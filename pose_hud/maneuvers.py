"""誘導の建物ブロック(1フェーズ=1関数の制御ループ)。

実機 I/O には依存せず、PoseSource / LookActuator / MoveActuator の抽象だけで
動く(ヘッドレスでテスト可能)。follow_path は体の移動追従、aim_at / turn_to は
視点合わせ。経路計画やフェーズの連結は pilot.Pilot が担う。
"""

import math
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .actuator import LookActuator, MoveActuator
from .controller import FaceControllers, NavControllers, PatrolGains
from .guidance import forward_factor, heading_error, pitch_error, wrap180
from .navigation import Path
from .pose import Pose
from .telemetry import AxisAccumulator, AxisMetrics, NullRecorder, Recorder


class PoseSource(Protocol):
    def get_latest(self) -> Pose | None: ...


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _next_frame(
    reader: PoseSource,
    last_t: int | None,
    last_time: float,
    *,
    wait_cap: float = 2.0,
    dt_cap: float = 0.2,
    poll: float = 0.002,
) -> tuple[Pose | None, float, float]:
    deadline = time.monotonic() + wait_cap
    while time.monotonic() < deadline:
        pose = reader.get_latest()
        if pose is not None and pose.time_ms != last_t:
            now = time.monotonic()
            return pose, min(now - last_time, dt_cap), now
        time.sleep(poll)
    return None, 0.0, time.monotonic()


@dataclass
class NavResult:
    reached: bool  # 経路が見つかった(到達可能)
    arrived: bool  # 最終ウェイポイント付近まで到達した
    reason: str  # "arrived" | "unreachable" | "no_pose" | "hud_lost" | "timeout"
    path: Path | None  # 計画した経路(follow() 直接指定時は None)
    elapsed: float  # 追従に要した秒
    frames: int  # 処理フレーム数
    yaw: AxisMetrics | None = (
        None  # 進行方向 yaw の応答指標(制御フレームが無ければ None)
    )


@dataclass
class AimResult:
    converged: bool  # yaw/pitch とも許容内を settle 回連続で達成した
    yaw_err: float  # 最終 yaw 誤差[deg]
    pitch_err: float  # 最終 pitch 誤差[deg]
    elapsed: float
    frames: int
    yaw: AxisMetrics | None = None  # yaw 軸の応答指標
    pitch: AxisMetrics | None = None  # pitch 軸の応答指標(pitch 未制御時は None)


def follow_path(
    reader: PoseSource,
    look: LookActuator,
    move: MoveActuator,
    waypoints: list[tuple[float, float]],
    gains: PatrolGains,
    nav: NavControllers,
    *,
    recorder: Recorder | None = None,
    announce: Callable[[str], None] | None = None,
    name: str = "",
) -> NavResult:
    rec = recorder or NullRecorder()
    track = not isinstance(rec, NullRecorder)
    say = announce or (lambda _m: None)
    wps = list(waypoints)
    if not wps:
        return NavResult(True, True, "arrived", None, 0.0, 0)

    nav.yaw.reset()
    nav.forward.reset()
    idx = 1 if len(wps) > 1 else 0
    last_t: int | None = None
    last_time = t0 = time.monotonic()
    frames = 0
    reason = "arrived"
    yaw_acc = AxisAccumulator() if track else None
    while idx < len(wps):
        pose, dt, now = _next_frame(reader, last_t, last_time)
        if pose is None:
            reason = "hud_lost"
            say(f"  [{name}] HUD lost, abort nav")
            break
        last_t, last_time = pose.time_ms, now
        frames += 1
        cur = (pose.position[0], pose.position[2])

        prev_idx = idx
        while idx < len(wps) - 1 and _dist(cur, wps[idx]) < gains.arrive:
            idx += 1
        if idx != prev_idx:
            nav.yaw.reset_derivative()  # 目標が急に変わったとき turn が跳ねるのを防ぐ
        target = wps[idx]
        final = idx == len(wps) - 1
        err, dist = heading_error(cur, pose.yaw_deg, target)
        if final and dist < gains.arrive:
            break

        turn = nav.yaw.update(err, dt)
        ff = forward_factor(err)
        speed = (nav.forward.update(dist, dt) if final else gains.speed) * ff
        look.look(turn)
        move.move(forward=speed)

        if track:
            rec.row(
                t=now - t0,
                phase="nav",
                target=name,
                wp=idx,
                dt=dt,
                x=pose.position[0],
                y=pose.position[1],
                z=pose.position[2],
                yaw=pose.yaw_deg,
                pitch=pose.pitch_deg,
                tx=target[0],
                tz=target[1],
                dist=dist,
                yaw_err=err,
                turn_p=nav.yaw.last_p,
                turn_i=nav.yaw.last_i,
                turn_d=nav.yaw.last_d,
                turn=turn,
                fwd=speed,
                fwd_factor=ff,
            )
            yaw_acc.update(err, turn, now - t0, dt, gains.face_tol)
        if now - t0 > gains.nav_timeout:
            reason = "timeout"
            say(f"  [{name}] nav timeout")
            break
    look.stop()
    move.stop()
    return NavResult(
        reached=True,
        arrived=(reason == "arrived"),
        reason=reason,
        path=None,
        elapsed=time.monotonic() - t0,
        frames=frames,
        yaw=yaw_acc.snapshot() if (track and frames) else None,
    )


def _face_loop(
    reader: PoseSource,
    look: LookActuator,
    gains: PatrolGains,
    face: FaceControllers,
    *,
    errors: Callable[[Pose], tuple[float, float]],
    control_pitch: bool,
    phase: str,
    extra: dict[str, float],
    recorder: Recorder | None,
    name: str,
) -> AimResult:
    """正対系ループの共通コア。errors(pose) が (yaw誤差, pitch誤差)[deg] を返す。

    control_pitch=False のときは pitch を制御せず(指令0)、収束判定も yaw のみ。
    extra は記録行に足す列(aim_at のターゲット座標など)。
    """
    rec = recorder or NullRecorder()
    track = not isinstance(
        rec, NullRecorder
    )  # 記録先が無ければ行組立も指標計算もしない
    face.yaw.reset()
    face.pitch.reset()
    last_t: int | None = None
    last_time = t0 = time.monotonic()
    frames = 0
    settle = 0
    converged = False
    yaw_err = pitch_err = 0.0
    yaw_acc = AxisAccumulator() if track else None
    pitch_acc = AxisAccumulator() if (track and control_pitch) else None
    while time.monotonic() - t0 < gains.face_timeout:
        pose, dt, now = _next_frame(reader, last_t, last_time)
        if pose is None:
            break
        last_t, last_time = pose.time_ms, now
        frames += 1
        yaw_err, pitch_err = errors(pose)

        pitch_ok = not control_pitch or abs(pitch_err) < gains.face_tol
        if abs(yaw_err) < gains.face_tol and pitch_ok:
            settle += 1
            if settle >= gains.settle:
                converged = True
                break
        else:
            settle = 0

        turn = face.yaw.update(yaw_err, dt)
        pitch_cmd = face.pitch.update(pitch_err, dt) if control_pitch else 0.0
        look.look(turn, pitch_cmd)

        if track:
            rec.row(
                t=now - t0,
                phase=phase,
                target=name,
                dt=dt,
                x=pose.position[0],
                y=pose.position[1],
                z=pose.position[2],
                yaw=pose.yaw_deg,
                pitch=pose.pitch_deg,
                **extra,
                yaw_err=yaw_err,
                pitch_err=pitch_err,
                turn_p=face.yaw.last_p,
                turn_i=face.yaw.last_i,
                turn_d=face.yaw.last_d,
                turn=turn,
                pitch_p=face.pitch.last_p,
                pitch_i=face.pitch.last_i,
                pitch_d=face.pitch.last_d,
                pitch_cmd=pitch_cmd,
            )
            yaw_acc.update(yaw_err, turn, now - t0, dt, gains.face_tol)
            if pitch_acc is not None:
                pitch_acc.update(pitch_err, pitch_cmd, now - t0, dt, gains.face_tol)
    look.stop()
    return AimResult(
        converged=converged,
        yaw_err=yaw_err,
        pitch_err=pitch_err,
        elapsed=time.monotonic() - t0,
        frames=frames,
        yaw=yaw_acc.snapshot() if (track and frames) else None,
        pitch=pitch_acc.snapshot() if (pitch_acc is not None and frames) else None,
    )


def aim_at(
    reader: PoseSource,
    look: LookActuator,
    target_xyz: tuple[float, float, float],
    gains: PatrolGains,
    face: FaceControllers,
    *,
    recorder: Recorder | None = None,
    name: str = "",
) -> AimResult:
    tgt_xz = (target_xyz[0], target_xyz[2])

    def errors(pose: Pose) -> tuple[float, float]:
        cur = (pose.position[0], pose.position[2])
        yaw_err, _ = heading_error(cur, pose.yaw_deg, tgt_xz)
        return yaw_err, pitch_error(pose.position, pose.forward, target_xyz)

    return _face_loop(
        reader,
        look,
        gains,
        face,
        errors=errors,
        control_pitch=True,
        phase="face",
        extra={"tx": target_xyz[0], "ty": target_xyz[1], "tz": target_xyz[2]},
        recorder=recorder,
        name=name,
    )


def turn_to(
    reader: PoseSource,
    look: LookActuator,
    yaw_deg: float,
    gains: PatrolGains,
    face: FaceControllers,
    *,
    pitch_deg: float | None = None,
    recorder: Recorder | None = None,
    name: str = "",
) -> AimResult:
    def errors(pose: Pose) -> tuple[float, float]:
        yaw_err = wrap180(yaw_deg - pose.yaw_deg)
        return yaw_err, 0.0 if pitch_deg is None else (pitch_deg - pose.pitch_deg)

    return _face_loop(
        reader,
        look,
        gains,
        face,
        errors=errors,
        control_pitch=pitch_deg is not None,
        phase="turn",
        extra={},
        recorder=recorder,
        name=name,
    )
