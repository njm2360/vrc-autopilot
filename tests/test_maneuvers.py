from vrc_autopilot.control.controller import (
    PatrolGains,
    face_controllers,
    strafe_controller,
)
from vrc_autopilot.control.maneuvers import strafe_align
from vrc_autopilot.core.pose import Pose


class _NullAct:
    def look(self, *a, **k) -> None:
        pass

    def move(self, *a, **k) -> None:
        pass

    def stop(self) -> None:
        pass


class _ScriptedWorld:
    """位置列を再生する PoseSource 兼 Clock。sleep で時間を進め、dt ごとに次フレームへ。

    yaw は固定(目標が真横にある構図なので lat_err は大きいまま=指令を出し続ける)。
    """

    def __init__(self, positions: list[tuple[float, float]], *, dt: float = 0.05):
        self.positions = positions
        self.dt = dt
        self.t = 0.0
        self.frame = 0

    # ---- Clock ----
    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds
        nxt = self.frame + 1
        while nxt < len(self.positions) and nxt * self.dt <= self.t + 1e-12:
            self.frame = nxt
            nxt += 1

    # ---- PoseSource ----
    def get_latest(self) -> Pose:
        x, z = self.positions[min(self.frame, len(self.positions) - 1)]
        fwd = (0.0, 0.0, 1.0)  # yaw=0(+Z向き)、pitch=0
        return Pose(
            time_ms=self.frame, position=(x, 1.5, z), forward=fwd, up=(0.0, 1.0, 0.0)
        )


def _gains() -> PatrolGains:
    # 短い窓・短い打切りでテストを速く回す
    return PatrolGains(align_timeout=0.6, align_stuck_time=0.2, align_stuck_eps=0.02)


# 目標は真横(+X 方向)遠方 → yaw 誤差 ≈90°、lat_err 大 → strafe 指令が出続ける
_TARGET = (5.0, 1.5, 0.0)


def test_stuck_not_triggered_by_in_place_oscillation():
    """その場往復(正味変位≈0 でも経路長は大)では、指令を出していてもスタック判定しない。"""
    gains = _gains()
    # x が 0.0/0.1 を往復(毎フレーム 0.1m 移動)
    positions = [(0.1 if i % 2 else 0.0, 0.0) for i in range(40)]
    world = _ScriptedWorld(positions)
    res = strafe_align(
        world,
        _NullAct(),
        _NullAct(),
        _TARGET,
        gains,
        face_controllers(gains),
        strafe_controller(gains),
        clock=world,
    )
    assert res.reason != "stuck"


def test_stuck_triggered_by_true_zero_motion():
    """本当に動けない(位置が完全固定)場合は、指令を出しているのでスタック判定する。"""
    gains = _gains()
    positions = [(0.0, 0.0) for _ in range(40)]
    world = _ScriptedWorld(positions)
    res = strafe_align(
        world,
        _NullAct(),
        _NullAct(),
        _TARGET,
        gains,
        face_controllers(gains),
        strafe_controller(gains),
        clock=world,
    )
    assert res.reason == "stuck"


def test_stuck_not_triggered_without_command():
    """指令を出していない(win_commanded=False)なら、動いていなくてもスタックにしない。"""
    gains = _gains()
    positions = [(0.0, 0.0) for _ in range(40)]
    world = _ScriptedWorld(positions)
    res = strafe_align(
        world,
        _NullAct(),
        _NullAct(),
        # 真正面かつ上方(+Z, 高所)。yaw 誤差0 → lat_err0 → strafe 指令0だが、
        # pitch 誤差が残るので収束はしない(win_commanded=False のまま静止)
        (0.0, 5.0, 5.0),
        gains,
        face_controllers(gains),
        strafe_controller(gains),
        clock=world,
    )
    assert res.reason != "stuck"
