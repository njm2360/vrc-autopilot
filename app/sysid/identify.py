"""実機プラントのシステム同定(プローブ注入 → 特性抽出 → モデル保存)。

階段状の指令列を1軸に注入して HUD ポーズ時系列を記録し、「指令→定常速度」の
静特性・むだ時間・フレーム間隔 dt 列を AxisModel / PlantModel に抽出する。
simplant.SimulatedVRChat がこのモデルを積分すると実機なしでゲイン検証できる。

むだ時間は OSC→ゲーム反映→描画→キャプチャ→デコードの合計、つまり制御器から
見えるループ遅延そのもの。視点軸に時間方向の平滑化(ランプ)があるとモデル化
から漏れる。手法の詳細は docs/system-identification.md を参照。
"""

import csv
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, median
from typing import Callable

import numpy as np

from ..control.guidance import wrap180
from ..control.maneuvers import PoseSource

logger = logging.getLogger(__name__)

AXES = ("yaw", "pitch", "forward", "strafe")
# VRChat の /input/ 軸名
AXIS_INPUT = {
    "yaw": "LookHorizontal",
    "pitch": "LookVertical",
    "forward": "Vertical",
    "strafe": "Horizontal",
}
_KIND = {"yaw": "angle", "pitch": "angle", "forward": "pos", "strafe": "pos"}
_UNIT = {"angle": "deg/s", "pos": "m/s"}
# むだ時間検出の変化しきい値(HUD デコードの量子化ノイズより十分大きく)
_ONSET_THRESHOLD = {"angle": 0.1, "pos": 0.01}
# これ以上はクランプ(±90°)張り付きとみなして identify_axis が除外する
_PITCH_SAT_DEG = 88.0


# ---------------------------------------------------------------------------
# プローブ(記録)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeSegment:
    """1区間の指令値と保持秒。"""

    cmd: float
    hold_s: float


@dataclass(frozen=True)
class ProbeSample:
    seg: int
    cmd: float
    t: float  # プローブ開始からの受信秒(実時間)
    time_ms: int
    x: float
    y: float
    z: float
    yaw: float
    pitch: float


@dataclass
class ProbeRun:
    """1軸ぶんのプローブ記録。"""

    axis: str
    samples: list[ProbeSample]
    seg_starts: list[tuple[int, float, float]]  # (seg, cmd, 送信t)


def look_schedule(
    levels: list[float], hold: float = 1.0, settle: float = 0.6
) -> list[ProbeSegment]:
    """視点軸用スケジュール: 各レベルを +v / -v の対で、間に 0 を挟む。

    0→v の遷移ごとにむだ時間が測れる。+と−を対にするので pitch は各レベルで
    ほぼ中央に戻る(クランプ ±90° に張り付かないよう hold は短めに)。
    """
    segs = [ProbeSegment(0.0, settle)]
    for v in levels:
        segs += [
            ProbeSegment(+v, hold),
            ProbeSegment(0.0, settle),
            ProbeSegment(-v, hold),
            ProbeSegment(0.0, settle),
        ]
    return segs


def move_schedule(
    levels: list[float], hold: float = 0.8, settle: float = 0.6
) -> list[ProbeSegment]:
    """移動軸用スケジュール: +v→-v の往復で元の位置に戻りながら両符号を測る。"""
    segs = [ProbeSegment(0.0, settle)]
    for v in levels:
        segs += [
            ProbeSegment(+v, hold),
            ProbeSegment(-v, hold),
            ProbeSegment(0.0, settle),
        ]
    return segs


def schedule_duration(segments: list[ProbeSegment]) -> float:
    return sum(s.hold_s for s in segments)


class _ProbeRecorder:
    """セグメント実行と記録の共有コア(クロックと停止条件を注入できる)。"""

    def __init__(
        self,
        reader: PoseSource,
        *,
        axis: str,
        monotonic: Callable[[], float],
        sleep: Callable[[float], None],
        wait_cap: float,
        poll: float,
        blind_cap: float = 0.3,
        total_segs: int | None = None,
        progress_every: float = 5.0,
    ):
        self.reader = reader
        self.axis = axis
        self.monotonic = monotonic
        self.sleep = sleep
        self.wait_cap = wait_cap
        self.poll = poll
        self.blind_cap = blind_cap
        self.total_segs = total_segs
        self.progress_every = progress_every
        self.samples: list[ProbeSample] = []
        self.starts: list[tuple[int, float, float]] = []
        self.t0 = monotonic()
        self.latest = None  # 直近の記録ポーズ
        self._last_ms: int | None = None
        self._last_frame = self.t0
        self._last_progress = self.t0
        self._seg = 0

    def run_segment(
        self,
        send: Callable[[float], None],
        cmd: float,
        hold_s: float,
        stop_when: Callable | None = None,
    ) -> None:
        """指令 cmd を保持しつつ記録する。stop_when(pose) が真になったら早期終了。

        新フレームのたびに指令を再送する(UDP 欠落対策)。ガード付きセグメントで
        HUD が blind_cap 以上途絶したら、ガードを見られないまま動き続けないよう
        指令を 0 に戻して打ち切る。
        """
        send(cmd)
        t_send = self.monotonic()
        self.starts.append((self._seg, cmd, t_send - self.t0))
        deadline = t_send + hold_s
        while (now := self.monotonic()) < deadline:
            pose = self.reader.get_latest()
            if pose is not None and pose.time_ms != self._last_ms:
                self._last_ms = pose.time_ms
                self._last_frame = now
                self.latest = pose
                p = pose.position
                self.samples.append(
                    ProbeSample(
                        self._seg,
                        cmd,
                        now - self.t0,
                        pose.time_ms,
                        p[0],
                        p[1],
                        p[2],
                        pose.yaw_deg,
                        pose.pitch_deg,
                    )
                )
                send(cmd)
                if stop_when is not None and stop_when(pose):
                    break
            elif now - self._last_frame > self.wait_cap:
                raise RuntimeError(
                    f"HUD lost during probe (axis={self.axis}, seg={self._seg})"
                )
            elif (
                stop_when is not None
                and cmd != 0.0
                and now - self._last_frame > self.blind_cap
            ):
                send(0.0)
                logger.warning(
                    "probe %s: HUD stalled %.2fs in guarded seg %d -- command cut",
                    self.axis,
                    now - self._last_frame,
                    self._seg,
                )
                break
            else:
                self.sleep(self.poll)
        self._seg += 1
        self._log_progress()

    def _log_progress(self) -> None:
        now = self.monotonic()
        if now - self._last_progress < self.progress_every:
            return
        self._last_progress = now
        total = f"/{self.total_segs}" if self.total_segs else ""
        logger.info(
            "probe %s: seg %d%s  t=%.0fs  %d samples",
            self.axis,
            self._seg,
            total,
            now - self.t0,
            len(self.samples),
        )

    def wait_first_pose(self):
        """最初のポーズを待って返す(wait_cap でタイムアウト)。"""
        deadline = self.t0 + self.wait_cap
        pose = self.reader.get_latest()
        while pose is None:
            if self.monotonic() > deadline:
                raise RuntimeError(f"no pose before probe (axis={self.axis})")
            self.sleep(self.poll)
            pose = self.reader.get_latest()
        return pose

    def result(self) -> ProbeRun:
        return ProbeRun(axis=self.axis, samples=self.samples, seg_starts=self.starts)


def run_axis_probe(
    reader: PoseSource,
    send: Callable[[float], None],
    segments: list[ProbeSegment],
    *,
    axis: str,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    wait_cap: float = 2.0,
    poll: float = 0.002,
) -> ProbeRun:
    """スケジュールを注入しながらポーズ時系列を記録する(1軸)。

    時間ベースなので視点軸向け。移動軸には位置ガードで行動範囲を絞る
    run_move_probe を使う。monotonic / sleep を差し替えるとヘッドレスで走る。
    """
    rec = _ProbeRecorder(
        reader,
        axis=axis,
        monotonic=monotonic,
        sleep=sleep,
        wait_cap=wait_cap,
        poll=poll,
        total_segs=len(segments),
    )
    try:
        for seg in segments:
            rec.run_segment(send, seg.cmd, seg.hold_s)
    finally:
        send(0.0)
    return rec.result()


def run_move_probe(
    reader: PoseSource,
    send: Callable[[float], None],
    levels: list[float],
    *,
    axis: str,
    max_travel: float = 0.6,
    hold: float = 1.2,
    settle: float = 0.6,
    passes: int = 2,
    home_tol: float = 0.08,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    wait_cap: float = 2.0,
    poll: float = 0.002,
    blind_cap: float = 0.3,
) -> ProbeRun:
    """移動軸(forward/strafe)の省スペースプローブ。

    開始位置(ホーム)から軸方向 ±max_travel の範囲内で往復して測る。位置
    ガードで指令を切り返すので、移動速度が未知でも実際の移動範囲は ±max_travel
    +行き過ぎマージン(≒ 最高速度×むだ時間)に収まる。各レベルで passes 回往復して
    サンプルを稼ぎ、最後にホームへ戻してから次のレベルへ移る。

    切り返し直後はむだ時間ぶん逆向きに動いているため、identify_axis はセグメント
    先頭(skip_min 秒以上)を捨てて定常部だけで速度を取る。hold は片道の上限秒
    (遅いレベルでは帯に届かずこの時間で切れる)。
    """
    rec = _ProbeRecorder(
        reader,
        axis=axis,
        monotonic=monotonic,
        sleep=sleep,
        wait_cap=wait_cap,
        poll=poll,
        blind_cap=blind_cap,
    )
    # ホーム位置と射影方向は最初のポーズの向き基準
    pose = rec.wait_first_pose()
    yr = math.radians(pose.yaw_deg)
    if axis == "forward":
        dx, dz = math.sin(yr), math.cos(yr)
    else:  # strafe: 右方向
        dx, dz = math.cos(yr), -math.sin(yr)
    hx, hz = pose.position[0], pose.position[2]

    def proj(p) -> float:
        return (p.position[0] - hx) * dx + (p.position[2] - hz) * dz

    try:
        rec.run_segment(send, 0.0, settle)
        for n, v in enumerate(levels, 1):
            logger.info("probe %s: level %+.2f (%d/%d)", axis, v, n, len(levels))
            rec.run_segment(send, +v, hold, stop_when=lambda p: proj(p) >= max_travel)
            for _ in range(passes):
                rec.run_segment(
                    send, -v, 2 * hold, stop_when=lambda p: proj(p) <= -max_travel
                )
                rec.run_segment(
                    send, +v, 2 * hold, stop_when=lambda p: proj(p) >= max_travel
                )
            # ホームへ戻してから次のレベルへ
            cur = proj(rec.latest) if rec.latest is not None else 0.0
            if cur > 0:
                rec.run_segment(
                    send, -v, 2 * hold, stop_when=lambda p: proj(p) <= home_tol
                )
            else:
                rec.run_segment(
                    send, +v, 2 * hold, stop_when=lambda p: proj(p) >= -home_tol
                )
            rec.run_segment(send, 0.0, settle)
    finally:
        send(0.0)
    return rec.result()


def run_pitch_probe(
    reader: PoseSource,
    send: Callable[[float], None],
    levels: list[float],
    *,
    hold: float = 0.8,
    settle: float = 0.6,
    span: float = 45.0,
    abs_limit: float = 85.0,
    home_tol: float = 5.0,
    home_level: float = 0.5,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    wait_cap: float = 2.0,
    poll: float = 0.002,
    blind_cap: float = 0.3,
) -> ProbeRun:
    """pitch の角度ガード付きプローブ。

    時間ベース(look_schedule)だと速いレベルで ±90° クランプに張り付き、静特性が
    黙って潰れる。そこでセグメント開始の pitch から ±span(絶対値では abs_limit)を
    超えたら指令を切る。± を対で振れば各レベル後はほぼ元の角度に戻り、残りの
    ドリフトはレベル間で開始時のホームへ戻して吸収する。+指令で pitch がどちらへ
    動くかは仮定せず、応答の符号を実測してホーミングに使う。hold は片道の上限秒
    (遅いレベルではガードに届かずこの時間で切れる)。
    """
    rec = _ProbeRecorder(
        reader,
        axis="pitch",
        monotonic=monotonic,
        sleep=sleep,
        wait_cap=wait_cap,
        poll=poll,
        blind_cap=blind_cap,
    )
    home = rec.wait_first_pose().pitch_deg

    def cur_pitch() -> float:
        return rec.latest.pitch_deg if rec.latest is not None else home

    def swing_guard(p0: float) -> Callable:
        return lambda p: abs(p.pitch_deg - p0) >= span or abs(p.pitch_deg) >= abs_limit

    sign = 0.0  # +指令で pitch が動く向き(応答から実測する)
    try:
        rec.run_segment(send, 0.0, settle)
        for n, v in enumerate(levels, 1):
            logger.info("probe pitch: level %+.2f (%d/%d)", v, n, len(levels))
            n0 = len(rec.samples)
            rec.run_segment(send, +v, hold, stop_when=swing_guard(cur_pitch()))
            seg = rec.samples[n0:]
            if len(seg) >= 2 and abs(seg[-1].pitch - seg[0].pitch) > 0.5:
                sign = math.copysign(1.0, seg[-1].pitch - seg[0].pitch)
            rec.run_segment(send, 0.0, settle)
            rec.run_segment(send, -v, hold, stop_when=swing_guard(cur_pitch()))
            rec.run_segment(send, 0.0, settle)
            # ホーム帯へ戻してから次のレベルへ
            cur = cur_pitch() - home
            if abs(cur) > home_tol and sign != 0.0:
                cmd = -sign * home_level if cur > 0 else sign * home_level
                rec.run_segment(
                    send,
                    cmd,
                    2 * hold,
                    stop_when=lambda p: abs(p.pitch_deg - home) <= home_tol
                    or abs(p.pitch_deg) >= abs_limit,
                )
                rec.run_segment(send, 0.0, settle)
    finally:
        send(0.0)
    return rec.result()


# ---------------------------------------------------------------------------
# 同定(抽出)
# ---------------------------------------------------------------------------


def _median3(ys: list[float]) -> list[float]:
    """3点メディアン平滑(両端は不変)。単調な区間(折れ点・平坦部含む)は
    厳密に不変で、孤立した外れ値だけを隣の値に置き換える。"""
    n = len(ys)
    if n < 3:
        return list(ys)
    return [ys[0]] + [median(ys[i - 1 : i + 2]) for i in range(1, n - 1)] + [ys[-1]]


def _fix_endpoint(
    x0: float, y0: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """端点の孤立外れ値補正。(x0,y0) が端点、(x1,y1)(x2,y2) は平滑済みの内側 2 点。

    端点はメディアン窓が組めず最も外れ値が残りやすい。隣接間隔が同程度
    (端まで一様なグリッド=同定が作る等間隔レベル)のときだけ、内側 2 点の
    線形外挿・隣接値・生値の中央値を取る(線形に伸びる端は不変)。間隔が不揃いな
    粗いカーブでは外挿を信用できない(ニー直後の端点を壊す)ので生値のまま返す。
    """
    d1, d2 = abs(x1 - x0), abs(x2 - x1)
    if d2 <= 0.0 or not (0.8 <= d1 / d2 <= 1.25):
        return y0
    pred = y1 + (y1 - y2) * (d1 / d2)
    return median([y0, y1, pred])


def _denoise_points(
    points: list[tuple[float, float]], axis: str
) -> list[tuple[float, float]]:
    """静特性(cmd 昇順)の孤立した外れ点を 3 点メディアンで除く。

    カーブは AxisModel.rate の順方向補間(cmd→速度)にしか使わないので単調性は
    課さず、1 レベルだけ飛んだ同定ノイズの除去に絞る(単調区間・不感帯・
    折れ点は厳密に保存され、ゲイン符号の仮定も不要)。
    置換で軸の最大 |rate| の 10% を超えて動いた点があれば同定データ品質を警告。
    """
    ys = [r for _, r in points]
    smoothed = _median3(ys)
    if len(points) >= 4:
        xs = [c for c, _ in points]
        smoothed[0] = _fix_endpoint(
            xs[0], ys[0], xs[1], smoothed[1], xs[2], smoothed[2]
        )
        smoothed[-1] = _fix_endpoint(
            xs[-1], ys[-1], xs[-2], smoothed[-2], xs[-3], smoothed[-3]
        )
    max_rate = max((abs(y) for y in ys), default=0.0)
    if max_rate > 0.0:
        worst = max(abs(a - b) for a, b in zip(ys, smoothed))
        if worst > 0.1 * max_rate:
            logger.warning(
                "axis %s: median filter adjusted static curve by %.0f%% of "
                "max |rate| -- identification data quality issue",
                axis,
                100.0 * worst / max_rate,
            )
    return [(c, y) for (c, _), y in zip(points, smoothed)]


def _responses(samples: list[ProbeSample], axis: str) -> np.ndarray:
    """サンプル列の応答量(先頭サンプル基準の相対値)。

    yaw は最短回りで unwrap。移動軸は先頭サンプル時点の体の向きを基準に、
    前方向(forward)/右方向(strafe)へ変位を射影する(仮に VRChat の +Horizontal
    が左向きだった場合は速度が負になるだけで、モデルとしては一貫する)。
    """
    if axis == "yaw":
        yaws = [s.yaw for s in samples]
        return np.cumsum([0.0] + [wrap180(b - a) for a, b in zip(yaws, yaws[1:])])
    if axis == "pitch":
        p0 = samples[0].pitch
        return np.array([s.pitch - p0 for s in samples])
    yr = math.radians(samples[0].yaw)
    if axis == "forward":
        dx, dz = math.sin(yr), math.cos(yr)
    else:  # strafe: 右方向
        dx, dz = math.cos(yr), -math.sin(yr)
    x0, z0 = samples[0].x, samples[0].z
    return np.array([(s.x - x0) * dx + (s.z - z0) * dz for s in samples])


@dataclass
class AxisModel:
    """1軸の同定結果: 静特性(指令→定常速度)の折れ線+むだ時間。"""

    axis: str
    unit: str  # "deg/s" | "m/s"
    points: list[tuple[float, float]]  # (指令, 定常速度) 指令の昇順
    deadtime_s: float = 0.0
    # rate() の補間配列キャッシュ。points はリスト差し替えで更新する前提
    # (ストレス試験の流儀)。要素の in-place 変更は検知できない
    _cache: tuple | None = field(default=None, init=False, repr=False, compare=False)

    def rate(self, cmd: float) -> float:
        """指令値 → 定常速度(測定点間は線形補間、範囲外は端の値)。"""
        c = self._cache
        if c is None or c[0] is not self.points:
            xs = np.asarray([p[0] for p in self.points], dtype=np.float64)
            ys = np.asarray([p[1] for p in self.points], dtype=np.float64)
            self._cache = c = (self.points, xs, ys)
        return float(np.interp(cmd, c[1], c[2]))


def identify_axis(
    run: ProbeRun,
    *,
    skip_frac: float = 0.4,
    skip_min: float = 0.2,
    min_samples: int = 3,
) -> AxisModel:
    """記録 1 本から静特性とむだ時間を抽出する。

    静特性: 各セグメントの後半の応答の傾きを最小二乗で取り、同一指令値は平均
    する。先頭は max(skip_frac×区間長, skip_min) 秒を過渡+むだ時間として捨てる
    (run_move_probe の切り返し直後の逆走を含む)。それだと定常サンプルが
    min_samples 未満しか残らない短いセグメントは棄却する(過渡だけの当てはめは
    速度を大きく誤るため。棄却が出た軸は件数を警告する)。
    むだ時間: 0→非0 の遷移で応答がしきい値を超えた時刻から、しきい値到達に
    かかる時間(しきい値/速度)を差し引いて逆算し、遷移全体の中央値を取る。
    """
    kind = _KIND[run.axis]
    by_seg: dict[int, list[ProbeSample]] = {}
    for s in run.samples:
        by_seg.setdefault(s.seg, []).append(s)
    start_by_seg = {i: (cmd, t) for i, cmd, t in run.seg_starts}

    # ---- 静特性 ----
    rates: dict[float, list[float]] = {}
    seg_rate: dict[int, float] = {}
    skipped = 0
    for i, samples in by_seg.items():
        cmd, t_send = start_by_seg[i]
        if run.axis == "pitch":
            # クランプに張り付いたサンプルは傾きを潰すので捨てる(張り付き前のランプは使える)
            kept = [s for s in samples if abs(s.pitch) < _PITCH_SAT_DEG]
            if len(kept) < len(samples):
                logger.warning(
                    "pitch seg %d (cmd=%+.2f): dropped %d/%d clamped samples",
                    i,
                    cmd,
                    len(samples) - len(kept),
                    len(samples),
                )
                samples = kept
            if not samples:
                continue
        span = samples[-1].t - t_send
        window = [s for s in samples if s.t >= t_send + max(skip_frac * span, skip_min)]
        if len(window) < min_samples:
            if cmd != 0.0:  # cmd=0 のセトリング区間は静特性に使わないので数えない
                skipped += 1
            continue
        t = np.array([s.t for s in window])
        resp = _responses(window, run.axis)
        rate = float(np.polyfit(t - t[0], resp, 1)[0])
        seg_rate[i] = rate
        if cmd != 0.0:
            rates.setdefault(cmd, []).append(rate)

    if skipped:
        n_cmd_segs = sum(1 for _, cmd, _ in run.seg_starts if cmd != 0.0)
        logger.warning(
            "axis %s: skipped %d/%d command segments with fewer than %d steady "
            "samples -- their command levels are missing from the static curve",
            run.axis,
            skipped,
            n_cmd_segs,
            min_samples,
        )
    if not rates:
        raise ValueError(f"no usable segments in probe run (axis={run.axis})")
    points = sorted((c, fmean(v)) for c, v in rates.items())
    if not any(c == 0.0 for c, _ in points):
        points = sorted(points + [(0.0, 0.0)])
    points = _denoise_points(points, run.axis)

    # ---- むだ時間 ----
    thr = _ONSET_THRESHOLD[kind]
    deads: list[float] = []
    prev_cmd: float | None = None
    for i, cmd, t_send in run.seg_starts:
        if prev_cmd == 0.0 and cmd != 0.0 and i in by_seg and (i - 1) in by_seg:
            steady = abs(seg_rate.get(i, 0.0))
            if steady > 1e-9:
                # 遷移直前の最後のサンプルを基準に応答を測る
                seq = [by_seg[i - 1][-1]] + by_seg[i]
                resp = np.abs(_responses(seq, run.axis))
                prev_s, prev_r = seq[0], float(resp[0])  # 遷移直前(応答≈0)
                for s, r in zip(seq[1:], resp[1:]):
                    r = float(r)
                    if r > thr:
                        # しきい値を跨ぐ時刻を前後サンプルで線形補間(1フレーム量子化遅れを除く)
                        frac = (thr - prev_r) / (r - prev_r) if r > prev_r else 0.0
                        t_cross = prev_s.t + frac * (s.t - prev_s.t)
                        deads.append(max(0.0, t_cross - t_send - thr / steady))
                        break
                    prev_s, prev_r = s, r
        prev_cmd = cmd
    deadtime = float(median(deads)) if deads else 0.0

    return AxisModel(
        axis=run.axis, unit=_UNIT[kind], points=points, deadtime_s=deadtime
    )


def extract_dts(run: ProbeRun, cap: float = 0.2) -> list[float]:
    """フレーム間隔 dt の列(記録全体。異常な間隔は cap で除外)。"""
    ts = [s.t for s in run.samples]
    return [b - a for a, b in zip(ts, ts[1:]) if 0.0 < b - a <= cap]


# ---------------------------------------------------------------------------
# プラントモデル(保存・読込)
# ---------------------------------------------------------------------------


@dataclass
class PlantModel:
    """同定済みプラント一式(軸モデル+フレーム間隔)。JSON で保存・読込できる。"""

    axes: dict[str, AxisModel]
    dt_mean: float = 0.05
    dt_seq: list[float] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def save(self, path) -> Path:
        path = Path(path)
        data = {
            "meta": self.meta,
            "dt_mean": self.dt_mean,
            "dt_seq": self.dt_seq,
            "axes": {
                name: {
                    "unit": m.unit,
                    "deadtime_s": m.deadtime_s,
                    "points": [[c, r] for c, r in m.points],
                }
                for name, m in self.axes.items()
            },
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "PlantModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # 外れ点の除去は同定時に済んでいる(3点メディアンは非冪等なので再適用しない)
        axes = {
            name: AxisModel(
                axis=name,
                unit=a["unit"],
                points=sorted((float(c), float(r)) for c, r in a["points"]),
                deadtime_s=float(a["deadtime_s"]),
            )
            for name, a in data["axes"].items()
        }
        return cls(
            axes=axes,
            dt_mean=float(data["dt_mean"]),
            dt_seq=[float(v) for v in data.get("dt_seq", [])],
            meta=data.get("meta", {}),
        )


def build_plant(
    runs: list[ProbeRun], *, meta: dict | None = None, max_dt_seq: int = 2000
) -> PlantModel:
    """複数軸のプローブ記録から PlantModel を組む。同定に失敗した軸はスキップする。"""
    axes: dict[str, AxisModel] = {}
    for r in runs:
        try:
            axes[r.axis] = identify_axis(r)
        except ValueError as e:
            logger.warning("axis %s: identify failed, skipping (%s)", r.axis, e)
    if not axes:
        raise ValueError("no axes could be identified")
    dts: list[float] = []
    for r in runs:
        dts.extend(extract_dts(r))
    return PlantModel(
        axes=axes,
        dt_mean=fmean(dts) if dts else 0.05,
        dt_seq=dts[:max_dt_seq],
        meta=meta or {},
    )


# ---------------------------------------------------------------------------
# 記録の CSV 保存・読込(あとから --from-log で再同定できるように)
# ---------------------------------------------------------------------------

_SAMPLE_FIELDS = ["seg", "cmd", "t", "time_ms", "x", "y", "z", "yaw", "pitch"]
_SEGMENT_FIELDS = ["seg", "cmd", "t_send"]


def save_run(run: ProbeRun, dir_) -> tuple[Path, Path]:
    """probe_{axis}.csv(サンプル)と segments_{axis}.csv(送信時刻)を書く。"""
    dir_ = Path(dir_)
    dir_.mkdir(parents=True, exist_ok=True)
    samples_path = dir_ / f"probe_{run.axis}.csv"
    with open(samples_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_SAMPLE_FIELDS)
        for s in run.samples:
            w.writerow([s.seg, s.cmd, s.t, s.time_ms, s.x, s.y, s.z, s.yaw, s.pitch])
    segments_path = dir_ / f"segments_{run.axis}.csv"
    with open(segments_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_SEGMENT_FIELDS)
        for seg, cmd, t_send in run.seg_starts:
            w.writerow([seg, cmd, t_send])
    return samples_path, segments_path


def load_run(dir_, axis: str) -> ProbeRun:
    dir_ = Path(dir_)
    samples: list[ProbeSample] = []
    with open(dir_ / f"probe_{axis}.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append(
                ProbeSample(
                    seg=int(row["seg"]),
                    cmd=float(row["cmd"]),
                    t=float(row["t"]),
                    time_ms=int(row["time_ms"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                    yaw=float(row["yaw"]),
                    pitch=float(row["pitch"]),
                )
            )
    starts: list[tuple[int, float, float]] = []
    with open(dir_ / f"segments_{axis}.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            starts.append((int(row["seg"]), float(row["cmd"]), float(row["t_send"])))
    return ProbeRun(axis=axis, samples=samples, seg_starts=starts)
