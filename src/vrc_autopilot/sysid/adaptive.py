"""適応プローブ(折れ点の二分探索 + 少数点の傾き + むだ時間バースト)。

静特性の形は「不感帯 + 線形」なので全域を刻まず、折れ点を二分探索で絞り、
線形域は少数レベルで傾きを取る。むだ時間は固定レベルの 0→v 遷移を多数回
連射し、分布(中央値・ばらつき・最悪値)として測る。フリーズ(HUD 途絶)で
あるレベルの定常データが全滅したときだけ測り直す(部分的な汚染は同定側が
セグメント単位で除外する)。

プローブ本体と同じく PoseSource・送信コールバック・クロック抽象にしか依存
しないので、SimulatedVRChat + SimClock でヘッドレスに検証できる。
"""

import dataclasses
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean

import numpy as np

from ..control.maneuvers import PoseSource
from .identify import (
    _KIND,
    _ONSET_THRESHOLD,
    FREEZE_FACTOR,
    ONSET_EPS_FRAC,
    AxisModel,
    ProbeRun,
    ProbeSample,
    deadtime_stats,
    freeze_gap,
    identify_axis,
    look_schedule,
    move_anchor,
    run_axis_probe,
    run_move_probe,
    run_pitch_probe,
    segment_rates,
)

logger = logging.getLogger(__name__)


def _clean_levels(run: ProbeRun, thr: float | None) -> set[float]:
    """定常窓に thr 超のギャップを含まないセグメントの |cmd| 集合(thr=None なら全採用)。"""
    rates, _ = segment_rates(run)
    return {
        round(abs(sr.cmd), 4)
        for sr in rates
        if sr.cmd != 0.0 and (thr is None or sr.max_gap <= thr)
    }


@dataclass(frozen=True)
class AdaptiveConfig:
    """適応プローブの調整定数。

    hold / settle が短いのは、プラントに速度ランプ・慣性が無い(0→v が
    1サンプルで定常に飛ぶ)ため。定常窓は skip_min+数フレームで足りる。
    """

    hold: float = 0.5  # 視点軸セグメントの保持秒
    move_hold: float = 1.0  # 移動軸の片道上限秒(位置ガードが先に効けば短い)
    settle: float = 0.3  # レベル間の 0 保持秒
    onset_tol: float = 0.01  # 折れ点の探索分解能(指令値)
    slope_levels: int = 3  # 折れ点と 1.0 の間に足す測定レベル数
    burst_n: int = 40  # むだ時間バーストの目標遷移回数
    burst_level: float = 0.8  # バーストの指令レベル(高いほど交差補正が小さい)
    burst_chunk: int = 6  # バーストのミニラン分割(測り直しを安くする)
    max_retries: int = 3  # フリーズ検知時の同一レベル測り直し回数
    freeze_factor: float = FREEZE_FACTOR
    max_travel: float = 3.0  # 移動軸プローブの往復範囲の片側幅[m]
    pitch_span: float = 70.0  # pitch を水平から振る角度幅[°](±80°クランプ手前まで)


@dataclass
class AdaptiveResult:
    """1軸の適応プローブ結果。run は save_run でそのまま保存できる。"""

    axis: str
    run: ProbeRun  # 全ミニランを連結した生記録
    model: AxisModel
    slope: float  # 折れ点より上の傾き [unit / 指令]
    deadtime_stats: dict  # {n, dropped, median, mean, std, p95, max}
    freezes: int  # 測り直しを引き起こしたフリーズ回数


class AdaptiveAxisProbe:
    """1軸ぶんの適応プローブ実行器。ミニラン(1〜数レベル)単位で測り、
    フリーズを検知したら測り直し、全記録を1本の ProbeRun に連結する。"""

    def __init__(
        self,
        reader: PoseSource,
        send: Callable[[float], None],
        axis: str,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        cfg: AdaptiveConfig | None = None,
    ):
        self.reader = reader
        self.send = send
        self.axis = axis
        self.monotonic = monotonic
        self.sleep = sleep
        self.cfg = cfg or AdaptiveConfig()
        self.freezes = 0
        self._samples: list[ProbeSample] = []
        self._starts: list[tuple[int, float, float]] = []
        self._seg_base = 0
        self._t0 = monotonic()
        self._anchor: tuple[float, float, float, float] | None = None

    def _mini_run(self, levels: list[float], *, passes: int = 1) -> ProbeRun:
        c = self.cfg
        if self.axis == "yaw":
            return run_axis_probe(
                self.reader,
                self.send,
                look_schedule(levels, hold=c.hold, settle=c.settle),
                axis="yaw",
                monotonic=self.monotonic,
                sleep=self.sleep,
            )
        if self.axis == "pitch":
            return run_pitch_probe(
                self.reader,
                self.send,
                levels,
                hold=c.hold,
                settle=c.settle,
                span=c.pitch_span,
                monotonic=self.monotonic,
                sleep=self.sleep,
            )
        # ホームと射影方向は軸で1回だけ決める(取り直すとホーミング誤差が累積する)
        if self._anchor is None:
            deadline = self.monotonic() + 2.0
            while (pose := self.reader.get_latest()) is None:
                if self.monotonic() > deadline:
                    raise RuntimeError(f"no pose before probe (axis={self.axis})")
                self.sleep(0.01)
            self._anchor = move_anchor(pose, self.axis)
        return run_move_probe(
            self.reader,
            self.send,
            levels,
            axis=self.axis,
            max_travel=c.max_travel,
            hold=c.move_hold,
            settle=c.settle,
            passes=passes,
            anchor=self._anchor,
            monotonic=self.monotonic,
            sleep=self.sleep,
        )

    def _absorb(self, run: ProbeRun, t_off: float) -> None:
        """ミニランを通し番号・通し時刻へ変換して連結する。"""
        for s in run.samples:
            self._samples.append(
                dataclasses.replace(s, seg=s.seg + self._seg_base, t=s.t + t_off)
            )
        for seg, cmd, t_send in run.seg_starts:
            self._starts.append((seg + self._seg_base, cmd, t_send + t_off))
        if run.seg_starts:
            self._seg_base += max(s for s, _, _ in run.seg_starts) + 1

    def _measure(self, levels: list[float], *, passes: int = 1) -> ProbeRun:
        """ミニランを実行する。各レベルに定常セグメントが1つでも残れば採用し、
        フリーズで全滅したレベルがあるときだけ測り直す(settle やホーム復帰の
        途絶では捨てない)。"""
        want = {round(abs(v), 4) for v in levels if v != 0.0}
        run = None
        for attempt in range(self.cfg.max_retries + 1):
            t_off = self.monotonic() - self._t0
            run = self._mini_run(levels, passes=passes)
            thr = freeze_gap(run, self.cfg.freeze_factor)
            if thr is None or not want or want <= _clean_levels(run, thr):
                self._absorb(run, t_off)
                return run
            self.freezes += 1
            wiped = sorted(want - _clean_levels(run, thr))
            logger.warning(
                "adaptive %s: freeze wiped level(s) %s (attempt %d/%d) -- re-measuring",
                self.axis,
                [f"{v:.2f}" for v in wiped],
                attempt + 1,
                self.cfg.max_retries + 1,
            )
        # リトライ上限: 汚染込みで採用(identify 側のフリーズ除外が最後の砦)
        logger.warning(
            "adaptive %s: freeze persisted after %d retries -- keeping last run",
            self.axis,
            self.cfg.max_retries,
        )
        self._absorb(run, self.monotonic() - self._t0)
        return run

    def measure_level(self, v: float, *, passes: int = 1) -> float:
        """±v を1レベル測り、両符号の定常速度の平均絶対値を返す。"""
        run = self._measure([v], passes=passes)
        rates = [sr.rate for sr in segment_rates(run)[0] if sr.cmd != 0.0]
        if not rates:
            logger.warning("adaptive %s: no steady segment at level %.2f", self.axis, v)
            return 0.0
        return fmean(abs(r) for r in rates)

    def burst(self, v: float, n_transitions: int) -> None:
        """0→v 遷移を n_transitions 回連射する(むだ時間の分布用)。"""
        per = 2 if self.axis in ("yaw", "pitch") else 1  # ±対は2遷移/レベル
        todo = math.ceil(n_transitions / per)
        while todo > 0:
            k = min(self.cfg.burst_chunk, todo)
            self._measure([v] * k, passes=0)
            todo -= k

    def result(self) -> ProbeRun:
        return ProbeRun(axis=self.axis, samples=self._samples, seg_starts=self._starts)


def probe_axis_adaptive(
    reader: PoseSource,
    send: Callable[[float], None],
    axis: str,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    cfg: AdaptiveConfig | None = None,
) -> AdaptiveResult:
    """1軸を適応プローブして AdaptiveResult(生記録+モデル+むだ時間分布)を返す。"""
    cfg = cfg or AdaptiveConfig()
    p = AdaptiveAxisProbe(reader, send, axis, monotonic=monotonic, sleep=sleep, cfg=cfg)
    kind = _KIND[axis]

    # 最大速度と「動いた」判定しきい値
    v_max = p.measure_level(1.0)
    seg_hold = cfg.move_hold if kind == "pos" else cfg.hold
    eps = max(ONSET_EPS_FRAC * v_max, 2.0 * _ONSET_THRESHOLD[kind] / seg_hold)
    if v_max <= eps:
        raise ValueError(f"axis {axis} does not respond at cmd=1.0")

    # 折れ点の二分探索。最小レベルの応答が線形外挿並みなら不感帯なしとみなして
    # 省略する(eps 比較だと不感帯なしの軸でも探索に落ちる)
    lo, hi = cfg.onset_tol, 1.0
    if p.measure_level(lo, passes=0) >= 0.5 * lo * v_max:
        lo = 0.0
    else:
        while hi - lo > cfg.onset_tol:
            mid = round((lo + hi) / 2.0, 4)
            if p.measure_level(mid, passes=0) > eps:
                hi = mid
            else:
                lo = mid

    # 線形域の傾きレベル(折れ点の少し上〜1.0 を等分。1.0 は測定済み)
    lo_lv = min(lo + max(2.0 * cfg.onset_tol, 0.05), 0.95)
    for v in np.linspace(lo_lv, 1.0, cfg.slope_levels + 1)[:-1]:
        p.measure_level(round(float(v), 4))

    p.burst(max(cfg.burst_level, min(lo + 0.2, 1.0)), cfg.burst_n)

    run = p.result()
    model = identify_axis(run, freeze_factor=cfg.freeze_factor)
    stats = deadtime_stats(run, cfg.freeze_factor)

    onset = model.onset
    above = [(c, r) for c, r in model.points if abs(c) > onset + cfg.onset_tol]
    slope = 0.0
    if above:
        xs = np.array([math.copysign(abs(c) - onset, c) for c, _ in above])
        ys = np.array([r for _, r in above])
        denom = float(xs @ xs)
        slope = float(xs @ ys / denom) if denom > 0.0 else 0.0
    logger.info("adaptive %s: onset %.3f slope %.1f", axis, onset, slope)

    return AdaptiveResult(
        axis=axis,
        run=run,
        model=model,
        slope=slope,
        deadtime_stats=stats,
        freezes=p.freezes,
    )
