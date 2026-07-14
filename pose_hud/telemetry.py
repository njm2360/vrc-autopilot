"""制御の記録と応答指標(チューニング用)。

Recorder はフレームごとの行の記録先(CSV 等)の抽象。AxisAccumulator は1軸の
誤差・指令の列から IAE/ITAE などの応答指標を積み、snapshot() で AxisMetrics に
固める。制御ループからは独立した純粋な計算で、recorder を付けたときだけ走る。
"""

from dataclasses import dataclass
from typing import Protocol


class Recorder(Protocol):
    def row(self, **kw) -> None: ...


class NullRecorder:
    def row(self, **kw) -> None:
        pass


@dataclass
class AxisMetrics:
    iae: float  # Σ|e|·dt(全体誤差・定常偏差)
    itae: float  # Σ t·|e|·dt(遅い収束ほど重く罰する)
    effort: float  # Σ|cmd|·dt(制御量。小さいほど省エネ・滑らか)
    overshoot: float  # 0 を最初に跨いだ後の反対符号ピーク[誤差と同単位]
    peak_err: float  # |e| の最大
    osc: int  # 誤差の符号反転回数(振動の目安)
    settle_time: float | None  # |e|<tol を最後に維持し始めた時刻[s](未収束は None)


class AxisAccumulator:
    def __init__(self):
        self.iae = self.itae = self.effort = 0.0
        self.overshoot = self.peak = 0.0
        self.osc = 0
        self.settle: float | None = None
        self._s0 = 0
        self._prev = 0
        self._n = 0

    def update(self, e: float, cmd: float, t: float, dt: float, tol: float) -> None:
        self._n += 1
        ae = abs(e)
        self.iae += ae * dt
        self.itae += t * ae * dt
        self.effort += abs(cmd) * dt
        if ae > self.peak:
            self.peak = ae
        if self._n == 1:
            self._s0 = 1 if e >= 0 else -1  # 初期誤差の符号(オーバーシュート基準)
        else:
            over = -self._s0 * e  # 目標(0)を跨いで反対側へ出た量
            if over > self.overshoot:
                self.overshoot = over
        sign = 1 if e > 0 else (-1 if e < 0 else 0)
        if sign and self._prev and sign != self._prev:
            self.osc += 1
        if sign:
            self._prev = sign
        if ae < tol:  # tol 未満を維持し始めた時刻を保持
            if self.settle is None:
                self.settle = t
        else:
            self.settle = None

    def snapshot(self) -> AxisMetrics:
        return AxisMetrics(
            iae=self.iae,
            itae=self.itae,
            effort=self.effort,
            overshoot=self.overshoot,
            peak_err=self.peak,
            osc=self.osc,
            settle_time=self.settle,
        )
