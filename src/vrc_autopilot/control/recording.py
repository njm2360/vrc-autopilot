"""制御の記録と応答指標(チューニング用)。

Recorder はフレームごとの行の記録先(CSV 等)の抽象。AxisAccumulator は1軸の
誤差・指令の列から IAE/ITAE などの応答指標を積算し、snapshot() で AxisMetrics
にまとめる。制御ループからは独立した純粋な計算で、recorder を付けたときだけ走る。
"""

import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Protocol


@dataclass
class ControlRow:
    t: float  # 開始からの経過秒
    phase: str  # nav / translate / face / turn / align
    wp: int | None = None  # 追従中のウェイポイント番号(nav/translate)
    dt: float | None = None  # 前フレームからの実経過秒
    x: float | None = None
    y: float | None = None
    z: float | None = None
    yaw: float | None = None
    pitch: float | None = None
    tx: float | None = None
    ty: float | None = None
    tz: float | None = None
    dist: float | None = None  # ターゲットまでの水平距離[m]
    yaw_err: float | None = None
    pitch_err: float | None = None
    lat_err: float | None = None  # 横方向誤差[m](alignのみ。+なら目標が右)
    fwd_err: float | None = None  # 目標までの前方距離[m](translateのみ)
    right_err: float | None = None  # 目標までの右方距離[m](translateのみ)
    turn_p: float | None = None
    turn_i: float | None = None
    turn_d: float | None = None
    turn: float | None = None  # yaw(LookHorizontal)PID内訳と出力
    pitch_p: float | None = None
    pitch_i: float | None = None
    pitch_d: float | None = None
    pitch_cmd: float | None = None
    strafe_p: float | None = None
    strafe_i: float | None = None
    strafe_d: float | None = None
    strafe: float | None = None  # Horizontal(横移動)PID内訳と出力(alignのみ)
    fwd: float | None = None  # Vertical(前進)出力
    fwd_factor: float | None = None  # 向きズレによる前進減衰係数


class Recorder(Protocol):
    def row(self, row: ControlRow) -> None: ...


class NullRecorder:
    def row(self, row: ControlRow) -> None:
        pass


class ListRecorder:
    """行を list[ControlRow] に貯める Recorder(テスト・オフライン解析用)。"""

    def __init__(self):
        self.rows: list[ControlRow] = []

    def row(self, row: ControlRow) -> None:
        self.rows.append(row)


class ControlLog:
    """行を CSV に書き出す Recorder(実機ログ・log-video の入力)。列は ControlRow に一元化。"""

    FIELDS = [f.name for f in fields(ControlRow)]

    @classmethod
    def timestamped(cls, dir_: str = "logs", prefix: str = "control") -> ControlLog:
        """dir_/<prefix>_<日時>.csv を開く(ディレクトリは自動生成)。"""
        return cls(Path(dir_) / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.csv")

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()

    def row(self, row: ControlRow) -> None:
        self._w.writerow(asdict(row))
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self) -> ControlLog:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class AxisMetrics:
    iae: float  # Σ|e|·dt(全体誤差・定常偏差)
    itae: float  # Σ t·|e|·dt(収束が遅いほど大きくなる)
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
        self._pe: float | None = None  # 直前の誤差(±180 ラップ検出用)
        self._n = 0

    def update(self, e: float, cmd: float, t: float, dt: float, tol: float) -> None:
        self._n += 1
        ae = abs(e)
        self.iae += ae * dt
        self.itae += t * ae * dt
        self.effort += abs(cmd) * dt
        if ae > self.peak:
            self.peak = ae
        # ±180 ラップの符号反転は目標跨ぎではない。幻のオーバーシュート/振動を
        # 数えないよう、該当フレームは集計から外し基準符号(_s0)を張り替える。
        wrap = self._pe is not None and abs(e - self._pe) > 180.0
        if self._n == 1:
            self._s0 = 1 if e >= 0 else -1  # 初期誤差の符号(オーバーシュート基準)
        elif wrap:
            self._s0 = 1 if e >= 0 else -1  # ラップ後の側へ基準を張り替える
        else:
            over = -self._s0 * e  # 目標(0)を跨いで反対側へ出た量
            if over > self.overshoot:
                self.overshoot = over
        sign = 1 if e > 0 else (-1 if e < 0 else 0)
        if not wrap and sign and self._prev and sign != self._prev:
            self.osc += 1
        if sign:
            self._prev = sign
        self._pe = e
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
