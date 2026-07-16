"""制御の記録と応答指標(チューニング用)。

Recorder はフレームごとの行の記録先(CSV 等)の抽象。AxisAccumulator は1軸の
誤差・指令の列から IAE/ITAE などの応答指標を積算し、snapshot() で AxisMetrics
にまとめる。制御ループからは独立した純粋な計算で、recorder を付けたときだけ走る。
"""

import csv
from dataclasses import dataclass
from typing import Protocol


class Recorder(Protocol):
    def row(self, **kw) -> None: ...


class NullRecorder:
    def row(self, **kw) -> None:
        pass


class ListRecorder:
    """行を list[dict] に貯める Recorder(テスト・オフライン解析用)。"""

    def __init__(self):
        self.rows: list[dict] = []

    def row(self, **kw) -> None:
        self.rows.append(kw)


class ControlLog:
    """行を CSV に書き出す Recorder(実機ログ・sim-video の入力)。列は FIELDS 固定。"""

    FIELDS = [
        "t",  # 開始からの経過秒
        "phase",  # nav / move / face / turn / align
        "target",  # ターゲット名
        "wp",  # 追従中のウェイポイント番号(nav/moveのみ)
        "dt",  # 前フレームからの実経過秒
        "x",
        "y",
        "z",
        "yaw",
        "pitch",
        "tx",
        "ty",
        "tz",
        "dist",  # ターゲットまでの水平距離[m]
        "yaw_err",
        "pitch_err",
        "lat_err",  # 横方向誤差[m](alignのみ。+なら目標が右)
        "fwd_err",  # 目標までの前方距離[m](moveのみ)
        "right_err",  # 目標までの右方距離[m](moveのみ)
        "turn_p",
        "turn_i",
        "turn_d",
        "turn",  # yaw(LookHorizontal)PID内訳と出力
        "pitch_p",
        "pitch_i",
        "pitch_d",
        "pitch_cmd",
        "strafe_p",
        "strafe_i",
        "strafe_d",
        "strafe",  # Horizontal(横移動)PID内訳と出力(alignのみ)
        "fwd",  # Vertical(前進)出力
        "fwd_factor",  # 向きズレによる前進減衰係数
    ]

    def __init__(self, path):
        self.path = path
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS, extrasaction="ignore")
        self._w.writeheader()

    def row(self, **kw) -> None:
        self._w.writerow({k: kw.get(k, "") for k in self.FIELDS})
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


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
        # ±180 を跨ぐと誤差符号が瞬間反転する(目標を跨いでいない)。この幻の
        # オーバーシュート/振動を数えないよう、ラップしたフレームは集計から外し、
        # 以降のフレームを誤判定しないよう基準符号(_s0)をラップ後の符号へ張り替える。
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
