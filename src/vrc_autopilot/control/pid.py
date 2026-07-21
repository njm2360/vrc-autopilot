import math
from dataclasses import dataclass, field


@dataclass
class PID:
    """離散 PID。

    update(error, dt) を毎周期呼ぶ。出力は [out_min, out_max] の範囲に収める。
    積分項は i_limit で絶対値を制限し、出力飽和中は積分を止める(ワインドアップ防止)。
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    out_min: float = -1.0
    out_max: float = 1.0
    i_limit: float = 1.0  # 積分項(ki*∫e)の絶対値上限
    # 非ゼロ出力の下限(不感帯補償)。>0 なら非ゼロ出力を最低 out_floor まで
    # 底上げする(VRChat 視点軸のような機器側不感帯を打ち消す)。小出力をゼロに
    # 潰す「出力不感帯」ではなく、その逆方向の補償であることに注意。
    out_floor: float = 0.0
    _i: float = field(default=0.0, init=False, repr=False)
    _prev: float | None = field(default=None, init=False, repr=False)
    # 直近 update() の内訳(ログ/デバッグ用)
    last_p: float = field(default=0.0, init=False, repr=False)
    last_i: float = field(default=0.0, init=False, repr=False)
    last_d: float = field(default=0.0, init=False, repr=False)
    last_out: float = field(default=0.0, init=False, repr=False)

    def reset(self) -> None:
        self._i = 0.0
        self._prev = None
        self.last_p = self.last_i = self.last_d = self.last_out = 0.0

    def reset_derivative(self) -> None:
        """微分履歴のみリセット(積分は保持)。目標急変時の微分キック抑制用。"""
        self._prev = None

    def update(self, error: float, dt: float) -> float:
        p = self.kp * error

        # 微分(計測ノイズをそのまま拾うので、必要なら呼び出し側で平滑化する)
        d = 0.0
        if self._prev is not None and dt > 0.0:
            # ラップした角度誤差(±180 跨ぎ)は生差が ~360 に飛んで微分キックになるので、
            # そのフレームだけ微分を捨てる(prev は更新して次フレームから復帰する)。
            if abs(error - self._prev) <= 180.0:
                d = self.kd * (error - self._prev) / dt
        self._prev = error

        # まず P+D と現在の積分で仮出力を作り、飽和していなければ積分を進める
        unsat = p + self.ki * self._i + d
        if dt > 0.0 and self.ki != 0.0:
            if self.out_min < unsat < self.out_max or (error * unsat) < 0.0:
                self._i += error * dt
                i_term = self.ki * self._i
                if i_term > self.i_limit:
                    self._i = self.i_limit / self.ki
                elif i_term < -self.i_limit:
                    self._i = -self.i_limit / self.ki

        i = self.ki * self._i
        out = max(self.out_min, min(self.out_max, p + i + d))
        # 不感帯補償: 非ゼロ出力を [out_floor, 1] へ線形リマップ(符号保持)
        if self.out_floor > 0.0 and abs(out) > 1e-3:
            out = math.copysign(
                self.out_floor + (1.0 - self.out_floor) * min(abs(out), 1.0), out
            )
            # レール(out_min/out_max)が ±1 より狭いと底上げで超えうるので再クランプ
            out = max(self.out_min, min(self.out_max, out))
        self.last_p, self.last_i, self.last_d, self.last_out = p, i, d, out
        return out
