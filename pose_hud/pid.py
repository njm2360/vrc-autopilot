import math
from dataclasses import dataclass, field


@dataclass
class PID:
    """離散 PID。

    update(error, dt) を毎周期呼ぶ。出力は [out_min, out_max] の範囲に収める。
    積分項は i_limit で絶対値を制限し、出力が上限/下限に張り付いている間は積分を
    止める(積分が溜まりすぎて指令が行き過ぎるのを防ぐ)。
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    out_min: float = -1.0
    out_max: float = 1.0
    i_limit: float = 1.0  # 積分項(ki*∫e)の絶対値上限
    # 出力の不感帯補償。>0 なら、非ゼロの出力を最低でも out_deadzone まで底上げする
    # (VRChat の視点軸のように、一定値以下がほとんど反応しない機器の不感帯を打ち消す)。
    out_deadzone: float = 0.0
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
        """微分履歴だけをリセットする(積分は保持)。目標が急に変わったときに
        微分項が大きく跳ねるのを抑えるために使う。"""
        self._prev = None

    def update(self, error: float, dt: float) -> float:
        p = self.kp * error

        # 微分(計測ノイズをそのまま拾うので、必要なら呼び出し側で平滑化する)
        d = 0.0
        if self._prev is not None and dt > 0.0:
            d = self.kd * (error - self._prev) / dt
        self._prev = error

        # まず P+D と現在の積分で仮出力を作り、飽和していなければ積分を進める
        unsat = p + self.ki * self._i + d
        if dt > 0.0 and self.ki != 0.0:
            if self.out_min < unsat < self.out_max or (error * self._i) < 0.0:
                self._i += error * dt
                i_term = self.ki * self._i
                if i_term > self.i_limit:
                    self._i = self.i_limit / self.ki
                elif i_term < -self.i_limit:
                    self._i = -self.i_limit / self.ki

        i = self.ki * self._i
        out = max(self.out_min, min(self.out_max, p + i + d))
        # 不感帯補償: 非ゼロの出力を最低 out_deadzone まで底上げして、ほとんど反応しない
        # 範囲を飛び越える。生の出力が大きいほど 1 に近づき、ごく小さい出力でちょうど
        # out_deadzone になる(符号は保持)。
        if self.out_deadzone > 0.0 and abs(out) > 1e-3:
            out = math.copysign(
                self.out_deadzone + (1.0 - self.out_deadzone) * min(abs(out), 1.0), out
            )
        self.last_p, self.last_i, self.last_d, self.last_out = p, i, d, out
        return out
