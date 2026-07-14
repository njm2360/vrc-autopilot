"""フィードバック制御器(制御ループへ注入する部品)。

制御ループは AxisController に (誤差, dt) を渡して指令 [-1,1] を得るだけ。PID の
ゲインや不感帯補償はここに閉じ込め、アクチュエータに合わせて差し替え・再調整する。

巡回制御のチューニング定数はすべて ``PatrolGains`` に集約する。CLI はこの既定値を
上書きするだけにして、数値の二重管理を避ける。
"""

from dataclasses import dataclass

from .pid import PID


@dataclass
class AxisController:
    """1軸のフィードバック制御器。誤差 → 指令[-1,1]。

    誤差の絶対値が tol 未満のときは指令を0にする(収束付近で行き過ぎたり、不感帯補償に
    よる最小の旋回がいつまでも残るのを止める)。不感帯補償や積分制限は内部の PID 側で
    設定する。
    """

    pid: PID
    tol: float = 0.0  # 収束とみなす範囲[誤差と同単位]。この範囲内は指令0。0で無効

    def update(self, error: float, dt: float) -> float:
        if self.tol > 0.0 and abs(error) < self.tol:
            return 0.0
        return self.pid.update(error, dt)

    def reset(self) -> None:
        self.pid.reset()

    def reset_derivative(self) -> None:
        self.pid.reset_derivative()

    # ログ用
    @property
    def last_p(self) -> float:
        return self.pid.last_p

    @property
    def last_i(self) -> float:
        return self.pid.last_i

    @property
    def last_d(self) -> float:
        return self.pid.last_d


@dataclass
class NavControllers:
    """移動追従フェーズの制御器。"""

    yaw: AxisController  # 進行方向へ向く(誤差=yaw[deg])
    forward: AxisController  # 最終ウェイポイントの減速(誤差=距離[m] → 速度)


@dataclass
class FaceControllers:
    """正対フェーズの制御器。"""

    yaw: AxisController  # 誤差=yaw[deg]
    pitch: AxisController  # 誤差=pitch[deg]


@dataclass
class PatrolGains:
    """巡回制御のチューニング定数一式(既定値はここに集約)。

    既定値は同定プラント(plant.json)上の自律探索(sim-face 相当)で調整した暫定値。
    実機(patrol-buttons)で答え合わせして更新すること(gain-tuning.md 参照)。
    """

    # ---- 移動・到達 ----
    speed: float = 0.9  # 巡航前進速度の上限(0..1)。狭所ではコーナー切りで壁に擦る
    #                     ことがあるが、経路追従で戻れるので許容する
    arrive: float = 0.35  # ウェイポイント到達半径[m]
    standoff: float = 1.0  # ボタン正面で止まる距離[m](Use到達距離内に収める)
    # ---- 収束判定・打切り ----
    face_tol: float = 2.0  # 正対(粗合わせ)とみなす角度[deg]
    settle: int = 3  # 収束判定に必要な連続フレーム数(face / align 共通)
    nav_timeout: float = 60.0  # 移動の打切り秒
    face_timeout: float = 12.0  # 正対の打切り秒
    # ---- 移動中(nav)の yaw: 穏やか・不感帯補償なし ----
    nav_turn_kp: float = 0.07
    nav_turn_ki: float = 0.025
    nav_turn_kd: float = 0.004
    # ---- 前進速度(最終ウェイポイントの減速): 誤差=距離[m] ----
    fwd_kp: float = 2.0
    fwd_kd: float = 0.05
    # ---- 正対(face)の yaw: 視点軸が反応しない範囲(≈0.50)を out_deadzone で飛び越える ----
    turn_kp: float = 0.08
    turn_ki: float = 0.01
    turn_kd: float = 0.004
    turn_ilim: float = 0.5  # yaw積分項の絶対上限
    turn_deadzone: float = 0.50  # 視点軸の不感帯補償(反応しない範囲=実測オンセット。0で無効)
    # ---- 正対(face)の pitch: pitch 軸にも不感帯(≈0.10)があるので補償する ----
    pitch_kp: float = 0.07
    pitch_ki: float = 0.015
    pitch_kd: float = 0.004
    pitch_ilim: float = 0.5
    pitch_deadzone: float = 0.11  # 実測オンセット。0=無効だと tol 直上で止まり未収束することがある
    # ---- 最終照準(align): 視点は回さず横移動で詰める ----
    align_tol: float = 0.02  # 横ずれの収束閾値[m]。0で align 無効
    align_timeout: float = 8.0  # 打切り秒
    align_stuck_time: float = 1.0  # 動けないままこの秒数経過で打切り(壁に阻まれた時)
    align_stuck_eps: float = 0.02  # 動けないとみなす移動距離[m]
    strafe_kp: float = 4.0  # 横移動の PID(誤差=横ずれ[m] → Horizontal 指令)
    strafe_ki: float = 0.8
    strafe_kd: float = 0.2
    strafe_ilim: float = 0.3  # 積分項の絶対上限
    strafe_deadzone: float = 0.0  # 移動軸の不感帯補償(実測で必要になったら)


def nav_controllers(g: PatrolGains) -> NavControllers:
    """移動追従用の制御器を組む。yaw は不感帯補償なし(連続追従では小さい指令が
    自然な遊びとして働き、暴れを防ぐ)。"""
    yaw = AxisController(
        PID(
            kp=g.nav_turn_kp,
            ki=g.nav_turn_ki,
            kd=g.nav_turn_kd,
            out_min=-1.0,
            out_max=1.0,
            i_limit=0.5,
        )
    )
    forward = AxisController(
        PID(kp=g.fwd_kp, ki=0.0, kd=g.fwd_kd, out_min=0.0, out_max=g.speed, i_limit=0.0)
    )
    return NavControllers(yaw=yaw, forward=forward)


def strafe_controller(g: PatrolGains) -> AxisController:
    """align フェーズの横移動制御器。誤差=横ずれ[m] → Horizontal指令[-1,1]。
    tol は横ずれの収束閾値(align_tol)。"""
    return AxisController(
        PID(
            kp=g.strafe_kp,
            ki=g.strafe_ki,
            kd=g.strafe_kd,
            out_min=-1.0,
            out_max=1.0,
            i_limit=g.strafe_ilim,
            out_deadzone=g.strafe_deadzone,
        ),
        tol=g.align_tol,
    )


def face_controllers(g: PatrolGains) -> FaceControllers:
    """正対用の制御器を組む。yaw/pitch とも誤差が tol 未満なら指令0。yaw は不感帯補償つき。"""
    yaw = AxisController(
        PID(
            kp=g.turn_kp,
            ki=g.turn_ki,
            kd=g.turn_kd,
            out_min=-1.0,
            out_max=1.0,
            i_limit=g.turn_ilim,
            out_deadzone=g.turn_deadzone,
        ),
        tol=g.face_tol,
    )
    pitch = AxisController(
        PID(
            kp=g.pitch_kp,
            ki=g.pitch_ki,
            kd=g.pitch_kd,
            out_min=-1.0,
            out_max=1.0,
            i_limit=g.pitch_ilim,
            out_deadzone=g.pitch_deadzone,
        ),
        tol=g.face_tol,
    )
    return FaceControllers(yaw=yaw, pitch=pitch)
