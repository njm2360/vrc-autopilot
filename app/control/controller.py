"""フィードバック制御器(制御ループへ注入する部品)。

制御ループは AxisController に (誤差, dt) を渡して指令 [-1,1] を得るだけ。PID の
ゲインや不感帯補償はここに閉じ込め、アクチュエータに合わせて差し替え・再調整する。

巡回制御のチューニング定数はすべて PatrolGains に集約する。CLI はこの既定値を
上書きするだけにして、数値の二重管理を避ける。
"""

from dataclasses import dataclass

from .pid import PID


@dataclass
class AxisController:
    """1軸のフィードバック制御器。誤差 → 指令[-1,1]。

    誤差の絶対値が tol 未満なら指令0(不感帯補償による微小指令の残留を止める)。
    不感帯補償や積分制限は内部の PID 側で設定する。
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
class TranslateControllers:
    """視点固定の並進フェーズの制御器。"""

    forward: AxisController  # 誤差=目標までの前方距離[m] → Vertical指令
    strafe: AxisController  # 誤差=目標までの右方距離[m] → Horizontal指令


@dataclass
class FaceControllers:
    """正対フェーズの制御器。"""

    yaw: AxisController  # 誤差=yaw[deg]
    pitch: AxisController  # 誤差=pitch[deg]


@dataclass
class PatrolGains:
    """巡回制御のチューニング定数一式(既定値はここに集約)。

    既定値は同定プラント(plant.json)上で全フェーズの制御ループを回して検証した値
    (根拠と安定範囲は gain-tuning.md)。実機で答え合わせして更新すること。
    """

    # ---- 移動・到達 ----
    speed: float = (
        0.9  # 巡航前進速度の上限(0..1)。狭所の壁擦りは経路追従で戻れるので許容
    )
    arrive_radius: float = 0.35  # ウェイポイント到達半径[m]
    nav_lookahead: float = 1.2  # 経路先読み(carrot)の弧長[m]。狭所は 0.8 程度に
    standoff: float = 1.0  # ボタン正面で止まる距離[m](Use到達距離内に収める)
    # ---- 収束判定・打切り ----
    face_tol: float = 0.2  # 正対とみなす角度[deg]。そのまま最終照準の精度になる
    settle_frames: int = 3  # 収束判定に必要な連続フレーム数(face / align 共通)
    nav_timeout: float = 60.0  # 移動の打切り秒
    face_timeout: float = 12.0  # 正対の打切り秒
    # ---- 移動中(nav)の yaw: face と同機構の不感帯補償つき(tol は入れない)。
    nav_turn_kp: float = 0.04
    nav_turn_ki: float = 0.025
    nav_turn_kd: float = 0.002
    nav_turn_deadzone: float = 0.50
    # ---- 視点固定の並進(hold-view move): 進行方向へ回さず forward/strafe を体フレームで
    #      合成して経路を追う。誤差=目標までの残距離[m]の体フレーム成分。指令上限は speed。 ----
    translate_kp: float = 1.0  # 安全域 0.45〜約2.5(下は不感帯で失速、上は振動)
    translate_ki: float = 0.0  # 定常外乱が無ければ0(入れると斜めで微小な行き過ぎ)
    translate_kd: float = 0.0  # 打ち消す遅れが無く GM を削るだけ(0.1 で GM ×9→×1.4)
    translate_ilim: float = 0.3
    # ---- 前進速度(最終ウェイポイントの減速): 誤差=距離[m] ----
    fwd_kp: float = 2.0
    fwd_kd: float = 0.05
    # ---- 正対(face)の yaw: 視点軸が反応しない範囲(0.50)を out_floor で飛び越える。
    #      kd は入れない(打ち消す遅れが無く、高周波ゲインを上げて GM を削るだけ) ----
    turn_kp: float = 0.05
    turn_ki: float = 0.005
    turn_kd: float = 0.0
    turn_ilim: float = 0.5  # yaw積分項の絶対上限
    turn_deadzone: float = 0.50
    # ---- 正対(face)の pitch: pitch 軸にも不感帯(0.10)があるので補償する。kd=0 は yaw と同じ理由 ----
    pitch_kp: float = 0.07
    pitch_ki: float = 0.008
    pitch_kd: float = 0.0
    pitch_ilim: float = 0.5
    pitch_deadzone: float = 0.10
    # ---- 最終照準(align): 視点は回さず横移動で詰める ----
    align_tol: float = 0.005  # 横ずれの収束閾値[m]。0で align 無効
    align_timeout: float = 8.0  # 打切り秒
    align_stuck_time: float = 1.0  # 動けないままこの秒数経過で打切り(壁に阻まれた時)
    align_stuck_eps: float = 0.02  # 動けないとみなす移動距離[m]
    strafe_kp: float = 4.0  # 横移動の PID(誤差=横ずれ[m] → Horizontal 指令)
    strafe_ki: float = 0.8
    strafe_kd: float = 0.1  # 0.2だとむだ時間増加時に不感帯ブーストと結合してチャタる
    strafe_ilim: float = 0.3  # 積分項の絶対上限
    strafe_deadzone: float = 0.10


def nav_controllers(g: PatrolGains) -> NavControllers:
    """移動追従用の制御器を組む。yaw は face と同じ不感帯補償つき(tol は入れない。
    根拠と安定範囲は gain-tuning.md の nav 節)。"""
    yaw = AxisController(
        PID(
            kp=g.nav_turn_kp,
            ki=g.nav_turn_ki,
            kd=g.nav_turn_kd,
            out_min=-1.0,
            out_max=1.0,
            i_limit=0.5,
            out_floor=g.nav_turn_deadzone,
        )
    )
    forward = AxisController(
        PID(kp=g.fwd_kp, ki=0.0, kd=g.fwd_kd, out_min=0.0, out_max=g.speed, i_limit=0.0)
    )
    return NavControllers(yaw=yaw, forward=forward)


def translate_controllers(g: PatrolGains) -> TranslateControllers:
    """視点固定の並進用の制御器を組む。前後・左右を同じゲインの独立 PID で詰める。

    指令は両軸とも ±speed に制限する。移動軸の不感帯は小さく(|指令|<0.10 で
    速度ゼロ)、kp·arrive_radius がこれを上回る限り目標手前で失速しないため補償は入れない
    (kp > 0.10/arrive_radius ≈ 0.29 を保つこと)。前後と左右で実速度は非対称(forward が
    strafe の約2倍)だが、各軸が独立に残距離を詰めるだけなので同ゲインで運用する。
    """

    def axis() -> AxisController:
        return AxisController(
            PID(
                kp=g.translate_kp,
                ki=g.translate_ki,
                kd=g.translate_kd,
                out_min=-g.speed,
                out_max=g.speed,
                i_limit=g.translate_ilim,
            )
        )

    return TranslateControllers(forward=axis(), strafe=axis())


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
            out_floor=g.strafe_deadzone,
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
            out_floor=g.turn_deadzone,
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
            out_floor=g.pitch_deadzone,
        ),
        tol=g.face_tol,
    )
    return FaceControllers(yaw=yaw, pitch=pitch)
