"""操作アクチュエータ: 視点(look)・移動(move)・押下(interact)を独立に差し替えるための IF。

制御ループは指令値 [-1,1] を出すだけで、実際の注入方法(OSC / DirectInput)は実装側が
吸収する。look と move は別プロトコルなので、視点はマウス・移動は OSC、のように片方だけ
差し替えられる。InteractActuator は単発の押下で、どの実装(OSC の /input/UseRight か
マウスクリックか)を使うかは呼び出し側が渡す。

osc.VRChatOSC は全プロトコルをそのまま満たす(OSC 経由ならアダプタ不要)。
HUD 表示切替はここに含めない(spec.HUD_ENABLE_PARAM を avatar_param で送る)。
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LookActuator(Protocol):
    def look(self, turn: float = 0.0, pitch: float = 0.0) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class MoveActuator(Protocol):
    def move(self, forward: float = 0.0, strafe: float = 0.0) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class InteractActuator(Protocol):
    def press(self) -> None: ...

    def release(self) -> None: ...

    def click(self) -> None: ...


class MouseLookActuator:
    """DirectInput(相対マウス移動)で視点を操作する LookActuator。

    制御指令 [-1,1] を1フレームあたりのマウス移動量[px]へ線形変換する(実質は速度指令)。
    VRChat デスクトップのマウス視点は加速なしが前提で、ウィンドウにフォーカスが必要。
    マウスに不感帯は無いので PID の out_floor は 0 でよい(ゲイン[px/指令]は
    OSC 版とは別物。実機で要校正)。move_rel は差し替え可能(テスト用)。
    """

    def __init__(
        self,
        yaw_gain: float = 40.0,
        pitch_gain: float = 40.0,
        invert_pitch: bool = True,  # 画面Yは下が正。pitch+(上)は dy<0
        move_rel=None,
    ):
        if move_rel is None:
            import pydirectinput

            pydirectinput.PAUSE = 0.0
            move_rel = pydirectinput.moveRel
        self.yaw_gain = yaw_gain
        self.pitch_gain = pitch_gain
        self.invert_pitch = invert_pitch
        self._move_rel = move_rel

    def look(self, turn: float = 0.0, pitch: float = 0.0) -> None:
        dx = int(round(turn * self.yaw_gain))
        dy = int(round(pitch * self.pitch_gain))
        if self.invert_pitch:
            dy = -dy
        if dx or dy:
            self._move_rel(dx, dy)

    def stop(self) -> None:
        pass


class MouseClickActuator:
    """pydirectinput の左クリックで interact する InteractActuator。"""

    def __init__(self):
        import pydirectinput

        pydirectinput.PAUSE = 0.0
        self._pdi = pydirectinput

    def press(self) -> None:
        self._pdi.mouseDown()

    def release(self) -> None:
        self._pdi.mouseUp()

    def click(self) -> None:
        self._pdi.click()
