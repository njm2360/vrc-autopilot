from pythonosc.udp_client import SimpleUDPClient


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class VRChatOSC:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.client = SimpleUDPClient(host, port)

    # ---- 連続軸(-1..1) ------------------------------------------------
    def axis(self, name: str, value: float) -> None:
        self.client.send_message(f"/input/{name}", _clamp(float(value)))

    def move(self, forward: float = 0.0, strafe: float = 0.0) -> None:
        """前後(forward)と左右移動(strafe)を同時指定。"""
        self.axis("Vertical", forward)
        self.axis("Horizontal", strafe)

    def look(self, turn: float = 0.0, pitch: float = 0.0) -> None:
        """水平旋回(+で右)・上下視点(+で上)。

        VRChat の /input 軸は最後に送った値を保持し続けるため、制御ループでは
        0 も明示的に送って止める必要がある。両軸とも毎回送信する。
        """
        self.axis("LookHorizontal", turn)
        self.axis("LookVertical", pitch)

    def look_vertical(self, pitch: float = 0.0) -> None:
        """上下視点。+で上。"""
        self.axis("LookVertical", pitch)

    def stop(self) -> None:
        """移動・旋回を全停止(軸を0に戻す)。"""
        self.move(0.0, 0.0)
        self.axis("LookHorizontal", 0.0)
        self.axis("LookVertical", 0.0)

    # ---- ボタン(0/1) --------------------------------------------------
    def button(self, name: str, pressed: bool) -> None:
        self.client.send_message(f"/input/{name}", 1 if pressed else 0)

    def jump(self) -> None:
        self.button("Jump", True)
        self.button("Jump", False)

    def press(self) -> None:
        self.button("UseRight", True)

    def release(self) -> None:
        self.button("UseRight", False)

    def click(self) -> None:
        self.press()
        self.release()

    # ---- アバターパラメータ --------------------------------------------
    def avatar_param(self, name: str, value) -> None:
        self.client.send_message(f"/avatar/parameters/{name}", value)

    def hud_enable(self, on: bool = True) -> None:
        self.avatar_param("HUD_Enable", bool(on))

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "VRChatOSC":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
