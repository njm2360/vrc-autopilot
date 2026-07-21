import time

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
        """前後(forward)と左右移動(strafe)"""
        self.axis("Vertical", forward)
        self.axis("Horizontal", strafe)

    def look(self, turn: float = 0.0, pitch: float = 0.0) -> None:
        """水平旋回(+で右)・上下視点(+で上)"""
        self.axis("LookHorizontal", turn)
        self.axis("LookVertical", pitch)

    def stop(self) -> None:
        """移動・旋回を全停止(軸を0に戻す)。"""
        self.move(0.0, 0.0)
        self.look(0.0, 0.0)

    # ---- ボタン(0/1) --------------------------------------------------
    def button(self, name: str, pressed: bool) -> None:
        self.client.send_message(f"/input/{name}", 1 if pressed else 0)

    def jump(self) -> None:
        self.button("Jump", True)
        self.button("Jump", False)

    def set_run(self, on: bool = True) -> None:
        self.button("Run", on)

    def press(self) -> None:
        self.button("UseRight", True)

    def release(self) -> None:
        self.button("UseRight", False)

    def click(self) -> None:
        self.press()
        time.sleep(0.05)
        self.release()

    # ---- アバターパラメータ --------------------------------------------
    def avatar_param(self, name: str, value) -> None:
        self.client.send_message(f"/avatar/parameters/{name}", value)

    def close(self) -> None:
        self.stop()
        self.set_run(False)
        self.client.close()

    def __enter__(self) -> VRChatOSC:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
