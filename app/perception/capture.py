import sys
from typing import Protocol

import numpy as np

from .spec import CAPTURE_H, CAPTURE_W


class FrameSource(Protocol):
    """グリッドを内包する領域(クライアント左上原点)を返すもの。"""

    def grab(self) -> np.ndarray:
        """HxWx3(以上) の uint8 画像を返す。frame[0,0] がクライアント左上。"""
        ...

    def close(self) -> None: ...


class WindowFocus(Protocol):
    """VRChat ウィンドウが最前面か(入力注入の前提)を返すもの。"""

    def is_active(self) -> bool: ...


class ArrayFrameSource:
    """固定の numpy 配列を返す、テスト/再生用のフレーム供給元。"""

    def __init__(self, frame: np.ndarray):
        self._frame = frame

    def set_frame(self, frame: np.ndarray) -> None:
        self._frame = frame

    def grab(self) -> np.ndarray:
        return self._frame

    def close(self) -> None:
        pass


class WindowNotFoundError(RuntimeError):
    """VRChat ウィンドウが見つからない。"""


def _enable_dpi_awareness() -> None:
    """プロセスを Per-Monitor DPI Aware にし、物理ピクセルで矩形を得られるようにする。

    VRChat はネイティブ解像度で描くため、論理ピクセルへ丸められるとブロック境界が壊れる。
    """
    import ctypes

    # Per-Monitor v2 (-4) → Per-Monitor (2) → System の順に試す
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except AttributeError, OSError:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return
    except AttributeError, OSError:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except AttributeError, OSError:
        pass


def _resolve_hwnd(title: str):
    import ctypes

    hwnd = ctypes.windll.user32.FindWindowW(None, title)
    if not hwnd:
        hwnd = _find_window_substring(title)
    return hwnd or None


def is_hwnd_foreground(hwnd) -> bool:
    import ctypes

    return bool(hwnd) and ctypes.windll.user32.GetForegroundWindow() == hwnd


def client_rect(hwnd) -> tuple[int, int, int, int] | None:
    """hwnd のクライアント領域 (left, top, width, height) をスクリーン座標(物理px)で返す。"""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32

    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None  # 最小化中など

    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None
    return pt.x, pt.y, width, height


def find_window_rect(title: str = "VRChat") -> tuple[int, int, int, int] | None:
    """タイトルからウィンドウを探しクライアント領域を返す(無ければ None)。"""
    hwnd = _resolve_hwnd(title)
    return client_rect(hwnd) if hwnd else None


def _find_window_substring(needle: str):
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    needle_low = needle.lower()
    found = []

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        if needle_low in buf.value.lower():
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(WNDENUMPROC(_cb), 0)
    return found[0] if found else None


class WindowsVRChatCapture:
    """VRChat のクライアント左上付近の小領域を mss で高速キャプチャする FrameSource。

    グリッドを内包する CAPTURE_W x CAPTURE_H 領域のみを掴むので 60fps を狙える。
    ウィンドウ矩形はキャッシュし、掴めなくなったら再解決する。
    """

    def __init__(self, window_title: str = "VRChat"):
        if sys.platform != "win32":
            raise RuntimeError(
                "WindowsVRChatCapture is Windows-only; inject a FrameSource"
            )
        import mss  # 遅延 import(テスト環境に mss/win 依存を持ち込まない)

        _enable_dpi_awareness()
        self.window_title = window_title
        self._sct = mss.mss()
        self._hwnd = None
        self._rect: tuple[int, int, int, int] | None = None

    def _resolve_rect(self) -> tuple[int, int, int, int] | None:
        self._hwnd = _resolve_hwnd(self.window_title)
        self._rect = client_rect(self._hwnd) if self._hwnd else None
        return self._rect

    def grab(self) -> np.ndarray:
        rect = self._rect or self._resolve_rect()
        if rect is None:
            raise WindowNotFoundError(f'window "{self.window_title}" not found')
        left, top, cw, ch = rect
        region = {
            "left": left,
            "top": top,
            "width": min(CAPTURE_W, cw),
            "height": min(CAPTURE_H, ch),
        }
        try:
            shot = self._sct.grab(region)
        except Exception:
            # ウィンドウが移動/クローズした可能性。次回再解決させる。
            self._hwnd = self._rect = None
            raise
        # mss は BGRA。RGB和で二値化するのでチャンネル順は不問。先頭3chのみ使う。
        return np.asarray(shot)[:, :, :3]

    def refresh_window(self) -> None:
        """ウィンドウ矩形を強制再解決する(解像度/位置変更後に呼ぶ)。"""
        self._resolve_rect()

    def is_active(self) -> bool:
        if self._hwnd is None:
            self._resolve_rect()
        return is_hwnd_foreground(self._hwnd)

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass
