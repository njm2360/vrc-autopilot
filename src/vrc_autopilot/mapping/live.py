from collections.abc import Callable

from .draw import draw_map
from .mapper import RoomMapper

# ライブ操作で使うキー。matplotlib 既定のショートカット(o=ズーム, r=ホーム等)と衝突する
# ので、該当キーを全 keymap から外してから使う。
_OUR_KEYS = {" ", "space", "o", "i", "z", "r", "q", "escape"}


def _free_our_keys(rcparams) -> None:
    for name, val in list(rcparams.items()):
        if not name.startswith("keymap."):
            continue
        kept = [k for k in val if k not in _OUR_KEYS]
        if kept != val:
            rcparams[name] = kept


class LiveMap:
    """録画中の地図ウィンドウ。表示不可なら headless=True で no-op になる。"""

    def __init__(
        self,
        on_key: Callable[[str], None] | None = None,
        show_occupancy: bool = False,
    ):
        self.show_occupancy = show_occupancy
        self.closed = False
        self.headless = False
        self._plt = None
        self.fig = None
        self.ax = None

        import matplotlib

        for backend in ("TkAgg", "QtAgg"):
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt

                _free_our_keys(plt.rcParams)
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(7, 7))
                self._plt = plt
                break
            except Exception:
                continue
        else:  # どのバックエンドも開けなかった
            self.headless = True
            return

        self.fig.canvas.mpl_connect("close_event", lambda _e: self._mark_closed())
        if on_key is not None:
            self.fig.canvas.mpl_connect(
                "key_press_event", lambda e: on_key(e.key or "")
            )
        try:
            self.fig.show()
        except Exception:
            self.headless = True

    def _mark_closed(self) -> None:
        self.closed = True

    def update(self, mapper: RoomMapper, title: str | None = None) -> None:
        """地図を再描画する(重いので数Hz程度に間引いて呼ぶ)。"""
        if self.headless or self.closed or len(mapper) == 0:
            return
        self.ax.clear()
        draw_map(self.ax, mapper, show_occupancy=self.show_occupancy, title=title)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def pump(self) -> None:
        """再描画せず GUI イベントだけ処理する(キー・閉じるを取りこぼさない)。"""
        if self.headless or self.closed:
            return
        try:
            self.fig.canvas.flush_events()
        except Exception:
            self.closed = True

    def close(self) -> None:
        if self._plt is not None and self.fig is not None:
            try:
                self._plt.close(self.fig)
            except Exception:
                pass
