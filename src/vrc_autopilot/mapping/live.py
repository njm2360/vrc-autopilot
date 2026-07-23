from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum, auto

from .draw import draw_map
from .mapper import RoomMapper


class Action(Enum):
    PAUSE = auto()
    OUTER = auto()
    INNER = auto()
    TOGGLE_MODE = auto()
    REWIND = auto()
    DISCARD = auto()
    SAVE = auto()


_BUTTONS = [
    (Action.PAUSE, "start"),
    (Action.TOGGLE_MODE, "inner"),
    (Action.REWIND, "rewind"),
    (Action.DISCARD, "discard"),
    (Action.SAVE, "save"),
]

_KEYMAP = {
    " ": Action.PAUSE,
    "space": Action.PAUSE,
    "o": Action.OUTER,
    "i": Action.INNER,
    "z": Action.REWIND,
    "r": Action.DISCARD,
    "s": Action.SAVE,
}

_OUR_KEYS = frozenset(_KEYMAP)


_FLASH_SEC = 2.5


class NoDisplayError(RuntimeError):
    pass


def _free_our_keys(rcparams) -> None:
    for name, val in list(rcparams.items()):
        if not name.startswith("keymap."):
            continue
        kept = [k for k in val if k not in _OUR_KEYS]
        if kept != val:
            rcparams[name] = kept


class LiveMap:
    """録画中の地図ウィンドウ。開けなければ NoDisplayError。

    操作ボタン・ショートカット・ステータスはすべてウィンドウ内で完結する
    """

    def __init__(
        self,
        on_action: Callable[[Action], None] | None = None,
        show_occupancy: bool = False,
    ):
        self.show_occupancy = show_occupancy
        self._on_action = on_action or (lambda _a: None)
        self.closed = False
        self._plt = None
        self.fig = None
        self.ax = None
        self._buttons: dict[Action, object] = {}
        self._flash_text = ""
        self._flash_until = 0.0

        import matplotlib

        for backend in ("TkAgg", "QtAgg"):
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt

                _free_our_keys(plt.rcParams)
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(7, 7.5))
                self._plt = plt
                break
            except Exception:
                continue
        else:
            raise NoDisplayError("no GUI backend available (tried TkAgg, QtAgg)")

        self.fig.canvas.mpl_connect("close_event", lambda _e: self._mark_closed())
        self.fig.canvas.mpl_connect("key_press_event", lambda e: self._on_key(e.key))
        self.fig.subplots_adjust(bottom=0.15)
        self._build_buttons()
        try:
            self.fig.canvas.manager.set_window_title("map_room")
        except Exception:
            pass
        try:
            self.fig.show()
        except Exception as exc:
            raise NoDisplayError(str(exc)) from exc

    def _on_key(self, key: str | None) -> None:
        action = _KEYMAP.get(key or "")
        if action is not None:
            self._on_action(action)

    def _build_buttons(self) -> None:
        from matplotlib.widgets import Button

        n = len(_BUTTONS)
        gap, left, span, h = 0.008, 0.03, 0.94, 0.06
        bw = (span - gap * (n - 1)) / n
        for i, (action, label) in enumerate(_BUTTONS):
            axb = self.fig.add_axes([left + i * (bw + gap), 0.03, bw, h])
            btn = Button(axb, label)
            btn.label.set_fontsize(12)
            btn.on_clicked(lambda _e, a=action: self._on_action(a))
            self._buttons[action] = btn

    def _mark_closed(self) -> None:
        self.closed = True

    def flash(self, text: str) -> None:
        self._flash_text = text
        self._flash_until = time.monotonic() + _FLASH_SEC

    def _status_line(self, mapper: RoomMapper, paused: bool) -> str:
        w, d = mapper.dimensions()
        if paused:
            state = "READY" if len(mapper) == 0 else "PAUSED"
        else:
            state = "REC"
        status = (
            f"[{state}]   mode={mapper.mode}   "
            f"{w:.2f} × {d:.2f} m   {len(mapper)} pts   "
            f"seg={mapper.num_segments}   path={mapper.path_length():.2f} m"
        )
        if time.monotonic() < self._flash_until:
            status += f"\n{self._flash_text}"
        return status

    def _update_buttons(self, mapper: RoomMapper, paused: bool) -> None:
        if paused:
            pause_label = "start" if len(mapper) == 0 else "resume"
        else:
            pause_label = "pause"
        self._buttons[Action.PAUSE].label.set_text(pause_label)
        other = "inner" if mapper.mode == "outer" else "outer"
        self._buttons[Action.TOGGLE_MODE].label.set_text(other)

    def update(self, mapper: RoomMapper, *, paused: bool = False) -> None:
        if self.closed:
            return
        self._update_buttons(mapper, paused)
        status = self._status_line(mapper, paused)
        self.ax.clear()
        if len(mapper) == 0:
            self.ax.set_title(status)
        else:
            draw_map(self.ax, mapper, show_occupancy=self.show_occupancy, title=status)
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:  # 描画中に窓を閉じた場合など。終了扱いにする
            self.closed = True

    def pump(self) -> None:
        if self.closed:
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
