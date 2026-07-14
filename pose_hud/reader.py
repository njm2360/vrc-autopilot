import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Iterator

import numpy as np

from .capture import FrameSource
from .decode import DecodeResult, DecodeStatus, decode_pose
from .pose import Pose

logger = logging.getLogger("pose_hud")

_STATS_WINDOW = 120  # fps 推定に使う直近サンプル数
_WARN_AFTER = 120  # 連続失敗がこれを超えたら一度だけ警告する


def _fps_from(times: deque[float]) -> float:
    """monotonic タイムスタンプ列から実効fpsを推定する。"""
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    return (len(times) - 1) / span if span > 0 else 0.0


@dataclass
class ReaderStats:
    """読み取り統計のスナップショット。"""

    frames_grabbed: int = 0  # キャプチャ回数(重複含む総取得数)
    decode_ok: int = 0  # 検証OK(重複含む)
    decode_fail: int = 0  # MAGIC/チェックサム不一致
    new_frames: int = 0  # 新規(time_ms 更新)ポーズ数
    duplicate_skipped: int = 0  # 同一 time_ms の二重読み
    consecutive_fail: int = 0  # 連続デコード失敗数(成功でリセット)
    last_status: DecodeStatus | None = None
    capture_fps: float = 0.0  # キャプチャの処理速度
    frame_fps: float = 0.0  # 新規ポーズの実効fps

    @property
    def success_rate(self) -> float:
        """検証OK率(全キャプチャに対する割合)。"""
        total = self.decode_ok + self.decode_fail
        return self.decode_ok / total if total else 0.0


class PoseReader:
    """VRChat HUD からポーズを読み続けるリーダ。

    使い方::

        reader = PoseReader()              # 既定で WindowsVRChatCapture を使用
        reader.start()
        pose = reader.get_latest()         # 最新ポーズ(なければ None)
        for pose in reader.poses():        # 新フレームをブロッキング取得
            ...
        reader.stop()

    テストや非Windows環境では ``source=ArrayFrameSource(frame)`` を注入する。
    """

    def __init__(self, source: FrameSource | None = None):
        if source is None:
            from .capture import WindowsVRChatCapture

            source = WindowsVRChatCapture()
        self.source = source

        self.stats = ReaderStats()
        self._latest: Pose | None = None
        self._last_time_ms: int | None = None

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._warned = False
        self._queue: queue.Queue[Pose] = queue.Queue()

        self._capture_times: deque[float] = deque(maxlen=_STATS_WINDOW)
        self._frame_times: deque[float] = deque(maxlen=_STATS_WINDOW)

    # ---- ライフサイクル -------------------------------------------------
    def start(self) -> "PoseReader":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="PoseReader", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, join: bool = True, timeout: float = 2.0) -> None:
        self._stop.set()
        if join and self._thread:
            self._thread.join(timeout)
        self.source.close()

    def __enter__(self) -> "PoseReader":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- 取得API -------------------------------------------------------
    def get_latest(self) -> Pose | None:
        """直近の有効ポーズ。まだ無ければ None。スレッドセーフ。"""
        with self._lock:
            return self._latest

    def get_stats(self) -> ReaderStats:
        """統計のコピーを返す。"""
        with self._lock:
            return replace(self.stats)

    def poses(self, timeout: float | None = None) -> Iterator[Pose]:
        """新規ポーズをブロッキングで yield し続けるジェネレータ。

        timeout 秒新フレームが来なければ終了(None なら stop() まで無限)。
        """
        while not self._stop.is_set():
            try:
                yield self._queue.get(timeout=timeout if timeout is not None else 0.5)
            except queue.Empty:
                if timeout is not None:
                    return
                continue

    # ---- 内部ループ ----------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> DecodeResult:
        """1フレームをデコードして統計・状態を更新する(単体テスト可能)。"""
        now = time.monotonic()
        result = decode_pose(frame)

        with self._lock:
            self.stats.frames_grabbed += 1
            self.stats.last_status = result.status
            self._capture_times.append(now)
            self._update_fps()

            if result.ok:
                self.stats.decode_ok += 1
                self.stats.consecutive_fail = 0
                self._warned = False
                pose = result.pose
                assert pose is not None
                if pose.time_ms == self._last_time_ms:
                    self.stats.duplicate_skipped += 1
                else:
                    self._last_time_ms = pose.time_ms
                    self._latest = pose
                    self.stats.new_frames += 1
                    self._frame_times.append(now)
                    self._queue.put(pose)
            else:
                self.stats.decode_fail += 1
                self.stats.consecutive_fail += 1
                self._maybe_warn()

        return result

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.source.grab()
            except Exception as exc:  # noqa: BLE001 - ウィンドウ消失等は回復対象
                logger.debug("grab failed: %s", exc)
                self._stop.wait(0.2)
                continue
            self.process_frame(frame)

    # ---- ヘルパ(ロック保持前提) ---------------------------------------
    def _maybe_warn(self) -> None:
        if not self._warned and self.stats.consecutive_fail >= _WARN_AFTER:
            self._warned = True
            logger.warning(
                "no valid HUD for %d consecutive frames (last=%s). "
                "menu open? HUD_Enable=false? wrong window?",
                self.stats.consecutive_fail,
                self.stats.last_status,
            )

    def _update_fps(self) -> None:
        self.stats.capture_fps = _fps_from(self._capture_times)
        self.stats.frame_fps = _fps_from(self._frame_times)
