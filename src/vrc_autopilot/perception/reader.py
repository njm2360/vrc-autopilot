import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np

from ..core.pose import Pose
from .capture import FrameSource
from .decode import DecodeResult, DecodeStatus, decode_frame
from .spec import HUD_ENABLE_PARAM

logger = logging.getLogger(__name__)

_STATS_WINDOW = 120  # fps 推定に使う直近サンプル数
_WARN_AFTER = 120  # 連続失敗がこれを超えたら一度だけ警告する
_QUEUE_MAX = 256  # poses() 用キューの上限(未消費なら古いものから捨てる)


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
    new_poses: int = 0  # 新規(time_ms 更新)ポーズ数
    duplicate_skipped: int = 0  # 同一 time_ms の二重読み
    consecutive_fail: int = 0  # 連続デコード失敗数(成功でリセット)
    last_status: DecodeStatus | None = None
    capture_fps: float = 0.0  # キャプチャの処理速度
    pose_fps: float = 0.0  # 新規ポーズの実効fps

    @property
    def success_rate(self) -> float:
        """検証OK率(全キャプチャに対する割合)。"""
        total = self.decode_ok + self.decode_fail
        return self.decode_ok / total if total else 0.0


class PoseReader:
    """VRChat HUD からポーズを読み続けるリーダ(既定は WindowsVRChatCapture)。

    start() 後、get_latest() で最新ポーズ、poses() で新規ポーズをブロッキング取得。
    テストや非Windows環境では source=ArrayFrameSource(frame) を注入する。
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
        self._stopped = False
        self._warned = False
        self._queue: queue.Queue[Pose] = queue.Queue(maxsize=_QUEUE_MAX)

        self._capture_times: deque[float] = deque(maxlen=_STATS_WINDOW)
        self._pose_times: deque[float] = deque(maxlen=_STATS_WINDOW)

    # ---- ライフサイクル -------------------------------------------------
    def start(self) -> PoseReader:
        if self._stopped:
            raise RuntimeError("PoseReader is single-use")
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="PoseReader", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, join: bool = True, timeout: float = 2.0) -> None:
        self._stopped = True
        self._stop.set()
        if self._thread:
            if join:
                self._thread.join(timeout)
        else:
            self.source.close()  # worker 未起動なのでここで閉じる

    def __enter__(self) -> PoseReader:
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
        result = decode_frame(frame)

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
                    self.stats.new_poses += 1
                    self._pose_times.append(now)
                    self._enqueue(pose)
            else:
                self.stats.decode_fail += 1
                self.stats.consecutive_fail += 1
                self._maybe_warn()

        return result

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    frame = self.source.grab()
                except Exception as exc:  # ウィンドウ消失等は回復対象
                    self._note_failure(f"capture failed: {exc} (VRChat window gone?)")
                    continue
                try:
                    self.process_frame(frame)
                except Exception as exc:  # フレーム不正(リサイズ等)は回復対象
                    self._note_failure(f"frame processing failed: {exc}")
        finally:
            self.source.close()  # grab と同じスレッドで閉じる(mss 非スレッドセーフ)

    def _note_failure(self, detail: str) -> None:
        """キャプチャ/処理の例外を失敗として計上し、少し待って再試行する。"""
        logger.debug(detail)
        with self._lock:
            self.stats.consecutive_fail += 1
            self._maybe_warn(detail)
        self._stop.wait(0.2)

    # ---- ヘルパ(ロック保持前提) ---------------------------------------
    def _enqueue(self, pose: Pose) -> None:
        """poses() 用キューへ追加。満杯なら最古を捨てて最新を優先する。

        get_latest() しか使わない消費者でもキューが無限に溜まらないようにする。
        """
        while True:
            try:
                self._queue.put_nowait(pose)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

    def _maybe_warn(self, detail: str | None = None) -> None:
        if not self._warned and self.stats.consecutive_fail >= _WARN_AFTER:
            self._warned = True
            logger.warning(
                "no valid HUD for %d consecutive frames (last=%s). "
                "menu open? %s=false? wrong window?",
                self.stats.consecutive_fail,
                detail or self.stats.last_status,
                HUD_ENABLE_PARAM,
            )

    def _update_fps(self) -> None:
        self.stats.capture_fps = _fps_from(self._capture_times)
        self.stats.pose_fps = _fps_from(self._pose_times)
