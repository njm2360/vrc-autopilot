import enum
from dataclasses import dataclass

import numpy as np

from ..core.pose import Pose
from .spec import (
    BLOCK,
    COLS,
    IDX_CHECKSUM,
    IDX_MAGIC,
    IDX_TIME,
    MAGIC,
    OFFSET_X,
    OFFSET_Y,
    ROWS,
    THRESHOLD,
)


class DecodeStatus(enum.Enum):
    OK = "ok"
    MAGIC_MISMATCH = "magic_mismatch"
    CHECKSUM_MISMATCH = "checksum_mismatch"


@dataclass(frozen=True)
class DecodeResult:
    status: DecodeStatus
    words: np.ndarray  # uint32[12] 生ワード(常に埋まる)
    pose: Pose | None = None

    @property
    def ok(self) -> bool:
        return self.status is DecodeStatus.OK


# ブロック中心座標と MSB-左のパック重み。
_half = BLOCK // 2
_CX = OFFSET_X + np.arange(COLS) * BLOCK + _half  # (cols,)
_CY = OFFSET_Y + np.arange(ROWS) * BLOCK + _half  # (rows,)
_WEIGHTS = np.uint64(1) << np.arange(COLS - 1, -1, -1, dtype=np.uint64)  # 列0が最上位


def sample_bits(frame: np.ndarray) -> np.ndarray:
    """フレームからブロック中心を一括サンプルし (rows, cols) の bool 配列を返す。

    frame: HxWx3 以上 (BGR/RGB どちらでも / アルファ付きも可)。
    グリッド原点はクライアント左上 = frame[0,0] を前提。
    """
    if frame.ndim != 3:
        raise ValueError(f"frame must be HxWxC, got shape {frame.shape}")
    need_h = int(_CY[-1]) + 1
    need_w = int(_CX[-1]) + 1
    if frame.shape[0] < need_h or frame.shape[1] < need_w:
        raise ValueError(
            f"frame {frame.shape[:2]} too small for grid (need >= {need_h}x{need_w})"
        )
    samples = frame[np.ix_(_CY, _CX)][:, :, :3].astype(np.uint16)
    rgb_sum = samples.sum(axis=2)  # (rows, cols)
    return rgb_sum > THRESHOLD


def pack_words(bits: np.ndarray) -> np.ndarray:
    """(rows, cols) bool 配列を uint32[rows] ワードにパックする(MSBが左端)。"""
    words = bits.astype(np.uint64) @ _WEIGHTS  # (rows,)
    return words.astype(np.uint32)


def decode_words(frame: np.ndarray) -> np.ndarray:
    """フレームから 12 ワードを復元する(検証なし)。uint32[12]。"""
    return pack_words(sample_bits(frame))


def validate_words(words: np.ndarray) -> DecodeStatus:
    """MAGIC と XOR チェックサムを検証する。"""
    if int(words[IDX_MAGIC]) != MAGIC:
        return DecodeStatus.MAGIC_MISMATCH
    xor = np.bitwise_xor.reduce(words[:IDX_CHECKSUM].astype(np.uint32))
    if np.uint32(xor) != words[IDX_CHECKSUM]:
        return DecodeStatus.CHECKSUM_MISMATCH
    return DecodeStatus.OK


def words_to_pose(words: np.ndarray) -> Pose:
    """検証済みワードから Pose を構築する(float32ビットパターンを解釈)。"""
    floats = words[2:11].astype(np.uint32).view(np.float32)
    return Pose(
        time_ms=int(words[IDX_TIME]),
        position=tuple(float(v) for v in floats[0:3]),
        forward=tuple(float(v) for v in floats[3:6]),
        up=tuple(float(v) for v in floats[6:9]),
    )


def decode_frame(frame: np.ndarray) -> DecodeResult:
    """フレームをデコードし検証まで行う(重複フレーム判定は PoseReader 側)。"""
    words = decode_words(frame)
    status = validate_words(words)
    pose = words_to_pose(words) if status is DecodeStatus.OK else None
    return DecodeResult(status=status, words=words, pose=pose)


def decode_pose(frame: np.ndarray) -> Pose | None:
    """検証込みでデコードし、有効なら Pose を返す(失敗は None)。"""
    return decode_frame(frame).pose
