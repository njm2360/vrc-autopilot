import numpy as np

from .spec import (
    BLOCK,
    CAPTURE_H,
    CAPTURE_W,
    COLS,
    IDX_CHECKSUM,
    MAGIC,
    OFFSET_X,
    OFFSET_Y,
    ROWS,
)


def pack_pose_words(
    time_ms: int,
    position: tuple[float, float, float],
    forward: tuple[float, float, float],
    up: tuple[float, float, float],
) -> np.ndarray:
    """6DoF から 12 ワード(uint32)を組み立てる。XORチェックサムも計算する。"""
    words = np.zeros(ROWS, dtype=np.uint32)
    words[0] = MAGIC
    words[1] = np.uint32(time_ms & 0xFFFFFFFF)
    words[2:5] = np.asarray(position, dtype=np.float32).view(np.uint32)
    words[5:8] = np.asarray(forward, dtype=np.float32).view(np.uint32)
    words[8:11] = np.asarray(up, dtype=np.float32).view(np.uint32)
    words[IDX_CHECKSUM] = np.bitwise_xor.reduce(words[:IDX_CHECKSUM])
    return words


def words_to_bits(words: np.ndarray) -> np.ndarray:
    """uint32[rows] を (rows, cols) bool グリッドへ展開する(MSBが左端)。"""
    shifts = np.arange(COLS - 1, -1, -1, dtype=np.uint32)
    return ((words[:, None].astype(np.uint32) >> shifts) & np.uint32(1)).astype(bool)


def render_grid(
    words: np.ndarray,
    canvas_shape: tuple[int, int] | None = None,
    origin: tuple[int, int] = (0, 0),
    background: int = 0,
    dtype=np.uint8,
) -> np.ndarray:
    """ワードを白黒ブロックグリッドとして描画した HxWx3 画像を返す。

    canvas_shape=(H, W) 未指定なら capture 領域サイズ。origin=(y, x) はキャンバス上の
    クライアント左上位置(通常 (0,0))。白=255, 黒=0。
    """
    if canvas_shape is None:
        canvas_shape = (CAPTURE_H, CAPTURE_W)
    bits = words_to_bits(words)  # (rows, cols)
    # ブロック拡大: 各ビットを block x block へ複製
    block_px = np.kron(bits, np.ones((BLOCK, BLOCK), dtype=bool))
    gh, gw = block_px.shape  # grid_h, grid_w

    canvas = np.full((*canvas_shape, 3), background, dtype=dtype)
    oy, ox = origin
    y0, x0 = oy + OFFSET_Y, ox + OFFSET_X
    if y0 + gh > canvas_shape[0] or x0 + gw > canvas_shape[1]:
        raise ValueError("grid does not fit in canvas at given origin")
    canvas[y0 : y0 + gh, x0 : x0 + gw, :] = np.where(block_px[..., None], 255, 0)
    return canvas


def render_pose(
    time_ms: int,
    position: tuple[float, float, float],
    forward: tuple[float, float, float],
    up: tuple[float, float, float],
    **render_kwargs,
) -> np.ndarray:
    """6DoF を直接グリッド画像へエンコードするショートカット。"""
    words = pack_pose_words(time_ms, position, forward, up)
    return render_grid(words, **render_kwargs)
