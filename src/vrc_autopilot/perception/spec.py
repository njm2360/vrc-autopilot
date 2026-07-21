# HUD ビットグリッドのプロトコル定数
MAGIC = 0x5AC3E7A1

# HUD有効化: Parameter名
HUD_ENABLE_PARAM = "PixelHUDEnable"

# ワード配列(uint32[12])のレイアウト
IDX_MAGIC = 0
IDX_TIME = 1
IDX_POS = slice(2, 5)  # x, y, z
IDX_FWD = slice(5, 8)  # forward x, y, z
IDX_UP = slice(8, 11)  # up x, y, z
IDX_CHECKSUM = 11
WORD_COUNT = 12

# グリッド
OFFSET_X = 0  # _OffsetX: グリッド左上のXオフセット
OFFSET_Y = 0  # _OffsetY: グリッド左上のYオフセット
BLOCK = 1  # _BlockPx: 1ビットの一辺。白=1, 黒=0
ROWS = WORD_COUNT
COLS = 32

GRID_W = COLS * BLOCK  # 32
GRID_H = ROWS * BLOCK  # 12

# キャプチャ切り出し領域
CAPTURE_W = OFFSET_X + GRID_W  # 32
CAPTURE_H = OFFSET_Y + GRID_H  # 12

# RGB和の二値化しきい値
THRESHOLD = 3 * 128
