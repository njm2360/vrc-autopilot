"""合成画像でのエンコード→デコード ラウンドトリップ検証(実VRChat不要 / CI可能)。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.perception.capture import ArrayFrameSource
from app.perception.decode import DecodeStatus, decode_pose, decode_words
from app.perception.encode import pack_pose_words, render_grid, render_pose
from app.perception.reader import PoseReader
from app.perception.spec import CAPTURE_H, CAPTURE_W, MAGIC

SAMPLE_POSES = [
    (0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    (123456, (1.5, -2.25, 42.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    (0xFFFFFFFF, (-1000.125, 7.0, 3.5), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (999, (12.34, 56.78, -90.12), (0.6, 0.0, 0.8), (0.0, 1.0, 0.0)),
]


@pytest.mark.parametrize("time_ms,pos,fwd,up", SAMPLE_POSES)
def test_roundtrip_pose(time_ms, pos, fwd, up):
    frame = render_pose(time_ms, pos, fwd, up)
    result = decode_pose(frame)

    assert result.status is DecodeStatus.OK
    p = result.pose
    assert p.time_ms == (time_ms & 0xFFFFFFFF)
    # float32 で往復するので厳密一致(丸めなし)
    assert p.position == tuple(np.float32(v) for v in pos)
    assert p.forward == tuple(np.float32(v) for v in fwd)
    assert p.up == tuple(np.float32(v) for v in up)


def test_yaw_pitch_values():
    # +X を向く => yaw = atan2(1, 0) = 90度
    frame = render_pose(0, (0, 0, 0), (1.0, 0.0, 0.0), (0, 1, 0))
    p = decode_pose(frame).pose
    assert math.isclose(p.yaw_deg, 90.0, abs_tol=1e-4)
    assert math.isclose(p.pitch_deg, 0.0, abs_tol=1e-4)

    # 45度上向き
    inv = 1.0 / math.sqrt(2)
    frame = render_pose(0, (0, 0, 0), (0.0, inv, inv), (0, 1, 0))
    p = decode_pose(frame).pose
    assert math.isclose(p.pitch_deg, 45.0, abs_tol=1e-3)


def test_words_roundtrip():
    words = pack_pose_words(42, (1.0, 2.0, 3.0), (0, 0, 1.0), (0, 1.0, 0))
    assert int(words[0]) == MAGIC
    decoded = decode_words(render_grid(words))
    np.testing.assert_array_equal(words, decoded)


def test_magic_mismatch_on_blank():
    frame = np.zeros((CAPTURE_H, CAPTURE_W, 3), dtype=np.uint8)
    result = decode_pose(frame)
    assert result.status is DecodeStatus.MAGIC_MISMATCH
    assert result.pose is None


def test_checksum_mismatch_detected():
    words = pack_pose_words(1, (1.0, 0, 0), (0, 0, 1.0), (0, 1.0, 0))
    words[3] ^= np.uint32(1)  # 位置ビットを1つ壊す(チェックサムはそのまま)
    frame = render_grid(words)
    result = decode_pose(frame)
    assert result.status is DecodeStatus.CHECKSUM_MISMATCH


def test_grid_offset_within_larger_canvas():
    # クライアント左上原点に描いたグリッドを、より大きなキャンバスでも読める
    words = pack_pose_words(7, (3.0, 2.0, 1.0), (0, 0, 1.0), (0, 1.0, 0))
    frame = render_grid(words, canvas_shape=(80, 160))
    assert decode_pose(frame).status is DecodeStatus.OK


def test_alpha_channel_frame():
    # mss は BGRA を返す。4ch でも先頭3chで読めること。
    frame3 = render_pose(5, (1.0, 1.0, 1.0), (0, 0, 1.0), (0, 1.0, 0))
    alpha = np.full((*frame3.shape[:2], 1), 255, dtype=np.uint8)
    frame4 = np.concatenate([frame3, alpha], axis=2)
    assert decode_pose(frame4).status is DecodeStatus.OK


def test_no_python_pixel_loop_is_fast():
    # ベクトル化のスモークテスト: 1000回デコードしても十分速い
    frame = render_pose(0, (1.0, 2.0, 3.0), (0, 0, 1.0), (0, 1.0, 0))
    import time

    t0 = time.perf_counter()
    for _ in range(1000):
        decode_pose(frame)
    dt = time.perf_counter() - t0
    assert dt < 2.0, f"1000 decodes took {dt:.2f}s (too slow?)"


# ---- PoseReader 統合(合成ソース) --------------------------------------
def test_pose_reader_with_array_source():
    frame = render_pose(100, (1.0, 2.0, 3.0), (0, 0, 1.0), (0, 1.0, 0))
    reader = PoseReader(source=ArrayFrameSource(frame))

    # スレッドを使わず単体で1フレーム処理
    result = reader.process_frame(frame)
    assert result.ok
    assert reader.get_latest().position == (
        np.float32(1.0),
        np.float32(2.0),
        np.float32(3.0),
    )
    assert reader.get_stats().new_frames == 1

    # 同一 time_ms は重複としてスキップ
    reader.process_frame(frame)
    stats = reader.get_stats()
    assert stats.duplicate_skipped == 1
    assert stats.new_frames == 1


def test_pose_reader_consecutive_fail():
    blank = np.zeros((CAPTURE_H, CAPTURE_W, 3), dtype=np.uint8)
    reader = PoseReader(source=ArrayFrameSource(blank))
    for _ in range(5):
        reader.process_frame(blank)
    stats = reader.get_stats()
    assert stats.consecutive_fail == 5
    assert stats.decode_fail == 5


def test_pose_reader_new_frame_resets_fail():
    good = render_pose(1, (0, 0, 0), (0, 0, 1.0), (0, 1.0, 0))
    blank = np.zeros((CAPTURE_H, CAPTURE_W, 3), dtype=np.uint8)
    reader = PoseReader(source=ArrayFrameSource(blank))
    reader.process_frame(blank)
    reader.process_frame(blank)
    assert reader.get_stats().consecutive_fail == 2
    reader.process_frame(good)
    assert reader.get_stats().consecutive_fail == 0


def test_pose_reader_generator():
    frame = render_pose(50, (5.0, 0, 0), (0, 0, 1.0), (0, 1.0, 0))
    reader = PoseReader(source=ArrayFrameSource(frame))
    reader.process_frame(frame)
    gen = reader.poses(timeout=0.1)
    assert next(gen).time_ms == 50
