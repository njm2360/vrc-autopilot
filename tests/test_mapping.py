"""RoomMapper の合成軌跡テスト(VRChat不要 / CI可能)。"""

from __future__ import annotations

import numpy as np
import pytest

from pose_hud.pose import Pose
from pose_hud.mapping import Bounds, RoomMapper


def rectangle_path(width=4.0, depth=6.0, step=0.1, x0=1.0, z0=-2.0):
    """(x0,z0) を角とする width x depth の矩形を反時計回りに一周する XZ 点列。"""
    corners = [
        (x0, z0),
        (x0 + width, z0),
        (x0 + width, z0 + depth),
        (x0, z0 + depth),
        (x0, z0),
    ]
    pts = []
    for (ax, az), (bx, bz) in zip(corners, corners[1:]):
        seg = np.hypot(bx - ax, bz - az)
        n = max(2, int(seg / step))
        for i in range(n):
            f = i / n
            pts.append((ax + (bx - ax) * f, az + (bz - az) * f))
    pts.append(corners[-1])
    return pts


def make_pose(x, z, y=1.6, t=0):
    return Pose(time_ms=t, position=(x, y, z), forward=(0, 0, 1.0), up=(0, 1.0, 0))


def _line(m, a, b, step=0.05):
    n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step))
    for i in range(n + 1):
        f = i / n
        m.add(a[0] + (b[0] - a[0]) * f, 1.6, a[1] + (b[1] - a[1]) * f)


# ---- セグメント分割(キャプチャ一時停止) --------------------------------
def test_break_segment_prevents_connecting_line():
    # 2本の離れた壁区間を、間に break を挟んで記録する
    m = RoomMapper(min_move=0.0)
    _line(m, (0.0, 0.0), (1.0, 0.0))     # 区間1
    m.break_segment()
    _line(m, (0.0, 5.0), (1.0, 5.0))     # 区間2(z=5 に離れている)
    assert m.num_segments == 2

    occ = m.occupancy_grid(cell=0.1, pad=0.5)
    # 両区間は占有されている
    assert occ.grid[occ.world_to_index(0.5, 0.0)]
    assert occ.grid[occ.world_to_index(0.5, 5.0)]
    # 間(z=2.5 付近)は繋がっていない=偽の壁が無い
    assert not occ.grid[occ.world_to_index(0.5, 2.5)]


def test_without_break_line_is_connected():
    # break を挟まないと2区間が1本の線で繋がる(対照)
    m = RoomMapper(min_move=0.0)
    _line(m, (0.0, 0.0), (1.0, 0.0))
    _line(m, (0.0, 5.0), (1.0, 5.0))     # break なし
    assert m.num_segments == 1
    occ = m.occupancy_grid(cell=0.1, pad=0.5)
    assert occ.grid[occ.world_to_index(0.5, 2.5)]   # 繋ぎ線が通る


def test_break_segment_excludes_gap_from_path_length():
    m = RoomMapper(min_move=0.0)
    _line(m, (0.0, 0.0), (1.0, 0.0))     # 長さ1
    m.break_segment()
    _line(m, (0.0, 5.0), (1.0, 5.0))     # 長さ1(gap ~5 は数えない)
    assert m.path_length() == pytest.approx(2.0, abs=0.05)


def test_break_on_empty_is_noop():
    m = RoomMapper(min_move=0.0)
    m.break_segment()                     # 点が無い状態でも安全
    m.add(1.0, 1.6, 2.0)
    assert m.num_segments == 1
    assert len(m) == 1


def test_double_break_does_not_skip_segment_ids():
    m = RoomMapper(min_move=0.0)
    m.add(0, 1.6, 0)
    m.break_segment()
    m.break_segment()                     # 連続break は1回分
    m.add(1, 1.6, 1)
    assert m.num_segments == 2


def test_segments_survive_save_load(tmp_path):
    m = RoomMapper(min_move=0.0)
    _line(m, (0.0, 0.0), (1.0, 0.0))
    m.break_segment()
    _line(m, (0.0, 5.0), (1.0, 5.0))
    m.save(tmp_path / "room")

    loaded = RoomMapper.load(tmp_path / "room")
    assert loaded.num_segments == 2
    assert len(loaded.segment_points()) == 2
    assert loaded.path_length() == pytest.approx(m.path_length())
    # 分割が保たれ、gap は繋がらない
    occ = loaded.occupancy_grid(cell=0.1, pad=0.5)
    assert not occ.grid[occ.world_to_index(0.5, 2.5)]


def test_dimensions_recovered_from_rectangle():
    m = RoomMapper(min_move=0.0)
    for i, (x, z) in enumerate(rectangle_path(4.0, 6.0)):
        m.add(x, 1.6, z, t=i)
    w, d = m.dimensions()
    assert w == pytest.approx(4.0, abs=0.05)
    assert d == pytest.approx(6.0, abs=0.05)


def test_bounds_and_area():
    m = RoomMapper.from_poses(
        (make_pose(x, z, t=i) for i, (x, z) in enumerate(rectangle_path(4.0, 6.0))),
        min_move=0.0,
    )
    b = m.bounds()
    assert isinstance(b, Bounds)
    assert b.xmin == pytest.approx(1.0, abs=0.05)
    assert b.zmax == pytest.approx(4.0, abs=0.05)  # z0=-2 + depth 6
    assert m.to_dict()["floor_area_bbox_m2"] == pytest.approx(24.0, rel=0.05)


def test_path_length_is_perimeter():
    m = RoomMapper(min_move=0.0)
    for x, z in rectangle_path(4.0, 6.0):
        m.add(x, 1.6, z)
    # 周長 2*(4+6)=20m
    assert m.path_length() == pytest.approx(20.0, abs=0.2)


def test_min_move_decimation():
    m = RoomMapper(min_move=0.05)
    m.add(0.0, 1.6, 0.0)
    added = m.add(0.01, 1.6, 0.0)   # 1cm 移動 -> 間引かれる
    assert added is False
    assert len(m) == 1
    assert m.add(0.1, 1.6, 0.0) is True  # 10cm -> 採用
    assert len(m) == 2


def test_height_range():
    m = RoomMapper(min_move=0.0)
    m.add(0, 1.5, 0)
    m.add(1, 1.9, 1)
    lo, hi = m.height_range()
    assert lo == pytest.approx(1.5)
    assert hi == pytest.approx(1.9)


def test_occupancy_grid_traces_path():
    m = RoomMapper(min_move=0.0)
    for x, z in rectangle_path(4.0, 6.0, step=0.05):
        m.add(x, 1.6, z)
    occ = m.occupancy_grid(cell=0.1, pad=0.5)
    assert occ.grid.any()
    # 経路は輪郭のみなので、内部が全部埋まることはない(掃過面積 < bbox面積)
    assert occ.visited_area < m.to_dict()["floor_area_bbox_m2"]
    # グリッド範囲が pad ぶん広い
    assert occ.bounds.width == pytest.approx(4.0 + 1.0, abs=0.15)


def test_empty_mapper():
    m = RoomMapper()
    assert len(m) == 0
    assert m.bounds() is None
    assert m.dimensions() == (0.0, 0.0)
    assert m.path_length() == 0.0
    with pytest.raises(ValueError):
        m.occupancy_grid()


def test_save_load_roundtrip(tmp_path):
    m = RoomMapper(min_move=0.0)
    for i, (x, z) in enumerate(rectangle_path(4.0, 6.0)):
        m.add(x, 1.6 + 0.001 * i, z, yaw=float(i % 360), t=i)
    npz = m.save(tmp_path / "room")
    assert npz.exists()
    assert (tmp_path / "room.json").exists()

    loaded = RoomMapper.load(tmp_path / "room")
    assert len(loaded) == len(m)
    np.testing.assert_allclose(loaded.xyz, m.xyz)
    np.testing.assert_allclose(loaded.yaw, m.yaw)
    assert loaded.dimensions() == pytest.approx(m.dimensions())


def test_single_point_grid():
    m = RoomMapper(min_move=0.0)
    m.add(1.0, 1.6, 2.0)
    occ = m.occupancy_grid(cell=0.1)
    assert occ.grid.sum() == 1


# ---- レンダラ(matplotlib) ----------------------------------------------
def test_render_map_png(tmp_path):
    pytest.importorskip("matplotlib")
    from pose_hud.mapping_render import render_map

    m = RoomMapper(min_move=0.0)
    for x, z in rectangle_path(4.0, 6.0):
        m.add(x, 1.6, z)
    out = render_map(m, tmp_path / "map", cell=0.1)
    assert out.exists() and out.stat().st_size > 0
