"""ナビゲーション(歩行可能グリッド生成・A*壁回避・操舵)のテスト。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pose_hud.mapping import RoomMapper
from pose_hud.navigation import NavGrid, plan_path


def _trace(corners, step=0.05):
    """角のリストを順に結ぶ密な軌跡点(閉ループにするため末尾に先頭を足す)。"""
    pts = []
    loop = list(corners) + [corners[0]]
    for (ax, az), (bx, bz) in zip(loop, loop[1:]):
        seg = math.hypot(bx - ax, bz - az)
        n = max(2, int(seg / step))
        for i in range(n):
            f = i / n
            pts.append((ax + (bx - ax) * f, az + (bz - az) * f))
    return pts


def rectangle_mapper(w=6.0, d=5.0):
    corners = [(0, 0), (w, 0), (w, d), (0, d)]
    m = RoomMapper(min_move=0.0)
    for x, z in _trace(corners):
        m.add(x, 1.6, z)
    return m


def l_shaped_mapper():
    # L字: 下アーム x[0,6] z[0,2] + 左アーム x[0,2] z[2,6]。凹角が (2,2)。
    corners = [(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)]
    m = RoomMapper(min_move=0.0)
    for x, z in _trace(corners):
        m.add(x, 1.6, z)
    return m


def test_rectangle_interior_free_exterior_blocked():
    grid = NavGrid.from_mapper(rectangle_mapper(6.0, 5.0), cell=0.1, avatar_radius=0.2)
    # 部屋の中央は歩ける
    r, c = grid.world_to_cell(3.0, 2.5)
    assert grid.is_free(r, c)
    # 壁の外は歩けない
    r, c = grid.world_to_cell(-1.0, 2.5)
    assert not grid.is_free(r, c)


def test_open_room_path_is_near_straight():
    grid = NavGrid.from_mapper(rectangle_mapper(6.0, 5.0), cell=0.1, avatar_radius=0.2)
    path = plan_path(grid, (1.0, 1.0), (5.0, 4.0))
    assert path is not None
    straight = math.hypot(5.0 - 1.0, 4.0 - 1.0)
    # 障害物なしなのでほぼ直線(グリッド/対角の誤差ぶんの余裕)
    assert path.length == pytest.approx(straight, rel=0.15)


def test_l_shape_forces_detour():
    grid = NavGrid.from_mapper(l_shaped_mapper(), cell=0.1, avatar_radius=0.2, gap_close=0.3)
    start = (5.0, 1.0)   # 下アームの右端
    goal = (1.0, 5.0)    # 左アームの上端
    path = plan_path(grid, start, goal)
    assert path is not None
    straight = math.hypot(goal[0] - start[0], goal[1] - start[1])  # ≈5.66
    # 凹角を回り込むので直線よりかなり長い
    assert path.length > straight * 1.1
    # 全経由点が歩けるセル(=壁の外/切り欠きを通らない)
    for x, z in path.waypoints:
        r, c = grid.world_to_cell(x, z)
        assert grid.is_free(r, c)


def test_l_shape_notch_is_blocked():
    grid = NavGrid.from_mapper(l_shaped_mapper(), cell=0.1, avatar_radius=0.2, gap_close=0.3)
    # 切り欠き(L外)の点は歩けない
    r, c = grid.world_to_cell(4.0, 4.0)
    assert not grid.is_free(r, c)


def _line(m, a, b, step=0.05):
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
    for i in range(n + 1):
        f = i / n
        m.add(a[0] + (b[0] - a[0]) * f, 1.6, a[1] + (b[1] - a[1]) * f)


def partitioned_room_mapper():
    # 10x6 の部屋 + 内壁(間仕切り): x=5 を z=0→4 まで(上に z4..6 の通路)
    m = RoomMapper(min_move=0.0)
    for a, b in [((0, 0), (10, 0)), ((10, 0), (10, 6)), ((10, 6), (0, 6)), ((0, 6), (0, 0))]:
        _line(m, a, b)
    m.break_segment()
    _line(m, (5, 0), (5, 4))       # 浮いた内壁(下側は壁、上側 z>4 は通路)
    return m


def test_interior_wall_is_blocked():
    grid = NavGrid.from_mapper(partitioned_room_mapper(), cell=0.1, avatar_radius=0.2)
    # 内壁の上(=障害物)は歩けない
    assert not grid.is_free(*grid.world_to_cell(5.0, 2.0))
    # 内壁のない開けた床は歩ける
    assert grid.is_free(*grid.world_to_cell(2.0, 3.0))
    assert grid.is_free(*grid.world_to_cell(8.0, 3.0))


def test_path_detours_around_interior_wall():
    grid = NavGrid.from_mapper(partitioned_room_mapper(), cell=0.1, avatar_radius=0.2)
    # 内壁で隔てられた左右を結ぶには、上の通路(z>4)へ迂回するしかない
    path = plan_path(grid, (2.0, 3.0), (8.0, 3.0))
    assert path is not None
    assert path.length > 6.0                    # 直進(6.0m)より長い=迂回している
    # 内壁(x=5, z<4)を越えるため、上の通路(z>4)まで回り込む経由点がある
    assert max(z for _, z in path.waypoints) > 4.0
    # 経由点自体は内壁 x=5, z<4 を踏まない
    for x, z in path.waypoints:
        if z < 3.8:
            assert abs(x - 5.0) > 0.35


def test_los_smoothing_reduces_waypoints_in_open_room():
    # 障害物のない部屋では A* のジグザグが直線1本に畳まれ、経由点が少ない
    grid = NavGrid.from_mapper(rectangle_mapper(8.0, 6.0), cell=0.1, avatar_radius=0.2)
    path = plan_path(grid, (1.0, 1.0), (7.0, 5.0))     # 斜め
    assert path is not None
    assert len(path.waypoints) <= 3                    # start と goal 近辺のみ
    # ほぼ直線(グリッド誤差の範囲)
    straight = math.hypot(6.0, 4.0)
    assert path.length == pytest.approx(straight, rel=0.1)


def test_los_smoothing_keeps_detour_around_wall():
    # 内壁があると見通しが切れるので、迂回のための経由点は残る
    grid = NavGrid.from_mapper(partitioned_room_mapper(), cell=0.1, avatar_radius=0.2)
    path = plan_path(grid, (2.0, 3.0), (8.0, 3.0))
    assert path is not None
    assert len(path.waypoints) >= 3                    # 直線1本にはならない


def test_goal_on_wall_routes_to_nearest_free():
    grid = NavGrid.from_mapper(rectangle_mapper(6.0, 5.0), cell=0.1, avatar_radius=0.2)
    # 壁の外にあるボタン → 最寄りの床へ迂回、goal_blocked フラグが立つ
    path = plan_path(grid, (3.0, 2.5), (6.5, 2.5))
    assert path is not None
    assert path.goal_blocked
    r, c = grid.world_to_cell(*path.reached_goal_cell)
    assert grid.is_free(r, c)


def test_unreachable_returns_none():
    grid = NavGrid.from_mapper(rectangle_mapper(6.0, 5.0), cell=0.1, avatar_radius=0.2)
    # 完全に部屋の外どうし(どちらも最寄り床には解決するが、別部屋想定の遠方)
    # ここでは開始を壁の中に置いても nearest_free で解決するので、
    # 代わりに free を全消しして到達不能を作る
    grid.free[:] = False
    assert plan_path(grid, (1.0, 1.0), (2.0, 2.0)) is None


# ---- クリアランス(線分密サンプル採点) ----------------------------------
def _sample_segments(waypoints, step=0.02):
    """経由点間の線分を step [m] 刻みで密サンプルした点列。

    経由点だけの採点では線分が角を掠めるのを見逃すため、必ず線分全体を評価する。
    """
    out = []
    for a, b in zip(waypoints, waypoints[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(d / step))
        for i in range(n):
            f = i / n
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    out.append(tuple(waypoints[-1]))
    return np.asarray(out)


def _min_wall_clearance(waypoints, wall_corners):
    """経路(線分密サンプル)から実際の壁軌跡までの最小距離 [m]。"""
    samples = _sample_segments(waypoints)
    walls = np.asarray(_trace(wall_corners, step=0.01))
    d = np.sqrt(((samples[:, None, :] - walls[None, :, :]) ** 2).sum(-1))
    return float(d.min(axis=1).min())


def _assert_segments_all_free(grid, waypoints):
    for x, z in _sample_segments(waypoints):
        r, c = grid.world_to_cell(x, z)
        assert grid.is_free(r, c), f"segment enters blocked cell at ({x:.2f}, {z:.2f})"


L_CORNERS = [(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)]


def test_path_keeps_margin_from_concave_corner():
    # L字の凹角 (2,2) を回り込む経路が、壁ギリギリ(radius+ラスタ誤差)を攻めない。
    # avatar_radius=0.2 に対し、線分密サンプルの最小クリアランスで radius+0.2 を要求。
    grid = NavGrid.from_mapper(l_shaped_mapper(), cell=0.1, avatar_radius=0.2, gap_close=0.3)
    path = plan_path(grid, (5.0, 1.0), (1.0, 5.0))
    assert path is not None
    assert _min_wall_clearance(path.waypoints, L_CORNERS) >= 0.4
    _assert_segments_all_free(grid, path.waypoints)


def test_los_segments_never_cross_blocked_cells():
    # 間仕切り部屋: 直線化後の各線分も(経由点だけでなく)全長にわたり free を通る
    grid = NavGrid.from_mapper(partitioned_room_mapper(), cell=0.1, avatar_radius=0.2)
    path = plan_path(grid, (2.0, 3.0), (8.0, 3.0))
    assert path is not None
    _assert_segments_all_free(grid, path.waypoints)


DUMBBELL_CORNERS = [
    (0, 0), (4, 0), (4, 1.7), (6, 1.7), (6, 0), (10, 0),
    (10, 4), (6, 4), (6, 2.4), (4, 2.4), (4, 4), (0, 4),
]


def dumbbell_mapper():
    # 2部屋を幅 0.7m の狭い通路(x∈[4,6], z∈[1.7,2.4])でつなぐ
    m = RoomMapper(min_move=0.0)
    for x, z in _trace(DUMBBELL_CORNERS):
        m.add(x, 1.6, z)
    return m


def test_narrow_corridor_stays_reachable_with_margin():
    # margin(壁際ソフトコスト)は狭通路を塞がない: radius=0.2 + 通路0.7m でも通れる
    grid = NavGrid.from_mapper(dumbbell_mapper(), cell=0.1, avatar_radius=0.2, gap_close=0.3)
    path = plan_path(grid, (2.0, 2.0), (8.0, 2.0))
    assert path is not None
    _assert_segments_all_free(grid, path.waypoints)
    # 通路内はほぼ中央を通る(半幅0.35に対して0.25以上のクリアランス)
    assert _min_wall_clearance(path.waypoints, DUMBBELL_CORNERS) >= 0.25


def test_open_room_margin_does_not_bend_far_path():
    # 壁から十分離れた直線経路は margin の影響を受けず直線のまま
    grid = NavGrid.from_mapper(rectangle_mapper(6.0, 5.0), cell=0.1, avatar_radius=0.2)
    path = plan_path(grid, (1.0, 1.0), (5.0, 4.0))
    assert path is not None
    straight = math.hypot(4.0, 3.0)
    assert path.length == pytest.approx(straight, rel=0.15)
