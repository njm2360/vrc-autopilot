"""オフラインの経路計画: 歩行可能グリッド生成(壁回避)+ A* + 経由点の直線化。

フレーム単位の照準幾何(誤差計算)は guidance.py。
"""

import heapq
import logging
import math
from collections import deque
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    generate_binary_structure,
    label,
)

from ..mapping.mapper import Bounds, RoomMapper

log = logging.getLogger(__name__)


def _dilate(mask: np.ndarray, iters: int, connectivity: int = 8) -> np.ndarray:
    """二値マスクを iters 回膨張(ラップなし)。connectivity=4 or 8。"""
    if iters <= 0:
        return mask.copy()
    struct = generate_binary_structure(2, 1 if connectivity == 4 else 2)
    return binary_dilation(mask, structure=struct, iterations=iters)


def _flood_from_border(passable: np.ndarray) -> np.ndarray:
    """グリッド外周から passable セルを通って到達できる領域(=外部)を返す(4連結)。"""
    lbl, _ = label(passable)  # 4連結の連結成分ラベリング
    border = (
        set(lbl[0]) | set(lbl[-1]) | set(lbl[:, 0]) | set(lbl[:, -1])
    )  # 外周に触れるラベル
    border.discard(0)  # 0 は非 passable(背景)
    return np.isin(lbl, list(border))


@dataclass
class NavGrid:
    """歩行可能セルのグリッド(True=歩ける)。行=Z, 列=X。"""

    free: np.ndarray
    cell: float
    bounds: Bounds
    solid: np.ndarray | None = None  # 固体領域(壁軌跡+外部+内壁の内側)。描画用

    @property
    def shape(self) -> tuple[int, int]:
        return self.free.shape

    def world_to_cell(self, x: float, z: float) -> tuple[int, int]:
        col = int((x - self.bounds.xmin) / self.cell)
        row = int((z - self.bounds.zmin) / self.cell)
        rows, cols = self.free.shape
        return (min(max(row, 0), rows - 1), min(max(col, 0), cols - 1))

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = self.bounds.xmin + (col + 0.5) * self.cell
        z = self.bounds.zmin + (row + 0.5) * self.cell
        return (x, z)

    def is_free(self, row: int, col: int) -> bool:
        rows, cols = self.free.shape
        return 0 <= row < rows and 0 <= col < cols and bool(self.free[row, col])

    def nearest_free(
        self, row: int, col: int, within: np.ndarray | None = None
    ) -> tuple[int, int] | None:
        """(row,col) から最も近い歩けるセルを BFS で探す。

        within(bool マスク)を渡すと、その中の歩けるセルだけを解とする(例: スタート
        と同じ連結成分に限定して、壁の裏側の床への吸着を防ぐ)。マスク外のセルは解に
        ならないが、BFS の波は通り抜ける。
        """

        def ok(r: int, c: int) -> bool:
            return bool(self.free[r, c]) and (within is None or bool(within[r, c]))

        if self.is_free(row, col) and ok(row, col):
            return (row, col)
        rows, cols = self.free.shape
        seen = np.zeros_like(self.free, dtype=bool)
        dq = deque([(row, col)])
        seen[row, col] = True
        while dq:
            r, c = dq.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc]:
                    seen[nr, nc] = True
                    if ok(nr, nc):
                        return (nr, nc)
                    dq.append((nr, nc))
        return None

    @classmethod
    def from_mapper(
        cls,
        mapper: RoomMapper,
        cell: float = 0.1,
        avatar_radius: float = 0.25,
        gap_close: float = 0.6,
    ) -> NavGrid:
        """歩行軌跡から歩行可能グリッドを構築する。

        軌跡は壁をなぞった跡で、外周だけでなく内壁・柱などの浮いた壁も障害物に含める。
        開けた床やドア越しの移動は記録時にペンアップして除外する運用が前提。
        幅 gap_close [m] 以下の記録の隙間は塞いで扱う(手順は本文コメント参照)。

        外周の軌跡に gap_close より大きい隙間が残ると外側が室内へ流れ込み、歩行可能
        セルがゼロになる(警告ログを出す)。gap_close を上げるかマップを取り直すこと。
        """
        pad = max(0.5, avatar_radius + gap_close / 2 + cell)
        occ = mapper.occupancy_grid(cell=cell, pad=pad)
        walked = occ.grid

        gap_cells = max(0, math.ceil(gap_close / (2 * cell)))
        struct = generate_binary_structure(2, 2)
        sealed = _dilate(walked, gap_cells, connectivity=8)
        exterior = _flood_from_border(~sealed)
        if gap_cells > 0:
            # flood は膨張カラーぶん壁の手前で止まる。壁(walked)を跨がない
            # 測地膨張で壁面まで戻し広げる(外側の帯を外部に正しく分類する)
            exterior = binary_dilation(
                exterior, structure=struct, iterations=gap_cells, mask=~walked
            )

        # 内壁(inner)が閉ループで囲む領域は固体(柱・間仕切りの中身)。記録の
        # 閉じ残しを gap_close まで塞いでから穴埋めし、膨張ぶんを戻す。
        # 閉じていない inner 線(単独の仕切り)は fill_holes で変化しない。
        inner = mapper.occupancy_grid(cell=cell, pad=pad, kind="inner").grid
        pockets = binary_fill_holes(_dilate(inner, gap_cells, connectivity=8))
        if gap_cells > 0:
            pockets = binary_erosion(pockets, structure=struct, iterations=gap_cells)

        # 固体 = 外側 + 壁(=軌跡そのもの) + 内壁の内側。walked を含めることで
        # 内壁・柱などの浮いた壁も障害物になる。そこから avatar_radius 未満に
        # 近づくセルを塞いだ残りが歩行可能な床。+cell/2 はラスタ化誤差の安全余裕
        solid = exterior | walked | pockets
        dist_m = distance_transform_edt(~solid) * cell
        free = dist_m >= avatar_radius + 0.5 * cell

        # 主床(最大の free を含む連結成分)以外の空間は、二重壁の隙間や封鎖
        # ポケット(歩いて入れない=地図化できない領域)なので固体に畳む
        lbl, n = label(~solid)
        if n > 1 and free.any():
            counts = np.bincount(lbl[free], minlength=n + 1)
            main = int(counts[1:].argmax()) + 1
            solid |= (lbl != 0) & (lbl != main)
            free &= lbl == main

        if not free.any():
            log.warning(
                "walkable grid is empty: the wall trace likely has a gap wider "
                "than gap_close=%.2fm and the exterior flooded the interior. "
                "Increase gap_close or re-record the map.",
                gap_close,
            )
        return cls(free=free, cell=cell, bounds=occ.bounds, solid=solid)


def _astar(
    free: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    cost_mult: np.ndarray | None = None,
):
    """8連結 A*。角抜け(壁の対角すり抜け)を禁止。セル列を返す(無ければ None)。

    cost_mult (>=1) を渡すと、そのセルへ入る移動コストが距離×cost_mult になる。
    壁際セルに大きい値を与えることで「通れるが避けたい」ソフトコストを表現できる
    (硬い障害物にはしないので、狭い通路の到達性は落ちない)。
    ペナルティは非負なのでユークリッド距離ヒューリスティックは許容的なまま。
    """
    rows, cols = free.shape
    if not free[start] or not free[goal]:
        return None
    if start == goal:
        return [start]

    gr, gc = goal
    open_heap = [(math.hypot(start[0] - gr, start[1] - gc), 0.0, start)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    gscore = {start: 0.0}
    SQRT2 = math.sqrt(2.0)
    neighbors = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, SQRT2),
        (1, -1, SQRT2),
        (-1, 1, SQRT2),
        (-1, -1, SQRT2),
    ]

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if g > gscore.get(cur, math.inf):  # より良い経路で再登録済み(stale)
            continue
        r, c = cur
        for dr, dc, cost in neighbors:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or not free[nr, nc]:
                continue
            if dr != 0 and dc != 0:  # 角抜け防止: 対角は両隣が空いている時のみ
                if not free[r + dr, c] or not free[r, c + dc]:
                    continue
            if cost_mult is not None:
                cost = cost * cost_mult[nr, nc]
            ng = g + cost
            if ng < gscore.get((nr, nc), math.inf):
                gscore[(nr, nc)] = ng
                came[(nr, nc)] = cur
                h = math.hypot(nr - gr, nc - gc)
                heapq.heappush(open_heap, (ng + h, ng, (nr, nc)))
    return None


def _segment_min_clearance(
    free: np.ndarray,
    clearance: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
) -> float:
    """セル a→b の線分を密サンプルし、通過セルの最小クリアランス[セル]を返す。

    塞がりセルに触れる、または壁の対角すり抜け(A*と同じ角抜け条件)がある場合は -1。
    サンプル間隔は 0.25 セルなので、経由点間の線分が角を掠めるケースも拾う
    """
    (r0, c0), (r1, c1) = a, b
    dist = math.hypot(r1 - r0, c1 - c0)
    steps = max(1, int(dist * 4))
    min_clear = math.inf
    pr, pc = r0, c0
    for i in range(steps + 1):
        t = i / steps
        r = int(round(r0 + (r1 - r0) * t))
        c = int(round(c0 + (c1 - c0) * t))
        if not free[r, c]:
            return -1.0
        if r != pr and c != pc:  # 対角のセル遷移: 両隣が空いていなければ角抜け
            if not free[pr, c] or not free[r, pc]:
                return -1.0
        cl = float(clearance[r, c])
        if cl < min_clear:
            min_clear = cl
        pr, pc = r, c
    return min_clear


def _los_simplify(
    free: np.ndarray,
    cells: list[tuple[int, int]],
    clearance: np.ndarray,
    margin_cells: float,
) -> list[tuple[int, int]]:
    """壁に遮られず、クリアランスを損なわない直線で経路を間引く。

    A* のジグザグ(対角の階段状)を直線で結び直して経由点を最小化する。ただし、
    ショートカットの直線が「元の A* 区間の最小クリアランス(margin_cells で頭打ち)」
    を下回って壁・角に寄る場合は採用しない。A* が角を大きく回った意図を直線化で
    打ち消さないため。狭い通路では元区間のクリアランス自体が小さいので、同等の
    直線は許容される(到達性は落ちない)。
    """
    if len(cells) <= 2:
        return cells
    out = [cells[0]]
    anchor = 0  # 現在の直線区間の起点
    sub_min = float(clearance[cells[0]])  # anchor..i を結ぶ A* 区間の最小クリアランス
    i = 1
    while i < len(cells) - 1:
        # 起点から次の点まで見通せる間は伸ばし、見通せなくなる直前で確定する
        cand_min = min(
            sub_min, float(clearance[cells[i]]), float(clearance[cells[i + 1]])
        )
        need = min(margin_cells, cand_min)
        seg_min = _segment_min_clearance(free, clearance, cells[anchor], cells[i + 1])
        if seg_min + 1e-9 < need:  # 壁越し(-1)またはクリアランス悪化
            out.append(cells[i])
            anchor = i
            sub_min = float(clearance[cells[i]])
        else:
            sub_min = cand_min
        i += 1
    out.append(cells[-1])
    return out


@dataclass
class Path:
    """計画された経路。"""

    waypoints: list[tuple[float, float]]  # XZ [m] の経由点列(start→goal付近)
    length: float  # 総距離 [m]
    snapped_goal_xz: tuple[float, float]  # 実際に到達するゴール寄りセルのXZ
    goal_blocked: bool  # 目標が壁または到達不能な孤立領域で、最寄りの床に迂回したか


def plan_path(
    grid: NavGrid,
    start_xz: tuple[float, float],
    goal_xz: tuple[float, float],
    *,
    margin: float = 0.3,
    margin_weight: float = 6.0,
    max_goal_divert: float = 1.0,
) -> Path | None:
    """start から goal まで壁を避けた経路を計画する。到達不能なら None。

    goal が歩けないセル(壁面のボタン等)や、スタートから到達できない孤立領域の
    セル(柱の内側など)なら、スタートと同じ連結成分内の最寄りの歩けるセルへ吸着して
    goal_blocked を立てる。ただし吸着先が goal から max_goal_divert [m] より離れる
    場合は本当に到達できない目標(仕切られた隣室など)なので None を返す。
    start / goal がマップ範囲外なら ValueError(グリッド端への暗黙のクランプはしない)。

    壁際はソフトコストで避ける: 塞がりセルからの距離が margin [m] 未満のセルは
    移動コストが最大 (1 + margin_weight) 倍になる(margin=0 で無効)。障害物を
    増やすわけではないので、margin より狭い通路も遠回りが無ければ通る。
    """
    b = grid.bounds
    for name, (x, z) in (("start", start_xz), ("goal", goal_xz)):
        if not (b.xmin <= x <= b.xmax and b.zmin <= z <= b.zmax):
            raise ValueError(
                f"{name} ({x:.2f}, {z:.2f}) is outside the map bounds "
                f"x[{b.xmin:.2f}, {b.xmax:.2f}] z[{b.zmin:.2f}, {b.zmax:.2f}]"
            )
    sc = grid.world_to_cell(*start_xz)
    gc = grid.world_to_cell(*goal_xz)
    if not grid.is_free(*sc):
        nf = grid.nearest_free(*sc)
        if nf is None:
            return None
        sc = nf

    # ゴールはスタートと同じ連結成分(4連結。角抜け禁止A*の到達性と一致)に解決する。
    # 壁面ボタンのゴールが壁の裏側(隣室・柱の内側)の床へ吸着して「到達不能」と
    # 誤判定されるのを防ぐ。
    comp, _ = label(grid.free)
    start_comp = comp[sc]
    goal_blocked = not grid.is_free(*gc) or comp[gc] != start_comp
    if goal_blocked:
        nf = grid.nearest_free(*gc, within=comp == start_comp)
        if nf is None:
            return None
        nx, nz = grid.cell_to_world(*nf)
        if math.hypot(nx - goal_xz[0], nz - goal_xz[1]) > max_goal_divert:
            return None  # 壁際のずれではなく、到達できない領域の目標
        gc = nf

    # 各 free セルから最寄りの塞がりセルまでの距離[セル](塞がりセルは 0)
    clearance = distance_transform_edt(grid.free)
    margin_cells = max(0.0, margin / grid.cell)
    cost_mult = None
    if margin_cells > 0.0 and margin_weight > 0.0:
        pen = np.clip(1.0 - clearance / margin_cells, 0.0, 1.0) ** 2
        cost_mult = 1.0 + margin_weight * pen

    cells = _astar(grid.free, sc, gc, cost_mult)
    if cells is None:
        return None
    # 見通しで直線化(ジグザグ除去)。クリアランスを損なう直線化はしない
    cells = _los_simplify(grid.free, cells, clearance, margin_cells)
    waypoints = [grid.cell_to_world(r, c) for (r, c) in cells]

    length = 0.0
    for a, b in pairwise(waypoints):
        length += math.hypot(b[0] - a[0], b[1] - a[1])
    return Path(
        waypoints=waypoints,
        length=length,
        snapped_goal_xz=grid.cell_to_world(*gc),
        goal_blocked=goal_blocked,
    )
