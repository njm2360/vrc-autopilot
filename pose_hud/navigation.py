import heapq
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
    label,
)

from .mapping import Bounds, RoomMapper


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

    def nearest_free(self, row: int, col: int) -> tuple[int, int] | None:
        """(row,col) から最も近い歩けるセルを BFS で探す。"""
        if self.is_free(row, col):
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
                    if self.free[nr, nc]:
                        return (nr, nc)
                    dq.append((nr, nc))
        return None

    @classmethod
    def from_mapper(
        cls,
        mapper: RoomMapper,
        cell: float = 0.1,
        avatar_radius: float = 0.25,
        gap_close: float = 0.3,
    ) -> "NavGrid":
        """歩行軌跡から歩行可能グリッドを構築する。

        軌跡は「壁をなぞった跡」。外周だけでなく**内壁・間仕切り・柱などの浮いた壁**も
        歩いた跡として障害物に含める(部屋を横切って壁を貫通する経路を防ぐ)。開けた床や
        ドア越しの移動は記録時に SPACE で一時停止して除外する運用が前提。

        1. 軌跡をグリッドに描き込む(=壁)。
        2. gap_close ぶん膨張して軌跡ループの隙間を塞ぎ、外周から流し込んで**外側**を判定。
        3. 外側と**すべての壁(軌跡)**を avatar_radius ぶん膨張させて塞ぐ(クリアランス確保)。
        4. 残りが歩行可能な床。内壁で仕切られた領域は互いに分断される(実際の壁どおり)。
        """
        pad = max(0.5, avatar_radius + gap_close + cell)
        occ = mapper.occupancy_grid(cell=cell, pad=pad)
        walked = occ.grid

        gap_cells = max(0, math.ceil(gap_close / cell))
        sealed = _dilate(walked, gap_cells, connectivity=8)
        exterior = _flood_from_border(~sealed)

        # 外側 + 壁(=軌跡そのもの)の両方に avatar_radius のクリアランスを付けて塞ぐ。
        # walked を含めることで内壁・柱などの浮いた壁も障害物になる。
        radius_cells = max(0, math.ceil(avatar_radius / cell))
        blocked = _dilate(exterior | walked, radius_cells, connectivity=8)
        free = ~blocked
        return cls(free=free, cell=cell, bounds=occ.bounds)


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
    reached_goal_cell: tuple[float, float]  # 実際に到達するゴール寄りセルのXZ
    goal_blocked: bool  # 目標が壁で、最寄り床に迂回したか


def plan_path(
    grid: NavGrid,
    start_xz: tuple[float, float],
    goal_xz: tuple[float, float],
    *,
    margin: float = 0.3,
    margin_weight: float = 6.0,
) -> Path | None:
    """start から goal まで壁を避けた経路を計画する。到達不能なら None。

    goal が歩けないセル(壁面のボタン等)なら、最寄りの歩けるセルまでの経路にする。

    壁際はソフトコストで避ける: 塞がりセルからの距離が ``margin`` [m] 未満のセルは
    移動コストが最大 (1 + margin_weight) 倍になる(margin=0 で無効)。障害物を
    増やすわけではないので、margin より狭い通路も通れる(遠回りが無ければ通る)。
    """
    sc = grid.world_to_cell(*start_xz)
    gc = grid.world_to_cell(*goal_xz)
    if not grid.is_free(*sc):
        nf = grid.nearest_free(*sc)
        if nf is None:
            return None
        sc = nf
    goal_blocked = not grid.is_free(*gc)
    if goal_blocked:
        nf = grid.nearest_free(*gc)
        if nf is None:
            return None
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
    for a, b in zip(waypoints, waypoints[1:]):
        length += math.hypot(b[0] - a[0], b[1] - a[1])
    return Path(
        waypoints=waypoints,
        length=length,
        reached_goal_cell=grid.cell_to_world(*gc),
        goal_blocked=goal_blocked,
    )


def _wrap180(deg: float) -> float:
    """角度を (-180, 180] に正規化。"""
    return (deg + 180.0) % 360.0 - 180.0


def heading_error(
    cur_xz: tuple[float, float], cur_yaw_deg: float, target_xz: tuple[float, float]
) -> tuple[float, float]:
    """target への (yaw誤差[deg], 水平距離[m]) を返す。yaw誤差は最短回り、+で右。"""
    dx = target_xz[0] - cur_xz[0]
    dz = target_xz[1] - cur_xz[1]
    dist = math.hypot(dx, dz)
    desired_yaw = math.degrees(math.atan2(dx, dz))
    return _wrap180(desired_yaw - cur_yaw_deg), dist


def pitch_error(
    eye_xyz: tuple[float, float, float],
    cur_forward: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
) -> float:
    """視線の pitch 誤差[deg]。+ は「もっと上を向く必要」。

    現在 pitch は forward.y から、目標 pitch は視点→ボタンの仰角から求める。
    """
    dx = target_xyz[0] - eye_xyz[0]
    dy = target_xyz[1] - eye_xyz[1]
    dz = target_xyz[2] - eye_xyz[2]
    horiz = math.hypot(dx, dz)
    desired_pitch = math.degrees(math.atan2(dy, horiz))
    fy = max(-1.0, min(1.0, cur_forward[1]))
    current_pitch = math.degrees(math.asin(fy))
    return desired_pitch - current_pitch


def aim_angle(
    eye_xyz: tuple[float, float, float],
    cur_forward: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
) -> float:
    """視線 forward と「視点→ボタン」方向との実際のなす角[deg](総合ずれの指標)。"""
    d = np.array(
        [
            target_xyz[0] - eye_xyz[0],
            target_xyz[1] - eye_xyz[1],
            target_xyz[2] - eye_xyz[2],
        ],
        dtype=np.float64,
    )
    n = np.linalg.norm(d)
    if n < 1e-9:
        return 0.0
    f = np.asarray(cur_forward, dtype=np.float64)
    f = f / (np.linalg.norm(f) + 1e-12)
    cos = float(np.clip(np.dot(d / n, f), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def forward_factor(yaw_err_deg: float, cutoff_deg: float = 90.0) -> float:
    """前進速度の減衰係数 [0,1]。正対で1、横向きで0(cos ベースで滑らか)。

    その場停止→旋回のガクつきを避けるため、向きのズレに応じて滑らかに減速する。
    |yaw_err| >= cutoff で 0。
    """
    a = abs(yaw_err_deg)
    if a >= cutoff_deg:
        return 0.0
    return max(0.0, math.cos(math.radians(a)))


def steering(
    cur_xz: tuple[float, float],
    cur_yaw_deg: float,
    target_xz: tuple[float, float],
    turn_gain: float = 0.03,
    face_thresh_deg: float = 35.0,
) -> tuple[float, float, float, float]:
    """現在位置・向きから target へ進むための (forward, turn, dist, yaw_err) を返す。

    純粋関数(OSCやスレッド非依存でテスト可能)。
    - forward: 目標へ十分正対している時のみ 1.0、そうでなければ 0.0(その場旋回)。
    - turn: LookHorizontal に渡す旋回量(+で右)。yaw誤差に比例、[-1,1]。
    yaw の規約は Pose と同じ atan2(fwd.x, fwd.z)(+Z基準・右が正)。
    """
    dx = target_xz[0] - cur_xz[0]
    dz = target_xz[1] - cur_xz[1]
    dist = math.hypot(dx, dz)
    desired_yaw = math.degrees(math.atan2(dx, dz))
    err = _wrap180(desired_yaw - cur_yaw_deg)
    turn = max(-1.0, min(1.0, turn_gain * err))
    forward = 1.0 if abs(err) < face_thresh_deg else 0.0
    return forward, turn, dist, err
