"""RoomMapper: 歩行軌跡から部屋の地図(床平面の間取り)を作る。

壁沿いに歩いた 6DoF ポーズ列を床平面(XZ)へ投影すると、その軌跡自体が部屋の輪郭に
なる。ここではその軌跡の蓄積・寸法計測・占有グリッドへの描き込み・保存/読込を扱う。
描画は mapping_render.py(matplotlib)に分離。

座標系は Unity 準拠(Y-up, 左手系, 単位メートル)。床平面は水平な XZ、Y は高さ。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .pose import Pose


@dataclass(frozen=True)
class Bounds:
    """床平面 XZ の軸並行バウンディングボックス [m]。"""

    xmin: float
    xmax: float
    zmin: float
    zmax: float

    @property
    def width(self) -> float:
        """X方向の広がり [m]。"""
        return self.xmax - self.xmin

    @property
    def depth(self) -> float:
        """Z方向の広がり [m]。"""
        return self.zmax - self.zmin

    def padded(self, pad: float) -> "Bounds":
        return Bounds(self.xmin - pad, self.xmax + pad, self.zmin - pad, self.zmax + pad)

    def as_extent(self) -> tuple[float, float, float, float]:
        """matplotlib imshow 用 (left, right, bottom, top)。"""
        return (self.xmin, self.xmax, self.zmin, self.zmax)


@dataclass
class OccupancyGrid:
    """歩いた経路をグリッドのセルに描き込んだ占有グリッド。"""

    grid: np.ndarray            # (rows=Z, cols=X) bool。True=通過
    cell: float                 # 1セルの辺長 [m]
    bounds: Bounds              # グリッドが覆う XZ 範囲(pad込み)

    def world_to_index(self, x: float, z: float) -> tuple[int, int]:
        col = int((x - self.bounds.xmin) / self.cell)
        row = int((z - self.bounds.zmin) / self.cell)
        return row, col

    @property
    def visited_area(self) -> float:
        """通過セルの総面積 [m^2](経路の掃過面積であって部屋面積ではない)。"""
        return float(self.grid.sum()) * self.cell * self.cell


class RoomMapper:
    """壁沿い歩行のポーズ列を蓄積して部屋地図を作るアキュムレータ。

    ライブ収集の例::

        mapper = RoomMapper()
        for pose in reader.poses():
            mapper.add_pose(pose)
        mapper.save("room")            # room.npz + room.json

    ``min_move`` [m] 未満しか動いていない点は間引く(静止時の 60fps 重複でメモリと
    ノイズが増えるのを防ぐ)。0 で無効。
    """

    def __init__(self, min_move: float = 0.02):
        self.min_move = min_move
        self._xyz: list[tuple[float, float, float]] = []
        self._yaw: list[float] = []
        self._t: list[int] = []
        self._seg: list[int] = []      # 各点のセグメントID(ペンアップで分割)
        self._cur_seg: int = 0
        self._last_xz: tuple[float, float] | None = None

    # ---- 追加 ----------------------------------------------------------
    def add_pose(self, pose: Pose) -> bool:
        """Pose を1点追加する。間引かれたら False。"""
        x, y, z = pose.position
        return self.add(x, y, z, pose.yaw_deg, pose.time_ms)

    def add(self, x: float, y: float, z: float, yaw: float = 0.0, t: int = 0) -> bool:
        if self._last_xz is not None and self.min_move > 0.0:
            dx = x - self._last_xz[0]
            dz = z - self._last_xz[1]
            if dx * dx + dz * dz < self.min_move * self.min_move:
                return False
        self._xyz.append((float(x), float(y), float(z)))
        self._yaw.append(float(yaw))
        self._t.append(int(t))
        self._seg.append(self._cur_seg)
        self._last_xz = (x, z)
        return True

    def break_segment(self) -> None:
        """ペンアップ。以後に追加する点を新しいセグメントにする。

        キャプチャ一時停止のように、軌跡が不連続になる箇所で呼ぶ。分割をまたぐ点どうしは
        線で繋がない(壁を横切る偽の線を防ぐ)。壁伝いに一周できず、いったん壁から離れて
        別の壁区間へ移動する場合に使う。
        """
        if self._seg and self._seg[-1] == self._cur_seg:
            self._cur_seg += 1
        self._last_xz = None

    def __len__(self) -> int:
        return len(self._xyz)

    @property
    def num_segments(self) -> int:
        """記録されたセグメント数(連続して歩いた区間の数)。"""
        return len(set(self._seg))

    def segment_points(self) -> list[np.ndarray]:
        """セグメントごとの XZ 点列のリスト(描画用。分割をまたいで繋がない)。"""
        pts = self.points
        if len(pts) == 0:
            return []
        seg = np.asarray(self._seg)
        return [pts[seg == s] for s in dict.fromkeys(self._seg)]

    # ---- 取り出し ------------------------------------------------------
    @property
    def xyz(self) -> np.ndarray:
        """(N, 3) の位置配列 [m]。"""
        return np.asarray(self._xyz, dtype=np.float64).reshape(-1, 3)

    @property
    def points(self) -> np.ndarray:
        """(N, 2) の床平面 XZ 軌跡 [m]。"""
        a = self.xyz
        return a[:, [0, 2]] if len(a) else a.reshape(-1, 2)

    @property
    def yaw(self) -> np.ndarray:
        return np.asarray(self._yaw, dtype=np.float64)

    @property
    def time_ms(self) -> np.ndarray:
        return np.asarray(self._t, dtype=np.uint32)

    # ---- 計測 ----------------------------------------------------------
    def bounds(self) -> Bounds | None:
        """XZ バウンディングボックス。点が無ければ None。"""
        pts = self.points
        if len(pts) == 0:
            return None
        return Bounds(
            float(pts[:, 0].min()), float(pts[:, 0].max()),
            float(pts[:, 1].min()), float(pts[:, 1].max()),
        )

    def dimensions(self) -> tuple[float, float]:
        """(幅X, 奥行Z) [m]。点が無ければ (0, 0)。"""
        b = self.bounds()
        return (b.width, b.depth) if b else (0.0, 0.0)

    def height_range(self) -> tuple[float, float]:
        """(最小Y, 最大Y) [m]。視点高さの変動(段差検出などの目安)。"""
        a = self.xyz
        if len(a) == 0:
            return (0.0, 0.0)
        return (float(a[:, 1].min()), float(a[:, 1].max()))

    def path_length(self) -> float:
        """歩いた総経路長 [m](セグメント分割をまたぐ区間は数えない)。"""
        pts = self.points
        if len(pts) < 2:
            return 0.0
        seg = np.asarray(self._seg)
        same = seg[:-1] == seg[1:]                     # 同一セグメント内の辺のみ
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return float(d[same].sum())

    # ---- 占有グリッド --------------------------------------------------
    def occupancy_grid(self, cell: float = 0.1, pad: float = 0.5) -> OccupancyGrid:
        """経路をセル解像度 ``cell`` [m] のグリッドに描き込んだ占有グリッドを返す。

        連続する点の間を線分として等間隔にサンプリングし、通過したセルを True にする。
        """
        b = self.bounds()
        if b is None:
            raise ValueError("no points to rasterize")
        gb = b.padded(pad)
        cols = max(1, int(np.ceil(gb.width / cell)) + 1)
        rows = max(1, int(np.ceil(gb.depth / cell)) + 1)
        grid = np.zeros((rows, cols), dtype=bool)

        samples = self._segment_samples(cell)          # (M, 2) XZ [m]
        ci = ((samples[:, 0] - gb.xmin) / cell).astype(np.intp)
        ri = ((samples[:, 1] - gb.zmin) / cell).astype(np.intp)
        np.clip(ci, 0, cols - 1, out=ci)
        np.clip(ri, 0, rows - 1, out=ri)
        grid[ri, ci] = True
        return OccupancyGrid(grid=grid, cell=cell, bounds=gb)

    def _segment_samples(self, step: float) -> np.ndarray:
        """全線分を等間隔サンプルした点列を返す(セグメントごとのループなしでベクトル化)。

        セグメント分割(ペンアップ)をまたぐ辺は繋がない。孤立点も落とさないよう、
        全ての元の点はそのまま含める。
        """
        pts = self.points
        if len(pts) <= 1:
            return pts.copy()
        seg = np.asarray(self._seg)
        same = seg[:-1] == seg[1:]                     # 同一セグメント内の辺のみ補間
        p0 = pts[:-1][same]
        p1 = pts[1:][same]
        if len(p0) == 0:
            return pts.copy()
        seg_len = np.linalg.norm(p1 - p0, axis=1)
        nsteps = np.maximum(1, np.ceil(seg_len / max(step * 0.5, 1e-9)).astype(np.intp))
        total = int(nsteps.sum())
        edge_id = np.repeat(np.arange(len(nsteps)), nsteps)
        start = np.repeat(np.cumsum(nsteps) - nsteps, nsteps)
        frac = (np.arange(total) - start) / np.repeat(nsteps, nsteps)
        interp = p0[edge_id] + (p1[edge_id] - p0[edge_id]) * frac[:, None]
        return np.concatenate([pts, interp])           # 元の点 + 辺の補間点

    # ---- 要約 / 保存 ---------------------------------------------------
    def to_dict(self) -> dict:
        b = self.bounds()
        w, d = self.dimensions()
        ymin, ymax = self.height_range()
        return {
            "points": len(self),
            "bounds": None if b is None else {
                "xmin": b.xmin, "xmax": b.xmax, "zmin": b.zmin, "zmax": b.zmax,
            },
            "width_x_m": w,
            "depth_z_m": d,
            "floor_area_bbox_m2": w * d,
            "path_length_m": self.path_length(),
            "height_min_m": ymin,
            "height_max_m": ymax,
            "segments": self.num_segments,
        }

    def save(self, path: str | Path) -> Path:
        """<path>.npz(軌跡)と <path>.json(要約)を保存し、npz のパスを返す。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        npz = path.with_suffix(".npz")
        np.savez(npz, xyz=self.xyz, yaw=self.yaw, time_ms=self.time_ms,
                 seg=np.asarray(self._seg, dtype=np.int32),
                 min_move=np.float64(self.min_move))
        path.with_suffix(".json").write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return npz

    @classmethod
    def load(cls, path: str | Path) -> "RoomMapper":
        """save() で書いた npz から復元する。"""
        data = np.load(Path(path).with_suffix(".npz"))
        m = cls(min_move=float(data["min_move"]) if "min_move" in data else 0.0)
        xyz = data["xyz"]
        yaw = data["yaw"] if "yaw" in data else np.zeros(len(xyz))
        t = data["time_ms"] if "time_ms" in data else np.zeros(len(xyz), np.uint32)
        seg = data["seg"] if "seg" in data else np.zeros(len(xyz), np.int32)
        m._xyz = [tuple(map(float, row)) for row in xyz]
        m._yaw = [float(v) for v in yaw]
        m._t = [int(v) for v in t]
        m._seg = [int(v) for v in seg]
        m._cur_seg = int(seg[-1]) if len(seg) else 0
        if m._xyz:
            m._last_xz = (m._xyz[-1][0], m._xyz[-1][2])
        return m

    @classmethod
    def from_poses(cls, poses: Iterable[Pose], min_move: float = 0.02) -> "RoomMapper":
        m = cls(min_move=min_move)
        for p in poses:
            m.add_pose(p)
        return m
