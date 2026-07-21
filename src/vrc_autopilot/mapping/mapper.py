import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..core.pose import Pose

Kind = Literal["outer", "inner"]
_KINDS: tuple[str, str] = ("outer", "inner")  # 保存時の int 対応: outer=0, inner=1


def _norm_kind(kind: str) -> str:
    k = str(kind).lower()
    if k not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    return k


def _signed_area(ring: np.ndarray) -> float:
    """XZ ポリゴンの符号付き面積(shoelace)。反時計回りで正。"""
    x, z = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))


def _close_ring(ring: np.ndarray, ccw: bool) -> np.ndarray:
    """向きを ccw に揃え、先頭点を末尾に付けて閉じた (M, 2) 配列を返す。"""
    r = np.asarray(ring, dtype=np.float64)
    if (_signed_area(r) < 0.0) == ccw:  # 望む向きと逆なら反転
        r = r[::-1]
    if not np.allclose(r[0], r[-1]):
        r = np.vstack([r, r[0]])
    return r


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

    def padded(self, pad: float) -> Bounds:
        return Bounds(
            self.xmin - pad, self.xmax + pad, self.zmin - pad, self.zmax + pad
        )

    def as_extent(self) -> tuple[float, float, float, float]:
        """matplotlib imshow 用 (left, right, bottom, top)。"""
        return (self.xmin, self.xmax, self.zmin, self.zmax)


@dataclass
class OccupancyGrid:
    """歩いた経路をグリッドのセルに描き込んだ占有グリッド。"""

    grid: np.ndarray  # (rows=Z, cols=X) bool。True=通過
    cell: float  # 1セルの辺長 [m]
    bounds: Bounds  # グリッドが覆う XZ 範囲(pad込み)

    def world_to_cell(self, x: float, z: float) -> tuple[int, int]:
        col = int((x - self.bounds.xmin) / self.cell)
        row = int((z - self.bounds.zmin) / self.cell)
        return row, col

    @property
    def visited_area(self) -> float:
        """通過セルの総面積 [m^2](経路の掃過面積であって部屋面積ではない)。"""
        return float(self.grid.sum()) * self.cell * self.cell


class RoomMapper:
    """壁沿い歩行のポーズ列を蓄積して部屋地図を作るアキュムレータ。

    min_move [m] 未満しか動いていない点は間引く(静止時の 60fps 重複によるノイズ増を
    防ぐ)。0 で無効。

    セグメントは kind を持つ(outer=外周 / inner=内壁)。外周は部屋の外リング、内壁は
    穴(柱・中庭など「回」型の内側)として room_polygon で扱う。set_mode でモード切替。
    壁から浮いた時の補正は rewind(末尾を距離ぶん消す)と discard_segment(現在セグメント
    を丸ごと破棄)。
    """

    def __init__(self, min_move: float = 0.02):
        self.min_move = min_move
        self._xyz: list[tuple[float, float, float]] = []
        self._yaw: list[float] = []
        self._t: list[int] = []
        self._seg: list[int] = []  # 各点のセグメントID(ペンアップで分割)
        self._cur_seg: int = 0
        self._kind: list[str] = [
            "outer"
        ]  # セグメントID -> kind。常に len == _cur_seg+1
        self._mode: str = "outer"  # 現在の記録モード(=_kind[_cur_seg])
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

    def _cur_has_points(self) -> bool:
        """現在セグメントに点があるか(点は順に追加されるので末尾で判定できる)。"""
        return bool(self._seg) and self._seg[-1] == self._cur_seg

    def break_segment(self) -> None:
        """ペンアップ。以後の点を新しいセグメント(同じモード)にする。

        軌跡が不連続になる箇所で呼ぶ。分割をまたぐ点どうしは線で繋がず、壁を横切る
        偽の線を防ぐ。
        """
        if self._cur_has_points():
            self._cur_seg += 1
            self._kind.append(self._mode)
        self._last_xz = None

    # ---- モード切替(外周 / 内壁) -------------------------------------
    @property
    def mode(self) -> str:
        """現在の記録モード("outer" or "inner")。"""
        return self._mode

    def set_mode(self, kind: str) -> None:
        """記録モードを切り替える(outer=外周 / inner=内壁・穴)。

        点のあるセグメントはペンアップしてから切り替え、1セグメント内に外周と内壁を
        混在させない。点が無ければ現在セグメントの kind を差し替えるだけ。
        """
        kind = _norm_kind(kind)
        if kind == self._mode:
            return
        if self._cur_has_points():
            self._cur_seg += 1
            self._kind.append(kind)
            self._last_xz = None
        else:
            self._kind[self._cur_seg] = kind
        self._mode = kind

    # ---- 再走行補正 ----------------------------------------------------
    def rewind(self, distance_m: float = 0.5) -> int:
        """現在セグメント末尾の点を、辿った距離が distance_m を超えるまで消す。

        壁から浮いた末尾区間を消して歩き直すための補正。前のセグメントには遡らない。
        消した点数を返す。
        """
        removed = 0
        acc = 0.0
        while self._cur_has_points():
            x, _, z = self._xyz[-1]
            self._pop_last()
            removed += 1
            if self._cur_has_points():
                px, _, pz = self._xyz[-1]
                acc += float(np.hypot(x - px, z - pz))
                if acc >= distance_m:
                    break
            else:
                break
        self._refresh_last_xz()
        return removed

    def discard_segment(self) -> int:
        """現在セグメントを丸ごと破棄する。ID とモードは維持され、同じセグメントに
        取り直せる。破棄した点数を返す。"""
        removed = 0
        while self._cur_has_points():
            self._pop_last()
            removed += 1
        self._last_xz = None
        return removed

    def _pop_last(self) -> None:
        self._xyz.pop()
        self._yaw.pop()
        self._t.pop()
        self._seg.pop()

    def _refresh_last_xz(self) -> None:
        if self._cur_has_points():
            x, _, z = self._xyz[-1]
            self._last_xz = (x, z)
        else:
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

    def segment_kinds(self) -> list[str]:
        """segment_points と同じ順で、各セグメントの kind を返す。"""
        return [self._kind[s] for s in dict.fromkeys(self._seg)]

    def _point_kinds(self) -> np.ndarray:
        """各点の kind(文字列)配列 (N,)。"""
        if not self._seg:
            return np.empty(0, dtype="<U5")
        return np.array([self._kind[s] for s in self._seg], dtype="<U5")

    # ---- 部屋ポリゴン(外周 - 穴) -------------------------------------
    def _rings_by_kind(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """(外周リング列, 穴リング列)。3点以上あるセグメントのみをリングとして採用。"""
        outer, inner = [], []
        for pts, kind in zip(self.segment_points(), self.segment_kinds(), strict=True):
            if len(pts) < 3:
                continue
            (outer if kind == "outer" else inner).append(pts)
        return outer, inner

    def room_polygon(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """描画・面積計算用の (外周リング列, 穴リング列)。

        各リングは閉じた (M, 2) XZ 配列。外周は反時計回り、穴は時計回りに向き付けする
        (nonzero 塗り規則で穴が抜ける)。
        """
        outer, inner = self._rings_by_kind()
        outer = [_close_ring(r, ccw=True) for r in outer]
        inner = [_close_ring(r, ccw=False) for r in inner]
        return outer, inner

    def room_area(self) -> float:
        """外周ポリゴン面積から穴の面積を引いた床面積 [m^2](リングが無ければ 0)。"""
        outer, inner = self._rings_by_kind()
        if not outer:
            return 0.0
        area = sum(abs(_signed_area(r)) for r in outer)
        area -= sum(abs(_signed_area(r)) for r in inner)
        return float(max(0.0, area))

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
            float(pts[:, 0].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].min()),
            float(pts[:, 1].max()),
        )

    def outer_bounds(self) -> Bounds | None:
        """外周(outer)区間だけの XZ バウンディングボックス。部屋の寸法は内壁を含めず
        外周から測る。外周点が無ければ全点にフォールバック。"""
        pts = self.points
        if len(pts) == 0:
            return None
        mask = self._point_kinds() == "outer"
        sel = pts[mask] if mask.any() else pts
        return Bounds(
            float(sel[:, 0].min()),
            float(sel[:, 0].max()),
            float(sel[:, 1].min()),
            float(sel[:, 1].max()),
        )

    def dimensions(self) -> tuple[float, float]:
        """(幅X, 奥行Z) [m]。外周から測る。点が無ければ (0, 0)。"""
        b = self.outer_bounds()
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
        same = seg[:-1] == seg[1:]  # 同一セグメント内の辺のみ
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return float(d[same].sum())

    # ---- 占有グリッド --------------------------------------------------
    def occupancy_grid(
        self, cell: float = 0.1, pad: float = 0.5, kind: Kind | None = None
    ) -> OccupancyGrid:
        """経路をセル解像度 cell [m] のグリッドに描き込んだ占有グリッドを返す。

        連続する点の間を線分として等間隔にサンプリングし、通過セルを True にする。
        kind 指定時はその種別のセグメントのみ。グリッドの寸法・原点は kind によらず
        全点の bounds で共通(重ね合わせ可能)。
        """
        b = self.bounds()
        if b is None:
            raise ValueError("no points to rasterize")
        gb = b.padded(pad)
        cols = max(1, int(np.ceil(gb.width / cell)) + 1)
        rows = max(1, int(np.ceil(gb.depth / cell)) + 1)
        grid = np.zeros((rows, cols), dtype=bool)

        samples = self._segment_samples(cell, kind=kind)  # (M, 2) XZ [m]
        if len(samples):
            ci = ((samples[:, 0] - gb.xmin) / cell).astype(np.intp)
            ri = ((samples[:, 1] - gb.zmin) / cell).astype(np.intp)
            np.clip(ci, 0, cols - 1, out=ci)
            np.clip(ri, 0, rows - 1, out=ri)
            grid[ri, ci] = True
        return OccupancyGrid(grid=grid, cell=cell, bounds=gb)

    def _segment_samples(self, step: float, kind: str | None = None) -> np.ndarray:
        """全線分を等間隔サンプルした点列(ベクトル化)。

        ペンアップをまたぐ辺は繋がない。孤立点を落とさないよう元の点は全て含める。
        kind 指定時はその種別のセグメントに限る。
        """
        pts = self.points
        if len(pts) == 0:
            return pts.copy()
        seg = np.asarray(self._seg)
        if kind is None:
            keep = np.ones(len(pts), dtype=bool)
        else:
            keep = np.asarray(self._kind, dtype=object)[seg] == kind
        if len(pts) == 1:
            return pts[keep].copy()
        same = (seg[:-1] == seg[1:]) & keep[:-1]  # 同一セグメント内の辺のみ補間
        p0 = pts[:-1][same]
        p1 = pts[1:][same]
        if len(p0) == 0:
            return pts[keep].copy()
        seg_len = np.linalg.norm(p1 - p0, axis=1)
        nsteps = np.maximum(1, np.ceil(seg_len / max(step * 0.5, 1e-9)).astype(np.intp))
        total = int(nsteps.sum())
        edge_id = np.repeat(np.arange(len(nsteps)), nsteps)
        start = np.repeat(np.cumsum(nsteps) - nsteps, nsteps)
        frac = (np.arange(total) - start) / np.repeat(nsteps, nsteps)
        interp = p0[edge_id] + (p1[edge_id] - p0[edge_id]) * frac[:, None]
        return np.concatenate([pts[keep], interp])  # 元の点 + 辺の補間点

    # ---- 要約 / 保存 ---------------------------------------------------
    def to_dict(self) -> dict:
        b = self.bounds()
        w, d = self.dimensions()
        ymin, ymax = self.height_range()
        kinds = self.segment_kinds()
        return {
            "points": len(self),
            "bounds": (
                None
                if b is None
                else {
                    "xmin": b.xmin,
                    "xmax": b.xmax,
                    "zmin": b.zmin,
                    "zmax": b.zmax,
                }
            ),
            "width_x_m": w,
            "depth_z_m": d,
            "floor_area_bbox_m2": w * d,
            "floor_area_polygon_m2": self.room_area(),
            "path_length_m": self.path_length(),
            "height_min_m": ymin,
            "height_max_m": ymax,
            "segments": self.num_segments,
            "outer_segments": kinds.count("outer"),
            "inner_segments": kinds.count("inner"),
        }

    def save(self, path: str | Path) -> Path:
        """<path>.npz(軌跡)と <path>.json(要約)を保存し、npz のパスを返す。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        npz = path.with_suffix(".npz")
        # 点の無い末尾セグメント(モード切替やペンアップ直後)の kind は保存しない。
        # load 側は _cur_seg = seg[-1] で復元するため、点のあるセグメント分だけ書く
        n_seg = (max(self._seg) + 1) if self._seg else 1
        kind = np.array([_KINDS.index(k) for k in self._kind[:n_seg]], dtype=np.int8)
        np.savez(
            npz,
            xyz=self.xyz,
            yaw=self.yaw,
            time_ms=self.time_ms,
            seg=np.asarray(self._seg, dtype=np.int32),
            kind=kind,
            min_move=np.float64(self.min_move),
        )
        path.with_suffix(".json").write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return npz

    @classmethod
    def load(cls, path: str | Path) -> RoomMapper:
        """save() で書いた npz から復元する。"""
        data = np.load(Path(path).with_suffix(".npz"))
        m = cls(min_move=float(data["min_move"]))
        seg = data["seg"]
        m._xyz = [tuple(map(float, row)) for row in data["xyz"]]
        m._yaw = [float(v) for v in data["yaw"]]
        m._t = [int(v) for v in data["time_ms"]]
        m._seg = [int(v) for v in seg]
        m._cur_seg = int(seg[-1]) if len(seg) else 0
        m._kind = [_KINDS[int(v)] for v in data["kind"]]
        m._mode = m._kind[m._cur_seg]
        if m._xyz:
            m._last_xz = (m._xyz[-1][0], m._xyz[-1][2])
        return m

    @classmethod
    def from_poses(cls, poses: Iterable[Pose], min_move: float = 0.02) -> RoomMapper:
        m = cls(min_move=min_move)
        for p in poses:
            m.add_pose(p)
        return m
