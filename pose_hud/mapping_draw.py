import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

from .mapping import RoomMapper

OUTER_COLOR = "#1f77b4"
INNER_COLOR = "#d62728"


def room_patch(mapper: RoomMapper) -> PathPatch | None:
    """外周-穴の複合パスを塗り潰す PathPatch(リングが無ければ None)。"""
    outer, inner = mapper.room_polygon()
    if not outer:
        return None
    verts: list[list[float]] = []
    codes: list[int] = []
    for ring in outer + inner:
        verts.extend(ring.tolist())
        codes.append(MplPath.MOVETO)
        codes.extend([MplPath.LINETO] * (len(ring) - 2))
        codes.append(MplPath.CLOSEPOLY)
    path = MplPath(np.asarray(verts), codes)
    return PathPatch(path, facecolor=OUTER_COLOR, edgecolor="none", alpha=0.12)


def draw_map(
    ax,
    mapper: RoomMapper,
    *,
    cell: float = 0.1,
    show_occupancy: bool = True,
    title: str | None = None,
) -> None:
    """``ax`` に地図を描く。呼び出し側で ``ax.clear()`` 済みを想定(ライブ更新用)。"""
    pts = mapper.points
    if len(pts) == 0:
        return
    w, d = mapper.dimensions()

    if show_occupancy and len(mapper) >= 2:
        occ = mapper.occupancy_grid(cell=cell)
        ax.imshow(
            occ.grid,
            origin="lower",
            extent=occ.bounds.as_extent(),
            cmap="Greys",
            alpha=0.35,
            interpolation="nearest",
            aspect="equal",
        )

    # 部屋ポリゴン(外周 - 穴)の塗り。「回」型なら中央が抜ける。
    patch = room_patch(mapper)
    if patch is not None:
        ax.add_patch(patch)

    # 歩行軌跡(=壁の輪郭)。セグメント分割をまたいでは繋がない。外周/内壁で色分け。
    seen = {"outer": False, "inner": False}
    for seg, kind in zip(mapper.segment_points(), mapper.segment_kinds()):
        color = OUTER_COLOR if kind == "outer" else INNER_COLOR
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            "-",
            color=color,
            lw=1.2,
            label=(None if seen[kind] else f"{kind} wall"),
        )
        seen[kind] = True
    ax.plot(pts[0, 0], pts[0, 1], "o", color="#2ca02c", ms=9, label="start")
    ax.plot(pts[-1, 0], pts[-1, 1], "s", color="#9467bd", ms=8, label="now")

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title or f"Room map  {w:.2f} x {d:.2f} m  ({len(mapper)} pts)")
    ax.legend(loc="upper right", fontsize=8)
