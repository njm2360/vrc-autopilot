from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ヘッドレス(PNG 保存)
import matplotlib.pyplot as plt

from .mapping import RoomMapper
from .mapping_draw import draw_map


def render_map(
    mapper: RoomMapper,
    out_path: str | Path,
    cell: float = 0.1,
    show_occupancy: bool = True,
    title: str | None = None,
) -> Path:
    if len(mapper) == 0:
        raise ValueError("no points to render")

    out_path = Path(out_path).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_map(ax, mapper, cell=cell, show_occupancy=show_occupancy, title=title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
