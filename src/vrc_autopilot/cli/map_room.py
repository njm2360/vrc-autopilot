from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from vrc_autopilot.cli._logging import setup_logging
from vrc_autopilot.mapping.draw import save_map_png
from vrc_autopilot.mapping.live import Action, LiveMap, NoDisplayError
from vrc_autopilot.mapping.mapper import RoomMapper
from vrc_autopilot.perception.capture import WindowsVRChatCapture
from vrc_autopilot.perception.reader import PoseReader

REWIND_DIST = 0.5  # 巻き戻す軌跡長 [m]
REDRAW_HZ = 5.0  # ライブ地図の再描画レート
MIN_MOVE = 0.02  # 間引き距離 [m]


def _positive_float(s: str) -> float:
    v = float(s)
    if v <= 0.0:
        raise argparse.ArgumentTypeError(f"0より大きい値を指定してください: {s}")
    return v


def _nonneg_float(s: str) -> float:
    v = float(s)
    if v < 0.0:
        raise argparse.ArgumentTypeError(f"0以上の値を指定してください: {s}")
    return v


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="壁沿いに歩いて部屋の地図を作成する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out", default="maps", metavar="DIR", help="出力先ディレクトリ"
    )
    parser.add_argument(
        "--min-move",
        type=_nonneg_float,
        default=MIN_MOVE,
        metavar="M",
        help="この距離未満の移動は間引く [m](0で無効)",
    )
    parser.add_argument(
        "--rewind-dist",
        type=_positive_float,
        default=REWIND_DIST,
        metavar="M",
        help="1回あたりの巻き戻し距離 [m]",
    )
    parser.add_argument(
        "--redraw-hz",
        type=_positive_float,
        default=REDRAW_HZ,
        metavar="HZ",
        help="ライブ地図の再描画レート [Hz]",
    )
    parser.add_argument(
        "--show-occupancy",
        action="store_true",
        help="ライブ地図に占有グリッドを描画する",
    )
    return parser.parse_args()


def _handle_action(
    action: Action,
    pause_evt: threading.Event,
    cmd_q: queue.Queue[Action],
) -> None:
    if action is Action.PAUSE:
        if pause_evt.is_set():
            pause_evt.clear()
        else:
            pause_evt.set()
    else:  # OUTER / INNER / TOGGLE_MODE / REWIND / DISCARD / SAVE
        cmd_q.put(action)


def _drain(q: queue.Queue[Action]) -> Iterator[Action]:
    while True:
        try:
            yield q.get_nowait()
        except queue.Empty:
            return


def _save_map(mapper: RoomMapper, out_root: Path) -> Path:
    out_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    npz = mapper.save(out_dir / "room")
    s = mapper.to_dict()
    if s["floor_area_polygon_m2"]:
        area = f"area {s['floor_area_polygon_m2']:.2f} m2 (polygon)"
    else:  # 外周が閉じていない等でポリゴン面積が取れない場合
        area = f"area {s['floor_area_bbox_m2']:.2f} m2 (bbox)"
    print(
        f"\nroom: {s['width_x_m']:.2f} x {s['depth_z_m']:.2f} m  "
        f"({area}, path {s['path_length_m']:.2f} m, {s['points']} pts, "
        f"{s['segments']} seg: outer={s['outer_segments']} inner={s['inner_segments']})"
    )
    print(f"saved: {npz}  {npz.with_suffix('.json')}")
    print(f"map:   {save_map_png(mapper, out_dir / 'room')}")
    return out_dir


def main() -> None:
    args = _parse_args()
    setup_logging()
    reader = PoseReader(source=WindowsVRChatCapture())
    mapper = RoomMapper(min_move=args.min_move)
    pause_evt = threading.Event()
    pause_evt.set()
    cmd_q: queue.Queue[Action] = queue.Queue()

    try:
        live = LiveMap(
            on_action=lambda a: _handle_action(a, pause_evt, cmd_q),
            show_occupancy=args.show_occupancy,
        )
    except NoDisplayError as exc:
        sys.exit(f"map_room needs a GUI: {exc}")

    reader.start()

    last_t: int | None = None
    last_redraw = 0.0
    was_paused = True
    dirty = False  # 最後の保存以降に地図が変わったか
    try:
        while not live.closed:
            for cmd in _drain(cmd_q):
                if cmd is Action.OUTER:
                    mapper.set_mode("outer")
                elif cmd is Action.INNER:
                    mapper.set_mode("inner")
                elif cmd is Action.TOGGLE_MODE:
                    mapper.set_mode("inner" if mapper.mode == "outer" else "outer")
                elif cmd is Action.REWIND:
                    n = mapper.rewind(args.rewind_dist)
                    dirty = dirty or n > 0
                    live.flash(f"rewind: dropped {n} pts")
                elif cmd is Action.DISCARD:
                    n = mapper.discard_segment()
                    dirty = dirty or n > 0
                    live.flash(f"discard: dropped {n} pts of the current segment")
                elif cmd is Action.SAVE:
                    if len(mapper) == 0:
                        live.flash("nothing to save yet")
                    else:
                        out_dir = _save_map(mapper, Path(args.out))
                        dirty = False
                        live.flash(f"saved: {out_dir}")
                last_redraw = 0.0

            pose = reader.get_latest()
            if pose is not None and pose.time_ms != last_t:
                last_t = pose.time_ms
                if pause_evt.is_set():
                    if not was_paused:
                        mapper.break_segment()
                        was_paused = True
                else:
                    was_paused = False
                    if mapper.add_pose(pose):
                        dirty = True

            now = time.monotonic()
            if now - last_redraw >= 1.0 / args.redraw_hz:
                live.update(mapper, paused=pause_evt.is_set())
                last_redraw = now
            else:
                live.pump()  # 再描画しない間もキー・閉じるは拾う
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        live.close()

    if len(mapper) == 0:
        print("no trajectory collected; nothing saved.")
        return
    if dirty:  # save ボタンで保存済みなら二重保存しない
        _save_map(mapper, Path(args.out))


if __name__ == "__main__":
    main()
