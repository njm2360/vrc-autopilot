import queue
import threading
import time
from datetime import datetime
from pathlib import Path

from pose_hud.capture import WindowsVRChatCapture
from pose_hud.cli._keys import key_events
from pose_hud.mapping import RoomMapper
from pose_hud.mapping_live import LiveMap
from pose_hud.reader import PoseReader

REWIND_DIST = 0.5  # z を1回押すたびに巻き戻す軌跡長 [m]
REDRAW_HZ = 5.0  # ライブ地図の再描画レート


def _handle_key(
    ch: str,
    pause_evt: threading.Event,
    stop_evt: threading.Event,
    cmd_q: "queue.Queue[str]",
) -> None:
    """1キーを解釈する。mapper は触らず、コマンドはキュー経由でメインスレッドへ渡す。

    コンソール(msvcrt)とライブ地図窓(matplotlib)の両方から呼ばれる。matplotlib は
    スペースを "space"、ESC を "escape" として渡してくる点に注意。
    """
    if ch in (" ", "space"):
        (pause_evt.clear if pause_evt.is_set() else pause_evt.set)()
    elif ch in ("o", "O"):
        cmd_q.put("outer")
    elif ch in ("i", "I"):
        cmd_q.put("inner")
    elif ch in ("z", "Z"):
        cmd_q.put("rewind")
    elif ch in ("r", "R"):
        cmd_q.put("redo")
    elif ch in ("q", "Q", "escape", "\x1b", "\x03"):  # q / ESC / Ctrl+C
        stop_evt.set()


def _key_thread(
    pause_evt: threading.Event, stop_evt: threading.Event, cmd_q: "queue.Queue[str]"
) -> None:
    for ch in key_events():
        _handle_key(ch, pause_evt, stop_evt, cmd_q)
        if stop_evt.is_set():
            return


def main() -> None:
    reader = PoseReader(source=WindowsVRChatCapture())
    mapper = RoomMapper()
    pause_evt = threading.Event()
    stop_evt = threading.Event()
    cmd_q: "queue.Queue[str]" = queue.Queue()

    live = LiveMap(on_key=lambda ch: _handle_key(ch, pause_evt, stop_evt, cmd_q))

    reader.start()
    threading.Thread(
        target=_key_thread, args=(pause_evt, stop_evt, cmd_q), daemon=True
    ).start()
    print(
        "recording...  SPACE=一時停止/再開  o=外周  i=内壁(穴)  "
        "z=巻き戻し  r=区間やり直し  q=保存して終了"
    )
    if live.headless:
        print("  (ライブ地図を表示できる GUI が無いためテキスト表示のみ)")

    last_t = None
    last_report = time.monotonic()
    last_redraw = 0.0
    was_paused = False
    try:
        while not stop_evt.is_set():
            if live.closed:  # 地図窓を閉じたら終了扱い
                break

            # キー操作コマンドはメインスレッドで mapper に反映(スレッド安全)
            while True:
                try:
                    cmd = cmd_q.get_nowait()
                except queue.Empty:
                    break
                if cmd == "outer":
                    mapper.set_mode("outer")
                elif cmd == "inner":
                    mapper.set_mode("inner")
                elif cmd == "rewind":
                    n = mapper.rewind(REWIND_DIST)
                    print(f"  rewind: {n} 点を巻き戻し(壁沿いに歩き直してください)")
                elif cmd == "redo":
                    n = mapper.redo_segment()
                    print(f"  redo: 現在セグメントの {n} 点を破棄")

            pose = reader.get_latest()
            if pose is not None and pose.time_ms != last_t:
                last_t = pose.time_ms
                if pause_evt.is_set():
                    if not was_paused:
                        mapper.break_segment()
                        was_paused = True
                else:
                    was_paused = False
                    mapper.add_pose(pose)

            now = time.monotonic()
            paused = pause_evt.is_set()
            if now - last_redraw >= 1.0 / REDRAW_HZ:
                w, d = mapper.dimensions()
                live.update(
                    mapper,
                    title=(
                        f"{'PAUSED ' if paused else ''}mode={mapper.mode}  "
                        f"{w:.2f}x{d:.2f} m  {len(mapper)} pts  "
                        f"seg={mapper.num_segments}"
                    ),
                )
                last_redraw = now
            else:
                live.pump()  # 再描画しない間もキー・閉じるは拾う

            if now - last_report >= 1.0:
                w, d = mapper.dimensions()
                state = "PAUSED" if paused else "rec   "
                print(
                    f"  [{state}] mode={mapper.mode:5s} pts={len(mapper):5d}  "
                    f"seg={mapper.num_segments}  bbox={w:5.2f}x{d:5.2f}m  "
                    f"path={mapper.path_length():6.2f}m"
                )
                last_report = now
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        live.close()

    if len(mapper) == 0:
        print("no trajectory collected; nothing saved.")
        return

    from pose_hud.mapping_render import render_map  # Agg 固定なので保存時のみ import

    out_dir = Path("maps") / datetime.now().strftime("%Y%m%d_%H%M%S")
    npz = mapper.save(out_dir / "room")
    s = mapper.to_dict()
    area = s["floor_area_polygon_m2"] or s["floor_area_bbox_m2"]
    print(
        f"\nroom: {s['width_x_m']:.2f} x {s['depth_z_m']:.2f} m  "
        f"(area {area:.2f} m2, path {s['path_length_m']:.2f} m, {s['points']} pts, "
        f"{s['segments']} seg: outer={s['outer_segments']} inner={s['inner_segments']})"
    )
    print(f"saved: {npz}  {npz.with_suffix('.json')}")
    print(f"map:   {render_map(mapper, out_dir / 'room')}")


if __name__ == "__main__":
    main()
