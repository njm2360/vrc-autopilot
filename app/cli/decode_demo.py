"""VRChat PoseTelemetryHUD デコーダの CLI デモ。

使い方:
    uv run decode-demo            # VRChat ウィンドウを読み続けて表示(Ctrl+C で停止)
    uv run decode-demo --stats    # 1秒ごとに統計(fps/成功率など)も表示
"""

import argparse
import time

from app.cli._logging import setup_logging
from app.perception.capture import WindowNotFoundError, WindowsVRChatCapture
from app.perception.reader import PoseReader


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="VRChat 6DoF HUD decoder demo")
    parser.add_argument("--stats", action="store_true", help="統計を定期表示")
    args = parser.parse_args()

    try:
        source = WindowsVRChatCapture()
    except RuntimeError as exc:
        parser.error(str(exc))

    reader = PoseReader(source=source)
    reader.start()
    print("reading VRChat HUD... (Ctrl+C to stop)")

    last_stats = time.monotonic()
    try:
        for pose in reader.poses():
            print(
                f"pos=({pose.position[0]:+8.3f}, {pose.position[1]:+8.3f}, "
                f"{pose.position[2]:+8.3f})  yaw={pose.yaw_deg:+7.2f}  "
                f"pitch={pose.pitch_deg:+6.2f}  t={pose.time_ms}"
            )
            if args.stats and time.monotonic() - last_stats >= 1.0:
                s = reader.get_stats()
                print(
                    f"  [stats] frame_fps={s.frame_fps:5.1f} capture_fps={s.capture_fps:5.1f} "
                    f"ok={s.success_rate:5.1%} dup={s.duplicate_skipped} "
                    f"consec_fail={s.consecutive_fail}"
                )
                last_stats = time.monotonic()
    except KeyboardInterrupt:
        pass
    except WindowNotFoundError as exc:
        print(f"error: {exc}")
    finally:
        reader.stop()
        s = reader.get_stats()
        print(
            f"\nstopped. grabbed={s.frames_grabbed} new_frames={s.new_frames} "
            f"ok_rate={s.success_rate:.1%}"
        )


if __name__ == "__main__":
    main()
