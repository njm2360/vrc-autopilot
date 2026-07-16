from app.cli._logging import setup_logging
from app.perception.capture import WindowsVRChatCapture
from app.cli._keys import key_events
from app.perception.reader import PoseReader
from app.spatial.triangulate import Sighting, triangulate


def main() -> None:
    setup_logging()
    reader = PoseReader(source=WindowsVRChatCapture())
    reader.start()
    sightings: list[Sighting] = []

    print("SPACE=capture  r=reset  q=quit")

    try:
        for ch in key_events():
            if ch in (" ", "\r", "\n"):
                pose = reader.get_latest()
                if pose is None or reader.get_stats().consecutive_fail > 5:
                    print("  [skip] cannot read HUD")
                    continue
                sightings.append(Sighting.from_pose(pose))
                if len(sightings) < 2:
                    print(f"  captured {len(sightings)} (need >= 2)")
                    continue
                res = triangulate(sightings)
                x, y, z = res.point
                warn = "" if res.well_conditioned else "  [warn] rays nearly parallel"
                print(
                    f"  ({x:+.3f}, {y:+.3f}, {z:+.3f}) m  "
                    f"residual={res.residual_rms * 100:.1f}cm  n={res.n}{warn}"
                )
            elif ch in ("r", "R"):
                sightings.clear()
                print("-- reset --")
            elif ch in ("q", "Q", "\x1b"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
