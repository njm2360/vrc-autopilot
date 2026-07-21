import argparse
import time
from datetime import datetime
from pathlib import Path

from vrc_autopilot.cli._logging import setup_logging
from vrc_autopilot.control.controller import PatrolGains
from vrc_autopilot.sysid.identify import AXIS_INPUT
from vrc_autopilot.sysid.worldcal import (
    REF_SPEED,
    WorldCalibration,
    run_world_calibration,
)


def _print_result(cal: WorldCalibration) -> None:
    for axis, est in cal.axes.items():
        bad = "" if est.usable else f"  [unusable: {est.reason}]"
        print(
            f"  {axis:8s} x{est.scale:.3f}  {REF_SPEED[axis] * est.scale:.2f} m/s  "
            f"deadtime {est.deadtime_s * 1000:.0f}ms{bad}"
        )
    for w in cal.warnings:
        print(f"  [warn] {w}")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=None,
        help="出力 JSON(既定: logs/worldcal_<日時>.json)",
    )
    parser.add_argument(
        "--max-travel",
        type=float,
        default=3.0,
        help="プローブの往復範囲の片側幅[m]",
    )
    parser.add_argument(
        "--hold", type=float, default=1.2, help="片道の上限秒(位置ガード優先)"
    )
    parser.add_argument("--settle", type=float, default=0.6, help="レベル間の 0 保持秒")
    parser.add_argument("--start-delay", type=float, default=3.0, help="開始前の猶予秒")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    from vrc_autopilot.control.osc import VRChatOSC
    from vrc_autopilot.perception.reader import PoseReader
    from vrc_autopilot.perception.spec import HUD_ENABLE_PARAM

    reader = PoseReader().start()
    osc = VRChatOSC(host=args.host, port=args.port)
    try:
        osc.avatar_param(HUD_ENABLE_PARAM, True)
        osc.set_run(True)  # 実運用と同じ Run 押しっぱなし条件で測る
        deadline = time.monotonic() + 10.0
        while reader.get_latest() is None:
            if time.monotonic() > deadline:
                raise SystemExit(
                    f"cannot read HUD (VRChat running? {HUD_ENABLE_PARAM}=true? wrong window?)"
                )
            time.sleep(0.1)
        print(
            f"[warn] will travel about ±{args.max_travel:.1f}m forward/back and "
            f"left/right (relative to facing). starting in {args.start_delay:.0f}s..."
        )
        time.sleep(args.start_delay)
        cal = run_world_calibration(
            reader,
            lambda axis, v: osc.axis(AXIS_INPUT[axis], v),
            monotonic=time.monotonic,
            sleep=time.sleep,
            max_travel=args.max_travel,
            hold=args.hold,
            settle=args.settle,
            meta={
                "created": datetime.now().isoformat(timespec="seconds"),
                "reference": dict(REF_SPEED),
            },
        )
    finally:
        osc.close()
        reader.stop()

    out = Path(
        args.out
        or Path("logs") / f"worldcal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    cal.save(out)
    print(f"\nworld calibration: {out}")
    _print_result(cal)

    try:
        applied = cal.apply(PatrolGains())
        g = applied.gains
        print(
            f"  gains: fwd_kp {g.fwd_kp:.2f}  translate_kp {g.translate_kp:.2f}  "
            f"strafe_kp {g.strafe_kp:.2f}  speed {g.speed:.2f}"
        )
        for n in applied.notes:
            print(f"  [note] {n}")
    except ValueError as e:
        print(f"  [unusable] {e}")


if __name__ == "__main__":
    main()
