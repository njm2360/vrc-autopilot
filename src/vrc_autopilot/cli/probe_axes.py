"""実機プローブ CLI: VRChat の各入力軸の応答特性を測り PlantModel(plant.json)を作る。

適応プローブ(app/sysid/adaptive.py)で折れ点を二分探索し、むだ時間は遷移
バーストで分布として測る。プチフリを検知したレベルは自動で測り直す。

plant.json は出力ディレクトリに生ログがある軸すべてから組むため、1軸だけの
取り直しや --from-log での再同定ができる。移動速度はワールド依存(ワールドを
移るたびに測り直すのでなく calibrate-world で倍率だけ補正する)。視点軸の速度と
不感帯はクライアント側の設定なのでワールド不変。むだ時間は fps 依存。
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from vrc_autopilot.cli._logging import setup_logging
from vrc_autopilot.sysid.adaptive import (
    AdaptiveConfig,
    AdaptiveResult,
    probe_axis_adaptive,
)
from vrc_autopilot.sysid.identify import (
    AXES,
    AXIS_INPUT,
    PlantModel,
    ProbeRun,
    build_plant,
    load_run,
    save_run,
)


def _plot_models(plant: PlantModel, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for name, m in plant.axes.items():
        xs = [p[0] for p in m.points]
        ys = [p[1] for p in m.points]
        fig, ax = plt.subplots(figsize=(6, 4))
        dense = np.linspace(min(xs), max(xs), 400)
        ax.plot(dense, [m.rate(c) for c in dense], "-", lw=1, color="#1f77b4")
        ax.plot(xs, ys, "o", ms=4, color="#d62728")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("command")
        ax.set_ylabel(m.unit)
        ax.set_title(
            f"{name}: onset {m.onset:.2f}  deadtime {m.deadtime_s * 1000:.0f} ms"
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=120)
        plt.close(fig)


def _print_summary(plant: PlantModel, out_dir: Path, plot: bool) -> None:
    path = plant.save(out_dir / "plant.json")
    print(f"\nplant model: {path}")
    print(f"  dt: mean {plant.dt_mean * 1000:.1f} ms  ({len(plant.dt_seq)} samples)")
    gains = {
        "yaw": "turn/nav_turn_deadzone",
        "pitch": "pitch_deadzone",
        "strafe": "strafe_deadzone",
    }
    for name, m in plant.axes.items():
        print(
            f"  {name:8s} {len(m.points):3d} pts  onset {m.onset:.3f}  "
            f"deadtime {m.deadtime_s * 1000:5.1f} ms  "
            f"rate(+1.0)={m.rate(1.0):+.2f} {m.unit}"
        )
        if name in gains:
            print(f"           [gains] set {gains[name]} = {m.onset:.2f}")
    if plot:
        _plot_models(plant, out_dir)
        print(f"  plots: {out_dir / '<axis>.png'}")


def _reuse_runs(out_dir: Path, skip: set[str]) -> dict[str, ProbeRun]:
    """out_dir にある既存生ログを拾う(今回測った軸は除く)。"""
    runs: dict[str, ProbeRun] = {}
    for axis in AXES:
        if axis in skip:
            continue
        try:
            runs[axis] = load_run(out_dir, axis)
            print(f"  [reuse] {axis}: existing raw log")
        except FileNotFoundError:
            pass
    return runs


def _print_adaptive(res: AdaptiveResult) -> None:
    st = res.deadtime_stats
    print(
        f"  onset {res.model.onset:.3f}  slope {res.slope:+.1f} {res.model.unit}/cmd  "
        f"points {len(res.model.points)}"
    )
    print(
        f"  deadtime median {st['median'] * 1000:.1f}ms  "
        f"mean {st['mean'] * 1000:.1f}  std {st['std'] * 1000:.1f}  "
        f"p95 {st['p95'] * 1000:.1f}  max {st['max'] * 1000:.1f}  "
        f"(n={st['n']}, freeze-dropped {st['dropped']})"
    )
    if res.freezes:
        print(f"  [warn] {res.freezes} freeze(s) triggered re-measurement")


def _run_live(axes: list[str], out_dir: Path, args) -> None:
    from vrc_autopilot.control.osc import VRChatOSC
    from vrc_autopilot.perception.reader import PoseReader
    from vrc_autopilot.perception.spec import HUD_ENABLE_PARAM

    reader = PoseReader().start()
    osc = VRChatOSC(host=args.host, port=args.port)
    cfg = AdaptiveConfig(
        burst_n=args.burst,
        onset_tol=args.onset_tol,
        max_travel=args.max_travel,
        pitch_span=args.pitch_span,
    )
    results: list[AdaptiveResult] = []
    try:
        osc.avatar_param(HUD_ENABLE_PARAM, True)
        osc.set_run(True)
        deadline = time.monotonic() + 10.0
        while reader.get_latest() is None:
            if time.monotonic() > deadline:
                raise SystemExit(
                    f"cannot read HUD (VRChat running? {HUD_ENABLE_PARAM}=true? wrong window?)"
                )
            time.sleep(0.1)
        if any(a in ("forward", "strafe") for a in axes):
            print(
                f"[warn] move axes travel about ±{args.max_travel:.1f}m "
                "forward/back and left/right (relative to facing)."
            )
        if "pitch" in axes:
            print(
                f"[warn] pitch homes to level via HUD, then sweeps ±"
                f"{args.pitch_span:.0f}° around the horizon (start view need not be level)."
            )
        print(f"starting in {args.start_delay:.0f}s...")
        time.sleep(args.start_delay)
        for axis in axes:

            def send(v: float, name: str = AXIS_INPUT[axis]) -> None:
                osc.axis(name, v)

            t_start = time.monotonic()
            print(f"probe {axis} (/input/{AXIS_INPUT[axis]})")
            try:
                res = probe_axis_adaptive(reader, send, axis, cfg=cfg)
            except KeyboardInterrupt:
                print(
                    f"\n[abort] dropping the {axis} recording; "
                    "building from the completed axes only"
                )
                break
            except (ValueError, RuntimeError) as e:
                print(f"  [fail] {axis}: {e} -- skipping this axis")
                osc.stop()
                continue
            osc.stop()
            save_run(res.run, out_dir)
            print(f"  {axis} done ({time.monotonic() - t_start:.0f}s)")
            _print_adaptive(res)
            results.append(res)
    finally:
        osc.close()
        reader.stop()

    extra = _reuse_runs(out_dir, {r.axis for r in results})
    runs = [r.run for r in results] + list(extra.values())
    if not runs:
        raise SystemExit("no axis completed; not writing plant.json")
    plant = build_plant(
        runs, meta={"created": datetime.now().isoformat(timespec="seconds")}
    )
    _print_summary(plant, out_dir, plot=not args.no_plot)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="VRChat 入力軸の応答特性を測定し PlantModel(plant.json)を作る"
    )
    parser.add_argument(
        "--axes",
        default=",".join(AXES),
        help=f"測定する軸のCSV(既定: {','.join(AXES)})",
    )
    parser.add_argument(
        "--out", default=None, help="出力ディレクトリ(既定: logs/probe_<日時>)"
    )
    parser.add_argument(
        "--from-log",
        metavar="DIR",
        default=None,
        help="生ログ CSV から同定だけやり直す(実機不要)",
    )
    parser.add_argument(
        "--burst", type=int, default=40, help="むだ時間バーストの遷移回数"
    )
    parser.add_argument(
        "--onset-tol", type=float, default=0.01, help="折れ点の探索分解能(指令値)"
    )
    parser.add_argument(
        "--pitch-span",
        type=float,
        default=70.0,
        help="pitch を水平から振る角度幅[°](±80°クランプ回避のガード)",
    )
    parser.add_argument(
        "--max-travel",
        type=float,
        default=3,
        help="移動軸プローブの往復範囲の片側幅[m](狭い場所では小さく)",
    )
    parser.add_argument("--start-delay", type=float, default=3.0, help="開始前の猶予秒")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--no-plot", action="store_true", help="プロットを出さない")
    args = parser.parse_args()

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    for a in axes:
        if a not in AXES:
            parser.error(f"unknown axis: {a} (choices: {', '.join(AXES)})")

    if args.from_log:
        src = Path(args.from_log)
        out_dir = Path(args.out) if args.out else src
        out_dir.mkdir(parents=True, exist_ok=True)
        runs = []
        for axis in axes:
            try:
                runs.append(load_run(src, axis))
            except FileNotFoundError:
                print(f"  [skip] {axis}: no recording")
        if not runs:
            raise SystemExit(f"no raw logs (probe_*.csv) found in {src}")
        plant = build_plant(runs, meta={"source": "from-log"})
        _print_summary(plant, out_dir, plot=not args.no_plot)
        return

    out_dir = Path(
        args.out or Path("logs") / f"probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_live(axes, out_dir, args)


if __name__ == "__main__":
    main()
