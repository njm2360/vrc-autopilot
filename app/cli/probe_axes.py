"""実機プローブ CLI: VRChat の各入力軸の応答特性を測って PlantModel にまとめる。

使い方(実機。開けた場所に立って実行):
    probe-axes                        # 4軸すべて測定 → logs/probe_*/plant.json
    probe-axes --axes yaw,strafe      # 軸を絞る
    probe-axes --axes pitch --out logs/probe_XXXX   # 1軸だけ測り直す
    probe-axes --from-log logs/probe_XXXX           # 生ログから同定だけやり直す

出力ディレクトリには生ログ CSV(probe_*.csv / segments_*.csv)、plant.json、
軸ごとの静特性プロット PNG が入る。plant.json は出力ディレクトリに生ログがある
軸すべてから組むので、1軸だけ取り直しても他の軸は残る。Ctrl-C で中断しても
完了済みの軸だけで同定する。plant.json は sim-face が読む。

必要スペース:
- yaw / pitch はその場で回るだけ。pitch は視線から ±pitch-span(既定 45°)の
  角度ガード内で振る。
- forward / strafe は開始位置から軸方向 ±max-travel(既定 3m)の範囲内で往復する
  (位置ガードで切り返す。行き過ぎマージン ≒ 最高速度×むだ時間)。狭い場所では
  --max-travel 0.4 程度まで詰められる(速い指令レベルの精度は粗くなる)。

測定値はワールド・アバター・fps に依存するので、環境が変わったら測り直す。
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from app.cli._logging import setup_logging
from app.sysid.identify import (
    AXES,
    AXIS_INPUT,
    PlantModel,
    ProbeRun,
    build_plant,
    load_run,
    look_schedule,
    run_axis_probe,
    run_move_probe,
    run_pitch_probe,
    save_run,
    schedule_duration,
)

# 1%刻みの全域掃引(1回しか測らないので細かく取る)
LOOK_LEVELS = ",".join(f"{v / 100:.2f}" for v in range(1, 101))
MOVE_LEVELS = LOOK_LEVELS


def _parse_levels(spec: str) -> list[float]:
    levels = [float(v) for v in spec.split(",") if v.strip()]
    if not levels or any(not 0.0 < v <= 1.0 for v in levels):
        raise SystemExit(f"レベルは 0<v<=1 のCSVで: {spec!r}")
    return levels


def _yaw_schedule(args) -> list:
    return look_schedule(_parse_levels(args.levels), args.hold, args.settle)


def _move_duration_cap(args) -> float:
    """移動軸の所要秒の上限(位置ガードで早く切り返せばこれより短い)。

    1レベル = 初回片道(hold) + 往復 passes 回(各 2×2hold) + 戻り(2hold) + settle。
    """
    per_level = args.move_hold * (1 + 4 * args.passes + 2) + args.settle
    return len(_parse_levels(args.move_levels)) * per_level


def _pitch_duration_cap(args) -> float:
    """pitch の所要秒の上限(角度ガードで早く切れればこれより短い)。

    1レベル = +振り(hold) + settle + −振り(hold) + settle + 戻り(2hold + settle)。
    """
    per_level = 2 * (args.pitch_hold + args.settle) + 2 * args.pitch_hold + args.settle
    return len(_parse_levels(args.levels)) * per_level + args.settle


def _axis_duration_cap(axis: str, args) -> float:
    if axis == "yaw":
        return schedule_duration(_yaw_schedule(args))
    if axis == "pitch":
        return _pitch_duration_cap(args)
    return _move_duration_cap(args)


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
        ax.set_title(f"{name}: deadtime {m.deadtime_s * 1000:.0f} ms")
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=120)
        plt.close(fig)


def _identify_and_save(
    runs: list[ProbeRun], out_dir: Path, *, source: str, plot: bool
) -> None:
    """runs に無い軸は out_dir の既存生ログから拾って plant.json を組む。"""
    by_axis = {r.axis: r for r in runs}
    for axis in AXES:
        if axis in by_axis:
            continue
        try:
            by_axis[axis] = load_run(out_dir, axis)
            print(f"  [reuse] {axis}: 既存の生ログ")
        except FileNotFoundError:
            pass
    runs = [by_axis[a] for a in AXES if a in by_axis]
    plant = build_plant(
        runs,
        meta={
            "created": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "axes": [r.axis for r in runs],
        },
    )
    path = plant.save(out_dir / "plant.json")
    print(f"\nplant model: {path}")
    print(f"  dt: mean {plant.dt_mean * 1000:.1f} ms  ({len(plant.dt_seq)} samples)")
    for name, m in plant.axes.items():
        print(
            f"  {name:8s} {len(m.points):3d} pts  deadtime {m.deadtime_s * 1000:5.0f} ms"
            f"  rate(+1.0)={m.rate(1.0):+.2f} {m.unit}"
            f"  rate(+0.5)={m.rate(0.5):+.2f} {m.unit}"
        )
    if plot:
        _plot_models(plant, out_dir)
        print(f"  plots: {out_dir}\\<axis>.png")


def _run_live(axes: list[str], out_dir: Path, args) -> list[ProbeRun]:
    from app.control.osc import VRChatOSC
    from app.perception.reader import PoseReader

    reader = PoseReader().start()
    osc = VRChatOSC(host=args.host, port=args.port)
    runs: list[ProbeRun] = []
    try:
        osc.hud_enable(True)
        deadline = time.monotonic() + 10.0
        while reader.get_latest() is None:
            if time.monotonic() > deadline:
                raise SystemExit("HUD が読めません(VRChat 起動中? HUD_Enable?)")
            time.sleep(0.1)
        total = sum(_axis_duration_cap(a, args) for a in axes)
        print(f"axes: {', '.join(axes)}  合計 ~{total:.0f}s 以内")
        if any(a in ("forward", "strafe") for a in axes):
            print(
                f"[注意] 移動軸は前後/左右に ±{args.max_travel:.1f}m ほど動きます"
                "(向いている方向基準)。"
            )
        if "pitch" in axes:
            print(f"[注意] pitch は現在の視線から ±{args.pitch_span:.0f}° 振ります。")
        print(f"{args.start_delay:.0f} 秒後に開始...")
        time.sleep(args.start_delay)
        for axis in axes:
            send = lambda v, name=AXIS_INPUT[axis]: osc.axis(name, v)
            t_start = time.monotonic()
            print(
                f"probe {axis} (/input/{AXIS_INPUT[axis]}): "
                f"~{_axis_duration_cap(axis, args):.0f}s 以内"
            )
            try:
                if axis == "yaw":
                    run = run_axis_probe(reader, send, _yaw_schedule(args), axis=axis)
                elif axis == "pitch":
                    run = run_pitch_probe(
                        reader,
                        send,
                        _parse_levels(args.levels),
                        hold=args.pitch_hold,
                        settle=args.settle,
                        span=args.pitch_span,
                    )
                else:
                    run = run_move_probe(
                        reader,
                        send,
                        _parse_levels(args.move_levels),
                        axis=axis,
                        max_travel=args.max_travel,
                        hold=args.move_hold,
                        settle=args.settle,
                        passes=args.passes,
                    )
            except KeyboardInterrupt:
                print(f"\n[中断] {axis} の記録は捨て、完了済みの軸だけで同定します")
                break
            osc.stop()
            paths = save_run(run, out_dir)
            print(
                f"  {axis} 完了 ({time.monotonic() - t_start:.0f}s): "
                f"{len(run.samples)} samples -> {paths[0]}"
            )
            runs.append(run)
    finally:
        osc.close()
        reader.stop()
    return runs


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
    parser.add_argument("--levels", default=LOOK_LEVELS, help="視点軸の指令レベルCSV")
    parser.add_argument(
        "--move-levels", default=MOVE_LEVELS, help="移動軸の指令レベルCSV"
    )
    parser.add_argument("--hold", type=float, default=1.0, help="yaw の保持秒")
    parser.add_argument(
        "--pitch-hold",
        type=float,
        default=0.8,
        help="pitch の片道上限秒(角度ガードが先に効けば短く切れる)",
    )
    parser.add_argument(
        "--pitch-span",
        type=float,
        default=45.0,
        help="pitch を開始視線から振る角度幅[°](±90°クランプ回避のガード)",
    )
    parser.add_argument(
        "--move-hold",
        type=float,
        default=1.2,
        help="移動軸の片道の上限秒(位置ガードが先に効けば短く切れる)",
    )
    parser.add_argument(
        "--max-travel",
        type=float,
        default=3,
        help="移動軸プローブの往復範囲の片側幅[m](狭い場所では小さく)",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="移動軸の1レベルあたり往復回数(1%%刻みなら隣接レベルが冗長性になるので1で十分)",
    )
    parser.add_argument("--settle", type=float, default=0.6, help="レベル間の 0 保持秒")
    parser.add_argument("--start-delay", type=float, default=3.0, help="開始前の猶予秒")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--no-plot", action="store_true", help="プロットを出さない")
    args = parser.parse_args()

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    for a in axes:
        if a not in AXES:
            parser.error(f"未知の軸: {a}(choices: {', '.join(AXES)})")

    if args.from_log:
        src = Path(args.from_log)
        out_dir = Path(args.out) if args.out else src
        out_dir.mkdir(parents=True, exist_ok=True)
        runs = []
        for axis in axes:
            try:
                runs.append(load_run(src, axis))
            except FileNotFoundError:
                print(f"  [skip] {axis}: 記録なし")
        if not runs:
            raise SystemExit(f"{src} に生ログ(probe_*.csv)が見つかりません")
        _identify_and_save(runs, out_dir, source="from-log", plot=not args.no_plot)
        return

    out_dir = Path(
        args.out or Path("logs") / f"probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = _run_live(axes, out_dir, args)
    if not runs and not any(out_dir.glob("probe_*.csv")):
        raise SystemExit("完了した軸がないため plant.json は作りません")
    _identify_and_save(runs, out_dir, source="probe", plot=not args.no_plot)


if __name__ == "__main__":
    main()
