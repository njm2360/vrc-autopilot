"""同定プラント(plant.json)上で巡回制御ループの安定余裕を出す解析 CLI

各ループ(face/nav/translate/align)の ωc/PM/GM/むだ時間余裕を表にし、任意で
ボード線図PNGを保存する。ゲインは patrol-buttons / sim-face と同じフラグで上書きできる
"""

import argparse
import dataclasses
import math
from pathlib import Path

from app.cli._logging import setup_logging
from app.cli.patrol_buttons import _add_gain_args
from app.control.controller import PatrolGains
from app.control.loop_analysis import analyze_patrol, save_bode_png
from app.sysid.identify import PlantModel


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="plant.json上で巡回制御ループの安定余裕(ωc/PM/GM)を出す"
    )
    parser.add_argument("--model", required=True, help="probe-axesが出力したplant.json")
    parser.add_argument("--out", default=None, help="ボード線図PNGの出力先(任意)")
    _add_gain_args(parser)
    args = parser.parse_args()

    gains = PatrolGains(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(PatrolGains)}
    )
    plant = PlantModel.load(args.model)

    print(f"model: {args.model}  (dt {plant.dt_mean * 1000:.1f} ms)")
    print(f"{'loop':18}{'wc':>7}{'PM':>6}{'GMdB':>7}{'GMx':>6}{'DMms':>7}{'Mr':>6}")
    for m in analyze_patrol(gains, plant):
        wc = f"{m.wc:.2f}" if m.wc else "-"
        pm = f"{m.pm_deg:.0f}" if m.pm_deg is not None else "-"
        gmdb = f"{20 * math.log10(m.gm):.1f}" if m.gm else "-"
        gmx = f"{m.gm:.1f}" if m.gm else "-"
        dm = f"{m.delay_margin_s * 1000:.0f}" if m.delay_margin_s else "-"
        print(f"{m.name:18}{wc:>7}{pm:>6}{gmdb:>7}{gmx:>6}{dm:>7}{m.mr:6.2f}")

    if args.out:
        save_bode_png(gains, plant, Path(args.out))
        print(f"bode: {args.out}")


if __name__ == "__main__":
    main()
