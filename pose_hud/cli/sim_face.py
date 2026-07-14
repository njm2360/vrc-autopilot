"""同定済みプラントに対する正対ループのオフライン検証(実機不要)。

probe-axes が作った plant.json を SimulatedVRChat に読み込み、本番と同じ
turn_to(_face_loop)を実時間で回して、初期誤差ごとの収束時間・振動回数・
オーバーシュートを表にする。PID ゲインは patrol-buttons と同じフラグで
上書きできるので、実機に持ち込む前にゲインの当たりをここで付ける。

例:
    sim-face --model logs/probe_XXXX/plant.json
    sim-face --model ... --turn-kp 0.05 --turn-deadzone 0.5 --yaw-err 30,5,2
"""

import argparse
import dataclasses
from pathlib import Path

from pose_hud.cli._ctl_log import ControlLog
from pose_hud.cli.patrol_buttons import _add_gain_args
from pose_hud.controller import PatrolGains, face_controllers
from pose_hud.maneuvers import turn_to
from pose_hud.simplant import SimulatedVRChat
from pose_hud.sysid import PlantModel
from pose_hud.telemetry import ListRecorder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="plant.json 上で正対(turn_to)ループを回しゲインを検証する"
    )
    parser.add_argument(
        "--model", required=True, help="probe-axes が出力した plant.json"
    )
    parser.add_argument(
        "--yaw-err",
        default="30,10,5,2",
        help="試す初期 yaw 誤差[deg]のCSV",
    )
    parser.add_argument(
        "--pitch-err",
        type=float,
        default=None,
        help="初期 pitch 誤差[deg](省略時は yaw のみ制御)",
    )
    parser.add_argument(
        "--log", default=None, help="全試行のフレーム記録 CSV の出力先(任意)"
    )
    _add_gain_args(parser)
    args = parser.parse_args()

    gains = PatrolGains(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(PatrolGains)}
    )
    plant = PlantModel.load(args.model)
    errs = [float(v) for v in args.yaw_err.split(",") if v.strip()]

    recorder = ControlLog(Path(args.log)) if args.log else ListRecorder()
    print(f"model: {args.model}  (dt {plant.dt_mean * 1000:.0f} ms)")
    print(
        f"gains: kp={gains.turn_kp} ki={gains.turn_ki} kd={gains.turn_kd} "
        f"deadzone={gains.turn_deadzone} tol={gains.face_tol}°"
    )
    print(f"{'err':>8}  {'result':6}  {'time':>6}  {'final':>7}  "
          f"{'osc':>3}  {'overshoot':>9}  {'settle':>6}")
    ok = 0
    for err in errs:
        sim = SimulatedVRChat(plant).start_realtime()
        try:
            res = turn_to(
                sim,
                sim,
                err,  # sim は yaw=0 で始まるので目標=初期誤差
                gains,
                face_controllers(gains),
                pitch_deg=args.pitch_err,  # sim は pitch=0 で始まるので目標=初期誤差
                recorder=recorder,
                name=f"err{err:g}",
            )
        finally:
            sim.close()
        ok += res.converged
        m = res.yaw
        print(
            f"{err:+7.1f}°  {'OK' if res.converged else 'NG':6}  "
            f"{res.elapsed:5.2f}s  {res.yaw_err:+6.2f}°  "
            f"{m.osc if m else '-':>3}  "
            f"{f'{m.overshoot:.2f}°' if m else '-':>9}  "
            f"{f'{m.settle_time:.2f}s' if m and m.settle_time is not None else '-':>6}"
        )
    print(f"\nconverged {ok}/{len(errs)}")
    if args.log:
        recorder.close()
        print(f"frame log: {args.log}")


if __name__ == "__main__":
    main()
