import argparse
import dataclasses
import math
from datetime import datetime
from pathlib import Path

from app.cli._logging import setup_logging
from app.control.actuator import MouseLookActuator
from app.control.controller import PatrolGains
from app.control.telemetry import ControlLog
from app.mapping.mapper import RoomMapper
from app.spatial.navigation import NavGrid, plan_path
from app.control.pilot import Pilot

# (name, xz, y, face_yaw_deg) face_yaw_deg=ボタンの向き(壁の外向き法線, +Z基準)
Target = tuple[str, tuple[float, float], float, float]


def _parse_targets(args) -> list[Target]:
    targets: list[Target] = []
    for i, spec in enumerate(args.target or []):
        parts = [float(v) for v in spec.split(",")]
        if len(parts) != 4:
            raise SystemExit(f"--target は 'x,y,z,face_yaw' 形式で: {spec!r}")
        targets.append((f"t{i + 1}", (parts[0], parts[2]), parts[1], parts[3]))
    return targets


def _standoff_xz(
    tgt: tuple[float, float], face_yaw: float, standoff: float
) -> tuple[float, float]:
    if standoff <= 0.0:
        return tgt
    y = math.radians(face_yaw)
    return (tgt[0] + math.sin(y) * standoff, tgt[1] + math.cos(y) * standoff)


def _plan_tour(
    grid: NavGrid,
    start: tuple[float, float],
    targets: list[Target],
    standoff: float = 0.0,
):
    cur = start
    legs = []
    for name, tgt, _y, face_yaw in targets:
        path = plan_path(grid, cur, _standoff_xz(tgt, face_yaw, standoff))
        legs.append((name, tgt, path))
        if path is not None:
            cur = path.reached_goal_cell
    return legs


def _render_plan(grid: NavGrid, start, legs, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = grid.bounds
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        grid.free,
        origin="lower",
        extent=b.as_extent(),
        cmap="Greys_r",
        alpha=0.5,
        interpolation="nearest",
        aspect="equal",
    )
    ax.plot(start[0], start[1], "o", color="#2ca02c", ms=10, label="start")
    for name, tgt, path in legs:
        ax.plot(tgt[0], tgt[1], "X", color="#d62728", ms=11)
        ax.annotate(name, tgt, textcoords="offset points", xytext=(6, 6), fontsize=8)
        if path is not None:
            wx = [p[0] for p in path.waypoints]
            wz = [p[1] for p in path.waypoints]
            ax.plot(wx, wz, "-", lw=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Patrol plan (white=walkable)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = out.with_suffix(".png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _run_live(grid, targets, args, gains: PatrolGains) -> None:
    log_path = Path("logs") / f"patrol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = ControlLog(log_path)
    print(f"control log: {log_path}")

    look = (
        MouseLookActuator(
            yaw_gain=args.mouse_yaw_gain, pitch_gain=args.mouse_pitch_gain
        )
        if args.look == "mouse"
        else None  # 省略時は OSC
    )
    pilot = Pilot.connect(grid, gains=gains, look=look, recorder=log)
    print(f"look={args.look}  waiting for HUD...")
    pilot.wait_for_hud()
    try:
        for name, tgt_xz, tgt_y, face_yaw in targets:
            print(f"-> {name} {tgt_xz} y={tgt_y} face_yaw={face_yaw:g}")
            pilot.visit((tgt_xz[0], tgt_y, tgt_xz[1]), face_yaw, name=name)
        print("patrol done.")
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        pilot.close()
        log.close()


def _add_gain_args(parser) -> None:
    d = PatrolGains()
    parser.add_argument(
        "--speed", type=float, default=d.speed, help="巡航前進速度の上限(0..1)"
    )
    parser.add_argument(
        "--arrive", type=float, default=d.arrive, help="ウェイポイント到達半径[m]"
    )
    parser.add_argument(
        "--nav-lookahead",
        type=float,
        default=d.nav_lookahead,
        help="経路先読み(carrot)の弧長[m]",
    )
    parser.add_argument(
        "--standoff",
        type=float,
        default=d.standoff,
        help="ボタン手前で止まる距離[m](0=直下まで詰める)",
    )
    parser.add_argument(
        "--face-tol", type=float, default=d.face_tol, help="正対とみなす角度[deg]"
    )
    parser.add_argument(
        "--settle",
        type=int,
        default=d.settle,
        help="収束に必要な、連続で正対を保ったフレーム数",
    )
    parser.add_argument(
        "--nav-timeout", type=float, default=d.nav_timeout, help="移動の打切り秒"
    )
    parser.add_argument(
        "--face-timeout", type=float, default=d.face_timeout, help="正対の打切り秒"
    )
    # 正対(face)の yaw: 視点軸は0.50以下が反応しないので out_deadzone で飛び越える(OSC用)。
    parser.add_argument("--turn-kp", type=float, default=d.turn_kp)
    parser.add_argument("--turn-ki", type=float, default=d.turn_ki)
    parser.add_argument("--turn-kd", type=float, default=d.turn_kd)
    parser.add_argument(
        "--turn-ilim", type=float, default=d.turn_ilim, help="yaw積分項の絶対上限"
    )
    parser.add_argument(
        "--turn-deadzone",
        type=float,
        default=d.turn_deadzone,
        help="正対 yaw の不感帯補償",
    )
    # 移動中(nav)の yaw: face と同機構の不感帯補償つき(kp は補償ぶん低め)。
    parser.add_argument("--nav-turn-kp", type=float, default=d.nav_turn_kp)
    parser.add_argument("--nav-turn-ki", type=float, default=d.nav_turn_ki)
    parser.add_argument("--nav-turn-kd", type=float, default=d.nav_turn_kd)
    parser.add_argument(
        "--nav-turn-deadzone",
        type=float,
        default=d.nav_turn_deadzone,
        help="nav yaw の不感帯補償",
    )
    # 視点固定の並進(move_to): 進行方向へ回さず前後+横で経路を追う。誤差=残距離[m]。
    parser.add_argument("--hmove-kp", type=float, default=d.hmove_kp)
    parser.add_argument("--hmove-ki", type=float, default=d.hmove_ki)
    parser.add_argument("--hmove-kd", type=float, default=d.hmove_kd)
    parser.add_argument(
        "--hmove-ilim", type=float, default=d.hmove_ilim, help="並進積分項の絶対上限"
    )
    parser.add_argument("--pitch-kp", type=float, default=d.pitch_kp)
    parser.add_argument("--pitch-ki", type=float, default=d.pitch_ki)
    parser.add_argument("--pitch-kd", type=float, default=d.pitch_kd)
    parser.add_argument(
        "--pitch-ilim", type=float, default=d.pitch_ilim, help="pitch積分項の絶対上限"
    )
    parser.add_argument(
        "--pitch-deadzone",
        type=float,
        default=d.pitch_deadzone,
        help="正対 pitch の不感帯補償(実測オンセット0.10。0で無効)",
    )
    parser.add_argument("--fwd-kp", type=float, default=d.fwd_kp)
    parser.add_argument("--fwd-kd", type=float, default=d.fwd_kd)
    # 最終照準(align): 残り yaw 誤差を横移動で吸収(視点軸の不感帯回避)
    parser.add_argument(
        "--align-tol",
        type=float,
        default=d.align_tol,
        help="align の横方向誤差の収束閾値[m](0=alignフェーズ無効)",
    )
    parser.add_argument(
        "--align-timeout", type=float, default=d.align_timeout, help="align の打切り秒"
    )
    parser.add_argument(
        "--align-stuck-time",
        type=float,
        default=d.align_stuck_time,
        help="指令を出しても動けない状態がこの秒数続いたら打切り(角のデッドロック防止)",
    )
    parser.add_argument(
        "--align-stuck-eps",
        type=float,
        default=d.align_stuck_eps,
        help="スタック判定の移動距離閾値[m]",
    )
    parser.add_argument("--strafe-kp", type=float, default=d.strafe_kp)
    parser.add_argument("--strafe-ki", type=float, default=d.strafe_ki)
    parser.add_argument("--strafe-kd", type=float, default=d.strafe_kd)
    parser.add_argument(
        "--strafe-ilim",
        type=float,
        default=d.strafe_ilim,
        help="strafe積分項の絶対上限",
    )
    parser.add_argument(
        "--strafe-deadzone",
        type=float,
        default=d.strafe_deadzone,
        help="align 横移動の不感帯補償(実測オンセット0.10。0だと tol 手前で失速する)",
    )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Patrol buttons on a saved room map, avoiding walls"
    )
    parser.add_argument(
        "--map", required=True, help="部屋マップ .npz(map_room.py 出力)"
    )
    parser.add_argument(
        "--target",
        action="append",
        metavar="X,Y,Z,FACE_YAW",
        help="ボタン座標と向き(複数可)。FACE_YAW=壁の外向き法線[deg](+Z基準)。"
        "その正面 standoff[m] に立つ",
    )
    parser.add_argument("--cell", type=float, default=0.1, help="グリッド解像度[m]")
    parser.add_argument(
        "--radius", type=float, default=0.25, help="アバター半径=壁クリアランス[m]"
    )
    parser.add_argument(
        "--gap-close", type=float, default=0.6, help="塞ぐ軌跡の隙間の最大幅[m]"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="計画のみ(操作しない)。マップ隣に plan.png を自動保存",
    )
    # アクチュエータ選択(視点のみ差し替え可。移動は OSC 固定)
    parser.add_argument(
        "--look",
        choices=("osc", "mouse"),
        default="osc",
        help="視点アクチュエータ(mouse=DirectInput相対マウス。要 pydirectinput)",
    )
    parser.add_argument(
        "--mouse-yaw-gain",
        type=float,
        default=40.0,
        help="マウス視点の水平ゲイン[px/指令]",
    )
    parser.add_argument(
        "--mouse-pitch-gain",
        type=float,
        default=40.0,
        help="マウス視点の上下ゲイン[px/指令]",
    )
    _add_gain_args(parser)
    args = parser.parse_args()

    # チューニング定数を1オブジェクトに集約(フラグは PatrolGains の既定を上書き)。
    gains = PatrolGains(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(PatrolGains)}
    )

    mapper = RoomMapper.load(args.map)
    grid = NavGrid.from_mapper(
        mapper, cell=args.cell, avatar_radius=args.radius, gap_close=args.gap_close
    )
    free_ratio = grid.free.mean()
    print(
        f"map: {len(mapper)}pts  grid {grid.shape[1]}x{grid.shape[0]}  "
        f"walkable {free_ratio:.0%}  dims {tuple(round(v, 2) for v in mapper.dimensions())}m"
    )

    targets = _parse_targets(args)
    if not targets:
        parser.error("--target x,y,z を1つ以上指定してください")

    p0 = mapper.points[0]
    start = (float(p0[0]), float(p0[1]))

    legs = _plan_tour(grid, start, targets, standoff=args.standoff)
    print(f"\nplan from {tuple(round(v, 2) for v in start)}:")
    for name, tgt, path in legs:
        if path is None:
            print(f"  {name} {tgt}: 到達不能")
        else:
            note = " (壁→最寄り床)" if path.goal_blocked else ""
            print(f"  {name} {tgt}: {len(path.waypoints)}wp / {path.length:.2f}m{note}")

    if args.dry_run:
        png = _render_plan(grid, start, legs, Path(args.map).with_name("plan.png"))
        print(f"plan figure: {png}")
        return

    _run_live(grid, targets, args, gains)


if __name__ == "__main__":
    main()
