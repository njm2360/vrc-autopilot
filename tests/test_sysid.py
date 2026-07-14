"""システム同定パイプラインの往復テスト。

既知の合成プラント(SimulatedVRChat)に模擬クロックでプローブをかけ、同定結果が
元の特性(静特性カーブ・むだ時間)と一致することを実機なしで検証する。
"""

import math

import pytest

from app.control.controller import (
    PatrolGains,
    face_controllers,
    nav_controllers,
)
from app.control.maneuvers import aim_at, follow_path, turn_to
from app.sysid.simplant import SimClock, SimulatedVRChat
from app.sysid.identify import (
    AxisModel,
    PlantModel,
    ProbeRun,
    build_plant,
    extract_dts,
    identify_axis,
    load_run,
    look_schedule,
    move_schedule,
    run_axis_probe,
    run_move_probe,
    run_pitch_probe,
    save_run,
)
from app.control.telemetry import ListRecorder

# VRChat 視点軸ふうの静特性: 0.55 以下はごく遅く、超えると急峻に立ち上がる
YAW_CURVE = [
    (-1.0, -90.0),
    (-0.6, -25.0),
    (-0.55, -1.2),
    (0.0, 0.0),
    (0.55, 1.2),
    (0.6, 25.0),
    (1.0, 90.0),
]
PITCH_CURVE = [(-1.0, -60.0), (0.0, 0.0), (1.0, 60.0)]
MOVE_CURVE = [(-1.0, -2.0), (0.0, 0.0), (1.0, 2.0)]
DEAD = 0.1
DT = 0.05


def make_plant(dead: float = DEAD, dt: float = DT) -> PlantModel:
    return PlantModel(
        axes={
            "yaw": AxisModel("yaw", "deg/s", YAW_CURVE, dead),
            "pitch": AxisModel("pitch", "deg/s", PITCH_CURVE, dead),
            "forward": AxisModel("forward", "m/s", MOVE_CURVE, dead),
            "strafe": AxisModel("strafe", "m/s", MOVE_CURVE, dead),
        },
        dt_mean=dt,
    )


def probe(sim: SimulatedVRChat, axis: str, segments):
    send = {
        "yaw": lambda v: sim.look(turn=v),
        "pitch": lambda v: sim.look(pitch=v),
        "forward": lambda v: sim.move(forward=v),
        "strafe": lambda v: sim.move(strafe=v),
    }[axis]
    clk = SimClock(sim)
    return run_axis_probe(
        sim, send, segments, axis=axis, monotonic=clk.monotonic, sleep=clk.sleep
    )


def test_yaw_static_curve_roundtrip():
    plant = make_plant()
    sim = SimulatedVRChat(plant)
    run = probe(
        sim, "yaw", look_schedule([0.3, 0.5, 0.55, 0.6, 0.8, 1.0], hold=1.0, settle=0.4)
    )
    model = identify_axis(run)
    assert model.unit == "deg/s"
    # プローブしたレベル(両符号)で元のカーブと一致する
    for cmd in (0.3, 0.5, 0.55, 0.6, 0.8, 1.0, -0.3, -0.55, -0.8, -1.0):
        true = plant.axes["yaw"].rate(cmd)
        assert model.rate(cmd) == pytest.approx(true, rel=0.05, abs=0.2)
    # 不感帯の折れ点(0.55→0.6 の急峻な立ち上がり)が再現されている
    assert abs(model.rate(0.55)) < 3.0
    assert model.rate(0.6) > 15.0


def test_deadtime_recovery():
    plant = make_plant(dead=0.1)
    sim = SimulatedVRChat(plant)
    run = probe(sim, "yaw", look_schedule([0.6, 0.8, 1.0], hold=1.0, settle=0.4))
    model = identify_axis(run)
    # 真値 0.1s に対しフレーム量子化(dt=0.05)の範囲で復元される
    assert 0.05 <= model.deadtime_s <= 0.22


def test_movement_projection_independent_of_heading():
    # 体の向きが斜めでも、左右移動の変位が右方向へ正しく射影される
    plant = make_plant()
    sim = SimulatedVRChat(plant, yaw=37.0)
    run = probe(sim, "strafe", move_schedule([0.5, 1.0], hold=0.8, settle=0.4))
    model = identify_axis(run)
    assert model.unit == "m/s"
    assert model.rate(1.0) == pytest.approx(2.0, rel=0.05)
    assert model.rate(-0.5) == pytest.approx(-1.0, rel=0.05)


def test_move_probe_stays_within_band():
    """省スペース移動プローブ: 位置ガードで移動範囲が ±max_travel+むだ時間マージンに収まる。"""
    plant = make_plant()  # 最高 2 m/s, むだ時間 0.1s
    sim = SimulatedVRChat(plant, yaw=20.0)
    clk = SimClock(sim)
    max_travel = 0.3
    run = run_move_probe(
        sim,
        lambda v: sim.move(strafe=v),
        [0.1, 0.5, 1.0],
        axis="strafe",
        max_travel=max_travel,
        monotonic=clk.monotonic,
        sleep=clk.sleep,
    )
    # ホーム(最初のサンプル位置)からの右方向変位が ±max_travel を大きく超えない。
    # 行き過ぎ上限 ≒ 最高速度 × (むだ時間 + 2フレーム)
    yr = math.radians(run.samples[0].yaw)
    dx, dz = math.cos(yr), -math.sin(yr)
    x0, z0 = run.samples[0].x, run.samples[0].z
    projs = [(s.x - x0) * dx + (s.z - z0) * dz for s in run.samples]
    margin = 2.0 * (DEAD + 2 * DT)
    assert max(abs(p) for p in projs) <= max_travel + margin
    # 終了時はホーム近くに戻っている
    assert abs(projs[-1]) < 0.15
    # 静特性は範囲内の短い往復からでも復元できる
    model = identify_axis(run)
    assert model.rate(1.0) == pytest.approx(2.0, rel=0.1)
    assert model.rate(0.5) == pytest.approx(1.0, rel=0.1)
    assert model.rate(-0.1) == pytest.approx(-0.2, rel=0.1)


def make_fast_pitch_plant(rate: float = 120.0) -> PlantModel:
    plant = make_plant()
    plant.axes["pitch"] = AxisModel(
        "pitch", "deg/s", [(-1.0, -rate), (0.0, 0.0), (1.0, rate)], DEAD
    )
    return plant


def test_pitch_probe_respects_guard_and_recovers_curve():
    """角度ガード付き pitch プローブ: 速い軸でもクランプに張り付かず特性を復元する。"""
    plant = make_fast_pitch_plant(120.0)
    sim = SimulatedVRChat(plant, pitch=10.0)  # ホームが中央からずれていてもよい
    clk = SimClock(sim)
    span = 30.0
    run = run_pitch_probe(
        sim,
        lambda v: sim.look(pitch=v),
        [0.25, 0.5, 1.0],
        hold=1.5,
        settle=0.4,
        span=span,
        monotonic=clk.monotonic,
        sleep=clk.sleep,
    )
    # 行き過ぎ上限 ≒ 最高速度 × (むだ時間 + 2フレーム)。クランプ(±89°)には遠い
    margin = 120.0 * (DEAD + 2 * DT)
    assert max(abs(s.pitch) for s in run.samples) <= 10.0 + span + 2 * margin
    assert max(abs(s.pitch) for s in run.samples) < 88.0
    model = identify_axis(run)
    assert model.rate(1.0) == pytest.approx(120.0, rel=0.1)
    assert model.rate(-1.0) == pytest.approx(-120.0, rel=0.1)
    assert model.rate(0.5) == pytest.approx(60.0, rel=0.1)
    assert 0.05 <= model.deadtime_s <= 0.22


def test_identify_excludes_pitch_clamp_saturation():
    """時間ベース加振で ±90° に張り付いた記録でも、張り付き前のランプから傾きを取れる。"""
    plant = make_fast_pitch_plant(120.0)  # 1.5s 保持で確実にクランプへ到達する
    sim = SimulatedVRChat(plant)
    run = probe(sim, "pitch", look_schedule([1.0], hold=1.5, settle=0.4))
    assert max(abs(s.pitch) for s in run.samples) >= 88.0  # 実際に張り付いている
    model = identify_axis(run)
    assert model.rate(1.0) == pytest.approx(120.0, rel=0.1)


class _StallingReader:
    """指定時刻以降、最後のポーズを返し続ける(HUD フリーズの模擬)。"""

    def __init__(self, src, clk: SimClock, stall_at: float):
        self.src, self.clk, self.stall_at = src, clk, stall_at
        self._frozen = None

    def get_latest(self):
        if self.clk.monotonic() >= self.stall_at:
            return self._frozen
        self._frozen = self.src.get_latest()
        return self._frozen


def test_guarded_probe_cuts_command_on_hud_stall():
    """HUD 途絶中はガードを評価できないので、blind_cap で指令を切って暴走距離を抑える。"""
    plant = make_plant()  # 最高 2 m/s
    sim = SimulatedVRChat(plant)
    clk = SimClock(sim)
    reader = _StallingReader(sim, clk, stall_at=0.9)  # 最初の +v 区間の途中で途絶
    try:
        run_move_probe(
            reader,
            lambda v: sim.move(forward=v),
            [1.0],
            axis="forward",
            max_travel=5.0,  # 位置ガードは発火させない(途絶時の挙動だけ見る)
            hold=2.0,
            monotonic=clk.monotonic,
            sleep=clk.sleep,
        )
    except RuntimeError:
        pass  # 途絶が wait_cap まで続けば HUD lost で落ちるのも正しい挙動
    # 指令はカットされ、hold いっぱい(2 m/s × ~1.9s ≈ 3.8m)までは走らない
    pose = sim.get_latest()
    assert abs(pose.position[2]) < 2.0


def test_build_plant_skips_failed_axis():
    """1軸の同定失敗(サンプル不足)が他軸を巻き込まない。"""
    plant = make_plant()
    sim = SimulatedVRChat(plant)
    good = probe(sim, "yaw", look_schedule([0.8], hold=1.0, settle=0.4))
    bad = ProbeRun(axis="pitch", samples=[], seg_starts=[])
    built = build_plant([good, bad])
    assert "yaw" in built.axes and "pitch" not in built.axes


def test_all_control_loops_run_against_sim():
    """本番の全制御ループ(移動追従+正対)が模擬プラント注入で無改造・非実時間で回る。"""
    plant = make_plant()
    gains = PatrolGains(nav_timeout=10.0, face_timeout=2.5)
    sim = SimulatedVRChat(plant)  # 原点, yaw=0(+Z向き)
    nav = follow_path(
        sim,
        sim,
        sim,
        [(0.0, 0.0), (0.0, 1.5)],
        gains,
        nav_controllers(gains),
        clock=SimClock(sim),
    )
    aim = aim_at(
        sim,
        sim,
        (1.0, 1.5, 4.0),
        gains,
        face_controllers(gains),
        clock=SimClock(sim),
        recorder=ListRecorder(),
    )
    assert nav.arrived and nav.reason == "arrived"
    assert aim.frames >= 5
    assert aim.yaw is not None and aim.pitch is not None
    assert aim.converged or abs(aim.yaw_err) < 19.0  # 初期誤差 ≈19.7° から減っている


def test_extract_dts():
    plant = make_plant(dt=0.05)
    sim = SimulatedVRChat(plant)
    run = probe(sim, "yaw", look_schedule([0.8], hold=1.0, settle=0.4))
    dts = extract_dts(run)
    assert len(dts) > 10
    assert all(dt == pytest.approx(0.05, abs=0.01) for dt in dts)


def test_plant_json_roundtrip(tmp_path):
    plant = make_plant()
    plant.dt_seq = [0.05, 0.06, 0.049]
    plant.meta = {"source": "test"}
    path = plant.save(tmp_path / "plant.json")
    loaded = PlantModel.load(path)
    assert loaded.dt_mean == pytest.approx(plant.dt_mean)
    assert loaded.dt_seq == pytest.approx(plant.dt_seq)
    assert loaded.meta == plant.meta
    assert set(loaded.axes) == set(plant.axes)
    for name, m in plant.axes.items():
        lm = loaded.axes[name]
        assert lm.unit == m.unit
        assert lm.deadtime_s == pytest.approx(m.deadtime_s)
        assert lm.points == pytest.approx(m.points)


def test_run_csv_roundtrip(tmp_path):
    plant = make_plant()
    sim = SimulatedVRChat(plant)
    run = probe(sim, "yaw", look_schedule([0.6, 1.0], hold=0.8, settle=0.4))
    save_run(run, tmp_path)
    loaded = load_run(tmp_path, "yaw")
    m1 = identify_axis(run)
    m2 = identify_axis(loaded)
    assert m2.points == pytest.approx(m1.points)
    assert m2.deadtime_s == pytest.approx(m1.deadtime_s)


def test_turn_to_runs_against_sim():
    """本番の正対ループ(turn_to)が模擬プラントで無改造・非実時間で回る(実時間はかからない)。"""
    plant = make_plant()
    gains = PatrolGains(face_timeout=2.0)
    sim = SimulatedVRChat(plant)
    res = turn_to(
        sim,
        sim,
        25.0,
        gains,
        face_controllers(gains),
        clock=SimClock(sim),
        recorder=ListRecorder(),
    )
    assert res.frames > 10
    assert res.yaw is not None  # 応答指標(osc / overshoot 等)が取れている
    assert abs(res.yaw_err) < 25.0  # 誤差は初期値から減っている
