"""システム同定パイプラインの往復テスト。

既知の合成プラント(SimulatedVRChat)に模擬クロックでプローブをかけ、同定結果が
元の特性(静特性カーブ・むだ時間)と一致することを実機なしで検証する。
"""

import math

import numpy as np
import pytest

from app.control.controller import (
    PatrolGains,
    face_controllers,
    nav_controllers,
)
from app.control.maneuvers import aim_at, follow_path, turn_to
from app.control.recording import ListRecorder
from app.sysid.identify import (
    AxisModel,
    PlantModel,
    ProbeRun,
    ProbeSample,
    _denoise_points,
    _median3,
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
from app.sysid.sim_plant import SimClock, SimulatedVRChat

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
    # 終了時はホーム帯の近くに戻っている(帯 + むだ時間と1フレームぶんの行き過ぎ)
    assert abs(projs[-1]) < 0.08 + 2.0 * (DEAD + DT)
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


def test_median3_kills_isolated_spike_and_preserves_shape():
    # 孤立スパイクは隣の値に潰れる
    assert _median3([0.0, 1.0, 2.0, 9.0, 4.0, 5.0]) == pytest.approx(
        [0.0, 1.0, 2.0, 4.0, 5.0, 5.0]
    )
    # 単調な区間(不感帯・折れ点含む)は厳密に不変
    knee = [-9.0, -4.0, 0.0, 0.0, 0.0, 4.0, 9.0]
    assert _median3(knee) == pytest.approx(knee)
    assert _median3([0.0, 1.0, 2.0, 3.0]) == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_denoise_fixes_endpoint_spike_on_uniform_grid():
    """等間隔グリッドの端の外れ値は内側からの外挿で潰れる(線形な端は不変)。"""
    pts = [(-1.0, -237.0), (-0.99, -180.0), (-0.98, -176.0), (-0.97, -172.0)]
    out = _denoise_points(pts, "pitch")
    assert out[0][1] == pytest.approx(-184.0)  # スパイク除去
    assert [y for _, y in out[1:]] == pytest.approx([-180.0, -176.0, -172.0])
    # 線形に伸びる端は不変
    lin = [(0.0, 0.0), (0.01, 1.0), (0.02, 2.0), (0.03, 3.0)]
    assert _denoise_points(lin, "yaw") == pytest.approx(lin)


def test_denoise_keeps_endpoints_on_irregular_grid():
    """間隔が不揃いなカーブ(レベル欠損・手書き)では端点を外挿で壊さない。"""
    pts = [(-1.0, -90.0), (-0.6, -25.0), (-0.55, -1.2), (0.0, 0.0)]
    out = _denoise_points(pts, "yaw")
    assert out == pytest.approx(pts)


def test_median3_is_sign_agnostic():
    """負ゲイン軸(単調減少カーブ)も一切歪めない。"""
    dec = [90.0, 25.0, 1.2, 0.0, -1.2, -25.0, -90.0]
    assert _median3(dec) == pytest.approx(dec)


def test_median3_does_not_staircase_noise():
    """小さな測定ノイズを階段状のプールへ均さない(値の多様性が保たれる)。"""
    rng = np.random.default_rng(0)
    true = np.linspace(0.0, 100.0, 51)
    noisy = (true + rng.normal(0.0, 1.0, true.size)).tolist()
    out = _median3(noisy)
    # 各点が近傍の実測値のまま残る(平坦なプールへ均されない)こと
    assert len(set(out)) >= 40
    assert max(abs(a - b) for a, b in zip(out, true, strict=True)) < 4.0


def _spiked_yaw() -> AxisModel:
    # yaw の 0.705 に孤立スパイク(周囲の傾きから大きく外れる)を仕込んだカーブ
    return AxisModel(
        "yaw",
        "deg/s",
        [
            (-1.0, -90.0),
            (0.0, 0.0),
            (0.55, 1.2),
            (0.705, 82.8),
            (0.715, 40.0),
            (1.0, 90.0),
        ],
        0.1,
    )


def test_save_load_roundtrip_is_exact(tmp_path):
    """load(save(x)) == x。読込時にフィルタを再適用しない(3点メディアンは
    非冪等なので、再適用すると保存・プロットしたカーブとシムが食い違う)。"""
    plant = make_plant()
    plant.axes["yaw"] = _spiked_yaw()  # 再フィルタされれば必ず動く点を含む
    path = plant.save(tmp_path / "plant.json")
    loaded = PlantModel.load(path)
    for name, m in plant.axes.items():
        assert loaded.axes[name].points == pytest.approx(m.points)


def test_identified_curve_matches_plant():
    """同定した静特性がプラント真値に一致し、階段状に潰れない。"""
    plant = make_plant()
    sim = SimulatedVRChat(plant)
    levels = [0.3, 0.5, 0.55, 0.6, 0.8, 1.0]
    run = probe(sim, "yaw", look_schedule(levels, hold=1.0, settle=0.4))
    model = identify_axis(run)
    true = plant.axes["yaw"]
    for c in levels:
        assert model.rate(c) == pytest.approx(true.rate(c), rel=0.1, abs=0.5)
    # 隣接レベルが同一値へプールされていない(0.55→0.6 の急峻な立ち上がりが残る)
    assert model.rate(0.6) - model.rate(0.55) > 10.0


def test_short_segments_are_rejected():
    """定常サンプルが足りないセグメントは緩和フィットせず棄却される。"""
    dt = 0.03
    samples: list[ProbeSample] = []
    # seg0: 十分長い(1.0s) cmd=+0.5、定常 10deg/s
    for k in range(34):
        t = k * dt
        samples.append(ProbeSample(0, 0.5, t, int(t * 1000), 0, 0, 0, 10.0 * t, 0))
    # seg1: 0.15s しかない cmd=+1.0(skip_min=0.2 を差し引くと空 → 棄却されるべき)
    t1 = 1.02
    for k in range(6):
        t = t1 + k * dt
        samples.append(
            ProbeSample(1, 1.0, t, int(t * 1000), 0, 0, 0, 10.2 + 200.0 * (t - t1), 0)
        )
    run = ProbeRun(
        axis="yaw",
        samples=samples,
        seg_starts=[(0, 0.5, 0.0), (1, 1.0, t1)],
    )
    model = identify_axis(run)
    cmds = [c for c, _ in model.points]
    assert 0.5 in cmds
    assert 1.0 not in cmds  # 短いセグメントの捏造レートが混入しない


def test_deadtime_interpolates_crossing():
    """しきい値到達時刻をサンプル間で線形補間し、量子化による上振れを避ける。

    既知むだ時間の合成ステップ応答(0→非0 遷移)を1本作り、推定が真値±半フレーム内。
    """
    dt = 0.05
    true_dead = 0.1
    rate = 30.0  # deg/s(しきい値 0.1deg には即到達=補間の効きを見る)
    # seg0: cmd=0 の基準サンプル、seg1: cmd=+1 でむだ時間後に一定速度で立ち上がる
    samples: list[ProbeSample] = []
    t_send0 = 0.0
    # 基準(遷移直前)サンプル
    samples.append(ProbeSample(0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0))
    t_send1 = dt
    n = 0
    for k in range(1, 30):
        t = t_send1 + k * dt
        elapsed = max(0.0, t - t_send1 - true_dead)
        yaw = rate * elapsed  # むだ時間後に線形に回る
        samples.append(ProbeSample(1, 1.0, t, int(t * 1000), 0.0, 0.0, 0.0, yaw, 0.0))
        n += 1
    run = ProbeRun(
        axis="yaw",
        samples=samples,
        seg_starts=[(0, 0.0, t_send0), (1, 1.0, t_send1)],
    )
    model = identify_axis(run)
    assert model.deadtime_s == pytest.approx(true_dead, abs=dt / 2)


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
