"""適応プローブ(折れ点二分探索+むだ時間バースト)とプチフリ耐性のテスト。

既知の合成プラント(SimulatedVRChat)に模擬クロックでかけ、折れ点・傾き・
むだ時間分布が復元されること、フリーズ(dt スパイク)混入時に測り直しと
同定側の除外が効くことを実機なしで検証する。
"""

import pytest

from vrc_autopilot.sysid.adaptive import (
    AdaptiveConfig,
    _clean_levels,
    probe_axis_adaptive,
)
from vrc_autopilot.sysid.identify import (
    AxisModel,
    PlantModel,
    ProbeRun,
    ProbeSample,
    build_plant,
    deadtime_samples,
    freeze_gap,
    identify_axis,
)
from vrc_autopilot.sysid.sim_plant import SimClock, SimulatedVRChat

# 不感帯 0.55 + 線形の視点軸ふう静特性(0.55 以下にごく遅いクロールあり)
YAW_CURVE = [
    (-1.0, -90.0),
    (-0.6, -25.0),
    (-0.55, -1.2),
    (0.0, 0.0),
    (0.55, 1.2),
    (0.6, 25.0),
    (1.0, 90.0),
]
MOVE_CURVE = [(-1.0, -2.0), (0.0, 0.0), (1.0, 2.0)]  # 不感帯なし
# 不感帯 0.10 + 線形の pitch(実機に近い速さ。クランプ ±90° に届きうる)
PITCH_CURVE = [(-1.0, -180.0), (-0.1, 0.0), (0.0, 0.0), (0.1, 0.0), (1.0, 180.0)]
DEAD = 0.1
DT = 0.05


def make_plant(dt_seq: list[float] | None = None) -> PlantModel:
    return PlantModel(
        axes={
            "yaw": AxisModel("yaw", "deg/s", YAW_CURVE, DEAD),
            "pitch": AxisModel("pitch", "deg/s", MOVE_CURVE, DEAD),
            "forward": AxisModel("forward", "m/s", MOVE_CURVE, DEAD),
            "strafe": AxisModel("strafe", "m/s", MOVE_CURVE, DEAD),
        },
        dt_mean=DT,
        dt_seq=dt_seq or [],
    )


def run_adaptive(plant: PlantModel, axis: str, **cfg_kw):
    sim = SimulatedVRChat(plant, use_dt_seq=bool(plant.dt_seq))
    clk = SimClock(sim)
    send = {
        "yaw": lambda v: sim.look(turn=v),
        "pitch": lambda v: sim.look(pitch=v),
        "forward": lambda v: sim.move(forward=v),
        "strafe": lambda v: sim.move(strafe=v),
    }[axis]
    return probe_axis_adaptive(
        sim,
        send,
        axis,
        monotonic=clk.monotonic,
        sleep=clk.sleep,
        cfg=AdaptiveConfig(**cfg_kw),
    )


def test_adaptive_finds_deadband_onset_and_slope():
    res = run_adaptive(make_plant(), "yaw")
    # 折れ点(0.55)が決め打ちなしで見つかる(逆外挿で真値近傍)
    assert 0.50 <= res.model.onset <= 0.57
    assert res.model.rate(1.0) == pytest.approx(90.0, rel=0.1)
    assert abs(res.model.rate(0.3)) < 3.0  # 不感帯内はほぼゼロ
    assert res.slope > 100.0


def test_adaptive_no_deadband_axis():
    res = run_adaptive(make_plant(), "forward")
    # 最小レベルの応答が線形外挿並み → 二分探索を省略して不感帯なし
    assert res.model.onset < 0.05
    assert res.model.rate(1.0) == pytest.approx(2.0, rel=0.1)


def test_adaptive_move_probe_stays_anchored():
    """ミニランを重ねてもホームが漂流せず、移動範囲が ±max_travel+マージンに収まる。"""
    plant = make_plant()
    sim = SimulatedVRChat(plant)
    clk = SimClock(sim)
    res = probe_axis_adaptive(
        sim,
        lambda v: sim.move(forward=v),
        "forward",
        monotonic=clk.monotonic,
        sleep=clk.sleep,
        cfg=AdaptiveConfig(max_travel=0.5, burst_n=10),
    )
    zs = [s.z for s in res.run.samples]
    margin = 2.0 * (DEAD + 2 * DT)  # 最高速度×(むだ時間+2フレーム)
    assert max(abs(z) for z in zs) <= 0.5 + margin
    assert abs(sim.get_latest().position[2]) < 0.08 + margin


def test_adaptive_deadtime_distribution():
    res = run_adaptive(make_plant(), "yaw", burst_n=20)
    st = res.deadtime_stats
    assert st["n"] >= 20
    # 真値 0.1s をフレーム量子化+ポーリングの範囲で復元
    assert st["median"] == pytest.approx(DEAD, abs=0.02)
    assert st["std"] < 0.02
    assert st["dropped"] == 0


def test_adaptive_no_wasteful_retry_under_sparse_freeze():
    # 散発フリーズ(約3秒ごとに0.3s)。各レベルに定常セグメントが残るので測り直さない。
    seq = [0.3 if (i % 60) == 59 else DT for i in range(4000)]
    res = run_adaptive(make_plant(seq), "yaw", burst_n=20)
    assert res.freezes == 0
    assert 0.50 <= res.model.onset <= 0.57
    assert res.model.rate(1.0) == pytest.approx(90.0, rel=0.1)
    assert res.deadtime_stats["median"] == pytest.approx(DEAD, abs=0.02)


def test_adaptive_retries_when_freeze_wipes_a_level():
    # 高頻度フリーズ(3フレームに1回)。レベルが全滅する回が出るので測り直しが走る。
    seq = [0.3 if (i % 3) == 2 else DT for i in range(12000)]
    res = run_adaptive(make_plant(seq), "yaw", burst_n=20)
    assert res.freezes >= 1
    assert 0.50 <= res.model.onset <= 0.57
    assert res.model.rate(1.0) == pytest.approx(90.0, rel=0.1)


def test_pitch_probe_homes_to_level_from_off_level_start():
    # 上を向いた(+35°)状態からでも HUD ホーミングで水平中心に振れ、線形・左右対称に
    # 復元される。むだ時間 15ms なら guard(±70°)のオーバーシュートは ~3° で ±80° に届かない。
    plant = PlantModel(
        axes={"pitch": AxisModel("pitch", "deg/s", PITCH_CURVE, 0.015)},
        dt_mean=0.017,
    )
    sim = SimulatedVRChat(plant, use_dt_seq=False, pitch=35.0)
    clk = SimClock(sim)
    res = probe_axis_adaptive(
        sim,
        lambda v: sim.look(pitch=v),
        "pitch",
        monotonic=clk.monotonic,
        sleep=clk.sleep,
        cfg=AdaptiveConfig(burst_n=12),
    )
    m = res.model
    assert m.onset == pytest.approx(0.10, abs=0.03)
    # 高指令まで線形・左右対称(クランプ張り付きがない)
    assert m.rate(1.0) == pytest.approx(180.0, rel=0.1)
    assert m.rate(-1.0) == pytest.approx(-180.0, rel=0.1)
    assert m.rate(0.8) == pytest.approx(-m.rate(-0.8), rel=0.1)
    # 記録は水平を中心に上下対称に振れている(片側クランプなら偏る)
    pitches = [s.pitch for s in res.run.samples]
    assert max(pitches) > 40.0 and min(pitches) < -40.0
    assert abs(max(pitches) + min(pitches)) < 30.0
    assert max(abs(p) for p in pitches) < 79.5  # クランプに張り付いていない


def test_clean_levels_keeps_partially_frozen_level():
    # 同じ指令レベルにクリーンな定常セグメントとフリーズ入りセグメントが1つずつ。
    # 1つでもクリーンなら測り直し対象にしない。
    dt = 0.017
    samples = _steady_samples(0, 0.5, 0.0, 10.0, 40, dt)
    samples += _steady_samples(1, 0.5, 1.0, 10.0, 40, dt, gap_at=30, gap=0.2)
    run = ProbeRun(
        axis="yaw", samples=samples, seg_starts=[(0, 0.5, 0.0), (1, 0.5, 1.0)]
    )
    assert 0.5 in _clean_levels(run, freeze_gap(run))


def test_clean_levels_flags_fully_wiped_level():
    # あるレベルの唯一の定常セグメントがフリーズ入り → クリーン集合から外れる。
    dt = 0.017
    samples = _steady_samples(0, 0.5, 0.0, 10.0, 40, dt)  # クリーンな 0.5
    samples += _steady_samples(
        1, 0.3, 1.0, 6.0, 40, dt, gap_at=30, gap=0.2
    )  # フリーズ入り 0.3
    run = ProbeRun(
        axis="yaw", samples=samples, seg_starts=[(0, 0.5, 0.0), (1, 0.3, 1.0)]
    )
    clean = _clean_levels(run, freeze_gap(run))
    assert 0.5 in clean
    assert 0.3 not in clean


def test_build_plant_carries_deadtime_stats():
    res = run_adaptive(make_plant(), "yaw")
    plant = build_plant([res.run])
    stats = plant.meta["deadtime_stats"]["yaw"]
    assert stats["median"] == pytest.approx(res.deadtime_stats["median"], abs=0.005)
    assert plant.axes["yaw"].onset == pytest.approx(res.model.onset)


def _steady_samples(seg, cmd, t0, rate, n, dt, gap_at=None, gap=0.0):
    """一定速度 rate で回る yaw サンプル列(gap_at 番目の後に gap 秒の途絶)。"""
    out = []
    t = t0
    for k in range(n):
        out.append(ProbeSample(seg, cmd, t, int(t * 1000), 0, 0, 0, rate * (t - t0), 0))
        t += dt + (gap if gap_at is not None and k == gap_at else 0.0)
    return out


def test_identify_drops_frozen_static_segment():
    """定常窓にフリーズを含むセグメントは静特性から除外される(クリーンな
    同一レベルの再測定だけが残る)。"""
    dt = 0.017
    samples = []
    # seg0: クリーン(rate 10)。seg1: 定常窓内に 0.2s ギャップ+崩れた傾き(rate 30)
    samples += _steady_samples(0, 0.5, 0.0, 10.0, 40, dt)
    samples += _steady_samples(1, 0.5, 1.0, 30.0, 40, dt, gap_at=25, gap=0.2)
    run = ProbeRun(
        axis="yaw", samples=samples, seg_starts=[(0, 0.5, 0.0), (1, 0.5, 1.0)]
    )
    model = identify_axis(run)
    assert dict(model.points)[0.5] == pytest.approx(10.0, rel=0.05)


def test_identify_drops_deadtime_transition_across_freeze():
    """しきい値交差がフリーズを跨いだ遷移はむだ時間の集計から除外される。"""
    dt = 0.017
    true_dead = 0.02
    rate = 30.0
    samples, starts = [], []
    t = 0.0
    seg = 0
    for k in range(4):
        contaminated = k == 1  # 2本目だけ遷移直後に 0.25s の途絶
        # settle(cmd=0、静止)
        for _ in range(12):
            samples.append(ProbeSample(seg, 0.0, t, int(t * 1000), 0, 0, 0, 0.0, 0))
            t += dt
        starts.append((seg, 0.0, t - 12 * dt))
        seg += 1
        # 指令セグメント: true_dead 後に一定速度。汚染版は最初のサンプルが遅れる
        # (フリーズ明けの初サンプルに応答が溜まって乗る)
        t_send = t
        t_k = t_send + (0.25 if contaminated else dt)
        for _ in range(30):
            y = rate * max(0.0, t_k - t_send - true_dead)
            samples.append(ProbeSample(seg, 1.0, t_k, int(t_k * 1000), 0, 0, 0, y, 0))
            t_k += dt
        starts.append((seg, 1.0, t_send))
        seg += 1
        t = t_k
    run = ProbeRun(axis="yaw", samples=samples, seg_starts=starts)
    deads = deadtime_samples(run)
    thr = 2.5 * dt
    dirty = [d for d in deads if d.cross_gap > thr]
    assert len(dirty) == 1  # 汚染遷移が cross_gap で識別できる
    model = identify_axis(run)
    assert model.deadtime_s == pytest.approx(true_dead, abs=dt)
