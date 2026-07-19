import math
from dataclasses import replace

import pytest

from app.control.controller import PatrolGains
from app.sysid.identify import AxisModel, PlantModel
from app.sysid.sim_plant import SimClock, SimulatedVRChat
from app.sysid.worldcal import (
    CAL_AXES,
    MOVE_DEADBAND_CMD,
    REF_DEADTIME_S,
    REF_SPEED,
    ScaleEstimate,
    WorldCalibration,
    check_deadtime,
    estimate_scale,
    probe_axis_speed,
    run_world_calibration,
    scale_gains,
)

DT = 0.017  # フレーム間隔(実機 ~17ms 相当)


# ---------------------------------------------------------------------------
# 合成プラント / プローブ結果のヘルパ
# ---------------------------------------------------------------------------


def _probe_model(
    axis: str, r_plus: float, r_minus: float, *, deadtime: float = 0.017
) -> AxisModel:
    """cmd=±1.0(と 0)だけを持つプローブ同定結果風の AxisModel。"""
    return AxisModel(
        axis, "m/s", [(-1.0, r_minus), (0.0, 0.0), (1.0, r_plus)], deadtime
    )


def _deadband_curve(top: float, deadband: float = MOVE_DEADBAND_CMD):
    """不感帯まで 0、その上は (1.0, top) まで線形の (cmd, rate) 折れ線(奇対称)。"""

    def rate(c: float) -> float:
        a = abs(c)
        if a <= deadband:
            return 0.0
        return math.copysign((a - deadband) / (1.0 - deadband) * top, c)

    cmds = [-1.0, -0.5, -0.15, -0.10, 0.0, 0.10, 0.15, 0.5, 1.0]
    return [(c, rate(c)) for c in cmds]


def _world(s_forward: float, s_strafe: float) -> PlantModel:
    """移動 2 軸が「不感帯 + 線形リマップ」で傾きだけ違うワールド。

    cmd=1.0 の速度は REF_SPEED[axis] * s。SimulatedVRChat は視点軸を参照しないので
    (move プローブは look を打たない)forward/strafe だけで足りる。
    """
    return PlantModel(
        axes={
            "forward": AxisModel(
                "forward",
                "m/s",
                _deadband_curve(REF_SPEED["forward"] * s_forward),
                REF_DEADTIME_S["forward"],
            ),
            "strafe": AxisModel(
                "strafe",
                "m/s",
                _deadband_curve(REF_SPEED["strafe"] * s_strafe),
                REF_DEADTIME_S["strafe"],
            ),
        },
        dt_mean=DT,
    )


def _sim_send(sim: SimulatedVRChat):
    return lambda axis, v: sim.move(**{axis: v})


# ===========================================================================
# estimate_scale: 倍率 = 中央値(rate/(cmd*ref)) over |cmd|>=0.99
# ===========================================================================


@pytest.mark.parametrize(
    "axis,s", [("forward", 2.5), ("forward", 1.0), ("strafe", 0.33), ("strafe", 1.7)]
)
def test_estimate_scale_exact_recovery_for_scaled_linear_plant(axis, s):
    ref = REF_SPEED[axis]
    est = estimate_scale(_probe_model(axis, ref * s, -ref * s))
    assert est.usable
    assert est.reason == ""
    assert est.scale == pytest.approx(s)
    assert est.rates == [(-1.0, -ref * s), (1.0, ref * s)]


def test_estimate_scale_uses_default_ref_speed_by_axis():
    for axis in CAL_AXES:
        ref = REF_SPEED[axis]
        est = estimate_scale(_probe_model(axis, ref * 1.4, -ref * 1.4))
        assert est.scale == pytest.approx(1.4)
    # 同じ実測速度でも軸の基準速度が違えば倍率は変わる
    e_f = estimate_scale(_probe_model("forward", 3.0, -3.0))  # ref 6.0 -> 0.5
    e_s = estimate_scale(_probe_model("strafe", 3.0, -3.0))  # ref 3.0 -> 1.0
    assert e_f.scale == pytest.approx(0.5)
    assert e_s.scale == pytest.approx(1.0)


def test_estimate_scale_ref_speed_override():
    m = _probe_model("forward", 8.0, -8.0)
    assert estimate_scale(m, ref_speed=8.0).scale == pytest.approx(1.0)
    assert estimate_scale(m, ref_speed=4.0).scale == pytest.approx(2.0)


def test_estimate_scale_unusable_without_full_command_point():
    m = AxisModel("forward", "m/s", [(-0.5, -3.0), (0.0, 0.0), (0.5, 3.0)], 0.03)
    est = estimate_scale(m)
    assert not est.usable
    assert "no full-command" in est.reason


def test_estimate_scale_unusable_on_reversed_response():
    ref = REF_SPEED["forward"]
    est = estimate_scale(_probe_model("forward", -ref, ref))
    assert not est.usable
    assert "non-positive" in est.reason


def test_estimate_scale_immobilized_below_s_min():
    ref = REF_SPEED["forward"]
    est = estimate_scale(_probe_model("forward", ref * 0.01, -ref * 0.01))
    assert not est.usable
    assert est.scale < 0.05
    assert "immobiliz" in est.reason.lower() or "locomotion" in est.reason
    # s_min を下げれば通る
    ok = estimate_scale(_probe_model("forward", ref * 0.01, -ref * 0.01), s_min=0.005)
    assert ok.usable
    assert ok.scale == pytest.approx(0.01)


def test_estimate_scale_flags_disagreement_between_signs():
    ref = REF_SPEED["forward"]
    est = estimate_scale(_probe_model("forward", ref * 1.0, -ref * 0.5))
    assert not est.usable
    assert "disagree" in est.reason
    # 中央値は参考値として残る
    assert est.scale == pytest.approx(0.75)


def test_estimate_scale_tolerates_probe_noise():
    """真のプラントは対称。数%の食い違いは計測ノイズなのでゲートを踏ませない。"""
    est = estimate_scale(_probe_model("forward", 6.000, -6.238))  # 4% 食い違い
    assert est.usable
    assert est.reason == ""
    assert est.scale == pytest.approx(1.02, abs=0.01)


def test_estimate_scale_agree_tol_is_configurable():
    ref = REF_SPEED["forward"]
    m = _probe_model("forward", ref * 1.0, -ref * 1.1)  # 10% 非対称
    assert estimate_scale(m).usable  # 既定 0.2 では通る
    assert not estimate_scale(m, agree_tol=0.05).usable


# ===========================================================================
# probe_axis_speed + run_world_calibration: SimulatedVRChat 上の end-to-end
# ===========================================================================


def test_probe_axis_speed_recovers_rate_and_deadtime():
    world = _world(1.0, 1.0)
    sim = SimulatedVRChat(world)
    clock = SimClock(sim)
    model, run = probe_axis_speed(
        sim,
        lambda v: sim.move(forward=v),
        axis="forward",
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    est = estimate_scale(model)
    assert est.usable
    assert est.scale == pytest.approx(1.0, rel=0.05)
    assert 0.0 < est.deadtime_s < 0.10  # 真値 17ms をフレーム量子化込みで復元
    assert clock.t < 30.0  # プローブは軸あたり数秒


@pytest.mark.parametrize(
    "s_f,s_s",
    [(0.33, 0.33), (1.0, 1.0), (2.5, 2.5), (0.33, 2.5), (2.5, 0.33)],
)
def test_run_world_calibration_recovers_both_axes_within_5pct(s_f, s_s):
    """不感帯+線形リマップのスケールワールドから倍率を 5% 以内で復元する(非対称含む)。"""
    world = _world(s_f, s_s)
    sim = SimulatedVRChat(world)
    clock = SimClock(sim)
    cal = run_world_calibration(
        sim,
        _sim_send(sim),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        meta={"case": "synthetic"},
    )
    assert set(cal.axes) == set(CAL_AXES)
    for axis, s in (("forward", s_f), ("strafe", s_s)):
        est = cal.axes[axis]
        assert est.usable, f"{axis}: {est.reason}"
        assert est.scale == pytest.approx(s, rel=0.05)
    assert cal.meta == {"case": "synthetic"}
    out = cal.apply(PatrolGains())
    assert out.s_forward == pytest.approx(s_f, rel=0.05)
    assert out.s_strafe == pytest.approx(s_s, rel=0.05)


def test_run_world_calibration_near_zero_is_unusable_and_apply_raises():
    world = _world(0.01, 0.01)
    sim = SimulatedVRChat(world)
    clock = SimClock(sim)
    cal = run_world_calibration(
        sim, _sim_send(sim), monotonic=clock.monotonic, sleep=clock.sleep
    )
    assert not cal.axes["forward"].usable
    assert not cal.axes["strafe"].usable
    with pytest.raises(ValueError, match="unusable"):
        cal.apply(PatrolGains())


def test_run_world_calibration_ref_speed_override():
    world = _world(1.0, 1.0)  # 実測は REF_SPEED ちょうど
    sim = SimulatedVRChat(world)
    clock = SimClock(sim)
    cal = run_world_calibration(
        sim,
        _sim_send(sim),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        ref_speed={
            "forward": REF_SPEED["forward"] / 2.0,
            "strafe": REF_SPEED["strafe"] / 2.0,
        },
    )
    assert cal.axes["forward"].scale == pytest.approx(2.0, rel=0.05)
    assert cal.axes["strafe"].scale == pytest.approx(2.0, rel=0.05)


# ===========================================================================
# scale_gains(移動系ゲインの 1/s 再スケール、視点系は不変)
# ===========================================================================


def test_scale_gains_halves_kp_for_2x_world():
    base = PatrolGains()
    out = scale_gains(base, 2.0, 2.0)
    assert out.gains.fwd_kp == pytest.approx(base.fwd_kp / 2.0)
    assert out.gains.strafe_kp == pytest.approx(base.strafe_kp / 2.0)
    assert out.gains.strafe_ki == pytest.approx(base.strafe_ki / 2.0)
    assert out.gains.translate_kp == pytest.approx(base.translate_kp / 2.0)
    # 視点系はワールド不変
    assert out.gains.turn_kp == base.turn_kp
    assert out.gains.nav_turn_kp == base.nav_turn_kp
    assert out.gains.pitch_kp == base.pitch_kp


def test_scale_gains_scales_every_move_pid_term():
    """既定で 0 の項も含めて移動系の kp/ki/kd は全て 1/s 倍する。"""
    base = replace(PatrolGains(), translate_ki=0.4, translate_kd=0.2, fwd_kd=0.05)
    out = scale_gains(base, 2.0, 2.0)
    for name in (
        "fwd_kp",
        "fwd_kd",
        "translate_kp",
        "translate_ki",
        "translate_kd",
        "strafe_kp",
        "strafe_ki",
        "strafe_kd",
    ):
        assert getattr(out.gains, name) == pytest.approx(getattr(base, name) / 2.0), (
            name
        )


def _cruise_speed(cmd: float, s: float, ref: float) -> float:
    """不感帯+線形プラントでの実巡航速度[m/s]。"""
    if abs(cmd) <= MOVE_DEADBAND_CMD:
        return 0.0
    return (abs(cmd) - MOVE_DEADBAND_CMD) / (1.0 - MOVE_DEADBAND_CMD) * ref * s


@pytest.mark.parametrize("s", [1.5, 2.0, 5.0, 9.0, 20.0])
def test_scale_gains_cruise_cap_holds_real_speed_at_tuned_level(s):
    """巡航指令は cmd 領域。不感帯があるので 1/s 倍では実巡航が保たれない。"""
    base = PatrolGains()
    ref = REF_SPEED["forward"]
    tuned = _cruise_speed(base.speed, 1.0, ref)
    out = scale_gains(base, s, s)
    assert _cruise_speed(out.gains.speed, s, ref) == pytest.approx(tuned)


@pytest.mark.parametrize("s", [9.0, 20.0, 100.0])
def test_scale_gains_cruise_cap_never_sinks_into_the_deadband(s):
    """速いワールドほど巡航指令は下がるが、不感帯に埋まったら動けなくなる。"""
    out = scale_gains(PatrolGains(), s, s)
    assert out.gains.speed > MOVE_DEADBAND_CMD


def test_scale_gains_caps_cruise_only_for_fast_worlds():
    base = PatrolGains()
    fast = scale_gains(base, 2.0, 2.0)
    assert fast.gains.speed < base.speed
    assert any("cruise cap" in n for n in fast.notes)

    slow = scale_gains(base, 0.5, 0.5)
    assert slow.gains.speed == base.speed
    assert not any("cruise cap" in n for n in slow.notes)


def test_scale_gains_cruise_cap_uses_faster_translation_axis():
    base = PatrolGains()
    out = scale_gains(base, 1.5, 1.0)  # forward だけ速い
    assert out.gains.speed == pytest.approx(
        MOVE_DEADBAND_CMD + (base.speed - MOVE_DEADBAND_CMD) / 1.5
    )
    assert out.gains.translate_kp == pytest.approx(base.translate_kp / 1.5)
    assert out.gains.strafe_kp == pytest.approx(base.strafe_kp)  # 自軸倍率 1.0


def test_scale_gains_applies_stall_floor_for_extreme_s():
    base = PatrolGains()
    out = scale_gains(base, 10.0, 10.0)
    fwd_floor = MOVE_DEADBAND_CMD * 1.2 / base.arrive_radius
    translate_floor = MOVE_DEADBAND_CMD * 1.2 * math.sqrt(2.0) / base.arrive_radius
    assert out.gains.fwd_kp == pytest.approx(fwd_floor)
    assert out.gains.translate_kp == pytest.approx(translate_floor)
    assert any("stall floor" in n for n in out.notes)
    assert out.gains.fwd_kp * base.arrive_radius > MOVE_DEADBAND_CMD


def test_scale_gains_no_floor_clamp_at_moderate_speed():
    base = PatrolGains()
    out = scale_gains(base, 2.0, 2.0)
    assert not any("stall floor" in n for n in out.notes)


def test_scale_gains_raises_on_non_positive_scale():
    base = PatrolGains()
    with pytest.raises(ValueError):
        scale_gains(base, 0.0, 1.0)
    with pytest.raises(ValueError):
        scale_gains(base, 1.0, -0.5)


# ===========================================================================
# check_deadtime
# ===========================================================================


def _est(
    axis: str,
    scale: float,
    *,
    usable: bool = True,
    reason: str = "",
    rates=None,
    deadtime_s: float = 0.032,
) -> ScaleEstimate:
    if rates is None:
        ref = REF_SPEED[axis]
        rates = [(-1.0, -scale * ref), (1.0, scale * ref)]
    return ScaleEstimate(
        axis=axis,
        scale=scale,
        rates=rates,
        deadtime_s=deadtime_s,
        usable=usable,
        reason=reason,
    )


def test_check_deadtime_warns_above_2x():
    est = _est("forward", 1.0, deadtime_s=0.10)
    warn = check_deadtime(est, ref_deadtime_s=0.030)
    assert warn is not None
    assert "forward" in warn and "3.3x" in warn


def test_check_deadtime_silent_below_threshold():
    est = _est("forward", 1.0, deadtime_s=0.032)
    assert check_deadtime(est, ref_deadtime_s=0.030) is None


def test_check_deadtime_silent_on_zero_reference():
    est = _est("forward", 1.0, deadtime_s=0.10)
    assert check_deadtime(est, ref_deadtime_s=0.0) is None
    # 実測むだ時間が 0(取れなかった)ときも黙る
    assert (
        check_deadtime(_est("forward", 1.0, deadtime_s=0.0), ref_deadtime_s=0.030)
        is None
    )


def test_check_deadtime_ratio_warn_configurable():
    est = _est("forward", 1.0, deadtime_s=0.05)  # ref の 1.67 倍
    assert check_deadtime(est, ref_deadtime_s=0.030) is None  # 既定 2.0 では黙る
    assert check_deadtime(est, ref_deadtime_s=0.030, ratio_warn=1.5) is not None


# ===========================================================================
# WorldCalibration: save → load ラウンドトリップ
# ===========================================================================


def _cal(**kw) -> WorldCalibration:
    s_f = kw.pop("s_forward", 2.0)
    s_s = kw.pop("s_strafe", 2.0)
    axes = {"forward": _est("forward", s_f), "strafe": _est("strafe", s_s)}
    axes.update(kw.pop("axes", {}))
    return WorldCalibration(axes=axes, **kw)


def test_worldcalibration_roundtrip_preserves_everything(tmp_path):
    cal = WorldCalibration(
        axes={
            "forward": _est(
                "forward",
                2.5,
                rates=[(-1.0, -15.6), (1.0, 15.0)],
                deadtime_s=0.041,
            ),
            "strafe": _est(
                "strafe", 1.9, usable=False, reason="immobilized?", deadtime_s=0.029
            ),
        },
        warnings=["axis forward: probed deadtime 90ms is 2.8x reference"],
        meta={"created": "2026-07-17T10:05:08", "reference": {"forward": 6.0}},
    )
    path = cal.save(tmp_path / "wc.json")
    back = WorldCalibration.load(path)

    assert set(back.axes) == set(cal.axes)
    for name in cal.axes:
        a, b = cal.axes[name], back.axes[name]
        assert b.axis == a.axis
        assert b.scale == pytest.approx(a.scale)
        assert b.deadtime_s == pytest.approx(a.deadtime_s)
        assert b.usable == a.usable
        assert b.reason == a.reason
        # rates は JSON の list ではなく tuple のリストとして復元される
        assert b.rates == a.rates
        assert all(isinstance(t, tuple) and len(t) == 2 for t in b.rates)
    assert back.warnings == cal.warnings
    assert back.meta == cal.meta


# ===========================================================================
# WorldCalibration.apply
# ===========================================================================


def test_apply_scales_gains_for_known_scale():
    base = PatrolGains()
    out = _cal(s_forward=2.0, s_strafe=2.0).apply(base)
    assert out.s_forward == pytest.approx(2.0)
    assert out.s_strafe == pytest.approx(2.0)
    assert out.gains.strafe_kp == pytest.approx(base.strafe_kp / 2.0)
    assert out.gains.fwd_kp == pytest.approx(base.fwd_kp / 2.0)
    assert out.gains.translate_kp == pytest.approx(base.translate_kp / 2.0)
    # 視点系はワールド不変
    assert out.gains.turn_kp == base.turn_kp
    assert out.gains.nav_turn_kp == base.nav_turn_kp
    assert out.gains.pitch_kp == base.pitch_kp


def test_apply_matches_scale_gains_directly():
    """apply の結果は scale_gains(base, s_f, s_s) と同一(直パイプラインとの等価性)。"""
    base = PatrolGains()
    applied = _cal(s_forward=1.5, s_strafe=1.0).apply(base)
    direct = scale_gains(base, 1.5, 1.0)
    assert applied.gains == direct.gains
    # notes 末尾は scale_gains の notes(warnings は前置される)
    if direct.notes:
        assert applied.notes[-len(direct.notes) :] == direct.notes


def test_apply_prepends_warnings_to_notes():
    base = PatrolGains()
    cal = _cal(s_forward=2.0, s_strafe=2.0, warnings=["deadtime warning present"])
    out = cal.apply(base)
    assert "deadtime warning present" in out.notes


def test_apply_raises_when_axis_unusable():
    base = PatrolGains()
    cal = _cal(
        axes={"forward": _est("forward", 0.02, usable=False, reason="immobilized?")}
    )
    with pytest.raises(ValueError, match="unusable"):
        cal.apply(base)


def test_apply_raises_when_axis_missing():
    base = PatrolGains()
    cal = WorldCalibration(axes={"forward": _est("forward", 2.0)})  # strafe 欠落
    with pytest.raises(ValueError, match="missing axes"):
        cal.apply(base)


# ===========================================================================
# Pilot への配線(world_cal=オブジェクト / ファイルパス / None)
# ===========================================================================


def _pilot_stubs():
    import numpy as np

    from app.core.pose import Pose
    from app.mapping.mapper import Bounds
    from app.spatial.navigation import NavGrid

    class FakeReader:
        def get_latest(self):
            return Pose(
                time_ms=1,
                position=(0.5, 1.6, 0.5),
                forward=(0.0, 0.0, 1.0),
                up=(0.0, 1.0, 0.0),
            )

        def stop(self, *a, **k):
            pass

    class RecActuator:
        def look(self, turn=0.0, pitch=0.0):
            pass

        def move(self, forward=0.0, strafe=0.0):
            pass

        def stop(self):
            pass

    grid = NavGrid(
        free=np.ones((10, 10), bool),
        cell=0.1,
        bounds=Bounds(0.0, 1.0, 0.0, 1.0),
    )
    return grid, FakeReader(), RecActuator(), RecActuator()


def test_pilot_applies_world_cal_object():
    from app.control.pilot import Pilot

    base = PatrolGains()
    grid, reader, look, move = _pilot_stubs()
    pilot = Pilot(grid, reader, look, move, world_cal=_cal(s_forward=2.0, s_strafe=2.0))
    assert pilot.gains.strafe_kp == pytest.approx(base.strafe_kp / 2.0)
    assert pilot.gains.fwd_kp == pytest.approx(base.fwd_kp / 2.0)
    assert pilot.gains.translate_kp == pytest.approx(base.translate_kp / 2.0)
    # 視点系は不変
    assert pilot.gains.turn_kp == base.turn_kp
    assert pilot.gains.nav_turn_kp == base.nav_turn_kp
    assert pilot.gains.pitch_kp == base.pitch_kp


def test_pilot_applies_world_cal_from_file_path(tmp_path):
    from app.control.pilot import Pilot

    base = PatrolGains()
    path = _cal(s_forward=2.0, s_strafe=2.0).save(tmp_path / "wc.json")
    grid, reader, look, move = _pilot_stubs()
    pilot = Pilot(grid, reader, look, move, world_cal=str(path))
    assert pilot.gains.strafe_kp == pytest.approx(base.strafe_kp / 2.0)
    assert pilot.gains.turn_kp == base.turn_kp


def test_pilot_no_world_cal_leaves_gains_default():
    from app.control.pilot import Pilot

    grid, reader, look, move = _pilot_stubs()
    pilot = Pilot(grid, reader, look, move)
    assert pilot.gains == PatrolGains()


def test_pilot_raises_on_unusable_world_cal():
    from app.control.pilot import Pilot

    grid, reader, look, move = _pilot_stubs()
    cal = _cal(
        axes={"forward": _est("forward", 0.02, usable=False, reason="immobilized?")}
    )
    with pytest.raises(ValueError, match="unusable"):
        Pilot(grid, reader, look, move, world_cal=cal)
