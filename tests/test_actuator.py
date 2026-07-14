"""アクチュエータ(look/move IF)と制御器ユニットのテスト。"""

from __future__ import annotations

import pytest

from pose_hud.actuator import LookActuator, MoveActuator, MouseLookActuator
from pose_hud.pid import PID
from pose_hud.controller import (
    AxisController,
    PatrolGains,
    face_controllers,
    nav_controllers,
)
from pose_hud.osc import VRChatOSC


def test_vrchat_osc_satisfies_both_actuator_protocols():
    # VRChatOSC は look/move/stop を備えるので両プロトコルを構造的に満たす。
    osc = VRChatOSC()
    assert isinstance(osc, LookActuator)
    assert isinstance(osc, MoveActuator)


def test_mouse_look_maps_command_to_relative_pixels():
    calls = []
    act = MouseLookActuator(
        yaw_gain=100.0, pitch_gain=50.0, invert_pitch=True,
        move_rel=lambda dx, dy: calls.append((dx, dy)),
    )
    act.look(turn=0.5, pitch=0.2)          # dx=50, dy=-(0.2*50)=-10(上は画面上=負)
    assert calls == [(50, -10)]


def test_mouse_look_skips_zero_motion():
    calls = []
    act = MouseLookActuator(move_rel=lambda dx, dy: calls.append((dx, dy)))
    act.look(0.0, 0.0)                     # 丸めて0なら送らない
    act.look(0.001, 0.0)                   # 0.001*40≈0 → 送らない
    assert calls == []


def test_mouse_look_pitch_not_inverted():
    calls = []
    act = MouseLookActuator(pitch_gain=100.0, invert_pitch=False,
                            move_rel=lambda dx, dy: calls.append((dx, dy)))
    act.look(0.0, 0.3)
    assert calls == [(0, 30)]


def test_axis_controller_gates_within_tol():
    ctl = AxisController(PID(kp=1.0), tol=2.0)
    assert ctl.update(1.0, 0.1) == 0.0     # |1|<2 → 指令0(PIDは進めない)
    out = ctl.update(5.0, 0.1)             # |5|>2 → 通常 PID
    assert out == pytest.approx(1.0)       # kp*5 クランプ


def test_axis_controller_passes_through_pid_logging():
    ctl = AxisController(PID(kp=0.5))
    ctl.update(1.0, 0.1)
    assert ctl.last_p == pytest.approx(0.5)


def test_nav_and_face_controllers_from_gains():
    g = PatrolGains(speed=0.6, face_tol=1.5, turn_deadzone=0.55)
    nav = nav_controllers(g)
    face = face_controllers(g)
    # 前進制御器は速度上限が speed
    assert nav.forward.pid.out_max == pytest.approx(0.6)
    # 正対 yaw は tol=face_tol・不感帯補償つき
    assert face.yaw.tol == pytest.approx(1.5)
    assert face.yaw.pid.out_deadzone == pytest.approx(0.55)
    # 移動 yaw は不感帯補償なし・指令0にする範囲なし
    assert nav.yaw.pid.out_deadzone == 0.0
    assert nav.yaw.tol == 0.0
