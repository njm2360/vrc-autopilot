"""照準幾何(guidance)のテスト。"""

from __future__ import annotations

import math

import pytest

from app.control.guidance import (
    aim_angle,
    forward_factor,
    heading_error,
    pitch_error,
    wrap180,
)


def test_wrap180():
    assert wrap180(0) == 0
    assert wrap180(190) == pytest.approx(-170)
    assert wrap180(-190) == pytest.approx(170)
    assert abs(wrap180(180)) == pytest.approx(180)   # 境界は ±180 どちらでも可


def test_heading_error_right_is_positive():
    # +Z を向いて原点。ターゲットが右(+X)→ +90度、距離5
    err, dist = heading_error((0.0, 0.0), 0.0, (5.0, 0.0))
    assert err == pytest.approx(90.0)
    assert dist == pytest.approx(5.0)


def test_heading_error_shortest_way():
    # 現在 yaw=170、目標方向 -170(=190)。最短回転は +20度(逆回りの -340度ではない)
    err, _ = heading_error((0.0, 0.0), 170.0, (math.sin(math.radians(-170)), math.cos(math.radians(-170))))
    assert abs(err) == pytest.approx(20.0, abs=1e-6)


def test_pitch_error_up_positive():
    # 視点(0,1,0)、水平を向いている。ボタンが上(y=3, 前方z=5)→ もっと上を向く必要=+
    eye = (0.0, 1.0, 0.0)
    forward = (0.0, 0.0, 1.0)             # pitch 0
    target = (0.0, 3.0, 5.0)
    err = pitch_error(eye, forward, target)
    assert err > 0
    assert err == pytest.approx(math.degrees(math.atan2(2.0, 5.0)), abs=1e-6)


def test_pitch_error_zero_when_aligned():
    eye = (0.0, 1.0, 0.0)
    target = (0.0, 1.0, 5.0)              # 同じ高さ真正面
    forward = (0.0, 0.0, 1.0)
    assert pitch_error(eye, forward, target) == pytest.approx(0.0, abs=1e-9)


def test_aim_angle_zero_when_forward_hits_target():
    eye = (0.0, 1.0, 0.0)
    target = (3.0, 2.0, 4.0)
    d = (3.0, 1.0, 4.0)
    n = math.sqrt(9 + 1 + 16)
    forward = (d[0] / n, d[1] / n, d[2] / n)
    assert aim_angle(eye, forward, target) == pytest.approx(0.0, abs=1e-3)


def test_aim_angle_from_pose():
    # forward が目標方向から 90度ずれている場合
    eye = (0.0, 1.0, 0.0)
    target = (0.0, 1.0, 5.0)             # 真正面(+Z)
    forward = (1.0, 0.0, 0.0)            # +X を向いている → 90度ずれ
    assert aim_angle(eye, forward, target) == pytest.approx(90.0, abs=1e-6)


def test_forward_factor_smooth_decay():
    assert forward_factor(0.0) == pytest.approx(1.0)
    assert forward_factor(60.0) == pytest.approx(0.5, abs=1e-6)
    assert forward_factor(90.0) == 0.0
    assert forward_factor(120.0) == 0.0
    # 単調減少(その場停止のような不連続がない)
    vals = [forward_factor(a) for a in range(0, 95, 5)]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
