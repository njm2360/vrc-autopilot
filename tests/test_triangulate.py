"""三角測量(視線レイの最小二乗交点)の合成テスト。"""

from __future__ import annotations

import numpy as np
import pytest

from pose_hud.pose import Pose
from pose_hud.triangulate import (
    Sighting,
    closest_point_to_rays,
    triangulate,
    triangulate_poses,
)


def sighting_towards(origin, target, label=""):
    o = np.asarray(origin, float)
    d = np.asarray(target, float) - o
    return Sighting(origin=tuple(o), direction=tuple(d), label=label)


def test_two_rays_exact_intersection():
    target = np.array([2.0, 3.0, 5.0])
    s1 = sighting_towards((0, 0, 0), target)
    s2 = sighting_towards((3, 0, 0), target)
    res = triangulate([s1, s2])
    np.testing.assert_allclose(res.point, target, atol=1e-6)
    assert res.residual_rms < 1e-6
    assert res.well_conditioned


def test_three_rays_with_noise_are_averaged():
    target = np.array([1.0, 2.0, 4.0])
    origins = [(-3, 0, 0), (3, 0, 1), (0, 0, -4)]
    sightings = []
    for i, o in enumerate(origins):
        d = target - np.asarray(o, float)
        d = d / np.linalg.norm(d)
        d[i % 3] += 0.01  # 各レイに少しだけ向きの誤差
        sightings.append(Sighting(origin=o, direction=tuple(d)))
    res = triangulate(sightings)
    assert res.n == 3
    np.testing.assert_allclose(res.point, target, atol=0.1)
    assert res.residual_rms < 0.1


def test_skew_rays_midpoint_and_residual():
    # x軸方向のレイ(原点)と、(0,0,1)を通るy軸方向のレイ(ねじれの位置)
    s1 = Sighting(origin=(0, 0, 0), direction=(1, 0, 0))
    s2 = Sighting(origin=(0, 0, 1), direction=(0, 1, 0))
    res = triangulate([s1, s2])
    np.testing.assert_allclose(res.point, (0, 0, 0.5), atol=1e-9)
    assert res.residual_rms == pytest.approx(0.5, abs=1e-9)
    assert res.max_pair_angle_deg == pytest.approx(90.0, abs=1e-6)


def test_parallel_rays_flagged_ill_conditioned():
    s1 = Sighting(origin=(0, 0, 0), direction=(1, 0, 0))
    s2 = Sighting(origin=(0, 1, 0), direction=(1, 0, 0))
    res = triangulate([s1, s2])
    assert not res.well_conditioned
    assert res.max_pair_angle_deg == pytest.approx(0.0, abs=1e-6)


def test_small_angle_flagged():
    target = np.array([0.0, 0.0, 10.0])
    # ほぼ同じ方向から狙う2点 -> 角度が小さく不十分
    s1 = sighting_towards((0.0, 0, 0), target)
    s2 = sighting_towards((0.05, 0, 0), target)
    res = triangulate([s1, s2], min_angle_deg=5.0)
    assert res.max_pair_angle_deg < 5.0
    assert not res.well_conditioned


def test_needs_two():
    with pytest.raises(ValueError):
        triangulate([Sighting(origin=(0, 0, 0), direction=(0, 0, 1))])
    with pytest.raises(ValueError):
        closest_point_to_rays(np.zeros((1, 3)), np.array([[0, 0, 1.0]]))


def test_from_pose_uses_forward_ray():
    target = np.array([0.0, 0.0, 5.0])
    # 原点から +Z を向くポーズ
    p1 = Pose(time_ms=1, position=(0, 0, 0), forward=(0, 0, 1.0), up=(0, 1, 0))
    # (2,0,5) から target(=(0,0,5)) を向く => -X 方向
    p2 = Pose(time_ms=2, position=(2, 0, 5), forward=(-1.0, 0, 0), up=(0, 1, 0))
    res = triangulate_poses([p1, p2])
    np.testing.assert_allclose(res.point, target, atol=1e-6)


def test_zero_direction_rejected():
    with pytest.raises(ValueError):
        Sighting(origin=(0, 0, 0), direction=(0, 0, 0)).direction_arr


def test_result_to_dict():
    res = triangulate([
        Sighting(origin=(0, 0, 0), direction=(0, 0, 1)),
        Sighting(origin=(1, 0, 0), direction=(0, 0, 1)),
    ])
    d = res.to_dict()
    assert set(d) >= {"point", "residual_rms_m", "n", "well_conditioned"}
    assert set(d["point"]) == {"x", "y", "z"}
