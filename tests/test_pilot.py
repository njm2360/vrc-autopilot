"""Pilot 誘導エンジンのテスト。

実機(VRChat/OSC/キャプチャ)なしで、注入した fake reader・記録アクチュエータだけで
建物ブロック(follow_path / aim_at)と Pilot の分岐を検証する。入出力が切れていて
ヘッドレスでテストできることの担保でもある。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pose_hud.controller import PatrolGains, face_controllers, nav_controllers
from pose_hud.pose import Pose
from pose_hud.mapping import Bounds
from pose_hud.navigation import NavGrid
from pose_hud.maneuvers import aim_at, follow_path, turn_to
from pose_hud.pilot import Pilot
from pose_hud.telemetry import AxisMetrics


def _pose(t: int, pos, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> Pose:
    """+Z 基準 yaw/pitch から forward を組んだ Pose。"""
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    cp = math.cos(p)
    fwd = (cp * math.sin(y), math.sin(p), cp * math.cos(y))
    return Pose(time_ms=t, position=tuple(pos), forward=fwd, up=(0.0, 1.0, 0.0))


class FakeReader:
    """スクリプトしたポーズ列を time_ms 昇順で返す(尽きたら最後を保持)。"""

    def __init__(self, poses):
        self._poses = list(poses)
        self._i = 0
        self.stopped = False

    def get_latest(self):
        if not self._poses:
            return None
        pose = self._poses[min(self._i, len(self._poses) - 1)]
        self._i += 1
        return pose

    def stop(self, *a, **k):
        self.stopped = True


class RecActuator:
    """look/move/stop の呼び出しを記録するアクチュエータ(LookActuator+MoveActuator)。"""

    def __init__(self):
        self.looks = []
        self.moves = []
        self.stops = 0

    def look(self, turn: float = 0.0, pitch: float = 0.0):
        self.looks.append((turn, pitch))

    def move(self, forward: float = 0.0, strafe: float = 0.0):
        self.moves.append((forward, strafe))

    def stop(self):
        self.stops += 1


class ListRec:
    def __init__(self):
        self.rows = []

    def row(self, **kw):
        self.rows.append(kw)


def _gains(**kw):
    return PatrolGains(**kw)


# ---- follow_path(体だけ誘導) ------------------------------------------
def test_follow_path_arrives_when_already_at_goal():
    g = _gains()
    reader = FakeReader([_pose(1, (0.0, 1.6, 2.0))])  # 最終WPの上に立っている
    look, move = RecActuator(), RecActuator()
    res = follow_path(
        reader, look, move, [(0.0, 0.0), (0.0, 2.0)], g, nav_controllers(g)
    )
    assert res.arrived and res.reason == "arrived"
    assert move.stops == 1 and look.stops == 1  # 終了時に必ず停止


def test_follow_path_scripted_walk_records_and_commands():
    g = _gains(arrive=0.35)
    # +Z へ直進する体を再現(最後は最終WP上=到達)
    poses = [_pose(i + 1, (0.0, 1.6, z)) for i, z in enumerate([0.0, 0.5, 1.0, 1.5, 2.0])]
    reader = FakeReader(poses)
    look, move = RecActuator(), RecActuator()
    rec = ListRec()
    res = follow_path(
        reader, look, move, [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)],
        g, nav_controllers(g), recorder=rec, name="walk",
    )
    assert res.arrived
    assert res.frames >= 4
    assert len(rec.rows) >= 3  # フレームごとに記録
    assert any(fwd > 0.0 for fwd, _ in move.moves)  # 正対中は前進指令が出る
    assert all(row["phase"] == "nav" for row in rec.rows)


def test_follow_path_empty_waypoints_is_noop_arrived():
    g = _gains()
    reader = FakeReader([_pose(1, (0.0, 1.6, 0.0))])
    look, move = RecActuator(), RecActuator()
    res = follow_path(reader, look, move, [], g, nav_controllers(g))
    assert res.arrived and res.frames == 0


# ---- aim_at(視点合わせだけ) --------------------------------------------
def test_aim_at_converges_when_aligned():
    g = _gains(settle=3, face_tol=1.0)
    # 既に真正面(+Z)を向いてターゲットも +Z・同じ高さ → yaw/pitch 誤差 0
    aligned = [_pose(i + 1, (0.0, 1.0, 0.0)) for i in range(5)]
    reader = FakeReader(aligned)
    look = RecActuator()
    res = aim_at(reader, look, (0.0, 1.0, 5.0), g, face_controllers(g))
    assert res.converged
    assert abs(res.yaw_err) < g.face_tol and abs(res.pitch_err) < g.face_tol
    assert look.stops == 1


# ---- turn_to(指定方向を向く) -------------------------------------------
def test_turn_to_yaw_only_converges():
    g = _gains(settle=3, face_tol=1.0)
    # 既に yaw=90° を向いている → 目標 90° に対し誤差0(pitch は無視)
    reader = FakeReader([_pose(i + 1, (0.0, 1.0, 0.0), yaw_deg=90.0) for i in range(5)])
    look = RecActuator()
    res = turn_to(reader, look, 90.0, g, face_controllers(g))
    assert res.converged and abs(res.yaw_err) < g.face_tol
    assert res.pitch_err == 0.0  # pitch_deg=None なので pitch は制御されない


def test_turn_to_yaw_and_pitch_converges():
    g = _gains(settle=3, face_tol=1.0)
    # yaw=45°, pitch=-20° を向いている → 同じ目標角なら両方誤差0で収束
    reader = FakeReader(
        [_pose(i + 1, (0.0, 1.0, 0.0), yaw_deg=45.0, pitch_deg=-20.0) for i in range(5)]
    )
    look = RecActuator()
    res = turn_to(reader, look, 45.0, g, face_controllers(g), pitch_deg=-20.0)
    assert res.converged
    assert abs(res.yaw_err) < g.face_tol and abs(res.pitch_err) < g.face_tol


def test_turn_to_pitch_none_leaves_pitch_metrics_none():
    g = _gains(settle=3, face_tol=1.0)
    reader = FakeReader([_pose(i + 1, (0.0, 1.0, 0.0), yaw_deg=90.0) for i in range(5)])
    # recorder を渡した時だけ指標が付く
    res = turn_to(reader, RecActuator(), 90.0, g, face_controllers(g), recorder=ListRec())
    assert isinstance(res.yaw, AxisMetrics)
    assert res.pitch is None  # pitch 未制御なら指標も None


# ---- チューニング指標(AxisMetrics) -----------------------------------
def test_metrics_only_when_recorder_attached():
    # recorder なし(=非ログ・非チューニング)なら指標計算は入らず None
    g = _gains(settle=3, face_tol=1.0)
    reader = FakeReader([_pose(i + 1, (0.0, 1.0, 0.0), yaw_deg=90.0) for i in range(5)])
    res = turn_to(reader, RecActuator(), 90.0, g, face_controllers(g))  # recorder 省略
    assert res.converged  # 制御・収束判定は従来どおり動く
    assert res.yaw is None and res.pitch is None  # 指標は積まれない


def test_metrics_populated_and_scorable():
    g = _gains(settle=3, face_tol=1.0)
    # yaw を +30°→0° へ寄せる(誤差が減っていく)スクリプト。整定して収束
    yaws = [30.0, 18.0, 8.0, 2.0, 0.3, 0.2, 0.1]
    reader = FakeReader([_pose(i + 1, (0.0, 1.0, 0.0), yaw_deg=y) for i, y in enumerate(yaws)])
    res = turn_to(reader, RecActuator(), 0.0, g, face_controllers(g), recorder=ListRec())
    m = res.yaw
    assert isinstance(m, AxisMetrics)
    assert m.iae > 0.0 and m.itae >= 0.0  # 誤差の積分が貯まる
    assert m.peak_err >= 29.0  # 初期誤差 30° 近辺がピーク
    assert m.settle_time is not None  # tol 未満に整定した時刻が取れる
    # 目的関数がそのまま組める(スカラー化できること)
    score = m.itae + 0.1 * m.effort
    assert score >= 0.0


def test_follow_path_exposes_yaw_metrics():
    g = _gains(arrive=0.35)
    poses = [_pose(i + 1, (0.0, 1.6, z)) for i, z in enumerate([0.0, 0.5, 1.0, 1.5, 2.0])]
    res = follow_path(
        FakeReader(poses), RecActuator(), RecActuator(),
        [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)], g, nav_controllers(g),
        recorder=ListRec(),
    )
    assert isinstance(res.yaw, AxisMetrics)
    assert res.yaw.effort >= 0.0


def test_turn_to_pitch_error_uses_target_angle():
    g = _gains(settle=1, face_tol=1.0)
    # yaw 合致・pitch は現在0°で目標+30° → pitch 誤差 ≈ +30(上を向く必要)
    reader = FakeReader([_pose(1, (0.0, 1.0, 0.0), yaw_deg=0.0, pitch_deg=0.0)])
    look = RecActuator()
    res = turn_to(reader, look, 0.0, g, face_controllers(g), pitch_deg=30.0)
    assert res.pitch_err == pytest.approx(30.0, abs=1e-6)


# ---- Pilot(高レベル・注入) --------------------------------------------
def _grid(free: np.ndarray) -> NavGrid:
    return NavGrid(free=free, cell=0.1, bounds=Bounds(0.0, 1.0, 0.0, 1.0))


def test_pilot_goto_no_pose():
    # HUD からポーズが取れない → no_pose で即返る(実機不要)
    pilot = Pilot(_grid(np.ones((10, 10), bool)), FakeReader([]), RecActuator(), RecActuator())
    res = pilot.goto((0.5, 0.5))
    assert not res.reached and res.reason == "no_pose"


def test_pilot_goto_unreachable():
    # free が全 False → 経路なし
    pilot = Pilot(
        _grid(np.zeros((10, 10), bool)),
        FakeReader([_pose(1, (0.5, 1.6, 0.5))]),
        RecActuator(), RecActuator(),
    )
    res = pilot.goto((0.9, 0.9))
    assert not res.reached and res.reason == "unreachable"


def test_pilot_is_usable_without_hardware_imports():
    # capture/osc を import せず、注入だけで Pilot が動く
    pilot = Pilot(
        _grid(np.ones((10, 10), bool)),
        FakeReader([_pose(1, (0.5, 1.6, 0.5))]),
        RecActuator(), RecActuator(),
    )
    assert pilot._owns_io is False
    res = pilot.goto((0.5, 0.5))  # start≈goal なのですぐ到達
    assert res.reached
