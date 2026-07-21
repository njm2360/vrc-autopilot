"""Pilot 誘導エンジンのテスト。

実機(VRChat/OSC/キャプチャ)なしで、注入した fake reader・記録アクチュエータだけで
制御ループ部品(follow_path / aim_at)と Pilot の分岐を検証する。実機 I/O から
切り離されていてヘッドレスでテストできることの担保でもある。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.control.controller import (
    PatrolGains,
    face_controllers,
    nav_controllers,
    strafe_controller,
    translate_controllers,
)
from app.control.maneuvers import (
    aim_at,
    follow_path,
    follow_path_translate,
    strafe_align,
    turn_to,
)
from app.control.pilot import Pilot
from app.control.recording import AxisMetrics
from app.core.pose import Pose
from app.mapping.mapper import Bounds
from app.spatial.navigation import NavGrid


def _pose(t: int, pos, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> Pose:
    """+Z 基準 yaw/pitch から forward を組んだ Pose。"""
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    cp = math.cos(p)
    fwd = (cp * math.sin(y), math.sin(p), cp * math.cos(y))
    return Pose(time_ms=t, position=tuple(pos), forward=fwd, up=(0.0, 1.0, 0.0))


class FakeReader:
    """あらかじめ用意したポーズ列を time_ms 昇順で返す(尽きたら最後を保持)。"""

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

    def row(self, row):
        self.rows.append(row)


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
    g = _gains(arrive_radius=0.35)
    # +Z へ進んでいく位置系列(最後は最終WP上=到達)
    poses = [
        _pose(i + 1, (0.0, 1.6, z)) for i, z in enumerate([0.0, 0.5, 1.0, 1.5, 2.0])
    ]
    reader = FakeReader(poses)
    look, move = RecActuator(), RecActuator()
    rec = ListRec()
    res = follow_path(
        reader,
        look,
        move,
        [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)],
        g,
        nav_controllers(g),
        recorder=rec,
        name="walk",
    )
    assert res.arrived
    assert res.frames >= 4
    assert len(rec.rows) >= 3  # フレームごとに記録
    assert any(fwd > 0.0 for fwd, _ in move.moves)  # 正対中は前進指令が出る
    assert all(row.phase == "nav" for row in rec.rows)


def test_follow_path_empty_waypoints_is_noop_arrived():
    g = _gains()
    reader = FakeReader([_pose(1, (0.0, 1.6, 0.0))])
    look, move = RecActuator(), RecActuator()
    res = follow_path(reader, look, move, [], g, nav_controllers(g))
    assert res.arrived and res.frames == 0


def test_follow_path_pitch_free_by_default():
    """pitch_target 未指定なら従来どおり pitch 指令は出さない(pitch=0 のまま)。"""
    g = _gains(arrive_radius=0.35)
    poses = [_pose(i + 1, (0.0, 1.6, z)) for i, z in enumerate([0.0, 0.5, 1.0, 2.0])]
    reader = FakeReader(poses)
    look, move = RecActuator(), RecActuator()
    res = follow_path(
        reader, look, move, [(0.0, 0.0), (0.0, 2.0)], g, nav_controllers(g)
    )
    assert res.arrived
    assert all(pitch == 0.0 for _, pitch in look.looks)  # pitch 軸は触らない
    assert res.pitch is None


def test_follow_path_pitch_target_commands_pitch_up_for_high_target():
    """頭上のターゲットを渡すと、移動中に pitch を上げる指令(+)が出て記録も残る。"""
    g = _gains(arrive_radius=0.35)
    # 眼高 1.6、ずっと水平を向いたまま +Z へ歩く。ターゲットは頭上(y=4.0)。
    poses = [_pose(i + 1, (0.0, 1.6, z)) for i, z in enumerate([0.0, 0.5, 1.0, 2.0])]
    reader = FakeReader(poses)
    look, move = RecActuator(), RecActuator()
    rec = ListRec()
    res = follow_path(
        reader,
        look,
        move,
        [(0.0, 0.0), (0.0, 2.0)],
        g,
        nav_controllers(g),
        pitch_target=(0.0, 4.0, 3.0),
        recorder=rec,
    )
    assert res.arrived
    assert any(pitch > 0.0 for _, pitch in look.looks)  # 上向き指令が出る
    assert res.pitch is not None  # 応答指標が付く
    assert all(row.pitch_err is not None and row.pitch_err > 0.0 for row in rec.rows)


def test_follow_path_translate_pitch_target_commands_pitch_without_yaw():
    """視点固定並進でも pitch だけは先合わせする(yaw 指令は 0 のまま)。"""
    g = _gains(arrive_radius=0.35)
    poses = [_pose(i + 1, (0.0, 1.6, z)) for i, z in enumerate([0.0, 0.5, 1.0, 2.0])]
    reader = FakeReader(poses)
    look, move = RecActuator(), RecActuator()
    rec = ListRec()
    res = follow_path_translate(
        reader,
        look,
        move,
        [(0.0, 0.0), (0.0, 2.0)],
        g,
        translate_controllers(g),
        pitch_target=(0.0, 4.0, 3.0),
        recorder=rec,
    )
    assert all(turn == 0.0 for turn, _ in look.looks)  # yaw は回さない
    assert any(pitch > 0.0 for _, pitch in look.looks)  # pitch は先合わせする
    assert res.pitch is not None  # 記録先があれば応答指標が付く


# ---- aim_at(視点合わせだけ) --------------------------------------------
def test_aim_at_converges_when_aligned():
    g = _gains(settle_frames=3, face_tol=1.0)
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
    g = _gains(settle_frames=3, face_tol=1.0)
    # 既に yaw=90° を向いている → 目標 90° に対し誤差0(pitch は無視)
    reader = FakeReader([_pose(i + 1, (0.0, 1.0, 0.0), yaw_deg=90.0) for i in range(5)])
    look = RecActuator()
    res = turn_to(reader, look, 90.0, g, face_controllers(g))
    assert res.converged and abs(res.yaw_err) < g.face_tol
    assert res.pitch_err == 0.0  # pitch_deg=None なので pitch は制御されない


def test_turn_to_yaw_and_pitch_converges():
    g = _gains(settle_frames=3, face_tol=1.0)
    # yaw=45°, pitch=-20° を向いている → 同じ目標角なら両方誤差0で収束
    reader = FakeReader(
        [_pose(i + 1, (0.0, 1.0, 0.0), yaw_deg=45.0, pitch_deg=-20.0) for i in range(5)]
    )
    look = RecActuator()
    res = turn_to(reader, look, 45.0, g, face_controllers(g), pitch_deg=-20.0)
    assert res.converged
    assert abs(res.yaw_err) < g.face_tol and abs(res.pitch_err) < g.face_tol


def test_turn_to_pitch_none_leaves_pitch_metrics_none():
    g = _gains(settle_frames=3, face_tol=1.0)
    reader = FakeReader([_pose(i + 1, (0.0, 1.0, 0.0), yaw_deg=90.0) for i in range(5)])
    # recorder を渡した時だけ指標が付く
    res = turn_to(
        reader, RecActuator(), 90.0, g, face_controllers(g), recorder=ListRec()
    )
    assert isinstance(res.yaw, AxisMetrics)
    assert res.pitch is None  # pitch 未制御なら指標も None


# ---- strafe_align(横移動による最終照準) --------------------------------
def test_strafe_align_converges_when_on_line():
    # 視線の延長上にターゲット(横ずれ0・pitch 0)→ 即収束
    g = _gains(settle_frames=3, align_tol=0.02)
    poses = [_pose(i + 1, (0.0, 1.0, 0.0)) for i in range(5)]
    look, move = RecActuator(), RecActuator()
    res = strafe_align(
        FakeReader(poses),
        look,
        move,
        (0.0, 1.0, 5.0),
        g,
        face_controllers(g),
        strafe_controller(g),
    )
    assert res.converged and res.reason == "converged"
    assert move.stops == 1 and look.stops == 1  # 終了時に必ず停止


def test_strafe_align_strafes_toward_error_side():
    # 目標が右(+X)にずれている → 横移動指令は正(右)
    g = _gains(settle_frames=3, align_tol=0.02)
    poses = [_pose(i + 1, (0.0, 1.0, 0.0)) for i in range(10)]
    look, move = RecActuator(), RecActuator()
    strafe_align(
        FakeReader(poses),
        look,
        move,
        (0.3, 1.0, 3.0),  # 横ずれ ≈ +0.3m
        g,
        face_controllers(g),
        strafe_controller(g),
    )
    cmds = [s for _f, s in move.moves]
    assert cmds and all(s > 0.0 for s in cmds)
    assert all(t == 0.0 for t, _p in look.looks)  # 視点(yaw)は回さない


def test_strafe_align_stuck_abort():
    # 位置が変わらない(壁に押し付け)のに誤差が残る → stuck で打ち切り
    g = _gains(
        settle_frames=3, align_tol=0.02, align_stuck_time=0.15, align_timeout=5.0
    )

    class FrozenReader:
        """毎回新しい time_ms を返すが位置は動かない(壁押し付けの再現)。"""

        def __init__(self):
            self.t = 0

        def get_latest(self):
            self.t += 1
            return _pose(self.t, (0.0, 1.0, 0.0))

    look, move = RecActuator(), RecActuator()
    res = strafe_align(
        FrozenReader(),
        look,
        move,
        (0.5, 1.0, 3.0),
        g,
        face_controllers(g),
        strafe_controller(g),
    )
    assert not res.converged and res.reason == "stuck"
    assert res.elapsed < 2.0  # align_timeout(5s)よりずっと早く抜ける
    assert move.stops == 1


# ---- follow_path_translate(視点を変えずに並進) -------------------------
def test_hold_view_strafes_to_side_without_turning():
    # 体は +Z を向いたまま(yaw=0)、右(+X)の目標へ。横移動指令が出て視点は一切回さない。
    g = _gains(arrive_radius=0.35)
    # +X へ進んでいく位置系列(最後は最終WP上=到達)
    poses = [
        _pose(i + 1, (x, 1.6, 0.0)) for i, x in enumerate([0.0, 0.5, 1.0, 1.5, 2.0])
    ]
    look, move = RecActuator(), RecActuator()
    res = follow_path_translate(
        FakeReader(poses),
        look,
        move,
        [(0.0, 0.0), (2.0, 0.0)],
        g,
        translate_controllers(g),
        name="m",
    )
    assert res.arrived
    assert move.moves[0][1] > 0.0  # 初手は右へ横移動(目標が右)
    assert all(abs(f) < 1e-9 for f, _s in move.moves)  # 前後成分は出ない(真横)
    assert look.looks and all(t == 0.0 and p == 0.0 for t, p in look.looks)  # 視点固定
    assert move.stops == 1 and look.stops == 1  # 終了時に必ず停止


def test_hold_view_moves_forward_when_target_ahead():
    # 目標が正面(+Z)→ 前進成分が出て横は出ない。視点は回さない。
    g = _gains(arrive_radius=0.35)
    poses = [
        _pose(i + 1, (0.0, 1.6, z)) for i, z in enumerate([0.0, 0.5, 1.0, 1.5, 2.0])
    ]
    look, move = RecActuator(), RecActuator()
    res = follow_path_translate(
        FakeReader(poses),
        look,
        move,
        [(0.0, 0.0), (0.0, 2.0)],
        g,
        translate_controllers(g),
    )
    assert res.arrived
    assert move.moves[0][0] > 0.0  # 初手は前進(目標が正面)
    assert all(abs(s) < 1e-9 for _f, s in move.moves)  # 横成分は出ない
    assert all(t == 0.0 for t, _p in look.looks)  # 視点(yaw)は回さない


def test_pilot_translate_to_reaches_holding_view():
    # Pilot.translate_to は plan_path で壁回避しつつ、視点を変えずに到達する(start≈goal で即到達)
    look, move = RecActuator(), RecActuator()
    pilot = Pilot(
        _grid(np.ones((10, 10), bool)),
        FakeReader([_pose(1, (0.5, 1.6, 0.5))]),
        look,
        move,
    )
    res = pilot.translate_to((0.5, 0.5))
    assert res.path_found and res.arrived
    assert all(t == 0.0 and p == 0.0 for t, p in look.looks)  # 視点固定


# ---- チューニング指標(AxisMetrics) -----------------------------------
def test_metrics_only_when_recorder_attached():
    # recorder なし(=非ログ・非チューニング)なら指標計算は入らず None
    g = _gains(settle_frames=3, face_tol=1.0)
    reader = FakeReader([_pose(i + 1, (0.0, 1.0, 0.0), yaw_deg=90.0) for i in range(5)])
    res = turn_to(reader, RecActuator(), 90.0, g, face_controllers(g))  # recorder 省略
    assert res.converged  # 制御・収束判定は従来どおり動く
    assert res.yaw is None and res.pitch is None  # 指標は積まれない


def test_metrics_populated_and_scorable():
    g = _gains(settle_frames=3, face_tol=1.0)
    # yaw を +30°→0° へ寄せる(誤差が減っていく)ポーズ列。整定して収束する
    yaws = [30.0, 18.0, 8.0, 2.0, 0.3, 0.2, 0.1]
    reader = FakeReader(
        [_pose(i + 1, (0.0, 1.0, 0.0), yaw_deg=y) for i, y in enumerate(yaws)]
    )
    res = turn_to(
        reader, RecActuator(), 0.0, g, face_controllers(g), recorder=ListRec()
    )
    m = res.yaw
    assert isinstance(m, AxisMetrics)
    assert m.iae > 0.0 and m.itae >= 0.0  # 誤差の積分が貯まる
    assert m.peak_err >= 29.0  # 初期誤差 30° 近辺がピーク
    assert m.settle_time is not None  # tol 未満に整定した時刻が取れる
    # 目的関数がそのまま組める(スカラー化できること)
    score = m.itae + 0.1 * m.effort
    assert score >= 0.0


def test_follow_path_exposes_yaw_metrics():
    g = _gains(arrive_radius=0.35)
    poses = [
        _pose(i + 1, (0.0, 1.6, z)) for i, z in enumerate([0.0, 0.5, 1.0, 1.5, 2.0])
    ]
    res = follow_path(
        FakeReader(poses),
        RecActuator(),
        RecActuator(),
        [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)],
        g,
        nav_controllers(g),
        recorder=ListRec(),
    )
    assert isinstance(res.yaw, AxisMetrics)
    assert res.yaw.effort >= 0.0


def test_turn_to_pitch_error_uses_target_angle():
    g = _gains(settle_frames=1, face_tol=1.0)
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
    pilot = Pilot(
        _grid(np.ones((10, 10), bool)), FakeReader([]), RecActuator(), RecActuator()
    )
    res = pilot.goto((0.5, 0.5))
    assert not res.path_found and res.reason == "no_pose"


def test_pilot_goto_unreachable():
    # free が全 False → 経路なし
    pilot = Pilot(
        _grid(np.zeros((10, 10), bool)),
        FakeReader([_pose(1, (0.5, 1.6, 0.5))]),
        RecActuator(),
        RecActuator(),
    )
    res = pilot.goto((0.9, 0.9))
    assert not res.path_found and res.reason == "unreachable"


def test_pilot_standoff_point_is_in_front_of_button():
    # ボタン(0.5,*,0.5)が +X 向き(face_yaw=90°)→ 法線上の正面点 (0.7, 0.5)。
    # 現在地には依存しない(壁裏回り込み防止)
    pilot = Pilot(
        _grid(np.ones((10, 10), bool)), FakeReader([]), RecActuator(), RecActuator()
    )
    assert pilot.standoff_point((0.5, 1.0, 0.5), 90.0, 0.2) == pytest.approx((0.7, 0.5))
    assert pilot.standoff_point((0.5, 1.0, 0.5), 180.0, 0.3) == pytest.approx(
        (0.5, 0.2)
    )  # -Z 向きの正面


def test_pilot_standoff_zero_targets_button_xz():
    pilot = Pilot(
        _grid(np.ones((10, 10), bool)), FakeReader([]), RecActuator(), RecActuator()
    )
    assert pilot.standoff_point((0.9, 1.0, 0.5), 90.0, 0.0) == (0.9, 0.5)


def test_pilot_standoff_default_uses_gains():
    g = _gains(standoff=0.2)
    pilot = Pilot(
        _grid(np.ones((10, 10), bool)),
        FakeReader([]),
        RecActuator(),
        RecActuator(),
        gains=g,
    )
    assert pilot.standoff_point((0.5, 1.0, 0.5), 90.0) == pytest.approx((0.7, 0.5))


class FakeInteract:
    """press/release/click を記録する InteractActuator。"""

    def __init__(self):
        self.presses = 0
        self.releases = 0
        self.clicks = 0

    def press(self):
        self.presses += 1

    def release(self):
        self.releases += 1

    def click(self):
        self.clicks += 1


def _pilot(free=None, poses=(), **kw) -> Pilot:
    free = np.ones((10, 10), bool) if free is None else free
    return Pilot(
        _grid(free), FakeReader(list(poses)), RecActuator(), RecActuator(), **kw
    )


# ---- 状態クエリ ----------------------------------------------------------
def test_pilot_state_queries():
    p = _pilot(poses=[_pose(1, (0.5, 1.6, 0.5), yaw_deg=90.0)])
    assert p.position() == (0.5, 1.6, 0.5)
    assert p.xz() == (0.5, 0.5)
    assert p.yaw() == pytest.approx(90.0)
    assert p.pitch() == pytest.approx(0.0)
    # (x,z) と (x,y,z) のどちらでも受ける
    assert p.distance_to((0.9, 0.5)) == pytest.approx(0.4)
    assert p.distance_to((0.9, 1.0, 0.5)) == pytest.approx(0.4)
    assert p.is_near((0.9, 0.5), 0.5) and not p.is_near((0.9, 0.5), 0.3)
    assert p.bearing_to((0.9, 0.5)) == pytest.approx(90.0)  # +X 方向
    assert p.yaw_error_to((0.9, 0.5)) == pytest.approx(0.0)  # 既に +X を向いている
    assert p.yaw_error_to((0.5, 0.9)) == pytest.approx(-90.0)  # +Z は左90°


def test_pilot_state_queries_no_pose():
    p = _pilot(poses=[])
    assert p.pose() is None and p.position() is None and p.xz() is None
    assert p.yaw() is None and p.distance_to((0.5, 0.5)) is None
    assert not p.is_near((0.5, 0.5), 10.0)
    assert p.bearing_to((0.5, 0.5)) is None and p.yaw_error_to((0.5, 0.5)) is None
    assert p.face_yaw_to((0.5, 0.5)) is None
    assert p.stats() is None  # FakeReader は統計未対応


def test_pilot_face_yaw_to_points_back_at_current_side():
    # 現在地(0.5,0.5)から見てボタン(0.5,*,0.9)の手前側は -Z 方向 → face_yaw=180°
    p = _pilot(poses=[_pose(1, (0.5, 1.6, 0.5))])
    yaw = p.face_yaw_to((0.5, 1.0, 0.9))
    assert abs(yaw) == pytest.approx(180.0)
    # その face_yaw で standoff すると現在地側の正面に立つ
    sx, sz = p.standoff_point((0.5, 1.0, 0.9), yaw, 0.4)
    assert (sx, sz) == pytest.approx((0.5, 0.5))


def test_pilot_pitch_error_to():
    # 同じ高さ正面 → 誤差0。上方 → 正(上を向く必要)
    p = _pilot(poses=[_pose(1, (0.0, 1.0, 0.0))])
    assert p.pitch_error_to((0.0, 1.0, 5.0)) == pytest.approx(0.0)
    assert p.pitch_error_to((0.0, 3.0, 5.0)) > 0.0


# ---- マップ切替 ----------------------------------------------------------
def test_pilot_use_grid_switches_active_map():
    # 全面 free で開始 → can_reach True。全面ブロックの階へ差し替えると False
    p = _pilot(free=np.ones((10, 10), bool), poses=[_pose(1, (0.5, 1.6, 0.5))])
    assert p.can_reach((0.85, 0.85))
    blocked = _grid(np.zeros((10, 10), bool))
    p.use_grid(blocked)
    assert p.grid is blocked
    assert not p.can_reach((0.85, 0.85))  # 差し替え後は新しい階で計画する


# ---- dry-run 経路計画 ----------------------------------------------------
def test_pilot_plan_without_moving():
    p = _pilot(poses=[_pose(1, (0.15, 1.6, 0.15))])
    path = p.plan((0.85, 0.85))
    assert path is not None and path.length > 0.0
    assert p.can_reach((0.85, 0.85))
    assert p.path_length((0.85, 0.85)) == pytest.approx(path.length)


def test_pilot_plan_no_pose_returns_none():
    p = _pilot(poses=[])
    assert p.plan((0.5, 0.5)) is None
    assert not p.can_reach((0.5, 0.5))
    assert p.path_length((0.5, 0.5)) is None


def test_pilot_can_reach_false_on_unreachable_and_out_of_bounds():
    p = _pilot(free=np.zeros((10, 10), bool), poses=[_pose(1, (0.5, 1.6, 0.5))])
    assert not p.can_reach((0.9, 0.9))  # free 無し
    p2 = _pilot(poses=[_pose(1, (0.5, 1.6, 0.5))])
    assert not p2.can_reach((5.0, 5.0))  # マップ範囲外(ValueError を吸収)


def test_pilot_plan_with_explicit_start_needs_no_pose():
    p = _pilot(poses=[])
    assert p.plan((0.85, 0.85), start=(0.15, 0.15)) is not None


# ---- interact(押下) -----------------------------------------------------
def test_pilot_click_delegates_to_interact():
    it = FakeInteract()
    p = _pilot(interact=it)
    p.press()
    p.release()
    p.click()
    assert (it.presses, it.releases, it.clicks) == (1, 1, 1)


def test_pilot_click_without_interact_raises():
    p = _pilot()
    with pytest.raises(RuntimeError):
        p.click()


def test_pilot_click_at_clicks_when_converged():
    # 既に正対(誤差0)→ aim 収束 → click。align は align_tol=0 で無効
    g = _gains(settle_frames=3, face_tol=1.0, align_tol=0.0)
    it = FakeInteract()
    poses = [_pose(i + 1, (0.0, 1.0, 0.0)) for i in range(6)]
    p = _pilot(poses=poses, interact=it, gains=g)
    res = p.click_at((0.0, 1.0, 5.0))
    assert res.clicked and res.reason == "clicked"
    assert it.clicks == 1


def test_pilot_click_at_skips_when_not_converged():
    # yaw が 90° ずれたまま動かない世界 → face_timeout → 収束せず click しない
    g = _gains(settle_frames=3, face_tol=1.0, align_tol=0.0, face_timeout=0.2)
    it = FakeInteract()

    class EndlessMisaligned:
        """毎回新しい time_ms を返すが向きは直らない(90°ずれ固定)。"""

        def __init__(self):
            self.t = 0

        def get_latest(self):
            self.t += 1
            return _pose(self.t, (0.0, 1.0, 0.0), yaw_deg=90.0)

    p = Pilot(
        _grid(np.ones((10, 10), bool)),
        EndlessMisaligned(),
        RecActuator(),
        RecActuator(),
        interact=it,
        gains=g,
    )
    res = p.click_at((0.0, 1.0, 5.0))
    assert not res.clicked and res.reason == "timeout"
    assert it.clicks == 0


def test_pilot_activate_full_sequence():
    # standoff 点(0.5,0.5)=現在地 → 即到達 → 正対済み → click
    g = _gains(settle_frames=3, face_tol=1.0, align_tol=0.0)
    it = FakeInteract()
    poses = [_pose(i + 1, (0.5, 1.6, 0.5)) for i in range(8)]
    p = _pilot(poses=poses, interact=it, gains=g)
    res = p.activate((0.5, 1.6, 0.9), 180.0, standoff=0.4)
    assert res.nav.arrived and res.clicked and res.reason == "clicked"
    assert it.clicks == 1


# ---- 中断(cancel/resume) ------------------------------------------------
def test_pilot_cancel_cancels_goto():
    poses = [_pose(i + 1, (0.15, 1.6, 0.15)) for i in range(10)]
    p = _pilot(poses=poses)
    p.cancel()
    res = p.goto((0.85, 0.85))
    assert res.path_found and not res.arrived and res.reason == "cancelled"
    p.resume()
    assert not p.cancelled


def test_pilot_cancel_cancels_aim():
    poses = [_pose(i + 1, (0.0, 1.0, 0.0), yaw_deg=90.0) for i in range(10)]
    p = _pilot(poses=poses)
    p.cancel()
    res = p.aim((0.0, 1.0, 5.0))
    assert not res.converged and res.reason == "cancelled"


def test_pilot_cancel_stops_patrol_between_targets():
    p = _pilot(poses=[_pose(1, (0.5, 1.6, 0.5))])
    p.cancel()
    results = p.patrol([("a", (0.5, 1.0, 0.9), 180.0), ("b", (0.9, 1.0, 0.5), 90.0)])
    assert results == []


# ---- 開ループ操作 --------------------------------------------------------
def test_pilot_stop_stops_both_actuators():
    p = _pilot()
    p.stop()
    assert p.look.stops == 1 and p.move.stops == 1


def test_pilot_move_for_commands_then_stops():
    p = _pilot()
    p.move_for(0.06, forward=0.5)
    assert p.move.moves and all(f == 0.5 for f, _s in p.move.moves)
    assert p.move.stops == 1


# ---- 待機系 --------------------------------------------------------------
def test_pilot_wait_until_near_immediate():
    p = _pilot(poses=[_pose(1, (0.5, 1.6, 0.5))])
    assert p.wait_until_near((0.5, 0.5), 0.1, timeout=0.2)
    assert not p.wait_until_near((0.9, 0.9), 0.1, timeout=0.1)


def test_pilot_is_hud_alive_detects_fresh_and_stale():
    fresh = _pilot(poses=[_pose(i + 1, (0.5, 1.6, 0.5)) for i in range(50)])
    assert fresh.is_hud_alive(timeout=0.5)
    stale = _pilot(poses=[])  # フレームが来ない
    assert not stale.is_hud_alive(timeout=0.1)


def test_pilot_is_usable_without_hardware_imports():
    # capture/osc を import せず、注入だけで Pilot が動く
    pilot = Pilot(
        _grid(np.ones((10, 10), bool)),
        FakeReader([_pose(1, (0.5, 1.6, 0.5))]),
        RecActuator(),
        RecActuator(),
    )
    assert pilot._owns_io is False
    res = pilot.goto((0.5, 0.5))  # start≈goal なのですぐ到達
    assert res.path_found
