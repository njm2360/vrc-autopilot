"""PID コントローラのテスト。"""

from __future__ import annotations

import pytest

from vrc_autopilot.control.pid import PID


def test_pid_proportional_only():
    pid = PID(kp=0.5)
    assert pid.update(2.0, 0.1) == pytest.approx(1.0)  # 0.5*2=1.0
    assert pid.update(-10.0, 0.1) == pytest.approx(-1.0)  # クランプ


def test_pid_output_clamped():
    pid = PID(kp=1.0, out_min=-0.5, out_max=0.5)
    assert pid.update(10.0, 0.1) == pytest.approx(0.5)
    assert pid.update(-10.0, 0.1) == pytest.approx(-0.5)


def test_pid_integral_accumulates_and_removes_steady_state():
    # P だけだと届かない小さな一定誤差でも、I があれば出力が育つ
    pid = PID(kp=0.01, ki=0.5, kd=0.0, i_limit=1.0)
    out1 = pid.update(1.0, 0.1)
    out2 = pid.update(1.0, 0.1)
    out3 = pid.update(1.0, 0.1)
    assert out1 < out2 < out3  # 積分で増加
    assert out3 > out1


def test_pid_integral_clamped_by_i_limit():
    pid = PID(kp=0.0, ki=1.0, kd=0.0, out_min=-10, out_max=10, i_limit=0.3)
    for _ in range(100):
        out = pid.update(1.0, 0.1)
    assert out == pytest.approx(0.3, abs=1e-9)  # i_limit で頭打ち


def test_pid_derivative_reacts_to_change():
    pid = PID(kp=0.0, ki=0.0, kd=1.0, out_min=-100, out_max=100)
    pid.update(0.0, 0.1)  # prev=0
    out = pid.update(1.0, 0.1)  # d = (1-0)/0.1 = 10
    assert out == pytest.approx(10.0)


def test_pid_exposes_term_breakdown():
    pid = PID(kp=0.1, ki=0.5, kd=0.2, out_min=-10, out_max=10)
    pid.update(2.0, 0.1)  # prev=None → d=0
    out = pid.update(3.0, 0.1)
    # 内訳の合計が出力に一致(飽和していない前提)
    assert pid.last_p == pytest.approx(0.1 * 3.0)
    assert pid.last_d == pytest.approx(0.2 * (3.0 - 2.0) / 0.1)
    assert pid.last_out == pytest.approx(out)
    assert pid.last_p + pid.last_i + pid.last_d == pytest.approx(out)


def test_pid_deadzone_compensation():
    # out_floor>0 なら微小な非ゼロ出力でも最低 out_floor まで底上げする(符号保持)
    pid = PID(kp=0.01, out_floor=0.55)
    out = pid.update(1.0, 0.1)  # 生出力 0.01 → out_floor まで底上げ
    assert out == pytest.approx(0.55 + (1 - 0.55) * 0.01, abs=1e-6)
    neg = pid.update(-1.0, 0.1)
    assert neg < 0 and abs(neg) == pytest.approx(0.55 + (1 - 0.55) * 0.01, abs=1e-6)
    # ちょうど0(誤差0)は0のまま(底上げしない)
    assert PID(kp=0.1, out_floor=0.55).update(0.0, 0.1) == 0.0
    # 生出力が飽和(1.0)なら 1.0 のまま
    big = PID(kp=1.0, out_floor=0.55).update(5.0, 0.1)
    assert big == pytest.approx(1.0)


def test_pid_reset_derivative_keeps_integral():
    pid = PID(kp=0.0, ki=1.0, kd=1.0, out_min=-10, out_max=10)
    pid.update(1.0, 0.1)
    pid.update(1.0, 0.1)
    i_before = pid.last_i
    pid.reset_derivative()  # 積分は保持、微分履歴のみクリア
    pid.update(1.0, 0.1)
    assert pid.last_d == 0.0  # prev=None → d=0(微分キックなし)
    assert pid.last_i >= i_before  # 積分は保持されて増えている


def test_pid_reset():
    pid = PID(kp=0.0, ki=1.0)
    pid.update(1.0, 0.1)
    pid.update(1.0, 0.1)
    pid.reset()
    # reset 後は積分ゼロから
    assert pid.update(1.0, 0.1) == pytest.approx(1.0 * 0.1)


def test_pid_antiwindup_keys_off_output_not_integral():
    """飽和方向と同符号の誤差では、積分は逆符号でも巻き上がらない(条件は出力符号基準)。

    負の積分(≈−0.4)を仕込んでから大きな正の誤差を与えると出力は +1 に張り付く。
    バグ版(条件が積分符号基準 error*_i<0)はここで積分を正へ巻き上げてしまうが、
    修正版(error*unsat<0)は飽和と同方向なので積分を凍結する。
    """
    pid = PID(kp=0.05, ki=0.005, kd=0.0, i_limit=0.5, out_min=-1.0, out_max=1.0)
    # 負の誤差で積分を負側へ仕込む(この間は出力が範囲内なので普通に積む)
    for _ in range(160):
        pid.update(-1.0, 0.1)
    i_seed = pid.last_i
    assert i_seed < -0.05  # 負の積分が溜まっている
    # 大きな正の誤差 → 出力は +1 に飽和(誤差と同方向)
    for _ in range(50):
        out = pid.update(40.0, 0.1)
        assert out == pytest.approx(1.0)
    # 積分は凍結され、正へ巻き上がっていない(バグ版なら ≈0 まで上がる)
    assert pid.last_i == pytest.approx(i_seed, abs=1e-9)
    assert pid.last_i < -0.05


def test_pid_deadzone_reclamped_to_tight_rails():
    """不感帯補償後の出力が out_min/out_max(±1 より狭い)を超えない。"""
    pid = PID(kp=1.0, out_floor=0.55, out_min=-0.8, out_max=0.8)
    for e in (0.05, 0.5, 5.0, -0.05, -5.0):
        out = pid.update(e, 0.1)
        assert -0.8 - 1e-12 <= out <= 0.8 + 1e-12
    # 大きな誤差では補償(0.55+0.45=1.0)がレールで頭打ちになる
    assert pid.update(100.0, 0.1) == pytest.approx(0.8)


def test_pid_no_derivative_spike_on_wrapped_error():
    """±180 を跨ぐラップ誤差(+179→−179)で微分キックが出ない。"""
    pid = PID(kp=0.0, ki=0.0, kd=1.0, out_min=-100, out_max=100)
    pid.update(179.0, 0.1)  # prev=None → d=0
    out = pid.update(-179.0, 0.1)  # 生差 -358 だがラップなので d をスキップ
    assert pid.last_d == 0.0
    assert out == pytest.approx(0.0)
    # ラップ後は普通に微分が復帰する(-179→-178 は d=(-178-(-179))/dt=10)
    pid.update(-178.0, 0.1)
    assert pid.last_d == pytest.approx(10.0)


def test_pid_converges_in_sim():
    """1次系(角度)を PID で 0 に収束できることをシミュレーションで確認。"""
    pid = PID(kp=0.05, ki=0.02, kd=0.01, out_min=-1, out_max=1, i_limit=0.5)
    angle = 60.0  # 初期誤差60度
    dt = 0.05
    for _ in range(800):
        cmd = pid.update(angle, dt)  # +で右回転が必要
        angle -= cmd * 90.0 * dt  # コマンドに比例して角度が減る(90deg/s @cmd=1)
    assert abs(angle) < 5.0  # 収束(振動を経て±5度以内)
