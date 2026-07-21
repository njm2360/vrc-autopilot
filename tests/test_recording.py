from vrc_autopilot.control.recording import AxisAccumulator


def _feed(errors, tol=0.5, dt=0.05):
    acc = AxisAccumulator()
    t = 0.0
    for e in errors:
        acc.update(e, cmd=0.0, t=t, dt=dt, tol=tol)
        t += dt
    return acc.snapshot()


def test_overshoot_and_osc_counted_normally():
    # 目標(0)を跨いで反対側へ出る → オーバーシュートと符号反転を数える
    m = _feed([10.0, 5.0, 1.0, -3.0, -1.0, 2.0])
    assert m.overshoot > 0.0  # 反対符号(-3)へ出た
    assert m.osc >= 2  # +→-→+ で2回


def test_wrapped_error_not_counted_as_overshoot_or_osc():
    """±180 を跨ぐ誤差の符号反転は目標を跨いでいない → 幻のオーバーシュート/振動を数えない。"""
    # +179 付近から -179 付近へラップし、その後 -側で単調に0へ収束する
    seq = [179.0, 179.5, -179.0, -170.0, -120.0, -60.0, -10.0, -1.0]
    m = _feed(seq, tol=0.5)
    # ラップ由来の ~360 の幻オーバーシュートを数えない(初期符号 + に対し僅かのみ)
    assert m.overshoot < 5.0
    assert m.osc == 0  # 符号反転はラップ1回だけ → 数えない
    assert m.peak_err >= 179.0  # |e| 系の指標はそのまま


def test_true_overshoot_still_seen_after_wrap():
    """ラップを挟んでも、実際に目標を跨ぐ本物のオーバーシュートは拾う。"""
    # -179→+179 のラップ後、0 を越えて反対符号(-20)へ本当に行き過ぎる
    seq = [-179.0, 179.0, 100.0, 20.0, -20.0, -5.0]
    m = _feed(seq, tol=0.5)
    assert m.overshoot >= 15.0  # 初期符号(-)に対し +20 側へ出た本物のオーバーシュート
