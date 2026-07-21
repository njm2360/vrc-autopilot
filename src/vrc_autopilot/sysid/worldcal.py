import json
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import median

from ..control.controller import PatrolGains
from ..control.maneuvers import PoseSource
from .identify import AxisModel, ProbeRun, identify_axis, run_move_probe

logger = logging.getLogger(__name__)

# キャリブレーション対象の軸(視点軸はワールド不変なので対象外)
CAL_AXES = ("forward", "strafe")

# 移動軸の入力不感帯オンセット(cmd 単位、クライアント側なのでワールド不変)
MOVE_DEADBAND_CMD = 0.10
# 失速下限クランプの安全率(kp·arrive_radius > 不感帯×これ を保証する)
_STALL_MARGIN = 1.2

# ゲイン調整時の基準値。再同定したら deadzone 定数と同様にここも更新する
# (gain-tuning.md「再同定後の再検証チェックリスト」)
REF_SPEED = {"forward": 6.0, "strafe": 3.0}  # cmd=1 の速度 [m/s]
REF_DEADTIME_S = {"forward": 0.016, "strafe": 0.030}


@dataclass(frozen=True)
class ScaleEstimate:
    """基準速度に対する 1 軸の速度倍率の測定結果。"""

    axis: str
    scale: float  # 速度倍率 s(±方向の中央値)。usable=False なら参考値
    rates: list[tuple[float, float]]  # 実測の (cmd, 速度[m/s])
    deadtime_s: float  # プローブで実測したむだ時間 [s]
    usable: bool  # ゲイン再スケールに使ってよいか
    reason: str  # usable=False の説明


def estimate_scale(
    model: AxisModel,
    *,
    ref_speed: float | None = None,
    s_min: float = 0.05,
    agree_tol: float = 0.2,
) -> ScaleEstimate:
    """±1.0 プローブの同定結果から速度倍率(実測速度/基準速度の中央値)を測る"""
    ref = REF_SPEED[model.axis] if ref_speed is None else ref_speed
    rates = [(c, r) for c, r in model.points if abs(c) >= 0.99]
    scales = [r / (c * ref) for c, r in rates]

    def bad(reason: str, scale: float = 0.0) -> ScaleEstimate:
        return ScaleEstimate(
            axis=model.axis,
            scale=scale,
            rates=rates,
            deadtime_s=model.deadtime_s,
            usable=False,
            reason=reason,
        )

    if not scales:
        return bad("no full-command measurement in probe")
    if any(s <= 0.0 for s in scales):
        return bad("non-positive response (reversed or absent)")
    s = float(median(scales))
    if s < s_min:
        return bad(f"scale {s:.3f} < {s_min} (immobilized or locomotion disabled?)", s)
    if len(scales) >= 2 and max(scales) / min(scales) > 1.0 + agree_tol:
        return bad(
            f"+/- speeds disagree ({min(scales):.3f} vs {max(scales):.3f}) -- "
            "blocked by an obstacle during probe? re-run in open space",
            s,
        )
    return ScaleEstimate(
        axis=model.axis,
        scale=s,
        rates=rates,
        deadtime_s=model.deadtime_s,
        usable=True,
        reason="",
    )


def probe_axis_speed(
    reader: PoseSource,
    send: Callable[[float], None],
    *,
    axis: str,
    max_travel: float = 3.0,
    hold: float = 1.2,
    settle: float = 0.6,
    monotonic,
    sleep,
) -> tuple[AxisModel, ProbeRun]:
    """移動1軸の cmd=±1.0 速度プローブ(位置ガード付き、往復で元の位置に戻る)"""
    run = run_move_probe(
        reader,
        send,
        [1.0],
        axis=axis,
        max_travel=max_travel,
        hold=hold,
        settle=settle,
        passes=0,  # +1.0 で行き、ホーム帰還の -1.0 で帰る(両符号を 1回ずつ)
        monotonic=monotonic,
        sleep=sleep,
    )
    return identify_axis(run), run


@dataclass(frozen=True)
class ScaledGains:
    """ゲイン再スケールの結果"""

    gains: PatrolGains
    s_forward: float
    s_strafe: float
    notes: list[str]


def scale_gains(
    base: PatrolGains,
    s_forward: float,
    s_strafe: float,
    *,
    move_deadband: float = MOVE_DEADBAND_CMD,
    cap_cruise: bool = True,
) -> ScaledGains:
    """移動系ゲインを速度倍率で再スケールした PatrolGains を導出する。

    kp/ki/kd を 1/s 倍して実効ループゲイン(ωc ≈ kp·K)を保つ。視点系はワールド
    不変なので触らない。失速下限(kp·arrive_radius > 移動不感帯)を割るスケールはクランプ
    する。cap_cruise は速いワールドで巡航指令を下げ、実巡航速度を調整済みワールドと
    揃える(carrot 先読みと到達の幾何を変えないため)。遅いワールドは安全側なので
    上げない。倍率は estimate_scale の usable な値だけを渡すこと(s≈0 は発散する)。
    """
    if s_forward <= 0.0 or s_strafe <= 0.0:
        raise ValueError(f"scales must be positive (got {s_forward}, {s_strafe})")
    notes: list[str] = []
    s_trans = max(s_forward, s_strafe)  # 並進 2 軸共用ゲインは速い側に合わせる

    def scaled(name: str, value: float, s: float, floor: float = 0.0) -> float:
        v = value / s
        if v < floor:
            notes.append(
                f"{name}: {value:.3g}/s={v:.3g} clamped to stall floor {floor:.3g}"
            )
            return floor
        return v

    fwd_floor = move_deadband * _STALL_MARGIN / base.arrive_radius
    fwd_kp = scaled("fwd_kp", base.fwd_kp, s_forward, fwd_floor)
    fwd_kd = base.fwd_kd / s_forward
    # translate は斜め45°で誤差が両体軸へ √2 分配されるので下限も √2 倍
    translate_floor = (
        move_deadband * _STALL_MARGIN * math.sqrt(2.0) / base.arrive_radius
    )
    translate_kp = scaled("translate_kp", base.translate_kp, s_trans, translate_floor)
    translate_ki = base.translate_ki / s_trans
    translate_kd = base.translate_kd / s_trans
    # align strafe は不感帯補償持ちなので失速下限は不要
    strafe_kp = base.strafe_kp / s_strafe
    strafe_ki = base.strafe_ki / s_strafe
    strafe_kd = base.strafe_kd / s_strafe

    speed = base.speed
    if cap_cruise and s_trans > 1.0:
        # 実巡航 v ∝ (cmd − 不感帯) は原点を通らないので逆変換に不感帯を残す。
        # speed/s だと不感帯側へ寄りすぎ、s≥9 で巡航指令が不感帯に埋まって動けない
        speed = move_deadband + (base.speed - move_deadband) / s_trans
        notes.append(
            f"speed: cruise cap {base.speed:.2f} -> {speed:.2f} "
            f"(keeps real cruise speed at tuned level)"
        )

    gains = replace(
        base,
        fwd_kp=fwd_kp,
        fwd_kd=fwd_kd,
        translate_kp=translate_kp,
        translate_ki=translate_ki,
        translate_kd=translate_kd,
        strafe_kp=strafe_kp,
        strafe_ki=strafe_ki,
        strafe_kd=strafe_kd,
        speed=speed,
    )
    return ScaledGains(gains=gains, s_forward=s_forward, s_strafe=s_strafe, notes=notes)


@dataclass
class WorldCalibration:
    axes: dict[str, ScaleEstimate]  # "forward" / "strafe"
    warnings: list[str] = field(default_factory=list)  # むだ時間増加など
    meta: dict = field(default_factory=dict)

    def apply(self, base: PatrolGains) -> ScaledGains:
        """移動系ゲインをこのワールドの倍率で再スケールした PatrolGains を導出する"""
        missing = [a for a in CAL_AXES if a not in self.axes]
        if missing:
            raise ValueError(f"calibration is missing axes: {missing}")
        bad = [a for a in CAL_AXES if not self.axes[a].usable]
        if bad:
            reasons = "; ".join(f"{a}: {self.axes[a].reason}" for a in bad)
            raise ValueError(f"calibration unusable ({reasons})")
        est_f, est_s = self.axes["forward"], self.axes["strafe"]
        sg = scale_gains(base, est_f.scale, est_s.scale)
        return ScaledGains(
            gains=sg.gains,
            s_forward=est_f.scale,
            s_strafe=est_s.scale,
            notes=list(self.warnings) + sg.notes,
        )

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "meta": self.meta,
            "warnings": self.warnings,
            "axes": {name: asdict(est) for name, est in self.axes.items()},
        }
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path) -> WorldCalibration:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        axes = {
            name: ScaleEstimate(
                axis=a["axis"],
                scale=float(a["scale"]),
                rates=[(float(c), float(r)) for c, r in a["rates"]],
                deadtime_s=float(a["deadtime_s"]),
                usable=bool(a["usable"]),
                reason=a.get("reason", ""),
            )
            for name, a in data["axes"].items()
        }
        return cls(
            axes=axes,
            warnings=list(data.get("warnings", [])),
            meta=data.get("meta", {}),
        )


def run_world_calibration(
    reader: PoseSource,
    send: Callable[[str, float], None],
    *,
    monotonic,
    sleep,
    max_travel: float = 3.0,
    hold: float = 1.2,
    settle: float = 0.6,
    ref_speed: dict[str, float] | None = None,
    ref_deadtime_s: dict[str, float] | None = None,
    meta: dict | None = None,
) -> WorldCalibration:
    """移動 2 軸の ±1.0 プローブを打ち、WorldCalibration を組む"""
    axes: dict[str, ScaleEstimate] = {}
    warnings: list[str] = []
    for axis in CAL_AXES:
        model, _run = probe_axis_speed(
            reader,
            lambda v, a=axis: send(a, v),
            axis=axis,
            max_travel=max_travel,
            hold=hold,
            settle=settle,
            monotonic=monotonic,
            sleep=sleep,
        )
        est = estimate_scale(
            model, ref_speed=None if ref_speed is None else ref_speed[axis]
        )
        axes[axis] = est
        ref_dead = (REF_DEADTIME_S if ref_deadtime_s is None else ref_deadtime_s)[axis]
        warn = check_deadtime(est, ref_deadtime_s=ref_dead)
        if warn:
            warnings.append(warn)
    return WorldCalibration(axes=axes, warnings=warnings, meta=meta or {})


def check_deadtime(
    est: ScaleEstimate, *, ref_deadtime_s: float, ratio_warn: float = 2.0
) -> str | None:
    if ref_deadtime_s <= 0.0 or est.deadtime_s <= 0.0:
        return None
    ratio = est.deadtime_s / ref_deadtime_s
    if ratio > ratio_warn:
        return (
            f"axis {est.axis}: probed deadtime {est.deadtime_s * 1000:.0f}ms is "
            f"{ratio:.1f}x reference ({ref_deadtime_s * 1000:.0f}ms) -- view-axis "
            "margins shrink; consider robust gains or full re-identification"
        )
    return None
