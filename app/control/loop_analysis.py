"""巡回制御ループの周波数応答解析(安定余裕・ボード線図)

PatrolGains と同定プラント(PlantModel)から各ループの開ループ伝達関数を小信号
線形化で組み、ωc/位相余裕/ゲイン余裕/むだ時間余裕を出す

    L(z) = C(z) · K · [T·z^-1/(1-z^-1)] · e^{-jωTd},   z = e^{jωT}

C は離散PID、K は静特性の傾き×(1-不感帯補償)、Td はむだ時間、T はフレーム周期
"""

import math
from dataclasses import dataclass

import numpy as np

from ..sysid.identify import PlantModel
from .controller import PatrolGains


@dataclass(frozen=True)
class LoopSpec:
    """1制御ループの線形化パラメータ(誤差→指令の PID と対象軸)。"""

    name: str
    axis: str
    kp: float
    ki: float
    kd: float
    compensated: bool  # 不感帯補償の有無(小信号ゲインが 1-onset 倍に圧縮される)


@dataclass
class LoopMargins:
    """1ループの安定余裕。角周波数は rad/s、時間は s。未交差は None。"""

    name: str
    axis: str
    K: float  # 実効ゲイン[軸単位/s per 指令]
    deadtime_s: float
    wc: float | None  # ゲイン交差角周波数
    pm_deg: float | None  # 位相余裕
    w180: float | None  # 位相交差角周波数
    gm: float | None  # ゲイン余裕(倍率。dB は 20·log10)
    delay_margin_s: float | None  # 追加で許容できるむだ時間
    mr: float  # 閉ループピーク |L/(1+L)| の最大


def patrol_loops(g: PatrolGains) -> list[LoopSpec]:
    """PatrolGains から巡回制御の全ループを組む(controller.py のビルダーと対応)。"""
    return [
        LoopSpec("face yaw", "yaw", g.turn_kp, g.turn_ki, g.turn_kd, True),
        LoopSpec("face pitch", "pitch", g.pitch_kp, g.pitch_ki, g.pitch_kd, True),
        LoopSpec("nav yaw", "yaw", g.nav_turn_kp, g.nav_turn_ki, g.nav_turn_kd, True),
        LoopSpec("nav forward", "forward", g.fwd_kp, 0.0, g.fwd_kd, False),
        LoopSpec(
            "translate forward",
            "forward",
            g.translate_kp,
            g.translate_ki,
            g.translate_kd,
            False,
        ),
        LoopSpec(
            "translate strafe",
            "strafe",
            g.translate_kp,
            g.translate_ki,
            g.translate_kd,
            False,
        ),
        LoopSpec("align strafe", "strafe", g.strafe_kp, g.strafe_ki, g.strafe_kd, True),
    ]


def _axis_gain(plant: PlantModel, spec: LoopSpec) -> float:
    """静特性の傾き(オンセットより上を最小二乗)から実効ゲイン K を出す。"""
    onset = plant.axes[spec.axis].onset
    pts = [(c, r) for c, r in plant.axes[spec.axis].points if abs(c) > onset + 0.02]
    xs = np.array([math.copysign(abs(c) - onset, c) for c, _ in pts])
    ys = np.array([r for _, r in pts])
    slope = float(xs @ ys / (xs @ xs))
    return slope * (1.0 - onset) if spec.compensated else slope


def loop_response(
    spec: LoopSpec, K: float, deadtime_s: float, T: float, w: np.ndarray
) -> np.ndarray:
    """開ループ L(e^{jωT}) を角周波数配列 w[rad/s] 上で評価する。"""
    w = np.asarray(w, float)
    z1 = np.exp(-1j * w * T)  # z^-1
    C = spec.kp + 0j
    if spec.ki:
        C = C + spec.ki * T / (1.0 - z1)  # 後退オイラー積分
    if spec.kd:
        C = C + (spec.kd / T) * (1.0 - z1)  # 後退差分微分
    integ = T * z1 / (1.0 - z1)  # プラント積分器(1サンプル遅れ込み)
    return C * K * integ * np.exp(-1j * w * deadtime_s)


def loop_margins(spec: LoopSpec, plant: PlantModel) -> LoopMargins:
    """1ループの安定余裕を求める。"""
    K = _axis_gain(plant, spec)
    Td = plant.axes[spec.axis].deadtime_s
    T = plant.dt_mean
    w = np.logspace(-1, math.log10(math.pi / T), 6000)
    L = loop_response(spec, K, Td, T, w)
    mag = np.abs(L)
    ph = np.unwrap(np.angle(L))
    wc = pm = w180 = gm = None
    ic = np.nonzero((mag[:-1] >= 1.0) & (mag[1:] < 1.0))[0]
    if len(ic):
        i = ic[0]
        f = (1.0 - mag[i]) / (mag[i + 1] - mag[i])
        wc = float(w[i] + f * (w[i + 1] - w[i]))
        pm = math.degrees((ph[i] + f * (ph[i + 1] - ph[i])) + math.pi)
    ip = np.nonzero((ph[:-1] >= -math.pi) & (ph[1:] < -math.pi))[0]
    if len(ip):
        i = ip[0]
        f = (-math.pi - ph[i]) / (ph[i + 1] - ph[i])
        w180 = float(w[i] + f * (w[i + 1] - w[i]))
        gm = 1.0 / float(mag[i] + f * (mag[i + 1] - mag[i]))
    dm = math.radians(pm) / wc if (pm is not None and wc) else None
    mr = float(np.max(np.abs(L / (1.0 + L))))
    return LoopMargins(spec.name, spec.axis, K, Td, wc, pm, w180, gm, dm, mr)


def analyze_patrol(g: PatrolGains, plant: PlantModel) -> list[LoopMargins]:
    """巡回制御の全ループの安定余裕をまとめて返す。"""
    return [loop_margins(s, plant) for s in patrol_loops(g)]


def save_bode_png(g: PatrolGains, plant: PlantModel, path) -> None:
    """全ループの開ループボード線図(|L|・∠L と ωc/PM/GM)を1枚のPNGに保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = plant.dt_mean
    w = np.logspace(-1, math.log10(math.pi / T), 2000)
    specs = patrol_loops(g)
    fig, axes = plt.subplots(4, 2, figsize=(14, 15))
    flat = axes.ravel()
    for ax, spec in zip(flat, specs, strict=False):  # flat は 8 枠、specs は 7
        m = loop_margins(spec, plant)
        L = loop_response(spec, m.K, m.deadtime_s, T, w)
        ax.semilogx(w, 20 * np.log10(np.abs(L)), color="tab:blue", lw=1.6)
        ax.axhline(0, color="gray", lw=0.7, ls=":")
        a2 = ax.twinx()
        a2.semilogx(
            w, np.degrees(np.unwrap(np.angle(L))), color="tab:red", lw=1.1, ls="--"
        )
        a2.axhline(-180, color="tab:red", lw=0.6, ls=":")
        if m.wc:
            ax.axvline(m.wc, color="k", lw=0.8)
            ax.plot(m.wc, 0, "ko", ms=4)
            a2.annotate(
                f"PM {m.pm_deg:.0f}deg",
                (m.wc, -180 + m.pm_deg),
                color="tab:red",
                fontsize=8,
                xytext=(3, 0),
                textcoords="offset points",
            )
        if m.gm:
            gdb = 20 * math.log10(m.gm)
            ax.plot(m.w180, -gdb, "bs", ms=4)
            ax.annotate(
                f"GM {gdb:.1f}dB x{m.gm:.1f}",
                (m.w180, -gdb),
                color="tab:blue",
                fontsize=8,
                xytext=(3, 5),
                textcoords="offset points",
            )
        ax.set_title(
            f"{spec.name}  kp={spec.kp} ki={spec.ki} kd={spec.kd} "
            f"K={m.K:.0f} Td={m.deadtime_s * 1000:.0f}ms",
            fontsize=9,
        )
        ax.set_xlabel("w [rad/s]")
        ax.set_ylabel("|L| [dB]", color="tab:blue")
        a2.set_ylabel("phase [deg]", color="tab:red")
        ax.set_ylim(-60, 40)
        a2.set_ylim(-300, -60)
        ax.grid(True, which="both", alpha=0.25)
    for ax in flat[len(specs) :]:
        ax.axis("off")
    fig.suptitle(
        f"Patrol control loops - open-loop Bode (plant dt={T * 1000:.0f}ms)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(path, dpi=110)
    plt.close(fig)
