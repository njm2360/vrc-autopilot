"""フレーム単位の照準幾何(現在ポーズ → 誤差・係数)。

navigation(オフラインの経路計画)とは別の関心事: 制御ループが毎フレーム呼ぶ
純粋関数群。角度の正規化 wrap180 もここが単一の出所。
"""

import math

import numpy as np


def wrap180(deg: float) -> float:
    """角度を (-180, 180] に正規化する(最短回りの誤差に使う)。"""
    return (deg + 180.0) % 360.0 - 180.0


def heading_error(
    cur_xz: tuple[float, float], cur_yaw_deg: float, target_xz: tuple[float, float]
) -> tuple[float, float]:
    """target への (yaw誤差[deg], 水平距離[m]) を返す。yaw誤差は最短回り、+で右。"""
    dx = target_xz[0] - cur_xz[0]
    dz = target_xz[1] - cur_xz[1]
    dist = math.hypot(dx, dz)
    desired_yaw = math.degrees(math.atan2(dx, dz))
    return wrap180(desired_yaw - cur_yaw_deg), dist


def pitch_error(
    eye_xyz: tuple[float, float, float],
    cur_forward: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
) -> float:
    """視線の pitch 誤差[deg]。+ は「もっと上を向く必要」。

    現在 pitch は forward.y から、目標 pitch は視点→ボタンの仰角から求める。
    """
    dx = target_xyz[0] - eye_xyz[0]
    dy = target_xyz[1] - eye_xyz[1]
    dz = target_xyz[2] - eye_xyz[2]
    horiz = math.hypot(dx, dz)
    desired_pitch = math.degrees(math.atan2(dy, horiz))
    fy = max(-1.0, min(1.0, cur_forward[1]))
    current_pitch = math.degrees(math.asin(fy))
    return desired_pitch - current_pitch


def aim_angle(
    eye_xyz: tuple[float, float, float],
    cur_forward: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
) -> float:
    """視線 forward と「視点→ボタン」方向との実際のなす角[deg](総合ずれの指標)。"""
    d = np.array(
        [
            target_xyz[0] - eye_xyz[0],
            target_xyz[1] - eye_xyz[1],
            target_xyz[2] - eye_xyz[2],
        ],
        dtype=np.float64,
    )
    n = np.linalg.norm(d)
    if n < 1e-9:
        return 0.0
    f = np.asarray(cur_forward, dtype=np.float64)
    f = f / (np.linalg.norm(f) + 1e-12)
    cos = float(np.clip(np.dot(d / n, f), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def forward_factor(yaw_err_deg: float, cutoff_deg: float = 90.0) -> float:
    """前進速度の減衰係数 [0,1]。正対で1、横向きで0(cos ベースで滑らか)。

    その場停止→旋回のガクつきを避けるため、向きのズレに応じて滑らかに減速する。
    |yaw_err| >= cutoff で 0。
    """
    a = abs(yaw_err_deg)
    if a >= cutoff_deg:
        return 0.0
    return max(0.0, math.cos(math.radians(a)))
