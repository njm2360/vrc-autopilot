"""パッケージ共通のドメイン型: 6DoF ポーズ。

HUD デコード(decode)が生成し、mapping / triangulate / maneuvers など全層で
使われる中核の型。座標系は Unity 準拠(Y-up, 左手系, 単位メートル)。
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose:
    """復元された 6DoF ポーズ(Unity座標系: Y-up, 左手系, 単位メートル)。"""

    time_ms: int  # _VRChatTimeNetworkMs (ラップあり)
    position: tuple[float, float, float]
    forward: tuple[float, float, float]
    up: tuple[float, float, float]

    @property
    def yaw_deg(self) -> float:
        """+Z 基準の yaw。atan2(fwd.x, fwd.z)。"""
        return math.degrees(math.atan2(self.forward[0], self.forward[2]))

    @property
    def pitch_deg(self) -> float:
        """上向きが正の pitch。asin(fwd.y)。"""
        return math.degrees(math.asin(max(-1.0, min(1.0, self.forward[1]))))

    @property
    def roll_deg(self) -> float:
        """up ベクトルから求めた roll。デスクトップでは常に≒0(VR対応用)。"""
        fwd = np.asarray(self.forward, dtype=np.float64)
        up = np.asarray(self.up, dtype=np.float64)
        # forward まわりで world-up を投影した右手系の傾き
        right = np.cross(up, fwd)
        world_up_proj = np.cross(fwd, right)
        return math.degrees(
            math.atan2(
                np.dot(right, [0.0, 1.0, 0.0]), np.dot(world_up_proj, [0.0, 1.0, 0.0])
            )
        )
