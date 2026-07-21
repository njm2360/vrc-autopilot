from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ..core.pose import Pose


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("direction vector has ~zero length")
    return v / n


@dataclass(frozen=True)
class Sighting:
    """1地点からの視線レイ(原点+正規化方向)と付随情報。"""

    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    label: str = ""
    time_ms: int = 0

    @classmethod
    def from_pose(cls, pose: Pose, label: str = "") -> Sighting:
        """Pose の位置=原点・forward=方向としてレイを作る。"""
        d = _normalize(np.asarray(pose.forward, dtype=np.float64))
        return cls(
            origin=tuple(float(v) for v in pose.position),
            direction=tuple(float(v) for v in d),
            label=label,
            time_ms=pose.time_ms,
        )

    @property
    def origin_arr(self) -> np.ndarray:
        return np.asarray(self.origin, dtype=np.float64)

    @property
    def direction_arr(self) -> np.ndarray:
        return _normalize(np.asarray(self.direction, dtype=np.float64))


@dataclass(frozen=True)
class TriangulationResult:
    point: tuple[float, float, float]  # 推定ボタン座標 [m]
    residual_rms: float  # 各レイへの垂直距離のRMS [m]
    ray_distances: tuple[float, ...]  # 各レイへの垂直距離 [m]
    n: int  # 使用レイ数
    max_pair_angle_deg: float  # レイ間の最大なす角 [deg]
    condition: float  # 正規方程式の条件数(大きいほど不安定)
    well_conditioned: bool  # 幾何が三角測量に十分か

    def to_dict(self) -> dict:
        return {
            "point": {"x": self.point[0], "y": self.point[1], "z": self.point[2]},
            "residual_rms_m": self.residual_rms,
            "ray_distances_m": list(self.ray_distances),
            "n": self.n,
            "max_pair_angle_deg": self.max_pair_angle_deg,
            "condition": self.condition,
            "well_conditioned": self.well_conditioned,
        }


MIN_GEOMETRY_ANGLE_DEG = 5.0


def closest_point_to_rays(
    origins: np.ndarray, directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if len(origins) < 2:
        raise ValueError("need at least 2 rays to triangulate")
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    n = len(origins)
    # A = Σ (I - d d^T) = N*I - D^T D
    A = n * np.eye(3) - dirs.T @ dirs
    # b = Σ (I - d d^T) o = Σ o - Σ d (d·o)
    dots = np.sum(dirs * origins, axis=1)  # (N,)
    b = origins.sum(axis=0) - (dirs * dots[:, None]).sum(axis=0)
    # 平行に近いと A が特異になりうるので lstsq で安定に解く
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    return p, A


def _perp_distances(p: np.ndarray, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    w = p[None, :] - origins  # (N,3)
    proj = np.sum(w * dirs, axis=1, keepdims=True) * dirs
    return np.linalg.norm(w - proj, axis=1)


def _max_pair_angle_deg(dirs: np.ndarray) -> float:
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    cos = np.clip(d @ d.T, -1.0, 1.0)
    ang = np.degrees(np.arccos(np.abs(cos)))  # 0..90
    return float(ang.max()) if len(d) > 1 else 0.0


def triangulate(
    sightings: Sequence[Sighting], min_angle_deg: float = MIN_GEOMETRY_ANGLE_DEG
) -> TriangulationResult:
    if len(sightings) < 2:
        raise ValueError("need at least 2 sightings to triangulate")
    origins = np.array([s.origin for s in sightings], dtype=np.float64)
    dirs = np.array([s.direction_arr for s in sightings], dtype=np.float64)

    p, A = closest_point_to_rays(origins, dirs)
    dists = _perp_distances(p, origins, dirs)
    rms = float(np.sqrt(np.mean(dists**2)))
    angle = _max_pair_angle_deg(dirs)
    cond = float(np.linalg.cond(A))
    well = angle >= min_angle_deg and np.isfinite(cond)

    return TriangulationResult(
        point=(float(p[0]), float(p[1]), float(p[2])),
        residual_rms=rms,
        ray_distances=tuple(float(d) for d in dists),
        n=len(sightings),
        max_pair_angle_deg=angle,
        condition=cond,
        well_conditioned=bool(well),
    )


def triangulate_poses(
    poses: Iterable[Pose], min_angle_deg: float = MIN_GEOMETRY_ANGLE_DEG
) -> TriangulationResult:
    """Pose 群から直接三角測量するショートカット。"""
    return triangulate([Sighting.from_pose(p) for p in poses], min_angle_deg)
