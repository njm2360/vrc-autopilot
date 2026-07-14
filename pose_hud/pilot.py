"""高レベル誘導ファサード。

Pilot は経路計画(navigation)と制御ループ(maneuvers)を束ね、goto / aim /
visit / patrol のフェーズ連結を提供する。実機 I/O(capture / osc / reader)は
connect() だけが知っており、コンストラクタ注入ならヘッドレスで動く。
"""

import time
from typing import Callable, Iterable

from .actuator import LookActuator, MouseLookActuator, MoveActuator
from .controller import PatrolGains, face_controllers, nav_controllers
from .maneuvers import AimResult, NavResult, PoseSource, aim_at, follow_path, turn_to
from .navigation import NavGrid, plan_path
from .telemetry import NullRecorder, Recorder


class Pilot:
    def __init__(
        self,
        grid: NavGrid,
        reader: PoseSource,
        look: LookActuator,
        move: MoveActuator,
        *,
        gains: PatrolGains | None = None,
        recorder: Recorder | None = None,
        announce: Callable[[str], None] | None = None,
        osc=None,
        owns_io: bool = False,
    ):
        self.grid = grid
        self.reader = reader
        self.look = look
        self.move = move
        self.gains = gains or PatrolGains()
        self.recorder = recorder or NullRecorder()
        self.announce = announce or (lambda _m: None)
        self._osc = osc
        self._owns_io = owns_io
        self.nav = nav_controllers(self.gains)
        self.face = face_controllers(self.gains)

    @classmethod
    def connect(
        cls,
        map_path=None,
        *,
        grid: NavGrid | None = None,
        cell: float = 0.1,
        radius: float = 0.25,
        gap_close: float = 0.3,
        gains: PatrolGains | None = None,
        look: str = "osc",
        mouse_yaw_gain: float = 40.0,
        mouse_pitch_gain: float = 40.0,
        recorder: Recorder | None = None,
        announce: Callable[[str], None] | None = None,
    ) -> "Pilot":
        if grid is None:
            if map_path is None:
                raise ValueError("map_path か grid のどちらかを指定してください")
            from .mapping import RoomMapper

            grid = NavGrid.from_mapper(
                RoomMapper.load(map_path),
                cell=cell,
                avatar_radius=radius,
                gap_close=gap_close,
            )
        from .capture import WindowsVRChatCapture
        from .osc import VRChatOSC
        from .reader import PoseReader

        reader = PoseReader(source=WindowsVRChatCapture()).start()
        osc = VRChatOSC()
        look_act: LookActuator = (
            osc
            if look == "osc"
            else MouseLookActuator(yaw_gain=mouse_yaw_gain, pitch_gain=mouse_pitch_gain)
        )
        osc.hud_enable(True)
        return cls(
            grid,
            reader,
            look_act,
            osc,
            gains=gains,
            recorder=recorder,
            announce=announce,
            osc=osc,
            owns_io=True,
        )

    def wait_for_hud(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.reader.get_latest() is not None:
                return True
            time.sleep(0.1)
        return False

    def goto(self, xz: tuple[float, float], *, name: str = "goto") -> NavResult:
        pose = self.reader.get_latest()
        if pose is None:
            self.announce(f"  [{name}] 現在位置が取れません(HUD?)")
            return NavResult(False, False, "no_pose", None, 0.0, 0)
        start = (pose.position[0], pose.position[2])
        path = plan_path(self.grid, start, xz)
        if path is None:
            self.announce(f"  [{name}] 経路なし(到達不能)")
            return NavResult(False, False, "unreachable", None, 0.0, 0)
        self.announce(
            f"  [{name}] 経路 {len(path.waypoints)}点 / {path.length:.1f}m"
            + ("(壁面→最寄り床へ)" if path.goal_blocked else "")
        )
        res = follow_path(
            self.reader,
            self.look,
            self.move,
            path.waypoints,
            self.gains,
            self.nav,
            recorder=self.recorder,
            announce=self.announce,
            name=name,
        )
        res.path = path
        return res

    def follow(
        self, waypoints: Iterable[tuple[float, float]], *, name: str = "follow"
    ) -> NavResult:
        return follow_path(
            self.reader,
            self.look,
            self.move,
            list(waypoints),
            self.gains,
            self.nav,
            recorder=self.recorder,
            announce=self.announce,
            name=name,
        )

    def aim(self, xyz: tuple[float, float, float], *, name: str = "aim") -> AimResult:
        return aim_at(
            self.reader,
            self.look,
            xyz,
            self.gains,
            self.face,
            recorder=self.recorder,
            name=name,
        )

    def turn_to(
        self, yaw_deg: float, pitch_deg: float | None = None, *, name: str = "turn"
    ) -> AimResult:
        return turn_to(
            self.reader,
            self.look,
            yaw_deg,
            self.gains,
            self.face,
            pitch_deg=pitch_deg,
            recorder=self.recorder,
            name=name,
        )

    def visit(
        self, xyz: tuple[float, float, float], *, name: str = "button"
    ) -> tuple[NavResult, AimResult | None]:
        nav = self.goto((xyz[0], xyz[2]), name=name)
        if not nav.reached:
            return nav, None
        aim = self.aim(xyz, name=name)
        self.announce(
            f"  [{name}] arrived. aim yaw_err={aim.yaw_err:+.2f}° "
            f"pitch_err={aim.pitch_err:+.2f}°"
        )
        return nav, aim

    def patrol(
        self, targets: Iterable[tuple[str, tuple[float, float, float]]]
    ) -> list[tuple[str, NavResult, AimResult | None]]:
        results = []
        for name, xyz in targets:
            nav, aim = self.visit(xyz, name=name)
            results.append((name, nav, aim))
        return results

    def close(self) -> None:
        try:
            self.look.stop()
        except Exception:
            pass
        if self._owns_io:
            if self._osc is not None:
                self._osc.close()
            self.reader.stop()

    def __enter__(self) -> "Pilot":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
