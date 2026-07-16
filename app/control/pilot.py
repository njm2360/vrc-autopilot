import logging
import math
import time
from typing import Iterable

from .actuator import LookActuator, MoveActuator
from .controller import (
    PatrolGains,
    face_controllers,
    nav_controllers,
    strafe_controller,
    translate_controllers,
)
from .maneuvers import (
    AimResult,
    NavResult,
    PoseSource,
    aim_at,
    follow_path,
    follow_path_hold_view,
    strafe_align,
    turn_to,
)
from ..spatial.navigation import NavGrid, plan_path
from .telemetry import NullRecorder, Recorder

logger = logging.getLogger(__name__)


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
        osc=None,
        owns_io: bool = False,
    ):
        self.grid = grid
        self.reader = reader
        self.look = look
        self.move = move
        self.gains = gains or PatrolGains()
        self.recorder = recorder or NullRecorder()
        self._osc = osc
        self._owns_io = owns_io
        self.nav = nav_controllers(self.gains)
        self.face = face_controllers(self.gains)
        self.strafe = strafe_controller(self.gains)
        self.translate = translate_controllers(self.gains)

    @classmethod
    def connect(
        cls,
        grid: NavGrid,
        *,
        gains: PatrolGains | None = None,
        look: LookActuator | None = None,
        recorder: Recorder | None = None,
    ) -> "Pilot":
        """実機 I/O(キャプチャ+OSC)を組んだ Pilot を作る(注入版は __init__)。

        look を渡すと視点だけ差し替えられる(例: MouseLookActuator)。省略時は OSC。
        """
        from ..perception.capture import WindowsVRChatCapture
        from .osc import VRChatOSC
        from ..perception.reader import PoseReader

        reader = PoseReader(source=WindowsVRChatCapture()).start()
        osc = VRChatOSC()
        osc.hud_enable(True)
        return cls(
            grid,
            reader,
            look or osc,
            osc,
            gains=gains,
            recorder=recorder,
            osc=osc,
            owns_io=True,
        )

    def wait_for_hud(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.reader.get_latest() is not None:
                return True
            time.sleep(0.1)
        logger.warning("HUD が %.0f 秒読めません(VRChat 起動中? HUD_Enable?)", timeout)
        return False

    def goto(self, xz: tuple[float, float], *, name: str = "goto") -> NavResult:
        pose = self.reader.get_latest()
        if pose is None:
            logger.warning("[%s] 現在位置が取れません(HUD?)", name)
            return NavResult(False, False, "no_pose", None, 0.0, 0)
        start = (pose.position[0], pose.position[2])
        path = plan_path(self.grid, start, xz)
        if path is None:
            logger.warning("[%s] 経路なし(到達不能)", name)
            return NavResult(False, False, "unreachable", None, 0.0, 0)
        logger.info(
            "[%s] 経路 %d点 / %.1fm%s",
            name, len(path.waypoints), path.length,
            "(壁面→最寄り床へ)" if path.goal_blocked else "",
        )
        res = follow_path(
            self.reader,
            self.look,
            self.move,
            path.waypoints,
            self.gains,
            self.nav,
            recorder=self.recorder,
            name=name,
        )
        res.path = path
        return res

    def move_to(self, xz: tuple[float, float], *, name: str = "move") -> NavResult:
        """視点を変えずに xz へ並進する。壁回避は goto と同じ plan_path が担う。

        進行方向へ視点を回さない点だけが goto と違う。前後+横移動で経路を追うので、
        経路が横方向に大きく曲がると goto より遅い(strafe が前進より遅いため)。
        """
        pose = self.reader.get_latest()
        if pose is None:
            logger.warning("[%s] 現在位置が取れません(HUD?)", name)
            return NavResult(False, False, "no_pose", None, 0.0, 0)
        start = (pose.position[0], pose.position[2])
        path = plan_path(self.grid, start, xz)
        if path is None:
            logger.warning("[%s] 経路なし(到達不能)", name)
            return NavResult(False, False, "unreachable", None, 0.0, 0)
        logger.info(
            "[%s] 経路 %d点 / %.1fm (視点固定)%s",
            name, len(path.waypoints), path.length,
            "(壁面→最寄り床へ)" if path.goal_blocked else "",
        )
        res = follow_path_hold_view(
            self.reader,
            self.look,
            self.move,
            path.waypoints,
            self.gains,
            self.translate,
            recorder=self.recorder,
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

    def align(
        self, xyz: tuple[float, float, float], *, name: str = "align"
    ) -> AimResult:
        return strafe_align(
            self.reader,
            self.look,
            self.move,
            xyz,
            self.gains,
            self.face,
            self.strafe,
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

    def _standoff_goal(
        self,
        xyz: tuple[float, float, float],
        standoff: float,
        face_yaw_deg: float,
    ) -> tuple[float, float]:
        if standoff <= 0.0:
            return (xyz[0], xyz[2])
        y = math.radians(face_yaw_deg)
        return (xyz[0] + math.sin(y) * standoff, xyz[2] + math.cos(y) * standoff)

    def visit(
        self,
        xyz: tuple[float, float, float],
        face_yaw_deg: float,
        *,
        name: str = "button",
        standoff: float | None = None,
    ) -> tuple[NavResult, AimResult | None]:
        d = self.gains.standoff if standoff is None else standoff
        nav = self.goto(self._standoff_goal(xyz, d, face_yaw_deg), name=name)
        if not nav.reached:
            return nav, None
        aim = self.aim(xyz, name=name)
        logger.info(
            "[%s] arrived. aim yaw_err=%+.2f° pitch_err=%+.2f° (%s)",
            name, aim.yaw_err, aim.pitch_err, aim.reason,
        )
        if self.gains.align_tol > 0.0:
            aim = self.align(xyz, name=name)
            logger.info(
                "[%s] align yaw_err=%+.2f° pitch_err=%+.2f° (%s)",
                name, aim.yaw_err, aim.pitch_err, aim.reason,
            )
        return nav, aim

    def patrol(
        self,
        targets: Iterable[tuple[str, tuple[float, float, float], float]],
    ) -> list[tuple[str, NavResult, AimResult | None]]:
        results = []
        for name, xyz, face_yaw in targets:
            nav, aim = self.visit(xyz, face_yaw, name=name)
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
