import logging
import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..core.pose import Pose
from ..spatial.navigation import NavGrid, Path, plan_path
from ..sysid.worldcal import WorldCalibration
from .actuator import InteractActuator, LookActuator, MoveActuator
from .controller import (
    PatrolGains,
    face_controllers,
    nav_controllers,
    strafe_controller,
    translate_controllers,
)
from .guidance import heading_error, pitch_error
from .guidance import standoff_point as _standoff_point
from .maneuvers import (
    AimResult,
    NavResult,
    PoseSource,
    aim_at,
    follow_path,
    follow_path_translate,
    strafe_align,
    turn_to,
)
from .recording import NullRecorder, Recorder

logger = logging.getLogger(__name__)


@dataclass
class ClickAtResult:
    """照準(aim/align)して押した結果。"""

    aim: AimResult
    clicked: bool

    @property
    def reason(self) -> str:
        return "clicked" if self.clicked else self.aim.reason


@dataclass
class ActivateResult:
    """移動+照準+押下の一連の結果。"""

    nav: NavResult
    aim: AimResult | None
    clicked: bool

    @property
    def reason(self) -> str:
        if self.clicked:
            return "clicked"
        if self.aim is not None:
            return self.aim.reason
        return self.nav.reason


class Pilot:
    def __init__(
        self,
        grid: NavGrid,
        reader: PoseSource,
        look: LookActuator,
        move: MoveActuator,
        *,
        interact: InteractActuator | None = None,
        gains: PatrolGains | None = None,
        world_cal: WorldCalibration | str | None = None,
        recorder: Recorder | None = None,
        osc=None,
        owns_io: bool = False,
    ):
        self.grid = grid
        self.reader = reader
        self.look = look
        self.move = move
        self.interact = interact
        self.gains = gains or PatrolGains()
        if world_cal is not None:
            if not isinstance(world_cal, WorldCalibration):
                world_cal = WorldCalibration.load(world_cal)
            applied = world_cal.apply(self.gains)
            self.gains = applied.gains
            logger.info(
                "world_cal applied: speed scale forward x%.2f / strafe x%.2f",
                applied.s_forward,
                applied.s_strafe,
            )
            for note in applied.notes:
                logger.warning("world_cal: %s", note)
        self.recorder = recorder or NullRecorder()
        self._osc = osc
        self._owns_io = owns_io
        self._cancel = threading.Event()
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
        world_cal: WorldCalibration | str | None = None,
        look: LookActuator | None = None,
        interact: InteractActuator | None = None,
        recorder: Recorder | None = None,
    ) -> Pilot:
        """実機 I/O(キャプチャ+OSC)を組んだ Pilot を作る(注入版は __init__)。

        look / interact を渡すと視点・押下だけ差し替えられる。省略時はどちらも OSC。
        world_cal は calibrate-world の JSON パス(またはロード済み WorldCalibration)。
        """
        from ..perception.capture import WindowsVRChatCapture
        from ..perception.reader import PoseReader
        from .osc import VRChatOSC

        reader = PoseReader(source=WindowsVRChatCapture()).start()
        osc = VRChatOSC()
        osc.hud_enable(True)
        osc.set_run(True)
        return cls(
            grid,
            reader,
            look or osc,
            osc,
            interact=interact or osc,
            gains=gains,
            world_cal=world_cal,
            recorder=recorder,
            osc=osc,
            owns_io=True,
        )

    # ---- 状態クエリ(sense) --------------------------------------------
    @staticmethod
    def _xz_of(target: Iterable[float]) -> tuple[float, float]:
        """(x,z) / (x,y,z) のどちらでも水平成分を取り出す。"""
        t = tuple(target)
        return (t[0], t[-1])

    def pose(self) -> Pose | None:
        """直近の 6DoF ポーズ(未取得なら None)。"""
        return self.reader.get_latest()

    def position(self) -> tuple[float, float, float] | None:
        """現在位置 (x, y, z) [m]。"""
        pose = self.pose()
        return pose.position if pose else None

    def xz(self) -> tuple[float, float] | None:
        """現在位置の水平成分 (x, z) [m]。"""
        pose = self.pose()
        return (pose.position[0], pose.position[2]) if pose else None

    def yaw(self) -> float | None:
        """現在の yaw [deg](+Z 基準)。"""
        pose = self.pose()
        return pose.yaw_deg if pose else None

    def pitch(self) -> float | None:
        """現在の pitch [deg](上向きが正)。"""
        pose = self.pose()
        return pose.pitch_deg if pose else None

    def distance_to(self, target: Iterable[float]) -> float | None:
        """target((x,z) または (x,y,z))までの水平距離[m]。ポーズ未取得なら None。"""
        cur = self.xz()
        if cur is None:
            return None
        tx, tz = self._xz_of(target)
        return math.hypot(tx - cur[0], tz - cur[1])

    def is_near(self, target: Iterable[float], radius: float) -> bool:
        """target から radius [m] 以内にいるか(ポーズ未取得なら False)。"""
        d = self.distance_to(target)
        return d is not None and d <= radius

    def bearing_to(self, target: Iterable[float]) -> float | None:
        """target を向くのに必要な yaw [deg](+Z 基準の絶対方位)。"""
        cur = self.xz()
        if cur is None:
            return None
        tx, tz = self._xz_of(target)
        return math.degrees(math.atan2(tx - cur[0], tz - cur[1]))

    def yaw_error_to(self, target: Iterable[float]) -> float | None:
        """target への yaw 誤差[deg](最短回り、+で右)。"""
        pose = self.pose()
        if pose is None:
            return None
        cur = (pose.position[0], pose.position[2])
        err, _ = heading_error(cur, pose.yaw_deg, self._xz_of(target))
        return err

    def pitch_error_to(self, target: tuple[float, float, float]) -> float | None:
        """target への pitch 誤差[deg](+はもっと上を向く必要)。"""
        pose = self.pose()
        if pose is None:
            return None
        return pitch_error(pose.position, pose.forward, target)

    def stats(self):
        """PoseReader の統計スナップショット(注入 reader が未対応なら None)。"""
        get = getattr(self.reader, "get_stats", None)
        return get() if callable(get) else None

    def wait_until_hud(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                return False
            if self.reader.get_latest() is not None:
                return True
            time.sleep(0.1)
        logger.warning(
            "no HUD for %.0fs (VRChat running? HUD_Enable=true? wrong window?)", timeout
        )
        return False

    def is_hud_alive(self, timeout: float = 1.0) -> bool:
        """新規フレームが timeout 秒以内に来るかを確認する(ブロッキング)。

        wait_until_hud と違い「一度でも読めたか」ではなく「今も更新されているか」を見る。
        長時間運転でメニュー開放・ウィンドウ消失を検知する用。
        """
        pose = self.reader.get_latest()
        last = pose.time_ms if pose else None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            p = self.reader.get_latest()
            if p is not None and p.time_ms != last:
                return True
            time.sleep(0.02)
        return False

    def wait_until(
        self,
        predicate: Callable[[], bool],
        timeout: float,
        *,
        poll: float = 0.05,
    ) -> bool:
        """predicate が真になるまで待つ(タイムアウト・cancel() で False)。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._cancel.is_set():
            if predicate():
                return True
            time.sleep(poll)
        return False

    def wait_until_near(
        self, target: Iterable[float], radius: float, timeout: float
    ) -> bool:
        """target から radius [m] 以内に入るまで待つ(外部要因で運ばれる時など)。"""
        t = tuple(target)
        return self.wait_until(lambda: self.is_near(t, radius), timeout)

    # ---- 幾何ヘルパ ----------------------------------------------------
    def standoff_point(
        self,
        xyz: tuple[float, float, float],
        face_yaw_deg: float,
        dist: float | None = None,
    ) -> tuple[float, float]:
        """目標の正面 dist [m](省略時 gains.standoff)に立つ位置の XZ。"""
        d = self.gains.standoff if dist is None else dist
        return _standoff_point(xyz, face_yaw_deg, d)

    def face_yaw_to(self, xyz: Iterable[float]) -> float | None:
        """現在位置から見た目標面の法線方向(目標→現在地の方位)[deg]。

        目標の向きが事前に分からない時、standoff_point / visit の face_yaw_deg に
        渡すと「今いる側の正面」に立てる。ポーズ未取得なら None。
        """
        cur = self.xz()
        if cur is None:
            return None
        tx, tz = self._xz_of(xyz)
        return math.degrees(math.atan2(cur[0] - tx, cur[1] - tz))

    # ---- 経路計画(dry-run) --------------------------------------------
    def plan(
        self,
        xz: tuple[float, float],
        *,
        start: tuple[float, float] | None = None,
    ) -> Path | None:
        """動かずに経路だけ計画する(到達不能・ポーズ未取得なら None)。

        start 省略時は現在位置。マップ範囲外の目標は plan_path 同様 ValueError。
        """
        if start is None:
            start = self.xz()
            if start is None:
                return None
        return plan_path(self.grid, start, xz)

    def can_reach(
        self,
        xz: tuple[float, float],
        *,
        start: tuple[float, float] | None = None,
    ) -> bool:
        """xz へ到達できるか(範囲外・ポーズ未取得も False。動かない)。"""
        try:
            return self.plan(xz, start=start) is not None
        except ValueError:
            return False

    def path_length(
        self,
        xz: tuple[float, float],
        *,
        start: tuple[float, float] | None = None,
    ) -> float | None:
        """xz までの経路長[m](到達不能なら None。動かない)。"""
        path = self.plan(xz, start=start)
        return path.length if path else None

    # ---- 中断 ----------------------------------------------------------
    def cancel(self) -> None:
        """実行中の maneuver を中断する(スレッドセーフ。以後の指令も止まる)。

        中断されたループは reason="cancelled" で返る。再開する時は resume()。
        即時のアクチュエータ停止(開ループ)は stop() を使う。
        """
        self._cancel.set()

    def resume(self) -> None:
        """cancel() の解除。以後の maneuver は通常どおり動く。"""
        self._cancel.clear()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ---- 移動(act) ----------------------------------------------------
    def goto(self, xz: tuple[float, float], *, name: str = "goto") -> NavResult:
        pose = self.reader.get_latest()
        if pose is None:
            logger.warning("[%s] no current pose (HUD?)", name)
            return NavResult(False, False, "no_pose", None, 0.0, 0)
        start = (pose.position[0], pose.position[2])
        path = plan_path(self.grid, start, xz)
        if path is None:
            logger.warning("[%s] no path (unreachable)", name)
            return NavResult(False, False, "unreachable", None, 0.0, 0)
        logger.info(
            "[%s] path %dwp / %.1fm%s",
            name,
            len(path.waypoints),
            path.length,
            " (goal on wall -> nearest floor)" if path.goal_blocked else "",
        )
        res = follow_path(
            self.reader,
            self.look,
            self.move,
            path.waypoints,
            self.gains,
            self.nav,
            recorder=self.recorder,
            cancel=self._cancel,
            name=name,
        )
        res.path = path
        return res

    def translate_to(
        self, xz: tuple[float, float], *, name: str = "translate"
    ) -> NavResult:
        """視点を回さず xz へ並進する(壁回避は goto と同じ plan_path)。

        前後+横移動で経路を追うため、横に曲がる経路では goto より遅い。
        """
        pose = self.reader.get_latest()
        if pose is None:
            logger.warning("[%s] no current pose (HUD?)", name)
            return NavResult(False, False, "no_pose", None, 0.0, 0)
        start = (pose.position[0], pose.position[2])
        path = plan_path(self.grid, start, xz)
        if path is None:
            logger.warning("[%s] no path (unreachable)", name)
            return NavResult(False, False, "unreachable", None, 0.0, 0)
        logger.info(
            "[%s] path %dwp / %.1fm (view locked)%s",
            name,
            len(path.waypoints),
            path.length,
            " (goal on wall -> nearest floor)" if path.goal_blocked else "",
        )
        res = follow_path_translate(
            self.reader,
            self.look,
            self.move,
            path.waypoints,
            self.gains,
            self.translate,
            recorder=self.recorder,
            cancel=self._cancel,
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
            cancel=self._cancel,
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
            cancel=self._cancel,
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
            cancel=self._cancel,
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
            cancel=self._cancel,
            name=name,
        )

    # ---- 開ループ操作 --------------------------------------------------
    def stop(self) -> None:
        """移動・視点指令を即座に止める(緊急停止。maneuver の中断は cancel())。"""
        try:
            self.look.stop()
        finally:
            self.move.stop()

    def move_for(
        self, duration: float, *, forward: float = 0.0, strafe: float = 0.0
    ) -> None:
        """開ループの時間指定移動(経路計画・帰還なし。段差越え等の細かい操作用)。

        cancel() で中断できる。終了時は必ず停止する。
        """
        deadline = time.monotonic() + duration
        try:
            while time.monotonic() < deadline and not self._cancel.is_set():
                # /input 軸は保持されるが、パケット取りこぼし対策で定期再送する
                self.move.move(forward=forward, strafe=strafe)
                time.sleep(0.05)
        finally:
            self.move.stop()

    # ---- 押下(interact) -----------------------------------------------
    def _require_interact(self) -> InteractActuator:
        if self.interact is None:
            raise RuntimeError(
                "no InteractActuator configured (use Pilot.connect or pass interact=)"
            )
        return self.interact

    def press(self) -> None:
        self._require_interact().press()

    def release(self) -> None:
        self._require_interact().release()

    def click(self) -> None:
        self._require_interact().click()

    # ---- 複合(移動+照準+押下) ---------------------------------------
    def _aim_sequence(self, xyz: tuple[float, float, float], name: str) -> AimResult:
        """aim → (align_tol > 0 なら)align の照準シーケンス。"""
        aim = self.aim(xyz, name=name)
        logger.info(
            "[%s] aim yaw_err=%+.2f° pitch_err=%+.2f° (%s)",
            name,
            aim.yaw_err,
            aim.pitch_err,
            aim.reason,
        )
        if self.gains.align_tol > 0.0:
            aim = self.align(xyz, name=name)
            logger.info(
                "[%s] align yaw_err=%+.2f° pitch_err=%+.2f° (%s)",
                name,
                aim.yaw_err,
                aim.pitch_err,
                aim.reason,
            )
        return aim

    def approach(
        self,
        xyz: tuple[float, float, float],
        face_yaw_deg: float,
        *,
        name: str = "button",
        standoff: float | None = None,
    ) -> tuple[NavResult, AimResult | None]:
        """正面へ移動して照準するだけ(押さない)。押下まで一括なら activate()。"""
        nav = self.goto(self.standoff_point(xyz, face_yaw_deg, standoff), name=name)
        if not nav.path_found:
            return nav, None
        return nav, self._aim_sequence(xyz, name)

    def click_at(
        self, xyz: tuple[float, float, float], *, name: str = "click_at"
    ) -> ClickAtResult:
        """その場で照準(aim/align)し、収束したら click する(移動しない)。"""
        self._require_interact()
        aim = self._aim_sequence(xyz, name)
        if not aim.converged:
            return ClickAtResult(aim, False)
        self.click()
        return ClickAtResult(aim, True)

    def activate(
        self,
        xyz: tuple[float, float, float],
        face_yaw_deg: float,
        *,
        name: str = "button",
        standoff: float | None = None,
    ) -> ActivateResult:
        """正面へ移動 → 照準 → 押下 の一連(ボタン1個を押し切る最上位API)。"""
        self._require_interact()
        nav, aim = self.approach(xyz, face_yaw_deg, name=name, standoff=standoff)
        if aim is None or not aim.converged:
            return ActivateResult(nav, aim, False)
        self.click()
        return ActivateResult(nav, aim, True)

    def patrol(
        self,
        targets: Iterable[tuple[str, tuple[float, float, float], float]],
    ) -> list[tuple[str, NavResult, AimResult | None]]:
        results = []
        for name, xyz, face_yaw in targets:
            if self.cancelled:
                break
            nav, aim = self.approach(xyz, face_yaw, name=name)
            results.append((name, nav, aim))
        return results

    # ---- ライフサイクル ------------------------------------------------
    def close(self) -> None:
        try:
            self.look.stop()
        except Exception:
            pass
        try:
            self.move.stop()
        except Exception:
            pass
        if self._owns_io:
            if self._osc is not None:
                self._osc.close()
            self.reader.stop()

    def __enter__(self) -> Pilot:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
