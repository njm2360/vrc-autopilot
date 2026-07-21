import logging
import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..core.pose import Pose
from ..perception.capture import WindowFocus
from ..perception.reader import ReaderStats
from ..perception.spec import HUD_ENABLE_PARAM
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
from .osc import VRChatOSC
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
    """VRChat 内で移動・照準・押下をまとめて呼び出すためのクラス。

    現在位置と向きは reader(6DoF ポーズ)から読み、look(視点)・move(移動)・
    interact(押下)の各アクチュエータへ指令を出す。実機の I/O ごと組むなら
    connect()、テストや部品差し替えは __init__ に直接渡す。

    position() や distance_to() などの状態クエリは動かずに現在値を返す。
    goto()/aim()/activate() などの操作はブロッキングで、別スレッドから cancel()
    すると reason="cancelled" で抜ける。
    """

    def __init__(
        self,
        grid: NavGrid,
        reader: PoseSource,
        look: LookActuator,
        move: MoveActuator,
        *,
        interact: InteractActuator | None = None,
        focus: WindowFocus | None = None,
        gains: PatrolGains | None = None,
        world_cal: WorldCalibration | str | None = None,
        recorder: Recorder | None = None,
    ):
        """各アクチュエータと reader を直接渡す注入版(実機 I/O 込みは connect())。

        interact 省略時は押下系(press/click/activate)が RuntimeError になる。
        world_cal を渡すと gains に速度スケールを反映する。
        """
        self.grid = grid
        self.reader = reader
        self.look = look
        self.move = move
        self.interact = interact
        self._focus = focus
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
        self._osc: VRChatOSC | None = None
        self._owns_io = False
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
        osc: VRChatOSC | None = None,
        gains: PatrolGains | None = None,
        world_cal: WorldCalibration | str | None = None,
        look: LookActuator | None = None,
        interact: InteractActuator | None = None,
        recorder: Recorder | None = None,
    ) -> Pilot:
        """実機 I/O(キャプチャ+OSC)を組んだ Pilot を作る(注入版は __init__)。

        osc を渡すと host/port を変えられる(省略時は 127.0.0.1:9000)。
        look / interact を渡すと視点・押下だけ差し替えられる。省略時はどれも OSC。
        world_cal は calibrate-world の JSON パス(またはロード済み WorldCalibration)。
        """
        from ..perception.capture import WindowsVRChatCapture
        from ..perception.reader import PoseReader

        capture = WindowsVRChatCapture()
        reader = PoseReader(source=capture).start()
        osc = osc or VRChatOSC()
        osc.avatar_param(HUD_ENABLE_PARAM, True)
        osc.set_run(True)
        try:
            pilot = cls(
                grid,
                reader,
                look or osc,
                osc,
                interact=interact or osc,
                focus=capture,
                gains=gains,
                world_cal=world_cal,
                recorder=recorder,
            )
        except BaseException:
            osc.close()
            reader.stop()
            raise
        pilot._osc = osc
        pilot._owns_io = True
        return pilot

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

    def stats(self) -> ReaderStats | None:
        """PoseReader の統計スナップショット(注入 reader が未対応なら None)。"""
        get = getattr(self.reader, "get_stats", None)
        return get() if callable(get) else None

    def wait_until_hud(self, timeout: float = 10.0) -> bool:
        """HUD(ポーズ)が最初に読めるまで待つ(timeout 超過・cancel() で False)。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                return False
            if self.reader.get_latest() is not None:
                return True
            time.sleep(0.1)
        logger.warning(
            "no HUD for %.0fs (VRChat running? %s=true? wrong window?)",
            timeout,
            HUD_ENABLE_PARAM,
        )
        return False

    def is_hud_alive(self, timeout: float = 1.0) -> bool:
        """新規フレームが timeout 秒以内に来るかを確認する(ブロッキング)。

        wait_until_hud が一度でも読めたかを見るのに対し、今も更新されているかを見る。
        長く走らせている間にメニューを開いたりウィンドウが消えたのを見つける用。
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

    def is_active(self) -> bool:
        """VRChatが最前面か。focus未注入ならTrue。"""
        return self._focus.is_active() if self._focus is not None else True

    def wait_until_active(self, timeout: float = 30.0) -> bool:
        """VRChatが最前面になるまで待つ。focus未注入ならTrue。"""
        if self._focus is None:
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                return False
            if self._focus.is_active():
                return True
            time.sleep(0.1)
        logger.warning(
            "VRChat not foreground for %.0fs (Alt+Tab to focus it?)", timeout
        )
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

        目標の向きが事前に分からない時、standoff_point / approach の face_yaw_deg に
        渡すと今いる側の正面に立てる。ポーズ未取得なら None。
        """
        cur = self.xz()
        if cur is None:
            return None
        tx, tz = self._xz_of(xyz)
        return math.degrees(math.atan2(cur[0] - tx, cur[1] - tz))

    # ---- マップ切替 ----------------------------------------------------
    def use_grid(self, grid: NavGrid) -> None:
        """以後の移動と経路計画で使う NavGrid を差し替える(階の切替用)。

        位置はワールド座標のまま、歩ける領域だけ変わる。切替点は呼び出し側で決める。
        """
        logger.info("grid switched: %dx%d cells", *grid.shape)
        self.grid = grid

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
        """cancel() 済みで、resume() するまで操作が中断される状態か。"""
        return self._cancel.is_set()

    # ---- 移動(act) ----------------------------------------------------
    def goto(
        self,
        xz: tuple[float, float],
        *,
        pitch_at: tuple[float, float, float] | None = None,
    ) -> NavResult:
        """xz へ経路計画して移動する(壁回避あり。視点は進行方向へ向く)。

        pitch_at((x,y,z))を渡すと、移動中にpitchを先合わせする
        """
        pose = self.reader.get_latest()
        if pose is None:
            logger.warning("[goto] no current pose (HUD?)")
            return NavResult(False, False, "no_pose", None, 0.0, 0)
        start = (pose.position[0], pose.position[2])
        path = plan_path(self.grid, start, xz)
        if path is None:
            logger.warning("[goto] no path (unreachable)")
            return NavResult(False, False, "unreachable", None, 0.0, 0)
        logger.info(
            "[goto] path %dwp / %.1fm%s",
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
            pitch_target=pitch_at,
            recorder=self.recorder,
            cancel=self._cancel,
        )
        res.path = path
        return res

    def translate_to(
        self,
        xz: tuple[float, float],
        *,
        pitch_at: tuple[float, float, float] | None = None,
    ) -> NavResult:
        """視点を回さず xz へ並進する(壁回避は goto と同じ plan_path)。

        前後+横移動で経路を追うため、横に曲がる経路では goto より遅い。
        pitch_at((x,y,z))を渡すと、移動中にpitchを先合わせする。
        """
        pose = self.reader.get_latest()
        if pose is None:
            logger.warning("[translate] no current pose (HUD?)")
            return NavResult(False, False, "no_pose", None, 0.0, 0)
        start = (pose.position[0], pose.position[2])
        path = plan_path(self.grid, start, xz)
        if path is None:
            logger.warning("[translate] no path (unreachable)")
            return NavResult(False, False, "unreachable", None, 0.0, 0)
        logger.info(
            "[translate] path %dwp / %.1fm (view locked)%s",
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
            pitch_target=pitch_at,
            recorder=self.recorder,
            cancel=self._cancel,
        )
        res.path = path
        return res

    def follow(
        self,
        waypoints: Iterable[tuple[float, float]],
        *,
        pitch_at: tuple[float, float, float] | None = None,
    ) -> NavResult:
        """与えた waypoints をそのまま追従する(経路計画なし。goto の低レベル版)。

        pitch_at((x,y,z))を渡すと、移動中にpitchを先合わせする。
        """
        return follow_path(
            self.reader,
            self.look,
            self.move,
            list(waypoints),
            self.gains,
            self.nav,
            pitch_target=pitch_at,
            recorder=self.recorder,
            cancel=self._cancel,
        )

    def aim(self, xyz: tuple[float, float, float]) -> AimResult:
        """target(x,y,z)へ視点(yaw/pitch)を向ける(体は動かさない)。"""
        return aim_at(
            self.reader,
            self.look,
            xyz,
            self.gains,
            self.face,
            recorder=self.recorder,
            cancel=self._cancel,
        )

    def align(self, xyz: tuple[float, float, float]) -> AimResult:
        """視点は回さず、体の横移動で target への横ずれを詰める(最終照準)。"""
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
        )

    def turn_to(self, yaw_deg: float, pitch_deg: float | None = None) -> AimResult:
        """指定した yaw(必要なら pitch)へ視点だけ回す(座標でなく角度で指定)。"""
        return turn_to(
            self.reader,
            self.look,
            yaw_deg,
            self.gains,
            self.face,
            pitch_deg=pitch_deg,
            recorder=self.recorder,
            cancel=self._cancel,
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
        """押しっぱなしにする(離すのは release。interact 未設定なら RuntimeError)。"""
        self._require_interact().press()

    def release(self) -> None:
        """press した押下を離す。"""
        self._require_interact().release()

    def click(self) -> None:
        """一回押して離す(interact 未設定なら RuntimeError)。"""
        self._require_interact().click()

    # ---- 複合(移動+照準+押下) ---------------------------------------
    def _aim_sequence(self, xyz: tuple[float, float, float]) -> AimResult:
        """aim → (align_tol > 0 なら)align の照準シーケンス。"""
        aim = self.aim(xyz)
        logger.info(
            "[aim] yaw_err=%+.2f° pitch_err=%+.2f° (%s)",
            aim.yaw_err,
            aim.pitch_err,
            aim.reason,
        )
        if self.gains.align_tol > 0.0:
            aim = self.align(xyz)
            logger.info(
                "[align] yaw_err=%+.2f° pitch_err=%+.2f° (%s)",
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
        standoff: float | None = None,
    ) -> tuple[NavResult, AimResult | None]:
        """正面へ移動して照準するだけ(押さない)。押下まで一括なら activate()。

        移動中に xyz へ pitch を先合わせするので、到着後の照準は主に yaw で済む。
        """
        nav = self.goto(self.standoff_point(xyz, face_yaw_deg, standoff), pitch_at=xyz)
        if not nav.path_found:
            return nav, None
        return nav, self._aim_sequence(xyz)

    def click_at(self, xyz: tuple[float, float, float]) -> ClickAtResult:
        """その場で照準(aim/align)し、収束したら click する(移動しない)。"""
        self._require_interact()
        aim = self._aim_sequence(xyz)
        if not aim.converged:
            return ClickAtResult(aim, False)
        self.click()
        return ClickAtResult(aim, True)

    def activate(
        self,
        xyz: tuple[float, float, float],
        face_yaw_deg: float,
        *,
        standoff: float | None = None,
    ) -> ActivateResult:
        """正面へ移動 → 照準 → 押下 の一連(ボタン1個を押し切る最上位API)。"""
        self._require_interact()
        nav, aim = self.approach(xyz, face_yaw_deg, standoff=standoff)
        if aim is None or not aim.converged:
            return ActivateResult(nav, aim, False)
        self.click()
        return ActivateResult(nav, aim, True)

    # ---- ライフサイクル ------------------------------------------------
    def close(self) -> None:
        """アクチュエータを止め、connect() で確保した I/O を閉じる。"""
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
