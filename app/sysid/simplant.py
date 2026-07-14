"""同定済みプラントの模擬(実機なしのゲイン検証用)。

sysid.PlantModel(軸ごとの静特性+むだ時間、フレーム間隔 dt 列)を積分して
6DoF ポーズを生成する。PoseSource / LookActuator / MoveActuator をすべて
満たすので、maneuvers の制御ループ(turn_to / aim_at / follow_path)へ
そのまま注入できる。

start_realtime() でバックグラウンドスレッドが dt 間隔で実時間ステップし、制御
ループを無改造で回せる(ループが実時間で測る dt とシミュレータの dt が一致する)。
テストや高速な探索では step() を手動で呼ぶ非実時間モードを使う。

モデル化しているのは静特性・むだ時間・フレーム間隔のみ。視点軸の時間方向の
平滑化(ランプ)・衝突・デコードノイズは含まない(プローブのステップ応答で
顕著なら要拡張)。
"""

import itertools
import math
import threading
from collections import deque

from ..control.guidance import wrap180
from ..core.pose import Pose
from .identify import PlantModel


class _DelayedCommand:
    """むだ時間つきの指令値。value(t) は時刻 t-deadtime までに設定された最新値。"""

    def __init__(self, deadtime_s: float):
        self.dead = deadtime_s
        self._hist: deque[tuple[float, float]] = deque([(-math.inf, 0.0)])

    def set(self, t: float, v: float) -> None:
        self._hist.append((t, float(v)))

    def value(self, t: float) -> float:
        cutoff = t - self.dead
        while len(self._hist) > 1 and self._hist[1][0] <= cutoff:
            self._hist.popleft()
        return self._hist[0][1]


class SimulatedVRChat:
    """PlantModel を積分する模擬 VRChat(PoseSource + LookActuator + MoveActuator)。

    stop() はアクチュエータ規約どおり「軸を0に戻す」。実時間スレッドの終了は
    close()(コンテキストマネージャでも可)。
    """

    def __init__(
        self,
        model: PlantModel,
        *,
        x: float = 0.0,
        y: float = 1.5,
        z: float = 0.0,
        yaw: float = 0.0,
        pitch: float = 0.0,
        use_dt_seq: bool = True,
    ):
        self.model = model
        self._x, self._y, self._z = x, y, z
        self._yaw, self._pitch = yaw, pitch
        self.now = 0.0
        self._time_ms = 0
        seq = model.dt_seq if (use_dt_seq and model.dt_seq) else []
        if not seq:
            seq = [model.dt_mean if model.dt_mean > 0.0 else 0.05]
        self._dts = itertools.cycle(seq)
        self._pending_dt: float | None = None
        self._cmds = {
            a: _DelayedCommand(model.axes[a].deadtime_s if a in model.axes else 0.0)
            for a in ("yaw", "pitch", "forward", "strafe")
        }
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._halt = threading.Event()
        self._pose = self._make_pose()

    # ---- PoseSource ------------------------------------------------------
    def get_latest(self) -> Pose | None:
        with self._lock:
            return self._pose

    # ---- LookActuator / MoveActuator -------------------------------------
    def look(self, turn: float = 0.0, pitch: float = 0.0) -> None:
        with self._lock:
            self._cmds["yaw"].set(self.now, turn)
            self._cmds["pitch"].set(self.now, pitch)

    def move(self, forward: float = 0.0, strafe: float = 0.0) -> None:
        with self._lock:
            self._cmds["forward"].set(self.now, forward)
            self._cmds["strafe"].set(self.now, strafe)

    def stop(self) -> None:
        self.look(0.0, 0.0)
        self.move(0.0, 0.0)

    # ---- ステップ実行 -----------------------------------------------------
    def next_frame_time(self) -> float:
        """次フレームのシミュレータ時刻(step() 前に覗ける。模擬時計用)。"""
        with self._lock:
            if self._pending_dt is None:
                self._pending_dt = next(self._dts)
            return self.now + self._pending_dt

    def step(self, dt: float | None = None) -> Pose:
        """1フレーム進める。dt 省略時は dt 列(なければ dt_mean)に従う。"""
        with self._lock:
            if dt is None:
                if self._pending_dt is None:
                    self._pending_dt = next(self._dts)
                dt = self._pending_dt
            self._pending_dt = None
            t = self.now
            wy = self._rate("yaw", t)
            wp = self._rate("pitch", t)
            vf = self._rate("forward", t)
            vs = self._rate("strafe", t)
            self._yaw = wrap180(self._yaw + wy * dt)
            self._pitch = max(-89.0, min(89.0, self._pitch + wp * dt))
            yr = math.radians(self._yaw)
            fx, fz = math.sin(yr), math.cos(yr)  # 前方向
            rx, rz = math.cos(yr), -math.sin(yr)  # 右方向
            self._x += (fx * vf + rx * vs) * dt
            self._z += (fz * vf + rz * vs) * dt
            self.now = t + dt
            self._time_ms += max(1, round(dt * 1000.0))
            self._pose = self._make_pose()
            return self._pose

    def _rate(self, axis: str, t: float) -> float:
        m = self.model.axes.get(axis)
        return m.rate(self._cmds[axis].value(t)) if m else 0.0

    def _make_pose(self) -> Pose:
        yr = math.radians(self._yaw)
        pr = math.radians(self._pitch)
        cp, sp = math.cos(pr), math.sin(pr)
        fwd = (cp * math.sin(yr), sp, cp * math.cos(yr))
        up = (-sp * math.sin(yr), cp, -sp * math.cos(yr))
        return Pose(
            time_ms=self._time_ms,
            position=(self._x, self._y, self._z),
            forward=fwd,
            up=up,
        )

    # ---- 実時間駆動 -------------------------------------------------------
    def start_realtime(self) -> "SimulatedVRChat":
        """dt 間隔で実時間ステップするバックグラウンドスレッドを開始する。"""
        if self._thread and self._thread.is_alive():
            return self
        self._halt.clear()
        self._thread = threading.Thread(
            target=self._run, name="SimulatedVRChat", daemon=True
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        while True:
            dt = self.next_frame_time() - self.now
            if self._halt.wait(dt):
                return
            self.step()

    def close(self) -> None:
        self._halt.set()
        if self._thread:
            self._thread.join(2.0)
            self._thread = None

    def __enter__(self) -> "SimulatedVRChat":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SimClock:
    def __init__(self, sim: SimulatedVRChat):
        self.sim = sim
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        target = self.t + seconds
        while self.sim.next_frame_time() <= target:
            self.t = self.sim.next_frame_time()
            self.sim.step()
        self.t = target
