import logging
import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from app.control.maneuvers import NavResult
from app.control.osc import VRChatOSC
from app.control.pilot import Pilot
from app.control.recording import ControlLog
from app.mapping.mapper import RoomMapper
from app.spatial.navigation import NavGrid

logging.basicConfig(level=logging.INFO)


class Stop(NamedTuple):
    name: str
    move: Callable[..., NavResult]  # pilot.goto / pilot.translate_to
    goal: tuple[float, float]
    buttons: list[tuple[str, tuple[float, float, float]]]


MAP = "room.npz"

BTN_AUTOPLAY = ("オートプレイ1h", (7.740, 7.405, 24.659))
BTN_RLT_FAST = ("ルレ高速1h", (7.740, 7.405, 23.488))
BTN_RLT_X25 = ("ルレx25", (7.740, 6.834, 21.505))
BTN_QVPEN = ("QvPenオフ", (-7.740, 7.248, 18.807))
BTN_MEMORIAL = ("記念アイテムオフ", (-19.870, 7.404, 24.229))

SPOT_AUTO_BUY = (6.43, 24.09)
WEST_HUB = (-6.740, 16.716)
X25_YAW = -45.0
QVPEN_YAW = 90.0
MEMORIAL_YAW = 45.0


def build_route(pilot: Pilot) -> list[Stop]:
    def standoff(
        xyz: tuple[float, float, float], face_yaw: float
    ) -> tuple[float, float]:
        r = math.radians(face_yaw)
        d = pilot.gains.standoff
        return (xyz[0] + math.sin(r) * d, xyz[2] + math.cos(r) * d)

    return [
        Stop("オート購入", pilot.goto, SPOT_AUTO_BUY, [BTN_AUTOPLAY, BTN_RLT_FAST]),
        Stop("ルレx25", pilot.goto, standoff(BTN_RLT_X25[1], X25_YAW), [BTN_RLT_X25]),
        Stop(
            "西壁ハブ",
            pilot.goto,
            WEST_HUB,
            [
                ("ログピックアップ", (-7.740, 7.313, 15.655)),
                ("効果音", (-7.740, 7.212, 16.716)),
                ("通知系サウンド", (-7.740, 7.008, 16.716)),
                ("BGM", (-7.740, 6.812, 16.716)),
                ("ポップアップ", (-7.740, 7.212, 17.912)),
                ("動画プレイヤー", (-7.740, 8.040, 17.912)),
            ],
        ),
        Stop(
            "QvPen", pilot.translate_to, standoff(BTN_QVPEN[1], QVPEN_YAW), [BTN_QVPEN]
        ),
        Stop(
            "記念アイテム",
            pilot.goto,
            standoff(BTN_MEMORIAL[1], MEMORIAL_YAW),
            [BTN_MEMORIAL],
        ),
    ]


def press(
    pilot: Pilot, interact: VRChatOSC, name: str, xyz: tuple[float, float, float]
) -> str:
    aim = pilot.aim(xyz, name=name)
    if pilot.gains.align_tol > 0.0:
        aim = pilot.align(xyz, name=name)
    if not aim.converged:
        return f"skipped ({aim.reason})"
    interact.click()
    return "clicked"


def main() -> None:
    grid = NavGrid.from_mapper(RoomMapper.load(MAP))
    log_path = Path(f"logs/buttons_{datetime.now():%Y%m%d_%H%M%S}.csv")
    log = ControlLog(log_path)
    skipped = 0
    try:
        interact = VRChatOSC()
        with Pilot.connect(grid, recorder=log) as pilot:
            pilot.wait_for_hud()
            for stop in build_route(pilot):
                nav = stop.move(stop.goal, name=stop.name)
                for name, xyz in stop.buttons:
                    if nav.arrived:
                        outcome = press(pilot, interact, name, xyz)
                    else:
                        outcome = f"skipped ({nav.reason})"
                    skipped += outcome != "clicked"
                    print(f"{name}: {outcome}", flush=True)
    finally:
        log.close()
    if skipped:
        print(f"{skipped} skipped")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
