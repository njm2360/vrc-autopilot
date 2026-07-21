from collections.abc import Callable
from typing import NamedTuple

from vrc_autopilot import NavGrid, NavResult, Pilot, RoomMapper


class Stop(NamedTuple):
    move: Callable[..., NavResult]  # pilot.goto / pilot.translate_to
    goal: tuple[float, float]
    buttons: list[tuple[float, float, float]]

    @property
    def pitch_hint(self) -> tuple[float, float, float] | None:
        return self.buttons[0] if self.buttons else None


MAP = "room.npz"

BTN_AUTOPLAY = (7.740, 7.405, 24.659)  # オートプレイ1h
BTN_RLT_FAST = (7.740, 7.405, 23.488)  # ルレ高速1h
BTN_RLT_X25 = (7.740, 6.834, 21.505)  # ルレx25
BTN_QVPEN = (-7.740, 7.248, 18.807)  # QvPenオフ
BTN_MEMORIAL = (-19.870, 7.404, 24.229)  # 記念アイテムオフ

SPOT_AUTO_BUY = (6.43, 24.09)
WEST_HUB = (-6.740, 16.716)
X25_YAW = -45.0
QVPEN_YAW = 90.0
MEMORIAL_YAW = 45.0


def build_route(pilot: Pilot) -> list[Stop]:
    standoff = pilot.standoff_point
    return [
        # オート購入
        Stop(pilot.goto, SPOT_AUTO_BUY, [BTN_AUTOPLAY, BTN_RLT_FAST]),
        # ルレx25
        Stop(pilot.goto, standoff(BTN_RLT_X25, X25_YAW), [BTN_RLT_X25]),
        # 西壁ハブ
        Stop(
            pilot.goto,
            WEST_HUB,
            [
                (-7.740, 7.313, 15.655),  # ログピックアップ
                (-7.740, 7.212, 16.716),  # 効果音
                (-7.740, 7.008, 16.716),  # 通知系サウンド
                (-7.740, 6.812, 16.716),  # BGM
                (-7.740, 7.212, 17.912),  # ポップアップ
                (-7.740, 8.040, 17.912),  # 動画プレイヤー
            ],
        ),
        # QvPen
        Stop(pilot.translate_to, standoff(BTN_QVPEN, QVPEN_YAW), [BTN_QVPEN]),
        # 記念アイテム
        Stop(pilot.goto, standoff(BTN_MEMORIAL, MEMORIAL_YAW), [BTN_MEMORIAL]),
    ]


def main() -> None:
    grid = NavGrid.from_mapper(RoomMapper.load(MAP))
    with Pilot.connect(grid) as pilot:
        pilot.wait_until_hud()
        pilot.wait_until_active()
        for stop in build_route(pilot):
            nav = stop.move(stop.goal, pitch_at=stop.pitch_hint)
            if nav.arrived:
                for xyz in stop.buttons:
                    pilot.click_at(xyz)


if __name__ == "__main__":
    main()
