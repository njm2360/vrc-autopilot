import csv

FIELDS = [
    "t",  # 開始からの経過秒
    "phase",  # nav / face / align
    "target",  # ターゲット名
    "wp",  # 追従中のウェイポイント番号(navのみ)
    "dt",  # 前フレームからの実経過秒
    "x",
    "y",
    "z",
    "yaw",
    "pitch",
    "tx",
    "ty",
    "tz",
    "dist",  # ターゲットまでの水平距離[m]
    "yaw_err",
    "pitch_err",
    "lat_err",  # 横方向誤差[m](alignのみ。+なら目標が右)
    "turn_p",
    "turn_i",
    "turn_d",
    "turn",  # yaw(LookHorizontal)PID内訳と出力
    "pitch_p",
    "pitch_i",
    "pitch_d",
    "pitch_cmd",
    "strafe_p",
    "strafe_i",
    "strafe_d",
    "strafe",  # Horizontal(横移動)PID内訳と出力(alignのみ)
    "fwd",  # Vertical(前進)出力
    "fwd_factor",  # 向きズレによる前進減衰係数
]


class ControlLog:
    def __init__(self, path):
        self.path = path
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=FIELDS, extrasaction="ignore")
        self._w.writeheader()

    def row(self, **kw) -> None:
        self._w.writerow({k: kw.get(k, "") for k in FIELDS})
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
