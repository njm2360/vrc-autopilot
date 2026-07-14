# dekapu-switch-bot

VRChat の PoseTelemetryHUD(画面左上の白黒ビットグリッド)をスクリーンキャプチャで読み取り、
視点の 6DoF(グローバル座標 + 向き)を復元し、OSC で移動・視点を操作する VRChat 自動化ツール群。

- グリッド/プロトコルの仕様: [docs/pose-telemetry-hud-spec.md](docs/pose-telemetry-hud-spec.md)
- 全体像とモジュール対応: [docs/architecture.md](docs/architecture.md)

## セットアップ

```bash
uv sync --group dev        # numpy, mss, python-osc, matplotlib, pytest
uv run pytest              # 合成画像でのラウンドトリップ検証(実VRChat不要)
```

前提(ライブ実行時): Windows で VRChat を**ネイティブ解像度・ウィンドウ表示**で起動し、
OSC で `/avatar/parameters/HUD_Enable = true`(HUD 表示 ON)にしておくこと。
ゲーム内メニューを開くと HUD が隠れて読めない(異常ではない)。

## CLI(console scripts)

`uv run <コマンド>` で実行する。実装は [app/cli/](app/cli/)。

| コマンド         | 用途                                                       |
| ---------------- | ---------------------------------------------------------- |
| `decode-demo`    | HUD を読み取り 6DoF を表示(動作確認・キャリブレーション)   |
| `map-room`       | 壁沿いに歩いて部屋マップを記録(引数なし・日時フォルダ出力) |
| `find-button`    | 複数地点からボタンを三角測量                               |
| `patrol-buttons` | マップ上でボタンを壁を避けて巡回(OSC + PID)                |

### `decode-demo` — 6DoF 表示

```bash
uv run decode-demo                 # 読み続けて表示(Ctrl+C で停止)
uv run decode-demo --stats         # fps・成功率などの統計も毎秒表示
```

出力: `pos=( +1.500, -2.250, +42.000)  yaw= +12.34  pitch= -5.67  t=123456`
(`pos` ワールド座標[m] / `yaw` +Z基準水平角[deg] / `pitch` 上向き正[deg] / `t` 同期時刻)。

グリッド定数(オフセット/ブロック)は [app/perception/spec.py](app/perception/spec.py) の確定値。変更する場合はシェーダー側と一致させること。

### `map-room` — 部屋マップ記録

壁沿いに歩き回った視点軌跡を床平面(XZ)へ投影して間取り図を作る。**歩いた経路そのものが
部屋の輪郭**になる。引数なし。

```bash
uv run map-room     # SPACE=一時停止/再開  q(or Ctrl+C)=保存して終了
```

**一時停止(SPACE)** は「壁伝いに一周できない壁」を撮るための機能。途中に障害物があって
壁から離れて別の壁区間へ移動する間だけ SPACE で止めれば、その移動が偽の壁として記録されない
(停止をまたぐ点どうしを線で繋がない=ペンアップ)。区間ごとに歩いて壁全体をカバーできる。

出力 `maps/<YYYYMMDD_HHMMSS>/`: `room.npz`(軌跡・セグメント)/ `room.json`(寸法・面積・
経路長・区間数)/ `room.png`(間取り図)。

### `find-button` — ボタン座標の三角測量

複数地点から「ボタンを画面中央(視点の正面)に捉えて」SPACE を押すと視線レイを1本記録する。
位置・角度を変えて2回以上押すと、レイの**最小二乗交点**=ボタン座標を stdout に表示する。

```bash
uv run find-button      # SPACE=キャプチャ  r=リセット(次のボタン)  q=終了
```

- 2地点の視線が**平行に近いと精度が出ない**。なるべく別方向から狙う(平行に近いと警告)。
- 3点あるとより安定。`residual`(各レイからのずれ)が小さいほど良い。

### `patrol-buttons` — 壁を避けてボタン巡回(OSC + PID)

保存した部屋マップを読み込み、指定座標のボタンへ壁を避けて移動する。歩行軌跡マップの内側=
歩ける床とみなし、行けない所(壁・部屋外)は A* で迂回する。移動・旋回は OSC(`/input/*`)で
注入し、位置は HUD デコードでフィードバックする。現状はボタンに到着して正対するまでで、
クリックそのものはまだ行わない。

```bash
# 計画のみ(VRChat不要)。--dry-run でマップと同じフォルダに plan.png を自動保存
uv run patrol-buttons --map maps/<日時>/room.npz --target 3.0,1.2,5.0,180 --target -1.0,1.0,2.0,90 --dry-run

# 実際に OSC で巡回(VRChat 起動 + HUD_Enable が必要)
uv run patrol-buttons --map maps/<日時>/room.npz --target 3.0,1.2,5.0,180 --target -1.0,1.0,2.0,90
```

- `--target X,Y,Z,FACE_YAW` … ボタン座標と向き(複数可, **高さ Y・向き FACE_YAW は必須**)。
  FACE_YAW はボタンの向き=壁の外向き法線[deg](+Z基準)。その正面 `--standoff`[m] に立って照準する
- `--radius 0.25` … アバター半径=壁クリアランス[m] / `--cell 0.1` … グリッド解像度[m]
- `--gap-close 0.3` … 軌跡ループの隙間を塞ぐ距離[m](歩き残しで外に漏れる時に増やす)
- 追従: `--speed`(巡航速度)、`--arrive`(WP到達半径)、`--face-tol`(正対とみなす角度[deg])、`--settle`
- PID ゲイン(**移動と正対で別**): 移動=`--nav-turn-kp/ki/kd`、正対=`--turn-kp/ki/kd`・`--pitch-kp/ki/kd`
- `--turn-deadzone 0.55` … 視点軸の不感帯補償(下記)。`--fwd-kp/kd` … 前進速度
- `--look osc|mouse` … 視点アクチュエータ。`mouse` は DirectInput 相対マウス(要 pydirectinput)。
  移動は OSC 固定。`--mouse-yaw-gain`/`--mouse-pitch-gain`[px/指令] で感度調整(**OSC とは別に要再調整**。
  マウスは不感帯が無いので `--turn-deadzone 0`)。

制御は経由点で止まらない**連続追従**。経路は**壁に遮られない範囲で直線に間引き**(ジグザグ除去)、
実測 dt でフィードバックする。移動中(nav)は穏やかな yaw ゲインで追従し、到着後(face)は
yaw/pitch を `--face-tol` 未満へ収束させる。

**視点軸の不感帯補償**: VRChat の `/input/LookHorizontal` は指令の絶対値が約0.55以下だとほとんど
反応しない。素の PID だと最後の数度がこの反応しない範囲の直下に張り付いて収束しない。
`--turn-deadzone`(既定0.55)で非ゼロ出力をこの値以上に底上げし、反応しない範囲を飛び越えて詰める
(移動中はそのまま自然な遊びとして残すため補償なし)。上下がなかなか合わない場合は `--pitch-deadzone 0.5` も有効。

**制御ログ**: `logs/patrol_<日時>.csv` に毎フレームの姿勢・誤差・PID内訳(P/I/D)・
OSC指令を書き出す。PID チューニングやぎこちなさの原因調査に使う。

到着後は yaw(水平)と pitch(上下, `/input/LookVertical`)を PID で合わせてボタンを正面に捉える。
高さ Y はこの pitch 合わせに使うため必須(`find-button` の座標 x,y,z をそのまま渡せる)。

## Python から使う

各機能はサブモジュールから直接 import する(`app` 直下に再エクスポートは無い)。

```python
from app.perception.reader import PoseReader
from app.mapping.mapper import RoomMapper
from app.spatial.navigation import NavGrid, plan_path
from app.control.osc import VRChatOSC

# 6DoF 読み取り
with PoseReader() as reader:                 # 既定で "VRChat" ウィンドウ
    for pose in reader.poses():
        print(pose.position, pose.yaw_deg, pose.pitch_deg)

# 経路計画 + OSC 操作
grid = NavGrid.from_mapper(RoomMapper.load("maps/room.npz"), avatar_radius=0.25)
path = plan_path(grid, start_xz=(1.0, 1.0), goal_xz=(3.0, 5.0))   # 壁を避けた経由点列
with VRChatOSC() as osc:                      # 127.0.0.1:9000
    osc.move(forward=1.0); osc.look(0.2)
```

非Windows/テストでは `app.perception.capture.ArrayFrameSource(frame)` を `PoseReader` に注入する。
モジュール対応とデータフローは [docs/architecture.md](docs/architecture.md) を参照。

## OSC 仕様(VRChat)

送信は [python-osc](https://pypi.org/project/python-osc/) を使い、既定で `127.0.0.1:9000`
(VRChat 受信、送信は 9001)。`app.control.osc.VRChatOSC` が薄いラッパ。
出典: VRChat OSC 公式ドキュメント(as-input-controller / osc-overview)。

- Axes(float −1..1、離したら0): `/input/Vertical`(+前/−後)、`/input/Horizontal`(+右/−左)、
  `/input/LookHorizontal`(+右旋回/−左)、`/input/LookVertical`(+上/−下)
- Buttons(int 0/1): `/input/Jump` ほか / アバター: `/avatar/parameters/HUD_Enable`

## 今後の課題

- ボタンの実クリック(OSC の Use 系操作)。
