# dekapu-switch-bot

VRChat の PoseTelemetryHUD(画面左上の白黒ビットグリッド)をスクリーンキャプチャで読み取り、
視点の 6DoF(グローバル座標 + 向き)を復元し、OSC で移動・視点を操作する VRChat 自動化ツール群。

- グリッド/プロトコルの仕様: [docs/pose-telemetry-hud-spec.md](docs/pose-telemetry-hud-spec.md)
- 全体像とモジュール対応: [docs/architecture.md](docs/architecture.md)
- プラント特性の測定手順: [docs/system-identification.md](docs/system-identification.md)
- 制御ゲインの根拠と調整手順: [docs/gain-tuning.md](docs/gain-tuning.md)
- オフライン検証の組み方(sim の回し方や指標、落とし穴): [docs/verification.md](docs/verification.md)

## セットアップ

Python 3.14 以上が必要。

```bash
uv sync --group dev        # numpy, scipy, mss, python-osc, matplotlib, pydirectinput, pytest
uv run pytest              # 208 テスト。合成画像のラウンドトリップから制御ループまで(実VRChat不要)
```

前提(ライブ実行時): VRChatをネイティブ解像度のウィンドウ表示で起動し、
OSCで`/avatar/parameters/HUD_Enable = true`(HUD表示ON)にしておくこと。
他のウィンドウ等でHUDのピクセルが隠れると正常に動作しない。

## CLI(console scripts)

`uv run <コマンド>` で実行する。実装は [app/cli/](app/cli/)、フラグ詳細は各 `--help`。

| コマンド          | 用途                                                                         |
| ----------------- | ---------------------------------------------------------------------------- |
| `decode-demo`     | HUD を読み取り 6DoF を表示(動作確認とキャリブレーション)                     |
| `map-room`        | 壁沿いに歩いて部屋マップを記録                                               |
| `find-button`     | 複数地点からボタンを三角測量                                                 |
| `patrol-buttons`  | マップ上でボタンを壁を避けて巡回(OSC + PID)                                  |
| `probe-axes`      | 入力軸の応答特性を測って `plant.json` に同定(制御ゲインの前提)               |
| `calibrate-world` | ワールドごとに変わる移動速度を測り、ゲインの倍率を補正                       |
| `sim-face`        | 同定プラント上で正対ループを回してゲインを検証(実機不要)                     |
| `bode-margins`    | 同定プラント上で全制御ループの安定余裕(ωc/PM/GM)とボード線図を出す(実機不要) |
| `log-video`       | 制御ログCSVを一人称3D+2D地図の動画(mp4)に再生                                |

視点軸の速度と不感帯はクライアント側の設定なのでワールドを移っても変わらないが、
移動速度はワールド依存になる。ワールドを移ったら `probe-axes` を全部やり直すのではなく、
`calibrate-world` で倍率だけ補正して `patrol-buttons --world-cal` に渡せばよい。
測定手順は [docs/system-identification.md](docs/system-identification.md)。

### `decode-demo` — 6DoF 表示

`pos`(ワールド座標[m])/ `yaw`(+Z基準[deg])/ `pitch`(上向き正[deg])/ `t`(同期時刻)を
読み続けて表示する(`--stats` を付けると fps と成功率も出る)。グリッド定数は
[app/perception/spec.py](app/perception/spec.py) の確定値。変更する場合はシェーダー側と一致させること。

### `map-room` — 部屋マップ記録

壁沿いに歩いた視点軌跡を床平面(XZ)へ投影して間取り図を作る(**歩いた経路そのものが部屋の輪郭**)。
SPACE の一時停止はペンアップ: 壁から離れて移動する間だけ止めれば偽の壁が記録されず、
壁を区間ごとに分けてカバーできる。出力は `maps/<日時>/` に room.npz / room.json / room.png。

### `find-button` — ボタン座標の三角測量

ボタンを画面中央に捉えて SPACE を押すと視線レイを記録し、2本以上の最小二乗交点=ボタン座標を表示する。
別方向から狙うほど精度が出る(平行に近いと警告)。

### `patrol-buttons` — 壁を避けてボタン巡回(OSC + PID)

部屋マップ上で指定座標のボタンへ A\* で壁を避けて移動し、正対まで行う。
移動・旋回は OSC、位置は HUD デコードでフィードバック。

```bash
# 計画のみ(VRChat不要)。plan.png をマップと同じフォルダに保存
uv run patrol-buttons --map maps/<日時>/room.npz --target 3.0,1.2,5.0,180 --dry-run

# 実際にOSCで巡回
uv run patrol-buttons --map maps/<日時>/room.npz --target 3.0,1.2,5.0,180 --target -1.0,1.0,2.0,90
```

- `--target X,Y,Z,FACE_YAW`(複数可): ボタン座標と向き。FACE_YAW は壁の外向き法線[deg](+Z基準)で、
  その正面 `--standoff`[m] に立って照準する。Y は pitch 合わせに必須(`find-button` の出力をそのまま渡せる)
- 制御の構成: 移動中(nav)は経路の少し先(`--nav-lookahead`)を狙う carrot 方式で連続追従し、
  到着後(face)に yaw/pitch を収束させる。視点軸は指令 0.50 以下が効かないため、
  不感帯補償(`--turn-deadzone` / `--nav-turn-deadzone`)で底上げする。
  ゲインの既定値と安定範囲、調整手順は [docs/gain-tuning.md](docs/gain-tuning.md)
- `--look mouse` で視点を DirectInput 相対マウスに切り替えられる。不感帯が無いので
  deadzone は 0 にし、ゲインは別物として再調整する
- `--world-cal` に `calibrate-world` の出力 JSON を渡すと、そのワールドの移動速度に
  合わせてゲインをスケールする
- 制御ログ: `logs/patrol_<日時>.csv` に毎フレームの姿勢と誤差、PID内訳、指令を書き出す
  (チューニングと原因調査用。`log-video` でそのまま再生できる)

### `log-video` — 制御ログを動画で再生

フレームCSV(実機の `logs/patrol_*.csv`またはsimの記録)を、一人称3D(レイキャスト)+
上面2D地図の並置mp4にする。経路への追従とぎこちなさの目視確認用。ffmpeg が PATH に必要。

```bash
uv run log-video --csv logs/patrol_<日時>.csv --map maps/<日時>/room.npz --out run.mp4
```

## プログラムから使う

高レベルAPIは `app.control.pilot.Pilot`(経路計画+移動+正対+最終照準を束ねる。
patrol-buttons CLI の中身と同一)。使い方は [examples/pilot_patrol.py](examples/pilot_patrol.py)、
モジュール対応は [docs/architecture.md](docs/architecture.md) を参照。

## OSC仕様

`app.control.osc.VRChatOSC` が薄いラッパ(python-osc、既定 127.0.0.1:9000)。
Axes(float −1..1、離したら0): `/input/Vertical` `/input/Horizontal` `/input/LookHorizontal`
`/input/LookVertical`。Buttons(int 0/1): `/input/Jump` `/input/Run` `/input/UseRight`。
アバター: `/avatar/parameters/HUD_Enable`

VRChat の `/input` 軸は最後に送った値を保持し続けるので、制御ループでは 0 を明示的に
送って止める必要がある(`VRChatOSC.look` が両軸とも毎回送っているのはこのため)。
