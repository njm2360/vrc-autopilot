# CLIの使用方法

各コマンドの使い方とフラグの補足。一覧は [README](../README.md)、フラグの網羅は各 `--help`。

## `decode-demo` — 6DoF 表示

`pos`(ワールド座標[m])/ `yaw`(+Z基準[deg])/ `pitch`(上向き正[deg])/ `t`(同期時刻)を読み続けて表示する(`--stats` を付けると fps と成功率も出る)。グリッド定数は [app/perception/spec.py](../app/perception/spec.py) の確定値。変更する場合はシェーダー側と一致させること。

## `map-room` — 部屋マップ記録

壁沿いに歩いた視点軌跡を床平面(XZ)へ投影して間取り図を作る(歩いた経路そのものが部屋の輪郭)。SPACE の一時停止はペンアップ: 壁から離れて移動する間だけ止めれば偽の壁が記録されず、壁を区間ごとに分けてカバーできる。出力は `maps/<日時>/` に room.npz / room.json / room.png。

## `find-button` — ボタン座標の三角測量

ボタンを画面中央に捉えて SPACE を押すと視線レイを記録し、2本以上の最小二乗交点=ボタン座標を表示する。別方向から狙うほど精度が出る(平行に近いと警告)。

## `patrol-buttons` — 壁を避けてボタン巡回(OSC + PID)

部屋マップ上で指定座標のボタンへ A\* で壁を避けて移動し、正対まで行う。移動・旋回は OSC、位置は HUD デコードでフィードバック。

```bash
# 計画のみ(VRChat不要)。plan.png をマップと同じフォルダに保存
uv run patrol-buttons --map maps/<日時>/room.npz --target 3.0,1.2,5.0,180 --dry-run

# 実際にOSCで巡回
uv run patrol-buttons --map maps/<日時>/room.npz --target 3.0,1.2,5.0,180 --target -1.0,1.0,2.0,90
```

- `--target X,Y,Z,FACE_YAW`(複数可): ボタン座標と向き。FACE_YAW は壁の外向き法線[deg](+Z基準)で、その正面 `--standoff`[m] に立って照準する。Y は pitch 合わせに必須(`find-button` の出力をそのまま渡せる)
- 制御の構成: 移動中(nav)は経路の少し先(`--nav-lookahead`)を狙う carrot 方式で連続追従し、到着後(face)に yaw/pitch を収束させる。視点軸は指令 0.50 以下が効かないため、不感帯補償(`--turn-deadzone` / `--nav-turn-deadzone`)で底上げする。ゲインの既定値と安定範囲、調整手順は [gain-tuning.md](gain-tuning.md)
- `--look mouse` で視点を DirectInput 相対マウスに切り替えられる。不感帯が無いので deadzone は 0 にし、ゲインは別物として再調整する
- `--world-cal` に `calibrate-world` の出力 JSON を渡すと、そのワールドの移動速度に合わせてゲインをスケールする
- 制御ログ: `logs/patrol_<日時>.csv` に毎フレームの姿勢と誤差、PID内訳、指令を書き出す(チューニングと原因調査用。`log-video` でそのまま再生できる)

## `log-video` — 制御ログを動画で再生

フレームCSV(実機の `logs/patrol_*.csv`またはsimの記録)を、一人称3D(レイキャスト)+上面2D地図の並置mp4にする。経路への追従とぎこちなさの目視確認用。ffmpeg が PATH に必要。

```bash
uv run log-video --csv logs/patrol_<日時>.csv --map maps/<日時>/room.npz --out run.mp4
```

`probe-axes` / `calibrate-world` の手順は [system-identification.md](system-identification.md)、`sim-face` / `bode-margins` の使い分けは [verification.md](verification.md) と [gain-tuning.md](gain-tuning.md) を参照。
