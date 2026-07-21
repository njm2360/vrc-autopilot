# CLIの使用方法

各コマンドの使い方とフラグの補足。一覧は [README](../README.md)、フラグの網羅は各 `--help`。

## `decode-demo` — 6DoF 表示

`pos`(ワールド座標[m])/ `yaw`(+Z基準[deg])/ `pitch`(上向き正[deg])/ `t`(同期時刻)を読み続けて表示する(`--stats` を付けると fps と成功率も出る)。グリッド定数は [src/vrc_autopilot/perception/spec.py](../src/vrc_autopilot/perception/spec.py) の確定値。変更する場合はシェーダー側と一致させること。

## `map-room` — 部屋マップ記録

壁沿いに歩いた視点軌跡を床平面(XZ)へ投影して間取り図を作る(歩いた経路そのものが部屋の輪郭)。SPACE の一時停止はペンアップ: 壁から離れて移動する間だけ止めれば偽の壁が記録されず、壁を区間ごとに分けてカバーできる。出力は `maps/<日時>/` に room.npz / room.json / room.png。

## `find-button` — ボタン座標の三角測量

ボタンを画面中央に捉えて SPACE を押すと視線レイを記録し、2本以上の最小二乗交点=ボタン座標を表示する。別方向から狙うほど精度が出る(平行に近いと警告)。

## `log-video` — 制御ログを動画で再生

フレームCSV(実機の `logs/patrol_*.csv`またはsimの記録)を、一人称3D(レイキャスト)+上面2D地図の並置mp4にする。経路への追従とぎこちなさの目視確認用。ffmpeg が PATH に必要。

```bash
uv run log-video --csv logs/patrol_<日時>.csv --map maps/<日時>/room.npz --out run.mp4
```

`probe-axes` / `calibrate-world` の手順は [system-identification.md](system-identification.md)、`bode-margins` とオフライン検証の使い分けは [verification.md](verification.md) と [gain-tuning.md](gain-tuning.md) を参照。
