# vrc-autopilot

[VRCPositionHUD](https://github.com/njm2360/vrc-position-hud)を利用してOSCで移動・視点を操作する自動化ツール

- 全体像とモジュール対応: [docs/architecture.md](docs/architecture.md)
- プラント特性の測定手順: [docs/system-identification.md](docs/system-identification.md)
- 制御ゲインの根拠と調整手順: [docs/gain-tuning.md](docs/gain-tuning.md)
- オフライン検証の組み方: [docs/verification.md](docs/verification.md)
- 各コマンドの使用方法: [docs/usage.md](docs/usage.md)

## 動作環境

- Python 3.14
- [uv](https://docs.astral.sh/uv/)

## サンプルコード

- [でかプ 軽量化スイッチ自動化](examples/dekapu/main.py)
    ※マップデータ同梱

## CLI

`uv run <コマンド>` で実行する。フラグ詳細は各 `--help` で確認可能

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

視点軸の速度と不感帯はクライアント側の設定なのでワールドを移っても変わらないが、移動速度はワールド依存になる。ワールドを移ったら `probe-axes` を全部やり直すのではなく、`calibrate-world` で倍率だけ補正して `patrol-buttons --world-cal` に渡せばよい。測定手順は [docs/system-identification.md](docs/system-identification.md)。

典型的な流れ(map-room でマップを作り、find-button で座標を出した後):

```bash
# 計画のみ(VRChat不要)。plan.png をマップと同じフォルダに保存
uv run patrol-buttons --map maps/<日時>/room.npz --target 3.0,1.2,5.0,180 --dry-run

# 実際にOSCで巡回
uv run patrol-buttons --map maps/<日時>/room.npz --target 3.0,1.2,5.0,180 --target -1.0,1.0,2.0,90
```
