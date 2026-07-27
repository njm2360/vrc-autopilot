# vrc-autopilot

[![PyPI](https://img.shields.io/pypi/v/vrc-autopilot)](https://pypi.org/project/vrc-autopilot/)
[![Python](https://img.shields.io/pypi/pyversions/vrc-autopilot)](https://pypi.org/project/vrc-autopilot/)
[![License](https://img.shields.io/pypi/l/vrc-autopilot)](LICENSE)
[![CI](https://github.com/njm2360/vrc-autopilot/actions/workflows/ci.yml/badge.svg)](https://github.com/njm2360/vrc-autopilot/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/njm2360/vrc-autopilot/graph/badge.svg)](https://codecov.io/gh/njm2360/vrc-autopilot)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Platform](https://img.shields.io/badge/platform-Windows-0078d4)

[VRCPositionHUD](https://github.com/njm2360/vrc-position-hud)を利用してOSCで移動・視点を操作する自動化ツール

> [!WARNING]
> 本ツールの使用によって生じたいかなる結果についても、作者は一切の責任を負いません。自己責任で使用してください。
> 使用にあたっては各ワールド・コミュニティのルールに従ってください。自動化を歓迎しない場所では使わないこと。

## 動作環境

- Windows 10/11
- Python 3.12+
- VRChat (デスクトップモード、OSC有効)
- [VRCPositionHUD](https://github.com/njm2360/vrc-position-hud) を組み込んだアバター

## インストール

```sh
uv add vrc-autopilot
```

## サンプルコード

移動・照準・押下は `Pilot` API を使用して記述します。

- [でかプ 軽量化スイッチ自動化](examples/dekapu/main.py)
  ※マップデータ同梱

## CLI

`uv run <コマンド>` で実行します。フラグ詳細は各 `--help` で確認可能です。

| コマンド          | 用途                                                               |
| ----------------- | ------------------------------------------------------------------ |
| `decode-demo`     | HUDを読み取り、座標と姿勢を表示するサンプル                        |
| `map-room`        | 壁沿いに歩いて部屋マップを記録する                                 |
| `find-button`     | 複数地点からボタンを三角測量して座標を推定する                     |
| `probe-axes`      | 入力軸の応答特性を測ってプラントモデルを同定する                   |
| `calibrate-world` | ワールドごとに変わる移動速度を測り、ゲインの倍率を補正             |
| `bode-margins`    | 同定プラント上で全制御ループの安定余裕(ωc/PM/GM)とボード線図を出す |
| `log-video`       | 制御ログCSVを一人称3D+2D地図の動画(mp4)に再生                      |

## 開発者向け

- 全体像とモジュール対応: [docs/architecture.md](docs/architecture.md)
- プラント特性の測定手順: [docs/system-identification.md](docs/system-identification.md)
- 制御ゲインの根拠と調整手順: [docs/gain-tuning.md](docs/gain-tuning.md)
- オフライン検証の組み方: [docs/verification.md](docs/verification.md)
- 各コマンドの使用方法: [docs/usage.md](docs/usage.md)
