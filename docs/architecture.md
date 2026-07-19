# アーキテクチャ概要

VRChat 自動化のデータフローと、それを実装する `app` パッケージ / CLI の対応。

## パイプライン

```txt
VRChat 画面(HUDビットグリッド)
    │  スクリーンキャプチャ(ネイティブ解像度・クライアント左上の小領域)
    ▼
[capture]  WindowsVRChatCapture ──► フレーム(numpy)
    │
    ▼
[decode]   decode_frame ──► DecodeResult(Pose + 検証結果)
    │
    ▼
[reader]   PoseReader(スレッドで読み続ける・統計・コールバック/ジェネレータ)
    │
    ├─► [mapping]     RoomMapper ──► 部屋の地図(占有グリッド/間取り図)
    ├─► [triangulate] Sighting/triangulate ──► ボタンのワールド座標(最小二乗交点)
    └─► [navigation]  NavGrid + plan_path ──► 壁を避けた経路
                          │
                          ▼
                     [pilot / maneuvers] goto / aim / patrol(フェーズ連結と制御ループ)
                          │  [guidance] で誤差を計算し、[controller] AxisController(PID) が指令[-1,1] に変換
                          ▼
                     [actuator]  LookActuator / MoveActuator
                          ├─ OSC:         VRChatOSC ──► /input/* を注入
                          └─ DirectInput: MouseLookActuator ──► 相対マウス(視点のみ)
```

制御ループ(patrol)はアクチュエータ(look / move)と制御器(AxisController)を注入で
受け取る。視点だけ OSC からマウスに差し替える、といった部品単位の入れ替えができる。HUD 表示切替は
アクチュエータではなく OSC 固有操作(`VRChatOSC.hud_enable`)。

## モジュール(`app/`)

### `core/` — 全層が参照する中核型

| モジュール     | 役割                    |
| -------------- | ----------------------- |
| `core/pose.py` | `Pose`(6DoF ドメイン型) |

### `perception/` — VRChat 画面 → Pose

| モジュール              | 役割                                                            |
| ----------------------- | --------------------------------------------------------------- |
| `perception/spec.py`    | グリッド/プロトコルの確定定数(モジュール定数。シェーダーと一致) |
| `perception/capture.py` | Windows/VRChat ウィンドウキャプチャ(DPI対応、`FrameSource`)     |
| `perception/decode.py`  | numpy ベクトル化デコード + 検証(`decode_frame` / `decode_pose`) |
| `perception/encode.py`  | 合成エンコーダ(`render_pose`)。テスト用                         |
| `perception/reader.py`  | `PoseReader`(スレッドで読み続ける・統計・ジェネレータ)          |

### `mapping/` — 部屋地図

| モジュール          | 役割                                                                   |
| ------------------- | ---------------------------------------------------------------------- |
| `mapping/mapper.py` | `RoomMapper`。軌跡→寸法・占有グリッド・外周/内壁ポリゴン・保存/読込    |
| `mapping/draw.py`   | 地図の描画・PNG 保存(バックエンド非依存の `draw_map` / `save_map_png`) |
| `mapping/live.py`   | 録画中のライブ地図表示(matplotlib インタラクティブ)                    |

### `spatial/` — 空間推定・経路計画

| モジュール               | 役割                                               |
| ------------------------ | -------------------------------------------------- |
| `spatial/triangulate.py` | 視線レイの最小二乗交点でボタン座標を推定           |
| `spatial/navigation.py`  | 歩行可能グリッド生成(壁回避)+ A\* + 経由点の直線化 |

### `control/` — 閉ループ制御 + VRChat 出力

| モジュール                 | 役割                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------- |
| `control/guidance.py`      | フレーム単位の照準幾何(`wrap180` / `heading_error` / `pitch_error` / `forward_factor`) |
| `control/pid.py`           | 汎用 PID(離散・積分の溜まりすぎ防止・不感帯補償)                                       |
| `control/controller.py`    | `AxisController`(PID+tolゲート)/ `PatrolGains` / 制御器ビルダー                        |
| `control/loop_analysis.py` | ループ整形解析(安定余裕・ボード線図。`analyze_patrol` / `save_bode_png`)               |
| `control/actuator.py`      | `LookActuator`/`MoveActuator` IF + `MouseLookActuator`(DirectInput)                    |
| `control/osc.py`           | VRChat への OSC 送信(look/move/stop で両 actuator IF を満たす)                         |
| `control/recording.py`     | `Recorder` IF(`ControlLog`=CSV / `ListRecorder`)+ `AxisMetrics`(応答指標)              |
| `control/maneuvers.py`     | 制御ループの部品(`follow_path`(carrot追従)/ `aim_at` / `strafe_align` / `turn_to`)     |
| `control/pilot.py`         | `Pilot` ファサード(経路計画+ループ連結。実機 I/O は `connect()` に集約)                |

### `sysid/` — システム同定・シミュレーション

| モジュール           | 役割                                                                                   |
| -------------------- | -------------------------------------------------------------------------------------- |
| `sysid/identify.py`  | システム同定(プローブ注入・静特性/むだ時間/dt 抽出・`PlantModel` JSON)                 |
| `sysid/sim_plant.py` | `SimulatedVRChat`(同定モデルを積分。PoseSource+両 Actuator を満たし制御ループに注入可) |

## CLI(`app/cli/`, console scripts)

| コマンド         | スクリプト              | 用途                                                              |
| ---------------- | ----------------------- | ----------------------------------------------------------------- |
| `decode-demo`    | `cli/decode_demo.py`    | HUD を読み取り 6DoF を表示(動作確認)                              |
| `map-room`       | `cli/map_room.py`       | 壁沿いに歩いて部屋マップを記録(SPACE一時停止・日時フォルダ出力)   |
| `find-button`    | `cli/find_button.py`    | 複数地点からボタンを三角測量(SPACE/r/q)                           |
| `patrol-buttons` | `cli/patrol_buttons.py` | マップ上でボタンを壁を避けて巡回(OSC + PID)                       |
| `probe-axes`     | `cli/probe_axes.py`     | 入力軸の応答特性を測定し `plant.json` に同定(--from-log で再同定) |
| `sim-face`       | `cli/sim_face.py`       | 同定プラント上で正対ループを回しゲインを検証(実機不要)            |
| `bode-margins`   | `cli/bode_margins.py`   | 同定プラント上で全ループの安定余裕とボード線図を出す(実機不要)    |
| `log-video`      | `cli/log_video.py`      | 制御ログCSVを一人称3D+2D地図の動画に再生(目視確認用)              |
