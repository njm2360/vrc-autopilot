# アーキテクチャ概要

VRChat 自動化のデータフローと、それを実装する `pose_hud` パッケージ / CLI の対応。

## パイプライン

```
VRChat 画面(HUDビットグリッド)
    │  スクリーンキャプチャ(ネイティブ解像度・クライアント左上の小領域)
    ▼
[capture]  WindowsVRChatCapture ──► フレーム(numpy)
    │
    ▼
[decode]   decode_pose ──► Pose(位置・前方・上方・時刻)+ 検証
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

制御ループ(patrol)は **アクチュエータ**(look / move)と **制御器**(AxisController)を注入で
受け取る。視点だけ OSC→マウスに差し替える、といった単位の入れ替えができる。HUD 表示切替は
アクチュエータではなく OSC 固有操作(`VRChatOSC.hud_enable`)。

## モジュール(`pose_hud/`)

各モジュールはサブモジュールから直接 import して使う(`__init__.py` は再エクスポートしない)。

| モジュール          | 役割                                                                                   |
| ------------------- | -------------------------------------------------------------------------------------- |
| `spec.py`           | グリッド/プロトコルの確定定数(モジュール定数。シェーダーと一致)                        |
| `pose.py`           | `Pose`(6DoF ドメイン型。全層が参照する中心の型)                                        |
| `decode.py`         | numpy ベクトル化デコード + 検証(`decode_pose`)                                         |
| `encode.py`         | 合成エンコーダ(`render_pose`)。テスト用                                                |
| `capture.py`        | Windows/VRChat ウィンドウキャプチャ(DPI対応、`FrameSource`)                            |
| `reader.py`         | `PoseReader`(スレッドで読み続ける・統計・ジェネレータ)                                 |
| `mapping.py`        | `RoomMapper`。軌跡→寸法・占有グリッド・保存/読込(ペンアップ分割対応)                   |
| `mapping_render.py` | 間取り図の描画(matplotlib)                                                             |
| `triangulate.py`    | 視線レイの最小二乗交点でボタン座標を推定                                               |
| `navigation.py`     | 歩行可能グリッド生成(壁回避)+ A* 経路計画 + 経由点の直線化                             |
| `guidance.py`       | フレーム単位の照準幾何(`wrap180` / `heading_error` / `pitch_error` / `forward_factor`) |
| `pid.py`            | 汎用 PID(離散・積分の溜まりすぎ防止・不感帯補償)                                       |
| `controller.py`     | `AxisController`(PID+tolゲート)/ `PatrolGains` / 制御器ビルダー                        |
| `actuator.py`       | `LookActuator`/`MoveActuator` IF + `MouseLookActuator`(DirectInput)                    |
| `osc.py`            | VRChat への OSC 送信(look/move/stop で両 actuator IF を満たす)                         |
| `telemetry.py`      | `Recorder` IF + `AxisMetrics`/`AxisAccumulator`(応答指標の累積)                        |
| `maneuvers.py`      | 制御ループの建物ブロック(`follow_path` / `aim_at` / `turn_to`。ヘッドレス可)           |
| `pilot.py`          | `Pilot` ファサード(経路計画+ループ連結。実機 I/O は `connect()` に集約)                |

## CLI(`pose_hud/cli/`, console scripts)

| コマンド         | スクリプト              | 用途                                                            |
| ---------------- | ----------------------- | --------------------------------------------------------------- |
| `decode-demo`    | `cli/decode_demo.py`    | HUD を読み取り 6DoF を表示(動作確認)                            |
| `map-room`       | `cli/map_room.py`       | 壁沿いに歩いて部屋マップを記録(SPACE一時停止・日時フォルダ出力) |
| `find-button`    | `cli/find_button.py`    | 複数地点からボタンを三角測量(SPACE/r/q)                         |
| `patrol-buttons` | `cli/patrol_buttons.py` | マップ上でボタンを壁を避けて巡回(OSC + PID)                     |
