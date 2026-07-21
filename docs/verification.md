# オフライン検証メモ

制御の変更やゲイン調整を実機なしで確かめるときの手引き。ゲインの数値と安定範囲、チューニング判断は [gain-tuning.md](gain-tuning.md) を正とし、ここでは検証の組み方(データの場所、sim の回し方、指標、落とし穴)だけをまとめる。

## データの場所

| データ         | 場所                                                        | 備考                                                                                                   |
| -------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 同定プラント   | `logs/probe_<日時>/plant.json`                              | `PlantModel.load()` で読む。軸ごとに静特性 `points[(cmd, rate)]`・むだ時間 `deadtime_s`・`dt_mean`。   |
| 同定の生データ | 同ディレクトリの `probe_*.csv` / `segments_*.csv` / `*.png` | 静特性プロットの目視確認用(3点メディアンは隣接2レベル連続の異常を検知できない)                         |
| 部屋マップ     | `maps/<日時>/room.npz`                                      | `RoomMapper.load()` → `NavGrid.from_mapper()` で歩行可能グリッド。`plan_path()` で実地図の経路が作れる |
| 実機制御ログ   | `logs/patrol_<日時>.csv`                                    | `ControlLog`(recording.py)形式。sim の記録と同じ列                                                     |

## 最小ハーネス

`SimulatedVRChat` が reader/look/move の3役を兼ねる。`SimClock` を渡すと仮想時間で回るので実時間よりずっと速い。実例は tests/test_sysid.py の `test_all_control_loops_run_against_sim`。

```python
from vrc_autopilot.control.controller import PatrolGains, nav_controllers
from vrc_autopilot.control.maneuvers import follow_path
from vrc_autopilot.control.recording import ListRecorder
from vrc_autopilot.sysid.identify import PlantModel
from vrc_autopilot.sysid.sim_plant import SimClock, SimulatedVRChat

plant = PlantModel.load("logs/probe_<日時>/plant.json")
gains = PatrolGains()                       # フィールド指定で部分上書き
sim = SimulatedVRChat(plant, x=0.0, z=0.0, yaw=0.0)
rec = ListRecorder()                        # rec.rows は ControlRow のリスト
res = follow_path(sim, sim, sim, waypoints, gains, nav_controllers(gains),
                  clock=SimClock(sim), recorder=rec)
```

注意点:

1. 1試行ごとに sim と controllers を作り直す。両方が状態(積分器・位置)を持つ。
2. 複数フェーズを連結するときは SimClock を1つだけ作って共有する。フェーズごとに作り直すと `SimClock.t` は0に戻るのに `sim.now` は進んだままで、次フェーズ冒頭の見かけ dt が巨大になり align の stuck 検出(1s)が誤発火する。
3. recorder の `t` はフェーズ相対(各マニューバが `now - t0` で記録する)。連結ログでは t が巻き戻る。log-video は読込時に単調化するが、自作スクリプトで `searchsorted` などを使うなら先に dt 列で単調化すること。
4. コーナー応答だけ見たいときは、初期ヨーを第1セグメントの方位に合わせて開始時の引き込み過渡を消す(`degrees(atan2(dx, dz))`)。

## ストレスのかけ方

メモリ上で書き換える。JSON は変更しない。

```python
plant = PlantModel.load(path)
for m in plant.axes.values():
    m.deadtime_s *= 2.0                     # むだ時間(×1.5〜×2 が標準)
m = plant.axes["yaw"]
m.points = [(c, r * 1.5) for c, r in m.points]  # ゲイン誤差(オンセットは保存され傾きだけ変わる)
```

評価順序と受け入れ基準(全収束・osc≤3・×1.5 で崖なし)は gain-tuning.md「探索の作法」。危険なのは実機がモデルより速い側で、遅い側は失敗しない。

## 指標

AxisMetrics(recording.py、マニューバの戻り値に付く): osc(誤差の符号反転回数)が振動の主指標で 2–3 以下が目安、2桁は発振。settle_time=None は未収束。同成績なら effort が小さい方を採る。

滑らかさはフレーム記録から次を計算する:

| 指標                    | 定義                                                              | 何を見るか             |
| ----------------------- | ----------------------------------------------------------------- | ---------------------- |
| yaw角加速度RMS [deg/s²] | yaw を2回差分して dt で正規化(dt スパイク行は中央値×3 閾値で除外) | 回頭の滑らかさ         |
| dz_cross                | \|turn\| が不感帯オンセット 0.50 を横断した回数                   | 階段回頭の頻度         |
| play_m [m]              | \|yaw_err\|>2° なのにヨーレート≈0 のまま進んだ距離                | 不感帯の遊びによる滑走 |
| xtrack [m]              | 現セグメント(wp[i-1]→wp[i])への点線分距離の最大                   | 経路への密着・内切り   |

基準値(現行既定ゲイン)。ここから大きく悪化したらリグレッションを疑う。プラントを再同定したら測り直すこと(むだ時間が変わると全部動く):

| 構成                           | acc_rms | dz_cross | play_m | xtrack |
| ------------------------------ | ------- | -------- | ------ | ------ |
| 円弧 r=3m・90°、0.5m間隔WP     | ≈1000   | 0        | ≈0.1   | ≈0.05  |
| 実地図 A\*経路(52m・コーナー6) | ≈810    | ≈7       | ≈2.6   | ≈0.20  |

指標のアーチファクト(実振動と混同しないこと):

- 密な階段状ポリライン(A\* 出力)や carrot の連続参照では osc が増えるが振幅は微小。acc_rms と peak_err で実態を見る。
- 複数WP経路の yaw_over/peak には目標の側変わりが混ざる(終端オーバーシュートではない)。
- box 経路の並進 overshoot は脚長のアーチファクト。

## 目視確認(log-video)

数値で判断がつかない挙動は動画で見る。`ListRecorder` の rows を `ControlLog.FIELDS` 順の CSV に書くか、最初から `ControlLog` を recorder に渡す。

```bash
uv run log-video --csv frames.csv --map maps/<日時>/room.npz --out out.mp4 --png-every 120
```

HUD は4軸バー(turn/pitch/fwd/strafe)+フェーズ(色分け)+誤差+アクティブ軸の PID 内訳+実ヨーレート/実速度。バーの赤い目盛りが不感帯オンセット(turn ±0.50、他 ±0.10)で、バーが目盛りに届いていない間は指令が効いていない。実機ログ(`logs/patrol_*.csv`)もそのまま読める。

## ControlLog CSV のフェーズ別の列

フェーズによって空欄になる列がある。読む側は NaN 耐性を持たせること。

| phase     | 位置・姿勢      | 目標     | 誤差                           | 指令・PID                         |
| --------- | --------------- | -------- | ------------------------------ | --------------------------------- |
| nav       | x,y,z,yaw,pitch | tx,tz,wp | dist,yaw_err                   | turn(+P/I/D), fwd, fwd_factor     |
| translate | 同上            | tx,tz,wp | fwd_err,right_err              | fwd, strafe                       |
| face      | 同上            | tx,ty,tz | yaw_err,pitch_err              | turn(+P/I/D), pitch_cmd(+P/I/D)   |
| turn      | 同上            | なし     | yaw_err,pitch_err              | turn(+P/I/D), pitch_cmd(+P/I/D)   |
| align     | 同上            | tx,ty,tz | dist,yaw_err,pitch_err,lat_err | strafe(+P/I/D), pitch_cmd(+P/I/D) |
