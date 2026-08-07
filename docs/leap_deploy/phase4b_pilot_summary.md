# Phase 4B Teacher Training Readiness and Pilot

## 基線與範圍

- Repository：`yummypenguin/Unilab_0806`
- Branch：`main`
- 起始 commit：`ec602a105fe379195a366c043e8ee4651eb6345d`
- 起始工作樹：乾淨
- Pilot：3 training seeds，每個 `2048 × 8 × 50 = 819,200` environment steps
- Evaluation：每個 run 的 5 checkpoints × 2 suites × 3 evaluation seeds
- Evaluation budget：每 row 256 envs × 800 deterministic steps
- 總 evaluation rows：90；non-finite rows：0；actor hash mismatches：0

本階段沒有執行正式長訓練、student distillation、export、影片或真機操作。

## Repository hygiene

Phase 4A 的兩個 tracked smoke run 已從工作樹移除，`.gitignore` 已加入 `logs/`。
以下兩個 checks 都由 `.gitignore:44` 命中：

```text
logs/hora_ppo/example/model_1.pt
logs/evaluation/example/eval_summary.json
```

`.gitattributes` 沒有修改，既有 `*.pt` 與 `events.out.tfevents.*` LFS rules 保留。

## Axis 與 diagnostics contract

`rotation_axis_command` 與 reward `rotation_axis` 現在各自 normalize 後必須在
`atol=1e-6` 內相同。相同軸及等比例軸通過；不同、反向或 zero axis fail closed。
預設仍是 `[0, 0, 1]`。

Rotation diagnostics 使用 reward path 收到的 raw ball angular velocity；
`rotation/axis_speed_mean` 沒有 clip。Control diagnostics 使用 delay queue 實際取出的
raw delayed action，以及 superclass clip/integrate 後實際送給 backend 的 target。
Drop termination 與 horizon truncation 分開累積計數。

## Pilot runs

| training seed | run | env steps | checkpoints |
|---:|---|---:|---|
| 1 | `2026-08-06_03-07-49_mujoco` | 819,200 | `0, 10, 20, 30, 40, 49` |
| 2 | `2026-08-06_03-13-58_mujoco` | 819,200 | `0, 10, 20, 30, 40, 49` |
| 3 | `2026-08-06_03-18-25_mujoco` | 819,200 | `0, 10, 20, 30, 40, 49` |

RSL-RL iteration index從0開始，因此50次 update後的 final checkpoint是
`model_49.pt`。為保留真實 checkpoint/resume metadata，evaluation使用
`10, 20, 30, 40, 49`，沒有將 final checkpoint偽裝成 `model_50.pt`。

## 跨 training seed 聚合結果

下表每列聚合3個 training seeds × 3個 evaluation seeds。`duration`單位為秒，
`axis speed`單位為 rad/s；所有 rate均為比例。

| checkpoint | suite | episodes | termination | axis speed | positive | reverse | high clip | duration | return | action sat. | target sat. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | nominal | 195,171 | 1.0000 | 0.6362 | 0.9520 | 0.0480 | 0.5503 | 0.4792 | -0.5655 | 0.0000 | 0.0008 |
| 10 | train_dr | 184,774 | 1.0000 | 0.5835 | 0.9050 | 0.0950 | 0.5048 | 0.5055 | -0.7452 | 0.0000 | 0.0008 |
| 20 | nominal | 320,624 | 1.0000 | 0.9765 | 0.9411 | 0.0589 | 0.6522 | 0.2867 | -0.5348 | 0.0003 | 0.0004 |
| 20 | train_dr | 291,379 | 1.0000 | 0.8857 | 0.8841 | 0.1159 | 0.5885 | 0.3153 | -0.7043 | 0.0003 | 0.0007 |
| 30 | nominal | 348,502 | 1.0000 | 1.1461 | 0.9476 | 0.0524 | 0.6773 | 0.2637 | -0.6122 | 0.0095 | 0.0009 |
| 30 | train_dr | 314,416 | 1.0000 | 1.0268 | 0.8888 | 0.1112 | 0.6075 | 0.2922 | -0.7733 | 0.0080 | 0.0012 |
| 40 | nominal | 359,186 | 1.0000 | 1.2762 | 0.9522 | 0.0478 | 0.6970 | 0.2558 | -0.7341 | 0.0368 | 0.0015 |
| 40 | train_dr | 323,447 | 1.0000 | 1.1335 | 0.8930 | 0.1070 | 0.6238 | 0.2840 | -0.8280 | 0.0320 | 0.0018 |
| 49 | nominal | 363,890 | 1.0000 | 1.3585 | 0.9561 | 0.0439 | 0.7112 | 0.2526 | -0.6674 | 0.0790 | 0.0025 |
| 49 | train_dr | 326,952 | 1.0000 | 1.2029 | 0.8959 | 0.1041 | 0.6361 | 0.2810 | -0.7477 | 0.0693 | 0.0028 |

## 各 seed learning trend

| seed | suite | axis speed 10 → 49 | duration 10 → 49 | termination | final action sat. | final target sat. |
|---:|---|---:|---:|---:|---:|---:|
| 1 | nominal | 0.6082 → 1.3381 | 0.3989 → 0.2499 | 1.0000 | 0.0650 | 0.0013 |
| 1 | train_dr | 0.5494 → 1.1901 | 0.4228 → 0.2779 | 1.0000 | 0.0568 | 0.0016 |
| 2 | nominal | 0.7210 → 1.3774 | 0.4669 → 0.2549 | 1.0000 | 0.0763 | 0.0017 |
| 2 | train_dr | 0.6698 → 1.2184 | 0.4951 → 0.2841 | 1.0000 | 0.0672 | 0.0020 |
| 3 | nominal | 0.5795 → 1.3599 | 0.5718 → 0.2528 | 1.0000 | 0.0957 | 0.0046 |
| 3 | train_dr | 0.5312 → 1.2002 | 0.5986 → 0.2810 | 1.0000 | 0.0838 | 0.0047 |

三個 seeds 都學到更高的正向 axis speed，但所有 checkpoint與suite的drop termination
rate都是100%。同時 episode duration隨訓練縮短，而不是穩定或增加。Return也沒有跨
seed一致改善。Final target saturation低於0.5%，action saturation約5.7%–9.6%，因此
主要失敗不是持續撞 joint limits；證據較符合「產生短暫旋轉後迅速掉落」。

Train-DR的 final axis speed相對 nominal低約11%–12%，reverse rate約從4%–5%增加到
10%–11%。DR沒有改變100% termination結論。

## Pilot decision

結論：**停止，不建議進入 Phase 4C正式 teacher training。**

這批 pilots不符合「明顯學習」條件：雖然 axis speed提高，但 termination沒有改善，
episode duration反而一致惡化。也沒有任何 checkpoint可作為可靠teacher candidate。
若只看 axis speed會偏好 late checkpoint，但這會忽略100%掉落與約0.25–0.28秒的
episode；因此不採用 evaluator ranking作為teacher選擇。

後續若另行授權新的實驗，優先診斷項目是 reward對drop/存活的平衡、DR是否過早、
action-delay比例、history timing與 observation sufficiency。這些是下一階段假設，
不是本次結果已證明的根因。

## Evaluation artifacts

每個目錄都包含 `evaluation_manifest.json`、`checkpoint_metrics.csv` 與
`checkpoint_metrics.json`：

- `logs/evaluation/leap_hora_phase4b/seed1/`
- `logs/evaluation/leap_hora_phase4b/seed2/`
- `logs/evaluation/leap_hora_phase4b/seed3/`

所有90 rows均為 deterministic、finite、no-render、no-export，且 actor hash before/after
相同。
