# LEAP HORA Teacher Runtime Integration Gate

## 範圍與基線

- Phase：4A（teacher runtime integration only）
- Repository provenance：`yummypenguin/Unilab_0806`
- 起始 branch：`main`
- 起始 commit：`a789f0213e232e3d485e136a6dae74a01e015bce`
- 起始工作樹：乾淨
- Task owner：`leap_inhand_ball_0730/mujoco` + `hora` profile
- 不包含：Phase 5、真機、正式長訓練、影片、policy export

## Runtime contract

Phase 4A 使用實際 Hydra owner compose、`resolve_rsl_rl_ppo_runtime()`、
`HoraPPO.construct_algorithm()`、`HoraSharedActorCritic` 與 RSL-RL
`OnPolicyRunner` 建立整合 gate。Wrapper reset contract 為：

| field | shape |
|---|---:|
| actor / policy | `[N, 105]` |
| critic | `[N, 105]` |
| `priv_info` | `[N, 9]` |
| `proprio_hist` | `[N, 30, 32]` |

`LeapInhandBall0730HoraRotationEnv` 明確宣告 `critic_info` 為 episode-static。
`HoraRslRlVecEnvWrapper.step()` 在呼叫 autoreset-capable `env.step()` 前保存該 episode
的 `critic_info`，並將它與 `final_observation` 一同組成 timeout bootstrap input。
因此 partial done batch 中 terminated、truncated 與仍在執行的 rows 不會誤用下一個
episode 的 privileged values。

純 CPU integration test 另外涵蓋：8 env × 4 step rollout、finite actions/values/returns、
一次真實 PPO update、runner checkpoint save/reload、deterministic action parity `< 1e-6`，
以及 legacy / malformed checkpoint fail-closed。

## Short training smoke

DR-off command：

```text
uv run train --algo ppo --task leap_inhand_ball_0730 --sim mujoco --profile hora algo.num_envs=32 algo.num_steps_per_env=8 algo.max_iterations=2 algo.save_interval=1 algo.algorithm.num_learning_epochs=1 algo.algorithm.num_mini_batches=1 env.hora_domain_rand.enabled=false training.no_play=true training.logger=no_print
```

結果：exit 0，512 env steps，產生 `model_0.pt` 與 `model_1.pt`；最後 checkpoint：
`logs/hora_ppo/LeapInhandBall0730HoraRotation/2026-08-06_01-50-54_mujoco/model_1.pt`。

DR-on command：

```text
uv run train --algo ppo --task leap_inhand_ball_0730 --sim mujoco --profile hora algo.num_envs=32 algo.num_steps_per_env=8 algo.max_iterations=2 algo.save_interval=1 algo.algorithm.num_learning_epochs=1 algo.algorithm.num_mini_batches=1 env.hora_domain_rand.enabled=true training.no_play=true training.logger=no_print
```

結果：exit 0，512 env steps，產生 `model_0.pt` 與 `model_1.pt`；最後 checkpoint：
`logs/hora_ppo/LeapInhandBall0730HoraRotation/2026-08-06_01-51-31_mujoco/model_1.pt`。

這兩個 smoke 只驗證 runtime integration 與 checkpoint plumbing；短程 reward 數值不是
task quality 或 sim-to-real performance 證據。

## Deterministic headless reload

```text
uv run eval --algo ppo --task leap_inhand_ball_0730 --sim mujoco --profile hora --render-mode none --load-run 2026-08-06_01-51-31_mujoco training.play_steps=20 training.play_env_num=4 training.no_play=false training.logger=no_print
```

結果：載入 DR-on `model_1.pt`，完成 20 deterministic、no-render steps；所有 runtime
tensors 保持 finite，policy state hash 在 rollout 前後相同。此分支不建立影片或 policy
export。

## Validation

```text
uv run pytest tests/leap_deploy -q --basetemp C:\tmp\unilab-phase4a-deploy
# 87 passed

uv run pytest tests/algos -q -k "hora or Hora" --basetemp C:\tmp\unilab-phase4a-hora
# 25 passed, 1 skipped, 244 deselected

uv run pytest tests/envs/test_leap_inhand_0730.py -q --basetemp C:\tmp\unilab-phase4a-0730
# 14 passed

git diff --check
# passed (Git emitted only LF-to-CRLF working-copy notices)
```

指定的 broad Ruff target 仍會報告 Phase 4A 起始 commit 已存在的
`src/unilab/envs/manipulation/leap_inhand/ball_grasp_allegro.py` import-order issue；Phase 4A
未修改該檔，也未跨越授權範圍處理此 baseline lint debt。
