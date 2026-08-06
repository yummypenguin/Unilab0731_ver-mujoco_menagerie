# LEAP-Hand Menagerie HORA 真機部署 Phase 0 稽核

## 稽核範圍與基線

- Repository：`yummypenguin/Unilab0731_ver-mujoco_menagerie`
- 起始 branch：`main`
- 起始 commit：`83f7d5fe21bf93bbc5d3ba149102c83364257ac2`
  (`83f7d5fe Add APPO training logs`)
- 起始工作樹：乾淨（`git status --short` 無輸出）
- Task：`LeapInhandBall0730Rotation`
- Environment：`src/unilab/envs/manipulation/leap_inhand/ball_rotation_0730.py`
- PPO owner：`conf/ppo/task/leap_inhand_ball_0730/mujoco.yaml`
- Scene：`src/unilab/assets/robots/leap_hand/scene_ball.xml`

本文件只記錄 Phase 0 的 repository 稽核結果。沒有修改 environment、訓練
config、reward、asset、checkpoint 或 deployment runtime，也沒有執行訓練或連接真機。

## 1. Menagerie actuator、joint 與 qpos contract

`scene_ball.xml` 依序 include `leap_hand.xml` 與 `ball.xml`。現有測試
`test_leap_scene_compiles_with_aligned_joint_and_actuator_order` 已把編譯後模型 contract
固定為：`nq=23`、`nv=22`、`nu=16`，前 16 個 joint/actuator 一一對應，
`actuator_trnid[:16, 0] == arange(16)`，且 `jnt_qposadr[:16] == arange(16)`。

| actuator index | actuator name | joint name | qpos address | ctrlrange [rad] |
|---:|---|---|---:|---:|
| 0 | `if_mcp_act` | `if_mcp` | 0 | `[-0.314, 2.230]` |
| 1 | `if_rot_act` | `if_rot` | 1 | `[-1.047, 1.047]` |
| 2 | `if_pip_act` | `if_pip` | 2 | `[-0.506, 1.885]` |
| 3 | `if_dip_act` | `if_dip` | 3 | `[-0.366, 2.042]` |
| 4 | `mf_mcp_act` | `mf_mcp` | 4 | `[-0.314, 2.230]` |
| 5 | `mf_rot_act` | `mf_rot` | 5 | `[-1.047, 1.047]` |
| 6 | `mf_pip_act` | `mf_pip` | 6 | `[-0.506, 1.885]` |
| 7 | `mf_dip_act` | `mf_dip` | 7 | `[-0.366, 2.042]` |
| 8 | `rf_mcp_act` | `rf_mcp` | 8 | `[-0.314, 2.230]` |
| 9 | `rf_rot_act` | `rf_rot` | 9 | `[-1.047, 1.047]` |
| 10 | `rf_pip_act` | `rf_pip` | 10 | `[-0.506, 1.885]` |
| 11 | `rf_dip_act` | `rf_dip` | 11 | `[-0.366, 2.042]` |
| 12 | `th_cmc_act` | `th_cmc` | 12 | `[-0.349, 2.094]` |
| 13 | `th_axl_act` | `th_axl` | 13 | `[-0.349, 2.094]` |
| 14 | `th_mcp_act` | `th_mcp` | 14 | `[-0.470, 2.443]` |
| 15 | `th_ipl_act` | `th_ipl` | 15 | `[-1.340, 1.880]` |

這些 ctrlrange 由 `leap_hand.xml` 的 actuator class defaults 擁有，並由測試確認等於
編譯後 `model.actuator_ctrlrange`。同一 XML 固定 joint `damping=0.03`、
`frictionloss=0.001`、`armature=0.01` 與 position actuator `kp=3.0`、`kv=0.01`。
模型沒有 actuator force limit。`LeapHandBaseEnv` 會拒絕 MuJoCo task-level joint
dynamics 或不相符的 kp/kd override；這些值不可轉寫成 Dynamixel register gain。

## 2. 現有 105-D observation 精確 layout

`AllegroRotationPPO` 定義：

- `_NUM_OBS_PER_STEP = 35`
- `_NUM_LAG_STEPS = 3`
- history shape：`[N, 3, 35]`
- flatten order：oldest → newest，得到 `[N, 105]`

每一幀的實際欄位是：

| frame offset | width | 欄位 | 實際語意 |
|---:|---:|---|---|
| `0:16` | 16 | hand qpos | `2 * (q - dof_mid) / (dof_range + 1e-8)` |
| `16:32` | 16 | `targets` | `info["prev_ctrl"]` 的原始 actuator target，單位 rad；**未正規化** |
| `32:35` | 3 | ball position | simulator world position，單位 m |

因此 flattened observation 是：

```text
[frame(t-2, 35), frame(t-1, 35), frame(t, 35)]
```

Reset 時 `build_obs_lag_history()` 會把由 cache state 建立的同一個 initial frame
複製到三個 history slot；不是零填充。Step 時 history 先左移，再把 current frame
寫入最後一格。

部署含義：現有 actor 直接讀取 ball world position，不能直接部署。即使新 HORA
observation 仍是 105 維，只要改成 `[q_norm, normalized_previous_target, axis]`，其欄位
語意已不同，現有 checkpoint 不能重用。

## 3. Action integration 與 `prev_ctrl` timing

實際 `NpEnv.step()` 順序是：

```text
apply_action(actions, state)
backend.step(ctrl, sim_substeps)
update_state(state)
steps += 1
compute truncation / autoreset
```

`AllegroBaseEnv.apply_action()` 執行：

```python
clipped_actions = np.clip(actions, -1.0, 1.0)
new_ctrl = prev_ctrl + action_scale * clipped_actions
new_ctrl = np.clip(new_ctrl, ctrl_lower, ctrl_upper)
state.info["prev_ctrl"] = new_ctrl
return new_ctrl
```

本 task 的 `action_scale = 1/24 = 0.041666666666666664 rad`，control period 是
`ctrl_dt=0.05 s`。Reset provider 將 cache row 的 16 維 hand qpos 同時設成初始
`prev_ctrl`。

重要時序差異：`apply_action()` 在 physics step 前就把 `info["prev_ctrl"]` 更新為
本次送出的 `target_t`；`update_state()` 之後的 `_compute_obs()` 再讀這個值。因此現有
step 回傳的 observation 內 target 欄是本次剛送出的 `target_t`，不是送出 action 前的
`target_(t-1)`。Phase 2 若要求 policy timestep frame 使用上一輪真正送出的 target，
必須在 owner layer 明確保存/組裝 pre-action target，不能只覆寫現有 `_compute_obs()`
欄位名稱後假設時序已符合。

## 4. Random cache reset

Owner config 固定：

```yaml
env:
  gen_grasp: false
  grasp_cache_path: robots/leap_hand/caches/ball_grasp_allegro_new_physics_0731_50k.npy
```

相對路徑由 `resolve_grasp_cache_path()` 以 `ASSETS_ROOT_PATH` 解析，實際 repository
asset 是：

```text
src/unilab/assets/robots/leap_hand/caches/
ball_grasp_allegro_new_physics_0731_50k.npy
```

Cache 只在首次需要時以 `np.load(...).astype(np.float64)` 載入並保存於
`env._grasp_cache`。每次 reset 對 `num_reset` 個環境執行：

```python
idx = np.random.randint(0, len(grasp_cache), size=num_reset)
sampled = grasp_cache[idx]
hand_qpos = sampled[:, :16]
ball_pos = sampled[:, 16:19]
ball_quat = sampled[:, 19:23]
```

這是每個 reset environment 獨立、均勻、有放回抽樣。每列 23 維，依序為
`hand qpos(16) + ball position(3) + ball quaternion wxyz(4)`。Reset plan 把它們組成
23-D qpos，並把 qvel 初始化為零（現有 owner 的 `ball_vel_noise=0`）。

`LeapBall0730ResetProvider` 另外把該 row 的 `ball_pos[:, 2]` 複製到
`info["initial_ball_z"]`，並把固定 cache-generation nominal hand pose 放入
`info["init_pose"]` 供 pose-difference reward 使用。

## 5. Reward、termination 與 truncation

PPO owner 與現有測試固定 reward：

```yaml
rotate:      1.25
obj_linvel: -0.3
pose_diff:  -0.3
torque:     -0.1
work:       -2.0
drop:        0.0
```

Rotation angular velocity clipping 是 `[-0.5, 0.5]`。目前沒有 middle-contact gate，
也沒有 `middle_contact_rotation_fraction` config。Reward 各項加權後再乘 `ctrl_dt`。

Task failure termination：

```python
threshold = info["initial_ball_z"] - 0.03
terminated = ball_pos[:, 2] <= threshold
```

邊界恰好下降 30 mm 也會 terminate。時間上限由 `max_episode_seconds=20.0` 與
`ctrl_dt=0.05` 推得 `max_episode_steps=400`。`NpEnv` 在每次 backend step 後令
`steps += 1`，當 `steps >= 400` 設 `truncated=True`；這不是 task failure
`terminated`。現有 autoreset 會保留 final observation 後重置完成的環境。

## 6. HORA teacher observation interface

HORA runtime 的 source of truth 是
`src/unilab/algos/torch/hora/observations.py`：

1. Actor observation 從 `state.obs["obs"]` 讀取。
2. Critic observation優先從 `state.obs["critic"]` 讀取；缺省時等於 actor obs。
3. Privileged info 優先從 `state.info["critic_info"]` 讀取，且必須是 NumPy array、
   batch dimension 與 actor observation 相同。
4. 若沒有有效 `critic_info`，而 `critic` 比 actor 多欄，runtime 會把 critic tail
   當作 privileged info；新 task 應提供明確 `critic_info`，避免依賴此 fallback。
5. Proprio history 從 `state.info["proprio_hist"]` 讀取；名稱不是
   `"proprio_history"`。

Wrapper 建立的 TensorDict keys 是：

```text
actor
policy
critic       # critic obs 存在時
priv_info    # 有 explicit/fallback privileged info 時
proprio_hist # info 提供 NumPy array 時
```

因此 Phase 2 environment contract 應為：

```python
state.obs = {"obs": actor_obs}
state.info["critic_info"] = privileged_info
state.info["proprio_hist"] = proprio_history
```

## 7. Stage-2 distillation checkpoint 與 normalizer

HORA stage-2 trainer 只訓練 actor 中名稱包含 `adapt_tconv` 的參數；teacher actor
weights 先從 teacher checkpoint 載入。PPO teacher key 是 `actor_state_dict`，APPO 是
`actor`，SAC 是 `actor` 但走 SAC-specific loader。

`HoraDistillationTrainer.save()` 目前保存以下 dictionary：

```text
model_state_dict       student actor 的完整 state_dict
history_normalizer     proprio-history EmpiricalNormalization state_dict
agent_steps
teacher_checkpoint
teacher_algo_family
teacher_metadata
distill_runtime_cfg
```

`history_normalizer` 是針對 `proprio_hist_shape = obs["proprio_hist"].shape[1:]`
建立的 `EmpiricalNormalization` state。載入 stage-2 checkpoint 時：

- `model_state_dict` 必須存在並以 `strict=True` 載入；
- `history_normalizer` 若存在便載入，否則保留新建 normalizer；
- `distill_runtime_cfg` 目前只持久化重建 student model 所需的 `algo.model`；
- env、reward 與 domain-randomization owner settings **不**寫入 stage-2 checkpoint，
  playback 會使用當下 compose 的 owner config。

Stage-2 檔名為 periodic `hora_stage2_<agent_steps>.pt` 與最終
`hora_stage2_last.pt`。Student inference 先用 history normalizer 處理
`proprio_hist`，actor 自身的 observation normalizer 則包含在
`model_state_dict` 的 model state 中。

## 8. 現有 deployment utilities 可重用範圍

`scripts/deploy/` 目前只有 G1-oriented utilities，沒有 LEAP hardware driver、LEAP
observation builder、joint mapping、calibration、safety、shadow mode 或 TorchScript
student exporter。

可重用的是設計方法，不是 G1 schema：

- `sim_prototype.py`
  - stateful history assembler；首次 reset 以 current segments 填滿；
  - oldest-first flatten；
  - layout dimension、finite value 與 model input width fail-fast 檢查；
  - deployment-path MuJoCo loop 與 latency/step 結構可作參考。
- `export_deploy_config.py`
  - 從編譯後 MuJoCo model 匯出 actuator/joint metadata；
  - manifest/layout 作單一 source of truth；
  - training/deployment alignment test 的思路可重用。

不可直接重用的部分包括 G1 的 29-DoF joint layout、free-base offsets、motion body
tracking、gyro/anchor observations、absolute `action * scale + default_angles` action
contract、ONNX runtime 與 C++ WBT schema。`export_motion_bin.py`、
`prepend_warmup.py`、`append_cooldown.py` 也是 motion-tracking/G1 工作流，不屬於 LEAP
HORA V1 runtime。

真機 deployment 尚缺：LEAP-specific deployment contract、独立 observation/history
builder、student export/runtime、MuJoCo parity harness、hardware driver、sim↔motor order
mapping、angle conversion、calibration/homing、safety state machine、shadow mode、live
rollout 與 rotation skill wrapper。

## 9. Menagerie joint names 與 real motor ID 候選 mapping

Repository 目前只有三組帶歷史來源的常數：

- `SOURCE_SIM_TO_REAL_INDICES`：舊 source Isaac Gym ordering 的 mapping；
- `SOURCE_REAL_TO_SIM_INDICES`：上述 source ordering 的 inverse；
- `UNILAB_SIM_JOINT_ORDER`：pre-Menagerie UniLab model 的 numeric qpos traversal order，
  註解明確限制為 cache/deployment provenance；
- 現行 MuJoCo source of truth 是 `MENAGERIE_SIM_JOINT_NAMES`。

Repository 內沒有 LEAP official API、沒有已校準的 real motor mapping，也沒有逐關節
物理驗證紀錄。因此下表只能作為 Phase 9 的**候選**，不能宣稱已驗證：

| Menagerie index | joint | candidate real motor ID | 狀態 |
|---:|---|---:|---|
| 0 | `if_mcp` | 1 | 未驗證 |
| 1 | `if_rot` | 0 | 未驗證 |
| 2 | `if_pip` | 2 | 未驗證 |
| 3 | `if_dip` | 3 | 未驗證 |
| 4 | `mf_mcp` | 5 | 未驗證 |
| 5 | `mf_rot` | 4 | 未驗證 |
| 6 | `mf_pip` | 6 | 未驗證 |
| 7 | `mf_dip` | 7 | 未驗證 |
| 8 | `rf_mcp` | 9 | 未驗證 |
| 9 | `rf_rot` | 8 | 未驗證 |
| 10 | `rf_pip` | 10 | 未驗證 |
| 11 | `rf_dip` | 11 | 未驗證 |
| 12 | `th_cmc` | 12 | 未驗證；拇指需特別確認 |
| 13 | `th_axl` | 13 | 未驗證；拇指需特別確認 |
| 14 | `th_mcp` | 14 | 未驗證；拇指需特別確認 |
| 15 | `th_ipl` | 15 | 未驗證；拇指需特別確認 |

候選 vector：

```python
MENAGERIE_TO_REAL_MOTOR_CANDIDATE = [
    1, 0, 2, 3,
    5, 4, 6, 7,
    9, 8, 10, 11,
    12, 13, 14, 15,
]
```

這個 vector 來自本次部署指南提供的 real motor ordering 假設，不是 repository 內已
證實的 calibration artifact。Phase 8/9 必須實作雙向 permutation、exact round-trip
test，並以一次只動一個 motor 的方式確認 joint identity、正負方向、offset 與 soft
limits。特別是 `th_cmc/th_axl/th_mcp/th_ipl` 不得只靠名稱猜測。

## 10. Phase 1 應建立的純函式與 tests

Phase 1 應只新增純 NumPy deployment contract，不修改 training environment。建議
`deploy_contract.py` 從 MuJoCo model 或 export manifest 接收 actuator bounds，不把
16 組 limits 再複製成散落常數。

必要 constants：

```text
NUM_JOINTS = 16
ACTOR_FRAME_DIM = 35
ACTOR_HISTORY_LEN = 3
ACTOR_OBS_DIM = 105
PROPRIO_FRAME_DIM = 32
PROPRIO_HISTORY_LEN = 30
PRIV_INFO_DIM = 9
CONTROL_DT = 0.05
ACTION_SCALE = 1 / 24
```

必要純函式：

- `normalize_joint_position(value, lower, upper)`
- `denormalize_joint_position(value, lower, upper)`
- `normalize_previous_target(value, lower, upper)`
- `denormalize_previous_target(value, lower, upper)`
- `build_actor_frame(measured_q, previous_target, target_axis, lower, upper)`
- `build_proprio_frame(measured_q, previous_target, lower, upper)`
- `integrate_incremental_action(previous_target, action, action_scale,
  target_lower, target_upper, rollout_gain=1.0)`
- `validate_axis(axis)`
- model/manifest bounds construction與 contract validation helper

History class 必須提供：

- `reset(current_frame)`：所有 slot 填入 current frame；
- `push(current_frame)`：丟棄 oldest，將 current 寫到 newest；
- `as_array_oldest_first()`；
- `flatten_oldest_first()`。

Phase 1 tests 至少應覆蓋：

1. constants 與 `[35, 105, 32, 30, 9, 16]` dimensions；
2. output dtype 與不必要 dtype promotion；
3. actor/proprio frame 欄位順序與 shape；
4. history oldest-first shift；
5. reset 使用 current frame 填滿，禁止 zero-fill；
6. action 先 clip 到 `[-1, 1]`；
7. target 再 clip 到 actuator bounds；
8. `rollout_gain` 只縮放 incremental delta；
9. joint/target normalization round-trip max error `< 1e-6`；
10. axis normalization、zero norm/NaN/Inf rejection；
11. 所有 public function 的 NaN/Inf、incorrect shape 與 mismatched bounds rejection；
12. bounds width 必須為正、lower/upper shape 必須是 `(16,)`；
13. input 不被 in-place 修改；
14. manifest/model bounds 與 actuator order 一致性。

## 11. Phase 0 結論與後續風險

- 現有 task、reward、cache reset、30 mm termination、400-step truncation 與 playback
  contract 均有 repository code/tests 支持；Phase 0 沒有改動它們。
- 現有 105-D actor observation 含 simulator-only ball world position，不能部署。
- 現有 frame 的 target 是 raw radians，而且 post-action observation 讀到 `target_t`；
  目標 HORA contract 要 normalized `target_(t-1)`，兩者都是 Phase 2 的高風險 parity
  邊界。
- HORA runtime 已支援 `critic_info` 與 `proprio_hist` info keys，不需要發明新 protocol。
- Stage-2 checkpoint 已保存 student model、history normalizer 與 model runtime config，
  但不保存 env/reward/DR owner config；export manifest 必須補齊部署所需 contract。
- Repository 沒有已驗證的 LEAP real-motor mapping。候選 mapping 在逐關節校準完成前
  不得用於 live policy command。
- Phase 1 可安全地先建立純 NumPy contract 與 unit tests；不應提前修改 environment、
  開始 HORA training、載入真機或建立 PR。

## 稽核依據

- `src/unilab/assets/robots/leap_hand/leap_hand.xml`
- `src/unilab/assets/robots/leap_hand/scene_ball.xml`
- `src/unilab/envs/manipulation/leap_inhand/base.py`
- `src/unilab/envs/manipulation/leap_inhand/ball_rotation_0730.py`
- `src/unilab/envs/manipulation/allegro_inhand/base.py`
- `src/unilab/envs/manipulation/allegro_inhand/rotation.py`
- `src/unilab/base/np_env.py`
- `src/unilab/algos/torch/hora/observations.py`
- `src/unilab/algos/torch/hora/models.py`
- `src/unilab/algos/torch/hora/rsl_rl.py`
- `src/unilab/algos/torch/hora/distill.py`
- `src/unilab/algos/torch/hora/distill_config.py`
- `scripts/train_hora_distill.py`
- `scripts/deploy/sim_prototype.py`
- `scripts/deploy/export_deploy_config.py`
- `tests/envs/test_leap_inhand.py`
- `tests/envs/test_leap_inhand_0730.py`
- `tests/algos/test_hora_contract.py`
- `tests/scripts/test_train_scripts.py`
