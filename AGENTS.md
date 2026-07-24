# UniLab Agent Principles

**Always use `uv run`, not python**.

UniLab 是一个 **高性能、模块化、contract 驱动** 的 RL infrastructure 仓库。

## Core Principles

1. **Contract first**: 不为了一次通过绕过 env / backend / runner contract。
2. **Fix at owner layer**: `scripts/` 只组装流程，不承载长期业务规则。
3. **Config first**: task / reward / backend 优先通过 Hydra + registry 表达。
4. **Backend isolation**: MuJoCo / Motrix 差异留在 backend 适配层和配置层。
5. **Evidence only**: support claim 只写仓库里已有的注册、配置、测试或 benchmark 事实。
6. **Validate near risk**: 在最接近风险的边界补验证，不只跑顶层命令。
7. **Cold-path asset access only**: asset/XML/model metadata 只允许在 init / materialization / cache 等低频路径处理；热路径不能解析 asset，也不能靠 `getattr` / `hasattr` 探测 backend 私有能力。

## High-Risk Areas

| 区域 | 不可破坏的不变量 |
|------|----------------|
| Env  | `NpEnvState.obs` 必须是 dict；`reset()` 返回 `(obs_dict, info_dict)`；`obs_groups_spec` 影响 wrapper 和 learner 维度。 |
| Config / Reward | reward 通过 Hydra 注入；后端切换必须通过 `task=<task>/<backend>` 选择 owner YAML，`training.sim_backend` 只是 owner YAML 的身份字段，不能单独 override 来切后端。算法超参数直接走 YAML compose，不经 Python 层解释。 |
| Backend | backend-specific 逻辑留在 backend / env 适配层，不向训练脚本扩散。env 层只能调用 `SimBackend`（`base.py`）中已声明的方法；若某方法只在 MuJoCo 或 Motrix 中存在，必须先将其加入 `SimBackend` 抽象接口（可抛 `NotImplementedError`），禁止直接在 env 里调用 backend 子类的私有方法（即"功能泄漏/feature leakage"）。新增 backend 专有能力时，需同步更新 `SimBackend`。 |
| Asset / Metadata | `ASSETS_ROOT_PATH`、`model_file`、XML / asset 元数据只允许在 init / materialization / cache 等低频路径访问；`step/reset/domain randomization` 等热路径不得解析 asset 或基于 asset 元数据做运行时分支。 |
| Asset / XML structure | `<keyframe>` 必须放在 task-level XML（`scene_*.xml` 或 `locomotion_task.xml` 等 fragment），**禁止放进 robot.xml**。robot.xml 是纯机器人描述（body / joint / actuator / sensor），跟 task / 场景无关；keyframe 是 task 起始姿态，属于场景或 task 资源。motrix 后端需要 keyframe 时通过 `scene.fragment_files` 引用 fragment XML。 |
| Async | 不绕开 runner lifecycle，也不另起 collector / learner 同步协议。 |
| Sim2Sim 契约 | 跨后端 play 时，影响策略 I/O / 网络结构的字段必须跨后端一致；不一致即 `CrossBackendIncompatibleError`。详见下方 Sim2Sim 章节。 |

## Sim2Sim 跨后端配置契约

`src/unilab/training/sim2sim.py` 按 dotted path 维护三类字段：

- **DENYLIST**（差异即 `CrossBackendIncompatibleError`）：`algo.obs_groups`、`env.control_config.action_scale`、`algo.policy.actor_hidden_dims` / `critic_hidden_dims`、`algo.empirical_normalization` / `algo.obs_normalization`、`env.sampling_mode`。`env.*` 子集对**任一方向**的不对称出现也 fail-closed；`algo` 专属字段目标缺省时按设计跳过（跨算法合法）。
- **WARNING_LIST**：`reward.*`、`env.control_config.simulate_action_latency`、`env.ctrl_dt`。
- **ALLOWLIST**（自由覆盖）：`training.sim_backend`、`env.scene`、`training.play_steps`、`env.domain_rand`、`env.noise_config`、`env.commands.vel_limit`。

训练时 `ExperimentTracker.start()` 把上述字段写入 `run_config.json` 的 `contract_snapshot`（不改 checkpoint 格式，旧 run 无 snapshot 时 fallback + warning）；五个 play 入口在建 env 前调用 `resolve_sim2sim_config` 校验，并用 `policy_load_dim_guard` 包裹 checkpoint 加载以把维度不匹配的隐晦报错重抛为显式诊断。设 `training.sim2sim_strict=false` 可把 DENYLIST 差异降级为 warning（默认 `true`）。DENYLIST 字段在每个后端 owner 配置中显式声明并保持跨后端一致（范例：`conf/ppo/task/g1_walk_flat/{mujoco,motrix}.yaml`）；跨后端契约审计见 `scripts/audit_sim2sim_contracts.py`。

## Dexterous Manipulation Training Guidance

本節是後續靈巧操作任務調整的研究基礎，不是不可變 contract。任何方法都必須以
checkpoint、metrics、影片、跨 seed 實驗或消融結果驗證，並保留既有 run 作為
baseline，禁止只因論文或直覺直接宣稱有效。

1. **Action representation**: 優先使用 joint-position target 加 PD/PID，而非讓
   policy 直接輸出 torque；同時必須保留真實 joint limit、velocity limit、actuator
   force/torque limit 與有效 self-collision，禁止讓 RL 利用不實物理。
2. **Phase decomposition**: 困難 manipulation 應拆成可觀測、可量測的階段，並使用
   curriculum reset 從 capture/return 等後段技能逐步擴展到完整動作。階段轉換應使用
   連續多步成立的事件 gate，不能只靠單一瞬時 reward 或角速度尖峰。
3. **Reward semantics**: dense shaping 只協助探索真正困難的局部技能，並搭配
   milestone bonus；避免堆疊大量 penalty。旋轉任務必須獎勵目標軸 signed angular
   speed/rotation progress，並抑制 orthogonal rotation，不能在需要旋轉的階段獎勵
   `exp(-||angular_velocity||^2)`；低角速度只應在 return/capture 階段要求。
4. **Diagnostics before tuning**: reward 變更前先記錄各 failure 類別、phase transition
   成功率、各 phase 耗時、目標軸與非目標軸角速度、trajectory error、contact event、
   action std/entropy 與 capture stability。聚合 total reward 不足以判斷卡點。
5. **Curriculum progression**: 固定 step threshold 可作 baseline，但正式設計優先考慮
   rolling performance gate；成功率連續達標才升級。高 level 仍保留早期 reset mixture，
   避免 catastrophic forgetting。resume 必須確認 curriculum progress 是否被還原。
6. **Exploration control**: exploration variance 往往比小幅網路寬度調整更重要。持續
   監控 policy standard deviation 與 entropy；std 無界增長且行為退化時，先處理分布
   尺度、entropy 設定或 schedule，不以增加訓練時間掩蓋。
7. **Nominal then robust**: 先在固定 nominal dynamics 學會完整任務，再逐步增加 mass、
   friction、COM、actuator、latency、observation noise 與尺寸 randomization。task
   acquisition 與 robustness 應分階段評估。
8. **Ablation discipline**: reward、observation、curriculum 與 regularization 的修改應做
   單變量 ablation，至少比較多個 random seeds；同時報告 success rate、time to first
   success、failure distribution 與最終穩定度，不只報最高 return。
9. **Observations**: 快速 launch、impact、roll 與 capture 任務應評估 joint velocity、
   object pose/velocity、previous action、contact、phase、target axis 等資訊。Observation
   contract 改變會使舊 checkpoint 不相容，必須同步更新雙 backend owner 與測試。
10. **Demonstrations**: grasp cache 是 reset-state prior，不是 state-action demonstration。
    BC/DAPG 需要 observation-action 軌跡；object-centric trajectory 可作 tracking target，
    但不能直接當 BC。只有 curriculum 仍無法探索 launch/rebound/roll 時，才優先投資
    scripted demonstration、BC，再做 PPO fine-tuning 或 decaying demo-gradient 方法。
11. **HER/off-policy boundary**: PPO 沒有 replay buffer，HER 也要求 goal-conditioned
    observation 與可重算 reward。SAC/TD3+HER 應作為獨立實驗，不得直接塞入含 phase
    state、one-shot bonus 與 timeout 的 PPO 任務。
12. **Visual and physical acceptance**: 每個 curriculum 邊界附近都要比較 checkpoint，
    並以原生 viewer 檢查接觸、穿透、動作順序與 reward hacking。影片看似成功不能取代
    collision、actuator limit、event metrics 與跨 backend 驗證。
13. **Stage training budgets**: 尚未確認方向的 reward、observation、curriculum 或控制
    實驗，必須先使用足以判斷趨勢的最小訓練預算；`num_envs`、`num_steps_per_env`、
    `max_iterations` 與 `save_interval` 應依單次 iteration 成本、統計需求和 checkpoint
    比較密度調整，不得把高成本長訓練當作預設。先以 smoke 驗證整合，再做短程診斷與
    小規模比較；只有 metrics、checkpoint 與 viewer 證據共同顯示方向正確且學習曲線尚未
    飽和時，才擴大到正式長訓練。`load_run`、`checkpoint`、warm-start 與 exploration
    `init_std` 皆為依實驗目的選用的工具，不是每次小規模試驗的固定參數。

## Pointers

- PPO: `scripts/train_rsl_rl.py`
- MLX PPO: `scripts/train_mlx_ppo.py`
- APPO: `scripts/train_appo.py`
- SAC / TD3: `scripts/train_offpolicy.py`
- env contract: `src/unilab/base/np_env.py`
- backend contract: `src/unilab/base/backend/base.py`
- training run helpers: `src/unilab/training/run.py`
- visualization helpers: `src/unilab/visualization/`
- shared numeric helpers: `src/unilab/utils/rotation.py`, `src/unilab/utils/geometry.py`
- MLX rotation helpers: `src/unilab/algos/mlx/common/rotation.py`
- config schema: `src/unilab/structured_configs.py`
- async runner: `src/unilab/ipc/async_runner.py`
- sim2sim 跨后端契约: `src/unilab/training/sim2sim.py`

## GitHub CLI (gh) 速查

### Issue 查看
```bash
gh issue view <number>
gh api repos/<owner>/<repo>/issues/<number> --jq '.body'
```

### PR 创建与管理
```bash
gh pr create --title "标题" --body "内容" --base main
gh pr list
gh pr view
```

### PR Gate

创建或更新 PR 前必须满足：

1. 最终提交已经完成，且 `git status --short --branch` 确认工作树干净。
2. 最终提交已经通过 `make test-all`。
3. 如果用户明确说明已经跑过 `make test-all`，不要重复跑；但必须在 PR body 的 Validation 里记录 `make test-all` 已完成。
4. 如果 `make test-all` 未通过且用户没有明确 override，不要创建或更新 PR。

### CI 工作流查看
```bash
gh run list
gh run list --workflow=<workflow-name>
gh run view <run-id>
gh run list --status=failure
```

### 常用组合
```bash
gh api repos/unilabsim/UniLab/issues/174 --jq '.title, .body'
git push -u origin fix/issue-174-mlx-ppo-config-alignment
gh pr create --title "fix: xxx" --body "Fixes #174" --base main
```

## Context

- 架构标准与验证详情：[docs/sphinx/source/zh_CN/4-developer_guide/0-index.md](docs/sphinx/source/zh_CN/4-developer_guide/0-index.md)
- 协作流程与 PR 规范：[docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md](docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md)
- 开发者入口（环境、命令、提交规范）：[CONTRIBUTING.md](CONTRIBUTING.md)
- 文档本地构建与发布到 UniLab-doc：[docs/sphinx/README.md#本地发布到-unilab-doc](docs/sphinx/README.md#本地发布到-unilab-doc)


User Instructions

These instructions apply to all future work in this repository:

1. The assistant may read any file without requesting permission.
2. Commands whose only purpose is reading or searching files may be executed without requesting permission.
3. Before executing any other command, the assistant must obtain the user's explicit consent.
4. Before creating, modifying, renaming, moving, or deleting any file or program code, the assistant must obtain the user's explicit consent.
5. Permission applies only to the specific command or file change described when consent is requested. Further commands or changes require new consent unless the user explicitly grants broader permission
