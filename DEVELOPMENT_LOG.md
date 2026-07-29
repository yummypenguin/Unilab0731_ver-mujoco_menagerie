# Development Log

This file records repository development performed with the coding assistant.
It was introduced after the LEAP Hand work had started, so the first entries
are retrospective and are based on the working tree, generated artifacts, and
validation output from the development session.

## Logging Rules

1. Read the applicable `AGENTS.md` before starting any repository task.
2. Record every completed code or asset addition, deletion, or modification in
   this file during the same task.
3. Each entry must state the scope, important design decisions, files or areas
   changed, validation performed, and known limitations.
4. Do not claim support based only on intent. Record registry, owner config,
   test, smoke-run, or benchmark evidence.
5. Preserve earlier entries. Corrections must be added as a new dated note
   instead of silently rewriting development history.

## 2026-07-13 - Repository Collaboration Rules (Retrospective)

### Scope

- Added persistent repository instructions for assistant access and approval.
- Reading files and running read-only search commands do not require approval.
- Non-read-only commands and file changes require explicit user approval for
  the stated scope.

### Files

- `USER_INSTRUCTIONS.md`
- `AGENTS.md` (user-instruction section was already present in the working tree)

### Notes

- These rules remain authoritative for future work alongside the repository
  engineering principles in `AGENTS.md`.

## 2026-07-13 - LEAP Hand Asset Import (Retrospective)

### Scope

- Imported the LEAP Hand model from the user-provided
  `LEAP_Hand_Sim-master.zip` source package.
- Preserved source URDFs and the upstream MIT license.
- Added a fixed-base 16-DoF MuJoCo model with position actuators, stable link
  and joint names, visual meshes, and simplified primitive collision geometry.
- Added LEAP-owned cube and ball assets so the task does not depend on Allegro
  or Sharpa object models.
- Added cube and ball task scenes. Task keyframes remain in scene XML and are
  not embedded in the robot XML.
- Converted the source scale-1.0 cube grasp cache to the UniLab/MuJoCo joint
  order and converted object quaternions from `xyzw` to `wxyz`.

### Files

- `src/unilab/assets/robots/leap_hand/leap_hand.xml`
- `src/unilab/assets/robots/leap_hand/cube.xml`
- `src/unilab/assets/robots/leap_hand/ball.xml`
- `src/unilab/assets/robots/leap_hand/scene.xml`
- `src/unilab/assets/robots/leap_hand/scene_ball.xml`
- `src/unilab/assets/robots/leap_hand/assets/*.stl`
- `src/unilab/assets/robots/leap_hand/source/*.urdf`
- `src/unilab/assets/robots/leap_hand/caches/cube_grasp_s10_1k.npy`
- `src/unilab/assets/robots/leap_hand/SOURCE.md`
- `src/unilab/assets/robots/leap_hand/LICENSE.txt`

### Import Tool Fix

- Updated `src/unilab/tools/import_robot.py` to detect whether the imported
  robot root has a free joint.
- The tuning scene now adds a temporary height joint only for floating-base
  robots. This prevents fixed-base hands from receiving an invalid synthetic
  base joint during tuning-scene compilation.
- Added fixed-base/free-joint regression coverage in
  `tests/test_import_robot.py`.

### Validation

- Both cube and ball scenes compiled with MuJoCo.
- Verified `nq=23`, `nv=22`, `nu=16`, one task-level keyframe, fixed hand base,
  aligned joint/actuator order, finite state after stepping, cache shape
  `(1024, 23)`, normalized object quaternions, and LEAP-owned object names.

## 2026-07-13 - LEAP In-Hand MuJoCo PPO Task (Retrospective)

### Scope

- Added `LeapInhandRotation` as a registry-backed manipulation environment.
- Reused the established Allegro 16-DoF in-hand rotation behavior while
  keeping LEAP-specific model names and joint mappings in the LEAP adapter.
- Generalized the Allegro base/object/log/grasp-generation names through class
  ownership fields so the shared behavior does not hard-code Allegro model
  names.
- Added a MuJoCo PPO owner with LEAP-specific control, reward, camera, cache,
  and disabled initial online domain-randomization settings.
- Registered the LEAP manipulation package during registry bootstrap.

### Files

- `src/unilab/envs/manipulation/leap_inhand/__init__.py`
- `src/unilab/envs/manipulation/leap_inhand/base.py`
- `src/unilab/envs/manipulation/leap_inhand/rotation.py`
- `src/unilab/envs/manipulation/__init__.py`
- `src/unilab/envs/manipulation/allegro_inhand/base.py`
- `src/unilab/envs/manipulation/allegro_inhand/rotation.py`
- `conf/ppo/task/leap_inhand/mujoco.yaml`
- `tests/envs/test_leap_inhand.py`

### Contract Decisions

- Policy observation dimension: 105.
- Action dimension: 16.
- Control action scale: `1/24`.
- MuJoCo physics timestep: `1/120` seconds.
- Control timestep: `0.05` seconds.
- Actor and critic hidden dimensions: `[512, 256, 128]`.
- The task starts with domain randomization disabled until baseline behavior is
  established.

### Validation

- MuJoCo reset/step contract passed with dict observations and finite states,
  rewards, and termination arrays.
- A 16-environment PPO smoke run completed and produced checkpoints.
- A 60-frame MuJoCo playback video was generated.
- Targeted tests previously reported 145 passed; a broader regression set
  reported 76 passed.
- Ruff passed. Pyright reported no errors. Mypy reported four pre-existing
  Windows `fcntl.flock` errors outside the LEAP changes.

## 2026-07-13 - LEAP Hand Motrix PPO and Sim2Sim Support (Retrospective)

### Scope

- Registered `LeapInhandRotation` for the Motrix backend using the same env
  adapter and policy-facing contract as MuJoCo.
- Added a Motrix PPO owner that inherits the MuJoCo owner and overrides only
  backend identity, physics timestep, playback spacing, and the explicitly
  measured environment count.
- Reused the same LEAP robot, cube, ball, and scene assets. No backend-specific
  asset fork was needed.
- Added Motrix reset/step coverage and a MuJoCo-to-Motrix Sim2Sim resolver test.
- Added LEAP ordering/labels to the support-matrix generator and regenerated
  the Chinese support matrix.
- Added support-matrix evidence tests for both PPO backends.

### Files

- `src/unilab/envs/manipulation/leap_inhand/rotation.py`
- `conf/ppo/task/leap_inhand/motrix.yaml`
- `tests/envs/test_leap_inhand.py`
- `tests/training/test_sim2sim_resolver.py`
- `src/unilab/utils/support_matrix.py`
- `tests/scripts/test_support_matrix.py`
- `docs/sphinx/source/zh_CN/5-reference/5-support_matrix.md`

### Contract Decisions

- Motrix physics timestep: `0.01` seconds.
- Motrix default environment count: 1024, retained after a successful default
  capacity smoke run.
- MuJoCo and Motrix keep identical observation groups, action scale, policy
  dimensions, network dimensions, normalization, reward, and control timestep.
- Motrix recording is not a completion requirement because renderer stability
  is separate from the environment/training contract.

### Validation

- Motrix materialization found 18 links, 16 hand joints, 16 actuators,
  `palm_lower`, and `leap_object`.
- LEAP/config/Sim2Sim target set: 11 passed, 122 deselected.
- PPO Sim2Sim audit verdict for `leap_inhand`: `TRANSFERABLE`, with no denylist
  or warning-list differences.
- 16-environment PPO smoke: 128 steps at approximately 332 steps/s.
- Default 1024-environment PPO smoke: 8192 steps at approximately 5823 steps/s;
  one iteration completed in approximately 1.41 seconds.
- Motrix checkpoint evaluation completed for 60 headless steps.
- A MuJoCo checkpoint from run
  `2026-07-13_19-06-56_mujoco` loaded into Motrix and completed 60 headless
  Sim2Sim steps.
- Ruff passed. Pyright reported 0 errors and 0 warnings for the LEAP package.
- Allegro, Sharpa, config, and Sim2Sim regression set: 173 passed, 5 deselected.
- Support-matrix and documentation consistency set: 25 passed.
- `git diff --check` reported no whitespace errors.

### Generated Artifacts

- 16-env Motrix smoke run:
  `logs/rsl_rl_ppo/LeapInhandRotation/2026-07-13_22-26-54_motrix`
- 1024-env Motrix smoke run:
  `logs/rsl_rl_ppo/LeapInhandRotation/2026-07-13_22-29-16_motrix`
- The 1024-env run produced `model_0.pt` (5,294,975 bytes).

### Known Limitation

- On the current Windows 11 system with Intel Arc graphics, Motrix recording
  has previously failed in WGPU DX12/Vulkan with GPU out-of-memory or renderer
  panics. Training, checkpoint loading, headless evaluation, and Sim2Sim are
  validated independently of that renderer issue. MuJoCo remains the reliable
  video-output path on this machine.

## 2026-07-13 - Development Log Introduced

### Scope

- Added this file at the user's request.
- Reconstructed all earlier LEAP Hand development entries from repository
  changes and validation evidence available in the current task.
- Established the requirement to read `AGENTS.md` before every task and update
  this log whenever code or assets are added, removed, or modified.

### Files

- `DEVELOPMENT_LOG.md`

### Validation

- Documentation-only addition; no runtime behavior changed.

## 2026-07-14 - User Acceptance: Full Training and Rendering

### Scope

- Recorded user-performed end-to-end acceptance of the LEAP Hand PPO task on
  both supported simulation backends.
- No program code, configuration, or assets changed in this entry.

### Commands Validated by the User

- `uv run train --algo ppo --task leap_inhand --sim motrix`
- `uv run train --algo ppo --task leap_inhand --sim mujoco`
- `uv run eval --algo ppo --task leap_inhand --sim mujoco --load-run -1`
- `uv run eval --algo ppo --task leap_inhand --sim motrix --load-run -1 --render-mode interactive`

### Result

- Full PPO training completed for both MuJoCo and Motrix.
- MuJoCo evaluation produced a clear rendered result.
- Motrix interactive evaluation produced a clear rendered result.
- The trained behavior shown by both backends met the user's requirements.

### Limitation Clarification

- The earlier Intel Arc/WGPU failure applies specifically to the Motrix
  `record` path observed in that earlier run.
- Motrix `interactive` rendering is now explicitly user-validated on the same
  system and must not be described as unsupported or generally unstable.

## 2026-07-14 - PowerShell Command Reference

### Scope

- Added a copy-and-paste command reference for the user's Windows 11
  PowerShell workflow.
- Covered dependency synchronization, LEAP Hand smoke and full PPO training,
  run/checkpoint inspection, explicit checkpoint evaluation, cross-backend
  Sim2Sim playback, resume training, TensorBoard, video handling, and
  Motrix/WGPU diagnostics.
- Documented that latest-run selection is task-wide and does not filter by
  backend, so explicit run ids are preferred for comparisons.

### Files

- `USEFUL_COMMANDS.md`
- `DEVELOPMENT_LOG.md`

### Validation

- Commands were derived from the checked-in CLI, owner configs, checkpoint
  resolver, repository documentation, and the user-validated LEAP workflow.
- Documentation-only change; no training or runtime code changed.

## 2026-07-14 - Known Viser Material and Geometry Display Issue

### Observation

- In the Viser interactive interface, the LEAP Hand and its cube appear mostly
  gray and differ substantially from the MuJoCo MP4 rendering.
- This is a visualization-path issue and is not evidence of a checkpoint,
  policy, or simulation-state failure.

### Likely Causes

- `MujocoViserScene` reads per-geom RGBA values without fully resolving colors
  assigned through MuJoCo material ids, so material-backed hand and object
  colors can fall back to gray.
- The Viser scene currently adds all MuJoCo geoms without applying the usual
  visual-group mask. LEAP collision primitives in group 3 can therefore be
  shown over the visual meshes in group 2.
- MuJoCo's native renderer used for MP4 output handles material resolution and
  default geom-group visibility, which explains the visual difference.

### Deferred Work

- Resolve effective geom color through `geom_matid` and `mat_rgba` when a
  material is assigned.
- Hide collision geom group 3 by default while retaining an optional collision
  visualization control for debugging.
- Add regression coverage for material colors, geom-group visibility, the LEAP
  cube color, and collision-geometry toggling.

### Status

- The user requested that this issue be recorded but not fixed yet.
- Only `DEVELOPMENT_LOG.md` was changed; no visualization code, configuration,
  model asset, or runtime behavior was modified.

## 2026-07-14 - LEAP Hand Toss, Passive Rebound, and Catch Curriculum

### Scope

- Added the registry-backed `LeapInhandToss` PPO task for MuJoCo and Motrix.
- Implemented a five-phase state machine: thumb support, ballistic flight,
  non-thumb impact, passive return, and stable capture.
- Added curriculum resets that first train capture, then return, then flight,
  and finally the complete thumb-launch sequence.
- Added a task-owned scene with index, middle, ring, and thumb contact sensors.
  The three non-thumb cube contact pairs use task-local solver parameters for
  passive rebound without changing the original LEAP in-hand task.
- Added matching MuJoCo and Motrix Hydra owners, command examples, support
  matrix entries, environment tests, and Sim2Sim contract coverage.

### Files

- `src/unilab/envs/manipulation/leap_inhand/toss.py`
- `src/unilab/envs/manipulation/leap_inhand/__init__.py`
- `src/unilab/envs/manipulation/allegro_inhand/rotation.py`
- `src/unilab/assets/robots/leap_hand/scene_toss.xml`
- `conf/ppo/task/leap_inhand_toss/mujoco.yaml`
- `conf/ppo/task/leap_inhand_toss/motrix.yaml`
- `tests/envs/test_leap_inhand.py`
- `tests/training/test_sim2sim_resolver.py`
- `src/unilab/utils/support_matrix.py`
- `tests/scripts/test_support_matrix.py`
- `docs/sphinx/source/zh_CN/5-reference/5-support_matrix.md`
- `USEFUL_COMMANDS.md`
- `DEVELOPMENT_LOG.md`

### Reward and Curriculum Decisions

- Policy observation dimension: 75; action dimension: 16.
- Dense reward is phase-specific and multiplied by `ctrl_dt`: thumb support
  and guard pose, ballistic position/velocity tracking with low cube angular
  velocity, return progress toward the grasp point, and stable capture.
- The only continuous negative regularizer is action-rate cost with scale
  `0.001`. Torque, linear velocity, angular velocity, workspace, and apex
  penalties are not accumulated every step.
- One-time bonuses are ready `0.2`, valid launch `0.5`, passive rebound `2.0`,
  and stable capture `10.0`. A single `10.0` failure cost terminates workspace
  escape, excessive apex, drop, or phase timeout.
- Passive rebound requires contact by any index/middle/ring fingertip, impact
  speed at least `0.05 m/s`, contacted fingertip speed at most `0.08 m/s`, and
  reversal toward the palm at at least `0.03 m/s`; all three fingers need not
  contact simultaneously.
- The allowed apex rise is the minimum nominal palm-to-non-thumb-fingertip
  distance. Both validated backends measured this limit as approximately
  `0.0874 m`; the reference parabola targets 20 percent of that height.
- Curriculum level thresholds are control steps 0, 1000, 2500, and 4000. The
  final reset mixture is 70 percent full support, 15 percent flight, 10 percent
  return, and 5 percent capture to retain earlier skills.

### Validation

- Ruff passed for all changed Python files in this task.
- Targeted environment, XML, registry, curriculum, passive-rebound, Sim2Sim,
  and support-matrix tests: 52 passed in 21.86 seconds.
- MuJoCo 16-environment, one-iteration PPO smoke completed: 128 steps at 252
  steps/s; run `2026-07-14_16-37-21_mujoco`.
- Motrix 16-environment, one-iteration PPO smoke completed: 128 steps at 222
  steps/s; run `2026-07-14_16-37-46_motrix`.
- Both smoke runs resolved 75 actor/critic inputs, 16 actions, curriculum level
  0 capture resets, finite reward output, and the same `0.0874 m` hand-length
  limit.
- The generated Chinese support matrix records tested PPO support for both
  backends. The first non-write generator attempt failed only because the
  Windows shell used CP1252; the UTF-8 `--write` invocation completed.

### Known Limitations

- A one-iteration smoke validates contracts and execution, not acquisition of
  the complete toss-and-catch behavior. Long training and rendered evaluation
  are still required to judge curriculum progression and reward balance.
- The task-local contact solver and passive-impact thresholds are initial
  physical tuning values. Learned trajectories may justify later adjustment,
  but changes should be based on logged event rates and playback evidence.

## 2026-07-14 - Assisted Rebound and Curriculum Checkpoint Visibility

### Scope

- Revised the toss task after the user clarified that index, middle, or ring
  fingers may actively help redirect the cube toward the palm.
- Replaced the passive-impact eligibility rule with an assisted-rebound rule.
  A valid impact now requires any non-thumb contact and sufficient incoming
  cube speed, but places no upper bound on the contacted fingertip speed.
- Retained the subsequent velocity-reversal requirement, so contact alone does
  not earn the rebound event bonus.
- Set the task owner checkpoint interval to 25 PPO iterations and documented
  checkpoint ranges for all four curriculum levels.

### Files

- `src/unilab/envs/manipulation/leap_inhand/toss.py`
- `src/unilab/assets/robots/leap_hand/scene_toss.xml`
- `conf/ppo/task/leap_inhand_toss/mujoco.yaml`
- `tests/envs/test_leap_inhand.py`
- `tests/training/test_sim2sim_resolver.py`
- `USEFUL_COMMANDS.md`
- `DEVELOPMENT_LOG.md`

### Behavior Correction

- The earlier entry's statement that rebound must use a nearly static finger
  barrier is superseded by this entry.
- The event is now named `assisted_rebound` in code and configuration. Its
  bonus remains `2.0`, and the TensorBoard metric remains
  `toss/rebound_event` for continuity.
- With 8 environment control steps per PPO iteration, checkpoints every 25
  iterations correspond to 200 control steps. This leaves multiple snapshots
  in Level 0, Level 1, Level 2, and Level 3 during a 1000-iteration run.

### Validation

- Ruff passed for the changed environment and test files.
- Targeted environment, config, Sim2Sim, and support-matrix tests: 52 passed in
  13.05 seconds.
- MuJoCo 16-environment, one-iteration PPO smoke completed at 182 steps/s; run
  `2026-07-14_16-51-50_mujoco` produced `model_0.pt`.
- Motrix 16-environment, one-iteration PPO smoke completed at 203 steps/s; run
  `2026-07-14_16-53-51_motrix` produced `model_0.pt`.
- Both backends retained 75 policy observations, 16 actions, finite Level 0
  rewards, and the same approximately `0.0874 m` hand-length limit.

## 2026-07-14 - Toss Checkpoint Command Reference Expansion

### Scope

- Expanded the Windows PowerShell reference for `LeapInhandToss` checkpoints.
- Added commands to list every checkpoint in the latest run, refresh the list
  every 10 seconds, automatically resolve and play the latest checkpoint, play
  an explicitly selected checkpoint, and record an explicitly selected
  checkpoint.
- Updated the task description from passive rebound to assisted rebound.

### Files

- `USEFUL_COMMANDS.md`
- `DEVELOPMENT_LOG.md`

### Validation

- Documentation-only change; no training, environment, or runtime code was
  modified or executed.

## 2026-07-14 - Relaxed Toss Phase Timeouts

### Scope

- Relaxed every `LeapInhandToss` phase timeout to give the policy more time to
  arrange support, contact the finger guard, redirect the cube, return it to
  the palm, and stabilize the final grasp.
- Increased the global episode duration from 8 to 15 seconds so it does not
  truncate the complete sequence before the phase-specific limits can apply.
- Kept the one-time failure penalty at `10.0`; only the time available before
  timeout changed.

### Values

- SUPPORT: 4.0 to 5.0 seconds (100 control steps).
- FLIGHT: 0.8 to 1.2 seconds (24 control steps).
- IMPACT: 0.25 to 0.5 seconds (10 control steps).
- RETURN: 1.5 to 2.5 seconds (50 control steps).
- CAPTURE: 2.0 to 3.0 seconds (60 control steps).
- Global episode: 8.0 to 15.0 seconds.

### Files

- `src/unilab/envs/manipulation/leap_inhand/toss.py`
- `conf/ppo/task/leap_inhand_toss/mujoco.yaml`
- `tests/envs/test_leap_inhand.py`
- `tests/training/test_sim2sim_resolver.py`
- `USEFUL_COMMANDS.md`
- `DEVELOPMENT_LOG.md`

### Validation

- Ruff passed for the changed environment and test files.
- Targeted environment, timeout, config, Sim2Sim, and support-matrix tests: 52
  passed in 7.77 seconds.
- MuJoCo 16-environment, one-iteration PPO smoke completed at 601 steps/s; run
  `2026-07-14_17-31-56_mujoco`.
- Motrix 16-environment, one-iteration PPO smoke completed at 370 steps/s; run
  `2026-07-14_17-33-47_motrix`.
- Both backends retained 75 policy observations, 16 actions, finite Level 0
  rewards, zero smoke-run failure events, and the same approximately `0.0874 m`
  hand-length limit.

## 2026-07-14 - Extended Formal Toss Training Command

### Scope

- Updated the PowerShell command reference with the recommended formal
  `LeapInhandToss` training commands for MuJoCo and Motrix.
- The commands use 1024 environments, 3500 PPO iterations, and curriculum
  boundaries `[0, 4000, 9000, 15000]` without changing repository defaults.
- Updated the Level 0 through Level 3 checkpoint examples for the extended
  curriculum while retaining the owner-defined 25-iteration save interval.

### Files

- `USEFUL_COMMANDS.md`
- `DEVELOPMENT_LOG.md`

### Validation

- Documentation-only change; the commands were not executed in this entry.

## 2026-07-14 - Official MuJoCo Viewer Command Reference

### Scope

- Added the `LeapInhandToss` command for the repository's official MuJoCo
  interactive viewer entrypoint, `scripts/play_interactive.py`.
- Documented explicit and latest-checkpoint forms for run
  `2026-07-14_17-37-26_mujoco` and the viewer pause, step, speed, camera, and
  exit controls.
- Clarified that this path uses `mujoco.viewer.launch_passive` and is distinct
  from Viser and recorded MP4 playback.

### Files

- `USEFUL_COMMANDS.md`
- `DEVELOPMENT_LOG.md`

### Validation

- Documentation-only change; the viewer command was not executed.

## 2026-07-15 - Deferred LEAP Hand and Cube Penetration Fix

### Observation

- Native MuJoCo checkpoint playback showed the thumb passing through the
  visible palm and the cube entering visible hand geometry.
- These behaviors are physically invalid and must be corrected before reward
  redesign or further formal training.

### Confirmed Causes

- LEAP hand collision geoms currently use `contype="0" conaffinity="1"`.
  Cube-hand contact is possible, but hand-hand self-collision is disabled, so
  the thumb is not physically blocked by the palm.
- The existing palm collision box is substantially misaligned with the palm
  STL bounds. The translated visual bounds are approximately
  `[-0.1001, -0.1000, -0.0347]` to `[0.0029, 0.0258, 0.0113]`, while the
  current primitive covers a different and thinner region.
- The source URDF specifies a `0.95 N m` effort limit for all 16 joints, but
  the converted MuJoCo position actuators currently have no `forcerange`.
  Large policy target errors can therefore produce unrealistic contact force.
- Simplified collision geometry can also make a physically valid contact look
  penetrated in the viewer when its envelope does not match the visual mesh.

### Deferred Implementation Plan

- Correct the palm collision envelope using aligned primitive geometry.
- Add collision masks that preserve cube contact with every hand collision
  geom while enabling selected distal-thumb-to-palm self-collision. Keep
  mechanically adjacent thumb-base contacts excluded until their geometry is
  proven non-overlapping.
- Add approximately 1 mm contact margins for the relevant hand and cube
  collision geoms.
- Apply the source `[-0.95, 0.95] N m` force range to all LEAP position
  actuators and verify the corresponding Motrix actuator limit behavior.
- Validate all grasp-cache states after the collision change; filter or
  regenerate states that begin with invalid penetration or unstable contact.
- Do not use reward penalties as the primary penetration fix. Physical
  collision geometry, masks, force limits, and solver behavior own this
  invariant.

### Planned Validation

- Compile the cube, ball, and toss scenes and verify joint/actuator ordering.
- Assert collision masks for cube-to-palm, cube-to-all-finger-links, and
  distal-thumb-to-palm contact.
- Assert all actuator force limits match the source URDF.
- Add dynamic MuJoCo tests that drive the cube toward the palm and the distal
  thumb toward the palm, then bound penetration depth and require finite
  simulation state.
- Run MuJoCo and Motrix reset/step tests plus 16-environment, one-iteration PPO
  smoke runs.
- Inspect collision group 3 in the native MuJoCo viewer and compare it with
  visual geometry before accepting the fix.

### Status

- Deferred at the user's request. No robot asset, object asset, environment,
  configuration, reward, or test code was changed, and no smoke run was
  executed in this entry.

## 2026-07-15 - Dexterous Manipulation Research Guidance

### Scope

- Added persistent research guidance to `AGENTS.md` for future dexterous
  manipulation task design and tuning.
- Recorded the preferred action representation, phase curriculum, target-axis
  rotation reward semantics, diagnostic metrics, performance-gated
  progression, exploration monitoring, nominal-to-randomized training,
  ablation discipline, observation design, and the applicability boundaries
  of demonstrations and HER/off-policy methods.
- Required physical validity and native-viewer inspection alongside reward and
  training metrics so learned behavior cannot rely on penetration, unlimited
  actuator force, or other simulation artifacts.

### Files

- `AGENTS.md`
- `DEVELOPMENT_LOG.md`

### Status

- Documentation-only change. No environment, reward, asset, configuration,
  checkpoint, or runtime behavior changed, and no training command was run.

## 2026-07-15 - LEAP Ball Rotation Task Referencing Allegro Behavior

### Scope

- Added the independent `LeapInhandBallRotation` environment and
  `leap_inhand_ball` PPO owner for MuJoCo and Motrix.
- Preserved the existing `LeapInhandRotation` cube task and `LeapInhandToss`;
  neither environment, owner config, nor scene was modified.
- Used only the LEAP-owned `scene_ball.xml` and `ball.xml`. The radius
  (`0.04 m`), mass (`0.05 kg`), and diagonal inertia (`0.0001 kg m^2`) are
  preserved from the user-provided LEAP `assets/ball.urdf`; the task does not
  include or reference `robots/allegro_hand/ball.xml`.
- Kept the Allegro rotation reward weights, target-axis angular-velocity
  bounds, 105-value observation history, 16-value position-target action,
  action scale, PPO network, and PPO optimization settings.

### Model-Binding Adaptations

- Used LEAP body, fingertip, joint, and actuator names through
  `LeapHandBaseEnv`.
- Kept the validated LEAP MuJoCo step (`1/120 s`), Motrix step (`0.01 s`), and
  PD gains (`kp=3.0`, `kd=0.1`).
- Adapted the drop threshold from Allegro's world-space `0.125 m` to LEAP's
  mounted scene height of `0.4 m`; this preserves drop semantics rather than
  copying an incompatible absolute coordinate.
- Reused the existing LEAP cube-grasp state cache as the initial hand-pose
  source because the dedicated sphere has a similar 40 mm radius. This must be
  validated for penetration and stability before formal training; a filtered
  ball-specific cache may still be required.

### Files

- `src/unilab/envs/manipulation/leap_inhand/ball_rotation.py`
- `src/unilab/envs/manipulation/leap_inhand/__init__.py`
- `src/unilab/assets/robots/leap_hand/ball.xml`
- `conf/ppo/task/leap_inhand_ball/mujoco.yaml`
- `conf/ppo/task/leap_inhand_ball/motrix.yaml`
- `tests/envs/test_leap_inhand.py`
- `tests/training/test_sim2sim_resolver.py`
- `src/unilab/utils/support_matrix.py`
- `tests/scripts/test_support_matrix.py`
- `docs/sphinx/source/zh_CN/5-reference/5-support_matrix.md`
- `USEFUL_COMMANDS.md`
- `DEVELOPMENT_LOG.md`

### Validation Status

- Added registry, MuJoCo reset/step, Allegro-contract comparison, Sim2Sim, and
  support-matrix regression coverage.
- Ruff passed for the changed Python files.
- Targeted environment, Sim2Sim, and support-matrix tests: 54 passed in
  10.11 seconds.
- The generated Chinese support matrix was refreshed successfully.
- MuJoCo smoke run `2026-07-15_16-32-29_mujoco` completed 128 steps at
  197 steps/s and wrote `model_0.pt`; it resolved 105 observations and 16
  actions. Its one-iteration work reward was unusually large (`-81.3794`), so
  the reused cube-grasp cache requires physical inspection before formal
  training.
- Motrix smoke run `2026-07-15_16-32-50_motrix` completed 128 steps at
  577 steps/s and wrote `model_0.pt`; it resolved the same 105 observations
  and 16 actions. Its one-iteration work reward was `-4.4755`.

### Clarification

- Earlier wording said the LEAP ball was aligned to the Allegro ball. That was
  misleading: the current `0.0001 kg m^2` inertia is also the value in the
  original LEAP zip's `assets/ball.urdf`. Allegro supplies the task behavior
  reference only; the runtime scene and object remain LEAP-owned assets.
## 2026-07-15 - Isolated LEAP Ball Grasp Cache Generator

### Scope

- Added the independent `LeapInhandBallGrasp` environment and
  `leap_inhand_ball_grasp` PPO owners for MuJoCo and Motrix.
- Kept the running `LeapInhandBallRotation` task, its owner YAML files, and its
  current cube-cache reference unchanged.
- Used the existing LEAP cube cache only as randomized proposal poses. Accepted
  states are settled and re-measured in the LEAP-owned ball scene before being
  written to new ball-cache filenames.
- Added backend-neutral gates for fingertip proximity, thumb-inclusive contact,
  ball height and drift, ball linear/angular speed, joint speed, joint limits,
  finite state, and absolute actuator work.
- Added quaternion normalization, cache-layout validation, and quantized
  duplicate suppression.
- Kept MuJoCo and Motrix outputs separate pending cross-backend validation.

### Files

- `src/unilab/envs/manipulation/leap_inhand/ball_grasp_gen.py`
- `src/unilab/envs/manipulation/leap_inhand/__init__.py`
- `conf/ppo/task/leap_inhand_ball_grasp/mujoco.yaml`
- `conf/ppo/task/leap_inhand_ball_grasp/motrix.yaml`
- `tests/envs/test_leap_inhand.py`
- `USEFUL_COMMANDS.md`
- `DEVELOPMENT_LOG.md`

### Physical Validation Boundary

- The current backend contract does not expose contact penetration depth.
  Penetration filtering was therefore not implemented through backend-private
  APIs. It remains blocked on the separately recorded LEAP collision-geometry,
  self-collision, contact-margin, and actuator-force fixes.
- The new cache is not connected to `leap_inhand_ball`; switching the training
  task to it requires a separate validation and configuration change.

### Validation

- No formatter, test, simulator, training, or cache-generation command was run
  while the user's 1024-environment MuJoCo training process was active.

### Post-Training Runtime Validation

- The cube-cache baseline completed run `2026-07-15_16-43-09_mujoco` at
  iteration 999/1000 with 8,192,000 total steps, mean reward `-1.13`, mean
  episode length `112.52`, and mean action standard deviation `1.07`.
- Ruff passed for the generator, registry import, and LEAP tests. The targeted
  LEAP suite passed: 17 tests in 0.98 seconds.
- A 64-environment, five-iteration generator smoke run completed without an
  import, registry, reset, step, or save-lifecycle error, but accepted zero
  grasps because all four fingertip contact sensors remained false.
- Forward-only MuJoCo inspection of all 1,024 cube-cache rows in
  `scene_ball.xml` found zero fingertip-sensor contacts. The states nevertheless
  averaged 1.69 contacts elsewhere, with minimum contact distance `-0.0111 m`.
  Contacts were concentrated on the middle, ring, and index MCP collision geoms
  and the thumb DIP collision geom.
- Representative-state geometry inspection found severe visual/collision
  misalignment. One thumb fingertip body was near
  `[-0.062, 0.005, 0.591] m`, while `thumb_tip_col` was near
  `[-0.023, -0.087, 0.642] m`, an approximately 0.114 m separation.
- Random joint perturbation did not provide a viable workaround. At `+/-1 rad`,
  only 1 of 5,000 proposals achieved thumb-plus-finger sensor contact without
  an MCP collision.
- Cache generation, the `leap_inhand_ball` cache-path switch, and the requested
  second 1,000-iteration run were stopped. Continuing would train against
  displaced invisible collision geometry and would not address the open-hand
  behavior seen in the official viewer.

### Required Follow-Up

- Correct LEAP collision geometry against the source URDF/mesh bounds and
  complete the previously deferred self-collision and actuator-force work.
- Re-run static, dynamic, viewer, and cross-backend validation before generating
  the ball-specific cache or starting the A/B training run.

## 2026-07-15 - Restore Source LEAP Collision and Runtime Effort Contract

### Source Evidence

- The user-provided `assets/leap_hand/robot.urdf` uses the corresponding STL
  mesh as both visual and collision geometry for every hand link, with identical
  origin and rotation.
- The source Isaac Gym task enables VHACD for those collision meshes and defaults
  `disable_self_collision` to `False`; neither source task YAML defines a custom
  body collision mask.
- Although the URDF declares `0.95 N m`, source
  `leapsim/tasks/leap_hand_rot.py` explicitly overwrites every DOF effort to
  `0.5 N m` at runtime. The user selected this original RL runtime value.

### Changes

- Replaced the converted LEAP box/sphere collision approximations with the exact
  source STL mesh, origin, and rotation paired with each visual geom.
- Restored self-collision-capable contact masks (`contype=1`, `conaffinity=1`)
  without adding custom shapes or collision-mask groups.
- Applied `[-0.5, 0.5] N m` force ranges to all 16 position actuators.
- Added asset-contract assertions for collision/visual mesh identity, world pose,
  contact masks, and actuator force limits.

### Validation Status

- Pending static compilation, dynamic contact, MuJoCo/Motrix reset-step, and
  throughput validation.
- Ball-cache generation and the second 1,000-iteration training run remain
  pending until the restored physical model passes those checks.

## 2026-07-15 - MuJoCo Adjacent-Link Contact Exclusions

### Scope

- Added 16 robot-specific MuJoCo contact exclusions covering only mechanically
  direct parent-child links: four pairs for each of the index, middle, ring,
  and thumb kinematic chains.
- Added an asset regression test that requires the exclusion set to match the
  approved pair list exactly and explicitly keeps palm-to-thumb-fingertip
  contact enabled.

### Source Boundary and Rationale

- The original LEAP URDF and task YAML do not explicitly contain these 16
  exclusions. The source Isaac Gym task enables self-collision and delegates
  asset filtering to the imported actor, but the exact importer-generated pair
  filters cannot be proven from the supplied source files alone.
- This is therefore an explicit MuJoCo compatibility adaptation. The restored
  source STL collision meshes overlap at several joint interfaces, so allowing
  direct parent-child mesh contact creates persistent internal contact forces
  at mechanically connected joints.
- The exclusions do not include the LEAP ball or cube, non-adjacent finger
  links, thumb fingertip-to-palm, or thumb DIP-to-palm contacts. Those contacts
  remain available to block physically invalid hand and object penetration.

### Files

- `src/unilab/assets/robots/leap_hand/leap_hand.xml`
- `tests/envs/test_leap_inhand.py`
- `DEVELOPMENT_LOG.md`

### Validation Status

- Ruff passed for the changed LEAP asset test. The targeted LEAP suite passed:
  18 tests in 1.27 seconds, including all three scene compilations, the exact
  exclusion-set assertion, MuJoCo reset/step, and Motrix reset/step.
- A 16-environment, one-iteration MuJoCo ball-rotation smoke completed at
  486 steps/s in run `2026-07-15_17-58-07_mujoco`. Its work reward was
  `-210.6169`, so the old cube cache is not physically acceptable with the
  restored collision model.
- A matching Motrix smoke completed at 572 steps/s in run
  `2026-07-15_18-01-54_motrix`. Its work reward was `-503.6713`, confirming
  that the cache problem is not isolated to the MuJoCo training path.
- Forward contact inspection of all 1,024 old cache states found zero contacts
  from excluded pairs. Hand self-contacts dropped from the earlier 9,440 to
  3,526, and maximum self-penetration dropped from approximately 30.6 mm to
  18.4 mm. The remaining contacts are primarily non-adjacent MCP-to-DIP and
  palm-to-thumb-PIP pairs and must not be hidden by broadening the exclusions.
- Old-cache object-to-hand maximum penetration was approximately 2.74 mm.
- An isolated 64-environment, 10-iteration MuJoCo ball-cache generator smoke
  collected 12 rows at `C:\tmp\leap_ball_grasp_diag.npy`; strict valid ratios
  reached approximately 9.6 to 17.0 percent after warmup. This confirms that
  fingertip contact sensing and cache collection now function.
- The isolated cache reduced maximum self-penetration to approximately 4.26 mm
  but still contained object-to-hand penetration up to approximately 5.20 mm.
  The current backend-neutral generator has no penetration-depth acceptance
  gate, so the isolated cache is diagnostic only and is not connected to the
  ball-rotation owner.

### Training Decision

- Formal cache generation and the next 1,000-iteration ball training remain
  blocked until penetration depth is included in cache acceptance or an
  equivalent validated cold-path post-filter is added. Native MuJoCo viewer
  inspection is also still required before physical acceptance.

## 2026-07-15 - MuJoCo Ball-Cache Penetration Acceptance Filter

### Scope

- Added a low-frequency `SimBackend` contact-penetration query contract and a
  MuJoCo implementation that reconstructs only selected candidate environment
  states and reports maximum hand self-penetration and object-to-hand
  penetration.
- Added a LEAP ball-cache acceptance filter immediately before successful
  candidates are written. Both maximum depths must be at most 1 mm; non-finite
  depths are rejected.
- Kept the penetration query out of the environment step and reward paths.
  It runs only for otherwise successful cache candidates at episode reset.
- Added rejection counts and maximum-depth diagnostics to grasp-generation
  logs.
- Explicitly disabled the filter in the Motrix owner because Motrix does not
  currently expose contact depth through the backend contract. Motrix output
  remains separate and must not be treated as penetration-validated.

### Files

- `src/unilab/base/backend/base.py`
- `src/unilab/base/backend/mujoco/backend.py`
- `src/unilab/envs/manipulation/leap_inhand/ball_grasp_gen.py`
- `conf/ppo/task/leap_inhand_ball_grasp/mujoco.yaml`
- `conf/ppo/task/leap_inhand_ball_grasp/motrix.yaml`
- `tests/base/test_sim_backend_smoke.py`
- `tests/envs/test_leap_inhand.py`
- `DEVELOPMENT_LOG.md`

### Validation Status

- Ruff passed for the backend contract, MuJoCo implementation, LEAP generator,
  and targeted tests.
- Backend and LEAP targeted tests passed: 39 tests in 4.53 seconds. Coverage
  includes non-negative MuJoCo penetration-depth results for selected LEAP
  cache rows and exact threshold behavior for self/object penetration.
- An isolated MuJoCo generator smoke used 64 environments, 50 iterations, a
  16-state target, and temporary output
  `C:\tmp\leap_ball_grasp_filtered_diag.npy`. Run id:
  `2026-07-15_18-19-50_mujoco`.
- The filter evaluated 41 candidates that had already passed the existing
  stability and contact gates. All 41 were rejected by the 1 mm penetration
  limits. Representative logged maxima were approximately 3.3 to 4.1 mm for
  hand self-penetration and 1.6 to 5.2 mm for object-to-hand penetration.
- Cache size remained zero and the temporary output file was not created. This
  confirms the filter fails closed and does not persist physically invalid
  candidates.

### Remaining Blocker

- The current proposal distribution starts from the old cube cache with only
  small joint and ball-position perturbations. It does not explore far enough
  from those penetrating poses to produce a candidate within the 1 mm limits.
- Do not weaken or bypass the penetration gate to force cache output. Formal
  ball-cache generation and 1,000-iteration ball training remain blocked until
  the proposal-generation method is redesigned and separately approved.

## 2026-07-15 - Canonical LEAP Ball-Grasp Proposal Generation

### Scope

- Removed the LEAP cube grasp cache as the proposal source for the independent
  ball-cache generator.
- Changed proposal generation to start from the task-owned `home` keyframe
  already materialized from `scene_ball.xml`. That keyframe contains the
  UniLab-adapted canonical pose and object start from the upstream LEAP
  `LeapHandRot.yaml` configuration.
- Retained configurable joint and ball-position perturbations, joint-limit
  clipping, stability/contact quality gates, duplicate suppression, and the
  MuJoCo-only 1 mm self/object penetration acceptance limits.
- Kept the ball-rotation and toss tasks unchanged. The generated cache remains
  isolated and is not connected to ball-rotation training before validation.

### Files

- `src/unilab/envs/manipulation/leap_inhand/ball_grasp_gen.py`
- `conf/ppo/task/leap_inhand_ball_grasp/mujoco.yaml`
- `tests/envs/test_leap_inhand.py`
- `DEVELOPMENT_LOG.md`

### Validation Status

- Ruff passed for the canonical proposal generator and LEAP targeted tests.
- LEAP targeted tests passed: 20 tests in 1.43 seconds. The added regression
  confirms proposals are built from the supplied canonical state, apply only
  explicit joint/object offsets, and clip joints to the configured limits.
- An isolated MuJoCo diagnostic used 64 environments, 50 iterations, a
  16-state target, and temporary output
  `C:\tmp\leap_ball_grasp_canonical_diag.npy`. Run id:
  `2026-07-15_21-42-13_mujoco`.
- The diagnostic did not create a cache. Fingertip proximity remained high
  (generally 0.94-1.00) and thumb contact reached approximately 0.09-0.27,
  but the at-least-two-contact condition remained 0.0. No candidate reached
  the later penetration acceptance filter.

### Remaining Blocker

- Independent symmetric noise around the canonical keyframe does not close a
  second finger onto the 40 mm ball. The next proposal revision must add a
  physically motivated finger-closing search or another approved structured
  proposal distribution; increasing formal generation length alone will not
  address the observed zero two-finger-contact rate.

## 2026-07-15 - Structured Ball-Grasp Proposal Diagnostics

### Scope

- Evaluated a broader canonical search before encoding a joint direction:
  joint noise `0.25 rad` and ball-position noise `15 mm`.
- Added an experimental structured proposal that scales each consecutive
  four-joint finger block by an independent factor in `[0.8, 1.2]`, then adds
  `0.03 rad` residual joint noise. This preserves each finger's canonical joint
  relationship without using the cube cache or hard-coded closing signs.
- Added configuration validation and regression coverage for the finger-block
  transformation.

### Validation

- Broad-search diagnostic run `2026-07-15_21-46-24_mujoco` used 64
  environments and 30 iterations. The two-contact ratio reached approximately
  0.002-0.037 and strict pre-penetration validity reached approximately 0.016.
  One candidate reached the penetration filter and was rejected at about
  3.3 mm self-penetration and 1.8 mm object penetration.
- Ruff passed and the expanded LEAP targeted set passed: 21 tests in 1.30
  seconds.
- Structured diagnostic run `2026-07-15_21-48-51_mujoco` used 64
  environments, 50 iterations, and a 16-state target. It produced no cache;
  the two-contact condition was almost always zero and no candidate reached
  physical acceptance.

### Conclusion

- Scaling a whole finger along the canonical joint vector is not an effective
  closing direction for this imported LEAP kinematic convention. The broad
  search is better at discovering multi-contact states, but its rare valid
  states still require local refinement to satisfy the 1 mm penetration gate.
- Formal 5,000-state generation remains blocked. Do not weaken the physical
  acceptance limits or connect either diagnostic output to ball rotation.

## 2026-07-15 - Adaptive Frontier Ball-Grasp Proposals

### Scope

- Removed the unsuccessful per-finger canonical-vector scaling experiment and
  restored the broader canonical proposal distribution.
- Added an in-memory adaptive frontier containing the lowest-penetration
  candidates that already pass the stability and contact gates but fail the
  strict physical acceptance gate.
- Each reset now mixes fresh canonical proposals with small local joint and
  ball-position perturbations around frontier states. The frontier is bounded,
  normalized, deduplicated, and ranked by the larger normalized self/object
  penetration depth.
- Kept the 1 mm self-penetration and object-penetration acceptance limits
  unchanged. Frontier candidates are never written to the cache unless they
  independently pass both limits.
- Kept all diagnostic output isolated under `C:\tmp`; no generated cache was
  connected to ball rotation or toss training.

### Files

- `src/unilab/envs/manipulation/leap_inhand/ball_grasp_gen.py`
- `conf/ppo/task/leap_inhand_ball_grasp/mujoco.yaml`
- `tests/envs/test_leap_inhand.py`
- `DEVELOPMENT_LOG.md`

### Validation

- Ruff passed for the modified generator, configuration, and tests.
- LEAP targeted tests passed: 21 tests in 1.30 seconds. Regression coverage
  checks frontier normalization, deduplication, capacity, and score ordering.
- Diagnostic run `2026-07-15_21-55-19_mujoco` used 64 environments, a
  16-state target, and an intended 200 iterations. Its captured output showed
  the frontier beginning to improve contact discovery, but this run did not
  produce an isolated cache.
- Completed diagnostic run `2026-07-15_21-56-38_mujoco` used 64 environments,
  80 iterations, and a 16-state target. `run_summary.json` reports status
  `completed`, 79 completed iterations, and checkpoint `model_79.pt`.
- Adaptive proposals raised the observed two-contact and strict pre-penetration
  candidate ratios to roughly 0.30-0.46. Object penetration frequently reached
  approximately 0.5-1.0 mm, while the best observed combined frontier score
  improved from about 3.29 to 2.15.
- No temporary cache file was created. The remaining best self-penetration was
  approximately 2.15 mm, above the unchanged 1 mm acceptance limit, so the
  generator correctly rejected every candidate.

### Remaining Blocker

- The adaptive frontier solves the earlier contact-exploration scarcity but
  plateaus above the self-penetration limit. More blind local perturbations are
  unlikely to identify whether the floor comes from one persistent collision
  pair, imported collision geometry, or an inadequate correction direction.
- Before changing proposal search or collision policy, identify the exact body
  pair responsible for the maximum self-penetration on the best frontier state.
  This requires a separately approved diagnostic extension. Formal 5,000-state
  cache generation and ball-rotation cache integration remain blocked.

## 2026-07-15 - Penetration Contact-Pair Diagnostics

### Scope

- Added a typed `ContactPenetrationDetail` backend record containing the maximum
  self/object penetration depth, body pair, and geom pair for one environment.
- Added the detail query to the `SimBackend` contract and implemented it in the
  MuJoCo backend. The existing depth-only API now derives its arrays from the
  same detail query, preserving its public return shape and values.
- Updated the LEAP ball-grasp generator to use the low-frequency detail query
  and print the responsible body/geom pairs whenever the global best frontier
  score improves.
- Kept all physical acceptance limits, contact classification, reward, task
  configuration, and formal cache paths unchanged.

### Files

- `src/unilab/base/backend/base.py`
- `src/unilab/base/backend/mujoco/backend.py`
- `src/unilab/envs/manipulation/leap_inhand/ball_grasp_gen.py`
- `tests/base/test_sim_backend_smoke.py`
- `DEVELOPMENT_LOG.md`

### Validation

- Ruff passed for all modified backend, generator, and targeted test files.
- MuJoCo penetration detail smoke passed: 1 test. It verifies environment ids,
  non-negative values, contact names when depth is positive, and equality with
  the existing depth-only API.
- LEAP targeted tests passed: 21 tests in 1.70 seconds.
- Isolated diagnostic run `2026-07-15_22-04-45_mujoco` completed 80 iterations
  with 64 environments and a 16-state target. The final checkpoint was
  `model_79.pt`; no temporary cache was created.
- Early improving self-penetration contacts identified same-finger, non-adjacent
  collision meshes: index `mcp_joint` / `index_mcp_col` against `dip` /
  `index_dip_col` at 3.293 mm, followed by ring `mcp_joint_3` /
  `ring_mcp_col` against `dip_3` / `ring_dip_col` at 2.980 mm.
- The run again reached a best frontier score of approximately 2.1455, but the
  corresponding stdout detail line fell inside the command runner's truncated
  output. A filtered deterministic diagnostic is required to capture every
  improvement line before choosing a geometry or proposal correction.
- Filtered run `2026-07-15_22-07-43_mujoco` terminated early because the
  Windows `findstr` phrase was split incorrectly by the command runner; it is
  not a valid diagnostic result.
- Filtered deterministic run `2026-07-15_22-08-07_mujoco` completed 45
  iterations and reproduced every improving frontier contact. The final best
  score was 2.1455: 2.145 mm self-penetration between ring bodies
  `mcp_joint_3` and `dip_3`, geoms `ring_mcp_col` and `ring_dip_col`, with only
  0.124 mm object penetration between `leap_object_col` and `thumb_tip_col`.
- XML inspection confirms `mcp_joint_3 -> pip_3 -> dip_3`; the penetrating
  links are non-adjacent and therefore are intentionally not covered by the
  direct parent-child contact excludes. Adding an exclude for this pair would
  permit physically invalid ring-finger folding and is not an acceptable fix.
- The next proposal correction should diagnose the associated ring joint
  values and steer candidate generation away from this self-collision while
  preserving thumb/ball contact. No formal cache was created or connected.

### Best-Candidate Joint Diagnostic

- Extended the existing frontier-improvement stdout diagnostic to include the
  candidate's 16 hand joint positions on the same line. This is diagnostic
  output only and does not alter proposal generation or acceptance behavior.
- Ruff passed and the LEAP targeted suite passed: 21 tests in 1.65 seconds.
- Filtered deterministic run `2026-07-15_22-11-17_mujoco` completed 45
  iterations, reproduced the 2.1455 best score, and created no temporary cache.
- The best candidate's ring block (array indices 8-11, actuator/joint order
  `9, 8, 10, 11`) was `[0.001562, 0.559920, 1.851274, 0.018000]`, versus the
  task-owned canonical values `[-0.108400, 0.965200, 1.731700, 0.107100]`.
- Ring joint `10` was only approximately 0.0337 rad below its 1.885 rad upper
  limit (about 98.6% through its configured range). Across improving candidates,
  joint `8` decreased from roughly 0.74 to 0.56 rad while joint `10` remained
  near 1.85 rad. This supports testing joints `8` and `10` as the local
  self-collision correction coordinates; it does not yet establish a safe
  hard-coded limit.

### Joint-Coordinate Finite-Difference Scan

- Added `build_joint_coordinate_probes` to the LEAP ball-grasp owner module. It
  produces a baseline plus signed coordinate probes and clips requested values
  to the robot's existing joint limits.
- Added `scripts/diagnose_leap_ball_grasp.py` as a cold-path assembly tool. It
  accepts a complete 23-value LEAP/ball qpos, resolves requested MuJoCo joint
  names through the backend contract, and reports maximum self/object contact
  details for every probe.
- Extended frontier diagnostics from 16 hand values to the complete 23-value
  state so the scanner preserves the exact ball pose and thumb contact.
- Ruff passed, the LEAP targeted suite passed with 22 tests in 1.87 seconds,
  and the diagnostic CLI help smoke passed.
- Full-state reproduction run `2026-07-15_22-15-56_mujoco` completed 45
  iterations, reproduced the 2.1455 frontier state, and created no cache.
- The joint `8`/`10` scan used signed 0.02, 0.05, and 0.10 rad probes. Moving
  joint `8` negative reduced ring MCP/DIP penetration from 2.145 mm to the
  point where index MCP/DIP became the maximum at 2.030 mm. Positive joint `8`
  perturbations worsened ring penetration up to 2.569 mm.
- Increasing joint `10` reduced ring penetration only slightly, reaching
  2.120 mm at its existing 1.885 rad upper limit; decreasing it worsened the
  result to approximately 2.166 mm. This disproves the earlier hypothesis that
  simply opening joint `10` would resolve the collision.
- Object/thumb penetration remained exactly 0.124 mm for all probes because
  these internal ring coordinates do not move the thumb. The next maximum
  reveals a second independent blocker: `index_mcp_col` against
  `index_dip_col`, approximately 2.030 mm. Index joints `0` and `2` require the
  same coordinate scan before designing a combined proposal correction.
- The first index joint `0`/`2` scan retained the original ring pose, so the
  2.145 mm ring collision masked index depths below that value. Joint `0`
  negative probes nevertheless made index MCP/DIP become the maximum at
  2.244 mm (`-0.05 rad`) and 2.487 mm (`-0.10 rad`), establishing that the
  negative direction is harmful. Positive joint `0` probes and all tested
  joint `2` probes remained below the ring maximum, so their exact index depth
  could not be measured from the maximum-only report.
- A conditioned scan must first apply the known ring joint `8` correction
  (`0.559920 -> 0.509920`) and then probe index joints `0` and `2`. This is
  required before selecting a combined correction direction.
- The conditioned index scan fixed ring joint `8` at 0.509920, making the
  2.030 mm index MCP/DIP collision visible. Index joint `0` positive probes
  reduced the maximum to 1.942 mm at `+0.02 rad`; at `+0.05 rad` the index
  collision fell below the ring collision, which became the maximum at
  1.922 mm. `+0.10 rad` produced the same ring-limited maximum, so `+0.05 rad`
  is sufficient for the current local correction without adding extra motion.
- Index joint `2` was much less effective: its best tested value was
  `+0.02 rad`, producing 2.013 mm, while larger positive and all negative
  probes were worse. Object/thumb penetration stayed 0.124 mm for all probes.
- The next conditioned scan should hold index joint `0` at its `+0.05 rad`
  correction (`-0.538329 -> -0.488329`) and continue probing ring joint `8`
  in the negative direction from 0.509920.
- With index joint `0` fixed at -0.488329, the second conditioned ring scan
  reduced the 1.922 mm baseline to 1.831 mm at ring joint `8 -0.02 rad`.
  At `-0.05 rad` (`0.509920 -> 0.459920`), index MCP/DIP became the maximum at
  1.808 mm. A larger `-0.10 rad` ring adjustment produced the same index-limited
  maximum, so the smaller `-0.05 rad` change is sufficient for this round.
- Object/thumb penetration again stayed 0.124 mm. The alternating maximum now
  returns to index MCP/DIP, so the next conditioned probe must hold ring joint
  `8` at 0.459920 and continue index joint `0` in the positive direction.

### Four-Round Alternating Coordinate Scan

- Ran the approved maximum of four additional conditioned scans, alternating
  index joint `0` positive corrections with ring joint `8` negative corrections.
- Round 1 changed index joint `0` from -0.488329 to -0.438329 and reduced the
  maximum from 1.808 mm to a ring-limited 1.692 mm.
- Round 2 changed ring joint `8` from 0.459920 to 0.409920 and reduced the
  maximum to an index-limited 1.580 mm.
- Round 3 changed index joint `0` from -0.438329 to -0.388329 and reduced the
  maximum to a ring-limited 1.455 mm.
- Round 4 changed ring joint `8` from 0.409920 to 0.359920 and reduced the
  maximum to an index-limited 1.344 mm. In every round, a 0.10 rad move did not
  improve beyond the 0.05 rad move because the other finger became limiting;
  the smaller sufficient correction was retained.
- Across the four rounds the maximum decreased monotonically from 1.808 mm to
  1.344 mm, but did not yet satisfy the 1 mm acceptance limit. Object/thumb
  penetration remained 0.124 mm in every static probe.
- These static contact scans do not prove that the corrected state still meets
  the generator's two-contact, thumb-contact, drift, velocity, work, and
  settling gates. Any candidate that reaches the penetration threshold must be
  revalidated through the full grasp-quality path before cache insertion.

### Second Alternating Coordinate Scan

- Ran a second approved set of up to four alternating scans, stopping as soon
  as the maximum self-penetration became less than 1 mm.
- Round 1 changed index joint `0` from -0.388329 to -0.338329 and reduced the
  maximum from 1.344 mm to a ring-limited 1.212 mm.
- Round 2 changed ring joint `8` from 0.359920 to 0.309920 and reduced the
  maximum to an index-limited 1.102 mm.
- Round 3 changed index joint `0` from -0.338329 to -0.288329 and reduced the
  maximum to a ring-limited 0.963393 mm, satisfying the strict 1 mm static
  self-penetration threshold. The fourth round was not run.
- A larger `+0.10 rad` index move did not improve beyond the selected
  `+0.05 rad` move. Object/thumb penetration remained 0.124192 mm and therefore
  also satisfied the 1 mm static object threshold.
- Relative to the original best frontier state, the final static correction is
  index joint `0: -0.538329 -> -0.288329` and ring joint
  `8: 0.559920 -> 0.309920`; all other hand and ball qpos values are unchanged.
- This state is penetration-valid only. It has not been written to cache and
  still requires the full settling/contact/stability/work quality validation.

### Full Settling and Quality Validation

- Added env-owned `diagnose_grasp_state`, which injects one complete 23-value
  candidate, disables autoreset, holds the candidate hand pose through the
  production PD/action path, settles for an explicit duration, and evaluates
  the existing LEAP grasp quality conditions plus penetration details.
- Extended `scripts/diagnose_leap_ball_grasp.py` with `--settle-seconds`. This
  mode explicitly supplies the grasp task's zero reward configuration, disables
  cache autosave/collection, and prints a structured JSON report.
- Added integration coverage for the diagnostic lifecycle and output contract.
  Ruff passed and the LEAP targeted suite passed with 23 tests in 1.89 seconds.
- Ran the corrected state through 0.5 seconds (10 control steps) of production
  PD settling. Self penetration remained valid at 0.942610 mm between
  `ring_mcp_col` and `ring_dip_col`; object penetration remained valid at
  0.415288 mm between `leap_object_col` and `thumb_dip_col`.
- Finite values, joint limits, object height, 0.226 mm ball drift, fingertip
  proximity, object linear/angular speed, joint speed, and work all passed.
- All four fingertip contact sensors were false after settling. Contact count
  was zero, including no thumb-tip contact, so the existing `contacts` and
  `thumb` gates failed and overall `quality_valid` was false.
- The corrected state was not written to cache. Penetration-only coordinate
  correction opened the index/ring kinematic chains enough to remove the
  contacts required for a valid grasp; further work must restore fingertip
  contact without undoing the self-penetration correction.

### Fingertip-to-Ball Signed Distance Diagnostics

- Added the low-frequency `SimBackend.get_geom_pair_distances` contract and a
  MuJoCo implementation using `mj_geomDistance`. It evaluates explicit geom
  pairs against selected environment physics states and returns signed surface
  distances; negative values indicate overlap.
- Extended settling diagnostics to report initial and settled contact flags and
  signed distances for index, middle, ring, and thumb tip collision geoms
  against `leap_object_col`.
- Ruff passed. The new MuJoCo geom-distance smoke passed, and the complete LEAP
  targeted suite passed with 23 tests in 2.02 seconds.
- Repeating the 0.5-second corrected-state diagnostic showed that initial
  middle and thumb contacts were both true. Their initial signed distances were
  -0.466 mm and -0.124 mm, respectively, satisfying the two-contact and thumb
  requirements at injection time.
- After settling, middle distance was +0.000384 mm (effectively touching but no
  longer overlapping) and thumb distance was +0.198 mm; both contact sensors
  became false. Index and ring remained far from the ball at +33.317 mm and
  +10.693 mm and were not the intended contact sources for this state.
- The evidence narrows contact restoration to middle and thumb distal/tip
  coordinates. Index/ring closure is unnecessary and would risk reintroducing
  the MCP/DIP self-collisions already corrected. No cache was written.

### Settling-Aware Distal Joint Probes

- Added env-owned `diagnose_joint_coordinate_probes`, which builds signed joint
  probes and independently runs every candidate through the full settling,
  contact, quality, and penetration diagnostic. Added the CLI
  `--settle-probes` switch and integration coverage for repeated lifecycle use.
- Ruff passed and the LEAP targeted suite passed with 23 tests in 2.04 seconds.
- Scanned middle joint `7` and thumb joint `15` at signed 0.005, 0.01, and
  0.02 rad deltas, with 0.5 seconds of independent settling per probe.
- Middle joint `7` positive was effective. The smallest `+0.005 rad` probe
  changed settled middle signed distance from +0.000384 mm to -0.0997 mm and
  restored `leap_middle_contact`; `+0.01` and `+0.02 rad` added unnecessary
  overlap. The selected middle correction is therefore `+0.005 rad`.
- Thumb joint `15` was ineffective in this local geometry. Its best tested
  `+0.02 rad` probe reduced the settled thumb gap only from 0.1975 mm to
  0.1896 mm, and no joint `15` probe restored thumb contact.
- All distal probes retained valid self/object penetration. The next thumb
  diagnostic should hold the selected middle joint `7 +0.005 rad` correction
  and scan upstream thumb joints `13` and `14`, which move the distal thumb
  chain rather than rotating only the fingertip link. No cache was written.

### Settling-Aware Upstream Thumb Probes

- Held the selected middle joint `7 +0.005 rad` correction and scanned thumb
  joints `13` and `14` at signed 0.005, 0.01, and 0.02 rad deltas, with each
  candidate independently settled for 0.5 seconds through the production
  control path.
- Thumb joint `13 -0.020 rad` was the first candidate to pass the complete
  quality path. It changed joint `13` from `1.326942` to `1.306942`, retained
  the restored middle-tip contact, and restored thumb-tip contact after
  settling. The final signed middle and thumb surface distances were
  -0.086403 mm and -0.132827 mm, respectively.
- The accepted probe had two fingertip contacts, including the required thumb
  contact. Self penetration was 0.942610 mm between `ring_mcp_col` and
  `ring_dip_col`; object penetration was 0.132827 mm between
  `leap_object_col` and the thumb tip. Both remained below the strict 1 mm
  limits.
- Ball drift was 0.305379 mm, and all finite-state, joint-limit, height,
  proximity, linear/angular speed, joint-speed, work, contact, thumb-contact,
  and penetration conditions passed. The diagnostic therefore reported
  `quality_valid=true`.
- No tested joint `14` probe restored thumb contact. This confirms joint `13`
  as the effective local thumb coordinate for this canonical candidate.
- The resulting corrected 16-joint hand pose combines index joint `0 +0.25
  rad`, middle joint `7 +0.005 rad`, ring joint `8 -0.25 rad`, and thumb joint
  `13 -0.020 rad` relative to the original frontier state. This is one valid
  deterministic 0.5-second candidate only; no cache file was created or
  connected to training.

### Independent Ball-Seed Proposal Configuration

- Extended the same corrected state to 1.5 seconds (30 production control
  steps). It retained middle and thumb contacts, two total contacts, 0.928111
  mm maximum self penetration, 0.132172 mm object penetration, and reported
  `quality_valid=true` without terminating during settling.
- Probed index joint `0`, middle joint `7`, ring joint `8`, and thumb joint `13`
  at signed 0.002 and 0.005 rad offsets with 1.5 seconds of settling per state.
  Fifteen of sixteen candidates passed. Only middle joint `7 -0.005 rad`
  failed, because middle contact opened and left only the thumb contact.
- Added a config-owned 23-value ball-grasp seed and per-joint asymmetric offset
  bounds to `LeapInhandBallGraspCfg`. The provider falls back to the task
  canonical keyframe when no seed is configured and never reads cube or toss
  cache data.
- Configured the MuJoCo owner to vary only the four validated coordinates:
  joints `0`, `8`, and `13` use +/-0.005 rad; joint `7` uses -0.002 to +0.005
  rad. Other joint offsets and ball-position offsets are zero for this first
  isolated cache, and broad frontier refinement is disabled.
- Added targeted tests for independent seed selection, quaternion
  normalization, asymmetric bounded sampling, zero offsets on unvalidated
  joints, incomplete-bound rejection, and the ball-owned cache path.
- No existing cache, `scene_ball.xml`, toss task, or ball-rotation task was
  modified. The formal 5,000-state cache remains uncreated and disconnected.

#### Validation

- Ruff passed for the modified generator and LEAP targeted tests. The complete
  LEAP targeted file passed with 26 tests in 2.25 seconds.
- The first approved isolated run reached the save path with every logged
  grasp-quality condition at 1.0, but Windows denied writes to
  `C:\tmp\leap_ball_grasp_seed_diag_16.npy`. It exited with `PermissionError`;
  no file was created there and this was not a proposal or physics failure.
- The approved retry wrote only to
  `D:\UniLab\logs\ball_grasp_diagnostics\leap_ball_grasp_seed_diag_16.npy`.
  Run `2026-07-15_22-53-27_mujoco` used 64 environments, a 16-state target,
  and a 20-iteration safety limit. At the first collection boundary it found
  64 unique candidates that passed the complete quality and 1 mm penetration
  path, saved the first 16, and stopped normally after iteration 2.
- The isolated output has shape `(16, 23)`, dtype `float32`, 16 unique finite
  rows, and unit quaternion norm for every row. It remains under `logs` and is
  not referenced by any training owner configuration.

## 2026-07-15 - Formal LEAP Ball-Grasp Cache Integration

### Scope

- Generated `robots/leap_hand/caches/ball_grasp_s10_5k.npy` from the validated
  independent LEAP ball seed and asymmetric four-joint proposal ranges. No
  cube or toss cache was used as proposal input.
- Changed `LeapInhandBallRotationCfg` and both PPO backend owner files to use
  the formal LEAP ball cache. The Motrix owner declares the path explicitly
  instead of relying only on inheritance from MuJoCo.
- Added regression coverage for cache shape, dtype, finite values, row
  uniqueness, normalized quaternions, Python default ownership, and explicit
  MuJoCo/Motrix owner paths.
- The cube cache remains present for LEAP cube rotation and toss tasks; it was
  not modified or deleted.

### Generation Evidence

- Formal run `2026-07-15_22-59-32_mujoco` used 1,024 environments, a 5,000
  state target, and a 100-iteration safety limit. It reached 5,327 accepted
  unique candidates after iteration 25, saved the first 5,000, and stopped
  normally.
- Logged grasp-quality conditions remained 1.0. The reported penetration batch
  had approximately 0.9 mm maximum self penetration, 0.2 mm maximum object
  penetration, 1.0 penetration-valid ratio, and zero penetration rejections.
- The saved cache has shape `(5000, 23)`, dtype `float32`, 5,000 unique finite
  rows, and quaternion norms exactly equal to 1.0 in the metadata check.

### Integration Validation

- Ruff passed for the changed ball-rotation owner module and LEAP tests.
- The complete LEAP targeted file passed with 27 tests in 1.97 seconds and no
  skips. This includes actual MuJoCo and Motrix reset/step coverage after the
  Python default and both backend owner configs were switched to the new ball
  cache.
- No formal ball-rotation RL training was executed as part of this integration.

## 2026-07-15 - LEAP Ball Rotation Viewer Telemetry

### Scope

- Added config-driven object rotation telemetry to the official MuJoCo
  interactive viewer. The overlay reports signed target-axis angular speed,
  total angular speed, signed RPM, a one-second signed-speed mean, and
  cumulative viewer-session turns.
- The viewer reads world-frame angular velocity through MuJoCo
  `mj_objectVelocity`; it does not change environment observations, rewards,
  checkpoint compatibility, simulation state, or training behavior. Paused
  frames update the instantaneous display without accumulating turns or the
  moving average.
- Enabled the overlay for `leap_inhand_ball` with body `leap_object`; the
  target axis comes from the task's existing `rotation_axis` config.
- Added a short white capsule marker on the LEAP ball's local +X surface. The
  marker is attached to the ball body but has `contype=0` and `conaffinity=0`,
  so it is visual-only. The existing sphere collision geom, explicit mass,
  inertia, friction, contact sensors, and reward are unchanged.
- Added regression coverage for signed telemetry calculations, RPM, paused
  accumulation behavior, overlay formatting, marker geom type/body ownership,
  disabled collision flags, and unchanged object mass/inertia. Additional
  integration coverage composes the LEAP ball owner and reads a known angular
  velocity from the real MuJoCo `scene_ball.xml` body state.

### Validation

- Ruff passed for the interactive viewer, viewer tests, and LEAP asset tests.
- The combined viewer and LEAP targeted suite passed with 43 tests in 6.29
  seconds. The real-scene velocity test injected `2 rad/s` around world Z and
  the viewer helper returned `[0, 0, 2]`; owner composition enabled telemetry
  for body `leap_object`.
- Asset compilation confirmed the marker is a capsule on the object body with
  both collision masks disabled, while object mass remained 0.05 kg and
  diagonal inertia remained `[0.0001, 0.0001, 0.0001]`.

## 2026-07-16 - LEAP Ball Rotation V2 Implementation

### Scope

- Added the independent `LeapInhandBallRotationV2` environment and PPO owner
  configs for MuJoCo and Motrix. The existing `LeapInhandBallRotation`
  implementation, `leap_inhand_ball` owner configs, ball asset, formal ball
  cache, toss task, and existing checkpoints were not modified.
- V2 observes joint velocity, previous action, palm-relative ball pose and
  velocity, target axis/speed, fingertip offsets, and contacts. Its policy
  observation is 99 dimensions and intentionally incompatible with V1
  checkpoints while leaving the V1 105-dimensional contract unchanged.
- Replaced the V1 stationary local optimum with signed target-axis progress,
  direction-aware orthogonal-spin quality, ball retention, small level-gated
  energy regularization, sustained-spin bonuses, quarter-turn milestones, and
  an explicit drop/workspace failure penalty.
- Added an episode-local performance curriculum with target speeds 0.10, 0.25,
  and 0.50 rad/s. An environment advances only after maintaining at least 60%
  of its current target for five control steps with bounded orthogonal speed.
- Added TensorBoard diagnostics for target-axis and orthogonal speed, positive
  progress, a one-second speed mean, completed turns, sustained fraction,
  quarter-turn events, drop rate, ball position error, curriculum level, and
  active target speed. The configured axis remains world-frame so reward,
  diagnostics, and the existing MuJoCo viewer overlay share one convention;
  its palm-frame representation is included in the policy observation.
- Added focused reward semantics, MuJoCo reset/step, and support-matrix tests,
  plus V2 TensorBoard instructions in `USEFUL_COMMANDS.md`.

### Validation

- Ruff passed for the V2 environment, LEAP package registration, LEAP targeted
  tests, and support-matrix tests after removing one unused V2 local variable.
- The targeted LEAP, support-matrix, and viewer suites passed with 51 tests in
  15.51 seconds. Coverage includes V2 registration, directed-spin reward
  semantics, 99-dimensional finite observations, and real MuJoCo/Motrix
  reset-step execution.
- The approved MuJoCo smoke run `2026-07-16_01-40-12_mujoco` completed one PPO
  iteration with 16 environments and 99-input actor/critic networks. All V2
  metrics were finite, drop rate was zero, and the untrained-policy batch
  reported approximately 0.0089 rad/s mean target-axis speed.
- The approved Motrix smoke run `2026-07-16_01-40-31_motrix` completed the same
  one-iteration, 16-environment contract with 99-input networks, finite V2
  metrics, and zero drop rate. These smoke values establish runtime viability,
  not learned rotation or cross-backend performance equivalence.
## 2026-07-17 - LEAP Ball Canonical-Pose Rotation Authority Diagnostic

### Scope

- Added `scripts/diagnose_leap_ball_rotation_authority.py` as an isolated
  MuJoCo diagnostic for canonical ball-grasp candidates. It does not modify
  the existing rotation task, reward, cache, scene keyframe, or candidate.
- The diagnostic loads the candidate's explicit qpos/control coordinate
  contract, settles 33 parallel environments from the same state, and applies
  one baseline plus signed `0.04 rad` position-target pulses to each of the 16
  LEAP joints.
- It samples world-Z angular speed and integrated rotation, orthogonal angular
  speed, ball displacement and height, fingertip contact retention, and
  self/object penetration. Positive and negative rotation authority require
  measurable directed motion while retaining the existing physical safety
  limits.
- Added focused tests for candidate validation, quaternion normalization,
  signed probe construction, joint-limit clipping, and authority
  classification. The summary ranks only physically safe probes; unsafe
  motions remain visible in the full report but cannot be labeled as the best
  positive or negative control direction.

### Validation

- Ruff passed for the diagnostic script and its focused test file. The
  targeted suite passed with 6 tests in 0.22 seconds after redirecting pytest
  temporary files into the project `logs` directory to avoid the host Windows
  Temp permission failure.
- Before the authority sweep, the same candidate passed an extended 3.0-second
  production-path settling check. It retained index, middle, and thumb
  contacts, drifted 0.352 mm, reached only 0.389 mm self penetration and
  0.102 mm object penetration, and did not terminate.
- The deterministic authority sweep used a 1.0-second settle and 0.25-second
  signed pulses sampled every 0.05 seconds. Twenty of 32 signed probes retained
  at least two contacts including the thumb, stayed within 5 mm of the anchor,
  and remained below both 1 mm penetration limits.
- LEAP thumb-base joint `12` provided safe authority in both world-Z
  directions. A `-0.04 rad` target pulse produced +0.015981 rad integrated
  rotation with +0.170904 rad/s peak speed, three retained contacts, and
  1.056 mm maximum ball displacement. A `+0.04 rad` pulse produced -0.011211
  rad integrated rotation with -0.083536 rad/s peak speed, two retained
  contacts, and 1.061 mm maximum ball displacement.
- The candidate therefore satisfies this local test's stable-grasp and
  bidirectional-rotation-authority criteria. This is local single-joint
  evidence, not proof of sustained multi-turn rotation or learned-policy
  performance.
## 2026-07-17 - LEAP Ball Fixed-Home Reset Integration

### Scope

- Promoted the validated `ball_candidate_01.json` qpos and control target into
  the task-level `scene_ball.xml` `home` keyframe. Robot XML, ball dynamics,
  collision geometry, reward, observations, controls, and checkpoints were not
  changed.
- Added a LEAP-owned `reset_source` selector with accepted values `home` and
  `cache`. The Python config retains `cache` as its compatibility default;
  both current `leap_inhand_ball` backend owners explicitly select `home`.
- `home` bypasses cache loading and therefore makes the existing reset builder
  set hand pose, ball pose, `prev_ctrl`, and `init_pose` from the scene
  keyframe. `cache` retains the existing
  `robots/leap_hand/caches/ball_grasp_s10_5k.npy` sampling path unchanged.
- The selector is scoped to V1 `LeapInhandBallRotation`. V2 keeps its existing
  cache provider and reset behavior; cube rotation, toss, and grasp generation
  are unaffected except that any explicit fallback to the shared ball scene
  home now sees the validated pose.
- Added regression coverage for candidate/keyframe identity, explicit owner
  selection, deterministic home reset on MuJoCo and Motrix, retained cache
  sampling, and invalid selector rejection. Validation evidence will be added
  after the approved checks and smoke runs complete.

### Validation

- Ruff passed for the LEAP ball reset owner and complete LEAP test file.
- The complete targeted LEAP suite passed with 35 tests in 4.71 seconds. This
  includes direct source-scene equality with the candidate, MuJoCo and Motrix
  fixed-home reset/step coverage, and an explicit MuJoCo cache-reset test that
  matched every sampled state back to `ball_grasp_s10_5k.npy`.
- MuJoCo's tracking-sensor XML materialization serializes some keyframe values
  to approximately six significant digits; runtime reset values remained
  within 3.94e-6 rad of the source candidate. The source `scene_ball.xml`
  compiles directly to the candidate values within 1e-7.
- The approved 16-environment, one-iteration MuJoCo PPO smoke completed as run
  `2026-07-17_18-24-52_mujoco`. It produced `model_0.pt`, used the unchanged
  105-dimensional observation and 16-dimensional action contracts, completed
  128 steps, and emitted finite reward metrics.
- The saved `run_config.json` explicitly records `reset_source: home`, proving
  the CLI/Hydra owner selected the fixed-home path rather than the retained
  cache path.

## 2026-07-18 - LEAP Allegro-Faithful Ball Rotation Task

### Scope

- Added the isolated `leap_inhand_ball_allegro` PPO task for MuJoCo and Motrix;
  the existing LEAP V1, V2, toss, grasp generation, home keyframe, and caches
  remain unchanged.
- Preserved the successful Allegro reward terms, weights, signed world-Z
  rotation axis, clipping, fixed-home reset, 20-second episode, 20 Hz control,
  and disabled reset randomization.
- Adapted Allegro's 5 mm drop margin to the LEAP home ball center, producing an
  exact world-Z termination threshold of `0.66997318983078 m`.
- Added range-equivalent pose scaling so an equal fraction of corresponding
  LEAP and Allegro joint travel receives the same raw-radian quadratic pose
  cost while retaining Allegro's `sum(error^2)` structure and `-0.3` weight.
- Matched the MuJoCo physics rate to Allegro at 200 Hz, retained LEAP's
  robot-specific `kp=3`, `kd=0.1`, and kept Motrix at its established 100 Hz
  integration rate.
- Added diagnostics for ball height, drop margin, termination rate, torque
  saturation, and Allegro-equivalent pose RMS, plus focused configuration,
  math, registry, support-matrix, and dual-backend reset/step coverage.

### Validation

- Ruff passed for the new environment module, LEAP package registration, LEAP
  targeted tests, and support-matrix tests after applying Ruff's required
  first-party/local import grouping.
- The complete targeted LEAP and support-matrix suites passed with 45 tests in
  16.67 seconds. Pytest emitted one non-functional warning because the existing
  project `.pytest_cache` directory was not writable; the approved per-run
  temporary directory under `logs` worked normally.
- Hydra composition passed for both owners. MuJoCo resolves to the exact
  `0.66997318983078 m` threshold, 200 Hz physics, 20 Hz control, fixed home,
  and 16 pose scales; Motrix resolves the same reward/reset contract with its
  established 100 Hz physics rate.
- The approved MuJoCo smoke run `2026-07-18_01-03-42_mujoco` completed one PPO
  iteration with 16 environments and the unchanged 105-observation/16-action
  contract. Diagnostics were finite, termination and torque saturation rates
  were zero, and ball height stayed above threshold with a 4.6 mm mean margin.
- The approved Motrix smoke run `2026-07-18_01-04-07_motrix` completed the same
  one-iteration contract with finite diagnostics, zero termination and torque
  saturation rates, and a 4.8 mm mean drop margin. These smoke runs establish
  runtime viability only, not learned rotation or backend performance parity.

## 2026-07-19 - LEAP Ball Rotation Drop and Palm-Termination Semantics

### Scope

- Restored `leap_inhand_ball_allegro` pose difference to the original Allegro
  raw-radian formula, `sum((dof_pos - init_pose)^2)`, and removed the prior
  LEAP/Allegro joint-range scaling configuration and diagnostics.
- Reassigned the height threshold from episode termination to a dense drop
  event. A ball-center fall of more than 7 mm from the fixed home height now
  activates `drop=-1.0` on every control step while the condition remains true.
- Added the passive `leap_palm_contact` task sensor between
  `palm_lower_collision` and `leap_object_col`. The Allegro-faithful LEAP task
  now terminates only when this palm contact is active; fingertip contacts do
  not terminate the episode.
- Added an overridable termination hook to the shared Allegro rotation env.
  Its default implementation retains the existing height-based behavior, so
  Allegro and other inherited rotation tasks are unchanged.
- Replaced the range-scaled pose diagnostic with raw pose L2 RMS and added
  separate drop-rate and palm-contact-rate diagnostics.

### Validation

- Ruff passed for the shared Allegro termination hook, LEAP task override, and
  targeted LEAP tests.
- The LEAP environment and support-matrix target set passed with 44 tests in
  41.51 seconds. Coverage includes the task-level palm sensor, the 7 mm drop
  boundary, an inactive palm sensor at the home reset, and MuJoCo/Motrix
  reset-step integration.
- Hydra composition passed for both owners. Both retain all Allegro reward
  terms, resolve `drop=-1.0` and `reset_z_threshold=0.66797318983078`, use the
  fixed home reset, and differ only in their established backend physics
  timestep.
- The MuJoCo smoke run `2026-07-19_02-41-54_mujoco` completed one PPO
  iteration with 16 environments. It reported zero drop and palm-contact
  rates, a 6.6 mm mean margin above the drop threshold, and finite raw-pose
  diagnostics.
- The Motrix smoke run `2026-07-19_02-41-53_motrix` completed the same
  one-iteration contract. It reported zero drop and palm-contact rates, a
  6.8 mm mean threshold margin, and finite raw-pose diagnostics.

## 2026-07-20 - LEAP Sustained +Z Ball Rotation Task

### Scope

- Added the independent `leap_inhand_ball_sustained` PPO task, registered as
  `LeapInhandBallSustainedRotation` for MuJoCo and Motrix. Existing LEAP ball
  V1, V2, Allegro-faithful, toss, caches, assets, and checkpoints are unchanged.
- Reused the validated fixed `scene_ball.xml` home reset and explicitly set
  `reset_source: home`; the retained ball-cache path remains selectable by
  other tasks but is not sampled by this task.
- Extended the V2 policy observation with the existing passive palm-contact
  sensor, producing a 100-value observation. The policy observes hand state,
  previous control/action, palm-relative ball pose and velocity, world +Z in
  the palm frame, fingertip geometry/contact, palm contact, and target speed.
- Added a four-stage within-episode curriculum: 1 second stable hold, 2 seconds
  at `+0.10 rad/s`, 4 seconds at `+0.25 rad/s`, and 10 seconds at
  `+0.50 rad/s`. Promotion requires an 80% target-speed EMA, bounded off-axis
  speed, at least two fingertip contacts, no palm contact, and at most 15 mm
  anchor-position error.
- Set the gate EMA time constant to 0.1 seconds. A 1-second EMA would add enough
  warm-up after each target increase to make the nominal 17-second curriculum
  impossible to finish inside the 20-second episode.
- Added dense target-axis spin quality, 15 mm retention shaping, smooth
  fingertip-support shaping, small action-rate and level-gated energy costs,
  stage bonuses, and a final 10-second success bonus. Hold stage spin reward is
  exactly zero; negative or off-axis spin cannot imitate successful +Z spin.
- Exposed the same effective coefficients under `reward.scales` for generic
  owner-config tooling while retaining typed runtime fields for term-specific
  units and event semantics; the mirror map is metadata and is not applied a
  second time.
- Unified drop, palm-contact, and 30 mm workspace failures behind the same
  three-control-step debounce, one-time `-5` event cost, and termination. This
  removes the previous incentive to select a cheaper failure mode.
- Added diagnostics for target/axis/off-axis speed and EMAs, axis purity,
  position/contact retention, each raw and debounced failure, stage progress,
  survival time, and consecutive 2/5/10-second sustained rotation.
- Added owner configs, registry/math/debounce/dual-backend tests, support-matrix
  metadata, and copy-paste commands for training, official MuJoCo viewer, and
  TensorBoard inspection.

### Validation

- Ruff passed for the new environment, package registration, LEAP tests,
  support-matrix tests, and support-matrix owner metadata.
- The targeted LEAP and support-matrix suites passed with 48 tests in 27.77
  seconds. Coverage includes +Z/off-axis reward semantics, failure debounce,
  registry discovery, fixed-home reset, 100-value observations, and finite
  MuJoCo/Motrix reset-step behavior.
- The two generic owner-compose cases passed (`2 passed`, 135 deselected),
  including the repository-wide non-empty `reward.scales` contract.
- Hydra composition passed for both owners. They resolve identical policy,
  reward, observation, fixed-home reset, world +Z, stage, and exploration
  contracts; only the allowed backend identity and integration timestep differ.
- The final MuJoCo smoke run `2026-07-20_01-50-42_mujoco` completed one PPO iteration
  with 16 environments and finite 100-observation/16-action learning metrics.
  It started at std 0.30, averaged 2.06 fingertip contacts, had 1.3 mm mean
  position error, and reported zero drop, palm, workspace, or total failures.
- The final Motrix smoke run `2026-07-20_01-50-42_motrix` completed the same contract.
  It started at std 0.30, averaged 2.09 fingertip contacts, had 1.6 mm mean
  position error, and also reported zero failures.
- Regenerated the Chinese support-matrix block after the new dual-backend owner
  was registered and tested. These smoke results prove runtime viability and
  initial support geometry only; they do not yet prove learned 10-second +Z
  rotation.

## 2026-07-20 - Sustained Rotation Stationary-Grip Reward Correction

### Baseline Evidence

- Diagnosed MuJoCo run `2026-07-20_02-00-23_mujoco`, trained for 300 iterations,
  1024 environments, and 2,457,600 environment steps.
- The policy converged to a stationary-grip local optimum: final mean episode
  length was 397.81/400 steps with zero final failure rate, 6.2 mm mean ball
  position error, 2.02 fingertip contacts, and no palm contact.
- Despite a final mean curriculum level of 0.92 and `0.0917 rad/s` mean target,
  actual world-Z speed was `-0.0019 rad/s`, its EMA was `-0.0030 rad/s`, and all
  2/5/10-second and final-success metrics remained zero throughout training.
- Final positive shaping was dominated by retention (`0.4178`) and fingertip
  support (`0.2496`), while spin quality was only `0.0041`. Mean policy std fell
  from 0.30 to 0.234, showing exploration collapsed around standing still.

### Scope

- Changed only `leap_inhand_ball_sustained` reward semantics. Observation,
  action, fixed-home reset, curriculum targets/durations, failure conditions,
  exploration settings, assets, caches, other tasks, and previous runs remain
  unchanged for this single-variable ablation.
- Replaced the multiplicative `axis_progress * axis_purity` signal with
  `axis_progress * (0.5 + 0.5 * axis_purity)`. Pure world-Z spin keeps the full
  signal, while off-axis exploration retains at least half instead of being
  numerically erased; reverse-Z spin remains negative.
- Level 0 now receives retention/support reward only while the complete hold
  gate is valid. In Levels 1-3, stationary grip receives no task reward;
  retention and fingertip support are paid only in proportion to positive
  target-axis progress.
- Increased the typed signed spin-progress coefficient from 1.0 to 1.5 and
  updated the generic `reward.scales` metadata mirror. Action-rate, level-gated
  energy costs, stage bonuses, final bonus, and unified failures are unchanged.
- Added direct reward-topology tests and per-level occupancy diagnostics, plus
  explicit axis-progress and visible-progress logging for the next ablation.

### Validation

- Ruff passed for the corrected environment and focused LEAP tests.
- The LEAP and support-matrix target set passed with 49 tests in 27.71 seconds.
  Direct numeric coverage proves zero rotating-stage reward at zero progress,
  full reward for pure +Z progress, 50% retained signal for strongly off-axis
  +Z exploration, negative reward for reverse-Z progress, and gate-only Level
  0 hold reward.
- Both generic owner-compose cases passed (`2 passed`, 135 deselected). Hydra
  composition for MuJoCo and Motrix resolves `spin_progress: 1.5`, contains no
  old `spin_quality` field, and preserves the same observation, action,
  curriculum, failure, and exploration contracts across backends.
- MuJoCo smoke run `2026-07-20_02-53-15_mujoco` completed one iteration with
  finite 100-observation/16-action metrics and std 0.30. With 56.25% of Level 0
  environments satisfying the full gate at the sampled step, hold reward was
  0.4211 rather than the old unconditional approximately 0.74; all failures
  remained zero.
- Motrix smoke run `2026-07-20_02-53-15_motrix` completed the same contract.
  Its sampled stage-valid fraction was 21.88%, hold reward was correspondingly
  0.1639, and all failure metrics were zero.
- Smoke tests establish corrected runtime reward topology, not learned
  sustained rotation. A fresh 300-iteration MuJoCo ablation is still required
  and must not resume the stationary baseline checkpoint.

## 2026-07-20 - Sustained Rotation Low-Speed Curriculum Bridge

### Training Evidence

- MuJoCo run `2026-07-20_02-54-59_mujoco` improved final world-Z speed EMA to
  `0.0259 rad/s` after the stationary-grip reward correction, but no environment
  reached curriculum Level 2 or sustained two seconds of valid rotation.
- Continuation run `2026-07-20_11-22-28_mujoco` reached a transient peak world-Z
  speed EMA of `0.0888 rad/s`, reduced final orthogonal-speed EMA to
  `0.0300 rad/s`, reduced final position error to `13.6 mm`, and eliminated
  final palm/drop failures. However, its final 50-iteration world-Z EMA mean
  plateaued at `0.0277 rad/s`; Level 2 occupancy and all sustained-success
  metrics remained zero while policy std declined to `0.149`.

### Scope

- Inserted a `0.04 rad/s` low-speed rotation stage between the hold stage and
  the existing `0.10 rad/s` stage. At the unchanged `0.80` sustain ratio, its
  promotion threshold is `0.032 rad/s`, close to the demonstrated policy range.
- The five-stage targets are now `[0.0, 0.04, 0.10, 0.25, 0.50] rad/s`, with
  durations `[1, 1, 2, 4, 10] s` and orthogonal tolerances
  `[0.10, 0.08, 0.05, 0.075, 0.10] rad/s`.
- Added a `0.10` milestone bonus for completing the bridge stage and preserved
  the previous bonuses `[0.25, 0.50, 1.0]` for the original speed stages.
  Energy regularization remains disabled through `0.10 rad/s`, then uses the
  existing `0.25` and `1.0` level scales.
- Generalized curriculum validation from exactly four levels to hold plus one
  or more strictly increasing rotation levels, and added a fail-fast check that
  stage bonus count equals promotion count.
- MuJoCo owns the complete curriculum configuration; Motrix inherits it through
  its existing owner default and retains only its backend-specific timestep.
  Reward topology, failure conditions, observations, actions, assets, reset
  source, PD gains, and domain randomization remain unchanged.

### Validation

- Ruff passed for the sustained-rotation environment and focused LEAP tests.
- `tests/envs/test_leap_inhand.py` passed all 45 tests, including the five-stage
  defaults, aligned stage-bonus contract, reward topology, collision/cache
  boundaries, and available backend reset-step coverage.
- MuJoCo and Motrix Hydra owner composition both succeeded. The composed configs
  resolve five targets/tolerances/energy scales and four promotion bonuses;
  Motrix retains its backend-specific `0.01 s` simulation timestep.
- MuJoCo smoke run `2026-07-20_12-00-08_mujoco` completed one iteration with
  finite 100-observation/16-action metrics, no failures, and diagnostics for
  curriculum Levels 0-4.
- Motrix smoke run `2026-07-20_12-00-29_motrix` completed the same contract with
  finite metrics, no failures, and diagnostics for curriculum Levels 0-4.
- Smoke tests validate runtime structure only. The bridge stage must still be
  evaluated through a fresh or checkpoint-initialized training ablation.

## 2026-07-20 - Sustained Rotation Mid-Speed Curriculum Bridge

### Training Evidence

- The first bridge run `2026-07-20_12-03-37_mujoco` unlocked the `0.10 rad/s`
  and `0.25 rad/s` stages. Its final Level 3 occupancy was 41.5%, and it produced
  the first nonzero two-second sustained-rotation metric, but never entered the
  `0.50 rad/s` stage or sustained valid rotation for five seconds.
- Continuation run `2026-07-20_14-12-58_mujoco` converged toward a stable
  low-speed strategy: final position error fell to 6.25 mm and world-Z speed
  EMA reached `0.0651 rad/s`, but the final 50-iteration Level 3 occupancy mean
  regressed to 7.07%, Level 4 remained zero, and policy std declined to 0.091.
- The evidence indicates that extending the unchanged configuration reinforces
  the `0.10 rad/s` local optimum rather than bridging the required promotion
  threshold from `0.08` to `0.20 rad/s`.

### Scope

- Inserted a `0.16 rad/s` target between `0.10` and `0.25 rad/s`. At the
  unchanged 0.80 sustain ratio, the new promotion threshold is `0.128 rad/s`.
- The six-stage targets are `[0.0, 0.04, 0.10, 0.16, 0.25, 0.50] rad/s`, with
  durations `[1, 1, 2, 2, 4, 10] s`, orthogonal tolerances
  `[0.10, 0.08, 0.05, 0.06, 0.075, 0.10] rad/s`, and promotion bonuses
  `[0.10, 0.25, 0.35, 0.50, 1.0]`.
- Energy regularization remains disabled through the new `0.16 rad/s` skill
  acquisition stage, then retains the existing 0.25 and 1.0 scales.
- Increased the episode budget from 20 to 25 seconds. The sequential stages now
  require 20 seconds, leaving five seconds for transitions and failed attempts
  without weakening the final ten-second requirement.
- Added validation that rejects curriculum durations whose sum is greater than
  or equal to the episode budget. Reward equations, failure conditions,
  observation/action dimensions, PD control, assets, reset source, and domain
  randomization remain unchanged.

### Validation

- Ruff passed for the sustained-rotation environment and focused LEAP tests.
- `tests/envs/test_leap_inhand.py` passed all 45 tests, including six-stage
  defaults, duration-budget coverage, reward topology, collision/cache
  boundaries, and available backend reset-step coverage.
- MuJoCo and Motrix Hydra owner composition both succeeded with six targets,
  durations, tolerances, and energy scales; five promotion bonuses; and a
  25-second episode budget.
- MuJoCo smoke run `2026-07-20_14-40-36_mujoco` completed one iteration with
  finite 100-observation/16-action metrics, no failures, and diagnostics for
  curriculum Levels 0-5.
- Motrix smoke run `2026-07-20_14-41-00_motrix` completed the same contract with
  finite metrics, no failures, and diagnostics for curriculum Levels 0-5.
- Smoke tests validate runtime structure only. The `0.16 rad/s` bridge still
  requires a checkpoint-initialized training ablation.

## 2026-07-20 - RSL-RL Policy-Only Warm Start

### Training Evidence

- Continuation run `2026-07-20_14-42-39_mujoco` inherited checkpoint `1150`
  from `2026-07-20_14-12-58_mujoco`, including its approximately `0.094`
  policy standard deviation and optimizer state.
- The continuation failed to cross the new `0.16 rad/s` curriculum bridge:
  final Level 2 occupancy was 68.7%, Levels 4-5 remained zero, target-axis
  speed EMA regressed to `0.0086 rad/s`, and final success remained zero.
- This evidence requires separating policy transfer from exact training-state
  resume so the learned controller can be reused without inheriting collapsed
  exploration or stale optimizer moments.

### Scope

- Added `algo.load_mode` with backward-compatible default `resume`.
- Added `warm_start_policy`, which transfers actor MLP weights and actor
  observation-normalizer state but excludes all `distribution.*` state. The
  configured action distribution therefore starts at its task-owned
  `init_std`; critic, optimizer, iteration, and logger counters also remain
  freshly initialized.
- Kept normal resume and playback checkpoint loading unchanged. The feature is
  generic to RSL-RL PPO and does not alter reward, curriculum, environment,
  backend, asset, reset, observation, or action contracts.

### Validation

- Focused Ruff passed for the training entry point, RSL-RL helper, and new
  warm-start tests. All four warm-start unit tests passed.
- Hydra composition preserved `resume` as the default, accepted the explicit
  `warm_start_policy` override, and resolved the sustained-rotation actor's
  configured initial standard deviation to `0.3`.
- MuJoCo smoke run `2026-07-20_15-20-59_mujoco` warm-started from checkpoint
  `1150`, completed one iteration, and wrote `model_0.pt` with checkpoint
  iteration `0` and action standard deviation `0.29906`.
- Motrix smoke run `2026-07-20_15-21-20_motrix` completed the same contract and
  wrote `model_0.pt` with checkpoint iteration `0` and action standard
  deviation `0.29986`.
- Direct checkpoint inspection measured the source standard deviation as
  `0.09441`. The new checkpoints' iteration `0` and approximately `0.30`
  standard deviations demonstrate that neither source iteration nor collapsed
  exploration was inherited. RSL-RL numbers a one-iteration fresh run from
  zero, so `model_0.pt` is the expected filename.
- These smoke runs establish checkpoint-loading semantics and cross-backend
  execution only; they do not establish improved sustained-rotation learning.

## 2026-07-20 - Sustained Rotation 0.07 rad/s Curriculum Bridge

### Training Evidence

- Policy-only warm-start run `2026-07-20_16-58-06_mujoco` with initial action
  standard deviation `0.15` improved Level 2 occupancy to 41.0% after 300
  iterations, but two-second sustained success and Level 3 occupancy remained
  zero.
- Its continuation run `2026-07-20_17-33-44_mujoco` reached 53.6% final Level
  2 occupancy while target-axis speed EMA regressed from `0.0453` to
  `0.0307 rad/s`; position error increased from 16.3 to 18.1 mm and all
  sustained-success metrics remained zero.
- The policy can repeatedly complete the `0.04 rad/s` stage but cannot bridge
  directly to the `0.10 rad/s` stage's `0.08 rad/s` promotion threshold.
  Continuing the unchanged task converges toward a stable low-speed strategy.

### Scope

- Inserted a `0.07 rad/s` stage between `0.04` and `0.10 rad/s`. With the
  unchanged `0.80` sustain ratio, its promotion threshold is `0.056 rad/s`,
  inside the range demonstrated transiently by the existing policy.
- The seven-stage targets are now
  `[0.0, 0.04, 0.07, 0.10, 0.16, 0.25, 0.50] rad/s`. The new stage requires
  1.5 seconds, uses `0.065 rad/s` orthogonal tolerance, has no energy penalty,
  and awards a `0.175` promotion bonus interpolated between the adjacent
  `0.10` and `0.25` bonuses.
- Total required stage time increases from 20.0 to 21.5 seconds, leaving 3.5
  seconds within the unchanged 25-second episode for transitions.
- Reward equations, sustain ratio, position/contact gates, failure conditions,
  observations, actions, control, reset source, assets, and domain
  randomization remain unchanged. Motrix inherits the curriculum from the
  MuJoCo owner and retains only its backend-specific timestep override.

### Validation

- Focused Ruff passed for the sustained-rotation environment and LEAP tests.
- `tests/envs/test_leap_inhand.py` passed all 47 tests, including seven-stage
  defaults, duration-budget coverage, reward topology, collision/cache
  boundaries, and available backend reset-step coverage.
- MuJoCo and Motrix Hydra composition both resolved targets
  `[0.0, 0.04, 0.07, 0.10, 0.16, 0.25, 0.50]`, durations
  `[1.0, 1.0, 1.5, 2.0, 2.0, 4.0, 10.0]`, and six promotion bonuses. Motrix
  retained its backend-specific `0.01 s` simulation timestep.
- MuJoCo smoke run `2026-07-20_18-06-41_mujoco` completed one iteration with
  finite 100-observation/16-action metrics, zero failures, and curriculum
  diagnostics for Levels 0-6.
- Motrix smoke run `2026-07-20_18-07-05_motrix` completed the same contract
  with finite metrics, zero failures, and diagnostics for Levels 0-6.
- Smoke tests validate runtime structure only. The `0.07 rad/s` bridge still
  requires a policy-only warm-start training ablation against the preserved
  pre-bridge checkpoint baseline.

## 2026-07-20 - Retention-Conditioned Positive Spin Reward

### Training Evidence

- Seven-stage bridge run `2026-07-20_18-16-42_mujoco` produced the first
  nonzero two-second sustained metric and briefly reached the `0.10 rad/s`
  stage, validating the `0.07 rad/s` curriculum direction.
- Continuation run `2026-07-20_19-02-45_mujoco` increased final `0.07 rad/s`
  stage occupancy to 77.6% and briefly reached 0.293% two-second sustained
  occupancy, but final position error increased to 21.7 mm while stage-valid
  occupancy fell to 9.2% and final two-second success remained zero.
- Positive spin reward remained available outside the 15 mm curriculum
  position gate, allowing return to increase without producing valid sustained
  rotation. This reward/curriculum mismatch, rather than another speed bridge,
  is the next isolated bottleneck.

### Scope

- Positive visible spin progress is now multiplied by the existing smooth
  retention signal before applying `spin_progress_scale`. Full positive spin
  reward is available at the rotation anchor and decays continuously as the
  object moves away.
- Zero and negative visible progress are unchanged. Reverse-axis rotation
  therefore retains its full negative reward and cannot reduce that cost by
  moving the object away from the anchor.
- No penalty was added. Rotation stability, hold reward, curriculum, stage
  bonuses, position/contact gates, failures, observations, actions, control,
  reset source, assets, and both backend configurations remain unchanged.

### Validation

- Focused Ruff passed for the sustained-rotation environment and LEAP tests.
- `tests/envs/test_leap_inhand.py` passed all 50 tests. Direct numeric coverage
  proves that positive visible progress receives full reward at retention 1.0,
  one-quarter reward at retention 0.25, and unchanged full negative reward for
  reverse-axis progress at retention 0.25.
- MuJoCo and Motrix Hydra composition preserved the seven-stage targets,
  `spin_progress_scale=1.5`, and Motrix's backend-specific `0.01 s` simulation
  timestep.
- MuJoCo smoke run `2026-07-20_20-02-14_mujoco` completed one iteration with
  finite 100-observation/16-action metrics, zero failures, and Level 0-6
  diagnostics.
- Motrix smoke run `2026-07-20_20-02-51_motrix` completed the same runtime
  contract with finite metrics, zero failures, and Level 0-6 diagnostics.
- One-iteration smoke remains in the hold stage, so it validates integration
  but not the positive-spin branch; that branch is covered directly by the
  numeric reward test. Learned retention-conditioned rotation still requires
  a policy-only warm-start training ablation.

## 2026-07-20 - Retention-Floored Positive Spin Reward

### Training Evidence

- Policy-only warm-start run `2026-07-20_20-38-59_mujoco` evaluated direct
  multiplication of positive spin progress by retention. Final target-axis
  speed EMA regressed to `0.0170 rad/s`, Level 2 occupancy fell to zero, mean
  return fell to `-3.01`, and raw drop rate increased to 2.12%.
- The direct retention factor is too aggressive for exploration: with the
  existing 15 mm retention sigma it pays only about 37% of positive spin
  reward at the valid-position boundary and about 12% at 22 mm. The policy
  preserved position but lost the rotation skill and did not recover over 300
  iterations.

### Scope

- Replaced the direct positive-spin retention multiplier with
  `0.5 + 0.5 * retention`. Positive spin therefore receives full reward at the
  anchor, about 68% at the 15 mm boundary, and never less than 50%, preserving
  a smooth incentive to return toward the anchor without erasing exploration.
- Zero and negative visible progress remain unchanged, so reverse-axis
  rotation retains its full negative reward at every object position.
- No coefficient, penalty, curriculum stage, bonus, gate, failure condition,
  observation, action, control, reset, asset, or backend configuration changed.

### Validation

- Focused Ruff passed for the sustained-rotation environment and LEAP tests.
- `tests/envs/test_leap_inhand.py` passed all 50 tests. Direct numeric coverage
  proves positive visible progress receives 100% reward at retention 1.0,
  62.5% reward at retention 0.25, and unchanged full negative reward for
  reverse-axis progress at retention 0.25.
- MuJoCo and Motrix Hydra composition preserved the seven-stage targets,
  `spin_progress_scale=1.5`, and Motrix's backend-specific `0.01 s` simulation
  timestep.
- MuJoCo smoke run `2026-07-20_21-25-34_mujoco` completed one iteration with
  finite 100-observation/16-action metrics, zero failures, and Level 0-6
  diagnostics.
- Motrix smoke run `2026-07-20_21-26-00_motrix` completed the same runtime
  contract with finite metrics, zero failures, and Level 0-6 diagnostics.
- One-iteration smoke remains in the hold stage, so it validates integration
  but not the positive-spin branch; that branch is covered directly by the
  numeric reward test. Learned retention-floored rotation still requires a
  policy-only warm-start training ablation.

## 2026-07-20 - RSL-RL Actor-Critic Warm Start

### Training Evidence

- Retention-floor run `2026-07-20_21-27-33_mujoco` was policy-only
  warm-started from bridge checkpoint `2026-07-20_18-16-42_mujoco/model_250.pt`.
  The source checkpoint had 48.5% Level 2 occupancy, 0.62% Level 3 occupancy,
  and `0.0537 rad/s` target-axis EMA. The adapted run ended with 1.20% Level 2,
  zero Level 3, `0.0163 rad/s` target-axis EMA, and a 1.37% per-step drop
  termination rate.
- Because policy-only warm start creates a fresh random critic while also
  changing reward semantics, this run does not isolate the reward formula
  from transfer instability. The next ablation must preserve both learned
  function approximators while resetting training progress and optimizer
  state.

### Scope

- Added generic RSL-RL PPO load mode `warm_start_actor_critic`.
- The mode transfers actor MLP and actor observation-normalizer state while
  excluding `distribution.*`, so action standard deviation remains at the
  destination task's configured `init_std`.
- It also transfers critic MLP and critic observation-normalizer state using a
  strict key-contract check. Optimizer state, iteration, logger counters, and
  action distribution remain freshly initialized.
- Existing `resume` and `warm_start_policy` semantics are unchanged. Reward,
  curriculum, environment, backend, observation, action, reset, assets, and
  existing checkpoints are unchanged.

### Validation

- Focused Ruff passed for the training entry point, RSL-RL checkpoint helper,
  structured PPO config, and warm-start tests.
- All six warm-start unit tests passed. Coverage verifies complete resume,
  policy-only transfer, actor+critic transfer, preservation of destination
  action distribution and fresh counters, critic incompatibility rejection,
  and unknown-mode rejection.
- MuJoCo and Motrix Hydra composition accepted
  `algo.load_mode=warm_start_actor_critic`, retained 16 environments for the
  smoke configuration, and preserved backend-owned simulation timesteps of
  `0.005 s` and `0.01 s`, respectively.
- MuJoCo smoke run `2026-07-20_23-05-49_mujoco` loaded actor and critic from
  `2026-07-20_18-16-42_mujoco/model_250.pt`, completed one iteration with 16
  environments, wrote `model_0.pt`, and reported destination action standard
  deviation `0.30`, finite 100-observation/16-action metrics, and zero
  failures.
- Motrix smoke run `2026-07-20_23-06-16_motrix` completed the same transfer
  contract, wrote `model_0.pt`, reported action standard deviation `0.30`, and
  produced finite metrics with zero failures.
- These smoke runs establish loading and cross-backend runtime semantics only.
  Whether actor+critic transfer prevents destructive adaptation under the
  changed reward still requires the planned MuJoCo training ablation.

## 2026-07-21 - Sustained Rotation 0.085 rad/s Curriculum Bridge

### Training Evidence

- Actor+critic warm-start run `2026-07-20_23-09-41_mujoco`, initialized from
  bridge checkpoint `2026-07-20_18-16-42_mujoco/model_250.pt`, avoided the
  policy-only transfer collapse. At checkpoint `275`, episode length reached
  500 steps, Level 3 (`0.10 rad/s`) occupancy reached 54.1%, two-second
  sustained occupancy reached 7.50%, and termination rate was zero.
- The run never reached Level 4. At checkpoint `275`, target-axis EMA was only
  `0.0385 rad/s`, while the existing `0.10 rad/s` stage requires
  `0.08 rad/s` continuously for two seconds. This isolates the next bottleneck
  at the transition from `0.07` to `0.10 rad/s` rather than retention, palm
  contact, or actor/critic transfer.

### Scope

- Inserted an episode-local `0.085 rad/s` bridge between `0.07` and
  `0.10 rad/s`. With the unchanged `0.80` sustain ratio, promotion requires
  `0.068 rad/s` continuously for one second.
- The eight-stage targets are now
  `[0.0, 0.04, 0.07, 0.085, 0.10, 0.16, 0.25, 0.50] rad/s`. The bridge uses
  `0.058 rad/s` orthogonal tolerance, zero energy regularization, and a `0.30`
  completion bonus interpolated between the preserved adjacent bonuses.
- Total required stage time increases from 21.5 to 22.5 seconds, leaving 2.5
  seconds inside the unchanged 25-second episode.
- Reward equations, sustain ratio, position/contact gates, failure conditions,
  observations, actions, control, reset source, assets, and backend behavior
  remain unchanged. Model dimensions therefore remain checkpoint-compatible.

### Validation

- Focused Ruff passed for the sustained-rotation environment and LEAP tests.
- `tests/envs/test_leap_inhand.py` passed all 50 tests, including the
  eight-stage typed defaults, 22.5-second duration budget, reward bonus
  cardinality, and available backend reset-step coverage.
- MuJoCo and Motrix Hydra composition both resolved targets
  `[0.0, 0.04, 0.07, 0.085, 0.10, 0.16, 0.25, 0.50]`, durations
  `[1.0, 1.0, 1.5, 1.0, 2.0, 2.0, 4.0, 10.0]`, and seven promotion bonuses.
  Motrix retained its backend-owned `0.01 s` simulation timestep.
- MuJoCo smoke run `2026-07-21_15-07-38_mujoco` actor+critic warm-started from
  `2026-07-20_23-09-41_mujoco/model_275.pt`, completed one iteration with 16
  environments, wrote `model_0.pt`, and produced finite 100-observation /
  16-action metrics, Level 0-7 diagnostics, and zero failures.
- Motrix smoke run `2026-07-21_15-08-17_motrix` completed the same checkpoint
  and curriculum contract with finite metrics, Level 0-7 diagnostics, and zero
  failures.
- Smoke tests validate integration and checkpoint compatibility only. Whether
  the new bridge unlocks the `0.10 rad/s` stage requires the planned MuJoCo
  actor+critic warm-start training ablation.

## 2026-07-21 - Sustained Target-Speed Duration Reward

### Training Evidence

- Full-resume run `2026-07-21_15-38-42_mujoco` repeatedly entered the
  `0.10 rad/s` stage and reached 66.3% Level 4 occupancy. At iteration 442,
  target-axis EMA reached `0.0871 rad/s`, stage-valid occupancy reached 93.3%,
  and two-second sustained occupancy reached 70.9%.
- The speed was transient. By iteration 450, Level 4 occupancy remained 66.3%
  while target-axis EMA fell to `0.0431 rad/s` and stage-valid occupancy fell
  to 0.10%. No environment reached Level 5 during the 300-iteration resume.
- Existing positive spin and rotation-stability rewards remain available at
  partial target speed and do not directly value an uninterrupted duration.
  The learned policy therefore accelerates through lower stages, then settles
  into a slower stable grasp.

### Scope

- Added Gaussian EMA target-speed quality with configurable tolerance ratio
  `0.25`. Quality is one at the active target and decays symmetrically for
  underspeed or overspeed rotation.
- Positive spin and rotation-stability rewards are multiplied by independent
  soft-floor weights for speed quality and consecutive valid-stage duration.
  Each weight ranges from 0.5 to 1.0, so low-speed, zero-duration positive spin
  receives 25% of the previous reward while exact sustained target speed
  receives 100%.
- Reverse-axis progress remains fully negative and is not attenuated by speed
  or duration quality. Hold reward, curriculum thresholds/durations, milestone
  bonuses, failures, observations, actions, control, reset, assets, and backend
  contracts remain unchanged.
- Added diagnostics for target-speed quality, normalized stage-duration
  progress, and consecutive valid seconds.

### Validation

- Focused Ruff passed for the sustained-rotation environment and LEAP tests.
- `tests/envs/test_leap_inhand.py` passed all 52 tests, including exact
  target-speed quality and duration-weight reward checks.
- MuJoCo and Motrix Hydra composition both resolved tolerance ratio `0.25`,
  identical eight-stage target speeds and reward settings, and retained their
  backend-owned simulation timesteps of `0.005 s` and `0.01 s`, respectively.
- MuJoCo smoke run `2026-07-21_16-40-18_mujoco` actor+critic warm-started from
  `2026-07-21_15-38-42_mujoco/model_575.pt`, completed one iteration with 16
  environments, emitted all three new rotation diagnostics, and reported zero
  failures.
- Motrix smoke run `2026-07-21_16-40-47_motrix` completed the same checkpoint
  compatibility contract with finite 100-observation / 16-action metrics, all
  new diagnostics, and zero failures.
- Smoke tests validate integration only. Whether speed- and duration-weighted
  shaping prevents late-stage slowdown requires a MuJoCo training ablation.

## 2026-07-21 - Cache-Reset Sustained LEAP Ball Rotation

### Scope

- Added independent task `leap_inhand_ball_sustained_cache`, initialized from
  `robots/leap_hand/caches/ball_grasp_official_50k.npy` rather than the fixed
  ball home keyframe. The original sustained task and its checkpoints remain
  unchanged.
- The new task inherits the sustained reward terms, exact weights, eight-stage
  curriculum, 100-dimensional observation, 16-dimensional action, control,
  palm-contact failure, and three-step failure debounce without modification.
- Each reset records its sampled cache ball center as `rotation_anchor_pos` and
  its sampled hand configuration as `init_pose`. Retention and workspace gates
  therefore remain episode-relative.
- Added an overridable sustained-task drop hook. The original task retains its
  absolute home-height threshold; the cache task fails after a 7 mm vertical
  drop relative to the sampled cache ball height.
- The formal task is configured for fresh PPO training. It contains no resume,
  warm-start, or checkpoint source setting.

### Validation

- Focused Ruff passed for the new environment, registry bootstrap, support
  matrix, and targeted tests.
- LEAP and support-matrix suites passed all 61 tests. Coverage verifies task
  registration, exact cache-row resets, per-reset `init_pose` and ball anchor,
  the 7 mm relative drop boundary, finite 100-observation reset/step behavior,
  and both configured backends.
- MuJoCo and Motrix Hydra composition resolved the new task with
  `reset_source=cache`, the 50k cache path, and 7 mm drop distance. Both reward
  and curriculum trees compare exactly equal to the original sustained task.
  Default `load_run=-1` and `checkpoint=-1` confirm fresh training semantics.
- Fresh-policy MuJoCo smoke run `2026-07-21_17-50-30_mujoco` loaded the
  `(50000, 23)` cache, completed one iteration with 16 environments, emitted
  finite 100-observation / 16-action metrics, and reported zero palm or
  workspace failures. Random exploration produced a 6.25% termination rate
  from the relative drop boundary.
- Fresh-policy Motrix smoke run `2026-07-21_17-51-06_motrix` completed the same
  reset and policy contract with finite metrics and zero palm failures. Its
  short-rollout drop and workspace drift were higher, so the MuJoCo-generated
  cache is not yet established as a Motrix-stable training distribution.
- A 512-state MuJoCo replay found zero initial palm contacts, zero samples with
  fewer than two fingertip contacts, and 2.67 mean fingertip contacts. With
  zero policy actions, 25.2% terminated within one second under the sustained
  task's `0.005 s` simulation timestep. Repeating at the cache-generation
  timestep of `1/120 s` produced a 25.0% termination rate; all terminations
  crossed the relative 7 mm drop boundary, while palm contact remained zero.
- The task integration is valid, but the replay result shows that the 50k
  cache contains a meaningful unstable tail under the sustained-task failure
  criterion. No reward, drop tolerance, or cache row was silently changed to
  hide this evidence. Formal training may either treat these rows as failure
  examples or first use a separately approved offline stability filter.

## 2026-07-21 - Cache Rotation Retention Grey-Zone Experiment

### Scope

- Added configurable positive `anchor_proximity` shaping to sustained LEAP ball
  rotation. It is linear from `1.0` at the 15 mm curriculum gate to `0.0` at
  the 30 mm workspace boundary and remains clipped to those limits.
- Enabled the term at scale `0.1` only for
  `leap_inhand_ball_sustained_cache`. The original home-reset sustained task
  keeps a zero scale, preserving its training behavior.
- Added separate reward and proximity diagnostics. Rotation reward,
  curriculum gates, termination conditions, observations, and checkpoint
  contracts are unchanged.
- This began as a temporary single-variable training experiment. The completed
  1,000-iteration ablation and deterministic checkpoint replay showed practical
  retention and curriculum improvements, so the user approved retaining it as
  the cache task's new baseline pending cross-seed confirmation.

### Validation

- Focused Ruff passed for the sustained environment and LEAP test module.
- The LEAP targeted suite passed all 57 tests. The new numeric test verifies
  proximity values `1.0`, `0.5`, and `0.0` at the 15 mm gate, 22.5 mm midpoint,
  and 30 mm workspace boundary respectively.
- MuJoCo and Motrix Hydra composition both resolved the cache task with
  `anchor_proximity_scale=0.1`; the inherited home-reset task retains `0.0`.
- MuJoCo smoke run `2026-07-21_22-47-15_mujoco` completed one iteration with
  finite 100-observation / 16-action metrics and reported mean
  `retention/anchor_proximity=0.9573`.
- Motrix smoke run `2026-07-21_22-50-49_motrix` completed the same contract and
  reported finite metrics with mean `retention/anchor_proximity=0.9375`.

## 2026-07-22 - Staged Training Budget Guidance

### Scope

- Added repository guidance requiring unproven training changes to start with
  the smallest budget that can diagnose direction, instead of defaulting to
  long runs such as 1,000 iterations.
- Training scale and checkpoint cadence must be selected from measured runtime,
  statistical needs, and comparison goals. Long training is reserved for
  changes supported by metrics, checkpoint inspection, and native-viewer
  evidence whose learning curves have not saturated.
- Clarified that checkpoint warm-start and exploration `init_std` overrides are
  optional experimental controls, not mandatory parts of every small run.

### Validation

- Documentation-only process update; no runtime behavior or configuration was
  changed.

## 2026-07-22 - Late-Stage Retention-Conditioned Spin Experiment

### Evidence

- A deterministic checkpoint-999 replay used 256 environments for 500 steps
  and classified 128,000 samples by the 15 mm retention gate and stage speed
  gate. Both gates failed in 72.46% of all samples, including 88.57% at Level 3
  and 83.79% at Level 4.
- Only-retention and only-speed failures were 5.47% and 4.52% respectively.
  The dominant bottleneck is therefore coupled late-stage displacement and
  insufficient sustained speed, not either gate in isolation.

### Scope

- Replaced the hard-coded `0.5 + 0.5 * retention` positive-spin weight with a
  validated per-stage floor: `floor + (1 - floor) * retention`.
- The original home-reset sustained task retains floor `0.50` at every level,
  preserving its reward behavior. The cache task keeps `0.50` for Levels 0-2
  and uses `0.25` for Levels 3-7, where deterministic replay found the coupled
  bottleneck.
- Reverse-axis progress remains fully penalized. Anchor proximity, stage gates,
  termination, observation, action, and checkpoint contracts are unchanged.
- Added runtime diagnostics for the active floor and effective retention weight.

### Validation

- Focused Ruff passed for the sustained environment and LEAP test module.
- The LEAP targeted suite passed all 58 tests, including numeric coverage that
  proves zero-retention positive spin receives weight `0.50` in the preserved
  early-stage behavior and `0.25` in the tightened late-stage behavior.
- MuJoCo and Motrix Hydra composition both resolved the cache task with floors
  `[0.50, 0.50, 0.50, 0.25, 0.25, 0.25, 0.25, 0.25]`.
- MuJoCo smoke run `2026-07-22_02-36-35_mujoco` completed one iteration with
  finite 100-observation / 16-action metrics and emitted both new diagnostics.
- Motrix smoke run `2026-07-22_02-37-01_motrix` completed the same contract with
  finite metrics and both new diagnostics.

## 2026-07-22 - LEAP Finger-Gaiting Rotation Task

### Evidence

- Native-viewer inspection showed that learned policies commonly produced an
  initial rotation burst and then stopped instead of reconnecting fingers for
  another push.
- Existing sustained-task diagnostics showed occasional two-second rotation but
  no five- or ten-second success. The prior reward observed global ball speed
  and treated two, three, or four fingertip contacts as the same saturated
  support value; it did not identify safe release/recontact sequences.

### Scope

- Added the independent `LeapInhandBallFingerGaitingRotation` task and
  `leap_inhand_ball_finger_gaiting` MuJoCo/Motrix owner configurations. Existing
  sustained and sustained-cache task configurations remain unchanged.
- Preserved the cache reset source, 100-dimensional policy observation, action
  contract, physical failure conditions, and base signed +Z spin reward.
- Added a debounced finger-handoff state machine. A qualified handoff requires
  one finger to remain released for at least two control steps, at least two
  other fingertip contacts throughout, retained/no-palm geometry, sufficient
  positive axis speed, and recontact before the ten-step timeout without losing
  more than the configured speed ratio. A global four-step cooldown prevents
  contact chatter from repeatedly earning events.
- Stage 0 requires stable three-finger support. Rotation stages require
  increasing numbers of qualified handoffs `[1, 1, 2, 2, 3, 4, 6]` in addition
  to the existing continuous speed, axis purity, retention, and duration gates.
  Stage durations increase through a final ten-second stage within a 35-second
  episode.
- Added positive stable-support and release-progress shaping plus a one-shot
  qualified-handoff bonus. No new penalty was introduced.
- Added per-finger contact duty and handoff metrics, aggregate qualified/useful
  handoff rates, release activity, inactive-finger fraction, stage handoff
  progress, total handoffs, longest valid rotation, and longest interval without
  a contact transition.
- Added no-op stage-skill hooks to the sustained base environment so specialized
  tasks can add gates and rewards without duplicating its update loop. The
  default hook returns all-valid/zero-reward arrays, preserving existing task
  behavior.

### Validation

- Focused Ruff checks passed. Ruff formatting was applied only to the sustained
  base, new finger-gaiting module, and LEAP test module; the support-matrix file
  was deliberately not globally reformatted because it contains unrelated
  pre-existing worktree changes.
- MuJoCo and Motrix Hydra composition both resolved
  `LeapInhandBallFingerGaitingRotation` with required handoffs
  `[0, 1, 1, 2, 2, 3, 4, 6]`.
- The first pytest attempt never entered a test because the Windows user temp
  directory denied fixture creation. Re-running with an isolated writable
  `C:\tmp` base directory passed all 61 LEAP tests.
- MuJoCo smoke run `2026-07-22_12-15-17_mujoco` completed one iteration with a
  finite 100-observation / 16-action contract and all gaiting diagnostics.
- Motrix smoke run `2026-07-22_12-15-48_motrix` completed the same contract and
  emitted the same diagnostics. Zero qualified handoffs in these eight-step
  random-policy smokes is expected and is not evidence of learnability.

## 2026-07-22 - Finger-Gaiting Markov Observation Ablation

### Evidence

- The 150-iteration baseline (`2026-07-22_12-19-14_mujoco`, resumed as
  `2026-07-22_12-36-43_mujoco`) learned stable support and ended with 97.07% of
  environments in Level 1, but only 0.049% in Level 2.
- The final target-axis EMA was `0.0066 rad/s` against a mean target of
  `0.0388 rad/s`; longest valid rotation was `0.364 s`, with zero two-, five-,
  or ten-second successes.
- Qualified handoff rate briefly peaked at `0.00537` near iteration 106 and
  collapsed to `0.000244`. Index/thumb contact duty approached 98%/99.9%, while
  the longest interval without any contact transition reached `9.56 s`.
- Reward eligibility depended on release activity, release duration, release
  start speed, cooldown, and completed stage handoffs, but none of those state
  variables were visible to the feed-forward policy. The task was therefore
  partially observable at the handoff reward boundary.

### Scope

- Kept reward, curriculum, termination, cache reset, action space, and both
  backend configurations unchanged for a single-variable observation ablation.
- Extended only `LeapInhandBallFingerGaitingRotation` from 100 to 114 policy
  observations. Existing sustained and sustained-cache tasks remain at 100.
- Added four release-active flags, four normalized release-duration values,
  four normalized release-start speeds, one normalized global cooldown, and one
  normalized current-stage handoff-progress value.
- Added a pure normalization helper with clipping and explicit zero progress for
  stages that require no handoff. Reset and runtime observations share this
  helper.
- This intentionally changes the new task's checkpoint contract. Earlier
  100-observation finger-gaiting checkpoints are retained as the baseline but
  cannot be resumed or warm-started into the 114-observation policy.

### Validation

- Ruff formatted the new finger-gaiting environment and its LEAP test module;
  the subsequent focused Ruff check passed.
- The focused LEAP suite passed all 64 tests. Runtime assertions confirm the
  finger-gaiting task returns 114 policy observations on both backends while
  the existing sustained-cache task remains at 100 observations.
- Hydra composition resolved `LeapInhandBallFingerGaitingRotation` for both
  owners: MuJoCo at `sim_dt=0.005` and Motrix at `sim_dt=0.01`.
- MuJoCo smoke run `2026-07-22_12-56-38_mujoco` completed one iteration with
  16 environments. Actor and critic both reported `in_features=114`, and all
  reward, retention, curriculum, and gaiting diagnostics were finite.
- Motrix smoke run `2026-07-22_12-56-55_motrix` completed the same contract;
  actor and critic both reported `in_features=114` and finite diagnostics.
- These random-policy smoke runs validate integration only. Their zero
  qualified handoffs do not establish or refute task learnability.

## 2026-07-22 - Stationary Finger-Handoff Curriculum Stage

### Evidence

- The 114-observation run `2026-07-22_12-59-49_mujoco` reached 97.08% Level 1
  but never reached Level 2. Its final target-axis EMA was `0.00254 rad/s`, mean
  total handoffs were `0.0089`, and longest valid rotation was `0.073 s`.
- The observation ablation exposed the handoff state but did not make release
  and recontact frequent enough to learn. The final release-active fraction was
  only `0.084%`.
- The previous Level 1 simultaneously required a qualified handoff and positive
  target rotation. This left the first handoff event too sparse to bootstrap.

### Scope

- Added one task-owned stationary handoff stage between stable support and the
  first `0.04 rad/s` rotation stage. This stage requires one safe release and
  recontact while retaining the ball, but imposes no signed rotation-speed or
  speed-recovery requirement.
- Shifted the existing rotation stages without changing their target speeds,
  tolerances, energy schedule, handoff counts, rewards, or final objective.
- Extended all stage-aligned curriculum, reward, and finger-gaiting arrays from
  eight to nine entries. The episode remains 35 seconds and total required
  stage duration is 30 seconds.
- Preserved the 114-observation contract, cache reset, termination conditions,
  physical model, control parameters, and both backend ownership structure.
- Generalized zero-speed stage validity and handoff qualification. Existing
  tasks are behaviorally unchanged because their only zero-speed stage is the
  original hold stage.
- Restricted stationary handoff qualification to zero-speed stages whose
  required handoff count is positive. The original hold stage cannot emit
  qualified handoffs, alter cooldown, or pollute handoff diagnostics.

### Validation

- Ruff formatted the changed Python files and the focused Ruff check passed.
- The combined LEAP/config run produced `205 passed, 5 failed`. All five
  failures pre-existed outside this change: one Allegro LEAP owner omits an
  explicit `training.sim_backend`, while the rotation-v2 and toss owners do not
  expose the generic `reward.scales` mapping expected by the repository-wide
  config test.
- Re-running only the LEAP suite and the new dual-backend curriculum compose
  assertion passed all 67 tests.
- Explicit Hydra composition resolved both backends to nine stages with targets
  `[0.0, 0.0, 0.04, 0.07, 0.085, 0.10, 0.16, 0.25, 0.50]`, required handoffs
  `[0, 1, 1, 1, 2, 2, 3, 4, 6]`, and 30 seconds of total stage duration.
- An initial smoke exposed that the generic zero-speed allowance also emitted a
  diagnostic-only handoff in Level 0. The implementation was tightened so only
  zero-speed stages with a positive handoff requirement receive the allowance.
- Final MuJoCo smoke run `2026-07-22_14-54-45_mujoco` completed one iteration
  with 16 environments, a 114-observation actor/critic contract, finite metrics,
  nine curriculum diagnostics, and zero Level-0 handoffs.
- Final Motrix smoke run `2026-07-22_14-55-02_motrix` completed the same runtime
  contract with finite metrics and zero Level-0 handoffs.

## 2026-07-22 - Per-Level Finger-Gaiting Gate Diagnostics

### Evidence

- The stationary-stage run `2026-07-22_14-58-36_mujoco` promoted 45.5% of
  environments into the first `0.04 rad/s` rotation stage but promoted none to
  the next stage.
- Existing duration diagnostics treated every nonzero curriculum level as a
  rotation stage. The new zero-speed handoff stage therefore produced false
  two-, five-, and ten-second rotation successes despite requiring no spin.
- Aggregate speed and stage-valid metrics mixed hold, stationary-handoff, and
  rotation stages, so they could not identify the first rotation stage's
  failing gate.

### Scope

- Changed rotation-duration diagnostics to accumulate only when the active
  target speed is positive. This corrects consecutive-valid, longest-rotation,
  and sustained-duration success metrics without changing reward or promotion.
- Added per-level target-axis EMA, orthogonal-speed EMA, support, retention,
  no-failure, speed, orthogonal, base-valid, completion-ready, and final
  stage-valid metrics. A separate sample fraction identifies the promotion-before
  level population used by the gate metrics; existing post-promotion curriculum
  level fractions retain their historical meaning. Empty levels report zero.
- In the finger-gaiting task, per-level completion readiness reports whether
  the required handoff count has been reached.
- Reward, curriculum, termination, cache reset, observation, action, and
  physical simulation contracts remain unchanged.

### Validation

- Ruff formatted the changed diagnostic code and the focused Ruff check passed.
- The LEAP suite plus the stationary-stage dual-backend config assertion passed
  all 68 tests.
- Explicit Hydra composition preserved the same nine target speeds and resolved
  `LeapInhandBallFingerGaitingRotation` for MuJoCo and Motrix.
- MuJoCo smoke run `2026-07-22_15-45-44_mujoco` completed one iteration with
  16 environments and finite Level 0-8 diagnostics. With every environment in
  Level 0, consecutive-valid time, longest rotation, and all sustained-duration
  success fractions remained zero as intended.
- Motrix smoke run `2026-07-22_15-46-00_motrix` completed the same 114-observation
  runtime contract and emitted finite Level 0-8 gate diagnostics with no false
  rotation duration.

## 2026-07-22 - Interactive Reward Value Telemetry

### Scope

- Added backend-neutral reward telemetry for MuJoCo interactive policy playback.
- Added an opt-in viewer overlay that shows the actual step reward, current
  episode return, each logged `reward/*` value, and its change from the previous
  reward-log sample.
- Kept existing object-rotation telemetry in the top-right while rendering the
  reward table in the top-left; both overlays now share one `set_texts` update.
- Added optional exact reward-term selection and a configurable display limit.
- Added the disabled-by-default interactive settings to PPO, APPO, off-policy,
  and HORA-distill root configs. Training rewards, observations, environments,
  checkpoints, and backend contracts are unchanged.

### Validation

- Ruff formatted the Python changes and the focused Ruff check passed.
- The first pytest invocation could not create its default Windows temp
  directory. Re-running with an isolated `C:\\tmp` base completed 30 focused
  interactive-playback and telemetry tests successfully.
- Hydra composition resolved `LeapInhandBallFingerGaitingRotation` with reward
  telemetry enabled, an exact reward-key selector, and an eight-term limit.
- The installed MuJoCo viewer exposes `Handle.set_texts` with list-of-overlays
  support, allowing reward values and object-rotation telemetry to coexist.

## 2026-07-22 - Direct Fixed-Speed LEAP Ball Rotation Acquisition Task

### Evidence

- Run `2026-07-22_15-47-22_mujoco` completed 100 iterations of the
  finger-gaiting curriculum. It retained the ball for 22.33 seconds on average
  with 10.48 mm mean anchor error, 2.12 fingertip contacts, zero palm-contact
  rate, and a 0.055% termination rate.
- Rotation acquisition still failed: the final target-axis EMA was
  `-0.00584 rad/s`, every sustained-duration success fraction was zero, and no
  environment reached Level 3.
- Level 2 support, retention, no-failure, and orthogonal gates passed at
  83.85%, 79.73%, 99.84%, and 93.20%, respectively, while the positive-speed
  gate passed only 1.97%. Qualified handoffs remained below 0.11%.
- The evidence identifies positive signed rotation, rather than grasp
  retention, as the immediate acquisition bottleneck.

### Scope

- Added `LeapInhandBallDirectRotation` as a separate task; all existing
  sustained and finger-gaiting tasks retain their original defaults.
- Added a default-disabled `direct_target_mode` to the sustained curriculum
  contract. Direct mode accepts exactly one positive target and treats Level 0
  as a rotation stage rather than a hold stage.
- Added a backend-neutral reward-adjustment hook to the sustained environment.
  Existing tasks receive an all-zero adjustment.
- The direct task starts from `ball_grasp_official_50k.npy`, targets world `+Z`
  at `0.30 rad/s`, and keeps the existing 20 Hz control, `kp=3`, `kd=0.1`,
  `1/24` action scale, 7 mm cache-relative drop, 30 mm workspace, and debounced
  palm-contact termination.
- Added direct stable-rotation and streak rewards plus explicit stall, reverse,
  object-center, object-linear-velocity, palm-contact, and orthogonal-speed
  terms. Existing action-rate, torque, work, and failure costs remain active.
- Removed handoff requirements from the learning objective. Natural
  simultaneous release/acquisition events and contact-transition gaps are
  diagnostic-only and are not observations, rewards, success gates, or
  termination conditions.
- Added MuJoCo and Motrix PPO owner configurations, focused reward/config
  tests, and short diagnostic train/view commands.

### Validation

- Ruff formatted the changed Python files and the focused Ruff check passed.
- The first pytest invocation was blocked before collection by the existing
  Windows temp-directory permission error. Re-running with a unique
  `C:\tmp` base and the cache provider disabled succeeded.
- The direct reward/config/runtime tests passed all 5 cases, including both
  backend owners. The complete LEAP environment suite passed all 69 tests.
- Hydra composition resolved both owners to one direct `0.30 rad/s` target,
  `direct_target_mode=true`, the shared `1/24` action scale, and the intended
  stable/stall/reverse coefficients.
- MuJoCo smoke run `2026-07-22_21-25-14_mujoco` completed one iteration with
  16 environments. Actor and critic used 100 observations; all direct reward,
  retention, failure, rotation, and natural-handoff diagnostics were finite.
- Motrix smoke run `2026-07-22_21-25-46_motrix` completed the same 100-observation
  runtime contract with finite direct reward and diagnostic output.
- Both random-policy smoke runs stayed inside the one-second stall grace
  period, so their zero stall term is expected. They validate integration only
  and do not establish task learnability.

## 2026-07-28: English Monthly Progress Presentation Revision

- Rebuilt `LEAP_Hand_Monthly_Progress_Report_English.pptx` as a 16-slide
  English report with a minimum editable text size of 14 pt.
- Corrected the ball-cache generation slide from the current local
  `leap_inhand_ball_grasp` implementation: seeded joint proposals, zero-action
  PD settling, strict physical/contact gates, survivor serialization, and
  reconstruction-time validation before atomic cache storage.
- Replaced the task-evolution diagram with the user-provided reference image
  and rewrote its adjacent explanation in English.
- Reworked later evidence slides around retention, rotation duration, axis
  speed, contact, and handoff diagnostics instead of iteration counts,
  environment steps, or aggregate reward comparisons.
- Rendered and visually inspected all slides. Artifact inspection found 16
  slides, 16 source-note sections, no missing `[Sources]` blocks, and no
  out-of-bounds objects.
- Corrected the typography conversion between Artifact Tool CSS pixels and
  PowerPoint points. The revised `_14pt` deck clamps editable text to 19 px,
  which exports as a verified minimum of 14.25 pt, and uses a wider footer
  frame for two-digit page numbers.
