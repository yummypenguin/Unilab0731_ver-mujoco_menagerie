# LEAP Allegro-equivalent grasp cache generation

`LeapInhandBallGraspAllegro` is a parallel cache-generation task. It preserves
the `AllegroInhandRotationGrasp` physical acceptance lifecycle and adds only a
final settled-row dataset deduplication layer. It does not replace or alter the
strict `LeapInhandBallGrasp` task.

## Nominal state and proposal distribution

The nominal state is the user-provided 23D qpos in
`conf/ppo/task/leap_inhand_ball_grasp_allegro/mujoco.yaml`:

- `[0:16]`: LEAP hand joint qpos
- `[16:19]`: ball world XYZ
- `[19:23]`: ball quaternion WXYZ

The source state's `ctrl`, `qvel`, and `time` are not used. Each reset samples
every hand joint uniformly within ±0.25 rad of the nominal hand qpos and clips
it to the actuator limits. Ball XYZ and the nominal non-identity quaternion are
fixed. Reset qvel is zero, and `prev_ctrl` is the sampled hand qpos.

External policy actions are ignored. Every control step applies zero action, so
the sampled `prev_ctrl` target remains fixed while MuJoCo settles the grasp.

## Physical acceptance lifecycle

The three online conditions are evaluated from the first control step:

1. All four fingertip body origins are strictly less than 0.1061 m from the
   ball center.
2. At least two of the four fingertip contact sensors are active.
3. Ball-center world Z is strictly greater than 0.6576576150346267 m.

Thumb contact is not required, and palm contact is not counted. There is no
warmup. An environment terminates immediately when any condition fails. The
episode timeout is 3.0 seconds, and success is exactly `truncated AND NOT
terminated`.

The generator saves the episode's final settled hand qpos, ball XYZ, and ball
WXYZ quaternion as a 23D float32 row. It does not save control targets,
velocity, time, contact flags, reward, or diagnostics.

This task intentionally does not use surface-gap, penetration, velocity, work,
drift, serialization round-trip, replay-validation, or frontier gates.

## Final-row deduplication

Deduplication occurs only after physical timeout success and float32 conversion.
The key contains 19 quantized values:

- 16 hand joints on a 0.01 rad grid
- ball XYZ on a 0.001 m grid

Ball quaternion is excluded because it changes the beach-ball texture
orientation without changing spherical contact geometry. Duplicate rows do not
enter the cache and do not count toward the 50,000-row target.

The output path is:

`src/unilab/assets/robots/leap_hand/caches/ball_grasp_allegro_dedup_50k.npy`

## Commands to run manually

Codex does not execute cache-generation training commands. The user runs these
steps in order.

### 1. Nominal-seed preflight

```powershell
uv run python scripts/check_leap_allegro_grasp_seed.py
```

### 2. Smoke generation to a separate path

```powershell
uv run train --algo ppo --task leap_inhand_ball_grasp_allegro --sim mujoco training.no_play=true algo.num_envs=64 env.grasp_collection_target=100 env.grasp_cache_path=robots/leap_hand/caches/ball_grasp_allegro_dedup_smoke_100.npy
```

### 3. Inspect the smoke cache

```powershell
uv run python scripts/inspect_leap_allegro_grasp_cache.py --path src/unilab/assets/robots/leap_hand/caches/ball_grasp_allegro_dedup_smoke_100.npy --expected-rows 100
```

### 4. Generate the final 50k cache

```powershell
uv run train --algo ppo --task leap_inhand_ball_grasp_allegro --sim mujoco training.no_play=true
```

### 5. Inspect the final cache

```powershell
uv run python scripts/inspect_leap_allegro_grasp_cache.py --path src/unilab/assets/robots/leap_hand/caches/ball_grasp_allegro_dedup_50k.npy --expected-rows 50000
```
