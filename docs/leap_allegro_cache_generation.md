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

1. All four fingertip body origins are strictly less than 0.1 m from the
   ball center.
2. At least two of the four fingertip contact sensors are active.
3. Ball-center world Z is strictly greater than that reset proposal's initial
   ball Z minus 0.005 m. With the current nominal pose this threshold is
   `0.664301098275159 - 0.005 = 0.659301098275159 m`.

After timeout success, all four fingertip collision surfaces must also be
strictly less than 0.005 m from the ball collision surface. A gap equal to
0.005 m is rejected. This final-row gate uses signed MuJoCo geom distances for
`index_tip_col`, `middle_tip_col`, `ring_tip_col`, and `thumb_tip_col` against
`leap_object_col`.

The online backend state is checked first for efficiency. After the accepted
state is serialized to its exact 23D float32 cache row, MuJoCo runs static FK on
that row and repeats the same strict 5 mm check before deduplication. This makes
the generator publication gate operate on the same representation reloaded by
the inspector.

Thumb contact is not required, and palm contact is not counted. There is no
warmup. An environment terminates immediately when any condition fails. The
episode timeout is 2.5 seconds (50 control steps at `ctrl_dt = 0.05` seconds),
and success is exactly `truncated AND NOT terminated`.

The generator saves the episode's final settled hand qpos, ball XYZ, and ball
WXYZ quaternion as a 23D float32 row. It does not save control targets,
velocity, time, contact flags, reward, or diagnostics.

This task intentionally does not use surface-gap, penetration, velocity, work,
drift, serialization round-trip, replay-validation, or frontier gates.

## Final-row deduplication

Deduplication occurs only after physical timeout success and float32 conversion.
The key contains 19 quantized values:

- 16 hand joints on a 0.001 rad grid
- ball XYZ on a 0.0005 m grid

Ball quaternion is excluded because it changes the beach-ball texture
orientation without changing spherical contact geometry. Duplicate rows do not
enter the cache and do not count toward the 50,000-row target.

The post-generation inspector applies the same nominal-height dataset gate.
Rows with ball Z less than or equal to `0.659301098275159 m` are rejected. This
is a static cache validation step; physics replay validation is not required.
It also runs static `mj_forward` for each row and enforces the same strict
5 mm fingertip-surface gap without advancing simulation time.

The output path is:

`src/unilab/assets/robots/leap_hand/caches/ball_grasp_allegro_new_physics_0731_50k.npy`

## Commands to run manually

Codex does not execute cache-generation training commands. The user runs these
steps in order.

### 1. Nominal-seed preflight

```powershell
uv run python scripts/check_leap_allegro_grasp_seed.py
```

### 2. Smoke generation to a separate path

```powershell
uv run train --algo ppo --task leap_inhand_ball_grasp_allegro --sim mujoco training.no_play=true algo.num_envs=64 env.grasp_collection_target=100 env.grasp_cache_path=robots/leap_hand/caches/ball_grasp_allegro_new_physics_0731_smoke_100.npy
```

### 3. Inspect the smoke cache

```powershell
uv run python scripts/inspect_leap_allegro_grasp_cache.py --path src/unilab/assets/robots/leap_hand/caches/ball_grasp_allegro_new_physics_0731_smoke_100.npy --expected-rows 100
```

### 4. Generate the final 50k cache

```powershell
uv run train --algo ppo --task leap_inhand_ball_grasp_allegro --sim mujoco training.no_play=true
```

### 5. Inspect the final cache

```powershell
uv run python scripts/inspect_leap_allegro_grasp_cache.py --path src/unilab/assets/robots/leap_hand/caches/ball_grasp_allegro_new_physics_0731_50k.npy --expected-rows 50000
```
