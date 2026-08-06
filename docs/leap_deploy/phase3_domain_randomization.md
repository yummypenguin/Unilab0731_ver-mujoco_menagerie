# LEAP HORA Phase 3 Domain Randomization

## Scope

This phase adds narrow sim-to-real randomization only to
`LeapInhandBall0730HoraRotation`. The original
`LeapInhandBall0730Rotation`, its reward, cache reset, 30 mm drop termination,
20 Hz control rate, action scale, and 20 second episode duration are unchanged.

## Reset-time physics randomization

Each reset environment is sampled independently. Autoreset samples and applies new
values only for the reset rows.

| Term | Range | Backend target |
|---|---:|---|
| Object mass ratio | `[0.90, 1.10]` | body `leap_object`, nominal mass multiplied by the ratio |
| Object friction scale | `[0.80, 1.20]` | collision geom `leap_object_col`, all three MuJoCo friction components multiplied by the same scale |
| Object COM offset | each axis `[-0.001, 0.001]` m | body `leap_object` inertial position (`body_ipos`) |
| Gravity direction | cone within 3 degrees of `-Z` | per-environment gravity vector with nominal magnitude preserved (9.81 m/s² in the current model) |
| Action delay | 0 or 1 control step | environment-side policy-action queue |

The provider constructs full body/geom tables from cold-path nominal metadata, then
changes only the object body or collision geom entry. Palm and finger masses, COMs,
and non-object friction values remain nominal. It uses UniLab's
`ResetRandomizationPayload` and backend capability contract; it does not modify
MuJoCo model arrays directly.

## Privileged information

`state.info["critic_info"]` remains nine-dimensional and records the values from the
same sample used to build the reset payload:

| Index | Meaning |
|---:|---|
| `0` | object mass ratio |
| `1` | object friction scale |
| `2:5` | object COM offset xyz in metres |
| `5:8` | normalized gravity direction xyz |
| `8` | action delay normalized by `max(action_delay_max_steps, 1)` |

The vector contains ratios and offsets, not absolute mass, friction, or inertial
position. No term is independently resampled while building `critic_info`.

## Joint measurement noise

When randomization is enabled, one uniform measurement error in
`[-0.003, 0.003]` rad is added to the measured joint position used to build the
current actor and proprio frames. The same noisy measurement is used in both
frames. Previous target and rotation-axis channels are unchanged, and normalization
uses `clip=False`.

Noise is observation-only. It is never written to backend state and is not used by
reward, termination, velocity, torque, or pose-difference calculations.

## Action-delay semantics

The queue has shape `[N, 2, 16]` and stores clipped policy actions. For delay zero,
the current action is integrated immediately. For delay one, the previous policy
action is integrated. Reset fills both queue slots with zero, so the first delayed
step leaves the applied target unchanged.

Observation timing remains:

```text
observation_t contains applied_target_(t-1)
policy proposes action_t
queue selects delayed_action_t
applied_target_t = integrate(applied_target_(t-1), delayed_action_t)
observation_(t+1) can contain applied_target_t
```

The queue delays policy actions, never an already-integrated absolute target.

## Nominal mode

Setting `env.hora_domain_rand.enabled=false` disables all six terms regardless of
their individual switches. The resulting privileged vector is exactly:

```text
[1, 1, 0, 0, 0, 0, 0, -1, 0]
```

Joint noise and action delay are zero, no HORA physics payload is produced, and the
Phase 2 observation/action contract is retained.

## Backend capability boundary

The current MuJoCo backend advertises reset support for `body_mass`, `body_ipos`,
`geom_friction`, and `gravity`. If an enabled HORA physics term is unavailable on a
future backend, environment initialization raises `NotImplementedError` naming the
backend, reset term, and related `env.hora_domain_rand` field. Joint noise and action
delay remain environment-side.

## Intentionally excluded

This phase does not add external pushes, initial ball velocity or height noise,
gain randomization, torque-limit randomization, motor-current models, tactile noise,
or camera noise. MuJoCo `kp=3.0` and `kv=0.01` are not randomized and do not
correspond to Dynamixel P/D registers.
