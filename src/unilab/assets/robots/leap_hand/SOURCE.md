# LEAP Hand Asset Provenance

`leap_hand.xml` and `assets/menagerie/*.obj` are based on Google DeepMind's
[`mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie):

- upstream commit: `71f066ad0be9cd271f7ed58c030243ef157af9f4`
- source robot: `leap_hand/right_hand.xml`
- adopted model: right hand
- upstream license: Menagerie LEAP Hand MIT license, preserved in `LICENSE.menagerie`

UniLab adopts the Menagerie body hierarchy, joint axes and ranges, inertias,
visual meshes, primitive collision geometry, collision filtering, contact
solver parameters, friction, and position actuators. Joint and actuator names
use Menagerie's semantic names. Selected body names and fingertip collision
geom names are retained for compatibility with UniLab task and sensor
contracts. The palm root remains at world `z=0.5` so existing scene, object,
camera, keyframe, and cache coordinate layouts remain unchanged.

The only intentional robot-physics deviation from the pinned Menagerie source
is joint `armature=0.01`. This value is present directly in `leap_hand.xml` so
direct XML consumers such as MuJoCo viewer and the pose editor see the same
dynamics as training. MuJoCo LEAP owners intentionally do not expose joint
`damping`, `frictionloss`, `armature`, actuator `kp`, or actuator `kv` in task
YAML. These are model-owned constants in `leap_hand.xml`: `0.03`, `0.001`,
`0.01`, `3.0`, and `0.01`, respectively. Runtime gain overrides for a LEAP
MuJoCo task fail closed instead of replacing the model values.

The historical LEAP Hand Sim URDF/STL sources remain under `source/` and the
legacy `assets/*.stl` files are retained until no repository consumer refers to
them. Their MIT license remains in `LICENSE.txt`.

Menagerie does not provide UniLab's manipulation objects or task state. The
dedicated `cube.xml`, `ball.xml`, textures, scenes, and caches therefore remain
UniLab-owned assets with unchanged object dimensions, masses, freejoint names,
and 23-value hand/object qpos layout.

`caches/cube_grasp_s10_1k.npy` is converted from the source scale-1.0 cube grasp cache.
Its hand joints are reordered to the MuJoCo qpos order in `leap_hand.xml`, and object
quaternions are converted from Isaac Gym `xyzw` to MuJoCo `wxyz`.

Caches generated before the Menagerie migration retain their shape and qpos
ordering but are not physics-valid training data for the migrated model. They
may only be used as proposals for re-settling, filtering, deduplication, and
validation.
