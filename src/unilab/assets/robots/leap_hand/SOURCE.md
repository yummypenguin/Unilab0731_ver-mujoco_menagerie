# LEAP Hand Asset Provenance

The robot URDF, STL meshes, and reference object URDFs in this directory are derived from
`leap-hand/LEAP_Hand_Sim`, copyright 2023 Ananye Agarwal, under the MIT License.
The complete license text is preserved in `LICENSE.txt`.

UniLab keeps the original URDF files under `source/`. `leap_hand.xml` is a MuJoCo conversion
with a fixed base, position actuators, stable names, and simplified primitive collision geometry.
The dedicated `cube.xml` and `ball.xml` assets use the source dimensions and mass without
depending on object assets owned by another UniLab robot.

`caches/cube_grasp_s10_1k.npy` is converted from the source scale-1.0 cube grasp cache.
Its hand joints are reordered to the MuJoCo qpos order in `leap_hand.xml`, and object
quaternions are converted from Isaac Gym `xyzw` to MuJoCo `wxyz`.
