# MetaWorld demonstration contract (extracted from the released code)

Source of truth, read-only:
`third_party/Metaworld/gen_demonstration_expert.py` and
`FlowPolicy/flow_policy_3d/env/metaworld/metaworld_wrapper.py`.
This note records exactly what a recorder must produce so our dataset and model see
the same observation as the paper's code. It is a **transcription**, not a redesign:
every number below is quoted from the released wrapper.

## Environment

| Item | Released value |
| --- | --- |
| Env class | `metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task + '-v2-goal-observable']()` |
| `_freeze_rand_vec` | `False` (random goal each reset) |
| Camera position override | `sim.model.cam_pos[2] = [0.6, 0.295, 0.8]` |
| Clip planes | `znear = 0.1`, `zfar = 1.5` |
| Render camera | `corner2` |
| Image size | `128 x 128` RGB |
| Episode length | `200` steps |
| Action dim | `4` |

## Robot state (`agent_pos`), 9-dim

```python
eef_pos                           = env.get_endeff_pos()                  # 3
finger_right, finger_left         = env._get_site_pos('rightEndEffector'),
                                    env._get_site_pos('leftEndEffector')   # 3 + 3
robot_state = np.concatenate([eef_pos, finger_right, finger_left])
```

## Point cloud, exactly 1024 XYZ points

1. Render depth from camera `corner2` at 128x128 (`sim.render(..., depth=True)`).
2. Unproject each pixel to metres with the camera intrinsics from the MuJoCo model
   (`mujoco_point_cloud.PointCloudGenerator`), giving a raw `N x 3` cloud.
3. Apply the fixed orientation transform (two rotations, degrees):

   ```python
   x_angle, y_angle = 61.4, -7
   pc_transform = Rx(x_angle) @ Ry(y_angle)
   point_cloud = point_cloud @ pc_transform.T
   ```

4. Scale by `[1, 1, 1]` and shift by `[0, 0, 0]` (identity in the released run).
5. Crop with the released bounds `TASK_BOUDNS = {'default': [-0.5, -1.5, -0.795, 1, -0.4, 100]}`
   interpreted as `x_min, y_min, z_min, x_max, y_max, z_max`, keeping points that satisfy
   **all** `> min` and **all** `< max`.
6. Farthest-point-sample the survivors to `num_points` (`'fps'` strategy; the wrapper
   defaults to `1024`, the task YAMLs use `512` — see DECISIONS D004).

The released `depth` field in the observation is the depth image flipped vertically
(`depth[::-1]`); it is stored in the Zarr file but not used by the policy.

## Episode filtering

An episode is kept only when it succeeds and succeeds **at least 5 times**:

```python
if not ep_success or ep_success_times < 5:   # drop the whole episode
```

Accepted episodes are concatenated and `meta/episode_ends` holds the **cumulative**
frame counts (`np.cumsum(lengths)`), which is the exclusive-end convention our dataset
uses (`data/dataset.py`).

## Expert action source

`metaworld.policies` scripted state machines; `load_mw_policy(task_name)` maps
`'peg-insert-side'` to `SawyerPegInsertionSideV2Policy` and every other task to
`Sawyer<Capitalised><Parts>V2Policy`. `action = policy.get_action(full_state)` where
`full_state` is the env's 20-dim `obs` vector, i.e. the scripted expert needs no
training.

## What this means for our reproduction

- Rendering and depth unprojection are the only simulator-specific steps; the policy
  itself never sees an image, only `point_cloud` plus the 9-dim robot state.
- If the legacy MuJoCo stack cannot be built here, the honest substitute is: same
  State/Action/PointCloud contract, same crop bounds and same episode filter, but a
  newer MuJoCo/MetaWorld API — and the substitution must be stated in
  `docs/DECISIONS.md` together with the difference (env version, camera matrix source).

## What we actually implemented (see DECISIONS D016)

The legacy stack was **not** needed: `mujoco 3.3.7` + `metaworld 3.0.0` install and render
headless here. The following parts of the released recipe are kept verbatim:

- camera `corner2`, the `cam_pos = [0.6, 0.295, 0.8]` override, `znear = 0.1` / `zfar = 1.5`,
  a 128x128 depth render, the fixed `Rx(61.4 deg) @ Ry(-7 deg)` orientation,
  farthest-point sampling to 1024 points, the 9-dim `agent_pos`, the scripted expert,
  the "keep an episode only if it succeeds at least 5 times" filter and the `data/` + `meta/`
  Zarr layout.

The following parts **had to change**, and the change is disclosed rather than hidden:

| Released behaviour | Ours | Why |
| --- | --- | --- |
| Depth from `mujoco-py` normalised by `near/(1-d(1-near/far))`, then `verticalFlip` | `mujoco.Renderer` depth used directly (already metric and upright) | Re-applying the old conversion would corrupt the cloud |
| Crop box `(-0.5, -1.5, -0.795, 1, -0.4, 100)` in the rotated frame | Crop box in **world** coordinates `(-0.35, 0.25, -0.05) .. (0.35, 0.95, 0.55)`, crop before the fixed rotation | The released box keeps ~0-100 of 16384 rendered pixels on this stack, i.e. it would delete the scene |
| `metaworld 2.0.0` env classes, `Sawyer*V2Policy` | `metaworld 3.0.0` `MT1(task)`, `Sawyer*V3Policy` | v2 classes and policies no longer ship |

Geometric validation of the new mapping (not a guess): after `cam_xmat`/`cam_xpos`
transformation, 1173 of the 16384 camera points land on the floor plane `z = -0.913` and
5233 on the table top `z = 0`, which matches the model geometry; the workspace box then
keeps 28.5% of the points.

Consequence for claims: field names, shapes, dimensions, success filtering and on-disk
layout match; the per-pixel clouds cannot match the paper's (different MuJoCo version,
different depth convention, re-derived crop box). Results must therefore be described as
"reproduced on a substituted simulator".

## Task availability probe (FPJ-027, `metaworld 3.0.0`)

`metaworld.MT1(task)` + the matching scripted expert, one 200-step episode, seed 0,
`MUJOCO_GL=egl`:

| Task | Expert class | Successes / 200 steps |
| --- | --- | --- |
| `reach-v3` | `SawyerReachV3Policy` | 150 |
| `button-press-wall-v3` | `SawyerButtonPressWallV3Policy` | 141 |
| `reach-wall-v3` | `SawyerReachWallV3Policy` | 159 |
| `push-v3` | `SawyerPushV3Policy` | 138 |
| `drawer-open-v3` | `SawyerDrawerOpenV3Policy` | 114 |
| `door-open-v3` | `SawyerDoorOpenV3Policy` | 30 |

So the plan's primary task (`button-press-wall`) and its stability task (`reach-wall`) both
exist on the modern stack with working experts; `door-open-v3` is a poor demonstration
source on this stack (30/200) and should not be used for the formal comparison.

Import note: the experts live in the `metaworld.policies` package (`from metaworld import
policies as ...; policies.SawyerReachV3Policy`), not in `metaworld.policies.policy`.

