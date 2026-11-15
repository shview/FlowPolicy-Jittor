# MetaWorld sim environment setup log

Date: 2026-11-14
Host: WSL2 Ubuntu 24.04.3 LTS, 24 cores, 7 GB RAM, no GPU (Mesa **llvmpipe** software rendering)
Scope: build an isolated Python env that runs MetaWorld and renders RGB + depth offscreen, then prove it.

---

## 1. Which path worked

**The modern stack worked on the first attempt. The legacy stack was never needed.**

| Item | Value |
| --- | --- |
| Path | **modern** (`mujoco` + `metaworld` 3.x + `gymnasium`) |
| New isolated env path | `/home/s/.local/share/flowpolicy/sim-env` |
| Python | **3.10.14** (venv created from `miniforge-py310`, `include-system-site-packages = false`) |
| Offscreen renderer | `MUJOCO_GL=egl` |
| Working GL backend | Mesa **llvmpipe** (software), EGL 4.5 |

Isolation is proven, not assumed:

- `/home/s/.local/share/flowpolicy/sim-env` did not exist before this run and was created new.
- `/home/s/.local/share/flowpolicy/miniforge-py310` (the Jittor env) was used **read-only, as the venv base
  interpreter only**. Nothing was installed into it: `find … -newermt "-3 hours"` returns no modified
  entries, and it contains no `mujoco`/`metaworld`/`gymnasium` packages.
- `sim-env/pyvenv.cfg` shows `include-system-site-packages = false`, so the two envs share no packages.
- Nothing was written inside `flowpolicy-upstream/`; that tree was only read
  (`gym_util/mujoco_point_cloud.py`) to check a depth-convention question — see §6.

---

## 2. Exact commands used, in order

```bash
BASE=/home/s/.local/share/flowpolicy/miniforge-py310/bin/python
SIM=/home/s/.local/share/flowpolicy/sim-env

# 0. confirm the base interpreter (read-only use of the Jittor env)
$BASE -V                                  # Python 3.10.14

# 1. create the NEW isolated venv (the Jittor env is the base only, never modified)
$BASE -m venv $SIM
$SIM/bin/python -V                        # Python 3.10.14
$SIM/bin/python -c "import sys; print(sys.executable, sys.prefix)"

# 2. upgrade packaging tools inside the venv only
$SIM/bin/python -m pip install --upgrade pip setuptools wheel

# 3. install the stack (this is the only install step)
$SIM/bin/python -m pip install \
  "mujoco==3.3.7" "metaworld==3.0.0" "numpy==1.26.4" \
  "imageio==2.37.0" "imageio-ffmpeg==0.6.0" "pillow==11.3.0"

# 4. install the probe and run it (EGL first, as instructed)
cp sim_probe.py /tmp/sim_probe.py
MUJOCO_GL=egl                     $SIM/bin/python /tmp/sim_probe.py   # SUCCESS
MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 $SIM/bin/python /tmp/sim_probe.py # SUCCESS (forced llvmpipe)
MUJOCO_GL=osmesa                  $SIM/bin/python /tmp/sim_probe.py   # fails: no libOSMesa on host (not needed)
```

No system packages, no compilers, no `patchelf`, no MuJoCo binaries, no GPU drivers were installed or
needed. `mujoco==3.3.7` ships self-contained manylinux wheels with EGL support built in.

### Final working versions (`pip freeze`, filtered)

```
glfw==2.11.0
gymnasium==1.4.0
imageio==2.37.0
imageio-ffmpeg==0.6.0
metaworld==3.0.0
mujoco==3.3.7
numpy==1.26.4
pillow==11.3.0
PyOpenGL==3.1.10
scipy==1.15.3
```

Full freeze in the venv (for exact reproduction):

```
absl-py==2.5.1
cloudpickle==3.1.2
etils==1.13.0
Farama-Notifications==0.0.6
fsspec==2026.9.0
glfw==2.11.0
gymnasium==1.4.0
imageio==2.37.0
imageio-ffmpeg==0.6.0
importlib_resources==7.1.0
metaworld==3.0.0
mujoco==3.3.7
numpy==1.26.4
packaging==26.3
pillow==11.3.0
PyOpenGL==3.1.10
scipy==1.15.3
typing_extensions==4.16.0
zipp==4.1.2
```

Note: `metaworld==3.0.0` has **no `__version__` attribute** (version comes from the installed dist).
There is also **no `metaworld.make(...)`**. Per-task envs are reached via the gymnasium registry:
`metaworld.register_mw_envs()` then `gym.make("Meta-World/MT1", env_name="reach-v3")`
(`metaworld.ALL_V3_ENVIRONMENTS['reach-v3']` exists but is an env *class* whose direct construction
asserts until a task is set — use `MT1`).

---

## 3. Probe output block

Command: `MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 /home/s/.local/share/flowpolicy/sim-env/bin/python /tmp/sim_probe.py`

```
task            : reach-v3 (SawyerReachEnvV3)
obs keys        : ['<flat vector>', 'named:state_achieved_goal', 'named:state_desired_goal', 'named:state_observation']
  obs[<flat vector>         ] shape=(39,) dtype=float64
  obs[named:state_achieved_goal] shape=(33,) dtype=float64
  obs[named:state_desired_goal] shape=(3,) dtype=float64
  obs[named:state_observation] shape=(39,) dtype=float64
action shape    : (4,)
image           : shape=(480, 480, 3) dtype=uint8 min=0 max=255 std=56.8725
depth range (m) : 1.106556 .. 6.822839 (median 1.95526, valid 144803/230400)
point cloud     : shape=(1024, 3) (metres, world frame)
point cloud xyz : min=[-3.9979, -0.5998, -0.913] max=[0.8243, 3.984, 0.6863] z_median=-0.6
landmark check  : floor plane z=-0.913 (from model), 41.5% of points land on it
MUJOCO_GL       : egl
GL renderer     : {'GL_VENDOR': 'Mesa', 'GL_RENDERER': 'llvmpipe (LLVM 20.1.2, 256 bits)',
                   'GL_VERSION': '4.5 (Compatibility Profile) Mesa 25.2.8-0ubuntu0.24.04.2'}
versions        : {'python': '3.10.14', 'mujoco': '3.3.7', 'mujoco(dist)': '3.3.7',
                   'metaworld': '3.0.0', 'gymnasium': '1.4.0', 'numpy': '1.26.4',
                   'imageio': '2.37.0', 'pillow': '11.3.0', 'scipy': '1.15.3'}
----------------------------------------------------------------------
[PASS] rgb_file_nonzero_bytes
[PASS] rgb_has_variance
[PASS] rgb_not_blank
[PASS] points_shape_1024x3
[PASS] points_dtype_float32
[PASS] points_finite
[PASS] points_nontrivial_extent
[PASS] depth_finite_nonzero
[PASS] points_metric_scale
[PASS] points_in_front_of_camera
[PASS] reconstruction_hits_floor_plane
======================================================================
PROBE RESULT: SUCCESS
======================================================================
```

Selected machine-readable fields:

| Field | Value |
| --- | --- |
| `env_id` / `env_name` | `Meta-World/MT1` / `reach-v3` |
| `camera` | `corner2` (model also has `topview`, `corner`, `corner3`, `behindGripper`, `gripperPOV`) |
| `action_space` | `Box(-1.0, 1.0, (4,), float32)` |
| `steps` / `sum_reward` | 5 small actions / 6.429 |
| `image_shape` / `dtype` | `(480, 480, 3)` / `uint8` |
| `depth_shape` / `dtype` | `(480, 480)` / `float32`, **metric metres** |
| `near_far_metres` | `[0.022579, 112.893347]` (default `vis.map.znear=0.01`, `zfar=50`, `stat.extent=2.258`) |
| `fovy_deg` | `60.0` |
| `cam_pos` | `[1.3, -0.2, 1.1]` |
| `points_shape` / `dtype` | `(1024, 3)` / `float32` |
| `points_dist_to_cam_median` | `2.1825` m |
| output files | `/tmp/sim_rgb.png` (176 181 B), `/tmp/sim_points.npy` (12 416 B) |

### Independent artefact verification (beyond the probe's own checks)

- `sim_rgb.png`: 480×480×3 uint8, per-channel std 50.6 / 58.3 / 60.2, **6 511 unique colours**, not blank.
  Row-brightness profile and an ASCII thumbnail show real scene structure (arm/table in the upper two
  thirds, flat background ≈ RGB 125 in the bottom third) rather than noise.
- The flat background rows agree with the depth buffer: those pixels sit at the far plane
  (112.897 m), i.e. RGB and depth describe the same scene.
- `sim_points.npy`: `(1024, 3)` float32, all finite, no duplicate rows, extent ≈ 4.8 m — metre scale.
- **Geometric ground truth check:** MetaWorld's scene contains a MuJoCo floor `PLANE` at
  `z = -0.913` and a table top at `z = 0.0`. The reconstructed cloud contains **425 points within 2 cm
  of z = -0.913** and **247 within 2 cm of z = 0.0**. The unprojection is therefore correct in metres,
  not merely plausible.

---

## 4. Offscreen renderer setting

**`MUJOCO_GL=egl`** is the setting that works. No X server, no `xvfb`, no `PYOPENGL_PLATFORM` override,
and no display dependency is required.

```bash
MUJOCO_GL=egl /home/s/.local/share/flowpolicy/sim-env/bin/python /tmp/sim_probe.py
# optional, to pin software rasterisation explicitly:
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl /home/s/.local/share/flowpolicy/sim-env/bin/python /tmp/sim_probe.py
```

Both variants pass. The GL strings confirm the required software path:

```
GL_VENDOR   = Mesa
GL_RENDERER = llvmpipe (LLVM 20.1.2, 256 bits)
GL_VERSION  = 4.5 (Compatibility Profile) Mesa 25.2.8-0ubuntu0.24.04.2
```

Backend results:

| `MUJOCO_GL` | Result | Notes |
| --- | --- | --- |
| `egl` | **works** | default choice; uses host libEGL (Mesa) → llvmpipe |
| `egl` + `LIBGL_ALWAYS_SOFTWARE=1` | **works** | same output, explicit llvmpipe |
| `glfw` | works | only because WSLg provides `DISPLAY=:0`; **do not depend on it** |
| `osmesa` | fails | host has no `libOSMesa`; `AttributeError: 'NoneType' object has no attribute 'glGetError'`. Not needed. |

Required env vars: only `MUJOCO_GL=egl`. Recommended for reproducibility:
`LIBGL_ALWAYS_SOFTWARE=1`. Do **not** set `DISPLAY`, and do not rely on WSLg.
(The probe reports `PYOPENGL_PLATFORM=egl` — it is set internally by MuJoCo's EGL backend, not by us.)

Runtime is comfortable: a 480×480 RGB + depth pair via llvmpipe takes well under a second.

---

## 5. Expert / scripted policy availability

**Yes — scripted experts are available and they actually succeed.** `metaworld==3.0.0` ships
`metaworld.policies` with **50 scripted state-machine policy classes** (53 files), one per v3 task,
each exposing `get_action(obs)`. No training and no checkpoint files are needed.

- Reach-related: `SawyerReachV3Policy`, `SawyerReachWallV3Policy`
  (file `metaworld/policies/sawyer_reach_v3_policy.py`).
- Naming convention: `Sawyer<CamelTask>V3Policy` (e.g. `SawyerPegInsertionSideV3Policy`,
  `SawyerDoorOpenV3Policy`, `SawyerBoxCloseV3Policy`, …).
- Usage: `from metaworld.policies import SawyerReachV3Policy; a = SawyerReachV3Policy().get_action(obs)`
  where `obs` is the env observation vector (39-dim for reach).

Verified live on `reach-v3` (same env, seed 0, driven by the scripted expert):

```
scripted policy: success=True at step=56 reward=10.000
```

So successful demonstration generation is possible: **scripted policy → env.step → record**.

Where the success signal lives (checked directly, this is what a recorder must read):

- `env.reset(...)` returns `info = {}` — the reset dict is **empty**, do not look for `success` there.
- `env.step(...)` returns a populated `info` with exactly these keys:
  `['success', 'near_object', 'grasp_success', 'grasp_reward', 'in_place_reward', 'obj_to_target', 'unscaled_reward']`.
- `success` is a **float** `0.0` / `1.0`, not a bool, so compare with `info["success"] > 0.5`.
- There is **no** `env.unwrapped._get_info()` on `SawyerReachEnvV3`; read the dict returned by `step`.

**Caveats vs the released FlowPolicy code:**

| Released code expects | This env provides |
| --- | --- |
| `metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[...]` | **absent** (`metaworld.ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE` exists instead, keys like `reach-v3-goal-observable`) |
| `SawyerReachV2Policy` / `load_mw_policy('reach')` → V2 class | **absent** — only `SawyerReachV3Policy` (50 V3 policies) |
| `gym==0.21.0` env API | `gymnasium==1.4.0` API (`reset() -> (obs, info)`, `step() -> 5-tuple`) |

The scripted-expert *capability* is present; only the V2 class names/IDs are gone.

---

## 6. Notes, hazards, and what to watch when reusing this env

Nothing failed, so there is no error text to report. Two integration hazards were found while
validating the probe against the released pipeline (documented here so they are not rediscovered later):

1. **Depth is already metric — do not convert it twice.** Modern `mujoco.Renderer(...)` with
   `enable_depth_rendering()` returns **metres** directly (it performs the reverse-Z → metric
   conversion internally). The released FlowPolicy helper
   (`flow_policy_3d/gym_util/mujoco_point_cloud.py`) instead receives *normalised* `[0,1]` depth from
   mujoco-py `sim.render(..., depth=True)` and applies its own
   `near / (1 - depth * (1 - near/far))`, then a `verticalFlip`. Re-applying that formula to a modern
   depth buffer will silently corrupt the cloud. Also, modern depth is **already upright**: the
   renderer applies `np.flipud` itself for EGL/OSMesa/GLFW, so no extra flip is needed.

2. **The released 1024-point recipe does not transfer as-is to v3.** Reproducing the released contract
   (`cam_pos = [0.6, 0.295, 0.8]`, `znear=0.1`, `zfar=1.5`, `corner2`, 128×128, then
   `Rx(61.4) @ Ry(-7)`, crop `x∈(-0.5,1) y∈(-1.5,-0.4) z∈(-0.795,100)`) renders fine
   (RGB 128×128 OK; depth 0.608…3.387 m, correctly clipped by the released far plane), but the released
   crop bounds then keep essentially nothing: **2 / 16384** points with the natural axis convention
   (best alternative axis variant: 102 / 16384). Rendering and geometry are correct — the *crop bounds*
   and/or the camera-frame sign convention are tied to the metaworld **2.0.0 / mujoco-py 2.1** model and
   depth convention, so they must be re-derived for v3 rather than copied. Treat this as a dataset-side
   task; the simulator side is proven working here.

Point-cloud unprojection used (and validated): intrinsics from `cam_fovy` (square pixels,
`fx = fy = 0.5*H/tan(fovy/2)`, principal point at the image centre), camera frame
`+X right, +Y up, looks down −Z`, then `world = cam_xpos + cam_xmat @ [x, −y, −depth]`.
`cam_xmat` is camera→world: `R @ [0,0,−1]` reproduces `mjvGLCamera.forward` exactly.

Fallback path status: **not attempted and not needed** (modern stack rendered on the first try).
If it is ever required, no MuJoCo 2.1 binaries or `patchelf`/`gcc` build steps exist in the current
env — a fresh Python 3.8 env would be needed, and `sim-env` must not be reused for it.

---

## 7. Probe script (`/tmp/sim_probe.py`)

Kept verbatim below so the proof can be re-run at any time. It is self-verifying: it exits non-zero if
any check fails.

```python
#!/usr/bin/env python
"""MetaWorld modern-stack offscreen probe: RGB frame + 1024-point depth cloud.

Run inside the isolated venv:
    MUJOCO_GL=egl /home/s/.local/share/flowpolicy/sim-env/bin/python /tmp/sim_probe.py

Writes /tmp/sim_rgb.png and /tmp/sim_points.npy and prints a machine-checkable
report. No display is required; rendering goes through MuJoCo's EGL/OSMesa
offscreen context.
"""
import json
import os
import platform
import sys
import warnings

warnings.filterwarnings("ignore")

MUJOCO_GL = os.environ.get("MUJOCO_GL", "<unset>")
H = W = 480
N_POINTS = 1024
N_STEPS = 5
CAMERA = "corner2"
ENV_NAME = "reach-v3"


def resample_to_n(points, n, rng):
    """Return exactly n rows from `points` (subsample, or tile if too few)."""
    if points.shape[0] >= n:
        idx = rng.choice(points.shape[0], size=n, replace=False)
    else:
        idx = rng.choice(points.shape[0], size=n, replace=True)
    return points[idx]


def main():
    import numpy as np
    import gymnasium as gym

    import mujoco
    import metaworld
    import imageio.v2 as imageio

    rng = np.random.default_rng(0)

    # ---- create task ------------------------------------------------------
    metaworld.register_mw_envs()
    env = gym.make("Meta-World/MT1", env_name=ENV_NAME)
    raw = env.unwrapped  # SawyerReachEnvV3, exposes mujoco model/data
    model, data = raw.model, raw.data
    task_name = f"{ENV_NAME} ({type(raw).__name__})"

    # ---- reset + a few small actions -------------------------------------
    obs, info = env.reset(seed=0)
    action_space = env.action_space
    nu = action_space.shape[0]
    act = np.zeros(nu, dtype=np.float64)
    act[0] = 0.01  # small, deterministic nudges (no physics blow-up)
    act[1] = -0.01
    act[2] = 0.01
    total_reward = 0.0
    for _ in range(N_STEPS):
        obs, reward, terminated, truncated, info = env.step(act)
        total_reward += float(reward)
        if terminated or truncated:
            break
    mujoco.mj_forward(model, data)  # make sure kinematics are current

    # MT1 returns a flat concatenated observation; also expose named components.
    def _obs_report(o, env_unwrapped):
        if isinstance(o, dict):
            return {k: tuple(np.asarray(v).shape) for k, v in sorted(o.items())}
        shapes = {"<flat vector>": (tuple(np.asarray(o).shape), str(np.asarray(o).dtype))}
        try:
            d = env_unwrapped._get_obs_dict()
            for k, v in sorted(d.items()):
                shapes[f"named:{k}"] = (tuple(np.asarray(v).shape), str(np.asarray(v).dtype))
        except Exception:
            pass
        return shapes

    obs_shapes = _obs_report(obs, raw)

    # ---- RGB offscreen render --------------------------------------------
    renderer = mujoco.Renderer(model, height=H, width=W)
    renderer.update_scene(data, camera=CAMERA)
    rgb = renderer.render().copy()

    # Which GL implementation is actually doing the rasterising?
    gl_strings = {}
    try:
        from OpenGL import GL as _GL

        gl_strings = {
            "GL_VENDOR": _GL.glGetString(_GL.GL_VENDOR).decode(),
            "GL_RENDERER": _GL.glGetString(_GL.GL_RENDERER).decode(),
            "GL_VERSION": _GL.glGetString(_GL.GL_VERSION).decode(),
        }
    except Exception as e:  # non-fatal, informational only
        gl_strings = {"error": f"{type(e).__name__}: {e}"}

    # ---- depth offscreen render (mujoco returns METRIC depth, metres) -----
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.close()

    imageio.imwrite("/tmp/sim_rgb.png", rgb)

    # ---- depth -> 3D point cloud (metres) --------------------------------
    cam_id = model.cam(CAMERA).id
    cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64).copy()
    cam_mat = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    fovy = float(model.cam_fovy[cam_id])  # vertical FOV, degrees
    fy = 0.5 * H / np.tan(np.deg2rad(fovy) * 0.5)
    fx = fy  # square pixels
    cx, cy = W * 0.5, H * 0.5

    extent = float(model.stat.extent)
    zfar = float(model.vis.map.zfar) * extent
    znear = float(model.vis.map.znear) * extent

    valid = np.isfinite(depth) & (depth > 0.0) & (depth < zfar * 0.999)
    n_valid = int(valid.sum())

    vs, us = np.nonzero(valid)  # row (v, top-down) and column (u, left-right)
    d_flat = depth[vs, us].astype(np.float64)
    # MuJoCo/OpenGL camera frame: looks down -Z, +X right, +Y up.
    x_cam = (us.astype(np.float64) + 0.5 - cx) / fx * d_flat
    y_cam = -(vs.astype(np.float64) + 0.5 - cy) / fy * d_flat
    z_cam = -d_flat
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)      # (N,3)
    points = pts_cam @ cam_mat.T + cam_pos                  # (N,3) world, metres
    points = points.astype(np.float64)

    points = resample_to_n(points, N_POINTS, rng)
    np.save("/tmp/sim_points.npy", points.astype(np.float32))

    # ---- independent landmark check (uses model geometry, not the math) ---
    # MetaWorld scenes contain a MuJoCo floor PLANE; if the reconstruction is
    # correct, some points must land on that plane's z.
    floor_z = None
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            floor_z = float(data.geom_xpos[i][2])
            break
    frac_near_floor = (
        float(np.mean(np.abs(points[:, 2] - floor_z) < 0.02)) if floor_z is not None else 0.0
    )
    cam_dists = np.linalg.norm(points - cam_pos, axis=1)

    # ---- report -----------------------------------------------------------
    import PIL
    import scipy
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    def _ver(name):
        try:
            return _pkg_version(name)
        except PackageNotFoundError:
            return "not installed"

    versions = {
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "mujoco(dist)": _ver("mujoco"),
        "metaworld": getattr(metaworld, "__version__", _ver("metaworld")),
        "gymnasium": gym.__version__,
        "numpy": np.__version__,
        "imageio": _ver("imageio"),
        "pillow": PIL.__version__,
        "scipy": scipy.__version__,
    }

    report = {
        "task": task_name,
        "env_id": "Meta-World/MT1",
        "env_name": ENV_NAME,
        "camera": CAMERA,
        "obs_keys": list(obs_shapes.keys()),
        "obs_shapes": obs_shapes,
        "action_shape": tuple(action_space.shape),
        "action_space": f"Box({action_space.low.min()}, {action_space.high.max()})",
        "steps": N_STEPS,
        "sum_reward": round(total_reward, 6),
        "image_shape": tuple(rgb.shape),
        "image_dtype": str(rgb.dtype),
        "image_min": int(rgb.min()),
        "image_max": int(rgb.max()),
        "image_std": round(float(rgb.astype(np.float64).std()), 4),
        "depth_shape": tuple(depth.shape),
        "depth_dtype": str(depth.dtype),
        "depth_min": round(float(depth[valid].min()), 6),
        "depth_max": round(float(depth[valid].max()), 6),
        "depth_median": round(float(np.median(depth[valid])), 6),
        "valid_depth_pixels": n_valid,
        "total_pixels": int(depth.size),
        "near_far_metres": [round(znear, 6), round(zfar, 6)],
        "fovy_deg": round(fovy, 4),
        "cam_pos": [round(float(v), 4) for v in cam_pos],
        "points_shape": tuple(points.shape),
        "points_dtype_expected": "float32 saved",
        "points_min": [round(float(v), 4) for v in points.min(axis=0)],
        "points_max": [round(float(v), 4) for v in points.max(axis=0)],
        "points_mean": [round(float(v), 4) for v in points.mean(axis=0)],
        "points_z_median": round(float(np.median(points[:, 2])), 4),
        "points_z_p05_p95": [
            round(float(np.percentile(points[:, 2], 5)), 4),
            round(float(np.percentile(points[:, 2], 95)), 4),
        ],
        "points_dist_to_cam_median": round(float(np.median(cam_dists)), 4),
        "floor_plane_z_ground_truth": floor_z,
        "frac_points_on_floor_plane": round(frac_near_floor, 4),
        "mujoco_gl": MUJOCO_GL,
        "renderer_gl": "MuJoCo EGL/OSMesa offscreen",
        "gl_strings": gl_strings,
        "env_vars": {
            k: os.environ.get(k)
            for k in ("MUJOCO_GL", "LIBGL_ALWAYS_SOFTWARE", "DISPLAY", "PYOPENGL_PLATFORM")
            if os.environ.get(k) is not None
        },
        "versions": versions,
        "files": {
            "rgb_png": os.path.getsize("/tmp/sim_rgb.png"),
            "points_npy": os.path.getsize("/tmp/sim_points.npy"),
        },
    }

    print("=" * 70)
    print("METAWORLD MODERN-STACK PROBE")
    print("=" * 70)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("-" * 70)
    print(f"task            : {report['task']}")
    print(f"obs keys        : {report['obs_keys']}")
    for k, s in obs_shapes.items():
        print(f"  obs[{k:22s}] shape={s[0]} dtype={s[1]}")
    print(f"action shape    : {report['action_shape']}")
    print(
        f"image           : shape={report['image_shape']} dtype={report['image_dtype']} "
        f"min={report['image_min']} max={report['image_max']} std={report['image_std']}"
    )
    print(
        f"depth range (m) : {report['depth_min']} .. {report['depth_max']} "
        f"(median {report['depth_median']}, valid {n_valid}/{report['total_pixels']})"
    )
    print(f"point cloud     : shape={report['points_shape']} (metres, world frame)")
    print(
        f"point cloud xyz : min={report['points_min']} max={report['points_max']} "
        f"z_median={report['points_z_median']}"
    )
    print(
        f"landmark check  : floor plane z={report['floor_plane_z_ground_truth']} (from model), "
        f"{report['frac_points_on_floor_plane'] * 100:.1f}% of points land on it"
    )
    print(f"MUJOCO_GL       : {MUJOCO_GL}")
    print(f"GL renderer     : {gl_strings}")
    print(f"versions        : {versions}")

    # ---- self-verification of the saved artifacts -------------------------
    img_back = imageio.imread("/tmp/sim_rgb.png")
    pts_back = np.load("/tmp/sim_points.npy")
    checks = {
        "rgb_file_nonzero_bytes": report["files"]["rgb_png"] > 1000,
        "rgb_has_variance": float(img_back.astype(np.float64).std()) > 1.0,
        "rgb_not_blank": int(img_back.max()) > int(img_back.min()) + 10,
        "points_shape_1024x3": pts_back.shape == (N_POINTS, 3),
        "points_dtype_float32": pts_back.dtype == np.float32,
        "points_finite": bool(np.isfinite(pts_back).all()),
        "points_nontrivial_extent": float(pts_back.max() - pts_back.min()) > 0.05,
        "depth_finite_nonzero": report["depth_max"] > 0.0 and report["depth_min"] > 0.0,
        # Metric-scale check: a metre-scale scene, not normalised [0,1] depth.
        "points_metric_scale": float(np.percentile(np.abs(pts_back[:, 2]), 90)) < 3.0,
        "points_in_front_of_camera": 0.2 < report["points_dist_to_cam_median"] < 10.0,
        # Non-circular check: reconstruction must hit the model's own floor plane.
        "reconstruction_hits_floor_plane": report["frac_points_on_floor_plane"] > 0.005,
    }
    print("-" * 70)
    for k, v in checks.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    all_ok = all(checks.values())
    print("=" * 70)
    print("PROBE RESULT:", "SUCCESS" if all_ok else "FAILED_CHECKS")
    print("=" * 70)
    env.close()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

Re-run with:

```bash
cp /path/to/sim_probe.py /tmp/sim_probe.py
MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 \
  /home/s/.local/share/flowpolicy/sim-env/bin/python /tmp/sim_probe.py
# exit code 0 + "PROBE RESULT: SUCCESS"
```

Note: WSL can clear `/tmp` between sessions, so re-copy the script before running.
