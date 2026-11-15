"""Write MetaWorld demonstration episodes into the released Zarr layout.

The released collector writes ``data/<array>`` for every trajectory array and
``meta/episode_ends`` for the cumulative episode boundaries; ``MetaworldZarrDataset``
reads both that layout and a flat one (DECISIONS D015).  This module owns the write
side so a recording run can be checked without trusting the simulator, and it never
overwrites an existing store.

Run the writer self-test (no simulator required):

    python scripts/record_demos.py --self-test --output data/_writer_selftest.zarr
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

POINT_COUNT = 1024
POINT_DIM = 3
MIN_SUCCESS_STEPS = 5

# Transcribed from flow_policy_3d/env/metaworld/metaworld_wrapper.py (see
# env/metaworld_contract.md).  Keep these numbers identical to the released code.
POINT_CLOUD_X_ANGLE = 61.4
POINT_CLOUD_Y_ANGLE = -7.0
DEFAULT_POINT_BOUNDS = (-0.5, -1.5, -0.795, 1.0, -0.4, 100.0)
CAMERA_NAME = "corner2"
IMAGE_SIZE = 128
EPISODE_LENGTH = 200


def point_cloud_transform(
    x_angle: float = POINT_CLOUD_X_ANGLE, y_angle: float = POINT_CLOUD_Y_ANGLE
) -> np.ndarray:
    """The released fixed rotation, ``Rx(x_angle) @ Ry(y_angle)``."""
    ax, ay = np.deg2rad(x_angle), np.deg2rad(y_angle)
    rx = np.array(
        [[1, 0, 0], [0, np.cos(ax), np.sin(ax)], [0, -np.sin(ax), np.cos(ax)]],
        dtype=np.float64,
    )
    ry = np.array(
        [[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]],
        dtype=np.float64,
    )
    return rx @ ry


def unproject_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Turn a metric depth image into XYZ points in the camera frame.

    ``intrinsics`` is ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.  MuJoCo cameras look
    down ``-z``, so the optical axis is negated to produce the usual camera frame.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must be [H, W], got {depth.shape}")
    height, width = depth.shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError(f"focal lengths must be positive, got fx={fx}, fy={fy}")
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    points = np.stack([x, y, -z], axis=-1).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def crop_points(
    points: np.ndarray,
    bounds: Sequence[float] = DEFAULT_POINT_BOUNDS,
) -> np.ndarray:
    """Keep points strictly inside the released task bounds."""
    x_min, y_min, z_min, x_max, y_max, z_max = (float(value) for value in bounds)
    keep = (
        (points[:, 0] > x_min)
        & (points[:, 1] > y_min)
        & (points[:, 2] > z_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] < y_max)
        & (points[:, 2] < z_max)
    )
    return points[keep]


def farthest_point_sample(points: np.ndarray, count: int) -> np.ndarray:
    """Greedy farthest-point sampling, deterministic and dependency-free."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != POINT_DIM:
        raise ValueError(f"points must be [N, 3], got {points.shape}")
    total = int(points.shape[0])
    count = int(count)
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if total < count:
        raise ValueError(
            f"cannot sample {count} points from a cloud of {total}; widen the crop bounds"
        )
    indices = np.empty(count, dtype=np.int64)
    distances = np.full(total, np.inf, dtype=np.float64)
    # Start from the point farthest from the centroid: stable and seed-free.
    centroid = points.mean(axis=0)
    current = int(np.argmax(((points - centroid) ** 2).sum(axis=1)))
    for slot in range(count):
        indices[slot] = current
        delta = ((points - points[current]) ** 2).sum(axis=1)
        distances = np.minimum(distances, delta)
        distances[current] = -1.0
        current = int(np.argmax(distances))
    return points[indices]


def point_cloud_from_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    *,
    bounds: Sequence[float] = DEFAULT_POINT_BOUNDS,
    count: int = POINT_COUNT,
    transform: np.ndarray | None = None,
) -> np.ndarray:
    """Depth image -> cropped -> ``count`` XYZ points, matching the released wrapper."""
    points = unproject_depth(depth, intrinsics)
    if transform is None:
        transform = point_cloud_transform()
    points = points @ np.asarray(transform, dtype=np.float64).T
    points = crop_points(points, bounds)
    sampled = farthest_point_sample(points, count)
    return sampled.astype(np.float32)


@dataclass
class Episode:
    """One recorded demonstration episode."""

    actions: np.ndarray
    states: np.ndarray
    point_clouds: np.ndarray
    success_steps: int = 0
    task: str = ""
    seed: int = 0
    extra: Dict[str, np.ndarray] = field(default_factory=dict)

    def validate(self, *, point_count: int = POINT_COUNT) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        states = np.asarray(self.states, dtype=np.float32)
        clouds = np.asarray(self.point_clouds, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"actions must be [T, A], got {actions.shape}")
        if states.ndim != 2:
            raise ValueError(f"states must be [T, S], got {states.shape}")
        if clouds.ndim != 3 or clouds.shape[2] != POINT_DIM:
            raise ValueError(f"point clouds must be [T, N, 3], got {clouds.shape}")
        if int(clouds.shape[1]) != point_count:
            raise ValueError(
                f"point clouds must contain {point_count} points, got {int(clouds.shape[1])}"
            )
        lengths = {actions.shape[0], states.shape[0], clouds.shape[0]}
        if len(lengths) != 1:
            raise ValueError(
                "actions, states and point clouds disagree on the episode length: "
                f"{sorted(lengths)}"
            )
        if actions.shape[0] == 0:
            raise ValueError("an episode must contain at least one step")
        for name, array in (("actions", actions), ("states", states), ("point_clouds", clouds)):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")
        if self.success_steps < 0:
            raise ValueError(f"success_steps must be >= 0, got {self.success_steps}")


def _require_zarr():
    try:
        import zarr  # noqa: F401
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError(
            "writing demonstrations requires zarr; install it in the collection "
            "environment, not in the Jittor environment"
        ) from error
    return zarr


def write_released_store(
    path: str | Path,
    episodes: Sequence[Episode],
    *,
    point_count: int = POINT_COUNT,
    overwrite: bool = False,
    compressor: str = "zstd",
    provenance: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Write ``episodes`` to ``path`` in the released layout and summarise it."""
    zarr = _require_zarr()
    if not episodes:
        raise ValueError("refusing to write a store without episodes")
    for episode in episodes:
        episode.validate(point_count=point_count)

    target = Path(path)
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing demonstration data: {target}"
        )

    actions = np.concatenate([episode.actions for episode in episodes], axis=0).astype(np.float32)
    states = np.concatenate([episode.states for episode in episodes], axis=0).astype(np.float32)
    clouds = np.concatenate([episode.point_clouds for episode in episodes], axis=0).astype(np.float32)
    lengths = [int(episode.actions.shape[0]) for episode in episodes]
    episode_ends = np.cumsum(lengths).astype(np.int64)

    root = zarr.open(str(target), mode="a")
    data = root.require_group("data")
    meta = root.require_group("meta")
    blosc = None
    if compressor:
        try:
            blosc = zarr.Blosc(cname=compressor, clevel=3, shuffle=1)
        except AttributeError:  # zarr >= 3 renamed the codec class
            blosc = None

    def put(group, key, value):
        group.create_dataset(key, data=value, chunks=value.shape, overwrite=True, compressor=blosc)

    put(data, "action", actions)
    put(data, "state", states)
    put(data, "point_cloud", clouds)
    put(meta, "episode_ends", episode_ends)
    put(meta, "success_steps", np.asarray([int(episode.success_steps) for episode in episodes], dtype=np.int32))

    # Provenance lives on disk so a synthetic store can never masquerade as real data.
    marker = {"synthetic": True, "writer": "scripts/record_demos.py"}
    if provenance:
        marker.update(provenance)
    root.attrs["provenance"] = marker

    return {
        "path": str(target),
        "layout": "data/ + meta/episode_ends",
        "episodes": len(episodes),
        "frames": int(actions.shape[0]),
        "episode_lengths": lengths,
        "episode_ends": episode_ends.tolist(),
        "action_shape": list(actions.shape),
        "state_shape": list(states.shape),
        "point_cloud_shape": list(clouds.shape),
        "action_range": [float(actions.min()), float(actions.max())],
        "state_range": [float(states.min()), float(states.max())],
        "point_cloud_range": [float(clouds.min()), float(clouds.max())],
        "success_steps": [int(episode.success_steps) for episode in episodes],
        "tasks": sorted({episode.task for episode in episodes if episode.task}),
        "seeds": [int(episode.seed) for episode in episodes],
    }


def _rollout_report(path: str | Path, summary: Dict[str, object]) -> Dict[str, object]:
    """Load the written store through the dataset and report what a trainer sees."""
    from flowpolicy_jittor.data import MetaworldZarrDataset

    dataset = MetaworldZarrDataset(
        path,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=4,
        num_points=POINT_COUNT,
        val_ratio=0.0,
    )
    sample = dataset[0]
    report = dict(summary)
    report["dataset"] = {
        "state_key": dataset.state_key,
        "episodes": int(dataset.n_episodes),
        "windows": int(dataset.num_windows),
        "episode_ends": dataset.episode_ends.tolist(),
        "action_dim": int(sample["action"].shape[-1]),
        "state_dim": int(sample["obs"]["agent_pos"].shape[-1]),
        "point_cloud_shape": list(np.asarray(sample["obs"]["point_cloud"]).shape),
        "finite": bool(
            np.all(np.isfinite(np.asarray(sample["action"])))
            and np.all(np.isfinite(np.asarray(sample["obs"]["point_cloud"])))
        ),
    }
    return report


def _synthetic_episodes(count: int = 2, length: int = 6, seed: int = 0) -> List[Episode]:
    """Random-but-finite episodes, used only to check the writer's own validation."""
    rng = np.random.default_rng(seed)
    episodes = []
    for index in range(count):
        actions = rng.normal(scale=0.1, size=(length, 4)).astype(np.float32)
        states = rng.normal(size=(length, 9)).astype(np.float32)
        clouds = rng.normal(scale=0.2, size=(length, POINT_COUNT, POINT_DIM)).astype(np.float32)
        episodes.append(
            Episode(
                actions=actions,
                states=states,
                point_clouds=clouds,
                success_steps=length,
                task="synthetic",
                seed=int(seed + index),
            )
        )
    return episodes


def collect_episodes(
    adapter,
    *,
    num_episodes: int,
    max_steps: int = EPISODE_LENGTH,
    min_success_steps: int = MIN_SUCCESS_STEPS,
    max_attempts: int | None = None,
    point_count: int = POINT_COUNT,
    log=print,
) -> tuple[List[Episode], Dict[str, object]]:
    """Roll out ``adapter`` until ``num_episodes`` good episodes are recorded.

    ``adapter`` is any object with ``task``, ``reset(seed)``, ``step(action)`` and
    ``expert_action(full_state)``.  ``reset`` returns an observation dict and
    ``step`` returns ``(observation, success)``; both observations must carry
    ``agent_pos``, ``full_state`` and ``point_cloud``.  The success filter mirrors the
    released collector: an episode is kept only when it succeeds at least
    ``min_success_steps`` times.
    """
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    budget = num_episodes * 10 if max_attempts is None else int(max_attempts)
    kept: List[Episode] = []
    attempts = 0
    rejected = 0
    while len(kept) < num_episodes and attempts < budget:
        attempts += 1
        observation = adapter.reset(attempts - 1)
        actions, states, clouds = [], [], []
        success_steps = 0
        steps = 0
        while steps < max_steps:
            actions.append(np.asarray(adapter.expert_action(observation["full_state"]), dtype=np.float32))
            states.append(np.asarray(observation["agent_pos"], dtype=np.float32))
            clouds.append(np.asarray(observation["point_cloud"], dtype=np.float32))
            next_observation, success = adapter.step(actions[-1])
            success_steps += int(success)
            steps += 1
            observation = next_observation
            if observation.get("done", False):
                break
        if success_steps < min_success_steps:
            rejected += 1
            log(
                f"episode attempt {attempts}: rejected ({success_steps} successful steps, "
                f"need {min_success_steps})"
            )
            continue
        episode = Episode(
            actions=np.stack(actions),
            states=np.stack(states),
            point_clouds=np.stack(clouds),
            success_steps=int(success_steps),
            task=str(getattr(adapter, "task", "")),
            seed=int(getattr(adapter, "seed", attempts - 1)),
        )
        episode.validate(point_count=point_count)
        kept.append(episode)
        log(
            f"episode {len(kept)}/{num_episodes}: kept, {steps} steps, "
            f"{success_steps} successful, attempt {attempts}"
        )
    if len(kept) < num_episodes:
        raise RuntimeError(
            f"recorded only {len(kept)}/{num_episodes} episodes in {attempts} attempts; "
            "the expert is not succeeding often enough to collect demonstrations"
        )
    return kept, {"attempts": attempts, "rejected": rejected, "kept": len(kept)}


# Workspace box in WORLD coordinates.  The released box is expressed in the
# metaworld 2.0.0 / mujoco-py frame and keeps almost nothing on this stack, so the
# box is re-derived here from the validated camera->world mapping (see DECISIONS
# D016).  It covers the tabletop region used by every MT1 task.
WORLD_BOUNDS = (-0.35, 0.25, -0.05, 0.35, 0.95, 0.55)


def _expert_policy_name(task: str) -> str:
    """``reach-v3`` -> ``SawyerReachV3Policy`` (released naming scheme, v3 policies)."""
    base = task.replace("-v3", "").replace("-v2", "")
    if base == "peg-insert-side":
        return "SawyerPegInsertionSideV3Policy"
    parts = [part.capitalize() for part in base.split("-")]
    return "Sawyer" + "".join(parts) + "V3Policy"


class MetaWorldAdapter:
    """MetaWorld v3 environment adapter exposing the released observation contract.

    Requires ``mujoco`` and ``metaworld`` in the *collection* environment, never in
    the Jittor environment.  ``MUJOCO_GL=egl`` selects headless software rendering.
    """

    def __init__(
        self,
        task: str = "reach-v3",
        *,
        seed: int = 0,
        size: int = IMAGE_SIZE,
        camera: str = CAMERA_NAME,
        bounds: Sequence[float] = WORLD_BOUNDS,
        point_count: int = POINT_COUNT,
        override_camera: bool = True,
    ):
        import metaworld
        import mujoco

        self._mujoco = mujoco
        self.task = task
        self.seed = int(seed)
        self.size = int(size)
        self.point_count = int(point_count)
        self.bounds = tuple(float(value) for value in bounds)
        self.transform = point_cloud_transform()

        # metaworld samples the task goals from the *global* numpy RNG while the
        # benchmark is built, so the same --seed only reproduces the goals when that
        # RNG is seeded first (measured: unseeded runs pick different goals).
        np.random.seed(self.seed)
        self._collection = metaworld.MT1(task)
        self.env = self._collection.train_classes[task]()
        self._tasks = list(self._collection.train_tasks)
        self.env.set_task(self._tasks[0])
        self.model = self.env.model
        self.data = self.env.data
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
        )
        if self.camera_id < 0:
            raise ValueError(f"{task} has no camera named {camera!r}")
        if override_camera:
            self.model.cam_pos[self.camera_id] = [0.6, 0.295, 0.8]
            self.model.vis.map.znear = 0.1
            self.model.vis.map.zfar = 1.5
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=self.size, width=self.size)

        import metaworld.policies as policies

        policy_name = _expert_policy_name(task)
        if not hasattr(policies, policy_name):
            raise ValueError(
                f"no scripted expert {policy_name} for task {task!r}; "
                "available: " + ", ".join(sorted(name for name in dir(policies) if name.endswith("V3Policy")))
            )
        self.policy = getattr(policies, policy_name)()
        self._observation: Dict[str, object] | None = None

    # ---------------------------------------------------------------- rendering
    def _depth_and_rgb(self) -> tuple[np.ndarray, np.ndarray]:
        self.renderer.update_scene(self.data, camera=self.camera_id)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb = self.renderer.render().copy()
        return depth, rgb

    def point_cloud(self) -> np.ndarray:
        depth, _ = self._depth_and_rgb()
        height, width = depth.shape
        focal = height / (2 * np.tan(np.deg2rad(float(self.model.cam_fovy[self.camera_id])) / 2))
        cx, cy = (width - 1) / 2, (height - 1) / 2
        us, vs = np.meshgrid(np.arange(width), np.arange(height))
        camera_points = np.stack(
            [(us - cx) * depth / focal, -(vs - cy) * depth / focal, -depth], axis=-1
        ).reshape(-1, 3)
        rotation = np.asarray(self.data.cam_xmat[self.camera_id]).reshape(3, 3)
        position = np.asarray(self.data.cam_xpos[self.camera_id])
        world = camera_points @ rotation.T + position
        x0, y0, z0, x1, y1, z1 = self.bounds
        keep = (
            (world[:, 0] > x0) & (world[:, 1] > y0) & (world[:, 2] > z0)
            & (world[:, 0] < x1) & (world[:, 1] < y1) & (world[:, 2] < z1)
        )
        cropped = world[keep] @ self.transform.T
        return farthest_point_sample(cropped, self.point_count).astype(np.float32)

    # ------------------------------------------------------------- environment
    def agent_pos(self) -> np.ndarray:
        eef = np.asarray(self.env.get_endeff_pos(), dtype=np.float32)
        right = np.asarray(self.env._get_site_pos("rightEndEffector"), dtype=np.float32)
        left = np.asarray(self.env._get_site_pos("leftEndEffector"), dtype=np.float32)
        return np.concatenate([eef, right, left])

    def _make_observation(self, obs: np.ndarray) -> Dict[str, object]:
        return {
            "agent_pos": self.agent_pos(),
            "full_state": np.asarray(obs, dtype=np.float32),
            "point_cloud": self.point_cloud(),
            "done": False,
        }

    def reset(self, index: int) -> Dict[str, object]:
        self.env.set_task(self._tasks[index % len(self._tasks)])
        obs, _info = self.env.reset(seed=self.seed + int(index))
        mujoco = self._mujoco
        mujoco.mj_forward(self.model, self.data)
        self._observation = self._make_observation(obs)
        return self._observation

    def expert_action(self, full_state) -> np.ndarray:
        return np.asarray(self.policy.get_action(np.asarray(full_state, dtype=np.float64)), dtype=np.float32)

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(np.asarray(action, dtype=np.float32))
        self._mujoco.mj_forward(self.model, self.data)
        self._observation = self._make_observation(obs)
        success = float(info.get("success", 0.0)) > 0.5
        self._observation["done"] = bool(terminated or truncated)
        return self._observation, success

    def close(self) -> None:
        try:
            self.renderer.close()
        finally:
            self.env.close()


class _FakeAdapter:
    """Deterministic stand-in used by ``--self-test``; never used for real data."""
    task = "synthetic/adapter"

    def __init__(self, *, fail_first: int = 1, length: int = 6, seed: int = 0):
        self.fail_first = fail_first
        self.length = length
        self.seed = seed
        self._episode = -1
        self._step = 0
        self._position = np.zeros(3, dtype=np.float32)

    def reset(self, index: int):
        self._episode = int(index)
        self._step = 0
        self._position = np.zeros(3, dtype=np.float32)
        return self._observation()

    def _observation(self) -> Dict[str, object]:
        depth = np.full((IMAGE_SIZE, IMAGE_SIZE), 0.6 + 0.01 * self._step, dtype=np.float64)
        focal = IMAGE_SIZE / (2 * np.tan(np.deg2rad(45.0) / 2))
        intrinsics = np.array(
            [
                [focal, 0, (IMAGE_SIZE - 1) / 2],
                [0, focal, (IMAGE_SIZE - 1) / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        cloud = point_cloud_from_depth(depth, intrinsics, count=POINT_COUNT)
        state = np.concatenate([self._position, self._position + 0.1, self._position - 0.1])
        return {
            "agent_pos": state.astype(np.float32),
            "full_state": np.concatenate([state, np.zeros(11)]).astype(np.float32),
            "point_cloud": cloud,
            "done": False,
        }

    def expert_action(self, full_state) -> np.ndarray:
        return np.array([0.01, -0.02, 0.0, 1.0], dtype=np.float32)

    def step(self, action):
        self._step += 1
        self._position = self._position + np.asarray(action[:3], dtype=np.float32)
        succeeded = self._step > self.fail_first
        observation = self._observation()
        observation["done"] = self._step >= self.length
        return observation, succeeded


def _collect_selftest() -> Dict[str, object]:
    """Rehearse adapter -> filter -> writer without a simulator."""
    adapter = _FakeAdapter(fail_first=1, length=6)
    episodes, stats = collect_episodes(
        adapter, num_episodes=2, max_steps=6, min_success_steps=MIN_SUCCESS_STEPS, log=lambda *_: None
    )
    strict_error = ""
    strict_adapter = _FakeAdapter(fail_first=5, length=6)
    try:
        collect_episodes(
            strict_adapter,
            num_episodes=1,
            max_steps=6,
            min_success_steps=MIN_SUCCESS_STEPS,
            max_attempts=3,
            log=lambda *_: None,
        )
    except RuntimeError as error:
        strict_error = str(error)

    # The writer must reject a bad episode instead of storing it.
    invalid = _synthetic_episodes(count=1, length=2)
    invalid[0].actions[0, 0] = np.nan
    invalid_error = ""
    empty_error = ""
    with tempfile.TemporaryDirectory() as directory:
        try:
            write_released_store(Path(directory) / "invalid.zarr", invalid)
        except ValueError as error:
            invalid_error = str(error)
        try:
            write_released_store(Path(directory) / "empty.zarr", [])
        except ValueError as error:
            empty_error = str(error)
    return {
        "episodes": len(episodes),
        "lengths": [int(episode.actions.shape[0]) for episode in episodes],
        "success_steps": [int(episode.success_steps) for episode in episodes],
        "state_dim": int(episodes[0].states.shape[1]),
        "attempts": stats["attempts"],
        "rejected": stats["rejected"],
        "too_few_episodes_error": strict_error,
        "invalid_episode_error": invalid_error,
        "empty_episode_list_error": empty_error,
    }


def _geometry_selftest() -> Dict[str, object]:
    """Exercise depth -> crop -> FPS without a simulator."""
    height = width = IMAGE_SIZE
    focal = width / (2 * np.tan(np.deg2rad(45.0) / 2))
    intrinsics = np.array(
        [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    depth = (0.55 + 0.25 * (us / width) + 0.1 * (vs / height)).astype(np.float64)
    depth[0, 0] = np.nan  # one dropped pixel must not poison the cloud

    raw = unproject_depth(depth, intrinsics)
    sampled = point_cloud_from_depth(depth, intrinsics, count=POINT_COUNT)
    again = point_cloud_from_depth(depth, intrinsics, count=POINT_COUNT)
    tight = (-0.01, -0.01, -0.01, 0.01, 0.01, 0.01)
    tight_error = ""
    try:
        point_cloud_from_depth(depth, intrinsics, bounds=tight, count=POINT_COUNT)
    except ValueError as error:
        tight_error = str(error)
    return {
        "depth_shape": list(depth.shape),
        "intrinsics_focal": float(focal),
        "raw_points": int(raw.shape[0]),
        "points_after_crop": int(crop_points(point_cloud_transform() @ raw.T).T.shape[0]),
        "sampled_shape": list(sampled.shape),
        "sampled_finite": bool(np.all(np.isfinite(sampled))),
        "sampled_deterministic": bool(np.array_equal(sampled, again)),
        "sampled_bounds": [float(sampled.min()), float(sampled.max())],
        "tight_bounds_error": tight_error,
    }


def _simulator_versions() -> Dict[str, str]:
    """Record the simulator stack next to the data it produced."""
    from importlib import metadata

    versions = {}
    for name in ("mujoco", "metaworld", "gymnasium", "numpy", "zarr"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def _store_success_steps(path: str | Path) -> list:
    """Per-episode successful-step counts, if the store records them."""
    zarr = _require_zarr()
    try:
        store = zarr.open(str(path), mode="r")
        node = store["meta"] if "meta" in store else store
        if "success_steps" not in node:
            return []
        return [int(value) for value in np.asarray(node["success_steps"][:])]
    except Exception:  # noqa: BLE001 - diagnostics only
        return []


def _store_provenance(path: str | Path) -> Dict[str, object]:
    """Best-effort read of the on-disk provenance marker."""
    zarr = _require_zarr()
    try:
        return dict(zarr.open(str(path), mode="r").attrs.get("provenance", {}))
    except Exception:  # noqa: BLE001 - provenance is diagnostics, never a gate
        return {}


def check_store(path: str | Path, *, point_count: int = POINT_COUNT) -> Dict[str, object]:
    """Validate an existing demonstration store against the training contract."""
    from flowpolicy_jittor.data import MetaworldZarrDataset

    try:
        dataset = MetaworldZarrDataset(
            path,
            horizon=4,
            n_obs_steps=2,
            n_action_steps=4,
            num_points=point_count,
            val_ratio=0.0,
        )
    except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
        # A contract violation is a verdict, not a crash: report it and exit 1.
        return {
            "path": str(path),
            "status": "failed",
            "problems": [f"{type(error).__name__}: {error}"],
            "episodes": 0,
            "windows": 0,
        }
    problems = []
    if int(dataset.num_windows) <= 0:
        problems.append("the store yields zero training windows")
    sample = dataset[0] if int(dataset.num_windows) else None
    report = {
        "path": str(path),
        "state_key": dataset.state_key,
        "episodes": int(dataset.n_episodes),
        "windows": int(dataset.num_windows),
        "episode_ends": dataset.episode_ends.tolist(),
        "episode_lengths": np.diff(np.concatenate([[0], dataset.episode_ends])).tolist(),
        "provenance": _store_provenance(path),
        "success_steps": _store_success_steps(path),
        "problems": problems,
    }
    if sample is not None:
        action = np.asarray(sample["action"])
        cloud = np.asarray(sample["obs"]["point_cloud"])
        state = np.asarray(sample["obs"]["agent_pos"])
        report["shapes"] = {
            "action": list(action.shape),
            "point_cloud": list(cloud.shape),
            "agent_pos": list(state.shape),
        }
        report["finite"] = bool(
            np.all(np.isfinite(action)) and np.all(np.isfinite(cloud)) and np.all(np.isfinite(state))
        )
        if not report["finite"]:
            problems.append("the first window contains non-finite values")
        if int(cloud.shape[-2]) != point_count:
            problems.append(
                f"point cloud has {int(cloud.shape[-2])} points, expected {point_count}"
            )
    if problems:
        report["status"] = "failed"
    else:
        report["status"] = "ok"
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="write synthetic episodes and re-read them")
    parser.add_argument("--check-store", type=Path, default=None, help="validate an existing .zarr store")
    parser.add_argument("--record", action="store_true", help="record real MetaWorld demonstrations")
    parser.add_argument("--task", default="reach-v3", help="MetaWorld task name, e.g. reach-v3")
    parser.add_argument("--episodes", type=int, default=1, help="how many successful episodes to keep")
    parser.add_argument("--max-steps", type=int, default=EPISODE_LENGTH, help="step cap per attempt")
    parser.add_argument("--seed", type=int, default=0, help="base environment seed")
    parser.add_argument("--output", type=Path, default=None, help="destination .zarr path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing store (never the default)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.check_store is not None:
        report = check_store(args.check_store)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "ok" else 1
    if args.record:
        if args.output is None:
            parser.error("--record requires --output")
        adapter = MetaWorldAdapter(args.task, seed=args.seed)
        try:
            episodes, stats = collect_episodes(
                adapter,
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                min_success_steps=MIN_SUCCESS_STEPS,
                log=lambda message: print(f"[collect] {message}", file=sys.stderr),
            )
        finally:
            adapter.close()
        summary = write_released_store(
            args.output,
            episodes,
            overwrite=args.overwrite,
            provenance={
                "synthetic": False,
                "task": args.task,
                "seed": int(args.seed),
                "camera": CAMERA_NAME,
                "image_size": IMAGE_SIZE,
                "world_bounds": list(WORLD_BOUNDS),
                "packages": _simulator_versions(),
            },
        )
        summary["task"] = args.task
        summary["seed"] = int(args.seed)
        summary["collection"] = stats
        summary["next_step"] = (
            "verify in the Jittor environment: python scripts/record_demos.py "
            f"--check-store {args.output}"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if not args.self_test:
        parser.error(
            "choose --self-test (write and re-read a synthetic store) or "
            "--check-store PATH; recording needs the simulator (see env/sim_setup_log.md)"
        )
    if args.output is None:
        parser.error("--self-test requires --output")

    episodes, collect_stats = collect_episodes(
        _FakeAdapter(fail_first=1, length=6),
        num_episodes=2,
        max_steps=6,
        min_success_steps=MIN_SUCCESS_STEPS,
        log=lambda message: print(f"[collect] {message}", file=sys.stderr),
    )
    summary = write_released_store(args.output, episodes, overwrite=args.overwrite)
    report = _rollout_report(args.output, summary)
    report["collection"] = collect_stats
    report["geometry"] = _geometry_selftest()
    report["failure_modes"] = _collect_selftest()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
