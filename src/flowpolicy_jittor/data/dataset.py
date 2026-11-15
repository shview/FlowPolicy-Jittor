"""Zarr-backed sequence dataset for the Jittor FlowPolicy port.

The released implementation lives in
``flow_policy_3d.dataset.metaworld_dataset.MetaworldDataset`` (upstream commit
``86d10c1``) and reads a Zarr replay buffer whose fields are
``action``, ``state``, ``point_cloud`` and ``episode_ends``.  Windows have a
fixed length, are allowed to start before or end after an episode, and are then
padded by repeating the first or last frame of that episode.

This module keeps the same data contract in NumPy/Jittor terms:

* ``point_cloud``: ``[T, N, 3]``
* ``agent_pos`` (upstream ``state``): ``[T, D]``
* ``action``: ``[T, A]``
* ``episode_ends``: strictly increasing end offsets, one per episode

Two behaviours are deliberate and documented in ``docs/DECISIONS.md``:

1. padding repeats the first/last frame instead of zero-filling, matching the
   released sampler;
2. normalizer statistics are fitted from **training episodes only**, while the
   released ``get_normalizer`` fits on the whole buffer and therefore lets
   validation episodes influence the scaling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import zarr

from jittor.dataset import Dataset

from ..model.normalizer import LinearNormalizer

ACTION_KEY = "action"
POINT_CLOUD_KEY = "point_cloud"
EPISODE_ENDS_KEY = "episode_ends"
#: The released MetaWorld dataset names the robot state ``state``; later configs
#: call the same field ``agent_pos``.  Both are accepted, ``agent_pos`` first.
STATE_KEYS: Tuple[str, ...] = ("agent_pos", "state")

DEFAULT_NUM_POINTS = 1024


def _positive_int(name: str, value: Any, minimum: int = 1) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def episode_validation_mask(
    n_episodes: int,
    val_ratio: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Return a boolean mask selecting whole validation episodes.

    Mirrors ``flow_policy_3d.common.sampler.get_val_mask``: at least one episode
    is held out when ``val_ratio > 0``, at least one episode stays in training,
    and the choice is reproducible for a fixed ``seed``.
    """
    n_episodes = _positive_int("n_episodes", n_episodes)
    val_ratio = float(val_ratio)
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must lie in [0, 1), got {val_ratio}")
    mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0.0:
        return mask
    n_val = min(max(1, int(round(n_episodes * val_ratio))), n_episodes - 1)
    rng = np.random.default_rng(seed)
    mask[rng.choice(n_episodes, size=n_val, replace=False)] = True
    return mask


def build_window_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: Optional[np.ndarray] = None,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    """Port of ``flow_policy_3d.common.sampler.create_indices``.

    Returns an ``[n_windows, 4]`` int64 array of
    ``(buffer_start, buffer_end, sample_start, sample_end)`` rows.  Rows describe
    a slice of the flat buffer plus the offset at which it belongs inside a
    ``sequence_length`` window; everything outside that slice is padding.
    """
    episode_ends = np.asarray(episode_ends)
    if episode_ends.ndim != 1 or episode_ends.size == 0:
        raise ValueError("episode_ends must be a non-empty 1-D array")
    sequence_length = _positive_int("sequence_length", sequence_length)
    if episode_mask is None:
        episode_mask = np.ones(episode_ends.shape[0], dtype=bool)
    episode_mask = np.asarray(episode_mask, dtype=bool)
    if episode_mask.shape != episode_ends.shape:
        raise ValueError(
            "episode_mask and episode_ends must have the same length, "
            f"got {episode_mask.shape} and {episode_ends.shape}"
        )

    pad_before = min(max(int(pad_before), 0), sequence_length - 1)
    pad_after = min(max(int(pad_after), 0), sequence_length - 1)

    rows = []
    previous_end = 0
    for index in range(len(episode_ends)):
        start_index = previous_end
        end_index = int(episode_ends[index])
        previous_end = end_index
        if not episode_mask[index]:
            continue
        episode_length = end_index - start_index
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        for offset in range(min_start, max_start + 1):
            buffer_start = max(offset, 0) + start_index
            buffer_end = min(offset + sequence_length, episode_length) + start_index
            sample_start = buffer_start - (offset + start_index)
            sample_end = sequence_length - (
                (offset + sequence_length + start_index) - buffer_end
            )
            rows.append([buffer_start, buffer_end, sample_start, sample_end])
    if not rows:
        return np.zeros((0, 4), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def _pad_sequence(    values: np.ndarray,
    sequence_length: int,
    sample_start: int,
    sample_end: int,
) -> np.ndarray:
    """Edge-pad ``values`` into a ``sequence_length`` window."""
    if values.shape[0] == sequence_length:
        return values
    window = np.empty((sequence_length,) + values.shape[1:], dtype=values.dtype)
    if sample_start > 0:
        window[:sample_start] = values[0]
    if sample_end < sequence_length:
        window[sample_end:] = values[-1]
    window[sample_start:sample_end] = values
    return window


def select_windows(dataset: "MetaworldZarrDataset", count: int, seed: int = 0) -> np.ndarray:
    """Deterministically pick ``count`` distinct window indices from ``dataset``.

    Used by the 32-window overfit smoke run: the selection must be identical on
    every machine for the same seed, otherwise the reported loss curve is not
    comparable.  Indices are returned sorted so batching order is stable too.
    """
    count = _positive_int("count", count)
    total = int(getattr(dataset, "num_windows", 0))
    if total <= 0:
        raise ValueError("dataset has no sequence windows to select")
    if count > total:
        raise ValueError(
            f"requested {count} windows but the dataset only has {total}"
        )
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(total, size=count, replace=False)).astype(np.int64)


class MetaworldZarrDataset(Dataset):
    """Sequence-window dataset over a MetaWorld Zarr replay buffer.

    ``mode="train"`` uses the training episodes (optionally downsampled to
    ``max_train_episodes``); ``mode="val"`` uses the complementary episodes.
    ``get_validation_dataset()`` returns the ``val`` view of the same buffer.

    Note:
        ``len(dataset)`` follows Jittor's batch semantics (``total_len`` divided
        by ``batch_size``).  Use :attr:`num_windows` whenever a real sample count
        is needed, for example to derive steps per epoch.
    """

    def __init__(
        self,
        zarr_path: Any,
        *,
        horizon: int = 4,
        n_obs_steps: int = 2,
        n_action_steps: int = 4,
        pad_before: Optional[int] = None,
        pad_after: Optional[int] = None,
        num_points: Optional[int] = DEFAULT_NUM_POINTS,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
        seed: int = 42,
        mode: str = "train",
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {"train", "val"}:
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")

        self.zarr_path = Path(zarr_path)
        self.horizon = _positive_int("horizon", horizon)
        self.n_obs_steps = _positive_int("n_obs_steps", n_obs_steps)
        self.n_action_steps = _positive_int("n_action_steps", n_action_steps)
        if self.n_obs_steps > self.horizon:
            raise ValueError("n_obs_steps must not exceed horizon")
        if self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps must not exceed horizon")
        # ``pad_before``/``pad_after`` default to the released task config:
        # ``n_obs_steps - 1`` and ``n_action_steps - 1``.
        self.pad_before = (
            self.n_obs_steps - 1 if pad_before is None else int(pad_before)
        )
        self.pad_after = (
            self.n_action_steps - 1 if pad_after is None else int(pad_after)
        )
        if self.pad_before < 0 or self.pad_after < 0:
            raise ValueError("pad_before and pad_after must be non-negative")
        self.num_points = (
            None if num_points is None else _positive_int("num_points", num_points)
        )
        self.val_ratio = float(val_ratio)
        self.max_train_episodes = (
            None
            if max_train_episodes is None
            else _positive_int("max_train_episodes", max_train_episodes)
        )
        self.seed = int(seed)
        self.mode = mode
        self.validate_finite = bool(validate_finite)

        self._store = zarr.open(str(self.zarr_path), mode="r")
        if not hasattr(self._store, "array_keys"):
            raise ValueError(
                f"zarr path must point to a group with named arrays: {self.zarr_path}"
            )
        self._arrays = self._open_arrays()
        self._state_key = self._resolve_state_key()
        self._episode_ends = np.asarray(self._arrays[EPISODE_ENDS_KEY][:])
        self._validate_episode_ends()

        self._val_mask = episode_validation_mask(
            self.n_episodes, self.val_ratio, self.seed
        )
        self._train_mask = ~self._val_mask if np.any(self._val_mask) else np.ones(
            self.n_episodes, dtype=bool
        )
        self._train_mask = self._downsample(self._train_mask)
        self.episode_mask = (
            self._train_mask.copy() if mode == "train" else self._val_mask.copy()
        )
        self._indices = build_window_indices(
            self._episode_ends,
            sequence_length=self.horizon,
            episode_mask=self.episode_mask,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
        )
        self.total_len = int(self._indices.shape[0])
        self.set_attrs(total_len=self.total_len)

    # ------------------------------------------------------------------ opening
    def _array_container(self) -> Any:
        """Return the group that holds the trajectory arrays.

        The released collector writes ``data/<array>`` plus
        ``meta/episode_ends``; hand-made stores keep everything at the root.  Both
        layouts are supported so real demonstration files load unchanged.
        """
        if "data" in self._store:
            candidate = self._store["data"]
            if hasattr(candidate, "array_keys"):
                return candidate
        return self._store

    def _episode_ends_array(self, container: Any) -> Any:
        # Membership tests only: zarr 2.x refuses to iterate a group that has a
        # child literally named "meta", which is exactly the released layout.
        for group in (container, self._store):
            if EPISODE_ENDS_KEY in group:
                return group[EPISODE_ENDS_KEY]
        if "meta" in self._store and EPISODE_ENDS_KEY in self._store["meta"]:
            return self._store["meta"][EPISODE_ENDS_KEY]
        raise KeyError(
            "zarr store is missing the episode_ends array; looked for "
            "episode_ends at the root, under data/, and under meta/"
        )

    def _open_arrays(self) -> Dict[str, Any]:
        container = self._array_container()
        missing = [key for key in (ACTION_KEY, POINT_CLOUD_KEY) if key not in container]
        if missing:
            raise KeyError(
                "zarr store is missing required arrays: " + ", ".join(sorted(missing))
            )
        if not any(key in container for key in STATE_KEYS):
            raise KeyError(
                "zarr store is missing the robot state array; expected one of: "
                + ", ".join(STATE_KEYS)
            )
        arrays = {key: container[key] for key in (ACTION_KEY, POINT_CLOUD_KEY)}
        state_key = next(key for key in STATE_KEYS if key in container)
        arrays[state_key] = container[state_key]
        arrays[EPISODE_ENDS_KEY] = self._episode_ends_array(container)

        action = arrays[ACTION_KEY]
        point_cloud = arrays[POINT_CLOUD_KEY]
        if action.ndim != 2:
            raise ValueError(
                f"{ACTION_KEY} must be [T, A], got shape {action.shape}"
            )
        if point_cloud.ndim != 3:
            raise ValueError(
                f"{POINT_CLOUD_KEY} must be [T, N, 3], got shape {point_cloud.shape}"
            )
        if int(point_cloud.shape[2]) != 3:
            raise ValueError(
                "point_cloud trailing dimension must be 3 (XYZ), got "
                f"{int(point_cloud.shape[2])}"
            )
        if self.num_points is not None and int(point_cloud.shape[1]) != self.num_points:
            raise ValueError(
                f"point_cloud must contain {self.num_points} points, got "
                f"{int(point_cloud.shape[1])}"
            )
        return arrays

    def _resolve_state_key(self) -> str:
        for key in STATE_KEYS:
            if key in self._arrays:
                return key
        raise KeyError(
            "zarr store is missing the robot state array; expected one of: "
            + ", ".join(STATE_KEYS)
        )

    def _validate_episode_ends(self) -> None:
        ends = self._episode_ends
        if ends.ndim != 1 or ends.size == 0:
            raise ValueError("episode_ends must be a non-empty 1-D array")
        if not np.issubdtype(ends.dtype, np.integer):
            raise TypeError(f"episode_ends must be an integer array, got {ends.dtype}")
        if int(ends[0]) <= 0:
            raise ValueError("episode_ends[0] must be positive (episode 0 is empty)")
        if np.any(np.diff(ends) <= 0):
            raise ValueError(
                "episode_ends must be strictly increasing; empty episodes are not "
                "supported"
            )
        total = int(ends[-1])
        for key in (ACTION_KEY, POINT_CLOUD_KEY, self._state_key):
            length = int(self._arrays[key].shape[0])
            if length != total:
                raise ValueError(
                    f"{key} has {length} frames but episode_ends ends at {total}"
                )
        state = self._arrays[self._state_key]
        if state.ndim != 2:
            raise ValueError(f"{self._state_key} must be [T, D], got {state.shape}")

    def _downsample(self, mask: np.ndarray) -> np.ndarray:
        if self.max_train_episodes is None:
            return mask
        count = int(mask.sum())
        if count <= self.max_train_episodes:
            return mask
        rng = np.random.default_rng(self.seed)
        selected = np.flatnonzero(mask)
        keep = rng.choice(selected, size=self.max_train_episodes, replace=False)
        result = np.zeros_like(mask)
        result[np.sort(keep)] = True
        return result

    # ---------------------------------------------------------------- accessors
    @property
    def state_key(self) -> str:
        return self._state_key

    @property
    def episode_ends(self) -> np.ndarray:
        return self._episode_ends

    @property
    def n_episodes(self) -> int:
        return int(self._episode_ends.shape[0])

    @property
    def num_windows(self) -> int:
        return int(self._indices.shape[0])

    @property
    def indices(self) -> np.ndarray:
        return self._indices

    @property
    def train_mask(self) -> np.ndarray:
        return self._train_mask

    @property
    def val_mask(self) -> np.ndarray:
        return self._val_mask

    def episode_span(self, index: int) -> Tuple[int, int]:
        start = 0 if index == 0 else int(self._episode_ends[index - 1])
        return start, int(self._episode_ends[index])

    def episode_of_window(self, index: int) -> int:
        """Return the episode that owns the first real frame of a window."""
        buffer_start, _, _, _ = self._indices[index]
        return int(np.searchsorted(self._episode_ends, buffer_start, side="right"))

    # -------------------------------------------------------------------- items
    def __getitem__(self, index: int) -> Dict[str, Any]:
        buffer_start, buffer_end, sample_start, sample_end = (
            int(value) for value in self._indices[index]
        )
        window = {}
        for key, array in (
            (ACTION_KEY, self._arrays[ACTION_KEY]),
            (self._state_key, self._arrays[self._state_key]),
            (POINT_CLOUD_KEY, self._arrays[POINT_CLOUD_KEY]),
        ):
            values = np.asarray(array[buffer_start:buffer_end], dtype=np.float32)
            window[key] = _pad_sequence(
                values, self.horizon, sample_start, sample_end
            )
            if self.validate_finite and not np.all(np.isfinite(window[key])):
                raise ValueError(
                    f"{key} window {index} contains NaN or Inf; refusing to train on it"
                )

        point_cloud = window[POINT_CLOUD_KEY][: self.n_obs_steps]
        agent_pos = window[self._state_key][: self.n_obs_steps]
        action = window[ACTION_KEY][: self.n_action_steps]
        return {
            "obs": {
                "point_cloud": np.ascontiguousarray(point_cloud),
                "agent_pos": np.ascontiguousarray(agent_pos),
            },
            "action": np.ascontiguousarray(action),
        }

    # ------------------------------------------------------------------ scaling
    def get_normalizer(self, mode: str = "limits") -> LinearNormalizer:
        """Fit the three-field limits normalizer from training episodes only."""
        if mode != "limits":
            raise NotImplementedError("only mode='limits' is validated")
        spans = [self.episode_span(i) for i in np.flatnonzero(self._train_mask)]
        if not spans:
            raise ValueError("no training episodes are available to fit the normalizer")
        action = np.concatenate(
            [self._arrays[ACTION_KEY][start:end] for start, end in spans]
        )
        state = np.concatenate(
            [self._arrays[self._state_key][start:end] for start, end in spans]
        )
        point_cloud = np.concatenate(
            [self._arrays[POINT_CLOUD_KEY][start:end] for start, end in spans]
        )
        normalizer = LinearNormalizer()
        normalizer.fit(
            {
                "action": np.asarray(action, dtype=np.float32),
                "agent_pos": np.asarray(state, dtype=np.float32),
                "point_cloud": np.asarray(point_cloud, dtype=np.float32),
            }
        )
        return normalizer

    def get_validation_dataset(self) -> "MetaworldZarrDataset":
        if not np.any(self._val_mask):
            raise ValueError(
                "validation split is empty; construct with val_ratio > 0 to use it"
            )
        dataset = MetaworldZarrDataset(
            self.zarr_path,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            num_points=self.num_points,
            val_ratio=self.val_ratio,
            max_train_episodes=self.max_train_episodes,
            seed=self.seed,
            mode="val",
            validate_finite=self.validate_finite,
        )
        if not np.array_equal(dataset.episode_mask, self._val_mask):
            raise RuntimeError("validation split is not reproducible for this seed")
        return dataset

    def close(self) -> None:
        store = getattr(self, "_store", None)
        if store is not None and hasattr(store, "close"):
            store.close()
