"""Tests for the Jittor Zarr sequence dataset (FPJ-022).

The fixtures write a small synthetic store that mirrors the released field
layout, so episode boundaries, padding, split leakage and shape errors are all
checked without MuJoCo or real demonstrations.
"""

from pathlib import Path

import numpy as np
import pytest

jt = pytest.importorskip("jittor")
zarr = pytest.importorskip("zarr")

from flowpolicy_jittor.data.dataset import (  # noqa: E402
    MetaworldZarrDataset,
    build_window_indices,
    episode_validation_mask,
)


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = previous


def _write_store(
    path: Path,
    episodes,
    *,
    num_points=8,
    action_dim=4,
    state_dim=9,
    state_key="agent_pos",
    point_channels=3,
    omit=(),
    episode_ends=None,
    layout="root",
    extra_arrays=(),
):
    """Write a synthetic store.

    ``episodes`` is a list of ``(length, base)`` pairs; frame ``t`` of an episode
    is filled with ``base + t`` so every window can be traced back to its frames.
    ``layout="released"`` mirrors the collector output: arrays under ``data/`` and
    ``episode_ends`` under ``meta/``.
    """
    lengths = [length for length, _ in episodes]
    bases = np.repeat(
        [base for _, base in episodes], lengths
    ).astype(np.float32)
    offsets = np.concatenate(
        [np.arange(length, dtype=np.float32) for length, _ in episodes]
    )
    values = bases + offsets
    total = int(values.shape[0])

    action = np.repeat(values[:, None], action_dim, axis=1).astype(np.float32)
    state = np.repeat(values[:, None], state_dim, axis=1).astype(np.float32)
    point_cloud = np.repeat(
        values[:, None, None], num_points, axis=1
    ).repeat(point_channels, axis=2).astype(np.float32)
    ends = np.cumsum(lengths).astype(np.int64) if episode_ends is None else np.asarray(
        episode_ends, dtype=np.int64
    )

    root = zarr.open(str(path), mode="w")
    arrays = {
        "action": action,
        "point_cloud": point_cloud,
        state_key: state,
    }
    if layout == "root":
        container = root
    elif layout == "released":
        container = root.create_group("data")
        meta = root.create_group("meta")
        meta.create_dataset("episode_ends", data=ends, chunks=ends.shape)
        for key in extra_arrays:
            value = np.zeros((total, 2), dtype=np.float32)
            container.create_dataset(key, data=value, chunks=value.shape)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    for key, value in arrays.items():
        if key in omit:
            continue
        container.create_dataset(key, data=value, chunks=value.shape)
    if layout == "root" and "episode_ends" not in omit:
        root.create_dataset("episode_ends", data=ends, chunks=ends.shape)
    return {
        "action": action,
        "point_cloud": point_cloud,
        "state": state,
        "episode_ends": ends,
        "total": total,
    }


def _dataset(path, **kwargs):
    # Default to inferring the point count so each fixture can choose its own N;
    # the 1024-point contract is covered by its own explicit test.
    kwargs.setdefault("num_points", None)
    kwargs.setdefault("validate_finite", True)
    return MetaworldZarrDataset(path, **kwargs)


# --------------------------------------------------------------------- windows
def test_window_count_matches_released_sampler(tmp_path: Path):
    path = tmp_path / "store.zarr"
    _write_store(path, [(3, 0.0), (5, 100.0), (4, 200.0)])
    dataset = _dataset(path, horizon=4, n_obs_steps=2, n_action_steps=4, val_ratio=0.0)

    expected = 0
    for length in (3, 5, 4):
        min_start = -1  # pad_before = n_obs_steps - 1
        max_start = length - 4 + 3  # horizon, pad_after = n_action_steps - 1
        expected += max(0, max_start - min_start + 1)
    assert dataset.num_windows == expected == 15
    assert dataset.n_episodes == 3


def test_windows_pad_by_repeating_episode_edges(tmp_path: Path):
    path = tmp_path / "store.zarr"
    store = _write_store(path, [(3, 0.0), (5, 100.0)], num_points=4)
    dataset = _dataset(path, horizon=4, n_obs_steps=2, n_action_steps=4, val_ratio=0.0)

    state = store["state"][:, 0]
    checked_padding = 0
    for row, (start, end, sample_start, sample_end) in enumerate(dataset.indices):
        assert sample_start >= 0 and sample_end <= dataset.horizon
        window = dataset[row]["obs"]["agent_pos"][:, 0]
        if sample_start > 0:
            np.testing.assert_allclose(
                window[:sample_start], state[start], err_msg=f"row {row}"
            )
            checked_padding += 1
        if sample_end < dataset.horizon:
            np.testing.assert_allclose(
                window[sample_end:], state[end - 1], err_msg=f"row {row}"
            )
            checked_padding += 1
    assert checked_padding > 0


def test_item_shapes_dtype_and_horizons(tmp_path: Path):
    path = tmp_path / "store.zarr"
    _write_store(path, [(6, 0.0)], num_points=1024)
    dataset = MetaworldZarrDataset(
        path, horizon=4, n_obs_steps=2, n_action_steps=4, val_ratio=0.0
    )
    sample = dataset[0]
    assert set(sample) == {"obs", "action"}
    assert set(sample["obs"]) == {"point_cloud", "agent_pos"}
    assert tuple(sample["obs"]["point_cloud"].shape) == (2, 1024, 3)
    assert tuple(sample["obs"]["agent_pos"].shape) == (2, 9)
    assert tuple(sample["action"].shape) == (4, 4)
    for key in ("point_cloud", "agent_pos"):
        assert sample["obs"][key].dtype == np.float32
    assert sample["action"].dtype == np.float32


def test_state_alias_matches_released_field_name(tmp_path: Path):
    path = tmp_path / "store.zarr"
    _write_store(path, [(5, 0.0)], num_points=4, state_key="state")
    dataset = _dataset(path, val_ratio=0.0)
    assert dataset.state_key == "state"
    sample = dataset[0]
    assert "agent_pos" in sample["obs"]


def test_build_window_indices_rejects_mismatched_mask():
    with pytest.raises(ValueError, match="same length"):
        build_window_indices(
            np.asarray([3, 5]), sequence_length=4, episode_mask=np.asarray([True])
        )


# ------------------------------------------------------------------ split rules
def test_validation_split_is_episode_level_and_disjoint(tmp_path: Path):
    path = tmp_path / "store.zarr"
    _write_store(path, [(6, 0.0), (6, 100.0), (6, 200.0), (6, 300.0), (6, 400.0)])
    train = _dataset(path, val_ratio=0.4, seed=42)
    mask = episode_validation_mask(5, 0.4, 42)
    assert np.array_equal(train.val_mask, mask)
    assert not np.any(train.train_mask & train.val_mask)
    assert np.all(train.train_mask | train.val_mask)
    for row in range(train.num_windows):
        assert train.train_mask[train.episode_of_window(row)]

    val = train.get_validation_dataset()
    assert np.array_equal(val.episode_mask, train.val_mask)
    for row in range(val.num_windows):
        assert val.val_mask[val.episode_of_window(row)]
    assert train.num_windows + val.num_windows > 0


def test_validation_dataset_and_downsampling_are_reproducible(tmp_path: Path):
    path = tmp_path / "store.zarr"
    _write_store(path, [(6, 0.0), (6, 100.0), (6, 200.0), (6, 300.0)])
    first = _dataset(path, val_ratio=0.25, max_train_episodes=2, seed=7)
    second = _dataset(path, val_ratio=0.25, max_train_episodes=2, seed=7)
    assert np.array_equal(first.train_mask, second.train_mask)
    assert int(first.train_mask.sum()) == 2
    with pytest.raises(ValueError, match="validation split is empty"):
        _dataset(path, val_ratio=0.0).get_validation_dataset()


# --------------------------------------------------------------------- failures
def test_wrong_point_dimension_fails_fast(tmp_path: Path):
    path = tmp_path / "store.zarr"
    _write_store(path, [(5, 0.0)], point_channels=4)
    with pytest.raises(ValueError, match="trailing dimension must be 3"):
        _dataset(path, num_points=8)


def test_wrong_point_count_fails_fast(tmp_path: Path):
    path = tmp_path / "store.zarr"
    _write_store(path, [(5, 0.0)], num_points=512)
    with pytest.raises(ValueError, match="must contain 1024 points"):
        MetaworldZarrDataset(path, horizon=4, n_obs_steps=2, n_action_steps=4)
    # Inferred point count is an explicit escape hatch, not a silent default.
    inferred = MetaworldZarrDataset(
        path, horizon=4, n_obs_steps=2, n_action_steps=4, num_points=None
    )
    assert tuple(inferred[0]["obs"]["point_cloud"].shape)[1] == 512


def test_empty_episode_fails_fast(tmp_path: Path):
    path = tmp_path / "store.zarr"
    _write_store(path, [(3, 0.0), (0, 0.0), (4, 100.0)], episode_ends=[3, 3, 7])
    with pytest.raises(ValueError, match="empty episodes"):
        _dataset(path)


def test_missing_required_arrays_fail_fast(tmp_path: Path):
    missing_action = tmp_path / "no-action.zarr"
    _write_store(missing_action, [(5, 0.0)], omit=("action",))
    with pytest.raises(KeyError, match="action"):
        _dataset(missing_action)

    missing_state = tmp_path / "no-state.zarr"
    _write_store(missing_state, [(5, 0.0)], omit=("agent_pos",))
    with pytest.raises(KeyError, match="robot state"):
        _dataset(missing_state)


def test_episode_ends_must_match_frames(tmp_path: Path):
    path = tmp_path / "store.zarr"
    _write_store(path, [(3, 0.0), (3, 100.0)], episode_ends=[3, 7])
    with pytest.raises(ValueError, match="episode_ends ends at 7"):
        _dataset(path)


def test_invalid_construction_arguments_are_rejected(tmp_path: Path):
    path = tmp_path / "store.zarr"
    _write_store(path, [(5, 0.0)])
    with pytest.raises(ValueError, match="mode must be"):
        _dataset(path, mode="test")
    with pytest.raises(ValueError, match="n_obs_steps must not exceed horizon"):
        _dataset(path, horizon=2, n_obs_steps=3)
    with pytest.raises(ValueError, match="n_action_steps must not exceed horizon"):
        _dataset(path, horizon=2, n_action_steps=3)


# ------------------------------------------------------------------- scaling
def test_normalizer_is_fitted_from_training_episodes_only(tmp_path: Path):
    path = tmp_path / "store.zarr"
    mask = episode_validation_mask(3, 0.34, 42)
    assert mask.sum() == 1
    bases = [0.0, 10.0, 20.0]
    bases[int(np.flatnonzero(mask)[0])] = 1_000_000.0
    store = _write_store(path, [(6, base) for base in bases], num_points=4)
    dataset = _dataset(path, val_ratio=0.34, seed=42)
    assert np.array_equal(dataset.val_mask, mask)
    train_spans = [dataset.episode_span(i) for i in np.flatnonzero(~mask)]
    train_values = np.concatenate(
        [store["state"][start:end, 0] for start, end in train_spans]
    )

    normalizer = dataset.get_normalizer()
    assert set(normalizer.keys()) == {"action", "agent_pos", "point_cloud"}
    stats = normalizer["agent_pos"].get_input_stats()
    input_max = float(np.asarray(stats["max"].numpy()).max())
    input_min = float(np.asarray(stats["min"].numpy()).min())
    # The outlier validation episode (base 1e6) must not shift the statistics.
    assert input_max < 1e3
    np.testing.assert_allclose(input_max, float(train_values.max()), rtol=0, atol=1e-4)
    np.testing.assert_allclose(input_min, float(train_values.min()), rtol=0, atol=1e-4)
    if mask[2]:
        assert input_max < 1e5


def test_get_normalizer_rejects_unknown_mode(tmp_path: Path):
    path = tmp_path / "store.zarr"
    _write_store(path, [(5, 0.0)])
    with pytest.raises(NotImplementedError, match="mode='limits'"):
        _dataset(path).get_normalizer(mode="gaussian")


# --------------------------------------------------------------- jittor/batches
def test_dataloader_batches_nested_dict(tmp_path: Path):
    path = tmp_path / "store.zarr"
    _write_store(path, [(6, 0.0), (6, 100.0)], num_points=16)
    dataset = _dataset(path, val_ratio=0.0)
    loader = jt.dataset.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=False
    )
    batches = list(loader)
    assert batches
    batch = batches[0]
    assert tuple(batch["obs"]["point_cloud"].shape) == (2, 2, 16, 3)
    assert tuple(batch["obs"]["agent_pos"].shape) == (2, 2, 9)
    assert tuple(batch["action"].shape) == (2, 4, 4)
    assert batch["action"].dtype == jt.float32


def test_len_follows_jittor_batch_semantics(tmp_path: Path):
    path = tmp_path / "store.zarr"
    _write_store(path, [(6, 0.0), (6, 100.0), (6, 200.0)])
    dataset = _dataset(path, val_ratio=0.0)
    assert dataset.num_windows > 0
    expected_batches = -(-dataset.num_windows // dataset.batch_size)
    assert len(dataset) == expected_batches
    dataset.set_attrs(batch_size=1)
    assert len(dataset) == dataset.num_windows


def test_released_horizon_batch_feeds_policy_loss(tmp_path: Path):
    from flowpolicy_jittor.model import FlowPolicy

    path = tmp_path / "store.zarr"
    _write_store(path, [(6, 0.0), (6, 100.0)], num_points=32)
    dataset = _dataset(path, val_ratio=0.0)
    loader = jt.dataset.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )
    batch = next(iter(loader))

    policy = FlowPolicy(
        shape_meta={
            "obs": {
                "point_cloud": {"shape": [32, 3], "type": "point_cloud"},
                "agent_pos": {"shape": [9], "type": "low_dim"},
            },
            "action": {"shape": [4]},
        },
        horizon=4,
        n_action_steps=4,
        n_obs_steps=2,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16),
        kernel_size=3,
        n_groups=4,
        encoder_output_dim=8,
        state_mlp_size=(8, 8),
        pointcloud_encoder_cfg={
            "in_channels": 3,
            "out_channels": 8,
            "use_layernorm": True,
            "final_norm": "layernorm",
            "normal_channel": False,
        },
        var_fc_hidden_dim=8,
    )
    policy.set_normalizer(dataset.get_normalizer())
    loss, logs = policy.compute_loss(
        batch,
        noise=jt.zeros((1, 4, 4), dtype="float32"),
        t=jt.array([0.3], dtype="float32"),
    )
    assert np.isfinite(float(loss.item()))
    assert np.isfinite(float(logs["endpoint_loss"].item()))


# ------------------------------------------------- released on-disk layout
def test_released_group_layout_loads_like_the_flat_layout(tmp_path: Path):
    flat_path = tmp_path / "flat.zarr"
    released_path = tmp_path / "released.zarr"
    episodes = [(5, 0.0), (6, 100.0), (4, 200.0)]
    _write_store(flat_path, episodes, num_points=16)
    _write_store(
        released_path,
        episodes,
        num_points=16,
        layout="released",
        extra_arrays=("img", "depth", "full_state"),
    )

    flat = _dataset(flat_path, val_ratio=0.0)
    released = _dataset(released_path, val_ratio=0.0)

    assert released.state_key == flat.state_key
    assert np.array_equal(released.episode_ends, flat.episode_ends)
    assert np.array_equal(released.indices, flat.indices)
    assert released.num_windows == flat.num_windows
    for row in range(flat.num_windows):
        left = released[row]
        right = flat[row]
        np.testing.assert_array_equal(left["action"], right["action"])
        np.testing.assert_array_equal(
            left["obs"]["point_cloud"], right["obs"]["point_cloud"]
        )
        np.testing.assert_array_equal(left["obs"]["agent_pos"], right["obs"]["agent_pos"])


def test_released_layout_can_keep_state_under_the_data_group(tmp_path: Path):
    path = tmp_path / "released.zarr"
    _write_store(path, [(5, 0.0), (5, 50.0)], num_points=16, layout="released", state_key="state")
    dataset = _dataset(path, val_ratio=0.0)
    assert dataset.state_key == "state"
    assert dataset.num_windows > 0


def test_missing_episode_ends_names_the_supported_locations(tmp_path: Path):
    path = tmp_path / "broken.zarr"
    _write_store(path, [(5, 0.0)], num_points=16, layout="released")
    store = zarr.open(str(path), mode="a")
    del store["meta"]["episode_ends"]
    with pytest.raises(KeyError, match="episode_ends"):
        _dataset(path, val_ratio=0.0)
