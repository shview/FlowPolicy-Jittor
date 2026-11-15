import numpy as np
import pytest

jt = pytest.importorskip("jittor")

from flowpolicy_jittor.model.normalizer import (  # noqa: E402
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = previous


def _limits_data():
    return np.asarray(
        [
            [[-2.0, 0.0, 10.0], [0.0, 2.0, 14.0]],
            [[2.0, 4.0, 18.0], [1.0, 1.0, 12.0]],
        ],
        dtype=np.float32,
    )


def test_limits_maps_each_feature_min_and_max():
    data = _limits_data()
    normalizer = SingleFieldLinearNormalizer.create_fit(data)
    normalized = normalizer.normalize(data).numpy()

    flat = normalized.reshape(-1, normalized.shape[-1])
    np.testing.assert_allclose(flat.min(axis=0), -1.0, atol=1e-6)
    np.testing.assert_allclose(flat.max(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        normalizer.unnormalize(normalized).numpy(), data, rtol=1e-6, atol=1e-6
    )


def test_limits_support_batch_and_time_dimensions():
    data = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    normalizer = SingleFieldLinearNormalizer.create_fit(data)
    result = normalizer(data)
    assert tuple(result.shape) == (2, 4, 3)


def test_constant_feature_maps_to_midpoint_and_roundtrips():
    data = np.asarray(
        [[1.0, 5.0], [3.0, 5.0], [7.0, 5.0]], dtype=np.float32
    )
    normalizer = SingleFieldLinearNormalizer.create_fit(data)
    normalized = normalizer.normalize(data)
    np.testing.assert_allclose(normalized.numpy()[:, 1], 0.0, atol=1e-7)
    np.testing.assert_allclose(
        normalizer.unnormalize(normalized).numpy(), data, atol=1e-6
    )


def test_identity_broadcasts_over_point_cloud_shape():
    points = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    identity = SingleFieldLinearNormalizer.create_identity()
    np.testing.assert_array_equal(identity.normalize(points).numpy(), points)
    np.testing.assert_array_equal(identity.unnormalize(points).numpy(), points)


def test_input_gradient_matches_scale_and_stats_are_stopped():
    normalizer = SingleFieldLinearNormalizer.create_fit(_limits_data())
    value = jt.array(_limits_data())
    gradient = jt.grad(normalizer.normalize(value).sum(), value)
    expected = np.broadcast_to(
        normalizer.scale.numpy(), tuple(map(int, value.shape))
    )
    np.testing.assert_allclose(gradient.numpy(), expected, atol=1e-6)
    assert all(
        item.is_stop_grad()
        for item in (
            normalizer.scale,
            normalizer.offset,
            normalizer.input_min,
            normalizer.input_max,
            normalizer.input_mean,
            normalizer.input_std,
        )
    )


def test_optional_identity_point_cloud_path():
    action = np.asarray([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    state = np.asarray([[-2.0, 0.0, 3.0], [2.0, 4.0, 5.0]], dtype=np.float32)
    points = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)

    normalizer = LinearNormalizer().fit(
        {"action": action, "agent_pos": state}
    )
    normalizer["point_cloud"] = SingleFieldLinearNormalizer.create_identity()

    normalized_obs = normalizer.normalize(
        {"agent_pos": state[None, ...], "point_cloud": points[None, ...]}
    )
    assert tuple(normalized_obs["agent_pos"].shape) == (1, 2, 3)
    np.testing.assert_array_equal(
        normalized_obs["point_cloud"].numpy(), points[None, ...]
    )
    normalized_action = normalizer["action"].normalize(action[None, ...])
    np.testing.assert_allclose(
        normalizer["action"].unnormalize(normalized_action).numpy(),
        action[None, ...],
        atol=1e-6,
    )


def test_official_policy_contract_fits_all_three_fields_with_limits():
    action = np.asarray([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    state = np.asarray([[-2.0, 0.0, 3.0], [2.0, 4.0, 5.0]], dtype=np.float32)
    points = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    normalizer = LinearNormalizer().fit(
        {"action": action, "agent_pos": state, "point_cloud": points}
    )
    for field, value in {
        "action": action,
        "agent_pos": state,
        "point_cloud": points,
    }.items():
        normalized = normalizer[field].normalize(value).numpy()
        flat = normalized.reshape((-1, normalized.shape[-1]))
        np.testing.assert_allclose(flat.min(axis=0), -1.0, atol=1e-6)
        np.testing.assert_allclose(flat.max(axis=0), 1.0, atol=1e-6)


def test_initialized_field_rejects_loaded_feature_dimension_change():
    existing = SingleFieldLinearNormalizer.create_fit(
        np.zeros((2, 4), dtype=np.float32)
    )
    other = SingleFieldLinearNormalizer.create_fit(
        np.zeros((2, 3), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="feature dimension"):
        existing.load_state_dict(other.state_dict(to="numpy"))


def test_prefixed_parent_state_rebuilds_dynamic_fields_strictly():
    source = LinearNormalizer().fit(
        {
            "action": np.asarray([[-1.0, 1.0], [2.0, 3.0]], dtype=np.float32),
            "agent_pos": np.asarray(
                [[-2.0, 0.0, 2.0], [4.0, 6.0, 8.0]], dtype=np.float32
            ),
            "point_cloud": np.arange(18, dtype=np.float32).reshape(2, 3, 3),
        }
    )
    parent_state = {
        **{f"normalizer.{key}": value for key, value in source.state_dict(to="numpy").items()},
        "model.weight": np.ones((2, 2), dtype=np.float32),
    }
    restored = LinearNormalizer()
    consumed = restored.load_prefixed_state_dict(parent_state, prefix="normalizer")
    assert consumed == {key for key in parent_state if key.startswith("normalizer.")}
    assert set(restored.keys()) == {"action", "agent_pos", "point_cloud"}
    assert "model.weight" not in consumed


def test_prefixed_parent_state_requires_matching_prefix():
    with pytest.raises(KeyError, match="no normalizer state"):
        LinearNormalizer().load_prefixed_state_dict(
            {"model.weight": np.ones((2, 2), dtype=np.float32)},
            prefix="normalizer.",
        )


def test_registered_fields_survive_state_dict_roundtrip():
    data = {
        "action": np.asarray([[-1.0, 2.0], [3.0, 6.0]], dtype=np.float32),
        "agent_pos": np.asarray(
            [[-2.0, 1.0, 4.0], [2.0, 5.0, 8.0]], dtype=np.float32
        ),
    }
    source = LinearNormalizer().fit(data)
    source["point_cloud"] = SingleFieldLinearNormalizer.create_identity()
    state = source.state_dict(to="numpy")
    assert set(state) == {
        f"normalizer_{field}.{name}"
        for field in ("action", "agent_pos", "point_cloud")
        for name in (
            "scale",
            "offset",
            "input_min",
            "input_max",
            "input_mean",
            "input_std",
        )
    }

    restored = LinearNormalizer()
    restored.load_state_dict(state)
    assert set(restored.keys()) == {"action", "agent_pos", "point_cloud"}
    for key, value in data.items():
        np.testing.assert_allclose(
            restored[key].normalize(value).numpy(),
            source[key].normalize(value).numpy(),
            atol=1e-7,
        )
    assert restored.trainable_parameters() == []


def test_default_field_path_roundtrips():
    data = np.asarray([[-2.0, 0.0], [2.0, 4.0]], dtype=np.float32)
    normalizer = LinearNormalizer().fit(data)
    normalized = normalizer.normalize(data)
    np.testing.assert_allclose(normalizer.unnormalize(normalized).numpy(), data)


def test_missing_field_and_wrong_feature_shape_fail_early():
    normalizer = LinearNormalizer().fit(
        {"action": np.zeros((2, 4), dtype=np.float32)}
    )
    with pytest.raises(KeyError, match="no fields"):
        normalizer.normalize({"agent_pos": np.zeros((2, 4), dtype=np.float32)})
    with pytest.raises(ValueError, match="final feature dimension"):
        normalizer["action"].normalize(np.zeros((2, 3), dtype=np.float32))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"mode": "gaussian"}, NotImplementedError),
        ({"last_n_dims": 2}, NotImplementedError),
        ({"fit_offset": False}, NotImplementedError),
    ],
)
def test_unvalidated_fit_modes_are_rejected(kwargs, error):
    with pytest.raises(error):
        SingleFieldLinearNormalizer.create_fit(_limits_data(), **kwargs)


def test_non_finite_fit_data_is_rejected():
    data = _limits_data()
    data[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SingleFieldLinearNormalizer.create_fit(data)


def test_load_state_dict_is_strict():
    normalizer = SingleFieldLinearNormalizer.create_fit(_limits_data())
    state = normalizer.state_dict(to="numpy")
    state.pop("offset")
    with pytest.raises(KeyError, match="state keys"):
        SingleFieldLinearNormalizer().load_state_dict(state)

