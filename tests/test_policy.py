import importlib.metadata
from pathlib import Path

import numpy as np
import pytest

jt = pytest.importorskip("jittor")

from flowpolicy_jittor import __version__  # noqa: E402
from flowpolicy_jittor.model import FlowPolicy, LinearNormalizer  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = previous


def _shape_meta(num_points=8):
    return {
        "obs": {
            "point_cloud": {"shape": [num_points, 3], "type": "point_cloud"},
            "agent_pos": {"shape": [9], "type": "low_dim"},
        },
        "action": {"shape": [4]},
    }


def _policy(*, action_slice_mode="code", target_mode="code", **overrides):
    config = dict(
        shape_meta=_shape_meta(),
        horizon=4,
        n_action_steps=4,
        n_obs_steps=2,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16, 32),
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
        action_slice_mode=action_slice_mode,
        target_mode=target_mode,
    )
    config.update(overrides)
    return FlowPolicy(**config)


def _data(seed=11, batch_size=2):
    rng = np.random.default_rng(seed)
    obs = {
        "point_cloud": rng.normal(size=(batch_size, 2, 8, 3)).astype(np.float32),
        "agent_pos": rng.normal(size=(batch_size, 2, 9)).astype(np.float32),
    }
    action = rng.normal(size=(batch_size, 4, 4)).astype(np.float32)
    return {"obs": obs, "action": action}


def _normalizer(data):
    return LinearNormalizer().fit(
        {
            "action": data["action"],
            "agent_pos": data["obs"]["agent_pos"],
            "point_cloud": data["obs"]["point_cloud"],
        }
    )


def _ready_policy(**kwargs):
    policy = _policy(**kwargs)
    data = _data()
    policy.set_normalizer(_normalizer(data))
    return policy, data


def _snapshot(policy):
    return {
        key: np.asarray(value).copy()
        for key, value in policy.state_dict(to="numpy").items()
    }


def _assert_snapshot(policy, expected):
    actual = policy.state_dict(to="numpy")
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_compute_loss_integrates_encoder_unet_and_cfm_with_gradients():
    policy, batch = _ready_policy()
    noise = jt.zeros((2, 4, 4), dtype="float32")
    time = jt.array([0.2, 0.7], dtype="float32")
    loss, logs = policy.compute_loss(batch, noise=noise, t=time)
    assert tuple(loss.shape) in {(), (1,)}
    assert set(logs) >= {"bc_loss", "endpoint_loss", "velocity_loss"}
    assert np.isfinite(float(loss.item()))

    active = policy.trainable_parameters()
    stopped = [item for item in policy.parameters() if item.is_stop_grad()]
    assert active
    assert stopped
    assert not {id(item) for item in active} & {id(item) for item in stopped}
    parameter = active[0]
    gradient = jt.grad(loss, parameter)
    assert np.all(np.isfinite(gradient.numpy()))


def test_top_level_adamw_updates_active_but_not_frozen_parameters():
    # Deterministic initialisation: FPJ-030 saw this assertion fail once in 20 runs, and
    # an unseeded parameter draw is one of the two plausible causes (the other is Jittor
    # graph state leaking between tests).  Seeding removes the first.
    jt.set_global_seed(0)
    policy, batch = _ready_policy()
    active = policy.trainable_parameters()
    frozen = [item for item in policy.parameters() if item.is_stop_grad()]
    before_active = [item.numpy().copy() for item in active]
    before_frozen = [item.numpy().copy() for item in frozen]
    optimizer = jt.optim.AdamW(active, lr=1e-3, weight_decay=0.1)
    loss, _ = policy.compute_loss(
        batch,
        noise=jt.zeros((2, 4, 4), dtype="float32"),
        t=jt.array([0.2, 0.7], dtype="float32"),
    )
    loss_value = float(loss.item())
    optimizer.step(loss)
    deltas = [
        float(np.abs(before - after.numpy()).max())
        for before, after in zip(before_active, active)
    ]
    # A measured assertion with diagnostics: an AdamW step at lr=1e-3 moves weights by
    # ~1e-3, so "nothing moved" means the graph was detached, not that the update was tiny.
    assert max(deltas) > 1e-6, (
        f"no trainable parameter moved (loss={loss_value}, max delta={max(deltas):.3e}, "
        f"{len(deltas)} parameters)"
    )
    assert all(
        np.array_equal(before, after.numpy())
        for before, after in zip(before_frozen, frozen)
    )


def test_one_step_prediction_matches_manual_update_and_is_deterministic():
    policy, batch = _ready_policy()
    noise = jt.zeros((2, 4, 4), dtype="float32")
    first = policy.predict_action(batch["obs"], noise=noise)
    second = policy.predict_action(batch["obs"], noise=noise)
    assert first["nfe"] == 1
    assert policy.last_nfe == 1
    assert tuple(first["action_pred"].shape) == (2, 4, 4)
    # Released code starts at n_obs_steps-1, so published 4/2/4 settings
    # return three actions.  This is tested rather than hidden.
    assert tuple(first["action"].shape) == (2, 3, 4)
    np.testing.assert_allclose(
        first["action_pred"].numpy(), second["action_pred"].numpy(), atol=0.0
    )

    _, condition = policy._prepare_observations(batch["obs"])
    time = jt.ones((2,), dtype="float32") * policy.loss_fn.eps
    velocity = policy.model(
        noise,
        time * policy.loss_fn.time_scale,
        global_cond=condition,
    )
    manual = policy.normalizer["action"].unnormalize(noise + velocity)
    np.testing.assert_allclose(first["action_pred"].numpy(), manual.numpy(), atol=1e-6)


def test_exact_slice_mode_returns_configured_action_count():
    policy, batch = _ready_policy(action_slice_mode="exact")
    result = policy.predict_action(
        batch["obs"], noise=jt.zeros((2, 4, 4), dtype="float32")
    )
    assert tuple(result["action"].shape) == (2, 4, 4)


def test_checkpoint_roundtrip_rebuilds_empty_normalizer(tmp_path: Path):
    source, batch = _ready_policy()
    noise = jt.zeros((2, 4, 4), dtype="float32")
    expected = source.predict_action(batch["obs"], noise=noise)["action_pred"].numpy()
    path = tmp_path / "policy.pkl"
    source.save_checkpoint(path, extra={"step": 7})

    restored = _policy()
    assert list(restored.normalizer.keys()) == []
    extra = restored.load_checkpoint(path)
    assert extra == {"step": 7}
    assert set(restored.normalizer.keys()) == {"action", "agent_pos", "point_cloud"}
    actual = restored.predict_action(batch["obs"], noise=noise)["action_pred"].numpy()
    np.testing.assert_allclose(actual, expected, atol=0.0)

    with pytest.raises(FileExistsError):
        source.save_checkpoint(path)


@pytest.mark.parametrize(
    "overrides,expected_field",
    [
        ({"action_slice_mode": "exact"}, "action_slice_mode"),
        ({"eps": 0.2}, "eps"),
        ({"num_inference_steps": 2}, "num_inference_steps"),
    ],
)
def test_checkpoint_rejects_behavior_configuration_mismatch(
    tmp_path: Path, overrides, expected_field
):
    source, _ = _ready_policy()
    path = tmp_path / "policy.pkl"
    source.save_checkpoint(path)
    target = _policy(**overrides)
    with pytest.raises(ValueError, match=expected_field):
        target.load_checkpoint(path)


def test_checkpoint_payload_records_reproduction_identity(tmp_path: Path):
    source, _ = _ready_policy()
    path = tmp_path / "policy.pkl"
    source.save_checkpoint(path)
    payload = jt.load(str(path))
    assert payload["schema_version"] == 2
    assert payload["implementation_version"] == __version__
    assert payload["upstream_commit"] == (
        "86d10c14972b4eab083abec9026749c340f8e5ed"
    )
    assert payload["jittor_version"] == importlib.metadata.version("jittor")
    assert payload["config"]["action_slice_mode"] == "code"
    assert payload["config"]["eps"] == pytest.approx(0.01)


def test_strict_policy_state_rejects_missing_and_wrong_shape():
    source, _ = _ready_policy()
    state = source.state_dict(to="numpy")
    missing = dict(state)
    missing.pop(next(key for key in missing if key.startswith("model.")))
    target, target_data = _ready_policy()
    before = {
        key: value.copy()
        for key, value in target.state_dict(to="numpy").items()
    }
    with pytest.raises(KeyError, match="keys mismatch"):
        target.load_policy_state_dict(missing)
    after = target.state_dict(to="numpy")
    assert set(after) == set(before)
    for key in before:
        np.testing.assert_array_equal(after[key], before[key])
    # The failed load must leave the target usable with its original statistics.
    target.predict_action(
        target_data["obs"], noise=jt.zeros((2, 4, 4), dtype="float32")
    )

    wrong = dict(state)
    key = next(key for key in wrong if key.startswith("model.") and wrong[key].ndim > 0)
    wrong[key] = wrong[key].reshape(-1)[:1]
    with pytest.raises(ValueError, match="shape mismatch"):
        _policy().load_policy_state_dict(wrong)


def test_normalizer_contract_and_shape_errors_fail_early():
    policy = _policy()
    data = _data()
    incomplete = LinearNormalizer().fit(
        {"action": data["action"], "agent_pos": data["obs"]["agent_pos"]}
    )
    with pytest.raises(KeyError, match="exactly"):
        policy.set_normalizer(incomplete)

    wrong_action = LinearNormalizer().fit(
        {
            "action": np.zeros((2, 4, 3), dtype=np.float32),
            "agent_pos": data["obs"]["agent_pos"],
            "point_cloud": data["obs"]["point_cloud"],
        }
    )
    with pytest.raises(ValueError, match="feature dimensions"):
        policy.set_normalizer(wrong_action)
    with pytest.raises(RuntimeError, match="not initialized"):
        policy.predict_action(data["obs"])

    policy.set_normalizer(_normalizer(data))
    bad_points = dict(data["obs"])
    bad_points["point_cloud"] = bad_points["point_cloud"][:, :, :7]
    with pytest.raises(ValueError, match="trailing shape"):
        policy.predict_action(bad_points)
    bad_action = data["action"][:, :3]
    with pytest.raises(ValueError, match="horizon"):
        policy.compute_loss({"obs": data["obs"], "action": bad_action})


def test_unvalidated_policy_modes_are_rejected():
    kwargs = dict(
        shape_meta=_shape_meta(),
        horizon=4,
        n_action_steps=4,
        n_obs_steps=2,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16, 32),
        n_groups=4,
        encoder_output_dim=8,
        state_mlp_size=(8, 8),
        var_fc_hidden_dim=8,
    )
    with pytest.raises(NotImplementedError, match="obs_as_global_cond"):
        FlowPolicy(**kwargs, obs_as_global_cond=False)
    with pytest.raises(ValueError, match="action_slice_mode"):
        FlowPolicy(**kwargs, action_slice_mode="silent-fix")
    with pytest.raises(ValueError, match="target_mode"):
        FlowPolicy(**kwargs, target_mode="implicit-ema")


def test_two_dimensional_action_shape_is_rejected():
    shape_meta = _shape_meta()
    shape_meta["action"]["shape"] = [2, 2]
    with pytest.raises(NotImplementedError, match="one-dimensional"):
        _policy(shape_meta=shape_meta)


def test_predict_action_result_is_stop_gradient():
    policy, batch = _ready_policy()
    result = policy.predict_action(
        batch["obs"], noise=jt.zeros((2, 4, 4), dtype="float32")
    )
    assert result["action"].is_stop_grad()
    assert result["action_pred"].is_stop_grad()


def test_detached_policy_requires_and_uses_a_separate_frozen_target():
    policy, batch = _ready_policy(target_mode="detached")
    with pytest.raises(ValueError, match="requires Trainer"):
        policy.compute_loss(batch)

    target = policy.make_target_policy()
    assert target is not policy
    assert target.obs_encoder is not policy.obs_encoder
    assert target.model is not policy.model
    assert target.trainable_parameters() == []
    loss, logs = policy.compute_loss(
        batch,
        target_policy=target,
        noise=jt.zeros((2, 4, 4), dtype="float32"),
        t=jt.array([0.2, 0.7], dtype="float32"),
    )
    assert np.isfinite(float(loss.item()))
    assert np.isfinite(float(logs["velocity_loss"].item()))
    assert all(parameter.is_stop_grad() for parameter in target.parameters())


def test_code_mode_stays_default_and_rejects_an_unused_target():
    policy, batch = _ready_policy()
    assert policy.loss_fn.target_mode == "code"
    target = policy.make_target_policy()
    with pytest.raises(ValueError, match="does not use"):
        policy.compute_loss(batch, target_policy=target)


def test_detached_target_architecture_mismatch_is_actionable():
    policy, batch = _ready_policy(target_mode="detached")
    target, _ = _ready_policy(down_dims=(8, 24, 32))
    for parameter in target.parameters():
        parameter.stop_grad()
    with pytest.raises(ValueError, match="architecture.*down_dims"):
        policy.compute_loss(batch, target_policy=target)


def test_checkpoint_config_is_detached_and_runtime_mutation_blocks_save(tmp_path):
    policy, _ = _ready_policy()
    exported = policy.checkpoint_config
    exported["eps"] = 0.5
    exported["shape_meta"]["action"][0] = 99
    assert policy.checkpoint_config["eps"] == pytest.approx(0.01)
    assert policy.checkpoint_config["shape_meta"]["action"] == [4]

    policy.action_slice_mode = "exact"
    with pytest.raises(RuntimeError, match="action_slice_mode"):
        policy.save_checkpoint(tmp_path / "mutated.pkl")
    assert not (tmp_path / "mutated.pkl").exists()


def test_loss_runtime_mutation_blocks_save(tmp_path):
    policy, _ = _ready_policy()
    policy.loss_fn.eps = 0.2
    with pytest.raises(RuntimeError, match="eps"):
        policy.save_checkpoint(tmp_path / "mutated-loss.pkl")


@pytest.mark.parametrize(
    "flag",
    ["use_down_condition", "use_mid_condition", "use_up_condition"],
)
def test_unet_condition_flag_mutation_blocks_save(tmp_path: Path, flag):
    policy, _ = _ready_policy()
    destination = tmp_path / f"mutated-{flag}.pkl"
    # The live module and the exported fingerprint must agree before mutation.
    assert policy.checkpoint_config[flag] is True
    assert getattr(policy.model, flag) is True

    setattr(policy.model, flag, False)

    # The fingerprint follows the live module instead of the construction snapshot.
    assert policy.checkpoint_config[flag] is False
    with pytest.raises(RuntimeError, match=flag):
        policy.save_checkpoint(destination)
    assert not destination.exists()
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_invalid_extra_is_rejected_before_checkpoint_state_changes(tmp_path):
    source, _ = _ready_policy()
    valid = tmp_path / "valid.pkl"
    source.save_checkpoint(valid)
    payload = dict(jt.load(str(valid)))
    payload["extra"] = ["not", "a", "mapping"]
    invalid = tmp_path / "invalid-extra.pkl"
    jt.save(payload, str(invalid))

    target, _ = _ready_policy()
    before = _snapshot(target)
    with pytest.raises(TypeError, match="extra must be a mapping"):
        target.load_checkpoint(invalid)
    _assert_snapshot(target, before)


@pytest.mark.parametrize(
    "bad_value,expected_error",
    [
        ("string", TypeError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (-np.inf, ValueError),
    ],
)
def test_invalid_state_dtype_and_values_are_rejected_atomically(
    tmp_path, bad_value, expected_error
):
    source, _ = _ready_policy()
    valid = tmp_path / "valid.pkl"
    source.save_checkpoint(valid)
    payload = dict(jt.load(str(valid)))
    policy_state = dict(payload["policy"])
    key = next(name for name in policy_state if name.startswith("model."))
    original = np.asarray(policy_state[key])
    if bad_value == "string":
        policy_state[key] = np.full(original.shape, "bad", dtype="<U3")
    else:
        corrupt = original.copy()
        corrupt.reshape(-1)[0] = bad_value
        policy_state[key] = corrupt
    payload["policy"] = policy_state
    invalid = tmp_path / f"invalid-{str(bad_value).replace('-', 'minus')}.pkl"
    jt.save(payload, str(invalid))

    target, _ = _ready_policy()
    before = _snapshot(target)
    with pytest.raises(expected_error):
        target.load_checkpoint(invalid)
    _assert_snapshot(target, before)


def test_commit_failure_rolls_back_model_and_normalizer(monkeypatch):
    source, _ = _ready_policy()
    target, _ = _ready_policy()
    before = _snapshot(target)
    state = source.state_dict(to="numpy")
    original_loader = FlowPolicy.load_state_dict
    call_count = {"value": 0}

    def fail_after_first_load(self, values):
        call_count["value"] += 1
        result = original_loader(self, values)
        if call_count["value"] == 1:
            raise RuntimeError("synthetic post-load failure")
        return result

    monkeypatch.setattr(FlowPolicy, "load_state_dict", fail_after_first_load)
    with pytest.raises(RuntimeError, match="synthetic post-load failure"):
        target.load_policy_state_dict(state)
    assert call_count["value"] == 2
    _assert_snapshot(target, before)


def test_atomic_save_preserves_existing_file_on_write_failure(tmp_path, monkeypatch):
    policy, _ = _ready_policy()
    destination = tmp_path / "policy.pkl"
    policy.save_checkpoint(destination)
    before = destination.read_bytes()

    def fail_save(*args, **kwargs):
        raise RuntimeError("synthetic save failure")

    monkeypatch.setattr(jt, "save", fail_save)
    with pytest.raises(RuntimeError, match="synthetic save failure"):
        policy.save_checkpoint(destination, overwrite=True)
    assert destination.read_bytes() == before
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []

