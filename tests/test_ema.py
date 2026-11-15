"""Tests for stored EMA shadows and the detached EMA target policy."""

import numpy as np
import pytest

jt = pytest.importorskip("jittor")

from flowpolicy_jittor.model import FlowPolicy, LinearNormalizer  # noqa: E402
from flowpolicy_jittor.training import (  # noqa: E402
    DetachedEmaTarget,
    EmaModel,
    Trainer,
)


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


def _policy(num_points=8, **overrides):
    config = dict(
        shape_meta=_shape_meta(num_points),
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
    )
    config.update(overrides)
    return FlowPolicy(**config)


def _batch(seed=5, batch_size=2, num_points=8):
    rng = np.random.default_rng(seed)
    return {
        "obs": {
            "point_cloud": rng.normal(size=(batch_size, 2, num_points, 3)).astype(np.float32),
            "agent_pos": rng.normal(size=(batch_size, 2, 9)).astype(np.float32),
        },
        "action": rng.normal(size=(batch_size, 4, 4)).astype(np.float32),
    }


def _ready_policy(num_points=8, **overrides):
    data = _batch(num_points=num_points)
    policy = _policy(num_points=num_points, **overrides)
    policy.set_normalizer(
        LinearNormalizer().fit(
            {
                "action": data["action"],
                "agent_pos": data["obs"]["agent_pos"],
                "point_cloud": data["obs"]["point_cloud"],
            }
        )
    )
    return policy, data


def _weights(policy):
    return {name: np.asarray(value).copy() for name, value in policy.state_dict(to="numpy").items()}


def test_ema_excludes_normalizer_and_var_est():
    policy, _ = _ready_policy()
    ema = EmaModel(policy, decay=0.9)
    names = set(ema.state_dict()["names"])
    assert names, "EMA must track at least one weight"
    # Nested paths must be excluded too: the policy prefixes the frozen branch with
    # ``model.``, so a prefix-only filter would silently average it.
    assert not any("normalizer" in name.split(".") for name in names)
    assert not any("var_est" in name.split(".") for name in names)
    # The encoder and the U-Net are both covered (not just one of them).
    assert any(name.startswith("obs_encoder") for name in names)
    assert any(name.startswith("model.") for name in names)


def test_trainer_updates_ema_and_the_shadow_is_an_average():
    policy, data = _ready_policy()
    before = _weights(policy)
    trainer = Trainer(
        policy, total_steps=4, lr=1e-2, warmup_steps=1, seed=0, ema_decay=0.5
    )
    trainer.fit(data, steps=3)
    assert trainer.ema is not None
    assert trainer.ema.optimization_step == 3
    assert all("ema_decay" in record for record in trainer.history)

    after_policy = _weights(policy)
    state = trainer.ema.state_dict()
    moved = 0
    for name, value in zip(state["names"], state["parameters"]):
        if np.array_equal(value, before[name]):
            continue
        moved += 1
        # The average must be an average: neither the untouched initialisation nor the
        # current online weight (unless the two happen to coincide).
        if not np.array_equal(after_policy[name], before[name]):
            assert not np.array_equal(value, after_policy[name]), f"{name} was copied, not averaged"
    assert moved >= len(state["names"]) // 2, f"only {moved}/{len(state['names'])} tensors moved"


def test_standalone_update_folds_the_current_weights_in():
    policy, data = _ready_policy()
    ema = EmaModel(policy, decay=0.5)
    trainer = Trainer(policy, total_steps=4, lr=1e-2, warmup_steps=1, seed=0)
    trainer.fit(data, steps=2)
    before = dict(zip(ema.state_dict()["names"], ema.state_dict()["parameters"]))
    assert ema.optimization_step == 0
    decay = ema.update(policy)
    assert 0.0 < decay <= 0.5
    assert ema.optimization_step == 1
    after = dict(zip(ema.state_dict()["names"], ema.state_dict()["parameters"]))
    changed = sum(1 for name in before if not np.array_equal(before[name], after[name]))
    assert changed >= len(before) // 2, f"only {changed}/{len(before)} tensors changed"


def test_effective_decay_ramps_up_and_respects_the_cap():
    policy, _ = _ready_policy()
    ema = EmaModel(policy, decay=0.999)
    decays = [ema.effective_decay()]
    for _ in range(30):
        ema.update(policy)
        decays.append(ema.effective_decay())
    assert decays[0] == pytest.approx(0.1, abs=1e-6)
    assert all(later >= earlier for earlier, later in zip(decays, decays[1:]))
    assert max(decays) <= 0.999
    assert decays[-1] > 0.7


def test_copy_store_and_restore_are_exact_round_trips():
    policy, data = _ready_policy()
    trainer = Trainer(policy, total_steps=3, lr=1e-2, warmup_steps=1, seed=1)
    trainer.fit(data, steps=2)
    ema = EmaModel(policy, decay=0.9)
    ema.update(policy)

    online_before = _weights(policy)
    ema.store(policy)
    ema.copy_to(policy)
    ema_weights = _weights(policy)
    assert any(
        not np.array_equal(online_before[name], ema_weights[name]) for name in online_before
    ), "copy_to must actually change the online weights"
    ema.restore(policy)
    restored = _weights(policy)
    for name, value in online_before.items():
        assert np.array_equal(restored[name], value), f"{name} was not restored exactly"


def test_ema_state_dict_round_trip_and_validation():
    policy, data = _ready_policy()
    ema = EmaModel(policy, decay=0.9)
    trainer = Trainer(policy, total_steps=3, lr=1e-2, warmup_steps=1, seed=2)
    trainer.fit(data, steps=2)
    ema.update(policy)
    state = ema.state_dict()

    fresh = EmaModel(policy, decay=0.5)
    fresh.load_state_dict(state)
    assert fresh.optimization_step == state["optimization_step"]
    assert fresh.state_dict()["decay"] == pytest.approx(0.9)
    for left, right in zip(fresh.state_dict()["parameters"], state["parameters"]):
        assert np.array_equal(left, right)

    broken = dict(state)
    broken["parameters"] = state["parameters"][:-1]
    with pytest.raises(ValueError, match="parameter list"):
        EmaModel(policy, decay=0.5).load_state_dict(broken)

    renamed = dict(state)
    renamed["names"] = [name + "_x" for name in state["names"]]
    with pytest.raises(ValueError, match="weight names"):
        EmaModel(policy, decay=0.5).load_state_dict(renamed)

    missing = {key: value for key, value in state.items() if key != "decay"}
    with pytest.raises(KeyError, match="EMA state keys"):
        EmaModel(policy, decay=0.5).load_state_dict(missing)


def test_stored_ema_rejects_nonfinite_and_late_metadata_atomically():
    policy, _ = _ready_policy()
    ema = EmaModel(policy, decay=0.9)
    before = ema.state_dict()

    late = ema.state_dict()
    late["parameters"][0] = np.asarray(late["parameters"][0]) + 1.0
    late["optimization_step"] = -1
    with pytest.raises(ValueError, match="optimization_step"):
        ema.load_state_dict(late)
    for left, right in zip(ema.state_dict()["parameters"], before["parameters"]):
        np.testing.assert_array_equal(left, right)

    nonfinite = ema.state_dict()
    nonfinite["parameters"][0].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ema.load_state_dict(nonfinite)
    for left, right in zip(ema.state_dict()["parameters"], before["parameters"]):
        np.testing.assert_array_equal(left, right)

    negative_warmup = ema.state_dict()
    negative_warmup["warmup_steps"] = -7
    with pytest.raises(ValueError, match="warmup_steps"):
        ema.load_state_dict(negative_warmup)
    assert ema.warmup_steps == before["warmup_steps"]


def test_invalid_decay_is_rejected():
    policy, _ = _ready_policy()
    for decay in (0.0, -0.5, 1.5, float("nan")):
        with pytest.raises(ValueError):
            EmaModel(policy, decay=decay)


def test_trainer_checkpoint_carries_and_restores_ema(tmp_path):
    policy, data = _ready_policy()
    trainer = Trainer(policy, total_steps=4, lr=1e-2, warmup_steps=1, seed=3, ema_decay=0.9)
    trainer.fit(data, steps=2)
    path = tmp_path / "trainer.ckpt"
    trainer.save_checkpoint(path)

    other_policy, _ = _ready_policy()
    other = Trainer(other_policy, total_steps=4, lr=1e-2, warmup_steps=1, seed=3, ema_decay=0.9)
    other.load_checkpoint(path)
    assert other.ema is not None
    assert other.ema.optimization_step == 2
    for left, right in zip(other.ema.state_dict()["parameters"], trainer.ema.state_dict()["parameters"]):
        assert np.array_equal(left, right)

    # A trainer without EMA must refuse a checkpoint that carries EMA weights.
    plain_policy, _ = _ready_policy()
    plain = Trainer(plain_policy, total_steps=4, lr=1e-2, warmup_steps=1, seed=3)
    with pytest.raises(ValueError, match="has no EMA"):
        plain.load_checkpoint(path)

    # And an EMA trainer must refuse a checkpoint that has none.
    plain_path = tmp_path / "plain.ckpt"
    plain.fit(data, steps=1)
    plain.save_checkpoint(plain_path)
    with pytest.raises(ValueError, match="no EMA state"):
        Trainer(_ready_policy()[0], total_steps=4, lr=1e-2, warmup_steps=1, seed=3, ema_decay=0.9).load_checkpoint(plain_path)


def test_trainer_without_ema_records_null():
    policy, data = _ready_policy()
    trainer = Trainer(policy, total_steps=2, lr=1e-2, warmup_steps=1, seed=4)
    trainer.fit(data, steps=1)
    assert trainer._trainer_payload()["ema"] is None
    record = trainer.history[-1]
    assert "ema_decay" not in record


def test_detached_target_is_separate_frozen_and_updates_only_through_ema():
    policy, data = _ready_policy(target_mode="detached")
    trainer = Trainer(policy, total_steps=2, lr=1e-2, seed=7, ema_decay=0.5)
    assert isinstance(trainer.ema, DetachedEmaTarget)
    target = trainer.ema.target_policy
    assert target.trainable_parameters() == []
    assert not {
        id(parameter) for parameter in policy.trainable_parameters()
    } & {id(parameter) for parameter in target.parameters()}

    target_before = _weights(target)
    noise, times = trainer._step_inputs(data["action"].shape, 1)
    loss, _ = policy.compute_loss(
        data, noise=noise, t=times, target_policy=target
    )
    trainer.optimizer.step(loss)
    for name, value in target_before.items():
        np.testing.assert_array_equal(_weights(target)[name], value)

    online_after = _weights(policy)
    decay = trainer.ema.update(policy)
    target_after = _weights(target)
    averaged = trainer.ema._names
    moved = 0
    for name in averaged:
        expected = decay * target_before[name] + (1.0 - decay) * online_after[name]
        np.testing.assert_allclose(target_after[name], expected, rtol=1e-5, atol=1e-6)
        moved += int(not np.array_equal(target_after[name], target_before[name]))
    assert moved > 0
    assert trainer.ema.optimization_step == 1
    assert all(parameter.is_stop_grad() for parameter in target.parameters())


def test_detached_trainer_requires_ema_decay():
    policy, _ = _ready_policy(target_mode="detached")
    with pytest.raises(ValueError, match="requires ema_decay"):
        Trainer(policy, total_steps=2)


def test_detached_checkpoint_restores_target_and_continues(tmp_path):
    policy, data = _ready_policy(target_mode="detached")
    trainer = Trainer(
        policy, total_steps=4, lr=1e-2, warmup_steps=1, seed=9, ema_decay=0.9
    )
    trainer.fit(data, steps=2)
    path = tmp_path / "detached.ckpt"
    trainer.save_checkpoint(path)

    other_policy, _ = _ready_policy(target_mode="detached")
    other = Trainer(
        other_policy, total_steps=4, lr=1e-2, warmup_steps=1, seed=9, ema_decay=0.9
    )
    other.load_checkpoint(path)
    assert other.global_step == 2
    assert other.ema.optimization_step == 2
    for name, value in _weights(trainer.ema.target_policy).items():
        np.testing.assert_array_equal(_weights(other.ema.target_policy)[name], value)
    record = other.train_step(data)
    assert record["step"] == 3
    assert other.ema.optimization_step == 3

    code_policy, _ = _ready_policy()
    code_trainer = Trainer(
        code_policy, total_steps=4, lr=1e-2, warmup_steps=1, seed=9, ema_decay=0.9
    )
    with pytest.raises(ValueError, match="target_mode"):
        code_trainer.load_checkpoint(path)

    payload = jt.load(str(path))
    payload["trainer"]["ema"]["kind"] = "stored-shadow"
    incompatible = tmp_path / "wrong-target-kind.ckpt"
    jt.save(payload, str(incompatible))
    fresh_policy, _ = _ready_policy(target_mode="detached")
    fresh = Trainer(
        fresh_policy,
        total_steps=4,
        lr=1e-2,
        warmup_steps=1,
        seed=9,
        ema_decay=0.9,
    )
    online_before = _weights(fresh_policy)
    target_before = _weights(fresh.ema.target_policy)
    optimizer_before = fresh.optimizer_state()
    with pytest.raises(ValueError, match="does not describe a detached target"):
        fresh.load_checkpoint(incompatible)
    assert fresh.global_step == 0
    assert fresh.history == []
    for name, value in online_before.items():
        np.testing.assert_array_equal(_weights(fresh_policy)[name], value)
    for name, value in target_before.items():
        np.testing.assert_array_equal(_weights(fresh.ema.target_policy)[name], value)
    optimizer_after = fresh.optimizer_state()
    assert optimizer_after["n_step"] == optimizer_before["n_step"]
    for key in ("m", "values"):
        for left, right in zip(optimizer_before[key], optimizer_after[key]):
            np.testing.assert_array_equal(left, right)


def test_detached_target_rejects_late_metadata_before_mutating_weights():
    policy, _ = _ready_policy(target_mode="detached")
    trainer = Trainer(policy, total_steps=2, ema_decay=0.9)
    state = trainer.ema.state_dict()
    first_name = next(iter(state["target_policy"]))
    state["target_policy"][first_name] = (
        np.asarray(state["target_policy"][first_name]) + 1.0
    )
    state["optimization_step"] = -1
    before = _weights(trainer.ema.target_policy)
    with pytest.raises(ValueError, match="optimization_step"):
        trainer.ema.load_state_dict(state)
    for name, value in before.items():
        np.testing.assert_array_equal(_weights(trainer.ema.target_policy)[name], value)
