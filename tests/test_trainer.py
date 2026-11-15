"""Tests for the FPJ-024 trainer: schedule, checkpoint, resume and guard rails."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

jt = pytest.importorskip("jittor")

from flowpolicy_jittor.model import FlowPolicy, LinearNormalizer  # noqa: E402
from flowpolicy_jittor.training import (  # noqa: E402
    TRAINER_KIND,
    Trainer,
    warmup_cosine_lr,
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


def _batch(seed=11, batch_size=3, num_points=8):
    rng = np.random.default_rng(seed)
    return {
        "obs": {
            "point_cloud": rng.normal(size=(batch_size, 2, num_points, 3)).astype(
                np.float32
            ),
            "agent_pos": rng.normal(size=(batch_size, 2, 9)).astype(np.float32),
        },
        "action": rng.normal(size=(batch_size, 4, 4)).astype(np.float32),
    }


def _normalizer(data):
    return LinearNormalizer().fit(
        {
            "action": data["action"],
            "agent_pos": data["obs"]["agent_pos"],
            "point_cloud": data["obs"]["point_cloud"],
        }
    )


def _ready_policy(seed=0, num_points=8, **overrides):
    jt.set_global_seed(seed)
    policy = _policy(num_points=num_points, **overrides)
    data = _batch(num_points=num_points)
    policy.set_normalizer(_normalizer(data))
    return policy, data


def _trainer(policy, **overrides):
    config = dict(
        total_steps=6,
        lr=1e-3,
        warmup_steps=2,
        min_lr_ratio=0.1,
        seed=5,
    )
    config.update(overrides)
    return Trainer(policy, **config)


def _snapshot(policy):
    return {key: np.asarray(value).copy() for key, value in policy.state_dict(to="numpy").items()}


# ------------------------------------------------------------------ schedule
def test_warmup_cosine_schedule_is_a_pure_function_of_the_step():
    base, total, warm = 1e-3, 10, 4
    kwargs = dict(base_lr=base, total_steps=total, warmup_steps=warm, min_lr_ratio=0.1)

    assert warmup_cosine_lr(0, **kwargs) == 0.0
    assert warmup_cosine_lr(1, **kwargs) == pytest.approx(base / warm)
    assert warmup_cosine_lr(warm, **kwargs) == pytest.approx(base)
    assert warmup_cosine_lr(total, **kwargs) == pytest.approx(base * 0.1)

    warmup_rates = [warmup_cosine_lr(step, **kwargs) for step in range(1, warm + 1)]
    assert all(left < right for left, right in zip(warmup_rates, warmup_rates[1:]))

    decay_rates = [warmup_cosine_lr(step, **kwargs) for step in range(warm, total + 1)]
    assert all(left > right for left, right in zip(decay_rates, decay_rates[1:]))

    # Same step twice must give the same value (no internal counters).
    assert warmup_cosine_lr(7, **kwargs) == warmup_cosine_lr(7, **kwargs)
    assert warmup_cosine_lr(7, **kwargs) > base * 0.1


def test_schedule_without_warmup_starts_at_the_base_rate():
    assert warmup_cosine_lr(1, base_lr=2e-3, total_steps=5, warmup_steps=0) == pytest.approx(
        2e-3
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(step=0, base_lr=0.0, total_steps=5),
        dict(step=0, base_lr=-1e-3, total_steps=5),
        dict(step=0, base_lr=1e-3, total_steps=0),
        dict(step=6, base_lr=1e-3, total_steps=5),
        dict(step=0, base_lr=1e-3, total_steps=5, warmup_steps=6),
        dict(step=0, base_lr=1e-3, total_steps=5, min_lr_ratio=1.5),
        dict(step=0, base_lr=float("nan"), total_steps=5),
    ],
)
def test_schedule_rejects_invalid_arguments(kwargs):
    with pytest.raises((TypeError, ValueError)):
        warmup_cosine_lr(**kwargs)


# ------------------------------------------------------------------- trainer
def test_trainer_updates_only_trainable_parameters():
    policy, batch = _ready_policy()
    trainable = list(policy.trainable_parameters())
    frozen = [item for item in policy.parameters() if item.is_stop_grad()]
    assert frozen, "the policy should expose stop-grad parameters"
    before_trainable = [item.numpy().copy() for item in trainable]
    before_frozen = [item.numpy().copy() for item in frozen]

    trainer = _trainer(policy)
    assert trainer.optimizer.param_groups[0]["params"] == trainable
    assert len(trainer.optimizer.param_groups[0]["params"]) == len(trainable)
    trainer.fit(batch, steps=3)

    assert any(
        not np.array_equal(before, after.numpy())
        for before, after in zip(before_trainable, trainable)
    )
    assert all(
        np.array_equal(before, after.numpy())
        for before, after in zip(before_frozen, frozen)
    )


def test_trainer_logs_step_learning_rate_and_loss(tmp_path: Path):
    policy, batch = _ready_policy()
    log_path = tmp_path / "steps.jsonl"
    trainer = _trainer(policy, total_steps=6, log_path=log_path, log_every=2)
    records = trainer.fit(batch)

    assert [record["step"] for record in records] == [1, 2, 3, 4, 5, 6]
    assert trainer.global_step == 6
    assert all(math.isfinite(record["loss"]) for record in records)
    assert records[0]["lr"] == pytest.approx(trainer.learning_rate_at(1))
    assert records[-1]["lr"] == pytest.approx(trainer.learning_rate_at(6))
    assert records[-1]["lr"] < records[1]["lr"]  # decay after warmup

    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [entry["step"] for entry in lines] == [2, 4, 6]
    assert lines[-1] == records[-1]


def test_trainer_refuses_to_run_past_total_steps():
    policy, batch = _ready_policy()
    trainer = _trainer(policy, total_steps=2)
    trainer.fit(batch, steps=2)
    with pytest.raises(RuntimeError, match="finished 2 of 2"):
        trainer.train_step(batch)
    with pytest.raises(ValueError, match="only 0 remain"):
        trainer.fit(batch, steps=1)


def test_non_finite_loss_stops_the_run_before_updating_state():
    policy, batch = _ready_policy()
    broken = {
        "obs": dict(batch["obs"]),
        "action": np.array(batch["action"], copy=True),
    }
    broken["action"][0, 0, 0] = np.nan
    trainer = _trainer(policy)

    with pytest.raises(RuntimeError, match="non-finite loss"):
        trainer.train_step(broken)
    assert trainer.global_step == 0
    assert trainer.history == []


# ---------------------------------------------------------------- checkpoint
def test_trainer_checkpoint_roundtrip_restores_schedule_and_ema_marker(tmp_path: Path):
    policy, batch = _ready_policy()
    trainer = _trainer(policy)
    trainer.fit(batch, steps=2)
    path = tmp_path / "trainer.ckpt"
    trainer.save_checkpoint(path)

    payload = jt.load(str(path))
    assert payload["kind"] == TRAINER_KIND
    assert payload["trainer"]["ema"] is None
    assert payload["trainer"]["global_step"] == 2
    assert payload["trainer"]["optimizer"]["n_step"] == 2
    assert set(payload["trainer"]) >= {
        "global_step",
        "optimizer",
        "ema",
        "base_lr",
        "warmup_steps",
        "total_steps",
        "min_lr_ratio",
    }
    assert payload["policy"]["schema_version"] >= 1

    fresh_policy, fresh_batch = _ready_policy()
    resumed = _trainer(fresh_policy)
    resumed.load_checkpoint(path)
    assert resumed.global_step == 2
    assert resumed.learning_rate == pytest.approx(trainer.learning_rate)
    for key, value in _snapshot(policy).items():
        np.testing.assert_array_equal(_snapshot(fresh_policy)[key], value)


def test_legacy_trainer_checkpoint_is_rejected_with_target_semantics_message(
    tmp_path: Path,
):
    policy, batch = _ready_policy()
    trainer = _trainer(policy)
    trainer.fit(batch, steps=1)
    current = tmp_path / "current.ckpt"
    legacy = tmp_path / "legacy.ckpt"
    trainer.save_checkpoint(current)
    payload = jt.load(str(current))
    payload["schema_version"] = 1
    payload["trainer"].pop("target_mode")
    jt.save(payload, str(legacy))
    with pytest.raises(ValueError, match="predates explicit target_mode semantics"):
        _trainer(_ready_policy()[0]).load_checkpoint(legacy)


def test_resume_restores_exact_state_and_continues_the_schedule(tmp_path: Path):
    """Resume must restore the exact checkpointed state and the remaining schedule.

    Bit-exact *trajectory* equality is deliberately not claimed for this path, and
    FPJ-031 measured why: checkpoint state restores bit-exactly and a resume in a fresh
    process is bit-exact, but a save/restore that happens after prior training in the
    same process drifted up to 5.5e-3 in an encoder weight and 9.5e-9 in the loss (three
    extra steps).  Two identical runs inside one process stay bit-exact, so the effect is
    specific to loading a checkpoint into an already-used process.  The tolerance below
    is therefore the promise; `scripts/determinism_probe.py` measures it.
    """
    baseline_policy, baseline_batch = _ready_policy()
    baseline = _trainer(baseline_policy)
    baseline.fit(baseline_batch, steps=6)
    expected = _snapshot(baseline_policy)

    first_policy, first_batch = _ready_policy()
    interrupted = _trainer(first_policy)
    interrupted.fit(first_batch, steps=3)
    saved = _snapshot(first_policy)
    saved_state = interrupted.optimizer_state()
    path = tmp_path / "resume.ckpt"
    interrupted.save_checkpoint(path)

    second_policy, second_batch = _ready_policy()
    resumed = _trainer(second_policy)
    resumed.load_checkpoint(path)

    # Restore is exact: same weights, same AdamW moments, same step counters.
    assert resumed.global_step == 3
    loaded = _snapshot(second_policy)
    for key in saved:
        np.testing.assert_array_equal(loaded[key], saved[key])
    loaded_state = resumed.optimizer_state()
    assert loaded_state["n_step"] == saved_state["n_step"] == 3
    for left, right in zip(loaded_state["m"], saved_state["m"]):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(loaded_state["values"], saved_state["values"]):
        np.testing.assert_array_equal(left, right)

    resumed.fit(second_batch, steps=3)
    assert resumed.global_step == baseline.global_step == 6
    assert [entry["step"] for entry in resumed.history] == [1, 2, 3, 4, 5, 6]
    assert [entry["lr"] for entry in resumed.history] == [
        entry["lr"] for entry in baseline.history
    ]
    for left, right in zip(resumed.history, baseline.history):
        assert left["loss"] == pytest.approx(right["loss"], rel=1e-3, abs=1e-9)

    actual = _snapshot(second_policy)
    assert set(actual) == set(expected)
    for key in expected:
        # Documented tolerance from FPJ-031, not "close enough": measured drift is 5.5e-3.
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-2, err_msg=key)


def test_saving_never_overwrites_silently(tmp_path: Path):
    policy, batch = _ready_policy()
    trainer = _trainer(policy)
    trainer.fit(batch, steps=1)
    path = tmp_path / "trainer.ckpt"
    trainer.save_checkpoint(path)
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        trainer.save_checkpoint(path)
    assert path.read_bytes() == original
    assert [item.name for item in tmp_path.iterdir() if item.name.startswith(".")] == []

    trainer.save_checkpoint(path, overwrite=True)
    assert path.exists()


def test_policy_and_trainer_checkpoints_are_not_interchangeable(tmp_path: Path):
    policy, batch = _ready_policy()
    trainer = _trainer(policy)
    trainer.fit(batch, steps=1)

    policy_path = tmp_path / "policy.ckpt"
    policy.save_checkpoint(policy_path)
    trainer_path = tmp_path / "trainer.ckpt"
    trainer.save_checkpoint(trainer_path)

    with pytest.raises(ValueError, match="not a trainer checkpoint"):
        trainer.load_checkpoint(policy_path)
    with pytest.raises((KeyError, ValueError)):
        policy.load_checkpoint(trainer_path)


def test_resume_rejects_a_different_schedule(tmp_path: Path):
    policy, batch = _ready_policy()
    trainer = _trainer(policy)
    trainer.fit(batch, steps=1)
    path = tmp_path / "trainer.ckpt"
    trainer.save_checkpoint(path)

    other_policy, _ = _ready_policy()
    other = _trainer(other_policy, lr=2e-3)
    with pytest.raises(ValueError, match="schedule does not match"):
        other.load_checkpoint(path)


def test_resume_rejects_a_different_policy_layout(tmp_path: Path):
    policy, batch = _ready_policy()
    trainer = _trainer(policy)
    trainer.fit(batch, steps=1)
    path = tmp_path / "trainer.ckpt"
    trainer.save_checkpoint(path)

    other_policy, _ = _ready_policy(num_points=16)
    other = _trainer(other_policy)
    # Either the policy configuration check or the parameter-layout check may
    # fire first; both must stop the resume before any optimizer state is applied.
    with pytest.raises(ValueError, match="shape_meta|parameter layout"):
        other.load_checkpoint(path)
    assert other.global_step == 0


def test_resume_restores_adamw_hyperparameters(tmp_path: Path):
    policy, batch = _ready_policy()
    trainer = _trainer(
        policy,
        eps=1e-7,
        betas=(0.8, 0.95),
        weight_decay=2e-4,
    )
    trainer.fit(batch, steps=1)
    path = tmp_path / "optimizer-hparams.ckpt"
    trainer.save_checkpoint(path)

    other_policy, _ = _ready_policy()
    other = _trainer(
        other_policy,
        eps=0.5,
        betas=(0.1, 0.2),
        weight_decay=0.9,
    )
    other.load_checkpoint(path)
    assert other.eps == pytest.approx(1e-7)
    assert other.betas == pytest.approx((0.8, 0.95))
    assert other.weight_decay == pytest.approx(2e-4)
    assert other.optimizer.eps == pytest.approx(other.eps)
    assert other.optimizer.betas == pytest.approx(other.betas)
    assert other.optimizer.weight_decay == pytest.approx(other.weight_decay)


def test_resume_provenance_must_match(tmp_path: Path):
    policy, batch = _ready_policy()
    trainer = _trainer(policy, provenance={"data_sha256": "a", "batch_size": 8})
    trainer.fit(batch, steps=1)
    path = tmp_path / "provenance.ckpt"
    trainer.save_checkpoint(path)
    other_policy, _ = _ready_policy()
    other = _trainer(
        other_policy, provenance={"data_sha256": "b", "batch_size": 8}
    )
    with pytest.raises(ValueError, match="provenance mismatch.*data_sha256"):
        other.load_checkpoint(path)
    assert other.global_step == 0


@pytest.mark.parametrize("counter", ["optimizer", "ema"])
def test_resume_rejects_counters_inconsistent_with_global_step(
    tmp_path: Path, counter: str
):
    policy, batch = _ready_policy(target_mode="detached")
    trainer = _trainer(policy, ema_decay=0.9)
    trainer.fit(batch, steps=1)
    valid = tmp_path / "counter-valid.ckpt"
    broken = tmp_path / f"counter-{counter}.ckpt"
    trainer.save_checkpoint(valid)
    payload = jt.load(str(valid))
    if counter == "optimizer":
        payload["trainer"]["optimizer"]["n_step"] = 99
    else:
        payload["trainer"]["ema"]["optimization_step"] = 99
    jt.save(payload, str(broken))

    target_policy, _ = _ready_policy(seed=99, target_mode="detached")
    target = _trainer(target_policy, ema_decay=0.9)
    online_before = _snapshot(target_policy)
    ema_before = {
        name: np.asarray(value).copy()
        for name, value in target.ema.target_policy.state_dict(to="numpy").items()
    }
    with pytest.raises(ValueError, match="does not match global_step"):
        target.load_checkpoint(broken)
    assert target.global_step == 0
    assert target.optimizer.n_step == 0
    assert target.ema.optimization_step == 0
    for name, value in online_before.items():
        np.testing.assert_array_equal(_snapshot(target_policy)[name], value)
    actual_ema = target.ema.target_policy.state_dict(to="numpy")
    for name, value in ema_before.items():
        np.testing.assert_array_equal(actual_ema[name], value)


def test_resume_rejects_invalid_history_and_log_every(tmp_path: Path):
    policy, batch = _ready_policy()
    trainer = _trainer(policy)
    trainer.fit(batch, steps=1)
    valid = tmp_path / "history-valid.ckpt"
    trainer.save_checkpoint(valid)
    payload = jt.load(str(valid))

    future = copy.deepcopy(payload)
    future["trainer"]["history"][0]["step"] = 999
    future["trainer"]["history"][0]["loss"] = float("nan")
    future_path = tmp_path / "history-future.ckpt"
    jt.save(future, str(future_path))
    with pytest.raises(ValueError, match=r"history\[0\].*step"):
        _trainer(_ready_policy()[0]).load_checkpoint(future_path)

    different_log = copy.deepcopy(payload)
    different_log["trainer"]["log_every"] = 999
    log_path = tmp_path / "log-every.ckpt"
    jt.save(different_log, str(log_path))
    with pytest.raises(ValueError, match="log_every does not match"):
        _trainer(_ready_policy()[0]).load_checkpoint(log_path)


@pytest.mark.parametrize("field", ["m", "values"])
def test_malformed_optimizer_tensor_is_rejected_without_partial_restore(
    tmp_path: Path, field: str
):
    policy, batch = _ready_policy()
    trainer = _trainer(policy)
    trainer.fit(batch, steps=1)
    valid = tmp_path / "valid.ckpt"
    broken = tmp_path / f"broken-{field}.ckpt"
    trainer.save_checkpoint(valid)
    payload = jt.load(str(valid))
    payload["trainer"]["optimizer"][field][0] = np.zeros((1,), dtype=np.float32)
    jt.save(payload, str(broken))

    target_policy, _ = _ready_policy(seed=99)
    target = _trainer(
        target_policy,
        eps=0.25,
        betas=(0.2, 0.3),
        weight_decay=0.4,
    )
    policy_before = _snapshot(target_policy)
    optimizer_before = target.optimizer_state()
    with pytest.raises(ValueError, match=rf"optimizer {field}\[0\].*shape"):
        target.load_checkpoint(broken)
    assert target.global_step == 0
    assert target.history == []
    for name, value in policy_before.items():
        np.testing.assert_array_equal(_snapshot(target_policy)[name], value)
    optimizer_after = target.optimizer_state()
    assert optimizer_after["n_step"] == optimizer_before["n_step"]
    assert optimizer_after["eps"] == optimizer_before["eps"]
    assert optimizer_after["betas"] == optimizer_before["betas"]
    assert optimizer_after["weight_decay"] == optimizer_before["weight_decay"]
    for key in ("m", "values"):
        for left, right in zip(optimizer_before[key], optimizer_after[key]):
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    "bad_value,error",
    [
        (np.asarray([np.nan], dtype=np.float32), ValueError),
        (np.asarray([1], dtype=np.int64), TypeError),
    ],
)
def test_optimizer_tensor_dtype_and_finite_values_are_validated(bad_value, error):
    policy, _ = _ready_policy()
    trainer = _trainer(policy)
    state = trainer.optimizer_state()
    state["values"][0] = np.broadcast_to(
        bad_value, tuple(state["param_shapes"][0])
    ).copy()
    with pytest.raises(error, match=r"values\[0\]"):
        trainer.load_optimizer_state(state)


def test_optimizer_state_rejects_live_hyperparameter_mutation():
    policy, _ = _ready_policy()
    trainer = _trainer(policy)
    trainer.optimizer.eps = 0.5
    with pytest.raises(RuntimeError, match="hyperparameters no longer match"):
        trainer.optimizer_state()


def test_optimizer_state_captures_adamw_moments():
    policy, batch = _ready_policy()
    trainer = _trainer(policy)
    trainer.fit(batch, steps=2)
    state = trainer.optimizer_state()
    assert state["n_step"] == 2
    assert len(state["m"]) == len(policy.trainable_parameters())
    assert any(np.any(np.asarray(moment) != 0.0) for moment in state["m"])
    assert all(
        list(np.asarray(moment).shape) == list(parameter.shape)
        for moment, parameter in zip(state["m"], policy.trainable_parameters())
    )


def test_sampled_times_stay_inside_the_flow_matcher_domain():
    """Regression: `rng.random()` can return a time below `eps` and crash the step.

    Real-data training hit exactly that (`t must lie in [0.01, 1]`), so every sampled
    time over a long range of steps is checked against the matcher's own bound.
    """
    policy, batch = _ready_policy()
    trainer = _trainer(policy, total_steps=400)
    eps = float(policy.loss_fn.eps)
    sampled = []
    for step in range(1, 401):
        noise, times = trainer._step_inputs((3, 4, 4), step)
        del noise
        sampled.append(np.asarray(times.numpy()))
    values = np.concatenate(sampled)
    assert values.min() >= eps
    assert values.max() <= 1.0
    assert (values > eps).any()
