import numpy as np
import pytest

jt = pytest.importorskip("jittor")

from flowpolicy_jittor.model.flow_matching import (  # noqa: E402
    ConsistencyFlowMatchingLoss,
)


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = previous


class StoredVelocity(jt.Module):
    def __init__(self, velocity):
        super().__init__()
        self.velocity = jt.array(np.asarray(velocity, dtype=np.float32)).stop_grad()

    def execute(self, x, timestep, local_cond=None, global_cond=None):
        del timestep, local_cond, global_cond
        assert tuple(x.shape) == tuple(self.velocity.shape)
        return self.velocity


class ZeroVelocity(jt.Module):
    def execute(self, x, timestep, local_cond=None, global_cond=None):
        del timestep, local_cond, global_cond
        return jt.zeros_like(x)


class ScaledVelocity(jt.Module):
    def __init__(self, velocity, scale=0.5):
        super().__init__()
        self.velocity = jt.array(np.asarray(velocity, dtype=np.float32)).stop_grad()
        self.scale = jt.array([scale], dtype="float32")

    def execute(self, x, timestep, local_cond=None, global_cond=None):
        del x, timestep, local_cond, global_cond
        return self.velocity * self.scale


class TimeVelocity(jt.Module):
    def execute(self, x, timestep, local_cond=None, global_cond=None):
        del local_cond, global_cond
        time = timestep.reshape((-1, 1, 1)) / 99.0
        return jt.ones_like(x) * time


def _target_noise():
    target = np.asarray(
        [
            [[1.0, -1.0], [2.0, 0.5]],
            [[-0.5, 2.5], [1.5, -2.0]],
        ],
        dtype=np.float32,
    )
    noise = np.asarray(
        [
            [[-1.0, 0.0], [0.5, -0.5]],
            [[1.0, -1.5], [-0.5, 1.0]],
        ],
        dtype=np.float32,
    )
    return target, noise


def test_exact_straight_line_velocity_has_zero_consistency_loss():
    target, noise = _target_noise()
    model = StoredVelocity(target - noise)
    loss_fn = ConsistencyFlowMatchingLoss(delta=0.1, alpha=1e-3)
    loss, parts = loss_fn(
        model,
        jt.array(target),
        noise=jt.array(noise),
        t=jt.array([0.2, 0.7]),
    )
    assert float(loss.item()) < 1e-12
    assert float(parts["endpoint_loss"].item()) < 1e-12
    assert float(parts["velocity_loss"].item()) < 1e-12


def test_zero_velocity_matches_manual_endpoint_formula():
    target, noise = _target_noise()
    t = np.asarray([0.2, 0.7], dtype=np.float32)
    delta = 0.1
    loss_fn = ConsistencyFlowMatchingLoss(delta=delta, alpha=0.25)
    loss, parts = loss_fn(
        ZeroVelocity(),
        jt.array(target),
        noise=jt.array(noise),
        t=jt.array(t),
    )
    expected = np.mean((delta * (noise - target)) ** 2)
    np.testing.assert_allclose(parts["endpoint_loss"].numpy(), expected, atol=1e-7)
    np.testing.assert_allclose(parts["velocity_loss"].numpy(), 0.0, atol=1e-8)
    np.testing.assert_allclose(loss.numpy(), expected, atol=1e-7)


def test_endpoint_and_velocity_components_match_manual_formula():
    target, noise = _target_noise()
    t = np.asarray([0.2, 0.7], dtype=np.float32)
    delta = 0.1
    alpha = 0.25
    r = t + delta
    segment_end = np.asarray([0.5, 1.0], dtype=np.float32)
    t_expand = t[:, None, None]
    r_expand = r[:, None, None]
    segment_expand = segment_end[:, None, None]
    x_t = t_expand * target + (1.0 - t_expand) * noise
    x_r = r_expand * target + (1.0 - r_expand) * noise
    v_t = np.ones_like(target) * t_expand
    v_r = np.ones_like(target) * r_expand
    f_t = x_t + (segment_expand - t_expand) * v_t
    f_r = x_r + (segment_expand - r_expand) * v_r
    expected_endpoint = np.mean((f_t - f_r) ** 2)
    expected_velocity = np.mean((v_t - v_r) ** 2)

    loss_fn = ConsistencyFlowMatchingLoss(delta=delta, alpha=alpha)
    loss, parts = loss_fn(
        TimeVelocity(),
        jt.array(target),
        noise=jt.array(noise),
        t=jt.array(t),
    )
    np.testing.assert_allclose(
        parts["endpoint_loss"].numpy(), expected_endpoint, atol=1e-7
    )
    np.testing.assert_allclose(
        parts["velocity_loss"].numpy(), expected_velocity, atol=1e-7
    )
    np.testing.assert_allclose(
        loss.numpy(), expected_endpoint + alpha * expected_velocity, atol=1e-7
    )


def test_inconsistent_velocity_is_positive_and_gradient_is_finite():
    target, noise = _target_noise()
    model = ScaledVelocity(target - noise, scale=0.4)
    loss_fn = ConsistencyFlowMatchingLoss(delta=0.1, alpha=1e-2)
    loss, parts = loss_fn(
        model,
        jt.array(target),
        noise=jt.array(noise),
        t=jt.array([0.2, 0.7]),
    )
    gradient = jt.grad(loss, model.scale)
    assert float(loss.item()) > 0.0
    assert float(parts["endpoint_loss"].item()) > 0.0
    assert np.all(np.isfinite(gradient.numpy()))
    assert np.any(np.abs(gradient.numpy()) > 0.0)


def test_random_sampling_is_reproducible_after_seed_reset():
    target, _ = _target_noise()
    loss_fn = ConsistencyFlowMatchingLoss()
    jt.set_global_seed(1234)
    first, first_parts = loss_fn(ZeroVelocity(), jt.array(target))
    jt.sync_all()
    first_values = (float(first.item()), float(first_parts["t_mean"].item()))
    jt.set_global_seed(1234)
    second, second_parts = loss_fn(ZeroVelocity(), jt.array(target))
    jt.sync_all()
    second_values = (float(second.item()), float(second_parts["t_mean"].item()))
    np.testing.assert_allclose(first_values, second_values, rtol=0.0, atol=0.0)


def test_boundary_zero_uses_analytic_endpoint_without_target_model():
    target, noise = _target_noise()
    loss_fn = ConsistencyFlowMatchingLoss(boundary=0, delta=0.1)
    loss, parts = loss_fn(
        StoredVelocity(target - noise),
        jt.array(target),
        noise=jt.array(noise),
        t=jt.array([0.2, 0.7]),
    )
    assert float(loss.item()) < 1e-12
    assert float(parts["velocity_loss"].item()) == 0.0


def test_detached_mode_requires_explicit_target_model():
    target, noise = _target_noise()
    loss_fn = ConsistencyFlowMatchingLoss(target_mode="detached")
    with pytest.raises(ValueError, match="explicit target_model"):
        loss_fn(
            StoredVelocity(target - noise),
            jt.array(target),
            noise=jt.array(noise),
            t=jt.array([0.2, 0.7]),
        )


def test_detached_target_model_runs_but_is_not_claimed_as_ema():
    target, noise = _target_noise()
    current = ScaledVelocity(target - noise, scale=0.6)
    target_model = StoredVelocity(target - noise)
    loss_fn = ConsistencyFlowMatchingLoss(
        target_mode="detached", delta=0.1, alpha=1e-2
    )
    loss, parts = loss_fn(
        current,
        jt.array(target),
        target_model=target_model,
        noise=jt.array(noise),
        t=jt.array([0.2, 0.7]),
    )
    assert float(loss.item()) > 0.0
    assert np.all(np.isfinite(parts["velocity_loss"].numpy()))


def test_detached_mode_rejects_current_model_as_target():
    target, noise = _target_noise()
    model = StoredVelocity(target - noise)
    loss_fn = ConsistencyFlowMatchingLoss(target_mode="detached")
    with pytest.raises(ValueError, match="distinct model"):
        loss_fn(
            model,
            jt.array(target),
            target_model=model,
            noise=jt.array(noise),
            t=jt.array([0.2, 0.7]),
        )


def test_segment_boundary_and_r_clamp_match_released_rules():
    loss_fn = ConsistencyFlowMatchingLoss(num_segments=2, delta=0.1)
    ends = loss_fn._segment_ends(jt.array([0.5, 0.50001, 0.99]))
    np.testing.assert_allclose(ends.numpy(), [0.5, 1.0, 1.0], atol=0.0)
    t, r = loss_fn._time_pair(2, jt.array([0.5, 0.99]))
    np.testing.assert_allclose(t.numpy(), [0.5, 0.99], atol=0.0)
    np.testing.assert_allclose(r.numpy(), [0.6, 1.0], atol=1e-7)


def test_full_finite_validation_is_debug_only():
    target, noise = _target_noise()
    target[0, 0, 0] = np.nan
    debug_loss = ConsistencyFlowMatchingLoss(validate_finite=True)
    with pytest.raises(ValueError, match="finite"):
        debug_loss(
            ZeroVelocity(),
            jt.array(target),
            noise=jt.array(noise),
            t=jt.array([0.2, 0.7]),
        )


def test_condition_mask_requires_matching_data_and_shape():
    target, noise = _target_noise()
    loss_fn = ConsistencyFlowMatchingLoss()
    with pytest.raises(ValueError, match="supplied together"):
        loss_fn(
            ZeroVelocity(),
            jt.array(target),
            noise=jt.array(noise),
            t=jt.array([0.2, 0.7]),
            condition_mask=jt.zeros(target.shape).bool(),
        )


def test_all_false_condition_mask_matches_unmasked_loss():
    target, noise = _target_noise()
    loss_fn = ConsistencyFlowMatchingLoss(delta=0.1)
    kwargs = {
        "noise": jt.array(noise),
        "t": jt.array([0.2, 0.7]),
    }
    unmasked, _ = loss_fn(ZeroVelocity(), jt.array(target), **kwargs)
    masked, _ = loss_fn(
        ZeroVelocity(),
        jt.array(target),
        condition_mask=jt.zeros(target.shape).bool(),
        condition_data=jt.array(target),
        **kwargs,
    )
    np.testing.assert_allclose(masked.numpy(), unmasked.numpy(), atol=1e-8)
    with pytest.raises(ValueError, match="identical shapes"):
        loss_fn(
            ZeroVelocity(),
            jt.array(target),
            noise=jt.array(noise),
            t=jt.array([0.2, 0.7]),
            condition_mask=jt.zeros((2, 2, 1)).bool(),
            condition_data=jt.array(target),
        )


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"eps": 0.0}, ValueError),
        ({"eps": 1.0}, ValueError),
        ({"num_segments": 0}, ValueError),
        ({"boundary": 1.1}, ValueError),
        ({"delta": 0.0}, ValueError),
        ({"alpha": -1.0}, ValueError),
        ({"target_mode": "ema"}, ValueError),
    ],
)
def test_invalid_hyperparameters_fail_early(kwargs, error):
    with pytest.raises(error):
        ConsistencyFlowMatchingLoss(**kwargs)


def test_invalid_target_noise_and_time_shapes_fail_early():
    target, noise = _target_noise()
    loss_fn = ConsistencyFlowMatchingLoss()
    with pytest.raises(ValueError, match=r"\[B,H,D\]"):
        loss_fn(ZeroVelocity(), jt.array(target[:, 0]))
    with pytest.raises(ValueError, match="identical shapes"):
        loss_fn(ZeroVelocity(), jt.array(target), noise=jt.array(noise[:, :, :1]))
    with pytest.raises(ValueError, match="t must have shape"):
        loss_fn(
            ZeroVelocity(),
            jt.array(target),
            noise=jt.array(noise),
            t=jt.array([0.2]),
        )
    with pytest.raises(ValueError, match="must lie"):
        loss_fn(
            ZeroVelocity(),
            jt.array(target),
            noise=jt.array(noise),
            t=jt.array([0.0, 0.7]),
        )

