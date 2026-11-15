import numpy as np
import pytest

jt = pytest.importorskip("jittor")

from flowpolicy_jittor.model.conditional_unet1d import (  # noqa: E402
    ConditionalResidualBlock1D,
    ConditionalUnet1D,
)


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = previous


def _small_model(global_cond_dim=6):
    return ConditionalUnet1D(
        input_dim=4,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16),
        kernel_size=3,
        num_groups=4,
        var_fc_hidden_dim=8,
    )


def test_residual_block_shape_condition_effect_and_gradient():
    jt.set_global_seed(11)
    block = ConditionalResidualBlock1D(
        in_channels=4,
        out_channels=8,
        cond_dim=6,
        num_groups=4,
    )
    x = jt.randn((2, 4, 4))
    cond_a = jt.zeros((2, 6))
    cond_b = jt.ones((2, 6))
    x.start_grad()
    cond_b.start_grad()

    output_a = block(x, cond_a)
    output_b = block(x, cond_b)
    assert tuple(output_a.shape) == (2, 8, 4)
    assert not np.allclose(output_a.numpy(), output_b.numpy())

    parameter = dict(block.named_parameters())["cond_encoder.1.weight"]
    x_grad, cond_grad, parameter_grad = jt.grad(
        (output_b**2).mean(), [x, cond_b, parameter]
    )
    assert np.isfinite(x_grad.numpy()).all()
    assert np.isfinite(cond_grad.numpy()).all()
    assert np.isfinite(parameter_grad.numpy()).all()


def test_unet_output_shape_and_scalar_batch_timestep_equivalence():
    jt.set_global_seed(12)
    model = _small_model()
    sample = jt.randn((2, 4, 4))
    condition = jt.randn((2, 6))
    scalar_output = model(sample, 0.25, global_cond=condition).numpy()
    batch_output = model(
        sample,
        jt.array(np.array([0.25, 0.25], dtype=np.float32)),
        global_cond=condition,
    ).numpy()

    assert scalar_output.shape == (2, 4, 4)
    np.testing.assert_allclose(
        scalar_output, batch_output, rtol=1e-6, atol=1e-6
    )


def test_unet_without_global_condition_is_supported():
    model = _small_model(global_cond_dim=None)
    output = model(jt.randn((2, 4, 4)), 0.5)
    assert tuple(output.shape) == (2, 4, 4)


def test_three_level_topology_supports_flowpolicy_horizon_four():
    model = ConditionalUnet1D(
        input_dim=4,
        global_cond_dim=6,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16, 32),
        kernel_size=3,
        num_groups=4,
        var_fc_hidden_dim=8,
    )
    output = model(
        jt.randn((1, 4, 4)),
        0.5,
        global_cond=jt.randn((1, 6)),
    )
    assert model.downsample_factor == 4
    assert tuple(output.shape) == (1, 4, 4)


def test_unet_repeated_forward_is_deterministic():
    jt.set_global_seed(13)
    model = _small_model()
    sample = jt.randn((2, 4, 4))
    condition = jt.randn((2, 6))
    first = model(sample, 0.5, global_cond=condition).numpy()
    second = model(sample, 0.5, global_cond=condition).numpy()
    np.testing.assert_array_equal(first, second)


def test_unet_sample_condition_and_parameter_gradients_are_finite():
    jt.set_global_seed(14)
    model = _small_model()
    sample = jt.randn((2, 4, 4))
    condition = jt.randn((2, 6))
    sample.start_grad()
    condition.start_grad()
    output = model(sample, jt.array([0.2, 0.8]), global_cond=condition)
    active_parameter = next(
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("final_conv") and name.endswith("weight")
    )
    sample_grad, condition_grad, parameter_grad = jt.grad(
        (output**2).mean(), [sample, condition, active_parameter]
    )
    assert np.isfinite(sample_grad.numpy()).all()
    assert np.isfinite(condition_grad.numpy()).all()
    assert np.isfinite(parameter_grad.numpy()).all()


def test_unused_var_est_is_kept_in_state_but_excluded_from_optimization():
    jt.set_global_seed(15)
    model = _small_model()
    frozen = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("var_est.")
    }
    assert frozen
    assert all(parameter.is_stop_grad() for parameter in frozen.values())
    assert set(frozen).issubset(model.state_dict())
    assert all(
        not parameter.is_stop_grad()
        for parameter in model.trainable_parameters()
    )
    assert not any(
        id(parameter) == id(frozen_parameter)
        for parameter in model.trainable_parameters()
        for frozen_parameter in frozen.values()
    )

    frozen_before = {
        name: parameter.numpy().copy()
        for name, parameter in frozen.items()
    }
    active_parameter = next(
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("final_conv") and name.endswith("weight")
    )
    active_before = active_parameter.numpy().copy()
    optimizer = jt.optim.AdamW(
        model.trainable_parameters(),
        lr=1e-3,
        weight_decay=0.1,
    )
    loss = (
        model(
            jt.randn((2, 4, 4)),
            jt.array([0.2, 0.8]),
            global_cond=jt.randn((2, 6)),
        )
        ** 2
    ).mean()
    optimizer.step(loss)

    for name, parameter in frozen.items():
        np.testing.assert_array_equal(parameter.numpy(), frozen_before[name])
    assert not np.array_equal(active_parameter.numpy(), active_before)


@pytest.mark.parametrize(
    "sample_shape,timestep,condition_shape,error",
    [
        ((2, 4), 0.5, (2, 6), "sample must have shape"),
        ((0, 4, 4), 0.5, (0, 6), "batch size must be positive"),
        ((2, 0, 4), 0.5, (2, 6), "horizon must be positive"),
        ((2, 4, 3), 0.5, (2, 6), "expected input_dim"),
        ((2, 3, 4), 0.5, (2, 6), "must be divisible"),
        ((2, 4, 4), jt.array([0.1, 0.2, 0.3]), (2, 6), "batch size"),
        ((2, 4, 4), 0.5, (2, 5), "global_cond shape"),
    ],
)
def test_unet_rejects_invalid_shapes(
    sample_shape, timestep, condition_shape, error
):
    model = _small_model()
    with pytest.raises(ValueError, match=error):
        model(
            jt.randn(sample_shape),
            timestep,
            global_cond=jt.randn(condition_shape),
        )


def test_unet_requires_configured_global_condition():
    with pytest.raises(ValueError, match="global_cond is required"):
        _small_model()(jt.randn((2, 4, 4)), 0.5)


def test_unet_rejects_local_condition_and_untested_condition_type():
    model = _small_model()
    with pytest.raises(NotImplementedError, match="local_cond"):
        model(
            jt.randn((2, 4, 4)),
            0.5,
            local_cond=jt.randn((2, 4, 2)),
            global_cond=jt.randn((2, 6)),
        )
    with pytest.raises(NotImplementedError, match="condition_type='film'"):
        ConditionalUnet1D(
            input_dim=4,
            global_cond_dim=6,
            diffusion_step_embed_dim=8,
            down_dims=(8, 16),
            num_groups=4,
            condition_type="add",
        )


@pytest.mark.parametrize(
    "flag_name",
    ["use_down_condition", "use_mid_condition", "use_up_condition"],
)
def test_unet_rejects_unvalidated_disabled_condition_paths(flag_name):
    kwargs = {flag_name: False}
    with pytest.raises(NotImplementedError, match="condition paths"):
        ConditionalUnet1D(
            input_dim=4,
            global_cond_dim=6,
            diffusion_step_embed_dim=8,
            down_dims=(8, 16),
            n_groups=4,
            **kwargs,
        )


def test_upstream_n_groups_name_and_compatibility_alias_agree():
    upstream_named = ConditionalUnet1D(
        input_dim=4,
        global_cond_dim=6,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16),
        n_groups=4,
        var_fc_hidden_dim=8,
    )
    compatibility_named = _small_model()
    assert upstream_named.n_groups == compatibility_named.n_groups == 4
    with pytest.raises(ValueError, match="different values"):
        ConditionalUnet1D(
            input_dim=4,
            global_cond_dim=6,
            diffusion_step_embed_dim=8,
            down_dims=(8, 16),
            n_groups=2,
            num_groups=4,
        )


@pytest.mark.parametrize("down_dims", [(8.5, 16), (8, 0)])
def test_unet_rejects_invalid_channel_sizes(down_dims):
    with pytest.raises(ValueError, match="positive integer"):
        ConditionalUnet1D(
            input_dim=4,
            global_cond_dim=6,
            diffusion_step_embed_dim=8,
            down_dims=down_dims,
            num_groups=4,
        )

