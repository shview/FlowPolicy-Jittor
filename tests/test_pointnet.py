import numpy as np
import pytest

jt = pytest.importorskip("jittor")

from flowpolicy_jittor.model.pointnet import (  # noqa: E402
    FlowPolicyEncoder,
    PointNetEncoderXYZ,
    StateMLP,
)


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = previous


def _encoder(num_points=16):
    return FlowPolicyEncoder(
        observation_space={
            "point_cloud": (num_points, 3),
            "agent_pos": (9,),
        },
        out_channel=64,
        state_mlp_size=(64, 64),
        pointcloud_encoder_cfg={
            "in_channels": 3,
            "out_channels": 64,
            "use_layernorm": True,
            "final_norm": "layernorm",
            "normal_channel": False,
        },
        use_pc_color=False,
        pointnet_type="mlp",
    )


@pytest.mark.parametrize("batch_size,num_points", [(1, 1), (2, 16), (3, 7)])
def test_pointnet_shape_for_single_and_multiple_points(batch_size, num_points):
    model = PointNetEncoderXYZ(out_channels=64)
    output = model(jt.randn((batch_size, num_points, 3)))
    assert tuple(output.shape) == (batch_size, 64)


def test_pointnet_is_invariant_to_point_permutation():
    jt.set_global_seed(31)
    model = PointNetEncoderXYZ(out_channels=64)
    points_np = np.random.default_rng(31).standard_normal(
        (2, 17, 3), dtype=np.float32
    )
    permutation = np.array([8, 2, 16, 0, 5, 11, 3, 14, 7, 1, 15, 6, 9, 4, 12, 10, 13])
    original = model(jt.array(points_np)).numpy()
    shuffled = model(jt.array(points_np[:, permutation, :])).numpy()
    np.testing.assert_allclose(original, shuffled, rtol=1e-6, atol=1e-6)


def test_pointnet_input_and_parameter_gradients_are_finite():
    jt.set_global_seed(32)
    model = PointNetEncoderXYZ(out_channels=16)
    points = jt.randn((2, 8, 3))
    points.start_grad()
    parameter = dict(model.named_parameters())["mlp.0.weight"]
    point_grad, parameter_grad = jt.grad(
        (model(points) ** 2).mean(), [points, parameter]
    )
    assert tuple(point_grad.shape) == (2, 8, 3)
    assert np.isfinite(point_grad.numpy()).all()
    assert np.isfinite(parameter_grad.numpy()).all()


def test_pointnet_repeated_forward_is_deterministic():
    jt.set_global_seed(33)
    model = PointNetEncoderXYZ(out_channels=32)
    points = jt.randn((2, 5, 3))
    first = model(points).numpy()
    second = model(points).numpy()
    np.testing.assert_array_equal(first, second)


def test_state_mlp_matches_released_9_to_64_to_64_shape():
    state_mlp = StateMLP(9, (64, 64))
    output = state_mlp(jt.randn((3, 9)))
    assert tuple(output.shape) == (3, 64)


def test_flowpolicy_encoder_combines_64_point_and_64_state_features():
    jt.set_global_seed(34)
    encoder = _encoder(num_points=16)
    output = encoder(
        {
            "point_cloud": jt.randn((2, 16, 3)),
            "agent_pos": jt.randn((2, 9)),
        }
    )
    assert encoder.extractor.output_dim == 64
    assert encoder.state_mlp.output_dim == 64
    assert encoder.output_shape() == 128
    assert tuple(output.shape) == (2, 128)


def test_flowpolicy_encoder_both_branches_receive_gradients():
    encoder = _encoder(num_points=8)
    points = jt.randn((2, 8, 3))
    state = jt.randn((2, 9))
    points.start_grad()
    state.start_grad()
    output = encoder({"point_cloud": points, "agent_pos": state})
    points_grad, state_grad = jt.grad((output**2).mean(), [points, state])
    assert np.isfinite(points_grad.numpy()).all()
    assert np.isfinite(state_grad.numpy()).all()


@pytest.mark.parametrize(
    "shape,error",
    [
        ((2, 3), r"shape \[B,N,3\]"),
        ((2, 4, 6), "expected 3 point channels"),
        ((2, 0, 3), "at least one point"),
    ],
)
def test_pointnet_rejects_invalid_input_shapes(shape, error):
    with pytest.raises(ValueError, match=error):
        PointNetEncoderXYZ()(jt.randn(shape))


def test_pointnet_rejects_rgb_and_unknown_normalization():
    with pytest.raises(NotImplementedError, match="XYZ path"):
        PointNetEncoderXYZ(in_channels=6)
    with pytest.raises(NotImplementedError, match="final_norm"):
        PointNetEncoderXYZ(final_norm="batchnorm")


def test_flowpolicy_encoder_rejects_unvalidated_branches_and_shape_mismatch():
    observation_space = {"point_cloud": (8, 3), "agent_pos": (9,)}
    with pytest.raises(NotImplementedError, match="RGB"):
        FlowPolicyEncoder(observation_space, use_pc_color=True)
    with pytest.raises(NotImplementedError, match="imagined-robot"):
        FlowPolicyEncoder(
            {**observation_space, "imagin_robot": (8, 3)}
        )

    encoder = _encoder(num_points=8)
    with pytest.raises(ValueError, match="expected point shape"):
        encoder(
            {
                "point_cloud": jt.randn((2, 7, 3)),
                "agent_pos": jt.randn((2, 9)),
            }
        )
    with pytest.raises(ValueError, match="batch sizes must match"):
        encoder(
            {
                "point_cloud": jt.randn((2, 8, 3)),
                "agent_pos": jt.randn((3, 9)),
            }
        )


def test_observation_space_rejects_fractional_dimensions():
    with pytest.raises(ValueError, match="positive integer"):
        FlowPolicyEncoder(
            {"point_cloud": (8.5, 3), "agent_pos": (9,)}
        )


def test_group_count_alias_rejects_even_explicit_default_conflict():
    from flowpolicy_jittor.model import ConditionalUnet1D

    with pytest.raises(ValueError, match="different values"):
        ConditionalUnet1D(
            input_dim=4,
            global_cond_dim=6,
            diffusion_step_embed_dim=8,
            down_dims=(8, 16),
            n_groups=8,
            num_groups=4,
            var_fc_hidden_dim=8,
        )

