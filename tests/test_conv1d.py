import numpy as np
import pytest

jt = pytest.importorskip("jittor")

from flowpolicy_jittor.model.conv1d import (  # noqa: E402
    Conv1dBlock,
    ConvTranspose1d,
    Downsample1d,
    Mish,
    Upsample1d,
)


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = previous


@pytest.mark.parametrize(
    "length,kernel_size,stride,padding,output_padding,dilation",
    [
        (4, 4, 2, 1, 0, 1),
        (5, 3, 1, 1, 0, 1),
        (6, 3, 2, 0, 1, 1),
        (7, 3, 1, 2, 0, 2),
    ],
)
def test_conv_transpose_1d_output_length(
    length, kernel_size, stride, padding, output_padding, dilation
):
    layer = ConvTranspose1d(
        2,
        3,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )
    x = jt.randn((2, 2, length))
    y = layer(x)

    expected = (
        (length - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )
    assert tuple(y.shape) == (2, 3, expected)


def test_conv_transpose_1d_backward_smoke():
    layer = ConvTranspose1d(3, 4, kernel_size=4, stride=2, padding=1)
    x = jt.randn((2, 3, 5))
    x.start_grad()
    loss = (layer(x) ** 2).mean()
    input_grad, weight_grad, bias_grad = jt.grad(
        loss,
        [x, layer.op.weight, layer.op.bias],
    )

    assert tuple(input_grad.shape) == tuple(x.shape)
    assert tuple(weight_grad.shape) == tuple(layer.op.weight.shape)
    assert tuple(bias_grad.shape) == tuple(layer.op.bias.shape)
    assert np.isfinite(input_grad.numpy()).all()
    assert np.isfinite(weight_grad.numpy()).all()
    assert np.isfinite(bias_grad.numpy()).all()


def test_up_and_down_sampling_restore_even_length():
    x = jt.randn((2, 8, 8))
    down = Downsample1d(8)
    up = Upsample1d(8)
    assert tuple(down(x).shape) == (2, 8, 4)
    assert tuple(up(down(x)).shape) == (2, 8, 8)


def test_local_mish_matches_definition():
    x_np = np.array([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=np.float32)
    expected = x_np * np.tanh(np.log1p(np.exp(x_np)))
    actual = Mish()(jt.array(x_np)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_local_mish_is_stable_for_large_inputs():
    x_np = np.array([-100.0, -20.0, 0.0, 20.0, 100.0], dtype=np.float32)
    expected = x_np * np.tanh(np.logaddexp(0.0, x_np))
    actual = Mish()(jt.array(x_np)).numpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_local_mish_gradient_is_finite():
    x = jt.array(
        np.array([-100.0, -10.0, 0.0, 10.0, 100.0], dtype=np.float32)
    )
    x.start_grad()
    gradient = jt.grad(Mish()(x).sum(), x).numpy()
    assert gradient.shape == (5,)
    assert np.isfinite(gradient).all()


def test_conv_block_preserves_length_and_changes_channels():
    block = Conv1dBlock(8, 16, kernel_size=3, num_groups=8)
    y = block(jt.randn((2, 8, 7)))
    assert tuple(y.shape) == (2, 16, 7)


def test_conv_block_rejects_invalid_group_count():
    with pytest.raises(ValueError, match="must be divisible"):
        Conv1dBlock(8, 10, kernel_size=3, num_groups=8)


def test_conv_block_rejects_even_kernel():
    with pytest.raises(ValueError, match="odd kernel_size"):
        Conv1dBlock(8, 16, kernel_size=4, num_groups=8)

