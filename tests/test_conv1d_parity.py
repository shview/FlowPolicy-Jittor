"""PyTorch/Jittor parity tests for the local ConvTranspose1d wrapper.

These tests deliberately live outside ``test_conv1d.py`` so a pure Jittor
environment still runs all framework-native tests without installing PyTorch.
Install the ``parity`` extra to run this module.
"""

import numpy as np
import pytest

jt = pytest.importorskip("jittor")
torch = pytest.importorskip("torch")

from flowpolicy_jittor.model.conv1d import ConvTranspose1d  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = previous


def _copy_torch_parameters(torch_layer, jittor_layer):
    weight = torch_layer.weight.detach().cpu().numpy()
    jittor_layer.op.weight.update(jt.array(weight[..., None]))
    if torch_layer.bias is not None:
        bias = torch_layer.bias.detach().cpu().numpy()
        jittor_layer.op.bias.update(jt.array(bias))


@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding,output_padding,"
    "groups,bias,dilation",
    [
        (3, 4, 4, 2, 1, 0, 1, True, 1),
        (3, 5, 3, 1, 1, 0, 1, False, 1),
        (4, 6, 3, 2, 1, 1, 2, True, 1),
        (2, 3, 3, 2, 2, 1, 1, True, 2),
    ],
)
def test_conv_transpose_1d_matches_torch_forward(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    output_padding,
    groups,
    bias,
    dilation,
):
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((2, in_channels, 5), dtype=np.float32)
    kwargs = dict(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )
    torch_layer = torch.nn.ConvTranspose1d(
        in_channels, out_channels, **kwargs
    )
    jittor_layer = ConvTranspose1d(in_channels, out_channels, **kwargs)
    _copy_torch_parameters(torch_layer, jittor_layer)

    torch_out = torch_layer(torch.from_numpy(x_np)).detach().cpu().numpy()
    jittor_out = jittor_layer(jt.array(x_np)).numpy()

    np.testing.assert_allclose(jittor_out, torch_out, rtol=1e-5, atol=1e-5)


def test_conv_transpose_1d_matches_torch_gradients():
    """Compare input, weight, and bias gradients for the upstream settings."""

    rng = np.random.default_rng(19)
    x_np = rng.standard_normal((2, 3, 5), dtype=np.float32)

    torch_layer = torch.nn.ConvTranspose1d(3, 4, 4, stride=2, padding=1)
    jittor_layer = ConvTranspose1d(3, 4, 4, stride=2, padding=1)
    _copy_torch_parameters(torch_layer, jittor_layer)

    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_out = torch_layer(torch_x)
    upstream_np = rng.standard_normal(
        tuple(torch_out.shape), dtype=np.float32
    )
    torch_out.backward(torch.from_numpy(upstream_np))

    jittor_x = jt.array(x_np)
    jittor_x.start_grad()
    jittor_out = jittor_layer(jittor_x)
    jittor_loss = (jittor_out * jt.array(upstream_np)).sum()
    input_grad, weight_grad, bias_grad = jt.grad(
        jittor_loss,
        [jittor_x, jittor_layer.op.weight, jittor_layer.op.bias],
    )

    np.testing.assert_allclose(
        input_grad.numpy(),
        torch_x.grad.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        weight_grad.numpy()[..., 0],
        torch_layer.weight.grad.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        bias_grad.numpy(),
        torch_layer.bias.grad.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
