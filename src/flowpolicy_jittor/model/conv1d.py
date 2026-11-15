"""Jittor 1D convolution blocks used by the FlowPolicy U-Net.

Jittor 1.3.11 does not provide ``nn.ConvTranspose1d``.  The wrapper below
follows Jittor's own ``Conv1d`` strategy: add a singleton spatial dimension,
run the two-dimensional operator, then remove that dimension.

API reference: Jittor commit ``ab31f57b351fccbf3a359698185d5cda8f49abba``,
``python/jittor/nn.py`` (``Conv1d`` and ``ConvTranspose``).  This project must
keep a forward-and-gradient parity test because the 1D wrapper is local code,
not an upstream Jittor layer.
"""

from __future__ import annotations

import jittor as jt
from jittor import nn


def _require_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _require_non_negative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


class ConvTranspose1d(jt.Module):
    """One-dimensional transposed convolution backed by ``nn.ConvTranspose``.

    Input and output layout is ``[batch, channels, length]``.  Integer-only
    convolution arguments are intentional: the current Jittor ``Conv1d``
    implementation does not safely support PyTorch-style tuples.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        in_channels = _require_positive_int("in_channels", in_channels)
        out_channels = _require_positive_int("out_channels", out_channels)
        kernel_size = _require_positive_int("kernel_size", kernel_size)
        stride = _require_positive_int("stride", stride)
        padding = _require_non_negative_int("padding", padding)
        output_padding = _require_non_negative_int(
            "output_padding", output_padding
        )
        groups = _require_positive_int("groups", groups)
        dilation = _require_positive_int("dilation", dilation)

        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError(
                "in_channels and out_channels must both be divisible by groups"
            )
        if output_padding >= stride and output_padding >= dilation:
            raise ValueError(
                "output_padding must be smaller than stride or dilation"
            )

        self.op = nn.ConvTranspose(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            output_padding=(output_padding, 0),
            groups=groups,
            bias=bias,
            dilation=(dilation, 1),
        )

    def execute(self, x: jt.Var) -> jt.Var:
        if len(x.shape) != 3:
            raise ValueError(
                "ConvTranspose1d expects [batch, channels, length], "
                f"got shape {tuple(x.shape)}"
            )
        return self.op(x.unsqueeze(-1)).squeeze(-1)


class Downsample1d(jt.Module):
    """Halve the sequence length with a stride-2 convolution."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        dim = _require_positive_int("dim", dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def execute(self, x: jt.Var) -> jt.Var:
        return self.conv(x)


class Upsample1d(jt.Module):
    """Double the sequence length with the upstream FlowPolicy parameters."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        dim = _require_positive_int("dim", dim)
        self.conv = ConvTranspose1d(
            dim,
            dim,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def execute(self, x: jt.Var) -> jt.Var:
        return self.conv(x)


class Mish(jt.Module):
    """Mish activation with a working Jittor 1.3.11 implementation.

    Jittor 1.3.11's ``nn.Mish`` calls the missing symbol ``jt.softplus``.
    The functional implementation is available as ``nn.softplus`` and is
    used here until the upstream layer is fixed.
    """

    def execute(self, x: jt.Var) -> jt.Var:
        return x * jt.tanh(nn.softplus(x))


class Conv1dBlock(jt.Module):
    """Conv1d -> GroupNorm -> Mish, matching the upstream implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        in_channels = _require_positive_int("in_channels", in_channels)
        out_channels = _require_positive_int("out_channels", out_channels)
        kernel_size = _require_positive_int("kernel_size", kernel_size)
        num_groups = _require_positive_int("num_groups", num_groups)
        if kernel_size % 2 == 0:
            raise ValueError(
                "Conv1dBlock requires an odd kernel_size so the sequence "
                "length is preserved"
            )
        if out_channels % num_groups != 0:
            raise ValueError(
                f"out_channels={out_channels} must be divisible by "
                f"num_groups={num_groups}"
            )

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(num_groups, out_channels),
            Mish(),
        )

    def execute(self, x: jt.Var) -> jt.Var:
        return self.block(x)

