"""Sinusoidal timestep embedding used by the FlowPolicy U-Net.

The frequency formula is intentionally identical to the upstream
``flow_policy_3d/model/flow/positional_embedding.py`` implementation at
FlowPolicy commit ``86d10c14972b4eab083abec9026749c340f8e5ed``.
"""

from __future__ import annotations

import math

import jittor as jt


class SinusoidalPosEmb(jt.Module):
    """Encode a one-dimensional batch of timesteps as sine/cosine features.

    FlowPolicy uses an even embedding dimension of 256.  Requiring an even
    dimension of at least four avoids two silent upstream edge cases: odd
    dimensions return fewer features than requested, and ``dim=2`` divides by
    zero in the frequency denominator.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError(f"dim must be an integer, got {type(dim).__name__}")
        if dim < 4 or dim % 2 != 0:
            raise ValueError(f"dim must be an even integer >= 4, got {dim}")
        self.dim = dim

    def execute(self, timesteps: jt.Var) -> jt.Var:
        if len(timesteps.shape) != 1:
            raise ValueError(
                "SinusoidalPosEmb expects timesteps with shape [batch], "
                f"got {tuple(timesteps.shape)}"
            )

        # The released FlowPolicy path uses float32 timesteps.  Casting here
        # keeps integer/float64 inputs deterministic in Jittor; it intentionally
        # does not reproduce PyTorch's float64 promotion for unused code paths.
        half_dim = self.dim // 2
        frequency_step = math.log(10000.0) / (half_dim - 1)
        frequencies = jt.exp(
            jt.arange(half_dim, dtype="float32") * -frequency_step
        )
        angles = timesteps.float32().reshape((-1, 1)) * frequencies.reshape(
            (1, -1)
        )
        return jt.concat((jt.sin(angles), jt.cos(angles)), dim=-1)

