"""Conditional one-dimensional U-Net used by FlowPolicy.

The down/middle/up topology and FiLM equation follow FlowPolicy commit
``86d10c14972b4eab083abec9026749c340f8e5ed``.  This first Jittor port only
implements the path used by the released FlowPolicy configuration:
global conditioning with ``condition_type="film"``.  Unsupported condition
types fail explicitly instead of silently taking an untested path.
"""

from __future__ import annotations

import numbers
from typing import Optional, Sequence, Union

import jittor as jt
from jittor import nn

from .conv1d import Conv1dBlock, Downsample1d, Mish, Upsample1d
from .positional_embedding import SinusoidalPosEmb


Timestep = Union[jt.Var, float, int]


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _resolve_group_count(
    n_groups: Optional[int], num_groups: Optional[int]
) -> int:
    """Accept the upstream name and the early-port compatibility alias."""
    if n_groups is None and num_groups is None:
        return 8
    if n_groups is None:
        return _positive_int("num_groups", num_groups)
    if num_groups is None:
        return _positive_int("n_groups", n_groups)
    if n_groups != num_groups:
        raise ValueError("n_groups and num_groups specify different values")
    return _positive_int("n_groups", n_groups)


class ConditionalResidualBlock1D(jt.Module):
    """Two convolution blocks with global FiLM modulation and a residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: Optional[int] = None,
        condition_type: str = "film",
        num_groups: Optional[int] = None,
    ) -> None:
        super().__init__()
        in_channels = _positive_int("in_channels", in_channels)
        out_channels = _positive_int("out_channels", out_channels)
        cond_dim = _positive_int("cond_dim", cond_dim)
        n_groups = _resolve_group_count(n_groups, num_groups)
        if condition_type != "film":
            raise NotImplementedError(
                "only condition_type='film' is validated in the Jittor port"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        self.condition_type = condition_type
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    num_groups=n_groups,
                ),
                Conv1dBlock(
                    out_channels,
                    out_channels,
                    kernel_size,
                    num_groups=n_groups,
                ),
            ]
        )
        self.cond_encoder = nn.Sequential(
            Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def execute(
        self, x: jt.Var, cond: Optional[jt.Var] = None
    ) -> jt.Var:
        if len(x.shape) != 3:
            raise ValueError(
                "ConditionalResidualBlock1D expects x as [B,C,H], "
                f"got {tuple(x.shape)}"
            )
        if int(x.shape[1]) != self.in_channels:
            raise ValueError(
                f"expected x with {self.in_channels} channels, "
                f"got {int(x.shape[1])}"
            )
        if cond is not None:
            if len(cond.shape) != 2:
                raise ValueError(
                    "FiLM condition must have shape [B,cond_dim], "
                    f"got {tuple(cond.shape)}"
                )
            if int(cond.shape[0]) != int(x.shape[0]):
                raise ValueError("x and condition batch sizes must match")
            if int(cond.shape[1]) != self.cond_dim:
                raise ValueError(
                    f"expected condition dim {self.cond_dim}, "
                    f"got {int(cond.shape[1])}"
                )

        out = self.blocks[0](x)
        if cond is not None:
            embedding = self.cond_encoder(cond.float32()).reshape(
                (int(x.shape[0]), 2, self.out_channels, 1)
            )
            scale = embedding[:, 0, :, :]
            bias = embedding[:, 1, :, :]
            # This intentionally matches the released code: scale*out+bias,
            # not the alternative (1+scale)*out+bias FiLM convention.
            out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(jt.Module):
    """FlowPolicy velocity network for action sequences.

    Inputs and outputs use ``[batch, horizon, action_dim]``.  The model
    validates horizon divisibility up front so down/up-sampling cannot create
    a late skip-connection mismatch.
    """

    def __init__(
        self,
        input_dim: int,
        local_cond_dim: Optional[int] = None,
        global_cond_dim: Optional[int] = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: Optional[int] = None,
        var_fc_hidden_dim: int = 512,
        condition_type: str = "film",
        use_down_condition: bool = True,
        use_mid_condition: bool = True,
        use_up_condition: bool = True,
        num_groups: Optional[int] = None,
    ) -> None:
        super().__init__()
        input_dim = _positive_int("input_dim", input_dim)
        diffusion_step_embed_dim = _positive_int(
            "diffusion_step_embed_dim", diffusion_step_embed_dim
        )
        var_fc_hidden_dim = _positive_int(
            "var_fc_hidden_dim", var_fc_hidden_dim
        )
        n_groups = _resolve_group_count(n_groups, num_groups)
        if local_cond_dim is not None:
            raise NotImplementedError(
                "local_cond is not used by released FlowPolicy and is not "
                "validated in this Jittor port"
            )
        if condition_type != "film":
            raise NotImplementedError(
                "only condition_type='film' is validated in the Jittor port"
            )
        if not (
            use_down_condition
            and use_mid_condition
            and use_up_condition
        ):
            raise NotImplementedError(
                "the released FlowPolicy configuration enables all three "
                "condition paths; disabled condition paths are not validated"
            )
        if global_cond_dim is not None:
            global_cond_dim = _positive_int(
                "global_cond_dim", global_cond_dim
            )
        if not down_dims:
            raise ValueError("down_dims must contain at least one channel size")
        normalized_down_dims = tuple(
            _positive_int(f"down_dims[{index}]", dim)
            for index, dim in enumerate(down_dims)
        )
        if any(dim % n_groups != 0 for dim in normalized_down_dims):
            raise ValueError(
                "every down_dims channel size must be divisible by n_groups"
            )

        self.input_dim = input_dim
        self.local_cond_dim = local_cond_dim
        self.global_cond_dim = global_cond_dim
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.down_dims = normalized_down_dims
        self.n_groups = n_groups
        self.condition_type = condition_type
        self.use_down_condition = bool(use_down_condition)
        self.use_mid_condition = bool(use_mid_condition)
        self.use_up_condition = bool(use_up_condition)
        self.downsample_factor = 2 ** max(len(normalized_down_dims) - 1, 0)

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # Kept for code-faithful state structure even though the released
        # FlowPolicy forward path never reads var_est.
        self.var_est = nn.Sequential(
            nn.Linear(input_dim, var_fc_hidden_dim),
            nn.SiLU(),
            nn.Linear(var_fc_hidden_dim, var_fc_hidden_dim),
            nn.SiLU(),
            nn.Linear(var_fc_hidden_dim, var_fc_hidden_dim),
            nn.SiLU(),
            nn.Linear(var_fc_hidden_dim, 1),
        )
        # The released forward never reads var_est.  PyTorch leaves these
        # parameters at grad=None, while Jittor AdamW may otherwise apply
        # weight decay to a missing-gradient parameter.  Keep the state-dict
        # structure for code compatibility, but freeze and exclude it from
        # effective optimization.
        for parameter in self.var_est.parameters():
            parameter.stop_grad()

        cond_dim = dsed + (global_cond_dim or 0)
        all_dims = (input_dim,) + normalized_down_dims
        in_out = tuple(zip(all_dims[:-1], all_dims[1:]))

        self.down_modules = nn.ModuleList([])
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        nn.Identity()
                        if is_last
                        else Downsample1d(dim_out),
                    ]
                )
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    condition_type=condition_type,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    condition_type=condition_type,
                ),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for dim_in, dim_out in reversed(in_out[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        Upsample1d(dim_in),
                    ]
                )
            )

        start_dim = normalized_down_dims[0]
        self.final_conv = nn.Sequential(
            Conv1dBlock(
                start_dim,
                start_dim,
                kernel_size=kernel_size,
                num_groups=n_groups,
            ),
            nn.Conv1d(start_dim, input_dim, kernel_size=1),
        )

    def _prepare_timesteps(
        self, timestep: Timestep, batch_size: int
    ) -> jt.Var:
        if isinstance(timestep, bool):
            raise TypeError("boolean timestep is not supported")
        if isinstance(timestep, numbers.Real):
            # The upstream helper creates a long tensor for Python numbers,
            # which truncates Python floats.  Preserving the float is an
            # intentional compatibility correction; the real training path
            # supplies a float Jittor Var rather than a Python scalar.
            timesteps = jt.array([float(timestep)], dtype="float32")
        elif isinstance(timestep, jt.Var):
            timesteps = timestep
        else:
            raise TypeError(
                "timestep must be a Python number or a Jittor Var"
            )

        if len(timesteps.shape) != 1:
            raise ValueError(
                "timestep must be scalar-like or have shape [B], "
                f"got {tuple(timesteps.shape)}"
            )
        count = int(timesteps.shape[0])
        if count == 1 and batch_size != 1:
            timesteps = timesteps.repeat(batch_size)
        elif count != batch_size:
            raise ValueError(
                f"timestep has {count} values for batch size {batch_size}"
            )
        return timesteps.float32()

    def _prepare_global_condition(
        self, global_cond: Optional[jt.Var], batch_size: int
    ) -> Optional[jt.Var]:
        if self.global_cond_dim is None:
            if global_cond is not None:
                raise ValueError(
                    "global_cond was provided but global_cond_dim is None"
                )
            return None
        if global_cond is None:
            raise ValueError(
                "global_cond is required when global_cond_dim is configured"
            )
        if len(global_cond.shape) != 2:
            raise ValueError(
                "global_cond must have shape [B,global_cond_dim], "
                f"got {tuple(global_cond.shape)}"
            )
        expected = (batch_size, self.global_cond_dim)
        actual = (int(global_cond.shape[0]), int(global_cond.shape[1]))
        if actual != expected:
            raise ValueError(
                f"expected global_cond shape {expected}, got {actual}"
            )
        return global_cond.float32()

    def execute(
        self,
        sample: jt.Var,
        timestep: Timestep,
        local_cond: Optional[jt.Var] = None,
        global_cond: Optional[jt.Var] = None,
        **_: object,
    ) -> jt.Var:
        if local_cond is not None:
            # The released configuration never uses local conditioning.  The
            # upstream local up-skip also contains an unreachable index check;
            # rejecting it avoids claiming compatibility with that buggy path.
            raise NotImplementedError("local_cond is not supported")
        if len(sample.shape) != 3:
            raise ValueError(
                "sample must have shape [B,H,input_dim], "
                f"got {tuple(sample.shape)}"
            )
        batch_size = int(sample.shape[0])
        horizon = int(sample.shape[1])
        action_dim = int(sample.shape[2])
        if batch_size <= 0:
            raise ValueError("sample batch size must be positive")
        if horizon <= 0:
            raise ValueError("sample horizon must be positive")
        if action_dim != self.input_dim:
            raise ValueError(
                f"expected input_dim {self.input_dim}, got {action_dim}"
            )
        if horizon % self.downsample_factor != 0:
            raise ValueError(
                f"horizon {horizon} must be divisible by U-Net factor "
                f"{self.downsample_factor}"
            )

        timesteps = self._prepare_timesteps(timestep, batch_size)
        condition = self._prepare_global_condition(global_cond, batch_size)
        timestep_embedding = self.diffusion_step_encoder(timesteps)
        global_feature = (
            timestep_embedding
            if condition is None
            else jt.concat((timestep_embedding, condition), dim=-1)
        )

        x = sample.float32().transpose(0, 2, 1)
        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            block_cond = global_feature if self.use_down_condition else None
            x = resnet(x, block_cond)
            x = resnet2(x, block_cond)
            skips.append(x)
            x = downsample(x)

        for middle in self.mid_modules:
            block_cond = global_feature if self.use_mid_condition else None
            x = middle(x, block_cond)

        for resnet, resnet2, upsample in self.up_modules:
            x = jt.concat((x, skips.pop()), dim=1)
            block_cond = global_feature if self.use_up_condition else None
            x = resnet(x, block_cond)
            x = resnet2(x, block_cond)
            x = upsample(x)

        x = self.final_conv(x)
        return x.transpose(0, 2, 1)

    def trainable_parameters(self):
        """Return parameters safe to pass to a Jittor optimizer.

        ``Module.parameters()`` includes stopped parameters in Jittor 1.3.11.
        Filtering here prevents the code-faithful but unused ``var_est``
        branch from allocating optimizer state or receiving weight decay.
        """
        return [
            parameter
            for parameter in self.parameters()
            if not parameter.is_stop_grad()
        ]

