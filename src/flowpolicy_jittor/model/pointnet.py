"""Point-cloud and robot-state encoders used by FlowPolicy.

This module ports the XYZ-only path from FlowPolicy commit ``86d10c1``
(``model/vision/pointnet_extractor.py``).  The released experiment config
uses three XYZ channels, a ``3 -> 64 -> 128 -> 256`` per-point MLP, max
pooling across points, and a 64-dimensional LayerNorm projection.  RGB and
imagined-robot branches are deliberately rejected until separately tested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import jittor as jt
from jittor import nn


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _shape_tuple(name: str, value: Sequence[int], rank: int) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of dimensions")
    shape = tuple(
        _positive_int(f"{name}[{i}]", dim) for i, dim in enumerate(value)
    )
    if len(shape) != rank:
        raise ValueError(f"{name} must have rank {rank}, got {shape}")
    return shape


class PointNetEncoderXYZ(jt.Module):
    """Permutation-invariant PointNet encoder for ``[B, N, 3]`` XYZ points.

    Defaults intentionally match the released FlowPolicy experiment config,
    not every default of the broader upstream helper class.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
        use_projection: bool = True,
    ) -> None:
        super().__init__()
        in_channels = _positive_int("in_channels", in_channels)
        out_channels = _positive_int("out_channels", out_channels)
        if in_channels != 3:
            raise NotImplementedError(
                "PointNetEncoderXYZ implements only the released XYZ path "
                f"with 3 channels, got {in_channels}"
            )
        if final_norm not in {"layernorm", "none"}:
            raise NotImplementedError(
                "final_norm must be 'layernorm' or 'none'"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_layernorm = bool(use_layernorm)
        self.final_norm = final_norm
        self.use_projection = bool(use_projection)

        hidden_dims = (64, 128, 256)
        layers = []
        dim_in = in_channels
        for dim_out in hidden_dims:
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(
                nn.LayerNorm(dim_out)
                if self.use_layernorm
                else nn.Identity()
            )
            layers.append(nn.ReLU())
            dim_in = dim_out
        self.mlp = nn.Sequential(*layers)

        if not self.use_projection:
            self.final_projection = nn.Identity()
            self.output_dim = hidden_dims[-1]
        elif final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(hidden_dims[-1], out_channels),
                nn.LayerNorm(out_channels),
            )
            self.output_dim = out_channels
        else:
            self.final_projection = nn.Linear(hidden_dims[-1], out_channels)
            self.output_dim = out_channels

    def execute(self, points: jt.Var) -> jt.Var:
        if len(points.shape) != 3:
            raise ValueError(
                "point cloud must have shape [B,N,3], "
                f"got {tuple(points.shape)}"
            )
        batch_size, num_points, channels = map(int, points.shape)
        if batch_size <= 0:
            raise ValueError("point-cloud batch size must be positive")
        if num_points <= 0:
            raise ValueError("point cloud must contain at least one point")
        if channels != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} point channels, got {channels}"
            )

        point_features = self.mlp(points.float32())
        # Upstream uses torch.max(x, 1)[0]: max over the unordered point axis.
        global_feature = point_features.max(dim=1)
        return self.final_projection(global_feature)


class StateMLP(jt.Module):
    """Robot-state MLP used by the released FlowPolicy configuration."""

    def __init__(
        self,
        input_dim: int,
        layer_dims: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        input_dim = _positive_int("input_dim", input_dim)
        normalized_dims = tuple(
            _positive_int(f"layer_dims[{index}]", int(dim))
            for index, dim in enumerate(layer_dims)
        )
        if not normalized_dims:
            raise ValueError("layer_dims must contain at least one output size")

        layers = []
        dim_in = input_dim
        for index, dim_out in enumerate(normalized_dims):
            layers.append(nn.Linear(dim_in, dim_out))
            if index < len(normalized_dims) - 1:
                layers.append(nn.ReLU())
            dim_in = dim_out
        self.input_dim = input_dim
        self.output_dim = normalized_dims[-1]
        self.mlp = nn.Sequential(*layers)

    def execute(self, state: jt.Var) -> jt.Var:
        if len(state.shape) != 2:
            raise ValueError(
                "robot state must have shape [B,state_dim], "
                f"got {tuple(state.shape)}"
            )
        if int(state.shape[0]) <= 0:
            raise ValueError("state batch size must be positive")
        if int(state.shape[1]) != self.input_dim:
            raise ValueError(
                f"expected state_dim {self.input_dim}, got {int(state.shape[1])}"
            )
        return self.mlp(state.float32())


class FlowPolicyEncoder(jt.Module):
    """Combine XYZ geometry and robot state into one observation feature.

    Defaults target the released configuration rather than full upstream API
    equivalence.  The released MetaWorld path uses point feature dim 64 plus state feature
    dim 64, producing 128 values per observation step.  Two observation
    steps are flattened by the policy later to form a 256-dimensional global
    U-Net condition.
    """

    def __init__(
        self,
        observation_space: Mapping[str, Sequence[int]],
        img_crop_shape: Optional[Sequence[int]] = None,
        out_channel: int = 64,
        state_mlp_size: Sequence[int] = (64, 64),
        pointcloud_encoder_cfg: Optional[Mapping[str, Any]] = None,
        use_pc_color: bool = False,
        pointnet_type: str = "mlp",
    ) -> None:
        super().__init__()
        del img_crop_shape  # Kept only for the upstream constructor contract.
        if not isinstance(observation_space, Mapping):
            raise TypeError("observation_space must be a mapping")
        if "point_cloud" not in observation_space or "agent_pos" not in observation_space:
            raise KeyError("observation_space requires point_cloud and agent_pos")
        if "imagin_robot" in observation_space:
            raise NotImplementedError(
                "the imagined-robot point branch is not validated"
            )
        if use_pc_color:
            raise NotImplementedError("RGB point features are not validated")
        if pointnet_type != "mlp":
            raise NotImplementedError("only pointnet_type='mlp' is validated")

        point_cloud_shape = _shape_tuple(
            "observation_space['point_cloud']",
            observation_space["point_cloud"],
            rank=2,
        )
        state_shape = _shape_tuple(
            "observation_space['agent_pos']",
            observation_space["agent_pos"],
            rank=1,
        )
        if point_cloud_shape[-1] != 3:
            raise NotImplementedError(
                "the released no-color path requires point shape [N,3]"
            )

        config = dict(pointcloud_encoder_cfg or {})
        configured_normal_channel = config.pop("normal_channel", False)
        if configured_normal_channel:
            raise NotImplementedError("point normals are not validated")
        in_channels = config.pop("in_channels", 3)
        configured_out = config.pop("out_channels", out_channel)
        use_layernorm = config.pop("use_layernorm", True)
        final_norm = config.pop("final_norm", "layernorm")
        use_projection = config.pop("use_projection", True)
        if config:
            raise TypeError(
                "unsupported pointcloud_encoder_cfg keys: "
                + ", ".join(sorted(config))
            )
        if int(configured_out) != int(out_channel):
            raise ValueError(
                "pointcloud_encoder_cfg.out_channels must match out_channel"
            )

        self.point_cloud_shape = point_cloud_shape
        self.state_shape = state_shape
        self.use_pc_color = False
        self.pointnet_type = pointnet_type
        self.extractor = PointNetEncoderXYZ(
            in_channels=int(in_channels),
            out_channels=int(configured_out),
            use_layernorm=bool(use_layernorm),
            final_norm=str(final_norm),
            use_projection=bool(use_projection),
        )
        self.state_mlp = StateMLP(state_shape[0], state_mlp_size)
        self.n_output_channels = (
            self.extractor.output_dim + self.state_mlp.output_dim
        )

    def execute(self, observations: Mapping[str, jt.Var]) -> jt.Var:
        if not isinstance(observations, Mapping):
            raise TypeError("observations must be a mapping")
        if "point_cloud" not in observations or "agent_pos" not in observations:
            raise KeyError("observations require point_cloud and agent_pos")
        points = observations["point_cloud"]
        state = observations["agent_pos"]
        if len(points.shape) != 3 or len(state.shape) != 2:
            raise ValueError(
                "point_cloud must be [B,N,3] and agent_pos must be [B,D]"
            )
        if int(points.shape[0]) != int(state.shape[0]):
            raise ValueError("point cloud and state batch sizes must match")
        if tuple(map(int, points.shape[1:])) != self.point_cloud_shape:
            raise ValueError(
                f"expected point shape {self.point_cloud_shape}, "
                f"got {tuple(map(int, points.shape[1:]))}"
            )
        if tuple(map(int, state.shape[1:])) != self.state_shape:
            raise ValueError(
                f"expected state shape {self.state_shape}, "
                f"got {tuple(map(int, state.shape[1:]))}"
            )

        point_feature = self.extractor(points)
        state_feature = self.state_mlp(state)
        return jt.concat((point_feature, state_feature), dim=-1)

    def output_shape(self) -> int:
        return self.n_output_channels

