"""Independent PyTorch mirror of the Jittor model, used only for parity checking (FPJ-028).

This file is deliberately a *second implementation*: it was written from the module
semantics (conv/groupnorm/Mish/FiLM/residual, PointNet MLP, state MLP, consistency flow
matching loss) so that agreement with the Jittor code is evidence and not a tautology.
Module attribute names are kept identical to the Jittor modules, which lets the Jittor
``state_dict`` be loaded with ``strict=True`` -- a mismatch in any weight name or shape
fails loudly instead of silently comparing the wrong tensors.

It runs in the separate ``parity-env`` virtualenv (PyTorch is never imported by the
Jittor training path).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn

torch.set_grad_enabled(True)
DTYPE = torch.float32


# --------------------------------------------------------------------------- blocks
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class ConvTranspose1d(nn.Module):
    """1D transposed convolution as a 2D operator on a singleton axis."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super().__init__()
        self.op = nn.ConvTranspose2d(
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

    def forward(self, x):
        return self.op(x.unsqueeze(-1)).squeeze(-1)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(num_groups, out_channels),
            Mish(),
        )

    def forward(self, x):
        return self.block(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half_dim = self.dim // 2
        frequency_step = math.log(10000.0) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, dtype=DTYPE, device=timesteps.device) * -frequency_step
        )
        angles = timesteps.float().reshape((-1, 1)) * frequencies.reshape((1, -1))
        return torch.concat((torch.sin(angles), torch.cos(angles)), dim=-1)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, num_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, num_groups=n_groups),
            ]
        )
        self.cond_encoder = nn.Sequential(Mish(), nn.Linear(cond_dim, out_channels * 2))
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond=None):
        out = self.blocks[0](x)
        if cond is not None:
            embedding = self.cond_encoder(cond.float()).reshape(
                (int(x.shape[0]), 2, self.out_channels, 1)
            )
            scale = embedding[:, 0, :, :]
            bias = embedding[:, 1, :, :]
            out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=128, down_dims=(512, 1024, 2048), kernel_size=5, n_groups=8, var_fc_hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim
        self.down_dims = tuple(down_dims)
        self.n_groups = n_groups
        self.downsample_factor = 2 ** max(len(self.down_dims) - 1, 0)

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        self.var_est = nn.Sequential(
            nn.Linear(input_dim, var_fc_hidden_dim),
            nn.SiLU(),
            nn.Linear(var_fc_hidden_dim, var_fc_hidden_dim),
            nn.SiLU(),
            nn.Linear(var_fc_hidden_dim, var_fc_hidden_dim),
            nn.SiLU(),
            nn.Linear(var_fc_hidden_dim, 1),
        )
        for parameter in self.var_est.parameters():
            parameter.requires_grad_(False)

        cond_dim = dsed + (global_cond_dim or 0)
        all_dims = (input_dim,) + self.down_dims
        in_out = tuple(zip(all_dims[:-1], all_dims[1:]))

        self.down_modules = nn.ModuleList([])
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                        nn.Identity() if is_last else Downsample1d(dim_out),
                    ]
                )
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for dim_in, dim_out in reversed(in_out[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                        Upsample1d(dim_in),
                    ]
                )
            )

        start_dim = self.down_dims[0]
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size, num_groups=n_groups),
            nn.Conv1d(start_dim, input_dim, kernel_size=1),
        )

    def forward(self, sample, timestep, global_cond=None):
        batch_size = int(sample.shape[0])
        condition = global_cond.float()
        timestep_embedding = self.diffusion_step_encoder(timestep.float())
        global_feature = torch.concat((timestep_embedding, condition), dim=-1)

        x = sample.float().transpose(1, 2)
        skips: List[torch.Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            skips.append(x)
            x = downsample(x)
        for middle in self.mid_modules:
            x = middle(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.concat((x, skips.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        return x.transpose(1, 2)


class PointNetEncoderXYZ(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, use_layernorm=True):
        super().__init__()
        hidden_dims = (64, 128, 256)
        layers = []
        dim_in = in_channels
        for dim_out in hidden_dims:
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.LayerNorm(dim_out) if use_layernorm else nn.Identity())
            layers.append(nn.ReLU())
            dim_in = dim_out
        self.mlp = nn.Sequential(*layers)
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dims[-1], out_channels), nn.LayerNorm(out_channels)
        )

    def forward(self, points):
        features = self.mlp(points.float())
        return self.final_projection(features.max(dim=1).values)


class StateMLP(nn.Module):
    def __init__(self, input_dim, layer_dims=(64, 64)):
        super().__init__()
        layers = []
        dim_in = input_dim
        for index, dim_out in enumerate(layer_dims):
            layers.append(nn.Linear(dim_in, dim_out))
            if index < len(layer_dims) - 1:
                layers.append(nn.ReLU())
            dim_in = dim_out
        self.mlp = nn.Sequential(*layers)

    def forward(self, state):
        return self.mlp(state.float())


class ObsEncoder(nn.Module):
    def __init__(self, state_dim, out_channels=64, state_mlp=(64, 64)):
        super().__init__()
        self.extractor = PointNetEncoderXYZ(out_channels=out_channels)
        self.state_mlp = StateMLP(state_dim, state_mlp)

    def forward(self, observations):
        point_feature = self.extractor(observations["point_cloud"])
        state_feature = self.state_mlp(observations["agent_pos"])
        return torch.concat((point_feature, state_feature), dim=-1)


class SingleFieldNormalizer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer("scale", torch.ones(size, dtype=DTYPE))
        self.register_buffer("offset", torch.zeros(size, dtype=DTYPE))
        self.register_buffer("input_min", torch.zeros(size, dtype=DTYPE))
        self.register_buffer("input_max", torch.zeros(size, dtype=DTYPE))
        self.register_buffer("input_mean", torch.zeros(size, dtype=DTYPE))
        self.register_buffer("input_std", torch.ones(size, dtype=DTYPE))

    def normalize(self, value):
        return value * self.scale + self.offset

    def unnormalize(self, value):
        return (value - self.offset) / self.scale


class LinearNormalizer(nn.Module):
    def __init__(self, sizes: Mapping[str, int]):
        super().__init__()
        for key, size in sizes.items():
            self.add_module(f"normalizer_{key}", SingleFieldNormalizer(int(size)))

    def field(self, key):
        return getattr(self, f"normalizer_{key}")


# ------------------------------------------------------------------------------ loss
class ConsistencyFlowLoss(nn.Module):
    """Paper-mirror of the consistency flow matching objective (code-faithful target)."""

    def __init__(self, eps=1e-2, num_segments=2, boundary=1.0, delta=1e-2, alpha=1e-5, time_scale=1000.0):
        super().__init__()
        self.eps = eps
        self.num_segments = num_segments
        self.boundary = boundary
        self.delta = delta
        self.alpha = alpha
        self.time_scale = time_scale

    def forward(self, model, target, global_cond, noise, t):
        batch_size, horizon, action_dim = target.shape
        r = torch.clamp(t + self.delta, min=self.eps, max=1.0)
        t_expand = t.reshape((-1, 1, 1)).expand(batch_size, horizon, action_dim)
        r_expand = r.reshape((-1, 1, 1)).expand(batch_size, horizon, action_dim)
        x_t = t_expand * target + (1.0 - t_expand) * noise
        x_r = r_expand * target + (1.0 - r_expand) * noise

        indices = torch.clamp(torch.ceil(t * float(self.num_segments)), min=1.0, max=float(self.num_segments))
        segment_ends = indices / float(self.num_segments)
        segment_ends_expand = segment_ends.reshape((-1, 1, 1)).expand(batch_size, horizon, action_dim)
        x_at_segment_ends = segment_ends_expand * target + (1.0 - segment_ends_expand) * noise

        v_t = model(x_t, t * self.time_scale, global_cond)
        v_r = model(x_r, r * self.time_scale, global_cond)
        v_r = torch.where(torch.isfinite(v_r), v_r, torch.zeros_like(v_r))

        f_t = x_t + (segment_ends_expand - t_expand) * v_t
        f_r_euler = x_r + (segment_ends_expand - r_expand) * v_r
        below_boundary_r = r_expand < self.boundary
        f_r = torch.where(below_boundary_r, f_r_euler, x_at_segment_ends)

        far_from_segment_end = (segment_ends - t) > (1.01 * self.delta)
        far_mask = far_from_segment_end.float().reshape((-1, 1, 1)).expand(batch_size, horizon, action_dim)
        below_boundary_t = (t_expand < self.boundary).float()
        velocity_error = below_boundary_t * far_mask * ((v_t - v_r) ** 2)
        velocity_per_sample = velocity_error.reshape((batch_size, -1)).mean(dim=1)

        endpoint_error = (f_t - f_r) ** 2
        endpoint_per_sample = endpoint_error.reshape((batch_size, -1)).mean(dim=1)
        total_per_sample = endpoint_per_sample + self.alpha * velocity_per_sample
        return total_per_sample.mean(), {
            "endpoint_loss": endpoint_per_sample.mean(),
            "velocity_loss": velocity_per_sample.mean(),
        }


# ----------------------------------------------------------------------------- model
class MirrorPolicy(nn.Module):
    def __init__(self, *, num_points, state_dim, action_dim, horizon, n_obs_steps, down_dims, kernel_size, n_groups, embed_dim, pointnet_out, state_mlp=(64, 64), var_fc_hidden_dim=512, normalizer_sizes, loss_config):
        super().__init__()
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.num_points = num_points
        self.state_dim = state_dim
        self.obs_encoder = ObsEncoder(state_dim, pointnet_out, state_mlp)
        obs_feature_dim = pointnet_out + state_mlp[-1]
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_feature_dim * n_obs_steps,
            diffusion_step_embed_dim=embed_dim,
            down_dims=tuple(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            var_fc_hidden_dim=var_fc_hidden_dim,
        )
        self.normalizer = LinearNormalizer(normalizer_sizes)
        self.loss_fn = ConsistencyFlowLoss(**loss_config)

    def prepare_observations(self, observations):
        points = self.normalizer.field("point_cloud").normalize(observations["point_cloud"])
        state = self.normalizer.field("agent_pos").normalize(observations["agent_pos"])
        batch_size = int(points.shape[0])
        flattened = {
            "point_cloud": points[:, : self.n_obs_steps].reshape((-1, self.num_points, 3)),
            "agent_pos": state[:, : self.n_obs_steps].reshape((-1, self.state_dim)),
        }
        features = self.obs_encoder(flattened)
        return batch_size, features.reshape((batch_size, -1))

    def compute_loss(self, observations, actions, noise, t):
        batch_size, global_cond = self.prepare_observations(observations)
        normalized = self.normalizer.field("action").normalize(actions)
        return self.loss_fn(self.model, normalized, global_cond, noise, t)

    def predict_action(self, observations, noise, nfe=1):
        batch_size, global_cond = self.prepare_observations(observations)
        latent = noise.clone()
        dt = 1.0 / float(nfe)
        for index in range(nfe):
            time_value = index / float(nfe) * (1.0 - self.loss_fn.eps) + self.loss_fn.eps
            time = torch.ones((batch_size,), dtype=DTYPE) * time_value
            velocity = self.model(latent, time * self.loss_fn.time_scale, global_cond)
            latent = latent + velocity * dt
        return self.normalizer.field("action").unnormalize(latent)


def load_mirror(bundle: Mapping[str, np.ndarray], config: Mapping[str, object]) -> MirrorPolicy:
    policy = MirrorPolicy(
        num_points=int(config["num_points"]),
        state_dim=int(config["state_dim"]),
        action_dim=int(config["action_dim"]),
        horizon=int(config["horizon"]),
        n_obs_steps=int(config["n_obs_steps"]),
        down_dims=tuple(config["down_dims"]),
        kernel_size=int(config["kernel_size"]),
        n_groups=int(config["n_groups"]),
        embed_dim=int(config["embed_dim"]),
        pointnet_out=int(config["pointnet_out"]),
        state_mlp=tuple(config["state_mlp"]),
        var_fc_hidden_dim=int(config["var_fc_hidden_dim"]),
        normalizer_sizes={
            "point_cloud": 3,
            "agent_pos": int(config["state_dim"]),
            "action": int(config["action_dim"]),
        },
        loss_config=dict(config["loss"]),
    ).to(DTYPE)
    weights = {key[2:]: torch.from_numpy(bundle[key]) for key in bundle if key.startswith("w:")}
    missing, unexpected = policy.load_state_dict(weights, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected weights from the Jittor state dict: {unexpected[:5]}")
    if missing:
        raise SystemExit(f"weights present in the mirror but not in the bundle: {missing[:5]}")
    return policy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    bundle = np.load(args.bundle, allow_pickle=False)
    config = json.loads(str(bundle["config_json"]))
    policy = load_mirror(bundle, config)

    observations = {
        "point_cloud": torch.from_numpy(bundle["obs_point_cloud"]),
        "agent_pos": torch.from_numpy(bundle["obs_agent_pos"]),
    }
    actions = torch.from_numpy(bundle["action"])
    noise = torch.from_numpy(bundle["noise"])
    times = torch.from_numpy(bundle["t"])

    with torch.no_grad():
        action_pred = policy.predict_action(observations, noise.clone(), nfe=int(config["nfe"]))

    policy.zero_grad(set_to_none=True)
    loss, parts = policy.compute_loss(observations, actions, noise, times)
    loss.backward()
    grads = {
        name: parameter.grad.detach().numpy().copy()
        for name, parameter in policy.named_parameters()
        if parameter.grad is not None
    }

    def compare(name: str, torch_value: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
        difference = np.abs(torch_value.astype(np.float64) - reference.astype(np.float64))
        denominator = np.maximum(np.abs(reference.astype(np.float64)), 1e-8)
        return {
            "max_abs": float(difference.max()) if difference.size else 0.0,
            "mean_abs": float(difference.mean()) if difference.size else 0.0,
            "max_rel": float((difference / denominator).max()) if difference.size else 0.0,
            "reference_scale": float(np.abs(reference).max()) if reference.size else 0.0,
        }

    report: Dict[str, object] = {
        "torch_version": torch.__version__,
        "action_pred": compare("action_pred", action_pred.numpy(), bundle["action_pred"]),
        "loss": compare("loss", np.asarray([float(loss)]), np.asarray([float(bundle["loss"])])),
        "gradients": {},
        "gradient_count": len(grads),
        "gradient_coverage": {},
    }
    reference_grads = {key[2:]: bundle[key] for key in bundle if key.startswith("g:")}
    worst: List[Tuple[str, float]] = []
    missing = sorted(set(reference_grads) - set(grads))
    extra = sorted(set(grads) - set(reference_grads))
    for name in sorted(set(reference_grads) & set(grads)):
        stats = compare(name, grads[name], reference_grads[name])
        worst.append((name, stats["max_abs"]))
        report["gradients"][name] = stats
    worst.sort(key=lambda item: -item[1])
    report["gradient_coverage"] = {
        "compared": len(worst),
        "missing_in_mirror": missing[:10],
        "extra_in_mirror": extra[:10],
        "worst": [{"name": name, "max_abs": value} for name, value in worst[:8]],
        "max_abs_over_all_gradients": worst[0][1] if worst else 0.0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "action_pred_max_abs": report["action_pred"]["max_abs"],
        "loss_max_abs": report["loss"]["max_abs"],
        "gradient_max_abs": report["gradient_coverage"]["max_abs_over_all_gradients"],
        "compared_gradients": report["gradient_coverage"]["compared"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
