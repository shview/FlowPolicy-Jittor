"""Top-level Jittor FlowPolicy integration.

The default behavior follows the released FlowPolicy implementation at commit
``86d10c1``.  In particular, the training loss uses the current velocity model
for both consistency predictions.  This is the code-faithful baseline; an EMA
target is not created implicitly here.

The upstream release also slices actions from ``n_obs_steps - 1``.  With its
published ``horizon=4, n_obs_steps=2, n_action_steps=4`` configuration, Python
therefore returns only three actions.  ``action_slice_mode='code'`` preserves
that behavior, while the explicit ``'exact'`` mode returns exactly the
configured number of actions and is treated as a documented correction.
"""

from __future__ import annotations

import copy
import importlib.metadata
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

import jittor as jt
import numpy as np

from flowpolicy_jittor import __version__ as IMPLEMENTATION_VERSION

from .conditional_unet1d import ConditionalUnet1D
from .flow_matching import ConsistencyFlowMatchingLoss
from .normalizer import LinearNormalizer
from .pointnet import FlowPolicyEncoder


CHECKPOINT_SCHEMA_VERSION = 2
UPSTREAM_FLOWPOLICY_COMMIT = "86d10c14972b4eab083abec9026749c340f8e5ed"


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _shape_tuple(name: str, value: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ValueError(f"{name}[{index}] must be a positive integer")
        result.append(int(item))
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return tuple(result)


def _var_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, jt.Var):
        return tuple(int(item) for item in value.shape)
    return tuple(int(item) for item in np.asarray(value).shape)


def _as_float_var(value: Any) -> jt.Var:
    if isinstance(value, jt.Var):
        return value.float32()
    return jt.array(np.asarray(value, dtype=np.float32))


def _plain_config(value: Any) -> Any:
    """Convert constructor metadata to deterministic checkpoint primitives."""
    if isinstance(value, Mapping):
        return {
            str(key): _plain_config(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_config(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"checkpoint configuration contains unsupported value {value!r}")


def _jittor_version() -> str:
    try:
        version = importlib.metadata.version("jittor")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "cannot create or load a checkpoint without installed Jittor metadata"
        ) from error
    version = str(version).strip()
    if not version or version.lower() == "unknown":
        raise RuntimeError("Jittor package version is unavailable")
    return version


def _state_array(key: str, value: Any) -> np.ndarray:
    """Convert one checkpoint tensor to a validated, detached NumPy copy."""
    if isinstance(value, jt.Var):
        array = np.asarray(value.numpy())
    else:
        array = np.asarray(value)
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError(
            f"checkpoint state {key!r} must have a real numeric dtype, "
            f"got {array.dtype}"
        )
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"checkpoint state {key!r} contains NaN or Inf")
    return array.copy()


class FlowPolicy(jt.Module):
    """Jittor policy with the public interface of the released implementation."""

    def __init__(
        self,
        shape_meta: Mapping[str, Any],
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        *,
        obs_as_global_cond: bool = True,
        diffusion_step_embed_dim: int = 128,
        down_dims: Sequence[int] = (512, 1024, 2048),
        kernel_size: int = 5,
        n_groups: int = 8,
        condition_type: str = "film",
        use_down_condition: bool = True,
        use_mid_condition: bool = True,
        use_up_condition: bool = True,
        encoder_output_dim: int = 64,
        state_mlp_size: Sequence[int] = (64, 64),
        crop_shape: Optional[Sequence[int]] = None,
        use_pc_color: bool = False,
        pointnet_type: str = "mlp",
        pointcloud_encoder_cfg: Optional[Mapping[str, Any]] = None,
        eps: float = 1e-2,
        num_segments: int = 2,
        boundary: float = 1.0,
        delta: float = 1e-2,
        alpha: float = 1e-5,
        num_inference_steps: int = 1,
        target_mode: str = "code",
        debug_validate_finite: bool = False,
        action_slice_mode: str = "code",
        var_fc_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if not isinstance(shape_meta, Mapping):
            raise TypeError("shape_meta must be a mapping")
        if "action" not in shape_meta or "obs" not in shape_meta:
            raise KeyError("shape_meta requires action and obs")
        if not isinstance(shape_meta["action"], Mapping) or "shape" not in shape_meta["action"]:
            raise KeyError("shape_meta['action'] requires shape")
        if not isinstance(shape_meta["obs"], Mapping):
            raise TypeError("shape_meta['obs'] must be a mapping")

        action_shape = _shape_tuple(
            "shape_meta['action']['shape']", shape_meta["action"]["shape"]
        )
        if len(action_shape) != 1:
            raise NotImplementedError(
                "only one-dimensional action shapes are validated because the released "
                f"normalizer fits the final axis; got {action_shape}"
            )
        action_dim = action_shape[0]

        obs_shapes: Dict[str, tuple[int, ...]] = {}
        for key in ("point_cloud", "agent_pos"):
            metadata = shape_meta["obs"].get(key)
            if not isinstance(metadata, Mapping) or "shape" not in metadata:
                raise KeyError(f"shape_meta['obs']['{key}'] requires shape")
            obs_shapes[key] = _shape_tuple(
                f"shape_meta['obs']['{key}']['shape']", metadata["shape"]
            )
        if len(obs_shapes["point_cloud"]) != 2 or obs_shapes["point_cloud"][-1] != 3:
            raise ValueError("point_cloud shape must be [N,3]")
        if len(obs_shapes["agent_pos"]) != 1:
            raise ValueError("agent_pos shape must be [D]")

        self.horizon = _positive_int("horizon", horizon)
        self.n_action_steps = _positive_int("n_action_steps", n_action_steps)
        self.n_obs_steps = _positive_int("n_obs_steps", n_obs_steps)
        if self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps must not exceed horizon")
        if not obs_as_global_cond:
            raise NotImplementedError(
                "the released experiment uses obs_as_global_cond=True; "
                "inpainting conditioning is not validated"
            )
        if action_slice_mode not in {"code", "exact"}:
            raise ValueError("action_slice_mode must be 'code' or 'exact'")
        if target_mode not in {"code", "detached"}:
            raise ValueError("target_mode must be 'code' or 'detached'")

        self.action_shape = action_shape
        self.action_dim = action_dim
        self.obs_shapes = obs_shapes
        self.obs_as_global_cond = True
        self.nfe = _positive_int("num_inference_steps", num_inference_steps)
        self.action_slice_mode = action_slice_mode
        self.last_nfe = 0

        encoder_config = dict(pointcloud_encoder_cfg or {})
        encoder_config.setdefault("in_channels", 3)
        encoder_config.setdefault("out_channels", encoder_output_dim)
        encoder_config.setdefault("use_layernorm", True)
        encoder_config.setdefault("final_norm", "layernorm")
        encoder_config.setdefault("normal_channel", False)
        normalized_down_dims = tuple(
            _positive_int(f"down_dims[{index}]", value)
            for index, value in enumerate(down_dims)
        )
        normalized_state_mlp = tuple(
            _positive_int(f"state_mlp_size[{index}]", value)
            for index, value in enumerate(state_mlp_size)
        )
        normalized_crop_shape = (
            None if crop_shape is None else _shape_tuple("crop_shape", crop_shape)
        )
        self.obs_encoder = FlowPolicyEncoder(
            observation_space=obs_shapes,
            img_crop_shape=normalized_crop_shape,
            out_channel=encoder_output_dim,
            state_mlp_size=normalized_state_mlp,
            pointcloud_encoder_cfg=encoder_config,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )
        self.obs_feature_dim = self.obs_encoder.output_shape()
        global_cond_dim = self.obs_feature_dim * self.n_obs_steps
        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=normalized_down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            var_fc_hidden_dim=var_fc_hidden_dim,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )
        self.normalizer = LinearNormalizer()
        self.loss_fn = ConsistencyFlowMatchingLoss(
            eps=eps,
            num_segments=num_segments,
            boundary=boundary,
            delta=delta,
            alpha=alpha,
            time_scale=99.0,
            target_mode=target_mode,
            validate_finite=debug_validate_finite,
        )
        self._construction_config = _plain_config(
            {
                "shape_meta": {
                    "action": list(self.action_shape),
                    "point_cloud": list(self.obs_shapes["point_cloud"]),
                    "agent_pos": list(self.obs_shapes["agent_pos"]),
                },
                "horizon": self.horizon,
                "n_action_steps": self.n_action_steps,
                "n_obs_steps": self.n_obs_steps,
                "obs_as_global_cond": True,
                "diffusion_step_embed_dim": diffusion_step_embed_dim,
                "down_dims": list(normalized_down_dims),
                "kernel_size": kernel_size,
                "n_groups": n_groups,
                "condition_type": condition_type,
                "use_down_condition": use_down_condition,
                "use_mid_condition": use_mid_condition,
                "use_up_condition": use_up_condition,
                "encoder_output_dim": encoder_output_dim,
                "state_mlp_size": list(normalized_state_mlp),
                "crop_shape": (
                    None if normalized_crop_shape is None else list(normalized_crop_shape)
                ),
                "use_pc_color": use_pc_color,
                "pointnet_type": pointnet_type,
                "pointcloud_encoder_cfg": encoder_config,
                "eps": self.loss_fn.eps,
                "num_segments": self.loss_fn.num_segments,
                "boundary": self.loss_fn.boundary,
                "delta": self.loss_fn.delta,
                "alpha": self.loss_fn.alpha,
                "time_scale": self.loss_fn.time_scale,
                "num_inference_steps": self.nfe,
                "target_mode": self.loss_fn.target_mode,
                "debug_validate_finite": self.loss_fn.validate_finite,
                "action_slice_mode": self.action_slice_mode,
                "var_fc_hidden_dim": var_fc_hidden_dim,
            }
        )

    def _runtime_checkpoint_config(self) -> Dict[str, Any]:
        """Rebuild behavior-sensitive metadata from the live policy state."""
        config = copy.deepcopy(self._construction_config)
        config.update(
            {
                "shape_meta": {
                    "action": list(self.action_shape),
                    "point_cloud": list(self.obs_shapes["point_cloud"]),
                    "agent_pos": list(self.obs_shapes["agent_pos"]),
                },
                "horizon": self.horizon,
                "n_action_steps": self.n_action_steps,
                "n_obs_steps": self.n_obs_steps,
                "obs_as_global_cond": self.obs_as_global_cond,
                "eps": self.loss_fn.eps,
                "num_segments": self.loss_fn.num_segments,
                "boundary": self.loss_fn.boundary,
                "delta": self.loss_fn.delta,
                "alpha": self.loss_fn.alpha,
                "time_scale": self.loss_fn.time_scale,
                "num_inference_steps": self.nfe,
                "target_mode": self.loss_fn.target_mode,
                "debug_validate_finite": self.loss_fn.validate_finite,
                "action_slice_mode": self.action_slice_mode,
                # Nested U-Net switches are read by ``forward`` on every call, so they
                # must come from the live module rather than the construction snapshot.
                "use_down_condition": bool(self.model.use_down_condition),
                "use_mid_condition": bool(self.model.use_mid_condition),
                "use_up_condition": bool(self.model.use_up_condition),
            }
        )
        return _plain_config(config)

    @property
    def checkpoint_config(self) -> Dict[str, Any]:
        """Return a detached copy; mutating it cannot alter policy behavior."""
        return copy.deepcopy(self._runtime_checkpoint_config())

    def _assert_runtime_configuration(self) -> Dict[str, Any]:
        runtime = self._runtime_checkpoint_config()
        baseline = self._construction_config
        if runtime != baseline:
            differing = sorted(
                key
                for key in set(runtime) | set(baseline)
                if runtime.get(key) != baseline.get(key)
            )
            raise RuntimeError(
                "policy behavior changed after construction; rebuild the policy "
                "instead of checkpointing mutated fields: " + ", ".join(differing)
            )
        return copy.deepcopy(runtime)

    def trainable_parameters(self):
        """Return all and only parameters safe for Jittor optimizers."""
        return [parameter for parameter in self.parameters() if not parameter.is_stop_grad()]

    def _target_constructor_kwargs(self) -> Dict[str, Any]:
        """Reconstruct this policy's architecture for an independent target copy."""
        config = self._assert_runtime_configuration()
        compact_shape = config.pop("shape_meta")
        config.pop("time_scale")
        config["shape_meta"] = {
            "action": {"shape": compact_shape["action"]},
            "obs": {
                "point_cloud": {"shape": compact_shape["point_cloud"]},
                "agent_pos": {"shape": compact_shape["agent_pos"]},
            },
        }
        # The target never computes its own training loss.  Keeping its loss in code
        # mode prevents accidentally nesting another target requirement.
        config["target_mode"] = "code"
        return config

    def make_target_policy(self) -> "FlowPolicy":
        """Return a separate, exact and fully frozen copy for detached EMA use."""
        self._require_normalizer()
        target = FlowPolicy(**self._target_constructor_kwargs())
        target.load_policy_state_dict(self.state_dict(to="numpy"))
        for parameter in target.parameters():
            parameter.stop_grad()
        target.eval()
        if any(not parameter.is_stop_grad() for parameter in target.parameters()):
            raise RuntimeError("failed to freeze every detached target parameter")
        return target

    def _validate_target_policy(self, target_policy: "FlowPolicy") -> None:
        if not isinstance(target_policy, FlowPolicy):
            raise TypeError("target_policy must be a FlowPolicy")
        if target_policy is self:
            raise ValueError("target_policy must be separate from the online policy")
        online = self.checkpoint_config
        target = target_policy.checkpoint_config
        online.pop("target_mode", None)
        target.pop("target_mode", None)
        if online != target:
            differing = sorted(
                key
                for key in set(online) | set(target)
                if online.get(key) != target.get(key)
            )
            raise ValueError(
                "target_policy architecture does not match the online policy: "
                + ", ".join(differing)
            )
        if any(not parameter.is_stop_grad() for parameter in target_policy.parameters()):
            raise ValueError("every target_policy parameter must have stop_grad enabled")

    def _require_normalizer(self) -> None:
        required = {"action", "agent_pos", "point_cloud"}
        missing = required - set(self.normalizer.keys())
        if missing:
            raise RuntimeError(
                "normalizer is not initialized for fields: " + ", ".join(sorted(missing))
            )

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        if not isinstance(normalizer, LinearNormalizer):
            raise TypeError("normalizer must be a LinearNormalizer")
        required = {"action", "agent_pos", "point_cloud"}
        if set(normalizer.keys()) != required:
            raise KeyError(
                "normalizer fields must be exactly action, agent_pos, point_cloud"
            )
        self._validate_normalizer_dimensions(normalizer)
        candidate = LinearNormalizer()
        candidate.load_state_dict(normalizer.state_dict(to="numpy"))
        self.normalizer = candidate

    def _validate_normalizer_dimensions(self, normalizer: LinearNormalizer) -> None:
        expected = {
            "action": self.action_dim,
            "agent_pos": self.obs_shapes["agent_pos"][-1],
            "point_cloud": self.obs_shapes["point_cloud"][-1],
        }
        mismatched = {
            key: (normalizer[key].feature_dim, feature_dim)
            for key, feature_dim in expected.items()
            if normalizer[key].feature_dim != feature_dim
        }
        if mismatched:
            raise ValueError(
                "normalizer feature dimensions do not match policy shapes: "
                f"{mismatched}"
            )

    def _prepare_observations(self, observations: Mapping[str, Any]):
        self._require_normalizer()
        if not isinstance(observations, Mapping):
            raise TypeError("observations must be a mapping")
        if "point_cloud" not in observations or "agent_pos" not in observations:
            raise KeyError("observations require point_cloud and agent_pos")
        points = _as_float_var(observations["point_cloud"])
        state = _as_float_var(observations["agent_pos"])
        if len(points.shape) != 4 or len(state.shape) != 3:
            raise ValueError(
                "point_cloud must be [B,T,N,3] and agent_pos must be [B,T,D]"
            )
        batch_size = int(points.shape[0])
        if batch_size <= 0 or int(state.shape[0]) != batch_size:
            raise ValueError("observation batch sizes must be positive and equal")
        if int(points.shape[1]) < self.n_obs_steps or int(state.shape[1]) < self.n_obs_steps:
            raise ValueError(
                f"observations must contain at least {self.n_obs_steps} time steps"
            )
        if tuple(int(item) for item in points.shape[2:]) != self.obs_shapes["point_cloud"]:
            raise ValueError(
                f"expected point_cloud trailing shape {self.obs_shapes['point_cloud']}"
            )
        if tuple(int(item) for item in state.shape[2:]) != self.obs_shapes["agent_pos"]:
            raise ValueError(
                f"expected agent_pos trailing shape {self.obs_shapes['agent_pos']}"
            )
        normalized = self.normalizer.normalize(
            {"point_cloud": points, "agent_pos": state}
        )
        flattened = {
            "point_cloud": normalized["point_cloud"][:, : self.n_obs_steps].reshape(
                (-1,) + self.obs_shapes["point_cloud"]
            ),
            "agent_pos": normalized["agent_pos"][:, : self.n_obs_steps].reshape(
                (-1,) + self.obs_shapes["agent_pos"]
            ),
        }
        features = self.obs_encoder(flattened)
        global_cond = features.reshape((batch_size, -1))
        expected_cond = self.obs_feature_dim * self.n_obs_steps
        if tuple(int(item) for item in global_cond.shape) != (batch_size, expected_cond):
            raise RuntimeError("observation encoder produced an unexpected condition shape")
        return batch_size, global_cond

    def _prepare_actions(self, actions: Any, batch_size: int) -> jt.Var:
        actions = _as_float_var(actions)
        if len(actions.shape) < 3:
            raise ValueError("actions must have shape [B,H,...action_shape]")
        if int(actions.shape[0]) != batch_size or int(actions.shape[1]) != self.horizon:
            raise ValueError(
                f"actions must have batch {batch_size} and horizon {self.horizon}"
            )
        trailing = tuple(int(item) for item in actions.shape[2:])
        if trailing != self.action_shape:
            raise ValueError(
                f"expected action trailing shape {self.action_shape}, got {trailing}"
            )
        flattened = actions.reshape((batch_size, self.horizon, self.action_dim))
        return self.normalizer["action"].normalize(flattened)

    def compute_loss(
        self,
        batch: Mapping[str, Any],
        *,
        noise: Optional[jt.Var] = None,
        t: Optional[jt.Var] = None,
        target_policy: Optional["FlowPolicy"] = None,
    ):
        if not isinstance(batch, Mapping) or "obs" not in batch or "action" not in batch:
            raise KeyError("batch requires obs and action")
        if self.loss_fn.target_mode == "code" and target_policy is not None:
            raise ValueError("target_mode='code' does not use a target_policy")
        if self.loss_fn.target_mode == "detached" and target_policy is None:
            raise ValueError(
                "target_mode='detached' requires Trainer(..., ema_decay=<value>) "
                "or an explicit frozen target_policy"
            )
        batch_size, global_cond = self._prepare_observations(batch["obs"])
        normalized_actions = self._prepare_actions(batch["action"], batch_size)
        target_global_cond = None
        target_model = None
        if target_policy is not None:
            self._validate_target_policy(target_policy)
            with jt.no_grad():
                target_batch_size, target_global_cond = target_policy._prepare_observations(
                    batch["obs"]
                )
                target_global_cond = target_global_cond.detach().stop_grad()
            if target_batch_size != batch_size:
                raise RuntimeError("target_policy observation batch size changed")
            target_model = target_policy.model
        loss, parts = self.loss_fn(
            self.model,
            normalized_actions,
            global_cond=global_cond,
            target_global_cond=target_global_cond,
            noise=noise,
            t=t,
            target_model=target_model,
        )
        logs = {"bc_loss": loss, **parts}
        return loss, logs

    def _action_bounds(self) -> tuple[int, int]:
        code_start = self.n_obs_steps - 1
        if self.action_slice_mode == "code":
            return code_start, code_start + self.n_action_steps
        exact_start = min(code_start, self.horizon - self.n_action_steps)
        return exact_start, exact_start + self.n_action_steps

    def predict_action(
        self,
        obs_dict: Mapping[str, Any],
        *,
        noise: Optional[Any] = None,
    ) -> Dict[str, Any]:
        with jt.no_grad():
            batch_size, global_cond = self._prepare_observations(obs_dict)
            expected_shape = (batch_size, self.horizon, self.action_dim)
            if noise is None:
                latent = jt.randn(expected_shape, dtype="float32")
            else:
                latent = _as_float_var(noise)
                if tuple(int(item) for item in latent.shape) != expected_shape:
                    raise ValueError(f"noise must have shape {expected_shape}")
            latent = latent.detach().stop_grad()
            dt = 1.0 / float(self.nfe)
            for index in range(self.nfe):
                time_value = (
                    index / float(self.nfe) * (1.0 - self.loss_fn.eps)
                    + self.loss_fn.eps
                )
                time = jt.ones((batch_size,), dtype="float32") * time_value
                velocity = self.model(
                    latent,
                    time * self.loss_fn.time_scale,
                    local_cond=None,
                    global_cond=global_cond,
                )
                latent = (latent + velocity * dt).detach().stop_grad()
            action_pred = self.normalizer["action"].unnormalize(latent)
            start, end = self._action_bounds()
            action = action_pred[:, start:end]
        self.last_nfe = self.nfe
        return {
            "action": action,
            "action_pred": action_pred,
            "nfe": self.last_nfe,
        }

    def _prepare_policy_state(self, state: Mapping[str, Any]):
        """Validate every state value without mutating this policy."""
        if not isinstance(state, Mapping) or not state:
            raise ValueError("policy state must be a non-empty mapping")
        if any(not isinstance(key, str) for key in state):
            raise TypeError("policy state keys must be strings")
        numeric_state = {
            key: _state_array(key, value)
            for key, value in state.items()
        }

        candidate_normalizer = LinearNormalizer()
        candidate_normalizer.load_prefixed_state_dict(
            numeric_state, prefix="normalizer"
        )
        required = {"action", "agent_pos", "point_cloud"}
        if set(candidate_normalizer.keys()) != required:
            raise KeyError(
                "checkpoint normalizer fields must be exactly action, agent_pos, point_cloud"
            )
        self._validate_normalizer_dimensions(candidate_normalizer)

        current = self.state_dict(to="numpy")
        expected = {
            key: np.asarray(value)
            for key, value in current.items()
            if not key.startswith("normalizer.")
        }
        expected.update(
            {
                f"normalizer.{key}": np.asarray(value)
                for key, value in candidate_normalizer.state_dict(to="numpy").items()
            }
        )
        expected_keys = set(expected)
        loaded_keys = set(numeric_state)
        if loaded_keys != expected_keys:
            raise KeyError(
                "policy state keys mismatch: "
                f"missing={sorted(expected_keys-loaded_keys)}, "
                f"unexpected={sorted(loaded_keys-expected_keys)}"
            )
        mismatched_shapes = {
            key: (numeric_state[key].shape, expected[key].shape)
            for key in expected_keys
            if numeric_state[key].shape != expected[key].shape
        }
        if mismatched_shapes:
            raise ValueError(f"policy state shape mismatch: {mismatched_shapes}")
        mismatched_dtypes = {
            key: (str(numeric_state[key].dtype), str(expected[key].dtype))
            for key in expected_keys
            if numeric_state[key].dtype != expected[key].dtype
        }
        if mismatched_dtypes:
            raise TypeError(f"policy state dtype mismatch: {mismatched_dtypes}")
        return candidate_normalizer, numeric_state

    @staticmethod
    def _state_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
        if set(actual) != set(expected):
            return False
        return all(
            np.array_equal(np.asarray(actual[key]), np.asarray(expected[key]))
            for key in expected
        )

    def _commit_policy_state(
        self,
        candidate_normalizer: LinearNormalizer,
        state: Mapping[str, np.ndarray],
    ) -> None:
        """Commit a prevalidated state, rolling back every module on failure."""
        original_normalizer = self.normalizer
        backup = {
            key: np.asarray(value).copy()
            for key, value in self.state_dict(to="numpy").items()
        }
        try:
            self.normalizer = candidate_normalizer
            self.load_state_dict(state)
            actual = self.state_dict(to="numpy")
            if not self._state_matches(actual, state):
                raise RuntimeError("Jittor did not load every checkpoint tensor exactly")
        except Exception:
            self.normalizer = original_normalizer
            self.load_state_dict(backup)
            restored = self.state_dict(to="numpy")
            if not self._state_matches(restored, backup):
                raise RuntimeError("checkpoint load failed and policy rollback also failed")
            raise

    def load_policy_state_dict(self, state: Mapping[str, Any]) -> None:
        """Strictly validate and atomically load a policy state mapping."""
        candidate_normalizer, numeric_state = self._prepare_policy_state(state)
        self._commit_policy_state(candidate_normalizer, numeric_state)

    def save_checkpoint(
        self,
        path: Any,
        *,
        overwrite: bool = False,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if extra is not None and not isinstance(extra, Mapping):
            raise TypeError("checkpoint extra must be a mapping")
        config = self._assert_runtime_configuration()
        policy_state = {
            key: np.asarray(value).copy()
            for key, value in self.state_dict(to="numpy").items()
        }
        # Saving must reject corrupt in-memory values before touching the target path.
        self._prepare_policy_state(policy_state)
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "upstream_commit": UPSTREAM_FLOWPOLICY_COMMIT,
            "jittor_version": _jittor_version(),
            "config": config,
            "policy": policy_state,
            "extra": dict(extra or {}),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            jt.save(payload, str(temporary))
            if destination.exists() and not overwrite:
                raise FileExistsError(destination)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load_checkpoint(self, path: Any) -> Dict[str, Any]:
        payload = jt.load(str(Path(path)))
        if not isinstance(payload, Mapping):
            raise TypeError("checkpoint payload must be a mapping")
        required = {
            "schema_version",
            "implementation_version",
            "upstream_commit",
            "jittor_version",
            "config",
            "policy",
            "extra",
        }
        if set(payload) != required:
            raise KeyError(
                "checkpoint payload keys mismatch: "
                f"missing={sorted(required-set(payload))}, "
                f"unexpected={sorted(set(payload)-required)}"
            )
        if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint schema version mismatch")
        if payload["implementation_version"] != IMPLEMENTATION_VERSION:
            raise ValueError(
                "checkpoint implementation version mismatch: "
                f"loaded={payload['implementation_version']!r}, "
                f"current={IMPLEMENTATION_VERSION!r}"
            )
        if payload["upstream_commit"] != UPSTREAM_FLOWPOLICY_COMMIT:
            raise ValueError("checkpoint upstream commit mismatch")
        current_jittor_version = _jittor_version()
        if payload["jittor_version"] != current_jittor_version:
            raise ValueError(
                "checkpoint Jittor version mismatch: "
                f"loaded={payload['jittor_version']!r}, "
                f"current={current_jittor_version!r}"
            )
        extra = payload["extra"]
        if not isinstance(extra, Mapping):
            raise TypeError("checkpoint extra must be a mapping")
        current_config = self._assert_runtime_configuration()
        loaded_config = _plain_config(payload["config"])
        if loaded_config != current_config:
            differing = sorted(
                key
                for key in set(loaded_config) | set(current_config)
                if loaded_config.get(key) != current_config.get(key)
            )
            raise ValueError(
                "checkpoint policy configuration mismatch in fields: "
                + ", ".join(differing)
            )
        candidate_normalizer, numeric_state = self._prepare_policy_state(
            payload["policy"]
        )
        self._commit_policy_state(candidate_normalizer, numeric_state)
        return dict(extra)

