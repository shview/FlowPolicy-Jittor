"""Consistency Flow Matching loss used by the released FlowPolicy policy.

The default path ports ``FlowPolicy.compute_loss`` from upstream commit
``86d10c1`` (``flow_policy_3d/policy/flowpolicy.py``).  That released code
evaluates both ``v_t`` and ``v_r`` with the current model and allows gradients
through both calls.  This differs from the paper's EMA-target notation and
from the repository's unused generic ``losses.py``, which detaches ``v_r``.

``target_mode='detached'`` requires an explicit, separate target model.  The
top-level policy/trainer own that target's encoder, weights, EMA update and
checkpoint lifecycle; this loss only consumes the already-detached target
prediction.
"""

from __future__ import annotations

from numbers import Real
from typing import Dict, Optional, Tuple

import jittor as jt
import numpy as np


def _positive_finite(name: str, value: float) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _finite_non_negative(name: str, value: float) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _shape(value: jt.Var) -> Tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def _finite(value: jt.Var) -> bool:
    return bool(np.all(np.isfinite(value.numpy())))


class ConsistencyFlowMatchingLoss(jt.Module):
    """Endpoint and velocity self-consistency loss for action trajectories.

    Parameters match the released policy defaults.  ``boundary`` is the time
    threshold below which the second Euler projection uses ``v_r``; at and
    above the threshold it uses the analytically known segment endpoint.
    """

    def __init__(
        self,
        *,
        eps: float = 1e-2,
        num_segments: int = 2,
        boundary: float = 1.0,
        delta: float = 1e-2,
        alpha: float = 1e-5,
        time_scale: float = 99.0,
        target_mode: str = "code",
        validate_finite: bool = False,
    ) -> None:
        super().__init__()
        self.eps = _positive_finite("eps", eps)
        if self.eps >= 1.0:
            raise ValueError("eps must be smaller than 1")
        if not isinstance(num_segments, int) or isinstance(num_segments, bool):
            raise TypeError("num_segments must be an integer")
        if num_segments <= 0:
            raise ValueError("num_segments must be positive")
        self.num_segments = int(num_segments)
        self.boundary = _finite_non_negative("boundary", boundary)
        if self.boundary > 1.0:
            raise ValueError("boundary must lie in [0, 1]")
        self.delta = _positive_finite("delta", delta)
        if self.delta > 1.0:
            raise ValueError("delta must not exceed 1")
        self.alpha = _finite_non_negative("alpha", alpha)
        self.time_scale = _positive_finite("time_scale", time_scale)
        if target_mode not in {"code", "detached"}:
            raise ValueError("target_mode must be 'code' or 'detached'")
        self.target_mode = target_mode
        self.validate_finite = bool(validate_finite)

    def _validate_trajectory(self, value: jt.Var, name: str) -> jt.Var:
        if not isinstance(value, jt.Var):
            value = jt.array(np.asarray(value, dtype=np.float32))
        value = value.float32()
        if len(value.shape) != 3:
            raise ValueError(f"{name} must have shape [B,H,D], got {_shape(value)}")
        if any(size <= 0 for size in _shape(value)):
            raise ValueError(f"{name} dimensions must be positive")
        # A full ``.numpy()`` finite check synchronizes and copies the entire
        # batch to the host.  Keep it as an explicit debug option rather than
        # putting a device sync in every training step.
        if self.validate_finite and not _finite(value):
            raise ValueError(f"{name} must contain only finite values")
        return value

    def _time_pair(self, batch_size: int, t: Optional[jt.Var]) -> Tuple[jt.Var, jt.Var]:
        if t is None:
            t_value = jt.rand((batch_size,), dtype="float32")
            t_value = t_value * (1.0 - self.eps) + self.eps
        else:
            if not isinstance(t, jt.Var):
                t = jt.array(np.asarray(t, dtype=np.float32))
            t_value = t.float32()
            if _shape(t_value) != (batch_size,):
                raise ValueError(
                    f"t must have shape [{batch_size}], got {_shape(t_value)}"
                )
            if not _finite(t_value):
                raise ValueError("t must contain only finite values")
            t_np = t_value.numpy()
            if np.any(t_np < self.eps) or np.any(t_np > 1.0):
                raise ValueError(f"t must lie in [{self.eps}, 1]")
        t_value = t_value.detach().stop_grad()
        r_value = jt.clamp(t_value + self.delta, min_v=self.eps, max_v=1.0)
        return t_value, r_value.stop_grad()

    def _segment_ends(self, t: jt.Var) -> jt.Var:
        # Uniform linspace + searchsorted(side='left') from the released code
        # is exactly ceil(t * K) / K for t in (0, 1].
        indices = jt.ceil(t * float(self.num_segments))
        indices = jt.clamp(indices, min_v=1.0, max_v=float(self.num_segments))
        return (indices / float(self.num_segments)).stop_grad()

    @staticmethod
    def _expand_time(value: jt.Var, horizon: int, action_dim: int) -> jt.Var:
        return value.reshape((-1, 1, 1)).repeat((1, horizon, action_dim))

    @staticmethod
    def _call_model(model, x, time, global_cond, local_cond):
        return model(
            x,
            time,
            local_cond=local_cond,
            global_cond=global_cond,
        )

    def execute(
        self,
        model,
        target: jt.Var,
        *,
        global_cond: Optional[jt.Var] = None,
        local_cond: Optional[jt.Var] = None,
        noise: Optional[jt.Var] = None,
        t: Optional[jt.Var] = None,
        condition_mask: Optional[jt.Var] = None,
        condition_data: Optional[jt.Var] = None,
        target_model=None,
        target_global_cond: Optional[jt.Var] = None,
        target_local_cond: Optional[jt.Var] = None,
    ) -> Tuple[jt.Var, Dict[str, jt.Var]]:
        target = self._validate_trajectory(target, "target")
        batch_size, horizon, action_dim = _shape(target)

        if noise is None:
            noise_value = jt.randn(target.shape, dtype="float32")
        else:
            noise_value = self._validate_trajectory(noise, "noise")
            if _shape(noise_value) != _shape(target):
                raise ValueError("noise and target must have identical shapes")
        noise_value = noise_value.detach().stop_grad()

        t_value, r_value = self._time_pair(batch_size, t)
        t_expand = self._expand_time(t_value, horizon, action_dim)
        r_expand = self._expand_time(r_value, horizon, action_dim)
        x_t = t_expand * target + (1.0 - t_expand) * noise_value
        x_r = r_expand * target + (1.0 - r_expand) * noise_value

        mask = None
        if condition_mask is not None or condition_data is not None:
            if condition_mask is None or condition_data is None:
                raise ValueError("condition_mask and condition_data must be supplied together")
            if not isinstance(condition_mask, jt.Var):
                condition_mask = jt.array(np.asarray(condition_mask, dtype=np.bool_))
            condition_data = self._validate_trajectory(condition_data, "condition_data")
            if _shape(condition_mask) != _shape(target):
                raise ValueError("condition_mask and target must have identical shapes")
            if _shape(condition_data) != _shape(target):
                raise ValueError("condition_data and target must have identical shapes")
            mask = condition_mask.bool()
            x_t = jt.where(mask, condition_data, x_t)
            x_r = jt.where(mask, condition_data, x_r)

        segment_ends = self._segment_ends(t_value)
        segment_ends_expand = self._expand_time(segment_ends, horizon, action_dim)
        x_at_segment_ends = (
            segment_ends_expand * target
            + (1.0 - segment_ends_expand) * noise_value
        )

        v_t = self._call_model(
            model,
            x_t,
            t_value * self.time_scale,
            global_cond,
            local_cond,
        )
        if _shape(v_t) != _shape(target):
            raise ValueError(
                f"model v_t shape must be {_shape(target)}, got {_shape(v_t)}"
            )

        if self.boundary == 0.0:
            v_r = None
        else:
            if self.target_mode == "detached":
                if target_model is None:
                    raise ValueError(
                        "target_mode='detached' requires an explicit target_model; "
                        "the loss module does not create or update EMA parameters"
                    )
                if target_model is model:
                    raise ValueError(
                        "detached target_model must be a distinct model; "
                        "using the current model is not an EMA target"
                    )
                with jt.no_grad():
                    v_r = self._call_model(
                        target_model,
                        x_r,
                        r_value * self.time_scale,
                        (
                            global_cond
                            if target_global_cond is None
                            else target_global_cond
                        ),
                        (
                            local_cond
                            if target_local_cond is None
                            else target_local_cond
                        ),
                    ).detach().stop_grad()
            else:
                if target_model is not None and target_model is not model:
                    raise ValueError(
                        "code mode uses the current model for both predictions"
                    )
                v_r = self._call_model(
                    model,
                    x_r,
                    r_value * self.time_scale,
                    global_cond,
                    local_cond,
                )
            if _shape(v_r) != _shape(target):
                raise ValueError(
                    f"model v_r shape must be {_shape(target)}, got {_shape(v_r)}"
                )
            v_r = jt.where(jt.isfinite(v_r), v_r, jt.zeros_like(v_r))

        if mask is not None:
            v_t = jt.where(mask, condition_data, v_t)
            if v_r is not None:
                v_r = jt.where(mask, condition_data, v_r)

        f_t = x_t + (segment_ends_expand - t_expand) * v_t
        if self.boundary == 0.0:
            f_r = x_at_segment_ends
            velocity_per_sample = jt.zeros((batch_size,), dtype="float32")
        else:
            f_r_euler = x_r + (segment_ends_expand - r_expand) * v_r
            below_boundary_r = r_expand < self.boundary
            f_r = jt.where(below_boundary_r, f_r_euler, x_at_segment_ends)

            far_from_segment_end = (
                (segment_ends - t_value) > (1.01 * self.delta)
            )
            far_mask = self._expand_time(
                far_from_segment_end.float32(), horizon, action_dim
            )
            below_boundary_t = (t_expand < self.boundary).float32()
            velocity_error = (
                below_boundary_t * far_mask * ((v_t - v_r) ** 2)
            )
            velocity_per_sample = velocity_error.reshape((batch_size, -1)).mean(dim=1)

        endpoint_error = (f_t - f_r) ** 2
        endpoint_per_sample = endpoint_error.reshape((batch_size, -1)).mean(dim=1)
        total_per_sample = endpoint_per_sample + self.alpha * velocity_per_sample
        total = total_per_sample.mean()
        components = {
            "loss": total,
            "endpoint_loss": endpoint_per_sample.mean(),
            "velocity_loss": velocity_per_sample.mean(),
            "t_mean": t_value.mean(),
            "r_mean": r_value.mean(),
        }
        return total, components

