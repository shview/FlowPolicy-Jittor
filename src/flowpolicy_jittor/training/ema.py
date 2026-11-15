"""EMA utilities for stored shadows and a live detached target policy.

The released code trains a single network and uses it for both the target velocity and
the online prediction; the paper describes an EMA target instead (see D005).  This
``EmaModel`` is the released-code experiment's stored shadow: it can be copied into
an online policy for evaluation but is not used by the loss. ``DetachedEmaTarget``
instead owns a separate frozen policy that the detached loss evaluates directly.

Design notes:

* only model weights are averaged.  The `normalizer` holds input statistics, not
  weights, and the frozen `var_est` module is an upstream-compatibility artefact that
  never takes part in the loss -- averaging either of them would be misleading.
* the effective decay follows the usual warmup ramp
  ``min(decay, (1 + step) / (10 + step))`` so the average is not dominated by the
  random initialisation during the first few steps.
* `store`/`restore` make evaluation exact: after ``restore`` the online weights are
  bit-identical to what they were before ``store``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import jittor as jt

# Matched against every dotted path segment, so `normalizer.*`, `model.var_est.*` and
# any future nesting are all excluded without hard-coding full names.
_EXCLUDED_SEGMENTS = frozenset({"normalizer", "var_est"})


def _excluded(name: str) -> bool:
    return any(segment in _EXCLUDED_SEGMENTS for segment in name.split("."))


def _finite_decay(value: Any) -> float:
    decay = float(value)
    if not math.isfinite(decay):
        raise ValueError(f"decay must be finite, got {value!r}")
    if not 0.0 < decay <= 1.0:
        raise ValueError(f"decay must be within (0, 1], got {decay}")
    return decay


def _float_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"EMA can only average floating point tensors, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError("EMA tensors must contain only finite values")
    return array.astype(np.float32, copy=True)


class EmaModel:
    """Shadow copy of the policy weights with a decayed average.

    ``EmaModel(policy)`` snapshots the current weights, ``update(policy)`` folds the
    online weights into the average, and ``copy_to(policy)`` writes the average back so
    the same rollout code can evaluate the EMA weights.
    """

    def __init__(self, policy: Any, *, decay: float = 0.999, warmup_steps: int = 0) -> None:
        if not hasattr(policy, "state_dict") or not hasattr(policy, "load_policy_state_dict"):
            raise TypeError("policy must expose state_dict and load_policy_state_dict")
        self.decay = _finite_decay(decay)
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, (int, np.integer)):
            raise TypeError("warmup_steps must be an integer")
        if int(warmup_steps) < 0:
            raise ValueError("warmup_steps must be >= 0")
        self.warmup_steps = int(warmup_steps)
        self.optimization_step = 0
        self._names: Tuple[str, ...] = ()
        self._shadows: Dict[str, np.ndarray] = {}
        self._backup: Optional[Dict[str, np.ndarray]] = None
        self._reset_from(policy)

    # ------------------------------------------------------------- snapshot
    def _select(self, state: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        selected: Dict[str, np.ndarray] = {}
        for name, value in state.items():
            if _excluded(name):
                continue
            selected[name] = _float_array(value)
        if not selected:
            raise ValueError("policy exposes no weights to average")
        return selected

    def _reset_from(self, policy: Any) -> None:
        selected = self._select(policy.state_dict(to="numpy"))
        self._names = tuple(sorted(selected))
        self._shadows = {name: selected[name].copy() for name in self._names}

    # ---------------------------------------------------------------- decay
    def effective_decay(self) -> float:
        """Decay actually applied on the next update.

        The usual ramp ``min(decay, (1 + step) / (10 + step))`` keeps the average from
        freezing the random initialisation; with ``warmup_steps > 0`` the ramp is
        additionally blended towards the configured decay over that many steps.
        """
        step = self.optimization_step
        ramp = float(min(self.decay, (1.0 + step) / (10.0 + step)))
        if self.warmup_steps <= 0:
            return ramp
        progress = min(1.0, step / float(self.warmup_steps))
        return float(ramp + (self.decay - ramp) * progress)

    # --------------------------------------------------------------- update
    def update(self, policy: Any) -> float:
        """Fold the current online weights into the average; returns the decay used."""
        decay = self.effective_decay()
        current = self._select(policy.state_dict(to="numpy"))
        if tuple(sorted(current)) != self._names:
            missing = sorted(set(self._names) - set(current))
            extra = sorted(set(current) - set(self._names))
            raise ValueError(
                "policy weights changed since the EMA was created: "
                f"missing={missing}, unexpected={extra}"
            )
        for name in self._names:
            shadow = self._shadows[name]
            online = current[name]
            if shadow.shape != online.shape:
                raise ValueError(
                    f"weight {name!r} changed shape: ema={shadow.shape}, online={online.shape}"
                )
            shadow *= decay
            shadow += (1.0 - decay) * online
        self.optimization_step += 1
        return decay

    # ------------------------------------------------------------ evaluation
    def copy_to(self, policy: Any) -> None:
        """Write the averaged weights into ``policy``."""
        merged = {name: value.copy() for name, value in policy.state_dict(to="numpy").items()}
        for name in self._names:
            merged[name] = self._shadows[name].copy()
        policy.load_policy_state_dict(merged)

    def store(self, policy: Any) -> None:
        """Save the online weights so ``restore`` can put them back exactly."""
        self._backup = {
            name: np.asarray(value).copy()
            for name, value in policy.state_dict(to="numpy").items()
        }

    def restore(self, policy: Any) -> None:
        if self._backup is None:
            raise RuntimeError("restore() requires store() first")
        policy.load_policy_state_dict(self._backup)

    # ------------------------------------------------------------ checkpoint
    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "optimization_step": self.optimization_step,
            "names": list(self._names),
            "parameters": [self._shadows[name].copy() for name in self._names],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {"decay", "warmup_steps", "optimization_step", "names", "parameters"}
        if not isinstance(state, Mapping):
            raise TypeError("EMA state must be a mapping")
        if set(state) != required:
            raise KeyError(
                "EMA state keys mismatch: "
                f"missing={sorted(required - set(state))}, "
                f"unexpected={sorted(set(state) - required)}"
            )
        names = tuple(state["names"])
        if names != self._names:
            raise ValueError(
                "EMA weight names do not match this policy: "
                f"stored={len(names)} current={len(self._names)}"
            )
        parameters = list(state["parameters"])
        if len(parameters) != len(self._names):
            raise ValueError("EMA parameter list does not match the stored names")
        shadows = {}
        for name, value in zip(self._names, parameters):
            array = _float_array(value)
            if array.shape != self._shadows[name].shape:
                raise ValueError(
                    f"EMA weight {name!r} has shape {array.shape}, "
                    f"expected {self._shadows[name].shape}"
                )
            shadows[name] = array
        decay = _finite_decay(state["decay"])
        warmup_steps = state["warmup_steps"]
        if isinstance(warmup_steps, bool) or not isinstance(
            warmup_steps, (int, np.integer)
        ) or int(warmup_steps) < 0:
            raise ValueError("EMA warmup_steps must be a non-negative integer")
        optimization_step = state["optimization_step"]
        if isinstance(optimization_step, bool) or not isinstance(
            optimization_step, (int, np.integer)
        ) or int(optimization_step) < 0:
            raise ValueError("EMA optimization_step must be >= 0")
        # Commit only after every tensor and scalar has passed validation.
        self._shadows = shadows
        self.decay = decay
        self.warmup_steps = int(warmup_steps)
        self.optimization_step = int(optimization_step)


class DetachedEmaTarget:
    """Live, frozen target policy updated only after each online optimizer step.

    The target starts as an exact copy of the online policy.  A training step reads
    the pre-update target for ``v_r``, updates the online policy, and then calls
    :meth:`update`; therefore the target cannot be changed by the online optimizer.
    """

    KIND = "flowpolicy-jittor.detached-ema-target"

    def __init__(
        self,
        policy: Any,
        *,
        decay: float = 0.999,
        warmup_steps: int = 0,
    ) -> None:
        if not hasattr(policy, "make_target_policy"):
            raise TypeError("policy must expose make_target_policy")
        self.decay = _finite_decay(decay)
        if isinstance(warmup_steps, bool) or not isinstance(
            warmup_steps, (int, np.integer)
        ):
            raise TypeError("warmup_steps must be an integer")
        if int(warmup_steps) < 0:
            raise ValueError("warmup_steps must be >= 0")
        self.warmup_steps = int(warmup_steps)
        self.optimization_step = 0
        self.target_policy = policy.make_target_policy()
        self._names = self._parameter_names(policy, self.target_policy)

    @staticmethod
    def _averaged_parameters(policy: Any) -> Dict[str, Any]:
        return {
            name: parameter
            for name, parameter in policy.named_parameters()
            if not _excluded(name)
        }

    @classmethod
    def _parameter_names(cls, online: Any, target: Any) -> Tuple[str, ...]:
        online_parameters = cls._averaged_parameters(online)
        target_parameters = cls._averaged_parameters(target)
        if set(online_parameters) != set(target_parameters):
            raise ValueError(
                "detached target parameter names do not match the online policy: "
                f"missing={sorted(set(online_parameters) - set(target_parameters))}, "
                f"unexpected={sorted(set(target_parameters) - set(online_parameters))}"
            )
        if not online_parameters:
            raise ValueError("policy exposes no weights for the detached EMA target")
        for name in online_parameters:
            if tuple(online_parameters[name].shape) != tuple(target_parameters[name].shape):
                raise ValueError(f"detached target parameter {name!r} has a different shape")
        return tuple(sorted(online_parameters))

    def effective_decay(self) -> float:
        step = self.optimization_step
        ramp = float(min(self.decay, (1.0 + step) / (10.0 + step)))
        if self.warmup_steps <= 0:
            return ramp
        progress = min(1.0, step / float(self.warmup_steps))
        return float(ramp + (self.decay - ramp) * progress)

    def update(self, policy: Any) -> float:
        """EMA-update the frozen target from the current online weights."""
        names = self._parameter_names(policy, self.target_policy)
        if names != self._names:
            raise ValueError("policy parameters changed since target initialization")
        online = self._averaged_parameters(policy)
        target = self._averaged_parameters(self.target_policy)
        decay = self.effective_decay()
        with jt.no_grad():
            for name in self._names:
                value = target[name] * decay + online[name].detach() * (1.0 - decay)
                target[name].update(value.detach())
        self.optimization_step += 1
        if any(
            not parameter.is_stop_grad()
            for parameter in self.target_policy.parameters()
        ):
            raise RuntimeError("detached target unexpectedly acquired trainable parameters")
        return decay

    def state_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.KIND,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "optimization_step": self.optimization_step,
            "names": list(self._names),
            "target_policy": {
                name: np.asarray(value).copy()
                for name, value in self.target_policy.state_dict(to="numpy").items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {
            "kind",
            "decay",
            "warmup_steps",
            "optimization_step",
            "names",
            "target_policy",
        }
        if not isinstance(state, Mapping):
            raise TypeError("detached EMA target state must be a mapping")
        if set(state) != required:
            raise KeyError(
                "detached EMA target state keys mismatch: "
                f"missing={sorted(required - set(state))}, "
                f"unexpected={sorted(set(state) - required)}"
            )
        if state["kind"] != self.KIND:
            raise ValueError(
                "EMA state kind does not describe a detached target: "
                f"{state['kind']!r}"
            )
        names = tuple(state["names"])
        if names != self._names:
            raise ValueError(
                "detached EMA target names do not match this policy: "
                f"stored={len(names)} current={len(self._names)}"
            )
        decay = _finite_decay(state["decay"])
        warmup_steps = state["warmup_steps"]
        if isinstance(warmup_steps, bool) or not isinstance(
            warmup_steps, (int, np.integer)
        ) or int(warmup_steps) < 0:
            raise ValueError("detached EMA warmup_steps must be a non-negative integer")
        optimization_step = state["optimization_step"]
        if isinstance(optimization_step, bool) or not isinstance(
            optimization_step, (int, np.integer)
        ) or int(optimization_step) < 0:
            raise ValueError(
                "detached EMA optimization_step must be a non-negative integer"
            )
        # The target policy loader is itself atomic.  Validate every scalar before it
        # so malformed metadata cannot leave a valid-but-uncommitted target state.
        self.target_policy.load_policy_state_dict(state["target_policy"])
        for parameter in self.target_policy.parameters():
            parameter.stop_grad()
        self.decay = decay
        self.warmup_steps = int(warmup_steps)
        self.optimization_step = int(optimization_step)
