"""Exponential moving average of the policy weights (paper-faithful EMA target, FPJ-028).

The released code trains a single network and uses it for both the target velocity and
the online prediction; the paper describes an EMA target instead (see D005).  This
module is that missing piece: a separate shadow copy of the **encoder and the U-Net**
that follows the online weights with a decayed average.

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
        self._shadows = shadows
        self.decay = _finite_decay(state["decay"])
        self.warmup_steps = int(state["warmup_steps"])
        self.optimization_step = int(state["optimization_step"])
        if self.optimization_step < 0:
            raise ValueError("EMA optimization_step must be >= 0")
