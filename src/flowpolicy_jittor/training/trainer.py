"""Reusable training loop for the Jittor FlowPolicy port.

Scope (FPJ-024 through FPJ-034):

* AdamW over ``policy.trainable_parameters()`` only, so frozen parameters never
  receive updates;
* a warmup + cosine learning-rate schedule that is a pure function of the step;
* deterministic per-step noise/time sampling derived from ``(seed, step)`` so an
  interrupted run can be resumed without relying on hidden RNG state;
* a JSONL step log with step, learning rate and loss;
* in released-code mode, an optional stored-but-unused EMA shadow;
* in detached mode, a required separate EMA target policy that supplies ``v_r``;
* a **trainer** checkpoint that stores the policy payload, AdamW moments,
  ``n_step``, the global step, the schedule configuration and the EMA state.

The trainer checkpoint is deliberately a different format from the policy
checkpoint; the two must never be interchangeable.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import jittor as jt

from .ema import DetachedEmaTarget, EmaModel

TRAINER_KIND = "flowpolicy-jittor.trainer"
TRAINER_SCHEMA_VERSION = 2
_REQUIRED_PAYLOAD_KEYS = {"kind", "schema_version", "policy", "trainer"}
_REQUIRED_TRAINER_KEYS = {
    "global_step",
    "base_lr",
    "total_steps",
    "warmup_steps",
    "min_lr_ratio",
    "seed",
    "log_every",
    "optimizer",
    "ema",
    "history",
    "target_mode",
    "provenance",
}
_SCHEDULE_KEYS = ("base_lr", "total_steps", "warmup_steps", "min_lr_ratio")


def _finite_float(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _int_at_least(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {number}")
    return number


def _plain_provenance(value: Any, *, path: str = "provenance") -> Any:
    """Normalize immutable resume metadata to deterministic primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return _finite_float(path, value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} keys must be strings")
        return {
            key: _plain_provenance(value[key], path=f"{path}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _plain_provenance(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains unsupported value {value!r}")


def warmup_cosine_lr(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> float:
    """Learning rate for the 1-based step about to run.

    ``step=0`` reports 0.0 (nothing has run yet), the rate ramps linearly to
    ``base_lr`` at ``warmup_steps``, then follows a cosine decay down to
    ``base_lr * min_lr_ratio`` at ``total_steps``.
    """
    base = _finite_float("base_lr", base_lr)
    if base <= 0:
        raise ValueError("base_lr must be positive")
    total = _int_at_least("total_steps", total_steps, 1)
    warmup = _int_at_least("warmup_steps", warmup_steps, 0)
    if warmup > total:
        raise ValueError("warmup_steps must not exceed total_steps")
    ratio = _finite_float("min_lr_ratio", min_lr_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("min_lr_ratio must be within [0, 1]")
    position = _int_at_least("step", step, 0)
    if position > total:
        raise ValueError(f"step must not exceed total_steps={total}, got {position}")

    floor = base * ratio
    if position == 0:
        return 0.0
    if warmup > 0 and position <= warmup:
        return base * position / warmup
    if warmup > 0:
        span = max(1, total - warmup)
        progress = (position - warmup) / span
    else:
        span = max(1, total - 1)
        progress = (position - 1) / span
    progress = min(1.0, max(0.0, progress))
    return floor + (base - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


class Trainer:
    """Minimal deterministic trainer for one FlowPolicy."""

    def __init__(
        self,
        policy: Any,
        *,
        total_steps: int,
        lr: float = 3e-3,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        weight_decay: float = 1e-6,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        seed: int = 0,
        log_path: Optional[Any] = None,
        log_every: int = 1,
        ema_decay: Optional[float] = None,
        ema_warmup_steps: int = 0,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not hasattr(policy, "compute_loss") or not hasattr(
            policy, "trainable_parameters"
        ):
            raise TypeError("policy must be a FlowPolicy-like object")
        self.policy = policy
        self.target_mode = getattr(getattr(policy, "loss_fn", None), "target_mode", None)
        if self.target_mode not in {"code", "detached"}:
            raise ValueError("policy.loss_fn.target_mode must be 'code' or 'detached'")
        if self.target_mode == "detached" and ema_decay is None:
            raise ValueError(
                "target_mode='detached' requires ema_decay so a separate target "
                "policy can be initialized and updated"
            )
        self.base_lr = _finite_float("lr", lr)
        self.total_steps = _int_at_least("total_steps", total_steps, 1)
        self.warmup_steps = _int_at_least("warmup_steps", warmup_steps, 0)
        self.min_lr_ratio = _finite_float("min_lr_ratio", min_lr_ratio)
        if self.warmup_steps > self.total_steps:
            raise ValueError("warmup_steps must not exceed total_steps")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be within [0, 1]")
        self.weight_decay = _finite_float("weight_decay", weight_decay)
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        self.betas = tuple(_finite_float("betas", value) for value in betas)
        if len(self.betas) != 2:
            raise ValueError("betas must have exactly two entries")
        if any(value < 0.0 or value >= 1.0 for value in self.betas):
            raise ValueError("betas must lie in [0, 1)")
        self.eps = _finite_float("eps", eps)
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        self.seed = _int_at_least("seed", seed, 0)
        self.log_every = _int_at_least("log_every", log_every, 1)
        if provenance is not None and not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping or None")
        self.provenance = _plain_provenance(provenance)
        self.global_step = 0
        self.history: List[Dict[str, Any]] = []
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        parameters = list(policy.trainable_parameters())
        if not parameters:
            raise ValueError("policy has no trainable parameters")
        self._parameter_count = len(parameters)
        self.optimizer = jt.optim.AdamW(
            parameters,
            lr=self.learning_rate_at(1),
            eps=self.eps,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        self.ema: Optional[Any] = None
        if ema_decay is not None:
            ema_type = DetachedEmaTarget if self.target_mode == "detached" else EmaModel
            self.ema = ema_type(
                self.policy, decay=ema_decay, warmup_steps=ema_warmup_steps
            )

    # ------------------------------------------------------------- schedule
    def learning_rate_at(self, step: int) -> float:
        return warmup_cosine_lr(
            step,
            base_lr=self.base_lr,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            min_lr_ratio=self.min_lr_ratio,
        )

    @property
    def learning_rate(self) -> float:
        """Rate that the next step will use."""
        return self.learning_rate_at(min(self.global_step + 1, self.total_steps))

    # ---------------------------------------------------------------- train
    def _step_inputs(self, action_shape: Sequence[int], step: int):
        """Deterministic noise and time for ``step``.

        Using ``(seed, step)`` instead of Jittor's global RNG keeps a resumed run
        bit-identical without checkpointing hidden RNG state.  The sampled times are
        clipped to the flow matcher's own lower bound: ``rng.random()`` occasionally
        returns a value below ``eps``, which the matcher rejects outright.
        """
        rng = np.random.default_rng([self.seed, int(step)])
        noise = rng.standard_normal(size=tuple(action_shape)).astype(np.float32)
        eps = float(getattr(getattr(self.policy, "loss_fn", None), "eps", 0.0))
        times = rng.random(size=(int(action_shape[0]),))
        if eps > 0.0:
            # Keep a small margin above `eps`: casting to float32 can round a value that
            # equals `eps` in float64 to just below it, which the matcher rejects.
            times = np.clip(times, eps * (1.0 + 1e-4), 1.0)
        times = times.astype(np.float32)
        return jt.array(noise), jt.array(times)

    def train_step(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        step = self.global_step + 1
        if step > self.total_steps:
            raise RuntimeError(
                f"trainer already finished {self.global_step} of {self.total_steps} steps"
            )
        if not isinstance(batch, Mapping) or "action" not in batch:
            raise KeyError("batch requires an action entry")
        action = np.asarray(batch["action"])
        if action.ndim != 3:
            raise ValueError(f"batch action must be [B, T, A], got shape {action.shape}")

        rate = self.learning_rate_at(step)
        self.optimizer.lr = rate
        noise, times = self._step_inputs(action.shape, step)
        target_policy = None
        if isinstance(self.ema, DetachedEmaTarget):
            target_policy = self.ema.target_policy
        loss, _ = self.policy.compute_loss(
            batch,
            noise=noise,
            t=times,
            target_policy=target_policy,
        )
        value = float(loss.item())
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite loss at step {step}: {value}")
        self.optimizer.step(loss)
        self.global_step = step
        ema_decay: Optional[float] = None
        if self.ema is not None:
            ema_decay = self.ema.update(self.policy)

        record = {"step": step, "lr": rate, "loss": value}
        if ema_decay is not None:
            record["ema_decay"] = ema_decay
        self.history.append(record)
        if self.log_path is not None and (
            step % self.log_every == 0 or step == self.total_steps
        ):
            self._append_log(record)
        return record

    def fit(self, batch: Mapping[str, Any], steps: Optional[int] = None) -> List[Dict[str, Any]]:
        remaining = self.total_steps - self.global_step
        count = remaining if steps is None else _int_at_least("steps", steps, 1)
        if count > remaining:
            raise ValueError(
                f"cannot run {count} more steps, only {remaining} remain of "
                f"total_steps={self.total_steps}"
            )
        return [self.train_step(batch) for _ in range(count)]

    def _append_log(self, record: Mapping[str, Any]) -> None:
        assert self.log_path is not None
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ----------------------------------------------------------- optimizer
    def optimizer_state(self) -> Dict[str, Any]:
        groups = self.optimizer.param_groups
        if len(groups) != 1:
            raise RuntimeError("trainer expects a single parameter group")
        group = groups[0]
        current = list(self.policy.trainable_parameters())
        if len(current) != len(group["params"]) or any(
            not (left is right) for left, right in zip(current, group["params"])
        ):
            raise RuntimeError(
                "optimizer parameters no longer match policy.trainable_parameters(); "
                "rebuild the trainer instead of mutating the policy"
            )
        live_hyperparameters = (
            float(self.optimizer.eps),
            tuple(float(value) for value in self.optimizer.betas),
            float(self.optimizer.weight_decay),
        )
        trainer_hyperparameters = (self.eps, self.betas, self.weight_decay)
        if live_hyperparameters != trainer_hyperparameters:
            raise RuntimeError(
                "optimizer hyperparameters no longer match the trainer; rebuild the "
                "trainer instead of mutating AdamW directly"
            )
        return {
            "param_shapes": [list(int(size) for size in value.shape) for value in current],
            "m": [np.asarray(value.numpy()).copy() for value in group["m"]],
            "values": [np.asarray(value.numpy()).copy() for value in group["values"]],
            "n_step": int(self.optimizer.n_step),
            "eps": self.eps,
            "betas": list(self.betas),
            "weight_decay": self.weight_decay,
        }

    def load_optimizer_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("optimizer state must be a mapping")
        required = {
            "param_shapes",
            "m",
            "values",
            "n_step",
            "eps",
            "betas",
            "weight_decay",
        }
        if set(state) != required:
            raise KeyError(
                "optimizer state keys mismatch: "
                f"missing={sorted(required - set(state))}, "
                f"unexpected={sorted(set(state) - required)}"
            )
        current = list(self.policy.trainable_parameters())
        shapes = [[int(size) for size in value.shape] for value in current]
        stored_shapes = [[int(size) for size in value] for value in state["param_shapes"]]
        if shapes != stored_shapes:
            raise ValueError(
                "trainer checkpoint parameter layout does not match this policy: "
                f"stored={len(stored_shapes)} parameters, current={len(shapes)}"
            )
        moments = state["m"]
        values = state["values"]
        if len(moments) != len(shapes) or len(values) != len(shapes):
            raise ValueError("optimizer moment lists do not match the parameter count")
        validated_moments = []
        validated_values = []
        for index, (moment, value) in enumerate(zip(moments, values)):
            arrays = []
            for label, candidate in (("m", moment), ("values", value)):
                array = np.asarray(candidate)
                if not np.issubdtype(array.dtype, np.floating):
                    raise TypeError(
                        f"optimizer {label}[{index}] must have a floating dtype, "
                        f"got {array.dtype}"
                    )
                if list(array.shape) != shapes[index]:
                    raise ValueError(
                        f"optimizer {label}[{index}] has shape {list(array.shape)}, "
                        f"expected {shapes[index]}"
                    )
                if not np.all(np.isfinite(array)):
                    raise ValueError(
                        f"optimizer {label}[{index}] contains NaN or Inf"
                    )
                arrays.append(array.astype(np.float32, copy=True))
            validated_moments.append(arrays[0])
            validated_values.append(arrays[1])

        n_step = _int_at_least("n_step", state["n_step"], 0)
        eps = _finite_float("optimizer eps", state["eps"])
        if eps <= 0.0:
            raise ValueError("optimizer eps must be positive")
        stored_betas = state["betas"]
        if not isinstance(stored_betas, Sequence) or isinstance(
            stored_betas, (str, bytes)
        ) or len(stored_betas) != 2:
            raise ValueError("optimizer betas must contain exactly two values")
        betas = tuple(_finite_float("optimizer beta", value) for value in stored_betas)
        if any(value < 0.0 or value >= 1.0 for value in betas):
            raise ValueError("optimizer betas must lie in [0, 1)")
        weight_decay = _finite_float(
            "optimizer weight_decay", state["weight_decay"]
        )
        if weight_decay < 0.0:
            raise ValueError("optimizer weight_decay must be non-negative")

        # Commit only after the complete optimizer payload has passed validation.
        group = self.optimizer.param_groups[0]
        group["m"] = [jt.array(value).stop_grad() for value in validated_moments]
        group["values"] = [
            jt.array(value).stop_grad() for value in validated_values
        ]
        self.optimizer.n_step = n_step
        self.eps = eps
        self.betas = betas
        self.weight_decay = weight_decay
        self.optimizer.eps = eps
        self.optimizer.betas = betas
        self.optimizer.weight_decay = weight_decay

    def _runtime_snapshot(self) -> Dict[str, Any]:
        """Capture all mutable trainer state for transactional checkpoint loading."""
        return {
            "policy": {
                name: np.asarray(value).copy()
                for name, value in self.policy.state_dict(to="numpy").items()
            },
            "optimizer": self.optimizer_state(),
            "ema": None if self.ema is None else copy.deepcopy(self.ema.state_dict()),
            "global_step": int(self.global_step),
            "history": copy.deepcopy(self.history),
            "optimizer_lr": float(self.optimizer.lr),
        }

    def _restore_runtime_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        self.policy.load_policy_state_dict(snapshot["policy"])
        self.load_optimizer_state(snapshot["optimizer"])
        if self.ema is not None and snapshot["ema"] is not None:
            self.ema.load_state_dict(snapshot["ema"])
        self.global_step = int(snapshot["global_step"])
        self.history = copy.deepcopy(snapshot["history"])
        self.optimizer.lr = float(snapshot["optimizer_lr"])

    # ------------------------------------------------------------ checkpoint
    def _policy_payload(self) -> Dict[str, Any]:
        """Reuse the strict policy checkpoint writer, then embed its payload."""
        handle, name = tempfile.mkstemp(prefix=".fpj-trainer-policy-", suffix=".ckpt")
        os.close(handle)
        temporary = Path(name)
        try:
            self.policy.save_checkpoint(temporary, overwrite=True)
            payload = jt.load(str(temporary))
        finally:
            temporary.unlink(missing_ok=True)
        if not isinstance(payload, Mapping):
            raise RuntimeError("policy checkpoint payload must be a mapping")
        return dict(payload)

    def _trainer_payload(self) -> Dict[str, Any]:
        return {
            "global_step": int(self.global_step),
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "seed": self.seed,
            "log_every": self.log_every,
            "optimizer": self.optimizer_state(),
            "target_mode": self.target_mode,
            "provenance": copy.deepcopy(self.provenance),
            # `None` means "no EMA was tracked"; it is never silently omitted.
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "history": self.history[-50:],
        }

    def _validate_history(
        self,
        history: Any,
        *,
        global_step: int,
        has_ema: bool,
    ) -> List[Dict[str, Any]]:
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
            raise TypeError("trainer checkpoint history must be a sequence")
        expected_count = min(global_step, 50)
        if len(history) != expected_count:
            raise ValueError(
                "trainer checkpoint history length does not match global_step: "
                f"stored={len(history)}, expected={expected_count}"
            )
        first_step = global_step - expected_count + 1
        validated = []
        for index, entry in enumerate(history):
            if not isinstance(entry, Mapping):
                raise TypeError(f"trainer history[{index}] must be a mapping")
            required = {"step", "lr", "loss"}
            optional = {"ema_decay"} if has_ema else set()
            if set(entry) != required | optional:
                raise KeyError(
                    f"trainer history[{index}] keys mismatch: "
                    f"expected={sorted(required | optional)}, got={sorted(entry)}"
                )
            step = _int_at_least(f"history[{index}].step", entry["step"], 1)
            expected_step = first_step + index
            if step != expected_step:
                raise ValueError(
                    f"trainer history[{index}] step {step} does not match "
                    f"expected {expected_step}"
                )
            lr = _finite_float(f"history[{index}].lr", entry["lr"])
            expected_lr = self.learning_rate_at(step)
            if not math.isclose(lr, expected_lr, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(
                    f"trainer history[{index}] learning rate does not match schedule"
                )
            record = {
                "step": step,
                "lr": lr,
                "loss": _finite_float(f"history[{index}].loss", entry["loss"]),
            }
            if has_ema:
                decay = _finite_float(
                    f"history[{index}].ema_decay", entry["ema_decay"]
                )
                if not 0.0 < decay <= 1.0:
                    raise ValueError("history ema_decay must lie in (0, 1]")
                record["ema_decay"] = decay
            validated.append(record)
        return validated

    def save_checkpoint(self, path: Any, *, overwrite: bool = False) -> None:
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": TRAINER_KIND,
            "schema_version": TRAINER_SCHEMA_VERSION,
            "policy": self._policy_payload(),
            "trainer": self._trainer_payload(),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
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
            raise TypeError("trainer checkpoint payload must be a mapping")
        if payload.get("kind") != TRAINER_KIND:
            raise ValueError(
                f"not a trainer checkpoint: {path} (kind={payload.get('kind')!r}); "
                "policy checkpoints must be loaded with FlowPolicy.load_checkpoint"
            )
        if set(payload) != _REQUIRED_PAYLOAD_KEYS:
            raise KeyError(
                "trainer checkpoint keys mismatch: "
                f"missing={sorted(_REQUIRED_PAYLOAD_KEYS - set(payload))}, "
                f"unexpected={sorted(set(payload) - _REQUIRED_PAYLOAD_KEYS)}"
            )
        if payload["schema_version"] != TRAINER_SCHEMA_VERSION:
            if payload["schema_version"] == 1:
                raise ValueError(
                    "trainer checkpoint schema 1 predates explicit target_mode "
                    "semantics; resume it with the matching older release or start "
                    "a new schema-2 run"
                )
            raise ValueError(
                "trainer checkpoint schema version mismatch: "
                f"loaded={payload['schema_version']!r}, "
                f"current={TRAINER_SCHEMA_VERSION!r}"
            )
        state = payload["trainer"]
        if not isinstance(state, Mapping):
            raise TypeError("trainer checkpoint state must be a mapping")
        if set(state) != _REQUIRED_TRAINER_KEYS:
            raise KeyError(
                "trainer state keys mismatch: "
                f"missing={sorted(_REQUIRED_TRAINER_KEYS - set(state))}, "
                f"unexpected={sorted(set(state) - _REQUIRED_TRAINER_KEYS)}"
            )

        stored_provenance = _plain_provenance(state["provenance"])
        if stored_provenance is not None and not isinstance(
            stored_provenance, Mapping
        ):
            raise TypeError("trainer checkpoint provenance must be a mapping or None")
        if stored_provenance != self.provenance:
            stored_mapping = stored_provenance or {}
            current_mapping = self.provenance or {}
            differing = sorted(
                key
                for key in set(stored_mapping) | set(current_mapping)
                if stored_mapping.get(key) != current_mapping.get(key)
            )
            raise ValueError(
                "trainer checkpoint resume provenance mismatch in fields: "
                + ", ".join(differing)
            )

        stored = {
            "base_lr": _finite_float("base_lr", state["base_lr"]),
            "total_steps": _int_at_least("total_steps", state["total_steps"], 1),
            "warmup_steps": _int_at_least("warmup_steps", state["warmup_steps"], 0),
            "min_lr_ratio": _finite_float("min_lr_ratio", state["min_lr_ratio"]),
        }
        current = {key: getattr(self, key) for key in _SCHEDULE_KEYS}
        if stored != current:
            differing = sorted(
                key for key in _SCHEDULE_KEYS if stored[key] != current[key]
            )
            raise ValueError(
                "trainer checkpoint schedule does not match this trainer: "
                + ", ".join(differing)
            )
        stored_target_mode = state["target_mode"]
        if stored_target_mode not in {"code", "detached"}:
            raise ValueError(
                f"trainer checkpoint target_mode is invalid: {stored_target_mode!r}"
            )
        if stored_target_mode != self.target_mode:
            raise ValueError(
                "trainer checkpoint target_mode does not match this policy: "
                f"stored={stored_target_mode!r}, current={self.target_mode!r}"
            )
        stored_ema = state["ema"]
        if stored_ema is not None and self.ema is None:
            raise ValueError(
                "trainer checkpoint carries an EMA state but this trainer has no EMA; "
                "construct Trainer(..., ema_decay=<value>) to resume it"
            )
        if stored_ema is None and self.ema is not None:
            raise ValueError(
                "trainer checkpoint has no EMA state but this trainer tracks one; "
                "resume with a trainer built without ema_decay, or retrain"
            )
        if self.target_mode == "detached" and not isinstance(
            self.ema, DetachedEmaTarget
        ):
            raise ValueError(
                "detached checkpoint requires a live DetachedEmaTarget controller"
            )
        if self.target_mode == "code" and isinstance(self.ema, DetachedEmaTarget):
            raise ValueError("code checkpoint cannot be loaded into a detached target")

        global_step = _int_at_least("global_step", state["global_step"], 0)
        if global_step > self.total_steps:
            raise ValueError("trainer checkpoint global_step exceeds total_steps")
        seed = _int_at_least("seed", state["seed"], 0)
        if seed != self.seed:
            raise ValueError(
                f"trainer checkpoint seed {seed} does not match trainer seed {self.seed}"
            )
        log_every = _int_at_least("log_every", state["log_every"], 1)
        if log_every != self.log_every:
            raise ValueError(
                "trainer checkpoint log_every does not match this trainer: "
                f"stored={log_every}, current={self.log_every}"
            )
        if not isinstance(state["optimizer"], Mapping):
            raise TypeError("optimizer state must be a mapping")
        optimizer_step = _int_at_least(
            "optimizer n_step", state["optimizer"].get("n_step"), 0
        )
        if optimizer_step != global_step:
            raise ValueError(
                "trainer checkpoint optimizer n_step does not match global_step: "
                f"optimizer={optimizer_step}, global_step={global_step}"
            )
        if stored_ema is not None:
            if not isinstance(stored_ema, Mapping):
                raise TypeError("EMA state must be a mapping")
            ema_step = _int_at_least(
                "EMA optimization_step", stored_ema.get("optimization_step"), 0
            )
            if ema_step != global_step:
                raise ValueError(
                    "trainer checkpoint EMA optimization_step does not match "
                    f"global_step: ema={ema_step}, global_step={global_step}"
                )
        history = self._validate_history(
            state["history"], global_step=global_step, has_ema=stored_ema is not None
        )

        # From this point onward every component load is transactional as a group.
        snapshot = self._runtime_snapshot()
        try:
            # Restore the policy payload through the strict policy loader first, so a
            # corrupted or reconfigured policy aborts before optimizer mutation.
            handle, name = tempfile.mkstemp(
                prefix=".fpj-trainer-restore-", suffix=".ckpt"
            )
            os.close(handle)
            temporary = Path(name)
            try:
                jt.save(dict(payload["policy"]), str(temporary))
                policy_extra = self.policy.load_checkpoint(temporary)
            finally:
                temporary.unlink(missing_ok=True)

            self.load_optimizer_state(state["optimizer"])
            if stored_ema is not None and self.ema is not None:
                self.ema.load_state_dict(stored_ema)
            self.global_step = global_step
            self.history = history
            self.optimizer.lr = self.learning_rate
            return dict(policy_extra)
        except Exception as error:
            try:
                self._restore_runtime_snapshot(snapshot)
            except Exception as rollback_error:
                raise RuntimeError(
                    "trainer checkpoint load failed and rollback also failed: "
                    f"{rollback_error}"
                ) from error
            raise
