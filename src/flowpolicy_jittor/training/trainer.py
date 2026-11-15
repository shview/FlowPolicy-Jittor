"""Reusable training loop for the Jittor FlowPolicy port.

Scope (FPJ-024):

* AdamW over ``policy.trainable_parameters()`` only, so frozen parameters never
  receive updates;
* a warmup + cosine learning-rate schedule that is a pure function of the step;
* deterministic per-step noise/time sampling derived from ``(seed, step)`` so an
  interrupted run can be resumed without relying on hidden RNG state;
* a JSONL step log with step, learning rate and loss;
* an optional paper-faithful EMA target over the encoder and the U-Net
  (``ema_decay``); when it is disabled the checkpoint records ``ema: null``
  explicitly, and when it is enabled the checkpoint carries the averaged weights;
* a **trainer** checkpoint that stores the policy payload, AdamW moments,
  ``n_step``, the global step, the schedule configuration and the EMA state.

The trainer checkpoint is deliberately a different format from the policy
checkpoint; the two must never be interchangeable.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import jittor as jt

from .ema import EmaModel

TRAINER_KIND = "flowpolicy-jittor.trainer"
TRAINER_SCHEMA_VERSION = 1
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
    ) -> None:
        if not hasattr(policy, "compute_loss") or not hasattr(
            policy, "trainable_parameters"
        ):
            raise TypeError("policy must be a FlowPolicy-like object")
        self.policy = policy
        self.base_lr = _finite_float("lr", lr)
        self.total_steps = _int_at_least("total_steps", total_steps, 1)
        self.warmup_steps = _int_at_least("warmup_steps", warmup_steps, 0)
        self.min_lr_ratio = _finite_float("min_lr_ratio", min_lr_ratio)
        if self.warmup_steps > self.total_steps:
            raise ValueError("warmup_steps must not exceed total_steps")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be within [0, 1]")
        self.weight_decay = _finite_float("weight_decay", weight_decay)
        self.betas = tuple(_finite_float("betas", value) for value in betas)
        if len(self.betas) != 2:
            raise ValueError("betas must have exactly two entries")
        self.eps = _finite_float("eps", eps)
        self.seed = _int_at_least("seed", seed, 0)
        self.log_every = _int_at_least("log_every", log_every, 1)
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
        self.ema: Optional[EmaModel] = None
        if ema_decay is not None:
            self.ema = EmaModel(
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
        loss, _ = self.policy.compute_loss(batch, noise=noise, t=times)
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
        group = self.optimizer.param_groups[0]
        for index, (moment, value) in enumerate(zip(moments, values)):
            array = np.asarray(moment, dtype=np.float32)
            if list(array.shape) != shapes[index]:
                raise ValueError(
                    f"optimizer moment {index} has shape {list(array.shape)}, "
                    f"expected {shapes[index]}"
                )
            group["m"][index] = jt.array(array).stop_grad()
            group["values"][index] = jt.array(
                np.asarray(value, dtype=np.float32)
            ).stop_grad()
        self.optimizer.n_step = _int_at_least("n_step", state["n_step"], 0)

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
            # `None` means "no EMA was tracked"; it is never silently omitted.
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "history": self.history[-50:],
        }

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

        global_step = _int_at_least("global_step", state["global_step"], 0)
        if global_step > self.total_steps:
            raise ValueError("trainer checkpoint global_step exceeds total_steps")
        seed = _int_at_least("seed", state["seed"], 0)
        if seed != self.seed:
            raise ValueError(
                f"trainer checkpoint seed {seed} does not match trainer seed {self.seed}"
            )

        # Restore the policy payload through the strict policy loader first, so a
        # corrupted or reconfigured policy aborts before any optimizer mutation.
        handle, name = tempfile.mkstemp(prefix=".fpj-trainer-restore-", suffix=".ckpt")
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
        self.history = [dict(entry) for entry in state["history"]]
        self.optimizer.lr = self.learning_rate
        return dict(policy_extra)
