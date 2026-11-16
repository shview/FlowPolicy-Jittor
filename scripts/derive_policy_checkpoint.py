"""Derive an evaluation-only policy checkpoint from a trainer checkpoint.

The training checkpoint contains both the online policy payload and the EMA state.
This tool can select either weight set and can relabel the explicit action-slice
configuration as ``code`` (the released three-step behavior) or ``exact`` (four
steps).  It changes no training state and never overwrites the source checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


TRAINER_KIND = "flowpolicy-jittor.trainer"
DETACHED_KIND = "flowpolicy-jittor.detached-ema-target"
POLICY_KEYS = {
    "schema_version",
    "implementation_version",
    "upstream_commit",
    "jittor_version",
    "config",
    "policy",
    "extra",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _validate_replacement(name: str, online: Any, replacement: Any) -> None:
    try:
        online_array = np.asarray(online)
        replacement_array = np.asarray(replacement)
    except Exception as exc:
        raise TypeError(f"parameter {name!r} cannot be converted to an array") from exc
    if online_array.shape != replacement_array.shape:
        raise ValueError(
            f"parameter {name!r} shape mismatch: "
            f"online={online_array.shape}, replacement={replacement_array.shape}"
        )
    if online_array.dtype != replacement_array.dtype:
        raise ValueError(
            f"parameter {name!r} dtype mismatch: "
            f"online={online_array.dtype}, replacement={replacement_array.dtype}"
        )
    try:
        finite = bool(np.isfinite(replacement_array).all())
    except TypeError as exc:
        raise TypeError(f"parameter {name!r} must have a numeric dtype") from exc
    if not finite:
        raise ValueError(f"parameter {name!r} replacement contains non-finite values")


def derive_payload(
    trainer_payload: Mapping[str, Any],
    *,
    weights: str,
    action_slice_mode: str,
    source_sha256: str,
) -> dict[str, Any]:
    if trainer_payload.get("kind") != TRAINER_KIND:
        raise ValueError("input is not a FlowPolicy-Jittor trainer checkpoint")
    if weights not in {"online", "ema"}:
        raise ValueError("weights must be 'online' or 'ema'")
    if action_slice_mode not in {"code", "exact"}:
        raise ValueError("action_slice_mode must be 'code' or 'exact'")
    policy_payload = copy.deepcopy(dict(_mapping(trainer_payload.get("policy"), "policy")))
    if set(policy_payload) != POLICY_KEYS:
        raise KeyError("trainer policy payload has unexpected keys")
    config = dict(_mapping(policy_payload["config"], "policy config"))
    state = dict(_mapping(policy_payload["policy"], "policy state"))
    target_mode = config.get("target_mode")
    if target_mode not in {"code", "detached"}:
        raise ValueError(f"unsupported target_mode in policy config: {target_mode!r}")

    if weights == "ema":
        trainer_state = _mapping(trainer_payload.get("trainer"), "trainer state")
        ema = _mapping(trainer_state.get("ema"), "trainer EMA state")
        if target_mode == "code":
            names = list(ema.get("names", []))
            parameters = list(ema.get("parameters", []))
            if not names or len(names) != len(parameters):
                raise ValueError("stored EMA names and parameters do not align")
            if len(names) != len(set(names)):
                raise ValueError("stored EMA names must be unique")
            missing = sorted(set(names) - set(state))
            if missing:
                raise KeyError(f"EMA names are absent from the policy state: {missing}")
            for name, value in zip(names, parameters):
                _validate_replacement(name, state[name], value)
                state[name] = copy.deepcopy(value)
        else:
            if ema.get("kind") != DETACHED_KIND:
                raise ValueError("detached trainer does not contain a detached EMA target")
            target = dict(_mapping(ema.get("target_policy"), "detached target policy"))
            if set(target) != set(state):
                raise KeyError("detached target state keys do not match online policy")
            for name, value in target.items():
                _validate_replacement(name, state[name], value)
            state = copy.deepcopy(target)

    config["action_slice_mode"] = action_slice_mode
    extra = dict(_mapping(policy_payload.get("extra"), "policy extra"))
    extra["derived_evaluation"] = {
        "source_kind": TRAINER_KIND,
        "source_sha256": source_sha256,
        "weights": weights,
        "action_slice_mode": action_slice_mode,
        "claim_boundary": (
            "evaluation-only derivative; training configuration and optimizer state "
            "remain those of the source trainer checkpoint"
        ),
    }
    policy_payload["config"] = config
    policy_payload["policy"] = state
    policy_payload["extra"] = extra
    return policy_payload


def write_checkpoint(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    import jittor as jt

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(handle)
    temporary = Path(name)
    try:
        jt.save(dict(payload), str(temporary))
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_report(path: Path, report: Mapping[str, Any], *, overwrite: bool) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(handle)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", choices=("online", "ema"), required=True)
    parser.add_argument(
        "--action-slice-mode", choices=("code", "exact"), default="code"
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional JSON sidecar for later evidence validation",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    import jittor as jt

    if args.out.resolve() == args.trainer.resolve():
        parser.error("--out must not overwrite the source trainer checkpoint")
    if args.report is not None:
        if args.report.resolve() in {args.trainer.resolve(), args.out.resolve()}:
            parser.error("--report must be separate from trainer and policy checkpoints")
        if args.report.exists() and not args.overwrite:
            parser.error(str(FileExistsError(args.report)))
    source_sha = sha256_file(args.trainer)
    try:
        payload = derive_payload(
            jt.load(str(args.trainer)),
            weights=args.weights,
            action_slice_mode=args.action_slice_mode,
            source_sha256=source_sha,
        )
        write_checkpoint(args.out, payload, overwrite=args.overwrite)
    except (FileExistsError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    report = {
        "source_trainer_sha256": source_sha,
        "derived_policy_sha256": sha256_file(args.out),
        "target_mode": payload["config"]["target_mode"],
        "weights": args.weights,
        "action_slice_mode": args.action_slice_mode,
        "path": str(args.out),
    }
    if args.report is not None:
        try:
            write_report(args.report, report, overwrite=args.overwrite)
        except FileExistsError as error:
            parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
