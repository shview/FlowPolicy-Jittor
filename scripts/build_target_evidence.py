"""Build a compact public evidence bundle for the FPJ-035 target comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable

from build_evidence_bundle import assert_public_safe


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> str:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return _sha256(path)


def _sanitize_train(train: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(train, dict):
        return {}
    result = {
        key: copy.deepcopy(train[key])
        for key in (
            "reused",
            "exit_code",
            "seconds",
            "wall_seconds",
            "seconds_per_step",
            "loss",
            "train_windows",
            "val_windows",
            "steps_per_epoch",
            "total_steps",
            "trainable_parameters",
            "data_sha256",
            "config",
            "config_sha256",
            "checkpoint_sha256",
            "trainer_checkpoint_sha256",
            "runtime",
            "ema",
        )
        if key in train
    }
    ema = result.get("ema")
    if isinstance(ema, dict):
        ema.pop("trainer_checkpoint", None)
    resume = train.get("resume")
    if isinstance(resume, dict):
        result["resume"] = {
            "resumed": bool(resume.get("resumed", False)),
            "start_epoch": resume.get("start_epoch"),
            "checkpoint_every": resume.get("checkpoint_every"),
            "claim_boundary": resume.get("claim_boundary"),
        }
    return result


def sanitize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    if not report.get("preflight", {}).get("passed"):
        raise ValueError("comparison preflight did not pass")
    if not report.get("matched_validation", {}).get("passed"):
        raise ValueError("matched comparison validation did not pass")
    config = copy.deepcopy(report.get("config", {}))
    if "data_dir" in config:
        config["data_dir"] = Path(str(config["data_dir"]).replace("\\", "/")).name
    cells = []
    for raw in report.get("cells", []):
        cell = {
            key: copy.deepcopy(raw[key])
            for key in (
                "target_mode",
                "task",
                "seed",
                "status",
                "episode_count",
                "success_episodes",
                "success_rate",
                "success_steps",
                "success_step_counts",
                "episode_outcomes",
                "done_reasons",
                "latency_ms_mean",
                "total_steps",
                "wall_seconds",
                "nfe",
                "num_points",
                "action_slice_mode",
                "checkpoint_reload_max_abs_delta",
            )
            if key in raw
        }
        cell["train"] = _sanitize_train(raw.get("train", {}))
        rollout = raw.get("rollout", {})
        if isinstance(rollout, dict):
            cell["rollout_process"] = {
                key: copy.deepcopy(rollout[key])
                for key in ("reused", "exit_code", "seconds")
                if key in rollout
            }
        cells.append(cell)
    result = {
        "schema_version": 1,
        "evidence_status": (
            "matched small-channel CPU target comparison; not a full-channel or paper-scale run"
        ),
        "target_modes": copy.deepcopy(report.get("target_modes", [])),
        "targets": copy.deepcopy(report.get("targets", {})),
        "preflight": copy.deepcopy(report["preflight"]),
        "matched_validation": copy.deepcopy(report["matched_validation"]),
        "config": config,
        "cells": cells,
        "prohibited_claims": copy.deepcopy(report.get("prohibited_claims", [])),
    }
    assert_public_safe(result)
    return result


def build_bundle(source_dir: Path, out_dir: Path, *, overwrite: bool) -> Dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty evidence directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    source_report = source_dir / "experiments.json"
    source_preflight = source_dir / "preflight.json"
    source_summary = source_dir / "SUMMARY.md"
    for path in (source_report, source_preflight, source_summary):
        if not path.is_file():
            raise FileNotFoundError(f"required comparison artifact is missing: {path}")

    report = sanitize_report(_load(source_report))
    summary = source_summary.read_text(encoding="utf-8")
    assert_public_safe({"summary": summary})
    hashes = {"comparison.json": _write(out_dir / "comparison.json", report)}
    (out_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")
    hashes["SUMMARY.md"] = _sha256(out_dir / "SUMMARY.md")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "task": report["config"].get("tasks"),
        "target_modes": report["target_modes"],
        "git_commit_at_generation": commit,
        "source_files_sha256": {
            path.name: _sha256(path)
            for path in (source_report, source_preflight, source_summary)
        },
        "files_sha256": hashes,
        "excluded": [
            "raw Zarr demonstrations",
            "policy/trainer checkpoints",
            "subprocess logs and absolute paths",
            "credentials and secrets",
        ],
        "claim_boundary": (
            "This bundle compares target semantics with matched small-channel CPU settings. "
            "It does not establish full-channel GPU behavior or reproduce paper success rates."
        ),
    }
    assert_public_safe(manifest)
    _write(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        manifest = build_bundle(args.source, args.out, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
