"""Build a public, path-sanitized evidence bundle from local experiment JSON files."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PRIVATE_PATTERNS = (
    re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    re.compile(r"/home/[^/]+/"),
    re.compile(r"/mnt/(?:workspace|[a-z])/", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
)


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _basename(value: Any) -> Any:
    if value is None:
        return None
    return Path(str(value).replace("\\", "/")).name


def sanitize_train_summary(value: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(value)
    for key in ("data", "checkpoint", "trainer_checkpoint"):
        if key in result:
            result[key] = _basename(result[key])
    ema = result.get("ema")
    if isinstance(ema, dict) and "trainer_checkpoint" in ema:
        ema["trainer_checkpoint"] = _basename(ema["trainer_checkpoint"])
    resume = result.get("resume")
    if isinstance(resume, dict):
        resume["resumed_from"] = _basename(resume.get("resumed_from"))
        for entry in resume.get("checkpoints_written_this_invocation", []):
            if isinstance(entry, dict):
                entry["path"] = _basename(entry.get("path"))
    result["evidence_status"] = "local low-budget resume smoke; not a formal success rate"
    return result


def sanitize_rollout(value: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(value)
    result["data"] = _basename(result.get("data"))
    result["checkpoint"] = _basename(result.get("checkpoint"))
    result["sim_python"] = "$FLOWPOLICY_SIM_PYTHON"
    result["evidence_status"] = (
        "five-episode local smoke; failures are retained; not a formal success-rate estimate"
    )
    return result


def sanitize_command(command: Any) -> Any:
    """Replace cloud interpreter and checkout paths with portable public names."""

    if not isinstance(command, list):
        return command
    result = []
    for raw in command:
        value = str(raw).replace("\\", "/")
        if value.endswith("/venvs/flowpolicy/bin/python"):
            value = "$FLOWPOLICY_TRAIN_PYTHON"
        elif value.endswith("/venvs/flowpolicy-sim/bin/python"):
            value = "$FLOWPOLICY_SIM_PYTHON"
        elif "/scripts/" in value:
            value = "scripts/" + value.rsplit("/scripts/", 1)[1]
        result.append(value)
    return result


def sanitize_experiments(value: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(value)
    for cell in result.get("cells", []):
        if not isinstance(cell, dict):
            continue
        for phase in ("train", "rollout"):
            details = cell.get(phase)
            if isinstance(details, dict):
                details["command"] = sanitize_command(details.get("command"))
    result["evidence_status"] = (
        "A10 full-channel limited-budget experiment; not the paper training recipe"
    )
    return result


def sanitize_formal_train_summary(value: Dict[str, Any]) -> Dict[str, Any]:
    result = sanitize_train_summary(value)
    result["evidence_status"] = (
        "A10 full-channel 30-epoch training cell; not a paper-scale run"
    )
    return result


def sanitize_formal_rollout(value: Dict[str, Any]) -> Dict[str, Any]:
    result = sanitize_rollout(value)
    result["evidence_status"] = (
        "twenty-episode A10 closed-loop cell in the documented substitute simulator"
    )
    return result


def assert_public_safe(value: Any) -> None:
    text = json.dumps(value, sort_keys=True)
    matches = [pattern.pattern for pattern in PRIVATE_PATTERNS if pattern.search(text)]
    if matches:
        raise ValueError("evidence contains a private path or secret-like value: " + ", ".join(matches))


def _write(path: Path, value: Any) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_formal_summary(experiments: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    task = experiments["tasks"]["reach-wall-v3"]
    rates = [100.0 * float(value) for value in task["success_rate_per_seed"]]
    mean = 100.0 * float(task["success_rate_mean"])
    std = 100.0 * float(task["success_rate_std"])
    cells = experiments["cells"]
    rows = []
    for cell in cells:
        rows.append(
            "| {seed} | {success}/{episodes} ({rate:.0f}%) | {loss:.6g} | "
            "{val:.6g} | {latency:.3f} | {delta:.3g} |".format(
                seed=cell["seed"],
                success=cell["success_episodes"],
                episodes=cell["episode_count"],
                rate=100.0 * float(cell["success_rate"]),
                loss=cell["train"]["loss"]["last_epoch_mean"],
                val=cell["train"]["loss"]["val"],
                latency=cell["latency_ms_mean"],
                delta=cell["checkpoint_reload_max_abs_delta"],
            )
        )
    return "\n".join(
        [
            "# FPJ-033 formal A10 evidence",
            "",
            "`reach-wall-v3` used the full 512/1024/2048 channel configuration, "
            "30 epochs, batch 64, seeds 0/1/2 and 20 rollout episodes per seed.",
            "",
            f"- success rate: **{mean:.2f}% ± {std:.2f}%** "
            f"({', '.join(f'{rate:.0f}%' for rate in rates)})",
            f"- mean policy latency: **{float(task['latency_ms_mean']):.3f} ms**, NFE=1",
            f"- hardware: {metadata['hardware']['gpu']}, "
            f"{metadata['hardware']['gpu_memory_mib']} MiB, "
            f"driver {metadata['hardware']['driver']}",
            f"- sampled peak GPU memory: "
            f"{metadata['measurements']['sampled_peak_gpu_memory_mib']} MiB",
            "- all 60 episodes ended at the declared 200-step cap; success is an "
            "episode-level observation and does not end the rollout early",
            "",
            "| seed | success | last train loss | val loss | latency ms | reload max abs delta |",
            "| --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            "## Interpretation boundary",
            "",
            "This is a full-channel, limited-budget Jittor run in the documented modern "
            "MetaWorld/MuJoCo substitute stack. It is not the paper's 3000-epoch, 37-task "
            "recipe, does not use the paper-style detached EMA target, and must not be "
            "compared numerically with the paper success-rate table.",
            "",
        ]
    )


def build_formal_bundle(
    *,
    source_dir: Path,
    dataset_fingerprint_path: Path,
    preflight_path: Path,
    metadata_path: Path,
    out_dir: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty evidence directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_experiments = _load(source_dir / "experiments.json")
    experiments = sanitize_experiments(raw_experiments)
    metadata = _load(metadata_path)
    payloads: Dict[str, Any] = {
        "experiments.json": experiments,
        "dataset_fingerprint.json": _load(dataset_fingerprint_path),
        "preflight.json": _load(preflight_path),
        "cloud_metadata.json": metadata,
    }
    for seed in (0, 1, 2):
        payloads[f"seed-{seed}-train-summary.json"] = sanitize_formal_train_summary(
            _load(source_dir / f"seed-{seed}-train-summary.json")
        )
        payloads[f"seed-{seed}-rollout.json"] = sanitize_formal_rollout(
            _load(source_dir / f"seed-{seed}-rollout.json")
        )
        payloads[f"seed-{seed}-epochs.json"] = json.loads(
            (source_dir / f"seed-{seed}-epochs.json").read_text(encoding="utf-8")
        )

    for value in payloads.values():
        assert_public_safe(value)
    hashes = {name: _write(out_dir / name, value) for name, value in payloads.items()}
    summary = render_formal_summary(experiments, metadata)
    assert_public_safe({"summary": summary})
    (out_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")
    hashes["SUMMARY.md"] = _sha256(out_dir / "SUMMARY.md")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "task": "reach-wall-v3",
        "git_commit_at_generation": commit,
        "git_commit_used_for_run": metadata["git_commit_used_for_run"],
        "bundle_status": "formal A10 full-channel limited-budget experiment complete",
        "source_files_sha256": {
            path.name: _sha256(path)
            for path in sorted(source_dir.iterdir())
            if path.is_file()
        },
        "files_sha256": hashes,
        "excluded": [
            "raw Zarr demonstrations",
            "policy/trainer checkpoints",
            "private absolute paths",
            "cloud credentials and secrets",
        ],
        "claim_boundary": (
            "Three seeds and sixty episodes are complete for a full-channel, 30-epoch "
            "A10 run in the substitute simulator. This is not the paper's 3000-epoch "
            "recipe or its reported success rate."
        ),
    }
    assert_public_safe(manifest)
    _write(out_dir / "manifest.json", manifest)
    return manifest


def build_bundle(
    *,
    probe_path: Path,
    fingerprint_path: Path,
    train_path: Path,
    rollout_path: Path,
    fresh_clone_path: Optional[Path] = None,
    out_dir: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty evidence directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "expert_probe.json": _load(probe_path),
        "dataset_fingerprint.json": _load(fingerprint_path),
        "local_train_summary.json": sanitize_train_summary(_load(train_path)),
        "local_rollout.json": sanitize_rollout(_load(rollout_path)),
    }
    if fresh_clone_path is not None:
        payloads["fresh_clone.json"] = _load(fresh_clone_path)
    for value in payloads.values():
        assert_public_safe(value)
    hashes = {name: _write(out_dir / name, value) for name, value in payloads.items()}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "task": "reach-wall-v3",
        "git_commit_at_generation": commit,
        "bundle_status": "preliminary local gates complete; full-channel cloud run pending",
        "files_sha256": hashes,
        "excluded": ["raw Zarr demonstrations", "policy/trainer checkpoints", "private paths", "secrets"],
        "claim_boundary": (
            "This bundle proves expert/data determinism, resumable local training and a "
            "five-episode closed-loop smoke. It is not the full-channel three-seed result "
            "and not the paper's training recipe."
        ),
    }
    assert_public_safe(manifest)
    _write(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--fingerprint", type=Path)
    parser.add_argument("--train-summary", type=Path)
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--fresh-clone", type=Path, default=None)
    parser.add_argument("--formal-source-dir", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.formal_source_dir is not None:
            if args.fingerprint is None or args.preflight is None or args.metadata is None:
                parser.error("formal mode requires --fingerprint, --preflight and --metadata")
            manifest = build_formal_bundle(
                source_dir=args.formal_source_dir,
                dataset_fingerprint_path=args.fingerprint,
                preflight_path=args.preflight,
                metadata_path=args.metadata,
                out_dir=args.out,
                overwrite=args.overwrite,
            )
        else:
            missing = [
                name
                for name, value in (
                    ("--probe", args.probe),
                    ("--fingerprint", args.fingerprint),
                    ("--train-summary", args.train_summary),
                    ("--rollout", args.rollout),
                )
                if value is None
            ]
            if missing:
                parser.error("preliminary mode requires " + ", ".join(missing))
            manifest = build_bundle(
                probe_path=args.probe,
                fingerprint_path=args.fingerprint,
                train_path=args.train_summary,
                rollout_path=args.rollout,
                fresh_clone_path=args.fresh_clone,
                out_dir=args.out,
                overwrite=args.overwrite,
            )
    except FileExistsError as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
