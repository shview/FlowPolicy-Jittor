"""Merge a verified one-seed target bundle with new matched seeds.

FPJ-037 deliberately keeps the already-published FPJ-036 seed-0 bundle
immutable.  This builder accepts that public comparison plus a raw
``experiments.json`` containing only the added seeds, verifies that the
experiment contracts match, recomputes all aggregate statistics, and writes a
new path-sanitized bundle.  It never copies checkpoints, Zarr stores, commands,
or absolute cloud paths into the public output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from build_evidence_bundle import assert_public_safe
from build_target_evidence import sanitize_report
from experiments import _aggregate


DEFAULT_STATUS = (
    "full-channel NVIDIA A10 matched target-semantics comparison; "
    "three training seeds"
)
DEFAULT_BOUNDARY = [
    "This is a selected-task, 30-epoch finite-budget comparison, not the paper's 3000-epoch 37-task result.",
    "The modern MetaWorld/MuJoCo stack is a documented substitute for the paper's older simulator stack.",
    "Success rates are recomputed from every declared closed-loop episode; training loss is not used as a proxy for task success.",
    "The comparison isolates released-code target semantics versus a detached EMA loss target; it does not prove either semantic is universally superior.",
]
DIAGNOSTIC_LABEL = re.compile(r"[a-z0-9][a-z0-9-]*\Z")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Any) -> dict[str, Any]:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _write_text(path: Path, value: str) -> dict[str, Any]:
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _without_seeds(shared: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(shared))
    result.pop("seeds", None)
    return result


def _validate_source(report: Mapping[str, Any], label: str) -> None:
    if not report.get("preflight", {}).get("passed"):
        raise ValueError(f"{label}: preflight did not pass")
    if not report.get("matched_validation", {}).get("passed"):
        raise ValueError(f"{label}: matched validation did not pass")
    modes = list(report.get("target_modes", []))
    if modes != ["code", "detached"]:
        raise ValueError(f"{label}: expected target modes ['code', 'detached'], got {modes}")
    cells = report.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{label}: cells must be a non-empty list")
    if any(not isinstance(cell, Mapping) for cell in cells):
        raise ValueError(f"{label}: every cell must be an object")
    failed = [cell for cell in cells if cell.get("status") != "ok"]
    if failed:
        raise ValueError(f"{label}: every cell must finish successfully")


def _validate_matched_pairs(report: Mapping[str, Any], label: str) -> None:
    shared = report.get("preflight", {}).get("shared_config", {})
    tasks = shared.get("tasks") if isinstance(shared, Mapping) else None
    cells = report.get("cells")
    if not isinstance(tasks, list) or not tasks or not isinstance(cells, list):
        raise ValueError(f"{label}: cannot derive matched-validation grid")
    try:
        expected = {
            (str(task), int(cell["seed"]))
            for task in tasks
            for cell in cells
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid task/seed in cells") from exc
    validation = report.get("matched_validation")
    pairs = validation.get("pairs") if isinstance(validation, Mapping) else None
    if not isinstance(pairs, list):
        raise ValueError(f"{label}: matched-validation pairs must be a list")
    actual: set[tuple[str, int]] = set()
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError(f"{label}: every matched-validation pair must be an object")
        try:
            key = (str(pair.get("task")), int(pair.get("seed")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: invalid matched-validation pair") from exc
        if key in actual:
            raise ValueError(f"{label}: duplicate matched-validation pair {key}")
        actual.add(key)
        checks = pair.get("checks")
        comparison = checks.get("code_vs_detached") if isinstance(checks, Mapping) else None
        if pair.get("base_mode") != "code" or not isinstance(comparison, Mapping):
            raise ValueError(f"{label}: malformed matched-validation pair {key}")
        if comparison.get("passed") is not True:
            raise ValueError(f"{label}: matched-validation pair {key} did not pass")
    if actual != expected:
        raise ValueError(
            f"{label}: matched-validation grid must be {sorted(expected)}, got {sorted(actual)}"
        )


def _episode_check(cell: Mapping[str, Any], label: str) -> None:
    outcomes = cell.get("episode_outcomes")
    if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)):
        raise ValueError(f"{label}: missing episode outcomes")
    if not outcomes:
        raise ValueError(f"{label}: episode outcomes must not be empty")
    if any(not isinstance(item, Mapping) for item in outcomes):
        raise ValueError(f"{label}: every episode outcome must be an object")
    invalid_success = [
        index
        for index, item in enumerate(outcomes)
        if not isinstance(item.get("success"), bool)
    ]
    if invalid_success:
        raise ValueError(
            f"{label}: episode success must be boolean at indexes {invalid_success}"
        )
    successes = sum(bool(item.get("success")) for item in outcomes)
    steps = sum(int(item.get("steps", -1)) for item in outcomes)
    if any(int(item.get("steps", -1)) < 0 for item in outcomes):
        raise ValueError(f"{label}: episode steps must be non-negative")
    reasons = Counter(str(item.get("done_reason")) for item in outcomes)
    if len(outcomes) != int(cell.get("episode_count", -1)):
        raise ValueError(f"{label}: episode_count does not match outcomes")
    if successes != int(cell.get("success_episodes", -1)):
        raise ValueError(f"{label}: success_episodes does not match outcomes")
    if steps != int(cell.get("total_steps", -1)):
        raise ValueError(f"{label}: total_steps does not match outcomes")
    if dict(reasons) != cell.get("done_reasons"):
        raise ValueError(f"{label}: done_reasons does not match outcomes")
    rate = successes / len(outcomes)
    if not math.isclose(
        rate,
        float(cell.get("success_rate", -1)),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label}: success_rate does not match outcomes")


def _diagnostic_entry(
    *,
    label: str,
    seed: int,
    derivation: Mapping[str, Any],
    rollout: Mapping[str, Any],
) -> dict[str, Any]:
    if not DIAGNOSTIC_LABEL.fullmatch(label):
        raise ValueError(f"invalid diagnostic label: {label!r}")
    target_mode = str(derivation.get("target_mode"))
    weights = str(derivation.get("weights"))
    action_slice_mode = str(derivation.get("action_slice_mode"))
    if target_mode not in {"code", "detached"}:
        raise ValueError(f"{label}: invalid target mode")
    if weights not in {"online", "ema"}:
        raise ValueError(f"{label}: invalid weight selection")
    if action_slice_mode not in {"code", "exact"}:
        raise ValueError(f"{label}: invalid action slice mode")
    model_config = rollout.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError(f"{label}: rollout model_config is missing")
    if str(model_config.get("target_mode")) != target_mode:
        raise ValueError(f"{label}: target mode differs between checkpoint and rollout")
    if str(rollout.get("action_slice_mode")) != action_slice_mode:
        raise ValueError(f"{label}: action slice mode differs between checkpoint and rollout")
    expected_bounds = [1, 5] if action_slice_mode == "code" else [0, 4]
    if list(rollout.get("policy_action_bounds", [])) != expected_bounds:
        raise ValueError(f"{label}: policy action bounds do not match the slice mode")
    derived_sha = str(derivation.get("derived_policy_sha256"))
    source_sha = str(derivation.get("source_trainer_sha256"))
    if not re.fullmatch(r"[0-9a-f]{64}", derived_sha):
        raise ValueError(f"{label}: invalid derived policy SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ValueError(f"{label}: invalid source trainer SHA256")
    if str(rollout.get("checkpoint_sha256")) != derived_sha:
        raise ValueError(f"{label}: rollout did not use the declared derived checkpoint")

    raw_episodes = rollout.get("episodes")
    if not isinstance(raw_episodes, Sequence) or isinstance(raw_episodes, (str, bytes)):
        raise ValueError(f"{label}: rollout episodes are missing")
    episode_outcomes = []
    for raw in raw_episodes:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label}: every rollout episode must be an object")
        if not isinstance(raw.get("success"), bool):
            raise ValueError(f"{label}: every rollout success value must be boolean")
        episode_outcomes.append(
            {
                key: copy.deepcopy(raw[key])
                for key in (
                    "episode",
                    "steps",
                    "success_steps",
                    "success",
                    "done_reason",
                    "policy_latency_ms",
                )
                if key in raw
            }
        )
    successes = sum(bool(item.get("success")) for item in episode_outcomes)
    total_steps = sum(int(item.get("steps", -1)) for item in episode_outcomes)
    if any(int(item.get("steps", -1)) < 0 for item in episode_outcomes):
        raise ValueError(f"{label}: episode steps must be non-negative")
    if len(episode_outcomes) != int(rollout.get("episode_count", -1)):
        raise ValueError(f"{label}: episode_count does not match episodes")
    if len(episode_outcomes) != 20:
        raise ValueError(f"{label}: diagnostics require 20 closed-loop episodes")
    if successes != int(rollout.get("success_episodes", -1)):
        raise ValueError(f"{label}: success_episodes does not match episodes")
    if total_steps != int(rollout.get("total_steps", -1)):
        raise ValueError(f"{label}: total_steps does not match episodes")
    done_reasons = dict(Counter(str(item.get("done_reason")) for item in episode_outcomes))
    latencies = [
        float(item["policy_latency_ms"]["mean"])
        for item in episode_outcomes
        if isinstance(item.get("policy_latency_ms"), Mapping)
        and item["policy_latency_ms"].get("mean") is not None
    ]
    if len(latencies) != len(episode_outcomes):
        raise ValueError(f"{label}: every episode must declare mean policy latency")
    checkpoint_reload = {
        key: rollout.get(key)
        for key in (
            "checkpoint_repeat_max_abs_delta",
            "checkpoint_same_object_reload_max_abs_delta",
            "checkpoint_fresh_policy_reload_max_abs_delta",
            "checkpoint_reload_max_abs_delta",
        )
    }
    if any(value is None or not math.isfinite(float(value)) for value in checkpoint_reload.values()):
        raise ValueError(f"{label}: checkpoint reload diagnostics must be finite")
    declared_seed = rollout.get("seed")
    if declared_seed is not None and int(declared_seed) != seed:
        raise ValueError(f"{label}: rollout seed differs from diagnostic specification")
    task = str(rollout.get("task"))
    if task != "reach-wall-v3":
        raise ValueError(f"{label}: unexpected diagnostic task {task!r}")
    return {
        "label": label,
        "task": task,
        "seed": seed,
        "noise_seed": int(rollout.get("noise_seed", seed)),
        "target_mode": target_mode,
        "weights": weights,
        "action_slice_mode": action_slice_mode,
        "source_trainer_sha256": source_sha,
        "derived_policy_sha256": derived_sha,
        "episode_count": len(episode_outcomes),
        "success_episodes": successes,
        "success_rate": successes / len(episode_outcomes),
        "total_steps": total_steps,
        "done_reasons": done_reasons,
        "latency_ms_mean": statistics.mean(latencies),
        "checkpoint_reload": checkpoint_reload,
        "episode_outcomes": episode_outcomes,
    }


def load_diagnostics(
    spec_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    spec = _load(spec_path)
    raw_entries = spec.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("diagnostics specification must contain non-empty entries")
    entries = []
    source_hashes: dict[str, str] = {}
    seen = set()
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            raise ValueError("every diagnostic specification entry must be an object")
        label = str(raw.get("label"))
        if label in seen:
            raise ValueError(f"duplicate diagnostic label: {label}")
        seen.add(label)
        derivation_path = (spec_path.parent / str(raw.get("derivation"))).resolve()
        rollout_path = (spec_path.parent / str(raw.get("rollout"))).resolve()
        if not derivation_path.is_file() or not rollout_path.is_file():
            raise FileNotFoundError(f"{label}: diagnostic input is missing")
        entries.append(
            _diagnostic_entry(
                label=label,
                seed=int(raw.get("seed")),
                derivation=_load(derivation_path),
                rollout=_load(rollout_path),
            )
        )
        source_hashes[f"{label}.derivation.json"] = _sha256(derivation_path)
        source_hashes[f"{label}.rollout.json"] = _sha256(rollout_path)
    result = {
        "schema_version": 1,
        "evidence_status": "evaluation-only checkpoint diagnostics; no retraining",
        "entries": entries,
        "claim_boundary": [
            "These rollouts isolate weight selection or action slicing on selected seed-1 trainer checkpoints.",
            "They are diagnostics, not additional independent training seeds and are excluded from the three-seed target aggregate.",
        ],
    }
    assert_public_safe(result)
    return result, source_hashes


def merge_reports(
    base_report: Mapping[str, Any], extension_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Return one independently aggregated public report."""

    _validate_source(base_report, "base")
    _validate_source(extension_report, "extension")
    base_shared = base_report["preflight"]["shared_config"]
    extension_shared = extension_report["preflight"]["shared_config"]
    if _without_seeds(base_shared) != _without_seeds(extension_shared):
        raise ValueError("base and extension experiment contracts differ beyond seeds")

    base_cells = copy.deepcopy(list(base_report["cells"]))
    extension_cells = copy.deepcopy(list(extension_report["cells"]))
    cells = base_cells + extension_cells
    for cell in cells:
        for key in ("nfe", "num_points", "action_slice_mode"):
            if cell.get(key) is None:
                cell[key] = copy.deepcopy(base_shared[key])
        if cell.get("checkpoint_reload_max_abs_delta") is None:
            reload_record = cell.get("checkpoint_reload")
            if isinstance(reload_record, Mapping):
                cell["checkpoint_reload_max_abs_delta"] = reload_record.get(
                    "checkpoint_reload_max_abs_delta"
                )
    keys = [
        (str(cell.get("target_mode")), str(cell.get("task")), int(cell.get("seed")))
        for cell in cells
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("base and extension contain duplicate target/task/seed cells")
    _validate_matched_pairs(base_report, "base")
    _validate_matched_pairs(extension_report, "extension")

    base_seeds = {int(cell["seed"]) for cell in base_cells}
    extension_seeds = {int(cell["seed"]) for cell in extension_cells}
    if base_seeds & extension_seeds:
        raise ValueError("base and extension seed sets must be disjoint")
    all_seeds = sorted(base_seeds | extension_seeds)
    if all_seeds != [0, 1, 2]:
        raise ValueError(f"expected the completed seed set [0, 1, 2], got {all_seeds}")

    expected = {
        (mode, task, seed)
        for mode in ("code", "detached")
        for task in base_shared["tasks"]
        for seed in all_seeds
    }
    if set(keys) != expected:
        raise ValueError("the merged report is not a complete target/task/seed grid")
    for cell in cells:
        _episode_check(
            cell,
            f"{cell['target_mode']}/{cell['task']}/seed-{cell['seed']}",
        )

    combined = _aggregate(
        sorted(cells, key=lambda cell: (cell["target_mode"], cell["task"], cell["seed"]))
    )
    combined["schema_version"] = 1
    combined["evidence_status"] = DEFAULT_STATUS
    combined["claim_boundary"] = list(DEFAULT_BOUNDARY)
    combined["prohibited_claims"] = sorted(
        set(base_report.get("prohibited_claims", []))
        | set(extension_report.get("prohibited_claims", []))
        | {"paper success rate reproduced", "the best cell is the result"}
    )

    shared = copy.deepcopy(dict(base_shared))
    shared["seeds"] = all_seeds
    combined["preflight"] = {
        "passed": True,
        "shared_config": shared,
        "shared_config_sha256": _canonical_sha(shared),
        "arms": copy.deepcopy(base_report["preflight"].get("arms", {})),
        "intended_differences": copy.deepcopy(
            base_report["preflight"].get("intended_differences", [])
        ),
        "source_contract_sha256": {
            "base": base_report["preflight"].get("shared_config_sha256"),
            "extension": extension_report["preflight"].get("shared_config_sha256"),
        },
    }
    pairs = copy.deepcopy(base_report["matched_validation"].get("pairs", []))
    pairs += copy.deepcopy(extension_report["matched_validation"].get("pairs", []))
    combined["matched_validation"] = {
        "applicable": True,
        "passed": True,
        "pairs": sorted(pairs, key=lambda pair: (pair["task"], pair["seed"])),
    }

    config = copy.deepcopy(dict(extension_report.get("config", {})))
    config["seeds"] = all_seeds
    config["data_dir"] = "$FLOWPOLICY_DATA_DIR"
    config["run_label"] = "fpj037-a10-full-channels-paired-target-seeds0-2"
    combined["config"] = config
    combined["checkpoint_hashes"] = {
        f"{cell['target_mode']}/{cell['task']}/seed-{cell['seed']}": {
            key: cell.get("train", {}).get(key)
            for key in ("checkpoint_sha256", "trainer_checkpoint_sha256")
            if cell.get("train", {}).get(key)
        }
        for cell in combined["cells"]
    }
    assert_public_safe(combined)
    return combined


def _summary(
    report: Mapping[str, Any], diagnostics: Mapping[str, Any] | None = None
) -> str:
    task = report["config"]["tasks"][0]
    lines = [
        "# FPJ-037 A10 three-seed target comparison",
        "",
        "- Task: `reach-wall-v3`; seeds 0/1/2; full 512/1024/2048 channels",
        "- Budget: 30 epochs, batch 64, 20 closed-loop episodes per seed, 200-step cap",
        "- Compared variable: released-code target semantics vs detached EMA loss target",
        "",
        "| target | per-seed successes | mean ± sample std | mean policy latency |",
        "|---|---:|---:|---:|",
    ]
    for mode in ("code", "detached"):
        entry = report["targets"][mode]["tasks"][task]
        successes = [
            f"{cell['success_episodes']}/{cell['episode_count']}"
            for cell in report["cells"]
            if cell["target_mode"] == mode and cell["task"] == task
        ]
        lines.append(
            f"| {mode} | {', '.join(successes)} | "
            f"{entry['success_rate_mean']:.2%} ± {entry['success_rate_std']:.2%} | "
            f"{entry['latency_ms_mean']:.3f} ms |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        *[f"- {item}" for item in report["claim_boundary"]],
    ]
    if diagnostics is not None:
        lines += [
            "",
            "## Evaluation-only diagnostics",
            "",
            "| variant | target | weights | slice | seed | successes | mean latency |",
            "|---|---|---|---|---:|---:|---:|",
        ]
        for entry in diagnostics["entries"]:
            lines.append(
                f"| {entry['label']} | {entry['target_mode']} | {entry['weights']} | "
                f"{entry['action_slice_mode']} | {entry['seed']} | "
                f"{entry['success_episodes']}/{entry['episode_count']} | "
                f"{entry['latency_ms_mean']:.3f} ms |"
            )
        lines += [
            "",
            *[f"- {item}" for item in diagnostics["claim_boundary"]],
        ]
    return "\n".join(lines) + "\n"


def build_bundle(
    base_path: Path,
    extension_path: Path,
    out_dir: Path,
    *,
    metadata_path: Path | None = None,
    diagnostics_spec_path: Path | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty evidence directory: {out_dir}")
    raw_extension = _load(extension_path)
    extension = sanitize_report(raw_extension)
    report = merge_reports(_load(base_path), extension)
    diagnostics = None
    diagnostic_source_hashes: dict[str, str] = {}
    if diagnostics_spec_path is not None:
        diagnostics, diagnostic_source_hashes = load_diagnostics(
            diagnostics_spec_path.resolve()
        )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.", suffix=".tmp", dir=out_dir.parent)
    )
    try:
        files = {
            "comparison.json": _write_json(staging / "comparison.json", report),
            "SUMMARY.md": _write_text(
                staging / "SUMMARY.md", _summary(report, diagnostics)
            ),
        }
        if diagnostics is not None:
            files["diagnostics.json"] = _write_json(
                staging / "diagnostics.json", diagnostics
            )
        if metadata_path is not None:
            metadata = _load(metadata_path)
            assert_public_safe(metadata)
            files["cloud_metadata.json"] = _write_json(
                staging / "cloud_metadata.json", metadata
            )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        manifest = {
            "schema_version": 1,
            "git_commit_at_generation": commit,
            "source_files_sha256": {
                "base_comparison.json": _sha256(base_path),
                "extension_experiments.json": _sha256(extension_path),
                **diagnostic_source_hashes,
            },
            "files": files,
            "excluded": [
                "raw Zarr demonstrations",
                "policy and trainer checkpoint bytes",
                "subprocess and GPU sample logs",
                "absolute cloud paths and instance identifiers",
                "credentials and secrets",
            ],
            "claim_boundary": list(DEFAULT_BOUNDARY),
        }
        assert_public_safe(manifest)
        _write_json(staging / "manifest.json", manifest)
        if out_dir.exists():
            out_dir.rmdir()
        staging.replace(out_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cloud-metadata", type=Path)
    parser.add_argument("--diagnostics-spec", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        manifest = build_bundle(
            args.base,
            args.extension,
            args.out,
            metadata_path=args.cloud_metadata,
            diagnostics_spec_path=args.diagnostics_spec,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
