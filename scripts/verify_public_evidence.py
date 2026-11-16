"""Independently verify the repository's compact public evidence bundles.

This verifier deliberately reads only files that are safe to publish.  Raw Zarr
stores, checkpoints and private cloud logs are neither required nor permitted.
It checks byte manifests first, then recomputes episode-level aggregates rather
than trusting the summary tables that were generated with the experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = (
    Path("evidence/fpj033/preliminary/manifest.json"),
    Path("evidence/fpj033/formal/manifest.json"),
    Path("evidence/fpj035/manifest.json"),
    Path("evidence/fpj036/manifest.json"),
    Path("evidence/fpj037/manifest.json"),
)
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".tar",
    ".tgz",
    ".zip",
}
PUBLIC_EVIDENCE_SUFFIXES = {".json", ".md"}
TARGET_GRID = {
    "fpj035": {"code": (0, 1, 2), "detached": (0, 1, 2)},
    "fpj036": {"code": (0,), "detached": (0,)},
    "fpj037": {"code": (0, 1, 2), "detached": (0, 1, 2)},
}
FPJ037_DIAGNOSTIC_CONTRACT = {
    "code-seed1-ema-code": {
        "target_mode": "code",
        "weights": "ema",
        "action_slice_mode": "code",
        "seed": 1,
    },
    "code-seed1-online-exact": {
        "target_mode": "code",
        "weights": "online",
        "action_slice_mode": "exact",
        "seed": 1,
    },
    "detached-seed1-ema-code": {
        "target_mode": "detached",
        "weights": "ema",
        "action_slice_mode": "code",
        "seed": 1,
    },
}
PRIVATE_PATTERNS = {
    "Windows absolute path": re.compile(r"(?<![A-Za-z])\b[A-Za-z]:[\\/](?!/)"),
    "cloud absolute path": re.compile(r"/(?:mnt/(?:workspace|systemDisk)|home/[^/\s]+)/"),
    "DSW instance identifier": re.compile(r"\bdsw-[a-z0-9]+\b", re.IGNORECASE),
    "workspace query identifier": re.compile(r"\bworkspaceId\b", re.IGNORECASE),
    "Alibaba account nickname": re.compile(r"\bnick\d{6,}\b", re.IGNORECASE),
    "AWS-style access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub personal token": re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
}


class EvidenceError(RuntimeError):
    """Raised when a public evidence invariant does not hold."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_payload(
    target: Path, expected_sha256: str
) -> tuple[bytes, str]:
    """Return bytes matching a declared hash.

    FPJ-033 through FPJ-036 were generated from CRLF files on Windows while
    Git stored normalized LF blobs.  The manifests are immutable, so a clean
    Linux checkout must recognize only that historical line-ending transform.
    New bundles are written as LF and should match raw bytes directly.
    """

    payload = target.read_bytes()
    if _sha256_bytes(payload) == expected_sha256:
        return payload, "raw"
    if target.suffix.lower() not in {".json", ".md"}:
        raise EvidenceError(
            f"SHA256 mismatch for {target}: expected {expected_sha256}, "
            f"got {_sha256_bytes(payload)}"
        )
    variants: list[tuple[bytes, str]] = []
    if b"\r\n" in payload:
        variants.append((payload.replace(b"\r\n", b"\n"), "crlf-to-lf"))
    else:
        variants.append((payload.replace(b"\n", b"\r\n"), "lf-to-legacy-crlf"))
    for candidate, normalization in variants:
        if _sha256_bytes(candidate) == expected_sha256:
            return candidate, normalization
    raise EvidenceError(
        f"SHA256 mismatch for {target}: expected {expected_sha256}, "
        f"got {_sha256_bytes(payload)}"
    )


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{path} must contain a JSON object")
    return value


def _close(actual: Any, expected: Any, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise EvidenceError(f"{label}: expected {expected!r}, got {actual!r}")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise EvidenceError(f"{label}: expected {expected!r}, got {actual!r}")


def _safe_relative(name: str, *, manifest: Path) -> Path:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise EvidenceError(f"{manifest}: unsafe manifest path {name!r}")
    return Path(*pure.parts)


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest = _load(path)
    if "files_sha256" in manifest:
        raw_entries = manifest["files_sha256"]
        if not isinstance(raw_entries, Mapping):
            raise EvidenceError(f"{path}: files_sha256 must be an object")
        entries = {
            str(name): {"sha256": digest}
            for name, digest in raw_entries.items()
        }
    elif "files" in manifest:
        entries = manifest["files"]
    else:
        raise EvidenceError(f"{path}: missing files_sha256/files map")
    if not isinstance(entries, Mapping) or not entries:
        raise EvidenceError(f"{path}: manifest file map must be non-empty")

    checked = []
    line_ending_compatibility = []
    for name, record in entries.items():
        if not isinstance(record, Mapping) or not isinstance(record.get("sha256"), str):
            raise EvidenceError(f"{path}: invalid hash record for {name!r}")
        relative = _safe_relative(str(name), manifest=path)
        target = path.parent / relative
        if not target.is_file():
            raise EvidenceError(f"{path}: missing declared file {name!r}")
        expected = record["sha256"]
        try:
            declared_payload, normalization = _manifest_payload(target, expected)
        except EvidenceError as exc:
            raise EvidenceError(f"{path}: {exc}") from exc
        if "size_bytes" in record and len(declared_payload) != int(record["size_bytes"]):
            raise EvidenceError(
                f"{path}: size mismatch for {name!r}: expected {record['size_bytes']}, "
                f"got {len(declared_payload)} after declared-byte normalization"
            )
        if normalization != "raw":
            line_ending_compatibility.append(
                {"file": str(relative).replace("\\", "/"), "mode": normalization}
            )
        checked.append(str(relative).replace("\\", "/"))
    return {
        "manifest": str(path),
        "files_checked": len(checked),
        "files": checked,
        "line_ending_compatibility": line_ending_compatibility,
    }


def _rollout_invariants(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    episodes = record.get("episode_outcomes", record.get("episodes"))
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        raise EvidenceError(f"{label}: missing episode outcomes")
    if not episodes:
        raise EvidenceError(f"{label}: episode outcomes must not be empty")
    if any(not isinstance(episode, Mapping) for episode in episodes):
        raise EvidenceError(f"{label}: every episode outcome must be an object")
    invalid_success = [
        index
        for index, episode in enumerate(episodes)
        if not isinstance(episode.get("success"), bool)
    ]
    if invalid_success:
        raise EvidenceError(
            f"{label}: episode success must be boolean at indexes {invalid_success}"
        )
    if int(record.get("episode_count", -1)) != len(episodes):
        raise EvidenceError(f"{label}: episode_count does not match episode list")
    successes = sum(bool(episode.get("success")) for episode in episodes)
    if int(record.get("success_episodes", -1)) != successes:
        raise EvidenceError(f"{label}: success_episodes does not match episode list")
    total_steps = sum(int(episode.get("steps", -1)) for episode in episodes)
    if any(int(episode.get("steps", -1)) < 0 for episode in episodes):
        raise EvidenceError(f"{label}: episode steps must be non-negative")
    if int(record.get("total_steps", -1)) != total_steps:
        raise EvidenceError(f"{label}: total_steps does not match episode list")
    done_reasons = Counter(str(episode.get("done_reason")) for episode in episodes)
    declared_reasons = record.get("done_reasons")
    if declared_reasons is not None and dict(done_reasons) != declared_reasons:
        raise EvidenceError(f"{label}: done_reasons does not match episode list")
    rate = successes / len(episodes) if episodes else 0.0
    if "success_rate" in record:
        _close(record["success_rate"], rate, f"{label}.success_rate")
    return {
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": rate,
        "total_steps": total_steps,
        "done_reasons": dict(done_reasons),
    }


def _sample_std(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _verify_summary(
    cells: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    ok = [cell for cell in cells if cell.get("status") == "ok"]
    if not ok:
        raise EvidenceError(f"{label}: no successful cells to aggregate")
    ordered = sorted(ok, key=lambda cell: int(cell["seed"]))
    rates = [float(cell["success_rate"]) for cell in ordered]
    seeds = [int(cell["seed"]) for cell in ordered]
    if list(summary.get("seeds", [])) != seeds:
        raise EvidenceError(f"{label}: seed list does not match cells")
    if list(summary.get("success_rate_per_seed", [])) != rates:
        raise EvidenceError(f"{label}: per-seed success rates do not match cells")
    _close(summary.get("success_rate_mean"), statistics.mean(rates), f"{label}.mean")
    _close(summary.get("success_rate_std"), _sample_std(rates), f"{label}.std")
    episodes_per_seed = sorted({int(cell["episode_count"]) for cell in ordered})
    if list(summary.get("episodes_per_seed", [])) != episodes_per_seed:
        raise EvidenceError(f"{label}: episodes_per_seed does not match cells")
    reasons = Counter()
    for cell in ordered:
        reasons.update(cell.get("done_reasons", {}))
    if dict(reasons) != summary.get("done_reasons"):
        raise EvidenceError(f"{label}: aggregate done_reasons do not match cells")
    return {
        "seeds": seeds,
        "episodes": sum(int(cell["episode_count"]) for cell in ordered),
        "successes": sum(int(cell["success_episodes"]) for cell in ordered),
        "rates": rates,
        "mean": statistics.mean(rates),
        "sample_std": _sample_std(rates),
    }


def verify_fpj033(root: Path) -> dict[str, Any]:
    base = root / "evidence/fpj033/formal"
    report = _load(base / "experiments.json")
    cells = report.get("cells")
    if not isinstance(cells, list) or len(cells) != 3:
        raise EvidenceError("FPJ-033 formal report must contain three cells")
    outcomes: dict[int, dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise EvidenceError("FPJ-033 formal cells must be objects")
        seed = int(cell["seed"])
        rollout = _load(base / f"seed-{seed}-rollout.json")
        result = _rollout_invariants(rollout, f"FPJ-033 seed {seed}")
        for key in ("episode_count", "success_episodes", "total_steps"):
            if int(cell[key]) != int(result[{"episode_count": "episodes", "success_episodes": "successes", "total_steps": "total_steps"}[key]]):
                raise EvidenceError(f"FPJ-033 seed {seed}: cell {key} does not match rollout")
        _close(cell["success_rate"], result["success_rate"], f"FPJ-033 seed {seed}")
        outcomes[seed] = result
    task = "reach-wall-v3"
    aggregate = _verify_summary(cells, report["tasks"][task], label="FPJ-033 formal")
    aggregate["episode_files"] = outcomes
    return aggregate


def _verify_matched_pairs(
    report: Mapping[str, Any], task_id: str, expected_seeds: Sequence[int]
) -> None:
    validation = report.get("matched_validation")
    if not isinstance(validation, Mapping) or validation.get("passed") is not True:
        raise EvidenceError(f"{task_id}: matched validation did not pass")
    pairs = validation.get("pairs")
    if not isinstance(pairs, list):
        raise EvidenceError(f"{task_id}: matched validation pairs must be a list")
    expected = {("reach-wall-v3", int(seed)) for seed in expected_seeds}
    actual: set[tuple[str, int]] = set()
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise EvidenceError(f"{task_id}: every matched validation pair must be an object")
        try:
            key = (str(pair.get("task")), int(pair.get("seed")))
        except (TypeError, ValueError) as exc:
            raise EvidenceError(f"{task_id}: matched validation seed is invalid") from exc
        if key in actual:
            raise EvidenceError(f"{task_id}: duplicate matched validation pair {key}")
        actual.add(key)
        if pair.get("base_mode") != "code":
            raise EvidenceError(f"{task_id}: matched pair {key} must use code as base")
        checks = pair.get("checks")
        comparison = checks.get("code_vs_detached") if isinstance(checks, Mapping) else None
        if not isinstance(comparison, Mapping) or comparison.get("passed") is not True:
            raise EvidenceError(f"{task_id}: matched pair {key} did not pass")
    if actual != expected:
        raise EvidenceError(
            f"{task_id}: matched validation grid must be {sorted(expected)}, got {sorted(actual)}"
        )


def _verify_target_comparison(root: Path, task_id: str) -> dict[str, Any]:
    if task_id not in TARGET_GRID:
        raise EvidenceError(f"unsupported target comparison bundle: {task_id}")
    report = _load(root / f"evidence/{task_id}/comparison.json")
    cells = report.get("cells")
    if not isinstance(cells, list):
        raise EvidenceError(f"{task_id}: cells must be a list")
    expected_grid = TARGET_GRID[task_id]
    expected_keys = {
        (mode, "reach-wall-v3", seed)
        for mode, seeds in expected_grid.items()
        for seed in seeds
    }
    if len(cells) != len(expected_keys):
        raise EvidenceError(
            f"{task_id}: expected {len(expected_keys)} cells, got {len(cells)}"
        )
    if report.get("target_modes") != ["code", "detached"]:
        raise EvidenceError(f"{task_id}: target_modes must be ['code', 'detached']")
    by_mode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    episodes = 0
    actual_keys: set[tuple[str, str, int]] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise EvidenceError(f"{task_id}: every cell must be an object")
        if cell.get("status") != "ok":
            raise EvidenceError(f"{task_id}: every cell must have status 'ok'")
        try:
            key = (str(cell.get("target_mode")), str(cell.get("task")), int(cell.get("seed")))
        except (TypeError, ValueError) as exc:
            raise EvidenceError(f"{task_id}: cell mode/task/seed is invalid") from exc
        if key in actual_keys:
            raise EvidenceError(f"{task_id}: duplicate cell {key}")
        actual_keys.add(key)
        result = _rollout_invariants(cell, f"{task_id} {cell.get('target_mode')} seed {cell.get('seed')}")
        if result["episodes"] != 20:
            raise EvidenceError(f"{task_id}: every cell must contain 20 episodes")
        episodes += result["episodes"]
        by_mode[str(cell["target_mode"])].append(cell)
    if actual_keys != expected_keys:
        raise EvidenceError(
            f"{task_id}: incomplete target/task/seed grid; "
            f"expected {sorted(expected_keys)}, got {sorted(actual_keys)}"
        )
    _verify_matched_pairs(
        report,
        task_id,
        sorted({seed for seeds in expected_grid.values() for seed in seeds}),
    )
    targets = report.get("targets")
    if not isinstance(targets, Mapping) or set(targets) != set(expected_grid):
        raise EvidenceError(f"{task_id}: targets summary must contain exactly code and detached")
    checked: dict[str, Any] = {"cells": len(cells), "episodes": episodes, "modes": {}}
    for mode, mode_cells in sorted(by_mode.items()):
        tasks = {str(cell["task"]) for cell in mode_cells}
        if tasks != {"reach-wall-v3"}:
            raise EvidenceError(f"{task_id}/{mode}: unexpected tasks {sorted(tasks)}")
        summary = report["targets"][mode]["tasks"]["reach-wall-v3"]
        checked["modes"][mode] = _verify_summary(
            mode_cells, summary, label=f"{task_id}/{mode}"
        )
    return checked


def verify_fpj034(root: Path) -> dict[str, Any]:
    report = _load(root / "evidence/fpj034/resource_probe.json")
    results = report["results"]
    online_counts = {int(value["online_trainable_parameters"]) for value in results.values()}
    if online_counts != {102676}:
        raise EvidenceError(f"FPJ-034: unexpected online parameter counts {online_counts}")
    detached = results["detached_ema"]
    if detached.get("target_trainable_parameters") != 0:
        raise EvidenceError("FPJ-034: detached target must have zero trainable parameters")
    if detached.get("ema_optimization_step") != 8:
        raise EvidenceError("FPJ-034: detached EMA step must be eight")
    expected_delta = (
        float(detached["construction_rss_delta_mib"])
        - float(results["code_no_ema"]["construction_rss_delta_mib"])
    )
    _close(
        report["detached_minus_code_no_ema"]["construction_rss_delta_mib"],
        expected_delta,
        "FPJ-034 construction RSS delta",
    )
    return {
        "online_trainable_parameters": 102676,
        "detached_target_trainable_parameters": 0,
        "ema_optimization_step": 8,
        "construction_rss_delta_mib": expected_delta,
    }


def verify_fpj037_diagnostics(root: Path) -> dict[str, Any]:
    report = _load(root / "evidence/fpj037/diagnostics.json")
    entries = report.get("entries")
    if not isinstance(entries, list) or len(entries) != len(FPJ037_DIAGNOSTIC_CONTRACT):
        raise EvidenceError("FPJ-037 diagnostics must contain exactly three entries")
    labels = [str(entry.get("label")) for entry in entries if isinstance(entry, Mapping)]
    if len(labels) != len(entries) or len(labels) != len(set(labels)):
        raise EvidenceError("FPJ-037 diagnostic labels must be unique objects")
    if set(labels) != set(FPJ037_DIAGNOSTIC_CONTRACT):
        raise EvidenceError(
            "FPJ-037 diagnostic labels do not match the declared variant contract"
        )
    comparison = _load(root / "evidence/fpj037/comparison.json")
    trainer_hashes = {
        (str(cell.get("target_mode")), int(cell.get("seed"))): str(
            cell.get("train", {}).get("trainer_checkpoint_sha256")
        )
        for cell in comparison.get("cells", [])
        if isinstance(cell, Mapping)
    }
    episodes = 0
    successes = 0
    variants = []
    for entry in entries:
        label = str(entry["label"])
        contract = FPJ037_DIAGNOSTIC_CONTRACT[label]
        for key, expected in contract.items():
            if entry.get(key) != expected:
                raise EvidenceError(
                    f"FPJ-037 diagnostic {label}: {key} must be {expected!r}"
                )
        if entry.get("task") != "reach-wall-v3":
            raise EvidenceError(f"FPJ-037 diagnostic {label}: unexpected task")
        source_sha = entry.get("source_trainer_sha256")
        derived_sha = entry.get("derived_policy_sha256")
        if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
            raise EvidenceError(f"FPJ-037 diagnostic {label}: invalid source trainer SHA256")
        if not isinstance(derived_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", derived_sha):
            raise EvidenceError(f"FPJ-037 diagnostic {label}: invalid derived policy SHA256")
        expected_source = trainer_hashes.get((str(entry["target_mode"]), int(entry["seed"])))
        if source_sha != expected_source:
            raise EvidenceError(
                f"FPJ-037 diagnostic {label}: source trainer is not linked to the main cell"
            )
        reloads = entry.get("checkpoint_reload")
        if not isinstance(reloads, Mapping) or not reloads:
            raise EvidenceError(f"FPJ-037 diagnostic {label}: missing checkpoint reload data")
        try:
            reload_values = [float(value) for value in reloads.values()]
        except (TypeError, ValueError) as exc:
            raise EvidenceError(
                f"FPJ-037 diagnostic {label}: invalid checkpoint reload value"
            ) from exc
        if not all(math.isfinite(value) and value >= 0.0 for value in reload_values):
            raise EvidenceError(
                f"FPJ-037 diagnostic {label}: checkpoint reload values must be finite and non-negative"
            )
        result = _rollout_invariants(entry, f"FPJ-037 diagnostic {entry['label']}")
        if result["episodes"] != 20:
            raise EvidenceError(f"FPJ-037 diagnostic {label}: expected 20 episodes")
        episodes += result["episodes"]
        successes += result["successes"]
        if entry.get("weights") not in {"online", "ema"}:
            raise EvidenceError(f"FPJ-037 diagnostic {entry['label']}: invalid weights")
        if entry.get("action_slice_mode") not in {"code", "exact"}:
            raise EvidenceError(
                f"FPJ-037 diagnostic {entry['label']}: invalid action slice mode"
            )
        variants.append(
            {
                "label": entry["label"],
                "target_mode": entry.get("target_mode"),
                "weights": entry.get("weights"),
                "action_slice_mode": entry.get("action_slice_mode"),
                "seed": entry.get("seed"),
                **result,
            }
        )
    return {
        "entries": len(entries),
        "episodes": episodes,
        "successes": successes,
        "variants": variants,
    }


def scan_public_tree(root: Path) -> dict[str, Any]:
    evidence = root / "evidence"
    checked = 0
    total_bytes = 0
    for path in sorted(p for p in evidence.rglob("*") if p.is_file()):
        relative = path.relative_to(root)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(
            part.lower().endswith(".zarr") for part in relative.parts
        ):
            raise EvidenceError(f"forbidden public evidence artifact: {relative}")
        if path.suffix.lower() not in PUBLIC_EVIDENCE_SUFFIXES:
            raise EvidenceError(f"unexpected public evidence artifact: {relative}")
        payload = path.read_bytes()
        total_bytes += len(payload)
        checked += 1
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvidenceError(f"invalid UTF-8 text in {relative}: {exc}") from exc
        for label, pattern in PRIVATE_PATTERNS.items():
            match = pattern.search(text)
            if match:
                raise EvidenceError(
                    f"{label} found in {relative}: {match.group(0)!r}"
                )
    return {"files_checked": checked, "bytes_checked": total_bytes}


def verify_repository(root: Path = REPO_ROOT) -> dict[str, Any]:
    root = root.resolve()
    manifests = [verify_manifest(root / relative) for relative in MANIFESTS]
    aggregates = {
        "fpj033": verify_fpj033(root),
        "fpj034": verify_fpj034(root),
        "fpj035": _verify_target_comparison(root, "fpj035"),
        "fpj036": _verify_target_comparison(root, "fpj036"),
    }
    aggregates["fpj037"] = _verify_target_comparison(root, "fpj037")
    aggregates["fpj037_diagnostics"] = verify_fpj037_diagnostics(root)
    safety = scan_public_tree(root)
    return {
        "status": "PASS",
        "root": str(root),
        "manifests": manifests,
        "aggregates": aggregates,
        "safety": safety,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    args = parser.parse_args(argv)
    try:
        report = verify_repository(args.root)
    except EvidenceError as exc:
        print(f"PUBLIC EVIDENCE: FAIL\n- {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        manifests = sum(item["files_checked"] for item in report["manifests"])
        aggregates = report["aggregates"]
        episode_counts = [
            f"{task_id.upper()}={aggregate['episodes']}"
            for task_id, aggregate in aggregates.items()
            if isinstance(aggregate, Mapping) and "episodes" in aggregate
        ]
        print("PUBLIC EVIDENCE: PASS")
        print(f"- manifest payloads: {manifests}")
        print(f"- episode outcomes: {', '.join(episode_counts)}")
        print(f"- safety files: {report['safety']['files_checked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
