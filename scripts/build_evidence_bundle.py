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


def assert_public_safe(value: Any) -> None:
    text = json.dumps(value, sort_keys=True)
    matches = [pattern.pattern for pattern in PRIVATE_PATTERNS if pattern.search(text)]
    if matches:
        raise ValueError("evidence contains a private path or secret-like value: " + ", ".join(matches))


def _write(path: Path, value: Any) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--train-summary", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--fresh-clone", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
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
