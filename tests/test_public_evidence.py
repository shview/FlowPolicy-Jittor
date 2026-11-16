import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.verify_public_evidence import (
    EvidenceError,
    _rollout_invariants,
    _verify_target_comparison,
    scan_public_tree,
    verify_fpj037_diagnostics,
    verify_manifest,
    verify_repository,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repository_public_evidence_passes_independent_verifier():
    report = verify_repository(ROOT)
    assert report["status"] == "PASS"
    assert report["aggregates"]["fpj033"]["episodes"] == 60
    assert report["aggregates"]["fpj035"]["episodes"] == 120
    assert report["aggregates"]["fpj036"]["episodes"] == 40
    assert report["aggregates"]["fpj037"]["episodes"] == 120
    assert report["aggregates"]["fpj037_diagnostics"]["episodes"] == 60


def test_repository_public_evidence_passes_lf_only_checkout(tmp_path):
    shutil.copytree(ROOT / "evidence", tmp_path / "evidence")
    for path in (tmp_path / "evidence").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".md"}:
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    report = verify_repository(tmp_path)
    assert report["status"] == "PASS"
    compatibility = [
        item
        for manifest in report["manifests"]
        for item in manifest["line_ending_compatibility"]
    ]
    assert compatibility


def test_manifest_verifier_detects_byte_mutation(tmp_path):
    payload = tmp_path / "payload.json"
    payload.write_text('{"value": 1}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    "payload.json": {
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                        "size_bytes": payload.stat().st_size,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert verify_manifest(manifest)["files_checked"] == 1
    payload.write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(EvidenceError, match="SHA256 mismatch"):
        verify_manifest(manifest)


def test_manifest_verifier_accepts_immutable_legacy_crlf_hash_on_lf_checkout(tmp_path):
    payload = tmp_path / "payload.json"
    payload.write_bytes(b'{"value": 1}\n')
    declared = b'{"value": 1}\r\n'
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    "payload.json": {
                        "sha256": hashlib.sha256(declared).hexdigest(),
                        "size_bytes": len(declared),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    report = verify_manifest(manifest)
    assert report["line_ending_compatibility"] == [
        {"file": "payload.json", "mode": "lf-to-legacy-crlf"}
    ]


def test_rollout_verifier_detects_selective_success_count():
    record = {
        "episode_count": 2,
        "success_episodes": 2,
        "total_steps": 4,
        "episodes": [
            {"success": True, "steps": 2, "done_reason": "step_cap"},
            {"success": False, "steps": 2, "done_reason": "step_cap"},
        ],
    }
    with pytest.raises(EvidenceError, match="success_episodes"):
        _rollout_invariants(record, "fixture")


def test_rollout_verifier_rejects_empty_or_malformed_episode_lists():
    with pytest.raises(EvidenceError, match="must not be empty"):
        _rollout_invariants(
            {"episode_count": 0, "success_episodes": 0, "total_steps": 0, "episodes": []},
            "fixture",
        )
    with pytest.raises(EvidenceError, match="must be an object"):
        _rollout_invariants(
            {
                "episode_count": 1,
                "success_episodes": 0,
                "total_steps": 0,
                "episodes": [None],
            },
            "fixture",
        )


def test_rollout_verifier_rejects_non_boolean_success():
    with pytest.raises(EvidenceError, match="success must be boolean"):
        _rollout_invariants(
            {
                "episode_count": 1,
                "success_episodes": 0,
                "total_steps": 2,
                "episodes": [
                    {"success": "false", "steps": 2, "done_reason": "step_cap"}
                ],
            },
            "fixture",
        )


def test_public_tree_rejects_private_cloud_path(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "bad.json").write_text(
        json.dumps({"path": "/mnt/workspace/private/run"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvidenceError, match="cloud absolute path"):
        scan_public_tree(tmp_path)


def test_public_tree_rejects_unknown_binary_artifact(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "weights.bin").write_bytes(b"not a public evidence document")
    with pytest.raises(EvidenceError, match="unexpected public evidence artifact"):
        scan_public_tree(tmp_path)


def test_repository_requires_fpj037_manifest(tmp_path):
    shutil.copytree(ROOT / "evidence", tmp_path / "evidence")
    (tmp_path / "evidence/fpj037/manifest.json").unlink()
    with pytest.raises(EvidenceError, match="cannot read JSON"):
        verify_repository(tmp_path)


def test_fpj037_comparison_requires_complete_two_mode_grid(tmp_path):
    shutil.copytree(ROOT / "evidence", tmp_path / "evidence")
    path = tmp_path / "evidence/fpj037/comparison.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["cells"] = [
        cell for cell in report["cells"] if cell["target_mode"] == "code"
    ]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(EvidenceError, match="expected 6 cells"):
        _verify_target_comparison(tmp_path, "fpj037")


def test_fpj037_diagnostics_require_exact_contract_and_source_link(tmp_path):
    shutil.copytree(ROOT / "evidence", tmp_path / "evidence")
    path = tmp_path / "evidence/fpj037/diagnostics.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["entries"][0]["source_trainer_sha256"] = "0" * 64
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(EvidenceError, match="not linked to the main cell"):
        verify_fpj037_diagnostics(tmp_path)

    report["entries"] = [
        {
            **report["entries"][0],
            "label": "invented",
            "target_mode": "invented",
            "seed": 99,
        }
    ]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(EvidenceError, match="exactly three entries"):
        verify_fpj037_diagnostics(tmp_path)
