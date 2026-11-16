import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_target_extension_evidence import build_bundle, merge_reports  # noqa: E402
from verify_public_evidence import (  # noqa: E402
    EvidenceError,
    _verify_target_comparison,
    verify_fpj037_diagnostics,
    verify_manifest,
)


def _base():
    return json.loads((ROOT / "evidence/fpj036/comparison.json").read_text(encoding="utf-8"))


def _extension(base):
    report = copy.deepcopy(base)
    report["cells"] = []
    for seed, successes in ((1, {"code": 8, "detached": 4}), (2, {"code": 12, "detached": 7})):
        for source in base["cells"]:
            cell = copy.deepcopy(source)
            cell["seed"] = seed
            count = successes[cell["target_mode"]]
            for index, episode in enumerate(cell["episode_outcomes"]):
                episode["success"] = index < count
                episode["success_steps"] = 1 if index < count else 0
            cell["success_episodes"] = count
            cell["success_rate"] = count / cell["episode_count"]
            cell["nfe"] = 1
            cell["num_points"] = 1024
            cell["action_slice_mode"] = "code"
            report["cells"].append(cell)
    report["preflight"]["shared_config"]["seeds"] = [1, 2]
    report["preflight"]["shared_config_sha256"] = "extension-contract"
    report["matched_validation"]["pairs"] = [
        copy.deepcopy(base["matched_validation"]["pairs"][0]) for _ in range(2)
    ]
    for pair, seed in zip(report["matched_validation"]["pairs"], (1, 2)):
        pair["seed"] = seed
    report["config"]["seeds"] = [1, 2]
    return report


def test_merge_recomputes_three_seed_aggregates_from_episodes():
    base = _base()
    result = merge_reports(base, _extension(base))
    code = result["targets"]["code"]["tasks"]["reach-wall-v3"]
    detached = result["targets"]["detached"]["tasks"]["reach-wall-v3"]
    assert code["seeds"] == [0, 1, 2]
    assert code["success_rate_per_seed"] == [0.5, 0.4, 0.6]
    assert code["success_rate_mean"] == pytest.approx(0.5)
    assert detached["success_rate_per_seed"] == [0.3, 0.2, 0.35]
    assert detached["success_rate_mean"] == pytest.approx(0.2833333333333333)
    assert len(result["matched_validation"]["pairs"]) == 3


def test_merge_rejects_contract_change_beyond_seed_list():
    base = _base()
    extension = _extension(base)
    extension["preflight"]["shared_config"]["batch_size"] = 32
    with pytest.raises(ValueError, match="contracts differ"):
        merge_reports(base, extension)


def test_merge_rejects_duplicate_seed_cells():
    base = _base()
    extension = _extension(base)
    for cell in extension["cells"]:
        if cell["seed"] == 1:
            cell["seed"] = 0
    with pytest.raises(ValueError, match="duplicate"):
        merge_reports(base, extension)


def test_merge_rejects_missing_matched_validation_pairs():
    base = _base()
    extension = _extension(base)
    extension["matched_validation"]["pairs"] = []
    with pytest.raises(ValueError, match="matched-validation grid"):
        merge_reports(base, extension)


def test_merge_rejects_non_boolean_episode_success():
    base = _base()
    extension = _extension(base)
    extension["cells"][0]["episode_outcomes"][0]["success"] = "false"
    with pytest.raises(ValueError, match="success must be boolean"):
        merge_reports(base, extension)


def test_bundle_writes_sanitized_manifest_without_overwriting_base(tmp_path):
    base = _base()
    extension_path = tmp_path / "experiments.json"
    extension_path.write_text(json.dumps(_extension(base)), encoding="utf-8")
    out = tmp_path / "evidence/fpj037"
    manifest = build_bundle(
        ROOT / "evidence/fpj036/comparison.json",
        extension_path,
        out,
    )
    assert set(manifest["files"]) == {"comparison.json", "SUMMARY.md"}
    comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["config"]["data_dir"] == "$FLOWPOLICY_DATA_DIR"
    assert comparison["targets"]["code"]["tasks"]["reach-wall-v3"][
        "success_rate_mean"
    ] == pytest.approx(0.5)
    assert verify_manifest(out / "manifest.json")["files_checked"] == 2
    assert _verify_target_comparison(tmp_path, "fpj037")["episodes"] == 120
    assert (ROOT / "evidence/fpj036/comparison.json").is_file()


def test_bundle_failure_leaves_no_partial_output(tmp_path):
    base = _base()
    extension_path = tmp_path / "experiments.json"
    extension_path.write_text(json.dumps(_extension(base)), encoding="utf-8")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"private_path": "/mnt/workspace/private/run"}),
        encoding="utf-8",
    )
    out = tmp_path / "evidence/fpj037"
    with pytest.raises(ValueError):
        build_bundle(
            ROOT / "evidence/fpj036/comparison.json",
            extension_path,
            out,
            metadata_path=metadata_path,
        )
    assert not out.exists()
    assert list(out.parent.glob(".fpj037.*.tmp")) == []


def test_bundle_adds_verified_evaluation_diagnostics(tmp_path):
    base = _base()
    extension_path = tmp_path / "experiments.json"
    extension_path.write_text(json.dumps(_extension(base)), encoding="utf-8")
    derivation = {
        "source_trainer_sha256": "a" * 64,
        "derived_policy_sha256": "b" * 64,
        "target_mode": "code",
        "weights": "ema",
        "action_slice_mode": "code",
        "path": "/private/checkpoint.ckpt",
    }
    cell = base["cells"][0]
    rollout = {
        "task": "reach-wall-v3",
        "checkpoint": "/private/checkpoint.ckpt",
        "checkpoint_sha256": "b" * 64,
        "model_config": {"target_mode": "code"},
        "action_slice_mode": "code",
        "policy_action_bounds": [1, 5],
        "episodes": cell["episode_outcomes"],
        "episode_count": cell["episode_count"],
        "success_episodes": cell["success_episodes"],
        "total_steps": cell["total_steps"],
        "checkpoint_repeat_max_abs_delta": 1e-6,
        "checkpoint_same_object_reload_max_abs_delta": 2e-6,
        "checkpoint_fresh_policy_reload_max_abs_delta": 3e-6,
        "checkpoint_reload_max_abs_delta": 3e-6,
    }
    (tmp_path / "derive.json").write_text(json.dumps(derivation), encoding="utf-8")
    (tmp_path / "rollout.json").write_text(json.dumps(rollout), encoding="utf-8")
    spec = {
        "entries": [
            {
                "label": "code-ema-code",
                "seed": 1,
                "derivation": "derive.json",
                "rollout": "rollout.json",
            }
        ]
    }
    spec_path = tmp_path / "diagnostics-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = tmp_path / "evidence/fpj037"
    manifest = build_bundle(
        ROOT / "evidence/fpj036/comparison.json",
        extension_path,
        out,
        diagnostics_spec_path=spec_path,
    )
    assert set(manifest["files"]) == {
        "comparison.json",
        "diagnostics.json",
        "SUMMARY.md",
    }
    diagnostics = json.loads((out / "diagnostics.json").read_text(encoding="utf-8"))
    entry = diagnostics["entries"][0]
    assert entry["success_episodes"] == 10
    assert entry["episode_count"] == 20
    assert entry["weights"] == "ema"
    assert "/private/" not in json.dumps(diagnostics)
    assert verify_manifest(out / "manifest.json")["files_checked"] == 3
    with pytest.raises(EvidenceError, match="exactly three entries"):
        verify_fpj037_diagnostics(tmp_path)
