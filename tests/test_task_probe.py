"""Tests for the pre-collection MetaWorld expert-quality gate (FPJ-033)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import task_probe  # noqa: E402


def _record(index, *, first=100, success_steps=10, error=None):
    return {
        "variant_index": index,
        "first_success_step": first,
        "success_steps": success_steps,
        "error": error,
    }


def test_policy_name_matches_v3_convention_and_special_case():
    assert task_probe.expert_policy_name("reach-wall-v3") == "SawyerReachWallV3Policy"
    assert task_probe.expert_policy_name("button-press-wall-v3") == "SawyerButtonPressWallV3Policy"
    assert task_probe.expert_policy_name("peg-insert-side-v3") == "SawyerPegInsertionSideV3Policy"


def test_summary_requires_explicit_rate_and_success_step_thresholds():
    records = [_record(index) for index in range(9)] + [
        _record(9, first=None, success_steps=0)
    ]
    strict = task_probe.summarize_records(
        records, min_success_steps=5, min_success_rate=1.0
    )
    assert strict["variants_reached_success"] == 9
    assert strict["variants_qualified"] == 9
    assert strict["qualified_rate"] == pytest.approx(0.9)
    assert strict["suitable_for_collection"] is False

    permissive = task_probe.summarize_records(
        records, min_success_steps=5, min_success_rate=0.9
    )
    assert permissive["suitable_for_collection"] is True


def test_error_disqualifies_even_when_rate_threshold_is_met():
    records = [_record(index) for index in range(10)]
    records[-1]["error"] = "AssertionError: bad observation"
    summary = task_probe.summarize_records(
        records, min_success_steps=5, min_success_rate=0.9
    )
    assert summary["errors"] == 1
    assert summary["suitable_for_collection"] is False


def test_empty_records_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        task_probe.summarize_records([], min_success_steps=5, min_success_rate=1.0)


def test_report_writer_refuses_implicit_overwrite(tmp_path):
    path = tmp_path / "probe.json"
    task_probe.write_report(path, {"value": 1}, overwrite=False)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        task_probe.write_report(path, {"value": 2}, overwrite=False)
    task_probe.write_report(path, {"value": 3}, overwrite=True)
    assert '"value": 3' in path.read_text(encoding="utf-8")
