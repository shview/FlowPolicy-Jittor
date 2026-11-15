"""Parity harness plumbing tests (FPJ-028).

The numerical comparison itself needs PyTorch, which lives in a separate virtualenv; it
runs through ``scripts/parity_check.py`` and is recorded in ``outputs/parity/``.  These
tests cover what can be checked in the Jittor environment: the bundle is complete and
consistent, and the harness refuses to report a pass it did not measure.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

jt = pytest.importorskip("jittor")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import parity_check  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = previous


def test_bundle_contains_every_weight_gradient_and_output(tmp_path):
    bundle_path = tmp_path / "bundle.npz"
    summary = parity_check.dump_bundle(bundle_path)
    bundle = np.load(bundle_path, allow_pickle=False)
    config = json.loads(str(bundle["config_json"]))

    assert summary["weight_count"] > 100
    assert summary["gradient_count"] > 100
    assert summary["action_pred_shape"] == [2, 4, 4]
    assert np.isfinite(summary["loss"]) and summary["loss"] > 0

    # Every gradient must belong to a real weight, and the ``var_est`` branch must be
    # excluded (it is frozen, exactly like PyTorch leaves it at grad=None).
    weight_names = set(config["weight_names"])
    gradient_names = set(config["gradient_names"])
    assert gradient_names <= weight_names
    assert not any(name.startswith("model.var_est") for name in gradient_names)
    assert any(name.startswith("model.down_modules") for name in gradient_names)
    assert any(name.startswith("obs_encoder.extractor") for name in gradient_names)

    # Inputs round-trip unchanged, so a future comparison cannot be blamed on data.
    expected = parity_check._inputs()
    for key, value in expected.items():
        assert np.array_equal(bundle[key], value), key

    for name in weight_names:
        assert "w:" + name in bundle, name
    for name in gradient_names:
        assert "g:" + name in bundle, name
        assert np.asarray(bundle["g:" + name]).shape == np.asarray(bundle["w:" + name]).shape


def test_verdict_fails_loudly_on_a_disagreeing_gradient():
    report = {
        "action_pred": {"max_abs": 1e-7, "mean_abs": 1e-8, "max_rel": 1e-6, "reference_scale": 1.0},
        "loss": {"max_abs": 1e-9, "mean_abs": 1e-9, "max_rel": 1e-9, "reference_scale": 1.0},
        "gradients": {
            "model.final_conv.1.weight": {
                "max_abs": 1e-9,
                "mean_abs": 1e-10,
                "max_rel": 1e-9,
                "reference_scale": 1e-3,
            },
            "model.mid_modules.0.blocks.0.block.0.weight": {
                "max_abs": 1e-2,
                "mean_abs": 1e-3,
                "max_rel": 5e-2,
                "reference_scale": 1e-3,
            },
        },
    }
    verdict = parity_check._verdict(report, absolute_tolerance=1e-7, relative_tolerance=1e-4)
    assert verdict["passed"] is False
    assert verdict["failed"] == ["grad:model.mid_modules.0.blocks.0.block.0.weight"]
    assert verdict["checks"]["action_pred"]["passed"] is True


def test_small_reference_tensors_are_not_absolved_by_the_absolute_floor():
    """A 1.2% relative change of a 1.6e-4 loss must fail; 2e-4 atol used to hide it."""
    report = {
        "action_pred": {"max_abs": 9.5e-7, "mean_abs": 1e-7, "max_rel": 4e-6, "reference_scale": 5.6},
        "loss": {"max_abs": 2.0e-6, "mean_abs": 2.0e-6, "max_rel": 1.2e-2, "reference_scale": 1.6e-4},
        "gradients": {},
    }
    verdict = parity_check._verdict(report, absolute_tolerance=1e-7, relative_tolerance=1e-4)
    assert verdict["checks"]["loss"]["passed"] is False
    assert verdict["failed"] == ["loss"]
    # The measured Float32 agreement of the real run stays comfortably inside.
    report["loss"].update({"max_abs": 7.3e-10, "max_rel": 4.5e-6})
    assert parity_check._verdict(report, 1e-7, 1e-4)["passed"] is True


def test_missing_torch_interpreter_is_reported_not_assumed(tmp_path):
    out = tmp_path / "parity"
    code = parity_check.main(
        ["--out", str(out), "--torch-python", str(tmp_path / "no-such-python")]
    )
    assert code == 1
    result = json.loads((out / "parity.json").read_text())
    assert result["parity"] == "not run"
    assert "no torch interpreter" in result["torch"]["reason"]
    # The Jittor half still has to be real work, not a stub.
    assert result["jittor"]["weight_count"] > 100


def test_refuses_to_overwrite_an_existing_bundle(tmp_path):
    out = tmp_path / "parity"
    out.mkdir()
    (out / "keep.txt").write_text("existing evidence")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        parity_check.main(["--out", str(out), "--torch-python", str(tmp_path / "missing")])
