"""Public evidence sanitization tests for FPJ-033."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_evidence_bundle as bundle  # noqa: E402


def test_train_and_rollout_paths_are_replaced_with_portable_names():
    train = bundle.sanitize_train_summary(
        {
            "data": "C:\\Users\\student\\private\\demo.zarr",
            "checkpoint": "/mnt/workspace/run/policy.ckpt",
            "trainer_checkpoint": "/home/student/run/trainer.ckpt",
            "ema": {"trainer_checkpoint": "/home/student/run/trainer.ckpt"},
            "resume": {
                "resumed_from": "/home/student/run/epoch-0002.ckpt",
                "checkpoints_written_this_invocation": [
                    {"path": "/home/student/run/epoch-0004.ckpt"}
                ],
            },
        }
    )
    assert train["data"] == "demo.zarr"
    assert train["checkpoint"] == "policy.ckpt"
    assert train["resume"]["resumed_from"] == "epoch-0002.ckpt"
    bundle.assert_public_safe(train)

    rollout = bundle.sanitize_rollout(
        {
            "data": "/home/student/demo.zarr",
            "checkpoint": "/mnt/workspace/policy.ckpt",
            "sim_python": "/home/student/sim-env/bin/python",
        }
    )
    assert rollout["sim_python"] == "$FLOWPOLICY_SIM_PYTHON"
    bundle.assert_public_safe(rollout)


def test_safety_check_rejects_private_paths_and_secret_shapes():
    with pytest.raises(ValueError, match="private path"):
        bundle.assert_public_safe({"path": "/home/student/private.json"})
    with pytest.raises(ValueError, match="secret-like"):
        bundle.assert_public_safe({"token": "ghp_abcdefghijklmnopqrstuvwxyz123456"})


def test_bundle_refuses_implicit_overwrite(tmp_path):
    out = tmp_path / "evidence"
    out.mkdir()
    (out / "old.json").write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"ok": True}), encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        bundle.build_bundle(
            probe_path=source,
            fingerprint_path=source,
            train_path=source,
            rollout_path=source,
            out_dir=out,
            overwrite=False,
        )
