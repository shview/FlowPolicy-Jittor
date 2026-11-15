"""Public evidence sanitization tests for FPJ-033."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_evidence_bundle as bundle  # noqa: E402
import build_target_evidence as target_bundle  # noqa: E402


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


def test_cloud_commands_are_made_portable_and_public_safe():
    value = bundle.sanitize_experiments(
        {
            "cells": [
                {
                    "train": {
                        "command": [
                            "/mnt/workspace/venvs/flowpolicy/bin/python",
                            "/mnt/workspace/projects/repo/scripts/train_smoke.py",
                        ]
                    },
                    "rollout": {
                        "command": [
                            "/mnt/workspace/venvs/flowpolicy-sim/bin/python",
                            "/mnt/workspace/projects/repo/scripts/rollout.py",
                        ]
                    },
                }
            ]
        }
    )
    assert value["cells"][0]["train"]["command"] == [
        "$FLOWPOLICY_TRAIN_PYTHON",
        "scripts/train_smoke.py",
    ]
    assert value["cells"][0]["rollout"]["command"] == [
        "$FLOWPOLICY_SIM_PYTHON",
        "scripts/rollout.py",
    ]
    bundle.assert_public_safe(value)


def test_target_comparison_bundle_keeps_outcomes_but_drops_private_paths(tmp_path):
    report = {
        "target_modes": ["code", "detached"],
        "targets": {"code": {"tasks": {}}, "detached": {"tasks": {}}},
        "preflight": {"passed": True, "shared_config_sha256": "shared"},
        "matched_validation": {"passed": True, "pairs": []},
        "config": {"tasks": ["reach-wall-v3"], "data_dir": "/home/student/data"},
        "cells": [
            {
                "target_mode": "code",
                "task": "reach-wall-v3",
                "seed": 0,
                "status": "ok",
                "episode_count": 1,
                "success_episodes": 0,
                "success_rate": 0.0,
                "episode_outcomes": [
                    {
                        "episode": 0,
                        "success": False,
                        "steps": 200,
                        "success_steps": 0,
                        "done_reason": "step_cap",
                    }
                ],
                "train": {
                    "command": ["/home/student/python", "/home/student/train.py"],
                    "log": "/home/student/train.log",
                    "checkpoint": "/home/student/policy.ckpt",
                    "checkpoint_sha256": "policy-hash",
                    "trainer_checkpoint_sha256": "trainer-hash",
                    "config": {"target_mode": "code"},
                    "ema": {"trainer_checkpoint": "/home/student/trainer.ckpt"},
                },
                "rollout": {"command": ["/home/student/python"], "exit_code": 0},
            }
        ],
        "prohibited_claims": ["paper success rate reproduced"],
    }
    sanitized = target_bundle.sanitize_report(report)
    assert sanitized["config"]["data_dir"] == "data"
    assert sanitized["cells"][0]["episode_outcomes"][0]["steps"] == 200
    assert "command" not in sanitized["cells"][0]["train"]
    assert "trainer_checkpoint" not in sanitized["cells"][0]["train"]["ema"]
    bundle.assert_public_safe(sanitized)
