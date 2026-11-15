"""Tests for the FPJ-023 32-window overfit smoke run.

The learning check runs the real script in a subprocess so it gets the OpenMP
thread cap that the CPU path needs; the in-process tests only cover the window
selection and the log contract.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

jt = pytest.importorskip("jittor")

from flowpolicy_jittor.data.dataset import MetaworldZarrDataset, select_windows  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "overfit_32_windows.py"
PYTHON = sys.executable


def _load_script_module():
    spec = importlib.util.spec_from_file_location("fpj023_overfit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script_module()


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fpj023")
    path = directory / "synthetic.zarr"
    script.write_synthetic_store(path, episodes=3, frames=20, num_points=16, seed=0)
    return path


@pytest.fixture(scope="module")
def dataset(store):
    return MetaworldZarrDataset(
        store, horizon=4, n_obs_steps=2, n_action_steps=4, num_points=16, val_ratio=0.0
    )


def test_select_windows_is_exact_and_reproducible(dataset):
    first = select_windows(dataset, 32, 7)
    second = select_windows(dataset, 32, 7)
    assert first.shape == (32,)
    assert len(set(first.tolist())) == 32
    assert np.array_equal(first, second)
    assert first.min() >= 0 and first.max() < dataset.num_windows
    # A different seed must produce a different subset, not the same one relabelled.
    assert not np.array_equal(first, select_windows(dataset, 32, 8))


def test_select_windows_rejects_impossible_requests(dataset):
    with pytest.raises(ValueError, match="only has"):
        select_windows(dataset, dataset.num_windows + 1, 0)
    with pytest.raises(ValueError):
        select_windows(dataset, 0, 0)


def test_run_overfit_writes_the_documented_log(store, tmp_path):
    log = tmp_path / "nested" / "overfit.json"
    record = script.run_overfit(
        zarr_path=store,
        steps=4,
        lr=3e-3,
        seed=0,
        num_points=16,
        log_path=log,
        enforce_reduction=False,
    )

    written = json.loads(log.read_text(encoding="utf-8"))
    assert written == json.loads(json.dumps(record))
    assert written["task"] == "FPJ-023"
    assert written["windows"] == 32
    assert len(written["window_indices"]) == 32
    assert written["label"].startswith("32-window smoke overfit")
    assert "paper-scale training" in written["label"]
    assert set(written["loss"]) >= {
        "initial",
        "min",
        "final",
        "reduction_ratio",
        "history",
    }
    assert set(written["action_error"]) >= {
        "initial",
        "final",
        "reduction_ratio",
        "baselines",
    }
    assert set(written["action_error"]["baselines"]) == {
        "zero_predictor_mae",
        "mean_action_predictor_mae",
    }
    assert np.isfinite(written["loss"]["initial"])
    assert np.isfinite(written["action_error"]["initial"])
    assert written["action_error"]["reduction_enforced"] is False
    assert written["loss"]["history"][0]["step"] == 1
    assert written["loss"]["history"][-1]["step"] == 4
    assert "paper success rate reproduced" in written["prohibited_claims"]


def test_overfit_run_rejects_a_non_learning_configuration(store):
    """One step cannot memorise 32 windows, so the guard must fire."""
    with pytest.raises(RuntimeError, match="did not halve the action error"):
        script.run_overfit(
            zarr_path=store,
            steps=1,
            lr=3e-3,
            seed=0,
            num_points=16,
            enforce_reduction=True,
        )


def test_script_reduces_action_error_by_half(tmp_path):
    log = tmp_path / "overfit.json"
    environment = dict(os.environ, OMP_NUM_THREADS="4")
    completed = subprocess.run(
        [
            PYTHON,
            str(SCRIPT),
            "--steps",
            "200",
            "--num-points",
            "32",
            "--log",
            str(log),
        ],
        cwd=str(REPO_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["windows"] == 32
    assert record["steps"] == 200
    assert record["action_error"]["reduction_ratio"] >= 0.5
    assert record["action_error"]["final"] < record["action_error"]["initial"]
    # Beating the untrained policy is not enough: the trained model must also
    # beat the best constant predictor on the same windows.
    baselines = record["action_error"]["baselines"]
    assert record["action_error"]["final"] < min(baselines.values())
    assert record["loss"]["min"] <= record["loss"]["initial"]
    assert np.isfinite([record["loss"]["initial"], record["loss"]["min"], record["loss"]["final"]]).all()
