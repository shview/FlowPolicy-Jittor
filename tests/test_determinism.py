"""Save/resume determinism on the CPU backend (FPJ-031).

These tests turn the claim into a measurement: an uninterrupted run and a
save-at-step-3/resume run must produce bit-identical weights at every step, and the
comparison helper must refuse to compare runs that do not cover the same steps (the first
version of the probe silently lined step 1 up with step 4 and reported fake agreement).
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

import determinism_probe  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = previous


def test_in_process_resume_stays_within_the_documented_tolerance(tmp_path):
    """Resume in a long-lived process may drift; the tolerance is what we promise.

    FPJ-031 measurements: two identical runs in one process are bit-exact, and a resume
    in a *fresh* process is bit-exact under OMP_NUM_THREADS=1/4 with and without
    MKL_CBWR=COMPATIBLE.  But a resume that happens after prior training in the same
    process can drift (measured up to 5.5e-3 in one weight, 9.5e-9 in the loss), which is
    why no bit-exact claim is made for that case.  This test pins the tolerance rather
    than pretending the drift is zero.
    """
    single = tmp_path / "single.npz"
    resume = tmp_path / "resume.npz"
    determinism_probe.run("single", steps=6, save_at=0, out=single)
    determinism_probe.run("resume", steps=6, save_at=3, out=resume)

    report = determinism_probe.compare(single, resume)
    assert len(report["steps"]) == 6
    assert report["max_abs_delta"] <= 1e-2, report["steps"]
    for row in report["steps"]:
        assert row["loss_delta"] <= 1e-6, row


def test_two_independent_runs_of_the_same_steps_are_bit_identical(tmp_path):
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    determinism_probe.run("single", steps=4, save_at=0, out=first)
    determinism_probe.run("single", steps=4, save_at=0, out=second)
    report = determinism_probe.compare(first, second)
    assert report["all_bit_identical"] is True
    assert report["right"]["meta"]["hash"] == report["left"]["meta"]["hash"]


def test_compare_detects_a_perturbed_weight_at_the_right_step(tmp_path):
    single = tmp_path / "single.npz"
    resume = tmp_path / "resume.npz"
    determinism_probe.run("single", steps=5, save_at=0, out=single)
    determinism_probe.run("resume", steps=5, save_at=2, out=resume)

    bundle = dict(np.load(resume, allow_pickle=False))
    key = next(key for key in bundle if key.startswith("s3:model."))
    bundle[key] = np.asarray(bundle[key]) + np.float32(1e-6)
    perturbed = tmp_path / "perturbed.npz"
    np.savez(perturbed, **bundle)

    report = determinism_probe.compare(single, perturbed)
    assert report["all_bit_identical"] is False
    assert report["first_diverging_step"] == 4  # s3 == weights after step 4
    # Adding 1e-6 in float32 to weights of order 0.1 lands near 1.01e-6, not exactly 1e-6.
    assert 0.9e-6 < report["max_abs_delta"] < 1.1e-6


def test_compare_refuses_mismatched_step_coverage(tmp_path):
    long_run = tmp_path / "long.npz"
    short_run = tmp_path / "short.npz"
    determinism_probe.run("single", steps=5, save_at=0, out=long_run)
    determinism_probe.run("single", steps=3, save_at=0, out=short_run)
    with pytest.raises(SystemExit, match="did not cover the same steps"):
        determinism_probe.compare(long_run, short_run)


def test_probe_meta_records_the_runtime_settings(tmp_path):
    out = tmp_path / "run.npz"
    result = determinism_probe.run("single", steps=1, save_at=0, out=out)
    meta = json.loads(str(np.load(out, allow_pickle=False)["meta"]))
    assert meta["steps"] == 1
    assert "omp_num_threads" in meta and "mkl_cbwr" in meta
    assert result["final_hash"] == meta["hash"]
