"""Experiment-grid aggregation tests (FPJ-029).

The grid itself needs demonstrations and a simulator, so these tests cover the part that
decides what the report *says*: aggregation across seeds, failure visibility and the
guards that stop a run from silently reusing or overwriting evidence.
"""

import argparse
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import experiments  # noqa: E402


def _cell(task, seed, rate, *, status="ok", reasons=None):
    return {
        "task": task,
        "seed": seed,
        "status": status,
        "success_rate": rate,
        "episode_count": 20,
        "success_episodes": int(rate * 20),
        "success_steps": [10] * int(rate * 20) + [0] * (20 - int(rate * 20)),
        "done_reasons": reasons or {"step_cap": 20},
        "latency_ms_mean": 2.9,
        "nfe": 1,
        "num_points": 1024,
        "checkpoint_reload_max_abs_delta": 0.0,
    }


def test_aggregate_reports_mean_and_standard_deviation_across_seeds():
    cells = [
        _cell("reach-v3", 0, 0.10),
        _cell("reach-v3", 1, 0.20),
        _cell("reach-v3", 2, 0.30),
        _cell("button-press-wall-v3", 0, 0.00, reasons={"step_cap": 20}),
    ]
    report = experiments._aggregate(cells)
    reach = report["tasks"]["reach-v3"]
    assert reach["success_rate_per_seed"] == [0.10, 0.20, 0.30]
    assert reach["success_rate_mean"] == pytest.approx(0.2)
    assert reach["success_rate_std"] == pytest.approx(0.1)
    assert reach["cells_ok"] == 3
    assert reach["episodes_per_seed"] == [20]
    assert reach["done_reasons"] == {"step_cap": 60}
    assert report["tasks"]["button-press-wall-v3"]["success_rate_mean"] == pytest.approx(0.0)
    # A single seed has no sample standard deviation -- it must be None, not 0.
    single = experiments._aggregate([_cell("reach-v3", 0, 0.5)])
    assert single["tasks"]["reach-v3"]["success_rate_std"] is None


def test_failed_cells_are_excluded_from_the_rate_but_stay_in_the_report():
    cells = [
        _cell("reach-v3", 0, 0.2),
        {"task": "reach-v3", "seed": 1, "status": "train_failed", "dir": "/tmp/x"},
    ]
    report = experiments._aggregate(cells)
    reach = report["tasks"]["reach-v3"]
    assert reach["cells_ok"] == 1
    assert reach["cells_failed"] == 1
    assert reach["success_rate_per_seed"] == [0.2]
    assert [cell["seed"] for cell in report["cells"]] == [0, 1]


def test_markdown_keeps_the_disclaimers_and_every_cell():
    args = argparse.Namespace(
        tasks=["reach-v3"], seeds=[0, 1], episodes=20, epochs=30,
        batch_size=32, down_dims=[32, 64], max_steps=200,
    )
    report = experiments._aggregate([_cell("reach-v3", 0, 0.1), _cell("reach-v3", 1, 0.0)])
    text = experiments._markdown(report, args)
    assert "| reach-v3 | 2/2 | 5.00% ± 7.07% |" in text
    assert "seed" in text and "| 0 | ok | 2/20 |" in text
    assert "## What this does not mean" in text
    assert "CPU only" in text and "not the paper's recipe" in text


def test_main_rejects_missing_data_duplicate_seeds_and_overwrite(tmp_path, capsys):
    out = tmp_path / "experiments"
    with pytest.raises(SystemExit, match="missing demonstration stores"):
        experiments.main(
            ["--tasks", "does-not-exist-v3", "--seeds", "0", "--out", str(out),
             "--data-dir", str(tmp_path)]
        )
    store = tmp_path / "reach-v3.zarr"
    store.mkdir()
    with pytest.raises(SystemExit, match="unique"):
        experiments.main(
            ["--tasks", "reach-v3", "--seeds", "0", "0", "--out", str(out),
             "--data-dir", str(tmp_path)]
        )
    out.mkdir()
    (out / "experiments.json").write_text("{}")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        experiments.main(
            ["--tasks", "reach-v3", "--seeds", "0", "--out", str(out),
             "--data-dir", str(tmp_path)]
        )


def test_report_records_the_prohibited_claims(tmp_path, monkeypatch):
    store = tmp_path / "reach-v3.zarr"
    store.mkdir()

    def fake_run_cell(args, task, seed):
        return _cell(task, seed, 0.25)

    monkeypatch.setattr(experiments, "run_cell", fake_run_cell)
    out = tmp_path / "experiments"
    code = experiments.main(
        ["--tasks", "reach-v3", "--seeds", "0", "1", "--out", str(out),
         "--data-dir", str(tmp_path), "--episodes", "20"]
    )
    assert code == 0
    report = json.loads((out / "experiments.json").read_text())
    assert report["tasks"]["reach-v3"]["success_rate_mean"] == pytest.approx(0.25)
    assert "paper success rate reproduced" in report["prohibited_claims"]
    assert report["config"]["episodes"] == 20
    assert (out / "SUMMARY.md").exists()
