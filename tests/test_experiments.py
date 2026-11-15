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


def _cell(task, seed, rate, *, status="ok", reasons=None, target_mode="code"):
    return {
        "target_mode": target_mode,
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
        "action_slice_mode": "code",
        "checkpoint_reload_max_abs_delta": 0.0,
        "train": {
            "seconds_per_step": 0.05,
            "loss": {"last_epoch_mean": 1e-4, "val": 2e-4},
            "config": {
                "data_sha256": "data-hash",
                "seed": seed,
                "target_mode": target_mode,
                "ema_decay": 0.999,
            },
            "data_sha256": "data-hash",
            "train_windows": 100,
            "total_steps": 30,
        },
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
        batch_size=32, down_dims=[32, 64], max_steps=200, device="cpu",
    )
    report = experiments._aggregate([_cell("reach-v3", 0, 0.1), _cell("reach-v3", 1, 0.0)])
    report["preflight"] = {"shared_config_sha256": "abc"}
    text = experiments._markdown(report, args)
    assert "| code | reach-v3 | 2/2 | 5.00% ± 7.07% |" in text
    assert "seed" in text and "| code | reach-v3 | 0 | ok | 2/20 |" in text
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

    def fake_run_cell(args, target_mode, task, seed):
        return _cell(task, seed, 0.25, target_mode=target_mode)

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


def test_training_command_enables_periodic_checkpoints(tmp_path):
    args = argparse.Namespace(
        data_dir=tmp_path,
        epochs=30,
        batch_size=64,
        lr=1e-3,
        warmup=20,
        down_dims=[512, 1024, 2048],
        embed_dim=128,
        encoder_out=64,
        pointnet_out=64,
        kernel_size=5,
        n_groups=8,
        var_fc_hidden_dim=512,
        num_points=1024,
        device="cuda",
        run_label="full-channel-limited-budget",
        checkpoint_every=5,
        ema_decay=0.999,
        ema_warmup_steps=10,
        overwrite=False,
    )
    resume = tmp_path / "epoch-0005.ckpt"
    command = experiments.train_command(
        args,
        "reach-wall-v3",
        0,
        tmp_path / "run",
        target_mode="detached",
        resume_checkpoint=resume,
    )
    position = command.index("--checkpoint-every")
    assert command[position + 1] == "5"
    assert command[command.index("--target-mode") + 1] == "detached"
    assert command[command.index("--ema-decay") + 1] == "0.999"
    assert command[command.index("--ema-warmup-steps") + 1] == "10"
    assert command[command.index("--resume") + 1] == str(resume)


def test_matched_validation_requires_only_target_mode_to_differ():
    cells = []
    for mode, rate in (("code", 0.2), ("detached", 0.3)):
        cells.append(_cell("reach-wall-v3", 0, rate, target_mode=mode))
    result = experiments._matched_validation(cells, ["code", "detached"])
    assert result["passed"] is True
    details = result["pairs"][0]["checks"]["code_vs_detached"]["details"]
    assert all(details.values())

    cells[1]["train"]["config"]["seed"] = 99
    result = experiments._matched_validation(cells, ["code", "detached"])
    assert result["passed"] is False
    assert (
        result["pairs"][0]["checks"]["code_vs_detached"]["details"]
        ["training_config_except_target_equal"]
        is False
    )


def test_comparison_main_records_shared_preflight_and_both_arms(tmp_path, monkeypatch):
    store = tmp_path / "reach-wall-v3.zarr"
    store.mkdir()
    (store / "metadata").write_bytes(b"same-data")

    def fake_run_cell(args, target_mode, task, seed):
        return _cell(task, seed, 0.25, target_mode=target_mode)

    monkeypatch.setattr(experiments, "run_cell", fake_run_cell)
    out = tmp_path / "comparison"
    code = experiments.main(
        [
            "--tasks", "reach-wall-v3",
            "--seeds", "0", "1", "2",
            "--target-modes", "code", "detached",
            "--ema-decay", "0.999",
            "--out", str(out),
            "--data-dir", str(tmp_path),
        ]
    )
    assert code == 0
    report = json.loads((out / "experiments.json").read_text(encoding="utf-8"))
    assert report["target_modes"] == ["code", "detached"]
    assert report["preflight"]["passed"] is True
    assert report["matched_validation"]["passed"] is True
    assert len(report["cells"]) == 6
    assert set(report["targets"]) == {"code", "detached"}
