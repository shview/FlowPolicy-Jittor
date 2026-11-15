"""End-to-end CLI checkpoint/resume tests for FPJ-033."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

jt = pytest.importorskip("jittor")
zarr = pytest.importorskip("zarr")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_smoke  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = previous


def _write_store(path: Path) -> None:
    lengths = [6, 6, 6, 6]
    total = sum(lengths)
    values = np.concatenate(
        [np.linspace(index, index + 0.5, length, dtype=np.float32) for index, length in enumerate(lengths)]
    )
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    data.create_dataset("action", data=np.repeat(values[:, None], 4, axis=1))
    data.create_dataset("state", data=np.repeat(values[:, None], 9, axis=1))
    points = np.repeat(values[:, None, None], 8, axis=1)
    points = np.repeat(points, 3, axis=2)
    data.create_dataset("point_cloud", data=points)
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=np.cumsum(lengths, dtype=np.int64))
    assert total == data["action"].shape[0]


def _args(store: Path, out: Path):
    return [
        "--data", str(store),
        "--epochs", "2",
        "--batch-size", "8",
        "--lr", "0.001",
        "--warmup", "1",
        "--seed", "7",
        "--num-points", "8",
        "--val-ratio", "0.25",
        "--down-dims", "8", "16", "32",
        "--embed-dim", "8",
        "--encoder-out", "8",
        "--pointnet-out", "8",
        "--kernel-size", "3",
        "--n-groups", "4",
        "--var-fc-hidden-dim", "8",
        "--checkpoint-every", "1",
        "--out", str(out),
    ]


def test_cli_planned_stop_and_resume_complete_the_same_schedule(tmp_path):
    store, out = tmp_path / "demo.zarr", tmp_path / "run"
    _write_store(store)

    assert train_smoke.main(_args(store, out) + ["--stop-after-epoch", "1"]) == 0
    checkpoint = out / "checkpoints" / "epoch-0001.ckpt"
    assert checkpoint.is_file()
    partial = json.loads((out / "partial_summary.json").read_text(encoding="utf-8"))
    assert partial["status"] == "stopped"
    assert partial["completed_epochs"] == 1
    assert not (out / "policy.ckpt").exists()

    assert train_smoke.main(_args(store, out) + ["--resume", str(checkpoint)]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    epochs = json.loads((out / "epochs.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]

    assert summary["resume"]["resumed"] is True
    assert summary["resume"]["start_epoch"] == 1
    assert summary["completed_epochs"] == 2
    assert summary["steps_this_invocation"] == summary["steps_per_epoch"]
    assert summary["seconds_per_step"] == summary["seconds_per_step_this_invocation"]
    assert summary["seconds_per_step"] == pytest.approx(
        summary["wall_seconds_this_invocation"] / summary["steps_this_invocation"]
    )
    assert progress["status"] == "completed"
    assert [entry["epoch"] for entry in epochs] == [1, 2]
    assert [entry["step"] for entry in records] == list(range(1, summary["total_steps"] + 1))
    assert (out / "policy.ckpt").is_file()
    assert (out / "trainer.ckpt").is_file()
    assert not (out / "partial_summary.json").exists()


def test_resume_output_cannot_be_combined_with_overwrite(tmp_path):
    target = tmp_path / "run"
    target.mkdir()
    with pytest.raises(SystemExit, match="cannot be used together"):
        train_smoke.prepare_output_dir(target, overwrite=True, resume=True)
