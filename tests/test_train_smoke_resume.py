"""End-to-end CLI checkpoint/resume tests for FPJ-033."""

import json
import hashlib
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


def _write_store(path: Path, *, offset: float = 0.0) -> None:
    lengths = [6, 6, 6, 6]
    total = sum(lengths)
    values = np.concatenate(
        [np.linspace(index, index + 0.5, length, dtype=np.float32) for index, length in enumerate(lengths)]
    ) + np.float32(offset)
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

    other_store = tmp_path / "other-demo.zarr"
    _write_store(other_store, offset=100.0)
    with pytest.raises(SystemExit, match="provenance mismatch.*data_sha256"):
        train_smoke.main(
            _args(other_store, out) + ["--resume", str(checkpoint)]
        )
    with pytest.raises(SystemExit, match="provenance mismatch.*batch_size"):
        train_smoke.main(
            _args(store, out)
            + ["--batch-size", "10", "--resume", str(checkpoint)]
        )

    assert train_smoke.main(_args(store, out) + ["--resume", str(checkpoint)]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    epochs = json.loads((out / "epochs.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]

    assert summary["resume"]["resumed"] is True
    assert summary["ema"]["role"] == "none"
    assert summary["config"]["ema_decay"] is None
    assert summary["config"]["ema_warmup_steps"] == 0
    assert summary["config"]["batch_size"] == 8
    assert summary["config"]["data_sha256"] == summary["data_sha256"]
    expected_config_hash = hashlib.sha256(
        json.dumps(
            summary["config"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    assert summary["config_sha256"] == expected_config_hash
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


def test_cli_detached_mode_requires_ema_and_records_target_role(tmp_path):
    store = tmp_path / "demo.zarr"
    _write_store(store)
    with pytest.raises(SystemExit, match="detached requires --ema-decay"):
        train_smoke.main(
            _args(store, tmp_path / "missing-ema")
            + ["--target-mode", "detached"]
        )
    with pytest.raises(SystemExit, match="warmup-steps requires --ema-decay"):
        train_smoke.main(
            _args(store, tmp_path / "warmup-without-ema")
            + ["--ema-warmup-steps", "2"]
        )

    out = tmp_path / "detached"
    assert train_smoke.main(
        _args(store, out)
        + ["--target-mode", "detached", "--ema-decay", "0.9"]
    ) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["target_mode"] == "detached"
    assert summary["config"]["ema_decay"] == pytest.approx(0.9)
    assert summary["config"]["ema_warmup_steps"] == 0
    assert summary["ema"]["role"] == "detached_loss_target"
    assert summary["ema"]["optimization_step"] == summary["total_steps"]
    payload = jt.load(str(out / "trainer.ckpt"))
    assert payload["trainer"]["target_mode"] == "detached"
    assert payload["trainer"]["ema"]["kind"].endswith("detached-ema-target")

    from flowpolicy_jittor.data import MetaworldZarrDataset

    dataset = MetaworldZarrDataset(
        store,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=4,
        num_points=8,
        val_ratio=0.25,
        seed=7,
        mode="train",
    )
    rollout_policy = train_smoke.build_policy(
        dataset,
        down_dims=(8, 16, 32),
        embed_dim=8,
        encoder_out=8,
        pointnet_out=8,
        kernel_size=3,
        n_groups=4,
        var_fc_hidden_dim=8,
        seed=7,
        target_mode="detached",
    )
    rollout_policy.load_checkpoint(out / "policy.ckpt")

    epoch_one = out / "checkpoints" / "epoch-0001.ckpt"
    with pytest.raises(SystemExit, match="provenance mismatch.*ema_decay"):
        train_smoke.main(
            _args(store, out)
            + [
                "--target-mode",
                "detached",
                "--ema-decay",
                "0.5",
                "--resume",
                str(epoch_one),
            ]
        )
