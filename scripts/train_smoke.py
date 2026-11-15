"""Small-channel training smoke test on real demonstrations (FPJ-027).

Trains the Jittor FlowPolicy on a recorded Zarr store with a small configuration,
logs step/lr/loss to JSONL, and saves a policy checkpoint.  This is a *smoke test*:
it proves the data -> model -> optimiser -> checkpoint path works on real data.  It is
not the paper's training recipe (small channels; EMA is optional). CPU remains the
default, while ``--device cuda`` enables an explicit cloud-GPU run.

Example:
    python scripts/train_smoke.py --data data/reach-v3.zarr --epochs 100 --out outputs/smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_CHANNELS = (512, 1024, 2048)
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def is_paper_architecture(args) -> bool:
    return (
        tuple(args.down_dims) == FULL_CHANNELS
        and args.embed_dim == 128
        and args.encoder_out == 64
        and args.pointnet_out == 64
        and args.kernel_size == 5
        and args.n_groups == 8
        and args.var_fc_hidden_dim == 512
    )


def run_note(args) -> str:
    if is_paper_architecture(args):
        return (
            f"paper-architecture {args.device.upper()} pilot; reduced batch/epochs and "
            "replacement simulator, NOT the paper training recipe"
        )
    return f"small-channel {args.device.upper()} smoke training; NOT the paper configuration"


def runtime_fingerprint(jt, device: str) -> Dict[str, object]:
    fingerprint: Dict[str, object] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "jittor": getattr(jt, "__version__", "unknown"),
        "device": device,
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    if device == "cuda":
        command = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        fingerprint["nvidia_smi"] = completed.stdout.strip() if completed.returncode == 0 else None
        fingerprint["nvidia_smi_error"] = completed.stderr.strip() if completed.returncode != 0 else None
    return fingerprint


def prepare_output_dir(path: Path, overwrite: bool, *, resume: bool = False) -> Path:
    """Create ``path`` unless it already holds results and overwriting is not allowed."""
    path = Path(path)
    if resume and overwrite:
        raise SystemExit("--resume and --overwrite cannot be used together")
    if path.exists() and any(path.iterdir()) and not overwrite and not resume:
        raise SystemExit(f"refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_atomic(path: Path, value: object) -> None:
    """Replace a JSON report atomically so interruption cannot leave half a file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_epoch_history(path: Path, *, completed_epochs: int) -> List[Dict[str, object]]:
    """Load and validate the epoch metrics paired with a periodic checkpoint."""

    if not path.is_file():
        raise SystemExit(
            f"resume checkpoint represents {completed_epochs} epochs but {path} is missing"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) < completed_epochs:
        raise SystemExit(
            f"epoch history has {len(value) if isinstance(value, list) else 'invalid'} entries; "
            f"resume checkpoint requires {completed_epochs}"
        )
    history = value[:completed_epochs]
    expected = list(range(1, completed_epochs + 1))
    actual = [entry.get("epoch") if isinstance(entry, dict) else None for entry in history]
    if actual != expected:
        raise SystemExit(f"epoch history is not contiguous: expected {expected}, got {actual}")
    return [dict(entry) for entry in history]


def collate(items: List[Dict[str, object]]) -> Dict[str, object]:
    """Stack dataset windows into a batch, preserving the nested obs layout."""
    batch: Dict[str, object] = {}
    observations = [item["obs"] for item in items]
    batch["obs"] = {
        key: np.stack([np.asarray(observation[key]) for observation in observations])
        for key in observations[0]
    }
    batch["action"] = np.stack([np.asarray(item["action"]) for item in items])
    return batch


def build_policy(
    dataset,
    *,
    down_dims,
    embed_dim,
    encoder_out,
    pointnet_out,
    kernel_size,
    n_groups,
    var_fc_hidden_dim,
    seed,
    action_slice_mode="code",
):
    import jittor as jt

    from flowpolicy_jittor.model import FlowPolicy

    sample = dataset[0]
    state = np.asarray(sample["obs"]["agent_pos"])
    cloud = np.asarray(sample["obs"]["point_cloud"])
    action = np.asarray(sample["action"])
    shape_meta = {
        "obs": {
            "point_cloud": {"shape": list(cloud.shape[-2:]), "type": "point_cloud"},
            "agent_pos": {"shape": list(state.shape[-1:]), "type": "low_dim"},
        },
        "action": {"shape": list(action.shape[-1:])},
    }
    jt.set_global_seed(seed)
    policy = FlowPolicy(
        shape_meta=shape_meta,
        horizon=dataset.horizon,
        n_action_steps=dataset.n_action_steps,
        n_obs_steps=dataset.n_obs_steps,
        down_dims=down_dims,
        diffusion_step_embed_dim=embed_dim,
        kernel_size=kernel_size,
        n_groups=n_groups,
        encoder_output_dim=encoder_out,
        state_mlp_size=(64, 64),
        pointcloud_encoder_cfg={
            "in_channels": 3,
            "out_channels": pointnet_out,
            "use_layernorm": True,
            "final_norm": "layernorm",
            "normal_channel": False,
        },
        var_fc_hidden_dim=var_fc_hidden_dim,
        action_slice_mode=action_slice_mode,
    )
    policy.set_normalizer(dataset.get_normalizer())
    return policy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--encoder-out", type=int, default=64)
    parser.add_argument("--pointnet-out", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--var-fc-hidden-dim", type=int, default=64)
    parser.add_argument("--run-label", default="smoke")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Jittor backend used for training",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="enables EMA target tracking with this decay (paper: 0.999)",
    )
    parser.add_argument("--ema-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        metavar="EPOCHS",
        help="save a resumable trainer checkpoint every N epochs (0 disables periodic saves)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume a periodic trainer checkpoint using the same --out and training config",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help="planned early stop used to test/preempt a run while keeping --epochs as the target",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    import jittor as jt

    from flowpolicy_jittor.data import MetaworldZarrDataset
    from flowpolicy_jittor.training import Trainer

    if args.device == "cuda" and not jt.has_cuda:
        raise SystemExit("--device cuda requested, but Jittor reports no CUDA backend")
    jt.flags.use_cuda = int(args.device == "cuda")
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.ema_decay is not None and not 0.0 < args.ema_decay < 1.0:
        raise SystemExit("--ema-decay must lie strictly between 0 and 1")
    if args.ema_warmup_steps < 0:
        raise SystemExit("--ema-warmup-steps must be non-negative")
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint-every must be non-negative")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        raise SystemExit("--stop-after-epoch must be within [1, --epochs]")
    if args.resume is not None and not args.resume.is_file():
        raise SystemExit(f"resume checkpoint does not exist: {args.resume}")
    out_dir = prepare_output_dir(
        args.out, args.overwrite, resume=args.resume is not None
    )
    log_path = out_dir / "train.jsonl"
    if args.overwrite and log_path.exists():
        # A fresh run must not append to (and thereby mix in) a previous run's log.
        log_path.unlink()

    dataset = MetaworldZarrDataset(
        args.data,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=4,
        num_points=args.num_points,
        val_ratio=args.val_ratio,
        seed=args.seed,
        mode="train",
    )
    validation = dataset.get_validation_dataset()
    windows = int(dataset.num_windows)
    steps_per_epoch = max(1, windows // args.batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs)
    policy = build_policy(
        dataset,
        down_dims=tuple(args.down_dims),
        embed_dim=args.embed_dim,
        encoder_out=args.encoder_out,
        pointnet_out=args.pointnet_out,
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
        var_fc_hidden_dim=args.var_fc_hidden_dim,
        seed=args.seed,
    )
    trainer = Trainer(
        policy,
        total_steps=total_steps,
        lr=args.lr,
        warmup_steps=args.warmup,
        min_lr_ratio=0.05,
        seed=args.seed,
        log_path=log_path,
        ema_decay=args.ema_decay,
        ema_warmup_steps=args.ema_warmup_steps,
    )

    rng = np.random.default_rng([args.seed, 0xA11CE])
    epochs_path = out_dir / "epochs.json"
    start_epoch = 0
    resumed_from = None
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        if trainer.global_step % steps_per_epoch != 0:
            raise SystemExit(
                "resume checkpoint is not at an epoch boundary: "
                f"global_step={trainer.global_step}, steps_per_epoch={steps_per_epoch}"
            )
        start_epoch = trainer.global_step // steps_per_epoch
        if start_epoch >= args.epochs:
            raise SystemExit(
                f"resume checkpoint already completed {start_epoch}/{args.epochs} epochs"
            )
        epochs_log = load_epoch_history(epochs_path, completed_epochs=start_epoch)
        resumed_from = str(args.resume)
        # Fresh runs use one stateful NumPy generator across epochs.  Replay only its
        # permutations so the next epoch receives the same order after a restart.
        for _ in range(start_epoch):
            rng.permutation(windows)
    else:
        epochs_log = []

    target_epoch = args.epochs
    if args.stop_after_epoch is not None:
        target_epoch = min(target_epoch, args.stop_after_epoch)
    if target_epoch <= start_epoch:
        raise SystemExit(
            f"planned stop epoch {target_epoch} is not after resumed epoch {start_epoch}"
        )
    started = time.perf_counter()
    start_step = trainer.global_step
    first_loss = None
    minimum_loss = None
    checkpoints = []
    for epoch in range(start_epoch, target_epoch):
        order = rng.permutation(windows)
        epoch_losses = []
        for step in range(steps_per_epoch):
            selection = order[step * args.batch_size : (step + 1) * args.batch_size]
            if selection.size == 0:
                continue
            batch = collate([dataset[int(index)] for index in selection])
            record = trainer.train_step(batch)
            loss = float(record["loss"])
            if first_loss is None:
                first_loss = loss
            minimum_loss = loss if minimum_loss is None else min(minimum_loss, loss)
            epoch_losses.append(loss)
        epochs_log.append(
            {
                "epoch": epoch + 1,
                "mean_loss": float(np.mean(epoch_losses)),
                "min_loss": float(np.min(epoch_losses)),
                "lr": float(trainer.learning_rate),
            }
        )
        write_json_atomic(epochs_path, epochs_log)
        epoch_number = epoch + 1
        should_checkpoint = (
            args.checkpoint_every > 0 and epoch_number % args.checkpoint_every == 0
        ) or (epoch_number == target_epoch and target_epoch < args.epochs)
        if should_checkpoint:
            checkpoint_dir = out_dir / "checkpoints"
            periodic = checkpoint_dir / f"epoch-{epoch_number:04d}.ckpt"
            trainer.save_checkpoint(periodic)
            checkpoints.append(
                {
                    "epoch": epoch_number,
                    "global_step": trainer.global_step,
                    "path": str(periodic),
                    "sha256": sha256_file(periodic),
                }
            )
            write_json_atomic(
                out_dir / "progress.json",
                {
                    "status": "running" if epoch_number < target_epoch else "stopped",
                    "target_epochs": args.epochs,
                    "completed_epochs": epoch_number,
                    "global_step": trainer.global_step,
                    "latest_checkpoint": checkpoints[-1],
                    "resumed_from": resumed_from,
                    "claim_boundary": (
                        "Trainer state and schedule are resumable; long-process CPU "
                        "continuation is validated within 1e-2, not bit-exact."
                    ),
                },
            )
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch + 1}/{args.epochs} mean {epochs_log[-1]['mean_loss']:.6e} "
                f"min {epochs_log[-1]['min_loss']:.6e}",
                flush=True,
            )
    elapsed = time.perf_counter() - started

    if target_epoch < args.epochs:
        partial = {
            "status": "stopped",
            "target_epochs": args.epochs,
            "completed_epochs": target_epoch,
            "global_step": trainer.global_step,
            "steps_this_invocation": trainer.global_step - start_step,
            "resumed_from": resumed_from,
            "checkpoints_written_this_invocation": checkpoints,
            "wall_seconds_this_invocation": elapsed,
            "next_step": (
                f"rerun with the same arguments and --resume "
                f"{checkpoints[-1]['path']} without --stop-after-epoch"
            ),
            "claim_boundary": (
                "This is an intentional partial run, not a completed training or rollout result."
            ),
        }
        write_json_atomic(out_dir / "partial_summary.json", partial)
        print(json.dumps(partial, indent=2))
        return 0

    val_loss = None
    if int(validation.num_windows) > 0:
        value_sample = min(int(validation.num_windows), 32)
        indices = np.linspace(0, int(validation.num_windows) - 1, value_sample).astype(int)
        batch = collate([validation[int(index)] for index in indices])
        loss, _logs = policy.compute_loss(batch)
        val_loss = float(loss.item())

    checkpoint = out_dir / "policy.ckpt"
    policy.save_checkpoint(checkpoint, overwrite=args.overwrite)
    # Always keep a final trainer checkpoint: even without EMA it carries AdamW moments,
    # global_step and the schedule.  The policy checkpoint remains the rollout artifact.
    trainer_checkpoint = out_dir / "trainer.ckpt"
    trainer.save_checkpoint(trainer_checkpoint, overwrite=args.overwrite)
    model_config = {
        "down_dims": args.down_dims,
        "embed_dim": args.embed_dim,
        "encoder_out": args.encoder_out,
        "pointnet_out": args.pointnet_out,
        "kernel_size": args.kernel_size,
        "n_groups": args.n_groups,
        "var_fc_hidden_dim": args.var_fc_hidden_dim,
        "lr": args.lr,
        "warmup": args.warmup,
        "num_points": args.num_points,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    config_sha256 = hashlib.sha256(
        json.dumps(model_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    paper_architecture = is_paper_architecture(args)
    summary = {
        "note": run_note(args),
        "run_label": args.run_label,
        "device": args.device,
        "paper_architecture": paper_architecture,
        "runtime": runtime_fingerprint(jt, args.device),
        "data": str(args.data),
        "data_sha256": sha256_tree(args.data),
        "train_windows": windows,
        "val_windows": int(validation.num_windows),
        "epochs": args.epochs,
        "completed_epochs": target_epoch,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "steps_this_invocation": trainer.global_step - start_step,
        "batch_size": args.batch_size,
        "config": model_config,
        "config_sha256": config_sha256,
        "trainable_parameters": int(
            sum(int(np.prod(parameter.shape)) for parameter in policy.trainable_parameters())
        ),
        "loss": {
            "first": first_loss,
            "min": minimum_loss,
            "last_epoch_mean": epochs_log[-1]["mean_loss"] if epochs_log else None,
            "val": val_loss,
        },
        # Backward-compatible names retain their old meaning for fresh runs; for a
        # resumed invocation the denominator must be the steps actually executed now,
        # not the schedule's total target.
        "wall_seconds": elapsed,
        "wall_seconds_this_invocation": elapsed,
        "seconds_per_step": elapsed / max(1, trainer.global_step - start_step),
        "seconds_per_step_this_invocation": elapsed / max(
            1, trainer.global_step - start_step
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "trainer_checkpoint": str(trainer_checkpoint),
        "trainer_checkpoint_sha256": sha256_file(trainer_checkpoint),
        "resume": {
            "resumed": args.resume is not None,
            "resumed_from": resumed_from,
            "start_epoch": start_epoch,
            "checkpoint_every": args.checkpoint_every,
            "checkpoints_written_this_invocation": checkpoints,
            "claim_boundary": (
                "State and schedule restore exactly; prior measurements allow up to 1e-2 "
                "long-process parameter drift, so bit-exact resume is not claimed."
            ),
        },
        "ema": {
            "enabled": trainer.ema is not None,
            "decay": args.ema_decay,
            "warmup_steps": args.ema_warmup_steps,
            "optimization_step": None if trainer.ema is None else trainer.ema.optimization_step,
            "effective_decay": None if trainer.ema is None else trainer.ema.effective_decay(),
            "trainer_checkpoint": str(trainer_checkpoint),
            "note": (
                "EMA weights are the target for the flow-matching update in the paper; "
                "this run keeps them in the trainer checkpoint only and evaluates the "
                "online policy"
            ),
        },
        "prohibited_claims": [
            (
                "paper architecture does not make this the paper training recipe; the batch, "
                "epochs, data and simulator remain reduced/substituted"
                if paper_architecture
                else f"not the paper training configuration (small channels, {args.device.upper()})"
            ),
            "with --ema-decay EMA weights are tracked but not used as the loss target",
            "no success rate is implied by the training loss",
        ],
    }
    write_json_atomic(out_dir / "summary.json", summary)
    write_json_atomic(epochs_path, epochs_log)
    write_json_atomic(
        out_dir / "progress.json",
        {
            "status": "completed",
            "target_epochs": args.epochs,
            "completed_epochs": target_epoch,
            "global_step": trainer.global_step,
            "policy_checkpoint": {
                "path": str(checkpoint),
                "sha256": summary["checkpoint_sha256"],
            },
            "trainer_checkpoint": {
                "path": str(trainer_checkpoint),
                "sha256": summary["trainer_checkpoint_sha256"],
            },
            "resumed_from": resumed_from,
            "claim_boundary": summary["resume"]["claim_boundary"],
        },
    )
    (out_dir / "partial_summary.json").unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
