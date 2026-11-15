"""32-window overfit smoke run for the Jittor FlowPolicy port (FPJ-023).

This is **not** paper-scale training.  It proves that the Zarr dataset, the
observation encoder, the conditional U-Net, the CFM loss and the AdamW update
form one working learning loop: 32 deterministic windows must be memorised, so
the loss has to drop clearly.

Usage::

    python scripts/overfit_32_windows.py --steps 120
    python scripts/overfit_32_windows.py --zarr data/x.zarr --num-points 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

# Jittor spins up one thread per core by default.  On this 24-core WSL box that
# turns a 0.15 s/step CPU smoke run into a 3 s/step one (measured), so the CPU
# overfit caps OpenMP *before* jittor is imported.  It is an environment tuning
# detail, not an algorithmic change.
os.environ.setdefault("OMP_NUM_THREADS", "4")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import jittor as jt  # noqa: E402

from flowpolicy_jittor.data.dataset import (  # noqa: E402
    MetaworldZarrDataset,
    select_windows,
)
from flowpolicy_jittor.model import FlowPolicy  # noqa: E402

LABEL = "32-window smoke overfit (NOT paper-scale training)"
WINDOW_COUNT = 32


def write_synthetic_store(
    path: Any,
    *,
    episodes: int = 4,
    frames: int = 40,
    num_points: int = 128,
    action_dim: int = 4,
    state_dim: int = 9,
    seed: int = 0,
) -> None:
    """Write a learnable synthetic MetaWorld-shaped store.

    Actions are a fixed linear map of the robot state plus noise, so the policy
    can actually fit them instead of only memorising noise.
    """
    import zarr

    rng = np.random.default_rng(seed)
    lengths = [frames] * episodes
    ends = np.cumsum(lengths).astype(np.int64)
    total = int(ends[-1])
    state = rng.normal(size=(total, state_dim)).astype(np.float32)
    point_cloud = rng.normal(size=(total, num_points, 3)).astype(np.float32)
    weight = rng.normal(size=(state_dim, action_dim)).astype(np.float32)
    action = (state @ weight + 0.01 * rng.normal(size=(total, action_dim))).astype(
        np.float32
    )

    store = zarr.open(str(path), mode="w")
    store.create_dataset("action", data=action)
    store.create_dataset("agent_pos", data=state)
    store.create_dataset("point_cloud", data=point_cloud)
    store.create_dataset("episode_ends", data=ends)


def build_policy(num_points: int, action_dim: int = 4, state_dim: int = 9) -> FlowPolicy:
    """Build the small-channel CPU policy used for the smoke overfit."""
    return FlowPolicy(
        shape_meta={
            "obs": {
                "point_cloud": {"shape": [num_points, 3], "type": "point_cloud"},
                "agent_pos": {"shape": [state_dim], "type": "low_dim"},
            },
            "action": {"shape": [action_dim]},
        },
        horizon=4,
        n_action_steps=4,
        n_obs_steps=2,
        diffusion_step_embed_dim=16,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=4,
        encoder_output_dim=32,
        state_mlp_size=(32, 32),
        pointcloud_encoder_cfg={
            "in_channels": 3,
            "out_channels": 32,
            "use_layernorm": True,
            "final_norm": "layernorm",
            "normal_channel": False,
        },
        var_fc_hidden_dim=16,
    )


def build_batch(dataset: MetaworldZarrDataset, indices: np.ndarray) -> Dict[str, Any]:
    """Stack the selected windows into one fixed in-memory batch."""
    samples = [dataset[int(index)] for index in indices]
    return {
        "obs": {
            "point_cloud": np.stack([item["obs"]["point_cloud"] for item in samples]),
            "agent_pos": np.stack([item["obs"]["agent_pos"] for item in samples]),
        },
        "action": np.stack([item["action"] for item in samples]),
    }


def action_error(policy: FlowPolicy, batch: Mapping[str, Any], noise) -> float:
    """Mean absolute error between the one-step prediction and the demo action.

    This is the honest learning signal for a memorisation test: the CFM loss is
    already ~1e-4 at initialisation because ``(r - t)`` is only ``delta``, so a
    "50% loss drop" is a weak and noisy criterion, while the predicted action
    error must fall by a large factor once the 32 windows are memorised.
    """
    prediction = policy.predict_action(batch["obs"], noise=noise)["action_pred"]
    target = jt.array(np.asarray(batch["action"], dtype=np.float32))
    return float(jt.abs(prediction - target).mean().item())


def _probe_noise(windows: int, horizon: int, action_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    return jt.array(
        rng.normal(size=(windows, horizon, action_dim)).astype(np.float32)
    ).detach().stop_grad()


def run_overfit(
    *,
    zarr_path: Any,
    steps: int,
    lr: float,
    seed: int,
    num_points: int,
    window_count: int = WINDOW_COUNT,
    log_path: Optional[Any] = None,
    log_every: int = 10,
    enforce_reduction: bool = True,
) -> Dict[str, Any]:
    """Run the overfit and return the structured log record."""
    if steps < 1:
        raise ValueError("steps must be positive")
    jt.flags.use_cuda = 0
    jt.set_global_seed(seed)

    dataset = MetaworldZarrDataset(
        zarr_path,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=4,
        num_points=num_points,
        val_ratio=0.0,
    )
    indices = select_windows(dataset, window_count, seed)
    batch = build_batch(dataset, indices)

    policy = build_policy(num_points)
    policy.set_normalizer(dataset.get_normalizer())
    optimizer = jt.optim.AdamW(
        policy.trainable_parameters(), lr=lr, weight_decay=1e-6
    )

    history = []
    noise = _probe_noise(window_count, policy.horizon, policy.action_dim, seed + 1)
    error_initial = action_error(policy, batch, noise)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        loss, _ = policy.compute_loss(batch)
        value = float(loss.item())
        if not np.isfinite(value):
            raise RuntimeError(f"non-finite loss at step {step}: {value}")
        optimizer.step(loss)
        if step == 1 or step % log_every == 0 or step == steps:
            history.append({"step": step, "loss": value})
    elapsed = time.perf_counter() - started

    error_final = action_error(policy, batch, noise)
    target = np.asarray(batch["action"], dtype=np.float32)
    mean_action = target.reshape(-1, target.shape[-1]).mean(axis=0)
    baselines = {
        "zero_predictor_mae": float(np.abs(target).mean()),
        "mean_action_predictor_mae": float(np.abs(target - mean_action).mean()),
    }
    losses = [entry["loss"] for entry in history]
    initial = losses[0]
    best = min(losses)
    final = losses[-1]
    if enforce_reduction and error_final > 0.5 * error_initial:
        raise RuntimeError(
            "32-window overfit did not halve the action error: "
            f"initial={error_initial:.6f} final={error_final:.6f}"
        )
    best_baseline = min(baselines.values())
    if enforce_reduction and error_final > 0.9 * best_baseline:
        raise RuntimeError(
            "32-window overfit is no better than a constant predictor: "
            f"final={error_final:.6f} baseline={best_baseline:.6f}"
        )

    record = {
        "task": "FPJ-023",
        "label": LABEL,
        "windows": int(window_count),
        "window_indices": [int(index) for index in indices],
        "seed": int(seed),
        "steps": int(steps),
        "learning_rate": float(lr),
        "num_points": int(num_points),
        "seconds": round(elapsed, 3),
        "action_error": {
            "initial": error_initial,
            "final": error_final,
            "reduction_ratio": round(1.0 - error_final / error_initial, 6),
            "reduction_enforced": bool(enforce_reduction),
            "baselines": baselines,
        },
        "loss": {
            "initial": initial,
            "min": best,
            "final": final,
            "reduction_ratio": round(1.0 - best / initial, 6),
            "history": history,
        },
        "prohibited_claims": [
            "paper success rate reproduced",
            "paper-scale training",
            "CUDA verified",
            "MetaWorld rollout completed",
        ],
    }
    if log_path is not None:
        destination = Path(log_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=LABEL)
    parser.add_argument("--zarr", type=Path, default=None, help="existing Zarr store")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=32)
    parser.add_argument("--synthetic-episodes", type=int, default=4)
    parser.add_argument("--synthetic-frames", type=int, default=40)
    parser.add_argument("--log", type=Path, default=None, help="JSON log path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.zarr is None:
        directory = tempfile.mkdtemp(prefix="fpj023-")
        zarr_path = Path(directory) / "synthetic.zarr"
        write_synthetic_store(
            zarr_path,
            episodes=args.synthetic_episodes,
            frames=args.synthetic_frames,
            num_points=args.num_points,
            seed=args.seed,
        )
        print(f"[data] synthetic store: {zarr_path}")
    else:
        zarr_path = args.zarr

    record = run_overfit(
        zarr_path=zarr_path,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        num_points=args.num_points,
        log_path=args.log,
    )
    print(f"[run] {LABEL}")
    print(
        "[action-error] initial={initial:.6f} final={final:.6f} "
        "reduction={reduction_ratio:.2%}".format(**record["action_error"])
    )
    print(
        "[baselines] zero={zero_predictor_mae:.6f} mean_action={mean_action_predictor_mae:.6f}".format(
            **record["action_error"]["baselines"]
        )
    )
    print(
        "[loss] initial={initial:.6f} min={min:.6f} final={final:.6f}".format(
            **record["loss"]
        )
    )
    print(f"[time] {record['seconds']}s for {record['steps']} steps "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})")
    if args.log is not None:
        print(f"[log] {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
