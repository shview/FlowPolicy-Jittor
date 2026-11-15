"""CUDA capability probe plus a full-channel configuration smoke run (FPJ-028).

Two questions, two answers:

1. *Can this machine run Jittor kernels on CUDA?*  The script asks Jittor, tries a real
   device kernel, and records the answer -- including the exact error when the answer is
   "no".  A CPU fallback is never reported as a CUDA result.
2. *Does the full paper channel configuration (512/1024/2048) actually run?*  The script
   builds that configuration, runs one real forward/backward/update and records the
   parameter count, timings and peak memory.

Example:
    python scripts/cuda_smoke.py --out outputs/cuda
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

FULL_CHANNELS = (512, 1024, 2048)


def _peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes.
    return float(usage) / 1024.0 if platform.system() != "Darwin" else float(usage) / (1024.0 * 1024.0)


def _short_error(error: BaseException) -> str:
    """Keep the informative tail: Jittor appends the real cause at the end."""
    text = " ".join(str(error).split())
    return text[-400:] if len(text) > 400 else text


def probe_cuda() -> Dict[str, Any]:
    """Return what CUDA actually does here; never assume."""
    import jittor as jt

    report: Dict[str, Any] = {"jittor_version": str(jt.__version__), "has_cuda_flag": bool(jt.has_cuda)}
    previous = int(jt.flags.use_cuda)
    try:
        jt.flags.use_cuda = 1
    except Exception as error:  # noqa: BLE001 - the flag itself refuses when no toolkit exists
        jt.flags.use_cuda = previous
        report.update(
            {
                "available": False,
                "stage": "set_use_cuda",
                "error": f"{type(error).__name__}: {_short_error(error)}",
            }
        )
        return report
    try:
        started = time.perf_counter()
        left = jt.randn((128, 128))
        right = jt.randn((128, 128))
        total = (left @ right).sum()
        total.sync()
        elapsed = time.perf_counter() - started
        report.update(
            {
                "available": True,
                "stage": "matmul",
                "matmul_sync_seconds": elapsed,
                "sum": float(total.item()),
                "peak_memory_mb": _peak_memory_mb(),
            }
        )
    except Exception as error:  # noqa: BLE001 - a failed kernel is the answer
        report.update(
            {
                "available": False,
                "stage": "matmul",
                "error": f"{type(error).__name__}: {_short_error(error)}",
            }
        )
    finally:
        jt.flags.use_cuda = previous
        try:
            jt.sync_all(True)
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass
    return report


def _synthetic_batch(batch_size: int, num_points: int, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "obs": {
            "point_cloud": rng.normal(size=(batch_size, 2, num_points, 3)).astype(np.float32),
            "agent_pos": rng.normal(size=(batch_size, 2, 9)).astype(np.float32),
        },
        "action": rng.normal(size=(batch_size, 4, 4)).astype(np.float32),
    }


def run_full_channels(
    *, batch_size: int, num_points: int, steps: int, seed: int, device: str = "cpu"
) -> Dict[str, Any]:
    import jittor as jt

    from flowpolicy_jittor.model import FlowPolicy, LinearNormalizer
    from flowpolicy_jittor.training import Trainer

    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {device}")
    jt.flags.use_cuda = int(device == "cuda")
    shape_meta = {
        "obs": {
            "point_cloud": {"shape": [num_points, 3], "type": "point_cloud"},
            "agent_pos": {"shape": [9], "type": "low_dim"},
        },
        "action": {"shape": [4]},
    }
    batch = _synthetic_batch(batch_size, num_points, seed=seed)
    policy = FlowPolicy(
        shape_meta=shape_meta,
        horizon=4,
        n_action_steps=4,
        n_obs_steps=2,
        down_dims=FULL_CHANNELS,
        kernel_size=5,
        n_groups=8,
        diffusion_step_embed_dim=128,
        encoder_output_dim=64,
        state_mlp_size=(64, 64),
        pointcloud_encoder_cfg={
            "in_channels": 3,
            "out_channels": 64,
            "use_layernorm": True,
            "final_norm": "layernorm",
            "normal_channel": False,
        },
        var_fc_hidden_dim=512,
    )
    policy.set_normalizer(
        LinearNormalizer().fit(
            {
                "action": batch["action"],
                "agent_pos": batch["obs"]["agent_pos"],
                "point_cloud": batch["obs"]["point_cloud"],
            }
        )
    )
    parameter_count = int(sum(int(np.prod(parameter.shape)) for parameter in policy.trainable_parameters()))
    trainer = Trainer(policy, total_steps=max(1, steps), lr=1e-4, warmup_steps=0, seed=seed)

    losses: List[float] = []
    started = time.perf_counter()
    for _ in range(steps):
        record = trainer.train_step(batch)
        losses.append(float(record["loss"]))
    train_seconds = time.perf_counter() - started

    inference_started = time.perf_counter()
    prediction = policy.predict_action(batch["obs"], noise=None)
    prediction["action"].sync()
    inference_seconds = time.perf_counter() - inference_started
    action_shape = list(np.asarray(prediction["action"].numpy()).shape)

    return {
        "device": device,
        "down_dims": list(FULL_CHANNELS),
        "num_points": int(num_points),
        "batch_size": int(batch_size),
        "trainable_parameters": parameter_count,
        "steps": int(steps),
        "losses": losses,
        "finite": bool(all(np.isfinite(losses))),
        "train_seconds": train_seconds,
        "seconds_per_step": train_seconds / max(1, steps),
        "inference_seconds": inference_seconds,
        "action_shape": action_shape,
        "peak_memory_mb": _peak_memory_mb(),
        "nfe": int(policy.nfe),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="backend for the full-channel forward/backward run; the CUDA probe always runs",
    )
    parser.add_argument("--skip-full-channels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    import jittor as jt

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    cuda = probe_cuda()
    print(f"[cuda] available={cuda['available']} stage={cuda.get('stage')} "
          f"error={cuda.get('error', '-')}")
    full: Optional[Dict[str, Any]] = None
    if not args.skip_full_channels:
        if args.device == "cuda" and not cuda.get("available"):
            raise SystemExit(
                "--device cuda requested, but the real CUDA matmul probe failed: "
                + str(cuda.get("error", "unknown error"))
            )
        full = run_full_channels(
            batch_size=args.batch_size,
            num_points=args.num_points,
            steps=args.steps,
            seed=args.seed,
            device=args.device,
        )
        print(f"[full-channels] device={full['device']} "
              f"params={full['trainable_parameters']} batch={full['batch_size']} "
              f"points={full['num_points']} s/step={full['seconds_per_step']:.2f} "
              f"peak={full['peak_memory_mb']:.0f}MB action={full['action_shape']} "
              f"finite={full['finite']}")

    report = {
        "cuda": cuda,
        "cuda_verified": bool(cuda.get("available")),
        "cuda_note": (
            "CUDA kernels ran on this machine"
            if cuda.get("available")
            else "CUDA was probed and is NOT available here; every other result in this "
            "repository is CPU-only and must be described that way"
        ),
        "full_channels": full,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "jittor_version": str(jt.__version__),
        "prohibited_claims": [
            "CUDA verified (unless cuda_verified is true)",
            "full-channel training (one step is not training)",
            "paper-scale throughput",
        ],
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cuda_verified": report["cuda_verified"], "report": str(args.out / "report.json")}))
    if full is not None and not full["finite"]:
        print("full-channel run produced a non-finite loss")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
