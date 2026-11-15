"""Measure small-CPU overhead of the detached EMA target (FPJ-034).

Each mode runs in a fresh process so Jittor's compiled graph/cache from one mode does
not contaminate another.  This is a resource probe, not a speed benchmark or a paper
training result.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _rss_mib() -> float:
    status = Path("/proc/self/status")
    if not status.is_file():
        raise RuntimeError("resource probe requires Linux /proc/self/status")
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS is unavailable in /proc/self/status")


def _peak_mib() -> float:
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _batch(seed: int = 34):
    rng = np.random.default_rng(seed)
    return {
        "obs": {
            "point_cloud": rng.normal(size=(2, 2, 8, 3)).astype(np.float32),
            "agent_pos": rng.normal(size=(2, 2, 9)).astype(np.float32),
        },
        "action": rng.normal(size=(2, 4, 4)).astype(np.float32),
    }


def _worker(mode: str) -> dict:
    import jittor as jt

    from flowpolicy_jittor.model import FlowPolicy, LinearNormalizer
    from flowpolicy_jittor.training import Trainer

    jt.flags.use_cuda = 0
    jt.set_global_seed(34)
    data = _batch()
    normalizer = LinearNormalizer().fit(
        {
            "action": data["action"],
            "agent_pos": data["obs"]["agent_pos"],
            "point_cloud": data["obs"]["point_cloud"],
        }
    )
    target_mode = "detached" if mode == "detached_ema" else "code"
    ema_decay = None if mode == "code_no_ema" else 0.999
    rss_before = _rss_mib()
    peak_before = _peak_mib()
    started = time.perf_counter()
    policy = FlowPolicy(
        shape_meta={
            "obs": {
                "point_cloud": {"shape": [8, 3], "type": "point_cloud"},
                "agent_pos": {"shape": [9], "type": "low_dim"},
            },
            "action": {"shape": [4]},
        },
        horizon=4,
        n_action_steps=4,
        n_obs_steps=2,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16, 32),
        kernel_size=3,
        n_groups=4,
        encoder_output_dim=8,
        state_mlp_size=(8, 8),
        pointcloud_encoder_cfg={
            "in_channels": 3,
            "out_channels": 8,
            "use_layernorm": True,
            "final_norm": "layernorm",
            "normal_channel": False,
        },
        var_fc_hidden_dim=8,
        target_mode=target_mode,
    )
    policy.set_normalizer(normalizer)
    trainer = Trainer(
        policy,
        total_steps=8,
        lr=1e-3,
        warmup_steps=1,
        seed=34,
        ema_decay=ema_decay,
    )
    construction_seconds = time.perf_counter() - started
    rss_after_construction = _rss_mib()
    peak_after_construction = _peak_mib()

    # Jittor compiles lazily.  Exclude three warm-up updates so compilation time is
    # not mislabeled as steady-state target overhead.
    for _ in range(3):
        trainer.train_step(data)
        jt.sync_all()
    step_times = []
    for _ in range(5):
        step_started = time.perf_counter()
        trainer.train_step(data)
        jt.sync_all()
        step_times.append(time.perf_counter() - step_started)
    return {
        "mode": mode,
        "target_mode": target_mode,
        "ema_enabled": ema_decay is not None,
        "construction_seconds": construction_seconds,
        "rss_before_mib": rss_before,
        "rss_after_construction_mib": rss_after_construction,
        "construction_rss_delta_mib": rss_after_construction - rss_before,
        "construction_peak_delta_mib": peak_after_construction - peak_before,
        "step_seconds": step_times,
        "mean_step_seconds": float(np.mean(step_times)),
        "median_step_seconds": float(np.median(step_times)),
        "final_rss_mib": _rss_mib(),
        "final_peak_mib": _peak_mib(),
        "online_trainable_parameters": int(
            sum(np.prod(parameter.shape) for parameter in policy.trainable_parameters())
        ),
        "target_trainable_parameters": (
            None
            if not hasattr(trainer.ema, "target_policy")
            else len(trainer.ema.target_policy.trainable_parameters())
        ),
        "ema_optimization_step": (
            None if trainer.ema is None else trainer.ema.optimization_step
        ),
    }


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("code_no_ema", "code_stored_ema", "detached_ema"))
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "fpj034" / "detached_target_probe.json",
    )
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(_worker(args.worker)))
        return 0

    results = {}
    for mode in ("code_no_ema", "code_stored_ema", "detached_ema"):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", mode],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        results[mode] = result
    code = results["code_no_ema"]
    detached = results["detached_ema"]
    report = {
        "schema_version": 1,
        "scope": "small CPU resource probe; not a paper recipe or formal experiment",
        "config": {
            "batch_size": 2,
            "num_points": 8,
            "horizon": 4,
            "down_dims": [8, 16, 32],
            "compile_warmup_steps": 3,
            "measured_steps": 5,
            "seed": 34,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "results": results,
        "detached_minus_code_no_ema": {
            "construction_rss_delta_mib": (
                detached["construction_rss_delta_mib"]
                - code["construction_rss_delta_mib"]
            ),
            "mean_step_seconds": (
                detached["mean_step_seconds"] - code["mean_step_seconds"]
            ),
            "mean_step_ratio": (
                detached["mean_step_seconds"] / code["mean_step_seconds"]
            ),
            "median_step_seconds": (
                detached["median_step_seconds"] - code["median_step_seconds"]
            ),
            "median_step_ratio": (
                detached["median_step_seconds"] / code["median_step_seconds"]
            ),
        },
    }
    _write_json_atomic(args.out, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
