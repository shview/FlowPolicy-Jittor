"""Formal closed-loop experiments: several tasks, several seeds, twenty episodes each (FPJ-029).

One command trains and evaluates every (task, seed) cell and writes a single aggregate
report, so the numbers in the README/defence cannot come from a hand-picked run:

    python scripts/experiments.py \
        --tasks reach-v3 button-press-wall-v3 \
        --seeds 0 1 2 --episodes 20 --epochs 30 \
        --out outputs/experiments

Design notes:

* each cell is a real subprocess run of ``scripts/train_smoke.py`` and
  ``scripts/rollout.py``; the report records the exact command line, the checkpoint and
  the per-episode outcome, so any cell can be re-run on its own;
* an existing cell is reused only with ``--reuse`` (and the reuse is recorded);
* training cells may run concurrently (``--parallel``), rollouts stay sequential because
  they drive the isolated simulator process;
* the aggregate reports mean and sample standard deviation across seeds, plus latency,
  NFE, step cap and the failure reasons -- never the best single run.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM_PYTHON = "/home/s/.local/share/flowpolicy/sim-env/bin/python"


def _log(message: str) -> None:
    print(message, flush=True)


def data_path(task: str, data_dir: Path) -> Path:
    return data_dir / f"{task}.zarr"


def train_command(args, task: str, seed: int, run_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/train_smoke.py"),
        "--data", str(data_path(task, args.data_dir)),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--warmup", str(args.warmup),
        "--seed", str(seed),
        "--down-dims", *[str(dim) for dim in args.down_dims],
        "--embed-dim", str(args.embed_dim),
        "--encoder-out", str(args.encoder_out),
        "--pointnet-out", str(args.pointnet_out),
        "--kernel-size", str(args.kernel_size),
        "--n-groups", str(args.n_groups),
        "--var-fc-hidden-dim", str(args.var_fc_hidden_dim),
        "--num-points", str(args.num_points),
        "--device", args.device,
        "--run-label", args.run_label,
        "--out", str(run_dir / "train"),
    ]


def rollout_command(args, task: str, seed: int, run_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/rollout.py"),
        "--data", str(data_path(task, args.data_dir)),
        "--checkpoint", str(run_dir / "train" / "policy.ckpt"),
        "--episodes", str(args.episodes),
        "--max-steps", str(args.max_steps),
        "--seed", str(seed),
        "--noise-seed", str(seed),
        "--task", task,
        "--sim-python", args.sim_python,
        "--down-dims", *[str(dim) for dim in args.down_dims],
        "--embed-dim", str(args.embed_dim),
        "--encoder-out", str(args.encoder_out),
        "--pointnet-out", str(args.pointnet_out),
        "--kernel-size", str(args.kernel_size),
        "--n-groups", str(args.n_groups),
        "--var-fc-hidden-dim", str(args.var_fc_hidden_dim),
        "--num-points", str(args.num_points),
        "--device", args.device,
        "--out", str(run_dir / "rollout"),
    ]


def _run(command: Sequence[str], log_path: Path, env: Dict[str, str]) -> Dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(list(command), capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\n" + completed.stdout + "\n[stderr]\n" + completed.stderr,
        encoding="utf-8",
    )
    return {
        "command": list(command),
        "exit_code": completed.returncode,
        "seconds": elapsed,
        "log": str(log_path),
    }


def _cell_dir(root: Path, task: str, seed: int) -> Path:
    return root / task / f"seed-{seed}"


def train_cell(args, task: str, seed: int) -> Dict[str, Any]:
    """Train one cell (or reuse its checkpoint); no rollout."""
    run_dir = _cell_dir(args.out, task, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.threads)
    env.setdefault("MUJOCO_GL", "egl")
    record: Dict[str, Any] = {"task": task, "seed": seed, "dir": str(run_dir)}

    checkpoint = run_dir / "train" / "policy.ckpt"
    if checkpoint.exists() and args.reuse:
        record["train"] = {"reused": True, "checkpoint": str(checkpoint)}
        _log(f"[{task} seed {seed}] reusing {checkpoint}")
        return record

    _log(f"[{task} seed {seed}] training for {args.epochs} epochs")
    record["train"] = _run(train_command(args, task, seed, run_dir), run_dir / "train.log", env)
    record["train"]["reused"] = False
    if record["train"]["exit_code"] != 0:
        record["status"] = "train_failed"
        return record
    summary_path = run_dir / "train" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        record["train"].update(
            {
                "wall_seconds": summary["wall_seconds"],
                "seconds_per_step": summary["seconds_per_step"],
                "loss": summary["loss"],
                "train_windows": summary["train_windows"],
            }
        )
    return record


def rollout_cell(args, record: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one trained cell; returns the completed record."""
    task, seed = record["task"], record["seed"]
    run_dir = Path(record["dir"])
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.threads)
    env.setdefault("MUJOCO_GL", "egl")

    _log(f"[{task} seed {seed}] rolling out {args.episodes} episodes")
    record["rollout"] = _run(rollout_command(args, task, seed, run_dir), run_dir / "rollout.log", env)
    if record["rollout"]["exit_code"] != 0:
        record["status"] = "rollout_failed"
        return record
    report = json.loads((run_dir / "rollout" / "rollout.json").read_text())
    episodes = report["episodes"]
    successes = [episode for episode in episodes if episode["success"]]
    latencies = [episode["policy_latency_ms"]["mean"] for episode in episodes if episode.get("policy_latency_ms")]
    record.update(
        {
            "status": "ok",
            "episode_count": int(report["episode_count"]),
            "success_episodes": int(report["success_episodes"]),
            "success_rate": float(report["success_episodes"]) / max(1, int(report["episode_count"])),
            "success_steps": [episode["success_steps"] for episode in episodes],
            "success_step_counts": [episode["steps"] for episode in successes],
            "done_reasons": _count(episode["done_reason"] for episode in episodes),
            "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
            "total_steps": int(report["total_steps"]),
            "wall_seconds": report["wall_seconds"],
            "nfe": report["nfe"],
            "num_points": report["num_points"],
            "action_slice_mode": report["action_slice_mode"],
            "checkpoint_reload_max_abs_delta": report["checkpoint_reload_max_abs_delta"],
        }
    )
    return record


def run_cell(args, task: str, seed: int) -> Dict[str, Any]:
    record = train_cell(args, task, seed)
    if record.get("status") == "train_failed":
        return record
    return rollout_cell(args, record)


def _count(values) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _aggregate(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, Dict[str, Any]] = {}
    for task in sorted({cell["task"] for cell in cells}):
        task_cells = [cell for cell in cells if cell["task"] == task]
        ok = [cell for cell in task_cells if cell.get("status") == "ok"]
        rates = [cell["success_rate"] for cell in ok]
        latencies = [cell["latency_ms_mean"] for cell in ok if cell["latency_ms_mean"] is not None]
        by_task[task] = {
            "seeds": [cell["seed"] for cell in task_cells],
            "cells_ok": len(ok),
            "cells_failed": len(task_cells) - len(ok),
            "episodes_per_seed": sorted({cell.get("episode_count") for cell in ok}),
            "success_rate_per_seed": rates,
            "success_rate_mean": statistics.fmean(rates) if rates else None,
            "success_rate_std": statistics.stdev(rates) if len(rates) > 1 else None,
            "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
            "done_reasons": _merge_counts(cell.get("done_reasons", {}) for cell in ok),
            "nfe": sorted({cell.get("nfe") for cell in ok}),
            "num_points": sorted({cell.get("num_points") for cell in ok}),
        }
    return {"tasks": by_task, "cells": cells}


def _merge_counts(counts_iter) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for counts in counts_iter:
        for key, value in counts.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def _markdown(report: Dict[str, Any], args) -> str:
    device = getattr(args, "device", "cpu")
    device_boundary = (
        "- CPU only: CUDA was not used for this grid, so no GPU throughput claim is made."
        if device == "cpu"
        else "- CUDA GPU run: the exact GPU, driver and checkpoint hashes are stored per training cell."
    )
    lines = [
        "# Formal experiment report (FPJ-029)",
        "",
        f"- tasks: {', '.join(sorted(report['tasks']))}",
        f"- seeds per task: {args.seeds}",
        f"- episodes per seed: {args.episodes}",
        f"- training budget: {args.epochs} epochs, batch {args.batch_size}, down_dims {args.down_dims}, device {device}",
        f"- rollout step cap: {args.max_steps}, NFE 1, action_slice_mode `code`",
        "",
        "| task | seeds ok | success rate (mean ± std) | latency ms | done reasons |",
        "| --- | --- | --- | --- | --- |",
    ]
    for task, entry in report["tasks"].items():
        mean = entry["success_rate_mean"]
        std = entry["success_rate_std"]
        mean_text = "n/a" if mean is None else f"{mean:.2%}"
        std_text = "" if std is None else f" ± {std:.2%}"
        latency = "n/a" if entry["latency_ms_mean"] is None else f"{entry['latency_ms_mean']:.2f}"
        reasons = ", ".join(f"{key}={value}" for key, value in sorted(entry["done_reasons"].items())) or "n/a"
        lines.append(
            f"| {task} | {entry['cells_ok']}/{len(entry['seeds'])} | {mean_text}{std_text} | {latency} | {reasons} |"
        )
    lines += [
        "",
        "## Per-seed cells",
        "",
        "| task | seed | status | success | episodes | success steps | reload delta |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in report["cells"]:
        if cell.get("status") == "ok":
            steps = ", ".join(str(value) for value in cell["success_steps"])
            lines.append(
                f"| {cell['task']} | {cell['seed']} | ok | {cell['success_episodes']}/{cell['episode_count']} "
                f"| {cell['episode_count']} | {steps} | {cell['checkpoint_reload_max_abs_delta']:.3g} |"
            )
        else:
            lines.append(f"| {cell['task']} | {cell['seed']} | {cell.get('status')} | - | - | - | - |")
    lines += [
        "",
        "## What this does not mean",
        "",
        device_boundary,
        "- Matching the paper architecture is not the paper's recipe; batch, epochs, data and simulator must be read from the report.",
        "- The simulator is a documented substitute (D016); success rates cannot be compared to the paper's numbers.",
        "- Ten to twenty demonstrations overfit quickly; a low rate is a data/scale statement, not a code verdict.",
        "",
    ]
    return "\n".join(lines)


def _train_wave(args, task: str, seeds: Sequence[int]) -> List[Dict[str, Any]]:
    """Train several seeds concurrently (the CPU is the bottleneck), roll them out in turn.

    Rollouts stay sequential on purpose: they drive the isolated simulator process and
    their latency numbers must not be distorted by CPU contention.
    """
    import concurrent.futures

    records: Dict[int, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(train_cell, args, task, seed): seed for seed in seeds}
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            records[seed] = future.result()
            _log(f"[{task} seed {seed}] train {records[seed].get('status', 'ok')}")
    ordered = [records[seed] for seed in seeds]
    return [rollout_cell(args, record) if record.get("status") != "train_failed" else record for record in ordered]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--down-dims", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--encoder-out", type=int, default=64)
    parser.add_argument("--pointnet-out", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--var-fc-hidden-dim", type=int, default=64)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--run-label", default="experiment-grid")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="train this many seeds of one task concurrently (rollouts stay sequential)",
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    if not args.seeds:
        raise SystemExit("--seeds must not be empty")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds must be unique")
    missing = [task for task in args.tasks if not data_path(task, args.data_dir).exists()]
    if missing:
        raise SystemExit(
            "missing demonstration stores for: " + ", ".join(missing)
            + " (record them with scripts/record_demos.py first)"
        )
    report_path = args.out / "experiments.json"
    if report_path.exists() and not args.overwrite and not args.reuse:
        raise SystemExit(f"refusing to overwrite existing report: {report_path}")

    _log(f"grid: {len(args.tasks)} tasks x {len(args.seeds)} seeds x {args.episodes} episodes")
    if args.parallel < 1:
        raise SystemExit("--parallel must be at least 1")
    if args.device == "cuda" and args.parallel != 1:
        raise SystemExit("CUDA experiment grids must use --parallel 1 to avoid sharing one GPU across seeds")
    cells: List[Dict[str, Any]] = []
    for task in args.tasks:
        if args.parallel > 1 and len(args.seeds) > 1:
            task_cells = _train_wave(args, task, args.seeds)
        else:
            task_cells = [run_cell(args, task, seed) for seed in args.seeds]
        for cell in task_cells:
            cells.append(cell)
            status = cell.get("status")
            _log(f"[{task} seed {cell['seed']}] {status}")
            if status != "ok":
                _log(f"[{task} seed {cell['seed']}] see {cell['dir']}")

    report = _aggregate(cells)
    report["config"] = {
        "tasks": args.tasks,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "down_dims": args.down_dims,
        "embed_dim": args.embed_dim,
        "encoder_out": args.encoder_out,
        "pointnet_out": args.pointnet_out,
        "kernel_size": args.kernel_size,
        "n_groups": args.n_groups,
        "var_fc_hidden_dim": args.var_fc_hidden_dim,
        "num_points": args.num_points,
        "device": args.device,
        "run_label": args.run_label,
        "threads": args.threads,
        "data_dir": str(args.data_dir),
    }
    report["prohibited_claims"] = [
        "paper success rate reproduced",
        "the best cell is the result",
        "paper architecture equals the full paper training recipe",
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out / "SUMMARY.md").write_text(_markdown(report, args), encoding="utf-8")
    for task, entry in report["tasks"].items():
        mean = entry["success_rate_mean"]
        text = "n/a" if mean is None else f"{mean:.2%}"
        _log(f"{task}: {text} over {entry['cells_ok']}/{len(entry['seeds'])} seeds")
    _log(f"report: {report_path}")
    return 0 if all(cell.get("status") == "ok" for cell in cells) else 1


if __name__ == "__main__":
    raise SystemExit(main())
