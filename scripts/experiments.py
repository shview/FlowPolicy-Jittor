"""Matched closed-loop experiments across tasks, seeds and target semantics.

One command trains and evaluates every (task, seed) cell and writes a single aggregate
report, so the numbers in the README/defence cannot come from a hand-picked run:

    python scripts/experiments.py \
        --tasks reach-wall-v3 --seeds 0 1 2 \
        --target-modes code detached --ema-decay 0.999 \
        --episodes 20 --epochs 10 --out outputs/fpj035

Design notes:

* each (target, task, seed) cell is a real subprocess run of ``scripts/train_smoke.py`` and
  ``scripts/rollout.py``; the report records the exact command line, the checkpoint and
  the per-episode outcome, so any cell can be re-run on its own;
* existing cells are reused only with ``--reuse`` after checking their provenance;
* ``--resume`` continues each incomplete cell from its latest trainer checkpoint;
* training cells may run concurrently (``--parallel``), rollouts stay sequential because
  they drive the isolated simulator process;
* the aggregate reports mean and sample standard deviation across seeds, plus latency,
  NFE, step cap and the failure reasons -- never the best single run.
"""

from __future__ import annotations

import argparse
import hashlib
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


def train_command(
    args,
    task: str,
    seed: int,
    run_dir: Path,
    *,
    target_mode: str = "code",
    resume_checkpoint: Optional[Path] = None,
) -> List[str]:
    command = [
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
        "--target-mode", target_mode,
        "--run-label", args.run_label,
        "--out", str(run_dir / "train"),
    ]
    checkpoint_every = int(getattr(args, "checkpoint_every", 0))
    if checkpoint_every > 0:
        command.extend(["--checkpoint-every", str(checkpoint_every)])
    ema_decay = getattr(args, "ema_decay", None)
    if ema_decay is not None:
        command.extend(["--ema-decay", str(ema_decay)])
        command.extend(["--ema-warmup-steps", str(getattr(args, "ema_warmup_steps", 0))])
    if resume_checkpoint is not None:
        command.extend(["--resume", str(resume_checkpoint)])
    elif getattr(args, "overwrite", False):
        command.append("--overwrite")
    return command


def rollout_command(
    args, task: str, seed: int, run_dir: Path, *, target_mode: str = "code"
) -> List[str]:
    command = [
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
        "--target-mode", target_mode,
        "--out", str(run_dir / "rollout"),
    ]
    if getattr(args, "overwrite", False):
        command.append("--overwrite")
    return command


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


def _cell_dir(root: Path, target_mode: str, task: str, seed: int) -> Path:
    return root / target_mode / task / f"seed-{seed}"


def _latest_resume_checkpoint(run_dir: Path) -> Optional[Path]:
    progress_path = run_dir / "train" / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        latest = progress.get("latest_checkpoint")
        if isinstance(latest, dict) and isinstance(latest.get("path"), str):
            candidate = Path(latest["path"])
            if candidate.is_file():
                return candidate
    candidates = sorted((run_dir / "train" / "checkpoints").glob("epoch-*.ckpt"))
    return candidates[-1] if candidates else None


def _load_train_summary(record: Dict[str, Any], summary_path: Path, *, reused: bool) -> None:
    if not summary_path.is_file():
        raise ValueError(f"training summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record.setdefault("train", {})
    record["train"].update(
        {
            "reused": reused,
            "wall_seconds": summary["wall_seconds"],
            "seconds_per_step": summary["seconds_per_step"],
            "loss": summary["loss"],
            "train_windows": summary["train_windows"],
            "val_windows": summary["val_windows"],
            "steps_per_epoch": summary["steps_per_epoch"],
            "total_steps": summary["total_steps"],
            "trainable_parameters": summary["trainable_parameters"],
            "data_sha256": summary["data_sha256"],
            "config": summary["config"],
            "config_sha256": summary["config_sha256"],
            "checkpoint": summary["checkpoint"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "trainer_checkpoint_sha256": summary["trainer_checkpoint_sha256"],
            "runtime": summary["runtime"],
            "ema": summary["ema"],
            "resume": summary["resume"],
        }
    )


def _validate_reused_train(args, record: Dict[str, Any]) -> None:
    config = record["train"]["config"]
    expected = {
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "down_dims": list(args.down_dims),
        "embed_dim": args.embed_dim,
        "encoder_out": args.encoder_out,
        "pointnet_out": args.pointnet_out,
        "kernel_size": args.kernel_size,
        "n_groups": args.n_groups,
        "var_fc_hidden_dim": args.var_fc_hidden_dim,
        "lr": args.lr,
        "warmup": args.warmup,
        "num_points": args.num_points,
        "seed": record["seed"],
        "target_mode": record["target_mode"],
        "ema_decay": args.ema_decay,
        "ema_warmup_steps": args.ema_warmup_steps,
    }
    mismatches = {
        key: {"stored": config.get(key), "requested": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"reused training provenance mismatch: {mismatches}")


def train_cell(args, target_mode: str, task: str, seed: int) -> Dict[str, Any]:
    """Train one cell (or reuse its checkpoint); no rollout."""
    run_dir = _cell_dir(args.out, target_mode, task, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.threads)
    env.setdefault("MUJOCO_GL", "egl")
    record: Dict[str, Any] = {
        "target_mode": target_mode,
        "task": task,
        "seed": seed,
        "dir": str(run_dir),
    }

    checkpoint = run_dir / "train" / "policy.ckpt"
    if checkpoint.exists() and (args.reuse or args.resume):
        _load_train_summary(record, run_dir / "train" / "summary.json", reused=True)
        _validate_reused_train(args, record)
        _log(f"[{target_mode} {task} seed {seed}] using completed {checkpoint}")
        return record

    resume_checkpoint = _latest_resume_checkpoint(run_dir) if args.resume else None
    _log(
        f"[{target_mode} {task} seed {seed}] "
        f"{'resuming' if resume_checkpoint else 'training'} for {args.epochs} epochs"
    )
    record["train"] = _run(
        train_command(
            args,
            task,
            seed,
            run_dir,
            target_mode=target_mode,
            resume_checkpoint=resume_checkpoint,
        ),
        run_dir / "train.log",
        env,
    )
    record["train"]["reused"] = False
    if record["train"]["exit_code"] != 0:
        record["status"] = "train_failed"
        return record
    _load_train_summary(record, run_dir / "train" / "summary.json", reused=False)
    return record


def rollout_cell(args, record: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one trained cell; returns the completed record."""
    target_mode, task, seed = record["target_mode"], record["task"], record["seed"]
    run_dir = Path(record["dir"])
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.threads)
    env.setdefault("MUJOCO_GL", "egl")

    report_path = run_dir / "rollout" / "rollout.json"
    if report_path.is_file() and (args.reuse or args.resume):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = {
            "task": task,
            "episode_count": args.episodes,
            "num_points": args.num_points,
            "target_mode": target_mode,
        }
        stored = {
            "task": report.get("task"),
            "episode_count": report.get("episode_count"),
            "num_points": report.get("num_points"),
            "target_mode": report.get("model_config", {}).get("target_mode"),
        }
        if stored != expected:
            raise ValueError(
                f"reused rollout provenance mismatch: stored={stored}, requested={expected}"
            )
        record["rollout"] = {"reused": True, "report": str(report_path)}
        _log(f"[{target_mode} {task} seed {seed}] using completed rollout")
    else:
        _log(f"[{target_mode} {task} seed {seed}] rolling out {args.episodes} episodes")
        record["rollout"] = _run(
            rollout_command(args, task, seed, run_dir, target_mode=target_mode),
            run_dir / "rollout.log",
            env,
        )
        record["rollout"]["reused"] = False
        if record["rollout"]["exit_code"] != 0:
            record["status"] = "rollout_failed"
            return record
        report = json.loads(report_path.read_text(encoding="utf-8"))
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
            "episode_outcomes": [
                {
                    "episode": index,
                    "success": bool(episode["success"]),
                    "steps": int(episode["steps"]),
                    "success_steps": int(episode["success_steps"]),
                    "done_reason": episode["done_reason"],
                    "policy_latency_ms": episode.get("policy_latency_ms"),
                }
                for index, episode in enumerate(episodes)
            ],
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


def run_cell(args, target_mode: str, task: str, seed: int) -> Dict[str, Any]:
    record = train_cell(args, target_mode, task, seed)
    if record.get("status") == "train_failed":
        return record
    return rollout_cell(args, record)


def _count(values) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _mean_std(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else None,
    }


def _aggregate_tasks(cells: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_task: Dict[str, Dict[str, Any]] = {}
    for task in sorted({cell["task"] for cell in cells}):
        task_cells = [cell for cell in cells if cell["task"] == task]
        ok = [cell for cell in task_cells if cell.get("status") == "ok"]
        rates = [cell["success_rate"] for cell in ok]
        latencies = [cell["latency_ms_mean"] for cell in ok if cell["latency_ms_mean"] is not None]
        seconds_per_step = [
            cell["train"]["seconds_per_step"]
            for cell in ok
            if cell.get("train", {}).get("seconds_per_step") is not None
        ]
        last_losses = [
            cell["train"]["loss"]["last_epoch_mean"]
            for cell in ok
            if cell.get("train", {}).get("loss", {}).get("last_epoch_mean") is not None
        ]
        val_losses = [
            cell["train"]["loss"]["val"]
            for cell in ok
            if cell.get("train", {}).get("loss", {}).get("val") is not None
        ]
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
            "train_seconds_per_step": _mean_std(seconds_per_step),
            "train_last_epoch_loss": _mean_std(last_losses),
            "validation_loss": _mean_std(val_losses),
        }
    return by_task


def _aggregate(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    target_modes = sorted({cell.get("target_mode", "code") for cell in cells})
    targets = {
        mode: {
            "tasks": _aggregate_tasks(
                [cell for cell in cells if cell.get("target_mode", "code") == mode]
            )
        }
        for mode in target_modes
    }
    report: Dict[str, Any] = {
        "target_modes": target_modes,
        "targets": targets,
        "cells": cells,
    }
    # Preserve the FPJ-029 single-mode report API for existing consumers.
    if len(target_modes) == 1:
        report["tasks"] = targets[target_modes[0]]["tasks"]
    return report


def _merge_counts(counts_iter) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for counts in counts_iter:
        for key, value in counts.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    path = Path(path)
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _preflight(args) -> Dict[str, Any]:
    shared = {
        "tasks": list(args.tasks),
        "data_sha256": {
            task: _sha256_tree(data_path(task, args.data_dir)) for task in args.tasks
        },
        "seeds": list(args.seeds),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup": args.warmup,
        "checkpoint_every": args.checkpoint_every,
        "down_dims": list(args.down_dims),
        "embed_dim": args.embed_dim,
        "encoder_out": args.encoder_out,
        "pointnet_out": args.pointnet_out,
        "kernel_size": args.kernel_size,
        "n_groups": args.n_groups,
        "var_fc_hidden_dim": args.var_fc_hidden_dim,
        "num_points": args.num_points,
        "device": args.device,
        "threads": args.threads,
        "ema_decay": args.ema_decay,
        "ema_warmup_steps": args.ema_warmup_steps,
        "rollout_noise_seed_rule": "same as training seed",
        "action_slice_mode": "code",
        "nfe": 1,
    }
    return {
        "passed": True,
        "shared_config": shared,
        "shared_config_sha256": _canonical_sha(shared),
        "arms": {
            mode: {
                "target_mode": mode,
                "ema_role": (
                    "detached_loss_target"
                    if mode == "detached"
                    else (
                        "stored_shadow_not_used_by_loss"
                        if args.ema_decay is not None
                        else "none"
                    )
                ),
            }
            for mode in args.target_modes
        },
        "intended_differences": ["target_mode", "ema_role/target implementation"],
    }


def _matched_validation(cells: List[Dict[str, Any]], target_modes: Sequence[str]) -> Dict[str, Any]:
    if len(target_modes) < 2:
        return {"applicable": False, "passed": True, "pairs": []}
    pairs = []
    passed = True
    indexed = {
        (cell.get("target_mode", "code"), cell["task"], cell["seed"]): cell
        for cell in cells
    }
    tasks_and_seeds = sorted({(cell["task"], cell["seed"]) for cell in cells})
    base_mode = target_modes[0]
    for task, seed in tasks_and_seeds:
        base = indexed.get((base_mode, task, seed))
        pair = {"task": task, "seed": seed, "base_mode": base_mode, "checks": {}}
        for mode in target_modes[1:]:
            other = indexed.get((mode, task, seed))
            label = f"{base_mode}_vs_{mode}"
            if not base or not other or base.get("status") != "ok" or other.get("status") != "ok":
                pair["checks"][label] = {
                    "passed": False,
                    "reason": "both matched cells must finish successfully",
                }
                passed = False
                continue
            base_config = dict(base["train"]["config"])
            other_config = dict(other["train"]["config"])
            base_target = base_config.pop("target_mode", None)
            other_target = other_config.pop("target_mode", None)
            protocol_keys = ("nfe", "num_points", "action_slice_mode")
            protocol_complete = all(
                key in base and key in other for key in protocol_keys
            )
            checks = {
                "target_mode_differs": base_target != other_target,
                "training_config_except_target_equal": base_config == other_config,
                "data_sha256_equal": base["train"]["data_sha256"] == other["train"]["data_sha256"],
                "train_windows_equal": base["train"]["train_windows"] == other["train"]["train_windows"],
                "total_steps_equal": base["train"]["total_steps"] == other["train"]["total_steps"],
                "episode_count_equal": base["episode_count"] == other["episode_count"],
                "rollout_protocol_equal": protocol_complete
                and tuple(base[key] for key in protocol_keys)
                == tuple(other[key] for key in protocol_keys),
            }
            check_passed = all(checks.values())
            pair["checks"][label] = {
                "passed": check_passed,
                "details": checks,
                "shared_training_config_sha256": _canonical_sha(base_config),
            }
            passed = passed and check_passed
        pairs.append(pair)
    return {"applicable": True, "passed": passed, "pairs": pairs}


def _markdown(report: Dict[str, Any], args) -> str:
    device = getattr(args, "device", "cpu")
    device_boundary = (
        "- CPU only: CUDA was not used for this grid, so no GPU throughput claim is made."
        if device == "cpu"
        else "- CUDA GPU run: the exact GPU, driver and checkpoint hashes are stored per training cell."
    )
    lines = [
        "# Matched target-semantics experiment (FPJ-035)",
        "",
        f"- target modes: {', '.join(report['target_modes'])}",
        f"- tasks: {', '.join(args.tasks)}",
        f"- seeds per task: {args.seeds}",
        f"- episodes per seed: {args.episodes}",
        f"- training budget: {args.epochs} epochs, batch {args.batch_size}, down_dims {args.down_dims}, device {device}",
        f"- rollout step cap: {args.max_steps}, NFE 1, action_slice_mode `code`",
        "",
        f"- matched preflight SHA-256: `{report['preflight']['shared_config_sha256']}`",
        "",
        "| target | task | seeds ok | success mean ± sample std | val loss mean ± sample std | train ms/step |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for mode in report["target_modes"]:
        for task, entry in report["targets"][mode]["tasks"].items():
            mean = entry["success_rate_mean"]
            std = entry["success_rate_std"]
            success_text = "n/a" if mean is None else f"{mean:.2%}"
            if std is not None:
                success_text += f" ± {std:.2%}"
            val = entry["validation_loss"]
            val_text = "n/a" if val["mean"] is None else f"{val['mean']:.3g}"
            if val["std"] is not None:
                val_text += f" ± {val['std']:.3g}"
            timing = entry["train_seconds_per_step"]
            timing_text = "n/a" if timing["mean"] is None else f"{timing['mean'] * 1000:.2f}"
            lines.append(
                f"| {mode} | {task} | {entry['cells_ok']}/{len(entry['seeds'])} | "
                f"{success_text} | {val_text} | {timing_text} |"
            )
    lines += [
        "",
        "## Per-seed cells",
        "",
        "| target | task | seed | status | success | val loss | train ms/step |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in report["cells"]:
        if cell.get("status") == "ok":
            lines.append(
                f"| {cell['target_mode']} | {cell['task']} | {cell['seed']} | ok | "
                f"{cell['success_episodes']}/{cell['episode_count']} | "
                f"{cell['train']['loss']['val']:.3g} | "
                f"{cell['train']['seconds_per_step'] * 1000:.2f} |"
            )
        else:
            lines.append(
                f"| {cell.get('target_mode')} | {cell['task']} | {cell['seed']} | "
                f"{cell.get('status')} | - | - | - |"
            )
    lines += [
        "",
        "## What this does not mean",
        "",
        device_boundary,
        "- This comparison isolates target semantics only within the declared reduced setup.",
        "- Matching the paper architecture is not the paper's recipe; batch, epochs, data and simulator must be read from the report.",
        "- The simulator is a documented substitute (D016); success rates cannot be compared to the paper's numbers.",
        "- Ten to twenty demonstrations overfit quickly; a low rate is a data/scale statement, not a code verdict.",
        "",
    ]
    return "\n".join(lines)


def _train_wave(
    args, target_mode: str, task: str, seeds: Sequence[int]
) -> List[Dict[str, Any]]:
    """Train several seeds concurrently (the CPU is the bottleneck), roll them out in turn.

    Rollouts stay sequential on purpose: they drive the isolated simulator process and
    their latency numbers must not be distorted by CPU contention.
    """
    import concurrent.futures

    records: Dict[int, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(train_cell, args, target_mode, task, seed): seed for seed in seeds
        }
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            records[seed] = future.result()
            _log(
                f"[{target_mode} {task} seed {seed}] "
                f"train {records[seed].get('status', 'ok')}"
            )
    ordered = [records[seed] for seed in seeds]
    return [rollout_cell(args, record) if record.get("status") != "train_failed" else record for record in ordered]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--target-modes",
        nargs="+",
        choices=("code", "detached"),
        default=["code"],
        help="one mode for a regular grid, or 'code detached' for a matched comparison",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="shared EMA decay; required when detached is selected (paper value: 0.999)",
    )
    parser.add_argument("--ema-warmup-steps", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        metavar="EPOCHS",
        help="periodic trainer checkpoint interval passed to each training cell",
    )
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume incomplete cells from their latest periodic trainer checkpoint",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint-every must be non-negative")
    if len(set(args.target_modes)) != len(args.target_modes):
        raise SystemExit("--target-modes must be unique")
    if args.ema_decay is not None and not 0.0 < args.ema_decay < 1.0:
        raise SystemExit("--ema-decay must lie strictly between 0 and 1")
    if "detached" in args.target_modes and args.ema_decay is None:
        raise SystemExit("detached target comparison requires --ema-decay")
    if args.ema_warmup_steps < 0:
        raise SystemExit("--ema-warmup-steps must be non-negative")
    if args.ema_decay is None and args.ema_warmup_steps != 0:
        raise SystemExit("--ema-warmup-steps requires --ema-decay")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite cannot be used together")
    if args.resume and args.reuse:
        raise SystemExit("--resume and --reuse cannot be used together")
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

    preflight = _preflight(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    _log(
        f"grid: {len(args.target_modes)} targets x {len(args.tasks)} tasks x "
        f"{len(args.seeds)} seeds x {args.episodes} episodes"
    )
    if args.parallel < 1:
        raise SystemExit("--parallel must be at least 1")
    if args.device == "cuda" and args.parallel != 1:
        raise SystemExit("CUDA experiment grids must use --parallel 1 to avoid sharing one GPU across seeds")
    cells: List[Dict[str, Any]] = []
    for target_mode in args.target_modes:
        for task in args.tasks:
            if args.parallel > 1 and len(args.seeds) > 1:
                task_cells = _train_wave(args, target_mode, task, args.seeds)
            else:
                task_cells = [
                    run_cell(args, target_mode, task, seed) for seed in args.seeds
                ]
            for cell in task_cells:
                cells.append(cell)
                status = cell.get("status")
                _log(f"[{target_mode} {task} seed {cell['seed']}] {status}")
                if status != "ok":
                    _log(
                        f"[{target_mode} {task} seed {cell['seed']}] see {cell['dir']}"
                    )

    report = _aggregate(cells)
    report["preflight"] = preflight
    report["matched_validation"] = _matched_validation(cells, args.target_modes)
    report["config"] = {
        "target_modes": args.target_modes,
        "ema_decay": args.ema_decay,
        "ema_warmup_steps": args.ema_warmup_steps,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "checkpoint_every": args.checkpoint_every,
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
        "small-channel CPU comparison predicts full-channel GPU behavior",
    ]
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out / "SUMMARY.md").write_text(_markdown(report, args), encoding="utf-8")
    for mode in report["target_modes"]:
        for task, entry in report["targets"][mode]["tasks"].items():
            mean = entry["success_rate_mean"]
            text = "n/a" if mean is None else f"{mean:.2%}"
            _log(
                f"{mode}/{task}: {text} over "
                f"{entry['cells_ok']}/{len(entry['seeds'])} seeds"
            )
    _log(f"report: {report_path}")
    complete = all(cell.get("status") == "ok" for cell in cells)
    return 0 if complete and report["matched_validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
