"""Probe a MetaWorld scripted expert across task variants before data collection.

This is a cheap gate: it runs the simulator and expert policy without rendering point
clouds or writing demonstrations.  A formal collection should not start from a single
easy variant, so the default requires all ten probed variants to reach success for at
least five environment steps.

Example (run in the isolated simulator environment)::

    MUJOCO_GL=egl python scripts/task_probe.py --task reach-wall-v3 \
        --variants 10 --output outputs/fpj033/reach-wall-expert-probe.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def expert_policy_name(task: str) -> str:
    """Map a MetaWorld v3 task name to its scripted-policy class name."""

    base = task
    for suffix in ("-v3", "-v2"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base == "peg-insert-side":
        return "SawyerPegInsertionSideV3Policy"
    return "Sawyer" + "".join(part.capitalize() for part in base.split("-")) + "V3Policy"


def summarize_records(
    records: Sequence[Dict[str, Any]],
    *,
    min_success_steps: int,
    min_success_rate: float,
) -> Dict[str, Any]:
    """Aggregate episode records and make the suitability threshold explicit."""

    if not records:
        raise ValueError("at least one task-variant record is required")
    qualified = [record for record in records if int(record.get("success_steps", 0)) >= min_success_steps]
    reached = [record for record in records if record.get("first_success_step") is not None]
    errors = [record for record in records if record.get("error")]
    total = len(records)
    qualified_rate = len(qualified) / total
    first_steps = [int(record["first_success_step"]) for record in reached]
    return {
        "variants_probed": total,
        "variants_reached_success": len(reached),
        "variants_qualified": len(qualified),
        "expert_success_rate": len(reached) / total,
        "qualified_rate": qualified_rate,
        "min_success_steps": int(min_success_steps),
        "min_success_rate": float(min_success_rate),
        "first_success_step_min": min(first_steps) if first_steps else None,
        "first_success_step_max": max(first_steps) if first_steps else None,
        "first_success_step_mean": float(np.mean(first_steps)) if first_steps else None,
        "errors": len(errors),
        "suitable_for_collection": bool(not errors and qualified_rate >= min_success_rate),
    }


def probe_task(
    task: str,
    *,
    variants: int,
    max_steps: int,
    seed: int,
    min_success_steps: int,
    min_success_rate: float,
) -> Dict[str, Any]:
    """Run the real scripted expert over distinct MT1 task variants."""

    if variants <= 0:
        raise ValueError("variants must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if min_success_steps <= 0 or min_success_steps > max_steps:
        raise ValueError("min_success_steps must be in [1, max_steps]")
    if not 0.0 < min_success_rate <= 1.0:
        raise ValueError("min_success_rate must be in (0, 1]")

    import metaworld
    import metaworld.policies as policies

    policy_name = expert_policy_name(task)
    if not hasattr(policies, policy_name):
        raise ValueError(f"MetaWorld has no scripted policy {policy_name} for {task!r}")

    # MT1 samples its task goals from NumPy's global RNG during construction.  Seed it
    # before creating the benchmark so the same CLI command selects the same variants.
    np.random.seed(seed)
    benchmark = metaworld.MT1(task)
    task_variants = list(benchmark.train_tasks)
    if variants > len(task_variants):
        raise ValueError(
            f"requested {variants} variants but {task!r} provides only {len(task_variants)}"
        )
    env = benchmark.train_classes[task]()
    policy_cls = getattr(policies, policy_name)
    records: List[Dict[str, Any]] = []
    try:
        for index in range(variants):
            reset_seed = seed + index
            record: Dict[str, Any] = {
                "variant_index": index,
                "reset_seed": reset_seed,
                "first_success_step": None,
                "success_steps": 0,
                "steps": 0,
                "terminated": False,
                "truncated": False,
                "error": None,
            }
            try:
                env.set_task(task_variants[index])
                observation, _info = env.reset(seed=reset_seed)
                policy = policy_cls()
                for step in range(1, max_steps + 1):
                    action = np.asarray(policy.get_action(observation), dtype=np.float32)
                    observation, _reward, terminated, truncated, info = env.step(action)
                    success = float(info.get("success", 0.0)) > 0.5
                    if success:
                        record["success_steps"] += 1
                        if record["first_success_step"] is None:
                            record["first_success_step"] = step
                    record["steps"] = step
                    record["terminated"] = bool(terminated)
                    record["truncated"] = bool(truncated)
                    if terminated or truncated:
                        break
            except Exception as exc:  # keep every failed variant visible in the report
                record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
    finally:
        env.close()

    summary = summarize_records(
        records,
        min_success_steps=min_success_steps,
        min_success_rate=min_success_rate,
    )
    return {
        "schema_version": 1,
        "task": task,
        "policy": policy_name,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "available_task_variants": len(task_variants),
        "packages": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "metaworld": importlib.metadata.version("metaworld"),
            "mujoco": importlib.metadata.version("mujoco"),
        },
        "summary": summary,
        "records": records,
        "claim_boundary": (
            "This probes the scripted expert without point-cloud rendering or policy training; "
            "it is a data-collection suitability gate, not a learned-policy result."
        ),
    }


def write_report(path: Path, report: Dict[str, Any], *, overwrite: bool) -> None:
    """Write JSON without silently replacing previous evidence."""

    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="reach-wall-v3")
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-success-steps", type=int, default=5)
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = probe_task(
        args.task,
        variants=args.variants,
        max_steps=args.max_steps,
        seed=args.seed,
        min_success_steps=args.min_success_steps,
        min_success_rate=args.min_success_rate,
    )
    if args.output is not None:
        try:
            write_report(args.output, report, overwrite=args.overwrite)
        except FileExistsError as exc:
            parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["summary"]["suitable_for_collection"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
