"""Determinism probe for save/resume on the CPU backend (FPJ-031).

Writes per-step snapshots so two runs can be compared afterwards:

    python scripts/determinism_probe.py --mode single --steps 6 --out /tmp/a.npz
    python scripts/determinism_probe.py --mode resume --steps 6 --save-at 3 --out /tmp/b.npz
    python scripts/determinism_probe.py --compare /tmp/a.npz /tmp/b.npz

``single`` runs all steps in one process; ``resume`` saves at ``--save-at``, loads into a
fresh policy/trainer and continues.  ``--compare`` reports, per step, the largest absolute
weight difference and the loss difference, plus which tensors diverge first -- so a claim
about bit-exactness is a measurement rather than an assumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

NUM_POINTS = 8


def _shape_meta() -> Dict[str, Any]:
    return {
        "obs": {
            "point_cloud": {"shape": [NUM_POINTS, 3], "type": "point_cloud"},
            "agent_pos": {"shape": [9], "type": "low_dim"},
        },
        "action": {"shape": [4]},
    }


def _batch(seed: int = 5, batch_size: int = 2) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "obs": {
            "point_cloud": rng.normal(size=(batch_size, 2, NUM_POINTS, 3)).astype(np.float32),
            "agent_pos": rng.normal(size=(batch_size, 2, 9)).astype(np.float32),
        },
        "action": rng.normal(size=(batch_size, 4, 4)).astype(np.float32),
    }


def _build_policy(seed: int):
    import jittor as jt

    from flowpolicy_jittor.model import FlowPolicy, LinearNormalizer

    jt.flags.use_cuda = 0
    jt.set_global_seed(seed)
    policy = FlowPolicy(
        shape_meta=_shape_meta(),
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
    )
    data = _batch()
    policy.set_normalizer(
        LinearNormalizer().fit(
            {
                "action": data["action"],
                "agent_pos": data["obs"]["agent_pos"],
                "point_cloud": data["obs"]["point_cloud"],
            }
        )
    )
    return policy, data


def _trainer(policy):
    import jittor as jt  # noqa: F401 - ensures the backend flag is set before use

    from flowpolicy_jittor.training import Trainer

    return Trainer(policy, total_steps=8, lr=1e-2, warmup_steps=2, seed=0)


def _snapshot(policy) -> Dict[str, np.ndarray]:
    return {name: np.asarray(value).copy() for name, value in policy.state_dict(to="numpy").items()}


def _hash(snapshot: Dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(snapshot):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(snapshot[name]).tobytes())
    return digest.hexdigest()[:16]


def run(mode: str, steps: int, save_at: int, out: Path) -> Dict[str, Any]:
    policy, data = _build_policy(0)
    trainer = _trainer(policy)
    snapshots: List[Dict[str, np.ndarray]] = []
    losses: List[float] = []

    def step_once() -> None:
        record = trainer.train_step(data)
        losses.append(float(record["loss"]))
        snapshots.append(_snapshot(policy))

    if mode == "single":
        for _ in range(steps):
            step_once()
    elif mode == "resume":
        for _ in range(save_at):
            step_once()
        checkpoint = out.parent / (out.stem + ".ckpt")
        trainer.save_checkpoint(checkpoint, overwrite=True)
        fresh_policy, _ = _build_policy(0)
        resumed = _trainer(fresh_policy)
        resumed.load_checkpoint(checkpoint)
        policy, trainer = fresh_policy, resumed
        # Keep the pre-save snapshots so snapshot index == completed step for both modes;
        # otherwise the comparison would silently line up step 1 with step save_at+1.
        for _ in range(save_at, steps):
            step_once()
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    payload: Dict[str, np.ndarray] = {
        "losses": np.asarray(losses, dtype=np.float64),
        "meta": np.asarray(
            json.dumps(
                {
                    "mode": mode,
                    "steps": steps,
                    "save_at": save_at if mode == "resume" else 0,
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                    "mkl_cbwr": os.environ.get("MKL_CBWR"),
                    "hash": _hash(snapshots[-1]),
                }
            )
        ),
    }
    for index, snapshot in enumerate(snapshots):
        for name, value in snapshot.items():
            payload[f"s{index}:{name}"] = value
    np.savez(out, **payload)
    return {"out": str(out), "losses": losses, "final_hash": _hash(snapshots[-1])}


def compare(left_path: Path, right_path: Path) -> Dict[str, Any]:
    left = np.load(left_path, allow_pickle=False)
    right = np.load(right_path, allow_pickle=False)
    left_meta = json.loads(str(left["meta"]))
    right_meta = json.loads(str(right["meta"]))
    left_losses = np.asarray(left["losses"])
    right_losses = np.asarray(right["losses"])
    def _indices(bundle) -> List[int]:
        found = {int(key[1 : key.index(":")]) for key in bundle.files if key.startswith("s") and ":" in key}
        return sorted(found)

    left_indices, right_indices = _indices(left), _indices(right)
    if left_indices != right_indices:
        raise SystemExit(
            f"cannot compare snapshot steps {left_indices} against {right_indices}: "
            "the two runs did not cover the same steps"
        )
    rows = []
    for index in left_indices:
        prefix = f"s{index}:"
        names = sorted(key[len(prefix):] for key in right.files if key.startswith(prefix))
        worst_name, worst_delta = None, 0.0
        bit_identical = True
        for name in names:
            a = np.asarray(left[prefix + name])
            b = np.asarray(right[prefix + name])
            if not np.array_equal(a, b):
                bit_identical = False
                delta = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
                if delta > worst_delta:
                    worst_name, worst_delta = name, delta
        rows.append(
            {
                "step": index + 1,
                "bit_identical": bit_identical,
                "max_abs_delta": worst_delta,
                "worst_tensor": worst_name,
                "left_loss": float(left_losses[index]),
                "right_loss": float(right_losses[index]),
                "loss_delta": float(abs(float(left_losses[index]) - float(right_losses[index]))),
            }
        )
    return {
        "left": {"meta": left_meta, "losses": left_losses.tolist()},
        "right": {"meta": right_meta, "losses": right_losses.tolist()},
        "steps": rows,
        "all_bit_identical": all(row["bit_identical"] for row in rows),
        "max_abs_delta": max((row["max_abs_delta"] for row in rows), default=0.0),
        "first_diverging_step": next((row["step"] for row in rows if not row["bit_identical"]), None),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "resume"), default=None)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--save-at", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--compare", nargs=2, type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.compare:
        report = compare(args.compare[0], args.compare[1])
        text = json.dumps(report, indent=2)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
        first = report["first_diverging_step"]
        print(
            f"bit_identical={report['all_bit_identical']} first_diverging_step={first} "
            f"max_abs_delta={report['max_abs_delta']:.3e} "
            f"(threads={report['right']['meta']['omp_num_threads']}, "
            f"cbwr={report['right']['meta']['mkl_cbwr']})"
        )
        for row in report["steps"]:
            print(
                f"  step {row['step']}: identical={row['bit_identical']} "
                f"max_abs={row['max_abs_delta']:.3e} loss_delta={row['loss_delta']:.3e} "
                f"({row['worst_tensor']})"
            )
        return 0

    if not args.mode or not args.out:
        raise SystemExit("--mode and --out are required unless --compare is used")
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.mode == "resume" and not 0 < args.save_at < args.steps:
        raise SystemExit("--save-at must be inside (0, steps)")
    result = run(args.mode, args.steps, args.save_at, args.out)
    print(json.dumps({**result, "losses": [round(value, 10) for value in result["losses"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
