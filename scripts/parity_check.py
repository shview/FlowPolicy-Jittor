"""Jittor vs PyTorch parity for the FlowPolicy forward path and the CFM gradient (FPJ-028).

The Jittor side dumps a *bundle* (weights, inputs, outputs, gradients); the PyTorch side
re-runs the same computation with an independently written mirror model
(``scripts/parity_torch_model.py``) and reports the differences.  PyTorch is never
imported by the Jittor process, so a parity claim always means two separate runtimes
agreed on the same numbers.

Example:
    python scripts/parity_check.py --out outputs/parity
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_TORCH_PYTHON = "/home/s/.local/share/flowpolicy/parity-env/bin/python"

CONFIG: Dict[str, Any] = {
    "num_points": 32,
    "state_dim": 9,
    "action_dim": 4,
    "horizon": 4,
    "n_obs_steps": 2,
    "down_dims": [16, 32, 64],
    "kernel_size": 3,
    "n_groups": 4,
    "embed_dim": 16,
    "pointnet_out": 8,
    "state_mlp": [8, 8],
    "var_fc_hidden_dim": 8,
    "nfe": 1,
    "seed": 0,
    "batch_size": 2,
}


def _shape_meta(config: Mapping = CONFIG) -> Dict[str, Any]:
    return {
        "obs": {
            "point_cloud": {"shape": [config["num_points"], 3], "type": "point_cloud"},
            "agent_pos": {"shape": [config["state_dim"]], "type": "low_dim"},
        },
        "action": {"shape": [config["action_dim"]]},
    }


def _inputs(config=CONFIG) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(config["seed"])
    batch = config["batch_size"]
    return {
        "obs_point_cloud": rng.normal(size=(batch, 2, config["num_points"], 3)).astype(np.float32),
        "obs_agent_pos": rng.normal(size=(batch, 2, config["state_dim"])).astype(np.float32),
        "action": rng.normal(size=(batch, config["horizon"], config["action_dim"])).astype(np.float32),
        "noise": rng.normal(size=(batch, config["horizon"], config["action_dim"])).astype(np.float32),
        # Deliberately away from 0/1 and from a segment boundary so every branch of the
        # consistency loss (far mask, below-boundary mask) is exercised.
        "t": np.asarray([0.32, 0.61], dtype=np.float32),
    }


def build_policy(config=CONFIG):
    import jittor as jt

    from flowpolicy_jittor.model import FlowPolicy, LinearNormalizer

    jt.flags.use_cuda = 0
    jt.set_global_seed(int(config["seed"]))
    policy = FlowPolicy(
        shape_meta=_shape_meta(config),
        horizon=config["horizon"],
        n_action_steps=config["horizon"],
        n_obs_steps=config["n_obs_steps"],
        down_dims=tuple(config["down_dims"]),
        kernel_size=config["kernel_size"],
        n_groups=config["n_groups"],
        diffusion_step_embed_dim=config["embed_dim"],
        encoder_output_dim=config["pointnet_out"],
        state_mlp_size=tuple(config["state_mlp"]),
        pointcloud_encoder_cfg={
            "in_channels": 3,
            "out_channels": config["pointnet_out"],
            "use_layernorm": True,
            "final_norm": "layernorm",
            "normal_channel": False,
        },
        var_fc_hidden_dim=config["var_fc_hidden_dim"],
    )
    data = _inputs(config)
    policy.set_normalizer(
        LinearNormalizer().fit(
            {
                "action": data["action"],
                "agent_pos": data["obs_agent_pos"],
                "point_cloud": data["obs_point_cloud"],
            }
        )
    )
    return policy


def dump_bundle(bundle_path: Path, config=CONFIG) -> Dict[str, Any]:
    import jittor as jt

    from flowpolicy_jittor.model import FlowPolicy  # noqa: F401 - documented import site

    policy = build_policy(config)
    data = _inputs(config)
    observations = {
        "point_cloud": jt.array(data["obs_point_cloud"]),
        "agent_pos": jt.array(data["obs_agent_pos"]),
    }
    noise = jt.array(data["noise"])
    times = jt.array(data["t"])
    actions = jt.array(data["action"])

    with jt.no_grad():
        prediction = policy.predict_action(observations, noise=noise)
    action_pred = np.asarray(prediction["action_pred"].numpy(), dtype=np.float32)

    loss, parts = policy.compute_loss(
        {"obs": observations, "action": jt.array(data["action"])}, noise=noise, t=times
    )
    loss_value = float(loss.item())

    named = [(name, parameter) for name, parameter in policy.named_parameters()]
    trainable = [(name, parameter) for name, parameter in named if not parameter.is_stop_grad()]
    gradients = jt.grad(loss, [parameter for _, parameter in trainable])
    weights = {
        name: np.asarray(value) for name, value in policy.state_dict(to="numpy").items()
    }
    grad_map = {
        name: np.asarray(value.numpy()) for (name, _), value in zip(trainable, gradients)
    }

    loss_config = {
        "eps": float(policy.loss_fn.eps),
        "num_segments": int(policy.loss_fn.num_segments),
        "boundary": float(policy.loss_fn.boundary),
        "delta": float(policy.loss_fn.delta),
        "alpha": float(policy.loss_fn.alpha),
        "time_scale": float(policy.loss_fn.time_scale),
    }
    config_out = dict(config)
    config_out["loss"] = loss_config
    config_out["weight_names"] = sorted(weights)
    config_out["gradient_names"] = sorted(grad_map)

    payload: Dict[str, np.ndarray] = {
        "config_json": np.asarray(json.dumps(config_out)),
        "action_pred": action_pred,
        "loss": np.asarray([loss_value], dtype=np.float64),
        "endpoint_loss": np.asarray([float(parts["endpoint_loss"].item())], dtype=np.float64),
        "velocity_loss": np.asarray([float(parts["velocity_loss"].item())], dtype=np.float64),
    }
    for key, value in data.items():
        payload[key] = value
    for name, value in weights.items():
        payload["w:" + name] = value
    for name, value in grad_map.items():
        payload["g:" + name] = value
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(bundle_path, **payload)
    return {
        "bundle": str(bundle_path),
        "loss": loss_value,
        "action_pred_shape": list(action_pred.shape),
        "weight_count": len(weights),
        "gradient_count": len(grad_map),
    }


def run_torch_side(bundle: Path, out: Path, torch_python: str) -> Dict[str, Any]:
    python = Path(torch_python)
    if not python.exists():
        return {"ran": False, "reason": f"no torch interpreter at {torch_python}"}
    script = REPO_ROOT / "scripts" / "parity_torch_model.py"
    completed = subprocess.run(
        [str(python), str(script), "--bundle", str(bundle), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if completed.returncode != 0:
        return {
            "ran": False,
            "reason": f"torch side exited {completed.returncode}",
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    return {"ran": True, "output": completed.stdout[-2000:]}


def _verdict(report: Dict[str, Any], absolute_tolerance: float, relative_tolerance: float) -> Dict[str, Any]:
    """Element-wise ``|a - b| <= atol + rtol * scale`` on the worst element.

    The scale is the largest reference magnitude of that tensor, so a tiny tensor is
    compared on an absolute basis and a large one on a relative basis.  The earlier
    version used ``atol + rtol * max(scale, 1.0)``, which made small tensors (a loss of
    1.6e-4, a gradient of 1e-4) effectively compare only against ``atol`` -- an
    anti-pattern probe that shifted a weight by 1e-3 changed a gradient by 24% relative
    (3e-7 absolute) and was still reported as a pass.  Keep the floor at ``atol`` only.
    """
    checks: Dict[str, Any] = {}

    def check(name: str, stats: Dict[str, float]) -> None:
        allowed = absolute_tolerance + relative_tolerance * stats["reference_scale"]
        checks[name] = {
            "max_abs": stats["max_abs"],
            "max_rel": stats.get("max_rel"),
            "reference_scale": stats["reference_scale"],
            "allowed_abs": allowed,
            "passed": bool(stats["max_abs"] <= allowed),
        }

    check("action_pred", report["action_pred"])
    check("loss", report["loss"])
    for name, stats in report["gradients"].items():
        check("grad:" + name, stats)
    failed = sorted(name for name, entry in checks.items() if not entry["passed"])
    return {"checks": checks, "failed": failed, "passed": not failed}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--torch-python", default=os.environ.get("FPJ_TORCH_PYTHON", DEFAULT_TORCH_PYTHON))
    parser.add_argument("--absolute-tolerance", type=float, default=1e-7)
    parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    summary = dump_bundle(args.out / "bundle.npz")
    print(f"[jittor] loss={summary['loss']:.6f} weights={summary['weight_count']} "
          f"gradients={summary['gradient_count']} action_pred={summary['action_pred_shape']}")

    torch_result = run_torch_side(args.out / "bundle.npz", args.out / "torch_report.json", args.torch_python)
    result: Dict[str, Any] = {"jittor": summary, "torch": torch_result}
    exit_code = 0
    if not torch_result.get("ran"):
        result["parity"] = "not run"
        print(f"[torch] parity NOT run: {torch_result.get('reason')}")
        exit_code = 1
    else:
        report = json.loads((args.out / "torch_report.json").read_text())
        verdict = _verdict(report, args.absolute_tolerance, args.relative_tolerance)
        result["torch_report"] = {
            "torch_version": report["torch_version"],
            "gradient_coverage": report["gradient_coverage"],
        }
        result["verdict"] = verdict
        print(f"[torch] {report['torch_version']} action_pred_max_abs={report['action_pred']['max_abs']:.3e} "
              f"loss_max_abs={report['loss']['max_abs']:.3e} "
              f"grad_max_abs={report['gradient_coverage']['max_abs_over_all_gradients']:.3e} "
              f"checked={len(verdict['checks'])} failed={len(verdict['failed'])}")
        if verdict["failed"]:
            print("failed checks: " + ", ".join(verdict["failed"][:12]))
            exit_code = 1
    (args.out / "parity.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"parity": result.get("parity", "run"), "report": str(args.out / "parity.json")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
