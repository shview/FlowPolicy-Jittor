"""Closed-loop MetaWorld rollout driven from the Jittor environment (FPJ-027).

The policy runs here (Jittor); the simulator runs in the isolated simulator virtualenv
behind ``scripts/sim_server.py``.  Nothing is imported across that boundary: plain JSON
lines with base64 ``.npy`` payloads carry the observations and actions.

This script is a smoke test of the closed loop.  It reports what actually happened --
including episodes that fail -- and never derives a success rate from the training loss.

Example:
    python scripts/rollout.py --data data/reach-v3.zarr \
        --checkpoint outputs/smoke/policy.ckpt --episodes 5 --out outputs/rollout
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

DEFAULT_SIM_PYTHON = "/home/s/.local/share/flowpolicy/sim-env/bin/python"


def decode_observation(payload: Dict[str, str]) -> Dict[str, np.ndarray]:
    observation = {}
    for key, value in payload.items():
        buffer = io.BytesIO(base64.b64decode(value))
        observation[key] = np.load(buffer, allow_pickle=False)
    return observation


class SimBridge:
    """Line protocol to ``sim_server.py`` running in the simulator virtualenv."""

    def __init__(self, *, sim_python: str, task: str, image_size: int, max_steps: int, seed: int, verbose: bool = False):
        self.process = subprocess.Popen(
            [
                sim_python,
                str(REPO_ROOT / "scripts" / "sim_server.py"),
                "--task",
                task,
                "--image-size",
                str(image_size),
                "--max-steps",
                str(max_steps),
                "--seed",
                str(seed),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None if verbose else subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
        )
        self.hello = self.request({"cmd": "hello"})

    def request(self, payload: Dict[str, object]) -> Dict[str, object]:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"simulator closed the bridge unexpectedly (exit={self.process.poll()})")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"simulator error: {reply['error']}")
        return reply

    def reset(self, index: int) -> Dict[str, np.ndarray]:
        reply = self.request({"cmd": "reset", "index": int(index)})
        return decode_observation(reply["obs"])

    def step(self, action: np.ndarray) -> Dict[str, object]:
        reply = self.request({"cmd": "step", "action": [float(value) for value in action]})
        return {
            "obs": decode_observation(reply["obs"]),
            "success": bool(reply["success"]),
            "done": bool(reply["done"]) or bool(reply.get("step_cap", False)),
            "step_cap": bool(reply.get("step_cap", False)),
        }

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.request({"cmd": "close"})
        except Exception:  # noqa: BLE001 - the bridge is best-effort on shutdown
            pass
        finally:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                self.process.kill()


def observation_window(history: List[Dict[str, np.ndarray]], n_obs_steps: int) -> Dict[str, np.ndarray]:
    window = history[-n_obs_steps:]
    return {key: np.stack([step[key] for step in window])[None] for key in window[0]}


def rollout_episode(
    policy,
    bridge: SimBridge,
    *,
    episode: int,
    index: int,
    warmup_obs: int,
    max_steps: int,
    noise_seed: int,
) -> Dict[str, object]:
    observation = bridge.reset(index)
    history = [observation] * warmup_obs
    n_action_steps = int(policy.n_action_steps)
    successes = 0
    steps = 0
    done_reason = None
    chunk_index = 0
    latencies: List[float] = []
    while steps < max_steps:
        window = observation_window(history, int(policy.n_obs_steps))
        rng = np.random.default_rng([noise_seed, episode, chunk_index])
        noise = rng.standard_normal(size=(1, int(policy.horizon), int(policy.action_dim))).astype(np.float32)
        started = time.perf_counter()
        prediction = policy.predict_action(window, noise=noise)
        latencies.append(time.perf_counter() - started)
        actions = np.asarray(prediction["action"])[0]
        chunk_index += 1
        for action in actions[:n_action_steps]:
            result = bridge.step(action)
            steps += 1
            successes += int(result["success"])
            history.append(result["obs"])
            if result["done"]:
                done_reason = "step_cap" if result.get("step_cap") else "env_done"
                break
        if done_reason is not None:
            break
        if steps >= max_steps:
            done_reason = "step_cap"
            break
    return {
        "episode": episode,
        "index": index,
        "steps": steps,
        "success_steps": successes,
        "success": bool(successes > 0),
        "done_reason": done_reason or "step_cap",
        "action_chunks": chunk_index,
        "policy_latency_ms": {
            "mean": 1000.0 * float(np.mean(latencies)) if latencies else None,
            "p95": 1000.0 * float(np.percentile(latencies, 95)) if latencies else None,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--task", default="reach-v3")
    parser.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--action-slice-mode", choices=("code", "exact"), default="code")
    parser.add_argument(
        "--target-mode",
        choices=("code", "detached"),
        default="code",
        help="must match the training checkpoint configuration",
    )
    parser.add_argument("--down-dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--encoder-out", type=int, default=64)
    parser.add_argument("--pointnet-out", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--var-fc-hidden-dim", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose-sim", action="store_true")
    args = parser.parse_args(argv)

    import jittor as jt

    from flowpolicy_jittor.data import MetaworldZarrDataset

    from train_smoke import build_policy, prepare_output_dir, sha256_file

    if args.device == "cuda" and not jt.has_cuda:
        raise SystemExit("--device cuda requested, but Jittor reports no CUDA backend")
    jt.flags.use_cuda = int(args.device == "cuda")
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be at least 1")
    if not Path(args.sim_python).exists():
        raise SystemExit(f"simulator interpreter not found: {args.sim_python}")
    out_dir = prepare_output_dir(args.out, args.overwrite)

    dataset = MetaworldZarrDataset(
        args.data,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=4,
        num_points=args.num_points,
        val_ratio=0.02,
        seed=args.seed,
        mode="train",
    )
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
        action_slice_mode=args.action_slice_mode,
        target_mode=args.target_mode,
    )
    checkpoint_extra = policy.load_checkpoint(args.checkpoint)

    # Separate three effects that were previously collapsed into one number:
    # repeated CUDA inference, loading into the same object, and loading into a fresh
    # object.  The distinction matters because a small backend-level inference drift
    # must not be reported as checkpoint corruption.
    sample = dataset[0]
    # `dataset[i]["obs"]` already carries the full observation window, so it only needs
    # a batch axis here.
    window = {key: np.asarray(sample["obs"][key])[None] for key in sample["obs"]}
    fixed_noise = np.random.default_rng([args.noise_seed, 12345]).standard_normal(
        size=(1, int(policy.horizon), int(policy.action_dim))
    ).astype(np.float32)
    first = np.asarray(policy.predict_action(window, noise=fixed_noise)["action"])
    repeated = np.asarray(policy.predict_action(window, noise=fixed_noise)["action"])
    repeat_delta = float(np.abs(first - repeated).max())
    checkpoint_extra = policy.load_checkpoint(args.checkpoint)
    same_object = np.asarray(policy.predict_action(window, noise=fixed_noise)["action"])
    same_object_reload_delta = float(np.abs(first - same_object).max())

    fresh_policy = build_policy(
        dataset,
        down_dims=tuple(args.down_dims),
        embed_dim=args.embed_dim,
        encoder_out=args.encoder_out,
        pointnet_out=args.pointnet_out,
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
        var_fc_hidden_dim=args.var_fc_hidden_dim,
        seed=args.seed,
        action_slice_mode=args.action_slice_mode,
        target_mode=args.target_mode,
    )
    fresh_policy.load_checkpoint(args.checkpoint)
    fresh = np.asarray(fresh_policy.predict_action(window, noise=fixed_noise)["action"])
    fresh_policy_reload_delta = float(np.abs(first - fresh).max())
    del fresh_policy

    bridge = SimBridge(
        sim_python=args.sim_python,
        task=args.task,
        image_size=args.image_size,
        max_steps=args.max_steps,
        seed=args.seed,
        verbose=args.verbose_sim,
    )
    episodes: List[Dict[str, object]] = []
    started = time.perf_counter()
    try:
        for index in range(args.episodes):
            record = rollout_episode(
                policy,
                bridge,
                episode=index,
                index=args.seed * 1000 + index,
                warmup_obs=int(policy.n_obs_steps),
                max_steps=args.max_steps,
                noise_seed=args.noise_seed,
            )
            episodes.append(record)
            print(
                f"[rollout] episode {index + 1}/{args.episodes}: steps={record['steps']} "
                f"success_steps={record['success_steps']} success={record['success']} "
                f"reason={record['done_reason']}",
                flush=True,
            )
    finally:
        bridge.close()
    elapsed = time.perf_counter() - started

    summary = {
        "task": args.task,
        "seed": args.seed,
        "noise_seed": args.noise_seed,
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_extra_keys": sorted(checkpoint_extra),
        "device": args.device,
        "model_config": {
            "down_dims": args.down_dims,
            "embed_dim": args.embed_dim,
            "encoder_out": args.encoder_out,
            "pointnet_out": args.pointnet_out,
            "kernel_size": args.kernel_size,
            "n_groups": args.n_groups,
            "var_fc_hidden_dim": args.var_fc_hidden_dim,
            "target_mode": args.target_mode,
        },
        "action_slice_mode": args.action_slice_mode,
        "policy_action_bounds": [int(value) for value in policy._action_bounds()],
        "nfe": int(policy.nfe),
        "policy_horizon": int(policy.horizon),
        "n_obs_steps": int(policy.n_obs_steps),
        "n_action_steps": int(policy.n_action_steps),
        "num_points": args.num_points,
        "simulator": bridge.hello,
        "sim_python": args.sim_python,
        "episodes": episodes,
        "episode_count": len(episodes),
        "success_episodes": int(sum(1 for record in episodes if record["success"])),
        "total_steps": int(sum(record["steps"] for record in episodes)),
        "wall_seconds": elapsed,
        "checkpoint_repeat_max_abs_delta": repeat_delta,
        "checkpoint_same_object_reload_max_abs_delta": same_object_reload_delta,
        "checkpoint_fresh_policy_reload_max_abs_delta": fresh_policy_reload_delta,
        # Backwards-compatible aggregate: the largest observed checkpoint-load delta.
        "checkpoint_reload_max_abs_delta": max(
            same_object_reload_delta, fresh_policy_reload_delta
        ),
        "prohibited_claims": [
            "fewer than the declared formal episode count is not a success-rate measurement",
            "matching the paper architecture does not imply matching its training recipe or simulator",
        ],
    }
    (out_dir / "rollout.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "episode_count",
                    "success_episodes",
                    "total_steps",
                    "wall_seconds",
                    "checkpoint_repeat_max_abs_delta",
                    "checkpoint_same_object_reload_max_abs_delta",
                    "checkpoint_fresh_policy_reload_max_abs_delta",
                    "checkpoint_reload_max_abs_delta",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
