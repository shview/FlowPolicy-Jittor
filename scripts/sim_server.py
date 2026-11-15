"""Line-based simulator server for the rollout bridge (runs in the sim environment).

The Jittor environment never imports mujoco or metaworld: it spawns this script with
the isolated simulator interpreter and exchanges one JSON object per line on
stdin/stdout.  Arrays travel as base64-encoded ``.npy`` payloads so nothing is
rounded.

Protocol (request -> response, one line each):
    {"cmd": "hello"}                      -> {"ok": true, "task": ..., "action_dim": 4, ...}
    {"cmd": "reset", "seed": 0}           -> {"obs": {...}, "success": false, "done": false}
    {"cmd": "step", "action": [..4..]}    -> {"obs": {...}, "reward": r, "success": bool, "done": bool}
    {"cmd": "close"}                      -> {"ok": true} then exit

``obs`` carries ``agent_pos`` (9,) and ``point_cloud`` (1024, 3) as float32 arrays.
Run it with ``MUJOCO_GL=egl`` for headless software rendering.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def encode(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array, dtype=np.float32), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode(payload: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(payload)), allow_pickle=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="reach-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--point-count", type=int, default=1024)
    args = parser.parse_args(argv)

    from record_demos import MetaWorldAdapter  # lives in scripts/, shared with recording

    adapter = MetaWorldAdapter(
        args.task,
        seed=args.seed,
        size=args.image_size,
        point_count=args.point_count,
    )
    step_index = 0
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            command = request.get("cmd")
            if command == "hello":
                reply = {
                    "ok": True,
                    "task": args.task,
                    "action_dim": int(adapter.env.action_space.shape[0]),
                    "point_count": int(args.point_count),
                    "image_size": int(args.image_size),
                    "seed": int(args.seed),
                    "variants": int(len(adapter._tasks)),
                }
            elif command == "reset":
                # `index` selects the task variant and offsets the reset seed, exactly as
                # the demonstration collector does, so replaying an index replays an episode.
                index = int(request.get("index", 0))
                observation = adapter.reset(index)
                step_index = 0
                reply = {
                    "obs": {
                        "agent_pos": encode(observation["agent_pos"]),
                        "point_cloud": encode(observation["point_cloud"]),
                    },
                    "success": False,
                    "done": False,
                    "step": step_index,
                }
            elif command == "step":
                action = np.asarray(request["action"], dtype=np.float32)
                observation, success = adapter.step(action)
                step_index += 1
                cap_reached = step_index >= args.max_steps
                reply = {
                    "obs": {
                        "agent_pos": encode(observation["agent_pos"]),
                        "point_cloud": encode(observation["point_cloud"]),
                    },
                    "success": bool(success),
                    "done": bool(observation["done"]),
                    "step_cap": bool(cap_reached),
                    "step": step_index,
                }
            elif command == "close":
                sys.stdout.write(json.dumps({"ok": True}) + "\n")
                sys.stdout.flush()
                break
            else:
                reply = {"error": f"unknown command {command!r}"}
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
