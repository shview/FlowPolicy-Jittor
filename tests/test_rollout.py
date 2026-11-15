"""Tests for the closed-loop rollout helpers (no simulator, no Jittor training)."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rollout  # noqa: E402
import train_smoke  # noqa: E402


class FakePolicy:
    """Minimal stand-in for the parts of FlowPolicy that rollout touches."""

    n_obs_steps = 2
    n_action_steps = 2
    horizon = 4
    action_dim = 4
    nfe = 1

    def __init__(self, actions=None):
        self.calls = []
        self._actions = actions

    def predict_action(self, obs_dict, *, noise=None):
        self.calls.append({"obs": obs_dict, "noise": None if noise is None else noise.copy()})
        if self._actions is None:
            action = np.zeros((1, self.n_action_steps, self.action_dim), dtype=np.float32)
        else:
            action = np.asarray(self._actions, dtype=np.float32)[None]
        return {"action": action}


class FakeBridge:
    """Scripted simulator: successes and dosages per step."""

    def __init__(self, *, success_steps=(), done_at=None):
        self.hello = {"task": "fake"}
        self.success_steps = set(success_steps)
        self.done_at = done_at
        self.index = None
        self.steps = 0
        self.closed = False

    def reset(self, index):
        self.index = int(index)
        self.steps = 0
        return {
            "agent_pos": np.zeros(9, dtype=np.float32),
            "point_cloud": np.zeros((8, 3), dtype=np.float32),
        }

    def step(self, action):
        self.steps += 1
        observation = {
            "agent_pos": np.full(9, self.steps, dtype=np.float32),
            "point_cloud": np.zeros((8, 3), dtype=np.float32),
        }
        return {
            "obs": observation,
            "success": self.steps in self.success_steps,
            "done": self.done_at is not None and self.steps >= self.done_at,
            "step_cap": False,
        }

    def close(self):
        self.closed = True


def test_observation_window_keeps_the_last_frames_in_order():
    history = [
        {"agent_pos": np.full(9, value, dtype=np.float32), "point_cloud": np.zeros((8, 3), dtype=np.float32)}
        for value in (1.0, 2.0, 3.0)
    ]
    window = rollout.observation_window(history, 2)
    assert window["agent_pos"].shape == (1, 2, 9)
    assert window["agent_pos"][0, :, 0].tolist() == [2.0, 3.0]
    assert window["point_cloud"].shape == (1, 2, 8, 3)


def test_rollout_episode_counts_successes_steps_and_chunks():
    bridge = FakeBridge(success_steps={1, 3})
    policy = FakePolicy(actions=np.zeros((2, 4), dtype=np.float32))
    record = rollout.rollout_episode(
        policy,
        bridge,
        episode=0,
        index=7,
        warmup_obs=2,
        max_steps=10,
        noise_seed=0,
    )
    assert record["steps"] == 10
    assert record["success_steps"] == 2
    assert record["success"] is True
    assert record["action_chunks"] == 5
    assert record["done_reason"] == "step_cap"
    assert record["index"] == 7
    assert len(policy.calls) == 5
    # The same episode index and noise seed must produce the same noise across runs.
    again = rollout.rollout_episode(
        FakePolicy(actions=np.zeros((2, 4), dtype=np.float32)),
        FakeBridge(success_steps={1, 3}),
        episode=0,
        index=7,
        warmup_obs=2,
        max_steps=10,
        noise_seed=0,
    )
    assert again["success_steps"] == record["success_steps"]


def test_rollout_episode_stops_when_the_environment_reports_done():
    bridge = FakeBridge(done_at=3)
    policy = FakePolicy(actions=np.zeros((2, 4), dtype=np.float32))
    record = rollout.rollout_episode(
        policy,
        bridge,
        episode=0,
        index=0,
        warmup_obs=2,
        max_steps=50,
        noise_seed=0,
    )
    assert record["done_reason"] == "env_done"
    assert record["steps"] == 3


def test_rollout_episode_reports_a_failure_honestly():
    bridge = FakeBridge(success_steps=())
    record = rollout.rollout_episode(
        FakePolicy(),
        bridge,
        episode=1,
        index=1,
        warmup_obs=2,
        max_steps=4,
        noise_seed=3,
    )
    assert record["success"] is False
    assert record["success_steps"] == 0


def test_decode_observation_round_trips_float32_payloads():
    buffer = io.BytesIO()
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.save(buffer, array, allow_pickle=False)
    decoded = rollout.decode_observation({"point_cloud": base64.b64encode(buffer.getvalue()).decode("ascii")})
    assert np.array_equal(decoded["point_cloud"], array)
    assert decoded["point_cloud"].dtype == np.float32


def test_prepare_output_dir_refuses_to_overwrite_without_permission(tmp_path):
    target = tmp_path / "results"
    train_smoke.prepare_output_dir(target, overwrite=False)
    assert target.is_dir()
    (target / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        train_smoke.prepare_output_dir(target, overwrite=False)
    train_smoke.prepare_output_dir(target, overwrite=True)
    assert (target / "summary.json").exists()


def test_collate_stacks_nested_observations():
    items = [
        {
            "obs": {
                "agent_pos": np.full(9, index, dtype=np.float32),
                "point_cloud": np.zeros((4, 3), dtype=np.float32),
            },
            "action": np.zeros((4, 4), dtype=np.float32) + index,
        }
        for index in range(3)
    ]
    batch = train_smoke.collate(items)
    assert batch["action"].shape == (3, 4, 4)
    assert batch["obs"]["agent_pos"].shape == (3, 9)
    assert batch["obs"]["point_cloud"].shape == (3, 4, 3)
    assert batch["obs"]["agent_pos"][:, 0].tolist() == [0.0, 1.0, 2.0]
