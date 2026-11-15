"""Full paper channel configuration and the CUDA capability report (FPJ-028)."""

import json
import os

import numpy as np
import pytest

jt = pytest.importorskip("jittor")

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from flowpolicy_jittor.model import FlowPolicy, LinearNormalizer  # noqa: E402
from flowpolicy_jittor.training import Trainer  # noqa: E402

FULL_CHANNELS = (512, 1024, 2048)


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = previous


def _shape_meta(num_points):
    return {
        "obs": {
            "point_cloud": {"shape": [num_points, 3], "type": "point_cloud"},
            "agent_pos": {"shape": [9], "type": "low_dim"},
        },
        "action": {"shape": [4]},
    }


def _batch(num_points, batch_size=1, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "obs": {
            "point_cloud": rng.normal(size=(batch_size, 2, num_points, 3)).astype(np.float32),
            "agent_pos": rng.normal(size=(batch_size, 2, 9)).astype(np.float32),
        },
        "action": rng.normal(size=(batch_size, 4, 4)).astype(np.float32),
    }


def _policy(down_dims, num_points, *, kernel_size=5, embed_dim=128):
    policy = FlowPolicy(
        shape_meta=_shape_meta(num_points),
        horizon=4,
        n_action_steps=4,
        n_obs_steps=2,
        down_dims=down_dims,
        kernel_size=kernel_size,
        n_groups=8,
        diffusion_step_embed_dim=embed_dim,
        encoder_output_dim=64,
        state_mlp_size=(64, 64),
        pointcloud_encoder_cfg={
            "in_channels": 3,
            "out_channels": 64,
            "use_layernorm": True,
            "final_norm": "layernorm",
            "normal_channel": False,
        },
        var_fc_hidden_dim=512 if down_dims == FULL_CHANNELS else 64,
    )
    return policy


def _parameter_count(policy):
    return int(sum(int(np.prod(parameter.shape)) for parameter in policy.trainable_parameters()))


def test_full_channel_configuration_builds_and_forwards():
    policy = _policy(FULL_CHANNELS, num_points=64)
    count = _parameter_count(policy)
    # The paper configuration is dominated by the U-Net; a five-times smaller model would
    # mean the channel tuple was silently reduced.
    assert count > 200_000_000, f"unexpected parameter count {count}"
    data = _batch(num_points=64)
    policy.set_normalizer(
        LinearNormalizer().fit(
            {
                "action": data["action"],
                "agent_pos": data["obs"]["agent_pos"],
                "point_cloud": data["obs"]["point_cloud"],
            }
        )
    )
    result = policy.predict_action(data["obs"], noise=jt.zeros((1, 4, 4), dtype="float32"))
    action = np.asarray(result["action"].numpy())
    assert action.shape == (1, 3, 4)  # code-faithful slice: 3 of 4 horizon steps
    assert np.isfinite(action).all()
    assert result["nfe"] == 1


def test_full_channel_backward_runs_with_a_smaller_batch():
    """Backward at 512/1024/2048 in one step would need the whole 8 GB; scale down the
    channel tuple and keep the same code path, then report the real full-channel step
    from scripts/cuda_smoke.py."""
    reduced = (128, 256, 512)
    policy = _policy(reduced, num_points=32)
    data = _batch(num_points=32)
    policy.set_normalizer(
        LinearNormalizer().fit(
            {
                "action": data["action"],
                "agent_pos": data["obs"]["agent_pos"],
                "point_cloud": data["obs"]["point_cloud"],
            }
        )
    )
    trainer = Trainer(policy, total_steps=2, lr=1e-4, warmup_steps=0, seed=0)
    record = trainer.train_step(data)
    assert np.isfinite(record["loss"])
    assert record["step"] == 1


def test_cuda_report_records_the_measured_answer_not_an_assumption():
    from cuda_smoke import probe_cuda

    report = probe_cuda()
    assert "available" in report
    assert isinstance(report["available"], bool)
    if not report["available"]:
        # A refusal must name its stage and carry the reason; it must never be a bare
        # False that a reader could mistake for "not tested".
        assert report["stage"] in {"set_use_cuda", "matmul"}
        assert report["error"]
    assert jt.flags.use_cuda == 0, "the probe must restore the CPU backend"


def test_cuda_smoke_writes_a_report(tmp_path):
    from cuda_smoke import main

    out = tmp_path / "cuda"
    code = main(["--out", str(out), "--skip-full-channels"])
    assert code == 0
    report = json.loads((out / "report.json").read_text())
    assert report["cuda_verified"] is report["cuda"]["available"]
    assert "prohibited_claims" in report
    if not report["cuda_verified"]:
        assert "NOT available" in report["cuda_note"]
    # Never overwrite silently.
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["--out", str(out), "--skip-full-channels"])
