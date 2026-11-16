import copy

import numpy as np
import pytest

from scripts.derive_policy_checkpoint import (
    DETACHED_KIND,
    TRAINER_KIND,
    derive_payload,
    write_report,
)


def _payload(target_mode="code"):
    state = {
        "model.weight": np.array([1.0, 2.0], dtype=np.float32),
        "normalizer.scale": np.array([3.0], dtype=np.float32),
    }
    ema = {
        "names": ["model.weight"],
        "parameters": [np.array([10.0, 20.0], dtype=np.float32)],
    }
    if target_mode == "detached":
        ema = {
            "kind": DETACHED_KIND,
            "target_policy": {
                "model.weight": np.array([30.0, 40.0], dtype=np.float32),
                "normalizer.scale": np.array([3.0], dtype=np.float32),
            },
        }
    return {
        "kind": TRAINER_KIND,
        "schema_version": 2,
        "policy": {
            "schema_version": 3,
            "implementation_version": "test",
            "upstream_commit": "test",
            "jittor_version": "test",
            "config": {"target_mode": target_mode, "action_slice_mode": "code"},
            "policy": state,
            "extra": {},
        },
        "trainer": {"ema": ema},
    }


def test_code_ema_replaces_only_averaged_weights_and_can_select_exact_slice():
    source = _payload("code")
    derived = derive_payload(
        source,
        weights="ema",
        action_slice_mode="exact",
        source_sha256="abc",
    )
    np.testing.assert_array_equal(derived["policy"]["model.weight"], [10.0, 20.0])
    np.testing.assert_array_equal(derived["policy"]["normalizer.scale"], [3.0])
    assert derived["config"]["action_slice_mode"] == "exact"
    assert derived["extra"]["derived_evaluation"]["weights"] == "ema"
    np.testing.assert_array_equal(
        source["policy"]["policy"]["model.weight"], [1.0, 2.0]
    )


def test_detached_ema_uses_complete_frozen_target_state():
    derived = derive_payload(
        _payload("detached"),
        weights="ema",
        action_slice_mode="code",
        source_sha256="abc",
    )
    np.testing.assert_array_equal(derived["policy"]["model.weight"], [30.0, 40.0])


def test_online_derivative_keeps_weights_and_source_is_not_mutated():
    source = _payload("code")
    before = copy.deepcopy(source)
    derived = derive_payload(
        source,
        weights="online",
        action_slice_mode="exact",
        source_sha256="abc",
    )
    np.testing.assert_array_equal(derived["policy"]["model.weight"], [1.0, 2.0])
    np.testing.assert_array_equal(
        source["policy"]["policy"]["model.weight"],
        before["policy"]["policy"]["model.weight"],
    )


def test_rejects_mismatched_detached_target_keys():
    source = _payload("detached")
    source["trainer"]["ema"]["target_policy"].pop("normalizer.scale")
    with pytest.raises(KeyError, match="keys do not match"):
        derive_payload(
            source,
            weights="ema",
            action_slice_mode="code",
            source_sha256="abc",
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (np.array([10.0, 20.0, 30.0], dtype=np.float32), "shape mismatch"),
        (np.array([10.0, 20.0], dtype=np.float64), "dtype mismatch"),
        (np.array([10.0, np.nan], dtype=np.float32), "non-finite"),
    ],
)
def test_rejects_invalid_code_ema_replacement(replacement, message):
    source = _payload("code")
    source["trainer"]["ema"]["parameters"][0] = replacement
    with pytest.raises((TypeError, ValueError), match=message):
        derive_payload(
            source,
            weights="ema",
            action_slice_mode="code",
            source_sha256="abc",
        )


def test_rejects_invalid_detached_ema_replacement_shape():
    source = _payload("detached")
    source["trainer"]["ema"]["target_policy"]["model.weight"] = np.array(
        [30.0, 40.0, 50.0], dtype=np.float32
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        derive_payload(
            source,
            weights="ema",
            action_slice_mode="code",
            source_sha256="abc",
        )


def test_report_sidecar_is_atomic_and_refuses_implicit_overwrite(tmp_path):
    path = tmp_path / "derive.json"
    write_report(path, {"weights": "ema"}, overwrite=False)
    assert path.read_text(encoding="utf-8") == '{\n  "weights": "ema"\n}\n'
    with pytest.raises(FileExistsError):
        write_report(path, {"weights": "online"}, overwrite=False)
    write_report(path, {"weights": "online"}, overwrite=True)
    assert '"online"' in path.read_text(encoding="utf-8")
