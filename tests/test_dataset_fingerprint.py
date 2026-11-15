"""Tests for the sanitized Zarr content fingerprint (FPJ-033)."""

import sys
from pathlib import Path

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import dataset_fingerprint  # noqa: E402


def _store(path, value=1.0):
    root = zarr.open_group(str(path), mode="w")
    root.attrs["task"] = "test-v3"
    data = root.create_group("data")
    data.create_dataset("action", data=np.full((3, 4), value, dtype=np.float32))
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=np.asarray([3], dtype=np.int64))


def test_fingerprint_is_directory_independent_and_value_sensitive(tmp_path):
    first, second, changed = tmp_path / "a.zarr", tmp_path / "b.zarr", tmp_path / "c.zarr"
    _store(first)
    _store(second)
    _store(changed, value=2.0)
    report_a = dataset_fingerprint.fingerprint_store(first)
    report_b = dataset_fingerprint.fingerprint_store(second)
    report_c = dataset_fingerprint.fingerprint_store(changed)
    assert report_a["content_sha256"] == report_b["content_sha256"]
    assert report_a["content_sha256"] != report_c["content_sha256"]
    assert report_a["store_name"] != report_b["store_name"]
    assert report_a["array_count"] == 2


def test_fingerprint_contains_no_parent_path(tmp_path):
    store = tmp_path / "private" / "demo.zarr"
    _store(store)
    report = dataset_fingerprint.fingerprint_store(store)
    assert report["store_name"] == "demo.zarr"
    assert str(tmp_path) not in str(report)
