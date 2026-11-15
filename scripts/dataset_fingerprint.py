"""Create a content fingerprint for a Zarr demonstration store without publishing it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import zarr


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def fingerprint_store(path: Path) -> Dict[str, Any]:
    """Hash logical array contents and metadata, independent of the store directory."""

    root = zarr.open_group(str(path), mode="r")
    arrays: Dict[str, Any] = {}

    def collect(name: str, obj: Any) -> None:
        if isinstance(obj, zarr.Array):
            values = np.ascontiguousarray(obj[:])
            arrays[name] = {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "chunks": list(obj.chunks) if obj.chunks is not None else None,
                "attrs": dict(obj.attrs),
                "nbytes": int(values.nbytes),
                "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
            }

    root.visititems(collect)
    if not arrays:
        raise ValueError(f"Zarr store has no arrays: {path}")
    payload = {
        "group_attrs": dict(root.attrs),
        "arrays": {name: arrays[name] for name in sorted(arrays)},
    }
    return {
        "schema_version": 1,
        "store_name": path.name,
        "array_count": len(arrays),
        "total_nbytes": sum(entry["nbytes"] for entry in arrays.values()),
        "content_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        **payload,
        "claim_boundary": (
            "The fingerprint identifies logical Zarr contents and metadata; it does not "
            "publish the demonstration arrays or prove learned-policy performance."
        ),
    }


def write_report(path: Path, report: Dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = fingerprint_store(args.store)
    if args.output is not None:
        try:
            write_report(args.output, report, overwrite=args.overwrite)
        except FileExistsError as exc:
            parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
