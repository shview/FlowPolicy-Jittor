"""Print an environment report for FlowPolicy-Jittor.

Normal mode only inspects package metadata. Strict modes import dependencies
and execute a small Jittor backend operation in isolated child processes.  A
first Jittor import may compile code and populate its oneDNN cache, so strict
mode is intentionally not side-effect-free.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Sequence, Tuple


PACKAGES = (
    "jittor",
    "numpy",
    "zarr",
    "yaml",
    "pytest",
    "torch",
    "pytorch3d",
    "mujoco_py",
    "gym",
    "metaworld",
)
TRAIN_IMPORTS = ("jittor", "numpy", "zarr", "yaml")
SIM_IMPORTS = (
    "numpy",
    "zarr",
    "torch",
    "pytorch3d",
    "mujoco_py",
    "gym",
    "metaworld",
)
TRAIN_PYTHON_RANGE = ((3, 8), (3, 12))
SIM_PYTHON_RANGE = ((3, 8), (3, 9))
PROBE_MARKER = "__FLOWPOLICY_IMPORT_PROBE__="
TRAIN_VERSION_RULES = {
    "jittor": ("exact", (1, 3, 11, 0)),
    "numpy": ("range", (1, 23), (2, 0)),
    "zarr": ("range", (2, 12), (3, 0)),
    "yaml": ("range", (6, 0), (7, 0)),
}


def package_status(import_name: str) -> Dict[str, Any]:
    available = importlib.util.find_spec(import_name) is not None
    result: Dict[str, Any] = {"available": available}
    if not available:
        return result

    distribution_name = "PyYAML" if import_name == "yaml" else import_name
    try:
        result["version"] = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        result["version"] = "unknown"
    return result


def import_probe(import_name: str, timeout: int = 600) -> Dict[str, Any]:
    """Import one dependency in a child process and return structured status."""

    probe_code = f"""
import importlib
import json
name = {import_name!r}
payload = {{"import_ok": False, "module": name}}
try:
    module = importlib.import_module(name)
    payload["import_ok"] = True
    payload["module_version"] = str(getattr(module, "__version__", "unknown"))
    if name == "jittor":
        payload["has_cuda"] = bool(getattr(module, "has_cuda", False))
        payload["use_cuda"] = int(getattr(module.flags, "use_cuda", 0))
        previous_use_cuda = int(module.flags.use_cuda)
        try:
            module.flags.use_cuda = 0
            cpu_array = ((module.array([1.0, 2.0]) ** 2).sum()).numpy()
            module.sync_all()
            cpu_value = float(cpu_array.reshape(-1)[0])
            payload["cpu_smoke_ok"] = bool(cpu_value == 5.0)
            payload["cpu_smoke_value"] = cpu_value

            if payload["has_cuda"]:
                try:
                    module.flags.use_cuda = 1
                    cuda_array = (
                        (module.array([1.0, 2.0]) ** 2).sum()
                    ).numpy()
                    module.sync_all()
                    cuda_value = float(cuda_array.reshape(-1)[0])
                    payload["cuda_smoke_ok"] = bool(cuda_value == 5.0)
                    payload["cuda_smoke_value"] = cuda_value
                except BaseException as cuda_exc:
                    # Keep a usable CPU probe successful under --allow-cpu,
                    # while strict CUDA mode will fail on cuda_smoke_ok below.
                    payload["cuda_smoke_ok"] = False
                    payload["cuda_smoke_error_type"] = type(
                        cuda_exc
                    ).__name__
                    payload["cuda_smoke_error"] = str(cuda_exc)
            else:
                payload["cuda_smoke_ok"] = False
        finally:
            module.flags.use_cuda = previous_use_cuda
except BaseException as exc:
    payload["import_ok"] = False
    payload["error_type"] = type(exc).__name__
    payload["error"] = str(exc)
print({PROBE_MARKER!r} + json.dumps(payload, ensure_ascii=False))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_code],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "import_ok": False,
            "module": import_name,
            "error_type": "TimeoutExpired",
            "error": f"import exceeded {timeout} seconds",
            "stdout": (exc.stdout or "")[-2000:],
            "stderr": (exc.stderr or "")[-2000:],
        }

    payload = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(PROBE_MARKER):
            try:
                payload = json.loads(line[len(PROBE_MARKER) :])
            except json.JSONDecodeError:
                payload = None
            break
    if payload is None:
        payload = {
            "import_ok": False,
            "module": import_name,
            "error_type": "ProbeProtocolError",
            "error": "child process did not emit a parseable result",
        }
    payload["returncode"] = completed.returncode
    if completed.stderr.strip():
        payload["stderr_tail"] = completed.stderr.strip()[-2000:]
    return payload


def nvidia_status() -> Dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False}
    command = [
        executable,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    return {
        "available": completed.returncode == 0,
        "output": completed.stdout.strip(),
        "error": completed.stderr.strip(),
    }


def version_in_range(
    version: Tuple[int, int],
    bounds: Tuple[Tuple[int, int], Tuple[int, int]],
) -> bool:
    lower, upper = bounds
    return lower <= version < upper


def numeric_version(version: str) -> Tuple[int, ...]:
    """Return the leading numeric release tuple without extra dependencies."""

    match = re.match(r"^(\d+(?:\.\d+)*)", version)
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def version_rule_error(import_name: str, version: str) -> str:
    rule = TRAIN_VERSION_RULES[import_name]
    parsed = numeric_version(version)
    if not parsed:
        return f"cannot parse {import_name} version {version!r}"
    if rule[0] == "exact":
        expected = rule[1]
        if parsed != expected:
            expected_text = ".".join(str(part) for part in expected)
            return (
                f"{import_name} {version} does not match required "
                f"version {expected_text}"
            )
    else:
        lower, upper = rule[1], rule[2]
        if not (parsed >= lower and parsed < upper):
            lower_text = ".".join(str(part) for part in lower)
            upper_text = ".".join(str(part) for part in upper)
            return (
                f"{import_name} {version} is outside "
                f">={lower_text},<{upper_text}"
            )
    return ""


def format_range(
    bounds: Tuple[Tuple[int, int], Tuple[int, int]]
) -> str:
    lower, upper = bounds
    return f">={lower[0]}.{lower[1]},<{upper[0]}.{upper[1]}"


def build_report(
    probe_names: Sequence[str] = (), probe_timeout: int = 600
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
        },
        "gpu": nvidia_status(),
        "packages": {name: package_status(name) for name in PACKAGES},
    }
    if probe_names:
        report["import_probes"] = {
            name: import_probe(name, timeout=probe_timeout)
            for name in probe_names
        }
    return report


def strict_errors(
    report: Dict[str, Any], mode: str, require_cuda: bool = True
) -> List[str]:
    errors: List[str] = []
    py_version = (sys.version_info.major, sys.version_info.minor)
    probes = report.get("import_probes", {})

    if mode == "train":
        bounds = TRAIN_PYTHON_RANGE
        required = TRAIN_IMPORTS
    elif mode == "sim":
        bounds = SIM_PYTHON_RANGE
        required = SIM_IMPORTS
        if platform.system() != "Linux":
            errors.append("simulation environment must run on Linux/WSL2")
    else:
        raise ValueError(f"unknown strict mode: {mode}")

    if not version_in_range(py_version, bounds):
        errors.append(
            f"Python {py_version[0]}.{py_version[1]} is outside "
            f"the {mode} range {format_range(bounds)}"
        )

    for name in required:
        status = probes.get(name, {})
        if not status.get("import_ok", False):
            detail = status.get("error", "import probe did not run")
            errors.append(f"cannot import {name}: {detail}")

    if mode == "train":
        for name in TRAIN_IMPORTS:
            package = report.get("packages", {}).get(name, {})
            if not package.get("available"):
                continue
            version_error = version_rule_error(
                name, str(package.get("version", "unknown"))
            )
            if version_error:
                errors.append(version_error)

    if mode == "train" and require_cuda:
        jittor_status = probes.get("jittor", {})
        if jittor_status.get("import_ok"):
            if not jittor_status.get("cpu_smoke_ok", False):
                errors.append("Jittor imported but its CPU smoke operation failed")
            if not jittor_status.get("has_cuda", False):
                errors.append("Jittor imported but cannot see a CUDA backend")
            elif not jittor_status.get("cuda_smoke_ok", False):
                errors.append(
                    "Jittor can see CUDA but a synchronized CUDA operation failed"
                )
    elif mode == "train":
        jittor_status = probes.get("jittor", {})
        if jittor_status.get("import_ok") and not jittor_status.get(
            "cpu_smoke_ok", False
        ):
            errors.append("Jittor imported but its CPU smoke operation failed")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--strict-train",
        action="store_true",
        help=(
            "import the Jittor training stack in child processes and fail "
            "when Python, packages, or the requested backend are not ready"
        ),
    )
    modes.add_argument(
        "--strict-sim",
        action="store_true",
        help=(
            "import the legacy simulation stack in child processes; EGL "
            "rendering is checked by a separate smoke test"
        ),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "with --strict-train, validate a CPU development environment "
            "without requiring a Jittor CUDA backend"
        ),
    )
    parser.add_argument(
        "--probe-timeout",
        type=int,
        default=600,
        metavar="SECONDS",
        help=(
            "maximum time for each strict-mode import/backend probe; "
            "first-time Jittor compilation can take several minutes"
        ),
    )
    args = parser.parse_args()

    mode = "train" if args.strict_train else "sim" if args.strict_sim else None
    if args.allow_cpu and mode != "train":
        parser.error("--allow-cpu requires --strict-train")
    if args.probe_timeout <= 0:
        parser.error("--probe-timeout must be a positive integer")
    if mode == "train":
        probe_names: Sequence[str] = TRAIN_IMPORTS
    elif mode == "sim":
        probe_names = SIM_IMPORTS
    else:
        probe_names = ()

    report = build_report(probe_names, probe_timeout=args.probe_timeout)
    require_cuda = mode == "train" and not args.allow_cpu
    errors = (
        strict_errors(report, mode, require_cuda=require_cuda) if mode else []
    )
    if mode:
        report["strict_check"] = {
            "mode": mode,
            "passed": not errors,
            "errors": errors,
            "cuda_required": require_cuda,
            "egl_rendering_checked": False,
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

