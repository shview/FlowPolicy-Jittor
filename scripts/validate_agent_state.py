"""Validate the repository's serial multi-agent coordination state."""

from __future__ import annotations

import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / ".agents" / "STATE.yaml"
QUEUE_PATH = ROOT / ".agents" / "QUEUE.md"
HANDOFFS_PATH = ROOT / ".agents" / "HANDOFFS.md"

STAGE_ORDER = (
    "implementation",
    "verification",
    "anti_pattern",
    "quality",
    "gate",
)
VALID_STAGES = set(STAGE_ORDER)
VALID_STATUSES = {
    "ready",
    "in_progress",
    "needs_fix",
    "waiting_user",
    "blocked",
    "done",
}
REQUIRED_CURRENT_KEYS = {
    "task_id",
    "title",
    "attempt",
    "stage",
    "status",
    "owner",
    "last_handoff_id",
    "next_stage",
    "role_handoffs",
    "objective",
    "scope",
    "acceptance",
    "required_commands",
}


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def _validate_relative_allowed_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return f"allowed path must be a non-empty string, got {value!r}"
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        return f"allowed path must stay inside the repository: {value!r}"
    return None


def validate() -> list[str]:
    errors: list[str] = []
    for path in (STATE_PATH, QUEUE_PATH, HANDOFFS_PATH):
        if not path.is_file():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")
    if errors:
        return errors

    try:
        state = _load_yaml(STATE_PATH)
    except (OSError, TypeError, yaml.YAMLError) as exc:
        return [f"cannot parse {STATE_PATH.relative_to(ROOT)}: {exc}"]

    if state.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if state.get("execution_mode") != "serial":
        errors.append("execution_mode must be 'serial'")
    if not isinstance(state.get("updated_on"), str):
        errors.append("updated_on must be an ISO date string")

    current = state.get("current")
    if not isinstance(current, Mapping):
        errors.append("current must be a mapping")
        return errors

    missing = sorted(REQUIRED_CURRENT_KEYS - set(current))
    if missing:
        errors.append("current is missing keys: " + ", ".join(missing))

    stage = current.get("stage")
    status = current.get("status")
    if stage not in VALID_STAGES:
        errors.append(f"invalid current.stage: {stage!r}")
    if status not in VALID_STATUSES:
        errors.append(f"invalid current.status: {status!r}")

    attempt = current.get("attempt")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        errors.append(f"current.attempt must be a positive integer, got {attempt!r}")

    owner = current.get("owner")
    if status == "in_progress":
        if not isinstance(owner, str) or not owner.strip():
            errors.append("in_progress requires a non-empty current.owner")
    elif status in {"ready", "needs_fix"} and owner is not None:
        errors.append(f"status {status!r} requires current.owner to be null")

    next_stage = current.get("next_stage")
    if stage in VALID_STAGES:
        stage_index = STAGE_ORDER.index(stage)
        expected_next = (
            STAGE_ORDER[stage_index + 1]
            if stage_index + 1 < len(STAGE_ORDER)
            else "next_task"
        )
        if next_stage != expected_next:
            errors.append(
                f"current.next_stage must be {expected_next!r} for stage {stage!r}, "
                f"got {next_stage!r}"
            )

    task_id = current.get("task_id")
    if not isinstance(task_id, str) or not re.fullmatch(r"FPJ-\d{3}", task_id):
        errors.append(f"invalid current.task_id: {task_id!r}")
    else:
        queue_text = QUEUE_PATH.read_text(encoding="utf-8")
        if task_id not in queue_text:
            errors.append(f"current task {task_id} is absent from .agents/QUEUE.md")
        current_markers = re.findall(
            r"^### \[>\] (FPJ-\d{3})\b", queue_text, flags=re.MULTILINE
        )
        if len(current_markers) != 1:
            errors.append(
                ".agents/QUEUE.md must contain exactly one '[>]' current task, "
                f"found {len(current_markers)}"
            )
        elif current_markers[0] != task_id:
            errors.append(
                f"queue current task {current_markers[0]!r} does not match {task_id!r}"
            )

    handoff_id = current.get("last_handoff_id")
    if not isinstance(handoff_id, str) or not handoff_id:
        errors.append("current.last_handoff_id must be a non-empty string")
    else:
        handoffs_text = HANDOFFS_PATH.read_text(encoding="utf-8")
        handoff_headings = re.findall(
            r"^## ([^\r\n]+)$", handoffs_text, flags=re.MULTILINE
        )
        duplicate_handoffs = sorted(
            {item for item in handoff_headings if handoff_headings.count(item) > 1}
        )
        if duplicate_handoffs:
            errors.append(
                "duplicate handoff ids: " + ", ".join(duplicate_handoffs)
            )
        if f"## {handoff_id}" not in handoffs_text:
            errors.append(
                f"last handoff {handoff_id!r} is absent from .agents/HANDOFFS.md"
            )

        role_handoffs = current.get("role_handoffs")
        if not isinstance(role_handoffs, Mapping):
            errors.append("current.role_handoffs must be a mapping")
        else:
            if set(role_handoffs) != VALID_STAGES:
                errors.append(
                    "current.role_handoffs must contain exactly: "
                    + ", ".join(STAGE_ORDER)
                )
            elif stage in VALID_STAGES:
                current_index = STAGE_ORDER.index(stage)
                for index, role in enumerate(STAGE_ORDER):
                    value = role_handoffs.get(role)
                    if index < current_index:
                        if not isinstance(value, str) or not value:
                            errors.append(
                                f"completed role {role!r} requires a handoff id"
                            )
                        elif f"## {value}" not in handoffs_text:
                            errors.append(
                                f"role handoff {value!r} for {role!r} is absent"
                            )
                    elif value is not None:
                        errors.append(
                            f"current or future role {role!r} must have null handoff"
                        )

    scope = current.get("scope")
    if not isinstance(scope, Mapping):
        errors.append("current.scope must be a mapping")
    else:
        allowed_paths = scope.get("allowed_paths")
        if not isinstance(allowed_paths, list) or not allowed_paths:
            errors.append("current.scope.allowed_paths must be a non-empty list")
        else:
            for value in allowed_paths:
                error = _validate_relative_allowed_path(value)
                if error:
                    errors.append(error)
        forbidden_paths = scope.get("forbidden_paths")
        if not isinstance(forbidden_paths, list) or not forbidden_paths:
            errors.append("current.scope.forbidden_paths must be a non-empty list")
        elif isinstance(allowed_paths, list):
            normalized_allowed = {str(value).replace("\\", "/") for value in allowed_paths}
            normalized_forbidden = {
                str(value).replace("\\", "/") for value in forbidden_paths
            }
            overlap = sorted(normalized_allowed & normalized_forbidden)
            if overlap:
                errors.append(
                    "allowed_paths and forbidden_paths overlap: " + ", ".join(overlap)
                )

    acceptance = current.get("acceptance")
    if not isinstance(acceptance, list) or not acceptance:
        errors.append("current.acceptance must be a non-empty list")

    required_commands = current.get("required_commands")
    if not isinstance(required_commands, list) or not required_commands:
        errors.append("current.required_commands must be a non-empty list")

    baseline = state.get("baseline")
    if not isinstance(baseline, Mapping):
        errors.append("baseline must be a mapping")
    else:
        result = baseline.get("result")
        if not isinstance(result, str) or not re.fullmatch(
            r"\d+ passed, \d+ skipped", result
        ):
            errors.append(
                "baseline.result must use the exact '<N> passed, <M> skipped' format"
            )

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Agent state validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    state = _load_yaml(STATE_PATH)
    current = state["current"]
    print(
        "Agent state OK: "
        f"task={current['task_id']} "
        f"stage={current['stage']} "
        f"status={current['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

