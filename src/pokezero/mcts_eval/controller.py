"""Resumable stage controller (plan D6/B2).

The controller's entire state is the set of persisted artifacts: every stage
writes its output atomically and writes a completion MARKER LAST, containing the
identity that stage is scoped to. On restart it validates and reuses complete
stages, then resumes the first incomplete one. Nothing is held in memory across
a restart, so a killed submitter loses no completed work.

Identity scoping (plan section 2.1) is what makes a concurrency change cheap:

    stages 1-4  -> experiment_id  (survive a resource-profile change)
    stages 5-9  -> execution_id   (re-run under a new resource profile)

A marker whose identity does not match at its own scope is stale: it is never
reused, and a *terminal* failure is never retried at all.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

MARKER_NAME = "_complete.json"
STATUS_NAME = "status.json"


class Stage(str, Enum):
    MATERIALIZE_CHECKPOINT = "materialize-checkpoint"
    VALIDATE_CONTRACT = "validate-contract"
    MECHANICS_SMOKE = "mechanics-smoke"
    BUILD_TIMING_CORPUS = "build-or-validate-timing-corpus"
    RUN_TIMING_LATTICE = "run-timing-lattice"
    MERGE_AND_SELECT = "merge-timing-and-select-candidates"
    RUN_FOULPLAY_MATRIX = "run-foulplay-matrix"
    MERGE_STRENGTH = "merge-strength-results"
    PUBLISH_REPORT = "publish-report"


STAGE_ORDER: tuple[Stage, ...] = tuple(Stage)
# Stages 1-4 are experiment-scoped; 5-9 execution-scoped (plan section 2.1).
EXPERIMENT_SCOPED: frozenset[Stage] = frozenset(STAGE_ORDER[:4])


class TerminalFailure(RuntimeError):
    """Never retried: provenance mismatch, unsupported contract, invalid matrix,
    conflicting duplicate result, deterministic engine failure, artifact error."""


class RetryableFailure(RuntimeError):
    """Transient scheduling/transport/storage/opponent failure — bounded retry."""


def stage_scope(stage: Stage) -> str:
    return "experiment_id" if stage in EXPERIMENT_SCOPED else "execution_id"


@dataclass(frozen=True)
class StageMarker:
    stage: str
    identity_field: str
    identity: str
    artifacts: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = list(self.artifacts)
        return payload


def stage_dir(root: str | Path, stage: Stage) -> Path:
    return Path(root) / stage.value


def write_marker(
    root: str | Path,
    stage: Stage,
    *,
    experiment_id: str,
    execution_id: str,
    artifacts: Sequence[str] = (),
) -> Path:
    """Write the completion marker LAST, atomically."""
    identity = experiment_id if stage in EXPERIMENT_SCOPED else execution_id
    marker = StageMarker(
        stage=stage.value,
        identity_field=stage_scope(stage),
        identity=identity,
        artifacts=tuple(artifacts),
    )
    directory = stage_dir(root, stage)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MARKER_NAME
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(marker.to_payload(), sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)
    return path


def read_marker(root: str | Path, stage: Stage) -> StageMarker | None:
    path = stage_dir(root, stage) / MARKER_NAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return StageMarker(
        stage=str(payload["stage"]),
        identity_field=str(payload["identity_field"]),
        identity=str(payload["identity"]),
        artifacts=tuple(payload.get("artifacts") or ()),
    )


def stage_is_complete(
    root: str | Path, stage: Stage, *, experiment_id: str, execution_id: str
) -> bool:
    """Complete only when the marker exists AND its identity matches at this
    stage's own scope, AND every artifact it claims still exists."""
    marker = read_marker(root, stage)
    if marker is None:
        return False
    expected = experiment_id if stage in EXPERIMENT_SCOPED else execution_id
    if marker.identity != expected or marker.identity_field != stage_scope(stage):
        return False
    return all(Path(artifact).exists() for artifact in marker.artifacts)


def next_incomplete_stage(
    root: str | Path, *, experiment_id: str, execution_id: str
) -> Stage | None:
    for stage in STAGE_ORDER:
        if not stage_is_complete(
            root, stage, experiment_id=experiment_id, execution_id=execution_id
        ):
            return stage
    return None


def write_status(
    root: str | Path,
    *,
    experiment_id: str,
    execution_id: str,
    stage: Stage | None,
    completed_tasks: int = 0,
    total_tasks: int = 0,
    retries: Mapping[str, int] | None = None,
    artifacts: Sequence[str] = (),
    terminal_failure: str | None = None,
) -> Path:
    """Machine-readable progress. Notifications CONSUME this artifact; they are
    never required for progress (plan B2)."""
    payload = {
        "experiment_id": experiment_id,
        "execution_id": execution_id,
        "stage": stage.value if stage else None,
        "state": "failed" if terminal_failure else ("complete" if stage is None else "running"),
        "completed_tasks": completed_tasks,
        "total_tasks": total_tasks,
        "retries": dict(retries or {}),
        "artifacts": list(artifacts),
        "terminal_failure": terminal_failure,
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    path = root_path / STATUS_NAME
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    return path


def run_pipeline(
    root: str | Path,
    *,
    experiment_id: str,
    execution_id: str,
    handlers: Mapping[Stage, Callable[[Path], Sequence[str]]],
    max_retries: int = 3,
) -> dict[str, Any]:
    """Drive stages in order, reusing complete ones and resuming the first gap.

    A handler returns the artifact paths its stage produced; the controller
    writes the marker afterwards, so a crash mid-handler leaves the stage
    incomplete and it re-runs cleanly. Retryable failures back off within the
    same run and shard id; terminal failures stop the run and are published.
    """
    root_path = Path(root)
    retries: dict[str, int] = {}
    for stage in STAGE_ORDER:
        if stage_is_complete(
            root_path, stage, experiment_id=experiment_id, execution_id=execution_id
        ):
            continue
        handler = handlers.get(stage)
        if handler is None:
            continue
        write_status(
            root_path,
            experiment_id=experiment_id,
            execution_id=execution_id,
            stage=stage,
            retries=retries,
        )
        attempt = 0
        while True:
            try:
                directory = stage_dir(root_path, stage)
                directory.mkdir(parents=True, exist_ok=True)
                artifacts = handler(directory)
                break
            except TerminalFailure as error:
                write_status(
                    root_path,
                    experiment_id=experiment_id,
                    execution_id=execution_id,
                    stage=stage,
                    retries=retries,
                    terminal_failure=f"{stage.value}: {error}",
                )
                raise
            except RetryableFailure:
                attempt += 1
                retries[stage.value] = attempt
                if attempt >= max_retries:
                    write_status(
                        root_path,
                        experiment_id=experiment_id,
                        execution_id=execution_id,
                        stage=stage,
                        retries=retries,
                        terminal_failure=f"{stage.value}: retry budget exhausted",
                    )
                    raise TerminalFailure(
                        f"{stage.value}: exhausted {max_retries} retries"
                    ) from None
        write_marker(
            root_path,
            stage,
            experiment_id=experiment_id,
            execution_id=execution_id,
            artifacts=artifacts,
        )
    write_status(
        root_path,
        experiment_id=experiment_id,
        execution_id=execution_id,
        stage=None,
        retries=retries,
    )
    return json.loads((root_path / STATUS_NAME).read_text(encoding="utf-8"))
