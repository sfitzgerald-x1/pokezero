#!/usr/bin/env python3
"""Run a frozen, fresh, mirrored MCTS-versus-MCTS comparison.

The manifest is intentionally small but complete enough to be an experiment
contract: it names two explicit model-leaf MCTS configurations, one checkpoint
digest, source/build identities, a seed list, and the decision cap.  A
source-different incumbent is allowed only through the explicit isolated-build
mode, where it serves decisions from its declared source-bound process.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero.mcts_eval.head_to_head import (  # noqa: E402
    HeadToHeadError,
    MctsPolicySpec,
    PublicOnlyMctsPolicy,
    complete_pair,
    load_pair,
    play_mirrored_pair,
    summarize_complete_pairs,
    write_game_immutable,
)
from pokezero.mcts_eval.scoring import bootstrap_indices, bootstrap_mean  # noqa: E402
from pokezero.engine_search import OPPONENT_REQUEST_ORDER_STATUS_VALUES  # noqa: E402


MANIFEST_SCHEMA_VERSION = "pokezero.mcts-h2h-manifest.v1"
COMPLETE_SCHEMA_VERSION = "pokezero.mcts-h2h-complete.v1"
PROGRESS_SCHEMA_VERSION = "pokezero.mcts-h2h-progress.v1"
BACKUP_REPAIR_PILOT_SCHEMA_VERSION = "pokezero.mcts-h2h-backup-repair-pilot.v4"
BACKUP_REPAIR_PILOT_READOUT_SCHEMA_VERSION = "pokezero.mcts-h2h-backup-repair-pilot-readout.v2"
OPPONENT_PRIOR_APPLICABILITY_SCHEMA_VERSION = (
    "pokezero.mcts-h2h-opponent-prior-applicability.v1"
)
OPPONENT_PRIOR_APPLICABILITY_READOUT_SCHEMA_VERSION = (
    "pokezero.mcts-h2h-opponent-prior-applicability-readout.v2"
)
OPPONENT_PRIOR_STRENGTH_PILOT_READOUT_SCHEMA_VERSION = (
    "pokezero.mcts-h2h-opponent-prior-strength-pilot-readout.v1"
)
OPPONENT_PRIOR_STRENGTH_PILOT_DECISIONS = (
    "ELIGIBLE_FOR_SEPARATE_CONFIRMATION_REGISTRATION",
    "NO_EXTENSION",
)
MODEL_WORLD_PARALLELISM_PILOT_SCHEMA_VERSION = (
    "pokezero.mcts-h2h-model-world-parallelism-pilot.v1"
)
MODEL_WORLD_PARALLELISM_PILOT_READOUT_SCHEMA_VERSION = (
    "pokezero.mcts-h2h-model-world-parallelism-pilot-readout.v1"
)
MODEL_WORLD_PARALLELISM_PILOT_DECISIONS = (
    "ELIGIBLE_FOR_SEPARATE_CONFIRMATION_REGISTRATION",
    "NO_EXTENSION",
)
OWN_POLICY_PRIOR_STUDY_SCHEMA_VERSION = "pokezero.mcts-h2h-own-policy-prior-study.v1"
OWN_POLICY_PRIOR_STUDY_READOUT_SCHEMA_VERSION = (
    "pokezero.mcts-h2h-own-policy-prior-study-readout.v1"
)
OWN_POLICY_PRIOR_STUDY_FAILURE_RETRY_POLICY = {
    "schema_version": "pokezero.mcts-h2h-failure-retry-policy.v1",
    "interrupted_before_runner_terminal": "resume_same_root_with_fresh_launcher_attempt",
    "nonzero_runner_exit": "terminal_failed_no_retry",
    "malformed_runner_terminal": "nonbankable_no_retry",
    "completed_game_units": "immutable_reuse_only",
}
OWN_POLICY_PRIOR_STUDY_DECISIONS = (
    "PREFLIGHT_PASS",
    "PREFLIGHT_NONPASS",
    "USE_GUIDED_MCTS_AS_RESEARCH_BASELINE",
    "GUIDANCE_HARMED_THIS_CONFIGURATION",
    "TARGET_SIZED_BENEFIT_RULED_OUT",
    "INCONCLUSIVE",
    "SHARD_COMPLETE",
)
MODEL_WORLD_PARALLELISM_PILOT_PAIRS = 8
MODEL_WORLD_PARALLELISM_PILOT_BOOTSTRAP_RESAMPLES = 10_000
MODEL_WORLD_PARALLELISM_PILOT_CONFIDENCE = 0.80
MODEL_WORLD_PARALLELISM_PILOT_MINIMUM_EFFECT_DELTA = 0.05
MODEL_WORLD_PARALLELISM_PILOT_FAILURE_RETRY_POLICY = {
    "schema_version": "pokezero.mcts-h2h-failure-retry-policy.v1",
    "interrupted_before_runner_terminal": "resume_same_root_with_fresh_launcher_attempt",
    "nonzero_runner_exit": "terminal_failed_no_retry",
    "malformed_runner_terminal": "nonbankable_no_retry",
    "completed_game_units": "immutable_reuse_only",
}
MODEL_WORLD_PARALLELISM_PILOT_DEADLINE_REQUIREMENTS = {
    "expected_decisions": 16,
    "native_batch_guard_ms": 64,
    "requested_ms": 1_000,
    "sims_per_world": 256,
    "worlds": 4,
}
OPPONENT_PRIOR_APPLICABILITY_CONTRACT = {
    "schema_version": OPPONENT_PRIOR_APPLICABILITY_SCHEMA_VERSION,
    "stage": "development_applicability",
    "minimum_candidate_opponent_prior_arm_decisions": 1,
    "maximum_incumbent_opponent_prior_arm_decisions": 0,
    # Interior simulated nodes can lack an authoritative live request after a
    # simulated replacement.  Root fallback is different: it means the actual
    # played decision could not retain its model prior, so it invalidates an
    # applicability result even if another decision used the opponent head.
    "maximum_candidate_root_prior_fallbacks": 0,
    "maximum_incumbent_root_prior_fallbacks": 0,
}

# This is intentionally a one-contrast contract rather than a tunable study
# registry.  The first strength read must isolate the batched-backup repair.
# The native bindings needed for the current PyTorch runtime require fresh,
# reviewed source revisions for *both* policies.  The B2 source-image builder
# deliberately produces depth-one, remote-free checkouts, so runtime admission
# cannot rely on Git ancestry.  Instead, the contract pins the two reviewed
# revisions and carries the static, identical compatibility-delta proof below.
# The runtime pair includes only the same PyTorch compatibility work plus the
# same source-local split-fallback instrumentation on each mechanics baseline.
# The historical runtime pair cannot serve this v2 protocol because it omits
# the root/branch counters; accepting it as scoped zero would be unsound.
# It must never call a generic later source revision the backup-repair
# treatment.
BACKUP_REPAIR_CANDIDATE_BASELINE_COMMIT = "df4e3ce15ee69f922f6ae1b81c7b5e9861828319"
BACKUP_REPAIR_INCUMBENT_BASELINE_COMMIT = "dacb6358d9b145ce069d6718662a38f581a38bc0"
BACKUP_REPAIR_CANDIDATE_RUNTIME_COMMIT = "2198e3c71114aef3f5f4c72f0d2f7d7d649fcbad"
BACKUP_REPAIR_INCUMBENT_RUNTIME_COMMIT = "82a7b555ce5373e939f996644397ba1c59cd8fc8"
BACKUP_REPAIR_RUNTIME_COMPATIBILITY = {
    "schema_version": "pokezero.mcts-h2h-backup-repair-runtime-compatibility.v2",
    "candidate": {
        "runtime_commit": BACKUP_REPAIR_CANDIDATE_RUNTIME_COMMIT,
        "mechanics_baseline_commit": BACKUP_REPAIR_CANDIDATE_BASELINE_COMMIT,
    },
    "incumbent": {
        "runtime_commit": BACKUP_REPAIR_INCUMBENT_RUNTIME_COMMIT,
        "mechanics_baseline_commit": BACKUP_REPAIR_INCUMBENT_BASELINE_COMMIT,
    },
    "allowed_changed_paths": [
        ".github/workflows/engine-fidelity-gates.yml",
        "docs/crate_model_integration.md",
        "rust/pokezero-search/Cargo.lock",
        "rust/pokezero-search/Cargo.toml",
        "rust/pokezero-search/README.md",
        "rust/pokezero-search/src/model.rs",
        "scripts/build_search_crate_model.sh",
        "scripts/public_projection_census.py",
        "src/pokezero/engine_search.py",
    ],
    # The two mechanics baselines differ outside the compatibility paths, so
    # Git's blob-id-bearing `index` lines differ even though the allowed
    # patches are byte-identical. This value is the SHA-256 of `git diff
    # <baseline> <runtime>` with only those `index` lines removed.
    "delta_normalization": "git-diff-with-index-lines-removed.v1",
    "identical_normalized_delta_sha256": "cab10cc252a2779b0dc1c62726390b52fa26a6dac9712a4d577bd94e2397f3d0",
}
BACKUP_REPAIR_PILOT_PAIRS = 12
BACKUP_REPAIR_CONFIRMATION_PAIRS = 50
BACKUP_REPAIR_BOOTSTRAP_RESAMPLES = 10_000
BACKUP_REPAIR_PILOT_CONFIDENCE = 0.80
BACKUP_REPAIR_MINIMUM_EFFECT_DELTA = 0.05
BACKUP_REPAIR_SUPERSEDED_STUDY = {
    "experiment_id": "mcts-backup-repair-pilot-c7f54d7e-20260908-r4",
    "terminal_handoff": "malformed_literal_backslash_n",
}
BACKUP_REPAIR_SUPERSEDED_PILOT_SEEDS = (
    2026090801,
    2026090802,
    2026090803,
    2026090804,
    2026090805,
    2026090806,
    2026090807,
    2026090808,
    2026090809,
    2026090810,
    2026090811,
    2026090812,
)
BACKUP_REPAIR_SUPERSEDED_CONFIRMATION_SEEDS = tuple(range(2026091001, 2026091051))
BACKUP_REPAIR_FAILURE_RETRY_POLICY = {
    "schema_version": "pokezero.mcts-h2h-failure-retry-policy.v1",
    "interrupted_before_runner_terminal": "resume_same_root_with_fresh_launcher_attempt",
    "nonzero_runner_exit": "terminal_failed_no_retry",
    "malformed_runner_terminal": "nonbankable_no_retry",
    "completed_game_units": "immutable_reuse_only",
}
DURABLE_LAUNCHER_ATTEMPT_SCHEMA_VERSION = "pokezero.mcts-h2h-launcher-attempt.v1"
DURABLE_LAUNCHER_ATTEMPT_ID_ENV = "POKEZERO_DURABLE_LAUNCHER_ATTEMPT_ID"
DURABLE_LAUNCHER_OUT_DIR_ENV = "POKEZERO_DURABLE_LAUNCHER_OUT_DIR"
DURABLE_LAUNCHER_ATTEMPT_RECEIPT_ENV = "POKEZERO_DURABLE_LAUNCHER_ATTEMPT_RECEIPT"
DURABLE_LAUNCHER_WRITER_LOCK_FD_ENV = "POKEZERO_DURABLE_LAUNCHER_WRITER_LOCK_FD"
RUNNER_TERMINAL_NAME = "runner-terminal.json"
ISOLATED_CONFIG_COMPATIBILITY_PROTOCOL = "disabled-diagnostic-omission.v1"
ISOLATED_WORKER_RESPONSE_TIMEOUT_SECONDS = 180.0
ISOLATED_DISABLED_DIAGNOSTIC_COMPATIBILITY_DEFAULTS = {
    "root_selector_q": False,
    "root_selector_shadow": False,
    # Older frozen sources predate this deadline control.  None is the
    # disabled setting and retains their fixed-work MCTS behavior exactly.
    "model_decision_time_ms": None,
    # A zero native-batch guard also retains the historical deadline behavior.
    # Any positive guard is a different scheduling policy and cannot be
    # silently omitted from a source-isolated comparison.
    "model_native_batch_guard_ms": 0,
    "model_world_workers": 1,
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_source_files(repo_root: Path, paths: list[Path]) -> str:
    """Hash source files with their repository-relative names and bytes."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(repo_root).as_posix()):
        relative = path.relative_to(repo_root).as_posix()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _source_provenance() -> dict[str, str]:
    """Bind Python/runtime source to a clean checkout or an explicit content receipt."""

    stamped = os.environ.get("POKEZERO_COMMIT", "").strip().lower()
    git_metadata = REPO_ROOT / ".git"
    if git_metadata.exists():
        try:
            head = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip().lower()
            dirty = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            tracked = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
                check=True,
                capture_output=True,
            ).stdout.split(b"\0")
        except (OSError, subprocess.CalledProcessError) as error:
            raise HeadToHeadError(
                f"cannot verify public source checkout before a scored comparison: {error}"
            ) from error
        if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
            raise HeadToHeadError("public source checkout HEAD is not a full lowercase Git commit id.")
        if stamped and stamped != head:
            raise HeadToHeadError(
                f"POKEZERO_COMMIT={stamped} does not match the executing checkout {head}."
            )
        if dirty:
            changed = ", ".join(line[3:].strip() for line in dirty.splitlines()[:8])
            raise HeadToHeadError(
                "refusing to score a dirty tracked public source tree"
                f" ({changed or 'tracked changes detected'})."
            )
        paths = [
            REPO_ROOT / value.decode("utf-8")
            for value in tracked
            if value and (REPO_ROOT / value.decode("utf-8")).is_file()
        ]
        return {
            "commit": head,
            "tree_sha256": _hash_source_files(REPO_ROOT, paths),
            "tree_status": "clean_tracked_checkout",
        }

    if len(stamped) != 40 or any(character not in "0123456789abcdef" for character in stamped):
        raise HeadToHeadError(
            "an image without .git must set POKEZERO_COMMIT to its full lowercase source commit."
        )
    roots = (REPO_ROOT / "src" / "pokezero", REPO_ROOT / "scripts")
    paths = [
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    # LocalShowdownEnv launches scripts/battle_bridge.mjs, which imports the
    # sibling request-boundary module. Include every in-repository JS module so
    # the image receipt covers that executable battle seam as well as Python.
    paths.extend(
        path
        for path in (REPO_ROOT / "scripts").glob("*.mjs")
        if path.is_file()
    )
    pyproject = REPO_ROOT / "pyproject.toml"
    if pyproject.is_file():
        paths.append(pyproject)
    if not paths:
        raise HeadToHeadError("cannot hash executable source in image without .git.")
    return {
        "commit": stamped,
        "tree_sha256": _hash_source_files(REPO_ROOT, paths),
        "tree_status": "explicit_hash_without_git_python_and_bridge",
    }


def _showdown_dependency_paths(root: Path) -> list[Path]:
    """Return every built Showdown byte the Gen 3 battle oracle can load."""

    required = (
        root / "dist" / "sim" / "index.js",
        root / "dist" / "sim" / "dex.js",
        root / "dist" / "sim" / "dex-data.js",
        root / "dist" / "data" / "moves.js",
        root / "dist" / "data" / "pokedex.js",
        root / "dist" / "data" / "typechart.js",
        root / "dist" / "data" / "abilities.js",
        root / "dist" / "data" / "items.js",
        root / "dist" / "data" / "mods" / "gen3" / "moves.js",
        root / "dist" / "data" / "mods" / "gen3" / "scripts.js",
        root / "dist" / "data" / "mods" / "gen3" / "abilities.js",
        root / "dist" / "data" / "mods" / "gen3" / "items.js",
        root / "data" / "random-battles" / "gen3" / "sets.json",
        root / "dist" / "data" / "random-battles" / "gen3" / "teams.js",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise HeadToHeadError(
            "cannot bind Showdown source; required runtime input is missing: "
            f"{missing[0].relative_to(root)}."
        )
    paths = set(required)
    # Dex.forGen(3) can resolve inherited mod layers through the built simulator.
    # Hash every built source byte, not a brittle hand-maintained transitive-load list.
    paths.update((root / "dist").rglob("*.js"))
    paths.update((root / "dist").rglob("*.json"))
    return sorted(path for path in paths if path.is_file())


def _showdown_source_provenance(showdown_root: str | Path) -> dict[str, Any]:
    """Require a clean checkout and bind all runtime-relevant Showdown content."""

    root = Path(showdown_root).expanduser().resolve()
    try:
        top_level = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ).resolve()
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise HeadToHeadError(
            f"cannot bind Showdown source identity from {root}: {error}"
        ) from error
    if top_level != root:
        raise HeadToHeadError("--showdown-root must be the root of its Showdown Git checkout.")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise HeadToHeadError("Showdown HEAD is not a full lowercase Git commit id.")
    if dirty:
        raise HeadToHeadError(
            "Showdown checkout is dirty; commit or restore it before an MCTS-vs-MCTS result."
        )
    digest = hashlib.sha256()
    for path in _showdown_dependency_paths(root):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return {"content_sha256": digest.hexdigest(), "git_commit": commit, "git_clean": True}


def _declared_showdown_source_sha256(manifest: Mapping[str, Any]) -> str:
    value = str(manifest.get("showdown_source_sha256", ""))
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise HeadToHeadError(
            "manifest.showdown_source_sha256 must be the 64-character hash of the "
            "clean Showdown runtime inputs."
        )
    return value


def _declared_source_tree_sha256(manifest: Mapping[str, Any]) -> str:
    value = str(manifest.get("source_tree_sha256", ""))
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise HeadToHeadError(
            "manifest.source_tree_sha256 must be the 64-character hash of the executing "
            "PokeZero source/image receipt."
        )
    return value


def _durable_output_root(value: str | Path) -> Path:
    """Keep resumable evidence outside the source tree that must remain clean."""

    root = Path(value).expanduser().resolve()
    try:
        root.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return root
    raise HeadToHeadError(
        "--out-dir must be outside the PokeZero source checkout; write durable games to "
        "/shared (or another external evidence root) so a resumed result never dirties its "
        "own source identity."
    )


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create ``path`` or require byte-identical prior contents."""

    path.parent.mkdir(parents=True, exist_ok=True)
    expected = _canonical_bytes(payload)
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing != expected:
            raise HeadToHeadError(
                f"refusing to replace extant immutable artifact {path}; its contents differ."
            )
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != expected:
                raise HeadToHeadError(
                    f"refusing to replace concurrently-created artifact {path}; it differs."
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _write_progress_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically update the runner's explicitly mutable liveness checkpoint.

    Game, receipt, summary, and terminal artifacts are immutable. This one
    progress path is deliberately different: it is a non-score-bearing status
    snapshot that advances after each committed decision. An unfamiliar or
    malformed existing file fails closed rather than being overwritten.
    """

    if path.parent.name != "progress" or path.name != "current.json":
        raise HeadToHeadError(f"progress checkpoint has an unexpected path: {path}")
    if payload.get("schema_version") != PROGRESS_SCHEMA_VERSION:
        raise HeadToHeadError("progress checkpoint has the wrong schema version.")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        existing = None
    except (OSError, json.JSONDecodeError) as error:
        raise HeadToHeadError(f"refusing to replace unreadable progress checkpoint {path}: {error}") from error
    if existing is not None:
        if not isinstance(existing, Mapping) or existing.get("schema_version") != PROGRESS_SCHEMA_VERSION:
            raise HeadToHeadError(f"refusing to replace incompatible progress checkpoint {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _isolated_worker_stderr_path(
    out_root: Path,
    *,
    attempt_id: str,
    seed: int,
    candidate_seat: str,
    role: str,
) -> Path:
    """Name a child log below an invocation-unique durable attempt directory.

    A resumed process may inherit the same PID as an interrupted one (notably
    PID 1 in a replacement container).  The child transport correctly creates
    its stderr log exclusively, so the invocation nonce—not that PID—must
    distinguish retry logs.  Prior logs are therefore retained as evidence,
    while the replacement worker always has a fresh path.
    """

    if len(attempt_id) != 32 or any(character not in "0123456789abcdef" for character in attempt_id):
        raise HeadToHeadError("isolated worker attempt id must be a lowercase UUID hex value.")
    if role not in {"candidate", "incumbent"}:
        raise HeadToHeadError(f"unknown isolated worker role {role!r}.")
    if candidate_seat not in {"p1", "p2"}:
        raise HeadToHeadError(f"unknown isolated candidate seat {candidate_seat!r}.")
    return (
        out_root
        / "worker-stderr"
        / f"attempt-{attempt_id}"
        / f"seed-{seed}-{candidate_seat}-{role}.log"
    )


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HeadToHeadError(f"{label} must be a JSON object.")
    return value


def _validated_deadline_totals(
    summary: Mapping[str, Any], *, role: str
) -> tuple[int, int]:
    """Derive audit totals from the validator's normalized decision records.

    The sealed decision artifact nests native invocations beneath
    ``engine_mcts.time_budget``.  ``validate_deadline_qualification`` checks
    that raw shape and returns a deliberately flatter, normalized ``decisions``
    list.  Consume that validated list here rather than duplicating the raw
    artifact path: this makes the H2H admission audit follow the exact schema
    that was used to recompute the immutable PASS summary.
    """

    decisions = summary.get("decisions")
    if not isinstance(decisions, list):
        raise HeadToHeadError(
            f"{role} deadline qualification recomputed decisions are malformed."
        )
    completed_iterations_total = 0
    worlds_searched_total = 0
    for decision_index, decision in enumerate(decisions):
        record = _mapping(
            decision,
            label=f"{role} deadline qualification recomputed decision {decision_index}",
        )
        worlds_searched = record.get("worlds_searched")
        invocations = record.get("native_invocations")
        if (
            isinstance(worlds_searched, bool)
            or not isinstance(worlds_searched, int)
            or worlds_searched < 0
            or not isinstance(invocations, list)
        ):
            raise HeadToHeadError(
                f"{role} deadline qualification recomputed decision {decision_index} is malformed."
            )
        worlds_searched_total += worlds_searched
        for invocation_index, invocation in enumerate(invocations):
            witness = _mapping(
                invocation,
                label=(
                    f"{role} deadline qualification recomputed decision "
                    f"{decision_index} invocation {invocation_index}"
                ),
            )
            completed_iterations = witness.get("completed_iterations")
            if (
                isinstance(completed_iterations, bool)
                or not isinstance(completed_iterations, int)
                or completed_iterations < 0
            ):
                raise HeadToHeadError(
                    f"{role} deadline qualification recomputed decision "
                    f"{decision_index} invocation {invocation_index} is malformed."
                )
            completed_iterations_total += completed_iterations
    return completed_iterations_total, worlds_searched_total


def _load_manifest(path: str | Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HeadToHeadError(f"cannot read MCTS-vs-MCTS manifest: {error}") from error
    manifest = _mapping(payload, label="manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise HeadToHeadError(
            f"manifest schema {manifest.get('schema_version')!r} is not "
            f"{MANIFEST_SCHEMA_VERSION!r}."
        )
    return manifest


def _runtime_spec(
    raw: Mapping[str, Any],
    *,
    role: str,
    checkpoint: str,
    checkpoint_sha256: str,
    source_commit: str,
    source_tree_sha256: str,
    engine_fingerprint: str,
    showdown_source_sha256: str,
    model_path: str,
    tables_path: str,
    device: str,
) -> tuple[MctsPolicySpec, Any]:
    """Build and freeze one model-leaf policy configuration from the manifest."""

    from pokezero.engine_search import EngineMctsConfig

    for field, actual in (
        ("source_commit", source_commit),
        ("engine_fingerprint", engine_fingerprint),
        ("checkpoint_sha256", checkpoint_sha256),
    ):
        declared = str(raw.get(field, ""))
        if declared != actual:
            raise HeadToHeadError(
                f"{role}.{field}={declared!r} does not match the active verified value {actual!r}."
            )
    settings = dict(_mapping(raw.get("engine_config"), label=f"{role}.engine_config"))
    if settings.pop("leaf_eval", None) != "model":
        raise HeadToHeadError(f"{role} must explicitly declare engine_config.leaf_eval='model'.")
    if settings.pop("strict_fallbacks", None) is not True:
        raise HeadToHeadError(
            f"{role} must explicitly declare engine_config.strict_fallbacks=true."
        )
    forbidden = {"checkpoint_path", "model_path", "tables_path", "model_device"}.intersection(settings)
    if forbidden:
        raise HeadToHeadError(
            f"{role}.engine_config may not override source-bound artifacts: {sorted(forbidden)}."
        )
    try:
        config = EngineMctsConfig(
            leaf_eval="model",
            strict_fallbacks=True,
            checkpoint_path=checkpoint,
            model_path=model_path,
            tables_path=tables_path,
            model_device=device,
            **settings,
        )
    except (TypeError, ValueError) as error:
        raise HeadToHeadError(f"invalid {role}.engine_config: {error}") from error
    frozen = asdict(config)
    try:
        json.dumps(frozen, sort_keys=True)
    except TypeError as error:
        raise HeadToHeadError(f"{role} configuration is not serializable: {error}") from error
    spec = MctsPolicySpec(
        config_id=str(raw.get("config_id", "")),
        policy_id=str(raw.get("policy_id", "")),
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        engine_fingerprint=engine_fingerprint,
        checkpoint_sha256=checkpoint_sha256,
        showdown_source_sha256=showdown_source_sha256,
        config=frozen,
    )
    return spec, config


def _seeds(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    values = manifest.get("seeds")
    if not isinstance(values, list) or not values:
        raise HeadToHeadError("manifest.seeds must be a non-empty JSON list.")
    seeds = tuple(int(value) for value in values)
    if len(set(seeds)) != len(seeds):
        raise HeadToHeadError("manifest.seeds contains a duplicate; each mirrored pair needs one key.")
    return seeds


def _bootstrap_settings(manifest: Mapping[str, Any]) -> tuple[int, int, float]:
    """Read the complete, predeclared bootstrap contract from a manifest."""
    bootstrap = _mapping(manifest.get("bootstrap"), label="manifest.bootstrap")
    try:
        resamples = int(bootstrap.get("resamples", 0))
        seed = int(bootstrap.get("seed", -1))
        confidence_level = float(bootstrap.get("confidence_level", 0.95))
    except (TypeError, ValueError) as error:
        raise HeadToHeadError("manifest.bootstrap values are malformed.") from error
    if resamples <= 0 or seed < 0 or not 0.0 < confidence_level < 1.0:
        raise HeadToHeadError(
            "manifest.bootstrap requires positive resamples, a non-negative seed, "
            "and a confidence level strictly between zero and one."
        )
    return resamples, seed, confidence_level


def _bootstrap_contract(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], int, int, float]:
    """Keep the validated manifest mapping available to downstream contracts."""

    bootstrap = _mapping(manifest.get("bootstrap"), label="manifest.bootstrap")
    resamples, seed, confidence_level = _bootstrap_settings(manifest)
    return bootstrap, resamples, seed, confidence_level


def _replacement_study_requires_durable_launcher(manifest: Mapping[str, Any]) -> bool:
    study = manifest.get("study")
    if isinstance(study, Mapping) and study.get("schema_version") == BACKUP_REPAIR_PILOT_SCHEMA_VERSION:
        return True
    world_parallelism = manifest.get("world_parallelism_pilot")
    return (
        isinstance(world_parallelism, Mapping)
        and world_parallelism.get("schema_version")
        == MODEL_WORLD_PARALLELISM_PILOT_SCHEMA_VERSION
    )


def _require_durable_launcher_handoff(out_root: Path) -> None:
    """Reject a v2 replacement run that was not launched with the durable handoff.

    A direct scorer invocation could otherwise resume a root after a terminal
    failure or malformed handoff and violate the contract's declared no-retry
    rule. The launcher owns its terminal record and writer lease; the scorer
    proves it received the matching immutable attempt context before any model
    or battle work begins.
    """

    terminal = out_root / RUNNER_TERMINAL_NAME
    if terminal.exists():
        raise HeadToHeadError(
            f"replacement study root already has a terminal handoff at {terminal}; no retry is permitted."
        )
    attempt_id = os.environ.get(DURABLE_LAUNCHER_ATTEMPT_ID_ENV, "")
    if not attempt_id or len(attempt_id) > 64 or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in attempt_id
    ):
        raise HeadToHeadError(
            "replacement study requires a safe durable-launcher attempt id."
        )
    expected_root = str(out_root)
    if os.environ.get(DURABLE_LAUNCHER_OUT_DIR_ENV) != expected_root:
        raise HeadToHeadError(
            "replacement study durable-launcher output root does not match --out-dir."
        )
    expected_receipt = out_root / "launcher-attempts" / f"{attempt_id}.json"
    supplied_receipt = os.environ.get(DURABLE_LAUNCHER_ATTEMPT_RECEIPT_ENV, "")
    if supplied_receipt != str(expected_receipt):
        raise HeadToHeadError(
            "replacement study durable-launcher receipt does not match its root and attempt."
        )
    lock_path = out_root / "runner-writer.lock"
    lock_fd_raw = os.environ.get(DURABLE_LAUNCHER_WRITER_LOCK_FD_ENV, "")
    try:
        lock_fd = int(lock_fd_raw)
    except ValueError as error:
        raise HeadToHeadError(
            "replacement study requires the inherited durable-launcher writer-lock fd."
        ) from error
    if lock_fd < 0 or str(lock_fd) != lock_fd_raw:
        raise HeadToHeadError(
            "replacement study requires the inherited durable-launcher writer-lock fd."
        )
    try:
        descriptor_status = os.fstat(lock_fd)
        path_status = lock_path.stat()
    except OSError as error:
        raise HeadToHeadError(
            f"replacement study cannot access its inherited writer-lock fd: {error}"
        ) from error
    if (
        descriptor_status.st_dev != path_status.st_dev
        or descriptor_status.st_ino != path_status.st_ino
    ):
        raise HeadToHeadError(
            "replacement study inherited writer-lock fd does not bind its output root."
        )
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        raise HeadToHeadError(
            f"replacement study cannot acquire its inherited writer lease: {error}"
        ) from error
    try:
        receipt = json.loads(expected_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HeadToHeadError(
            f"replacement study cannot read its durable-launcher attempt receipt: {error}"
        ) from error
    if not isinstance(receipt, Mapping):
        raise HeadToHeadError("replacement study durable-launcher receipt is not an object.")
    runner_script = Path(__file__).resolve()
    if (
        receipt.get("schema_version") != DURABLE_LAUNCHER_ATTEMPT_SCHEMA_VERSION
        or receipt.get("attempt_id") != attempt_id
        or receipt.get("runner_script") != str(runner_script)
        or receipt.get("runner_script_sha256") != _sha256_file(runner_script)
        or receipt.get("writer_lock") != str(lock_path)
    ):
        raise HeadToHeadError(
            "replacement study durable-launcher receipt does not bind this scorer and root."
        )


def _backup_repair_pilot_contract(
    manifest: Mapping[str, Any],
    *,
    seeds: tuple[int, ...],
    bootstrap: Mapping[str, Any],
    candidate_raw: Mapping[str, Any],
    incumbent_raw: Mapping[str, Any],
    candidate_config: Any,
    incumbent_config: Any,
) -> dict[str, Any] | None:
    """Validate the fixed first-strength-pilot contract before it can run.

    Generic MCTS-vs-MCTS manifests remain useful for diagnostics.  A manifest
    opting into this schema, however, receives a deliberately narrow contract:
    corrected batched backups versus the frozen predecessor, the exact reviewed
    runtime pair and its constrained compatibility delta, a fresh pilot roster
    and separately reserved confirmation roster, equal configured work, a
    non-bankable predecessor record, and an executable failure/retry policy.
    This prevents a result from being retrospectively called the backup-repair
    pilot after configuration, source, seed, failure-handling, or analysis drift.
    """

    study = manifest.get("study")
    if study is None:
        return None
    payload = _mapping(study, label="manifest.study")
    if payload.get("schema_version") != BACKUP_REPAIR_PILOT_SCHEMA_VERSION:
        raise HeadToHeadError(
            "manifest.study is not the registered backup-repair pilot schema."
        )
    if payload.get("stage") != "pilot":
        raise HeadToHeadError("backup-repair study must declare stage='pilot'.")
    if payload.get("replaces_nonbankable_study") != BACKUP_REPAIR_SUPERSEDED_STUDY:
        raise HeadToHeadError(
            "backup-repair pilot must explicitly supersede the recorded non-bankable study."
        )
    if payload.get("failure_retry_policy") != BACKUP_REPAIR_FAILURE_RETRY_POLICY:
        raise HeadToHeadError(
            "backup-repair pilot must use the exact predeclared failure/retry policy."
        )
    if payload.get("runtime_compatibility") != BACKUP_REPAIR_RUNTIME_COMPATIBILITY:
        raise HeadToHeadError(
            "backup-repair pilot must declare the exact reviewed runtime compatibility delta."
        )
    candidate_commit = str(candidate_raw.get("source_commit", ""))
    incumbent_commit = str(incumbent_raw.get("source_commit", ""))
    if candidate_commit != BACKUP_REPAIR_CANDIDATE_RUNTIME_COMMIT:
        raise HeadToHeadError(
            "backup-repair pilot candidate must use the exact reviewed corrected-backup runtime commit."
        )
    if incumbent_commit != BACKUP_REPAIR_INCUMBENT_RUNTIME_COMMIT:
        raise HeadToHeadError(
            "backup-repair pilot incumbent must use the exact reviewed predecessor runtime commit."
        )
    if str(candidate_raw.get("config_id", "")) == str(incumbent_raw.get("config_id", "")):
        raise HeadToHeadError(
            "backup-repair pilot candidate and incumbent require distinct config_id values."
        )
    if asdict(candidate_config) != asdict(incumbent_config):
        raise HeadToHeadError(
            "backup-repair pilot policies must use identical EngineMctsConfig values; "
            "the repair source is the only treatment."
        )
    if len(seeds) != BACKUP_REPAIR_PILOT_PAIRS:
        raise HeadToHeadError(
            f"backup-repair pilot requires exactly {BACKUP_REPAIR_PILOT_PAIRS} mirrored pairs."
        )
    if int(bootstrap.get("resamples", 0)) != BACKUP_REPAIR_BOOTSTRAP_RESAMPLES:
        raise HeadToHeadError(
            "backup-repair pilot requires exactly 10,000 bootstrap resamples."
        )
    if float(bootstrap.get("confidence_level", -1.0)) != BACKUP_REPAIR_PILOT_CONFIDENCE:
        raise HeadToHeadError(
            "backup-repair pilot requires its predeclared 80% bootstrap interval."
        )
    if float(payload.get("minimum_effect_delta", -1.0)) != BACKUP_REPAIR_MINIMUM_EFFECT_DELTA:
        raise HeadToHeadError(
            "backup-repair pilot requires its predeclared +0.05 minimum effect."
        )
    confirmation_raw = payload.get("reserved_confirmation_seeds")
    if not isinstance(confirmation_raw, list):
        raise HeadToHeadError(
            "backup-repair pilot must reserve its disjoint confirmation seed roster."
        )
    try:
        confirmation = tuple(int(value) for value in confirmation_raw)
    except (TypeError, ValueError) as error:
        raise HeadToHeadError("backup-repair confirmation seeds must be integer values.") from error
    if len(confirmation) != BACKUP_REPAIR_CONFIRMATION_PAIRS or len(set(confirmation)) != len(
        confirmation
    ):
        raise HeadToHeadError(
            f"backup-repair pilot must reserve exactly {BACKUP_REPAIR_CONFIRMATION_PAIRS} "
            "unique confirmation seeds."
        )
    overlap = sorted(set(seeds).intersection(confirmation))
    if overlap:
        raise HeadToHeadError(
            "backup-repair pilot and confirmation rosters overlap: "
            + ", ".join(str(value) for value in overlap[:8])
        )
    superseded = set(BACKUP_REPAIR_SUPERSEDED_PILOT_SEEDS).union(
        BACKUP_REPAIR_SUPERSEDED_CONFIRMATION_SEEDS
    )
    reused = sorted(set(seeds).union(confirmation).intersection(superseded))
    if reused:
        raise HeadToHeadError(
            "backup-repair replacement rosters reuse seeds reserved by the non-bankable study: "
            + ", ".join(str(value) for value in reused[:8])
        )
    return {
        "schema_version": BACKUP_REPAIR_PILOT_SCHEMA_VERSION,
        "stage": "pilot",
        "candidate_runtime_commit": candidate_commit,
        "incumbent_runtime_commit": incumbent_commit,
        "runtime_compatibility": BACKUP_REPAIR_RUNTIME_COMPATIBILITY,
        "pilot_seeds": list(seeds),
        "reserved_confirmation_seeds": list(confirmation),
        "bootstrap_resamples": BACKUP_REPAIR_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": int(bootstrap["seed"]),
        "confidence_level": BACKUP_REPAIR_PILOT_CONFIDENCE,
        "minimum_effect_delta": BACKUP_REPAIR_MINIMUM_EFFECT_DELTA,
        "replaces_nonbankable_study": dict(BACKUP_REPAIR_SUPERSEDED_STUDY),
        "failure_retry_policy": dict(BACKUP_REPAIR_FAILURE_RETRY_POLICY),
    }


def _backup_repair_pilot_readout(
    *,
    contract: Mapping[str, Any],
    summary: Mapping[str, Any],
    games: list[Any],
) -> dict[str, Any]:
    """Make the frozen pilot decision explicit without hiding a null result."""

    pair_scores_raw = summary.get("pair_scores")
    if not isinstance(pair_scores_raw, list):
        raise HeadToHeadError("completed backup-repair pilot summary has no pair_scores list.")
    try:
        pair_scores = tuple(float(value) for value in pair_scores_raw)
    except (TypeError, ValueError) as error:
        raise HeadToHeadError("completed backup-repair pilot pair scores are malformed.") from error
    pilot_seeds = tuple(int(value) for value in contract["pilot_seeds"])
    if len(pair_scores) != len(pilot_seeds):
        raise HeadToHeadError("backup-repair pilot summary does not cover every registered pair.")
    interval = bootstrap_mean(
        pair_scores,
        bootstrap_indices(
            sample_size=len(pair_scores),
            resamples=int(contract["bootstrap_resamples"]),
            seed=int(contract["bootstrap_seed"]),
        ),
        confidence_level=float(contract["confidence_level"]),
    )
    delta = {
        "point": interval.point - 0.5,
        "low": interval.low - 0.5,
        "high": interval.high - 0.5,
    }
    candidate_fallbacks = sum(game.candidate_telemetry.fallback_decisions for game in games)
    incumbent_fallbacks = sum(game.incumbent_telemetry.fallback_decisions for game in games)
    candidate_prior_fallbacks = sum(game.candidate_telemetry.prior_fallbacks for game in games)
    incumbent_prior_fallbacks = sum(game.incumbent_telemetry.prior_fallbacks for game in games)
    candidate_root_prior_fallbacks = sum(
        game.candidate_telemetry.root_prior_fallbacks for game in games
    )
    incumbent_root_prior_fallbacks = sum(
        game.incumbent_telemetry.root_prior_fallbacks for game in games
    )
    candidate_branch_prior_fallbacks = sum(
        game.candidate_telemetry.branch_prior_fallbacks for game in games
    )
    incumbent_branch_prior_fallbacks = sum(
        game.incumbent_telemetry.branch_prior_fallbacks for game in games
    )
    no_decision_or_root_fallbacks = (
        candidate_fallbacks == 0
        and incumbent_fallbacks == 0
        and candidate_root_prior_fallbacks == 0
        and incumbent_root_prior_fallbacks == 0
    )
    clears_effect = delta["point"] >= float(contract["minimum_effect_delta"])
    interval_above_neutral = delta["low"] > 0.0
    eligible = no_decision_or_root_fallbacks and clears_effect and interval_above_neutral
    return {
        "schema_version": BACKUP_REPAIR_PILOT_READOUT_SCHEMA_VERSION,
        "contract": dict(contract),
        "complete_pairs": len(pair_scores),
        "candidate_score": interval.to_payload(),
        "candidate_score_delta_from_neutral": delta,
        "fallback_counts": {
            "candidate_decision_fallbacks": candidate_fallbacks,
            "incumbent_decision_fallbacks": incumbent_fallbacks,
            "candidate_prior_fallbacks": candidate_prior_fallbacks,
            "incumbent_prior_fallbacks": incumbent_prior_fallbacks,
            "candidate_root_prior_fallbacks": candidate_root_prior_fallbacks,
            "incumbent_root_prior_fallbacks": incumbent_root_prior_fallbacks,
            # Interior simulated nodes lack an authoritative Showdown request
            # after a simulated replacement. Their fail-closed uniform-prior
            # fallbacks remain visible here but do not falsely invalidate the
            # live root action selected for this game.
            "candidate_branch_prior_fallbacks": candidate_branch_prior_fallbacks,
            "incumbent_branch_prior_fallbacks": incumbent_branch_prior_fallbacks,
        },
        "promotion_checks": {
            "all_registered_pairs_complete": len(pair_scores) == len(pilot_seeds),
            "no_decision_or_root_prior_fallbacks": no_decision_or_root_fallbacks,
            "point_estimate_at_least_minimum_effect": clears_effect,
            "interval_wholly_above_neutral": interval_above_neutral,
        },
        "decision": "ELIGIBLE_FOR_RESERVED_CONFIRMATION" if eligible else "INCONCLUSIVE_OR_NOT_PROMOTED",
    }


def _validated_deadline_qualification(
    raw: object,
    *,
    role: str,
    expected_workers: int,
    expected_provenance: Mapping[str, Any],
    evidence_parent: Path = Path("/shared/scott-experiment"),
) -> dict[str, Any]:
    """Revalidate the exact soft-clock qualification before a scored pilot.

    The two-worker candidate is useful only because R16 and R17 demonstrated
    more completed, source-valid model work under the *same* soft clock.  Do
    not let a later game manifest merely cite those roots by name: re-read their
    immutable hashes, requirements, terminal marker, and every durable decision
    unit before any pilot game is admitted.
    """

    payload = _mapping(raw, label=f"world_parallelism_pilot.deadline_qualifications.{role}")
    if set(payload) != {"root", "manifest_sha256", "pass_sha256"}:
        raise HeadToHeadError(
            f"{role} deadline qualification must contain exactly root, manifest_sha256, and pass_sha256."
        )
    root_raw = payload.get("root")
    if not isinstance(root_raw, str):
        raise HeadToHeadError(f"{role} deadline qualification root must be a string.")
    root = Path(root_raw).resolve()
    try:
        root.relative_to(evidence_parent.resolve())
    except ValueError as error:
        raise HeadToHeadError(
            f"{role} deadline qualification root must be below {evidence_parent}."
        ) from error
    for field in ("manifest_sha256", "pass_sha256"):
        value = payload[field]
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise HeadToHeadError(f"{role} deadline qualification {field} must be a SHA-256 hex digest.")
    manifest_path = root / "MANIFEST.json"
    pass_path = root / "PASS.json"
    try:
        if _sha256_file(manifest_path) != payload["manifest_sha256"]:
            raise HeadToHeadError(f"{role} deadline qualification manifest digest differs.")
        if _sha256_file(pass_path) != payload["pass_sha256"]:
            raise HeadToHeadError(f"{role} deadline qualification PASS digest differs.")
        manifest = _mapping(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            label=f"{role} deadline qualification manifest",
        )
        passed = _mapping(
            json.loads(pass_path.read_text(encoding="utf-8")),
            label=f"{role} deadline qualification PASS",
        )
    except (OSError, json.JSONDecodeError) as error:
        raise HeadToHeadError(
            f"cannot read {role} deadline qualification evidence: {error}"
        ) from error
    expected_requirements = {
        **MODEL_WORLD_PARALLELISM_PILOT_DEADLINE_REQUIREMENTS,
        "model_world_workers": expected_workers,
    }
    if manifest.get("requirements") != expected_requirements:
        raise HeadToHeadError(f"{role} deadline qualification requirements differ.")
    if passed.get("state") != "PASS" or passed.get("marker") != "DEADLINE_QUALIFICATION_PASS":
        raise HeadToHeadError(f"{role} deadline qualification is not a terminal PASS.")
    if passed.get("manifest") != manifest:
        raise HeadToHeadError(f"{role} deadline qualification PASS does not bind its manifest.")

    # R16/R17 established a *specific* soft-clock result.  A different
    # checkpoint, observation vocabulary, Showdown runtime, native engine, or
    # engine-search implementation may have the same row counts while pricing
    # a different game.  Prove the qualification's frozen mechanics match the
    # live scored policy before any games are admitted.  The scorer itself is
    # intentionally allowed to be newer: its source tree changes to add this
    # guard, while the engine-search and native fingerprints must not.
    qualification_checkpoint = _mapping(
        manifest.get("checkpoint"), label=f"{role} deadline qualification checkpoint"
    )
    for field in (
        "checkpoint_sha256",
        "observation_contract_sha256",
        "observation_contract",
        "model_device",
        "showdown_source_sha256",
        "exporter_revision",
    ):
        if qualification_checkpoint.get(field) != expected_provenance[field]:
            raise HeadToHeadError(
                f"{role} deadline qualification checkpoint {field} differs from the live pilot."
            )
    qualification_showdown = _mapping(
        manifest.get("showdown_source"), label=f"{role} deadline qualification Showdown source"
    )
    if (
        qualification_showdown.get("content_sha256")
        != expected_provenance["showdown_content_sha256"]
        or qualification_showdown.get("git_clean") is not True
    ):
        raise HeadToHeadError(
            f"{role} deadline qualification Showdown source differs from the live pilot."
        )
    active_source = _mapping(
        manifest.get("active_source"), label=f"{role} deadline qualification active source"
    )
    source_commit = active_source.get("commit")
    source_tree = active_source.get("execution_tree_sha256")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or not isinstance(source_tree, str)
        or len(source_tree) != 64
        or any(character not in "0123456789abcdef" for character in source_tree)
    ):
        raise HeadToHeadError(f"{role} deadline qualification active source is malformed.")
    qualification_receipt = _mapping(
        manifest.get("source_receipt"), label=f"{role} deadline qualification source receipt"
    )
    if (
        qualification_receipt.get("schema_version")
        != "pokezero.mcts-deadline-source-receipt.v1"
        or qualification_receipt.get("complete") is not True
        or qualification_receipt.get("source_commit") != source_commit
        or qualification_receipt.get("execution_tree_sha256") != source_tree
        or qualification_receipt.get("engine_fingerprint")
        != expected_provenance["engine_fingerprint"]
    ):
        raise HeadToHeadError(
            f"{role} deadline qualification source receipt is not self-consistent with live engine mechanics."
        )
    mechanics = _mapping(
        manifest.get("deadline_mechanics"), label=f"{role} deadline qualification mechanics"
    )
    engine_build = _mapping(
        mechanics.get("engine_build_fingerprint"),
        label=f"{role} deadline qualification native engine fingerprint",
    )
    if (
        engine_build.get("fingerprint") != expected_provenance["engine_fingerprint"]
        or mechanics.get("engine_search_sha256")
        != expected_provenance["engine_search_sha256"]
    ):
        raise HeadToHeadError(
            f"{role} deadline qualification engine mechanics differ from the live pilot."
        )
    if manifest.get("search_config") != expected_provenance["search_config"]:
        raise HeadToHeadError(
            f"{role} deadline qualification search configuration differs from the live pilot."
        )
    decisions_dir = root / "decisions"
    try:
        decision_paths = sorted(decisions_dir.glob("*.json"))
        if len(decision_paths) != expected_requirements["expected_decisions"]:
            raise HeadToHeadError(
                f"{role} deadline qualification has {len(decision_paths)} durable decisions, "
                f"expected {expected_requirements['expected_decisions']}."
            )
        rows = [
            _mapping(json.loads(path.read_text(encoding="utf-8")), label=str(path))
            for path in decision_paths
        ]
        from pokezero.mcts_eval.deadline_qualification import (  # noqa: PLC0415
            DeadlineQualificationRequirements,
            validate_deadline_qualification,
        )

        recomputed = validate_deadline_qualification(
            rows,
            requirements=DeadlineQualificationRequirements(**expected_requirements),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise HeadToHeadError(
            f"cannot revalidate {role} deadline qualification decisions: {error}"
        ) from error
    if passed.get("summary") != recomputed:
        raise HeadToHeadError(f"{role} deadline qualification summary differs from durable decisions.")
    completed_iterations_total, worlds_searched_total = _validated_deadline_totals(
        recomputed, role=role
    )
    return {
        "root": str(root),
        "manifest_sha256": payload["manifest_sha256"],
        "pass_sha256": payload["pass_sha256"],
        "requirements": expected_requirements,
        "completed_iterations_total": completed_iterations_total,
        "worlds_searched_total": worlds_searched_total,
        "qualification_provenance": {
            "source_commit": source_commit,
            "source_tree_sha256": source_tree,
            "checkpoint_sha256": qualification_checkpoint["checkpoint_sha256"],
            "observation_contract_sha256": qualification_checkpoint[
                "observation_contract_sha256"
            ],
            "showdown_content_sha256": qualification_showdown["content_sha256"],
            "engine_fingerprint": engine_build["fingerprint"],
            "engine_search_sha256": mechanics["engine_search_sha256"],
        },
    }


def _world_parallelism_expected_config(
    values: Mapping[str, Any], *, workers: int
) -> dict[str, Any]:
    """Freeze every behavior-bearing engine knob for the first pilot.

    Qualification manifests predate several configuration fields, so comparing
    a short hand-maintained subset would let a shared PUCT or approximation
    change sneak in under the same-clock claim.  Start from the current
    dataclass defaults, set every intentional non-default, then retain only the
    source-bound artifact paths resolved by the runner.  The only allowed arm
    difference is handled by the caller.
    """

    from pokezero.engine_search import EngineMctsConfig  # noqa: PLC0415

    expected = asdict(
        EngineMctsConfig(
            worlds=4,
            search_time_ms=100,
            threads=1,
            strict_fallbacks=True,
            leaf_eval="model",
            model_device="cpu",
            checkpoint_path=values.get("checkpoint_path"),
            model_path=values.get("model_path"),
            tables_path=values.get("tables_path"),
            search_sims=256,
            search_batch=16,
            search_depth=2,
            model_priors=False,
            use_opponent_priors=False,
            early_stop=False,
            model_decision_time_ms=1_000,
            model_native_batch_guard_ms=64,
            model_world_workers=workers,
            depth_min=None,
            worlds_min=None,
        )
    )
    if set(values) != set(expected):
        raise HeadToHeadError(
            "model-world parallelism pilot engine configuration field set differs from the active source."
        )
    return expected


def _world_parallelism_qualification_provenance(
    *,
    checkpoint_contract: Mapping[str, Any],
    showdown_source: Mapping[str, Any],
    engine_fingerprint: str,
) -> dict[str, Any]:
    """Derive the non-negotiable R16/R17-to-live-pilot compatibility keys."""

    required_checkpoint_fields = (
        "checkpoint_sha256",
        "observation_contract_sha256",
        "observation_contract",
        "model_device",
        "showdown_source_sha256",
        "exporter_revision",
    )
    missing = [field for field in required_checkpoint_fields if field not in checkpoint_contract]
    if missing:
        raise HeadToHeadError(
            "live checkpoint contract lacks required world-parallelism qualification field "
            f"{missing[0]!r}."
        )
    showdown_content = showdown_source.get("content_sha256")
    if not isinstance(showdown_content, str) or len(showdown_content) != 64:
        raise HeadToHeadError("live Showdown source lacks a content SHA-256.")
    if not isinstance(engine_fingerprint, str) or not engine_fingerprint:
        raise HeadToHeadError("live native engine lacks a fingerprint.")
    return {
        **{field: checkpoint_contract[field] for field in required_checkpoint_fields},
        "showdown_content_sha256": showdown_content,
        "engine_fingerprint": engine_fingerprint,
        "engine_search_sha256": _sha256_file(REPO_ROOT / "src" / "pokezero" / "engine_search.py"),
        "search_config": {
            "worlds": 4,
            "sims": 256,
            "batch": 16,
            "depth": 2,
            "early_stop": False,
            "model_decision_time_ms": 1_000,
            "model_native_batch_guard_ms": 64,
            "model_priors": False,
            "use_opponent_priors": False,
            # Filled per role before a qualification is revalidated.
            "model_world_workers": None,
        },
    }


def _world_parallelism_pilot_contract(
    manifest: Mapping[str, Any],
    *,
    seeds: tuple[int, ...],
    bootstrap: Mapping[str, Any],
    candidate_raw: Mapping[str, Any],
    incumbent_raw: Mapping[str, Any],
    candidate_config: Any,
    incumbent_config: Any,
    execution_mode: str,
    checkpoint_contract: Mapping[str, Any],
    showdown_source: Mapping[str, Any],
    engine_fingerprint: str | None,
) -> dict[str, Any] | None:
    """Freeze the first same-soft-clock model-world strength pilot.

    This is deliberately a single two-worker-versus-one-worker contrast.  It
    inherits its timing eligibility from two independently complete deadline
    qualifications and cannot be relabelled as a hard-latency study or used to
    tune any other MCTS setting.
    """

    raw = manifest.get("world_parallelism_pilot")
    if raw is None:
        return None
    pilot = _mapping(raw, label="manifest.world_parallelism_pilot")
    expected_fields = {
        "schema_version",
        "stage",
        "failure_retry_policy",
        "minimum_effect_delta",
        "terminal_decisions",
        "deadline_qualifications",
    }
    if set(pilot) != expected_fields:
        raise HeadToHeadError(
            "model-world parallelism pilot has unexpected or missing contract fields."
        )
    if pilot["schema_version"] != MODEL_WORLD_PARALLELISM_PILOT_SCHEMA_VERSION:
        raise HeadToHeadError("model-world parallelism pilot schema is unrecognized.")
    if pilot["stage"] != "pilot":
        raise HeadToHeadError("model-world parallelism pilot must declare stage='pilot'.")
    if pilot["failure_retry_policy"] != MODEL_WORLD_PARALLELISM_PILOT_FAILURE_RETRY_POLICY:
        raise HeadToHeadError("model-world parallelism pilot failure/retry policy differs.")
    if tuple(pilot["terminal_decisions"]) != MODEL_WORLD_PARALLELISM_PILOT_DECISIONS:
        raise HeadToHeadError("model-world parallelism pilot terminal decisions differ.")
    minimum_effect = pilot["minimum_effect_delta"]
    if (
        isinstance(minimum_effect, bool)
        or not isinstance(minimum_effect, (int, float))
        or float(minimum_effect) != MODEL_WORLD_PARALLELISM_PILOT_MINIMUM_EFFECT_DELTA
    ):
        raise HeadToHeadError(
            "model-world parallelism pilot must retain its +0.05 minimum effect."
        )
    if execution_mode != "in_process":
        raise HeadToHeadError("model-world parallelism pilot requires one current in-process source.")
    if engine_fingerprint is None:
        raise HeadToHeadError("model-world parallelism pilot requires an active native engine fingerprint.")
    identity_fields = ("source_commit", "source_tree_sha256", "engine_fingerprint")
    if any(candidate_raw.get(field) != incumbent_raw.get(field) for field in identity_fields):
        raise HeadToHeadError(
            "model-world parallelism pilot requires the same verified source identity for both arms."
        )
    if candidate_raw.get("config_id") == incumbent_raw.get("config_id"):
        raise HeadToHeadError(
            "model-world parallelism pilot requires distinct candidate and incumbent config_id values."
        )
    if len(seeds) != MODEL_WORLD_PARALLELISM_PILOT_PAIRS:
        raise HeadToHeadError(
            f"model-world parallelism pilot requires exactly {MODEL_WORLD_PARALLELISM_PILOT_PAIRS} mirrored pairs."
        )
    if (
        int(bootstrap.get("resamples", 0)) != MODEL_WORLD_PARALLELISM_PILOT_BOOTSTRAP_RESAMPLES
        or float(bootstrap.get("confidence_level", -1.0))
        != MODEL_WORLD_PARALLELISM_PILOT_CONFIDENCE
    ):
        raise HeadToHeadError(
            "model-world parallelism pilot requires 10,000 bootstrap resamples at 80% confidence."
        )
    candidate_values = asdict(candidate_config)
    incumbent_values = asdict(incumbent_config)
    changed_fields = {
        field
        for field in candidate_values
        if candidate_values.get(field) != incumbent_values.get(field)
    }
    if changed_fields != {"model_world_workers"}:
        raise HeadToHeadError(
            "model-world parallelism pilot permits only model_world_workers to differ."
        )
    qualification_provenance = _world_parallelism_qualification_provenance(
        checkpoint_contract=checkpoint_contract,
        showdown_source=showdown_source,
        engine_fingerprint=engine_fingerprint,
    )
    for role, values, workers in (
        ("candidate", candidate_values, 2),
        ("incumbent", incumbent_values, 1),
    ):
        expected = _world_parallelism_expected_config(values, workers=workers)
        if values != expected:
            raise HeadToHeadError(
                f"model-world parallelism pilot {role} does not match the exact qualified soft-clock configuration."
            )
    qualifications = _mapping(
        pilot["deadline_qualifications"], label="model-world parallelism pilot deadline qualifications"
    )
    if set(qualifications) != {"candidate", "incumbent"}:
        raise HeadToHeadError(
            "model-world parallelism pilot requires candidate and incumbent deadline qualifications."
        )
    return {
        "schema_version": MODEL_WORLD_PARALLELISM_PILOT_SCHEMA_VERSION,
        "stage": "pilot",
        "pilot_seeds": list(seeds),
        "bootstrap_resamples": MODEL_WORLD_PARALLELISM_PILOT_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": int(bootstrap["seed"]),
        "confidence_level": MODEL_WORLD_PARALLELISM_PILOT_CONFIDENCE,
        "minimum_effect_delta": MODEL_WORLD_PARALLELISM_PILOT_MINIMUM_EFFECT_DELTA,
        "failure_retry_policy": dict(MODEL_WORLD_PARALLELISM_PILOT_FAILURE_RETRY_POLICY),
        "deadline_qualifications": {
            "candidate": _validated_deadline_qualification(
                qualifications["candidate"],
                role="candidate",
                expected_workers=2,
                expected_provenance={
                    **qualification_provenance,
                    "search_config": {
                        **qualification_provenance["search_config"], "model_world_workers": 2
                    },
                },
            ),
            "incumbent": _validated_deadline_qualification(
                qualifications["incumbent"],
                role="incumbent",
                expected_workers=1,
                expected_provenance={
                    **qualification_provenance,
                    "search_config": {
                        **qualification_provenance["search_config"], "model_world_workers": 1
                    },
                },
            ),
        },
    }


def _world_parallelism_pilot_readout(
    *,
    contract: Mapping[str, Any],
    summary: Mapping[str, Any],
    games: list[Any],
) -> dict[str, Any]:
    """Make a clean positive, null, or failure-contaminated pilot terminal."""

    pair_scores_raw = summary.get("pair_scores")
    if not isinstance(pair_scores_raw, list):
        raise HeadToHeadError("model-world parallelism pilot summary has no pair_scores list.")
    try:
        pair_scores = tuple(float(value) for value in pair_scores_raw)
    except (TypeError, ValueError) as error:
        raise HeadToHeadError("model-world parallelism pilot pair scores are malformed.") from error
    pilot_seeds = tuple(int(value) for value in contract["pilot_seeds"])
    if len(pair_scores) != len(pilot_seeds):
        raise HeadToHeadError("model-world parallelism pilot summary does not cover every registered pair.")
    interval = bootstrap_mean(
        pair_scores,
        bootstrap_indices(
            sample_size=len(pair_scores),
            resamples=int(contract["bootstrap_resamples"]),
            seed=int(contract["bootstrap_seed"]),
        ),
        confidence_level=float(contract["confidence_level"]),
    )
    delta = {
        "point": interval.point - 0.5,
        "low": interval.low - 0.5,
        "high": interval.high - 0.5,
    }
    def total(role: str, field: str) -> int:
        return sum(int(getattr(getattr(game, f"{role}_telemetry"), field)) for game in games)

    candidate_iterations = total("candidate", "total_iterations")
    incumbent_iterations = total("incumbent", "total_iterations")
    candidate_worlds = total("candidate", "worlds_searched")
    incumbent_worlds = total("incumbent", "worlds_searched")
    candidate_root_fallbacks = total("candidate", "root_prior_fallbacks")
    incumbent_root_fallbacks = total("incumbent", "root_prior_fallbacks")
    all_pairs_complete = len(pair_scores) == len(pilot_seeds)
    clean_live_roots = candidate_root_fallbacks == 0 and incumbent_root_fallbacks == 0
    both_arms_searched = (
        candidate_iterations > 0
        and incumbent_iterations > 0
        and candidate_worlds > 0
        and incumbent_worlds > 0
    )
    clears_effect = delta["point"] >= float(contract["minimum_effect_delta"])
    interval_above_neutral = delta["low"] > 0.0
    eligible = (
        all_pairs_complete
        and clean_live_roots
        and both_arms_searched
        and clears_effect
        and interval_above_neutral
    )
    return {
        "schema_version": MODEL_WORLD_PARALLELISM_PILOT_READOUT_SCHEMA_VERSION,
        "complete": True,
        "contract": dict(contract),
        "candidate_score": interval.to_payload(),
        "candidate_score_delta_from_neutral": delta,
        "work": {
            "candidate_iterations": candidate_iterations,
            "incumbent_iterations": incumbent_iterations,
            "candidate_worlds_searched": candidate_worlds,
            "incumbent_worlds_searched": incumbent_worlds,
            "candidate_decision_wall_summary": summary.get("candidate_decision_wall_summary"),
            "incumbent_decision_wall_summary": summary.get("incumbent_decision_wall_summary"),
        },
        "promotion_checks": {
            "all_registered_pairs_complete": all_pairs_complete,
            "zero_root_prior_fallbacks": clean_live_roots,
            "both_arms_completed_model_search": both_arms_searched,
            "point_estimate_at_least_minimum_effect": clears_effect,
            "interval_wholly_above_neutral": interval_above_neutral,
        },
        "decision": (
            "ELIGIBLE_FOR_SEPARATE_CONFIRMATION_REGISTRATION" if eligible else "NO_EXTENSION"
        ),
    }


def _own_policy_prior_study_contract(
    manifest: Mapping[str, Any],
    *,
    seeds: tuple[int, ...],
    bootstrap: Mapping[str, Any],
    candidate_raw: Mapping[str, Any],
    incumbent_raw: Mapping[str, Any],
    candidate_config: Any,
    incumbent_config: Any,
    execution_mode: str,
) -> dict[str, Any] | None:
    """Bind the own-policy-prior ablation to one complete, comparable run.

    This study deliberately does not reuse the model-world contract: that
    contract froze both self-prior switches off to isolate worker count.  Here
    the two policies share a source-isolated build and every configuration
    field except ``model_priors`` must be byte-for-byte equivalent.
    """

    raw = manifest.get("own_policy_prior_study")
    if raw is None:
        return None
    study = dict(_mapping(raw, label="manifest.own_policy_prior_study"))
    required_fields = {
        "schema_version",
        "stage",
        "failure_retry_policy",
        "minimum_effect_delta",
        "max_p95_decision_wall_seconds",
        "max_guided_to_uniform_mean_wall_ratio",
        "terminal_decisions",
    }
    shard_fields = {"shard_index", "full_roster_sha256"}
    if set(study) not in (required_fields, required_fields | shard_fields):
        raise HeadToHeadError("own-policy-prior study has unexpected or missing contract fields.")
    if study["schema_version"] != OWN_POLICY_PRIOR_STUDY_SCHEMA_VERSION:
        raise HeadToHeadError("own-policy-prior study has an unrecognized schema.")
    stage = study["stage"]
    expected_pairs = {"preflight": 4, "strength": 400, "strength_shard": 200}
    if stage not in expected_pairs:
        raise HeadToHeadError("own-policy-prior study stage must be 'preflight' or 'strength'.")
    if len(seeds) != expected_pairs[stage] or len(set(seeds)) != len(seeds):
        raise HeadToHeadError(
            f"own-policy-prior {stage} study requires exactly {expected_pairs[stage]} unique mirrored pairs."
        )
    if stage == "strength_shard":
        if set(study) != required_fields | shard_fields:
            raise HeadToHeadError("own-policy-prior strength shard requires sealed shard identity fields.")
        if study["shard_index"] not in (0, 1) or not isinstance(study["full_roster_sha256"], str) or len(study["full_roster_sha256"]) != 64:
            raise HeadToHeadError("own-policy-prior strength shard identity is malformed.")
    elif set(study) != required_fields:
        raise HeadToHeadError("own-policy-prior non-shard study has unexpected shard identity fields.")
    if study["failure_retry_policy"] != OWN_POLICY_PRIOR_STUDY_FAILURE_RETRY_POLICY:
        raise HeadToHeadError("own-policy-prior study failure/retry policy differs.")
    if tuple(study["terminal_decisions"]) != OWN_POLICY_PRIOR_STUDY_DECISIONS:
        raise HeadToHeadError("own-policy-prior study terminal decisions differ.")
    if (
        int(bootstrap.get("resamples", 0)) != 10_000
        or float(bootstrap.get("confidence_level", -1.0)) != 0.95
    ):
        raise HeadToHeadError(
            "own-policy-prior study requires 10,000 bootstrap resamples at 95% confidence."
        )
    if int(manifest.get("max_decision_rounds", 0)) != 250:
        raise HeadToHeadError("own-policy-prior study requires max_decision_rounds=250.")
    minimum_effect = study["minimum_effect_delta"]
    if (
        isinstance(minimum_effect, bool)
        or not isinstance(minimum_effect, (int, float))
        or float(minimum_effect) != 0.05
    ):
        raise HeadToHeadError("own-policy-prior study requires its registered +0.05 effect.")
    p95_limit = study["max_p95_decision_wall_seconds"]
    if (
        isinstance(p95_limit, bool)
        or not isinstance(p95_limit, (int, float))
        or float(p95_limit) != 1.20
    ):
        raise HeadToHeadError("own-policy-prior study requires the registered 1.20-second p95 limit.")
    ratio_limit = study["max_guided_to_uniform_mean_wall_ratio"]
    if (
        isinstance(ratio_limit, bool)
        or not isinstance(ratio_limit, (int, float))
        or float(ratio_limit) != 1.05
    ):
        raise HeadToHeadError("own-policy-prior study requires the registered 1.05 mean-wall ratio limit.")
    if execution_mode != "isolated_build":
        raise HeadToHeadError("own-policy-prior study requires source-isolated policies for both arms.")
    identity_fields = ("source_commit", "source_tree_sha256", "engine_fingerprint")
    if any(candidate_raw.get(field) != incumbent_raw.get(field) for field in identity_fields):
        raise HeadToHeadError(
            "own-policy-prior study requires the same verified source identity for both arms."
        )
    if candidate_raw.get("config_id") == incumbent_raw.get("config_id"):
        raise HeadToHeadError("own-policy-prior study requires distinct candidate and incumbent config_id values.")
    candidate_values = asdict(candidate_config)
    incumbent_values = asdict(incumbent_config)
    changed_fields = {
        field for field in candidate_values if candidate_values.get(field) != incumbent_values.get(field)
    }
    if changed_fields != {"model_priors"}:
        raise HeadToHeadError(
            "own-policy-prior study permits only model_priors to differ between arms."
        )
    if (
        candidate_values.get("model_priors") is not True
        or incumbent_values.get("model_priors") is not False
        or candidate_values.get("use_opponent_priors") is not False
        or incumbent_values.get("use_opponent_priors") is not False
        or candidate_values.get("model_world_workers") != 1
        or incumbent_values.get("model_world_workers") != 1
        or candidate_values.get("model_decision_time_ms") != 1_000
        or incumbent_values.get("model_decision_time_ms") != 1_000
        or candidate_values.get("model_native_batch_guard_ms") != 64
        or incumbent_values.get("model_native_batch_guard_ms") != 64
        or candidate_values.get("worlds") != 4
        or incumbent_values.get("worlds") != 4
        or candidate_values.get("search_sims") != 256
        or incumbent_values.get("search_sims") != 256
        or candidate_values.get("search_batch") != 16
        or incumbent_values.get("search_batch") != 16
        or candidate_values.get("search_depth") != 2
        or incumbent_values.get("search_depth") != 2
        or candidate_values.get("c_puct") != 1.4
        or incumbent_values.get("c_puct") != 1.4
        or candidate_values.get("early_stop") is not False
        or incumbent_values.get("early_stop") is not False
        or candidate_values.get("leaf_eval") != "model"
        or incumbent_values.get("leaf_eval") != "model"
        or candidate_values.get("model_device") != "cpu"
        or incumbent_values.get("model_device") != "cpu"
        or candidate_values.get("override_telemetry") is not True
        or incumbent_values.get("override_telemetry") is not True
    ):
        raise HeadToHeadError(
            "own-policy-prior study configuration differs from its exact one-worker soft-clock contrast."
        )
    return {
        "schema_version": OWN_POLICY_PRIOR_STUDY_SCHEMA_VERSION,
        "stage": stage,
        "registered_seeds": list(seeds),
        "bootstrap": dict(bootstrap),
        "minimum_effect_delta": 0.05,
        "max_p95_decision_wall_seconds": 1.20,
        "max_guided_to_uniform_mean_wall_ratio": 1.05,
        "failure_retry_policy": dict(OWN_POLICY_PRIOR_STUDY_FAILURE_RETRY_POLICY),
        "terminal_decisions": list(OWN_POLICY_PRIOR_STUDY_DECISIONS),
        "shard_index": study.get("shard_index"),
        "full_roster_sha256": study.get("full_roster_sha256"),
    }


def _own_policy_prior_cap_interval(
    *, games: list[Any], seeds: tuple[int, ...], bootstrap: Mapping[str, Any], capped_score: float
) -> dict[str, float]:
    """Re-score only capped games at a stated guided-arm value, per pair."""

    outcome_score = {"win": 1.0, "tie": 0.5, "loss": 0.0, "cap": capped_score}
    scores: list[float] = []
    for seed in seeds:
        pair = [game for game in games if game.seed == seed]
        if len(pair) != 2:
            raise HeadToHeadError("own-policy-prior cap sensitivity has an incomplete mirrored pair.")
        try:
            scores.append(sum(outcome_score[game.result.outcome] for game in pair) / 2.0)
        except KeyError as error:
            raise HeadToHeadError("own-policy-prior cap sensitivity saw an unknown game outcome.") from error
    interval = bootstrap_mean(
        scores,
        bootstrap_indices(
            sample_size=len(scores), resamples=int(bootstrap["resamples"]), seed=int(bootstrap["seed"])
        ),
        confidence_level=float(bootstrap["confidence_level"]),
    )
    return {"point": interval.point - 0.5, "low": interval.low - 0.5, "high": interval.high - 0.5}


def _own_policy_prior_study_readout(
    *, contract: Mapping[str, Any], summary: Mapping[str, Any], games: list[Any]
) -> dict[str, Any]:
    """Write the frozen validity and outcome decision for own policy guidance."""

    seeds = tuple(int(value) for value in contract["registered_seeds"])
    scores = summary.get("pair_scores")
    if not isinstance(scores, list) or len(scores) != len(seeds):
        raise HeadToHeadError("own-policy-prior study summary does not cover every registered pair.")
    candidate_score = _mapping(summary.get("candidate_score"), label="own-policy-prior candidate_score")
    try:
        delta = {name: float(candidate_score[name]) - 0.5 for name in ("point", "low", "high")}
    except (KeyError, TypeError, ValueError) as error:
        raise HeadToHeadError("own-policy-prior study has malformed paired score interval.") from error
    if any(not math.isfinite(value) for value in delta.values()):
        raise HeadToHeadError("own-policy-prior study paired score interval is non-finite.")

    def nonnegative_summary_int(name: str) -> int:
        value = summary.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise HeadToHeadError(f"own-policy-prior study summary has invalid {name}={value!r}.")
        return value

    candidate_root_fallbacks = nonnegative_summary_int("candidate_root_prior_fallbacks")
    incumbent_root_fallbacks = nonnegative_summary_int("incumbent_root_prior_fallbacks")
    candidate_iterations = nonnegative_summary_int("candidate_iterations")
    incumbent_iterations = nonnegative_summary_int("incumbent_iterations")
    candidate_worlds = nonnegative_summary_int("candidate_worlds_searched")
    incumbent_worlds = nonnegative_summary_int("incumbent_worlds_searched")
    candidate_override_measured = nonnegative_summary_int(
        "candidate_override_measured_decisions"
    )
    incumbent_override_measured = nonnegative_summary_int(
        "incumbent_override_measured_decisions"
    )
    candidate_model_overrides = nonnegative_summary_int("candidate_model_override_decisions")
    incumbent_model_overrides = nonnegative_summary_int("incumbent_model_override_decisions")
    candidate_walls = _mapping(
        summary.get("candidate_decision_wall_summary"), label="own-policy-prior candidate wall summary"
    )
    incumbent_walls = _mapping(
        summary.get("incumbent_decision_wall_summary"), label="own-policy-prior incumbent wall summary"
    )
    try:
        candidate_count = int(candidate_walls["count"])
        incumbent_count = int(incumbent_walls["count"])
        candidate_p95 = float(candidate_walls["p95_s"])
        incumbent_p95 = float(incumbent_walls["p95_s"])
        if candidate_count <= 0 or incumbent_count <= 0:
            raise ValueError("empty searched-decision wall series")
        candidate_mean = float(candidate_walls["total_s"]) / candidate_count
        incumbent_mean = float(incumbent_walls["total_s"]) / incumbent_count
    except (KeyError, TypeError, ValueError) as error:
        raise HeadToHeadError("own-policy-prior study has malformed decision wall summary.") from error
    if (
        not all(math.isfinite(value) and value >= 0.0 for value in (candidate_p95, incumbent_p95, candidate_mean, incumbent_mean))
        or incumbent_mean <= 0.0
    ):
        raise HeadToHeadError("own-policy-prior study has unusable decision wall measurements.")
    mean_ratio = candidate_mean / incumbent_mean
    cap_games = sum(1 for game in games if bool(game.terminal_capped))
    all_pairs_complete = len(games) == 2 * len(seeds)
    clean_roots = candidate_root_fallbacks == 0 and incumbent_root_fallbacks == 0
    both_arms_searched = all(value > 0 for value in (
        candidate_iterations, incumbent_iterations, candidate_worlds, incumbent_worlds
    ))
    guided_arm_has_model_action_witness = candidate_override_measured > 0
    timing_eligible = (
        candidate_p95 <= float(contract["max_p95_decision_wall_seconds"])
        and incumbent_p95 <= float(contract["max_p95_decision_wall_seconds"])
        and mean_ratio <= float(contract["max_guided_to_uniform_mean_wall_ratio"])
    )
    cap_sensitivity = (
        {
            "guided_loss": _own_policy_prior_cap_interval(
                games=games, seeds=seeds, bootstrap=contract["bootstrap"], capped_score=0.0
            ),
            "guided_win": _own_policy_prior_cap_interval(
                games=games, seeds=seeds, bootstrap=contract["bootstrap"], capped_score=1.0
            ),
        }
        if cap_games
        else None
    )
    validity = (
        all_pairs_complete
        and clean_roots
        and both_arms_searched
        and guided_arm_has_model_action_witness
        and timing_eligible
    )
    stage = str(contract["stage"])
    if stage == "preflight":
        decision = "PREFLIGHT_PASS" if validity else "PREFLIGHT_NONPASS"
    elif stage == "strength_shard":
        decision = "SHARD_COMPLETE" if validity else "INCONCLUSIVE"
    elif not validity:
        decision = "INCONCLUSIVE"
    else:
        cap_direction_stable = cap_sensitivity is None or (
            cap_sensitivity["guided_loss"]["low"] > 0.0
            and cap_sensitivity["guided_win"]["low"] > 0.0
        )
        cap_harm_stable = cap_sensitivity is None or (
            cap_sensitivity["guided_loss"]["high"] < 0.0
            and cap_sensitivity["guided_win"]["high"] < 0.0
        )
        cap_target_stable = cap_sensitivity is None or (
            cap_sensitivity["guided_loss"]["high"] < float(contract["minimum_effect_delta"])
            and cap_sensitivity["guided_win"]["high"] < float(contract["minimum_effect_delta"])
        )
        if (
            delta["point"] >= float(contract["minimum_effect_delta"])
            and delta["low"] > 0.0
            and cap_direction_stable
        ):
            decision = "USE_GUIDED_MCTS_AS_RESEARCH_BASELINE"
        elif delta["high"] < 0.0 and cap_harm_stable:
            decision = "GUIDANCE_HARMED_THIS_CONFIGURATION"
        elif delta["high"] < float(contract["minimum_effect_delta"]) and cap_target_stable:
            decision = "TARGET_SIZED_BENEFIT_RULED_OUT"
        else:
            decision = "INCONCLUSIVE"
    return {
        "schema_version": OWN_POLICY_PRIOR_STUDY_READOUT_SCHEMA_VERSION,
        "complete": True,
        "contract": dict(contract),
        "candidate_score_delta_from_neutral": delta,
        "work": {
            "candidate_iterations": candidate_iterations,
            "incumbent_iterations": incumbent_iterations,
            "candidate_worlds_searched": candidate_worlds,
            "incumbent_worlds_searched": incumbent_worlds,
            "candidate_override_measured_decisions": candidate_override_measured,
            "incumbent_override_measured_decisions": incumbent_override_measured,
            "candidate_model_override_decisions": candidate_model_overrides,
            "incumbent_model_override_decisions": incumbent_model_overrides,
            "candidate_decision_wall_summary": dict(candidate_walls),
            "incumbent_decision_wall_summary": dict(incumbent_walls),
            "guided_to_uniform_mean_wall_ratio": mean_ratio,
            "capped_games": cap_games,
        },
        "validity_checks": {
            "all_registered_pairs_complete": all_pairs_complete,
            "zero_root_prior_fallbacks": clean_roots,
            "both_arms_completed_model_search": both_arms_searched,
            "guided_arm_has_model_action_witness": guided_arm_has_model_action_witness,
            "timing_eligible": timing_eligible,
        },
        "cap_sensitivity_delta_from_neutral": cap_sensitivity,
        "decision": decision,
        "marker": f"OWN_POLICY_PRIOR_STUDY_{decision}",
    }


def _opponent_prior_applicability_contract(
    manifest: Mapping[str, Any],
    *,
    candidate_raw: Mapping[str, Any],
    incumbent_raw: Mapping[str, Any],
    candidate_config: Any,
    incumbent_config: Any,
    execution_mode: str,
) -> dict[str, Any] | None:
    """Freeze the one-setting current-source opponent-prior applicability read.

    This is deliberately not a strength-study schema.  Its only conclusion is
    whether the flag-on candidate actually supplied a non-uniform model-priced
    opponent arm to the tree, while the otherwise-identical flag-off twin did
    not.  Scoring still remains a later, separately registered decision.
    """

    raw = manifest.get("opponent_prior_applicability")
    if raw is None:
        return None
    contract = _mapping(raw, label="manifest.opponent_prior_applicability")
    if dict(contract) != OPPONENT_PRIOR_APPLICABILITY_CONTRACT:
        raise HeadToHeadError(
            "opponent-prior applicability must use the exact declared development contract."
        )
    if execution_mode != "isolated_build":
        raise HeadToHeadError(
            "opponent-prior applicability requires source-isolated policies for both arms."
        )
    identity_fields = ("source_commit", "source_tree_sha256", "engine_fingerprint")
    if any(candidate_raw.get(field) != incumbent_raw.get(field) for field in identity_fields):
        raise HeadToHeadError(
            "opponent-prior applicability requires the same verified source identity for both arms."
        )
    if candidate_raw.get("config_id") == incumbent_raw.get("config_id"):
        raise HeadToHeadError(
            "opponent-prior applicability requires distinct candidate and incumbent config_id values."
        )
    candidate_values = asdict(candidate_config)
    incumbent_values = asdict(incumbent_config)
    changed_fields = {
        field
        for field in candidate_values
        if candidate_values.get(field) != incumbent_values.get(field)
    }
    if changed_fields != {"use_opponent_priors"}:
        raise HeadToHeadError(
            "opponent-prior applicability permits only use_opponent_priors to differ between arms."
        )
    if (
        candidate_values.get("model_priors") is not True
        or incumbent_values.get("model_priors") is not True
        or candidate_values.get("use_opponent_priors") is not True
        or incumbent_values.get("use_opponent_priors") is not False
    ):
        raise HeadToHeadError(
            "opponent-prior applicability requires model priors on and a true-versus-false opponent-prior contrast."
        )
    # The native tree has always applied these priors independently of whether
    # the report exposes its arms.  The application witness, however, is
    # produced only by the pure `override_telemetry` report path.  Require it
    # for both arms here rather than spend a development roster on a counter
    # that would necessarily remain zero.  It is unchanged between arms by
    # the one-setting check above and does not alter search selection.
    if (
        candidate_values.get("override_telemetry") is not True
        or incumbent_values.get("override_telemetry") is not True
    ):
        raise HeadToHeadError(
            "opponent-prior applicability requires override_telemetry=true on both arms "
            "to retain the native applied-prior witness."
        )
    return dict(contract)


def _opponent_prior_applicability_readout(
    *, contract: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Produce a terminal applied/not-applied result before any score is read."""

    def counter(name: str) -> int:
        value = summary.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise HeadToHeadError(
                f"opponent-prior applicability summary has invalid {name}={value!r}."
            )
        return value

    def status_counter(name: str) -> dict[str, int]:
        value = summary.get(name)
        if not isinstance(value, Mapping):
            raise HeadToHeadError(
                f"opponent-prior applicability summary has no {name} mapping."
            )
        validated: dict[str, int] = {}
        for status, count in value.items():
            if status not in OPPONENT_REQUEST_ORDER_STATUS_VALUES:
                raise HeadToHeadError(
                    f"opponent-prior applicability summary has unknown {name} status "
                    f"{status!r}."
                )
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise HeadToHeadError(
                    f"opponent-prior applicability summary has invalid {name} "
                    f"count {status!r}={count!r}."
                )
            validated[str(status)] = count
        return dict(sorted(validated.items()))

    candidate_count = counter("candidate_opponent_prior_arm_decisions")
    incumbent_count = counter("incumbent_opponent_prior_arm_decisions")
    candidate_root_prior_fallbacks = counter("candidate_root_prior_fallbacks")
    incumbent_root_prior_fallbacks = counter("incumbent_root_prior_fallbacks")
    candidate_order_statuses = status_counter("candidate_opponent_request_order_statuses")
    incumbent_order_statuses = status_counter("incumbent_opponent_request_order_statuses")
    candidate_root_fallback_statuses = status_counter(
        "candidate_opponent_request_order_root_fallback_statuses"
    )
    incumbent_root_fallback_statuses = status_counter(
        "incumbent_opponent_request_order_root_fallback_statuses"
    )
    for role, root_fallbacks, statuses, root_statuses in (
        (
            "candidate",
            candidate_root_prior_fallbacks,
            candidate_order_statuses,
            candidate_root_fallback_statuses,
        ),
        (
            "incumbent",
            incumbent_root_prior_fallbacks,
            incumbent_order_statuses,
            incumbent_root_fallback_statuses,
        ),
    ):
        if sum(root_statuses.values()) != root_fallbacks:
            raise HeadToHeadError(
                "opponent-prior applicability summary has unclassified "
                f"{role} root fallbacks."
            )
        if any(count > statuses.get(status, 0) for status, count in root_statuses.items()):
            raise HeadToHeadError(
                "opponent-prior applicability summary has "
                f"{role} root-fallback statuses beyond their status denominator."
            )
    candidate_applied = (
        candidate_count
        >= int(contract["minimum_candidate_opponent_prior_arm_decisions"])
    )
    incumbent_remained_off = (
        incumbent_count
        <= int(contract["maximum_incumbent_opponent_prior_arm_decisions"])
    )
    live_roots_clean = (
        candidate_root_prior_fallbacks
        <= int(contract["maximum_candidate_root_prior_fallbacks"])
        and incumbent_root_prior_fallbacks
        <= int(contract["maximum_incumbent_root_prior_fallbacks"])
    )
    status = (
        "PASS"
        if candidate_applied and incumbent_remained_off and live_roots_clean
        else "NONPASS"
    )
    return {
        "schema_version": OPPONENT_PRIOR_APPLICABILITY_READOUT_SCHEMA_VERSION,
        "complete": True,
        "contract": dict(contract),
        "candidate_opponent_prior_arm_decisions": candidate_count,
        "incumbent_opponent_prior_arm_decisions": incumbent_count,
        "candidate_root_prior_fallbacks": candidate_root_prior_fallbacks,
        "incumbent_root_prior_fallbacks": incumbent_root_prior_fallbacks,
        "candidate_opponent_request_order_statuses": candidate_order_statuses,
        "incumbent_opponent_request_order_statuses": incumbent_order_statuses,
        "candidate_opponent_request_order_root_fallback_statuses": (
            candidate_root_fallback_statuses
        ),
        "incumbent_opponent_request_order_root_fallback_statuses": (
            incumbent_root_fallback_statuses
        ),
        "checks": {
            "candidate_applied_model_priced_opponent_arm": candidate_applied,
            "incumbent_remained_flag_off": incumbent_remained_off,
            "live_root_priors_remained_clean": live_roots_clean,
            "root_fallbacks_have_source_order_statuses": True,
        },
        "status": status,
        "marker": f"OPPONENT_PRIOR_APPLICABILITY_{status}",
    }


def _opponent_prior_strength_pilot_contract(
    manifest: Mapping[str, Any],
    *,
    seeds: tuple[int, ...],
    bootstrap: Mapping[str, Any],
    applicability_contract: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the declared terminal decision for an opponent-prior pilot.

    Applicability and strength answer different questions.  A flag-on policy can
    demonstrably price opponent arms without having enough paired strength
    evidence to justify a confirmation roster.  Keep the latter's exact,
    predeclared threshold in the runner so a generic summary cannot be mistaken
    for an approval.
    """

    trial_raw = manifest.get("trial_protocol")
    if trial_raw is None:
        return None
    trial = _mapping(trial_raw, label="manifest.trial_protocol")
    raw = trial.get("readout_contract")
    if raw is None:
        return None
    if applicability_contract is None:
        raise HeadToHeadError(
            "opponent-prior strength pilot requires the matching applicability contract."
        )
    contract = dict(_mapping(raw, label="manifest.trial_protocol.readout_contract"))
    required_fields = {
        "schema_version",
        "bootstrap",
        "candidate_allowed_opponent_request_order_statuses",
        "candidate_observed_source_order_statuses_at_least",
        "candidate_opponent_prior_arm_decisions_at_least",
        "candidate_score_reference",
        "delta_definition",
        "incumbent_allowed_opponent_request_order_statuses",
        "incumbent_opponent_prior_arm_decisions_at_most",
        "interval_lower_strictly_above",
        "minimum_effect_delta",
        "terminal_decisions",
    }
    if set(contract) != required_fields:
        raise HeadToHeadError(
            "opponent-prior strength pilot readout contract has unexpected or missing fields."
        )
    if contract["schema_version"] != OPPONENT_PRIOR_STRENGTH_PILOT_READOUT_SCHEMA_VERSION:
        raise HeadToHeadError("opponent-prior strength pilot has an unrecognized readout schema.")
    declared_bootstrap = dict(_mapping(contract["bootstrap"], label="readout_contract.bootstrap"))
    if declared_bootstrap != dict(bootstrap):
        raise HeadToHeadError(
            "opponent-prior strength pilot readout bootstrap must exactly match manifest.bootstrap."
        )

    def nonnegative_int(name: str, *, minimum: int = 0) -> int:
        value = contract[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise HeadToHeadError(
                f"opponent-prior strength pilot {name} must be an integer at least {minimum}."
            )
        return value

    def allowed_statuses(name: str) -> tuple[str, ...]:
        values = contract[name]
        if not isinstance(values, list) or tuple(values) != ("resolved",):
            raise HeadToHeadError(
                f"opponent-prior strength pilot {name} must be exactly ['resolved']."
            )
        return ("resolved",)

    candidate_allowed_statuses = allowed_statuses(
        "candidate_allowed_opponent_request_order_statuses"
    )
    incumbent_allowed_statuses = allowed_statuses(
        "incumbent_allowed_opponent_request_order_statuses"
    )
    candidate_observed_minimum = nonnegative_int(
        "candidate_observed_source_order_statuses_at_least", minimum=1
    )
    candidate_arm_minimum = nonnegative_int(
        "candidate_opponent_prior_arm_decisions_at_least", minimum=1
    )
    incumbent_arm_maximum = nonnegative_int(
        "incumbent_opponent_prior_arm_decisions_at_most"
    )
    if incumbent_arm_maximum != 0:
        raise HeadToHeadError(
            "opponent-prior strength pilot incumbent arm maximum must remain zero."
        )
    for name, expected in (
        ("candidate_score_reference", 0.5),
        ("interval_lower_strictly_above", 0.0),
    ):
        value = contract[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != expected:
            raise HeadToHeadError(
                f"opponent-prior strength pilot {name} must remain {expected:g}."
            )
    if contract["delta_definition"] != "mean_paired_candidate_score_minus_0.5":
        raise HeadToHeadError("opponent-prior strength pilot has an unrecognized delta definition.")
    minimum_effect = contract["minimum_effect_delta"]
    if (
        isinstance(minimum_effect, bool)
        or not isinstance(minimum_effect, (int, float))
        or not 0.0 < float(minimum_effect) <= 0.5
    ):
        raise HeadToHeadError(
            "opponent-prior strength pilot minimum effect must be in (0, 0.5]."
        )
    if tuple(contract["terminal_decisions"]) != OPPONENT_PRIOR_STRENGTH_PILOT_DECISIONS:
        raise HeadToHeadError(
            "opponent-prior strength pilot must declare the exact eligible/no-extension decisions."
        )
    return {
        "schema_version": OPPONENT_PRIOR_STRENGTH_PILOT_READOUT_SCHEMA_VERSION,
        "registered_seeds": list(seeds),
        "bootstrap": declared_bootstrap,
        "candidate_allowed_opponent_request_order_statuses": list(candidate_allowed_statuses),
        "candidate_observed_source_order_statuses_at_least": candidate_observed_minimum,
        "candidate_opponent_prior_arm_decisions_at_least": candidate_arm_minimum,
        "candidate_score_reference": 0.5,
        "delta_definition": "mean_paired_candidate_score_minus_0.5",
        "incumbent_allowed_opponent_request_order_statuses": list(incumbent_allowed_statuses),
        "incumbent_opponent_prior_arm_decisions_at_most": incumbent_arm_maximum,
        "interval_lower_strictly_above": 0.0,
        "minimum_effect_delta": float(minimum_effect),
        "terminal_decisions": list(OPPONENT_PRIOR_STRENGTH_PILOT_DECISIONS),
    }


def _opponent_prior_strength_pilot_readout(
    *,
    contract: Mapping[str, Any],
    summary: Mapping[str, Any],
    applicability_readout: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the terminal strength decision without upgrading a thin signal."""

    raw_scores = summary.get("pair_scores")
    if not isinstance(raw_scores, list):
        raise HeadToHeadError("opponent-prior strength pilot summary has no pair_scores list.")
    try:
        scores = tuple(float(value) for value in raw_scores)
    except (TypeError, ValueError) as error:
        raise HeadToHeadError("opponent-prior strength pilot pair scores are malformed.") from error
    if not scores or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores):
        raise HeadToHeadError("opponent-prior strength pilot pair scores must be finite values in [0, 1].")
    registered_seeds = tuple(int(value) for value in contract["registered_seeds"])
    if len(scores) != len(registered_seeds):
        raise HeadToHeadError(
            "opponent-prior strength pilot summary does not cover every registered pair."
        )
    if applicability_readout.get("status") not in {"PASS", "NONPASS"}:
        raise HeadToHeadError("opponent-prior strength pilot has no terminal applicability status.")

    def count(name: str) -> int:
        value = applicability_readout.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise HeadToHeadError(
                f"opponent-prior strength pilot applicability readout has invalid {name}={value!r}."
            )
        return value

    def statuses(name: str) -> dict[str, int]:
        raw = applicability_readout.get(name)
        if not isinstance(raw, Mapping):
            raise HeadToHeadError(
                f"opponent-prior strength pilot applicability readout has no {name} mapping."
            )
        result: dict[str, int] = {}
        for status, value in raw.items():
            if status not in OPPONENT_REQUEST_ORDER_STATUS_VALUES:
                raise HeadToHeadError(
                    f"opponent-prior strength pilot has unknown request-order status {status!r}."
                )
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise HeadToHeadError(
                    f"opponent-prior strength pilot has invalid {name} count {status!r}={value!r}."
                )
            result[str(status)] = value
        return dict(sorted(result.items()))

    candidate_statuses = statuses("candidate_opponent_request_order_statuses")
    incumbent_statuses = statuses("incumbent_opponent_request_order_statuses")
    candidate_arms = count("candidate_opponent_prior_arm_decisions")
    incumbent_arms = count("incumbent_opponent_prior_arm_decisions")
    interval = bootstrap_mean(
        scores,
        bootstrap_indices(
            sample_size=len(scores),
            resamples=int(contract["bootstrap"]["resamples"]),
            seed=int(contract["bootstrap"]["seed"]),
        ),
        confidence_level=float(contract["bootstrap"]["confidence_level"]),
    )
    reference = float(contract["candidate_score_reference"])
    delta = {
        "point": interval.point - reference,
        "low": interval.low - reference,
        "high": interval.high - reference,
    }
    candidate_orders_allowed = set(candidate_statuses).issubset(
        set(contract["candidate_allowed_opponent_request_order_statuses"])
    )
    incumbent_orders_allowed = set(incumbent_statuses).issubset(
        set(contract["incumbent_allowed_opponent_request_order_statuses"])
    )
    checks = {
        "all_registered_pairs_complete": len(scores) == len(registered_seeds),
        "applicability_passed": applicability_readout.get("status") == "PASS",
        "candidate_source_order_statuses_allowed": candidate_orders_allowed,
        "candidate_observed_source_order_statuses_sufficient": (
            sum(candidate_statuses.values())
            >= int(contract["candidate_observed_source_order_statuses_at_least"])
        ),
        "candidate_opponent_prior_arm_decisions_sufficient": (
            candidate_arms >= int(contract["candidate_opponent_prior_arm_decisions_at_least"])
        ),
        "incumbent_source_order_statuses_allowed": incumbent_orders_allowed,
        "incumbent_opponent_prior_arm_decisions_bounded": (
            incumbent_arms <= int(contract["incumbent_opponent_prior_arm_decisions_at_most"])
        ),
        "point_estimate_at_least_minimum_effect": (
            delta["point"] >= float(contract["minimum_effect_delta"])
        ),
        "interval_lower_strictly_above_reference": (
            delta["low"] > float(contract["interval_lower_strictly_above"])
        ),
    }
    eligible = all(checks.values())
    decision = (
        "ELIGIBLE_FOR_SEPARATE_CONFIRMATION_REGISTRATION" if eligible else "NO_EXTENSION"
    )
    return {
        "schema_version": OPPONENT_PRIOR_STRENGTH_PILOT_READOUT_SCHEMA_VERSION,
        "complete": True,
        "contract": dict(contract),
        "complete_pairs": len(scores),
        "candidate_score": interval.to_payload(),
        "candidate_score_delta_from_neutral": delta,
        "candidate_opponent_prior_arm_decisions": candidate_arms,
        "incumbent_opponent_prior_arm_decisions": incumbent_arms,
        "candidate_opponent_request_order_statuses": candidate_statuses,
        "incumbent_opponent_request_order_statuses": incumbent_statuses,
        "promotion_checks": checks,
        "status": "PASS" if eligible else "NONPASS",
        "decision": decision,
        "marker": f"OPPONENT_PRIOR_STRENGTH_PILOT_{decision}",
    }


def _required_sha256(value: object, *, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise HeadToHeadError(f"{label} must be a 64-character lowercase SHA-256 hash.")
    return digest


def _isolated_policy_identity(raw: Mapping[str, Any], *, role: str) -> tuple[str, str, str]:
    """Read the remote source receipt declared by one isolated policy.

    The worker independently recomputes these values before it constructs a
    policy.  This helper only freezes the expected receipt in the host's
    immutable experiment contract.
    """

    commit = str(raw.get("source_commit", ""))
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise HeadToHeadError(
            f"isolated {role}.source_commit must be a full lowercase Git commit id."
        )
    tree_sha256 = _required_sha256(
        raw.get("source_tree_sha256"), label=f"isolated {role}.source_tree_sha256"
    )
    fingerprint = str(raw.get("engine_fingerprint", ""))
    if not fingerprint:
        raise HeadToHeadError(f"isolated {role}.engine_fingerprint must be non-empty.")
    return commit, tree_sha256, fingerprint


def _validate_isolated_receipt(
    receipt: object,
    *,
    policy: MctsPolicySpec,
    role: str,
    bootstrap_sha256: str,
) -> Mapping[str, Any]:
    """Check the durable child receipt before a game is accepted or resumed."""

    payload = _mapping(receipt, label="isolated worker receipt")
    if dict(_mapping(payload.get("policy"), label="isolated worker receipt.policy")) != (
        policy.to_payload()
    ):
        raise HeadToHeadError(
            f"isolated worker receipt policy does not exactly match the declared {role}."
        )
    if payload.get("commit") != policy.source_commit:
        raise HeadToHeadError(f"isolated worker receipt source commit differs from {role}.")
    if payload.get("tree_sha256") != policy.source_tree_sha256:
        raise HeadToHeadError(f"isolated worker receipt source tree differs from {role}.")
    if payload.get("tree_status") != "clean_tracked_checkout":
        raise HeadToHeadError("isolated worker receipt does not attest a clean source checkout.")
    if payload.get("engine_fingerprint") != policy.engine_fingerprint:
        raise HeadToHeadError(f"isolated worker receipt engine differs from {role}.")
    if payload.get("worker_bootstrap_sha256") != bootstrap_sha256:
        raise HeadToHeadError(
            "isolated worker receipt bootstrap differs from the host's declared adapter."
        )
    if payload.get("reset_protocol") != "policy_method_or_fresh_source_policy.v1":
        raise HeadToHeadError(
            "isolated worker receipt does not attest the required source-safe reset protocol."
        )
    compatibility = _mapping(
        payload.get("config_compatibility"), label="isolated worker receipt.config_compatibility"
    )
    if compatibility.get("protocol") != ISOLATED_CONFIG_COMPATIBILITY_PROTOCOL:
        raise HeadToHeadError(
            "isolated worker receipt does not attest the required config compatibility protocol."
        )
    omitted = compatibility.get("omitted_disabled_fields")
    if (
        not isinstance(omitted, list)
        or any(not isinstance(field_name, str) for field_name in omitted)
        or omitted != sorted(set(omitted))
    ):
        raise HeadToHeadError(
            "isolated worker receipt has an invalid omitted diagnostic-field record."
        )
    for field_name in omitted:
        expected = ISOLATED_DISABLED_DIAGNOSTIC_COMPATIBILITY_DEFAULTS.get(field_name)
        if field_name not in ISOLATED_DISABLED_DIAGNOSTIC_COMPATIBILITY_DEFAULTS:
            raise HeadToHeadError(
                "isolated worker receipt omitted an unsupported config field "
                f"{field_name!r}."
            )
        if field_name not in policy.config or policy.config[field_name] != expected:
            raise HeadToHeadError(
                "isolated worker receipt omitted a diagnostic field that is not explicitly "
                f"disabled in the declared {role}."
            )
    return dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--manifest", required=True, help="frozen MCTS-vs-MCTS experiment JSON")
    parser.add_argument("--out-dir", required=True, help="new or exact-resume durable result directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--isolated-candidate-source-root",
        help=(
            "clean fixed PokeZero checkout for the candidate; required together with the "
            "isolated incumbent so neither source is represented by the host engine"
        ),
    )
    parser.add_argument(
        "--isolated-candidate-python",
        help="Python executable from the isolated candidate build",
    )
    parser.add_argument(
        "--isolated-incumbent-source-root",
        help=(
            "clean historical PokeZero checkout for the incumbent; enables the only "
            "source-different execution mode"
        ),
    )
    parser.add_argument(
        "--isolated-incumbent-python",
        help="Python executable from the isolated incumbent build (required with its source root)",
    )
    parser.add_argument(
        "--isolated-worker-timeout-seconds",
        type=float,
        default=ISOLATED_WORKER_RESPONSE_TIMEOUT_SECONDS,
        help=(
            "per-decision request/response deadline for each isolated worker "
            f"(default: {ISOLATED_WORKER_RESPONSE_TIMEOUT_SECONDS:g} seconds)"
        ),
    )
    parser.add_argument("--skip-build-check", action="store_true", help="dry inspection only; never scored")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.skip_build_check:
        raise HeadToHeadError("--skip-build-check is not permitted for an MCTS-vs-MCTS result.")
    isolated_values = (
        args.isolated_candidate_source_root,
        args.isolated_candidate_python,
        args.isolated_incumbent_source_root,
        args.isolated_incumbent_python,
    )
    if any(isolated_values) and not all(isolated_values):
        raise HeadToHeadError(
            "source-different mode requires isolated candidate and incumbent source roots plus "
            "their Python executables; do not let the host engine stand in for either build."
        )
    if args.isolated_worker_timeout_seconds <= 0:
        raise HeadToHeadError("--isolated-worker-timeout-seconds must be positive.")
    execution_mode = "isolated_build" if all(isolated_values) else "in_process"
    out_root = _durable_output_root(args.out_dir)
    manifest = _load_manifest(args.manifest)
    if _replacement_study_requires_durable_launcher(manifest):
        _require_durable_launcher_handoff(out_root)
    declared_showdown_source_sha256 = _declared_showdown_source_sha256(manifest)
    declared_source_tree_sha256 = _declared_source_tree_sha256(manifest)
    seeds = _seeds(manifest)
    bootstrap, resamples, bootstrap_seed, bootstrap_confidence_level = _bootstrap_contract(manifest)
    max_decision_rounds = int(manifest.get("max_decision_rounds", 0))
    if max_decision_rounds <= 0:
        raise HeadToHeadError("manifest.max_decision_rounds must be positive.")

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from engine_build_fingerprint import assert_fresh, compute_fingerprint  # noqa: PLC0415
    from pokezero.dex import load_showdown_dex_cached  # noqa: PLC0415
    from pokezero.engine_search import (  # noqa: PLC0415
        EngineMctsPolicy,
        EnvTier2AnnotationSource,
    )
    from pokezero.local_showdown import (  # noqa: PLC0415
        LocalShowdownConfig,
        LocalShowdownEnv,
        env_config_from_checkpoint_provenance,
    )
    from pokezero.mcts_eval.lattice import materialize_search_artifacts  # noqa: PLC0415
    from pokezero.mcts_eval.isolated_policy import (  # noqa: PLC0415
        IsolatedMctsPolicy,
        IsolatedPolicyLaunch,
    )
    from pokezero.mcts_eval.resolver import resolve_checkpoint_contract  # noqa: PLC0415
    from pokezero.neural_policy import (  # noqa: PLC0415
        category_vocab_from_model_config,
        feature_masks_from_model_config,
        load_transformer_model_config,
        observation_spec_from_model_config,
    )
    from pokezero.randbat import load_gen3_randbat_source_cached  # noqa: PLC0415
    from pokezero.rollout import (  # noqa: PLC0415
        RolloutConfig,
        RolloutDecisionProgress,
        RolloutDriver,
    )

    source = _source_provenance()
    source_commit = source["commit"]
    source_tree_sha256 = source["tree_sha256"]
    if declared_source_tree_sha256 != source_tree_sha256:
        raise HeadToHeadError(
            "manifest source-tree identity does not match the executing source/image receipt: "
            f"declared {declared_source_tree_sha256}, active {source_tree_sha256}."
        )
    if execution_mode == "in_process":
        assert_fresh()
        engine_fingerprint = str(compute_fingerprint()["fingerprint"])
    else:
        # The host only drives the shared Showdown battle in this mode. Search
        # decisions (and their native extensions) are both bound in child
        # receipts, so a host engine fingerprint would be misleading.
        engine_fingerprint = None
    checkpoint_sha256 = _sha256_file(args.checkpoint)
    showdown_source = _showdown_source_provenance(args.showdown_root)
    showdown_source_sha256 = str(showdown_source["content_sha256"])
    if declared_showdown_source_sha256 != showdown_source_sha256:
        raise HeadToHeadError(
            "manifest Showdown source identity does not match --showdown-root: "
            f"declared {declared_showdown_source_sha256}, active {showdown_source_sha256}."
        )
    contract = resolve_checkpoint_contract(
        args.checkpoint,
        model_device=args.device,
        showdown_root=args.showdown_root,
        showdown_source_sha256=showdown_source_sha256,
        expected_showdown_source_sha256=declared_showdown_source_sha256,
    )
    artifacts = materialize_search_artifacts(contract, showdown_root=args.showdown_root)
    candidate_raw = _mapping(manifest.get("candidate"), label="manifest.candidate")
    incumbent_raw = _mapping(manifest.get("incumbent"), label="manifest.incumbent")
    isolated_candidate_source_root: Path | None = None
    isolated_incumbent_source_root: Path | None = None
    isolated_bootstrap_sha256: str | None = None
    if execution_mode == "isolated_build":
        isolated_candidate_source_root = Path(args.isolated_candidate_source_root).expanduser().resolve()
        isolated_incumbent_source_root = Path(args.isolated_incumbent_source_root).expanduser().resolve()
        if not isolated_candidate_source_root.is_dir():
            raise HeadToHeadError(
                "--isolated-candidate-source-root does not name an existing directory."
            )
        if not isolated_incumbent_source_root.is_dir():
            raise HeadToHeadError(
                "--isolated-incumbent-source-root does not name an existing directory."
            )
        candidate_commit, candidate_tree_sha256, candidate_fingerprint = _isolated_policy_identity(
            candidate_raw, role="candidate"
        )
        incumbent_commit, incumbent_tree_sha256, incumbent_fingerprint = _isolated_policy_identity(
            incumbent_raw, role="incumbent"
        )
        candidate, candidate_config = _runtime_spec(
            candidate_raw,
            role="isolated candidate",
            checkpoint=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            source_commit=candidate_commit,
            source_tree_sha256=candidate_tree_sha256,
            engine_fingerprint=candidate_fingerprint,
            showdown_source_sha256=showdown_source_sha256,
            model_path=artifacts["model_path"],
            tables_path=artifacts["tables_path"],
            device=args.device,
        )
        incumbent, incumbent_config = _runtime_spec(
            incumbent_raw,
            role="isolated incumbent",
            checkpoint=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            source_commit=incumbent_commit,
            source_tree_sha256=incumbent_tree_sha256,
            engine_fingerprint=incumbent_fingerprint,
            showdown_source_sha256=showdown_source_sha256,
            model_path=artifacts["model_path"],
            tables_path=artifacts["tables_path"],
            device=args.device,
        )
        isolated_bootstrap_sha256 = _sha256_file(
            REPO_ROOT / "scripts" / "mcts_isolated_policy_worker.py"
        )
    else:
        assert engine_fingerprint is not None
        candidate, candidate_config = _runtime_spec(
            candidate_raw,
            role="candidate",
            checkpoint=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            source_commit=source_commit,
            source_tree_sha256=source_tree_sha256,
            engine_fingerprint=engine_fingerprint,
            showdown_source_sha256=showdown_source_sha256,
            model_path=artifacts["model_path"],
            tables_path=artifacts["tables_path"],
            device=args.device,
        )
        incumbent, incumbent_config = _runtime_spec(
            incumbent_raw,
            role="incumbent",
            checkpoint=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            source_commit=source_commit,
            source_tree_sha256=source_tree_sha256,
            engine_fingerprint=engine_fingerprint,
            showdown_source_sha256=showdown_source_sha256,
            model_path=artifacts["model_path"],
            tables_path=artifacts["tables_path"],
            device=args.device,
        )

    pilot_contract = _backup_repair_pilot_contract(
        manifest,
        seeds=seeds,
        bootstrap=bootstrap,
        candidate_raw=candidate_raw,
        incumbent_raw=incumbent_raw,
        candidate_config=candidate_config,
        incumbent_config=incumbent_config,
    )
    if pilot_contract is not None:
        if execution_mode != "isolated_build" or isolated_candidate_source_root is None or isolated_incumbent_source_root is None:
            raise HeadToHeadError(
                "backup-repair pilot requires isolated source builds for both runtime revisions."
            )
    world_parallelism_pilot_contract = _world_parallelism_pilot_contract(
        manifest,
        seeds=seeds,
        bootstrap=bootstrap,
        candidate_raw=candidate_raw,
        incumbent_raw=incumbent_raw,
        candidate_config=candidate_config,
        incumbent_config=incumbent_config,
        execution_mode=execution_mode,
        checkpoint_contract=contract.to_manifest(),
        showdown_source=showdown_source,
        engine_fingerprint=engine_fingerprint,
    )
    own_policy_prior_study_contract = _own_policy_prior_study_contract(
        manifest,
        seeds=seeds,
        bootstrap=bootstrap,
        candidate_raw=candidate_raw,
        incumbent_raw=incumbent_raw,
        candidate_config=candidate_config,
        incumbent_config=incumbent_config,
        execution_mode=execution_mode,
    )
    opponent_prior_applicability_contract = _opponent_prior_applicability_contract(
        manifest,
        candidate_raw=candidate_raw,
        incumbent_raw=incumbent_raw,
        candidate_config=candidate_config,
        incumbent_config=incumbent_config,
        execution_mode=execution_mode,
    )
    opponent_prior_strength_pilot_contract = _opponent_prior_strength_pilot_contract(
        manifest,
        seeds=seeds,
        bootstrap=bootstrap,
        applicability_contract=opponent_prior_applicability_contract,
    )

    model_config = load_transformer_model_config(args.checkpoint)
    vocabulary = category_vocab_from_model_config(model_config, args.showdown_root)
    env_config = env_config_from_checkpoint_provenance(
        LocalShowdownConfig(
            showdown_root=args.showdown_root,
            set_belief_source=True,
            category_vocab=vocabulary,
        ),
        feature_masks_from_model_config(model_config),
        required_specs=observation_spec_from_model_config(model_config),
        required_vocabs=vocabulary,
        context="MCTS-vs-MCTS paired runner",
    )
    dex = None
    set_source = None
    if execution_mode == "in_process":
        dex = load_showdown_dex_cached(args.showdown_root)
        set_source = load_gen3_randbat_source_cached(args.showdown_root)

    _write_immutable_json(
        out_root / "manifest.json",
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "declared_manifest": manifest,
            "active_source_commit": source_commit,
            "active_source": source,
            "active_engine_fingerprint": engine_fingerprint,
            "active_checkpoint_sha256": checkpoint_sha256,
            "active_showdown_source": showdown_source,
            "candidate": candidate.to_payload(),
            "incumbent": incumbent.to_payload(),
            "seeds": list(seeds),
            "execution_mode": execution_mode,
            "backup_repair_pilot_contract": pilot_contract,
            "world_parallelism_pilot_contract": world_parallelism_pilot_contract,
            "own_policy_prior_study_contract": own_policy_prior_study_contract,
            "opponent_prior_strength_pilot_contract": opponent_prior_strength_pilot_contract,
            "isolated_policies": (
                {
                    "candidate": {
                        "source_root": str(isolated_candidate_source_root),
                        "python": str(args.isolated_candidate_python),
                    },
                    "incumbent": {
                        "source_root": str(isolated_incumbent_source_root),
                        "python": str(args.isolated_incumbent_python),
                    },
                    "worker_bootstrap_sha256": isolated_bootstrap_sha256,
                }
                if execution_mode == "isolated_build"
                else None
            ),
        },
    )

    isolated_worker_attempt_id: str | None = None
    if execution_mode == "isolated_build":
        # This record is intentionally separate from the experiment manifest:
        # it describes one resumable host invocation, whereas the manifest is
        # the stable experiment contract.  Every invocation gets new child-log
        # paths and preserves the earlier attempt's stderr evidence.
        isolated_worker_attempt_id = uuid.uuid4().hex
        _write_immutable_json(
            out_root
            / "worker-stderr"
            / f"attempt-{isolated_worker_attempt_id}"
            / "ATTEMPT.json",
            {
                "schema_version": "pokezero.mcts-h2h-isolated-worker-attempt.v1",
                "attempt_id": isolated_worker_attempt_id,
                "host_pid": os.getpid(),
                "host_source": source,
                "candidate_provenance_sha256": candidate.provenance_sha256,
                "incumbent_provenance_sha256": incumbent.provenance_sha256,
                "request_response_timeout_seconds": args.isolated_worker_timeout_seconds,
            },
        )

    isolated_workers: dict[tuple[int, str], dict[str, IsolatedMctsPolicy]] = {}

    def write_progress(
        event: str,
        *,
        seed: int,
        candidate_seat: str,
        decision: RolloutDecisionProgress | None = None,
        completed_candidate_seats: tuple[str, ...] | None = None,
    ) -> None:
        """Publish a non-score-bearing, atomically replaceable runner heartbeat."""

        payload: dict[str, Any] = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "event": event,
            "seed": seed,
            "candidate_seat": candidate_seat,
            "max_decision_rounds": max_decision_rounds,
            "execution_mode": execution_mode,
            "candidate_provenance_sha256": candidate.provenance_sha256,
            "incumbent_provenance_sha256": incumbent.provenance_sha256,
        }
        if decision is not None:
            payload.update(
                {
                    "battle_id": decision.battle_id,
                    "decision_round_index": decision.decision_round_index,
                    "decision_round_count": decision.decision_round_count,
                    "requested_players": list(decision.requested_players),
                    "terminal": decision.terminal,
                    "terminal_capped": decision.terminal_capped,
                    "terminal_winner": decision.terminal_winner,
                }
            )
        if completed_candidate_seats is not None:
            payload["completed_candidate_seats"] = list(completed_candidate_seats)
        _write_progress_json(out_root / "progress" / "current.json", payload)

    def isolated_policy_for(
        *,
        policy: MctsPolicySpec,
        role: str,
        source_root: Path,
        python: str,
        annotations: Any,
        seed: int,
        candidate_seat: str,
    ) -> IsolatedMctsPolicy:
        assert isolated_bootstrap_sha256 is not None
        assert isolated_worker_attempt_id is not None
        return IsolatedMctsPolicy(
            IsolatedPolicyLaunch(
                policy=policy,
                command=(python, str(REPO_ROOT / "scripts" / "mcts_isolated_policy_worker.py")),
                worker_config={
                    "source_root": str(source_root),
                    "showdown_root": str(Path(args.showdown_root).expanduser().resolve()),
                    "worker_bootstrap_sha256": isolated_bootstrap_sha256,
                },
                response_timeout_seconds=args.isolated_worker_timeout_seconds,
                stderr_path=_isolated_worker_stderr_path(
                    out_root,
                    attempt_id=isolated_worker_attempt_id,
                    seed=seed,
                    candidate_seat=candidate_seat,
                    role=role,
                ),
                working_directory=source_root,
            ),
            annotation_source=annotations,
        )

    def session_factory(seed: int, candidate_seat: str):
        write_progress("game_started", seed=seed, candidate_seat=candidate_seat)
        env = LocalShowdownEnv(env_config)
        annotations = EnvTier2AnnotationSource(env)
        if execution_mode == "isolated_build":
            assert isolated_candidate_source_root is not None
            assert isolated_incumbent_source_root is not None
            assert isolated_bootstrap_sha256 is not None
            isolated_candidate = isolated_policy_for(
                policy=candidate,
                role="candidate",
                source_root=isolated_candidate_source_root,
                python=str(args.isolated_candidate_python),
                annotations=annotations,
                seed=seed,
                candidate_seat=candidate_seat,
            )
            isolated_incumbent = isolated_policy_for(
                policy=incumbent,
                role="incumbent",
                source_root=isolated_incumbent_source_root,
                python=str(args.isolated_incumbent_python),
                annotations=annotations,
                seed=seed,
                candidate_seat=candidate_seat,
            )
            isolated_workers[(seed, candidate_seat)] = {
                "candidate": isolated_candidate,
                "incumbent": isolated_incumbent,
            }
            candidate_policy = PublicOnlyMctsPolicy(isolated_candidate)
            incumbent_policy = PublicOnlyMctsPolicy(isolated_incumbent)
        else:
            assert dex is not None and set_source is not None
            candidate_policy = PublicOnlyMctsPolicy(
                EngineMctsPolicy(
                    dex=dex,
                    set_source=set_source,
                    config=candidate_config,
                    policy_id=candidate.policy_id,
                    annotation_source=annotations,
                )
            )
            incumbent_policy = PublicOnlyMctsPolicy(
                EngineMctsPolicy(
                    dex=dex,
                    set_source=set_source,
                    config=incumbent_config,
                    policy_id=incumbent.policy_id,
                    annotation_source=annotations,
                )
            )
        other_seat = "p2" if candidate_seat == "p1" else "p1"
        driver = RolloutDriver(
            env=env,
            policies={candidate_seat: candidate_policy, other_seat: incumbent_policy},
            config=RolloutConfig(
                max_decision_rounds=max_decision_rounds,
                format_id="gen3randombattle",
                record_policy_timing=True,
                hide_opponent_legal_action_masks=True,
                decision_sink=lambda decision: write_progress(
                    "decision_committed",
                    seed=seed,
                    candidate_seat=candidate_seat,
                    decision=decision,
                ),
            ),
        )
        return driver, candidate_policy, incumbent_policy

    all_games = []
    for seed in seeds:
        completed = load_pair(
            out_root, seed=seed, candidate=candidate, incumbent=incumbent
        )
        if execution_mode == "isolated_build":
            assert isolated_bootstrap_sha256 is not None
            for candidate_seat in completed:
                for role, policy in (("candidate", candidate), ("incumbent", incumbent)):
                    receipt_path = (
                        out_root
                        / "worker-receipts"
                        / f"seed-{seed}-{candidate_seat}-{role}.json"
                    )
                    try:
                        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as error:
                        raise HeadToHeadError(
                            "completed isolated game has no readable matching worker receipt "
                            f"at {receipt_path}: {error}"
                        ) from error
                    _validate_isolated_receipt(
                        receipt,
                        policy=policy,
                        role=role,
                        bootstrap_sha256=isolated_bootstrap_sha256,
                    )

        def on_game(game):
            if execution_mode == "isolated_build":
                assert isolated_bootstrap_sha256 is not None
                workers = isolated_workers.pop((game.seed, game.candidate_seat), None)
                if workers is None:
                    raise HeadToHeadError(
                        "isolated workers were not retained for their completed game receipts."
                    )
                for role, policy in (("candidate", candidate), ("incumbent", incumbent)):
                    worker = workers.get(role)
                    if worker is None:
                        raise HeadToHeadError(
                            f"isolated {role} worker is missing for its completed game receipt."
                        )
                    receipt = _validate_isolated_receipt(
                        worker.worker_receipt,
                        policy=policy,
                        role=role,
                        bootstrap_sha256=isolated_bootstrap_sha256,
                    )
                    _write_immutable_json(
                        out_root
                        / "worker-receipts"
                        / f"seed-{game.seed}-{game.candidate_seat}-{role}.json",
                        receipt,
                    )
            write_game_immutable(out_root, game)
            write_progress(
                "game_persisted",
                seed=game.seed,
                candidate_seat=game.candidate_seat,
            )

        games = play_mirrored_pair(
            seed=seed,
            candidate=candidate,
            incumbent=incumbent,
            candidate_factory=lambda: None,
            incumbent_factory=lambda: None,
            driver_factory=lambda *_: None,
            completed=completed,
            session_factory=session_factory,
            on_game=on_game,
            execution_mode=execution_mode,
        )
        complete_pair(games, seed=seed, candidate=candidate, incumbent=incumbent)
        write_progress(
            "pair_completed",
            seed=seed,
            candidate_seat="both",
            completed_candidate_seats=tuple(sorted(game.candidate_seat for game in games)),
        )
        all_games.extend(games)
        print(f"completed mirrored pair seed={seed}", flush=True)

    summary = summarize_complete_pairs(
        all_games,
        seeds=seeds,
        candidate=candidate,
        incumbent=incumbent,
        bootstrap_resamples=resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence_level=bootstrap_confidence_level,
    )
    _write_immutable_json(out_root / "summary.json", summary)
    pilot_readout_path: Path | None = None
    if pilot_contract is not None:
        pilot_readout_path = out_root / "PILOT_READOUT.json"
        _write_immutable_json(
            pilot_readout_path,
            _backup_repair_pilot_readout(
                contract=pilot_contract,
                summary=summary,
                games=all_games,
            ),
        )
    world_parallelism_pilot_readout_path: Path | None = None
    if world_parallelism_pilot_contract is not None:
        world_parallelism_pilot_readout_path = out_root / "WORLD_PARALLELISM_PILOT_READOUT.json"
        _write_immutable_json(
            world_parallelism_pilot_readout_path,
            _world_parallelism_pilot_readout(
                contract=world_parallelism_pilot_contract,
                summary=summary,
                games=all_games,
            ),
        )
    own_policy_prior_study_readout_path: Path | None = None
    if own_policy_prior_study_contract is not None:
        own_policy_prior_study_readout_path = out_root / "OWN_POLICY_PRIOR_STUDY_READOUT.json"
        _write_immutable_json(
            own_policy_prior_study_readout_path,
            _own_policy_prior_study_readout(
                contract=own_policy_prior_study_contract,
                summary=summary,
                games=all_games,
            ),
        )
    opponent_prior_applicability_readout_path: Path | None = None
    applicability_readout: dict[str, Any] | None = None
    if opponent_prior_applicability_contract is not None:
        opponent_prior_applicability_readout_path = (
            out_root / "OPPONENT_PRIOR_APPLICABILITY.json"
        )
        applicability_readout = _opponent_prior_applicability_readout(
            contract=opponent_prior_applicability_contract,
            summary=summary,
        )
        _write_immutable_json(
            opponent_prior_applicability_readout_path,
            applicability_readout,
        )
        if applicability_readout["status"] != "PASS":
            raise HeadToHeadError(
                "opponent-prior applicability is terminal NONPASS; "
                "the source-isolated contrast did not prove applied, clean opponent priors."
            )
    opponent_prior_strength_pilot_readout_path: Path | None = None
    if opponent_prior_strength_pilot_contract is not None:
        if applicability_readout is None:
            raise HeadToHeadError(
                "opponent-prior strength pilot requires a completed applicability readout."
            )
        opponent_prior_strength_pilot_readout_path = (
            out_root / "OPPONENT_PRIOR_STRENGTH_PILOT_READOUT.json"
        )
        _write_immutable_json(
            opponent_prior_strength_pilot_readout_path,
            _opponent_prior_strength_pilot_readout(
                contract=opponent_prior_strength_pilot_contract,
                summary=summary,
                applicability_readout=applicability_readout,
            ),
        )
    _write_immutable_json(
        out_root / "COMPLETE.json",
        {
            "schema_version": COMPLETE_SCHEMA_VERSION,
            "summary_sha256": _sha256_file(out_root / "summary.json"),
            "games": len(all_games),
            "pairs": len(seeds),
            "candidate_provenance_sha256": candidate.provenance_sha256,
            "incumbent_provenance_sha256": incumbent.provenance_sha256,
            "backup_repair_pilot_readout_sha256": (
                _sha256_file(pilot_readout_path) if pilot_readout_path is not None else None
            ),
            "world_parallelism_pilot_readout_sha256": (
                _sha256_file(world_parallelism_pilot_readout_path)
                if world_parallelism_pilot_readout_path is not None
                else None
            ),
            "own_policy_prior_study_readout_sha256": (
                _sha256_file(own_policy_prior_study_readout_path)
                if own_policy_prior_study_readout_path is not None
                else None
            ),
            "opponent_prior_applicability_readout_sha256": (
                _sha256_file(opponent_prior_applicability_readout_path)
                if opponent_prior_applicability_readout_path is not None
                else None
            ),
            "opponent_prior_strength_pilot_readout_sha256": (
                _sha256_file(opponent_prior_strength_pilot_readout_path)
                if opponent_prior_strength_pilot_readout_path is not None
                else None
            ),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HeadToHeadError as error:
        print(f"MCTS-vs-MCTS REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
