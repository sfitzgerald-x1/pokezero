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


MANIFEST_SCHEMA_VERSION = "pokezero.mcts-h2h-manifest.v1"
COMPLETE_SCHEMA_VERSION = "pokezero.mcts-h2h-complete.v1"
BACKUP_REPAIR_PILOT_SCHEMA_VERSION = "pokezero.mcts-h2h-backup-repair-pilot.v4"
BACKUP_REPAIR_PILOT_READOUT_SCHEMA_VERSION = "pokezero.mcts-h2h-backup-repair-pilot-readout.v2"
OPPONENT_PRIOR_APPLICABILITY_SCHEMA_VERSION = (
    "pokezero.mcts-h2h-opponent-prior-applicability.v1"
)
OPPONENT_PRIOR_APPLICABILITY_READOUT_SCHEMA_VERSION = (
    "pokezero.mcts-h2h-opponent-prior-applicability-readout.v1"
)
OPPONENT_PRIOR_APPLICABILITY_CONTRACT = {
    "schema_version": OPPONENT_PRIOR_APPLICABILITY_SCHEMA_VERSION,
    "stage": "development_applicability",
    "minimum_candidate_opponent_prior_arm_decisions": 1,
    "maximum_incumbent_opponent_prior_arm_decisions": 0,
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


def _replacement_study_requires_durable_launcher(manifest: Mapping[str, Any]) -> bool:
    study = manifest.get("study")
    return isinstance(study, Mapping) and study.get("schema_version") == BACKUP_REPAIR_PILOT_SCHEMA_VERSION


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

    candidate_count = counter("candidate_opponent_prior_arm_decisions")
    incumbent_count = counter("incumbent_opponent_prior_arm_decisions")
    candidate_applied = (
        candidate_count
        >= int(contract["minimum_candidate_opponent_prior_arm_decisions"])
    )
    incumbent_remained_off = (
        incumbent_count
        <= int(contract["maximum_incumbent_opponent_prior_arm_decisions"])
    )
    status = "PASS" if candidate_applied and incumbent_remained_off else "NONPASS"
    return {
        "schema_version": OPPONENT_PRIOR_APPLICABILITY_READOUT_SCHEMA_VERSION,
        "complete": True,
        "contract": dict(contract),
        "candidate_opponent_prior_arm_decisions": candidate_count,
        "incumbent_opponent_prior_arm_decisions": incumbent_count,
        "checks": {
            "candidate_applied_model_priced_opponent_arm": candidate_applied,
            "incumbent_remained_flag_off": incumbent_remained_off,
        },
        "status": status,
        "marker": f"OPPONENT_PRIOR_APPLICABILITY_{status}",
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
    resamples, bootstrap_seed, bootstrap_confidence_level = _bootstrap_settings(manifest)
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
    from pokezero.rollout import RolloutConfig, RolloutDriver  # noqa: PLC0415

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
    opponent_prior_applicability_contract = _opponent_prior_applicability_contract(
        manifest,
        candidate_raw=candidate_raw,
        incumbent_raw=incumbent_raw,
        candidate_config=candidate_config,
        incumbent_config=incumbent_config,
        execution_mode=execution_mode,
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
    opponent_prior_applicability_readout_path: Path | None = None
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
                "the candidate did not prove applied model-priced opponent arms."
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
            "opponent_prior_applicability_readout_sha256": (
                _sha256_file(opponent_prior_applicability_readout_path)
                if opponent_prior_applicability_readout_path is not None
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
