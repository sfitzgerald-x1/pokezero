#!/usr/bin/env python3
"""Run the fixed source-root model-leaf versus rollout-leaf diagnostic.

This is deliberately a *root-selection* diagnostic, not a strength evaluator.
It replays eleven predeclared public decision roots from the completed R4 audit:

* ``model_control_a`` and ``model_control_b`` use the exact same model leaf;
  their selection witnesses must match exactly, proving the replay boundary is
  deterministic before the leaf comparison is interpreted.
* ``rollout_leaf`` changes only the leaf evaluator.  It has a mandatory live
  rollout witness, so a configuration receipt cannot be mistaken for evidence
  that the rollout pricer actually ran.

Every completed root is atomically durable.  An interrupted launcher resumes
only validated root units under the same manifest; it never adopts a partial
or differently configured run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

from pokezero.actions import ACTION_COUNT  # noqa: E402
from pokezero.engine_search import require_rollout_leaf_witness  # noqa: E402
from pokezero.mcts_eval.lattice import _LiveEngineTimingDecider  # noqa: E402
from pokezero.mcts_eval.manifest import SearchConfig  # noqa: E402
from pokezero.mcts_eval.resolver import (  # noqa: E402
    ContractError,
    resolve_checkpoint_contract,
    sha256_file,
)
from pokezero.mcts_eval.source_root_replay import (  # noqa: E402
    SourceRootReplayError,
    source_bound_replay_prefix,
)
from pokezero.public_decision_corpus import PublicDecisionRecord  # noqa: E402


SCHEMA_VERSION = "pokezero.source-root-leaf-ablation.v2"
SOURCE_WRAPPER_SCHEMA = "pokezero.mcts-guided-vs-raw-public-decision.v1"
ARMS = ("model_control_a", "model_control_b", "rollout_leaf")
SEARCH = {"depth": 6, "sims": 4096, "batch": 16, "worlds": 4}
ROLLOUT = {
    "rollout_count": 32,
    "rollout_max_plies": 200,
    "rollout_policy": "uniform",
    "rollout_threads": 12,
    "rollout_threads_cpu_budget_ack": True,
}
HISTORICAL_MODEL_CONFIG = {
    "leaf_eval": "model",
    "model_priors": True,
    "use_opponent_priors": False,
    "override_telemetry": True,
    "early_stop": False,
    "model_decision_time_ms": None,
    "model_world_workers": 1,
    "search_depth": 6,
    "search_sims": 4096,
    "search_batch": 16,
    "worlds": 4,
    "c_puct": 1.4,
    "deep_ko_split": True,
    "leaf_batch": 1,
    "fpu_reduction": None,
}


@dataclass(frozen=True, order=True)
class SourceRoot:
    seed: int
    seat: str
    turn_index: int

    def to_dict(self) -> dict[str, object]:
        return {"seed": self.seed, "seat": self.seat, "turn_index": self.turn_index}


# Fixed before this script is ever run.  The five omitted R4 roots have an
# unresolved opponent event and therefore cannot be repaired without crossing
# the source actor's information boundary.
TARGETS = (
    SourceRoot(2026092004, "p1", 19),
    SourceRoot(2026092004, "p1", 20),
    SourceRoot(2026092005, "p1", 21),
    SourceRoot(2026092005, "p1", 25),
    SourceRoot(2026092005, "p2", 1),
    SourceRoot(2026092005, "p2", 6),
    SourceRoot(2026092006, "p1", 2),
    SourceRoot(2026092006, "p1", 3),
    SourceRoot(2026092007, "p1", 9),
    SourceRoot(2026092007, "p1", 19),
    SourceRoot(2026092007, "p2", 9),
)


class AblationError(RuntimeError):
    """A non-bankable source-root leaf-ablation result."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _create_terminal(path: Path, payload: Mapping[str, Any]) -> None:
    """Create the terminal record once; never replace a completed result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AblationError(f"refusing to replace terminal artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise AblationError(f"refusing to replace terminal artifact: {path}") from error
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AblationError(f"cannot read JSON artifact {path}: {error}") from error


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--source-root", required=True, help="Completed R4 durable root.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--expected-source-commit",
        help=(
            "Optional full commit that the executing image must contain. Required for a "
            "source-repair measurement so a repaired engine cannot be mistaken for a historical replay."
        ),
    )
    parser.add_argument(
        "--expected-engine-fingerprint",
        help=(
            "Optional repaired native-engine fingerprint. Omit only for strict historical-engine replay."
        ),
    )
    parser.add_argument("--model-device", default="cuda", choices=("cuda",))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)
    if len(args.expected_checkpoint_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.expected_checkpoint_sha256
    ):
        parser.error("--expected-checkpoint-sha256 must be lowercase SHA-256")
    for option in ("expected_source_commit", "expected_engine_fingerprint"):
        value = getattr(args, option)
        if value is not None and not _is_lower_hex(value, 40 if option == "expected_source_commit" else 64):
            parser.error(f"--{option.replace('_', '-')} must be lowercase hexadecimal")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if args.finalize_only and args.shard_index != 0:
        parser.error("--finalize-only requires shard index zero")
    return args


def _root_directory(out_root: Path, root: SourceRoot) -> Path:
    return out_root / "roots" / f"seed-{root.seed}-{root.seat}-turn-{root.turn_index:03d}"


def _historical_config_projection(candidate_config: Any) -> dict[str, Any]:
    if not isinstance(candidate_config, Mapping):
        raise AblationError("source candidate has no model-search config")
    projected = {field: candidate_config.get(field) for field in HISTORICAL_MODEL_CONFIG}
    if projected != HISTORICAL_MODEL_CONFIG:
        raise AblationError("source candidate does not use the historical fixed-work model config")
    return projected


def _is_lower_hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _hash_source_files(repo_root: Path, paths: Sequence[Path]) -> str:
    """Hash executable source with names as well as bytes."""

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
    """Bind the executing Python/bridge source to a clean checkout or image receipt."""

    stamped = os.environ.get("POKEZERO_COMMIT", "").strip().lower()
    if (ROOT / ".git").exists():
        try:
            commit = subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip().lower()
            dirty = subprocess.run(
                ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            tracked = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files", "-z"],
                check=True,
                capture_output=True,
            ).stdout.split(b"\0")
        except (OSError, subprocess.CalledProcessError) as error:
            raise AblationError("cannot bind the executing source checkout") from error
        if not _is_lower_hex(commit, 40):
            raise AblationError("executing source HEAD is not a full lowercase Git commit")
        if stamped and stamped != commit:
            raise AblationError("POKEZERO_COMMIT does not match the executing checkout")
        if dirty:
            raise AblationError("refusing to run against a dirty executing source checkout")
        paths = [
            ROOT / value.decode("utf-8")
            for value in tracked
            if value and (ROOT / value.decode("utf-8")).is_file()
        ]
        return {
            "commit": commit,
            "tree_sha256": _hash_source_files(ROOT, paths),
            "tree_status": "clean_tracked_checkout",
        }

    if not _is_lower_hex(stamped, 40):
        raise AblationError("an image without .git must set POKEZERO_COMMIT to its full lowercase commit")
    roots = (ROOT / "src" / "pokezero", ROOT / "scripts")
    paths = [
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    paths.extend(path for path in (ROOT / "scripts").glob("*.mjs") if path.is_file())
    if (ROOT / "pyproject.toml").is_file():
        paths.append(ROOT / "pyproject.toml")
    if not paths:
        raise AblationError("cannot hash executable source in image without .git")
    return {
        "commit": stamped,
        "tree_sha256": _hash_source_files(ROOT, paths),
        "tree_status": "explicit_hash_without_git_python_and_bridge",
    }


def _showdown_dependency_paths(root: Path) -> list[Path]:
    """Return every Showdown byte the Gen 3 battle oracle can load."""

    required = (
        root / "dist" / "sim" / "index.js", root / "dist" / "sim" / "dex.js",
        root / "dist" / "sim" / "dex-data.js", root / "dist" / "data" / "moves.js",
        root / "dist" / "data" / "pokedex.js", root / "dist" / "data" / "typechart.js",
        root / "dist" / "data" / "abilities.js", root / "dist" / "data" / "items.js",
        root / "dist" / "data" / "mods" / "gen3" / "moves.js",
        root / "dist" / "data" / "mods" / "gen3" / "scripts.js",
        root / "dist" / "data" / "mods" / "gen3" / "abilities.js",
        root / "dist" / "data" / "mods" / "gen3" / "items.js",
        root / "data" / "random-battles" / "gen3" / "sets.json",
        root / "dist" / "data" / "random-battles" / "gen3" / "teams.js",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise AblationError(
            f"cannot bind Showdown runtime; required input is missing: {missing[0].relative_to(root)}"
        )
    paths = set(required)
    paths.update((root / "dist").rglob("*.js"))
    paths.update((root / "dist").rglob("*.json"))
    return sorted(path for path in paths if path.is_file())


def _showdown_source_provenance(showdown_root: str | Path) -> dict[str, Any]:
    root = Path(showdown_root).expanduser().resolve()
    try:
        top_level = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"], cwd=root, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        ).resolve()
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip().lower()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise AblationError(f"cannot bind Showdown runtime identity from {root}") from error
    if top_level != root or not _is_lower_hex(commit, 40) or dirty:
        raise AblationError("Showdown must be a clean root Git checkout with a full lowercase commit")
    digest = hashlib.sha256()
    for path in _showdown_dependency_paths(root):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return {"content_sha256": digest.hexdigest(), "git_commit": commit, "git_clean": True}


def _validate_historical_baseline(source_root: Path) -> dict[str, Any]:
    """Bind the new deterministic control to the archived R4 search contract.

    This proves configuration equivalence, not bit-for-bit historical replay:
    public records intentionally do not retain private simulator state or the
    historical decision RNG stream.  The distinction is persisted so identical
    controls cannot be misreported as historical reproduction.
    """

    rows: list[dict[str, Any]] = []
    for path in sorted(source_root.glob("seeds/seed-*/manifest.json")):
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            raise AblationError(f"source manifest is malformed: {path}")
        candidate = payload.get("candidate")
        if not isinstance(candidate, Mapping):
            raise AblationError(f"source manifest has no candidate: {path}")
        config = _historical_config_projection(candidate.get("config"))
        checkpoint_sha256 = candidate.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
            raise AblationError(f"source manifest has no valid candidate checkpoint: {path}")
        seed = payload.get("study", {}).get("seed") if isinstance(payload.get("study"), Mapping) else None
        if seed not in {2026092004, 2026092005, 2026092006, 2026092007}:
            raise AblationError(f"source manifest has an unexpected seed: {path}")
        rows.append({
            "seed": seed,
            "config_sha256": _sha256(config),
            "checkpoint_sha256": checkpoint_sha256,
            "manifest_sha256": sha256_file(path),
            "active_source": payload.get("active_source"),
            "active_engine_fingerprint": payload.get("active_engine_fingerprint"),
            "active_showdown_source": payload.get("active_showdown_source"),
        })
    if len(rows) != 4 or {row["seed"] for row in rows} != {2026092004, 2026092005, 2026092006, 2026092007}:
        raise AblationError("source historical baseline manifests are incomplete")
    if len({row["config_sha256"] for row in rows}) != 1:
        raise AblationError("source historical baseline config differs across seeds")
    if len({row["checkpoint_sha256"] for row in rows}) != 1:
        raise AblationError("source historical baseline checkpoint differs across seeds")
    runtime_rows: list[dict[str, Any]] = []
    for row in rows:
        source = row["active_source"]
        engine = row["active_engine_fingerprint"]
        showdown = row["active_showdown_source"]
        if (
            not isinstance(source, Mapping)
            or not _is_lower_hex(source.get("commit"), 40)
            or not _is_lower_hex(source.get("tree_sha256"), 64)
            or not isinstance(source.get("tree_status"), str)
            or not _is_lower_hex(engine, 64)
            or not isinstance(showdown, Mapping)
            or not _is_lower_hex(showdown.get("content_sha256"), 64)
            or not _is_lower_hex(showdown.get("git_commit"), 40)
            or showdown.get("git_clean") is not True
        ):
            raise AblationError(f"source historical baseline has malformed runtime identity for seed {row['seed']}")
        runtime_rows.append({
            "source": dict(source),
            "engine_fingerprint": engine,
            "showdown_source": dict(showdown),
        })
    if len({_canonical_json(runtime) for runtime in runtime_rows}) != 1:
        raise AblationError("source historical baseline runtime differs across seeds")
    return {
        "historical_config": dict(HISTORICAL_MODEL_CONFIG),
        "historical_config_sha256": rows[0]["config_sha256"],
        "historical_candidate_checkpoint_sha256": rows[0]["checkpoint_sha256"],
        "source_manifest_sha256_by_seed": {
            str(row["seed"]): row["manifest_sha256"] for row in sorted(rows, key=lambda row: row["seed"])
        },
        "historical_runtime": runtime_rows[0],
        "historical_reproduction": False,
        "historical_reproduction_reason": "public-source replay uses a newly pinned decision RNG",
    }


def _execution_runtime(
    args: argparse.Namespace, historical_baseline: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the actual engine stack and reject a different battle oracle.

    The source-replay repair deliberately runs newer, source-bound Python code,
    so it is preserved as a distinct execution identity rather than pretending
    to reproduce the archived R4 source tree.  The battle oracle and native
    engine must nevertheless remain exactly the archived R4 versions.
    """

    historical = historical_baseline.get("historical_runtime")
    if not isinstance(historical, Mapping):
        raise AblationError("historical runtime identity is missing")
    expected_showdown = historical.get("showdown_source")
    historical_engine = historical.get("engine_fingerprint")
    if (
        not isinstance(expected_showdown, Mapping)
        or not _is_lower_hex(expected_showdown.get("content_sha256"), 64)
        or not _is_lower_hex(historical_engine, 64)
    ):
        raise AblationError("historical runtime identity is malformed")
    source = _source_provenance()
    if args.expected_source_commit is not None and source["commit"] != args.expected_source_commit:
        raise AblationError("executing source commit differs from the explicitly bound source-repair commit")
    showdown = _showdown_source_provenance(args.showdown_root)
    if showdown["content_sha256"] != expected_showdown["content_sha256"]:
        raise AblationError("active Showdown runtime differs from the archived R4 battle oracle")
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from engine_build_fingerprint import assert_fresh, compute_fingerprint  # noqa: PLC0415

        assert_fresh()
        engine = compute_fingerprint()
    except BaseException as error:  # assert_fresh may raise SystemExit for stale artifacts.
        raise AblationError("installed native engine failed its freshness check") from error
    fingerprint = engine.get("fingerprint")
    expected_engine = args.expected_engine_fingerprint or historical_engine
    if fingerprint != expected_engine:
        if args.expected_engine_fingerprint is None:
            raise AblationError("active native engine differs from the archived R4 engine fingerprint")
        raise AblationError("active native engine differs from the explicitly bound source-repair fingerprint")
    return {
        "source": source,
        "engine_build": dict(engine),
        "engine_fingerprint": fingerprint,
        "historical_engine_fingerprint": historical_engine,
        "expected_engine_fingerprint": expected_engine,
        "engine_identity_mode": "historical_replay" if args.expected_engine_fingerprint is None else "source_repair",
        "showdown_source": showdown,
    }


def _read_source_wrapper(path: Path) -> tuple[PublicDecisionRecord, Mapping[str, Any]]:
    wrapper = _read_json(path)
    if not isinstance(wrapper, Mapping) or set(wrapper) != {
        "candidate_provenance_sha256", "candidate_seat", "raw_provenance_sha256", "record", "schema_version", "seed"
    }:
        raise AblationError(f"source record wrapper has an unsupported shape: {path}")
    if wrapper.get("schema_version") != SOURCE_WRAPPER_SCHEMA:
        raise AblationError(f"source record wrapper schema differs: {path}")
    payload = wrapper.get("record")
    if not isinstance(payload, Mapping):
        raise AblationError(f"source record wrapper has no record: {path}")
    try:
        record = PublicDecisionRecord.from_dict(payload)
    except (TypeError, ValueError) as error:
        raise AblationError(f"source record is invalid: {path}: {error}") from error
    if wrapper.get("seed") != record.seed or wrapper.get("candidate_seat") != record.acting_player:
        raise AblationError(f"source record wrapper disagrees with its public record: {path}")
    return record, wrapper


def _source_wrapper_path(source_root: Path, root: SourceRoot) -> Path:
    matches = sorted(
        source_root.glob(
            "seeds/seed-"
            f"{root.seed}/public-decision-records/seed-{root.seed}-{root.seat}/"
            f"turn-{root.turn_index:03d}-*.json"
        )
    )
    if len(matches) != 1:
        raise AblationError(f"source root has {len(matches)} records at declared address {root}")
    return matches[0]


def _load_source_records(
    source_root: Path,
) -> tuple[
    dict[SourceRoot, tuple[PublicDecisionRecord, Mapping[str, Any]]],
    dict[int, tuple[PublicDecisionRecord, ...]],
    list[dict[str, Any]],
]:
    complete = _read_json(source_root / "COMPLETE.json")
    if complete != {
        "games": 8,
        "pairs": 4,
        "schema_version": "pokezero.root-action-audit-recovery-complete.v1",
        "seeds": [2026092004, 2026092005, 2026092006, 2026092007],
        "status": "COMPLETE",
    }:
        raise AblationError("source root is not the exact completed R4 recovery")
    selected: dict[SourceRoot, tuple[PublicDecisionRecord, Mapping[str, Any]]] = {}
    wrappers: list[Mapping[str, Any]] = []
    for root in TARGETS:
        record, wrapper = _read_source_wrapper(_source_wrapper_path(source_root, root))
        if SourceRoot(record.seed, record.acting_player, record.turn_index) != root:
            raise AblationError(f"source record address disagrees with declared target {root}")
        selected[root] = (record, wrapper)
        wrappers.append(wrapper)
    candidate_hashes = {wrapper["candidate_provenance_sha256"] for wrapper in wrappers}
    raw_hashes = {wrapper["raw_provenance_sha256"] for wrapper in wrappers}
    if len(candidate_hashes) != 1 or len(raw_hashes) != 1:
        raise AblationError("source records disagree on R4 policy provenance")
    by_seed: dict[int, dict[str, PublicDecisionRecord]] = {}
    source_inventory: dict[str, dict[str, Any]] = {}

    def register_source(path: Path, record: PublicDecisionRecord, wrapper: Mapping[str, Any]) -> None:
        by_seed.setdefault(record.seed, {})[record.decision_id] = record
        identity = {
            "seed": record.seed,
            "seat": record.acting_player,
            "turn_index": record.turn_index,
            "decision_id": record.decision_id,
            "record_sha256": _sha256(record.to_dict()),
            "wrapper_sha256": sha256_file(path),
        }
        prior = source_inventory.setdefault(record.decision_id, identity)
        if prior != identity:
            raise AblationError(f"source record identity is inconsistent: {record.decision_id}")

    for root, (record, wrapper) in selected.items():
        register_source(_source_wrapper_path(source_root, root), record, wrapper)
    # Retain only the source-owned earlier decision records that an explicit
    # placeholder repair may consult.  Scanning every captured record is both
    # unnecessary and makes recovery depend on hundreds of unrelated files.
    for target, (record, _) in selected.items():
        for action_round in record.public_resolved_action_rounds:
            identifier = action_round.actions.get(record.acting_player)
            if (
                identifier is None
                or identifier.kind != "event"
                or identifier.event_id != "unresolved-public-event"
            ):
                continue
            source_root_address = SourceRoot(record.seed, record.acting_player, action_round.turn_index)
            source_record, source_wrapper = _read_source_wrapper(
                _source_wrapper_path(source_root, source_root_address)
            )
            if source_wrapper["candidate_provenance_sha256"] not in candidate_hashes or source_wrapper[
                "raw_provenance_sha256"
            ] not in raw_hashes:
                raise AblationError(f"{target}: repair source record provenance drifted")
            register_source(_source_wrapper_path(source_root, source_root_address), source_record, source_wrapper)
    return (
        selected,
        {seed: tuple(records.values()) for seed, records in by_seed.items()},
        [source_inventory[key] for key in sorted(source_inventory)],
    )


def _decision_seed(record: PublicDecisionRecord) -> int:
    """Fixed per root and identical for every arm of that root."""

    return int.from_bytes(hashlib.sha256(record.decision_id.encode("utf-8")).digest()[:8], "big")


def _rollout_seed(record: PublicDecisionRecord) -> int:
    """A separate, pinned leaf-pricing RNG root for the same public decision."""

    return int.from_bytes(
        hashlib.sha256(f"rollout-leaf:{record.decision_id}".encode("utf-8")).digest()[:8], "big"
    )


def _new_decider(
    contract: Any, args: argparse.Namespace, *, rollout_leaf_eval: bool, rollout_seed: int
) -> _LiveEngineTimingDecider:
    return _LiveEngineTimingDecider(
        contract,
        args.showdown_root,
        model_decision_time_ms=None,
        model_world_workers=1,
        model_priors=True,
        use_opponent_priors=False,
        override_telemetry=True,
        rollout_leaf_eval=rollout_leaf_eval,
        rollout_seed=rollout_seed,
        **ROLLOUT,
    )


def _finite(value: Any, *, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise AblationError(f"{field} must be finite")
    return float(value)


def _selection_witness(telemetry: Mapping[str, Any], *, rollout_leaf_eval: bool) -> dict[str, Any]:
    if telemetry.get("invalid_actions") != 0 or telemetry.get("fallbacks") != 0 or telemetry.get("prior_fallbacks") != 0:
        raise AblationError("source root action or prior fallback occurred")
    if not isinstance(telemetry.get("root_action"), str) or not telemetry["root_action"]:
        raise AblationError("source root has no serializable selected action")
    engine = telemetry.get("engine_mcts")
    if not isinstance(engine, Mapping):
        raise AblationError("source root has no engine MCTS telemetry")
    try:
        require_rollout_leaf_witness({"engine_mcts": engine}, rollout_leaf_eval=rollout_leaf_eval)
    except Exception as error:
        raise AblationError(f"rollout leaf witness is invalid: {error}") from error
    override = engine.get("override")
    if not isinstance(override, Mapping):
        raise AblationError("source root has no override telemetry")
    allocation = override.get("root_allocation")
    if not isinstance(allocation, Mapping) or set(allocation) != {"worlds", "prior_authority", "prior_cause", "arms"}:
        raise AblationError("source root has malformed root allocation")
    if allocation.get("prior_authority") is not True or allocation.get("prior_cause") is not None:
        raise AblationError("source root model priors were not authoritative")
    if not isinstance(allocation.get("worlds"), int) or allocation["worlds"] != SEARCH["worlds"]:
        raise AblationError("source root did not complete all requested worlds")
    arms = allocation.get("arms")
    if not isinstance(arms, list) or len(arms) < 2:
        raise AblationError("source root has fewer than two allocated actions")
    normalized_arms: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_action_indices: set[int] = set()
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise AblationError("source root allocation arm is malformed")
        move = arm.get("move")
        if not isinstance(move, str) or not move or move in seen:
            raise AblationError("source root allocation moves are malformed")
        seen.add(move)
        action_index = arm.get("action_index")
        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or not 0 <= action_index < ACTION_COUNT
        ):
            raise AblationError("source root allocation action indices are malformed")
        if action_index in seen_action_indices:
            raise AblationError("source root allocation action indices are not unique")
        seen_action_indices.add(action_index)
        visit_share = _finite(arm.get("visit_share"), field="root visit share")
        reported_prior = _finite(arm.get("reported_prior"), field="root reported prior")
        model_prior = _finite(arm.get("model_prior"), field="root model prior")
        q = arm.get("q")
        normalized_arms.append({
            "move": move,
            "action_index": action_index,
            "visit_share": visit_share,
            "reported_prior": reported_prior,
            "model_prior": model_prior,
            "q": None if q is None else _finite(q, field="root Q"),
        })
    if not math.isclose(sum(arm["visit_share"] for arm in normalized_arms), 1.0, abs_tol=2e-5):
        raise AblationError("source root visits do not conserve one")
    if not math.isclose(sum(arm["reported_prior"] for arm in normalized_arms), 1.0, abs_tol=2e-5):
        raise AblationError("source root reported priors do not conserve one")
    if not math.isclose(sum(arm["model_prior"] for arm in normalized_arms), 1.0, abs_tol=2e-5):
        raise AblationError("source root model priors do not conserve one")
    return {
        "root_action": telemetry["root_action"],
        "total_iterations": telemetry.get("total_iterations"),
        "model_evals": telemetry.get("model_evals"),
        "max_depth_reached": telemetry.get("max_depth_reached"),
        "root_allocation": {"worlds": allocation["worlds"], "arms": normalized_arms},
        "rollout_leaf": engine.get("rollout_leaf"),
    }


def _control_projection(witness: Mapping[str, Any]) -> dict[str, Any]:
    """Fields that must match for two deterministic model-leaf controls."""

    return {
        "root_action": witness["root_action"],
        "total_iterations": witness["total_iterations"],
        "model_evals": witness["model_evals"],
        "max_depth_reached": witness["max_depth_reached"],
        "root_allocation": witness["root_allocation"],
        "rollout_leaf": witness["rollout_leaf"],
    }


def _validate_persisted_selection(selection: Any, *, rollout_leaf_eval: bool) -> Mapping[str, Any]:
    """Revalidate a durable selection witness before a resume can trust it."""

    if not isinstance(selection, Mapping):
        raise AblationError("durable selection is not a mapping")
    root_action = selection.get("root_action")
    if not isinstance(root_action, str) or not root_action:
        raise AblationError("durable selection has no root action")
    for field in ("total_iterations", "model_evals", "max_depth_reached"):
        value = selection.get(field)
        if not isinstance(value, int) or value <= 0:
            raise AblationError(f"durable selection has invalid {field}")
    allocation = selection.get("root_allocation")
    if not isinstance(allocation, Mapping) or set(allocation) != {"worlds", "arms"}:
        raise AblationError("durable selection has malformed root allocation")
    if allocation.get("worlds") != SEARCH["worlds"]:
        raise AblationError("durable selection did not complete all requested worlds")
    arms = allocation.get("arms")
    if not isinstance(arms, list) or len(arms) < 2:
        raise AblationError("durable selection has fewer than two allocated actions")
    seen: set[str] = set()
    seen_action_indices: set[int] = set()
    normalized: list[dict[str, float | str | None]] = []
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise AblationError("durable selection allocation arm is malformed")
        move = arm.get("move")
        if not isinstance(move, str) or not move or move in seen:
            raise AblationError("durable selection allocation moves are malformed")
        seen.add(move)
        action_index = arm.get("action_index")
        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or not 0 <= action_index < ACTION_COUNT
        ):
            raise AblationError("durable selection allocation action indices are malformed")
        if action_index in seen_action_indices:
            raise AblationError("durable selection allocation action indices are not unique")
        seen_action_indices.add(action_index)
        normalized.append({
            "move": move,
            "action_index": action_index,
            "visit_share": _finite(arm.get("visit_share"), field="durable root visit share"),
            "reported_prior": _finite(arm.get("reported_prior"), field="durable reported prior"),
            "model_prior": _finite(arm.get("model_prior"), field="durable model prior"),
            "q": None if arm.get("q") is None else _finite(arm["q"], field="durable root Q"),
        })
    for field in ("visit_share", "reported_prior", "model_prior"):
        if not math.isclose(sum(float(arm[field]) for arm in normalized), 1.0, abs_tol=2e-5):
            raise AblationError(f"durable selection {field} values do not conserve one")
    try:
        require_rollout_leaf_witness(
            {"engine_mcts": {"rollout_leaf": selection.get("rollout_leaf")}},
            rollout_leaf_eval=rollout_leaf_eval,
        )
    except Exception as error:
        raise AblationError(f"durable rollout leaf witness is invalid: {error}") from error
    return selection


def _run_root(
    *, root: SourceRoot, record: PublicDecisionRecord, source_records: Sequence[PublicDecisionRecord],
    contract: Any, args: argparse.Namespace, manifest_sha256: str,
) -> dict[str, Any]:
    try:
        prefix = source_bound_replay_prefix(record, source_records=source_records)
    except SourceRootReplayError as error:
        raise AblationError(f"{record.decision_id}: source-bound replay repair failed: {error}") from error
    decision_seed = _decision_seed(record)
    rollout_seed = _rollout_seed(record)
    config = SearchConfig(**SEARCH)
    arm_rows: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        rollout_leaf_eval = arm == "rollout_leaf"
        decider = _new_decider(
            contract, args, rollout_leaf_eval=rollout_leaf_eval, rollout_seed=rollout_seed
        )
        try:
            started = time.perf_counter()
            telemetry = decider.prepare_public_decision(
                record,
                config,
                public_action_rounds=prefix.public_action_rounds,
                decision_rng_seed=decision_seed,
            )()
            witness = _selection_witness(telemetry, rollout_leaf_eval=rollout_leaf_eval)
            arm_rows[arm] = {"wall_seconds": round(time.perf_counter() - started, 6), "selection": witness}
        finally:
            decider.close()
    if _control_projection(arm_rows["model_control_a"]["selection"]) != _control_projection(
        arm_rows["model_control_b"]["selection"]
    ):
        raise AblationError(f"{record.decision_id}: model controls are not deterministic")
    if (
        arm_rows["rollout_leaf"]["selection"]["total_iterations"]
        != arm_rows["model_control_a"]["selection"]["total_iterations"]
    ):
        raise AblationError(f"{record.decision_id}: rollout arm changed fixed native search work")
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "COMPLETE",
        "manifest_sha256": manifest_sha256,
        "source": root.to_dict(),
        "record_sha256": _sha256(record.to_dict()),
        "decision_id": record.decision_id,
        "decision_rng_seed": decision_seed,
        "rollout_rng_seed": rollout_seed,
        "repairs": [repair.to_dict() for repair in prefix.repairs],
        "arms": arm_rows,
    }


def _manifest(
    *, args: argparse.Namespace, contract: Any,
    selected: Mapping[SourceRoot, tuple[PublicDecisionRecord, Mapping[str, Any]]],
    historical_baseline: Mapping[str, Any],
    execution_runtime: Mapping[str, Any],
    source_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    wrappers = [wrapper for _, wrapper in selected.values()]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(Path(args.source_root).resolve()),
        "source_complete_sha256": sha256_file(Path(args.source_root) / "COMPLETE.json"),
        "source_record_inventory": [dict(item) for item in source_inventory],
        "source_policy_provenance": {
            "candidate": wrappers[0]["candidate_provenance_sha256"],
            "raw": wrappers[0]["raw_provenance_sha256"],
        },
        "execution_runtime": dict(execution_runtime),
        "checkpoint": contract.to_manifest(),
        "historical_baseline": dict(historical_baseline),
        "targets": [root.to_dict() for root in TARGETS],
        "search": dict(SEARCH),
        "model_policy": {"model_priors": True, "use_opponent_priors": False, "override_telemetry": True},
        "rollout": dict(ROLLOUT),
        "arms": list(ARMS),
        "execution_plan": {
            "shard_count": args.shard_count,
            "root_assignment": {
                str(index): [root.to_dict() for root_index, root in enumerate(TARGETS) if root_index % args.shard_count == index]
                for index in range(args.shard_count)
            },
        },
    }


def _prepare_root(
    out_root: Path, manifest: Mapping[str, Any], *, resume: bool, require_existing_manifest: bool
) -> None:
    terminals = [out_root / name for name in ("PASS.json", "NONPASS.json") if (out_root / name).exists()]
    if terminals:
        raise AblationError(f"out root is terminal: {', '.join(path.name for path in terminals)}")
    path = out_root / "MANIFEST.json"
    if path.exists():
        if not resume:
            raise AblationError("out root exists; pass --resume to revalidate durable units")
        if _read_json(path) != manifest:
            raise AblationError("out root manifest differs from this frozen contract")
    elif out_root.exists() and any(out_root.iterdir()):
        raise AblationError("nonempty out root has no manifest")
    elif require_existing_manifest:
        raise AblationError("sharded workers require a prior single --prepare-only initialization")
    else:
        _atomic_json(path, manifest)
    _atomic_json(out_root / "RUNNING.json", {"schema_version": SCHEMA_VERSION, "state": "RUNNING"})


def _validate_completed_root(
    payload: Any, *, root: SourceRoot, record: PublicDecisionRecord, manifest_sha256: str
) -> None:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION or payload.get("state") != "COMPLETE":
        raise AblationError(f"{root}: durable root has invalid state")
    if payload.get("source") != root.to_dict() or payload.get("record_sha256") != _sha256(record.to_dict()):
        raise AblationError(f"{root}: durable root does not bind its source record")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise AblationError(f"{root}: durable root does not bind this run manifest")
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise AblationError(f"{root}: durable root has malformed arm coverage")
    controls: list[Mapping[str, Any]] = []
    for arm in ARMS:
        row = arms[arm]
        if not isinstance(row, Mapping) or not isinstance(row.get("selection"), Mapping):
            raise AblationError(f"{root}: durable arm is malformed")
        selection = _validate_persisted_selection(
            row["selection"], rollout_leaf_eval=arm == "rollout_leaf"
        )
        if arm.startswith("model_control"):
            controls.append(selection)
    if _control_projection(controls[0]) != _control_projection(controls[1]):
        raise AblationError(f"{root}: durable model controls disagree")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    out_root = Path(args.out_root).resolve()
    selected, by_seed, source_inventory = _load_source_records(source_root)
    historical_baseline = _validate_historical_baseline(source_root)
    execution_runtime = _execution_runtime(args, historical_baseline)
    try:
        contract = resolve_checkpoint_contract(
            args.checkpoint,
            expected_sha256=args.expected_checkpoint_sha256,
            model_device=args.model_device,
            showdown_root=args.showdown_root,
            showdown_source_sha256=execution_runtime["showdown_source"]["content_sha256"],
            expected_showdown_source_sha256=historical_baseline["historical_runtime"]["showdown_source"]["content_sha256"],
        )
    except ContractError as error:
        raise AblationError(f"checkpoint contract failed: {error}") from error
    if contract.checkpoint_sha256 != historical_baseline["historical_candidate_checkpoint_sha256"]:
        raise AblationError("live checkpoint does not match the archived R4 candidate checkpoint")
    manifest = _manifest(
        args=args,
        contract=contract,
        selected=selected,
        historical_baseline=historical_baseline,
        execution_runtime=execution_runtime,
        source_inventory=source_inventory,
    )
    manifest_sha256 = _sha256(manifest)
    _prepare_root(
        out_root,
        manifest,
        resume=args.resume,
        require_existing_manifest=args.shard_count > 1 and not args.prepare_only,
    )
    if args.prepare_only:
        return {
            "schema_version": SCHEMA_VERSION,
            "state": "PREPARED",
            "target_count": len(TARGETS),
        }
    completed: list[dict[str, Any]] = []
    owned_targets = tuple(
        root for index, root in enumerate(TARGETS) if index % args.shard_count == args.shard_index
    )
    if args.finalize_only:
        owned_targets = ()
    for root in owned_targets:
        record, _ = selected[root]
        root_dir = _root_directory(out_root, root)
        complete_path = root_dir / "COMPLETE.json"
        if complete_path.exists():
            payload = _read_json(complete_path)
            _validate_completed_root(
                payload, root=root, record=record, manifest_sha256=manifest_sha256
            )
        else:
            _atomic_json(out_root / "progress" / "current.json", {
                "schema_version": SCHEMA_VERSION,
                "state": "RUNNING",
                "completed_roots": sum(
                    (_root_directory(out_root, known_root) / "COMPLETE.json").exists()
                    for known_root in TARGETS
                ),
                "total_roots": len(TARGETS),
                "current": root.to_dict(),
                "worker_shard": {"index": args.shard_index, "count": args.shard_count},
            })
            payload = _run_root(
                root=root, record=record, source_records=by_seed[root.seed], contract=contract,
                args=args, manifest_sha256=manifest_sha256,
            )
            _create_terminal(complete_path, payload)
        completed.append(dict(payload))
    if args.shard_count > 1 and not args.finalize_only:
        _atomic_json(out_root / "shards" / f"shard-{args.shard_index}.json", {
            "schema_version": SCHEMA_VERSION,
            "state": "COMPLETE",
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "manifest_sha256": manifest_sha256,
            "roots": [root.to_dict() for root in owned_targets],
        })
        return {
            "schema_version": SCHEMA_VERSION,
            "state": "SHARD_COMPLETE",
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "root_count": len(completed),
        }
    all_completed: list[dict[str, Any]] = []
    if args.shard_count > 1:
        for shard_index in range(args.shard_count):
            receipt_path = out_root / "shards" / f"shard-{shard_index}.json"
            receipt = _read_json(receipt_path)
            expected_roots = [
                root.to_dict() for root_index, root in enumerate(TARGETS)
                if root_index % args.shard_count == shard_index
            ]
            if receipt != {
                "schema_version": SCHEMA_VERSION,
                "state": "COMPLETE",
                "shard_index": shard_index,
                "shard_count": args.shard_count,
                "manifest_sha256": manifest_sha256,
                "roots": expected_roots,
            }:
                raise AblationError(f"shard {shard_index}: receipt differs from frozen execution plan")
    for root in TARGETS:
        record, _ = selected[root]
        complete_path = _root_directory(out_root, root) / "COMPLETE.json"
        if not complete_path.exists():
            raise AblationError(f"{root}: finalization is missing a completed durable root")
        payload = _read_json(complete_path)
        _validate_completed_root(
            payload, root=root, record=record, manifest_sha256=manifest_sha256
        )
        all_completed.append(dict(payload))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "state": "PASS",
        "marker": "SOURCE_ROOT_LEAF_ABLATION_PASS",
        "root_count": len(all_completed),
        "targets": [root.to_dict() for root in TARGETS],
        "complete_root_sha256": _sha256(all_completed),
        "scope": {
            "selection_only": True,
            "requires_independent_paired_continuations_for_action_quality": True,
        },
    }
    _atomic_json(out_root / "SUMMARY.json", summary)
    _create_terminal(out_root / "PASS.json", summary)
    (out_root / "RUNNING.json").unlink(missing_ok=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out_root = Path(args.out_root).resolve()
    try:
        summary = _run(args)
    except Exception as error:
        try:
            if not (out_root / "PASS.json").exists() and not (out_root / "NONPASS.json").exists():
                _atomic_json(out_root / "last_error.json", {
                    "schema_version": SCHEMA_VERSION,
                    "state": "RETRYABLE_ERROR",
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
        except Exception:
            pass
        print(f"NONPASS: {error}", file=sys.stderr)
        return 1
    print("WROTE SOURCE ROOT LEAF ABLATION PASS", _canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
