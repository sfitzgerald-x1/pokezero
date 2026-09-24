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
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

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


SCHEMA_VERSION = "pokezero.source-root-leaf-ablation.v1"
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if args.finalize_only and (args.shard_index != 0 or args.shard_count != 1):
        parser.error("--finalize-only requires the default single finalizer shard")
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
        seed = payload.get("study", {}).get("seed") if isinstance(payload.get("study"), Mapping) else None
        if seed not in {2026092004, 2026092005, 2026092006, 2026092007}:
            raise AblationError(f"source manifest has an unexpected seed: {path}")
        rows.append({"seed": seed, "config_sha256": _sha256(config)})
    if len(rows) != 4 or {row["seed"] for row in rows} != {2026092004, 2026092005, 2026092006, 2026092007}:
        raise AblationError("source historical baseline manifests are incomplete")
    if len({row["config_sha256"] for row in rows}) != 1:
        raise AblationError("source historical baseline config differs across seeds")
    return {
        "historical_config": dict(HISTORICAL_MODEL_CONFIG),
        "historical_config_sha256": rows[0]["config_sha256"],
        "historical_reproduction": False,
        "historical_reproduction_reason": "public-source replay uses a newly pinned decision RNG",
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


def _load_source_records(source_root: Path) -> tuple[dict[SourceRoot, tuple[PublicDecisionRecord, Mapping[str, Any]]], dict[int, tuple[PublicDecisionRecord, ...]]]:
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
    for record, _ in selected.values():
        by_seed.setdefault(record.seed, {})[record.decision_id] = record
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
            by_seed.setdefault(record.seed, {})[source_record.decision_id] = source_record
    return selected, {seed: tuple(records.values()) for seed, records in by_seed.items()}


def _decision_seed(record: PublicDecisionRecord) -> int:
    """Fixed per root and identical for every arm of that root."""

    return int.from_bytes(hashlib.sha256(record.decision_id.encode("utf-8")).digest()[:8], "big")


def _new_decider(contract: Any, args: argparse.Namespace, *, rollout_leaf_eval: bool, decision_seed: int) -> _LiveEngineTimingDecider:
    return _LiveEngineTimingDecider(
        contract,
        args.showdown_root,
        model_decision_time_ms=None,
        model_world_workers=1,
        model_priors=True,
        use_opponent_priors=False,
        override_telemetry=True,
        rollout_leaf_eval=rollout_leaf_eval,
        rollout_seed=decision_seed,
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
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise AblationError("source root allocation arm is malformed")
        move = arm.get("move")
        if not isinstance(move, str) or not move or move in seen:
            raise AblationError("source root allocation moves are malformed")
        seen.add(move)
        visit_share = _finite(arm.get("visit_share"), field="root visit share")
        reported_prior = _finite(arm.get("reported_prior"), field="root reported prior")
        model_prior = _finite(arm.get("model_prior"), field="root model prior")
        q = arm.get("q")
        normalized_arms.append({
            "move": move,
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
    normalized: list[dict[str, float | str | None]] = []
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise AblationError("durable selection allocation arm is malformed")
        move = arm.get("move")
        if not isinstance(move, str) or not move or move in seen:
            raise AblationError("durable selection allocation moves are malformed")
        seen.add(move)
        normalized.append({
            "move": move,
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
    *, root: SourceRoot, record: PublicDecisionRecord, source_records: Sequence[PublicDecisionRecord], contract: Any, args: argparse.Namespace
) -> dict[str, Any]:
    try:
        prefix = source_bound_replay_prefix(record, source_records=source_records)
    except SourceRootReplayError as error:
        raise AblationError(f"{record.decision_id}: source-bound replay repair failed: {error}") from error
    decision_seed = _decision_seed(record)
    config = SearchConfig(**SEARCH)
    arm_rows: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        rollout_leaf_eval = arm == "rollout_leaf"
        decider = _new_decider(
            contract, args, rollout_leaf_eval=rollout_leaf_eval, decision_seed=decision_seed
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
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "COMPLETE",
        "source": root.to_dict(),
        "record_sha256": _sha256(record.to_dict()),
        "decision_id": record.decision_id,
        "decision_rng_seed": decision_seed,
        "repairs": [repair.to_dict() for repair in prefix.repairs],
        "arms": arm_rows,
    }


def _manifest(
    *, args: argparse.Namespace, contract: Any,
    selected: Mapping[SourceRoot, tuple[PublicDecisionRecord, Mapping[str, Any]]],
    historical_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    wrappers = [wrapper for _, wrapper in selected.values()]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(Path(args.source_root).resolve()),
        "source_complete_sha256": sha256_file(Path(args.source_root) / "COMPLETE.json"),
        "source_policy_provenance": {
            "candidate": wrappers[0]["candidate_provenance_sha256"],
            "raw": wrappers[0]["raw_provenance_sha256"],
        },
        "checkpoint": contract.to_manifest(),
        "historical_baseline": dict(historical_baseline),
        "targets": [root.to_dict() for root in TARGETS],
        "search": dict(SEARCH),
        "model_policy": {"model_priors": True, "use_opponent_priors": False, "override_telemetry": True},
        "rollout": dict(ROLLOUT),
        "arms": list(ARMS),
    }


def _prepare_root(out_root: Path, manifest: Mapping[str, Any], *, resume: bool) -> None:
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
    else:
        _atomic_json(path, manifest)
    _atomic_json(out_root / "RUNNING.json", {"schema_version": SCHEMA_VERSION, "state": "RUNNING"})


def _validate_completed_root(payload: Any, *, root: SourceRoot, record: PublicDecisionRecord) -> None:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION or payload.get("state") != "COMPLETE":
        raise AblationError(f"{root}: durable root has invalid state")
    if payload.get("source") != root.to_dict() or payload.get("record_sha256") != _sha256(record.to_dict()):
        raise AblationError(f"{root}: durable root does not bind its source record")
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
    selected, by_seed = _load_source_records(source_root)
    historical_baseline = _validate_historical_baseline(source_root)
    try:
        contract = resolve_checkpoint_contract(
            args.checkpoint,
            expected_sha256=args.expected_checkpoint_sha256,
            model_device=args.model_device,
            showdown_root=args.showdown_root,
        )
    except ContractError as error:
        raise AblationError(f"checkpoint contract failed: {error}") from error
    manifest = _manifest(
        args=args,
        contract=contract,
        selected=selected,
        historical_baseline=historical_baseline,
    )
    _prepare_root(out_root, manifest, resume=args.resume)
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
            _validate_completed_root(payload, root=root, record=record)
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
                root=root, record=record, source_records=by_seed[root.seed], contract=contract, args=args
            )
            _create_terminal(complete_path, payload)
        completed.append(dict(payload))
    if args.shard_count > 1:
        _atomic_json(out_root / "shards" / f"shard-{args.shard_index}.json", {
            "schema_version": SCHEMA_VERSION,
            "state": "COMPLETE",
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
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
    for root in TARGETS:
        record, _ = selected[root]
        complete_path = _root_directory(out_root, root) / "COMPLETE.json"
        if not complete_path.exists():
            raise AblationError(f"{root}: finalization is missing a completed durable root")
        payload = _read_json(complete_path)
        _validate_completed_root(payload, root=root, record=record)
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
