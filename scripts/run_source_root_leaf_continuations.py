#!/usr/bin/env python3
"""Score changed source-root leaf selections with paired terminal continuations.

This is the causal second stage of ``run_source_root_leaf_ablation.py``.  The
ablation tells us whether replacing only the leaf evaluator changes a root
selection.  It intentionally does *not* claim that a changed selection is
better.  This runner makes that latter comparison without persisting a
simulator snapshot or the fixed opponent action:

* it consumes only a terminal, source-bound leaf-ablation root;
* it includes every root where the model and rollout leaf selected different
  actions--never a convenient outcome-selected subset;
* each root restores the same replayed source boundary for both candidate
  actions, holds one independently sampled opponent action fixed, and uses
  common continuation RNGs within each target/trial;
* every root is a create-only terminal unit, allowing a restarted sharded
  launcher to reuse finished roots and redo only the interrupted one.

The result is an action-quality diagnostic, not a game-strength measurement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if SCRIPTS.is_dir():
    sys.path.insert(0, str(SCRIPTS))

from pokezero.local_showdown import (  # noqa: E402
    LocalShowdownConfig,
    LocalShowdownEnv,
    env_config_from_checkpoint_provenance,
)
from pokezero.mcts_eval.lattice import _LiveEngineTimingDecider  # noqa: E402
from pokezero.mcts_eval.source_root_replay import source_bound_replay_prefix  # noqa: E402
from pokezero.neural_policy import (  # noqa: E402
    TransformerSoftmaxPolicy,
    category_vocab_from_model_config,
    feature_masks_from_model_config,
    load_transformer_checkpoint,
    load_transformer_model_config,
    observation_spec_from_model_config,
)
from pokezero.policy import RandomLegalPolicy  # noqa: E402
from pokezero.public_replay_materializer import replay_public_action_rounds  # noqa: E402
from pokezero.rollout import RolloutConfig  # noqa: E402
from pokezero.sealed_override_continuation import (  # noqa: E402
    evaluate_sealed_root_action_grid,
)
from pokezero.showdown import showdown_choice_for_action  # noqa: E402

# The first-stage runner is the frozen source/input contract.  Reusing its
# validators is intentional: a continuation study may not adopt a look-alike
# source tree, target list, or source-repair rule.
from run_source_root_leaf_ablation import (  # noqa: E402
    SCHEMA_VERSION as ABLATION_SCHEMA_VERSION,
    SourceRoot,
    TARGETS,
    _canonical_json,
    _load_source_records,
    _read_json,
    _sha256,
    _validate_completed_root as _validate_leaf_completed_root,
)


SCHEMA_VERSION = "pokezero.source-root-leaf-continuation.v1"
LEAF_ARMS = ("model_control_a", "model_control_b", "rollout_leaf")
CONTINUATION_TARGETS = ("policy_consistent", "uniform_own")
CONTINUATION_RNG_SEEDS = tuple(range(16))
INITIAL_MAX_CONTINUATION_DECISION_ROUNDS = 250
EXPANDED_MAX_CONTINUATION_DECISION_ROUNDS = 1024


class ContinuationError(RuntimeError):
    """The continuation experiment cannot safely produce a result."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ContinuationError(f"refusing to replace terminal artifact: {path}")
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
            raise ContinuationError(f"refusing to replace terminal artifact: {path}") from error
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_require_identical_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a terminal artifact once, or safely reuse an identical one.

    A process can be interrupted after publishing one terminal witness but
    before its sibling witness.  Retrying must complete that publication
    window without replacing the first witness or silently adopting a drifted
    one.
    """

    if path.exists():
        if _read_json(path) != payload:
            raise ContinuationError(f"existing terminal artifact differs from frozen result: {path}")
        return
    _write_create_only_json(path, payload)


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace only the explicitly nonterminal liveness pointer."""

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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--leaf-ablation-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
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
    if re.fullmatch(r"[0-9a-f]{40}", args.expected_source_commit) is None:
        parser.error("--expected-source-commit must be a lowercase full Git SHA")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if args.finalize_only and args.shard_index != 0:
        parser.error("--finalize-only requires shard index zero")
    return args


def _root_directory(out_root: Path, root: SourceRoot) -> Path:
    return out_root / "roots" / f"seed-{root.seed}-{root.seat}-turn-{root.turn_index:03d}"


def _selection_projection(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContinuationError("leaf-ablation arm has no selection witness")
    return value


def _load_changed_leaf_roots(
    leaf_root: Path,
    source_records: Mapping[SourceRoot, tuple[Any, Mapping[str, Any]]],
) -> tuple[dict[SourceRoot, Mapping[str, Any]], Mapping[str, Any], str]:
    """Validate the finished leaf ablation and select every changed root."""

    manifest_path = leaf_root / "MANIFEST.json"
    pass_path = leaf_root / "PASS.json"
    summary_path = leaf_root / "SUMMARY.json"
    manifest = _read_json(manifest_path)
    passed = _read_json(pass_path)
    summary = _read_json(summary_path)
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != ABLATION_SCHEMA_VERSION:
        raise ContinuationError("leaf-ablation manifest has an unsupported schema")
    if not isinstance(passed, Mapping) or passed != summary:
        raise ContinuationError("leaf-ablation PASS and SUMMARY disagree")
    if passed.get("state") != "PASS" or passed.get("root_count") != len(TARGETS):
        raise ContinuationError("leaf-ablation root is not a complete PASS")
    if passed.get("targets") != [root.to_dict() for root in TARGETS]:
        raise ContinuationError("leaf-ablation PASS target registration drifted")
    manifest_sha256 = _sha256(manifest)
    completed: list[Mapping[str, Any]] = []
    changed: dict[SourceRoot, Mapping[str, Any]] = {}
    for root in TARGETS:
        record, _ = source_records[root]
        complete_path = _root_directory(leaf_root, root) / "COMPLETE.json"
        payload = _read_json(complete_path)
        _validate_leaf_completed_root(payload, root=root, record=record, manifest_sha256=manifest_sha256)
        if not isinstance(payload, Mapping):
            raise ContinuationError("leaf-ablation completed root is not a mapping")
        completed.append(payload)
        arms = payload.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(LEAF_ARMS):
            raise ContinuationError(f"{root}: leaf-ablation arms drifted")
        model = _selection_projection(arms["model_control_a"].get("selection"))
        control = _selection_projection(arms["model_control_b"].get("selection"))
        rollout = _selection_projection(arms["rollout_leaf"].get("selection"))
        if model != control:
            raise ContinuationError(f"{root}: model-leaf reproducibility control differs")
        if model.get("total_iterations") != rollout.get("total_iterations"):
            raise ContinuationError(f"{root}: rollout leaf changed fixed search work")
        model_action, rollout_action = model.get("root_action"), rollout.get("root_action")
        if not isinstance(model_action, str) or not model_action or not isinstance(rollout_action, str) or not rollout_action:
            raise ContinuationError(f"{root}: leaf-ablation action is malformed")
        if model_action != rollout_action:
            changed[root] = payload
    if passed.get("complete_root_sha256") != _sha256(completed):
        raise ContinuationError("leaf-ablation PASS root hash differs from durable units")
    if not changed:
        raise ContinuationError("leaf-ablation selected no changed roots to score")
    return changed, manifest, manifest_sha256


def _source_code_provenance(*, expected_commit: str) -> Mapping[str, str]:
    stamped = os.environ.get("POKEZERO_COMMIT", "").strip().lower()
    if stamped != expected_commit:
        raise ContinuationError("executing source commit does not match the frozen contract")
    paths = (ROOT / "scripts" / "run_source_root_leaf_continuations.py",)
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(payload)
    return {"commit": stamped, "runner_sha256": digest.hexdigest()}


def _manifest(
    *, args: argparse.Namespace, changed: Mapping[SourceRoot, Mapping[str, Any]],
    leaf_manifest: Mapping[str, Any], leaf_manifest_sha256: str,
) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(Path(args.source_root).resolve()),
        "source_complete_sha256": _sha256_file(Path(args.source_root) / "COMPLETE.json"),
        "leaf_ablation_root": str(Path(args.leaf_ablation_root).resolve()),
        "leaf_ablation_manifest_sha256": leaf_manifest_sha256,
        "leaf_ablation_pass_sha256": _sha256_file(Path(args.leaf_ablation_root) / "PASS.json"),
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "source_code": dict(_source_code_provenance(expected_commit=args.expected_source_commit)),
        "changed_targets": [root.to_dict() for root in changed],
        "continuation": {
            "targets": list(CONTINUATION_TARGETS),
            "rng_seeds": list(CONTINUATION_RNG_SEEDS),
            "initial_max_decision_rounds": INITIAL_MAX_CONTINUATION_DECISION_ROUNDS,
            "expanded_max_decision_rounds": EXPANDED_MAX_CONTINUATION_DECISION_ROUNDS,
            "opponent_selector": {
                "kind": "sampled_raw_policy",
                "deterministic": False,
                "sampling_temperature": 1.0,
                "shared_across_subject_actions": True,
            },
            "scope": "changed-root-action-quality-only",
        },
        "execution_plan": {
            "shard_count": args.shard_count,
            "root_assignment": {
                str(index): [root.to_dict() for number, root in enumerate(changed) if number % args.shard_count == index]
                for index in range(args.shard_count)
            },
        },
        # Bind the exact first-stage protocol, but do not copy its potentially
        # large source inventory into a second artifact.
        "leaf_ablation_contract": {
            "schema_version": leaf_manifest.get("schema_version"),
            "search": leaf_manifest.get("search"),
            "model_policy": leaf_manifest.get("model_policy"),
            "rollout": leaf_manifest.get("rollout"),
        },
    }


def _prepare(out_root: Path, manifest: Mapping[str, Any], *, resume: bool, require_existing: bool) -> None:
    terminals = [out_root / name for name in ("PASS.json", "NONPASS.json") if (out_root / name).exists()]
    if terminals:
        raise ContinuationError(f"out root is terminal: {', '.join(path.name for path in terminals)}")
    path = out_root / "MANIFEST.json"
    if path.exists():
        if not resume:
            raise ContinuationError("out root exists; pass --resume to validate completed roots")
        if _read_json(path) != manifest:
            raise ContinuationError("out root manifest differs from this frozen contract")
    elif out_root.exists() and any(out_root.iterdir()):
        raise ContinuationError("nonempty out root has no manifest")
    elif require_existing:
        raise ContinuationError("sharded workers require a prior --prepare-only initialization")
    else:
        _write_create_only_json(path, manifest)
    _write_progress(out_root / "RUNNING.json", {"schema_version": SCHEMA_VERSION, "state": "RUNNING"})


def _choice_index(env: LocalShowdownEnv, seat: str, choice: str) -> int:
    observation = env.observe(seat)
    matches = []
    for index, legal in enumerate(observation.legal_action_mask):
        if not legal:
            continue
        rendered = showdown_choice_for_action(env._state_for_player(seat), index)
        if rendered == choice:
            matches.append(index)
    if len(matches) != 1:
        raise ContinuationError(f"{seat}: source choice {choice!r} maps to {len(matches)} legal actions")
    return matches[0]


def _selector_seed(record: Any) -> int:
    return int.from_bytes(hashlib.sha256(f"opponent:{record.decision_id}".encode()).digest()[:8], "big")


def _source_histories(replay: Any, current: Mapping[str, Any]) -> Mapping[str, tuple[Any, ...]]:
    histories: dict[str, list[Any]] = {"p1": [], "p2": []}
    for turn_index in sorted(replay.replay_observations):
        observations = replay.replay_observations[turn_index]
        for seat in histories:
            observation = observations.get(seat)
            if observation is not None:
                histories[seat].append(observation)
    for seat, observation in current.items():
        if seat not in histories:
            raise ContinuationError("source root requested an unsupported player")
        histories[seat].append(observation)
    if any(not history for history in histories.values()):
        raise ContinuationError("source root has an empty policy history")
    return {seat: tuple(history) for seat, history in histories.items()}


def _new_sampled_policy(
    *, model: Any, result: Any, device: str, seat: str, history: Sequence[Any], deterministic: bool,
) -> TransformerSoftmaxPolicy:
    return TransformerSoftmaxPolicy(
        model=model,
        result=result,
        device=device,
        deterministic=deterministic,
        exploration_epsilon=0.0,
        sampling_temperature=1.0,
        family_gated_selection=False,
        _history_by_player={seat: list(history)},
    )


@dataclass(frozen=True)
class _RootPlan:
    root: SourceRoot
    model_choice: str
    rollout_choice: str
    model_action: int
    rollout_action: int
    histories: Mapping[str, tuple[Any, ...]]
    snapshot: Any
    opponent_seat: str
    opponent_action: int
    source_battle_id: str


def _root_plan(
    *, root: SourceRoot, record: Any, source_records: Sequence[Any], leaf_payload: Mapping[str, Any],
    env_config: Any, model: Any, result: Any, device: str,
) -> _RootPlan:
    prefix = source_bound_replay_prefix(record, source_records=source_records)
    env = LocalShowdownEnv(env_config)
    try:
        replay = replay_public_action_rounds(
            env,
            seed=record.seed,
            format_id="gen3randombattle",
            public_action_rounds=prefix.public_action_rounds,
            start_override=None,
        )
        if replay.terminal is not None:
            raise ContinuationError(f"{root}: replay reached terminal before the source boundary")
        if tuple(replay.requested_players) != ("p1", "p2"):
            raise ContinuationError(f"{root}: source boundary is not simultaneous")
        current = {seat: env.observe(seat) for seat in ("p1", "p2")}
        if not _LiveEngineTimingDecider._same_source_public_observation(
            current[root.seat], record.observation, record.public_belief_view
        ):
            raise ContinuationError(f"{root}: replayed source observation or belief drifted")
        histories = _source_histories(replay, current)
        arms = leaf_payload["arms"]
        model_choice = arms["model_control_a"]["selection"]["root_action"]
        rollout_choice = arms["rollout_leaf"]["selection"]["root_action"]
        model_action = _choice_index(env, root.seat, model_choice)
        rollout_action = _choice_index(env, root.seat, rollout_choice)
        if model_action == rollout_action:
            raise ContinuationError(f"{root}: changed root mapped to identical action indexes")
        opponent_seat = "p2" if root.seat == "p1" else "p1"
        # The opposing source action is intentionally never written.  Its
        # deterministic selector seed derives only from the public root id and
        # it is shared across the two subject actions and all suffix trials.
        selector = _new_sampled_policy(
            model=model,
            result=result,
            device=device,
            seat=opponent_seat,
            history=histories[opponent_seat][:-1],
            deterministic=False,
        )
        opponent_action = selector.select_action(
            current[opponent_seat], rng=random.Random(_selector_seed(record))
        ).action_index
        snapshotter = getattr(env, "snapshot_actionable_boundary", None)
        if not callable(snapshotter):
            raise ContinuationError(f"{root}: environment cannot snapshot the source boundary")
        snapshot = snapshotter()
        return _RootPlan(
            root=root,
            model_choice=model_choice,
            rollout_choice=rollout_choice,
            model_action=model_action,
            rollout_action=rollout_action,
            histories=histories,
            snapshot=snapshot,
            opponent_seat=opponent_seat,
            opponent_action=opponent_action,
            source_battle_id=record.battle_id,
        )
    finally:
        env.close()


def _evaluate_root(
    *, plan: _RootPlan, record: Any, env_config: Any, model: Any, result: Any, device: str,
) -> Mapping[str, Any]:
    subject = plan.root.seat
    opponent = plan.opponent_seat

    def sampled_raw(seat: str) -> TransformerSoftmaxPolicy:
        return _new_sampled_policy(
            model=model, result=result, device=device, seat=seat, history=plan.histories[seat], deterministic=False,
        )

    def policy_consistent() -> Mapping[str, Any]:
        return {"p1": sampled_raw("p1"), "p2": sampled_raw("p2")}

    def uniform_own() -> Mapping[str, Any]:
        return {subject: RandomLegalPolicy(), opponent: sampled_raw(opponent)}

    grid = evaluate_sealed_root_action_grid(
        snapshot=plan.snapshot,
        source_battle_id=plan.source_battle_id,
        source_seed=record.seed,
        source_decision_round=record.turn_index,
        subject_player=subject,
        actions={"model_leaf": plan.model_action, "rollout_leaf": plan.rollout_action},
        opponent_player=opponent,
        opponent_action=plan.opponent_action,
        continuation_policy_factories={
            "policy_consistent": policy_consistent,
            "uniform_own": uniform_own,
        },
        continuation_rng_seeds=CONTINUATION_RNG_SEEDS,
        search_evidence={
            "model_leaf_choice": plan.model_choice,
            "rollout_leaf_choice": plan.rollout_choice,
            "selection_changed": True,
            "leaf_only_intervention": True,
        },
        env_factory=lambda: LocalShowdownEnv(env_config),
        rollout_config=RolloutConfig(
            max_decision_rounds=INITIAL_MAX_CONTINUATION_DECISION_ROUNDS,
            format_id="gen3randombattle",
            record_policy_timing=False,
            hide_opponent_legal_action_masks=True,
        ),
        max_continuation_decision_rounds=INITIAL_MAX_CONTINUATION_DECISION_ROUNDS,
        expanded_max_continuation_decision_rounds=EXPANDED_MAX_CONTINUATION_DECISION_ROUNDS,
    )
    if grid.get("opponent_action_held_fixed") is not True or "opponent_action" in grid or "snapshot" in grid:
        raise ContinuationError(f"{plan.root}: continuation grid leaked or omitted its fixed-opponent binding")
    return grid


def _validate_completed_root(
    payload: Any,
    *,
    root: SourceRoot,
    record: Any,
    leaf_payload: Mapping[str, Any],
    expected_actions: Mapping[str, int],
    manifest_sha256: str,
) -> None:
    """Validate a reusable root against all frozen inputs and producer bounds."""

    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION or payload.get("state") != "COMPLETE":
        raise ContinuationError(f"{root}: completed continuation root has invalid state")
    if payload.get("source") != root.to_dict() or payload.get("manifest_sha256") != manifest_sha256:
        raise ContinuationError(f"{root}: completed continuation root binding drifted")
    if payload.get("source_record_sha256") != _sha256(record.to_dict()):
        raise ContinuationError(f"{root}: completed continuation root source record drifted")
    if payload.get("leaf_complete_sha256") != _sha256(leaf_payload):
        raise ContinuationError(f"{root}: completed continuation root leaf selection drifted")
    grid = payload.get("grid")
    if not isinstance(grid, Mapping) or grid.get("schema_version") != "pokezero.sealed-root-action-grid.v2":
        raise ContinuationError(f"{root}: completed continuation grid is invalid")
    if grid.get("opponent_action_held_fixed") is not True or "opponent_action" in grid or "snapshot" in grid:
        raise ContinuationError(f"{root}: completed grid leaks or omits fixed-opponent evidence")
    expected_grid_identity = {
        "source_battle_id": record.battle_id,
        "source_seed": record.seed,
        "source_decision_round": record.turn_index,
        "subject_player": root.seat,
        "opponent_player": "p2" if root.seat == "p1" else "p1",
    }
    if any(grid.get(key) != value for key, value in expected_grid_identity.items()):
        raise ContinuationError(f"{root}: completed grid source identity drifted")
    arms = leaf_payload.get("arms")
    if not isinstance(arms, Mapping):
        raise ContinuationError(f"{root}: changed leaf payload lacks arm evidence")
    expected_evidence = {
        "model_leaf_choice": arms["model_control_a"]["selection"]["root_action"],
        "rollout_leaf_choice": arms["rollout_leaf"]["selection"]["root_action"],
        "selection_changed": True,
        "leaf_only_intervention": True,
    }
    if grid.get("search_evidence") != expected_evidence:
        raise ContinuationError(f"{root}: completed grid search evidence drifted")
    actions = grid.get("actions")
    expected_action_rows = [
        {"action_label": label, "action_index": expected_actions[label]}
        for label in ("model_leaf", "rollout_leaf")
    ]
    if actions != expected_action_rows:
        raise ContinuationError(f"{root}: completed grid action registration drifted")
    targets = grid.get("continuation_targets")
    if not isinstance(targets, list) or [entry.get("target") for entry in targets if isinstance(entry, Mapping)] != list(CONTINUATION_TARGETS):
        raise ContinuationError(f"{root}: completed grid continuation target registration drifted")
    for target in targets:
        if not isinstance(target, Mapping) or not isinstance(target.get("trials"), list):
            raise ContinuationError(f"{root}: completed target has invalid trials")
        if len(target["trials"]) != len(CONTINUATION_RNG_SEEDS) or [trial.get("continuation_rng_seed") for trial in target["trials"] if isinstance(trial, Mapping)] != list(CONTINUATION_RNG_SEEDS):
            raise ContinuationError(f"{root}: continuation RNG schedule drifted")
        for trial in target["trials"]:
            if not isinstance(trial, Mapping) or not isinstance(trial.get("outcomes"), list):
                raise ContinuationError(f"{root}: continuation trial is malformed")
            outcomes = trial["outcomes"]
            if len(outcomes) != 2 or [outcome.get("action_label") for outcome in outcomes if isinstance(outcome, Mapping)] != ["model_leaf", "rollout_leaf"]:
                raise ContinuationError(f"{root}: continuation paired actions drifted")
            for action_label, outcome in zip(("model_leaf", "rollout_leaf"), outcomes, strict=True):
                if not isinstance(outcome, Mapping) or outcome.get("action_index") != expected_actions[action_label]:
                    raise ContinuationError(f"{root}: continuation action index drifted")
                continuation = outcome.get("continuation") if isinstance(outcome, Mapping) else None
                terminal = continuation.get("terminal") if isinstance(continuation, Mapping) else None
                if not isinstance(terminal, Mapping) or terminal.get("capped") is not False:
                    raise ContinuationError(f"{root}: continuation is incomplete or capped")
                if "winner" not in terminal or terminal.get("winner") not in ("p1", "p2", None):
                    raise ContinuationError(f"{root}: continuation terminal winner is invalid")
                if isinstance(terminal.get("turn_count"), bool) or not isinstance(terminal.get("turn_count"), int) or terminal["turn_count"] < 0:
                    raise ContinuationError(f"{root}: continuation terminal turn count is invalid")
                decision_count = continuation.get("decision_round_count")
                fixed_step_terminal = continuation.get("terminal_after_fixed_joint_step")
                cap_retry = continuation.get("cap_retry")
                initial_ceiling = continuation.get("initial_max_continuation_decision_rounds")
                effective_ceiling = continuation.get("effective_max_continuation_decision_rounds")
                if isinstance(decision_count, bool) or not isinstance(decision_count, int) or decision_count < 0:
                    raise ContinuationError(f"{root}: continuation decision count is invalid")
                if not isinstance(fixed_step_terminal, bool) or not isinstance(cap_retry, bool):
                    raise ContinuationError(f"{root}: continuation terminal metadata is invalid")
                if initial_ceiling != INITIAL_MAX_CONTINUATION_DECISION_ROUNDS:
                    raise ContinuationError(f"{root}: continuation initial cap drifted")
                expected_ceiling = EXPANDED_MAX_CONTINUATION_DECISION_ROUNDS if cap_retry else INITIAL_MAX_CONTINUATION_DECISION_ROUNDS
                if effective_ceiling != expected_ceiling or decision_count > expected_ceiling:
                    raise ContinuationError(f"{root}: continuation effective cap drifted")
                if cap_retry and decision_count <= INITIAL_MAX_CONTINUATION_DECISION_ROUNDS:
                    raise ContinuationError(f"{root}: continuation cap retry has no expanded work")
                if fixed_step_terminal and (decision_count != 0 or cap_retry):
                    raise ContinuationError(f"{root}: fixed-step terminal metadata drifted")


def _run(args: argparse.Namespace) -> Mapping[str, Any]:
    source_root = Path(args.source_root).resolve()
    leaf_root = Path(args.leaf_ablation_root).resolve()
    out_root = Path(args.out_root).resolve()
    if _sha256_file(Path(args.checkpoint)) != args.expected_checkpoint_sha256:
        raise ContinuationError("checkpoint SHA-256 does not match the frozen contract")
    source_selected, source_by_seed, _ = _load_source_records(source_root)
    changed, leaf_manifest, leaf_manifest_sha256 = _load_changed_leaf_roots(leaf_root, source_selected)
    leaf_checkpoint = leaf_manifest.get("checkpoint")
    if not isinstance(leaf_checkpoint, Mapping) or leaf_checkpoint.get("checkpoint_sha256") != args.expected_checkpoint_sha256:
        raise ContinuationError("continuation checkpoint does not match the source-root leaf-ablation checkpoint")
    manifest = _manifest(
        args=args, changed=changed, leaf_manifest=leaf_manifest, leaf_manifest_sha256=leaf_manifest_sha256,
    )
    manifest_sha256 = _sha256(manifest)
    _prepare(out_root, manifest, resume=args.resume, require_existing=args.shard_count > 1 and not args.prepare_only)
    if args.prepare_only:
        return {"schema_version": SCHEMA_VERSION, "state": "PREPARED", "root_count": len(changed)}
    if args.finalize_only:
        owned = ()
    else:
        owned = tuple(root for number, root in enumerate(changed) if number % args.shard_count == args.shard_index)

    model_config = load_transformer_model_config(args.checkpoint)
    vocabulary = category_vocab_from_model_config(model_config, args.showdown_root)
    env_config = env_config_from_checkpoint_provenance(
        LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True, category_vocab=vocabulary),
        feature_masks_from_model_config(model_config),
        required_specs=observation_spec_from_model_config(model_config),
        required_vocabs=vocabulary,
        context="source-root leaf continuation diagnostic",
    )
    model, result = load_transformer_checkpoint(args.checkpoint, map_location=args.device)
    completed: list[Mapping[str, Any]] = []
    for root in owned:
        record, _ = source_selected[root]
        complete_path = _root_directory(out_root, root) / "COMPLETE.json"
        # Rebuild the source boundary even while resuming.  This validates the
        # durable action indexes against the live, source-bound replay rather
        # than trusting labels copied from a prior process.
        plan = _root_plan(
            root=root, record=record, source_records=source_by_seed[root.seed], leaf_payload=changed[root],
            env_config=env_config, model=model, result=result, device=args.device,
        )
        expected_actions = {"model_leaf": plan.model_action, "rollout_leaf": plan.rollout_action}
        if complete_path.exists():
            payload = _read_json(complete_path)
            _validate_completed_root(
                payload, root=root, record=record, leaf_payload=changed[root],
                expected_actions=expected_actions, manifest_sha256=manifest_sha256,
            )
        else:
            _write_progress(out_root / "progress" / "current.json", {
                "schema_version": SCHEMA_VERSION, "state": "RUNNING",
                "completed_roots": sum((_root_directory(out_root, known) / "COMPLETE.json").exists() for known in changed),
                "total_roots": len(changed), "current": root.to_dict(),
                "worker_shard": {"index": args.shard_index, "count": args.shard_count},
            })
            payload = {
                "schema_version": SCHEMA_VERSION,
                "state": "COMPLETE",
                "manifest_sha256": manifest_sha256,
                "source": root.to_dict(),
                "source_record_sha256": _sha256(record.to_dict()),
                "leaf_complete_sha256": _sha256(changed[root]),
                "grid": _evaluate_root(plan=plan, record=record, env_config=env_config, model=model, result=result, device=args.device),
            }
            _write_create_only_json(complete_path, payload)
        completed.append(payload)
    if args.shard_count > 1 and not args.finalize_only:
        _write_or_require_identical_json(out_root / "shards" / f"shard-{args.shard_index}.json", {
            "schema_version": SCHEMA_VERSION, "state": "COMPLETE", "shard_index": args.shard_index,
            "shard_count": args.shard_count, "manifest_sha256": manifest_sha256,
            "roots": [root.to_dict() for root in owned],
        })
        return {"schema_version": SCHEMA_VERSION, "state": "SHARD_COMPLETE", "root_count": len(completed)}
    all_completed: list[Mapping[str, Any]] = []
    for root in changed:
        record, _ = source_selected[root]
        plan = _root_plan(
            root=root, record=record, source_records=source_by_seed[root.seed], leaf_payload=changed[root],
            env_config=env_config, model=model, result=result, device=args.device,
        )
        payload = _read_json(_root_directory(out_root, root) / "COMPLETE.json")
        _validate_completed_root(
            payload, root=root, record=record, leaf_payload=changed[root],
            expected_actions={"model_leaf": plan.model_action, "rollout_leaf": plan.rollout_action},
            manifest_sha256=manifest_sha256,
        )
        all_completed.append(payload)
    summary = {
        "schema_version": SCHEMA_VERSION, "state": "PASS", "marker": "SOURCE_ROOT_LEAF_CONTINUATION_PASS",
        "root_count": len(all_completed), "continuation_unit_count": len(all_completed) * len(CONTINUATION_TARGETS) * len(CONTINUATION_RNG_SEEDS) * 2,
        "complete_root_sha256": _sha256(all_completed),
        "scope": "source-root action quality only; not game strength",
    }
    _write_or_require_identical_json(out_root / "SUMMARY.json", summary)
    _write_or_require_identical_json(out_root / "PASS.json", summary)
    (out_root / "RUNNING.json").unlink(missing_ok=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out_root = Path(args.out_root).resolve()
    try:
        result = _run(args)
    except Exception as error:
        try:
            if not (out_root / "PASS.json").exists() and not (out_root / "NONPASS.json").exists():
                _write_progress(out_root / "last_error.json", {
                    "schema_version": SCHEMA_VERSION, "state": "RETRYABLE_ERROR",
                    "error_type": type(error).__name__, "error": str(error),
                })
        except Exception:
            pass
        print(f"NONPASS: {error}", file=sys.stderr)
        return 1
    print("WROTE SOURCE ROOT LEAF CONTINUATION PASS", _canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
