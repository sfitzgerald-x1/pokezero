"""A3 timing-lattice execution (plan section 4).

End-to-end decision wall is defined by the plan as

    acting-player request available -> validated Showdown choice string ready

so the timer spans observation/legal-action construction, belief-world
construction, native search, root-action mapping, and choice serialization. It
EXCLUDES prefix replay (the corpus record's public prefix is replayed first to
warm the incremental fold) and network delivery, which the offline harness does
not have.

Early stop: after at least ``MIN_DECISIONS_BEFORE_EARLY_STOP`` decisions, a cell
whose running mean reaches the gate is persisted as ``gate_failed`` with its
partial sample and full telemetry. That is a RESULT, not a stage failure — the
plan is explicit that upper cells are allowed to fail the gate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean, median
import time
from typing import Any, Callable, Mapping, Sequence

from .manifest import SearchConfig
from .report import (
    DECISION_WALL_GATE_S,
    MIN_CORPUS_DECISIONS_BEFORE_EARLY_STOP,
    TimingRow,
)
from ..observation import (
    OBSERVATION_SCHEMA_VERSION_V2_2,
    OBSERVATION_SCHEMA_VERSION_V3,
    OBSERVATION_SCHEMA_VERSION_V4,
)
from .resolver import CheckpointContract, ContractError
from .timing_corpus import TimingDecisionRecord
from ..showdown import showdown_choice_for_action


PreparedDecision = Callable[[], dict[str, Any]]
PreparedDecider = Callable[[TimingDecisionRecord, SearchConfig], PreparedDecision]


@dataclass(frozen=True)
class _PublicReplayStep:
    """One ephemeral replay action with only the actor's public observation.

    ``PolicyContext.trajectory`` is normally a ``BattleTrajectory``.  The
    engine's opponent-order reconstruction needs just a structural subset of
    it: both players' action indexes, the acting player's historical public
    observations, and public action identifiers.  Supplying an intentionally
    smaller runtime object prevents a timing adapter from smuggling the
    opponent's private requests into policy context.
    """

    player_id: str
    turn_index: int
    action_index: int
    observation: Any | None = None


@dataclass(frozen=True)
class _PublicReplayTrajectory:
    """Public-only trajectory shape consumed by opponent-order reconstruction."""

    battle_id: str
    format_id: str
    seed: int
    steps: tuple[_PublicReplayStep, ...]
    metadata: Mapping[str, Any]
    terminal: None = None


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def time_lattice_cell(
    config: SearchConfig,
    *,
    records: Sequence[TimingDecisionRecord],
    contract: CheckpointContract,
    showdown_root: str | None = None,
    decide: Callable[[TimingDecisionRecord, SearchConfig], dict[str, Any]] | None = None,
    prepared_decider: PreparedDecider | None = None,
    gate_s: float = DECISION_WALL_GATE_S,
) -> TimingRow:
    """Time one lattice cell over the corpus.

    ``decide`` performs one decision and returns its telemetry; it is injected so
    the timing loop itself is testable without the native crate.
    ``prepared_decider`` separates replay/setup from the timed decision. It must
    return a zero-argument callable whose invocation is the measured span. The
    default replays the corpus prefix and warms the incremental fold before the
    stopwatch starts, then times only engine-MCTS decision work.
    """
    if decide is not None and prepared_decider is not None:
        raise ValueError("pass either decide or prepared_decider, not both.")

    close: Callable[[], None] | None = None
    if prepared_decider is None:
        if decide is None:
            live_decider = _default_decider(contract, showdown_root)
            prepared_decider = live_decider.prepare
            close = live_decider.close
        else:
            prepared_decider = lambda record, config: lambda: decide(record, config)

    walls: list[float] = []
    depths: list[int] = []
    cap_hits = 0
    fallbacks = 0
    prior_fallbacks = 0
    invalid_actions = 0
    total_iterations = model_evals = 0
    encode_s = model_s = tree_s = 0.0
    fold_clone_s = render_s = fold_advance_s = tensor_s = action_map_s = 0.0
    row_input_s = products_s = row_write_s = 0.0
    root_actions: list[str] = []
    gate_failed = False

    try:
        for index, record in enumerate(records, start=1):
            timed_decision = prepared_decider(record, config)
            started = time.perf_counter()
            telemetry = timed_decision()
            walls.append(time.perf_counter() - started)

            realized = int(telemetry.get("max_depth_reached", 0))
            depths.append(realized)
            if realized >= config.depth - 1:
                cap_hits += 1
            fallbacks += int(telemetry.get("fallbacks", 0))
            prior_fallbacks += int(telemetry.get("prior_fallbacks", 0))
            invalid_actions += int(telemetry.get("invalid_actions", 0))
            total_iterations += int(telemetry.get("total_iterations", 0))
            model_evals += int(telemetry.get("model_evals", 0))
            encode_s += float(telemetry.get("encode_s", 0.0))
            model_s += float(telemetry.get("model_s", 0.0))
            tree_s += float(telemetry.get("tree_s", 0.0))
            fold_clone_s += float(telemetry.get("fold_clone_s", 0.0))
            render_s += float(telemetry.get("render_s", 0.0))
            fold_advance_s += float(telemetry.get("fold_advance_s", 0.0))
            tensor_s += float(telemetry.get("tensor_s", 0.0))
            action_map_s += float(telemetry.get("action_map_s", 0.0))
            row_input_s += float(telemetry.get("row_input_s", 0.0))
            products_s += float(telemetry.get("products_s", 0.0))
            row_write_s += float(telemetry.get("row_write_s", 0.0))
            root_actions.append(str(telemetry.get("root_action", "")))

            if index >= MIN_CORPUS_DECISIONS_BEFORE_EARLY_STOP and mean(walls) >= gate_s:
                # A result, not a harness failure: persist the partial sample.
                gate_failed = True
                break
    finally:
        if close is not None:
            close()

    return TimingRow(
        config_id=config.config_id,
        depth=config.depth,
        sims=config.sims,
        decisions_timed=len(walls),
        mean_wall_s=mean(walls) if walls else 0.0,
        median_wall_s=median(walls) if walls else 0.0,
        p95_wall_s=_percentile(walls, 0.95),
        max_wall_s=max(walls) if walls else 0.0,
        realized_depth_mean=mean(depths) if depths else 0.0,
        realized_depth_max=max(depths) if depths else 0,
        cap_hit_rate=(cap_hits / len(depths)) if depths else 0.0,
        total_iterations=total_iterations,
        model_evals=model_evals,
        encode_s=encode_s,
        model_s=model_s,
        tree_s=tree_s,
        fold_clone_s=fold_clone_s,
        render_s=render_s,
        fold_advance_s=fold_advance_s,
        tensor_s=tensor_s,
        action_map_s=action_map_s,
        row_input_s=row_input_s,
        products_s=products_s,
        row_write_s=row_write_s,
        fallbacks=fallbacks,
        prior_fallbacks=prior_fallbacks,
        invalid_actions=invalid_actions,
        gate_failed=gate_failed,
        provenance_exact=True,
        root_argmax_by_decision=tuple(root_actions),
    )


class _LiveEngineTimingDecider:
    """Replay one corpus record into the genuine model-backed engine policy.

    The corpus stores public action identifiers, never stale request-local
    indexes. Each preparation therefore replays those identifiers into a fresh
    deterministic Showdown battle and confirms the resulting request is the
    recorded one. Prefix replay and its incremental-fold warm-up happen before
    the timer; constructing the current observation and action map remains in
    the timed callback, matching the study's ``request available -> choice
    ready`` boundary.
    """

    _FORMAT_ID = "gen3randombattle"
    _STATS_FIELDS = (
        "fallback_decisions",
        "prior_fallbacks",
        "total_iterations",
        "model_evals",
        "encode_wall_seconds",
        "model_wall_seconds",
        "tree_wall_seconds",
        "fold_clone_wall_seconds",
        "render_wall_seconds",
        "fold_advance_wall_seconds",
        "tensor_wall_seconds",
        "action_map_wall_seconds",
        "row_input_wall_seconds",
        "products_wall_seconds",
        "row_write_wall_seconds",
    )

    def __init__(self, contract: CheckpointContract, showdown_root: str | None) -> None:
        from ..collection import env_config_with_policy_spec_masks
        from ..dex import load_showdown_dex_cached
        from ..engine_search import EnvTier2AnnotationSource
        from ..local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from ..randbat import load_gen3_randbat_source_cached

        self._contract = contract
        self._artifacts = materialize_search_artifacts(contract, showdown_root=showdown_root)
        self._env_config = env_config_with_policy_spec_masks(
            LocalShowdownConfig(showdown_root=showdown_root, set_belief_source=True),
            [f"neural:{contract.checkpoint_path}"],
            context="MCTS timing replay",
        )
        self._showdown_root = self._env_config.resolved_showdown_root()
        self._env = LocalShowdownEnv(self._env_config)
        self._dex = load_showdown_dex_cached(self._showdown_root)
        self._set_source = load_gen3_randbat_source_cached(self._showdown_root)
        self._annotation_source = EnvTier2AnnotationSource(self._env)
        self._policies: dict[str, Any] = {}
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True

    def _policy_for(self, config: SearchConfig) -> Any:
        from ..engine_search import EngineMctsConfig, EngineMctsPolicy

        policy = self._policies.get(config.config_id)
        if policy is not None:
            return policy
        if config.inference_mode != "local":
            raise ContractError(
                f"timing config {config.config_id} requests {config.inference_mode!r}; "
                "the current engine-MCTS model path is local-only."
            )
        policy = EngineMctsPolicy(
            dex=self._dex,
            set_source=self._set_source,
            config=EngineMctsConfig(
                worlds=config.worlds,
                search_time_ms=100,
                threads=1,
                leaf_eval="model",
                model_path=self._artifacts["model_path"],
                checkpoint_path=self._contract.checkpoint_path,
                model_device=self._contract.model_device,
                tables_path=self._artifacts["tables_path"],
                search_sims=config.sims,
                search_batch=config.batch,
                search_depth=config.depth,
                model_priors=True,
                early_stop=False,
            ),
            policy_id=f"mcts-timing-{config.config_id}",
            annotation_source=self._annotation_source,
        )
        # The model module's one-time initialization is explicitly outside the
        # decision window. The model's actual leaf forwards remain inside it.
        policy.warm_model_runtime()
        self._policies[config.config_id] = policy
        return policy

    @staticmethod
    def _candidate_payloads(observation: Any) -> tuple[dict[str, Any], ...]:
        candidates = getattr(observation, "metadata", {}).get("action_candidates", ())
        if not isinstance(candidates, Sequence):
            raise ContractError("replayed request has no action_candidates sequence")
        normalized: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ContractError("replayed request has non-mapping action candidate")
            normalized.append(dict(candidate))
        return tuple(normalized)

    @classmethod
    def _snapshot_stats(cls, policy: Any) -> dict[str, Any]:
        stats = policy.stats
        snapshot = {field: getattr(stats, field) for field in cls._STATS_FIELDS}
        snapshot["depth_reached_histogram"] = dict(stats.depth_reached_histogram)
        return snapshot

    @staticmethod
    def _changed_depth(before: Mapping[int, int], after: Mapping[int, int]) -> int:
        changed = [
            int(depth)
            for depth, count in after.items()
            if int(count) > int(before.get(depth, 0))
        ]
        return max(changed, default=0)

    @staticmethod
    def _public_replay_trajectory(
        record: TimingDecisionRecord, replayed: Any
    ) -> _PublicReplayTrajectory:
        """Build the exact public history native opponent priors need.

        The public replay resolves action indexes in its sampled world.  Keep
        them so the existing production permutation walk can decode historical
        opponent switches, and retain *only* the acting player's observation at
        each prior request.  Those observations are the public information the
        production policy itself had at that boundary.
        """

        steps: list[_PublicReplayStep] = []
        replay_actions = getattr(replayed, "replay_actions", None)
        replay_observations = getattr(replayed, "replay_observations", None)
        if not isinstance(replay_actions, Mapping) or not isinstance(replay_observations, Mapping):
            raise ContractError(
                f"{record.decision_id}: replay did not retain action/observation history"
            )
        for action_round in record.public_resolved_action_rounds:
            turn_index = action_round.turn_index
            actions = replay_actions.get(turn_index)
            observations = replay_observations.get(turn_index)
            if not isinstance(actions, Mapping) or set(actions) != set(action_round.actions):
                raise ContractError(
                    f"{record.decision_id}: replay action history differs at turn {turn_index}"
                )
            if not isinstance(observations, Mapping) or set(observations) != set(action_round.actions):
                raise ContractError(
                    f"{record.decision_id}: replay observation history differs at turn {turn_index}"
                )
            for player_id in sorted(str(player) for player in actions):
                action_index = actions[player_id]
                if not isinstance(action_index, int):
                    raise ContractError(
                        f"{record.decision_id}: replay action index is not an integer"
                    )
                observation = observations[player_id] if player_id == record.seat else None
                if player_id == record.seat and observation is None:
                    raise ContractError(
                        f"{record.decision_id}: replay omitted the acting player's prior observation"
                    )
                steps.append(
                    _PublicReplayStep(
                        player_id=player_id,
                        turn_index=turn_index,
                        action_index=action_index,
                        observation=observation,
                    )
                )
        return _PublicReplayTrajectory(
            battle_id=record.battle_id,
            format_id=_LiveEngineTimingDecider._FORMAT_ID,
            seed=record.battle_seed,
            steps=tuple(steps),
            metadata={
                "public_resolved_action_rounds": [
                    action_round.to_dict()
                    for action_round in record.public_resolved_action_rounds
                ]
            },
        )

    def prepare(self, record: TimingDecisionRecord, config: SearchConfig) -> PreparedDecision:
        """Replay + validate the public prefix, returning one timed decision."""
        from ..policy import PolicyContext
        from ..public_replay_materializer import PublicReplayError, replay_public_action_rounds

        if self._closed:
            raise RuntimeError("timing decider is closed")
        policy = self._policy_for(config)
        try:
            replayed = replay_public_action_rounds(
                self._env,
                seed=record.battle_seed,
                format_id=self._FORMAT_ID,
                public_action_rounds=record.public_resolved_action_rounds,
                start_override=None,
            )
        except PublicReplayError as error:
            raise ContractError(
                f"{record.decision_id}: public prefix cannot be materialized: {error.reason}"
            ) from error
        if replayed.terminal is not None:
            raise ContractError(f"{record.decision_id}: public replay reached a terminal state")
        if record.seat not in replayed.requested_players:
            raise ContractError(
                f"{record.decision_id}: replay requested {replayed.requested_players}, "
                f"not acting seat {record.seat}"
            )
        warm_state = self._env.public_materialization_state(record.seat)
        if tuple(warm_state.replay.public_lines) != record.event_prefix:
            raise ContractError(
                f"{record.decision_id}: replayed public prefix differs from corpus witness"
            )
        # Do this before the stopwatch. A subsequent policy decision sees the
        # same fully consumed fold and therefore pays no prefix-fold cost.
        policy.warm_public_prefix_for_replay(
            battle_id=record.battle_id,
            player_id=record.seat,
            decision_round_index=record.turn_index,
            public_materialization_state=warm_state,
        )
        trajectory = self._public_replay_trajectory(record, replayed)

        def timed_decision() -> dict[str, Any]:
            observation = self._env.observe(record.seat)
            legal_mask = tuple(bool(value) for value in observation.legal_action_mask)
            if legal_mask != record.legal_action_mask:
                raise ContractError(
                    f"{record.decision_id}: replayed legal-action mask differs from corpus"
                )
            if self._candidate_payloads(observation) != record.action_candidates:
                raise ContractError(
                    f"{record.decision_id}: replayed action candidates differ from corpus"
                )
            public_state = self._env.public_materialization_state(record.seat)
            if tuple(public_state.replay.public_lines) != record.event_prefix:
                raise ContractError(
                    f"{record.decision_id}: public prefix changed before timed decision"
                )
            context = PolicyContext(
                player_id=record.seat,
                decision_round_index=record.turn_index,
                battle_id=record.battle_id,
                format_id=self._FORMAT_ID,
                seed=record.battle_seed,
                observation=observation,
                requested_players=tuple(replayed.requested_players),
                trajectory=trajectory,  # type: ignore[arg-type]  # public-only runtime history
                requested_legal_action_masks={record.seat: legal_mask},
                requested_observations={record.seat: observation},
                public_materialization_state=public_state,
            )
            before = self._snapshot_stats(policy)
            decision = policy.select_action_with_context(
                context, rng=random.Random(record.bot_rng_seed)
            )
            after = self._snapshot_stats(policy)
            action_index = int(decision.action_index)
            try:
                choice = showdown_choice_for_action(
                    self._env._state_for_player(record.seat), action_index
                )
            except ValueError as error:
                raise ContractError(
                    f"{record.decision_id}: selected action cannot be serialized to Showdown"
                ) from error
            return {
                "root_action": choice,
                "max_depth_reached": self._changed_depth(
                    before["depth_reached_histogram"], after["depth_reached_histogram"]
                ),
                "fallbacks": int(after["fallback_decisions"] - before["fallback_decisions"]),
                "prior_fallbacks": int(after["prior_fallbacks"] - before["prior_fallbacks"]),
                "invalid_actions": int(
                    action_index < 0
                    or action_index >= len(legal_mask)
                    or not legal_mask[action_index]
                ),
                "total_iterations": int(after["total_iterations"] - before["total_iterations"]),
                "model_evals": int(after["model_evals"] - before["model_evals"]),
                "encode_s": float(after["encode_wall_seconds"] - before["encode_wall_seconds"]),
                "model_s": float(after["model_wall_seconds"] - before["model_wall_seconds"]),
                "tree_s": float(after["tree_wall_seconds"] - before["tree_wall_seconds"]),
                "fold_clone_s": float(after["fold_clone_wall_seconds"] - before["fold_clone_wall_seconds"]),
                "render_s": float(after["render_wall_seconds"] - before["render_wall_seconds"]),
                "fold_advance_s": float(after["fold_advance_wall_seconds"] - before["fold_advance_wall_seconds"]),
                "tensor_s": float(after["tensor_wall_seconds"] - before["tensor_wall_seconds"]),
                "action_map_s": float(after["action_map_wall_seconds"] - before["action_map_wall_seconds"]),
                "row_input_s": float(after["row_input_wall_seconds"] - before["row_input_wall_seconds"]),
                "products_s": float(after["products_wall_seconds"] - before["products_wall_seconds"]),
                "row_write_s": float(after["row_write_wall_seconds"] - before["row_write_wall_seconds"]),
            }

        return timed_decision


def _default_decider(
    contract: CheckpointContract, showdown_root: str | None
) -> _LiveEngineTimingDecider:
    """Build the real replay-backed engine-MCTS timing adapter."""
    return _LiveEngineTimingDecider(contract, showdown_root)


def materialize_search_artifacts(
    contract: CheckpointContract, *, showdown_root: str | None
) -> dict[str, str]:
    """Export (once) the TorchScript trace + encoder tables this contract needs.

    Reuse is keyed by the contract's export key, so a different checkpoint,
    device, observation contract, Showdown source, or exporter revision cannot
    silently adopt someone else's artifact.
    """
    import subprocess
    import sys
    from pathlib import Path as _Path

    from .resolver import export_reuse_key, validate_encoder_tables

    key = export_reuse_key(contract)[:16]
    root = _Path(contract.checkpoint_path).parent / f".mcts-eval-artifacts-{key}"
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "model_ts.pt"
    tables_path = root / "encoder_tables.json"
    repo = _Path(__file__).resolve().parents[3]

    # Map the checkpoint's schema to the exporter's CLI choice, BEFORE the TorchScript export
    # below: an unmapped schema is a terminal contract error either way, and resolving it first
    # means it fails in milliseconds instead of after burning a full model trace.
    # A bare "everything else is v2.2" default silently exported the wrong tables for any newer
    # schema — v4 got past the resolver's schema gate and then died in validate_encoder_tables
    # on a v2.2-vs-v4 mismatch, making the gate a dead end. Raising closes the CLASS, not the
    # instance: otherwise the only thing standing between the next schema and that same dead
    # end is SUPPORTED_OBSERVATION_SCHEMAS upstream, an implicit coupling one edit away from
    # being wrong again.
    _EXPORTER_SCHEMA_CHOICES = {
        OBSERVATION_SCHEMA_VERSION_V4: "v4",
        OBSERVATION_SCHEMA_VERSION_V3: "v3",
        OBSERVATION_SCHEMA_VERSION_V2_2: "v2.2",
    }
    try:
        schema = _EXPORTER_SCHEMA_CHOICES[contract.schema_version]
    except KeyError:
        raise ContractError(
            f"no encoder-table exporter choice for observation schema "
            f"{contract.schema_version!r} (known: "
            f"{', '.join(sorted(_EXPORTER_SCHEMA_CHOICES))}). Add the mapping when "
            f"adding the schema; exporting v2.2 tables by default silently produces "
            f"the wrong string->row map."
        ) from None

    if not model_path.is_file():
        subprocess.run(
            # TorchScript traces bake device constants (see model.rs: "Artifacts
            # are PER-DEVICE"), so a CPU trace loaded on CUDA dies inside the
            # interpreter. Trace on the device the search will run on; the export
            # reuse key already includes model_device, so the cache stays correct.
            [sys.executable, str(repo / "scripts" / "export_model.py"),
             "--checkpoint", contract.checkpoint_path, "--out-dir", str(root),
             "--formats", "ts", "--device", contract.model_device],
            check=True,
        )
        produced = next(root.glob("*_ts.pt"), None) or next(root.glob("*.pt"), None)
        if produced and produced != model_path:
            produced.rename(model_path)
    if not tables_path.is_file():
        subprocess.run(
            # Always derive the layout from the checkpoint, never from the schema
            # default: a region-trimmed model has a narrower transition region, so
            # schema-default tables describe an observation it cannot consume.
            [sys.executable, str(repo / "scripts" / "export_encoder_tables.py"),
             "--showdown-root", showdown_root or "", "--observation-schema", schema,
             "--checkpoint", contract.checkpoint_path,
             "--out", str(tables_path)],
            check=True,
        )
    # Fail closed on root/leaf drift, against the checkpoint's own contract. A
    # REUSED artifact is covered because the contract's vocabulary is in the reuse
    # key, so tables exported against a different enumeration resolve to a
    # different path and can never be adopted here.
    validate_encoder_tables(contract, tables_path)
    return {"model_path": str(model_path), "tables_path": str(tables_path)}
