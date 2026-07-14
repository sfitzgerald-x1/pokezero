"""Pure prior and public-belief uncertainty profiling for adaptive root budgets.

The metric layer in this module intentionally has no dependency on a model, a
simulator, or FoulPlay. Callers inject untempered priors and initial-sweep
candidate values. That makes the reported gate surface reproducible from the
public corpus and testable without a privileged battle state.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from .actions import ACTION_COUNT, MOVE_ACTION_COUNT
from .determinization import BeliefWorldSamplingProfile, belief_world_sampling_profile
from .policy import PolicyContext
from .public_decision_corpus import (
    PUBLIC_DECISION_CORPUS_SCHEMA_SHA256,
    PublicDecisionCorpus,
    PublicDecisionCorpusStream,
    PublicDecisionRecord,
    canonical_json_sha256,
)
from .public_replay_materializer import public_event_prefix_summary


PRIOR_BELIEF_PROFILE_SCHEMA_VERSION = "pokezero.prior-belief-profile.v1"
MINIMUM_PROFILE_DECISIONS = 2_000
EARLY_PHASE_MAX_TURN = 5
MID_PHASE_MAX_TURN = 15

PriorEvaluator = Callable[[tuple[Any, ...]], Sequence[float]]
CandidateValueEvaluator = Callable[
    [PublicDecisionRecord], Sequence["WorldScenarioEvaluation"] | "CandidateValueEvaluation"
]
ProfileProgressCallback = Callable[[int, PublicDecisionRecord], None]


def _records_with_progress(
    records: Iterable[PublicDecisionRecord],
    progress_callback: ProfileProgressCallback,
) -> Iterable[PublicDecisionRecord]:
    """Report a record only after its profile attempt has completed."""

    for completed_count, record in enumerate(records, start=1):
        yield record
        progress_callback(completed_count, record)


@dataclass(frozen=True)
class WorldScenarioEvaluation:
    """The mandatory initial candidate sweep for one public world/scenario.

    ``candidate_values`` must be values before any PUCT revisits. The profiler
    stores this mapping verbatim in the selection-context row so an adaptive
    visit decision can be reconstructed later without the simulator.
    """

    world_index: int
    scenario_index: int
    scenario_label: str
    scenario_weight: float
    candidate_values: Mapping[int, float]
    public_event_canonicalizations: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.world_index < 0 or self.scenario_index < 0:
            raise ValueError("world_index and scenario_index must be non-negative.")
        if not self.scenario_label:
            raise ValueError("scenario_label must be non-empty.")
        if not math.isfinite(self.scenario_weight) or self.scenario_weight <= 0.0:
            raise ValueError("scenario_weight must be finite and positive.")
        normalized = {int(action): float(value) for action, value in self.candidate_values.items()}
        if not normalized:
            raise ValueError("initial candidate sweep must contain at least one candidate value.")
        for action, value in normalized.items():
            if not 0 <= action < ACTION_COUNT:
                raise ValueError(f"candidate action index must be between 0 and {ACTION_COUNT - 1}.")
            if not math.isfinite(value):
                raise ValueError("candidate values must be finite.")
        object.__setattr__(self, "candidate_values", normalized)
        object.__setattr__(
            self,
            "public_event_canonicalizations",
            tuple(dict(value) for value in self.public_event_canonicalizations),
        )


@dataclass(frozen=True)
class CandidateValueEvaluation:
    """Evaluator outcome with a public reason when no replay context is available."""

    contexts: tuple[WorldScenarioEvaluation, ...]
    skip_reason: str | None = None
    failure_reasons: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if self.contexts and self.skip_reason is not None:
            raise ValueError("candidate evaluation cannot have both contexts and a skip reason.")
        if not self.contexts and not self.skip_reason:
            raise ValueError("empty candidate evaluation requires a skip reason.")
        normalized = {str(reason): int(count) for reason, count in (self.failure_reasons or {}).items()}
        if any(count <= 0 for count in normalized.values()):
            raise ValueError("candidate evaluation failure counts must be positive.")
        object.__setattr__(self, "failure_reasons", normalized)


@dataclass(frozen=True)
class PriorBeliefProfileConfig:
    """Explicit gate thresholds and phase boundaries recorded with each profile."""

    entropy_thresholds: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    margin_thresholds: tuple[float, ...] = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4)
    world_sample_cap: int = 4
    early_phase_max_turn: int = EARLY_PHASE_MAX_TURN
    mid_phase_max_turn: int = MID_PHASE_MAX_TURN
    opponent_legal_mask_mode: str = "hidden"
    root_noise_enabled: bool = False

    def __post_init__(self) -> None:
        if self.world_sample_cap <= 0:
            raise ValueError("world_sample_cap must be positive.")
        if self.early_phase_max_turn < 0 or self.mid_phase_max_turn < self.early_phase_max_turn:
            raise ValueError("phase boundaries must satisfy 0 <= early <= mid.")
        if self.opponent_legal_mask_mode != "hidden":
            raise ValueError("prior/belief profiling refuses privileged opponent legal-mask mode.")
        if self.root_noise_enabled:
            raise ValueError("prior/belief profiling requires root noise to be off.")
        for name, values in (("entropy_thresholds", self.entropy_thresholds), ("margin_thresholds", self.margin_thresholds)):
            if not values:
                raise ValueError(f"{name} must be non-empty.")
            if any(value < 0.0 or not math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain finite non-negative thresholds.")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "entropy_metric": "raw-shannon-nats-from-untempered-legal-priors",
            "normalized_entropy_metric": "raw-shannon-nats/log(legal-action-count)",
            "root_noise": {"enabled": False, "root_dirichlet_alpha": None},
        }


@dataclass(frozen=True)
class _PublicReplayStep:
    player_id: str
    turn_index: int
    action_index: int | None
    observation: Any | None = None


@dataclass(frozen=True)
class _PublicReplayTrajectory:
    steps: tuple[_PublicReplayStep, ...]
    metadata: Mapping[str, Any]


def phase_for_turn(
    turn_index: int,
    *,
    early_phase_max_turn: int = EARLY_PHASE_MAX_TURN,
    mid_phase_max_turn: int = MID_PHASE_MAX_TURN,
) -> str:
    """Return the stable phase bucket used by Step 2 reports.

    Turns 0..5 are early, 6..15 are mid, and later turns are late by
    default. The boundaries are profile configuration, not implicit policy
    behavior, and are emitted in every report.
    """

    if turn_index < 0:
        raise ValueError("turn_index must be non-negative.")
    if early_phase_max_turn < 0 or mid_phase_max_turn < early_phase_max_turn:
        raise ValueError("phase boundaries must satisfy 0 <= early <= mid.")
    if turn_index <= early_phase_max_turn:
        return "early"
    if turn_index <= mid_phase_max_turn:
        return "mid"
    return "late"


def public_policy_context(record: PublicDecisionRecord) -> PolicyContext:
    """Build the minimal trajectory shape used by public belief determinization.

    The public belief sampler receives only actor observations at their source
    decision rounds. Resolved public action identifiers are intentionally not
    converted to request-local indexes here; sampled-world replay resolves them
    only after materializing that world.
    """

    own_history = tuple(
        _PublicReplayStep(
            player_id=record.acting_player,
            turn_index=entry.turn_index,
            action_index=None,
            observation=entry.observation.to_observation(belief_view=record.public_belief_view),
        )
        for entry in record.history
    )
    trajectory = _PublicReplayTrajectory(
        steps=own_history,
        # Determinization reads this through the normal trajectory adapter.
        # It is the corpus's public-only representation, never request-local IDs.
        metadata={
            "public_resolved_action_rounds": [
                action_round.to_dict() for action_round in record.public_resolved_action_rounds
            ]
        },
    )
    return PolicyContext(
        player_id=record.acting_player,
        decision_round_index=record.turn_index,
        battle_id=record.battle_id,
        format_id=record.format_id,
        seed=record.seed,
        observation=record.observations()[-1],
        requested_players=(record.acting_player,),
        trajectory=trajectory,  # type: ignore[arg-type]
        requested_legal_action_masks={record.acting_player: record.current_legal_action_mask},
        requested_observations={record.acting_player: record.observations()[-1]},
    )


def public_belief_sampling_profile(
    record: PublicDecisionRecord,
    *,
    sample_cap: int,
    set_source: Any | None = None,
) -> BeliefWorldSamplingProfile:
    """Resolve dynamic ``K`` from the record's public belief view only."""

    profile = belief_world_sampling_profile(
        public_policy_context(record),
        sample_cap=sample_cap,
        set_source=set_source,
    )
    if profile is None:
        raise ValueError(f"public decision {record.decision_id} is missing a valid public belief profile.")
    return profile


def profile_public_decisions(
    records: Iterable[PublicDecisionRecord],
    *,
    prior_evaluator: PriorEvaluator,
    candidate_value_evaluator: CandidateValueEvaluator,
    config: PriorBeliefProfileConfig = PriorBeliefProfileConfig(),
    belief_set_source: Any | None = None,
    provenance: Mapping[str, Any] | None = None,
    provenance_factory: Callable[[], Mapping[str, Any]] | None = None,
    profile_scope: Mapping[str, Any] | None = None,
    progress_callback: ProfileProgressCallback | None = None,
) -> dict[str, Any]:
    """Profile public decisions with raw priors and initial candidate values.

    No policy temperature is accepted here. ``prior_evaluator`` must supply the
    checkpoint's raw (temperature 1.0), root-noise-free prior vector. Candidate
    values are likewise injected from a public-prefix hidden-mode evaluator.
    """

    if provenance is not None and provenance_factory is not None:
        raise ValueError("provide either provenance or provenance_factory, not both.")
    if progress_callback is not None:
        records = _records_with_progress(records, progress_callback)
    decision_rows: list[dict[str, Any]] = []
    selection_context_rows: list[dict[str, Any]] = []
    skipped_decision_rows: list[dict[str, Any]] = []
    corpus_decision_count = 0
    corpus_by_phase: Counter[str] = Counter()
    event_bearing_by_phase: Counter[str] = Counter()
    public_events_by_phase: Counter[str] = Counter()
    unsupported_prefixes_by_phase: Counter[str] = Counter()
    unsupported_events_by_phase: Counter[str] = Counter()
    for record in records:
        phase = phase_for_turn(
            record.turn_index,
            early_phase_max_turn=config.early_phase_max_turn,
            mid_phase_max_turn=config.mid_phase_max_turn,
        )
        event_summary = public_event_prefix_summary(record.public_resolved_action_rounds)
        corpus_decision_count += 1
        corpus_by_phase[phase] += 1
        event_bearing_by_phase[phase] += int(event_summary["public_event_count"] > 0)
        public_events_by_phase[phase] += int(event_summary["public_event_count"])
        unsupported_prefixes_by_phase[phase] += int(event_summary["unsupported_public_event_count"] > 0)
        unsupported_events_by_phase[phase] += int(event_summary["unsupported_public_event_count"])
        legal_actions = tuple(index for index, legal in enumerate(record.current_legal_action_mask) if legal)
        if not legal_actions:
            skipped_decision_rows.append(
                _skipped_decision_row(record, phase=phase, reason="no_legal_acting_player_actions", event_summary=event_summary)
            )
            continue
        try:
            priors = _normalized_raw_legal_priors(prior_evaluator(record.observations()), legal_actions)
        except (TypeError, ValueError):
            skipped_decision_rows.append(
                _skipped_decision_row(record, phase=phase, reason="prior_evaluator_contract_failure", event_summary=event_summary)
            )
            continue
        raw_entropy = _shannon_entropy(priors[index] for index in legal_actions)
        normalized_entropy = raw_entropy / math.log(len(legal_actions)) if len(legal_actions) > 1 else 0.0
        ranked_priors = sorted(((index, priors[index]) for index in legal_actions), key=lambda item: (-item[1], item[0]))
        top1_action, top1_prior = ranked_priors[0]
        top2_action, top2_prior = ranked_priors[1] if len(ranked_priors) > 1 else (None, 0.0)
        try:
            belief = public_belief_sampling_profile(
                record,
                sample_cap=config.world_sample_cap,
                set_source=belief_set_source,
            )
        except ValueError:
            skipped_decision_rows.append(
                _skipped_decision_row(record, phase=phase, reason="invalid_public_belief", event_summary=event_summary)
            )
            continue
        try:
            evaluated = candidate_value_evaluator(record)
            outcome = _candidate_evaluation_outcome(evaluated)
        except (TypeError, ValueError):
            skipped_decision_rows.append(
                _skipped_decision_row(
                    record,
                    phase=phase,
                    reason="candidate_evaluator_contract_failure",
                    event_summary=event_summary,
                )
            )
            continue
        contexts = outcome.contexts
        if not contexts:
            skipped_decision_rows.append(
                _skipped_decision_row(
                    record,
                    phase=phase,
                    reason=outcome.skip_reason or "no_public_replay_contexts",
                    event_summary=event_summary,
                    failure_reasons=outcome.failure_reasons,
                )
            )
            continue
        margins: list[tuple[float, float]] = []
        prepared_context_rows: list[dict[str, Any]] = []
        try:
            for context in contexts:
                _validate_context_legal_candidates(record, context)
                margin, value_top1, value_top2 = initial_candidate_value_top_two_margin(context.candidate_values)
                if margin is not None:
                    margins.append((context.scenario_weight, margin))
                margin_available = margin is not None
                prepared_context_rows.append(
                {
                    "decision_id": record.decision_id,
                    "phase": phase,
                    "turn_index": record.turn_index,
                    "world_index": context.world_index,
                    "scenario_index": context.scenario_index,
                    "scenario_label": context.scenario_label,
                    "scenario_weight": context.scenario_weight,
                    "resolved_dynamic_k": belief.sample_count,
                    "belief_public_checksum": belief.public_checksum,
                    "legal_action_indices": list(legal_actions),
                    "raw_policy_entropy": raw_entropy,
                    "normalized_policy_entropy": normalized_entropy,
                    "initial_candidate_values": [
                        {"action_index": action, "value": value}
                        for action, value in sorted(context.candidate_values.items())
                    ],
                    "initial_candidate_value_top1": value_top1,
                    "initial_candidate_value_top2": value_top2,
                    "initial_candidate_value_top_two_margin": margin,
                    "candidate_margin_available": margin_available,
                    "candidate_value_context_kind": "competitive" if margin_available else "forced",
                    "public_event_canonicalizations": [
                        dict(value) for value in context.public_event_canonicalizations
                    ],
                    "public_event_canonicalization_count": len(context.public_event_canonicalizations),
                    "selection_gate_inputs": {
                        "policy_entropy": raw_entropy,
                        "value_margin": margin,
                        "action_priors": [priors[index] for index in legal_actions],
                        "root_noise_enabled": False,
                        "opponent_legal_mask_mode": "hidden",
                    },
                }
                )
        except (TypeError, ValueError):
            skipped_decision_rows.append(
                _skipped_decision_row(
                    record,
                    phase=phase,
                    reason="candidate_legality_or_contract_failure",
                    event_summary=event_summary,
                )
            )
            continue
        selection_context_rows.extend(prepared_context_rows)
        weighted_margin = (
            sum(weight * margin for weight, margin in margins) / sum(weight for weight, _ in margins)
            if margins
            else None
        )
        decision_rows.append(
            {
                "decision_id": record.decision_id,
                "battle_id": record.battle_id,
                "seed": record.seed,
                "format_id": record.format_id,
                "acting_player": record.acting_player,
                "turn_index": record.turn_index,
                "phase": phase,
                "legal_action_count": len(legal_actions),
                "legal_action_indices": list(legal_actions),
                "raw_policy_entropy": raw_entropy,
                "normalized_policy_entropy": normalized_entropy,
                "top1_action_index": top1_action,
                "top1_prior": top1_prior,
                "top2_action_index": top2_action,
                "top2_prior": top2_prior,
                "top2_prior_mass": top1_prior + top2_prior,
                "initial_candidate_value_top_two_margin": weighted_margin,
                "candidate_margin_available": weighted_margin is not None,
                "belief_combination_count": belief.combination_count,
                "belief_uncertainty_bits": belief.uncertainty_bits,
                "belief_uncertain_slot_count": belief.uncertain_slot_count,
                "resolved_dynamic_k": belief.sample_count,
                "belief_public_checksum": belief.public_checksum,
                "selection_context_count": len(contexts),
                **event_summary,
            }
        )
    if corpus_decision_count == 0:
        raise ValueError("prior/belief profile requires at least one public decision.")
    profile_config = config.to_dict()
    provenance_payload = dict(provenance_factory() if provenance_factory is not None else provenance or {})
    report_core = {
        "schema_version": PRIOR_BELIEF_PROFILE_SCHEMA_VERSION,
        "profile_config": profile_config,
        "profile_config_sha256": canonical_json_sha256(profile_config),
        "checkpoint_sha256": provenance_payload.get("checkpoint_sha256"),
        "corpus_sha256": provenance_payload.get("corpus_source_sha256"),
        "selected_content_sha256": provenance_payload.get("selected_content_sha256"),
        "corpus_selection": provenance_payload.get("corpus_selection"),
        "public_corpus_schema_sha256": PUBLIC_DECISION_CORPUS_SCHEMA_SHA256,
        "root_noise": {"enabled": False, "root_dirichlet_alpha": None},
        "opponent_legal_mask_mode": "hidden",
        "corpus_decision_count": corpus_decision_count,
        "decision_count": len(decision_rows),
        "skipped_decision_count": len(skipped_decision_rows),
        "skipped_decision_rows": skipped_decision_rows,
        "selection_context_count": len(selection_context_rows),
        "decision_rows": decision_rows,
        "selection_context_rows": selection_context_rows,
        "threshold_sweeps": _threshold_sweeps(selection_context_rows, config=config),
        "decision_normalized_threshold_sweeps": _threshold_sweeps(
            decision_rows,
            config=config,
            unit="decision",
        ),
        "representativeness": _representativeness_summary(
            corpus_by_phase=corpus_by_phase,
            event_bearing_by_phase=event_bearing_by_phase,
            public_events_by_phase=public_events_by_phase,
            unsupported_prefixes_by_phase=unsupported_prefixes_by_phase,
            unsupported_events_by_phase=unsupported_events_by_phase,
            decision_rows=decision_rows,
            skipped_decision_rows=skipped_decision_rows,
        ),
        "provenance": provenance_payload,
    }
    if profile_scope is not None:
        report_core["profile_scope"] = dict(profile_scope)
    return {**report_core, "profile_sha256": canonical_json_sha256(report_core)}


def profile_public_corpus(
    corpus: PublicDecisionCorpus | PublicDecisionCorpusStream,
    *,
    prior_evaluator: PriorEvaluator,
    candidate_value_evaluator: CandidateValueEvaluator,
    config: PriorBeliefProfileConfig = PriorBeliefProfileConfig(),
    belief_set_source: Any | None = None,
    provenance: Mapping[str, Any] | None = None,
    minimum_decisions: int = MINIMUM_PROFILE_DECISIONS,
    progress_callback: ProfileProgressCallback | None = None,
) -> dict[str, Any]:
    """Profile a corpus only when it satisfies the Step 2 2,000-decision floor."""

    if minimum_decisions < MINIMUM_PROFILE_DECISIONS:
        raise ValueError(f"minimum_decisions may not be lower than {MINIMUM_PROFILE_DECISIONS}.")
    return _profile_public_corpus(
        corpus,
        prior_evaluator=prior_evaluator,
        candidate_value_evaluator=candidate_value_evaluator,
        config=config,
        belief_set_source=belief_set_source,
        provenance=provenance,
        minimum_decisions=minimum_decisions,
        profile_scope=None,
        progress_callback=progress_callback,
    )


def profile_public_corpus_shard(
    corpus: PublicDecisionCorpusStream,
    *,
    prior_evaluator: PriorEvaluator,
    candidate_value_evaluator: CandidateValueEvaluator,
    config: PriorBeliefProfileConfig = PriorBeliefProfileConfig(),
    belief_set_source: Any | None = None,
    provenance: Mapping[str, Any] | None = None,
    progress_callback: ProfileProgressCallback | None = None,
    source_start_decision: int | None = None,
    source_corpus_sha256: str | None = None,
) -> dict[str, Any]:
    """Profile one deterministic corpus range without treating it as capstone-ready.

    A replay-from-root value sweep is intentionally expensive. This helper is a
    map-stage primitive: each shard records its own bounded selection and may
    contain fewer than 2,000 valid replay contexts. Only
    :func:`merge_public_corpus_profile_shards` can re-establish the Step 2
    floor for capstone use.
    """

    if corpus.selected_decision_limit is None:
        raise ValueError("profile shards require a bounded corpus range.")
    if source_start_decision is not None and source_start_decision < 0:
        raise ValueError("source_start_decision must be non-negative when provided.")
    if source_corpus_sha256 is not None and (
        len(source_corpus_sha256) != 64 or any(character not in "0123456789abcdef" for character in source_corpus_sha256)
    ):
        raise ValueError("source_corpus_sha256 must be a lowercase SHA-256 digest when provided.")
    requested_count = corpus.selected_decision_limit
    scope = {
        "kind": "shard",
        # A locally materialized shard starts at zero in its snapshot file, but
        # retains its original position for merge-contiguity validation.
        "start_decision": (
            corpus.selected_decision_start if source_start_decision is None else source_start_decision
        ),
        "requested_decision_count": requested_count,
    }
    return _profile_public_corpus(
        corpus,
        prior_evaluator=prior_evaluator,
        candidate_value_evaluator=candidate_value_evaluator,
        config=config,
        belief_set_source=belief_set_source,
        provenance={
            **dict(provenance or {}),
            **({"source_corpus_sha256": source_corpus_sha256} if source_corpus_sha256 is not None else {}),
        },
        minimum_decisions=None,
        profile_scope=scope,
        progress_callback=progress_callback,
    )


def _profile_public_corpus(
    corpus: PublicDecisionCorpus | PublicDecisionCorpusStream,
    *,
    prior_evaluator: PriorEvaluator,
    candidate_value_evaluator: CandidateValueEvaluator,
    config: PriorBeliefProfileConfig,
    belief_set_source: Any | None,
    provenance: Mapping[str, Any] | None,
    minimum_decisions: int | None,
    profile_scope: Mapping[str, Any] | None,
    progress_callback: ProfileProgressCallback | None,
) -> dict[str, Any]:
    base_provenance = dict(provenance or {})
    if isinstance(corpus, PublicDecisionCorpusStream):
        records = corpus.iter_decisions()

        def streaming_provenance() -> Mapping[str, Any]:
            selected_count = corpus.selected_decision_count
            if minimum_decisions is not None and selected_count < minimum_decisions:
                raise ValueError(
                    f"prior/belief profiling requires at least {minimum_decisions} valid p1 decisions; "
                    f"corpus contains {selected_count}."
                )
            return {
                **base_provenance,
                "corpus_source_sha256": base_provenance.get("source_corpus_sha256", corpus.source_file_sha256),
                "selected_content_sha256": corpus.selected_content_sha256,
                "corpus_selection": {
                    "max_decisions": corpus.selected_decision_limit,
                    "selected_decision_count": selected_count,
                },
                "corpus_manifest": dict(corpus.manifest),
            }

        provenance_factory = streaming_provenance
        corpus_count = lambda: corpus.selected_decision_count
    else:
        if minimum_decisions is not None and len(corpus.decisions) < minimum_decisions:
            raise ValueError(
                f"prior/belief profiling requires at least {minimum_decisions} valid p1 decisions; "
                f"corpus contains {len(corpus.decisions)}."
            )
        records = corpus.decisions
        provenance_factory = lambda: {
            **base_provenance,
            "corpus_source_sha256": base_provenance.get("source_corpus_sha256", corpus.source_file_sha256),
            "selected_content_sha256": corpus.selected_content_sha256,
            "corpus_selection": {
                "max_decisions": corpus.selected_decision_limit,
                "selected_decision_count": len(corpus.decisions),
            },
            "corpus_manifest": dict(corpus.manifest),
        }
        corpus_count = lambda: len(corpus.decisions)
    report = profile_public_decisions(
        records,
        prior_evaluator=prior_evaluator,
        candidate_value_evaluator=candidate_value_evaluator,
        config=config,
        belief_set_source=belief_set_source,
        provenance_factory=provenance_factory,
        profile_scope=profile_scope,
        progress_callback=progress_callback,
    )
    if minimum_decisions is not None and report["decision_count"] < minimum_decisions:
        raise ValueError(
            f"prior/belief profiling requires at least {minimum_decisions} successfully profiled p1 decisions; "
            f"corpus contains {corpus_count()} rows but only {report['decision_count']} produced public replay contexts."
        )
    return report


def merge_public_corpus_profile_shards(
    shards: Sequence[Mapping[str, Any]],
    *,
    minimum_decisions: int = MINIMUM_PROFILE_DECISIONS,
) -> dict[str, Any]:
    """Merge contiguous profile-map outputs into one capstone-eligible report.

    The merge refuses gaps, overlap, incompatible model/configuration inputs,
    or tampered shard hashes. It recomputes aggregate sweeps and phase coverage
    from the persisted per-decision rows instead of averaging shard summaries.
    """

    if minimum_decisions < MINIMUM_PROFILE_DECISIONS:
        raise ValueError(f"minimum_decisions may not be lower than {MINIMUM_PROFILE_DECISIONS}.")
    if not shards:
        raise ValueError("at least one profile shard is required.")

    first = _validated_profile_shard(shards[0])
    profile_config = _profile_mapping(first, "profile_config")
    provenance = _profile_mapping(first, "provenance")
    expected_start = 0
    decision_rows: list[dict[str, Any]] = []
    selection_context_rows: list[dict[str, Any]] = []
    skipped_decision_rows: list[dict[str, Any]] = []
    corpus_by_phase: Counter[str] = Counter()
    event_bearing_by_phase: Counter[str] = Counter()
    public_events_by_phase: Counter[str] = Counter()
    unsupported_prefixes_by_phase: Counter[str] = Counter()
    unsupported_events_by_phase: Counter[str] = Counter()
    shard_digests: list[str] = []
    decision_ids: set[str] = set()

    for index, raw_shard in enumerate(shards):
        shard = _validated_profile_shard(raw_shard)
        if shard.get("profile_config") != profile_config:
            raise ValueError(f"profile shard {index} configuration does not match shard 0.")
        if _profile_input_identity(shard) != _profile_input_identity(first):
            raise ValueError(f"profile shard {index} checkpoint or public-corpus provenance does not match shard 0.")
        scope = _profile_mapping(shard, "profile_scope")
        start = _profile_nonnegative_int(scope.get("start_decision"), field=f"profile shard {index} start_decision")
        requested = _profile_positive_int(
            scope.get("requested_decision_count"), field=f"profile shard {index} requested_decision_count"
        )
        selection = _profile_mapping(shard, "corpus_selection")
        selected = _profile_nonnegative_int(
            selection.get("selected_decision_count"), field=f"profile shard {index} selected_decision_count"
        )
        if start != expected_start:
            raise ValueError(f"profile shard {index} is not contiguous with the preceding range.")
        if selected != requested:
            raise ValueError(f"profile shard {index} did not complete its requested corpus range.")
        expected_start += selected
        selection_digest = shard.get("selected_content_sha256")
        if not isinstance(selection_digest, str) or len(selection_digest) != 64:
            raise ValueError(f"profile shard {index} is missing its selected-content digest.")
        shard_digests.append(selection_digest)
        shard_decision_rows = [dict(row) for row in _profile_sequence(shard, "decision_rows")]
        shard_skipped_rows = [dict(row) for row in _profile_sequence(shard, "skipped_decision_rows")]
        for row in (*shard_decision_rows, *shard_skipped_rows):
            decision_id = row.get("decision_id")
            if not isinstance(decision_id, str) or not decision_id:
                raise ValueError(f"profile shard {index} contains a row without a decision_id.")
            if decision_id in decision_ids:
                raise ValueError(f"profile shard {index} duplicates a decision covered by an earlier shard.")
            decision_ids.add(decision_id)
        decision_rows.extend(shard_decision_rows)
        selection_context_rows.extend(dict(row) for row in _profile_sequence(shard, "selection_context_rows"))
        skipped_decision_rows.extend(shard_skipped_rows)
        _merge_representativeness_counters(
            shard,
            corpus_by_phase=corpus_by_phase,
            event_bearing_by_phase=event_bearing_by_phase,
            public_events_by_phase=public_events_by_phase,
            unsupported_prefixes_by_phase=unsupported_prefixes_by_phase,
            unsupported_events_by_phase=unsupported_events_by_phase,
        )

    if len(decision_rows) < minimum_decisions:
        raise ValueError(
            f"merged profile requires at least {minimum_decisions} successfully profiled p1 decisions; "
            f"shards produced {len(decision_rows)}."
        )
    merged_selection = {
        "max_decisions": expected_start,
        "selected_decision_count": expected_start,
        "selection_kind": "contiguous_shards",
        "shard_count": len(shards),
    }
    merged_provenance = {
        **provenance,
        "corpus_selection": merged_selection,
        "shard_profile_sha256": [str(shard["profile_sha256"]) for shard in shards],
    }
    report_core = {
        "schema_version": PRIOR_BELIEF_PROFILE_SCHEMA_VERSION,
        "profile_config": dict(profile_config),
        "profile_config_sha256": canonical_json_sha256(profile_config),
        "checkpoint_sha256": first.get("checkpoint_sha256"),
        "corpus_sha256": first.get("corpus_sha256"),
        "selected_content_sha256": canonical_json_sha256(shard_digests),
        "corpus_selection": merged_selection,
        "public_corpus_schema_sha256": PUBLIC_DECISION_CORPUS_SCHEMA_SHA256,
        "root_noise": {"enabled": False, "root_dirichlet_alpha": None},
        "opponent_legal_mask_mode": "hidden",
        "corpus_decision_count": expected_start,
        "decision_count": len(decision_rows),
        "skipped_decision_count": len(skipped_decision_rows),
        "skipped_decision_rows": skipped_decision_rows,
        "selection_context_count": len(selection_context_rows),
        "decision_rows": decision_rows,
        "selection_context_rows": selection_context_rows,
        "threshold_sweeps": _threshold_sweeps(selection_context_rows, config=_profile_config_from_payload(profile_config)),
        "decision_normalized_threshold_sweeps": _threshold_sweeps(
            decision_rows,
            config=_profile_config_from_payload(profile_config),
            unit="decision",
        ),
        "representativeness": _representativeness_summary(
            corpus_by_phase=corpus_by_phase,
            event_bearing_by_phase=event_bearing_by_phase,
            public_events_by_phase=public_events_by_phase,
            unsupported_prefixes_by_phase=unsupported_prefixes_by_phase,
            unsupported_events_by_phase=unsupported_events_by_phase,
            decision_rows=decision_rows,
            skipped_decision_rows=skipped_decision_rows,
        ),
        "provenance": merged_provenance,
        "profile_scope": {"kind": "merged-shards", "shard_count": len(shards)},
    }
    return {**report_core, "profile_sha256": canonical_json_sha256(report_core)}


def _validated_profile_shard(raw_shard: Mapping[str, Any]) -> Mapping[str, Any]:
    shard = dict(raw_shard)
    if shard.get("schema_version") != PRIOR_BELIEF_PROFILE_SCHEMA_VERSION:
        raise ValueError("profile shard has an unsupported schema version.")
    scope = _profile_mapping(shard, "profile_scope")
    if scope.get("kind") != "shard":
        raise ValueError("profile merge accepts only explicit shard reports.")
    profile_sha256 = shard.get("profile_sha256")
    report_core = {name: value for name, value in shard.items() if name != "profile_sha256"}
    if not isinstance(profile_sha256, str) or profile_sha256 != canonical_json_sha256(report_core):
        raise ValueError("profile shard hash does not match its payload.")
    return shard


def _profile_input_identity(profile: Mapping[str, Any]) -> tuple[object, ...]:
    provenance = _profile_mapping(profile, "provenance")
    return (
        profile.get("checkpoint_sha256"),
        profile.get("corpus_sha256"),
        profile.get("public_corpus_schema_sha256"),
        profile.get("root_noise"),
        profile.get("opponent_legal_mask_mode"),
        provenance.get("checkpoint_sha256"),
        provenance.get("source_corpus_sha256"),
        provenance.get("belief_set_source_hash"),
        provenance.get("opponent_legal_mask_mode"),
        provenance.get("root_noise_enabled"),
        provenance.get("opponent_scenarios"),
        provenance.get("corpus_manifest"),
    )


def _merge_representativeness_counters(
    shard: Mapping[str, Any],
    *,
    corpus_by_phase: Counter[str],
    event_bearing_by_phase: Counter[str],
    public_events_by_phase: Counter[str],
    unsupported_prefixes_by_phase: Counter[str],
    unsupported_events_by_phase: Counter[str],
) -> None:
    representativeness = _profile_mapping(shard, "representativeness")
    for row in _profile_sequence(representativeness, "by_phase"):
        phase = str(row.get("phase"))
        if phase not in {"early", "mid", "late"}:
            raise ValueError("profile shard representativeness has an unknown phase.")
        corpus_by_phase[phase] += _profile_nonnegative_int(row.get("corpus_decision_count"), field="corpus_decision_count")
        event_bearing_by_phase[phase] += _profile_nonnegative_int(
            row.get("event_bearing_prefix_count"), field="event_bearing_prefix_count"
        )
        public_events_by_phase[phase] += _profile_nonnegative_int(row.get("public_event_count"), field="public_event_count")
        unsupported_prefixes_by_phase[phase] += _profile_nonnegative_int(
            row.get("unsupported_event_prefix_count"), field="unsupported_event_prefix_count"
        )
        unsupported_events_by_phase[phase] += _profile_nonnegative_int(
            row.get("unsupported_public_event_count"), field="unsupported_public_event_count"
        )


def _profile_config_from_payload(payload: Mapping[str, Any]) -> PriorBeliefProfileConfig:
    return PriorBeliefProfileConfig(
        entropy_thresholds=tuple(float(value) for value in _profile_value_sequence(payload, "entropy_thresholds")),
        margin_thresholds=tuple(float(value) for value in _profile_value_sequence(payload, "margin_thresholds")),
        world_sample_cap=_profile_positive_int(payload.get("world_sample_cap"), field="world_sample_cap"),
        early_phase_max_turn=_profile_nonnegative_int(payload.get("early_phase_max_turn"), field="early_phase_max_turn"),
        mid_phase_max_turn=_profile_nonnegative_int(payload.get("mid_phase_max_turn"), field="mid_phase_max_turn"),
        opponent_legal_mask_mode=str(payload.get("opponent_legal_mask_mode")),
        root_noise_enabled=bool(payload.get("root_noise_enabled")),
    )


def _profile_mapping(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"profile {field} must be a mapping.")
    return value


def _profile_sequence(payload: Mapping[str, Any], field: str) -> Sequence[Mapping[str, Any]]:
    value = _profile_value_sequence(payload, field)
    if any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"profile {field} must contain mappings.")
    return value


def _profile_value_sequence(payload: Mapping[str, Any], field: str) -> Sequence[Any]:
    value = payload.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"profile {field} must be a sequence.")
    return value


def _profile_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"profile {field} must be a non-negative integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"profile {field} must be a non-negative integer.") from exc
    if result < 0:
        raise ValueError(f"profile {field} must be a non-negative integer.")
    return result


def _profile_positive_int(value: Any, *, field: str) -> int:
    result = _profile_nonnegative_int(value, field=field)
    if result <= 0:
        raise ValueError(f"profile {field} must be a positive integer.")
    return result


def initial_candidate_value_top_two_margin(candidate_values: Mapping[int, float]) -> tuple[float | None, float, float | None]:
    """Match the root search mandatory-sweep ranking: value descending, action ascending ties."""

    ranked = sorted(((int(action), float(value)) for action, value in candidate_values.items()), key=lambda item: (-item[1], item[0]))
    if not ranked:
        raise ValueError("candidate_values must not be empty.")
    top1 = ranked[0][1]
    top2 = ranked[1][1] if len(ranked) > 1 else None
    return (top1 - top2 if top2 is not None else None, top1, top2)


def _normalized_raw_legal_priors(raw_priors: Sequence[float], legal_actions: Sequence[int]) -> tuple[float, ...]:
    if len(raw_priors) != ACTION_COUNT:
        raise ValueError(f"raw prior evaluator must return {ACTION_COUNT} values.")
    values = tuple(float(value) for value in raw_priors)
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("raw prior evaluator must return finite non-negative probabilities.")
    total = sum(values[index] for index in legal_actions)
    if total <= 0.0:
        uniform = 1.0 / len(legal_actions)
        return tuple(uniform if index in legal_actions else 0.0 for index in range(ACTION_COUNT))
    return tuple(values[index] / total if index in legal_actions else 0.0 for index in range(ACTION_COUNT))


def _shannon_entropy(probabilities: Sequence[float]) -> float:
    return -sum(probability * math.log(probability) for probability in probabilities if probability > 0.0)


def _validate_context_legal_candidates(record: PublicDecisionRecord, context: WorldScenarioEvaluation) -> None:
    illegal = sorted(
        action
        for action in context.candidate_values
        if action < 0 or action >= len(record.current_legal_action_mask) or not record.current_legal_action_mask[action]
    )
    if illegal:
        raise ValueError(
            f"initial candidate sweep for {record.decision_id} contains acting-player illegal actions: {illegal}"
        )


def _candidate_evaluation_outcome(
    value: Sequence[WorldScenarioEvaluation] | CandidateValueEvaluation,
) -> CandidateValueEvaluation:
    if isinstance(value, CandidateValueEvaluation):
        return value
    contexts = tuple(value)
    if not all(isinstance(context, WorldScenarioEvaluation) for context in contexts):
        raise ValueError("candidate evaluator must return WorldScenarioEvaluation contexts.")
    if not contexts:
        return CandidateValueEvaluation(contexts=(), skip_reason="no_public_replay_contexts")
    return CandidateValueEvaluation(contexts=contexts)


def _skipped_decision_row(
    record: PublicDecisionRecord,
    *,
    phase: str,
    reason: str,
    event_summary: Mapping[str, Any],
    failure_reasons: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "decision_id": record.decision_id,
        "battle_id": record.battle_id,
        "turn_index": record.turn_index,
        "phase": phase,
        "reason": reason,
        "failure_reasons": dict(failure_reasons or {}),
        **dict(event_summary),
    }


def _threshold_sweeps(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    config: PriorBeliefProfileConfig,
    unit: str = "selection_context",
) -> list[dict[str, Any]]:
    if unit not in {"selection_context", "decision"}:
        raise ValueError("threshold sweep unit must be selection_context or decision.")
    rows: list[dict[str, Any]] = []
    for phase in ("early", "mid", "late"):
        phase_rows = [row for row in source_rows if row["phase"] == phase]
        for threshold in config.entropy_thresholds:
            rows.append(
                _sweep_row(
                    phase_rows,
                    phase=phase,
                    gate="entropy",
                    entropy_threshold=threshold,
                    margin_threshold=None,
                    unit=unit,
                )
            )
        for threshold in config.margin_thresholds:
            rows.append(
                _sweep_row(
                    phase_rows,
                    phase=phase,
                    gate="margin",
                    entropy_threshold=None,
                    margin_threshold=threshold,
                    unit=unit,
                )
            )
        for entropy_threshold in config.entropy_thresholds:
            for margin_threshold in config.margin_thresholds:
                rows.append(
                    _sweep_row(
                        phase_rows,
                        phase=phase,
                        gate="entropy_or_margin",
                        entropy_threshold=entropy_threshold,
                        margin_threshold=margin_threshold,
                        unit=unit,
                    )
                )
    return rows


def _sweep_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    gate: str,
    entropy_threshold: float | None,
    margin_threshold: float | None,
    unit: str,
) -> dict[str, Any]:
    contested = 0
    margin_eligible_rows = [
        row for row in rows if bool(row.get("candidate_margin_available"))
    ]
    forced_or_insufficient_count = len(rows) - len(margin_eligible_rows)
    for row in rows:
        entropy_contested = entropy_threshold is not None and float(row["raw_policy_entropy"]) >= entropy_threshold
        margin_contested = (
            margin_threshold is not None
            and bool(row.get("candidate_margin_available"))
            and float(row["initial_candidate_value_top_two_margin"]) <= margin_threshold
        )
        if (gate == "entropy" and entropy_contested) or (gate == "margin" and margin_contested) or (
            gate == "entropy_or_margin" and (entropy_contested or margin_contested)
        ):
            contested += 1
    count = len(rows)
    denominator = len(margin_eligible_rows) if gate == "margin" else count
    return {
        "phase": phase,
        "gate": gate,
        "entropy_threshold": entropy_threshold,
        "margin_threshold": margin_threshold,
        "rate_unit": "per_selection_context" if unit == "selection_context" else "per_decision_normalized",
        "selection_context_count": count if unit == "selection_context" else None,
        "decision_count": count if unit == "decision" else None,
        "margin_eligible_selection_context_count": len(margin_eligible_rows) if unit == "selection_context" else None,
        "margin_eligible_decision_count": len(margin_eligible_rows) if unit == "decision" else None,
        "forced_or_insufficient_context_count": forced_or_insufficient_count if unit == "selection_context" else None,
        "forced_or_insufficient_decision_count": forced_or_insufficient_count if unit == "decision" else None,
        "contested_count": contested,
        "contested_fraction": (contested / denominator) if denominator else None,
        "entropy_metric": "raw-shannon-nats-from-untempered-legal-priors",
        "margin_metric": "initial-candidate-value-top-two",
        "decision_aggregation": (
            "not_applicable"
            if unit == "selection_context"
            else "raw entropy per decision; scenario-weighted candidate margin over available contexts; one gate event per decision"
        ),
    }


def _representativeness_summary(
    *,
    corpus_by_phase: Mapping[str, int],
    event_bearing_by_phase: Mapping[str, int],
    public_events_by_phase: Mapping[str, int],
    unsupported_prefixes_by_phase: Mapping[str, int],
    unsupported_events_by_phase: Mapping[str, int],
    decision_rows: Sequence[Mapping[str, Any]],
    skipped_decision_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report which public prefix classes survived evaluation in each phase.

    The counters are accumulated while records are streamed so profiling does not
    retain the full history-heavy corpus just to build this summary.
    """

    profiled_by_phase = Counter(str(row["phase"]) for row in decision_rows)
    skipped_by_phase: dict[str, list[Mapping[str, Any]]] = {phase: [] for phase in ("early", "mid", "late")}
    for row in skipped_decision_rows:
        skipped_by_phase[str(row["phase"])].append(row)
    rows: list[dict[str, Any]] = []
    for phase in ("early", "mid", "late"):
        skipped = skipped_by_phase[phase]
        rows.append(
            {
                "phase": phase,
                "corpus_decision_count": int(corpus_by_phase.get(phase, 0)),
                "profiled_decision_count": profiled_by_phase[phase],
                "skipped_decision_count": len(skipped),
                "event_bearing_prefix_count": int(event_bearing_by_phase.get(phase, 0)),
                "public_event_count": int(public_events_by_phase.get(phase, 0)),
                "unsupported_event_prefix_count": int(unsupported_prefixes_by_phase.get(phase, 0)),
                "unsupported_public_event_count": int(unsupported_events_by_phase.get(phase, 0)),
                "skip_reason_counts": dict(sorted(Counter(str(row["reason"]) for row in skipped).items())),
            }
        )
    return {
        "definition": "event-bearing and unsupported-event counts are derived only from persisted public event identifiers; skip reasons are evaluator/public-prefix outcomes.",
        "by_phase": rows,
    }
