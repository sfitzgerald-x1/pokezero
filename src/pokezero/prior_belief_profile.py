"""Pure prior and public-belief uncertainty profiling for adaptive root budgets.

The metric layer in this module intentionally has no dependency on a model, a
simulator, or FoulPlay. Callers inject untempered priors and initial-sweep
candidate values. That makes the reported gate surface reproducible from the
public corpus and testable without a privileged battle state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Mapping, Sequence

from .actions import ACTION_COUNT, MOVE_ACTION_COUNT
from .determinization import BeliefWorldSamplingProfile, belief_world_sampling_profile
from .policy import PolicyContext
from .public_decision_corpus import (
    PUBLIC_DECISION_CORPUS_SCHEMA_SHA256,
    PublicDecisionCorpus,
    PublicDecisionRecord,
    canonical_json_sha256,
)


PRIOR_BELIEF_PROFILE_SCHEMA_VERSION = "pokezero.prior-belief-profile.v1"
MINIMUM_PROFILE_DECISIONS = 2_000
EARLY_PHASE_MAX_TURN = 5
MID_PHASE_MAX_TURN = 15

PriorEvaluator = Callable[[tuple[Any, ...]], Sequence[float]]
CandidateValueEvaluator = Callable[[PublicDecisionRecord], Sequence["WorldScenarioEvaluation"]]


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
    trajectory = _PublicReplayTrajectory(steps=own_history)
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
    records: Sequence[PublicDecisionRecord],
    *,
    prior_evaluator: PriorEvaluator,
    candidate_value_evaluator: CandidateValueEvaluator,
    config: PriorBeliefProfileConfig = PriorBeliefProfileConfig(),
    belief_set_source: Any | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Profile public decisions with raw priors and initial candidate values.

    No policy temperature is accepted here. ``prior_evaluator`` must supply the
    checkpoint's raw (temperature 1.0), root-noise-free prior vector. Candidate
    values are likewise injected from a public-prefix hidden-mode evaluator.
    """

    if not records:
        raise ValueError("prior/belief profile requires at least one public decision.")
    decision_rows: list[dict[str, Any]] = []
    selection_context_rows: list[dict[str, Any]] = []
    skipped_decision_rows: list[dict[str, Any]] = []
    for record in records:
        legal_actions = tuple(index for index, legal in enumerate(record.current_legal_action_mask) if legal)
        if not legal_actions:
            raise ValueError(f"public decision {record.decision_id} has no legal acting-player actions.")
        priors = _normalized_raw_legal_priors(prior_evaluator(record.observations()), legal_actions)
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
            skipped_decision_rows.append(_skipped_decision_row(record, reason="invalid_public_belief"))
            continue
        phase = phase_for_turn(
            record.turn_index,
            early_phase_max_turn=config.early_phase_max_turn,
            mid_phase_max_turn=config.mid_phase_max_turn,
        )
        contexts = tuple(candidate_value_evaluator(record))
        if not contexts:
            skipped_decision_rows.append(_skipped_decision_row(record, reason="no_public_replay_contexts"))
            continue
        margins: list[tuple[float, float]] = []
        for context in contexts:
            _validate_context_legal_candidates(record, context)
            margin, value_top1, value_top2 = initial_candidate_value_top_two_margin(context.candidate_values)
            if margin is not None:
                margins.append((context.scenario_weight, margin))
            margin_available = margin is not None
            selection_context_rows.append(
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
                    "selection_gate_inputs": {
                        "policy_entropy": raw_entropy,
                        "value_margin": margin,
                        "action_priors": [priors[index] for index in legal_actions],
                        "root_noise_enabled": False,
                        "opponent_legal_mask_mode": "hidden",
                    },
                }
            )
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
            }
        )
    profile_config = config.to_dict()
    provenance_payload = dict(provenance or {})
    report_core = {
        "schema_version": PRIOR_BELIEF_PROFILE_SCHEMA_VERSION,
        "profile_config": profile_config,
        "profile_config_sha256": canonical_json_sha256(profile_config),
        "checkpoint_sha256": provenance_payload.get("checkpoint_sha256"),
        "corpus_sha256": provenance_payload.get("corpus_sha256"),
        "public_corpus_schema_sha256": PUBLIC_DECISION_CORPUS_SCHEMA_SHA256,
        "root_noise": {"enabled": False, "root_dirichlet_alpha": None},
        "opponent_legal_mask_mode": "hidden",
        "corpus_decision_count": len(records),
        "decision_count": len(decision_rows),
        "skipped_decision_count": len(skipped_decision_rows),
        "skipped_decision_rows": skipped_decision_rows,
        "selection_context_count": len(selection_context_rows),
        "decision_rows": decision_rows,
        "selection_context_rows": selection_context_rows,
        "threshold_sweeps": _threshold_sweeps(selection_context_rows, config=config),
        "provenance": provenance_payload,
    }
    return {**report_core, "profile_sha256": canonical_json_sha256(report_core)}


def profile_public_corpus(
    corpus: PublicDecisionCorpus,
    *,
    prior_evaluator: PriorEvaluator,
    candidate_value_evaluator: CandidateValueEvaluator,
    config: PriorBeliefProfileConfig = PriorBeliefProfileConfig(),
    belief_set_source: Any | None = None,
    provenance: Mapping[str, Any] | None = None,
    minimum_decisions: int = MINIMUM_PROFILE_DECISIONS,
) -> dict[str, Any]:
    """Profile a corpus only when it satisfies the Step 2 2,000-decision floor."""

    if minimum_decisions < MINIMUM_PROFILE_DECISIONS:
        raise ValueError(f"minimum_decisions may not be lower than {MINIMUM_PROFILE_DECISIONS}.")
    if len(corpus.decisions) < minimum_decisions:
        raise ValueError(
            f"prior/belief profiling requires at least {minimum_decisions} valid p1 decisions; "
            f"corpus contains {len(corpus.decisions)}."
        )
    report = profile_public_decisions(
        corpus.decisions,
        prior_evaluator=prior_evaluator,
        candidate_value_evaluator=candidate_value_evaluator,
        config=config,
        belief_set_source=belief_set_source,
        provenance={
            **dict(provenance or {}),
            "corpus_sha256": corpus.corpus_sha256,
            "corpus_manifest": dict(corpus.manifest),
        },
    )
    if report["decision_count"] < minimum_decisions:
        raise ValueError(
            f"prior/belief profiling requires at least {minimum_decisions} successfully profiled p1 decisions; "
            f"corpus contains {len(corpus.decisions)} rows but only {report['decision_count']} produced public replay contexts."
        )
    return report


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
    illegal = sorted(action for action in context.candidate_values if not record.current_legal_action_mask[action])
    if illegal:
        raise ValueError(
            f"initial candidate sweep for {record.decision_id} contains acting-player illegal actions: {illegal}"
        )


def _skipped_decision_row(record: PublicDecisionRecord, *, reason: str) -> dict[str, Any]:
    return {
        "decision_id": record.decision_id,
        "battle_id": record.battle_id,
        "turn_index": record.turn_index,
        "reason": reason,
    }


def _threshold_sweeps(
    selection_context_rows: Sequence[Mapping[str, Any]],
    *,
    config: PriorBeliefProfileConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in ("early", "mid", "late"):
        phase_rows = [row for row in selection_context_rows if row["phase"] == phase]
        for threshold in config.entropy_thresholds:
            rows.append(
                _sweep_row(
                    phase_rows,
                    phase=phase,
                    gate="entropy",
                    entropy_threshold=threshold,
                    margin_threshold=None,
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
) -> dict[str, Any]:
    contested = 0
    margin_eligible_rows = [
        row for row in rows if bool(row.get("candidate_margin_available"))
    ]
    forced_or_insufficient_context_count = len(rows) - len(margin_eligible_rows)
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
        "selection_context_count": count,
        "margin_eligible_selection_context_count": len(margin_eligible_rows),
        "forced_or_insufficient_context_count": forced_or_insufficient_context_count,
        "contested_count": contested,
        "contested_fraction": (contested / denominator) if denominator else None,
        "entropy_metric": "raw-shannon-nats-from-untempered-legal-priors",
        "margin_metric": "initial-candidate-value-top-two",
    }
