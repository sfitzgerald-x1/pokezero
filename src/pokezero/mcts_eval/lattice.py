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

from statistics import mean, median
import time
from typing import Any, Callable, Sequence

from .manifest import SearchConfig
from .report import (
    DECISION_WALL_GATE_S,
    MIN_CORPUS_DECISIONS_BEFORE_EARLY_STOP,
    TimingRow,
)
from .resolver import CheckpointContract
from .timing_corpus import TimingDecisionRecord


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
    gate_s: float = DECISION_WALL_GATE_S,
) -> TimingRow:
    """Time one lattice cell over the corpus.

    ``decide`` performs one decision and returns its telemetry; it is injected so
    the timing loop itself is testable without the native crate. The default
    builds the controlled engine-MCTS policy for this cell.
    """
    if decide is None:
        decide = _default_decider(contract, showdown_root)

    walls: list[float] = []
    depths: list[int] = []
    cap_hits = 0
    fallbacks = 0
    invalid_actions = 0
    encode_s = model_s = tree_s = 0.0
    root_actions: list[str] = []
    gate_failed = False

    for index, record in enumerate(records, start=1):
        started = time.perf_counter()
        telemetry = decide(record, config)
        walls.append(time.perf_counter() - started)

        realized = int(telemetry.get("max_depth_reached", 0))
        depths.append(realized)
        if realized >= config.depth - 1:
            cap_hits += 1
        fallbacks += int(telemetry.get("fallbacks", 0))
        invalid_actions += int(telemetry.get("invalid_actions", 0))
        encode_s += float(telemetry.get("encode_s", 0.0))
        model_s += float(telemetry.get("model_s", 0.0))
        tree_s += float(telemetry.get("tree_s", 0.0))
        root_actions.append(str(telemetry.get("root_action", "")))

        if index >= MIN_CORPUS_DECISIONS_BEFORE_EARLY_STOP and mean(walls) >= gate_s:
            # A result, not a harness failure: persist the partial sample.
            gate_failed = True
            break

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
        encode_s=encode_s,
        model_s=model_s,
        tree_s=tree_s,
        fallbacks=fallbacks,
        invalid_actions=invalid_actions,
        gate_failed=gate_failed,
        provenance_exact=True,
        root_argmax_by_decision=tuple(root_actions),
    )


def _default_decider(
    contract: CheckpointContract, showdown_root: str | None
) -> Callable[[TimingDecisionRecord, SearchConfig], dict[str, Any]]:
    """Build the controlled engine-MCTS policy and run one corpus decision.

    Prefix replay warms the incremental fold BEFORE the caller's timer starts
    for the next record; within a decision only the request->choice span is
    timed, matching the plan's definition.
    """

    def decide(record: TimingDecisionRecord, config: SearchConfig) -> dict[str, Any]:
        raise NotImplementedError(
            "the live decider requires the model-enabled crate plus an exported "
            "TorchScript artifact and encoder tables; pass decide=... explicitly "
            "until the study image is available."
        )

    return decide
