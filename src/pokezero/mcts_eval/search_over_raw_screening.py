"""Reconciled Phase A screening and staged selection, without runtime imports.

The legacy betting contracts remain historical; this kernel is for the new
design only. Calling it does not certify calibration, freeze a runtime ledger,
allocate sources, open a holdout, or authorize collection/Phase B. Percentile
bootstrap nominal levels are not a finite-sample validity theorem.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
import uuid
from typing import Mapping, Sequence

ARMS = ("incumbent", "reference")
DEPLOYABLE_LEAVES = ("model", "hp_fraction")
BOOTSTRAP_DRAWS = 10000
LOWER_ORDER_INDEX = 500


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def interval(row):
    require(isinstance(row, (tuple, list)) and len(row) == 2
        and all(finite(x) for x in row) and -1 <= row[0] <= row[1] <= 1,
        "finite ordered contrast interval in [-1,1] required")
    return tuple(row)


def action_score(row):
    if row is None:
        return (0., 1.)
    require(row.get("status") in {"COMPLETE", "CAPPED", "REFUSED", "UNSTARTED"}
        and type(row.get("validated")) is bool, "explicit status/validation required")
    score = row.get("score")
    require(score is None or (finite(score) and score in (0, .5, 1)), "invalid win score")
    if row["validated"]:
        require(row["status"] == "COMPLETE" and score is not None,
            "only complete terminal scores can be validated")
        return (score, score)
    require(score is None or row["status"] == "COMPLETE", "uncertain status cannot carry a score")
    return (0., 1.)


def paired_root_interval(candidate_action: int, raw_action: int,
                         outcomes: Mapping[tuple[int, int], Mapping]) -> tuple[float, float]:
    """Eight paired continuation replicates; missing raw affects every consumer.

    An action identity denotes one shared outcome cell. The actual collector
    must bind that identity to its receipt; this arithmetic cannot prove
    provenance. Unvalidated COMPLETE is uncertain, not a scored result.
    """
    require(all(type(a) is int and a >= 0 for a in (candidate_action, raw_action)),
        "bound nonnegative action identities required")
    require(all(type(key) is tuple and len(key) == 2 and type(key[0]) is int
        and key[0] >= 0 and type(key[1]) is int and 0 <= key[1] < 8 for key in outcomes),
        "invalid action/replicate cell identity")
    # Validate referenced cells even for aliases: bad receipts are not excused
    # by an algebraically zero contrast. Other actions belong to other claims.
    scores = {a: [action_score(outcomes.get((a, r))) for r in range(8)]
              for a in {candidate_action, raw_action}}
    if candidate_action == raw_action:
        return (0., 0.)
    candidate, raw = scores[candidate_action], scores[raw_action]
    return (math.fsum(c[0] - r[1] for c, r in zip(candidate, raw)) / 8,
            math.fsum(c[1] - r[0] for c, r in zip(candidate, raw)) / 8)


def cluster_intervals(roster: tuple[tuple[int, tuple[str, ...]], ...],
                      roots: Mapping[str, Sequence[float]]) -> tuple[tuple[float, float], ...]:
    """Fixed root quotas within seeds, equal seed weights; no source allocation."""
    require(type(roster) is tuple and 32 <= len(roster) <= 200, "fixed >=32 seed roster required")
    require(all(type(row) is tuple and len(row) == 2 and type(row[0]) is int
        and 0 <= row[0] < 2**32 and type(row[1]) is tuple and row[1]
        and all(type(root) is str and root for root in row[1]) for row in roster),
        "uint32 seeds with fixed nonempty root quotas required")
    seeds = [seed for seed, _ in roster]
    slots = [root for _, quota in roster for root in quota]
    require(len(set(seeds)) == len(seeds) and len(set(slots)) == len(slots), "duplicate seed/root")
    require(set(roots) <= set(slots), "unregistered root")
    result = []
    for _, quota in roster:
        rows = [interval(roots.get(root, (-1., 1.))) for root in quota]
        result.append((math.fsum(row[0] for row in rows) / len(rows),
                       math.fsum(row[1] for row in rows) / len(rows)))
    return tuple(result)


@dataclass(frozen=True)
class ScreeningDesign:
    namespace: str
    seeds: tuple[int, ...]

    def __post_init__(self):
        require(type(self.namespace) is str, "canonical analysis UUID required")
        try:
            canonical = str(uuid.UUID(self.namespace))
        except (ValueError, AttributeError):
            raise ValueError("canonical analysis UUID required") from None
        require(canonical == self.namespace, "canonical analysis UUID required")
        require(type(self.seeds) is tuple and 32 <= len(self.seeds) <= 200
            and all(type(s) is int and 0 <= s < 2**32 for s in self.seeds)
            and len(set(self.seeds)) == len(self.seeds), "fixed unique uint32 seeds required")

    @property
    def random_seed(self):
        payload = json.dumps([self.namespace, "phase-a-screening-bootstrap"],
            separators=(",", ":")).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def screen(design: ScreeningDesign, intervals_by_arm: Mapping[str, Sequence | None]) -> dict:
    """One complete-panel screening calculation, NOT an execution ledger.

    The caller must provide seed intervals in the frozen design's order. None
    represents an unavailable arm, not permission to remove it. Sharing the
    resample indices preserves covariance from shared raw control cells.
    """
    require(isinstance(design, ScreeningDesign) and set(intervals_by_arm) == set(ARMS),
        "both declared arms required")
    n = len(design.seeds)
    rows = {}
    unavailable = {}
    for arm in ARMS:
        supplied = intervals_by_arm[arm]
        unavailable[arm] = supplied is None
        if supplied is None:
            rows[arm] = ((-1., 1.),) * n
        else:
            require(len(supplied) == n, "full fixed seed denominator required")
            rows[arm] = tuple(interval(row) for row in supplied)
    lows = {arm: tuple(row[0] for row in rows[arm]) for arm in ARMS}
    boot = {arm: [] for arm in ARMS}
    rng = random.Random(design.random_seed)
    for _ in range(BOOTSTRAP_DRAWS):
        indices = rng.choices(range(n), k=n)
        for arm in ARMS:
            boot[arm].append(math.fsum(lows[arm][i] for i in indices) / n)
    result = {}
    for arm in ARMS:
        lower = sorted(boot[arm])[LOWER_ORDER_INDEX]
        positive = not unavailable[arm] and lower > 0
        result[arm] = dict(status="UNAVAILABLE" if unavailable[arm] else
            "SCREEN_POSITIVE" if positive else "NOT_SCREEN_POSITIVE",
            nominal_alpha=.05, lower_percentile_endpoint=lower, screen_positive=positive,
            identification_interval=[math.fsum(row[i] for row in rows[arm]) / n for i in (0, 1)],
            source_seeds=n, uncertain_seeds=sum(row[0] != row[1] for row in rows[arm]))
    return dict(schema="pokezero.search-over-raw.screening-kernel.v1", arms=result,
        method="paired_seed_cluster_percentile_bootstrap", bootstrap_draws=BOOTSTRAP_DRAWS,
        lower_order_index=LOWER_ORDER_INDEX, resample_indices_shared=True,
        any_arm_screen_positive=any(row["screen_positive"] for row in result.values()),
        both_arms_screen_positive=all(row["screen_positive"] for row in result.values()),
        family_error_control_claim=False, calibration_certified=False,
        confirmatory_strength=False, collection_authorized=False, phase_b_authorized=False)


@dataclass(frozen=True)
class StageSummary:
    stage: str
    arm: str
    belief: str
    leaf: str
    seconds: float | None
    lower_mean: float
    accepted_units: int
    status: str
    lane: str = "deployment"

    def __post_init__(self):
        require(self.stage in {"A2", "A3"} and self.arm in ARMS
            and self.belief in {"public", "oracle"}
            and self.leaf in {*DEPLOYABLE_LEAVES, "raw_rollout"}, "invalid stage/configuration")
        require(self.lane in {"deployment", "wall_clock_diagnostic", "fixed_work_diagnostic"},
            "explicit resource lane required")
        if self.lane == "fixed_work_diagnostic":
            require(self.stage == "A2" and self.belief == "public" and self.seconds is None,
                "fixed-work diagnostic cannot have a search-decision ceiling")
        else:
            require(finite(self.seconds) and self.seconds in (1, 3, 10)
                and self.leaf != "raw_rollout", "terminal rollouts are fixed-work only")
        if self.lane == "deployment":
            require(self.belief == "public" and self.leaf in DEPLOYABLE_LEAVES,
                "diagnostics cannot be deployment summaries")
        if self.stage == "A3":
            require(self.lane == "deployment", "A3 admits deployment ceilings only")
        require(finite(self.lower_mean) and -1 <= self.lower_mean <= 1, "invalid score")
        require(type(self.accepted_units) is int and self.accepted_units >= 0
            and self.status in {"QUALIFIED", "UNQUALIFIED_AT_CEILING", "DIAGNOSTIC_INCOMPLETE",
                                "UNSTARTED", "OPERATIONAL_STOP"},
            "explicit work/qualification status required")
        require(self.status != "DIAGNOSTIC_INCOMPLETE" or self.lane == "fixed_work_diagnostic",
            "diagnostic incompleteness requires fixed-work lane")
        require(self.lane != "fixed_work_diagnostic" or self.status != "UNQUALIFIED_AT_CEILING",
            "fixed-work diagnostics have no decision ceiling")
        require(self.status != "UNSTARTED" or self.accepted_units == 0,
            "unstarted configuration cannot have accepted work")
        require(self.status != "QUALIFIED" or self.accepted_units > 0,
            "qualified configuration requires accepted work")
        require(self.status != "QUALIFIED" or self.lane != "fixed_work_diagnostic"
            or self.accepted_units >= 64, "fixed-work diagnostic requires its 64 accepted-unit target")


def _check_stage(rows, stage):
    require(all(isinstance(row, StageSummary) and row.stage == stage for row in rows), "wrong stage")
    identities = [(r.arm, r.belief, r.leaf, r.lane, r.seconds) for r in rows]
    require(len(set(identities)) == len(identities), "duplicate configuration")
    require(not any(row.status == "OPERATIONAL_STOP" for row in rows),
        "operational integrity failure stops selection")


def choose_evaluators(rows: Sequence[StageSummary]) -> tuple[tuple[str, str | None], ...]:
    """A2 only; A1 and oracle/terminal diagnostic results cannot select deployables."""
    _check_stage(rows, "A2")
    eligible = [r for r in rows if r.lane == "deployment"
                and r.belief == "public" and r.leaf in DEPLOYABLE_LEAVES]
    require({(r.arm, r.leaf) for r in eligible} == {(a, l) for a in ARMS for l in DEPLOYABLE_LEAVES}
        and all(r.seconds == 1 for r in eligible), "exact public A2 roster at 1 second required")
    result = []
    for arm in ARMS:
        available = [r for r in eligible if r.arm == arm and r.status == "QUALIFIED"]
        best = max(available, key=lambda r: (r.lower_mean, -DEPLOYABLE_LEAVES.index(r.leaf))) if available else None
        result.append((arm, best.leaf if best else None))
    return tuple(result)


def choose_ceilings(evaluators: tuple[tuple[str, str | None], ...],
                    rows: Sequence[StageSummary]) -> tuple[tuple[str, str | None, float | None], ...]:
    """A3 cannot reselect evaluator; absent arms remain explicit unavailable arms."""
    require(type(evaluators) is tuple and len(evaluators) == 2
        and all(type(row) is tuple and len(row) == 2 for row in evaluators)
        and tuple(row[0] for row in evaluators) == ARMS
        and all(row[1] is None or row[1] in DEPLOYABLE_LEAVES for row in evaluators),
        "both frozen A2 choices required")
    _check_stage(rows, "A3")
    frozen = dict(evaluators)
    require(all(r.belief == "public" and r.leaf == frozen[r.arm] for r in rows),
        "A3 cannot change evaluator or promote diagnostics")
    expected = {(arm, ceiling) for arm, leaf in evaluators if leaf is not None for ceiling in (1, 3, 10)}
    require({(r.arm, r.seconds) for r in rows} == expected, "full frozen A3 ceiling roster required")
    result = []
    for arm in ARMS:
        available = [r for r in rows if r.arm == arm and r.status == "QUALIFIED"]
        best = max(available, key=lambda r: (r.lower_mean, -r.seconds)) if available else None
        result.append((arm, best.leaf if best else None, best.seconds if best else None))
    return tuple(result)
