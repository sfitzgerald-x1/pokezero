"""Prospective statistical kernels, NOT scientific collection admission.

Two one-sided search-minus-raw claims against Foul Play spend alpha/2 each.
Fixed positive stakes are frozen before outcomes, not selected from live data.
For D in [-1,1] with E[D_t | past] <= 0, product(1+stake*D_t) is a
nonnegative supermartingale. Lower uncertainty endpoints decrease it pointwise,
so arbitrary outcome-dependent missingness cannot manufacture a crossing.
Ville plus the union bound controls either false claim, even with a shared raw
control. Empirical Bernstein assumes iid clusters, uses a union bound over
registered looks, and abstains on uncertain prefixes: its variance penalty is
NOT monotone under arbitrary lower-endpoint imputation.

Sources: https://arxiv.org/abs/2010.09686 (betting) and
https://arxiv.org/abs/0907.3740 (Maurer/Pontil Theorem4). Simulation is not proof.
Neither bootstrap intervals nor t-tests are admitted as calibrated kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Mapping, Sequence

from .search_over_raw import require

ARMS = ("incumbent", "reference")


def _finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_intervals(rows: Sequence[Sequence[float]]) -> None:
    require(bool(rows), "nonempty fixed-order contrast prefix required")
    require(all(len(row) == 2 and all(_finite_number(x) for x in row)
        and -1 <= row[0] <= row[1] <= 1 for row in rows), "invalid contrast interval")


def primary_cluster_intervals(cells: Mapping[tuple[str, str], Mapping]) -> dict:
    """Six primary cells: each arm/raw against FoulPlay, both seats.

    Absent/refused/capped/unvalidated COMPLETE cells retain [0,1]. A shared
    uncertain raw cell enters BOTH contrasts with raw's upper endpoint for the
    lower bound. This is partial identification, never a reported loss score.
    Secondary-opponent cells are not inputs to these primary contrasts.
    """
    expected = {(arm, seat) for arm in ("raw", *ARMS) for seat in ("p1", "p2")}
    require(set(cells) <= expected, "unregistered primary cell")
    intervals = {}
    for key in expected:
        row = cells.get(key)
        if row is None:
            intervals[key] = (0., 1.)
            continue
        status = row.get("status")
        require(status in {"COMPLETE", "REFUSED", "CAPPED", "UNSTARTED"}, "invalid cell status")
        require(type(row.get("validated")) is bool, "explicit validation flag required")
        score = row.get("score")
        if status == "COMPLETE" and row["validated"]:
            require(_finite_number(score) and score in (0, .5, 1), "invalid validated win score")
            intervals[key] = (score, score)
        else:
            require(not row["validated"], "only COMPLETE cells can be validated")
            require(score is None or (status == "COMPLETE" and _finite_number(score)
                and score in (0, .5, 1)), "uncertain terminal has invalid score")
            intervals[key] = (0., 1.)
    return {arm: (
        math.fsum(intervals[(arm, seat)][0] - intervals[("raw", seat)][1]
            for seat in ("p1", "p2")) / 2,
        math.fsum(intervals[(arm, seat)][1] - intervals[("raw", seat)][0]
            for seat in ("p1", "p2")) / 2) for arm in ARMS}


@dataclass(frozen=True)
class JointDesign:
    """Synthetic/prospective analysis specification; no model/worker admission."""
    seeds: tuple[int, ...]
    looks: tuple[int, ...]
    method: str = "fixed_stake_betting"
    stakes: tuple[float, float] = (.4, .4)
    alpha: float = .05
    refusal_cap: int = 2

    def __post_init__(self):
        require(type(self.seeds) is tuple and len(self.seeds) in (64, 128, 256)
            and len(set(self.seeds)) == len(self.seeds)
            and all(type(s) is int and 0 <= s < 2**32 for s in self.seeds),
            "fixed unique64/128/256 uint32 seed roster required")
        require(type(self.looks) is tuple and bool(self.looks)
            and all(type(n) is int and 2 <= n <= len(self.seeds) for n in self.looks)
            and tuple(sorted(set(self.looks))) == self.looks and self.looks[-1] == len(self.seeds),
            "sorted distinct registered looks ending at cap required")
        require(self.method in {"fixed_stake_betting", "hoeffding", "empirical_bernstein"},
            "uncalibrated or unknown analysis method")
        require(type(self.stakes) is tuple and len(self.stakes) == 2
            and all(_finite_number(s) and 0 < s < 1 for s in self.stakes),
            "two frozen strictly positive stakes below1 required")
        require(_finite_number(self.alpha) and 0 < self.alpha <= .05, "family alpha must be in (0,.05]")
        require(type(self.refusal_cap) is int and 0 <= self.refusal_cap <= len(self.seeds),
            "invalid refused-cluster cap")


def log_wealth(rows: Sequence[Sequence[float]], stake: float) -> float:
    validate_intervals(rows)
    require(_finite_number(stake) and 0 < stake < 1, "stake must be in (0,1)")
    return math.fsum(math.log1p(stake * row[0]) for row in rows)


def lower_confidence_bound(rows: Sequence[Sequence[float]], alpha: float,
                           method: str) -> float | None:
    validate_intervals(rows)
    require(_finite_number(alpha) and 0 < alpha < 1, "invalid tail probability")
    n = len(rows)
    if method == "hoeffding":
        return max(-1., statistics.mean(row[0] for row in rows)
            - math.sqrt(2 * math.log(1 / alpha) / n))
    require(method == "empirical_bernstein", "unknown confidence bound")
    if n < 2 or any(row[0] != row[1] for row in rows):
        return None
    values = [row[0] for row in rows]
    logarithm = math.log(2 / alpha)
    radius = math.sqrt(2 * statistics.variance(values) * logarithm / n)
    radius += 14 * logarithm / (3 * (n - 1))  # range2, not range1
    return max(-1., statistics.mean(values) - radius)


def adjudicate(design: JointDesign, records: Sequence[Mapping]) -> dict:
    """Recompute scheduled crossings; stop after both successes or third refusal.

    Caller supplies the exact prefix of the registered seed order. Extra records
    after a terminal boundary fail, never silently trim/reorder evidence. Each
    unresolved attempted cluster is counted once even if several cells fail.
    Unknown primary intervals cannot be labelled non-refused to evade the cap;
    a secondary-only failure may count with precise primary contrasts. Provenance or
    worker failure immediately invalidates the attempt; no retries are implied.
    Missing suffix seeds retain [-1,1] in full-roster identification bounds.
    """
    require(len(records) <= len(design.seeds), "prefix exceeds fixed roster")
    prefixes = {arm: [] for arm in ARMS}
    crossings = {arm: None for arm in ARMS}
    refused = 0
    terminal = None
    last_bounds = {arm: None for arm in ARMS}
    for ordinal, row in enumerate(records, 1):
        require(terminal is None, "evidence continues past terminal boundary")
        require(type(row.get("seed")) is int and row["seed"] == design.seeds[ordinal - 1],
            "records must follow exact registered seed order")
        require(type(row.get("refused")) is bool and type(row.get("provenance_ok")) is bool,
            "explicit cluster refusal/provenance flags required")
        require(set(row.get("contrasts", {})) == set(ARMS), "both primary contrasts required")
        for arm in ARMS:
            interval = row["contrasts"][arm]
            validate_intervals([interval])
            prefixes[arm].append(interval)
        require(row["refused"] or all(prefixes[arm][-1][0] == prefixes[arm][-1][1]
            for arm in ARMS), "uncertain attempted cluster must count against refusal cap")
        refused += row["refused"]
        if not row["provenance_ok"]:
            terminal = "INVALID_PROVENANCE_OR_WORKER_STATE"
        elif refused > design.refusal_cap:
            terminal = "REFUSAL_CAP_EXCEEDED"
        elif ordinal in design.looks:
            for index, arm in enumerate(ARMS):
                if design.method == "fixed_stake_betting":
                    crossed = log_wealth(prefixes[arm], design.stakes[index]) >= math.log(2 / design.alpha)
                else:
                    last_bounds[arm] = lower_confidence_bound(prefixes[arm],
                        design.alpha / (2 * len(design.looks)), design.method)
                    crossed = last_bounds[arm] is not None and last_bounds[arm] > 0
                if crossed and crossings[arm] is None:
                    crossings[arm] = ordinal
            if all(crossings.values()):
                terminal = "BOTH_PRIMARY_THRESHOLDS_CROSSED"
            elif ordinal == len(design.seeds):
                terminal = "CAP_REACHED"
    missing = len(design.seeds) - len(records)
    full_bounds = {arm: [
        (math.fsum(x[0] for x in prefixes[arm]) - missing) / len(design.seeds),
        (math.fsum(x[1] for x in prefixes[arm]) + missing) / len(design.seeds)] for arm in ARMS}
    return dict(status=terminal or "PREFIX_NOT_TERMINAL", primary_crossings=crossings,
        refused_clusters=refused, observed_clusters=len(records), roster_clusters=len(design.seeds),
        full_roster_identification_intervals=full_bounds, latest_lower_bounds=last_bounds,
        alpha_procedure="fixed Bonferroni, NOT Holm", family_alpha=design.alpha,
        scientific_admission=False, phase_b_authorized=False, retries_permitted=False)
