"""Synthetic stopped-scan sensitivity, NOT full Phase A calibration.

No model, source game, panel, holdout, or scientific receipt is read. Families
below are assumptions, not fits to the completed 320 panel. The real wall cap,
source validation/scheduling and cross-stage configuration selection are NOT
modeled, so this program cannot certify or authorize a screen. In particular,
an outcome-independent root-count stop is not a simulation of an eight-hour
collection attempt. Preserve earlier toy outputs as historical evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math

import numpy as np


@dataclass(frozen=True)
class Family:
    name: str
    effects: tuple[float, float] = (0., 0.)
    skew_cluster: bool = False
    shared_candidate: bool = False
    shared_membership: bool = False
    missing_positive: bool = False
    unavailable: bool = False

    def __post_init__(self):
        if self.shared_candidate and self.effects[0] != self.effects[1]:
            raise ValueError("one aliased candidate action cannot have two effects")


# Declared before any outcome simulations; none is an empirical distribution.
FAMILIES = (
    Family("symmetric_null"),
    Family("skew_seed_null", skew_cluster=True),
    Family("shared_candidate_null", shared_candidate=True),
    Family("shared_membership_null", shared_membership=True),
    Family("positive_score_missing_null", missing_positive=True),
    Family("unavailable_selector_null", unavailable=True),
    Family("partial_null", effects=(0., .10)),
    *(Family(f"plus_{round(delta * 100):02}", effects=(delta, delta))
      for delta in (.05, .10, .20)),
)
TARGET, CAP, MIN_DEVIATIONS, MIN_SOURCES = 100, 2000, 20, 32


def first_stop(actions):
    """Action-only first hit: both targets, root cap or three unavailable rows.

    0 is raw, positive is a distinct action, -1 is unavailable. Unknown actions
    consume the denominator and never count as agreements or deviations.
    This does not model later source-gate invalidation or elapsed wall time.
    """
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 2 or len(actions) > CAP:
        raise ValueError("at most 2000 two-leaf action rows required")
    if not np.issubdtype(actions.dtype, np.integer) or np.any(actions < -1):
        raise ValueError("integer action identities >= -1 required")
    unavailable = (actions < 0).any(axis=1)
    counts = (actions > 0).cumsum(axis=0)
    for i in range(len(actions)):
        if i >= 2 and unavailable[i-2:i+1].all():
            return i + 1, "THREE_CONSECUTIVE_UNAVAILABLE"
        if i + 1 >= CAP:
            return i + 1, "ROOT_CAP"
        if (counts[i] >= TARGET).all():
            return i + 1, "DEVIATION_TARGET"
    return len(actions), "SYNTHETIC_STREAM_EXHAUSTED"


def flip_scores(raw, delta, uniforms):
    """Valid 0/1 action scores with E[candidate-raw]=delta, P(tie)=.6.

    Requires marginal raw win probability .5. Conditional flip probabilities
    .4+delta (raw lost) and .4-delta (raw won) give EXACT win-score effects,
    not a delta attenuated by the tie probability or multiplied by 50.
    """
    delta = np.asarray(delta)
    if np.any(np.abs(delta) > .4):
        raise ValueError("effect outside realizable flip family")
    flip = uniforms < np.where(raw == 0, .4 + delta, .4 - delta)
    return np.where(flip, 1 - raw, raw)


def intervals(actions, scores, validated):
    """Vector equivalent of eight-pair interval arithmetic, shared raw cells.

    scores/validated have shape (roots, actions, 8). An action alias is zero
    even with no games; missing or unvalidated cells are [0,1], not losses.
    """
    n = len(actions)
    if scores.shape != validated.shape or scores.shape[0] != n or scores.shape[2] != 8:
        raise ValueError("eight paired action-level score cells required")
    result = np.empty((n, 2, 2), dtype=float)
    raw_low = np.where(validated[:, 0], scores[:, 0], 0.)
    raw_high = np.where(validated[:, 0], scores[:, 0], 1.)
    for leaf in range(2):
        a = np.maximum(actions[:, leaf], 0)
        known = validated[np.arange(n), a]
        candidate = scores[np.arange(n), a]
        result[:, leaf, 0] = (np.where(known, candidate, 0.) - raw_high).mean(axis=1)
        result[:, leaf, 1] = (np.where(known, candidate, 1.) - raw_low).mean(axis=1)
        result[actions[:, leaf] == 0, leaf] = 0.
        result[actions[:, leaf] < 0, leaf] = (-1., 1.)
    return result


def panel(rng, family, p, roots_per_source):
    """Generate common controls BEFORE leaf contrasts; select by actions only."""
    seeds = np.arange(CAP) // roots_per_source
    dev = rng.random((CAP, 2)) < p
    if family.shared_membership:
        dev[:, 1] = dev[:, 0]
    actions = dev * np.array([1, 2], dtype=int)
    if family.shared_candidate:
        actions[dev[:, 1], 1] = 1
    if family.unavailable:
        actions[rng.random((CAP, 2)) < .03] = -1
    n, reason = first_stop(actions)
    actions, seeds = actions[:n], seeds[:n]
    raw = (rng.random((n, 8)) < .5).astype(float)
    scores = np.empty((n, 3, 8)); scores[:, 0] = raw
    shock = np.zeros(seeds.max() + 1)
    if family.skew_cluster:
        # Mean zero, rare negative seeds: .8*(+.1)+.2*(-.4)=0.
        shock = np.where(rng.random(len(shock)) < .8, .1, -.4)
    for leaf in range(2):
        delta = family.effects[leaf] + shock[seeds]
        scores[:, leaf+1] = flip_scores(raw, delta[:, None], rng.random((n, 8)))
    validated = np.ones(scores.shape, dtype=bool)
    if family.missing_positive:
        # Outcome-dependent missingness of ACTION CELLS, including shared raw.
        validated &= ~((scores == 1) & (rng.random(scores.shape) < .25))
    return actions, seeds, intervals(actions, scores, validated), reason


def lower_endpoints(bounds, seeds, indices):
    """Pooled-root bootstrap, same cluster indices for both leaves/endpoints."""
    groups = np.unique(seeds)
    sums = np.stack([bounds[seeds == s, :, 0].sum(axis=0) for s in groups])
    counts = np.array([(seeds == s).sum() for s in groups])
    means = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]
    return np.sort(means, axis=0)[int(.05 * len(indices))]


def rate(hits, n):
    """Wilson interval reports Monte Carlo uncertainty, not scientific evidence."""
    p = hits / n; z = 1.96; den = 1 + z*z/n
    middle = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return dict(hits=int(hits), replicates=n, estimate=p,
                monte_carlo_wilson_95=[middle-half, middle+half])


def run(*, replicates, draws, seed):
    if replicates < 1 or draws < 20:
        raise ValueError("positive cohorts and >=20 bootstrap draws required")
    rng = np.random.default_rng(seed)
    rows = []
    for family in FAMILIES:
        for p in (.02, .10, .20):
            for size in (6, 48):
                hits = np.zeros(2, dtype=int); false_any = 0
                eligible = np.zeros(2, dtype=int); missing_source_minimum = 0
                scanned = []; target = 0
                stops = {reason: 0 for reason in ("ROOT_CAP", "DEVIATION_TARGET",
                    "THREE_CONSECUTIVE_UNAVAILABLE", "SYNTHETIC_STREAM_EXHAUSTED")}
                for _ in range(replicates):
                    actions, seeds, bounds, reason = panel(rng, family, p, size)
                    k = len(np.unique(seeds)); counts = (actions > 0).sum(axis=0)
                    can_read = (counts >= MIN_DEVIATIONS) & (k >= MIN_SOURCES)
                    positive = np.zeros(2, dtype=bool)
                    if can_read.any():
                        indices = rng.integers(k, size=(draws, k))
                        positive = can_read & (lower_endpoints(bounds, seeds, indices) > 0)
                    hits += positive
                    false_any += bool((positive & (np.asarray(family.effects) <= 0)).any())
                    eligible += can_read; missing_source_minimum += k < MIN_SOURCES
                    scanned.append(len(actions))
                    target += bool((counts >= TARGET).all())
                    stops[reason] += 1
                rows.append(dict(family=family.name, conditional_win_score_effects=family.effects,
                    deviation_membership="identical" if family.shared_membership else "independent",
                    assumed_deviation_rate=p, assumed_roots_per_source=size,
                    endpoint_positive_by_leaf={leaf: dict(rate(int(hits[j]), replicates),
                        interpretation="false_positive_frequency" if family.effects[j] <= 0 else
                            "detection_frequency") for j, leaf in enumerate(("model", "hp_fraction"))},
                    any_null_leaf_positive=rate(false_any, replicates),
                    count_and_source_eligibility_by_leaf={leaf: rate(int(eligible[j]), replicates)
                        for j, leaf in enumerate(("model", "hp_fraction"))},
                    below_32_source_minimum=rate(missing_source_minimum, replicates),
                    stop_reasons={reason: rate(count, replicates) for reason, count in stops.items()},
                    both_targets_reached=rate(target, replicates), mean_scanned_roots=float(np.mean(scanned))))
    return dict(schema="pokezero.sparse-stopped-scan-sensitivity.v1", seed=seed,
        replicates=replicates, bootstrap_draws=draws, results=rows,
        score_scale="candidate win score minus raw win score; multiply by 100 for percentage points",
        source_lengths_are_assumed=True, wall_runtime_modeled=False,
        source_gate_scheduling_modeled=False, configuration_selection_modeled=False,
        equal_seed_estimand_calibrated=False, simulation_is_not_a_proof=True,
        full_phase_a_calibration=False, screening_authorized=False,
        holdout_authorized=False, phase_b_authorized=False, scientific_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026101702)
    args = parser.parse_args()
    print(json.dumps(run(replicates=args.replicates, draws=args.draws, seed=args.seed), indent=2))
