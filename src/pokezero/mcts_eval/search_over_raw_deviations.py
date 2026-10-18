"""Deviation-triggered exploration arithmetic, not a collection entry point.

The caller must validate original source, live-gate and continuation receipts.
This module cannot establish provenance, launch work, open a holdout or certify
an adaptive bootstrap. Unavailable selections never become agreement zeros.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence

from .search_over_raw_screening import paired_root_interval

LEAVES = ("model", "hp_fraction")
INFORMATION_ARMS = ("public", "oracle")
TARGET_DEVIATIONS = 100
MAX_ROOTS = 2000
MAX_WALL_SECONDS = 28800
MIN_SCREEN_DEVIATIONS = 20
MIN_SOURCE_CLUSTERS = 32


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _action(value):
    return type(value) is int and value >= 0


@dataclass(frozen=True)
class ScannedRoot:
    """One reserved eligible boundary, including failed selector attempts.

    A missing action means unavailable OR not independently gate-validated;
    the calling reader must not pass a merely returned but unvalidated action.
    Root identity is the original seed/request index, not a sampled quota slot.
    """

    seed: int
    request_index: int
    raw: int
    model: int | None
    hp_fraction: int | None

    def __post_init__(self):
        _require(type(self.seed) is int and 0 <= self.seed < 2**32,
                 "original uint32 source seed required")
        _require(type(self.request_index) is int and self.request_index > 0,
                 "eligible non-opening source boundary required")
        _require(_action(self.raw) and all(value is None or _action(value)
                 for value in (self.model, self.hp_fraction)), "invalid action identity")

    def deviation(self, leaf):
        action = self.action(leaf)
        return None if action is None else action != self.raw

    @property
    def candidates(self):
        return LEAVES

    def action(self, name):
        _require(name in self.candidates, "unsupported candidate")
        return getattr(self, name)

    @property
    def continuation_actions(self):
        """Unique actions for eight paired CRN replicates, or no games at all.

        If just one leaf differs, the agreeing leaf needs no separate action.
        If both leaves select the same differing action, share that action bank.
        Unknown selections do not trigger a made-up action or prove agreement.
        """
        actions = {a for a in (self.model, self.hp_fraction) if a is not None}
        if not actions - {self.raw}:
            return ()
        return tuple(sorted(actions | {self.raw}))


@dataclass(frozen=True)
class ScannedInformationRoot:
    """Actual information-arm identities, never aliases for model/HP leaves."""
    seed: int
    request_index: int
    raw: int
    public: int | None
    oracle: int | None

    def __post_init__(self):
        _require(type(self.seed) is int and 0 <= self.seed < 2**32
                 and type(self.request_index) is int and self.request_index > 0,
                 "original nonopening source boundary required")
        _require(_action(self.raw) and all(a is None or _action(a)
                 for a in (self.public, self.oracle)), "invalid action identity")

    @property
    def candidates(self):
        return INFORMATION_ARMS

    action = ScannedRoot.action
    deviation = ScannedRoot.deviation

    @property
    def continuation_actions(self):
        actions = {a for a in (self.public, self.oracle) if a is not None}
        return tuple(sorted(actions | {self.raw})) if actions - {self.raw} else ()


def root_intervals(root: ScannedRoot, outcomes: Mapping):
    """Retain missing/refused/unvalidated games as intervals, not losses.

    The shared action identity makes an accepted selector/raw alias exactly
    zero even when no games were played. The existing paired reader checks
    supplied cells, including malformed receipts on an alias.
    """
    _require(isinstance(root, (ScannedRoot, ScannedInformationRoot)), "validated scan identity required")
    return {leaf: (-1., 1.) if root.action(leaf) is None else
            paired_root_interval(root.action(leaf), root.raw, outcomes)
            for leaf in root.candidates}


def validate_scan(roots: Sequence[ScannedRoot]):
    _require(all(isinstance(root, (ScannedRoot, ScannedInformationRoot)) for root in roots), "scan rows required")
    _require(not roots or all(r.candidates == roots[0].candidates for r in roots),
             "cannot mix scientific comparisons")
    identities = [(root.seed, root.request_index) for root in roots]
    _require(len(identities) == len(set(identities)), "duplicate source boundary; no retries")
    _require(len(roots) <= MAX_ROOTS, "hard scanned-root cap exceeded")
    last = {}
    for root in roots:
        _require(root.request_index > last.get(root.seed, 0), "source boundaries out of order")
        last[root.seed] = root.request_index


def stop_reason(roots: Sequence[ScannedRoot], elapsed_seconds: float):
    """Check before reserving another boundary; failures consume the cap too.

    The source-cluster minimum is an interpretation rule, not permission to
    overrun Scott's target, hard root cap or calendar cap. No retries/redraws.
    """
    validate_scan(roots)
    _require(type(elapsed_seconds) in (int, float) and math.isfinite(elapsed_seconds)
             and elapsed_seconds >= 0, "finite elapsed wall time required")
    if elapsed_seconds >= MAX_WALL_SECONDS:
        return "WALL_CAP"
    if len(roots) >= MAX_ROOTS:
        return "ROOT_CAP"
    if all(sum(root.deviation(leaf) is True for root in roots) >= TARGET_DEVIATIONS
           for leaf in (roots[0].candidates if roots else LEAVES)):
        return "DEVIATION_TARGET"
    return None


def _aggregate(groups, leaf, weighting):
    """All three quantities use the same weights and the same denominator."""
    total = sum(len(group) for group in groups)
    _require(total > 0 and all(groups), "nonempty scanned source clusters required")
    rows = []
    for group in groups:
        weight = 1 / total if weighting == "pooled_roots" else 1 / (len(groups) * len(group))
        rows.extend((root, intervals[leaf], weight) for root, intervals in group)
    known_deviation = math.fsum(w for r, _, w in rows if r.deviation(leaf) is True)
    unknown = math.fsum(w for r, _, w in rows if r.deviation(leaf) is None)
    overall = tuple(math.fsum(w * bounds[j] for _, bounds, w in rows) for j in (0, 1))
    # Unknown actions make the membership of the conditional subset unknown.
    # No deviations is an undefined conditional mean, NOT a zero outcome.
    conditional = None if unknown or not known_deviation else tuple(
        math.fsum(w * bounds[j] for r, bounds, w in rows if r.deviation(leaf) is True)
        / known_deviation for j in (0, 1))
    return dict(deviation_rate=(known_deviation, min(1., known_deviation + unknown)),
                conditional_contrast=conditional, overall_effect=overall,
                conditional_mean_defined=not unknown and known_deviation > 0,
                conditional_mean_status="UNCERTAIN_MEMBERSHIP" if unknown else
                    "DEFINED" if known_deviation else "UNDEFINED_NO_DEVIATIONS")


def summarize(roots: Sequence[ScannedRoot], intervals: Sequence[Mapping], *,
              weighting="pooled_roots", bootstrap_seed: int, bootstrap_draws=10000):
    """Joint seed-cluster bootstrap of the stopped scan, explicitly descriptive.

    Never multiply marginal interval endpoints: calculate the overall effect
    from the same resampled clusters. Selection/count-dependent stopping still
    needs whole-procedure calibration before any screening or holdout read.
    Empty planned sources cannot be added as fake zero observations. A panel
    missing source coverage is reported incomplete, not silently qualified.
    """
    validate_scan(roots)
    _require(roots and len(roots) == len(intervals), "all scanned roots need intervals")
    candidates = roots[0].candidates
    _require(weighting in {"pooled_roots", "equal_seed"}, "explicit weighting required")
    _require(type(bootstrap_seed) is int and type(bootstrap_draws) is int
             and bootstrap_draws >= 20, "bound bootstrap seed and >=20 draws required")
    grouped = {}
    for root, bounds in zip(roots, intervals):
        _require(set(bounds) == set(candidates), "both candidates required on every scanned root")
        for leaf in candidates:
            value = bounds[leaf]
            _require(isinstance(value, (tuple, list)) and len(value) == 2
                     and all(type(x) in (int, float) and math.isfinite(x) for x in value)
                     and -1 <= value[0] <= value[1] <= 1, "finite contrast interval required")
            if root.deviation(leaf) is None:
                _require(tuple(value) == (-1., 1.), "unavailable selection must stay uncertain")
            elif root.deviation(leaf) is False:
                _require(tuple(value) == (0., 0.), "accepted action alias must be exactly zero")
        grouped.setdefault(root.seed, []).append((root, bounds))
    groups = list(grouped.values())
    metrics = ("deviation_rate", "conditional_contrast", "overall_effect")
    boot = {leaf: {metric: [[], []] for metric in metrics} for leaf in candidates}
    conditional_statuses = {leaf: {status: 0 for status in
        ("DEFINED", "UNDEFINED_NO_DEVIATIONS", "UNCERTAIN_MEMBERSHIP")} for leaf in candidates}
    rng = random.Random(bootstrap_seed)
    for _ in range(bootstrap_draws):
        sampled = [groups[i] for i in rng.choices(range(len(groups)), k=len(groups))]
        for leaf in candidates:
            estimate = _aggregate(sampled, leaf, weighting)
            conditional_statuses[leaf][estimate["conditional_mean_status"]] += 1
            for metric in metrics:
                if estimate[metric] is None:
                    continue
                for endpoint in (0, 1):
                    boot[leaf][metric][endpoint].append(estimate[metric][endpoint])
    result = {}
    for leaf in candidates:
        estimate = _aggregate(groups, leaf, weighting)
        deviations = sum(root.deviation(leaf) is True for root in roots)
        unknown = sum(root.deviation(leaf) is None for root in roots)
        ci = {metric: None if len(boot[leaf][metric][0]) != bootstrap_draws else
                      [sorted(boot[leaf][metric][0])[int(.025 * bootstrap_draws)],
                       sorted(boot[leaf][metric][1])[min(bootstrap_draws - 1,
                                                                 int(.975 * bootstrap_draws))]]
              for metric in metrics}
        lower = sorted(boot[leaf]["overall_effect"][0])[int(.05 * bootstrap_draws)]
        result[leaf] = dict(identification_intervals=estimate,
            descriptive_seed_bootstrap_95=ci, deviations=deviations,
            unavailable_selections=unknown, one_sided_lower_endpoint=lower,
            conditional_bootstrap_draw_statuses=conditional_statuses[leaf],
            conditional_interval_status="DEFINED_ALL_DRAWS" if
                conditional_statuses[leaf]["DEFINED"] == bootstrap_draws else
                "UNAVAILABLE_SOME_DRAWS_UNDEFINED_OR_MEMBERSHIP_UNCERTAIN",
            minimum_count_satisfied=deviations >= MIN_SCREEN_DEVIATIONS,
            source_minimum_satisfied=len(groups) >= MIN_SOURCE_CLUSTERS,
            strictly_positive_lower_endpoint=lower > 0,
            screening_authorized=False, holdout_authorized=False)
    return dict(schema="pokezero.search-over-raw.deviation-summary.v1",
        **({"leaves":result} if candidates == LEAVES else {"candidates":result}),
        scanned_roots=len(roots), observed_source_clusters=len(groups), weighting=weighting,
        conditional_weighting="pooled deviations" if weighting == "pooled_roots" else
            "deviations weighted by inverse source root count",
        bootstrap_draws=bootstrap_draws, bootstrap_seed=bootstrap_seed,
        resample_indices_shared=True, adaptive_stopping_validity_certified=False,
        complete_case_inference_authorized=False, calibration_certified=False,
        collection_authorized=False, phase_b_authorized=False, scientific_strength_claim=False)
