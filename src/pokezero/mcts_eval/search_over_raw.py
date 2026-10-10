"""Prospective Phase A contracts and paired raw-policy continuation audits.

No historical outcome pooling, search implementation, automatic training or
full-game launch. Private source snapshots belong only to the outcome auditor;
search adapters must receive their separately captured public requests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
import statistics
from typing import Any, Callable, Mapping, Sequence
import uuid


# Opening fixtures and the reference runtime's twenty synthetic startup games.
# This is a minimum exclusion, not the complete historical exposure inventory.
ENGINEERING_EXCLUDED_SEEDS = (2026101009, *range(2026100400, 2026100420))


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rng_seed(namespace: str, *parts: object) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps([namespace, *parts],
        separators=(",", ":")).encode()).digest()[:8], "big")


@dataclass(frozen=True)
class SearchConfiguration:
    arm: str
    belief: str = "public"
    leaf: str = "model"
    seconds: float = 10.0
    workers: int = 1

    def __post_init__(self) -> None:
        require(self.arm in {"raw", "incumbent", "reference"}, "unknown policy arm")
        require(self.belief in {"public", "oracle"}, "unknown information mode")
        require(self.leaf in {"model", "hp_fraction", "raw_rollout"}, "unknown leaf evaluator")
        require(type(self.seconds) in (int, float) and math.isfinite(self.seconds)
            and 0 < self.seconds <= 10, "decision ceiling must be in (0,10]")
        require(type(self.workers) is int and 1 <= self.workers <= 20,
            "worker count must be an integer in [1,20]")
        require(self.arm != "raw" or (self.belief == "public" and self.leaf == "model"
            and self.workers == 1), "raw control cannot use oracle or search options")

    @property
    def identity(self) -> str:
        return digest(asdict(self))

    @property
    def deployable(self) -> bool:
        return self.belief == "public" and self.arm != "raw"


def phase_a_contract(namespace: str, *, excluded_seeds: Sequence[int],
                     configurations: Sequence[SearchConfiguration], roots: int = 400,
                     seeds_per_panel: int = 32, continuations: int = 8) -> dict[str, Any]:
    """Allocate independent panels before collection; no replacement seeds.

    Freshness is relative to the explicitly bound exposure inventory, not a
    claim about any undisclosed historical data. The CLI binds that inventory.
    Root slots are quotas within fresh raw-vs-raw source battles. A collector
    selects requests using seed-derived random priorities across the complete
    public request catalog, not search changes or continuation outcomes.
    Short/failed source battles leave missing slots, never replacement seeds.
    """
    require(str(uuid.UUID(namespace)) == namespace, "canonical fresh UUID required")
    require(type(roots) is int and 300 <= roots <= 500 and roots % 2 == 0,
        "even 300–500 root slots required")
    require(type(seeds_per_panel) is int and seeds_per_panel >= 32
        and seeds_per_panel <= roots // 2, "at least32 source seeds in each panel")
    require(type(continuations) is int and continuations == 8, "eight paired continuations required")
    require(all(type(s) is int and 0 <= s < 2**32 for s in excluded_seeds),
        "exposure inventory must contain uint32 seeds")
    require(configurations and len({c.identity for c in configurations}) == len(configurations),
        "nonempty unique configuration roster required")
    require(any(c.arm == "raw" for c in configurations), "raw control is required")
    require(any(c.arm == "incumbent" and c.deployable for c in configurations)
        and any(c.arm == "reference" and c.deployable for c in configurations),
        "both deployable search arms required")
    excluded = set(excluded_seeds) | set(ENGINEERING_EXCLUDED_SEEDS)
    used = set(excluded)
    panels = {}
    ordinal = 0
    for panel in ("exploration", "validation"):
        seeds = []
        while len(seeds) < seeds_per_panel:
            seed = rng_seed(namespace, "source", ordinal) % 2**32
            ordinal += 1
            if seed not in used:
                used.add(seed)
                seeds.append(seed)
        slots = [dict(root_id=f"{panel}:{seeds[i % seeds_per_panel]}:{i // seeds_per_panel}",
            source_seed=seeds[i % seeds_per_panel], root_slot=i // seeds_per_panel)
            for i in range(roots // 2)]
        panels[panel] = dict(seeds=seeds, root_slots=slots)
    return dict(schema="pokezero.search-over-raw.phase-a.v1", namespace=namespace,
        status="PREPARED_NOT_COLLECTED", panels=panels,
        configurations=[dict(asdict(c), identity=c.identity, deployable=c.deployable)
                        for c in configurations],
        continuation_replicates=continuations,
        source_policy="raw_argmax_both_seats", candidate_seat="p1",
        source_root_rule="seed-derived priorities over the completed raw source request catalog; missing slots are not replaced",
        continuation_policy="raw_argmax_after_initial_sampled_opponent_reply",
        paired_randomness="root/replicate/step/domain; independent of selected action",
        estimand="equal-source-seed mean continuation win-score gain versus raw",
        failure_policy="uncertain outcomes; no complete-case gate or redraw",
        selection_rule="exploration only; freeze one deployable configuration before validation",
        historical_outcomes_pooled=False, phase_b_authorized=False,
        excluded_seeds=sorted(excluded),
        excluded_seed_inventory_sha256=digest(sorted(excluded)))


def select_source_requests(namespace: str, source_seed: int,
                           eligible_requests: Sequence[int], quota: int) -> tuple[int, ...]:
    """Sample across the battle, never just its easy first decisions.

    The collector must freeze the full request catalog before search/audits;
    this function deliberately accepts no actions, values or outcome fields.
    """
    require(type(source_seed) is int and 0 <= source_seed < 2**32, "uint32 source seed required")
    require(type(quota) is int and quota > 0, "positive root quota required")
    require(len(set(eligible_requests)) == len(eligible_requests)
        and all(type(r) is int and r >= 0 for r in eligible_requests),
        "unique nonnegative source request indices required")
    return tuple(sorted(eligible_requests, key=lambda r:
        (rng_seed(namespace, "root-selection", source_seed, r), r))[:quota])


def _raw_action(legal: Sequence[int], priors: Sequence[float]) -> int:
    require(len(legal) == len(priors) and len(set(legal)) == len(legal) and bool(legal),
        "nonempty unique legal actions aligned with priors required")
    require(all(type(a) is int and a >= 0 for a in legal), "invalid legal action")
    require(all(type(p) in (int, float) and math.isfinite(p) and p >= 0 for p in priors)
        and math.fsum(priors) > 0, "invalid legal policy distribution")
    return legal[max(range(len(legal)), key=lambda i: (priors[i], -legal[i]))]


def paired_continuations(*, env: Any, snapshot: object, subject: str,
                         actions: Mapping[str, int], evaluator: Callable,
                         namespace: str, root_id: str, max_boundaries: int = 200,
                         outcome_sink=None) -> dict:
    """Oct5 audit law, retaining identical actions and terminal uncertainty.

    Oracle snapshots are used AFTER action selection. The evaluator returns
    (legal_actions, normalized-or-unnormalized legal priors). Each unique action
    is run once per replicate; shared actions share the exact outcome receipt.
    No committed private opponent action is supplied to a search adapter.
    """
    require(subject in {"p1", "p2"}, "invalid candidate seat")
    require(actions and all(type(a) is int and a >= 0 for a in actions.values()),
        "frozen legal action indices required")
    require(type(max_boundaries) is int and max_boundaries > 0, "positive continuation cap required")
    rows = []
    for action in sorted(set(actions.values())):
        for replicate in range(8):
            env.restore(snapshot)
            require(env.terminal() is None, "continuation source root is already terminal")
            result = None
            for step in range(max_boundaries + 1):
                terminal = env.terminal()
                if terminal is not None:
                    result = dict(status="CAPPED" if terminal.capped else "COMPLETE",
                        signed_outcome=None if terminal.capped else
                        0 if terminal.winner is None else 1 if terminal.winner == subject else -1,
                        boundaries=step)
                    break
                if step == max_boundaries:
                    break
                requested = env.requested_players()
                require(requested and set(requested) <= {"p1", "p2"},
                    "invalid nonterminal request boundary")
                if step == 0:
                    require(subject in requested, "source root omits the candidate request")
                choices = {}
                for seat in requested:
                    legal, priors = evaluator(env.observe(seat))
                    argmax = _raw_action(legal, priors)
                    if step == 0 and seat == subject:
                        require(action in legal, "selected action is illegal at source root")
                        choices[seat] = action
                    elif step == 0:
                        draw = random.Random(rng_seed(namespace, root_id, replicate, step,
                            "opponent")).random() * math.fsum(priors)
                        cumulative = 0.0
                        choices[seat] = legal[-1]
                        for candidate, prior in zip(legal, priors):
                            cumulative += prior
                            if draw < cumulative:
                                choices[seat] = candidate
                                break
                    else:
                        choices[seat] = argmax
                env.reseed_simulator_rng(rng_seed(namespace, root_id, replicate, step, "chance"))
                env.step(choices)
            rows.append(dict(action=action, replicate=replicate,
                **(result or dict(status="CAPPED", signed_outcome=None, boundaries=max_boundaries))))
            if outcome_sink is not None:
                outcome_sink(dict(root_id=root_id, **rows[-1]))
    return dict(root_id=root_id, actions=dict(actions), outcomes=rows,
        status="COMPLETE" if all(r["status"] == "COMPLETE" for r in rows) else "UNCERTAIN")


def root_contrast(audit: Mapping, configuration: str) -> tuple[float, float]:
    """Sharp paired win-score bounds; a capped continuation is not a loss."""
    require(set(audit["actions"]) >= {"raw", configuration}, "paired control/arm missing")
    rows = {(r["action"], r["replicate"]): r for r in audit["outcomes"]}
    require(len(rows) == len(audit["outcomes"]), "duplicate continuation receipt")
    expected = {(a, r) for a in set(audit["actions"].values()) for r in range(8)}
    require(set(rows) == expected, "missing or extra continuation receipt")
    for row in rows.values():
        require((row["status"] == "COMPLETE" and type(row["signed_outcome"]) is int
                 and row["signed_outcome"] in (-1, 0, 1))
                or (row["status"] in {"CAPPED", "REFUSED"} and row["signed_outcome"] is None),
                "invalid terminal outcome/status")
    if audit["actions"][configuration] == audit["actions"]["raw"]:
        return (0.0, 0.0)
    lower = upper = 0.0
    for replicate in range(8):
        def interval(action):
            row = rows[(action, replicate)]
            return ((row["signed_outcome"] + 1) / 2,) * 2 if row["status"] == "COMPLETE" else (0., 1.)
        candidate = interval(audit["actions"][configuration])
        raw = interval(audit["actions"]["raw"])
        lower += (candidate[0] - raw[1]) / 8
        upper += (candidate[1] - raw[0]) / 8
    return lower, upper


def panel_summary(contract: Mapping, *, panel: str, configuration: str,
                  root_intervals: Mapping[str, Sequence[float]], bootstrap_reps: int = 10000) -> dict:
    """Keep full source-seed denominator; uncertainty disables validation gate."""
    require(panel in contract["panels"], "unknown panel")
    require(configuration in {c["identity"] for c in contract["configurations"]},
        "unregistered configuration")
    require(type(bootstrap_reps) is int and bootstrap_reps >= 1000, "at least1000 bootstrap draws required")
    slots = contract["panels"][panel]["root_slots"]
    require(set(root_intervals) <= {r["root_id"] for r in slots}, "unregistered root evidence")
    by_seed = {seed: [] for seed in contract["panels"][panel]["seeds"]}
    unknown = 0
    for root in slots:
        interval = root_intervals.get(root["root_id"], (-1., 1.))
        require(len(interval) == 2 and all(type(v) in (int, float) and math.isfinite(v)
            for v in interval) and -1 <= interval[0] <= interval[1] <= 1,
            "invalid continuation contrast interval")
        unknown += interval[0] != interval[1]
        by_seed[root["source_seed"]].append(interval)
    means = [(statistics.mean(x[0] for x in rows), statistics.mean(x[1] for x in rows))
             for rows in by_seed.values()]
    result = dict(panel=panel, configuration=configuration, root_slots=len(slots),
        source_seeds=len(means), uncertain_roots=unknown,
        identification_interval=[statistics.mean(x[0] for x in means),
                                 statistics.mean(x[1] for x in means)],
        bootstrap_interval=None, gate_positive=False, confirmatory_strength=False)
    if unknown == 0:
        values = [x[0] for x in means]
        rng = random.Random(rng_seed(contract["namespace"], panel, configuration, "bootstrap"))
        boot = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(bootstrap_reps))
        result["bootstrap_interval"] = [boot[int(.025 * bootstrap_reps)],
                                        boot[min(bootstrap_reps - 1, int(.975 * bootstrap_reps))]]
    return result


def freeze_selection(contract: Mapping, configuration: str, *, exploration_summary: Mapping) -> dict:
    require(exploration_summary["panel"] == "exploration"
        and exploration_summary["configuration"] == configuration,
        "selection must use matching exploration evidence only")
    require(exploration_summary["uncertain_roots"] == 0, "incomplete exploration cannot select a winner")
    config = next((c for c in contract["configurations"] if c["identity"] == configuration), None)
    require(config is not None and config["deployable"], "oracle/raw configuration cannot be promoted")
    return dict(schema="pokezero.search-over-raw.selection.v1", contract_sha256=digest(contract),
        configuration=configuration, exploration_summary_sha256=digest(exploration_summary),
        validation_opened=False, phase_b_authorized=False)


def validation_gate(contract: Mapping, selection: Mapping, summary: Mapping) -> dict:
    require(selection["contract_sha256"] == digest(contract), "selection/contract mismatch")
    require(summary["panel"] == "validation" and summary["configuration"] == selection["configuration"],
        "held-out evidence must match the frozen exploration selection")
    config = next(c for c in contract["configurations"] if c["identity"] == selection["configuration"])
    require(config["deployable"], "oracle configuration cannot pass validation")
    interval = summary["bootstrap_interval"]
    return dict(status="PHASE_A_GAIN_VALIDATED" if summary["uncertain_roots"] == 0
        and interval is not None and interval[0] > 0 else "NO_VALIDATED_GAIN",
        phase_b_authorized=False, full_game_strength_established=False,
        explanation="Phase B still requires calibration, reliability and source qualification gates")
