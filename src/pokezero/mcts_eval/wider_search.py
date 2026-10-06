"""Whole-game paired search comparison; independent unit is a battle seed.

No selected disagreement roots, raw takeover, redraw or refusal-as-loss scoring.
The two seats of a seed are dependent and must remain one analysis cluster.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import time
from typing import Any

ARMS = ("deep_incumbent", "paper_reference")
SEATS = ("p1", "p2")


def study_seed(index: int, *, qualification: bool = False) -> int:
    if type(index) is not int or index < 0:
        raise ValueError("invalid seed ordinal")
    domain = "qualification" if qualification else "confirmation"
    digest = hashlib.sha256(f"pokezero.wider-search.20261006.v1:{domain}:{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def game_identity(seed: int, seat: str, arm: str) -> str:
    if seat not in SEATS or arm not in ARMS:
        raise ValueError("invalid registered game")
    return f"seed-{seed}-{seat}-{arm}"


def play_game(env: Any, *, subject: str, decision_id: str, selector: Any,
              opponent: Any, emit: Any, max_boundaries: int, wall_seconds: float,
              start_boundary: int = 0, prior_selections: int = 0,
              elapsed_before_resume: float = 0.) -> dict:
    from .followthrough import continuation_seed, sampled_opponent
    if subject not in SEATS or max_boundaries <= 0 or wall_seconds <= 0:
        raise ValueError("invalid whole-game bounds")
    if (type(start_boundary) is not int or not 0 <= start_boundary <= max_boundaries
            or type(prior_selections) is not int or not 0 <= prior_selections <= start_boundary
            or type(elapsed_before_resume) not in (int, float)
            or not math.isfinite(elapsed_before_resume) or elapsed_before_resume < 0
            or start_boundary == 0 and (prior_selections or elapsed_before_resume)):
        raise ValueError('invalid retained whole-game prefix')
    started, selections = time.perf_counter(), prior_selections
    elapsed = lambda: elapsed_before_resume + time.perf_counter() - started
    for boundary in range(start_boundary, max_boundaries + 1):
        terminal = env.terminal()
        if terminal is not None:
            return dict(status="CAPPED" if terminal.capped else "COMPLETE",
                winner=terminal.winner, signed_outcome=None if terminal.capped else
                0 if terminal.winner is None else 1 if terminal.winner == subject else -1,
                boundaries=boundary, own_decisions=selections,
                elapsed_seconds=elapsed())
        if boundary == max_boundaries or elapsed() >= wall_seconds:
            return dict(status="CAPPED", signed_outcome=None, boundaries=boundary,
                own_decisions=selections, reason="registered_whole_game_safety_cap",
                elapsed_seconds=elapsed())
        requested = tuple(env.requested_players())
        if not requested:
            raise RuntimeError("nonterminal game has no requested actor")
        actions, evidence = {}, {}
        for actor in requested:
            observation = env.observe(actor)
            if actor == subject:
                action, row = selector(observation, boundary,
                    continuation_seed(decision_id, 0, boundary, "search"))
                selections += 1
            else:
                action, row = sampled_opponent(opponent, observation,
                    continuation_seed(decision_id, 0, boundary, "opponent"))
            if (type(action) is not int or not 0 <= action < len(observation.legal_action_mask)
                    or not observation.legal_action_mask[action]):
                raise RuntimeError("whole-game selector returned an illegal action")
            actions[actor], evidence[actor] = action, row
        chance_seed = continuation_seed(decision_id, 0, boundary, "chance")
        emit(dict(boundary=boundary, actions=actions, evidence=evidence, chance_seed=chance_seed))
        env.reseed_simulator_rng(chance_seed)
        env.step(actions)
    raise AssertionError("unreachable whole-game loop")


def exact_signflip_p(differences: list[float]) -> float:
    """Two-sided, seed-cluster sign-flip test, exact integer convolution.

    Assumes independent clusters and exchangeability of the two arm labels
    under the null. Does not pretend individual turns/seats are independent.
    Outcomes are quarter-unit contrasts (two seats; draws score one half).
    """
    if not differences:
        raise ValueError("empty analysis")
    values = []
    for difference in differences:
        scaled = round(4 * difference)
        if not math.isfinite(difference) or abs(difference) > 1 or abs(scaled - 4 * difference) > 1e-9:
            raise ValueError("invalid paired seed contrast")
        values.append(scaled)
    observed = abs(sum(values))
    distribution = Counter({0: 1})
    for value in values:
        updated = Counter()
        for total, count in distribution.items():
            updated[total + value] += count
            updated[total - value] += count
        distribution = updated
    return sum(count for total, count in distribution.items() if abs(total) >= observed) / (2 ** len(values))


def analyze(seeds: list[int], rows: list[dict], *, alpha: float = .05) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or not 0 < alpha < 1:
        raise ValueError("invalid fixed seed roster")
    expected = {game_identity(seed, seat, arm) for seed in seeds for seat in SEATS for arm in ARMS}
    indexed = {}
    for row in rows:
        identity = game_identity(row["seed"], row["subject"], row["arm"])
        if identity != row["identity"] or identity not in expected or identity in indexed:
            raise ValueError("extra, duplicate or drifted game identity")
        if row["status"] == "COMPLETE":
            if type(row.get("signed_outcome")) is not int or row["signed_outcome"] not in (-1, 0, 1):
                raise ValueError("invalid complete game outcome")
        elif row.get("signed_outcome") is not None:
            raise ValueError("noncomplete game must never be scored")
        indexed[identity] = row
    contrasts = []
    for seed in seeds:
        group = [indexed.get(game_identity(seed, seat, arm)) for arm in ARMS for seat in SEATS]
        if any(row is None or row["status"] != "COMPLETE" for row in group):
            continue
        scores = {arm: sum((indexed[game_identity(seed, seat, arm)]["signed_outcome"] + 1) / 2
            for seat in SEATS) / 2 for arm in ARMS}
        contrasts.append(dict(seed=seed, scores=scores, difference=scores[ARMS[1]] - scores[ARMS[0]]))
    missing = len(seeds) - len(contrasts)
    total = sum(row["difference"] for row in contrasts)
    result = dict(registered_seeds=len(seeds), registered_games=len(expected),
        observed_games=len(indexed), complete_games=sum(row["status"] == "COMPLETE" for row in rows),
        complete_seed_clusters=len(contrasts), missing_seed_clusters=missing, contrasts=contrasts,
        mean_delta_worst_case_bounds=[(total - missing) / len(seeds), (total + missing) / len(seeds)],
        inferential_test_allowed=False, statistically_supported_advantage=False)
    if missing:
        result["status"] = "INCOMPLETE_NO_COMPLETE_CASE_INFERENCE"
        return result
    delta = total / len(seeds)
    # Distribution-free confidence bound for independent bounded seed clusters.
    # Conservative: do not turn failure to reject into equivalence or a small-effect claim.
    radius = math.sqrt(2 * math.log(2 / alpha) / len(seeds))
    interval = [max(-1., delta - radius), min(1., delta + radius)]
    p = exact_signflip_p([row["difference"] for row in contrasts])
    result.update(status="COMPLETE_FIXED_ROSTER", inferential_test_allowed=True,
        mean_win_score_delta=delta, bounded_mean_confidence_interval=interval,
        confidence_level=1-alpha, exact_seed_cluster_signflip_p=p,
        statistically_supported_advantage=p <= alpha and interval[0] > 0,
        limitations=["sign-flip inference assumes arm-label exchangeability under the null",
            "bounded mean interval assumes independent seed clusters and is conservative",
            "same champion opponent, not Foul Play or all-opponent superiority",
            "fixed N; no stopping when a favorable p-value first appears"])
    return result
