"""Bounded, frozen-first-action continuations; not a whole-game strength claim.

The oracle environment owns the true source world. Selectors receive only the
acting observation through their adapter; no snapshot crosses that boundary.
The same opponent policy and independent per-boundary RNG domains are used in
all arms. A cap/refusal is never scored as a completed game or replaced by raw.
"""
from __future__ import annotations

import hashlib
import random
import time
from typing import Any, Callable


def continuation_seed(decision_id: str, replicate: int, boundary: int, domain: str) -> int:
    if domain not in ("search", "opponent", "chance"):
        raise ValueError("unknown follow-through RNG domain")
    raw = f"pokezero.followthrough.v1:{decision_id}:{replicate}:{boundary}:{domain}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def masked_argmax(evaluator: Any, observation: Any) -> tuple[int, dict[str, Any]]:
    legal, evaluated = evaluator(observation)
    index = max(range(len(legal)), key=lambda j: (evaluated.priors[j], -legal[j]))
    return legal[index], {"selector": "raw_masked_argmax", "legal": legal,
        "priors": evaluated.priors, "signed_value": evaluated.value}


def sampled_opponent(evaluator: Any, observation: Any, seed: int) -> tuple[int, dict[str, Any]]:
    legal, evaluated = evaluator(observation)
    action = random.Random(seed).choices(legal, weights=evaluated.priors, k=1)[0]
    return action, {"selector": "champion_full_masked_policy_sample", "seed": seed,
        "legal": legal, "priors": evaluated.priors, "signed_value": evaluated.value}


def play_continuation(env: Any, *, subject: str, first_action: int, decision_id: str,
        replicate: int, subject_selector: Callable, opponent_evaluator: Any,
        emit: Callable, max_boundaries: int = 200, wall_seconds: float = 3600.) -> dict[str, Any]:
    """Search every subsequent own request, including one-sided switch requests.

    ``subject_selector`` receives only observation/boundary/search-seed. Its
    production adapter may obtain that actor's public materialization state,
    but must not read the oracle snapshot or opponent request.
    """
    if subject not in ("p1", "p2") or type(first_action) is not int:
        raise ValueError("invalid frozen first intervention")
    if max_boundaries <= 0 or wall_seconds <= 0:
        raise ValueError("invalid continuation safety cap")
    started = time.perf_counter()
    selections = 0
    first_applied = False
    for boundary in range(max_boundaries + 1):
        terminal = env.terminal()
        if terminal is not None:
            return {"status": "CAPPED" if terminal.capped else "COMPLETE",
                "winner": terminal.winner, "turn_count": terminal.turn_count,
                "signed_outcome": None if terminal.capped else 0 if terminal.winner is None else
                    1 if terminal.winner == subject else -1,
                "boundaries": boundary, "followthrough_decisions": selections,
                "first_action_applied": first_applied,
                "elapsed_seconds": time.perf_counter() - started}
        if boundary == max_boundaries or time.perf_counter() - started >= wall_seconds:
            return {"status": "CAPPED", "signed_outcome": None, "boundaries": boundary,
                "followthrough_decisions": selections, "first_action_applied": first_applied,
                "reason": "registered_continuation_safety_cap",
                "elapsed_seconds": time.perf_counter() - started}
        requested = tuple(env.requested_players())
        if not requested or (boundary == 0 and subject not in requested):
            raise RuntimeError("nonterminal boundary lacks the registered acting request")
        actions = {}
        evidence = {}
        for player in requested:
            observation = env.observe(player)
            if player == subject and boundary == 0:
                action, row = first_action, {"selector": "frozen_historical_first_action"}
                first_applied = True
            elif player == subject:
                seed = continuation_seed(decision_id, replicate, boundary, "search")
                action, row = subject_selector(observation, boundary, seed)
                selections += 1
            else:
                action, row = sampled_opponent(opponent_evaluator, observation,
                    continuation_seed(decision_id, replicate, boundary, "opponent"))
            if type(action) is not int or not 0 <= action < len(observation.legal_action_mask) or not observation.legal_action_mask[action]:
                raise RuntimeError("continuation selector returned an illegal action")
            actions[player], evidence[player] = action, row
        chance_seed = continuation_seed(decision_id, replicate, boundary, "chance")
        emit({"boundary": boundary, "actions": actions, "evidence": evidence,
            "chance_seed": chance_seed})
        env.reseed_simulator_rng(chance_seed)
        env.step(actions)
    raise AssertionError("unreachable continuation loop")
