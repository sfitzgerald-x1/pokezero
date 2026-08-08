#!/usr/bin/env python3
"""Reference controls for measuring `opponent_request_order` against Showdown.

WHAT THIS IS: three known-wrong reference implementations plus the written
contract for the measurement. Import the controls; the game-driving loop is the
caller's.

WHAT THIS IS NOT: a harness. It does not drive games and it reports no numbers.
An earlier version of this file printed "this harness drives real
gen3randombattle games" and then exited 0 having measured nothing, with its
--games/--seed-start flags parsed and ignored. That is the exact failure this
branch spent eight review rounds on, so the banner is gone and the flags with
it.

WHY THE CONTROLS MATTER. `engine_search.opponent_request_order` was wrong five
times, and each wrong version passed the tests written for it. The only thing
that ever caught the defect was scoring against the live opponent request order
WITH known-wrong baselines on the identical rows -- if the controls do not land
far from zero, the corpus does not discriminate and a clean subject score means
nothing. Rounds 6 and 7 measured 13,614 and 7,267 decisions, zero wrong, with
these controls at 81-96% wrong.

THE CONTRACT:

    subject       engine_search.opponent_request_order(context, base_party)
    ground truth  the opponent's own |request| side.pokemon species order,
                  read live (e.g. env._latest_requests[opponent])
    base_party    the opponent's FIRST live request order -- the packed
                  team-index order, since Showdown's battle-start switch-in is
                  a self-swap
    driver        real gen3randombattle games; rounds 6 and 7 used
                  rollout.RolloutDriver with both seats instrumented
    pass          0 wrong. None (fail-closed) is acceptable; report the rate
    valid only if the controls below score far from 0 wrong on the same rows

No circularity: the subject reads only our own observations, the opponent's
recorded action indices, and public protocol lines -- never the opponent's
request.
"""

from __future__ import annotations

__all__ = ["CONTRACT", "wrong_unevolved", "wrong_one_swap", "wrong_inverted"]

CONTRACT = {
    "subject": "engine_search.opponent_request_order(context, base_party)",
    "ground_truth": "opponent |request| side.pokemon species order, read live",
    "base_party": "the opponent's FIRST live request order (packed team-index order)",
    "pass": "0 wrong; report the None rate",
    "validity": "the controls must score far from 0 wrong on the same rows",
}


def wrong_unevolved(party, _truth):
    """Attempt 2: the packed party order, never evolved. ~96% wrong."""
    return list(party)


def wrong_one_swap(party, truth):
    """Attempt 3, and FORMERLY the crate's fallback: one slot-0 swap. ~91% wrong.

    Correct only while the opponent has made at most one switch-in.

    The crate no longer substitutes this when `ctx["opponent_request_order"]`
    is absent -- it fails closed to an all-`None` action map, leaving the node
    uniform and counting the refusal in `prior_fallbacks`
    (`LeafContext::root_opponent_order`). It stayed a control here because the
    controls are what make the subject's score readable.
    """
    if not truth:
        return list(party)
    order = list(party)
    active = truth[0]
    if active in order:
        index = order.index(active)
        if index:
            order[0], order[index] = order[index], order[0]
    return order


def wrong_inverted(party, truth):
    """The inverted permutation: self-consistent, passes shape checks, ~84% wrong."""
    inverse = [None] * len(truth)
    for position, species in enumerate(truth):
        if species in party:
            inverse[party.index(species)] = party[position]
    return inverse
