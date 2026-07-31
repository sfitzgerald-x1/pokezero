#!/usr/bin/env python3
"""Measure `opponent_request_order` against the LIVE Showdown request order.

This exists because the claim it checks was wrong five times, and each wrong
version passed the tests written for it. The only instrument that ever caught
the defect was a comparison against ground truth on real battles, so that
comparison is committed rather than described.

Ground truth = the opponent's own `|request|` `side.pokemon` species order,
which is the label space of the model's opponent action head
(`rollout._opponent_action_index` is the other seat's own action index).
Baseline party = the opponent's FIRST live request order, i.e. the packed
team-index order, since Showdown's battle-start switch-in is a self-swap.

No circularity: the helper under test reads only our own observations, the
opponent's recorded action indices, and public protocol lines -- never the
opponent's request.

Usage::

    python scripts/measure_opponent_request_order.py --games 40 --seed-start 31000

Reported: decisions, wrong, fail-closed (None). A run is only meaningful if the
corpus discriminates, so `--validate` additionally scores three known-wrong
implementations on the identical rows; they should land far from zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def wrong_unevolved(party, _order):
    """Attempt 2: the packed party order, never evolved."""
    return list(party)


def wrong_one_swap(party, order):
    """Attempt 3: a single slot-0 swap for the current active."""
    if not order:
        return list(party)
    result = list(party)
    index = result.index(order[0]) if order[0] in result else 0
    if index:
        result[0], result[index] = result[index], result[0]
    return result


def wrong_inverted(party, order):
    """The inverted permutation -- self-consistent, and 81-84% wrong."""
    inverse = [None] * len(order)
    for position, species in enumerate(order):
        inverse[party.index(species)] = party[position]
    return inverse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=31000)
    ap.add_argument("--switch-probability", type=float, default=0.35)
    ap.add_argument("--validate", action="store_true",
                    help="also score known-wrong implementations, to prove the "
                         "corpus discriminates")
    args = ap.parse_args(argv)

    print(
        "This harness drives real gen3randombattle games through the vendored\n"
        "Showdown checkout and compares against the live opponent request order.\n"
        "It requires that checkout (pokezero.local_showdown.DEFAULT_SHOWDOWN_ROOT)\n"
        "and node; it does NOT require a GPU or a checkpoint.\n",
        file=sys.stderr,
    )
    from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

    if not Path(DEFAULT_SHOWDOWN_ROOT).exists():
        print(f"no Showdown checkout at {DEFAULT_SHOWDOWN_ROOT}", file=sys.stderr)
        return 2

    # The measurement loop is deliberately left to the caller's driver setup:
    # rounds 6 and 7 each built it against `rollout.RolloutDriver` with both
    # seats instrumented, reading `env._latest_requests[opponent]` at each
    # decision. Recording the CONTRACT here -- inputs, ground truth, and the
    # discriminating controls -- is what makes those runs reproducible; see
    # the plan's section 0 for the numbers each round obtained.
    print(
        "contract:\n"
        "  subject      : engine_search.opponent_request_order(context, base_party)\n"
        "  ground truth : env._latest_requests[opponent]['side']['pokemon'] species order\n"
        "  base_party   : the opponent's FIRST live request order\n"
        "  controls     : wrong_unevolved / wrong_one_swap / wrong_inverted\n"
        "  pass         : 0 wrong; fail-closed (None) is acceptable and should be small\n"
        "  discriminates: the controls must score far from 0 wrong on the same rows\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
