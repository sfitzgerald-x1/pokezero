"""Run a fixed baseline policy (random-legal / simple-legal / max-damage) as an online Showdown
bot, so it can play foul-play on a local server for calibration. Additive: reuses the existing
OnlineBattleAgent + ShowdownClient; imports no foul-play code.

  python scripts/play_online_baseline.py --policy max-damage --showdown-root ROOT \
      --websocket ws://localhost:8000/showdown/websocket --username MaxDmgBot \
      --accept --no-login --max-games 100
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys

from pokezero.dex import load_showdown_dex_cached
from pokezero.online_client import DEFAULT_FORMAT, DEFAULT_LOCAL_WS, OnlineBattleAgent, ShowdownClient
from pokezero.policy import MaxDamagePolicy, RandomLegalPolicy, SimpleLegalPolicy
from pokezero.randbat_vocab import gen3_category_vocabulary
from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC


def _build_policy(name: str, showdown_root: str, seed: int | None):
    if name == "random-legal":
        return RandomLegalPolicy()
    if name == "simple-legal":
        return SimpleLegalPolicy()
    if name == "max-damage":
        return MaxDamagePolicy(showdown_root=showdown_root)
    raise SystemExit(f"unknown --policy {name!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=("random-legal", "simple-legal", "max-damage"))
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--websocket", default=DEFAULT_LOCAL_WS)
    parser.add_argument("--username", required=True)
    parser.add_argument("--format", dest="battle_format", default=DEFAULT_FORMAT)
    parser.add_argument("--max-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-login", dest="no_login", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--accept", action="store_true", help="accept incoming challenges (default)")
    mode.add_argument("--challenge", metavar="USER", help="challenge a specific user")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    agent = OnlineBattleAgent(
        policy=_build_policy(args.policy, args.showdown_root, args.seed),
        vocab=gen3_category_vocabulary(args.showdown_root),
        dex=load_showdown_dex_cached(args.showdown_root),
        our_name=args.username,
        spec=DEFAULT_REPLAY_OBSERVATION_SPEC,
        rng=random.Random(args.seed),
    )
    client = ShowdownClient(
        agent,
        username=args.username,
        websocket_url=args.websocket,
        battle_format=args.battle_format,
        accept_challenges=not args.challenge,
        challenge_user=args.challenge,
        max_games=None if args.max_games == 0 else args.max_games,
        skip_login=args.no_login,
    )
    asyncio.run(client.run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
