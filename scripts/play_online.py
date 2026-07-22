#!/usr/bin/env python3
"""Run a PokeZero checkpoint as a bot on a live Pokemon Showdown server.

Development (local server): start one in the vendored checkout, then point the bot at it:

    # terminal 1 — a local server on :8000
    cd /path/to/pokemon-showdown && node pokemon-showdown start --no-security

    # terminal 2 — the bot accepts a challenge; then in a browser client (localhost:8000),
    # log in as another name and challenge "PokeZeroBot" to a gen3randombattle
    PYTHONPATH=src python scripts/play_online.py \
        --showdown-root /path/to/pokemon-showdown \
        --websocket ws://localhost:8000/showdown/websocket \
        --username PokeZeroBot --accept

In the wild (official server) requires a registered account with battle permission and
respecting Showdown's automation rules:

    PYTHONPATH=src python scripts/play_online.py \
        --showdown-root /path/to/pokemon-showdown \
        --websocket wss://sim3.psim.us/showdown/websocket \
        --username <account> --password-env PS_PASSWORD --search --max-games 5
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    import sys

    sys.path.insert(0, str(_SRC))

from pokezero.online_client import (  # noqa: E402
    DEFAULT_FORMAT,
    DEFAULT_LOCAL_WS,
    DEFAULT_LOGIN_URL,
    run_online,
)

_DEFAULT_CHECKPOINT = "runs/promoted/strongest-vs-max-damage.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play a PokeZero checkpoint on a Showdown server.")
    parser.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT, help="Transformer checkpoint to play.")
    parser.add_argument(
        "--showdown-root",
        default=os.environ.get("POKEZERO_SHOWDOWN_ROOT"),
        help="Built Pokemon Showdown checkout (for the dex/vocab); or set POKEZERO_SHOWDOWN_ROOT.",
    )
    parser.add_argument("--websocket", default=DEFAULT_LOCAL_WS, help="Server websocket URL.")
    parser.add_argument("--login-url", default=DEFAULT_LOGIN_URL, help="Login (action.php) URL.")
    parser.add_argument("--username", required=True, help="Bot display name / account.")
    parser.add_argument("--password-env", help="Env var holding the account password (registered names).")
    parser.add_argument("--format", dest="battle_format", default=DEFAULT_FORMAT, help="Battle format id.")
    parser.add_argument("--seed", type=int, default=None, help="Policy RNG seed.")
    parser.add_argument("--sample", action="store_true", help="Sample moves (default: greedy).")
    parser.add_argument("--max-games", type=int, default=1, help="Disconnect after N games (0 = unlimited).")
    parser.add_argument(
        "--history-mask-k",
        type=int,
        default=None,
        help=(
            "History-truncation probe (docs/history_truncation_probe_plan.md): mask the "
            "checkpoint's transition-history region to the most-recent K tokens at decision "
            "time. Eval-only; omit for full (128) history."
        ),
    )
    parser.add_argument(
        "--no-login",
        action="store_true",
        help="Skip the login assertion (for a local server started with --no-security).",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--accept", action="store_true", help="Accept incoming challenges (default).")
    mode.add_argument("--search", action="store_true", help="Search the ladder.")
    mode.add_argument("--challenge", metavar="USER", help="Challenge a specific user.")

    args = parser.parse_args(argv)
    if not args.showdown_root:
        parser.error("--showdown-root is required (or set POKEZERO_SHOWDOWN_ROOT).")
    password = os.environ.get(args.password_env) if args.password_env else None

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(
        run_online(
            checkpoint=args.checkpoint,
            showdown_root=args.showdown_root,
            username=args.username,
            password=password,
            websocket_url=args.websocket,
            login_url=args.login_url,
            battle_format=args.battle_format,
            accept_challenges=not (args.search or args.challenge),
            search_ladder=args.search,
            challenge_user=args.challenge,
            max_games=None if args.max_games == 0 else args.max_games,
            deterministic=not args.sample,
            seed=args.seed,
            skip_login=args.no_login,
            history_mask_k=args.history_mask_k,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
