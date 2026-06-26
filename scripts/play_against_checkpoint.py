#!/usr/bin/env python3
"""Play a Gen 3 random battle against a trained PokeZero checkpoint, in your terminal.

    PYTHONPATH=src python scripts/play_against_checkpoint.py \
        --showdown-root /path/to/pokemon-showdown

Defaults to the promoted "strongest vs max-damage" checkpoint. You pick a move or switch
each turn from a numbered menu; the checkpoint plays the other side. Needs a built Pokemon
Showdown checkout (the same one self-play uses) for the local battle bridge.
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import replace
from pathlib import Path

# Allow running as `python scripts/play_against_checkpoint.py` without PYTHONPATH=src.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    import sys

    sys.path.insert(0, str(_SRC))

from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv  # noqa: E402
from pokezero.neural_policy import load_transformer_policy  # noqa: E402
from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC  # noqa: E402

_DEFAULT_CHECKPOINT = "runs/promoted/strongest-vs-max-damage.pt"


def _pct(fraction: float | None) -> str:
    return f"{round((fraction or 0.0) * 100)}% HP"


def _status(mon: dict) -> str:
    status = mon.get("status")
    return "" if not status or status == "none" else f" [{status}]"


def _render(observation, human: str) -> None:
    md = observation.metadata
    me, opp = md.get("self_active") or {}, md.get("opponent_active") or {}
    print("\n" + "=" * 64)
    header = f"Turn {md.get('turn_number', 0)}"
    weather = md.get("weather")
    if weather:
        header += f"   weather: {weather}"
    print(header)
    print(f"  Opponent  {opp.get('species', '?'):<14} {_pct(opp.get('hp_fraction'))}{_status(opp)}")
    print(f"  You       {me.get('species', '?'):<14} {_pct(me.get('hp_fraction'))}{_status(me)}")
    events = []
    for line in md.get("recent_public_events") or []:
        cleaned = " ".join(part for part in line.split("|") if part).strip()
        if cleaned and not cleaned.startswith(("turn", "upkeep")):
            events.append(cleaned)
    for line in events[-3:]:
        print(f"    · {line}")
    print("-" * 64)


def _action_label(candidate: dict) -> str:
    if candidate.get("kind") == "move":
        return f"move   {candidate.get('move_name') or candidate.get('move_id')}"
    mon = candidate.get("pokemon") or {}
    return f"switch {mon.get('species', '?'):<12} ({_pct(mon.get('hp_fraction'))}{_status(mon)})"


def _prompt_human_action(observation) -> int:
    legal = [c for c in observation.metadata.get("action_candidates", []) if c.get("legal")]
    for index, candidate in enumerate(legal, start=1):
        print(f"   {index}) {_action_label(candidate)}")
    while True:
        try:
            raw = input("Your choice (number, or 'q' to quit): ").strip().lower()
        except EOFError as exc:  # piped input exhausted
            raise KeyboardInterrupt from exc
        if raw in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if raw.isdigit() and 1 <= int(raw) <= len(legal):
            return int(legal[int(raw) - 1]["action_index"])
        print(f"   Please enter 1-{len(legal)}.")


def _bot_action(policy, observation, rng: random.Random) -> int:
    return policy.select_action(observation, rng=rng).action_index


def play(
    *,
    checkpoint: str,
    showdown_root: str | None,
    seed: int,
    human_player: str,
    deterministic: bool,
) -> None:
    policy = load_transformer_policy(checkpoint, deterministic=deterministic)
    config = policy.result.model_config
    # Feed the model the observation shape it was trained on (it may predate later feature slots).
    spec = replace(
        DEFAULT_REPLAY_OBSERVATION_SPEC,
        categorical_feature_count=config.categorical_feature_count,
        numeric_feature_count=config.numeric_feature_count,
    )
    bot_player = "p1" if human_player == "p2" else "p2"
    env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root, observation_spec=spec))
    rng = random.Random(seed)

    print(f"Gen 3 random battle — you are {human_player} vs {config.policy_id} ({bot_player}).")
    env.reset(seed=seed)
    try:
        requested = env.requested_players()
        observations = {player: env.observe(player) for player in requested}
        terminal = env.terminal()
        while terminal is None and requested:
            actions: dict[str, int] = {}
            for player in requested:
                observation = observations.get(player) or env.observe(player)
                if player == human_player:
                    _render(observation, human_player)
                    actions[player] = _prompt_human_action(observation)
                else:
                    actions[player] = _bot_action(policy, observation, rng)
            result = env.step(actions)
            requested = result.requested_players
            observations = result.observations
            terminal = result.terminal
    except KeyboardInterrupt:
        print("\nGG — quit.")
        env.close()
        return

    if terminal is None:
        print("\nBattle ended without a result.")
    elif terminal.winner is None:
        print("\nIt's a tie!")
    elif terminal.winner == human_player:
        print(f"\nYou won! (turn {terminal.turn_count})")
    else:
        print(f"\nThe checkpoint won. (turn {terminal.turn_count})")
    env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play a Gen 3 random battle against a PokeZero checkpoint.")
    parser.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT, help="Transformer checkpoint to play against.")
    parser.add_argument(
        "--showdown-root",
        default=os.environ.get("POKEZERO_SHOWDOWN_ROOT"),
        help="Built Pokemon Showdown checkout root (or set POKEZERO_SHOWDOWN_ROOT).",
    )
    parser.add_argument("--seed", type=int, default=random.randint(1, 10_000_000), help="Battle seed.")
    parser.add_argument("--player", choices=("p1", "p2"), default="p2", help="Which side you play.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Let the checkpoint sample its moves (default: greedy / strongest play).",
    )
    args = parser.parse_args(argv)
    if not args.showdown_root:
        parser.error("--showdown-root is required (or set POKEZERO_SHOWDOWN_ROOT).")
    play(
        checkpoint=args.checkpoint,
        showdown_root=args.showdown_root,
        seed=args.seed,
        human_player=args.player,
        deterministic=not args.sample,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
