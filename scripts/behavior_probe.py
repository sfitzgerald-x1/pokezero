"""Self-play behavioral-fingerprint probe for foundation checkpoints.

Where ``collapse_probe`` measures collapse *signatures* over a fixed driver-driven state corpus
(no neural self-play), this probe plays actual self-play games with the checkpoint driving BOTH
seats and records how the policy behaves in its own games — the fun-to-watch behavioral data that
shifts as training progresses:

  - move_usage  — distribution over EVERY move the policy actually picks (a proper distribution
                  over move choices; forced replacements are excluded). Watching this reshape over
                  milestones shows the policy's move preferences drift (e.g. setup rising/vanishing,
                  a few attacks crowding out the rest = the peaked-usage face of mode collapse).
  - pivot_rate  — how often the active Pokemon voluntarily switches out BEFORE it faints. A
                  voluntary pivot is a switch chosen while the mon could still attack
                  (request_kind == "move"); a forced replacement is the switch after a faint
                  (request_kind == "force_switch"). pivot_rate = pivots / (pivots + forced), i.e.
                  the fraction of "exits from the field" that were pivots rather than KOs. An
                  aggressive/max-damage-like policy trends toward 0; a defensive pivoting policy
                  trends up. Also reports raw per-game rates.

Games are deterministic given (checkpoint, seed) — the agent is loaded with deterministic=True, so
both seats play argmax and the series is reproducible + comparable across checkpoints.

Note: gen3 has no U-turn/Volt Switch, so voluntary mid-turn switches are genuine pivots; the one
non-faint source of a force_switch request is Baton Pass, which slightly inflates the forced count.
The final KO that ends a game raises no replacement request, so forced_switches undercounts true
faints by ~1/game — a consistent bias that preserves cross-checkpoint comparability.

Usage:
    python scripts/behavior_probe.py \
      --checkpoint checkpoints/pokezero-no-belief-gen3-500k.pt=500k \
      --checkpoint checkpoints/pokezero-no-belief-gen3-1m.pt=1M \
      --showdown-root /path/to/pokemon-showdown \
      --games 40 --out evals/behavior_signals.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pokezero.checkpoint_factors import choice_label
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.online_client import build_agent
from pokezero.showdown import observation_from_player_state

MAX_STEPS = 400


def _self_play_behavior(agent, showdown_root: str, num_games: int, seed_start: int,
                        belief_set_source: bool | None) -> dict:
    """Play `num_games` deterministic self-play games; tally move usage + pivot/faint switches."""
    move_counts: Counter[str] = Counter()
    move_decisions = 0          # decisions where the policy picked a move (proper move-usage base)
    voluntary_pivots = 0        # switch chosen while the active mon could still attack
    forced_switches = 0         # replacement after a faint (~faints, minus the game-ending KO)
    total_turns = 0
    games_played = 0

    for index in range(num_games):
        seed = seed_start + index
        config = LocalShowdownConfig(showdown_root=showdown_root, set_belief_source=belief_set_source)
        env = LocalShowdownEnv(config)
        env.reset(seed=seed)
        last_turn = 0

        for _ in range(MAX_STEPS):
            if env.terminal() is not None:
                break
            requested = env.requested_players()
            actions = {}
            for player in requested:
                state = env._state_for_player(player)
                if state.request is None or state.request_kind in {"wait", "none", "team_preview"}:
                    continue
                if not any(state.legal_action_mask):
                    continue
                obs = observation_from_player_state(
                    state, category_vocab=agent.vocab, spec=agent.spec, dex=agent.dex
                )
                idx = agent.policy.select_action(obs, rng=agent.rng).action_index
                actions[player] = idx
                last_turn = max(last_turn, state.turn_number)

                label = choice_label(state, idx)
                is_switch = label.startswith("switch:")
                if state.request_kind == "force_switch":
                    forced_switches += 1
                elif is_switch:
                    voluntary_pivots += 1
                else:  # a move played on a normal turn
                    move_decisions += 1
                    move_counts[label[len("move:"):] if label.startswith("move:") else label] += 1
            if not actions:
                break
            env.step(actions)

        total_turns += last_turn
        games_played += 1

    exits = voluntary_pivots + forced_switches
    g = games_played or 1
    move_usage = {
        move: round(count / move_decisions, 4)
        for move, count in move_counts.most_common()
    } if move_decisions else {}
    return {
        "games": games_played,
        "move_decisions": move_decisions,
        "distinct_moves": len(move_counts),
        "move_usage": move_usage,
        "avg_turns": round(total_turns / g, 2),
        "voluntary_pivots": voluntary_pivots,
        "forced_switches": forced_switches,
        "pivots_per_game": round(voluntary_pivots / g, 3),
        "forced_switches_per_game": round(forced_switches / g, 3),
        "pivot_rate": round(voluntary_pivots / exits, 4) if exits else 0.0,
    }


def probe_checkpoint(label: str, checkpoint: str, showdown_root: str, num_games: int,
                     seed_start: int, belief_set_source: bool | None) -> dict:
    agent = build_agent(checkpoint, showdown_root, our_name="behavior", deterministic=True)
    row = {"label": label, "checkpoint": checkpoint}
    row.update(_self_play_behavior(agent, showdown_root, num_games, seed_start, belief_set_source))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="PATH[=LABEL]",
                        help="checkpoint to probe; repeatable. Optional =LABEL for display.")
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--games", type=int, default=40, help="self-play games per checkpoint")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--belief-set-source", action="store_true",
                        help="populate candidate-set beliefs in the env (match belief-on checkpoints)")
    parser.add_argument("--top", type=int, default=15, help="top-N moves to print per checkpoint")
    parser.add_argument("--out", default=None, help="write the full result JSON here")
    args = parser.parse_args()

    belief = True if args.belief_set_source else None
    rows = []
    for spec in args.checkpoint:
        path, _, label = spec.partition("=")
        label = label or Path(path).stem
        print(f"[behavior] probing {label} ({args.games} self-play games)…", file=sys.stderr)
        row = probe_checkpoint(label, path, args.showdown_root, args.games, args.seed_start, belief)
        rows.append(row)
        top = list(row["move_usage"].items())[: args.top]
        top_str = ", ".join(f"{m} {p:.0%}" for m, p in top)
        print(f"  pivot_rate={row['pivot_rate']} "
              f"(pivots/g={row['pivots_per_game']} forced/g={row['forced_switches_per_game']}) "
              f"avg_turns={row['avg_turns']} distinct_moves={row['distinct_moves']}", file=sys.stderr)
        print(f"  top moves: {top_str}", file=sys.stderr)

    payload = {"games_per_checkpoint": args.games, "checkpoints": rows}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[behavior] wrote {out}", file=sys.stderr)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
