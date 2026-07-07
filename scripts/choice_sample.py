"""Record each checkpoint's full choice distribution on a fixed set of mid-game states, and
save it in the repo as a committed, diffable behavioral fingerprint.

Samples the decision at turn `--turn` (default 10, for some game depth) from `--num-games`
unique games (deterministic heuristic drivers, so the state set is reproducible and stable as
new checkpoints are added). For every state it records, per checkpoint, the probability the
policy assigns to each legal choice (labeled: move:Surf, switch:Blissey, ...).

  python scripts/choice_sample.py \
      --checkpoint checkpoints/curated/current-v2-500k.pt=v2-500k \
      --checkpoint checkpoints/curated/current-v2-600k.pt=v2-600k \
      --showdown-root /Users/scott/workspace/pokerena/vendor/pokemon-showdown \
      --out evals/turn10_choice_sample.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from pokezero.actions import ACTION_COUNT
from pokezero.checkpoint_factors import (
    active_status,
    choice_label,
    hp_fraction,
    sample_states_at_turn,
)
from pokezero.neural_policy import evaluate_transformer_action_priors
from pokezero.online_client import build_agent
from pokezero.showdown import observation_from_player_state


def _mon_snapshot(mon) -> dict:
    condition = mon.condition
    return {
        "species": mon.species,
        "hp": round(hp_fraction(condition), 3),
        "status": active_status(condition),
        "fainted": "fnt" in str(condition or ""),
        "active": mon.active,
    }


def _team_snapshot(team, active_toxic_stage: int) -> list:
    """Snapshot a team; annotate the active mon with the badly-poisoned counter."""
    mons = [_mon_snapshot(m) for m in team]
    for mon in mons:
        if mon["active"]:
            mon["toxic_stage"] = active_toxic_stage
    return mons


def _active_detail(state) -> dict:
    """Our active mon's full context from the request: moveset (with pp/disabled), item,
    ability, status, and stat changes."""
    request = state.request if isinstance(state.request, Mapping) else {}
    active = request.get("active")
    moves = []
    if isinstance(active, (list, tuple)) and active and isinstance(active[0], Mapping):
        for move in active[0].get("moves") or []:
            if isinstance(move, Mapping):
                moves.append(
                    {
                        "name": move.get("move") or move.get("id"),
                        "pp": move.get("pp"),
                        "maxpp": move.get("maxpp"),
                        "disabled": bool(move.get("disabled")),
                    }
                )
    item = ability = None
    side = request.get("side") if isinstance(request, Mapping) else None
    roster = side.get("pokemon") if isinstance(side, Mapping) else None
    if isinstance(roster, (list, tuple)):
        for mon in roster:
            if isinstance(mon, Mapping) and mon.get("active"):
                item = mon.get("item")
                ability = mon.get("baseAbility") or mon.get("ability")
                break
    active_mon = state.self_active
    return {
        "species": active_mon.species if active_mon is not None else None,
        "item": item,
        "ability": ability,
        "status": active_status(active_mon.condition) if active_mon is not None else "none",
        "toxic_stage": state.self_toxic_stage,
        "boosts": {k: v for k, v in state.self_active_boosts.items() if v},
        "moves": moves,
    }


def _state_snapshot(state) -> dict:
    """Board context for reading the plot: both teams (HP/status), active-mon detail, field."""
    return {
        "turn": state.turn_number,
        "weather": state.weather,
        "self_side_conditions": list(state.self_side_conditions),
        "opponent_side_conditions": list(state.opponent_side_conditions),
        "opponent_boosts": {k: v for k, v in state.opponent_active_boosts.items() if v},
        "active_detail": _active_detail(state),
        "self_team": _team_snapshot(state.self_team, state.self_toxic_stage),
        "opponent_team": _team_snapshot(state.opponent_team, state.opponent_toxic_stage),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="PATH[=LABEL]")
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--num-games", type=int, default=25)
    parser.add_argument("--turn", type=int, default=10)
    parser.add_argument("--player", default="p1")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--out", default="evals/turn10_choice_sample.json")
    parser.add_argument(
        "--allow-legacy-checkpoints",
        action="store_true",
        help=(
            "allow no-belief/pre-v2 checkpoints for explicit historical reproduction; "
            "do not use for current longitudinal evals"
        ),
    )
    args = parser.parse_args()

    specs = []
    for raw in args.checkpoint:
        path, _, label = raw.partition("=")
        specs.append((label or Path(path).stem, path))
    if not args.allow_legacy_checkpoints:
        try:
            from pokezero.opponents import require_current_family_checkpoint_paths
        except ImportError:
            print("[warn] current-family gate unavailable in this image; skipping schema check "
                  "(the deployed image is the checkpoint's own, so schema matches by construction)",
                  file=sys.stderr)
        else:
            require_current_family_checkpoint_paths(
                (path for _, path in specs),
                context="choice-sample probe",
            )

    # Build agents up front (cached) so state capture can use the checkpoint's own observation
    # spec. This matters for v2.2 (turn-merged) runs: the capture env must populate
    # state.turn_merged_tokens, which the v2.2 encode requires. All checkpoints in one invocation
    # share a schema family (enforced by the current-family gate above), so the first checkpoint's
    # spec drives capture.
    agents: dict[str, object] = {}

    def _agent(path: str):
        if path not in agents:
            agents[path] = build_agent(path, args.showdown_root, our_name="sample", deterministic=True)
        return agents[path]

    capture_spec = _agent(specs[0][1]).spec
    print(f"[sample] collecting turn-{args.turn} {args.player} states from {args.num_games} unique "
          f"games (schema {capture_spec.schema_version})…")
    states = sample_states_at_turn(
        args.showdown_root,
        num_games=args.num_games,
        turn=args.turn,
        player=args.player,
        seed_start=args.seed_start,
        observation_spec=capture_spec,
    )
    print(f"[sample] captured {len(states)} states (seeds {[s for _, s in states]})")

    # Per-state legal choices + context (checkpoint-independent).
    records = []
    for state, seed in states:
        legal = [i for i in range(ACTION_COUNT) if state.legal_action_mask[i]]
        opponent = state.opponent_active
        records.append(
            {
                "seed": seed,
                "turn": state.turn_number,
                "player": args.player,
                "active": state.self_active.species,
                "active_condition": state.self_active.condition,
                "opponent_active": opponent.species if opponent is not None else None,
                "legal_choices": [{"index": i, "label": choice_label(state, i)} for i in legal],
                "state": _state_snapshot(state),
                "checkpoints": {},
            }
        )

    for label, path in specs:
        print(f"[sample] scoring {label}…")
        agent = _agent(path)
        for record, (state, _) in zip(records, states):
            obs = observation_from_player_state(
                state, category_vocab=agent.vocab, spec=agent.spec, dex=agent.dex,
                **({"feature_masks": agent.feature_masks} if agent.feature_masks is not None else {}),
            )
            probs = evaluate_transformer_action_priors(
                model=agent.policy.model, result=agent.policy.result, observations=[obs]
            )
            record["checkpoints"][label] = {
                choice["label"]: round(probs[choice["index"]], 4) for choice in record["legal_choices"]
            }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Per-checkpoint action-probability fingerprint on a fixed set of mid-game decision "
            "states. Sampled at the given turn from unique games via deterministic heuristic "
            "drivers, so the state set is stable and new checkpoints can be appended for comparison."
        ),
        "turn": args.turn,
        "player": args.player,
        "num_games": args.num_games,
        "seed_start": args.seed_start,
        "checkpoints": [label for label, _ in specs],
        "states": records,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[out] wrote {out_path} ({len(records)} states x {len(specs)} checkpoints)")

    # Console preview: top choice per checkpoint on the first few states.
    print("\n=== preview (top choice per checkpoint) ===")
    for record in records[:6]:
        print(f"  seed {record['seed']} t{record['turn']}: {record['active']} vs {record['opponent_active']}")
        for label, _ in specs:
            dist = record["checkpoints"][label]
            top = max(dist.items(), key=lambda kv: kv[1])
            print(f"    {label:<6} -> {top[0]} ({top[1]:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
