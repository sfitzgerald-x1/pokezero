"""Encode foul-play mirror capture transcripts into RolloutRecord JSONL for behavior cloning.

For each captured seat, replay its protocol up to every decision, build our observation, decode
the /choose it played into an action index, and emit a per-game BattleTrajectory. Both mirror
seats are teacher labels, so both capture files contribute. Output feeds
`train_transformer_policy(..., objective="behavior-cloning")` directly.

  python scripts/encode_foulplay_capture.py \
      --showdown-root /path/to/pokemon-showdown \
      --capture /tmp/fpmirror/capA.jsonl=FoulPlayA \
      --capture /tmp/fpmirror/capB.jsonl=FoulPlayB \
      --out data/foulplay_bc/games.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero.dex import load_showdown_dex_cached
from pokezero.env import TerminalState
from pokezero.randbat import load_gen3_randbat_source_cached
from pokezero.randbat_vocab import gen3_category_vocabulary
from pokezero.showdown import (
    DEFAULT_REPLAY_OBSERVATION_SPEC,
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
)
from pokezero.teacher_capture import action_index_from_choice_string, parse_capture_transcript
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


def _encode_capture(path, username, *, vocab, dex, spec, stats, set_source=None):
    records = []
    for game in parse_capture_transcript(path):
        if not game.decisions:
            continue
        trajectory = BattleTrajectory(
            battle_id=f"{game.room}:{username}", format_id="gen3randombattle", seed=0
        )
        steps = []
        last_perspective = None
        for counter, decision in enumerate(game.decisions):
            stats["total"] += 1
            try:
                replay = parse_showdown_replay(decision.protocol_lines, battle_id=decision.room)
                state = normalize_for_player(
                    replay,
                    player_id="bot",
                    player_name=username,
                    format_id="gen3randombattle",
                    set_source=set_source,
                )
            except ValueError:
                stats["parse_err"] += 1
                continue
            action_index = action_index_from_choice_string(state, decision.choice)
            if action_index is None:
                stats["undecoded"] += 1
                continue
            observation = observation_from_player_state(
                state, category_vocab=vocab, spec=spec, dex=dex
            )
            if not observation.legal_action_mask[action_index]:
                stats["illegal"] += 1
                continue
            last_perspective = observation.perspective
            steps.append(
                TrajectoryStep(
                    player_id=observation.perspective.showdown_slot,
                    turn_index=counter,
                    observation=observation,
                    legal_action_mask=tuple(observation.legal_action_mask),
                    action_index=action_index,
                    reward=0.0,
                    metadata={"policy_id": "foul-play"},
                )
            )
            stats["encoded"] += 1

        if not steps or last_perspective is None:
            continue
        # Terminal reward from this seat's perspective: +1 win / -1 loss / 0 tie or unknown.
        won = game.winner == username
        lost = game.winner is not None and not won
        outcome = 1.0 if won else (-1.0 if lost else 0.0)
        steps[-1] = _with_reward(steps[-1], outcome)
        winner_slot = (
            last_perspective.showdown_slot if won
            else last_perspective.opponent_showdown_slot if lost
            else None
        )
        for step in steps:
            trajectory.append(step)
        terminal = TerminalState(winner=winner_slot, turn_count=steps[-1].turn_index + 1, capped=False)
        trajectory.record_terminal(terminal)
        records.append(
            RolloutRecord(
                battle_id=trajectory.battle_id,
                seed=0,
                format_id="gen3randombattle",
                policy_ids={last_perspective.showdown_slot: "foul-play"},
                decision_round_count=len(steps),
                elapsed_seconds=0.0,
                terminal=terminal,
                trajectory=trajectory,
            )
        )
        stats["games"] += 1
    return records


def _with_reward(step: TrajectoryStep, reward: float) -> TrajectoryStep:
    return TrajectoryStep(
        player_id=step.player_id,
        turn_index=step.turn_index,
        observation=step.observation,
        legal_action_mask=step.legal_action_mask,
        action_index=step.action_index,
        reward=reward,
        opponent_action_index=step.opponent_action_index,
        action_probability=step.action_probability,
        value_estimate=step.value_estimate,
        metadata=step.metadata,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--capture", action="append", required=True, metavar="PATH=USERNAME")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    vocab = gen3_category_vocabulary(args.showdown_root)
    dex = load_showdown_dex_cached(args.showdown_root)
    spec = DEFAULT_REPLAY_OBSERVATION_SPEC
    stats = {"total": 0, "encoded": 0, "undecoded": 0, "illegal": 0, "parse_err": 0, "games": 0}

    # Match the self-play encoder: when POKEZERO_BELIEF_SET_SOURCE is enabled, encode opponent
    # candidate-set beliefs (possible_moves/candidate_set_count) so the BC clone trains on the same
    # observation the belief-on self-play line uses. Off by default -> belief-blind (legacy) data.
    set_source = None
    if os.environ.get("POKEZERO_BELIEF_SET_SOURCE", "0").strip().lower() in {"1", "true", "yes", "on"}:
        set_source = load_gen3_randbat_source_cached(args.showdown_root)
        print("[encode] belief set source ENABLED (candidate-set beliefs populated)")

    all_records = []
    for spec_str in args.capture:
        path, _, username = spec_str.partition("=")
        if not username:
            raise SystemExit(f"--capture needs PATH=USERNAME, got {spec_str!r}")
        print(f"[encode] {path} (seat {username})…")
        all_records.extend(
            _encode_capture(path, username, vocab=vocab, dex=dex, spec=spec, stats=stats, set_source=set_source)
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        for record in all_records:
            write_rollout_record(handle, record)

    print(
        f"[encode] games={stats['games']} decisions={stats['total']} "
        f"encoded={stats['encoded']} undecoded={stats['undecoded']} "
        f"illegal={stats['illegal']} parse_err={stats['parse_err']}"
    )
    print(f"[out] wrote {out_path} ({len(all_records)} trajectories)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
