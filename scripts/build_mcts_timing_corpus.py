#!/usr/bin/env python3
"""Build ``pokezero.engine-mcts-timing-corpus.v3`` from held-out games (plan A2).

Each record carries the acting seat's public protocol prefix through its request,
the canonical public action identifiers needed to replay it,
the REQUEST-DERIVED action candidates and legal mask (the field
``public-decision-corpus.v1`` lacks, and the reason the plan forbids reusing
it), the seeds needed to reproduce the game, and the public belief inputs.

Games are played from a held-out seed band with the study checkpoint, so the
decisions are drawn from the distribution the timing lattice will be asked
about. Every record is labeled on all seven strata axes from public state.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from pokezero.mcts_eval.timing_corpus import (  # noqa: E402
    CorpusError,
    TimingDecisionRecord,
    build_corpus,
    label_strata,
    validate_representative_timing_panel,
    write_corpus,
)
from pokezero.public_action_capture import public_action_round_from_protocol_lines  # noqa: E402
from pokezero.public_replay_materializer import (  # noqa: E402
    PublicReplayError,
    public_event_prefix_summary,
    replay_public_action_rounds,
)


def _remaining_and_hp(state: Any) -> tuple[int, float]:
    """Living count + team HP fraction from PUBLIC state.

    Showdown reports HP in ``condition``: "cur/max", "cur/max sta", or "0 fnt".
    Reading it (rather than guessing at numeric fields) is what makes the strata
    labels real — a silent zero here collapses every record into one bucket.
    """
    alive, current, total = 0, 0.0, 0.0
    for mon in getattr(state, "self_team", ()) or ():
        condition = (getattr(mon, "condition", None) or "").strip()
        head = condition.split(" ")[0] if condition else ""
        if "/" in head:
            cur_text, _, max_text = head.partition("/")
            try:
                cur, mx = float(cur_text), float(max_text)
            except ValueError:
                continue
        elif head in {"0", ""}:
            cur, mx = 0.0, 1.0
        else:
            continue
        mx = mx or 1.0
        current += cur
        total += mx
        if cur > 0 and not condition.endswith("fnt"):
            alive += 1
    return (alive or 1), (current / total if total else 1.0)


def _prefix_replayability_reason(
    env: Any,
    *,
    seed: int,
    public_rounds: tuple[Any, ...],
    expected_public_lines: dict[str, tuple[str, ...]],
) -> str | None:
    """Return a named rejection unless the public prefix replays identically.

    Request-local action indexes are intentionally absent from the corpus. A
    parseable public action identifier is therefore not sufficient: it must
    also reproduce the public protocol and future request shape in a fresh
    MCTS-shaped world.
    """

    try:
        replay_public_action_rounds(
            env,
            seed=seed,
            format_id="gen3randombattle",
            public_action_rounds=public_rounds,
            start_override=None,
        )
    except PublicReplayError as error:
        return f"public_replay:{error.reason}"
    for player, expected in expected_public_lines.items():
        actual = tuple(env.public_materialization_state(player).replay.public_lines)
        if actual != expected:
            return "public_replay:public_prefix_mismatch"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--showdown-root", default=os.environ.get("POKEZERO_SHOWDOWN_ROOT"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--seed-start", type=int, default=900_000)
    parser.add_argument("--decisions", type=int, default=256)
    parser.add_argument("--max-decision-rounds", type=int, default=250)
    args = parser.parse_args(argv)

    from pokezero.collection import policy_from_spec
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
    from pokezero.collection import env_config_with_policy_spec_masks

    spec = f"neural:{args.checkpoint}"
    # The corpus must use the exact belief-aware observation path consumed by
    # the timing lattice. Otherwise a replayable collection record can still
    # fail in the model-backed MCTS adapter.
    env_config = env_config_with_policy_spec_masks(
        LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True),
        [spec],
        context="timing corpus",
    )
    policy = policy_from_spec(spec)

    records: list[TimingDecisionRecord] = []
    games_played = 0
    replay_rejections: dict[str, int] = {}
    import random

    for offset in range(args.games):
        seed = args.seed_start + offset
        env = LocalShowdownEnv(env_config)
        env.reset(seed=seed)
        rngs = {"p1": random.Random(seed * 2 + 1), "p2": random.Random(seed * 2 + 2)}
        public_rounds = []
        turn = 0
        while turn < args.max_decision_rounds and env.terminal() is None:
            requested = env.requested_players()
            expected_public_lines = {
                player: tuple(env.public_materialization_state(player).replay.public_lines)
                for player in requested
            }
            probe = LocalShowdownEnv(env_config)
            try:
                replay_reason = _prefix_replayability_reason(
                    probe,
                    seed=seed,
                    public_rounds=tuple(public_rounds),
                    expected_public_lines=expected_public_lines,
                )
            finally:
                probe.close()
            if replay_reason is not None:
                replay_rejections[replay_reason] = replay_rejections.get(replay_reason, 0) + 1
                print(
                    f"seed {seed}: excluding unreplayable timing prefix at turn {turn}: {replay_reason}",
                    flush=True,
                )
            actions: dict[str, int] = {}
            for player in ("p1", "p2"):
                if player not in requested:
                    continue
                observation = env.observe(player)
                mask = tuple(bool(v) for v in observation.legal_action_mask)
                if not any(mask):
                    continue
                decision = policy.select_action(observation, rng=rngs[player])
                actions[player] = decision.action_index
                candidates = tuple(
                    dict(c) if isinstance(c, dict) else {"value": str(c)}
                    for c in (getattr(observation, "metadata", None) or {}).get("action_candidates", ()) or ()
                )
                if not candidates or replay_reason is not None:
                    # Continue the held-out game so later prefixes remain
                    # available, but do not admit a decision whose public
                    # reconstruction already diverged.
                    continue
                state = env._state_for_player(player)
                remaining, hp_fraction = _remaining_and_hp(state)
                records.append(
                    TimingDecisionRecord(
                        decision_id=f"s{seed:07d}-t{turn:03d}-{player}",
                        battle_id=f"corpus-{seed}",
                        seat=player,
                        turn_index=turn,
                        team_seed=seed,
                        battle_seed=seed,
                        bot_rng_seed=seed * 2 + (1 if player == "p1" else 2),
                        # ``protocol_lines`` also contains both players' raw request JSON.
                        # The replay snapshot projects those down to public protocol facts,
                        # so retaining it is both the privacy boundary and the exact input the
                        # live fold will consume when the timing harness reconstructs this turn.
                        event_prefix=tuple(
                            env.public_materialization_state(player).replay.public_lines
                        ),
                        public_resolved_action_rounds=tuple(public_rounds),
                        action_candidates=candidates,
                        legal_action_mask=mask,
                        public_belief_inputs={"turn": turn, "request_kind": str(getattr(state, "request_kind", ""))},
                        strata=label_strata(
                            remaining=remaining,
                            team_hp_fraction=hp_fraction,
                            boosts=getattr(state, "self_active_boosts", None),
                            forced_switch=str(getattr(state, "request_kind", "")) == "forceSwitch",
                            hidden_world_count=1,
                            turn_index=turn,
                            legal_action_count=sum(mask),
                        ),
                    )
                )
            if not actions:
                break
            protocol_line_count = len(env.protocol_lines)
            env.step(actions)
            action_round = public_action_round_from_protocol_lines(
                env.protocol_lines[protocol_line_count:],
                turn_index=turn,
                requested_players=requested,
            )
            event_summary = public_event_prefix_summary((action_round,))
            if event_summary["unsupported_public_event_count"]:
                print(
                    f"seed {seed}: stopping unreplayable prefix at turn {turn}: "
                    f"{event_summary['unsupported_public_event_ids']}",
                    flush=True,
                )
                break
            public_rounds.append(action_round)
            turn += 1
        print(f"seed {seed}: {len(records)} decisions so far", flush=True)
        games_played = offset + 1
        # A raw count is not enough: one long battle can fill the candidate
        # pool while omitting a seat, late prefix, or narrow legal mask.  Check
        # the exact deterministic selection before ending collection early.
        if len(records) >= args.decisions * 2:
            try:
                _, candidate = build_corpus(
                    records,
                    held_out_seed_start=args.seed_start,
                    held_out_seed_end=args.seed_start + games_played,
                    count=args.decisions,
                )
                validate_representative_timing_panel(candidate)
            except CorpusError as error:  # coverage is retried with the next held-out game
                print(f"seed {seed}: corpus coverage incomplete ({error})", flush=True)
            else:
                break

    manifest, selected = build_corpus(
        records,
        held_out_seed_start=args.seed_start,
        held_out_seed_end=args.seed_start + games_played,
        count=min(args.decisions, len(records)),
    )
    coverage = validate_representative_timing_panel(selected)
    write_corpus(args.out, manifest, selected)
    print(json.dumps({"decisions": manifest.decision_count, "sha256": manifest.corpus_sha256[:16],
                      "buckets": {k: v for k, v in manifest.bucket_counts.items() if v},
                      "representativeness": coverage,
                      "replay_rejections": dict(sorted(replay_rejections.items()))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
