#!/usr/bin/env python3
"""Build ``pokezero.engine-mcts-timing-corpus.v1`` from held-out games (plan A2).

Each record carries the acting seat's public event prefix through its request,
the REQUEST-DERIVED action candidates and legal mask (the field
``public-decision-corpus.v1`` lacks, and the reason the plan forbids reusing
it), the seeds needed to reproduce the game, and the public belief inputs.

Games are played from a held-out seed band with the study checkpoint, so the
decisions are drawn from the distribution the timing lattice will be asked
about. Every record is labeled on all six strata axes from public state.
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
    TimingDecisionRecord,
    build_corpus,
    label_strata,
    write_corpus,
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
    env_config = env_config_with_policy_spec_masks(
        LocalShowdownConfig(showdown_root=args.showdown_root), [spec], context="timing corpus"
    )
    policy = policy_from_spec(spec)

    records: list[TimingDecisionRecord] = []
    import random

    for offset in range(args.games):
        seed = args.seed_start + offset
        env = LocalShowdownEnv(env_config)
        env.reset(seed=seed)
        rngs = {"p1": random.Random(seed * 2 + 1), "p2": random.Random(seed * 2 + 2)}
        turn = 0
        while turn < args.max_decision_rounds and env.terminal() is None:
            requested = env.requested_players()
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
                if player != "p1":
                    continue
                candidates = tuple(
                    dict(c) if isinstance(c, dict) else {"value": str(c)}
                    for c in (getattr(observation, "metadata", None) or {}).get("action_candidates", ()) or ()
                )
                if not candidates:
                    continue
                state = env._state_for_player(player)
                remaining, hp_fraction = _remaining_and_hp(state)
                records.append(
                    TimingDecisionRecord(
                        decision_id=f"s{seed:07d}-t{turn:03d}",
                        battle_id=f"corpus-{seed}",
                        seat=player,
                        turn_index=turn,
                        team_seed=seed,
                        battle_seed=seed,
                        bot_rng_seed=seed,
                        event_prefix=tuple(str(line) for line in (env.public_log() if hasattr(env, "public_log") else [])[-400:]),
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
                        ),
                    )
                )
            if not actions:
                break
            env.step(actions)
            turn += 1
        print(f"seed {seed}: {len(records)} decisions so far", flush=True)
        if len(records) >= args.decisions * 2:
            break

    manifest, selected = build_corpus(
        records,
        held_out_seed_start=args.seed_start,
        held_out_seed_end=args.seed_start + args.games,
        count=min(args.decisions, len(records)),
    )
    write_corpus(args.out, manifest, selected)
    print(json.dumps({"decisions": manifest.decision_count, "sha256": manifest.corpus_sha256[:16],
                      "buckets": {k: v for k, v in manifest.bucket_counts.items() if v}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
