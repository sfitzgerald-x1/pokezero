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
    """Living count + team HP fraction for the acting side, from public state."""
    team = getattr(state, "self_team", None) or getattr(state, "team", None) or []
    alive, total, current = 0, 0.0, 0.0
    for mon in team:
        hp = float(getattr(mon, "current_hp", 0) or 0)
        maxhp = float(getattr(mon, "max_hp", 0) or 0) or 1.0
        if hp > 0:
            alive += 1
        current += hp
        total += maxhp
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
        try:
            observation = env.reset(seed=seed)
        except TypeError:
            observation = env.reset()
        rng = random.Random(seed)
        turn = 0
        prefix: list[str] = []
        while turn < args.max_decision_rounds:
            mask = tuple(bool(v) for v in observation.legal_action_mask)
            if not any(mask):
                break
            candidates = tuple(
                dict(c) if isinstance(c, dict) else {"value": str(c)}
                for c in (observation.metadata or {}).get("action_candidates", ()) or ()
            )
            if candidates:
                remaining, hp_fraction = _remaining_and_hp(observation)
                forced = not any(mask[:4])  # no move slot legal -> forced switch
                records.append(
                    TimingDecisionRecord(
                        decision_id=f"s{seed:07d}-t{turn:03d}",
                        battle_id=f"corpus-{seed}",
                        seat="p1",
                        turn_index=turn,
                        team_seed=seed,
                        battle_seed=seed,
                        bot_rng_seed=seed,
                        event_prefix=tuple(prefix[-400:]),
                        action_candidates=candidates,
                        legal_action_mask=mask,
                        public_belief_inputs={"turn": turn},
                        strata=label_strata(
                            remaining=remaining,
                            team_hp_fraction=hp_fraction,
                            boosts=(observation.metadata or {}).get("boosts"),
                            forced_switch=forced,
                            hidden_world_count=1,
                            turn_index=turn,
                        ),
                    )
                )
            decision = policy.select_action(observation, rng=rng)
            step = env.step(decision.action_index)
            observation = step[0] if isinstance(step, tuple) else step
            done = bool(getattr(step[2] if isinstance(step, tuple) and len(step) > 2 else step, "__bool__", lambda: False)())
            prefix.extend(getattr(env, "last_lines", ()) or ())
            turn += 1
            if done:
                break
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
