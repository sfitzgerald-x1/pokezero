#!/usr/bin/env python
"""Replay exact differential boundaries and attest Python-to-Rust damage inputs.

This intentionally shares the differential's deterministic action selection and
world-construction path.  It is an evidence tool, not a second simulator: each
target's result contains the full stored-stat and active-stage comparison at the
native state passed to branch generation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from engine_build_fingerprint import assert_fresh  # noqa: E402
from engine_transition_differential import (  # noqa: E402
    _prepare_boundary,
    _true_teams_from_bridge_snapshot,
    unpack_team,
)
from pokezero.dex import load_showdown_dex  # noqa: E402
from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy  # noqa: E402
from pokezero.engine_stat_attestation import attest_damage_stat_inputs  # noqa: E402
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT, LocalShowdownConfig, LocalShowdownEnv  # noqa: E402
from pokezero.randbat import Gen3RandbatSource  # noqa: E402


def _target(value: str) -> tuple[int, int]:
    seed, separator, step = value.partition("/")
    if not separator or not seed.isdigit() or not step.isdigit():
        raise argparse.ArgumentTypeError("targets must be SEED/STEP")
    return int(seed), int(step)


def attest_target(
    *,
    env: LocalShowdownEnv,
    policy: EngineMctsPolicy,
    dex: Any,
    seed: int,
    target_step: int,
    max_steps: int,
) -> dict[str, object]:
    """Recreate one differential boundary and return its construction evidence."""

    env.reset(seed=seed, format_id="gen3randombattle")
    true_teams = _true_teams_from_bridge_snapshot(env.snapshot().bridge_snapshot)
    packed = {slot: true_teams[slot]["packed"] for slot in ("p1", "p2")}
    override = BattleStartOverride(player_teams=packed)
    teams = {slot: unpack_team(packed[slot]) for slot in ("p1", "p2")}
    rng = random.Random(seed ^ 0x5EED)
    cumulative = list(env.protocol_lines)

    for step in range(1, max_steps + 1):
        if env.terminal() is not None:
            return {
                "seed": seed,
                "step": target_step,
                "status": "terminal_before_target",
                "terminal_at_or_before_step": step - 1,
            }
        requested = tuple(env.requested_players())
        actions: dict[str, int] = {}
        for player in requested:
            legal = [index for index, allowed in enumerate(env.legal_actions(player)) if allowed]
            if not legal:
                return {"seed": seed, "step": target_step, "status": "no_legal_action"}
            actions[player] = rng.choice(legal)

        prepared = None
        if set(requested) == {"p1", "p2"}:
            prepared = _prepare_boundary(
                env=env,
                flags_policy=policy,
                override=override,
                teams=teams,
                dex=dex,
                actions=actions,
                cumulative=cumulative,
                counts=Counter(),
                approximate_sleep=False,
                hidden_counter_support=True,
            )
        if step == target_step:
            if prepared is None:
                return {
                    "seed": seed,
                    "step": target_step,
                    "status": "unmaterializable_target",
                    "requested_players": list(requested),
                }
            attestations = [
                attest_damage_stat_inputs(spec, state).to_dict()
                for spec, state in zip(prepared["specs"], prepared["states"], strict=True)
            ]
            return {
                "seed": seed,
                "step": target_step,
                "status": "attested",
                "turn": prepared["turn"],
                "gating": prepared["gating"],
                "attestations": attestations,
            }

        before = len(cumulative)
        env.step(actions)
        cumulative.extend(str(line) for line in env.protocol_lines[before:])
    return {"seed": seed, "step": target_step, "status": "max_steps_exceeded"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--showdown-root", default=DEFAULT_SHOWDOWN_ROOT)
    parser.add_argument("--target", type=_target, action="append", required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    assert_fresh()

    dex = load_showdown_dex(args.showdown_root)
    env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True))
    policy = EngineMctsPolicy(
        dex=dex,
        set_source=Gen3RandbatSource.from_showdown_root(args.showdown_root),
        config=EngineMctsConfig(worlds=1, search_time_ms=1),
    )
    try:
        rows = [
            attest_target(
                env=env,
                policy=policy,
                dex=dex,
                seed=seed,
                target_step=step,
                max_steps=args.max_steps,
            )
            for seed, step in args.target
        ]
    finally:
        env.close()
    payload = {"schema_version": "pokezero.materialized_damage_stats.v1", "targets": rows}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if all(row["status"] == "attested" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
