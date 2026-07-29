#!/usr/bin/env python
"""Replay a recorded divergent boundary back through the engine.

**This is step ONE of triaging any residue class, not the fallback after a
diagnosis fails.** Twice in a row a residue label was wrong and the replay was
right: #893 re-attributed a "spikes" class to Whirlwind from branch-count
reasoning, and #895 then disproved that same lead by replay before finding the
real bug (Protect blocks phazing) during source verification. A class label is a
*hypothesis about a symptom*; the replay is the experiment.

What it does, for each selected row of an
``engine_transition_differential.py --json`` report:

1. deserializes every recorded candidate ``engine_state`` (the full
   hidden-counter sweep, not just the first rung — those states ARE the union
   the matcher judged);
2. re-runs the exact recorded joint action through
   ``poke_engine.generate_instructions`` and through the instruction->event
   mapper (``pokezero_search.branch_events``);
3. prints every branch's attributed components beside the observation's, plus
   the Showdown protocol slice, so the disagreement is visible per component
   rather than inferred from a class name.

Nothing here mutates state or re-runs Showdown: it is a pure re-execution of a
state the differential already captured, so it is cheap and exactly reproducible.

Usage::

    PYTHONPATH=src python scripts/replay_residue.py --report residue_300.json \\
        [--seed 1310000] [--step 193] [--class component_magnitude:psn] [--limit 3]

Exit code is 0 always — this is a diagnostic, not a gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import poke_engine  # noqa: E402
import pokezero_search  # noqa: E402

from engine_transition_differential import (  # noqa: E402
    _split_components,
    damage_components,
)


def _fmt(components: Sequence[tuple[str, int]]) -> str:
    if not components:
        return "(none)"
    return ", ".join(f"{source or 'move'}={delta:+d}" for source, delta in components)


def replay_row(row: Mapping[str, Any], *, max_branches: int) -> None:
    seed = row.get("seed")
    step = row.get("step")
    choices = row.get("choices") or {}
    print("=" * 100)
    print(
        f"seed {seed}  step {step}  class={row.get('divergence_class')}  "
        f"gating={row.get('gating', '?')}  choices={choices}"
    )

    protocol = [l for l in (row.get("protocol") or []) if not l.startswith("|request|")]
    print("\n-- Showdown step --")
    for line in protocol:
        print(f"   {line[:150]}")

    # Seed the running HP with the recorded PRE-STATE, exactly as the matcher
    # does. Without it the first HP line for a slot only sets the baseline and
    # its delta is dropped — which hides the step's primary move damage and
    # makes the replay disagree with the verdict it is supposed to explain.
    pre = row.get("pre_features") or {}
    pre_hp = {"p1": int(pre.get("p1_hp", 0)), "p2": int(pre.get("p2_hp", 0))}
    print(f"\n-- pre-state HP: {pre_hp} --")
    observed = damage_components(protocol, pre_hp)
    print("\n-- observed components --")
    for slot in ("p1", "p2"):
        exact, rolled = _split_components(observed[slot])
        print(f"   {slot}: exact={_fmt(sorted(exact.elements()))}  rolled={_fmt(rolled)}")

    states = row.get("engine_states") or ([row["engine_state"]] if row.get("engine_state") else [])
    if not states:
        print("\n!! no engine_state recorded — re-run the differential with --keep-repro")
        return
    print(f"\n-- engine: {len(states)} candidate state(s) (hidden-counter sweep) --")

    party = row.get("party_display") or {"p1": [], "p2": []}
    ctx = json.dumps({"p1": party.get("p1", []), "p2": party.get("p2", []), "turn": 0})

    # The recorded choices are keyed by PLAYER SLOT; branch_events wants
    # side_one/side_two. The differential records slot_sides on the row when it
    # can; default to the identity mapping it uses in the overwhelming majority.
    slot_sides = row.get("slot_sides") or {"p1": "side_one", "p2": "side_two"}
    s1 = choices["p1"] if slot_sides.get("p1") == "side_one" else choices["p2"]
    s2 = choices["p2"] if slot_sides.get("p2") == "side_two" else choices["p1"]

    for index, state_str in enumerate(states):
        try:
            rendered = json.loads(
                pokezero_search.branch_events(state_str, s1, s2, ctx, True, True)
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # noqa: BLE001
            print(f"   candidate {index}: branch_events failed: {type(error).__name__}: {error}")
            continue
        branches = rendered.get("branches") or []
        print(f"   candidate {index}: {len(branches)} branch(es)")
        for branch in branches[:max_branches]:
            pct = float(branch.get("percentage") or 0.0)
            lossy = branch.get("lossy") or []
            engine = damage_components(
                branch.get("events") or [],
                {
                    ("p1" if slot_sides.get("p1") == "side_one" else "p2"): pre_hp["p1"],
                    ("p2" if slot_sides.get("p2") == "side_two" else "p1"): pre_hp["p2"],
                },
            )
            bits = []
            for slot in ("p1", "p2"):
                exact, rolled = _split_components(engine[slot])
                bits.append(
                    f"{slot} exact={_fmt(sorted(exact.elements()))} rolled={_fmt(rolled)}"
                )
            marker = f"  LOSSY={lossy}" if lossy else ""
            print(f"      pct={pct:6.2f}  " + "  |  ".join(bits) + marker)

    # Raw instruction list of the first candidate: the ground truth the mapper
    # renders from, and the thing to read when a rendering looks wrong.
    try:
        state = poke_engine.State.from_string(states[0])
        raw = poke_engine.generate_instructions(state, s1, s2)
        print("\n-- raw instructions, candidate 0, highest-probability branch --")
        best = max(raw, key=lambda b: float(b.percentage))
        for instruction in best.instruction_list:
            print(f"      {instruction}")
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:  # noqa: BLE001
        print(f"\n-- raw instructions unavailable: {type(error).__name__}: {error}")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--class", dest="klass", default=None,
                        help="substring match on divergence_class")
    parser.add_argument("--limit", type=int, default=3, help="rows to replay")
    parser.add_argument("--max-branches", type=int, default=8)
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text())
    rows = [r for r in (report.get("repros") or []) if r.get("kind") == "transition_diverged"]
    if args.seed is not None:
        rows = [r for r in rows if r.get("seed") == args.seed]
    if args.step is not None:
        rows = [r for r in rows if r.get("step") == args.step]
    if args.klass:
        rows = [r for r in rows if args.klass in str(r.get("divergence_class", ""))]

    if not rows:
        print("no matching residue rows (was the report written with --keep-repro > 0?)")
        return 0
    print(f"{len(rows)} matching row(s); replaying {min(len(rows), args.limit)}\n")
    for row in rows[: args.limit]:
        replay_row(row, max_branches=args.max_branches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
