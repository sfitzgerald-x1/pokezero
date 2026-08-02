"""Pin for a Sleep-Talk-called phazing move with an unidentifiable callee.

Roar and Whirlwind are both Sleep-Talk-callable in gen 3 and produce identical
instruction tails, so `identify_sleep_talk_called` cannot prove which ran and
the renderer takes its unidentified-callee path.

Two wrong answers were shipped before this pin existed, which is why it exists:

* emitting HP lines against `before[]` captured BEFORE the tail, which spans two
  different Pokemon once the tail drags one in -- producing
  `|-heal|p2a: Snorlax|100/100|[from] residual` for a mon that never healed and
  never got a switch line;
* bailing on the whole block, which leaves no `|drag|` at all, so the consumer's
  running HP stays on the OUTGOING Pokemon and the next residual is measured
  against it -- a Leftovers tick decomposing to -14, verbatim the impossible
  component of reports/c52_impossible_heal_component.json.

The callee is unprovable. The drag is not: it names no move, so rendering it
invents no attribution, and it re-baselines the consumer.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import pokezero_search  # noqa: E402
from engine_transition_differential import damage_component_events  # noqa: E402
from pokezero.poke_engine_adapter import (  # noqa: E402
    BattleSpec,
    MoveSpec,
    PokemonSpec,
    SideSpec,
    build_poke_engine_state,
)
from test_instruction_event_mapping import CTX  # noqa: E402


def _mon(species, moves, *, hp=100, speed=100, status="none", item=None):
    return PokemonSpec(
        id=species, level=100, types=("normal",), hp=hp, maxhp=100, attack=100,
        defense=100, special_attack=100, special_defense=100, speed=speed,
        status=status, ability=None, item=item, sleep_turns=0,
        moves=tuple(MoveSpec(id=m, pp=32) for m in moves),
    )


class SleepTalkPhazeDragTests(unittest.TestCase):
    def _branch(self, benched_hp: int):
        spec = BattleSpec(
            side_one=SideSpec(
                pokemon=(_mon("rattata", ("sleeptalk", "roar", "whirlwind"),
                              speed=500, status="sleep"),),
                volatile_statuses=(), side_conditions={}, boosts={},
            ),
            side_two=SideSpec(
                pokemon=(_mon("chansey", ("splash",), hp=50),
                         _mon("snorlax", ("splash",), hp=benched_hp, item="leftovers")),
                volatile_statuses=(), side_conditions={}, boosts={},
            ),
        )
        state = build_poke_engine_state(spec).to_string()
        report = json.loads(
            pokezero_search.branch_events(state, "sleeptalk", "splash", CTX, True, True)
        )
        for branch in report["branches"]:
            if any("|drag|" in line for line in branch["events"]):
                return branch
        self.fail(f"no dragging branch rendered: {report['branches']}")

    def test_the_drag_is_rendered(self) -> None:
        branch = self._branch(30)
        drags = [line for line in branch["events"] if line.startswith("|drag|")]
        self.assertEqual(len(drags), 1, branch)

    def test_no_hp_line_names_a_mon_that_never_entered(self) -> None:
        """The first bug: an HP line for the incoming mon, with no switch line
        anywhere, because before[] belonged to the outgoing one."""

        branch = self._branch(100)
        residuals = [line for line in branch["events"] if "[from] residual" in line]
        self.assertEqual(residuals, [], branch)

    def test_the_residual_decomposes_against_the_incoming_mon(self) -> None:
        """The second bug: with no drag line the consumer measured the incoming
        mon's Leftovers tick from the OUTGOING mon's HP, yielding -14 -- a
        negative Leftovers, which is impossible."""

        branch = self._branch(30)
        components = damage_component_events(
            [line for line in branch["events"] if line.strip()],
            {"p1": 100, "p2": 50},
        )
        self.assertEqual(
            [(c.source, c.delta) for c in (components.get("p2") or [])],
            [("itemleftovers", 6)],
            branch,
        )


if __name__ == "__main__":
    unittest.main()
