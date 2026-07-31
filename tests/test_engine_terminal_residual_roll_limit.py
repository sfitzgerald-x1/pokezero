"""Consumer ablation pin for the withdrawn terminal-residual roll experiment.

This asserts the documented compact behavior and deliberately does not present
it as a simulator-fidelity result. See docs/terminal_residual_roll_branching_limit.md.
"""

from __future__ import annotations

import unittest

try:
    import poke_engine
except ImportError:  # pragma: no cover - native wheel absent
    poke_engine = None


def _fainted_dummy():
    return poke_engine.Pokemon(id="pikachu", level=1, hp=0)


def _comparison_limit_state():
    attacker = poke_engine.Pokemon(
        id="piloswine", level=87, types=("ice", "ground"), hp=154, maxhp=316,
        ability="oblivious", item="leftovers", attack=224, defense=180,
        special_attack=140, special_defense=180, speed=137,
        moves=[poke_engine.Move(id="earthquake", pp=16)],
    )
    defender = poke_engine.Pokemon(
        id="slaking", level=100, types=("normal", "typeless"), hp=182, maxhp=362,
        ability="truant", item="choiceband", attack=300, defense=201,
        special_attack=200, special_defense=180, speed=201, status="toxic",
        moves=[poke_engine.Move(id="splash", pp=16)],
    )
    return poke_engine.State(
        side_one=poke_engine.Side(active_index="0", pokemon=[attacker] + [_fainted_dummy()] * 5),
        side_two=poke_engine.Side(
            active_index="0", pokemon=[defender] + [_fainted_dummy()] * 5,
            side_conditions=poke_engine.SideConditions(toxic_count=2),
        ),
        weather="none", terrain="none", trick_room=False,
    )


def _damage_to(branch, side: str) -> list[int]:
    prefix = f"Damage {side}:"
    return [
        int(str(instruction).rsplit(":", 1)[1])
        for instruction in branch.instruction_list
        if str(instruction).startswith(prefix)
    ]


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class TerminalResidualRollComparisonLimitTests(unittest.TestCase):
    def test_withdrawn_split_is_an_explicit_unpatched_ablation(self) -> None:
        state = _comparison_limit_state()
        before = str(state)
        branches = list(poke_engine.generate_instructions(state, "earthquake", "splash"))

        self.assertEqual(str(state), before, "branch generation must restore the input state")
        self.assertAlmostEqual(sum(float(branch.percentage) for branch in branches), 100.0)
        self.assertEqual(len(branches), 2, "no production residual roll splitter is installed")
        self.assertTrue(
            any(
                _damage_to(branch, "SideTwo")[0] == 113
                and any(str(item) == "Heal SideOne: 19" for item in branch.instruction_list)
                for branch in branches
            )
        )
