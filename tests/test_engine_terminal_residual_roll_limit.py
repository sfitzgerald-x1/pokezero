"""Consumer ablation pin for the withdrawn terminal-residual roll experiment.

This asserts the documented compact behavior and deliberately does not present
it as a simulator-fidelity result. See docs/terminal_residual_roll_branching_limit.md.
"""

from __future__ import annotations

import unittest
from pathlib import Path

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


def _heals_to(branch, side: str) -> list[int]:
    prefix = f"Heal {side}:"
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

        # WHAT THIS PIN IS FOR, asserted directly instead of by proxy.
        #
        # It used to assert `len(branches) == 2` with the message "no production residual roll
        # splitter is installed", and a specific 113/19 damage-heal pair. Both were proxies for
        # "the withdrawn terminal-toxic experiment is not in the patch stack", and both went
        # stale when OTHER, deliberate splitters shipped -- `crit-kill-split` (C27, #1007) added
        # a 1/16 crit branch and `residual-lethality-partition` (#1069) split the non-crit mass,
        # so this state now yields three branches at 58.594 / 35.156 / 6.250 percent and the
        # single 113 representative became 109 and 116.
        #
        # None of that is the withdrawn experiment coming back, and refreshing the numbers would
        # turn an ablation pin into a change-detector that any future legitimate splitter breaks
        # again. The withdrawn thing has a name, so assert its absence by name.
        patch_dir = Path(__file__).resolve().parents[1] / "third_party"
        withdrawn = patch_dir / "poke-engine-gen3-terminal-toxic-roll-split.patch"
        self.assertFalse(
            withdrawn.exists(),
            f"{withdrawn.name} is back in the patch stack. It treated a pre-move Toxic "
            "arithmetic threshold as the decision boundary, which cannot model gen3's "
            "end-of-turn queue -- see docs/terminal_residual_roll_branching_limit.md. If it was "
            "reinstated deliberately, that document and this pin both need rewriting.",
        )

        # And the durable half of the old proxy: the engine still uses a COMPACT representative
        # for residual-lethal non-direct damage. Every branch's residual heal is the single
        # representative value, not a fan -- that is the comparison limit the doc records, and it
        # is what the withdrawn patch would have changed.
        heals = {tuple(_heals_to(branch, "SideOne")) for branch in branches}
        self.assertEqual(
            heals,
            {(19,), ()},
            "residual heals are no longer a single compact representative per branch",
        )
