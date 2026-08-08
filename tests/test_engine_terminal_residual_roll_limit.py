"""Consumer ablation pin for the withdrawn terminal-residual roll experiment.

This asserts the documented compact behavior -- the exact branch partition the engine
produces for a residual-lethal non-direct roll -- and deliberately does not present
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

        # RE-DERIVED, not loosened. An earlier revision of this change replaced these numbers
        # with (a) the withdrawn patch file's absence from third_party/ and (b) a set of residual
        # heals. Review demonstrated both are unable to catch the withdrawn behaviour: running
        # this same fixture under POKEZERO_ENUMERATE_ROLLS emits the 17-arm roll fan the doc
        # attributes to the withdrawn patch, and that version PASSED while the numeric pin below
        # correctly FAILED (17 != 2). The filename check is theatre -- patches apply from
        # `third_party/poke-engine-gen3-patches.txt`, the hunks could fold into any listed patch
        # (#1152's crit-straddle patch already lives in this code path), and the test exercises a
        # prebuilt WHEEL, not repo source. The heals set collapsed cardinality, so a fan and a
        # 2-way split were indistinguishable.
        #
        # The expectations are derived from the roll lattice, independently of the
        # implementation, mirroring the native twin
        # `rust/pokezero-search/tests/gen3_battle_end_residuals.rs`, which was RE-DERIVED rather
        # than relaxed when `residual-lethality-partition` (#1062) landed:
        #
        #   max non-crit roll 123; residual tick 66; defender HP 182, so the residual-KO
        #   threshold is 182 - 66 = 116. Of the sixteen rolls floor(123 * r / 100), r in 85..=100,
        #   six reach 116 and ten fall below.
        #     surviving arm  109 (mean of the ten below) + 66 tick, attacker Leftovers 19 ticks
        #                    because the battle has not ended        (1 - 1/16) * 10/16 = 58.5938%
        #     residual-KO    116 + 66 == 182 exactly; NO Leftovers heal, because side two has no
        #                    other live Pokemon so stop_residuals_if_battle_ended fires before
        #                    order 10 reaches side one              (1 - 1/16) *  6/16 = 35.1562%
        #     crit arm       max_crit 246 >= 182 kills on the hit, so it keeps the full base crit
        #                    rate, unscaled by the non-crit split               1/16 =  6.2500%
        #
        # A fan fails this. So does a partition-arithmetic change. So does the crit arm being
        # rescaled by the non-crit split.
        observed = {
            (
                tuple(_damage_to(branch, "SideTwo")),
                tuple(_heals_to(branch, "SideOne")),
                round(float(branch.percentage), 4),
            )
            for branch in branches
        }
        self.assertEqual(
            observed,
            {
                ((109, 66), (19,), 58.5938),
                ((116, 66), (), 35.1562),
                ((182,), (), 6.25),
            },
        )

        # Secondary breadcrumb only -- explicitly NOT what holds the line, per the precedent in
        # tests/test_roll_enumeration_scope.py where a file scan was defeated in review and
        # demoted to a change ledger.
        withdrawn = (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "poke-engine-gen3-terminal-toxic-roll-split.patch"
        )
        self.assertFalse(
            withdrawn.exists(),
            f"{withdrawn.name} is back by name; see docs/terminal_residual_roll_branching_limit.md",
        )
