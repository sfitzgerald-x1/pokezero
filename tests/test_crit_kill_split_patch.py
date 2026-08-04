"""Pins for the C27/C31 threshold-partition patches.

WHAT THIS IS AND IS NOT. `compare_health_with_damage_multiples` is a private Rust
function in the vendored engine; it is not callable from Python, and the vendored
tree is a gitignored derivation. So these are **contract and source pins**, not
behavioural ones:

* the contract pin replicates the identity's intended arithmetic in Python and
  asserts the property C27 fixed — a roll landing EXACTLY on the defender's HP is
  lethal, and its mass is counted;
* the source pins assert the patches are present in the stack with the reasoning
  that motivated them, so a silent revert fails here rather than in a sweep.

A behavioural pin was attempted and abandoned honestly. Finding a real boundary
where a legal roll lands exactly on the defender's remaining HP requires pairing
`calculate_damage` output with the RIGHT defender, and the obvious heuristic pairs
it with the pre-state active — which is the wrong mon whenever the boundary opens
with a switch. Rather than pin a case that looked right and was not, the gap is
recorded: a behavioural pin needs a boundary constructed for the purpose, not
mined from the archive by magnitude matching.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _patch_stack():
    """Load the applicator so the registration guard reads the SAME list it does."""

    spec = importlib.util.spec_from_file_location(
        "apply_poke_engine_patches", ROOT / "scripts" / "apply_poke_engine_patches.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec: the module defines a frozen dataclass, and
    # dataclasses resolves the defining module out of sys.modules.
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


PATCH_LIST = ROOT / "third_party" / "poke-engine-gen3-patches.txt"
KILL_SPLIT_PATCH = ROOT / "third_party" / "poke-engine-gen3-crit-kill-split.patch"
SUBSTITUTE_PATCH = ROOT / "third_party" / "poke-engine-gen3-substitute-hp-gate.patch"
BELLYDRUM_PATCH = ROOT / "third_party" / "poke-engine-gen3-bellydrum-roll-gate.patch"

ROLLS = range(85, 101)


def kill_rolls(max_damage: int, health: int) -> int:
    """The identity's intended accounting: exactly-on-the-line is LETHAL.

    gen3 damage is ``floor(base * random(85, 100) / 100)``. A roll that leaves the
    defender on exactly zero killed it, so it belongs in the kill class and its
    1/16 of mass belongs to the kill branch.
    """

    return sum(1 for roll in ROLLS if max_damage * roll // 100 >= health)


def survive_rolls(max_damage: int, health: int) -> int:
    return sum(1 for roll in ROLLS if max_damage * roll // 100 < health)


class KillSplitContractTests(unittest.TestCase):
    def test_every_roll_lands_in_exactly_one_class(self) -> None:
        """The defect was a roll in NEITHER class, so mass must always total 16."""

        for max_damage in range(20, 400, 7):
            for health in range(10, max_damage + 40, 11):
                with self.subTest(max_damage=max_damage, health=health):
                    self.assertEqual(
                        kill_rolls(max_damage, health)
                        + survive_rolls(max_damage, health),
                        16,
                    )

    def test_an_exactly_lethal_roll_is_counted_as_a_kill(self) -> None:
        """base 100 against 90 HP: the loop steps by exactly 1, so roll 90 lands
        on the line. The old `damage > health` test dropped it from both buckets,
        sending its mass to the survive branch — an exactly-lethal roll priced as
        a survival."""

        exact = [roll for roll in ROLLS if 100 * roll // 100 == 90]
        self.assertEqual(exact, [90], "expected a roll landing exactly on HP")
        self.assertEqual(kill_rolls(100, 90), 11)
        strictly_greater = sum(1 for roll in ROLLS if 100 * roll // 100 > 90)
        self.assertEqual(
            strictly_greater, 10, "the pre-C27 accounting undercounts by the orphan"
        )

    def test_a_straddle_splits_mass_across_both_classes(self) -> None:
        self.assertGreater(kill_rolls(200, 190), 0)
        self.assertGreater(survive_rolls(200, 190), 0)

    def test_no_straddle_puts_all_mass_on_one_side(self) -> None:
        self.assertEqual(kill_rolls(100, 500), 0)
        self.assertEqual(survive_rolls(100, 500), 16)
        self.assertEqual(kill_rolls(500, 100), 16)
        self.assertEqual(survive_rolls(500, 100), 0)


class PatchStackPinTests(unittest.TestCase):
    def test_both_partition_patches_are_registered_in_order(self) -> None:
        # Must use patch_names(), the applicator's own parse, NOT .split() on the
        # raw text. .split() tokenises the comment prose too, so a patch mentioned
        # in a comment but de-registered still satisfied assertIn -- exactly the
        # failure class that let #1017 register a patch twice and break the build.
        stack = _patch_stack().patch_names(PATCH_LIST)
        self.assertIn("poke-engine-gen3-crit-kill-split.patch", stack)
        self.assertIn("poke-engine-gen3-substitute-hp-gate.patch", stack)
        self.assertLess(
            stack.index("poke-engine-gen3-crit-kill-split.patch"),
            stack.index("poke-engine-gen3-substitute-hp-gate.patch"),
            "C31 extends the predicate C27's context established",
        )

    def test_kill_split_patch_carries_the_orphan_fix(self) -> None:
        patch = KILL_SPLIT_PATCH.read_text(encoding="utf-8")
        # The `> health` test became a bare `else`, so on-the-line is a kill.
        self.assertIn("-        } else if damage > health_f32 {", patch)
        self.assertIn("+        } else {", patch)
        self.assertIn("EXACTLY on the defender", patch)
        # And the crit arm gained the identity case A already used.
        self.assertIn("compare_health_with_damage_multiples(max_crit_damage", patch)

    def test_substitute_patch_extends_the_existing_predicate(self) -> None:
        patch = SUBSTITUTE_PATCH.read_text(encoding="utf-8")
        self.assertIn("Choices::SUBSTITUTE", patch)
        self.assertIn("pending_hp_reading_move", patch)
        # It must EXTEND the Flail/Reversal predicate, not stand beside it.
        self.assertIn("Choices::FLAIL | Choices::REVERSAL", patch)

    def test_bellydrum_patch_extends_the_same_predicate(self) -> None:
        """Belly Drum is the Substitute case at maxhp/2 instead of maxhp/4.

        Pinned for the same reason the Substitute arm is: the allowlist is an
        enumeration, and it has now been found short twice. A new arm must
        EXTEND `pending_hp_reading_move` rather than fork a second predicate,
        because the 16-roll preservation it gates is keyed on that one call.
        """
        patch = BELLYDRUM_PATCH.read_text(encoding="utf-8")
        self.assertIn("Choices::BELLYDRUM", patch)
        self.assertIn("pending_hp_reading_move", patch)
        # Extends, not replaces: all three prior arms must survive.
        self.assertIn(
            "Choices::FLAIL | Choices::REVERSAL | Choices::SUBSTITUTE | Choices::BELLYDRUM",
            patch,
        )
        # The gate is maxhp/2, and the patch must say so rather than leaving the
        # reader to infer it from Substitute's maxhp/4.
        self.assertIn("maxhp / 2", patch)


if __name__ == "__main__":
    unittest.main()
