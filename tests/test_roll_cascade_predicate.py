"""Pins for the roll-cascade fallback, written from the #1010 review findings.

The reviewer's counterexample is the first test: an observed Leftovers tick
against an engine Wish plus Leech Seed, nets equal. Different mechanics, and
the first version of the predicate accepted them.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("etd", ROOT / "scripts" / "engine_transition_differential.py")
D = importlib.util.module_from_spec(spec); sys.modules["etd"] = D; spec.loader.exec_module(D)


def comp(source: str, delta: int, index: int = 0):
    return D.DamageComponent(source=source, delta=delta, event_index=index)


def cascade(observed, engine, legal=None):
    return D._roll_cascade_equivalent(
        observed, engine, support=None, target_side="side_one", pre_legal=legal
    )


class RollCascadeTests(unittest.TestCase):
    def test_mismatched_heal_sources_are_rejected(self) -> None:
        """The review's counterexample. Nets agree at -75 on both sides, and the
        engine's heals are a different MECHANISM -- Wish plus Leech Seed against
        a Leftovers tick. Equal sums are not one transition at two rolls."""

        observed = [comp("", -100), comp("itemleftovers", 25)]
        engine = [comp("", -110), comp("movewish", 20), comp("leechseed", 15)]
        self.assertEqual(sum(c.delta for c in observed), sum(c.delta for c in engine))
        self.assertFalse(cascade(observed, engine))

    def test_the_genuine_cascade_is_accepted(self) -> None:
        """s18000588/37: one legal roll apart, the Wish stops capping, and a
        Leftovers residual fills the 4 HP it left."""

        observed = [comp("", -41), comp("movewish_to_full", 119)]
        engine = [comp("", -45), comp("movewish", 119), comp("itemleftovers_to_full", 4)]
        self.assertTrue(cascade(observed, engine))

    def test_the_direct_component_faces_the_ordinary_roll_check(self) -> None:
        """The first version returned before the per-component loop, so the
        direct damage never met ``roll_components_agree`` at all -- only a
        proportional band inside the cascade helper.

        Note what this does NOT assert. An empty ``legal`` set is not a veto:
        the module documents ``legal`` as "an ADDITIONAL accept path, never a
        veto", because it is computed from the pre-state with an assumed move
        order and letting it reject would fail boundaries where the two sims
        agree to the HP point. The ordinary path passes with ``legal=set()``
        too. Requiring a veto here would be inconsistent with the matcher.

        What the routing buys is that a direct component too far apart to be
        one roll is now rejected by the same rule every other roll-scaled
        component obeys."""

        heals_observed = [comp("movewish_to_full", 119)]
        heals_engine = [comp("movewish", 119), comp("itemleftovers_to_full", 4)]
        # one roll apart: accepted
        self.assertTrue(cascade([comp("", -41)] + heals_observed,
                                [comp("", -45)] + heals_engine))
        # far outside any roll, conservation still satisfied: rejected
        self.assertFalse(cascade([comp("", -10), comp("movewish_to_full", 119)],
                                 [comp("", -45), comp("movewish", 119),
                                  comp("itemleftovers_to_full", 35)]))

    def test_two_extra_components_are_rejected(self) -> None:
        observed = [comp("", -41), comp("movewish_to_full", 119)]
        engine = [comp("", -49), comp("movewish", 119), comp("itemleftovers", 4), comp("heal", 4)]
        self.assertFalse(cascade(observed, engine))

    def test_identical_directs_with_a_shifted_shared_heal_are_rejected(self) -> None:
        """Re-review hole (a). The direct components are IDENTICAL, so there is
        no roll gap and no cascade -- but comparing heal SOURCES only let the
        16 HP shift hide inside the 'extra' component."""

        self.assertFalse(cascade(
            [comp("", -100), comp("movewish", 60)],
            [comp("", -100), comp("movewish", 44), comp("itemleftovers", 16)],
        ))
        self.assertFalse(cascade(
            [comp("", -100), comp("itemleftovers", 25)],
            [comp("", -100), comp("itemleftovers", 9), comp("leechseed", 16)],
        ))

    def test_a_discrete_mechanic_difference_is_rejected(self) -> None:
        """Re-review hole (b). Conservation holds and the extra equals the roll
        gap, but no shared heal flipped its cap -- so nothing stopped topping
        the mon out and there is no cascade. The engine simply sees a Leech Seed
        the observation never did."""

        self.assertFalse(cascade(
            [comp("", -100), comp("itemleftovers", 25)],
            [comp("", -106), comp("itemleftovers", 25), comp("leechseed", 6)],
        ))

    def test_the_extra_component_must_be_the_roll_gap(self) -> None:
        """C66 measured this identity and named it as what licenses the match;
        the first implementation shipped bare conservation instead."""

        self.assertFalse(cascade(
            [comp("", -41), comp("movewish_to_full", 119)],
            [comp("", -45), comp("movewish", 111), comp("itemleftovers_to_full", 12)],
        ))

    def test_unequal_totals_are_rejected(self) -> None:
        observed = [comp("", -41), comp("movewish_to_full", 119)]
        engine = [comp("", -45), comp("movewish", 119), comp("itemleftovers_to_full", 40)]
        self.assertFalse(cascade(observed, engine))

    def test_a_negative_heal_is_rejected(self) -> None:
        """An impossible component must never license a match (C52)."""

        observed = [comp("", -41), comp("movewish_to_full", 119)]
        engine = [comp("", -45), comp("movewish", 119), comp("itemleftovers", -4)]
        self.assertFalse(cascade(observed, engine))


if __name__ == "__main__":
    unittest.main()
