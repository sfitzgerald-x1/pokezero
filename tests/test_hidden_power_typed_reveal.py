"""Hidden Power's TYPE must survive the parse-to-belief path.

gen3 randbats uses per-type move ids (``hiddenpowergrass`` and 12 others across 727 variant
occurrences), and the engine masks the type only for the generic gen6+ ``hiddenpower`` id
(``sim/battle-actions.ts``: ``if (move.id === 'hiddenpower') movename = 'Hidden Power'``). That
branch never fires in gen3, so the full name goes out on the wire and the type is public on
first use.

That matters more than most reveals: 28 of the 130 Hidden-Power-carrying species run more than
one type, and on those the typed name frequently SOLVES the set outright.

The failure mode is silent. A bare ``"Hidden Power"`` narrows NOTHING -- not "less", nothing --
so a path that delivers the untyped form loses a full set-solve with no error and no signal.
A belief that is quietly worse is harder to notice than one that is absent, which is why this
is pinned by behaviour rather than left to the protocol.
"""
import os
import unittest

from pokezero.belief import PublicBattleBeliefEngine
from pokezero.randbat import load_gen3_randbat_source_cached
from pokezero.showdown import parse_showdown_replay

SHOWDOWN_ROOT = os.environ.get(
    "POKEZERO_SHOWDOWN_ROOT", "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
)

# Charizard runs Hidden Power Flying AND Hidden Power Grass, so the type is genuinely
# discriminating here rather than implied by the species.
OPENING = [
    "|start",
    "|switch|p1a: Furret|Furret, L88, M|100/100",
    "|switch|p2a: Charizard|Charizard, L82, M|100/100",
]

try:
    _SOURCE = load_gen3_randbat_source_cached(SHOWDOWN_ROOT)
except Exception:  # pragma: no cover - no checkout in this environment
    _SOURCE = None


@unittest.skipIf(_SOURCE is None, "needs a pokemon-showdown checkout")
class HiddenPowerTypedRevealTest(unittest.TestCase):
    def _candidates(self, move: str) -> int:
        replay = parse_showdown_replay(
            [*OPENING, f"|move|p2a: Charizard|{move}|p1a: Furret", "|turn|1"],
            battle_id="battle-gen3randombattle-hp",
        )
        engine = PublicBattleBeliefEngine.from_events(
            replay.public_events, format_id="gen3randombattle", set_source=_SOURCE
        )
        return engine.snapshot().side("p2")[0].candidate_set_count

    def test_the_pool_really_carries_typed_hidden_power_ids(self) -> None:
        """Premise check: if the generator ever collapsed to a single `hiddenpower` id, the
        narrowing below would be unreachable and the rest of this file would pass vacuously."""
        typed = {
            move
            for universe in _SOURCE.universes.values()
            for variant in universe.variants
            for move in variant.moves
            if move.startswith("hiddenpower") and move != "hiddenpower"
        }
        self.assertGreater(len(typed), 5, f"expected many typed ids, got {sorted(typed)}")

        multi = [
            key
            for key, universe in _SOURCE.universes.items()
            if len(
                {m for v in universe.variants for m in v.moves if m.startswith("hiddenpower")}
            )
            > 1
        ]
        self.assertIn("charizard", multi, "the probe species must run more than one HP type")

    def test_a_typed_hidden_power_narrows_the_candidate_set(self) -> None:
        baseline = self._candidates("Tackle")
        typed = self._candidates("Hidden Power Grass")
        self.assertLess(typed, baseline, "the typed name failed to narrow at all")

    def test_the_untyped_form_narrows_nothing_and_that_is_the_hazard(self) -> None:
        """Pins the SILENCE, so a regression to the untyped form is visible here.

        This asserts current reality rather than a desired behaviour: a bare "Hidden Power"
        carries no type, so belief correctly cannot narrow on it. The point of the assertion is
        that the two forms differ enormously, which is what makes an upstream change that starts
        emitting the untyped name a real loss rather than a cosmetic one.
        """
        baseline = self._candidates("Tackle")
        self.assertEqual(
            self._candidates("Hidden Power"),
            baseline,
            "untyped Hidden Power unexpectedly narrowed -- the hazard model has changed",
        )
        self.assertLess(
            self._candidates("Hidden Power Grass"),
            self._candidates("Hidden Power"),
            "the typed and untyped forms must differ, or this guard is pointless",
        )

    def test_an_untyped_reveal_degrades_safely_instead_of_excluding(self) -> None:
        """The degradation is SAFE, and that is the property worth pinning.

        `_revealed_move_matches_variant` treats a bare `hiddenpower` as matching ANY typed
        variant, so a lost type yields NO narrowing rather than a wrong exclusion. That is the
        right shape -- a belief that is absent costs less than one that is wrong -- but it is
        only right by construction, and a "tighter" future matcher that required an exact id
        would silently eliminate every real variant and collapse the set to the inconsistent
        fallback.
        """
        from pokezero.randbat import _revealed_move_matches_variant

        typed_pool = {"hiddenpowergrass", "surf"}
        self.assertTrue(
            _revealed_move_matches_variant("Hidden Power", typed_pool),
            "an untyped reveal excluded a typed variant -- it must degrade, not eliminate",
        )
        self.assertTrue(_revealed_move_matches_variant("Hidden Power Grass", typed_pool))
        self.assertFalse(_revealed_move_matches_variant("Hidden Power Fire", typed_pool))

        # End to end: the untyped form must never empty or contradict the set.
        self.assertGreater(self._candidates("Hidden Power"), 0)

    def test_each_type_selects_a_different_survivor_set(self) -> None:
        """Two types on one species must not collapse to the same answer, which would mean the
        type was being dropped somewhere between parse and variant matching."""
        self.assertNotEqual(
            self._candidates("Hidden Power Grass"),
            self._candidates("Hidden Power Flying"),
        )


if __name__ == "__main__":
    unittest.main()
