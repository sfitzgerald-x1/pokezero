"""Hidden Power's type is PRIVATE in gen3, and the belief matcher must degrade rather than exclude.

An earlier version of this file asserted the opposite -- that gen3 puts the typed name on the
wire and the type is public on first use -- and built three tests on it. That premise is wrong,
and this docstring exists partly to stop it being rediscovered:

    data/mods/gen3/scripts.ts  ``if (move.id === 'hiddenpower') movename = 'Hidden Power';``

gen3 carries its OWN copy of the masking branch, so every ``|move|`` line is untyped. The choice
parser collapses typed ids too (``sim/side.ts``, ``sim/pokemon.ts`` ``hasMove``). Verified against
this repo's real captures: all 13 Hidden Power ``|move|`` lines in
``tests/fixtures/showdown/capture/`` are bare ``Hidden Power``, including the Sleep-Talk-called
one. ``tests/test_engine_env.py::test_hidden_power_type_is_not_publicly_revealed`` already pinned
this; the earlier docstring contradicted a pinned test.

The pool DOES use per-type ids -- 13 of them across 727 variant occurrences, on 130 species, 28
of which run more than one type. That is set DATA, not the wire, and conflating the two is what
produced the wrong premise. It is why the matcher has to be lenient: the belief engine sees
``hiddenpower`` and the candidate variants carry ``hiddenpowergrass``.

So the property worth pinning is not that the type survives -- it never arrives. It is that the
untyped reveal DEGRADES SAFELY: no narrowing, rather than excluding every typed variant. A
"tighter" matcher requiring an exact id would collapse all 130 Hidden-Power-carrying species to
the inconsistent fallback, silently, on a channel the belief system feeds.
"""
import os
import pathlib
import re
import unittest

from pokezero.belief import PublicBattleBeliefEngine
from pokezero.randbat import load_gen3_randbat_source_cached
from pokezero.showdown import parse_showdown_replay

SHOWDOWN_ROOT = os.environ.get(
    "POKEZERO_SHOWDOWN_ROOT", "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
)
CAPTURES = pathlib.Path(__file__).parent / "fixtures" / "showdown" / "capture"

OPENING = [
    "|start",
    "|switch|p1a: Furret|Furret, L88, M|100/100",
    "|switch|p2a: Charizard|Charizard, L82, M|100/100",
]

try:
    _SOURCE = load_gen3_randbat_source_cached(SHOWDOWN_ROOT)
except Exception:  # pragma: no cover - no checkout in this environment
    _SOURCE = None


class HiddenPowerProtocolPremiseTest(unittest.TestCase):
    """The premise the rest of this file rests on, checked against real server output.

    Needs no Showdown checkout: the captures are committed. That is deliberate -- the premise is
    the thing that was wrong, so it should be the thing hardest to skip.
    """

    def test_gen3_move_lines_never_carry_the_hidden_power_type(self) -> None:
        lines = []
        for log in sorted(CAPTURES.glob("lines-battle-gen3randombattle-*.log")):
            lines += [
                line
                for line in log.read_text(encoding="utf-8").splitlines()
                if re.match(r"\|move\|[^|]*\|Hidden Power", line)
            ]
        self.assertTrue(lines, "no Hidden Power move lines in the captures to check")
        typed = [line for line in lines if not re.search(r"\|Hidden Power(\||$)", line)]
        self.assertEqual(
            typed, [], "a gen3 capture carried a TYPED Hidden Power -- the premise has changed"
        )


@unittest.skipIf(_SOURCE is None, "needs a pokemon-showdown checkout")
class HiddenPowerBeliefDegradationTest(unittest.TestCase):
    def _candidates(self, move: "str | None") -> int:
        """Candidate count after revealing `move`; `None` reveals nothing at all.

        The no-reveal baseline is the honest one. An earlier version used Tackle -- which NO
        Charizard variant carries -- so its baseline was the INCONSISTENT-reveal fallback and
        happened to equal the real answer through a second, unrelated mechanism.
        """
        reveal = [f"|move|p2a: Charizard|{move}|p1a: Furret"] if move else []
        replay = parse_showdown_replay(
            [*OPENING, *reveal, "|turn|1"],
            battle_id="battle-gen3randombattle-hp",
        )
        engine = PublicBattleBeliefEngine.from_events(
            replay.public_events, format_id="gen3randombattle", set_source=_SOURCE
        )
        return engine.snapshot().side("p2")[0].candidate_set_count

    def test_the_pool_carries_typed_ids_even_though_the_wire_does_not(self) -> None:
        """The asymmetry that forces the matcher to be lenient. If the generator ever collapsed
        to a single `hiddenpower` id, the leniency would be unnecessary rather than load-bearing.
        """
        typed = {
            move
            for universe in _SOURCE.universes.values()
            for variant in universe.variants
            for move in variant.moves
            if move.startswith("hiddenpower") and move != "hiddenpower"
        }
        self.assertGreater(len(typed), 5, f"expected many typed ids, got {sorted(typed)}")
        self.assertNotIn(
            "hiddenpower", {m for u in _SOURCE.universes.values() for v in u.variants for m in v.moves}
        )

    def test_an_untyped_reveal_degrades_safely_instead_of_excluding(self) -> None:
        """THE load-bearing guard, and the only scenario in this file that occurs in production.

        `_revealed_move_matches_variant` treats a bare `hiddenpower` as matching ANY typed
        variant, so the reveal yields NO narrowing rather than a wrong exclusion -- an absent
        belief costs less than a wrong one. But that is right only by construction: a matcher
        tightened to require an exact id would eliminate every real variant on all 130
        Hidden-Power-carrying species and collapse them to the inconsistent fallback.
        """
        from pokezero.randbat import _revealed_move_matches_variant

        typed_pool = {"hiddenpowergrass", "surf"}
        self.assertTrue(
            _revealed_move_matches_variant("Hidden Power", typed_pool),
            "an untyped reveal excluded a typed variant -- it must degrade, not eliminate",
        )
        # ...and leniency is not blanket permissiveness: a DIFFERENT typed id still excludes.
        self.assertTrue(_revealed_move_matches_variant("Hidden Power Grass", typed_pool))
        self.assertFalse(_revealed_move_matches_variant("Hidden Power Fire", typed_pool))

    def test_the_untyped_reveal_costs_a_solve_that_the_typed_form_would_have_bought(self) -> None:
        """Quantifies what gen3's masking actually costs belief, against a NO-REVEAL baseline
        (see `_candidates`). The earlier version used Tackle as the baseline move -- which no
        Charizard variant carries -- so it measured against the inconsistent-reveal fallback and
        coincided with the real answer through a second, unrelated mechanism.

        This is the information the belief system does NOT get. It is not a defect to fix; it is
        a bound to know, and it is why item and damage evidence carry more weight than they
        would if the type were public.
        """
        baseline = self._candidates(None)
        untyped = self._candidates("Hidden Power")
        typed = self._candidates("Hidden Power Grass")
        self.assertEqual(untyped, baseline, "untyped Hidden Power unexpectedly narrowed")
        self.assertLess(typed, untyped, "the typed form must narrow, or the cost claim is empty")

    def test_each_type_would_select_a_DIFFERENT_survivor_set(self) -> None:
        """Guards the matcher, not the protocol: if two types on one species produced the same
        survivors, the type would be getting dropped between parse and variant matching.

        Asserts survivor IDENTITY, not just count. A mutation swapping grass and flying upstream
        leaves both counts unchanged and would otherwise pass.
        """
        survivors = {}
        for hp_type in ("Grass", "Flying"):
            replay = parse_showdown_replay(
                [*OPENING, f"|move|p2a: Charizard|Hidden Power {hp_type}|p1a: Furret", "|turn|1"],
                battle_id="battle-gen3randombattle-hp",
            )
            engine = PublicBattleBeliefEngine.from_events(
                replay.public_events, format_id="gen3randombattle", set_source=_SOURCE
            )
            belief = engine.snapshot().side("p2")[0]
            survivors[hp_type] = {
                tuple(sorted(v.get("moves") or ())) for v in belief.candidate_variants
            }
            self.assertTrue(survivors[hp_type], f"Hidden Power {hp_type} left no survivors")
            for moves in survivors[hp_type]:
                self.assertIn(
                    f"hiddenpower{hp_type.lower()}",
                    moves,
                    f"a survivor of Hidden Power {hp_type} does not carry that move",
                )
        self.assertNotEqual(survivors["Grass"], survivors["Flying"])


if __name__ == "__main__":
    unittest.main()
