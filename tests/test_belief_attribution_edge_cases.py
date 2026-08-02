"""Belief attribution edge cases: called moves, Transform, and item mutation.

These all share one failure mode — a protocol line whose SUBJECT is not the mon the fact
belongs to. Trace was one instance (see test_belief_trace_attribution); these pin the
behaviour of the rest so a future change has to break a test to change a semantic.

Everything here is checked for REACHABILITY in the gen3 randbats pool first. Skill Swap,
Role Play, Mimic, Metronome, Assist and Imprison are absent from the pool entirely and are
deliberately not covered. Frisk does not exist in gen3, so there is no benign ``-item``
reveal: every reachable ``-item`` line is a Trick/Thief/Covet mutation.

Charizard is the probe species throughout — 6 variants over 5 distinct items, so an item
that is honoured for variant matching cuts the candidate set 6 -> 1, and one that is
suppressed leaves it at 6. That makes "did this reveal narrow?" directly observable.
"""
import unittest

from pokezero.belief import PublicBattleBeliefEngine
from pokezero.randbat import load_gen3_randbat_source_cached
from pokezero.showdown import parse_showdown_replay

SHOWDOWN_ROOT = "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"

try:
    _SOURCE = load_gen3_randbat_source_cached(SHOWDOWN_ROOT)
except Exception:  # pragma: no cover - no checkout in this environment
    _SOURCE = None

OPENING = [
    "|start",
    "|switch|p1a: Furret|Furret, L88, M|100/100",
    "|switch|p2a: Charizard|Charizard, L82, M|100/100",
]


def _engine(lines, *, with_source: bool = False):
    return PublicBattleBeliefEngine.from_events(
        parse_showdown_replay(lines, battle_id="battle-gen3randombattle-edge").public_events,
        format_id="gen3randombattle" if with_source else None,
        set_source=_SOURCE if with_source else None,
    )


def _mon(engine, slot, species):
    for mon in engine.snapshot().side(slot):
        if mon.species.lower().startswith(species.lower()):
            return mon
    raise AssertionError(f"no {species} on {slot}")


def _moves(mon) -> set[str]:
    return {m.lower().replace(" ", "") for m in mon.revealed_moves}


class CalledMoveTest(unittest.TestCase):
    """Sleep Talk draws from the user's OWN moveset, so the called move is a real reveal."""

    def test_sleep_talk_called_move_is_recorded(self) -> None:
        engine = _engine(
            [
                "|start",
                "|switch|p1a: Furret|Furret, L88, M|100/100",
                "|switch|p2a: Snorlax|Snorlax, L79, M|100/100",
                "|move|p2a: Snorlax|Body Slam|p1a: Furret|[from]Sleep Talk",
                "|turn|1",
            ]
        )
        # Not skipped as "called": unlike Trace/Transform the move genuinely belongs to
        # the user, so discarding it would throw away a certain reveal.
        self.assertIn("bodyslam", _moves(_mon(engine, "p2", "Snorlax")))


class TransformTest(unittest.TestCase):
    """Transform borrows the TARGET's moves; they are not the transformer's set."""

    LINES = [
        "|start",
        "|switch|p1a: Furret|Furret, L88, M|100/100",
        "|switch|p2a: Ditto|Ditto, L83|100/100",
        "|-transform|p2a: Ditto|p1a: Furret",
        "|move|p2a: Ditto|Quick Attack|p1a: Furret",
        "|turn|1",
    ]

    def test_borrowed_move_is_not_credited_to_the_transformer(self) -> None:
        ditto = _mon(_engine(self.LINES), "p2", "Ditto")
        self.assertNotIn("quickattack", _moves(ditto))

    def test_transform_identity_is_tracked(self) -> None:
        ditto = _mon(_engine(self.LINES), "p2", "Ditto")
        self.assertTrue(ditto.transformed)
        self.assertTrue((ditto.transform_species or "").lower().startswith("furret"))


@unittest.skipIf(_SOURCE is None, "needs a pokemon-showdown checkout")
class ItemMutationTest(unittest.TestCase):
    """Trick/Knock Off must not corrupt the candidate set with a non-generator item."""

    def _charizard(self, lines):
        return _mon(_engine([*OPENING, *lines, "|turn|1"], with_source=True), "p2", "Charizard")

    def test_baseline_is_unnarrowed(self) -> None:
        self.assertEqual(self._charizard([]).candidate_set_count, 6)

    def test_a_genuine_item_reveal_narrows(self) -> None:
        """Control: an eaten berry is the generator's own assignment, so it must narrow."""
        mon = self._charizard(["|-enditem|p2a: Charizard|Salac Berry|[eat]"])
        self.assertEqual(mon.candidate_set_count, 1)
        self.assertFalse(mon.item_mutated)

    def test_tricked_item_does_not_corrupt_the_candidate_set(self) -> None:
        """A Choice Band Tricked ONTO a mon that could legitimately carry one.

        The dangerous outcome would be narrowing to the Choice Band variant on the
        strength of an item the generator never assigned. It must stay unnarrowed
        instead — and must NOT collapse to the inconsistent fallback either.
        """
        mon = self._charizard(
            [
                "|move|p1a: Furret|Trick|p2a: Charizard",
                "|-activate|p1a: Furret|move: Trick|[of] p2a: Charizard",
                "|-item|p2a: Charizard|Choice Band|[from] move: Trick",
                "|-item|p1a: Furret|Leftovers|[from] move: Trick",
            ]
        )
        self.assertTrue(mon.item_mutated)
        self.assertEqual(mon.candidate_set_count, 6)
        self.assertEqual(mon.uncertainty, 1.0)

    def test_knock_off_marks_mutation_and_suppresses_matching(self) -> None:
        """Current semantics, pinned deliberately.

        The knocked-off item IS the generator's assignment, so honouring it would cut
        6 -> 1 (see test_a_genuine_item_reveal_narrows, same item value). It is suppressed
        because ``item_mutated`` gates the whole revealed_item channel off. That is the
        conservative choice, not an obviously correct one — this test exists so that
        changing it is a deliberate act with a visible diff.
        """
        mon = self._charizard(
            [
                "|move|p1a: Furret|Knock Off|p2a: Charizard",
                "|-enditem|p2a: Charizard|Salac Berry|[from] move: Knock Off|[of] p1a: Furret",
            ]
        )
        self.assertTrue(mon.item_mutated)
        self.assertTrue(mon.item_removed)
        self.assertEqual(mon.candidate_set_count, 6)


@unittest.skipIf(_SOURCE is None, "needs a pokemon-showdown checkout")
class InteractionItemInferenceTest(unittest.TestCase):
    """gen3 has no Frisk: items are inferred from INTERACTIONS, in both directions.

    A Leftovers heal reveals the item outright; a residual phase that passes with the mon
    below full HP and no heal rules it out. The negative direction is the subtler one and
    the easier to regress, since it depends on gen3's residual ORDER — the Leftovers slot
    runs before status/Leech chip, so a mon chipped only during residuals was at full HP
    when its slot ran and yields no evidence either way.
    """

    def _charizard(self, lines):
        return _mon(_engine([*OPENING, *lines], with_source=True), "p2", "Charizard")

    def test_leftovers_proc_reveals_the_item(self) -> None:
        mon = self._charizard(
            [
                "|move|p1a: Furret|Quick Attack|p2a: Charizard",
                "|-damage|p2a: Charizard|70/100",
                "|-heal|p2a: Charizard|76/100|[from] item: Leftovers",
                "|upkeep",
                "|turn|2",
            ]
        )
        self.assertEqual(mon.candidate_set_count, 1)
        self.assertEqual(mon.possible_items, ("Leftovers",))

    def test_a_residual_phase_without_a_heal_rules_leftovers_out(self) -> None:
        mon = self._charizard(
            [
                "|move|p1a: Furret|Quick Attack|p2a: Charizard",
                "|-damage|p2a: Charizard|70/100",
                "|turn|2",
                "|move|p1a: Furret|Quick Attack|p2a: Charizard",
                "|-damage|p2a: Charizard|50/100",
                "|upkeep",
                "|turn|3",
            ]
        )
        self.assertIn("leftovers", mon.ruled_out_items)
        self.assertEqual(mon.candidate_set_count, 5)
        self.assertNotIn("Leftovers", mon.possible_items)


if __name__ == "__main__":
    unittest.main()
