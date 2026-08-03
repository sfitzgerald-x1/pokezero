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
import os
import unittest

from pokezero.belief import PublicBattleBeliefEngine
from pokezero.randbat import load_gen3_randbat_source_cached
from pokezero.showdown import parse_showdown_replay

SHOWDOWN_ROOT = os.environ.get(
    "POKEZERO_SHOWDOWN_ROOT", "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
)

try:
    _SOURCE = load_gen3_randbat_source_cached(SHOWDOWN_ROOT)
except Exception:  # pragma: no cover - no checkout in this environment
    _SOURCE = None

OPENING = [
    "|start",
    "|switch|p1a: Furret|Furret, L88, M|100/100",
    "|switch|p2a: Charizard|Charizard, L82, M|100/100",
]


def _engine(lines, *, with_source: bool = False, narrow: bool = False):
    return PublicBattleBeliefEngine.from_events(
        parse_showdown_replay(lines, battle_id="battle-gen3randombattle-edge").public_events,
        format_id="gen3randombattle" if with_source else None,
        set_source=_SOURCE if with_source else None,
        item_belief_narrowing=narrow,
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


KNOCK_OFF = [
    "|move|p1a: Furret|Knock Off|p2a: Charizard",
    "|-enditem|p2a: Charizard|Salac Berry|[from] move: Knock Off|[of] p1a: Furret",
]
# Furret's Choice Band goes to Charizard; Charizard's Leftovers goes to Furret. Each -item
# line's SUBJECT is the new holder and the item it names is the PARTNER's assignment.
TRICK = [
    "|move|p1a: Furret|Trick|p2a: Charizard",
    "|-activate|p1a: Furret|move: Trick|[of] p2a: Charizard",
    "|-item|p2a: Charizard|Choice Band|[from] move: Trick",
    "|-item|p1a: Furret|Leftovers|[from] move: Trick",
]


@unittest.skipIf(_SOURCE is None, "needs a pokemon-showdown checkout")
class ItemMutationTest(unittest.TestCase):
    """Trick/Knock Off must not corrupt the candidate set with a non-generator item.

    Two facts hide behind ``item_mutated`` and only one of them justifies suppression:
    "what it holds NOW is not the generator's assignment" (always true after a mutation) and
    "what it WAS holding is now known with certainty" (true only when the same protocol line
    named it). The second is a legitimate matching key and lives in ``original_public_item``;
    honouring it rides the ``item_belief_narrowing`` switch, so every assertion below is
    written twice, once per switch state.
    """

    def _charizard(self, lines, *, narrow: bool = False):
        return _mon(
            _engine([*OPENING, *lines, "|turn|1"], with_source=True, narrow=narrow),
            "p2",
            "Charizard",
        )

    def _furret(self, lines, *, narrow: bool = False):
        return _mon(
            _engine([*OPENING, *lines, "|turn|1"], with_source=True, narrow=narrow),
            "p1",
            "Furret",
        )

    def test_baseline_is_unnarrowed(self) -> None:
        self.assertEqual(self._charizard([]).candidate_set_count, 6)

    def test_a_genuine_item_reveal_narrows(self) -> None:
        """Control: an eaten berry is the generator's own assignment, so it must narrow."""
        mon = self._charizard(["|-enditem|p2a: Charizard|Salac Berry|[eat]"])
        self.assertEqual(mon.candidate_set_count, 1)
        self.assertFalse(mon.item_mutated)

    def test_tricked_item_does_not_corrupt_the_candidate_set(self) -> None:
        """A Choice Band Tricked ONTO a mon that could legitimately carry one.

        The dangerous outcome would be narrowing to the Choice Band variant on the strength
        of an item the generator never assigned. Charizard genuinely has a Choice Band
        variant, so a receiver that honoured its new item would pin the WRONG one — with
        four wrong moves attached — and look confident doing it.

        Unchanged by the switch, and that is the point: the received item lands in
        ``revealed_item``/``current_public_item``, never in ``original_public_item``, so it
        is not a matching key in either state. What DOES move here is the rest of the
        candidate set, because the partner line reveals Charizard's real item — see
        ``test_trick_narrows_the_target_from_the_partner_line``. This test therefore isolates
        the receiver rule by dropping the partner line.
        """
        received_only = [line for line in TRICK if not line.startswith("|-item|p1a: Furret|")]
        for narrow in (False, True):
            with self.subTest(item_belief_narrowing=narrow):
                mon = self._charizard(received_only, narrow=narrow)
                self.assertTrue(mon.item_mutated)
                self.assertEqual(mon.current_public_item, "Choice Band")
                self.assertIsNone(mon.original_public_item)
                # Not narrowed, and not collapsed to the inconsistent fallback either.
                self.assertEqual(mon.candidate_set_count, 6)
                self.assertEqual(mon.uncertainty, 1.0)

    def test_knock_off_records_the_original_item(self) -> None:
        """``|-enditem|<target>|<ITEM>|[from] move: Knock Off`` names the TARGET's own item.

        ``data/moves.ts`` knockoff.onAfterHit takes the item from the mon it just hit and
        prints it, so the removal is simultaneously a certain reveal of the generator's
        assignment. Recording that is unconditional; only USING it is switched.
        """
        for narrow in (False, True):
            with self.subTest(item_belief_narrowing=narrow):
                mon = self._charizard(KNOCK_OFF, narrow=narrow)
                self.assertTrue(mon.item_mutated)
                self.assertTrue(mon.item_removed)
                self.assertEqual(mon.original_public_item, "Salac Berry")

    def test_knock_off_narrows_to_the_true_variant_when_enabled(self) -> None:
        """The knocked-off item IS the assignment, so it pins the same variant an eaten
        Salac Berry does (test_a_genuine_item_reveal_narrows, same item value)."""
        mon = self._charizard(KNOCK_OFF, narrow=True)
        self.assertEqual(mon.candidate_set_count, 1)
        self.assertEqual(mon.possible_items, ("Salac Berry",))
        self.assertLess(mon.uncertainty, 1.0)

    def test_knock_off_stays_suppressed_with_the_switch_off(self) -> None:
        """The pre-switch behaviour, kept pinned.

        Before ``item_belief_narrowing`` existed, ``item_mutated`` gated the entire
        ``revealed_item`` channel off and this reveal was discarded. That is still what the
        default resolves to, byte for byte, because three lineages are training against these
        encodes right now.
        """
        mon = self._charizard(KNOCK_OFF, narrow=False)
        baseline = self._charizard([])
        self.assertEqual(mon.candidate_set_count, 6)
        self.assertEqual(mon.uncertainty, 1.0)
        self.assertEqual(mon.possible_items, baseline.possible_items)
        self.assertEqual(mon.possible_moves, baseline.possible_moves)
        self.assertEqual(mon.candidate_variants, baseline.candidate_variants)

    def test_trick_narrows_the_target_from_the_partner_line(self) -> None:
        """The redirect: ``|-item|<source>|<ITEM>|`` names the TARGET's original item.

        trick.onHit hands ``yourItem`` (the target's) to the source and ``myItem`` (the
        source's) to the target, so BOTH -item lines are cross-attributed exactly like the
        Traced ability line. Charizard's real assignment is the Leftovers announced on
        Furret's line, and honouring it cuts 6 -> 1 even though Charizard is now visibly
        holding a Choice Band.
        """
        mon = self._charizard(TRICK, narrow=True)
        self.assertEqual(mon.original_public_item, "Leftovers")
        self.assertEqual(mon.current_public_item, "Choice Band")
        self.assertEqual(mon.candidate_set_count, 1)
        self.assertEqual(mon.possible_items, ("Leftovers",))

    def test_trick_narrows_the_source_too_and_never_the_line_subject(self) -> None:
        """The other direction of the same swap, and the cross-attribution guard.

        Furret's assignment is the Choice Band announced on CHARIZARD's line. Crediting each
        line's subject instead would be wrong in both directions at once — the tell is that
        Charizard's own -item line must not leave Charizard holding a Choice Band as its
        assignment.
        """
        self.assertEqual(self._furret(TRICK, narrow=True).original_public_item, "Choice Band")
        self.assertNotEqual(
            self._charizard(TRICK, narrow=True).original_public_item, "Choice Band"
        )

    def test_an_unpaired_trick_line_records_nothing_rather_than_the_subject(self) -> None:
        """The -item lines carry no ``[of]`` tag, so the pairing comes from the -activate line.

        With no pairing available the attribution is impossible, and falling back to the
        subject is precisely the bug the redirect exists to avoid.
        """
        unpaired = [line for line in TRICK if not line.startswith("|-activate|")]
        mon = self._charizard(unpaired, narrow=True)
        self.assertIsNone(mon.original_public_item)
        self.assertEqual(mon.candidate_set_count, 6)

    def test_a_second_trick_cannot_overwrite_the_first_reveal(self) -> None:
        """A mon already carrying somebody else's item names THAT item on the swap line.

        Only the first swap sees a generator assignment, so the later ones must record
        nothing — for either participant — rather than publishing a laundered item as an
        assignment.
        """
        mon = self._charizard(
            [
                *TRICK,
                "|turn|2",
                "|move|p1a: Furret|Trick|p2a: Charizard",
                "|-activate|p1a: Furret|move: Trick|[of] p2a: Charizard",
                "|-item|p2a: Charizard|Leftovers|[from] move: Trick",
                "|-item|p1a: Furret|Choice Band|[from] move: Trick",
            ],
            narrow=True,
        )
        self.assertEqual(mon.original_public_item, "Leftovers")
        self.assertEqual(mon.candidate_set_count, 1)

    def test_knocking_off_a_tricked_item_reveals_nothing(self) -> None:
        """The removal names what it holds, which after a Trick is not its assignment."""
        mon = self._charizard(
            [
                *TRICK,
                "|turn|2",
                "|switch|p1a: Sableye|Sableye, L88, M|100/100",
                "|move|p1a: Sableye|Knock Off|p2a: Charizard",
                "|-enditem|p2a: Charizard|Choice Band|[from] move: Knock Off|[of] p1a: Sableye",
            ],
            narrow=True,
        )
        # The Trick's Leftovers survives; the knocked-off Choice Band is refused outright.
        self.assertEqual(mon.original_public_item, "Leftovers")

    def test_a_trick_that_returns_no_item_still_reveals_the_subjects_own(self) -> None:
        """When the source holds nothing, trick.onHit emits a SELF-attributed silent enditem.

        ``|-enditem|<target>|<yourItem>|[silent]|[from] move: Trick`` — unlike the -item
        lines this one names the subject's own item, so it takes the Knock Off path.
        """
        mon = self._charizard(
            [
                "|move|p1a: Furret|Trick|p2a: Charizard",
                "|-activate|p1a: Furret|move: Trick|[of] p2a: Charizard",
                "|-enditem|p2a: Charizard|Petaya Berry|[silent]|[from] move: Trick",
            ],
            narrow=True,
        )
        self.assertEqual(mon.original_public_item, "Petaya Berry")
        self.assertTrue(mon.item_removed)
        self.assertEqual(mon.candidate_set_count, 1)


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


@unittest.skipIf(_SOURCE is None, "needs a pokemon-showdown checkout")
class SwitchOffIsByteIdenticalTest(unittest.TestCase):
    """The load-bearing guard: with the switch off the ENCODE must not move.

    Honouring a certain item reveal narrows candidate sets that previously stayed wide, and
    the belief state feeds ``NUMERIC_CANDIDATE_SET_COUNT`` (5) and ``NUMERIC_UNCERTAINTY`` (6)
    — frozen legacy positions present in every schema — plus the possible-* counts. Three
    lineages are training against these encodes right now, so "off" has to mean bit-for-bit
    unchanged, not merely "roughly the same".

    The control is Trick with its ``-activate`` pairing line removed. That is the one surface
    where the reveal can be made genuinely unavailable without touching anything else the
    encoder reads: the two ``-item`` lines carry no ``[of]`` tag, so with no pairing in hand
    the engine records nothing and the belief is exactly the pre-change belief. (Knock Off has
    no such degenerate form — it names its own subject — which is precisely why the gate has
    to be a switch there rather than a missing attribution.) The second test proves this
    comparison has teeth: with the switch ON the same pair must diverge.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary

        cls.dex = load_showdown_dex_cached(SHOWDOWN_ROOT)
        cls.vocab = gen3_category_vocabulary(SHOWDOWN_ROOT)

    def _encode(self, lines, *, narrow: bool):
        """Encode under v2.1 — the widest schema whose encode needs no turn-merged stream.

        The belief columns at issue are frozen legacy positions: identical indices under v2,
        v2.1 and v2.2 and carried into the v3/v4 grouped projections.
        """
        from pokezero.showdown import (
            V2_1_REPLAY_OBSERVATION_SPEC,
            normalize_for_player,
            observation_from_player_state,
        )

        replay = parse_showdown_replay(
            [*OPENING, *lines, "|turn|1"], battle_id="battle-gen3randombattle-edge"
        )
        engine = PublicBattleBeliefEngine.from_events(
            replay.public_events,
            format_id="gen3randombattle",
            set_source=_SOURCE,
            item_belief_narrowing=narrow,
        )
        state = normalize_for_player(
            replay,
            player_id="p1",
            configured_showdown_slot="p1",
            format_id="gen3randombattle",
            set_source=_SOURCE,
            belief_engine=engine,
        )
        return observation_from_player_state(
            state,
            category_vocab=self.vocab,
            dex=self.dex,
            spec=V2_1_REPLAY_OBSERVATION_SPEC,
        )

    UNPAIRED = [line for line in TRICK if not line.startswith("|-activate|")]

    def test_the_reveal_is_actually_available_with_the_switch_off(self) -> None:
        """Non-vacuity: the fact IS learned, it is simply not used."""
        paired = _mon(_engine([*OPENING, *TRICK], with_source=True, narrow=False), "p2", "Charizard")
        unpaired = _mon(
            _engine([*OPENING, *self.UNPAIRED], with_source=True, narrow=False), "p2", "Charizard"
        )
        self.assertEqual(paired.original_public_item, "Leftovers")
        self.assertIsNone(unpaired.original_public_item)
        self.assertEqual(paired.candidate_set_count, unpaired.candidate_set_count)

    def test_switch_off_encodes_byte_identically_to_a_belief_that_cannot_know(self) -> None:
        known = self._encode(TRICK, narrow=False)
        unknowable = self._encode(self.UNPAIRED, narrow=False)
        self.assertEqual(known.numeric_features, unknowable.numeric_features)
        self.assertEqual(known.categorical_ids, unknowable.categorical_ids)

    def test_the_same_comparison_diverges_with_the_switch_on(self) -> None:
        """Without this the equality above would pass even if the switch did nothing."""
        known = self._encode(TRICK, narrow=True)
        unknowable = self._encode(self.UNPAIRED, narrow=True)
        self.assertNotEqual(known.numeric_features, unknowable.numeric_features)

    def test_knock_off_moves_the_encode_only_when_enabled(self) -> None:
        off = self._encode(KNOCK_OFF, narrow=False)
        on = self._encode(KNOCK_OFF, narrow=True)
        self.assertNotEqual(off.numeric_features, on.numeric_features)


if __name__ == "__main__":
    unittest.main()
