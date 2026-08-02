"""Choice Band is ruled out by two different moves in one stay on the field.

``data/conditions.ts`` choicelock records ``activeMove.id`` in ``onStart`` and fails any
later move whose id differs in ``onBeforeMove``; ``choiceband.onModifyMove`` adds the
volatile on the holder's first move and ``choiceband.onStart`` removes it on switch-in.
``data/mods/gen3`` overrides neither, so gen3 inherits the lock verbatim and a mon seen
SELECTING two different moves without leaving the field cannot have been holding one.

This is the negative direction only, and it joins the existing non-proc pruning family
(Leftovers / Lum / pinch berries) rather than starting a new one: certain, free, and derived
from public ``|move|`` lines with no damage analysis and hence no precision gate. Choice Band
is the pool's second most common item — 160 variants against Leftovers' 1371 — so it is the
only member of that family that moves a large share of the pool.

What must NOT count as a second selection, each covered below: a switch (the lock resets), a
``[from]lockedmove`` continuation, a Sleep-Talk-CALLED move (Sleep Talk chose it, not the
player), and Struggle (forced, and explicitly exempted by choicelock itself).

Probe species, all checked for reachability in the gen3 randbats pool:
  * Charizard — 6 variants over 5 items INCLUDING Choice Band, so the rule-out is visible in
    ``possible_items`` and ``candidate_set_count``, not just in ``ruled_out_items``;
  * Exeggutor — carries the pool's ONLY two-turn move (Solar Beam; Thrash, Outrage, Petal
    Dance, Rollout, Uproar, Fly, Dig and Razor Wind are all absent from gen3 randbats);
  * Snorlax — Rest + Sleep Talk, the pool's called-move surface.
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

LEAD = ["|start", "|switch|p1a: Furret|Furret, L88, M|100/100"]
CHARIZARD_IN = "|switch|p2a: Charizard|Charizard, L82, M|100/100"

# Both are real Charizard set members, and they co-occur on the Choice Band variant — so the
# rule-out is not doing the work the move reveals already did.
TWO_SELECTIONS = [
    CHARIZARD_IN,
    "|move|p2a: Charizard|Earthquake|p1a: Furret",
    "|turn|2",
    "|move|p2a: Charizard|Rock Slide|p1a: Furret",
]


def _engine(lines, *, narrow: bool):
    return PublicBattleBeliefEngine.from_events(
        parse_showdown_replay(
            [*LEAD, *lines, "|turn|9"], battle_id="battle-gen3randombattle-choicelock"
        ).public_events,
        format_id="gen3randombattle",
        set_source=_SOURCE,
        item_belief_narrowing=narrow,
    )


def _mon(lines, species, *, narrow: bool = True):
    for belief in _engine(lines, narrow=narrow).snapshot().side("p2"):
        if belief.species.lower().startswith(species.lower()):
            return belief
    raise AssertionError(f"no {species} on p2")


@unittest.skipIf(_SOURCE is None, "needs a pokemon-showdown checkout")
class ChoiceLockRuleOutTest(unittest.TestCase):
    def test_two_different_selections_in_one_stay_rule_choice_band_out(self) -> None:
        narrowed = _mon(TWO_SELECTIONS, "Charizard", narrow=True)
        plain = _mon(TWO_SELECTIONS, "Charizard", narrow=False)
        self.assertIn("choiceband", narrowed.ruled_out_items)
        self.assertNotIn("Choice Band", narrowed.possible_items)
        # The move reveals alone leave the Choice Band variant standing; the lock removes it.
        self.assertIn("Choice Band", plain.possible_items)
        self.assertEqual(narrowed.candidate_set_count, plain.candidate_set_count - 1)
        self.assertLess(narrowed.uncertainty, plain.uncertainty)

    def test_the_same_move_twice_rules_nothing_out(self) -> None:
        """The control: repetition is exactly what a locked mon looks like."""
        mon = _mon(
            [
                CHARIZARD_IN,
                "|move|p2a: Charizard|Earthquake|p1a: Furret",
                "|turn|2",
                "|move|p2a: Charizard|Earthquake|p1a: Furret",
            ],
            "Charizard",
        )
        self.assertEqual(mon.ruled_out_items, ())
        self.assertIn("Choice Band", mon.possible_items)

    def test_a_switch_between_the_two_selections_resets_the_lock(self) -> None:
        """``choiceband.onStart`` removes the volatile, so each stay locks independently."""
        mon = _mon(
            [
                CHARIZARD_IN,
                "|move|p2a: Charizard|Earthquake|p1a: Furret",
                "|turn|2",
                "|switch|p2a: Snorlax|Snorlax, L79, M|100/100",
                "|turn|3",
                CHARIZARD_IN,
                "|turn|4",
                "|move|p2a: Charizard|Rock Slide|p1a: Furret",
            ],
            "Charizard",
        )
        self.assertEqual(mon.ruled_out_items, ())
        self.assertIn("Choice Band", mon.possible_items)

    def test_struggle_does_not_break_the_lock(self) -> None:
        """choicelock's onBeforeMove exempts Struggle by id, and it is never a set member."""
        mon = _mon(
            [
                CHARIZARD_IN,
                "|move|p2a: Charizard|Earthquake|p1a: Furret",
                "|turn|2",
                "|move|p2a: Charizard|Struggle|p1a: Furret",
            ],
            "Charizard",
        )
        self.assertEqual(mon.ruled_out_items, ())

    def test_a_locked_continuation_is_the_same_move_continuing(self) -> None:
        """Solar Beam's release carries ``[from]lockedmove`` and is not a fresh selection.

        The paired assertion keeps this honest: the very same Exeggutor DOES rule Choice Band
        out once it genuinely selects twice, so the negative above is not a dead probe.
        """
        charging = _mon(
            [
                "|switch|p2a: Exeggutor|Exeggutor, L79, M|100/100",
                "|move|p2a: Exeggutor|Solar Beam||[still]",
                "|-prepare|p2a: Exeggutor|Solar Beam",
                "|turn|2",
                "|move|p2a: Exeggutor|Solar Beam|p1a: Furret|[from]lockedmove",
            ],
            "Exeggutor",
        )
        self.assertEqual(charging.ruled_out_items, ())

        selecting = _mon(
            [
                "|switch|p2a: Exeggutor|Exeggutor, L79, M|100/100",
                "|move|p2a: Exeggutor|Psychic|p1a: Furret",
                "|turn|2",
                "|move|p2a: Exeggutor|Giga Drain|p1a: Furret",
            ],
            "Exeggutor",
        )
        self.assertIn("choiceband", selecting.ruled_out_items)

    def test_a_sleep_talk_called_move_is_not_a_free_selection(self) -> None:
        """Sleep Talk picked the callee, so the callee cannot testify about the lock.

        Two DIFFERENT called moves here — Body Slam and Curse — against one repeated free
        selection. If the callee counted, this would rule Choice Band out.
        """
        mon = _mon(
            [
                "|switch|p2a: Snorlax|Snorlax, L79, M|100/100 slp",
                "|move|p2a: Snorlax|Sleep Talk|p2a: Snorlax",
                "|move|p2a: Snorlax|Body Slam|p1a: Furret|[from]Sleep Talk",
                "|turn|2",
                "|move|p2a: Snorlax|Sleep Talk|p2a: Snorlax",
                "|move|p2a: Snorlax|Curse|p2a: Snorlax|[from]Sleep Talk",
            ],
            "Snorlax",
        )
        self.assertEqual(mon.ruled_out_items, ())
        # Non-vacuity: the called moves really were seen, they simply did not testify.
        self.assertIn("Body Slam", mon.revealed_moves)
        self.assertIn("Curse", mon.revealed_moves)

    def test_the_rule_is_frozen_once_the_held_item_is_mutated(self) -> None:
        """After a Trick the mon is locked (or not) by somebody ELSE's item.

        Same freeze the rest of the non-proc family observes: pruning describes the original
        assignment, and once that is gone the move stream stops being evidence about it.
        """
        mon = _mon(
            [
                CHARIZARD_IN,
                "|move|p1a: Furret|Trick|p2a: Charizard",
                "|-activate|p1a: Furret|move: Trick|[of] p2a: Charizard",
                "|-item|p2a: Charizard|Choice Band|[from] move: Trick",
                "|-item|p1a: Furret|Leftovers|[from] move: Trick",
                "|turn|2",
                "|move|p2a: Charizard|Earthquake|p1a: Furret",
                "|turn|3",
                "|move|p2a: Charizard|Rock Slide|p1a: Furret",
            ],
            "Charizard",
        )
        self.assertEqual(mon.ruled_out_items, ())


@unittest.skipIf(_SOURCE is None, "needs a pokemon-showdown checkout")
class SwitchOffIsByteIdenticalTest(unittest.TestCase):
    """With the switch off the encode must not move, because three lineages are training.

    ``ruled_out_items`` is the ONLY state this rule can write, and the parent commit had no
    Choice Band rule at all, so its value there was ``()`` on every battle. Asserting that the
    switch-off engine still produces ``()`` — and that every surface derived from it matches
    an engine that never saw a second selection — is therefore a complete statement of
    byte-identity with the parent, not a proxy for one. The second test is what keeps it from
    being vacuous: with the switch ON the same battle must move the encoded belief columns.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary

        cls.dex = load_showdown_dex_cached(SHOWDOWN_ROOT)
        cls.vocab = gen3_category_vocabulary(SHOWDOWN_ROOT)

    def _encode(self, lines, *, narrow: bool):
        from pokezero.showdown import (
            V2_1_REPLAY_OBSERVATION_SPEC,
            normalize_for_player,
            observation_from_player_state,
        )

        replay = parse_showdown_replay(
            [*LEAD, *lines, "|turn|9"], battle_id="battle-gen3randombattle-choicelock"
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

    def test_switch_off_writes_nothing_and_leaves_every_surface_at_the_parent_value(self) -> None:
        off = _mon(TWO_SELECTIONS, "Charizard", narrow=False)
        self.assertEqual(off.ruled_out_items, ())
        self.assertIn("Choice Band", off.possible_items)
        # The bookkeeping DID run — the conclusion is available, it is simply not applied.
        on = _mon(TWO_SELECTIONS, "Charizard", narrow=True)
        self.assertEqual(on.ruled_out_items, ("choiceband",))

    def test_the_encoded_belief_columns_move_only_when_enabled(self) -> None:
        off = self._encode(TWO_SELECTIONS, narrow=False)
        on = self._encode(TWO_SELECTIONS, narrow=True)
        self.assertNotEqual(off.numeric_features, on.numeric_features)

    def test_a_battle_the_rule_cannot_conclude_on_encodes_identically_either_way(self) -> None:
        """Sensitivity control: with no second selection the switch is a no-op end to end."""
        one_selection = [
            CHARIZARD_IN,
            "|move|p2a: Charizard|Earthquake|p1a: Furret",
            "|turn|2",
            "|move|p2a: Charizard|Earthquake|p1a: Furret",
        ]
        off = self._encode(one_selection, narrow=False)
        on = self._encode(one_selection, narrow=True)
        self.assertEqual(off.numeric_features, on.numeric_features)
        self.assertEqual(off.categorical_ids, on.categorical_ids)


if __name__ == "__main__":
    unittest.main()
