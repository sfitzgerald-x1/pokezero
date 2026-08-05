"""Leftovers non-proc pruning must not rule out the TRUE item when it had no opportunity.

`ruled_out_items` is documented as "deterministic non-proc pruning results", and the candidate
filter consumes it, so a wrong entry is not a loose belief -- it is a belief that has EXCLUDED the
truth. Two ways that happened, both found by the V1 coherence sweep
(`scripts/belief_coherence_gate.py`, 565 violations over 250 games):

1. **A non-Leftovers heal reached full HP before the Leftovers slot.** Wish is the common case and
   it is engine-ordered first. Orders must be read from the GEN3-EFFECTIVE chain, since
   `data/mods/gen3/scripts.ts` is `inherit: 'gen4'` and gen4 overrides both: wish
   `onResidualOrder: 7` (`data/mods/gen4/moves.ts`), leftovers `10 / subOrder 4`
   (`data/mods/gen4/items.ts`), Leech Seed `10/5`, brn/psn/tox `10/6`. The shared `data/moves.ts` /
   `data/items.ts` values (4 and 5) are gen5+ and are NOT what gen3 runs.
2. **A stale pre-residual snapshot.** (Called "the root cause" here once. It was not -- see 3.)
   `_hp_after_actions` was not cleared per
   turn, so a mon damaged on an earlier turn that takes no action-phase hit on this one was still
   judged against the old snapshot. The code's own comment states the policy as "no snapshot => no
   evidence (conservative)", but a STALE snapshot is not no-snapshot. It is now cleared at
   `|upkeep`.

3. **A residual HP change classified as action-phase because it carried no `[from]` tag.** The
   Leech Seed drain heal on the DRAINER is emitted `[silent]` with no source tag at all
   (`sim/battle.ts:2293-2296`), so a tag-based classifier read it as action-phase and overwrote the
   pre-slot snapshot with a residual-phase HP. 85 violations at 400 games on seed 31337, 65 on
   555001, all family 5 `ruled_out_item`. Leech Seed is on 12 pool species.

   The fix is not another tag: the same engine branch emits `[silent]` for REST, which really is
   action-phase and is on 46 pool species, so a blanket `[silent]` rule trades one unsound
   elimination for another. The engine's own action->residual boundary marker -- the bare `|` line
   from `case 'residual': this.add('')` -- separates them exactly, and an unenumerated residual
   source now DECLINES rather than reading as action-phase.

The consequence measured in live games: for a species whose every variant holds Leftovers
(Octillery) the rule-out emptied the candidate set, forced the inconsistent fallback and pinned
`uncertainty` to 1.0; one turn later the same belief carried `revealed_item="Leftovers"` AND
`ruled_out_items=("leftovers",)` at the same time. On a mixed-item species it drops the true variant
instead, which is the unrecoverable direction.

The tests that keep this honest are the SOUND-ELIMINATION ones at the bottom: the rule must still
fire whenever the mon really was below full at its own 10/4 slot, or the "fix" is just the pruning
switched off. Two of them exist because a previous version of this fix over-declined -- it keyed off
"healed to full at any point this turn" instead of "was full when the slot ran", and so refused two
eliminations that the engine's ordering makes sound.
"""

from __future__ import annotations

import unittest

from pokezero.belief import PublicBattleBeliefEngine, RevealedPokemonBelief
from pokezero.showdown import parse_showdown_replay


# The sim's action -> residual boundary: ``sim/battle.ts:2836``, ``case 'residual': this.add('')``.
# Named rather than inlined because a bare "|" in a fixture reads like a typo, and because every
# fixture that omits it is testing the phase-independent tag path instead (which is a real path --
# see ``test_a_post_slot_tag_is_honoured_even_without_the_phase_marker`` -- but a different one).
_RESIDUAL_MARKER = "|"


def _engine_from(lines: list[str]) -> PublicBattleBeliefEngine:
    replay = parse_showdown_replay(
        ["|player|p1|PokeZeroBot|1", "|player|p2|Rival|2", *lines], battle_id="b"
    )
    engine = PublicBattleBeliefEngine()
    for event in replay.public_events:
        engine.ingest_event(event)
    return engine


def _opponent(engine: PublicBattleBeliefEngine, species: str) -> RevealedPokemonBelief:
    for belief in engine.snapshot().sides["p2"]:
        if belief.species == species:
            return belief
    raise AssertionError(f"no belief for {species}")


class LeftoversNonProcSoundnessTest(unittest.TestCase):
    def test_wish_heal_to_full_does_not_rule_out_leftovers(self) -> None:
        """The exact live sequence that produced the first 565 violations.

        Octillery is damaged during the action phase, then Wish tops it back to full before the
        Leftovers slot. Leftovers has nothing to heal, so it is silent -- which is not evidence.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Stantler|Stantler, L88|300/300",
                "|switch|p2a: Octillery|Octillery, L87, F|272/272",
                "|turn|1",
                "|move|p1a: Stantler|Return|p2a: Octillery",
                "|-damage|p2a: Octillery|169/272",
                "|-heal|p2a: Octillery|272/272|[from] move: Wish|[wisher] Umbreon",
                "|upkeep",
            ]
        )
        octillery = _opponent(engine, "Octillery")
        self.assertNotIn(
            "leftovers",
            octillery.ruled_out_items,
            "Leftovers had no room to heal (Wish reached full first), so its silence is not "
            "evidence of absence",
        )

    def test_a_heal_to_full_by_any_source_blocks_the_rule_out(self) -> None:
        """The class, not just Wish: a self-heal to full removes the opportunity identically."""
        engine = _engine_from(
            [
                "|switch|p1a: Stantler|Stantler, L88|300/300",
                "|switch|p2a: Porygon2|Porygon2, L80|267/267",
                "|turn|1",
                "|move|p1a: Stantler|Return|p2a: Porygon2",
                "|-damage|p2a: Porygon2|180/267",
                "|move|p2a: Porygon2|Recover|p2a: Porygon2",
                "|-heal|p2a: Porygon2|267/267",
                "|upkeep",
            ]
        )
        self.assertNotIn("leftovers", _opponent(engine, "Porygon2").ruled_out_items)

    def test_a_stale_snapshot_from_an_earlier_turn_is_not_this_turns_evidence(self) -> None:
        """The live Togetic reproducer, end to end.

        Turn 1 damages it and Wish tops it off. On turn 2 it takes no action-phase HP change, is at
        full HP when its 10/4 slot runs, and only falls below full afterwards from toxic (10/6). The
        elimination the old code made was wrong.

        This test does NOT isolate the per-turn clear -- its docstring used to claim it did, and that
        claim was false: with the classifier fixed, turn 1's snapshot ends at 1.0 (Wish is pre-slot),
        so deleting the clear leaves this green. See
        ``test_the_pre_slot_snapshot_is_cleared_at_upkeep`` for the clear's own guard and for why a
        behavioural fixture cannot discriminate it.

        It is kept because it is the shape live data produced, and because it pins that the psn chip
        at 10/6 does not disturb the snapshot.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Togetic|Togetic, L84, F|264/264",
                "|switch|p2a: Rapidash|Rapidash, L82, M|244/244",
                "|turn|1",
                "|move|p2a: Rapidash|Return|p1a: Togetic",
                "|-damage|p1a: Togetic|135/264",
                "|-heal|p1a: Togetic|264/264|[from] move: Wish|[wisher] Blissey",
                "|upkeep",
                "|turn|2",
                "|move|p2a: Rapidash|Toxic|p1a: Togetic",
                "|-status|p1a: Togetic|tox",
                "|move|p1a: Togetic|Seismic Toss|p2a: Rapidash",
                "|-damage|p2a: Rapidash|147/244",
                "|-damage|p1a: Togetic|248/264 tox|[from] psn",
                "|upkeep",
            ]
        )
        togetic = [b for b in engine.snapshot().sides["p1"] if b.species == "Togetic"][0]
        self.assertIsNone(
            togetic.revealed_item,
            "precondition: item must not be revealed, or the sweep skips this mon",
        )
        self.assertNotIn(
            "leftovers",
            togetic.ruled_out_items,
            "a stale snapshot from turn 1 was used as turn 2's evidence",
        )

    def test_reaching_full_then_being_chipped_by_a_POST_slot_residual_declines(self) -> None:
        """Wish (7) to full, then toxic (10/6) chip: full at the slot, so silence is not evidence.

        The docstring here used to say this "isolates the HEAL-TO-FULL guard". It does not, and the
        guard it named is gone. What makes this sound is the ORDER: Wish resolves before the item
        slot so it updates the snapshot to 1.0, and psn resolves after it so it leaves the snapshot
        alone. The mon really was at full when Leftovers was offered.

        Kill-confirmed by moving ``[from] psn`` out of the at/after list (the elimination then
        fires), and separately by dropping ``[from] move: Wish`` from the pre-slot list (the snapshot
        is discarded instead of reaching 1.0).

        Contrast ``test_reaching_full_then_being_chipped_by_a_PRE_slot_residual_eliminates``: same
        shape, chip ordered before the slot instead of after, opposite and equally required verdict.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Stantler|Stantler, L88|300/300",
                "|switch|p2a: Octillery|Octillery, L87, F|272/272",
                "|turn|1",
                "|move|p1a: Stantler|Return|p2a: Octillery",
                "|-damage|p2a: Octillery|169/272",
                _RESIDUAL_MARKER,
                "|-heal|p2a: Octillery|272/272|[from] move: Wish|[wisher] Umbreon",
                "|-damage|p2a: Octillery|255/272|[from] psn",
                "|upkeep",
            ]
        )
        octillery = _opponent(engine, "Octillery")
        self.assertNotIn("leftovers", octillery.ruled_out_items)

    def test_revealed_item_is_never_simultaneously_ruled_out(self) -> None:
        """A belief must not hold "it is Leftovers" and "it is not Leftovers" at once.

        This self-contradiction was live: the sweep captured a belief with
        revealed_item="Leftovers" and ruled_out_items=("leftovers",) on the same mon.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Stantler|Stantler, L88|300/300",
                "|switch|p2a: Octillery|Octillery, L87, F|272/272",
                "|turn|1",
                "|move|p1a: Stantler|Return|p2a: Octillery",
                "|-damage|p2a: Octillery|169/272",
                "|-heal|p2a: Octillery|272/272|[from] move: Wish|[wisher] Umbreon",
                "|upkeep",
                "|turn|2",
                "|move|p1a: Stantler|Return|p2a: Octillery",
                "|-damage|p2a: Octillery|186/272",
                "|-heal|p2a: Octillery|203/272|[from] item: Leftovers",
                "|upkeep",
            ]
        )
        octillery = _opponent(engine, "Octillery")
        self.assertEqual(octillery.revealed_item, "Leftovers")
        self.assertNotIn("leftovers", octillery.ruled_out_items)

    def test_liquid_ooze_leech_seed_damage_is_not_pre_residual_evidence(self) -> None:
        """A residual-phase HP change misclassified as action-phase, found at 400 games.

        Liquid Ooze turns a LEECH SEED drain into damage on the drainer
        (``data/mods/gen4/abilities.ts``, ``canOoze = ['drain', 'leechseed']``; gen3 inherits gen4),
        and Leech Seed is residual 10/subOrder 5 -- AFTER the Leftovers slot at 10/4. Untagged, that
        damage overwrote the pre-residual snapshot, so a mon at FULL HP when Leftovers ran looked
        like it ended a damaged turn with no heal.

        This one broke CONTAINMENT rather than merely widening: Flygon is a mixed-item species, so
        the rule-out dropped the true variant instead of emptying the set, leaving a confidently
        wrong single-candidate pin.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Swalot|Swalot, L84, M|300/300",
                "|switch|p2a: Flygon|Flygon, L78, F|253/253",
                "|turn|1",
                "|move|p2a: Flygon|Substitute|p2a: Flygon",
                "|-damage|p1a: Swalot|10/300|[from] Leech Seed|[of] p2a: Flygon",
                "|-damage|p2a: Flygon|220/253|[from] ability: Liquid Ooze|[of] p1a: Swalot",
                "|upkeep",
            ]
        )
        flygon = [b for b in engine.snapshot().sides["p2"] if b.species == "Flygon"][0]
        self.assertNotIn(
            "leftovers",
            flygon.ruled_out_items,
            "Liquid Ooze damage from a Leech Seed drain lands AFTER the Leftovers slot, so it is "
            "not evidence about the pre-residual HP",
        )

    def test_a_genuine_non_proc_still_rules_leftovers_out(self) -> None:
        """The true positive must survive, or the fix is just the pruning switched off.

        Damaged during the action phase, ends the turn still below full, no Leftovers heal: the mon
        had a real opportunity and did not take it, so Leftovers is soundly eliminated. Without this
        the other four tests would pass on a no-op.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Stantler|Stantler, L88|300/300",
                "|switch|p2a: Octillery|Octillery, L87, F|272/272",
                "|turn|1",
                "|move|p1a: Stantler|Return|p2a: Octillery",
                "|-damage|p2a: Octillery|169/272",
                "|upkeep",
            ]
        )
        octillery = _opponent(engine, "Octillery")
        self.assertIn(
            "leftovers",
            octillery.ruled_out_items,
            "a real end-of-turn non-proc must still eliminate Leftovers",
        )

    # ------------------------------------------------------------------ the [silent] drain heal

    def test_the_silent_leech_seed_drain_heal_on_the_drainer_is_not_pre_slot_evidence(self) -> None:
        """The defect that survived the tag-based classifier: 85 violations / 400 games, seed 31337.

        ``Battle.heal`` emits the Leech Seed drain heal on the DRAINER with no source tag at all --
        ``sim/battle.ts:2293-2296``, ``case 'leechseed': case 'rest': this.add('-heal', target,
        target.getHealth, '[silent]')`` -- so a classifier that keys off ``[from]`` tags read it as
        action-phase and wrote a residual-phase HP into the pre-slot snapshot.

        The shape below is the one that turns that into a WRONG elimination. Flygon is at full HP and
        poisoned. In the residual phase, ordered by ``comparePriority``, Flygon's own handlers run as
        a group (Leftovers 10/4 first -- no room, so silent -- then psn 10/6), and the seeded Swalot's
        leechseed 10/5 heals Flygon back up part-way. Flygon was demonstrably FULL when its item slot
        was offered, so the silence is not evidence; the drain heal to 249/253 is what made it look
        like "ended a damaged turn with no heal".

        Flygon is a mixed-item species, so this broke CONTAINMENT rather than merely widening: it
        dropped the true variant and left a confidently wrong single-candidate pin.

        Kill-confirmed two ways, each failing this test alone: (1) delete the ``_HP_SNAPSHOT_DISCARD``
        return at the end of ``_hp_snapshot_action``, and (2) delete the bare-``|`` branch of
        ``_track_residual_phase`` so the phase is never entered.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Swalot|Swalot, L84, M|300/300",
                "|switch|p2a: Flygon|Flygon, L78, F|253/253",
                "|turn|1",
                "|move|p2a: Flygon|Leech Seed|p1a: Swalot",
                "|-start|p1a: Swalot|move: Leech Seed",
                "|-status|p2a: Flygon|psn",
                _RESIDUAL_MARKER,
                # Flygon's own group: 10/4 Leftovers is silent (full), 10/6 psn chips.
                "|-damage|p2a: Flygon|240/253 psn|[from] psn",
                # Swalot's group: 10/5 leechseed, whose drain heal lands on Flygon untagged.
                "|-damage|p1a: Swalot|281/300|[from] Leech Seed|[of] p2a: Flygon",
                "|-heal|p2a: Flygon|249/253 psn|[silent]",
                "|upkeep",
            ]
        )
        flygon = [b for b in engine.snapshot().sides["p2"] if b.species == "Flygon"][0]
        self.assertIsNone(flygon.revealed_item, "precondition: the sweep must not skip this mon")
        self.assertNotIn(
            "leftovers",
            flygon.ruled_out_items,
            "the [silent] Leech Seed drain heal is a residual-phase change with no determinable "
            "position against the drainer's own 10/4 slot, so it is not pre-slot evidence",
        )

    def test_a_discarded_snapshot_actually_clears_the_earlier_value(self) -> None:
        """The DISCARD branch must overwrite the snapshot, not merely decline to update it.

        ``_hp_snapshot_action`` returning ``discard`` is worthless unless ``_apply_hp_observation``
        acts on it, and that application was a SURVIVING MUTANT: deleting
        ``elif action == _HP_SNAPSHOT_DISCARD: self._hp_after_actions[key] = None`` left the whole
        suite green. The Flygon test above did not bind it, and it is worth being precise about why --
        Flygon is at FULL when the untagged heal arrives, so its snapshot is already 1.0 from the
        switch-in and the psn line only has to be classified ``keep``. Discarding 1.0 or keeping 1.0
        both decline. That test binds the classifier's RETURN VALUE; this one binds its APPLICATION.

        Here the mon is damaged in the action phase FIRST, so there is a below-full snapshot that the
        discard has to actively destroy. Tangela is a live pool Leech Seed carrier (12 species carry
        the move) whose sets hold Leftovers, so this is the shape the 85 violations were made of.

        Ordering: the seeded Shuckle's leechseed handler (10/5) runs before Tangela's own item slot
        (10/4) because ``comparePriority`` puts SPEED ahead of subOrder across two Pokemon, and the
        drain heal takes Tangela to full. Tangela's Leftovers is then offered with no room and stays
        silent -- so its silence is not evidence, and the elimination would be wrong.

        Kill-confirmed: deleting the discard assignment yields ``('leftovers',)`` and fails this test
        alone.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Tangela|Tangela, L84, F|277/277",
                "|switch|p2a: Shuckle|Shuckle, L88, M|198/198",
                "|turn|1",
                "|move|p1a: Tangela|Leech Seed|p2a: Shuckle",
                "|-start|p2a: Shuckle|move: Leech Seed",
                "|move|p2a: Shuckle|Rock Slide|p1a: Tangela",
                "|-damage|p1a: Tangela|240/277",
                _RESIDUAL_MARKER,
                "|-damage|p2a: Shuckle|174/198|[from] Leech Seed|[of] p1a: Tangela",
                "|-heal|p1a: Tangela|277/277|[silent]",
                "|upkeep",
            ]
        )
        tangela = [b for b in engine.snapshot().sides["p1"] if b.species == "Tangela"][0]
        self.assertIsNone(tangela.revealed_item, "precondition: the sweep must not skip this mon")
        self.assertEqual(
            tangela.ruled_out_items,
            (),
            "the untagged drain heal made the pre-slot HP undeterminable, so the turn must yield "
            "no evidence -- the below-full action-phase snapshot has to be discarded, not kept",
        )

    def test_rest_is_an_ACTION_phase_silent_heal_and_still_updates_the_snapshot(self) -> None:
        """Why ``[silent]`` cannot simply be added to the at/after list.

        The same ``Battle.heal`` branch that emits the untagged drain heal also emits REST's heal
        (``case 'leechseed': case 'rest':``), and Rest is an action-phase move carried by 46 pool
        species. Treating ``[silent]`` as residual would leave Rest's heal out of the snapshot, so the
        snapshot would keep the PRE-Rest value and the rule would eliminate Leftovers on a mon that
        was at full HP when its slot ran -- the same unsound direction, on a much more common shape.

        Kill-confirmed by adding ``"[silent]"`` to ``_AT_OR_AFTER_ITEM_SLOT_RESIDUAL_HP_TAGS``: this
        test fails and no other does.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Stantler|Stantler, L88|300/300",
                "|switch|p2a: Blastoise|Blastoise, L82, M|264/264",
                "|turn|1",
                "|move|p1a: Stantler|Return|p2a: Blastoise",
                "|-damage|p2a: Blastoise|150/264",
                "|move|p2a: Blastoise|Rest|p2a: Blastoise",
                "|-status|p2a: Blastoise|slp|[from] move: Rest",
                "|-heal|p2a: Blastoise|264/264|[silent]",
                _RESIDUAL_MARKER,
                "|upkeep",
            ]
        )
        self.assertNotIn(
            "leftovers",
            _opponent(engine, "Blastoise").ruled_out_items,
            "Rest healed to full during the ACTION phase, so the item slot found no room",
        )

    def test_a_post_slot_tag_is_honoured_even_without_the_phase_marker(self) -> None:
        """The at/after tags do not depend on the phase marker, on purpose.

        psn/brn/tox, Leech Seed damage and the Leftovers heal are residual-ONLY effects in gen3 --
        none of them can produce an HP line during the action phase -- so their tag settles the
        classification by itself. Applying them without consulting the phase keeps these cases
        correct on any event stream that reached the engine with the bare ``|`` markers stripped,
        which is why the marker is load-bearing only for the untagged residual sources.

        This fixture deliberately omits the marker. Kill-confirmed by removing ``[from] psn`` from
        ``_AT_OR_AFTER_ITEM_SLOT_RESIDUAL_HP_TAGS``.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Stantler|Stantler, L88|300/300",
                "|switch|p2a: Octillery|Octillery, L87, F|272/272",
                "|turn|1",
                "|move|p1a: Stantler|Toxic|p2a: Octillery",
                "|-status|p2a: Octillery|tox",
                "|-damage|p2a: Octillery|255/272 tox|[from] psn",
                "|upkeep",
            ]
        )
        self.assertNotIn(
            "leftovers",
            _opponent(engine, "Octillery").ruled_out_items,
            "toxic chips at 10/6, after the 10/4 item slot, so the mon was full when it ran",
        )

    # ------------------------------------------------- eliminations that MUST still happen

    def test_reaching_full_then_being_chipped_by_a_PRE_slot_residual_eliminates(self) -> None:
        """Wish (7) to full, then Sandstorm (field 8) chip: BELOW full at the slot, so eliminate.

        The mirror of the post-slot case, and one of the two eliminations the removed
        ``_healed_to_full_this_turn`` set wrongly declined -- it asked "was this mon healed to full at
        any point this turn", which is not the question. Both Wish and the sandstorm chip resolve
        before the 10/4 slot, so the item was offered at 255/272 and stayed silent. That is exactly
        the evidence the rule exists to use.

        Sandstorm is pool-reachable: all 15 Tyranitar sets carry Sand Stream, the pool's only source
        of it. Kill-confirmed by re-adding the ``_healed_to_full_this_turn`` clause, and separately by
        dropping ``[from] Sandstorm`` from ``_PRE_ITEM_SLOT_RESIDUAL_HP_TAGS`` -- the second is the
        coverage that entry had none of.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Tyranitar|Tyranitar, L74, M|265/265",
                "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar",
                "|switch|p2a: Octillery|Octillery, L87, F|272/272",
                "|turn|1",
                "|move|p1a: Tyranitar|Rock Slide|p2a: Octillery",
                "|-damage|p2a: Octillery|169/272",
                _RESIDUAL_MARKER,
                "|-heal|p2a: Octillery|272/272|[from] move: Wish|[wisher] Umbreon",
                "|-damage|p2a: Octillery|255/272|[from] Sandstorm",
                "|upkeep",
            ]
        )
        self.assertIn(
            "leftovers",
            _opponent(engine, "Octillery").ruled_out_items,
            "the mon was below full when its 10/4 slot ran and no Leftovers heal followed",
        )

    def test_a_partial_wish_heal_still_supplies_pre_slot_evidence(self) -> None:
        """Wish's pre-slot entry, which had no coverage of its own.

        Every other Wish fixture heals to FULL, and those all pass whether Wish is classified as
        pre-slot (snapshot -> 1.0) or as unclassifiable (snapshot -> discarded): both decline. So the
        ``[from] move: Wish`` entry survived deletion silently.

        Wish restores half of max HP, so a mon low enough is still below full afterwards. At order 7
        that lands before the 10/4 slot, which means Leftovers WAS offered -- at 236/272, with room to
        heal -- and stayed silent. Sound elimination.

        Kill-confirmed by dropping ``[from] move: Wish`` from ``_PRE_ITEM_SLOT_RESIDUAL_HP_TAGS``:
        this test fails and only this one.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Stantler|Stantler, L88|300/300",
                "|switch|p2a: Octillery|Octillery, L87, F|272/272",
                "|turn|1",
                "|move|p1a: Stantler|Return|p2a: Octillery",
                "|-damage|p2a: Octillery|100/272",
                _RESIDUAL_MARKER,
                "|-heal|p2a: Octillery|236/272|[from] move: Wish|[wisher] Umbreon",
                "|upkeep",
            ]
        )
        self.assertIn(
            "leftovers",
            _opponent(engine, "Octillery").ruled_out_items,
            "Wish resolves at order 7, so the item slot was offered at 236/272 with room to heal",
        )

    def test_healing_to_full_and_then_being_hit_in_the_action_phase_eliminates(self) -> None:
        """Recover to full, then take a hit: below full at the slot, so eliminate.

        The other elimination ``_healed_to_full_this_turn`` wrongly declined. Ordering does not even
        enter into it -- both lines are action-phase -- which is what made that set plainly wrong
        rather than merely conservative.

        Kill-confirmed by re-adding the ``_healed_to_full_this_turn`` clause: this test and the
        Sandstorm one above fail, and nothing else does.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Stantler|Stantler, L88|300/300",
                "|switch|p2a: Porygon2|Porygon2, L80|180/267",
                "|turn|1",
                "|move|p2a: Porygon2|Recover|p2a: Porygon2",
                "|-heal|p2a: Porygon2|267/267",
                "|move|p1a: Stantler|Return|p2a: Porygon2",
                "|-damage|p2a: Porygon2|180/267",
                _RESIDUAL_MARKER,
                "|upkeep",
            ]
        )
        self.assertIn(
            "leftovers",
            _opponent(engine, "Porygon2").ruled_out_items,
            "healed to full EARLIER in the turn is not the same as full when the slot ran",
        )

    def test_a_pre_slot_sandstorm_crossing_still_eliminates_the_pinch_berries(self) -> None:
        """The pinch rule reads the same snapshot, so Sandstorm's entry has to bind there too.

        gen3's pinch berries share the item slot exactly (`data/mods/gen3/items.ts` sets
        ``onUpdate: undefined`` plus ``onResidualOrder: 10, onResidualSubOrder: 4``). A sandstorm chip
        at field order 8 takes the mon under 25% BEFORE the berry is offered, so the berry's silence
        is real evidence. Kill-confirmed by dropping ``[from] Sandstorm`` from the pre-slot list.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Tyranitar|Tyranitar, L74, M|265/265",
                "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar",
                "|switch|p2a: Octillery|Octillery, L87, F|272/272",
                "|turn|1",
                "|move|p1a: Tyranitar|Rock Slide|p2a: Octillery",
                "|-damage|p2a: Octillery|72/272",
                _RESIDUAL_MARKER,
                "|-damage|p2a: Octillery|55/272|[from] Sandstorm",
                "|upkeep",
            ]
        )
        ruled_out = _opponent(engine, "Octillery").ruled_out_items
        for berry in ("salacberry", "petayaberry", "liechiberry"):
            self.assertIn(berry, ruled_out)

    # --------------------------------------------------------- the per-turn clear, white-box

    def test_the_pre_slot_snapshot_is_cleared_at_upkeep(self) -> None:
        """The per-turn clear, asserted as the state invariant it is.

        This is a white-box assertion on purpose, and the reason is worth writing down: with the
        classifier fixed, a stale snapshot can only ever be HIGHER than the mon's true HP at the next
        turn's item slot. HP only rises through a line that updates the snapshot (action-phase heals,
        Wish, weather), discards it (the untagged drain heal), or reveals the item outright (the
        Leftovers heal, after which the sweep skips the mon). So a stale value can only make both
        rules DECLINE, never fire wrongly -- which is exactly why no behavioural fixture can
        distinguish "cleared" from "stale" any more, and why the previous test that claimed to isolate
        this line did not.

        The line stays because the policy it implements is not an optimization: the rules are supposed
        to reason from what THIS turn published, not from an inference chain about HP monotonicity
        across turns. Deleting it fails this test and only this test.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Stantler|Stantler, L88|300/300",
                "|switch|p2a: Octillery|Octillery, L87, F|272/272",
                "|turn|1",
                "|move|p1a: Stantler|Return|p2a: Octillery",
                "|-damage|p2a: Octillery|169/272",
                _RESIDUAL_MARKER,
                "|upkeep",
            ]
        )
        self.assertEqual(
            engine._hp_after_actions,
            {},
            "the pre-slot HP snapshot is per-turn evidence and must not survive |upkeep",
        )

    def test_the_snapshot_is_populated_before_upkeep(self) -> None:
        """Reachability for the test above: an empty dict must mean CLEARED, not never-written."""
        engine = _engine_from(
            [
                "|switch|p1a: Stantler|Stantler, L88|300/300",
                "|switch|p2a: Octillery|Octillery, L87, F|272/272",
                "|turn|1",
                "|move|p1a: Stantler|Return|p2a: Octillery",
                "|-damage|p2a: Octillery|169/272",
            ]
        )
        self.assertTrue(engine._hp_after_actions)
        self.assertAlmostEqual(
            next(v for k, v in engine._hp_after_actions.items() if "octillery" in k.lower()),
            169 / 272,
            places=6,
        )


class HpSnapshotClassifierTest(unittest.TestCase):
    """The classifier's own table, for the entries a battle fixture cannot reliably reach.

    Two entries needed this. ``[from] confusion`` only matters on a turn whose FIRST published line
    is the confusion self-hit -- rare enough that a short sweep does not hit it (measured: 1 line in
    35,283 on seed 555001, 0 on 31337), so no fixture-based test binds it. And the
    ``_ACTION_PHASE_EVENT_TYPES`` set is what re-opens the action phase after ``turnLoop``'s copy of
    the bare marker; its members are load-bearing for every action-phase HP line in the format, which
    makes a targeted assertion clearer than the diffuse failure they produce.
    """

    def _action(self, raw_line: str, *, residual: bool) -> str:
        from pokezero.belief import _hp_snapshot_action

        return _hp_snapshot_action(raw_line, in_residual_phase=residual)

    def test_confusion_self_damage_is_action_phase_whatever_the_tracked_phase_says(self) -> None:
        """``confusion.onBeforeMove`` deals it inside move execution, so it is always pre-slot.

        A confused mon that hits itself publishes ``|-activate|…|confusion`` and ``|-damage|…|[from]
        confusion`` and NO ``|move|`` line, so on that turn nothing has re-opened the action phase
        yet. Without this entry the snapshot is discarded instead of updated -- declined evidence, not
        a wrong belief, but a measurable divergence from the sim, and the differential asserts zero.

        Kill-confirmed by emptying ``_ACTION_PHASE_ONLY_HP_TAGS``.
        """
        from pokezero.belief import _HP_SNAPSHOT_UPDATE

        line = "|-damage|p2a: Octillery|169/272|[from] confusion"
        self.assertEqual(self._action(line, residual=True), _HP_SNAPSHOT_UPDATE)
        self.assertEqual(self._action(line, residual=False), _HP_SNAPSHOT_UPDATE)

    def test_an_HP_line_with_no_text_follows_the_phase_and_declines_in_residuals(self) -> None:
        """The least-information case must not be the most permissive one.

        With no line text there is no tag, but the phase is still known: on the action side every
        change precedes every residual regardless of source, so it updates; in the residual phase
        nothing is left to classify with, so it declines. The branch previously returned ``update``
        unconditionally — the same "assume action-phase" shape as the four defects this rule has
        already had — and it was a surviving mutant: flipping it to ``discard`` left all 90 tests
        green.

        Kill-confirmed by returning ``_HP_SNAPSHOT_UPDATE`` unconditionally.
        """
        from pokezero.belief import _HP_SNAPSHOT_DISCARD, _HP_SNAPSHOT_UPDATE, _hp_snapshot_action

        for empty in (None, ""):
            self.assertEqual(
                _hp_snapshot_action(empty, in_residual_phase=False), _HP_SNAPSHOT_UPDATE
            )
            self.assertEqual(
                _hp_snapshot_action(empty, in_residual_phase=True), _HP_SNAPSHOT_DISCARD
            )

    def test_an_unrecognized_residual_phase_source_is_declined_not_assumed(self) -> None:
        """The default that makes the tag lists safe to be incomplete.

        Kill-confirmed by returning ``_HP_SNAPSHOT_UPDATE`` at the end of ``_hp_snapshot_action``.
        """
        from pokezero.belief import _HP_SNAPSHOT_DISCARD, _HP_SNAPSHOT_UPDATE

        line = "|-damage|p2a: Octillery|169/272|[from] SomeFutureResidual"
        self.assertEqual(self._action(line, residual=True), _HP_SNAPSHOT_DISCARD)
        self.assertEqual(self._action(line, residual=False), _HP_SNAPSHOT_UPDATE)

    def test_every_action_phase_event_type_reopens_the_action_phase(self) -> None:
        """Each member of ``_ACTION_PHASE_EVENT_TYPES`` must actually close the residual phase.

        These are the line types that can only appear while actions are resolving, and the first of
        them after ``turnLoop``'s bare marker is what tells the engine the turn has begun.

        The expected members are written out rather than read from the set: the first version of this
        test iterated ``_ACTION_PHASE_EVENT_TYPES`` itself, so removing a member removed it from the
        iteration too and the mutation survived. That is the vacuous-assertion shape the plan's §3
        forbids, reproduced inside the test written to prevent it.

        Kill-confirmed by removing any single member: this test then fails naming that member.
        """
        from pokezero.belief import _ACTION_PHASE_EVENT_TYPES, PublicBattleBeliefEngine

        self.assertEqual(
            _ACTION_PHASE_EVENT_TYPES,
            frozenset({"move", "switch", "drag", "replace", "cant"}),
            "the action-phase line types changed; the differential in test_belief_coherence.py is "
            "what should justify the new set",
        )
        for event_type in ["move", "switch", "drag", "replace", "cant", "turn"]:
            engine = PublicBattleBeliefEngine()
            engine.ingest_event({"event_type": "unknown", "raw_line": "|"})
            self.assertTrue(engine._in_residual_phase, "the bare marker must open the phase")
            engine.ingest_event({"event_type": event_type, "raw_line": f"|{event_type}|p1a: X|Y"})
            self.assertFalse(
                engine._in_residual_phase,
                f"|{event_type}| is action-phase-only and must close the residual phase",
            )

    def test_a_residual_phase_line_does_not_reopen_the_action_phase(self) -> None:
        """Reachability for the test above: the phase must not be closed by everything."""
        from pokezero.belief import PublicBattleBeliefEngine

        engine = PublicBattleBeliefEngine()
        engine.ingest_event({"event_type": "unknown", "raw_line": "|"})
        engine.ingest_event(
            {"event_type": "-damage", "raw_line": "|-damage|p1a: X|100/200|[from] psn"}
        )
        self.assertTrue(engine._in_residual_phase)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
