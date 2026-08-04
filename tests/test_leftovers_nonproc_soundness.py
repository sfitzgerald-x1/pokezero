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
2. **A stale pre-residual snapshot -- the root cause.** `_hp_after_actions` was not cleared per
   turn, so a mon damaged on an earlier turn that takes no action-phase hit on this one was still
   judged against the old snapshot. The code's own comment states the policy as "no snapshot => no
   evidence (conservative)", but a STALE snapshot is not no-snapshot. It is now cleared at
   `|upkeep`.

The consequence measured in live games: for a species whose every variant holds Leftovers
(Octillery) the rule-out emptied the candidate set, forced the inconsistent fallback and pinned
`uncertainty` to 1.0; one turn later the same belief carried `revealed_item="Leftovers"` AND
`ruled_out_items=("leftovers",)` at the same time. On a mixed-item species it drops the true variant
instead, which is the unrecoverable direction.

The last test here is the one that keeps this honest: the rule must still FIRE on a genuine
non-proc, or the "fix" is just the pruning switched off.
"""

from __future__ import annotations

import unittest

from pokezero.belief import PublicBattleBeliefEngine, RevealedPokemonBelief
from pokezero.showdown import parse_showdown_replay


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
        """Isolates the PER-TURN CLEAR of ``_hp_after_actions`` -- only it can block this.

        The live reproducer the first version of this fix missed (Togetic, true item Leftovers).
        Turn 1 damages it and Wish tops it off, leaving a stale snapshot. On turn 2 it takes NO
        action-phase HP change, is at full HP when the Leftovers slot runs (10/4), and only falls
        below full afterwards from toxic (10/6). So the mon ends turn 2 damaged with no Leftovers
        heal, and nothing healed it to full DURING turn 2 -- both of the other guards pass, and the
        elimination is still wrong.

        This is why "ends the turn below full" was an unsound proxy: it is not the same question as
        "was below full when the Leftovers slot ran".
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

    def test_reaching_full_then_being_chipped_does_not_rule_out_leftovers(self) -> None:
        """Isolates the HEAL-TO-FULL guard: only it can block this case.

        Wish tops the mon off before the Leftovers slot (so Leftovers had no room and stayed
        silent), and a later residual then chips it back BELOW full. Because the turn does not end
        at full HP, the end-of-turn guard cannot help here — the heal-to-full tracking is what makes
        this sound.
        """
        engine = _engine_from(
            [
                "|switch|p1a: Stantler|Stantler, L88|300/300",
                "|switch|p2a: Octillery|Octillery, L87, F|272/272",
                "|turn|1",
                "|move|p1a: Stantler|Return|p2a: Octillery",
                "|-damage|p2a: Octillery|169/272",
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
