"""Leftovers non-proc pruning must not rule out the TRUE item when it had no opportunity.

`ruled_out_items` is documented as "deterministic non-proc pruning results", and the candidate
filter consumes it, so a wrong entry is not a loose belief -- it is a belief that has EXCLUDED the
truth. Two ways that happened, both found by the V1 coherence sweep
(`scripts/belief_coherence_gate.py`, 565 violations over 250 games):

1. **A non-Leftovers heal reached full HP before the Leftovers slot.** Wish is the common case and
   it is engine-ordered FIRST (`data/moves.ts` wish `onResidualOrder: 4` vs `data/items.ts`
   leftovers `onResidualOrder: 5`), so a Wish that tops the mon off leaves Leftovers nothing to heal
   and it emits no line. Reading that silence as proof of absence ruled out the real item.
2. **A stale pre-residual snapshot.** `_hp_after_actions` is not cleared per turn, so a mon damaged
   on an earlier turn that takes no action-phase hit on this one was still judged against the old
   snapshot. The code's own comment states the policy as "no snapshot => no evidence
   (conservative)", but a STALE snapshot is not no-snapshot.

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

    def test_stale_pre_residual_snapshot_does_not_rule_out_leftovers(self) -> None:
        """Isolates the END-OF-TURN guard: only it can block this case.

        Turn 1 damages the mon and Wish tops it off, which leaves a stale ``_hp_after_actions``
        snapshot of 169/272 and — crucially — does NOT reveal the item, so the sweep still runs on
        turn 2. Turn 2 has no HP lines at all, so the per-turn heal tracking is empty and the mon
        ends at full HP. Without the ``hp_fraction < 1.0`` requirement, turn 1's snapshot silently
        acts as turn-2 evidence and Leftovers is eliminated.

        Written this way deliberately: the first draft of this test put a LEFTOVERS heal on turn 1,
        which set ``revealed_item`` and made the sweep skip the mon entirely — the test passed
        without ever reaching the path it claimed to cover, and both guard mutations survived it.
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
                "|move|p2a: Octillery|Thunder Wave|p1a: Stantler",
                "|upkeep",
            ]
        )
        octillery = _opponent(engine, "Octillery")
        self.assertIsNone(
            octillery.revealed_item,
            "precondition: the item must NOT be revealed, or the sweep skips this mon and the "
            "test proves nothing",
        )
        self.assertNotIn("leftovers", octillery.ruled_out_items)

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
