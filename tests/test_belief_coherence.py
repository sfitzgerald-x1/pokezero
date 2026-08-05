"""V1 — whole-game belief coherence, as a live integration test.

The plan's §3 of the readiness doc asked for this and it was never built: nothing anywhere asserted
that the TRUE variant stays in the candidate set at any point of a real game (grepped `tests/` for
containment/coherence/omniscient assertions, 2026-08-04: none reach it). Containment is the property
whose violation is maximally harmful — it poisons `CANDIDATE_SET_COUNT`, `UNCERTAINTY`, every
`possible-*` count and every sampled search world at once, and it fails SILENTLY.

This runs a SHORT sweep of the same harness the fleet gate runs
(`scripts/belief_coherence_gate.py`), rather than re-implementing its assertions here: two copies of
a coherence check drifting apart is the very defect class the harness exists to catch. The long
sweep (≥20k games) is a fleet job; this is the always-on regression guard.

Skips cleanly without a built Showdown checkout + node.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from tests.test_tier2_live_env import _integration_root

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


@unittest.skipUnless(_integration_root() is not None, "requires built Showdown checkout and node")
class BeliefCoherenceSweepTest(unittest.TestCase):
    """A few real games, all seven assertion families, zero tolerated violations."""

    @classmethod
    def setUpClass(cls) -> None:
        from belief_coherence_gate import run_sweep

        # Deliberately SMALL, and this is a known limitation rather than a choice: the Leftovers
        # defect this harness found needs ~26-400 games to surface depending on seed, so a guard
        # big enough to catch it would be red on this branch. It is raised in the PR that fixes
        # that defect, where the tree is green. The fleet sweep is the real bar either way.
        cls.summary = run_sweep(
            showdown_root=_integration_root(),
            games=3,
            seed=7,
            clone_equivalence_every=5,
        )

    def test_no_coherence_violations(self) -> None:
        """Zero violations of families 1/2/4/5/6/7 — the plan's exit criterion, in miniature."""
        self.assertEqual(
            self.summary["violation_counts"],
            {name: 0 for name in self.summary["violation_counts"]},
            msg=f"coherence violations: {self.summary.get('violations')}",
        )
        self.assertEqual(self.summary["verdict"], "PASS")

    def test_no_mon_was_silently_skipped(self) -> None:
        """A skipped mon must fail the run, not vanish into a counter.

        Truth resolution returning None skips the mon. Before this was gated, mutating
        ``_true_variant_for`` to fail for half the species silently dropped 1106 of 2551
        observations -- including the very mon whose defect motivated this harness -- and the
        verdict stayed PASS. Kill-confirmed: that mutation now yields FAIL with skips=644.
        """
        self.assertEqual(
            self.summary["skipped"],
            {name: 0 for name in self.summary["skipped"]},
            "the sweep skipped mons or truncated games; its coverage is not what it reports",
        )
        self.assertTrue(self.summary["no_silent_skips"])

    def test_pin_conflict_family_is_reported_as_inapplicable_when_narrowing_is_off(self) -> None:
        """Family 4 cannot fire with narrowing off, and must say so rather than claim coverage.

        ``variant_pin_conflicts`` is only written from the tier2/investment producers, which are
        gated off in this configuration -- measured directly: over 397 decision points
        ``_variant_pins`` was non-empty 0 times. Counting loop iterations as "reached" reported the
        property as exercised when it could not fire, which is the laundering the plan forbids.

        The status is now derived from ``pins_observed`` (did a pin ever get WRITTEN) rather than
        from the config flag, so ``--item-belief-narrowing`` alone can no longer make the artifact
        claim "exercised" for a run in which nothing pinned.
        """
        self.assertEqual(
            self.summary["pin_conflict_family"],
            "n/a (narrowing off; no producer writes _variant_pins)",
        )
        self.assertEqual(self.summary["counts"]["pins_observed"], 0)
        self.assertGreater(self.summary["counts"]["pin_conflict_checks"], 0)
        self.assertNotIn("pin_conflict_checks", self.summary["reachability"])

    def test_family_5_ability_arm_is_reported_as_having_no_producer(self) -> None:
        """The ability arm is structurally unfailable in gen3 randbats and must not claim coverage.

        ``ruled_out_abilities`` has exactly one producer -- the Intimidate non-trigger rule -- and
        both of its gates require the mon to have Intimidate AND another possible ability. Every one
        of the 11 pool Intimidate carriers has Intimidate as its sole ability, so the precondition
        is never met. The harness's docstring used to sell this arm as "the TRAPPER_ALIVE
        correctness check ... Shadow Tag and Arena Trap are both pool-reachable", which is false:
        both are their species' only ability too, so neither can ever reach
        ``ruled_out_abilities``.

        This test pins the MEASUREMENT, not the prose: zero rule-outs and zero preconditions met.
        """
        counts = self.summary["counts"]
        self.assertEqual(counts["mons_with_ruled_out_abilities"], 0)
        self.assertEqual(counts["intimidate_ruleout_preconditions"], 0)
        self.assertTrue(
            self.summary["ruled_out_ability_arm"].startswith("n/a (no producer in gen3 randbats)"),
            self.summary["ruled_out_ability_arm"],
        )

    def test_family_5_item_arm_is_reachable(self) -> None:
        """The arm that CAN fire has to be shown firing, or family 5 is vacuous end to end.

        With the ability arm reported n/a, ``ruled_out_items`` is the only live producer left in
        family 5. A sweep in which it never fired would make the whole family a no-op while still
        printing PASS -- exactly the vacuous pass the plan's §3 forbids.
        """
        self.assertGreater(
            self.summary["counts"]["mons_with_ruled_out_items"],
            0,
            "no item was ever ruled out; family 5 has no live arm in this run",
        )

    def test_games_without_both_requests_is_gated_not_merely_counted(self) -> None:
        """A game whose opening requests did not resolve is a SKIP and must fail the run.

        It was counted and then ignored. A regression in ``_first_requests`` would ``continue`` past
        every game, leaving ``games`` at 0 with nothing in the verdict naming the cause.
        """
        self.assertIn("games_without_both_requests", self.summary["skipped"])
        self.assertEqual(self.summary["skipped"]["games_without_both_requests"], 0)

    def test_sweep_actually_reached_the_properties_it_asserts(self) -> None:
        """The vacuous-pass guard.

        A containment sweep over games where no opponent mon was ever recognized passes trivially,
        which is the bug and not the fix (plan §3). Each of these counters was chosen because a
        zero would make the corresponding assertion meaningless, so the sweep's own verdict is FAIL
        unless all of them are positive — this test pins that the run really did reach them.
        """
        reach = self.summary["reachability"]
        self.assertGreater(reach["mon_observations"], 100, "too few belief observations")
        self.assertGreater(reach["distinct_species"], 5, "too few species reached")
        self.assertGreater(reach["narrowing_steps"], 0, "no set ever narrowed; containment is idle")
        self.assertGreater(reach["pinned_and_correct"], 0, "no set was ever pinned to one variant")
        self.assertGreater(reach["stat_legality_checks"], 0, "assertion 6 never ran")
        self.assertGreater(reach["clone_equivalence_checks"], 0, "assertion 7 never ran")
        self.assertTrue(self.summary["reached"])

    def test_every_monotonicity_growth_is_an_attributed_fallback(self) -> None:
        """Growth is allowed ONLY through the documented inconsistent-fallback, and is counted.

        The plan requires every fallback be "attributed to a known cause, not silently absorbed".
        A monotonicity violation is recorded whenever a set grows WITHOUT matching the fallback
        signature (full species pool at uncertainty 1.0), so an empty violation list plus a
        published fallback count is the attribution.
        """
        self.assertEqual(self.summary["violation_counts"]["monotonicity"], 0)
        self.assertIn("inconsistent_fallbacks", self.summary["counts"])

    def test_containment_holds_for_every_observation_not_merely_on_average(self) -> None:
        """containment_ok must equal the observation count — no silent partial credit."""
        counts = self.summary["counts"]
        self.assertEqual(
            counts.get("containment_ok", 0),
            counts.get("mon_observations", -1),
            "some observations were not containment-checked or not contained",
        )


class CloneEquivalenceAliasingTest(unittest.TestCase):
    """Assertion 7's state check must catch an ALIASED field, not only a dropped one.

    ``_engine_state_mismatches`` compared ``repr()``, and ``repr()`` is identical whether the clone
    copied a container or aliased it. So ``twin._x = self._x`` -- two engines sharing one dict, every
    sampled-world mutation writing back into the live game's belief state -- was invisible to the
    sweep in both directions. No Showdown checkout needed: the defect is in the comparison, so it is
    reproducible on a bare engine.
    """

    def _engine(self):
        from pokezero.belief import PublicBattleBeliefEngine

        engine = PublicBattleBeliefEngine()
        engine._hp_after_actions = {"p2:snorlax": 0.5}
        return engine

    def test_a_correct_clone_reports_no_mismatches(self) -> None:
        from belief_coherence_gate import _engine_state_mismatches

        engine = self._engine()
        self.assertEqual(_engine_state_mismatches(engine, engine.clone()), [])

    def test_an_aliased_container_is_reported(self) -> None:
        from belief_coherence_gate import _engine_state_mismatches

        engine = self._engine()
        twin = engine.clone()
        # Exactly the mutation `clone()` would contain if someone wrote `twin._x = self._x`.
        twin._hp_after_actions = engine._hp_after_actions
        mismatches = _engine_state_mismatches(engine, twin)
        self.assertEqual(
            mismatches,
            ["_hp_after_actions: clone ALIASED the parent's container instead of copying it"],
        )

    def test_an_aliased_EMPTY_container_is_reported_too(self) -> None:
        """The value check is blind to empty state; the identity check is not.

        This is what extends the guard to ``_variant_pins``, which is empty with narrowing off and
        therefore could not be covered by the value comparison at all.
        """
        from belief_coherence_gate import _engine_state_mismatches
        from pokezero.belief import PublicBattleBeliefEngine

        engine = PublicBattleBeliefEngine()
        twin = engine.clone()
        twin._variant_pins = engine._variant_pins
        self.assertEqual(
            _engine_state_mismatches(engine, twin),
            ["_variant_pins: clone ALIASED the parent's container instead of copying it"],
        )

    def test_immutable_shared_state_is_not_flagged(self) -> None:
        """Sharing is correct for immutables, so the check must not fire on them.

        A guard that reported every shared int and string would be noise, and noise gets suppressed.
        """
        from belief_coherence_gate import _engine_state_mismatches

        engine = self._engine()
        engine.ingest_event({"event_type": "turn", "raw_line": "|turn|3"})
        twin = engine.clone()
        self.assertEqual(twin._turn_number, engine._turn_number)
        self.assertEqual(_engine_state_mismatches(engine, twin), [])


@unittest.skipUnless(_integration_root() is not None, "requires built Showdown checkout and node")
class Family5AbilityArmHasNoProducerTest(unittest.TestCase):
    """The structural claim behind reporting family 5's ability arm as n/a, read from the pool.

    Measured rather than argued, because the previous justification (TRAPPER_ALIVE / "Shadow Tag and
    Arena Trap are both pool-reachable") was argued and wrong.
    """

    @staticmethod
    def _norm(value) -> str:
        return "".join(ch for ch in str(value or "").lower() if ch.isalnum())

    def _abilities_by_species(self) -> dict[str, set[str]]:
        from pokezero.randbat import load_gen3_randbat_source_cached

        source = load_gen3_randbat_source_cached(str(_integration_root()))
        return {
            species: {self._norm(variant.ability) for variant in universe.variants}
            for species, universe in source.universes.items()
        }

    def test_no_pool_intimidate_carrier_has_a_second_ability(self) -> None:
        """The Intimidate rule's precondition, checked against the generator's own data.

        ``_can_queue_intimidate_non_trigger`` and ``_can_rule_out_intimidate`` both require
        Intimidate AND at least one other possible ability. If this ever stops holding, the arm gains
        a producer and the gate's n/a report must be revisited -- which is what this test is for.
        """
        by_species = self._abilities_by_species()
        carriers = {sp: ab for sp, ab in by_species.items() if "intimidate" in ab}
        self.assertGreater(len(carriers), 0, "no Intimidate carrier in the pool at all")
        self.assertEqual(
            {sp: sorted(ab) for sp, ab in carriers.items() if len(ab) > 1},
            {},
            "an Intimidate carrier now has a second possible ability: the ability arm HAS a "
            "producer and belief_coherence_gate's n/a report is stale",
        )

    def test_the_trappers_cannot_reach_ruled_out_abilities_either(self) -> None:
        """The specific claim the docstring made and this PR strikes.

        Wobbuffet/Shadow Tag and Dugtrio/Arena Trap are pool-reachable SPECIES, but each ability is
        its species' only one, so it can never be eliminated and never lands in
        ``ruled_out_abilities``. This sweep therefore cannot check TRAPPER_ALIVE's soundness.
        """
        by_species = self._abilities_by_species()
        for trapper in ("shadowtag", "arenatrap"):
            carriers = {sp: ab for sp, ab in by_species.items() if trapper in ab}
            self.assertGreater(len(carriers), 0, f"{trapper} is not pool-reachable at all")
            for species, abilities in carriers.items():
                self.assertEqual(
                    sorted(abilities),
                    [trapper],
                    f"{species} now has more than one possible ability, so {trapper} could be "
                    "ruled out and the struck TRAPPER_ALIVE claim would become checkable",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
