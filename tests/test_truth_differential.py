"""Gates for the truth-injection differential runner.

The load-bearing one is :class:`ForcedRefusalTests`. An instrument that cannot
report failure will report success -- this campaign has hit that repeatedly,
including a capture harness that returned 0 records and looked clean because a
method was patched on the module instead of the instance. So before any
"0 truth-rejections" reading is trusted, a forced rejection must be shown to be
counted AND attributed, and the production arm must be shown to be untouched by
the forcing.
"""

from __future__ import annotations

import random
import sys
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _showdown_root import requires_showdown, showdown_root_str  # noqa: E402

from pokezero.truth_differential import (  # noqa: E402
    STAGES,
    TruthDecisionRecord,
    TruthDifferentialProbe,
    TruthRefusal,
    TruthWorldBuilder,
    aggregate_records,
    identity_witness,
    mechanism_family,
    probe_policy_config,
    stage_for_predicate,
)


# --- doubles -----------------------------------------------------------------


class _Stats:
    """The subset of ``EngineMctsStats`` the probe reads, with the same units."""

    def __init__(self) -> None:
        self.world_failure_reasons: Counter = Counter()
        self.fallback_reasons: Counter = Counter()
        self.choices_unmapped_causes: Counter = Counter()
        self.unmapped_choices: Counter = Counter()
        self.lossy_subcase_renders: Counter = Counter()
        self.worlds_attempted = 0
        self.worlds_constructed = 0
        self.worlds_searched = 0
        self.attribution_unsafe_renders = 0
        self.fallback_decisions = 0
        self.searched_decisions = 0


class _ScriptedPolicy:
    """A policy whose per-decision counter movements are dictated by a script."""

    def __init__(self, script: Any = None, policy_id: str = "scripted") -> None:
        self.policy_id = policy_id
        self.stats = _Stats()
        self.script = script
        self.calls: list[Any] = []
        self._fixed_override: Any = None

    def select_action_with_context(self, context: Any, *, rng: random.Random) -> Any:
        self.calls.append(context)
        if self.script is not None:
            self.script(self.stats, context)
        return object()


def _clean_world(stats: _Stats, _context: Any) -> None:
    stats.worlds_attempted += 1
    stats.worlds_constructed += 1
    stats.worlds_searched += 1
    stats.searched_decisions += 1


def _refuse_construction(stats: _Stats, _context: Any) -> None:
    stats.worlds_attempted += 1
    stats.world_failure_reasons["forced_instrument_test: construction refused"] += 1
    stats.fallback_reasons["no_worlds_constructed"] += 1
    stats.fallback_decisions += 1


def _refuse_abort_and_map(stats: _Stats, _context: Any) -> None:
    """Two predicates on one decision -- the de-censoring case."""

    stats.worlds_attempted += 1
    stats.worlds_constructed += 1
    stats.world_failure_reasons[
        "crate_search: attribution-unsafe renderer branch rejected before tree/model fold: heal_zero_marker"
    ] += 1
    stats.lossy_subcase_renders["shape_partial"] += 3
    stats.choices_unmapped_causes["all_unmapped_legality_mismatch"] += 1
    stats.fallback_reasons["crate_search_failed"] += 1
    stats.fallback_decisions += 1


def _sampler_only_failure(stats: _Stats, _context: Any) -> None:
    for _ in range(32):
        stats.worlds_attempted += 1
        stats.world_failure_reasons["belief_sample: opponent belief could not be materialized"] += 1
    stats.fallback_reasons["no_worlds_constructed"] += 1
    stats.fallback_decisions += 1


class _Observation:
    def __init__(self, metadata: Any) -> None:
        self.metadata = metadata
        self.legal_action_mask = (True,) * 10


class _Context:
    def __init__(self, battle_id: str = "b1", round_index: int = 3, seat: str = "p1") -> None:
        self.battle_id = battle_id
        self.decision_round_index = round_index
        self.player_id = seat
        self.format_id = "gen3randombattle"
        self.observation = _Observation(
            {
                "self_team": [{"species": "Absol", "active": True, "condition": "100/100"}],
                "opponent_team": [{"species": "Zapdos", "active": True, "condition": "80/100"}],
                "action_candidates": ["move 1", "switch 2"],
            }
        )
        self.public_materialization_state = None


class _StubBuilder:
    def __init__(self, override: Any = None, failure: str | None = None) -> None:
        self.override = override if override is not None else _StubOverride()
        self.failure = failure

    def reset(self) -> None:
        pass

    def override_for(self, _context: Any) -> tuple[Any, str | None]:
        if self.failure:
            return None, self.failure
        return self.override, None


class _StubOverride:
    player_teams = {"p1": "packed-p1", "p2": "packed-p2"}


def _probe(primary_script, truth_script, **kwargs) -> tuple[TruthDifferentialProbe, list]:
    records: list = []
    probe = TruthDifferentialProbe(
        primary=_ScriptedPolicy(primary_script, "primary"),
        truth_policy=_ScriptedPolicy(truth_script, "truth"),
        truth_builder=_StubBuilder(),
        records=records,
        seed=4242,
        **kwargs,
    )
    return probe, records


# --- taxonomy ----------------------------------------------------------------


class TaxonomyTests(unittest.TestCase):
    def test_every_stage_is_registered(self) -> None:
        self.assertEqual(len(STAGES), len(set(STAGES)))
        for stage in ("construction", "crate_search", "choice_mapping", "decision"):
            self.assertIn(stage, STAGES)

    def test_stage_is_read_off_the_counter_key_prefix(self) -> None:
        self.assertEqual(stage_for_predicate("crate_search: boom"), "crate_search")
        self.assertEqual(stage_for_predicate("root_inputs: KeyError: x"), "root_inputs")
        self.assertEqual(
            stage_for_predicate("materialization_blocker: toxic-stage-unknown"), "construction"
        )

    def test_the_renderer_abort_gets_its_own_family(self) -> None:
        predicate = (
            "crate_search: attribution-unsafe renderer branch rejected before "
            "tree/model fold: heal_zero_marker"
        )
        self.assertEqual(
            mechanism_family("crate_search", predicate), "renderer:attribution_unsafe"
        )
        self.assertEqual(
            mechanism_family("crate_search", "crate_search: TypeError: nope"),
            "crate_search:other",
        )

    def test_construction_families_keep_the_actionable_slug(self) -> None:
        self.assertEqual(
            mechanism_family("construction", "materialization_blocker: baton-pass:substitute"),
            "construction:materialization_blocker",
        )
        self.assertEqual(
            mechanism_family("construction", "attract_patch_unavailable"), "engine_capability"
        )
        self.assertEqual(
            mechanism_family(
                "construction", "engine_capability_unavailable: PokeEngineUnavailableError"
            ),
            "engine_capability",
        )


class ProbeConfigTests(unittest.TestCase):
    def test_the_truth_arm_draws_exactly_one_world_once(self) -> None:
        from pokezero.engine_search import EngineMctsConfig

        production = EngineMctsConfig(
            leaf_eval="hp_fraction_crate", worlds=8, sample_retry_factor=4, search_sims=256
        )
        probe = probe_policy_config(production, sims=16)
        self.assertEqual(probe.worlds, 1)
        # Retrying a FIXED override re-runs identical construction and would
        # multiply one refusal into four identical inventory rows.
        self.assertEqual(probe.sample_retry_factor, 1)
        self.assertEqual(probe.search_sims, 16)
        self.assertFalse(probe.early_stop)
        self.assertEqual(probe.search_depth, production.search_depth)


# --- the instrument must be able to report failure ---------------------------


class ForcedRefusalTests(unittest.TestCase):
    def test_a_clean_truth_world_reports_no_rejection(self) -> None:
        probe, records = _probe(_clean_world, _clean_world)
        probe.select_action_with_context(_Context(), rng=random.Random(0))
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].truth_rejected)
        self.assertEqual(records[0].refusals, [])
        self.assertEqual(probe.errors, [])
        summary = aggregate_records(records)
        self.assertEqual(summary["truth_rejection_rate"], 0.0)
        self.assertEqual(summary["distinct_open_predicates"], 0)

    def test_a_forced_construction_refusal_is_counted_and_attributed(self) -> None:
        """The null-world control for the reading above.

        A zero from an instrument that cannot produce a one is not a measurement.
        """

        probe, records = _probe(_clean_world, _refuse_construction)
        probe.select_action_with_context(_Context(), rng=random.Random(0))
        record = records[0]
        self.assertTrue(record.truth_rejected)
        self.assertEqual(record.truth_fallback_reason, "no_worlds_constructed")
        predicates = {(r.stage, r.predicate) for r in record.refusals}
        self.assertIn(
            ("construction", "forced_instrument_test: construction refused"), predicates
        )
        self.assertIn(("decision", "fallback:no_worlds_constructed"), predicates)
        summary = aggregate_records(records)
        self.assertEqual(summary["truth_rejection_rate"], 1.0)
        self.assertEqual(summary["distinct_open_predicates"], 2)

    def test_forcing_the_truth_arm_leaves_the_production_arm_byte_identical(self) -> None:
        """The forcing must not leak into the thing being measured."""

        clean, clean_records = _probe(_clean_world, _clean_world)
        forced, forced_records = _probe(_clean_world, _refuse_construction)
        for probe in (clean, forced):
            for index in range(4):
                probe.select_action_with_context(
                    _Context(round_index=index), rng=random.Random(index)
                )
        self.assertEqual(
            vars(clean.primary.stats).keys(), vars(forced.primary.stats).keys()
        )
        for key, value in vars(clean.primary.stats).items():
            self.assertEqual(value, getattr(forced.primary.stats, key), key)
        self.assertEqual(
            [r.production_fallback_reason for r in clean_records],
            [r.production_fallback_reason for r in forced_records],
        )
        self.assertTrue(all(r.truth_rejected for r in forced_records))
        self.assertFalse(any(r.truth_rejected for r in clean_records))

    def test_every_predicate_on_a_decision_is_reported_not_the_first(self) -> None:
        """De-censoring, at the resolution offline injection actually buys."""

        probe, records = _probe(_clean_world, _refuse_abort_and_map)
        probe.select_action_with_context(_Context(), rng=random.Random(0))
        record = records[0]
        stages = {refusal.stage for refusal in record.refusals}
        self.assertEqual(stages, {"crate_search", "choice_mapping", "decision"})
        # BRANCH RENDERS are carried, and are NOT refusals.
        self.assertEqual(record.lossy_subcase_renders, {"shape_partial": 3})
        self.assertNotIn(
            "shape_partial", {refusal.predicate for refusal in record.refusals}
        )

    def test_an_unavailable_truth_is_an_instrument_gap_not_an_acceptance(self) -> None:
        records: list = []
        probe = TruthDifferentialProbe(
            primary=_ScriptedPolicy(_clean_world, "primary"),
            truth_policy=_ScriptedPolicy(_clean_world, "truth"),
            truth_builder=_StubBuilder(failure="opening request missing or short for p2"),
            records=records,
            seed=1,
        )
        probe.select_action_with_context(_Context(), rng=random.Random(0))
        self.assertFalse(records[0].truth_available)
        summary = aggregate_records(records)
        self.assertEqual(summary["truth_probed_decisions"], 0)
        # The rate has no denominator, so it must be None rather than a clean 0.
        self.assertIsNone(summary["truth_rejection_rate"])
        self.assertEqual(summary["truth_unavailable_decisions"], 1)

    def test_a_probe_exception_is_recorded_and_never_reaches_the_game(self) -> None:
        def explode(_stats: _Stats, _context: Any) -> None:
            raise RuntimeError("probe blew up")

        probe, records = _probe(_clean_world, explode)
        probe.select_action_with_context(_Context(), rng=random.Random(0))
        self.assertEqual(len(probe.errors), 1)
        self.assertIn("probe blew up", probe.errors[0])


class SamplerResidualTests(unittest.TestCase):
    def test_sampler_search_failure_is_isolated_from_truth_rejection(self) -> None:
        probe, records = _probe(_sampler_only_failure, _clean_world)
        probe.select_action_with_context(_Context(), rng=random.Random(0))
        record = records[0]
        self.assertTrue(record.sampler_search_failure)
        self.assertFalse(record.truth_rejected)
        summary = aggregate_records(records)
        self.assertEqual(summary["sampler_search_failure_decisions"], 1)
        self.assertEqual(summary["truth_rejected_decisions"], 0)
        self.assertEqual(summary["sampler_search_failure_rate"], 1.0)

    def test_a_guard_failure_alongside_sampling_is_not_a_sampler_failure(self) -> None:
        def mixed(stats: _Stats, context: Any) -> None:
            _sampler_only_failure(stats, context)
            stats.world_failure_reasons["materialization_blocker: toxic-stage-unknown"] += 1

        probe, records = _probe(mixed, _clean_world)
        probe.select_action_with_context(_Context(), rng=random.Random(0))
        self.assertFalse(records[0].sampler_search_failure)


class AggregationTests(unittest.TestCase):
    def test_the_three_units_are_never_merged_into_one_table(self) -> None:
        record = TruthDecisionRecord(
            battle_id="b", seed=1, seat="p1", round=0, turn=0, truth_rejected=True,
            truth_fallback_reason="crate_search_failed",
            refusals=[
                TruthRefusal("crate_search", "crate_search: x", "crate_search:other", 1),
                TruthRefusal("decision", "fallback:crate_search_failed", "decision_literal", 1),
            ],
            lossy_subcase_renders={"shape_partial": 7},
        )
        summary = aggregate_records([record])
        predicate_names = {row["predicate"] for row in summary["predicates"]}
        self.assertNotIn("shape_partial", predicate_names)
        self.assertEqual(summary["lossy_subcase_renders"], {"shape_partial": 7})
        for row in summary["predicates"]:
            self.assertIn("counter_units", row)
            self.assertIn("decisions", row)

    def test_records_round_trip_through_dicts(self) -> None:
        record = TruthDecisionRecord(battle_id="b", seed=1, seat="p2", round=2, turn=2)
        self.assertEqual(
            aggregate_records([record])["decisions_seen"],
            aggregate_records([record.to_dict()])["decisions_seen"],
        )

    def test_one_exemplar_per_predicate_first_occurrence_wins(self) -> None:
        probe, records = _probe(_clean_world, _refuse_construction)
        for index in range(3):
            probe.select_action_with_context(
                _Context(battle_id=f"b{index}", round_index=index), rng=random.Random(index)
            )
        self.assertIsNotNone(records[0].exemplar)
        self.assertIsNone(records[1].exemplar)
        summary = aggregate_records(records)
        for row in summary["predicates"]:
            self.assertEqual(row["exemplar"]["battle_id"], "b0")
            self.assertIn("truth_packed_teams", row["exemplar"])


class IdentityWitnessTests(unittest.TestCase):
    def test_the_witness_names_the_loaded_tree_and_the_crate(self) -> None:
        witness = identity_witness()
        self.assertTrue(witness["truth_differential_present"])
        self.assertTrue(witness["engine_search_fixed_override_hook"])
        self.assertTrue(witness["truth_differential_file"].endswith("truth_differential.py"))
        # A content fingerprint, not just a path: a stale .pyc has the right
        # __file__ and the wrong bytes.
        self.assertIn("truth_differential", witness["source_sha256"])
        self.assertEqual(len(witness["source_sha256"]["truth_differential"]), 16)


class PlanTests(unittest.TestCase):
    """The census plan's coverage claims, checked on the plan itself."""

    @requires_showdown()
    def test_every_variant_is_covered_on_both_seats_at_least_five_times(self) -> None:
        from pokezero.randbat import load_gen3_randbat_source_cached
        from truth_differential_census import build_plan

        set_source = load_gen3_randbat_source_cached(showdown_root_str())
        plan = build_plan(set_source=set_source, passes=5, seed_base=9_800_000)
        coverage = plan["coverage"]
        self.assertEqual(coverage["variants_uncovered"], [])
        self.assertEqual(coverage["variants_on_one_seat_only"], [])
        self.assertGreaterEqual(coverage["min_games_per_variant"], 5)
        self.assertEqual(coverage["variants_covered"], plan["variant_count"])
        # Species Clause is not enforced by gen3customgame, so the harness owes it.
        for game in plan["games"]:
            for seat, packed in game["packed"].items():
                species = [entry.split("|")[0] for entry in packed.split("]")]
                self.assertEqual(len(species), 6, (game["seed"], seat))
                self.assertEqual(len(set(species)), 6, (game["seed"], seat, species))

    @requires_showdown()
    def test_the_plan_is_deterministic(self) -> None:
        from pokezero.randbat import load_gen3_randbat_source_cached
        from truth_differential_census import build_plan

        set_source = load_gen3_randbat_source_cached(showdown_root_str())
        first = build_plan(set_source=set_source, passes=2, seed_base=1000)
        second = build_plan(set_source=set_source, passes=2, seed_base=1000)
        self.assertEqual(first["games"], second["games"])


class TruthSourceTests(unittest.TestCase):
    """The injected 'truth' must actually BE the truth.

    Cross-checked against the bridge snapshot's generator-internal sets --
    a second, independent reader of both sides' hidden state.
    """

    @requires_showdown()
    def test_the_builder_reproduces_both_sides_generator_sets(self) -> None:
        from pokezero.golden_corpus import _true_teams_from_bridge_snapshot
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from pokezero.randbat import load_gen3_randbat_source_cached

        set_source = load_gen3_randbat_source_cached(showdown_root_str())
        env = LocalShowdownEnv(
            LocalShowdownConfig(showdown_root=Path(showdown_root_str()), set_belief_source=True)
        )
        try:
            env.reset(seed=970001, format_id="gen3randombattle")
            builder = TruthWorldBuilder(env, set_source=set_source)
            packed, failure = builder.packed_teams("battle")
            self.assertIsNone(failure)
            self.assertIsNotNone(packed)
            oracle = _true_teams_from_bridge_snapshot(env.snapshot().bridge_snapshot)
            for slot in ("p1", "p2"):
                mine = _normalized_packed(packed[slot])
                theirs = _normalized_packed(oracle[slot]["packed"])
                self.assertEqual(
                    [row[:5] for row in mine],
                    [row[:5] for row in theirs],
                    f"{slot}: injected truth disagrees with the generator's own sets",
                )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()


def _normalized_packed(packed: str) -> list[tuple[str, ...]]:
    """Compare packed teams modulo Showdown's ``toID`` and EV/IV formatting."""

    rows = []
    for entry in packed.split("]"):
        parts = entry.split("|")
        parts += [""] * (12 - len(parts))
        identifier = lambda text: "".join(  # noqa: E731 - Showdown's toID, inline
            character for character in text.lower() if character.isalnum()
        )
        rows.append(
            (
                identifier(parts[1] or parts[0]),
                identifier(parts[2]),
                identifier(parts[3]),
                ",".join(sorted(identifier(move) for move in parts[4].split(",") if move)),
                parts[10],
            )
        )
    return sorted(rows)


if __name__ == "__main__":
    unittest.main()
