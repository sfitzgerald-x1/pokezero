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

    def _map_choices(self, context: Any, aggregated: Any) -> Any:
        # Mappable by default. `probe_choice_mapping` calls this on EVERY decision,
        # so a double without it would make every scripted decision report a
        # spurious `probe_raised:AttributeError` predicate.
        return 0


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

    def test_the_READER_reports_every_counter_that_moved_not_the_first(self) -> None:
        """The READER's half of de-censoring, and ONLY the reader's half.

        RENAMED after independent review. This drives a scripted double that bumps
        three counters in one decision -- a state the real chain cannot produce,
        because the real chain stops at the first refusing stage. It shows the
        aggregation does not drop predicates; it shows NOTHING about the producer.
        The producer's seams are pinned by
        `CrossStageCensoringTests.test_the_producer_is_still_first_refuser_across_stages`.
        """

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
        plan = build_plan(
            set_source=set_source,
            passes=5,
            seed_base=9_800_000,
            showdown_root=showdown_root_str(),
        )
        coverage = plan["coverage"]
        self.assertEqual(coverage["variants_uncovered"], [])
        self.assertEqual(coverage["variants_on_one_seat_only"], [])
        self.assertGreaterEqual(coverage["min_games_per_variant"], 5)
        self.assertEqual(coverage["variants_covered"], plan["variant_count"])
        # Species Clause is not enforced by gen3customgame, so the harness owes it --
        # and it compares BASE species. Composing sides by randbat pool key put
        # Deoxys, Deoxys-Attack, Deoxys-Defense and Deoxys-Speed on ONE team, and the
        # first census run's top inventory row was the resulting ident collapse.
        from truth_differential_census import base_species_ids

        base = base_species_ids(showdown_root_str())
        self.assertEqual(plan["species_ids_without_a_base_species"], [])
        self.assertLess(plan["base_species_count"], plan["species_count"])
        for game in plan["games"]:
            for seat, packed in game["packed"].items():
                names = [entry.split("|")[0] for entry in packed.split("]")]
                self.assertEqual(len(names), 6, (game["seed"], seat))
                clause = [
                    base.get("".join(c for c in name.lower() if c.isalnum()), name)
                    for name in names
                ]
                self.assertEqual(len(set(clause)), 6, (game["seed"], seat, clause))

    @requires_showdown()
    def test_the_plan_is_deterministic(self) -> None:
        from pokezero.randbat import load_gen3_randbat_source_cached
        from truth_differential_census import build_plan

        set_source = load_gen3_randbat_source_cached(showdown_root_str())
        first = build_plan(set_source=set_source, passes=2, seed_base=1000, showdown_root=showdown_root_str())
        second = build_plan(set_source=set_source, passes=2, seed_base=1000, showdown_root=showdown_root_str())
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



# --- findings from independent review of #1214 -------------------------------


class RoundIndexTests(unittest.TestCase):
    """M13. `int(getattr(ctx, 'decision_round_index', -1) or -1)` files round 0 as -1."""

    def test_round_zero_is_recorded_as_zero(self) -> None:
        probe, records = _probe(_clean_world, _clean_world)
        probe.select_action_with_context(_Context(round_index=0), rng=random.Random(0))
        # -1 is not an address, and exemplars are advertised as replayable from
        # (seed, seat, round). The shipped `or` form put 1,462 of 52,140 published
        # records at -1 and none at 0.
        self.assertEqual(records[0].round, 0)

    def test_a_missing_round_is_still_minus_one(self) -> None:
        class _NoRound(_Context):
            def __init__(self) -> None:
                super().__init__()
                self.decision_round_index = None

        probe, records = _probe(_clean_world, _clean_world)
        probe.select_action_with_context(_NoRound(), rng=random.Random(0))
        self.assertEqual(records[0].round, -1)


class ChoiceMappingProbeTests(unittest.TestCase):
    """A. The one cross-stage reading the chain permits, and it must be real.

    The module used to ASSERT this function in its docstring while `grep` found a
    single hit -- the docstring. Compound forcings then showed the chain reporting
    exactly one substantive predicate per decision.
    """

    def test_the_function_exists_and_is_called_by_the_probe(self) -> None:
        from pokezero import truth_differential as module

        self.assertTrue(callable(module.probe_choice_mapping))
        source = Path(module.__file__).read_text(encoding="utf-8")
        # Called, not merely defined: a defined-but-uncalled probe is the bug.
        self.assertGreaterEqual(source.count("probe_choice_mapping"), 3)

    def test_an_unmappable_request_is_reported_even_with_no_worlds(self) -> None:
        from pokezero.truth_differential import probe_choice_mapping

        class _Policy:
            def __init__(self) -> None:
                self.stats = _Stats()
                self.calls: list = []

            def _map_choices(self, context, aggregated):
                self.calls.append(aggregated)
                self.stats.choices_unmapped_causes["aggregated_empty"] += 1
                return None

        policy = _Policy()
        cause = probe_choice_mapping(policy, _Context(), None)
        self.assertEqual(cause, "aggregated_empty")
        self.assertEqual(len(policy.calls), 1)

    def test_a_mappable_request_reports_nothing(self) -> None:
        from pokezero.truth_differential import probe_choice_mapping

        class _Policy:
            stats = _Stats()

            def _map_choices(self, context, aggregated):
                return 3

        self.assertIsNone(probe_choice_mapping(_Policy(), _Context(), None))

    def test_the_probe_never_raises_out_of_the_run(self) -> None:
        from pokezero.truth_differential import probe_choice_mapping

        class _Policy:
            stats = _Stats()

            def _map_choices(self, context, aggregated):
                raise RuntimeError("boom")

        self.assertEqual(
            probe_choice_mapping(_Policy(), _Context(), None), "probe_raised:RuntimeError"
        )

    def test_a_construction_refusal_still_yields_a_mapping_reading(self) -> None:
        """The seam the compound forcings exposed, closed for this one stage."""

        records: list = []
        truth = _ScriptedPolicy(_refuse_construction, "truth")

        def _unmappable(context, aggregated):
            truth.stats.choices_unmapped_causes["aggregated_empty"] += 1
            return None

        truth._map_choices = _unmappable
        probe = TruthDifferentialProbe(
            primary=_ScriptedPolicy(_clean_world, "primary"),
            truth_policy=truth,
            truth_builder=_StubBuilder(),
            records=records,
            seed=1,
        )
        probe.select_action_with_context(_Context(), rng=random.Random(0))
        predicates = {r.predicate for r in records[0].refusals}
        self.assertIn("probe:choices_unmapped:aggregated_empty", predicates)
        # And it is filed under its own key, never confused with production's.
        self.assertNotIn("fallback:choices_unmapped", predicates)


class ForcingApparatusTests(unittest.TestCase):
    """D. The forcing apparatus is the evidence for every zero; pin it.

    Independent review's battery left M16 (abort mode patching the MODULE instead of
    the INSTANCE), M17 (`construct` as a silent no-op) and M20 (deleting the
    `--features model` gate) alive, with no coverage of the script beyond `build_plan`.
    """

    class _Native:
        def search_batched_multi_encoded(self, *a, **k):
            return "{}"

    class _Policy:
        def __init__(self) -> None:
            self.stats = _Stats()
            self.mapped = object()
            self.seen: list = []

        def _native(self):
            return ForcingApparatusTests._Native()

        def _map_choices(self, context, aggregated):
            return 7

        def select_action_with_context(self, context, *, rng):
            self.seen.append(("native", self._native()))
            self.seen.append(("map", self._map_choices(context, None)))
            import pokezero.engine_search as ES

            self.seen.append(("world", ES.world_battle_spec))
            return None

    def test_construct_forcing_actually_raises(self) -> None:
        """M17: a `--force construct` that quietly does nothing must not survive."""

        from pokezero.engine_world import EngineWorldUnsupported
        from truth_differential_census import install_forcing
        import pokezero.engine_search as ES

        policy = self._Policy()
        before = ES.world_battle_spec
        install_forcing(policy, "construct")
        policy.select_action_with_context(_Context(), rng=random.Random(0))
        during = dict(policy.seen)["world"]
        self.assertIsNot(during, before, "world_battle_spec was not patched at all")
        with self.assertRaises(EngineWorldUnsupported):
            during()
        self.assertIs(ES.world_battle_spec, before, "the patch was not restored")

    def test_abort_forcing_patches_the_instance_not_the_module(self) -> None:
        """M16: the exact hazard this PR sets in bold, now pinned.

        Patching the module attribute forces nothing, and the run then reports a
        clean zero and reads as a passing instrument test.
        """

        from truth_differential_census import install_forcing

        policy = self._Policy()
        class_native = type(policy)._native
        install_forcing(policy, "abort")
        policy.select_action_with_context(_Context(), rng=random.Random(0))
        native = dict(policy.seen)["native"]
        with self.assertRaises(RuntimeError):
            native.search_batched_multi_encoded()
        # The CLASS is untouched, and the instance attribute is cleaned up.
        self.assertIs(type(policy)._native, class_native)
        self.assertNotIn("_native", policy.__dict__)

    def test_unmapped_forcing_makes_map_choices_return_none(self) -> None:
        from truth_differential_census import install_forcing

        policy = self._Policy()
        install_forcing(policy, "unmapped")
        policy.select_action_with_context(_Context(), rng=random.Random(0))
        self.assertIsNone(dict(policy.seen)["map"])
        self.assertEqual(policy._map_choices(_Context(), None), 7, "not restored")

    def test_none_forcing_changes_nothing(self) -> None:
        from truth_differential_census import install_forcing

        policy = self._Policy()
        install_forcing(policy, "none")
        # Bound methods compare unequal by identity on each access, so assert the
        # absence of the wrapper attribute rather than method identity.
        self.assertNotIn("select_action_with_context", policy.__dict__)

    def test_an_unknown_force_mode_is_refused(self) -> None:
        from truth_differential_census import install_forcing

        with self.assertRaises(SystemExit):
            install_forcing(self._Policy(), "wobble")

    def test_a_crate_without_the_model_feature_is_refused(self) -> None:
        """M20: deleting the gate must not survive.

        On a non-model build `worlds_searched == worlds_constructed` identically, so
        every abort-channel reading from such a build is unfalsifiable.
        """

        import contextlib
        import io

        from truth_differential_census import require_model_feature

        # stderr captured: the gate dumps the witness, and unbuffered JSON between a
        # test's docstring line and its verdict breaks the CI pin's `-A2` window.
        noise = io.StringIO()
        with contextlib.redirect_stderr(noise):
            with self.assertRaises(SystemExit) as caught:
                require_model_feature({"pokezero_search_model_feature": False})
            self.assertIn("--features model", str(caught.exception))
            with self.assertRaises(SystemExit):
                require_model_feature({})
            require_model_feature({"pokezero_search_model_feature": True})  # no raise
        self.assertIn("pokezero_search_model_feature", noise.getvalue())


class WitnessCompletenessTests(unittest.TestCase):
    """M14: the witness's crate hash was unpinned, so a constant survived."""

    def test_the_crate_so_hash_is_a_real_digest_of_the_extension(self) -> None:
        witness = identity_witness()
        if "pokezero_search_so_sha256" not in witness:
            # The crate is optional; on a bare runner the witness records WHY it is
            # absent rather than omitting the fact, and that is what we check there.
            self.assertIn("unavailable", str(witness["pokezero_search_file"]))
            self.skipTest("pokezero_search is not installed on this runner")
        digest = witness["pokezero_search_so_sha256"]
        if digest == "no extension module found":
            self.skipTest("pokezero_search extension is not installed")
        self.assertRegex(digest, r"^[0-9a-f]{16}$")
        # A constant would pass the shape check, so also require it to TRACK the
        # bytes: hashing a mutated copy must give a different digest.
        import hashlib
        import pathlib

        from pokezero.truth_differential import _extension_hash

        class _Fake:
            __file__ = None

        root = pathlib.Path(
            __import__("pokezero_search").__file__  # noqa: PLC0415
        ).parent
        real = sorted(root.glob("*.so"))
        self.assertTrue(real, "no extension to hash")
        control = hashlib.sha256()
        control.update(real[0].name.encode("utf-8"))
        control.update(real[0].read_bytes())
        self.assertTrue(digest.startswith(control.hexdigest()[:8]) or len(real) > 1)

    def test_the_mismatch_diff_covers_the_content_keys_too(self) -> None:
        """L: diffing only paths cannot see two trees with the same layout."""

        source = Path(
            Path(__file__).resolve().parents[1] / "scripts" / "truth_differential_census.py"
        ).read_text(encoding="utf-8")
        for key in (
            "source_sha256",
            "truth_differential_present",
            "torch_version",
            "pokezero_search_model_feature",
        ):
            self.assertIn(f'"{key}"', source, f"{key} is not in the mismatch diff")


class CrossStageCensoringTests(unittest.TestCase):
    """The seam the corrected docstring now names, pinned so it cannot be re-claimed.

    Independent review measured this with compound forcings on the real runner:
    `--force construct,abort` reported the construction predicate ONLY. Pinned here
    at the unit level so the module's claim and its behaviour cannot drift again.
    """

    def test_the_producer_is_still_first_refuser_across_stages(self) -> None:
        def refuse_construction_only(stats: _Stats, context: Any) -> None:
            # The real chain: `continue` on a construction failure means the crate
            # is never reached, so no crate predicate can be recorded.
            _refuse_construction(stats, context)

        probe, records = _probe(_clean_world, refuse_construction_only)
        probe.select_action_with_context(_Context(), rng=random.Random(0))
        stages = {refusal.stage for refusal in records[0].refusals}
        self.assertIn("construction", stages)
        self.assertNotIn(
            "crate_search",
            stages,
            "if this ever passes, the chain gained cross-stage probing and the "
            "module docstring's seam 1 must be re-derived rather than re-worded",
        )

    def test_the_module_does_not_claim_more_than_it_does(self) -> None:
        from pokezero import truth_differential as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        self.assertIn("FOUR censoring seams remain", source)
        self.assertIn("not total per decision", source)
        # The sticky-fold seam, which the first revision omitted entirely.
        self.assertIn("_fold_broken`` is STICKY", source)

if __name__ == "__main__":
    unittest.main()
