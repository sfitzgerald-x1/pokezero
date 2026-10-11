from copy import deepcopy
from dataclasses import replace
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.mcts_eval.paper_reference import ReferenceRefusal, SamplingDeadlineExceeded
from pokezero.mcts_eval.paper_reference_factory import PublicRootWorldFactory
from pokezero.mcts_eval.paper_reference_parallel import PreparedDecision
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
from pokezero.mcts_eval.paper_reference_showdown import ShowdownTrajectoryWorld
from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_belief_diagnostics import (
    SCHEMA, FIELDS, BeliefDiagnosticWorkerFactory, PublicBeliefDiagnosticSearchAdapter,
    OracleBeliefDiagnosticSearchAdapter, _BeliefDiagnosticRuntime,
    capture_sampled_team, compare_sampled_teams, sampled_team_origin,
    selected_team_origin, team_fingerprints, truth_record)
from pokezero.mcts_eval.search_over_raw_oracle import TeamOracle, team_sha256
from pokezero.showdown_fixture import FixturePokemon


def team():
    return tuple(FixturePokemon(species, ("tackle", "protect"), ability="Pressure",
        item="Leftovers", level=80) for species in
        ("Snorlax", "Zapdos", "Suicune", "Celebi", "Jirachi", "Blissey"))


def truth():
    with patch.object(TeamOracle, "validate"):
        return truth_record(TeamOracle("p1", "a"*64, "b"*64, "c"*64, team(), team_sha256(team())),
            request=object(), source=object())


def row(ordinal=0, sampled=None):
    members = team_fingerprints(sampled or team())
    return dict(ordinal=ordinal, status="ROOT_VALIDATED", released=True,
        packed_team_sha256=team_sha256(sampled or team()),
        sampled_team_origin=dict(schema=SCHEMA, original_team_sha256=team_sha256(sampled or team()),
            members=members, members_sha256=digest(members)),
        sampled_original_team=dict(schema=SCHEMA, role="HYPOTHETICAL_DRAW_ONLY",
            root_binding="b"*64, subject="p1", information_key="0102", set_source_hash="a"*64,
            members=members, members_sha256=digest(members),
            original_team_sha256=team_sha256(sampled or team()),
            instrumentation_can_change_deadlines=True, qualifies_uninstrumented_runtime=False))


class BeliefDiagnosticsTests(unittest.TestCase):
    def compare(self, rows, **kwargs):
        return compare_sampled_teams(rows, truth=kwargs.get("truth", truth()), information_key="0102")

    def test_order_free_members_moves_and_supported_traits_only(self):
        reordered = tuple(reversed(team()))
        self.assertEqual(team_fingerprints(team()), team_fingerprints(reordered))
        changed = (replace(team()[0], moves=("protect", "tackle")), *team()[1:])
        result = self.compare([row(sampled=changed)])
        self.assertEqual(result["agreement_intervals"]["moves"], [1, 1])
        self.assertNotIn("exact_set", result["agreement_intervals"])
        self.assertFalse(result["exact_nature_gender_spread_identity_established"])

    def test_missing_species_disagrees_not_dropped(self):
        changed = (replace(team()[0], species="Gengar"), *team()[1:])
        result = self.compare([row(sampled=changed)])
        self.assertEqual(result["member_denominator"], 6)
        self.assertEqual(result["agreement_intervals"]["species"], [5/6, 5/6])
        self.assertEqual(result["agreement_intervals"]["item"], [5/6, 5/6])

    def test_all_unvalidated_worlds_keep_full_denominator(self):
        rows = [row(), dict(ordinal=1, status="DEADLINE_CANCELLED"),
            dict(ordinal=2, status="REFUSED"), dict(ordinal=3, status="STARTED")]
        result = self.compare(rows)
        self.assertEqual((result["attempted_worlds"], result["validated_worlds"], result["unresolved_worlds"]), (4, 1, 3))
        self.assertTrue(all(v == [.25, 1.] for v in result["agreement_intervals"].values()))
        self.assertFalse(result["significance_claim"])

    def test_no_draws_are_unknown_not_perfect(self):
        self.assertTrue(all(v == [0, 1] for v in self.compare([])["agreement_intervals"].values()))

    def test_trait_mismatch_is_separate(self):
        changed = (replace(team()[0], item="Lum Berry"), *team()[1:])
        result = self.compare([row(sampled=changed)])
        self.assertEqual(result["agreement_intervals"]["item"], [5/6, 5/6])
        self.assertEqual(result["agreement_intervals"]["ability"], [1, 1])

    def test_missing_validated_diagnostic_cannot_silently_drop(self):
        with self.assertRaisesRegex(ValueError, "lacks required diagnostic"):
            self.compare([dict(ordinal=0, status="ROOT_VALIDATED")])

    def test_unvalidated_measurements_and_duplicate_ordinals_refuse(self):
        bad = row(); bad["status"] = "DEADLINE_CANCELLED"
        for rows in ([bad], [row(), row()], [dict(ordinal=True, status="STARTED")]):
            with self.assertRaises(ValueError): self.compare(rows)

    def test_root_source_subject_and_information_drift_refuse(self):
        for key, value in (("root_binding", "d"*64), ("subject", "p2"),
                ("set_source_hash", "d"*64), ("information_key", "ffff"),
                ("role", "CONTROLLER_TRUTH_ONLY")):
            bad = row(); bad["sampled_original_team"][key] = value
            with self.assertRaisesRegex(ValueError, "boundary drift"): self.compare([bad])

    def test_malformed_fingerprints_and_truth_mutation_refuse(self):
        bad = row(); bad["sampled_original_team"]["members"][0]["moves"] = "nan"
        with self.assertRaises(ValueError): self.compare([bad])
        bad_truth = truth(); bad_truth["members"].append(bad_truth["members"][0])
        with self.assertRaises(ValueError): self.compare([], truth=bad_truth)
        oracle = TeamOracle("p1", "a"*64, "b"*64, "c"*64, team(), "f"*64)
        with self.assertRaisesRegex(ValueError, "immutable"):
            truth_record(oracle, request=object(), source=object())

    def test_default_observer_is_noop_without_snapshot_or_clock_call(self):
        factory = object.__new__(PublicRootWorldFactory)
        world, evidence = Mock(), dict(status="ROOT_VALIDATED")
        self.assertIs(factory._observe_team(world, evidence), world)
        world.close.assert_not_called()
        self.assertEqual(evidence, dict(status="ROOT_VALIDATED"))

    def test_fresh_enable_once_no_retroactive_enable(self):
        factory = object.__new__(PublicRootWorldFactory)
        factory.active, factory.receipts = False, []
        factory.enable_team_diagnostics("b"*64)
        with self.assertRaises(ReferenceRefusal): factory.enable_team_diagnostics("b"*64)
        for active, receipts, binding in ((True, [], "b"*64), (False, [{}], "b"*64), (False, [], "X"*64)):
            fresh = object.__new__(PublicRootWorldFactory)
            fresh.active, fresh.receipts = active, receipts
            with self.assertRaises(ReferenceRefusal): fresh.enable_team_diagnostics(binding)

    def test_clock_expiry_closes_world_and_removes_unaccepted_measurement(self):
        factory = object.__new__(PublicRootWorldFactory)
        factory.team_diagnostic_root_binding = "b"*64
        factory.check_sampling_deadline = Mock(side_effect=SamplingDeadlineExceeded("expired"))
        world, evidence = Mock(), dict(status="ROOT_VALIDATED")
        def capture(*args): evidence["sampled_original_team"] = {"temporary": True}
        with patch("pokezero.mcts_eval.search_over_raw_belief_diagnostics.capture_sampled_team", side_effect=capture):
            with self.assertRaises(SamplingDeadlineExceeded): factory._observe_team(world, evidence)
        world.close.assert_called_once()
        self.assertNotIn("sampled_original_team", evidence)

    def test_owned_world_capture_uses_hypothetical_not_truth(self):
        env = Mock(); env._search_snapshot_permitted = True
        # Match the real public-materialization lifecycle: actor-only opening
        # request; current opponent request is NOT original-team provenance.
        env.snapshot.return_value = SimpleNamespace(first_requests={"p1": {}},
            latest_requests={"p2": {"item": "mutated"}})
        world = ShowdownTrajectoryWorld(env, subject="p1", evaluator=Mock(), release=Mock())
        state = SimpleNamespace(key=b"\x01\x02")
        factory = SimpleNamespace(env=env, root=state, team_diagnostic_root_binding="b"*64,
            set_source=SimpleNamespace(metadata=SimpleNamespace(source_hash="a"*64)))
        evidence = row(); evidence.pop("sampled_original_team")
        with patch("pokezero.mcts_eval.paper_reference_showdown.decision_state", return_value=state):
            capture_sampled_team(factory, world, evidence)
        env.snapshot.assert_not_called()
        self.assertEqual(evidence["sampled_original_team"]["members"], truth()["members"])
        self.assertEqual(world.evaluator.call_count, 0)

    def test_actual_draw_hash_and_selected_particle_ancestry_are_required(self):
        from pokezero.mcts_eval.paper_reference_sampling import HiddenTeamDraw
        draw = HiddenTeamDraw(team(), (), (), team_sha256(team()))
        origin = sampled_team_origin(draw)
        self.assertEqual(origin, row()["sampled_team_origin"])
        with self.assertRaisesRegex(ValueError, "packed identity"):
            sampled_team_origin(replace(draw, packed_team_sha256="f"*64))
        nested = dict(substitute_particle_conditioning=dict(anchor=dict(
            substitute_particle_conditioning=dict(anchor=row()))))
        self.assertEqual(selected_team_origin(nested), origin)
        nested["substitute_particle_conditioning"]["anchor"]["substitute_particle_conditioning"]["anchor"]["packed_team_sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "actual sampled-team origin"): selected_team_origin(nested)

    def test_report_cannot_mix_measurement_with_another_origin(self):
        bad = row()
        bad["sampled_original_team"] = row(sampled=(replace(team()[0], item="Lum Berry"), *team()[1:]))["sampled_original_team"]
        with self.assertRaisesRegex(ValueError, "differs from selected"): self.compare([bad])

    def test_truth_requires_public_root_and_source_validation(self):
        oracle = TeamOracle("p1", "a"*64, "b"*64, "c"*64, team(), team_sha256(team()))
        request, source = object(), object()
        with patch.object(TeamOracle, "validate", side_effect=ValueError("binding changed")) as validate:
            with self.assertRaisesRegex(ValueError, "binding changed"):
                truth_record(oracle, request=request, source=source)
        validate.assert_called_once_with(request, source)

    def test_runtime_returns_same_prepared_sampler_and_rng(self):
        owned = Mock(); base = Mock()
        rng = random.Random(9); before = rng.getstate()
        prepared = PreparedDecision(Mock(), Mock(), Mock(return_value=owned))
        base.prepare.return_value = prepared
        base.prepared_factory = Mock()
        request = Mock()
        with patch("pokezero.mcts_eval.search_over_raw_oracle.public_root_binding", return_value="b"*64):
            observed = _BeliefDiagnosticRuntime(base).prepare(request)
        self.assertIs(observed, prepared)
        self.assertIs(observed.sample_world(rng), owned)
        observed.sample_world.assert_called_once_with(rng)
        self.assertEqual(rng.getstate(), before)
        base.prepared_factory.enable_team_diagnostics.assert_called_once_with("b"*64)

    def test_factory_is_spawn_safe_and_rejects_arbitrary_runtimes(self):
        import pickle
        base = ShowdownWorkerFactory("weights", "a"*64, "server", "b"*64)
        factory = BeliefDiagnosticWorkerFactory(base)
        self.assertEqual(pickle.loads(pickle.dumps(factory)), factory)
        self.assertNotIn(b"opponent_team", pickle.dumps(factory))
        with self.assertRaises(ValueError): BeliefDiagnosticWorkerFactory(object())

    def test_raw_is_not_a_belief_search_diagnostic(self):
        with self.assertRaisesRegex(ValueError, "search diagnostic arm required"):
            PublicBeliefDiagnosticSearchAdapter(SearchConfiguration("raw"))

    def test_actual_factory_direct_path_preserves_rng_world_and_sampler(self):
        from pokezero.mcts_eval.paper_reference_sampling import HiddenTeamDraw
        def run(observed):
            factory = object.__new__(PublicRootWorldFactory)
            env = Mock(); env._search_snapshot_permitted = True
            factory.env, factory.evaluator = env, Mock()
            factory.set_source = SimpleNamespace(metadata=SimpleNamespace(source_hash="a"*64))
            factory.state = SimpleNamespace(player_id="p1")
            factory.root = SimpleNamespace(key=b"\x01\x02")
            factory.active, factory.receipts, factory.sampling_deadline_at = False, [], None
            factory.pending_transition, factory.diagnostic_oracle_team = None, None
            factory.own_team, factory.known = team(), ()
            factory.anchor_gender_guidance_rows, factory.necessary_public_encore_support = [], {}
            draw = HiddenTeamDraw(team(), (), (), team_sha256(team()))
            factory.sampler = Mock(); factory.sampler.draw.return_value = draw
            if observed: factory.enable_team_diagnostics("b"*64)
            rng = random.Random(123)
            with patch("pokezero.mcts_eval.paper_reference_factory._public_reference_encore", return_value=None), \
                    patch("pokezero.mcts_eval.paper_reference_factory.induced_sleep_certificates", return_value={}), \
                    patch("pokezero.mcts_eval.paper_reference_factory.decision_state", return_value=factory.root), \
                    patch("pokezero.mcts_eval.paper_reference_showdown.decision_state", return_value=factory.root):
                world = factory(rng)
            factory.sampler.draw.assert_called_once_with((), rng)
            env.snapshot.assert_not_called()
            self.assertIs(world.env, env)
            self.assertEqual(factory.evaluator.call_count, 0)
            self.assertTrue(factory.active)
            world.close()
            self.assertFalse(factory.active)
            self.assertTrue(factory.receipts[0]["released"])
            return rng.getstate(), env.materialize_public_world.call_args, factory.receipts[0]
        default_state, default_call, default_receipt = run(False)
        observed_state, observed_call, observed_receipt = run(True)
        self.assertEqual(default_state, observed_state)
        self.assertEqual(default_call, observed_call)
        self.assertNotIn("sampled_team_origin", default_receipt)
        self.assertEqual(observed_receipt["sampled_team_origin"], row()["sampled_team_origin"])
        self.assertEqual(observed_receipt["sampled_original_team"]["members"], truth()["members"])

    def test_public_factory_composes_all_leaves_without_truth(self):
        from pokezero.mcts_eval.search_over_raw_leaves import ReferenceLeafWorkerFactory
        contract = SimpleNamespace(checkpoint_path="weights", checkpoint_sha256="a"*64,
            showdown_source_sha256="b"*64)
        factory = ShowdownWorkerFactory("weights", "a"*64, "server", "b"*64)
        for leaf in ("model", "hp_fraction", "raw_rollout"):
            with patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch") as pool:
                adapter = PublicBeliefDiagnosticSearchAdapter(SearchConfiguration("reference", leaf=leaf, workers=20),
                    checkpoint_contract=contract, showdown_root="server", reference_factory=factory)
                instrumented = pool.call_args.args[1]
                self.assertIsInstance(instrumented, BeliefDiagnosticWorkerFactory)
                base = instrumented.base
                self.assertIsInstance(base, ShowdownWorkerFactory if leaf == "model" else ReferenceLeafWorkerFactory)
                self.assertFalse(adapter.runtime_configuration["belief_diagnostics"]["truth_sent_to_public_worker"])
                self.assertFalse(adapter.runtime_configuration["belief_diagnostics"]["qualifies_uninstrumented_runtime"])
                self.assertEqual(adapter.runtime_sha256, digest(adapter.runtime_configuration))
                adapter.close()

    def test_oracle_factory_keeps_explicit_oracle_and_leaf_layers(self):
        from pokezero.mcts_eval.search_over_raw_oracle import OracleWorkerFactory
        from pokezero.mcts_eval.search_over_raw_leaves import ReferenceLeafWorkerFactory
        contract = SimpleNamespace(checkpoint_path="weights", checkpoint_sha256="a"*64,
            showdown_source_sha256="b"*64)
        factory = ShowdownWorkerFactory("weights", "a"*64, "server", "b"*64)
        source = SimpleNamespace(metadata=SimpleNamespace(source_hash="b"*64))
        oracle = TeamOracle("p1", "b"*64, "c"*64, "d"*64, team(), team_sha256(team()))
        for leaf in ("model", "hp_fraction", "raw_rollout"):
            with patch("pokezero.randbat.load_gen3_randbat_source_cached", return_value=source), \
                    patch("pokezero.mcts_eval.paper_reference_parallel.ParallelTrajectorySearch") as pool:
                adapter = OracleBeliefDiagnosticSearchAdapter(SearchConfiguration("reference", "oracle", leaf, workers=20),
                    oracle=oracle, checkpoint_contract=contract, showdown_root="server", reference_factory=factory)
                wrapped = pool.call_args.args[1]
                self.assertIsInstance(wrapped, BeliefDiagnosticWorkerFactory)
                if leaf != "model":
                    self.assertIsInstance(wrapped.base, ReferenceLeafWorkerFactory)
                    oracle_layer = wrapped.base.base
                else: oracle_layer = wrapped.base
                self.assertIsInstance(oracle_layer, OracleWorkerFactory)
                self.assertEqual(oracle_layer.base, factory)
                self.assertEqual(adapter.runtime_configuration["team_oracle"], oracle.receipt())
                adapter.close()

    def test_population_draw_binds_selected_retained_anchor(self):
        from pokezero.mcts_eval.paper_reference_particles import HypotheticalHistoryPopulation
        factory = object.__new__(PublicRootWorldFactory)
        env = Mock(); env._search_snapshot_permitted = True
        factory.env, factory.evaluator = env, Mock()
        factory.state = SimpleNamespace(player_id="p1")
        factory.root = SimpleNamespace(key=b"\x01\x02")
        factory.set_source = SimpleNamespace(metadata=SimpleNamespace(source_hash="a"*64))
        factory.active, factory.receipts, factory.sampling_deadline_at = True, [], None
        factory.team_diagnostic_root_binding = "b"*64
        population = object.__new__(HypotheticalHistoryPopulation)
        snapshot = object()
        population.factory, population.draws = factory, 0
        population.particles = [SimpleNamespace(snapshot=snapshot, ancestor=7,
            anchor_receipt=row(), steps=(), substitute_hp={})]
        population.receipt = dict(final_normalized_weights=[1.])
        evidence = dict(ordinal=0, status="STARTED")
        with patch("pokezero.mcts_eval.paper_reference_particles.public_history", return_value=("public",)), \
                patch("pokezero.mcts_eval.paper_reference_particles.decision_state", return_value=factory.root), \
                patch("pokezero.mcts_eval.paper_reference_showdown.decision_state", return_value=factory.root):
            world = factory._observe_team(population.draw(random.Random(9), evidence), evidence)
        env.restore_search_snapshot.assert_called_once_with(snapshot)
        env.snapshot.assert_not_called()
        self.assertEqual(evidence["substitute_particle_conditioning"]["selected_anchor_ancestor"], 7)
        self.assertEqual(evidence["sampled_original_team"]["members"], row()["sampled_original_team"]["members"])
        world.close()
        self.assertTrue(evidence["released"])


if __name__ == "__main__": unittest.main()
