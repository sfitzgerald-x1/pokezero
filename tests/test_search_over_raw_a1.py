"""Staged A1 contracts/lifecycle with synthetic environments; no real seeds."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import qualify_search_over_raw_a1 as driver
from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy
from pokezero.engine_world import EngineWorld
from pokezero.env import BattleStartOverride
from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_belief_diagnostics import IncumbentBeliefRecorder
from pokezero.showdown_fixture import pack_team
from tests.test_search_over_raw_belief_diagnostics import team, row, truth
from tests.test_engine_search import _FakeContext, _FakeObservation
from tests import test_engine_search as engine_tests


def progress():
    return dict(stage="test", source_games_completed=0, selections_completed=0,
        continuations_completed=0, roots_completed=0, fixed_roster=driver.planned_cells(),
        continuation_roster=driver.planned_continuations(), source_roster={str(s): dict(
            status="UNSTARTED_UNCERTAIN") for s in driver.SEEDS})


class IncumbentRecordingTests(unittest.TestCase):
    def setUp(self):
        self.request = SimpleNamespace(state=SimpleNamespace(player_id="p1"), set_source_hash="a"*64,
            observation=object(), pending_transition=None)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("pokezero.mcts_eval.search_over_raw_oracle.public_root_binding", return_value="b"*64))
        self.stack.enter_context(patch("pokezero.mcts_eval.paper_reference_showdown.decision_state",
            return_value=SimpleNamespace(key=bytes.fromhex("0102"))))
        self.stack.enter_context(patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture",
            return_value=self.request))
        self.recorder = IncumbentBeliefRecorder(self.request)
        self.context = SimpleNamespace(player_id="p1", public_materialization_state=object(), observation=object())
        self.override = BattleStartOverride(player_teams={"p1": pack_team(team()), "p2": pack_team(team())},
            observation_format_id="gen3randombattle")
        self.world = EngineWorld(None, {"p1": "side_one", "p2": "side_two"},
            {"p1": (), "p2": tuple(r["species"] for r in row()["sampled_original_team"]["members"])})

    def test_actual_packed_draw_and_construction_are_joined_without_truth(self):
        self.recorder.attempt(self.context, 0, self.override)
        self.recorder.constructed(self.context, self.world, object())
        measured = self.recorder.finish(1, 1)
        self.assertEqual(measured[0]["sampled_original_team"]["members"], row()["sampled_original_team"]["members"])
        self.assertEqual(measured[0]["sampled_original_team"]["role"], "HYPOTHETICAL_DRAW_ONLY")

    def test_failed_sampling_and_construction_remain_full_attempt_uncertainty(self):
        self.recorder.attempt(self.context, 0, None)
        self.recorder.attempt(self.context, 1, self.override)
        self.assertEqual([r["status"] for r in self.recorder.finish(2, 0)], ["REFUSED", "STARTED"])

    def test_missing_attempt_constructed_and_duplicate_callbacks_cannot_pass(self):
        self.recorder.attempt(self.context, 0, self.override)
        for counts in ((2, 0), (1, 1)):
            with self.assertRaisesRegex(ValueError, "reconciliation"):
                self.recorder.finish(*counts)
        self.recorder.constructed(self.context, self.world, object())
        with self.assertRaises(ValueError):
            self.recorder.constructed(self.context, self.world, object())
        with self.assertRaisesRegex(ValueError, "reconciliation"):
            self.recorder.finish(1, 1)

    def test_swallowed_observer_errors_are_still_diagnostic_refusals(self):
        policy = EngineMctsPolicy(dex=None, set_source=None, module=object())
        policy._world_attempt_observer = self.recorder.attempt
        with self.assertWarns(Warning):
            policy._notify_world_attempt_observer(self.context, 9, self.override)
        with self.assertRaisesRegex(ValueError, "reconciliation"):
            self.recorder.finish(1, 0)

    def test_constructed_party_mismatch_refuses(self):
        self.recorder.attempt(self.context, 0, self.override)
        bad = EngineWorld(None, self.world.slot_sides, {"p2": ("gengar",)})
        with self.assertRaises(ValueError):
            self.recorder.constructed(self.context, bad, object())

    def test_cosmetic_engine_species_matches_actual_original_draw(self):
        from dataclasses import replace
        from pokezero.mcts_eval.search_over_raw_belief_diagnostics import team_fingerprints
        cosmetic = (replace(team()[0], species="Unown-C"), *team()[1:])
        override = BattleStartOverride(player_teams={"p1": pack_team(team()), "p2": pack_team(cosmetic)},
            observation_format_id="gen3randombattle")
        self.recorder.attempt(self.context, 0, override)
        self.assertIn("unown", [r["species"] for r in team_fingerprints(cosmetic)])
        # Use all actual members, changing only the cosmetic spelling.
        world = EngineWorld(None, self.world.slot_sides, {"p2": tuple("unownc" if r["species"] == "unown"
            else r["species"] for r in team_fingerprints(cosmetic))})
        self.recorder.constructed(self.context, world, object())
        self.recorder.finish(1, 1)

    def test_failed_selection_exports_swallowed_errors_missing_attempts_and_releases_hooks(self):
        from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
        from pokezero.mcts_eval.search_over_raw_belief_diagnostics import PublicBeliefDiagnosticSearchAdapter
        adapter = object.__new__(PublicBeliefDiagnosticSearchAdapter)
        native = EngineMctsPolicy(dex=None, set_source=None, module=object())
        adapter._native = native
        def failed_select(_self, context, **kwargs):
            adapter._belief_recorder = self.recorder
            adapter._belief_before = (0, 0)
            native._world_attempt_observer, native._world_observer = self.recorder.attempt, self.recorder.constructed
            self.recorder.attempt(self.context, 0, self.override)
            native.stats.worlds_attempted = 2
            adapter.last_failure = dict(status="UNCERTAIN_REFUSED")
            raise ValueError("synthetic native failure")
        with patch.object(PublicModelSearchAdapter, "select", new=failed_select):
            with self.assertRaisesRegex(ValueError, "native failure"):
                adapter.select(self.context, root_id="test", selection_seed=0)
        e = adapter.last_failure["attempted_belief_evidence"]
        self.assertEqual(e["unobserved_attempt_ordinals"], [1])
        self.assertEqual(len(e["observed_draws"]), 1)
        self.assertFalse(e["diagnostic_qualified"])
        self.assertIsNone(native._world_attempt_observer)
        self.assertIsNone(native._world_observer)

    def test_actual_engine_loop_calls_both_hooks_and_preserves_sampler_rng_and_actions(self):
        candidates = [{"action_index": 0, "kind": "move", "legal": True, "move_id": "earthquake"},
            {"action_index": 1, "kind": "move", "legal": True, "move_id": "surf"}]
        context = _FakeContext(_FakeObservation((True, True, False, False, False, False, False, False, False), candidates))
        observations = []
        for enabled in (False, True):
            module = Mock()
            module.monte_carlo_tree_search.return_value = engine_tests.OwnSideSelectionTests._Result()
            policy = EngineMctsPolicy(dex=None, set_source=None, module=module,
                config=EngineMctsConfig(worlds=3, sample_retry_factor=1))
            recorder = IncumbentBeliefRecorder(self.request)
            if enabled:
                policy._world_attempt_observer, policy._world_observer = recorder.attempt, recorder.constructed
            rng, samples = random.Random(7), []
            def sample(**kwargs):
                samples.append(kwargs["rng"].random())
                return self.override, None
            with patch("pokezero.engine_search._gen3_randbat_belief_start_override_result", side_effect=sample), \
                 patch("pokezero.engine_search.world_battle_spec", return_value=self.world) as materialize, \
                 patch("pokezero.engine_search.build_poke_engine_state", return_value=object()):
                selected = policy.select_action_with_context(context, rng=rng)
            observations.append((selected.action_index, samples, rng.getstate(), materialize.call_count))
            if enabled:
                self.assertEqual(len(recorder.finish(policy.stats.worlds_attempted, policy.stats.worlds_constructed)), 3)
        self.assertEqual(observations[0], observations[1])


class StagedA1Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name).resolve()

    def test_exact_five_model_selectors_keep_full_allocations(self):
        configs = driver.configurations()
        self.assertEqual([(c.arm, c.belief) for c in configs], [("raw", "public"),
            ("incumbent", "public"), ("incumbent", "oracle"), ("reference", "public"), ("reference", "oracle")])
        self.assertTrue(all(c.seconds == 1. and c.leaf == "model" for c in configs))
        self.assertTrue(all(c.workers == (20 if c.arm == "reference" else 1) for c in configs))
        self.assertEqual(len(driver.planned_cells()), 20)
        self.assertEqual(len(driver.planned_continuations()), 160)

    def test_exposure_refuses_overlap_and_includes_closed_pools_and_accidental_fixtures(self):
        path = self.output / "exposure.json"
        driver.save_new(path, dict(seeds=[123]))
        excluded, _ = driver.exposure_inventory([path])
        self.assertTrue(set(range(4711, 4736)) <= set(excluded))
        self.assertIn(2026101023, excluded)
        self.assertFalse(set(driver.SEEDS) & set(excluded))
        other = self.output / "overlap.json"
        driver.save_new(other, dict(seeds=[driver.SEEDS[0]]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            driver.exposure_inventory([other])

    def test_public_constructor_never_receives_truth_and_oracle_is_explicit(self):
        oracle = object()
        with patch("pokezero.mcts_eval.search_over_raw_belief_diagnostics.PublicBeliefDiagnosticSearchAdapter") as public, \
             patch("pokezero.mcts_eval.search_over_raw_belief_diagnostics.OracleBeliefDiagnosticSearchAdapter") as privileged:
            driver.make_adapter(driver.configurations()[1], oracle=oracle, checkpoint_contract="contract")
            self.assertNotIn("oracle", public.call_args.kwargs)
            driver.make_adapter(driver.configurations()[2], oracle=oracle, checkpoint_contract="contract")
            self.assertIs(privileged.call_args.kwargs["oracle"], oracle)

    def root_fixture(self, *, fail=None, capped=False):
        root_id = f"excluded:{driver.SEEDS[0]}:0"
        root = dict(root_id=root_id, public_record_sha256="f"*64,
            public_record=dict(seed=driver.SEEDS[0], recorded_action_index=0))
        p = progress()
        lifecycle = []
        class Env:
            def restore(self, snapshot):
                self.done = False
            def terminal(self):
                return SimpleNamespace(capped=capped, winner="p1") if self.done else None
            def requested_players(self):
                return ("p1", "p2")
            def observe(self, seat):
                return object()
            def reseed_simulator_rng(self, seed):
                pass
            def step(self, actions):
                self.done = True
        class Adapter:
            last_failure = None
            def __init__(self, cfg):
                self.cfg = cfg
                self.runtime_configuration = dict(configuration=asdict(cfg))
            def select(self, context, **kwargs):
                lifecycle.append(("select", self.cfg.arm, self.cfg.belief))
                if fail == (self.cfg.arm, self.cfg.belief):
                    self.last_failure = dict(status="UNCERTAIN_REFUSED", retry_authorized=False)
                    raise ValueError("synthetic selection failure")
                return dict(root_id=kwargs["root_id"], configuration_sha256=self.cfg.identity,
                    runtime_sha256=digest(self.runtime_configuration), action=0, status="SELECTED", elapsed_seconds=.01,
                    evidence=dict(total_iterations=1, result=dict(trajectories=1),
                        belief_draws=[row()], worker_receipts=[dict(evidence=dict(draws=[row()]))]))
            def close(self):
                lifecycle.append(("close", self.cfg.arm, self.cfg.belief))
        stack = ExitStack()
        self.addCleanup(stack.close)
        request = SimpleNamespace(state=SimpleNamespace(player_id="p1"), observation=object())
        stack.enter_context(patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture", return_value=request))
        stack.enter_context(patch("pokezero.mcts_eval.search_over_raw_oracle.TeamOracle.capture", return_value=object()))
        stack.enter_context(patch("pokezero.mcts_eval.search_over_raw_belief_diagnostics.truth_record", return_value=truth()))
        stack.enter_context(patch("pokezero.mcts_eval.paper_reference_showdown.decision_state",
            return_value=SimpleNamespace(key=bytes.fromhex("0102"))))
        stack.enter_context(patch.object(driver, "make_adapter", side_effect=lambda cfg, **kwargs: Adapter(cfg)))
        kwargs = dict(root=root, context=SimpleNamespace(public_materialization_state=object(), observation=
            SimpleNamespace(legal_action_mask=(True, True))), pending=None, snapshot=object(), output=self.output,
            progress=p, env=Env(), evaluator=lambda obs: ((0, 1), SimpleNamespace(priors=(.6, .4))),
            factory=object(), contract=object(), source=object(), showdown="none", verify=Mock())
        return kwargs, p, lifecycle

    def test_one_root_runs_all_five_selectors_then_eight_shared_terminal_outcomes(self):
        kwargs, p, events = self.root_fixture()
        driver.collect_root(**kwargs)
        self.assertEqual((p["selections_completed"], p["roots_completed"], p["continuations_completed"]), (5, 1, 8))
        self.assertEqual(len(events), 10)
        self.assertTrue(all(events[i][0] == "close" for i in range(1, 10, 2)))
        cells = [r for r in p["continuation_roster"] if r["root_id"] == kwargs["root"]["root_id"]]
        self.assertEqual(len(cells), 40)
        self.assertTrue(all(c["status"] == "COMPLETE" for c in cells))
        self.assertEqual(len(list(self.output.glob("outcome-*.json"))), 8)
        self.assertTrue((self.output / "audit.json").exists())

    def test_selection_failure_stops_immediately_cleans_and_keeps_remaining_roster(self):
        kwargs, p, events = self.root_fixture(fail=("incumbent", "oracle"))
        with self.assertRaisesRegex(ValueError, "synthetic selection"):
            driver.collect_root(**kwargs)
        self.assertEqual((p["selections_completed"], p["roots_completed"], p["continuations_completed"]), (2, 0, 0))
        self.assertEqual(events[-1], ("close", "incumbent", "oracle"))
        self.assertFalse(any(e[1] == "reference" for e in events))
        self.assertFalse((self.output / "continuation-attempt.json").exists())
        self.assertTrue(all(c["contrast_interval"] == [-1, 1] for c in p["fixed_roster"]))

    def test_capped_continuation_stops_and_preserves_all_aliases_as_uncertain(self):
        kwargs, p, _ = self.root_fixture(capped=True)
        with self.assertRaisesRegex(ValueError, "capped/refused"):
            driver.collect_root(**kwargs)
        self.assertEqual(p["roots_completed"], 0)
        self.assertEqual(p["continuations_completed"], 0)
        self.assertFalse((self.output / "audit.json").exists())
        self.assertEqual(sum(c["status"] == "CAPPED" for c in p["continuation_roster"]), 5)
        self.assertTrue(all(c["signed_outcome"] is None for c in p["continuation_roster"]))

    def test_missing_agreement_refuses_before_any_continuation(self):
        kwargs, p, events = self.root_fixture()
        with patch.object(driver, "agreement", side_effect=ValueError("missing actual diagnostic")):
            with self.assertRaisesRegex(ValueError, "actual diagnostic"):
                driver.collect_root(**kwargs)
        self.assertEqual(p["selections_completed"], 1)
        self.assertFalse((self.output / "continuation-attempt.json").exists())
        self.assertEqual(events[-1], ("close", "incumbent", "public"))

    def registration(self):
        path = self.output / "prior.json"
        driver.save_new(path, dict(seeds=[123]))
        excluded, exposure = driver.exposure_inventory([path])
        return dict(schema=driver.SCHEMA, namespace=driver.NAMESPACE,
            configurations=[asdict(c) for c in driver.configurations()], source_contract=driver.source_contract(),
            seeds=list(driver.SEEDS), fixture_seed=driver.SEEDS[0], candidate_seat="p1", initial_dispatch_workers=6,
            required_independent_review=True, retry_authorized=False, phase_a_admission=False,
            phase_b_authorized=False, scientific_strength_evidence=False, qualifies_uninstrumented_runtime=False,
            representative_runtime_evidence=False, historical_attempt_reused=False,
            attempt_directory=str(self.output), source_root=str(driver.ROOT), source_commit="test-head",
            excluded_seeds=excluded, excluded_seed_inventory_sha256=digest(excluded), exposure_registrations=exposure,
            input_sha256={**exposure, **{str(p): driver.sha256_file(p) for p in driver.bound_files()}})

    def test_registration_rejects_old_runner_extra_config_and_admission(self):
        r = self.registration()
        driver.validate_contract(r, self.output)
        for mutation in (dict(schema="pokezero.search-over-raw.real-source-feasibility.v3"),
                         dict(seeds=[2026101022, 2026101023]), dict(phase_a_admission=True),
                         dict(initial_dispatch_workers=20), dict(required_independent_review=False)):
            bad = deepcopy(r)
            bad.update(mutation)
            with self.assertRaisesRegex(ValueError, "core drift"):
                driver.validate_contract(bad, self.output)
        bad = deepcopy(r)
        bad["configurations"].append(bad["configurations"][-1])
        with self.assertRaisesRegex(ValueError, "core drift"):
            driver.validate_contract(bad, self.output)

    def prepare_run(self, r, review):
        driver.save_new(self.output / "registration.json", r)
        driver.save_new(self.output / "registration-binding.json", dict(registration_sha256=digest(r)))
        path = self.output / "review.json"
        driver.save_new(path, review)
        return path

    def clearance(self, r):
        return dict(disposition="CLEAR_ONE_STAGED_A1_ENGINEERING_ATTEMPT",
            registration_sha256=digest(r), source_commit=r["source_commit"], independent_reviewer="test-other-agent",
            scientific_admission=False)

    def test_missing_wrong_review_does_not_claim_or_enter_runtime(self):
        r = self.registration()
        bad = self.clearance(r)
        bad["registration_sha256"] = "0"*64
        path = self.prepare_run(r, bad)
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure") as runtime:
            with self.assertRaisesRegex(ValueError, "independent staged review"):
                driver.run(self.output, path)
        runtime.assert_not_called()
        self.assertFalse((self.output / "attempt.json").exists())

    def test_terminal_failure_exit1_keeps_full_unknown_roster_and_never_reopens(self):
        r = self.registration()
        path = self.prepare_run(r, self.clearance(r))
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure", side_effect=ValueError("fake failure")) as runtime:
            self.assertEqual(driver.run(self.output, path), 1)
            terminal = json.loads((self.output / "terminal.json").read_text())
            self.assertEqual((terminal["exit_code"], terminal["status"]), (1, "FAILED_NO_RETRY"))
            self.assertEqual(len(terminal["fixed_roster"]), 20)
            self.assertEqual(len(terminal["continuation_roster"]), 160)
            self.assertEqual(terminal["roots_completed"], 0)
            with self.assertRaisesRegex(ValueError, "no retry"):
                driver.run(self.output, path)
            self.assertEqual(runtime.call_count, 1)

    def full_artifact_fixture(self):
        p = progress()
        for seed in driver.SEEDS:
            source_dir = self.output / f"source-{seed}"
            source_dir.mkdir()
            indices = [1, 2]
            selected = driver.select_source_requests(driver.NAMESPACE, seed, indices, 2)
            records = {i: dict(seed=seed, recorded_action_index=0, source_request_index=i) for i in indices}
            eligible = [dict(source_request_index=i, public_record_sha256=digest(records[i])) for i in indices]
            roots = [dict(root_id=f"excluded:{seed}:{slot}", source_request_index=i,
                public_record=records[i], public_record_sha256=digest(records[i])) for slot, i in enumerate(selected)]
            source = dict(contract_sha256=digest(driver.source_contract()), source_seed=seed, panel="excluded",
                status="COMPLETE", source_terminal_complete=True, source_policy="raw_argmax_both_seats",
                eligible_requests=2, eligible_public_records=eligible, eligible_catalog_sha256=digest(eligible),
                requested_root_slots=2, missing_root_ids=[], roots=roots)
            driver.save_new(source_dir / "source.json", source)
            p["source_roster"][str(seed)]["status"] = "COMPLETE"
            p["source_games_completed"] += 1
            for root in roots:
                kwargs, _, _ = self.root_fixture()
                d = source_dir / f"root-{root['root_id'].rsplit(':', 1)[1]}"
                d.mkdir()
                kwargs.update(root=root, output=d, progress=p)
                with patch("builtins.print"):
                    driver.collect_root(**kwargs)
        return p

    def test_complete_durable_roster_reconciles_twenty_selectors_and_all_160_aliases(self):
        p = self.full_artifact_fixture()
        driver.verify_completion(self.output, p)
        self.assertEqual((p["roots_completed"], p["selections_completed"], p["continuations_completed"]), (4, 20, 32))
        p["continuation_roster"][-1]["signed_outcome"] = -1
        with self.assertRaisesRegex(ValueError, "alias"):
            driver.verify_completion(self.output, p)

    def test_durable_verifier_rejects_missing_world_measurement_and_root_drift(self):
        p = self.full_artifact_fixture()
        d = self.output / f"source-{driver.SEEDS[0]}" / "root-0"
        original = json.loads((d / "root-binding.json").read_text())
        original["information_key"] = "wrong-root"
        # In-memory loader mutation avoids overwriting append-only receipts.
        old_load = json.loads
        def modified(payload, **kwargs):
            value = old_load(payload, **kwargs)
            if type(value) is dict and value.get("root_id") == f"excluded:{driver.SEEDS[0]}:0" and "information_key" in value:
                return original
            return value
        with patch.object(driver.json, "loads", side_effect=modified):
            with self.assertRaisesRegex(ValueError, "boundary drift"):
                driver.verify_completion(self.output, p)

    def test_source_priority_and_missing_slots_refuse_without_replacement(self):
        p = self.full_artifact_fixture()
        source = json.loads((self.output / f"source-{driver.SEEDS[0]}" / "source.json").read_text())
        source["roots"].reverse()
        with self.assertRaisesRegex(ValueError, "priority"):
            driver.validate_source(source, driver.SEEDS[0])

    def test_failure_receipt_write_error_cannot_skip_cleanup_or_mask_selection_failure(self):
        kwargs, p, events = self.root_fixture(fail=("incumbent", "oracle"))
        save = driver.save_new
        def faulty(path, value):
            if str(path).endswith("-failure.json"):
                raise OSError("synthetic full disk")
            save(path, value)
        with patch.object(driver, "save_new", side_effect=faulty):
            with self.assertRaisesRegex(ValueError, "synthetic selection"):
                driver.collect_root(**kwargs)
        self.assertEqual(events[-1], ("close", "incumbent", "oracle"))
        self.assertEqual(p["evidence_write_failures"], ["OSError"])

    def test_complete_but_all_forced_roots_do_not_qualify(self):
        p = self.full_artifact_fixture()
        saved = driver.selection_work
        def forced(cfg, selected, legal):
            return saved(cfg, selected, 1)
        old_load = json.loads
        def forced_artifact(payload, **kwargs):
            value = old_load(payload, **kwargs)
            if type(value) is dict and "legal_choices" in value:
                value.update(legal_choices=1)
                if "forced_action" in value:
                    value.update(forced_action=True, substantive_search_exercised=False)
            return value
        with patch.object(driver, "selection_work", side_effect=forced), patch.object(driver.json, "loads", side_effect=forced_artifact):
            with self.assertRaisesRegex(ValueError, "NO_SUBSTANTIVE_A1_EXERCISE"):
                driver.verify_completion(self.output, p)


if __name__ == "__main__":
    unittest.main()
