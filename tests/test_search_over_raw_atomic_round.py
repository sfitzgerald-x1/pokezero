"""Synthetic contract/lifecycle and explicit diagnostics plumbing regressions."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import diagnose_search_over_raw_atomic_round as driver
from pokezero.engine_search import EngineMctsPolicy, EngineSearchFallbackError
from pokezero.mcts_eval.lattice import _LiveEngineTimingDecider
from pokezero.mcts_eval.manifest import SearchConfig
from pokezero.mcts_eval.policy_opponent_profile import make_profile_decider
from pokezero.mcts_eval.resolver import ContractError
from pokezero.mcts_eval.search_over_raw import ENGINEERING_EXCLUDED_SEEDS, SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter, _incumbent_runtime
from pokezero.mcts_eval.search_over_raw_source import AuditedRawPolicy, collect_raw_source
from pokezero.policy_opponent_diagnostics import PHASES, PolicyOpponentDiagnostics
from test_policy_opponent_diagnostics import arguments
from test_search_over_raw_source import FakeChampion, SourceEnv, SHA


PROBE_SEED = 2026101021
PROBE_NAMESPACE = "7d2df0c6-8c3f-4ce7-a801-50e1e9cf928b"


def synthetic_source(seed=driver.SEED, namespace=driver.NAMESPACE):
    return collect_raw_source(driver.source_contract(seed, namespace), panel="excluded", source_seed=seed,
        env=SourceEnv(), policies={s: AuditedRawPolicy(FakeChampion(), checkpoint_sha256=SHA)
            for s in ("p1", "p2")})


class PlumbingTests(unittest.TestCase):
    def test_adapter_disabled_call_shape_and_identity_are_preserved(self):
        contract = SimpleNamespace(checkpoint_sha256=SHA, showdown_source_sha256="source")
        identities = []
        for sink in (None, PolicyOpponentDiagnostics()):
            with patch("pokezero.mcts_eval.search_over_raw_adapters._incumbent_runtime",
                    return_value=(Mock(), Mock(), "config")) as runtime:
                adapter = PublicModelSearchAdapter(driver.CONFIGURATION, checkpoint_contract=contract,
                    showdown_root="showdown", policy_opponent_diagnostics=sink)
                self.addCleanup(adapter.close)
                expected = {"leaf": "raw_rollout"}
                if sink is not None:
                    expected["policy_opponent_diagnostics"] = sink
                self.assertEqual(runtime.call_args.kwargs, expected)
                self.assertEqual("callback_diagnostics" in adapter.runtime_configuration, sink is not None)
                identities.append(adapter.runtime_sha256)
        self.assertNotEqual(*identities)

    def test_wrong_adapter_arm_leaf_or_custom_sink_refuses_before_runtime(self):
        class Custom(PolicyOpponentDiagnostics):
            pass
        for cfg, sink in [(SearchConfiguration("raw"), PolicyOpponentDiagnostics()),
                (SearchConfiguration("reference", workers=20), PolicyOpponentDiagnostics()),
                (SearchConfiguration("incumbent"), PolicyOpponentDiagnostics()),
                (driver.CONFIGURATION, object()), (driver.CONFIGURATION, Custom())]:
            with patch("pokezero.mcts_eval.search_over_raw_adapters._incumbent_runtime") as runtime:
                with self.assertRaisesRegex(ValueError, "callback diagnostics"):
                    PublicModelSearchAdapter(cfg, checkpoint_contract=None, showdown_root="",
                        policy_opponent_diagnostics=sink)
                runtime.assert_not_called()

    def test_incumbent_factory_forwards_only_opted_in_sink(self):
        for sink in (None, PolicyOpponentDiagnostics()):
            with patch("pokezero.mcts_eval.policy_opponent_profile.make_profile_decider") as factory:
                _incumbent_runtime(None, "showdown", 10., leaf="raw_rollout", policy_opponent_diagnostics=sink)
                self.assertEqual("policy_opponent_diagnostics" in factory.call_args.kwargs, sink is not None)
                if sink is not None:
                    self.assertIs(factory.call_args.kwargs["policy_opponent_diagnostics"], sink)
                factory.return_value._policy_for.assert_called_once_with(
                    SearchConfig(depth=6, sims=4096, batch=16, worlds=4, inference_mode="local"))

    def test_profile_forwards_sink_and_preserves_raw_allocations(self):
        sink = PolicyOpponentDiagnostics()
        with patch("pokezero.mcts_eval.policy_opponent_profile._LiveEngineTimingDecider") as factory:
            make_profile_decider(None, "showdown", arm="incumbent_mcts", mode="matched_deadline",
                opponent_seed=7, deadline_ms=10000, native_batch_guard_ms=64,
                model_leaf_override="raw_policy_terminal", policy_opponent_diagnostics=sink)
        kwargs = factory.call_args.kwargs
        self.assertIs(kwargs["policy_opponent_diagnostics"], sink)
        self.assertEqual((kwargs["model_world_workers"], kwargs["rollout_count"], kwargs["rollout_threads"],
            kwargs["rollout_max_plies"], kwargs["rollout_branch_on_damage"]), (1, 1, 1, 250, True))

    def test_profile_and_lattice_reject_sink_without_callback_runtime(self):
        with self.assertRaisesRegex(ContractError, "callback diagnostics"):
            make_profile_decider(None, "", arm="raw_policy", mode="matched_deadline",
                opponent_seed=0, deadline_ms=10000, native_batch_guard_ms=64,
                policy_opponent_diagnostics=PolicyOpponentDiagnostics())
        with self.assertRaisesRegex(ValueError, "callback diagnostics"):
            _LiveEngineTimingDecider(None, "", policy_opponent_diagnostics=PolicyOpponentDiagnostics())
        with self.assertRaisesRegex(TypeError, "exact aggregate sink"):
            EngineMctsPolicy(dex=None, set_source=None, policy_opponent_diagnostics=Mock())

    def test_engine_callback_receives_exact_sink_without_default_kwargs_change(self):
        args = arguments()
        policy = object.__new__(EngineMctsPolicy)
        policy._policy_opponent_result, policy._policy_opponent_model = args["result"], args["model"]
        tokens = args["result"].model_config.category_vocab
        policy._tables_json = json.dumps(dict(vocab=dict(tokens=tokens,
            oov_buckets=args["result"].model_config.category_oov_buckets,
            index={token.lower(): i for i, token in enumerate(tokens, 1)})))
        policy._set_source, policy._dex = args["set_source"], args["dex"]
        policy._config, policy._policy_opponent_inference_lock = SimpleNamespace(model_device="cpu"), None
        context = SimpleNamespace(public_materialization_state=SimpleNamespace(replay=SimpleNamespace(
            public_lines=args["public_lines"], hp_visibility=args["hp_visibility"])),
            battle_id="fixture", seed=7, format_id="gen3randombattle")
        for sink in (None, PolicyOpponentDiagnostics()):
            if sink is not None:
                policy._policy_opponent_diagnostics = sink
            with patch("pokezero.policy_opponent.make_policy_opponent_callback") as factory:
                policy._checkpoint_policy_callback(context, "p2", raw_argmax=True)
            self.assertEqual("diagnostics" in factory.call_args.kwargs, sink is not None)
            if sink is not None:
                self.assertIs(factory.call_args.kwargs["diagnostics"], sink)

    def test_lattice_passes_sink_to_engine_without_default_call_shape_change(self):
        decider = object.__new__(_LiveEngineTimingDecider)
        values = dict(contract=SimpleNamespace(checkpoint_path="weights", model_device="cpu"),
            artifacts=dict(model_path="model", tables_path="tables"), dex=None, set_source=None,
            annotation_source=None, model_priors=True, use_opponent_priors=False,
            override_telemetry=True, record_joint_actions=True, model_leaf_override="raw_policy_terminal",
            model_decision_time_ms=10000, model_native_batch_guard_ms=64, model_world_workers=1,
            rollout_leaf_eval=False, rollout_count=1, rollout_max_plies=250,
            rollout_policy="raw_argmax", rollout_seed=7, rollout_threads=1,
            rollout_threads_cpu_budget_ack=False, rollout_branch_on_damage=True)
        for key, value in values.items():
            setattr(decider, "_" + key, value)
        decider._policy_opponent_kwargs = {"strict_fallbacks": True}
        for sink in (None, PolicyOpponentDiagnostics()):
            decider._policies = {}
            decider._policy_opponent_diagnostics = sink
            with patch("pokezero.engine_search.EngineMctsPolicy") as factory:
                decider._policy_for(SearchConfig(depth=6, sims=4096, batch=16, worlds=4))
            self.assertEqual("policy_opponent_diagnostics" in factory.call_args.kwargs, sink is not None)
            if sink is not None:
                self.assertIs(factory.call_args.kwargs["policy_opponent_diagnostics"], sink)


class DiagnosticDriverTests(unittest.TestCase):
    def setUp(self):
        quiet = patch.object(driver, "print", create=True)
        quiet.start()
        self.addCleanup(quiet.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name).resolve()
        self.exposure = self.output / "prior.json"
        driver.save_new(self.exposure, dict(seeds=[123]))
        excluded, bindings = driver.exposure_inventory([self.exposure], seed=PROBE_SEED)
        self.registration = dict(driver.core(PROBE_SEED, PROBE_NAMESPACE), source_root=str(driver.ROOT),
            attempt_directory=str(self.output), exposure_registrations=bindings,
            excluded_seeds=excluded, excluded_seed_inventory_sha256=digest(excluded),
            input_sha256={**bindings, **{str(p): driver.sha256_file(p) for p in driver.bound_driver_files()}})

    def setup_run(self):
        driver.save_new(self.output / "registration.json", self.registration)
        driver.save_new(self.output / "registration-binding.json", dict(registration_sha256=digest(self.registration)))

    def test_fixed_new_seed_budget_order_and_same_root_law(self):
        self.assertIn(driver.SEED, ENGINEERING_EXCLUDED_SEEDS)
        self.assertEqual(driver.SEED, 2026101020)
        self.assertEqual(driver.MODES, ("instrumented", "disabled"))
        self.assertEqual(asdict(driver.CONFIGURATION), dict(arm="incumbent", belief="public",
            leaf="raw_rollout", seconds=10., workers=1))
        self.assertEqual(driver.core()["native_allocation"]["batch"], 16)
        self.assertEqual(driver.core()["native_allocation"]["worlds"], 4)
        self.assertEqual(len(driver.source_contract()["panels"]["excluded"]["root_slots"]), 1)
        fresh = driver.core(PROBE_SEED, PROBE_NAMESPACE)
        self.assertEqual(fresh["seeds"], [PROBE_SEED])
        self.assertEqual(fresh["native_allocation"], driver.core()["native_allocation"])
        self.assertNotEqual(fresh["selection_seed"], driver.core()["selection_seed"])
        self.assertEqual(fresh["source_contract"]["panels"]["excluded"]["root_slots"][0]["root_id"],
            f"excluded:{PROBE_SEED}:0")
        self.assertIn(driver.SEED, self.registration["excluded_seeds"])
        for seed, namespace in ((True, PROBE_NAMESPACE), (-1, PROBE_NAMESPACE),
                (2**32, PROBE_NAMESPACE), (PROBE_SEED, "not-a-uuid"),
                (PROBE_SEED, PROBE_NAMESPACE.upper()), (PROBE_SEED, driver.NAMESPACE),
                (driver.SEED, PROBE_NAMESPACE)):
            with self.subTest(seed=seed, namespace=namespace), self.assertRaises(ValueError):
                driver.core(seed, namespace)

    def test_core_changes_and_typed_aliases_refuse(self):
        driver.validate_contract(self.registration, self.output)
        for change in (dict(seeds=[2026101018]), dict(mode_order=["disabled", "instrumented"]),
                dict(selection_seed=True), dict(phase_a_admission=True), dict(retry_authorized=True),
                dict(causal_overhead_evidence=True), dict(representative_runtime_evidence=True)):
            r = deepcopy(self.registration)
            r.update(change)
            with self.assertRaisesRegex(ValueError, "core changed"):
                driver.validate_contract(r, self.output)

    def test_exposure_overlap_missing_roster_and_mandatory_bindings_refuse(self):
        with self.assertRaisesRegex(ValueError, "explicit exposure"):
            driver.exposure_inventory([])
        for seeds in ([driver.SEED], [True], []):
            path = self.output / (digest(seeds) + ".json")
            driver.save_new(path, dict(seeds=seeds))
            with self.assertRaises(ValueError):
                driver.exposure_inventory([path])
        exposed = self.output / "fresh-exposed.json"
        driver.save_new(exposed, dict(seeds=[PROBE_SEED]))
        with self.assertRaisesRegex(ValueError, "already exposed"):
            driver.exposure_inventory([exposed], seed=PROBE_SEED)
        with patch.object(driver, "prepare_binding") as prepare:
            for seed in ENGINEERING_EXCLUDED_SEEDS:
                namespace = driver.NAMESPACE if seed == driver.SEED else PROBE_NAMESPACE
                with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "terminal or excluded"):
                    driver.register(output=self.output, exposure_registrations=[self.exposure],
                        seed=seed, namespace=namespace)
            prepare.assert_not_called()
        r = deepcopy(self.registration)
        r["input_sha256"].pop(str(driver.bound_driver_files()[0]))
        with self.assertRaisesRegex(ValueError, "mandatory driver"):
            driver.validate_contract(r, self.output)

    def test_copied_attempt_directory_and_exclusion_drift_refuse(self):
        for change in (dict(attempt_directory="other"), dict(source_root="other"),
                dict(excluded_seed_inventory_sha256="0"*64)):
            r = deepcopy(self.registration)
            r.update(change)
            with self.assertRaises(ValueError):
                driver.validate_contract(r, self.output)

    def test_source_is_full_catalog_priority_sampled_and_capped_not_redrawn(self):
        source = synthetic_source()
        self.assertEqual(driver.validate_source(source), source["roots"][0])
        fresh = synthetic_source(PROBE_SEED, PROBE_NAMESPACE)
        self.assertEqual(driver.validate_source(fresh, seed=PROBE_SEED, namespace=PROBE_NAMESPACE),
            fresh["roots"][0])
        with self.assertRaises(ValueError):
            driver.validate_source(fresh)
        for field, value in (("status", "UNCERTAIN_SOURCE"), ("source_terminal_complete", False),
                ("eligible_catalog_sha256", "0"*64), ("roots", [])):
            changed = deepcopy(source)
            changed[field] = value
            with self.assertRaises(ValueError):
                driver.validate_source(changed)

    def test_selected_root_private_payload_and_priority_drift_refuse(self):
        source = synthetic_source()
        source["roots"][0]["source_request_index"] += 1
        with self.assertRaisesRegex(ValueError, "priority"):
            driver.validate_source(source)
        source = synthetic_source()
        source["roots"][0]["public_record"]["private_snapshot"] = "private"
        with self.assertRaises(ValueError):
            driver.validate_source(source)

    def test_claim_precedes_runtime_and_terminal_without_claim_cannot_reopen(self):
        self.setup_run()
        def measured(r, output, progress):
            self.assertTrue((output / "attempt.json").exists())
        with patch.object(driver, "verify_inputs"), patch.object(driver, "verify_completion"), \
                patch.object(driver, "measure", side_effect=measured):
            self.assertEqual(driver.run(self.output), 0)
        before = (self.output / "terminal.json").read_bytes()
        (self.output / "attempt.json").unlink()
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure") as measured:
            with self.assertRaisesRegex(ValueError, "no retry"):
                driver.run(self.output)
            measured.assert_not_called()
        self.assertEqual(before, (self.output / "terminal.json").read_bytes())

    def test_binding_drift_and_provenance_fail_before_attempt(self):
        self.setup_run()
        with patch.object(driver, "load_binding", return_value="bad"), patch.object(driver, "measure") as measured:
            with self.assertRaisesRegex(ValueError, "registration binding"):
                driver.run(self.output)
            measured.assert_not_called()
        with patch.object(driver, "verify_inputs", side_effect=ValueError("drift")):
            with self.assertRaisesRegex(ValueError, "drift"):
                driver.run(self.output)
        self.assertFalse((self.output / "attempt.json").exists())
        historical = dict(self.registration)
        historical.update(driver.core())
        with patch.object(driver, "load_binding", return_value=digest(historical)), \
                patch.object(driver, "validate_contract"), patch.object(driver, "verify_inputs"), \
                patch.object(driver, "measure") as measured, \
                patch.object(driver.json, "loads", return_value=historical):
            with self.assertRaisesRegex(ValueError, "terminal or excluded"):
                driver.run(self.output)
            measured.assert_not_called()
        self.assertFalse((self.output / "attempt.json").exists())

    def test_refusal_retains_diagnostic_and_unknown_control(self):
        self.setup_run()
        diagnostic = dict(schema="engine-search-fallback-refusal-v1", reason="deadline",
            diagnostic_only_not_policy_input=True, engine_mcts={"total_iterations": 0})
        def failed(r, output, progress):
            progress.update(stage="instrumented:select", current_mode="instrumented")
            progress["modes"]["instrumented"].update(status="ATTEMPTED_UNCERTAIN", diagnostics=dict(calls=7))
            raise EngineSearchFallbackError("fixture", diagnostic=diagnostic)
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure", side_effect=failed):
            self.assertEqual(driver.run(self.output), 1)
        terminal = json.loads((self.output / "terminal.json").read_text())
        self.assertEqual(terminal["engine_fallback_diagnostic"], diagnostic)
        self.assertEqual(terminal["modes"]["instrumented"]["status"], "REFUSED_UNCERTAIN")
        self.assertEqual(terminal["modes"]["disabled"]["status"], "UNSTARTED_UNCERTAIN")
        self.assertFalse(terminal["representative_runtime_evidence"])

    def fixtures(self, *, failed=False, forced=False, cleanup_failure=False, construction_failure=False):
        source, calls, resources = synthetic_source(PROBE_SEED, PROBE_NAMESPACE), [], []
        source_root = source["roots"][0]
        context = SimpleNamespace(observation=SimpleNamespace(legal_action_mask=(True, not forced)))
        registration = self.registration
        class Resource:
            def close(inner):
                resources.append(type(inner).__name__)
                if cleanup_failure:
                    raise RuntimeError("cleanup fixture")
        class Archive(Resource):
            capture_public = capture_private = Mock()
            def selected(inner, root):
                return context, None, object()
        class Adapter(Resource):
            def __init__(inner, cfg, **kwargs):
                if construction_failure:
                    raise RuntimeError("construction fixture")
                inner.sink = kwargs.get("policy_opponent_diagnostics")
                inner.runtime_configuration = driver.expected_runtime(registration, enabled=inner.sink is not None)
                inner.runtime_sha256 = digest(inner.runtime_configuration)
            def select(inner, ctx, **kwargs):
                calls.append(kwargs)
                if inner.sink is not None:
                    for phase in PHASES:
                        with inner.sink.phase(phase):
                            pass
                if failed:
                    raise EngineSearchFallbackError("selection fixture", diagnostic=dict(
                        schema="engine-search-fallback-refusal-v1", reason="deadline",
                        diagnostic_only_not_policy_input=True, engine_mcts=dict(total_iterations=0)))
                return dict(status="SELECTED", root_id=source_root["root_id"],
                    configuration_sha256=driver.CONFIGURATION.identity,
                    runtime_sha256=inner.runtime_sha256, action=0, elapsed_seconds=.1,
                    evidence=dict(total_iterations=1, root_action="action:0"))
        stack = ExitStack()
        self.addCleanup(stack.close)
        for target, value in (
            ("pokezero.local_showdown.LocalShowdownEnv", Resource),
            ("pokezero.collection.env_config_with_policy_spec_masks", Mock()),
            ("pokezero.neural_policy.load_transformer_policy", Mock(return_value=SimpleNamespace(
                result=SimpleNamespace(belief_set_source_hash="source")))),
            ("pokezero.mcts_eval.search_over_raw_source.AuditedRawPolicy", Mock()),
            ("pokezero.mcts_eval.search_over_raw_source.collect_raw_source", Mock(return_value=source)),
            ("pokezero.mcts_eval.search_over_raw_archive.SealedSourceArchive", Archive),
            ("pokezero.mcts_eval.search_over_raw_adapters.PublicModelSearchAdapter", Adapter),
            ("pokezero.mcts_eval.resolver.resolve_checkpoint_contract", Mock(return_value=SimpleNamespace(
                to_manifest=lambda: {})))):
            if value is Resource:
                value = lambda *a, **k: Resource()
            stack.enter_context(patch(target, value))
        stack.enter_context(patch.object(driver, "_runtime"))
        stack.enter_context(patch.object(driver, "verify_inputs"))
        self.validator = stack.enter_context(patch("pokezero.mcts_eval.policy_opponent_profile.validate_selection"))
        self.registration.update(checkpoint="weights", showdown_root="showdown", checkpoint_sha256=SHA,
            set_source_hash="source", model_path="model", encoder_tables="tables", checkpoint_contract={})
        return calls, resources

    def test_actual_measure_lifecycle_saves_both_modes_and_same_prospective_root_seed(self):
        calls, resources = self.fixtures()
        self.setup_run()
        self.assertEqual(driver.run(self.output), 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(resources.count("Adapter"), 2)
        self.assertEqual(resources.count("Archive"), 1)
        self.assertEqual(resources.count("Resource"), 1)
        self.assertEqual(self.validator.call_count, 2)
        kwargs = self.validator.call_args.kwargs
        self.assertEqual((kwargs["deadline_ms"], kwargs["native_batch_guard_ms"], kwargs["model_leaf_override"]),
            (10000, 64, "raw_policy_terminal"))
        self.assertEqual(kwargs["opponent_seed"], self.registration["selection_seed"])
        self.assertEqual(kwargs["config"], SearchConfig(depth=6, sims=4096, batch=16, worlds=4))
        self.assertEqual(sum(kwargs["mask"]), 2)

    def test_measure_refusal_saves_sink_does_not_run_control_and_preserves_primary_on_cleanup_error(self):
        calls, resources = self.fixtures(failed=True, cleanup_failure=True)
        self.setup_run()
        self.assertEqual(driver.run(self.output), 1)
        terminal = json.loads((self.output / "terminal.json").read_text())
        self.assertEqual(len(calls), 1)
        self.assertEqual(terminal["stage"], "instrumented:select")
        self.assertEqual(terminal["modes"]["disabled"]["status"], "UNSTARTED_UNCERTAIN")
        self.assertEqual(len(terminal["cleanup_failures"]), 3)
        sink = json.loads((self.output / "instrumented-diagnostics.json").read_text())
        self.assertEqual(sink["phases"]["callback_total"]["calls"], 1)

    def test_forced_priority_sampled_root_stops_without_easier_replacement(self):
        calls, resources = self.fixtures(forced=True)
        self.setup_run()
        self.assertEqual(driver.run(self.output), 1)
        self.assertEqual(calls, [])
        self.assertFalse((self.output / "instrumented-attempt.json").exists())

    def test_completion_rejects_missing_control_and_forged_disabled_timing(self):
        calls, resources = self.fixtures()
        progress = dict(stage="initial", modes={mode: dict(status="UNSTARTED_UNCERTAIN") for mode in driver.MODES})
        driver.measure(self.registration, self.output, progress)
        driver.verify_completion(self.registration, self.output, progress)
        path = self.output / "disabled-diagnostics.json"
        original = path.read_text()
        path.write_text(json.dumps(dict(enabled=True)))
        with self.assertRaisesRegex(ValueError, "progress/evidence"):
            driver.verify_completion(self.registration, self.output, progress)
        path.write_text(original)
        source_root = json.loads((self.output / "source.json").read_text())["roots"][0]
        with patch.object(driver, "validate_source", return_value=source_root), \
                patch.object(driver, "digest", return_value="bad"):
            with self.assertRaises(ValueError):
                driver.verify_completion(self.registration, self.output, progress)
        (self.output / "disabled-close.json").unlink()
        with self.assertRaises(FileNotFoundError):
            driver.verify_completion(self.registration, self.output, progress)

    def test_native_refusal_survives_secondary_diagnostic_write_failure(self):
        calls, resources = self.fixtures(failed=True)
        self.setup_run()
        saved = driver.save_new
        def failing_save(path, value):
            if Path(path).name == "instrumented-diagnostics.json":
                raise OSError("fixture diagnostic write failed")
            return saved(path, value)
        with patch.object(driver, "save_new", side_effect=failing_save):
            self.assertEqual(driver.run(self.output), 1)
        terminal = json.loads((self.output / "terminal.json").read_text())
        self.assertEqual(terminal["error_type"], "EngineSearchFallbackError")
        self.assertEqual(terminal["engine_fallback_diagnostic"]["reason"], "deadline")
        self.assertEqual(terminal["diagnostic_failures"][0]["error_type"], "OSError")
        self.assertEqual(terminal["modes"]["instrumented"]["diagnostics"]["phases"]["callback_total"]["calls"], 1)
        self.assertEqual(len(calls), 1)

    def test_completion_rejects_coherently_rehashed_runtime_allocation(self):
        calls, resources = self.fixtures()
        progress = dict(stage="initial", modes={mode: dict(status="UNSTARTED_UNCERTAIN") for mode in driver.MODES})
        driver.measure(self.registration, self.output, progress)
        runtime_path, selected_path = self.output / "instrumented-runtime.json", self.output / "instrumented-selected.json"
        runtime = json.loads(runtime_path.read_text())
        selected = json.loads(selected_path.read_text())
        runtime["runtime_configuration"]["incumbent"]["batch"] = 1
        runtime["runtime_sha256"] = selected["runtime_sha256"] = digest(runtime["runtime_configuration"])
        runtime_path.write_text(json.dumps(runtime))
        selected_path.write_text(json.dumps(selected))
        with self.assertRaisesRegex(ValueError, "frozen core"):
            driver.verify_completion(self.registration, self.output, progress)

    def test_diagnostic_schema_counts_and_nonfinite_seconds_refuse(self):
        sink = PolicyOpponentDiagnostics()
        for phase in PHASES:
            with sink.phase(phase):
                pass
        driver.validate_diagnostic(sink.snapshot())
        for key, value in (("elapsed_seconds", float("nan")), ("calls", True),
                ("timed_calls", 0), ("failed_calls", 1)):
            snapshot = sink.snapshot()
            snapshot["phases"]["callback_total"][key] = value
            with self.assertRaises(ValueError):
                driver.validate_diagnostic(snapshot)
        snapshot = sink.snapshot()
        snapshot["phases"].pop("model_evaluation")
        with self.assertRaises(ValueError):
            driver.validate_diagnostic(snapshot)

    def test_source_checkpoint_cap_and_selected_public_identity_refuse(self):
        source = synthetic_source()
        driver.validate_source(source, checkpoint_sha256=SHA)
        for field, value in (("checkpoint_sha256", "b"*64), ("decision_boundaries", 251),
                ("elapsed_seconds", float("inf")), ("verified_raw_decisions", False)):
            changed = deepcopy(source)
            changed[field] = value
            with self.assertRaisesRegex(ValueError, "source checkpoint"):
                driver.validate_source(changed, checkpoint_sha256=SHA)
        changed = deepcopy(source)
        changed["roots"][0]["public_record"]["seed"] = 123
        with self.assertRaises(ValueError):
            driver.validate_source(changed)

    def test_construction_failure_retains_elapsed_and_zero_call_sink(self):
        calls, resources = self.fixtures(construction_failure=True)
        self.setup_run()
        self.assertEqual(driver.run(self.output), 1)
        terminal = json.loads((self.output / "terminal.json").read_text())
        mode = terminal["modes"]["instrumented"]
        self.assertTrue(driver.finite_seconds(mode["construction_seconds"]))
        self.assertNotIn("selection_wall_seconds", mode)
        self.assertEqual(mode["diagnostics"]["phases"]["callback_total"]["calls"], 0)
        self.assertEqual(calls, [])

    def test_completion_rejects_mismatched_two_legal_selected_actions(self):
        calls, resources = self.fixtures()
        progress = dict(stage="initial", modes={mode: dict(status="UNSTARTED_UNCERTAIN") for mode in driver.MODES})
        driver.measure(self.registration, self.output, progress)
        path = self.output / "instrumented-selected.json"
        selected = json.loads(path.read_text())
        selected["action"] = progress["modes"]["instrumented"]["action"] = 1
        path.write_text(json.dumps(selected))
        with self.assertRaisesRegex(ValueError, "selected action"):
            driver.verify_completion(self.registration, self.output, progress)

    def test_diagnostic_mandatory_phases_and_model_output_counts_conserve(self):
        sink = PolicyOpponentDiagnostics()
        for phase in PHASES:
            with sink.phase(phase):
                pass
        for phase in ("payload_binding", "view_reconstruction", "distribution_binding",
                "observation_and_surface", "model_evaluation", "output_certification"):
            snapshot = sink.snapshot()
            snapshot["phases"][phase].update(calls=0, timed_calls=0)
            with self.assertRaisesRegex(ValueError, "conserve"):
                driver.validate_diagnostic(snapshot)


if __name__ == "__main__":
    unittest.main()
