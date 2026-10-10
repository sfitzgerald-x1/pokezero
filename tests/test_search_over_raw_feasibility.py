"""Fresh feasibility contract/lifecycle probes; no scientific workload."""
from dataclasses import asdict
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_search_over_raw_feasibility as driver
from pokezero.engine_search import EngineSearchFallbackError
from pokezero.mcts_eval.search_over_raw import (
    ENGINEERING_EXCLUDED_SEEDS, digest, paired_continuations, phase_a_contract,
)


def registration(output):
    exposure = output / "prior-exposure.json"
    if not exposure.exists():
        driver.save_new(exposure, dict(seeds=[123]))
    excluded, bindings = driver.exposure_inventory([exposure])
    return dict(schema=driver.SCHEMA, namespace=driver.NAMESPACE,
        configurations=[asdict(c) for c in driver.configurations()],
        source_contract=driver.source_contract(), seeds=list(driver.SEEDS),
        fixture_seed=driver.SEEDS[0], candidate_seat="p1", initial_dispatch_workers=6,
        retry_authorized=False, phase_a_admission=False, phase_b_authorized=False,
        scientific_strength_evidence=False, representative_runtime_evidence=False,
        historical_attempt_reused=False, attempt_directory=str(output.resolve()),
        source_root=str(driver.ROOT), exposure_registrations=bindings, excluded_seeds=excluded,
        excluded_seed_inventory_sha256=digest(excluded), input_sha256={**bindings,
            **{str(p): driver.sha256_file(p) for p in driver.bound_driver_files()}})


class RealSourceFeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.registration = registration(self.output)

    def setup_run(self):
        driver.save_new(self.output / "registration.json", self.registration)
        driver.save_new(self.output / "registration-binding.json", dict(registration_sha256=digest(self.registration)))

    def test_all_19_selectors_and_full_1_vs_20_allocations_are_frozen(self):
        configs = driver.configurations()
        self.assertEqual(len(configs), 19)
        self.assertEqual(configs[0].arm, "raw")
        self.assertEqual([(c.seconds, c.leaf, c.arm) for c in configs[1:]], [
            (seconds, leaf, arm) for seconds in (1., 3., 10.)
            for leaf in ("model", "hp_fraction", "raw_rollout")
            for arm in ("incumbent", "reference")])
        self.assertTrue(all(c.belief == "public" and c.workers ==
            (20 if c.arm == "reference" else 1) for c in configs))

    def test_both_reserved_seeds_are_excluded_even_if_never_started(self):
        self.assertTrue(set(driver.SEEDS) <= set(ENGINEERING_EXCLUDED_SEEDS))
        c = phase_a_contract(driver.NAMESPACE, excluded_seeds=[], configurations=driver.configurations())
        self.assertFalse(set(driver.SEEDS) & {seed for p in c["panels"].values() for seed in p["seeds"]})

    def test_fresh_post_optimization_scope_preserves_all_prior_seed_exclusions(self):
        self.assertEqual(driver.SCHEMA, "pokezero.search-over-raw.real-source-feasibility.v3")
        self.assertEqual(driver.NAMESPACE, "8319a3be-b806-43eb-a3dc-ebcffb559093")
        self.assertEqual(driver.SEEDS, (2026101022, 2026101023))
        self.assertTrue(set(range(2026101013, 2026101024)) <= set(ENGINEERING_EXCLUDED_SEEDS))
        self.assertTrue(set(range(2026101013, 2026101022)) <= set(self.registration["excluded_seeds"]))
        self.assertFalse(set(driver.SEEDS) & set(self.registration["excluded_seeds"]))

    def test_historical_schema_namespace_and_seeds_cannot_reopen_under_new_driver(self):
        for change in (
                dict(schema="pokezero.search-over-raw.real-source-feasibility.v1"),
                dict(namespace="7f133933-252b-4db6-becf-b8b5bda6a017"),
                dict(seeds=[2026101015, 2026101016], fixture_seed=2026101015),
                dict(schema="pokezero.search-over-raw.real-source-feasibility.v2"),
                dict(namespace="6ff7b2be-433a-4961-95b7-d0d47e0604ce"),
                dict(seeds=[2026101018, 2026101019], fixture_seed=2026101018),
                dict(seeds=[2026101020, 2026101021], fixture_seed=2026101020)):
            with self.subTest(change=change):
                r = copy.deepcopy(self.registration)
                r.update(change)
                with self.assertRaisesRegex(ValueError, "core changed"):
                    driver.validate_contract(r, self.output)

    def test_four_roots_have_76_selector_and_608_alias_continuation_cells(self):
        cells = driver.planned_cells()
        continuations = driver.planned_continuations()
        self.assertEqual(len(cells), 76)
        self.assertEqual(len(continuations), 608)
        self.assertEqual(len({r["root_id"] for r in cells}), 4)
        self.assertTrue(all(r["contrast_interval"] == [-1., 1.] for r in cells))
        self.assertTrue(all(r["signed_outcome"] is None for r in continuations))

    def test_contract_validates_and_typed_core_drift_refuses(self):
        driver.validate_contract(self.registration, self.output)
        for change in (dict(namespace="other"), dict(seeds=[True, driver.SEEDS[1]]),
                       dict(candidate_seat="p2"), dict(initial_dispatch_workers=20), dict(initial_dispatch_workers=6.0)):
            r = copy.deepcopy(self.registration)
            r.update(change)
            with self.assertRaisesRegex(ValueError, "core changed"):
                driver.validate_contract(r, self.output)
        r = copy.deepcopy(self.registration)
        r["source_contract"]["panels"]["excluded"]["root_slots"][0]["root_slot"] = False
        with self.assertRaisesRegex(ValueError, "core changed"):
            driver.validate_contract(r, self.output)

    def test_oracle_budget_leaf_worker_and_reordered_configs_refuse(self):
        for field, value in (("belief", "oracle"), ("seconds", 10.), ("leaf", "hp_fraction"), ("workers", 2)):
            r = copy.deepcopy(self.registration)
            r["configurations"][1][field] = value
            with self.assertRaisesRegex(ValueError, "core changed"):
                driver.validate_contract(r, self.output)
        r = copy.deepcopy(self.registration)
        r["configurations"][1:3] = r["configurations"][2:0:-1]
        with self.assertRaisesRegex(ValueError, "core changed"):
            driver.validate_contract(r, self.output)

    def test_no_scientific_admission_or_retry_flags(self):
        for field in ("retry_authorized", "phase_a_admission", "phase_b_authorized",
                      "scientific_strength_evidence", "representative_runtime_evidence", "historical_attempt_reused"):
            r = copy.deepcopy(self.registration)
            r[field] = True
            with self.assertRaisesRegex(ValueError, "core changed"):
                driver.validate_contract(r, self.output)

    def test_exposure_overlap_missing_inventory_and_noninteger_seeds_refuse(self):
        with self.assertRaisesRegex(ValueError, "explicit exposure"):
            driver.exposure_inventory([])
        for seeds in ([driver.SEEDS[0]], [True], []):
            path = self.output / ("bad-" + digest(seeds) + ".json")
            driver.save_new(path, dict(seeds=seeds))
            with self.assertRaises(ValueError):
                driver.exposure_inventory([path])

    def test_exposure_binding_drift_refuses_before_runtime(self):
        r = copy.deepcopy(self.registration)
        r["excluded_seeds"] += [999]
        with self.assertRaisesRegex(ValueError, "exposure inventory drift"):
            driver.validate_contract(r, self.output)
        r = copy.deepcopy(self.registration)
        path = next(iter(r["exposure_registrations"]))
        r["input_sha256"][path] = "0" * 64
        with self.assertRaisesRegex(ValueError, "exposure inventory drift"):
            driver.validate_contract(r, self.output)

    def test_driver_checkout_and_copied_attempt_directory_refuse(self):
        for key, value in (("source_root", "other"), ("input_sha256", {}),
                           ("attempt_directory", str(self.output / "other"))):
            r = copy.deepcopy(self.registration)
            r[key] = value
            with self.assertRaises(ValueError):
                driver.validate_contract(r, self.output)

    def test_every_direct_runtime_pin_is_required_before_claim(self):
        pins = driver.bound_driver_files()
        self.assertEqual(len(pins), len(set(pins)))
        self.assertIn(driver.ROOT / "src/pokezero/randbat.py", pins)
        self.assertIn(driver.ROOT / "src/pokezero/belief.py", pins)
        for path in pins:
            with self.subTest(path=path):
                r = copy.deepcopy(self.registration)
                r["input_sha256"].pop(str(path))
                with self.assertRaisesRegex(ValueError, "mandatory driver hash"):
                    driver.validate_contract(r, self.output)
                r["input_sha256"][str(path)] = "0" * 64
                with self.assertRaisesRegex(ValueError, "mandatory driver hash"):
                    driver.validate_contract(r, self.output)

    def test_claim_precedes_runtime_and_success_is_only_engineering(self):
        self.setup_run()
        def measured(r, output, progress):
            self.assertTrue((output / "attempt.json").exists())
            progress["roots_completed"] = 4
            progress["selections_completed"] = 76
            for cell in progress["fixed_roster"]:
                cell.update(status="MEASURED_ENGINEERING_ONLY", contrast_interval=[0., 0.])
        with patch.object(driver, "verify_inputs"), patch.object(driver, "verify_completion"), patch.object(driver, "measure", side_effect=measured):
            self.assertEqual(driver.run(self.output), 0)
        terminal = json.loads((self.output / "terminal.json").read_text())
        self.assertEqual(terminal["status"], "COMPLETE_REAL_SOURCE_ENGINEERING_ONLY")
        self.assertFalse(terminal["phase_a_admission"])
        self.assertFalse(terminal["representative_runtime_evidence"])

    def test_failure_retains_every_unknown_suffix_and_poison_diagnostic(self):
        self.setup_run()
        diagnostic = dict(schema="engine-search-fallback-refusal-v1", reason="deadline",
            diagnostic_only_not_policy_input=True, engine_mcts={"total_iterations": 0})
        def failed(r, output, progress):
            cell = progress["fixed_roster"][1]
            progress["stage"] = cell["root_id"] + ":" + cell["configuration"]
            progress["current_operation"] = dict(kind="selector", root_id=cell["root_id"],
                configuration=cell["configuration"], phase="select")
            progress["adapter_failure"] = dict(status="UNCERTAIN_REFUSED", retry_authorized=False)
            raise EngineSearchFallbackError("fixture only", diagnostic=diagnostic)
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure", side_effect=failed):
            self.assertEqual(driver.run(self.output), 1)
        terminal = json.loads((self.output / "terminal.json").read_text())
        self.assertEqual(terminal["engine_fallback_diagnostic"], diagnostic)
        self.assertEqual(len(terminal["fixed_roster"]), 76)
        self.assertEqual(terminal["fixed_roster"][1]["selection_status"], "REFUSED_UNCERTAIN")
        self.assertTrue(all(r["contrast_interval"] == [-1., 1.] for r in terminal["fixed_roster"]))
        self.assertEqual(len(terminal["continuation_roster"]), 608)

    def test_terminal_without_claim_cannot_reopen(self):
        self.setup_run()
        driver.save_new(self.output / "terminal.json", dict(status="FAILED_NO_RETRY"))
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure") as measured:
            with self.assertRaises(FileExistsError):
                driver.run(self.output)
            measured.assert_not_called()
        self.assertFalse((self.output / "attempt.json").exists())

    def test_claimed_attempt_cannot_run_a_second_time(self):
        self.setup_run()
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure", side_effect=ValueError("fixture")) as measured:
            self.assertEqual(driver.run(self.output), 1)
            old = (self.output / "terminal.json").read_bytes()
            with self.assertRaises(FileExistsError):
                driver.run(self.output)
            measured.assert_called_once()
        self.assertEqual(old, (self.output / "terminal.json").read_bytes())

    def test_provenance_drift_blocks_before_claim(self):
        self.setup_run()
        with patch.object(driver, "verify_inputs", side_effect=ValueError("input drift")), patch.object(driver, "measure") as measured:
            with self.assertRaisesRegex(ValueError, "input drift"):
                driver.run(self.output)
            measured.assert_not_called()
        self.assertFalse((self.output / "attempt.json").exists())

    def test_shared_actions_have_one_outcome_and_no_invented_replicates(self):
        progress = dict(continuation_roster=driver.planned_continuations(), continuations_completed=0)
        root_id = progress["continuation_roster"][0]["root_id"]
        other = driver.configurations()[1].identity
        actions = {"raw": 0, other: 0}
        row = dict(root_id=root_id, action=0, replicate=0, status="COMPLETE", signed_outcome=1)
        driver.record_outcome(self.output, progress, actions, row, .1)
        done = [r for r in progress["continuation_roster"] if r["status"] == "COMPLETE"]
        self.assertEqual({r["configuration"] for r in done}, {"raw", other})
        self.assertEqual(len(done), 2)
        self.assertEqual(progress["continuations_completed"], 1)
        self.assertEqual(len(list(self.output.glob("outcome-*.json"))), 1)

    def test_capped_continuation_stops_at_first_row_with_durable_attempt(self):
        progress = dict(continuation_roster=driver.planned_continuations(), continuations_completed=0)
        root_id = progress["continuation_roster"][0]["root_id"]
        actions = {"raw": 0, driver.configurations()[1].identity: 1}
        class CappedEnv:
            restores = 0
            def restore(self, snapshot): self.restores += 1
            def terminal(self):
                # paired helper requires a nonterminal source before stepping;
                # hit the explicit cap by returning no terminal/request success.
                return None
            def requested_players(self): return ("p1",)
            def observe(self, seat): return object()
            def reseed_simulator_rng(self, seed): pass
            def step(self, choices): pass
        env = CappedEnv()
        with self.assertRaisesRegex(ValueError, "capped continuation"):
            paired_continuations(env=env, snapshot=object(), subject="p1", actions=actions,
                evaluator=lambda observation: ([0, 1], [.5, .5]), namespace=driver.NAMESPACE,
                root_id=root_id, max_boundaries=1,
                attempt_sink=lambda row: driver.record_continuation_attempt(self.output, progress, actions, row),
                outcome_sink=lambda row: driver.record_outcome(self.output, progress, actions, row, .1))
        self.assertEqual(env.restores, 1)
        self.assertEqual(len(list(self.output.glob("outcome-*.json"))), 1)
        self.assertTrue((self.output / "continuation-0-0-attempt.json").exists())
        self.assertEqual(progress["continuation_roster"][0]["status"], "CAPPED")

    def test_exception_before_outcome_retains_refused_alias_and_unstarted_suffix(self):
        self.setup_run()
        def failed(r, output, progress):
            row = progress["continuation_roster"][0]
            progress["stage"] = row["root_id"] + ":continuations"
            driver.record_continuation_attempt(output, progress, {"raw": 0},
                dict(root_id=row["root_id"], action=0, replicate=0))
            raise ValueError("fixture restore failed")
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure", side_effect=failed):
            self.assertEqual(driver.run(self.output), 1)
        terminal = json.loads((self.output / "terminal.json").read_text())
        self.assertEqual(terminal["continuation_roster"][0]["status"], "REFUSED_UNCERTAIN")
        self.assertEqual(terminal["continuation_roster"][1]["status"], "UNSTARTED_UNCERTAIN")

    def test_forced_roots_are_preserved_not_claimed_as_search_work(self):
        cfg = driver.configurations()[1]
        work = driver.selection_work(cfg, dict(evidence=dict(total_iterations=0)), 1)
        self.assertTrue(work["forced_action"])
        self.assertFalse(work["substantive_search_exercised"])
        work = driver.selection_work(cfg, dict(evidence=dict(total_iterations=2)), 9)
        self.assertEqual(work["completed_search_units"], 2)
        self.assertTrue(work["substantive_search_exercised"])
        self.assertFalse(work["leaf_exercise_qualified"])

    def test_registration_receipt_prevents_input_map_removal_before_claim(self):
        self.setup_run()
        original = json.loads((self.output / "registration.json").read_text())
        original["input_sha256"].pop(str(Path(driver.__file__).resolve()))
        # Deliberate malformed-input fixture, not historical evidence.
        (self.output / "registration.json").write_text(json.dumps(original))
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure") as measured:
            with self.assertRaisesRegex(ValueError, "input-map or contract drift"):
                driver.run(self.output)
            measured.assert_not_called()
        self.assertFalse((self.output / "attempt.json").exists())

    def test_optimistic_mutable_counters_cannot_establish_complete_harness(self):
        self.setup_run()
        def unverified(r, output, progress):
            progress["roots_completed"] = 4
            for cell in progress["fixed_roster"]:
                cell["status"] = "MEASURED_ENGINEERING_ONLY"
        with patch.object(driver, "verify_inputs"), patch.object(driver, "measure", side_effect=unverified):
            self.assertEqual(driver.run(self.output), 1)
        terminal = json.loads((self.output / "terminal.json").read_text())
        self.assertEqual(terminal["error_type"], "FileNotFoundError")
        self.assertEqual(terminal["source_games_completed"], 0)

    def complete_durable_fixture(self):
        """Synthetic evidence only: exercise reconciliation, not native work."""
        progress = dict(source_games_completed=2, roots_completed=4,
            selections_completed=76, continuations_completed=32,
            source_roster={str(seed): dict(status="COMPLETE") for seed in driver.SEEDS},
            fixed_roster=driver.planned_cells(), continuation_roster=driver.planned_continuations())
        contract = driver.source_contract()
        public = dict(recorded_action_index=0)
        for seed in driver.SEEDS:
            source_dir = self.output / f"source-{seed}"
            source_dir.mkdir()
            catalog = [dict(source_request_index=i, public_record_sha256=digest(public)) for i in (1, 2, 3)]
            indices = driver.select_source_requests(driver.NAMESPACE, seed, [1, 2, 3], 2)
            roots = [dict(root_id=f"excluded:{seed}:{slot}", source_request_index=index,
                public_record=public, public_record_sha256=digest(public)) for slot, index in enumerate(indices)]
            driver.save_new(source_dir / "source.json", dict(contract_sha256=digest(contract),
                source_seed=seed, panel="excluded", status="COMPLETE", source_terminal_complete=True,
                source_policy="raw_argmax_both_seats", eligible_requests=3, eligible_public_records=catalog,
                eligible_catalog_sha256=digest(catalog), roots=roots, requested_root_slots=2, missing_root_ids=[]))
            for root in roots:
                root_dir = source_dir / f"root-{root['root_id'].rsplit(':', 1)[1]}"
                root_dir.mkdir()
                actions = {"raw" if c.arm == "raw" else c.identity: 0 for c in driver.configurations()}
                outcomes = [dict(action=0, replicate=i, status="COMPLETE", signed_outcome=1, boundaries=1)
                    for i in range(8)]
                driver.save_new(root_dir / "audit.json", dict(root_id=root["root_id"], status="COMPLETE",
                    actions=actions, outcomes=outcomes, contrasts={key: [0., 0.] for key in actions}))
                for cfg in driver.configurations():
                    key = "raw" if cfg.arm == "raw" else cfg.identity
                    runtime = dict(configuration=asdict(cfg))
                    driver.save_new(root_dir / f"{key}-runtime.json", dict(runtime_configuration=runtime))
                    driver.save_new(root_dir / f"{key}-selected.json", dict(status="SELECTED",
                        root_id=root["root_id"] + ":" + key, configuration_sha256=cfg.identity,
                        runtime_sha256=digest(runtime), action=0))
                    driver.save_new(root_dir / f"{key}-close.json", dict(seconds=.1))
                for row in outcomes:
                    driver.save_new(root_dir / f"outcome-0-{row['replicate']}.json", dict(root_id=root["root_id"], **row))
                    driver.save_new(root_dir / f"continuation-0-{row['replicate']}-attempt.json",
                        dict(root_id=root["root_id"], action=0, replicate=row["replicate"]))
        for cell in progress["fixed_roster"]:
            cell.update(status="MEASURED_ENGINEERING_ONLY", selection_status="SELECTED", action=0,
                contrast_interval=[0., 0.])
        for cell in progress["continuation_roster"]:
            cell.update(status="COMPLETE", action=0, signed_outcome=1)
        return progress

    def test_complete_durable_sources_selectors_and_shared_outcomes_reconcile(self):
        progress = self.complete_durable_fixture()
        driver.verify_completion(self.output, progress)
        self.assertEqual(progress["continuations_completed"], 32)
        self.assertEqual(len(progress["continuation_roster"]), 608)

    def test_durable_completion_rejects_source_runtime_contrast_and_outcome_tampering(self):
        progress = self.complete_durable_fixture()
        source_dir = self.output / f"source-{driver.SEEDS[0]}"
        root_dir = source_dir / "root-0"
        cases = [
            (source_dir / "source.json", lambda value: value.update(eligible_catalog_sha256="0" * 64)),
            (root_dir / "raw-selected.json", lambda value: value.update(runtime_sha256="0" * 64)),
            (root_dir / "audit.json", lambda value: value["contrasts"].update(raw=[1., 1.])),
            (root_dir / "audit.json", lambda value: value["outcomes"].append(dict(value["outcomes"][0]))),
            (root_dir / "outcome-0-0.json", lambda value: value.update(signed_outcome=-1)),
            (root_dir / "continuation-0-0-attempt.json", lambda value: value.update(replicate=1)),
        ]
        for path, change in cases:
            with self.subTest(path=path.name):
                original = path.read_text()
                value = json.loads(original)
                change(value)
                # Only disposable synthetic test artifacts are deliberately changed.
                path.write_text(json.dumps(value))
                try:
                    with self.assertRaises(ValueError):
                        driver.verify_completion(self.output, progress)
                finally:
                    path.write_text(original)
        driver.verify_completion(self.output, progress)

    def test_cleanup_preserves_primary_failure_and_attempts_all_owned_resources(self):
        calls = []
        class Resource:
            def __init__(self, name): self.name = name
            def close(self):
                calls.append(self.name)
                raise RuntimeError("fixture cleanup")
        progress = dict(stage="fixture")
        try:
            raise ValueError("first fixture failure")
        except ValueError:
            driver.close_resources((("one", Resource("one")), ("two", Resource("two"))), progress)
        self.assertEqual(calls, ["one", "two"])
        self.assertEqual(len(progress["cleanup_failures"]), 2)
        with self.assertRaises(RuntimeError):
            driver.close_resources((("one", Resource("one")),), progress)
        self.assertEqual(progress["current_operation"]["kind"], "cleanup")

    def test_actual_measure_lifecycle_source_raw_selection_and_cleanup_failures(self):
        # Exercise the real orchestration (not a replacement measure callback),
        # with owned disposable mock environments and zero model/search work.
        for mode in ("source", "raw", "close", "raw_and_close"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
                output = Path(directory)
                r = registration(output)
                r.update(checkpoint="fixture.pt", showdown_root="/fixture-showdown", set_source_hash="source",
                    checkpoint_sha256="a" * 64, model_path="fixture-model", encoder_tables="fixture-tables",
                    checkpoint_contract={"fixture": "contract"}, factory_options={})
                driver.save_new(output / "registration.json", r)
                driver.save_new(output / "registration-binding.json", dict(registration_sha256=digest(r)))
                public = dict(recorded_action_index=0)
                contract = driver.source_contract()
                seed = driver.SEEDS[0]
                rows = [dict(source_request_index=i, public_record_sha256=digest(public)) for i in (1, 2, 3)]
                selected = driver.select_source_requests(driver.NAMESPACE, seed, [1, 2, 3], 2)
                source = dict(contract_sha256=digest(contract), source_seed=seed, panel="excluded",
                    status="COMPLETE", source_terminal_complete=True, source_policy="raw_argmax_both_seats",
                    eligible_requests=3, eligible_public_records=rows, eligible_catalog_sha256=digest(rows),
                    requested_root_slots=2, missing_root_ids=[], elapsed_seconds=.1, decision_boundaries=4,
                    roots=[dict(root_id=f"excluded:{seed}:{slot}", source_request_index=index,
                        public_record=public, public_record_sha256=digest(public)) for slot, index in enumerate(selected)])
                context = SimpleNamespace(observation=SimpleNamespace(legal_action_mask=(True, True)))
                counts = dict(source=0, select=0, env_close=0, archive_close=0)
                class Adapter:
                    def __init__(self, cfg, **kwargs):
                        self.cfg = cfg
                        self.runtime_configuration = dict(configuration=asdict(cfg))
                        self.last_failure = dict(status="UNCERTAIN_REFUSED")
                    def select(self, context, **kwargs):
                        counts["select"] += 1
                        if mode in ("raw", "raw_and_close"):
                            raise ValueError("first raw fixture failure")
                        return dict(root_id=kwargs["root_id"], configuration_sha256=self.cfg.identity,
                            runtime_sha256=digest(self.runtime_configuration), status="SELECTED", action=0,
                            elapsed_seconds=.1, exceeded_nominal_seconds=False, evidence={})
                    def close(self):
                        if mode in ("close", "raw_and_close"):
                            raise RuntimeError("secondary fixture cleanup")
                def collect(*args, **kwargs):
                    counts["source"] += 1
                    if mode == "source": raise ValueError("source fixture failure")
                    return source
                def env_close(): counts["env_close"] += 1
                def archive_close(): counts["archive_close"] += 1
                policy = SimpleNamespace(result=SimpleNamespace(belief_set_source_hash="source"))
                patches.enter_context(patch.object(driver, "_runtime"))
                patches.enter_context(patch.object(driver, "verify_inputs"))
                patches.enter_context(patch("pokezero.collection.env_config_with_policy_spec_masks", return_value=None))
                patches.enter_context(patch("pokezero.local_showdown.LocalShowdownEnv", return_value=SimpleNamespace(close=env_close)))
                patches.enter_context(patch("pokezero.neural_policy.load_transformer_policy", return_value=policy))
                patches.enter_context(patch("pokezero.mcts_eval.paper_reference_showdown.ChampionEvaluator"))
                patches.enter_context(patch("pokezero.mcts_eval.paper_reference_runtime.ShowdownWorkerFactory"))
                patches.enter_context(patch("pokezero.mcts_eval.resolver.resolve_checkpoint_contract",
                    return_value=SimpleNamespace(to_manifest=lambda: r["checkpoint_contract"])))
                patches.enter_context(patch("pokezero.mcts_eval.search_over_raw_source.AuditedRawPolicy",
                    side_effect=lambda policy, **kwargs: SimpleNamespace(policy=policy)))
                patches.enter_context(patch("pokezero.mcts_eval.search_over_raw_source.collect_raw_source", side_effect=collect))
                patches.enter_context(patch("pokezero.mcts_eval.search_over_raw_archive.SealedSourceArchive",
                    return_value=SimpleNamespace(capture_public=lambda: None, capture_private=lambda: None,
                        selected=lambda root: (context, None, object()), close=archive_close)))
                patches.enter_context(patch("pokezero.mcts_eval.search_over_raw_adapters.PublicModelSearchAdapter", Adapter))
                self.assertEqual(driver.run(output), 1)
                terminal = json.loads((output / "terminal.json").read_text())
                self.assertEqual(counts["source"], 1)
                self.assertEqual(counts["env_close"], 1)
                self.assertEqual(counts["archive_close"], 1)
                raw = terminal["fixed_roster"][0]
                if mode == "source":
                    self.assertEqual(terminal["source_roster"][str(seed)]["status"], "REFUSED_UNCERTAIN")
                    self.assertEqual(counts["select"], 0)
                elif mode == "close":
                    self.assertEqual(raw["selection_status"], "SELECTED")
                    self.assertEqual(raw["cleanup_status"], "FAILED")
                    self.assertEqual(terminal["selections_completed"], 1)
                else:
                    self.assertEqual(raw["selection_status"], "REFUSED_UNCERTAIN")
                    self.assertEqual(terminal["error_type"], "ValueError")
                    if mode == "raw_and_close": self.assertTrue(terminal["cleanup_failures"])
                self.assertEqual(terminal["fixed_roster"][1]["selection_status"], "UNSTARTED")


if __name__ == "__main__":
    unittest.main()
