"""Authored deadline reports and actual Python input hooks; no source/model runs."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_native_atomic_work import (
    state_hash, validate_timed_native_report,
)
from pokezero.mcts_eval.search_over_raw_world_inputs import (
    NativeInputHandle, NativeWorldInputs, WorldInputSearchAdapter, WorldInputFixedWorkAdapter,
    validate_timed_native_inputs,
)
from test_search_over_raw_native_atomic_work import native_report
from test_search_over_raw_world_inputs import BINDING, construct, arguments, native_manifest, FakeOverride, world


def timed_report(completed=16, requested=64, budget=8, elapsed=8.5, exhausted=True, cancelled=False):
    report = native_report(completed)
    report["atomic_work"]["requested_iterations"] = requested
    report.update(requested_iterations=requested, remaining_iterations=requested-completed,
        time_budget_enabled=True, time_budget_ms=budget, time_budget_elapsed_ms=elapsed,
        time_budget_batch_overshoot_ms=max(0.,elapsed-budget), time_budget_exhausted=exhausted)
    if cancelled:
        work = report["atomic_work"]
        before = deepcopy(work["final"])
        work["rounds"].append(dict(ordinal=len(work["rounds"]), status="CANCELLED",
            before_collection=deepcopy(before), after_collection=deepcopy(before),
            after=deepcopy(before), accepted=0, attempted_paths=[]))
    return report


def timed_arguments(text="complete-state", requested=64, budget=8, leaf=None):
    args = arguments(text=text, iterations=requested)
    return args + [0, True, False, None, False, leaf, 1, 250, "uniform", 0, 1, False, budget]


def input_evidence(completed=16, *, requested=64, unique=False, cancelled=False,
                   normal_report=True, dispatched=True):
    recorder = NativeWorldInputs(BINDING,16, record_atomic_work=True, record_timed_atomic_work=True)
    for ordinal in range(4):
        construct(recorder, text=f"state-{ordinal}" if unique else "complete-state")
    report = timed_report(completed, requested=requested, cancelled=cancelled)
    if unique: report["atomic_work"]["input_state_utf8_sha256"] = state_hash("state-0")
    base = SimpleNamespace(search_batched_multi_encoded=Mock(return_value=json.dumps(report)))
    if dispatched:
        NativeInputHandle(base,recorder).search_batched_multi_encoded(*timed_arguments(
            text="state-0" if unique else "complete-state", requested=requested))
    manifest = native_manifest(recorder, completed if dispatched else 0) if normal_report else None
    calls = manifest["calls"] if manifest else recorder.calls
    invocations = []
    for call in calls:
        row = call["atomic_work_validation"]
        invocations.append(dict(status="completed",world_seed=7,multiplicity=call["multiplicity"],
            requested_iterations=call["requested_iterations"], completed_iterations=row["completed_iterations"],
            remaining_iterations=row["remaining_iterations"], time_budget_ms=row["native_budget_ms"],
            time_budget_elapsed_ms=row["native_elapsed_ms"],
            time_budget_batch_overshoot_ms=row["native_overshoot_ms"], time_budget_exhausted=row["exhausted"],
            root_visits=dict(side_one=row["completed_iterations"],side_two=row["completed_iterations"])))
    return dict(total_iterations=completed if dispatched else 0, consumed_world_inputs=manifest,
        engine_mcts=dict(time_budget=dict(scope="whole_model_decision",requested_ms=1000,
            native_batch_guard_ms=64,deadline_elapsed_ms=1001.,deadline_overshoot_ms=1.,exhausted=True,
            worlds_budget_skipped=4-sum(call["multiplicity"] for call in calls if call["completed_iterations"] > 0),
            native_invocations=invocations))), recorder, base


def read_inputs(evidence, **overrides):
    return validate_timed_native_inputs(evidence, **dict(dict(binding=BINDING,
        iterations_per_world=16,batch_size=16,expected_worlds=4,nominal_seconds=1.,
        adapter_elapsed_seconds=1.2),**overrides))


class TimedNativeReportTests(unittest.TestCase):
    def read(self, report, **overrides):
        return validate_timed_native_report(report, **dict(dict(state_sha256=state_hash("complete-state"),
            requested_iterations=64,batch_size=16,chance_seed=7,time_budget_ms=8),**overrides))

    def test_prefix_and_cancelled_tail_not_fixed_allocation(self):
        result = self.read(timed_report(cancelled=True))
        self.assertEqual((result["completed_iterations"], result["remaining_iterations"],
            result["committed_rounds"],result["cancelled_rounds"]), (16,48,1,1))
        self.assertEqual(result["native_overshoot_ms"], .5)
        self.assertFalse(result["runtime_qualified"])
        self.assertFalse(result["owning_process_validated"])

    def test_zero_deadline_without_batches_and_cancelled_tail(self):
        for cancelled in (False,True):
            result = self.read(timed_report(0,cancelled=cancelled))
            self.assertEqual(result["completed_iterations"],0)
            self.assertEqual(result["status"],"ZERO_COMMIT_DEADLINE_RETURN")

    def test_full_work_final_batch_can_overrun_without_exhaustion_flag(self):
        result = self.read(timed_report(64,exhausted=False))
        self.assertEqual(result["completed_iterations"],64)
        self.assertEqual(result["native_overshoot_ms"],.5)
        self.assertFalse(result["on_time_deployability_established"])

    def test_no_partial_commit_or_false_deadline_completion(self):
        cases = [timed_report(3), timed_report(0,exhausted=False), timed_report(elapsed=7.9)]
        for report in cases:
            with self.subTest(report=report["iterations"]),self.assertRaises(ValueError): self.read(report)

    def test_row_saturated_commit_does_not_require_traversal_saturation(self):
        # One complete traversal expands two fully priced chance rows. Native
        # stops collection at the row cap, not at a deadline inside the round.
        report = timed_report(completed=1)
        report["atomic_work"]["batch_size"] = 2
        result = self.read(report, batch_size=2)
        self.assertEqual(result["completed_iterations"], 1)
        self.assertEqual(result["committed_rounds"], 1)
        with self.assertRaisesRegex(ValueError, "whole batch"):
            self.read(timed_report(completed=1))
        forged = deepcopy(report)
        forged["atomic_work"]["rounds"][0]["row_value_bits"].append(0)
        with self.assertRaises(ValueError): self.read(forged, batch_size=2)

    def test_foreign_args_and_malformed_duration_count_or_scope_refuse(self):
        for key, value in dict(time_budget_ms=True, time_budget_enabled=False,
            remaining_iterations=True, requested_iterations=16, time_budget_elapsed_ms=float("nan"),
            time_budget_batch_overshoot_ms=0., time_budget_exhausted=1).items():
            report=timed_report(); report[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): self.read(report)
        with self.assertRaises(ValueError): self.read(timed_report(),time_budget_ms=9)


class TimedNativeInputTests(unittest.TestCase):
    def test_actual_python_abi_builder_places_model_and_hp_budget_at_bound_slot(self):
        from pokezero.engine_search import EngineMctsConfig, native_search_args
        for leaf in (None, "hp_fraction"):
            cfg=EngineMctsConfig(leaf_eval="model",model_priors=True,strict_fallbacks=True,
                search_sims=16,search_batch=16,model_leaf_override=leaf,
                model_path="authored-never-loaded-model",checkpoint_path="authored-never-loaded-checkpoint",
                tables_path="authored-never-loaded-tables")
            source=arguments()
            args=native_search_args(cfg,dict(state_str=source[0],ctx_json=source[5],seed=7,side_key="side_one"),
                tables_json=source[3],root_inputs=source[4],rust_fold=source[6],early_stop_min_sims=0,time_budget_ms=8)
            self.assertEqual((len(args),args[12],args[17],args[24]),(25,0,leaf,8))
            recorder=NativeWorldInputs(BINDING,16,record_atomic_work=True,record_timed_atomic_work=True)
            construct(recorder)
            handle=NativeInputHandle(SimpleNamespace(search_batched_multi_encoded=Mock(
                return_value=json.dumps(timed_report(requested=16)))),recorder)
            handle.search_batched_multi_encoded(*args)
            manifest=native_manifest(recorder)
            self.assertEqual(manifest["calls"][0]["native_call_parameters"]["time_budget_ms"],8)

    def test_actual_hooks_and_raw_return_bind_collapsed_population(self):
        evidence,recorder,base=input_evidence(cancelled=True)
        result=read_inputs(evidence)
        self.assertEqual(base.search_batched_multi_encoded.call_count,1)
        actual_args,actual_kwargs=base.search_batched_multi_encoded.call_args
        expected_args=timed_arguments()
        self.assertEqual(actual_args[:6],tuple(expected_args[:6]))
        self.assertEqual(actual_args[6].to_payload(),expected_args[6].to_payload())
        self.assertEqual(actual_args[7:],tuple(expected_args[7:]))
        self.assertEqual(actual_kwargs,dict(record_atomic_work=True))
        self.assertEqual((result["attempted_worlds"],result["dispatched_worlds"],result["searched_worlds"]),(4,4,4))
        self.assertEqual(result["completed_iterations"],16)
        self.assertAlmostEqual(result["adapter_overshoot_seconds"],.2)
        self.assertEqual(recorder.calls[0]["native_call_parameters"]["time_budget_ms"],8)
        for flag in ("scoring_authorized","runtime_qualified","owning_process_validated",
                "scientific_collection_authorized","on_time_deployability_established"):
            self.assertFalse(result[flag])

    def test_all_skipped_and_zero_return_visible_not_productive(self):
        for dispatched in (True,False):
            evidence,_,_=input_evidence(0,dispatched=dispatched)
            result=read_inputs(evidence)
            self.assertEqual(result["status"],"UNQUALIFIED_AT_CEILING")
            self.assertEqual(result["searched_worlds"],0)
            self.assertEqual(result["constructed_not_dispatched_worlds"],0 if dispatched else 4)

    def test_unconsumed_constructed_worlds_kept_in_population(self):
        evidence,_,_=input_evidence(unique=True,requested=16)
        result=read_inputs(evidence)
        self.assertEqual(result["constructed_not_dispatched_worlds"],3)
        self.assertEqual(result["dispatched_worlds"],1)

    def test_guarded_native_prefix_can_finish_before_nominal_ceiling(self):
        evidence,_,_=input_evidence()
        budget=evidence["engine_mcts"]["time_budget"]
        budget.update(deadline_elapsed_ms=940.,deadline_overshoot_ms=0.)
        result=read_inputs(evidence,adapter_elapsed_seconds=.95)
        self.assertEqual(result["adapter_overshoot_seconds"],0.)
        self.assertEqual(result["status"],"OBSERVED_WORK_NOT_QUALIFIED")

    def test_mixed_positive_and_zero_return_keeps_dispatched_zero_in_skips(self):
        evidence,_,_=input_evidence(unique=True,requested=16)
        recorder=NativeWorldInputs(BINDING,16,record_atomic_work=True,record_timed_atomic_work=True)
        for ordinal in range(4):construct(recorder,text=f"state-{ordinal}")
        for ordinal, completed in enumerate((16,0)):
            report=timed_report(completed,requested=16)
            report["atomic_work"]["input_state_utf8_sha256"]=state_hash(f"state-{ordinal}")
            handle=NativeInputHandle(SimpleNamespace(search_batched_multi_encoded=Mock(return_value=json.dumps(report))),recorder)
            handle.search_batched_multi_encoded(*timed_arguments(text=f"state-{ordinal}",requested=16))
        evidence["consumed_world_inputs"]=native_manifest(recorder,16)
        second=deepcopy(evidence["engine_mcts"]["time_budget"]["native_invocations"][0])
        second.update(completed_iterations=0,remaining_iterations=16,root_visits=dict(side_one=0,side_two=0))
        evidence["engine_mcts"]["time_budget"]["native_invocations"].append(second)
        result=read_inputs(evidence)
        self.assertEqual((result["dispatched_worlds"],result["searched_worlds"],
            result["constructed_not_dispatched_worlds"]),(2,1,2))
        self.assertEqual(evidence["engine_mcts"]["time_budget"]["worlds_budget_skipped"],3)
        evidence["engine_mcts"]["time_budget"]["worlds_budget_skipped"]=2
        with self.assertRaises(ValueError):read_inputs(evidence)

    def test_dispatch_abi_drift_delegates_but_invalidates_observer(self):
        for mutation in ("missing", "extra", "boolbudget", "earlystop", "rollout"):
            recorder=NativeWorldInputs(BINDING,16,record_atomic_work=True,record_timed_atomic_work=True)
            construct(recorder)
            args=timed_arguments(requested=16)
            if mutation=="missing":args.pop()
            if mutation=="extra":args.append(None)
            if mutation=="boolbudget":args[24]=True
            if mutation=="earlystop":args[12]=1
            if mutation=="rollout":args[17]="raw_policy_terminal"
            base=SimpleNamespace(search_batched_multi_encoded=Mock(return_value=json.dumps(timed_report(requested=16))))
            NativeInputHandle(base,recorder).search_batched_multi_encoded(*args)
            base.search_batched_multi_encoded.assert_called_once_with(*args,record_atomic_work=True)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):native_manifest(recorder)

    def test_exception_and_unvalidated_return_never_classified_zero(self):
        for failure in (RuntimeError("authored failure"),'{"iterations":0}'):
            recorder=NativeWorldInputs(BINDING,16,record_atomic_work=True,record_timed_atomic_work=True)
            construct(recorder)
            base=SimpleNamespace(search_batched_multi_encoded=Mock(
                side_effect=failure if isinstance(failure,Exception) else None,
                return_value=failure if isinstance(failure,str) else None))
            if isinstance(failure,Exception):
                with self.assertRaises(RuntimeError):NativeInputHandle(base,recorder).search_batched_multi_encoded(*timed_arguments(requested=16))
            else: NativeInputHandle(base,recorder).search_batched_multi_encoded(*timed_arguments(requested=16))
            with self.assertRaises(ValueError):native_manifest(recorder,0)

    def test_full_roster_and_clock_mutations_rejected(self):
        for mutation in ("drop", "duplicate", "notconsumed", "alias", "error", "count", "report",
                "params", "invocation", "requested", "skipped", "overshoot", "elapsed", "exhausted", "refused",
                "differentconstruction", "differentstate"):
            evidence,_,_=input_evidence()
            manifest=evidence["consumed_world_inputs"]; budget=evidence["engine_mcts"]["time_budget"]
            if mutation=="drop":manifest["rows"].pop()
            if mutation=="duplicate":manifest["calls"].append(deepcopy(manifest["calls"][0]))
            if mutation=="notconsumed":manifest["rows"][0]["consumed"]=False
            if mutation=="alias":manifest["calls"][0]["construction_ordinals"][0]=False
            if mutation=="error":manifest["timed_atomic_observation"]["hook_errors"]=["ValueError"]
            if mutation=="count":manifest["timed_atomic_observation"]["searched"]=False
            if mutation=="report":manifest["calls"][0]["native_report_utf8"]="{}"
            if mutation=="params":manifest["calls"][0]["native_call_parameters"]["time_budget_ms"]=9
            if mutation=="invocation":budget["native_invocations"][0]["completed_iterations"]=0
            if mutation=="requested":budget["requested_ms"]=500
            if mutation=="skipped":budget["worlds_budget_skipped"]=1
            if mutation=="overshoot":budget["deadline_overshoot_ms"]=0
            if mutation=="elapsed":budget["deadline_elapsed_ms"]=1300;budget["deadline_overshoot_ms"]=300
            if mutation=="exhausted":budget["exhausted"]=False
            if mutation=="refused":manifest["rows"][0]["status"]="REFUSED"
            if mutation=="differentconstruction":manifest["rows"][0]["constructed_sha256"]="c"*64
            if mutation=="differentstate":manifest["rows"][0]["state_sha256"]="d"*64
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,KeyError,IndexError)):read_inputs(evidence)

    def test_no_attempts_is_only_a_normal_exhausted_readout_not_owner_proof(self):
        recorder=NativeWorldInputs(BINDING,16,record_atomic_work=True,record_timed_atomic_work=True)
        evidence,_,_=input_evidence(0,dispatched=False)
        evidence["consumed_world_inputs"]=recorder.finish(dict(total_iterations=0),attempted=0,constructed=0,searched=0)
        evidence["engine_mcts"]["time_budget"]["worlds_budget_skipped"]=0
        result=read_inputs(evidence)
        self.assertEqual(result["not_attempted_worlds"],4)
        self.assertFalse(result["owning_process_validated"])
        evidence["engine_mcts"]["time_budget"]["exhausted"]=False
        with self.assertRaises(ValueError):read_inputs(evidence)

    def test_opt_in_not_legacy_or_reference_or_rollout_or_fixed_work(self):
        def parent_init(adapter, configuration, **kwargs):
            adapter.configuration=configuration;adapter.runtime_configuration={}
        from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
        with patch.object(PublicModelSearchAdapter,"__init__",parent_init):
            for leaf in ("model","hp_fraction"):
                cfg=SearchConfiguration("incumbent",leaf=leaf)
                adapter=WorldInputSearchAdapter(cfg,timed_native_atomic_work=True)
                self.assertIn("timed_native_atomic_work",adapter.runtime_configuration)
                self.assertTrue(adapter._timed_native_atomic_work)
            legacy=WorldInputSearchAdapter(SearchConfiguration("incumbent"))
            self.assertNotIn("native_atomic_work",legacy.runtime_configuration)
            self.assertFalse(legacy._timed_native_atomic_work)
            for adapter_type,cfg in ((WorldInputSearchAdapter,SearchConfiguration("reference",workers=20)),
                    (WorldInputSearchAdapter,SearchConfiguration("incumbent",leaf="raw_rollout")),
                    (WorldInputFixedWorkAdapter,SearchConfiguration("incumbent"))):
                with self.assertRaises(ValueError):adapter_type(cfg,timed_native_atomic_work=True)


class StrictZeroWorkSelectorTests(unittest.TestCase):
    """Genuine adapter select/cleanup and strict type; authored policy/native return."""
    def setUp(self):
        from test_search_over_raw_archive import ExplorationRootCrosswalkTests
        fixture=ExplorationRootCrosswalkTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.context=fixture.context

    def adapter(self, mutation=None, timed=True):
        from pokezero.engine_search import EngineSearchFallbackError
        from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
        cfg=SearchConfiguration("incumbent",seconds=1.)
        def parent_init(adapter, configuration, **kwargs):
            adapter.configuration=configuration;adapter.runtime_configuration={}
        with patch.object(PublicModelSearchAdapter,"__init__",parent_init):
            adapter=WorldInputSearchAdapter(cfg,timed_native_atomic_work=timed)
        adapter.contract=SimpleNamespace(showdown_source_sha256="a"*64)
        adapter._closed=adapter._poisoned=False
        adapter._pending_value_diagnostics=adapter._value_diagnostics=None
        adapter._attempted=set();adapter._attempted_statistics_roots=set()
        adapter._search_config=SimpleNamespace()
        adapter.last_failure=None
        class Policy:
            _fixed_override=_ladder_sims_override=_value_diagnostic_session=None
            _world_attempt_observer=_world_observer=None
            _world_construction_observer=_native_construction_observer=None
            _config=SimpleNamespace(model_world_workers=1,early_stop=False,search_sims=16,
                model_decision_time_ms=1000,search_batch=16,worlds=4)
            stats=SimpleNamespace(worlds_attempted=0,worlds_constructed=0,worlds_searched=0,total_iterations=0)
            def reset(self):pass
            def _native(self):return SimpleNamespace(search_batched_multi_encoded=Mock(
                return_value=json.dumps(timed_report(0))))
            def select_action_with_context(self, context, rng):
                for ordinal in range(4):
                    self.stats.worlds_attempted+=1
                    self._world_attempt_observer(context,ordinal,FakeOverride(dict(p1="own",p2="hypothetical")))
                    self.stats.worlds_constructed+=1
                    self._world_observer(context,world(),SimpleNamespace(to_string=lambda:"complete-state"))
                self._native().search_batched_multi_encoded(*timed_arguments())
                evidence,_,_=input_evidence(0)
                diagnostic=dict(schema="engine-search-fallback-refusal-v1",diagnostic_only_not_policy_input=True,
                    battle_id=context.battle_id,round=context.decision_round_index,seat=context.player_id,
                    reason="model_time_budget_no_completed_worlds",world_failures={},
                    engine_mcts=dict(fallback="model_time_budget_no_completed_worlds",
                        time_budget=evidence["engine_mcts"]["time_budget"]))
                if mutation=="root":diagnostic["battle_id"]="another-root"
                if mutation=="seat":diagnostic["seat"]="p2"
                if mutation=="round":diagnostic["round"]=False
                if mutation=="worldfailure":diagnostic["world_failures"]={"operational":1}
                if mutation=="reason":diagnostic["reason"]="crate_search_failed"
                if mutation=="hookerror":adapter._input_recorder.errors.append("ValueError")
                if mutation=="counter":self.stats.worlds_attempted+=1
                if mutation=="boolcounter":self.stats.total_iterations=False
                if mutation=="positive":self.stats.total_iterations=16
                if mutation=="refusedcall":adapter._input_recorder.calls[0]["status"]="REFUSED"
                error=EngineSearchFallbackError("authored strict fallback",diagnostic=diagnostic)
                adapter.authored_error=error
                if mutation=="outer":raise RuntimeError("outer operational failure") from error
                if mutation=="subclass":
                    class DifferentFailure(EngineSearchFallbackError):pass
                    error=DifferentFailure("outer subclass",diagnostic=diagnostic)
                    adapter.authored_error=error
                raise error
        adapter._native=Policy()
        adapter._decider=SimpleNamespace(_snapshot_stats=lambda policy:dict(depth_reached_histogram={}))
        return adapter

    def select(self, adapter):
        # Authored clock values are shared with the actual parent's start seam;
        # no real timing, production model or qualification is implied.
        with patch("pokezero.mcts_eval.search_over_raw_world_inputs.time.perf_counter",side_effect=[100.,101.2]):
            return adapter.select(self.context,root_id="root:configuration",statistics_root_id="root",selection_seed=7)

    def test_normal_strict_zero_returns_no_action_and_consumed_poisoned_attempt(self):
        adapter=self.adapter()
        result=self.select(adapter)
        self.assertEqual(result["status"],"UNQUALIFIED_AT_CEILING")
        self.assertIsNone(result["action"])
        self.assertEqual(result["timed_input_readout"]["completed_iterations"],0)
        self.assertEqual(result["original_failure_receipt"]["status"],"UNCERTAIN_REFUSED")
        self.assertEqual(result["strict_engine_diagnostic"]["reason"],"model_time_budget_no_completed_worlds")
        self.assertTrue(adapter._poisoned)
        for flag in ("retry_authorized","continuation_authorized","runtime_qualified",
                "owning_process_validated","scientific_collection_authorized"):
            self.assertFalse(result[flag])
        self.assertIsNone(adapter._native._world_attempt_observer)
        self.assertIsNone(adapter._native._world_observer)
        self.assertNotIn("_native",vars(adapter._native))
        with self.assertRaises(ValueError):self.select(adapter)

    def test_legacy_strict_zero_remains_failure_without_option(self):
        adapter=self.adapter(timed=False)
        from pokezero.engine_search import EngineSearchFallbackError
        with self.assertRaises(EngineSearchFallbackError):self.select(adapter)
        self.assertTrue(adapter._poisoned)
        self.assertFalse(hasattr(adapter,"last_unqualified"))

    def test_malformed_foreign_or_operational_failure_preserves_original(self):
        for mutation in ("root","seat","round","worldfailure","reason","hookerror",
                "counter","boolcounter","positive","refusedcall","outer","subclass"):
            adapter=self.adapter(mutation)
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError) as captured:self.select(adapter)
            expected=adapter.authored_error if mutation!="outer" else adapter.authored_error.__context__
            if mutation!="outer":self.assertIs(captured.exception,expected)
            else:
                self.assertEqual(str(captured.exception),"outer operational failure")
                self.assertIs(captured.exception.__cause__,adapter.authored_error)
            self.assertTrue(adapter._poisoned)
            self.assertFalse(hasattr(adapter,"last_unqualified"))
            self.assertIsNone(adapter._native._world_attempt_observer)
            if mutation not in ("reason","outer","subclass"):
                self.assertTrue(any("verification failed" in note for note in captured.exception.__notes__))
