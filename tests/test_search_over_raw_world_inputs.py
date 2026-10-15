"""Synthetic owned objects and mocked runtimes only; never starts a game."""
from copy import deepcopy
from dataclasses import dataclass
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.mcts_eval.paper_reference_parallel import PreparedDecision
from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_world_inputs import (
    NativeInputHandle, NativeWorldInputs, ReferenceInputWorld, WorldInputRuntime,
    _WorldInputAdapterMixin, _manifest, compare_world_inputs, reference_manifest,
)

BINDING = dict(root_binding="a"*64, subject="p1", set_source_hash="b"*64, information_key="abc")
CONTEXT = SimpleNamespace(player_id="p1")


@dataclass(frozen=True)
class FakeOverride:
    player_teams: dict


@dataclass(frozen=True)
class FakeWorld:
    spec: dict
    party_species: dict
    slot_sides: dict


def world(hp=80):
    return FakeWorld(dict(hp=hp, hidden_duration=3, pp=15),
        dict(p1=("own",), p2=("sampled",)), dict(p1="side_one", p2="side_two"))


def construct(recorder, text="complete-state", hp=80):
    recorder.attempt(CONTEXT, len(recorder.rows), FakeOverride(dict(p1="own", p2="hypothetical")))
    recorder.constructed(CONTEXT, world(hp), SimpleNamespace(to_string=lambda: text))


def arguments(text="complete-state", iterations=16, chance_seed=7):
    return [text, iterations, 16, '{"tables":true}', '{"public":true}',
        '{"p1":["own"],"p2":["sampled"],"turn":3}',
        SimpleNamespace(to_payload=lambda: {"history":["public"], "annotations":{0:[0., 1.]}}),
        6, 1., chance_seed, True, True]


def native_manifest(recorder, work=16):
    return recorder.finish(dict(total_iterations=work), attempted=len(recorder.rows),
        constructed=sum(row["status"] == "CONSTRUCTED" for row in recorder.rows),
        searched=sum(row["multiplicity"] for row in recorder.calls
            if row["status"] == "RETURNED"
            and (row["completed_iterations"] > 0 or not row["deadline_limited"])))


def snapshot(hp=80, pp=15, seed=7):
    return SimpleNamespace(bridge_snapshot=dict(battle=dict(hp=hp, pp=pp, prng=[seed]),
        boundaryRequests=dict(p1={"moves":["move"]}), terminalScheduled=False),
        protocol_lines=("public",), latest_requests={"p1":{}}, first_requests={"p1":{}},
        request_history={"p1":({},)}, latest_turn=3,
        format_id="gen3randombattle", observation_format_id="gen3randombattle",
        replay={"events":["public"], "hp_visibility":{"p1":"exact"}},
        belief_engine={"pending_switches":[], "pp_ledger":{("p2", "sampled"):15}},
        terminal=None, search_choice_cache={"p1":{0:"move 1"}},
        annotation_cache={"assessed_until":1, "residuals":{0:0.}},
        battle_token="owned-token", battle_id="owned-battle")


def reference_world(snap=None, permitted=True):
    env = SimpleNamespace(_search_snapshot_permitted=permitted,
        snapshot=Mock(return_value=snap or snapshot()))
    return SimpleNamespace(env=env, subject="p1", closed=False,
        frame=Mock(return_value="frame"), evaluate=Mock(return_value="evaluation"),
        advance=Mock(return_value="successor"), close=Mock())


class NativeInputTests(unittest.TestCase):
    def test_zero_iteration_deadline_return_is_visible_but_not_searched(self):
        recorder = NativeWorldInputs(BINDING, 16)
        construct(recorder, text="first"); construct(recorder, text="last")
        base = SimpleNamespace(search_batched_multi_encoded=Mock(
            side_effect=['{"iterations":16}', '{"iterations":0}']))
        handle = NativeInputHandle(base, recorder)
        handle.search_batched_multi_encoded(*arguments(text="first"))
        handle.search_batched_multi_encoded(*arguments(text="last"))
        report = recorder.finish(dict(total_iterations=16), attempted=2, constructed=2, searched=1)
        self.assertEqual([call["completed_iterations"] for call in report["calls"]], [16, 0])
        self.assertEqual([row["consumed"] for row in report["rows"]], [True, True])
        with self.assertRaises(ValueError): recorder.finish(dict(total_iterations=16),
            attempted=2, constructed=2, searched=2)

    def test_actual_fold_history_and_tables_bound_but_leaf_parameter_not_matched(self):
        hashes = []
        for mutation in (None, "fold", "tables", "leaf"):
            recorder = NativeWorldInputs(BINDING, 16)
            construct(recorder)
            args = arguments()
            if mutation == "fold": args[6] = SimpleNamespace(to_payload=lambda: {"history":["changed"]})
            if mutation == "tables": args[3] = '{"tables":false}'
            if mutation == "leaf": args[11] = False
            NativeInputHandle(SimpleNamespace(search_batched_multi_encoded=lambda *a:
                '{"iterations":16}'), recorder).search_batched_multi_encoded(*args)
            hashes.append(native_manifest(recorder)["rows"][0]["input_sha256"])
        self.assertEqual(hashes[0], hashes[3])
        self.assertEqual(len(set(hashes)), 3)

    def test_zero_iteration_without_deadline_preserves_native_counter_semantics(self):
        recorder = NativeWorldInputs(BINDING, 16, deadline_limited=False)
        construct(recorder)
        NativeInputHandle(SimpleNamespace(search_batched_multi_encoded=lambda *a:
            '{"iterations":0}'), recorder).search_batched_multi_encoded(*arguments())
        report = recorder.finish(dict(total_iterations=0), attempted=1, constructed=1, searched=1)
        self.assertEqual(report["calls"][0]["completed_iterations"], 0)
        self.assertFalse(report["calls"][0]["deadline_limited"])
        self.assertFalse(report["runtime_qualified"])

    def test_dispatch_delegates_exact_arguments_and_binds_consumed_state(self):
        recorder = NativeWorldInputs(BINDING, 16)
        construct(recorder)
        base = SimpleNamespace(search_batched_multi_encoded=Mock(return_value='{"iterations":16}'))
        args = arguments()
        result = NativeInputHandle(base, recorder).search_batched_multi_encoded(*args, leaf="model")
        self.assertEqual(result, '{"iterations":16}')
        base.search_batched_multi_encoded.assert_called_once_with(*args, leaf="model")
        report = native_manifest(recorder)
        self.assertTrue(report["rows"][0]["consumed"])
        self.assertEqual(report["calls"][0]["report_sha256"], digest({"iterations":16}))
        self.assertFalse(report["runtime_qualified"])

    def test_duplicate_worlds_preserve_both_construction_ordinals(self):
        recorder = NativeWorldInputs(BINDING, 16)
        construct(recorder); construct(recorder)
        base = SimpleNamespace(search_batched_multi_encoded=lambda *a: '{"iterations":32}')
        NativeInputHandle(base, recorder).search_batched_multi_encoded(*arguments(iterations=32))
        report = native_manifest(recorder, 32)
        self.assertEqual(report["calls"][0]["construction_ordinals"], [0, 1])
        self.assertEqual(len(report["rows"]), 2)

    def test_actual_report_utf8_hash_is_not_a_reconstructed_json_hash(self):
        import hashlib
        raw = ' { "iterations" : 16 }\n'
        recorder = NativeWorldInputs(BINDING, 16)
        construct(recorder)
        NativeInputHandle(SimpleNamespace(search_batched_multi_encoded=lambda *a: raw),
            recorder).search_batched_multi_encoded(*arguments())
        call = native_manifest(recorder)["calls"][0]
        self.assertEqual(call["report_utf8_sha256"], hashlib.sha256(raw.encode()).hexdigest())
        self.assertNotEqual(call["report_utf8_sha256"], call["report_sha256"])

    def test_construction_is_not_consumption(self):
        recorder = NativeWorldInputs(BINDING, 16)
        construct(recorder)
        report = native_manifest(recorder, 0)
        self.assertFalse(report["rows"][0]["consumed"])

    def test_ambiguous_constructed_state_does_not_change_native_call_but_invalidates_witness(self):
        recorder = NativeWorldInputs(BINDING, 16)
        construct(recorder, hp=80); construct(recorder, hp=40)
        base = SimpleNamespace(search_batched_multi_encoded=Mock(return_value='{"iterations":32}'))
        NativeInputHandle(base, recorder).search_batched_multi_encoded(*arguments(iterations=32))
        base.search_batched_multi_encoded.assert_called_once()
        with self.assertRaises(ValueError): native_manifest(recorder, 32)

    def test_swallowed_observer_error_cannot_certify(self):
        recorder = NativeWorldInputs(BINDING, 16)
        with self.assertRaises(ValueError): recorder.attempt(CONTEXT, 1, None)
        with self.assertRaises(ValueError): native_manifest(recorder, 0)

    def test_missing_construction_and_work_mismatch_rejected(self):
        recorder = NativeWorldInputs(BINDING, 16)
        construct(recorder)
        base = SimpleNamespace(search_batched_multi_encoded=lambda *a: '{"iterations":16}')
        NativeInputHandle(base, recorder).search_batched_multi_encoded(*arguments())
        with self.assertRaises(ValueError): native_manifest(recorder, 17)
        with self.assertRaises(ValueError): recorder.finish(dict(total_iterations=16),
            attempted=2, constructed=1, searched=1)

    def test_native_exception_retains_refusal_and_same_exception(self):
        recorder = NativeWorldInputs(BINDING, 16)
        construct(recorder)
        failure = RuntimeError("synthetic failure")
        base = SimpleNamespace(search_batched_multi_encoded=Mock(side_effect=failure))
        with self.assertRaises(RuntimeError) as raised:
            NativeInputHandle(base, recorder).search_batched_multi_encoded(*arguments())
        self.assertIs(raised.exception, failure)
        self.assertEqual(recorder.calls[0]["status"], "REFUSED")
        with self.assertRaises(ValueError): native_manifest(recorder, 0)


class ReferenceInputTests(unittest.TestCase):
    def test_all_python_side_inputs_change_hash_and_only_ownership_metadata_excluded(self):
        def fingerprint(snap):
            draw = dict(status="ROOT_VALIDATED")
            ReferenceInputWorld(reference_world(snap), draw).frame()
            self.assertEqual(draw["consumed_world_input"]["ownership_metadata_excluded"],
                ["battle_token", "battle_id"])
            return draw["consumed_world_input"]["input_sha256"]
        original = fingerprint(snapshot())
        changes = dict(format_id="changed", observation_format_id="changed",
            replay={"events":["different"]}, belief_engine={"pp_ledger":{("p2", "sampled"):14}},
            terminal={"winner":"p2"}, search_choice_cache={"p1":{0:"move 2"}},
            annotation_cache={"assessed_until":2})
        for key, value in changes.items():
            with self.subTest(field=key):
                snap = snapshot(); setattr(snap, key, value)
                self.assertNotEqual(original, fingerprint(snap))
        snap = snapshot(); snap.battle_id = "new-owner"; snap.battle_token = "new-token"
        self.assertEqual(original, fingerprint(snap))

    def test_contract_drift_unknown_state_and_cycles_cannot_certify(self):
        for mutation in ("field", "object", "cycle"):
            snap = snapshot()
            if mutation == "field": snap.future_input = "unknown"
            if mutation == "object": snap.belief_engine = object()
            if mutation == "cycle": snap.belief_engine["cycle"] = snap.belief_engine
            draw = dict(status="ROOT_VALIDATED")
            with self.assertRaises(ValueError): ReferenceInputWorld(reference_world(snap), draw).frame()
            self.assertNotIn("consumed_world_input", draw)

    def test_real_python_state_types_hash_without_new_battle_or_source_reads(self):
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.local_showdown import SnapshotAnnotationCache
        from pokezero.tier2 import Tier2LiveTracker
        from pokezero.investment import InvestmentLiveTracker
        from pokezero.dex import ShowdownDex
        snap = snapshot()
        dex = ShowdownDex({}, {}, {})
        snap.belief_engine = PublicBattleBeliefEngine(format_id="gen3randombattle")
        snap.annotation_cache = SnapshotAnnotationCache(
            {"p1":Tier2LiveTracker(perspective_slot="p1", own_team=(), dex=dex, whitelist={})},
            {"p1":InvestmentLiveTracker(perspective_slot="p1", own_team=(), dex=dex)})
        draw = dict(status="ROOT_VALIDATED")
        ReferenceInputWorld(reference_world(snap), draw).frame()
        snap.belief_engine._stay_locked_move["sampled"] = "move"
        changed = dict(status="ROOT_VALIDATED")
        ReferenceInputWorld(reference_world(snap), changed).frame()
        self.assertNotEqual(draw["consumed_world_input"]["input_sha256"],
            changed["consumed_world_input"]["input_sha256"])

    def test_snapshot_taken_at_actual_first_frame_and_no_additional_sampling(self):
        owned = reference_world()
        draw = dict(ordinal=0, status="ROOT_VALIDATED")
        wrapped = ReferenceInputWorld(owned, draw)
        owned.env.snapshot.assert_not_called()
        self.assertEqual(wrapped.frame(), "frame")
        wrapped.frame(); wrapped.evaluate("state"); wrapped.advance("a", "b", None); wrapped.close()
        owned.env.snapshot.assert_called_once()
        owned.close.assert_called_once()
        self.assertTrue(draw["consumed_world_input"]["consumed"])

    def test_live_source_environment_and_unbound_consumption_refused(self):
        for operation in (lambda w: w.frame(), lambda w: w.evaluate("state"),
                          lambda w: w.advance("a", "b", None)):
            owned = reference_world(permitted=False)
            wrapped = ReferenceInputWorld(owned, dict(status="ROOT_VALIDATED"))
            with self.assertRaises(ValueError): operation(wrapped)
            owned.env.snapshot.assert_not_called()

    def test_pp_hidden_hp_and_rng_are_not_removed_to_force_matching(self):
        hashes = []
        for snap in (snapshot(), snapshot(hp=40), snapshot(pp=14), snapshot(seed=8)):
            draw = dict(status="ROOT_VALIDATED")
            ReferenceInputWorld(reference_world(snap), draw).frame()
            hashes.append(draw["consumed_world_input"]["input_sha256"])
        self.assertEqual(len(set(hashes)), 4)

    def test_actual_prepared_sampler_is_wrapped_and_receipt_owner_checked(self):
        owned = reference_world()
        factory = SimpleNamespace(active=False, receipts=[], env=owned.env,
            state=SimpleNamespace(player_id="p1"))
        def sample(rng):
            factory.receipts.append(dict(ordinal=0, status="ROOT_VALIDATED"))
            return owned
        sampler = Mock(side_effect=sample)
        prepared = PreparedDecision("root", "evaluate", sampler)
        base = SimpleNamespace(prepared_factory=factory, prepare=Mock(return_value=prepared), close=Mock())
        runtime = WorldInputRuntime(base)
        with patch("pokezero.mcts_eval.search_over_raw_world_inputs._binding", return_value=BINDING):
            wrapped = runtime.prepare("request")
        rng = object()
        result = wrapped.sample_world(rng)
        sampler.assert_called_once_with(rng)
        self.assertIsInstance(result, ReferenceInputWorld)
        self.assertEqual(result.frame(), "frame")
        self.assertIn("consumed_world_input", factory.receipts[0])
        runtime.close(); base.close.assert_called_once()

    def test_incremental_exchange_order_does_not_change_manifest(self):
        receipts = []
        for worker, ordinal in ((1, 0), (0, 0), (0, 1)):
            draw = dict(ordinal=ordinal, status="ROOT_VALIDATED", released=True)
            ReferenceInputWorld(reference_world(), draw).frame()
            receipts.append(dict(worker=worker, evidence=dict(draws=[draw])))
        left = reference_manifest(dict(worker_receipts=receipts), BINDING)
        right = reference_manifest(dict(worker_receipts=list(reversed(receipts))), BINDING)
        self.assertEqual(left, right)
        self.assertEqual([(r["worker"], r["ordinal"]) for r in left["rows"]], [(0,0), (0,1), (1,0)])

    def test_missing_duplicate_unreleased_and_gap_fail(self):
        draw = dict(ordinal=0, status="ROOT_VALIDATED", released=True)
        ReferenceInputWorld(reference_world(), draw).frame()
        good = dict(worker=0, evidence=dict(draws=[draw]))
        cases = [dict(worker_receipts=[good, good]),
            dict(worker_receipts=[dict(worker=0, evidence=dict(draws=[dict(draw, ordinal=1)]))]),
            dict(worker_receipts=[dict(worker=0, evidence=dict(draws=[dict(draw, released=False)]))]),
            dict(worker_receipts=[dict(worker=0, evidence=dict(draws=[dict(ordinal=0, status="ROOT_VALIDATED")]))])]
        for evidence in cases:
            with self.assertRaises(ValueError): reference_manifest(evidence, BINDING)


class WorldComparisonTests(unittest.TestCase):
    def row(self, ordinal=0, sha="c"*64, consumed=True):
        return dict(worker=0, ordinal=ordinal, consumed=consumed, input_sha256=sha)

    def test_full_match_is_within_arm_not_cross_engine_equivalence(self):
        manifest = _manifest("incumbent", BINDING, [self.row()])
        result = compare_world_inputs(manifest, deepcopy(manifest))
        self.assertTrue(result["entire_population_matched"])
        self.assertFalse(result["equal_work_claim"])
        self.assertFalse(result["cross_engine_equivalence"])
        with self.assertRaises(ValueError): compare_world_inputs(manifest, dict(manifest, arm="reference"))

    def test_different_prefix_lengths_are_reported_not_erased_or_equality_gate(self):
        left = _manifest("reference", BINDING, [self.row(), self.row(1)])
        right = _manifest("reference", BINDING, [self.row()])
        result = compare_world_inputs(left, right)
        self.assertEqual((result["matched"], result["uncertain"], result["mismatched"]), (1, 1, 0))
        self.assertFalse(result["entire_population_matched"])

    def test_state_mismatch_unknown_empty_and_root_drift_visible(self):
        left = _manifest("incumbent", BINDING, [self.row()])
        right = _manifest("incumbent", BINDING, [self.row(sha="d"*64), self.row(1, consumed=False)])
        result = compare_world_inputs(left, right)
        self.assertEqual((result["mismatched"], result["uncertain"]), (1, 1))
        empty = _manifest("incumbent", BINDING, [])
        self.assertFalse(compare_world_inputs(empty, empty)["entire_population_matched"])
        with self.assertRaises(ValueError): compare_world_inputs(left, dict(right, root_binding="e"*64))

    def test_bad_digest_and_undeclared_exclusion_fail(self):
        good = _manifest("reference", BINDING, [self.row()])
        for bad in (_manifest("reference", BINDING, [self.row(sha="not-a-sha")]),
                    dict(good, snapshot_fields_excluded=["pp"])):
            with self.assertRaises(ValueError): compare_world_inputs(good, bad)


class AdapterHookTests(unittest.TestCase):
    def test_production_mixin_installs_call_wrapper_and_restores_method_on_failure(self):
        class Policy:
            _world_attempt_observer = _world_observer = None
            _config = SimpleNamespace(model_world_workers=1, early_stop=False, search_sims=16,
                model_decision_time_ms=1000)
            stats = SimpleNamespace(worlds_attempted=0, worlds_constructed=0, worlds_searched=0)
            def _native(self): return "genuine-native-handle"
        class Base:
            def _prepare_incumbent(self, request): pass
            def select(self, context, **kwargs):
                self._prepare_incumbent("request")
                self.assert_wrapper = self._native._native()
                raise RuntimeError("synthetic stopped selector")
        class Adapter(_WorldInputAdapterMixin, Base): pass
        adapter = object.__new__(Adapter)
        adapter._native = Policy()
        adapter._expected_public_request_sha256 = None  # This isolated hook fixture bypasses __init__.
        adapter._timed_native_atomic_work = False
        adapter._compact_native_atomic_work = False
        adapter.last_failure = None
        with patch("pokezero.mcts_eval.search_over_raw_world_inputs._binding", return_value=BINDING):
            with self.assertRaises(RuntimeError): adapter.select(None)
        self.assertIsInstance(adapter.assert_wrapper, NativeInputHandle)
        self.assertEqual(adapter._native._native(), "genuine-native-handle")
        self.assertNotIn("_native", vars(adapter._native))
        self.assertIsNone(adapter._native._world_observer)
        self.assertIsNone(adapter._native._world_attempt_observer)


class ActualAdapterPathTests(unittest.TestCase):
    def test_real_incumbent_adapter_captures_calls_and_cleans_hooks(self):
        from pokezero.mcts_eval.search_over_raw import SearchConfiguration
        from pokezero.mcts_eval.search_over_raw_world_inputs import WorldInputSearchAdapter
        from pokezero.policy import PolicyContext, PolicyDecision
        from pokezero.trajectory import BattleTrajectory
        class Policy:
            _world_attempt_observer = _world_observer = None
            _fixed_override = None
            _config = SimpleNamespace(model_world_workers=1, early_stop=False, search_sims=16,
                model_decision_time_ms=1000)
            stats = SimpleNamespace(worlds_attempted=0, worlds_constructed=0, worlds_searched=0,
                total_iterations=0)
            def reset(self): pass
            def _native(self):
                return SimpleNamespace(search_batched_multi_encoded=lambda *a, **k: '{"iterations":16}')
            def select_action_with_context(self, context, rng):
                self._world_attempt_observer(context, 0, FakeOverride({"p1":"own", "p2":"sampled"}))
                self.stats.worlds_attempted += 1
                self._world_observer(context, world(), SimpleNamespace(to_string=lambda: "complete-state"))
                self.stats.worlds_constructed += 1
                self._native().search_batched_multi_encoded(*arguments())
                self.stats.worlds_searched += 1
                self.stats.total_iterations += 16
                return PolicyDecision(0, "synthetic-native", metadata={"engine_mcts":{}})
        policy = Policy()
        stats = lambda p: dict(depth_reached_histogram={}, fallback_decisions=0,
            prior_fallbacks=0, total_iterations=p.stats.total_iterations, model_evals=1)
        decider = SimpleNamespace(_snapshot_stats=stats, _changed_depth=lambda *a: 1, close=Mock())
        contract = SimpleNamespace(checkpoint_path="mock", checkpoint_sha256="a"*64,
            showdown_source_sha256="b"*64)
        observation = SimpleNamespace(legal_action_mask=(True, True))
        context = PolicyContext("p1", 0, "mock", "gen3randombattle", 7,
            observation, ("p1",), BattleTrajectory("mock", "gen3randombattle", 7),
            requested_observations={"p1":observation}, public_materialization_state="mock-public")
        request = SimpleNamespace(state=SimpleNamespace(player_id="p1"), observation=observation,
            set_source_hash="b"*64)
        with patch("pokezero.mcts_eval.search_over_raw_adapters._incumbent_runtime",
                return_value=(decider, policy, "mock-config")), \
             patch("pokezero.mcts_eval.head_to_head.public_only_context", side_effect=lambda c:c), \
             patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture", return_value=request), \
             patch("pokezero.mcts_eval.search_over_raw_world_inputs._binding", return_value=BINDING), \
             patch("pokezero.mcts_eval.search_over_raw_adapters.PublicModelSearchAdapter._validate_incumbent_selection"):
            adapter = WorldInputSearchAdapter(SearchConfiguration("incumbent"),
                checkpoint_contract=contract, showdown_root="mock-showdown")
            try:
                row = adapter.select(context, root_id="root:model", statistics_root_id="root", selection_seed=7)
                manifest = row["evidence"]["consumed_world_inputs"]
                self.assertTrue(manifest["rows"][0]["consumed"])
                self.assertEqual(manifest["calls"][0]["completed_iterations"], 16)
                self.assertEqual(row["statistics_root_id"], "root")
                self.assertIn("consumed_world_inputs", adapter.runtime_configuration)
                self.assertIsNone(policy._world_observer)
                self.assertNotIn("_native", vars(policy))
            finally:
                adapter.close()
        decider.close.assert_called_once()


if __name__ == "__main__": unittest.main()
