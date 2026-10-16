"""Real Python constructor hooks over authored state, no scientific source."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
from pokezero.mcts_eval.search_over_raw_world_inputs import (
    NativeWorldInputs, public_request_witness, WorldInputFixedWorkAdapter, _state_payload)
from pokezero.mcts_eval.search_over_raw_ancestry import NativeAncestryContract
from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
from pokezero.mcts_eval.search_over_raw_public_fold import (
    capture_request, read_construction_public_fold, read_dispatch_inputs, runtime_receipt)
from pokezero.mcts_eval.search_over_raw_diagnostic_producer import fixed_work_runtime_recipe, diagnostic_manifest
import test_search_over_raw_archive as archive_fixture
import test_search_over_raw_construction as construction_fixture
import test_search_over_raw_workflow_preflight as preflight_fixture
import test_search_over_raw_consumed_materialization as consumed_fixture


class ConstructionPublicFoldTests(unittest.TestCase):
    def setUp(self):
        self.f = archive_fixture.ExplorationRootCrosswalkTests()
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.context = self.f.context
        self.request = PublicRootRequest.capture(self.context.public_materialization_state,
            self.context.observation)
        self.witness = public_request_witness(self.request)

    def recorder(self, enabled=True, pending=None):
        request = PublicRootRequest.capture(self.context.public_materialization_state,
            self.context.observation, pending_transition=pending)
        witness = public_request_witness(request)
        return NativeWorldInputs(witness["consumed_binding"], 16, deadline_limited=False,
            record_atomic_work=True, record_construction=True, expected_public_request=witness,
            pending_transition=pending, record_public_fold=enabled)

    def construct(self, recorder):
        with patch.object(construction_fixture, "CONTEXT", self.context):
            return construction_fixture.construct_actual_python(recorder)

    def read(self, row, request=None):
        return read_construction_public_fold(row.get("public_fold_capture"), ordinal=row["ordinal"],
            construction_receipt=row["construction_receipt"], consumed_binding=self.witness["consumed_binding"],
            public_request_sha256=self.witness["public_request_sha256"],
            expected_request=self.request if request is None else request)

    def dispatch_fixture(self, enabled=True):
        from test_search_over_raw_world_inputs import arguments
        recorder = self.recorder(enabled)
        for _ in range(4):
            world, state = self.construct(recorder)
        args = arguments(text=state.to_string(), iterations=64)
        args[5] = json.dumps({seat:list(world.party_species[seat]) for seat in ("p1", "p2")} | {"turn":3})
        return recorder, args, recorder.dispatch(args)

    def test_dispatch_retains_exact_already_hashed_payload_only_in_explicit_fixed_lane(self):
        recorder, args, call = self.dispatch_fixture()
        payload = read_dispatch_inputs(call, rows=recorder.rows, expected_tables={"tables":True})
        self.assertEqual(digest(payload), call["input_sha256"])
        self.assertEqual(payload["fold"], _state_payload(args[6].to_payload()))
        self.assertTrue(call["dispatch_inputs"]["private_controller_only"])
        self.assertEqual(call["construction_ordinals"], list(range(4)))
        # Returned data cannot mutate the retained call or original inputs.
        payload["context"]["p1"].clear()
        read_dispatch_inputs(call, rows=recorder.rows)
        ordinary, _, ordinary_call = self.dispatch_fixture(False)
        self.assertNotIn("dispatch_inputs", ordinary_call)
        self.assertEqual(call["input_sha256"], ordinary_call["input_sha256"])
        self.assertFalse(ordinary.record_public_fold)
        self.assertTrue(runtime_receipt()["native_dispatch_inputs"]["fixed_work_only"])

    def test_dispatch_reads_native_fold_once_and_does_not_reconstruct_after_hash(self):
        recorder, args, _ = self.dispatch_fixture()
        # Use a fresh recorder: each construction can be dispatched only once.
        recorder = self.recorder()
        for _ in range(4): self.construct(recorder)
        class Fold:
            calls = 0
            def to_payload(self):
                self.calls += 1
                return {"history":[self.calls]}
        fold = Fold(); args[6] = fold
        call = recorder.dispatch(args)
        self.assertEqual(fold.calls, 1)
        self.assertEqual(read_dispatch_inputs(call, rows=recorder.rows)["fold"],
            _state_payload({"history":[1]}))

    def test_dispatch_reader_rejects_drift_missing_capture_and_wrong_external_tables(self):
        recorder, _, original = self.dispatch_fixture()
        for mode in ("missing", "state", "fold", "tables", "context", "seed", "call_seed", "world", "timed", "scope"):
            call = deepcopy(original)
            capture = call["dispatch_inputs"]
            if mode == "missing": call.pop("dispatch_inputs")
            elif mode == "state": capture["payload"]["state"] = "changed"
            elif mode == "fold": capture["payload"]["fold"] = {"mapping":[]}
            elif mode == "tables": capture["payload"]["tables"] = {"tables":False}
            elif mode == "context": capture["payload"]["context"]["turn"] += 1
            elif mode == "seed": capture["payload"]["chance_seed"] += 1
            elif mode == "call_seed": call["native_call_parameters"]["chance_seed"] += 1
            elif mode == "world": capture["payload"]["world_sha256"] = "a"*64
            elif mode == "timed": call["deadline_limited"] = True
            else: capture["private_controller_only"] = False
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                read_dispatch_inputs(call, rows=recorder.rows)
        with self.assertRaisesRegex(ValueError, "independently pinned tables"):
            read_dispatch_inputs(original, rows=recorder.rows, expected_tables={"tables":False})
        rows = deepcopy(recorder.rows)
        rows[0]["construction_receipt"]["events"][1]["value"]["native_state"] = "substitution"
        with self.assertRaisesRegex(ValueError, "original construction"):
            read_dispatch_inputs(original, rows=rows)

    def test_public_fold_scope_cannot_be_enabled_after_recorder_creation(self):
        recorder, args, _ = self.dispatch_fixture(False)
        recorder.record_public_fold = True
        with self.assertRaisesRegex(ValueError, "public-fold capture scope"):
            recorder.dispatch(args)

    def test_actual_attempt_and_constructor_capture_exact_full_public_request(self):
        recorder = self.recorder(); self.construct(recorder)
        row = recorder.rows[0]; capture = row["public_fold_capture"]
        checked = self.read(row)
        self.assertTrue(checked["full_public_payload_bound"])
        self.assertEqual(capture["public_request_payload"], _state_payload(self.request))
        encoded = str(capture)
        self.assertNotIn("LocalShowdownSnapshot", encoded)
        self.assertNotIn("private-source-snapshot", encoded)
        self.assertNotIn("RandbatSetSource", encoded)
        self.assertIsNone(self.request.state.belief_engine.set_source)
        self.assertFalse(checked["process_ownership_validated"])
        self.assertFalse(checked["labels_produced"])
        self.assertFalse(checked["runtime_qualified"])

    def test_capture_preserves_original_slot_roster_and_is_default_off(self):
        recorder = self.recorder()
        for _ in range(4): self.construct(recorder)
        self.assertEqual([self.read(row)["construction_ordinal"] for row in recorder.rows], list(range(4)))
        ordinary = self.recorder(False); self.construct(ordinary)
        self.assertNotIn("public_fold_capture", ordinary.rows[0])

    def test_history_changed_after_attempt_fails_not_root_only_matching(self):
        recorder = self.recorder()
        original = construction_fixture.build_poke_engine_state
        def changed(*args, **kwargs):
            state = original(*args, **kwargs)
            public = self.context.public_materialization_state
            object.__setattr__(public, "self_move_states", {"changed-history": ()})
            return state
        with patch.object(construction_fixture, "build_poke_engine_state", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "public fold changed"):
                self.construct(recorder)
        self.assertTrue(recorder.errors)
        self.assertNotIn("public_fold_capture", recorder.rows[0])

    def test_missing_payload_witness_pending_coordinate_and_scope_substitutions_refuse(self):
        recorder = self.recorder(); self.construct(recorder)
        original = recorder.rows[0]
        for mode in ("missing", "payload", "witness", "coordinate", "bool-coordinate", "construction", "scope"):
            row = deepcopy(original)
            capture = row["public_fold_capture"]
            if mode == "missing": row.pop("public_fold_capture")
            elif mode == "payload": capture["public_request_payload"]["fields"]["pending_transition"] = "substituted"
            elif mode == "witness": capture["witness"]["consumed_binding"]["subject"] = "p2"
            elif mode == "coordinate": capture["construction_ordinal"] = 1
            elif mode == "bool-coordinate": capture["construction_ordinal"] = False
            elif mode == "construction": capture["construction_capture_sha256"] = "a" * 64
            else: capture["process_ownership_validated"] = True
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.read(row)
        altered = replace(self.request, pending_transition="different-original-context")
        with self.assertRaisesRegex(ValueError, "original typed consumed public fold"):
            self.read(original, altered)

    def test_unreviewed_timed_or_nonconstruction_capture_refuses(self):
        for kwargs in (dict(deadline_limited=True, record_construction=True),
                dict(deadline_limited=False, record_construction=False)):
            with self.assertRaisesRegex(ValueError, "public fold retention"):
                NativeWorldInputs(self.witness["consumed_binding"],16,
                    expected_public_request=self.witness,record_public_fold=True,**kwargs)

    def test_actual_disposable_owner_validates_all_folds_and_refuses_durably_missing_fold(self):
        fixture = consumed_fixture.ArchivedDiagnosticAttachmentTests()
        self.addCleanup(fixture.doCleanups)
        archive, selected, reader = fixture.fixture(record_public_fold=True)
        self.assertEqual(reader(selected), digest(selected))
        self.assertTrue(reader.last_validation["construction_public_fold_capture_validated"])
        rows = selected["evidence"]["consumed_world_inputs"]["rows"]
        self.assertEqual([row["public_fold_capture"]["construction_ordinal"] for row in rows], list(range(4)))
        from pokezero.mcts_eval.search_over_raw_consumed_materialization import _original_diagnostic_owner_guard
        guard = _original_diagnostic_owner_guard(selected, [], [], reader)
        original_readout = deepcopy(reader.last_validation)
        reader.native_public_fold_required = False
        with self.assertRaises(ValueError): guard({str(reader.directory):original_readout})
        reader.native_public_fold_required = True
        changed = deepcopy(selected)
        changed["evidence"]["consumed_world_inputs"]["rows"][2].pop("public_fold_capture")
        # Model a consistently rehashed child return. Actual ownership receipts
        # remain untouched; the independent reader must still reject missing
        # observed public evidence rather than trust a normal exit or summary.
        for name in ("worker-result.json", "terminal.json"):
            path = reader.directory / name
            record = json.loads(path.read_text()); record["result"] = changed
            path.write_text(json.dumps(record))
        with self.assertRaises(ValueError): reader(changed)
        self.assertIsNone(reader.last_validation)

    def test_completed_owner_rejects_durably_missing_dispatch_payload(self):
        fixture = consumed_fixture.ArchivedDiagnosticAttachmentTests()
        self.addCleanup(fixture.doCleanups)
        _, selected, reader = fixture.fixture(record_public_fold=True)
        self.assertEqual(reader(selected), digest(selected))
        manifest = selected["evidence"]["consumed_world_inputs"]
        for call in manifest["calls"]:
            read_dispatch_inputs(call, rows=manifest["rows"], expected_tables={"tables":True})
        changed = deepcopy(selected)
        changed["evidence"]["consumed_world_inputs"]["calls"][0].pop("dispatch_inputs")
        # A consistently rewritten child result cannot bypass the independent
        # owning reader's required actual-input content check.
        for name in ("worker-result.json", "terminal.json"):
            path = reader.directory / name
            record = json.loads(path.read_text()); record["result"] = changed
            path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "dispatch input capture"):
            reader(changed)
        self.assertIsNone(reader.last_validation)

    def test_prospective_runtime_matches_actual_adapter_and_is_not_original_runtime(self):
        h = preflight_fixture.DiagnosticWorkflowPreflightTests()
        self.addCleanup(h.doCleanups); h.prepare()
        cfg = SearchConfiguration("incumbent", seconds=1.)
        ancestry = NativeAncestryContract(17,capture_native_edges=True)
        expected = fixed_work_runtime_recipe(cfg,h.checkpoint,native_ancestry_contract=ancestry,
            capture_native_public_fold=True)
        with patch.object(WorldInputFixedWorkAdapter,"_make_incumbent_runtime",return_value=(None,None,None)):
            adapter = WorldInputFixedWorkAdapter(cfg,checkpoint_contract=h.checkpoint,
                showdown_root=h.checkpoint.showdown_root,fixed_work=FixedWorkDiagnosticContract(),
                native_ancestry_contract=ancestry,expected_public_request_sha256=self.witness["public_request_sha256"],
                capture_native_public_fold=True)
            try: self.assertEqual(digest(adapter.runtime_configuration),digest(expected))
            finally: adapter.close()
        self.assertEqual(expected["native_public_fold"],runtime_receipt())
        self.assertNotEqual(digest(expected),digest(fixed_work_runtime_recipe(cfg,h.checkpoint,native_ancestry_contract=ancestry)))

    def test_direct_manifest_checks_flag_runtime_parity_before_any_launch(self):
        h = preflight_fixture.DiagnosticWorkflowPreflightTests()
        self.addCleanup(h.doCleanups); fixture = h.prepare()
        cfg = SearchConfiguration("incumbent", seconds=1.)
        original = h.recipes[cfg.identity]
        valid = replace(original, capture_native_public_fold=True,
            runtime={**original.runtime, "native_public_fold":runtime_receipt()})
        cases = [valid,
            replace(valid, runtime=original.runtime),
            replace(original, runtime=valid.runtime),
            replace(original, runtime={**original.runtime,"native_public_fold":None}),
            replace(valid, runtime={**valid.runtime,"native_public_fold":{}})]
        # File contents/reviews need no mutation: this is the direct pure
        # registry path, not the separate whole-entry admission path.
        with patch("subprocess.Popen") as launch:
            for index, recipe in enumerate(cases):
                recipes = {**h.recipes, cfg.identity:recipe}
                arguments = dict(checkpoint_contract=h.checkpoint, showdown_root=h.checkpoint.showdown_root,
                    file_hashes={})
                if index == 0:
                    manifest = diagnostic_manifest(fixture.ledger.contract,recipes,**arguments)
                    self.assertTrue(next(r for r in manifest["recipes"] if r["configuration_sha256"]==cfg.identity)
                        ["capture_native_public_fold"])
                else:
                    with self.subTest(index=index),self.assertRaisesRegex(ValueError,"public-fold runtime"):
                        diagnostic_manifest(fixture.ledger.contract,recipes,**arguments)
        launch.assert_not_called()
