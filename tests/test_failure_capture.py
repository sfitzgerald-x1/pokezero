"""Capture contracts: authored failures, not scientific measurements."""
import json
import os
from pathlib import Path
import random
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.failure_capture import (CaptureFailed, Quarantine, boundary,
    canonical, current, decision_scope, digest_file, record_refusal, snapshot,
    record_native_refusals, require_capture_admission, require_fixture_coverage, runtime_build)


class FailureCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.q = Quarantine(self.path, build={"fingerprint": "authored-build"})
        self.context = SimpleNamespace(battle_id="b", seed=17, player_id="p1",
            decision_round_index=2, public_materialization_state=SimpleNamespace(
                self_request={"active": [{"moves": ["surf"]}]},
                replay=SimpleNamespace(public_lines=["|turn|3"])))
        self.rng = random.Random(17)

    def bundles(self):
        return [json.loads(p.read_text()) for p in self.path.glob("failure-*.json")]

    def test_snapshot_memo_is_detached_and_never_hides_later_input_changes(self):
        shared = {'world_spec':{'hp':[200,180]},'draw_state':[1,2,3]}
        batch = [{'origin':shared,'branch':0},{'origin':shared,'branch':1}]
        first = snapshot(batch)
        expected = [{'origin':snapshot(shared),'branch':0},{'origin':snapshot(shared),'branch':1}]
        self.assertEqual(canonical(first),canonical(expected))
        frozen = canonical(first)
        shared['world_spec']['hp'][1] = 0
        shared['draw_state'].append(4)
        self.assertEqual(canonical(first),frozen)
        second = snapshot(batch)
        self.assertEqual(second[0]['origin']['world_spec']['hp'],[200,0])
        self.assertNotEqual(canonical(second),frozen)

    def test_cyclic_capture_input_fails_closed_without_a_partial_update(self):
        cycle = [];cycle.append(cycle)
        with self.assertRaises(CaptureFailed):
            with decision_scope(self.q,self.context,self.rng):
                current().update(cyclic=cycle)
        self.assertEqual(self.q.failures,1)

    def test_memo_preserves_supported_typed_values_and_detaches_numpy_inputs(self):
        from dataclasses import dataclass
        from enum import Enum
        import numpy as np
        class Mode(Enum):
            VALUE = 'declared'
        @dataclass
        class Record:
            value: list
        array = np.array([[1.0,2.0]],dtype='<f8')
        payload = dict(mode=Mode.VALUE,records=[Record([1]),Record([2])],
            ordering=frozenset((3,1)),array=array,alias=array,tuple_value=(True,1,1.0,-0.0))
        captured = snapshot(payload)
        expected_array = dict(encoding='numpy-c-order-hex',dtype='<f8',shape=[1,2],
                              data=array.tobytes().hex())
        self.assertEqual(captured,dict(mode='declared',records=[{'value':[1]},{'value':[2]}],
            ordering=[1,3],array=expected_array,alias=expected_array,
            tuple_value=[True,1,1.0,-0.0]))
        frozen = canonical(captured)
        array[0,0] = 99
        self.assertEqual(canonical(captured),frozen)
        self.assertNotEqual(snapshot(payload)['array']['data'],expected_array['data'])

    def test_position_capture_matches_complete_snapshot_across_mutation_and_batch_reset(self):
        from collections import Counter
        from pokezero.shallow_exact_native import NativeExactBackend
        from pokezero.failure_capture import unavailable
        backend = object.__new__(NativeExactBackend)
        backend.stage_seconds = Counter()
        backend.capture_context_updates = 0
        shared = dict(world_ordinal=2,world_spec={'hp':[200,180]},
            rng_state_before_draw=[1,[2,3]],row_action='switch mewtwo',
            column_action='aerialace',branch_index=12,generating_pre_state='root',generating_turn=1)
        position = SimpleNamespace(origin=shared,state_str='post',instruction_list=('Damage SideOne: 1',))
        with decision_scope(self.q,self.context,self.rng):
            for change in ('first','same','mutated','integer_to_bool','integer_to_float',
                           'negative_zero','positive_zero','batch_reset','new_world','no_origin'):
                if change == 'mutated':
                    shared['world_spec']['hp'][1] = 179
                    shared['rng_state_before_draw'][1].append(4)
                elif change == 'batch_reset':
                    backend._capture_batch_context()
                elif change == 'integer_to_bool':
                    shared['world_spec']['hp'][1] = True
                elif change == 'integer_to_float':
                    shared['world_spec']['hp'][1] = 1.0
                elif change == 'negative_zero':
                    shared['world_spec']['hp'][1] = -0.0
                elif change == 'positive_zero':
                    shared['world_spec']['hp'][1] = 0.0
                elif change == 'new_world':
                    shared['world_ordinal'] = 3
                elif change == 'no_origin':
                    position.origin = None
                reference = snapshot(current().fields)
                reference.update(snapshot({
                    **{key:unavailable('this root has no generating branch address') for key in
                       ('row_action','column_action','branch_index','generating_pre_state','generating_turn')},
                    **(position.origin or {}),'engine_state':position.state_str,
                    'instruction_list':list(position.instruction_list),
                    'offending_instruction':unavailable('no failing instruction reported')}))
                backend._capture_position(position)
                self.assertEqual(canonical(current().fields),canonical(reference),change)
            retained = canonical(current().fields)
            shared['world_spec']['hp'][0] = 0
            self.assertEqual(canonical(current().fields),retained)  # Detached from later input mutation.
        self.assertEqual(backend.capture_context_updates,10)

    def test_snapshot_comparison_never_hides_typed_or_in_place_changes(self):
        from pokezero.shallow_exact_native import _same_capture_snapshot
        original = {'nested':[True,1,1.0,-0.0,{'move':'surf'}]}
        saved = snapshot(original)
        self.assertTrue(_same_capture_snapshot(saved, original))
        for value in (1,1.0,0.0,None,(True,1,1.0,-0.0,{'move':'surf'})):
            changed = {'nested':value}
            self.assertFalse(_same_capture_snapshot(saved,changed),value)
        for before,after in ((True,1),(1,1.0),(-0.0,0.0),('1',1),
                             ([1],(1,)),({'1':2},{1:2})):
            with self.subTest(before=before,after=after):
                self.assertFalse(_same_capture_snapshot(snapshot({'x':before}),{'x':after}))
        original['nested'][4]['move'] = 'icebeam'
        self.assertFalse(_same_capture_snapshot(saved,original))
        # A saved field changed in place must also be observed: no cached bytes.
        saved['nested'][4]['move'] = 'thunderbolt'
        self.assertFalse(_same_capture_snapshot(saved,original))

    def test_snapshot_comparison_matches_independent_allocations_and_alias_patterns(self):
        from pokezero.shallow_exact_native import _same_capture_snapshot
        shared = {'species':'independently allocated long species label','hp':[200,180]}
        payload = {'world':[shared,shared],'draw':[17,[2,3,4]]}
        encoded = json.dumps(payload)
        first = json.loads(encoded)
        second = json.loads(encoded)
        self.assertIsNot(first['world'][0]['species'],second['world'][0]['species'])
        self.assertTrue(_same_capture_snapshot(snapshot(first),second))
        self.assertTrue(_same_capture_snapshot(snapshot(payload),second))

    def test_snapshot_comparison_rejects_nonfinite_cycles_and_custom_reducers(self):
        from pokezero.shallow_exact_native import _same_capture_snapshot
        class Hostile:
            def __reduce__(self):
                raise AssertionError('comparison must not invoke reducers')
        class Integer(int):
            pass
        for value in (float('nan'),float('inf'),-float('inf'),Hostile(),Integer(1)):
            self.assertFalse(_same_capture_snapshot({'x':1},{'x':value}))
        cycle=[];cycle.append(cycle)
        self.assertFalse(_same_capture_snapshot({'x':[]},{'x':cycle}))
        for value in (float('nan'),float('inf'),-float('inf')):
            self.assertFalse(_same_capture_snapshot(snapshot({'x':value}),{'x':value}))

    def test_early_failure_has_request_prefix_seed_and_explicit_unavailable_fields(self):
        error = ValueError("before world construction")
        with self.assertRaises(ValueError) as raised:
            with decision_scope(self.q, self.context, self.rng):
                raise error
        self.assertIs(raised.exception, error)
        bundle, = self.bundles()
        self.assertEqual(bundle["request"], self.context.public_materialization_state.self_request)
        self.assertEqual(bundle["public_log_prefix"], ["|turn|3"])
        self.assertEqual(bundle["root"]["seed"], 17)
        self.assertEqual(bundle["engine_state"]["availability"], "unavailable")
        self.assertEqual(bundle["instruction_list"]["availability"], "unavailable")
        self.assertEqual(bundle["reason_codes"], ["decision.contract_failure"])
        self.assertEqual(len(list(self.path.glob("decision-terminal-*.json"))), 1)

    def test_scientific_engine_owner_cannot_silently_disable_capture(self):
        from pokezero.engine_search import EngineMctsPolicy
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "POKEZERO_QUARANTINE_DIR"):
                EngineMctsPolicy(dex=None, set_source=None, scientific_capture=True)
        with patch.dict(os.environ, {"POKEZERO_QUARANTINE_DIR":str(self.path)}, clear=True):
            with self.assertRaisesRegex(ValueError, "admission manifest"):
                EngineMctsPolicy(dex=None, set_source=None, scientific_capture=True)

    def test_capture_precedes_world_drop_and_does_not_consume_rng_or_replace_error(self):
        class WorldUnsupported(ValueError):
            reason = "authored_surface"
        error = WorldUnsupported("exact offending field")
        @boundary("world.construct", arguments=("payload", "override"))
        def construct(payload, override, rng):
            raise error
        before = self.rng.getstate()
        with decision_scope(self.q, self.context, self.rng):
            current().update(engine_state="previous world's state")
            try:
                construct({"turn": 3}, {"teams": "sampled"}, self.rng)
            except WorldUnsupported as caught:
                self.assertIs(caught, error)
                bundle, = self.bundles()  # Already durable INSIDE the catch/drop seam.
                self.assertEqual(bundle["reason_codes"], ["world.authored_surface"])
                self.assertEqual(bundle["world.construct"]["override"], {"teams": "sampled"})
                self.assertEqual(bundle["engine_state"]["availability"], "unavailable")
        self.assertEqual(self.rng.getstate(), before)

    def test_native_origin_is_preserved_before_outer_generic_failure(self):
        error = ValueError("native contract")
        error.pokezero_failure_witness = json.dumps(dict(
            reason_codes=["crate.leaf_encode"], engine_state="native serialized state",
            instruction_list=["Damage SideOne: 3"],
            offending_instruction={"availability": "unavailable", "reason": "not uniquely identified"}))
        with self.assertRaises(ValueError):
            with decision_scope(self.q, self.context, self.rng):
                record_refusal("crate", error)
                raise error
        bundle, = self.bundles()
        self.assertEqual(bundle["reason_codes"], ["crate.leaf_encode"])
        self.assertEqual(bundle["engine_state"], "native serialized state")
        self.assertEqual(bundle["instruction_list"], ["Damage SideOne: 3"])
        self.assertEqual(self.q.events, 1)
        self.assertEqual(self.q.reason_counts, {"crate.leaf_encode": 1})

    def test_multicode_events_are_not_confused_with_exception_or_boolean_counts(self):
        with decision_scope(self.q, self.context, self.rng):
            record_refusal("renderer", codes=["renderer.a", "renderer.b", "renderer.a"])
        self.assertEqual(self.q.telemetry(), dict(events=1, capture_successes=1,
            capture_failures=0, reason_counts={"renderer.a": 1, "renderer.b": 1}))

    def test_successful_native_uniform_fallback_is_captured_without_becoming_an_exception(self):
        event = dict(reason_codes=["crate.prior.unmapped_action"], engine_state="actual-prior-state",
                     instruction_list=["SleepTalkCalled SideOne EARTHQUAKE"],
                     batch_row=3, prior_seat="acting", action_map=[1, None])
        with decision_scope(self.q, self.context, self.rng):
            record_native_refusals(dict(prior_fallbacks=1, capture_refusals=[event]))
            bundle, = self.bundles()
            self.assertEqual(bundle['engine_state'], 'actual-prior-state')
            self.assertEqual(bundle['origin_witness']['action_map'], [1, None])
        terminal, = self.path.glob('decision-terminal-*.json')
        self.assertEqual(json.loads(terminal.read_text())['status'], 'RETURNED')

    def test_native_refusal_event_survives_a_later_world_abort_once(self):
        error = ValueError("later world abort")
        error.pokezero_refusal_events = json.dumps([dict(
            reason_codes=["crate.prior.unmapped_action"], engine_state="prior-state", instruction_list=[])])
        with self.assertRaises(ValueError):
            with decision_scope(self.q, self.context, self.rng):
                record_refusal("crate", error)
                raise error
        self.assertEqual(len(self.bundles()), 2)
        self.assertEqual(self.q.reason_counts['crate.prior.unmapped_action'], 1)

    def test_counter_only_prior_fallback_is_a_capture_failure_not_healthy_silence(self):
        with self.assertRaises(CaptureFailed):
            with decision_scope(self.q, self.context, self.rng):
                record_native_refusals(dict(prior_fallbacks=1))
        self.assertEqual(self.q.failures, 1)

    def test_structurally_marked_native_capture_loss_is_counted_once_and_propagated(self):
        error = BaseException("native evidence attachment failed")
        error.pokezero_capture_failed = True
        error.pokezero_failure_witness = json.dumps(dict(
            reason_codes=['crate.model_search'], engine_state='outer-native-root'))
        with self.assertRaises(BaseException) as raised:
            with decision_scope(self.q, self.context, self.rng):
                record_refusal("crate", error)
                raise error
        self.assertIs(raised.exception, error)
        self.assertEqual(self.q.failures, 1)
        bundle, = self.bundles()
        self.assertEqual(bundle['reason_codes'], ['capture.failed', 'crate.model_search'])

    def test_capture_failure_cannot_be_swallowed_as_an_ordinary_world_failure(self):
        with self.assertRaises(CaptureFailed):
            with decision_scope(self.q, self.context, self.rng):
                with patch.object(self.q, "_write", side_effect=OSError("authored full disk")):
                    try:
                        record_refusal("renderer", ValueError("original condition"))
                    except Exception:
                        self.fail("capture loss must not enter legacy catch-and-continue")
        self.assertEqual(self.q.failures, 1)

    def test_missing_native_witness_fields_are_not_invented(self):
        with decision_scope(self.q, self.context, self.rng):
            record_refusal("crate", ValueError("generator unwound before branches existed"))
        bundle, = self.bundles()
        self.assertEqual(bundle["offending_instruction"]["availability"], "unavailable")
        self.assertEqual(bundle["instruction_list"]["availability"], "unavailable")

    def test_failure_files_are_private_immutable_and_hash_linked(self):
        with decision_scope(self.q, self.context, self.rng):
            receipt = record_refusal("renderer", codes=["renderer.authored"])
        path = self.path / receipt["file"]
        self.assertEqual(digest_file(path), receipt["sha256"])
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertIn("input_receipt", json.loads(path.read_text()))

    def test_nonfinite_input_is_preserved_bit_exactly_not_lost_to_json_nan(self):
        value = float("nan")
        encoded = json.loads(canonical({"invalid": value}))
        self.assertEqual(encoded["invalid"], dict(
            encoding="ieee754-binary64-big-endian-hex", data=struct.pack(">d", value).hex()))

    def test_runtime_build_hashes_compiled_child_not_only_package_init(self):
        package = self.path / "native_package"
        package.mkdir()
        init = package / "__init__.py"
        child = package / "pokezero_search.authored.so"
        init.write_text("# authored import wrapper\n")
        child.write_bytes(b"authored compiled binary")
        build = runtime_build(SimpleNamespace(__file__=str(init)))
        self.assertEqual(build['native_binaries']['pokezero_search'][child.name], digest_file(child))
        child.write_bytes(b"different compiled binary")
        self.assertNotEqual(runtime_build(SimpleNamespace(__file__=str(init)))['fingerprint'],
                            build['fingerprint'])

    def test_terminal_capture_loss_is_counted_and_cannot_return_a_decision(self):
        write = self.q._write
        def failing_terminal(kind, payload):
            if kind == "decision-terminal":
                raise OSError("authored terminal disk failure")
            return write(kind, payload)
        with patch.object(self.q, "_write", side_effect=failing_terminal):
            with self.assertRaises(CaptureFailed):
                with decision_scope(self.q, self.context, self.rng):
                    pass
        self.assertEqual(self.q.telemetry()["capture_failures"], 1)
        self.assertEqual(len(list(self.path.glob("decision-input-*.json"))), 1)
        self.assertEqual(len(list(self.path.glob("decision-terminal-*.json"))), 0)

    def test_position_rebind_uses_failed_world_not_last_constructed_world(self):
        from collections import Counter
        from pokezero.shallow_exact_native import NativeExactBackend, NativePosition
        backend = object.__new__(NativeExactBackend)
        backend.stage_seconds = Counter()
        backend.capture_context_updates = 0
        position = NativePosition("world-zero-state", {}, 3, None, None, "side_one",
            origin={"world_ordinal": 0, "world_spec": {"id": "zero"},
                    "engine_state": "stale-origin-state"})
        with decision_scope(self.q, self.context, self.rng):
            current().update(world_ordinal=3, world_spec={"id": "three"})
            backend._capture_position(position)
            record_refusal("crate", codes=["crate.authored"])
        bundle, = self.bundles()
        self.assertEqual(bundle["world_ordinal"], 0)
        self.assertEqual(bundle["world_spec"], {"id": "zero"})
        self.assertEqual(bundle["engine_state"], "world-zero-state")

    def test_incumbent_world_keeps_construction_rng_and_override_through_later_attempts(self):
        from pokezero.engine_search import EngineMctsPolicy
        policy = object.__new__(EngineMctsPolicy)
        policy._capture_world_origins = {}
        worlds = [SimpleNamespace(spec={"world": i}) for i in range(4)]
        states = [SimpleNamespace(to_string=lambda i=i:f"state-{i}") for i in range(4)]
        with decision_scope(self.q, self.context, self.rng):
            for i, (world, state) in enumerate(zip(worlds, states)):
                current().update(world_ordinal=i, rng_state_before_draw=[i], player_teams={"p2":f"team-{i}"})
                policy._remember_capture_world(world, state)
            current().update(world_ordinal=4, player_teams={"availability":"unavailable"})
            policy._bind_capture_world(worlds[0], states[0])
            record_refusal("crate", codes=["crate.authored"])
        bundle, = self.bundles()
        self.assertEqual(bundle["world_ordinal"], 0)
        self.assertEqual(bundle["rng_state_before_draw"], [0])
        self.assertEqual(bundle["player_teams"], {"p2":"team-0"})
        self.assertEqual(bundle["engine_state"], "state-0")

    def test_child_leaf_keeps_its_generating_cell_not_the_last_enumerated_cell(self):
        from pokezero.shallow_exact_native import NativeExactBackend, NativePosition
        from tests.test_shallow_exact_native import Encoder, Fold, Model, Native
        with patch.dict(os.environ, POKEZERO_ENUMERATE_ROLLS="1"):
            backend = NativeExactBackend(Native(), Model())
        root = NativePosition("original-root", {"p1":("a",),"p2":("b",)}, 3,
                              Fold(), Encoder(), "side_one")
        with decision_scope(self.q, self.context, self.rng):
            earlier = list(backend.outcomes(root, "surf", "switch blissey"))[0].state
            list(backend.outcomes(root, "different-row", "different-column"))
            backend._capture_position(earlier)
            record_refusal("crate", codes=["crate.authored"])
        bundle, = self.bundles()
        self.assertEqual(bundle['row_action'], 'surf')
        self.assertEqual(bundle['column_action'], 'switch blissey')
        self.assertEqual(bundle['branch_index'], 0)
        self.assertEqual(bundle['generating_pre_state'], 'original-root')
        self.assertEqual(bundle['generating_turn'], 3)

    def test_batch_context_has_no_stale_last_leaf_address_or_world(self):
        from pokezero.shallow_exact_native import NativeExactBackend
        backend = object.__new__(NativeExactBackend)
        keys = ('world_spec','world_ordinal','generating_pre_state','generating_turn',
                'row_action','column_action','branch_index')
        with decision_scope(self.q, self.context, self.rng):
            current().update(**{k:'last-leaf' for k in keys})
            backend._capture_batch_context()
            record_refusal('crate', codes=['crate.authored'])
        bundle, = self.bundles()
        for key in keys:
            self.assertEqual(bundle[key]['availability'], 'unavailable')

    def test_malformed_native_capture_fails_closed_instead_of_generic_refusal(self):
        error = ValueError("native failure")
        error.pokezero_failure_witness = "not JSON"
        with self.assertRaises(CaptureFailed):
            with decision_scope(self.q, self.context, self.rng):
                raise error
        self.assertEqual(self.q.failures, 1)
        self.assertEqual(self.bundles(), [])

    def test_fixture_gate_verifies_artifacts_class_root_build_and_regression_exit(self):
        root = dict(battle_id="captured-root", seed=17)
        bundle = self.q.write("bundle", dict(root=root, reason_codes=["renderer.authored"]))
        diagnostic = self.q.write("receipt", dict(root=root, status="REFUSAL_RECOVERED",
            refusal_codes=["renderer.authored"], scientific_writes=0, actions_selected=0, games_scored=0))
        fixture = self.q.write("fixture", dict(root=root, bundle_sha256=bundle["sha256"],
            diagnostic_receipt_sha256=diagnostic["sha256"]))
        verification = self.q.write("verified", dict(build_fingerprint="build", exit_code=0,
            reason_codes=["renderer.authored"], test="authored regression", fixture_sha256=fixture["sha256"]))
        row = dict(reason_codes=["renderer.authored"], bundle=bundle, diagnostic_receipt=diagnostic,
                   fixture=fixture, verification=verification)
        def gate(row, code="renderer.authored", build="build"):
            return require_fixture_coverage([code], {"fixtures":[row]},
                build_fingerprint=build, evidence_root=self.path)
        gate(row)
        for changed in ({**row, "bundle":{**bundle, "sha256":"0"*64}},
                        {**row, "verification":self.q.write("bad-test", dict(build_fingerprint="build",
                            exit_code=1, reason_codes=["renderer.authored"], test="failed",
                            fixture_sha256=fixture["sha256"]))}):
            with self.assertRaisesRegex(ValueError, "fixture gate"):
                gate(changed)
        with self.assertRaisesRegex(ValueError, "fixture gate"):
            gate(row, build="different-build")
        with self.assertRaisesRegex(ValueError, "fixture gate"):
            gate({**row, "reason_codes":["renderer.missing"]}, code="renderer.missing")

    def test_inventory_must_bind_actual_closed_run_diagnostics_before_fixture_gate(self):
        code = "renderer.authored"
        root = dict(battle_id="b", seed=17)
        bundle = self.q.write("bundle", dict(root=root, reason_codes=[code]))
        terminal = self.q.write("source-terminal", dict(run_id="r1", status="CLOSED", exit_code=1))
        diagnostic = self.q.write("diagnostic", dict(root=root, status="REFUSAL_RECOVERED",
            build_fingerprint="old-diagnostic-build", reason_codes=[code],
            source_terminal_sha256=terminal["sha256"], source_bundle_sha256=bundle["sha256"],
            scientific_writes=0, actions_selected=0, games_scored=0))
        source = dict(run_id="r1", reason_codes=[code], terminal=terminal,
                      diagnostic_receipt=diagnostic, bundle=bundle)
        fixture = self.q.write("fixture", dict(root=root, bundle_sha256=bundle["sha256"],
            diagnostic_receipt_sha256=diagnostic["sha256"]))
        verification = self.q.write("verified", dict(build_fingerprint="new-runtime", exit_code=0,
            reason_codes=[code], test="authored regression", fixture_sha256=fixture["sha256"]))
        fixture_entry = dict(reason_codes=[code], bundle=bundle, diagnostic_receipt=diagnostic,
                             fixture=fixture, verification=verification)
        def manifest_for(source, fixtures=()):
            inventory = self.q.write("inventory", dict(schema="pokezero.observed-refusal-inventory.v1",
                path="p2-informed-one-ply", reason_codes=[code], source_runs=[source]))
            admission = self.q.write("admission", dict(schema="pokezero.capture-admission.v1",
                build_fingerprint="new-runtime", observed_reason_codes=[code], observed_inventory=inventory,
                fixtures=list(fixtures)))
            return self.path / admission["file"]
        # Inventory qualification is distinct from fixture qualification. It
        # does NOT qualify an implementation just because a failure is known.
        with self.assertRaisesRegex(ValueError, "fixture gate"):
            require_capture_admission(manifest_for(source), build={"fingerprint":"new-runtime"})
        accepted = require_capture_admission(manifest_for(source, [fixture_entry]),
                                             build={"fingerprint":"new-runtime"})
        self.assertEqual(accepted["reason_codes"], [code])
        for mutation in ({**source, "run_id":"invented"},
                         {**source, "reason_codes":["renderer.invented"]},
                         {**source, "terminal":self.q.write("false-clean-exit",
                             dict(run_id="r1", status="CLOSED", exit_code=0))}):
            with self.assertRaisesRegex(ValueError, "observed inventory"):
                require_capture_admission(manifest_for(mutation), build={"fingerprint":"new-runtime"})
        successful_terminal = self.q.write("successful-source-terminal", dict(run_id="r1", status="CLOSED", exit_code=0))
        successful_diagnostic = self.q.write("successful-source-diagnostic", {
            **json.loads((self.path / diagnostic['file']).read_text()),
            "source_terminal_sha256":successful_terminal['sha256']})
        # A successfully completed incumbent run can still contain real
        # dropped-world refusals. Its inventory is admissible, but a fixture
        # for the newly linked receipt is still required.
        with self.assertRaisesRegex(ValueError, "fixture gate"):
            require_capture_admission(manifest_for({**source, "terminal":successful_terminal,
                "diagnostic_receipt":successful_diagnostic}), build={"fingerprint":"new-runtime"})
        with self.assertRaisesRegex(ValueError, "requires"):
            require_capture_admission(None, build={"fingerprint":"new-runtime"})


class FirstEngineCaptureAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.q = Quarantine(self.root, build={"fingerprint": "runtime"})
        self.history = self.root / "history"
        self.current = self.history / "current"
        self.current.mkdir(parents=True)
        self.legacy_source = self.q.write("legacy-source", dict(
            authored_fixture=True, fallback_occurrences=3, canonical_reason_codes=None))

    def manifest(self, *, audit_changes=None, inventory_changes=None,
                 manifest_changes=None, legacy_changes=None):
        legacy = self.q.write("legacy", dict({
            "schema": "pokezero.legacy-refusal-history.v1", "path": "engine-mcts",
            "capture_completeness": "unavailable", "sources": [self.legacy_source],
        }, **(legacy_changes or {})))
        audit = self.q.write("audit", dict({
            "schema": "pokezero.engine-mcts-capture-history.v1", "path": "engine-mcts",
            "adoption_boundary": "authored fixture: first capture-enabled roster",
            "audited_locations": ["history"], "capture_history_root": "history",
            "prospective_capture_directory": "history/current", "legacy_history": legacy,
        }, **(audit_changes or {})))
        inventory = self.q.write("inventory", dict({
            "schema": "pokezero.observed-refusal-inventory.v1", "path": "engine-mcts",
            "reason_codes": [], "source_runs": [], "capture_history_audit": audit,
        }, **(inventory_changes or {})))
        link = self.q.write("admission", dict({
            "schema": "pokezero.capture-admission.v1", "build_fingerprint": "runtime",
            "admission_mode": "first-capture-enabled-engine-mcts",
            "observed_reason_codes": [], "observed_inventory": inventory, "fixtures": [],
        }, **(manifest_changes or {})))
        return self.root / link["file"]

    def admit(self, path, **kwargs):
        return require_capture_admission(path, build={"fingerprint": "runtime"},
            scientific_path="engine-mcts", capture_directory=self.current, **kwargs)

    def test_first_engine_admission_preserves_nonzero_legacy_disclosure(self):
        admitted = self.admit(self.manifest())
        self.assertEqual(admitted["reason_codes"], [])
        source = json.loads((self.root / self.legacy_source["file"]).read_text())
        self.assertEqual(source["fallback_occurrences"], 3)
        self.assertIsNone(source["canonical_reason_codes"])

    def test_p2_and_unspecified_empty_histories_still_refuse(self):
        path = self.manifest()
        with self.assertRaises(ValueError):
            require_capture_admission(path, build={"fingerprint": "runtime"})
        with self.assertRaises(ValueError):
            self.admit(self.manifest(manifest_changes={"admission_mode": None}))

    def test_first_mode_is_not_a_generic_new_path_bypass(self):
        with self.assertRaises(ValueError):
            require_capture_admission(self.manifest(), build={"fingerprint": "runtime"},
                scientific_path="different-path", capture_directory=self.current)

    def test_malformed_or_unbound_audits_refuse(self):
        mutations = ({"adoption_boundary": ""}, {"audited_locations": []},
            {"audited_locations": [None]}, {"path": "p2-informed-one-ply"},
            {"capture_history_root": ".."}, {"capture_history_root": "missing"},
            {"prospective_capture_directory": "history"},
            {"prospective_capture_directory": "history/different"})
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.admit(self.manifest(audit_changes=change))
        with self.assertRaises(ValueError):
            require_capture_admission(self.manifest(), build={"fingerprint": "runtime"},
                scientific_path="engine-mcts")

    def test_malformed_inventory_lists_and_fixture_types_refuse(self):
        for change in ({"source_runs": {}}, {"reason_codes": {}},
                       {"reason_codes": ["omitted.class"]}, {"path": "different"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.admit(self.manifest(inventory_changes=change))
        with self.assertRaises(ValueError):
            self.admit(self.manifest(manifest_changes={"fixtures": {}}))

    def test_legacy_evidence_cannot_disappear_or_be_called_complete(self):
        for change in ({"sources": []}, {"capture_completeness": "complete"},
                       {"path": "different"},
                       {"sources": [{**self.legacy_source, "sha256": "0" * 64}]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.admit(self.manifest(legacy_changes=change))

    def test_actual_prior_capture_cannot_be_omitted_or_reset_by_new_build(self):
        previous = Quarantine(self.history / "previous", build={"fingerprint": "old"})
        previous.write("failure", dict(root={"seed": 17}, reason_codes=["crate.actual"]))
        with self.assertRaisesRegex(ValueError, "unrepresented refusal"):
            self.admit(self.manifest())
        path = self.manifest(manifest_changes={"build_fingerprint": "new"})
        with self.assertRaisesRegex(ValueError, "unrepresented refusal"):
            require_capture_admission(path, build={"fingerprint": "new"},
                scientific_path="engine-mcts", capture_directory=self.current)

    def test_current_roster_captures_gate_the_next_roster_not_the_next_game(self):
        path = self.manifest()
        self.admit(path)
        active = Quarantine(self.current, build={"fingerprint": "runtime"})
        active.write("failure", dict(root={"seed": 18}, reason_codes=["crate.new"]))
        self.admit(path)  # Same frozen roster may finish with declared fallbacks.
        next_directory = self.history / "next"
        next_directory.mkdir()
        next_path = self.manifest(audit_changes={"prospective_capture_directory": "history/next"})
        with self.assertRaisesRegex(ValueError, "unrepresented refusal"):
            require_capture_admission(next_path, build={"fingerprint": "runtime"},
                scientific_path="engine-mcts", capture_directory=next_directory)

    def test_linked_prior_history_cannot_be_silently_skipped(self):
        archive = self.root / "archived-history"
        previous = Quarantine(archive, build={"fingerprint": "old"})
        previous.write("failure", dict(root={"seed": 19}, reason_codes=["crate.actual"]))
        (self.history / "previous").symlink_to(archive, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "uninspected directory link"):
            self.admit(self.manifest())

    def test_build_binding_remains_required_for_empty_first_admission(self):
        with self.assertRaisesRegex(ValueError, "actual runtime build"):
            require_capture_admission(self.manifest(), build={"fingerprint": "different"},
                scientific_path="engine-mcts", capture_directory=self.current)

    def test_known_codes_still_need_real_closed_run_and_fixture_links(self):
        previous = Quarantine(self.history / "previous", build={"fingerprint": "old"})
        previous.write("failure", dict(root={"seed": 17}, reason_codes=["crate.actual"]))
        path = self.manifest(inventory_changes={
            "reason_codes": ["crate.actual"],
            "source_runs": [{"run_id": "missing-links", "terminal": {}}]},
            manifest_changes={"observed_reason_codes": ["crate.actual"]})
        with self.assertRaises(ValueError):
            self.admit(path)


if __name__ == "__main__":
    unittest.main()
