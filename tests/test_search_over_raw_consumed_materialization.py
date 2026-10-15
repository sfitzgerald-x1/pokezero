"""Owned disposable process joins, authored native/bridge data, no science."""
from copy import deepcopy
from dataclasses import dataclass, replace, FrozenInstanceError
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_consumed_materialization import (
    join_consumed_materializations, join_owned_ancestry_roots, join_owned_raw_replays, _observe_raw_replay_boundaries,
    OwnedNativeReplayPlan, plan_owned_native_replays, build_owned_native_replay_jobs,
    reconcile_owned_replay_diagnostic,
    reconcile_archived_exploration_diagnostic,
)
from pokezero.mcts_eval.search_over_raw_fixed_work_run import FixedWorkRunValidator
from pokezero.mcts_eval.search_over_raw_materialization_run import MaterializationRunValidator
import test_search_over_raw_fixed_work_run as fixed_work_fixtures
import test_search_over_raw_materialization_run as materialization_fixtures
from test_search_over_raw_supervision import INPUT_BINDING
from test_engine_world import _payload, _dex
from test_search_over_raw_ancestry import AuthoredAncestryJob, CONTRACT
from test_search_over_raw_construction import map_fixture
from test_search_over_raw_native_boundary import report
from pokezero.mcts_eval.search_over_raw_boundary_run import (
    OwnedRawBoundaryJob, RawBoundaryRunValidator, supervise_raw_boundary_run, raw_root_json,
    OwnedNativePathBoundaryJob, NativePathBoundaryRunValidator, supervise_native_path_boundary_run,
)
import test_search_over_raw_showdown_boundary as showdown_fixtures
import pickle
import shutil


@dataclass(frozen=True)
class AuthoredArchivedNativeJob:
    """Disposable process, actual request/capture hooks, authored native result."""
    configuration: object
    context: object
    statistics_root_id: str
    seed: int
    public_witness: dict
    capture_native_public_fold: bool = False
    synthetic_process_fixture = True

    def __call__(self, ownership, phase):
        from types import SimpleNamespace
        from pokezero.mcts_eval.search_over_raw_world_inputs import NativeWorldInputs, NativeInputHandle
        from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
        from test_search_over_raw_world_inputs import arguments
        from test_search_over_raw_native_atomic_work import native_report
        import test_search_over_raw_construction as construction
        import test_search_over_raw_ancestry as ancestry
        phase("startup")
        context = self.context
        engine = context.public_materialization_state.belief_engine.clone()
        engine.set_source = SimpleNamespace(metadata=SimpleNamespace(
            source_hash=self.public_witness["consumed_binding"]["set_source_hash"]))
        context = replace(context, public_materialization_state=replace(
            context.public_materialization_state, belief_engine=engine))
        phase("selection")
        binding = self.public_witness["consumed_binding"]
        recorder = NativeWorldInputs(binding, 16, deadline_limited=False,
            record_atomic_work=True, record_construction=True, expected_public_request=self.public_witness,
            record_public_fold=self.capture_native_public_fold)
        with patch.object(construction, "CONTEXT", context):
            for _ in range(4): world, _ = construction.construct_actual_python(recorder)
        args = arguments(iterations=64); args[5] = json.dumps(dict(world.party_species, turn=3))
        NativeInputHandle(SimpleNamespace(search_batched_multi_encoded=lambda *a,**k:json.dumps(native_report(64))),
            recorder).search_batched_multi_encoded(*args)
        manifest = recorder.finish(dict(total_iterations=64), attempted=4, constructed=4, searched=4)
        with patch.object(ancestry, "INPUT_BINDING", binding):
            exports = [ancestry.export_fixture(call,i) for i,call in enumerate(manifest["calls"])]
        contract = FixedWorkDiagnosticContract()
        selected = dict(root_id=self.statistics_root_id+":"+self.configuration.identity,
            statistics_root_id=self.statistics_root_id, configuration_sha256=self.configuration.identity,
            selection_seed=self.seed, runtime_sha256="a"*64,
            fixed_work_diagnostic=dict(contract=contract.receipt(), accepted_actual=64),
            evidence=dict(total_iterations=64, consumed_world_inputs=manifest, native_searched_ancestry=exports))
        phase("cleanup")
        return selected


class ArchivedDiagnosticAttachmentTests(unittest.TestCase):
    def fixture(self, record_public_fold=False):
        from test_search_over_raw_archive import ExplorationRootCrosswalkTests
        from pokezero.mcts_eval.search_over_raw_archive import prepare_archived_fixed_work_job
        from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
        from pokezero.mcts_eval.search_over_raw_resources import derive_fixed_work_accounting
        from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
        from pokezero.mcts_eval.search_over_raw import SearchConfiguration
        archive = ExplorationRootCrosswalkTests(); archive.setUp(); self.addCleanup(archive.doCleanups)
        owner = fixed_work_fixtures.FixedWorkRunReaderTests(); owner.setUp(); self.addCleanup(owner.doCleanups)
        cfg = SearchConfiguration("incumbent", seconds=1.)
        prepared, bound = prepare_archived_fixed_work_job(archive.archive, archive.source,
            contract=archive.contract, root_id=archive.root_id, configuration=cfg,
            checkpoint_contract=SimpleNamespace(showdown_source_sha256="a"*64),
            showdown_root=str(owner.showdown), runtime_sha256="a"*64)
        witness = bound["public_request_witness"]
        job = AuthoredArchivedNativeJob(cfg, prepared.context, archive.root_id, archive.context.seed, witness,
            capture_native_public_fold=record_public_fold)
        directory = owner.root / "archived-native"
        terminal = _supervise_owned_job(job, directory=directory, binding_sha256="a"*64,
            contract=FixedWorkDiagnosticContract(), kind="fixed-work", work_phase="selection",
            verify_admission=lambda:True, validate_result=lambda selected:derive_fixed_work_accounting(
                selected["evidence"], arm="incumbent", binding=witness["consumed_binding"],
                statistics_root_id=archive.root_id, leaf="model",
                native_public_fold_required=record_public_fold,
                expected_public_request_sha256=witness["public_request_sha256"]))
        self.assertEqual(terminal["supervisor_exit_code"],0,terminal)
        reader = FixedWorkRunValidator(directory=directory, job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256="a"*64, configuration=cfg, root_id=archive.root_id+":"+cfg.identity,
            statistics_root_id=archive.root_id, selection_seed=archive.context.seed, runtime_sha256="a"*64,
            consumed_binding=witness["consumed_binding"], public_request_sha256=witness["public_request_sha256"],
            file_hashes=owner.files, checkpoint_path=str(owner.checkpoint), checkpoint_sha256=owner.files[str(owner.checkpoint)],
            showdown_root=str(owner.showdown), allow_synthetic=True,native_public_fold_required=record_public_fold)
        return archive, terminal["result"], reader

    def attach(self, fixture, extractor=None):
        archive, selected, reader = fixture
        return reconcile_archived_exploration_diagnostic(selected, [], [], archive=archive.archive,
            source=archive.source, contract=archive.contract, root_id=archive.root_id,
            native_validator=reader, ancestry_contract=CONTRACT,
            native_extract=extractor or OwnedAncestryRootJoinTests.extractor)

    def test_actual_archive_sampling_and_completed_owner_attach_without_scoring_or_qualification(self):
        fixture = self.fixture(); result = self.attach(fixture)
        self.assertTrue(result["root_to_panel_attachment_validated"])
        self.assertTrue(result["synthetic_scope"])
        self.assertEqual(result["costed_unique_jobs"],1)
        self.assertEqual(result["deployment_accepted_units_added"],0)
        self.assertEqual(len(result["archive_crosswalk"]["missing_source_root_ids"]),6)
        self.assertEqual(result["owned_replay_join"]["required_construction_sample_records"],8)
        self.assertEqual(result["owned_replay_join"]["replay_ordinal_records"],0)
        for key in ("source_policy_validated", "source_terminal_validated", "source_freshness_validated",
                    "runtime_qualified", "branch_transition_validated", "qualification_changed",
                    "scored_root_intervals_changed", "scientific_collection_authorized", "holdout_authorized"):
            self.assertFalse(result[key])

    def test_incomplete_or_foreign_prospective_full_request_cannot_attach(self):
        fixture = self.fixture(); reader = fixture[2]
        for field, value in (("public_request_sha256",None), ("public_request_sha256","d"*64),
                             ("statistics_root_id","same-root"), ("seed",1000), ("seed",False)):
            before = getattr(reader,field); setattr(reader,field,value)
            with self.assertRaisesRegex(ValueError,"prospective owning"):
                self.attach(fixture)
            setattr(reader,field,before)
        reader.allow_synthetic = False
        with self.assertRaisesRegex(ValueError,"uncleared owning"):
            self.attach(fixture)

    def test_late_archive_mutation_after_final_owning_callbacks_cannot_clear(self):
        import pokezero.mcts_eval.search_over_raw_consumed_materialization as coordinator
        fixture = self.fixture(); archive = fixture[0]; calls = []
        actual = coordinator.reconcile_owned_replay_diagnostic
        def completed(*args, **kwargs):
            result = actual(*args, **kwargs)
            calls.append(result)
            if len(calls) == 2:
                archive.archive._private[0].audit_only["canary"] = "late-changed-boundary"
            return result
        with patch.object(coordinator, "reconcile_owned_replay_diagnostic", side_effect=completed):
            with self.assertRaisesRegex(ValueError,"archived root changed after final"):
                self.attach(fixture)

    def test_final_archive_callback_cannot_mutate_original_owner_receipt_scope_readout_or_model(self):
        class SubstitutedReader(FixedWorkRunValidator):pass
        for target in ("receipt","scope","readout","model","class"):
            with self.subTest(target=target):
                fixture=self.fixture();archive,selected,reader=fixture
                original=archive.archive.selected;calls=0
                def last(root):
                    nonlocal calls
                    calls+=1;value=original(root)
                    if calls==3:
                        if target=="receipt":
                            path=reader.directory/"terminal.json";path.write_bytes(path.read_bytes()+b" ")
                        if target=="scope":reader.allow_synthetic=False
                        if target=="class":reader.__class__=SubstitutedReader
                        if target=="readout":reader.last_validation["costs"]["complete_workflow_cost"]=True
                        if target=="model":
                            path=Path(reader.checkpoint_path);path.write_bytes(path.read_bytes()+b"late mutation")
                    return value  # The actual archived root remains unchanged.
                with patch.object(archive.archive,"selected",side_effect=last):
                    with self.assertRaisesRegex(ValueError,"original diagnostic owner|original diagnostic owning readouts"):
                        self.attach(fixture)
                self.assertEqual(calls,3)


@dataclass(frozen=True)
class DifferentOverlayJob(materialization_fixtures.SyntheticMaterializationJob):
    """Different overlay is submitted at the actual Python constructor seam."""
    def input_binding(self):
        payload = _payload(_dex()); payload["turn"] = 77
        # Bypass only the parent fixture's fixed authored payload substitution.
        from pokezero.mcts_eval.search_over_raw_materialization_run import OwnedMaterializationJob
        with patch("pokezero.local_showdown._public_materialization_payload", return_value=payload):
            return OwnedMaterializationJob.input_binding(self)

    def __call__(self, ownership, phase):
        payload = _payload(_dex()); payload["turn"] = 77
        from pokezero.mcts_eval.search_over_raw_materialization_run import OwnedMaterializationJob
        with patch("pokezero.local_showdown._public_materialization_payload", return_value=payload):
            return OwnedMaterializationJob.__call__(self, ownership, phase)


class OwnedConstructorJoinTests(unittest.TestCase):
    def fixtures(self, *, different=False, allow_native=True, allow_reference=True):
        native = fixed_work_fixtures.FixedWorkRunReaderTests(); native.setUp(); self.addCleanup(native.doCleanups)
        selected, inputs = native.run_fixture()
        reader = FixedWorkRunValidator(**inputs, allow_synthetic=allow_native)
        material = materialization_fixtures.MaterializationRunTests(); material.setUp(); self.addCleanup(material.doCleanups)
        job = replace(material.job(), root_binding=deepcopy(INPUT_BINDING))
        if different:
            job = DifferentOverlayJob(job.config, job.state, job.override, job.seed, job.root_binding, job.runtime)
        job, terminal = material.run_job(job)
        self.assertEqual(terminal["supervisor_exit_code"], 0, terminal)
        material_reader = material.reader(job, allow_synthetic=allow_reference)
        rows = [dict(construction_ordinal=0, result=terminal["result"], validator=material_reader)]
        return selected, rows, reader

    def test_owned_inputs_join_actual_raw_root_without_dropping_collapsed_or_unmaterialized_ordinals(self):
        native, reference, reader = self.fixtures()
        before = digest(dict(native=native, result=reference[0]["result"]))
        result = join_consumed_materializations(native, reference, native_validator=reader)
        self.assertEqual(result["required_construction_ordinals"], 4)
        self.assertEqual(len(result["rows"]), 4)
        self.assertEqual(result["actual_native_calls"], 1)
        self.assertEqual(result["matched_constructor_ordinals"], 1)
        row = result["rows"][0]
        self.assertEqual(row["status"], "MATCHED_CONSTRUCTOR_INPUTS_NOT_FULL_WORLD")
        self.assertEqual(row["collapsed_construction_ordinals"], [0,1,2,3])
        self.assertEqual(row["native_call_multiplicity"], 4)
        self.assertEqual(row["produced_bridge_snapshot_sha256"],
            digest(reference[0]["result"]["produced_bridge_snapshot"]))
        self.assertEqual([row["status"] for row in result["rows"][1:]], ["UNMATERIALIZED_UNCERTAIN"]*3)
        self.assertEqual(digest(dict(native=native, result=reference[0]["result"])), before)
        self.assertTrue(result["synthetic_scope"])
        for flag in ("private_state_exported", "native_ancestry_coverage_validated", "party_mapping_validated",
                "branch_transition_validated", "full_world_equivalence", "resource_cost_complete",
                "runtime_qualified", "scientific_collection_authorized"):
            self.assertFalse(result[flag])

    def test_actual_differing_overlay_is_input_mismatch_not_full_equivalence_gate(self):
        native, reference, reader = self.fixtures(different=True)
        result = join_consumed_materializations(native, reference, native_validator=reader)
        self.assertEqual(result["matched_constructor_ordinals"], 0)
        self.assertEqual(result["rows"][0]["status"], "INPUT_MISMATCH")
        self.assertEqual(result["rows"][0]["mismatched_inputs"], ["public_overlay"])
        self.assertTrue(result["rows"][0]["native_run_validated"])
        self.assertTrue(result["rows"][0]["reference_run_validated"])

    def test_default_synthetic_reader_scope_never_promotes_constructor_join(self):
        for allow_native, allow_reference in ((False,True), (True,False)):
            with self.subTest(native=allow_native, reference=allow_reference):
                native, reference, reader = self.fixtures(allow_native=allow_native, allow_reference=allow_reference)
                result = join_consumed_materializations(native, reference, native_validator=reader)
                self.assertEqual(result["matched_constructor_ordinals"], 0)
                self.assertEqual(result["rows"][0]["status"], "UNVALIDATED_SCOPE")

    def test_missing_duplicate_unexpected_and_substituted_roots_are_not_repaired(self):
        native, reference, reader = self.fixtures()
        empty = join_consumed_materializations(native, [], native_validator=reader)
        self.assertEqual(empty["matched_constructor_ordinals"], 0)
        self.assertTrue(all(row["status"] == "UNMATERIALIZED_UNCERTAIN" for row in empty["rows"]))
        for rows in ([reference[0],reference[0]], [dict(reference[0],construction_ordinal=True)],
                [dict(reference[0],construction_ordinal=4)]):
            with self.assertRaises(ValueError): join_consumed_materializations(native, rows, native_validator=reader)
        altered = deepcopy(reference[0]["result"])
        altered["produced_bridge_snapshot"]["battle"]["turn"] = 88
        altered["produced_bridge_snapshot_sha256"] = digest(altered["produced_bridge_snapshot"])
        with self.assertRaises(ValueError): join_consumed_materializations(native,
            [dict(reference[0],result=altered)], native_validator=reader)

    def test_numeric_alias_wrappers_cannot_replace_validated_inputs_or_actual_raw_root(self):
        native, reference, reader = self.fixtures()
        original = reference[0]["result"]
        for mode in ("overlay", "snapshot"):
            altered = deepcopy(original)
            if mode == "overlay":
                altered["materialization_inputs"]["materialize"]["publicState"]["weatherFromAbility"] = 0
                self.assertEqual(altered["materialization_inputs"], original["materialization_inputs"])
                self.assertNotEqual(digest(altered["materialization_inputs"]), digest(original["materialization_inputs"]))
            else:
                altered["produced_bridge_snapshot"]["battle"]["ended"] = 0
                altered["produced_bridge_snapshot_sha256"] = digest(altered["produced_bridge_snapshot"])
                self.assertEqual(altered["produced_bridge_snapshot"], original["produced_bridge_snapshot"])
                self.assertNotEqual(altered["produced_bridge_snapshot_sha256"], original["produced_bridge_snapshot_sha256"])
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                join_consumed_materializations(native, [dict(reference[0],result=altered)], native_validator=reader)

    def test_explicit_collapsed_aliases_do_not_inflate_actual_reference_run_count(self):
        native, reference, reader = self.fixtures()
        result = join_consumed_materializations(native,
            [reference[0], dict(reference[0],construction_ordinal=1)], native_validator=reader)
        self.assertEqual(result["materialization_ordinal_records"], 2)
        self.assertEqual(result["materialization_runs_recorded"], 1)
        self.assertEqual(result["matched_constructor_ordinals"], 2)

    def test_controller_runtime_binds_actual_join_reader(self):
        from pathlib import Path
        from pokezero.mcts_eval.search_over_raw_reconciled import controller_input_hashes
        package = Path(__file__).resolve().parents[1] / "src/pokezero/mcts_eval"
        hashes = controller_input_hashes()
        for name in ("search_over_raw_fixed_work_run.py", "search_over_raw_consumed_materialization.py"):
            path = package / name
            self.assertEqual(hashes[str(path)], hashlib.sha256(path.read_bytes()).hexdigest())


class OwnedAncestryRootJoinTests(unittest.TestCase):
    """Actual disposable owners, authored root/path/extractor, never physics."""
    fixtures = OwnedConstructorJoinTests.fixtures
    def ancestry_fixtures(self, **kwargs):
        with patch.object(fixed_work_fixtures, "AuthoredFixedWorkJob", AuthoredAncestryJob):
            return self.fixtures(**kwargs)

    @staticmethod
    def extractor(state, root):
        value = report(state, root)
        _, native, _ = map_fixture()
        for seat in ("p1", "p2"):
            side = value[seat]
            side["party_identities"] = deepcopy(native[seat]["party_identities"])
            side["active_identity"] = deepcopy(side["party_identities"][0])
            side["active_moves"][0]["engine_id"] = "move"
            for key in ("engine_options", "search_arms", "effective_legal_choices"):
                side[key] = [dict(kind="move", slot=0, engine_id="move")]
        # This authored export is an exact UTF-8 witness, not just parsed JSON.
        # report() includes set-derived flags: canonicalize here so a separate
        # owning reader reproduces the same bytes under another hash seed.
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def join(self, native, reference, reader, extractor=None):
        return join_owned_ancestry_roots(native, reference, native_validator=reader,
            ancestry_contract=CONTRACT, native_extract=extractor or self.extractor)

    def test_join_owned_roots_and_every_sample_boundary_without_multiplicity_inflation(self):
        native, reference, reader = self.ancestry_fixtures()
        calls = []
        def extract(state, root):
            calls.append((state, root)); return self.extractor(state, root)
        result = self.join(native, reference, reader, extract)
        self.assertEqual(result["owned_unique_party_mapping_joins"], 1)
        self.assertEqual(result["rows"][0]["mapping"]["native_to_original"], dict(p1=[1,0], p2=[0,1]))
        self.assertEqual(result["rows"][0]["status"], "OWNED_ROOT_UNIQUE_SET_JOIN_NOT_TRANSITION")
        self.assertEqual((result["actual_native_calls"], result["actual_model_evaluations"],
            result["actual_sampled_events"], result["construction_sample_records"]), (1,2,2,8))
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0], ("complete-state", True))
        self.assertEqual(set(calls[1:]), {("branch-0", False), ("branch-1", False)})
        for event in result["rows"][0]["sampled_events"]:
            self.assertEqual(event["required_boundaries"], 2)
            self.assertEqual(len(event["boundaries"]), 2)
            self.assertIsNone(event["boundaries"][1]["native"]["branch_transition"])
            for boundary in event["boundaries"]:
                state = "complete-state" if boundary["ordinal"] == 0 else "branch-" + str(event["evaluation_ordinal"])
                self.assertEqual(boundary["native_export_utf8_sha256"],
                    hashlib.sha256(self.extractor(state, boundary["ordinal"] == 0).encode()).hexdigest())
        for row in result["rows"][1:]:
            self.assertEqual(row["status"], "UNMATERIALIZED_UNCERTAIN")
            self.assertEqual(len(row["sampled_events"]), 2)
            self.assertTrue(all(e["boundaries"] is None for e in row["sampled_events"]))
        for flag in ("native_extractor_runtime_qualified", "party_mapping_validated", "branch_transition_validated",
                "independent_terminal_labels", "full_world_equivalence", "resource_cost_complete",
                "runtime_qualified", "scientific_collection_authorized"):
            self.assertFalse(result[flag])

    def test_unvalidated_or_unmaterialized_rows_never_call_extractor(self):
        for mode in ("scope", "mismatch", "missing"):
            native, reference, reader = self.ancestry_fixtures(allow_native=mode != "scope", different=mode == "mismatch")
            if mode == "missing": reference = []
            def forbidden(*args): self.fail("unqualified row invoked extraction")
            result = self.join(native, reference, reader, forbidden)
            self.assertEqual(result["owned_unique_party_mapping_joins"], 0)
            self.assertEqual(result["actual_sampled_events"], 2)
            self.assertEqual(result["construction_sample_records"], 8)

    def test_incomplete_identity_is_uncertain_without_dropping_events(self):
        def missing_identity():
            root, native, construction = map_fixture()
            root["battle"]["sides"][0]["pokemon"][1]["set"].pop("nature")
            return root, native, construction
        with patch("test_search_over_raw_construction.map_fixture", missing_identity):
            native, reference, reader = self.ancestry_fixtures()
        result = self.join(native, reference, reader)
        self.assertEqual(result["rows"][0]["status"], "PARTY_IDENTITY_UNCERTAIN")
        self.assertIsNone(result["rows"][0]["mapping"])
        self.assertEqual(len(result["rows"][0]["sampled_events"]), 2)
        self.assertTrue(all(len(e["boundaries"]) == 2 for e in result["rows"][0]["sampled_events"]))

    def test_extractor_substitutions_and_mutating_input_callbacks_fail_closed(self):
        native, reference, reader = self.ancestry_fixtures()
        for mode in ("state", "duplicate_json", "promoted", "private_input"):
            def extract(state, root):
                raw = self.extractor(state, root)
                if mode == "duplicate_json": return raw[:-1] + ',"root":true}'
                value = json.loads(raw)
                if mode == "state": value["state_utf8_sha256"] = "c"*64
                if mode == "promoted": value["runtime_qualified"] = True
                if mode == "private_input": native["evidence"]["native_searched_ancestry"][0]["world_seed"] += 1
                return json.dumps(value)
            before = deepcopy(native)
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.join(native, reference, reader, extract)
            native.clear(); native.update(before)

    def test_current_native_species_mismatch_is_uncertain_not_a_universal_gate(self):
        native, reference, reader = self.ancestry_fixtures()
        def extract(state, root):
            value = json.loads(self.extractor(state, root))
            value["p1"]["party_identities"][1]["engine_id"] = "blissey"
            return json.dumps(value)
        result = self.join(native, reference, reader, extract)
        self.assertEqual(result["rows"][0]["status"], "PARTY_IDENTITY_UNCERTAIN")
        self.assertIn("p1:current_species_slot_mismatch", result["rows"][0]["limitations"])
        self.assertEqual(result["construction_sample_records"], 8)

    def test_owned_receipt_drift_during_extraction_is_revalidated(self):
        native, reference, reader = self.ancestry_fixtures()
        path = reference[0]["validator"].directory / "terminal.json"
        before = path.read_bytes()
        def extract(state, root):
            terminal = json.loads(before); terminal["worker_exit_code"] = True
            path.write_text(json.dumps(terminal))
            return self.extractor(state, root)
        try:
            with self.assertRaises(ValueError): self.join(native, reference, reader, extract)
        finally:
            path.write_bytes(before)


@dataclass(frozen=True)
class AuthoredRawReplayJob(OwnedRawBoundaryJob):
    synthetic_process_fixture = True
    def _components(self): return materialization_fixtures.SyntheticEnv(self.config)


@dataclass(frozen=True)
class AuthoredNativeReplayJob(OwnedNativePathBoundaryJob):
    synthetic_process_fixture = True
    def _components(self): return materialization_fixtures.SyntheticEnv(self.config)


from pokezero.mcts_eval.search_over_raw_native_prefix import OwnedNativePrefixBoundaryJob


@dataclass(frozen=True)
class AuthoredNativePrefixReplayJob(OwnedNativePrefixBoundaryJob):
    synthetic_process_fixture = True
    def _components(self): return materialization_fixtures.SyntheticEnv(self.config)


class ObservedReplayFieldsTests(unittest.TestCase):
    """Authored observations, never independent ownership or engine semantics."""
    def observe(self, root, after, commands, actions, *, frozen=None, fields=None):
        frozen = showdown_fixtures.mapping(root) if frozen is None else frozen
        native = [dict(ordinal=0, native={}), dict(ordinal=1, native=dict(action_identity=actions, **(fields or {})))]
        reference = [dict(snapshot=root, observed=dict(incoming_choices=None)),
            dict(snapshot=after, observed=dict(incoming_choices=commands))]
        return _observe_raw_replay_boundaries(native, reference, frozen, 2)

    def test_switch_command_uses_before_party_order_and_p2_identity(self):
        root = showdown_fixtures.snapshot(); after = deepcopy(root)
        side = after["battle"]["sides"][0]
        side["pokemon"].reverse(); side["team"] = "21"
        for current, mon in enumerate(side["pokemon"]): mon.update(position=current, isActive=current == 0)
        after["boundaryRequests"] = showdown_fixtures.requests(after["battle"]["sides"])
        actions = dict(p1=dict(kind="switch", slot=1, engine_id="charmander"),
            p2=dict(kind="move", slot=0, engine_id="splash"))
        result = self.observe(root, after, dict(p1="switch 2", p2="move 1"), actions)
        self.assertEqual(result[1]["observed_required_fields"]["action_identity"], "MATCH")
        self.assertEqual(result[1]["reference_projection"]["showdown"]["active_identities"]["p1"]["slot"], 1)
        # After switching, command 2 points to the opposite original slot.
        actions["p1"]["slot"] = 0
        self.assertEqual(self.observe(root, after, dict(p1="switch 2", p2="move 1"), actions)[1]
            ["observed_required_fields"]["action_identity"], "MISMATCH")

    def test_locked_move_request_index_and_terminal_arrival_are_not_post_state_catalogs(self):
        root = showdown_fixtures.snapshot(); frozen = showdown_fixtures.mapping(root)
        root["battle"]["sides"][0]["pokemon"][0]["moveSlots"].append(dict(id="tackle", pp=20, disabled=False))
        root["boundaryRequests"]["p1"]["active"][0] = dict(moves=[dict(id="tackle")], trapped=True)
        after = deepcopy(root); after["battle"]["ended"] = True; after["boundaryRequests"] = {}
        actions = dict(p1=dict(kind="move", slot=1, engine_id="tackle"),
            p2=dict(kind="move", slot=0, engine_id="splash"))
        result = self.observe(root, after, dict(p1="move 1", p2="move 1"), actions, frozen=frozen)
        self.assertEqual(result[1]["observed_required_fields"]["action_identity"], "MATCH")
        self.assertTrue(result[1]["reference_projection"]["reference_terminal"])
        actions["p2"]["engine_id"] = "wrong-p2"
        self.assertEqual(self.observe(root, after, dict(p1="move 1", p2="move 1"), actions, frozen=frozen)[1]
            ["observed_required_fields"]["action_identity"], "MISMATCH")

    def test_pp_class_does_not_erase_required_mismatches_or_claim_transition(self):
        root = showdown_fixtures.snapshot(); actions = dict(p1=dict(kind="move", slot=0, engine_id="seismictoss"),
            p2=dict(kind="move", slot=0, engine_id="splash"))
        fields = dict(hp=dict(p1=[99,100], p2=[100,100]), status=dict(p1="burn",p2="none"),
            effective_legal_choices=dict(p1=[],p2=[]), native_pp=dict(p1=[],p2=[]))
        result = self.observe(root, deepcopy(root), dict(p1="move 1",p2="move 1"), actions, fields=fields)
        self.assertEqual(result[0]["observed_required_fields"]["action_identity"], "NOT_APPLICABLE")
        for key in ("hp","status","effective_legal_choices"):
            self.assertEqual(result[1]["observed_required_fields"][key], "MISMATCH")
        self.assertEqual(result[1]["documented_differences"][0]["classification"], "engine_pp_model")
        self.assertEqual(result[1]["observed_required_fields"]["branch_transition"], "UNCHECKABLE")
        for flag in ("independently_validated","matched_causal_trajectory","branch_transition_validated",
                "runtime_qualified","scientific_collection_authorized"):
            self.assertFalse(result[1][flag])

    def test_missing_map_ambiguous_native_action_and_withheld_legality_stay_uncertain(self):
        root = showdown_fixtures.snapshot(); frozen = showdown_fixtures.mapping(root)
        native = [dict(ordinal=0,native={}), dict(ordinal=1,native=dict(action_identity=None))]
        reference = [dict(snapshot=root,observed=dict(incoming_choices=None)),
            dict(snapshot=deepcopy(root),observed=dict(incoming_choices=dict(p1="move 1",p2="move 1")))]
        for mapping, rows in ((None,reference), (frozen,None), (frozen,reference)):
            result = _observe_raw_replay_boundaries(native,rows,mapping,2)
            self.assertEqual(len(result),2)
            self.assertEqual(result[1]["observed_required_fields"]["action_identity"], "UNCHECKABLE")
        root["boundaryRequests"]["p1"]["active"][0]["maybeDisabled"] = True
        actions = dict(p1=dict(kind="move",slot=0,engine_id="seismictoss"),
            p2=dict(kind="move",slot=0,engine_id="splash"))
        result = self.observe(root,deepcopy(root),dict(p1="move 1",p2="move 1"),actions,frozen=frozen)
        self.assertEqual(result[1]["observed_required_fields"]["action_identity"], "UNCHECKABLE")
        self.assertEqual(result[1]["observed_required_fields"]["effective_legal_choices"], "UNCHECKABLE")
        with self.assertRaises(ValueError): _observe_raw_replay_boundaries(native,reference[:1],frozen,2)
        bad = deepcopy(reference); bad[1]["snapshot"]["battle"]["sides"][0]["pokemon"][0]["hp"] = True
        with self.assertRaises(ValueError): _observe_raw_replay_boundaries(native,bad,frozen,2)

    def test_wait_omission_is_observation_not_native_commitment_validation(self):
        root = showdown_fixtures.snapshot()
        root["boundaryRequests"]["p2"].pop("active"); root["boundaryRequests"]["p2"]["wait"] = True
        actions = dict(p1=dict(kind="move",slot=0,engine_id="seismictoss"),
            p2=dict(kind="none",slot=None,engine_id="No Move"))
        result = self.observe(root,deepcopy(root),dict(p1="move 1"),actions)
        self.assertEqual(result[1]["observed_required_fields"]["action_identity"], "MATCH")
        self.assertFalse(result[1]["matched_causal_trajectory"])


class OwnedRawReplayJoinTests(unittest.TestCase):
    extractor = staticmethod(OwnedAncestryRootJoinTests.extractor)

    def fixture(self, *, wrong_origin=False, wrong_roster=False, scope=True, wrong_action=False, withheld=False,
            native_driven=False, tampered_projection=False, tampered_mapping=False, skip_replay=False, with_replay_log=False):
        native = fixed_work_fixtures.FixedWorkRunReaderTests(); native.setUp(); self.addCleanup(native.doCleanups)
        with patch.object(fixed_work_fixtures, "AuthoredFixedWorkJob", AuthoredAncestryJob):
            selected, inputs = native.run_fixture()
        native_reader = FixedWorkRunValidator(**inputs, allow_synthetic=True)
        material = materialization_fixtures.MaterializationRunTests(); material.setUp(); self.addCleanup(material.doCleanups)
        for side in material.snapshot["battle"]["sides"]:
            for mon in side["pokemon"]:
                mon.update(moveSlots=[dict(id="move", pp=15, disabled=False)], hpType="Ice", hpPower=70)
        material.snapshot["boundaryRequests"] = showdown_fixtures.requests(material.snapshot["battle"]["sides"])
        if with_replay_log: material.snapshot["battle"]["log"] = []
        if withheld: material.snapshot["boundaryRequests"]["p2"]["active"][0]["maybeTrapped"] = True
        material.configure("complete")
        job, terminal = material.run_job(replace(material.job(), root_binding=deepcopy(INPUT_BINDING)))
        self.assertEqual(terminal["supervisor_exit_code"], 0, terminal)
        references = [dict(construction_ordinal=0, result=terminal["result"], validator=material.reader(job))]
        roots = join_owned_ancestry_roots(selected, references, native_validator=native_reader,
            ancestry_contract=CONTRACT, native_extract=self.extractor)
        self.planner_config = material.config
        if skip_replay: return selected, references, [], native_reader
        event = roots["rows"][0]["sampled_events"][0]
        root = terminal["result"]["produced_bridge_snapshot"]
        origin = dict(materialization_capture_sha256=digest(terminal["result"]["capture"]),
            produced_bridge_snapshot_sha256=digest(root), native_selected_sha256=digest(selected),
            native_capture_sha256="e"*64 if wrong_origin else event["capture_sha256"],
            construction_ordinal=0, evaluation_ordinal=event["evaluation_ordinal"])
        clone = dict(battleId="joined-owned-raw", cloneId="joined-path",
            snapshotJsonSha256=hashlib.sha256(raw_root_json(root).encode()).hexdigest())
        if native_driven:
            boundaries, mapping = deepcopy(event["boundaries"]), deepcopy(roots["rows"][0]["mapping"])
            if tampered_projection: boundaries[0]["native"]["hp"]["p1"][0] -= 1
            if tampered_mapping: mapping["native_to_original"]["p1"].reverse()
            replay_job = AuthoredNativeReplayJob(material.config, deepcopy(root), deepcopy(INPUT_BINDING), origin,
                deepcopy(material.runtime), clone, ["root", "after"], boundaries, mapping)
            supervise, reader_class = supervise_native_path_boundary_run, NativePathBoundaryRunValidator
        else:
            replay_job = AuthoredRawReplayJob(material.config, deepcopy(root), deepcopy(INPUT_BINDING), origin,
                deepcopy(material.runtime), clone, ["root"] if wrong_roster else ["root", "after"],
                [] if wrong_roster else [dict(p1="switch 2" if wrong_action else "move 1", p2="move 1")])
            supervise, reader_class = supervise_raw_boundary_run, RawBoundaryRunValidator
        directory = material.root / "raw-replay"
        replay = supervise(replay_job, directory=directory, binding_sha256="a"*64,
            verify_admission=lambda:True, validate_result=lambda _:True)
        self.assertEqual(replay["supervisor_exit_code"], 0, replay)
        self.last_replay_job = replay_job
        reader = reader_class(directory=directory, job_sha256=hashlib.sha256(pickle.dumps(replay_job)).hexdigest(),
            binding_sha256="a"*64, expected_inputs=replay_job.input_binding(), expected_raw_root=root,
            runtime=material.runtime, allow_synthetic=scope)
        replays = [dict(construction_ordinal=0, evaluation_ordinal=event["evaluation_ordinal"], result=replay["result"], validator=reader)]
        return selected, references, replays, native_reader

    def join(self, fixture):
        native, references, replays, reader = fixture
        return join_owned_raw_replays(native, references, replays, native_validator=reader,
            ancestry_contract=CONTRACT, native_extract=self.extractor)

    def test_independent_three_owner_join_keeps_full_constructor_sample_roster(self):
        result = self.join(self.fixture())
        self.assertEqual((result["required_construction_sample_records"], result["replay_ordinal_records"],
            result["actual_replay_runs"], result["owned_root_origin_and_coordinate_joins"]), (8,1,1,1))
        self.assertTrue(result["synthetic_scope"])
        self.assertEqual(result["required_boundary_records"], 16)
        self.assertEqual(result["required_arriving_action_records"], 8)
        self.assertEqual(result["observed_arriving_action_counts"], dict(MATCH=1, MISMATCH=0, UNCHECKABLE=7))
        matched = [r for r in result["records"] if r["owned_root_origin_and_coordinates_join"]]
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(matched[0]["reference_boundaries"]), 2)
        self.assertTrue(matched[0]["initial_roundtrip_equal"])
        self.assertTrue(matched[0]["reference_boundaries"][1]["snapshot"]["battle"]["ended"])
        self.assertEqual(sum(r["status"] == "UNREPLAYED_UNCERTAIN" for r in result["records"]), 7)
        observations = matched[0]["boundary_observations"]
        self.assertEqual(observations[0]["observed_required_fields"]["action_identity"], "NOT_APPLICABLE")
        self.assertEqual(observations[1]["observed_required_fields"]["action_identity"], "MATCH")
        # The authored reference terminal differs from the native after-state;
        # observed action equality must not erase that legal-set mismatch.
        self.assertEqual(observations[1]["observed_required_fields"]["effective_legal_choices"], "MISMATCH")
        self.assertTrue(all(b["observed_required_fields"]["branch_transition"] == "UNCHECKABLE"
            and b["independently_validated"] is False for b in observations))
        for flag in ("native_joint_actions_matched", "branch_transition_validated", "independent_terminal_labels",
                "full_world_equivalence", "resource_cost_complete", "runtime_qualified", "scientific_collection_authorized"):
            self.assertFalse(result[flag])

    def test_actual_commands_mismatch_and_withheld_requests_remain_visible(self):
        for kwargs, expected in ((dict(wrong_action=True), "MISMATCH"), (dict(withheld=True), "UNCHECKABLE")):
            result = self.join(self.fixture(**kwargs))
            self.assertEqual(result["observed_arriving_action_counts"][expected], 1 if expected == "MISMATCH" else 8)
            self.assertEqual(result["required_arriving_action_records"], 8)
            self.assertFalse(result["native_joint_actions_matched"])

    def test_native_driven_producer_reader_and_actual_native_origin_join(self):
        fixture = self.fixture(native_driven=True)
        result = self.join(fixture)
        self.assertEqual(result["native_action_driven_origin_joins"], 1)
        self.assertEqual(result["observed_arriving_action_counts"], dict(MATCH=1,MISMATCH=0,UNCHECKABLE=7))
        result_inputs = fixture[2][0]["result"]
        self.assertNotIn("choices",result_inputs)
        self.assertEqual(len(result_inputs["native_boundaries"]),2)
        reader = fixture[2][0]["validator"]
        launch = json.loads((reader.directory / "bridge-launch.json").read_text())
        self.assertEqual(launch["schema"], "pokezero.search-over-raw.native-path-boundary-bridge-launch.v1")
        self.assertFalse(result["native_joint_actions_matched"])
        self.assertFalse(result["branch_transition_validated"])

    def test_owning_native_reader_cannot_certify_substituted_native_projections_or_map(self):
        for kwargs, expected in ((dict(tampered_projection=True), "actual_native_boundary_projections"),
                (dict(tampered_mapping=True), "actual_unique_party_mapping")):
            result = self.join(self.fixture(native_driven=True, **kwargs))
            self.assertEqual(result["native_action_driven_origin_joins"],0)
            supplied = [r for r in result["records"] if r["replay_run_validated"]][0]
            self.assertEqual(supplied["status"], "ORIGIN_MISMATCH")
            self.assertIn(expected, supplied["origin_mismatches"])
            self.assertEqual(result["required_arriving_action_records"],8)

    def test_native_producer_stops_unavailable_and_unreached_full_paths_without_fallback(self):
        fixture = self.fixture(native_driven=True)
        base = self.last_replay_job
        for mode in ("unavailable", "unreached"):
            boundaries = deepcopy(base.native_boundaries)
            requests = list(base.request_ids)
            if mode == "unavailable": boundaries[1]["native"]["action_identity"]["p2"]["engine_id"] = "unavailable"
            else:
                extra = deepcopy(boundaries[1]); extra["ordinal"] = extra["binding"]["ordinal"] = 2
                boundaries.append(extra); requests.append("unreached")
            job = replace(base, native_boundaries=boundaries, request_ids=requests)
            directory = fixture[2][0]["validator"].directory.parent / ("native-failure-"+mode)
            receipt = supervise_native_path_boundary_run(job, directory=directory, binding_sha256="a"*64,
                verify_admission=lambda:True, validate_result=lambda _:True)
            self.assertEqual(receipt["status"], "OPERATIONAL_STOP", receipt)
            self.assertEqual(receipt["supervisor_exit_code"], 1)
            self.assertIsNone(receipt["result"])
            records = json.loads((directory / "bridge-transport.json").read_text())["records"]
            choices = [r for r in records if r["kind"] == "command" and r["payload"]["type"] == "choices"]
            self.assertEqual(len(choices), 0 if mode == "unavailable" else 1)
            with self.assertRaises(FileExistsError):
                supervise_native_path_boundary_run(job, directory=directory, binding_sha256="a"*64,
                    verify_admission=lambda:True, validate_result=lambda _:True)

    def test_distinct_native_contract_rejects_legacy_law_and_source_or_path_mutations(self):
        fixture = self.fixture(native_driven=True)
        job, reader = self.last_replay_job, fixture[2][0]["validator"]
        with self.assertRaises(ValueError):
            supervise_raw_boundary_run(job, directory=reader.directory.parent/"wrong-law",binding_sha256="a"*64,
                verify_admission=lambda:True,validate_result=lambda _:True)
        for mode in ("caller-choices","source","ambiguous-action","numeric-coordinate"):
            inputs = job.input_binding()
            if mode == "caller-choices": inputs["choices"] = [dict(p1="move 1",p2="move 1")]
            if mode == "source": inputs["command_derivation_sources"][next(iter(inputs["command_derivation_sources"]))] = "f"*64
            if mode == "ambiguous-action": inputs["native_boundaries"][1]["native"]["action_identity"] = None
            if mode == "numeric-coordinate": inputs["native_boundaries"][1]["binding"]["ordinal"] = 1.0
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                NativePathBoundaryRunValidator(directory=reader.directory,job_sha256=reader.job_sha,
                    binding_sha256=reader.binding,expected_inputs=inputs,expected_raw_root=job.raw_root,runtime=job.runtime)
        result = self.join(self.fixture(native_driven=True,scope=False))
        self.assertEqual(result["native_action_driven_origin_joins"],0)
        self.assertEqual(result["observed_arriving_action_counts"]["UNCHECKABLE"],8)

    def test_late_native_projection_or_map_mutation_is_rejected_after_final_extraction(self):
        fixture = self.fixture(native_driven=True)
        native, refs, replays, reader = fixture
        raw_reader = replays[0]["validator"]
        original_inputs = deepcopy(raw_reader.expected)
        for target in ("projection", "mapping"):
            calls = 0
            def late_extract(state, root):
                nonlocal calls
                calls += 1
                if calls == 4:
                    if target == "projection": raw_reader.expected["native_boundaries"][1]["native"]["action_identity"]["p1"]["slot"] = 1
                    else: raw_reader.expected["party_mapping"]["native_to_original"]["p1"].reverse()
                return self.extractor(state, root)
            try:
                with self.subTest(target=target),self.assertRaises(ValueError):
                    join_owned_raw_replays(native, refs, replays,native_validator=reader,
                        ancestry_contract=CONTRACT,native_extract=late_extract)
                self.assertGreaterEqual(calls,4)
            finally:
                raw_reader.expected = deepcopy(original_inputs)

    def test_origin_and_shortened_native_roster_are_mismatches_not_completed_paths(self):
        for kwargs, reason in ((dict(wrong_origin=True), "native_capture_sha256"),
                (dict(wrong_roster=True), "full_native_coordinate_roster")):
            result = self.join(self.fixture(**kwargs))
            supplied = [r for r in result["records"] if r["replay_run_validated"]][0]
            self.assertEqual(supplied["status"], "ORIGIN_MISMATCH")
            self.assertIn(reason, supplied["origin_mismatches"])
            self.assertEqual(result["owned_root_origin_and_coordinate_joins"], 0)
            self.assertEqual(len(result["records"]), 8)

    def test_unvalidated_missing_foreign_and_duplicate_replays_never_disappear(self):
        fixture = self.fixture(scope=False)
        result = self.join(fixture)
        self.assertEqual(result["owned_root_origin_and_coordinate_joins"], 0)
        self.assertEqual(result["records"][0]["status"], "UNVALIDATED_REPLAY_SCOPE")
        native, refs, replays, reader = fixture
        result = self.join((native, refs, [], reader))
        self.assertEqual(len(result["records"]), 8)
        self.assertTrue(all(r["status"] == "UNREPLAYED_UNCERTAIN" for r in result["records"]))
        for rows in ([replays[0], replays[0]], [dict(replays[0], evaluation_ordinal=999)]):
            with self.assertRaises(ValueError): self.join((native, refs, rows, reader))

    def test_final_extraction_cannot_mutate_replay_receipts_wrapper_or_scope(self):
        fixture = self.fixture()
        native, refs, replays, reader = fixture
        raw_reader = replays[0]["validator"]
        terminal_path = raw_reader.directory / "terminal.json"
        terminal_bytes = terminal_path.read_bytes()
        original_result = deepcopy(replays[0]["result"])
        for target in ("terminal", "wrapper", "scope"):
            calls = 0
            def late_extract(state, root):
                nonlocal calls
                calls += 1
                if calls == 4:  # First extraction of the second/final roots pass.
                    if target == "terminal":
                        terminal = json.loads(terminal_bytes)
                        terminal["worker_exit_code"] = True
                        terminal_path.write_text(json.dumps(terminal))
                    elif target == "wrapper":
                        replays[0]["result"]["root_origin"]["native_capture_sha256"] = "f"*64
                    else:
                        raw_reader.allow_synthetic = False
                return self.extractor(state, root=root)
            try:
                with self.subTest(target=target), self.assertRaises(ValueError):
                    join_owned_raw_replays(native, refs, replays, native_validator=reader,
                        ancestry_contract=CONTRACT, native_extract=late_extract)
                self.assertGreaterEqual(calls, 4)
            finally:
                terminal_path.write_bytes(terminal_bytes)
                replays[0]["result"] = deepcopy(original_result)
                raw_reader.allow_synthetic = True


class OwnedReplayDiagnosticTests(unittest.TestCase):
    fixture = OwnedRawReplayJoinTests.fixture
    extractor = staticmethod(OwnedAncestryRootJoinTests.extractor)

    def reconcile(self, fixture, extractor=None):
        selected, references, replays, reader = fixture
        return reconcile_owned_replay_diagnostic(selected, references, replays, native_validator=reader,
            ancestry_contract=CONTRACT, native_extract=extractor or self.extractor)

    def test_actual_owners_full_roster_and_separate_costs_never_admit_panel_or_score(self):
        fixture = self.fixture(native_driven=True)
        result = self.reconcile(fixture)
        joined = result["owned_replay_join"]
        self.assertEqual((joined["required_construction_sample_records"], joined["required_boundary_records"]), (8,16))
        self.assertEqual(joined["replay_ordinal_records"], 1)
        self.assertEqual(result["costed_unique_jobs"], 3)
        self.assertEqual(result["actual_fixed_work"]["accepted_units"], 64)
        self.assertEqual(result["deployment_accepted_units_added"], 0)
        self.assertEqual(result["diagnostic_selection_sha256"], digest(fixture[0]))
        owners = result["owning_jobs"]
        self.assertEqual({row["owner"]["role"] for row in owners}, {"native-fixed-work", "materialization", "replay"})
        costs = [row["owner"]["readout"]["costs"] for row in owners]
        totals = result["covered_cost_subtotals"]
        for key in ("job_cpu_seconds_including_reaped_children", "supervisor_thread_cpu_seconds"):
            self.assertEqual(totals[key], sum(row[key] for row in costs))
        self.assertEqual(totals["accumulated_job_wall_exposure_seconds"], sum(row["elapsed_wall_seconds"] for row in costs))
        self.assertIsNone(totals["campaign_elapsed_wall_seconds"])
        self.assertIsNone(totals["concurrent_tree_peak_bytes"])
        self.assertFalse(totals["descendant_close_cpu_added_again"])
        self.assertTrue(result["synthetic_scope"])
        for key in ("root_to_panel_attachment_validated", "scored_root_intervals_changed", "qualification_changed",
                "branch_transition_validated", "independent_terminal_labels", "complete_attempt_population_cost",
                "resource_cost_complete", "runtime_qualified", "scientific_collection_authorized", "holdout_authorized"):
            self.assertFalse(result[key])
        self.assertTrue(result["uncovered_cost_scope"])

    def test_collapsed_materialization_aliases_keep_denominators_without_cpu_duplication(self):
        fixture = self.fixture(native_driven=True)
        original = self.reconcile(fixture)
        fixture[1].append(dict(fixture[1][0], construction_ordinal=1))
        aliased = self.reconcile(fixture)
        self.assertEqual(aliased["costed_unique_jobs"], 3)
        self.assertEqual(aliased["covered_cost_subtotals"], original["covered_cost_subtotals"])
        owner = next(row for row in aliased["owning_jobs"] if row["owner"]["role"] == "materialization")
        self.assertEqual(owner["coordinates"], [dict(construction_ordinal=0), dict(construction_ordinal=1)])
        self.assertEqual(aliased["owned_replay_join"]["required_construction_sample_records"], 8)

    def test_missing_replays_and_uncleared_scope_are_not_zero_cost_or_missing_roster(self):
        fixture = self.fixture(native_driven=True, scope=False)
        result = self.reconcile(fixture)
        replay = next(row for row in result["owning_jobs"] if row["owner"]["role"] == "replay")
        self.assertIsNone(replay["owner"]["validation_witness"])
        self.assertEqual(result["owned_replay_join"]["owned_root_origin_and_coordinate_joins"], 0)
        selected, references, _, reader = fixture
        missing = self.reconcile((selected, references, [], reader))
        self.assertEqual(missing["costed_unique_jobs"], 2)
        self.assertEqual(missing["owned_replay_join"]["required_construction_sample_records"], 8)
        self.assertEqual(missing["owned_replay_join"]["observed_arriving_action_counts"]["UNCHECKABLE"], 8)
        self.assertFalse(missing["complete_attempt_population_cost"])

    def test_stale_or_forged_reader_readout_is_replaced_and_bad_costs_fail_closed(self):
        fixture = self.fixture(native_driven=True)
        selected, references, replays, native_reader = fixture
        for reader in (native_reader, references[0]["validator"], replays[0]["validator"]):
            reader.last_validation = {"resource_cost_complete":True, "costs":{"cpu":0}}
        result = self.reconcile(fixture)
        self.assertTrue(all(row["owner"]["readout"]["resource_cost_complete"] is False for row in result["owning_jobs"]))
        reader = replays[0]["validator"]
        path = reader.directory / "terminal.json"
        original = json.loads(path.read_text())
        for key, value in (("cpu_seconds_including_reaped_children", None), ("supervisor_thread_cpu_seconds", True),
                ("elapsed_wall_seconds", -1), ("maxrss_scope", "exact tree peak")):
            path.write_text(json.dumps(dict(original, **{key:value})))
            with self.subTest(key=key), self.assertRaises(ValueError): self.reconcile(fixture)
            self.assertIsNone(reader.last_validation)
        path.write_text(json.dumps(original))
        malformed = deepcopy(original)
        malformed["phase_costs"]["boundaries_wall_seconds"] += 1
        path.write_text(json.dumps(malformed))
        with self.assertRaises(ValueError): self.reconcile(fixture)
        self.assertIsNone(reader.last_validation)
        path.write_text(json.dumps(original))

    def test_unknown_sampler_cpu_is_not_zero_or_complete_measurement(self):
        from pokezero.mcts_eval.search_over_raw_resources import summarize_memory_trace
        fixture = self.fixture(native_driven=True)
        reader = fixture[2][0]["validator"]
        memory_path, terminal_path = reader.directory / "memory-trace.json", reader.directory / "terminal.json"
        trace, terminal = json.loads(memory_path.read_text()), json.loads(terminal_path.read_text())
        self.assertTrue(trace["samples"])
        trace["samples"][0]["sampling_child_cpu_seconds"] = None
        terminal["memory_measurement"] = summarize_memory_trace(trace)
        memory_path.write_text(json.dumps(trace)); terminal_path.write_text(json.dumps(terminal))
        result = self.reconcile(fixture)
        totals = result["covered_cost_subtotals"]
        self.assertIsNone(totals["sampling_child_cpu_seconds"])
        self.assertFalse(totals["all_costed_jobs_sampler_cpu_known"])
        self.assertGreaterEqual(totals["known_sampling_child_cpu_seconds"], 0)
        self.assertFalse(result["resource_cost_complete"])

    def test_final_extractor_cannot_mutate_cost_receipt_or_reader_scope(self):
        for mode in ("cost", "scope"):
            fixture = self.fixture(native_driven=True)
            calls = 0
            reader = fixture[2][0]["validator"]
            def extract(state, root):
                nonlocal calls
                calls += 1
                # Each full replay join makes two 2-state extraction passes.
                # Mutate in the final join, after initial costs were copied.
                if calls == 8:
                    if mode == "scope": reader.allow_synthetic = False
                    else:
                        path = reader.directory / "terminal.json"
                        terminal = json.loads(path.read_text())
                        terminal["supervisor_thread_cpu_seconds"] += .25
                        path.write_text(json.dumps(terminal))
                return self.extractor(state, root)
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.reconcile(fixture, extract)

    def test_foreign_selected_or_missing_terminal_cannot_be_costed(self):
        fixture = self.fixture(native_driven=True)
        selected, references, replays, reader = fixture
        changed = deepcopy(selected); changed["selection_seed"] += 1
        with self.assertRaises(ValueError): self.reconcile((changed, references, replays, reader))
        path = replays[0]["validator"].directory / "terminal.json"
        path.rename(path.with_name("withheld-terminal.json"))
        with self.assertRaises(FileNotFoundError): self.reconcile(fixture)

    def test_same_directory_conflicting_scope_and_duplicate_launch_cannot_double_cost(self):
        fixture = self.fixture(native_driven=True)
        reader = fixture[1][0]["validator"]
        def clone(directory, scope):
            return MaterializationRunValidator(directory=directory, job_sha256=reader.job_sha,
                binding_sha256=reader.binding, expected_inputs=reader.expected, runtime=reader.runtime,
                allow_synthetic=scope)
        alias = dict(fixture[1][0], construction_ordinal=1, validator=clone(reader.directory, False))
        fixture[1].append(alias)
        with self.assertRaisesRegex(ValueError, "conflicting diagnostic"): self.reconcile(fixture)
        # A second directory holding the same *actual* completed generation
        # is independently readable but is not another supervised CPU bill.
        copied = reader.directory.with_name("copied-completed-fixture")
        shutil.copytree(reader.directory, copied)
        alias["validator"] = clone(copied, True)
        with self.assertRaisesRegex(ValueError, "overlapping diagnostic"): self.reconcile(fixture)


class OwnedNativeReplayPlanningTests(unittest.TestCase):
    """Planning from disposable owners does not execute its returned jobs."""
    extractor = staticmethod(OwnedAncestryRootJoinTests.extractor)

    def fixture(self, **kwargs):
        owner = OwnedRawReplayJoinTests()
        self.addCleanup(owner.doCleanups)
        native, refs, replays, reader = owner.fixture(skip_replay=True,**kwargs)
        self.config = owner.planner_config
        self.attempt_root = refs[0]["validator"].directory.parent / "planned-native"
        return native, refs, reader

    def plan(self, fixture, extractor=None):
        native, refs, reader = fixture
        return plan_owned_native_replays(native, refs, native_validator=reader, ancestry_contract=CONTRACT,
            native_extract=extractor or self.extractor, config=self.config, attempt_root=self.attempt_root)

    def build(self, plan, fixture):
        native, refs, reader = fixture
        return build_owned_native_replay_jobs(plan, native, refs, native_validator=reader, ancestry_contract=CONTRACT,
            native_extract=self.extractor, config=self.config)

    def test_full_roster_deterministic_frozen_specs_and_build_do_not_launch(self):
        fixture = self.fixture()
        with patch("subprocess.Popen", side_effect=AssertionError("planner launched a process")), \
                patch.object(OwnedNativePathBoundaryJob, "_components", side_effect=AssertionError("planner opened simulator")):
            plan = self.plan(fixture)
            again = self.plan(fixture)
            jobs = self.build(plan, fixture)
        self.assertEqual(plan, again)
        result = plan.receipt()
        self.assertEqual((result["actual_native_calls"],result["actual_model_evaluations"],result["actual_sampled_events"]),(1,2,2))
        self.assertEqual((result["required_construction_ordinals"],result["required_construction_sample_records"],
            result["required_boundary_records"],result["required_arriving_action_records"]),(4,8,16,8))
        self.assertEqual(result["actual_materialization_runs"],1)
        self.assertEqual(result["planned_job_records"],2)
        self.assertEqual(len(jobs),2)
        self.assertEqual(sum(row["job_spec"] is None for row in result["records"]),6)
        self.assertFalse(self.attempt_root.exists())
        for row in jobs:
            self.assertIs(type(row["job"]),OwnedNativePathBoundaryJob)
            self.assertEqual(row["plan_sha256"],plan.sha256)
            spec = next(r["job_spec"] for r in result["records"] if r["attempt_directory"] == row["attempt_directory"])
            self.assertEqual(digest(row["job"].input_binding()),spec["expected_inputs_sha256"])
            self.assertNotIn("choices",row["job"].input_binding())
        with self.assertRaises(FrozenInstanceError): plan.payload_utf8 = "{}"
        result["records"].clear()
        self.assertEqual(len(plan.receipt()["records"]),8)
        for flag in ("source_generation","retries_permitted","branch_transition_validated","resource_cost_complete",
                "runtime_qualified","scientific_collection_authorized"):
            self.assertFalse(plan.receipt()[flag])
        self.assertEqual(plan.receipt()["process_launches"],0)

    def test_existing_claim_blocks_coordinate_without_replacement_or_shortened_roster(self):
        fixture = self.fixture()
        plan = self.plan(fixture)
        original = next(r for r in plan.receipt()["records"] if r["job_spec"] is not None)
        Path(original["attempt_directory"]).mkdir(parents=True)
        with self.assertRaises(ValueError): self.build(plan,fixture)
        new = self.plan(fixture).receipt()
        claimed = next(r for r in new["records"] if r["attempt_directory"] == original["attempt_directory"])
        self.assertIn("ATTEMPT_ALREADY_CLAIMED",claimed["blockers"])
        self.assertIsNone(claimed["job_spec"])
        self.assertEqual(new["planned_job_records"],1)
        self.assertEqual(new["required_boundary_records"],16)
        self.assertEqual([r["attempt_directory"] for r in new["records"]],
            [r["attempt_directory"] for r in plan.receipt()["records"]])

    def test_forged_origins_projections_maps_sources_or_dropped_rows_rejected_by_owner_rejoin(self):
        fixture = self.fixture()
        plan = self.plan(fixture)
        for mode in ("origin","projection","map","source","dropped","numeric-coordinate","qualified"):
            value = plan.receipt()
            row = next(r for r in value["records"] if r["job_spec"] is not None)
            spec = row["job_spec"]
            if mode == "origin": spec["root_origin"]["native_capture_sha256"] = "f"*64
            if mode == "projection": spec["native_boundaries"][0]["native"]["hp"]["p1"][0] -= 1
            if mode == "map": spec["party_mapping"]["native_to_original"]["p1"].reverse()
            if mode == "source": value["controller_sources"][next(iter(value["controller_sources"]))] = "f"*64
            if mode == "dropped": value["records"].pop()
            if mode == "numeric-coordinate": row["construction_ordinal"] = 0.0
            if mode == "qualified": value["scientific_collection_authorized"] = True
            forged = OwnedNativeReplayPlan(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False))
            with self.subTest(mode=mode),self.assertRaises(ValueError): self.build(forged,fixture)

    def test_absent_uncleared_mapping_or_ambiguous_action_retains_all_records(self):
        fixture = self.fixture()
        native, refs, reader = fixture
        reader.allow_synthetic = False
        uncleared = self.plan(fixture).receipt()
        reader.allow_synthetic = True
        missing = self.plan((native,[],reader)).receipt()
        def ambiguous(state, root):
            value = json.loads(self.extractor(state,root))
            value["p1"]["active_moves"][1]["engine_id"] = "move"
            return json.dumps(value)
        ambiguous_plan = self.plan(fixture,ambiguous).receipt()
        def mismatched_species(state, root):
            value = json.loads(self.extractor(state,root))
            value["p1"]["party_identities"][1]["engine_id"] = "blissey"
            return json.dumps(value)
        unmapped = self.plan(fixture,mismatched_species).receipt()
        for result in (uncleared,missing,ambiguous_plan,unmapped):
            self.assertEqual((result["planned_job_records"],len(result["records"]),result["required_boundary_records"]),(0,8,16))
        self.assertTrue(any("NATIVE_ACTION_AMBIGUOUS" in r["blockers"] for r in ambiguous_plan["records"]))
        self.assertTrue(any("UNIQUE_PARTY_MAPPING_UNAVAILABLE" in r["blockers"] for r in unmapped["records"]))

    def test_withheld_initial_legality_is_blocked_not_action_reselection(self):
        result = self.plan(self.fixture(withheld=True)).receipt()
        self.assertEqual(result["planned_job_records"],0)
        self.assertEqual(result["required_arriving_action_records"],8)
        self.assertEqual(sum("INITIAL_NATIVE_ACTION_UNAVAILABLE_OR_UNSUPPORTED" in row["blockers"]
            for row in result["records"]),2)

    def test_collapsed_materializer_aliases_do_not_inflate_actual_work(self):
        native, refs, reader = self.fixture()
        refs = refs + [dict(refs[0],construction_ordinal=1)]
        result = self.plan((native,refs,reader)).receipt()
        self.assertEqual((result["materialization_ordinal_records"],result["actual_materialization_runs"]),(2,1))
        self.assertEqual((result["actual_sampled_events"],result["planned_job_records"]),(2,4))
        self.assertEqual(result["required_construction_sample_records"],8)

    def test_late_owner_mutation_cannot_return_a_stale_plan(self):
        fixture = self.fixture()
        native, refs, _ = fixture
        before = deepcopy(refs[0]["result"])
        calls = 0
        def late(state,root):
            nonlocal calls
            calls += 1
            if calls == 4: refs[0]["result"]["produced_bridge_snapshot_sha256"] = "f"*64
            return self.extractor(state,root)
        try:
            with self.assertRaises(ValueError): self.plan(fixture,late)
            self.assertGreaterEqual(calls,4)
        finally:
            refs[0]["result"] = before

    def test_late_ancestor_and_child_symlink_redirection_rejected_after_final_callback(self):
        fixture = self.fixture()
        native, refs, reader = fixture
        for stage, trigger in (("plan",4),("build",10)):
            for target_kind in ("ancestor","child"):
                self.attempt_root = refs[0]["validator"].directory.parent / ("late-"+stage+"-"+target_kind)
                plan = self.plan(fixture)
                first = next(r for r in plan.receipt()["records"] if r["job_spec"] is not None)
                link = self.attempt_root if target_kind == "ancestor" else Path(first["attempt_directory"])
                target = refs[0]["validator"].directory.parent / ("redirected-"+stage+"-"+target_kind)
                target.mkdir()
                calls = 0
                def late(state,root):
                    nonlocal calls
                    calls += 1
                    if calls == trigger:
                        link.parent.mkdir(parents=True,exist_ok=True)
                        link.symlink_to(target,target_is_directory=True)
                    return self.extractor(state,root)
                try:
                    with self.subTest(stage=stage,target=target_kind),self.assertRaisesRegex(ValueError,"redirected"):
                        if stage == "plan": self.plan(fixture,late)
                        else:
                            build_owned_native_replay_jobs(plan,native,refs,native_validator=reader,
                                ancestry_contract=CONTRACT,native_extract=late,config=self.config)
                    self.assertGreaterEqual(calls,trigger)
                finally:
                    if link.is_symlink(): link.unlink()

    def test_config_and_current_runtime_byte_drift_refuse_existing_plan(self):
        fixture = self.fixture()
        plan = self.plan(fixture)
        before = self.config
        self.config = replace(before,read_timeout_seconds=before.read_timeout_seconds+1)
        with self.assertRaises(ValueError): self.build(plan,fixture)
        self.config = before
        bridge = Path(fixture[1][0]["validator"].runtime["argv"][1])
        raw = bridge.read_bytes()
        try:
            bridge.write_bytes(raw+b"\n")
            with self.assertRaises(ValueError): self.build(plan,fixture)
        finally:
            bridge.write_bytes(raw)


if __name__ == "__main__":
    unittest.main()
