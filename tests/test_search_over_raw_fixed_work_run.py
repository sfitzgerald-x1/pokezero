"""Actual disposable supervisor/worker receipts; authored atomic data, no game."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
from pokezero.mcts_eval.search_over_raw_fixed_work_run import FixedWorkRunValidator, fixed_work_reader_source_hashes
from pokezero.mcts_eval.search_over_raw_resources import derive_fixed_work_accounting
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
from pokezero.mcts_eval.paper_reference_parallel import record_owned_process, close_owned_generations
from test_search_over_raw_supervision import BINDING, INPUT_BINDING, selection_inputs


@dataclass(frozen=True)
class AuthoredFixedWorkJob:
    configuration: object
    public_request_sha256: str | None = None
    synthetic_process_fixture = True

    def __call__(self, ownership, phase):
        phase("startup")
        children, receipts = {}, {}
        if self.configuration.arm == "reference":
            for i in range(20):
                child = subprocess.Popen([sys.executable, "-c", "import sys,os; sys.stdin.buffer.read(); os.setsid()"],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                children[i] = child
                receipts[i] = record_owned_process(child.pid, ownership)
                child.stdin.close()
        phase("selection")
        evidence = selection_inputs(self.configuration.arm)
        if self.public_request_sha256 is not None:
            witness = dict(schema="pokezero.search-over-raw.public-request-consumption.v1",
                public_request_sha256=self.public_request_sha256, consumed_binding=INPUT_BINDING,
                full_supported_public_state_and_pending=True, private_source_snapshot_included=False,
                runtime_qualified=False)
            if self.configuration.arm=="reference":
                evidence["adapter_public_request_witness"]=deepcopy(witness)
                for row in evidence["worker_receipts"]:
                    row["evidence"]["worker_public_request_preparation"]=dict(worker=row["worker"],witness=deepcopy(witness))
            else:
                evidence["consumed_world_inputs"]["public_request_observations"] = [
                    dict(ordinal=i, witness=deepcopy(witness)) for i in range(4)]
        if children:
            evidence["worker_pids"] = [child.pid for child in children.values()]
        contract = FixedWorkDiagnosticContract()
        selected = dict(root_id="same-root:" + self.configuration.identity,
            statistics_root_id="same-root", configuration_sha256=self.configuration.identity,
            selection_seed=7, runtime_sha256=BINDING,
            fixed_work_diagnostic=dict(contract=contract.receipt(), accepted_actual=contract.accepted_boundary(self.configuration.arm)),
            evidence=evidence)
        phase("cleanup")
        if children:
            exits = {i: child.wait() for i, child in children.items()}
            close_owned_generations(receipts, {i: SimpleNamespace(pid=child.pid, exitcode=exits[i])
                for i, child in children.items()})
        return selected


class FixedWorkRunReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.checkpoint = self.root / "not-a-model"
        self.checkpoint.write_bytes(b"synthetic unused model")
        self.showdown = self.root / "not-a-simulator"
        dist = self.showdown / "dist/sim"
        dist.mkdir(parents=True)
        for name in ("index.js", "state.js"):
            (dist / name).write_text("// synthetic unused simulator")
        self.files = fixed_work_reader_source_hashes()
        for path in [self.checkpoint, Path(sys.executable).resolve(), *dist.glob("*.js")]:
            self.files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def run_fixture(self, arm="incumbent", public_request_sha256=None):
        cfg = SearchConfiguration(arm, seconds=1., workers=20 if arm == "reference" else 1)
        job = AuthoredFixedWorkJob(cfg, public_request_sha256) if public_request_sha256 is not None else AuthoredFixedWorkJob(cfg)
        contract = FixedWorkDiagnosticContract()
        directory = self.root / arm
        terminal = _supervise_owned_job(job, directory=directory, binding_sha256=BINDING,
            verify_admission=lambda:None, contract=contract, kind="fixed-work", work_phase="selection",
            validate_result=lambda selected:derive_fixed_work_accounting(selected["evidence"],
                arm=arm, binding=INPUT_BINDING, statistics_root_id="same-root", leaf="model"))
        self.assertEqual(terminal["supervisor_exit_code"], 0, terminal)
        inputs = dict(directory=directory, job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256=BINDING, configuration=cfg, root_id="same-root:"+cfg.identity,
            statistics_root_id="same-root", selection_seed=7, runtime_sha256=BINDING,
            consumed_binding=INPUT_BINDING, file_hashes=self.files,
            checkpoint_path=str(self.checkpoint), checkpoint_sha256=self.files[str(self.checkpoint)],
            showdown_root=str(self.showdown))
        if public_request_sha256 is not None:
            inputs["public_request_sha256"] = public_request_sha256
        return terminal["result"], inputs

    def test_catalog_fingerprint_format_does_not_relax_digest_or_exact_binding(self):
        selected, inputs = self.run_fixture()
        binding = dict(INPUT_BINDING, set_source_hash="0123456789abcdef")
        reader = FixedWorkRunValidator(**dict(inputs, consumed_binding=binding), allow_synthetic=True)
        self.assertEqual(reader.consumed, binding)
        # Valid formatting cannot substitute a different actual catalog receipt.
        with self.assertRaises(ValueError):
            reader(selected)
        self.assertIsNone(reader.last_validation)
        for fingerprint in ("", "a"*15, "a"*17, "a"*63, "a"*65, "A"*16,
                "g"*16, True, 123, {"hash":"a"*16}):
            with self.subTest(fingerprint=fingerprint), self.assertRaisesRegex(ValueError, "external consumed root"):
                FixedWorkRunValidator(**dict(inputs, consumed_binding=dict(binding,
                    set_source_hash=fingerprint)))
        with self.assertRaisesRegex(ValueError, "external consumed root"):
            FixedWorkRunValidator(**dict(inputs, consumed_binding=dict(binding, root_binding="a"*16)))
        with self.assertRaisesRegex(ValueError, "external fixed-work input bindings"):
            FixedWorkRunValidator(**dict(inputs, job_sha256="a"*16))

    def test_full_public_request_is_optional_prospective_scope_not_a_legacy_promotion(self):
        selected, inputs = self.run_fixture(public_request_sha256="c"*64)
        reader = FixedWorkRunValidator(**inputs, allow_synthetic=True)
        self.assertEqual(reader(selected), digest(selected))
        self.assertEqual(reader.last_validation["full_public_request_sha256"], "c"*64)
        wrong = dict(inputs, public_request_sha256="d"*64)
        with self.assertRaisesRegex(ValueError, "public request/pending transport drift"):
            FixedWorkRunValidator(**wrong, allow_synthetic=True)(selected)
        legacy, legacy_inputs = self.run_fixture(arm="reference")
        with self.assertRaises(ValueError):
            FixedWorkRunValidator(**legacy_inputs, public_request_sha256="c"*64)(legacy)

    def test_reference_actual_owned_twenty_worker_request_roster_is_prospectively_bound(self):
        selected,inputs=self.run_fixture(arm="reference",public_request_sha256="c"*64)
        reader=FixedWorkRunValidator(**inputs,allow_synthetic=True)
        self.assertEqual(reader(selected),digest(selected))
        self.assertEqual(reader.last_validation["full_public_request_sha256"],"c"*64)
        self.assertEqual(len(reader.last_validation["owned_generation_pids"]),20)
        self.assertEqual(reader.last_validation["actual_work"]["accepted_units"],80)
        wrong=dict(inputs,public_request_sha256="d"*64)
        with self.assertRaisesRegex(ValueError,"actual adapter public capture"):
            FixedWorkRunValidator(**wrong,allow_synthetic=True)(selected)
        from pokezero.mcts_eval.search_over_raw_world_inputs import validate_fixed_work_inputs
        for mutation in ("absent_adapter","absent_worker","wrong_worker","foreign_digest","pending_only_digest"):
            changed=deepcopy(selected["evidence"])
            if mutation=="absent_adapter":changed.pop("adapter_public_request_witness")
            else:
                record=changed["worker_receipts"][-1]["evidence"]
                if mutation=="absent_worker":record.pop("worker_public_request_preparation")
                elif mutation=="wrong_worker":record["worker_public_request_preparation"]["worker"]=False
                else:record["worker_public_request_preparation"]["witness"]["public_request_sha256"]="e"*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate_fixed_work_inputs(changed,arm="reference",binding=INPUT_BINDING,
                    statistics_root_id="same-root",expected_public_request_sha256="c"*64)

    def test_full_request_ordinal_gaps_and_coercion_aliases_fail_fresh_durable_read(self):
        selected, inputs = self.run_fixture(public_request_sha256="c"*64)
        reader = FixedWorkRunValidator(**inputs, allow_synthetic=True)
        original_worker, original_terminal = self.load(inputs,"worker-result.json"), self.load(inputs,"terminal.json")
        for mutation in ("missing", "ordinal-bool", "scope-bool", "hash"):
            changed = deepcopy(selected)
            observations = changed["evidence"]["consumed_world_inputs"]["public_request_observations"]
            if mutation == "missing": observations.pop()
            if mutation == "ordinal-bool": observations[0]["ordinal"] = False
            if mutation == "scope-bool": observations[0]["witness"]["full_supported_public_state_and_pending"] = 1
            if mutation == "hash": observations[0]["witness"]["public_request_sha256"] = "d"*64
            self.write(inputs, "worker-result.json", dict(original_worker,result=changed))
            self.write(inputs, "terminal.json", dict(original_terminal,result=changed))
            with self.assertRaises(ValueError): reader(changed)
            self.assertIsNone(reader.last_validation)

    def load(self, inputs, name):
        return json.loads((inputs["directory"] / name).read_bytes())

    def write(self, inputs, name, value):
        (inputs["directory"] / name).write_text(json.dumps(value))

    def test_actual_owned_synthetic_rosters_do_not_clear_default_and_both_allocations_reconcile(self):
        for arm in ("incumbent", "reference"):
            with self.subTest(arm=arm):
                selected, inputs = self.run_fixture(arm)
                ordinary = FixedWorkRunValidator(**inputs)
                self.assertIsNone(ordinary(selected))
                reader = FixedWorkRunValidator(**inputs, allow_synthetic=True)
                self.assertEqual(reader(selected), digest(selected))
                result = reader.last_validation
                self.assertTrue(result["process_receipts_validated"])
                self.assertEqual(result["actual_work"]["accepted_units"], 64 if arm == "incumbent" else 80)
                for key in ("full_world_equivalence", "resource_cost_complete", "runtime_qualified",
                        "scientific_collection_authorized", "native_ancestry_coverage_validated"):
                    self.assertFalse(result[key])

    def test_missing_wrong_job_binding_source_model_and_changed_inputs_reject(self):
        selected, inputs = self.run_fixture()
        for key, value in (("job_sha256", "f"*64), ("binding_sha256", "f"*64),
                ("runtime_sha256", "f"*64), ("selection_seed", 8)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                FixedWorkRunValidator(**{**inputs, key:value}, allow_synthetic=True)(selected)
        reader = FixedWorkRunValidator(**inputs, allow_synthetic=True)
        reader.consumed["subject"] = "p2"
        with self.assertRaises(ValueError): reader(selected)
        for omitted in (str(self.checkpoint), str(Path(sys.executable).resolve()),
                str(self.showdown / "dist/sim/state.js"), str(Path(__file__).parents[1] / "src/pokezero/mcts_eval/search_over_raw_world_inputs.py")):
            files = deepcopy(self.files); files.pop(omitted)
            with self.subTest(omitted=omitted), self.assertRaises(ValueError):
                FixedWorkRunValidator(**{**inputs, "file_hashes":files})
        self.checkpoint.write_bytes(b"changed")
        with self.assertRaises(ValueError): FixedWorkRunValidator(**inputs)(selected)

    def test_contradictory_declared_worker_configuration_rejected(self):
        selected, inputs = self.run_fixture()
        configuration = SearchConfiguration("incumbent", seconds=1., workers=20)
        with self.assertRaises(ValueError): FixedWorkRunValidator(**{**inputs,
            "configuration":configuration, "root_id":"same-root:"+configuration.identity})

    def test_normal_exit_phase_cpu_memory_and_exact_return_tampering_reject(self):
        selected, inputs = self.run_fixture()
        original = self.load(inputs, "terminal.json")
        edits = [lambda t:t.update(worker_exit_code=True), lambda t:t.update(worker_exit_code=7),
            lambda t:t.update(watchdog_reached=True), lambda t:t.update(cpu_seconds_including_reaped_children=None),
            lambda t:t["phase_costs"].update(selection_wall_seconds=999.),
            lambda t:t.update(result={}), lambda t:t.update(resource_cost_complete=True),
            lambda t:t.update(memory_measurement=None, resource_measurement_error_type=None),
            lambda t:t["observed_work_accounting"].update(accepted_units=65)]
        for edit in edits:
            changed = deepcopy(original); edit(changed); self.write(inputs, "terminal.json", changed)
            with self.assertRaises(ValueError): FixedWorkRunValidator(**inputs, allow_synthetic=True)(selected)
        self.write(inputs, "terminal.json", original)
        self.assertEqual(FixedWorkRunValidator(**inputs, allow_synthetic=True)(selected), digest(selected))
        (inputs["directory"] / "worker-result.json").unlink()
        with self.assertRaises(FileNotFoundError): FixedWorkRunValidator(**inputs)(selected)

    def test_rehashed_raw_report_substitution_cannot_change_the_bound_actual_return(self):
        selected, inputs = self.run_fixture()
        worker, terminal = self.load(inputs, "worker-result.json"), self.load(inputs, "terminal.json")
        changed = deepcopy(selected)
        call = changed["evidence"]["consumed_world_inputs"]["calls"][0]
        call["native_report_utf8"] += " "
        # Canonical JSON and all commit receipts remain unchanged; actual bytes do not.
        worker["result"] = terminal["result"] = changed
        self.write(inputs, "worker-result.json", worker); self.write(inputs, "terminal.json", terminal)
        with self.assertRaises(ValueError): FixedWorkRunValidator(**inputs, allow_synthetic=True)(changed)

    def test_raw_json_numeric_type_alias_cannot_join_typed_atomic_evidence(self):
        from pokezero.mcts_eval.search_over_raw_world_inputs import validate_fixed_work_inputs
        evidence = selection_inputs("incumbent")
        call = evidence["consumed_world_inputs"]["calls"][0]
        parsed = json.loads(call["native_report_utf8"])
        parsed["atomic_work"]["rounds"][0]["commits"][0]["path"][0]["decision"] = False
        call["native_report_utf8"] = json.dumps(parsed)
        call["report_utf8_sha256"] = hashlib.sha256(call["native_report_utf8"].encode()).hexdigest()
        # Python equality would conceal False == 0; exact canonical evidence must not.
        self.assertEqual(parsed, call["native_report"])
        self.assertNotEqual(digest(parsed), digest(call["native_report"]))
        with self.assertRaises(ValueError): validate_fixed_work_inputs(evidence, arm="incumbent",
            binding=INPUT_BINDING, statistics_root_id="same-root", construction_required=True)

    def test_reference_exports_must_join_exactly_twenty_normal_owned_generations(self):
        selected, inputs = self.run_fixture("reference")
        path = sorted((inputs["directory"] / "owned-groups").glob("closed-group-*.json"))[0]
        closed = json.loads(path.read_bytes()); closed["worker_exit_code"] = True
        path.write_text(json.dumps(closed))
        with self.assertRaises(ValueError): FixedWorkRunValidator(**inputs, allow_synthetic=True)(selected)
        closed["worker_exit_code"] = 0; path.write_text(json.dumps(closed))
        selected["evidence"]["worker_pids"][0] = 123
        worker, terminal = self.load(inputs, "worker-result.json"), self.load(inputs, "terminal.json")
        worker["result"] = terminal["result"] = selected
        self.write(inputs, "worker-result.json", worker); self.write(inputs, "terminal.json", terminal)
        with self.assertRaises(ValueError): FixedWorkRunValidator(**inputs, allow_synthetic=True)(selected)

    def test_mutation_during_independent_resource_validation_rejects(self):
        selected, inputs = self.run_fixture()
        from pokezero.mcts_eval.search_over_raw_fixed_work_run import derive_fixed_work_accounting as derive
        def mutate(*args, **kwargs):
            value = derive(*args, **kwargs)
            selected["selection_seed"] = 99
            return value
        with patch("pokezero.mcts_eval.search_over_raw_fixed_work_run.derive_fixed_work_accounting", side_effect=mutate):
            with self.assertRaises(ValueError): FixedWorkRunValidator(**inputs, allow_synthetic=True)(selected)


if __name__ == "__main__":
    unittest.main()
