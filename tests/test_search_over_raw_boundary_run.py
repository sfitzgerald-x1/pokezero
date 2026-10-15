"""Actual POSIX ownership and real bridge entrypoint, fake simulator only."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import time
import unittest

from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_boundary_run import (
    OwnedBoundaryJob, BoundaryRunValidator, BoundaryRunContract, supervise_boundary_run,
    OwnedRawBoundaryJob, RawBoundaryRunValidator, RawBoundaryRunContract, supervise_raw_boundary_run, raw_root_json,
)
from pokezero.mcts_eval.search_over_raw_outcome_run import OwningRunValidator, outcome_runtime_binding
from pokezero.mcts_eval.search_over_raw_outcomes import _snapshot_json
from test_search_over_raw_outcomes import EntrypointSyntheticEnv, BINDING
import test_search_over_raw_outcome_run as outcome_fixtures


@dataclass(frozen=True)
class SyntheticBoundaryJob(OwnedBoundaryJob):
    mode: str = "complete"
    synthetic_process_fixture = True

    def _components(self):
        env = EntrypointSyntheticEnv(self.config)
        if self.mode == "missing-close":
            read = env._read_event
            def altered(*,timeout):
                event = read(timeout=timeout)
                return dict(type="error") if event == dict(type="closed") else event
            env._read_event = altered
        return env

    def _advance(self, env, choices):
        if self.mode == "hang": time.sleep(60.)
        if self.mode == "failure": raise RuntimeError("synthetic transition failure")
        return super()._advance(env,choices)


class BoundaryRunTests(unittest.TestCase):
    setUp = outcome_fixtures.OutcomeRunTests.setUp
    load = outcome_fixtures.OutcomeRunTests.load
    save = outcome_fixtures.OutcomeRunTests.save

    def job(self, mode="complete", *, requests=None, choices=None):
        initial = _snapshot_json(self.snapshot)
        clone = dict(battleId="boundary-owned-fixture",cloneId="path-fixture",
            snapshotJsonSha256=hashlib.sha256(initial.encode()).hexdigest())
        return SyntheticBoundaryJob(self.config,self.snapshot,deepcopy(BINDING),deepcopy(self.runtime),clone,
            ["root","terminal"] if requests is None else requests,
            [dict(p1="move 1",p2="move 1")] if choices is None else choices,mode)

    def run_job(self, job=None, validate_result=lambda result:True, **kwargs):
        job = self.job() if job is None else job
        receipt = supervise_boundary_run(job,directory=self.directory,binding_sha256="a"*64,
            verify_admission=lambda:True,validate_result=validate_result,**kwargs)
        return job,receipt

    def reader(self, job, *, allow_synthetic=True, expected=None):
        return BoundaryRunValidator(directory=self.directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256="a"*64,root_binding=BINDING,expected_snapshot=self.snapshot,runtime=self.runtime,
            expected_inputs=job.input_binding() if expected is None else expected,allow_synthetic=allow_synthetic)

    def test_actual_one_shot_boundary_job_wait4_and_distinct_reader(self):
        job,receipt = self.run_job()
        self.assertEqual(receipt["status"],"BOUNDARY_RUN_COMPLETE",receipt)
        self.assertEqual(receipt["worker_exit_code"],0)
        self.assertIn("boundaries_wall_seconds",receipt["phase_costs"])
        self.assertNotIn("continuations_wall_seconds",receipt["phase_costs"])
        result=receipt["result"]; capture=result["capture"]
        self.assertEqual(self.reader(job)(capture),digest(capture))
        self.assertIsNone(self.reader(job,allow_synthetic=False)(capture))
        closed=self.load("bridge-close.json");launch=self.load("bridge-launch.json")
        self.assertEqual(launch["schema"],"pokezero.search-over-raw.boundary-bridge-launch.v1")
        self.assertEqual(closed["schema"],"pokezero.search-over-raw.boundary-bridge-close.v1")
        self.assertNotIn("bank_sha256",closed)
        self.assertEqual(closed["capture_sha256"],digest(capture))
        self.assertTrue(closed["reaped"]);self.assertTrue(closed["normal_close"])
        self.assertFalse(closed["forced_cleanup"]);self.assertTrue(closed["pumps_drained"])
        self.assertEqual([e["boundaryGeneration"] for e in capture["snapshots"]],[0,1])
        self.assertNotIn("currentSnapshotJson\"",(self.directory/"bridge-transport.json").read_text())
        self.assertEqual(self.directory.stat().st_mode&0o777,0o700)
        for flag in ("construction_provenance_validated","native_ancestry_coverage_validated",
            "branch_transition_validated","independent_terminal_truth","runtime_qualified",
            "resource_cost_complete","scientific_collection_authorized"):
            self.assertFalse(result[flag])
        outcome=OwningRunValidator(directory=self.directory,job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256="a"*64,root_binding=BINDING,expected_snapshot=self.snapshot,runtime=self.runtime,
            allow_synthetic=True)
        with self.assertRaises(ValueError):outcome(capture)
        with self.assertRaises(FileExistsError):self.run_job(job)

    def test_root_only_trace_does_not_imply_native_path_coverage_or_terminal_truth(self):
        job,receipt=self.run_job(self.job(requests=["root"],choices=[]))
        self.assertEqual(receipt["status"],"BOUNDARY_RUN_COMPLETE",receipt)
        self.assertEqual(self.reader(job)(receipt["result"]["capture"]),digest(receipt["result"]["capture"]))
        self.assertFalse(receipt["result"]["native_ancestry_coverage_validated"])
        self.assertFalse(json.loads(receipt["result"]["capture"]["snapshots"][0]["currentSnapshotJson"])["battle"]["ended"])

    def test_failed_unreached_or_incomplete_actor_trace_cannot_clear_or_retry(self):
        jobs=(self.job("failure"),self.job("missing-close"),self.job(choices=[dict(p1="move 1")]),
            self.job(requests=["root","next","unreached"],choices=[dict(p1="move 1",p2="move 1")]*2))
        for index,job in enumerate(jobs):
            self.directory=self.root/f"failure-{index}"
            _,receipt=self.run_job(job)
            with self.subTest(index=index):
                self.assertEqual(receipt["status"],"OPERATIONAL_STOP",receipt)
                self.assertEqual(receipt["worker_exit_code"],1)
                self.assertIsNone(receipt["result"])
                closed=self.load("bridge-close.json")
                self.assertFalse(closed["normal_close"]);self.assertTrue(closed["reaped"])
                with self.assertRaises(FileExistsError):self.run_job(job)

    def test_external_reader_rejects_same_child_close_runtime_contract_and_schema_drift(self):
        job,receipt=self.run_job();capture=receipt["result"]["capture"]
        cases=(("claim.json","contract",dict(BoundaryRunContract().receipt(),max_transitions=63)),
            ("claim.json","binding_sha256","b"*64),("launch.json","pid",False),
            ("worker-result.json","worker_pid",1),("terminal.json","status","OUTCOME_RUN_COMPLETE"),
            ("terminal.json","worker_exit_code",False),("terminal.json","watchdog_reached",True),
            ("bridge-launch.json","schema","pokezero.search-over-raw.outcome-bridge-launch.v1"),
            ("bridge-launch.json","parent_pid",1),("bridge-launch.json","argv",["wrong"]),
            ("bridge-close.json","pid",1),("bridge-close.json","normal_close",False),
            ("bridge-close.json","forced_cleanup",True),("bridge-close.json","exit_code",False),
            ("bridge-close.json","wait_ownership_lost",True),("bridge-close.json","pumps_drained",False),
            ("bridge-close.json","capture_sha256","b"*64))
        for name,key,value in cases:
            original=self.load(name);self.save(name,dict(original,**{key:value}))
            with self.subTest(name=name,key=key),self.assertRaises(ValueError):self.reader(job)(capture)
            self.save(name,original)
        # Even a self-consistently rehashed stored transport must match capture.
        transport=self.load("bridge-transport.json");closed=self.load("bridge-close.json")
        other=deepcopy(transport);other["records"][0]["payload"]["cloneId"]="substitute"
        self.save("bridge-transport.json",other)
        self.save("bridge-close.json",dict(closed,transport_sha256=digest(other["records"])))
        with self.assertRaises(ValueError):self.reader(job)(capture)
        self.save("bridge-transport.json",transport);self.save("bridge-close.json",closed)
        self.assertEqual(self.reader(job)(capture),digest(capture))

    def test_frozen_joint_choice_and_request_roster_cannot_be_inferred_from_capture(self):
        job,receipt=self.run_job();capture=receipt["result"]["capture"]
        for key,value in (("choices",[dict(p1="move 2",p2="move 1")]),
            ("request_ids",["root","other"]),("clone_binding",dict(job.clone_binding,cloneId="other"))):
            changed=deepcopy(job.input_binding());changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.reader(job,expected=changed)(capture)
        changed=deepcopy(capture);changed["snapshots"][0]["currentSnapshotJson"]="{}"
        with self.assertRaises(ValueError):self.reader(job)(changed)
        checker=self.reader(job);checker.expected["choices"][0]["p1"]="move 2"
        with self.assertRaises(ValueError):checker(capture)

    def test_watchdog_or_actual_nonzero_bridge_exit_never_means_complete(self):
        job,receipt=self.run_job(self.job("hang"),_synthetic_watchdog_seconds=.5)
        self.assertEqual(receipt["status"],"BOUNDARY_RUN_INCOMPLETE",receipt)
        self.assertEqual(receipt["supervisor_exit_code"],124);self.assertIsNone(receipt["result"])
        self.directory=self.root/"nonzero"
        index=self.root/"showdown/dist/sim/index.js"
        index.write_text("const fixtureExit=process.exit.bind(process);process.exit=code=>fixtureExit(code===0?7:code);\n"
            +index.read_text())
        self.runtime=outcome_runtime_binding(self.config,checkpoint_path=self.runtime["checkpoint_path"])
        _,receipt=self.run_job()
        self.assertEqual(receipt["status"],"OPERATIONAL_STOP",receipt)
        closed=self.load("bridge-close.json")
        self.assertEqual(closed["exit_code"],7);self.assertFalse(closed["normal_close"])
        self.assertTrue(closed["reaped"]);self.assertIsNone(receipt["result"])

    def test_exact_external_inputs_and_single_command_shapes_required_before_launch(self):
        job,receipt=self.run_job()
        for key in job.input_binding():
            changed=job.input_binding();changed.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):self.reader(job,expected=changed)
        for key,value in (("max_transitions",True),("max_transitions",63),("choices",[]),
            ("choices",[dict(p1="move 1\nmove 2")]),("request_ids",["root","root"]),
            ("request_ids",("root","terminal")),("choices",[dict(p3="move 1")])):
            changed=job.input_binding();changed[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.reader(job,expected=changed)
        self.directory=self.root/"invalid-before-launch"
        with self.assertRaises(ValueError):self.run_job(self.job(choices=[dict(p1="move 1\nmove 2",p2="move 1")]))
        self.assertFalse(self.directory.exists())

    def test_result_mutating_or_refusing_callback_cannot_publish_accepted_terminal(self):
        def mutate(result):result["request_ids"]=["substitution"];return True
        for index,validator in enumerate((mutate,lambda result:False)):
            self.directory=self.root/f"acceptance-{index}"
            _,receipt=self.run_job(validate_result=validator)
            self.assertEqual(receipt["status"],"OPERATIONAL_STOP",receipt)
            self.assertEqual(receipt["worker_exit_code"],0)  # Exit0 is not acceptance.
            self.assertIsNone(receipt["result"])
            self.assertTrue(self.load("bridge-close.json")["normal_close"])


@dataclass(frozen=True)
class SyntheticRawBoundaryJob(OwnedRawBoundaryJob):
    mode: str = "complete"
    synthetic_process_fixture = True
    _components = SyntheticBoundaryJob._components
    def _advance(self, env, choices):
        if self.mode == "failure": raise RuntimeError("synthetic transition failure")
        return OwnedRawBoundaryJob._advance(self, env, choices)


class RawBoundaryRunTests(unittest.TestCase):
    setUp = outcome_fixtures.OutcomeRunTests.setUp
    load = outcome_fixtures.OutcomeRunTests.load
    save = outcome_fixtures.OutcomeRunTests.save

    def job(self, *, requests=None, choices=None, mode="complete"):
        root = deepcopy(self.snapshot.bridge_snapshot)
        origin = dict(materialization_capture_sha256="b"*64, produced_bridge_snapshot_sha256=digest(root),
            native_selected_sha256="c"*64, native_capture_sha256="d"*64, construction_ordinal=0, evaluation_ordinal=1)
        clone = dict(battleId="raw-boundary-owned", cloneId="raw-path",
            snapshotJsonSha256=hashlib.sha256(raw_root_json(root).encode()).hexdigest())
        return SyntheticRawBoundaryJob(self.config, root, deepcopy(BINDING), origin, deepcopy(self.runtime), clone,
            ["root", "terminal"] if requests is None else requests,
            [dict(p1="move 1", p2="move 1")] if choices is None else choices, mode)

    def run_job(self, job=None, validate_result=lambda _:True, **kwargs):
        job = self.job() if job is None else job
        receipt = supervise_raw_boundary_run(job, directory=self.directory, binding_sha256="a"*64,
            verify_admission=lambda:True, validate_result=validate_result, **kwargs)
        return job, receipt

    def reader(self, job, *, allow_synthetic=True, expected=None, root=None):
        return RawBoundaryRunValidator(directory=self.directory, job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256="a"*64, expected_inputs=job.input_binding() if expected is None else expected,
            expected_raw_root=job.raw_root if root is None else root, runtime=self.runtime, allow_synthetic=allow_synthetic)

    def test_distinct_raw_root_owner_has_no_python_actor_snapshot_and_no_source_generation(self):
        job, receipt = self.run_job()
        self.assertEqual(receipt["status"], "BOUNDARY_RUN_COMPLETE", receipt)
        result = receipt["result"]
        self.assertNotIn("snapshot_identity", result)
        self.assertNotIn("source_battle_token", result)
        self.assertEqual(result["raw_root_sha256"], digest(job.raw_root))
        self.assertEqual(self.reader(job)(result["capture"]), digest(result["capture"]))
        self.assertIsNone(self.reader(job, allow_synthetic=False)(result["capture"]))
        self.assertEqual(self.load("bridge-launch.json")["schema"], "pokezero.search-over-raw.raw-boundary-bridge-launch.v1")
        self.assertEqual(self.load("bridge-close.json")["schema"], "pokezero.search-over-raw.raw-boundary-bridge-close.v1")
        self.assertEqual(self.load("claim.json")["contract"], RawBoundaryRunContract().receipt())
        commands = [row["payload"]["type"] for row in self.load("bridge-transport.json")["records"] if row["kind"] == "command"]
        self.assertEqual(commands, ["clone_snapshot", "snapshot_owned_boundary", "choices", "snapshot_owned_boundary",
            "release_clone", "close"])
        self.assertFalse(result["construction_provenance_validated"])
        self.assertFalse(result["branch_transition_validated"])
        self.assertFalse(result["scientific_collection_authorized"])
        with self.assertRaises(FileExistsError): self.run_job(job)

    def test_raw_root_origin_and_alias_substitution_cannot_clear(self):
        job, receipt = self.run_job(); capture = receipt["result"]["capture"]
        for key, value in (("evaluation_ordinal", 2), ("evaluation_ordinal", 1.0), ("construction_ordinal", False),
                ("materialization_capture_sha256", "e"*64), ("native_capture_sha256", "e"*64)):
            expected = job.input_binding(); expected["root_origin"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.reader(job, expected=expected)(capture)
        root = deepcopy(job.raw_root); root["battle"]["log"] = ["substituted root"]
        with self.assertRaises(ValueError): self.reader(job, root=root)(capture)
        stored = self.load("worker-result.json"); altered = deepcopy(stored)
        altered["result"]["max_transitions"] = 64.0
        self.save("worker-result.json", altered)
        with self.assertRaises(ValueError): self.reader(job)(capture)
        self.save("worker-result.json", stored)
        self.assertEqual(self.reader(job)(capture), digest(capture))

    def test_raw_contract_cannot_be_read_as_python_snapshot_or_vice_versa(self):
        job, receipt = self.run_job(); capture = receipt["result"]["capture"]
        legacy = BoundaryRunValidator(directory=self.directory, job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256="a"*64, root_binding=BINDING, expected_snapshot=self.snapshot, runtime=self.runtime,
            expected_inputs=BoundaryRunTests.job(self).input_binding(), allow_synthetic=True)
        with self.assertRaises(ValueError): legacy(capture)
        with self.assertRaises(ValueError): supervise_boundary_run(job, directory=self.root / "legacy", binding_sha256="a"*64,
            verify_admission=lambda:True, validate_result=lambda _:True)
        self.assertFalse((self.root / "legacy").exists())

    def test_unreached_actors_failures_and_acceptance_mutation_never_shorten_path(self):
        cases = (self.job(choices=[dict(p1="move 1")]), self.job(mode="failure"), self.job(mode="missing-close"),
            self.job(requests=["root", "next", "unreached"], choices=[dict(p1="move 1", p2="move 1")]*2))
        for i, job in enumerate(cases):
            self.directory = self.root / f"raw-failure-{i}"
            _, receipt = self.run_job(job)
            self.assertEqual(receipt["status"], "OPERATIONAL_STOP", receipt)
            self.assertEqual(receipt["worker_exit_code"], 1)
            self.assertIsNone(receipt["result"])
        self.directory = self.root / "raw-mutation"
        def mutate(result): result["root_origin"]["evaluation_ordinal"] = 99; return True
        _, receipt = self.run_job(validate_result=mutate)
        self.assertEqual(receipt["status"], "OPERATIONAL_STOP", receipt)
        self.assertEqual(receipt["worker_exit_code"], 0)
        self.assertIsNone(receipt["result"])

    def test_raw_input_guards_precede_launch_and_private_source_state_is_not_reconstructed(self):
        job = self.job()
        job.raw_root["battle"]["ended"] = True
        with self.assertRaises(ValueError): self.run_job(job)
        self.assertFalse(self.directory.exists())
        job = self.job(); job.root_origin["construction_ordinal"] = True
        with self.assertRaises(ValueError): self.run_job(job)
        self.assertFalse(self.directory.exists())


if __name__ == "__main__":unittest.main()
