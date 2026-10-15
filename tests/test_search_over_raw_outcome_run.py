"""Disposable synthetic bridges under actual POSIX supervision; no games/model."""
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import re
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from pokezero.local_showdown import LocalShowdownConfig
from pokezero.mcts_eval.search_over_raw import digest
from pokezero.mcts_eval.search_over_raw_outcomes import OutcomeReceiptValidator
from pokezero.mcts_eval.search_over_raw_outcome_run import (
    OwnedContinuationJob, OwningRunValidator, outcome_runtime_binding,
    verify_outcome_runtime, supervise_outcome_run,
)
from test_search_over_raw_outcomes import EntrypointSyntheticEnv, parser_snapshot, BINDING, CONTRACT, NAMESPACE


class ThreadStartupTests(unittest.TestCase):
    """Both real startup callers; mocked models/envs, never simulator games."""

    def check_startup(self, caller, mode, *, real_torch=False):
        from pokezero import neural_policy
        from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory, _ShowdownRuntime
        from pokezero.mcts_eval.search_over_raw_world_inputs import _state_payload
        calls = []
        threads = {"interop": 4}
        def set_interop(value):
            calls.append(("interop", value))
            if mode == "locked":
                raise RuntimeError("authored incompatible inter-op pool")
            threads["interop"] = value
        torch = SimpleNamespace(set_num_threads=lambda n: calls.append(("intra", n)),
            get_num_interop_threads=lambda: threads["interop"], set_num_interop_threads=set_interop)
        with tempfile.TemporaryDirectory(prefix="pokezero-thread-startup-") as temporary, ExitStack() as stack:
            checkpoint = Path(temporary) / "authored-checkpoint"; checkpoint.write_bytes(b"not-a-model")
            weights = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            source = SimpleNamespace(metadata=SimpleNamespace(source_hash="a"*64))
            policy = SimpleNamespace(weights_sha256=weights, result=SimpleNamespace(belief_set_source_hash="a"*64))
            config = LocalShowdownConfig(set_belief_source=False)
            env = Mock(belief_set_source_hash="a"*64)
            stack.enter_context(patch.dict(os.environ, {
                "POKEZERO_TORCH_NUM_THREADS": "1",
                "POKEZERO_TORCH_NUM_INTEROP_THREADS": "1" if mode == "environment" else ""}))
            stack.enter_context(patch.object(neural_policy, "_TORCH_THREAD_ENV_APPLIED", False))
            if not real_torch:
                def require():
                    calls.append(("require", None))
                    neural_policy._apply_torch_thread_env(torch)
                    return torch
                stack.enter_context(patch.object(neural_policy, "require_torch", side_effect=require))
            loader = stack.enter_context(patch.object(neural_policy, "load_transformer_policy", return_value=policy))
            stack.enter_context(patch("pokezero.local_showdown.LocalShowdownEnv", return_value=env))
            stack.enter_context(patch("pokezero.collection.env_config_with_policy_spec_masks", return_value=config))
            stack.enter_context(patch("pokezero.randbat.load_gen3_randbat_source_cached", return_value=source))
            stack.enter_context(patch("pokezero.mcts_eval.paper_reference_showdown.ChampionEvaluator", return_value=Mock()))
            stack.enter_context(patch("pokezero.mcts_eval.paper_reference_runtime.ChampionEvaluator", return_value=Mock()))
            if caller == "outcome":
                snapshot = SimpleNamespace(belief_engine=SimpleNamespace(set_source=source))
                job = OwnedContinuationJob(config, snapshot, {},
                    dict(checkpoint_path=str(checkpoint), files={str(checkpoint):weights},
                        config_sha256=digest(_state_payload(config))), "p1", {"raw":0}, "authored")
                startup = job._components
            else:
                binding = ShowdownWorkerFactory(str(checkpoint), weights, temporary, "a"*64)
                startup = lambda: _ShowdownRuntime(binding, 0)
            if mode == "locked":
                with self.assertRaisesRegex(RuntimeError, "incompatible inter-op pool"):
                    startup()
                loader.assert_not_called()
                env.reset.assert_not_called()
            else:
                result = startup()
                loader.assert_called_once()
                if caller == "outcome":
                    self.assertIs(result[0], env); env.reset.assert_not_called()
                else:
                    self.assertIs(result.env, env); env.reset.assert_called_once_with(seed=2026100400)
                if real_torch:
                    self.assertEqual(neural_policy.torch.get_num_threads(), 1)
                    self.assertEqual(neural_policy.torch.get_num_interop_threads(), 1)
                else:
                    self.assertEqual(threads["interop"], 1)
            if not real_torch:
                self.assertEqual(calls[0], ("require", None))
                self.assertEqual([c for c in calls if c[0] == "interop"], [("interop", 1)])

    def test_outcome_environment_pool_is_not_configured_twice(self):
        self.check_startup("outcome", "environment")

    def test_reference_environment_pool_is_not_configured_twice(self):
        self.check_startup("reference", "environment")

    def test_outcome_default_pool_is_changed_to_one(self):
        self.check_startup("outcome", "default")

    def test_reference_default_pool_is_changed_to_one(self):
        self.check_startup("reference", "default")

    def test_outcome_locked_incompatible_pool_refuses_before_model_or_env(self):
        self.check_startup("outcome", "locked")

    def test_reference_locked_incompatible_pool_refuses_before_model_or_env(self):
        self.check_startup("reference", "locked")

    def check_fresh_interpreter(self, caller):
        command = "from test_search_over_raw_outcome_run import ThreadStartupTests; " + \
            f"ThreadStartupTests().check_startup({caller!r}, 'environment', real_torch=True)"
        result = subprocess.run([sys.executable, "-B", "-c", command], capture_output=True, text=True, timeout=30.)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_torch_outcome_startup_in_fresh_interpreter(self):
        self.check_fresh_interpreter("outcome")

    def test_actual_torch_reference_startup_in_fresh_interpreter(self):
        self.check_fresh_interpreter("reference")


@dataclass(frozen=True)
class SyntheticContinuationJob(OwnedContinuationJob):
    mode: str = "complete"
    synthetic_process_fixture = True

    def _components(self):
        env = EntrypointSyntheticEnv(self.config)
        if self.mode == "missing-close":
            read = env._read_event
            def altered(*, timeout):
                event = read(timeout=timeout)
                return dict(type="error") if event == dict(type="closed") else event
            env._read_event = altered
        if self.mode == "transport-drift":
            read = env._read_event
            def altered(*, timeout):
                event = read(timeout=timeout)
                if event and event.get("type") == "cloned":
                    event["snapshotJsonSha256"] = "b"*64
                return event
            env._read_event = altered
        def evaluator(observation):
            if self.mode == "evaluator-failure": raise RuntimeError("synthetic evaluator failure")
            if self.mode == "hang": time.sleep(60.)
            if self.mode == "runtime-drift":
                Path(self.runtime["checkpoint_path"]).write_text("drift")
            return [0, 1], [.25, .75]
        return env, evaluator


class OutcomeRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pokezero-owned-outcome-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sim = self.root / "showdown" / "dist" / "sim"; sim.mkdir(parents=True)
        text = Path(__file__).with_name("test_owned_continuation.mjs").read_text()
        for variable, name in (("fakeRuntime", "index.js"), ("fakeState", "state.js")):
            (sim / name).write_text(re.search(r"const " + variable + r" = `([\s\S]*?)`;", text).group(1))
        checkpoint = self.root / "synthetic-checkpoint"; checkpoint.write_bytes(b"not-a-model")
        self.config = LocalShowdownConfig(showdown_root=str(self.root / "showdown"),
            set_belief_source=False, read_timeout_seconds=2.)
        env = EntrypointSyntheticEnv(self.config)
        self.snapshot = parser_snapshot(env)
        self.runtime = outcome_runtime_binding(self.config, checkpoint_path=checkpoint)
        self.directory = self.root / "attempt"

    def job(self, mode="complete"):
        return SyntheticContinuationJob(self.config, self.snapshot, deepcopy(BINDING), deepcopy(self.runtime),
            "p1", {"raw": 0}, NAMESPACE, mode)

    def run_job(self, job=None, validate_result=lambda result: True, **kwargs):
        job = self.job() if job is None else job
        receipt = supervise_outcome_run(job, directory=self.directory, binding_sha256="a"*64,
            verify_admission=lambda: True, validate_result=validate_result, **kwargs)
        return job, receipt

    def reader(self, job, *, allow_synthetic=True):
        return OwningRunValidator(directory=self.directory,
            job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(), binding_sha256="a"*64,
            root_binding=BINDING, expected_snapshot=self.snapshot, runtime=self.runtime,
            allow_synthetic=allow_synthetic)

    def load(self, name): return json.loads((self.directory / name).read_bytes())
    def save(self, name, value): (self.directory / name).write_text(json.dumps(value))

    def test_actual_external_worker_bridge_trace_normal_wait4_join_and_reader(self):
        job, receipt = self.run_job()
        self.assertEqual(receipt["status"], "OUTCOME_RUN_COMPLETE", receipt)
        self.assertEqual(receipt["worker_exit_code"], 0)
        self.assertIn("continuations_wall_seconds", receipt["phase_costs"])
        self.assertNotIn("selection_wall_seconds", receipt["phase_costs"])
        self.assertFalse(receipt["resource_cost_complete"]); self.assertFalse(receipt["runtime_qualified"])
        bank = receipt["result"]["bank"]
        self.assertEqual(len(bank["cells"]), 8)
        self.assertEqual(self.reader(job)(bank), digest(bank))
        self.assertIsNone(self.reader(job, allow_synthetic=False)(bank))
        validator = OutcomeReceiptValidator(bank, expected_snapshot=self.snapshot, root_binding=BINDING,
            runtime_sha256=digest(self.runtime), validate_provenance_and_cleanup=self.reader(job))
        for row in receipt["result"]["outcomes"]["outcomes"]:
            complete = dict(root_id=BINDING["root_id"], **row)
            self.assertEqual(validator(complete, CONTRACT), digest(complete))
        launch, closed = self.load("bridge-launch.json"), self.load("bridge-close.json")
        self.assertEqual(launch["parent_pid"], receipt["worker_pid"])
        self.assertEqual(closed["exit_code"], 0); self.assertTrue(closed["reaped"])
        self.assertTrue(closed["pumps_drained"]); self.assertFalse(closed["forced_cleanup"])
        self.assertEqual((self.directory / "job.pickle").stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.directory.stat().st_mode & 0o777, 0o700)
        self.assertNotIn("snapshotJson\"", (self.directory / "bridge-transport.json").read_text())
        self.assertEqual(self.load("bridge-transport.json")["records"][-1],
            dict(kind="event", payload=dict(type="closed")))

    def test_nonzero_failure_or_forced_close_cannot_clear_bank_or_restart_attempt(self):
        for mode in ("missing-close", "transport-drift", "evaluator-failure", "runtime-drift"):
            # One create-only directory per explicitly synthetic test, not scientific retries.
            self.directory = self.root / mode
            job, receipt = self.run_job(self.job(mode))
            with self.subTest(mode=mode):
                self.assertEqual(receipt["status"], "OPERATIONAL_STOP", receipt)
                self.assertEqual(receipt["worker_exit_code"], 1)
                self.assertIsNone(receipt["result"])
                closed = self.load("bridge-close.json")
                self.assertFalse(closed["normal_close"])
                self.assertTrue(closed["reaped"])
                self.assertTrue(closed["forced_cleanup"])
                with self.assertRaises((ValueError, FileExistsError)):
                    self.run_job(job)

    def test_external_reader_rejects_receipt_mutation_and_hash_substitution(self):
        job, receipt = self.run_job(); bank = receipt["result"]["bank"]
        cases = (
            ("claim.json", "job_sha256", "b"*64), ("claim.json", "binding_sha256", "b"*64),
            ("launch.json", "pid", False), ("launch.json", "birth_identity", ""),
            ("worker-result.json", "exit_code", False), ("worker-result.json", "worker_pid", 1),
            ("terminal.json", "worker_exit_code", None), ("terminal.json", "watchdog_reached", True),
            ("terminal.json", "cleanup_error_type", "failure"), ("terminal.json", "runtime_qualified", True),
            ("bridge-launch.json", "parent_pid", 1), ("bridge-launch.json", "argv", ["wrong"]),
            ("bridge-close.json", "pid", 1), ("bridge-close.json", "normal_close", False),
            ("bridge-close.json", "forced_cleanup", True), ("bridge-close.json", "reaped", False),
            ("bridge-close.json", "wait_ownership_lost", True),
            ("bridge-close.json", "exit_code", False), ("bridge-close.json", "pumps_drained", False),
            ("bridge-close.json", "bank_sha256", "b"*64), ("bridge-close.json", "launch_sha256", "b"*64),
        )
        for name, key, value in cases:
            original = self.load(name); changed = deepcopy(original); changed[key] = value
            self.save(name, changed)
            with self.subTest(name=name, key=key), self.assertRaises(ValueError): self.reader(job)(bank)
            self.save(name, original)
        trace = self.load("bridge-transport.json"); closed = self.load("bridge-close.json")
        for mode in ("drop", "substitute", "missing-closed"):
            changed = deepcopy(trace)
            if mode == "drop": changed["records"].pop(0)
            if mode == "substitute": changed["records"][0]["payload"]["battleId"] = "wrong"
            if mode == "missing-closed": changed["records"].pop()
            self.save("bridge-transport.json", changed)
            self.save("bridge-close.json", dict(closed, transport_sha256=digest(changed["records"])))
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.reader(job)(bank)
        self.save("bridge-transport.json", trace); self.save("bridge-close.json", closed)
        other = deepcopy(bank); other["cells"][0]["records"][0]["payload"]["cloneId"] = "different"
        with self.assertRaises(ValueError): self.reader(job)(other)
        self.assertEqual(self.reader(job)(bank), digest(bank))

    def test_runtime_root_job_bytes_and_acceptance_callback_drift_refuse(self):
        job, receipt = self.run_job(); bank = receipt["result"]["bank"]
        payload = (self.directory / "job.pickle").read_bytes()
        (self.directory / "job.pickle").write_bytes(b"substituted")
        with self.assertRaises(ValueError): self.reader(job)(bank)
        (self.directory / "job.pickle").write_bytes(payload)
        checker = self.reader(job)
        self.snapshot.request_history["p1"][0]["active"].append(dict(drift=True))
        with self.assertRaises(ValueError): checker(bank)
        self.snapshot.request_history["p1"][0]["active"].pop()
        checkpoint = Path(self.runtime["checkpoint_path"]); before = checkpoint.read_bytes()
        checkpoint.write_bytes(b"drift")
        with self.assertRaises(ValueError): self.reader(job)(bank)
        checkpoint.write_bytes(before)
        self.directory = self.root / "acceptance-refusal"
        _, stopped = self.run_job(validate_result=lambda result: False)
        self.assertEqual(stopped["status"], "OPERATIONAL_STOP")
        self.assertEqual(stopped["worker_exit_code"], 0)  # Actual exit0 is not acceptance.
        self.assertIsNone(stopped["result"])

    def test_runtime_manifest_requires_exact_current_config_and_required_files(self):
        self.assertEqual(verify_outcome_runtime(self.runtime, self.config), digest(self.runtime))
        changed = deepcopy(self.runtime); changed["files"].pop(changed["argv"][0])
        with self.assertRaises(ValueError): verify_outcome_runtime(changed, self.config)
        changed = deepcopy(self.runtime); changed["argv"][0] += "-different"
        with self.assertRaises(ValueError): verify_outcome_runtime(changed, self.config)
        changed = deepcopy(self.runtime)
        module = next(p for p in changed["files"] if p.endswith("/search_over_raw.py"))
        changed["files"].pop(module)
        with self.assertRaisesRegex(ValueError, "inventory"): verify_outcome_runtime(changed, self.config)
        added = self.root / "showdown" / "dist" / "sim" / "new-runtime.js"
        added.write_text("module.exports = {}")
        with self.assertRaisesRegex(ValueError, "inventory"): verify_outcome_runtime(self.runtime, self.config)

    def test_result_mutating_callback_cannot_publish_accepted_terminal(self):
        def mutate(result):
            result["subject"] = "p2"
            return True
        _,receipt = self.run_job(validate_result=mutate)
        self.assertEqual(receipt["status"],"OPERATIONAL_STOP",receipt)
        self.assertEqual(receipt["worker_exit_code"],0)
        self.assertIsNone(receipt["result"])

    def test_emergency_worker_stop_is_not_normal_close_or_scoring(self):
        _, receipt = self.run_job(self.job("hang"), _synthetic_watchdog_seconds=.5)
        self.assertEqual(receipt["status"], "OUTCOME_RUN_INCOMPLETE", receipt)
        self.assertTrue(receipt["watchdog_reached"])
        self.assertEqual(receipt["supervisor_exit_code"], 124)
        self.assertNotEqual(receipt["worker_exit_code"], 0)
        self.assertFalse(receipt["cpu_wall_cost_complete"])
        self.assertIsNone(receipt["result"])

    def test_closed_acknowledgement_cannot_override_actual_nonzero_child_exit(self):
        index = self.root / "showdown" / "dist" / "sim" / "index.js"
        index.write_text("const fixtureExit = process.exit.bind(process); "
            "process.exit = code => fixtureExit(code === 0 ? 7 : code);\n" + index.read_text())
        # Freeze the modified SYNTHETIC fixture before this one-shot test launch.
        self.runtime = outcome_runtime_binding(self.config, checkpoint_path=self.runtime["checkpoint_path"])
        _, receipt = self.run_job()
        self.assertEqual(receipt["status"], "OPERATIONAL_STOP", receipt)
        self.assertEqual(receipt["worker_exit_code"], 1)
        closed = self.load("bridge-close.json")
        self.assertEqual(closed["exit_code"], 7)
        self.assertTrue(closed["reaped"]); self.assertFalse(closed["normal_close"])
        trace = self.load("bridge-transport.json")["records"]
        self.assertEqual(trace[-1], dict(kind="event", payload=dict(type="closed")))
        self.assertIsNone(receipt["result"])

    def test_already_reaped_child_loses_wait_ownership_and_is_never_signalled(self):
        from pokezero.mcts_eval.search_over_raw_outcome_run import ContinuationProcessOwner
        process = SimpleNamespace(pid=123, args=[], stdin=Mock(), returncode=None)
        owner = object.__new__(ContinuationProcessOwner)
        owner.process, owner.reaped, owner.wait_ownership_lost = process, None, False
        owner.phase, owner.bank_sha, owner.records = "started", "a"*64, []
        owner.launch, owner.runtime, owner.runtime_sha = {}, {}, "a"*64
        owner.directory, owner.ownership = self.root, dict(registration_sha256="a"*64)
        env = SimpleNamespace(_process=process, _battle_token=None, _bridge_evidence_observer=None,
            config=self.config, _send_command=Mock(), _read_event=lambda **kw: dict(type="closed"))
        owner.env = env
        with patch("pokezero.mcts_eval.search_over_raw_outcome_run.verify_outcome_runtime", return_value="a"*64), \
             patch("pokezero.mcts_eval.search_over_raw_outcome_run.os.wait4", side_effect=ChildProcessError), \
             patch("pokezero.mcts_eval.search_over_raw_outcome_run.os.kill") as kill:
            with self.assertRaises(ChildProcessError): owner.close(env)
        kill.assert_not_called()
        self.assertTrue(owner.wait_ownership_lost); self.assertIsNone(owner.reaped)
        self.assertEqual(owner.phase, "failed")
        closed = json.loads((self.root / "bridge-close.json").read_bytes())
        self.assertTrue(closed["wait_ownership_lost"])
        self.assertIsNone(closed["exit_code"]); self.assertFalse(closed["normal_close"])

    def test_canonical_champion_evaluation_is_adapted_to_unchanged_prior_sequence(self):
        # Exercise the actual production component seam with mocked load only.
        # No transformer forward, actual model weights or simulator is loaded.
        job = OwnedContinuationJob(self.config, self.snapshot, BINDING, self.runtime, "p1", {"raw": 0}, NAMESPACE)
        source = SimpleNamespace(metadata=SimpleNamespace(source_hash="source"))
        self.snapshot.belief_engine.set_source = source
        policy = SimpleNamespace(weights_sha256=self.runtime["files"][self.runtime["checkpoint_path"]],
            result=SimpleNamespace(belief_set_source_hash="source"))
        champion = Mock(return_value=([0, 1], SimpleNamespace(priors=(.25, .75), value=.9)))
        env = SimpleNamespace(belief_set_source_hash="source", close=Mock())
        with patch("pokezero.collection.env_config_with_policy_spec_masks", return_value=self.config), \
             patch("pokezero.neural_policy.require_torch", return_value=Mock()), \
             patch("pokezero.neural_policy.load_transformer_policy", return_value=policy), \
             patch("pokezero.mcts_eval.paper_reference_showdown.ChampionEvaluator", return_value=champion), \
             patch("pokezero.local_showdown.LocalShowdownEnv", return_value=env):
            actual_env, evaluator = job._components()
            self.assertIs(actual_env, env)
            self.assertEqual(evaluator("observation"), ([0, 1], (.25, .75)))
        champion.assert_called_once_with("observation")


if __name__ == "__main__": unittest.main()
