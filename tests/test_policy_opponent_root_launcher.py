"""Portable launcher boundary checks; fixtures are not real image/profile proof."""

from copy import deepcopy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_policy_opponent_root_profile as launch
from tests.test_policy_opponent_registration import registration
from pokezero.mcts_eval.resolver import ContractError


def image_fixture():
    return {"schema_version": "pokezero.b2-source-image-receipt.v8", "complete": True,
        "immutable_image": "registry/example@sha256:" + "a" * 64, "image_digest": "sha256:" + "a" * 64,
        "source_commit": "b" * 40, "model_runtime": {"schema_version": "pokezero.b2-model-runtime.v4",
            "copied_runtime_probe": True, "system_site_packages": True,
            "runtime_path": "/opt/pokezero-b2-model-runtime", "python_version": "3.14.0",
            "source": {"commit": "b" * 40, "tree_sha256": "c" * 64, "tree_status": "clean_tracked_checkout"},
            "torch": {"version": "fixture", "module_sha256": "d" * 64}, "engine_fingerprint": "e" * 64,
            "native_extensions": [{"module": name, "relative_path": f"lib/{name}/native.so", "sha256": "f" * 64}
                                  for name in ("poke_engine", "pokezero_search")]}}


def image_args(base, payload):
    path = base / "receipt.json"
    path.write_text(json.dumps(payload))
    return SimpleNamespace(image_receipt=str(path), image_receipt_sha256=launch.sha256_file(path))


def execution_fixture(base):
    roster, payload = registration()
    reg_sha = launch.immutable_json(base / "REGISTRATION.json", payload)
    preflight = {"schema_version": "pokezero.paper-policy-opponent-live-replay.v1",
        "root_denominator": 32, "replacement_roots": [], "roots": [
            {"decision_id": row["decision_id"], "state": "LIVE_PUBLIC_REPLAY_VALID",
             "requested_players": ["p1", "p2"]} for row in roster["profile_roots"]]}
    live_sha = launch.immutable_json(base / "LIVE_REPLAY.json", preflight)
    start = {"study_started_unix_s": time.time(), "registration_sha256": reg_sha,
             "live_replay_sha256": live_sha, "reserved_cpus": 2, "includes_initialization": True}
    start_sha = launch.immutable_json(base / "STUDY_START.json", start)
    return roster, payload, preflight, start, SimpleNamespace(registration_sha256=reg_sha, study_start_sha256=start_sha)


class RootLauncherTests(unittest.TestCase):
    def run_fixture(self, base, *, first_exit=0, first_timeout=False, final_drift=False):
        """Execute the real publication/control loop, but no policies or images."""
        roster, payload = registration()
        runtime = deepcopy(payload["expected_runtime"])
        runtime.pop("model_sha256")
        runtime.pop("tables_sha256")
        checkpoint = base / "fixture.pt"
        checkpoint.write_bytes(b"NOT A CHAMPION")
        out = base / "new-study"
        args = SimpleNamespace(out_root=str(out), roster="fixture-roster", source_root="fixture-source",
            checkpoint=str(checkpoint), showdown_root="fixture-showdown", ownership_token="fixture-token")
        preflight = {"schema_version": "pokezero.paper-policy-opponent-live-replay.v1",
            "root_denominator": 32, "replacement_roots": [], "roots": [
                {"decision_id": row["decision_id"], "state": "REFUSED", "phase": "fixture-refusal",
                 "refusal": "deliberate synthetic input refusal"} for row in roster["profile_roots"]]}
        def create_worker(command):
            ordinal = int(command[command.index("--worker-ordinal") + 1])
            source = roster["profile_roots"][ordinal]
            reg_sha = command[command.index("--registration-sha256") + 1]
            if ordinal == 0 and first_timeout:
                launch.immutable_json(out / "measurements/00/fixed_work-raw_policy.json", {"fixture_partial": True})
            else:
                launch.immutable_json(out / f"roots/{ordinal:02d}.json", {"registration_sha256": reg_sha,
                    "source": source, "result": {"decision_id": source["decision_id"], "state": "REFUSED", "modes": {}}})
            return SimpleNamespace(pid=ordinal, returncode=first_exit if ordinal == 0 else 0)
        contract = SimpleNamespace(checkpoint_path=out / "inputs/champion.pt")
        source = {"commit": runtime["source_commit"], "tree_sha256": "d" * 64 if final_drift else runtime["source_tree_sha256"]}
        with ExitStack() as stack:
            patches = {"bound_inputs": Mock(return_value=(roster, contract, runtime)),
                "resolve_checkpoint_contract": Mock(return_value=contract),
                "materialize_search_artifacts": Mock(return_value={"model_path": "fixture-model", "tables_path": "fixture-tables"}),
                "env_config_with_policy_spec_masks": Mock(return_value=object()),
                "LocalShowdownEnv": Mock(return_value=unittest.mock.MagicMock()),
                "qualify_live_replay": Mock(return_value=preflight), "verify_source_files": Mock(),
                "launch_root_process": Mock(side_effect=create_worker),
                "communicate_bounded": Mock(side_effect=lambda process, _: (b"", b"", first_timeout and process.pid == 0)),
                "verify_image_runtime": Mock(return_value=({"immutable_image": runtime["immutable_image"]}, source,
                    runtime["engine_fingerprint"], {})),
                "sha256_file": Mock(side_effect=lambda path: launch.CHAMPION_SHA256 if str(path).endswith("champion.pt") else "a" * 64)}
            for name, replacement in patches.items():
                stack.enter_context(patch.object(launch, name, replacement))
            stack.enter_context(patch.object(launch, "PROCESS_STARTED_UNIX", time.time()))
            stack.enter_context(patch.object(launch, "PROCESS_STARTED_MONOTONIC", time.monotonic()))
            launch.run(args)
        return out

    def test_immutable_writer_never_replaces_existing_bytes_or_leaves_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unit.json"
            digest = launch.immutable_json(path, {"first": 1})
            raw = path.read_bytes()
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            with self.assertRaises(FileExistsError):
                launch.immutable_json(path, {"second": 2})
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(list(path.parent.iterdir()), [path])
            with self.assertRaises(ValueError):
                launch.immutable_json(path.parent / "nan.json", {"value": float("nan")})
            self.assertFalse((path.parent / "nan.json").exists())

    def test_receipt_requires_external_byte_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            args = image_args(Path(directory), image_fixture())
            self.assertEqual(launch.load_image_receipt(args), image_fixture())
            Path(args.image_receipt).write_bytes(Path(args.image_receipt).read_bytes() + b"\n")
            with self.assertRaisesRegex(ContractError, "byte hash drift"):
                launch.load_image_receipt(args)

    def test_incomplete_receipts_refuse_before_importing_or_loading_runtime(self):
        mutations = [lambda p: p.update(complete=1), lambda p: p.update(immutable_image="registry/image:tag"),
            lambda p: p.update(image_digest="sha256:" + "b" * 64), lambda p: p.update(model_runtime=[]),
            lambda p: p["model_runtime"].update(source={}), lambda p: p["model_runtime"].update(torch=None),
            lambda p: p["model_runtime"].update(runtime_path="/tmp/pretend"),
            lambda p: p["model_runtime"].update(copied_runtime_probe=False),
            lambda p: p["model_runtime"]["source"].update(tree_status="dirty"),
            lambda p: p.update(source_commit="c" * 40)]
        with tempfile.TemporaryDirectory() as directory:
            for mutate in mutations:
                payload = image_fixture()
                mutate(payload)
                args = image_args(Path(directory), payload)
                with patch.object(launch, "_source_provenance") as source:
                    with self.assertRaises(ContractError):
                        launch.verify_image_runtime(args)
                    source.assert_not_called()

    def test_both_native_extension_bindings_and_safe_paths_are_required(self):
        mutations = [lambda rows: rows.pop(), lambda rows: rows[0].pop("relative_path"),
            lambda rows: rows[0].update(relative_path="../escape.so"),
            lambda rows: rows[0].update(relative_path="/tmp/native.so"),
            lambda rows: rows[0].update(sha256=None), lambda rows: rows[0].update(module=[]),
            lambda rows: rows[1].update(module="poke_engine")]
        with tempfile.TemporaryDirectory() as directory:
            for mutate in mutations:
                payload = image_fixture()
                mutate(payload["model_runtime"]["native_extensions"])
                with self.assertRaisesRegex(ContractError, "native consumers"):
                    launch.load_image_receipt(image_args(Path(directory), payload))

    def test_attested_receipt_is_not_proof_of_current_platform(self):
        payload = image_fixture()
        with tempfile.TemporaryDirectory() as directory:
            args = image_args(Path(directory), payload)
            with patch.object(launch, "_source_provenance", return_value=payload["model_runtime"]["source"]), \
                    patch.object(launch.platform, "system", return_value="Darwin"), \
                    patch.object(launch.importlib, "import_module") as native:
                with self.assertRaisesRegex(ContractError, "platform/interpreter"):
                    launch.verify_image_runtime(args)
                native.assert_not_called()

    def test_study_clock_and_replay_have_independent_external_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roster, payload, preflight, start, args = execution_fixture(base)
            self.assertEqual(launch.load_execution_evidence(base, args, roster), (payload, preflight, start))
            for filename, reason in (("STUDY_START.json", "study-start"), ("LIVE_REPLAY.json", "qualification"),
                                     ("REGISTRATION.json", "registration")):
                path = base / filename
                raw = path.read_bytes()
                path.write_bytes(raw + b"\n")
                with self.assertRaisesRegex(ContractError, reason):
                    launch.load_execution_evidence(base, args, roster)
                path.write_bytes(raw)

    def test_replay_panel_and_start_malformed_bindings_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roster, _, preflight, start, args = execution_fixture(base)
            mutations = [lambda p: p.update(root_denominator=True), lambda p: p["roots"].pop(),
                lambda p: p["roots"].reverse(), lambda p: p.update(replacement_roots=["new"]),
                lambda p: p["roots"][0].update(requested_players=[]), lambda p: p["roots"][0].update(state="PASS")]
            for mutate in mutations:
                changed = deepcopy(preflight)
                mutate(changed)
                path = base / "LIVE_REPLAY.json"
                path.write_text(json.dumps(changed))
                changed_start = {**start, "live_replay_sha256": launch.sha256_file(path)}
                (base / "STUDY_START.json").write_text(json.dumps(changed_start))
                args.study_start_sha256 = launch.sha256_file(base / "STUDY_START.json")
                with self.assertRaises(ContractError):
                    launch.load_execution_evidence(base, args, roster)
            (base / "LIVE_REPLAY.json").write_text(json.dumps(preflight))
            start["live_replay_sha256"] = launch.sha256_file(base / "LIVE_REPLAY.json")
            for field, value in (("reserved_cpus", True), ("registration_sha256", "a" * 64),
                                 ("study_started_unix_s", time.time() + 100), ("includes_initialization", False)):
                changed = {**start, field: value}
                (base / "STUDY_START.json").write_text(json.dumps(changed))
                args.study_start_sha256 = launch.sha256_file(base / "STUDY_START.json")
                with self.assertRaises(ContractError):
                    launch.load_execution_evidence(base, args, roster)

    def test_terminal_identity_and_all_six_arm_dispositions_are_required(self):
        roster, _ = registration()
        source = roster["profile_roots"][0]
        unit = {"registration_sha256": "a" * 64, "source": source, "result": {
            "decision_id": source["decision_id"], "state": "COMPLETE", "modes": {
                mode: {arm: {"state": "COMPLETE"} for arm in launch.ARMS} for mode in launch.MODES}}}
        self.assertEqual(launch.validate_terminal(unit, source=source, registration_sha="a" * 64), unit["result"])
        mutations = [lambda p: p["result"].update(decision_id="b" * 64), lambda p: p.update(source={}),
            lambda p: p["result"].update(modes={}), lambda p: p["result"]["modes"]["fixed_work"].pop("raw_policy"),
            lambda p: p["result"]["modes"]["fixed_work"]["raw_policy"].update(state="REFUSED"),
            lambda p: p["result"].update(state="REFUSED")]
        for mutate in mutations:
            changed = deepcopy(unit)
            mutate(changed)
            with self.assertRaises(ContractError):
                launch.validate_terminal(changed, source=source, registration_sha="a" * 64)

    def test_timeout_preserves_late_terminal_and_completed_arm_receipts(self):
        roster, _ = registration()
        source = roster["profile_roots"][0]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "roots/00.json"
            arm = base / "measurements/00/fixed_work-raw_policy.json"
            launch.immutable_json(root, {"late": "terminal"})
            launch.immutable_json(arm, {"completed": "arm"})
            original = {path: path.read_bytes() for path in (root, arm)}
            launch.record_timeout(base, ordinal=0, source=source, registration_sha="a" * 64)
            self.assertEqual({path: path.read_bytes() for path in (root, arm)}, original)
            self.assertTrue(json.loads((base / "caps/00.json").read_bytes())["partial_arm_receipts_retained"])
            launch.record_timeout(base, ordinal=1, source=roster["profile_roots"][1], registration_sha="a" * 64)
            self.assertEqual(json.loads((base / "roots/01.json").read_bytes())["result"]["state"], "CAP_UNRESOLVED")

    def test_communicate_handles_exit_races_and_escalates_only_owned_group(self):
        process = Mock(pid=43210)
        process.communicate.side_effect = [subprocess.TimeoutExpired("owned", 1), (b"done", b"")]
        with patch.object(launch.os, "killpg", side_effect=ProcessLookupError) as kill:
            self.assertEqual(launch.communicate_bounded(process, 1), (b"done", b"", True))
            kill.assert_called_once_with(43210, signal.SIGTERM)
        process.communicate.side_effect = [subprocess.TimeoutExpired("owned", 1),
            subprocess.TimeoutExpired("owned", 5), (b"killed", b"")]
        with patch.object(launch.os, "killpg") as kill:
            self.assertEqual(launch.communicate_bounded(process, 1), (b"killed", b"", True))
            self.assertEqual([call.args for call in kill.call_args_list], [(43210, signal.SIGTERM), (43210, signal.SIGKILL)])

    def test_outer_cap_really_terminates_and_reaps_nested_owned_session(self):
        script = ("import sys,signal,time; sys.path[:0]=sys.argv[1:]; "
            "import run_policy_opponent_root_profile as l; "
            "signal.signal(signal.SIGTERM,l.terminate_owned_root); "
            "p=l.launch_root_process([sys.executable,'-c','import time; time.sleep(30)']); "
            "print(p.pid,flush=True); time.sleep(30)")
        process = subprocess.Popen([sys.executable, "-S", "-c", script, str(launch.ROOT / "src"),
            str(launch.ROOT / "scripts")], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        stdout, stderr, timed_out = launch.communicate_bounded(process, 2)
        self.assertTrue(timed_out)
        self.assertEqual(process.returncode, 143, stderr.decode())
        nested_pid = int(stdout.strip())
        with self.assertRaises(ProcessLookupError):
            os.kill(nested_pid, 0)

    def test_interrupt_during_observation_or_grace_cleans_up_owned_process(self):
        for first in ([KeyboardInterrupt(), (b"", b"")],
                      [subprocess.TimeoutExpired("owned", 1), KeyboardInterrupt(), (b"", b"")]):
            process = Mock(pid=43210)
            process.poll.return_value = None
            process.communicate.side_effect = first
            with patch.object(launch.os, "killpg") as kill:
                with self.assertRaises(KeyboardInterrupt):
                    launch.communicate_bounded(process, 1)
                self.assertTrue(kill.called)
                self.assertTrue(all(call.args == (43210, signal.SIGTERM) for call in kill.call_args_list))

    def test_reaped_root_is_never_signalled_by_study_cleanup(self):
        process = Mock(pid=43210)
        process.poll.return_value = 0
        with patch.object(launch, "ACTIVE_ROOT_PROCESS", process), patch.object(launch.os, "killpg") as kill:
            with self.assertRaises(SystemExit):
                launch.terminate_owned_root(signal.SIGTERM, None)
            kill.assert_not_called()

    def test_actual_cli_bootstrap_releases_inherited_termination_mask(self):
        code = """import runpy, signal, sys
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
sys.argv = [sys.argv[1], '--help']
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as error:
    assert error.code == 0
print('SIGTERM_BLOCKED', signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, set()))
"""
        result = subprocess.run([sys.executable, "-S", "-c", code,
            str(launch.ROOT / "scripts/run_policy_opponent_root_profile.py")], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIGTERM_BLOCKED False", result.stdout)

    def test_existing_output_root_refuses_before_any_runtime_work(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out_root=directory)
            with patch.object(launch, "bound_inputs") as bound:
                for function in (launch.run, launch.supervise):
                    with self.assertRaisesRegex(ContractError, "already exists"):
                        function(args)
                bound.assert_not_called()

    def test_real_control_loop_preserves_original_denominator_and_partial_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            out = self.run_fixture(Path(directory), first_timeout=True, first_exit=143)
            readout = json.loads((out / "PROFILE_READOUT.json").read_bytes())
            self.assertEqual(len(list((out / "roots").glob("*.json"))), 32)
            self.assertEqual(len(readout["root_dispositions"]), 32)
            self.assertEqual(readout["root_dispositions"][0]["state"], "CAP_UNRESOLVED")
            self.assertEqual(readout["timed_out_arm_receipts"], ["measurements/00/fixed_work-raw_policy.json"])
            self.assertEqual(readout["replacement_roots"], [])
            self.assertFalse(readout["strength_qualified"])
            self.assertFalse(readout["advance_authorized"])
            for mode in readout["cells"].values():
                for cell in mode.values():
                    self.assertEqual((cell["completed"], cell["uncompleted"], cell["root_denominator"]), (0, 32, 32))
                    self.assertIsNone(cell["p95"])
            self.assertIsNone(launch.ACTIVE_ROOT_PROCESS)

    def test_worker_nonzero_or_final_source_drift_never_creates_readout(self):
        for options, reason in (({"first_exit": 17}, "nonzero: 17"), ({"final_drift": True}, "binding drift")):
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                with self.assertRaisesRegex(ContractError, reason):
                    self.run_fixture(base, **options)
                self.assertFalse((base / "new-study/PROFILE_READOUT.json").exists())
                self.assertTrue((base / "new-study/roots/00.json").is_file())
                if options.get("first_exit"):
                    self.assertEqual(json.loads((base / "new-study/logs/00.json").read_bytes())["exit_code"], 17)

    def test_both_handled_signals_are_masked_until_root_ownership_publication(self):
        before = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        def spawn(*args, **kwargs):
            current = signal.pthread_sigmask(signal.SIG_BLOCK, set())
            self.assertTrue({signal.SIGTERM, signal.SIGINT}.issubset(current))
            return SimpleNamespace(pid=123)
        with patch.object(launch.subprocess, "Popen", side_effect=spawn), patch.object(launch, "ACTIVE_ROOT_PROCESS", None):
            process = launch.launch_root_process(["fixture"])
            self.assertIs(launch.ACTIVE_ROOT_PROCESS, process)
        self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, set()), before)

    def test_real_outer_sigterm_cascades_into_study_and_nested_root(self):
        study = ("import sys,signal,time; import run_policy_opponent_root_profile as l; "
            "signal.signal(signal.SIGTERM,l.terminate_owned_root); "
            "p=l.launch_root_process([sys.executable,'-c','import time; time.sleep(30)']); "
            "print(p.pid,flush=True); time.sleep(30)")
        outer = """import sys,signal,subprocess,time
sys.path[:0] = sys.argv[1:3]
import run_policy_opponent_root_profile as l
signal.signal(signal.SIGTERM, l.terminate_owned_study)
l.ACTIVE_STUDY_PROCESS = subprocess.Popen([sys.executable, '-S', '-c', 'import sys; sys.path[:0]=' + repr(sys.path[:2]) + ';' + sys.argv[3]], stdout=subprocess.PIPE, start_new_session=True)
print(l.ACTIVE_STUDY_PROCESS.pid, flush=True)
print(l.ACTIVE_STUDY_PROCESS.stdout.readline().decode().strip(), flush=True)
time.sleep(30)
"""
        process = subprocess.Popen([sys.executable, "-S", "-c", outer, str(launch.ROOT / "src"),
            str(launch.ROOT / "scripts"), study], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        stdout, stderr, timed_out = launch.communicate_bounded(process, 2)
        self.assertTrue(timed_out)
        self.assertEqual(process.returncode, 143, stderr.decode())
        self.assertEqual(len(stdout.splitlines()), 2)
        for line in stdout.splitlines():
            with self.assertRaises(ProcessLookupError):
                os.kill(int(line), 0)


if __name__ == "__main__":
    unittest.main()
