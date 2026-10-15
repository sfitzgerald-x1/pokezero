"""Prospective owning-run evidence for isolated continuation captures.

This is not collection authority or runtime qualification. The external reader
joins immutable job/launch/wait4 receipts to a separately observed bridge trace,
the exact root and runtime bytes, and strict normal close. Physics, inference,
full dependency qualification and aggregate resource cost remain separate.
Local receipt hashes detect drift; they are not operator authentication.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import time

from .search_over_raw import digest, paired_continuations, require
from .search_over_raw_outcomes import (
    ContinuationCapture, OwnedContinuationLifecycle, SCHEMA, _sha, snapshot_identity,
)
from .search_over_raw_world_inputs import _state_payload
from .paper_reference_parallel import owned_process_identity
from .search_over_raw_supervision import _save_new, _supervise_owned_job


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runtime_inventory(argv, checkpoint, extras):
    directory = Path(__file__).resolve().parent
    paths = {Path(argv[0]), Path(checkpoint), Path(argv[1]),
        *(directory.parent.rglob("*.py")), *(Path(argv[1]).parent.glob("*.mjs")),
        *(p for p in (Path(argv[3]) / "dist").rglob("*") if p.is_file()),
        *(Path(p) for p in extras)}
    return {str(p.resolve()) for p in paths}


def outcome_runtime_binding(config, *, checkpoint_path, extra_files=()):
    """Freeze current bytes for prospective review, never load a model or engine.

    All local Python sources, bridge modules and files in built Showdown dist
    are included. External/native dependency qualification is NOT established.
    """
    node = shutil.which(config.node_binary)
    require(node is not None, "explicit available node executable required")
    node = str(Path(node).resolve())
    bridge = str(config.resolved_bridge_path().resolve())
    showdown = str(config.resolved_showdown_root().resolve())
    checkpoint = str(Path(checkpoint_path).resolve())
    argv = [node, bridge, "--showdown-root", showdown]
    extras = sorted({str(Path(p).resolve()) for p in extra_files})
    paths = _runtime_inventory(argv, checkpoint, extras)
    require(str(Path(showdown) / "dist/sim/index.js") in paths
        and str(Path(showdown) / "dist/sim/state.js") in paths, "built simulator byte bindings required")
    return dict(schema="pokezero.search-over-raw.outcome-runtime.v1",
        config_sha256=digest(_state_payload(config)),
        argv=argv, checkpoint_path=checkpoint, extra_files=extras,
        files={p: _file_hash(p) for p in sorted(paths)},
        runtime_qualified=False, dependency_closure_qualified=False)


def verify_outcome_runtime(runtime, config=None):
    require(type(runtime) is dict and set(runtime) == {"schema", "config_sha256", "argv", "checkpoint_path", "extra_files",
        "files", "runtime_qualified", "dependency_closure_qualified"}
        and runtime["schema"] == "pokezero.search-over-raw.outcome-runtime.v1"
        and _sha(runtime["config_sha256"]) and runtime["runtime_qualified"] is False
        and runtime["dependency_closure_qualified"] is False, "unpromoted prospective runtime binding required")
    argv, files = runtime["argv"], runtime["files"]
    require(type(argv) is list and len(argv) == 4 and argv[2] == "--showdown-root"
        and all(type(p) is str and p and str(Path(p).resolve()) == p for p in (argv[0], argv[1], argv[3]))
        and type(files) is dict and files and type(runtime["checkpoint_path"]) is str,
        "exact executable/bridge/checkpoint binding required")
    required = {argv[0], argv[1], runtime["checkpoint_path"],
        str(Path(argv[3]) / "dist/sim/index.js"), str(Path(argv[3]) / "dist/sim/state.js"),
        str(Path(__file__).resolve()), str(Path(__file__).with_name("search_over_raw_outcomes.py").resolve()),
        str(Path(__file__).parents[1] / "local_showdown.py"),
        str(Path(argv[1]).with_name("battle_bridge_owned_continuation.mjs")),
        str(Path(argv[1]).with_name("battle_bridge_owned_snapshot.mjs"))}
    require(required <= set(files), "required owning-run/runtime files omitted")
    extras = runtime["extra_files"]
    require(type(extras) is list and all(type(p) is str and str(Path(p).resolve()) == p for p in extras)
        and extras == sorted(set(extras))
        and set(files) == _runtime_inventory(argv, runtime["checkpoint_path"], extras),
        "full declared outcome runtime inventory drift")
    require(all(type(path) is str and str(Path(path).resolve()) == path and _sha(value)
        and _file_hash(path) == value for path, value in files.items()), "outcome runtime byte drift")
    if config is not None:
        node = shutil.which(config.node_binary)
        require(node is not None and digest(_state_payload(config)) == runtime["config_sha256"]
            and [str(Path(node).resolve()), str(config.resolved_bridge_path().resolve()), "--showdown-root",
                str(config.resolved_showdown_root().resolve())] == argv, "outcome runtime config/argv drift")
    return digest(runtime)


def _redact(kind, payload):
    value = deepcopy(dict(payload))
    # Hypothetical constructor inputs/requests belong in the private witness,
    # not the separately persisted process trace.
    for field in ("players", "publicState", "boundaryRequests"):
        if field in value:
            value[field + "_sha256"] = digest(value.pop(field))
    if "snapshotJson" in value:
        value["snapshotJsonSha256"] = hashlib.sha256(value.pop("snapshotJson").encode()).hexdigest()
    if "currentSnapshotJson" in value:
        encoded = value.pop("currentSnapshotJson")
        require(type(encoded) is str and value.get("currentSnapshotJsonSha256")
            == hashlib.sha256(encoded.encode()).hexdigest(), "owned boundary snapshot byte drift")
    if "snapshot" in value:
        value["snapshot_sha256"] = digest(value.pop("snapshot"))
    if kind == "event" and value.get("type") == "stream" and value.get("stream") != "omniscient":
        value["lines_sha256"] = digest(value.pop("lines", None))
    return dict(kind=kind, payload=value)


class ContinuationProcessOwner:
    """One bridge child of the externally supervised auditor; exclusive wait4.

    Captures process-level transport separately from the producer's cell bank.
    Inherited process group keeps emergency cleanup under the external owner.
    Normal close never uses Popen.poll/wait, terminate or kill as success.
    """
    launch_schema = "pokezero.search-over-raw.outcome-bridge-launch.v1"
    close_schema = "pokezero.search-over-raw.outcome-bridge-close.v1"
    capture_hash_key = "bank_sha256"
    def __init__(self, *, directory, ownership, runtime, config):
        self.directory = Path(directory)
        self.ownership, self.runtime = deepcopy(ownership), deepcopy(runtime)
        self.config, self.runtime_sha = config, verify_outcome_runtime(runtime, config)
        require(ownership["controller_pid"] == os.getpid() and os.getpgrp() == os.getpid(),
            "external owned auditor process group required")
        self.process, self.env, self.launch, self.reaped = None, None, None, None
        self.wait_ownership_lost = False
        self.records, self.bank_sha, self.phase = [], None, "empty"

    @contextmanager
    def installed(self, env):
        require(env._process is None and env._battle_token is None
            and getattr(env, "_bridge_process_observer", None) is None and self.env is None,
            "empty exclusively owned continuation environment required")
        self.env = env
        env._bridge_process_observer = self
        try:
            yield self
        finally:
            require(env._bridge_process_observer is self, "outcome process owner changed")
            # An unreaped child remains attached until external emergency cleanup.
            if env._process is None:
                env._bridge_process_observer = None

    def started(self, env, process):
        require(self.phase == "empty" and env is self.env and self.process is None,
            "outcome bridge restart forbidden")
        self.process, self.phase = process, "started"
        verify_outcome_runtime(self.runtime, env.config)
        argv = list(process.args)
        argv[0] = str(Path(shutil.which(argv[0]) or argv[0]).resolve())
        argv[1], argv[3] = str(Path(argv[1]).resolve()), str(Path(argv[3]).resolve())
        birth = owned_process_identity(process.pid)
        require(birth and argv == self.runtime["argv"] and os.getpgid(process.pid) == os.getpid(),
            "actual outcome child/executable/group binding drift")
        self.launch = dict(schema=self.launch_schema, pid=process.pid,
            parent_pid=os.getpid(), group=os.getpid(), birth_identity=birth, argv=argv,
            binding_sha256=self.ownership["registration_sha256"], runtime_sha256=self.runtime_sha)
        _save_new(self.directory / "bridge-launch.json", self.launch)

    def command(self, payload):
        require(self.phase in ("started", "closing"), "bridge command outside owning run")
        self.records.append(_redact("command", payload))

    def event(self, payload):
        require(self.phase in ("started", "closing"), "bridge event outside owning run")
        self.records.append(_redact("event", payload))

    def seal(self, bank):
        require(self.phase == "started" and self.bank_sha is None and self.env._battle_token is None
            and self.env._bridge_evidence_observer is None, "released capture required before close")
        self._match_capture(bank)
        self.bank_sha = digest(bank)

    def _match_capture(self, bank):
        _match_transport(bank, self.records)

    def _wait(self, seconds):
        until = time.monotonic() + seconds
        while self.reaped is None:
            try:
                pid, status, usage = os.wait4(self.process.pid, os.WNOHANG)
            except ChildProcessError:
                self.wait_ownership_lost = True
                raise  # ECHILD invalidates the unreaped-child PID safety proof.
            if pid:
                self.reaped = (os.waitstatus_to_exitcode(status), usage)
                self.process.returncode = self.reaped[0]
                break
            require(time.monotonic() < until, "outcome bridge reap timed out")
            time.sleep(.01)

    def _detach(self):
        from ..local_showdown import _close_process_pipes
        env = self.env
        for thread in (env._stdout_thread, env._stderr_thread):
            if thread is not None:
                thread.join(timeout=1.)
        pumps_drained = all(thread is None or not thread.is_alive()
            for thread in (env._stdout_thread, env._stderr_thread))
        _close_process_pipes(self.process)
        env._process = env._stdout_queue = env._stdout_thread = env._stderr_thread = None
        return pumps_drained

    def close(self, env):
        require(env is self.env and self.phase == "started" and self.process is env._process,
            "one strict close of the actual outcome child required")
        primary, forced = None, False
        self.phase = "closing"
        try:
            self._require_close_ready(env)
            verify_outcome_runtime(self.runtime, env.config)
            self._normal_close_ack(env)
            self.process.stdin.close()
            self._wait(10.)
            require(self.reaped[0] == 0, "outcome bridge nonzero exit")
            verify_outcome_runtime(self.runtime, env.config)
            # Threads must drain before checking for unseen post-close output.
            for thread in (env._stdout_thread, env._stderr_thread):
                thread.join(timeout=1.)
            tail = []
            while True:
                try: tail.append(env._stdout_queue.get_nowait())
                except queue.Empty: break
            require(tail == [None], "unseen or undrained bridge output after close")
        except BaseException as error:
            primary, forced = error, True
            try:
                if self.reaped is None and not self.wait_ownership_lost:
                    # This direct child cannot reuse its PID before our wait4.
                    try: os.kill(self.process.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: self._wait(1.)
                    except ValueError:
                        try: os.kill(self.process.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                        self._wait(1.)
            except BaseException as cleanup:
                if hasattr(error, "add_note"):
                    error.add_note("Outcome process cleanup failed: " + str(cleanup))
        drained = False
        try:
            drained = self._detach() if self.reaped is not None else False
        except BaseException as cleanup:
            if primary is None:
                primary = cleanup
            elif hasattr(primary, "add_note"):
                primary.add_note("Outcome transport cleanup failed: " + str(cleanup))
        if primary is None and not drained:
            primary = ValueError("outcome bridge reader pumps remain live")
        complete = primary is None
        self.phase = "closed" if complete else "failed"
        usage = None if self.reaped is None else self.reaped[1]
        closed = dict(
            schema=self.close_schema, launch_sha256=digest(self.launch),
            **{self.capture_hash_key: self.bank_sha}, transport_sha256=digest(self.records),
            pid=self.process.pid, parent_pid=os.getpid(), binding_sha256=self.ownership["registration_sha256"],
            runtime_sha256=self.runtime_sha, normal_close=complete, forced_cleanup=forced,
            exit_code=None if self.reaped is None else self.reaped[0], reaped=self.reaped is not None,
            wait_ownership_lost=self.wait_ownership_lost,
            pumps_drained=drained, error_type=None if primary is None else type(primary).__name__,
            cpu_seconds=None if usage is None else usage.ru_utime + usage.ru_stime,
            resource_cost_complete=False, runtime_qualified=False)
        try:
            _save_new(self.directory / "bridge-transport.json", dict(records=self.records))
            _save_new(self.directory / "bridge-close.json", closed)
        except BaseException as recording:
            self.phase = "failed"
            if primary is None:
                primary = recording
            elif hasattr(primary, "add_note"):
                primary.add_note("Outcome cleanup evidence failed: " + str(recording))
        if primary is not None:
            raise primary

    def _require_close_ready(self, env):
        require(self.bank_sha is not None and env._battle_token is None
            and env._bridge_evidence_observer is None, "sealed released bank required")

    def _normal_close_ack(self, env):
        env._send_command(dict(type="close"))
        event = env._read_event(timeout=min(env.config.read_timeout_seconds, 10.))
        require(event == dict(type="closed"), "exact normal closed acknowledgement required")


def _match_transport(bank, records):
    from .search_over_raw_outcomes import RAW_LABEL_SCHEMA
    require(type(bank) is dict and bank.get("schema") in {SCHEMA, RAW_LABEL_SCHEMA} and type(bank.get("cells")) is list,
        "prospective outcome bank required")
    tokens = [r["payload"].get("battleId") for r in records
        if r["kind"] == "command" and r["payload"].get("type") == "clone_snapshot"]
    cells = {cell["battle_token"]: cell for cell in bank["cells"]}
    require(tokens and len(tokens) == len(set(tokens)) == len(cells) == len(bank["cells"])
        and set(tokens) == set(cells), "exact freshly isolated process/cell roster required")
    expected = [r for token in tokens for r in cells[token]["records"] if r["kind"] in ("command", "event")]
    require(records == expected, "bank differs from separately observed process transport")
    for token in tokens:
        rows = [r for r in cells[token]["records"] if r["kind"] in ("command", "event")]
        require(rows[0]["kind"] == "command" and rows[0]["payload"].get("type") == "clone_snapshot"
            and rows[-1]["kind"] == "event" and rows[-1]["payload"].get("type") == "clone_released"
            and rows[-1]["payload"].get("battleId") == token, "released isolated cell transport required")


@dataclass(frozen=True)
class OwnedContinuationJob:
    """Already allocated frozen auditor root, never a source environment.

    Loading the canonical champion and cloning this root require separate caller
    authority. Tests override only _components with an explicitly synthetic job.
    """
    config: object
    snapshot: object
    root_binding: dict
    runtime: dict
    subject: str
    actions: dict
    namespace: str

    def input_binding(self):
        require(type(self.root_binding) is dict and self.snapshot.terminal is None
            and self.subject in ("p1", "p2") and type(self.actions) is dict and self.actions
            and all(type(name) is str and name and type(action) is int and action >= 0
                for name, action in self.actions.items())
            and type(self.namespace) is str and self.namespace, "frozen continuation job inputs required")
        return dict(root_binding=deepcopy(self.root_binding), snapshot_identity=snapshot_identity(self.snapshot),
            source_battle_token=self.snapshot.battle_token, runtime_sha256=verify_outcome_runtime(self.runtime, self.config),
            subject=self.subject, actions=deepcopy(self.actions), namespace=self.namespace,
            max_boundaries=250, replicates=8)

    def _components(self):
        from ..local_showdown import LocalShowdownEnv
        from ..collection import env_config_with_policy_spec_masks
        from ..neural_policy import load_transformer_policy, require_torch
        from .paper_reference_showdown import ChampionEvaluator
        checkpoint = Path(self.runtime["checkpoint_path"])
        torch = require_torch()
        torch.set_num_threads(1)
        # require_torch may already apply the process-start inter-op contract.
        # Re-setting the same pool is forbidden by PyTorch; keep the required
        # one-thread law without swallowing a genuine configuration failure.
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        config = env_config_with_policy_spec_masks(self.config, [f"neural:{checkpoint}"],
            context="prospective isolated outcome auditor")
        require(digest(_state_payload(config)) == self.runtime["config_sha256"],
            "auditor config must already bind checkpoint observation provenance")
        policy = load_transformer_policy(checkpoint, device="cpu", deterministic=True,
            exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False)
        evaluator = ChampionEvaluator(policy)
        require(policy.weights_sha256 == self.runtime["files"][str(checkpoint)], "actual champion weights drift")
        env = LocalShowdownEnv(config)
        try:
            require(env.belief_set_source_hash == policy.result.belief_set_source_hash
                == self.snapshot.belief_engine.set_source.metadata.source_hash,
                "actual auditor/checkpoint/frozen catalog binding drift")
        except BaseException:
            env.close()
            raise
        def priors(observation):
            legal, evaluated = evaluator(observation)
            return legal, evaluated.priors
        return env, priors

    def __call__(self, ownership, phase):
        phase("startup")
        before = self.input_binding()
        env, evaluator = self._components()
        observer = ContinuationCapture(self.root_binding, before["runtime_sha256"])
        lifecycle = OwnedContinuationLifecycle(env, expected_snapshot=self.snapshot, root_binding=self.root_binding)
        owner = ContinuationProcessOwner(directory=Path(ownership["directory"]).parent,
            ownership=ownership, runtime=self.runtime, config=self.config)
        primary = None
        with owner.installed(env):
            try:
                phase("continuations")
                with observer.installed(env):
                    rows = paired_continuations(env=env, snapshot=self.snapshot, subject=self.subject,
                        actions=self.actions, evaluator=evaluator, namespace=self.namespace,
                        root_id=self.root_binding["root_id"], max_boundaries=250,
                        continuation_observer=observer, continuation_lifecycle=lifecycle)
                require(self.input_binding() == before, "outcome inputs changed during continuations")
                bank = observer.export()
                owner.seal(bank)
            except BaseException as error:
                primary = error
                raise
            finally:
                phase("cleanup")
                if env._process is not None:
                    try: env.close()
                    except BaseException as cleanup:
                        if primary is None: raise
                        if hasattr(primary, "add_note"):
                            primary.add_note("Outcome owner cleanup failed: " + str(cleanup))
        require(self.input_binding() == before, "outcome inputs changed after bridge reap")
        return dict(**before, bank=bank, bank_sha256=digest(bank), outcomes=rows,
            synthetic_process_fixture=getattr(self, "synthetic_process_fixture", False) is True,
            runtime_qualified=False, resource_cost_complete=False)


@dataclass(frozen=True)
class ContinuationRunContract:
    emergency_watchdog_seconds: float = 1800.

    def receipt(self):
        require(type(self.emergency_watchdog_seconds) in (int, float)
            and math.isfinite(self.emergency_watchdog_seconds) and self.emergency_watchdog_seconds > 10,
            "predeclared outcome-run emergency bound required")
        return dict(schema="pokezero.search-over-raw.outcome-run-contract.v1",
            emergency_watchdog_seconds=self.emergency_watchdog_seconds, max_boundaries=250,
            replicates=8, source_generation=False, retries_permitted=False, collection_authorized=False)


def supervise_outcome_run(job, *, directory, binding_sha256, verify_admission, validate_result,
                          contract=None, _synthetic_watchdog_seconds=None):
    """Non-collecting implementation seam; callers still need execution authority."""
    contract = ContinuationRunContract() if contract is None else contract
    require(type(contract) is ContinuationRunContract, "distinct continuation run contract required")
    contract.receipt()
    require((type(job) is OwnedContinuationJob or isinstance(job, OwnedContinuationJob)
        and getattr(job, "synthetic_process_fixture", False) is True) and callable(validate_result),
        "canonical outcome job or explicitly synthetic fixture required")
    before = job.input_binding()
    def verify():
        require(job.input_binding() == before, "outcome launch/acceptance inputs changed")
        return verify_admission()
    def validate(result):
        require(all(result.get(key) == value for key, value in before.items())
            and result["bank_sha256"] == digest(result["bank"])
            and result["runtime_qualified"] is False and result["resource_cost_complete"] is False,
            "outcome job result/input drift")
        frozen_result = digest(result)
        require(validate_result(result) is not False, "independent outcome result validator refused")
        require(digest(result) == frozen_result, "outcome result validator mutated worker evidence")
        require(job.input_binding() == before, "outcome acceptance inputs changed")
    return _supervise_owned_job(job, directory=directory, binding_sha256=binding_sha256,
        verify_admission=verify, contract=contract, kind="outcome-run", work_phase="continuations",
        validate_result=validate, _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)


class OwningRunValidator:
    """Independent reader of external launch/terminal and bridge wait witnesses.

    Inputs must be the caller's prospectively bound job/root/runtime, not values
    inferred from the bank. No missing receipt or forced cleanup can score.
    """
    contract_class = ContinuationRunContract
    supervision_schema = "pokezero.search-over-raw.outcome-run-supervision.v1"
    complete_status = "OUTCOME_RUN_COMPLETE"
    launch_schema = ContinuationProcessOwner.launch_schema
    close_schema = ContinuationProcessOwner.close_schema
    capture_hash_key = "bank_sha256"
    work_phase = "continuations"
    def __init__(self, *, directory, job_sha256, binding_sha256, root_binding,
                 expected_snapshot, runtime, allow_synthetic=False, contract=None):
        require(_sha(job_sha256) and _sha(binding_sha256), "external owning-run input bindings required")
        require(type(allow_synthetic) is bool, "explicit synthetic reader scope required")
        self.directory = Path(directory).resolve()
        self.allow_synthetic = allow_synthetic
        contract = self.contract_class() if contract is None else contract
        require(type(contract) is self.contract_class, "external distinct run contract required")
        self.contract = contract.receipt()
        self.job_sha, self.binding = job_sha256, binding_sha256
        self.root, self.snapshot, self.runtime = deepcopy(root_binding), expected_snapshot, deepcopy(runtime)
        self.inputs = digest(dict(root=self.root, snapshot=snapshot_identity(expected_snapshot),
            source_token=expected_snapshot.battle_token, runtime=self.runtime))

    def _join_result(self, result, bank, runtime_sha):
        require(result["bank"] == bank and result["bank_sha256"] == digest(bank)
            and result["root_binding"] == self.root and result["snapshot_identity"] == snapshot_identity(self.snapshot)
            and result["source_battle_token"] == self.snapshot.battle_token
            and bank["root_binding"] == self.root and bank["runtime_sha256"] == runtime_sha,
            "exact bank/root/worker evidence join required")

    def _match_capture(self, bank, records):
        _match_transport(bank, records)

    def _check_inputs(self):
        require(digest(dict(root=self.root, snapshot=snapshot_identity(self.snapshot),
            source_token=self.snapshot.battle_token, runtime=self.runtime)) == self.inputs, "owning-run inputs changed")

    def __call__(self, bank):
        self.last_validation = None
        self._check_inputs()
        runtime_sha = verify_outcome_runtime(self.runtime)
        hashes = {}
        def read(name):
            payload = (self.directory / name).read_bytes()
            hashes[name] = hashlib.sha256(payload).hexdigest()
            return json.loads(payload)
        claim, launch, worker, terminal = (read(name) for name in
            ("claim.json", "launch.json", "worker-result.json", "terminal.json"))
        bridge, closed, transport = (read(name) for name in
            ("bridge-launch.json", "bridge-close.json", "bridge-transport.json"))
        require(_file_hash(self.directory / "job.pickle") == self.job_sha == claim["job_sha256"]
            and claim["binding_sha256"] == self.binding and claim["retry_authorized"] is False
            and digest(claim["contract"]) == digest(self.contract),
            "external job/claim drift")
        from .search_over_raw_resources import measurement_protocol, resource_source_hash, validate_memory_receipt
        require(claim.get("resource_measurement_protocol") == measurement_protocol()
            and claim.get("resource_measurement_source_sha256") == resource_source_hash(), "RSS runtime protocol drift")
        pid = launch["pid"]
        require(type(pid) is int and pid > 0 and launch["group"] == pid
            and type(launch["supervisor_pid"]) is int and launch["supervisor_pid"] > 0
            and launch["supervisor_pid"] != pid
            and type(launch["birth_identity"]) is str and launch["birth_identity"]
            and launch["binding_sha256"] == self.binding
            and type(worker["worker_pid"]) is int and type(terminal["worker_pid"]) is int
            and worker["worker_pid"] == terminal["worker_pid"] == pid
            and worker["binding_sha256"] == terminal["binding_sha256"] == self.binding,
            "worker launch/wait ownership drift")
        require(terminal["schema"] == self.supervision_schema
            and terminal["status"] == self.complete_status and terminal["supervisor_exit_code"] == 0
            and all(type(value) is int and value == 0 for value in
                (terminal["supervisor_exit_code"], terminal["worker_exit_code"], worker["exit_code"]))
            and terminal["watchdog_reached"] is False and terminal["error_type"] is None
            and terminal["cleanup_error_type"] is None and worker["error_type"] is None
            and terminal["owned_groups"] == [pid] and terminal["cpu_wall_cost_complete"] is True
            and terminal["resource_cost_complete"] is False and terminal["runtime_qualified"] is False,
            "normal externally reaped outcome worker required")
        if terminal["memory_measurement"] is not None:
            require(terminal["resource_measurement_error_type"] is None, "conflicting RSS measurement failure")
            memory_payload = (self.directory / "memory-trace.json").read_bytes()
            validate_memory_receipt(self.directory,binding=self.binding,main_launch=launch,
                expected_summary=terminal["memory_measurement"])
            hashes["memory-trace.json"] = hashlib.sha256(memory_payload).hexdigest()
        else:
            require(type(terminal["resource_measurement_error_type"]) is str
                and terminal["resource_measurement_error_type"], "missing RSS error cannot be silent success")
        watchdog = terminal["watchdog_seconds"]
        require(type(watchdog) in (int, float) and math.isfinite(watchdog) and watchdog > 0
            and (watchdog == self.contract["emergency_watchdog_seconds"]
                or claim["synthetic_process_fixture"] is True and watchdog < self.contract["emergency_watchdog_seconds"]),
            "actual outcome watchdog differs from external contract")
        from .search_over_raw_resources import read_completed_process_cost
        costs = read_completed_process_cost(terminal, worker, work_phase=self.work_phase)
        result = terminal["result"]
        require(digest(worker["result"]) == digest(result)
            and type(claim["synthetic_process_fixture"]) is bool
            and result["synthetic_process_fixture"] is claim["synthetic_process_fixture"],
            "exact bank/root/worker evidence join required")
        self._join_result(result, bank, runtime_sha)
        require(bridge["schema"] == self.launch_schema
            and type(bridge["pid"]) is int and bridge["pid"] > 0 and bridge["pid"] != pid
            and bridge["parent_pid"] == bridge["group"] == closed["parent_pid"] == pid
            and type(bridge["birth_identity"]) is str and bridge["birth_identity"]
            and bridge["argv"] == self.runtime["argv"]
            and bridge["binding_sha256"] == closed["binding_sha256"] == self.binding
            and bridge["runtime_sha256"] == closed["runtime_sha256"] == runtime_sha
            and closed["schema"] == self.close_schema
            and closed["pid"] == bridge["pid"] and closed["launch_sha256"] == digest(bridge)
            and closed[self.capture_hash_key] == digest(bank) and closed["normal_close"] is True
            and closed["forced_cleanup"] is False and closed["reaped"] is True
            and closed["wait_ownership_lost"] is False
            and type(closed["exit_code"]) is int and closed["exit_code"] == 0 and closed["pumps_drained"] is True
            and closed["error_type"] is None and closed["resource_cost_complete"] is False
            and closed["runtime_qualified"] is False, "strict same-child normal close required")
        records = transport["records"]
        require(closed["transport_sha256"] == digest(records)
            and records[-2:] == [dict(kind="command", payload=dict(type="close")),
                dict(kind="event", payload=dict(type="closed"))], "actual close transport required")
        self._match_capture(bank, records[:-2])
        self._check_inputs()
        verify_outcome_runtime(self.runtime)
        require(all(_file_hash(self.directory / name) == value for name, value in hashes.items())
            and _file_hash(self.directory / "job.pickle") == self.job_sha, "owning-run receipts changed during validation")
        self.last_validation = dict(schema="pokezero.search-over-raw.owning-process-readout.v1",
            capture_sha256=digest(bank), receipt_file_sha256=hashes, costs=costs,
            main_launch=deepcopy(launch), synthetic_process_fixture=claim["synthetic_process_fixture"],
            process_receipts_validated=True, resource_cost_complete=False, runtime_qualified=False,
            scientific_collection_authorized=False)
        return digest(bank) if not claim["synthetic_process_fixture"] or self.allow_synthetic else None
