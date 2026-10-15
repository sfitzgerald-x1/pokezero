"""Independent reader of fixed-work ownership and actual accepted-work receipts.

This runs no source, simulator or model and never deserializes a pickle. Caller-supplied job and file
bindings must be prospective, not copied from the result being checked. Native
FFI runs inside the isolated Python job; it is not a bridge child. Reference
workers are independently declared and normally closed process generations.
Declared file bytes are checked, not a loaded-binary dependency closure. Neither
this reader nor complete CPU attribution qualifies a runtime or simulator.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .search_over_raw import digest, require
from .search_over_raw_fixed_work import FixedWorkDiagnosticContract
from .search_over_raw_outcomes import _sha
from .search_over_raw_resources import (
    derive_fixed_work_accounting, measurement_protocol, resource_source_hash,
    validate_memory_receipt,
)
from .search_over_raw_supervision import _phase_costs
from .search_over_raw_world_inputs import validate_fixed_work_inputs, valid_catalog_fingerprint


def fixed_work_reader_source_paths():
    """Enumerate current required paths afresh; no byte-validation shortcut."""
    from .search_over_raw_reconciled import controller_input_paths
    required = {str(path) for path in controller_input_paths()}
    directory = Path(__file__).resolve().parent
    paths = list(directory.parent.rglob("*.py"))
    paths += list((directory.parents[2] / "scripts").glob("*.mjs"))
    paths += list((directory.parents[2] / "rust/pokezero-search/src").glob("*.rs"))
    required.update(str(path.resolve()) for path in paths)
    return required


def fixed_work_reader_source_hashes():
    """Required controller bytes; models/dependencies are additional bindings."""
    return {path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in fixed_work_reader_source_paths()}


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class FixedWorkRunValidator:
    """One exact selected receipt, joined to its external owned run.

    Re-reading completed evidence never queries or signals historical PIDs.
    Missing or nonzero exits are failures, never inferred from disappearance.
    Synthetic fixtures return None unless explicitly opted into test scope.
    """
    def __init__(self, *, directory, job_sha256, binding_sha256, configuration,
                 root_id, statistics_root_id, selection_seed, runtime_sha256,
                 consumed_binding, file_hashes, checkpoint_path, checkpoint_sha256,
                 showdown_root, allow_synthetic=False, contract=None, public_request_sha256=None,
                 native_public_fold_required=False):
        from .search_over_raw import SearchConfiguration
        require(type(configuration) is SearchConfiguration
            and configuration.arm in {"incumbent", "reference"}
            and configuration.belief == "public" and configuration.seconds == 1.
            and configuration.workers == (20 if configuration.arm == "reference" else 1),
            "canonical fixed-work configuration required")
        require(all(_sha(value) for value in (job_sha256, binding_sha256, runtime_sha256))
            and type(statistics_root_id) is str and bool(statistics_root_id)
            and root_id == statistics_root_id + ":" + configuration.identity
            and type(selection_seed) is int and type(allow_synthetic) is bool,
            "external fixed-work input bindings required")
        require(type(consumed_binding) is dict and set(consumed_binding) == {
            "root_binding", "subject", "set_source_hash", "information_key"}
            and _sha(consumed_binding["root_binding"])
            and valid_catalog_fingerprint(consumed_binding["set_source_hash"])
            and consumed_binding["subject"] in {"p1", "p2"}
            and type(consumed_binding["information_key"]) is str
            and bool(consumed_binding["information_key"]), "external consumed root required")
        contract = FixedWorkDiagnosticContract() if contract is None else contract
        require(type(contract) is FixedWorkDiagnosticContract, "distinct fixed-work contract required")
        self.directory = Path(directory).resolve()
        self.job_sha, self.binding, self.configuration = job_sha256, binding_sha256, configuration
        self.root_id, self.statistics_root_id, self.seed = root_id, statistics_root_id, selection_seed
        self.runtime_sha, self.consumed = runtime_sha256, deepcopy(consumed_binding)
        self.file_hashes, self.allow_synthetic = deepcopy(file_hashes), allow_synthetic
        require(type(checkpoint_path) is str and str(Path(checkpoint_path).resolve()) == checkpoint_path
            and _sha(checkpoint_sha256) and type(showdown_root) is str
            and str(Path(showdown_root).resolve()) == showdown_root, "external model/simulator file scope required")
        self.checkpoint_path, self.checkpoint_sha = checkpoint_path, checkpoint_sha256
        self.showdown_root = showdown_root
        self.contract = contract
        require(public_request_sha256 is None or _sha(public_request_sha256),
            "prospective full public request binding required")
        self.public_request_sha256 = public_request_sha256
        require(type(native_public_fold_required) is bool and (not native_public_fold_required or
            configuration.arm == "incumbent" and configuration.leaf == "model" and public_request_sha256 is not None),
            "prospective public-fold owning-reader scope required")
        self.native_public_fold_required = native_public_fold_required
        self.expected = dict(root_id=root_id, statistics_root_id=statistics_root_id,
            configuration_sha256=configuration.identity, selection_seed=selection_seed,
            runtime_sha256=runtime_sha256)
        self.frozen = self._inputs()
        self.last_validation = None
        self._files()

    def _files(self):
        import sys
        required = fixed_work_reader_source_hashes()
        dist = Path(self.showdown_root) / "dist"
        simulator = [path for path in dist.rglob("*") if path.is_file()]
        require((dist / "sim/index.js").is_file() and (dist / "sim/state.js").is_file(),
            "built simulator binding files required")
        required.update({str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in simulator})
        required[self.checkpoint_path] = self.checkpoint_sha
        executable = str(Path(sys.executable).resolve())
        required[executable] = hashlib.sha256(Path(executable).read_bytes()).hexdigest()
        require(type(self.file_hashes) is dict and required.items() <= self.file_hashes.items()
            and all(type(path) is str and str(Path(path).resolve()) == path and _sha(value)
                and hashlib.sha256(Path(path).read_bytes()).hexdigest() == value
                for path, value in self.file_hashes.items()), "declared fixed-work file byte drift")

    def _inputs(self):
        return digest(dict(expected=self.expected, consumed=self.consumed, files=self.file_hashes,
            public_request_sha256=self.public_request_sha256,
            native_public_fold_required=self.native_public_fold_required,
            checkpoint_path=self.checkpoint_path, checkpoint_sha256=self.checkpoint_sha,
            showdown_root=self.showdown_root))

    def _selected_result(self, result):
        require("owned_reference_value_diagnostic" not in result,
            "owned reference label results require their distinct completed reader")
        return result

    def __call__(self, selected):
        self.last_validation = None
        require(self._inputs() == self.frozen, "fixed-work reader inputs changed")
        self._files()
        raw_saved = deepcopy(selected)
        raw_sha = digest(raw_saved)
        saved = deepcopy(self._selected_result(raw_saved))
        selected_sha = digest(saved)
        hashes = {}
        def read(name):
            payload = (self.directory / name).read_bytes()
            hashes[name] = hashlib.sha256(payload).hexdigest()
            return json.loads(payload)
        claim, launch, worker, terminal = (read(name) for name in
            ("claim.json", "launch.json", "worker-result.json", "terminal.json"))
        require(hashlib.sha256((self.directory / "job.pickle").read_bytes()).hexdigest()
            == self.job_sha == claim["job_sha256"] and claim["binding_sha256"] == self.binding
            and claim["contract"] == self.contract.receipt() and claim["retry_authorized"] is False
            and type(claim["synthetic_process_fixture"]) is bool
            and claim["resource_measurement_protocol"] == measurement_protocol()
            and claim["resource_measurement_source_sha256"] == resource_source_hash(),
            "external fixed-work job/claim/protocol drift")
        pid = launch["pid"]
        require(type(pid) is int and pid > 0 and type(launch["group"]) is int and launch["group"] == pid
            and type(launch["supervisor_pid"]) is int and launch["supervisor_pid"] > 0
            and launch["supervisor_pid"] != pid and type(launch["birth_identity"]) is str
            and bool(launch["birth_identity"]) and launch["binding_sha256"] == self.binding
            and all(type(row["worker_pid"]) is int and row["worker_pid"] == pid
                and row["binding_sha256"] == self.binding for row in (worker, terminal)),
            "fixed-work actual worker launch/terminal ownership drift")
        require(terminal["schema"] == "pokezero.search-over-raw.fixed-work-supervision.v1"
            and terminal["status"] == "DIAGNOSTIC_COMPLETE"
            and all(type(value) is int and value == 0 for value in
                (terminal["supervisor_exit_code"], terminal["worker_exit_code"], worker["exit_code"]))
            and terminal["watchdog_reached"] is False and terminal["error_type"] is None
            and terminal["cleanup_error_type"] is None and worker["error_type"] is None
            and terminal["cpu_wall_cost_complete"] is True
            and terminal["resource_cost_complete"] is False and terminal["peak_tree_memory_measured"] is False
            and terminal["deployment_eligible"] is False and terminal["runtime_qualified"] is False
            and terminal["retries_permitted"] is False and terminal["holdout_authorized"] is False,
            "normal externally reaped fixed-work completion required")
        watchdog = terminal["watchdog_seconds"]
        require(_finite_nonnegative(watchdog) and watchdog > 0
            and (watchdog == self.contract.emergency_watchdog_seconds
                or claim["synthetic_process_fixture"] and watchdog < self.contract.emergency_watchdog_seconds)
            and all(_finite_nonnegative(terminal[key]) for key in
                ("elapsed_wall_seconds", "cpu_seconds_including_reaped_children", "supervisor_thread_cpu_seconds", "maxrss"))
            and terminal["cpu_scope"] == "isolated job wait4; reaped descendants only"
            and terminal["maxrss_units"] in {"bytes", "KiB"}, "actual fixed-work CPU/watchdog witness required")
        costs = _phase_costs(worker["phases"], worker["terminal_monotonic_seconds"])
        reported = terminal["phase_costs"]
        require(type(reported) is dict and set(reported) == set(costs) | {
            "process_bootstrap_wall_seconds", "receipt_reap_and_verification_wall_seconds"}
            and all(reported[key] == value for key, value in costs.items())
            and all(_finite_nonnegative(value) for value in reported.values())
            and sum(costs.values()) + reported["process_bootstrap_wall_seconds"] <= watchdog
            and sum(reported.values()) <= terminal["elapsed_wall_seconds"], "fixed-work phase accounting drift")
        owned = self.directory / "owned-groups"
        paths = sorted(owned.glob("group-*.json"))
        require(not list((owned / "active").glob("group-*.json"))
            and len(paths) == (20 if self.configuration.arm == "reference" else 0),
            "fixed-work generation roster or active cleanup drift")
        pids = []
        for path in paths:
            child, closed = read("owned-groups/" + path.name), read("owned-groups/closed-" + path.name)
            child_pid = child["pid"]
            require(type(child_pid) is int and child_pid > 0 and child_pid != pid
                and type(child["group"]) is int and child["group"] == child_pid
                and child["controller_pid"] == pid and child["registration_sha256"] == self.binding
                and child["protocol"] == "active-generations-v1"
                and type(child["birth_identity"]) is str and bool(child["birth_identity"])
                and type(child["generation"]) is str and bool(child["generation"])
                and path.name == f"group-{child_pid}-{child['generation']}.json"
                and closed["launch_sha256"] == hashes["owned-groups/" + path.name]
                and closed["pid"] == child_pid and closed["controller_pid"] == pid
                and closed["generation"] == child["generation"] and closed["group_live"] is False
                and type(closed["worker_exit_code"]) is int and closed["worker_exit_code"] == 0,
                "fixed-work normal same-generation close required")
            pids.append(child_pid)
        # The terminal's owned_groups is the active cleanup set AFTER normal
        # generations were retired. It is not the historical cost population.
        # Retired generations remain independently checked above and in RSS.
        require(len(set(pids)) == len(pids) and terminal["owned_groups"] == [pid],
            "fixed-work owned process population drift")
        if terminal["memory_measurement"] is not None:
            require(terminal["resource_measurement_error_type"] is None, "conflicting fixed-work RSS failure")
            read("memory-trace.json")
            validate_memory_receipt(self.directory, binding=self.binding, main_launch=launch,
                expected_summary=terminal["memory_measurement"])
        else:
            require(type(terminal["resource_measurement_error_type"]) is str
                and bool(terminal["resource_measurement_error_type"]), "missing RSS evidence is not success")
        require(digest(worker["result"]) == digest(terminal["result"]) == raw_sha
            and type(saved) is dict and all(saved.get(key) == value for key, value in self.expected.items())
            and type(saved["selection_seed"]) is int
            and saved["fixed_work_diagnostic"]["contract"] == self.contract.receipt()
            and type(saved["fixed_work_diagnostic"]["accepted_actual"]) is int
            and saved["fixed_work_diagnostic"]["accepted_actual"] == self.contract.accepted_boundary(self.configuration.arm),
            "external selected/config/root/runtime or exact worker evidence drift")
        from .search_over_raw_resources import read_completed_process_cost
        costs = read_completed_process_cost(terminal, worker, work_phase="selection")
        if self.configuration.arm == "reference":
            actual_pids = saved["evidence"].get("worker_pids")
            require(type(actual_pids) is list and len(actual_pids) == 20
                and all(type(p) is int for p in actual_pids) and sorted(actual_pids) == sorted(pids),
                "reference atomic exports lack their actual owned worker roster")
        manifest_sha = validate_fixed_work_inputs(saved["evidence"], arm=self.configuration.arm,
            binding=self.consumed, statistics_root_id=self.statistics_root_id,
            construction_required=self.configuration.arm == "incumbent",
            expected_public_request_sha256=self.public_request_sha256,
            native_public_fold_required=self.native_public_fold_required)
        accounting = derive_fixed_work_accounting(saved["evidence"], arm=self.configuration.arm,
            binding=self.consumed, statistics_root_id=self.statistics_root_id, leaf=self.configuration.leaf,
            native_public_fold_required=self.native_public_fold_required,
            expected_public_request_sha256=self.public_request_sha256)
        require(accounting == terminal["observed_work_accounting"], "independent fixed-work cost reconciliation drift")
        self._files()
        require(self._inputs() == self.frozen and digest(selected) == raw_sha and all(hashlib.sha256((self.directory / name).read_bytes())
            .hexdigest() == value for name, value in hashes.items())
            and hashlib.sha256((self.directory / "job.pickle").read_bytes()).hexdigest() == self.job_sha
            and sorted(owned.glob("group-*.json")) == paths
            and not list((owned / "active").glob("group-*.json")), "fixed-work evidence changed during validation")
        self.last_validation = dict(schema="pokezero.search-over-raw.fixed-work-owning-readout.v1",
            selected_sha256=selected_sha, consumed_manifest_sha256=manifest_sha,
            receipt_file_sha256=hashes, actual_work=accounting, costs=costs,
            main_launch=deepcopy(launch), process_receipts_validated=True,
            owned_generation_pids=sorted(pids),
            full_public_request_sha256=self.public_request_sha256,
            construction_public_fold_capture_validated=self.native_public_fold_required,
            declared_file_bytes_validated=True, loaded_binary_dependency_closure_validated=False,
            native_ancestry_coverage_validated=False, full_world_equivalence=False,
            resource_cost_complete=False, runtime_qualified=False, scientific_collection_authorized=False,
            synthetic_process_fixture=claim["synthetic_process_fixture"])
        return selected_sha if not claim["synthetic_process_fixture"] or self.allow_synthetic else None
