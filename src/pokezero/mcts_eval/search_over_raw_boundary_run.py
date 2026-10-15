"""Distinct one-shot ownership for private, prospectively frozen boundary traces.

This proves transport and normal process cleanup, not searched-line selection,
native ancestry coverage, construction provenance, dynamics or terminal truth.
No source environment is reset, no policy is loaded, and no retry is offered.
Actual simulator execution still requires separately reviewed caller authority.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import hashlib
import math
from pathlib import Path
import time

from .search_over_raw import digest, require
from .search_over_raw_boundary_transport import (
    BoundaryTransportCapture, _binding, _identity, _requests, _sha, read_owned_boundary_transport,
)
from .search_over_raw_outcome_run import (
    ContinuationProcessOwner, OwningRunValidator, verify_outcome_runtime,
)
from .search_over_raw_outcomes import _snapshot_json, snapshot_identity
from .search_over_raw_supervision import _supervise_owned_job


def _validate_inputs(value):
    require(type(value) is dict and set(value) == {"root_binding","snapshot_identity","source_battle_token",
        "runtime_sha256","clone_binding","request_ids","choices","max_transitions"}
        and type(value["root_binding"]) is dict and value["root_binding"]
        and type(value["snapshot_identity"]) is dict and type(value["source_battle_token"]) is str
        and value["source_battle_token"] and type(value["max_transitions"]) is int
        and value["max_transitions"] == 64,"exact external boundary input schema required")
    _binding(value["clone_binding"])
    require(value["source_battle_token"] != value["clone_binding"]["battleId"],
        "distinct source and clone identities required")
    _validate_path(value)


def _validate_path(value):
    _binding(value["clone_binding"])
    requests,choices = value["request_ids"],value["choices"]
    require(type(value["max_transitions"]) is int and value["max_transitions"] == 64
        and type(requests) is list and 1 <= len(requests) <= 65
        and all(_identity(v) for v in requests) and len(set(requests)) == len(requests)
        and type(choices) is list and len(choices) == len(requests)-1
        and all(type(row) is dict and row and set(row) <= {"p1","p2"}
            and all(type(v) is str and v.strip() and "\n" not in v and "\r" not in v
                for v in row.values()) for row in choices),"full frozen single-command joint roster required")


@dataclass(frozen=True)
class BoundaryRunContract:
    emergency_watchdog_seconds: float = 1800.

    def receipt(self):
        require(type(self.emergency_watchdog_seconds) in (int,float)
            and math.isfinite(self.emergency_watchdog_seconds) and self.emergency_watchdog_seconds > 10,
            "predeclared boundary-run emergency bound required")
        return dict(schema="pokezero.search-over-raw.boundary-run-contract.v1",
            emergency_watchdog_seconds=self.emergency_watchdog_seconds, max_transitions=64,
            source_generation=False, retries_permitted=False, collection_authorized=False)


class BoundaryProcessOwner(ContinuationProcessOwner):
    """Reuse exclusive wait4/strict close, never the outcome-bank work law."""
    launch_schema = "pokezero.search-over-raw.boundary-bridge-launch.v1"
    close_schema = "pokezero.search-over-raw.boundary-bridge-close.v1"
    capture_hash_key = "capture_sha256"

    def __init__(self, *, initial_snapshot_json, clone_binding, request_ids, **kwargs):
        super().__init__(**kwargs)
        self.initial = initial_snapshot_json
        self.clone, self.requests = deepcopy(clone_binding), list(request_ids)

    def _match_capture(self, capture):
        require(getattr(self.env,"_bridge_boundary_observer",None) is None,
            "private boundary tap must be removed before sealing")
        read_owned_boundary_transport(capture,process_records=self.records,initial_snapshot_json=self.initial,
            clone_binding=self.clone,request_ids=self.requests)


@dataclass(frozen=True)
class OwnedBoundaryJob:
    """Replay explicit joint commands from an already frozen private root.

    The caller must freeze the full intended reached roster, including the root.
    Unreached/unsupported paths fail closed rather than shorten that roster.
    This job alone cannot establish how the native path/commands were selected.
    """
    config: object
    snapshot: object
    root_binding: dict
    runtime: dict
    clone_binding: dict
    request_ids: list
    choices: list
    process_owner_class = BoundaryProcessOwner

    def initial_snapshot_json(self):
        return _snapshot_json(self.snapshot)

    def input_binding(self):
        _binding(self.clone_binding)
        require(type(self.root_binding) is dict and self.root_binding and self.snapshot.terminal is None
            and _sha(_snapshot_json(self.snapshot)) == self.clone_binding["snapshotJsonSha256"]
            and self.snapshot.battle_token != self.clone_binding["battleId"],
            "distinct owned clone of a frozen nonterminal root required")
        value = dict(root_binding=deepcopy(self.root_binding),snapshot_identity=snapshot_identity(self.snapshot),
            source_battle_token=self.snapshot.battle_token,runtime_sha256=verify_outcome_runtime(self.runtime,self.config),
            clone_binding=deepcopy(self.clone_binding),request_ids=list(self.request_ids),choices=deepcopy(self.choices),
            max_transitions=64)
        require(type(self.request_ids) is list,"prospective boundary request list required")
        _validate_inputs(value)
        return value

    def _components(self):
        from ..local_showdown import LocalShowdownEnv
        return LocalShowdownEnv(self.config)

    def transition_count(self):
        return len(self.choices)

    def commands_at_boundary(self, ordinal, current):
        return deepcopy(self.choices[ordinal])

    def check_reached_boundary(self, current):
        pass

    def _advance(self, env, choices):
        # Read transport only: no policy inference, observation reconstruction,
        # source reset, synthetic terminal score or implicit next action.
        env._send_command(dict(type="choices",battleId=self.clone_binding["battleId"],choices=choices))
        deadline = time.monotonic()+env.config.read_timeout_seconds
        while True:
            remaining = deadline-time.monotonic()
            require(remaining > 0,"boundary transition observation timed out")
            event = env._read_event(timeout=remaining)
            if event is None: continue
            require(event.get("battleId") == self.clone_binding["battleId"] and event.get("type") != "error",
                "foreign or failed boundary transition")
            if event.get("type") in {"ready","terminal"}: return

    def __call__(self, ownership, phase):
        phase("startup")
        before = self.input_binding()
        env = self._components()
        tap = BoundaryTransportCapture(clone_binding=self.clone_binding,request_ids=self.request_ids)
        owner = self.process_owner_class(directory=Path(ownership["directory"]).parent,ownership=ownership,
            runtime=self.runtime,config=self.config,initial_snapshot_json=self.initial_snapshot_json(),
            clone_binding=self.clone_binding,request_ids=self.request_ids)
        primary = None
        with owner.installed(env):
            try:
                phase("boundaries")
                with tap.installed(env):
                    env._start_bridge()  # Empty bridge, never reset/start/materialize a source game.
                    env._battle_token = self.clone_binding["battleId"]
                    event = env._bridge_request_event(dict(type="clone_snapshot",
                        battleId=self.clone_binding["battleId"],cloneId=self.clone_binding["cloneId"],
                        snapshotJson=self.initial_snapshot_json()),"cloned")
                    require(all(event.get(k) == v for k,v in self.clone_binding.items()),"actual clone drift")
                    current = tap.snapshot()
                    self.check_reached_boundary(current)
                    for ordinal in range(self.transition_count()):
                        choices = self.commands_at_boundary(ordinal, current)
                        require(set(choices) == set(_requests(current)),
                            "frozen joint command must cover every actual requested actor before send")
                        self._advance(env,choices)
                        current = tap.snapshot()
                        self.check_reached_boundary(current)
                    env._bridge_request_event(dict(type="release_clone",**self.clone_binding),"clone_released")
                    env._battle_token = None
                require(digest(self.input_binding()) == digest(before),"boundary job inputs changed during replay")
                capture = tap.export()
                owner.seal(capture)
            except BaseException as error:
                primary = error
                raise
            finally:
                phase("cleanup")
                if env._process is not None:
                    try: env.close()
                    except BaseException as cleanup:
                        if primary is None: raise
                        if hasattr(primary,"add_note"): primary.add_note("Boundary owner cleanup failed: "+str(cleanup))
        require(digest(self.input_binding()) == digest(before),"boundary job inputs changed after bridge reap")
        return dict(**before,capture=capture,capture_sha256=digest(capture),
            synthetic_process_fixture=getattr(self,"synthetic_process_fixture",False) is True,
            construction_provenance_validated=False,native_ancestry_coverage_validated=False,
            branch_transition_validated=False,independent_terminal_truth=False,
            runtime_qualified=False,resource_cost_complete=False,scientific_collection_authorized=False)


def supervise_boundary_run(job, *, directory, binding_sha256, verify_admission, validate_result,
                           contract=None, _synthetic_watchdog_seconds=None):
    contract = BoundaryRunContract() if contract is None else contract
    require(type(contract) is BoundaryRunContract,"distinct boundary-run contract required")
    contract.receipt()
    require((type(job) is OwnedBoundaryJob or isinstance(job,OwnedBoundaryJob)
        and getattr(job,"synthetic_process_fixture",False) is True) and callable(validate_result),
        "canonical frozen boundary job or explicitly synthetic fixture required")
    before = job.input_binding()
    def verify():
        require(job.input_binding() == before,"boundary launch/acceptance inputs changed")
        return verify_admission()
    def validate(result):
        require(type(result) is dict and all(result.get(k) == v for k,v in before.items())
            and result["capture_sha256"] == digest(result["capture"])
            and all(result[k] is False for k in ("construction_provenance_validated",
                "native_ancestry_coverage_validated","branch_transition_validated","independent_terminal_truth",
                "runtime_qualified","resource_cost_complete","scientific_collection_authorized")),
            "unpromoted boundary job result required")
        frozen_result = digest(result)
        require(validate_result(result) is not False,"independent boundary result validator refused")
        require(digest(result) == frozen_result,"boundary result validator mutated worker evidence")
        require(job.input_binding() == before,"boundary acceptance inputs changed")
    return _supervise_owned_job(job,directory=directory,binding_sha256=binding_sha256,verify_admission=verify,
        contract=contract,kind="boundary-run",work_phase="boundaries",validate_result=validate,
        _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)


class BoundaryRunValidator(OwningRunValidator):
    work_phase = "boundaries"
    """Read boundary-specific worker/bridge receipts, not outcome-bank receipts.

    Success certifies transport ownership only. Local hashes are drift guards,
    not operator authentication or independent simulation of the same branch.
    """
    contract_class = BoundaryRunContract
    supervision_schema = "pokezero.search-over-raw.boundary-run-supervision.v1"
    complete_status = "BOUNDARY_RUN_COMPLETE"
    launch_schema = BoundaryProcessOwner.launch_schema
    close_schema = BoundaryProcessOwner.close_schema
    capture_hash_key = "capture_sha256"

    def __init__(self, *, expected_inputs, **kwargs):
        super().__init__(**kwargs)
        self.expected = deepcopy(expected_inputs)
        _validate_inputs(self.expected)
        require(self.expected.get("root_binding") == self.root
            and self.expected.get("snapshot_identity") == snapshot_identity(self.snapshot)
            and self.expected.get("source_battle_token") == self.snapshot.battle_token
            and self.expected.get("runtime_sha256") == digest(self.runtime),"external boundary job binding required")
        self.expected_sha = digest(self.expected)

    def _join_result(self, result, capture, runtime_sha):
        require(digest(self.expected) == self.expected_sha
            and all(result.get(k) == v for k,v in self.expected.items())
            and result["capture"] == capture and result["capture_sha256"] == digest(capture)
            and result["runtime_sha256"] == runtime_sha
            and all(result[k] is False for k in ("construction_provenance_validated",
                "native_ancestry_coverage_validated","branch_transition_validated","independent_terminal_truth",
                "runtime_qualified","resource_cost_complete","scientific_collection_authorized")),
            "exact unpromoted boundary/root/worker evidence join required")

    def _match_capture(self, capture, records):
        trace = read_owned_boundary_transport(capture,process_records=records,
            initial_snapshot_json=_snapshot_json(self.snapshot),clone_binding=self.expected["clone_binding"],
            request_ids=self.expected["request_ids"])
        require([v["incoming_choices"] for v in trace["observed_snapshots"][1:]] == self.expected["choices"],
            "actual joint choices differ from prospectively frozen path")


RAW_INPUT_KIND = "pokezero.search-over-raw.raw-materialized-boundary-input.v1"
ROOT_ORIGIN_FIELDS = {"materialization_capture_sha256", "produced_bridge_snapshot_sha256",
    "native_selected_sha256", "native_capture_sha256", "construction_ordinal", "evaluation_ordinal"}
UNQUALIFIED = ("construction_provenance_validated", "native_ancestry_coverage_validated",
    "branch_transition_validated", "independent_terminal_truth", "runtime_qualified",
    "resource_cost_complete", "scientific_collection_authorized")


def raw_root_json(root):
    """Canonical encoding of the returned RAW object, never Python actor state.

    The materializer exports an object plus its canonical digest, not original
    wire bytes. This newly declared clone encoding is distinct from that digest.
    It does not change, settle or fill in any simulator fields.
    """
    require(type(root) is dict and _requests(root) and root["battle"]["ended"] is False,
        "actual nonterminal raw root with paired requests required")
    digest(root)
    return json.dumps(root, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _validate_raw_inputs(value):
    require(type(value) is dict and set(value) == {"input_kind", "root_binding", "raw_root_sha256",
        "root_origin", "runtime_sha256", "clone_binding", "request_ids", "choices", "max_transitions"}
        and value["input_kind"] == RAW_INPUT_KIND and type(value["root_binding"]) is dict
        and value["root_binding"] and type(value["root_origin"]) is dict
        and set(value["root_origin"]) == ROOT_ORIGIN_FIELDS,
        "distinct externally frozen raw-root boundary inputs required")
    origin = value["root_origin"]
    is_sha = lambda v: type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
    require(is_sha(value["runtime_sha256"]) and is_sha(value["raw_root_sha256"])
        and all(is_sha(origin[k]) for k in ROOT_ORIGIN_FIELDS if k.endswith("sha256"))
        and origin["produced_bridge_snapshot_sha256"] == value["raw_root_sha256"]
        and type(origin["construction_ordinal"]) is int and 0 <= origin["construction_ordinal"] < 4
        and type(origin["evaluation_ordinal"]) is int and origin["evaluation_ordinal"] >= 0,
        "exact observed materialization/native-path origin binding required")
    # Reuse only path shape; do not synthesize even an empty Python snapshot.
    _validate_path(value)


@dataclass(frozen=True)
class RawBoundaryRunContract(BoundaryRunContract):
    def receipt(self):
        return dict(super().receipt(), schema="pokezero.search-over-raw.raw-boundary-run-contract.v1",
            input_kind=RAW_INPUT_KIND, python_actor_snapshot_required=False)


class RawBoundaryProcessOwner(BoundaryProcessOwner):
    launch_schema = "pokezero.search-over-raw.raw-boundary-bridge-launch.v1"
    close_schema = "pokezero.search-over-raw.raw-boundary-bridge-close.v1"


@dataclass(frozen=True)
class OwnedRawBoundaryJob:
    """Distinct raw-object path; origin hashes alone do not prove prior ownership.

    The controller must independently join root_origin to materialization and
    ancestry readers. Explicit choices remain prospective and may not be drawn
    from the resulting replay to fabricate successful path coverage.
    """
    config: object
    raw_root: dict
    root_binding: dict
    root_origin: dict
    runtime: dict
    clone_binding: dict
    request_ids: list
    choices: list
    process_owner_class = RawBoundaryProcessOwner
    _components = OwnedBoundaryJob._components
    _advance = OwnedBoundaryJob._advance
    __call__ = OwnedBoundaryJob.__call__
    transition_count = OwnedBoundaryJob.transition_count
    commands_at_boundary = OwnedBoundaryJob.commands_at_boundary
    check_reached_boundary = OwnedBoundaryJob.check_reached_boundary

    def initial_snapshot_json(self):
        return raw_root_json(self.raw_root)

    def input_binding(self):
        require(type(self.request_ids) is list and type(self.choices) is list,
            "prospective raw-root path roster required")
        raw = self.initial_snapshot_json()
        value = dict(input_kind=RAW_INPUT_KIND, root_binding=deepcopy(self.root_binding),
            raw_root_sha256=digest(self.raw_root), root_origin=deepcopy(self.root_origin),
            runtime_sha256=verify_outcome_runtime(self.runtime, self.config),
            clone_binding=deepcopy(self.clone_binding), request_ids=list(self.request_ids),
            choices=deepcopy(self.choices), max_transitions=64)
        _validate_raw_inputs(value)
        require(_sha(raw) == self.clone_binding["snapshotJsonSha256"], "canonical raw clone bytes drift")
        return value


def _join_raw_result(result, expected, capture):
    require(type(result) is dict and set(result) == set(expected) | set(UNQUALIFIED) |
        {"capture", "capture_sha256", "synthetic_process_fixture"}
        and all(digest(result[k]) == digest(v) for k, v in expected.items())
        and digest(result["capture"]) == digest(capture) and result["capture_sha256"] == digest(capture)
        and type(result["synthetic_process_fixture"]) is bool and all(result[k] is False for k in UNQUALIFIED),
        "exact unpromoted raw-root boundary result required")


def supervise_raw_boundary_run(job, *, directory, binding_sha256, verify_admission, validate_result,
                               contract=None, _synthetic_watchdog_seconds=None):
    contract = RawBoundaryRunContract() if contract is None else contract
    require(type(contract) is RawBoundaryRunContract and (type(job) is OwnedRawBoundaryJob or
        isinstance(job, OwnedRawBoundaryJob) and getattr(job, "synthetic_process_fixture", False) is True)
        and callable(validate_result), "distinct raw boundary job and contract required")
    before = job.input_binding()
    def verify():
        require(digest(job.input_binding()) == digest(before), "raw boundary admission input drift")
        return verify_admission()
    def validate(result):
        _join_raw_result(result, before, result["capture"])
        frozen = digest(result)
        require(validate_result(result) is not False, "raw boundary result refused")
        require(digest(result) == frozen and digest(job.input_binding()) == digest(before),
            "raw boundary acceptance mutation")
    # Generic supervision supplies wait ownership, not the snapshot/work law.
    return _supervise_owned_job(job, directory=directory, binding_sha256=binding_sha256,
        verify_admission=verify, contract=contract, kind="boundary-run", work_phase="boundaries",
        validate_result=validate, _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)


class RawBoundaryRunValidator(OwningRunValidator):
    work_phase = "boundaries"
    contract_class = RawBoundaryRunContract
    supervision_schema = BoundaryRunValidator.supervision_schema
    complete_status = BoundaryRunValidator.complete_status
    launch_schema = RawBoundaryProcessOwner.launch_schema
    close_schema = RawBoundaryProcessOwner.close_schema
    capture_hash_key = "capture_sha256"
    validate_expected = staticmethod(_validate_raw_inputs)

    def __init__(self, *, directory, job_sha256, binding_sha256, expected_inputs, expected_raw_root,
                 runtime, allow_synthetic=False, contract=None):
        is_sha = lambda v: type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
        require(is_sha(job_sha256) and is_sha(binding_sha256) and type(allow_synthetic) is bool,
            "external raw-root job/attempt inputs required")
        self.directory, self.job_sha, self.binding = Path(directory).resolve(), job_sha256, binding_sha256
        self.allow_synthetic = allow_synthetic
        self.expected, self.raw_root, self.runtime = deepcopy(expected_inputs), deepcopy(expected_raw_root), deepcopy(runtime)
        contract = self.contract_class() if contract is None else contract
        require(type(contract) is self.contract_class, "distinct external raw boundary contract required")
        self.contract = contract.receipt()
        self.inputs = digest(dict(expected=self.expected, raw_root=self.raw_root, runtime=self.runtime))
        self._check_inputs()

    def _check_inputs(self):
        self.validate_expected(self.expected)
        raw = raw_root_json(self.raw_root)
        require(digest(dict(expected=self.expected, raw_root=self.raw_root, runtime=self.runtime)) == self.inputs
            and self.expected["runtime_sha256"] == digest(self.runtime)
            and self.expected["raw_root_sha256"] == digest(self.raw_root)
            and self.expected["clone_binding"]["snapshotJsonSha256"] == _sha(raw),
            "external raw root/runtime/clone inputs changed")

    def _join_result(self, result, capture, runtime_sha):
        require(runtime_sha == self.expected["runtime_sha256"], "raw boundary runtime drift")
        _join_raw_result(result, self.expected, capture)

    def _match_capture(self, capture, records):
        trace = read_owned_boundary_transport(capture, process_records=records,
            initial_snapshot_json=raw_root_json(self.raw_root), clone_binding=self.expected["clone_binding"],
            request_ids=self.expected["request_ids"])
        require(digest([v["incoming_choices"] for v in trace["observed_snapshots"][1:]]) == digest(self.expected["choices"]),
            "actual raw-root replay choices differ from frozen path")


NATIVE_PATH_INPUT_KIND = "pokezero.search-over-raw.native-action-boundary-input.v1"


def native_command_source_hashes():
    paths = [Path(__file__), Path(__file__).with_name("search_over_raw_showdown_boundary.py"),
        Path(__file__).with_name("search_over_raw_native_boundary.py")]
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _validate_native_path_inputs(value):
    from .search_over_raw_native_boundary import _choice
    require(type(value) is dict and set(value) == {"input_kind", "root_binding", "raw_root_sha256",
        "root_origin", "runtime_sha256", "clone_binding", "request_ids", "native_boundaries", "party_mapping", "max_transitions",
        "command_derivation_sources"}
        and value["input_kind"] == NATIVE_PATH_INPUT_KIND
        and digest(value["command_derivation_sources"]) == digest(native_command_source_hashes()),
        "distinct source-bound native-action path input law required")
    # Validate the shared origin without changing the explicit-command law.
    origin = value["root_origin"]
    sha = lambda v: type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
    require(type(value["root_binding"]) is dict and value["root_binding"] and sha(value["runtime_sha256"])
        and sha(value["raw_root_sha256"]) and type(origin) is dict and set(origin) == ROOT_ORIGIN_FIELDS
        and all(sha(origin[k]) for k in ROOT_ORIGIN_FIELDS if k.endswith("sha256"))
        and origin["produced_bridge_snapshot_sha256"] == value["raw_root_sha256"]
        and type(origin["construction_ordinal"]) is int and 0 <= origin["construction_ordinal"] < 4
        and type(origin["evaluation_ordinal"]) is int and origin["evaluation_ordinal"] >= 0,
        "frozen native path origin required")
    _binding(value["clone_binding"])
    requests, boundaries = value["request_ids"], value["native_boundaries"]
    require(type(value["max_transitions"]) is int and value["max_transitions"] == 64
        and type(requests) is list and 2 <= len(requests) <= 65 and all(_identity(v) for v in requests)
        and len(set(requests)) == len(requests) and type(boundaries) is list and len(boundaries) == len(requests)
        and type(value["party_mapping"]) is dict
        and value["party_mapping"].get("root_snapshot_sha256") == value["raw_root_sha256"],
        "full original native path and root-bound party map required")
    for ordinal, row in enumerate(boundaries):
        require(type(row) is dict and set(row) == {"ordinal", "binding", "native", "native_export_sha256",
            "native_export_utf8_sha256", "independently_validated", "party_mapping_validated", "runtime_qualified",
            "scientific_collection_authorized"} and type(row["ordinal"]) is int and row["ordinal"] == ordinal
            and type(row["binding"]) is dict and row["binding"].get("ordinal") == ordinal
            and type(row["binding"].get("ordinal")) is int
            and row["binding"].get("capture_sha256") == origin["native_capture_sha256"]
            and row["binding"].get("evaluation_ordinal") == origin["evaluation_ordinal"]
            and type(row["binding"].get("evaluation_ordinal")) is int
            and row["binding"].get("root_binding") == value["root_binding"].get("root_binding")
            and sha(row["native_export_sha256"]) and sha(row["native_export_utf8_sha256"])
            and type(row["native"]) is dict and all(row[k] is False for k in
                ("independently_validated", "party_mapping_validated", "runtime_qualified", "scientific_collection_authorized")),
            "unpromoted original native coordinate inputs required")
        actions = row["native"].get("action_identity")
        if not ordinal:
            require(actions is None, "root has no arriving native action")
        else:
            require(type(actions) is dict and set(actions) == {"p1", "p2"}, "ambiguous native action cannot drive replay")
            for action in actions.values(): _choice(action)
    digest(value)


@dataclass(frozen=True)
class NativePathBoundaryRunContract(BoundaryRunContract):
    def receipt(self):
        return dict(super().receipt(), schema="pokezero.search-over-raw.native-path-boundary-run-contract.v1",
            input_kind=NATIVE_PATH_INPUT_KIND, python_actor_snapshot_required=False,
            caller_command_roster=False, unavailable_action_fallback=False)


class NativePathBoundaryProcessOwner(BoundaryProcessOwner):
    launch_schema = "pokezero.search-over-raw.native-path-boundary-bridge-launch.v1"
    close_schema = "pokezero.search-over-raw.native-path-boundary-bridge-close.v1"


@dataclass(frozen=True)
class OwnedNativePathBoundaryJob:
    """Frozen native targets; commands are resolved before each actual send.

    Prior ownership is checked by the controller, not by caller-supplied hashes.
    Stochastic divergence can make a later action unavailable: then the full
    path fails, with no substitute, retry or shortened successful denominator.
    """
    config: object
    raw_root: dict
    root_binding: dict
    root_origin: dict
    runtime: dict
    clone_binding: dict
    request_ids: list
    native_boundaries: list
    party_mapping: dict
    process_owner_class = NativePathBoundaryProcessOwner
    _components = OwnedBoundaryJob._components
    _advance = OwnedBoundaryJob._advance
    __call__ = OwnedBoundaryJob.__call__

    def initial_snapshot_json(self): return raw_root_json(self.raw_root)

    def transition_count(self): return len(self.native_boundaries)-1

    def commands_at_boundary(self, ordinal, current):
        from .search_over_raw_showdown_boundary import commands_for_native_actions
        return commands_for_native_actions(current, self.party_mapping,
            self.native_boundaries[ordinal+1]["native"]["action_identity"])

    def check_reached_boundary(self, current):
        from .search_over_raw_showdown_boundary import _mapped_sides
        _mapped_sides(current, self.party_mapping)

    def input_binding(self):
        from .search_over_raw_showdown_boundary import _mapped_sides
        raw = self.initial_snapshot_json()
        value = dict(input_kind=NATIVE_PATH_INPUT_KIND, root_binding=deepcopy(self.root_binding),
            raw_root_sha256=digest(self.raw_root), root_origin=deepcopy(self.root_origin),
            runtime_sha256=verify_outcome_runtime(self.runtime, self.config), clone_binding=deepcopy(self.clone_binding),
            request_ids=deepcopy(self.request_ids), native_boundaries=deepcopy(self.native_boundaries),
            party_mapping=deepcopy(self.party_mapping), max_transitions=64,
            command_derivation_sources=native_command_source_hashes())
        _validate_native_path_inputs(value)
        _mapped_sides(self.raw_root, self.party_mapping)
        require(_sha(raw) == self.clone_binding["snapshotJsonSha256"], "native replay canonical clone bytes drift")
        return value


def supervise_native_path_boundary_run(job, *, directory, binding_sha256, verify_admission,
                                      validate_result, contract=None, _synthetic_watchdog_seconds=None):
    contract = NativePathBoundaryRunContract() if contract is None else contract
    require(type(contract) is NativePathBoundaryRunContract and (type(job) is OwnedNativePathBoundaryJob or
        isinstance(job, OwnedNativePathBoundaryJob) and getattr(job, "synthetic_process_fixture", False) is True)
        and callable(validate_result), "distinct native-path job and contract required")
    before = job.input_binding()
    def verify():
        require(digest(job.input_binding()) == digest(before), "native-path admission inputs changed")
        return verify_admission()
    def validate(result):
        _join_raw_result(result, before, result["capture"])
        frozen = digest(result)
        require(validate_result(result) is not False, "native-path result refused")
        require(digest(result) == frozen and digest(job.input_binding()) == digest(before), "native-path acceptance mutation")
    return _supervise_owned_job(job, directory=directory, binding_sha256=binding_sha256, verify_admission=verify,
        contract=contract, kind="boundary-run", work_phase="boundaries", validate_result=validate,
        _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)


class NativePathBoundaryRunValidator(RawBoundaryRunValidator):
    contract_class = NativePathBoundaryRunContract
    launch_schema = NativePathBoundaryProcessOwner.launch_schema
    close_schema = NativePathBoundaryProcessOwner.close_schema
    validate_expected = staticmethod(_validate_native_path_inputs)

    def _check_inputs(self):
        from .search_over_raw_showdown_boundary import _mapped_sides
        super()._check_inputs()
        _mapped_sides(self.raw_root, self.expected["party_mapping"])

    def _match_capture(self, capture, records):
        from .search_over_raw_showdown_boundary import commands_for_native_actions, _mapped_sides
        trace = read_owned_boundary_transport(capture, process_records=records,
            initial_snapshot_json=raw_root_json(self.raw_root), clone_binding=self.expected["clone_binding"],
            request_ids=self.expected["request_ids"])
        snapshots = [json.loads(row["currentSnapshotJson"]) for row in capture["snapshots"]]
        for snapshot in snapshots: _mapped_sides(snapshot, self.expected["party_mapping"])
        for ordinal, observed in enumerate(trace["observed_snapshots"][1:]):
            expected = commands_for_native_actions(snapshots[ordinal], self.expected["party_mapping"],
                self.expected["native_boundaries"][ordinal+1]["native"]["action_identity"])
            require(digest(observed["incoming_choices"]) == digest(expected),
                "actual command differs from frozen native action's pre-state resolution")
