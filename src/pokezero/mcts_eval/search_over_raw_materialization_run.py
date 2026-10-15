"""Distinct one-shot ownership for an explicitly sampled hypothetical root.

No random source generation, selection, policy/model load, retry or collection.
Normal ownership proves transport provenance, not native/reference physics or
equivalence. Only the observed raw bridge root is exported by this contract;
Python's substituted actor request is not promoted into simulator evidence.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path

from .search_over_raw import digest, require
from .search_over_raw_materialization import (
    materialization_inputs, observe_materialized_world, read_materialization, validate_materialization_inputs,
)
from .search_over_raw_outcome_run import ContinuationProcessOwner, OwningRunValidator, verify_outcome_runtime
from .search_over_raw_outcomes import _sha
from .search_over_raw_supervision import _supervise_owned_job

UNQUALIFIED = ("construction_provenance_validated","native_ancestry_coverage_validated",
    "full_world_equivalence","branch_transition_validated","runtime_qualified",
    "resource_cost_complete","scientific_collection_authorized")


def _validate_inputs(value):
    require(type(value) is dict and set(value)=={"materialization_inputs","runtime_sha256"}
        and _sha(value["runtime_sha256"]),"distinct external materialization-run inputs required")
    validate_materialization_inputs(value["materialization_inputs"])


@dataclass(frozen=True)
class MaterializationRunContract:
    emergency_watchdog_seconds: float = 1800.

    def receipt(self):
        require(type(self.emergency_watchdog_seconds) in (int,float)
            and math.isfinite(self.emergency_watchdog_seconds) and self.emergency_watchdog_seconds>10,
            "predeclared materialization emergency bound required")
        return dict(schema="pokezero.search-over-raw.materialization-run-contract.v1",
            emergency_watchdog_seconds=self.emergency_watchdog_seconds,max_roots=1,
            explicit_sampled_teams_only=True,source_generation=False,retries_permitted=False,
            collection_authorized=False)


class MaterializationProcessOwner(ContinuationProcessOwner):
    launch_schema="pokezero.search-over-raw.materialization-bridge-launch.v1"
    close_schema="pokezero.search-over-raw.materialization-bridge-close.v1"
    capture_hash_key="capture_sha256"

    def __init__(self,*,expected_inputs,**kwargs):
        super().__init__(**kwargs)
        self.expected=deepcopy(expected_inputs);validate_materialization_inputs(self.expected)
        self.snapshot=None;self.empty_reset_close=False;self.reset_started=False

    def begin_owned_reset(self,env,seed,format_id,start_override):
        from dataclasses import asdict
        from ..env import BattleStartOverride
        tap=getattr(env,"_bridge_evidence_observer",None)
        require(env is self.env and self.phase=="empty" and not self.reset_started
            and env._process is None and self.process is None and env._battle_token is None
            and env._battle_counter==0 and type(seed) is int and seed==self.expected["seed"]
            and type(start_override) is BattleStartOverride and asdict(start_override)==self.expected["draw"]
            and format_id==self.expected["draw"]["format_id"]
            and tap is not None and tap.env is env and tap.inputs==self.expected,
            "one exact reviewed materialization reset required; no source generation or restart")
        self.reset_started=True

    def close(self,env):
        if self.phase=="empty":
            # Existing _reset calls close() before its very first _start_bridge.
            # Only that observed, single fresh-shell no-op is admissible. It is
            # not a child cleanup receipt and cannot clear a later failed run.
            tap=getattr(env,"_bridge_evidence_observer",None)
            require(env is self.env and env._process is None and self.process is None
                and self.launch is None and self.bank_sha is None and not self.empty_reset_close
                and env._battle_counter==1 and env._battle_token=="b1"
                and tap is not None and tap.env is env and tap.inputs==self.expected,
                "only the observed initial empty-shell reset close is allowed")
            self.empty_reset_close=True
            return
        return super().close(env)

    def _match_capture(self,capture):
        require(self.snapshot is not None,"actual produced root required before seal")
        read_materialization(capture,expected_inputs=self.expected,process_records=self.records,
            expected_snapshot=self.snapshot)


@dataclass(frozen=True)
class OwnedMaterializationJob:
    config: object
    state: object
    override: object
    seed: int
    root_binding: dict
    runtime: dict
    options: dict | None = None

    def input_binding(self):
        value=dict(materialization_inputs=materialization_inputs(state=self.state,override=self.override,
            seed=self.seed,root_binding=self.root_binding,options=self.options),
            runtime_sha256=verify_outcome_runtime(self.runtime,self.config))
        _validate_inputs(value)
        return value

    def _components(self):
        from ..local_showdown import LocalShowdownEnv
        return LocalShowdownEnv(self.config)

    def __call__(self,ownership,phase):
        phase("startup");before=self.input_binding();env=self._components()
        owner=MaterializationProcessOwner(directory=Path(ownership["directory"]).parent,ownership=ownership,
            runtime=self.runtime,config=self.config,expected_inputs=before["materialization_inputs"])
        primary=None
        with owner.installed(env):
            try:
                phase("materialization")
                observed=observe_materialized_world(env,state=self.state,override=self.override,seed=self.seed,
                    root_binding=self.root_binding,options=self.options)
                require(observed["inputs"]==before["materialization_inputs"]
                    and self.input_binding()==before,"actual materialization input drift")
                capture=observed["capture"]
                # Do not export Python latest_requests as authoritative state.
                root=deepcopy(observed["snapshot"].bridge_snapshot)
                owner.snapshot=root;owner.seal(capture)
            except BaseException as error:
                primary=error
                raise
            finally:
                phase("cleanup")
                if env._process is not None:
                    try:env.close()
                    except BaseException as cleanup:
                        if primary is None:raise
                        if hasattr(primary,"add_note"):primary.add_note("Materialization cleanup failed: "+type(cleanup).__name__)
        require(self.input_binding()==before,"materialization inputs changed after reap")
        return dict(**before,capture=capture,capture_sha256=digest(capture),produced_bridge_snapshot=root,
            produced_bridge_snapshot_sha256=digest(root),
            synthetic_process_fixture=getattr(self,"synthetic_process_fixture",False) is True,
            **{key:False for key in UNQUALIFIED})


@dataclass(frozen=True)
class OwnedRecordedMaterializationJob:
    """Exact owned constructor for a native run's already consumed payload."""
    config: object
    recorded_inputs: dict
    runtime: dict

    def input_binding(self):
        value=dict(materialization_inputs=deepcopy(self.recorded_inputs),
            runtime_sha256=verify_outcome_runtime(self.runtime,self.config))
        _validate_inputs(value)
        require("recorded_native_origin" in self.recorded_inputs,"recorded native input law required")
        return value

    def _components(self):
        from ..local_showdown import LocalShowdownEnv
        return LocalShowdownEnv(self.config)

    def __call__(self,ownership,phase):
        from .search_over_raw_materialization import observe_recorded_materialized_world
        phase("startup");before=self.input_binding();env=self._components()
        owner=MaterializationProcessOwner(directory=Path(ownership["directory"]).parent,ownership=ownership,
            runtime=self.runtime,config=self.config,expected_inputs=before["materialization_inputs"])
        primary=None
        with owner.installed(env):
            try:
                phase("materialization")
                observed=observe_recorded_materialized_world(env,inputs=before["materialization_inputs"])
                require(digest(self.input_binding())==digest(before),"recorded constructor input changed")
                capture,root=observed["capture"],observed["snapshot"]
                owner.snapshot=root;owner.seal(capture)
            except BaseException as failure:
                primary=failure;raise
            finally:
                phase("cleanup")
                if env._process is not None:
                    try:env.close()
                    except BaseException as cleanup:
                        if primary is None:raise
                        primary.add_note("Recorded materialization cleanup failed: "+type(cleanup).__name__)
        require(digest(self.input_binding())==digest(before),"recorded materialization inputs changed after reap")
        return dict(**before,capture=capture,capture_sha256=digest(capture),produced_bridge_snapshot=root,
            produced_bridge_snapshot_sha256=digest(root),
            synthetic_process_fixture=getattr(self,"synthetic_process_fixture",False) is True,
            **{key:False for key in UNQUALIFIED})


def supervise_materialization_run(job,*,directory,binding_sha256,verify_admission,validate_result,
                                 contract=None,_synthetic_watchdog_seconds=None):
    contract=MaterializationRunContract() if contract is None else contract
    require(type(contract) is MaterializationRunContract,"distinct materialization contract required")
    contract.receipt()
    require((type(job) in (OwnedMaterializationJob,OwnedRecordedMaterializationJob)
        or isinstance(job,(OwnedMaterializationJob,OwnedRecordedMaterializationJob))
        and getattr(job,"synthetic_process_fixture",False) is True) and callable(validate_result),
        "canonical frozen materialization job or explicit synthetic fixture required")
    before=job.input_binding()
    def verify():
        require(job.input_binding()==before,"materialization launch/acceptance inputs changed")
        return verify_admission()
    def validate(result):
        _join_result(result,before,result["capture"])
        frozen=digest(result)
        require(validate_result(result) is not False,"independent materialization result refused")
        require(digest(result)==frozen and job.input_binding()==before,"materialization acceptance drift")
    return _supervise_owned_job(job,directory=directory,binding_sha256=binding_sha256,verify_admission=verify,
        contract=contract,kind="materialization-run",work_phase="materialization",validate_result=validate,
        _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)


def _join_result(result,expected,capture):
    require(type(result) is dict and set(result)==set(expected)|set(UNQUALIFIED)|{
        "capture","capture_sha256","produced_bridge_snapshot","produced_bridge_snapshot_sha256",
        "synthetic_process_fixture"} and all(digest(result.get(k))==digest(v) for k,v in expected.items())
        and digest(result["capture"])==digest(capture) and result["capture_sha256"]==digest(capture)
        and type(result["synthetic_process_fixture"]) is bool
        and all(result[k] is False for k in UNQUALIFIED),"exact unpromoted materialization worker join required")
    root=result["produced_bridge_snapshot"]
    require(type(root) is dict and result["produced_bridge_snapshot_sha256"]==digest(root),
        "observed raw bridge root binding required")
    read_materialization(capture,expected_inputs=expected["materialization_inputs"],
        process_records=capture["records"],expected_snapshot=root)


class MaterializationRunValidator(OwningRunValidator):
    """Independent same-child reader; no caller-supplied simulator root needed.

    Reuses only generic launch/wait4/close receipt law. The distinct input/result
    law requires the produced root to be the actual captured snapshot response.
    No full Python snapshot reconstruction or scientific qualification implied.
    """
    contract_class=MaterializationRunContract
    supervision_schema="pokezero.search-over-raw.materialization-run-supervision.v1"
    complete_status="MATERIALIZATION_RUN_COMPLETE"
    launch_schema=MaterializationProcessOwner.launch_schema
    close_schema=MaterializationProcessOwner.close_schema
    capture_hash_key="capture_sha256"
    work_phase="materialization"

    def __init__(self,*,directory,job_sha256,binding_sha256,expected_inputs,runtime,
                 allow_synthetic=False,contract=None):
        require(_sha(job_sha256) and _sha(binding_sha256) and type(allow_synthetic) is bool,
            "external materialization job/attempt bindings required")
        self.directory=Path(directory).resolve();self.job_sha=job_sha256;self.binding=binding_sha256
        self.allow_synthetic=allow_synthetic;self.expected=deepcopy(expected_inputs);self.runtime=deepcopy(runtime)
        _validate_inputs(self.expected)
        require(self.expected["runtime_sha256"]==digest(self.runtime),"external materialization runtime join required")
        contract=MaterializationRunContract() if contract is None else contract
        require(type(contract) is MaterializationRunContract,"distinct reader contract required")
        self.contract=contract.receipt();self.inputs=digest(dict(expected=self.expected,runtime=self.runtime))

    def _check_inputs(self):
        _validate_inputs(self.expected)
        require(digest(dict(expected=self.expected,runtime=self.runtime))==self.inputs,"external materialization input drift")

    def _join_result(self,result,capture,runtime_sha):
        require(runtime_sha==self.expected["runtime_sha256"],"materialization runtime drift")
        _join_result(result,self.expected,capture)

    def _match_capture(self,capture,records):
        # The raw root is taken from the observed response, whose byte digest
        # must match the independently stored process transport.
        read_materialization(capture,expected_inputs=self.expected["materialization_inputs"],
            process_records=records,expected_snapshot=capture["raw_events"][1]["snapshot"])
