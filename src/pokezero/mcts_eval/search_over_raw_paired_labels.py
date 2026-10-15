"""Eight raw/raw terminal probes of a retained paired native AFTER endpoint.

Separate from the legacy source-history route and reference prediction-input
equality. Native prediction/correspondence/source ownership are controller joins;
these labels alone establish none of those or scientific execution authority.
"""
from copy import deepcopy
from dataclasses import dataclass

from .search_over_raw import digest, require, _run_continuation_cell
from .search_over_raw_outcomes import (
    ContinuationCapture, OwnedContinuationLifecycle, RAW_LABEL_SCHEMA, RAW_LABEL_POLICY,
    validate_observed_outcome, paired_player_names,
)
from .search_over_raw_outcome_run import (
    OwnedContinuationJob, ContinuationProcessOwner, ContinuationRunContract,
    OwningRunValidator,
)
from .search_over_raw_paired_replay import context_identity
from .search_over_raw_supervision import _supervise_owned_job

SCHEMA="pokezero.search-over-raw.paired-native-after-labels.v1"


def label_contract(inputs):
    return dict(root_binding=inputs["root_binding"],root_id=inputs["root_binding"]["root_id"],
        namespace=inputs["namespace"],subject=inputs["subject"],max_boundaries=250,replicates=8,
        selection_seed=inputs["root_binding"]["source_seed"],policy=RAW_LABEL_POLICY,
        paired_player_names=inputs["paired_player_names"])


def read_labels(result,*,expected,snapshot):
    require(type(result) is dict and set(result)==set(expected)|{"bank","bank_sha256","outcomes",
        "synthetic_process_fixture","runtime_qualified","resource_cost_complete"}
        and all(digest(result[k])==digest(v) for k,v in expected.items())
        and result["bank_sha256"]==digest(result["bank"])
        and result["runtime_qualified"] is False and result["resource_cost_complete"] is False
        and type(result["synthetic_process_fixture"]) is bool,"exact paired label worker result required")
    bank=result["bank"];rows=result["outcomes"]
    require(bank["schema"]==RAW_LABEL_SCHEMA and bank["root_binding"]==expected["root_binding"]
        and bank["runtime_sha256"]==expected["runtime_sha256"]
        and type(rows) is list and len(rows)==len(bank["cells"])==8,
        "all eight original paired-label cells required")
    contract=label_contract(expected)
    for replicate,(row,cell) in enumerate(zip(rows,bank["cells"])):
        require(type(row["replicate"]) is int and row["replicate"]==replicate and row["action"]==0
            and validate_observed_outcome(row,contract,cell,expected_snapshot=snapshot,
                runtime_sha256=expected["runtime_sha256"])==digest(row),
            "original raw/raw terminal or capped-null label readback required")
    return deepcopy(rows)


@dataclass(frozen=True)
class PairedLabelRunContract(ContinuationRunContract):
    def receipt(self):
        return dict(super().receipt(),schema=SCHEMA+".contract",policy=RAW_LABEL_POLICY,
            attribution="actual_paired_serializer_and_parser_names",native_prediction_input_parity_required=False)


@dataclass(frozen=True)
class OwnedPairedLabelJob(OwnedContinuationJob):
    """Use only a separately owning-reader-validated designated AFTER snapshot."""
    def input_binding(self):
        value=super().input_binding()
        require(self.actions=={"raw":0} and type(self.root_binding.get("source_seed")) is int,
            "paired labels have one raw/raw cell per replicate and a frozen source seed")
        value.update(paired_endpoint=context_identity(self.snapshot),paired_player_names=paired_player_names(self.snapshot),
            paired_label_schema=SCHEMA,policy=RAW_LABEL_POLICY)
        return value

    def __call__(self,ownership,phase):
        from pathlib import Path
        phase("startup");before=self.input_binding();env,evaluator=self._components()
        observer=ContinuationCapture(self.root_binding,before["runtime_sha256"])
        lifecycle=OwnedContinuationLifecycle(env,expected_snapshot=self.snapshot,root_binding=self.root_binding)
        owner=ContinuationProcessOwner(directory=Path(ownership["directory"]).parent,ownership=ownership,
            runtime=self.runtime,config=self.config)
        rows=[];primary=None
        with owner.installed(env):
            try:
                phase("continuations")
                with observer.installed(env):
                    for replicate in range(8):
                        law=dict(root_id=self.root_binding["root_id"],action=0,replicate=replicate,
                            subject=self.subject,namespace=self.namespace,max_boundaries=250,policy=RAW_LABEL_POLICY)
                        lifecycle.prepare(self.snapshot,law)
                        try:
                            observer.begin_cell(self.snapshot,**law)
                            outcome=_run_continuation_cell(env,self.snapshot,self.subject,0,replicate,evaluator,
                                self.namespace,self.root_binding["root_id"],250,observer,lifecycle,raw_from_start=True)
                            lifecycle.release()
                        except BaseException as error:
                            lifecycle.abort(error);raise
                        row=dict(root_id=self.root_binding["root_id"],action=0,replicate=replicate,**outcome)
                        observer.finish_cell(row);rows.append(row)
                bank=dict(observer.export(),schema=RAW_LABEL_SCHEMA)
                owner.seal(bank)
            except BaseException as error:
                primary=error;raise
            finally:
                phase("cleanup")
                if env._process is not None:
                    try:env.close()
                    except BaseException as cleanup:
                        if primary is None:raise
                        primary.add_note("Paired label cleanup failed: "+type(cleanup).__name__)
        require(digest(self.input_binding())==digest(before),"paired label inputs changed during execution")
        result=dict(**before,bank=bank,bank_sha256=digest(bank),outcomes=rows,
            synthetic_process_fixture=getattr(self,"synthetic_process_fixture",False) is True,
            runtime_qualified=False,resource_cost_complete=False)
        read_labels(result,expected=before,snapshot=self.snapshot)
        return result


def supervise_paired_label_run(job,*,directory,binding_sha256,verify_admission,validate_result,contract=None,
        _synthetic_watchdog_seconds=None):
    contract=PairedLabelRunContract() if contract is None else contract
    require(type(contract) is PairedLabelRunContract and (type(job) is OwnedPairedLabelJob or
        isinstance(job,OwnedPairedLabelJob) and getattr(job,"synthetic_process_fixture",False) is True),
        "distinct paired-label job and contract required")
    before=job.input_binding()
    def verify():
        require(digest(job.input_binding())==digest(before),"paired label admission drift")
        return verify_admission()
    def validate(result):
        read_labels(result,expected=before,snapshot=job.snapshot);frozen=digest(result)
        require(validate_result(result) is not False and digest(result)==frozen
            and digest(job.input_binding())==digest(before),"paired label acceptance refused or changed")
    return _supervise_owned_job(job,directory=directory,binding_sha256=binding_sha256,verify_admission=verify,
        contract=contract,kind="outcome-run",work_phase="continuations",validate_result=validate,
        _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)


class PairedLabelRunValidator(OwningRunValidator):
    contract_class=PairedLabelRunContract

    def __init__(self,*,expected_inputs,**kwargs):
        self.expected=deepcopy(expected_inputs)
        self.expected_sha=digest(self.expected)
        super().__init__(**kwargs)

    def _check_inputs(self):
        super()._check_inputs()
        require(digest(self.expected)==self.expected_sha and self.expected["root_binding"]==self.root
            and self.expected["paired_endpoint"]==context_identity(self.snapshot)
            and self.expected["paired_player_names"]==paired_player_names(self.snapshot)
            and self.expected["runtime_sha256"]==digest(self.runtime),"original paired label reader inputs drift")

    def _join_result(self,result,bank,runtime_sha):
        super()._join_result(result,bank,runtime_sha)
        read_labels(result,expected=self.expected,snapshot=self.snapshot)


def prepare_paired_label_job(replay_reader,replay_result,*,config,runtime,root_binding,subject,namespace):
    """Read original replay ownership immediately; no prefix/JSON snapshot input."""
    from .search_over_raw_paired_replay import PairedPrefixRunValidator
    require(type(replay_reader) is PairedPrefixRunValidator,"original paired replay owning reader required")
    snapshot=replay_reader.retained_endpoint(replay_result)
    require(digest(runtime)==digest(replay_reader.runtime),"label runtime differs from original paired replay")
    require(type(root_binding) is dict and "paired_replay_origin" not in root_binding,
        "caller cannot replace the original replay-to-label provenance join")
    root=deepcopy(root_binding)
    root["paired_replay_origin"]=dict(result_sha256=digest(replay_result),
        capture_sha256=replay_result["capture_sha256"],inputs_sha256=digest(replay_reader.expected),
        native_origin=deepcopy(replay_reader.expected["root_origin"]),
        designated_native_boundary=deepcopy(replay_reader.expected["native_boundaries"][-1]["binding"]))
    job=OwnedPairedLabelJob(config,snapshot,root,deepcopy(runtime),subject,{"raw":0},namespace)
    job.input_binding()
    return job
