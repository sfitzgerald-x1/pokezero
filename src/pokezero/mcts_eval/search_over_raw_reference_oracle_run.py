"""Explicit twenty-worker A1 original-team observation and timed ownership.

No collection, retry, holdout or runtime qualification. Original-team projection
is transported, never a private source snapshot. Archive origin is a separate
trusted controller join; exported hashes are observed consistency, not physics.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass,replace

from .search_over_raw import SearchConfiguration,digest,require
from .search_over_raw_outcomes import _sha
from .search_over_raw_oracle import (TeamOracle,OracleRootRequest,OracleWorkerFactory,
    DiagnosticTeamOracleSearchAdapter,_OracleSampleWorld)
from .search_over_raw_world_inputs import (WorldInputRuntime,_WorldInputAdapterMixin,
    public_request_witness,_state_payload)
from .search_over_raw_reference_timed_run import (ReferenceTimedRunContract,ReferenceTimedAdapterJob,
    ReferenceTimedRunValidator,_runtime_law)
from .search_over_raw_timed_reference import _read_reference_timed_selection
from .search_over_raw_supervision import _supervise_owned_job

SCHEMA="pokezero.search-over-raw.reference-oracle-consumption.v1"
REFERENCE_ORACLE_CAPTURE=dict(schema=SCHEMA,original_opponent_team_only=True,
    at_actual_preparation=True,at_each_actual_sampling_attempt=True,at_every_worker_export=True,
    at_owned_materialized_draw=True,
    includes_workers_without_draws=True,installed_before_chance_planning=True,
    no_sampled_fallback=True,deployable=False,runtime_qualified=False)


class ReferenceOracleWorldInputRuntime(WorldInputRuntime):
    def __init__(self,base,*,expected_oracle_receipt,expected_oracle_payload_sha256,
            expected_reconstruction_catalog_sha256,**kwargs):
        super().__init__(base,**kwargs)
        require(self.expected_public_request_sha256 is not None
            and type(expected_oracle_receipt) is dict and _sha(expected_oracle_payload_sha256)
            and _sha(expected_reconstruction_catalog_sha256),"prospective reference original-team bindings required")
        self.oracle_receipt=deepcopy(expected_oracle_receipt)
        self.oracle_sha=expected_oracle_payload_sha256
        self.catalog_sha=expected_reconstruction_catalog_sha256

    def _public_input(self,request):
        require(type(request) is OracleRootRequest and type(request.oracle) is TeamOracle
            and digest(request.oracle.receipt()) == digest(self.oracle_receipt)
            and digest(_state_payload(request.oracle)) == self.oracle_sha,
            "actual reference worker original-team wrapper drift")
        require(digest(_state_payload(self.base.base.source)) == self.catalog_sha,
            "actual reference worker reconstruction catalog drift")
        return request.public

    def _wrap_sample_world(self,original,sample):
        # The outer observer must preserve writes such as sampling_deadline_at
        # and phase-timing attributes on the guarded underlying world factory.
        return _OracleSampleWorld(original,sample)

    def prepare(self,request):
        prepared=super().prepare(request)  # Actual preparation keeps the oracle wrapper.
        factory=self.base.base.prepared_factory
        factory.enable_team_diagnostics(request.oracle.root_binding)
        original=prepared.evidence
        def evidence():
            self._public_input(request)
            require(factory.team_diagnostic_root_binding == request.oracle.root_binding
                and factory.collect_sampled_team_traits is True,
                "reference actual original-team materialization observer drift")
            result=original()
            self._public_input(request)
            require("worker_original_team_preparation" not in result,
                "reference original-team worker witness collision")
            prep=result.get("team_oracle_preparation")
            require(type(prep) is dict and prep.get("oracle_payload_sha256") == self.oracle_sha
                and prep.get("reconstruction_catalog_sha256") == self.catalog_sha
                and digest(prep.get("oracle")) == digest(self.oracle_receipt),
                "actual reference preparation differs from prospective original team/catalog")
            result["worker_original_team_preparation"]=dict(worker=self.worker_index,
                preparation_sha256=digest(prep),materialized_draw_observer=True,
                root_binding=request.oracle.root_binding)
            return result
        return replace(prepared,evidence=evidence)


@dataclass(frozen=True)
class ReferenceOracleWorldInputWorkerFactory:
    base: OracleWorkerFactory
    expected_public_request_sha256: str
    expected_oracle_receipt: dict
    expected_oracle_payload_sha256: str
    expected_reconstruction_catalog_sha256: str

    def __post_init__(self):
        require(type(self.base) is OracleWorkerFactory and all(_sha(v) for v in (
            self.expected_public_request_sha256,self.expected_oracle_payload_sha256,
            self.expected_reconstruction_catalog_sha256)),"distinct actual original-team worker factory required")

    def __call__(self,index):
        return ReferenceOracleWorldInputRuntime(self.base(index),worker_index=index,
            expected_public_request_sha256=self.expected_public_request_sha256,
            expected_oracle_receipt=self.expected_oracle_receipt,
            expected_oracle_payload_sha256=self.expected_oracle_payload_sha256,
            expected_reconstruction_catalog_sha256=self.expected_reconstruction_catalog_sha256)


class ReferenceTimedOracleSearchAdapter(_WorldInputAdapterMixin,DiagnosticTeamOracleSearchAdapter):
    def _check_world_input_scope(self,configuration):
        _configuration(configuration)

    def __init__(self,configuration,*,expected_public_request_sha256,
            expected_reconstruction_catalog_sha256,**kwargs):
        require(_sha(expected_reconstruction_catalog_sha256)
            and kwargs.get("value_diagnostics") is None and kwargs.get("native_ancestry_contract") is None
            and kwargs.get("owned_process_receipts") is not None,
            "reference A1 requires full catalog and indexed owned workers, not label/ancestry lanes")
        self._expected_catalog_sha=expected_reconstruction_catalog_sha256
        super().__init__(configuration,expected_public_request_sha256=expected_public_request_sha256,
            timed_reference_atomic_work=True,**kwargs)
        require(digest(_state_payload(self._oracle_source)) == self._expected_catalog_sha,
            "actual reference oracle adapter catalog differs")
        self.runtime_configuration["reference_team_oracle_observation"]=deepcopy(REFERENCE_ORACLE_CAPTURE)
        self.runtime_configuration["oracle_payload_sha256"]=digest(_state_payload(self.oracle))
        self.runtime_configuration["oracle_reconstruction_catalog_sha256"]=self._expected_catalog_sha
        self.runtime_sha256=digest(self.runtime_configuration)

    def _reference_worker_factory(self,factory):
        base=DiagnosticTeamOracleSearchAdapter._reference_worker_factory(self,factory)
        return ReferenceOracleWorldInputWorkerFactory(base,self._expected_public_request_sha256,
            self.oracle.receipt(),digest(_state_payload(self.oracle)),self._expected_catalog_sha)

    def _selection_evidence(self,evidence,request):
        result=super()._selection_evidence(evidence,request)
        validate_reference_oracle_inputs(result,oracle_receipt=self.oracle.receipt(),
            public_request=public_request_witness(request),
            expected_oracle_payload_sha256=digest(_state_payload(self.oracle)),
            expected_reconstruction_catalog_sha256=self._expected_catalog_sha)
        return result


def validate_reference_oracle_inputs(evidence,*,oracle_receipt,public_request,
        expected_oracle_payload_sha256,expected_reconstruction_catalog_sha256):
    """All twenty actual preparations and every OWN export, with empty work retained."""
    before=digest(dict(evidence=evidence,oracle=oracle_receipt,public=public_request,
        payload=expected_oracle_payload_sha256,catalog=expected_reconstruction_catalog_sha256))
    binding=public_request["consumed_binding"]
    require(type(oracle_receipt) is dict and set(oracle_receipt) == {"schema","information_scope",
        "root_binding","own_team_sha256","opponent_team_sha256","set_source_hash","deployable",
        "source_snapshot_transported","committed_opponent_action_transported"}
        and oracle_receipt["schema"] == "pokezero.search-over-raw.team-oracle.v1"
        and oracle_receipt["information_scope"] == "original_opponent_team_only"
        and oracle_receipt["root_binding"] == binding["root_binding"]
        and oracle_receipt["set_source_hash"] == binding["set_source_hash"]
        and all(_sha(oracle_receipt[k]) for k in ("own_team_sha256","opponent_team_sha256"))
        and all(oracle_receipt[k] is False for k in ("deployable","source_snapshot_transported",
            "committed_opponent_action_transported"))
        and all(_sha(v) for v in (expected_oracle_payload_sha256,expected_reconstruction_catalog_sha256))
        and digest(evidence.get("team_oracle")) == digest(oracle_receipt),
        "reference external original-team/public/catalog scope drift")
    workers={};exports=attempts=consumed=0
    require(type(evidence.get("worker_receipts")) is list,"actual reference OWN exports required")
    for row in evidence["worker_receipts"]:
        worker=row.get("worker");raw=row.get("evidence",{});prep=raw.get("team_oracle_preparation")
        observed=raw.get("worker_original_team_preparation")
        require(type(worker) is int and 0 <= worker < 20 and type(observed) is dict
            and set(observed) == {"worker","preparation_sha256","materialized_draw_observer","root_binding"}
            and observed["materialized_draw_observer"] is True
            and observed["root_binding"] == oracle_receipt["root_binding"] and type(observed["worker"]) is int
            and observed["worker"] == worker and observed["preparation_sha256"] == digest(prep)
            and type(prep) is dict and set(prep) == {"schema","oracle","public_request_witness",
                "actual_sampler_team_sha256","actual_construction_sha256","oracle_payload_sha256",
                "reconstruction_catalog_sha256","installed_before_chance_planning",
                "source_snapshot_transported","committed_opponent_action_transported","deployable","runtime_qualified"}
            and prep["schema"] == "pokezero.search-over-raw.reference-oracle-preparation.v1"
            and digest(prep["oracle"]) == digest(oracle_receipt)
            and digest(raw.get("team_oracle")) == digest(oracle_receipt)
            and digest(prep["public_request_witness"]) == digest(public_request)
            and prep["actual_sampler_team_sha256"] == oracle_receipt["opponent_team_sha256"]
            and _sha(prep["actual_construction_sha256"])
            and prep["oracle_payload_sha256"] == expected_oracle_payload_sha256
            and prep["reconstruction_catalog_sha256"] == expected_reconstruction_catalog_sha256
            and prep["installed_before_chance_planning"] is True
            and all(prep[k] is False for k in ("source_snapshot_transported",
                "committed_opponent_action_transported","deployable","runtime_qualified")),
            "reference original-team preparation/export/index differs from external truth")
        require(worker not in workers or workers[worker] == digest(prep),
            "reference worker changed its preparation between OWN exports")
        workers[worker]=digest(prep);exports+=1
        # Incremental exports contain only their actual draw slice; never infer
        # a draw from a declaration or require an unused future sampled prefix.
        require(type(raw.get("draws")) is list,"actual oracle draw roster required")
        for draw in raw["draws"]:
            require(draw.get("diagnostic_oracle_team_sha256") == oracle_receipt["opponent_team_sha256"]
                and draw.get("information_scope") == "original_opponent_team_only",
                "reference actual draw lost original-team truth")
            require("packed_team_sha256" not in draw or
                draw["packed_team_sha256"] == oracle_receipt["opponent_team_sha256"],
                "reference actual sampled team contradicts original team")
            attempts+=1
            if draw.get("consumed_world_input") is not None:
                from .search_over_raw_belief_diagnostics import selected_team_origin,validate_members
                origin=selected_team_origin(draw)
                actual=draw.get("sampled_original_team")
                require(type(actual) is dict and actual.get("schema") == "pokezero.search-over-raw.sampled-original-team.v1"
                    and actual.get("role") == "HYPOTHETICAL_DRAW_ONLY"
                    and actual.get("root_binding") == binding["root_binding"]
                    and actual.get("subject") == binding["subject"]
                    and actual.get("set_source_hash") == binding["set_source_hash"]
                    and actual.get("information_key") == binding["information_key"]
                    and actual.get("original_team_sha256") == origin["original_team_sha256"]
                        == oracle_receipt["opponent_team_sha256"]
                    and digest(actual.get("members")) == actual.get("members_sha256") == origin["members_sha256"]
                    and actual.get("instrumentation_can_change_deadlines") is True
                    and actual.get("qualifies_uninstrumented_runtime") is False,
                    "consumed reference world lacks actual selected-team materialization observation")
                validate_members(actual["members"])
                consumed+=1
    require(set(workers) == set(range(20)) and before == digest(dict(evidence=evidence,
        oracle=oracle_receipt,public=public_request,payload=expected_oracle_payload_sha256,
        catalog=expected_reconstruction_catalog_sha256)),"reference original-team roster or late input drift")
    return dict(schema=SCHEMA,prepared_workers=20,worker_exports=exports,attempted_draws=attempts,
        consumed_worlds=consumed,team_consumed=consumed > 0,original_team_preparation_validated=True,
        observed_draw_truth_validated=True,atomic_prefix_validated=False,owning_process_validated=False,
        deployable=False,runtime_qualified=False,scientific_collection_authorized=False)


@dataclass(frozen=True)
class ReferenceOracleTimedRunContract(ReferenceTimedRunContract):
    def receipt(self):
        return dict(**{**super().receipt(),"schema":"pokezero.search-over-raw.reference-oracle-timed-contract.v1",
            "decision_ceiling_seconds":[1]},information_scope="original_opponent_team_only",deployable=False)


def _configuration(configuration):
    require(type(configuration) is SearchConfiguration and configuration.arm == "reference"
        and configuration.belief == "oracle" and configuration.leaf == "model"
        and configuration.workers == 20 and configuration.seconds == 1.,
        "A1 reference original-team model/one-second/twenty-worker configuration required")


def _runtime(configuration,runtime,runtime_sha):
    _runtime_law(configuration,runtime,runtime_sha,configuration_check=_configuration)
    require("full_public_request_observation" in runtime
        and digest(runtime.get("reference_team_oracle_observation")) == digest(REFERENCE_ORACLE_CAPTURE)
        and _sha(runtime.get("oracle_payload_sha256"))
        and _sha(runtime.get("oracle_reconstruction_catalog_sha256")),
        "A1 reference full original-team observation required")


def read_reference_oracle_timed_selection(selected,*,oracle_receipt,public_request,
        expected_oracle_payload_sha256,expected_reconstruction_catalog_sha256,contract,**kwargs):
    def inputs():
        return digest(dict(selected=selected,oracle=oracle_receipt,public=public_request,
            payload=expected_oracle_payload_sha256,catalog=expected_reconstruction_catalog_sha256,
            contract=contract.receipt(),arguments=_state_payload(kwargs)))
    require(type(contract) is ReferenceOracleTimedRunContract,"distinct reference A1 timed contract required")
    before=inputs()
    runtime=kwargs["runtime_configuration"]
    _runtime(kwargs["configuration"],runtime,kwargs["runtime_sha256"])
    require(digest(runtime.get("team_oracle")) == digest(oracle_receipt)
        and runtime["oracle_payload_sha256"] == expected_oracle_payload_sha256
        and runtime["oracle_reconstruction_catalog_sha256"] == expected_reconstruction_catalog_sha256
        and kwargs.get("public_request_sha256") == public_request.get("public_request_sha256")
        and digest(kwargs.get("consumed_binding")) == digest(public_request.get("consumed_binding"))
        and digest(selected.get("reference_timed_run_contract")) == digest(contract.receipt())
        and digest(selected.get("observed_runtime_configuration")) == digest(runtime),
        "reference A1 prospective runtime/truth/public/contract bindings differ")
    accounting=_read_reference_timed_selection(selected,configuration_check=_configuration,**kwargs)
    truth=validate_reference_oracle_inputs(selected["evidence"],oracle_receipt=oracle_receipt,
        public_request=public_request,expected_oracle_payload_sha256=expected_oracle_payload_sha256,
        expected_reconstruction_catalog_sha256=expected_reconstruction_catalog_sha256)
    require(inputs() == before,"reference A1 selection/external arguments changed during read")
    return dict(**{**accounting,"schema":"pokezero.search-over-raw.reference-oracle-timed-observed-work.v1"},
        original_team_consumption=truth,deployable=False,archive_truth_correspondence_validated=False)


@dataclass(frozen=True,kw_only=True)
class ReferenceOracleTimedAdapterJob(ReferenceTimedAdapterJob):
    oracle: TeamOracle
    expected_oracle_receipt: dict
    expected_reconstruction_catalog_sha256: str
    contract: ReferenceOracleTimedRunContract=ReferenceOracleTimedRunContract()
    _runtime_check=staticmethod(_runtime)
    _contract_class=ReferenceOracleTimedRunContract

    def __post_init__(self):
        require(type(self.oracle) is TeamOracle and type(self.expected_oracle_receipt) is dict
            and _sha(self.expected_reconstruction_catalog_sha256) and _sha(self.expected_public_request_sha256),
            "prospective exact reference A1 original-team/public/catalog inputs required")
        object.__setattr__(self,"oracle",deepcopy(self.oracle))
        object.__setattr__(self,"expected_oracle_receipt",deepcopy(self.expected_oracle_receipt))
        require(digest(self.oracle.receipt()) == digest(self.expected_oracle_receipt)
            == digest(self.expected_runtime_configuration.get("team_oracle"))
            and self.expected_runtime_configuration.get("oracle_payload_sha256") == digest(_state_payload(self.oracle))
            and self.expected_runtime_configuration.get("oracle_reconstruction_catalog_sha256")
                == self.expected_reconstruction_catalog_sha256,"reference A1 runtime original team/catalog differs")
        super().__post_init__()
        require(self.oracle.root_binding == self._consumed_binding["root_binding"]
            and self.oracle.subject == self._consumed_binding["subject"]
            and self.oracle.set_source_hash == self._consumed_binding["set_source_hash"],
            "reference A1 team projection differs from public root")

    def _inputs(self):
        return digest(dict(lifecycle=super()._inputs(),oracle=_state_payload(self.oracle),
            oracle_receipt=self.expected_oracle_receipt,catalog=self.expected_reconstruction_catalog_sha256))

    def _check_source(self,source):
        super()._check_source(source)
        require(digest(_state_payload(source)) == self.expected_reconstruction_catalog_sha256,
            "actual reference A1 catalog differs from external catalog")
        self.oracle.validate(self._public_request,source)

    def _make_adapter(self,ownership):
        return ReferenceTimedOracleSearchAdapter(self.configuration,oracle=self.oracle,
            checkpoint_contract=self.checkpoint_contract,showdown_root=self.showdown_root,
            reference_factory=self.reference_factory,initial_dispatch_workers=6,owned_process_receipts=ownership,
            expected_public_request_sha256=self.expected_public_request_sha256,
            expected_reconstruction_catalog_sha256=self.expected_reconstruction_catalog_sha256)

    def _check_adapter_scope(self,adapter):
        require(self._inputs() == self._frozen_binding and type(adapter.oracle) is TeamOracle
            and digest(_state_payload(adapter.oracle)) == digest(_state_payload(self.oracle)),
            "actual reference A1 team inputs changed")
        self._check_source(adapter._oracle_source)

    def read_selection(self,selected):
        require(self._inputs() == self._frozen_binding,"reference A1 owning job inputs changed")
        return read_reference_oracle_timed_selection(selected,oracle_receipt=self.expected_oracle_receipt,
            public_request=public_request_witness(self._public_request),
            expected_oracle_payload_sha256=digest(_state_payload(self.oracle)),
            expected_reconstruction_catalog_sha256=self.expected_reconstruction_catalog_sha256,
            contract=self.contract,configuration=self.configuration,runtime_configuration=self.expected_runtime_configuration,
            runtime_sha256=self.expected_runtime_sha256,root_id=self.root_id,statistics_root_id=self.statistics_root_id,
            selection_seed=self.selection_seed,consumed_binding=self._consumed_binding,
            legal_action_mask=self._legal_action_mask,public_request_sha256=self.expected_public_request_sha256)


def supervise_reference_oracle_timed(job,*,directory,binding_sha256,verify_admission,
        _synthetic_watchdog_seconds=None):
    require(type(job) is ReferenceOracleTimedAdapterJob,"distinct reference A1 owning job required")
    return _supervise_owned_job(job,directory=directory,binding_sha256=binding_sha256,
        verify_admission=verify_admission,contract=job.contract,kind="reference-oracle-timed-run",
        work_phase="selection",validate_result=job.read_selection,
        _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)


class ReferenceOracleTimedRunValidator(ReferenceTimedRunValidator):
    _runtime_check=staticmethod(_runtime)
    _contract_class=ReferenceOracleTimedRunContract
    _kind="reference-oracle-timed-run"
    _complete_status="REFERENCE_ORACLE_TIMED_RUN_COMPLETE"
    _readout_schema="pokezero.search-over-raw.reference-oracle-timed-owning-readout.v1"

    def __init__(self,*,oracle_receipt,public_request,expected_oracle_payload_sha256,
            expected_reconstruction_catalog_sha256,**kwargs):
        require(type(oracle_receipt) is dict and type(public_request) is dict
            and all(_sha(v) for v in (expected_oracle_payload_sha256,expected_reconstruction_catalog_sha256))
            and _sha(kwargs.get("public_request_sha256"))
            and kwargs["public_request_sha256"] == public_request.get("public_request_sha256")
            and digest(kwargs.get("consumed_binding")) == digest(public_request.get("consumed_binding")),
            "external reference A1 team/public/catalog inputs required")
        self.oracle_receipt=deepcopy(oracle_receipt);self.public_request=deepcopy(public_request)
        self.oracle_sha=expected_oracle_payload_sha256;self.catalog_sha=expected_reconstruction_catalog_sha256
        super().__init__(**kwargs)

    def _inputs(self):
        return digest(dict(lifecycle=super()._inputs(),oracle=self.oracle_receipt,public=self.public_request,
            payload=self.oracle_sha,catalog=self.catalog_sha))

    def _read_work(self,saved):
        return read_reference_oracle_timed_selection(saved,oracle_receipt=self.oracle_receipt,
            public_request=self.public_request,expected_oracle_payload_sha256=self.oracle_sha,
            expected_reconstruction_catalog_sha256=self.catalog_sha,contract=self.contract,
            configuration=self.configuration,runtime_configuration=self.runtime,runtime_sha256=self.runtime_sha,
            root_id=self.root_id,statistics_root_id=self.statistics_root_id,selection_seed=self.seed,
            consumed_binding=self.consumed,legal_action_mask=self.mask,
            public_request_sha256=self.reference_public_request_sha)
