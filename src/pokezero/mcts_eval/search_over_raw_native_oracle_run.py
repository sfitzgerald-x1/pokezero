"""Distinct, nondeployable A1 native timed ownership and independent reader.

This transports a previously extracted original-team projection, not an archive
or private snapshot. Its origin must be validated separately by the archive
reader. No collection, retry, stage admission or runtime qualification follows.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass

from .search_over_raw import SearchConfiguration, digest, require
from .search_over_raw_outcomes import _sha
from .search_over_raw_oracle import TeamOracle, opening_team
from .search_over_raw_native_oracle import (NativeTimedOracleSearchAdapter,
    NATIVE_ORACLE_CAPTURE, validate_native_oracle_inputs)
from .search_over_raw_world_inputs import _state_payload, public_request_witness
from .search_over_raw_construction import SCHEMA as CONSTRUCTION_SCHEMA, construction_source_hashes
from .search_over_raw_timed_run import (NativeTimedRunContract, NativeTimedAdapterJob,
    NativeTimedRunValidator, _runtime_law, _read_native_timed_selection)
from .search_over_raw_supervision import _supervise_owned_job


@dataclass(frozen=True)
class NativeOracleTimedRunContract(NativeTimedRunContract):
    def receipt(self):
        return dict(**{**super().receipt(),
            "schema":"pokezero.search-over-raw.native-oracle-timed-contract.v1",
            "decision_ceiling_seconds":[1]}, information_scope="original_opponent_team_only",
            deployable=False, archive_truth_correspondence_validated=False)


def _configuration(configuration):
    require(type(configuration) is SearchConfiguration and configuration.arm == "incumbent"
        and configuration.belief == "oracle" and configuration.leaf == "model"
        and configuration.workers == 1 and configuration.seconds == 1.,
        "A1 native original-team model/one-second/single-worker configuration required")


def _runtime(configuration, runtime, runtime_sha):
    _runtime_law(configuration, runtime, runtime_sha, configuration_check=_configuration)
    require("full_public_request_observation" in runtime
        and digest(runtime.get("native_team_oracle_observation")) == digest(NATIVE_ORACLE_CAPTURE)
        and digest(runtime.get("native_oracle_construction")) == digest(dict(
            schema=CONSTRUCTION_SCHEMA, source_hashes=construction_source_hashes(),
            private_controller_only=True, instrumentation_can_change_deadlines=True, runtime_qualified=False)),
        "A1 native original-team and actual constructor observation required")


def read_native_oracle_timed_selection(selected, *, oracle_receipt, public_request,
        expected_override_sha256, expected_reconstruction_catalog_sha256, **kwargs):
    """Same native prefix/selector arithmetic, plus an external original-team join."""
    before=digest(dict(selected=selected,oracle=oracle_receipt,public=public_request,
        override=expected_override_sha256,catalog=expected_reconstruction_catalog_sha256,inputs=_state_payload(kwargs)))
    require(_sha(expected_override_sha256) and _sha(expected_reconstruction_catalog_sha256)
        and kwargs.get("public_request_sha256") == public_request.get("public_request_sha256")
        and digest(kwargs.get("consumed_binding")) == digest(public_request.get("consumed_binding"))
        and digest(kwargs.get("runtime_configuration",{}).get("team_oracle")) == digest(oracle_receipt),
        "external A1 timed public/catalog/override bindings required")
    accounting=_read_native_timed_selection(selected, runtime_check=_runtime,
        contract_class=NativeOracleTimedRunContract, **kwargs)
    evidence=selected["evidence"]
    require(digest(evidence.get("team_oracle")) == digest(oracle_receipt),
        "A1 selection original-team receipt differs from external truth projection")
    truth=validate_native_oracle_inputs(evidence["consumed_world_inputs"],oracle_receipt=oracle_receipt,
        public_request=public_request,expected_override_sha256=expected_override_sha256,
        expected_reconstruction_catalog_sha256=expected_reconstruction_catalog_sha256)
    require(before == digest(dict(selected=selected,oracle=oracle_receipt,public=public_request,
        override=expected_override_sha256,catalog=expected_reconstruction_catalog_sha256,inputs=_state_payload(kwargs))),
        "A1 native timed reader inputs changed during validation")
    return dict(**{**accounting,"schema":"pokezero.search-over-raw.native-oracle-timed-observed-work.v1"},
        original_team_consumption=truth, deployable=False,
        archive_truth_correspondence_validated=False)


@dataclass(frozen=True, kw_only=True)
class NativeOracleTimedAdapterJob(NativeTimedAdapterJob):
    oracle: TeamOracle
    expected_oracle_receipt: dict
    expected_override_sha256: str
    expected_reconstruction_catalog_sha256: str
    contract: NativeOracleTimedRunContract = NativeOracleTimedRunContract()
    _configuration_check=staticmethod(_configuration)
    _runtime_check=staticmethod(_runtime)
    _contract_class=NativeOracleTimedRunContract

    def __post_init__(self):
        require(type(self.oracle) is TeamOracle and type(self.expected_oracle_receipt) is dict
            and _sha(self.expected_override_sha256) and _sha(self.expected_reconstruction_catalog_sha256)
            and _sha(self.expected_public_request_sha256), "exact prospective A1 team/public bindings required")
        object.__setattr__(self,"oracle",deepcopy(self.oracle))
        object.__setattr__(self,"expected_oracle_receipt",deepcopy(self.expected_oracle_receipt))
        require(digest(self.oracle.receipt()) == digest(self.expected_oracle_receipt),
            "A1 job original-team projection differs from external receipt")
        super().__post_init__()
        witness=public_request_witness(self._public_request)
        require(self.oracle.root_binding == self._consumed_binding["root_binding"]
            and self.oracle.subject == self._consumed_binding["subject"]
            and self.oracle.set_source_hash == self._consumed_binding["set_source_hash"]
            and witness["public_request_sha256"] == self.expected_public_request_sha256
            and digest(self.expected_runtime_configuration.get("team_oracle")) == digest(self.expected_oracle_receipt),
            "A1 team projection differs from prospective public root")

    def _inputs(self):
        return digest(dict(lifecycle=super()._inputs(),oracle_payload=_state_payload(self.oracle),
            oracle_receipt=self.expected_oracle_receipt,override=self.expected_override_sha256,
            reconstruction_catalog=self.expected_reconstruction_catalog_sha256))

    def _check_source(self, source):
        super()._check_source(source)
        require(digest(_state_payload(source)) == self.expected_reconstruction_catalog_sha256,
            "actual A1 reconstruction catalog differs from external catalog digest")
        self.oracle.validate(self._public_request,source)
        from ..env import BattleStartOverride
        from ..showdown_fixture import pack_team
        own=opening_team(self._public_request.state.self_initial_request,self.oracle.subject,source)
        opponent="p2" if self.oracle.subject == "p1" else "p1"
        override=BattleStartOverride(player_teams={self.oracle.subject:pack_team(own),
            opponent:pack_team(self.oracle.opponent_team)},observation_format_id="gen3randombattle")
        require(digest(asdict(override)) == self.expected_override_sha256,
            "actual A1 reconstructed override differs from external original teams")

    def _make_adapter(self):
        return NativeTimedOracleSearchAdapter(self.configuration,oracle=self.oracle,
            expected_public_request_sha256=self.expected_public_request_sha256,
            checkpoint_contract=self.checkpoint_contract,showdown_root=self.showdown_root)

    def _check_adapter_scope(self, adapter):
        require(self._inputs() == self._frozen_inputs and type(adapter.oracle) is TeamOracle
            and digest(_state_payload(adapter.oracle)) == digest(_state_payload(self.oracle)),
            "actual A1 adapter original-team projection drift")
        self._check_source(adapter._oracle_source)

    def read_selection(self, selected):
        require(self._inputs() == self._frozen_inputs,"A1 native timed job inputs changed")
        return read_native_oracle_timed_selection(selected,oracle_receipt=self.expected_oracle_receipt,
            public_request=public_request_witness(self._public_request),
            expected_override_sha256=self.expected_override_sha256,
            expected_reconstruction_catalog_sha256=self.expected_reconstruction_catalog_sha256,
            configuration=self.configuration,runtime_configuration=self.expected_runtime_configuration,
            runtime_sha256=self.expected_runtime_sha256,root_id=self.root_id,
            statistics_root_id=self.statistics_root_id,selection_seed=self.selection_seed,
            consumed_binding=self._consumed_binding,legal_action_mask=self._legal_action_mask,
            decision_round=self.context.decision_round_index,contract=self.contract,
            public_request_sha256=self.expected_public_request_sha256)


def supervise_native_oracle_timed(job, *, directory, binding_sha256, verify_admission,
                                  _synthetic_watchdog_seconds=None):
    require(type(job) is NativeOracleTimedAdapterJob,"distinct A1 native timed owning job required")
    return _supervise_owned_job(job,directory=directory,binding_sha256=binding_sha256,
        verify_admission=verify_admission,contract=job.contract,kind="native-oracle-timed-run",
        work_phase="selection",validate_result=job.read_selection,
        _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)


class NativeOracleTimedRunValidator(NativeTimedRunValidator):
    _runtime_check=staticmethod(_runtime)
    _contract_class=NativeOracleTimedRunContract
    _kind="native-oracle-timed-run"
    _complete_status="NATIVE_ORACLE_TIMED_RUN_COMPLETE"
    _readout_schema="pokezero.search-over-raw.native-oracle-timed-owning-readout.v1"

    def __init__(self, *, oracle_receipt, public_request, expected_override_sha256,
            expected_reconstruction_catalog_sha256, **kwargs):
        require(_sha(expected_override_sha256) and _sha(expected_reconstruction_catalog_sha256)
            and type(oracle_receipt) is dict and type(public_request) is dict
            and _sha(kwargs.get("public_request_sha256"))
            and kwargs["public_request_sha256"] == public_request.get("public_request_sha256")
            and digest(kwargs.get("consumed_binding")) == digest(public_request.get("consumed_binding")),
            "A1 owning reader external truth/public/catalog bindings required")
        self.oracle_receipt=deepcopy(oracle_receipt);self.public_request=deepcopy(public_request)
        self.override_sha=expected_override_sha256;self.catalog_sha=expected_reconstruction_catalog_sha256
        super().__init__(**kwargs)

    def _inputs(self):
        return digest(dict(lifecycle=super()._inputs(),oracle_receipt=self.oracle_receipt,
            public_request=self.public_request,override=self.override_sha,catalog=self.catalog_sha))

    def _files(self):
        super()._files()
        from pathlib import Path
        import hashlib
        # Explicit byte inventory of this reader and observer, not a claim that
        # the running interpreter loaded a complete native dependency closure.
        import sys
        required={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        observer=Path(sys.modules[NativeTimedOracleSearchAdapter.__module__].__file__).resolve()
        required[str(observer)]=hashlib.sha256(observer.read_bytes()).hexdigest()
        require(required.items() <= self.files.items(),"A1 oracle reader/observer source inventory required")

    def _read_work(self, saved):
        return read_native_oracle_timed_selection(saved,oracle_receipt=self.oracle_receipt,
            public_request=self.public_request,expected_override_sha256=self.override_sha,
            expected_reconstruction_catalog_sha256=self.catalog_sha,configuration=self.configuration,
            runtime_configuration=self.runtime,runtime_sha256=self.runtime_sha,root_id=self.root_id,
            statistics_root_id=self.statistics_root_id,selection_seed=self.seed,consumed_binding=self.consumed,
            legal_action_mask=self.mask,decision_round=self.round,contract=self.contract,
            public_request_sha256=self.timed_public_request_sha)
