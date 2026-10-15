"""Owned shared raw control. Wiring only: never collection or qualification.

Capture the actual public request at the evaluator seam; one canonical champion
forward, deterministic masked argmax, fresh adapter, no search or retry. The
independent reader checks receipt arithmetic, source bytes and actual process
ownership, not evaluator truth or loaded-library closure.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import math
from pathlib import Path

from .search_over_raw import SearchConfiguration, digest, require
from .search_over_raw_adapters import PublicModelSearchAdapter, _raw_action
from .search_over_raw_supervision import _supervise_owned_job
from .search_over_raw_timed_run import NativeTimedRunValidator
from .search_over_raw_world_inputs import _binding, public_request_witness, valid_catalog_fingerprint

CAPTURE = dict(schema="pokezero.search-over-raw.raw-public-capture.v1",
    actual_adapter_request=True, runtime_qualified=False)


@dataclass(frozen=True)
class RawSelectionRunContract:
    emergency_watchdog_seconds: int = 1800

    def __post_init__(self):
        require(type(self.emergency_watchdog_seconds) is int and self.emergency_watchdog_seconds == 1800,
            "raw owner watchdog requires prospective amendment")

    def receipt(self):
        return dict(schema="pokezero.search-over-raw.raw-selection-contract.v1", **asdict(self),
            selector="deterministic_masked_argmax", model_forwards=1, search_target=None,
            watchdog_is_external=True, retries_permitted=False,
            runtime_qualified=False, collection_authorized=False)


def raw_runtime(configuration, checkpoint_sha256, set_source_hash):
    return dict(configuration=asdict(configuration), checkpoint_sha256=checkpoint_sha256,
        set_source_hash=set_source_hash, root_statistics="independent_per_root", incumbent=None,
        reference_factory=None, initial_dispatch_workers=None, raw_public_request_capture=deepcopy(CAPTURE))


def _runtime(configuration, runtime, sha):
    require(type(configuration) is SearchConfiguration and configuration.arm == "raw"
        and configuration.belief == "public" and configuration.leaf == "model" and configuration.workers == 1,
        "unchanged public single-forward raw control required")
    require(type(runtime) is dict and digest(runtime) == sha
        and digest(runtime) == digest(raw_runtime(configuration, runtime.get("checkpoint_sha256"),
            runtime.get("set_source_hash")))
        and type(runtime["checkpoint_sha256"]) is str and len(runtime["checkpoint_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in runtime["checkpoint_sha256"])
        and valid_catalog_fingerprint(runtime["set_source_hash"]), "raw runtime drift")


class RawSelectionAdapter(PublicModelSearchAdapter):
    def __init__(self, configuration, **kwargs):
        require(configuration.arm == "raw", "raw-only public capture adapter")
        super().__init__(configuration, **kwargs)
        self.runtime_configuration["raw_public_request_capture"] = deepcopy(CAPTURE)
        self.runtime_sha256 = digest(self.runtime_configuration)

    def _selection_evidence(self, evidence, request):
        return dict(evidence, consumed_binding=_binding(request),
            public_request_witness=public_request_witness(request))


def read_raw_selection(selected, *, configuration, runtime, root_id, statistics_root_id,
                       selection_seed, consumed_binding, legal_action_mask, public_request_sha256,
                       contract=RawSelectionRunContract()):
    _runtime(configuration, runtime, digest(runtime))
    mask = tuple(legal_action_mask)
    require(type(statistics_root_id) is str and statistics_root_id
        and root_id == statistics_root_id+":raw" and type(selection_seed) is int
        and 0 <= selection_seed < 2**64 and all(type(v) is bool for v in mask) and sum(mask) >= 2,
        "non-forced raw root/seed/mask required")
    require(type(selected) is dict and selected.get("status") == "SELECTED"
        and selected.get("root_id") == root_id and selected.get("statistics_root_id") == statistics_root_id
        and type(selected.get("selection_seed")) is int and selected["selection_seed"] == selection_seed
        and selected.get("configuration_sha256") == configuration.identity
        and selected.get("runtime_sha256") == digest(runtime)
        and digest(selected.get("observed_runtime_configuration")) == digest(runtime)
        and digest(selected.get("raw_selection_run_contract")) == digest(contract.receipt())
        and selected.get("independent_root_statistics") is True
        and selected.get("scientific_strength_evidence") is False
        and type(selected.get("elapsed_seconds")) in (int,float)
        and math.isfinite(selected["elapsed_seconds"]) and selected["elapsed_seconds"] >= 0
        and type(selected.get("nominal_search_seconds")) in (int,float)
        and selected["nominal_search_seconds"] == configuration.seconds
        and selected.get("exceeded_nominal_seconds") is False,
        "raw selected receipt binding/status/clock drift")
    evidence = selected.get("evidence")
    require(type(evidence) is dict and set(evidence) == {"selector", "model_forwards", "legal_actions",
        "priors", "signed_value", "consumed_binding", "public_request_witness"}
        and evidence["selector"] == "deterministic_masked_argmax"
        and type(evidence["model_forwards"]) is int and evidence["model_forwards"] == 1
        and type(evidence["legal_actions"]) is list
        and all(type(a) is int for a in evidence["legal_actions"])
        and evidence["legal_actions"] == [i for i,v in enumerate(mask) if v]
        and digest(evidence["consumed_binding"]) == digest(consumed_binding), "raw forward/legal/public binding drift")
    witness = evidence["public_request_witness"]
    require(type(witness) is dict and set(witness) == {"schema", "public_request_sha256", "consumed_binding",
        "full_supported_public_state_and_pending", "private_source_snapshot_included", "runtime_qualified"}
        and witness["schema"] == "pokezero.search-over-raw.public-request-consumption.v1"
        and witness["full_supported_public_state_and_pending"] is True
        and witness["private_source_snapshot_included"] is False and witness["runtime_qualified"] is False
        and digest(witness["consumed_binding"]) == digest(consumed_binding)
        and witness["public_request_sha256"] == public_request_sha256
        and type(public_request_sha256) is str and len(public_request_sha256) == 64
        and all(c in "0123456789abcdef" for c in public_request_sha256),
        "raw actual public request differs from external request")
    priors = evidence["priors"]
    require(type(priors) is list and len(priors) == len(evidence["legal_actions"])
        and all(type(p) in (int,float) and math.isfinite(p) and p >= 0 for p in priors)
        and abs(sum(priors)-1.) <= 1e-5
        and type(evidence["signed_value"]) in (int,float) and math.isfinite(evidence["signed_value"])
        and -1 <= evidence["signed_value"] <= 1
        and type(selected.get("action")) is int
        and selected["action"] == _raw_action(evidence["legal_actions"], priors), "raw masked argmax/value drift")
    return dict(schema="pokezero.search-over-raw.raw-selection-readout.v1", status="SELECTED",
        action=selected["action"], accepted_units=0, observed_model_forwards=1,
        selected_sha256=digest(selected), public_request_sha256=public_request_sha256,
        runtime_qualified=False, scoring_authorized=False, independent_evaluator_correctness=False)


@dataclass(frozen=True)
class RawSelectionJob:
    configuration: object
    checkpoint_contract: object
    showdown_root: str
    context: object
    root_id: str
    selection_seed: int
    statistics_root_id: str
    expected_runtime_configuration: dict
    expected_public_request_sha256: str
    pending_transition: object = None
    contract: RawSelectionRunContract = RawSelectionRunContract()

    def __post_init__(self):
        from .head_to_head import public_only_context
        from .paper_reference_runtime import PublicRootRequest
        require(type(self.contract) is RawSelectionRunContract, "distinct raw contract required")
        object.__setattr__(self, "expected_runtime_configuration", deepcopy(self.expected_runtime_configuration))
        _runtime(self.configuration, self.expected_runtime_configuration, digest(self.expected_runtime_configuration))
        require(self.expected_runtime_configuration["checkpoint_sha256"] == self.checkpoint_contract.checkpoint_sha256
            and self.expected_runtime_configuration["set_source_hash"] == self.checkpoint_contract.showdown_source_sha256
            and str(Path(self.checkpoint_contract.checkpoint_path).resolve()) == self.checkpoint_contract.checkpoint_path
            and str(Path(self.showdown_root).resolve()) == self.showdown_root, "raw external model/catalog/path drift")
        public = public_only_context(self.context)
        request = PublicRootRequest.capture(public.public_materialization_state, public.observation,
            pending_transition=self.pending_transition)
        require(public_request_witness(request)["public_request_sha256"] == self.expected_public_request_sha256
            and request.set_source_hash == self.checkpoint_contract.showdown_source_sha256
            and self.root_id == self.statistics_root_id+":raw" and type(self.selection_seed) is int
            and 0 <= self.selection_seed < 2**64 and sum(request.observation.legal_action_mask) >= 2,
            "raw archived public root/seed drift")
        object.__setattr__(self, "context", replace(public, observation=request.observation,
            requested_observations={public.player_id:request.observation}, public_materialization_state=request.state))
        object.__setattr__(self, "_request", request)
        object.__setattr__(self, "_consumed_binding", _binding(request))
        object.__setattr__(self, "_mask", tuple(request.observation.legal_action_mask))
        object.__setattr__(self, "_frozen", self._inputs())

    def _inputs(self):
        # This request is already sanitized: recapture would require the
        # deliberately stripped catalog. Hash the live transported fields, not
        # a second capture or a planned checksum standing in for actual input.
        request = replace(self._request, state=self.context.public_materialization_state,
            observation=self.context.observation, pending_transition=self.pending_transition)
        return digest(dict(configuration=asdict(self.configuration), runtime=self.expected_runtime_configuration,
            root=self.root_id, statistics_root=self.statistics_root_id, seed=self.selection_seed,
            public=public_request_witness(request), expected_public=self.expected_public_request_sha256,
            contract=self.contract.receipt(), showdown=self.showdown_root,
            checkpoint=dict(path=self.checkpoint_contract.checkpoint_path,
                sha256=self.checkpoint_contract.checkpoint_sha256, source=self.checkpoint_contract.showdown_source_sha256)))

    def read_selection(self, selected):
        require(self._inputs() == self._frozen, "raw job changed after creation")
        return read_raw_selection(selected, configuration=self.configuration, runtime=self.expected_runtime_configuration,
            root_id=self.root_id, statistics_root_id=self.statistics_root_id, selection_seed=self.selection_seed,
            consumed_binding=self._consumed_binding, legal_action_mask=self._mask,
            public_request_sha256=self.expected_public_request_sha256, contract=self.contract)

    def __call__(self, ownership, phase):
        adapter, primary_error = None, None
        phase("startup")
        try:
            from ..neural_policy import load_transformer_policy
            from ..randbat import load_gen3_randbat_source_cached
            from .paper_reference_showdown import ChampionEvaluator
            require(self._inputs() == self._frozen, "raw job changed before startup")
            checkpoint = Path(self.checkpoint_contract.checkpoint_path)
            require(hashlib.sha256(checkpoint.read_bytes()).hexdigest() == self.checkpoint_contract.checkpoint_sha256,
                "raw champion checkpoint drift")
            policy = load_transformer_policy(checkpoint, device="cpu", deterministic=True,
                exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False)
            source = load_gen3_randbat_source_cached(self.showdown_root)
            require(source.metadata.source_hash == self._consumed_binding["set_source_hash"], "raw owner catalog drift")
            state = self.context.public_materialization_state
            engine = state.belief_engine.clone(); engine.set_source = source
            context = replace(self.context, public_materialization_state=replace(state, belief_engine=engine))
            adapter = RawSelectionAdapter(self.configuration, checkpoint_contract=self.checkpoint_contract,
                showdown_root=self.showdown_root, evaluator=ChampionEvaluator(policy))
            require(digest(adapter.runtime_configuration) == digest(self.expected_runtime_configuration)
                and adapter.runtime_sha256 == digest(self.expected_runtime_configuration), "actual raw runtime drift")
            phase("selection")
            selected = adapter.select(context, root_id=self.root_id, selection_seed=self.selection_seed,
                statistics_root_id=self.statistics_root_id, pending_transition=self.pending_transition)
            selected["observed_runtime_configuration"] = deepcopy(adapter.runtime_configuration)
            selected["raw_selection_run_contract"] = self.contract.receipt()
            self.read_selection(selected)
            return selected
        except BaseException as error:
            primary_error = error
            raise
        finally:
            phase("cleanup")
            if adapter is not None:
                try:
                    adapter.close()
                except BaseException as error:
                    if primary_error is None: raise
                    primary_error.add_note("Raw adapter cleanup failed: "+type(error).__name__)


def supervise_raw_selection(job, *, directory, binding_sha256, verify_admission):
    require(type(job) is RawSelectionJob, "actual shared raw job required")
    return _supervise_owned_job(job, directory=directory, binding_sha256=binding_sha256,
        verify_admission=verify_admission, contract=job.contract, kind="raw-selection-run",
        work_phase="selection", validate_result=job.read_selection)


class RawSelectionRunValidator(NativeTimedRunValidator):
    """Shared lifecycle guards only; no native/search-work law is inherited."""
    _runtime_check = staticmethod(_runtime)
    _contract_class = RawSelectionRunContract
    _kind = "raw-selection-run"
    _complete_status = "RAW_SELECTION_RUN_COMPLETE"
    _readout_schema = "pokezero.search-over-raw.raw-owning-readout.v1"

    @staticmethod
    def _root_suffix(configuration):
        return "raw"

    def __init__(self, *, public_request_sha256, **kwargs):
        require(type(public_request_sha256) is str and len(public_request_sha256) == 64
            and all(c in "0123456789abcdef" for c in public_request_sha256), "external raw public request hash required")
        self.public_request_sha = public_request_sha256
        super().__init__(**kwargs)

    def _inputs(self):
        return digest(dict(lifecycle=super()._inputs(), public_request_sha256=self.public_request_sha))

    def _read_work(self, saved):
        return read_raw_selection(saved, configuration=self.configuration, runtime=self.runtime,
            root_id=self.root_id, statistics_root_id=self.statistics_root_id, selection_seed=self.seed,
            consumed_binding=self.consumed, legal_action_mask=self.mask,
            public_request_sha256=self.public_request_sha, contract=self.contract)
