"""Distinct native timed ownership; prefix work is not runtime qualification.

This module grants no source generation, collection, holdout or retry authority.
The job transports an already allocated public root. The independent reader
never unpickles jobs, signals historical PIDs, or loads a model/simulator.
Declared file bytes are not a loaded-binary dependency closure. Optional full
request observation binds the archived public transport, not private truth.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

from .search_over_raw import SearchConfiguration, digest, require
from .search_over_raw_fixed_work_run import fixed_work_reader_source_paths, _finite_nonnegative
from .search_over_raw_outcomes import _sha
from .search_over_raw_resources import (
    measurement_protocol, resource_source_hash, read_completed_process_cost, validate_memory_receipt,
)
from .search_over_raw_supervision import _supervise_owned_job
from .search_over_raw_world_inputs import (_binding, validate_timed_native_inputs,public_request_witness,
    validate_timed_public_request,TIMED_NATIVE_PUBLIC_CAPTURE,valid_catalog_fingerprint,
    native_atomic_runtime_scopes)


@dataclass(frozen=True)
class NativeTimedRunContract:
    iterations_per_world: int = 4096
    batch_size: int = 16
    worlds: int = 4
    emergency_watchdog_seconds: int = 1800

    def __post_init__(self):
        require(all(type(v) is int for v in asdict(self).values())
            and tuple(asdict(self).values()) == (4096, 16, 4, 1800),
            "native timed allocation requires a prospective amendment")

    def receipt(self):
        return dict(schema="pokezero.search-over-raw.native-timed-contract.v1", **asdict(self),
            decision_ceiling_seconds=[1, 3, 10], fixed_work_target=None,
            accepted_unit="complete_native_iteration_with_committed_backup",
            watchdog_is_external=True, zero_work_status="UNQUALIFIED_AT_CEILING",
            retries_permitted=False, runtime_qualified=False, collection_authorized=False)


def _configuration(configuration):
    require(type(configuration) is SearchConfiguration and configuration.arm == "incumbent"
        and configuration.belief == "public" and configuration.leaf in {"model", "hp_fraction"}
        and configuration.workers == 1 and configuration.seconds in (1., 3., 10.),
        "native timed public model/HP and exact 1/3/10 ceiling required")


def _runtime(configuration, runtime, runtime_sha):
    _runtime_law(configuration, runtime, runtime_sha, configuration_check=_configuration,
        compact_native_atomic_work=_compact_runtime(runtime))
    require(not any(k in runtime for k in ("team_oracle", "native_team_oracle_observation",
        "native_oracle_construction")), "public timed runtime cannot carry oracle observations")


def _compact_runtime(runtime):
    """Decode only the frozen prospective recipe, never selector evidence."""
    require(type(runtime) is dict and ("compact_native_atomic_work" not in runtime
        or runtime["compact_native_atomic_work"] is True), "native timed compact recipe marker drift")
    return "compact_native_atomic_work" in runtime


def _runtime_law(configuration, runtime, runtime_sha, *, configuration_check,
                 compact_native_atomic_work=False):
    configuration_check(configuration)
    require(type(compact_native_atomic_work) is bool
        and _compact_runtime(runtime) is compact_native_atomic_work,
        "native timed compact recipe scope drift")
    atomic_scope, timed_scope = native_atomic_runtime_scopes(compact=compact_native_atomic_work)
    require(type(runtime) is dict and _sha(runtime_sha) and digest(runtime) == runtime_sha
        and digest(runtime.get("configuration")) == digest(asdict(configuration))
        and digest(runtime.get("incumbent")) == digest(dict(depth=6, sims=4096, batch=16,
            worlds=4, native_batch_guard_ms=64))
        and digest(runtime.get("timed_native_atomic_work")) == digest(timed_scope)
        and "fixed_work_diagnostic" not in runtime and "native_searched_ancestry" not in runtime,
        "prospective distinct native timed runtime/allocation drift")
    require(digest(runtime.get("consumed_world_inputs")) == digest(dict(
        schema="pokezero.search-over-raw.consumed-world-inputs.v1",initial_inputs_only=True,
        snapshot_fields_excluded=[],ownership_metadata_excluded=[],instrumentation_can_change_deadlines=True,
        qualifies_uninstrumented_runtime=False))
        and digest(runtime.get("native_atomic_work")) == digest(atomic_scope)
        and runtime.get("root_statistics") == "independent_per_root",
        "native timed observation scope drift")
    if configuration.leaf == "hp_fraction":
        require(digest(runtime.get("incumbent_leaf")) == digest(dict(leaf="hp_fraction",
            tree="unchanged_encoded_model_tree",priors="unchanged_champion",model_forwards="retained",
            value_frame="side_one_absolute")),"native timed HP runtime leaf drift")
    else:
        require("incumbent_leaf" not in runtime,"native timed model runtime contains an alternative leaf")
    if "full_public_request_observation" in runtime:
        require(digest(runtime["full_public_request_observation"]) == digest(TIMED_NATIVE_PUBLIC_CAPTURE),
            "distinct timed native full-public observation scope drift")


def read_native_timed_selection(selected, *, configuration, runtime_configuration, runtime_sha256,
                                root_id, statistics_root_id, selection_seed, consumed_binding,
                                legal_action_mask, decision_round, contract, public_request_sha256=None):
    """Recompute prefix and selector law; ownership is a separate read below."""
    return _read_native_timed_selection(selected, configuration=configuration,
        runtime_configuration=runtime_configuration, runtime_sha256=runtime_sha256,
        root_id=root_id, statistics_root_id=statistics_root_id, selection_seed=selection_seed,
        consumed_binding=consumed_binding, legal_action_mask=legal_action_mask,
        decision_round=decision_round, contract=contract, public_request_sha256=public_request_sha256,
        runtime_check=_runtime, contract_class=NativeTimedRunContract)


def _read_native_timed_selection(selected, *, configuration, runtime_configuration, runtime_sha256,
                                root_id, statistics_root_id, selection_seed, consumed_binding,
                                legal_action_mask, decision_round, contract, public_request_sha256,
                                runtime_check, contract_class):
    require(type(contract) is contract_class, "distinct native timed contract required")
    runtime_check(configuration, runtime_configuration, runtime_sha256)
    require(type(statistics_root_id) is str and statistics_root_id
        and root_id == statistics_root_id+":"+configuration.identity
        and type(selection_seed) is int and 0 <= selection_seed < 2**64
        and type(legal_action_mask) in (list, tuple) and all(type(v) is bool for v in legal_action_mask)
        and sum(legal_action_mask) >= 2 and (decision_round is None or type(decision_round) is int),
        "prospective non-forced native timed root and legal surface required")
    expected = dict(root_id=root_id, statistics_root_id=statistics_root_id,
        configuration_sha256=configuration.identity, selection_seed=selection_seed,
        runtime_sha256=runtime_sha256)
    require(type(selected) is dict and digest({k:selected.get(k) for k in expected}) == digest(expected)
        and digest(selected.get("native_timed_run_contract")) == digest(contract.receipt())
        and digest(selected.get("observed_runtime_configuration")) == digest(runtime_configuration)
        and selected.get("independent_root_statistics") is True
        and selected.get("scientific_strength_evidence") is False
        and _finite_nonnegative(selected.get("elapsed_seconds"))
        and type(selected.get("nominal_search_seconds")) in (int, float)
        and selected["nominal_search_seconds"] == configuration.seconds
        and selected.get("exceeded_nominal_seconds") is (selected["elapsed_seconds"] > configuration.seconds),
        "exact selected native timed root/runtime/clock drift")
    evidence = selected.get("evidence")
    readout = validate_timed_native_inputs(evidence, binding=consumed_binding,
        iterations_per_world=contract.iterations_per_world, batch_size=contract.batch_size,
        expected_worlds=contract.worlds, nominal_seconds=configuration.seconds,
        adapter_elapsed_seconds=selected["elapsed_seconds"],
        compact_native_atomic_work=_compact_runtime(runtime_configuration))
    require((public_request_sha256 is not None) == ("full_public_request_observation" in runtime_configuration),
        "timed full-public runtime requires externally bound request hash")
    public_readout = None if public_request_sha256 is None else validate_timed_public_request(evidence,
        binding=consumed_binding,expected_sha256=public_request_sha256)
    calls = evidence["consumed_world_inputs"]["calls"]
    expected_hp = []
    model_evaluation_rows = 0
    for call in calls:
        report = call["native_report"]
        require(type(report.get("model_evals")) is int and report["model_evals"] >= 0,
            "actual native timed model-evaluation row count malformed")
        # One report per actual call, never multiplied by collapsed worlds.
        # Cancelled/zero-commit calls may still report real model row work.
        model_evaluation_rows += report["model_evals"]
        if configuration.leaf == "hp_fraction":
            from ..engine_search import validate_native_hp_leaf_witness
            validate_native_hp_leaf_witness(report)
            expected_hp.append(dict(world_seed=call["native_call_parameters"]["chance_seed"],
                belief_multiplicity=call["multiplicity"],completed_iterations=report["iterations"],
                model_evals=report["model_evals"],hp_leaf_rows_priced=report["hp_leaf_rows_priced"]))
        else:
            require("model_leaf_override" not in report,
                "model timed prefix contains an unrequested leaf override")
    if readout["completed_iterations"] == 0:
        diagnostic, original = selected.get("strict_engine_diagnostic"), selected.get("original_failure_receipt")
        require(selected.get("status") == "UNQUALIFIED_AT_CEILING" and "action" in selected and selected["action"] is None
            and digest(selected.get("timed_input_readout")) == digest(readout)
            and type(diagnostic) is dict and diagnostic.get("schema") == "engine-search-fallback-refusal-v1"
            and diagnostic.get("diagnostic_only_not_policy_input") is True
            and diagnostic.get("reason") == "model_time_budget_no_completed_worlds"
            and diagnostic.get("battle_id") == "search-over-raw-root:"+statistics_root_id
            and diagnostic.get("seat") == consumed_binding["subject"]
            and digest(diagnostic.get("round")) == digest(decision_round)
            and type(diagnostic.get("world_failures")) is dict and not diagnostic["world_failures"]
            and digest(diagnostic.get("engine_mcts")) == digest(evidence["engine_mcts"])
            and evidence["engine_mcts"].get("fallback") == diagnostic["reason"]
            and type(original) is dict and original.get("root_id") == root_id
            and original.get("error_type") == "EngineSearchFallbackError"
            and original.get("status") == "UNCERTAIN_REFUSED" and original.get("retry_authorized") is False
            and digest(original.get("engine_fallback_diagnostic")) == digest(diagnostic)
            and selected.get("attempt_consumed") is True and selected.get("retry_authorized") is False
            and selected.get("continuation_authorized") is False
            and selected.get("runtime_qualified") is False
            and selected.get("owning_process_validated") is False
            and selected.get("scientific_collection_authorized") is False,
            "zero work lacks exact bound normal strict-deadline provenance")
    else:
        require(selected.get("status") == "SELECTED" and type(selected.get("action")) is int
            and 0 <= selected["action"] < len(legal_action_mask) and legal_action_mask[selected["action"]]
            and evidence.get("root_action") == f"action:{selected['action']}"
            and "strict_engine_diagnostic" not in selected and "original_failure_receipt" not in selected,
            "productive native timed selector/status/action drift")
        require(type(evidence.get("model_evals")) is int and evidence["model_evals"] == model_evaluation_rows,
            "productive model-evaluation rows differ from actual returned calls")
        from .manifest import SearchConfig
        from .policy_opponent_profile import validate_selection
        validate_selection(evidence, arm="incumbent_mcts", mode="matched_deadline",
            config=SearchConfig(depth=6, sims=4096, batch=16, worlds=4, inference_mode="local"),
            mask=legal_action_mask, opponent_seed=selection_seed, deadline_ms=round(configuration.seconds*1000),
            native_batch_guard_ms=64,
            **({"model_leaf_override":"hp_fraction"} if configuration.leaf == "hp_fraction" else {}))
        expected_joint = [dict(world_seed=call["native_call_parameters"]["chance_seed"],
            belief_multiplicity=call["multiplicity"], witness=call["native_report"]["joint_action_witness"])
            for call in calls]
        require(digest(evidence["engine_mcts"]["joint_actions"]["native_invocations"]) == digest(expected_joint),
            "selector joint witness differs from actual native returned prefix")
        if configuration.leaf == "hp_fraction":
            require(digest(evidence["engine_mcts"]["model_leaf_override"]["native_invocations"]) == digest(expected_hp),
                "HP selector witness differs from actual native reports")
    return dict(schema="pokezero.search-over-raw.native-timed-observed-work.v1",
        input_readout=readout, accepted_units=readout["completed_iterations"],
        accepted_unit="complete_native_iteration_with_committed_backup",
        observed_native_model_evaluation_rows=model_evaluation_rows,
        model_evaluation_scope="backend-reported evaluated rows; not independent inference-operation proof",
        selected_status=selected["status"], evidence_sha256=digest(evidence),
        public_request_consumption=public_readout,
        full_public_request_consumption_validated=public_readout is not None,
        selector_structure_validated=readout["completed_iterations"] > 0,
        independent_evaluator_correctness=False, runtime_qualified=False, scoring_authorized=False,
        scientific_collection_authorized=False)


@dataclass(frozen=True)
class NativeTimedAdapterJob:
    _configuration_check = staticmethod(_configuration)
    _runtime_check = staticmethod(_runtime)
    _contract_class = NativeTimedRunContract
    configuration: object
    checkpoint_contract: object
    showdown_root: str
    context: object
    root_id: str
    selection_seed: int
    statistics_root_id: str
    expected_runtime_configuration: dict
    pending_transition: object = None
    contract: NativeTimedRunContract = NativeTimedRunContract()
    expected_public_request_sha256: str | None = None

    def __post_init__(self):
        from .head_to_head import public_only_context
        from .paper_reference_runtime import PublicRootRequest
        self._configuration_check(self.configuration)
        require(type(self.contract) is self._contract_class, "distinct native timed job contract required")
        public = public_only_context(self.context)
        request = PublicRootRequest.capture(public.public_materialization_state, public.observation,
            pending_transition=self.pending_transition)
        public = replace(public, observation=request.observation,
            requested_observations={public.player_id:request.observation}, public_materialization_state=request.state)
        object.__setattr__(self, "context", public)
        object.__setattr__(self, "expected_runtime_configuration", deepcopy(self.expected_runtime_configuration))
        self._runtime_check(self.configuration, self.expected_runtime_configuration, self.expected_runtime_sha256)
        require(request.set_source_hash == self.checkpoint_contract.showdown_source_sha256
            and self.expected_runtime_configuration.get("set_source_hash") == request.set_source_hash
            and self.expected_runtime_configuration.get("checkpoint_sha256") == self.checkpoint_contract.checkpoint_sha256,
            "timed public root/model catalog differs from pinned runtime")
        object.__setattr__(self, "_consumed_binding", _binding(request))
        object.__setattr__(self, "_public_request", request)
        object.__setattr__(self, "_legal_action_mask", tuple(request.observation.legal_action_mask))
        require(type(self.statistics_root_id) is str and self.statistics_root_id
            and self.root_id == self.statistics_root_id+":"+self.configuration.identity
            and type(self.selection_seed) is int and 0 <= self.selection_seed < 2**64
            and all(type(v) is bool for v in self._legal_action_mask) and sum(self._legal_action_mask) >= 2,
            "exact non-forced timed root/selection seed required")
        require((self.expected_public_request_sha256 is not None) ==
            ("full_public_request_observation" in self.expected_runtime_configuration),
            "native job full-public runtime/hash scope mismatch")
        if self.expected_public_request_sha256 is not None:
            require(public_request_witness(request)["public_request_sha256"] == self.expected_public_request_sha256,
                "native timed job differs from archived full public request")
        object.__setattr__(self,"_frozen_inputs",self._inputs())

    def _inputs(self):
        # Reuse the captured sanitized request; recapturing catalog-stripped
        # state would lose its public source hash. Hash live transported fields.
        witness = None if self.expected_public_request_sha256 is None else public_request_witness(replace(
            self._public_request,state=self.context.public_materialization_state,
            observation=self.context.observation,pending_transition=self.pending_transition))
        return digest(dict(configuration=asdict(self.configuration),runtime=self.expected_runtime_configuration,
            root=self.root_id,statistics_root=self.statistics_root_id,seed=self.selection_seed,
            consumed=self._consumed_binding,mask=self._legal_action_mask,contract=self.contract.receipt(),
            showdown=self.showdown_root,checkpoint_sha=self.checkpoint_contract.checkpoint_sha256,
            checkpoint_path=getattr(self.checkpoint_contract,"checkpoint_path",None),
            source_sha=self.checkpoint_contract.showdown_source_sha256,
            player=self.context.player_id,decision_round=self.context.decision_round_index,
            public_request_sha=self.expected_public_request_sha256,live_public_witness=witness))

    @property
    def expected_runtime_sha256(self):
        return digest(self.expected_runtime_configuration)

    def read_selection(self, selected):
        require(self._inputs() == self._frozen_inputs,"native timed job public inputs changed")
        return read_native_timed_selection(selected, configuration=self.configuration,
            runtime_configuration=self.expected_runtime_configuration, runtime_sha256=self.expected_runtime_sha256,
            root_id=self.root_id, statistics_root_id=self.statistics_root_id, selection_seed=self.selection_seed,
            consumed_binding=self._consumed_binding, legal_action_mask=self._legal_action_mask,
            decision_round=self.context.decision_round_index, contract=self.contract,
            public_request_sha256=self.expected_public_request_sha256)

    def _make_adapter(self):
        from .search_over_raw_world_inputs import WorldInputSearchAdapter
        return WorldInputSearchAdapter(self.configuration, timed_native_atomic_work=True,
            **({"compact_native_atomic_work":True} if _compact_runtime(self.expected_runtime_configuration) else {}),
            **({"expected_public_request_sha256":self.expected_public_request_sha256}
                if self.expected_public_request_sha256 is not None else {}),
            checkpoint_contract=self.checkpoint_contract, showdown_root=self.showdown_root)

    def _check_source(self, source):
        require(source.metadata.source_hash == self._consumed_binding["set_source_hash"], "timed worker catalog drift")

    def _check_adapter_scope(self, adapter):
        pass

    def __call__(self, ownership, phase):
        adapter = None
        primary_error = None
        phase("startup")
        try:
            require(self._inputs() == self._frozen_inputs,"native timed job public inputs changed before startup")
            from ..randbat import load_gen3_randbat_source_cached
            source = load_gen3_randbat_source_cached(self.showdown_root)
            self._check_source(source)
            state = self.context.public_materialization_state
            engine = state.belief_engine.clone(); engine.set_source = source
            context = replace(self.context, public_materialization_state=replace(state, belief_engine=engine))
            adapter = self._make_adapter()
            self._check_adapter_scope(adapter)
            require(digest(adapter.runtime_configuration) == self.expected_runtime_sha256
                and adapter.runtime_sha256 == self.expected_runtime_sha256, "actual timed runtime differs from review")
            cfg = adapter._native._config
            require(all(type(getattr(cfg,k)) is int and getattr(cfg,k) == v for k,v in dict(
                search_depth=6, search_sims=4096, search_batch=16, worlds=4, model_world_workers=1,
                model_decision_time_ms=round(self.configuration.seconds*1000),model_native_batch_guard_ms=64).items())
                and cfg.early_stop is False and cfg.strict_fallbacks is True and cfg.model_priors is True
                and cfg.record_joint_actions is True and cfg.policy_opponent is False and cfg.leaf_eval == "model"
                and cfg.model_leaf_override == ("hp_fraction" if self.configuration.leaf == "hp_fraction" else None)
                and getattr(adapter._native,"_ladder_sims_override",None) is None,
                "actual native timed allocation differs from prospective law")
            phase("selection")
            selected = adapter.select(context, root_id=self.root_id, selection_seed=self.selection_seed,
                statistics_root_id=self.statistics_root_id, pending_transition=self.pending_transition)
            self._check_adapter_scope(adapter)
            require(digest(adapter.runtime_configuration) == self.expected_runtime_sha256
                and adapter.runtime_sha256 == self.expected_runtime_sha256, "timed runtime changed during selection")
            selected["native_timed_run_contract"] = self.contract.receipt()
            selected["observed_runtime_configuration"] = deepcopy(adapter.runtime_configuration)
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
                    if primary_error is None:
                        self._check_adapter_scope(adapter)
                        require(self._inputs() == self._frozen_inputs,"native timed job public inputs changed during cleanup")
                except BaseException as cleanup:
                    if primary_error is None:
                        raise
                    primary_error.add_note("Native timed adapter cleanup failed: "+type(cleanup).__name__)


def supervise_native_timed(job, *, directory, binding_sha256, verify_admission,
                           _synthetic_watchdog_seconds=None):
    require(type(job) is NativeTimedAdapterJob, "native timed owning job required")
    return _supervise_owned_job(job, directory=directory, binding_sha256=binding_sha256,
        verify_admission=verify_admission, contract=job.contract, kind="native-timed-run",
        work_phase="selection", validate_result=job.read_selection,
        _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)


class NativeTimedRunValidator:
    """Read one exact completed owned timed job; never use fixed-work quotas."""
    _runtime_check = staticmethod(_runtime)
    _contract_class = NativeTimedRunContract
    _kind = "native-timed-run"
    _complete_status = "NATIVE_TIMED_RUN_COMPLETE"
    _readout_schema = "pokezero.search-over-raw.native-timed-owning-readout.v1"
    _native_public_binding = True

    @staticmethod
    def _root_suffix(configuration):
        return configuration.identity

    def _read_work(self, saved):
        return read_native_timed_selection(saved, configuration=self.configuration,
            runtime_configuration=self.runtime,runtime_sha256=self.runtime_sha,root_id=self.root_id,
            statistics_root_id=self.statistics_root_id,selection_seed=self.seed,consumed_binding=self.consumed,
            legal_action_mask=self.mask,decision_round=self.round,contract=self.contract,
            public_request_sha256=self.timed_public_request_sha)

    def _read_owned(self, saved, terminal, pid, read, hashes):
        owned = self.directory/"owned-groups"
        inventory = sorted(str(p.relative_to(owned)) for p in owned.rglob("*") if p.is_file())
        require(owned.is_dir() and not inventory and terminal["owned_groups"] == [pid],
            "native timed job unexpectedly contains descendants or active/historical generations")
        return []

    def __init__(self, *, directory, job_sha256, binding_sha256, configuration,
                 root_id, statistics_root_id, selection_seed, runtime_configuration,
                 runtime_sha256, consumed_binding, legal_action_mask, decision_round,
                 file_hashes, checkpoint_path, checkpoint_sha256, showdown_root,
                 allow_synthetic=False, contract=None, public_request_sha256=None):
        require(public_request_sha256 is None or self._native_public_binding and _sha(public_request_sha256),
            "native timed full-public request binding required")
        self.timed_public_request_sha = public_request_sha256
        if self._native_public_binding:
            require((public_request_sha256 is not None) == ("full_public_request_observation" in runtime_configuration),
                "external native timed runtime/hash scope mismatch")
        self._runtime_check(configuration, runtime_configuration, runtime_sha256)
        require(all(_sha(v) for v in (job_sha256,binding_sha256,checkpoint_sha256))
            and type(allow_synthetic) is bool and type(consumed_binding) is dict
            and set(consumed_binding) == {"root_binding","subject","set_source_hash","information_key"}
            and _sha(consumed_binding["root_binding"])
            and valid_catalog_fingerprint(consumed_binding["set_source_hash"])
            and consumed_binding["subject"] in {"p1","p2"}
            and type(consumed_binding["information_key"]) is str and consumed_binding["information_key"]
            and type(statistics_root_id) is str and statistics_root_id
            and root_id == statistics_root_id+":"+self._root_suffix(configuration)
            and type(selection_seed) is int and 0 <= selection_seed < 2**64
            and type(legal_action_mask) in (list,tuple) and all(type(v) is bool for v in legal_action_mask)
            and sum(legal_action_mask) >= 2 and (decision_round is None or type(decision_round) is int),
            "external timed job/root/seed/legal surface bindings required")
        require(type(checkpoint_path) is str and str(Path(checkpoint_path).resolve()) == checkpoint_path
            and type(showdown_root) is str and str(Path(showdown_root).resolve()) == showdown_root,
            "external native timed model/simulator paths required")
        require(runtime_configuration.get("checkpoint_sha256") == checkpoint_sha256
            and runtime_configuration.get("set_source_hash") == consumed_binding["set_source_hash"],
            "timed external runtime model/catalog binding drift")
        self.contract = self._contract_class() if contract is None else contract
        require(type(self.contract) is self._contract_class, "distinct timed owning contract required")
        self.directory = Path(directory).resolve()
        self.job_sha, self.binding = job_sha256,binding_sha256
        self.configuration, self.runtime_sha = configuration,runtime_sha256
        self.runtime = deepcopy(runtime_configuration)
        self.root_id, self.statistics_root_id, self.seed = root_id,statistics_root_id,selection_seed
        self.consumed, self.mask, self.round = deepcopy(consumed_binding),tuple(legal_action_mask),decision_round
        self.files = deepcopy(file_hashes)
        self.checkpoint_path, self.checkpoint_sha, self.showdown_root = checkpoint_path,checkpoint_sha256,showdown_root
        self.allow_synthetic = allow_synthetic
        self.frozen = self._inputs()
        self.last_validation = None
        self._files()

    def _inputs(self):
        return digest(dict(directory=str(self.directory),job=self.job_sha,binding=self.binding,
            configuration=asdict(self.configuration),runtime=self.runtime,runtime_sha=self.runtime_sha,
            root=self.root_id,statistics_root=self.statistics_root_id,seed=self.seed,consumed=self.consumed,
            mask=self.mask,round=self.round,files=self.files,checkpoint_path=self.checkpoint_path,
            checkpoint_sha=self.checkpoint_sha,showdown=self.showdown_root,
            allow_synthetic=self.allow_synthetic,contract=self.contract.receipt(),
            timed_public_request_sha256=self.timed_public_request_sha))

    def _files(self):
        import sys
        required = fixed_work_reader_source_paths()  # Path inventory only, not its fixed-work law.
        dist = Path(self.showdown_root)/"dist"
        require((dist/"sim/index.js").is_file() and (dist/"sim/state.js").is_file(),
            "built native timed simulator binding required")
        required.update(str(p.resolve()) for p in dist.rglob("*") if p.is_file())
        executable = str(Path(sys.executable).resolve())
        required.update((self.checkpoint_path,executable))
        # Hash every declared canonical file once in this invocation. Nothing
        # is cached across calls: pre/post-callback drift guards remain fresh.
        # This is not an atomic filesystem snapshot or dependency qualification.
        require(type(self.files) is dict and required <= set(self.files)
            and self.files.get(self.checkpoint_path) == self.checkpoint_sha
            and all(type(p) is str and str(Path(p).resolve()) == p and _sha(v)
                and hashlib.sha256(Path(p).read_bytes()).hexdigest() == v for p,v in self.files.items()),
            "declared native timed file byte drift")

    def __call__(self, selected):
        self.last_validation = None
        require(self._inputs() == self.frozen,"native timed reader inputs changed")
        self._files()
        saved = deepcopy(selected); selected_sha = digest(saved); hashes = {}
        def read(name):
            payload = (self.directory/name).read_bytes()
            hashes[name] = hashlib.sha256(payload).hexdigest()
            return json.loads(payload)
        claim,launch,worker,terminal = (read(name) for name in
            ("claim.json","launch.json","worker-result.json","terminal.json"))
        require(hashlib.sha256((self.directory/"job.pickle").read_bytes()).hexdigest() == self.job_sha
            == claim["job_sha256"] and claim["binding_sha256"] == self.binding
            and digest(claim["contract"]) == digest(self.contract.receipt())
            and claim["retry_authorized"] is False and type(claim["synthetic_process_fixture"]) is bool
            and digest(claim["resource_measurement_protocol"]) == digest(measurement_protocol())
            and claim["resource_measurement_source_sha256"] == resource_source_hash(),
            "external native timed job/claim/protocol drift")
        pid = launch["pid"]
        require(type(pid) is int and pid > 0 and type(launch["group"]) is int and launch["group"] == pid
            and type(launch["supervisor_pid"]) is int and launch["supervisor_pid"] > 0
            and launch["supervisor_pid"] != pid and type(launch["birth_identity"]) is str and launch["birth_identity"]
            and launch["binding_sha256"] == self.binding
            and all(type(row["worker_pid"]) is int and row["worker_pid"] == pid
                and row["binding_sha256"] == self.binding for row in (worker,terminal)),
            "actual native timed launch/terminal ownership drift")
        require(terminal["schema"] == f"pokezero.search-over-raw.{self._kind}-supervision.v1"
            and terminal["status"] == self._complete_status
            and all(type(v) is int and v == 0 for v in (
                terminal["supervisor_exit_code"],terminal["worker_exit_code"],worker["exit_code"]))
            and terminal["watchdog_reached"] is False and terminal["error_type"] is None
            and terminal["cleanup_error_type"] is None and worker["error_type"] is None
            and terminal["deployment_eligible"] is False and terminal["runtime_qualified"] is False
            and terminal["retries_permitted"] is False and terminal["holdout_authorized"] is False,
            "normal externally reaped native timed completion required; failure is not zero work")
        watchdog = terminal["watchdog_seconds"]
        require(_finite_nonnegative(watchdog) and watchdog > 0
            and (watchdog == self.contract.emergency_watchdog_seconds or claim["synthetic_process_fixture"]
                and watchdog < self.contract.emergency_watchdog_seconds), "native timed external watchdog drift")
        costs = read_completed_process_cost(terminal,worker,work_phase="selection")
        require(_finite_nonnegative(worker["worker_wall_seconds"])
            and worker["worker_wall_seconds"] <= terminal["elapsed_wall_seconds"]
            and digest(saved) == digest(worker["result"]) == digest(terminal["result"]),
            "exact native timed returned receipt/worker clock drift")
        owned = self.directory/"owned-groups"
        def owned_inventory():
            return sorted(str(p.relative_to(owned)) for p in owned.rglob("*") if p.is_file())
        inventory = owned_inventory()
        generation_pids = self._read_owned(saved, terminal, pid, read, hashes)
        if terminal["memory_measurement"] is not None:
            require(terminal["resource_measurement_error_type"] is None,"conflicting native timed RSS failure")
            read("memory-trace.json")
            validate_memory_receipt(self.directory,binding=self.binding,main_launch=launch,
                expected_summary=terminal["memory_measurement"])
        else:
            require(type(terminal["resource_measurement_error_type"]) is str
                and terminal["resource_measurement_error_type"],"missing native timed RSS evidence is not success")
        accounting = self._read_work(saved)
        require(digest(accounting) == digest(terminal["observed_work_accounting"])
            and saved["elapsed_seconds"] <= costs["phase_costs"]["selection_wall_seconds"]+.001,
            "independent native timed work/owned selection clock reconciliation drift")
        self._files()
        require(self._inputs() == self.frozen and digest(selected) == selected_sha
            and all(hashlib.sha256((self.directory/name).read_bytes()).hexdigest() == value for name,value in hashes.items())
            and hashlib.sha256((self.directory/"job.pickle").read_bytes()).hexdigest() == self.job_sha
            and owned.is_dir() and owned_inventory() == inventory, "timed evidence changed during validation")
        self.last_validation = dict(schema=self._readout_schema,
            selected_sha256=selected_sha,receipt_file_sha256=hashes,actual_work=accounting,costs=costs,
            main_launch=deepcopy(launch),process_receipts_validated=True,owned_generation_pids=generation_pids,
            declared_file_bytes_validated=True,loaded_binary_dependency_closure_validated=False,
            full_public_request_correspondence_validated=False,source_freshness_validated=False,
            independent_evaluator_correctness=False,on_time_deployability_established=False,
            resource_cost_complete=False,runtime_qualified=False,scientific_collection_authorized=False,
            synthetic_process_fixture=claim["synthetic_process_fixture"])
        return selected_sha if not claim["synthetic_process_fixture"] or self.allow_synthetic else None
