"""Reference timed owner and independent normally-closed generation reader.

Non-collecting integration only. No quota, runtime qualification, loaded-library
closure, independent evaluator truth, fresh-source or holdout authority follows.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from .search_over_raw import SearchConfiguration, digest, require
from .search_over_raw_timed_run import NativeTimedRunValidator
from .search_over_raw_timed_reference import read_reference_timed_selection, SCHEMA
from .search_over_raw_supervision import _supervise_owned_job
from .search_over_raw_world_inputs import _binding, public_request_witness, TIMED_REFERENCE_PUBLIC_CAPTURE


@dataclass(frozen=True)
class ReferenceTimedRunContract:
    workers: int = 20
    batch_size: int = 10
    initial_dispatch_workers: int = 6
    emergency_watchdog_seconds: int = 1800

    def __post_init__(self):
        require(all(type(v) is int for v in asdict(self).values())
            and tuple(asdict(self).values()) == (20, 10, 6, 1800),
            "reference timed allocation requires a prospective amendment")

    def receipt(self):
        return dict(schema="pokezero.search-over-raw.reference-timed-contract.v1", **asdict(self),
            decision_ceiling_seconds=[1, 3, 10], fixed_work_target=None,
            accepted_unit="complete_reference_trajectory_with_own_backup_and_cleanup",
            watchdog_is_external=True, zero_work_status="UNQUALIFIED_AT_CEILING",
            retries_permitted=False, runtime_qualified=False, collection_authorized=False)


def _runtime(configuration, runtime, runtime_sha):
    _runtime_law(configuration,runtime,runtime_sha,configuration_check=_public_configuration)
    require(not any(k in runtime for k in ("team_oracle","reference_team_oracle_observation")),
        "public reference timed owner cannot carry oracle truth")


def _public_configuration(configuration):
    from .search_over_raw_timed_reference import _public_configuration as check
    check(configuration)


def _runtime_law(configuration, runtime, runtime_sha, *, configuration_check):
    from .paper_reference_runtime import ShowdownWorkerFactory
    configuration_check(configuration)
    require(type(runtime) is dict and digest(runtime) == runtime_sha
        and digest(runtime.get("configuration")) == digest(asdict(configuration))
        and runtime.get("incumbent") is None and runtime.get("root_statistics") == "independent_per_root"
        and type(runtime.get("initial_dispatch_workers")) is int and runtime["initial_dispatch_workers"] == 6
        and digest(runtime.get("timed_reference_atomic_work")) == digest(dict(schema=SCHEMA,
            zero_work_not_scored=True, fresh_single_attempt=True,
            on_time_deployability_established=False, runtime_qualified=False))
        and digest(runtime.get("owned_process_receipts")) == digest(dict(enabled=True,
            publishes_before_detachment=True, qualifies_uninstrumented_runtime=False))
        and not any(k in runtime for k in ("fixed_work_diagnostic", "visited_value_diagnostics",
            "callback_diagnostics", "native_searched_ancestry")),
        "distinct reference timed owner runtime drift")
    require("full_public_request_observation" not in runtime or
        digest(runtime["full_public_request_observation"]) == digest(TIMED_REFERENCE_PUBLIC_CAPTURE),
        "reference timed full public observation scope drift")
    factory = runtime.get("reference_factory")
    require(type(factory) is dict, "reviewed reference factory required")
    rebuilt = ShowdownWorkerFactory(**factory)
    require(digest(asdict(rebuilt)) == digest(factory)
        and rebuilt.checkpoint_sha256 == runtime.get("checkpoint_sha256")
        and rebuilt.set_source_hash == runtime.get("set_source_hash")
        and str(Path(rebuilt.checkpoint).resolve()) == rebuilt.checkpoint
        and str(Path(rebuilt.showdown_root).resolve()) == rebuilt.showdown_root,
        "reference owner factory/model/catalog/path drift")


@dataclass(frozen=True)
class ReferenceTimedAdapterJob:
    _runtime_check=staticmethod(_runtime)
    _contract_class=ReferenceTimedRunContract
    configuration: object
    checkpoint_contract: object
    showdown_root: str
    reference_factory: object
    context: object
    root_id: str
    selection_seed: int
    statistics_root_id: str
    expected_runtime_configuration: dict
    pending_transition: object = None
    contract: ReferenceTimedRunContract = ReferenceTimedRunContract()
    expected_public_request_sha256: str | None = None

    def __post_init__(self):
        from .head_to_head import public_only_context
        from .paper_reference_runtime import PublicRootRequest, ShowdownWorkerFactory
        require(type(self.reference_factory) is ShowdownWorkerFactory
            and type(self.contract) is self._contract_class, "distinct reference owner job required")
        object.__setattr__(self, "expected_runtime_configuration", deepcopy(self.expected_runtime_configuration))
        self._runtime_check(self.configuration, self.expected_runtime_configuration, self.expected_runtime_sha256)
        factory = self.reference_factory
        require(digest(asdict(factory)) == digest(self.expected_runtime_configuration["reference_factory"])
            and (factory.checkpoint, factory.checkpoint_sha256, factory.showdown_root, factory.set_source_hash)
            == (self.checkpoint_contract.checkpoint_path, self.checkpoint_contract.checkpoint_sha256,
                self.showdown_root, self.checkpoint_contract.showdown_source_sha256),
            "actual reference factory differs from external runtime")
        public = public_only_context(self.context)
        request = PublicRootRequest.capture(public.public_materialization_state, public.observation,
            pending_transition=self.pending_transition)
        public = replace(public, observation=request.observation,
            requested_observations={public.player_id:request.observation}, public_materialization_state=request.state)
        require(request.set_source_hash == factory.set_source_hash
            and type(self.statistics_root_id) is str and self.statistics_root_id
            and self.root_id == self.statistics_root_id+":"+self.configuration.identity
            and type(self.selection_seed) is int and 0 <= self.selection_seed < 2**64
            and all(type(v) is bool for v in request.observation.legal_action_mask)
            and sum(request.observation.legal_action_mask) >= 2,
            "non-forced public reference root/catalog/seed drift")
        object.__setattr__(self, "context", public)
        object.__setattr__(self, "_public_request", request)
        object.__setattr__(self, "_consumed_binding", _binding(request))
        object.__setattr__(self, "_legal_action_mask", tuple(request.observation.legal_action_mask))
        require((self.expected_public_request_sha256 is not None) ==
            ("full_public_request_observation" in self.expected_runtime_configuration),
            "reference job full request scope/digest drift")
        if self.expected_public_request_sha256 is not None:
            require(public_request_witness(request)["public_request_sha256"] == self.expected_public_request_sha256,
                "reference job differs from full archived public request")
        object.__setattr__(self, "_frozen_binding", self._inputs())

    def _inputs(self):
        public = None if self.expected_public_request_sha256 is None else public_request_witness(
            replace(self._public_request, state=self.context.public_materialization_state,
                observation=self.context.observation, pending_transition=self.pending_transition))
        return digest(dict(configuration=asdict(self.configuration),factory=asdict(self.reference_factory),
            runtime=self.expected_runtime_configuration,root=self.root_id,statistics_root=self.statistics_root_id,
            seed=self.selection_seed,consumed=self._consumed_binding,mask=self._legal_action_mask,
            contract=self.contract.receipt(),showdown=self.showdown_root,
            checkpoint=dict(path=self.checkpoint_contract.checkpoint_path,
                sha256=self.checkpoint_contract.checkpoint_sha256,
                source=self.checkpoint_contract.showdown_source_sha256),
            player=self.context.player_id, decision_round=getattr(self.context,"decision_round",None),
            expected_public_request_sha256=self.expected_public_request_sha256, live_public_request=public))

    @property
    def expected_runtime_sha256(self):
        return digest(self.expected_runtime_configuration)

    def read_selection(self, selected):
        require(self._inputs() == self._frozen_binding, "reference job bindings changed after creation")
        self._runtime_check(self.configuration, self.expected_runtime_configuration, self.expected_runtime_sha256)
        require(digest(selected.get("reference_timed_run_contract")) == digest(self.contract.receipt())
            and digest(selected.get("observed_runtime_configuration")) == self.expected_runtime_sha256,
            "reference owner selected contract/runtime drift")
        return read_reference_timed_selection(selected, configuration=self.configuration,
            runtime_configuration=self.expected_runtime_configuration, runtime_sha256=self.expected_runtime_sha256,
            root_id=self.root_id, statistics_root_id=self.statistics_root_id, selection_seed=self.selection_seed,
            consumed_binding=self._consumed_binding, legal_action_mask=self._legal_action_mask,
            public_request_sha256=self.expected_public_request_sha256)

    def __call__(self, ownership, phase):
        adapter, primary_error = None, None
        phase("startup")
        try:
            from ..randbat import load_gen3_randbat_source_cached
            require(self._inputs() == self._frozen_binding, "reference job bindings changed before startup")
            self._runtime_check(self.configuration, self.expected_runtime_configuration, self.expected_runtime_sha256)
            require(digest(asdict(self.reference_factory)) == digest(self.expected_runtime_configuration["reference_factory"]),
                "reference factory changed before startup")
            source = load_gen3_randbat_source_cached(self.showdown_root)
            self._check_source(source)
            state = self.context.public_materialization_state
            engine = state.belief_engine.clone(); engine.set_source = source
            context = replace(self.context, public_materialization_state=replace(state, belief_engine=engine))
            adapter = self._make_adapter(ownership)
            def actual_runtime():
                self._check_adapter_scope(adapter)
                require(digest(adapter.runtime_configuration) == self.expected_runtime_sha256
                    and adapter.runtime_sha256 == self.expected_runtime_sha256
                    and adapter._pool.workers == 20 and adapter._pool.batch_size == 10
                    and adapter._pool.initial_dispatch_workers == 6,
                    "actual reference timed runtime/allocation differs from review")
            actual_runtime()
            phase("selection")
            selected = adapter.select(context, root_id=self.root_id, selection_seed=self.selection_seed,
                statistics_root_id=self.statistics_root_id, pending_transition=self.pending_transition)
            actual_runtime()
            selected["reference_timed_run_contract"] = self.contract.receipt()
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
                except BaseException as cleanup:
                    if primary_error is None:
                        raise
                    primary_error.add_note("Reference timed adapter cleanup failed: "+type(cleanup).__name__)
            if primary_error is None:
                if adapter is not None:self._check_adapter_scope(adapter)
                require(self._inputs() == self._frozen_binding,
                    "reference job public inputs changed during selection/cleanup")

    def _check_source(self,source):
        require(source.metadata.source_hash == self._consumed_binding["set_source_hash"], "reference owner catalog drift")

    def _check_adapter_scope(self,adapter):
        pass

    def _make_adapter(self,ownership):
        from .search_over_raw_world_inputs import WorldInputSearchAdapter
        return WorldInputSearchAdapter(self.configuration,timed_reference_atomic_work=True,
            checkpoint_contract=self.checkpoint_contract,showdown_root=self.showdown_root,
            reference_factory=self.reference_factory,initial_dispatch_workers=6,owned_process_receipts=ownership,
            expected_public_request_sha256=self.expected_public_request_sha256)


def supervise_reference_timed(job, *, directory, binding_sha256, verify_admission,
                              _synthetic_watchdog_seconds=None):
    require(type(job) is ReferenceTimedAdapterJob, "reference timed owning job required")
    return _supervise_owned_job(job, directory=directory, binding_sha256=binding_sha256,
        verify_admission=verify_admission, contract=job.contract, kind="reference-timed-run",
        work_phase="selection", validate_result=job.read_selection,
        _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)


class ReferenceTimedRunValidator(NativeTimedRunValidator):
    """Reuse lifecycle/cost/byte guards only, never the native accepted-work law."""
    _runtime_check = staticmethod(_runtime)
    _contract_class = ReferenceTimedRunContract
    _kind = "reference-timed-run"
    _complete_status = "REFERENCE_TIMED_RUN_COMPLETE"
    _readout_schema = "pokezero.search-over-raw.reference-timed-owning-readout.v1"
    _native_public_binding = False

    def __init__(self, *, public_request_sha256=None, **kwargs):
        require(public_request_sha256 is None or (type(public_request_sha256) is str
            and len(public_request_sha256) == 64
            and all(c in "0123456789abcdef" for c in public_request_sha256)),
            "external reference full public digest required")
        self.reference_public_request_sha = public_request_sha256
        super().__init__(**kwargs)
        require((public_request_sha256 is not None) == ("full_public_request_observation" in self.runtime),
            "reference owning reader full scope/digest drift")
        factory = self.runtime["reference_factory"]
        require(factory["checkpoint"] == self.checkpoint_path and factory["showdown_root"] == self.showdown_root,
            "reference external file paths differ from factory")

    def _inputs(self):
        return digest(dict(lifecycle=super()._inputs(),
            reference_public_request_sha256=self.reference_public_request_sha))

    def _read_work(self, saved):
        self._runtime_check(self.configuration, self.runtime, self.runtime_sha)
        require(digest(saved.get("reference_timed_run_contract")) == digest(self.contract.receipt())
            and digest(saved.get("observed_runtime_configuration")) == self.runtime_sha,
            "reference owner selected contract/runtime drift")
        return read_reference_timed_selection(saved, configuration=self.configuration,
            runtime_configuration=self.runtime, runtime_sha256=self.runtime_sha, root_id=self.root_id,
            statistics_root_id=self.statistics_root_id, selection_seed=self.seed,
            consumed_binding=self.consumed, legal_action_mask=self.mask,
            public_request_sha256=self.reference_public_request_sha)

    def _read_owned(self, saved, terminal, pid, read, hashes):
        owned = self.directory/"owned-groups"
        paths = sorted(owned.glob("group-*.json"))
        expected_files = sorted([p.name for p in paths]+["closed-"+p.name for p in paths])
        actual_files = sorted(str(p.relative_to(owned)) for p in owned.rglob("*") if p.is_file())
        require(owned.is_dir() and len(paths) == 20 and actual_files == expected_files
            and terminal["owned_groups"] == [pid], "reference requires exactly twenty retired generations, no active/extras")
        by_index = {}
        for path in paths:
            child = read("owned-groups/"+path.name)
            closed = read("owned-groups/closed-"+path.name)
            child_pid, index = child["pid"], child.get("worker_index")
            require(type(child_pid) is int and child_pid > 0 and child_pid != pid
                and type(index) is int and 0 <= index < 20 and index not in by_index
                and type(child["group"]) is int and child["group"] == child_pid
                and type(child["controller_pid"]) is int and child["controller_pid"] == pid
                and child["registration_sha256"] == self.binding and child["protocol"] == "active-generations-v1"
                and type(child["birth_identity"]) is str and child["birth_identity"]
                and type(child["generation"]) is str and child["generation"]
                and path.name == f"group-{child_pid}-{child['generation']}.json"
                and closed["launch_sha256"] == hashes["owned-groups/"+path.name]
                and type(closed["pid"]) is int and closed["pid"] == child_pid
                and type(closed["controller_pid"]) is int and closed["controller_pid"] == pid
                and closed["generation"] == child["generation"] and closed["group_live"] is False
                and type(closed["worker_exit_code"]) is int and closed["worker_exit_code"] == 0
                and closed.get("cleanup_protocol") == "live_leader_ack_before_reap.v2"
                and closed.get("clean_acknowledgment") is True and closed.get("forced_cleanup") is False
                and closed.get("runtime_close_failed") is False
                and type(closed.get("cleanup_exception_count")) is int and closed["cleanup_exception_count"] == 0,
                "normal same-generation reference close/index/ownership required")
            by_index[index] = child_pid
        ordered = [by_index[i] for i in range(20)]
        actual = saved["evidence"].get("worker_pids")
        require(len(set(ordered)) == 20 and type(actual) is list and len(actual) == 20
            and all(type(p) is int for p in actual) and actual == ordered,
            "reference raw worker positions differ from actual indexed owned generations")
        return ordered
