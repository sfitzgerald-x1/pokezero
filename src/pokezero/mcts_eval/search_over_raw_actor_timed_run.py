"""Opt-in timed model/HP selection with the original actor-root annotations.

This transports a genuine actor boundary, not a sampled opponent observer seed.
It does not evolve residual trackers, implement sampled-opponent history, run
terminal labels, or qualify physics, full cost or scientific collection.
Historical timed jobs and recipes cannot silently adopt this owning kind.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, fields

from .search_over_raw import digest, require
from .search_over_raw_annotation_boundary import ActorAnnotationBoundary, SCHEMA as BOUNDARY_SCHEMA
from .search_over_raw_outcomes import _sha
from .search_over_raw_supervision import _supervise_owned_job
from .search_over_raw_timed_run import (
    NativeTimedAdapterJob, NativeTimedRunContract, NativeTimedRunValidator,
    _configuration, _runtime, _read_native_timed_selection, _compact_runtime,
)
from .search_over_raw_world_inputs import (
    WorldInputSearchAdapter, TIMED_NATIVE_PUBLIC_CAPTURE, _state_payload, native_atomic_runtime_scopes,
)

RUNTIME_KEY = "original_actor_annotation_handoff"
KIND = "actor-annotated-native-timed-run"
UNAVAILABLE_REASONS = frozenset({"annotation_cache_absent", "actor_tracker_absent",
    "actor_tracker_unassessed", "archived_request_mismatch", "archived_observation_mismatch"})


class ActorInputUnavailable(ValueError):
    """Known prelaunch input gap, never a worker/provenance failure."""
    def __init__(self, reason):
        require(reason in UNAVAILABLE_REASONS, "known actor input-unavailable reason required")
        self.reason = reason
        super().__init__("ACTOR_INPUT_UNAVAILABLE: "+reason)


def _same_original(value, original):
    return type(value) is type(original) and (value == original
        if type(original) in (str, int, float, bool, type(None)) else value is original)


def annotation_runtime_scope():
    return dict(schema="pokezero.search-over-raw.actor-timed-annotations.v1",
        boundary_schema=BOUNDARY_SCHEMA, root_annotations_only=True,
        sampled_opponent_history_implemented=False, branch_residual_tracker_evolved=False,
        instrumentation_can_change_deadlines=True, runtime_qualified=False)


def actor_timed_runtime_recipe(configuration, checkpoint_contract, *, compact_native_atomic_work=False):
    """No model load or future-root fields; a distinct prospective recipe."""
    _configuration(configuration)
    require(type(compact_native_atomic_work) is bool, "explicit prospective actor encoding required")
    value = dict(configuration=asdict(configuration),
        checkpoint_sha256=checkpoint_contract.checkpoint_sha256,
        set_source_hash=checkpoint_contract.showdown_source_sha256,
        root_statistics="independent_per_root",
        incumbent=dict(depth=6, sims=4096, batch=16, worlds=4, native_batch_guard_ms=64),
        reference_factory=None, initial_dispatch_workers=None,
        consumed_world_inputs=dict(schema="pokezero.search-over-raw.consumed-world-inputs.v1",
            initial_inputs_only=True, snapshot_fields_excluded=[], ownership_metadata_excluded=[],
            instrumentation_can_change_deadlines=True, qualifies_uninstrumented_runtime=False),
        native_atomic_work=dict(schema="native-atomic-backups-v1",
            scope="observed_float32_backup_consistency; not independent terminal labels",
            instrumentation_can_change_deadlines=True, runtime_qualified=False),
        timed_native_atomic_work=dict(schema="pokezero.search-over-raw.timed-native-prefix.v1",
            native_budget_abi_slot=24, zero_work_not_scored=True,
            on_time_deployability_established=False, runtime_qualified=False),
        full_public_request_observation=deepcopy(TIMED_NATIVE_PUBLIC_CAPTURE))
    if configuration.leaf == "hp_fraction":
        value["incumbent_leaf"] = dict(leaf="hp_fraction", tree="unchanged_encoded_model_tree",
            priors="unchanged_champion", model_forwards="retained", value_frame="side_one_absolute")
    value[RUNTIME_KEY] = annotation_runtime_scope()
    if compact_native_atomic_work:
        value["compact_native_atomic_work"] = True
        value["native_atomic_work"], value["timed_native_atomic_work"] = native_atomic_runtime_scopes(compact=True)
    _actor_runtime(configuration, value, digest(value))
    return value


def _actor_runtime(configuration, runtime, runtime_sha):
    _runtime(configuration, runtime, runtime_sha)
    require(digest(runtime.get(RUNTIME_KEY)) == digest(annotation_runtime_scope())
        and digest(runtime.get("full_public_request_observation")) == digest(TIMED_NATIVE_PUBLIC_CAPTURE),
        "explicit original-actor/full-public timed recipe required")


def _annotation_witness(boundary_sha, request_sha, player):
    require(_sha(boundary_sha) and _sha(request_sha) and player in {"p1", "p2"},
        "prospective original actor boundary bindings required")
    return dict(**annotation_runtime_scope(), original_boundary_sha256=boundary_sha,
        public_request_sha256=request_sha, player=player)


class ActorAnnotatedTimedAdapter(WorldInputSearchAdapter):
    """The retained actor source is installed only during this one selection."""
    def __init__(self, configuration, *, annotation_boundary, **kwargs):
        _configuration(configuration)
        require(type(annotation_boundary) is ActorAnnotationBoundary
            and kwargs.get("expected_public_request_sha256") is not None
            and kwargs.get("timed_native_atomic_work", True) is True
            and all(kwargs.get(k) is None for k in ("native_ancestry_contract", "value_diagnostics",
                "reference_factory", "policy_opponent_diagnostics"))
            and not kwargs.get("capture_native_public_fold", False),
            "distinct actor-only public timed adapter required")
        self.annotation_boundary = self._original_boundary = annotation_boundary
        self._annotation_restore = None
        self._annotation_attempted = False
        kwargs["timed_native_atomic_work"] = True
        super().__init__(configuration, **kwargs)
        self.runtime_configuration[RUNTIME_KEY] = annotation_runtime_scope()
        self.runtime_sha256 = digest(self.runtime_configuration)
        self._original_actor_native = self._native
        self._actor_runtime_original = (self.runtime_configuration, self.runtime_sha256)
        self._actor_refs = tuple((name, getattr(self, name)) for name in
            ("configuration", "contract", "_expected_public_request_sha256", "_compact_native_atomic_work"))
        self._actor_native_config = self._native._config
        self._actor_values = self._actor_value_inputs()

    def _actor_value_inputs(self):
        return digest(dict(configuration=_state_payload(self.configuration),
            checkpoint=_state_payload(self.contract), native_config=_state_payload(self._native._config)))

    def _actor_inputs(self):
        runtime, sha = self._actor_runtime_original
        require(self.annotation_boundary is self._original_boundary
            and self._native is self._original_actor_native
            and self.runtime_configuration is runtime and digest(runtime) == sha
            and self.runtime_sha256 == sha
            and all(_same_original(getattr(self, name), original) for name, original in self._actor_refs)
            and self._native._config is self._actor_native_config
            and self._actor_value_inputs() == self._actor_values,
            "original actor timed adapter inputs changed")
        self.annotation_boundary._check()

    def _prepare_incumbent(self, request):
        self._actor_inputs()
        self.annotation_boundary.validate(request, self.contract)
        native = self._native
        self._annotation_restore = (native, native._annotation_source)
        native._annotation_source = self.annotation_boundary
        super()._prepare_incumbent(request)

    def select(self, context, **kwargs):
        require(not self._annotation_attempted, "actor timed adapter is single-attempt; no retry")
        self._annotation_attempted = True
        native, original_source = self._native, self._native._annotation_source
        try:
            self._actor_inputs()
            selected = super().select(context, **kwargs)
            self._actor_inputs()
            require(self._annotation_restore is not None
                and self._annotation_restore[0] is native
                and self._annotation_restore[1] is original_source
                and self._native._annotation_source is self.annotation_boundary,
                "actual original actor source changed during timed selection")
            selected["evidence"][RUNTIME_KEY] = _annotation_witness(
                self.annotation_boundary._original_sha,
                self.annotation_boundary.public_request_sha256, context.player_id)
            return selected
        except BaseException:
            self._poisoned = True
            raise
        finally:
            # Callback-visible bookkeeping is not cleanup authority. Restore
            # the actual original object even if that slot was nullified.
            native._annotation_source = original_source
            self._annotation_restore = None


def _read_actor_work(selected, *, annotation_sha256, **kwargs):
    before = digest(selected)
    expected = _annotation_witness(annotation_sha256, kwargs["public_request_sha256"],
        kwargs["consumed_binding"]["subject"])
    require(digest(selected["evidence"].get(RUNTIME_KEY)) == digest(expected),
        "actual actor-root annotation witness differs from prospective handoff")
    result = _read_native_timed_selection(selected, runtime_check=_actor_runtime,
        contract_class=NativeTimedRunContract, **kwargs)
    require(digest(selected) == before, "actor timed selection changed during independent read")
    result["original_actor_annotation_handoff"] = expected
    return result


@dataclass(frozen=True, kw_only=True)
class ActorAnnotatedTimedJob(NativeTimedAdapterJob):
    annotation_boundary: ActorAnnotationBoundary
    synthetic_process_fixture: bool = False
    _runtime_check = staticmethod(_actor_runtime)

    def __post_init__(self):
        require(type(self.annotation_boundary) is ActorAnnotationBoundary
            and type(self.synthetic_process_fixture) is bool
            and self.expected_public_request_sha256 is not None,
            "original typed actor boundary/full-request owning job required")
        object.__setattr__(self, "_actor_boundary_original", self.annotation_boundary)
        super().__post_init__()
        self.annotation_boundary.validate(self._public_request, self.checkpoint_contract)
        object.__setattr__(self, "_actor_job_refs", tuple((f.name, getattr(self, f.name)) for f in fields(self))
            + tuple((name, getattr(self, name)) for name in
                ("_public_request", "_consumed_binding", "_legal_action_mask", "_frozen_inputs")))
        object.__setattr__(self, "_actor_job_values", self._actor_value_inputs())

    def _actor_value_inputs(self):
        return digest(dict(fields={f.name:_state_payload(getattr(self, f.name)) for f in fields(self)},
            consumed=self._consumed_binding, request=_state_payload(self._public_request)))

    def _check(self):
        require(all(_same_original(getattr(self, name), original) for name, original in self._actor_job_refs)
            and self._actor_value_inputs() == self._actor_job_values
            and self._inputs() == self._frozen_inputs, "original actor timed job changed")

    def __call__(self, ownership, phase):
        self._check()
        cleanup_errors = []
        def guarded_phase(name):
            try:
                self._check()
                phase(name)
                self._check()
            except BaseException as error:
                if name != "cleanup":
                    raise
                # The base job notifies cleanup BEFORE closing its original
                # adapter. Defer notification/guard failures until close ran.
                cleanup_errors.append(error)
        try:
            result = super().__call__(ownership, guarded_phase)
            self._check()
            return result
        finally:
            self._check()
            if cleanup_errors:
                raise cleanup_errors[0]

    def _inputs(self):
        require(self.annotation_boundary is self._actor_boundary_original,
            "original actor timed job boundary replaced")
        self.annotation_boundary._check()
        return digest(dict(base=super()._inputs(), annotation=self.annotation_boundary._original_sha,
            checkpoint=_state_payload(self.checkpoint_contract), synthetic=self.synthetic_process_fixture))

    def _make_adapter(self):
        return ActorAnnotatedTimedAdapter(self.configuration, annotation_boundary=self.annotation_boundary,
            **({"compact_native_atomic_work":True} if _compact_runtime(self.expected_runtime_configuration) else {}),
            expected_public_request_sha256=self.expected_public_request_sha256,
            checkpoint_contract=self.checkpoint_contract, showdown_root=self.showdown_root)

    def _check_adapter_scope(self, adapter):
        require(type(adapter) is ActorAnnotatedTimedAdapter
            and adapter.annotation_boundary is self.annotation_boundary,
            "original actor timed owning adapter replaced")
        adapter._actor_inputs()

    def read_selection(self, selected):
        self._check()
        result = _read_actor_work(selected, annotation_sha256=self.annotation_boundary._original_sha,
            configuration=self.configuration, runtime_configuration=self.expected_runtime_configuration,
            runtime_sha256=self.expected_runtime_sha256, root_id=self.root_id,
            statistics_root_id=self.statistics_root_id, selection_seed=self.selection_seed,
            consumed_binding=self._consumed_binding, legal_action_mask=self._legal_action_mask,
            decision_round=self.context.decision_round_index, contract=self.contract,
            public_request_sha256=self.expected_public_request_sha256)
        self._check()
        return result


def prepare_archived_actor_timed_job(archive, source, *, contract, root_id, configuration,
        checkpoint_contract, showdown_root, runtime_configuration, source_config, source_config_sha256,
        synthetic_process_fixture=False):
    """Conditional original-p1 reconstruction; no bridge, model or source launch.

    The retained snapshot is after both seats' policy activity, not necessarily
    the original p1 encoder boundary. Only exact original observation/request
    correspondence and unchanged already-assessed trackers admit transport.
    Missing or mismatching inputs are unavailable, never repaired here.
    """
    from ..local_showdown import LocalShowdownEnv, LocalShowdownConfig, LocalShowdownSnapshot, SnapshotAnnotationCache
    from ..transitions import extract_transition_tokens
    from .paper_reference_runtime import PublicRootRequest
    from .search_over_raw_archive import (
        SealedSourceArchive, archived_exploration_root_binding, retained_archive_root_identity)
    from .search_over_raw_world_inputs import public_request_witness
    require(type(archive) is SealedSourceArchive and type(source_config) is LocalShowdownConfig
        and type(synthetic_process_fixture) is bool and (not synthetic_process_fixture or
            contract.get("execution_ready") is False and contract.get("collection_authorized") is False)
        and _sha(source_config_sha256) and digest(_state_payload(source_config)) == source_config_sha256
        and str(source_config.resolved_showdown_root().resolve()) == showdown_root == checkpoint_contract.showdown_root,
        "prospectively bound original source configuration required")
    root = next(r for r in source["roots"] if r["root_id"] == root_id)
    index = root["source_request_index"]
    require(type(index) is int and index in archive._public and index in archive._private,
        "original archived actor root unavailable")
    context, pending, _ = archive._public[index]
    snapshot = archive._private[index]
    require(type(snapshot) is LocalShowdownSnapshot and context.player_id == "p1",
        "original typed actor snapshot required")
    if snapshot.annotation_cache is None: raise ActorInputUnavailable("annotation_cache_absent")
    require(type(snapshot.annotation_cache) is SnapshotAnnotationCache, "original typed annotation cache required")
    refs = (archive._public, archive._private, archive._public[index], snapshot, context, pending)
    owners = (context, snapshot, snapshot.annotation_cache, source_config, checkpoint_contract)
    field_refs = tuple((owner, f.name, getattr(owner, f.name)) for owner in owners for f in fields(owner))
    source_roots_original = source["roots"]
    frozen = digest(dict(archive=retained_archive_root_identity(archive, index), source=source,
        contract=contract, checkpoint=_state_payload(checkpoint_contract), config=_state_payload(source_config),
        runtime=runtime_configuration, configuration=_state_payload(configuration)))
    def unchanged():
        require(source["roots"] is source_roots_original and any(actual is root for actual in source["roots"])
            and all(_same_original(getattr(owner, name), original) for owner, name, original in field_refs)
            and all(actual is original for actual, original in zip(
            (archive._public, archive._private, archive._public[index], archive._private[index],
                archive._public[index][0], archive._public[index][1]), refs))
            and digest(dict(archive=retained_archive_root_identity(archive, index), source=source,
                contract=contract, checkpoint=_state_payload(checkpoint_contract), config=_state_payload(source_config),
                runtime=runtime_configuration, configuration=_state_payload(configuration))) == frozen,
            "original archived actor timed preparation changed")
    unchanged()
    before = archived_exploration_root_binding(archive, source, contract=contract, root_id=root_id)
    unchanged()
    before_sha = digest(before)
    selected = archive.selected(root)
    unchanged()
    require(all(actual is original for actual, original in zip(selected, (context, pending, snapshot))),
        "original archived actor extraction replaced")
    env = LocalShowdownEnv(source_config)
    boundary = None
    try:
        require(env._process is None and env._battle_token is None, "process-free actor preparer required")
        env._restore_local_snapshot_state(snapshot)
        token_count = len(extract_transition_tokens(snapshot.replay, perspective_slot="p1"))
        active_trackers = []
        if env.tier2_residuals_active(): active_trackers.append(env._tier2_trackers.get("p1"))
        if env.investment_active() or env.investment_belief_narrowing_active():
            active_trackers.append(env._investment_trackers.get("p1"))
        unchanged()
        if any(tracker is None for tracker in active_trackers): raise ActorInputUnavailable("actor_tracker_absent")
        require(all(type(tracker._assessed_until) is int and 0 <= tracker._assessed_until <= token_count
            for tracker in active_trackers), "malformed original actor assessment position")
        if any(tracker._assessed_until < token_count for tracker in active_trackers):
            raise ActorInputUnavailable("actor_tracker_unassessed")
        def annotation_inputs():
            return digest(dict(belief=_state_payload(env._belief_engine), cache=_state_payload(env._annotation_cache())))
        annotations = annotation_inputs()
        actual_request = PublicRootRequest.capture(env.public_materialization_state("p1"),
            context.observation, pending_transition=pending)
        original_request = PublicRootRequest.capture(context.public_materialization_state,
            context.observation, pending_transition=pending)
        unchanged()
        require(annotation_inputs() == annotations, "actor annotation/belief changed while checking request")
        if public_request_witness(actual_request)["public_request_sha256"] != public_request_witness(
                original_request)["public_request_sha256"]:
            raise ActorInputUnavailable("archived_request_mismatch")
        from ..public_decision_corpus import PublicObservation
        observed = env.observe("p1")
        unchanged()
        require(annotation_inputs() == annotations, "actor annotation/belief changed while checking observation")
        if digest(PublicObservation.from_observation(observed).to_dict()) != digest(
                PublicObservation.from_observation(context.observation).to_dict()):
            raise ActorInputUnavailable("archived_observation_mismatch")
        boundary = ActorAnnotationBoundary.capture(env, context, pending_transition=pending)
        original_request = PublicRootRequest.capture(context.public_materialization_state,
            context.observation, pending_transition=pending)
        boundary.validate(original_request, checkpoint_contract)
        require(annotation_inputs() == annotations and env._process is None and env._battle_token is None,
            "LABEL_INPUT_UNAVAILABLE: archived actor observation changed original annotation/belief state")
        require(boundary.public_request_sha256 == before["public_request_witness"]["public_request_sha256"]
            == public_request_witness(original_request)["public_request_sha256"],
            "LABEL_INPUT_UNAVAILABLE: reconstructed actor differs from original archive request")
        unchanged()
    finally:
        try:
            env.close()
        finally:
            unchanged()
            if boundary is not None: boundary._check()
    job = ActorAnnotatedTimedJob(configuration=configuration, checkpoint_contract=checkpoint_contract,
        showdown_root=showdown_root, context=deepcopy(context), pending_transition=deepcopy(pending),
        root_id=root_id+":"+configuration.identity, statistics_root_id=root_id,
        selection_seed=before["slot"]["source_seed"], expected_runtime_configuration=runtime_configuration,
        expected_public_request_sha256=boundary.public_request_sha256, annotation_boundary=boundary,
        synthetic_process_fixture=synthetic_process_fixture)
    def original_job():
        require(type(job) is ActorAnnotatedTimedJob and job.annotation_boundary is boundary
            and job.configuration is configuration and job.checkpoint_contract is checkpoint_contract
            and job.showdown_root == showdown_root and job.root_id == root_id+":"+configuration.identity
            and job.statistics_root_id == root_id and type(job.selection_seed) is int
            and job.synthetic_process_fixture is synthetic_process_fixture
            and job.selection_seed == before["slot"]["source_seed"] == context.seed
            and job.context.player_id == "p1" and type(job.context.decision_round_index) is int
            and job.context.decision_round_index == index
            and digest(_state_payload(job.pending_transition)) == digest(_state_payload(pending))
            and digest(job.expected_runtime_configuration) == digest(runtime_configuration)
            and digest(job.contract.receipt()) == digest(NativeTimedRunContract().receipt())
            and public_request_witness(job._public_request)["public_request_sha256"]
                == boundary.public_request_sha256 == before["public_request_witness"]["public_request_sha256"],
            "returned actor timed job differs from original prospective archive handoff")
        job._check()
    original_job()
    job._check()
    unchanged()
    final = archived_exploration_root_binding(archive, source, contract=contract, root_id=root_id)
    require(digest(before) == before_sha == digest(final),
        "archived actor root or returned binding changed during final preparation")
    original_job()
    unchanged()
    require(digest(before) == before_sha, "returned actor archive binding changed after final callbacks")
    return job, before


def supervise_actor_timed(job, *, directory, binding_sha256, verify_admission):
    require(type(job) is ActorAnnotatedTimedJob, "distinct original-actor timed owning job required")
    job._check()
    def admission():
        job._check()
        result = verify_admission()
        job._check()
        return result
    result = _supervise_owned_job(job, directory=directory, binding_sha256=binding_sha256,
        verify_admission=admission, contract=job.contract, kind=KIND,
        work_phase="selection", validate_result=job.read_selection)
    job._check()
    if result["supervisor_exit_code"] == 0:
        require(digest(job.read_selection(result["result"])) == digest(result["observed_work_accounting"]),
            "actor timed supervisor final accounting changed")
    job._check()
    return result


class ActorAnnotatedTimedRunValidator(NativeTimedRunValidator):
    _runtime_check = staticmethod(_actor_runtime)
    _kind = KIND
    _complete_status = "ACTOR_ANNOTATED_NATIVE_TIMED_RUN_COMPLETE"
    _readout_schema = "pokezero.search-over-raw.actor-timed-owning-readout.v1"

    def __init__(self, *, annotation_sha256, **kwargs):
        require(_sha(annotation_sha256) and kwargs.get("public_request_sha256") is not None,
            "prospective original actor timed reader bindings required")
        self.annotation_sha256 = annotation_sha256
        super().__init__(**kwargs)

    def _inputs(self):
        return digest(dict(base=super()._inputs(), annotation=self.annotation_sha256))

    def _check(self):
        require(set(vars(self)) == {name for name, _ in self._actor_reader_refs}
                | {"last_validation", "_actor_reader_refs", "_actor_reader_scope"}
            and all(name not in vars(self) and getattr(type(self), name) is original
                for name, original in self._actor_reader_scope)
            and all(_same_original(getattr(self, name), original) for name, original in self._actor_reader_refs)
            and self._inputs() == self.frozen, "original actor timed reader changed")

    def _files(self):
        # The base constructor invokes this after all external bindings and the
        # value digest exist. Freeze before its first file boundary, not after.
        if not hasattr(self, "_actor_reader_refs"):
            self._actor_reader_refs = tuple((name, value) for name, value in vars(self).items()
                if name != "last_validation")
            self._actor_reader_scope = tuple((name, getattr(type(self), name)) for name in
                ("_kind", "_complete_status", "_readout_schema", "_native_public_binding",
                 "_contract_class", "_runtime_check", "_read_work", "_read_owned", "_files"))
        self._check()
        super()._files()
        self._check()

    def __call__(self, selected):
        self.last_validation = None
        try:
            self._check()
            result = super().__call__(selected)
            self._check()
            return result
        except BaseException:
            self.last_validation = None
            raise

    def _read_work(self, saved):
        return _read_actor_work(saved, annotation_sha256=self.annotation_sha256,
            configuration=self.configuration, runtime_configuration=self.runtime, runtime_sha256=self.runtime_sha,
            root_id=self.root_id, statistics_root_id=self.statistics_root_id, selection_seed=self.seed,
            consumed_binding=self.consumed, legal_action_mask=self.mask, decision_round=self.round,
            contract=self.contract, public_request_sha256=self.timed_public_request_sha)
