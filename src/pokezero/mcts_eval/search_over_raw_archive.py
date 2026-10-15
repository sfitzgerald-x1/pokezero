"""Trusted in-memory source boundary archive; never a search input container.

The public policy callback and private pre-step callback have separate types.
Only selected roots can retrieve an auditor snapshot, after binding their
canonical public records. No private snapshot is exported or serialized.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import resource
import sys

from .search_over_raw import digest, require


class SealedSourceArchive:
    def __reduce_ex__(self, protocol):
        raise TypeError("private source archive cannot be serialized")

    def __init__(self, *, subject="p1"):
        from .paper_reference_substitute import SubstituteHistoryTracker
        require(subject in {"p1", "p2"}, "invalid source subject")
        self.subject = subject
        self._public = {}
        self._private = {}
        self._previous = None
        self._substitute = SubstituteHistoryTracker()

    def capture_public(self, context, action):
        from .paper_reference_pending import (
            PendingPolicyTransition, FaintReplacementTransition,
            requires_baton_interruption_replay, requires_faint_encore_replay)
        from .paper_reference_runtime import PublicRootRequest
        from ..public_decision_corpus import PublicObservation, _public_belief_view

        require(context.player_id == self.subject, "opponent context cannot enter actor archive")
        index = context.decision_round_index
        require(index not in self._public, "duplicate public source boundary")
        state = context.public_materialization_state
        pending = self._substitute.certificate(state)
        if pending is None and (state.deferred_opponent_action_player is not None
                                or requires_baton_interruption_replay(state)):
            pending = self._previous
        if requires_faint_encore_replay(state):
            require(self._previous is not None, "forced Encore source lacks prior public root")
            previous = self._previous
            pending = FaintReplacementTransition(previous.before_state, previous.before_observation,
                previous.own_action, previous.set_source_hash, previous.prior_transition)
        request = PublicRootRequest.capture(state, context.observation, pending_transition=pending)
        # Keep only actor observations/actions and resolved PUBLIC rounds. The
        # original rollout may carry opponent request-local action indices.
        own_steps = tuple(replace(step, observation=PublicObservation.from_observation(step.observation)
            .to_observation(belief_view=_public_belief_view(step.observation.metadata)))
            for step in context.trajectory.steps if step.player_id == self.subject)
        public_context = replace(context, observation=request.observation,
            requested_observations={self.subject: request.observation},
            requested_legal_action_masks={self.subject: request.observation.legal_action_mask},
            trajectory=replace(context.trajectory, steps=own_steps))
        self._public[index] = (public_context, pending, action)
        self._substitute.accept_own(request, action)
        if not state.self_request.get("forceSwitch") and state.deferred_opponent_action_player is None:
            self._previous = PendingPolicyTransition.capture(request, action)

    def capture_private(self, boundary):
        if self.subject not in boundary.requested_players:
            self._substitute.accept_opponent_only()
            return
        index = boundary.decision_round_index
        require(index in self._public and index not in self._private,
            "private boundary missing or duplicates its public binding")
        context, _, action = self._public[index]
        require((boundary.seed, boundary.battle_id) == (context.seed, context.battle_id)
            and boundary.decisions[self.subject].action_index == action,
            "private source identity/action drift")
        self._private[index] = boundary.snapshot

    def selected(self, root):
        from ..public_decision_corpus import (
            PublicDecisionRecord, PublicObservation, PublicActorObservation, _public_belief_view)
        require(root["public_record_sha256"] == digest(root["public_record"]), "public root digest drift")
        record = PublicDecisionRecord.from_dict(root["public_record"])
        require(digest(record.to_dict()) == root["public_record_sha256"]
            and type(root["source_request_index"]) is int,
            "selected source record must be canonical; no parser coercion aliases")
        index = root["source_request_index"]
        require(index == record.turn_index and index in self._public and index in self._private,
            "selected source boundary unavailable")
        context, pending, action = self._public[index]
        require(digest((record.seed, record.battle_id, record.acting_player, record.recorded_action_index)) ==
                digest((context.seed, context.battle_id, self.subject, action))
            and record.format_id == context.format_id
            and digest(record.observation.to_dict()) == digest(PublicObservation.from_observation(context.observation).to_dict()),
            "selected canonical public root differs from sealed source")
        history = [PublicActorObservation(step.turn_index, PublicObservation.from_observation(step.observation)).to_dict()
            for step in context.trajectory.steps]
        require(digest([row.to_dict() for row in record.history]) == digest(history)
            and digest(record.public_belief_view) == digest(_public_belief_view(context.observation.metadata))
            and digest([row.to_dict() for row in record.public_resolved_action_rounds]) ==
                digest(context.trajectory.metadata.get("public_resolved_action_rounds", [])),
            "selected public history/belief differs from sealed source")
        return context, pending, self._private[index]

    def close(self):
        self._public.clear()
        self._private.clear()
        self._previous = None
        self._substitute.transition = None


def retained_archive_root_identity(archive, index):
    """Callback-free retained root fingerprint; never exports private state."""
    from copy import copy
    from .paper_reference_runtime import PublicRootRequest
    from .search_over_raw_world_inputs import public_request_witness, _state_payload
    require(type(archive) is SealedSourceArchive and type(index) is int,
        "exact retained source root required")
    context,pending,action = archive._public[index]
    state = context.public_materialization_state
    engine = copy(state.belief_engine); engine.set_source = None
    request = PublicRootRequest(replace(state,belief_engine=engine),context.observation,
        state.belief_engine.set_source.metadata.source_hash,pending)
    return digest(dict(request=public_request_witness(request),
        snapshot=_state_payload(archive._private[index]),action=action,
        trajectory=_state_payload(context.trajectory),seed=context.seed,
        round=context.decision_round_index,battle=context.battle_id,format=context.format_id,
        subject=archive.subject,requested_masks=context.requested_legal_action_masks))


def archived_exploration_root_binding(archive, source, *, contract, root_id):
    """Private structural crosswalk from an actual archive to public transport.

    No archive data enters a policy here and no source game is opened. The
    source receipt proves only the selected/missing partition, NOT source
    policy, terminal truth, freshness, runtime or collection qualification.
    Holdout sources are deliberately outside this entry point.
    """
    from .search_over_raw_reconciled import validate_contract
    from .search_over_raw_source import validate_source_receipt
    from .search_over_raw_world_inputs import public_request_witness, _state_payload
    from .paper_reference_runtime import PublicRootRequest
    from ..public_decision_corpus import PublicDecisionRecord
    require(type(archive) is SealedSourceArchive and archive.subject == "p1",
        "actual p1 source archive required")
    validate_contract(contract)
    require(type(source) is dict and type(source.get("source_seed")) is int
        and type(root_id) is str and root_id and source.get("panel") == "exploration",
        "exploration-only source identity required; holdout remains closed")
    seed = source["source_seed"]
    require(seed in contract["panels"]["exploration"]["seeds"], "unregistered exploration source")
    validate_source_receipt(source, seed, contract=contract, panel="exploration")
    require(type(source.get("eligible_requests")) is int
        and type(source.get("requested_root_slots")) is int,
        "canonical source counts required")
    slots = [s for s in contract["panels"]["exploration"]["root_slots"] if s["root_id"] == root_id]
    roots = [r for r in source["roots"] if r["root_id"] == root_id]
    require(len(slots) == len(roots) == 1 and slots[0]["source_seed"] == seed,
        "selected exploration slot required; missing slots remain uncertain")
    root, slot = deepcopy(roots[0]), deepcopy(slots[0])
    record = PublicDecisionRecord.from_dict(root["public_record"])
    context, pending, snapshot = archive.selected(root)
    require(type(context.seed) is int and context.seed == seed
        and type(context.decision_round_index) is int
        and context.decision_round_index == root["source_request_index"], "actual archived source identity drift")
    request = PublicRootRequest.capture(context.public_materialization_state,
        context.observation, pending_transition=pending)
    witness = public_request_witness(request)
    result = dict(schema="pokezero.search-over-raw.private-archived-exploration-root.v1",
        private_controller_only=True, contract_sha256=digest(contract), source_receipt_sha256=digest(source),
        slot=slot, source_request_index=root["source_request_index"],
        public_record_sha256=root["public_record_sha256"], public_request_witness=witness,
        public_evidence_binding=dict(root_id=root_id, source_seed=seed,
            public_record_sha256=root["public_record_sha256"],
            information_key=witness["consumed_binding"]["information_key"],
            legal_action_mask=list(record.current_legal_action_mask)),
        private_auditor_snapshot_sha256=digest(_state_payload(snapshot)),
        missing_source_root_ids=deepcopy(source["missing_root_ids"]),
        source_policy_validated=False, source_terminal_validated=False,
        source_freshness_validated=False, runtime_qualified=False,
        collection_authorized=False, holdout_authorized=False)
    return result


def prepare_archived_fixed_work_job(archive, source, *, contract, root_id, configuration,
                                   checkpoint_contract, showdown_root, runtime_sha256,
                                   native_ancestry_contract=None, capture_native_public_fold=False):
    """Prepare one public native diagnostic job; never launch or reserve it."""
    from .search_over_raw_supervision import FixedWorkAdapterJob
    from .search_over_raw import SearchConfiguration
    before = archived_exploration_root_binding(archive, source, contract=contract, root_id=root_id)
    require(type(configuration) is SearchConfiguration and configuration.arm == "incumbent"
        and before["slot"]["root_slot"] == 0,
        "archived native diagnostic requires the predeclared slot-zero subset")
    root = next(r for r in source["roots"] if r["root_id"] == root_id)
    context, pending, _ = archive.selected(root)
    job = FixedWorkAdapterJob(configuration=configuration, checkpoint_contract=checkpoint_contract,
        showdown_root=showdown_root, context=deepcopy(context), pending_transition=deepcopy(pending),
        root_id=root_id + ":" + configuration.identity, statistics_root_id=root_id,
        selection_seed=before["slot"]["source_seed"], expected_runtime_sha256=runtime_sha256,
        native_ancestry_contract=native_ancestry_contract,
        capture_native_public_fold=capture_native_public_fold,
        expected_public_request_sha256=before["public_request_witness"]["public_request_sha256"])
    require(digest(before) == digest(archived_exploration_root_binding(archive, source,
        contract=contract, root_id=root_id)), "archived root changed during job preparation")
    return job, before


def prepare_archived_reference_fixed_work_job(archive, source, *, contract, root_id, configuration,
        checkpoint_contract, showdown_root, runtime_sha256, reference_factory,owned_reference_value_recipe=None):
    """Slot-zero public fixed work, with actual twenty-worker request binding.

    No source game, draw, worker, admission or deployment selector is created.
    The pending transition stays in the complete public transport witness;
    private source snapshots never enter the returned job.
    """
    from .search_over_raw import SearchConfiguration
    from .search_over_raw_supervision import FixedWorkAdapterJob
    from .paper_reference_runtime import ShowdownWorkerFactory,PublicRootRequest
    from .search_over_raw_world_inputs import public_request_witness,_state_payload
    before=archived_exploration_root_binding(archive,source,contract=contract,root_id=root_id)
    require(type(configuration) is SearchConfiguration and configuration.arm=="reference"
        and configuration.belief=="public" and configuration.seconds==1. and configuration.workers==20
        and before["slot"]["root_slot"]==0,"archived reference fixed work requires public slot-zero diagnostic")
    require(type(reference_factory) is ShowdownWorkerFactory
        and reference_factory.checkpoint==checkpoint_contract.checkpoint_path
        and reference_factory.checkpoint_sha256==checkpoint_contract.checkpoint_sha256
        and reference_factory.showdown_root==showdown_root==checkpoint_contract.showdown_root
        and reference_factory.set_source_hash==checkpoint_contract.showdown_source_sha256,
        "reference diagnostic factory must retain original champion/simulator/catalog")
    factory_sha=digest(_state_payload(reference_factory))
    if owned_reference_value_recipe is not None:
        from .search_over_raw_diagnostic_producer import _check_reference_values
        _check_reference_values(configuration,owned_reference_value_recipe,checkpoint_contract)
    root=next(r for r in source["roots"] if r["root_id"]==root_id)
    context,pending,_=archive.selected(root)
    witness=before["public_request_witness"]
    # Freeze the expected envelope from the ORIGINAL archive and arguments,
    # not from whatever internally valid object a constructor may return.
    from dataclasses import replace
    from .head_to_head import public_only_context
    public=public_only_context(context)
    original_request=PublicRootRequest.capture(public.public_materialization_state,public.observation,
        pending_transition=pending)
    original_context=replace(public,observation=original_request.observation,
        requested_observations={public.player_id:original_request.observation},
        public_materialization_state=original_request.state)
    expected=dict(configuration=_state_payload(configuration),checkpoint_type=type(checkpoint_contract),
        checkpoint=_state_payload(vars(checkpoint_contract)),showdown_root=showdown_root,
        context=_state_payload(original_context),pending=_state_payload(pending),
        root_id=root_id+":"+configuration.identity,statistics_root_id=root_id,
        selection_seed=before["slot"]["source_seed"],runtime_sha256=runtime_sha256,
        public_request_sha256=witness["public_request_sha256"],factory_sha256=factory_sha,
        owned_reference_value_recipe=_state_payload(owned_reference_value_recipe))
    job=FixedWorkAdapterJob(configuration=configuration,checkpoint_contract=checkpoint_contract,
        showdown_root=showdown_root,context=deepcopy(context),pending_transition=deepcopy(pending),
        root_id=root_id+":"+configuration.identity,statistics_root_id=root_id,
        selection_seed=before["slot"]["source_seed"],expected_runtime_sha256=runtime_sha256,
        reference_factory=deepcopy(reference_factory),expected_public_request_sha256=witness["public_request_sha256"],
        owned_reference_value_recipe=deepcopy(owned_reference_value_recipe))
    require(digest(before)==digest(archived_exploration_root_binding(archive,source,contract=contract,root_id=root_id))
        and factory_sha==digest(_state_payload(reference_factory))==digest(_state_payload(job.reference_factory))
        and digest(witness)==digest(public_request_witness(PublicRootRequest(
            state=job.context.public_materialization_state,observation=job.context.observation,
            set_source_hash=job._set_source_hash,pending_transition=job.pending_transition))),
        "archived reference diagnostic original request/factory changed during preparation")
    actual=dict(configuration=_state_payload(job.configuration),checkpoint_type=type(job.checkpoint_contract),
        checkpoint=_state_payload(vars(job.checkpoint_contract)),showdown_root=job.showdown_root,
        context=_state_payload(job.context),pending=_state_payload(job.pending_transition),
        root_id=job.root_id,statistics_root_id=job.statistics_root_id,selection_seed=job.selection_seed,
        runtime_sha256=job.expected_runtime_sha256,public_request_sha256=job.expected_public_request_sha256,
        factory_sha256=digest(_state_payload(job.reference_factory)),
        owned_reference_value_recipe=_state_payload(job.owned_reference_value_recipe))
    # The class identity is an object guard, not JSON to be supplied by a user.
    require(actual.pop("checkpoint_type") is expected.pop("checkpoint_type")
        and digest(actual)==digest(expected) and job.native_ancestry_contract is None
        and digest(job._consumed_binding)==digest(witness["consumed_binding"])
        and job._set_source_hash==witness["consumed_binding"]["set_source_hash"],
        "reference diagnostic returned job differs from original prospective envelope")
    return job,before


def prepare_archived_raw_job(archive, source, *, contract, root_id, configuration,
                             checkpoint_contract, showdown_root, runtime_configuration):
    """Prepare the shared control from an actual exploration archive; no launch."""
    from .search_over_raw_raw_run import RawSelectionJob
    before = archived_exploration_root_binding(archive, source, contract=contract, root_id=root_id)
    root = next(r for r in source["roots"] if r["root_id"] == root_id)
    context, pending, _ = archive.selected(root)
    job = RawSelectionJob(configuration=configuration, checkpoint_contract=checkpoint_contract,
        showdown_root=showdown_root, context=deepcopy(context), pending_transition=deepcopy(pending),
        root_id=root_id+":raw", statistics_root_id=root_id, selection_seed=before["slot"]["source_seed"],
        expected_runtime_configuration=runtime_configuration,
        expected_public_request_sha256=before["public_request_witness"]["public_request_sha256"])
    require(digest(before) == digest(archived_exploration_root_binding(archive, source,
        contract=contract, root_id=root_id)), "archived raw root changed during job preparation")
    return job, before


def prepare_archived_native_timed_job(archive, source, *, contract, root_id, configuration,
                                      checkpoint_contract, showdown_root, runtime_configuration):
    """Public timed job bound to the archived full request, without launch."""
    from .search_over_raw_timed_run import NativeTimedAdapterJob
    before = archived_exploration_root_binding(archive,source,contract=contract,root_id=root_id)
    root = next(r for r in source["roots"] if r["root_id"] == root_id)
    context,pending,_ = archive.selected(root)
    job = NativeTimedAdapterJob(configuration=configuration,checkpoint_contract=checkpoint_contract,
        showdown_root=showdown_root,context=deepcopy(context),pending_transition=deepcopy(pending),
        root_id=root_id+":"+configuration.identity,statistics_root_id=root_id,
        selection_seed=before["slot"]["source_seed"],expected_runtime_configuration=runtime_configuration,
        expected_public_request_sha256=before["public_request_witness"]["public_request_sha256"])
    frozen = job._inputs()
    require(digest(before) == digest(archived_exploration_root_binding(archive,source,
        contract=contract,root_id=root_id)) and job._inputs() == frozen,
        "archived native timed root/job changed during final preparation")
    return job,before


def capture_archived_team_oracle(archive, source, *, contract, root_id, set_source):
    """Original opponent sets only from a bound exploration archive; no launch.

    The private snapshot is retained by the trusted archive and never returned.
    This prepares a diagnostic projection, not a qualified oracle selector.
    """
    from .paper_reference_runtime import PublicRootRequest
    from .search_over_raw_oracle import TeamOracle
    from .search_over_raw_world_inputs import _state_payload,public_request_witness
    from ..randbat import Gen3RandbatSource
    require(type(set_source) is Gen3RandbatSource,"actual pinned oracle reconstruction catalog required")
    catalog_sha = digest(_state_payload(set_source))
    before = archived_exploration_root_binding(archive,source,contract=contract,root_id=root_id)
    root = next(r for r in source["roots"] if r["root_id"] == root_id)
    context,pending,snapshot = archive.selected(root)
    request = PublicRootRequest.capture(context.public_materialization_state,context.observation,
        pending_transition=pending)
    require(digest(public_request_witness(request)) == digest(before["public_request_witness"]),
        "oracle projection differs from actual archived public transport")
    oracle = TeamOracle.capture(snapshot,request,set_source)
    frozen = digest(_state_payload(oracle))
    # Validate again from the actual archive; extraction callbacks cannot replace
    # the earlier root or catalog and establish a new success baseline.
    after = archived_exploration_root_binding(archive,source,contract=contract,root_id=root_id)
    context,pending,snapshot = archive.selected(root)
    actual_request = PublicRootRequest.capture(context.public_materialization_state,context.observation,
        pending_transition=pending)
    actual = TeamOracle.capture(snapshot,actual_request,set_source)
    require(digest(after) == digest(before) and digest(_state_payload(set_source)) == catalog_sha
        and digest(_state_payload(actual)) == frozen and digest(_state_payload(oracle)) == frozen
        and digest(before) == digest(archived_exploration_root_binding(archive,source,
            contract=contract,root_id=root_id)),"oracle archive/catalog/projection changed during extraction")
    require(digest(_state_payload(set_source)) == catalog_sha
        and digest(_state_payload(actual)) == frozen and digest(_state_payload(oracle)) == frozen,
        "oracle catalog/projection changed during final archive validation")
    return deepcopy(oracle),dict(schema="pokezero.search-over-raw.archived-team-oracle.v1",
        root_id=root_id,archived_root_sha256=digest(before),public_request_witness=before["public_request_witness"],
        reconstruction_catalog_sha256=catalog_sha,team_oracle=oracle.receipt(),
        original_team_only=True,source_snapshot_transported=False,committed_opponent_action_transported=False,
        diagnostic_only=True,deployable=False,execution_ready=False,runtime_qualified=False,
        collection_authorized=False,holdout_authorized=False)


def prepare_archived_reference_timed_job(archive, source, *, contract, root_id, configuration,
                                         checkpoint_contract, showdown_root, reference_factory,
                                         runtime_configuration):
    """Full public request for twenty worker preparations; no launch/admission."""
    from .search_over_raw_reference_timed_run import ReferenceTimedAdapterJob
    before = archived_exploration_root_binding(archive,source,contract=contract,root_id=root_id)
    root = next(r for r in source["roots"] if r["root_id"] == root_id)
    context,pending,_ = archive.selected(root)
    job = ReferenceTimedAdapterJob(configuration=configuration,checkpoint_contract=checkpoint_contract,
        showdown_root=showdown_root,reference_factory=reference_factory,
        context=deepcopy(context),pending_transition=deepcopy(pending),
        root_id=root_id+":"+configuration.identity,statistics_root_id=root_id,
        selection_seed=before["slot"]["source_seed"],expected_runtime_configuration=runtime_configuration,
        expected_public_request_sha256=before["public_request_witness"]["public_request_sha256"])
    frozen = job._inputs()
    require(digest(before) == digest(archived_exploration_root_binding(archive,source,
        contract=contract,root_id=root_id)) and job._inputs() == frozen,
        "archived reference timed root/job changed during final preparation")
    return job,before


def prepare_archived_native_oracle_timed_job(archive, source, *, contract, root_id,
        configuration, checkpoint_contract, showdown_root, runtime_configuration, set_source):
    """Join the original-team extraction to a distinct A1 job; no launch/admission."""
    from .search_over_raw_native_oracle_run import NativeOracleTimedAdapterJob,NativeOracleTimedRunContract
    from .paper_reference_runtime import PublicRootRequest
    from .search_over_raw_oracle import opening_team
    from .search_over_raw_world_inputs import _state_payload,public_request_witness
    from ..env import BattleStartOverride
    from ..showdown_fixture import pack_team
    before=archived_exploration_root_binding(archive,source,contract=contract,root_id=root_id)
    oracle,projection=capture_archived_team_oracle(archive,source,contract=contract,
        root_id=root_id,set_source=set_source)
    original_truth=digest(_state_payload(oracle));original_projection=digest(projection)
    root=next(r for r in source["roots"] if r["root_id"] == root_id)
    context,pending,_=archive.selected(root)
    request=PublicRootRequest.capture(context.public_materialization_state,context.observation,pending_transition=pending)
    witness=public_request_witness(request)
    require(projection["archived_root_sha256"] == digest(before)
        and digest(witness) == digest(before["public_request_witness"])
        and digest(witness) == digest(projection["public_request_witness"]),
        "A1 native job differs from original archived team/public extraction")
    own=opening_team(request.state.self_initial_request,oracle.subject,set_source)
    opponent="p2" if oracle.subject == "p1" else "p1"
    override=BattleStartOverride(player_teams={oracle.subject:pack_team(own),opponent:pack_team(oracle.opponent_team)},
        observation_format_id="gen3randombattle")
    expected_override=digest(asdict(override))
    prospective=dict(configuration=asdict(configuration),oracle_payload=_state_payload(oracle),
        oracle_receipt=projection["team_oracle"],override=expected_override,
        catalog=projection["reconstruction_catalog_sha256"],public_request=witness,
        public_request_sha256=witness["public_request_sha256"],root=root_id+":"+configuration.identity,
        statistics_root=root_id,seed=before["slot"]["source_seed"],runtime=runtime_configuration,
        consumed_binding=witness["consumed_binding"],legal_mask=list(request.observation.legal_action_mask),
        player=context.player_id,decision_round=context.decision_round_index,
        timed_contract=NativeOracleTimedRunContract().receipt(),
        showdown=showdown_root,checkpoint_path=getattr(checkpoint_contract,"checkpoint_path",None),
        checkpoint_sha=checkpoint_contract.checkpoint_sha256,source_sha=checkpoint_contract.showdown_source_sha256)
    prospective_sha=digest(prospective)
    require(digest(_state_payload(oracle)) == original_truth and digest(projection) == original_projection,
        "original A1 team projection changed before job construction")
    job=NativeOracleTimedAdapterJob(configuration=configuration,checkpoint_contract=checkpoint_contract,
        showdown_root=showdown_root,context=deepcopy(context),pending_transition=deepcopy(pending),
        root_id=root_id+":"+configuration.identity,statistics_root_id=root_id,
        selection_seed=before["slot"]["source_seed"],expected_runtime_configuration=runtime_configuration,
        expected_public_request_sha256=witness["public_request_sha256"],oracle=oracle,
        expected_oracle_receipt=projection["team_oracle"],expected_override_sha256=expected_override,
        expected_reconstruction_catalog_sha256=projection["reconstruction_catalog_sha256"])
    returned=dict(configuration=asdict(job.configuration),oracle_payload=_state_payload(job.oracle),
        oracle_receipt=job.expected_oracle_receipt,override=job.expected_override_sha256,
        catalog=job.expected_reconstruction_catalog_sha256,
        public_request=public_request_witness(replace(job._public_request,
            state=job.context.public_materialization_state,observation=job.context.observation,
            pending_transition=job.pending_transition)),
        public_request_sha256=job.expected_public_request_sha256,root=job.root_id,
        statistics_root=job.statistics_root_id,seed=job.selection_seed,runtime=job.expected_runtime_configuration,
        consumed_binding=job._consumed_binding,legal_mask=list(job._legal_action_mask),
        player=job.context.player_id,decision_round=job.context.decision_round_index,
        timed_contract=job.contract.receipt(),
        showdown=job.showdown_root,checkpoint_path=getattr(job.checkpoint_contract,"checkpoint_path",None),
        checkpoint_sha=job.checkpoint_contract.checkpoint_sha256,
        source_sha=job.checkpoint_contract.showdown_source_sha256)
    require(digest(returned) == prospective_sha and digest(prospective) == prospective_sha
        and digest(_state_payload(oracle)) == original_truth and digest(projection) == original_projection
        and job._inputs() == job._frozen_inputs,
        "returned A1 job differs from original extracted truth/public/runtime bindings")
    frozen=job._inputs();job._check_source(set_source)
    require(digest(before) == digest(archived_exploration_root_binding(archive,source,
        contract=contract,root_id=root_id)) and job._inputs() == frozen
        and digest(_state_payload(set_source)) == projection["reconstruction_catalog_sha256"]
        and digest(prospective) == prospective_sha and digest(projection) == original_projection
        and digest(_state_payload(oracle)) == original_truth,
        "A1 archived root/job/catalog changed during final preparation")
    return job,dict(schema="pokezero.search-over-raw.archived-native-oracle-timed-job.v1",
        archived_root_binding=before,team_projection=projection,job_inputs_sha256=frozen,
        original_team_only=True,private_snapshot_transported=False,deployable=False,
        runtime_qualified=False,collection_authorized=False,holdout_authorized=False)


def prepare_archived_reference_oracle_timed_job(archive, source, *, contract, root_id,
        configuration, checkpoint_contract, showdown_root, reference_factory,
        runtime_configuration, set_source):
    """Freeze archive-derived A1 truth before construction; no launch/admission."""
    from .search_over_raw_reference_oracle_run import ReferenceOracleTimedAdapterJob,ReferenceOracleTimedRunContract
    from .paper_reference_runtime import PublicRootRequest
    from .search_over_raw_world_inputs import _state_payload,public_request_witness
    before=archived_exploration_root_binding(archive,source,contract=contract,root_id=root_id)
    oracle,projection=capture_archived_team_oracle(archive,source,contract=contract,
        root_id=root_id,set_source=set_source)
    original_truth=digest(_state_payload(oracle));original_projection=digest(projection)
    root=next(r for r in source["roots"] if r["root_id"] == root_id)
    context,pending,_=archive.selected(root)
    request=PublicRootRequest.capture(context.public_materialization_state,context.observation,pending_transition=pending)
    witness=public_request_witness(request)
    require(projection["archived_root_sha256"] == digest(before)
        and digest(witness) == digest(before["public_request_witness"])
        and digest(witness) == digest(projection["public_request_witness"]),
        "A1 reference job differs from original archived team/public extraction")
    def proposed(*,configuration,oracle,receipt,catalog,witness,root,statistics_root,seed,runtime,
            factory,checkpoint,showdown,player,decision_round,timed_contract):
        return dict(configuration=asdict(configuration),oracle_payload=_state_payload(oracle),
            oracle_receipt=receipt,catalog=catalog,public_request=witness,
            public_request_sha256=witness["public_request_sha256"],root=root,statistics_root=statistics_root,
            seed=seed,runtime=runtime,reference_factory=asdict(factory),consumed_binding=witness["consumed_binding"],
            legal_mask=list(request.observation.legal_action_mask),player=player,decision_round=decision_round,
            timed_contract=timed_contract,showdown=showdown,checkpoint_path=checkpoint.checkpoint_path,
            checkpoint_sha=checkpoint.checkpoint_sha256,source_sha=checkpoint.showdown_source_sha256)
    prospective=proposed(configuration=configuration,oracle=oracle,receipt=projection["team_oracle"],
        catalog=projection["reconstruction_catalog_sha256"],witness=witness,root=root_id+":"+configuration.identity,
        statistics_root=root_id,seed=before["slot"]["source_seed"],runtime=runtime_configuration,
        factory=reference_factory,checkpoint=checkpoint_contract,showdown=showdown_root,player=context.player_id,
        decision_round=context.decision_round_index,timed_contract=ReferenceOracleTimedRunContract().receipt())
    prospective_sha=digest(prospective)
    require(digest(_state_payload(oracle)) == original_truth and digest(projection) == original_projection,
        "original reference A1 projection changed before construction")
    job=ReferenceOracleTimedAdapterJob(configuration=configuration,checkpoint_contract=checkpoint_contract,
        showdown_root=showdown_root,reference_factory=reference_factory,
        context=deepcopy(context),pending_transition=deepcopy(pending),
        root_id=root_id+":"+configuration.identity,statistics_root_id=root_id,
        selection_seed=before["slot"]["source_seed"],expected_runtime_configuration=runtime_configuration,
        expected_public_request_sha256=witness["public_request_sha256"],oracle=oracle,
        expected_oracle_receipt=projection["team_oracle"],
        expected_reconstruction_catalog_sha256=projection["reconstruction_catalog_sha256"])
    actual_public=public_request_witness(replace(job._public_request,
        state=job.context.public_materialization_state,observation=job.context.observation,
        pending_transition=job.pending_transition))
    returned=proposed(configuration=job.configuration,oracle=job.oracle,receipt=job.expected_oracle_receipt,
        catalog=job.expected_reconstruction_catalog_sha256,witness=actual_public,root=job.root_id,
        statistics_root=job.statistics_root_id,seed=job.selection_seed,runtime=job.expected_runtime_configuration,
        factory=job.reference_factory,checkpoint=job.checkpoint_contract,showdown=job.showdown_root,
        player=job.context.player_id,decision_round=job.context.decision_round_index,timed_contract=job.contract.receipt())
    require(digest(returned) == prospective_sha and digest(prospective) == prospective_sha
        and digest(_state_payload(oracle)) == original_truth and digest(projection) == original_projection
        and job.expected_public_request_sha256 == witness["public_request_sha256"]
        and digest(job._consumed_binding) == digest(witness["consumed_binding"])
        and tuple(job._legal_action_mask) == tuple(request.observation.legal_action_mask)
        and job._inputs() == job._frozen_binding,
        "returned reference A1 job differs from original extracted truth/public/runtime")
    frozen=job._inputs();job._check_source(set_source)
    require(digest(before) == digest(archived_exploration_root_binding(archive,source,contract=contract,root_id=root_id))
        and job._inputs() == frozen and digest(_state_payload(set_source)) == projection["reconstruction_catalog_sha256"]
        and digest(prospective) == prospective_sha and digest(projection) == original_projection
        and digest(_state_payload(oracle)) == original_truth,
        "reference A1 archive/job/catalog changed during final preparation")
    return job,dict(schema="pokezero.search-over-raw.archived-reference-oracle-timed-job.v1",
        archived_root_binding=before,team_projection=projection,job_inputs_sha256=frozen,
        original_team_only=True,private_snapshot_transported=False,deployable=False,
        runtime_qualified=False,collection_authorized=False,holdout_authorized=False)


def parent_high_water_bytes():
    """Parent-only resident high-water mark; workers retain their own limits."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class SealedComparisonBank:
    """Fixed-roster, same-process auditor storage for staged A1/A2/A3.

    Retains data, never source environments. No persistence, eviction, redraw,
    runtime admission, or public-adapter entry point. A future reviewed producer
    must own/close this bank and use one separately owned same-format audit shell.
    Parent RSS is checked before/after copies and before retrieval. The limit is
    prospective engineering protection, not evidence of acceptable measured use.
    """
    def __init__(self, root_slots, *, parent_memory_limit_bytes, memory_usage=parent_high_water_bytes):
        slots = deepcopy(list(root_slots))
        require(slots and len(slots) <= 200 and len({s["root_id"] for s in slots}) == len(slots)
            and all(s["root_id"] == f"exploration:{s['source_seed']}:{s['root_slot']}" for s in slots),
            "fixed unique exploration slots required")
        require(type(parent_memory_limit_bytes) is int and 0 < parent_memory_limit_bytes <= 8 * 1024**3,
            "explicit parent memory limit at most8GiB required")
        self._slots = {s["root_id"]: s for s in slots}
        self._entries = {}
        self._selected = {}
        self._missing = set()
        self._sources = {}
        self._memory_usage = memory_usage
        self._limit = parent_memory_limit_bytes
        self._state = "CAPTURING"
        self._guard()

    def __reduce_ex__(self, protocol):
        raise TypeError("private comparison bank cannot be persisted")

    def _guard(self):
        require(self._state != "CLOSED", "comparison bank is closed")
        measured = self._memory_usage()
        require(type(measured) is int and measured >= 0, "invalid parent memory measurement")
        if measured > self._limit:
            self.close()
            raise ValueError("parent memory limit exceeded; no eviction or replay")

    def capture(self, archive, root):
        self._guard()
        require(self._state == "CAPTURING" and root["root_id"] in self._slots
            and root["root_id"] not in self._entries, "unregistered, duplicate or sealed root capture")
        slot = self._slots[root["root_id"]]
        require(root["public_record"]["seed"] == slot["source_seed"], "root source seed drift")
        require(self._selected.get(root["root_id"]) == root,
            "capture differs from validated priority-selected source root")
        context, pending, snapshot = archive.selected(root)
        require("snapshot_id" not in snapshot.bridge_snapshot and "battle" in snapshot.bridge_snapshot,
            "bridge-local handle is not a portable auditor snapshot")
        # Copy protects retained data from later callbacks and restore branches.
        # The public root is the only part returned by manifest().
        self._entries[root["root_id"]] = deepcopy((root, context, pending, snapshot))
        self._guard()

    def validate_roster(self, root_slots):
        self._guard()
        rows = list(root_slots)
        require(len(rows) == len(self._slots)
            and {s["root_id"]: s for s in rows} == self._slots,
            "comparison bank differs from executable fixed roster")

    def account_source(self, source, *, contract):
        from .search_over_raw_source import validate_source_receipt
        self._guard()
        require(self._state == "CAPTURING" and source["source_seed"] not in self._sources,
            "source already accounted or bank sealed")
        self.validate_roster(contract["panels"]["exploration"]["root_slots"])
        validate_source_receipt(source, source["source_seed"], contract=contract, panel="exploration")
        self._selected.update({root["root_id"]: deepcopy(root) for root in source["roots"]})
        self._missing.update(source["missing_root_ids"])
        self._sources[source["source_seed"]] = digest(source)
        self._guard()

    def seal(self):
        self._guard()
        require(self._state == "CAPTURING" and set(self._entries) == set(self._selected)
            and set(self._entries).isdisjoint(self._missing)
            and set(self._entries) | self._missing == set(self._slots),
            "same-root barrier requires all fixed slots accounted; no replacement")
        self._state = "SEALED"
        return self.manifest()

    def selected_for_auditor(self, root_id):
        self._guard()
        require(self._state == "SEALED" and root_id in self._entries,
            "sealed registered comparison root required")
        value = deepcopy(self._entries[root_id])
        self._guard()
        return value

    def manifest(self):
        self._guard()
        return dict(schema="pokezero.search-over-raw.comparison-bank.v1", state=self._state,
            capacity=len(self._slots), captured_roots=len(self._entries),
            source_validated_missing_root_ids=sorted(self._missing), source_receipt_bindings=dict(self._sources),
            root_public_bindings={root: entry[0]["public_record_sha256"]
                for root, entry in self._entries.items()},
            parent_memory_limit_bytes=self._limit, parent_high_water_bytes=self._memory_usage(),
            private_snapshot_persisted=False, source_environments_retained=False,
            replay_authorized=False, runtime_authorized=False)

    def close(self):
        self._entries.clear()
        self._selected.clear()
        self._missing.clear()
        self._sources.clear()
        self._state = "CLOSED"
