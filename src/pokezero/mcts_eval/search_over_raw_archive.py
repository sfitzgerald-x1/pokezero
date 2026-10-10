"""Trusted in-memory source boundary archive; never a search input container.

The public policy callback and private pre-step callback have separate types.
Only selected roots can retrieve an auditor snapshot, after binding their
canonical public records. No private snapshot is exported or serialized.
"""
from dataclasses import replace

from .search_over_raw import digest, require


class SealedSourceArchive:
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
        from ..public_decision_corpus import PublicDecisionRecord, PublicObservation
        require(root["public_record_sha256"] == digest(root["public_record"]), "public root digest drift")
        record = PublicDecisionRecord.from_dict(root["public_record"])
        index = root["source_request_index"]
        require(index == record.turn_index and index in self._public and index in self._private,
            "selected source boundary unavailable")
        context, pending, action = self._public[index]
        require((record.seed, record.battle_id, record.acting_player, record.recorded_action_index) ==
                (context.seed, context.battle_id, self.subject, action)
            and record.observation.to_dict() == PublicObservation.from_observation(context.observation).to_dict(),
            "selected canonical public root differs from sealed source")
        return context, pending, self._private[index]

    def close(self):
        self._public.clear()
        self._private.clear()
        self._previous = None
        self._substitute.transition = None
