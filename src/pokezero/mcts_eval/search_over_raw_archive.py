"""Trusted in-memory source boundary archive; never a search input container.

The public policy callback and private pre-step callback have separate types.
Only selected roots can retrieve an auditor snapshot, after binding their
canonical public records. No private snapshot is exported or serialized.
"""
from copy import deepcopy
from dataclasses import replace
import resource
import sys

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
        from ..public_decision_corpus import (
            PublicDecisionRecord, PublicObservation, PublicActorObservation, _public_belief_view)
        require(root["public_record_sha256"] == digest(root["public_record"]), "public root digest drift")
        record = PublicDecisionRecord.from_dict(root["public_record"])
        index = root["source_request_index"]
        require(index == record.turn_index and index in self._public and index in self._private,
            "selected source boundary unavailable")
        context, pending, action = self._public[index]
        require((record.seed, record.battle_id, record.acting_player, record.recorded_action_index) ==
                (context.seed, context.battle_id, self.subject, action)
            and record.format_id == context.format_id
            and record.observation.to_dict() == PublicObservation.from_observation(context.observation).to_dict(),
            "selected canonical public root differs from sealed source")
        history = [PublicActorObservation(step.turn_index, PublicObservation.from_observation(step.observation)).to_dict()
            for step in context.trajectory.steps]
        require([row.to_dict() for row in record.history] == history
            and record.public_belief_view == _public_belief_view(context.observation.metadata)
            and [row.to_dict() for row in record.public_resolved_action_rounds] ==
                context.trajectory.metadata.get("public_resolved_action_rounds", []),
            "selected public history/belief differs from sealed source")
        return context, pending, self._private[index]

    def close(self):
        self._public.clear()
        self._private.clear()
        self._previous = None
        self._substitute.transition = None


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
