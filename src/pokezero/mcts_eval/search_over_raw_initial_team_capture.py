"""Passive public-only A1 direct-check integration; no entry or launch authority.

Search receives no truth. Existing callbacks copy its actual attempted initial
teams. The already-owned pre-step boundary subsequently supplies the source
join. A durable reader must still validate the original source transport and
runtime before interpreting these local receipts. No continuation labels,
posterior calibration, current-state fidelity or playing-strength claim.
"""
from copy import deepcopy
import hashlib
import json

from ..root_prior_correspondence import boundary_fingerprint
from .search_over_raw_information_worlds import OneWorldTeamCapture
from .search_over_raw_initial_teams import compare_initial_teams


def digest(value):
    """Strict canonical JSON identity, not operator authentication."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class InitialTeamSourceCheck:
    """Instrument exactly one public production call, restoring both hooks.

    ``publish(kind, round, payload)`` must be a create-only durable sink outside
    live-selections. Attempts publish even if search fails before an owned
    pre-step boundary arrives; they cannot silently become accepted matches.
    """

    def __init__(self, *, source_seed, publish, boundary_key=boundary_fingerprint):
        require(type(source_seed) is int and 0 <= source_seed < 2**32, "source seed required")
        self.seed, self.publish, self.boundary_key = source_seed, publish, boundary_key
        self.attempts, self.rows, self.raw_actions = {}, {}, {}
        self.truth = None

    def select(self, policy, context, *, rng, raw_action):
        index = context.decision_round_index
        require(context.seed == self.seed and context.player_id == "p1"
                and type(index) is int and index >= 0 and index not in self.attempts,
                "original public decision; no retry required")
        require(policy._fixed_override is None and policy._world_attempt_observer is None
                and policy._world_observer is None, "public truth/observer boundary required")
        capture = OneWorldTeamCapture(context, policy._config, boundary_key=self.boundary_key)
        self.raw_actions[index] = raw_action
        row = dict(binding=deepcopy(capture.binding), status="UNAVAILABLE", capture=None,
                   errors=[], raw_action=raw_action)
        self.attempts[index] = row  # Reserve before any production work.
        names = ("worlds_attempted", "worlds_constructed", "worlds_searched", "total_iterations")
        before = [getattr(policy.stats, name) for name in names]
        require(all(type(n) is int and n >= 0 for n in before), "original production counters required")
        original = (policy._world_attempt_observer, policy._world_observer)

        def observer(method):
            def observe(*args):
                try:
                    method(*args)
                except Exception as error:
                    row["errors"].append(type(error).__name__)
                    raise  # Production owns telemetry exception handling.
            return observe

        policy._world_attempt_observer = observer(capture.attempt)
        policy._world_observer = observer(capture.world)
        returned = False
        try:
            selected = policy.select_action_with_context(context, rng=rng)
            returned = True
            return selected
        finally:
            policy._world_attempt_observer, policy._world_observer = original
            row["observed_attempts"] = deepcopy(capture.attempts)
            row["observed_constructed"] = deepcopy(capture.constructed)
            try:
                after = [getattr(policy.stats, name) for name in names]
                require(all(type(n) is int and n >= b for n,b in zip(after,before)),
                        "production counters changed type or decreased")
                deltas = dict(zip(("attempts", "constructed", "searched", "iterations"),
                                  [n-b for n,b in zip(after,before)]))
                row["work"] = deltas
                require(returned and not row["errors"] and policy._fixed_override is None,
                        "successful original public selection and complete observers required")
                row["capture"] = capture.finish(context, policy._config, **deltas)
                row["status"] = "CAPTURED"
            except Exception as error:
                row["errors"].append(type(error).__name__)
            self.publish("attempt", index, deepcopy(row))

    def capture_private(self, boundary):
        from ..golden_corpus import _true_teams_from_bridge_snapshot
        from ..rollout import RolloutSealedPreStepBoundary
        require(type(boundary) is RolloutSealedPreStepBoundary and boundary.seed == self.seed,
                "owned original source pre-step boundary required")
        index = boundary.decision_round_index
        if "p1" not in boundary.requested_players:
            return
        require(index in self.attempts and index not in self.rows,
                "one original pre-step join per public selection required")
        attempt = self.attempts[index]
        require(boundary.battle_id == attempt["binding"]["battle_id"]
                and boundary.decisions["p1"].action_index == self.raw_actions[index],
                "source identity/raw action changed")
        bridge = boundary.snapshot.bridge_snapshot
        require(type(bridge) is dict, "original bridge snapshot required")
        if self.truth is None:
            require(index == 0, "truth must bind at original opening, never a later roster")
            teams = _true_teams_from_bridge_snapshot(bridge)
            self.truth = dict(source_seed=self.seed, battle_id=boundary.battle_id,
                opening_snapshot=deepcopy(bridge), opening_snapshot_sha256=digest(bridge),
                initial_packed_teams={seat:teams[seat]["packed"] for seat in ("p1","p2")})
            self.publish("truth", 0, deepcopy(self.truth))
        require(boundary.battle_id == self.truth["battle_id"], "original truth battle changed")
        row = dict(binding=deepcopy(attempt["binding"]), raw_action=self.raw_actions[index],
            source_snapshot_sha256=digest(bridge), truth_sha256=digest(self.truth),
            attempt_sha256=digest(attempt), status="UNAVAILABLE", comparisons={},
            opponent_primary="p2", own_positive_control="p1", sides_pooled=False,
            original_source_transport_validated=False, current_materialization_compared=False,
            posterior_calibration_established=False, scientific_strength_claim=False)
        if attempt["status"] == "CAPTURED":
            captured = attempt["capture"]
            ordinal = captured["constructed"][0]["attempt_ordinal"]
            sampled = captured["attempts"][ordinal]["initial_packed_teams"]
            for seat in ("p2", "p1"):
                try:
                    row["comparisons"][seat] = compare_initial_teams(
                        sampled[seat], self.truth["initial_packed_teams"][seat])
                except ValueError:
                    row["comparisons"][seat] = None  # Full fixed roster remains unknown.
            if all(value is not None for value in row["comparisons"].values()):
                row["status"] = "COMPARED"
        self.rows[index] = row
        self.publish("comparison", index, deepcopy(row))

    def manifest(self):
        require(self.truth is not None and self.attempts.keys() == self.rows.keys(),
                "complete source boundary joins required")
        return dict(schema="pokezero.search-over-raw.initial-team-source-check.v1",
            truth_sha256=digest(self.truth), source_seed=self.seed,
            invocations=[dict(round=i, attempt_sha256=digest(self.attempts[i]),
                              comparison_sha256=digest(self.rows[i])) for i in sorted(self.rows)],
            source_play_uses_search=False, source_truth_sent_to_public=False,
            original_source_transport_validated=False)
