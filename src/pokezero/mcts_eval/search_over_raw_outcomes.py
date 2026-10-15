"""Prospective continuation capture and independent observed-protocol reader.

The reader derives labels from actual omniscient win/tie lines, not the producer's
TerminalState. It checks the continuation law against observed decisions and
transport acknowledgements. It does not certify simulator physics, model
inference, legal-mask correctness, process provenance or cleanup. A bank can
score only after a separate owning-run provenance/cleanup validator clears its
exact capture. Historical labels cannot acquire evidence through this module.
Private snapshots stay in the outcome auditor; only their digests are exported.
Reused-shell and legacy restore captures cannot score. Prospective captures
require the reviewed fresh clone/strict release lifecycle, independently checked
owning-run provenance and cleanup, not retrospective token tagging.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import random
import uuid

from .search_over_raw import _raw_action, digest, require, rng_seed
from .search_over_raw_world_inputs import _snapshot_payload, _state_payload

SCHEMA = "pokezero.search-over-raw.observed-continuations.v2"
RAW_LABEL_SCHEMA = "pokezero.search-over-raw.observed-raw-labels.v1"
RAW_LABEL_POLICY = "raw_argmax_both_seats_from_step_zero.v1"


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _seed_string(seed):
    # Independent reconstruction of the local bridge's integer seed encoding.
    value = hashlib.sha256(str(seed).encode()).digest()
    return ",".join(str(int.from_bytes(value[i:i+2], "big")) for i in range(0, 8, 2))


def snapshot_identity(snapshot):
    return dict(root_snapshot_sha256=digest(_snapshot_payload(snapshot)),
        bridge_snapshot_sha256=digest(snapshot.bridge_snapshot),
        root_player_lines=[line for line in snapshot.protocol_lines if line.startswith("|player|")])


def _snapshot_json(snapshot):
    return json.dumps(snapshot.bridge_snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _clone_id(law):
    return "continuation-" + digest(law)


def _root_requested(snapshot):
    bridge = snapshot.bridge_snapshot
    require(type(bridge) is dict and set(bridge) == {"battle", "boundaryRequests", "terminalScheduled"}
        and bridge["terminalScheduled"] is False and type(bridge["battle"]) is dict
        and bridge["battle"].get("ended") is not True
        and type(bridge["boundaryRequests"]) is dict, "nonterminal frozen bridge root required")
    requested = []
    for seat in ("p1", "p2"):
        request = bridge["boundaryRequests"].get(seat)
        if request is None:
            continue
        require(type(request) is dict, "frozen request object required")
        if request.get("wait") or request.get("teamPreview"):
            continue
        force = request.get("forceSwitch")
        active = request.get("active")
        if (type(force) is list and any(force)) or (type(active) is list and active):
            requested.append(seat)
    require(requested, "frozen root has no actionable request")
    return requested


class OwnedContinuationLifecycle:
    """Auditor-only opt-in lifecycle, not source generation or process ownership.

    Use an empty LocalShowdownEnv with its original checkpoint-compatible config.
    The bridge may be started without reset. Process/runtime provenance and final
    process cleanup must still be validated independently by the owning run.
    A transport failure permanently poisons this object; no token/cell retry.
    """
    def __init__(self, env, *, expected_snapshot, root_binding):
        from ..local_showdown import LocalShowdownEnv
        require(isinstance(env, LocalShowdownEnv) and env._battle_token is None
            and env._bridge_evidence_observer is None, "empty dedicated outcome environment required")
        require(type(root_binding) is dict and expected_snapshot.terminal is None,
            "bound nonterminal outcome root required")
        self.env, self.snapshot = env, expected_snapshot
        self.binding = deepcopy(root_binding)
        self._original_binding = root_binding
        self._frozen_inputs = self._inputs()
        self.phase, self.poisoned, self.used = "idle", False, set()
        self.token, self.clone_id, self.snapshot_sha, self.observer = None, None, None, None

    def _inputs(self):
        return digest(dict(binding=self._original_binding, snapshot=snapshot_identity(self.snapshot),
            source_battle_token=self.snapshot.battle_token))

    def _check_inputs(self, snapshot=None):
        require(self._inputs() == self._frozen_inputs and (snapshot is None
            or snapshot_identity(snapshot) == snapshot_identity(self.snapshot)
            and snapshot.battle_token == self.snapshot.battle_token), "owned root inputs changed")

    def prepare(self, snapshot, law):
        require(not self.poisoned and self.phase in ("idle", "released")
            and self.env._battle_token is None, "owned lifecycle failed or already active")
        self._check_inputs(snapshot)
        require(type(law) is dict and (set(law) == {"root_id", "action", "replicate", "subject", "namespace", "max_boundaries"}
            or set(law) == {"root_id", "action", "replicate", "subject", "namespace", "max_boundaries", "policy"}
                and law["policy"] == RAW_LABEL_POLICY and law["action"] == 0)
            and law["root_id"] == self.binding["root_id"] and type(law["action"]) is int and law["action"] >= 0
            and type(law["replicate"]) is int and 0 <= law["replicate"] < 8
            and law["subject"] in _root_requested(snapshot)
            and type(law["namespace"]) is str and bool(law["namespace"])
            and type(law["max_boundaries"]) is int and law["max_boundaries"] == 250,
            "frozen observed continuation law required")
        observer = self.env._bridge_evidence_observer
        require(isinstance(observer, ContinuationCapture) and observer.env is self.env
            and observer.active is None and observer.binding == self.binding,
            "installed bound continuation capture required")
        clone_id = _clone_id(law)
        require(clone_id not in self.used, "owned continuation cell retry forbidden")
        self.used.add(clone_id)  # Reserve before startup, send or acknowledgement.
        self.clone_id, self.token = clone_id, "outcome-" + uuid.uuid4().hex
        self.snapshot_sha = hashlib.sha256(_snapshot_json(snapshot).encode()).hexdigest()
        self.observer, self.phase = observer, "prepared"
        try:
            if self.env._process is None:
                self.env._validate_runtime()
                self.env._start_bridge()  # Empty bridge only; never reset/start.
            self.env._battle_token = self.token
        except BaseException:
            self.poisoned, self.phase = True, "failed"
            raise

    def restore(self, snapshot):
        require(self.phase == "prepared" and self.env._battle_token == self.token,
            "prepared owned continuation required")
        self._check_inputs(snapshot)
        require(self.observer.active is not None and _clone_id(self.observer.active["law"]) == self.clone_id,
            "capture must begin before clone submission")
        self.phase = "submitted"
        event = self.env._bridge_request_event(dict(type="clone_snapshot", battleId=self.token,
            cloneId=self.clone_id, snapshotJson=_snapshot_json(snapshot)), "cloned")
        require(event.get("type") == "cloned" and all(event.get(key) == value
            for key, value in self._binding().items()), "clone acknowledgement identity drift")
        self.phase = "live"
        require(event == dict(type="cloned", **self._binding(), requested=_root_requested(snapshot)),
            "clone acknowledgement/root request drift")
        self.env._restore_local_snapshot_state(snapshot)
        require(list(self.env.requested_players()) == event["requested"], "Python/frozen clone request mismatch")

    def _binding(self):
        return dict(battleId=self.token, cloneId=self.clone_id, snapshotJsonSha256=self.snapshot_sha)

    def release(self):
        require(self.phase == "live" and self.env._battle_token == self.token, "live owned clone required")
        self.phase = "releasing"  # No retry even if send/ack/EOF is ambiguous.
        start = len(self.observer.active["records"])
        event = self.env._bridge_request_event(dict(type="release_clone", **self._binding()), "clone_released")
        require(event == dict(type="clone_released", **self._binding()), "strict release acknowledgement drift")
        ends = [record["payload"].get("stream") for record in self.observer.active["records"][start:]
            if record["kind"] == "event" and record["payload"].get("battleId") == self.token
            and record["payload"].get("type") == "stream_end"]
        require(len(ends) == 3 and all(type(seat) is str for seat in ends)
            and sorted(ends) == ["omniscient", "p1", "p2"], "all three distinct owned streams must end before release")
        self.env._battle_token, self.phase = None, "released"
        self._check_inputs()

    def abort(self, primary):
        self.poisoned = True
        try:
            failure = [record["payload"] for record in (self.observer.active or {}).get("records", [])
                if record["kind"] == "event" and record["payload"].get("type") == "clone_failed"]
            if failure:
                require(len(failure) == 1 and all(failure[0].get(key) == value for key, value in self._binding().items())
                    and failure[0].get("cleanupReleased") is True, "unvalidated bridge failure cleanup; owning supervisor required")
                self.env._battle_token, self.phase = None, "released"
            elif self.phase == "live":
                self.release()  # One cleanup attempt for a local/evaluator error.
            elif self.phase not in ("prepared", "released"):
                raise ValueError("ambiguous owned transport failure; owning supervisor required")
            elif self.phase == "prepared":
                self.env._battle_token = None  # No clone command was submitted.
        except BaseException as cleanup:
            if hasattr(primary, "add_note"):
                primary.add_note("Owned cleanup failed: " + str(cleanup))
        self.phase = "failed"


class ContinuationCapture:
    """Observe real continuation/transport seams; default-off and fail closed.

    Install only on the auditor's local environment. Context removal is not a
    process-cleanup receipt. In-flight failed cells cannot be exported as done.
    Transport snapshots/private requests are hashed, never exported wholesale.
    """
    def __init__(self, root_binding, runtime_sha256):
        require(type(root_binding) is dict and _sha(runtime_sha256), "bound outcome runtime required")
        self.binding, self.runtime = deepcopy(root_binding), runtime_sha256
        self.cells, self.active, self.env = {}, None, None

    @contextmanager
    def installed(self, env):
        require(self.env is None and hasattr(env, "_bridge_evidence_observer")
            and env._bridge_evidence_observer is None, "exclusive bridge audit seam required")
        self.env = env
        env._bridge_evidence_observer = self
        try:
            yield self
        finally:
            require(env._bridge_evidence_observer is self, "bridge audit observer changed")
            env._bridge_evidence_observer = None
            self.env = None

    def begin_cell(self, snapshot, **law):
        require(self.env is not None and self.active is None and snapshot.terminal is None,
            "new nonterminal owned continuation required")
        require(law["root_id"] == self.binding["root_id"], "captured root identity drift")
        key = (law["action"], law["replicate"])
        require(type(key[0]) is int and key[0] >= 0 and type(key[1]) is int and 0 <= key[1] < 8
            and key not in self.cells, "continuation retry/duplicate capture forbidden")
        self.active = dict(law=deepcopy(law), battle_token=self.env._battle_token,
            source_battle_token=snapshot.battle_token,
            **snapshot_identity(snapshot), records=[])

    def _record(self, kind, payload):
        require(self.active is not None, "transport outside active continuation")
        self.active["records"].append(dict(kind=kind, payload=deepcopy(payload)))

    def command(self, payload):
        saved = deepcopy(dict(payload))
        if "snapshot" in saved:
            saved["snapshot_sha256"] = digest(saved.pop("snapshot"))
        if "snapshotJson" in saved:
            encoded = saved.pop("snapshotJson")
            require(type(encoded) is str, "received private snapshot JSON bytes required")
            saved["snapshotJsonSha256"] = hashlib.sha256(encoded.encode()).hexdigest()
        self._record("command", saved)

    def event(self, payload):
        saved = deepcopy(dict(payload))
        if "currentSnapshotJson" in saved:
            encoded = saved.pop("currentSnapshotJson")
            require(type(encoded) is str and saved.get("currentSnapshotJsonSha256")
                == hashlib.sha256(encoded.encode()).hexdigest(), "owned boundary snapshot byte drift")
        # Player request streams are private and not required for terminal
        # attribution. Preserve their existence and hash, not their plaintext.
        if saved.get("type") == "stream" and saved.get("stream") != "omniscient":
            saved["lines_sha256"] = digest(saved.pop("lines", None))
        self._record("event", saved)

    def actions(self, actions, choices):
        self._record("actions", dict(actions=dict(actions), choices=dict(choices)))

    def policy_receipt(self, observation, legal, priors):
        return dict(observation_sha256=digest(_state_payload(observation)),
            legal_actions=list(legal), priors=list(priors))

    def boundary(self, step, requested, policies, choices):
        self._record("boundary", dict(step=step, requested=list(requested),
            policies=policies, actions=choices))

    def finish_cell(self, row):
        require(self.active is not None, "missing active continuation")
        law = self.active["law"]
        require(all(row[k] == law[k] for k in ("root_id", "action", "replicate")),
            "captured outcome identity drift")
        self.active["row_sha256"] = digest(row)
        self.cells[law["action"], law["replicate"]] = self.active
        self.active = None

    def export(self):
        require(self.active is None, "failed continuation cannot supply a complete capture")
        return dict(schema=SCHEMA, root_binding=deepcopy(self.binding), runtime_sha256=self.runtime,
            cells=[deepcopy(self.cells[key]) for key in sorted(self.cells)],
            scope="observed_protocol_and_decisions", runtime_qualified=False,
            inference_correctness_established=False, cleanup_validated=False)


def _players(lines):
    require(type(lines) is list and lines, "root player protocol required")
    players = {}
    for line in lines:
        require(type(line) is str, "invalid player protocol")
        parts = line.split("|")
        require(len(parts) >= 4 and parts[:2] == ["", "player"]
            and parts[2] in ("p1", "p2") and parts[3], "invalid root player protocol")
        require(parts[2] not in players or players[parts[2]] == parts[3], "player identity drift")
        players[parts[2]] = parts[3]
    require(set(players) == {"p1", "p2"} and players["p1"] != players["p2"],
        "two distinct known player names required")
    return players


def paired_player_names(snapshot):
    """Actual serializer identity for a paired initializer without player lines.

    This is attribution metadata, never fabricated protocol or policy history.
    A different policy/parser identity cannot supply a signed outcome.
    """
    sides=snapshot.bridge_snapshot.get("battle",{}).get("sides")
    require(type(sides) is list and len(sides)==2
        and all(type(side) is dict and side.get("id")==seat and type(side.get("name")) is str
            and bool(side["name"]) for seat,side in zip(("p1","p2"),sides)),
        "actual paired serialized player identities required")
    names={side["id"]:side["name"] for side in sides}
    require(names["p1"]!=names["p2"] and names==dict(snapshot.replay.players),
        "paired simulator/parser player identities differ")
    lines=[line for line in snapshot.protocol_lines if line.startswith("|player|")]
    require(not lines or _players(lines)==names,"paired retained player protocol conflicts with serializer")
    return names


def _decisions(boundary, law, requested):
    require(type(boundary) is dict and set(boundary) == {"step", "requested", "policies", "actions"}
        and boundary["requested"] == requested and type(boundary["step"]) is int,
        "actual request/decision boundary drift")
    step, policies, actions = boundary["step"], boundary["policies"], boundary["actions"]
    require(type(policies) is dict and type(actions) is dict
        and set(policies) == set(actions) == set(requested), "decision seat roster drift")
    for seat in requested:
        policy = policies[seat]
        require(type(policy) is dict and set(policy) == {"observation_sha256", "legal_actions", "priors"}
            and _sha(policy["observation_sha256"]), "actual policy observation binding required")
        legal, priors = policy["legal_actions"], policy["priors"]
        require(type(legal) is list and type(priors) is list, "retained policy distribution required")
        expected = _raw_action(legal, priors)
        if law.get("policy") == RAW_LABEL_POLICY:
            pass  # Both actors use the independently reconstructed argmax.
        elif step == 0 and seat == law["subject"]:
            require(law["action"] in legal, "initial selected action is illegal")
            expected = law["action"]
        elif step == 0:
            draw = random.Random(rng_seed(law["namespace"], law["root_id"], law["replicate"],
                step, "opponent")).random() * math.fsum(priors)
            cumulative, expected = 0., legal[-1]
            for action, prior in zip(legal, priors):
                cumulative += prior
                if draw < cumulative:
                    expected = action
                    break
        require(type(actions[seat]) is int and actions[seat] == expected,
            "observed decision violates the frozen continuation policy")


def validate_observed_outcome(row, contract, capture, *, expected_snapshot, runtime_sha256):
    """Check exact supplied label against observed protocol; return digest or None.

    Missing terminal evidence is unvalidated, not a synthetic draw. Structural,
    policy, identity, acknowledgement or conflicting-terminal drift fails the
    attempt. The external snapshot/runtime bindings must come from the owning
    run, not from the capture being checked. Does NOT validate run cleanup.
    """
    fields={"root_binding", "root_id", "namespace", "subject",
        "max_boundaries", "replicates", "selection_seed", "policy"}
    require(type(contract) is dict and (set(contract)==fields or set(contract)==fields|{"paired_player_names"}
        and contract["policy"]==RAW_LABEL_POLICY)
        and type(contract["max_boundaries"]) is int and contract["max_boundaries"] == 250
        and type(contract["replicates"]) is int and contract["replicates"] == 8
        and contract["subject"] in ("p1", "p2") and type(contract["namespace"]) is str
        and contract["namespace"] and contract["policy"] in {
            "raw_argmax_after_initial_sampled_opponent_reply", RAW_LABEL_POLICY}
        and type(contract["root_binding"]) is dict
        and contract["root_binding"]["root_id"] == contract["root_id"]
        and type(contract["selection_seed"]) is int
        and type(contract["root_binding"]["source_seed"]) is int
        and contract["selection_seed"] == contract["root_binding"]["source_seed"], "frozen continuation law required")
    require(type(row) is dict and set(row) == {
        "root_id", "action", "replicate", "status", "signed_outcome", "boundaries"}, "exact outcome row required")
    require(type(row["action"]) is int and row["action"] >= 0 and type(row["replicate"]) is int
        and 0 <= row["replicate"] < 8 and type(row["boundaries"]) is int
        and 0 <= row["boundaries"] <= contract["max_boundaries"], "invalid continuation cell")
    require(row["status"] in {"COMPLETE", "CAPPED", "REFUSED", "UNSTARTED"}
        and ((row["status"] == "COMPLETE" and type(row["signed_outcome"]) is int
              and row["signed_outcome"] in (-1, 0, 1))
             or (row["status"] != "COMPLETE" and row["signed_outcome"] is None)), "invalid outcome status/label")
    raw_label = contract["policy"] == RAW_LABEL_POLICY
    if capture is None or row["status"] != "COMPLETE" and not (raw_label and row["status"] == "CAPPED"):
        return None
    require(type(capture) is dict and set(capture) == {"law", "battle_token", "source_battle_token", "root_snapshot_sha256",
        "bridge_snapshot_sha256", "root_player_lines", "records", "row_sha256"}, "exact protocol capture required")
    law = dict(root_id=contract["root_id"], namespace=contract["namespace"], subject=contract["subject"],
        max_boundaries=contract["max_boundaries"], action=row["action"], replicate=row["replicate"])
    if raw_label:
        require(row["action"] == 0, "raw label cell has no forced action")
        law["policy"] = RAW_LABEL_POLICY
    require(digest(capture["law"]) == digest(law) and row["root_id"] == law["root_id"]
        and capture["row_sha256"] == digest(row) and _sha(runtime_sha256), "outcome/law/runtime binding drift")
    identity = snapshot_identity(expected_snapshot)
    require(expected_snapshot.terminal is None and not any(line.startswith("|win|") or line == "|tie"
        or line.startswith("|tie|") for line in expected_snapshot.protocol_lines), "nonterminal source root required")
    require(all(capture[k] == v for k, v in identity.items()), "restored root snapshot drift")
    if "paired_player_names" in contract:
        players=paired_player_names(expected_snapshot)
        require(digest(contract["paired_player_names"])==digest(players),"paired attribution contract drift")
    else:
        players = _players(capture["root_player_lines"])
    token = capture["battle_token"]
    require(type(token) is str and token and capture["source_battle_token"] == expected_snapshot.battle_token,
        "actual battle routing token/source identity required")
    records = capture["records"]
    require(type(records) is list and records, "actual transport records required")
    if not any(record.get("kind") == "command" and record.get("payload", {}).get("type") == "clone_snapshot"
        for record in records if type(record) is dict and type(record.get("payload")) is dict):
        return None  # Historical restore/token assertions cannot acquire a lifecycle.
    cloned_binding = dict(battleId=token, cloneId=_clone_id(law),
        snapshotJsonSha256=identity["bridge_snapshot_sha256"])
    require(token != capture["source_battle_token"], "source shell cannot become an owned outcome clone")
    root_requested = _root_requested(expected_snapshot)
    phase, requested, step, decisions, sent, acknowledgements = "clone", None, 0, None, None, set()
    ended_streams = set()
    terminal_labels, terminal_boundary = [], False
    for record in records:
        require(type(record) is dict and set(record) == {"kind", "payload"}
            and type(record["payload"]) is dict, "malformed transport record")
        kind, payload = record["kind"], record["payload"]
        if kind == "boundary":
            require(phase == "boundary" and payload.get("step") == step
                and step < law["max_boundaries"], "missing/duplicate/out-of-order policy boundary")
            _decisions(payload, law, requested)
            decisions, phase = payload, "reseed"
        elif kind == "actions":
            require(phase == "actions" and set(payload) == {"actions", "choices"}
                and digest(payload["actions"]) == digest(decisions["actions"])
                and type(payload["choices"]) is dict and set(payload["choices"]) == set(requested)
                and all(type(c) is str and c.strip() for c in payload["choices"].values()),
                "actual action-index/transport translation drift")
            sent, phase = payload["choices"], "choices"
        elif kind == "command":
            require(payload.get("battleId") == token, "command battle identity drift")
            if phase == "clone":
                require(payload == dict(type="clone_snapshot", **cloned_binding),
                    "clone must consume the bound snapshot bytes and cell identity")
                phase = "cloned"
            elif phase == "reseed":
                seed = _seed_string(rng_seed(law["namespace"], law["root_id"], law["replicate"], step, "chance"))
                require(payload == dict(type="reseed", battleId=token, seed=seed), "chance seed drift")
                phase = "reseeded"
            elif phase == "choices":
                require(payload == dict(type="choices", battleId=token, choices=sent), "sent choices drift")
                acknowledgements, phase = set(), "result"
            elif phase == "terminal" or raw_label and phase == "boundary" and step == law["max_boundaries"]:
                require(payload == dict(type="release_clone", **cloned_binding), "strict clone release command required")
                phase = "releasing"
            else:
                raise ValueError("unexpected continuation command")
        elif kind == "event":
            # Warm-process stale routing is observed, but cannot supply terminal
            # facts or acknowledgements for the active restored cell.
            if payload.get("battleId") not in (None, token):
                continue
            event_type = payload.get("type")
            require(event_type != "error", "simulator error cannot supply an accepted outcome")
            require(payload.get("battleId") == token, "unrouted simulator event")
            require(phase != "done", "own-shell event after strict release acknowledgement")
            if event_type == "cloned":
                require(phase == "cloned" and payload == dict(type="cloned", **cloned_binding,
                    requested=root_requested), "clone acknowledgement/root request drift")
                requested = payload.get("requested")
                require(type(requested) is list and requested and len(set(requested)) == len(requested)
                    and set(requested) <= {"p1", "p2"} and law["subject"] in requested, "invalid restored request")
                phase = "boundary"
            elif event_type == "reseeded":
                require(phase == "reseeded" and payload.get("seed") == seed, "reseed acknowledgement drift")
                phase = "actions"
            elif event_type == "choice_ack":
                seat = payload.get("player")
                require(phase == "result" and seat in sent and seat not in acknowledgements
                    and payload.get("choice") == sent[seat], "choice acknowledgement drift")
                acknowledgements.add(seat)
            elif event_type == "stream":
                if payload.get("stream") != "omniscient":
                    require(_sha(payload.get("lines_sha256")), "private stream digest required")
                    continue
                lines = payload.get("lines")
                require(type(lines) is list and all(type(line) is str for line in lines), "invalid omniscient protocol")
                for line in lines:
                    require(not line.startswith("|error|"), "protocol error invalidates outcome")
                    require(not line.startswith("|player|"), "player identity changed after restored root")
                    if line.startswith("|win|") or line == "|tie" or line.startswith("|tie|"):
                        require(phase == "result", "terminal line outside an actual step")
                        if line.startswith("|win|"):
                            name = line.split("|", 2)[2]
                            require(name in players.values(), "unknown winner cannot become a draw")
                            winner = next(seat for seat in players if players[seat] == name)
                            terminal_labels.append(1 if winner == law["subject"] else -1)
                        else:
                            terminal_labels.append(0)
                        require(len(terminal_labels) == 1, "duplicate/conflicting terminal protocol")
            elif event_type in {"ready", "terminal"}:
                require(phase == "result" and acknowledgements == set(sent), "boundary before all choice acknowledgements")
                step += 1
                if event_type == "terminal":
                    terminal_boundary, phase = True, "terminal"
                else:
                    require(not terminal_labels, "terminal protocol followed by a nonterminal boundary")
                    requested = payload.get("requested")
                    require(type(requested) is list and requested and len(set(requested)) == len(requested)
                        and set(requested) <= {"p1", "p2"}, "invalid next request")
                    phase = "boundary"
            elif event_type == "stream_end":
                stream = payload.get("stream")
                require(phase == "releasing" and type(stream) is str and stream in {"omniscient", "p1", "p2"}
                    and stream not in ended_streams, "unexpected or duplicate owned stream ending")
                ended_streams.add(stream)
            elif event_type == "clone_released":
                require(phase == "releasing" and ended_streams == {"omniscient", "p1", "p2"}
                    and payload == dict(type="clone_released", **cloned_binding), "unvalidated strict release acknowledgement")
                phase = "done"
            else:
                raise ValueError("unrecognized continuation event")
        else:
            raise ValueError("unrecognized continuation record")
    capped = raw_label and row["status"] == "CAPPED" and step == law["max_boundaries"] and not terminal_boundary
    require(phase == "done" and (terminal_boundary or capped) and row["boundaries"] == step,
        "complete outcome lacks its terminal boundary and strict release")
    if capped:
        require(not terminal_labels and row["signed_outcome"] is None, "capped raw label cannot hide a terminal")
        return digest(row)
    if not terminal_labels:
        return None
    require(row["signed_outcome"] == terminal_labels[0], "supplied outcome contradicts simulator protocol")
    return digest(row)


class OutcomeReceiptValidator:
    """Reconciliation callback with mandatory separate provenance/cleanup check.

    A callable returning the exact bank digest must independently verify the
    owning run's runtime/process/capture provenance, freshly isolated per-cell
    shell ownership and cleanup. None leaves all
    labels uncertain. A reader alone or a supplied success flag cannot score.
    """
    def __init__(self, bank, *, expected_snapshot, root_binding, runtime_sha256,
                 validate_provenance_and_cleanup):
        require(callable(validate_provenance_and_cleanup), "independent owning-run validator required")
        original_binding = deepcopy(root_binding)
        original_inputs = digest(dict(binding=original_binding, snapshot=snapshot_identity(expected_snapshot),
            source_battle_token=expected_snapshot.battle_token, runtime=runtime_sha256))
        require(type(bank) is dict and set(bank) == {"schema", "root_binding", "runtime_sha256", "cells",
            "scope", "runtime_qualified", "inference_correctness_established", "cleanup_validated"}
            and bank["schema"] == SCHEMA and digest(bank["root_binding"]) == digest(original_binding)
            and bank["runtime_sha256"] == runtime_sha256 and _sha(runtime_sha256)
            and bank["scope"] == "observed_protocol_and_decisions"
            and all(bank[k] is False for k in ("runtime_qualified", "inference_correctness_established", "cleanup_validated")),
            "exact unpromoted outcome capture/runtime binding required")
        frozen = deepcopy(bank)
        before = digest(frozen)
        verified = validate_provenance_and_cleanup(deepcopy(frozen))
        require(verified is None or verified == before, "owning-run validation must bind the exact capture")
        require(digest(bank) == before and digest(dict(binding=root_binding,
            snapshot=snapshot_identity(expected_snapshot), source_battle_token=expected_snapshot.battle_token,
            runtime=runtime_sha256)) == original_inputs, "owning-run callback changed capture or external inputs")
        require(type(frozen["cells"]) is list, "capture cell bank required")
        self.cells = {}
        for capture in frozen["cells"]:
            key = (capture["law"]["action"], capture["law"]["replicate"])
            require(type(key[0]) is int and key[0] >= 0 and type(key[1]) is int and 0 <= key[1] < 8
                and key not in self.cells and capture["law"]["root_id"] == original_binding["root_id"],
                "duplicate or unbound captured continuation")
            self.cells[key] = capture
        self.validated = verified is not None
        self.identity = original_binding
        self.snapshot, self.runtime = expected_snapshot, runtime_sha256
        self._approved_cells = self._cells_digest()
        self._approved_inputs = original_inputs
        # A bridge token is not a restore generation. Refuse to upgrade warm
        # reuse through even a generic successful owning-run callback.
        tokens = [cell["battle_token"] for cell in self.cells.values()]
        self._isolated_cells = len(tokens) == len(set(tokens)) and all(
            token != expected_snapshot.battle_token for token in tokens)

    def _cells_digest(self):
        return digest([[list(key), self.cells[key]] for key in sorted(self.cells)])

    def _inputs_digest(self):
        return digest(dict(binding=self.identity, snapshot=snapshot_identity(self.snapshot),
            source_battle_token=self.snapshot.battle_token, runtime=self.runtime))

    def __call__(self, row, contract):
        require(self._cells_digest() == self._approved_cells and self._inputs_digest() == self._approved_inputs,
            "approved outcome capture/input evidence changed")
        require(contract["root_binding"] == self.identity, "root binding changed after owning-run validation")
        witness = validate_observed_outcome(row, contract, self.cells.get((row["action"], row["replicate"])),
            expected_snapshot=self.snapshot, runtime_sha256=self.runtime)
        require(self._cells_digest() == self._approved_cells and self._inputs_digest() == self._approved_inputs,
            "approved outcome capture/input evidence changed during validation")
        return witness if self.validated and self._isolated_cells else None
