"""One-shot raw sources inside the trusted exploration auditor.

No CLI, allocation, registration, retry or holdout entry. Archives and source
trajectories remain in the current owned process; only hashes/public partitions
are persisted. Production calls require separate reviewed exploration authority.
An explicit authored-component route is unadmitted and cannot qualify a runtime.
"""
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import threading
import time

from .search_over_raw import digest, require
from .search_over_raw_archive import (SealedSourceArchive, archived_exploration_root_binding,
    retained_archive_root_identity)
from .search_over_raw_continuation_admission import _path, _fresh_review, NAMESPACE_SCHEMA, _FLAGS
from .search_over_raw_ledger import _load, _write_new
from .search_over_raw_nested_ownership import current_scope
from .search_over_raw_outcome_run import ContinuationProcessOwner, verify_outcome_runtime
from .search_over_raw_outcomes import _sha
from .search_over_raw_reconciled import ReconciledPhaseALedger, controller_input_hashes
from .search_over_raw_source import AuditedRawPolicy, collect_raw_source, validate_source_receipt
from .search_over_raw_world_inputs import _state_payload


def source_recipe(config, checkpoint_contract, bridge_runtime, *, capture_actor_history=False):
    """Reviewable recipe only; does not load a policy or reset an environment."""
    from .resolver import CheckpointContract
    from ..local_showdown import LocalShowdownConfig
    require(type(config) is LocalShowdownConfig and type(checkpoint_contract) is CheckpointContract
        and checkpoint_contract.model_device == "cpu"
        and str(config.resolved_showdown_root().resolve()) == checkpoint_contract.showdown_root,
        "exact CPU source/checkpoint/environment recipe required")
    verify_outcome_runtime(bridge_runtime, config)
    require(bridge_runtime["checkpoint_path"] == checkpoint_contract.checkpoint_path
        and bridge_runtime["files"][checkpoint_contract.checkpoint_path] == checkpoint_contract.checkpoint_sha256,
        "source checkpoint bytes differ from bridge inventory")
    for path,sha in ((checkpoint_contract.model_path,checkpoint_contract.model_sha256),
                     (checkpoint_contract.tables_path,checkpoint_contract.tables_sha256)):
        require(path is None and sha is None or bridge_runtime["files"].get(path) == sha,
            "source exported checkpoint inputs omitted")
    require(type(capture_actor_history) is bool, "explicit source actor capture option required")
    result = dict(schema="pokezero.search-over-raw.source-recipe.v1",
        config_sha256=digest(_state_payload(config)), checkpoint=checkpoint_contract.to_manifest(),
        bridge_runtime_sha256=digest(bridge_runtime), format_id="gen3randombattle", max_decision_rounds=250,
        policy=dict(device="cpu",deterministic=True,exploration_epsilon=0.,sampling_temperature=1.,
            family_gated_selection=False,history_mask_k=None,window_size=1,forward="original_default",
            record_policy_distribution=True), source_policy="raw_argmax_both_seats",
        archive_subject="p1", retries_permitted=False, holdout_authorized=False,
        runtime_qualified=False, collection_authorized=False)
    if capture_actor_history:
        result["actor_history_capture"] = dict(actor="p1", actual_seat_stream=True,
            actual_observation_and_raw_input=True, private_retention=True,
            original_observation_origin_retained=True,
            encoding_mismatch_is_unavailable=True, policy_encoding_changed=False)
    return result


def source_review_template(contract, recipe, *, qualification_evidence_hashes):
    """Prospective source gate; success flags are a separate review decision."""
    from .search_over_raw_reconciled import validate_contract
    validate_contract(contract)
    require(type(qualification_evidence_hashes) is dict and qualification_evidence_hashes
        and all(type(p) is str and str(_path(p)) == p and type(sha) is str and len(sha) == 64
            and hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
            for p,sha in qualification_evidence_hashes.items()), "separate source qualification evidence required")
    code = controller_input_hashes()
    require(not set(code) & set(qualification_evidence_hashes), "source code is not runtime qualification")
    core = {k:v for k,v in contract.items() if k not in {
        "input_hashes","execution_ready","collection_authorized","remaining_gates"}}
    return dict(schema="pokezero.search-over-raw.source-review.v1",status="UNREVIEWED_SOURCE_RECIPE",
        contract_recipe_sha256=digest(core), recipe=deepcopy(recipe), controller_input_hashes=code,
        qualification_evidence_hashes=deepcopy(qualification_evidence_hashes),
        reviewed_before_first_source_work=False, source_runtime_and_cleanup_reviewed=False,
        source_policy_and_terminal_reviewed=False, cost_scope="source-owned components only; not full exploration cost",
        collection_authorized=False, holdout_authorized=False, retry_authorized=False)


def _source_trace(kind, payload):
    # Do not copy arbitrary request, team, choice, snapshot or error strings.
    # Bind whole bytes and retain only identity/terminal/control fields needed
    # by the source reader. Transport provenance is not simulator physics.
    value = dict(kind=kind, payload_sha256=digest(dict(payload)), type=payload.get("type"))
    for key in ("battleId","formatid","seed","stream","player"):
        if key in payload: value[key] = deepcopy(payload[key])
    if kind=="command" and payload.get("type")=="choices":
        choices=payload.get("choices")
        require(type(choices) is dict and choices and set(choices)<={"p1","p2"},
            "source choice seat roster malformed")
        value["players"]=sorted(choices)
    if kind == "event" and payload.get("type") == "snapshot_actionable_boundary":
        require(type(payload.get("available")) is bool, "source snapshot availability malformed")
        value["available"] = payload["available"]
        if payload["available"]:
            requested, snapshot = payload.get("requested"), payload.get("snapshot")
            require(type(requested) is list and requested and requested == sorted(set(requested))
                and set(requested) <= {"p1", "p2"} and type(snapshot) is dict,
                "source snapshot response malformed")
            value.update(requested=deepcopy(requested), snapshot_sha256=digest(snapshot))
    if kind == "event" and payload.get("type") == "stream" and payload.get("stream") == "omniscient":
        lines = payload.get("lines")
        require(type(lines) is list and all(type(line) is str for line in lines), "source protocol lines malformed")
        markers=[line for line in lines if line.startswith("|win|") or line == "|tie|"]
        require(all(line in {"|win|PokeZero p1","|win|PokeZero p2","|tie|"} for line in markers),
            "unknown source terminal identity")
        value["terminal_markers"] = markers
    return value


class SourceProcessOwner(ContinuationProcessOwner):
    """Strict one-source bridge lifecycle; reuse only wait/pipe machinery."""
    launch_schema = "pokezero.search-over-raw.source-bridge-launch.v1"
    close_schema = "pokezero.search-over-raw.source-bridge-close.v1"
    capture_hash_key = "source_sha256"

    def __init__(self, *, source_seed, actor_history=None, **kwargs):
        super().__init__(**kwargs)
        require(type(source_seed) is int and 0 <= source_seed < 2**32, "exact fixed source seed required")
        self.seed, self.reset_started, self.token = source_seed, False, None
        from .search_over_raw_source_actor_history import SourceActorHistoryCapture
        require(actor_history is None or type(actor_history) is SourceActorHistoryCapture,
            "exact private prospective actor capture required")
        self.actor_history = actor_history

    @contextmanager
    def installed(self, env):
        with super().installed(env):
            if self.actor_history is not None:
                self.actor_history.bind(self, env, self.records)
            yield self

    def actor_observation(self, env, actor, observation):
        if self.actor_history is not None:
            self.actor_history.observation(env, actor, observation)

    def begin_owned_reset(self, env, seed, format_id, start_override):
        require(env is self.env and self.phase == "empty" and not self.reset_started
            and self.process is None and env._process is None and env._battle_token is None
            and type(seed) is int and seed == self.seed and format_id == "gen3randombattle"
            and start_override is None, "one exact fresh raw source reset required; no restart")
        self.reset_started = True

    def command(self, payload):
        require(self.phase in {"started","closing"}, "source command outside owning run")
        require(payload.get("type") in {"start","choices","snapshot_actionable_boundary","end","close"},
            "non-source bridge command forbidden")
        if payload["type"] == "start":
            from ..local_showdown import showdown_seed_from_int
            require(self.token is None and self.reset_started and payload.get("seed") == showdown_seed_from_int(self.seed)
                and payload.get("formatid") == "gen3randombattle"
                and payload.get("players") == {"p1":"PokeZero p1","p2":"PokeZero p2"},
                "source start differs from fixed seed/format/raw teams")
            self.token = payload["battleId"]
        elif payload["type"] != "close":
            require(self.token is not None and payload.get("battleId") == self.token, "source command token drift")
        self.records.append(_source_trace("command", payload))

    def event(self, payload):
        require(self.phase in {"started","closing"}, "source event outside owning run")
        if payload.get("type") != "closed":
            require(self.token is not None and payload.get("battleId") == self.token, "source event token drift")
        record = _source_trace("event", payload)
        if payload.get("type") == "snapshot_actionable_boundary":
            # Observed here, before the rollout's private pre-step callback and
            # before committing this boundary's choices. Never supplied by it.
            record.update(record_index=len(self.records), decision_round_index=sum(
                r["kind"] == "command" and r["type"] == "choices" for r in self.records))
        self.records.append(record)
        if self.actor_history is not None:
            self.actor_history.event(payload, len(self.records) - 1)

    def seal(self, source):
        require(self.phase == "started" and self.bank_sha is None and self.token == self.env._battle_token
            and self.env._bridge_evidence_observer is None, "one actual source capture required")
        _match_source_trace(source,self.records,self.seed)
        if self.actor_history is not None:
            self.actor_history.seal()
        self.bank_sha = digest(source)

    def _require_close_ready(self, env):
        require(self.bank_sha is not None and env._battle_token == self.token and self.token is not None
            and env._bridge_evidence_observer is None, "sealed source before strict close required")

    def _normal_close_ack(self, env):
        # End this exact live or terminal stream before closing the bridge.
        env._send_command(dict(type="end",battleId=self.token))
        deadline = time.monotonic() + min(env.config.read_timeout_seconds,10.)
        while True:
            require(time.monotonic() < deadline, "source end acknowledgement timed out")
            event = env._read_event(timeout=max(.001,deadline-time.monotonic()))
            if event is None: continue
            if event.get("type") == "ended":
                require(event == dict(type="ended",battleId=self.token), "source end acknowledgement drift")
                break
            require(event.get("type") == "stream_end" and event.get("battleId") == self.token,
                "unexpected source teardown output")
        ends = [r.get("stream") for r in self.records if r["kind"] == "event" and r["type"] == "stream_end"]
        require(sorted(ends) == ["omniscient","p1","p2"], "all three source streams must end exactly once")
        env._battle_token = None
        super()._normal_close_ack(env)

    def close(self, env):
        # LocalShowdown's initial reset closes its empty shell. It is not a
        # second lifecycle nor evidence of a completed source.
        if self.phase == "empty" and self.process is None and env._process is None:
            require(env is self.env and self.reset_started, "only initial source shell close is allowed")
            return
        super().close(env)


def _match_source_trace(source, records, seed):
    from ..local_showdown import showdown_seed_from_int
    starts = [r for r in records if r["kind"] == "command" and r["type"] == "start"]
    require(len(starts) == 1 and starts[0].get("formatid") == "gen3randombattle"
        and starts[0].get("seed") == showdown_seed_from_int(seed), "actual single source start required")
    token = starts[0]["battleId"]
    started = [r for r in records if r["kind"] == "event" and r["type"] == "started"]
    require(len(started) == 1 and started[0].get("battleId") == token
        and started[0].get("seed") == starts[0]["seed"] and started[0].get("formatid") == "gen3randombattle",
        "observed source start acknowledgement drift")
    require(all(r.get("battleId") == token for r in records if "battleId" in r)
        and not any(r["type"] == "error" for r in records), "source transport refused or token drifted")
    markers = [line for r in records for line in r.get("terminal_markers",[])]
    terminals = [r for r in records if r["kind"] == "event" and r["type"] == "terminal"]
    if source["status"] == "COMPLETE":
        require(len(markers) == len(terminals) == 1 and markers[0] in {
            "|win|PokeZero p1","|win|PokeZero p2","|tie|"}, "observed source terminal required")
    else:
        require(source["status"] == "UNCERTAIN_SOURCE" and not markers and not terminals,
            "capped source cannot acquire terminal evidence")


def _read_source_teardown(records, driver):
    """Reconstruct the owner's normal close law, not merely its summary."""
    token=next(r["battleId"] for r in records if r["kind"]=="command" and r["type"]=="start")
    def positions(kind,event):
        return [i for i,r in enumerate(records) if r["kind"]==kind and r["type"]==event]
    ends,ended,closes,closed=(positions(k,e) for k,e in (
        ("command","end"),("event","ended"),("command","close"),("event","closed")))
    require(len(ends)==len(ended)==len(closes)==len(closed)==1
        and ends[0]<ended[0]==len(records)-3 and closes[0]==len(records)-2 and closed[0]==len(records)-1,
        "source end/ended/close/closed order differs")
    require(records[ends[0]]==_source_trace("command",dict(type="end",battleId=token))
        and records[ended[0]]==_source_trace("event",dict(type="ended",battleId=token))
        and records[closes[0]]==_source_trace("command",dict(type="close"))
        and records[closed[0]]==_source_trace("event",dict(type="closed"))
        and all(r["kind"]=="event" and r["type"]=="stream_end" for r in records[ends[0]+1:ended[0]]),
        "source teardown has unobserved/intervening work")
    eof=positions("event","stream_end")
    require(len(eof)==3 and sorted(records[i].get("stream") for i in eof)==["omniscient","p1","p2"]
        and all(i<ended[0] and records[i]==_source_trace("event",dict(type="stream_end",battleId=token,
            stream=records[i]["stream"])) for i in eof), "source stream EOF roster/order differs")
    choices=[r for r in records[:ends[0]] if r["kind"]=="command" and r["type"]=="choices"]
    acknowledgements=[r for r in records[:ends[0]] if r["kind"]=="event" and r["type"]=="choice_ack"]
    require(len(choices)==driver["decision_boundaries"] and all(
        sum(seat in r["players"] for r in choices)==sum(r.get("player")==seat for r in acknowledgements)
        ==driver["decisions_by_seat"][seat] for seat in ("p1","p2")),
        "source actual choices/acknowledgements differ from committed driver work")


def read_source_snapshot_trace(records, *, decision_boundaries):
    """Reconstruct exact received snapshots before each committed source step.

    Old traces without these observations remain lifecycle evidence only; this
    never upgrades a historical public fold. Private payloads are NOT exported.
    """
    require(type(records) is list and type(decision_boundaries) is int and decision_boundaries > 0,
        "exact source trace/decision count required")
    observations, pending, current, round_index = {}, False, None, 0
    for index, row in enumerate(records):
        kind, event = row["kind"], row["type"]
        if kind == "command" and event == "snapshot_actionable_boundary":
            require(not pending and current is None, "duplicate source boundary snapshot command")
            pending = True
        elif kind == "event" and event == "snapshot_actionable_boundary":
            require(pending and current is None and row.get("available") is True
                and type(row.get("record_index")) is int and row["record_index"] == index
                and type(row.get("decision_round_index")) is int and row["decision_round_index"] == round_index
                and _sha(row.get("snapshot_sha256")) and type(row.get("requested")) is list
                and row["requested"] and row["requested"] == sorted(set(row["requested"]))
                and set(row["requested"]) <= {"p1", "p2"},
                "unbound/unavailable source snapshot observation")
            current, pending = row, False
        elif kind == "command" and event == "choices":
            require(not pending, "source choices committed before snapshot response")
            if current is not None:
                require(current["requested"] == row["players"], "source snapshot/committed seat roster drift")
                observations[round_index] = deepcopy(current)
                current = None
            round_index += 1
    require(not pending and current is None and round_index == decision_boundaries,
        "unfinished source snapshot/decision roster")
    require(not observations or set(observations) == set(range(decision_boundaries)),
        "partial observed source snapshot history")
    return observations


def read_archived_source_snapshot_join(archive, source, records):
    """Join original retained private boundaries to independently received bytes.

    Callers still need the actual source owner/lifecycle. Return hashes only.
    No source request, Pokemon state, team, or private history leaves the join.
    """
    require(type(archive) is SealedSourceArchive and type(source) is dict,
        "exact original archive/source partition required")
    observed = read_source_snapshot_trace(records, decision_boundaries=source["decision_boundaries"])
    result = {}
    for root in source["roots"]:
        index = root["source_request_index"]
        require(type(index) is int and index in observed and index in archive._private,
            "LABEL_INPUT_UNAVAILABLE: original source snapshot transport not retained")
        snapshot = archive._private[index]
        bridge = getattr(snapshot, "bridge_snapshot", None)
        require(type(bridge) is dict and digest(bridge) == observed[index]["snapshot_sha256"],
            "original source archive differs from received snapshot bytes")
        result[root["root_id"]] = dict(source_request_index=index,
            source_record_index=observed[index]["record_index"],
            source_snapshot_sha256=observed[index]["snapshot_sha256"],
            retained_archive_root_sha256=retained_archive_root_identity(archive, index))
    return result


def _policies_identity(policies):
    require(type(policies) is dict and set(policies) == {"p1","p2"}
        and all(type(p) is AuditedRawPolicy for p in policies.values()), "exact two audited raw policies required")
    rows = {}
    for seat,auditor in policies.items():
        p = auditor.policy
        require(p.deterministic is True and p.exploration_epsilon == 0 and p.sampling_temperature == 1
            and p.family_gated_selection is False and p.history_mask_k is None
            and p.result.model_config.window_size == 1 and p.forward_fn == p._default_forward
            and p.record_policy_distribution is True, "raw source policy law changed")
        rows[seat] = dict(auditor_id=id(auditor),policy_id=id(p),actor_history_id=id(auditor.actor_history),checkpoint=p.weights_sha256,
            config=_state_payload(p.result.model_config),forward_id=id(getattr(p.forward_fn,"__func__",p.forward_fn)))
    return digest(rows)


def _production_components(config, checkpoint):
    from ..local_showdown import LocalShowdownEnv
    from ..collection import env_config_with_policy_spec_masks
    from ..neural_policy import load_transformer_policy
    from .resolver import resolve_checkpoint_contract
    actual = resolve_checkpoint_contract(checkpoint.checkpoint_path,expected_sha256=checkpoint.checkpoint_sha256,
        model_device="cpu",showdown_root=checkpoint.showdown_root,showdown_source_sha256=checkpoint.showdown_source_sha256,
        model_path=checkpoint.model_path,tables_path=checkpoint.tables_path)
    require(digest(actual.to_manifest()) == digest(checkpoint.to_manifest()), "loaded source checkpoint contract drift")
    derived = env_config_with_policy_spec_masks(config,["neural:"+checkpoint.checkpoint_path],context="owned exploration source")
    require(digest(_state_payload(derived)) == digest(_state_payload(config)) and config.set_belief_source is True,
        "source environment must already match checkpoint encoding and public catalog")
    policies = {seat:load_transformer_policy(Path(checkpoint.checkpoint_path),device="cpu",deterministic=True,
        exploration_epsilon=0.,sampling_temperature=1.,family_gated_selection=False) for seat in ("p1","p2")}
    require(all(p.result.belief_set_source_hash == checkpoint.showdown_source_sha256 for p in policies.values()),
        "source policies disagree with reviewed catalog")
    return LocalShowdownEnv(config),policies


def _retained_archive_identity(archive, source):
    return digest(dict(subject=archive.subject,public_indices=sorted(archive._public),
        private_indices=sorted(archive._private),roots={r["root_id"]:
            retained_archive_root_identity(archive,r["source_request_index"]) for r in source["roots"]}))


class TrustedOwnedSource:
    """Process-local archive handle; cannot cross worker transport."""
    def __init__(self, ledger, archive, source, directory, claim_bytes, runtime, bindings, actor_history=None):
        self.ledger,self.archive,self.source,self.directory = ledger,archive,deepcopy(source),directory
        self.claim_bytes,self.runtime,self.bindings = claim_bytes,deepcopy(runtime),deepcopy(bindings)
        self._actor_history = actor_history
        self.pid = os.getpid()
        self._scope = current_scope()
        self._objects = (ledger,archive,directory,self.runtime,self.bindings,self.source,actor_history)
        self._owner_bytes = {p.name:p.read_bytes() for p in directory.glob("*.json")}
        self._archive_sha = _retained_archive_identity(archive,self.source)
        self.frozen = self._identity()

    def _identity(self):
        return digest(dict(source=self.source,bindings=self.bindings,runtime=self.runtime,
            retained_archive_sha256=_retained_archive_identity(self.archive,self.source),
            actor_history_sha256=self._actor_history.identity() if self._actor_history is not None else None,
            directory=str(self.directory),claim_sha256=hashlib.sha256(self.claim_bytes).hexdigest(),
            contract_sha256=self.ledger.contract_sha256,ledger_directory=str(self.ledger.directory)))

    def _original_owner(self):
        require(current_scope() is self._scope and self._scope.pid==os.getpid()==self.pid
            and self._scope.thread_id==threading.get_ident() and os.getpgrp()==self.pid,
            "trusted source cannot leave its original auditor/thread/group")
        require(_retained_archive_identity(self.archive,self.source)==self._archive_sha,
            "original retained source archive changed")
        require(type(self.ledger) is ReconciledPhaseALedger and type(self.archive) is SealedSourceArchive
            and all(actual is original for actual,original in zip((self.ledger,self.archive,self.directory,
                self.runtime,self.bindings,self.source,self._actor_history),self._objects)) and self._identity()==self.frozen,
            "original trusted source identities changed")
        require({p.name:p.read_bytes() for p in self.directory.glob("*.json")}==self._owner_bytes,
            "original source owner receipt inventory changed")
        if self._actor_history is not None:
            self._actor_history.verify(_load(self.directory/"bridge-transport.json")["records"])
            require(len(self._actor_history._policy_inputs) == _load(self.directory/"driver.json")["decisions_by_seat"]["p1"],
                "source actor witness omitted an actual committed raw input")
        claim=_load(self.directory/"claim.json")
        require(claim["auditor_pid"]==self._scope.pid and claim["auditor_binding_sha256"]==self._scope.binding,
            "source claim differs from original auditor scope")

    def __reduce_ex__(self, protocol): raise TypeError("trusted source handle cannot be serialized")

    def validate(self):
        self._original_owner()
        self.ledger.verify()
        require((self.directory/"claim.json").read_bytes() == self.claim_bytes
            and self._identity() == self.frozen,
            "original source handle/claim changed")
        owner_bytes = {p.name:p.read_bytes() for p in self.directory.glob("*.json")}
        read_source_owner(self.directory,self.source,self.runtime,self.ledger.contract)
        for root in self.source["roots"]:
            actual = archived_exploration_root_binding(self.archive,self.source,contract=self.ledger.contract,root_id=root["root_id"])
            require(digest(actual) == self.bindings[root["root_id"]], "retained source archive drift")
        self.ledger.verify()
        require((self.directory/"claim.json").read_bytes() == self.claim_bytes
            and self._identity() == self.frozen,
            "source inputs changed during archive callbacks")
        require({p.name:p.read_bytes() for p in self.directory.glob("*.json")} == owner_bytes,
            "source owner receipts changed during archive callbacks")
        read_source_owner(self.directory,self.source,self.runtime,self.ledger.contract)
        self._original_owner()
        return self

    def close(self): self.archive.close()

    def require_inputs(self, ledger, archive, source):
        """Only the original handle may supply a fresh selector's source."""
        require(type(self) is TrustedOwnedSource and ledger is self.ledger
            and archive is self.archive and source is self.source,
            "exact original source owner/ledger/archive/public partition required")
        self.validate()
        return self

    def public_history(self, root_id):
        """Read-only prospective public prefix from this exact retained source."""
        from .search_over_raw_source_public_history import archived_public_history
        self.validate()
        root = next((r for r in self.source["roots"] if r["root_id"] == root_id), None)
        require(root is not None, "original selected exploration root required")
        records = _load(self.directory/"bridge-transport.json")["records"]
        joined = read_archived_source_snapshot_join(self.archive, self.source, records)
        result = archived_public_history(self.archive, root, joined[root_id], self.runtime)
        self.validate()
        require(read_archived_source_snapshot_join(self.archive, self.source, records) == joined,
            "source snapshot join changed during public projection")
        return result

    def actor_program(self, root_id, *, config):
        """Original-subject inputs only, with a source-owned selected cutoff.

        Never use this original p1 program as a sampled p2 private history.
        A mismatch with the existing source encoding remains unavailable.
        """
        from .search_over_raw_source_actor_history import read_source_actor_program, retained_program_identity
        self.validate()
        require(self._actor_history is not None, "LABEL_INPUT_UNAVAILABLE: source actor capture absent")
        root = next((r for r in self.source["roots"] if r["root_id"] == root_id), None)
        require(root is not None, "original selected source root required")
        records = _load(self.directory/"bridge-transport.json")["records"]
        joined = read_archived_source_snapshot_join(self.archive, self.source, records)
        result = read_source_actor_program(self._actor_history, records=records,
            cutoff_record_index=joined[root_id]["source_record_index"], config=config)
        frozen = retained_program_identity(result)
        self.validate()
        final_join = read_archived_source_snapshot_join(self.archive, self.source, records)
        # The LAST join callback can mutate an earlier original input. Do not
        # adopt its new baseline or rely only on equality of its returned map.
        self._original_owner()
        require(final_join == joined and retained_program_identity(result) == frozen
            and digest(_state_payload(config)) == result.receipt["config_sha256"]
            and digest(records) == digest(_load(self.directory/"bridge-transport.json")["records"]),
            "source actor program changed during final source checks")
        result.receipt["process_ownership_validated"] = True
        result.receipt["source_prefix_join_validated"] = True
        return result


class TrustedExplorationPanel:
    """All fixed source attempts retained in one auditor, including caps.

    This is a private roster coordinator, not an allocation, review or launch
    permission. A partial roster cannot authorize an irreversible stage write.
    """
    def __init__(self, ledger, owners):
        require(type(ledger) is ReconciledPhaseALedger and type(owners) is tuple,
            "exact ledger and retained source-owner tuple required")
        seeds=ledger.contract["panels"]["exploration"]["seeds"]
        require(len(owners)==len(seeds) and all(type(o) is TrustedOwnedSource and o.ledger is ledger
            and o.source["source_seed"]==seed for o,seed in zip(owners,seeds)),
            "complete original fixed source roster required before stage writing")
        recipes={(c["recipe_sha256"],c["runtime_sha256"],c["source_review_sha256"])
            for o in owners for c in (_load(o.directory/"claim.json"),)}
        require(len(recipes)==1,"all fixed panel sources require one prospectively reviewed source recipe")
        self.source_recipe=next(iter(recipes))
        self.ledger,self.owners=ledger,owners
        self._original_ledger,self._original_owners=ledger,owners
        self._frozen_owners=tuple(owners)
        self._scope=current_scope()
        self.by_root={r["root_id"]:o for o in owners for r in o.source["roots"]}
        missing=[root for o in owners for root in o.source["missing_root_ids"]]
        slots=[r["root_id"] for r in ledger.contract["panels"]["exploration"]["root_slots"]]
        require(len(self.by_root)+len(missing)==len(slots) and len(set(missing))==len(missing)
            and not set(self.by_root)&set(missing) and set(self.by_root)|set(missing)==set(slots),
            "selected and missing source roots must partition the original fixed panel")
        self.missing_root_ids=tuple(missing)
        self._original_by_root=self.by_root
        self.frozen=self._identity()
        self.validate()

    def __reduce_ex__(self, protocol):raise TypeError("trusted exploration panel cannot be serialized")

    def _identity(self):
        return digest(dict(ledger=id(self.ledger),contract=self.ledger.contract_sha256,
            source_recipe=self.source_recipe,
            owners=[(id(o),o._identity()) for o in self.owners],
            roots={r:id(o) for r,o in self.by_root.items()},missing=self.missing_root_ids))

    def _original_inputs(self):
        require(self.ledger is self._original_ledger and self.owners is self._original_owners
            and self.by_root is self._original_by_root and len(self.owners)==len(self._frozen_owners)
            and all(a is b for a,b in zip(self.owners,self._frozen_owners))
            and current_scope() is self._scope and self._identity()==self.frozen,
            "original fixed panel identities changed")
        # No archive extraction callbacks here. Check every earlier owner after
        # the last owner/caller callback, not against a newly adopted baseline.
        for owner in self.owners:owner._original_owner()

    def validate(self):
        self._original_inputs()
        for owner in self.owners:owner.validate()
        self.ledger.verify()
        self._original_inputs()
        return self

    def require_stage_inputs(self, ledger, plans, *, input_unavailable=None, configuration=None):
        unavailable={} if input_unavailable is None else input_unavailable
        require(type(unavailable) is dict and not set(plans)&set(unavailable)
            and ledger is self.ledger and type(plans) is dict and set(plans)|set(unavailable)==set(self.by_root)
            and all(plan.source_owner is self.by_root[root] and plan.archive is self.by_root[root].archive
                and digest(plan.source)==digest(self.by_root[root].source) for root,plan in plans.items()),
            "stage must partition every captured root into owned work or validated unavailable input")
        frozen=digest(unavailable)
        self.validate()
        if unavailable:
            from .search_over_raw_selector_producer import read_actor_input_unavailable
            require(configuration is not None, "unavailable partition requires its exact configuration")
            for root,receipt in unavailable.items():
                require(receipt["root_id"] == root, "unavailable root coordinate drift")
                read_actor_input_unavailable(ledger,self.by_root[root],receipt,configuration=configuration)
        self._original_inputs()
        require(digest(unavailable) == frozen, "unavailable source partition changed during validation")
        return self

    def receipt(self):
        self._original_inputs()
        return dict(schema="pokezero.search-over-raw.retained-exploration-panel.v1",
            contract_sha256=self.ledger.contract_sha256,source_count=len(self.owners),
            source_recipe_sha256=self.source_recipe[0],source_runtime_sha256=self.source_recipe[1],
            source_review_sha256=self.source_recipe[2],
            source_sha256_by_seed={str(o.source["source_seed"]):digest(o.source) for o in self.owners},
            observed_root_ids=list(self.by_root),missing_root_ids=list(self.missing_root_ids),
            source_roster_complete=True,source_pipeline_validated=True,
            runtime_qualified=False,collection_authorized=False,holdout_authorized=False)


def collect_owned_exploration_panel(ledger, **source_inputs):
    """Collect only an already fixed roster; no redraw, registration or retry."""
    require("source_seed" not in source_inputs,"panel uses only its original registered source seeds")
    owners=[]
    try:
        for seed in ledger.contract["panels"]["exploration"]["seeds"]:
            # Each one-shot driver consumes its own claim BEFORE reset. A
            # normal cap is missing data; an operational failure stops here.
            owners.append(collect_owned_exploration_source(ledger,source_seed=seed,**source_inputs))
        return TrustedExplorationPanel(ledger,tuple(owners))
    except BaseException as error:
        for owner in owners:owner.close()
        if not (ledger.directory/"operational-stop.json").exists():
            try:ledger.stop(type(error).__name__)
            except BaseException as stopping:error.add_note("Panel stop recording failed: "+type(stopping).__name__)
        raise


def read_source_owner(directory, source, runtime, contract):
    """Independent durable source lifecycle reader; no physics/inference claim."""
    directory = _path(directory)
    original = {p.name:p.read_bytes() for p in directory.glob("*.json")}
    claim,driver,launch,closed,trace = (_load(directory/name) for name in (
        "claim.json","driver.json","bridge-launch.json","bridge-close.json","bridge-transport.json"))
    require(set(original)=={"claim.json","driver.json","bridge-launch.json","bridge-close.json",
        "bridge-transport.json","source.json"} and type(source) is dict
        and source.get("schema")=="pokezero.search-over-raw.source.v1"
        and source["contract_sha256"]==digest(contract) and type(source["source_seed"]) is int
        and source["source_seed"] in contract["panels"]["exploration"]["seeds"]
        and source["source_policy"]=="raw_argmax_both_seats" and source["replacement_seeds"]==[]
        and source["scientific_strength_evidence"] is False and source["search_invoked"] is False
        and source["checkpoint_sha256"]==runtime["files"][runtime["checkpoint_path"]]
        and type(source["elapsed_seconds"]) in (float,int) and math.isfinite(source["elapsed_seconds"])
        and source["elapsed_seconds"]>=0, "exact unpromoted original source partition required")
    require(set(claim)=={"schema","contract_sha256","source_seed","source_sha256","recipe_sha256",
        "runtime_sha256","source_review_sha256","auditor_binding_sha256","auditor_pid",
        "synthetic_process_fixture","retry_authorized","holdout_authorized"}
        and claim["schema"]=="pokezero.search-over-raw.source-attempt.v1"
        and type(claim["synthetic_process_fixture"]) is bool
        and claim["retry_authorized"] is False and claim["holdout_authorized"] is False
        and all(_sha(claim[k]) for k in ("recipe_sha256","source_review_sha256","auditor_binding_sha256")),
        "exact one-shot source owner claim required")
    require(digest(_load(directory/"source.json")) == digest(source), "durable source public partition drift")
    require(claim["contract_sha256"] == digest(contract) and claim["source_seed"] == source["source_seed"]
        and source["panel"] == "exploration" and claim["source_sha256"] is None
        and claim["runtime_sha256"] == verify_outcome_runtime(runtime)
        and driver["source_sha256"] == digest(source), "source owner/driver input mismatch")
    slots = [r["root_id"] for r in contract["panels"]["exploration"]["root_slots"] if r["source_seed"] == source["source_seed"]]
    require(set(driver)=={"schema","decision_boundaries","decisions_by_seat","trajectory_sha256",
        "terminal_capped","source_sha256"} and driver["schema"]=="pokezero.search-over-raw.source-driver.v1"
        and _sha(driver["trajectory_sha256"]) and type(driver["decisions_by_seat"]) is dict
        and set(driver["decisions_by_seat"])=={"p1","p2"}
        and type(driver["decision_boundaries"]) is int and 0 < driver["decision_boundaries"] <= 250
        and all(type(n) is int and 0<=n<=driver["decision_boundaries"] for n in driver["decisions_by_seat"].values())
        and type(source["requested_root_slots"]) is int and source["requested_root_slots"]==len(slots)
        and type(source["eligible_requests"]) is int and source["eligible_requests"]==len(source["eligible_public_records"])
        and type(source["decision_boundaries"]) is int and type(source["verified_raw_decisions"]) is int
        and source["decision_boundaries"] == driver["decision_boundaries"]
        and source["verified_raw_decisions"] == sum(driver["decisions_by_seat"].values()),
        "actual source driver decision accounting differs")
    if source["status"] == "COMPLETE":
        validate_source_receipt(source,source["source_seed"],contract=contract,panel="exploration")
        require(driver["terminal_capped"] is False, "completed source driver was capped")
    else:
        require(source["status"] == "UNCERTAIN_SOURCE" and driver["terminal_capped"] is True
            and driver["decision_boundaries"] == 250 and source["roots"] == []
            and source["missing_root_ids"] == slots and source["eligible_public_records"] == []
            and source["eligible_catalog_sha256"] == digest([]) and source["source_terminal_complete"] is False,
            "normally capped source must retain every fixed slot as missing")
    require(launch["schema"] == SourceProcessOwner.launch_schema and closed["schema"] == SourceProcessOwner.close_schema
        and type(launch["pid"]) is int and launch["pid"] > 0
        and type(claim["auditor_pid"]) is int and claim["auditor_pid"]>0
        and type(launch["group"]) is int and type(launch["parent_pid"]) is int
        and launch["group"] == launch["parent_pid"] == claim["auditor_pid"] and launch["pid"]!=claim["auditor_pid"]
        and launch["argv"] == runtime["argv"] and launch["runtime_sha256"] == digest(runtime)
        and closed["launch_sha256"] == digest(launch) and closed["source_sha256"] == digest(source)
        and closed["transport_sha256"] == digest(trace["records"])
        and type(closed["exit_code"]) is int and closed["exit_code"] == 0 and closed["reaped"] is True
        and closed["normal_close"] is True and closed["forced_cleanup"] is False
        and closed["wait_ownership_lost"] is False and closed["pumps_drained"] is True
        and closed["error_type"] is None
        and closed["pid"] == launch["pid"] and closed["parent_pid"] == launch["parent_pid"]
        and closed["binding_sha256"] == launch["binding_sha256"] == claim["auditor_binding_sha256"]
        and closed["runtime_sha256"] == digest(runtime), "actual normally reaped source bridge required")
    _match_source_trace(source,trace["records"],source["source_seed"])
    _read_source_teardown(trace["records"],driver)
    snapshots = read_source_snapshot_trace(trace["records"], decision_boundaries=driver["decision_boundaries"])
    require(trace["records"][-1] == _source_trace("event",dict(type="closed"))
        and {p.name:p.read_bytes() for p in directory.glob("*.json")} == original, "source owner bytes changed while reading")
    return dict(source_sha256=digest(source), source_pipeline_validated=True,
        source_snapshot_transport_validated=bool(snapshots), source_inference_truth_validated=False,
        source_physics_validated=False, full_resource_cost_complete=False, runtime_qualified=False,
        collection_authorized=False, holdout_authorized=False)


def collect_owned_exploration_source(ledger, *, stage, configuration, source_seed, config, checkpoint_contract,
        bridge_runtime, source_review_path, history_review_path, _synthetic_factory=None,
        capture_actor_history=False):
    """Run one previously allocated source once, entirely inside its auditor."""
    from ..local_showdown import LocalShowdownEnv
    require(type(ledger) is ReconciledPhaseALedger, "actual reconciled ledger required")
    scope = current_scope()
    require(scope is not None and scope.pid == os.getpid() and os.getpgrp() == scope.pid,
        "source driver must remain inside an externally owned auditor")
    require(stage in {"A1","A2"} and configuration in {c.identity for c in ledger._stage_configs(stage)}
        and type(source_seed) is int and source_seed in ledger.contract["panels"]["exploration"]["seeds"]
        and not (ledger.directory/"a2-freeze.json").exists() and not (ledger.directory/"selection.json").exists(),
        "unfrozen fixed exploration source only; no holdout")
    synthetic = _synthetic_factory is not None
    if synthetic:
        require(isinstance(_synthetic_factory,type) and getattr(_synthetic_factory,"synthetic_process_fixture",False) is True
            and ledger.contract["execution_ready"] is False and ledger.contract["collection_authorized"] is False,
            "explicit unadmitted authored source components required")
    else: ledger.verify_exploration(stage,configuration)
    recipe = source_recipe(config,checkpoint_contract,bridge_runtime,capture_actor_history=capture_actor_history)
    review_path = _path(source_review_path);review_bytes = review_path.read_bytes();review = _load(review_path)
    require(ledger.contract["input_hashes"].get(str(review_path)) == hashlib.sha256(review_bytes).hexdigest(),
        "source review must precede ledger/source construction")
    expected = source_review_template(ledger.contract,recipe,qualification_evidence_hashes=review["qualification_evidence_hashes"])
    expected.update(status="CLEAR_SOURCE_RECIPE",reviewed_before_first_source_work=True,
        source_runtime_and_cleanup_reviewed=True,source_policy_and_terminal_reviewed=True)
    require(digest(review) == digest(expected), "unknown or mismatched source recipe review")
    history,sha = _fresh_review(ledger,history_review_path)
    from .search_over_raw import SearchConfiguration
    from .search_over_raw_raw_run import raw_runtime
    require(history["raw_runtime_sha256"] == digest(raw_runtime(SearchConfiguration("raw",seconds=1.),
        checkpoint_contract.checkpoint_sha256,checkpoint_contract.showdown_source_sha256)),
        "source champion/catalog differs from shared raw recipe")
    namespace = _path(ledger.directory/"continuation-admission"/"namespace.json");namespace_bytes = namespace.read_bytes()
    require(digest(_load(namespace)) == digest(dict(schema=NAMESPACE_SCHEMA,contract_sha256=ledger.contract_sha256,
        review_path=str(_path(history_review_path)),review_sha256=sha,review=history,**_FLAGS)), "fresh source namespace required")
    original_inputs = digest(dict(recipe=recipe,config=_state_payload(config),checkpoint=checkpoint_contract.to_manifest(),runtime=bridge_runtime))
    def guard():
        ledger.verify()
        if not synthetic: ledger.verify_exploration(stage,configuration)
        require(review_path.read_bytes() == review_bytes and namespace.read_bytes() == namespace_bytes
            and digest(dict(recipe=source_recipe(config,checkpoint_contract,bridge_runtime,
                capture_actor_history=capture_actor_history),config=_state_payload(config),
                checkpoint=checkpoint_contract.to_manifest(),runtime=bridge_runtime)) == original_inputs,
            "source admission/runtime bytes changed")
    guard()
    parent = _path(ledger.directory/"source-attempts");parent.mkdir(exist_ok=True,mode=0o700)
    directory = _path(parent/str(source_seed))
    try: directory.mkdir(exist_ok=False,mode=0o700)
    except FileExistsError:
        ledger.stop("FileExistsError")  # Stranded or consumed source attempts are never reopened.
        raise
    claim = dict(schema="pokezero.search-over-raw.source-attempt.v1",contract_sha256=ledger.contract_sha256,
        source_seed=source_seed,source_sha256=None,recipe_sha256=digest(recipe),runtime_sha256=digest(bridge_runtime),
        source_review_sha256=hashlib.sha256(review_bytes).hexdigest(),auditor_binding_sha256=scope.binding,
        auditor_pid=scope.pid,synthetic_process_fixture=synthetic,retry_authorized=False,holdout_authorized=False)
    _write_new(directory/"claim.json",claim);claim_bytes = (directory/"claim.json").read_bytes()
    archive,env,owner = SealedSourceArchive(),None,None
    try:
        factory = _synthetic_factory or _production_components
        env,raw_policies = factory(config,checkpoint_contract)
        require((isinstance(env,LocalShowdownEnv) if synthetic else type(env) is LocalShowdownEnv)
            and digest(_state_payload(env.config)) == recipe["config_sha256"] and set(raw_policies) == {"p1","p2"},
            "source components differ from reviewed recipe")
        from .search_over_raw_source_actor_history import SourceActorHistoryCapture
        actor_history = SourceActorHistoryCapture(config) if capture_actor_history else None
        policies = {seat:AuditedRawPolicy(raw_policies[seat],checkpoint_sha256=checkpoint_contract.checkpoint_sha256,
            public_context_sink=archive.capture_public if seat == "p1" else None,
            actor_history=actor_history if seat == "p1" else None) for seat in ("p1","p2")}
        frozen_policies = _policies_identity(policies)
        ownership = dict(controller_pid=scope.pid,registration_sha256=scope.binding,directory=str(scope.directory/"owned-groups"))
        owner = SourceProcessOwner(source_seed=source_seed,directory=directory,ownership=ownership,
            runtime=bridge_runtime,config=config,actor_history=actor_history)
        driver = {}
        def accept_result(result):
            from ..trajectory import trajectory_to_dict
            require(not driver, "source result cannot be supplied twice")
            counts = {seat:sum(step.player_id == seat for step in result.trajectory.steps) for seat in ("p1","p2")}
            require(all(counts[seat] == policies[seat].verified_decisions for seat in counts),
                "source policy counters differ from actual committed trajectory")
            driver.update(schema="pokezero.search-over-raw.source-driver.v1",
                decision_boundaries=result.decision_round_count,decisions_by_seat=counts,
                trajectory_sha256=digest(trajectory_to_dict(result.trajectory)),terminal_capped=result.terminal.capped)
        with owner.installed(env):
            guard()
            require((directory/"claim.json").read_bytes() == claim_bytes, "source claim changed before reset")
            source = collect_raw_source(ledger.contract,panel="exploration",source_seed=source_seed,env=env,policies=policies,
                max_decision_rounds=250,sealed_pre_step_sink=archive.capture_private,_trusted_result_sink=accept_result)
            require(_policies_identity(policies) == frozen_policies and driver, "source policy inputs changed during rollout")
            owner.seal(source)
            env.close()
        driver["source_sha256"] = digest(source)
        _write_new(directory/"source.json",source)
        _write_new(directory/"driver.json",driver)
        guard()
        require((directory/"claim.json").read_bytes() == claim_bytes, "source claim changed during final validation")
        owner_bytes={p.name:p.read_bytes() for p in directory.glob("*.json")}
        archive_sha=_retained_archive_identity(archive,source)
        read_source_owner(directory,source,bridge_runtime,ledger.contract)
        bindings = {r["root_id"]:digest(archived_exploration_root_binding(archive,source,
            contract=ledger.contract,root_id=r["root_id"])) for r in source["roots"]}
        guard()
        require({p.name:p.read_bytes() for p in directory.glob("*.json")}==owner_bytes,
            "source receipts changed while constructing archive crosswalks")
        require(_retained_archive_identity(archive,source)==archive_sha,
            "retained source roots changed while constructing archive crosswalks")
        if source["status"] == "UNCERTAIN_SOURCE":
            # The independent terminal reader above proved normal cap and the
            # complete missing quota. None of its request snapshots is an
            # accepted root. Release that unusable private history BEFORE the
            # retained handle freezes its empty archive identity; never clear
            # an already accepted handle or change source/terminal evidence.
            require(source["roots"] == [] and bindings == {},"capped source cannot retain an accepted root")
            archive.close()
        handle = TrustedOwnedSource(ledger,archive,source,directory,claim_bytes,bridge_runtime,bindings,
            actor_history=actor_history)
        return handle.validate()
    except BaseException as error:
        archive.close()
        if env is not None and env._process is not None:
            try: env.close()
            except BaseException as cleanup:error.add_note("Source cleanup failed: "+type(cleanup).__name__)
        if not (ledger.directory/"operational-stop.json").exists():
            try: ledger.stop(type(error).__name__)
            except BaseException as stopping:error.add_note("Source stop recording failed: "+type(stopping).__name__)
        raise
