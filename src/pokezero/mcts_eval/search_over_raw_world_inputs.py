"""Opt-in actual world-input witnesses, not a world sampler or equivalence gate.

Initial inputs are compared within an arm, not across engines or across visited
leaves. No source truth enters public transport. Instrumentation is inside the
decision clock and qualifies only its own runtime. Private state is hashed in
the owning process; no snapshot is transported to an actor or master.
"""
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from enum import Enum
import json
import hashlib
import math
import time
from pathlib import PurePath

from .search_over_raw import digest, require

SCHEMA = "pokezero.search-over-raw.consumed-world-inputs.v1"
OWNERSHIP_METADATA = ["battle_token", "battle_id"]
SNAPSHOT_INPUT_FIELDS = ("format_id", "observation_format_id", "bridge_snapshot",
    "protocol_lines", "latest_requests", "first_requests", "request_history", "replay",
    "belief_engine", "latest_turn", "terminal", "search_choice_cache", "annotation_cache")
STATE_OBJECT_TYPES = frozenset({"pokezero.belief.PublicBattleBeliefEngine",
    "pokezero.tier2.Tier2LiveTracker", "pokezero.tier2._IncrementalContextFold",
    "pokezero.investment.InvestmentLiveTracker"})
TIMED_NATIVE_PUBLIC_CAPTURE = dict(schema="pokezero.search-over-raw.public-request-consumption.v1",
    at_actual_adapter_capture=True, at_each_actual_native_sampling_attempt=True,
    pending_transition_included=True, private_source_snapshot_included=False, runtime_qualified=False)
TIMED_REFERENCE_PUBLIC_CAPTURE = dict(schema="pokezero.search-over-raw.public-request-consumption.v1",
    at_actual_adapter_capture=True, at_each_actual_reference_worker_preparation=True,
    checked_at_each_worker_export=True, includes_workers_without_draws=True,
    pending_transition_included=True, private_source_snapshot_included=False, runtime_qualified=False)


def _state_payload(value, active=None):
    """Deterministic full state, never repr/pickle or a partial public projection.

    All fields of known mutable state objects are retained, including caches.
    This can conservatively report a mismatch for non-causal cache differences;
    it cannot certify equality by silently deleting one. Unsupported objects or
    cycles refuse the witness. Hashing the full catalogs/dex costs real time.
    """
    active = set() if active is None else active
    kind = type(value)
    name = kind.__module__ + "." + kind.__qualname__
    if isinstance(value, Enum):
        return dict(enum=name, value=_state_payload(value.value, active))
    if value is None or kind in (str, int, float, bool):
        return value
    if isinstance(value, PurePath):
        return dict(path=str(value))
    require(id(value) not in active, "cyclic consumed input state")
    active.add(id(value))
    try:
        if isinstance(value, Mapping):
            entries = [[_state_payload(key, active), _state_payload(item, active)]
                for key, item in value.items()]
            entries.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, allow_nan=False))
            return dict(mapping=entries)
        if kind in (list, tuple, set, frozenset):
            items = [_state_payload(item, active) for item in value]
            if kind in (set, frozenset):
                items.sort(key=lambda item: json.dumps(item, sort_keys=True, allow_nan=False))
            return dict(container=kind.__name__, items=items)
        if is_dataclass(value) and not isinstance(value, type):
            return dict(dataclass=name, fields={field.name: _state_payload(
                getattr(value, field.name), active) for field in fields(value)})
        require(name in STATE_OBJECT_TYPES and hasattr(value, "__dict__")
            and not any("__slots__" in cls.__dict__ for cls in kind.__mro__),
            "unsupported consumed input state type")
        return dict(object=name, fields=_state_payload(vars(value), active))
    finally:
        active.remove(id(value))


def _snapshot_payload(snapshot):
    names = {field.name for field in fields(snapshot)} if is_dataclass(snapshot) else set(vars(snapshot))
    require(names == set(SNAPSHOT_INPUT_FIELDS) | set(OWNERSHIP_METADATA),
        "reference snapshot field contract drift")
    return {key: _state_payload(getattr(snapshot, key)) for key in SNAPSHOT_INPUT_FIELDS}


def valid_catalog_fingerprint(value):
    """Catalog identity, not necessarily a full SHA-256 file digest.

    randbat._source_hash emits 16 lowercase hex characters. Existing authored
    receipt fixtures also use full 64-character digests. Neither representation
    permits truncating or padding another binding: callers still compare the
    complete original fingerprint exactly.
    """
    return type(value) is str and len(value) in (16, 64) and all(
        character in "0123456789abcdef" for character in value)


def _binding(request):
    from .search_over_raw_oracle import public_root_binding
    from .paper_reference_showdown import decision_state
    subject = request.state.player_id
    return dict(root_binding=public_root_binding(request), subject=subject,
        set_source_hash=request.set_source_hash,
        information_key=decision_state(request.observation, player=subject).key.hex())


def public_request_witness(request):
    """Full public-only transport, including pending transition, not source truth."""
    from .paper_reference_runtime import PublicRootRequest
    require(type(request) is PublicRootRequest and request.state.replay.requests == {}
        and request.state.belief_engine.set_source is None, "sanitized actual public request required")
    return dict(schema="pokezero.search-over-raw.public-request-consumption.v1",
        public_request_sha256=digest(_state_payload(request)), consumed_binding=_binding(request),
        full_supported_public_state_and_pending=True, private_source_snapshot_included=False,
        runtime_qualified=False)


def validate_timed_public_request(evidence, *, binding, expected_sha256):
    """Actual capture plus the variable attempt roster, not a fixed-world gate."""
    manifest = evidence["consumed_world_inputs"]
    capture = manifest.get("adapter_public_request_witness")
    expected = dict(schema="pokezero.search-over-raw.public-request-consumption.v1",
        public_request_sha256=expected_sha256, consumed_binding=deepcopy(binding),
        full_supported_public_state_and_pending=True, private_source_snapshot_included=False,
        runtime_qualified=False)
    require(type(expected_sha256) is str and len(expected_sha256) == 64
        and all(c in "0123456789abcdef" for c in expected_sha256)
        and type(capture) is dict and digest(capture) == digest(expected),
        "actual timed adapter public capture differs from prospective archive")
    rows, observations = manifest["rows"], manifest.get("public_request_observations")
    require(type(observations) is list and len(observations) == len(rows)
        and all(type(row) is dict and set(row) == {"ordinal","witness"}
            and type(row["ordinal"]) is int and row["ordinal"] == i
            and type(rows[i]["ordinal"]) is int and rows[i]["ordinal"] == i
            and digest(row["witness"]) == digest(expected) for i,row in enumerate(observations)),
        "every actual timed native attempt requires exact public consumption witness")
    return dict(public_request_sha256=expected_sha256,observed_attempts=len(rows),
        adapter_capture_validated=True, full_public_request_consumption_validated=True,
        runtime_qualified=False, collection_authorized=False)


def _manifest(arm, binding, rows):
    return dict(schema=SCHEMA, arm=arm, **binding, rows=deepcopy(rows),
        snapshot_fields_excluded=[], initial_inputs_only=True,
        ownership_metadata_excluded=OWNERSHIP_METADATA if arm == "reference" else [],
        cross_engine_equivalence=False, instrumentation_can_change_deadlines=True,
        qualifies_uninstrumented_runtime=False, runtime_qualified=False)


class NativeWorldInputs:
    """Join actual constructed state to the serial native call boundary.

    Hook exceptions are retained because the engine swallows telemetry errors.
    Duplicate construction ordinals survive native multiplicity collapsing.
    This seam supports the plan's single native worker, fixed-depth/no-ladder
    configuration only; it cannot silently attest a different dispatcher.
    """
    def __init__(self, binding, iterations_per_world, *, deadline_limited=True, record_atomic_work=False,
                 record_construction=False, expected_public_request=None, pending_transition=None,
                 record_timed_atomic_work=False, record_public_fold=False,
                 record_compact_atomic_work=False):
        require(type(iterations_per_world) is int and iterations_per_world > 0,
            "positive native allocation required")
        self.binding, self.iterations_per_world = deepcopy(binding), iterations_per_world
        require(type(deadline_limited) is bool, "native deadline scope required")
        self.deadline_limited = deadline_limited
        require(type(record_atomic_work) is bool, "native atomic observation scope required")
        self.record_atomic_work = record_atomic_work
        require(type(record_compact_atomic_work) is bool and (not record_compact_atomic_work
            or record_atomic_work), "compact native observation requires explicit atomic scope")
        self.record_compact_atomic_work = record_compact_atomic_work
        require(type(record_timed_atomic_work) is bool and (not record_timed_atomic_work
            or (deadline_limited and record_atomic_work)), "timed atomic observation requires a deadline")
        self.record_timed_atomic_work = record_timed_atomic_work
        self._original_atomic_scope = (record_atomic_work, record_timed_atomic_work, record_compact_atomic_work)
        require(type(record_construction) is bool,"explicit construction observation scope required")
        self.record_construction,self.construction = record_construction,None
        require(type(record_public_fold) is bool and (not record_public_fold or
            record_construction and expected_public_request is not None and not deadline_limited),
            "public fold retention requires prospective fixed construction/full-request observation")
        self.record_public_fold = record_public_fold
        self.public_fold_attempt = None
        self.rows, self.calls, self.errors = [], [], []
        self.expected_public_request = deepcopy(expected_public_request)
        self.pending_transition = pending_transition
        self.public_request_observations = []

    def attempt(self, context, ordinal, override):
        try:
            require(type(ordinal) is int and ordinal == len(self.rows)
                and context.player_id == self.binding["subject"], "world attempt identity drift")
            if self.expected_public_request is not None:
                from .paper_reference_runtime import PublicRootRequest
                actual = public_request_witness(PublicRootRequest.capture(
                    context.public_materialization_state, context.observation,
                    pending_transition=self.pending_transition))
                require(digest(actual) == digest(self.expected_public_request),
                    "native sampler public request differs from archived transport")
                self.public_request_observations.append(dict(ordinal=ordinal, witness=actual))
            if self.record_public_fold:
                from .search_over_raw_public_fold import capture_request
                self.public_fold_attempt = capture_request(context, self.pending_transition)
                require(digest(self.public_fold_attempt["witness"]) == digest(self.expected_public_request),
                    "captured attempt public fold differs from original full request")
            self.rows.append(dict(worker=0, ordinal=ordinal,
                status="REFUSED" if override is None else "STARTED",
                draw_sha256=None if override is None else digest(asdict(override)),
                constructed_sha256=None, consumed=False))
            if self.record_construction:
                from .search_over_raw_construction import NativeConstructionCapture
                self.construction = None if override is None else NativeConstructionCapture(binding=self.binding,
                    ordinal=ordinal,draw_sha256=self.rows[-1]["draw_sha256"])
        except Exception as error:
            self.errors.append(type(error).__name__)
            raise

    def constructed(self, context, world, state):
        try:
            require(bool(self.rows) and self.rows[-1]["status"] == "STARTED"
                and context.player_id == self.binding["subject"], "unbound constructed world")
            row = self.rows[-1]
            text = state.to_string()
            require(type(text) is str and bool(text), "native world has no serialized state")
            world_payload = asdict(world)
            if self.record_construction:
                require(self.construction is not None,"actual construction hooks missing")
                row["construction_receipt"] = self.construction.export(expected_world=world_payload,expected_state=text)
            if self.record_public_fold:
                from .search_over_raw_public_fold import capture_request, construction_capture
                row["public_fold_capture"] = construction_capture(self.public_fold_attempt,
                    capture_request(context, self.pending_transition), ordinal=row["ordinal"],
                    construction_receipt=row["construction_receipt"])
            row.update(status="CONSTRUCTED", constructed_sha256=digest(world_payload),
                state_sha256=digest(text), party_species=deepcopy(world_payload["party_species"]),
                slot_sides=deepcopy(world_payload["slot_sides"]))
            if self.record_atomic_work:
                from .search_over_raw_native_atomic_work import state_hash
                row["native_state_utf8_sha256"] = state_hash(text)
        except Exception as error:
            self.errors.append(type(error).__name__)
            raise

    def world_built(self, value):
        require(self.record_construction and self.construction is not None,"unbound world assembly hook")
        self.construction.world_built(value)

    def native_built(self, value):
        require(self.record_construction and self.construction is not None,"unbound native assembly hook")
        self.construction.native_built(value)

    def _check_atomic_scope(self):
        actual = (self.record_atomic_work, self.record_timed_atomic_work, self.record_compact_atomic_work)
        require(all(type(value) is bool for value in actual) and actual == self._original_atomic_scope,
            "original native observation encoding scope changed")

    def dispatch(self, args):
        """Observe EXACT args sent to native, before the call, without draws."""
        self._check_atomic_scope()
        require(len(args) >= 12 and type(args[0]) is str and type(args[1]) is int
            and type(args[9]) is int, "unsupported native call contract")
        if self.record_timed_atomic_work:
            # The current Python builder puts the budget at slot 24, after all
            # earlier inert optional slots. Do not infer it from the last arg.
            require(len(args) == 25 and type(args[12]) is int and args[12] == 0
                and args[17] in (None, "hp_fraction") and type(args[24]) is int
                and args[24] > 0, "unsupported native timed ABI or early-stop/rollout mode")
        ctx = json.loads(args[5])
        # Bind the ACTUAL native fold object, not a reconstructed public fold.
        # The native API exposes all its state through to_payload(). No leaf
        # intervention parameters enter the matched-world/history key.
        fold_payload = args[6].to_payload()
        require(isinstance(fold_payload, Mapping), "native fold payload unavailable")
        candidates = [row for row in self.rows if row["status"] == "CONSTRUCTED"
            and not row["consumed"] and row["state_sha256"] == digest(args[0])
            and all(list(row["party_species"][seat]) == ctx[seat] for seat in ("p1", "p2"))]
        require(candidates and args[1] == self.iterations_per_world * len(candidates),
            "native dispatch/constructed multiplicity drift")
        require(len({row["constructed_sha256"] for row in candidates}) == 1,
            "identical native states conceal different constructed worlds")
        witness = dict(call_ordinal=len(self.calls),
            construction_ordinals=[row["ordinal"] for row in candidates],
            multiplicity=len(candidates), requested_iterations=args[1],
            deadline_limited=self.deadline_limited,
            input_sha256=digest(dict(state=args[0], context=ctx,
                public_inputs=json.loads(args[4]), chance_seed=args[9],
                fold=_state_payload(fold_payload), tables=json.loads(args[3]),
                world_sha256=candidates[0]["constructed_sha256"])),
            status="DISPATCHED", completed_iterations=None)
        self.calls.append(witness)
        if self.record_atomic_work:
            from .search_over_raw_native_atomic_work import state_hash
            witness["native_call_parameters"] = dict(state_sha256=state_hash(args[0]),
                requested_iterations=args[1], batch_size=args[2], chance_seed=args[9])
            if self.record_timed_atomic_work:
                witness["native_call_parameters"]["time_budget_ms"] = args[24]
            if self.record_compact_atomic_work:
                witness["compact_atomic_observation"] = True
        for row in candidates:
            row.update(consumed=True, input_sha256=witness["input_sha256"],
                call_ordinal=witness["call_ordinal"])
        return witness

    def finish(self, evidence, *, attempted, constructed, searched):
        self._check_atomic_scope()
        require(not self.errors and type(attempted) is int and attempted == len(self.rows)
            and type(constructed) is int and constructed == sum(
                row["status"] == "CONSTRUCTED" for row in self.rows),
            "native input hook/counter reconciliation failed")
        require(all(row["status"] == "RETURNED" for row in self.calls)
            and type(searched) is int and searched == sum(row["multiplicity"] for row in self.calls
                if row["completed_iterations"] > 0 or not row["deadline_limited"])
            and sum(row["completed_iterations"] for row in self.calls) == evidence["total_iterations"],
            "native consumed work/counter reconciliation failed")
        result = dict(_manifest("incumbent", self.binding, self.rows), calls=deepcopy(self.calls),
            construction_observed=self.record_construction)
        if self.record_compact_atomic_work:
            from .search_over_raw_native_atomic_compact import SCHEMA as COMPACT_SCHEMA
            result["native_atomic_encoding"] = COMPACT_SCHEMA
        if self.record_timed_atomic_work:
            result["timed_atomic_observation"] = dict(attempted=attempted, constructed=constructed,
                searched=searched, hook_errors=[], complete_counter_reconciliation=True,
                normal_native_returns=True, runtime_qualified=False)
        if self.expected_public_request is not None:
            require(len(self.public_request_observations) == len(self.rows),
                "every actual native attempt needs public transport observation")
            result["public_request_observations"] = deepcopy(self.public_request_observations)
            if self.record_timed_atomic_work:
                result["adapter_public_request_witness"] = deepcopy(self.expected_public_request)
        return result


class NativeInputHandle:
    """Delegates the genuine model unchanged; records call inputs and returns."""
    def __init__(self, base, recorder, *, value_session=None):
        self.base, self.recorder = base, recorder
        self.value_session = value_session

    def __getattr__(self, name):
        return getattr(self.base, name)

    def search_batched_multi_encoded(self, *args, **kwargs):
        original_base, original_recorder, original_session = self.base, self.recorder, self.value_session
        original_scope = original_recorder._original_atomic_scope
        def guard():
            self.recorder._check_atomic_scope()
            require(self.recorder._original_atomic_scope == original_scope,
                "original native observation encoding scope changed")
            if original_session is not None or original_scope[2]:
                require(self.base is original_base and self.recorder is original_recorder
                    and self.value_session is original_session,
                    "original native value/compact handle/base/recorder/session changed")
        guard()
        if self.recorder.record_atomic_work:
            require("record_atomic_work" not in kwargs or kwargs["record_atomic_work"] is True,
                "native atomic observation cannot be disabled")
            kwargs = {**kwargs, "record_atomic_work": True}
        if self.recorder.record_compact_atomic_work:
            require("record_compact_atomic_work" not in kwargs or kwargs["record_compact_atomic_work"] is True,
                "compact native observation cannot be disabled")
            require(original_session is None,
                "compact native format is not admitted to existing value/ancestry sessions")
            # The normal builder stops at budget slot24; engine ancestry can
            # inject its bank separately as a keyword (native positional31).
            # Neither path is prospectively admitted to this compact observer.
            require(len(args) <= 25 and kwargs.get("visited_value_bank") is None,
                "compact native format excludes visited banks and extended positional ABI")
            kwargs = {**kwargs, "record_compact_atomic_work": True}
        else:
            require(kwargs.get("record_compact_atomic_work", False) is False,
                "unreviewed compact native observation cannot replace v1")
        witness = None
        try:
            witness = self.recorder.dispatch(args)
        except Exception as error:
            self.recorder.errors.append(type(error).__name__)
        handle = None
        try:
            guard()
            if original_session is not None:
                from .search_over_raw_native_fixed_values import NativeFixedWorkValueSession
                require(type(original_session) is NativeFixedWorkValueSession and witness is not None
                    and self.recorder.record_atomic_work is True
                    and self.recorder.deadline_limited is False and "visited_value_bank" not in kwargs,
                    "native fixed-work labels require original actual dispatch, no competing bank or deadline")
                handle = original_session.begin(dict(state_str=args[0], seed=args[9],
                    construction_ordinals=deepcopy(witness["construction_ordinals"])))
                require(digest(original_session.rows[-1]["parameters"]) ==
                    digest(witness["native_call_parameters"]), "native label allocation differs from observed call")
                kwargs = {**kwargs, "visited_value_bank": handle}
            guard()
            result = self.base.search_batched_multi_encoded(*args, **kwargs)
            guard()
            if handle is not None:
                original_session.complete(handle, result)
        except BaseException:
            if witness is not None:
                witness["status"] = "REFUSED"
            if original_session is not None:
                original_session.close()
            raise
        if witness is not None:
            try:
                require(type(result) is str, "actual native UTF-8 report required")
                raw_sha = hashlib.sha256(result.encode("utf-8")).hexdigest()
                report = json.loads(result)
                completed = report["iterations"]
                require(type(completed) is int and 0 <= completed <= args[1],
                    "invalid completed native work")
                witness.update(status="RETURNED", completed_iterations=completed,
                    report_sha256=digest(report),
                    report_utf8_sha256=raw_sha)
                if self.recorder.record_atomic_work:
                    reader = _native_atomic_reader(compact=self.recorder.record_compact_atomic_work,
                        timed=self.recorder.record_timed_atomic_work)
                    witness["atomic_work_validation"] = reader(report,
                        **witness["native_call_parameters"])
                    witness["native_report"] = report
                    # A parsed-JSON hash is not the hash bound by the native
                    # ancestry bank. Preserve the exact return at this seam;
                    # formatting is evidence, not reconstructed serialization.
                    witness["native_report_utf8"] = result
            except Exception as error:
                self.recorder.errors.append(type(error).__name__)
                if original_session is not None:
                    original_session.close()
                    raise
        if original_session is not None:
            try:
                guard()
            except BaseException:
                original_session.close()
                raise
        return result


class ReferenceInputWorld:
    """Capture the owned initial world when _trajectory first consumes it."""
    def __init__(self, world, draw):
        self.world, self.draw = world, draw
        self.observed = False

    def __getattr__(self, name):
        return getattr(self.world, name)

    def frame(self):
        if not self.observed:
            owned = self.world
            while hasattr(owned, "world"):
                owned = owned.world
            require(not owned.closed and owned.env._search_snapshot_permitted is True
                and self.draw["status"] == "ROOT_VALIDATED", "input snapshot is not an owned sampled root")
            snapshot = owned.env.snapshot()
            # Include replay, full belief bookkeeping, annotation trackers,
            # formats and terminal state as well as simulator PP/PRNG. Only
            # outer ownership handles are explicitly excluded as metadata.
            payload = dict(snapshot=_snapshot_payload(snapshot), subject=owned.subject)
            self.draw["consumed_world_input"] = dict(input_sha256=digest(payload),
                consumed=True, snapshot_fields_excluded=[],
                ownership_metadata_excluded=list(OWNERSHIP_METADATA),
                owned_hypothetical_only=True, runtime_qualified=False)
            self.observed = True
        return self.world.frame()

    def evaluate(self, *args):
        require(self.observed, "reference evaluator consumed an unbound initial world")
        return self.world.evaluate(*args)

    def advance(self, *args):
        require(self.observed, "reference transition consumed an unbound initial world")
        return self.world.advance(*args)

    def close(self):
        self.world.close()


@dataclass(frozen=True)
class WorldInputWorkerFactory:
    base: object
    expected_public_request_sha256: str | None = None

    def __call__(self, index):
        return WorldInputRuntime(self.base(index),
            expected_public_request_sha256=self.expected_public_request_sha256, worker_index=index)


class WorldInputRuntime:
    def __init__(self, base, *, expected_public_request_sha256=None, worker_index=None):
        self.base = base
        require(expected_public_request_sha256 is None or (
            type(expected_public_request_sha256) is str and len(expected_public_request_sha256) == 64
            and all(c in "0123456789abcdef" for c in expected_public_request_sha256)
            and type(worker_index) is int and 0 <= worker_index < 20),
            "full reference worker request requires prospective digest and actual index")
        self.expected_public_request_sha256 = expected_public_request_sha256
        self.worker_index = worker_index

    def _public_input(self, request):
        return request

    def _wrap_sample_world(self, original, sample):
        return sample

    def prepare(self, request):
        public = self._public_input(request)
        witness = None
        if self.expected_public_request_sha256 is not None:
            witness = public_request_witness(public)
            require(witness["public_request_sha256"] == self.expected_public_request_sha256,
                "reference worker consumed a different full public request")
        prepared = self.base.prepare(request)
        if witness is not None:
            require(digest(public_request_witness(self._public_input(request))) == digest(witness),
                "reference public request changed during worker preparation")
        from .search_over_raw_atomic_work import ReferenceAtomicWork
        require(prepared.atomic_work is None, "existing atomic work hook cannot be overwritten")
        atomic_work = ReferenceAtomicWork(_binding(public))
        runtime = self.base
        while not hasattr(runtime, "prepared_factory") and hasattr(runtime, "base"):
            runtime = runtime.base
        factory = runtime.prepared_factory
        require(not factory.active and not factory.receipts, "fresh world-input factory required")
        def sample(rng):
            before = len(factory.receipts)
            world = prepared.sample_world(rng)
            try:
                owned = world
                while hasattr(owned, "world"):
                    owned = owned.world
                require(len(factory.receipts) == before + 1
                    and factory.receipts[-1]["ordinal"] == before
                    and owned.env is factory.env and owned.subject == factory.state.player_id,
                    "actual sampled-world receipt identity drift")
                return ReferenceInputWorld(world, factory.receipts[-1])
            except BaseException as primary:
                try:
                    world.close()
                except BaseException as cleanup:
                    primary.add_note("Reference input world cleanup also failed: "+type(cleanup).__name__)
                raise
        def evidence():
            result = prepared.evidence()
            require(type(result) is dict and "atomic_work" not in result, "atomic evidence collision")
            result["atomic_work"] = atomic_work.evidence(result["draws"])
            if witness is not None:
                require(digest(public_request_witness(self._public_input(request))) == digest(witness),
                    "reference public request changed after worker preparation")
                require("worker_public_request_preparation" not in result,
                    "reference full request evidence collision")
                result["worker_public_request_preparation"] = dict(
                    worker=self.worker_index, witness=deepcopy(witness))
            return result
        return replace(prepared, sample_world=self._wrap_sample_world(prepared.sample_world,sample),
            evidence=evidence, atomic_work=atomic_work)

    def close(self):
        self.base.close()


def reference_manifest(evidence, binding):
    rows, seen = [], set()
    for receipt in evidence["worker_receipts"]:
        worker = receipt["worker"]
        require(type(worker) is int and 0 <= worker < 20, "invalid reference worker")
        for draw in receipt["evidence"]["draws"]:
            ordinal = draw["ordinal"]
            require(type(ordinal) is int and ordinal >= 0 and (worker, ordinal) not in seen,
                "duplicate or invalid reference world ordinal")
            seen.add((worker, ordinal))
            witness = draw.get("consumed_world_input")
            consumed = witness is not None
            require(not consumed or (draw["status"] == "ROOT_VALIDATED" and draw.get("released") is True
                and witness.get("consumed") is True and witness.get("snapshot_fields_excluded") == []
                and witness.get("ownership_metadata_excluded") == OWNERSHIP_METADATA
                and witness.get("owned_hypothetical_only") is True
                and witness.get("runtime_qualified") is False), "invalid reference consumption witness")
            require(draw["status"] != "ROOT_VALIDATED" or consumed,
                "validated reference world lacks actual consumption witness")
            require(draw["status"] in {"ROOT_VALIDATED", "DEADLINE_CANCELLED", "REFUSED", "STARTED"},
                "unknown reference world status")
            if consumed:
                sha = witness.get("input_sha256")
                require(type(sha) is str and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha),
                    "invalid actual reference input digest")
            rows.append(dict(worker=worker, ordinal=ordinal, status=draw["status"], consumed=consumed,
                input_sha256=witness["input_sha256"] if consumed else None))
    for worker in {row["worker"] for row in rows}:
        ordinals = sorted(row["ordinal"] for row in rows if row["worker"] == worker)
        require(ordinals == list(range(len(ordinals))), "reference world ordinal gap")
    return _manifest("reference", binding, sorted(rows, key=lambda r: (r["worker"], r["ordinal"])))


def compare_world_inputs(left, right):
    """Report all coordinates, never cherry-pick a matched common prefix."""
    fields = ("schema", "arm", "root_binding", "subject", "set_source_hash", "information_key")
    require(all(left.get(key) == right.get(key) for key in fields) and left.get("schema") == SCHEMA
        and left.get("arm") in {"incumbent", "reference"}, "world comparison root/arm drift")
    def indexed(manifest):
        require(manifest.get("snapshot_fields_excluded") == []
            and manifest.get("ownership_metadata_excluded") == (
                OWNERSHIP_METADATA if manifest["arm"] == "reference" else [])
            and manifest.get("cross_engine_equivalence") is False
            and manifest.get("qualifies_uninstrumented_runtime") is False, "world scope drift")
        result = {}
        for row in manifest["rows"]:
            key = (row["worker"], row["ordinal"])
            require(all(type(v) is int and v >= 0 for v in key) and key not in result
                and type(row["consumed"]) is bool, "invalid world coordinate")
            if row["consumed"]:
                sha = row.get("input_sha256")
                require(type(sha) is str and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha),
                    "consumed world has no input digest")
            result[key] = row
        return result
    lrows, rrows = indexed(left), indexed(right)
    rows = []
    for worker, ordinal in sorted(lrows.keys() | rrows.keys()):
        lrow, rrow = lrows.get((worker, ordinal)), rrows.get((worker, ordinal))
        both = lrow is not None and rrow is not None and lrow["consumed"] and rrow["consumed"]
        result = ("MATCH" if lrow["input_sha256"] == rrow["input_sha256"] else "MISMATCH") if both else "UNCERTAIN"
        rows.append(dict(worker=worker, ordinal=ordinal, comparison=result))
    return dict(schema="pokezero.search-over-raw.world-input-comparison.v1", arm=left["arm"], rows=rows,
        matched=sum(row["comparison"] == "MATCH" for row in rows),
        mismatched=sum(row["comparison"] == "MISMATCH" for row in rows),
        uncertain=sum(row["comparison"] == "UNCERTAIN" for row in rows),
        entire_population_matched=bool(rows) and all(row["comparison"] == "MATCH" for row in rows),
        matched_input_scope="initial consumed world inputs within one arm",
        equal_work_claim=False, cross_engine_equivalence=False, runtime_qualified=False)


def _native_atomic_reader(*, compact, timed):
    """Choose from prospective authority, never from a returned report schema."""
    require(type(compact) is bool and type(timed) is bool, "explicit native encoding scope required")
    if compact:
        from .search_over_raw_native_atomic_compact import (
            validate_compact_native_report, validate_timed_compact_native_report)
        return validate_timed_compact_native_report if timed else validate_compact_native_report
    from .search_over_raw_native_atomic_work import validate_native_report, validate_timed_native_report
    return validate_timed_native_report if timed else validate_native_report


def native_atomic_runtime_scopes(*, compact=False):
    """Prospective format scopes shared by recipe and adapter construction."""
    require(type(compact) is bool, "explicit native runtime encoding required")
    from .search_over_raw_native_atomic_work import SCHEMA as FULL_SCHEMA
    if compact:
        from .search_over_raw_native_atomic_compact import SCHEMA as COMPACT_SCHEMA
    return dict(schema=COMPACT_SCHEMA if compact else FULL_SCHEMA,
        scope="observed_float32_backup_consistency; not independent terminal labels",
        instrumentation_can_change_deadlines=True, runtime_qualified=False), dict(
        schema="pokezero.search-over-raw.timed-native-prefix.compact-v2" if compact
            else "pokezero.search-over-raw.timed-native-prefix.v1",
        native_budget_abi_slot=24, zero_work_not_scored=True,
        on_time_deployability_established=False, runtime_qualified=False)


def _native_encoding_scope(manifest, compact):
    require(type(compact) is bool, "explicit native encoding scope required")
    calls = manifest.get("calls", [])
    require(type(calls) is list and all(type(call) is dict for call in calls),
        "native call encoding roster missing")
    if compact:
        from .search_over_raw_native_atomic_compact import SCHEMA as COMPACT_SCHEMA
        require(manifest.get("native_atomic_encoding") == COMPACT_SCHEMA,
            "prospective compact native manifest encoding drift")
        require(all(call.get("compact_atomic_observation") is True for call in calls),
            "compact native call observation absent")
    else:
        require("native_atomic_encoding" not in manifest
            and all("compact_atomic_observation" not in call for call in calls),
            "compact native observation lacks prospective admission")


def validate_timed_native_inputs(evidence, *, binding, iterations_per_world, batch_size,
                                expected_worlds, nominal_seconds, adapter_elapsed_seconds,
                                compact_native_atomic_work=False):
    """Independent deadline-prefix reader, never fixed-work or qualification.

    Accepts only a normal, reconciled input observer return. Operational
    failures have no readout here: missing/malformed evidence raises, rather
    than manufacturing an unqualified zero-work measurement. Ownership must
    still be established by a separate prospective run reader.
    """
    reader = _native_atomic_reader(compact=compact_native_atomic_work, timed=True)
    sha = lambda v: type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
    finite = lambda v: type(v) in (int, float) and math.isfinite(v) and v >= 0
    require(type(evidence) is dict and type(binding) is dict and set(binding) == {
        "root_binding", "subject", "set_source_hash", "information_key"}
        and all(type(v) is int and v > 0 for v in (iterations_per_world, batch_size, expected_worlds))
        and finite(nominal_seconds) and nominal_seconds > 0 and finite(adapter_elapsed_seconds),
        "prospective native timed input/allocation scope required")
    manifest = evidence.get("consumed_world_inputs")
    require(type(manifest) is dict and manifest.get("arm") == "incumbent"
        and digest({k: manifest.get(k) for k in binding}) == digest(binding)
        and manifest.get("schema") == SCHEMA and manifest.get("initial_inputs_only") is True
        and manifest.get("instrumentation_can_change_deadlines") is True
        and manifest.get("runtime_qualified") is False, "timed native manifest binding drift")
    compare_world_inputs(manifest, manifest)  # Validates all coordinates, not a matched subset.
    rows, calls = manifest["rows"], manifest.get("calls")
    observed = manifest.get("timed_atomic_observation")
    require(type(rows) is list and len(rows) <= expected_worlds and type(calls) is list
        and type(observed) is dict and observed.get("hook_errors") == []
        and observed.get("complete_counter_reconciliation") is True
        and observed.get("normal_native_returns") is True and observed.get("runtime_qualified") is False,
        "timed native normal-completion evidence missing")
    _native_encoding_scope(manifest, compact_native_atomic_work)
    for ordinal, row in enumerate(rows):
        require(type(row) is dict and type(row.get("worker")) is int and row["worker"] == 0
            and type(row.get("ordinal")) is int and row["ordinal"] == ordinal
            and row.get("status") == "CONSTRUCTED" and type(row.get("consumed")) is bool
            and all(sha(row.get(k)) for k in ("draw_sha256", "constructed_sha256", "state_sha256",
                "native_state_utf8_sha256")), "timed native attempted roster is refused or incomplete")
    covered, completed, searched, readouts = [], 0, 0, []
    for ordinal, call in enumerate(calls):
        require(type(call) is dict and type(call.get("call_ordinal")) is int
            and call["call_ordinal"] == ordinal and call.get("status") == "RETURNED"
            and call.get("deadline_limited") is True, "timed native call did not return normally")
        indices = call.get("construction_ordinals")
        require(type(indices) is list and indices and all(type(i) is int and 0 <= i < len(rows) for i in indices)
            and len(set(indices)) == len(indices) and not set(indices) & set(covered)
            and type(call.get("multiplicity")) is int and call["multiplicity"] == len(indices)
            and type(call.get("requested_iterations")) is int
            and call["requested_iterations"] == iterations_per_world*len(indices)
            and all(rows[i]["consumed"] is True and type(rows[i].get("call_ordinal")) is int
                and rows[i]["call_ordinal"] == ordinal and rows[i].get("input_sha256") == call.get("input_sha256")
                for i in indices) and sha(call.get("input_sha256"))
            and len({rows[i]["constructed_sha256"] for i in indices}) == 1
            and len({rows[i]["state_sha256"] for i in indices}) == 1,
            "timed native multiplicity/coverage drift")
        params, report, raw = call.get("native_call_parameters"), call.get("native_report"), call.get("native_report_utf8")
        require(type(params) is dict and set(params) == {
            "state_sha256", "requested_iterations", "batch_size", "chance_seed", "time_budget_ms"}
            and type(params["requested_iterations"]) is int and params["requested_iterations"] == call["requested_iterations"]
            and type(params["batch_size"]) is int and params["batch_size"] == batch_size
            and all(rows[i]["native_state_utf8_sha256"] == params["state_sha256"] for i in indices)
            and type(report) is dict and digest(report) == call.get("report_sha256")
            and type(raw) is str and digest(json.loads(raw)) == digest(report)
            and hashlib.sha256(raw.encode()).hexdigest() == call.get("report_utf8_sha256"),
            "timed native actual-call/raw report drift")
        result = reader(report, **params)
        require(digest(result) == digest(call.get("atomic_work_validation"))
            and type(call.get("completed_iterations")) is int
            and call["completed_iterations"] == result["completed_iterations"],
            "timed native fresh arithmetic readout drift")
        completed += result["completed_iterations"]
        searched += len(indices) if result["completed_iterations"] > 0 else 0
        covered.extend(indices); readouts.append(result)
    require(all(row["consumed"] == (i in covered) for i, row in enumerate(rows))
        and type(evidence.get("total_iterations")) is int and evidence["total_iterations"] == completed
        and all(type(observed.get(k)) is int and observed[k] == v
            for k, v in dict(attempted=len(rows), constructed=len(rows), searched=searched).items()),
        "timed native full roster/counter drift")
    budget = evidence.get("engine_mcts", {}).get("time_budget")
    require(type(budget) is dict and budget.get("scope") == "whole_model_decision"
        and type(budget.get("requested_ms")) is int and budget["requested_ms"] == round(nominal_seconds*1000)
        and type(budget.get("native_batch_guard_ms")) is int and budget["native_batch_guard_ms"] == 64
        and all(finite(budget.get(k)) for k in ("deadline_elapsed_ms", "deadline_overshoot_ms"))
        and type(budget.get("exhausted")) is bool and type(budget.get("worlds_budget_skipped")) is int
        and budget["worlds_budget_skipped"] == len(rows)-searched
        and abs(budget["deadline_overshoot_ms"]-max(0.,budget["deadline_elapsed_ms"]-budget["requested_ms"])) <= .001
        and budget["deadline_elapsed_ms"] <= adapter_elapsed_seconds*1000+.001,
        "timed native whole-decision clock/skip drift")
    invocations = budget.get("native_invocations")
    require(type(invocations) is list and len(invocations) == len(calls), "timed native invocation roster drift")
    for call, result, invocation in zip(calls, readouts, invocations):
        expected = dict(status="completed", world_seed=call["native_call_parameters"]["chance_seed"],
            multiplicity=call["multiplicity"], requested_iterations=call["requested_iterations"],
            completed_iterations=result["completed_iterations"], remaining_iterations=result["remaining_iterations"],
            time_budget_ms=result["native_budget_ms"], time_budget_elapsed_ms=result["native_elapsed_ms"],
            time_budget_batch_overshoot_ms=result["native_overshoot_ms"], time_budget_exhausted=result["exhausted"],
            root_visits=dict(side_one=result["completed_iterations"], side_two=result["completed_iterations"]))
        require(digest(invocation) == digest(expected), "timed native engine/actual return drift")
    unfinished = len(covered) < expected_worlds or completed < expected_worlds*iterations_per_world
    require(not unfinished or budget["exhausted"] is True,
        "unfinished native population lacks deadline exhaustion")
    # The engine reserves 64 ms before dispatch. A native deadline or a
    # guarded skip may therefore exhaust usable work BEFORE the nominal wall.
    exhaustion_reason = bool(budget["worlds_budget_skipped"] or any(r["exhausted"] for r in readouts))
    require((not exhaustion_reason or budget["exhausted"] is True)
        and (not budget["exhausted"] or exhaustion_reason
            or budget["deadline_elapsed_ms"]+.001 >= budget["requested_ms"])
        and (budget["exhausted"] or budget["deadline_elapsed_ms"] <= budget["requested_ms"]+.001)
        and sum(r["native_elapsed_ms"] for r in readouts) <= budget["deadline_elapsed_ms"]+.001*(len(readouts)+1)
        and all(r["native_budget_ms"] <= budget["requested_ms"]-64+1 for r in readouts),
        "whole-decision guard/exhaustion or serial native clock drift")
    return dict(schema="pokezero.search-over-raw.timed-native-input-readout.v1",
        expected_worlds=expected_worlds, attempted_worlds=len(rows), dispatched_worlds=len(covered),
        searched_worlds=searched, not_attempted_worlds=expected_worlds-len(rows),
        constructed_not_dispatched_worlds=len(rows)-len(covered), completed_iterations=completed,
        native_calls=readouts, nominal_seconds=nominal_seconds, adapter_elapsed_seconds=adapter_elapsed_seconds,
        adapter_overshoot_seconds=max(0., adapter_elapsed_seconds-nominal_seconds),
        status="UNQUALIFIED_AT_CEILING" if completed == 0 else "OBSERVED_WORK_NOT_QUALIFIED",
        readout_scope="normal-return prefix structure; separate owner and selector validation required",
        owning_process_validated=False, on_time_deployability_established=False,
        runtime_qualified=False, scoring_authorized=False, scientific_collection_authorized=False)


def validate_fixed_work_inputs(evidence, *, arm, binding, statistics_root_id, construction_required=False,
                               expected_public_request_sha256=None, native_public_fold_required=False,
                               compact_native_atomic_work=False):
    """Observed input and commit reconciliation before result persistence.

    The actual adapter observes dispatch/return and owned first-use seams. This
    independent reader checks the complete fixed allocation, reference OWN
    commits/exports/cleanup and native float32 atomic backups. Independent
    terminal-label truth and runtime qualification remain separate.
    """
    require(arm in {"incumbent", "reference"} and type(evidence) is dict
        and type(statistics_root_id) is str and bool(statistics_root_id)
        and type(binding) is dict and set(binding) == {
            "root_binding", "subject", "set_source_hash", "information_key"},
        "fixed-work input evidence required")
    require(type(compact_native_atomic_work) is bool and (not compact_native_atomic_work
        or arm == "incumbent" and not native_public_fold_required),
        "prospective compact fixed-work scope required; existing public-fold lane remains v1")
    manifest = evidence.get("consumed_world_inputs")
    require(type(manifest) is dict and all(manifest.get(key) == value for key, value in binding.items())
        and manifest.get("arm") == arm, "diagnostic consumed-input root binding drift")
    _native_encoding_scope(manifest, compact_native_atomic_work)
    comparison = compare_world_inputs(manifest, manifest)
    require(comparison["entire_population_matched"] and manifest.get("initial_inputs_only") is True
        and manifest.get("instrumentation_can_change_deadlines") is True
        and manifest.get("runtime_qualified") is False,
        "diagnostic lacks complete consumed-input witnesses")
    rows = manifest["rows"]
    require(type(native_public_fold_required) is bool and (not native_public_fold_required or
        arm == "incumbent" and construction_required and expected_public_request_sha256 is not None),
        "prospective native public-fold validation scope required")
    if native_public_fold_required:
        from .search_over_raw_public_fold import read_construction_public_fold
        for ordinal, row in enumerate(rows):
            read_construction_public_fold(row.get("public_fold_capture"), ordinal=ordinal,
                construction_receipt=row["construction_receipt"], consumed_binding=binding,
                public_request_sha256=expected_public_request_sha256)
    else:
        require(all("public_fold_capture" not in row for row in rows),
            "unreviewed public-fold capture cannot enter ordinary fixed-work evidence")
    require(len(rows) == (4 if arm == "incumbent" else 80), "diagnostic input allocation incomplete")
    if expected_public_request_sha256 is not None:
        require(type(expected_public_request_sha256) is str
            and len(expected_public_request_sha256) == 64
            and all(c in "0123456789abcdef" for c in expected_public_request_sha256),
            "prospective public request digest required")
        if arm == "reference":
            from .search_over_raw_timed_reference import validate_reference_public_request
            # The request law applies to fixed work too; no timed-prefix or
            # ceiling inference enters this independent twenty-worker check.
            validate_reference_public_request(evidence,binding=binding,expected_sha256=expected_public_request_sha256)
        else:
            observations = manifest.get("public_request_observations")
            require(type(observations) is list and len(observations) == 4,
                "full native public request observation roster required")
            expected_witness = dict(schema="pokezero.search-over-raw.public-request-consumption.v1",
                public_request_sha256=expected_public_request_sha256, consumed_binding=binding,
                full_supported_public_state_and_pending=True, private_source_snapshot_included=False,
                runtime_qualified=False)
            for ordinal, row in enumerate(observations):
                require(type(row) is dict and set(row) == {"ordinal", "witness"}
                    and type(row["ordinal"]) is int and row["ordinal"] == ordinal
                    and digest(row["witness"]) == digest(expected_witness),
                    "actual native public request/pending transport drift")
    if arm == "reference":
        from .search_over_raw_atomic_work import validate_reference_atomic_work
        validate_reference_atomic_work(evidence, binding=binding,
            battle_id="search-over-raw-root:" + statistics_root_id)
        require(digest(reference_manifest(evidence, binding)) == digest(manifest),
            "reference consumed-input manifest differs from actual atomic draw receipts")
        require({(r["worker"], r["ordinal"]) for r in rows} == {
            (worker, ordinal) for worker in range(20) for ordinal in range(4)}
            and all(r["status"] == "ROOT_VALIDATED" for r in rows)
            and type(evidence.get("result", {}).get("trajectories")) is int
            and evidence["result"]["trajectories"] == 80, "reference fixed input/work allocation drift")
    else:
        require(type(construction_required) is bool,"external construction scope required")
        if construction_required or manifest.get("construction_observed") is True:
            from .search_over_raw_construction import construction_source_hashes,read_native_construction
            require(manifest.get("construction_observed") is True,"prospective native job omitted construction observations")
            for row in rows:
                read_native_construction(row.get("construction_receipt"),binding=binding,ordinal=row["ordinal"],
                    draw_sha256=row["draw_sha256"],constructed_sha256=row["constructed_sha256"],
                    state_utf8_sha256=row["native_state_utf8_sha256"],source_hashes=construction_source_hashes())
        calls = manifest.get("calls")
        require(type(calls) is list and bool(calls) and len(calls) <= 4
            and {(r["worker"], r["ordinal"]) for r in rows} == {(0, i) for i in range(4)}
            and all(r["status"] == "CONSTRUCTED" for r in rows), "native input roster drift")
        covered = []
        indexed = {row["ordinal"]: row for row in rows}
        for ordinal, call in enumerate(calls):
            indices = call.get("construction_ordinals")
            require(type(indices) is list and indices and all(type(i) is int and 0 <= i < 4 for i in indices)
                and type(call.get("call_ordinal")) is int and call["call_ordinal"] == ordinal
                and type(call.get("multiplicity")) is int and call["multiplicity"] == len(indices)
                and type(call.get("requested_iterations")) is int
                and call["requested_iterations"] == 16 * len(indices)
                and type(call.get("completed_iterations")) is int
                and call["completed_iterations"] == call["requested_iterations"]
                and call.get("deadline_limited") is False and call.get("status") == "RETURNED"
                and type(call.get("report_sha256")) is str and len(call["report_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in call["report_sha256"]),
                "native fixed allocation lacks actual full return witness")
            require(all(indexed[i].get("call_ordinal") == ordinal
                and indexed[i]["input_sha256"] == call.get("input_sha256") for i in indices),
                "native input row/call binding drift")
            reader = _native_atomic_reader(compact=compact_native_atomic_work, timed=False)
            parameters, native_report = call.get("native_call_parameters"), call.get("native_report")
            raw_report = call.get("native_report_utf8")
            require(type(parameters) is dict and set(parameters) == {
                "state_sha256", "requested_iterations", "batch_size", "chance_seed"}
                and type(native_report) is dict and digest(native_report) == call["report_sha256"]
                and type(raw_report) is str and digest(json.loads(raw_report)) == digest(native_report)
                and hashlib.sha256(raw_report.encode("utf-8")).hexdigest() == call.get("report_utf8_sha256")
                and parameters["requested_iterations"] == call["requested_iterations"]
                and parameters["batch_size"] == 16
                and all(indexed[i].get("native_state_utf8_sha256") == parameters["state_sha256"] for i in indices),
                "native atomic actual-call/input binding drift")
            validated = reader(native_report, **parameters)
            require(validated == call.get("atomic_work_validation")
                and validated["completed_iterations"] == call["completed_iterations"],
                "native atomic validation receipt drift")
            covered.extend(indices)
        require(sorted(covered) == list(range(4))
            and type(evidence.get("total_iterations")) is int and evidence["total_iterations"] == 64,
            "native fixed input/work coverage drift")
    return digest(manifest)


class _WorldInputAdapterMixin:
    """Explicit prospective adapter runtime; legacy selectors stay unchanged."""
    _ancestry_contract = None
    _capture_native_public_fold = False
    _native_input_type = NativeWorldInputs

    def _check_world_input_scope(self, configuration):
        require(configuration.arm in {"incumbent", "reference"}
            and configuration.belief == "public", "world-input adapter is public search only")

    def _native_input_options(self, request):
        return {}

    def __init__(self, configuration, *, native_ancestry_contract=None,
                 expected_public_request_sha256=None, timed_native_atomic_work=False,
                 timed_reference_atomic_work=False, capture_native_public_fold=False,
                 compact_native_atomic_work=False, **kwargs):
        self._check_world_input_scope(configuration)
        if native_ancestry_contract is not None:
            from .search_over_raw_ancestry import NativeAncestryContract
            require(type(native_ancestry_contract) is NativeAncestryContract
                and configuration.arm == "incumbent" and configuration.leaf == "model"
                and isinstance(self, FixedWorkDiagnosticAdapter) and kwargs.get("value_diagnostics") is None,
                "prospective ancestry requires fixed-work incumbent model only; no label session")
        self._ancestry_contract = native_ancestry_contract
        require(type(capture_native_public_fold) is bool and (not capture_native_public_fold or
            native_ancestry_contract is not None and expected_public_request_sha256 is not None),
            "public fold capture requires reviewed fixed-model ancestry/full-request scope")
        self._capture_native_public_fold = capture_native_public_fold
        require(type(timed_native_atomic_work) is bool and (not timed_native_atomic_work or (
            configuration.arm == "incumbent" and configuration.leaf in {"model", "hp_fraction"}
            and not isinstance(self, FixedWorkDiagnosticAdapter) and kwargs.get("value_diagnostics") is None)),
            "timed atomic observation is native model/HP only, not a fixed-work or label lane")
        self._timed_native_atomic_work = timed_native_atomic_work
        require(type(compact_native_atomic_work) is bool and (not compact_native_atomic_work or
            timed_native_atomic_work and configuration.belief == "public"
            and native_ancestry_contract is None and not capture_native_public_fold
            and kwargs.get("value_diagnostics") is None),
            "compact native adapter is the separate public timed model/HP lane only")
        self._compact_native_atomic_work = compact_native_atomic_work
        require(type(timed_reference_atomic_work) is bool and (not timed_reference_atomic_work or (
            configuration.arm == "reference" and configuration.leaf in {"model", "hp_fraction"}
            and configuration.seconds in (1., 3., 10.) and configuration.workers == 20
            and kwargs.get("initial_dispatch_workers", 6) == 6
            and not isinstance(self, FixedWorkDiagnosticAdapter) and kwargs.get("value_diagnostics") is None)),
            "timed reference observation requires the twenty-worker public model/HP lane")
        self._timed_reference_atomic_work = timed_reference_atomic_work
        require(expected_public_request_sha256 is None or (
            (isinstance(self, FixedWorkDiagnosticAdapter) or (configuration.arm == "incumbent"
                and timed_native_atomic_work) or timed_reference_atomic_work)
            and type(expected_public_request_sha256) is str and len(expected_public_request_sha256) == 64
            and all(c in "0123456789abcdef" for c in expected_public_request_sha256)),
            "full public request observation requires a prospective timed/fixed observation lane")
        self._expected_public_request_sha256 = expected_public_request_sha256
        super().__init__(configuration, **kwargs)
        self.runtime_configuration["consumed_world_inputs"] = dict(schema=SCHEMA,
            initial_inputs_only=True, snapshot_fields_excluded=[],
            ownership_metadata_excluded=OWNERSHIP_METADATA if configuration.arm == "reference" else [],
            instrumentation_can_change_deadlines=True, qualifies_uninstrumented_runtime=False)
        if configuration.arm == "reference":
            from .search_over_raw_atomic_work import SCHEMA as ATOMIC_SCHEMA
            self.runtime_configuration["reference_atomic_work"] = dict(schema=ATOMIC_SCHEMA,
                scope="actual_own_backup_and_cleanup; not independent simulator labels",
                instrumentation_can_change_deadlines=True, runtime_qualified=False)
            if timed_reference_atomic_work:
                self.runtime_configuration["timed_reference_atomic_work"] = dict(
                    schema="pokezero.search-over-raw.timed-reference-prefix.v1",
                    zero_work_not_scored=True, fresh_single_attempt=True,
                    on_time_deployability_established=False, runtime_qualified=False)
        elif hasattr(self, "fixed_work") or timed_native_atomic_work:
            atomic_scope, timed_scope = native_atomic_runtime_scopes(compact=compact_native_atomic_work)
            self.runtime_configuration["native_atomic_work"] = atomic_scope
            if compact_native_atomic_work:
                self.runtime_configuration["compact_native_atomic_work"] = True
            if timed_native_atomic_work:
                self.runtime_configuration["timed_native_atomic_work"] = timed_scope
            else:
                from .search_over_raw_construction import SCHEMA as CONSTRUCTION_SCHEMA,construction_source_hashes
                self.runtime_configuration["native_construction"] = dict(schema=CONSTRUCTION_SCHEMA,
                    source_hashes=construction_source_hashes(),private_controller_only=True,
                    instrumentation_can_change_deadlines=True,runtime_qualified=False)
        self.runtime_sha256 = digest(self.runtime_configuration)
        if expected_public_request_sha256 is not None:
            self.runtime_configuration["full_public_request_observation"] = deepcopy(
                TIMED_REFERENCE_PUBLIC_CAPTURE if configuration.arm == "reference" else TIMED_NATIVE_PUBLIC_CAPTURE
            ) if configuration.arm == "reference" or timed_native_atomic_work else dict(
                schema="pokezero.search-over-raw.public-request-consumption.v1",
                at_each_actual_native_sampling_attempt=True, pending_transition_included=True,
                private_source_snapshot_included=False, runtime_qualified=False)
            self.runtime_sha256 = digest(self.runtime_configuration)
        if self._ancestry_contract is not None:
            from .search_over_raw_ancestry import SCHEMA as ANCESTRY_SCHEMA
            self.runtime_configuration["native_searched_ancestry"] = dict(schema=ANCESTRY_SCHEMA,
                contract=asdict(self._ancestry_contract), private_controller_only=True,
                decision_deadline=None, emergency_watchdog="external_only",
                event_population="actual_model_evaluations_including_cancelled_rounds",
                labels_used_for_backup=False, instrumentation_can_change_selection=True,
                runtime_qualified=False)
            self.runtime_sha256 = digest(self.runtime_configuration)
        if self._capture_native_public_fold:
            from .search_over_raw_public_fold import runtime_receipt
            self.runtime_configuration["native_public_fold"] = runtime_receipt()
            self.runtime_sha256 = digest(self.runtime_configuration)

    def _reference_worker_factory(self, factory):
        return WorldInputWorkerFactory(super()._reference_worker_factory(factory),
            expected_public_request_sha256=self._expected_public_request_sha256)

    def _search_reference(self, request, root, statistics_root, selection_seed, started):
        self._reference_public_capture = None
        if self._expected_public_request_sha256 is not None:
            self._reference_public_capture = public_request_witness(request)
            require(self._reference_public_capture["public_request_sha256"] == self._expected_public_request_sha256,
                "reference adapter captured a different full public request")
        if not self._timed_reference_atomic_work:
            return super()._search_reference(request, root, statistics_root, selection_seed, started)
        self._timed_reference_binding = dict(battle_id="search-over-raw-root:"+statistics_root,
            selection_seed=selection_seed, absolute_deadline=started+self.configuration.seconds)
        return self._pool.search(self._diagnostic_request(request), root,
            battle_id="search-over-raw-root:"+statistics_root, seed=selection_seed,
            deadline_seconds=self.configuration.seconds, deadline_at=started+self.configuration.seconds,
            adapter_started_at=started, timed_zero_work=True)

    def _reference_zero_selection(self, measured, request, *, root_id, statistics_root_id,
                                  statistics_root, selection_seed, started):
        require(self._timed_reference_atomic_work and measured.result.trajectories == 0
            and self._value_diagnostics is None, "zero reference selection requires the explicit timed lane")
        if self.configuration.leaf != "model":
            from .search_over_raw_leaves import validate_leaf_work
            validate_leaf_work(measured, self.configuration.leaf)
        # Validate before classifying; an invalid/operational result still raises.
        evidence = self._selection_evidence(asdict(measured), request)
        require(evidence["timed_reference_readout"]["status"] == "UNQUALIFIED_AT_CEILING",
            "reference zero result is not a normally completed deadline prefix")
        self._poisoned = True
        elapsed = time.perf_counter()-started
        return dict(root_id=root_id, statistics_root_id=statistics_root_id or statistics_root,
            selection_seed=selection_seed, configuration_sha256=self.configuration.identity,
            runtime_sha256=self.runtime_sha256, action=None, status="UNQUALIFIED_AT_CEILING",
            elapsed_seconds=elapsed, **self._execution_receipt(elapsed, evidence), evidence=evidence,
            independent_root_statistics=True, scientific_strength_evidence=False,
            attempt_consumed=True, retry_authorized=False, continuation_authorized=False,
            runtime_qualified=False)

    def select(self, context, **kwargs):
        self._input_recorder = None
        self._input_restore = None
        self._ancestry_session = None
        self._selection_started_monotonic = None
        try:
            return super().select(context, **kwargs)
        except BaseException as error:
            if self._timed_native_atomic_work:
                try:
                    result = self._zero_work_unqualification(error, context, kwargs)
                    if result is not None:
                        return result
                except Exception as verification_error:
                    # Preserve the original strict error, including its native
                    # diagnostic. Verification failure is not normal zero work.
                    error.add_note("Timed unqualification verification failed: " + type(verification_error).__name__)
            if self._input_recorder is not None and self.last_failure is not None:
                self.last_failure["attempted_world_inputs"] = dict(
                    _manifest("incumbent", self._input_recorder.binding, self._input_recorder.rows),
                    calls=deepcopy(self._input_recorder.calls),
                    callback_error_types=list(self._input_recorder.errors),
                    complete_counter_reconciliation=False)
            raise
        finally:
            if self._input_restore is not None:
                policy, original, existed = self._input_restore
                if self._ancestry_session is not None:
                    self._ancestry_session.close()
                    policy._value_diagnostic_session = None
                policy._world_attempt_observer = None
                policy._world_observer = None
                policy._world_construction_observer = None
                policy._native_construction_observer = None
                if existed:
                    policy._native = original
                else:
                    del policy._native

    def _zero_work_unqualification(self, error, context, arguments):
        """Classify this consumed one-shot attempt, never rescue a selector.

        Exact strict engine exceptions only; walking arbitrary cause chains
        could hide a genuine outer operational failure behind a deadline note.
        The adapter remains poisoned and cannot retry this or another root.
        """
        from ..engine_search import EngineSearchFallbackError
        if type(error) is not EngineSearchFallbackError:
            return None
        raw = getattr(error, "engine_search_fallback_diagnostic", None)
        require(type(raw) is str, "strict deadline diagnostic absent")
        diagnostic = json.loads(raw)
        if diagnostic.get("reason") != "model_time_budget_no_completed_worlds":
            return None
        root_id = arguments["root_id"]
        statistics_root = arguments.get("statistics_root_id") or root_id
        recorder = self._input_recorder
        require(recorder is not None and recorder.record_timed_atomic_work
            and self._poisoned and type(root_id) is str and root_id in self._attempted
            and statistics_root in self._attempted_statistics_roots
            and diagnostic.get("schema") == "engine-search-fallback-refusal-v1"
            and diagnostic.get("diagnostic_only_not_policy_input") is True
            and diagnostic.get("battle_id") == "search-over-raw-root:" + statistics_root
            and diagnostic.get("seat") == context.player_id == recorder.binding["subject"]
            and digest(diagnostic.get("round")) == digest(getattr(context, "decision_round_index", None))
            and type(diagnostic.get("world_failures")) is dict and not diagnostic["world_failures"]
            and not recorder.errors and type(self._selection_started_monotonic) in (int, float),
            "strict zero-work diagnostic is not the current normal deadline attempt")
        engine = diagnostic.get("engine_mcts")
        require(type(engine) is dict and engine.get("fallback") == diagnostic["reason"],
            "strict zero-work engine diagnostic drift")
        policy = self._native
        require(type(policy.stats.total_iterations) is int and type(self._timed_work_before) is int
            and all(type(v) is int for v in self._input_before)
            and all(type(getattr(policy.stats, name)) is int for name in (
                "worlds_attempted", "worlds_constructed", "worlds_searched")),
            "strict zero-work counters have numeric aliases")
        work = policy.stats.total_iterations-self._timed_work_before
        require(type(work) is int and work == 0, "strict deadline contains productive work")
        evidence = dict(total_iterations=work, engine_mcts=deepcopy(engine))
        before = self._input_before
        evidence["consumed_world_inputs"] = recorder.finish(evidence,
            attempted=policy.stats.worlds_attempted-before[0],
            constructed=policy.stats.worlds_constructed-before[1],
            searched=policy.stats.worlds_searched-before[2])
        elapsed = time.perf_counter()-self._selection_started_monotonic
        readout = validate_timed_native_inputs(evidence, binding=recorder.binding,
            iterations_per_world=recorder.iterations_per_world, batch_size=policy._config.search_batch,
            expected_worlds=policy._config.worlds, nominal_seconds=self.configuration.seconds,
            adapter_elapsed_seconds=elapsed, compact_native_atomic_work=self._compact_native_atomic_work)
        require(readout["status"] == "UNQUALIFIED_AT_CEILING", "strict zero-work classification drift")
        result = dict(root_id=root_id, configuration_sha256=self.configuration.identity,
            **({"statistics_root_id": arguments["statistics_root_id"], "selection_seed": arguments["selection_seed"]}
                if arguments.get("statistics_root_id") is not None else {}),
            runtime_sha256=self.runtime_sha256, action=None, status="UNQUALIFIED_AT_CEILING",
            elapsed_seconds=elapsed, nominal_search_seconds=self.configuration.seconds,
            exceeded_nominal_seconds=elapsed > self.configuration.seconds,
            evidence=evidence, timed_input_readout=readout, strict_engine_diagnostic=deepcopy(diagnostic),
            original_failure_receipt=deepcopy(self.last_failure), independent_root_statistics=True,
            attempt_consumed=True, retry_authorized=False, continuation_authorized=False,
            owning_process_validated=False, runtime_qualified=False, scientific_strength_evidence=False,
            scientific_collection_authorized=False)
        self.last_unqualified = deepcopy(result)
        return result

    def _prepare_incumbent(self, request):
        super()._prepare_incumbent(request)
        policy = self._native
        if self._ancestry_contract is not None:
            require(policy._value_diagnostic_session is None, "existing private native session cannot be overwritten")
        require(policy._world_attempt_observer is None and policy._world_observer is None,
            "existing world-input hooks cannot be overwritten")
        require(getattr(policy,"_world_construction_observer",None) is None
            and getattr(policy,"_native_construction_observer",None) is None,"existing constructor hooks cannot be overwritten")
        require(policy._config.model_world_workers == 1 and not policy._config.early_stop
            and getattr(policy, "_ladder_sims_override", None) is None,
            "world-input witness requires the registered serial no-early-stop dispatcher")
        recorder = self._native_input_type(_binding(request), policy._config.search_sims,
            deadline_limited=policy._config.model_decision_time_ms is not None,
            record_atomic_work=hasattr(self, "fixed_work") or self._timed_native_atomic_work,
            record_timed_atomic_work=self._timed_native_atomic_work,
            record_construction=hasattr(self,"fixed_work"),
            record_public_fold=self._capture_native_public_fold,
            expected_public_request=public_request_witness(request)
                if self._expected_public_request_sha256 is not None else None,
            pending_transition=request.pending_transition
                if self._expected_public_request_sha256 is not None else None,
            **({"record_compact_atomic_work":True} if self._compact_native_atomic_work else {}),
            **self._native_input_options(request))
        if self._expected_public_request_sha256 is not None:
            require(recorder.expected_public_request["public_request_sha256"] == self._expected_public_request_sha256,
                "adapter full public request differs from prospective archived root")
        self._input_before = (policy.stats.worlds_attempted, policy.stats.worlds_constructed,
            policy.stats.worlds_searched)
        if self._timed_native_atomic_work:
            self._timed_work_before = policy.stats.total_iterations
        original = policy._native
        self._input_restore = (policy, original, "_native" in vars(policy))
        self._input_recorder = recorder
        policy._world_attempt_observer = recorder.attempt
        policy._world_observer = recorder.constructed
        if recorder.record_construction:
            policy._world_construction_observer = recorder.world_built
            policy._native_construction_observer = recorder.native_built
        policy._native = lambda: NativeInputHandle(original(), recorder)
        if self._ancestry_contract is not None:
            from .search_over_raw_ancestry import NativeAncestrySession
            policy.prepare_visited_value_runtime()
            self._ancestry_session = NativeAncestrySession(self._ancestry_contract,
                root_key=recorder.binding["information_key"])
            policy._value_diagnostic_session = self._ancestry_session

    def _selection_evidence(self, evidence, request):
        if self.configuration.arm == "incumbent":
            before = self._input_before
            policy = self._native
            manifest = self._input_recorder.finish(evidence,
                attempted=policy.stats.worlds_attempted-before[0],
                constructed=policy.stats.worlds_constructed-before[1],
                searched=policy.stats.worlds_searched-before[2])
            if self._ancestry_contract is not None:
                evidence = {**evidence, "native_searched_ancestry": policy._value_diagnostic_session.exports()}
        else:
            manifest = reference_manifest(evidence, _binding(request))
            if self._expected_public_request_sha256 is not None:
                require(digest(public_request_witness(request)) == digest(self._reference_public_capture),
                    "reference adapter public request changed during selection")
                evidence = {**evidence,"adapter_public_request_witness":deepcopy(self._reference_public_capture)}
            if self._timed_reference_atomic_work:
                from .search_over_raw_timed_reference import validate_reference_timed_inputs
                evidence = {**evidence, "consumed_world_inputs": manifest}
                expected = self._timed_reference_binding
                require(evidence["reference_timed_prefix"]["absolute_deadline_monotonic"] ==
                    expected["absolute_deadline"], "reference timed original adapter deadline drift")
                readout = validate_reference_timed_inputs(evidence, binding=_binding(request),
                    battle_id=expected["battle_id"], selection_seed=expected["selection_seed"],
                    nominal_seconds=self.configuration.seconds,
                    public_request_sha256=self._expected_public_request_sha256)
                evidence = {**evidence, "timed_reference_readout": readout}
                self._poisoned = True
        return super()._selection_evidence({**evidence, "consumed_world_inputs": manifest}, request)


from .search_over_raw_adapters import PublicModelSearchAdapter
from .search_over_raw_fixed_work import FixedWorkDiagnosticAdapter


class WorldInputSearchAdapter(_WorldInputAdapterMixin, PublicModelSearchAdapter):
    """Public deployment lane with measured world-input instrumentation."""


class WorldInputFixedWorkAdapter(_WorldInputAdapterMixin, FixedWorkDiagnosticAdapter):
    """Fixed-work evaluator probe with the same owned-input witnesses."""
