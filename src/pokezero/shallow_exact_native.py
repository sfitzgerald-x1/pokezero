"""Native state/history binding for bounded shallow-exact engineering pilots.

No gameplay-policy registration is provided. A fresh worker must set the exact
damage-roll environment flag before loading either engine, behaviorally verify
the native enumeration, and run under an owned-process hard deadline. The
constructor checks the flag but cannot prove a process-local Rust OnceLock was
not already initialized; the behavioral probe remains mandatory for a pilot.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import marshal
import os
import struct
import time
from typing import Any, Sequence

from .shallow_exact import ChanceBranch, ExactSearchRefused
from .shallow_leaf_rows import DecisionLeafValues, encoded_leaf_row
from .failure_capture import current, record_refusal, unavailable


class NativeRenderRefused(ExactSearchRefused):
    """The unchanged strict guard, with its actual branch witness retained."""

    def __init__(self, witness: dict[str, Any]):
        super().__init__("native event rendering is not lossless for this pilot")
        self.witness = witness


@dataclass(frozen=True)
class NativePosition:
    state_str: str
    party_species: dict[str, tuple[str, ...]]
    turn: int
    fold: Any
    encoder: Any
    self_side: str
    # Complete synthesized history relative to this encoder's original root.
    lines: tuple[str, ...] = ()
    teams_hp: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    instruction_list: tuple[str, ...] | None = None
    origin: dict[str, Any] | None = None


def validate_encoder_frame(root_inputs: str, self_side: str) -> None:
    """The native encoder chooses its frame from showdown_slot, not a label."""
    slot = json.loads(root_inputs).get("observation_metadata", {}).get("showdown_slot")
    expected = {"p1": "side_one", "p2": "side_two"}.get(slot)
    if expected is None or self_side != expected:
        raise ExactSearchRefused("fixture self_side disagrees with encoder showdown_slot")


def _branch_action_token(action: str) -> str:
    """Translate native option display strings to branch_events parser tokens."""
    if action == "No Move":
        return "none"
    return action.removeprefix("switch ")


def _same_capture_value(saved, value):
    """Exact plain-JSON equality; unsupported types take the full snapshot path.

    Python equality alone merges True/1/1.0 and signed zero, which would lose
    evidence of invalid type/value changes inside an otherwise identical world.
    """
    kind = type(value)
    if type(saved) is not kind:
        return False
    if kind is dict:
        return (all(type(k) is str for k in value) and saved.keys() == value.keys()
                and all(_same_capture_value(saved[k], v) for k, v in value.items()))
    if kind is list:
        return len(saved) == len(value) and all(_same_capture_value(a, b) for a, b in zip(saved, value))
    if kind is float:
        return math.isfinite(value) and saved.hex() == value.hex()
    if kind in (str, int, bool, type(None)):
        return saved == value
    return False


def _same_capture_snapshot(saved, value):
    """Conservatively compare against a detached DecisionCapture snapshot.

    Container snapshots contain built-in JSON values, with nonfinite floats
    represented by explicit marker dictionaries. Marshal's C encoder preserves
    scalar/container types and float bits, so equality cannot merge True/1/1.0,
    signed zero, tuple/list or changed mapping keys. It never loads or executes
    a serialized object. Unsupported values take the ordinary snapshot path.

    Format 2 deliberately avoids object-identity references, so separately
    allocated equal inputs still match. Mapping-order differences may return
    false for equal JSON: that merely refreshes the detached snapshot. Serialize
    BOTH sides every time; no cache can hide mutation. The saved side must come
    from DecisionCapture. No serialized bytes are loaded or written to evidence.
    """
    if type(saved) is not type(value):
        return False
    if type(value) not in (dict, list):
        return _same_capture_value(saved, value)
    try:
        return marshal.dumps(saved, 2) == marshal.dumps(value, 2)
    except (ValueError, TypeError, OverflowError):
        return False


def _drain_model_transport(model):
    """Optional diagnostics never replace a model result or its exception."""
    try:
        drain = getattr(model, 'drain_transport_timings', None)
        if not callable(drain):
            return dict(status='UNAVAILABLE_API')
        values = drain()
        if (type(values) not in (tuple, list) or len(values) != 4
                or any(type(values[i]) is not int or values[i] < 0 for i in (0, 2))
                or any(type(values[i]) not in (int, float)
                       or not math.isfinite(values[i]) or values[i] < 0 for i in (1, 3))):
            return dict(status='UNAVAILABLE_INVALID_COUNTERS')
        return dict(status='MEASURED',packed_decode_calls=values[0],packed_decode_seconds=values[1],
                    native_tensor_forward_output_calls=values[2],native_tensor_forward_output_seconds=values[3])
    except Exception as error:
        return dict(status='UNAVAILABLE_ERROR',error_type=type(error).__name__)


class NativeExactBackend:
    """Rust transitions/CPU leaf encode plus an existing batched native model.

    Fold objects are cloned through their payload for every chance successor.
    Neither shared mutable histories nor state-string-only deduplication is safe.
    Values are converted from the encoder's self-seat frame to side-one tanh.
    Any lossy/unsafe render refuses this strict engineering pilot; the normal
    incumbent policy and its historical acceptance semantics are untouched.

    The Python event/leaf seam does not carry the production search's private
    typed active-status transitions. This is not a broad leaf-byte-parity
    certificate. The reviewed cost domain is initially status-free Earthquake
    versus Recover, no switches/status-affecting abilities, items or history.
    Broader mechanics require an explicit congruence check or typed seam first.

    Opt-in leaf memoization keys the five complete typed forward inputs and
    fixed model binding, after constructing every fold/history and encoding
    every leaf. It resets per decision and changes no tree expansion. Forward
    batch cardinalities may change floating-point kernel roundoff, so native
    numerical parity still needs a declared tolerance; this is not a bitwise
    cross-batch arithmetic claim.

    Separate default-off packed transport retains every encode and batch. It
    passes the encoder's canonical i32/f64/i16/bool little-endian bytes to the
    additive model API, which performs the SAME effective i64/f32/i64/bool/bool
    conversions before the unchanged forward. It is not a new encoder, does
    not compose with memoization in this pilot, and rejects malformed rows.
    Explicit bulk_encode_workers additionally uses the same native CPU encoder
    over contiguous same-encoder runs, preserving every state/fold/line/turn and
    value frame. Missing APIs refuse instead of silently falling back to serial.
    """

    def __init__(self, native: Any, model: Any, *, window: int = 1,
                 max_outcomes_per_call: int = 100_000, memoize_leaves: bool = False,
                 packed_leaves: bool = False,
                 bulk_encode_workers: int | None = None,
                 model_sha256: str | None = None, observation_spec: dict | None = None):
        if os.environ.get("POKEZERO_ENUMERATE_ROLLS") != "1":
            raise ExactSearchRefused("fresh worker must opt into exact damage rolls before engine load")
        if window != 1:
            raise ExactSearchRefused("native pilot supports the champion one-window contract only")
        if type(max_outcomes_per_call) is not int or max_outcomes_per_call <= 0:
            raise ExactSearchRefused("positive native outcome-call limit required")
        if type(memoize_leaves) is not bool:
            raise ExactSearchRefused("leaf memoization must be an explicit boolean")
        if type(packed_leaves) is not bool:
            raise ExactSearchRefused("packed leaves must be an explicit boolean")
        if packed_leaves and memoize_leaves:
            raise ExactSearchRefused("packed pilot does not compose with leaf memoization")
        if bulk_encode_workers is not None and (type(bulk_encode_workers) is not int
                or not 1 <= bulk_encode_workers <= 32 or not packed_leaves):
            raise ExactSearchRefused("bulk encode requires packed leaves and 1..32 explicit workers")
        if packed_leaves and not callable(getattr(model, "eval_obs_packed", None)):
            raise ExactSearchRefused("packed pilot requires the additive native model API")
        self._row_spec = dict(observation_spec or {})
        if (memoize_leaves or packed_leaves) and (set(self._row_spec) != {
                "window", "tokens", "categorical_features", "numeric_features"}
                or any(type(value) is not int or value <= 0 for value in self._row_spec.values())
                or self._row_spec["window"] != window):
            raise ExactSearchRefused("memoized/packed leaves require the fixed complete observation spec")
        self.native, self.model = native, model
        self.packed_leaves = packed_leaves
        self.bulk_encode_workers = bulk_encode_workers
        self._leaf_values = DecisionLeafValues(model, model_sha256=model_sha256) if memoize_leaves else None
        self._model_sha256 = model_sha256
        self.max_outcomes_per_call = max_outcomes_per_call
        self.render_refusals: Counter[str] = Counter()
        self.render_subcases: Counter[str] = Counter()
        self.stage_seconds: Counter[str] = Counter()
        self.encode_calls = 0
        self.forward_batches = 0
        self.unique_leaf_evaluations = 0
        self.leaf_cache_hits = 0
        self.capture_context_updates = 0
        self.captured_leaf_rows = 0
        self.model_call_timings = []

    def _capture_position(self, state):
        capture = current()
        if capture:
            started = time.perf_counter()
            dynamic = {"row_action", "column_action", "branch_index",
                       "generating_pre_state", "generating_turn"}
            origin = {}
            for key, value in (state.origin or {}).items():
                # The saved origin is mostly identical for every successor in
                # one world. Compare against the detached snapshot, so changes
                # (including in-place mutations or batch unavailable markers)
                # still force a fresh copy. Never cache by world ordinal alone.
                if (key in dynamic or key not in capture.fields
                        or not _same_capture_snapshot(capture.fields[key], value)):
                    origin[key] = value
            capture.update(**{
                **{key:unavailable("this root has no generating branch address") for key in
                   ("row_action", "column_action", "branch_index", "generating_pre_state", "generating_turn")},
                **origin, "engine_state":state.state_str,
                "instruction_list":(list(state.instruction_list) if state.instruction_list is not None
                                  else unavailable("root position: no generating transition")),
                "offending_instruction":unavailable("no failing instruction reported")})
            self.stage_seconds['capture_context_snapshot'] += time.perf_counter()-started
            self.capture_context_updates += 1

    def begin_decision(self):
        """No value reuse crosses a decision, model or observation contract."""
        if self._leaf_values is not None:
            self._leaf_values.begin_decision(self.model)
        self.render_refusals.clear()
        self.render_subcases.clear()
        self.stage_seconds.clear()
        self.encode_calls = self.forward_batches = self.unique_leaf_evaluations = self.leaf_cache_hits = 0
        self.capture_context_updates = self.captured_leaf_rows = 0
        self.model_call_timings.clear()

    def _capture_batch_context(self):
        if current():
            current().update(**{key:unavailable("batch context: exact per-row evidence retained in leaf_inputs")
                for key in ("engine_state", "instruction_list", "world_spec", "world_ordinal",
                            "row_action", "column_action", "branch_index", "generating_pre_state", "generating_turn")})

    def terminal_value(self, state: NativePosition) -> float | None:
        self._capture_position(state)
        if state.teams_hp is not None:
            exhausted = tuple(all(hp == 0 for hp in side) for side in state.teams_hp)
            if all(exhausted):
                # The native terminal-options seam historically returns side
                # two on simultaneous exhaustion. Complete HP evidence owns
                # this known Gen3 draw; do not accept that seat-biased sign.
                return 0.0
            if any(exhausted):
                return -1.0 if exhausted[0] else 1.0
        started = time.perf_counter()
        result = float(self.native.env_battle_over(state.state_str))
        self.stage_seconds["native_options_terminal"] += time.perf_counter() - started
        if not math.isfinite(result):
            raise ExactSearchRefused("nonfinite native terminal result")
        if result != 0:
            raise ExactSearchRefused("native terminal sign lacks complete team-HP attestation")
        return None

    def actions(self, state: NativePosition, *, root: bool):
        self._capture_position(state)
        started = time.perf_counter()
        report = json.loads(self.native.env_options(state.state_str, root=root))
        self.stage_seconds["native_options_terminal"] += time.perf_counter() - started
        return tuple(report["p1"]), tuple(report["p2"])

    def outcomes(self, state: NativePosition, row_action: str, column_action: str):
        self._capture_position(state)
        started = time.perf_counter()
        ctx = json.dumps({**state.party_species, "turn": state.turn})
        if current():
            current().update(engine_state=state.state_str, row_action=row_action,
                column_action=column_action, event_context=json.loads(ctx),
                instruction_list=unavailable("native transition has not returned"),
                offending_instruction=unavailable("native transition has not returned"))
        # Keep display identities in matrices and sampling receipts. The legacy
        # parser needs bare switch species and "none" for the "No Move" option
        # of a passive seat (e.g. the opponent during a forced replacement).
        raw = self.native.branch_events(
            state.state_str, _branch_action_token(row_action),
            _branch_action_token(column_action), ctx,
            branch_on_damage=True, include_post_state=True,
        )
        branches = json.loads(raw)["branches"]
        self.stage_seconds["native_transition_render_serialize"] += time.perf_counter() - started
        # This is a post-return guard, not a bound on allocations inside Rust.
        # The owned worker hard deadline and memory limit own that residual risk.
        if len(branches) > self.max_outcomes_per_call:
            raise ExactSearchRefused("native outcome-call materialization cap reached")
        for branch_index, branch in enumerate(branches):
            started = time.perf_counter()
            self.render_subcases.update(branch.get("lossy_subcases") or ())
            reasons = list(branch.get("lossy") or ())
            if current():
                current().update(branch_index=branch_index,
                    instruction_list=branch.get("instruction_list", unavailable("binary omitted instruction list")))
            if branch.get("attribution_unsafe"):
                reasons.extend(branch.get("attribution_unsafe_reasons") or ("unsafe_attribution",))
            if reasons:
                self.render_refusals.update(reasons)
                post_state = branch.get("post_state")
                witness = dict(
                    schema="pokezero.native-render-refusal.v1",
                    root_state_sha256=sha256(state.state_str.encode()).hexdigest(),
                    turn=state.turn, row_action=row_action, column_action=column_action,
                    native_row_action=_branch_action_token(row_action),
                    native_column_action=_branch_action_token(column_action),
                    branch_index=branch_index, branch_count=len(branches),
                    percentage=branch.get("percentage"),
                    lossy=branch.get("lossy") or [],
                    attribution_unsafe=bool(branch.get("attribution_unsafe")),
                    attribution_unsafe_reasons=branch.get("attribution_unsafe_reasons") or [],
                    lossy_subcases=branch.get("lossy_subcases") or [],
                    post_state_sha256=(sha256(post_state.encode()).hexdigest()
                                       if isinstance(post_state, str) else None),
                    events=branch.get("events"),
                    engine_state=state.state_str,
                    instruction_list=branch.get("instruction_list", unavailable("binary omitted instruction list")),
                    native_origin=branch.get("refusal_witness"),
                    offending_instruction=unavailable("branch-level refusal, no unique instruction"),
                )
                error = NativeRenderRefused(witness)
                record_refusal("renderer", error, codes=["renderer." + r for r in
                    (reasons + list(branch.get("lossy_subcases") or []))], witness=witness)
                raise error
            post_state = branch.get("post_state")
            if not isinstance(post_state, str) or not post_state:
                raise ExactSearchRefused("native outcome lacks complete post-state")
            events = tuple(branch["events"])
            if any(not isinstance(line, str) for line in events):
                raise ExactSearchRefused("native outcome contains a non-string history event")
            fold = self.native.FoldState.from_payload(state.fold.to_payload())
            fold.advance_in_place(list(events))
            teams_hp = tuple(tuple(row["hp"] for row in branch["post"][side]["pokemon"])
                             for side in ("p1", "p2"))
            if any(not side or any(type(hp) is not int or hp < 0 for hp in side)
                   for side in teams_hp):
                raise ExactSearchRefused("native outcome lacks complete nonnegative team HP")
            child = ChanceBranch(NativePosition(
                post_state, state.party_species,
                state.turn + int(bool(branch["turn_completed"])), fold,
                state.encoder, state.self_side, state.lines + events, teams_hp,
                tuple(branch["instruction_list"]) if "instruction_list" in branch else None,
                {**(state.origin or {}), "row_action":row_action, "column_action":column_action,
                 "branch_index":branch_index, "generating_pre_state":state.state_str,
                 "generating_turn":state.turn},
            ), float(branch["percentage"]) / 100.0)
            self.stage_seconds["fold_history_advance"] += time.perf_counter() - started
            yield child

    def evaluate_leaves(self, states: Sequence[NativePosition]) -> Sequence[float]:
        if not states:
            return []
        if current():
            capture_started = time.perf_counter()
            current().update(leaf_inputs=[dict(engine_state=s.state_str, turn=s.turn, origin=s.origin,
                fold_payload=s.fold.to_payload(), lines=list(s.lines),
                instruction_list=s.instruction_list or unavailable("no transition instructions supplied")) for s in states],
                engine_state=unavailable("batched encode: exact inputs retained in leaf_inputs"),
                world_spec=unavailable("batched encode: per-row worlds retained in leaf_inputs"),
                world_ordinal=unavailable("batched encode: per-row ordinals retained in leaf_inputs"),
                row_action=unavailable("batched encode: per-row address retained in leaf_inputs"),
                column_action=unavailable("batched encode: per-row address retained in leaf_inputs"),
                branch_index=unavailable("batched encode: per-row address retained in leaf_inputs"),
                instruction_list=unavailable("batched encode: per-input lists retained in leaf_inputs"),
                offending_instruction=unavailable("no failing batch row reported"))
            self._capture_batch_context()
            self.stage_seconds['capture_batch_snapshot'] += time.perf_counter()-capture_started
            self.captured_leaf_rows += len(states)
        if self.bulk_encode_workers is not None:
            return self._evaluate_bulk_leaves(states)
        categorical, numeric, token_types, attention = [], [], [], []
        packed_columns = ([], [], [], []) if self.packed_leaves else None
        encoded_rows = []
        started = time.perf_counter()
        for state in states:
            self._capture_position(state)
            if state.self_side not in ("side_one", "side_two"):
                raise ExactSearchRefused("unknown encoder value perspective")
            encoded_at = time.perf_counter()
            row = state.encoder.encode_leaf(state.state_str, state.fold, state.turn,
                                            list(state.lines) or None, False)
            self.stage_seconds["native_leaf_encode"] += time.perf_counter() - encoded_at
            self.encode_calls += 1
            packing_at = time.perf_counter()
            if packed_columns is not None:
                spec = self._row_spec
                wt = spec["window"] * spec["tokens"]
                for name, width, column in zip(
                        ("categorical_ids", "numeric_features", "token_type_ids", "attention_mask"),
                        (wt * spec["categorical_features"] * 4,
                         wt * spec["numeric_features"] * 8, wt * 2, wt), packed_columns):
                    data = row[name]
                    if type(data) is not bytes or len(data) != width:
                        raise ExactSearchRefused("packed native row has wrong bytes type or width: " + name)
                    column.append(data)
                self.stage_seconds["python_leaf_pack"] += time.perf_counter() - packing_at
                continue
            if self._leaf_values is not None:
                encoded_rows.append(encoded_leaf_row(row, model_sha256=self._model_sha256, **self._row_spec))
                continue
            categorical.extend(value[0] for value in struct.iter_unpack("<i", row["categorical_ids"]))
            numeric.extend(value[0] for value in struct.iter_unpack("<d", row["numeric_features"]))
            token_types.extend(value[0] for value in struct.iter_unpack("<h", row["token_type_ids"]))
            if any(value not in (0, 1) for value in row["attention_mask"]):
                raise ExactSearchRefused("native attention mask is not boolean")
            attention.extend(bool(value) for value in row["attention_mask"])
            self.stage_seconds["python_leaf_pack"] += time.perf_counter() - packing_at
        if self._leaf_values is not None:
            self._capture_batch_context()
            if current():
                current().update(engine_state=unavailable("model batch: see leaf_inputs"),
                    instruction_list=unavailable("model batch: see per-row leaf_inputs"),
                    row_action=unavailable("model batch: see per-row leaf_inputs"),
                    column_action=unavailable("model batch: see per-row leaf_inputs"),
                    branch_index=unavailable("model batch: see per-row leaf_inputs"))
            previous_forward_seconds = self._leaf_values.forward_seconds
            try:
                values = self._leaf_values.evaluate(encoded_rows, self.model)
            finally:
                forward_seconds = self._leaf_values.forward_seconds - previous_forward_seconds
                self.stage_seconds["champion_forward"] += forward_seconds
                self.stage_seconds["cpu_encode_and_pack"] += time.perf_counter() - started - forward_seconds
                self.forward_batches = self._leaf_values.forward_batches
                self.unique_leaf_evaluations = self._leaf_values.unique_evaluations
                self.leaf_cache_hits = self._leaf_values.cache_hits
            return [value * (1 if state.self_side == "side_one" else -1)
                    for state, value in zip(states, values)]
        if packed_columns is not None:
            packing_at = time.perf_counter()
            packed = tuple(b"".join(column) for column in packed_columns)
            history = b"\x01" * (len(states) * self._row_spec["window"])
            self.stage_seconds["python_leaf_pack"] += time.perf_counter() - packing_at
        self.stage_seconds["cpu_encode_and_pack"] += time.perf_counter() - started
        started = time.perf_counter()
        self._capture_batch_context()
        if current():
            current().update(engine_state=unavailable("model batch: see leaf_inputs"),
                instruction_list=unavailable("model batch: see per-row leaf_inputs"),
                row_action=unavailable("model batch: see per-row leaf_inputs"),
                column_action=unavailable("model batch: see per-row leaf_inputs"),
                branch_index=unavailable("model batch: see per-row leaf_inputs"))
        if packed_columns is not None:
            values, _, _, _ = self.model.eval_obs_packed(len(states), *packed, history, byte_order="little")
        else:
            values, _, _, _ = self.model.eval_obs_flat(
                len(states), categorical, numeric, token_types, attention,
                [True] * len(states),
            )
        self.stage_seconds["champion_forward"] += time.perf_counter() - started
        self.forward_batches += 1
        if len(values) != len(states):
            raise ExactSearchRefused("native value head changed batch cardinality")
        self.unique_leaf_evaluations += len(states)
        return [float(value) * (1 if state.self_side == "side_one" else -1)
                for state, value in zip(states, values)]

    def _evaluate_bulk_leaves(self, states: Sequence[NativePosition]) -> Sequence[float]:
        """Explicit packed CPU path; retain every row and its encoder identity."""
        started = time.perf_counter()
        if len(states) > 4096 or any(s.self_side not in ('side_one', 'side_two') for s in states):
            raise ExactSearchRefused("bulk encode requires <=4096 leaves with known value frames")
        columns = ([], [], [], [])
        spec = self._row_spec
        wt = spec['window']*spec['tokens']
        names = ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask')
        widths = (wt*spec['categorical_features']*4, wt*spec['numeric_features']*8, wt*2, wt)
        offset = 0
        while offset < len(states):
            encoder = states[offset].encoder
            stop = offset+1
            while stop < len(states) and states[stop].encoder is encoder:
                stop += 1
            group = states[offset:stop]
            method = getattr(encoder, 'encode_leaves_packed', None)
            if not callable(method):
                raise ExactSearchRefused("bulk encode API missing; no serial fallback")
            encoded_at = time.perf_counter()
            if current():
                current().update(leaf_encode_group_start=offset, leaf_encode_group_stop=stop)
            row = method([s.state_str for s in group], [s.fold for s in group],
                         [s.turn for s in group], [list(s.lines) or None for s in group],
                         workers=self.bulk_encode_workers, engine_authoritative=False)
            self.stage_seconds['native_leaf_encode'] += time.perf_counter()-encoded_at
            self.encode_calls += len(group)
            packing_at = time.perf_counter()
            if (not isinstance(row, dict) or type(row.get('batch_rows')) is not int
                    or row['batch_rows'] != len(group)):
                raise ExactSearchRefused("bulk encoder changed batch cardinality")
            for name, width, column in zip(names, widths, columns):
                data = row.get(name)
                if type(data) is not bytes or len(data) != width*len(group):
                    raise ExactSearchRefused("bulk native row has wrong bytes type or width: "+name)
                column.append(data)
            self.stage_seconds['python_leaf_pack'] += time.perf_counter()-packing_at
            offset = stop
        packing_at = time.perf_counter()
        packed = tuple(b''.join(column) for column in columns)
        history = b'\x01'*(len(states)*spec['window'])
        self.stage_seconds['python_leaf_pack'] += time.perf_counter()-packing_at
        self.stage_seconds['cpu_encode_and_pack'] += time.perf_counter()-started
        previous_transport = _drain_model_transport(self.model)  # Reset only diagnostic counters.
        returned = False
        forward_at = time.perf_counter()
        try:
            values, _, _, _ = self.model.eval_obs_packed(len(states), *packed, history, byte_order='little')
            returned = True
        finally:
            elapsed = time.perf_counter()-forward_at
            self.stage_seconds['champion_forward'] += elapsed
            transport = (_drain_model_transport(self.model) if previous_transport['status'] == 'MEASURED'
                         else previous_transport)
            self.model_call_timings.append(dict(batch_rows=len(states),returned=returned,
                wrapper_wall_seconds=elapsed,transport=transport))
        self.forward_batches += 1
        if len(values) != len(states):
            raise ExactSearchRefused("native value head changed batch cardinality")
        self.unique_leaf_evaluations += len(states)
        return [float(value)*(1 if state.self_side == 'side_one' else -1)
                for state, value in zip(states, values)]
