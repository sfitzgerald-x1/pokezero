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
import json
import math
import os
import struct
import time
from typing import Any, Sequence

from .shallow_exact import ChanceBranch, ExactSearchRefused
from .shallow_leaf_rows import DecisionLeafValues, encoded_leaf_row


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


def validate_encoder_frame(root_inputs: str, self_side: str) -> None:
    """The native encoder chooses its frame from showdown_slot, not a label."""
    slot = json.loads(root_inputs).get("observation_metadata", {}).get("showdown_slot")
    expected = {"p1": "side_one", "p2": "side_two"}.get(slot)
    if expected is None or self_side != expected:
        raise ExactSearchRefused("fixture self_side disagrees with encoder showdown_slot")


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
    """

    def __init__(self, native: Any, model: Any, *, window: int = 1,
                 max_outcomes_per_call: int = 100_000, memoize_leaves: bool = False,
                 model_sha256: str | None = None, observation_spec: dict | None = None):
        if os.environ.get("POKEZERO_ENUMERATE_ROLLS") != "1":
            raise ExactSearchRefused("fresh worker must opt into exact damage rolls before engine load")
        if window != 1:
            raise ExactSearchRefused("native pilot supports the champion one-window contract only")
        if type(max_outcomes_per_call) is not int or max_outcomes_per_call <= 0:
            raise ExactSearchRefused("positive native outcome-call limit required")
        if type(memoize_leaves) is not bool:
            raise ExactSearchRefused("leaf memoization must be an explicit boolean")
        self._row_spec = dict(observation_spec or {})
        if memoize_leaves and (set(self._row_spec) != {
                "window", "tokens", "categorical_features", "numeric_features"}
                or any(type(value) is not int or value <= 0 for value in self._row_spec.values())
                or self._row_spec["window"] != window):
            raise ExactSearchRefused("memoized leaves require the fixed complete observation spec")
        self.native, self.model = native, model
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

    def begin_decision(self):
        """No value reuse crosses a decision, model or observation contract."""
        if self._leaf_values is not None:
            self._leaf_values.begin_decision(self.model)
        self.render_refusals.clear()
        self.render_subcases.clear()
        self.stage_seconds.clear()
        self.encode_calls = self.forward_batches = self.unique_leaf_evaluations = self.leaf_cache_hits = 0

    def terminal_value(self, state: NativePosition) -> float | None:
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
        started = time.perf_counter()
        report = json.loads(self.native.env_options(state.state_str, root=root))
        self.stage_seconds["native_options_terminal"] += time.perf_counter() - started
        return tuple(report["p1"]), tuple(report["p2"])

    def outcomes(self, state: NativePosition, row_action: str, column_action: str):
        started = time.perf_counter()
        ctx = json.dumps({**state.party_species, "turn": state.turn})
        # The legacy branch_events entry point accepts bare switch species;
        # unlike env_step, it does not strip the public display prefix itself.
        raw = self.native.branch_events(
            state.state_str, row_action.removeprefix("switch "),
            column_action.removeprefix("switch "), ctx,
            branch_on_damage=True, include_post_state=True,
        )
        branches = json.loads(raw)["branches"]
        self.stage_seconds["native_transition_render_serialize"] += time.perf_counter() - started
        # This is a post-return guard, not a bound on allocations inside Rust.
        # The owned worker hard deadline and memory limit own that residual risk.
        if len(branches) > self.max_outcomes_per_call:
            raise ExactSearchRefused("native outcome-call materialization cap reached")
        for branch in branches:
            started = time.perf_counter()
            self.render_subcases.update(branch.get("lossy_subcases") or ())
            reasons = list(branch.get("lossy") or ())
            if branch.get("attribution_unsafe"):
                reasons.extend(branch.get("attribution_unsafe_reasons") or ("unsafe_attribution",))
            if reasons:
                self.render_refusals.update(reasons)
                raise ExactSearchRefused("native event rendering is not lossless for this pilot")
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
            ), float(branch["percentage"]) / 100.0)
            self.stage_seconds["fold_history_advance"] += time.perf_counter() - started
            yield child

    def evaluate_leaves(self, states: Sequence[NativePosition]) -> Sequence[float]:
        if not states:
            return []
        categorical, numeric, token_types, attention = [], [], [], []
        encoded_rows = []
        started = time.perf_counter()
        for state in states:
            if state.self_side not in ("side_one", "side_two"):
                raise ExactSearchRefused("unknown encoder value perspective")
            row = state.encoder.encode_leaf(state.state_str, state.fold, state.turn,
                                            list(state.lines) or None, False)
            self.encode_calls += 1
            if self._leaf_values is not None:
                encoded_rows.append(encoded_leaf_row(row, model_sha256=self._model_sha256, **self._row_spec))
                continue
            categorical.extend(value[0] for value in struct.iter_unpack("<i", row["categorical_ids"]))
            numeric.extend(value[0] for value in struct.iter_unpack("<d", row["numeric_features"]))
            token_types.extend(value[0] for value in struct.iter_unpack("<h", row["token_type_ids"]))
            if any(value not in (0, 1) for value in row["attention_mask"]):
                raise ExactSearchRefused("native attention mask is not boolean")
            attention.extend(bool(value) for value in row["attention_mask"])
        if self._leaf_values is not None:
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
        self.stage_seconds["cpu_encode_and_pack"] += time.perf_counter() - started
        started = time.perf_counter()
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
