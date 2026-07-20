"""Incremental fold state for the transition/tendency extraction (track B).

The engine-swap plan's revised encoder contract (``docs/test_time_search_plan_v3.md``,
"Encoder contract" + "Schema v2") makes the **fold-state advance** the unit of
correctness: ``advance(fold_state, events) -> (fold_state', products')`` where the
fold state is a first-class serializable struct carrying everything cumulative the
production encoder holds across decisions. Production today re-folds the FULL public
stream on every observe (``normalize_for_player`` -> ``extract_transition_products``
-> ``_fold_replay`` over all lines); this module is the incremental accumulator that
replaces that quadratic re-fold with an O(slice) advance, plus the state export
schema v2 stores per corpus row.

The batch fold in :mod:`pokezero.transitions` / :mod:`pokezero.turn_merged` is
deliberately untouched: it is the differential ORACLE. The closure argument and the
component-by-component carryability verdicts live in ``docs/fold_closure_probe.md``;
the differential proof (batch fold over every real-game prefix == incremental fold
over the slices, at every decision boundary, for every observation-visible product)
is ``tests/test_transitions_fold.py``.

Design points (each mirrors a probe verdict):

- **Open window carried, boundary views virtually closed.** A decision boundary can
  land mid-chunk (forceSwitch pause inside the killing move's chunk); lines after
  the boundary may still attach to the still-open window. ``advance`` therefore
  never finalizes the open window; :meth:`FoldState.products` computes the
  boundary-visible view (the batch fold's trailing ``close_window()``) without
  mutating carried state.
- **Pursuit lookback is a ring buffer.** ``_flag_pursuit_intercepts`` is a
  backward scan that breaks at the first ``_PURSUIT_SCAN_BOUNDARY`` line — and
  deliberately NOT at blank ``|`` chunk separators — so the carried buffer holds
  the raw lines since the last boundary-type line and the intercept flag is set at
  window-open time (provably the same scan set).
- **Turn-merged tokens finalize at ``|turn|``/``|win|``.** Merge groups are
  contiguous same-turn window runs and every input of the NEGATED gate freezes
  when the next turn starts, so finalized merged tokens are immutable; the open
  group is re-merged virtually per boundary. The merge itself calls
  ``turn_merged._merge_turn`` so the two paths cannot drift on merge semantics.
- **Tendencies are running counters.** The ``(side, turn)`` opportunity dedupe is
  a last-counted-turn scalar (turns are monotone); outcome-dependent counters
  accumulate at window close with the open window contributing virtually.
- **Tier-2 annotations are an overlay.** The live trackers
  (``Tier2LiveTracker`` / ``InvestmentLiveTracker``) stay the assessment engines
  (they need the belief engine + dex, which are runtime dependencies, not fold
  state); their per-token-index conclusions enter through
  :meth:`FoldState.apply_annotations` and are joined onto the merged tail via the
  same representative-index arithmetic as ``annotate_turn_merged_tokens``. The
  observation's full-stream pinned surfaces (``showdown.py`` CB-pinned /
  investment-pinned) are maintained as monotone bounded reductions.
- ``|t:|`` wall-clock lines are filtered inside :meth:`FoldState.advance` (the
  schema-v2 byte-determinism rule); the production parser never forwards them, so
  this only matters for raw-log slices.

Read-only imports of private helpers from the oracle modules are intentional:
no oracle module is modified, and reusing ``_merge_turn`` / ``_Window`` / the parse
helpers is what makes the incremental path definitionally aligned with the batch
path rather than a re-implementation that could drift.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence

from .belief import _CALLER_MOVES, _called_move_source
from .showdown import (
    _condition_features,
    _line_mentions_baton_pass,
    _normalize_identifier,
    _side_condition_identifier,
    _slot_from_ident,
    _species_from_details,
    _species_from_ident,
    _update_side_conditions,
    _update_weather,
)
from .transitions import (
    DAMAGE_OUTCOME_ABSORBED,
    DAMAGE_OUTCOME_BLOCKED,
    DAMAGE_OUTCOME_BROKE_SUB,
    DAMAGE_OUTCOME_ENDURED,
    DAMAGE_OUTCOME_HIT_SUB,
    DAMAGE_OUTCOME_IMMUNE,
    EFFECTIVENESS_IMMUNE,
    EFFECTIVENESS_RESISTED,
    EFFECTIVENESS_SUPER,
    SIDE_EFFECT_BOOST,
    SIDE_EFFECT_CHARGING,
    SIDE_EFFECT_DRAIN,
    SIDE_EFFECT_HAZARD_CLEAR,
    SIDE_EFFECT_HAZARD_SET,
    SIDE_EFFECT_HEAL,
    SIDE_EFFECT_STATUS_INFLICTED,
    SIDE_EFFECT_WEATHER_SET,
    SWITCH_REASON_BATON_PASS,
    SWITCH_REASON_LEAD,
    SWITCH_REASON_REPLACEMENT,
    SWITCH_REASON_VOLUNTARY,
    TOKEN_KIND_CANT,
    TOKEN_KIND_MOVE,
    TOKEN_KIND_SWITCH,
    OpponentMonTendency,
    OpponentWeatherReveal,
    TendencyStats,
    TransitionToken,
    _CANT_NO_CHOICE_REASONS,
    _MonCounters,
    _PURSUIT_SCAN_BOUNDARY,
    _SELF_COST_FROM_TAGS,
    _SELF_FAINT_COST_MOVES,
    _StayRecord,
    _Window,
    _from_tag_payload,
    _is_absorb_signature,
    _is_absorb_start,
    _of_tag_slot,
    _other_side,
    _validated_slot,
)
from .turn_merged import (
    PHASE_EXTRA,
    PHASE_LEAD,
    SUB_BLOCK_ABSENT,
    SUB_BLOCK_ACTION,
    TurnMergedToken,
    TurnSubBlock,
    _action_sub_block,
    _expansion_length,
    _merge_turn,
    _single_phase,
)

# Default tail bounds. The merged tail must cover the largest merged-token budget any
# observation spec can read (spec.transition_token_count == 128); the per-action tail
# must cover the v2/v2.1 per-action budget (also <= 128) AND the flatten expansion of
# the merged tail (<= 8 per-action tokens per merged turn), AND give the annotation
# overlay comfortable headroom to land while its tokens are still identifiable.
DEFAULT_MERGED_TAIL_LIMIT = 128
DEFAULT_ACTION_TAIL_LIMIT = 512

# Annotation overlay values: (residual, residual_valid, cb_bit, investment) — exactly
# the four fields the annotation layers may rewrite (turn_merged.annotate docstring).
AnnotationValues = tuple[Optional[float], bool, bool, float]


@dataclass(frozen=True)
class FoldProducts:
    """The observation-visible boundary products (see docs/fold_closure_probe.md #10).

    ``transition_tokens`` / ``turn_merged_tokens`` are bounded TAILS (most recent
    ``*_tail_limit`` entries) — exactly the ``[-budget:]`` windows any encoder reads —
    with the annotation overlay applied; the ``*_total`` counts recover the
    attention-mask fill (``min(total, budget, count)``). ``tendency_stats`` is the
    full dataclass the stats token encodes. The pinned surfaces mirror the
    v2.1/v2.2 encode's full-stream reductions (normalized species identifiers).
    """

    transition_tokens: tuple[TransitionToken, ...]
    transition_token_total: int
    turn_merged_tokens: tuple[TurnMergedToken, ...]
    turn_merged_total: int
    tendency_stats: TendencyStats
    cb_pinned_species: frozenset[str]
    investment_pinned: Mapping[str, float]


@dataclass
class FoldState:
    """Serializable incremental fold state (one per perspective, per battle).

    Mutating entry points (:meth:`advance`, :meth:`apply_annotations`) are PURE:
    they clone, mutate the clone, and return it — each search chance-child advances
    its OWN copy per the plan's search-tree contract. In-place variants exist for
    hot loops (:meth:`advance_in_place`).
    """

    perspective_slot: str
    merged_tail_limit: int = DEFAULT_MERGED_TAIL_LIMIT
    action_tail_limit: int = DEFAULT_ACTION_TAIL_LIMIT

    # --- fold core (transitions._fold_replay main-loop locals, probe #1) ---
    event_index: int = 0  # count of processed (post-|t:|-filter) lines
    side_condition_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"p1": {}, "p2": {}}
    )
    weather: Optional[str] = None
    turn_number: int = 0
    hp_fraction: dict[str, float] = field(default_factory=dict)
    occupant: dict[str, _StayRecord] = field(default_factory=dict)
    transformed: dict[str, bool] = field(default_factory=lambda: {"p1": False, "p2": False})
    pending_baton_pass: dict[str, bool] = field(default_factory=lambda: {"p1": False, "p2": False})
    pending_faint_replacement: dict[str, bool] = field(
        default_factory=lambda: {"p1": False, "p2": False}
    )
    lead_seen: dict[str, bool] = field(default_factory=lambda: {"p1": False, "p2": False})
    pending_charge: dict[str, Optional[str]] = field(
        default_factory=lambda: {"p1": None, "p2": None}
    )
    current_window: Optional[_Window] = None
    # Pursuit ring buffer (probe #3): raw lines since the last _PURSUIT_SCAN_BOUNDARY
    # line (blank | separators included — they do NOT break the batch scan).
    pursuit_buffer: list[str] = field(default_factory=list)

    # --- merge staging (probe #8) ---
    lead_done: bool = False
    pending_windows: list[_Window] = field(default_factory=list)
    # Finalized merged tokens: (token, first_rep_index, second_rep_index) — rep indices
    # are absolute per-action token indices of each ACTION sub-block's representative
    # (annotate_turn_merged_tokens' cursor arithmetic), None for non-ACTION sub-blocks.
    merged_done: deque = field(default_factory=deque)
    merged_total: int = 0
    # Absolute per-action index just past the finalized merged tokens' flatten coverage.
    expansion_cursor: int = 0

    # --- per-action token tail ---
    action_tail: deque = field(default_factory=deque)
    action_total: int = 0

    # --- tendency running state (probe #4/#5/#6) ---
    opponent_switch_count: int = 0
    opponent_decision_opportunities: int = 0
    last_opponent_opportunity_turn: Optional[int] = None
    blocked_on_our_attack_count: int = 0
    pursuit_intercept_predict_count: int = 0
    my_switch_turn_count: int = 0
    mon_counters: dict[tuple[str, str], _MonCounters] = field(default_factory=dict)
    # (side, weather id) -> OR(from_ability); the consumer's order-independent reduction.
    weather_reveals: dict[tuple[str, str], bool] = field(default_factory=dict)

    # --- bounded recent-turn slices for the merge (probe #7) ---
    turn_start_occupants: dict[int, dict[str, str]] = field(default_factory=dict)
    completed_turns: set = field(default_factory=set)
    fainted_turns: set = field(default_factory=set)

    # --- Tier-2 annotation overlay (probe #9) ---
    annotations: dict[int, AnnotationValues] = field(default_factory=dict)
    # rep per-action index -> (merged seq number, "first" | "second"), finalized only.
    rep_index_map: dict[int, tuple[int, str]] = field(default_factory=dict)
    cb_pinned: set = field(default_factory=set)
    # normalized defender species -> (last annotated index, code)
    investment_pinned_state: dict[str, tuple[int, float]] = field(default_factory=dict)

    # ------------------------------------------------------------------ construction

    @classmethod
    def initial(
        cls,
        *,
        perspective_slot: str,
        merged_tail_limit: int = DEFAULT_MERGED_TAIL_LIMIT,
        action_tail_limit: int = DEFAULT_ACTION_TAIL_LIMIT,
    ) -> "FoldState":
        return cls(
            perspective_slot=_validated_slot(perspective_slot),
            merged_tail_limit=int(merged_tail_limit),
            action_tail_limit=int(action_tail_limit),
        )

    @property
    def opponent_slot(self) -> str:
        return _other_side(self.perspective_slot)

    # ------------------------------------------------------------------ public API

    def advance(self, raw_lines: Sequence[str]) -> tuple["FoldState", FoldProducts]:
        """Fold a slice of new public lines; returns ``(new_state, products)``.

        Pure: ``self`` is untouched. ``raw_lines`` is the inter-decision event slice
        (public protocol lines appended since the previous advance); ``|t:|``
        wall-clock lines are filtered here for byte-determinism.
        """
        state = self._clone()
        state.advance_in_place(raw_lines)
        return state, state.products()

    def advance_in_place(self, raw_lines: Sequence[str]) -> None:
        """In-place :meth:`advance` for hot loops (no clone, no products build)."""
        for raw_line in raw_lines:
            parts = raw_line.split("|")
            event_type = parts[1] if len(parts) > 1 else ""
            if event_type == "t:":
                continue  # wall-clock line: never battle state (schema-v2 filter)
            self._process_line(raw_line, parts, event_type)
            self.event_index += 1
            # Pursuit ring buffer maintenance AFTER processing: boundary-type lines
            # clear it (they stop the batch backward scan); everything else — blank
            # separators included — joins the scan set.
            if event_type in _PURSUIT_SCAN_BOUNDARY:
                self.pursuit_buffer.clear()
            else:
                self.pursuit_buffer.append(raw_line)

    def apply_annotations(
        self, overlay: Mapping[int, AnnotationValues]
    ) -> "FoldState":
        """Layer tracker conclusions (per absolute token index) onto the fold. Pure.

        ``overlay`` maps per-action token indices to their
        ``(residual, residual_valid, cb_bit, investment)`` values — the four fields
        the annotation layers may rewrite. Per-index values are immutable once
        applied (the trackers assess each strike exactly once); a changed value
        raises. Indices must still be identifiable (within the action tail or the
        currently open window) — guaranteed in per-boundary operation.
        """
        state = self._clone()
        state.apply_annotations_in_place(overlay)
        return state

    def apply_annotations_in_place(self, overlay: Mapping[int, AnnotationValues]) -> None:
        for index in sorted(overlay):
            raw_residual, raw_valid, raw_cb, raw_investment = overlay[index]
            # Canonicalize value types exactly as from_payload does, so a state built
            # from loosely-typed overlays (int-flag trackers, JSON round-trips)
            # serializes byte-identically to one built from bool/float values.
            values = (
                raw_residual if raw_residual is None else float(raw_residual),
                bool(raw_valid),
                bool(raw_cb),
                float(raw_investment),
            )
            existing = self.annotations.get(index)
            if existing is not None:
                if tuple(existing) != values:
                    raise ValueError(
                        f"annotation for token index {index} changed after application "
                        f"({existing!r} -> {values!r}); tracker conclusions are per-index "
                        "immutable."
                    )
                continue
            token = self._token_identity(index)
            self.annotations[index] = values  # type: ignore[assignment]
            residual, residual_valid, cb_bit, investment = values
            if (
                cb_bit
                and token.kind == TOKEN_KIND_MOVE
                and token.actor_slot == self.opponent_slot
            ):
                self.cb_pinned.add(_normalize_identifier(token.actor_species))
            if (
                investment
                and token.kind == TOKEN_KIND_MOVE
                and token.actor_slot == self.perspective_slot
                and token.defender_species
            ):
                key = _normalize_identifier(token.defender_species)
                previous = self.investment_pinned_state.get(key)
                if previous is None or index >= previous[0]:
                    # Same clamp as the production pinned reduction (showdown.py
                    # tier2_investment_pinned): logic-identical under future drift of
                    # the code vocabulary; a no-op for today's {-1,-0.5,0.5,1} codes.
                    self.investment_pinned_state[key] = (
                        index,
                        max(-1.0, min(1.0, float(investment))),
                    )
        self._prune_annotations()

    def products(self) -> FoldProducts:
        """The boundary-visible products (virtual close + virtual merge; no mutation)."""
        virtual_windows = list(self.pending_windows)
        virtual_token: Optional[TransitionToken] = None
        if self.current_window is not None:
            virtual_windows.append(self.current_window)
            virtual_token = _token_from_window(self.current_window)

        # Per-action tail (annotated).
        tail_start = self.action_total - len(self.action_tail)
        action_tokens = [
            self._annotated_token(tail_start + offset, token)
            for offset, token in enumerate(self.action_tail)
        ]
        total_actions = self.action_total
        if virtual_token is not None:
            action_tokens.append(self._annotated_token(self.action_total, virtual_token))
            total_actions += 1
        action_tokens = action_tokens[-self.action_tail_limit :]

        # Merged stream: finalized tail + virtual merge of the open run.
        merged_tokens = [
            self._annotated_merged(token, first_rep, second_rep)
            for token, first_rep, second_rep in self.merged_done
        ]
        virtual_merged, _ = _merge_window_run(
            virtual_windows,
            lead_done=self.lead_done,
            turn_start_occupants=self.turn_start_occupants,
            completed_turns=self.completed_turns,
            fainted_turns=self.fainted_turns,
        )
        cursor = self.expansion_cursor
        for token in virtual_merged:
            annotated, cursor = self._annotate_with_cursor(token, cursor)
            merged_tokens.append(annotated)
        merged_total = self.merged_total + len(virtual_merged)
        merged_tokens = merged_tokens[-self.merged_tail_limit :]

        return FoldProducts(
            transition_tokens=tuple(action_tokens),
            transition_token_total=total_actions,
            turn_merged_tokens=tuple(merged_tokens),
            turn_merged_total=merged_total,
            tendency_stats=self._tendency_stats(self.current_window),
            cb_pinned_species=frozenset(self.cb_pinned),
            investment_pinned={
                species: code
                for species, (_, code) in sorted(self.investment_pinned_state.items())
            },
        )

    # ------------------------------------------------------------------ fold core

    def _process_line(self, raw_line: str, parts: list[str], event_type: str) -> None:
        """One line of transitions._fold_replay's loop, against carried state."""
        perspective = self.perspective_slot
        opponent = self.opponent_slot

        if event_type in {"", "upkeep"}:
            self._close_window()
            if event_type == "upkeep":
                self.completed_turns.add(self.turn_number)
            return

        if event_type == "turn":
            self._close_window()
            self.completed_turns.add(self.turn_number)
            # The pending group is complete: |turn|N+1 guarantees no more turn-N
            # windows AND freezes the NEGATED gate's inputs for turn N (probe #7/#8).
            self._flush_pending()
            try:
                self.turn_number = int(parts[2])
            except (IndexError, TypeError, ValueError):
                pass
            for side, stay in self.occupant.items():
                self._counters_for(side, stay.species).turns_active += 1
            self.turn_start_occupants[self.turn_number] = {
                side: stay.species for side, stay in self.occupant.items()
            }
            self._prune_turn_maps()
            return

        if event_type == "win":
            self._close_window()
            self.completed_turns.add(self.turn_number)
            self._flush_pending()
            return

        if event_type == "move" and len(parts) >= 4:
            side = _slot_from_ident(parts[2]) or ""
            if side not in {"p1", "p2"}:
                return
            stay = self.occupant.get(side)
            species = stay.species if stay is not None else _species_from_ident(parts[2])
            called_source = _called_move_source(raw_line)
            called = called_source in _CALLER_MOVES
            move_id = _normalize_identifier(parts[3])
            locked = called_source == "lockedmove" or self.pending_charge[side] == move_id
            self.pending_charge[side] = None
            if stay is not None and not stay.moved:
                stay.moved = True
                self._counters_for(side, stay.species).stayed_and_attacked += 1
            self.pending_baton_pass[side] = move_id == "batonpass"
            defender = (_slot_from_ident(parts[4]) if len(parts) > 4 else None) or _other_side(side)
            defender_stay = self.occupant.get(defender)
            defender_species = defender_stay.species if defender_stay is not None else None
            own, opp, current_weather = self._context_trio()
            window = _Window(
                event_index=self.event_index,
                turn=self.turn_number,
                side=side,
                species=species,
                kind=TOKEN_KIND_MOVE,
                action=move_id,
                defender_side=defender,
                defender_species=defender_species,
                called=called,
                transformed=self.transformed[side],
                own_spikes_layers=own,
                opp_spikes_layers=opp,
                weather=current_weather,
            )
            window.locked_continuation = locked
            # Pursuit intercept, resolved at open time from the ring buffer (probe #3):
            # the batch post-pass scans raw_lines[:event_index] backward and breaks at
            # the first boundary-type line — exactly the buffer's contents, reversed.
            if move_id == "pursuit" and defender is not None:
                for buffered in reversed(self.pursuit_buffer):
                    buffered_parts = buffered.split("|")
                    if (
                        len(buffered_parts) >= 4
                        and (buffered_parts[1] if len(buffered_parts) > 1 else "") == "-activate"
                        and _slot_from_ident(buffered_parts[2]) == defender
                        and _side_condition_identifier(buffered_parts[3]) == "pursuit"
                    ):
                        window.pursuit_intercept = True
                        break
            self._open_window(window)
            return

        if event_type in {"switch", "drag", "replace"} and len(parts) >= 4:
            side = _slot_from_ident(parts[2]) or ""
            if side not in {"p1", "p2"}:
                return
            is_lead = not self.lead_seen[side]
            self.lead_seen[side] = True
            is_faint_replacement = self.pending_faint_replacement[side]
            other_pending = (
                is_faint_replacement and self.pending_faint_replacement[_other_side(side)]
            )
            self.pending_faint_replacement[side] = False
            is_baton_pass = self.pending_baton_pass[side] or _line_mentions_baton_pass(parts)
            self.pending_baton_pass[side] = False
            self.pending_charge[side] = None
            voluntary = (
                event_type == "switch" and not is_lead and not is_faint_replacement and not is_baton_pass
            )
            if is_lead:
                switch_reason = SWITCH_REASON_LEAD
            elif is_faint_replacement:
                switch_reason = SWITCH_REASON_REPLACEMENT
            elif is_baton_pass:
                switch_reason = SWITCH_REASON_BATON_PASS
            else:
                switch_reason = SWITCH_REASON_VOLUNTARY
            previous = self.occupant.get(side)
            if previous is not None and voluntary and not previous.moved:
                self._counters_for(side, previous.species).switched_out_before_attacking += 1
            species = _species_from_details(parts[3]) or _species_from_ident(parts[2])
            self.occupant[side] = _StayRecord(species=species)
            self.transformed[side] = False
            condition = _condition_features(parts[4] if len(parts) > 4 else None)
            if condition.hp_fraction is not None:
                self.hp_fraction[side] = condition.hp_fraction
            if event_type in {"drag", "replace"}:
                self._close_window()
                return
            own, opp, current_weather = self._context_trio()
            window = _Window(
                event_index=self.event_index,
                turn=self.turn_number,
                side=side,
                species=species,
                kind=TOKEN_KIND_SWITCH,
                action=species,
                defender_side=None,
                own_spikes_layers=own,
                opp_spikes_layers=opp,
                weather=current_weather,
            )
            window.voluntary_switch = voluntary
            window.switch_reason = switch_reason
            window.other_side_pending_replacement = other_pending
            self._open_window(window)
            return

        if event_type == "cant" and len(parts) >= 4:
            side = _slot_from_ident(parts[2]) or ""
            if side not in {"p1", "p2"}:
                return
            stay = self.occupant.get(side)
            species = stay.species if stay is not None else _species_from_ident(parts[2])
            self.pending_charge[side] = None
            own, opp, current_weather = self._context_trio()
            self._open_window(
                _Window(
                    event_index=self.event_index,
                    turn=self.turn_number,
                    side=side,
                    species=species,
                    kind=TOKEN_KIND_CANT,
                    action=_side_condition_identifier(parts[3]),
                    defender_side=None,
                    transformed=self.transformed[side],
                    own_spikes_layers=own,
                    opp_spikes_layers=opp,
                    weather=current_weather,
                )
            )
            return

        # --- Non-action lines: window accumulation, then global state updates. ---
        current = self.current_window
        target = _slot_from_ident(parts[2]) if len(parts) > 2 else None
        from_payload = _from_tag_payload(raw_line)

        if event_type == "-transform" and target in {"p1", "p2"}:
            self.transformed[target] = True

        if event_type == "-damage" and target in {"p1", "p2"} and len(parts) >= 4:
            condition = _condition_features(parts[3])
            new_fraction = condition.hp_fraction
            if current is not None and target == current.defender_side:
                if from_payload is None:
                    if current.kind == TOKEN_KIND_MOVE and new_fraction is not None:
                        previous_fraction = self.hp_fraction.get(target, 1.0)
                        delta = previous_fraction - new_fraction
                        if delta > 0:
                            current.damage_fraction += delta
                        current.defender_hit_by_move = True
                else:
                    current.defender_hit_by_move = False
            if (
                current is not None
                and target == current.side
                and current.kind == TOKEN_KIND_MOVE
                and new_fraction is not None
            ):
                normalized_from = (
                    _side_condition_identifier(from_payload) if from_payload is not None else None
                )
                if from_payload is None or normalized_from in _SELF_COST_FROM_TAGS:
                    cost_delta = self.hp_fraction.get(target, 1.0) - new_fraction
                    if cost_delta > 0:
                        current.self_hp_cost += cost_delta
            if new_fraction is not None:
                self.hp_fraction[target] = new_fraction

        elif event_type in {"-heal", "-sethp"} and target in {"p1", "p2"} and len(parts) >= 4:
            condition = _condition_features(parts[3])
            if (
                event_type == "-sethp"
                and current is not None
                and target == current.side
                and current.kind == TOKEN_KIND_MOVE
                and condition.hp_fraction is not None
                and from_payload is not None
                and _side_condition_identifier(from_payload) == "painsplit"
            ):
                sethp_delta = self.hp_fraction.get(target, 1.0) - condition.hp_fraction
                if sethp_delta > 0:
                    current.self_hp_cost += sethp_delta
            if condition.hp_fraction is not None:
                self.hp_fraction[target] = condition.hp_fraction
            is_silent = "[silent]" in raw_line
            if current is not None and event_type == "-heal" and target == current.side and not is_silent:
                if from_payload is not None and _normalize_identifier(from_payload) == "drain":
                    current.upgrade_side_effect(SIDE_EFFECT_DRAIN)
                elif from_payload is None:
                    current.upgrade_side_effect(SIDE_EFFECT_HEAL)

        elif event_type == "faint" and target in {"p1", "p2"}:
            if (
                current is not None
                and target == current.side
                and current.kind == TOKEN_KIND_MOVE
                and current.action in _SELF_FAINT_COST_MOVES
            ):
                current.self_hp_cost += self.hp_fraction.get(target, 1.0)
            self.hp_fraction[target] = 0.0
            self.pending_faint_replacement[target] = True
            self.fainted_turns.add(self.turn_number)
            if current is not None and target == current.defender_side and current.defender_hit_by_move:
                current.ko = True

        elif event_type == "-status" and target in {"p1", "p2"}:
            if current is not None and target != current.side:
                current.upgrade_side_effect(SIDE_EFFECT_STATUS_INFLICTED)

        elif event_type in {"-boost", "-unboost", "-setboost"}:
            if current is not None and from_payload is None:
                current.upgrade_side_effect(SIDE_EFFECT_BOOST)

        elif event_type == "-sidestart":
            if current is not None:
                current.upgrade_side_effect(SIDE_EFFECT_HAZARD_SET)

        elif event_type == "-sideend":
            if current is not None:
                current.upgrade_side_effect(SIDE_EFFECT_HAZARD_CLEAR)

        elif event_type == "-weather" and len(parts) >= 3:
            identifier = _normalize_identifier(parts[2])
            is_upkeep = "[upkeep]" in raw_line
            if identifier and identifier != "none" and not is_upkeep:
                if current is not None:
                    current.upgrade_side_effect(SIDE_EFFECT_WEATHER_SET)
                from_ability = from_payload is not None and from_payload.lower().startswith("ability:")
                setter = _of_tag_slot(raw_line) or (current.side if current is not None else None)
                if setter in {"p1", "p2"}:
                    key = (setter, identifier)
                    self.weather_reveals[key] = self.weather_reveals.get(key, False) or from_ability

        elif event_type == "-prepare":
            if current is not None:
                current.upgrade_side_effect(SIDE_EFFECT_CHARGING)
            prepare_side = _slot_from_ident(parts[2]) if len(parts) > 2 else None
            if prepare_side in {"p1", "p2"} and len(parts) >= 4:
                self.pending_charge[prepare_side] = _normalize_identifier(parts[3])

        elif event_type == "-crit":
            if current is not None and target == current.defender_side:
                current.crit = True

        elif event_type == "-miss":
            if current is not None and _slot_from_ident(parts[2]) == current.side:
                current.miss = True

        elif event_type == "-fail":
            # Spec v3 window-scoped marker — mirrors transitions._fold_replay exactly
            # (no side condition; see the batch handler's rationale).
            if current is not None:
                current.fail = True

        elif event_type == "-supereffective":
            if current is not None and target == current.defender_side:
                current.effectiveness = EFFECTIVENESS_SUPER

        elif event_type == "-resisted":
            if current is not None and target == current.defender_side:
                current.effectiveness = EFFECTIVENESS_RESISTED

        elif event_type == "-immune":
            if current is not None and target == current.defender_side:
                current.effectiveness = EFFECTIVENESS_IMMUNE
                if _is_absorb_signature(from_payload):
                    current.upgrade_outcome(DAMAGE_OUTCOME_ABSORBED)
                else:
                    current.upgrade_outcome(DAMAGE_OUTCOME_IMMUNE)

        elif event_type == "-hitcount" and len(parts) >= 4:
            if current is not None:
                try:
                    current.n_hits = max(1, int(parts[3]))
                except (TypeError, ValueError):
                    pass

        elif event_type == "-activate" and len(parts) >= 4:
            identifier = _side_condition_identifier(parts[3])
            if current is not None and target == current.defender_side:
                if identifier in {"protect", "detect"}:
                    current.upgrade_outcome(DAMAGE_OUTCOME_BLOCKED)
                elif identifier == "substitute":
                    current.upgrade_outcome(DAMAGE_OUTCOME_HIT_SUB)
                elif identifier == "endure":
                    current.upgrade_outcome(DAMAGE_OUTCOME_ENDURED)

        elif event_type == "-end" and len(parts) >= 4:
            if (
                current is not None
                and target == current.defender_side
                and _side_condition_identifier(parts[3]) == "substitute"
            ):
                current.upgrade_outcome(DAMAGE_OUTCOME_BROKE_SUB)

        if (
            current is not None
            and target == current.defender_side
            and event_type in {"-heal", "-start"}
            and (_is_absorb_signature(from_payload) or _is_absorb_start(event_type, parts))
        ):
            current.upgrade_outcome(DAMAGE_OUTCOME_ABSORBED)

        _update_side_conditions(parts, self.side_condition_counts)
        self.weather = _update_weather(parts, self.weather)

    def _context_trio(self) -> tuple[int, int, Optional[str]]:
        own = int(self.side_condition_counts[self.perspective_slot].get("spikes", 0))
        opp = int(self.side_condition_counts[self.opponent_slot].get("spikes", 0))
        return own, opp, self.weather

    def _counters_for(self, side: str, species: str) -> _MonCounters:
        return self.mon_counters.setdefault((side, species), _MonCounters())

    def _open_window(self, window: _Window) -> None:
        self._close_window()
        self.current_window = window

    def _close_window(self) -> None:
        window = self.current_window
        if window is None:
            return
        self.current_window = None
        self.pending_windows.append(window)
        token = _token_from_window(window)
        self.action_tail.append(token)
        while len(self.action_tail) > self.action_tail_limit:
            self.action_tail.popleft()
        self.action_total += 1
        self._accumulate_tendency(window, token)

    # ------------------------------------------------------------------ tendencies

    def _accumulate_tendency(self, window: _Window, token: TransitionToken) -> None:
        """One (token, window) pair of _tendency_stats_from_fold's loop, at close."""
        opponent = self.opponent_slot
        voluntary_switch = token.kind == TOKEN_KIND_SWITCH and window.voluntary_switch
        is_decision = (
            (token.kind == TOKEN_KIND_MOVE and not token.called and not window.locked_continuation)
            or voluntary_switch
            or (token.kind == TOKEN_KIND_CANT and token.action not in _CANT_NO_CHOICE_REASONS)
        )
        if (
            is_decision
            and token.actor_slot == opponent
            and self.last_opponent_opportunity_turn != token.turn
        ):
            # The batch (side, turn) set dedupe: turns are monotone and tokens arrive
            # in order, so a last-counted scalar is exact (probe #4). Only the
            # opponent side of the pair is ever read.
            self.opponent_decision_opportunities += 1
            self.last_opponent_opportunity_turn = token.turn
        if token.actor_slot == opponent:
            if voluntary_switch:
                self.opponent_switch_count += 1
            if token.kind == TOKEN_KIND_MOVE and token.pursuit_intercept:
                self.pursuit_intercept_predict_count += 1
        else:
            if voluntary_switch:
                self.my_switch_turn_count += 1
            if token.kind == TOKEN_KIND_MOVE and token.damage_outcome == DAMAGE_OUTCOME_BLOCKED:
                self.blocked_on_our_attack_count += 1

    def _tendency_stats(self, virtual_window: Optional[_Window]) -> TendencyStats:
        opponent = self.opponent_slot
        switches = self.opponent_switch_count
        opportunities = self.opponent_decision_opportunities
        blocked = self.blocked_on_our_attack_count
        pursuit = self.pursuit_intercept_predict_count
        my_switches = self.my_switch_turn_count
        if virtual_window is not None:
            token = _token_from_window(virtual_window)
            voluntary_switch = token.kind == TOKEN_KIND_SWITCH and virtual_window.voluntary_switch
            is_decision = (
                (
                    token.kind == TOKEN_KIND_MOVE
                    and not token.called
                    and not virtual_window.locked_continuation
                )
                or voluntary_switch
                or (token.kind == TOKEN_KIND_CANT and token.action not in _CANT_NO_CHOICE_REASONS)
            )
            if (
                is_decision
                and token.actor_slot == opponent
                and self.last_opponent_opportunity_turn != token.turn
            ):
                opportunities += 1
            if token.actor_slot == opponent:
                if voluntary_switch:
                    switches += 1
                if token.kind == TOKEN_KIND_MOVE and token.pursuit_intercept:
                    pursuit += 1
            else:
                if voluntary_switch:
                    my_switches += 1
                if token.kind == TOKEN_KIND_MOVE and token.damage_outcome == DAMAGE_OUTCOME_BLOCKED:
                    blocked += 1

        mon_tendencies = tuple(
            OpponentMonTendency(
                slot=side,
                species=species,
                switched_out_before_attacking=counters.switched_out_before_attacking,
                stayed_and_attacked=counters.stayed_and_attacked,
                turns_active=counters.turns_active,
            )
            for (side, species), counters in sorted(self.mon_counters.items())
            if side == opponent
        )
        reveals_by_weather = {
            weather: from_ability
            for (side, weather), from_ability in self.weather_reveals.items()
            if side == opponent
        }
        weather_reveals = tuple(
            OpponentWeatherReveal(weather=weather, from_ability=from_ability)
            for weather, from_ability in sorted(reveals_by_weather.items())
        )
        return TendencyStats(
            perspective_slot=self.perspective_slot,
            opponent_slot=opponent,
            opponent_switch_count=switches,
            opponent_decision_opportunities=opportunities,
            opponent_mon_tendencies=mon_tendencies,
            opponent_weather_reveals=weather_reveals,
            blocked_on_our_attack_count=blocked,
            pursuit_intercept_predict_count=pursuit,
            my_switch_turn_count=my_switches,
        )

    # ------------------------------------------------------------------ merge staging

    def _flush_pending(self) -> None:
        if not self.pending_windows:
            return
        merged, self.lead_done = _merge_window_run(
            self.pending_windows,
            lead_done=self.lead_done,
            turn_start_occupants=self.turn_start_occupants,
            completed_turns=self.completed_turns,
            fainted_turns=self.fainted_turns,
        )
        self.pending_windows = []
        for token in merged:
            first_rep: Optional[int] = None
            second_rep: Optional[int] = None
            for position, sub in (("first", token.first), ("second", token.second)):
                if sub.status != SUB_BLOCK_ACTION:
                    continue
                rep = self.expansion_cursor + _representative_offset(sub)
                if position == "first":
                    first_rep = rep
                else:
                    second_rep = rep
                self.rep_index_map[rep] = (self.merged_total, position)
                self.expansion_cursor += _expansion_length(sub)
            self.merged_done.append((token, first_rep, second_rep))
            self.merged_total += 1
        while len(self.merged_done) > self.merged_tail_limit:
            self.merged_done.popleft()
        # The flatten bijection: the flushed merged tokens' expansions cover exactly
        # the flushed windows' per-action tokens.
        if self.expansion_cursor != self.action_total:
            raise AssertionError(
                "fold-state invariant violated: merged flatten coverage "
                f"({self.expansion_cursor}) != emitted per-action tokens ({self.action_total})."
            )
        self._prune_rep_index_map()
        self._prune_annotations()

    def _prune_turn_maps(self) -> None:
        keep_from = self.turn_number - 1
        for turn in [t for t in self.turn_start_occupants if t < keep_from]:
            del self.turn_start_occupants[turn]
        self.completed_turns = {t for t in self.completed_turns if t >= keep_from}
        self.fainted_turns = {t for t in self.fainted_turns if t >= keep_from}

    def _prune_rep_index_map(self) -> None:
        oldest_seq = self.merged_total - len(self.merged_done)
        for index in [i for i, (seq, _) in self.rep_index_map.items() if seq < oldest_seq]:
            del self.rep_index_map[index]

    def _prune_annotations(self) -> None:
        tail_start = self.action_total - len(self.action_tail)
        for index in [
            i for i in self.annotations if i < tail_start and i not in self.rep_index_map
        ]:
            del self.annotations[index]

    # ------------------------------------------------------------------ annotation join

    def _token_identity(self, index: int) -> TransitionToken:
        tail_start = self.action_total - len(self.action_tail)
        if tail_start <= index < self.action_total:
            return self.action_tail[index - tail_start]
        if index == self.action_total and self.current_window is not None:
            return _token_from_window(self.current_window)
        raise ValueError(
            f"annotation index {index} is outside the identifiable range "
            f"[{tail_start}, {self.action_total}] — apply annotations per boundary "
            "(or raise action_tail_limit)."
        )

    def _annotated_token(self, index: int, token: TransitionToken) -> TransitionToken:
        values = self.annotations.get(index)
        if values is None:
            return token
        residual, residual_valid, cb_bit, investment = values
        return replace(
            token,
            residual=residual,
            residual_valid=residual_valid,
            cb_bit=cb_bit,
            investment=investment,
        )

    def _annotated_merged(
        self,
        token: TurnMergedToken,
        first_rep: Optional[int],
        second_rep: Optional[int],
    ) -> TurnMergedToken:
        updates: dict[str, TurnSubBlock] = {}
        for position, sub, rep in (
            ("first", token.first, first_rep),
            ("second", token.second, second_rep),
        ):
            if rep is None:
                continue
            values = self.annotations.get(rep)
            if values is None:
                continue
            residual, residual_valid, cb_bit, investment = values
            if (
                residual == sub.residual
                and residual_valid == sub.residual_valid
                and cb_bit == sub.cb_bit
                and investment == sub.investment
            ):
                continue
            updates[position] = replace(
                sub,
                residual=residual,
                residual_valid=residual_valid,
                cb_bit=cb_bit,
                investment=investment,
            )
        return replace(token, **updates) if updates else token

    def _annotate_with_cursor(
        self, token: TurnMergedToken, cursor: int
    ) -> tuple[TurnMergedToken, int]:
        """Annotate a virtual (unfinalized) merged token, advancing the flatten cursor."""
        first_rep: Optional[int] = None
        second_rep: Optional[int] = None
        for position, sub in (("first", token.first), ("second", token.second)):
            if sub.status != SUB_BLOCK_ACTION:
                continue
            rep = cursor + _representative_offset(sub)
            if position == "first":
                first_rep = rep
            else:
                second_rep = rep
            cursor += _expansion_length(sub)
        return self._annotated_merged(token, first_rep, second_rep), cursor

    # ------------------------------------------------------------------ cloning

    def _clone(self) -> "FoldState":
        clone = FoldState(
            perspective_slot=self.perspective_slot,
            merged_tail_limit=self.merged_tail_limit,
            action_tail_limit=self.action_tail_limit,
        )
        clone.event_index = self.event_index
        clone.side_condition_counts = {
            side: dict(counts) for side, counts in self.side_condition_counts.items()
        }
        clone.weather = self.weather
        clone.turn_number = self.turn_number
        clone.hp_fraction = dict(self.hp_fraction)
        clone.occupant = {
            side: _StayRecord(species=stay.species, moved=stay.moved)
            for side, stay in self.occupant.items()
        }
        clone.transformed = dict(self.transformed)
        clone.pending_baton_pass = dict(self.pending_baton_pass)
        clone.pending_faint_replacement = dict(self.pending_faint_replacement)
        clone.lead_seen = dict(self.lead_seen)
        clone.pending_charge = dict(self.pending_charge)
        clone.current_window = (
            replace(self.current_window) if self.current_window is not None else None
        )
        clone.pursuit_buffer = list(self.pursuit_buffer)
        clone.lead_done = self.lead_done
        clone.pending_windows = list(self.pending_windows)  # closed windows are immutable
        clone.merged_done = deque(self.merged_done)
        clone.merged_total = self.merged_total
        clone.expansion_cursor = self.expansion_cursor
        clone.action_tail = deque(self.action_tail)
        clone.action_total = self.action_total
        clone.opponent_switch_count = self.opponent_switch_count
        clone.opponent_decision_opportunities = self.opponent_decision_opportunities
        clone.last_opponent_opportunity_turn = self.last_opponent_opportunity_turn
        clone.blocked_on_our_attack_count = self.blocked_on_our_attack_count
        clone.pursuit_intercept_predict_count = self.pursuit_intercept_predict_count
        clone.my_switch_turn_count = self.my_switch_turn_count
        clone.mon_counters = {
            key: _MonCounters(
                switched_out_before_attacking=value.switched_out_before_attacking,
                stayed_and_attacked=value.stayed_and_attacked,
                turns_active=value.turns_active,
            )
            for key, value in self.mon_counters.items()
        }
        clone.weather_reveals = dict(self.weather_reveals)
        clone.turn_start_occupants = {
            turn: dict(occupants) for turn, occupants in self.turn_start_occupants.items()
        }
        clone.completed_turns = set(self.completed_turns)
        clone.fainted_turns = set(self.fainted_turns)
        clone.annotations = dict(self.annotations)
        clone.rep_index_map = dict(self.rep_index_map)
        clone.cb_pinned = set(self.cb_pinned)
        clone.investment_pinned_state = dict(self.investment_pinned_state)
        return clone

    # ------------------------------------------------------------------ serialization

    def to_payload(self) -> dict:
        """JSON-safe, deterministic export (schema v2 stores this per corpus row)."""
        return {
            "schema": "pokezero.fold-state.v1",
            "perspective_slot": self.perspective_slot,
            "merged_tail_limit": self.merged_tail_limit,
            "action_tail_limit": self.action_tail_limit,
            "event_index": self.event_index,
            "side_condition_counts": {
                side: dict(sorted(counts.items()))
                for side, counts in sorted(self.side_condition_counts.items())
            },
            "weather": self.weather,
            "turn_number": self.turn_number,
            "hp_fraction": dict(sorted(self.hp_fraction.items())),
            "occupant": {
                side: {"species": stay.species, "moved": stay.moved}
                for side, stay in sorted(self.occupant.items())
            },
            "transformed": dict(sorted(self.transformed.items())),
            "pending_baton_pass": dict(sorted(self.pending_baton_pass.items())),
            "pending_faint_replacement": dict(sorted(self.pending_faint_replacement.items())),
            "lead_seen": dict(sorted(self.lead_seen.items())),
            "pending_charge": dict(sorted(self.pending_charge.items())),
            "current_window": (
                _window_to_payload(self.current_window)
                if self.current_window is not None
                else None
            ),
            "pursuit_buffer": list(self.pursuit_buffer),
            "lead_done": self.lead_done,
            "pending_windows": [_window_to_payload(window) for window in self.pending_windows],
            "merged_done": [
                {
                    "token": _merged_token_to_payload(token),
                    "first_rep": first_rep,
                    "second_rep": second_rep,
                }
                for token, first_rep, second_rep in self.merged_done
            ],
            "merged_total": self.merged_total,
            "expansion_cursor": self.expansion_cursor,
            "action_tail": [_transition_token_to_payload(token) for token in self.action_tail],
            "action_total": self.action_total,
            "opponent_switch_count": self.opponent_switch_count,
            "opponent_decision_opportunities": self.opponent_decision_opportunities,
            "last_opponent_opportunity_turn": self.last_opponent_opportunity_turn,
            "blocked_on_our_attack_count": self.blocked_on_our_attack_count,
            "pursuit_intercept_predict_count": self.pursuit_intercept_predict_count,
            "my_switch_turn_count": self.my_switch_turn_count,
            "mon_counters": {
                f"{side}|{species}": [
                    counters.switched_out_before_attacking,
                    counters.stayed_and_attacked,
                    counters.turns_active,
                ]
                for (side, species), counters in sorted(self.mon_counters.items())
            },
            "weather_reveals": {
                f"{side}|{weather}": from_ability
                for (side, weather), from_ability in sorted(self.weather_reveals.items())
            },
            "turn_start_occupants": {
                str(turn): dict(sorted(occupants.items()))
                for turn, occupants in sorted(self.turn_start_occupants.items())
            },
            "completed_turns": sorted(self.completed_turns),
            "fainted_turns": sorted(self.fainted_turns),
            "annotations": {
                str(index): list(values) for index, values in sorted(self.annotations.items())
            },
            "rep_index_map": {
                str(index): [seq, position]
                for index, (seq, position) in sorted(self.rep_index_map.items())
            },
            "cb_pinned": sorted(self.cb_pinned),
            "investment_pinned": {
                species: [index, code]
                for species, (index, code) in sorted(self.investment_pinned_state.items())
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping) -> "FoldState":
        if payload.get("schema") != "pokezero.fold-state.v1":
            raise ValueError(f"unsupported fold-state payload schema: {payload.get('schema')!r}.")
        state = cls.initial(
            perspective_slot=str(payload["perspective_slot"]),
            merged_tail_limit=int(payload["merged_tail_limit"]),
            action_tail_limit=int(payload["action_tail_limit"]),
        )
        state.event_index = int(payload["event_index"])
        state.side_condition_counts = {
            side: {name: int(count) for name, count in counts.items()}
            for side, counts in payload["side_condition_counts"].items()
        }
        for side in ("p1", "p2"):
            state.side_condition_counts.setdefault(side, {})
        state.weather = payload["weather"]
        state.turn_number = int(payload["turn_number"])
        state.hp_fraction = {side: float(value) for side, value in payload["hp_fraction"].items()}
        state.occupant = {
            side: _StayRecord(species=str(entry["species"]), moved=bool(entry["moved"]))
            for side, entry in payload["occupant"].items()
        }
        state.transformed = {side: bool(v) for side, v in payload["transformed"].items()}
        state.pending_baton_pass = {side: bool(v) for side, v in payload["pending_baton_pass"].items()}
        state.pending_faint_replacement = {
            side: bool(v) for side, v in payload["pending_faint_replacement"].items()
        }
        state.lead_seen = {side: bool(v) for side, v in payload["lead_seen"].items()}
        state.pending_charge = dict(payload["pending_charge"])
        state.current_window = (
            _window_from_payload(payload["current_window"])
            if payload["current_window"] is not None
            else None
        )
        state.pursuit_buffer = [str(line) for line in payload["pursuit_buffer"]]
        state.lead_done = bool(payload["lead_done"])
        state.pending_windows = [_window_from_payload(entry) for entry in payload["pending_windows"]]
        state.merged_done = deque(
            (
                _merged_token_from_payload(entry["token"]),
                entry["first_rep"],
                entry["second_rep"],
            )
            for entry in payload["merged_done"]
        )
        state.merged_total = int(payload["merged_total"])
        state.expansion_cursor = int(payload["expansion_cursor"])
        state.action_tail = deque(
            _transition_token_from_payload(entry) for entry in payload["action_tail"]
        )
        state.action_total = int(payload["action_total"])
        state.opponent_switch_count = int(payload["opponent_switch_count"])
        state.opponent_decision_opportunities = int(payload["opponent_decision_opportunities"])
        raw_last = payload["last_opponent_opportunity_turn"]
        state.last_opponent_opportunity_turn = int(raw_last) if raw_last is not None else None
        state.blocked_on_our_attack_count = int(payload["blocked_on_our_attack_count"])
        state.pursuit_intercept_predict_count = int(payload["pursuit_intercept_predict_count"])
        state.my_switch_turn_count = int(payload["my_switch_turn_count"])
        state.mon_counters = {}
        for key, values in payload["mon_counters"].items():
            side, species = key.split("|", 1)
            state.mon_counters[(side, species)] = _MonCounters(
                switched_out_before_attacking=int(values[0]),
                stayed_and_attacked=int(values[1]),
                turns_active=int(values[2]),
            )
        state.weather_reveals = {}
        for key, from_ability in payload["weather_reveals"].items():
            side, weather = key.split("|", 1)
            state.weather_reveals[(side, weather)] = bool(from_ability)
        state.turn_start_occupants = {
            int(turn): dict(occupants)
            for turn, occupants in payload["turn_start_occupants"].items()
        }
        state.completed_turns = set(int(turn) for turn in payload["completed_turns"])
        state.fainted_turns = set(int(turn) for turn in payload["fainted_turns"])
        state.annotations = {
            int(index): (
                values[0] if values[0] is None else float(values[0]),
                bool(values[1]),
                bool(values[2]),
                float(values[3]),
            )
            for index, values in payload["annotations"].items()
        }
        state.rep_index_map = {
            int(index): (int(entry[0]), str(entry[1]))
            for index, entry in payload["rep_index_map"].items()
        }
        state.cb_pinned = set(str(species) for species in payload["cb_pinned"])
        state.investment_pinned_state = {
            str(species): (int(entry[0]), float(entry[1]))
            for species, entry in payload["investment_pinned"].items()
        }
        return state


# ---------------------------------------------------------------------------
# Module-level helpers.
# ---------------------------------------------------------------------------


def _token_from_window(window: _Window) -> TransitionToken:
    """The batch fold's window -> token mapping (transitions._fold_replay:899-923)."""
    return TransitionToken(
        turn=window.turn,
        actor_slot=window.side,
        actor_species=window.species,
        kind=window.kind,
        action=window.action,
        called=window.called,
        transformed=window.transformed,
        damage_fraction=window.damage_fraction,
        self_hp_cost=window.self_hp_cost,
        damage_outcome=window.outcome,
        crit=window.crit,
        miss=window.miss,
        fail=window.fail,
        ko=window.ko,
        pursuit_intercept=window.pursuit_intercept,
        n_hits=window.n_hits,
        effectiveness=window.effectiveness,
        side_effect=window.side_effect,
        own_spikes_layers=window.own_spikes_layers,
        opp_spikes_layers=window.opp_spikes_layers,
        weather=window.weather,
        defender_species=window.defender_species if window.kind == TOKEN_KIND_MOVE else None,
    )


def _merge_window_run(
    windows: Sequence[_Window],
    *,
    lead_done: bool,
    turn_start_occupants: Mapping[int, dict],
    completed_turns,
    fainted_turns,
) -> tuple[list[TurnMergedToken], bool]:
    """turn_merged._merge_fold's body over one contiguous window run.

    Batch output == concatenation over runs because merge groups are contiguous
    same-turn window runs (turn numbers are non-decreasing and runs are flushed at
    every ``|turn|``/``|win|`` line, so no group spans a run boundary), each
    ``_merge_turn`` reads only its group + the (frozen, probe #7) per-turn maps, and
    the lead pass consumes only the stream-initial LEAD run — tracked by
    ``lead_done`` across runs.
    """
    windows = list(windows)
    tokens: list[TurnMergedToken] = []
    index = 0
    if not lead_done:
        leads: list[_Window] = []
        while index < len(windows) and windows[index].switch_reason == SWITCH_REASON_LEAD:
            leads.append(windows[index])
            index += 1
        if leads:
            first = leads[0]
            second = leads[1] if len(leads) > 1 else None
            tokens.append(
                TurnMergedToken(
                    turn=first.turn,
                    phase=PHASE_LEAD,
                    first=_action_sub_block(first),
                    second=(
                        _action_sub_block(second)
                        if second is not None
                        else TurnSubBlock(
                            status=SUB_BLOCK_ABSENT, actor_slot=_other_side(first.side)
                        )
                    ),
                    own_spikes_layers=first.own_spikes_layers,
                    opp_spikes_layers=first.opp_spikes_layers,
                    weather=first.weather,
                )
            )
            for extra in leads[2:]:  # unreachable in singles; bijection safety valve
                tokens.append(_single_phase(extra, PHASE_EXTRA))
        if windows:
            # The batch lead pass runs exactly once, over the stream-initial run.
            lead_done = True

    while index < len(windows):
        turn = windows[index].turn
        turn_windows: list[_Window] = []
        while index < len(windows) and windows[index].turn == turn:
            turn_windows.append(windows[index])
            index += 1
        tokens.extend(
            _merge_turn(
                turn,
                turn_windows,
                dict(turn_start_occupants),
                consumption_confirmed=(turn in completed_turns or turn in fainted_turns),
            )
        )
    return tokens, lead_done


def _representative_offset(sub: TurnSubBlock) -> int:
    """annotate_turn_merged_tokens' representative offset within a sub-block's expansion."""
    offset = 0
    if sub.cant_reason is not None:
        offset += 1  # the cant token precedes the representative
        if sub.called:
            offset += 1  # so does the synthesized Sleep Talk click
    return offset


def _window_to_payload(window: _Window) -> dict:
    # ``fail`` (spec v3) serializes OMIT-WHEN-DEFAULT: the fold-state payload schema
    # ("pokezero.fold-state.v1") is a Rust-parity surface pinned byte-exactly by the
    # committed golden-corpus sample, which predates the field (and contains no |-fail|
    # lines). Emitting the key only when True keeps every pre-v3 payload byte-identical
    # while round-tripping the marker for games that do fail; the key joins the fixed
    # field set when the fold-state schema bumps with the Rust mirror + corpus
    # regeneration at v3 (docs/observation_v3_spec.md, coordination section).
    payload = {
        "event_index": window.event_index,
        "turn": window.turn,
        "side": window.side,
        "species": window.species,
        "kind": window.kind,
        "action": window.action,
        "defender_side": window.defender_side,
        "defender_species": window.defender_species,
        "called": window.called,
        "transformed": window.transformed,
        "own_spikes_layers": window.own_spikes_layers,
        "opp_spikes_layers": window.opp_spikes_layers,
        "weather": window.weather,
        "damage_fraction": window.damage_fraction,
        "self_hp_cost": window.self_hp_cost,
        "outcome": window.outcome,
        "crit": window.crit,
        "miss": window.miss,
        "ko": window.ko,
        "pursuit_intercept": window.pursuit_intercept,
        "n_hits": window.n_hits,
        "effectiveness": window.effectiveness,
        "side_effect": window.side_effect,
        "defender_hit_by_move": window.defender_hit_by_move,
        "voluntary_switch": window.voluntary_switch,
        "locked_continuation": window.locked_continuation,
        "switch_reason": window.switch_reason,
        "other_side_pending_replacement": window.other_side_pending_replacement,
    }
    if window.fail:
        payload["fail"] = True
    return payload


def _window_from_payload(payload: Mapping) -> _Window:
    return _Window(
        event_index=int(payload["event_index"]),
        turn=int(payload["turn"]),
        side=str(payload["side"]),
        species=str(payload["species"]),
        kind=str(payload["kind"]),
        action=str(payload["action"]),
        defender_side=payload["defender_side"],
        defender_species=payload["defender_species"],
        called=bool(payload["called"]),
        transformed=bool(payload["transformed"]),
        own_spikes_layers=int(payload["own_spikes_layers"]),
        opp_spikes_layers=int(payload["opp_spikes_layers"]),
        weather=payload["weather"],
        damage_fraction=float(payload["damage_fraction"]),
        self_hp_cost=float(payload["self_hp_cost"]),
        outcome=str(payload["outcome"]),
        crit=bool(payload["crit"]),
        miss=bool(payload["miss"]),
        fail=bool(payload.get("fail", False)),
        ko=bool(payload["ko"]),
        pursuit_intercept=bool(payload["pursuit_intercept"]),
        n_hits=int(payload["n_hits"]),
        effectiveness=str(payload["effectiveness"]),
        side_effect=str(payload["side_effect"]),
        defender_hit_by_move=bool(payload["defender_hit_by_move"]),
        voluntary_switch=bool(payload["voluntary_switch"]),
        locked_continuation=bool(payload["locked_continuation"]),
        switch_reason=payload["switch_reason"],
        other_side_pending_replacement=bool(payload["other_side_pending_replacement"]),
    )


_TRANSITION_TOKEN_FIELDS = (
    "turn",
    "actor_slot",
    "actor_species",
    "kind",
    "action",
    "called",
    "transformed",
    "damage_fraction",
    "damage_outcome",
    "crit",
    "miss",
    "ko",
    "pursuit_intercept",
    "n_hits",
    "effectiveness",
    "side_effect",
    "self_hp_cost",
    "own_spikes_layers",
    "opp_spikes_layers",
    "weather",
    "defender_species",
    "residual",
    "residual_valid",
    "cb_bit",
    "investment",
)


def _transition_token_to_payload(token: TransitionToken) -> dict:
    payload = {name: getattr(token, name) for name in _TRANSITION_TOKEN_FIELDS}
    # Omit-when-default (spec v3): see _window_to_payload's rationale.
    if token.fail:
        payload["fail"] = True
    return payload


def _transition_token_from_payload(payload: Mapping) -> TransitionToken:
    return TransitionToken(
        fail=bool(payload.get("fail", False)),
        **{name: payload[name] for name in _TRANSITION_TOKEN_FIELDS},
    )


_SUB_BLOCK_FIELDS = (
    "status",
    "actor_slot",
    "actor_species",
    "kind",
    "action",
    "called",
    "transformed",
    "damage_fraction",
    "self_hp_cost",
    "damage_outcome",
    "crit",
    "miss",
    "ko",
    "pursuit_intercept",
    "n_hits",
    "effectiveness",
    "side_effect",
    "defender_species",
    "cant_reason",
    "baton_pass_species",
    "residual",
    "residual_valid",
    "cb_bit",
    "investment",
)


def _sub_block_to_payload(sub: TurnSubBlock) -> dict:
    payload = {name: getattr(sub, name) for name in _SUB_BLOCK_FIELDS}
    # Omit-when-default (spec v3): see _window_to_payload's rationale.
    if sub.fail:
        payload["fail"] = True
    return payload


def _sub_block_from_payload(payload: Mapping) -> TurnSubBlock:
    return TurnSubBlock(
        fail=bool(payload.get("fail", False)),
        **{name: payload[name] for name in _SUB_BLOCK_FIELDS},
    )


def _merged_token_to_payload(token: TurnMergedToken) -> dict:
    return {
        "turn": token.turn,
        "phase": token.phase,
        "first": _sub_block_to_payload(token.first),
        "second": _sub_block_to_payload(token.second),
        "own_spikes_layers": token.own_spikes_layers,
        "opp_spikes_layers": token.opp_spikes_layers,
        "weather": token.weather,
    }


def _merged_token_from_payload(payload: Mapping) -> TurnMergedToken:
    return TurnMergedToken(
        turn=int(payload["turn"]),
        phase=str(payload["phase"]),
        first=_sub_block_from_payload(payload["first"]),
        second=_sub_block_from_payload(payload["second"]),
        own_spikes_layers=int(payload["own_spikes_layers"]),
        opp_spikes_layers=int(payload["opp_spikes_layers"]),
        weather=payload["weather"],
    )
