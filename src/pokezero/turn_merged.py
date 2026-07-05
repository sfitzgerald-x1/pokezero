"""Turn-merged transition tokens (v2.1 batch 3): one token per TURN, two sub-blocks.

Pure extraction layer over the same replay fold as :mod:`pokezero.transitions` — no
observation-encoder changes live here (the encode/schema integration is staged
separately and rebases onto the post-#512 dual-schema encode).

Design (Scott-approved, 2026-07-05):

- One :class:`TurnMergedToken` per battle turn carries the two DECLARED actions as
  sub-blocks: ``first`` = first mover, ``second`` = second mover — speed order becomes
  EXPLICIT structure (in the per-action stream it is only positional). Each sub-block
  carries the per-action token fields (kind / move-or-species / outcome / damage / crit
  / miss / ko / effectiveness / n_hits / side-effect / Tier-2 residual+valid+cb). The
  context trio (own layers / opp layers / weather) is stored ONCE per token, captured at
  the FIRST sub-block's declaration; the positional ``turn`` is stored once per token.
- A sub-block may be NEGATED: the side's declared action was consumed with **zero
  protocol trace**. Engine-verified (vendored gen3 Showdown, 2026-07-05): when a mon
  faints mid-turn before the opponent's declared action executes — hazard sack (switch-in
  faints to Spikes) or a faster Explosion/attack KO — the opponent's declared action
  emits NO ``|move|`` line at all, even for non-targeted moves (a declared Spikes layer
  also fizzles). The engine pauses with a forceSwitch/wait cycle, completes the
  replacement, and goes straight to ``|upkeep``. Hazard-sacking is therefore a true free
  pivot, and the negated sub-block is what makes it learnable. NEGATED (declared, never
  executed) is distinct from ABSENT (no declaration expected: the empty half of a
  single-replacement token).
- REPLACEMENT phases (post-faint switch-ins) stay their own tokens — EXCEPT the cold
  pair: after Explosion/Selfdestruct or any simultaneous double-faint, BOTH sides
  replace blind in ONE engine forceSwitch cycle (verified: the two ``|switch|`` lines
  are emitted back-to-back, order arbitrary). That phase is represented as one merged
  pair token with two switch sub-blocks in engine emission order. Sequential faints
  (move KO now, residual faint later the same turn) are two separate engine request
  cycles and stay two single tokens — the cold-pair signal is "was the OTHER side also
  waiting on a replacement when this switch-in was emitted", not log adjacency.
- Lead send-outs are one PHASE_LEAD pair token (both sides send blind, engine emission
  order) — the same cold-pair shape as a double replacement. (Refinement over the
  approved sketch, which left leads unspecified; justification: identical simultaneity
  semantics, one token saved, and turn-0 keeps no fake speed order.)
- Intra-turn oddities (engine-verified emission shapes):
  * Pursuit interception: Pursuit executes BEFORE the declared switch, so Pursuit is the
    FIRST sub-block and the intercepted switch (which still completes) is the SECOND.
    When Pursuit KOs the switching mon, the engine's own hint line says "Previously
    chosen switches continue in Gen 2-4" and the declared switch completes in the same
    breath with NO forceSwitch cycle — that continuation is folded back in as the
    target's declared switch sub-block, not a replacement token.
  * ``|cant``` (sleep / para / flinch / recharge / broken Focus Punch) IS that side's
    action sub-block (kind ``cant``).
  * A RestTalk turn (three per-action tokens: ``|cant|slp`` + the Sleep Talk click + the
    called execution) collapses to ONE sub-block: the called execution with
    ``called=True`` and ``cant_reason="slp"``; the click is protocol-constant and is
    resynthesized exactly on flatten.
  * Baton Pass: the mid-turn completion switch (engine-verified: forceSwitch pause, then
    ``|switch ... [from] Baton Pass``, then the slower opponent acts against the NEW
    mon) folds into the passer's sub-block as ``baton_pass_species``.
  * Wish / end-of-turn residuals are NOT actions and never form sub-blocks (the
    underlying fold already excludes the residual phase from action windows).
- K budget: the transition budget flag counts TOKENS, and a turn-merged token covers a
  whole turn — the old 64-action horizon is ≈ 32 turn tokens. See the encode-side
  documentation for the loud unit-change note on K=64 configs.

Equivalence gate: :func:`flatten_turn_merged_tokens` reconstructs the per-action token
stream exactly, modulo ONE documented merge: the context trio is stored once per merged
token, so a second sub-block (or second half of a pair) reconstructs with the first
mover's trio. The two can differ only when the first mover's own action changed the trio
(hazard set/clear or weather set — including ability weather on a switch-in), which is
exactly the ``side_effect`` ∈ {hazard-set, hazard-clear, weather-set} predicate the
bijection test uses as its allowance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .belief import _CALLER_MOVES
from .showdown import ShowdownReplayState
from .transitions import (
    DAMAGE_OUTCOME_NORMAL,
    EFFECTIVENESS_NEUTRAL,
    SIDE_EFFECT_HAZARD_CLEAR,
    SIDE_EFFECT_HAZARD_SET,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_WEATHER_SET,
    SWITCH_REASON_BATON_PASS,
    SWITCH_REASON_LEAD,
    SWITCH_REASON_REPLACEMENT,
    TOKEN_KIND_CANT,
    TOKEN_KIND_MOVE,
    TOKEN_KIND_SWITCH,
    TendencyStats,
    TransitionToken,
    _fold_replay,
    _other_side,
    _tendency_stats_from_fold,
    _validated_slot,
    _Window,
)

# Sub-block status. ACTION: an executed declared action. NEGATED: the side declared an
# action this turn but the engine consumed it with no protocol trace (mid-turn faint
# fizzle — see module docstring). ABSENT: no declaration expected (the empty half of a
# single replacement token).
SUB_BLOCK_ACTION = "action"
SUB_BLOCK_NEGATED = "negated"
SUB_BLOCK_ABSENT = "absent"

# Token phases. TURN: a numbered battle turn's declared-action pair. LEAD: the turn-0
# blind send-out pair. REPLACEMENT: a post-faint switch-in phase (single, or the
# cold double-faint pair). EXTRA: safety valve for an action sequence the merger does
# not recognize (never emitted on the verified corpus; preserves bijection if a future
# protocol shape appears).
PHASE_TURN = "turn"
PHASE_LEAD = "lead"
PHASE_REPLACEMENT = "replacement"
PHASE_EXTRA = "extra"


@dataclass(frozen=True)
class TurnSubBlock:
    """One side's action within a merged token (the per-action fields minus the
    per-token context trio / positional pair, plus the collapse fields).

    ``cant_reason`` is the RestTalk collapse: the ``|cant|`` reason that preceded a
    Sleep Talk click chain (the representative action is the called execution, or the
    click itself when the call produced no execution). ``baton_pass_species`` is the
    Baton Pass collapse: the incoming species of the mid-turn completion switch.
    NEGATED/ABSENT sub-blocks carry only ``actor_slot`` (and, for NEGATED, the species
    whose declared action was consumed, when the fold knows it) — every other field
    holds its neutral default.
    """

    status: str
    actor_slot: str
    actor_species: str = ""
    kind: str = ""
    action: str = ""
    called: bool = False
    transformed: bool = False
    damage_fraction: float = 0.0
    damage_outcome: str = DAMAGE_OUTCOME_NORMAL
    crit: bool = False
    miss: bool = False
    ko: bool = False
    pursuit_intercept: bool = False
    n_hits: int = 1
    effectiveness: str = EFFECTIVENESS_NEUTRAL
    side_effect: str = SIDE_EFFECT_NONE
    cant_reason: Optional[str] = None
    baton_pass_species: Optional[str] = None
    residual: Optional[float] = None
    residual_valid: bool = False
    cb_bit: bool = False


@dataclass(frozen=True)
class TurnMergedToken:
    """One turn / lead / replacement phase: up to two sub-blocks in execution order.

    ``first`` is always a real (ACTION) sub-block; ``second`` is ACTION, NEGATED, or
    ABSENT. Sub-block order is engine emission order, which for a normal turn IS the
    resolution (speed) order — explicit added information vs the per-action stream.
    The context trio is captured at the first sub-block's declaration; ``turn`` is the
    battle turn the phase belongs to (replacements after a residual faint keep the turn
    of the faint).
    """

    turn: int
    phase: str
    first: TurnSubBlock
    second: TurnSubBlock
    own_spikes_layers: int = 0
    opp_spikes_layers: int = 0
    weather: Optional[str] = None


def extract_turn_merged_tokens(
    replay: ShowdownReplayState,
    *,
    perspective_slot: str,
) -> tuple[TurnMergedToken, ...]:
    """The turn-merged token stream, in chronological phase order."""
    perspective = _validated_slot(perspective_slot)
    fold = _fold_replay(replay, perspective_slot=perspective)
    return _merge_fold(fold)


def extract_turn_merged_and_tendencies(
    replay: ShowdownReplayState,
    *,
    perspective_slot: str,
) -> tuple[tuple[TurnMergedToken, ...], TendencyStats]:
    """Both encode-side products from ONE fold of the replay (hot-path shape, matching
    ``extract_transitions_and_tendencies``)."""
    perspective = _validated_slot(perspective_slot)
    fold = _fold_replay(replay, perspective_slot=perspective)
    return _merge_fold(fold), _tendency_stats_from_fold(fold, perspective_slot=perspective)


def flatten_turn_merged_tokens(
    tokens: tuple[TurnMergedToken, ...] | list[TurnMergedToken],
) -> tuple[TransitionToken, ...]:
    """Reconstruct the per-action token stream (the bijection-test inverse).

    Exact except for the documented trio merge: tokens rebuilt from a ``second``
    sub-block inherit the merged token's (first-mover) context trio.
    """
    out: list[TransitionToken] = []
    for token in tokens:
        for sub in (token.first, token.second):
            if sub.status != SUB_BLOCK_ACTION:
                continue
            out.extend(_expand_sub_block(token, sub))
    return tuple(out)


def _expand_sub_block(token: TurnMergedToken, sub: TurnSubBlock) -> list[TransitionToken]:
    trio = {
        "own_spikes_layers": token.own_spikes_layers,
        "opp_spikes_layers": token.opp_spikes_layers,
        "weather": token.weather,
    }
    expanded: list[TransitionToken] = []
    if sub.cant_reason is not None:
        expanded.append(
            TransitionToken(
                turn=token.turn,
                actor_slot=sub.actor_slot,
                actor_species=sub.actor_species,
                kind=TOKEN_KIND_CANT,
                action=sub.cant_reason,
                transformed=sub.transformed,
                **trio,
            )
        )
        if sub.called:
            # The Sleep Talk click line: protocol-constant (no damage/effect events of
            # its own — the merger verified default fields before collapsing).
            expanded.append(
                TransitionToken(
                    turn=token.turn,
                    actor_slot=sub.actor_slot,
                    actor_species=sub.actor_species,
                    kind=TOKEN_KIND_MOVE,
                    action="sleeptalk",
                    transformed=sub.transformed,
                    **trio,
                )
            )
    expanded.append(
        TransitionToken(
            turn=token.turn,
            actor_slot=sub.actor_slot,
            actor_species=sub.actor_species,
            kind=sub.kind,
            action=sub.action,
            called=sub.called,
            transformed=sub.transformed,
            damage_fraction=sub.damage_fraction,
            damage_outcome=sub.damage_outcome,
            crit=sub.crit,
            miss=sub.miss,
            ko=sub.ko,
            pursuit_intercept=sub.pursuit_intercept,
            n_hits=sub.n_hits,
            effectiveness=sub.effectiveness,
            side_effect=sub.side_effect,
            residual=sub.residual,
            residual_valid=sub.residual_valid,
            cb_bit=sub.cb_bit,
            **trio,
        )
    )
    if sub.baton_pass_species is not None:
        expanded.append(
            TransitionToken(
                turn=token.turn,
                actor_slot=sub.actor_slot,
                actor_species=sub.baton_pass_species,
                kind=TOKEN_KIND_SWITCH,
                action=sub.baton_pass_species,
                **trio,
            )
        )
    return expanded


# ---------------------------------------------------------------------------
# Merge internals.
# ---------------------------------------------------------------------------


@dataclass
class _Chain:
    """One side's declared action within a turn: representative window + collapse."""

    side: str
    start_index: int  # event index of the chain's first window (ordering key)
    representative: _Window
    cant_reason: Optional[str] = None
    baton_pass_species: Optional[str] = None


def _merge_fold(fold) -> tuple[TurnMergedToken, ...]:
    windows = list(fold.windows)
    tokens: list[TurnMergedToken] = []

    index = 0
    # Lead phase: the opening blind send-out pair (windows before |turn|1).
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
                    else TurnSubBlock(status=SUB_BLOCK_ABSENT, actor_slot=_other_side(first.side))
                ),
                own_spikes_layers=first.own_spikes_layers,
                opp_spikes_layers=first.opp_spikes_layers,
                weather=first.weather,
            )
        )
        for extra in leads[2:]:  # unreachable in singles; bijection safety valve
            tokens.append(_single_phase(extra, PHASE_EXTRA))

    # Remaining windows, grouped by turn (non-decreasing in the stream).
    while index < len(windows):
        turn = windows[index].turn
        turn_windows: list[_Window] = []
        while index < len(windows) and windows[index].turn == turn:
            turn_windows.append(windows[index])
            index += 1
        tokens.extend(_merge_turn(turn, turn_windows, fold.turn_start_occupants))
    return tuple(tokens)


def _merge_turn(
    turn: int,
    turn_windows: list[_Window],
    turn_start_occupants: dict[int, dict[str, str]],
) -> list[TurnMergedToken]:
    declared: list[_Window] = []
    replacements: list[_Window] = []
    for position, window in enumerate(turn_windows):
        if window.switch_reason == SWITCH_REASON_REPLACEMENT:
            # Pursuit KO-intercept continuation: the engine completes the previously
            # chosen switch in the same breath (no forceSwitch cycle) — that IS the
            # side's declared action, not a replacement phase.
            previous = turn_windows[position - 1] if position > 0 else None
            if (
                previous is not None
                and previous.kind == TOKEN_KIND_MOVE
                and previous.pursuit_intercept
                and previous.ko
                and previous.side != window.side
            ):
                declared.append(window)
            else:
                replacements.append(window)
        else:
            declared.append(window)

    chains: list[_Chain] = []
    extras: list[_Window] = []
    for side in ("p1", "p2"):
        side_windows = [window for window in declared if window.side == side]
        if not side_windows:
            continue
        chain, side_extras = _reduce_side_chain(side, side_windows)
        chains.append(chain)
        extras.extend(side_extras)
    chains.sort(key=lambda chain: chain.start_index)

    phases: list[tuple[int, TurnMergedToken]] = []
    if chains:
        first_chain = chains[0]
        if len(chains) > 1:
            second = _chain_sub_block(chains[1])
        else:
            second = _negated_sub_block(
                _other_side(first_chain.side), turn, turn_start_occupants
            )
        anchor = first_chain.representative
        phases.append(
            (
                first_chain.start_index,
                TurnMergedToken(
                    turn=turn,
                    phase=PHASE_TURN,
                    first=_chain_sub_block(first_chain),
                    second=second,
                    own_spikes_layers=anchor.own_spikes_layers,
                    opp_spikes_layers=anchor.opp_spikes_layers,
                    weather=anchor.weather,
                ),
            )
        )
    for extra in extras:
        phases.append((extra.event_index, _single_phase(extra, PHASE_EXTRA)))

    # Replacement phases: cold pairs merge, sequential faints stay single.
    position = 0
    while position < len(replacements):
        window = replacements[position]
        partner = replacements[position + 1] if position + 1 < len(replacements) else None
        if (
            window.other_side_pending_replacement
            and partner is not None
            and partner.side == _other_side(window.side)
        ):
            phases.append(
                (
                    window.event_index,
                    TurnMergedToken(
                        turn=turn,
                        phase=PHASE_REPLACEMENT,
                        first=_action_sub_block(window),
                        second=_action_sub_block(partner),
                        own_spikes_layers=window.own_spikes_layers,
                        opp_spikes_layers=window.opp_spikes_layers,
                        weather=window.weather,
                    ),
                )
            )
            position += 2
        else:
            phases.append((window.event_index, _single_phase(window, PHASE_REPLACEMENT)))
            position += 1

    phases.sort(key=lambda item: item[0])
    return [token for _, token in phases]


def _reduce_side_chain(side: str, seq: list[_Window]) -> tuple[_Chain, list[_Window]]:
    """Collapse one side's declared windows of a turn into a representative chain.

    Recognized shapes (everything else falls into the EXTRA safety valve):
    ``[any]``; ``[cant, click]`` / ``[cant, click, called-exec]`` (RestTalk — the click
    must be protocol-constant to collapse); a trailing Baton Pass completion switch
    after a move.
    """
    start_index = seq[0].event_index
    cant_reason: Optional[str] = None
    rest = seq
    if (
        len(rest) >= 2
        and rest[0].kind == TOKEN_KIND_CANT
        and rest[1].kind == TOKEN_KIND_MOVE
        and rest[1].action in _CALLER_MOVES
        and _is_protocol_constant(rest[0])
    ):
        if len(rest) >= 3 and rest[2].kind == TOKEN_KIND_MOVE and rest[2].called:
            if _is_protocol_constant(rest[1]):
                cant_reason = rest[0].action
                rest = rest[2:]
        else:
            cant_reason = rest[0].action
            rest = rest[1:]

    representative = rest[0]
    rest = rest[1:]

    baton_pass_species: Optional[str] = None
    if (
        rest
        and representative.kind == TOKEN_KIND_MOVE
        and rest[0].kind == TOKEN_KIND_SWITCH
        and rest[0].switch_reason == SWITCH_REASON_BATON_PASS
    ):
        baton_pass_species = rest[0].action
        rest = rest[1:]

    chain = _Chain(
        side=side,
        start_index=start_index,
        representative=representative,
        cant_reason=cant_reason,
        baton_pass_species=baton_pass_species,
    )
    return chain, list(rest)


def _is_protocol_constant(window: _Window) -> bool:
    """True when a chain-interior window carries no information beyond its identity
    (so flatten can resynthesize it exactly). ``transformed`` is inherited from the
    representative and deliberately not checked."""
    return (
        window.damage_fraction == 0.0
        and window.outcome == DAMAGE_OUTCOME_NORMAL
        and not window.crit
        and not window.miss
        and not window.ko
        and not window.pursuit_intercept
        and window.n_hits == 1
        and window.effectiveness == EFFECTIVENESS_NEUTRAL
        and window.side_effect == SIDE_EFFECT_NONE
        and not window.called
    )


def _action_sub_block(window: _Window, **collapse) -> TurnSubBlock:
    return TurnSubBlock(
        status=SUB_BLOCK_ACTION,
        actor_slot=window.side,
        actor_species=window.species,
        kind=window.kind,
        action=window.action,
        called=window.called,
        transformed=window.transformed,
        damage_fraction=window.damage_fraction,
        damage_outcome=window.outcome,
        crit=window.crit,
        miss=window.miss,
        ko=window.ko,
        pursuit_intercept=window.pursuit_intercept,
        n_hits=window.n_hits,
        effectiveness=window.effectiveness,
        side_effect=window.side_effect,
        **collapse,
    )


def _chain_sub_block(chain: _Chain) -> TurnSubBlock:
    return _action_sub_block(
        chain.representative,
        cant_reason=chain.cant_reason,
        baton_pass_species=chain.baton_pass_species,
    )


def _negated_sub_block(
    side: str, turn: int, turn_start_occupants: dict[int, dict[str, str]]
) -> TurnSubBlock:
    species = turn_start_occupants.get(turn, {}).get(side, "")
    return TurnSubBlock(status=SUB_BLOCK_NEGATED, actor_slot=side, actor_species=species)


def _single_phase(window: _Window, phase: str) -> TurnMergedToken:
    return TurnMergedToken(
        turn=window.turn,
        phase=phase,
        first=_action_sub_block(window),
        second=TurnSubBlock(status=SUB_BLOCK_ABSENT, actor_slot=_other_side(window.side)),
        own_spikes_layers=window.own_spikes_layers,
        opp_spikes_layers=window.opp_spikes_layers,
        weather=window.weather,
    )
