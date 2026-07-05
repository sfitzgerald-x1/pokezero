"""Turn-merged transition tokens (v2.1 batch 3): one token per TURN, two sub-blocks.

Pure extraction layer over the same replay fold as :mod:`pokezero.transitions`; the
encode side lives in ``showdown.py`` as observation schema v2.2 — the third entry in the
checkpoint-driven dual-schema table #512 built (v2/v2.1 artifacts stay first-class; a
checkpoint's stamped schema selects the encode).

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
  whole turn — the old 64-action horizon is ≈ 32 turn tokens. See the v2.2 encode
  docstring in ``showdown.py`` for the loud unit-change note on K=64 configs.

Equivalence gate: :func:`flatten_turn_merged_tokens` reconstructs the per-action token
stream exactly, modulo ONE documented merge: the context trio is stored once per merged
token, so a second sub-block (or second half of a pair) reconstructs with the first
mover's trio. The two can differ only when the first mover's own action changed the trio
(hazard set/clear or weather set — including ability weather on a switch-in), which is
exactly the ``side_effect`` ∈ {hazard-set, hazard-clear, weather-set} predicate the
bijection test uses as its allowance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

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
# action this turn and the engine PROVABLY consumed it with no protocol trace (mid-turn
# faint fizzle, or the turn closed without it — see module docstring). PENDING: the turn
# is still open with no consumption proof — the side's action simply has not resolved
# yet (a replay prefix cut at a mid-turn forceSwitch boundary, e.g. the Baton Pass
# completion choice; review MED-1 — encoding these as negated would assert the
# free-pivot semantics exactly where they are false). ABSENT: no declaration expected
# (the empty half of a single replacement token).
SUB_BLOCK_ACTION = "action"
SUB_BLOCK_NEGATED = "negated"
SUB_BLOCK_PENDING = "pending"
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
    # Fraction of the ACTOR'S max HP lost to its own action (v2.2 SELF_HP_COST; see
    # transitions._SELF_COST_FROM_TAGS for the source classification).
    self_hp_cost: float = 0.0
    damage_outcome: str = DAMAGE_OUTCOME_NORMAL
    crit: bool = False
    miss: bool = False
    ko: bool = False
    pursuit_intercept: bool = False
    n_hits: int = 1
    effectiveness: str = EFFECTIVENESS_NEUTRAL
    side_effect: str = SIDE_EFFECT_NONE
    # Defender identity at declaration (v2.1 batch 1 field; move sub-blocks only). For a
    # hazard-sack redirect this is what would name the mid-turn replacement — but the
    # engine-verified disposition is a full fizzle, so a NEGATED sub-block never has one.
    defender_species: Optional[str] = None
    cant_reason: Optional[str] = None
    baton_pass_species: Optional[str] = None
    residual: Optional[float] = None
    residual_valid: bool = False
    cb_bit: bool = False
    # Defender-side investment conclusion code (#513; as-of-strike, rides assessed OWN
    # move sub-blocks and describes the struck defender). Populated only via
    # annotate_turn_merged_tokens from an investment-annotated per-action stream.
    investment: float = 0.0


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


def extract_transition_products(
    replay: ShowdownReplayState,
    *,
    perspective_slot: str,
) -> tuple[tuple[TransitionToken, ...], tuple[TurnMergedToken, ...], TendencyStats]:
    """Per-action tokens + turn-merged tokens + tendencies from ONE fold.

    The v2.2 observe path needs all three: the per-action stream stays the Tier-2
    annotation substrate (and the per-mon pinned-bit derivation source), the merged
    stream is what the transition block encodes. Merging is O(windows) on top of the
    shared fold.
    """
    perspective = _validated_slot(perspective_slot)
    fold = _fold_replay(replay, perspective_slot=perspective)
    return (
        fold.tokens,
        _merge_fold(fold),
        _tendency_stats_from_fold(fold, perspective_slot=perspective),
    )


def annotate_turn_merged_tokens(
    merged: tuple[TurnMergedToken, ...],
    annotated_per_action: tuple[TransitionToken, ...],
) -> tuple[TurnMergedToken, ...]:
    """Copy Tier-2 annotations (residual / validity / cb_bit / investment) onto the
    merged stream.

    ``annotated_per_action`` must be the annotated form of exactly the per-action stream
    this merged stream was built from (``Tier2LiveTracker.annotate`` + the #513
    investment tracker only rewrite the four Tier-2-family fields). The flatten expansion order gives the exact positional
    correspondence between sub-blocks and per-action tokens; chain-interior tokens (the
    cant line and the protocol-constant Sleep Talk click, plus Baton Pass completions)
    are never assessed strikes, so only each sub-block's representative is mapped.
    """
    total = sum(
        _expansion_length(sub)
        for token in merged
        for sub in (token.first, token.second)
        if sub.status == SUB_BLOCK_ACTION
    )
    if total != len(annotated_per_action):
        raise ValueError(
            "annotate_turn_merged_tokens: merged stream does not correspond to the "
            "annotated per-action stream (length mismatch after flatten)."
        )
    out: list[TurnMergedToken] = []
    cursor = 0
    for token in merged:
        updates: dict[str, TurnSubBlock] = {}
        for position, sub in (("first", token.first), ("second", token.second)):
            if sub.status != SUB_BLOCK_ACTION:
                continue
            representative_offset = 0
            if sub.cant_reason is not None:
                representative_offset += 1  # the cant token precedes the representative
                if sub.called:
                    representative_offset += 1  # so does the synthesized click
            source = annotated_per_action[cursor + representative_offset]
            cursor += _expansion_length(sub)
            if (
                source.residual == sub.residual
                and source.residual_valid == sub.residual_valid
                and source.cb_bit == sub.cb_bit
                and source.investment == sub.investment
            ):
                continue
            updates[position] = replace(
                sub,
                residual=source.residual,
                residual_valid=source.residual_valid,
                cb_bit=source.cb_bit,
                investment=source.investment,
            )
        out.append(replace(token, **updates) if updates else token)
    return tuple(out)


def _expansion_length(sub: "TurnSubBlock") -> int:
    """How many per-action tokens this sub-block flattens to."""
    length = 1
    if sub.cant_reason is not None:
        length += 1
        if sub.called:
            length += 1
    if sub.baton_pass_species is not None:
        length += 1
    return length


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
            # its own — the merger verified default fields before collapsing). The click
            # is self-targeted, so its defender identity is the actor itself.
            expanded.append(
                TransitionToken(
                    turn=token.turn,
                    actor_slot=sub.actor_slot,
                    actor_species=sub.actor_species,
                    kind=TOKEN_KIND_MOVE,
                    action="sleeptalk",
                    transformed=sub.transformed,
                    defender_species=sub.actor_species,
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
            self_hp_cost=sub.self_hp_cost,
            damage_outcome=sub.damage_outcome,
            crit=sub.crit,
            miss=sub.miss,
            ko=sub.ko,
            pursuit_intercept=sub.pursuit_intercept,
            n_hits=sub.n_hits,
            effectiveness=sub.effectiveness,
            side_effect=sub.side_effect,
            defender_species=sub.defender_species,
            residual=sub.residual,
            residual_valid=sub.residual_valid,
            cb_bit=sub.cb_bit,
            investment=sub.investment,
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
        tokens.extend(
            _merge_turn(
                turn,
                turn_windows,
                fold.turn_start_occupants,
                consumption_confirmed=(
                    turn in fold.completed_turns or turn in fold.fainted_turns
                ),
            )
        )
    return tuple(tokens)


def _merge_turn(
    turn: int,
    turn_windows: list[_Window],
    turn_start_occupants: dict[int, dict[str, str]],
    *,
    consumption_confirmed: bool,
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
            # NEGATED only on proof of consumption (the turn closed, or a mid-turn
            # faint — which the engine turns into a full cancel of every remaining
            # action); an open turn with neither is a PENDING resolution, not a
            # free pivot (review MED-1: the Baton Pass completion boundary).
            second = _missing_sub_block(
                _other_side(first_chain.side),
                turn,
                turn_start_occupants,
                status=SUB_BLOCK_NEGATED if consumption_confirmed else SUB_BLOCK_PENDING,
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
    ``[any]``; ``[cant, sleeptalk-click]`` / ``[cant, sleeptalk-click, called-exec]``
    (RestTalk — the cant/click must be protocol-constant to collapse, and the click must
    be Sleep Talk, the only caller flatten resynthesizes); a trailing Baton Pass
    completion switch after a move.
    """
    start_index = seq[0].event_index
    cant_reason: Optional[str] = None
    rest = seq
    if (
        len(rest) >= 2
        and rest[0].kind == TOKEN_KIND_CANT
        and rest[1].kind == TOKEN_KIND_MOVE
        # Sleep Talk is the only reachable caller in gen3 randbats (design doc, verified
        # on the movepools) AND the only click flatten resynthesizes — restricting the
        # collapse to it makes that invariant structural (review NIT); any other caller
        # chain would fall to the EXTRA safety valve, preserving bijection.
        and rest[1].action == "sleeptalk"
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
        and window.self_hp_cost == 0.0
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
        self_hp_cost=window.self_hp_cost,
        damage_outcome=window.outcome,
        crit=window.crit,
        miss=window.miss,
        ko=window.ko,
        pursuit_intercept=window.pursuit_intercept,
        n_hits=window.n_hits,
        effectiveness=window.effectiveness,
        side_effect=window.side_effect,
        defender_species=window.defender_species,
        **collapse,
    )


def _chain_sub_block(chain: _Chain) -> TurnSubBlock:
    return _action_sub_block(
        chain.representative,
        cant_reason=chain.cant_reason,
        baton_pass_species=chain.baton_pass_species,
    )


def _missing_sub_block(
    side: str,
    turn: int,
    turn_start_occupants: dict[int, dict[str, str]],
    *,
    status: str,
) -> TurnSubBlock:
    species = turn_start_occupants.get(turn, {}).get(side, "")
    return TurnSubBlock(status=status, actor_slot=side, actor_species=species)


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
