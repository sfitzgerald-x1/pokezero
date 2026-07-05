"""Transition-token and tendency-stats extraction (next-train readiness, PR B).

Pure functions over parsed replay state (:class:`~pokezero.showdown.ShowdownReplayState`).
No observation-encoder changes and no spec bump live here — encoding the tokens into the
new observation layout is PR C's scope.

Spec: ``docs/observation_compression_design.md`` corrections layer (items 9-14 govern the
token schema and emission rules) plus ``docs/gen3_interaction_inventory.md`` (context trio,
charge-turn rule, Pursuit event-order detection).

Emission rules implemented (corrections item 11 + 14):
- One token per declared action: every ``|move|`` line and every ``|switch|`` line —
  turn-1 lead send-outs, faint-replacements, Baton Pass completions, and
  Pursuit-intercepted switches each emit their own switch token.
- ``|cant|`` no-action turns (sleep / para / flinch / recharge / freeze) emit a token with
  the reason as the action id (corrections item 14).
- ``|drag|`` (Roar / Whirlwind) does NOT emit a token: the drag is RNG-forced, not a
  declared action — the phazer's move token is the declared action.
- A Sleep Talk turn emits THREE tokens, matching the engine's three protocol lines:
  the ``|cant|slp`` token, the Sleep Talk click token, and the called execution with
  ``called=True`` (detected via ``[from] Sleep Talk`` / ``[from]move: Sleep Talk``), so
  the damage-carrying token is self-describing without charging set evidence to a click
  that never happened. Tendency opportunity counting collapses the turn to ONE decision
  (see :class:`TendencyStats`).
- Pursuit interception is detected via the engine's explicit marker
  ``|-activate|<target>|move: Pursuit`` (emitted on every interception — through a
  Substitute and on Baton Pass switch-outs included). Still protocol-tautological, no
  mechanics model. Residual-phase ordering heuristics are deliberately NOT used: real
  faint-replacements arrive BEFORE ``|upkeep|``, so any pre-upkeep ordering rule would
  misread plain Pursuit KOs as intercepts.

Attribution rules:
- Damage fractions come only from untagged ``|-damage|`` lines on the pending move's
  defender. Chip damage (``[from] psn/brn/Sandstorm/Spikes/Leech Seed/...``) and recoil
  are ``[from]``-tagged and NEVER produce tokens or damage fractions; a tagged hit also
  vetoes KO attribution to the move (a chip faint is not a move KO).
- Side effects attach only within the acting move's own contiguous event chunk: a
  window closes at the next action line, at the blank ``|`` chunk separator the engine
  emits between action chunks and the residual phase, and at ``|upkeep|`` — so
  residual-phase events (Leech Seed transfers, Yawn's delayed sleep) can never stamp a
  category onto an unrelated action token. ``[silent]``-tagged heals are excluded from
  attribution entirely (Leech Seed's recipient heal is ``[silent]``; Rest's heal is too,
  so Rest carries side_effect "none" at Tier 1 — its effect stays visible through the
  status/sleep channels).
- Healing is a side-effect category only, never a magnitude field (every gen-3 heal
  magnitude is derivable from public state).
- History/stats attribution keys on slot + BASE species (the protocol ident) — the
  Transform identity rule. A transformed actor's move tokens carry ``transformed=True``
  and keep the base species.

Tier-2 deferrals (reserved here, populated by PR D behind its precision gate):
- ``residual`` / ``residual_valid`` are reserved zero/None fields (corrections item 10);
  no expected-damage computation exists in this module.
- The damage-calc midground comparison (opponent-move-better-vs-incoming) is
  Tier-2-gated (corrections item 12) and not computed.
- The typing-explained immune split needs a type chart and is Tier-2-gated (corrections
  item 14); ``TendencyStats`` therefore carries no typing-explained-immune counter.
- Pursuit KO-intercept switch COMPLETION semantics (whether the declared replacement's
  entry counters double-count) remain the open sim experiment (inventory item 6); the
  intercept flag itself is exact via the ``-activate`` marker regardless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .belief import _CALLER_MOVES, _called_move_source
from .showdown import (
    ShowdownReplayState,
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

# Damage-outcome enum (corrections item 9). Two evidence classes: negation
# (blocked/immune/absorbed — no damage event occurred) and truncation
# (hit-sub/broke-sub/endured — full damage was dealt to a proxy or clipped).
DAMAGE_OUTCOME_NORMAL = "normal"
DAMAGE_OUTCOME_BLOCKED = "blocked"
DAMAGE_OUTCOME_IMMUNE = "immune"
DAMAGE_OUTCOME_ABSORBED = "absorbed"
DAMAGE_OUTCOME_HIT_SUB = "hit-sub"
DAMAGE_OUTCOME_BROKE_SUB = "broke-sub"
DAMAGE_OUTCOME_ENDURED = "endured"
# When several outcome signals fire in one action window, the lowest rank wins
# (e.g. hit-sub then |-end|Substitute upgrades to broke-sub).
_OUTCOME_RANK = {
    DAMAGE_OUTCOME_ABSORBED: 0,
    DAMAGE_OUTCOME_IMMUNE: 1,
    DAMAGE_OUTCOME_BLOCKED: 2,
    DAMAGE_OUTCOME_BROKE_SUB: 3,
    DAMAGE_OUTCOME_HIT_SUB: 4,
    DAMAGE_OUTCOME_ENDURED: 5,
    DAMAGE_OUTCOME_NORMAL: 9,
}

# Side-effect category (single value per token; the docs' consolidated vocabulary).
SIDE_EFFECT_NONE = "none"
SIDE_EFFECT_STATUS_INFLICTED = "status-inflicted"
SIDE_EFFECT_HAZARD_SET = "hazard-set"
SIDE_EFFECT_HAZARD_CLEAR = "hazard-clear"
SIDE_EFFECT_WEATHER_SET = "weather-set"
SIDE_EFFECT_BOOST = "boost"
SIDE_EFFECT_DRAIN = "drain"
SIDE_EFFECT_HEAL = "heal"
SIDE_EFFECT_CHARGING = "charging"
# Deterministic winner when one action produces several category signals (e.g. Giga
# Drain's damage + drain heal, Rest's self-status + heal).
_SIDE_EFFECT_RANK = {
    SIDE_EFFECT_CHARGING: 0,
    SIDE_EFFECT_DRAIN: 1,
    SIDE_EFFECT_HAZARD_SET: 2,
    SIDE_EFFECT_HAZARD_CLEAR: 3,
    SIDE_EFFECT_WEATHER_SET: 4,
    SIDE_EFFECT_STATUS_INFLICTED: 5,
    SIDE_EFFECT_HEAL: 6,
    SIDE_EFFECT_BOOST: 7,
    SIDE_EFFECT_NONE: 99,
}

EFFECTIVENESS_NEUTRAL = "neutral"
EFFECTIVENESS_SUPER = "super"
EFFECTIVENESS_RESISTED = "resisted"
EFFECTIVENESS_IMMUNE = "immune"

TOKEN_KIND_MOVE = "move"
TOKEN_KIND_SWITCH = "switch"
TOKEN_KIND_CANT = "cant"

# Absorb-class abilities (negation outcome distinct from immune: the defender gained).
_ABSORB_ABILITIES = frozenset({"voltabsorb", "waterabsorb", "flashfire"})
# |cant| reasons that are still a decision opportunity (the player could have switched);
# a recharge turn is a locked no-choice turn and is excluded from opportunity counts.
_CANT_NO_CHOICE_REASONS = frozenset({"recharge"})

_FROM_TAG_RE = re.compile(r"\[from\]\s*([^|\[\]]*)")
_OF_TAG_RE = re.compile(r"\[of\]\s*(p[12])")


@dataclass(frozen=True)
class TransitionToken:
    """One declared action from the public log (corrections item 9's canonical fields).

    The positional pair is (absolute ``turn``, turns-ago); ``turns_ago`` is computed by
    the consumer at encode time (it depends on the observation turn), so only ``turn`` is
    recorded here. Within-turn resolution order is the tuple order returned by
    :func:`extract_transition_tokens` (ties in ``turn`` break by position).

    ``actor_slot``/``actor_species`` follow the Transform identity rule: always the
    protocol ident's side and BASE species, never the acting (copied) species.

    ``action`` carries THREE vocabularies, disambiguated by ``kind`` — a normalized move
    id (``"rockslide"``) for moves, a display-form species (``"Starmie"``, matching
    ``ShowdownPokemon.species``) for switches, and a raw reason id (``"slp"``) for cant
    tokens. PR C's encoder MUST branch on ``kind`` and embed per-kind; the vocabularies
    are deliberately not merged into one id space here.

    ``residual``/``residual_valid`` are reserved Tier-2 fields (corrections item 10):
    always ``None``/``False`` in Tier 1; PR D populates them behind its precision gate.
    """

    turn: int
    actor_slot: str  # protocol side, "p1" / "p2"
    actor_species: str  # base species (display form, e.g. "Charizard")
    kind: str  # TOKEN_KIND_MOVE / TOKEN_KIND_SWITCH / TOKEN_KIND_CANT
    action: str  # move id ("flamethrower"), incoming species ("Starmie"), or cant reason
    called: bool = False  # Sleep Talk execution ([from] Sleep Talk / [from]move: Sleep Talk)
    transformed: bool = False  # actor was transformed when acting
    damage_fraction: float = 0.0  # fraction of defender max HP from untagged -damage lines
    damage_outcome: str = DAMAGE_OUTCOME_NORMAL
    crit: bool = False
    miss: bool = False
    ko: bool = False  # the move's own damage fainted the defender (chip faints excluded)
    pursuit_intercept: bool = False  # |-activate|<target>|move: Pursuit marker; see docstring
    n_hits: int = 1  # from |-hitcount| (Bonemerang: always exactly 2 in this format)
    effectiveness: str = EFFECTIVENESS_NEUTRAL
    side_effect: str = SIDE_EFFECT_NONE
    # Context trio (gen3 inventory: the principled derivability exception), captured at
    # action-declaration time (before the action's own effects land). "own" is the
    # perspective side.
    own_spikes_layers: int = 0
    opp_spikes_layers: int = 0
    weather: Optional[str] = None
    # Defender identity (spec v2.1): the BASE species (display form, Transform identity
    # rule — the slot occupant recorded at its switch-in, never the acting/copied species)
    # occupying the defender side when a MOVE was declared. None on switch/cant tokens and
    # when the extractor cannot resolve the occupant. Extracted for every replay (pure
    # extraction, schema-independent); only a v2.1 encode reads it — the defender is
    # inferable from interleaved switch tokens EXCEPT when K-truncation drops the anchoring
    # switch, and damage_fraction is defender-relative.
    defender_species: Optional[str] = None
    # Tier-2 fields (populated only by pokezero.tier2 behind the #505 precision gate;
    # always None/False/False from this module's Tier-1 extraction). ``cb_bit`` is the
    # two-strike Choice Band conclusion for the ACTING mon as of this strike — monotone
    # within a battle, set on assessed opponent move tokens only.
    residual: Optional[float] = None
    residual_valid: bool = False
    cb_bit: bool = False


@dataclass(frozen=True)
class OpponentMonTendency:
    """Per-opponent-mon tendency triple, keyed slot + base species (Transform rule)."""

    slot: str
    species: str
    switched_out_before_attacking: int
    stayed_and_attacked: int
    turns_active: int


@dataclass(frozen=True)
class OpponentWeatherReveal:
    """The opponent set this weather this game; ``from_ability`` marks a permanent
    (ability-sourced) reveal vs a 5-turn move weather."""

    weather: str
    from_ability: bool


@dataclass(frozen=True)
class TendencyStats:
    """(count, opportunity) evidence-mass pairs (never bare rates) per the design doc.

    Tier-1 prediction-channel inputs only: ``blocked_on_our_attack_count`` (a "no-predict"
    input) and ``pursuit_intercept_predict_count`` (the opponent's doubled Pursuit — an
    affirmative switch-predict observation) plus the raw ``my_switch_turn_count``
    denominator. The damage-calc midground comparison and the typing-explained immune
    split are Tier-2-gated and deliberately absent (corrections items 12 and 14).
    """

    perspective_slot: str
    opponent_slot: str
    # Global switch tendency: voluntary opponent switches / opponent decision
    # opportunities. An opportunity is a (side, turn) with at least one controllable
    # decision token — a non-called, non-locked move; a voluntary switch; or a |cant|
    # whose reason leaves the switch choice open (sleep/para/flinch, NOT recharge) —
    # counted at most ONCE per side per turn, so a RestTalk turn (cant + click + called
    # execution, three tokens) is exactly one opportunity. Locked continuations (Solar
    # Beam release, Thrash-class [from]lockedmove) contribute zero; lead send-outs,
    # faint-replacements, drags, and Baton Pass completions are not stay-or-switch
    # decisions and count on neither side of the pair.
    opponent_switch_count: int
    opponent_decision_opportunities: int
    opponent_mon_tendencies: tuple[OpponentMonTendency, ...]
    opponent_weather_reveals: tuple[OpponentWeatherReveal, ...]
    # Tier-1 midground/prediction inputs (corrections item 13 routing).
    blocked_on_our_attack_count: int
    pursuit_intercept_predict_count: int
    my_switch_turn_count: int


@dataclass
class _Window:
    """Mutable accumulator for one declared action's event window."""

    event_index: int
    turn: int
    side: str
    species: str
    kind: str
    action: str
    defender_side: Optional[str]
    defender_species: Optional[str] = None
    called: bool = False
    transformed: bool = False
    own_spikes_layers: int = 0
    opp_spikes_layers: int = 0
    weather: Optional[str] = None
    damage_fraction: float = 0.0
    outcome: str = DAMAGE_OUTCOME_NORMAL
    crit: bool = False
    miss: bool = False
    ko: bool = False
    pursuit_intercept: bool = False
    n_hits: int = 1
    effectiveness: str = EFFECTIVENESS_NEUTRAL
    side_effect: str = SIDE_EFFECT_NONE
    # KO guard: True while the last damage the defender took in this window was the
    # move's own (untagged); a tagged (chip) hit flips it back off.
    defender_hit_by_move: bool = False
    # Tendency meta (not token fields).
    voluntary_switch: bool = False
    # Locked continuation (Solar Beam release / [from]lockedmove): the |move| line is
    # real history but not a controllable decision — excluded from opportunities.
    locked_continuation: bool = False

    def upgrade_outcome(self, outcome: str) -> None:
        if _OUTCOME_RANK[outcome] < _OUTCOME_RANK[self.outcome]:
            self.outcome = outcome

    def upgrade_side_effect(self, category: str) -> None:
        if _SIDE_EFFECT_RANK[category] < _SIDE_EFFECT_RANK[self.side_effect]:
            self.side_effect = category


@dataclass
class _StayRecord:
    species: str
    moved: bool = False


@dataclass
class _MonCounters:
    switched_out_before_attacking: int = 0
    stayed_and_attacked: int = 0
    turns_active: int = 0


@dataclass
class _FoldResult:
    tokens: tuple[TransitionToken, ...] = ()
    windows: tuple[_Window, ...] = ()
    # (side, weather id, from_ability) reveal records, in event order.
    weather_reveals: tuple[tuple[str, str, bool], ...] = ()
    # (side, species) -> counters.
    mon_counters: dict[tuple[str, str], _MonCounters] = field(default_factory=dict)


def extract_transition_tokens(
    replay: ShowdownReplayState,
    *,
    perspective_slot: str,
) -> tuple[TransitionToken, ...]:
    """One :class:`TransitionToken` per declared action, in within-turn resolution order.

    ``perspective_slot`` orients the context trio (``own_spikes_layers`` is that side's
    hazard layers); actor attribution stays in absolute protocol slots.
    """
    return _fold_replay(replay, perspective_slot=perspective_slot).tokens


def extract_tendency_stats(
    replay: ShowdownReplayState,
    *,
    perspective_slot: str,
) -> TendencyStats:
    """(count, opportunity) tendency pairs for ``perspective_slot`` against its opponent."""
    perspective = _validated_slot(perspective_slot)
    fold = _fold_replay(replay, perspective_slot=perspective)
    return _tendency_stats_from_fold(fold, perspective_slot=perspective)


def extract_transitions_and_tendencies(
    replay: ShowdownReplayState,
    *,
    perspective_slot: str,
) -> tuple[tuple[TransitionToken, ...], TendencyStats]:
    """Both extraction products from ONE fold of the replay.

    The per-observe hot path (``normalize_for_player``) needs both; folding the whole log
    twice doubled an O(events) pass for no reason (~2x the encode-side history cost at the
    stall-game tail).
    """
    perspective = _validated_slot(perspective_slot)
    fold = _fold_replay(replay, perspective_slot=perspective)
    return fold.tokens, _tendency_stats_from_fold(fold, perspective_slot=perspective)


def _tendency_stats_from_fold(fold: _FoldResult, *, perspective_slot: str) -> TendencyStats:
    perspective = perspective_slot
    opponent = _other_side(perspective)

    opponent_switches = 0
    blocked_on_our_attack = 0
    pursuit_predicts = 0
    my_switch_turns = 0
    # One decision opportunity per (side, turn) at most: a RestTalk turn emits three
    # tokens (cant + click + called execution) but is a single controllable decision.
    opportunity_turns: set[tuple[str, int]] = set()
    for token, window in zip(fold.tokens, fold.windows):
        voluntary_switch = token.kind == TOKEN_KIND_SWITCH and window.voluntary_switch
        is_decision = (
            (token.kind == TOKEN_KIND_MOVE and not token.called and not window.locked_continuation)
            or voluntary_switch
            or (token.kind == TOKEN_KIND_CANT and token.action not in _CANT_NO_CHOICE_REASONS)
        )
        if is_decision:
            opportunity_turns.add((token.actor_slot, token.turn))
        if token.actor_slot == opponent:
            if voluntary_switch:
                opponent_switches += 1
            if token.kind == TOKEN_KIND_MOVE and token.pursuit_intercept:
                pursuit_predicts += 1
        else:
            if voluntary_switch:
                my_switch_turns += 1
            if token.kind == TOKEN_KIND_MOVE and token.damage_outcome == DAMAGE_OUTCOME_BLOCKED:
                blocked_on_our_attack += 1
    opportunities = sum(1 for side, _ in opportunity_turns if side == opponent)

    mon_tendencies = tuple(
        OpponentMonTendency(
            slot=side,
            species=species,
            switched_out_before_attacking=counters.switched_out_before_attacking,
            stayed_and_attacked=counters.stayed_and_attacked,
            turns_active=counters.turns_active,
        )
        for (side, species), counters in sorted(fold.mon_counters.items())
        if side == opponent
    )

    reveals_by_weather: dict[str, bool] = {}
    for side, weather, from_ability in fold.weather_reveals:
        if side != opponent:
            continue
        reveals_by_weather[weather] = reveals_by_weather.get(weather, False) or from_ability
    weather_reveals = tuple(
        OpponentWeatherReveal(weather=weather, from_ability=from_ability)
        for weather, from_ability in sorted(reveals_by_weather.items())
    )

    return TendencyStats(
        perspective_slot=perspective,
        opponent_slot=opponent,
        opponent_switch_count=opponent_switches,
        opponent_decision_opportunities=opportunities,
        opponent_mon_tendencies=mon_tendencies,
        opponent_weather_reveals=weather_reveals,
        blocked_on_our_attack_count=blocked_on_our_attack,
        pursuit_intercept_predict_count=pursuit_predicts,
        my_switch_turn_count=my_switch_turns,
    )


def _fold_replay(replay: ShowdownReplayState, *, perspective_slot: str) -> _FoldResult:
    perspective = _validated_slot(perspective_slot)
    opponent = _other_side(perspective)

    raw_lines = tuple(event.raw_line for event in replay.public_events)
    side_condition_counts: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
    weather: Optional[str] = None
    turn_number = 0
    hp_fraction: dict[str, float] = {}
    occupant: dict[str, _StayRecord] = {}
    transformed: dict[str, bool] = {"p1": False, "p2": False}
    pending_baton_pass: dict[str, bool] = {"p1": False, "p2": False}
    pending_faint_replacement: dict[str, bool] = {"p1": False, "p2": False}
    lead_seen: dict[str, bool] = {"p1": False, "p2": False}
    # Two-turn charge state (|-prepare|): the side's next use of the same move is the
    # locked release, not a fresh decision. Cleared by any intervening action.
    pending_charge: dict[str, Optional[str]] = {"p1": None, "p2": None}

    windows: list[_Window] = []
    current: Optional[_Window] = None
    weather_reveals: list[tuple[str, str, bool]] = []
    mon_counters: dict[tuple[str, str], _MonCounters] = {}

    def counters_for(side: str, species: str) -> _MonCounters:
        return mon_counters.setdefault((side, species), _MonCounters())

    def open_window(window: _Window) -> None:
        nonlocal current
        if current is not None:
            windows.append(current)
        current = window

    def close_window() -> None:
        nonlocal current
        if current is not None:
            windows.append(current)
            current = None

    def context_trio() -> tuple[int, int, Optional[str]]:
        own = int(side_condition_counts[perspective].get("spikes", 0))
        opp = int(side_condition_counts[opponent].get("spikes", 0))
        return own, opp, weather

    for index, raw_line in enumerate(raw_lines):
        parts = raw_line.split("|")
        event_type = parts[1] if len(parts) > 1 else ""

        # Blank ``|`` chunk separators and ``|upkeep|`` bound the acting move's own
        # contiguous event chunk: nothing in the residual phase may attach to a window.
        if event_type in {"", "upkeep"}:
            close_window()
            continue

        if event_type == "turn":
            close_window()
            try:
                turn_number = int(parts[2])
            except (IndexError, TypeError, ValueError):
                pass
            for side, stay in occupant.items():
                counters_for(side, stay.species).turns_active += 1
            continue

        if event_type == "win":
            close_window()
            continue

        if event_type == "move" and len(parts) >= 4:
            side = _slot_from_ident(parts[2]) or ""
            if side not in {"p1", "p2"}:
                continue
            stay = occupant.get(side)
            species = stay.species if stay is not None else _species_from_ident(parts[2])
            called_source = _called_move_source(raw_line)
            called = called_source in _CALLER_MOVES
            move_id = _normalize_identifier(parts[3])
            # Locked continuation: a Thrash-class [from]lockedmove line, or the release
            # of a two-turn charge this side prepared (Solar Beam) — no fresh decision.
            locked = called_source == "lockedmove" or pending_charge[side] == move_id
            pending_charge[side] = None
            if stay is not None and not stay.moved:
                stay.moved = True
                counters_for(side, stay.species).stayed_and_attacked += 1
            # Any non-Baton-Pass move clears a stale pending-BP flag (mirrors the parser).
            pending_baton_pass[side] = move_id == "batonpass"
            defender = (_slot_from_ident(parts[4]) if len(parts) > 4 else None) or _other_side(side)
            # Defender identity (v2.1 token field): the extractor knows both actives at
            # declaration time. The occupant record carries the BASE species from the
            # switch-in details (Transform identity rule, robust to nicknamed idents).
            # NO ident-derived fallback (#512 review): on a truncated log whose lead
            # switch predates the fold, the target ident's tail is the NICKNAME, and a
            # ``species:<nickname>`` label would land in the OOV bucket — an absent
            # defender beats a nickname-shaped species. Unreachable on well-formed logs
            # (leads always fold first; the corpus gate asserts full coverage there).
            defender_stay = occupant.get(defender)
            defender_species = defender_stay.species if defender_stay is not None else None
            own, opp, current_weather = context_trio()
            window = _Window(
                event_index=index,
                turn=turn_number,
                side=side,
                species=species,
                kind=TOKEN_KIND_MOVE,
                action=move_id,
                defender_side=defender,
                defender_species=defender_species,
                called=called,
                transformed=transformed[side],
                own_spikes_layers=own,
                opp_spikes_layers=opp,
                weather=current_weather,
            )
            window.locked_continuation = locked
            open_window(window)
            continue

        if event_type in {"switch", "drag", "replace"} and len(parts) >= 4:
            side = _slot_from_ident(parts[2]) or ""
            if side not in {"p1", "p2"}:
                continue
            is_lead = not lead_seen[side]
            lead_seen[side] = True
            is_faint_replacement = pending_faint_replacement[side]
            pending_faint_replacement[side] = False
            is_baton_pass = pending_baton_pass[side] or _line_mentions_baton_pass(parts)
            pending_baton_pass[side] = False
            pending_charge[side] = None
            voluntary = (
                event_type == "switch" and not is_lead and not is_faint_replacement and not is_baton_pass
            )
            previous = occupant.get(side)
            if previous is not None and voluntary and not previous.moved:
                counters_for(side, previous.species).switched_out_before_attacking += 1
            species = _species_from_details(parts[3]) or _species_from_ident(parts[2])
            occupant[side] = _StayRecord(species=species)
            transformed[side] = False
            condition = _condition_features(parts[4] if len(parts) > 4 else None)
            if condition.hp_fraction is not None:
                hp_fraction[side] = condition.hp_fraction
            if event_type in {"drag", "replace"}:
                # Forced RNG switch (Roar/Whirlwind): not a declared action, no token —
                # the phazer's move token is the declared action. (|replace| is
                # unreachable in gen 3; treated the same way for safety.)
                close_window()
                continue
            own, opp, current_weather = context_trio()
            window = _Window(
                event_index=index,
                turn=turn_number,
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
            open_window(window)
            continue

        if event_type == "cant" and len(parts) >= 4:
            side = _slot_from_ident(parts[2]) or ""
            if side not in {"p1", "p2"}:
                continue
            stay = occupant.get(side)
            species = stay.species if stay is not None else _species_from_ident(parts[2])
            # A prevented turn interrupts a two-turn charge; the next use is a fresh choice.
            pending_charge[side] = None
            own, opp, current_weather = context_trio()
            open_window(
                _Window(
                    event_index=index,
                    turn=turn_number,
                    side=side,
                    species=species,
                    kind=TOKEN_KIND_CANT,
                    action=_side_condition_identifier(parts[3]),
                    defender_side=None,
                    transformed=transformed[side],
                    own_spikes_layers=own,
                    opp_spikes_layers=opp,
                    weather=current_weather,
                )
            )
            continue

        # --- Non-action lines: window accumulation, then global state updates. ---
        target = _slot_from_ident(parts[2]) if len(parts) > 2 else None
        from_payload = _from_tag_payload(raw_line)

        if event_type == "-transform" and target in {"p1", "p2"}:
            transformed[target] = True

        if event_type == "-damage" and target in {"p1", "p2"} and len(parts) >= 4:
            condition = _condition_features(parts[3])
            new_fraction = condition.hp_fraction
            if current is not None and target == current.defender_side:
                if from_payload is None:
                    if current.kind == TOKEN_KIND_MOVE and new_fraction is not None:
                        previous_fraction = hp_fraction.get(target, 1.0)
                        delta = previous_fraction - new_fraction
                        if delta > 0:
                            current.damage_fraction += delta
                        current.defender_hit_by_move = True
                else:
                    # Chip landed on the defender after the move's own damage: a
                    # subsequent faint is the chip's, not the move's.
                    current.defender_hit_by_move = False
            if new_fraction is not None:
                hp_fraction[target] = new_fraction

        elif event_type in {"-heal", "-sethp"} and target in {"p1", "p2"} and len(parts) >= 4:
            condition = _condition_features(parts[3])
            if condition.hp_fraction is not None:
                hp_fraction[target] = condition.hp_fraction
            # [silent] heals (Leech Seed transfers, Rest) are excluded from attribution
            # entirely — same hygiene class as [from]-tagged chip damage.
            is_silent = "[silent]" in raw_line
            if current is not None and event_type == "-heal" and target == current.side and not is_silent:
                if from_payload is not None and _normalize_identifier(from_payload) == "drain":
                    current.upgrade_side_effect(SIDE_EFFECT_DRAIN)
                elif from_payload is None:
                    current.upgrade_side_effect(SIDE_EFFECT_HEAL)

        elif event_type == "faint" and target in {"p1", "p2"}:
            hp_fraction[target] = 0.0
            pending_faint_replacement[target] = True
            if current is not None and target == current.defender_side and current.defender_hit_by_move:
                current.ko = True

        elif event_type == "-status" and target in {"p1", "p2"}:
            if current is not None and target != current.side:
                current.upgrade_side_effect(SIDE_EFFECT_STATUS_INFLICTED)

        elif event_type in {"-boost", "-unboost", "-setboost"}:
            # Tagged boosts (item procs like Salac Berry) are not the action's own
            # side effect — same attribution hygiene as [from]-tagged chip damage.
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
                    weather_reveals.append((setter, identifier, from_ability))

        elif event_type == "-prepare":
            if current is not None:
                current.upgrade_side_effect(SIDE_EFFECT_CHARGING)
            prepare_side = _slot_from_ident(parts[2]) if len(parts) > 2 else None
            if prepare_side in {"p1", "p2"} and len(parts) >= 4:
                pending_charge[prepare_side] = _normalize_identifier(parts[3])

        elif event_type == "-crit":
            if current is not None and target == current.defender_side:
                current.crit = True

        elif event_type == "-miss":
            if current is not None and _slot_from_ident(parts[2]) == current.side:
                current.miss = True

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

        _update_side_conditions(parts, side_condition_counts)
        weather = _update_weather(parts, weather)

    close_window()

    _flag_pursuit_intercepts(windows, raw_lines)

    tokens = tuple(
        TransitionToken(
            turn=window.turn,
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
            own_spikes_layers=window.own_spikes_layers,
            opp_spikes_layers=window.opp_spikes_layers,
            weather=window.weather,
            defender_species=window.defender_species if window.kind == TOKEN_KIND_MOVE else None,
        )
        for window in windows
    )
    return _FoldResult(
        tokens=tokens,
        windows=tuple(windows),
        weather_reveals=tuple(weather_reveals),
        mon_counters=mon_counters,
    )


# Line types that end the backward scan for the interception marker: the marker is
# emitted by the engine immediately before the Pursuit move executes, so any earlier
# action line means there was no interception of THIS Pursuit.
_PURSUIT_SCAN_BOUNDARY = frozenset({"move", "switch", "drag", "replace", "cant", "turn", "upkeep"})


def _flag_pursuit_intercepts(windows: list[_Window], raw_lines: Sequence[str]) -> None:
    """Pursuit-intercept detection via the engine's explicit interception marker.

    The engine emits ``|-activate|<target>|move: Pursuit`` (``onBeforeSwitchOut`` in the
    vendored ``data/moves.ts``) on every interception, immediately before the Pursuit
    move line — including interceptions through a Substitute (no untagged ``-damage``)
    and of Baton Pass switch-outs. A Pursuit move token is an intercept iff that marker
    for its defender directly precedes it (scanning back past non-action lines only).
    Plain Pursuit KOs never emit the marker, so faint-replacements — which the engine
    places BEFORE ``|upkeep|`` — can never false-positive; ordering heuristics are
    deliberately not used.
    """
    for window in windows:
        if window.kind != TOKEN_KIND_MOVE or window.action != "pursuit":
            continue
        defender = window.defender_side
        if defender is None:
            continue
        for raw_line in reversed(raw_lines[: window.event_index]):
            parts = raw_line.split("|")
            event_type = parts[1] if len(parts) > 1 else ""
            if event_type in _PURSUIT_SCAN_BOUNDARY:
                break
            if (
                event_type == "-activate"
                and len(parts) >= 4
                and _slot_from_ident(parts[2]) == defender
                and _side_condition_identifier(parts[3]) == "pursuit"
            ):
                window.pursuit_intercept = True
                break


def _from_tag_payload(raw_line: str) -> Optional[str]:
    """Raw ``[from]`` payload text (e.g. ``ability: Volt Absorb``, ``psn``) or None."""
    match = _FROM_TAG_RE.search(raw_line)
    if match is None:
        return None
    payload = match.group(1).strip()
    return payload or None


def _of_tag_slot(raw_line: str) -> Optional[str]:
    match = _OF_TAG_RE.search(raw_line)
    return match.group(1) if match is not None else None


def _is_absorb_signature(from_payload: Optional[str]) -> bool:
    if from_payload is None or not from_payload.lower().startswith("ability:"):
        return False
    ability = _normalize_identifier(from_payload.split(":", 1)[1])
    return ability in _ABSORB_ABILITIES


def _is_absorb_start(event_type: str, parts: Sequence[str]) -> bool:
    """``|-start|pXa: Mon|ability: Flash Fire`` — the boost-state form of an absorb."""
    if event_type != "-start" or len(parts) < 4:
        return False
    return _side_condition_identifier(parts[3]) in _ABSORB_ABILITIES


def _validated_slot(slot: str) -> str:
    if slot not in {"p1", "p2"}:
        raise ValueError(f"perspective_slot must be 'p1' or 'p2', got {slot!r}.")
    return slot


def _other_side(slot: str) -> str:
    return "p2" if slot == "p1" else "p1"
