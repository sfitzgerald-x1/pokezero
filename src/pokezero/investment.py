"""Defender-side investment inference (v2.1 batch 2 — the CB sibling, inverted).

Infers, from OUR strikes against an opponent mon, whether that defender's
variant-conditioned stats are pinned (``docs/observation_compression_design.md``
corrections item 1: HP and Atk are variant-conditioned; Def/SpA/SpD/Spe carry only the
Hidden-Power IV-override wobble). Two conclusion axes per opponent mon:

- **HP investment** — the defender's max HP (the damage-fraction denominator): full
  85-EV vs generator-trimmed (Sub+Flail/Reversal, Sub+pinch-berry, Belly Drum trim
  loops, plus the HP-IV 30 Hidden-Power overrides). Family separations are 1-4 HP
  points, far below base Tier 2's CB margins — the evidence is exact-damage LATTICE
  MEMBERSHIP, not magnitude exceedance: our attacker stats are exactly known (own
  request), so each candidate defender variant admits exactly 16 legal per-hit damage
  values; ``observed_fraction * candidate_max_hp`` must land ON one of them.
- **Defensive pinning** — Def (physical strikes) / SpD (special strikes) narrowed
  within the variant family via the same lattice check (IV-31 vs Hidden-Power IV-30
  spreads differ by one stored point on some species/levels, which shifts the
  truncation cascade).

Fraction-exactness precondition: the local BattleStream env consumes the OMNISCIENT
protocol channel (``local_showdown._apply_event``; channel -1 carries the secret
``h/maxhp`` form for BOTH sides — ``sim/battle-stream.ts``), so ``damage_fraction`` on
our strikes is an exact rational with the defender's true max HP as denominator. That
is the production training-collection view, and the gate harness measures exactly it.
A consumer feeding percent-quantized PLAYER views (online websocket play) must set
``InvestmentConfig.fraction_granularity`` to the protocol quantum (0.01): the pin rules
stay sound but the ±1%-of-max-HP tolerance window then covers every family (their
separations are smaller), so conclusions correctly never fire — this module cannot
manufacture set knowledge a real player view does not contain.

Evidence discipline (same as base Tier 2's CB bit):

- conservative conditioning — the shared :func:`pokezero.tier2._strike_damage` core
  (NOT a fork): crit-conditioned rolls on ``|-crit|`` strikes, screens/stages/burn/
  weather/Facade/pinch conditioning, and the structural disqualifiers (transform,
  typechange, Trick/Knock Off item mutation, Trace-class ability override, truncated
  outcomes, multi-hit, KO-clipped observations);
- a strike pins a family only when it is consistent with EXACTLY ONE value, every
  other candidate value is rejected by a margin (``rejection_margin_hp``) beyond the
  tolerance window, and at least two candidate values existed BEFORE the strike
  (belief-elimination alone never fires the bit — mirrors ``cb-pinned-by-elimination``);
- an observation consistent with NO candidate variant is off-model and yields no
  evidence in either direction (the precision guard);
- monotone accrual with a TWO-STRIKE rule: the same value must pin on two independent
  clean strikes; any conflicting pin or margin-rejection of a previously pinned value
  before conclusion permanently blocks that axis for the mon.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

from .belief import PublicBattleBeliefEngine, belief_key
from .dex import ShowdownDex, normalize_id
from .gen3_damage import RandbatsSpread, gen3_stat, randbats_spread_details
from .showdown import ShowdownReplayState
from .tier2 import (
    CandidateVariant,
    OwnMon,
    StrikeContext,
    StrikeParticipant,
    _IncrementalContextFold,
    _other,
    _species_key,
    _strike_contexts,
    _strike_damage,
    variant_has_physical_attack,
)
from .transitions import (
    DAMAGE_OUTCOME_ABSORBED,
    DAMAGE_OUTCOME_BLOCKED,
    DAMAGE_OUTCOME_IMMUNE,
    TOKEN_KIND_MOVE,
    TransitionToken,
    _fold_replay,
)

# Fixed exact-damage moves that make legal HP-denominator probes: the dealt damage is
# a public constant (Seismic Toss / Night Shade deal the ATTACKER's level — ours,
# exactly known), so the "lattice" is a single value. The remaining fixed-damage
# callbacks depend on hidden or path-dependent state and are excluded outright.
FIXED_EXACT_DAMAGE_MOVES = frozenset({"seismictoss", "nightshade", "dragonrage", "sonicboom"})
_FIXED_CONSTANT_DAMAGE = {"dragonrage": 40, "sonicboom": 20}
UNMODELED_DAMAGE_MOVES = frozenset({"counter", "mirrorcoat", "psywave", "superfang", "endeavor"})

_TRUNCATION_OUTCOMES = frozenset({"hit-sub", "broke-sub", "endured"})
_NEGATION_OUTCOMES = frozenset({DAMAGE_OUTCOME_BLOCKED, DAMAGE_OUTCOME_IMMUNE, DAMAGE_OUTCOME_ABSORBED})

HP_CLASS_FULL = "full"
HP_CLASS_TRIMMED = "trimmed"
DEFENSE_CLASS_FULL = "full"
DEFENSE_CLASS_REDUCED = "reduced"

# Observation-column codes (single reserved scalar, NUMERIC_TT_INVESTMENT_BIT): the
# HP conclusion owns the unit magnitudes, defensive pins the half magnitudes; an HP
# conclusion takes precedence when both exist. Zero = no damage-evidence conclusion.
INVESTMENT_CODE_HP_FULL = 1.0
INVESTMENT_CODE_HP_TRIMMED = -1.0
INVESTMENT_CODE_DEFENSE_FULL = 0.5
INVESTMENT_CODE_DEFENSE_REDUCED = -0.5


@dataclass(frozen=True)
class InvestmentConfig:
    # Lattice-membership tolerance in HP points: strictly a float-noise guard on exact
    # rational fractions (parse + delta accumulation error is <1e-9 HP at gen3 scales).
    lattice_tolerance_hp: float = 1e-6
    # Granularity of the observed HP fractions. 0.0 = exact secret-channel fractions
    # (the local training env and the gate harness). Percent-quantized player views
    # must use 0.01, widening the tolerance window by 0.01 * candidate max HP.
    fraction_granularity: float = 0.0
    # A candidate value is REJECTED (evidence against) only when its best roll misses
    # by at least this many HP points beyond the tolerance window; the in-between band
    # is treated as ambiguous and yields no pin from the strike.
    rejection_margin_hp: float = 0.25
    # Two-strike rule (mirrors Tier2Config.required_cb_strikes): one lattice edge case
    # cannot conclude an axis.
    required_pin_strikes: int = 2
    # Passed through to the shared damage core (our own pinch abilities / Flail
    # breakpoints; own-side fractions are exact but the band costs nothing).
    pinch_ambiguity_band: float = 0.04


@dataclass(frozen=True)
class DefenderStrikeAssessment:
    """One assessed strike of ours against an opponent defender."""

    token_index: int
    turn: int
    defender_key: str  # belief key, e.g. "p2:snorlax"
    move_id: str
    category: Optional[str] = None  # "Physical" | "Special"
    defense_stat_key: Optional[str] = None  # "def" | "spd"
    observed_fraction: float = 0.0
    candidate_hp_values: tuple[int, ...] = ()
    consistent_hp_values: tuple[int, ...] = ()
    candidate_defense_values: tuple[int, ...] = ()
    consistent_defense_values: tuple[int, ...] = ()
    hp_pin: Optional[int] = None
    hp_pin_class: Optional[str] = None  # "full" | "trimmed" | None (mixed EV classes)
    defense_pin: Optional[int] = None
    off_model: bool = False  # observation consistent with NO candidate variant
    margin_ambiguous: bool = False  # some candidate sat between tolerance and rejection
    disqualifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class InvestmentConclusion:
    """Monotone per-defender conclusions (as of the end of the assessed prefix)."""

    defender_key: str
    hp_value: Optional[int] = None
    hp_class: Optional[str] = None  # "full" | "trimmed" | None (value pinned, class mixed)
    hp_pin_turns: tuple[int, ...] = ()
    hp_blocked: bool = False  # conflicting HP evidence observed before conclusion
    defense_values: Mapping[str, int] = field(default_factory=dict)  # stat key -> value
    defense_classes: Mapping[str, str] = field(default_factory=dict)  # stat key -> class
    defense_pin_turns: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    defense_blocked: tuple[str, ...] = ()


@dataclass(frozen=True)
class InvestmentInference:
    perspective_slot: str
    opponent_slot: str
    strikes: tuple[DefenderStrikeAssessment, ...]
    conclusions: Mapping[str, InvestmentConclusion]
    # token index -> observation-column code, as of that strike (monotone within the
    # battle: set on our assessed move tokens from the concluding strike onward).
    token_codes: Mapping[int, float]


def conclusion_column_code(conclusion: InvestmentConclusion) -> float:
    """Project a conclusion onto the single reserved observation column.

    HP class conclusions dominate (unit magnitude, sign = full/trimmed); otherwise a
    defensive pin encodes at half magnitude (sign = full/reduced; a reduced pin on
    EITHER defense stat wins the sign — both signal a Hidden-Power IV override).
    Value-only pins (class mixed across surviving spreads) encode zero: the column
    carries the investment CLASS, not the integer.
    """
    if conclusion.hp_value is not None:
        if conclusion.hp_class == HP_CLASS_FULL:
            return INVESTMENT_CODE_HP_FULL
        if conclusion.hp_class == HP_CLASS_TRIMMED:
            return INVESTMENT_CODE_HP_TRIMMED
    if conclusion.defense_values:
        classes = set(conclusion.defense_classes.values())
        if DEFENSE_CLASS_REDUCED in classes:
            return INVESTMENT_CODE_DEFENSE_REDUCED
        if classes == {DEFENSE_CLASS_FULL}:
            return INVESTMENT_CODE_DEFENSE_FULL
    return 0.0


# --- Per-defender monotone ledger. ---


@dataclass
class _AxisLedger:
    pins: list[tuple[int, int, Optional[str]]] = field(default_factory=list)  # (turn, value, class)
    blocked: bool = False
    concluded_value: Optional[int] = None
    concluded_class: Optional[str] = None
    concluded_turns: tuple[int, ...] = ()

    def observe(
        self,
        *,
        pin_value: Optional[int],
        pin_class: Optional[str],
        rejected_values: Sequence[int],
        turn: int,
        required: int,
    ) -> None:
        if self.concluded_value is not None:
            return  # frozen: conclusions are monotone within a battle
        if self.blocked:
            return
        # Margin-rejection of a previously pinned value is conflicting evidence even
        # when the strike itself pins nothing (its consistent set was plural).
        if any(value == pinned for _, pinned, _ in self.pins for value in rejected_values):
            self.blocked = True
            return
        if pin_value is None:
            return
        if any(pinned != pin_value for _, pinned, _ in self.pins):
            self.blocked = True
            return
        self.pins.append((turn, pin_value, pin_class))
        if len(self.pins) >= required:
            classes = {pin_class for _, _, pin_class in self.pins}
            self.concluded_value = pin_value
            self.concluded_class = classes.pop() if len(classes) == 1 else None
            self.concluded_turns = tuple(turn for turn, _, _ in self.pins)


@dataclass
class _DefenderLedger:
    hp: _AxisLedger = field(default_factory=_AxisLedger)
    defense: dict[str, _AxisLedger] = field(default_factory=dict)

    def defense_axis(self, stat_key: str) -> _AxisLedger:
        return self.defense.setdefault(stat_key, _AxisLedger())

    def conclusion(self, defender_key: str) -> InvestmentConclusion:
        defense_values = {
            stat: axis.concluded_value
            for stat, axis in self.defense.items()
            if axis.concluded_value is not None
        }
        defense_classes = {
            stat: axis.concluded_class
            for stat, axis in self.defense.items()
            if axis.concluded_value is not None and axis.concluded_class is not None
        }
        defense_turns = {
            stat: axis.concluded_turns
            for stat, axis in self.defense.items()
            if axis.concluded_value is not None
        }
        return InvestmentConclusion(
            defender_key=defender_key,
            hp_value=self.hp.concluded_value,
            hp_class=self.hp.concluded_class,
            hp_pin_turns=self.hp.concluded_turns,
            hp_blocked=self.hp.blocked,
            defense_values=defense_values,
            defense_classes=defense_classes,
            defense_pin_turns=defense_turns,
            defense_blocked=tuple(sorted(stat for stat, axis in self.defense.items() if axis.blocked)),
        )


# --- Candidate spread evaluation. ---


def _variant_spread(
    variant: CandidateVariant,
    species_key: str,
    dex: ShowdownDex,
    cache: dict[tuple, RandbatsSpread],
) -> Optional[RandbatsSpread]:
    info = dex.species_info(species_key)
    if info is None or not info.base_stats:
        return None
    cache_key = (species_key, variant.level, variant.moves, variant.item)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    spread = randbats_spread_details(
        info.base_stats,
        level=variant.level,
        moves=variant.moves,
        item=variant.item,
        has_physical_attack=variant_has_physical_attack(variant.moves, dex),
    )
    cache[cache_key] = spread
    return spread


@dataclass(frozen=True)
class _EvaluatedDefender:
    variant: CandidateVariant
    spread: RandbatsSpread
    rolls: tuple[int, ...]
    ambiguous: bool


def _hp_class_of(spread: RandbatsSpread) -> str:
    return HP_CLASS_FULL if int(spread.evs.get("hp", 85)) == 85 else HP_CLASS_TRIMMED


def _classify_values(values: set[str]) -> Optional[str]:
    return values.pop() if len(values) == 1 else None


# --- Strike assessment. ---


def _assess_defender_strike(
    *,
    token: TransitionToken,
    token_index: int,
    context: StrikeContext,
    engine: PublicBattleBeliefEngine,
    opponent: str,
    own_by_species: Mapping[str, OwnMon],
    dex: ShowdownDex,
    config: InvestmentConfig,
    spread_cache: dict[tuple, RandbatsSpread],
) -> Optional[DefenderStrikeAssessment]:
    move_id = normalize_id(token.action)
    defender_species_key = _species_key(context.defender_species)
    defender_key = belief_key(opponent, context.defender_species)
    base = DefenderStrikeAssessment(
        token_index=token_index,
        turn=token.turn,
        defender_key=defender_key,
        move_id=move_id,
        observed_fraction=token.damage_fraction,
    )

    if move_id == "struggle":
        return replace(base, disqualifiers=("struggle",))
    if move_id in UNMODELED_DAMAGE_MOVES:
        return replace(base, disqualifiers=("unmodeled-damage",))
    fixed_move = move_id in FIXED_EXACT_DAMAGE_MOVES
    info = dex.move_info(move_id)
    if not fixed_move and not move_id.startswith("hiddenpower"):
        if info is None:
            return replace(base, disqualifiers=("unknown-move",))
        if info.gen3_category not in {"Physical", "Special"}:
            return None  # status move: no damage channel at all
        if info.base_power <= 0 and move_id not in {"return", "flail", "reversal", "pursuit"}:
            return replace(base, disqualifiers=("no-base-power",))

    # No damage event: misses and negation outcomes carry no lattice evidence.
    if token.miss or token.damage_outcome in _NEGATION_OUTCOMES or token.damage_fraction <= 0:
        return replace(base, disqualifiers=("no-damage-event",))
    if token.called:
        return replace(base, disqualifiers=("called",))
    if token.ko:
        # The observation is clipped at the defender's remaining HP: lattice
        # membership is undefined (mirrors the CB non-KO requirement).
        return replace(base, disqualifiers=("ko-clipped",))
    if token.damage_outcome in _TRUNCATION_OUTCOMES:
        return replace(base, disqualifiers=("truncated-outcome",))
    if max(1, token.n_hits) != 1:
        return replace(base, disqualifiers=("multi-hit",))
    if token.transformed or context.attacker_transformed:
        return replace(base, disqualifiers=("transformed-attacker",))
    if context.defender_transformed:
        return replace(base, disqualifiers=("transformed-defender",))
    if context.attacker_type_changed or context.defender_type_changed:
        return replace(base, disqualifiers=("type-changed",))
    if context.attacker_item_mutated or context.defender_item_mutated:
        return replace(base, disqualifiers=("item-mutated",))
    if context.attacker_ability_overridden or context.defender_ability_overridden:
        return replace(base, disqualifiers=("ability-overridden",))

    own = own_by_species.get(_species_key(context.attacker_species))
    if own is None or not own.stats:
        return replace(base, disqualifiers=("unknown-attacker",))
    defender_info = dex.species_info(defender_species_key)
    if defender_info is None or not defender_info.base_stats:
        return replace(base, disqualifiers=("unknown-defender",))

    belief = next(
        (
            mon
            for mon in engine.snapshot().side(opponent)
            if _species_key(mon.species) == defender_species_key
        ),
        None,
    )
    if belief is None:
        return replace(base, disqualifiers=("no-belief",))
    variants = [CandidateVariant.from_mapping(payload) for payload in belief.candidate_variants]
    if not variants:
        return replace(base, disqualifiers=("no-candidates",))

    # Category / defense-stat resolution. Fixed exact-damage moves probe only the HP
    # denominator (the constant tests no defense stat), so their defense axis is None.
    fixed_value: Optional[int] = None
    if fixed_move:
        fixed_value = _FIXED_CONSTANT_DAMAGE.get(move_id, own.level)
        category = info.gen3_category if info is not None else None
        defense_stat_key = None
    elif move_id.startswith("hiddenpower"):
        own_hp_move = next((move for move in own.moves if move.startswith("hiddenpower")), None)
        own_hp_info = dex.move_info(own_hp_move) if own_hp_move else None
        category = own_hp_info.gen3_category if own_hp_info is not None else None
        if category not in {"Physical", "Special"}:
            return replace(base, disqualifiers=("unknown-category",))
        defense_stat_key = "def" if category == "Physical" else "spd"
    else:
        category = info.gen3_category if info is not None else None
        if category not in {"Physical", "Special"}:
            return replace(base, disqualifiers=("unknown-category",))
        defense_stat_key = "def" if category == "Physical" else "spd"
    base = replace(base, category=category, defense_stat_key=defense_stat_key)

    attacker = StrikeParticipant(
        species_key=_species_key(own.species),
        level=own.level,
        stats=own.stats,
        ability=normalize_id(own.ability or ""),
        item=normalize_id(own.item or ""),
        moves=own.moves,
    )

    evaluated: list[_EvaluatedDefender] = []
    ambiguous = False
    for variant in variants:
        spread = _variant_spread(variant, defender_species_key, dex, spread_cache)
        if spread is None:
            # An unevaluable candidate could be the true one: pinning among the rest
            # would be unsound, so the whole strike is discarded.
            return replace(base, disqualifiers=("no-computable-variant",))
        if fixed_move:
            rolls: tuple[int, ...] = (int(fixed_value),)
            variant_ambiguous = False
        else:
            damage = _strike_damage(
                attacker=attacker,
                defender=StrikeParticipant(
                    species_key=defender_species_key,
                    level=variant.level,
                    stats=spread.stats,
                    ability=variant.ability,
                    item=variant.item,
                    moves=variant.moves,
                ),
                defender_types=defender_info.types,
                observed_move=move_id,
                token=token,
                context=context,
                dex=dex,
                require_move_in_set=bool(own.moves),
                pinch_ambiguity_band=config.pinch_ambiguity_band,
                split_choice_band_baseline=False,
            )
            if damage is None or not damage.rolls:
                return replace(base, disqualifiers=("no-computable-strike",))
            rolls = damage.rolls
            variant_ambiguous = damage.ambiguous
        ambiguous = ambiguous or variant_ambiguous
        evaluated.append(_EvaluatedDefender(variant=variant, spread=spread, rolls=rolls, ambiguous=variant_ambiguous))

    if ambiguous:
        return replace(base, disqualifiers=("conditioning-ambiguity",))

    # Lattice-membership classification per candidate variant.
    consistent: list[_EvaluatedDefender] = []
    rejected: list[_EvaluatedDefender] = []
    margin_ambiguous = False
    for entry in evaluated:
        max_hp = int(entry.spread.stats.get("hp", 0))
        if max_hp <= 0:
            return replace(base, disqualifiers=("no-computable-variant",))
        observed_hp = token.damage_fraction * max_hp
        distance = min(abs(observed_hp - roll) for roll in entry.rolls)
        tolerance = config.lattice_tolerance_hp + config.fraction_granularity * max_hp
        if distance <= tolerance:
            consistent.append(entry)
        elif distance >= tolerance + config.rejection_margin_hp:
            rejected.append(entry)
        else:
            margin_ambiguous = True

    candidate_hp_values = tuple(sorted({int(entry.spread.stats["hp"]) for entry in evaluated}))
    consistent_hp_values = tuple(sorted({int(entry.spread.stats["hp"]) for entry in consistent}))
    if defense_stat_key is not None:
        candidate_defense_values = tuple(
            sorted({int(entry.spread.stats[defense_stat_key]) for entry in evaluated})
        )
        consistent_defense_values = tuple(
            sorted({int(entry.spread.stats[defense_stat_key]) for entry in consistent})
        )
    else:
        candidate_defense_values = ()
        consistent_defense_values = ()
    base = replace(
        base,
        candidate_hp_values=candidate_hp_values,
        candidate_defense_values=candidate_defense_values,
        consistent_hp_values=consistent_hp_values,
        consistent_defense_values=consistent_defense_values,
        margin_ambiguous=margin_ambiguous,
    )

    if not consistent:
        # Off-model: something unmodeled happened. Never evidence (precision guard).
        return replace(base, off_model=True, disqualifiers=("no-consistent-variant",))
    if margin_ambiguous:
        return replace(base, disqualifiers=("margin-ambiguity",))

    hp_pin: Optional[int] = None
    hp_pin_class: Optional[str] = None
    if len(consistent_hp_values) == 1 and len(candidate_hp_values) >= 2:
        hp_pin = consistent_hp_values[0]
        hp_pin_class = _classify_values({_hp_class_of(entry.spread) for entry in consistent})
    defense_pin: Optional[int] = None
    if len(consistent_defense_values) == 1 and len(candidate_defense_values) >= 2:
        defense_pin = consistent_defense_values[0]
    return replace(base, hp_pin=hp_pin, hp_pin_class=hp_pin_class, defense_pin=defense_pin)


def _defense_class(
    *, dex: ShowdownDex, species_key: str, level: int, stat_key: str, value: int
) -> Optional[str]:
    """Full (85 EV / 31 IV baseline) vs reduced (Hidden-Power IV override) defense."""
    info = dex.species_info(species_key)
    if info is None or not info.base_stats:
        return None
    baseline = gen3_stat(int(info.base_stats.get(stat_key, 0)), 31, 85, level)
    if value == baseline:
        return DEFENSE_CLASS_FULL
    if value < baseline:
        return DEFENSE_CLASS_REDUCED
    return None


@dataclass
class _InferenceState:
    """Shared accrual state between the batch entry point and the live tracker."""

    ledgers: dict[str, _DefenderLedger] = field(default_factory=dict)
    strikes: list[DefenderStrikeAssessment] = field(default_factory=list)
    token_codes: dict[int, float] = field(default_factory=dict)

    def ledger(self, defender_key: str) -> _DefenderLedger:
        return self.ledgers.setdefault(defender_key, _DefenderLedger())

    def apply(
        self,
        assessment: DefenderStrikeAssessment,
        *,
        dex: ShowdownDex,
        defender_level: Optional[int],
        config: InvestmentConfig,
    ) -> None:
        self.strikes.append(assessment)
        if not assessment.disqualifiers and not assessment.off_model:
            ledger = self.ledger(assessment.defender_key)
            rejected_hp = tuple(
                value
                for value in assessment.candidate_hp_values
                if value not in assessment.consistent_hp_values
            )
            ledger.hp.observe(
                pin_value=assessment.hp_pin,
                pin_class=assessment.hp_pin_class,
                rejected_values=rejected_hp,
                turn=assessment.turn,
                required=config.required_pin_strikes,
            )
            if assessment.defense_stat_key is not None:
                rejected_defense = tuple(
                    value
                    for value in assessment.candidate_defense_values
                    if value not in assessment.consistent_defense_values
                )
                axis = ledger.defense_axis(assessment.defense_stat_key)
                defense_class: Optional[str] = None
                if assessment.defense_pin is not None and defender_level is not None:
                    defense_class = _defense_class(
                        dex=dex,
                        species_key=assessment.defender_key.split(":", 1)[1],
                        level=defender_level,
                        stat_key=assessment.defense_stat_key,
                        value=assessment.defense_pin,
                    )
                axis.observe(
                    pin_value=assessment.defense_pin,
                    pin_class=defense_class,
                    rejected_values=rejected_defense,
                    turn=assessment.turn,
                    required=config.required_pin_strikes,
                )
        # As-of-strike code (mirrors the CB bit): EVERY assessed strike token of a
        # concluded defender carries the code, disqualified strikes included.
        existing = self.ledgers.get(assessment.defender_key)
        if existing is not None:
            code = conclusion_column_code(existing.conclusion(assessment.defender_key))
            if code != 0.0:
                self.token_codes[assessment.token_index] = code

    def conclusions(self) -> dict[str, InvestmentConclusion]:
        return {key: ledger.conclusion(key) for key, ledger in self.ledgers.items()}


def infer_investment(
    replay: ShowdownReplayState,
    *,
    perspective_slot: str,
    own_team: Sequence[OwnMon],
    dex: ShowdownDex,
    set_source: Any = None,
    config: InvestmentConfig | None = None,
    format_id: str = "gen3randombattle",
) -> InvestmentInference:
    """Run the defender-side investment inference for one perspective over a replay.

    Mirrors :func:`pokezero.tier2.infer_tier2`'s evaluation timing: each of OUR strikes
    is assessed against the belief state after ingesting the public events up to the
    next declared action (the strike's own protocol chunk included).
    """
    config = config or InvestmentConfig()
    if perspective_slot not in {"p1", "p2"}:
        raise ValueError(f"perspective_slot must be 'p1' or 'p2', got {perspective_slot!r}.")
    opponent = _other(perspective_slot)

    fold = _fold_replay(replay, perspective_slot=perspective_slot)
    raw_lines = tuple(event.raw_line for event in replay.public_events)
    contexts = _strike_contexts(raw_lines)
    own_by_species = {_species_key(mon.species): mon for mon in own_team}

    engine = PublicBattleBeliefEngine(format_id=format_id, set_source=set_source)
    events = replay.public_events
    fed = 0
    spread_cache: dict[tuple, RandbatsSpread] = {}
    state = _InferenceState()
    defender_levels: dict[str, int] = {}

    windows = list(fold.windows)
    for index, (token, window) in enumerate(zip(fold.tokens, windows)):
        feed_until = windows[index + 1].event_index if index + 1 < len(windows) else len(events)
        while fed < feed_until:
            engine.ingest_event(events[fed])
            fed += 1
        if token.kind != TOKEN_KIND_MOVE or token.actor_slot != perspective_slot:
            continue
        context = contexts.get(window.event_index)
        if context is None or not context.defender_species:
            continue
        assessment = _assess_defender_strike(
            token=token,
            token_index=index,
            context=context,
            engine=engine,
            opponent=opponent,
            own_by_species=own_by_species,
            dex=dex,
            config=config,
            spread_cache=spread_cache,
        )
        if assessment is None:
            continue
        defender_level = _defender_level(
            engine, opponent, context.defender_species, defender_levels
        )
        state.apply(assessment, dex=dex, defender_level=defender_level, config=config)

    while fed < len(events):
        engine.ingest_event(events[fed])
        fed += 1

    return InvestmentInference(
        perspective_slot=perspective_slot,
        opponent_slot=opponent,
        strikes=tuple(state.strikes),
        conclusions=state.conclusions(),
        token_codes=dict(state.token_codes),
    )


def _defender_level(
    engine: PublicBattleBeliefEngine,
    opponent: str,
    defender_species: str,
    cache: dict[str, int],
) -> Optional[int]:
    key = _species_key(defender_species)
    cached = cache.get(key)
    if cached is not None:
        return cached
    belief = next(
        (mon for mon in engine.snapshot().side(opponent) if _species_key(mon.species) == key),
        None,
    )
    if belief is None:
        return None
    for payload in belief.candidate_variants:
        level = payload.get("level")
        if isinstance(level, int) and level > 0:
            cache[key] = level
            return level
    return None


class InvestmentLiveTracker:
    """Incremental live consumer: accrues defender-side investment conclusions.

    The per-battle sibling of :class:`pokezero.tier2.Tier2LiveTracker` — shares the
    caller's belief engine, folds each protocol line once, and assesses each of OUR
    move tokens once at the first ``observe`` call that sees it. ``token_codes`` maps
    token indices to the reserved investment-column code as of that strike (monotone).
    """

    def __init__(
        self,
        *,
        perspective_slot: str,
        own_team: Sequence[OwnMon],
        dex: ShowdownDex,
        config: InvestmentConfig | None = None,
    ) -> None:
        if perspective_slot not in {"p1", "p2"}:
            raise ValueError(f"perspective_slot must be 'p1' or 'p2', got {perspective_slot!r}.")
        self._perspective = perspective_slot
        self._opponent = _other(perspective_slot)
        self._own_by_species = {_species_key(mon.species): mon for mon in own_team}
        self._dex = dex
        self._config = config or InvestmentConfig()
        self._fold = _IncrementalContextFold()
        self._spread_cache: dict[tuple, RandbatsSpread] = {}
        self._state = _InferenceState()
        self._defender_levels: dict[str, int] = {}
        self._assessed_until = 0

    @property
    def token_codes(self) -> dict[int, float]:
        return dict(self._state.token_codes)

    @property
    def conclusions(self) -> dict[str, InvestmentConclusion]:
        return self._state.conclusions()

    def observe(
        self,
        replay: ShowdownReplayState,
        tokens: Sequence[TransitionToken],
        belief_engine: PublicBattleBeliefEngine,
    ) -> dict[int, float]:
        """Assess any new own strikes; returns the current token-code mapping.

        ``tokens`` must be the transition extraction for the same ``replay``;
        ``belief_engine`` is the caller's persistent engine, already fed through the
        replay's boundary (the same live-vs-batch evidence-monotone caveat as the
        Tier-2 tracker: the live boundary may carry strictly more belief evidence).
        """
        raw_lines = tuple(event.raw_line for event in replay.public_events)
        self._fold.process(raw_lines)
        if len(self._fold.token_line_indices) != len(tokens):
            raise ValueError(
                "transition tokens do not align with the tracker's protocol fold "
                f"({len(tokens)} tokens vs {len(self._fold.token_line_indices)} action lines)."
            )
        for index in range(self._assessed_until, len(tokens)):
            token = tokens[index]
            if token.kind != TOKEN_KIND_MOVE or token.actor_slot != self._perspective:
                continue
            context = self._fold.contexts.get(self._fold.token_line_indices[index])
            if context is None or not context.defender_species:
                continue
            assessment = _assess_defender_strike(
                token=token,
                token_index=index,
                context=context,
                engine=belief_engine,
                opponent=self._opponent,
                own_by_species=self._own_by_species,
                dex=self._dex,
                config=self._config,
                spread_cache=self._spread_cache,
            )
            if assessment is None:
                continue
            defender_level = _defender_level(
                belief_engine, self._opponent, context.defender_species, self._defender_levels
            )
            self._state.apply(
                assessment, dex=self._dex, defender_level=defender_level, config=self._config
            )
        self._assessed_until = len(tokens)
        return dict(self._state.token_codes)
