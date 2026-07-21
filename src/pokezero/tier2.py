"""Tier-2 residual damage attribution + Choice Band inference (next-train readiness PR D).

Consumes PR B's transition tokens (:mod:`pokezero.transitions`) and PR A's belief ledger
(:mod:`pokezero.belief`), and populates the Tier-2 reserved fields the corrections layer
defines (items 7, 9, 10 of ``docs/observation_compression_design.md``):

- ``residual`` / ``residual_valid`` on opponent attack tokens: signed fraction of the
  defender's max HP, observed minus the expected MEDIAN roll under the
  candidate-conservative baseline (no Choice Band, no unconfirmed hidden modifier), with
  every deterministic public modifier conditioned — stat stages, screens, burn (Guts
  exempt), sun/rain Fire/Water, Solar Beam halving in rain/sand/hail, Facade's status
  doubling, Flash Fire's tracked volatile, Explosion's defense halving, crit-expected
  medians on ``|-crit|`` strikes, and Pursuit's intercept power. Multi-hit
  (Bonemerang) strikes are masked: production validity ships only populations the
  gate's calibration arm covers. Anything the candidate set leaves ambiguous
  invalidates the residual instead of guessing.
- the per-opponent-mon **Choice Band bit**: on a whitelisted fixed-power physical move,
  observed damage must exceed the maximum explainable non-CB roll (max over surviving
  candidate variants' abilities and items, max roll, all public modifiers) by a margin,
  on TWO independent clean strikes (crit / multi-hit / truncated-outcome / called /
  boosted-stage / screened strikes never count). A strike that exceeds even the best CB
  explanation is off-model and never counts either (precision guard).

Base Tier 2 populates residuals on OPPONENT attacks only (correction item 10): the
defender is the perspective player's own mon, whose exact stats/ability/item are known
from its own request — no defender-side ambiguity. Trick carriers are handled upstream:
the generator assigns Trick users Choice Band unconditionally, so the randbats candidate
variants already pin their item at the Tier-1/exact layer (``randbat._possible_items``);
this module adds no duplicate rule. All four corrections-item-9 slots are materialized
in the observation: residual (117) + validity (118) + the as-of-strike CB bit (119,
populated here under the same tier2 gate + mask), while the investment bit (120) is a
true always-zero reserve held for the H3 defender-side/investment inference.

Expected damage comes from :mod:`pokezero.gen3_damage` (see that module's docstring for
the engine-choice rationale: the vendored Showdown sim is the calibration target;
poke-engine's gen3 damage path measurably diverges from it and is not a repo dep).

Evaluation timing: each strike is assessed against the belief state after ingesting the
public events up to the next declared action (i.e. the strike's own protocol chunk,
including its own move reveal). Live consumers that re-extract from the full prefix per
observation see a superset of this evidence for older strikes; the strike-time framing
here is the conservative, reproducible one the precision gate measures.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

from .belief import PublicBattleBeliefEngine, belief_key
from .dex import ShowdownDex, normalize_id, resolve_move_base_power
from .gen3_damage import (
    Gen3DamageContext,
    gen3_damage_rolls,
    median_damage,
    randbats_spread_stats,
)
from .randbat import canonical_gen3_randbat_species_id
from .showdown import (
    ShowdownReplayState,
    _condition_features,
    _side_condition_identifier,
    _slot_from_ident,
    _species_from_details,
    _species_from_ident,
    _update_side_conditions,
)
from .transitions import (
    DAMAGE_OUTCOME_ABSORBED,
    DAMAGE_OUTCOME_BLOCKED,
    DAMAGE_OUTCOME_IMMUNE,
    DAMAGE_OUTCOME_NORMAL,
    TOKEN_KIND_MOVE,
    TransitionToken,
    _fold_replay,
)

# Fixed-damage moves (Showdown ``move.damage`` / ``damageCallback``): no base power, no
# set information; the randbats generator counts them as "damage", NOT "Physical", which
# also matters for the Atk-zeroing spread rule. Gen3-complete closed set.
FIXED_DAMAGE_MOVES = frozenset(
    {"counter", "mirrorcoat", "seismictoss", "nightshade", "dragonrage", "sonicboom", "superfang", "psywave", "endeavor"}
)
# Physical damaging moves whose dex base power is 0 (basePowerCallback). Return is
# whitelist-INCLUDED at a fixed 102 (randbats happiness is maxed — inventory decision);
# the others resolve from public HP (flail/reversal) and are CB-excluded.
RETURN_BASE_POWER = 102
_BP_CALLBACK_PHYSICAL = frozenset({"return", "frustration", "flail", "reversal"})

# CB whitelist exclusions (spec Tier-2 (b) + inventory dispositions): HP-scaled
# (flail/reversal), IV-scaled (hiddenpower*, via prefix), scenario-scaled (pursuit),
# multi-hit (bonemerang — the pool's only one), fixed-damage callbacks, Struggle.
# Weather Ball is unreachable in the gen3 randbats pool (verified against sets.json);
# it is excluded defensively against set drift (weather-dependent BP AND type).
CB_EXCLUDED_MOVES = frozenset(
    {"flail", "reversal", "frustration", "pursuit", "bonemerang", "struggle", "weatherball"}
) | FIXED_DAMAGE_MOVES

# Gen 3 type-boost items are 1.1x on the matching offensive stat (mods/gen3/items.ts).
TYPE_BOOST_ITEMS: Mapping[str, str] = {
    "silkscarf": "Normal",
    "blackbelt": "Fighting",
    "sharpbeak": "Flying",
    "poisonbarb": "Poison",
    "softsand": "Ground",
    "hardstone": "Rock",
    "silverpowder": "Bug",
    "spelltag": "Ghost",
    "metalcoat": "Steel",
    "charcoal": "Fire",
    "mysticwater": "Water",
    "seaincense": "Water",
    "miracleseed": "Grass",
    "magnet": "Electric",
    "twistedspoon": "Psychic",
    "nevermeltice": "Ice",
    "dragonfang": "Dragon",
    "blackglasses": "Dark",
}
# Gen3 Sea Incense is 1.05, not 1.1 (mods/gen3/items.ts). Dormant — the generator never
# assigns it — kept exact against set drift.
_TYPE_BOOST_FACTORS: Mapping[str, float] = {"seaincense": 1.05}

_PINCH_ABILITY_TYPES: Mapping[str, str] = {
    "overgrow": "Grass",
    "blaze": "Fire",
    "torrent": "Water",
    "swarm": "Bug",
}
_WEATHER_SUPPRESSORS = frozenset({"cloudnine", "airlock"})
_SOLAR_BEAM_WEAK_WEATHERS = frozenset({"raindance", "sandstorm", "hail"})
_GUTS_STATUSES = frozenset({"brn", "par", "psn", "tox", "slp", "frz"})
_FACADE_STATUSES = frozenset({"brn", "par", "psn", "tox"})
_TRUNCATION_OUTCOMES = frozenset({"hit-sub", "broke-sub", "endured"})
_NEGATION_OUTCOMES = frozenset({DAMAGE_OUTCOME_BLOCKED, DAMAGE_OUTCOME_IMMUNE, DAMAGE_OUTCOME_ABSORBED})


@dataclass(frozen=True)
class Tier2Config:
    # CB exceedance margin: >= protocol quantization (1% of defender max HP) plus one
    # HP point of float/rounding slack (correction item 10).
    cb_margin_fraction: float = 0.01
    cb_margin_hp: float = 1.0
    # Residual baseline validity: candidate-variant medians must agree within this
    # envelope, else the residual is ambiguous and masked.
    baseline_agreement_hp: float = 1.0
    baseline_agreement_fraction: float = 0.02
    # Two-strike rule (spec Tier-2 (c)): one calc edge case cannot flip the bit.
    required_cb_strikes: int = 2
    # |3*hp_fraction - 1| band around the pinch-ability 1/3 boundary treated as
    # ambiguous (opponent HP is percent-quantized in the player view).
    pinch_ambiguity_band: float = 0.04
    # Known-set calibration mode: apply Choice Band in the baseline (the truth source
    # pins the item, so the baseline should be the true interpretation).
    baseline_includes_choice_band: bool = False


@dataclass(frozen=True)
class OwnMon:
    """The perspective player's own mon — exact server-provided identity.

    ``moves`` (normalized ids, from the request's move list) lets the own side act as
    the ATTACKER in the shared damage core (defender-side investment inference): Hidden
    Power resolves its type against the own set exactly like variant sets resolve it.
    Empty means unknown/legacy callers — own-attacker assessment then skips the strike.
    """

    species: str
    level: int
    stats: Mapping[str, int]  # includes "hp" = max HP
    ability: Optional[str] = None
    item: Optional[str] = None
    moves: tuple[str, ...] = ()


def own_team_from_request(request: Mapping[str, Any]) -> tuple[OwnMon, ...]:
    """Extract exact own-team info from a Showdown request payload (``side.pokemon``)."""
    side = request.get("side")
    rows = side.get("pokemon") if isinstance(side, Mapping) else None
    team: list[OwnMon] = []
    if not isinstance(rows, list):
        return ()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        details = str(row.get("details") or "")
        species = _species_from_details(details) if details else ""
        if not species:
            continue
        level = 100
        for chunk in details.split(",")[1:]:
            chunk = chunk.strip()
            if chunk.startswith("L"):
                try:
                    level = int(chunk[1:])
                except ValueError:
                    pass
        stats = {
            str(stat): int(value)
            for stat, value in (row.get("stats") or {}).items()
            if isinstance(value, int)
        }
        condition = str(row.get("condition") or "")
        max_hp: Optional[int] = None
        head = condition.split()[0] if condition else ""
        if "/" in head:
            try:
                max_hp = int(head.split("/", 1)[1])
            except ValueError:
                max_hp = None
        if max_hp is not None:
            stats["hp"] = max_hp
        team.append(
            OwnMon(
                species=species,
                level=level,
                stats=stats,
                ability=str(row.get("baseAbility") or row.get("ability") or "") or None,
                item=str(row.get("item") or "") or None,
                moves=tuple(
                    canonical_move_id(str(move)) for move in (row.get("moves") or []) if str(move)
                ),
            )
        )
    return tuple(team)


# --- Public-modifier context fold (protocol-tautological state only). ---


@dataclass(frozen=True)
class StrikeContext:
    attacker_species: str
    defender_species: str
    attacker_status: Optional[str] = None
    defender_status: Optional[str] = None
    attacker_boosts: Mapping[str, int] = field(default_factory=dict)
    defender_boosts: Mapping[str, int] = field(default_factory=dict)
    attacker_hp_fraction: float = 1.0
    attacker_flash_fire: bool = False
    attacker_transformed: bool = False
    defender_transformed: bool = False
    # Color Change / Conversion-class ``|-start|...|typechange`` volatiles and Forecast
    # ``|-formechange|`` (Castform): the mon's live typing no longer matches its
    # species, so species-derived STAB / effectiveness are unsafe.
    attacker_type_changed: bool = False
    defender_type_changed: bool = False
    # Trick / Knock Off mutated the mon's held item: candidate-variant item modifiers
    # no longer describe the current holder (corrections item 15's mutation rule).
    attacker_item_mutated: bool = False
    defender_item_mutated: bool = False
    # Trace (or any acquisition-tagged |-ability|): the side's LIVE ability is no
    # longer its set/request ability — every ability-conditioned modifier is suspect,
    # on both sides of the matchup. Structural, not species-keyed.
    attacker_ability_overridden: bool = False
    defender_ability_overridden: bool = False
    defender_screens: tuple[str, ...] = ()


def _other(side: str) -> str:
    return "p2" if side == "p1" else "p1"


def _species_key(species: str) -> str:
    return canonical_gen3_randbat_species_id(species)


# Request payloads (unlike sets.json / protocol lines) suffix some move ids with their
# resolved base power: "return102", "frustration102", and (in some formats)
# "hiddenpowerice70". Canonicalize so dex lookups and set matching work either way.
_BP_SUFFIXED_PREFIXES = ("return", "frustration", "hiddenpower")


def canonical_move_id(move: str) -> str:
    move_id = normalize_id(move)
    for prefix in _BP_SUFFIXED_PREFIXES:
        if move_id.startswith(prefix):
            stripped = move_id.rstrip("0123456789")
            return stripped if stripped else move_id
    return move_id


# Moves whose gen3 ``onTryHit`` shatters the defender's Reflect/Light Screen BEFORE
# dealing damage (data/mods/gen3/moves.ts brickbreak onTryHit: ``foe.removeSideCondition``
# for both screens, "before you hit"). The screens are still present in ``side_counts``
# at the ``|move|`` line where the strike context is snapshotted, but the strike itself
# lands unscreened, so the assessed move's own context must drop them. Brick Break is the
# only gen3 move with this behaviour; later-gen shatterers (Psychic Fangs, etc.) are
# out of pool.
_SCREEN_SHATTERING_MOVES = frozenset({"brickbreak"})


class _IncrementalContextFold:
    """Incremental public-modifier context fold over the protocol lines.

    Tracks only protocol-tautological state: stat stages (with Baton Pass inheritance
    and Haze clears), screens, statuses per (side, species), Flash Fire volatiles,
    Transform, Forecast/Color Change type changes, Trick/Knock Off item mutation,
    Trace-class ability overrides, active occupants, and public HP fractions. Each
    line is processed exactly once; a :class:`StrikeContext` snapshot is stored for
    every token-emitting action line (``|move|``/``|switch|``/``|cant|`` — drags emit
    no token), BEFORE the action's own effects land, and ``token_line_indices`` maps
    transition-token positions to their protocol line indices (the emission rules
    mirror ``transitions._fold_replay``).
    """

    def __init__(self) -> None:
        self.occupant: dict[str, str] = {}
        self.status: dict[tuple[str, str], Optional[str]] = {}
        self.boosts: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
        self.pending_bp: dict[str, bool] = {"p1": False, "p2": False}
        self.flash_fire: dict[str, bool] = {"p1": False, "p2": False}
        self.transformed: dict[str, bool] = {"p1": False, "p2": False}
        self.type_changed: dict[str, bool] = {"p1": False, "p2": False}
        self.ability_overridden: dict[str, bool] = {"p1": False, "p2": False}
        # Item mutation follows the MON (persists across switches), keyed (side, species).
        self.item_mutated: set[tuple[str, str]] = set()
        self.hp: dict[str, float] = {"p1": 1.0, "p2": 1.0}
        self.side_counts: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
        self.contexts: dict[int, StrikeContext] = {}
        self.token_line_indices: list[int] = []
        self._processed = 0

    def clone(self) -> "_IncrementalContextFold":
        """Return an independent continuation state without re-folding its prefix.

        ``StrikeContext`` instances are immutable after construction, so the clone can
        safely share them. Mutable accumulator containers must be copied because a
        search branch can process a different protocol suffix from the same snapshot.
        """

        cloned = _IncrementalContextFold.__new__(_IncrementalContextFold)
        cloned.occupant = dict(self.occupant)
        cloned.status = dict(self.status)
        cloned.boosts = {side: dict(boosts) for side, boosts in self.boosts.items()}
        cloned.pending_bp = dict(self.pending_bp)
        cloned.flash_fire = dict(self.flash_fire)
        cloned.transformed = dict(self.transformed)
        cloned.type_changed = dict(self.type_changed)
        cloned.ability_overridden = dict(self.ability_overridden)
        cloned.item_mutated = set(self.item_mutated)
        cloned.hp = dict(self.hp)
        cloned.side_counts = {side: dict(counts) for side, counts in self.side_counts.items()}
        cloned.contexts = dict(self.contexts)
        cloned.token_line_indices = list(self.token_line_indices)
        cloned._processed = self._processed
        return cloned

    def process(self, raw_lines: Sequence[str]) -> None:
        """Fold any not-yet-seen suffix of ``raw_lines`` (lines are processed once)."""
        for index in range(self._processed, len(raw_lines)):
            self._process_line(index, raw_lines[index])
        self._processed = len(raw_lines)

    def _snapshot(
        self,
        attacker: str,
        defender: Optional[str],
        attacker_species: str,
        *,
        shatters_screens: bool = False,
    ) -> StrikeContext:
        defender = defender or _other(attacker)
        defender_species = self.occupant.get(defender, "")
        boosts = self.boosts
        return StrikeContext(
            attacker_species=attacker_species,
            defender_species=defender_species,
            attacker_status=self.status.get((attacker, _species_key(attacker_species))),
            defender_status=self.status.get((defender, _species_key(defender_species))),
            attacker_boosts=dict(boosts[attacker]),
            defender_boosts=dict(boosts[defender]),
            attacker_hp_fraction=self.hp.get(attacker, 1.0),
            attacker_flash_fire=self.flash_fire[attacker],
            attacker_transformed=self.transformed[attacker],
            defender_transformed=self.transformed[defender],
            attacker_type_changed=self.type_changed[attacker],
            defender_type_changed=self.type_changed[defender],
            attacker_item_mutated=(attacker, _species_key(attacker_species)) in self.item_mutated,
            defender_item_mutated=(defender, _species_key(defender_species)) in self.item_mutated,
            attacker_ability_overridden=self.ability_overridden[attacker],
            defender_ability_overridden=self.ability_overridden[defender],
            # A screen-shattering move (Brick Break) removes the defender's screens in
            # its ``onTryHit``, BEFORE dealing damage, so its own strike lands unscreened
            # even though ``side_counts`` still carries the screens at this ``|move|`` line.
            # Drop them from this strike's context to stay engine-exact; the shatter
            # ``-sideend`` lines fold in below, so the NEXT strike's snapshot sees no screens.
            defender_screens=()
            if shatters_screens
            else tuple(
                name for name in ("reflect", "lightscreen") if self.side_counts[defender].get(name)
            ),
        )

    def _record_action(self, index: int, event_type: str, parts: Sequence[str], side: str) -> None:
        species = self.occupant.get(side) or _species_from_ident(parts[2]) or ""
        shatters_screens = False
        if event_type == "move":
            defender = _slot_from_ident(parts[4]) if len(parts) > 4 else None
            shatters_screens = normalize_id(parts[3]) in _SCREEN_SHATTERING_MOVES
        elif event_type == "switch":
            species = _species_from_details(parts[3]) or _species_from_ident(parts[2]) or ""
            defender = None
        else:  # cant
            defender = None
        self.contexts[index] = self._snapshot(side, defender, species, shatters_screens=shatters_screens)
        self.token_line_indices.append(index)

    def _process_line(self, index: int, raw_line: str) -> None:
        parts = raw_line.split("|")
        event_type = parts[1] if len(parts) > 1 else ""
        side = _slot_from_ident(parts[2]) if len(parts) > 2 else None

        # Token-emitting action lines (mirrors transitions._fold_replay: drags and
        # replaces emit no token) snapshot BEFORE the action's own effects land.
        if event_type in {"move", "switch", "cant"} and side in {"p1", "p2"} and len(parts) >= 4:
            self._record_action(index, event_type, parts, side)

        occupant = self.occupant
        status = self.status
        boosts = self.boosts
        pending_bp = self.pending_bp
        flash_fire = self.flash_fire
        transformed = self.transformed
        type_changed = self.type_changed
        ability_overridden = self.ability_overridden
        item_mutated = self.item_mutated
        hp = self.hp
        side_counts = self.side_counts

        if event_type in {"switch", "drag", "replace"} and side in {"p1", "p2"} and len(parts) >= 4:
            species = _species_from_details(parts[3]) or _species_from_ident(parts[2])
            inherited = pending_bp[side] and event_type == "switch"
            if not inherited:
                boosts[side] = {}
            pending_bp[side] = False
            flash_fire[side] = False
            transformed[side] = False
            type_changed[side] = False
            ability_overridden[side] = False
            occupant[side] = species
            condition = _condition_features(parts[4] if len(parts) > 4 else None)
            hp[side] = condition.hp_fraction if condition.hp_fraction is not None else 1.0
            status[(side, _species_key(species))] = condition.status if condition.status != "none" else None
        elif event_type == "move" and side in {"p1", "p2"} and len(parts) >= 4:
            pending_bp[side] = normalize_id(parts[3]) == "batonpass"
        elif event_type == "faint" and side in {"p1", "p2"}:
            flash_fire[side] = False
            transformed[side] = False
            type_changed[side] = False
            ability_overridden[side] = False
            hp[side] = 0.0
        elif event_type == "-transform" and side in {"p1", "p2"}:
            transformed[side] = True
            # Transform copies the target's current public boost stages in Gen 3.
            target = _slot_from_ident(parts[3]) if len(parts) >= 4 else None
            if target in {"p1", "p2"}:
                boosts[side] = dict(boosts[target])
        elif event_type == "-formechange" and side in {"p1", "p2"}:
            # Forecast (Castform): the forme change carries a live typing change.
            type_changed[side] = True
        elif event_type in {"-item", "-enditem"} and side in {"p1", "p2"}:
            if "move: Trick" in raw_line or "move: Knock Off" in raw_line:
                mutated_species = occupant.get(side) or _species_from_ident(parts[2])
                item_mutated.add((side, _species_key(mutated_species)))
        elif event_type == "-ability" and side in {"p1", "p2"}:
            # A plain |-ability| line is a reveal; an ACQUISITION-tagged one replaces
            # the side's live ability until it leaves the field (Trace here; Role
            # Play / Skill Swap defensively — both unreachable in this pool). The
            # rule is structural: a traced ability breaks every ability-conditioned
            # modifier regardless of which species did the tracing.
            if "Trace" in raw_line or "move: Role Play" in raw_line or "move: Skill Swap" in raw_line:
                ability_overridden[side] = True
        elif event_type == "-status" and side in {"p1", "p2"} and len(parts) >= 4:
            species = occupant.get(side) or _species_from_ident(parts[2])
            status[(side, _species_key(species))] = parts[3].strip() or None
        elif event_type == "-curestatus" and side in {"p1", "p2"}:
            species = _species_from_ident(parts[2])
            status[(side, _species_key(species))] = None
        elif event_type == "-cureteam" and side in {"p1", "p2"}:
            for key in [key for key in status if key[0] == side]:
                status[key] = None
        elif event_type in {"-boost", "-unboost"} and side in {"p1", "p2"} and len(parts) >= 5:
            try:
                amount = int(parts[4].split()[0])
            except ValueError:
                amount = 0
            stat = parts[3].strip()
            delta = amount if event_type == "-boost" else -amount
            boosts[side][stat] = max(-6, min(6, boosts[side].get(stat, 0) + delta))
        elif event_type == "-setboost" and side in {"p1", "p2"} and len(parts) >= 5:
            try:
                boosts[side][parts[3].strip()] = max(-6, min(6, int(parts[4].split()[0])))
            except ValueError:
                pass
        elif event_type == "-clearallboost":
            boosts["p1"] = {}
            boosts["p2"] = {}
        elif event_type == "-clearboost" and side in {"p1", "p2"}:
            boosts[side] = {}
        elif event_type in {"-clearnegativeboost", "-restoreboost"} and side in {"p1", "p2"}:
            # White Herb (Deoxys lock): silently zeroes all NEGATIVE stages — without
            # this, Superpower/Psycho Boost self-drops read as stale stat stages.
            boosts[side] = {stat: stage for stat, stage in boosts[side].items() if stage > 0}
        elif event_type == "-copyboost" and side in {"p1", "p2"} and len(parts) >= 4:
            source = _slot_from_ident(parts[3])
            if source in {"p1", "p2"}:
                boosts[side] = dict(boosts[source])
        elif event_type in {"-damage", "-heal", "-sethp"} and side in {"p1", "p2"} and len(parts) >= 4:
            condition = _condition_features(parts[3])
            if condition.hp_fraction is not None:
                hp[side] = condition.hp_fraction
        elif event_type == "-start" and side in {"p1", "p2"} and len(parts) >= 4:
            identifier = _side_condition_identifier(parts[3])
            if identifier == "flashfire":
                flash_fire[side] = True
            elif identifier == "typechange":
                # Color Change (Kecleon) / Conversion-class: live typing diverges
                # from the species, so species-derived STAB/effectiveness are wrong.
                type_changed[side] = True
        _update_side_conditions(parts, side_counts)


def _strike_contexts(raw_lines: Sequence[str]) -> dict[int, StrikeContext]:
    """Batch wrapper over the incremental fold (contexts keyed by action-line index)."""
    fold = _IncrementalContextFold()
    fold.process(raw_lines)
    return fold.contexts


# --- Candidate variants and the data-derived CB whitelist. ---


@dataclass(frozen=True)
class CandidateVariant:
    moves: tuple[str, ...]  # normalized ids
    ability: str  # normalized id ("" when unknown)
    item: str  # normalized id ("" when unknown)
    level: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, default_level: int = 100) -> "CandidateVariant":
        raw_moves = payload.get("moves")
        moves = tuple(canonical_move_id(str(move)) for move in raw_moves) if isinstance(raw_moves, (list, tuple)) else ()
        level = payload.get("level")
        return cls(
            moves=moves,
            ability=normalize_id(str(payload.get("ability") or "")),
            item=normalize_id(str(payload.get("item") or "")),
            level=int(level) if isinstance(level, int) and level > 0 else default_level,
        )


def variant_has_physical_attack(moves: Sequence[str], dex: ShowdownDex) -> bool:
    """The generator's ``counter.get('Physical')`` truthiness for a move list."""
    for move in moves:
        move_id = canonical_move_id(move)
        if move_id in FIXED_DAMAGE_MOVES:
            continue
        info = dex.move_info(move_id)
        if info is None:
            continue
        if info.gen3_category != "Physical":
            continue
        if info.base_power > 0 or move_id in _BP_CALLBACK_PHYSICAL:
            return True
    return False


def _variant_stats(variant: CandidateVariant, species_key: str, dex: ShowdownDex,
                   cache: dict[tuple, Mapping[str, int]]) -> Optional[Mapping[str, int]]:
    info = dex.species_info(species_key)
    if info is None or not info.base_stats:
        return None
    cache_key = (species_key, variant.level, variant.moves, variant.item)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    stats = randbats_spread_stats(
        info.base_stats,
        level=variant.level,
        moves=variant.moves,
        item=variant.item,
        has_physical_attack=variant_has_physical_attack(variant.moves, dex),
    )
    cache[cache_key] = stats
    return stats


def build_cb_whitelist(
    universes: Mapping[str, Any],
    dex: ShowdownDex,
    *,
    atk_tolerance: float = 0.02,
) -> dict[str, frozenset[str]]:
    """Data-derived (species -> moves) pairs eligible for CB residual evidence.

    A pair qualifies iff the move is a fixed-power physical damaging move outside the
    exclusion classes (HP-scaled, IV-scaled, scenario-scaled, multi-hit, fixed-damage;
    Return is included at its fixed 102), and every candidate variant of the species
    that carries the move computes the same Attack stat within ``atk_tolerance`` —
    i.e. the observed move itself pins the attacker's Attack. ``universes`` is
    ``Gen3RandbatSource.universes`` (or any mapping of objects with ``.variants``
    exposing ``moves`` / ``ability`` / ``item`` / ``level``).
    """
    stats_cache: dict[tuple, Mapping[str, int]] = {}
    whitelist: dict[str, frozenset[str]] = {}
    for species_key, universe in universes.items():
        key = _species_key(species_key)
        variants = [
            CandidateVariant(
                moves=tuple(normalize_id(str(move)) for move in getattr(variant, "moves", ())),
                ability=normalize_id(str(getattr(variant, "ability", "") or "")),
                item=normalize_id(str(getattr(variant, "item", "") or "")),
                level=int(getattr(variant, "level", 100) or 100),
            )
            for variant in getattr(universe, "variants", ())
        ]
        eligible: set[str] = set()
        all_moves = {move for variant in variants for move in variant.moves}
        for move_id in all_moves:
            if move_id in CB_EXCLUDED_MOVES or move_id.startswith("hiddenpower"):
                continue
            info = dex.move_info(move_id)
            if info is None or info.gen3_category != "Physical":
                continue
            base_power = RETURN_BASE_POWER if move_id == "return" else info.base_power
            if base_power <= 0:
                continue
            carriers = [variant for variant in variants if move_id in variant.moves]
            if not carriers:
                continue
            atk_values = []
            for variant in carriers:
                stats = _variant_stats(variant, key, dex, stats_cache)
                if stats is None:
                    atk_values = []
                    break
                atk_values.append(stats["atk"])
            if not atk_values:
                continue
            if max(atk_values) - min(atk_values) > atk_tolerance * max(atk_values):
                continue
            eligible.add(move_id)
        if eligible:
            whitelist[key] = frozenset(eligible)
    return whitelist


# --- Per-variant expected damage under public conditioning. ---


@dataclass(frozen=True)
class _VariantDamage:
    variant: CandidateVariant
    rolls: tuple[int, ...]  # per-hit rolls under the FULL variant interpretation
    baseline_rolls: tuple[int, ...]  # Choice Band neutralized (unless config says else)
    ambiguous: bool  # a conditioning input sat in an ambiguity band


def _weather_suppressed(variant_ability: str, own_ability: Optional[str]) -> bool:
    if variant_ability in _WEATHER_SUPPRESSORS:
        return True
    return normalize_id(own_ability or "") in _WEATHER_SUPPRESSORS


@dataclass(frozen=True)
class StrikeParticipant:
    """One side of a strike for the shared damage core, direction-neutral.

    Base Tier 2 (opponent attacks us) builds the ATTACKER from a candidate variant and
    the DEFENDER from the exact own mon; the investment inference (we attack them,
    :mod:`pokezero.investment`) builds the ATTACKER from the exact own mon and the
    DEFENDER from a candidate variant. Same conditioning code either way — the
    defender-side extension must never fork the modifier chain.

    ``stats`` are stored stats; the defender's must include ``"hp"`` (max HP).
    ``moves`` are normalized ids used to resolve the observed move against the
    participant's set (Hidden Power type resolution); empty means no set constraint.
    """

    species_key: str
    level: int
    stats: Mapping[str, int]
    ability: str  # normalized id ("" when unknown)
    item: str  # normalized id ("" when unknown)
    moves: tuple[str, ...] = ()


@dataclass(frozen=True)
class _StrikeDamage:
    rolls: tuple[int, ...]  # per-hit rolls under the FULL interpretation
    baseline_rolls: tuple[int, ...]  # Choice Band neutralized when split is requested
    ambiguous: bool  # a conditioning input sat in an ambiguity band


def _participant_move_id(participant: StrikeParticipant, observed_move: str) -> Optional[str]:
    """The participant's own id for the observed move (Hidden Power resolves per set)."""
    if observed_move.startswith("hiddenpower"):
        for move in participant.moves:
            if move.startswith("hiddenpower"):
                return move
        return None
    return observed_move if observed_move in participant.moves else None


def _strike_damage(
    *,
    attacker: StrikeParticipant,
    defender: StrikeParticipant,
    defender_types: tuple[str, ...],
    observed_move: str,
    token: TransitionToken,
    context: StrikeContext,
    dex: ShowdownDex,
    require_move_in_set: bool,
    pinch_ambiguity_band: float,
    split_choice_band_baseline: bool,
) -> Optional[_StrikeDamage]:
    """Direction-neutral expected-damage core (the single Tier-2 conditioning chain).

    ``context`` is always oriented with the ATTACKER as the acting side (the fold
    snapshots ``attacker_*`` fields for whichever side declared the move).
    ``split_choice_band_baseline`` requests the CB-neutralized baseline alongside the
    full interpretation (base Tier 2's residual baseline); when False the baseline
    equals the full rolls.
    """
    move_id = _participant_move_id(attacker, observed_move)
    if move_id is None:
        if require_move_in_set:
            return None
        move_id = observed_move
    info = dex.move_info(move_id)
    if info is None or info.gen3_category not in {"Physical", "Special"}:
        return None
    category = info.gen3_category
    move_type = info.type
    ambiguous = False

    # Base power, with public conditioning.
    if move_id == "return":
        base_power = RETURN_BASE_POWER
    elif move_id in {"flail", "reversal"}:
        base_power = resolve_move_base_power(info, context.attacker_hp_fraction)
        low = resolve_move_base_power(info, max(0.0, context.attacker_hp_fraction - 0.01))
        high = resolve_move_base_power(info, min(1.0, context.attacker_hp_fraction + 0.01))
        if low != high:
            ambiguous = True  # percent quantization straddles a breakpoint
    elif move_id == "pursuit":
        base_power = 80 if token.pursuit_intercept else info.base_power
    else:
        base_power = info.base_power
    if base_power <= 0:
        return None

    # BasePower-event mods (mods/gen4 inherited by gen3): Solar Beam weather, Facade,
    # the pinch abilities (onBasePower), and Thick Fat (onSourceBasePower).
    base_power_mods: list[tuple[float, float]] = []
    weather = normalize_id(token.weather or "")
    suppressed = _weather_suppressed(attacker.ability, defender.ability)
    if move_id == "solarbeam" and weather in _SOLAR_BEAM_WEAK_WEATHERS and not suppressed:
        base_power_mods.append((0.5, 1))
    if move_id == "facade" and (context.attacker_status or "") in _FACADE_STATUSES:
        base_power_mods.append((2, 1))
    ability = attacker.ability
    pinch_type = _PINCH_ABILITY_TYPES.get(ability)
    if pinch_type is not None and move_type == pinch_type:
        fraction = context.attacker_hp_fraction
        if abs(3.0 * fraction - 1.0) <= pinch_ambiguity_band:
            ambiguous = True
            base_power_mods.append((1.5, 1))  # conservative-max interpretation
        elif fraction <= 1.0 / 3.0:
            base_power_mods.append((1.5, 1))
    defender_ability = defender.ability
    if defender_ability == "thickfat" and move_type in {"Fire", "Ice"}:
        base_power_mods.append((0.5, 1))

    # ModifyDamagePhase1 mods beyond the screen (which gen3_damage applies itself):
    # the Flash Fire volatile (mods/gen4/abilities.ts, inherited).
    phase1_mods: list[tuple[float, float]] = []
    if context.attacker_flash_fire and move_type == "Fire":
        phase1_mods.append((1.5, 1))

    attack_stat_key = "atk" if category == "Physical" else "spa"
    defense_stat_key = "def" if category == "Physical" else "spd"
    attack = attacker.stats.get(attack_stat_key)
    defense = defender.stats.get(defense_stat_key)
    max_hp = defender.stats.get("hp")
    if attack is None or defense is None or max_hp is None:
        return None

    # Attack-stat modifier chain (ModifyAtk/SpA events: abilities then items). Hustle
    # is the one DIRECT-modify handler (it truncates the stat itself; chained handlers
    # accumulate into one finalModify — Hustle+CB truncates twice in the engine).
    attack_mods: list[tuple[float, float]] = []
    attack_direct_mods: list[tuple[float, float]] = []
    guts_active = ability == "guts" and (context.attacker_status or "") in _GUTS_STATUSES
    if guts_active:
        attack_mods.append((1.5, 1))
    if ability == "hustle" and category == "Physical":
        attack_direct_mods.append((1.5, 1))
    if ability in {"hugepower", "purepower"} and category == "Physical":
        attack_mods.append((2, 1))
    # Gen3 Plus/Minus check ALL actives, not allies (mods/gen3/abilities.ts) — in
    # singles the partner is the OPPOSING active (the defender here). Reachable via
    # Plusle/Minun; the inventory's "inert in singles" note is dex-level, not
    # engine-level.
    partner = {"minus": "plus", "plus": "minus"}.get(ability)
    if partner is not None and category == "Special" and defender_ability == partner:
        attack_mods.append((1.5, 1))

    item_mods: list[tuple[float, float]] = []
    cb_mods: list[tuple[float, float]] = []
    item = attacker.item
    if item == "choiceband" and category == "Physical":
        cb_mods.append((1.5, 1))
    elif item == "thickclub" and category == "Physical" and attacker.species_key in {"marowak", "cubone"}:
        item_mods.append((2, 1))
    elif item == "lightball" and category == "Special" and attacker.species_key == "pikachu":
        item_mods.append((2, 1))
    elif item == "souldew" and category == "Special" and attacker.species_key in {"latias", "latios"}:
        item_mods.append((1.5, 1))
    else:
        boost_type = TYPE_BOOST_ITEMS.get(item)
        if boost_type is not None and boost_type == move_type:
            item_mods.append((_TYPE_BOOST_FACTORS.get(item, 1.1), 1))

    defense_mods: list[tuple[float, float]] = []
    if defender_ability == "marvelscale" and context.defender_status and category == "Physical":
        defense_mods.append((1.5, 1))
    if (
        defender.item == "souldew"
        and category == "Special"
        and defender.species_key in {"latias", "latios"}
    ):
        defense_mods.append((1.5, 1))

    weather_mod: Optional[tuple[float, float]] = None
    if not suppressed:
        if weather == "sunnyday":
            if move_type == "Fire":
                weather_mod = (1.5, 1)
            elif move_type == "Water":
                weather_mod = (0.5, 1)
        elif weather == "raindance":
            if move_type == "Water":
                weather_mod = (1.5, 1)
            elif move_type == "Fire":
                weather_mod = (0.5, 1)

    attacker_info = dex.species_info(attacker.species_key)
    stab = bool(attacker_info and move_type in attacker_info.types)
    effectiveness = dex.effectiveness(move_type, defender_types)
    burned = (
        category == "Physical"
        and (context.attacker_status or "") == "brn"
        and not guts_active
        and ability != "guts"
    )
    screen = ("reflect" if category == "Physical" else "lightscreen") in context.defender_screens

    def build(mods_with_item: Sequence[tuple[float, float]]) -> tuple[int, ...]:
        return gen3_damage_rolls(
            Gen3DamageContext(
                level=attacker.level,
                base_power=base_power,
                category=category,
                attack=attack,
                defense=defense,
                attack_boost=int(context.attacker_boosts.get(attack_stat_key, 0)),
                defense_boost=int(context.defender_boosts.get(defense_stat_key, 0)),
                attack_mods=tuple(mods_with_item),
                attack_direct_mods=tuple(attack_direct_mods),
                defense_mods=tuple(defense_mods),
                base_power_mods=tuple(base_power_mods),
                phase1_mods=tuple(phase1_mods),
                stab=stab,
                effectiveness=effectiveness,
                burned=burned,
                screen=screen,
                weather_mod=weather_mod,
                crit=token.crit,
                explosion_def_halving=move_id in {"explosion", "selfdestruct"} and category == "Physical",
            )
        )

    full_rolls = build(attack_mods + item_mods + cb_mods)
    if cb_mods and split_choice_band_baseline:
        baseline_rolls = build(attack_mods + item_mods)
    else:
        baseline_rolls = full_rolls
    return _StrikeDamage(rolls=full_rolls, baseline_rolls=baseline_rolls, ambiguous=ambiguous)


def _variant_damage(
    *,
    variant: CandidateVariant,
    attacker_species_key: str,
    observed_move: str,
    token: TransitionToken,
    context: StrikeContext,
    own: OwnMon,
    own_types: tuple[str, ...],
    dex: ShowdownDex,
    config: Tier2Config,
    stats_cache: dict[tuple, Mapping[str, int]],
    require_move_in_set: bool,
) -> Optional[_VariantDamage]:
    """Base Tier-2 direction: candidate-variant attacker vs exact own defender."""
    stats = _variant_stats(variant, attacker_species_key, dex, stats_cache)
    if stats is None:
        return None
    damage = _strike_damage(
        attacker=StrikeParticipant(
            species_key=attacker_species_key,
            level=variant.level,
            stats=stats,
            ability=variant.ability,
            item=variant.item,
            moves=variant.moves,
        ),
        defender=StrikeParticipant(
            species_key=_species_key(own.species),
            level=own.level,
            stats=own.stats,
            ability=normalize_id(own.ability or ""),
            item=normalize_id(own.item or ""),
        ),
        defender_types=own_types,
        observed_move=observed_move,
        token=token,
        context=context,
        dex=dex,
        require_move_in_set=require_move_in_set,
        pinch_ambiguity_band=config.pinch_ambiguity_band,
        split_choice_band_baseline=not config.baseline_includes_choice_band,
    )
    if damage is None:
        return None
    return _VariantDamage(
        variant=variant,
        rolls=damage.rolls,
        baseline_rolls=damage.baseline_rolls,
        ambiguous=damage.ambiguous,
    )


# --- Strike assessment and the top-level inference. ---


@dataclass(frozen=True)
class StrikeAssessment:
    token_index: int
    turn: int
    attacker_key: str  # belief key, e.g. "p2:snorlax"
    move_id: str
    observed_hp: Optional[int] = None
    defender_max_hp: Optional[int] = None
    expected_median_hp: Optional[float] = None
    max_non_cb_hp: Optional[int] = None
    max_cb_hp: Optional[int] = None
    baseline_rolls: tuple[int, ...] = ()
    residual: Optional[float] = None  # signed fraction of defender max HP
    residual_valid: bool = False
    cb_eligible: bool = False
    cb_exceeded: bool = False
    disqualifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Tier2Inference:
    perspective_slot: str
    opponent_slot: str
    tokens: tuple[TransitionToken, ...]  # residual fields populated where valid
    strikes: tuple[StrikeAssessment, ...]
    cb_bits: Mapping[str, bool]
    cb_strike_turns: Mapping[str, tuple[int, ...]]


def infer_tier2(
    replay: ShowdownReplayState,
    *,
    perspective_slot: str,
    own_team: Sequence[OwnMon],
    dex: ShowdownDex,
    set_source: Any = None,
    whitelist: Mapping[str, frozenset[str]] | None = None,
    config: Tier2Config | None = None,
    format_id: str = "gen3randombattle",
) -> Tier2Inference:
    """Run Tier-2 residual + CB inference for one perspective over a replay prefix."""
    config = config or Tier2Config()
    if perspective_slot not in {"p1", "p2"}:
        raise ValueError(f"perspective_slot must be 'p1' or 'p2', got {perspective_slot!r}.")
    opponent = _other(perspective_slot)
    if whitelist is None:
        universes = getattr(set_source, "universes", None)
        whitelist = build_cb_whitelist(universes, dex) if isinstance(universes, Mapping) else {}

    fold = _fold_replay(replay, perspective_slot=perspective_slot)
    raw_lines = tuple(event.raw_line for event in replay.public_events)
    contexts = _strike_contexts(raw_lines)
    own_by_species = {_species_key(mon.species): mon for mon in own_team}

    engine = PublicBattleBeliefEngine(format_id=format_id, set_source=set_source)
    events = replay.public_events
    fed = 0
    stats_cache: dict[tuple, Mapping[str, int]] = {}
    strikes: list[StrikeAssessment] = []
    residuals: dict[int, tuple[Optional[float], bool]] = {}
    cb_turns: dict[str, list[int]] = {}
    cb_non_ko: set[str] = set()
    # As-of-strike CB conclusion per assessed token index (corrections item 9's CB
    # slot): monotone within a battle — once a mon concludes, every later assessed
    # strike token of that mon carries the bit.
    cb_bit_indices: set[int] = set()

    windows = list(fold.windows)
    for index, (token, window) in enumerate(zip(fold.tokens, windows)):
        feed_until = windows[index + 1].event_index if index + 1 < len(windows) else len(events)
        while fed < feed_until:
            engine.ingest_event(events[fed])
            fed += 1
        if token.kind != TOKEN_KIND_MOVE or token.actor_slot != opponent:
            continue
        context = contexts.get(window.event_index)
        if context is None:
            continue
        assessment = _assess_strike(
            token=token,
            token_index=index,
            context=context,
            engine=engine,
            opponent=opponent,
            own_by_species=own_by_species,
            dex=dex,
            whitelist=whitelist,
            config=config,
            stats_cache=stats_cache,
        )
        if assessment is None:
            continue
        strikes.append(assessment)
        if assessment.residual_valid and assessment.residual is not None:
            residuals[index] = (assessment.residual, True)
        if assessment.cb_eligible and assessment.cb_exceeded:
            cb_turns.setdefault(assessment.attacker_key, []).append(assessment.turn)
            if not token.ko:
                cb_non_ko.add(assessment.attacker_key)
        if (
            len(cb_turns.get(assessment.attacker_key, ())) >= config.required_cb_strikes
            and assessment.attacker_key in cb_non_ko
        ):
            cb_bit_indices.add(index)

    # Feed any trailing events so the engine ends at the true boundary (parity with a
    # from_events construction; no evaluation depends on it).
    while fed < len(events):
        engine.ingest_event(events[fed])
        fed += 1

    tokens = tuple(
        replace(
            token,
            residual=residuals[index][0] if index in residuals else token.residual,
            residual_valid=True if index in residuals else token.residual_valid,
            cb_bit=index in cb_bit_indices,
        )
        if index in residuals or index in cb_bit_indices
        else token
        for index, token in enumerate(fold.tokens)
    )
    # The bit needs the two-strike count AND at least one NON-KO exceedance: a KO-
    # clipped observation is understated, which weakens the off-model upper guard on
    # that strike, so KO strikes alone may never flip the bit.
    cb_bits = {
        key: len(turns) >= config.required_cb_strikes and key in cb_non_ko
        for key, turns in cb_turns.items()
    }
    return Tier2Inference(
        perspective_slot=perspective_slot,
        opponent_slot=opponent,
        tokens=tokens,
        strikes=tuple(strikes),
        cb_bits=cb_bits,
        cb_strike_turns={key: tuple(turns) for key, turns in cb_turns.items()},
    )


def apply_residuals(
    tokens: Sequence[TransitionToken], inference: Tier2Inference
) -> tuple[TransitionToken, ...]:
    """Copy the inference's residual fields onto a caller-held token tuple.

    Convenience for the observation-encoder wiring (PR #502's reserved slots): the
    caller's tokens must be the same extraction (same length/order) this inference ran
    over.
    """
    if len(tokens) != len(inference.tokens):
        raise ValueError("token sequence does not match the inference's extraction.")
    return tuple(
        replace(
            token,
            residual=inferred.residual,
            residual_valid=inferred.residual_valid,
            cb_bit=inferred.cb_bit,
        )
        for token, inferred in zip(tokens, inference.tokens)
    )


_WHITELIST_CACHE: dict[str, Mapping[str, frozenset[str]]] = {}
_WHITELIST_CACHE_LOCK = threading.Lock()


def cb_whitelist_for_source(set_source: Any, dex: ShowdownDex) -> Mapping[str, frozenset[str]]:
    """Process-wide cached CB whitelist for a randbats set source.

    Building the whitelist enumerates every species' candidate universe (seconds); the
    live env would otherwise pay it per battle. Keyed by the source's provenance hash,
    mirroring ``load_gen3_randbat_source_cached``.
    """
    universes = getattr(set_source, "universes", None)
    if not isinstance(universes, Mapping):
        return {}
    metadata = getattr(set_source, "metadata", None)
    key = getattr(metadata, "source_hash", None) or f"id:{id(set_source)}"
    with _WHITELIST_CACHE_LOCK:
        cached = _WHITELIST_CACHE.get(key)
        if cached is not None:
            return cached
    built = build_cb_whitelist(universes, dex)
    with _WHITELIST_CACHE_LOCK:
        return _WHITELIST_CACHE.setdefault(key, built)


class Tier2LiveTracker:
    """Incremental live consumer: annotates transition tokens with Tier-2 residuals.

    The batch entry point (:func:`infer_tier2`) refolds the whole replay per call —
    fine for the offline gate, quadratic if an env did it per observation. This
    tracker is the per-battle live form used by the collection env:

    - it SHARES the caller's belief engine (the env already feeds it each protocol
      line exactly once) rather than running a second candidate-filtering pass;
    - each protocol line is context-folded once (``_IncrementalContextFold``);
    - each opponent move token is assessed once, at the first ``annotate`` call that
      sees it — i.e. against the belief state at the first observation boundary after
      the strike.

    Live-vs-batch invariant (evidence-monotone consistency, NOT exact equality): the
    live boundary can carry strictly MORE belief evidence than ``infer_tier2``'s
    next-action cutoff for the same strike (end-of-turn non-proc pruning lands inside
    the same observation window), so live may conservatively diverge from batch —
    standing a CB strike down (e.g. ``cb-pinned-by-elimination`` once Leftovers is
    pruned) or masking a residual — but never the reverse, and residual VALUES are
    identical wherever both sides are valid. Most games are outright equal; the
    divergence classes are rare (4 perspective-level cases in a 120-perspective
    review sweep) and always in the more-evidence direction.

    Per-``annotate`` work is O(new lines + new strikes); memory is O(actions).
    """

    def __init__(
        self,
        *,
        perspective_slot: str,
        own_team: Sequence[OwnMon],
        dex: ShowdownDex,
        whitelist: Mapping[str, frozenset[str]],
        config: Tier2Config | None = None,
    ) -> None:
        if perspective_slot not in {"p1", "p2"}:
            raise ValueError(f"perspective_slot must be 'p1' or 'p2', got {perspective_slot!r}.")
        self._perspective = perspective_slot
        self._opponent = _other(perspective_slot)
        self._own_by_species = {_species_key(mon.species): mon for mon in own_team}
        self._dex = dex
        self._whitelist = whitelist
        self._config = config or Tier2Config()
        self._fold = _IncrementalContextFold()
        self._stats_cache: dict[tuple, Mapping[str, int]] = {}
        self._assessed_until = 0
        self._residuals: dict[int, float] = {}
        self._cb_turns: dict[str, list[int]] = {}
        self._cb_non_ko: set[str] = set()
        self._cb_bit_indices: set[int] = set()

    def clone(self) -> "Tier2LiveTracker":
        """Clone the incremental state for one independent search branch.

        The dex, whitelist, and stat-cache values are read-only after creation. The
        branch-specific fold and inference ledgers are copied so sibling branches
        cannot influence one another's residual or Choice Band conclusions.
        """

        cloned = Tier2LiveTracker.__new__(Tier2LiveTracker)
        cloned._perspective = self._perspective
        cloned._opponent = self._opponent
        cloned._own_by_species = self._own_by_species
        cloned._dex = self._dex
        cloned._whitelist = self._whitelist
        cloned._config = self._config
        cloned._fold = self._fold.clone()
        cloned._stats_cache = dict(self._stats_cache)
        cloned._assessed_until = self._assessed_until
        cloned._residuals = dict(self._residuals)
        cloned._cb_turns = {key: list(turns) for key, turns in self._cb_turns.items()}
        cloned._cb_non_ko = set(self._cb_non_ko)
        cloned._cb_bit_indices = set(self._cb_bit_indices)
        return cloned

    @property
    def cb_bits(self) -> dict[str, bool]:
        """Per-opponent-mon CB bit under the two-strike + non-KO rules (diagnostics;
        the observation currently reserves residual/validity slots only)."""
        return {
            key: len(turns) >= self._config.required_cb_strikes and key in self._cb_non_ko
            for key, turns in self._cb_turns.items()
        }

    def annotate(
        self,
        replay: ShowdownReplayState,
        tokens: Sequence[TransitionToken],
        belief_engine: PublicBattleBeliefEngine,
    ) -> tuple[TransitionToken, ...]:
        """Assess any new opponent strikes and return tokens with residual fields set.

        ``tokens`` must be the transition extraction for the same ``replay`` (the env's
        ``normalize_for_player`` output); ``belief_engine`` is the caller's persistent
        engine, already fed through the replay's boundary.
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
            if token.kind != TOKEN_KIND_MOVE or token.actor_slot != self._opponent:
                continue
            context = self._fold.contexts.get(self._fold.token_line_indices[index])
            if context is None:
                continue
            assessment = _assess_strike(
                token=token,
                token_index=index,
                context=context,
                engine=belief_engine,
                opponent=self._opponent,
                own_by_species=self._own_by_species,
                dex=self._dex,
                whitelist=self._whitelist,
                config=self._config,
                stats_cache=self._stats_cache,
            )
            if assessment is None:
                continue
            if assessment.residual_valid and assessment.residual is not None:
                self._residuals[index] = assessment.residual
            if assessment.cb_eligible and assessment.cb_exceeded:
                self._cb_turns.setdefault(assessment.attacker_key, []).append(assessment.turn)
                if not token.ko:
                    self._cb_non_ko.add(assessment.attacker_key)
            if (
                len(self._cb_turns.get(assessment.attacker_key, ())) >= self._config.required_cb_strikes
                and assessment.attacker_key in self._cb_non_ko
            ):
                self._cb_bit_indices.add(index)
        self._assessed_until = len(tokens)
        if not self._residuals and not self._cb_bit_indices:
            return tuple(tokens)
        return tuple(
            replace(
                token,
                residual=self._residuals.get(index, token.residual),
                residual_valid=True if index in self._residuals else token.residual_valid,
                cb_bit=index in self._cb_bit_indices,
            )
            if index in self._residuals or index in self._cb_bit_indices
            else token
            for index, token in enumerate(tokens)
        )


def _assess_strike(
    *,
    token: TransitionToken,
    token_index: int,
    context: StrikeContext,
    engine: PublicBattleBeliefEngine,
    opponent: str,
    own_by_species: Mapping[str, OwnMon],
    dex: ShowdownDex,
    whitelist: Mapping[str, frozenset[str]],
    config: Tier2Config,
    stats_cache: dict[tuple, Mapping[str, int]],
) -> Optional[StrikeAssessment]:
    move_id = normalize_id(token.action)
    attacker_species_key = _species_key(token.actor_species)
    attacker_key = belief_key(opponent, token.actor_species)
    base = StrikeAssessment(
        token_index=token_index,
        turn=token.turn,
        attacker_key=attacker_key,
        move_id=move_id,
    )
    disqualifiers: list[str] = []

    if move_id == "struggle":
        return replace(base, disqualifiers=("struggle",))
    info = dex.move_info(move_id)
    is_hidden_power = move_id.startswith("hiddenpower")
    if not is_hidden_power:
        if info is None:
            return replace(base, disqualifiers=("unknown-move",))
        if info.gen3_category not in {"Physical", "Special"}:
            return None  # status move: no damage channel at all
        if info.base_power <= 0 and move_id not in _BP_CALLBACK_PHYSICAL and move_id != "pursuit":
            return replace(base, disqualifiers=("no-base-power",))

    # Negation outcomes and misses: no damage event occurred — no evidence either way.
    if token.miss or token.damage_outcome in _NEGATION_OUTCOMES or token.damage_fraction <= 0:
        return replace(base, disqualifiers=("no-damage-event",))

    if token.transformed or context.attacker_transformed:
        return replace(base, disqualifiers=("transformed-attacker",))
    if context.defender_transformed:
        return replace(base, disqualifiers=("transformed-defender",))
    if context.attacker_type_changed or context.defender_type_changed:
        # Color Change / Conversion-class typechange, or a Forecast forme change:
        # STAB and effectiveness are species-derived here, and the live typing no
        # longer matches the species.
        return replace(base, disqualifiers=("type-changed",))
    if context.attacker_item_mutated or context.defender_item_mutated:
        # Trick / Knock Off changed the held item mid-game: candidate-variant item
        # modifiers describe the ORIGINAL assignment, not the current holder
        # (corrections item 15). Conservative: no damage inference on such mons.
        return replace(base, disqualifiers=("item-mutated",))
    if context.attacker_ability_overridden or context.defender_ability_overridden:
        # Trace-class acquisition replaced a live ability (announced |-ability| with
        # an acquisition tag): every ability-conditioned modifier — on either side —
        # may now be wrong. Symmetric to item-mutated; structural, not species-keyed.
        return replace(base, disqualifiers=("ability-overridden",))

    own = own_by_species.get(_species_key(context.defender_species))
    if own is None or "hp" not in own.stats:
        return replace(base, disqualifiers=("unknown-defender",))
    max_hp = int(own.stats["hp"])
    own_info = dex.species_info(_species_key(own.species))
    own_types = own_info.types if own_info is not None else ()
    observed_hp = int(round(token.damage_fraction * max_hp))

    truncated = token.damage_outcome in _TRUNCATION_OUTCOMES
    if truncated:
        disqualifiers.append("truncated-outcome")

    belief = next(
        (
            mon
            for mon in engine.snapshot().side(opponent)
            if _species_key(mon.species) == attacker_species_key
        ),
        None,
    )
    if belief is None:
        return replace(base, observed_hp=observed_hp, defender_max_hp=max_hp, disqualifiers=("no-belief",))
    variants = [CandidateVariant.from_mapping(payload) for payload in belief.candidate_variants]
    if not variants:
        return replace(base, observed_hp=observed_hp, defender_max_hp=max_hp, disqualifiers=("no-candidates",))

    evaluated: list[_VariantDamage] = []
    for variant in variants:
        damage = _variant_damage(
            variant=variant,
            attacker_species_key=attacker_species_key,
            observed_move=move_id,
            token=token,
            context=context,
            own=own,
            own_types=own_types,
            dex=dex,
            config=config,
            stats_cache=stats_cache,
            # Called executions (Sleep Talk) charge no set evidence, so belief did not
            # prune by the executed move; require the move in-set locally instead.
            require_move_in_set=True,
        )
        if damage is not None and damage.rolls:
            # A variant that cannot deal the observed damage at all (e.g. its Hidden
            # Power type is immune against us) is impossible given the observed hit;
            # drop it rather than invalidating the strike.
            evaluated.append(damage)
    if not evaluated:
        return replace(base, observed_hp=observed_hp, defender_max_hp=max_hp, disqualifiers=("no-computable-variant",))

    ambiguous = any(entry.ambiguous for entry in evaluated)
    n_hits = max(1, token.n_hits)

    # Residual: candidate-conservative baseline medians must agree.
    medians = [median_damage(entry.baseline_rolls) * n_hits for entry in evaluated]
    median_low, median_high = min(medians), max(medians)
    agreement = max(config.baseline_agreement_hp, config.baseline_agreement_fraction * max(median_high, 1.0))
    baseline_agrees = (median_high - median_low) <= agreement
    expected_median = (median_low + median_high) / 2.0

    # Multi-hit (Bonemerang) residuals are masked: the summed-roll population is
    # excluded from the gate's calibration arm, and production validity must exactly
    # match the calibrated population. Crit strikes STAY valid — they are calibrated
    # (crit-conditioned) alongside plain strikes.
    residual_valid = (
        baseline_agrees and not ambiguous and not truncated and not token.ko and n_hits == 1
    )
    residual = (observed_hp - expected_median) / max_hp if residual_valid else None
    if not baseline_agrees:
        disqualifiers.append("baseline-disagreement")
    if ambiguous:
        disqualifiers.append("conditioning-ambiguity")
    if token.ko:
        disqualifiers.append("ko-clipped")

    # Choice Band evaluation.
    cb_eligible = True
    if token.crit:
        cb_eligible = False
        disqualifiers.append("crit")
    if token.called:
        cb_eligible = False
        disqualifiers.append("called")
    if token.damage_outcome != DAMAGE_OUTCOME_NORMAL:
        cb_eligible = False
    if n_hits != 1:
        cb_eligible = False
        disqualifiers.append("multi-hit")
    if move_id not in whitelist.get(attacker_species_key, frozenset()):
        cb_eligible = False
        disqualifiers.append("not-whitelisted")
    if context.attacker_boosts.get("atk", 0) != 0 or context.defender_boosts.get("def", 0) != 0:
        cb_eligible = False
        disqualifiers.append("stat-stages")
    if "reflect" in context.defender_screens:
        cb_eligible = False
        disqualifiers.append("screen")
    if ambiguous:
        cb_eligible = False
    non_cb = [entry for entry in evaluated if entry.variant.item != "choiceband"]
    cb = [entry for entry in evaluated if entry.variant.item == "choiceband"]
    if not cb:
        cb_eligible = False
        disqualifiers.append("cb-not-a-candidate")
    if not non_cb:
        cb_eligible = False
        disqualifiers.append("cb-pinned-by-elimination")

    max_non_cb = max((max(entry.rolls) for entry in non_cb), default=None)
    max_cb = max((max(entry.rolls) for entry in cb), default=None)
    margin = config.cb_margin_fraction * max_hp + config.cb_margin_hp
    cb_exceeded = False
    if cb_eligible and max_non_cb is not None and max_cb is not None:
        exceeds_non_cb = observed_hp > max_non_cb + margin
        # Off-model guard: damage beyond even the best CB explanation means something
        # unmodeled happened — never count it. Caveat: on a KO the observed value is
        # clipped at the defender's remaining HP, which understates true damage and
        # can slip an off-model hit under this ceiling; the bit therefore requires at
        # least one NON-KO exceedance among its strikes (see infer_tier2).
        within_cb = observed_hp <= max_cb + margin
        cb_exceeded = exceeds_non_cb and within_cb
        if exceeds_non_cb and not within_cb:
            disqualifiers.append("exceeds-cb-explanation")

    single_variant_rolls = evaluated[0].baseline_rolls if len(evaluated) == 1 else ()
    return replace(
        base,
        observed_hp=observed_hp,
        defender_max_hp=max_hp,
        expected_median_hp=expected_median if baseline_agrees else None,
        max_non_cb_hp=max_non_cb,
        max_cb_hp=max_cb,
        baseline_rolls=single_variant_rolls,
        residual=residual,
        residual_valid=residual_valid,
        cb_eligible=cb_eligible,
        cb_exceeded=cb_exceeded,
        disqualifiers=tuple(disqualifiers),
    )
