"""Small local Showdown dex loader for scripted policies and metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import threading
from typing import Any, Mapping, Optional, Sequence


PHYSICAL_TYPES = {"Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel"}
SPECIAL_TYPES = {"Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark"}


@dataclass(frozen=True)
class MoveInfo:
    id: str
    name: str
    type: str
    category: str
    gen3_category: str
    base_power: int
    accuracy: float
    priority: int
    recoil: bool
    drain: bool
    heal: bool
    status: Optional[str]
    boosts: Mapping[str, int]
    target: str
    selfdestruct: bool
    # Unified move-effect label (primary OR secondary): a status (par/brn/...), a volatile
    # (substitute/leechseed/flinch/...), or a target-explicit, magnitude-enumerated stat change
    # (raise_self_atk / raise_self_atk_sharply / raise_self_atk_max / lower_foe_def_sharply /
    # raise_self_all / lower_self_atkdef / ...). effect_chance is its probability (100 = guaranteed).
    effect_chance: int = 0
    effect_label: str = ""
    # Fraction of the user's max HP the move spends upfront as a deterrent (Belly Drum 0.5,
    # Substitute 0.25, Explosion/Self-Destruct 1.0). Recoil (damage-proportional) is separate.
    self_hp_cost: float = 0.0
    # Base PP from the dex (0 when unknown). Randbat catalog max PP is 3-PP-Up maxed:
    # floor(pp * 8/5) — see max_move_pp(); the opponent PP ledger divides by that.
    pp: int = 0

    @property
    def max_pp(self) -> int:
        """Catalog max PP for randbats sets (3 PP Ups): floor(base * 8/5); 0 when unknown."""
        return (self.pp * 8) // 5 if self.pp > 0 else 0


@dataclass(frozen=True)
class SpeciesInfo:
    id: str
    name: str
    types: tuple[str, ...]
    base_stats: Mapping[str, int]
    weight_kg: float = 0.0


@dataclass(frozen=True)
class ShowdownDex:
    moves: Mapping[str, MoveInfo]
    species: Mapping[str, SpeciesInfo]
    type_chart: Mapping[str, Mapping[str, int]]

    def move_info(self, move: str | None) -> MoveInfo | None:
        if not move:
            return None
        return self.moves.get(normalize_id(move))

    def species_info(self, species: str | None) -> SpeciesInfo | None:
        if not species:
            return None
        return self.species.get(normalize_id(species))

    def effectiveness(self, move_type: str | None, defender_types: tuple[str, ...]) -> float:
        if not move_type or not defender_types:
            return 1.0
        multiplier = 1.0
        normalized_move_type = normalize_id(move_type)
        for defender_type in defender_types:
            damage_taken = self.type_chart.get(normalize_id(defender_type), {})
            modifier = damage_taken.get(normalized_move_type, 0)
            if modifier == 1:
                multiplier *= 2.0
            elif modifier == 2:
                multiplier *= 0.5
            elif modifier == 3:
                return 0.0
        return multiplier


_DEX_CACHE: dict[Path, ShowdownDex] = {}
_DEX_CACHE_LOCK = threading.Lock()


def load_showdown_dex_cached(showdown_root: Path | str) -> ShowdownDex:
    root = Path(showdown_root).expanduser().resolve()
    with _DEX_CACHE_LOCK:
        cached = _DEX_CACHE.get(root)
        if cached is not None:
            return cached
    loaded = load_showdown_dex(root)
    with _DEX_CACHE_LOCK:
        return _DEX_CACHE.setdefault(root, loaded)


def load_showdown_dex(showdown_root: Path | str) -> ShowdownDex:
    root = Path(showdown_root).expanduser().resolve()
    sim_entry = root / "dist" / "sim" / "index.js"
    moves_path = root / "dist" / "data" / "moves.js"
    pokedex_path = root / "dist" / "data" / "pokedex.js"
    if not sim_entry.exists() or not moves_path.exists() or not pokedex_path.exists():
        raise FileNotFoundError(
            "Built Pokemon Showdown data is missing. Expected dist/sim/index.js, "
            "dist/data/moves.js, and dist/data/pokedex.js under the Showdown root."
        )
    # Source data from the gen3-modded Dex (Dex.forGen(3)) rather than the raw base data files:
    # the base files carry current-generation attributes (e.g. Fairy typing, modern move stats,
    # post-gen3 type-chart changes) that are wrong for a Gen 3 battle. forGen(3) resolves the
    # gen3 typings, base stats, move data, and type chart with full inheritance applied.
    #
    # We iterate the base data KEYS (not gen3.moves.all()) and resolve each through gen3: the .all()
    # iterator collapses Hidden Power's type variants onto a single id ("hiddenpower"), which would
    # drop hiddenpower<type> entries. The base keys keep every variant; gen3.moves.get() returns the
    # gen3-correct, per-variant data.
    script = """
const root = process.argv[1];
const {Moves} = require(root + '/dist/data/moves.js');
const {Pokedex} = require(root + '/dist/data/pokedex.js');
const {Dex} = require(root + '/dist/sim');
const gen3 = Dex.forGen(3);
const out = {moves: {}, species: {}, typeChart: {}};
for (const id of Object.keys(Moves)) {
  const move = gen3.moves.get(id);
  if (!move) continue;
  const boosts = move.boosts || (move.secondary && move.secondary.boosts) || {};
  // Emit the raw effect components; the single move-effect label (type/target/magnitude) and the
  // effect chance are derived in Python (testable, with per-move overrides for custom-onHit moves).
  const secondaries = [];
  for (const s of [move.secondary, ...(Array.isArray(move.secondaries) ? move.secondaries : [])]) {
    if (!s) continue;
    secondaries.push({
      chance: (typeof s.chance === 'number') ? s.chance : 100,
      status: s.status || null,
      volatileStatus: s.volatileStatus || null,
      boosts: s.boosts || {},
      selfBoosts: (s.self && s.self.boosts) || {},
    });
  }
  out.moves[id] = {
    id,
    name: move.name,
    type: move.type,
    category: move.category,
    basePower: move.basePower || 0,
    accuracy: move.accuracy === true ? 100 : (move.accuracy || 0),
    priority: move.priority || 0,
    pp: move.pp || 0,
    recoil: Boolean(move.recoil || move.hasCrashDamage),
    drain: Boolean(move.drain),
    heal: Boolean(move.heal),
    status: move.status || (move.secondary && move.secondary.status) || null,
    boosts,
    topStatus: move.status || null,
    topVolatile: move.volatileStatus || null,
    topBoosts: move.boosts || {},
    selfBoosts: (move.self && move.self.boosts) || {},
    secondaries,
    target: move.target || '',
    selfdestruct: Boolean(move.selfdestruct)
  };
}
for (const id of Object.keys(Pokedex)) {
  const species = gen3.species.get(id);
  if (!species) continue;
  out.species[id] = {
    id,
    name: species.name,
    types: species.types || [],
    baseStats: species.baseStats || {},
    weightkg: species.weightkg || 0
  };
}
for (const type of gen3.types.all()) {
  out.typeChart[type.id] = type.damageTaken || {};
}
console.log(JSON.stringify(out));
"""
    result = subprocess.run(
        ["node", "-e", script, str(root)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return showdown_dex_from_payload(json.loads(result.stdout))


def showdown_dex_from_payload(payload: Mapping[str, Any]) -> ShowdownDex:
    moves = payload.get("moves")
    species = payload.get("species")
    type_chart = payload.get("typeChart")
    if not isinstance(moves, Mapping) or not isinstance(species, Mapping) or not isinstance(type_chart, Mapping):
        raise ValueError("Showdown dex payload must contain moves, species, and typeChart objects.")
    return ShowdownDex(
        moves={
            normalize_id(move_id): _move_info_from_payload(str(move_id), _mapping(raw_move))
            for move_id, raw_move in moves.items()
        },
        species={
            normalize_id(species_id): _species_info_from_payload(str(species_id), _mapping(raw_species))
            for species_id, raw_species in species.items()
        },
        type_chart={
            normalize_id(type_id): {
                normalize_id(attack_type): int(modifier)
                for attack_type, modifier in _mapping(raw_damage_taken).items()
                if isinstance(modifier, int)
            }
            for type_id, raw_damage_taken in type_chart.items()
        },
    )


def gen3_move_category(move_type: str, category: str) -> str:
    if category == "Status":
        return "Status"
    if move_type in PHYSICAL_TYPES:
        return "Physical"
    if move_type in SPECIAL_TYPES:
        return "Special"
    return category or "Status"


def normalize_id(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


_STAT_BOOST_KEYS = ("atk", "def", "spa", "spd", "spe")

# Custom-onHit moves whose effect / HP cost is not in declarative move data, so we name them
# explicitly. Belly Drum maximizes Attack for 50% HP; Substitute's volatile is declarative but its
# 25% HP cost is in onHit.
_MOVE_EFFECT_OVERRIDES: dict[str, dict[str, Any]] = {
    "bellydrum": {"effect_label": "raise_self_atk_max", "effect_chance": 100, "self_hp_cost": 0.5},
    "substitute": {"self_hp_cost": 0.25},
    # Curse is type-dependent and exposes only volatileStatus:"curse" in move data, so its static
    # label is empty; the real label is resolved from the user's type at encode time (see
    # resolve_move_effect / DYNAMIC_MOVE_EFFECT_LABELS).
    "curse": {"effect_label": "", "effect_chance": 0},
}

# Effect labels that resolve_move_effect can produce for type-dependent moves (not present on any
# MoveInfo.effect_label statically), so the vocabulary still enumerates them.
DYNAMIC_MOVE_EFFECT_LABELS: tuple[str, ...] = ("curse", "curse_setup")


def _stat_effect_label(boosts: Mapping[str, Any] | None, who: str) -> str:
    """A target-explicit, magnitude-enumerated stat-change label, or '' if no stat change.

    Direction (raise/lower) is uniform within a Gen 3 boost set; magnitude maps |1|->'' (rose),
    |2|->'_sharply' (rose sharply), >=|3|->'_max' (maximized). Stats are sorted and concatenated,
    collapsing the five-stat omniboost to 'all'.
    """
    entries = [(str(k), int(v)) for k, v in (boosts or {}).items() if isinstance(v, (int, float)) and v]
    if not entries:
        return ""
    magnitude = max(abs(v) for _, v in entries)
    suffix = "" if magnitude <= 1 else ("_sharply" if magnitude == 2 else "_max")
    stats = sorted(k for k, _ in entries)
    stat_str = "all" if set(stats) >= set(_STAT_BOOST_KEYS) else "".join(stats)
    direction = "raise" if entries[0][1] > 0 else "lower"
    return f"{direction}_{who}_{stat_str}{suffix}"


def _secondary_effect_label(secondary: Mapping[str, Any]) -> str:
    if secondary.get("volatileStatus"):
        return str(secondary["volatileStatus"])
    if secondary.get("status"):
        return str(secondary["status"])
    return _stat_effect_label(secondary.get("boosts"), "foe") or _stat_effect_label(
        secondary.get("selfBoosts"), "self"
    )


def _compute_move_effect(payload: Mapping[str, Any]) -> tuple[str, int]:
    """Derive the unified (label, chance) for a move from its raw effect components.

    The strongest secondary effect (highest chance) is used if present, carrying its own chance —
    a damaging move's rider is its notable effect even when guaranteed (chance 100). Only if there
    is no labeled secondary does the move's guaranteed primary effect (status / volatile / stat
    change, target from `target`; or a guaranteed self-boost like Overheat) apply, at chance 100.
    """
    best: tuple[int, str] | None = None
    for secondary in payload.get("secondaries", []) or []:
        if not isinstance(secondary, Mapping):
            continue
        label = _secondary_effect_label(secondary)
        if not label:
            continue
        chance = int(secondary.get("chance") or 100)
        if best is None or chance >= best[0]:
            best = (chance, label)
    if best is not None:
        return best[1], best[0]
    if payload.get("topStatus"):
        return str(payload["topStatus"]), 100
    if payload.get("topVolatile"):
        return str(payload["topVolatile"]), 100
    who = "self" if str(payload.get("target") or "") == "self" else "foe"
    label = _stat_effect_label(payload.get("topBoosts"), who)
    if label:
        return label, 100
    label = _stat_effect_label(payload.get("selfBoosts"), "self")
    if label:
        return label, 100
    return "", 0


def _move_info_from_payload(move_id: str, payload: Mapping[str, Any]) -> MoveInfo:
    move_type = str(payload.get("type") or "")
    category = str(payload.get("category") or "Status")
    effect_label, effect_chance = _compute_move_effect(payload)
    self_hp_cost = 1.0 if payload.get("selfdestruct") else 0.0
    override = _MOVE_EFFECT_OVERRIDES.get(normalize_id(payload.get("id") if payload.get("id") is not None else move_id))
    if override:
        effect_label = override.get("effect_label", effect_label)
        effect_chance = override.get("effect_chance", effect_chance)
        self_hp_cost = override.get("self_hp_cost", self_hp_cost)
    return MoveInfo(
        id=normalize_id(payload.get("id") if payload.get("id") is not None else move_id),
        name=str(payload.get("name") or move_id),
        type=move_type,
        category=category,
        gen3_category=gen3_move_category(move_type, category),
        base_power=int(payload.get("basePower") or 0),
        accuracy=_accuracy_value(payload.get("accuracy")),
        priority=int(payload.get("priority") or 0),
        recoil=bool(payload.get("recoil")),
        drain=bool(payload.get("drain")),
        heal=bool(payload.get("heal")),
        status=_optional_str(payload.get("status")),
        boosts={
            str(stat): int(value)
            for stat, value in _mapping(payload.get("boosts", {})).items()
            if isinstance(value, int)
        },
        target=str(payload.get("target") or ""),
        selfdestruct=bool(payload.get("selfdestruct")),
        effect_chance=int(effect_chance),
        effect_label=str(effect_label),
        self_hp_cost=float(self_hp_cost),
        pp=int(payload.get("pp") or 0),
    )


_HP_BASE_POWER_LOW = frozenset({"reversal", "flail"})  # low user HP -> high base power
_HP_BASE_POWER_HIGH = frozenset({"eruption", "waterspout"})  # high user HP -> high base power


def resolve_move_base_power(move: "MoveInfo", user_hp_fraction: float | None = None) -> int:
    """Base power, resolving HP-variable moves from the user's current HP fraction at encode time.

    Reversal/Flail scale inversely with the user's remaining HP (Gen 3 breakpoints on 48*HP/maxHP);
    Eruption/Water Spout scale directly (150*HP/maxHP). Their static dex base power is 0, so without
    this they would mislead the model. All other moves return their fixed base power.
    """
    if user_hp_fraction is None:
        return move.base_power
    fraction = max(0.0, min(1.0, user_hp_fraction))
    if move.id in _HP_BASE_POWER_LOW:
        # Gen 3 buckets the floored value p = floor(48 * curHP / maxHP), not the raw ratio.
        scaled = int(48 * fraction)
        if scaled <= 1:
            return 200
        if scaled <= 4:
            return 150
        if scaled <= 9:
            return 100
        if scaled <= 16:
            return 80
        if scaled <= 32:
            return 40
        return 20
    if move.id in _HP_BASE_POWER_HIGH:
        return max(1, int(150 * fraction))
    return move.base_power


def resolve_move_effect(move: "MoveInfo", user_types: Sequence[str] = ()) -> tuple[str, int, float]:
    """(effect_label, effect_chance, self_hp_cost), resolving type-dependent moves by the user's type.

    Curse is the one Gen 3 case: a Ghost user lays a foe curse for 50% of its own HP, while a
    non-Ghost user gets the +Atk/+Def/-Spe self setup. A mon's typing is stable within a battle, so
    resolving from the user's types at encode time is well-defined; all other moves pass through.
    """
    if move.id == "curse":
        if any(str(type_name).lower() == "ghost" for type_name in user_types):
            return "curse", 100, 0.5
        return "curse_setup", 100, 0.0
    return move.effect_label, move.effect_chance, move.self_hp_cost


def _species_info_from_payload(species_id: str, payload: Mapping[str, Any]) -> SpeciesInfo:
    base_stats = payload.get("baseStats")
    return SpeciesInfo(
        id=normalize_id(payload.get("id") if payload.get("id") is not None else species_id),
        name=str(payload.get("name") or species_id),
        types=tuple(str(item) for item in _sequence(payload.get("types", ()))),
        base_stats={
            str(stat): int(value)
            for stat, value in _mapping(base_stats if base_stats is not None else {}).items()
            if isinstance(value, int)
        },
        weight_kg=float(payload.get("weightkg") or 0.0),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected object payload.")
    return value


def _sequence(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("expected array payload.")
    return tuple(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _accuracy_value(value: Any) -> float:
    if value is True:
        return 100.0
    if value is False or value is None or value == "":
        return 0.0
    return float(value)
