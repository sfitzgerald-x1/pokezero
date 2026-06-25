"""Small local Showdown dex loader for scripted policies and metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import threading
from typing import Any, Mapping, Optional


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
    secondary_chance: int = 0  # strongest secondary-effect chance in percent (0-100)
    secondary_effect: str = ""  # effect class of that secondary (frz / flinch / lower_def / ...)


@dataclass(frozen=True)
class SpeciesInfo:
    id: str
    name: str
    types: tuple[str, ...]
    base_stats: Mapping[str, int]


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
    moves_path = root / "dist" / "data" / "moves.js"
    pokedex_path = root / "dist" / "data" / "pokedex.js"
    typechart_path = root / "dist" / "data" / "typechart.js"
    if not moves_path.exists() or not pokedex_path.exists() or not typechart_path.exists():
        raise FileNotFoundError(
            "Built Pokemon Showdown dex data is missing. Expected dist/data/moves.js, "
            "dist/data/pokedex.js, and dist/data/typechart.js under the Showdown root."
        )
    script = """
const root = process.argv[1];
const {Moves} = require(root + '/dist/data/moves.js');
const {Pokedex} = require(root + '/dist/data/pokedex.js');
const {TypeChart} = require(root + '/dist/data/typechart.js');
const out = {moves: {}, species: {}, typeChart: {}};
for (const [id, move] of Object.entries(Moves)) {
  const boosts = move.boosts || (move.secondary && move.secondary.boosts) || {};
  // Secondary-effect TYPE + chance. A secondary object without an explicit chance is guaranteed
  // (100%). We summarize the strongest secondary as a single effect label (status / flinch /
  // confusion / stat change) so the model can distinguish e.g. a freeze chance from an
  // attack-raise chance; secondaryChance carries its probability.
  // Stat-change labels are target-explicit: a secondary's `boosts` hits the foe, while `self.boosts`
  // hits the user — so Psychic's 10% -SpD is lower_foe_spd, but Meteor Mash's 20% +Atk is
  // raise_self_atk. Statuses/flinch/confusion are always foe-targeted (no direction needed).
  const statLabel = (boosts, who) => {
    for (const [stat, val] of Object.entries(boosts || {})) {
      if (val < 0) return 'lower_' + who + '_' + stat;
      if (val > 0) return 'raise_' + who + '_' + stat;
    }
    return '';
  };
  const effectLabel = (s) => {
    if (!s) return '';
    if (s.volatileStatus) return String(s.volatileStatus);   // flinch, confusion, ...
    if (s.status) return String(s.status);                   // brn, par, frz, psn, slp, tox
    return statLabel(s.boosts, 'foe') || (s.self && statLabel(s.self.boosts, 'self')) || 'other';
  };
  const secondaries = [];
  if (move.secondary) secondaries.push(move.secondary);
  if (Array.isArray(move.secondaries)) secondaries.push(...move.secondaries);
  let secondaryChance = 0, secondaryEffect = '';
  for (const s of secondaries) {
    if (!s) continue;
    const c = (typeof s.chance === 'number') ? s.chance : 100;
    if (c >= secondaryChance) { secondaryChance = c; secondaryEffect = effectLabel(s); }
  }
  // Guaranteed self-stat drawbacks (Overheat -2 SpA, Superpower -1 Atk/-1 Def, ...) live in
  // move.self.boosts at 100% rather than in a probabilistic secondary — capture them too.
  if (!secondaryEffect && move.self && move.self.boosts) {
    const l = statLabel(move.self.boosts, 'self');
    if (l) { secondaryEffect = l; secondaryChance = 100; }
  }
  out.moves[id] = {
    id,
    name: move.name,
    type: move.type,
    category: move.category,
    basePower: move.basePower || 0,
    accuracy: move.accuracy === true ? 100 : (move.accuracy || 0),
    priority: move.priority || 0,
    recoil: Boolean(move.recoil || move.hasCrashDamage),
    drain: Boolean(move.drain),
    heal: Boolean(move.heal),
    status: move.status || (move.secondary && move.secondary.status) || null,
    boosts,
    secondaryChance,
    secondaryEffect,
    target: move.target || '',
    selfdestruct: Boolean(move.selfdestruct)
  };
}
for (const [id, species] of Object.entries(Pokedex)) {
  out.species[id] = {
    id,
    name: species.name,
    types: species.types || [],
    baseStats: species.baseStats || {}
  };
}
for (const [id, typeInfo] of Object.entries(TypeChart)) {
  out.typeChart[id] = typeInfo.damageTaken || {};
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


def _move_info_from_payload(move_id: str, payload: Mapping[str, Any]) -> MoveInfo:
    move_type = str(payload.get("type") or "")
    category = str(payload.get("category") or "Status")
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
        secondary_chance=int(payload.get("secondaryChance") or 0),
        secondary_effect=str(payload.get("secondaryEffect") or ""),
    )


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
