"""Exact Gen 3 damage math + randbats spread stats, replicating the vendored Showdown sim.

Engine choice (next-train readiness PR D). The frozen spec's corrections layer names
"poke-engine queries" as the Tier-2 expected-damage source, but poke-engine is not a
dependency of this repo, its Python binding exposes no per-move expected-damage query
against a protocol-reconstructed state, and the feasibility assessment
(``docs/poke_engine_assessment.md``) records a REPRODUCED one-turn gen3 damage mismatch
between poke-engine 0.0.47 and the vendored Showdown simulator — the engine that
generates every training game. Tier-2 residuals must be calibrated against the sim that
produces the observations, so this module replicates the vendored sim's gen3 damage
chain in pure Python instead:

- base formula: ``sim/battle-actions.ts getDamage`` —
  ``tr(tr(tr(tr(2*L/5+2) * P * A) / D) / 50)`` with boosted stats and stat-modify events;
- modifier chain: ``data/mods/gen3/scripts.ts modifyDamage`` — burn, weather, +2, crit x2,
  STAB, type effectiveness, screens (base-data Reflect/Light Screen hook the final
  ModifyDamage event and skip crits), then the 85-100 randomizer;
- fixed-point rounding: ``Battle.modify`` / ``Battle.chainModify`` 4096-based math and
  ``Pokemon.calculateStat``'s boost table.

Validation is two-sided: unit fixtures in ``tests/test_gen3_damage.py`` whose expected
values are cross-checked against the live vendored sim (one-turn fixture battles), and
the Tier-2 gate harness (``scripts/tier2_gate.py``), whose calibration arm asserts that
observed damage on clean strikes in full controlled games lands exactly on one of the 16
predicted rolls.

Everything here is pure and dependency-free (no dex, no engine): callers supply stats,
base power, and the public-modifier set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence


def _tr(value: float) -> int:
    """JS ``Math.trunc`` (Showdown's ``battle.trunc`` for gen 3+): truncate toward zero."""
    return int(value)


def modify(value: int, numerator: float, denominator: float = 1) -> int:
    """``Battle.modify``: 4096-based fixed-point multiply with round-half-down."""
    modifier = _tr(numerator * 4096 / denominator)
    return _tr((_tr(value * modifier) + 2048 - 1) / 4096)


def chain_modifier(mods: Sequence[tuple[float, float]]) -> float:
    """Accumulate ``Battle.chainModify`` calls into one event modifier.

    Mirrors the engine exactly: each step truncates to 4096ths and the running product
    rounds at 4096 scale, so ``apply_chain`` reproduces the engine's per-event rounding.
    """
    modifier = 1.0
    for numerator, denominator in mods:
        previous = _tr(modifier * 4096)
        next_mod = _tr(numerator * 4096 / denominator)
        modifier = ((previous * next_mod + 2048) >> 12) / 4096
    return modifier


def apply_chain(value: int, mods: Sequence[tuple[float, float]]) -> int:
    """runEvent + finalModify: chain the modifiers, then ``modify`` the value once."""
    if not mods:
        return value
    return modify(value, chain_modifier(mods))


_BOOST_TABLE = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def boosted_stat(stat: int, boost: int) -> int:
    """``Pokemon.calculateStat``: apply a stat stage to a stored stat (gen 3 table)."""
    boost = max(-6, min(6, int(boost)))
    if boost >= 0:
        return int(stat * _BOOST_TABLE[boost])
    return int(stat / _BOOST_TABLE[-boost])


# --- Stored-stat computation (gen 3 formula; randbats sets are always neutral-nature). ---


def gen3_stat(base: int, iv: int, ev: int, level: int) -> int:
    """Non-HP stored stat: ``floor(floor(2*base + iv + floor(ev/4)) * level / 100 + 5)``."""
    return int(int(2 * base + iv + int(ev / 4)) * level / 100 + 5)


def gen3_hp_stat(base: int, iv: int, ev: int, level: int) -> int:
    """HP stat: ``floor(floor(2*base + iv + floor(ev/4) + 100) * level / 100 + 10)``."""
    return int(int(2 * base + iv + int(ev / 4) + 100) * level / 100 + 10)


# Gen 3+ Hidden Power IV overrides, transcribed from the vendored
# ``data/typechart.ts`` ``HPivs`` tables (unlisted stats stay 31). The randbats
# generator applies these whenever the set carries a Hidden Power move.
HIDDEN_POWER_IVS: Mapping[str, Mapping[str, int]] = {
    "bug": {"atk": 30, "def": 30, "spd": 30},
    "dark": {},
    "dragon": {"atk": 30},
    "electric": {"spa": 30},
    "fighting": {"def": 30, "spa": 30, "spd": 30, "spe": 30},
    "fire": {"atk": 30, "spa": 30, "spe": 30},
    "flying": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30},
    "ghost": {"def": 30, "spd": 30},
    "grass": {"atk": 30, "spa": 30},
    "ground": {"spa": 30, "spd": 30},
    "ice": {"atk": 30, "def": 30},
    "poison": {"def": 30, "spa": 30, "spd": 30},
    "psychic": {"atk": 30, "spe": 30},
    "rock": {"def": 30, "spd": 30, "spe": 30},
    "steel": {"spd": 30},
    "water": {"atk": 30, "def": 30, "spa": 30},
}

_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
_PINCH_BERRIES = frozenset({"salacberry", "petayaberry", "liechiberry"})


def _normalize(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def hidden_power_type(moves: Sequence[str]) -> Optional[str]:
    """The Hidden Power type carried by a normalized move list, or None."""
    for move in moves:
        normalized = _normalize(move)
        if normalized.startswith("hiddenpower") and len(normalized) > len("hiddenpower"):
            return normalized[len("hiddenpower"):]
    return None


def randbats_spread_stats(
    base_stats: Mapping[str, int],
    *,
    level: int,
    moves: Sequence[str],
    item: Optional[str],
    has_physical_attack: bool,
) -> dict[str, int]:
    """Actual stats for a gen3 randbats set, replicating the generator's spread logic.

    Mirrors ``data/random-battles/gen3/teams.ts`` exactly (corrections layer item 1):
    85 EVs / 31 IVs / neutral nature everywhere, THEN

    - Hidden Power IV overrides for the carried HP type;
    - the first HP-trim loop (Sub+Flail/Reversal want ``hp % 4 > 0``; Sub+pinch-berry
      want ``hp % 4 == 0``; Belly Drum wants ``hp % 2 > 0``);
    - Atk zeroing on no-physical-attack, non-Transform sets (EV 0; IV 0, or IV-28 when
      a Hidden Power override pinned the Atk IV);
    - the second HP-trim pass (Sub+Endeavor/Flail/Reversal drop one more step when
      ``hp % 4 == 0``; Sub+pinch loops until ``hp % 4 == 0``).

    ``has_physical_attack`` is the generator's ``counter.get('Physical')`` truthiness —
    the caller derives it from move categories (fixed-damage callback moves such as
    Counter/Seismic Toss do NOT count, matching the generator's ``queryMoves``).
    """
    evs = {stat: 85 for stat in _STAT_KEYS}
    ivs = {stat: 31 for stat in _STAT_KEYS}
    normalized_moves = {_normalize(move) for move in moves}
    hp_type = hidden_power_type(tuple(normalized_moves))
    if hp_type is not None:
        for stat, value in HIDDEN_POWER_IVS.get(hp_type, {}).items():
            ivs[stat] = value

    def hp_value() -> int:
        return gen3_hp_stat(int(base_stats.get("hp", 0)), ivs["hp"], evs["hp"], level)

    item_id = _normalize(item or "")
    has_substitute = "substitute" in normalized_moves
    flail_reversal = bool(normalized_moves & {"flail", "reversal"})
    pinch_item = item_id in _PINCH_BERRIES

    # First HP-trim loop (teams.ts "Prepare optimal HP").
    while evs["hp"] > 1:
        hp = hp_value()
        if has_substitute and flail_reversal:
            if hp % 4 > 0:
                break
        elif has_substitute and pinch_item:
            if hp % 4 == 0:
                break
        elif "bellydrum" in normalized_moves:
            if hp % 2 > 0:
                break
        else:
            break
        evs["hp"] -= 4

    # Minimize confusion damage: no physical attacks and no Transform -> zero Atk.
    if not has_physical_attack and "transform" not in normalized_moves:
        evs["atk"] = 0
        ivs["atk"] = (ivs["atk"] or 31) - 28 if hp_type is not None else 0

    # Second HP-trim pass.
    hp = hp_value()
    if has_substitute and bool(normalized_moves & {"endeavor", "flail", "reversal"}):
        if hp % 4 == 0:
            evs["hp"] -= 4
    elif has_substitute and pinch_item:
        while hp % 4 > 0:
            evs["hp"] -= 4
            hp = hp_value()

    stats = {
        stat: gen3_stat(int(base_stats.get(stat, 0)), ivs[stat], evs[stat], level)
        for stat in _STAT_KEYS
        if stat != "hp"
    }
    # Shedinja (the only base-1-HP species): the engine pins max HP to 1.
    stats["hp"] = 1 if int(base_stats.get("hp", 0)) == 1 else hp_value()
    return stats


# --- The damage chain itself. ---

ROLL_NUMERATORS = tuple(range(85, 101))  # (100 - random(16)) in protocol order


@dataclass(frozen=True)
class Gen3DamageContext:
    """Inputs to one gen3 damage computation, all public or candidate-conditioned.

    ``attack``/``defense`` are STORED stats (spread stats, no stages). Stat-modify
    events (``attack_mods``/``defense_mods``) are (numerator, denominator) chain
    entries in engine order — Choice Band (1.5, 1), Guts (1.5, 1), Huge/Pure Power
    (2, 1), pinch abilities (1.5, 1), Flash Fire volatile (1.5, 1), Thick Fat on the
    attacker's stat (0.5, 1), gen3 type-boost items (1.1, 1), Marvel Scale defense
    (1.5, 1), Soul Dew SpA/SpD (1.5, 1), Thick Club / Light Ball (2, 1).

    ``base_power_mods`` cover BasePower-event conditioning: Solar Beam in
    rain/sand/hail (0.5, 1) and Facade with a status (2, 1). ``weather_mod`` is the
    WeatherModifyDamage entry (Fire/Water in sun/rain: (1.5, 1) or (0.5, 1)).

    ``burned`` must already encode the Guts exemption (burn halving is skipped when
    the attacker's ability is Guts); ``screen`` is the category-matching screen on
    the defender's side (crit bypass is handled here, per the vendored base-data
    Reflect/Light Screen handlers).
    """

    level: int
    base_power: int
    category: str  # "Physical" | "Special"
    attack: int
    defense: int
    attack_boost: int = 0
    defense_boost: int = 0
    attack_mods: tuple[tuple[float, float], ...] = ()
    defense_mods: tuple[tuple[float, float], ...] = ()
    base_power_mods: tuple[tuple[float, float], ...] = ()
    stab: bool = False
    effectiveness: float = 1.0  # product of 2x / 0.5x steps; 0 means immune
    burned: bool = False
    screen: bool = False
    weather_mod: tuple[float, float] | None = None
    crit: bool = False
    explosion_def_halving: bool = False
    n_hits: int = 1


def gen3_damage_rolls(context: Gen3DamageContext) -> tuple[int, ...]:
    """The 16 possible per-hit damage values (roll 85..100), engine-exact.

    Returns an empty tuple when the move can deal no damage (immune, or zero base
    power). Values are single-hit; multi-hit callers roll each hit independently
    (``n_hits`` is carried for callers, not applied here).
    """
    if context.effectiveness <= 0 or context.base_power <= 0:
        return ()
    base_power = max(1, apply_chain(int(context.base_power), context.base_power_mods))

    attack_boost = context.attack_boost
    defense_boost = context.defense_boost
    if context.crit:
        # Gen 3 crits ignore harmful attack stages and helpful defense stages.
        if attack_boost < 0:
            attack_boost = 0
        if defense_boost > 0:
            defense_boost = 0
    attack = boosted_stat(context.attack, attack_boost)
    attack = apply_chain(attack, context.attack_mods)
    defense = boosted_stat(context.defense, defense_boost)
    defense = apply_chain(defense, context.defense_mods)
    if context.explosion_def_halving:
        defense = max(1, _tr(defense / 2))

    base = _tr(_tr(_tr(_tr(2 * context.level / 5 + 2) * base_power * attack) / defense) / 50)

    # data/mods/gen3/scripts.ts modifyDamage, in order.
    if context.burned:
        base = modify(base, 0.5)
    # (ModifyDamagePhase1: no handler in the gen3 randbats universe.)
    if context.weather_mod is not None:
        base = modify(base, *context.weather_mod)
    if context.category == "Physical" and base == 0:
        base = 1
    base += 2
    if context.crit:
        base = modify(base, 2)
    # (ModifyDamagePhase2: no handler in the gen3 randbats universe; floor is a no-op
    # on ints.)
    if context.stab:
        base = modify(base, 1.5)
    effectiveness = context.effectiveness
    while effectiveness >= 2:
        base *= 2
        effectiveness /= 2
    while effectiveness <= 0.5:
        base = _tr(base / 2)
        effectiveness *= 2
    if context.screen and not context.crit:
        # Base-data Reflect / Light Screen hook the final ModifyDamage event.
        base = modify(base, 0.5)

    rolls = []
    for numerator in ROLL_NUMERATORS:
        rolled = _tr(_tr(base * numerator) / 100)
        rolls.append(rolled if rolled > 0 else 1)
    return tuple(rolls)


def median_damage(rolls: Sequence[int]) -> float:
    """Median of the 16 rolls (mean of the two central order statistics)."""
    if not rolls:
        return 0.0
    ordered = sorted(rolls)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass(frozen=True)
class DamageSummary:
    rolls: tuple[int, ...] = field(default_factory=tuple)

    @property
    def min(self) -> int:
        return min(self.rolls) if self.rolls else 0

    @property
    def max(self) -> int:
        return max(self.rolls) if self.rolls else 0

    @property
    def median(self) -> float:
        return median_damage(self.rolls)


def summarize_damage(context: Gen3DamageContext) -> DamageSummary:
    return DamageSummary(rolls=gen3_damage_rolls(context))
