"""Fail-closed complete represented-state correspondence, not fidelity evidence.

Native Gen3 source layout: poke-engine 0.0.47 plus the bound PokeZero stack.
All 12 Pokemon are constructed explicitly. Raw states remain attached to every
result. A refusal is a panel outcome, not permission to replace that root.

The native wrapper recomputes conditional mechanics on EVERY Into<State> and
does not expose/serialize damage_dealt. Completed ordinary boundaries have no
pending cross-action carry; suspended boundaries require a separate interface.
Natural sleep/confusion/Encore elapsed-hazard counters are not a bijection to
JAX sampled remaining durations. This module names that gap; it does not erase
the latent duration or declare either representation structurally impossible.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import hashlib
import json
import math
import operator
import struct
from typing import Any


# Every field must have an explicit policy, including external chance addresses.
FIELD_POLICY = {
    **{name: "native party field" for name in (
        "species", "level", "types", "stats", "hp", "max_hp", "moves", "pp",
        "disabled", "item", "ability")},
    "hidden_power_type": "native Hidden Power variant, not metadata alias",
    "hidden_power_power": "native fixed70; refuse non70 when referenced",
    "happiness": "native Return102/Frustration1; refuse unequal effective power",
    "status": "native major status; sleep origin/duration requires explicit refusal",
    "status_counter": "active Toxic count; natural sleep latent convention unresolved",
    "sleep_skipped": "Rest provenance/refund unresolved when sleeping; otherwise stale",
    "choice_move": "Choice Band disabled-slot representation; ambiguous locks refuse",
    "last_move": "native selected slot only when conditional tracking is enabled",
    "encore_move": "elapsed-hazard versus sampled duration unresolved while active",
    "encore_counter": "elapsed-hazard versus sampled duration unresolved while active",
    "charge_move": "single Solar Beam charge volatile only",
    "charge_counter": "one remaining Solar Beam release; other nonzero values refuse",
    "boosts": "native active side boosts; nonzero bench state refuses",
    "volatile": "explicit supported bitset; unknown bits refuse",
    "confusion_counter": "elapsed-hazard versus sampled duration unresolved while active",
    "protect_counter": "native active side Protect streak; nonzero bench state refuses",
    "substitute_hp": "native active substitute health and volatile",
    "active": "native party index",
    "entry_pending": "unresolved initial events refuse; never silently initialize",
    "active_turns": "completed ordinary boundary age, next action increments before use",
    "trapped": "native force_trapped",
    "trap_source": "active opposing source association; dangling/bench links refuse",
    "forced_switch": "derived request; native force_switch specifically means deferred residual",
    "baton_pass": "live suspended action boundary refuses with raw state retained",
    "queued_actor": "live suspended action boundary refuses with raw state retained",
    "queued_actions": "future-reused while suspended; otherwise stale inactive values",
    "queued_stall": "future-reused while suspended; otherwise stale inactive values",
    "pending_residual": "future-reused conditional chance/queue interface unresolved",
    "residual_draws": "future-reused while suspended; otherwise consumed inactive draws",
    "phazing_draws": "future-reused while suspended; otherwise consumed inactive draws",
    "secondary_draws": "future-reused while suspended; otherwise consumed inactive draws",
    "sleep_talk_draws": "future-reused while suspended; otherwise consumed inactive draws",
    "replacement_count": "future tie address while suspended; otherwise inactive address",
    "spikes": "native side condition",
    "wish_counter": "nonzero Wish needs missing caster/recipient heal-amount correspondence",
    "weather": "native field weather",
    "weather_counter": "native field weather duration",
    "turn": "external ordinary-turn address; startup must already be resolved",
    "keys": "external PRNG address; exact-law driver supplies chance, not key enumeration",
    "done": "derived from all12 party HP; inconsistency refuses",
    "winner": "derived from all12 party HP; inconsistency refuses",
    "truant_loaf": "native active TRUANT residual clock; bench clock overwritten on entry",
}
NATIVE_PATHS = {
    "species": ("side[].pokemon[].id",), "level": ("side[].pokemon[].level",),
    "hidden_power_type": ("side[].pokemon[].moves[].id: HiddenPower variant",),
    "hidden_power_power": ("MOVES[HiddenPower*].base_power=70",),
    "happiness": ("MOVES[Return|Frustration].base_power",),
    "types": ("side[].pokemon[].types",),
    "stats": ("side[].pokemon[].maxhp", "side[].pokemon[].{attack,defense,special_attack,special_defense,speed}"),
    "hp": ("side[].pokemon[].hp",), "max_hp": ("side[].pokemon[].maxhp",),
    "status": ("side[].pokemon[].status",),
    "status_counter": ("side[].side_conditions.toxic_count", "UNRESOLVED: rest_turns/sleep_turns"),
    "sleep_skipped": ("UNRESOLVED: side[].pokemon[].rest_sleep_pending_refund",),
    "moves": ("side[].pokemon[].moves[].id",), "pp": ("side[].pokemon[].moves[].pp",),
    "disabled": ("side[].pokemon[].moves[].disabled",),
    "choice_move": ("side[].pokemon[].moves[].disabled: Choice Band lock mask",),
    "last_move": ("side[].last_used_move", "State.use_last_used_move: recomputed from all12 inventories"),
    "encore_move": ("UNRESOLVED: side[].last_used_move + ENCORE",),
    "encore_counter": ("UNRESOLVED: side[].volatile_status_durations.encore",),
    "charge_move": ("side[].volatile_statuses.SOLARBEAM",),
    "charge_counter": ("side[].volatile_statuses.SOLARBEAM: one-release boundary",),
    "item": ("side[].pokemon[].item",), "ability": ("side[].pokemon[].ability",),
    "boosts": ("side[].{attack,defense,special_attack,special_defense,speed,accuracy,evasion}_boost",),
    "volatile": ("side[].volatile_statuses: explicitly enumerated bits",),
    "confusion_counter": ("UNRESOLVED: side[].volatile_status_durations.confusion",),
    "protect_counter": ("side[].side_conditions.protect",),
    "substitute_hp": ("side[].substitute_health", "side[].volatile_statuses.SUBSTITUTE"),
    "active": ("side[].active_index",), "entry_pending": ("UNRESOLVED: pending initial event queue",),
    "active_turns": ("NO_NATIVE_FIELD: inactive age at admitted complete ordinary boundary",),
    "trapped": ("side[].force_trapped",),
    "trap_source": ("side[].volatile_statuses.TRAPPED", "opposing side.active_index"),
    "forced_switch": ("derived active.hp<=0 + live reserve; NOT native force_switch",),
    "baton_pass": ("UNRESOLVED: side[].baton_passing + suspended turn",),
    "queued_actor": ("UNRESOLVED: side[].switch_out_move_second_saved_move + current actor",),
    "queued_actions": ("UNRESOLVED while suspended; otherwise stale",),
    "queued_stall": ("UNRESOLVED while suspended; otherwise stale",),
    "pending_residual": ("UNRESOLVED: side[].force_switch + suspended action/chance conditioning",),
    **{name: ("EXTERNAL_CHANCE: future-reused while suspended; otherwise consumed inactive draw",)
       for name in ("residual_draws", "phazing_draws", "secondary_draws", "sleep_talk_draws")},
    "replacement_count": ("EXTERNAL_CHANCE: conditional tie-stream address",),
    "spikes": ("side[].side_conditions.spikes",),
    "wish_counter": ("UNRESOLVED: side[].wish=(counter,stored_heal_amount)",),
    "weather": ("State.weather.weather_type",), "weather_counter": ("State.weather.turns_remaining",),
    "turn": ("EXTERNAL_CHANCE: turn address; startup already resolved",),
    "keys": ("EXTERNAL_CHANCE: PRNG address, not exact semantic outcome state",),
    "done": ("derived from complete party HP",), "winner": ("derived from complete party HP",),
    "truant_loaf": ("side[].volatile_statuses.TRUANT",),
}
SIDE_NAMES = ("side_one", "side_two")
BOOST_NAMES = ("attack", "defense", "special_attack", "special_defense", "speed", "accuracy", "evasion")
STATUS_NAMES = ("none", "paralyze", "burn", "poison", "toxic", "sleep", "freeze")
WEATHER_NAMES = ("none", "rain", "sun", "sand", "hail")
CONDITIONS = ("aurora_veil", "crafty_shield", "healing_wish", "light_screen", "lucky_chant",
    "lunar_dance", "mat_block", "mist", "protect", "quick_guard", "reflect", "safeguard",
    "spikes", "stealth_rock", "sticky_web", "tailwind", "toxic_count", "toxic_spikes", "wide_guard")
DURATIONS = ("confusion", "encore", "lockedmove", "slowstart", "taunt", "yawn")
VOLATILES = {1: "flinch", 2: "flashfire", 4: "leechseed", 8: "ingrain"}
MON_FIELDS = ("id", "level", "type0", "type1", "base_type0", "base_type1", "hp", "maxhp",
    "ability", "base_ability", "item", "nature", "evs", "attack", "defense", "special_attack",
    "special_defense", "speed", "status", "rest_turns", "sleep_turns", "weight_kg")
INT_MON = frozenset(("level", "hp", "maxhp", "attack", "defense", "special_attack",
    "special_defense", "speed", "rest_turns", "sleep_turns"))
CONDITIONAL_LAST = frozenset(("encore", "fakeout"))
CONDITIONAL_DAMAGE = frozenset(("counter", "mirrorcoat", "focuspunch"))
DYNAMIC_INVENTORY = frozenset(("transform", "mimic", "sketch", "metronome", "assist", "mirrormove", "naturepower"))
SHAPES = {name: (2, 6) for name in ("species", "level", "hidden_power_type", "hidden_power_power",
    "happiness", "hp", "max_hp", "status", "status_counter", "sleep_skipped", "choice_move", "last_move",
    "encore_move", "encore_counter", "charge_move", "charge_counter", "item", "ability", "volatile",
    "confusion_counter", "protect_counter", "substitute_hp", "trap_source", "truant_loaf")}
SHAPES.update({name: (2,) for name in ("active", "entry_pending", "active_turns", "trapped", "forced_switch",
    "baton_pass", "queued_actions", "queued_stall", "phazing_draws", "secondary_draws", "sleep_talk_draws",
    "spikes", "wish_counter", "keys")})
SHAPES.update({name: () for name in ("queued_actor", "pending_residual", "replacement_count", "weather",
    "weather_counter", "turn", "done", "winner")})
SHAPES.update(types=(2, 6, 2), stats=(2, 6, 6), moves=(2, 6, 4), pp=(2, 6, 4), disabled=(2, 6, 4),
    boosts=(2, 6, 7), residual_draws=(2, 12))


@dataclass(frozen=True)
class Gap:
    field: str
    code: str
    detail: str


@dataclass(frozen=True)
class Refused:
    gaps: tuple[Gap, ...]
    raw_jax: tuple = ()
    raw_native: str | None = None
    status: str = "REFUSED"


@dataclass(frozen=True)
class Certificate:
    field_policies: tuple[tuple[str, str], ...]
    field_native_paths: tuple[tuple[str, tuple[str, ...]], ...]
    input_fields: tuple
    static_native: tuple
    metadata_sha256: str
    conditional_last: bool
    conditional_damage: bool
    inactive_addresses: tuple
    native_layout: str = "poke-engine0.0.47.gen3.pokezero-state-wire.v1"


@dataclass(frozen=True)
class NativeInput:
    state: Any
    certificate: Certificate
    raw_native: str


@dataclass(frozen=True)
class CanonicalFuture:
    key: tuple[tuple[str, Any], ...]
    certificate: Certificate
    raw_jax: tuple = ()
    raw_native: str | None = None


def _freeze(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return tuple((str(k), _freeze(v)) for k, v in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (bool, str, float)) or value is None:
        return value
    return operator.index(value)


def _sample(state, index):
    if index is not None and (type(index) is not int or index < 0):
        raise ValueError("explicit nonnegative batch index required")
    values = state if isinstance(state, dict) else state._asdict()
    return {name: _freeze(value if index is None else value[index]) for name, value in values.items()}


def _all(values):
    if isinstance(values, tuple):
        return all(_all(v) for v in values)
    return not values


def _shape(value):
    if not isinstance(value, tuple):
        if not isinstance(value, (int, bool)):
            raise ValueError("JAX state fields must be integral/boolean arrays")
        return ()
    children = {_shape(v) for v in value}
    if len(children) != 1:
        raise ValueError("ragged/empty state array")
    return (len(value), *children.pop())


def _f32(value):
    value = struct.unpack("f", struct.pack("f", float(value)))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite native f32")
    return value


def _metadata_sha(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _flat(value, prefix=""):
    if isinstance(value, dict):
        for k, v in sorted(value.items()):
            yield from _flat(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            yield from _flat(v, f"{prefix}[{i}]")
    else:
        yield prefix, value


def _vocab(metadata, kind, value):
    names = metadata["vocabulary"][kind]
    if type(value) is not int or not 0 <= value < len(names):
        raise ValueError(f"invalid {kind} vocabulary id {value!r}")
    return "none" if names[value] == "_none" else names[value]


def _move(metadata, data, side, slot, move):
    name = _vocab(metadata, "moves", move)
    if name.startswith("hiddenpower"):
        typ = _vocab(metadata, "types", data["hidden_power_type"][side][slot])
        name = "hiddenpower" + ("dark" if typ == "none" else typ)
    return name


def _prepare(data, metadata, certificate=None):
    """Make complete expected native fields, or enumerate named mapping gaps."""
    gaps = []
    add = lambda field, code, detail: gaps.append(Gap(field, code, detail))
    expected = set(FIELD_POLICY) - {"truant_loaf"}  # supported on either side of clock PR landing
    if set(data) - set(FIELD_POLICY) or expected - set(data):
        add("schema", "UNCLASSIFIED_JAX_FIELD", repr(sorted(set(data) ^ expected)))
        return None, gaps, (), False, False
    for name, value in data.items():
        if _shape(value) != SHAPES[name]:
            add(name, "INVALID_FIELD_SHAPE", f"Expected {SHAPES[name]}, received {_shape(value)}")
    if gaps:
        return None, gaps, (), False, False
    if not 0 <= data["weather"] < len(WEATHER_NAMES) or not 0 <= data["weather_counter"] <= 127:
        add("weather", "INVALID_WEATHER_CLOCK", "Unknown weather id or native i8 clock range")
        return None, gaps, (), False, False
    if any(age < 0 for age in data["active_turns"]):
        add("active_turns", "INVALID_ACTIVE_AGE", "Negative active age is not an ordinary boundary")
    if certificate is not None and _metadata_sha(metadata) != certificate.metadata_sha256:
        add("metadata", "METADATA_BINDING_CHANGED", "Vocabulary/static-field source differs from construction certificate")
    if data["pending_residual"] or any(data["baton_pass"]):
        add("pending_residual", "LIVE_SUSPENSION_INTERFACE", "Saved chance arrays and queued action are future-reused; native wrapper lacks the complete conditioning/carry interface")
    if any(data["entry_pending"]):
        add("entry_pending", "UNRESOLVED_ENTRY", "Initial entry events must be resolved by the bound caller, never silently by this mapper")
    inventory = {_move(metadata, data, side, p, move) for side in range(2) for p in range(6)
                 for move in data["moves"][side][p]}
    use_last = bool(inventory & CONDITIONAL_LAST)
    use_damage = bool(inventory & CONDITIONAL_DAMAGE)
    if inventory & DYNAMIC_INVENTORY:
        add("moves", "DYNAMIC_HISTORY_CONTEXT_UNRESOLVED", "Dynamic move acquisition/callers defeat the fixed-inventory inactive-history proof; do not erase lastMove/damage conditioning")
    if inventory & {"attract"} or any(_vocab(metadata, "abilities", a) == "cutecharm" for party in data["ability"] for a in party):
        add("species", "GENDER_NOT_RETAINED", "Accepted pool lacks per-Pokemon gender; Attract/Cute Charm can read it")
    if "lowkick" in inventory and "oracle_species" not in metadata:
        add("species", "WEIGHT_NOT_BOUND", "Low Kick needs source-bound species weights; do not substitute native default1kg")
    alive = [any(hp > 0 for hp in party) for party in data["hp"]]
    done = not all(alive)
    winner = -1 if not done else 0 if alive[0] else 1 if alive[1] else 2
    if bool(data["done"]) != done or data["winner"] != winner:
        add("done/winner", "INCONSISTENT_TERMINAL", "Terminal flags do not match complete party HP")
    result = {"weather": WEATHER_NAMES[data["weather"]], "weather_turns_remaining": data["weather_counter"],
        "terrain": "none", "terrain_turns_remaining": 0, "trick_room": False,
        "trick_room_turns_remaining": 0, "team_preview": False,
        "derived_done": done, "derived_winner": winner,
        "conditional_last": use_last, "conditional_damage": use_damage}
    static = dict(certificate.static_native) if certificate else {}
    for side, side_name in enumerate(SIDE_NAMES):
        active = data["active"][side]
        if type(active) is not int or not 0 <= active < 6:
            raise ValueError("invalid active index")
        mons = []
        for p in range(6):
            path = f"{side_name}.pokemon[{p}]"
            names = [_move(metadata, data, side, p, m) for m in data["moves"][side][p]]
            if any(m.startswith("hiddenpower") for m in names) and data["hidden_power_power"][side][p] not in (0, 70):
                add(path+".hidden_power_power", "NATIVE_POWER_CONVENTION", "Native Hidden Power variant has fixed70 power")
            happy = max(0, min(255, data["happiness"][side][p]))
            if ("return" in names and max(1, happy*10//25) != 102) or ("frustration" in names and max(1, (255-happy)*10//25) != 1):
                add(path+".happiness", "NATIVE_POWER_CONVENTION", "Native fixed Return/Frustration power differs from retained happiness")
            status = data["status"][side][p]
            if not 0 <= status < len(STATUS_NAMES):
                raise ValueError("unknown major status id")
            if status == 5:
                add(path+".status_counter", "SLEEP_ORIGIN_AND_LATENT_DURATION", "JAX lacks Rest origin; sampled remaining duration is not native elapsed-hazard sleep state")
            for name in ("confusion_counter", "encore_move", "encore_counter"):
                if data[name][side][p]:
                    add(path+"."+name, "LATENT_DURATION_CONVENTION", "Sampled remaining duration is not an elapsed-hazard/pending-expiry state")
            if p != active:
                for name in ("boosts", "volatile", "protect_counter", "substitute_hp", "choice_move", "last_move", "charge_move", "charge_counter", "trap_source"):
                    value = data[name][side][p]
                    empty = value == -1 if name == "trap_source" else _all(value)
                    if not empty:
                        add(path+"."+name, "BENCH_STATE_NOT_REPRESENTED", "Native stores this condition only on the active side")
                if status == 4 and data["status_counter"][side][p]:
                    add(path+".status_counter", "BENCH_TOXIC_COUNT_NOT_REPRESENTED", "Native switches reset the side Toxic counter; a nonzero stored bench count cannot be silently erased")
            types = tuple("typeless" if (t := _vocab(metadata, "types", value)) == "none" else t for value in data["types"][side][p])
            stats = data["stats"][side][p]
            if stats[0] != data["max_hp"][side][p]:
                add(path+".stats[0]", "DISTINCT_HP_STAT", "Native exposes maxhp, not an additional distinct immutable HP stat")
            ability = _vocab(metadata, "abilities", data["ability"][side][p])
            extras = static.get(path, ("serious", (85,)*6, "none", 1.0, types, ability))
            if certificate is None and "oracle_species" in metadata:
                species = _vocab(metadata, "species", data["species"][side][p])
                extras = (*extras[:3], _f32(metadata["oracle_species"][species]["weight_kg"]), *extras[4:])
            static[path] = extras
            mon = dict(id=_vocab(metadata, "species", data["species"][side][p]), level=data["level"][side][p],
                types=types, base_types=extras[4], hp=data["hp"][side][p], maxhp=data["max_hp"][side][p],
                ability=ability, base_ability=extras[5], item=_vocab(metadata, "items", data["item"][side][p]),
                nature=extras[0], evs=extras[1], gender=extras[2], weight_kg=extras[3],
                attack=stats[1], defense=stats[2], special_attack=stats[3], special_defense=stats[4], speed=stats[5],
                status=STATUS_NAMES[status], rest_turns=0, sleep_turns=0, rest_sleep_pending_refund=0,
                pre_transform="", terastallized=False, tera_type="normal",
                moves=tuple(dict(id=name, disabled=bool(disabled), pp=pp) for name, disabled, pp in
                    zip(names, data["disabled"][side][p], data["pp"][side][p], strict=True)))
            if any(not -32768 <= mon[k] <= 32767 for k in ("hp", "maxhp", "attack", "defense", "special_attack", "special_defense", "speed")) or any(not 0 <= m["pp"] <= 127 for m in mon["moves"]):
                add(path, "NATIVE_INTEGER_RANGE", "State cannot be represented without narrowing native integers")
            if data["choice_move"][side][p]:
                if any(data["disabled"][side][p]) or mon["item"] != "choiceband":
                    add(path+".choice_move", "AMBIGUOUS_DISABLED_LOCK", "Native disabled slots cannot distinguish independent disable and Choice lock")
                lock = data["choice_move"][side][p]
                if sum(m == lock for m in data["moves"][side][p]) != 1:
                    add(path+".choice_move", "AMBIGUOUS_LOCK_SLOT", "Choice lock does not identify exactly one native slot")
                mon["moves"] = tuple(dict(m, disabled=m["disabled"] or original != lock)
                    for m, original in zip(mon["moves"], data["moves"][side][p], strict=True))
            mons.append(mon)
        bits = data["volatile"][side][active]
        if bits & ~sum(VOLATILES):
            add(side_name+".volatile", "UNMAPPED_VOLATILE_BITS", f"Unknown or live Focus Punch bits {bits & ~sum(VOLATILES)}")
        volatiles = {name for bit, name in VOLATILES.items() if bits & bit}
        if data["substitute_hp"][side][active] > 0:
            volatiles.add("substitute")
        if data.get("truant_loaf", ((False,)*6,)*2)[side][active] and mons[active]["ability"] == "truant":
            volatiles.add("truant")
        charge, ticks = data["charge_move"][side][active], data["charge_counter"][side][active]
        if charge or ticks:
            if _vocab(metadata, "moves", charge) != "solarbeam" or ticks != 1:
                add(side_name+".charge", "UNMAPPED_CHARGE_STATE", "Only the complete-boundary single Solar Beam release is mapped")
            else:
                volatiles.add("solarbeam")
        source = data["trap_source"][side][active]
        if source >= 0:
            if source != data["active"][1-side] or data["hp"][1-side][source] <= 0:
                add(side_name+".trap_source", "UNMAPPED_TRAP_SOURCE", "Move trap source must be the live opposing active")
            volatiles.add("trapped")
        if data["wish_counter"][side]:
            add(side_name+".wish_counter", "WISH_HEAL_AMOUNT_NOT_RETAINED", "JAX recipient-dependent healing cannot manufacture native stored heal amount/history")
        conditions = {k: 0 for k in CONDITIONS}
        conditions.update(spikes=data["spikes"][side], protect=data["protect_counter"][side][active],
            toxic_count=data["status_counter"][side][active] if data["status"][side][active] == 4 else 0)
        last = "move:none"
        if use_last:
            used = data["last_move"][side][active]
            if used and _vocab(metadata, "moves", used) == "struggle":
                last = "move:struggle"
            elif used:
                matches = [i for i, move in enumerate(data["moves"][side][active]) if move == used]
                if len(matches) != 1:
                    add(side_name+".last_move", "AMBIGUOUS_LAST_SLOT", "Native history needs a unique slot, not only move identity")
                else:
                    last = f"move:{matches[0]}"
            elif data["active_turns"][side] == 0:
                last = f"switch:{active}"
            else:
                add(side_name+".last_move", "LAST_ACTION_HISTORY_NOT_RETAINED", "No PP-spending move plus elapsed active turns cannot distinguish native Switch/None history")
        expected_request = data["hp"][side][active] <= 0 and alive[side]
        if bool(data["forced_switch"][side]) != expected_request and not data["pending_residual"]:
            add(side_name+".forced_switch", "INCONSISTENT_REQUEST", "Replacement request disagrees with active HP")
        result[side_name] = dict(pokemon=tuple(mons), active_index=str(active),
            side_conditions=conditions, volatile_status_durations={k: 0 for k in DURATIONS},
            volatile_statuses=tuple(sorted(volatiles)), substitute_health=data["substitute_hp"][side][active],
            **{name+"_boost": data["boosts"][side][active][i] for i, name in enumerate(BOOST_NAMES)},
            wish=(0, 0), future_sight=(0, "0"), force_switch=False,
            switch_out_move_second_saved_move="none", baton_passing=False, shed_tailing=False,
            force_trapped=bool(data["trapped"][side]), last_used_move=last, slow_uturn_move=False)
    return result, gaps, tuple(sorted(static.items())), use_last, use_damage


def native_input_from_jax(state, index, metadata, *, module=None):
    """Construct all12 native mons and a field certificate, or return REFUSED."""
    raw = ()
    try:
        data = _sample(state, index)
        raw = _freeze(data)
        expected, gaps, static, last, damage = _prepare(data, metadata)
    except (ValueError, KeyError, IndexError, TypeError) as error:
        return Refused((Gap("schema", "INVALID_INPUT", str(error)),), raw)
    if gaps:
        return Refused(tuple(gaps), raw)
    inactive = tuple((name, data[name]) for name in ("keys", "turn", "queued_actor", "queued_actions", "queued_stall",
        "residual_draws", "phazing_draws", "secondary_draws", "sleep_talk_draws", "replacement_count"))
    certificate = Certificate(tuple((k, FIELD_POLICY[k]) for k in sorted(data)),
        tuple((k, NATIVE_PATHS[k]) for k in sorted(data)), raw, static, _metadata_sha(metadata), last, damage, inactive)
    module = import_module("poke_engine") if module is None else module
    try:
        arguments = {k: v for k, v in expected.items() if k not in (*SIDE_NAMES, "derived_done", "derived_winner", "conditional_last", "conditional_damage")}
        for name in SIDE_NAMES:
            side = dict(expected[name])
            side["pokemon"] = [module.Pokemon(**dict(mon, moves=[module.Move(**move) for move in mon["moves"]])) for mon in side["pokemon"]]
            side["side_conditions"] = module.SideConditions(**side["side_conditions"])
            side["volatile_status_durations"] = module.VolatileStatusDurations(**side["volatile_status_durations"])
            side["volatile_statuses"] = set(side["volatile_statuses"])
            arguments[name] = module.Side(**side)
        native = module.State(**arguments)
        raw_native = native.to_string()
    except (ValueError, TypeError, AttributeError) as error:
        return Refused((Gap("native", "CONSTRUCTION_FAILED", str(error)),), raw)
    actual = canonical_native_future(native, certificate)
    if isinstance(actual, Refused):
        return actual
    if actual.key != tuple(_flat(expected)):
        return Refused((Gap("native", "CONSTRUCTION_NOT_LOSSLESS", repr(field_differences(tuple(_flat(expected)), actual.key))),), raw, raw_native)
    return NativeInput(native, certificate, raw_native)


def canonical_jax_future(state, index, metadata, *, certificate):
    """Immutable complete represented key; retain every raw JAX field separately."""
    raw = ()
    try:
        data = _sample(state, index)
        raw = _freeze(data)
        expected, gaps, _, last, damage = _prepare(data, metadata, certificate)
    except (ValueError, KeyError, IndexError, TypeError) as error:
        return Refused((Gap("schema", "INVALID_FUTURE", str(error)),), raw)
    if last != certificate.conditional_last or damage != certificate.conditional_damage:
        gaps.append(Gap("moves", "CONDITIONAL_MECHANICS_CHANGED", "Move inventory changed native tracking context; require a new correspondence certificate"))
    return Refused(tuple(gaps), raw) if gaps else CanonicalFuture(tuple(_flat(expected)), certificate, raw)


def _bool(value):
    if value not in ("true", "false"):
        raise ValueError("invalid native bool")
    return value == "true"


def _native_wire(raw):
    parts = raw.split("/")
    if len(parts) != 6:
        raise ValueError("unknown native State wire layout")
    result = {}
    for side_name, wire in zip(SIDE_NAMES, parts[:2], strict=True):
        fields = wire.split("=")
        if len(fields) != 29:
            raise ValueError("unknown native Side wire layout")
        mons = []
        for mon_wire in fields[:6]:
            m = mon_wire.split(",")
            if len(m) < 29:
                raise ValueError("incomplete native Pokemon wire layout")
            mon = {name: int(value) if name in INT_MON else _f32(value) if name == "weight_kg" else
                   tuple(map(int, value.split(";"))) if name == "evs" else value.lower()
                   for name, value in zip(MON_FIELDS, m[:22], strict=True)}
            mon["types"] = (mon.pop("type0"), mon.pop("type1"))
            mon["base_types"] = (mon.pop("base_type0"), mon.pop("base_type1"))
            mon["moves"] = tuple(dict(id=mv[0].lower(), disabled=_bool(mv[1]), pp=int(mv[2]))
                for value in m[22:26] if len(mv := value.split(";")) == 3)
            if len(mon["moves"]) != 4:
                raise ValueError("invalid native move layout")
            mon.update(terastallized=_bool(m[26]), tera_type=m[27].lower(), gender=m[28].lower(),
                pre_transform="", rest_sleep_pending_refund=0)
            for trailing in m[29:]:
                if trailing.startswith("refund:"):
                    mon["rest_sleep_pending_refund"] = int(trailing.removeprefix("refund:"))
                elif not mon["pre_transform"]:
                    mon["pre_transform"] = trailing
                else:
                    raise ValueError("unknown optional native Pokemon field")
            mons.append(mon)
        conditions = fields[7].split(";")
        durations = fields[9].split(";")
        if len(conditions) != len(CONDITIONS) or len(durations) != len(DURATIONS):
            raise ValueError("unknown native condition/duration layout")
        result[side_name] = dict(pokemon=tuple(mons), active_index=fields[6],
            side_conditions=dict(zip(CONDITIONS, map(int, conditions), strict=True)),
            volatile_status_durations=dict(zip(DURATIONS, map(int, durations), strict=True)),
            volatile_statuses=tuple(sorted(v.lower() for v in fields[8].split(":") if v)),
            substitute_health=int(fields[10]),
            **{name+"_boost": int(fields[11+i]) for i, name in enumerate(BOOST_NAMES)},
            wish=(int(fields[18]), int(fields[19])), future_sight=(int(fields[20]), fields[21]),
            force_switch=_bool(fields[22]), switch_out_move_second_saved_move=fields[23].lower(),
            baton_passing=_bool(fields[24]), shed_tailing=_bool(fields[25]), force_trapped=_bool(fields[26]),
            last_used_move=fields[27].lower(), slow_uturn_move=_bool(fields[28]))
    for index, name in ((2, "weather"), (3, "terrain")):
        field = parts[index].split(";")
        if len(field) != 2:
            raise ValueError("invalid native field clock")
        result[name] = field[0].lower()
        result[name+"_turns_remaining"] = int(field[1])
    trick = parts[4].split(";")
    if len(trick) != 2:
        raise ValueError("invalid native trick room")
    result.update(trick_room=_bool(trick[0]), trick_room_turns_remaining=int(trick[1]), team_preview=_bool(parts[5]))
    return result


def canonical_native_future(native_state, certificate):
    """Parse complete bound wire state; unknown live native-only fields refuse."""
    raw = native_state if isinstance(native_state, str) else native_state.to_string()
    gaps = []
    try:
        values = _native_wire(raw)
        inventory = {m["id"] for side in SIDE_NAMES for mon in values[side]["pokemon"] for m in mon["moves"]}
        use_last, use_damage = bool(inventory & CONDITIONAL_LAST), bool(inventory & CONDITIONAL_DAMAGE)
        if (use_last, use_damage) != (certificate.conditional_last, certificate.conditional_damage):
            gaps.append(Gap("moves", "CONDITIONAL_MECHANICS_CHANGED", "PyState.into recomputes tracking from all12 move inventories"))
        for name in SIDE_NAMES:
            side = values[name]
            allowed = set(VOLATILES.values()) | {"substitute", "truant", "solarbeam", "trapped"}
            unknown = set(side["volatile_statuses"]) - allowed
            if unknown:
                gaps.append(Gap(name+".volatile_statuses", "NATIVE_UNREPRESENTED_VOLATILE", repr(sorted(unknown))))
            for field, value in side["side_conditions"].items():
                if value and field not in {"spikes", "protect", "toxic_count"}:
                    gaps.append(Gap(name+".side_conditions."+field, "NATIVE_UNREPRESENTED_CONDITION", repr(value)))
            for field, value in side["volatile_status_durations"].items():
                if value:
                    gaps.append(Gap(name+".volatile_status_durations."+field, "LATENT_DURATION_CONVENTION", repr(value)))
            for field in ("wish", "future_sight", "force_switch", "baton_passing", "shed_tailing", "slow_uturn_move"):
                value = side[field]
                if value not in (False, (0, 0), (0, "0")):
                    gaps.append(Gap(name+"."+field, "NATIVE_LIVE_FUTURE_FIELD", repr(value)))
            if side["switch_out_move_second_saved_move"] != "none":
                gaps.append(Gap(name+".switch_out_move_second_saved_move", "LIVE_SUSPENSION_INTERFACE", "Saved action must not be erased"))
            for p, mon in enumerate(side["pokemon"]):
                for field in ("pre_transform", "rest_sleep_pending_refund", "rest_turns", "sleep_turns"):
                    if mon[field]:
                        gaps.append(Gap(f"{name}.pokemon[{p}].{field}", "NATIVE_LIVE_FUTURE_FIELD", repr(mon[field])))
                if mon["status"] == "sleep":
                    gaps.append(Gap(f"{name}.pokemon[{p}].status", "SLEEP_ORIGIN_AND_LATENT_DURATION", "Native Rest/elapsed sleep cannot be silently projected"))
                if mon["terastallized"]:
                    gaps.append(Gap(f"{name}.pokemon[{p}].terastallized", "NATIVE_UNREPRESENTED_FIELD", "No JAX Gen3 Tera state"))
        if values["terrain"] != "none" or values["terrain_turns_remaining"] or values["trick_room"] or values["trick_room_turns_remaining"] or values["team_preview"]:
            gaps.append(Gap("native.field", "NATIVE_UNREPRESENTED_FIELD", "Terrain/trick-room/team-preview has no JAX correspondence"))
        alive = [any(mon["hp"] > 0 for mon in values[name]["pokemon"]) for name in SIDE_NAMES]
        done = not all(alive)
        values.update(derived_done=done, derived_winner=-1 if not done else 0 if alive[0] else 1 if alive[1] else 2,
            conditional_last=use_last, conditional_damage=use_damage)
    except (ValueError, IndexError, TypeError) as error:
        gaps.append(Gap("native.schema", "UNKNOWN_NATIVE_LAYOUT", str(error)))
    if gaps:
        return Refused(tuple(gaps), certificate.input_fields, raw)
    return CanonicalFuture(tuple(_flat(values)), certificate, raw_native=raw)


def field_differences(left, right):
    """Keep exact field paths and both values; no tolerances or HP-only fallback."""
    a, b = dict(left.key if isinstance(left, CanonicalFuture) else left), dict(right.key if isinstance(right, CanonicalFuture) else right)
    return tuple((path, a.get(path, "<MISSING>"), b.get(path, "<MISSING>"))
                 for path in sorted(a.keys() | b.keys()) if a.get(path, "<MISSING>") != b.get(path, "<MISSING>"))
