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
    "base_ability": "immutable original constructor ability, retained separately from current ability",
    "hidden_power_type": "native Hidden Power variant, not metadata alias",
    "hidden_power_power": "native60/70 variant; refuse other referenced powers",
    "happiness": "native Return102; Frustration catalog has no bound happiness parameter",
    "gender": "retained immutable -1unknown/0genderless/1male/2female; unknown permitted only under gender-insensitive inventory proof",
    "status": "native major status; sleep origin/duration requires explicit refusal",
    "status_counter": "active Toxic quotient min(count,14); natural sleep latent convention unresolved",
    "sleep_skipped": "Rest provenance/refund unresolved when sleeping; otherwise stale",
    "choice_move": "Choice Band disabled-slot representation; ambiguous locks refuse",
    "last_move": "inactive caller-identity observation; all Encore/history consumers use side-owned kind/index",
    "last_action_kind": "explicit side-owned NONE/SWITCH/MOVE/STRUGGLE; native tracking follows inventory flag",
    "last_action_index": "exact dispatch move slot or committed switch party index; -1 for NONE/STRUGGLE",
    "encore_move": "elapsed-hazard versus sampled duration unresolved while active",
    "encore_counter": "elapsed-hazard versus sampled duration unresolved while active",
    "charge_move": "single Solar Beam charge volatile only",
    "charge_counter": "one remaining Solar Beam release; other nonzero values refuse",
    "boosts": "native active side boosts; nonzero bench state refuses",
    "volatile": "explicit supported bitset; unknown bits refuse",
    "confusion_counter": "elapsed-hazard versus sampled duration unresolved while active",
    "protect_counter": "JAX denominator0/2/4/8 -> native capped exponent0/1/2/3; other live values refuse",
    "substitute_hp": "native active substitute health and volatile; absent volatile leaves only inert raw native storage",
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
    "wish_counter": "native side Wish counter, independent of retained amount and current recipient",
    "wish_amount": "native stored caster half-max-HP, retained through expiry and switches; healing uses recipient",
    "weather": "native field weather",
    "weather_counter": "native duration; non-NONE JAX0 permanent maps to native-1, positive clocks unchanged",
    "turn": "external ordinary-turn address; startup must already be resolved",
    "keys": "external PRNG address; exact-law driver supplies chance, not key enumeration",
    "done": "derived from all12 party HP; inconsistency refuses",
    "winner": "derived from all12 party HP; inconsistency refuses",
    "truant_loaf": "native active TRUANT residual clock; bench clock overwritten on entry",
}
NATIVE_PATHS = {
    "species": ("side[].pokemon[].id",), "level": ("side[].pokemon[].level",),
    "hidden_power_type": ("side[].pokemon[].moves[].id: HiddenPower variant",),
    "hidden_power_power": ("MOVES[HiddenPower*60|HiddenPower*70].base_power",),
    "happiness": ("MOVES[Return|Frustration].base_power",),
    "gender": ("side[].pokemon[].gender",),
    "types": ("side[].pokemon[].types",),
    "stats": ("side[].pokemon[].maxhp", "side[].pokemon[].{attack,defense,special_attack,special_defense,speed}"),
    "hp": ("side[].pokemon[].hp",), "max_hp": ("side[].pokemon[].maxhp",),
    "status": ("side[].pokemon[].status",),
    "status_counter": ("side[].side_conditions.toxic_count", "UNRESOLVED: rest_turns/sleep_turns"),
    "sleep_skipped": ("UNRESOLVED: side[].pokemon[].rest_sleep_pending_refund",),
    "moves": ("side[].pokemon[].moves[].id",), "pp": ("side[].pokemon[].moves[].pp",),
    "disabled": ("side[].pokemon[].moves[].disabled",),
    "choice_move": ("side[].pokemon[].moves[].disabled: Choice Band lock mask",),
    "last_move": ("INACTIVE: diagnostic caller identity; native history uses last_action_kind/index",),
    "last_action_kind": ("side[].last_used_move: variant",),
    "last_action_index": ("side[].last_used_move: exact move/party index",),
    "encore_move": ("UNRESOLVED: side[].last_used_move + ENCORE",),
    "encore_counter": ("UNRESOLVED: side[].volatile_status_durations.encore",),
    "charge_move": ("side[].volatile_statuses.SOLARBEAM",),
    "charge_counter": ("side[].volatile_statuses.SOLARBEAM: one-release boundary",),
    "item": ("side[].pokemon[].item",), "ability": ("side[].pokemon[].ability",),
    "base_ability": ("side[].pokemon[].base_ability",),
    "boosts": ("side[].{attack,defense,special_attack,special_defense,speed,accuracy,evasion}_boost",),
    "volatile": ("side[].volatile_statuses: explicitly enumerated bits",),
    "confusion_counter": ("UNRESOLVED: side[].volatile_status_durations.confusion",),
    "protect_counter": ("side[].side_conditions.protect: denominator-to-exponent, capped3",),
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
    "wish_counter": ("side[].wish[0]",),
    "wish_amount": ("side[].wish[1]",),
    "weather": ("State.weather.weather_type",), "weather_counter": ("State.weather.turns_remaining: active JAX0 -> native-1",),
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
    "happiness", "gender", "hp", "max_hp", "status", "status_counter", "sleep_skipped", "choice_move", "last_move",
    "encore_move", "encore_counter", "charge_move", "charge_counter", "item", "ability", "base_ability", "volatile",
    "confusion_counter", "protect_counter", "substitute_hp", "trap_source", "truant_loaf")}
SHAPES.update({name: (2,) for name in ("active", "entry_pending", "active_turns", "trapped", "forced_switch",
    "last_action_kind", "last_action_index",
    "baton_pass", "queued_actions", "queued_stall", "phazing_draws", "secondary_draws", "sleep_talk_draws",
    "spikes", "wish_counter", "wish_amount", "keys")})
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
    raw_actions: tuple | None = None


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
class NativeActions:
    choices: tuple[str, str]
    raw_actions: tuple[int, int]
    raw_jax: tuple
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
        power = data["hidden_power_power"][side][slot]
        name = "hiddenpower" + ("dark" if typ == "none" else typ) + str(power if power > 0 else 70)
    return name


def _prepare(data, metadata, certificate=None, *, boundary="ordinary",
             original_static=None, original_genders=None):
    """Make complete expected native fields, or enumerate named mapping gaps."""
    if boundary not in ("ordinary", "suspended"):
        raise ValueError("unknown correspondence boundary policy")
    gaps = []
    add = lambda field, code, detail: gaps.append(Gap(field, code, detail))
    if certificate is not None and (original_static is not None or original_genders is not None):
        raise ValueError("ambiguous original static provenance")
    if "base_ability" not in data:
        add("base_ability", "ORIGINAL_BASE_ABILITY_REQUIRED", "Do not infer an original ability from a possibly copied current ability")
        return None, gaps, (), False, False
    if not {"last_action_kind", "last_action_index"} <= data.keys():
        add("last_action", "LAST_ACTION_HISTORY_REQUIRED", "Retain explicit side-owned kind/index; never infer history from identity or active age")
        return None, gaps, (), False, False
    if "wish_amount" not in data:
        add("wish_amount", "WISH_HISTORY_REQUIRED", "Retain the actual stored Wish history, including after counter expiry; never infer it from current HP")
        return None, gaps, (), False, False
    if original_static is not None:
        required = {f"{side}.pokemon[{p}]" for side in SIDE_NAMES for p in range(6)}
        if set(dict(original_static)) != required or original_genders is None:
            add("source", "INCOMPLETE_ORIGINAL_STATIC", "All12 constructor records and retained genders required")
            return None, gaps, (), False, False
    expected = set(FIELD_POLICY) - {"truant_loaf", "gender"}  # legacy fields remain explicitly unknown
    if set(data) - set(FIELD_POLICY) or expected - set(data):
        add("schema", "UNCLASSIFIED_JAX_FIELD", repr(sorted(set(data) ^ expected)))
        return None, gaps, (), False, False
    for name, value in data.items():
        if _shape(value) != SHAPES[name]:
            add(name, "INVALID_FIELD_SHAPE", f"Expected {SHAPES[name]}, received {_shape(value)}")
    if gaps:
        return None, gaps, (), False, False
    for side, (kind, index) in enumerate(zip(data["last_action_kind"], data["last_action_index"])):
        valid = type(kind) is int and type(index) is int and (
            (kind in (0, 3) and index == -1) or (kind == 1 and 0 <= index < 6)
            or (kind == 2 and 0 <= index < 4))
        if not valid:
            add(f"side[{side}].last_action", "INVALID_LAST_ACTION_HISTORY", "Expected NONE/STRUGGLE(-1), SWITCH(0..5), or MOVE(0..3)")
    for side,(counter,amount) in enumerate(zip(data["wish_counter"],data["wish_amount"])):
        if (type(counter) is not int or type(amount) is not int
                or not 0 <= counter <= 127 or not 0 <= amount <= 32767):
            add(f"side[{side}].wish", "INVALID_WISH_HISTORY", "Expected nonnegative native i8 counter and i16 stored amount")
    if gaps:
        return None, gaps, (), False, False
    if not 0 <= data["weather"] < len(WEATHER_NAMES) or not 0 <= data["weather_counter"] <= 127:
        add("weather", "INVALID_WEATHER_CLOCK", "Unknown weather id or native i8 clock range")
        return None, gaps, (), False, False
    if any(age < 0 for age in data["active_turns"]):
        add("active_turns", "INVALID_ACTIVE_AGE", "Negative active age is not an ordinary boundary")
    if certificate is not None and _metadata_sha(metadata) != certificate.metadata_sha256:
        add("metadata", "METADATA_BINDING_CHANGED", "Vocabulary/static-field source differs from construction certificate")
    if boundary == "ordinary" and (data["pending_residual"] or any(data["baton_pass"])):
        add("pending_residual", "LIVE_SUSPENSION_INTERFACE", "Saved chance arrays and queued action are future-reused; native wrapper lacks the complete conditioning/carry interface")
    if boundary == "suspended":
        direct = (sum(data["baton_pass"]) == 1 and data["forced_switch"] == data["baton_pass"]
            and data["replacement_count"] == 0)
        chain = (not any(data["baton_pass"]) and sum(data["forced_switch"]) == 1
            and 1 <= data["replacement_count"] <= 5 and data["queued_actor"] == -1
            and all(not forced or data["hp"][side][data["active"][side]] == 0
                for side, forced in enumerate(data["forced_switch"])))
        if not data["pending_residual"] or not (direct or chain):
            add("pending_residual", "UNSUPPORTED_SUSPENSION_CAUSE", "Direct Baton or its counted hazard-KO replacement chain required")
        if certificate is None:
            add("metadata", "SUSPENSION_PROVENANCE_REQUIRED", "The original construction certificate retains native-only static fields")
    if any(data["entry_pending"]):
        add("entry_pending", "UNRESOLVED_ENTRY", "Initial entry events must be resolved by the bound caller, never silently by this mapper")
    inventory = {_move(metadata, data, side, p, move) for side in range(2) for p in range(6)
                 for move in data["moves"][side][p]}
    use_last = bool(inventory & CONDITIONAL_LAST)
    use_damage = bool(inventory & CONDITIONAL_DAMAGE)
    if boundary == "suspended" and (use_last or use_damage):
        add("moves", "SUSPENDED_HISTORY_UNSUPPORTED", "Ordinary-boundary history inactivity is not a suspended-turn proof")
    if boundary == "suspended" and (data["queued_actor"] not in (-1, 0, 1)
            or (data["queued_actor"] >= 0 and data["baton_pass"][data["queued_actor"]])
            or (data["queued_actor"] == -1 and data["queued_actions"] != (0, 0))):
        add("queued_actor", "UNSUPPORTED_SUSPENSION_QUEUE", "Queued actor must be the waiter, or an explicitly empty queue")
    if inventory & DYNAMIC_INVENTORY:
        add("moves", "DYNAMIC_HISTORY_CONTEXT_UNRESOLVED", "Dynamic move acquisition/callers defeat the fixed-inventory inactive-history proof; do not erase lastMove/damage conditioning")
    genders = data.get("gender", ((-1,)*6,)*2)
    if original_genders is not None and genders != original_genders:
        add("gender", "RETAINED_GENDER_CHANGED", "Gender differs from original constructor provenance")
    if certificate is not None and genders != dict(certificate.input_fields).get("gender", ((-1,)*6,)*2):
        add("gender", "RETAINED_GENDER_CHANGED", "Unknown and genderless are distinct immutable constructor values")
    if any(type(g) is not int or g not in (-1, 0, 1, 2) for party in genders for g in party):
        add("gender", "INVALID_RETAINED_GENDER", "Expected immutable codes-1/0/1/2")
    retained_static = certificate.static_native if certificate is not None else original_static or ()
    base_cute = any(extras[5] == "cutecharm" for _, extras in retained_static)
    sensitive = bool(inventory & {"attract"}) or base_cute or any(
        _vocab(metadata, "abilities", a) == "cutecharm" for party in data["ability"] for a in party)
    if sensitive and any(g == -1 for party in genders for g in party):
        add("gender", "GENDER_NOT_RETAINED", "Attract/Cute Charm inventory can read any retained party gender; unknown is not genderless")
    if "lowkick" in inventory and "oracle_species" not in metadata:
        add("species", "WEIGHT_NOT_BOUND", "Low Kick needs source-bound species weights; do not substitute native default1kg")
    alive = [any(hp > 0 for hp in party) for party in data["hp"]]
    done = not all(alive)
    winner = -1 if not done else 0 if alive[0] else 1 if alive[1] else 2
    if bool(data["done"]) != done or data["winner"] != winner:
        add("done/winner", "INCONSISTENT_TERMINAL", "Terminal flags do not match complete party HP")
    # JAX's non-NONE zero clock is permanent. Native Gen3 weather abilities
    # write WEATHER_ABILITY_TURNS=-1; use that convention at construction AND
    # in the JAX future key, not a quotient over arbitrary native clocks.
    # Raw JAX0 remains retained in the certificate/representative; native wire
    # clocks remain unmodified, including malformed/different-clock evidence.
    native_weather_clock = -1 if data["weather"] and data["weather_counter"] == 0 else data["weather_counter"]
    result = {"weather": WEATHER_NAMES[data["weather"]], "weather_turns_remaining": native_weather_clock,
        "terrain": "none", "terrain_turns_remaining": 0, "trick_room": False,
        "trick_room_turns_remaining": 0, "team_preview": False,
        "derived_done": done, "derived_winner": winner,
        "conditional_last": use_last, "conditional_damage": use_damage}
    static = dict(retained_static)
    for side, side_name in enumerate(SIDE_NAMES):
        active = data["active"][side]
        if type(active) is not int or not 0 <= active < 6:
            raise ValueError("invalid active index")
        mons = []
        for p in range(6):
            path = f"{side_name}.pokemon[{p}]"
            names = [_move(metadata, data, side, p, m) for m in data["moves"][side][p]]
            if any(m.startswith("hiddenpower") for m in names) and data["hidden_power_power"][side][p] not in (0, 60, 70):
                add(path+".hidden_power_power", "NATIVE_POWER_CONVENTION", "Native Hidden Power variants represent only60/70 power")
            happy = max(0, min(255, data["happiness"][side][p]))
            if ("return" in names and max(1, happy*10//25) != 102) or "frustration" in names:
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
            base_ability = _vocab(metadata, "abilities", data["base_ability"][side][p])
            extras = static.get(path, ("serious", (85,)*6, "none", 1.0, types, base_ability))
            if base_ability != extras[5]:
                add(path+".base_ability", "RETAINED_BASE_ABILITY_CHANGED", "Original ability differs from retained constructor evidence")
            gender = {-1:"none", 0:"none", 1:"male", 2:"female"}.get(genders[side][p], "none")
            if certificate is not None and gender != extras[2]:
                add(path+".gender", "RETAINED_GENDER_CHANGED", "Gender is immutable across admitted Gen3 futures")
            extras = (*extras[:2], gender, *extras[3:])
            if certificate is None and original_static is None and "oracle_species" in metadata:
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
        conditions = {k: 0 for k in CONDITIONS}
        # These are documented future-law quotients, NOT raw-state equality.
        # Native Protect uses min(streak,3) as a success exponent; JAX stores
        # the next attempt denominator. Native Toxic saturates14, JAX15, but
        # both next ticks use min(counter,14)+1 and preserve that saturation.
        denominator = data["protect_counter"][side][active]
        protect = {0: 0, 2: 1, 4: 2, 8: 3}.get(denominator)
        if protect is None:
            add(side_name+".protect_counter", "INVALID_PROTECT_DENOMINATOR", "Live JAX denominator must be0/2/4/8")
        toxic = data["status_counter"][side][active] if data["status"][side][active] == 4 else 0
        if toxic < 0:
            add(side_name+".status_counter", "INVALID_TOXIC_COUNTER", "Negative Toxic count")
        conditions.update(spikes=data["spikes"][side], protect=protect if protect is not None else 0,
            toxic_count=min(toxic, 14))
        last = "move:none"
        if use_last:
            kind, index = data["last_action_kind"][side], data["last_action_index"][side]
            if kind == 3:
                last = "move:struggle"
            elif kind == 2:
                last = f"move:{index}"
            elif kind == 1:
                last = f"switch:{index}"
        expected_request = data["hp"][side][active] <= 0 and alive[side]
        if bool(data["forced_switch"][side]) != expected_request and not data["pending_residual"]:
            add(side_name+".forced_switch", "INCONSISTENT_REQUEST", "Replacement request disagrees with active HP")
        saved_move = "none"
        if boundary == "suspended" and data["queued_actor"] == side:
            slot = data["queued_actions"][side]
            if type(slot) is not int or not 0 <= slot < 4:
                add(side_name+".queued_actions", "INVALID_QUEUED_SLOT", "Saved action must identify an exact move slot")
            else:
                saved_move = mons[active]["moves"][slot]["id"]
        result[side_name] = dict(pokemon=tuple(mons), active_index=str(active),
            side_conditions=conditions, volatile_status_durations={k: 0 for k in DURATIONS},
            volatile_statuses=tuple(sorted(volatiles)), substitute_health=data["substitute_hp"][side][active],
            **{name+"_boost": data["boosts"][side][active][i] for i, name in enumerate(BOOST_NAMES)},
            wish=(data["wish_counter"][side], data["wish_amount"][side]), future_sight=(0, "0"),
            force_switch=bool(data["forced_switch"][side]) if boundary == "suspended" else False,
            switch_out_move_second_saved_move=saved_move,
            baton_passing=bool(data["baton_pass"][side]) if boundary == "suspended" else False, shed_tailing=False,
            force_trapped=bool(data["trapped"][side]), last_used_move=last, slow_uturn_move=False)
    return result, gaps, tuple(sorted(static.items())), use_last, use_damage


def _construct_native(expected, module):
    """Shared physical constructor; boundary policy is applied before this call."""
    arguments = {k: v for k, v in expected.items() if k not in (*SIDE_NAMES, "derived_done", "derived_winner", "conditional_last", "conditional_damage")}
    for name in SIDE_NAMES:
        side = dict(expected[name])
        side["pokemon"] = [module.Pokemon(**dict(mon, moves=[module.Move(**move) for move in mon["moves"]])) for mon in side["pokemon"]]
        side["side_conditions"] = module.SideConditions(**side["side_conditions"])
        side["volatile_status_durations"] = module.VolatileStatusDurations(**side["volatile_status_durations"])
        side["volatile_statuses"] = set(side["volatile_statuses"])
        arguments[name] = module.Side(**side)
    return module.State(**arguments)


def native_input_from_jax(state, index, metadata, *, module=None):
    """Construct all12 native mons and a field certificate, or return REFUSED."""
    raw = ()
    try:
        data = _sample(state, index)
        raw = _freeze(data)
    except (ValueError, KeyError, IndexError, TypeError) as error:
        return Refused((Gap("schema", "INVALID_INPUT", str(error)),), raw)
    return _native_input_from_data(data, metadata, module=module)


def _native_input_from_data(data, metadata, *, module=None,
                            original_static=None, original_genders=None):
    raw = _freeze(data)
    try:
        expected, gaps, static, last, damage = _prepare(data, metadata,
            original_static=original_static, original_genders=original_genders)
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
        native = _construct_native(expected, module)
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
    return _canonical_native_future(native_state, certificate)


def _canonical_native_future(native_state, certificate, *, boundary="ordinary"):
    if boundary not in ("ordinary", "suspended"):
        raise ValueError("unknown correspondence boundary policy")
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
            # In the bound Gen3 source, ordinary departure removes SUBSTITUTE
            # but leaves substitute_health's scalar storage behind. Damage
            # reads it only under the volatile guard. A new Substitute sets
            # it to maxhp//4 (the reversible instruction stores new-old).
            # Thus this absent-volatile value cannot affect a future law.
            # Do not erase an active/Baton-carried Substitute, or malformed
            # storage outside native nonnegative i16. Existing suspension
            # guards below still refuse the whole state; original wire bytes
            # remain raw_native regardless of any canonicalization.
            if not 0 <= side['substitute_health'] <= 32767:
                gaps.append(Gap(name+'.substitute_health', 'INVALID_NATIVE_SUBSTITUTE_HEALTH',
                                repr(side['substitute_health'])))
            elif 'substitute' not in side['volatile_statuses']:
                side['substitute_health'] = 0
            allowed = set(VOLATILES.values()) | {"substitute", "truant", "solarbeam", "trapped"}
            unknown = set(side["volatile_statuses"]) - allowed
            if unknown:
                gaps.append(Gap(name+".volatile_statuses", "NATIVE_UNREPRESENTED_VOLATILE", repr(sorted(unknown))))
            for field, value in side["side_conditions"].items():
                if value and field not in {"spikes", "protect", "toxic_count"}:
                    gaps.append(Gap(name+".side_conditions."+field, "NATIVE_UNREPRESENTED_CONDITION", repr(value)))
            if side["side_conditions"]["protect"] < 0 or side["side_conditions"]["toxic_count"] < 0:
                gaps.append(Gap(name+".side_conditions", "INVALID_NATIVE_COUNTER", "Negative Protect/Toxic count"))
            side["side_conditions"]["protect"] = min(side["side_conditions"]["protect"], 3)
            side["side_conditions"]["toxic_count"] = min(side["side_conditions"]["toxic_count"], 14)
            for field, value in side["volatile_status_durations"].items():
                if value:
                    gaps.append(Gap(name+".volatile_status_durations."+field, "LATENT_DURATION_CONVENTION", repr(value)))
            counter,amount = side["wish"]
            if not 0 <= counter <= 127 or not 0 <= amount <= 32767:
                gaps.append(Gap(name+".wish", "INVALID_NATIVE_WISH_HISTORY",
                                "Expected nonnegative native i8 counter and i16 stored amount"))
            for field in ("future_sight", "force_switch", "baton_passing", "shed_tailing", "slow_uturn_move"):
                if boundary == "suspended" and field in ("force_switch", "baton_passing"):
                    continue  # explicit capsule policy validates these live fields
                value = side[field]
                if value not in (False, (0, 0), (0, "0")):
                    gaps.append(Gap(name+"."+field, "NATIVE_LIVE_FUTURE_FIELD", repr(value)))
            if boundary == "ordinary" and side["switch_out_move_second_saved_move"] != "none":
                gaps.append(Gap(name+".switch_out_move_second_saved_move", "LIVE_SUSPENSION_INTERFACE", "Saved action must not be erased"))
            for p, mon in enumerate(side["pokemon"]):
                original = dict(certificate.static_native)[f"{name}.pokemon[{p}]"]
                expected_gender = original[2]
                if mon["gender"] != expected_gender:
                    gaps.append(Gap(f"{name}.pokemon[{p}].gender", "NATIVE_GENDER_CHANGED", "Retained constructor gender changed"))
                if mon["base_ability"] != original[5]:
                    gaps.append(Gap(f"{name}.pokemon[{p}].base_ability", "NATIVE_BASE_ABILITY_CHANGED", "Retained original ability changed"))
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


def native_actions_from_jax(state, index, actions, native_input):
    """Exact nine-slot action identities, never public-string first-match coercion.

    Mirrors state.usable_moves/legal_actions and the selected ability traps in
    transition.action_mask. Native MoveChoice::from_string scans EVERY inactive
    Pokemon before moves and scans move slots in order, even disabled/0PP slots.
    Duplicate strings therefore refuse rather than silently choosing a slot.
    Public generate_instructions takes bare switch species IDs, not the
    human-facing `switch <species>` strings exported by search option displays.
    """
    raw = ()
    original = _freeze(actions)
    def refuse(code, detail):
        return Refused((Gap("actions", code, detail),), raw,
            native_input.raw_native if isinstance(native_input, NativeInput) else None,
            raw_actions=original)
    try:
        if not isinstance(native_input, NativeInput):
            return refuse("INPUT_NOT_ADMITTED", "A bound NativeInput certificate is required")
        data = _sample(state, index)
        raw = _freeze(data)
        if raw != native_input.certificate.input_fields or native_input.state.to_string() != native_input.raw_native:
            return refuse("ACTION_ROOT_BINDING_CHANGED", "Actions must address the exact constructed JAX/native root")
        if not isinstance(original, tuple) or len(original) != 2 or any(type(a) is not int or not 0 <= a <= 8 for a in original):
            return refuse("INVALID_NINE_ACTION_REQUEST", "Two integer actions in0..8 required")
        if data["done"]:
            return refuse("TERMINAL_NO_ACTION", "Terminal states have no scientific action request")
        native = _native_wire(native_input.raw_native)
        active = data["active"]
        hp = [data["hp"][side][active[side]] for side in range(2)]
        forced = [bool(data["forced_switch"][s]) or hp[s] <= 0 for s in range(2)]
        replacement = any(forced)
        result = []
        for side, action in enumerate(original):
            if replacement and not forced[side]:
                result.append("none")  # genuine ignored waiter, never Struggle
                continue
            p = active[side]
            own = native[SIDE_NAMES[side]]["pokemon"][p]
            foe = native[SIDE_NAMES[1-side]]["pokemon"][active[1-side]]
            bench = [i for i in range(6) if i != p]
            usable = [pp > 0 and not disabled and (not data["choice_move"][side][p] or move == data["choice_move"][side][p])
                for move, pp, disabled in zip(data["moves"][side][p], data["pp"][side][p], data["disabled"][side][p], strict=True)]
            charge = data["charge_move"][side][p]
            if charge:
                usable = [move == charge for move in data["moves"][side][p]]
                if not any(usable):
                    return refuse("COPIED_CHARGE_ACTION_UNREPRESENTED", "JAX fallback release has no matching public native move slot")
            if action < 4:
                if forced[side]:
                    return refuse("ILLEGAL_REPLACEMENT_MOVE", repr(original))
                if not any(usable):
                    if action != 0:
                        return refuse("ILLEGAL_STRUGGLE_SLOT", "Synthetic Struggle occupies canonical slot0 only")
                    result.append("struggle")
                    continue
                if not usable[action]:
                    return refuse("ILLEGAL_MOVE_SLOT", repr(original))
                name = own["moves"][action]["id"]
                if name in {"none", "struggle"}:
                    return refuse("RESERVED_MOVE_ID", "Real move slot must not turn into a public wait/synthetic action")
                if sum(move["id"] == name for move in own["moves"]) != 1:
                    return refuse("DUPLICATE_MOVE_ID", "Public native string cannot bind one selected move slot")
                if any(native[SIDE_NAMES[side]]["pokemon"][b]["id"] == name for b in bench):
                    return refuse("MOVE_SWITCH_STRING_COLLISION", "Native parser scans all inactive species before move slots")
                result.append(name)
            else:
                target = bench[action-4]
                target_mon = native[SIDE_NAMES[side]]["pokemon"][target]
                grounded = "flying" not in own["types"] and own["ability"] != "levitate"
                ability_trap = hp[side] > 0 and hp[1-side] > 0 and (foe["ability"] == "shadowtag" or
                    (foe["ability"] == "magnetpull" and "steel" in own["types"]) or
                    (foe["ability"] == "arenatrap" and grounded))
                linked = data["trap_source"][side][p] == active[1-side] and hp[1-side] > 0
                trapped = data["trapped"][side] or bool(data["volatile"][side][p] & 8) or linked or bool(charge) or ability_trap
                if target_mon["hp"] <= 0 or (trapped and not forced[side]):
                    return refuse("ILLEGAL_SWITCH", repr(original))
                name = target_mon["id"]
                if name in {"none", "struggle"} or sum(native[SIDE_NAMES[side]]["pokemon"][b]["id"] == name for b in bench) != 1:
                    return refuse("DUPLICATE_OR_RESERVED_SWITCH_ID", "Public parser must identify exactly this inactive party slot, including fainted duplicates")
                result.append(name)
        return NativeActions(tuple(result), original, raw, native_input.raw_native)
    except (ValueError, IndexError, TypeError, KeyError, AttributeError) as error:
        return refuse("INVALID_ACTION_BINDING", str(error))
