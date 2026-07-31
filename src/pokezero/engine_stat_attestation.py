"""Audit the ``BattleSpec`` to native ``State`` transport seam.

This diagnostic compares the values the Python adapter forwards with the native
``State`` it constructs.  A clean result establishes only that transport seam:
it does not attest how a belief world derived its ``BattleSpec`` and it does not
attest native branch generation or Gen 3 damage arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any, Mapping, Sequence

from .dex import normalize_id
from .poke_engine_adapter import TYPELESS, BattleSpec, build_poke_engine_state

_STAT_FIELDS = ("attack", "defense", "special_attack", "special_defense", "speed")
_BOOST_FIELDS = (
    "attack_boost",
    "defense_boost",
    "special_attack_boost",
    "special_defense_boost",
    "speed_boost",
    "accuracy_boost",
    "evasion_boost",
)
_PRE_TRANSFORM_MOVE_SLOTS = 4


def _id(value: object | None) -> str:
    """Use the adapter's lowercase Showdown-id convention for native enums."""

    text = str(value or "")
    # PyO3 enum display has historically been either ``tackle`` or
    # ``MoveName.TACKLE``.  The latter must compare as the same production id.
    return normalize_id(text.rsplit(".", 1)[-1])


def _types(values: Sequence[object]) -> tuple[str, str]:
    normalized = tuple(_id(value) for value in values)
    if len(normalized) == 1:
        return normalized + (TYPELESS,)
    return normalized  # Adapter validation limits this to exactly two slots.


def _gender(value: object | None) -> str:
    """Normalize the adapter's compact gender inputs and native display values."""

    normalized = _id(value)
    return {
        "": "none",
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female",
        "n": "none",
        "none": "none",
    }.get(normalized, f"invalid:{value!r}")


def _float32(value: object) -> float:
    """Canonicalize Python inputs to the binding's native ``f32`` precision."""

    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def _pre_transform_from_spec(member: Any) -> dict[str, object] | None:
    """Canonicalize the exact restoration snapshot forwarded by the adapter."""

    base = member.pre_transform
    if base is None:
        return None
    moves = [
        (_id(move.id), int(move.pp))
        for move in base.moves[:_PRE_TRANSFORM_MOVE_SLOTS]
    ]
    moves.extend(
        [("none", 0)] * (_PRE_TRANSFORM_MOVE_SLOTS - len(moves))
    )
    return {
        "id": _id(base.id),
        **{field: int(getattr(base, field)) for field in _STAT_FIELDS},
        "moves": tuple(moves),
    }


def _pre_transform_from_native(value: object | None) -> dict[str, object] | None:
    """Parse the binding's public PreTransform wire record without hiding corruption."""

    raw = str(value or "")
    if not raw:
        return None
    fields = raw.split(";")
    if len(fields) != 6 + _PRE_TRANSFORM_MOVE_SLOTS:
        return {"invalid_wire": raw}
    try:
        moves: list[tuple[str, int]] = []
        for slot in fields[6:]:
            move_id, separator, pp = slot.partition(":")
            if not separator:
                return {"invalid_wire": raw}
            moves.append((_id(move_id), int(pp)))
        return {
            "id": _id(fields[0]),
            **{
                field: int(fields[index])
                for index, field in enumerate(_STAT_FIELDS, start=1)
            },
            "moves": tuple(moves),
        }
    except (TypeError, ValueError):
        return {"invalid_wire": raw}


def _native_active_index(value: object) -> int | str:
    """The binding exposes a string slot index; preserve malformed values visibly."""

    try:
        return int(str(value))
    except (TypeError, ValueError):
        return f"invalid:{value!r}"


def _native_side_conditions(value: object) -> dict[str, int]:
    """Read every nonzero exposed condition, including unexpected native fields."""

    conditions: dict[str, int] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        candidate = getattr(value, name)
        # SideConditions is an integer-only record.  Ignore methods/properties
        # outside that record, but do not restrict names to the expected map:
        # an extra live native condition is a transport mismatch.
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            continue
        if candidate:
            conditions[name] = int(candidate)
    return conditions


def _native_durations(value: object | None) -> dict[str, int]:
    """Read every nonzero exposed volatile counter, including unknown additions."""

    return {} if value is None else _native_side_conditions(value)


def _last_used_move(value: object | None) -> str:
    """Normalize the binding's empty last-used-move sentinel."""

    raw = str(value or "")
    if not raw:
        return "move:none"
    prefix, separator, index = raw.partition(":")
    return f"{prefix}:{_id(index)}" if separator else _id(raw)


@dataclass(frozen=True)
class BattleSpecTransportAttestation:
    """Exact comparison of adapter-forwarded inputs and native construction output."""

    expected: Mapping[str, Mapping[str, object]]
    actual: Mapping[str, Mapping[str, object]]
    mismatches: tuple[str, ...]

    @property
    def matches(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": "BattleSpec_to_native_State_transport_only",
            "does_not_attest": [
                "belief_world_to_BattleSpec_derivation",
                "native_branch_generation",
                "native_Gen3_damage_arithmetic",
            ],
            "matches": self.matches,
            "expected": {key: dict(value) for key, value in self.expected.items()},
            "actual": {key: dict(value) for key, value in self.actual.items()},
            "mismatches": list(self.mismatches),
        }


def attest_battle_spec_transport(spec: BattleSpec, state: Any) -> BattleSpecTransportAttestation:
    """Compare all damage-relevant fields forwarded from ``BattleSpec`` to ``State``.

    This includes each party member because a damaged bench can become the active
    defender on a later branch.  It deliberately stops at the constructed state;
    it cannot show that the world builder chose the right values or that Rust
    subsequently applies them with correct mechanics.
    """

    expected: dict[str, dict[str, object]] = {}
    actual: dict[str, dict[str, object]] = {}
    mismatches: list[str] = []

    expected["field"] = {
        "weather": _id(spec.weather),
        # The adapter omits the constructor keyword for no-weather states; the
        # native default is 0 rather than BattleSpec's -1 sentinel.
        "weather_turns_remaining": 0 if _id(spec.weather) == "none" else int(spec.weather_turns_remaining),
        "terrain": _id(spec.terrain),
        "trick_room": bool(spec.trick_room),
    }
    actual["field"] = {
        "weather": _id(state.weather),
        "weather_turns_remaining": int(state.weather_turns_remaining),
        "terrain": _id(state.terrain),
        "trick_room": bool(state.trick_room),
    }
    _compare_mapping("field", expected["field"], actual["field"], mismatches)

    for side_name, side_spec in (("side_one", spec.side_one), ("side_two", spec.side_two)):
        native_side = getattr(state, side_name)
        expected_side = {
            "active_index": int(side_spec.active_index),
            "side_conditions": {
                key: int(value) for key, value in side_spec.side_conditions.items() if int(value)
            },
            "volatile_statuses": tuple(sorted(_id(value) for value in side_spec.volatile_statuses)),
            # Substitute HP and Wish are direct HP-transition inputs.  The
            # remaining values below are also explicitly forwarded by the
            # adapter, so compare them while this seam is open.
            "substitute_health": int(side_spec.substitute_health),
            "force_switch": bool(side_spec.force_switch),
            "wish": tuple(int(value) for value in side_spec.wish),
            "baton_passing": bool(side_spec.baton_passing),
            "slow_uturn_move": bool(side_spec.slow_uturn_move),
            "switch_out_move_second_saved_move": _id(side_spec.switch_out_move_second_saved_move)
            or "none",
            "last_used_move": _last_used_move(side_spec.last_used_move),
            "volatile_status_durations": {
                key: int(value)
                for key, value in side_spec.volatile_status_durations.items()
                if int(value)
            },
        }
        actual_side = {
            "active_index": _native_active_index(native_side.active_index),
            "side_conditions": _native_side_conditions(native_side.side_conditions),
            "volatile_statuses": tuple(sorted(_id(value) for value in native_side.volatile_statuses)),
            "substitute_health": int(getattr(native_side, "substitute_health", 0)),
            "force_switch": bool(getattr(native_side, "force_switch", False)),
            "wish": tuple(int(value) for value in getattr(native_side, "wish", (0, 0))),
            "baton_passing": bool(getattr(native_side, "baton_passing", False)),
            "slow_uturn_move": bool(getattr(native_side, "slow_uturn_move", False)),
            "switch_out_move_second_saved_move": _id(
                getattr(native_side, "switch_out_move_second_saved_move", "none")
            )
            or "none",
            "last_used_move": _last_used_move(getattr(native_side, "last_used_move", "")),
            "volatile_status_durations": _native_durations(
                getattr(native_side, "volatile_status_durations", None)
            ),
        }
        expected[f"{side_name}.transport"] = expected_side
        actual[f"{side_name}.transport"] = actual_side
        _compare_mapping(f"{side_name}.transport", expected_side, actual_side, mismatches)

        expected_boosts = {
            field: int(side_spec.boosts.get(field.removesuffix("_boost"), 0))
            for field in _BOOST_FIELDS
        }
        actual_boosts = {field: int(getattr(native_side, field)) for field in _BOOST_FIELDS}
        expected[f"{side_name}.active_boosts"] = expected_boosts
        actual[f"{side_name}.active_boosts"] = actual_boosts
        _compare_mapping(f"{side_name}.active_boosts", expected_boosts, actual_boosts, mismatches)

        native_party = tuple(native_side.pokemon)
        if len(native_party) < len(side_spec.pokemon):
            mismatches.append(
                f"{side_name}.pokemon length: expected at least {len(side_spec.pokemon)}, got {len(native_party)}"
            )
        for index, member in enumerate(side_spec.pokemon):
            key = f"{side_name}.pokemon[{index}]"
            expected_member = {
                "id": _id(member.id),
                "level": int(member.level),
                "hp": int(member.hp),
                "maxhp": int(member.maxhp),
                **{field: int(getattr(member, field)) for field in _STAT_FIELDS},
                "ability": _id(member.ability or "none"),
                "base_ability": _id(member.base_ability or member.ability or "none"),
                "item": _id(member.item or "none"),
                "nature": _id(member.nature or "serious"),
                "gender": _gender(member.gender),
                "status": _id(member.status),
                "types": _types(member.types),
                "base_types": _types(member.types if member.base_types is None else member.base_types),
                "moves": tuple(
                    (_id(move.id), int(move.pp), bool(move.disabled)) for move in member.moves
                ),
                # Low Kick's Gen 3 base power depends on this value.
                "weight_kg": _float32(member.weight_kg or 0.0),
                "rest_turns": int(member.rest_turns),
                "sleep_turns": int(member.sleep_turns),
                "pre_transform": _pre_transform_from_spec(member),
            }
            expected[key] = expected_member
            if index >= len(native_party):
                actual[key] = {"missing": True}
                continue
            native_member = native_party[index]
            actual_member = {
                "id": _id(native_member.id),
                "level": int(native_member.level),
                "hp": int(native_member.hp),
                "maxhp": int(native_member.maxhp),
                **{field: int(getattr(native_member, field)) for field in _STAT_FIELDS},
                "ability": _id(native_member.ability),
                "base_ability": _id(native_member.base_ability),
                "item": _id(native_member.item),
                "nature": _id(native_member.nature),
                "gender": _gender(native_member.gender),
                "status": _id(native_member.status),
                "types": _types(native_member.types),
                "base_types": _types(native_member.base_types),
                "moves": tuple(
                    (_id(move.id), int(move.pp), bool(getattr(move, "disabled", False)))
                    for move in native_member.moves
                ),
                "weight_kg": _float32(native_member.weight_kg),
                "rest_turns": int(native_member.rest_turns),
                "sleep_turns": int(native_member.sleep_turns),
                "pre_transform": _pre_transform_from_native(native_member.pre_transform),
            }
            actual[key] = actual_member
            _compare_mapping(key, expected_member, actual_member, mismatches)
        # The adapter pads a short fixture party with fainted ``NONE`` records
        # because native sides always carry six slots.  Those are transport
        # defaults, not an omitted BattleSpec member; any live extra member is
        # instead a concrete construction discrepancy.
        for index, native_member in enumerate(native_party[len(side_spec.pokemon):], start=len(side_spec.pokemon)):
            if _id(native_member.id) != "none" or int(native_member.hp) != 0:
                mismatches.append(
                    f"{side_name}.pokemon[{index}]: unexpected live native padding "
                    f"id={_id(native_member.id)!r} hp={int(native_member.hp)}"
                )
    return BattleSpecTransportAttestation(
        expected=expected, actual=actual, mismatches=tuple(mismatches)
    )


def _compare_mapping(
    path: str,
    expected: Mapping[str, object],
    actual: Mapping[str, object],
    mismatches: list[str],
) -> None:
    """Report both missing/extra fields and unequal values with stable paths."""

    for field in sorted(set(expected) | set(actual)):
        if field not in expected:
            mismatches.append(f"{path}.{field}: unexpected native value {actual[field]!r}")
        elif field not in actual:
            mismatches.append(f"{path}.{field}: missing native value; expected {expected[field]!r}")
        elif expected[field] != actual[field]:
            mismatches.append(
                f"{path}.{field}: expected {expected[field]!r}, got {actual[field]!r}"
            )


# Retained as a compatibility alias for early diagnostic callers.  New callers
# should use the transport-named API so they do not overstate its evidence.
DamageStatAttestation = BattleSpecTransportAttestation


def attest_damage_stat_inputs(spec: BattleSpec, state: Any) -> BattleSpecTransportAttestation:
    """Compatibility wrapper for :func:`attest_battle_spec_transport`."""

    return attest_battle_spec_transport(spec, state)


def build_and_attest_battle_spec_transport(
    spec: BattleSpec, *, module: Any | None = None
) -> tuple[Any, BattleSpecTransportAttestation]:
    """Construct native branch inputs and audit only their transport fidelity."""

    state = build_poke_engine_state(spec, module=module)
    return state, attest_battle_spec_transport(spec, state)


def attest_battle_spec_transport_variants(
    specs: Sequence[BattleSpec],
    states: Sequence[Any],
    *,
    variant_construction: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Attest aligned candidate worlds without hiding a dropped construction.

    Hidden-counter support may deliberately reject one candidate as illegal for
    the native constructor.  That is diagnostic data, not permission to zip
    later states against earlier specs.  Callers receive an explicit structured
    result and must not claim a full candidate-world transport clearance.
    """

    construction = [dict(row) for row in variant_construction]
    dropped = [row for row in construction if row.get("status") != "constructed"]
    if dropped or len(specs) != len(states):
        return {
            "status": "dropped_variant_construction",
            "requested_variants": len(construction) if construction else len(specs),
            "comparison_states": len(states),
            "hidden_counter_variants": max(0, (len(construction) or len(specs)) - 1),
            "variant_construction": construction,
        }
    attestations = [
        attest_battle_spec_transport(spec, state).to_dict()
        for spec, state in zip(specs, states, strict=True)
    ]
    return {
        "status": "transport_attested" if all(row["matches"] for row in attestations) else "transport_mismatch",
        "requested_variants": len(specs),
        "comparison_states": len(attestations),
        "hidden_counter_variants": max(0, len(attestations) - 1),
        "variant_construction": construction,
        "attestations": attestations,
    }


def build_and_attest_damage_stat_inputs(
    spec: BattleSpec, *, module: Any | None = None
) -> tuple[Any, BattleSpecTransportAttestation]:
    """Compatibility wrapper for the transport-named helper."""

    return build_and_attest_battle_spec_transport(spec, module=module)
