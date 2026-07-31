"""Audit the Python-to-Rust damage-stat construction seam.

This is diagnostic-only.  The engine receives a constructed ``State`` before it
enumerates branches; attest that state rather than trying to infer its inputs
from a damage discrepancy after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .poke_engine_adapter import TYPELESS, BattleSpec, build_poke_engine_state

_STAT_FIELDS = ("attack", "defense", "special_attack", "special_defense", "speed")
_BOOST_FIELDS = (
    "attack_boost",
    "defense_boost",
    "special_attack_boost",
    "special_defense_boost",
    "speed_boost",
)


@dataclass(frozen=True)
class DamageStatAttestation:
    """Exact comparison of the adapter spec and native branch-input state."""

    expected: Mapping[str, Mapping[str, object]]
    actual: Mapping[str, Mapping[str, object]]
    mismatches: tuple[str, ...]

    @property
    def matches(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict[str, object]:
        return {
            "matches": self.matches,
            "expected": {key: dict(value) for key, value in self.expected.items()},
            "actual": {key: dict(value) for key, value in self.actual.items()},
            "mismatches": list(self.mismatches),
        }


def attest_damage_stat_inputs(spec: BattleSpec, state: Any) -> DamageStatAttestation:
    """Compare the complete adapter payload to the Rust branch-input state.

    Stored Attack/Defense/Special Attack/Special Defense/Speed values remain on
    each Pokemon.  Stages remain on its active Side.  Ability, item, status and
    types are also read because they are damage-relevant native inputs that can
    modify a stored stat or the damage formula.  The result covers both active
    and benched members: a future switch must not surface a latent corruption.
    """

    expected: dict[str, dict[str, object]] = {}
    actual: dict[str, dict[str, object]] = {}
    mismatches: list[str] = []
    expected["field"] = {
        "weather": str(spec.weather),
        # The adapter intentionally omits this keyword for no-weather states;
        # the native default is 0 rather than BattleSpec's -1 sentinel.
        "weather_turns_remaining": (
            0 if str(spec.weather) == "none" else int(spec.weather_turns_remaining)
        ),
    }
    actual["field"] = {
        "weather": str(state.weather),
        "weather_turns_remaining": int(state.weather_turns_remaining),
    }
    for field, expected_value in expected["field"].items():
        actual_value = actual["field"][field]
        if expected_value != actual_value:
            mismatches.append(f"field.{field}: expected {expected_value!r}, got {actual_value!r}")
    for side_name, side_spec in (("side_one", spec.side_one), ("side_two", spec.side_two)):
        native_side = getattr(state, side_name)
        expected_boosts = {
            field: int(side_spec.boosts.get(field.removesuffix("_boost"), 0))
            for field in _BOOST_FIELDS
        }
        actual_boosts = {field: int(getattr(native_side, field)) for field in _BOOST_FIELDS}
        expected[f"{side_name}.active_boosts"] = expected_boosts
        actual[f"{side_name}.active_boosts"] = actual_boosts
        for field in _BOOST_FIELDS:
            if expected_boosts[field] != actual_boosts[field]:
                mismatches.append(
                    f"{side_name}.{field}: expected {expected_boosts[field]}, got {actual_boosts[field]}"
                )

        expected_side_modifiers = {
            "side_conditions": {
                key: int(value) for key, value in side_spec.side_conditions.items() if int(value)
            },
            "volatile_statuses": tuple(sorted(str(value) for value in side_spec.volatile_statuses)),
        }
        native_conditions = native_side.side_conditions
        actual_side_modifiers = {
            "side_conditions": {
                key: int(getattr(native_conditions, key))
                for key in expected_side_modifiers["side_conditions"]
                if int(getattr(native_conditions, key))
            },
            "volatile_statuses": tuple(sorted(str(value) for value in native_side.volatile_statuses)),
        }
        expected[f"{side_name}.damage_modifiers"] = expected_side_modifiers
        actual[f"{side_name}.damage_modifiers"] = actual_side_modifiers
        for field, expected_value in expected_side_modifiers.items():
            actual_value = actual_side_modifiers[field]
            if expected_value != actual_value:
                mismatches.append(
                    f"{side_name}.damage_modifiers.{field}: expected {expected_value!r}, got {actual_value!r}"
                )

        native_party = tuple(native_side.pokemon)
        # The adapter pads short fixture parties with inert fainted placeholders.
        # A materialized world supplies all six members, but the diagnostic is
        # also useful for focused one-member fixtures, so only a truncated
        # native party is corruption.
        if len(native_party) < len(side_spec.pokemon):
            mismatches.append(
                f"{side_name}.pokemon length: expected at least {len(side_spec.pokemon)}, got {len(native_party)}"
            )
        for index, member in enumerate(side_spec.pokemon):
            key = f"{side_name}.pokemon[{index}]"
            expected_member = {
                "id": member.id,
                "hp": int(member.hp),
                "maxhp": int(member.maxhp),
                **{field: int(getattr(member, field)) for field in _STAT_FIELDS},
                "ability": member.ability or "none",
                "item": member.item or "none",
                "status": member.status,
                "types": tuple(member.types)
                if len(member.types) == 2
                else tuple(member.types) + (TYPELESS,),
            }
            expected[key] = expected_member
            if index >= len(native_party):
                actual[key] = {"missing": True}
                continue
            native_member = native_party[index]
            actual_member = {
                "id": str(native_member.id),
                "hp": int(native_member.hp),
                "maxhp": int(native_member.maxhp),
                **{field: int(getattr(native_member, field)) for field in _STAT_FIELDS},
                "ability": str(native_member.ability),
                "item": str(native_member.item),
                "status": str(native_member.status),
                "types": tuple(str(value) for value in native_member.types),
            }
            actual[key] = actual_member
            for field, expected_value in expected_member.items():
                actual_value = actual_member[field]
                if expected_value != actual_value:
                    mismatches.append(
                        f"{key}.{field}: expected {expected_value!r}, got {actual_value!r}"
                    )
    return DamageStatAttestation(expected=expected, actual=actual, mismatches=tuple(mismatches))


def build_and_attest_damage_stat_inputs(
    spec: BattleSpec, *, module: Any | None = None
) -> tuple[Any, DamageStatAttestation]:
    """Construct native branch inputs and return their materialization audit."""

    state = build_poke_engine_state(spec, module=module)
    return state, attest_damage_stat_inputs(spec, state)
