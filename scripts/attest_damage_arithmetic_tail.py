#!/usr/bin/env python
"""Attest whether retained damage-tail rows are arithmetic or branch-composition defects.

The differential deliberately accepts a direct hit when the observed magnitude
falls in the 16 Gen 3 legal rolls.  That test alone cannot explain a later
poison/burn residual: the native engine currently emits one representative
nonterminal damage instruction, so it does not cross product that damage roll with
secondary-effect chance branches.  This tool makes the distinction explicit.

For a recorded repro it compares three independently useful values from one
pre-hit state:

* the observed Showdown direct HP delta;
* the pure-Python Gen 3 oracle transcribed from Showdown; and
* the native ``calculate_damage`` maximum plus every rendered instruction
  branch.

It only calls the pure oracle exact when no modifier outside its basic,
fully-visible context is active.  Complex rows remain evidence with an honest
``comparison_limit`` rather than a guessed arithmetic verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import poke_engine  # noqa: E402
import pokezero_search  # noqa: E402

from engine_build_fingerprint import assert_fresh  # noqa: E402
from pokezero.dex import (  # noqa: E402
    ShowdownDex,
    load_showdown_dex,
    normalize_id,
    resolve_move_base_power,
)
from pokezero.gen3_damage import Gen3DamageContext, gen3_damage_rolls  # noqa: E402


def _target(value: str) -> tuple[int, int]:
    seed, separator, step = value.partition("/")
    if not separator or not seed.isdigit() or not step.isdigit():
        raise argparse.ArgumentTypeError("targets must be SEED/STEP")
    return int(seed), int(step)


def _slot(value: str) -> str:
    return value.split(":", 1)[0].strip()[:2]


def _hp(condition: str) -> int:
    head = condition.strip().split(" ", 1)[0]
    if head in {"0", "0.0"} or "fnt" in condition:
        return 0
    value, _, _ = head.partition("/")
    return int(value)


def _has_from(parts: Sequence[str]) -> bool:
    return any(part.strip().startswith("[from]") for part in parts[4:])


@dataclass(frozen=True)
class DirectHit:
    actor: str
    target: str
    move: str
    damage: int
    critical: bool
    secondary_status: str | None


def observed_direct_hit(row: Mapping[str, Any]) -> DirectHit | None:
    """Return the first direct move hit from a Showdown protocol slice.

    Switch rows must seed the target's running HP from the switch condition;
    otherwise the apparent direct delta would be measured against the outgoing
    Pokemon.  This mirrors the differential's event component parser but keeps
    only the one event the arithmetic oracle can adjudicate.
    """

    features = row.get("pre_features")
    if not isinstance(features, Mapping):
        return None
    running = {
        "p1": int(features.get("p1_hp") or 0),
        "p2": int(features.get("p2_hp") or 0),
    }
    active_move: tuple[str, str] | None = None
    last_direct: DirectHit | None = None
    critical_targets: set[str] = set()
    lines = row.get("protocol")
    if not isinstance(lines, Sequence):
        return None
    for line in lines:
        if not isinstance(line, str):
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        tag = parts[1]
        if tag in {"switch", "drag", "replace"} and len(parts) > 4:
            running[_slot(parts[2])] = _hp(parts[4])
            continue
        if tag == "move" and len(parts) > 3:
            active_move = (_slot(parts[2]), normalize_id(parts[3]))
            critical_targets.clear()
            continue
        if tag == "-crit" and len(parts) > 2:
            critical_targets.add(_slot(parts[2]))
            continue
        if tag == "-status" and len(parts) > 3 and last_direct is not None:
            if _slot(parts[2]) == last_direct.target:
                return DirectHit(
                    actor=last_direct.actor,
                    target=last_direct.target,
                    move=last_direct.move,
                    damage=last_direct.damage,
                    critical=last_direct.critical,
                    secondary_status=normalize_id(parts[3]),
                )
            continue
        if tag != "-damage" or len(parts) < 4 or _has_from(parts):
            continue
        target = _slot(parts[2])
        new_hp = _hp(parts[3])
        before = running.get(target)
        running[target] = new_hp
        if before is None or active_move is None or target == active_move[0]:
            continue
        last_direct = DirectHit(
            actor=active_move[0],
            target=target,
            move=active_move[1],
            damage=before - new_hp,
            critical=target in critical_targets,
            secondary_status=None,
        )
    return last_direct


def _native_member(member: Any) -> dict[str, object]:
    return {
        "id": normalize_id(str(member.id)),
        "level": int(member.level),
        "hp": int(member.hp),
        "maxhp": int(member.maxhp),
        "attack": int(member.attack),
        "defense": int(member.defense),
        "special_attack": int(member.special_attack),
        "special_defense": int(member.special_defense),
        "speed": int(member.speed),
        "ability": normalize_id(str(member.ability)),
        "item": normalize_id(str(member.item)),
        "status": normalize_id(str(member.status)),
        "types": tuple(normalize_id(str(value)) for value in member.types),
    }


def _side_snapshot(state: Any, side: str) -> dict[str, object]:
    native_side = getattr(state, "side_one" if side == "p1" else "side_two")
    member = native_side.pokemon[int(native_side.active_index)]
    conditions = native_side.side_conditions
    return {
        "active": _native_member(member),
        "boosts": {
            "attack": int(native_side.attack_boost),
            "defense": int(native_side.defense_boost),
            "special_attack": int(native_side.special_attack_boost),
            "special_defense": int(native_side.special_defense_boost),
            "speed": int(native_side.speed_boost),
        },
        "screens": {
            "reflect": int(getattr(conditions, "reflect", 0)),
            "lightscreen": int(getattr(conditions, "light_screen", 0)),
        },
        "volatiles": tuple(sorted(normalize_id(str(value)) for value in native_side.volatile_statuses)),
    }


def _weather_modifier(weather: str, move_type: str) -> tuple[float, float] | None:
    if weather in {"sun", "sunnyday"}:
        if move_type == "fire":
            return (1.5, 1)
        if move_type == "water":
            return (0.5, 1)
    if weather in {"rain", "raindance"}:
        if move_type == "water":
            return (1.5, 1)
        if move_type == "fire":
            return (0.5, 1)
    return None


def _basic_oracle(
    *,
    state: Any,
    direct: DirectHit,
    dex: ShowdownDex,
) -> tuple[dict[str, object], tuple[int, ...] | None, str | None]:
    """Build the exact simple-case Showdown oracle or return its limit reason."""

    info = dex.move_info(direct.move)
    if info is None or info.gen3_category not in {"Physical", "Special"}:
        return {}, None, "move_not_a_standard_damaging_move"
    attacker_side = _side_snapshot(state, direct.actor)
    defender_side = _side_snapshot(state, direct.target)
    attacker = attacker_side["active"]
    defender = defender_side["active"]
    assert isinstance(attacker, Mapping) and isinstance(defender, Mapping)
    category = info.gen3_category
    move_type = normalize_id(info.type)
    weather = normalize_id(str(state.weather))
    context = {
        "move": direct.move,
        "base_power": resolve_move_base_power(
            info, int(attacker["hp"]) / max(1, int(attacker["maxhp"]))
        ),
        "category": category,
        "move_type": move_type,
        "weather": weather,
        "attacker": dict(attacker),
        "defender": dict(defender),
        "attacker_boosts": dict(attacker_side["boosts"]),
        "defender_boosts": dict(defender_side["boosts"]),
        "defender_screens": dict(defender_side["screens"]),
        "attacker_volatiles": list(attacker_side["volatiles"]),
        "defender_volatiles": list(defender_side["volatiles"]),
    }
    relevant_abilities = {
        "guts", "hustle", "hugepower", "purepower", "marvelscale", "thickfat",
        "blaze", "overgrow", "torrent", "swarm", "airlock", "cloudnine",
    }
    type_boost_items = {
        "blackbelt", "blackglasses", "charcoal", "dragonfang", "hardstone", "magnet",
        "metalcoat", "miracleseed", "mysticwater", "nevermeltice", "pinkbow",
        "poisonbarb", "sharpbeak", "silkscarf", "silverpowder", "softsand", "spelltag",
        "twistedspoon",
    }
    reasons: list[str] = []
    attacker_ability = str(attacker["ability"])
    defender_ability = str(defender["ability"])
    attacker_item = str(attacker["item"])
    defender_item = str(defender["item"])
    if attacker_ability in relevant_abilities:
        reasons.append(f"attacker_ability:{attacker_ability}")
    if defender_ability in {"marvelscale", "thickfat", "airlock", "cloudnine"}:
        reasons.append(f"defender_ability:{defender_ability}")
    if attacker_item in type_boost_items:
        reasons.append(f"attacker_item:{attacker_item}")
    if defender_item == "souldew":
        reasons.append("defender_item:souldew")
    if attacker_item in {"thickclub", "lightball", "souldew"}:
        reasons.append(f"attacker_item:{attacker_item}")
    if category == "Physical" and str(attacker["status"]) == "burn":
        reasons.append("attacker_burn")
    if int(defender_side["screens"]["reflect" if category == "Physical" else "lightscreen"]):
        reasons.append("defender_screen")
    if "flashfire" in attacker_side["volatiles"]:
        reasons.append("attacker_flashfire")
    if direct.move in {"solarbeam", "facade", "flail", "reversal", "eruption", "waterspout"}:
        reasons.append(f"variable_or_conditioned_move:{direct.move}")
    if reasons:
        return context, None, ",".join(reasons)

    attack_key = "attack" if category == "Physical" else "special_attack"
    defense_key = "defense" if category == "Physical" else "special_defense"
    attack_boost_key = "attack" if category == "Physical" else "special_attack"
    defense_boost_key = "defense" if category == "Physical" else "special_defense"
    attack_mods: list[tuple[float, float]] = []
    if attacker_item == "choiceband" and category == "Physical":
        attack_mods.append((1.5, 1))
    oracle = Gen3DamageContext(
        level=int(attacker["level"]),
        base_power=int(context["base_power"]),
        category=category,
        attack=int(attacker[attack_key]),
        defense=int(defender[defense_key]),
        attack_boost=int(attacker_side["boosts"][attack_boost_key]),
        defense_boost=int(defender_side["boosts"][defense_boost_key]),
        attack_mods=tuple(attack_mods),
        stab=move_type in set(attacker["types"]),
        effectiveness=dex.effectiveness(info.type, tuple(str(value) for value in defender["types"])),
        weather_mod=_weather_modifier(weather, move_type),
        crit=direct.critical,
    )
    return context, gen3_damage_rolls(oracle), None


def _branch_direct_damage(events: Sequence[object], target: str) -> int | None:
    running: int | None = None
    for event in events:
        if not isinstance(event, str):
            continue
        parts = event.split("|")
        if len(parts) < 3:
            continue
        if parts[1] in {"switch", "drag", "replace"} and len(parts) > 4 and _slot(parts[2]) == target:
            running = _hp(parts[4])
            continue
        if parts[1] != "-damage" or len(parts) < 4 or _has_from(parts) or _slot(parts[2]) != target:
            continue
        new_hp = _hp(parts[3])
        if running is not None:
            return running - new_hp
    return None


def _branch_has_status(events: Sequence[object], target: str, status: str) -> bool:
    return any(
        isinstance(event, str)
        and event.startswith("|-status|")
        and len(event.split("|")) > 3
        and _slot(event.split("|")[2]) == target
        and normalize_id(event.split("|")[3]) == status
        for event in events
    )


def _branch_target_fainted(events: Sequence[object], target: str) -> bool:
    return any(
        isinstance(event, str)
        and event.startswith("|faint|")
        and len(event.split("|")) > 2
        and _slot(event.split("|")[2]) == target
        for event in events
    )


def _branch_report(row: Mapping[str, Any], direct: DirectHit, dex: ShowdownDex) -> dict[str, object]:
    state_text = row.get("engine_state")
    choices = row.get("choices")
    if not isinstance(state_text, str) or not isinstance(choices, Mapping):
        return {"verdict": "comparison_limit", "reason": "missing_repro_engine_state_or_choices"}
    side_one_choice = choices.get("p1")
    side_two_choice = choices.get("p2")
    if not isinstance(side_one_choice, str) or not isinstance(side_two_choice, str):
        return {"verdict": "comparison_limit", "reason": "invalid_repro_choices"}
    context = json.dumps({"p1": [], "p2": [], "turn": 0})
    state_texts = row.get("engine_states")
    if not isinstance(state_texts, Sequence) or isinstance(state_texts, (str, bytes)):
        state_texts = [state_text]
    branch_rows: list[dict[str, object]] = []
    native_rolls: tuple[int, int] | None = None
    oracle_context: dict[str, object] = {}
    oracle_rolls: tuple[int, ...] | None = None
    oracle_limit: str | None = None
    for candidate_index, candidate_state in enumerate(state_texts):
        if not isinstance(candidate_state, str):
            continue
        rendered = json.loads(
            pokezero_search.branch_events(candidate_state, side_one_choice, side_two_choice, context, True, True)
        )
        branches = rendered.get("branches") if isinstance(rendered, Mapping) else None
        if not isinstance(branches, Sequence):
            continue
        for branch in branches:
            if not isinstance(branch, Mapping):
                continue
            events = branch.get("events")
            legal_state = branch.get("legal_roll_state")
            if not isinstance(events, Sequence) or not isinstance(legal_state, str):
                continue
            direct_damage = _branch_direct_damage(events, direct.target)
            if direct_damage is None:
                continue
            if native_rolls is None:
                state = poke_engine.State.from_string(legal_state)
                raw = poke_engine.calculate_damage(state, side_one_choice, side_two_choice, True)
                # The binding returns one ``[normal, critical]`` pair per acting
                # side.  The direct protocol event identifies which choice actually
                # landed, so never flatten both sides into an ambiguous pair.
                actor_index = 0 if direct.actor == "p1" else 1
                actor_rolls = raw[actor_index]
                if len(actor_rolls) != 2:
                    return {
                        "verdict": "comparison_limit",
                        "reason": "native_damage_binding_missing_normal_critical_pair",
                    }
                native_rolls = tuple(int(value) for value in actor_rolls)
                oracle_context, oracle_rolls, oracle_limit = _basic_oracle(
                    state=state, direct=direct, dex=dex
                )
            branch_rows.append(
                {
                    "candidate_index": candidate_index,
                    "percentage": float(branch.get("percentage") or 0.0),
                    "direct_damage": direct_damage,
                    "target_fainted": _branch_target_fainted(events, direct.target),
                    "has_observed_secondary": bool(
                        direct.secondary_status
                        and _branch_has_status(events, direct.target, direct.secondary_status)
                    ),
                    "lossy": list(branch.get("lossy") or []),
                }
            )
    if not branch_rows:
        return {"verdict": "comparison_limit", "reason": "no_rendered_direct_damage_branch"}
    observed_in_oracle = bool(oracle_rolls and direct.damage in oracle_rolls)
    native_max = None
    native_representative = None
    if native_rolls is not None:
        native_max = native_rolls[1] if direct.critical else native_rolls[0]
        native_representative = int(native_max * 0.925)
    all_branch_damage = sorted({int(item["direct_damage"]) for item in branch_rows})
    nonterminal_damage = sorted(
        {int(item["direct_damage"]) for item in branch_rows if not item["target_fainted"]}
    )
    secondary_rows = [row for row in branch_rows if row["has_observed_secondary"]]
    coupled_observed_damage = any(int(row["direct_damage"]) == direct.damage for row in secondary_rows)
    if oracle_rolls is None:
        verdict = "comparison_limit"
    elif tuple(oracle_rolls)[-1] != native_max:
        verdict = "native_arithmetic_disagreement"
    elif not observed_in_oracle:
        verdict = "showdown_outside_transcribed_oracle"
    elif len(nonterminal_damage) == 1 and direct.damage != nonterminal_damage[0]:
        # The raw generator uses one representative (average-like) damage
        # instruction, while the binding exposes the true maximum separately.
        # A state-local rounding detail can make that representative differ by
        # one from ``floor(max * .925)``; the shared fact that matters is that
        # the branch fan-out has only ONE nonterminal damage value and cannot
        # pair the observed legal roll with its observed secondary effect.
        verdict = "fixed_single_roll_composition"
    else:
        verdict = "no_arithmetic_disagreement"
    return {
        "verdict": verdict,
        "observed": {
            "actor": direct.actor,
            "target": direct.target,
            "move": direct.move,
            "damage": direct.damage,
            "critical": direct.critical,
            "secondary_status": direct.secondary_status,
        },
        "oracle_context": oracle_context,
        "oracle_rolls": list(oracle_rolls) if oracle_rolls is not None else None,
        "oracle_limit": oracle_limit,
        "observed_in_oracle": observed_in_oracle if oracle_rolls is not None else None,
        "native_rolls": list(native_rolls) if native_rolls is not None else None,
        "native_representative_damage": native_representative,
        "native_max_equals_oracle_max": (
            tuple(oracle_rolls)[-1] == native_max if oracle_rolls is not None and native_max is not None else None
        ),
        "rendered_direct_damages": all_branch_damage,
        "rendered_nonterminal_direct_damages": nonterminal_damage,
        "secondary_branch_count": len(secondary_rows),
        "secondary_branch_has_observed_damage": coupled_observed_damage if direct.secondary_status else None,
        "branches": branch_rows,
    }


def _rows(paths: Iterable[Path], targets: set[tuple[int, int]]) -> list[Mapping[str, Any]]:
    matches: list[Mapping[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("repros") or []:
            if not isinstance(row, Mapping):
                continue
            identity = (int(row.get("seed") or -1), int(row.get("step") or -1))
            if identity in targets:
                matches.append(row)
    return matches


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--target", type=_target, action="append", required=True)
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    assert_fresh()
    targets = set(args.target)
    rows = _rows(args.report, targets)
    by_identity = {(int(row["seed"]), int(row["step"])): row for row in rows}
    dex = load_showdown_dex(args.showdown_root)
    results: list[dict[str, object]] = []
    for identity in sorted(targets):
        row = by_identity.get(identity)
        if row is None:
            results.append({"seed": identity[0], "step": identity[1], "verdict": "comparison_limit", "reason": "target_not_retained"})
            continue
        direct = observed_direct_hit(row)
        if direct is None:
            results.append({"seed": identity[0], "step": identity[1], "verdict": "comparison_limit", "reason": "no_standard_direct_hit"})
            continue
        result = _branch_report(row, direct, dex)
        result.update({"seed": identity[0], "step": identity[1]})
        results.append(result)
    payload = {
        "schema_version": "pokezero.damage-arithmetic-tail-attestation/v1",
        "targets": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
