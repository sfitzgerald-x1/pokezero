"""Track C (engine swap plan v3): Showdown vs poke-engine fidelity differential.

One-turn differential: run a curated (teams, choices, seed) fixture through the
real Showdown sim (:mod:`pokezero.showdown_fixture` oracle) and enumerate the
same joint action's outcome branches in poke-engine
(``generate_instructions``); the observed Showdown outcome must appear in the
engine's branch support.

Matching rules:

- statuses, faints, weather, and side-condition **presence** match exactly;
- active HP matches within a relative tolerance band, because poke-engine
  branches carry one representative (average) damage roll per (crit,
  secondary) combination while Showdown samples a concrete roll from the
  0.85-1.0 spread;
- everything else in the branch (probability) is recorded, not asserted.

The deliverable is the report: per-mechanic match rates, and for every
divergence a self-contained repro (engine state string, choices, branch
features, Showdown protocol) so a mismatch becomes a fixture, not an anecdote.

This module owns no search or training behavior; it is measurement only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .dex import ShowdownDex, load_showdown_dex, normalize_id
from .engine_world import _build_pokemon_spec, hidden_power_engine_id  # shared construction path
from .poke_engine_adapter import BattleSpec, SideSpec, build_poke_engine_state
from .showdown_fixture import DEFAULT_GEN3_CUSTOM_FORMAT, FixturePokemon, OneTurnFixtureResult

# Showdown protocol status codes -> poke-engine status names (upper-cased by the
# binding's accessors).
_STATUS_TO_ENGINE = {
    "": "NONE",
    "par": "PARALYZE",
    "brn": "BURN",
    "psn": "POISON",
    "tox": "TOXIC",
    "slp": "SLEEP",
    "frz": "FREEZE",
}

_WEATHER_TO_ENGINE = {
    "": "NONE",
    "none": "NONE",
    "sandstorm": "SAND",
    "raindance": "RAIN",
    "sunnyday": "SUN",
    "hail": "HAIL",
}

# Engine SideConditions fields checked for presence parity with |-sidestart|.
_SIDE_CONDITION_FIELDS = ("spikes", "reflect", "light_screen", "safeguard", "mist")
_SIDESTART_IDS = {
    "spikes": "spikes",
    "reflect": "reflect",
    "lightscreen": "light_screen",
    "light screen": "light_screen",
    "safeguard": "safeguard",
    "mist": "mist",
}

# Relative HP tolerance for damage matching: engine branches carry the average
# roll (~0.925 of base); Showdown samples 0.85-1.0, so the max roll lands ~8.1%
# above the representative and rounding of residuals stacks on top (observed
# up to ~13% on large hits like Explosion). 16% keeps roll spread inside the
# band while still catching wrong-mechanic deltas (wrong residual fraction,
# wrong type effectiveness) which move HP by 50%+.
_DAMAGE_TOLERANCE = 0.16
_MIN_TOLERANCE_HP = 5


@dataclass(frozen=True)
class FidelityCase:
    """One curated differential case (single turn, move choices only)."""

    name: str
    mechanic: str
    p1_team: Sequence[FixturePokemon]
    p2_team: Sequence[FixturePokemon]
    p1_move: str
    p2_move: str
    seeds: Sequence[int] = (11, 12, 13, 14, 15, 16, 17, 18)
    notes: str = ""
    # Initial engine-state weather for entry effects the one-turn oracle applies
    # before the fixture turn (e.g. Sand Stream on switch-in). ``-1`` turns =
    # indefinite, matching Gen 3 ability weather.
    spec_weather: str = "none"


@dataclass
class TurnFeatures:
    """Comparable outcome features of one resolved turn."""

    p1_hp: int
    p2_hp: int
    p1_status: str
    p2_status: str
    fainted: frozenset[str]
    weather: str
    side_conditions: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def presence(self) -> dict[str, tuple[str, ...]]:
        return {
            side: tuple(sorted(k for k, v in conditions.items() if v))
            for side, conditions in self.side_conditions.items()
        }


# ---------------------------------------------------------------------------------------------
# Showdown side: protocol -> features.
# ---------------------------------------------------------------------------------------------


def showdown_turn_features(result: OneTurnFixtureResult) -> TurnFeatures:
    """Fold the omniscient protocol into comparable end-of-turn features."""

    hp: dict[str, int] = {}
    status: dict[str, str] = {}
    fainted: set[str] = set()
    weather = ""
    side_conditions: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
    actives: dict[str, str] = {}

    for line in result.protocol_lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag == "switch" and len(parts) >= 4:
            slot = parts[2].split(":", 1)[0].strip()  # "p1a"
            side = slot[:2]
            actives[side] = slot
            hp[side], status[side] = _parse_condition(parts[4] if len(parts) > 4 else parts[3])
        elif tag in ("-damage", "-heal") and len(parts) >= 4:
            slot = parts[2].split(":", 1)[0].strip()
            side = slot[:2]
            if actives.get(side) == slot:
                hp[side], status[side] = _parse_condition(parts[3])
        elif tag == "-status" and len(parts) >= 4:
            side = parts[2].split(":", 1)[0].strip()[:2]
            status[side] = parts[3].strip()
        elif tag == "-curestatus" and len(parts) >= 3:
            side = parts[2].split(":", 1)[0].strip()[:2]
            status[side] = ""
        elif tag == "faint" and len(parts) >= 3:
            side = parts[2].split(":", 1)[0].strip()[:2]
            fainted.add(side)
            hp[side] = 0
        elif tag == "-weather" and len(parts) >= 3:
            token = normalize_id(parts[2])
            weather = "" if token == "none" else token
        elif tag == "-sidestart" and len(parts) >= 4:
            side = parts[2].split(":", 1)[0].strip()[:2]
            condition = _SIDESTART_IDS.get(normalize_id(parts[3].split(":")[-1]))
            if condition:
                counts = side_conditions[side]
                counts[condition] = counts.get(condition, 0) + 1
        elif tag == "-sideend" and len(parts) >= 4:
            side = parts[2].split(":", 1)[0].strip()[:2]
            condition = _SIDESTART_IDS.get(normalize_id(parts[3].split(":")[-1]))
            if condition:
                side_conditions[side][condition] = 0

    return TurnFeatures(
        p1_hp=hp.get("p1", -1),
        p2_hp=hp.get("p2", -1),
        p1_status=_STATUS_TO_ENGINE.get(status.get("p1", ""), f"?{status.get('p1')}"),
        p2_status=_STATUS_TO_ENGINE.get(status.get("p2", ""), f"?{status.get('p2')}"),
        fainted=frozenset(fainted),
        weather=_WEATHER_TO_ENGINE.get(weather, f"?{weather}"),
        side_conditions=side_conditions,
    )


def _parse_condition(condition: str) -> tuple[int, str]:
    condition = condition.strip()
    hp_part, _, status_part = condition.partition(" ")
    if status_part.strip() == "fnt" or hp_part == "0":
        return 0, ""
    current, _, _ = hp_part.partition("/")
    try:
        return int(current), status_part.strip()
    except ValueError:
        return -1, status_part.strip()


# ---------------------------------------------------------------------------------------------
# Engine side: fixture teams -> BattleSpec -> branch features.
# ---------------------------------------------------------------------------------------------


def fixture_battle_spec(
    p1_team: Sequence[FixturePokemon],
    p2_team: Sequence[FixturePokemon],
    *,
    dex: ShowdownDex,
    weather: str = "none",
) -> BattleSpec:
    """Fresh-battle spec for curated teams (full HP, catalog PP, no overlay)."""

    def side(team: Sequence[FixturePokemon], slot: str) -> SideSpec:
        party = tuple(
            _build_pokemon_spec(mon, None, dex=dex, slot=slot, is_self=False)
            for mon in team
        )
        return SideSpec(pokemon=party, active_index=0)

    return BattleSpec(
        side_one=side(p1_team, "p1"),
        side_two=side(p2_team, "p2"),
        weather=weather,
        weather_turns_remaining=-1,
    )


def engine_state_features(state: Any) -> TurnFeatures:
    """Fold an engine state (initial or branch-applied) into comparable features.

    Shared with the multi-turn differential (:mod:`pokezero.engine_fidelity_multiturn`),
    which folds every intermediate state of a followed trajectory, not just
    one-turn branch outcomes.
    """

    p1_active = state.side_one.pokemon[int(str(state.side_one.active_index))]
    p2_active = state.side_two.pokemon[int(str(state.side_two.active_index))]
    fainted = frozenset(
        side for side, mon in (("p1", p1_active), ("p2", p2_active)) if mon.hp <= 0
    )
    return TurnFeatures(
        p1_hp=int(p1_active.hp),
        p2_hp=int(p2_active.hp),
        p1_status=str(p1_active.status).upper(),
        p2_status=str(p2_active.status).upper(),
        fainted=fainted,
        weather=str(state.weather).upper(),
        side_conditions={
            "p1": _engine_side_conditions(state.side_one),
            "p2": _engine_side_conditions(state.side_two),
        },
    )


def engine_branch_features(state: Any, p1_move: str, p2_move: str, *, module: Any) -> list[dict[str, Any]]:
    """Enumerate (probability, TurnFeatures) for the joint action's branches."""

    branches = module.generate_instructions(state, normalize_id(p1_move), normalize_id(p2_move))
    rows: list[dict[str, Any]] = []
    for branch in branches:
        applied = state.apply_instructions(branch)
        rows.append({
            "percentage": float(branch.percentage),
            "features": engine_state_features(applied),
            "raw": str(branch.instruction_list),
        })
    return rows


def _engine_side_conditions(side: Any) -> dict[str, int]:
    conditions = getattr(side, "side_conditions", None)
    if conditions is None:
        return {}
    return {name: int(getattr(conditions, name, 0) or 0) for name in _SIDE_CONDITION_FIELDS}


# ---------------------------------------------------------------------------------------------
# Matching.
# ---------------------------------------------------------------------------------------------


def match_branch(
    observed: TurnFeatures,
    branches: Sequence[Mapping[str, Any]],
    *,
    p1_start_hp: int,
    p2_start_hp: int,
) -> tuple[Mapping[str, Any] | None, list[str]]:
    """Find a branch consistent with the observed turn; else explain the misses."""

    reasons: list[str] = []
    for row in branches:
        candidate: TurnFeatures = row["features"]
        mismatch = _mismatch_reason(
            observed, candidate, p1_start_hp=p1_start_hp, p2_start_hp=p2_start_hp
        )
        if mismatch is None:
            return row, []
        reasons.append(f"pct={row['percentage']:.2f}: {mismatch}")
    return None, reasons


def _mismatch_reason(
    observed: TurnFeatures,
    branch: TurnFeatures,
    *,
    p1_start_hp: int,
    p2_start_hp: int,
) -> str | None:
    if observed.p1_status != branch.p1_status:
        return f"p1 status {observed.p1_status} != {branch.p1_status}"
    if observed.p2_status != branch.p2_status:
        return f"p2 status {observed.p2_status} != {branch.p2_status}"
    if observed.fainted != branch.fainted:
        return f"fainted {sorted(observed.fainted)} != {sorted(branch.fainted)}"
    if observed.weather != branch.weather:
        return f"weather {observed.weather} != {branch.weather}"
    if observed.presence() != branch.presence():
        return f"side conditions {observed.presence()} != {branch.presence()}"
    for side, obs_hp, br_hp, start in (
        ("p1", observed.p1_hp, branch.p1_hp, p1_start_hp),
        ("p2", observed.p2_hp, branch.p2_hp, p2_start_hp),
    ):
        if obs_hp < 0:
            continue
        if obs_hp == 0 or br_hp == 0:
            if obs_hp != br_hp:
                return f"{side} hp {obs_hp} != {br_hp} (faint boundary)"
            continue
        damage_scale = max(abs(start - br_hp), _MIN_TOLERANCE_HP)
        if abs(obs_hp - br_hp) > max(_MIN_TOLERANCE_HP, _DAMAGE_TOLERANCE * damage_scale):
            return f"{side} hp {obs_hp} outside tolerance of {br_hp} (start {start})"
    return None


# ---------------------------------------------------------------------------------------------
# Curated cases (Gen 3 randbats-reachable mechanics first).
# ---------------------------------------------------------------------------------------------


def _mon(species: str, moves: Sequence[str], *, level: int = 80, ability: str | None = None,
         item: str | None = "Leftovers", ivs: Mapping[str, int] | None = None) -> FixturePokemon:
    return FixturePokemon(
        species=species, moves=tuple(moves), ability=ability, item=item, level=level,
        evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")}, ivs=ivs,
    )


def curated_cases() -> tuple[FidelityCase, ...]:
    swampert = _mon("Swampert", ["Earthquake", "Ice Beam", "Protect", "Toxic"], ability="Torrent")
    snorlax = _mon("Snorlax", ["Body Slam", "Shadow Ball", "Rest", "Curse"], ability="Immunity")
    gengar = _mon("Gengar", ["Thunderbolt", "Will-O-Wisp", "Explosion", "Ice Punch"], ability="Levitate")
    skarmory = _mon("Skarmory", ["Spikes", "Drill Peck", "Roar", "Protect"], ability="Keen Eye")
    celebi = _mon("Celebi", ["Leech Seed", "Psychic", "Recover", "Calm Mind"], ability="Natural Cure")
    tyranitar = _mon("Tyranitar", ["Rock Slide", "Earthquake", "Crunch", "Pursuit"], ability="Sand Stream")
    starmie = _mon("Starmie", ["Surf", "Psychic", "Thunder Wave", "Recover"], ability="Natural Cure")
    jirachi = _mon("Jirachi", ["Reflect", "Light Screen", "Psychic", "Wish"], ability="Serene Grace")
    hp_grass_swampert_check = _mon(
        "Starmie", ["Hidden Power", "Recover", "Surf", "Psychic"], ability="Natural Cure",
        ivs={"hp": 31, "atk": 30, "def": 31, "spa": 30, "spd": 31, "spe": 31},  # HP Grass
    )

    return (
        FidelityCase("basic_damage", "damage", [swampert], [snorlax], "move earthquake", "move bodyslam",
                     notes="STAB EQ vs Body Slam; crit and 30% para branches"),
        FidelityCase("ground_immunity", "immunity", [swampert], [gengar], "move earthquake", "move thunderbolt",
                     notes="EQ into Levitate = no damage; TBolt immune into ground? (no: Swampert immune to TBolt)"),
        FidelityCase("toxic_status", "status", [swampert], [snorlax], "move toxic", "move curse",
                     notes="Toxic application (Immunity blocks? — Snorlax Immunity: poison-proof; expect no status)"),
        FidelityCase("toxic_residual", "status", [swampert], [starmie], "move toxic", "move recover",
                     notes="Toxic lands; first residual 1/16 end of turn"),
        FidelityCase("burn_application", "status", [gengar], [snorlax], "move willowisp", "move curse",
                     notes="75% burn chance in gen3; branch support must include miss"),
        FidelityCase("thunder_wave_para", "status", [starmie], [snorlax], "move thunderwave", "move bodyslam",
                     notes="Para + full-para branch on the same turn"),
        FidelityCase("spikes_set", "side_condition", [skarmory], [swampert], "move spikes", "move icebeam",
                     notes="Spikes layer 1 sidestart parity"),
        FidelityCase("reflect_set", "side_condition", [jirachi], [snorlax], "move reflect", "move bodyslam",
                     notes="Faster Jirachi sets the screen first, so same-turn physical halving is exercised"),
        FidelityCase("light_screen_set", "side_condition", [jirachi], [starmie], "move lightscreen", "move surf",
                     notes="Special screen halving"),
        FidelityCase("leech_seed_drain", "residual", [celebi], [swampert], "move leechseed", "move earthquake",
                     notes="Seed + drain routing to seeder same turn"),
        FidelityCase("sand_stream_chip", "weather", [tyranitar], [starmie], "move rockslide", "move surf",
                     spec_weather="sand",
                     notes="Sand Stream fires on entry before the fixture turn, so the engine state starts in sand"),
        FidelityCase("explosion_faints", "faint", [gengar], [snorlax], "move explosion", "move bodyslam",
                     notes="Attacker faints; Normal-immune Gengar ordering; defense halving in gen3"),
        FidelityCase("protect_blocks", "protect", [swampert], [snorlax], "move protect", "move bodyslam",
                     notes="Protect negates; no damage branches for p1"),
        FidelityCase("hidden_power_type", "hidden_power", [hp_grass_swampert_check], [swampert], "move hiddenpower", "move icebeam",
                     notes="HP Grass IVs -> 4x on Swampert; engine move id mapping for hiddenpower"),
        FidelityCase("rest_full_heal", "sleep", [swampert], [snorlax], "move earthquake", "move rest",
                     notes="Damaged Snorlax rests after being hit: full heal + SLEEP status. Rest/sleep "
                           "turn COUNTS are invisible to TurnFeatures and not asserted here"),
    )


# ---------------------------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------------------------


def run_case(
    case: FidelityCase,
    *,
    dex: ShowdownDex,
    module: Any,
    config: Any | None = None,
) -> dict[str, Any]:
    """Run one curated case across its seeds; return the per-case report row."""

    from .showdown_fixture import run_one_turn_fixture

    spec = fixture_battle_spec(case.p1_team, case.p2_team, dex=dex, weather=case.spec_weather)
    state = build_poke_engine_state(spec, module=module)
    p1_start = spec.side_one.pokemon[0].hp
    p2_start = spec.side_two.pokemon[0].hp

    def engine_move(choice: str, mon: FixturePokemon) -> str:
        move_id = normalize_id(choice.split(None, 1)[1])
        if move_id.startswith("hiddenpower"):
            # Same translation the production world constructor applies.
            return hidden_power_engine_id(move_id, mon.ivs)
        return move_id

    try:
        branches = engine_branch_features(
            state,
            engine_move(case.p1_move, case.p1_team[0]),
            engine_move(case.p2_move, case.p2_team[0]),
            module=module,
        )
    except Exception as error:  # engine-side hard failure IS a finding
        return {
            "case": case.name, "mechanic": case.mechanic, "status": "engine_error",
            "error": f"{type(error).__name__}: {error}", "engine_state": state.to_string(),
        }

    seeds_report = []
    matched = 0
    for seed in case.seeds:
        result = run_one_turn_fixture(
            p1_team=case.p1_team, p2_team=case.p2_team,
            p1_choice=case.p1_move, p2_choice=case.p2_move,
            seed=seed, config=config,
        )
        if result.error_lines:
            seeds_report.append({"seed": seed, "status": "showdown_error", "errors": list(result.error_lines)})
            continue
        observed = showdown_turn_features(result)
        row, misses = match_branch(observed, branches, p1_start_hp=p1_start, p2_start_hp=p2_start)
        if row is not None:
            matched += 1
            seeds_report.append({"seed": seed, "status": "matched", "branch_pct": row["percentage"]})
        else:
            seeds_report.append({
                "seed": seed, "status": "diverged",
                "observed": _features_payload(observed),
                "branch_misses": misses,
                "protocol": list(result.protocol_lines),
            })
    return {
        "case": case.name, "mechanic": case.mechanic, "notes": case.notes,
        "status": "ok" if matched == len(case.seeds) else "diverged",
        "matched": matched, "total": len(case.seeds),
        "engine_state": state.to_string(),
        "engine_branches": [
            {"pct": row["percentage"], **_features_payload(row["features"]), "raw": row["raw"]}
            for row in branches
        ],
        "seeds": seeds_report,
    }


def _features_payload(features: TurnFeatures) -> dict[str, Any]:
    return {
        "p1_hp": features.p1_hp, "p2_hp": features.p2_hp,
        "p1_status": features.p1_status, "p2_status": features.p2_status,
        "fainted": sorted(features.fainted), "weather": features.weather,
        "side_conditions": features.presence(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Showdown vs poke-engine one-turn fidelity differential")
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default=None, help="run only cases whose name or mechanic contains this")
    args = parser.parse_args(argv)

    import poke_engine

    from .local_showdown import LocalShowdownConfig

    dex = load_showdown_dex(args.showdown_root)
    config = LocalShowdownConfig(showdown_root=args.showdown_root)
    cases = [
        case for case in curated_cases()
        if args.only is None or args.only in case.name or args.only in case.mechanic
    ]
    rows = []
    for case in cases:
        row = run_case(case, dex=dex, module=poke_engine, config=config)
        rows.append(row)
        print(f"{row['case']}: {row['status']} ({row.get('matched', 0)}/{row.get('total', 0)})")

    diverged = [r for r in rows if r["status"] != "ok"]
    report = {
        "cases": len(rows),
        "clean": len(rows) - len(diverged),
        "diverged": [r["case"] for r in diverged],
        "results": rows,
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"{report['clean']}/{report['cases']} cases clean -> {args.out}")
    return 1 if diverged else 0


if __name__ == "__main__":
    raise SystemExit(main())
