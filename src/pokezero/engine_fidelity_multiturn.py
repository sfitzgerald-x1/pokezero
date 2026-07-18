"""Track C tier-2 wave 1: multi-turn Showdown vs poke-engine fidelity differential.

Extends :mod:`pokezero.engine_fidelity` from one isolated turn to short scripted
trajectories (3-7 decision boundaries). Per step, the same joint action is run
through the real Showdown sim (:func:`pokezero.showdown_fixture.run_multi_turn_fixture`)
and enumerated in poke-engine (``generate_instructions``); the observed step must
land in the engine's branch support, and the engine then CONTINUES from the
matched branch's applied state (``apply_instructions`` returns a new state — the
binding is immutable-style). A miss diverges the case at that step and records a
self-contained repro; later steps are not scored.

What multi-turn adds over the one-turn harness:

- timed counts, not just presence: engine screen counters ticking down per turn
  and screen EXPIRY (damage un-halves after turn 5), toxic escalation
  (1/16 -> 2/16 -> 3/16), rest/sleep-talk wake timing — validated both through
  the differential itself (a wrong counter changes damage/status and misses the
  branch support) and through per-step engine counter traces asserted on
  fully-matched seeds;
- mid-turn force-switch boundaries (Baton Pass): only one seat acts, and the
  engine resolution call must RE-SUPPLY the waiting seat's saved move
  (``side.switch_out_move_second_saved_move``) — passing ``"none"`` makes the
  engine silently drop that move (caller-contract sharp edge, see findings doc);
- drift correction: Showdown samples concrete damage rolls while the followed
  engine branch carries the representative (average) roll, so absolute HP drifts
  apart over turns even when every mechanic matches. Observed HP is shifted by
  the accumulated (engine - showdown) offset per side before matching, which
  reduces the per-step comparison to this step's delta. The offset is reset for
  a side whenever its active changes (fresh mons start in sync).

This module owns no search or training behavior; it is measurement only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .dex import ShowdownDex, load_showdown_dex, normalize_id
from .engine_fidelity import (
    TurnFeatures,
    _features_payload,
    _mon,
    engine_state_features,
    fixture_battle_spec,
    match_branch,
    showdown_turn_features,
)
from .engine_world import hidden_power_engine_id
from .poke_engine_adapter import MoveSpec, build_poke_engine_state
from .showdown_fixture import (
    FixturePokemon,
    MultiTurnFixtureResult,
    OneTurnFixtureResult,
    run_multi_turn_fixture,
)

# Engine "no action" move string (waiting seat of a force-switch boundary whose
# saved move has already resolved or never existed).
_ENGINE_NO_MOVE = "none"


@dataclass(frozen=True)
class MultiTurnCase:
    """One curated multi-turn differential case (scripted boundary sequence)."""

    name: str
    mechanic: str
    p1_team: Sequence[FixturePokemon]
    p2_team: Sequence[FixturePokemon]
    # Showdown choice strings per boundary; None = seat is waiting (mid-turn
    # force switch on the other side). "switch N" uses ORIGINAL team order.
    turns: Sequence[tuple[str | None, str | None]]
    seeds: Sequence[int] = (21, 22, 23, 24)
    notes: str = ""
    spec_weather: str = "none"
    # Engine timed-count expectations keyed by telemetry field (e.g.
    # "p1.reflect"), one value per step; checked only on fully-matched seeds
    # whose trajectory stayed on the deterministic path the trace encodes.
    expected_traces: Mapping[str, Sequence[int]] = field(default_factory=dict)
    # Non-empty = shipped-but-skipped with a precise statement of what is missing.
    skip_reason: str = ""


# ---------------------------------------------------------------------------------------------
# Engine side: choices, telemetry, drift correction.
# ---------------------------------------------------------------------------------------------


def engine_step_choices(
    state: Any,
    step_choices: Mapping[str, str],
    *,
    p1_team: Sequence[FixturePokemon],
    p2_team: Sequence[FixturePokemon],
) -> tuple[str, str]:
    """Translate one boundary's Showdown choices into engine move strings.

    - ``move X`` -> engine move id (Hidden Power translated via IVs, mirroring
      the one-turn harness and the production world constructor);
    - ``switch N`` (1-based, original team order) -> bare species id when that
      side is resolving a force switch, ``switch <species>`` otherwise;
    - an absent choice (waiting seat) -> the side's saved move re-supplied, or
      ``"none"`` when there is none. poke-engine postpones the slower seat's
      move across a Baton Pass switch-out and only executes it if the resolution
      call passes it again; ``"none"`` silently drops it.
    """

    engine_moves = []
    for player, side, team in (("p1", state.side_one, p1_team), ("p2", state.side_two, p2_team)):
        choice = step_choices.get(player)
        if choice is None:
            saved = normalize_id(str(side.switch_out_move_second_saved_move))
            engine_moves.append(saved if saved and saved != "none" else _ENGINE_NO_MOVE)
            continue
        kind, _, argument = choice.strip().partition(" ")
        if kind == "move":
            move_id = normalize_id(argument)
            if move_id.startswith("hiddenpower"):
                active = team[int(str(side.active_index))]
                move_id = hidden_power_engine_id(move_id, active.ivs)
            engine_moves.append(move_id)
        elif kind == "switch":
            species = normalize_id(team[int(argument) - 1].species)
            engine_moves.append(species if side.force_switch else f"switch {species}")
        else:
            raise ValueError(f"Unsupported fixture choice for {player}: {choice!r}")
    return engine_moves[0], engine_moves[1]


# Per-step engine telemetry: the timed counts the one-turn harness cannot see.
_TELEMETRY_SIDE_CONDITIONS = ("reflect", "light_screen", "spikes", "toxic_count")
_TELEMETRY_BOOSTS = ("attack_boost", "special_attack_boost")


def engine_step_telemetry(state: Any) -> dict[str, int]:
    """Timed/hidden engine counters after a step, keyed ``p1.<field>``/``p2.<field>``."""

    telemetry: dict[str, int] = {}
    for label, side in (("p1", state.side_one), ("p2", state.side_two)):
        active = side.pokemon[int(str(side.active_index))]
        for name in _TELEMETRY_SIDE_CONDITIONS:
            telemetry[f"{label}.{name}"] = int(getattr(side.side_conditions, name, 0) or 0)
        for name in _TELEMETRY_BOOSTS:
            telemetry[f"{label}.{name}"] = int(getattr(side, name, 0) or 0)
        telemetry[f"{label}.rest_turns"] = int(active.rest_turns)
        telemetry[f"{label}.sleep_turns"] = int(active.sleep_turns)
        telemetry[f"{label}.encored"] = int("ENCORE" in {str(v).upper() for v in side.volatile_statuses})
    return telemetry


# Showdown protocol boost stat ids -> engine Side boost field prefixes.
_BOOST_IDS = {
    "atk": "attack", "def": "defense", "spa": "special_attack",
    "spd": "special_defense", "spe": "speed", "accuracy": "accuracy", "evasion": "evasion",
}
_ENGINE_BOOST_FIELDS = tuple(f"{name}_boost" for name in _BOOST_IDS.values())


def observed_boost_deltas(step_lines: Sequence[str]) -> dict[str, dict[str, int]]:
    """Per-side stat-stage deltas applied during one step (|-boost|/|-unboost|).

    Stat stages are invisible to :class:`TurnFeatures`, which makes branches that
    differ only in boosts (Sleep Talk calling Curse vs calling Rest, which
    no-ops) observationally identical and lets the trajectory bind to the wrong
    applied state. Per-step DELTAS disambiguate without tracking absolute stages
    across switches (Baton Pass keeps stages with no protocol echo; a regular
    switch clears them — both are delta-0 events on the untouched side).
    """

    deltas: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
    for line in step_lines:
        parts = line.split("|")
        if len(parts) < 5 or parts[1] not in ("-boost", "-unboost"):
            continue
        side = parts[2].split(":", 1)[0].strip()[:2]
        stat = _BOOST_IDS.get(parts[3].strip())
        if stat is None or side not in deltas:
            continue
        amount = int(parts[4])
        deltas[side][stat] = deltas[side].get(stat, 0) + (amount if parts[1] == "-boost" else -amount)
    return {side: {stat: v for stat, v in stats.items() if v} for side, stats in deltas.items()}


def engine_boost_deltas(before: Any, after: Any) -> dict[str, dict[str, int]]:
    """Per-side stat-stage deltas between an engine state and a branch's applied state."""

    deltas: dict[str, dict[str, int]] = {}
    for label, side_before, side_after in (
        ("p1", before.side_one, after.side_one),
        ("p2", before.side_two, after.side_two),
    ):
        stats: dict[str, int] = {}
        for name in _ENGINE_BOOST_FIELDS:
            delta = int(getattr(side_after, name, 0) or 0) - int(getattr(side_before, name, 0) or 0)
            if delta:
                stats[name[: -len("_boost")]] = delta
        deltas[label] = stats
    return deltas


def drift_adjusted(
    observed: TurnFeatures,
    *,
    showdown_prev: TurnFeatures,
    engine_prev: TurnFeatures,
    p1_active_changed: bool,
    p2_active_changed: bool,
) -> TurnFeatures:
    """Shift observed HP by the accumulated (engine - showdown) roll drift.

    The followed engine branch carries average damage rolls, so after N turns
    the absolute HP gap can exceed the one-turn tolerance band even with every
    mechanic correct. Shifting the observation by the pre-step offset makes the
    matcher compare THIS step's delta. Faint boundaries stay exact (0 is never
    shifted), and a side whose active changed this step is not adjusted — the
    incoming mon's HP is in sync with the engine's.
    """

    def adjust(obs_hp: int, prev_obs: int, prev_eng: int, active_changed: bool) -> int:
        if obs_hp <= 0 or active_changed or prev_obs < 0:
            return obs_hp
        return max(1, obs_hp + (prev_eng - prev_obs))

    return dataclasses.replace(
        observed,
        p1_hp=adjust(observed.p1_hp, showdown_prev.p1_hp, engine_prev.p1_hp, p1_active_changed),
        p2_hp=adjust(observed.p2_hp, showdown_prev.p2_hp, engine_prev.p2_hp, p2_active_changed),
    )


def step_changed_active(protocol_lines: Sequence[str], side: str) -> bool:
    """Whether this step's protocol swapped the given side's active Pokemon."""

    prefixes = (f"|switch|{side}a", f"|drag|{side}a", f"|replace|{side}a")
    return any(line.startswith(prefixes) for line in protocol_lines)


# ---------------------------------------------------------------------------------------------
# Showdown side: cumulative protocol fold.
# ---------------------------------------------------------------------------------------------


def cumulative_features(
    result: MultiTurnFixtureResult, lines: Sequence[str], step_lines: Sequence[str]
) -> TurnFeatures:
    """Fold the battle-so-far protocol into end-of-step features.

    Reuses the one-turn extractor over the CUMULATIVE prefix so running state
    (HP of a side untouched this step, persisting status/weather/screens) is
    carried instead of reported as unknown. ``fainted`` is folded per step:
    a faint is an event of the step it happens in, not a permanent flag.
    """

    folded = showdown_turn_features(_as_one_turn(result, lines))
    step_only = showdown_turn_features(_as_one_turn(result, step_lines))
    return dataclasses.replace(folded, fainted=step_only.fainted)


def _as_one_turn(result: MultiTurnFixtureResult, lines: Sequence[str]) -> OneTurnFixtureResult:
    return OneTurnFixtureResult(
        format_id=result.format_id,
        seed=result.seed,
        choices={},
        protocol_lines=tuple(lines),
        p1_request=None,
        p2_request=None,
        terminal=result.terminal,
    )


# ---------------------------------------------------------------------------------------------
# Curated multi-turn cases (tier-2 wave 1).
# ---------------------------------------------------------------------------------------------


def curated_multiturn_cases() -> tuple[MultiTurnCase, ...]:
    swampert = _mon("Swampert", ["Earthquake", "Ice Beam", "Protect", "Toxic"], ability="Torrent")
    jirachi = _mon("Jirachi", ["Reflect", "Calm Mind", "Psychic", "Wish"], ability="Serene Grace")
    snorlax_strength = _mon("Snorlax", ["Strength", "Body Slam", "Rest", "Curse"], ability="Immunity")
    snorlax_resttalk = _mon("Snorlax", ["Body Slam", "Rest", "Sleep Talk", "Curse"], ability="Immunity")
    snorlax_curse = _mon("Snorlax", ["Curse", "Body Slam", "Rest", "Sleep Talk"], ability="Immunity")
    skarmory = _mon("Skarmory", ["Drill Peck", "Spikes", "Roar", "Protect"], ability="Keen Eye")
    starmie = _mon("Starmie", ["Recover", "Surf", "Psychic", "Thunder Wave"], ability="Natural Cure")
    starmie_bare = _mon("Starmie", ["Recover", "Ice Beam", "Psychic", "Thunder Wave"],
                        ability="Natural Cure", item=None)
    celebi = _mon("Celebi", ["Calm Mind", "Baton Pass", "Psychic", "Recover"], ability="Natural Cure")
    tyranitar = _mon("Tyranitar", ["Curse", "Rock Slide", "Earthquake", "Crunch"], ability="Sand Stream")
    politoed = _mon("Politoed", ["Encore", "Surf", "Ice Beam", "Hypnosis"], ability="Water Absorb")
    ampharos = _mon("Ampharos", ["Growl", "Thunderbolt", "Thunder Wave", "Light Screen"], ability="Static")

    return (
        MultiTurnCase(
            "reflect_expiry", "screen_duration", [jirachi], [snorlax_strength],
            turns=[("move reflect", "move strength")] + [("move calmmind", "move strength")] * 6,
            expected_traces={"p1.reflect": (4, 3, 2, 1, 0, 0, 0)},
            notes="Reflect turn 1, same physical attack every turn: halved damage turns 2-5, "
                  "full damage turns 6-7 (crits pierce screens in gen3 and are separate "
                  "branches). Engine reflect counter must tick 5->0; expiry is validated by "
                  "the damage jump, not presence alone.",
        ),
        MultiTurnCase(
            "toxic_escalation", "residual_escalation", [swampert], [starmie],
            turns=[("move toxic", "move recover")] * 3,
            # Seeds screened so Toxic (85% acc) lands on step 1; the miss branch
            # is one-turn coverage, the escalating counter is what's new here.
            seeds=(22, 23, 25, 26),
            expected_traces={"p2.toxic_count": (1, 2, 3)},
            notes="Toxic then stall: residuals 1/16, 2/16, 3/16 with the patched Leftovers "
                  "ordering (heal BEFORE toxic damage) in both sims; Recover resets HP to "
                  "full so each step's residual fraction is exact.",
        ),
        MultiTurnCase(
            "resttalk_cycle", "sleep_talk", [skarmory], [snorlax_resttalk],
            turns=[
                ("move drillpeck", "move bodyslam"),
                ("move drillpeck", "move rest"),
                ("move drillpeck", "move sleeptalk"),
                ("move drillpeck", "move sleeptalk"),
                ("move drillpeck", "move sleeptalk"),
                ("move drillpeck", "move curse"),
            ],
            expected_traces={"p2.rest_turns": (0, 3, 2, 1, 0, 0)},
            notes="Damage turn, Rest (full heal + SLEEP + rest_turns=3), then Sleep Talk "
                  "turns (engine branches: called Body Slam / called Curse / called Rest "
                  "no-ops, 1/3 each) through the wake turn. Doubles as the Rest/Sleep Talk "
                  "PP-underflow canary (see pp_underflow_canary).",
        ),
        MultiTurnCase(
            "baton_pass_transfer", "baton_pass", [celebi, starmie], [snorlax_curse],
            turns=[
                ("move calmmind", "move curse"),
                ("move calmmind", "move curse"),
                ("move batonpass", "move curse"),
                ("switch 2", None),
                ("move surf", "move curse"),
            ],
            expected_traces={
                "p1.special_attack_boost": (1, 2, 2, 2, 2),
                # Step 3's Curse is postponed across the Baton Pass switch and
                # lands during the step-4 resolution.
                "p2.attack_boost": (1, 2, 2, 3, 4),
            },
            notes="Calm Mind x2 + Baton Pass: boosts must survive the mid-turn switch on the "
                  "engine side (side boost telemetry) AND on the Showdown side via the "
                  "step-5 boosted Surf (+2 doubles the damage — far outside the roll band).",
        ),
        MultiTurnCase(
            "encore_lock", "encore", [politoed], [ampharos],
            turns=[
                ("move surf", "move growl"),
                ("move encore", "move thunderbolt"),
                ("move surf", "move growl"),
            ],
            expected_traces={"p2.encored": (0, 1, 1)},
            notes="Ampharos' Growl becomes last_used_move on step 1 (the engine auto-tracks "
                  "it when Encore is in a moveset), Encore on step 2 must redirect the "
                  "already-chosen Thunderbolt to Growl (no 2x hit on Politoed), step 3 stays "
                  "locked (Showdown's request offers only Growl). Gen 3 Encore's random 2-6 "
                  "turn duration is NOT validated — the window here is the guaranteed prefix.",
        ),
        MultiTurnCase(
            "sand_chip_multi", "weather_residual", [tyranitar], [starmie_bare],
            turns=[
                ("move rockslide", "move icebeam"),
                ("move curse", "move recover"),
                ("move curse", "move recover"),
            ],
            spec_weather="sand",
            notes="Sand Stream chip 1/16 on itemless Starmie every turn; step 1's resisted "
                  "Ice Beam dents sand-immune Tyranitar so its Leftovers netting (including "
                  "the clamp back to full) is observable on steps 2-3. Starmie's Recover "
                  "resets to full each turn, so end-of-step HP exposes the weather residual "
                  "exactly.",
        ),
    )


# ---------------------------------------------------------------------------------------------
# PP-underflow canary (engine-only; attached to resttalk_cycle's report row).
# ---------------------------------------------------------------------------------------------


def pp_underflow_canary(*, dex: ShowdownDex, module: Any) -> dict[str, Any]:
    """Probe the historical Rest/Sleep Talk PP panic directly against the engine.

    Engine-only (no Showdown seat): a rest-talk Snorlax with Rest at 0 PP is
    forced through Rest, two Sleep Talk turns, and an MCTS burst — the paths
    that decrement the called move's PP. A pyo3 panic surfaces as a
    ``BaseException`` (``PanicException``), so anything raised here is caught
    and reported as a CONFIRMED engine bug with the repro state, never as a
    harness crash.
    """

    skarmory = _mon("Skarmory", ["Drill Peck", "Spikes", "Roar", "Protect"], ability="Keen Eye")
    snorlax = _mon("Snorlax", ["Body Slam", "Rest", "Sleep Talk", "Curse"], ability="Immunity")
    spec = fixture_battle_spec([skarmory], [snorlax], dex=dex)
    resting = spec.side_two.pokemon[0]
    resting = dataclasses.replace(
        resting,
        hp=resting.maxhp // 2,
        moves=tuple(
            MoveSpec(id=move.id, pp=0 if move.id == "rest" else move.pp, disabled=move.disabled)
            for move in resting.moves
        ),
    )
    spec = dataclasses.replace(
        spec, side_two=dataclasses.replace(spec.side_two, pokemon=(resting,))
    )
    state = build_poke_engine_state(spec, module=module)
    report: dict[str, Any] = {"panic": False}
    try:
        state = state.apply_instructions(module.generate_instructions(state, "protect", "rest")[0])
        active = state.side_two.pokemon[0]
        # The engine uses the 0-PP Rest and decrements PAST zero; record the
        # wraparound value (observed -1 on the patched 0.0.47 wheel).
        report["rest_pp_after_zero_pp_rest"] = int(active.moves[1].pp)
        report["status_after_rest"] = str(active.status).upper()
        for step in (1, 2):
            branches = module.generate_instructions(state, "drillpeck", "sleeptalk")
            report[f"sleeptalk_step_{step}_branches"] = len(branches)
            state = state.apply_instructions(branches[0])
        result = module.monte_carlo_tree_search(state, 200)
        report["mcts_visits"] = int(result.total_visits)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:  # pyo3 panics do NOT derive from Exception
        report["panic"] = True
        report["error"] = f"{type(error).__name__}: {error}"
        report["engine_state"] = state.to_string()
    return report


# ---------------------------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------------------------


def _engine_branch_rows(state: Any, p1_move: str, p2_move: str, *, module: Any) -> list[dict[str, Any]]:
    """Branch rows that also carry the applied state, for trajectory continuation."""

    rows: list[dict[str, Any]] = []
    for branch in module.generate_instructions(state, p1_move, p2_move):
        applied = state.apply_instructions(branch)
        rows.append({
            "percentage": float(branch.percentage),
            "features": engine_state_features(applied),
            "boost_deltas": engine_boost_deltas(state, applied),
            "applied": applied,
            "raw": str(branch.instruction_list),
        })
    return rows


def match_step_branch(
    observed: TurnFeatures,
    adjusted: TurnFeatures,
    branches: Sequence[Mapping[str, Any]],
    *,
    observed_boosts: Mapping[str, Mapping[str, int]],
    p1_start_hp: int,
    p2_start_hp: int,
) -> tuple[Mapping[str, Any] | None, str | None, list[str]]:
    """One step's branch match: boost-delta filter, then features (adjusted, then raw).

    Returns ``(row, variant, misses)``. The drift-adjusted observation is
    preferred (it removes accumulated roll drift); the raw observation is a
    fallback for sync events — a heal to full clamps both sims to max HP and
    zeroes the true drift, making the carried offset stale for exactly one step.
    A wrong-mechanic delta has to evade both bands, which stay far below
    mechanic-sized errors.
    """

    candidates = [row for row in branches if row["boost_deltas"] == observed_boosts]
    if not candidates:
        available = [f"pct={row['percentage']:.2f}: {row['boost_deltas']}" for row in branches]
        return None, None, [f"observed boost deltas {dict(observed_boosts)} not in branch support: {available}"]
    row, misses = match_branch(adjusted, candidates, p1_start_hp=p1_start_hp, p2_start_hp=p2_start_hp)
    if row is not None:
        return row, "drift_adjusted", []
    if adjusted != observed:
        raw_row, raw_misses = match_branch(
            observed, candidates, p1_start_hp=p1_start_hp, p2_start_hp=p2_start_hp
        )
        if raw_row is not None:
            return raw_row, "raw", []
        misses = [f"adjusted: {reason}" for reason in misses]
        misses += [f"raw: {reason}" for reason in raw_misses]
    return None, None, misses


def match_seed_trajectory(
    case: MultiTurnCase,
    result: MultiTurnFixtureResult,
    *,
    initial_state: Any,
    module: Any,
) -> dict[str, Any]:
    """Follow one seed's Showdown trajectory through the engine's branch support."""

    state = initial_state
    engine_prev = engine_state_features(state)
    showdown_prev = cumulative_features(result, result.initial_protocol_lines, ())
    cumulative: list[str] = list(result.initial_protocol_lines)
    steps_report: list[dict[str, Any]] = []
    seed_status = "matched"

    for index, step in enumerate(result.steps, start=1):
        cumulative.extend(step.protocol_lines)
        observed = cumulative_features(result, cumulative, step.protocol_lines)
        try:
            p1_move, p2_move = engine_step_choices(
                state, step.choices, p1_team=case.p1_team, p2_team=case.p2_team
            )
            branches = _engine_branch_rows(state, p1_move, p2_move, module=module)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # engine panic/rejection IS a finding
            seed_status = "engine_error"
            steps_report.append({
                "step": index, "status": "engine_error",
                "error": f"{type(error).__name__}: {error}",
                "choices": dict(step.choices),
                "engine_state": state.to_string(),
            })
            break
        adjusted = drift_adjusted(
            observed,
            showdown_prev=showdown_prev,
            engine_prev=engine_prev,
            p1_active_changed=step_changed_active(step.protocol_lines, "p1"),
            p2_active_changed=step_changed_active(step.protocol_lines, "p2"),
        )
        observed_boosts = observed_boost_deltas(step.protocol_lines)
        row, variant, misses = match_step_branch(
            observed, adjusted, branches,
            observed_boosts=observed_boosts,
            p1_start_hp=engine_prev.p1_hp, p2_start_hp=engine_prev.p2_hp,
        )
        if row is None:
            seed_status = "diverged"
            steps_report.append({
                "step": index, "status": "diverged",
                "choices": dict(step.choices),
                "engine_moves": [p1_move, p2_move],
                "observed": _features_payload(observed),
                "observed_drift_adjusted": _features_payload(adjusted),
                "observed_boost_deltas": observed_boosts,
                "branch_misses": misses,
                "engine_state": state.to_string(),
                "protocol": list(step.protocol_lines),
            })
            break
        state = row["applied"]
        engine_prev = row["features"]
        showdown_prev = observed
        steps_report.append({
            "step": index, "status": "matched",
            "branch_pct": row["percentage"],
            "matched_variant": variant,
            "telemetry": engine_step_telemetry(state),
        })
    else:
        if len(result.steps) < len(case.turns):
            seed_status = "ended_early"

    trace_mismatches = []
    if seed_status == "matched":
        trace_mismatches = check_expected_traces(case.expected_traces, steps_report)
        if trace_mismatches:
            seed_status = "trace_mismatch"
    return {
        "seed": result.seed,
        "status": seed_status,
        "steps_matched": sum(1 for row in steps_report if row["status"] == "matched"),
        "steps_total": len(case.turns),
        "trace_mismatches": trace_mismatches,
        "steps": steps_report,
    }


def check_expected_traces(
    expected_traces: Mapping[str, Sequence[int]], steps_report: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Compare engine counter traces of a fully-matched seed against expectations."""

    mismatches: list[str] = []
    for key, expected in expected_traces.items():
        actual = tuple(int(row["telemetry"].get(key, -1)) for row in steps_report)
        if actual != tuple(expected):
            mismatches.append(f"{key}: expected {tuple(expected)}, engine {actual}")
    return mismatches


def run_multiturn_case(
    case: MultiTurnCase,
    *,
    dex: ShowdownDex,
    module: Any,
    config: Any | None = None,
) -> dict[str, Any]:
    """Run one curated multi-turn case across its seeds; return the report row."""

    if case.skip_reason:
        return {"case": case.name, "mechanic": case.mechanic, "status": "skipped",
                "reason": case.skip_reason, "notes": case.notes}

    spec = fixture_battle_spec(case.p1_team, case.p2_team, dex=dex, weather=case.spec_weather)
    seeds_report = []
    matched = 0
    for seed in case.seeds:
        try:
            result = run_multi_turn_fixture(
                p1_team=case.p1_team, p2_team=case.p2_team,
                turns=case.turns, seed=seed, config=config,
            )
        except Exception as error:  # bad scripted choice / bridge error: report, don't kill the sweep
            seeds_report.append({"seed": seed, "status": "showdown_error",
                                 "errors": [f"{type(error).__name__}: {error}"]})
            continue
        if result.error_lines:
            seeds_report.append({"seed": seed, "status": "showdown_error",
                                 "errors": list(result.error_lines)})
            continue
        initial_state = build_poke_engine_state(spec, module=module)
        seed_row = match_seed_trajectory(case, result, initial_state=initial_state, module=module)
        if seed_row["status"] == "matched":
            matched += 1
        seeds_report.append(seed_row)

    row = {
        "case": case.name, "mechanic": case.mechanic, "notes": case.notes,
        "status": "ok" if matched == len(case.seeds) else "diverged",
        "matched": matched, "total": len(case.seeds),
        "seeds": seeds_report,
    }
    if case.name == "resttalk_cycle":
        row["pp_underflow_canary"] = pp_underflow_canary(dex=dex, module=module)
        if row["pp_underflow_canary"]["panic"]:
            row["status"] = "engine_error"
    return row


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Showdown vs poke-engine multi-turn fidelity differential (tier-2 wave 1)"
    )
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default=None, help="run only cases whose name or mechanic contains this")
    args = parser.parse_args(argv)

    import poke_engine

    from .local_showdown import LocalShowdownConfig

    dex = load_showdown_dex(args.showdown_root)
    config = LocalShowdownConfig(showdown_root=args.showdown_root)
    cases = [
        case for case in curated_multiturn_cases()
        if args.only is None or args.only in case.name or args.only in case.mechanic
    ]
    rows = []
    for case in cases:
        row = run_multiturn_case(case, dex=dex, module=poke_engine, config=config)
        rows.append(row)
        print(f"{row['case']}: {row['status']} ({row.get('matched', 0)}/{row.get('total', 0)})")

    not_clean = [r for r in rows if r["status"] not in ("ok", "skipped")]
    report = {
        "cases": len(rows),
        "clean": sum(1 for r in rows if r["status"] == "ok"),
        "skipped": [r["case"] for r in rows if r["status"] == "skipped"],
        "diverged": [r["case"] for r in not_clean],
        "results": rows,
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"{report['clean']}/{report['cases']} cases clean -> {args.out}")
    return 1 if not_clean else 0


if __name__ == "__main__":
    raise SystemExit(main())
