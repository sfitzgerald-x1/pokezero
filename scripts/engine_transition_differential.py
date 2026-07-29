#!/usr/bin/env python
"""Game-level Showdown-vs-poke-engine TRANSITION differential over fresh random games.

This is the tier-2 real-game sweep that ``docs/engine_fidelity_findings.md``
lists as "Next": instead of curated fixtures, it plays whole
``gen3randombattle`` games in the real Node sim with uniform-random legal
actions and, at EVERY full decision boundary (both seats act), asks one
question:

    does the transition Showdown actually took lie in the branch support
    ``poke_engine.generate_instructions`` enumerates for the same joint action,
    from the same state?

Design points that make this different from
:mod:`pokezero.engine_fidelity_multiturn`:

* **Fresh world per boundary, not a followed engine trajectory.** The engine
  state is rebuilt at every boundary through the PRODUCTION world constructor
  (``engine_world.world_battle_spec`` with the game's TRUE packed teams as a
  fixed ``BattleStartOverride`` — the omniscient world, no belief sampling).
  Roll drift therefore never accumulates, and the constructor itself is under
  test on live states rather than on hand-built specs.
* **Pre-state gate.** A boundary only receives a transition verdict when the
  constructed engine pre-state matches Showdown's observed pre-state exactly
  (active HP, status, weather, side-condition presence). Boundaries that fail
  the gate are reported separately as world-construction divergences, so
  constructor error is never charged to the engine's transition model (and
  never silently passes either).
* **Fail-closed skips are counted, not hidden.** ``EngineWorldUnsupported``
  reasons, single-seat (force-switch) boundaries and unmappable choices each
  get their own counted bucket.

Matching reuses the shipped matchers: exact boost-delta filter
(:func:`pokezero.engine_fidelity_multiturn.observed_boost_deltas`), then exact
status/faint/weather/side-condition equality with the +/-16%-of-this-turn's-
damage HP band (:func:`pokezero.engine_fidelity._mismatch_reason`). Because the
pre-state is exact, the band scales to a single turn's damage — but it is still
a band, and a sub-band systematic damage error would pass (documented limit,
inherited from the one-turn harness).

Read-only: no training, no search, no production behavior. Measurement only.

Usage::

    PYTHONPATH=src python scripts/engine_transition_differential.py \\
        --showdown-root <showdown> --games 200 --seed-start 900000 \\
        --json report.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import types
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import poke_engine  # noqa: E402

from pokezero.dex import load_showdown_dex, normalize_id  # noqa: E402
from pokezero.engine_fidelity import (  # noqa: E402
    _DAMAGE_TOLERANCE,
    _MIN_TOLERANCE_HP,
    TurnFeatures,
    _engine_side_conditions,
    _features_payload,
    showdown_turn_features,
)
from pokezero.engine_fidelity_multiturn import observed_boost_deltas  # noqa: E402
from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy  # noqa: E402
from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    unpack_team,
    world_battle_spec,
)
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.golden_corpus import _true_teams_from_bridge_snapshot  # noqa: E402
from pokezero.local_showdown import (  # noqa: E402
    DEFAULT_SHOWDOWN_ROOT,
    LocalShowdownConfig,
    LocalShowdownEnv,
    _public_materialization_payload,
)
from pokezero.poke_engine_adapter import build_poke_engine_state  # noqa: E402
from pokezero.randbat import Gen3RandbatSource, canonical_gen3_randbat_species_id  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fidelity_gate_events import truant_loaf_slots  # noqa: E402

_ENGINE_BOOST_FIELDS = (
    "attack_boost",
    "defense_boost",
    "special_attack_boost",
    "special_defense_boost",
    "speed_boost",
    "accuracy_boost",
    "evasion_boost",
)


# ---------------------------------------------------------------------------------------------
# Feature extraction keyed by PLAYER SLOT (the engine's side_one/side_two assignment
# is an EngineWorld detail; the protocol is always p1/p2).
# ---------------------------------------------------------------------------------------------


def _sides_by_slot(state: Any, slot_sides: Mapping[str, str]) -> dict[str, Any]:
    sides = {"side_one": state.side_one, "side_two": state.side_two}
    return {slot: sides[slot_sides[slot]] for slot in ("p1", "p2")}


def engine_features_by_slot(state: Any, slot_sides: Mapping[str, str]) -> TurnFeatures:
    sides = _sides_by_slot(state, slot_sides)
    actives = {
        slot: side.pokemon[int(str(side.active_index))] for slot, side in sides.items()
    }
    return TurnFeatures(
        p1_hp=int(actives["p1"].hp),
        p2_hp=int(actives["p2"].hp),
        p1_status=str(actives["p1"].status).upper(),
        p2_status=str(actives["p2"].status).upper(),
        fainted=frozenset(slot for slot, mon in actives.items() if mon.hp <= 0),
        weather=str(state.weather).upper(),
        side_conditions={slot: _engine_side_conditions(side) for slot, side in sides.items()},
    )


def engine_boost_deltas_by_slot(
    before: Any, after: Any, slot_sides: Mapping[str, str]
) -> dict[str, dict[str, int]]:
    pre = _sides_by_slot(before, slot_sides)
    post = _sides_by_slot(after, slot_sides)
    deltas: dict[str, dict[str, int]] = {}
    for slot in ("p1", "p2"):
        stats: dict[str, int] = {}
        for name in _ENGINE_BOOST_FIELDS:
            delta = int(getattr(post[slot], name, 0) or 0) - int(getattr(pre[slot], name, 0) or 0)
            if delta:
                stats[name[: -len("_boost")]] = delta
        deltas[slot] = stats
    return deltas


def _fold(lines: Sequence[str]) -> TurnFeatures:
    return showdown_turn_features(types.SimpleNamespace(protocol_lines=tuple(lines)))


# ---------------------------------------------------------------------------------------------
# Choice resolution: chosen action index -> engine move/switch string.
# ---------------------------------------------------------------------------------------------


class UnmappableChoice(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def engine_choice_for_action(
    *,
    action_index: int,
    candidates: Sequence[Mapping[str, Any]],
    engine_side: Any,
    party_species: Sequence[str],
) -> str:
    """Translate one seat's chosen action index into an engine choice string.

    Move ids are resolved against the BUILT engine active's own move list (which
    is how Hidden Power's typed+BP engine id is recovered from the request's
    plain ``hiddenpower``); switch targets are resolved against the engine party
    order, with the cosmetic-forme collapse the world constructor applies.
    """

    candidate = next(
        (
            row
            for row in candidates
            if isinstance(row, Mapping) and row.get("action_index") == action_index
        ),
        None,
    )
    if candidate is None:
        raise UnmappableChoice("no_candidate_row")
    kind = candidate.get("kind")
    if kind == "move":
        move_id = normalize_id(str(candidate.get("move_id") or ""))
        if not move_id:
            raise UnmappableChoice("blank_move_id")
        active = engine_side.pokemon[int(str(engine_side.active_index))]
        engine_moves = [normalize_id(str(move.id)) for move in active.moves]
        if move_id in engine_moves:
            return move_id
        if move_id.startswith("hiddenpower"):
            typed = [m for m in engine_moves if m.startswith("hiddenpower")]
            if len(typed) == 1:
                return typed[0]
            raise UnmappableChoice("hidden_power_ambiguous")
        # Struggle / recharge and other pseudo-moves are engine-legal without a
        # party move slot; pass them through and let the engine reject.
        if move_id in {"struggle", "recharge"}:
            return move_id
        raise UnmappableChoice(f"move_not_in_engine_set:{move_id}")
    if kind == "switch":
        pokemon = candidate.get("pokemon")
        species = (
            normalize_id(str(pokemon.get("species") or ""))
            if isinstance(pokemon, Mapping)
            else ""
        )
        if not species:
            raise UnmappableChoice("blank_switch_species")
        # gen3 ``MoveChoice::from_string`` (third_party/poke-engine-src/src/gen3/
        # state.rs:51) resolves a switch from the BARE species id — the
        # ``"switch <species>"`` form raises ValueError("Invalid move for sN").
        party = [normalize_id(s) for s in party_species]
        if species in party:
            return species
        canonical = canonical_gen3_randbat_species_id(species)
        matches = [s for s in party if canonical_gen3_randbat_species_id(s) == canonical]
        if len(matches) == 1:
            return matches[0]
        raise UnmappableChoice("switch_species_not_in_party")
    raise UnmappableChoice(f"unknown_kind:{kind}")


# ---------------------------------------------------------------------------------------------
# Per-boundary evaluation.
# ---------------------------------------------------------------------------------------------


def _prestate_mismatch(observed: TurnFeatures, engine: TurnFeatures) -> str | None:
    """Exact pre-state comparison (no damage band: both sides claim the SAME state)."""

    if observed.p1_status != engine.p1_status:
        return f"p1 status {observed.p1_status} != {engine.p1_status}"
    if observed.p2_status != engine.p2_status:
        return f"p2 status {observed.p2_status} != {engine.p2_status}"
    if observed.weather != engine.weather:
        return f"weather {observed.weather} != {engine.weather}"
    if observed.presence() != engine.presence():
        return f"side conditions {observed.presence()} != {engine.presence()}"
    for side, obs_hp, eng_hp in (
        ("p1", observed.p1_hp, engine.p1_hp),
        ("p2", observed.p2_hp, engine.p2_hp),
    ):
        if obs_hp < 0:
            return f"{side} hp unknown in protocol fold"
        if obs_hp != eng_hp:
            return f"{side} hp {obs_hp} != {eng_hp}"
    return None


def _transition_mismatch(
    observed: TurnFeatures,
    branch: TurnFeatures,
    *,
    start_hp: Mapping[str, int],
    branch_maxhp: Mapping[str, int],
    active_changed: Mapping[str, bool],
) -> str | None:
    """This step's observed delta vs one engine branch, with the real-game carve-outs.

    Differences from :func:`pokezero.engine_fidelity._mismatch_reason`, each
    forced by a live-game shape the curated fixtures never reach:

    * a side that FAINTED this step is not status-compared — Showdown's
      ``0 fnt`` condition string carries no status, while the engine keeps the
      status on the fainted mon (the known faint-pattern conflation);
    * a side whose ACTIVE CHANGED anchors its damage band on the incoming mon's
      max HP rather than on the outgoing mon's HP, which is not a damage scale
      at all.
    """

    for slot, obs_status, br_status in (
        ("p1", observed.p1_status, branch.p1_status),
        ("p2", observed.p2_status, branch.p2_status),
    ):
        if slot in observed.fainted or slot in branch.fainted:
            continue
        if obs_status != br_status:
            return f"{slot} status {obs_status} != {br_status}"
    if observed.fainted != branch.fainted:
        return f"fainted {sorted(observed.fainted)} != {sorted(branch.fainted)}"
    if observed.weather != branch.weather:
        return f"weather {observed.weather} != {branch.weather}"
    if observed.presence() != branch.presence():
        return f"side conditions {observed.presence()} != {branch.presence()}"
    for slot, obs_hp, br_hp in (
        ("p1", observed.p1_hp, branch.p1_hp),
        ("p2", observed.p2_hp, branch.p2_hp),
    ):
        if obs_hp < 0:
            continue
        if obs_hp == 0 or br_hp == 0:
            if obs_hp != br_hp:
                return f"{slot} hp {obs_hp} != {br_hp} (faint boundary)"
            continue
        anchor = branch_maxhp[slot] if active_changed[slot] else start_hp[slot]
        damage_scale = max(abs(anchor - br_hp), _MIN_TOLERANCE_HP)
        if abs(obs_hp - br_hp) > max(_MIN_TOLERANCE_HP, _DAMAGE_TOLERANCE * damage_scale):
            return f"{slot} hp {obs_hp} outside tolerance of {br_hp} (anchor {anchor})"
    return None



def classify_divergence(step_lines: Sequence[str], misses: Sequence[str]) -> str:
    """Coarse, evidence-based bucket for one divergent boundary.

    Ordered most-specific-first; every bucket is falsifiable from the step's own
    protocol so the ledger's per-class rates are auditable.
    """

    fainted = any(line.startswith("|faint|") for line in step_lines)
    upkeep = any(line.strip() == "|upkeep" for line in step_lines)
    if fainted and not upkeep:
        # Showdown defers the end-of-turn residual block past a mid-turn faint
        # (the switch request comes first); poke-engine runs it in the same ply.
        return "faint_ply_residual_deferral"
    if any("[from] Spikes" in line for line in step_lines):
        return "spikes_entry_damage"
    if any("[from] psn" in line or "[from] brn" in line or "[from] tox" in line for line in step_lines):
        return "status_residual"
    if any("|-crit|" in line for line in step_lines):
        return "crit_roll_band"
    if misses and "boost deltas" in misses[0]:
        return "boost_delta_support"
    if misses and " status " in misses[0]:
        return "status_support"
    if misses and "fainted " in misses[0]:
        return "faint_boundary"
    if misses and " hp " in misses[0]:
        return "damage_band"
    return "unclassified"

def _active_maxhp_by_slot(state: Any, slot_sides: Mapping[str, str]) -> dict[str, int]:
    sides = _sides_by_slot(state, slot_sides)
    return {
        slot: int(side.pokemon[int(str(side.active_index))].maxhp) for slot, side in sides.items()
    }


def evaluate_boundary(
    *,
    state: Any,
    slot_sides: Mapping[str, str],
    choices: Mapping[str, str],
    pre_features: TurnFeatures,
    observed: TurnFeatures,
    observed_boosts: Mapping[str, Mapping[str, int]],
    active_changed: Mapping[str, bool],
) -> tuple[str, list[str], int]:
    """Return ``(verdict, misses, branch_count)`` for one full decision boundary."""

    side_one_choice = choices["p1"] if slot_sides["p1"] == "side_one" else choices["p2"]
    side_two_choice = choices["p2"] if slot_sides["p2"] == "side_two" else choices["p1"]
    branches = poke_engine.generate_instructions(state, side_one_choice, side_two_choice)

    rows: list[dict[str, Any]] = []
    for branch in branches:
        applied = state.apply_instructions(branch)
        rows.append(
            {
                "percentage": float(branch.percentage),
                "features": engine_features_by_slot(applied, slot_sides),
                "boost_deltas": engine_boost_deltas_by_slot(state, applied, slot_sides),
                "maxhp": _active_maxhp_by_slot(applied, slot_sides),
            }
        )

    # A REGULAR switch clears stat stages with no protocol echo (Showdown emits
    # nothing; the engine emits reset_boosts instructions), so a side whose
    # active changed is exempt from the exact boost-delta filter.
    def _comparable(deltas: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
        return {
            slot: ({} if active_changed[slot] else dict(deltas.get(slot) or {}))
            for slot in ("p1", "p2")
        }

    normalized_observed = _comparable(observed_boosts)
    candidates = [row for row in rows if _comparable(row["boost_deltas"]) == normalized_observed]
    if not candidates:
        misses = [
            f"observed boost deltas {normalized_observed} not in branch support: "
            + "; ".join(
                f"pct={row['percentage']:.2f}: {_comparable(row['boost_deltas'])}" for row in rows
            )
        ]
        return "diverged", misses, len(rows)

    start_hp = {"p1": pre_features.p1_hp, "p2": pre_features.p2_hp}
    misses = []
    for row in candidates:
        reason = _transition_mismatch(
            observed,
            row["features"],
            start_hp=start_hp,
            branch_maxhp=row["maxhp"],
            active_changed=active_changed,
        )
        if reason is None:
            return "matched", [], len(rows)
        misses.append(f"pct={row['percentage']:.2f}: {reason}")
    return "diverged", misses, len(rows)


# ---------------------------------------------------------------------------------------------
# Game driver.
# ---------------------------------------------------------------------------------------------


def run_game(
    *,
    env: LocalShowdownEnv,
    flags_policy: EngineMctsPolicy,
    seed: int,
    dex: Any,
    max_steps: int,
    keep_repro: int,
    repros: list[dict[str, Any]],
    approximate_sleep: bool,
) -> Counter:
    counts: Counter = Counter()
    env.reset(seed=seed, format_id="gen3randombattle")
    true_teams = _true_teams_from_bridge_snapshot(env.snapshot().bridge_snapshot)
    packed = {slot: true_teams[slot]["packed"] for slot in ("p1", "p2")}
    override = BattleStartOverride(player_teams=packed)
    teams = {slot: unpack_team(packed[slot]) for slot in ("p1", "p2")}
    rng = random.Random(seed ^ 0x5EED)

    cumulative: list[str] = list(env.protocol_lines)
    cursor = len(cumulative)
    steps = 0

    while env.terminal() is None and steps < max_steps:
        steps += 1
        requested = tuple(env.requested_players())
        actions: dict[str, int] = {}
        for player in requested:
            mask = env.legal_actions(player)
            legal = [i for i, allowed in enumerate(mask) if allowed]
            if not legal:
                counts["abort:no_legal_action"] += 1
                return counts
            actions[player] = rng.choice(legal)

        prepared: dict[str, Any] | None = None
        if set(requested) == {"p1", "p2"}:
            counts["boundaries_full_round"] += 1
            prepared = _prepare_boundary(
                env=env,
                flags_policy=flags_policy,
                override=override,
                teams=teams,
                dex=dex,
                actions=actions,
                cumulative=cumulative,
                counts=counts,
                approximate_sleep=approximate_sleep,
            )
        else:
            counts["skip:single_seat_boundary"] += 1

        env.step(actions)
        step_lines = tuple(str(line) for line in env.protocol_lines[cursor:])
        cursor = len(env.protocol_lines)
        cumulative.extend(step_lines)

        if prepared is None:
            continue

        observed = _fold(cumulative)
        observed = TurnFeatures(
            p1_hp=observed.p1_hp,
            p2_hp=observed.p2_hp,
            p1_status=observed.p1_status,
            p2_status=observed.p2_status,
            fainted=_fold(step_lines).fainted,
            weather=observed.weather,
            side_conditions=observed.side_conditions,
        )
        active_changed = {
            slot: any(
                line.startswith((f"|switch|{slot}a", f"|drag|{slot}a", f"|replace|{slot}a"))
                for line in step_lines
            )
            for slot in ("p1", "p2")
        }
        try:
            verdict, misses, branch_count = evaluate_boundary(
                state=prepared["state"],
                slot_sides=prepared["slot_sides"],
                choices=prepared["choices"],
                pre_features=prepared["pre_features"],
                observed=observed,
                observed_boosts=observed_boost_deltas(step_lines),
                active_changed=active_changed,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # pyo3 panics do not derive from Exception
            counts["engine_error"] += 1
            # Strip operands so the reason histogram stays low-cardinality
            # ("Invalid move for s1: firepunch" -> "invalid_move").
            detail = "invalid_move" if "Invalid move for" in str(error) else "other"
            counts[f"engine_error:{type(error).__name__}:{detail}"] += 1
            if detail == "invalid_move":
                bad = str(error).split(": ", 1)[-1].strip()
                counts[f"engine_error_choice:{bad}"] += 1
            if len(repros) < keep_repro:
                repros.append(
                    {
                        "kind": "engine_error",
                        "seed": seed,
                        "step": steps,
                        "error": f"{type(error).__name__}: {error}",
                        "choices": prepared["choices"],
                        "engine_state": prepared["state"].to_string(),
                    }
                )
            continue

        counts[f"transition:{verdict}"] += 1
        if verdict == "diverged":
            counts[f"divergence_class:{classify_divergence(step_lines, misses)}"] += 1
        if verdict == "diverged" and len(repros) < keep_repro:
            repros.append(
                {
                    "kind": "transition_diverged",
                    "seed": seed,
                    "step": steps,
                    "choices": prepared["choices"],
                    "engine_state": prepared["state"].to_string(),
                    "pre_features": _features_payload(prepared["pre_features"]),
                    "observed": _features_payload(observed),
                    "observed_boost_deltas": observed_boost_deltas(step_lines),
                    "active_changed": active_changed,
                    "divergence_class": classify_divergence(step_lines, misses),
                    "branch_count": branch_count,
                    "branch_misses": misses[:12],
                    "protocol": list(step_lines),
                }
            )
    if steps >= max_steps:
        counts["abort:max_steps"] += 1
    return counts


def _prepare_boundary(
    *,
    env: LocalShowdownEnv,
    flags_policy: EngineMctsPolicy,
    override: BattleStartOverride,
    teams: Mapping[str, tuple],
    dex: Any,
    actions: Mapping[str, int],
    cumulative: Sequence[str],
    counts: Counter,
    approximate_sleep: bool,
) -> dict[str, Any] | None:
    """Build the engine world + resolve both choices, or return None with a counted skip."""

    try:
        mstate = env.public_materialization_state("p1")
    except Exception as error:  # noqa: BLE001 — a materialization refusal is a skip
        counts[f"skip:no_materialization:{type(error).__name__}"] += 1
        return None

    observation = env.observe("p1")
    context = types.SimpleNamespace(observation=observation, player_id="p1")
    # Production derivation of the public item/Transform/Encore signals the
    # payload cannot carry (engine_search.EngineMctsPolicy._public_effect_signals):
    # reused verbatim so the differential builds the same world the live searcher
    # would. It needs only ``observation.metadata`` + ``player_id``.
    blocked, encored, removed, overridden = flags_policy._public_effect_signals(context)

    candidates_by_slot: dict[str, Sequence[Mapping[str, Any]]] = {}
    recharging: list[str] = []
    for slot in ("p1", "p2"):
        metadata = env.observe(slot).metadata
        rows = metadata.get("action_candidates")
        if not isinstance(rows, Sequence):
            counts["skip:no_action_candidates"] += 1
            return None
        candidates_by_slot[slot] = rows
        chosen = next(
            (r for r in rows if isinstance(r, Mapping) and r.get("action_index") == actions[slot]),
            None,
        )
        if (
            isinstance(chosen, Mapping)
            and chosen.get("kind") == "move"
            and normalize_id(str(chosen.get("move_id") or "")) == "recharge"
        ):
            recharging.append(slot)

    try:
        payload = _public_materialization_payload(mstate)
        truant = truant_loaf_slots(list(cumulative), payload, teams)
    except Exception:  # noqa: BLE001
        truant = []

    try:
        world = world_battle_spec(
            mstate,
            override,
            dex=dex,
            approximate_sleep_turns=approximate_sleep,
            approximate_substitute_health=True,
            blocked_slots=blocked,
            encored_moves=encored,
            removed_item_species=removed,
            current_item_overrides=overridden,
            recharging_slots=tuple(recharging),
            truant_slots=tuple(truant),
        )
        state = build_poke_engine_state(world.spec, module=poke_engine)
    except EngineWorldUnsupported as error:
        counts[f"skip:world_unsupported:{error.reason}"] += 1
        return None
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:  # noqa: BLE001
        counts[f"skip:world_error:{type(error).__name__}"] += 1
        return None

    sides = _sides_by_slot(state, world.slot_sides)
    choices: dict[str, str] = {}
    for slot in ("p1", "p2"):
        try:
            choices[slot] = engine_choice_for_action(
                action_index=actions[slot],
                candidates=candidates_by_slot[slot],
                engine_side=sides[slot],
                party_species=world.party_species[slot],
            )
        except UnmappableChoice as error:
            counts[f"skip:unmappable_choice:{error.reason}"] += 1
            return None

    pre_features = engine_features_by_slot(state, world.slot_sides)
    observed_pre = _fold(cumulative)
    mismatch = _prestate_mismatch(observed_pre, pre_features)
    if mismatch is not None:
        counts["world_prestate_mismatch"] += 1
        counts[f"world_prestate_mismatch:{mismatch.split(' ')[0]}_{mismatch.split(' ')[1]}"] += 1
        return None

    counts["boundaries_measured"] += 1
    return {
        "state": state,
        "slot_sides": world.slot_sides,
        "choices": choices,
        "pre_features": pre_features,
    }


# ---------------------------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--showdown-root", default=DEFAULT_SHOWDOWN_ROOT)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=900000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--keep-repro", type=int, default=25)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--approximate-sleep",
        action="store_true",
        help="approximate hidden sleep counters instead of failing the world closed "
             "(default: strict — a publicly-asleep mon with an unknown counter is a "
             "counted SKIP, never a guessed world)",
    )
    args = parser.parse_args(argv)

    dex = load_showdown_dex(args.showdown_root)
    env = LocalShowdownEnv(
        LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True)
    )
    flags_policy = EngineMctsPolicy(
        dex=dex,
        set_source=Gen3RandbatSource.from_showdown_root(args.showdown_root),
        config=EngineMctsConfig(worlds=1, search_time_ms=1),
    )

    totals: Counter = Counter()
    repros: list[dict[str, Any]] = []
    started = time.perf_counter()
    games_done = 0
    try:
        for offset in range(args.games):
            seed = args.seed_start + offset
            totals.update(
                run_game(
                    env=env,
                    flags_policy=flags_policy,
                    seed=seed,
                    dex=dex,
                    max_steps=args.max_steps,
                    keep_repro=args.keep_repro,
                    repros=repros,
                    approximate_sleep=args.approximate_sleep,
                )
            )
            games_done += 1
            if args.progress_every and games_done % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"[{games_done}/{args.games}] {elapsed:.0f}s "
                    f"({games_done / elapsed * 3600:.0f} games/h) "
                    f"measured={totals['boundaries_measured']} "
                    f"matched={totals['transition:matched']} "
                    f"diverged={totals['transition:diverged']}",
                    flush=True,
                )
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    measured = totals["boundaries_measured"]
    diverged = totals["transition:diverged"] + totals["engine_error"]
    report = {
        "games": games_done,
        "seed_start": args.seed_start,
        "approximate_sleep_turns": bool(args.approximate_sleep),
        "elapsed_seconds": round(elapsed, 2),
        "games_per_hour": round(games_done / elapsed * 3600, 1) if elapsed else None,
        "boundaries_full_round": totals["boundaries_full_round"],
        "boundaries_measured": measured,
        "transitions_matched": totals["transition:matched"],
        "transitions_diverged": totals["transition:diverged"],
        "engine_errors": totals["engine_error"],
        "divergent_transitions_per_game": round(diverged / games_done, 4) if games_done else None,
        "measured_fraction_of_full_rounds": (
            round(measured / totals["boundaries_full_round"], 4)
            if totals["boundaries_full_round"]
            else None
        ),
        "divergence_classes": {
            key.split(":", 1)[1]: value
            for key, value in sorted(totals.items())
            if key.startswith("divergence_class:")
        },
        "counters": dict(sorted(totals.items())),
        "repros": repros,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "repros"}, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"-> {args.json}")
    return 1 if diverged else 0


if __name__ == "__main__":
    raise SystemExit(main())
