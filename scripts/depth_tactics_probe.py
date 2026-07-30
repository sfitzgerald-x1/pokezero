#!/usr/bin/env python
"""Constructed-tactics depth probe: does depth work AT ALL in the model-leaf search?

Context (docs/mcts_dual_grid_findings.md): measured playing strength is
depth-invariant (d1≈d2≈d4≈d6 at s1024, n=400/cell), but at s1024 the sim budget
starves the cap (mean reached depth ~2.5), so the flat ladders cannot say
whether depth WORKS. This probe removes the game-distribution confound with
CONSTRUCTED positions carrying a provable forced win that requires seeing N
plies ahead: a depth-d search with d < N cannot distinguish the winning move,
a depth-d search with d >= N must find it (the forced line ends in an exact
terminal branch inside the tree's own horizon).

Design contract per position (see ``POSITIONS``):

- The opponent side has exactly ONE legal action each turn (one mon, one move,
  no bench), so the game tree is single-agent for forcing purposes and
  "forced" needs no simultaneous-move caveats.
- Forcing-ness is verified against THE ENGINE'S OWN GAME — exhaustive
  enumeration over ``pe.generate_instructions`` (branch_on_damage=True,
  exactly the branching the search tree uses at plies 1-2 plus KO-splits),
  every branch, no sampling — by ``solve_forced``. The hand-derived line in
  the position's docstring is the human-readable account of the same
  arithmetic (damage numbers cross-checked against ``pe.calculate_damage``).
- ``needed_depth`` is the measured flip point of an exhaustive fixed-horizon
  solver with the crate's own HP-fraction leaf (``solve_horizon_argmax``):
  the horizon at which the optimal root move becomes the forced-win move and
  stays it. d < needed_depth must argmax the trap move; d >= needed_depth
  must argmax the winning move.

The probe then runs the REAL search at d1/d2/d4/d6, s4096 (budget chosen so
the cap binds, not the budget; ``max_depth_reached`` is recorded per cell):

- ``hp_fraction_crate`` leaf via ``pokezero_search.puct_search_multi`` — the
  control arm, known to convert depth into strength;
- ``model`` leaf via ``NativeLeafModel.search_batched_multi_encoded`` — the
  full production pipeline (live root fold, per-branch synthesized events,
  checkpoint-latched native leaf encode, TorchScript leaf eval), driven
  through a real ``LocalShowdownEnv`` boundary exactly like production
  (``EngineMctsPolicy`` internals are reused verbatim for root inputs, fold,
  world construction and choice mapping).

Positions are materialized through the scenario-studio seam: packed Custom
Game teams (``BattleStartOverride``) + the typed scenario bridge patch
(current HP), so the observation the model sees is a REAL env observation of
the constructed position, not a synthetic tensor.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/depth_tactics_probe.py \
        --checkpoint local-artifacts/v3hist-k64-enthalf-5m-20260723-iteration-2657.pt \
        --out docs/audit_artifacts/depth-tactics-20260729/probe.json

    # design-only (no checkpoint needed): verify forcing + horizon flips
    PYTHONPATH=src .venv/bin/python scripts/depth_tactics_probe.py --design-only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEPTHS = (1, 2, 4, 6)
SIMS = 4096
SEARCH_SEEDS = (7, 1337, 900913)
C_PUCT = 1.4

# ---------------------------------------------------------------------------
# Position specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonSpec:
    species: str
    moves: tuple[str, ...]
    ability: str
    level: int = 100
    item: str | None = None
    current_hp: int | None = None  # None = full
    evs: Mapping[str, int] | None = None
    ivs: Mapping[str, int] | None = None
    # Per-move current PP override (e.g. {"hyperbeam": 1} pins a single use);
    # unlisted moves keep their base PP.
    move_pp: Mapping[str, int] | None = None


@dataclass(frozen=True)
class PositionSpec:
    name: str
    title: str
    # Hand-derived account of the forced line (the report's human-readable
    # arithmetic; the solver is the machine check of the same facts).
    derivation: str
    p1: tuple[MonSpec, ...]
    p2: tuple[MonSpec, ...]
    # Engine move-choice string of the forced-win root move and the trap move
    # (the locally attractive move that loses by force), e.g. "return102",
    # "switch cloyster".
    win_move: str
    trap_move: str
    # Solver horizon at which the argmax must flip to win_move (design-time
    # expectation; the probe MEASURES the flip and reports both).
    expected_needed_depth: int
    # Max turns for the forcing proof (>= the forced line's length; the
    # solver proves win-by-force within this horizon on every branch).
    proof_horizon: int = 6
    seed: int = 4242


# ---------------------------------------------------------------------------
# Materialization through the env (scenario-studio seam)
# ---------------------------------------------------------------------------


def _fixture(mon: MonSpec):
    from pokezero.showdown_fixture import FixturePokemon

    return FixturePokemon(
        species=mon.species,
        moves=mon.moves,
        ability=mon.ability,
        item=mon.item,
        level=mon.level,
        evs=mon.evs,
        ivs=mon.ivs,
    )


def start_override(spec: PositionSpec):
    from pokezero.env import BattleStartOverride
    from pokezero.showdown_fixture import pack_team

    return BattleStartOverride(
        player_teams={
            "p1": pack_team(tuple(_fixture(m) for m in spec.p1)),
            "p2": pack_team(tuple(_fixture(m) for m in spec.p2)),
        }
    )


def bridge_patch(spec: PositionSpec, dex) -> dict[str, Any]:
    """The typed scenario-materialization patch (current HP only; slot 0 active).

    The bridge requires an integer HP for every mon and an integer PP for
    every move slot; PP is set to the move's base PP (always <= the packed
    team's PP-Up maxpp).
    """

    def base_pp(move: str) -> int:
        info = dex.move_info(move)
        return int(getattr(info, "pp", 0) or 16)

    def side(mons: tuple[MonSpec, ...]) -> dict[str, Any]:
        return {
            "activeSlot": 0,
            "sideConditions": {},
            "activeVolatiles": [],
            "pokemon": [
                {
                    "slot": index,
                    **({"hp": mon.current_hp} if mon.current_hp is not None else {}),
                    "status": {"id": "", "sleepTurnsRemaining": None, "toxicStage": None},
                    "moves": [
                        {"id": move, "pp": int((mon.move_pp or {}).get(move, base_pp(move)))}
                        for move in mon.moves
                    ],
                }
                for index, mon in enumerate(mons)
            ],
        }

    return {
        "turn": 1,
        "field": {"weather": "", "turnsRemaining": 0, "permanent": False},
        "sides": {"p1": side(spec.p1), "p2": side(spec.p2)},
    }


def _request_max_hp(env, player: str) -> list[int]:
    """Max HP per party slot, read from the freshly-reset battle's own request."""

    side = (env._latest_requests or {}).get(player, {}).get("side", {})  # noqa: SLF001
    values: list[int] = []
    for mon in side.get("pokemon", ()):  # condition: "319/319" or "0 fnt"
        condition = str(mon.get("condition") or "")
        total = condition.split(" ")[0]
        values.append(int(total.split("/")[1]) if "/" in total else 0)
    return values


def materialize_context(env, spec: PositionSpec, dex, *, battle_id: str):
    """Reset the env into the constructed position and build the production
    PolicyContext for the p1 seat (rollout.py's construction, verbatim shape)."""

    from pokezero.policy import PolicyContext
    from pokezero.trajectory import BattleTrajectory

    override = start_override(spec)
    env.reset_with_start_override(seed=spec.seed, start_override=override)
    # The bridge requires an explicit integer HP for every mon: fill full-HP
    # mons with the battle's own max HP (read off the reset request).
    max_hp = {player: _request_max_hp(env, player) for player in ("p1", "p2")}
    filled = PositionSpec(
        **{
            **spec.__dict__,
            "p1": tuple(
                MonSpec(**{**mon.__dict__, "current_hp": mon.current_hp or max_hp["p1"][i]})
                for i, mon in enumerate(spec.p1)
            ),
            "p2": tuple(
                MonSpec(**{**mon.__dict__, "current_hp": mon.current_hp or max_hp["p2"][i]})
                for i, mon in enumerate(spec.p2)
            ),
        }
    )
    env.materialize_scenario_state(scenario_state=bridge_patch(filled, dex))
    observation = env.observe("p1")
    context = PolicyContext(
        player_id="p1",
        decision_round_index=0,
        battle_id=battle_id,
        format_id="gen3customgame",
        seed=spec.seed,
        observation=observation,
        requested_players=("p1", "p2"),
        trajectory=BattleTrajectory(
            battle_id=battle_id,
            format_id="gen3customgame",
            seed=spec.seed,
            steps=[],
            terminal=None,
            metadata={},
        ),
        requested_legal_action_masks={"p1": tuple(observation.legal_action_mask)},
        requested_observations={"p1": observation},
        public_materialization_state=env.public_materialization_state("p1"),
    )
    return context, override


def build_world_state(policy, context, override):
    """The exact belief-world the policy would search: fixed_override world.

    Reuses EngineMctsPolicy's own construction path (signals + world spec +
    engine state) so the probed state is byte-identical to production's.
    """

    from pokezero.engine_world import world_battle_spec
    from pokezero.poke_engine_adapter import build_poke_engine_state

    blocked, encored, removed, overridden, transformed = policy._public_effect_signals(context)
    world = world_battle_spec(
        context.public_materialization_state,
        override,
        dex=policy._dex,
        approximate_sleep_turns=policy._config.approximate_sleep_turns,
        approximate_substitute_health=policy._config.approximate_substitute_health,
        approximate_partial_trap_turns=policy._config.approximate_partial_trap_turns,
        approximate_hidden_duration_volatiles=policy._config.approximate_hidden_duration_volatiles,
        blocked_slots=blocked,
        encored_moves=encored,
        removed_item_species=removed,
        current_item_overrides=overridden,
        recharging_slots=policy._recharging_slots(context),
        truant_slots=policy._truant_loaf_slots(context),
        transformed_slots=transformed,
        rng=random.Random(spec_rng_seed(context)),
    )
    return world, build_poke_engine_state(world.spec, module=policy._module)


def spec_rng_seed(context) -> int:
    return (hash((context.battle_id, context.player_id)) & 0x7FFFFFFF) or 1


# ---------------------------------------------------------------------------
# Exhaustive forcing solver (the engine's own game, every branch)
# ---------------------------------------------------------------------------


def _engine_choice(display: str) -> str:
    """env_options display -> MoveChoice::from_string token.

    The crate displays "No Move" for MoveChoice::None (parse token "none")
    and prefixes switches with "switch " (the Python binding's from_string
    matches the bare species name).
    """

    token = display.strip()
    if token.lower() == "no move":
        return "none"
    if token.lower().startswith("switch "):
        return token[len("switch "):]
    return token


def _options(state_str: str, *, root: bool) -> tuple[list[str], list[str], float]:
    import pokezero_search

    payload = json.loads(pokezero_search.env_options(state_str, root))
    return (
        [_engine_choice(o) for o in payload["p1"]],
        [_engine_choice(o) for o in payload["p2"]],
        float(payload["battle_over"]),
    )


def _apply_branch(pe, state, branch):
    return state.apply_instructions(branch)


def solve_forced(
    pe,
    state,
    horizon: int,
    *,
    root: bool = True,
    _cache: dict | None = None,
) -> dict[str, bool]:
    """Per side-one root option: is the option a FORCED WIN within `horizon`?

    Forced win for an option o: for EVERY chance branch of (o, opponent's
    single action), either the battle is over with side one winning, or some
    side-one continuation is again a forced win within the remaining horizon.
    Requires the side-two option surface to be a singleton at every reached
    decision (the design contract); raises otherwise.
    """
    import pokezero_search

    cache = _cache if _cache is not None else {}

    def forced_win(state_str: str, depth_left: int) -> bool:
        over = pokezero_search.env_battle_over(state_str)
        if over > 0.0:
            return True
        if over < 0.0:
            return False
        if depth_left == 0:
            return False
        key = (state_str, depth_left)
        if key in cache:
            return cache[key]
        s1_opts, s2_opts, _ = _options(state_str, root=False)
        if len(s2_opts) != 1:
            raise AssertionError(
                f"design contract violated: side two has {len(s2_opts)} options: {s2_opts}"
            )
        state_obj = pe.State.from_string(state_str)
        result = any(
            _option_forced(state_obj, o, s2_opts[0], depth_left)
            for o in s1_opts
        )
        cache[key] = result
        return result

    def _option_forced(state_obj, s1_move: str, s2_move: str, depth_left: int) -> bool:
        branches = pe.generate_instructions(state_obj, s1_move, s2_move)
        if not branches:
            return False
        for branch in branches:
            nxt = state_obj.apply_instructions(branch)
            if not forced_win(nxt.to_string(), depth_left - 1):
                return False
        return True

    s1_opts, s2_opts, over = _options(state.to_string(), root=root)
    if over != 0.0:
        raise AssertionError("battle already over at the probed root")
    if len(s2_opts) != 1:
        raise AssertionError(
            f"design contract violated at root: side two options {s2_opts}"
        )
    return {
        option: _option_forced(state, option, s2_opts[0], horizon)
        for option in s1_opts
    }


def solve_win_bounds(pe, state, horizon: int) -> dict[str, dict[str, float]]:
    """Per side-one root option: exact bounds on the achievable win probability.

    The engine branches accuracy, secondary effects, and crit/damage splits
    that flip a KO, so "forced" claims are probability statements over the
    engine's own exact branch percentages:

    - ``p_win_lower``: max over side-one strategies of the probability of a
      WIN terminal within `horizon` (non-terminal leaves count 0). A value of
      1.0 is a strict all-branches forced win.
    - ``p_win_upper``: same, but non-terminal leaves count 1 (the opponent
      cannot do better than this even granting every unresolved line). A trap
      arm with upper bound ~1/16 loses on everything but the crit lottery;
      0.0 is a strict forced loss.
    """
    import pokezero_search

    cache: dict = {}

    def bounds(state_str: str, depth_left: int) -> tuple[Fraction, Fraction]:
        over = pokezero_search.env_battle_over(state_str)
        if over > 0.0:
            return Fraction(1), Fraction(1)
        if over < 0.0:
            return Fraction(0), Fraction(0)
        if depth_left == 0:
            return Fraction(0), Fraction(1)
        key = (state_str, depth_left)
        if key in cache:
            return cache[key]
        s1_opts, s2_opts, _ = _options(state_str, root=False)
        if len(s2_opts) != 1:
            raise AssertionError("design contract violated (side two options)")
        state_obj = pe.State.from_string(state_str)
        pairs = [
            _edge_bounds(state_obj, o, s2_opts[0], depth_left) for o in s1_opts
        ]
        result = (max(p[0] for p in pairs), max(p[1] for p in pairs))
        cache[key] = result
        return result

    def _edge_bounds(state_obj, s1_move, s2_move, depth_left) -> tuple[Fraction, Fraction]:
        branches = pe.generate_instructions(state_obj, s1_move, s2_move)
        if not branches:
            return Fraction(0), Fraction(1)
        total = sum(Fraction(str(round(b.percentage, 4))) for b in branches)
        lower = Fraction(0)
        upper = Fraction(0)
        for branch in branches:
            probability = Fraction(str(round(branch.percentage, 4))) / total
            nxt = state_obj.apply_instructions(branch)
            lo, hi = bounds(nxt.to_string(), depth_left - 1)
            lower += probability * lo
            upper += probability * hi
        return lower, upper

    s1_opts, s2_opts, _ = _options(state.to_string(), root=True)
    return {
        option: dict(zip(("p_win_lower", "p_win_upper"),
                         map(float, _edge_bounds(state, option, s2_opts[0], horizon))))
        for option in s1_opts
    }


# ---------------------------------------------------------------------------
# Fixed-horizon exact solver with the crate's HP-fraction leaf
# ---------------------------------------------------------------------------


def _hp_fraction_value(pe, state) -> Fraction:
    """The crate's HpFractionEval, exactly (lib.rs side_hp_fraction)."""

    def side_fraction(side) -> Fraction:
        hp = Fraction(0)
        maxhp = Fraction(0)
        for mon in side.pokemon:
            if mon.maxhp <= 0 or str(mon.id) in ("", "none"):
                continue
            hp += max(mon.hp, 0)
            maxhp += mon.maxhp
        return hp / maxhp if maxhp > 0 else Fraction(1, 2)

    return Fraction(1, 2) + (side_fraction(state.side_one) - side_fraction(state.side_two)) / 2


def solve_horizon_values(pe, state, horizon: int) -> dict[str, Fraction]:
    """Exact expectimax value per side-one ROOT option at a fixed horizon.

    Semantics mirror the search tree's ideal limit: terminal branches are
    priced {0,1}; the horizon leaf is the HP-fraction value; chance nodes are
    exact expectations; side-one decision nodes take the max over options
    (side two is a singleton by the design contract). This is what an
    infinitely-sampled depth-`horizon` HP-leaf search converges to, so its
    argmax flip point is the position's `needed_depth` for the control arm.
    """
    import pokezero_search

    cache: dict = {}

    def value(state_str: str, depth_left: int) -> Fraction:
        over = pokezero_search.env_battle_over(state_str)
        if over > 0.0:
            return Fraction(1)
        if over < 0.0:
            return Fraction(0)
        state_obj = pe.State.from_string(state_str)
        if depth_left == 0:
            return _hp_fraction_value(pe, state_obj)
        key = (state_str, depth_left)
        if key in cache:
            return cache[key]
        s1_opts, s2_opts, _ = _options(state_str, root=False)
        if len(s2_opts) != 1:
            raise AssertionError("design contract violated (side two options)")
        result = max(
            _edge_value(state_obj, o, s2_opts[0], depth_left) for o in s1_opts
        )
        cache[key] = result
        return result

    def _edge_value(state_obj, s1_move, s2_move, depth_left) -> Fraction:
        branches = pe.generate_instructions(state_obj, s1_move, s2_move)
        if not branches:
            return _hp_fraction_value(pe, state_obj)
        total_pct = sum(Fraction(str(round(b.percentage, 4))) for b in branches)
        acc = Fraction(0)
        for branch in branches:
            probability = Fraction(str(round(branch.percentage, 4))) / total_pct
            nxt = state_obj.apply_instructions(branch)
            acc += probability * value(nxt.to_string(), depth_left - 1)
        return acc

    s1_opts, s2_opts, _ = _options(state.to_string(), root=True)
    return {
        option: _edge_value(state, option, s2_opts[0], horizon) for option in s1_opts
    }


def horizon_argmax_table(pe, state, max_horizon: int) -> list[dict[str, Any]]:
    rows = []
    for horizon in range(1, max_horizon + 1):
        values = solve_horizon_values(pe, state, horizon)
        best = max(values, key=values.get)
        rows.append(
            {
                "horizon": horizon,
                "argmax": best,
                "values": {k: float(v) for k, v in values.items()},
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Search runners
# ---------------------------------------------------------------------------


def run_hp_crate(state_str: str, depth: int, sims: int, seed: int) -> dict[str, Any]:
    import pokezero_search

    report = json.loads(
        pokezero_search.puct_search_multi(
            state_str, sims, max_depth=depth, c_puct=C_PUCT, seed=seed, deep_ko_split=True
        )
    )
    return report


def run_model(policy, context, world, state_str: str, depth: int, sims: int, batch: int,
              seed: int, *, model_priors: bool = True) -> dict[str, Any]:
    """The full in-crate model pipeline on the constructed position.

    Mirrors EngineMctsPolicy._search_model's crate invocation exactly (same
    root inputs surface, same live fold, same latched tables, same ctx);
    captures the crate's raw report instead of discarding it.
    """
    import pokezero_search

    policy._validate_model_root_observation(context.observation)
    live_fold = policy._advance_live_fold(context)
    if live_fold is None:
        raise RuntimeError("live fold broken on the constructed position")
    rust_fold = pokezero_search.FoldState.from_payload(live_fold.to_payload())
    root_inputs = policy._root_inputs_json(context)
    native = policy._native()
    replay = context.public_materialization_state.replay
    turn = int(getattr(replay, "turn_number", 0) or 0)
    ctx_json = json.dumps(
        {
            "p1": list(world.party_species["p1"]),
            "p2": list(world.party_species["p2"]),
            "turn": turn,
        }
    )
    report = json.loads(
        native.search_batched_multi_encoded(
            state_str,
            sims,
            batch,
            policy._tables_json,
            root_inputs,
            ctx_json,
            rust_fold,
            depth,
            C_PUCT,
            seed,
            True,  # deep_ko_split
            model_priors,
        )
    )
    return report


# ---------------------------------------------------------------------------
# Position roster
# ---------------------------------------------------------------------------

# NOTE ON LEGALITY: Custom Game materialization does not enforce randbats
# legality; species/moves are kept inside the gen3 randbats universe so the
# model's vocabulary sees familiar tokens (the leaf encoder's vocab-drift
# guard warns on any OOV token), but movesets are composed for forcing-ness
# (the point is a unit test of the tree, not an eval).
#
# Shared design conventions (see module docstring):
# - the opponent side is a single mon with a single move (usually a fixed-
#   damage clock: Seismic Toss / Night Shade = exactly 100 at L100), so the
#   game tree is single-agent and every "forced" claim is branch-exhaustive;
# - the engine's own game applies avg damage trunc(0.925*max) per branch and
#   splits only at KO-straddles (min = trunc(0.85*max)), so windows below are
#   stated in terms of (avg, min, max) of the engine's calculate_damage rolls;
# - current HP values are chosen so no unintended KO-straddle branch exists
#   along the analyzed lines (the solver enumerates whatever the engine
#   actually branches, so a missed straddle shows up as a probability < 1
#   forcing failure, not a silent error).
POSITIONS: tuple[PositionSpec, ...] = (
    # ------------------------------------------------------------------
    # P1 — needs depth 2. The Hyper Beam recharge trap.
    # Snorlax (210/461 hp, slower) holds {return, hyperbeam};
    # Registeel (85/301 hp, faster) is a Seismic Toss clock (100/turn).
    # Rolls (engine): return max 50 / min 42 / avg 46; hyperbeam max 73 /
    # min 62 / avg 67. Registeel at 85: hyperbeam cannot KO (73 < 85, no
    # straddle) but LOOKS better at depth 1 (67 > 46 damage on an equal
    # incoming toss). Forced lines:
    #   return: T1 toss(A 110) + return(B 39); T2 toss(A 10) + return —
    #           min 42 >= 39, certain KO => WIN at ply 2 (terminal).
    #   hyperbeam: T1 toss(A 110) + hb(B 18); T2 recharge + toss(A 10);
    #           T3 toss kills A before it acts (B faster) => forced LOSS.
    PositionSpec(
        name="hb-recharge-trap",
        title="Hyper Beam recharge trap (needs d2)",
        derivation=(
            "toss=100 exact; return avg 46 (max 50/min 42), hb avg 67 (max 73/min 62). "
            "B at 85: hb leaves 18 and costs T2 to recharge; A (210) dies to the 3rd toss "
            "before acting on T3. return leaves 39, and 39 <= min 42 makes the T2 KO certain. "
            "d1 sees hb 67 > return 46; d2 sees return's terminal win and hb's forced loss."
        ),
        p1=(MonSpec("Snorlax", ("return", "hyperbeam"), "Thick Fat", current_hp=210),),
        p2=(MonSpec("Registeel", ("seismictoss",), "Clear Body", current_hp=85),),
        win_move="return",
        trap_move="hyperbeam",
        expected_needed_depth=2,
        proof_horizon=4,
    ),
    # ------------------------------------------------------------------
    # P2 — needs depth 3 (depth 2 ties by construction). The Swords Dance race.
    # Snorlax (310/461) holds {return, swordsdance}; Registeel (160/301) tosses.
    # return avg 46; +2 return avg 92 (max 100/min 85). A survives three
    # tosses (310 > 300) and dies to the 4th before acting.
    #   swordsdance: T1 SD; T2 +2 return (B 68); T3 +2 return — min 85 >= 68,
    #           certain KO with A at 10 hp => WIN at ply 3.
    #   return spam: 3 x 46 = 138 < 160 on every branch; A dies during T4
    #           => forced LOSS (every continuation).
    # Depth-2 cumulative damage TIES exactly (46+46 = 0+92); the tie is
    # broken toward swordsdance by its T2 crit-KO branch (+2 crit 200 >= 160,
    # a 1/16 exact-terminal branch), so the measured argmax flip is h2 and
    # this is a depth-2 cell. d1 strictly prefers return.
    PositionSpec(
        name="sd-race",
        title="Swords Dance race (needs d2: exact damage ties, crit branch decides)",
        derivation=(
            "toss=100 exact; return avg 46, +2 return avg 92/min 85/max 100. B at 160: "
            "spam deals 138 by T3 (A dies during T4) — forced loss; SD->+2x2 deals 184 with "
            "the T3 KO certain (68 remaining <= min 85) — forced win. 46+46 == 0+92 makes "
            "depth 2 a tie by construction; depth 1 strictly prefers return."
        ),
        p1=(MonSpec("Snorlax", ("return", "swordsdance"), "Thick Fat", current_hp=310),),
        p2=(MonSpec("Registeel", ("seismictoss",), "Clear Body", current_hp=160),),
        win_move="swordsdance",
        trap_move="return",
        expected_needed_depth=2,
        proof_horizon=5,
    ),
    # ------------------------------------------------------------------
    # P3 — needs depth 4. The Perish Song clock.
    # Lapras (310/401, faster) holds {perishsong, surf}; Blissey (620/651) is
    # a Seismic Toss clock (fixed 100). Registeel (95/301) waits on the bench
    # as one-toss fodder. Surf cannot race 620 hp (engine avg 75; five surfs
    # reach 375) and never gets within surf's crit range, so the surf arm has
    # NO crit lottery: its loss is strict. The engine's perish cascade
    # (PERISH4->3->...) faints the perished active four plies after the song.
    #   perishsong T1: A eats tosses T1-T3 (310 -> 10), dies to the 4th
    #     mid-T4, Registeel replaces it untouched, B's count expires at the
    #     end of T4 => terminal WIN at ply 4 on every branch.
    #   Any delay is fatal by construction: singing at T2 lets the 5th toss
    #     empty our side mid-T5 before B's count (from T2) expires at the end
    #     of T5 — Registeel is one-toss fodder, so the replacement does not
    #     buy the missing turn. Every surf-first and switch-first line is a
    #     strict forced loss within 5 plies.
    # Depths 1-3 see only surf's hp progress vs perish's zero damage. NOTE
    # tree plies vs game turns: A's mid-T4 faint inserts a forced-replacement
    # DECISION node, so the win terminal sits at tree ply 5 — the measured
    # flip is h5 and the {1,2,4,6} grid resolves it at d6, not d4.
    PositionSpec(
        name="perish-clock",
        title="Perish Song clock (needs d5: 4 turns + forced-replacement ply)",
        derivation=(
            "toss=100 exact; surf avg 75 (five surfs = 375 < 620, crit never in range => the "
            "trap arm is a STRICT forced loss, no crit lottery). A at 310 survives exactly "
            "three tosses. Perish sung at T1 faints B at the end of T4 while benched "
            "Registeel survives untouched => terminal WIN at ply 4. Sung at T2+, our side "
            "is emptied mid-T5 (A dead mid-T4, one-toss fodder dead mid-T5) before the "
            "count expires => forced loss. d1-d3 see only surf's hp progress."
        ),
        p1=(
            MonSpec("Lapras", ("perishsong", "surf"), "Water Absorb", current_hp=310),
            MonSpec("Registeel", ("amnesia",), "Clear Body", current_hp=95),
        ),
        p2=(MonSpec("Blissey", ("seismictoss",), "Natural Cure", current_hp=620),),
        win_move="perishsong",
        trap_move="surf",
        expected_needed_depth=5,
        proof_horizon=6,
    ),
    # ------------------------------------------------------------------
    # P4 — control, needs depth 1 only. Immediate Hyper Beam KO.
    # Same species as P1, Registeel at 55: hyperbeam min 62 >= 55 => certain
    # KO now (gen3 skips the recharge when the target faints); return max 50
    # cannot KO and the second toss kills A first. EVERY depth must pick
    # hyperbeam — a positive control that the probe is not simply
    # anti-hyperbeam and that d4/d6 do not overthink a won position.
    PositionSpec(
        name="immediate-ko-control",
        title="Immediate KO control (needs d1; all depths must agree)",
        derivation=(
            "hb min 62 >= 55: certain KO at T1 (no recharge on a KO in gen3); return max 50 "
            "< 55 cannot KO and A (110) dies to the second toss. All depths must argmax "
            "hyperbeam."
        ),
        p1=(MonSpec("Snorlax", ("return", "hyperbeam"), "Thick Fat", current_hp=110),),
        p2=(MonSpec("Registeel", ("seismictoss",), "Clear Body", current_hp=55),),
        win_move="hyperbeam",
        trap_move="return",
        expected_needed_depth=1,
        proof_horizon=3,
    ),
    # ------------------------------------------------------------------
    # P5 — argmax flips at depth 3, terminal proof lands at depth 4.
    # Deeper Swords Dance race: A (410/461) survives four tosses; B at 250
    # requires the full SD + three +2 returns (92*3 = 276 >= 250, final KO
    # certain: 250-184 = 66 <= min 85). Spam deals 4*46 = 184 < 250 and A
    # dies during T5 => forced loss. The HP-leaf argmax already flips at
    # horizon 3 (SD line's cumulative 138 ties spam, pulls ahead through the
    # boost), so this cell separates "finds the win via the hp gradient"
    # (d4) from "proves it via the terminal" (d4+ with deeper margin).
    PositionSpec(
        name="sd-race-deep",
        title="Deep Swords Dance race (flip d3, terminal d4)",
        derivation=(
            "toss=100; A at 410 survives four tosses. B at 250: spam totals 184 by T4 — "
            "forced loss during T5; SD + 3x(+2 return avg 92) totals 276 with the T4 KO "
            "certain (66 <= min 85) — forced win at ply 4. Cumulative-damage flip: h2 ties "
            "(92=92), h3 SD leads (184 vs 138) before any terminal exists."
        ),
        p1=(MonSpec("Snorlax", ("return", "swordsdance"), "Thick Fat", current_hp=410),),
        p2=(MonSpec("Registeel", ("seismictoss",), "Clear Body", current_hp=250),),
        win_move="swordsdance",
        trap_move="return",
        expected_needed_depth=3,
        proof_horizon=6,
    ),
    # ------------------------------------------------------------------
    # P6 — needs depth 2; different species/tokens than P1, same recharge
    # mechanism with the speed relation INVERTED (our side faster this time).
    # Tauros (210/291, faster) holds {return, hyperbeam}; Umbreon (310/331)
    # tosses. Measured rolls: return max 120 / avg 111 / min 102; hyperbeam
    # max 177 / avg 163. A acts on T1-T3 and dies to the third toss at the
    # end of T3.
    #   return: 111+111 by T2, remaining 88 <= min 102 => certain KO at T3
    #           with A at 10 => forced WIN at ply 3 (the h2 hp gradient
    #           already separates: 88 remaining vs 147).
    #   hyperbeam: 163, recharge, then one return reaches 274 < 283 on every
    #           non-crit branch; the third toss kills A at the end of T3
    #           => loses everything but the crit lotteries. Hyper Beam is
    #           pinned to 1 PP: with a second use, hb/recharge/hb = 326 >=
    #           310 would win and the trap would not be a trap.
    PositionSpec(
        name="hb-recharge-trap-b",
        title="Hyper Beam recharge trap, second species set (needs d2)",
        derivation=(
            "toss=100; return avg 111/min 102/max 120, hb avg 163/max 177 (measured on the "
            "materialized state). B at 310: hb+return = 274 and even max rolls reach 283 < "
            "310 => the hb line cannot finish before A dies end-T3 (hb pinned to 1 PP so a "
            "second beam cannot); return x2 leaves 88 <= min 102 => certain T3 KO. d1 sees "
            "hb 163 > return 111; d2 the gradient (88 vs 147 remaining), d3 the terminal."
        ),
        p1=(MonSpec("Tauros", ("return", "hyperbeam"), "Intimidate", current_hp=210,
                    move_pp={"hyperbeam": 1}),),
        p2=(MonSpec("Umbreon", ("seismictoss",), "Synchronize", current_hp=310),),
        win_move="return",
        trap_move="hyperbeam",
        expected_needed_depth=2,
        proof_horizon=4,
    ),
)


def design_report(spec: PositionSpec, state, pe) -> dict[str, Any]:
    """Forcing proof + horizon flip table for one materialized position."""

    forced_win = solve_forced(pe, state, spec.proof_horizon)
    win_bounds = solve_win_bounds(pe, state, spec.proof_horizon)
    table = horizon_argmax_table(pe, state, spec.proof_horizon)
    flip = next(
        (row["horizon"] for row in table if row["argmax"] == spec.win_move), None
    )
    stable = all(
        row["argmax"] == spec.win_move for row in table if flip and row["horizon"] >= flip
    )
    return {
        "position": spec.name,
        "forced_win": forced_win,
        "win_bounds": win_bounds,
        "horizon_table": table,
        "argmax_flip_horizon": flip,
        "flip_stable": stable,
        "win_move_forced": bool(forced_win.get(spec.win_move)),
        "win_move_p_win_lower": win_bounds.get(spec.win_move, {}).get("p_win_lower"),
        "trap_move_p_win_upper": win_bounds.get(spec.trap_move, {}).get("p_win_upper"),
        "expected_needed_depth": spec.expected_needed_depth,
    }


def _measured_rolls(pe, state, moves: Sequence[str], opp_move: str) -> dict[str, Any]:
    """Non-crit/crit max rolls per own move on the materialized state."""

    rolls: dict[str, Any] = {}
    for move in moves:
        try:
            noncrit, _ = pe.calculate_damage(state, move, opp_move, False)
            crit, _ = pe.calculate_damage(state, move, opp_move, True)
        except Exception as error:  # noqa: BLE001 — status moves have no rolls
            rolls[move] = {"error": str(error)[:80]}
            continue
        rolls[move] = {"noncrit": noncrit, "crit": crit}
    return rolls


def build_policy(args, *, leaf: str):
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy
    from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT
    from pokezero.randbat import load_gen3_randbat_source_cached

    showdown_root = args.showdown_root or str(DEFAULT_SHOWDOWN_ROOT)
    common = dict(worlds=1, search_time_ms=10, search_sims=args.sims,
                  search_depth=2, search_batch=args.batch)
    if leaf == "model":
        from pokezero.mcts_eval.lattice import materialize_search_artifacts
        from pokezero.mcts_eval.resolver import resolve_checkpoint_contract

        contract = resolve_checkpoint_contract(
            args.checkpoint, model_device="cpu", showdown_root=showdown_root
        )
        artifacts = materialize_search_artifacts(contract, showdown_root=showdown_root)
        config = EngineMctsConfig(
            leaf_eval="model",
            checkpoint_path=args.checkpoint,
            model_path=artifacts["model_path"],
            tables_path=artifacts["tables_path"],
            model_device="cpu",
            **common,
        )
        provenance = {
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": contract.checkpoint_sha256,
            "model_path": artifacts["model_path"],
            "tables_path": artifacts["tables_path"],
            "schema": contract.schema_version,
            "transition_token_budget": contract.feature_masks.get("transition_token_budget"),
        }
    else:
        config = EngineMctsConfig(leaf_eval="hp_fraction_crate", **common)
        provenance = {}
    policy = EngineMctsPolicy(
        dex=load_showdown_dex_cached(showdown_root),
        set_source=load_gen3_randbat_source_cached(showdown_root),
        config=config,
        policy_id=f"depth-tactics-{leaf}",
    )
    return policy, provenance


def build_env(args, *, checkpoint: str | None):
    """Env for materialization. With a checkpoint, latch its full observation
    contract (masks + spec + vocab) exactly like the acceptance harness."""

    from pokezero.local_showdown import (
        DEFAULT_SHOWDOWN_ROOT,
        LocalShowdownConfig,
        LocalShowdownEnv,
        env_config_from_checkpoint_provenance,
    )

    showdown_root = args.showdown_root or str(DEFAULT_SHOWDOWN_ROOT)
    base = LocalShowdownConfig(showdown_root=showdown_root, set_belief_source=True)
    if checkpoint:
        from pokezero.neural_policy import (
            category_vocab_from_model_config,
            feature_masks_from_model_config,
            load_transformer_model_config,
            observation_spec_from_model_config,
        )

        model_config = load_transformer_model_config(checkpoint)
        vocab = category_vocab_from_model_config(model_config, showdown_root)
        base = LocalShowdownConfig(
            showdown_root=showdown_root, set_belief_source=True, category_vocab=vocab
        )
        env_config = env_config_from_checkpoint_provenance(
            base,
            feature_masks_from_model_config(model_config),
            required_specs=observation_spec_from_model_config(model_config),
            required_vocabs=vocab,
            context="depth tactics probe",
        )
        return LocalShowdownEnv(env_config)
    return LocalShowdownEnv(base)


def probe_position(env, spec: PositionSpec, args, policies) -> dict[str, Any]:
    import poke_engine as pe

    from pokezero.dex import load_showdown_dex_cached
    from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

    dex = load_showdown_dex_cached(args.showdown_root or str(DEFAULT_SHOWDOWN_ROOT))
    context, override = materialize_context(env, spec, dex, battle_id=f"tactics-{spec.name}")
    reference_policy = policies.get("hp_fraction_crate") or next(iter(policies.values()))
    world, state = build_world_state(reference_policy, context, override)
    state_str = state.to_string()

    record: dict[str, Any] = {
        "position": spec.name,
        "title": spec.title,
        "derivation": spec.derivation,
        "win_move": spec.win_move,
        "trap_move": spec.trap_move,
        "state_str": state_str,
        "rolls": _measured_rolls(
            pe, state, [m for m in json.loads(
                __import__("pokezero_search").env_options(state_str, True))["p1"]],
            json.loads(__import__("pokezero_search").env_options(state_str, True))["p2"][0],
        ),
        "design": design_report(spec, state, pe),
    }
    if args.design_only:
        return record

    searches: dict[str, Any] = {}
    for leaf, policy in policies.items():
        cells: dict[str, Any] = {}
        for depth in DEPTHS:
            runs = []
            for seed in SEARCH_SEEDS:
                if leaf == "model":
                    report = run_model(
                        policy, context, world, state_str, depth, args.sims,
                        args.batch, seed, model_priors=not args.no_model_priors,
                    )
                else:
                    report = run_hp_crate(state_str, depth, args.sims, seed)
                arms = report["side_one"]
                total = max(sum(a["visits"] for a in arms), 1)
                runs.append(
                    {
                        "seed": seed,
                        "chosen": arms[0]["move"],
                        "arms": [
                            {
                                "move": a["move"],
                                "visits": a["visits"],
                                "share": round(a["visits"] / total, 4),
                                "q": a["q"],
                            }
                            for a in arms
                        ],
                        "root_value": report.get("root_value"),
                        "max_depth_reached": report.get("max_depth_reached"),
                        "terminal_branches": report.get("terminal_branches"),
                        "deep_ko_triggers": report.get("deep_ko_triggers"),
                        "iterations": report.get("iterations"),
                        "model_evals": report.get("model_evals"),
                        "root_priors": report.get("root_priors"),
                        "elapsed_s": report.get("elapsed_s"),
                    }
                )
            chosen = [r["chosen"] for r in runs]
            cells[f"d{depth}"] = {
                "runs": runs,
                "chosen_win": sum(c == spec.win_move for c in chosen),
                "chosen_trap": sum(c == spec.trap_move for c in chosen),
                "unanimous": len(set(chosen)) == 1,
            }
        searches[leaf] = cells
    record["searches"] = searches
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Constructed-tactics depth probe (see module docstring)."
    )
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint for the model leaf; omit for hp-only")
    parser.add_argument("--showdown-root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--design-only", action="store_true",
                        help="verify forcing + horizon flips only (no searches)")
    parser.add_argument("--hp-only", action="store_true")
    parser.add_argument("--no-model-priors", action="store_true")
    parser.add_argument("--sims", type=int, default=SIMS)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--positions", nargs="*", default=None)
    args = parser.parse_args()

    wanted = set(args.positions or [p.name for p in POSITIONS])
    roster = [p for p in POSITIONS if p.name in wanted]
    if not roster:
        raise SystemExit(f"no positions matched {sorted(wanted)}")

    policies: dict[str, Any] = {}
    if not args.design_only:
        policies["hp_fraction_crate"], _ = build_policy(args, leaf="hp_fraction_crate")
        provenance: dict[str, Any] = {}
        if args.checkpoint and not args.hp_only:
            policies["model"], provenance = build_policy(args, leaf="model")
    else:
        policies["hp_fraction_crate"], _ = build_policy(args, leaf="hp_fraction_crate")
        provenance = {}

    import subprocess

    fingerprint = json.loads(
        (Path(sys.prefix) / ".engine-build-fingerprint.json").read_text()
    ) if (Path(sys.prefix) / ".engine-build-fingerprint.json").exists() else {}
    payload: dict[str, Any] = {
        "probe": "depth-tactics",
        "sims": args.sims,
        "batch": args.batch,
        "depths": list(DEPTHS),
        "seeds": list(SEARCH_SEEDS),
        "c_puct": C_PUCT,
        "engine_fingerprint": fingerprint.get("fingerprint"),
        "engine_patches": len(fingerprint.get("patches", []) or []),
        "commit": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip(),
        "checkpoint_provenance": provenance,
        "positions": [],
    }

    env = build_env(args, checkpoint=None if args.design_only or args.hp_only else args.checkpoint)
    try:
        for spec in roster:
            print(f"=== {spec.name} ===", flush=True)
            record = probe_position(env, spec, args, policies)
            design = record["design"]
            print(
                f"  forced win[{spec.win_move}]={design['win_move_forced']} "
                f"(p_lower={design['win_move_p_win_lower']}) "
                f"trap[{spec.trap_move}] p_upper={design['trap_move_p_win_upper']} "
                f"flip@h{design['argmax_flip_horizon']} (expected {spec.expected_needed_depth})",
                flush=True,
            )
            for leaf, cells in (record.get("searches") or {}).items():
                summary = " ".join(
                    f"d{d}:{cells[f'd{d}']['chosen_win']}/{len(SEARCH_SEEDS)}win"
                    for d in DEPTHS
                )
                print(f"  [{leaf}] {summary}", flush=True)
            payload["positions"].append(record)
    finally:
        env.close()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
