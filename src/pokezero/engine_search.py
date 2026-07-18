"""POC: native-engine MCTS over belief-sampled worlds (engine swap plan v3).

FoulPlay's architecture on pokezero's belief engine: per decision, sample K
determinized worlds from the public belief (the existing
``gen3_randbat_belief_start_override`` planner), construct each as a
poke-engine state via the track-A world constructor, run poke-engine's native
multi-ply MCTS per world for a fixed time budget, and aggregate the acting
side's root visit distributions across worlds. ~10⁵ simulations per decision
in a few hundred milliseconds — the speed regime the plan targets.

Deliberate POC boundaries (the tradeoffs this exists to measure):

- **No learned model in the loop.** Leaves are priced by poke-engine's
  handcrafted evaluation and the tree explores without our policy priors.
  This isolates the speed question from the strength question; the paired
  +10pt result came from the learned value, so POC speed does NOT imply POC
  strength. Track B (encoder) is what puts the learned model on this path.
- **Fail-closed construction.** Decisions whose worlds cannot be expressed
  exactly (see ``engine_world``'s reason taxonomy) fall back to uniform
  legal; the bench reports the rate and taxonomy rather than hiding it.
- **Uniform world weights.** FoulPlay weights worlds by sample likelihood;
  the belief planner does not expose one yet.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .dex import ShowdownDex, normalize_id
from .determinization import (
    _gen3_randbat_belief_start_override_result,
    _move_from_public_event_line,
)
from .public_action_capture import public_action_rounds_from_trajectory_metadata
from .engine_world import EngineWorld, EngineWorldUnsupported, world_battle_spec
from .poke_engine_adapter import build_poke_engine_state
from .policy import PolicyContext, PolicyDecision, legal_action_indices


@dataclass(frozen=True)
class EngineMctsConfig:
    worlds: int = 4
    search_time_ms: int = 100
    threads: int = 1
    # Documented approximation (see engine_world): model publicly-asleep mons
    # as freshly asleep instead of failing the whole world closed. Without it
    # the fallback rate is dominated by sleep (~60% of decisions in smokes).
    approximate_sleep_turns: bool = True
    # Belief sampling is stochastic; failed draws are retried up to
    # worlds * sample_retry_factor total attempts (mirrors the W1 retry fix).
    sample_retry_factor: int = 4
    # Documented approximation: a public Substitute is modeled at fresh
    # (maxhp/4) health, since remaining sub HP is not tracked publicly.
    approximate_substitute_health: bool = True

    def __post_init__(self) -> None:
        if self.worlds <= 0 or self.search_time_ms <= 0 or self.threads <= 0:
            raise ValueError("worlds, search_time_ms, and threads must be positive.")


@dataclass
class EngineMctsStats:
    """Cumulative per-policy telemetry; every fallback is counted, never hidden."""

    decisions: int = 0
    searched_decisions: int = 0
    fallback_decisions: int = 0
    worlds_attempted: int = 0
    worlds_searched: int = 0
    total_iterations: int = 0
    search_wall_seconds: float = 0.0
    decision_wall_seconds: float = 0.0
    world_failure_reasons: Counter = field(default_factory=Counter)
    fallback_reasons: Counter = field(default_factory=Counter)
    unmapped_choices: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "decisions": self.decisions,
            "searched_decisions": self.searched_decisions,
            "fallback_decisions": self.fallback_decisions,
            "fallback_rate": self.fallback_decisions / self.decisions if self.decisions else 0.0,
            "worlds_attempted": self.worlds_attempted,
            "worlds_searched": self.worlds_searched,
            "total_iterations": self.total_iterations,
            "search_wall_seconds": self.search_wall_seconds,
            "decision_wall_seconds": self.decision_wall_seconds,
            "world_failure_reasons": dict(self.world_failure_reasons),
            "fallback_reasons": dict(self.fallback_reasons),
            "unmapped_choices": dict(self.unmapped_choices),
        }
        if self.searched_decisions:
            payload["iterations_per_searched_decision"] = (
                self.total_iterations / self.searched_decisions
            )
            payload["search_wall_per_searched_decision"] = (
                self.search_wall_seconds / self.searched_decisions
            )
        if self.decisions:
            payload["wall_per_decision"] = self.decision_wall_seconds / self.decisions
        return payload


class EngineMctsPolicy:
    """ContextAwarePolicy running poke-engine MCTS over belief-sampled worlds."""

    def __init__(
        self,
        *,
        dex: ShowdownDex,
        set_source: Any,
        config: EngineMctsConfig | None = None,
        module: Any | None = None,
        policy_id: str = "engine-mcts",
        fixed_override: Any | None = None,
    ) -> None:
        if module is None:
            import poke_engine as module  # noqa: PLC0415 — optional native dependency

        self.policy_id = policy_id
        # Test/scenario hook: bypass belief sampling and use this override as
        # every world (custom-game sweeps where the catalog cannot sample).
        self._fixed_override = fixed_override
        self._dex = dex
        self._set_source = set_source
        self._config = config or EngineMctsConfig()
        self._module = module
        self.stats = EngineMctsStats()

    # Policy protocol (context-free path): uniform legal. Only reached if the
    # rollout driver cannot supply a context, which the bench never does.
    def select_action(self, observation, *, rng: random.Random) -> PolicyDecision:
        legal = legal_action_indices(observation.legal_action_mask)
        return PolicyDecision(action_index=rng.choice(legal), policy_id=self.policy_id)

    def select_action_with_context(
        self, context: PolicyContext, *, rng: random.Random
    ) -> PolicyDecision:
        started = time.perf_counter()
        decision = self._search(context, rng=rng)
        self.stats.decisions += 1
        self.stats.decision_wall_seconds += time.perf_counter() - started
        return decision

    # -----------------------------------------------------------------------------------------

    def _search(self, context: PolicyContext, *, rng: random.Random) -> PolicyDecision:
        if context.public_materialization_state is None:
            return self._fallback(context, rng, "no_public_state")
        blocked_slots, encored_moves = self._public_effect_signals(context)
        recharging_slots = self._recharging_slots(context)
        truant_slots = self._truant_loaf_slots(context)

        worlds: list[tuple[EngineWorld, Any]] = []
        attempts_budget = self._config.worlds * self._config.sample_retry_factor
        attempts = 0
        while len(worlds) < self._config.worlds and attempts < attempts_budget:
            attempts += 1
            self.stats.worlds_attempted += 1
            if self._fixed_override is not None:
                override, sample_failure = self._fixed_override, None
            else:
                override, sample_failure = _gen3_randbat_belief_start_override_result(
                    context=context,
                    set_source=self._set_source,
                    rng=rng,
                    witnessed_fallback=True,
                )
            if override is None:
                self.stats.world_failure_reasons[
                    f"belief_sample: {sample_failure or 'unknown'}"
                ] += 1
                continue
            try:
                world = world_battle_spec(
                    context.public_materialization_state,
                    override,
                    dex=self._dex,
                    approximate_sleep_turns=self._config.approximate_sleep_turns,
                    approximate_substitute_health=self._config.approximate_substitute_health,
                    blocked_slots=blocked_slots,
                    encored_moves=encored_moves,
                    recharging_slots=recharging_slots,
                    truant_slots=truant_slots,
                    rng=rng,
                )
                state = build_poke_engine_state(world.spec, module=self._module)
            except EngineWorldUnsupported as error:
                key = error.reason
                if key in ("volatile_unsupported", "hidden_power_iv_mismatch", "wish_carrier_ambiguous"):
                    key = f"{error.reason}: {error.detail}"
                self.stats.world_failure_reasons[key] += 1
                continue
            worlds.append((world, state))

        if not worlds:
            return self._fallback(context, rng, "no_worlds_constructed")

        aggregated: Counter = Counter()
        search_started = time.perf_counter()
        for world, state in worlds:
            result = self._module.monte_carlo_tree_search(
                state, self._config.search_time_ms, threads=self._config.threads
            )
            own_side = (
                result.side_one
                if world.slot_sides[context.player_id] == "side_one"
                else result.side_two
            )
            total = max(result.total_visits, 1)
            for entry in own_side:
                aggregated[entry.move_choice] += entry.visits / total
            self.stats.total_iterations += result.total_visits
            self.stats.worlds_searched += 1
        self.stats.search_wall_seconds += time.perf_counter() - search_started

        action_index = self._map_choices(context, aggregated)
        if action_index is None:
            return self._fallback(context, rng, "choices_unmapped")

        self.stats.searched_decisions += 1
        return PolicyDecision(
            action_index=action_index,
            policy_id=self.policy_id,
            metadata={
                "engine_mcts": {
                    "worlds_searched": len(worlds),
                    "aggregated_choices": {
                        choice: round(weight, 4) for choice, weight in aggregated.most_common()
                    },
                }
            },
        )



    # Gen 3 pool's only recharge move; the recharge turn itself is public.
    _RECHARGE_MOVES = frozenset({"hyperbeam"})

    def _recharging_slots(self, context: PolicyContext) -> tuple[str, ...]:
        """Slots publicly forced to recharge THIS turn (Hyper Beam landed last round).

        Turn-exact signal: the round-indexed public action record (not the
        rolling event window) must show the opponent's action in the
        immediately-preceding round was a recharge move, and the rolling
        window must not carry a miss marker for it (a missed Hyper Beam does
        not recharge in gen3). If the record is unavailable the signal stays
        off — fail-open to the pre-fix behavior rather than inventing a lock.
        """

        opponent_slot = "p2" if context.player_id == "p1" else "p1"
        trajectory = getattr(context, "trajectory", None)
        round_index = getattr(context, "decision_round_index", None)
        if trajectory is None or not isinstance(round_index, int):
            return ()
        rounds = public_action_rounds_from_trajectory_metadata(trajectory)
        previous = rounds.get(round_index - 1)
        if previous is None:
            return ()
        action = previous.actions.get(opponent_slot)
        if action is None or action.kind != "move":
            return ()
        if normalize_id(str(action.move_id or "")) not in self._RECHARGE_MOVES:
            return ()
        # The round record proves the move happened but stores no hit/miss and
        # no actor identity. Require the ANCHOR: the |move| line must still be
        # visible in the rolling event window, its actor must match the
        # CURRENT active opponent (species continuity — double-faint guard),
        # and no adjacent |-miss| may follow. If the anchor scrolled out we
        # cannot verify the hit, so the lock stays OFF (fail-open to the
        # pre-fix behavior — never a wrong lock on a missed Hyper Beam).
        metadata = context.observation.metadata
        if not isinstance(metadata, Mapping):
            return ()
        belief_view = metadata.get("belief_view")
        opponents = belief_view.get("opponent_pokemon") if isinstance(belief_view, Mapping) else None
        active_species = next(
            (
                str(mon.get("species") or "")
                for mon in opponents or ()
                if isinstance(mon, Mapping) and mon.get("active")
            ),
            "",
        )
        if not active_species:
            return ()
        events = metadata.get("recent_public_events")
        if not isinstance(events, Sequence):
            return ()
        lines = [str(line) for line in events]
        for index in range(len(lines) - 1, -1, -1):
            parts = lines[index].split("|")
            if len(parts) < 4 or parts[1] != "move":
                continue
            if normalize_id(parts[3]) not in self._RECHARGE_MOVES:
                continue
            actor = parts[2]
            actor_species = actor.split(":", 1)[-1].strip() if ":" in actor else actor
            if normalize_id(actor_species) != normalize_id(active_species):
                return ()
            if not actor.strip().lower().startswith(opponent_slot):
                return ()
            if any(rest.startswith(f"|-miss|{actor}") for rest in lines[index + 1 : index + 3]):
                return ()
            return (opponent_slot,)
        return ()


    def _truant_loaf_slots(self, context: PolicyContext) -> tuple[str, ...]:
        """Slots whose active is a Truant mon that ACTED last round (loafs now).

        The alternation is public: a Truant mon that publicly moved in the
        immediately-preceding round loafs this turn. Evidence of acting comes
        from the round-indexed public action record (turn-exact). Without
        clear acted-last-round evidence the volatile stays off (fail-open:
        the mon is modeled as free to act — the pre-fix behavior).
        """

        trajectory = getattr(context, "trajectory", None)
        round_index = getattr(context, "decision_round_index", None)
        if trajectory is None or not isinstance(round_index, int):
            return ()
        rounds = public_action_rounds_from_trajectory_metadata(trajectory)
        previous = rounds.get(round_index - 1)
        if previous is None:
            return ()
        metadata = context.observation.metadata
        if not isinstance(metadata, Mapping):
            return ()
        slots: list[str] = []
        opponent_slot = "p2" if context.player_id == "p1" else "p1"
        belief_view = metadata.get("belief_view")
        opponents = belief_view.get("opponent_pokemon") if isinstance(belief_view, Mapping) else None
        for mon in opponents or ():
            if not isinstance(mon, Mapping) or not mon.get("active"):
                continue
            ability = normalize_id(str(mon.get("revealed_ability") or ""))
            possible = [normalize_id(str(a)) for a in mon.get("possible_abilities") or ()]
            if ability == "truant" or (not ability and possible == ["truant"]):
                action = previous.actions.get(opponent_slot)
                if action is not None and action.kind == "move":
                    slots.append(opponent_slot)
        # Self seat: our own Truant mon's phase from our own action record.
        self_team = metadata.get("self_team")
        if isinstance(self_team, Sequence):
            for row in self_team:
                if not isinstance(row, Mapping) or not row.get("active"):
                    continue
                if normalize_id(str(row.get("ability") or "")) == "truant":
                    action = previous.actions.get(context.player_id)
                    if action is not None and action.kind == "move":
                        slots.append(context.player_id)
        return tuple(slots)

    def _public_effect_signals(
        self, context: PolicyContext
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Public-information signals engine_world cannot see in the payload.

        - blocked_slots: the opponent's active is publicly Transformed (the
          belief engine tracks it; the payload does not) — the sampled world
          cannot express the copied moveset/stats, so construction must fail
          closed rather than search a silently wrong world.
        - encored_moves: the opponent's publicly-observed last move, consumed
          by engine_world only when that side carries the encore volatile.
        """

        blocked: dict[str, str] = {}
        encored: dict[str, str] = {}
        metadata = context.observation.metadata
        if not isinstance(metadata, Mapping):
            return blocked, encored
        opponent_slot = "p2" if context.player_id == "p1" else "p1"
        belief_view = metadata.get("belief_view")
        opponents = belief_view.get("opponent_pokemon") if isinstance(belief_view, Mapping) else None
        active_species: str | None = None
        for mon in opponents or ():
            if not isinstance(mon, Mapping):
                continue
            if mon.get("item_mutated"):
                # Trick/Knock Off mutated the held item: the sampled set's item
                # no longer matches the current holder (rule-outs stay frozen
                # to the ORIGINAL assignment upstream). Pool scope: 6 sets set
                # this flag (4 Knock Off + 2 Trick). Knock Off removals ARE
                # representable (item publicly None) — recovering them needs a
                # belief_view field distinguishing removal from swap; until
                # then fail closed over constructing wrong items.
                blocked[opponent_slot] = f"item mutated on {mon.get('species')}"
            if not mon.get("active"):
                continue
            active_species = str(mon.get("species") or "") or None
            if mon.get("transformed"):
                target = mon.get("transform_species") or "?"
                blocked[opponent_slot] = f"active transformed into {target}"
        if active_species:
            events = metadata.get("recent_public_events")
            for line in reversed(list(events) if isinstance(events, Sequence) else []):
                move = _move_from_public_event_line(
                    str(line),
                    opponent_slot=opponent_slot,
                    self_slot=context.player_id,
                    species=active_species,
                )
                if move is not None:
                    encored[opponent_slot] = move
                    break
        return blocked, encored

    def _map_choices(
        self, context: PolicyContext, aggregated: Mapping[str, float]
    ) -> Optional[int]:
        candidates = context.observation.metadata.get("action_candidates")
        if not isinstance(candidates, Sequence):
            return None
        mask = context.observation.legal_action_mask

        move_index_by_id: dict[str, int] = {}
        hidden_power_index: Optional[int] = None
        switch_index_by_species: dict[str, int] = {}
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not candidate.get("legal"):
                continue
            index = candidate.get("action_index")
            if not isinstance(index, int) or not (0 <= index < len(mask)) or not mask[index]:
                continue
            if candidate.get("kind") == "move":
                move_id = normalize_id(str(candidate.get("move_id") or ""))
                if move_id:
                    move_index_by_id[move_id] = index
                    if move_id.startswith("hiddenpower"):
                        hidden_power_index = index
            elif candidate.get("kind") == "switch":
                pokemon = candidate.get("pokemon")
                species = (
                    normalize_id(str(pokemon.get("species") or ""))
                    if isinstance(pokemon, Mapping)
                    else ""
                )
                if species:
                    switch_index_by_species[species] = index

        best_index: Optional[int] = None
        best_weight = 0.0
        for choice, weight in aggregated.items():
            index: Optional[int] = None
            if choice.startswith("switch "):
                index = switch_index_by_species.get(normalize_id(choice[len("switch "):]))
            else:
                move_id = normalize_id(choice)
                index = move_index_by_id.get(move_id)
                if index is None and move_id.startswith("hiddenpower"):
                    # Engine ids are typed+BP; the request reports plain "hiddenpower".
                    index = hidden_power_index
            if index is None:
                self.stats.unmapped_choices[choice] += 1
                continue
            if weight > best_weight:
                best_weight = weight
                best_index = index
        return best_index

    def _fallback(
        self, context: PolicyContext, rng: random.Random, reason: str
    ) -> PolicyDecision:
        self.stats.fallback_decisions += 1
        self.stats.fallback_reasons[reason] += 1
        legal = legal_action_indices(context.observation.legal_action_mask)
        return PolicyDecision(
            action_index=rng.choice(legal),
            policy_id=self.policy_id,
            metadata={"engine_mcts": {"fallback": reason}},
        )


# ---------------------------------------------------------------------------------------------
# Bench CLI.
# ---------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="POC bench: poke-engine MCTS over belief worlds vs a baseline"
    )
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=7000)
    parser.add_argument("--opponent", choices=("random-legal", "simple-legal"), default="simple-legal")
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--search-time-ms", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--strict-sleep", action="store_true",
                        help="fail worlds closed on publicly-asleep mons instead of approximating")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from .dex import load_showdown_dex
    from .local_showdown import LocalShowdownConfig, LocalShowdownEnv
    from .policy import RandomLegalPolicy, SimpleLegalPolicy
    from .randbat import Gen3RandbatSource
    from .rollout import RolloutConfig, RolloutDriver

    dex = load_showdown_dex(args.showdown_root)
    set_source = Gen3RandbatSource.from_showdown_root(args.showdown_root)
    policy = EngineMctsPolicy(
        dex=dex,
        set_source=set_source,
        config=EngineMctsConfig(
            worlds=args.worlds,
            search_time_ms=args.search_time_ms,
            threads=args.threads,
            approximate_sleep_turns=not args.strict_sleep,
        ),
    )
    opponent = RandomLegalPolicy() if args.opponent == "random-legal" else SimpleLegalPolicy()

    env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=args.showdown_root))
    driver = RolloutDriver(
        env=env,
        policies={"p1": policy, "p2": opponent},
        config=RolloutConfig(format_id="gen3randombattle"),
    )
    wins = 0
    games = []
    try:
        for offset in range(args.games):
            seed = args.seed_start + offset
            result = driver.run(seed=seed, battle_id=f"engine-mcts-bench-{seed}")
            won = result.terminal.winner == "p1"
            wins += int(won)
            games.append({
                "seed": seed,
                "winner": result.terminal.winner,
                "decision_rounds": result.decision_round_count,
            })
            print(f"seed {seed}: winner={result.terminal.winner} rounds={result.decision_round_count}")
    finally:
        env.close()

    report = {
        "config": {
            "worlds": args.worlds,
            "search_time_ms": args.search_time_ms,
            "threads": args.threads,
            "approximate_sleep_turns": not args.strict_sleep,
            "opponent": args.opponent,
            "games": args.games,
        },
        "wins": wins,
        "win_rate": wins / args.games if args.games else 0.0,
        "games": games,
        "engine_mcts": policy.stats.to_dict(),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "games"}, indent=2))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
