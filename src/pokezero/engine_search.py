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
from .determinization import _gen3_randbat_belief_start_override_result
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
    ) -> None:
        if module is None:
            import poke_engine as module  # noqa: PLC0415 — optional native dependency

        self.policy_id = policy_id
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

        worlds: list[tuple[EngineWorld, Any]] = []
        attempts_budget = self._config.worlds * self._config.sample_retry_factor
        attempts = 0
        while len(worlds) < self._config.worlds and attempts < attempts_budget:
            attempts += 1
            self.stats.worlds_attempted += 1
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
