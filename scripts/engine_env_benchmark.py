#!/usr/bin/env python
"""Micro-benchmark: engine-as-environment vs LocalShowdownEnv self-play.

Plays the SAME seeds through both backends with the SAME checkpoint and reports
games/sec plus a per-decision latency breakdown. Both arms run in this process,
single-threaded, so the comparison is apples-to-apples; the only difference is
which :class:`~pokezero.env.PokeZeroEnv` implementation advances the battle.

The breakdown separates what the env does from what the policy does:

  * ``env step``      — advancing one ply (engine transition, or a Showdown
                        bridge boundary round-trip)
  * ``env encode``    — producing one player's observation
  * ``policy fwd``    — the model forward, identical work in both arms
  * ``other``         — everything left over inside a game (bookkeeping, action
                        translation, public-info replay)
  * ``teams``         — per-game team generation (engine arm only; the Showdown
                        arm generates teams inside its own reset)

Checkpoint: any budget-0 v2.2 checkpoint works. ``--init-checkpoint`` writes a
FRESHLY-INITIALIZED model at a given architecture instead — for a wall-time
measurement policy quality is irrelevant, only the forward's shape and cost
are, and no trained budget-0 checkpoint may exist yet.

Usage:
    python scripts/engine_env_benchmark.py --games 200 --checkpoint <path>
    python scripts/engine_env_benchmark.py --init-checkpoint <path>   # then rerun
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pokezero.local_showdown import CONFIG_DEFAULT_OBSERVATION_SPEC  # noqa: E402
from pokezero.observation import ObservationFeatureMasks  # noqa: E402


# ---------------------------------------------------------------------------
# Fresh checkpoint (no trained budget-0 model exists yet)
# ---------------------------------------------------------------------------


def write_initialized_checkpoint(path: Path, reference: Path | None) -> None:
    """Save an untrained model at a real architecture, stamped budget 0."""
    import dataclasses

    from pokezero.neural_policy import (
        EntityTokenTransformerPolicy,
        TransformerEpochMetrics,
        TransformerTrainingConfig,
        TransformerTrainingResult,
        load_transformer_model_config,
        save_transformer_checkpoint,
    )

    reference = reference or (REPO / "checkpoints" / "pz-v2-2-1m.pt")
    config = load_transformer_model_config(reference)
    config = dataclasses.replace(
        config,
        policy_id="engine-env-benchmark-init",
        transition_token_budget=0,
    )
    model = EntityTokenTransformerPolicy(config)
    result = TransformerTrainingResult(
        model_config=config,
        training_config=TransformerTrainingConfig(),
        epochs=(
            TransformerEpochMetrics(
                epoch=0, examples=0, loss=0.0, policy_loss=0.0, policy_accuracy=0.0
            ),
        ),
    )
    save_transformer_checkpoint(path, model, result=result)
    print(
        f"wrote freshly-initialized checkpoint {path}\n"
        f"  architecture copied from {reference.name}: "
        f"dim={config.embedding_dim} layers={config.transformer_layers} "
        f"heads={config.attention_heads} ff={config.feedforward_dim}\n"
        f"  transition_token_budget=0 (k=0), schema={config.observation_schema_version}\n"
        "  WEIGHTS ARE UNTRAINED — valid for timing, meaningless for strength."
    )


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    label: str
    games: int = 0
    decisions: int = 0
    plies: int = 0
    turns: int = 0
    wall: float = 0.0
    reset: float = 0.0
    step: float = 0.0
    encode: float = 0.0
    policy: float = 0.0
    teams: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)
    game_seconds: list[float] = field(default_factory=list)

    @property
    def games_per_second(self) -> float:
        return self.games / self.wall if self.wall > 0 else 0.0

    @property
    def decisions_per_second(self) -> float:
        return self.decisions / self.wall if self.wall > 0 else 0.0

    def per_decision_us(self, seconds: float) -> float:
        return seconds / self.decisions * 1e6 if self.decisions else 0.0

    @property
    def other(self) -> float:
        """Wall time inside a game not attributed to a measured call."""
        return max(0.0, self.wall - (self.step + self.policy + self.reset))

    @property
    def env_seconds(self) -> float:
        """Everything the ENV costs: total minus the model forward.

        This is the number the env swap actually moves. `games/sec` includes
        the policy forward, which is identical work in both arms and therefore
        dilutes the ratio toward 1.
        """
        return max(0.0, self.wall - self.policy)


class _TimedPolicy:
    """Wraps a policy so the model forward is measured, not estimated."""

    def __init__(self, inner):
        self._inner = inner
        self.seconds = 0.0
        self.calls = 0

    def select_action(self, observation, *, rng):
        started = time.perf_counter()
        try:
            return self._inner.select_action(observation, rng=rng)
        finally:
            self.seconds += time.perf_counter() - started
            self.calls += 1


def _play(env, policies, seed, format_id, max_rounds, arm: ArmResult) -> None:
    """One self-play game, timing the env's own calls.

    Deliberately NOT `RolloutDriver.run`: the driver builds trajectory records
    whose cost is identical for both arms and would only dilute the signal.
    The decision loop is otherwise the driver's exact shape — in particular it
    consumes the observations `step()` RETURNS rather than calling `observe()`
    again.

    That detail is load-bearing. Both envs eagerly observe the next requested
    players inside `step()`, so an extra `observe()` in the loop would be
    served from cache by one env (EngineEnv memoizes per decision point) and
    recomputed from scratch by the other (LocalShowdownEnv re-derives), making
    the step/encode split incomparable. Attributing encode-inside-step to
    `step` for BOTH arms keeps the totals symmetric; the engine arm's internal
    counters supply the finer split.
    """
    rngs = {player: random.Random(f"{seed}:{player}".__hash__() & 0xFFFFFFFF) for player in ("p1", "p2")}

    started = time.perf_counter()
    env.reset(seed=seed, format_id=format_id)
    requested = env.requested_players()
    observations = {player: env.observe(player) for player in requested}
    arm.reset += time.perf_counter() - started

    rounds = 0
    while env.terminal() is None and requested and rounds < max_rounds:
        actions = {}
        for player in requested:
            arm.decisions += 1
            decision = policies[player].select_action(observations[player], rng=rngs[player])
            actions[player] = decision.action_index
        step_started = time.perf_counter()
        result = env.step(actions)
        arm.step += time.perf_counter() - step_started
        arm.plies += 1
        rounds += 1
        requested = tuple(result.requested_players)
        observations = dict(result.observations)

    terminal = env.terminal()
    arm.turns += int(terminal.turn_count) if terminal is not None else 0
    arm.games += 1


def _random_legal_policy():
    from pokezero.policy import PolicyDecision

    class _Policy:
        def select_action(self, observation, *, rng):
            legal = [i for i, ok in enumerate(observation.legal_action_mask) if ok]
            return PolicyDecision(
                action_index=rng.choice(legal) if legal else 0, policy_id="random-legal"
            )

    return _Policy()


def _neural_policy(checkpoint: Path):
    from pokezero.neural_policy import load_transformer_policy

    # deterministic=False (sampling) matches collection-time behavior; the
    # forward cost is the same either way, but the action distribution — and
    # therefore game length — should look like collection, not like eval.
    return load_transformer_policy(checkpoint, deterministic=False, exploration_epsilon=0.0)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def run_engine_arm(args, policy_factory, masks, observation_spec) -> ArmResult:
    from pokezero.engine_env import EngineEnv, EngineEnvConfig

    env = EngineEnv(
        EngineEnvConfig(
            showdown_root=args.showdown_root,
            node_binary=args.node_binary,
            feature_masks=masks,
            encoder_tables=args.encoder_tables,
            observation_spec=observation_spec,
        )
    )
    arm = ArmResult("engine")
    policies = {player: _TimedPolicy(policy_factory()) for player in ("p1", "p2")}
    try:
        # Warm the bridge, the dex, the encoder tables and the torch graph so
        # the measured window is steady-state, not first-call cost.
        _play(env, policies, args.seed_start - 1, args.format_id, args.max_decision_rounds, ArmResult("warmup"))
        base = json.loads(json.dumps(env.timings.as_dict()))
        for policy in policies.values():
            policy.seconds = 0.0
            policy.calls = 0

        wall_started = time.perf_counter()
        for index in range(args.games):
            game_started = time.perf_counter()
            _play(env, policies, args.seed_start + index, args.format_id, args.max_decision_rounds, arm)
            arm.game_seconds.append(time.perf_counter() - game_started)
        arm.wall = time.perf_counter() - wall_started
    finally:
        env.close()

    timings = env.timings.as_dict()
    delta = {key: timings[key] - base.get(key, 0.0) for key in timings}
    arm.policy = sum(policy.seconds for policy in policies.values())
    # `arm.step` / `arm.encode` stay as the OUTER measurement of env.step() and
    # env.observe(), which is what the Showdown arm reports too — symmetric and
    # complete. The env's internal counters are finer but do not cover
    # everything inside those calls (root-inputs JSON, encoder rebuild), so
    # they are reported as a sub-split, not substituted for the total.
    arm.teams = delta["teams_s"]
    arm.extra = {
        "step+observe (total)": arm.step,
        "  engine transition": delta["step_s"],
        "  native encode": delta["encode_s"],
        "  buffer materialize": delta["materialize_s"],
        "  action map": delta["action_map_s"],
        "  public-info ledger": delta["ledger_s"],
        "reset (total)": arm.reset,
        "  team generation": delta["teams_s"],
    }
    return arm


def run_showdown_arm(args, policy_factory, masks, observation_spec) -> ArmResult:
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv

    env = LocalShowdownEnv(
        LocalShowdownConfig(
            showdown_root=args.showdown_root,
            node_binary=args.node_binary,
            feature_masks=masks,
            observation_spec=observation_spec,
        )
    )
    arm = ArmResult("showdown")
    policies = {player: _TimedPolicy(policy_factory()) for player in ("p1", "p2")}
    try:
        _play(env, policies, args.seed_start - 1, args.format_id, args.max_decision_rounds, ArmResult("warmup"))
        for policy in policies.values():
            policy.seconds = 0.0
            policy.calls = 0

        wall_started = time.perf_counter()
        for index in range(args.games):
            game_started = time.perf_counter()
            _play(env, policies, args.seed_start + index, args.format_id, args.max_decision_rounds, arm)
            arm.game_seconds.append(time.perf_counter() - game_started)
        arm.wall = time.perf_counter() - wall_started
    finally:
        env.close()

    arm.policy = sum(policy.seconds for policy in policies.values())
    arm.extra = {
        "bridge step + observe (boundary round-trip, parse, python encode)": arm.step,
        "reset (incl. team generation + first observe)": arm.reset,
    }
    arm.teams = arm.reset  # informational: Showdown generates teams in reset()
    return arm


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(engine: ArmResult, showdown: ArmResult, args) -> None:
    def ratio(a: float, b: float) -> str:
        return f"{b / a:.2f}x" if a > 0 else "n/a"

    print("\n" + "=" * 78)
    print("ENGINE-AS-ENVIRONMENT MICRO-BENCHMARK")
    print("=" * 78)
    print(f"games/arm            {args.games}")
    print(f"seeds                {args.seed_start}..{args.seed_start + args.games - 1} (identical both arms)")
    print(f"format               {args.format_id}")
    print(f"policy               {args.policy_label}")
    print(f"transition budget    k=0 (Markov-only)")
    import torch  # noqa: PLC0415
    print(f"threads              1 worker; torch intra-op threads={torch.get_num_threads()}")

    print(f"\n{'':34s} {'engine':>14s} {'showdown':>14s} {'speedup':>10s}")
    print("-" * 78)
    print(f"{'games / sec':34s} {engine.games_per_second:14.3f} {showdown.games_per_second:14.3f} "
          f"{ratio(showdown.games_per_second, engine.games_per_second):>10s}")
    print(f"{'decisions / sec':34s} {engine.decisions_per_second:14.1f} {showdown.decisions_per_second:14.1f} "
          f"{ratio(showdown.decisions_per_second, engine.decisions_per_second):>10s}")
    print(f"{'sec / game (mean)':34s} {engine.wall / max(engine.games,1):14.4f} "
          f"{showdown.wall / max(showdown.games,1):14.4f} "
          f"{ratio(engine.wall / max(engine.games,1), showdown.wall / max(showdown.games,1)):>10s}")
    median_e = statistics.median(engine.game_seconds) if engine.game_seconds else 0.0
    median_s = statistics.median(showdown.game_seconds) if showdown.game_seconds else 0.0
    print(f"{'sec / game (median)':34s} {median_e:14.4f} {median_s:14.4f} {ratio(median_e, median_s):>10s}")
    print(f"{'env sec / game (policy excluded)':34s} {engine.env_seconds / max(engine.games,1):14.4f} "
          f"{showdown.env_seconds / max(showdown.games,1):14.4f} "
          f"{ratio(engine.env_seconds / max(engine.games,1), showdown.env_seconds / max(showdown.games,1)):>10s}")
    print(f"{'decisions / game':34s} {engine.decisions / max(engine.games,1):14.1f} "
          f"{showdown.decisions / max(showdown.games,1):14.1f}")
    print(f"{'turns / game':34s} {engine.turns / max(engine.games,1):14.1f} "
          f"{showdown.turns / max(showdown.games,1):14.1f}")

    print(f"\nPER-DECISION LATENCY (microseconds)")
    print(f"{'':34s} {'engine':>14s} {'showdown':>14s} {'speedup':>10s}")
    print("-" * 78)
    for label, e_value, s_value in (
        ("env step+encode (per ply)", engine.step / max(engine.plies, 1) * 1e6,
         showdown.step / max(showdown.plies, 1) * 1e6),
        ("env step+encode (per decision)", engine.per_decision_us(engine.step),
         showdown.per_decision_us(showdown.step)),
        ("policy forward (per decision)", engine.per_decision_us(engine.policy),
         showdown.per_decision_us(showdown.policy)),
        ("env reset (per decision)", engine.per_decision_us(engine.reset),
         showdown.per_decision_us(showdown.reset)),
        ("other (per decision)", engine.per_decision_us(engine.other),
         showdown.per_decision_us(showdown.other)),
        ("ENV TOTAL (policy excluded)", engine.per_decision_us(engine.env_seconds),
         showdown.per_decision_us(showdown.env_seconds)),
        ("TOTAL (per decision)", engine.per_decision_us(engine.wall),
         showdown.per_decision_us(showdown.wall)),
    ):
        print(f"{label:34s} {e_value:14.1f} {s_value:14.1f} {ratio(e_value, s_value):>10s}")

    for arm in (engine, showdown):
        print(f"\n{arm.label.upper()} ARM — component split "
              f"(share of {arm.wall:.2f}s wall, us/decision)")
        print("-" * 78)
        for label, seconds in sorted(arm.extra.items(), key=lambda kv: -kv[1]):
            print(f"  {label:44s} {seconds:8.3f}s  {seconds / arm.wall * 100:5.1f}%  "
                  f"{arm.per_decision_us(seconds):8.1f} us")
        print(f"  {'policy forward':44s} {arm.policy:8.3f}s  {arm.policy / arm.wall * 100:5.1f}%  "
              f"{arm.per_decision_us(arm.policy):8.1f} us")
        print(f"  {'unattributed':44s} {arm.other:8.3f}s  {arm.other / arm.wall * 100:5.1f}%  "
              f"{arm.per_decision_us(arm.other):8.1f} us")

    engine_16k = 16000 / engine.games_per_second if engine.games_per_second else float("inf")
    showdown_16k = 16000 / showdown.games_per_second if showdown.games_per_second else float("inf")
    print(f"\nEXTRAPOLATION — 16k-game k=0 smoke, single worker, this machine")
    print("-" * 78)
    print(f"  engine   {engine_16k / 3600:8.2f} h  ({engine_16k:9.0f} s)")
    print(f"  showdown {showdown_16k / 3600:8.2f} h  ({showdown_16k:9.0f} s)")
    print(f"  saving   {(showdown_16k - engine_16k) / 3600:8.2f} h  "
          f"({ratio(engine_16k, showdown_16k)} faster)")
    print("  Single-threaded on one box; the production collector fans out across")
    print("  workers, so treat this as a per-worker ratio, not a fleet ETA.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--format", dest="format_id", default="gen3randombattle")
    parser.add_argument("--max-decision-rounds", type=int, default=250)
    parser.add_argument("--showdown-root", type=Path, default=None)
    parser.add_argument("--node-binary", default="node")
    parser.add_argument("--encoder-tables", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Budget-0 v2.2 checkpoint. Omit for a random-legal policy.")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Write a freshly-initialized budget-0 checkpoint here and exit.")
    parser.add_argument("--init-reference", type=Path, default=None,
                        help="Architecture source for --init-checkpoint.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.init_checkpoint is not None:
        write_initialized_checkpoint(args.init_checkpoint, args.init_reference)
        return 0

    masks = ObservationFeatureMasks(transition_token_budget=0)
    # ONE spec, named once, passed to BOTH arms. Left implicit, the two arms answered "nobody said"
    # from different places and this script silently compared two schemas -- which is the one thing
    # it exists not to do. `EngineEnvConfig` resolves through `engine_env._default_observation_spec()`
    # (a read of `DEFAULT_REPLAY_OBSERVATION_SPEC`, so it follows the process default), while
    # `LocalShowdownConfig.observation_spec` NAMES `CONFIG_DEFAULT_OBSERVATION_SPEC` and does not.
    # Those were the same value until the default rotated to v4, at which point the engine arm went
    # to v4 (132 numeric) and the Showdown arm stayed at v2.2 (155) -- a latent A/B corruption that
    # would have shown up as a backend difference rather than as an error.
    #
    # Taken from the SHOWDOWN side and inherited by the engine side, which is the invariant
    # `rollout_cli.py` already states in as many words: "the engine backend inherits exactly the
    # contract the Showdown backend would have written -- which is what makes the two backends'
    # shards schema-identical."
    #
    # Deliberately NOT `observation_spec_for_schema(OBSERVATION_SCHEMA_VERSION)`, which was my first
    # cut. That reads the global, and the census counted it: a new `bare-const` row in scripts/,
    # taking the ledger UP by one against a HIGH_WATER_MARK that only ever lowers. Fixing a
    # two-answers bug by adding a third read of the global is the wrong direction, and the gate said
    # so before this was committed.
    #
    # Consequence worth knowing: this benchmark measures the CONFIG default, so after a rotation it
    # keeps measuring v2.2 while `neural train` collects at the new default. That is a real gap, and
    # it belongs to the two-answers split rather than to this script -- closing it means the engine
    # and Showdown sides sharing one constant, not this line reaching for the global.
    observation_spec = CONFIG_DEFAULT_OBSERVATION_SPEC
    if args.checkpoint is not None:
        from pokezero.neural_policy import load_transformer_model_config

        config = load_transformer_model_config(args.checkpoint)
        if int(config.transition_token_budget) != 0:
            raise SystemExit(
                f"checkpoint {args.checkpoint} is stamped "
                f"transition_token_budget={config.transition_token_budget}, not 0; "
                "the k=0 arm needs a budget-0 checkpoint (see --init-checkpoint)."
            )
        args.policy_label = f"neural:{args.checkpoint.name} (dim={config.embedding_dim}, layers={config.transformer_layers})"
        policy_factory = lambda: _neural_policy(args.checkpoint)  # noqa: E731
    else:
        args.policy_label = "random-legal (no model forward)"
        policy_factory = _random_legal_policy

    print(f"engine arm: {args.games} games...")
    engine = run_engine_arm(args, policy_factory, masks, observation_spec)
    print(f"showdown arm: {args.games} games...")
    showdown = run_showdown_arm(args, policy_factory, masks, observation_spec)
    report(engine, showdown, args)

    if args.json_out is not None:
        payload = {
            arm.label: {
                "games": arm.games,
                "decisions": arm.decisions,
                "plies": arm.plies,
                "wall_s": arm.wall,
                "games_per_second": arm.games_per_second,
                "decisions_per_second": arm.decisions_per_second,
                "step_s": arm.step,
                "encode_s": arm.encode,
                "policy_s": arm.policy,
                "teams_s": arm.teams,
                "other_s": arm.other,
                "components": arm.extra,
            }
            for arm in (engine, showdown)
        }
        payload["config"] = {
            "games": args.games,
            "seed_start": args.seed_start,
            "format_id": args.format_id,
            "policy": args.policy_label,
            "transition_token_budget": 0,
        }
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
