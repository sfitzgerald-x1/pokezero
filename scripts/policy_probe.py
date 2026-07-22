"""Engineered-scenario policy probe — measure how a checkpoint's action distribution
responds to a single controlled feature, and track that response across checkpoints.

The idea: hold a real mid-game battle state fixed, vary ONE thing (here: the badly-poisoned
"toxic" counter on our own active mon, swept from healthy -> deepening ramp), and read how
the policy's probability of SWITCHING OUT moves. Run it on matched current-family v2+
milestones to watch a tactic emerge: a model that has learned toxic management should
shift probability toward switching as the ramp deepens; one that hasn't will be flat.

This is a *behavioral* probe, not a win-rate eval. Absolute numbers matter less than
(a) the slope across toxic depth within a checkpoint and (b) how that slope changes
between checkpoints over training.

Caveats (documented so the numbers aren't over-read):
  - Single-frame observation (window of 1): we probe one engineered state, not a full
    history window. Consistent across checkpoints, but out-of-distribution vs live play
    where the model carries history. Use for *relative* comparison.
  - HP is held fixed while the toxic counter rises, deliberately, to isolate the toxic
    signal from the confound of "low HP -> switch". A real toxic'd mon also loses HP;
    this probe answers "does the counter alone move the policy".

Usage:
  python scripts/policy_probe.py \
      --checkpoint checkpoints/curated/current-v2-500k.pt=v2-500k \
      --checkpoint checkpoints/curated/current-v2-600k.pt=v2-600k \
      --showdown-root /Users/scott/workspace/pokerena/vendor/pokemon-showdown \
      [--capture-seed 7] [--out runs/probes/toxic-<date>.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

from pokezero.actions import ACTION_COUNT, MOVE_ACTION_COUNT
from pokezero.local_showdown import (
    LocalShowdownConfig,
    LocalShowdownEnv,
    env_config_with_checkpoint_masks,
)
from pokezero.neural_policy import evaluate_transformer_action_priors
from pokezero.online_client import build_agent
from pokezero.opponents import require_current_family_checkpoint_paths
from pokezero.showdown import (
    CATEGORY_SECONDARY,
    NUMERIC_TOXIC_STAGE,
    SELF_POKEMON_TOKEN_OFFSET,
    numeric_index_for_schema,
    observation_from_player_state,
)

# Toxic ramp depths to probe. 0 = "just poisoned, first tick pending"; 15 = the encoder's
# saturation point (the /15 normalization). Healthy baseline is probed separately.
TOXIC_STAGES = (0, 1, 3, 6, 9, 12, 15)

# Stally, poison-SUSCEPTIBLE walls where toxic management is genuinely a decision: a toxic'd
# wall loses the longevity that is its whole job, and switching out resets the gen-3 toxic
# counter, so a strong policy should lean toward switching as the ramp deepens. Crucially these
# are NOT Poison/Steel (which are immune to toxic — engineering toxic onto them is an impossible,
# out-of-distribution state). Priority-ordered; capture takes the first that appears active.
DEFAULT_STALLERS = (
    "vaporeon", "blissey", "suicune", "milotic", "snorlax",
    "slowbro", "umbreon", "regice", "lapras", "hypno",
)


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _toxic_immune(dex, species: str) -> bool:
    info = dex.species_info(species)
    types = info.types if info else ()
    return any(str(t).lower() in ("poison", "steel") for t in types)


def _hp_fraction(condition: str | None) -> float:
    parts = str(condition or "").split()
    if parts and "/" in parts[0]:
        num, _, den = parts[0].partition("/")
        try:
            return max(0.0, min(1.0, float(num) / float(den)))
        except (ValueError, ZeroDivisionError):
            return 0.0
    return 0.0


def _hp_token(condition: str | None) -> str:
    """The bare HP token ('180/281') from a condition string, dropping any status word."""
    parts = str(condition or "100/100").split()
    return parts[0] if parts else "100/100"


def _numeric_feature(observation, token: int, legacy_index: int) -> float:
    physical_index = numeric_index_for_schema(observation.schema_version, legacy_index)
    return float(observation.numeric_features[token][physical_index])


def capture_base_state(
    showdown_root: str,
    driver_checkpoint: str,
    seed: int,
    targets,
    *,
    max_seeds: int = 80,
    min_turn: int = 5,
):
    """Scan real self-play games until one of the target staller species is OUR active mon at a
    clean decision: not fainted, healthy (HP > 50%), at least one legal switch, and (asserted)
    poison-susceptible. The board, stats, and moves are whatever the real game produced — only
    the toxic status is engineered downstream. Prefer a mid-game state (turn >= min_turn). Either
    seat may surface the staller. Returns (state, protocol_lines, active, found_seed, found_player)."""
    agent = build_agent(driver_checkpoint, showdown_root, our_name="probe", deterministic=True)
    policy, dex = agent.policy, agent.dex
    target_set = {_norm(t) for t in targets}

    # The driver checkpoint reads env.observe() tensors, so the env must encode with the
    # masks AND the observation schema/width it trained under (the same latch the shared
    # harnesses apply; build_agent already resolved agent.spec from the checkpoint).
    env_config = LocalShowdownConfig(showdown_root=showdown_root)
    if agent.feature_masks is not None:
        env_config = env_config_with_checkpoint_masks(
            env_config,
            agent.feature_masks,
            context="policy_probe capture driver",
            required_specs=agent.spec,
        )

    for game_seed in range(seed, seed + max_seeds):
        env = LocalShowdownEnv(env_config)
        env.reset(seed=game_seed)
        rng = random.Random(game_seed)
        best = None  # last qualifying capture this game, used if we never reach min_turn

        for _ in range(400):
            if env.terminal() is not None:
                break
            requested = env.requested_players()
            for player in ("p1", "p2"):
                if player not in requested:
                    continue
                state = env._state_for_player(player)
                active = state.self_active
                if active is None or _norm(active.species) not in target_set:
                    continue
                if _toxic_immune(dex, active.species):
                    continue  # never engineer toxic onto an immune mon
                mask = state.legal_action_mask
                if (
                    "fnt" not in str(active.condition or "")
                    and _hp_fraction(active.condition) > 0.5
                    and any(mask[MOVE_ACTION_COUNT:ACTION_COUNT])
                    and state.request_kind not in {"wait", "none"}
                ):
                    best = (state, tuple(env.protocol_lines), active, game_seed, player)
                    if state.turn_number >= min_turn:
                        return best
            actions = {}
            for player in requested:
                obs = env.observe(player)
                if not any(obs.legal_action_mask):
                    continue
                actions[player] = policy.select_action(obs, rng=rng).action_index
            if not actions:
                break
            env.step(actions)
        if best is not None:
            return best
    raise RuntimeError(
        f"no target staller ({', '.join(sorted(target_set))}) reached a clean decision in "
        f"{max_seeds} games from seed {seed}; widen --active-species or raise --max-seeds"
    )


def engineer_toxic(state, stage: int | None):
    """Return a copy of `state` with the active mon badly poisoned at the given ramp `stage`
    (status:tox categorical + the NUMERIC_TOXIC_STAGE ramp + belief status), HP held fixed.
    `stage=None` returns the healthy baseline (status cleared, ramp 0)."""
    active = state.self_active
    hp = _hp_token(active.condition)
    status_word = "tox" if stage is not None else ""
    new_condition = f"{hp} {status_word}".strip()

    new_team = tuple(
        dataclasses.replace(p, condition=new_condition) if p is active else p
        for p in state.self_team
    )
    belief = state.belief_view
    new_self_belief = tuple(
        dataclasses.replace(
            b,
            status=("tox" if stage is not None else None),
            condition=new_condition,
        )
        if b.active
        else b
        for b in belief.self_pokemon
    )
    new_belief = dataclasses.replace(belief, self_pokemon=new_self_belief)
    return dataclasses.replace(
        state,
        self_team=new_team,
        belief_view=new_belief,
        self_toxic_stage=(stage or 0),
    )


def _active_token_index(state) -> int:
    for offset, pokemon in enumerate(state.self_team):
        if pokemon.active:
            return SELF_POKEMON_TOKEN_OFFSET + offset
    raise RuntimeError("no active mon in self_team")


def probe_checkpoint(label: str, checkpoint: str, showdown_root: str, base_state):
    """For one checkpoint, sweep healthy + toxic depths and return per-scenario action stats.
    Verifies the engineered toxic features actually landed in the observation tensor."""
    agent = build_agent(checkpoint, showdown_root, our_name="probe", deterministic=True)
    tox_vocab_id = agent.vocab.encode("status:tox")
    none_vocab_id = agent.vocab.encode("status:none")

    def evaluate(stage):
        state = engineer_toxic(base_state, stage)
        obs = observation_from_player_state(
            state, category_vocab=agent.vocab, spec=agent.spec, dex=agent.dex,
            **({"feature_masks": agent.feature_masks} if agent.feature_masks is not None else {}),
        )
        token = _active_token_index(state)
        seen_status = obs.categorical_ids[token][CATEGORY_SECONDARY]
        seen_ramp = _numeric_feature(obs, token, NUMERIC_TOXIC_STAGE)
        # Verify the engineering landed: at a toxic stage the active token must read status:tox
        # and the ramp must equal stage/15; healthy must read status:none and ramp 0.
        if stage is None:
            assert seen_status == none_vocab_id, f"healthy: status not 'none' (got id {seen_status})"
            assert abs(seen_ramp) < 1e-6, f"healthy: ramp not 0 (got {seen_ramp})"
        else:
            assert seen_status == tox_vocab_id, f"stage {stage}: status:tox not set (got id {seen_status})"
            assert abs(seen_ramp - stage / 15.0) < 1e-6, f"stage {stage}: ramp {seen_ramp} != {stage/15.0}"

        probs = evaluate_transformer_action_priors(
            model=agent.policy.model, result=agent.policy.result, observations=[obs]
        )
        p_switch = sum(probs[MOVE_ACTION_COUNT:ACTION_COUNT])
        p_move = sum(probs[:MOVE_ACTION_COUNT])
        argmax = max(range(ACTION_COUNT), key=lambda i: probs[i])
        return {
            "p_switch": round(p_switch, 4),
            "p_move": round(p_move, 4),
            "argmax_is_switch": argmax >= MOVE_ACTION_COUNT,
            "argmax_action": argmax,
        }

    results = {"healthy": evaluate(None)}
    for stage in TOXIC_STAGES:
        results[f"tox_{stage}"] = evaluate(stage)

    # Temporal probe: a real toxic tell is the counter CLIMBING across turns, which a single
    # static frame cannot express. Build a full history window and compare three trajectories on
    # the final decision: all-healthy, the counter escalating 1..window, and a deep constant.
    window = agent.policy.result.model_config.window_size

    def evaluate_history(stages):
        obs_window = []
        for stage in stages:
            state = engineer_toxic(base_state, stage)
            obs_window.append(
                observation_from_player_state(
                    state, category_vocab=agent.vocab, spec=agent.spec, dex=agent.dex,
                    **({"feature_masks": agent.feature_masks} if agent.feature_masks is not None else {}),
                )
            )
        probs = evaluate_transformer_action_priors(
            model=agent.policy.model, result=agent.policy.result, observations=obs_window
        )
        return round(sum(probs[MOVE_ACTION_COUNT:ACTION_COUNT]), 4)

    escalating = [min(15, i) for i in range(1, window + 1)]
    temporal = {
        "healthy_history": evaluate_history([None] * window),
        "escalating_history": evaluate_history(escalating),
        "deep_constant_history": evaluate_history([15] * window),
    }
    return {
        "label": label,
        "checkpoint": checkpoint,
        "window_size": window,
        "scenarios": results,
        "temporal": temporal,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="PATH[=LABEL]",
        help="checkpoint to probe; repeatable. Optional =LABEL for display.",
    )
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--capture-seed", type=int, default=7, help="first game seed to scan")
    parser.add_argument(
        "--active-species",
        default=",".join(DEFAULT_STALLERS),
        help="comma-separated staller species to capture as our active mon (priority order)",
    )
    parser.add_argument("--max-seeds", type=int, default=80, help="games to scan for a target staller")
    parser.add_argument("--out", default=None, help="write the full result JSON here")
    parser.add_argument(
        "--allow-legacy-checkpoints",
        action="store_true",
        help=(
            "allow no-belief/pre-v2 checkpoints for explicit historical reproduction; "
            "do not use for current longitudinal evals"
        ),
    )
    args = parser.parse_args()

    specs = []
    for raw in args.checkpoint:
        path, _, label = raw.partition("=")
        specs.append((label or Path(path).stem, path))
    if not args.allow_legacy_checkpoints:
        require_current_family_checkpoint_paths(
            (path for _, path in specs),
            context="policy probe",
        )

    targets = tuple(s.strip() for s in args.active_species.split(",") if s.strip())
    driver = specs[0][1]
    print(f"[capture] scanning real self-play games (from seed {args.capture_seed}) for a staller: {', '.join(targets)}…")
    base_state, protocol_lines, active, found_seed, found_player = capture_base_state(
        args.showdown_root, driver, args.capture_seed, targets, max_seeds=args.max_seeds
    )
    print(
        f"[capture] base state: {active.species} ({_hp_token(active.condition)} HP) active as "
        f"{found_player} in game seed {found_seed}, turn {base_state.turn_number}, "
        f"{sum(base_state.legal_action_mask[MOVE_ACTION_COUNT:ACTION_COUNT])} legal switches "
        f"— poisonable ✓"
    )

    probes = [probe_checkpoint(label, path, args.showdown_root, base_state) for label, path in specs]

    scenarios = ["healthy"] + [f"tox_{s}" for s in TOXIC_STAGES]
    for probe in probes:
        print(f"\n=== {probe['label']} ===")
        print(f"  {'scenario':<12}{'P(switch)':>11}{'P(move)':>10}  argmax")
        for name in scenarios:
            row = probe["scenarios"][name]
            tag = "SWITCH" if row["argmax_is_switch"] else "move"
            print(f"  {name:<12}{row['p_switch']:>11.3f}{row['p_move']:>10.3f}  {tag} (a{row['argmax_action']})")
        healthy = probe["scenarios"]["healthy"]["p_switch"]
        deep = probe["scenarios"][f"tox_{TOXIC_STAGES[-1]}"]["p_switch"]
        print(f"  -> single-frame ΔP(switch) healthy→tox_{TOXIC_STAGES[-1]}: {deep - healthy:+.3f}")
        t = probe["temporal"]
        print(
            f"  -> temporal (window {probe['window_size']}) P(switch): "
            f"healthy {t['healthy_history']:.3f} | escalating {t['escalating_history']:.3f} "
            f"| deep {t['deep_constant_history']:.3f}  "
            f"(Δ esc−healthy {t['escalating_history'] - t['healthy_history']:+.3f})"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scenario": "toxic_counter_escalation",
            "capture_seed_start": args.capture_seed,
            "found_seed": found_seed,
            "found_player": found_player,
            "base_active_species": active.species,
            "base_turn": base_state.turn_number,
            "toxic_stages": list(TOXIC_STAGES),
            "probes": probes,
            "protocol_lines": list(protocol_lines),
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\n[out] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
