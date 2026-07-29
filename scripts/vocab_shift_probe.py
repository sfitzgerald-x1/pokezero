"""Quantify what a category-vocabulary shift does to a checkpoint's play.

Paired by construction: ONE checkpoint, ONE trajectory, TWO encodings — the checkpoint's own
stamped enumeration versus whatever this build enumerates today. Weights and states are
identical on both sides, so there is no game-sampling variance to bound and no seed budget to
argue about; every difference is the enumeration and nothing else.

Reports, per decision boundary:
  * whether the two encodings differ AT ALL (how often the shift is even reachable), and
  * whether the argmax action differs (the decision-level effect).

The game is driven by the CHECKPOINT-anchored side, so the build-anchored decisions are
counterfactuals measured along the correct trajectory.

Used for docs/encoder_vocab_provenance_20260729.md, where a 1216-token checkpoint on a
1217-token build gave 19.5 % of encodings differing and 3.7 % of actions changing — refuting
the assumption that a single inserted token is negligible. Sorted insertion renumbers the
whole alphabetical tail after it, which here included all four `weather:` tokens.

Usage::

    PYTHONPATH=src python scripts/vocab_shift_probe.py \
        --checkpoint runs/.../model.pt --showdown-root /path/to/pokemon-showdown --games 60
"""
from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--showdown-root", required=True)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=9_000_000)
    args = ap.parse_args(argv)

    from pokezero.dex import load_showdown_dex_cached
    from pokezero.local_showdown import (
        LocalShowdownConfig,
        LocalShowdownEnv,
        env_config_with_checkpoint_masks,
    )
    from pokezero.neural_policy import (
        category_vocab_from_model_config,
        feature_masks_from_model_config,
        load_transformer_model_config,
        load_transformer_policy,
        observation_spec_from_model_config,
    )
    from pokezero.randbat_vocab import gen3_category_vocabulary
    from pokezero.showdown import observation_from_player_state

    config = load_transformer_model_config(args.checkpoint)
    spec = observation_spec_from_model_config(config)
    masks = feature_masks_from_model_config(config)
    turn_merged = spec.schema_version in (
        "pokezero.observation.v2.2",
        "pokezero.observation.v3",
    )

    vocab_ckpt = category_vocab_from_model_config(config, args.showdown_root)
    vocab_build = gen3_category_vocabulary(args.showdown_root, include_turn_merged=turn_merged)
    print(f"checkpoint enumeration: {len(vocab_ckpt.tokens)} tokens")
    print(f"build enumeration     : {len(vocab_build.tokens)} tokens")
    shifted = [
        (i, t)
        for i, t in enumerate(vocab_build.tokens)
        if i < len(vocab_ckpt.tokens) and vocab_ckpt.tokens[i] != t
    ]
    print(f"first divergence index: {shifted[0][0] if shifted else None}")
    print(f"renumbered EXISTING tokens: {len(shifted)}  (plus 1 newly inserted = 13-token tail)")
    print(f"  e.g. {shifted[0][0]}: checkpoint={vocab_ckpt.tokens[shifted[0][0]]!r} build={shifted[0][1]!r}")
    print()

    env_config = env_config_with_checkpoint_masks(
        LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True),
        masks,
        context="vocab probe",
        required_specs=spec,
        required_vocabs=vocab_ckpt,
    )
    assert env_config.category_vocab == vocab_ckpt, "latch did not adopt the checkpoint vocab"

    policy = load_transformer_policy(args.checkpoint, deterministic=True)
    dex = load_showdown_dex_cached(args.showdown_root)
    env = LocalShowdownEnv(env_config)

    decisions = 0
    encodings_differ = 0
    argmax_differ = 0
    games_played = 0

    for game in range(args.games):
        seed = args.seed_start + game
        try:
            env.reset(seed=seed)
        except Exception as exc:  # noqa: BLE001
            print(f"  seed {seed}: reset failed ({type(exc).__name__}); skipping", flush=True)
            continue
        games_played += 1
        steps = 0
        while env.terminal() is None and steps < 400:
            steps += 1
            players = env.requested_players()
            if not players:
                break
            actions = {}
            for player in players:
                state = env._state_for_player(player)
                obs_a = observation_from_player_state(
                    state, category_vocab=vocab_ckpt, spec=spec, dex=dex, feature_masks=masks
                )
                obs_b = observation_from_player_state(
                    state, category_vocab=vocab_build, spec=spec, dex=dex, feature_masks=masks
                )
                decisions += 1
                if list(obs_a.categorical_ids) != list(obs_b.categorical_ids):
                    encodings_differ += 1
                # Deterministic policy, but rng is required; use a fresh identical
                # stream for each side so the pairing cannot leak through it.
                a = policy.select_action(obs_a, rng=random.Random(0)).action_index
                b = policy.select_action(obs_b, rng=random.Random(0)).action_index
                if a != b:
                    argmax_differ += 1
                actions[player] = a
            try:
                env.step(actions)
            except Exception as exc:  # noqa: BLE001
                print(f"  seed {seed}: step failed ({type(exc).__name__}); ending game", flush=True)
                break
        if (game + 1) % 10 == 0:
            print(
                f"  [{game + 1}/{args.games}] decisions={decisions} "
                f"encodings_differ={encodings_differ} argmax_differ={argmax_differ}",
                flush=True,
            )
    env.close()

    print()
    print("=" * 66)
    print(f"games played              : {games_played}")
    print(f"decision boundaries       : {decisions}")
    pct = (100.0 * encodings_differ / decisions) if decisions else 0.0
    print(f"encodings differed        : {encodings_differ}  ({pct:.2f}% of decisions)")
    pct2 = (100.0 * argmax_differ / decisions) if decisions else 0.0
    print(f"argmax action differed    : {argmax_differ}  ({pct2:.2f}% of decisions)")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
