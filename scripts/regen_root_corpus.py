"""Regenerate search-over-raw source games as a golden corpus, for the root-parity gate.

The exploration collector plays each source game with the champion checkpoint as a
deterministic masked argmax on both seats from a seeded local Showdown battle. That is
reproducible from the seed alone. This script replays the given seeds through the same
production env + policy and writes the decisions as a golden corpus (public materialization
payload, true teams, observation arrays, fold rows), so ``scripts/leaf_root_parity.py`` can
byte-diff the native depth-0 root encoding against the Python observation by column name.

Also matches each archived exploration root (``run/sources/<seed>/source.json``) to its
regenerated row by observation-array equality and reports the match count, which is the
acceptance test for the regeneration.

Usage:
    PYTHONPATH=src python scripts/regen_root_corpus.py --checkpoint <transformer-policy.pt> \
        --showdown-root <built-showdown> --run-dir <exploration run dir> --seeds 1505790335 ... \
        --out <corpus dir> [--max-decision-rounds 250]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero.actions import ACTION_COUNT  # noqa: E402
from pokezero.collection import env_config_with_policy_spec_masks  # noqa: E402
from pokezero.golden_corpus import (  # noqa: E402
    GoldenGame,
    GoldenGameRecord,
    _CapturingPolicy,
    _decision_row_from_context,
    _true_teams_from_bridge_snapshot,
    write_golden_corpus,
)
from pokezero.golden_corpus_fold import FoldSurfaceRecorder, build_fold_rows  # noqa: E402
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv  # noqa: E402
from pokezero.neural_policy import load_transformer_policy  # noqa: E402
from pokezero.observation import TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS  # noqa: E402
from pokezero.rollout import RolloutConfig, continue_rollout_from_current_state  # noqa: E402


def archived_roots(run_dir: Path, seed: int) -> list[dict[str, Any]]:
    path = run_dir / "sources" / str(seed) / "source.json"
    if not path.exists():
        return []
    return list(json.load(open(path))["roots"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None, help="exploration run dir with sources/<seed>/source.json")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--format-id", default="gen3randombattle")
    parser.add_argument("--max-decision-rounds", type=int, default=250)
    parser.add_argument("--namespace", default="regen", help="battle id prefix")
    args = parser.parse_args(argv)

    base = LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True)
    config = env_config_with_policy_spec_masks(base, ["neural:" + str(args.checkpoint)], context="regen_root_corpus")
    env = LocalShowdownEnv(config)
    turn_merged_active = config.observation_spec.schema_version in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS
    collected: list[GoldenGame] = []
    matches: dict[int, dict[str, Any]] = {}
    try:
        belief_hash = env.belief_set_source_hash
        for seed in args.seeds:
            battle_id = f"{args.namespace}:{args.format_id}:{seed}"
            env.reset(seed=seed, format_id=args.format_id)
            true_teams = _true_teams_from_bridge_snapshot(env.snapshot().bridge_snapshot)
            captures = []
            recorder = FoldSurfaceRecorder(env)

            def _sink(context, decision, captures=captures, recorder=recorder):
                captures.append((context, decision))
                recorder.record(context.player_id)

            policies = {
                seat: _CapturingPolicy(
                    load_transformer_policy(args.checkpoint, device="cpu", deterministic=True, exploration_epsilon=0.0,
                                            sampling_temperature=1.0, family_gated_selection=False),
                    _sink,
                )
                for seat in ("p1", "p2")
            }
            result = continue_rollout_from_current_state(
                env=env, policies=policies,
                config=RolloutConfig(max_decision_rounds=args.max_decision_rounds, format_id=args.format_id,
                                     hide_opponent_legal_action_masks=True),
                seed=seed, battle_id=battle_id, reset_policies=True,
            )
            rows = tuple(_decision_row_from_context(context, decision, battle_seed=seed) for context, decision in captures)
            fold_rows = build_fold_rows(
                replays=[context.public_materialization_state.replay for context, _ in captures],
                surfaces=recorder.surfaces, turn_merged_active=turn_merged_active,
            )
            record = GoldenGameRecord(
                battle_seed=seed, battle_id=battle_id, format_id=args.format_id,
                policy_ids={"p1": policies["p1"].policy_id, "p2": policies["p2"].policy_id},
                true_teams=true_teams,
                terminal={"winner": result.terminal.winner, "turn_count": result.terminal.turn_count, "capped": result.terminal.capped},
            )
            collected.append(GoldenGame(record=record, rows=rows, fold_rows=fold_rows))
            # Match archived roots by observation arrays.
            if args.run_dir is not None:
                for root in archived_roots(args.run_dir, seed):
                    want = numpy.asarray(root["public_record"]["observation"]["numeric_features"], dtype=numpy.float32)
                    hit = None
                    for row in rows:
                        if row.player_id != root["public_record"]["acting_player"]:
                            continue
                        got = numpy.asarray(row.arrays.numeric_features, dtype=numpy.float32)
                        if got.shape == want.shape and numpy.array_equal(got, want):
                            hit = row; break
                    matches[root["root_id"]] = dict(
                        matched=hit is not None, decision_round_index=(hit.decision_round_index if hit else None),
                        seat=root["public_record"]["acting_player"], source_request_index=root["source_request_index"],
                        chosen_action=(hit.chosen_action_index if hit else None),
                        recorded_action=root["public_record"].get("recorded_action_index"),
                        turn_index=root["public_record"].get("turn_index"))
            seed_roots = [m for r, m in matches.items() if r.split(":")[1] == str(seed)]
            print(json.dumps(dict(seed=seed, decisions=len(rows), turns=result.terminal.turn_count, winner=result.terminal.winner,
                                  archived_roots=len(seed_roots), roots_matched=sum(1 for m in seed_roots if m["matched"]))), flush=True)
    finally:
        env.close()
    spec = config.observation_spec
    masks = config.feature_masks
    header = {
        "generator": {"script": "scripts/regen_root_corpus.py", "seeds": list(args.seeds), "format_id": args.format_id,
                      "max_decision_rounds": args.max_decision_rounds, "policy": "neural:" + str(args.checkpoint),
                      "hide_opponent_legal_action_masks": True},
        "observation": {"schema_version": spec.schema_version, "token_count": spec.token_count,
                        "categorical_feature_count": spec.categorical_feature_count,
                        "numeric_feature_count": spec.numeric_feature_count, "action_count": ACTION_COUNT,
                        "feature_masks": {"stats_block": masks.opponent_tendency_stats_block, "exact_state": masks.exact_state,
                                          "transition_token_budget": min(masks.transition_token_budget, spec.transition_token_count),
                                          "tier2_residuals": masks.tier2_residuals, "tier2_investment": masks.tier2_investment}},
        "belief_set_source": {"enabled": config.belief_set_source_enabled(), "source_hash": belief_hash},
    }
    summary = write_golden_corpus(args.out, header=header, games=collected)
    (args.out / "archived_root_matches.json").write_text(json.dumps(matches, indent=2, sort_keys=True) + "\n")
    total = len(matches); hit = sum(1 for m in matches.values() if m["matched"])
    print(json.dumps(dict(corpus=str(args.out), games=len(collected), rows=sum(len(g.rows) for g in collected),
                          archived_roots=total, archived_roots_matched=hit)))
    return 0 if (total == 0 or hit == total) else 1


if __name__ == "__main__":
    raise SystemExit(main())
