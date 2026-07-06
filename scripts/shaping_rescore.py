"""Stage-0 shaping rescoring: score a shaping-weights config against frozen rollout records.

Given rollout-record JSONL file(s) (e.g. the frozen pools under runs/) and a shaping config,
compute every per-turn dense shaping term the config WOULD have produced at
collection time (the exact pokezero.shaping code path collection uses) and report:

  - per-turn |shaping| distribution (mean / p50 / p95 / max),
  - total shaped-sum vs terminal-return ratio per game (does dense signal swamp the outcome?),
  - per-category contribution shares (hp vs faint vs status vs hazard vs action classes),
  - a potential telescoping check
    (sum_k gamma^k f_k == gamma^K * Phi_T - Phi_0 per player-episode),
  - the fraction of turns with |shaping| above a threshold fraction of the terminal return.

This is the tool for killing bad weight configs WITHOUT training: a config whose per-game
shaped total rivals the terminal +-1, or whose per-turn terms are mostly noise-sized, is
wrong before any GPU is spent. JSON + human summary out.

Usage:
    python scripts/shaping_rescore.py \
      --records runs/pool-self-v2-20260705/records.jsonl \
      --shaping-weights wse-arm1 --discount 0.9999 --out evals/shaping_rescore.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pokezero.collection import iter_rollout_records
from pokezero.shaping import (
    ShapingConfig,
    action_class_components_by_step_index,
    action_class_names,
    component_names,
    ground_truth_components_by_step_index,
    parse_shaping_spec,
    potential_from_components,
    shaping_rewards_by_step_index,
    shaping_terms,
)

TELESCOPING_TOLERANCE = 1e-9


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def rescore_records(
    records_paths: list[Path],
    *,
    config: ShapingConfig,
    gamma: float,
    terminal_threshold_frac: float,
) -> dict:
    weights = config.component_weights()
    action_weights = config.action_class_weights()
    names = component_names()
    action_names = action_class_names()

    all_terms_abs: list[float] = []
    category_abs_totals: dict[str, float] = {name: 0.0 for name in (*names, *action_names)}
    games = 0
    episodes = 0
    threshold_hits = 0
    term_count = 0
    max_telescoping_error = 0.0
    shaped_total_over_terminal: list[float] = []
    shaped_magnitude_over_terminal: list[float] = []
    per_game_rows: list[dict] = []

    for records_path in records_paths:
        for record in iter_rollout_records(records_path):
            games += 1
            components_by_step = ground_truth_components_by_step_index(record)
            action_components_by_step = action_class_components_by_step_index(record)
            dense_rewards_by_step = shaping_rewards_by_step_index(record, config=config, gamma=gamma)
            step_indices_by_player: dict[str, list[int]] = {}
            for step_index, step in enumerate(record.trajectory.steps):
                step_indices_by_player.setdefault(step.player_id, []).append(step_index)

            terminal = record.terminal or record.trajectory.terminal
            game_row = {
                "battle_id": record.battle_id,
                "seed": record.seed,
                "winner": terminal.winner if terminal is not None else None,
                "capped": bool(terminal.capped) if terminal is not None else False,
                "players": {},
            }

            for player_id, step_indices in sorted(step_indices_by_player.items()):
                episodes += 1
                potentials = [
                    potential_from_components(components_by_step[index], config)
                    for index in step_indices
                ]
                terminal_potential = 0.0 if config.terminal_mode == "zero" else potentials[-1]
                potential_terms = shaping_terms(potentials, gamma=gamma, terminal_potential=terminal_potential)
                terms = [dense_rewards_by_step.get(index, 0.0) for index in step_indices]

                # Per-category terms: the potential is linear in its components, and
                # action classes are direct linear terms, so this split is exact.
                for name in names:
                    weight = weights[name]
                    if weight == 0.0:
                        continue
                    series = [weight * components_by_step[index][name] for index in step_indices]
                    terminal_component = 0.0 if config.terminal_mode == "zero" else series[-1]
                    for term in shaping_terms(series, gamma=gamma, terminal_potential=terminal_component):
                        category_abs_totals[name] += abs(term)
                for name in action_names:
                    weight = action_weights[name]
                    if weight == 0.0:
                        continue
                    for index in step_indices:
                        category_abs_totals[name] += abs(weight * action_components_by_step[index].get(name, 0.0))

                potential_discounted_total = sum((gamma**k) * term for k, term in enumerate(potential_terms))
                expected_total = (gamma ** len(potential_terms)) * terminal_potential - potentials[0]
                max_telescoping_error = max(max_telescoping_error, abs(potential_discounted_total - expected_total))
                discounted_total = sum((gamma**k) * term for k, term in enumerate(terms))

                terminal_return = 0.0
                if terminal is not None and not terminal.capped and terminal.winner is not None:
                    terminal_return = 1.0 if terminal.winner == player_id else -1.0
                reference = abs(terminal_return) if terminal_return != 0.0 else 1.0
                if terminal_return != 0.0:
                    # Net effect on the episode return. For terminal_mode='zero' this is
                    # -Phi(initial) ~ 0 by construction (PBRS neutrality); 'carry' keeps
                    # the accumulated potential and lands nonzero.
                    shaped_total_over_terminal.append(abs(discounted_total) / reference)
                    # Total dense-signal MAGNITUDE injected along the episode — the
                    # interference read that stays meaningful under 'zero'.
                    shaped_magnitude_over_terminal.append(
                        sum(abs(term) for term in terms) / reference
                    )

                for term in terms:
                    all_terms_abs.append(abs(term))
                    term_count += 1
                    if abs(term) > terminal_threshold_frac * reference:
                        threshold_hits += 1

                game_row["players"][player_id] = {
                    "decisions": len(terms),
                    "phi_initial": potentials[0],
                    "phi_final": potentials[-1],
                    "potential_discounted_total": potential_discounted_total,
                    "shaped_discounted_total": discounted_total,
                    "terminal_return": terminal_return,
                }
            per_game_rows.append(game_row)

    if games == 0:
        raise ValueError("no rollout records found in the given paths.")

    sorted_abs = sorted(all_terms_abs)
    category_total = sum(category_abs_totals.values())
    sorted_ratios = sorted(shaped_total_over_terminal)
    sorted_magnitudes = sorted(shaped_magnitude_over_terminal)
    return {
        "shaping_config": config.to_dict(),
        "gamma": gamma,
        "records": [str(path) for path in records_paths],
        "games": games,
        "player_episodes": episodes,
        "turns_scored": term_count,
        "per_turn_abs_shaping": {
            "mean": sum(sorted_abs) / len(sorted_abs) if sorted_abs else None,
            "p50": _percentile(sorted_abs, 0.50),
            "p95": _percentile(sorted_abs, 0.95),
            "max": sorted_abs[-1] if sorted_abs else None,
        },
        "shaped_total_vs_terminal": {
            "mean": (sum(sorted_ratios) / len(sorted_ratios)) if sorted_ratios else None,
            "p50": _percentile(sorted_ratios, 0.50),
            "p95": _percentile(sorted_ratios, 0.95),
            "max": sorted_ratios[-1] if sorted_ratios else None,
            "decided_episodes": len(sorted_ratios),
        },
        "shaped_magnitude_vs_terminal": {
            "mean": (sum(sorted_magnitudes) / len(sorted_magnitudes)) if sorted_magnitudes else None,
            "p50": _percentile(sorted_magnitudes, 0.50),
            "p95": _percentile(sorted_magnitudes, 0.95),
            "max": sorted_magnitudes[-1] if sorted_magnitudes else None,
        },
        "category_abs_contribution_share": {
            name: (category_abs_totals[name] / category_total if category_total > 0 else 0.0)
            for name in (*names, *action_names)
        },
        "telescoping": {
            "max_abs_error": max_telescoping_error,
            "tolerance": TELESCOPING_TOLERANCE,
            "passed": max_telescoping_error <= TELESCOPING_TOLERANCE,
        },
        "terminal_threshold_frac": terminal_threshold_frac,
        "fraction_turns_above_threshold": (threshold_hits / term_count) if term_count else None,
        "games_detail": per_game_rows,
    }


def print_summary(report: dict) -> None:
    print(f"games: {report['games']}  player_episodes: {report['player_episodes']}  turns: {report['turns_scored']}")
    per_turn = report["per_turn_abs_shaping"]
    print(
        "per-turn |shaping|: "
        f"mean={_fmt(per_turn['mean'])} p50={_fmt(per_turn['p50'])} "
        f"p95={_fmt(per_turn['p95'])} max={_fmt(per_turn['max'])}"
    )
    ratio = report["shaped_total_vs_terminal"]
    print(
        "|shaped total| / |terminal| (net effect; potential-only configs telescope under terminal_mode=zero): "
        f"mean={_fmt(ratio['mean'])} p50={_fmt(ratio['p50'])} p95={_fmt(ratio['p95'])} "
        f"max={_fmt(ratio['max'])} (over {ratio['decided_episodes']} decided episodes)"
    )
    magnitude = report["shaped_magnitude_vs_terminal"]
    print(
        "sum|shaping| / |terminal| (dense-signal magnitude): "
        f"mean={_fmt(magnitude['mean'])} p50={_fmt(magnitude['p50'])} "
        f"p95={_fmt(magnitude['p95'])} max={_fmt(magnitude['max'])}"
    )
    shares = report["category_abs_contribution_share"]
    nonzero = {name: share for name, share in shares.items() if share > 0}
    print("category |contribution| shares: " + (
        "  ".join(f"{name}={share:.3f}" for name, share in sorted(nonzero.items(), key=lambda kv: -kv[1]))
        if nonzero
        else "(all zero)"
    ))
    telescoping = report["telescoping"]
    print(
        f"telescoping: max_abs_error={telescoping['max_abs_error']:.3e} "
        f"({'PASS' if telescoping['passed'] else 'FAIL'} at {telescoping['tolerance']:.0e})"
    )
    print(
        f"turns with |shaping| > {report['terminal_threshold_frac']:.0%} of terminal: "
        f"{_fmt(report['fraction_turns_above_threshold'])}"
    )


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", action="append", required=True, type=Path, help="Rollout-record JSONL. May repeat.")
    parser.add_argument(
        "--shaping-weights",
        required=True,
        help="Shaping config: preset (wse-arm1), inline JSON, or @/path/to.json.",
    )
    parser.add_argument("--discount", type=float, default=1.0, help="Shaping gamma (use the training discount).")
    parser.add_argument(
        "--terminal-threshold-frac",
        type=float,
        default=0.01,
        help="Report the fraction of turns whose |shaping| exceeds this fraction of the terminal return.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    config = parse_shaping_spec(args.shaping_weights)
    if config is None:
        raise SystemExit("--shaping-weights 'none' scores nothing; give a real config.")
    report = rescore_records(
        list(args.records),
        config=config,
        gamma=args.discount,
        terminal_threshold_frac=args.terminal_threshold_frac,
    )
    print_summary(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
