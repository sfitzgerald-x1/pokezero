#!/usr/bin/env python3
"""Defender-side investment inference gate harness (v2.1 batch 2).

Sibling of ``scripts/tier2_gate.py`` (PR D's pattern), inverted to the defender:
generates omniscient controlled games on the local Showdown BattleStream (both seats
seeded uniform-random-legal with a move bias), then

1. runs ``pokezero.investment.infer_investment`` from each perspective with the real
   randbats candidate-set source, and scores every per-opponent-mon conclusion against
   ground truth read from the opposing seat's own opening request (true max HP / Def /
   SpD / EV spread — omniscient truth, no oracle leakage into the inference):
   PRECISION per conclusion type is the gate metric (false pins are the poison);
   recall/coverage are informational;
2. runs a known-set control pass (defender candidates pinned to the true set) and
   checks LATTICE CALIBRATION: on clean assessed strikes the observed damage must be
   lattice-consistent with the true variant (the inverted-side analog of the Tier-2
   roll-match arm — any inconsistency is a conditioning bug and a false-pin source);
3. verifies the fraction-exactness premise directly: every ``|-damage|`` /
   ``|switch|`` condition denominator in the omniscient log must equal the mon's true
   max HP;
4. cross-checks ``randbats_spread_details`` against server-computed stats AND the
   spread's own max-HP replication (the family universe must be engine-exact);
5. writes the JSON summary artifact quoted in the PR, with the gate verdict and the
   mask recommendation.

Usage:
    uv run python scripts/investment_gate.py --games 120 --seed 11 \
        --showdown-root ~/workspace/pokerena/vendor/pokemon-showdown \
        --out runs/investment-gate-2026-07-04
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pokezero.dex import load_showdown_dex_cached  # noqa: E402
from pokezero.gen3_damage import randbats_spread_details  # noqa: E402
from pokezero.investment import InvestmentConfig, infer_investment  # noqa: E402
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv  # noqa: E402
from pokezero.randbat import canonical_gen3_randbat_species_id, load_gen3_randbat_source_cached  # noqa: E402
from pokezero.showdown import _species_from_ident, _slot_from_ident, parse_showdown_replay  # noqa: E402
from pokezero.tier2 import own_team_from_request, variant_has_physical_attack  # noqa: E402
from tier2_gate import TrueVariantSource, _first_requests, _play_game, _team_truth  # noqa: E402


def _denominator_mismatches(lines: tuple[str, ...], truth: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Check the omniscient-channel premise: HP denominators are true max HP."""
    mismatches: list[dict[str, Any]] = []
    for line in lines:
        parts = line.split("|")
        event = parts[1] if len(parts) > 1 else ""
        if event in {"-damage", "-heal", "-sethp"} and len(parts) >= 4:
            ident, condition = parts[2], parts[3]
        elif event in {"switch", "drag"} and len(parts) >= 5:
            ident, condition = parts[2], parts[4]
        else:
            continue
        side = _slot_from_ident(ident)
        if side not in {"p1", "p2"}:
            continue
        head = condition.split()[0] if condition else ""
        if "/" not in head:
            continue
        try:
            denominator = int(head.split("/", 1)[1])
        except ValueError:
            continue
        species_key = canonical_gen3_randbat_species_id(_species_from_ident(ident))
        row = truth.get(side, {}).get(species_key)
        if row is None or row.get("maxhp") is None:
            continue
        if denominator != row["maxhp"]:
            mismatches.append(
                {"line": line, "expected_maxhp": row["maxhp"], "denominator": denominator}
            )
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--move-bias", type=float, default=0.75)
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.showdown_root.expanduser().resolve()
    dex = load_showdown_dex_cached(root)
    source = load_gen3_randbat_source_cached(root)
    rng = random.Random(args.seed)
    config = InvestmentConfig()

    # Precision ledgers per conclusion type.
    hp_value_tp = hp_value_fp = 0
    hp_class_tp = hp_class_fp = 0
    defense_tp = defense_fp = 0
    fp_detail: list[dict[str, Any]] = []
    tp_sample: list[dict[str, Any]] = []
    hp_blocked_mons = 0
    defense_blocked_mons = 0
    # Review LOW-1: a conclusion whose canonical key misses the truth mapping must be
    # COUNTED and asserted zero, never silently dropped — a canonicalization divergence
    # would otherwise hide exactly the false pins this gate exists to catch.
    unmatched_conclusions = 0
    unmatched_detail: list[dict[str, Any]] = []

    # Coverage / recall denominators.
    opponent_mons_seen = 0
    hp_mixed_mons_seen = 0  # candidate universe carries >= 2 max-HP families
    hp_mixed_mons_concluded = 0
    trimmed_truth_seen = 0
    trimmed_truth_concluded = 0
    strike_count = 0
    clean_strike_count = 0
    hp_pin_strikes = 0
    defense_pin_strikes = 0
    disqualifier_histogram: dict[str, int] = {}

    # Known-set calibration arm.
    control_clean = 0
    control_off_model = 0
    control_mismatch_examples: list[dict[str, Any]] = []

    # Premise + spread checks.
    denominator_mismatch_total = 0
    denominator_examples: list[dict[str, Any]] = []
    spread_checked = 0
    spread_mismatches: list[dict[str, Any]] = []
    turns_played: list[int] = []
    games_completed = 0

    hp_family_counts: dict[str, int] = {}
    for species_key, universe in source.universes.items():
        info = dex.species_info(species_key)
        if info is None or not info.base_stats:
            continue
        values = set()
        for variant in universe.variants:
            spread = randbats_spread_details(
                info.base_stats,
                level=variant.level,
                moves=variant.moves,
                item=variant.item,
                has_physical_attack=variant_has_physical_attack(variant.moves, dex),
            )
            values.add(spread.stats["hp"])
        hp_family_counts[species_key] = len(values)

    env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=root, set_belief_source=False))
    try:
        for game_index in range(args.games):
            game_seed = args.seed * 100_000 + game_index
            lines = _play_game(
                env, seed=game_seed, rng=rng, max_steps=args.max_steps, move_bias=args.move_bias
            )
            games_completed += 1
            replay = parse_showdown_replay(lines)
            turns_played.append(replay.turn_number)
            first_requests = _first_requests(lines)
            if "p1" not in first_requests or "p2" not in first_requests:
                continue
            truth = {slot: _team_truth(first_requests[slot]) for slot in ("p1", "p2")}

            mismatches = _denominator_mismatches(lines, truth)
            denominator_mismatch_total += len(mismatches)
            if mismatches and len(denominator_examples) < 8:
                denominator_examples.extend(mismatches[: 8 - len(denominator_examples)])

            # Spread replication cross-check (stats AND max HP, via the details path).
            for slot in ("p1", "p2"):
                for species_key, row in truth[slot].items():
                    info = dex.species_info(species_key)
                    if info is None or not row["stats"]:
                        continue
                    spread = randbats_spread_details(
                        info.base_stats,
                        level=row["level"],
                        moves=row["moves"],
                        item=row["item"],
                        has_physical_attack=variant_has_physical_attack(row["moves"], dex),
                    )
                    spread_checked += 1
                    diffs = {
                        stat: (spread.stats[stat], row["stats"][stat])
                        for stat in ("atk", "def", "spa", "spd", "spe")
                        if stat in row["stats"] and spread.stats[stat] != row["stats"][stat]
                    }
                    if row["maxhp"] is not None and spread.stats["hp"] != row["maxhp"]:
                        diffs["hp"] = (spread.stats["hp"], row["maxhp"])
                    if diffs:
                        spread_mismatches.append(
                            {"game": game_index, "species": species_key,
                             "diffs": {k: list(v) for k, v in diffs.items()}}
                        )

            revealed = {
                slot: {
                    canonical_gen3_randbat_species_id(mon.species)
                    for mon in replay.public_revealed.get(slot, ())
                }
                for slot in ("p1", "p2")
            }

            for perspective in ("p1", "p2"):
                opponent = "p2" if perspective == "p1" else "p1"
                own_team = own_team_from_request(first_requests[perspective])
                opponent_truth = truth[opponent]

                opponent_mons_seen += len(revealed[opponent])
                for key in revealed[opponent]:
                    row = opponent_truth.get(key)
                    if hp_family_counts.get(key, 1) >= 2:
                        hp_mixed_mons_seen += 1
                    if row is not None and row["stats"]:
                        info = dex.species_info(key)
                        if info is not None:
                            spread = randbats_spread_details(
                                info.base_stats,
                                level=row["level"],
                                moves=row["moves"],
                                item=row["item"],
                                has_physical_attack=variant_has_physical_attack(row["moves"], dex),
                            )
                            if spread.evs["hp"] < 85:
                                trimmed_truth_seen += 1

                inference = infer_investment(
                    replay,
                    perspective_slot=perspective,
                    own_team=own_team,
                    dex=dex,
                    set_source=source,
                    config=config,
                )
                strike_count += len(inference.strikes)
                for strike in inference.strikes:
                    for reason in strike.disqualifiers:
                        disqualifier_histogram[reason] = disqualifier_histogram.get(reason, 0) + 1
                    if not strike.disqualifiers and not strike.off_model:
                        clean_strike_count += 1
                        if strike.hp_pin is not None:
                            hp_pin_strikes += 1
                        if strike.defense_pin is not None:
                            defense_pin_strikes += 1

                for key, conclusion in inference.conclusions.items():
                    species_key = canonical_gen3_randbat_species_id(key.split(":", 1)[1])
                    row = opponent_truth.get(species_key)
                    if conclusion.hp_blocked:
                        hp_blocked_mons += 1
                    if conclusion.defense_blocked:
                        defense_blocked_mons += 1
                    if row is None:
                        unmatched_conclusions += 1
                        unmatched_detail.append({
                            "game": game_index,
                            "perspective": perspective,
                            "kind": "truth-row-missing",
                            "conclusion_key": key,
                            "canonical_species": species_key,
                        })
                        continue
                    info = dex.species_info(species_key)
                    true_spread = (
                        randbats_spread_details(
                            info.base_stats,
                            level=row["level"],
                            moves=row["moves"],
                            item=row["item"],
                            has_physical_attack=variant_has_physical_attack(row["moves"], dex),
                        )
                        if info is not None
                        else None
                    )
                    record = {
                        "game": game_index,
                        "perspective": perspective,
                        "mon": species_key,
                        "true_maxhp": row["maxhp"],
                        "true_moves": row["moves"],
                        "true_item": row["item"],
                    }
                    if conclusion.hp_value is not None:
                        if hp_family_counts.get(species_key, 1) >= 2:
                            hp_mixed_mons_concluded += 1
                        correct_value = conclusion.hp_value == row["maxhp"]
                        if correct_value:
                            hp_value_tp += 1
                        else:
                            hp_value_fp += 1
                            fp_detail.append({**record, "type": "hp-value",
                                              "pinned": conclusion.hp_value})
                        if conclusion.hp_class is not None and true_spread is not None:
                            true_class = "full" if true_spread.evs["hp"] == 85 else "trimmed"
                            if conclusion.hp_class == true_class:
                                hp_class_tp += 1
                                if true_class == "trimmed":
                                    trimmed_truth_concluded += 1
                            else:
                                hp_class_fp += 1
                                fp_detail.append({**record, "type": "hp-class",
                                                  "pinned": conclusion.hp_class,
                                                  "true_class": true_class})
                        if correct_value and len(tp_sample) < 10:
                            tp_sample.append({**record, "type": "hp",
                                              "pinned": conclusion.hp_value,
                                              "class": conclusion.hp_class,
                                              "turns": list(conclusion.hp_pin_turns)})
                    for stat_key, value in conclusion.defense_values.items():
                        true_value = row["stats"].get(stat_key)
                        if true_value is None:
                            # Same LOW-1 shape as the row miss: a defense conclusion for
                            # a stat the truth row lacks must be scored, not dropped.
                            unmatched_conclusions += 1
                            unmatched_detail.append({
                                "game": game_index,
                                "perspective": perspective,
                                "kind": "truth-stat-missing",
                                "conclusion_key": key,
                                "canonical_species": species_key,
                                "stat": stat_key,
                            })
                            continue
                        if value == true_value:
                            defense_tp += 1
                            if len(tp_sample) < 10:
                                tp_sample.append({**record, "type": f"defense-{stat_key}",
                                                  "pinned": value,
                                                  "class": conclusion.defense_classes.get(stat_key)})
                        else:
                            defense_fp += 1
                            fp_detail.append({**record, "type": f"defense-{stat_key}",
                                              "pinned": value, "true_value": true_value})

                # Known-set control: pin the defender candidates to the true sets and
                # measure lattice consistency of the true variant on clean strikes.
                truth_source = TrueVariantSource(opponent_truth)
                control = infer_investment(
                    replay,
                    perspective_slot=perspective,
                    own_team=own_team,
                    dex=dex,
                    set_source=truth_source,
                    config=config,
                )
                for strike in control.strikes:
                    if strike.disqualifiers and not strike.off_model:
                        continue
                    control_clean += 1
                    if strike.off_model:
                        control_off_model += 1
                        if len(control_mismatch_examples) < 8:
                            control_mismatch_examples.append(
                                {
                                    "game": game_index,
                                    "perspective": perspective,
                                    "defender": strike.defender_key,
                                    "move": strike.move_id,
                                    "observed_fraction": strike.observed_fraction,
                                    "candidate_hp_values": list(strike.candidate_hp_values),
                                }
                            )
    finally:
        env.close()

    hp_value_predictions = hp_value_tp + hp_value_fp
    hp_class_predictions = hp_class_tp + hp_class_fp
    defense_predictions = defense_tp + defense_fp
    total_fp = hp_value_fp + hp_class_fp + defense_fp
    consistency = (
        (control_clean - control_off_model) / control_clean if control_clean else None
    )

    precision_pass = hp_value_predictions >= 3 and total_fp == 0 and unmatched_conclusions == 0
    calibration_pass = (
        control_clean >= 200
        and consistency is not None
        and consistency >= 0.98
        and not spread_mismatches
        and denominator_mismatch_total == 0
    )
    verdict = "PASS" if precision_pass and calibration_pass else "FAIL"

    summary = {
        "generated": _dt.date.today().isoformat(),
        "harness": "scripts/investment_gate.py",
        "policy": f"move-biased random-legal (move_bias={args.move_bias}), both seats",
        "games_requested": args.games,
        "games_completed": games_completed,
        "seed": args.seed,
        "mean_turns": round(sum(turns_played) / len(turns_played), 2) if turns_played else None,
        "randbat_source_hash": source.metadata.source_hash,
        "config": {
            "lattice_tolerance_hp": config.lattice_tolerance_hp,
            "fraction_granularity": config.fraction_granularity,
            "rejection_margin_hp": config.rejection_margin_hp,
            "required_pin_strikes": config.required_pin_strikes,
        },
        "strikes_assessed": strike_count,
        "strikes_clean": clean_strike_count,
        "strikes_hp_pin": hp_pin_strikes,
        "strikes_defense_pin": defense_pin_strikes,
        "strike_disqualifier_histogram": dict(
            sorted(disqualifier_histogram.items(), key=lambda kv: -kv[1])
        ),
        "conclusions": {
            "hp_value": {
                "predictions": hp_value_predictions,
                "true_positives": hp_value_tp,
                "false_positives": hp_value_fp,
                "precision": hp_value_tp / hp_value_predictions if hp_value_predictions else None,
            },
            "hp_class": {
                "predictions": hp_class_predictions,
                "true_positives": hp_class_tp,
                "false_positives": hp_class_fp,
                "precision": hp_class_tp / hp_class_predictions if hp_class_predictions else None,
            },
            "defense": {
                "predictions": defense_predictions,
                "true_positives": defense_tp,
                "false_positives": defense_fp,
                "precision": defense_tp / defense_predictions if defense_predictions else None,
            },
            "blocked_mons": {"hp": hp_blocked_mons, "defense": defense_blocked_mons},
            "unmatched_conclusions": unmatched_conclusions,
            "unmatched_detail": unmatched_detail,
            "false_positive_detail": fp_detail,
            "true_positive_sample": tp_sample,
        },
        "coverage": {
            "opponent_mons_seen": opponent_mons_seen,
            "hp_mixed_family_mons_seen": hp_mixed_mons_seen,
            "hp_mixed_family_mons_concluded": hp_mixed_mons_concluded,
            "hp_conclusion_rate_on_mixed": (
                hp_mixed_mons_concluded / hp_mixed_mons_seen if hp_mixed_mons_seen else None
            ),
            "trimmed_truth_mons_seen": trimmed_truth_seen,
            "trimmed_truth_mons_concluded": trimmed_truth_concluded,
        },
        "calibration_known_set": {
            "clean_strikes": control_clean,
            "off_model_strikes": control_off_model,
            "true_variant_consistency": consistency,
            "mismatch_examples": control_mismatch_examples,
            "spread_mons_checked": spread_checked,
            "spread_stat_mismatches": len(spread_mismatches),
            "spread_mismatch_examples": spread_mismatches[:8],
            "hp_denominator_mismatches": denominator_mismatch_total,
            "hp_denominator_examples": denominator_examples,
        },
        "gate": {
            "precision_pass": precision_pass,
            "calibration_pass": calibration_pass,
            "verdict": verdict,
            "criteria": {
                "precision": "zero false pins across all conclusion types, zero unmatched "
                             "conclusions (every conclusion scored against truth), and "
                             ">= 3 hp-value conclusions",
                "calibration": ">= 200 clean control strikes, true-variant consistency >= 0.98, "
                               "zero spread mismatches, zero HP-denominator mismatches",
            },
            "mask_recommendation": (
                "populate NUMERIC_TT_INVESTMENT_BIT behind tier2_investment"
                if verdict == "PASS"
                else "keep NUMERIC_TT_INVESTMENT_BIT zero-masked"
            ),
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["gate"], indent=2))
    print(
        f"hp-value: {hp_value_predictions} predictions "
        f"(tp={hp_value_tp}, fp={hp_value_fp}); hp-class fp={hp_class_fp}; "
        f"defense: {defense_predictions} predictions (fp={defense_fp})"
    )
    print(f"control: {control_clean} clean strikes, consistency={consistency}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
