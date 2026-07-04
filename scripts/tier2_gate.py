#!/usr/bin/env python3
"""Tier-2 gate harness (next-train readiness PR D).

Generates omniscient controlled games on the local Showdown BattleStream (both seats
driven by a seeded uniform-random-legal policy), then:

1. runs the Tier-2 inference (``pokezero.tier2.infer_tier2``) from the protocol-line
   view with the real randbats candidate-set source, and scores the Choice Band bit
   against ground-truth items read from each seat's own opening request (the server
   knows the true teams — omniscient truth, no oracle leakage into the inference);
2. runs a known-set control pass (a set source that pins each opponent mon to its true
   set) and checks residual calibration: on clean strikes the observed damage must land
   exactly on one of the 16 predicted rolls, the roll positions should be ~uniform, and
   the log-ratio to the expected median should be ~0-mean;
3. cross-checks ``randbats_spread_stats`` against the server-computed stats of every
   mon in every game (the generator-spread replication must be exact);
4. writes a small JSON summary (the artifact quoted in the PR) with the gate verdict.

Usage:
    uv run python scripts/tier2_gate.py --games 48 --seed 7 \
        --showdown-root ~/workspace/pokerena/vendor/pokemon-showdown \
        --out runs/tier2-gate-2026-07-04
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero.actions import MOVE_ACTION_COUNT  # noqa: E402
from pokezero.belief import CandidateSetSummary  # noqa: E402
from pokezero.dex import load_showdown_dex_cached, normalize_id  # noqa: E402
from pokezero.gen3_damage import ROLL_NUMERATORS, randbats_spread_stats  # noqa: E402
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv  # noqa: E402
from pokezero.randbat import canonical_gen3_randbat_species_id, load_gen3_randbat_source_cached  # noqa: E402
from pokezero.showdown import parse_showdown_replay  # noqa: E402
from pokezero.tier2 import (  # noqa: E402
    Tier2Config,
    build_cb_whitelist,
    canonical_move_id,
    own_team_from_request,
    infer_tier2,
    variant_has_physical_attack,
)

_STAT_KEYS = ("atk", "def", "spa", "spd", "spe")


class TrueVariantSource:
    """A ``PokemonSetSource`` that pins every known species to its true generated set.

    Used only by the known-set calibration arm: candidate ambiguity is removed so the
    residual distribution isolates the damage model. Species names collide across
    teams only for the perspective's OWN side, whose beliefs the inference never reads.
    """

    def __init__(self, variants_by_species: Mapping[str, Mapping[str, Any]]) -> None:
        self._variants = dict(variants_by_species)

    def summarize(self, *, format_id, species, revealed_moves, **kwargs) -> CandidateSetSummary | None:
        variant = self._variants.get(canonical_gen3_randbat_species_id(species))
        if variant is None:
            return None
        return CandidateSetSummary(
            species=species,
            candidate_count=1,
            uncertainty=0.0,
            possible_abilities=(str(variant.get("ability") or ""),),
            possible_items=(str(variant.get("item") or ""),),
            possible_moves=tuple(variant.get("moves") or ()),
            candidate_variants=(dict(variant),),
        )


def _first_requests(lines: tuple[str, ...]) -> dict[str, Mapping[str, Any]]:
    first: dict[str, Mapping[str, Any]] = {}
    for line in lines:
        if not line.startswith("|request|"):
            continue
        try:
            payload = json.loads(line[len("|request|"):])
        except json.JSONDecodeError:
            continue
        side = payload.get("side") if isinstance(payload, Mapping) else None
        slot = side.get("id") if isinstance(side, Mapping) else None
        if slot in {"p1", "p2"} and slot not in first:
            first[slot] = payload
    return first


def _team_truth(request: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """species_key -> true variant payload {moves, ability, item, level, stats, maxhp}."""
    truth: dict[str, dict[str, Any]] = {}
    side = request.get("side")
    rows = side.get("pokemon") if isinstance(side, Mapping) else None
    if not isinstance(rows, list):
        return truth
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        details = str(row.get("details") or "")
        species = details.split(",", 1)[0].strip()
        if not species:
            continue
        level = 100
        for chunk in details.split(",")[1:]:
            chunk = chunk.strip()
            if chunk.startswith("L"):
                try:
                    level = int(chunk[1:])
                except ValueError:
                    pass
        condition = str(row.get("condition") or "")
        max_hp: Optional[int] = None
        head = condition.split()[0] if condition else ""
        if "/" in head:
            try:
                max_hp = int(head.split("/", 1)[1])
            except ValueError:
                max_hp = None
        truth[canonical_gen3_randbat_species_id(species)] = {
            "species": species,
            "moves": [canonical_move_id(str(move)) for move in (row.get("moves") or [])],
            "ability": str(row.get("baseAbility") or row.get("ability") or ""),
            "item": str(row.get("item") or ""),
            "level": level,
            "stats": {
                stat: int(value) for stat, value in (row.get("stats") or {}).items() if isinstance(value, int)
            },
            "maxhp": max_hp,
        }
    return truth


def _play_game(
    env: LocalShowdownEnv, *, seed: int, rng: random.Random, max_steps: int, move_bias: float = 0.75
) -> tuple[str, ...]:
    env.reset(seed=seed)
    steps = 0
    while steps < max_steps and env.terminal() is None:
        requested = env.requested_players()
        if not requested:
            break
        actions = {}
        for player in requested:
            mask = env.observe(player).legal_action_mask
            legal = [index for index, allowed in enumerate(mask) if allowed]
            if not legal:
                return env.protocol_lines
            # Move-biased random policy: with probability ``move_bias`` pick among
            # legal MOVE actions (indices < MOVE_ACTION_COUNT) when any exist. Random
            # play is switch-heavy (5 of 9 actions), which starves the CB arm of
            # repeat strikes; the bias raises strike density without changing the
            # damage distribution any strike is drawn from.
            moves = [index for index in legal if index < MOVE_ACTION_COUNT]
            if moves and rng.random() < move_bias:
                actions[player] = rng.choice(moves)
            else:
                actions[player] = rng.choice(legal)
        env.step(actions)
        steps += 1
    return env.protocol_lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--move-bias", type=float, default=0.75)
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.showdown_root.expanduser().resolve()
    dex = load_showdown_dex_cached(root)
    source = load_gen3_randbat_source_cached(root)
    whitelist = build_cb_whitelist(source.universes, dex)
    rng = random.Random(args.seed)
    config = Tier2Config()
    calibration_config = Tier2Config(baseline_includes_choice_band=True)

    cb_tp = cb_fp = 0
    fp_detail: list[dict[str, Any]] = []
    tp_detail: list[dict[str, Any]] = []
    true_cb_total = 0
    true_cb_seen = 0
    true_cb_hit = 0
    strike_count = 0
    eligible_strike_count = 0
    residual_valid_count = 0
    disqualifier_histogram: dict[str, int] = {}

    roll_hits = 0
    roll_total = 0
    roll_histogram = [0] * len(ROLL_NUMERATORS)
    log_ratios: list[float] = []
    mismatch_examples: list[dict[str, Any]] = []
    spread_checked = 0
    spread_mismatches: list[dict[str, Any]] = []
    turns_played: list[int] = []
    games_completed = 0

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

            # Spread replication cross-check against server-computed stats.
            for slot in ("p1", "p2"):
                for species_key, row in truth[slot].items():
                    info = dex.species_info(species_key)
                    if info is None or not row["stats"]:
                        continue
                    computed = randbats_spread_stats(
                        info.base_stats,
                        level=row["level"],
                        moves=row["moves"],
                        item=row["item"],
                        has_physical_attack=variant_has_physical_attack(row["moves"], dex),
                    )
                    spread_checked += 1
                    diffs = {
                        stat: (computed[stat], row["stats"][stat])
                        for stat in _STAT_KEYS
                        if stat in row["stats"] and computed[stat] != row["stats"][stat]
                    }
                    if row["maxhp"] is not None and computed["hp"] != row["maxhp"]:
                        diffs["hp"] = (computed["hp"], row["maxhp"])
                    if diffs:
                        spread_mismatches.append(
                            {"game": game_index, "species": species_key, "diffs": {k: list(v) for k, v in diffs.items()}}
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

                inference = infer_tier2(
                    replay,
                    perspective_slot=perspective,
                    own_team=own_team,
                    dex=dex,
                    set_source=source,
                    whitelist=whitelist,
                    config=config,
                )
                strike_count += len(inference.strikes)
                eligible_strike_count += sum(1 for strike in inference.strikes if strike.cb_eligible)
                residual_valid_count += sum(1 for strike in inference.strikes if strike.residual_valid)
                for strike in inference.strikes:
                    for reason in strike.disqualifiers:
                        disqualifier_histogram[reason] = disqualifier_histogram.get(reason, 0) + 1

                cb_truth_keys = {
                    key for key, row in opponent_truth.items() if normalize_id(row["item"]) == "choiceband"
                }
                true_cb_total += len(cb_truth_keys)
                seen_cb = cb_truth_keys & revealed[opponent]
                true_cb_seen += len(seen_cb)
                for key, bit in inference.cb_bits.items():
                    if not bit:
                        continue
                    species_key = canonical_gen3_randbat_species_id(key.split(":", 1)[1])
                    record = {
                        "game": game_index,
                        "perspective": perspective,
                        "mon": species_key,
                        "true_item": opponent_truth.get(species_key, {}).get("item"),
                        "strike_turns": list(inference.cb_strike_turns.get(key, ())),
                    }
                    if species_key in cb_truth_keys:
                        cb_tp += 1
                        true_cb_hit += 1
                        tp_detail.append(record)
                    else:
                        cb_fp += 1
                        fp_detail.append(record)

                # Known-set control pass: pin the opponent's true sets.
                truth_source = TrueVariantSource(opponent_truth)
                control = infer_tier2(
                    replay,
                    perspective_slot=perspective,
                    own_team=own_team,
                    dex=dex,
                    set_source=truth_source,
                    whitelist=whitelist,
                    config=calibration_config,
                )
                for strike in control.strikes:
                    if not strike.residual_valid or not strike.baseline_rolls:
                        continue
                    if strike.observed_hp is None or strike.expected_median_hp is None:
                        continue
                    token = control.tokens[strike.token_index]
                    if token.crit or token.n_hits != 1:
                        continue  # calibration is over clean single-hit, non-crit strikes
                    roll_total += 1
                    rolls = strike.baseline_rolls
                    if strike.observed_hp in rolls:
                        roll_hits += 1
                        roll_histogram[rolls.index(strike.observed_hp)] += 1
                    elif len(mismatch_examples) < 8:
                        mismatch_examples.append(
                            {
                                "game": game_index,
                                "perspective": perspective,
                                "attacker": strike.attacker_key,
                                "move": strike.move_id,
                                "observed": strike.observed_hp,
                                "rolls": list(rolls),
                            }
                        )
                    if strike.expected_median_hp > 0 and strike.observed_hp > 0:
                        log_ratios.append(math.log(strike.observed_hp / strike.expected_median_hp))
    finally:
        env.close()

    predictions = cb_tp + cb_fp
    precision = cb_tp / predictions if predictions else None
    recall_seen = true_cb_hit / true_cb_seen if true_cb_seen else None
    recall_all = true_cb_hit / true_cb_total if true_cb_total else None
    mean_log = sum(log_ratios) / len(log_ratios) if log_ratios else None
    std_log = (
        math.sqrt(sum((value - mean_log) ** 2 for value in log_ratios) / len(log_ratios))
        if log_ratios and mean_log is not None
        else None
    )
    roll_match_rate = roll_hits / roll_total if roll_total else None

    precision_pass = predictions > 0 and cb_fp == 0 and cb_tp >= 3
    calibration_pass = (
        roll_total >= 200
        and roll_match_rate is not None
        and roll_match_rate >= 0.98
        and mean_log is not None
        and abs(mean_log) <= 0.01
        and not spread_mismatches
    )
    verdict = "PASS" if precision_pass and calibration_pass else "FAIL"

    summary = {
        "generated": _dt.date.today().isoformat(),
        "harness": "scripts/tier2_gate.py",
        "policy": f"move-biased random-legal (move_bias={args.move_bias}), both seats",
        "games_requested": args.games,
        "games_completed": games_completed,
        "seed": args.seed,
        "mean_turns": round(sum(turns_played) / len(turns_played), 2) if turns_played else None,
        "randbat_source_hash": source.metadata.source_hash,
        "whitelist_species": len(whitelist),
        "whitelist_pairs": sum(len(moves) for moves in whitelist.values()),
        "strikes_assessed": strike_count,
        "strikes_cb_eligible": eligible_strike_count,
        "strikes_residual_valid": residual_valid_count,
        "strike_disqualifier_histogram": dict(sorted(disqualifier_histogram.items(), key=lambda kv: -kv[1])),
        "cb": {
            "predictions": predictions,
            "true_positives": cb_tp,
            "false_positives": cb_fp,
            "precision": precision,
            "true_cb_mons_total": true_cb_total,
            "true_cb_mons_seen": true_cb_seen,
            "recall_seen": recall_seen,
            "recall_all": recall_all,
            "false_positive_detail": fp_detail,
            "true_positive_sample": tp_detail[:10],
        },
        "calibration_known_set": {
            "clean_strikes": roll_total,
            "roll_match_rate": roll_match_rate,
            "mean_log_ratio": mean_log,
            "std_log_ratio": std_log,
            "roll_histogram": roll_histogram,
            "mismatch_examples": mismatch_examples,
            "spread_mons_checked": spread_checked,
            "spread_stat_mismatches": len(spread_mismatches),
            "spread_mismatch_examples": spread_mismatches[:8],
        },
        "gate": {
            "precision_pass": precision_pass,
            "calibration_pass": calibration_pass,
            "verdict": verdict,
            "criteria": {
                "precision": "zero false positives and >= 3 true positives",
                "calibration": ">= 200 clean strikes, roll-match >= 0.98, |mean log-ratio| <= 0.01, zero spread mismatches",
            },
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["gate"], indent=2))
    print(f"cb: {summary['cb']['predictions']} predictions, precision={precision}, recall_seen={recall_seen}")
    print(f"calibration: {roll_total} strikes, roll_match={roll_match_rate}, mean_log={mean_log}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
