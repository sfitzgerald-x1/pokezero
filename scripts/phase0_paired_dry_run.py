#!/usr/bin/env python3
"""Read a live fixed primary roster through its frozen descriptive reader.

This never attests final execution or enables inference. Public terminal checks
are provisional saved-metadata checks, not the independent final audit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def checked_outcome(reader, row, seat, *, round_cap, candidate_name="PokeZeroBot",
                    opponent_name="FoulPlayBot", format_id="gen3randombattle"):
    """Return an unknown cell on a conflicting terminal, never a legacy draw."""
    seed = row["seed"]
    unknown = reader.Outcome(seed, seat, "FAILED")
    witness = row.get("terminal_witness")
    if not isinstance(witness, dict):
        return unknown
    rounds = row.get("decision_rounds")
    capped, tied = row.get("capped"), row.get("tied")
    players = witness.get("players")
    expected_players = {seat: candidate_name, "p2" if seat == "p1" else "p1": opponent_name}
    if (type(rounds) is not int or not 0 < rounds <= round_cap
            or type(capped) is not bool or type(tied) is not bool
            or witness.get("schema") != "pokezero.controlled-terminal-witness.v1"
            or witness.get("capped") is not capped
            or type(witness.get("decision_rounds")) is not int
            or witness["decision_rounds"] != rounds
            or type(witness.get("max_decision_rounds")) is not int
            or witness["max_decision_rounds"] != round_cap
            or not isinstance(players, dict) or set(players) != {"p1", "p2"}
            or any(not isinstance(x, str) or not x for x in players.values())
            or players["p1"] == players["p2"] or players != expected_players
            or row.get("battle_id") != f"battle-{format_id}-controlled-{seed}"
            or type(row.get("pokezero_won")) is not bool
            or "winner" not in row):
        return unknown
    status, terminal = witness.get("status"), witness.get("public_terminal_line")
    winner = row.get("winner")
    if capped:
        if (status == "ROUND_CAP" and rounds == round_cap and not tied
                and terminal is None and winner is None
                and witness.get("winner_player") is None and not row["pokezero_won"]):
            return reader.Outcome(seed, seat, "CAPPED")
        return unknown
    score = row.get("pokezero_score")
    if type(score) not in (int, float):
        return unknown
    if (status == "PUBLIC_TIE" and terminal in ("|tie", "|tie|")
            and tied and winner is None and witness.get("winner_player") is None
            and score == 0.5 and not row["pokezero_won"]):
        return reader.Outcome(seed, seat, "COMPLETE", 0.5, True)
    if (status == "PUBLIC_WIN" and not tied and winner in players.values()
            and terminal == "|win|" + winner
            and witness.get("winner_player") in ("p1", "p2")
            and players[witness["winner_player"]] == winner
            and score == int(winner == players[seat])
            and row["pokezero_won"] == (winner == players[seat])):
        return reader.Outcome(seed, seat, "COMPLETE", score, True)
    return unknown


def read_panel(entry, reader):
    seeds = list(range(entry["seed_start"], entry["seed_start"] + entry["seed_count"]))
    common = entry["common_arguments"]
    round_cap = int(common[common.index("--max-decision-rounds") + 1])
    arms = {"candidate": [], "raw": []}
    snapshots = {}
    for shard in entry["shards"]:
        name, seat = shard["name"], shard["seat"]
        arm = "candidate" if name == "candidate-" + seat else "raw"
        if name not in {arm + "-" + seat} or seat not in ("p1", "p2"):
            raise ValueError("unexpected registered shard")
        path = Path(entry["run_directory"]) / (name + ".json")
        if not path.exists():
            continue
        payload = path.read_bytes()  # One atomically replaced summary snapshot.
        doc = json.loads(payload)
        rows = doc["game_results"]
        if (type(doc.get("completed_games")) is not int
                or doc["completed_games"] != len(rows) or len(rows) > len(seeds)
                or [r["seed"] for r in rows] != seeds[:len(rows)]
                or any(type(r["seed"]) is not int for r in rows)
                or doc.get("pokezero_player") != seat
                or doc.get("opponent_policy_id") != "foul-play"
                or doc.get("policy_mode") != ("raw" if arm == "raw" else "engine-mcts")):
            raise ValueError("summary is not the registered arm/seat chronological prefix")
        def argument(flag, default):
            return common[common.index(flag) + 1] if flag in common else default
        outcomes = [checked_outcome(reader, r, seat, round_cap=round_cap,
                    candidate_name=argument("--pokezero-username", "PokeZeroBot"),
                    opponent_name=argument("--foulplay-username", "FoulPlayBot"),
                    format_id=argument("--format", "gen3randombattle")) for r in rows]
        arms[arm].extend(outcomes)
        snapshots[name] = dict(
            sha256=hashlib.sha256(payload).hexdigest(), recorded=len(rows),
            provisional_complete=sum(o.validated for o in outcomes),
            capped_unknown=sum(o.status == "CAPPED" for o in outcomes),
            conflicting_unknown=sum(o.status == "FAILED" for o in outcomes),
        )
    result = reader.summarize_capped_pairs(seeds, arms["candidate"], arms["raw"],
                                          normal_complete_attempt=False)
    assert result["primary_test"] is None and not result["screening_rule_passed"]
    return dict(schema="pokezero.phase0-paired-live-dry-run.v1", snapshots=snapshots,
                planned_cells=4 * len(seeds),
                recorded_cells=sum(s["recorded"] for s in snapshots.values()),
                independent_final_validation=False, result=result,
                interpretation="Descriptive dry run only. Every missing/capped/conflicting cell is [0,1]. No interim test, stopping decision, final execution attestation or advancement.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True, type=Path)
    args = parser.parse_args()
    entry_payload = args.entry.read_bytes()
    entry = json.loads(entry_payload)
    frozen = entry["analysis"]["reader"]
    if frozen["path"] != "src/pokezero/mcts_eval/fullgame_strength.py":
        raise ValueError("unexpected frozen reader path")
    path = Path(entry["runtime"]) / frozen["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != frozen["sha256"]:
        raise ValueError("frozen reader hash mismatch")
    spec = importlib.util.spec_from_file_location("phase0_frozen_reader", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = read_panel(entry, module)
    result["entry_sha256"] = hashlib.sha256(entry_payload).hexdigest()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
