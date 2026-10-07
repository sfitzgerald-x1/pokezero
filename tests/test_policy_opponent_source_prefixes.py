"""Prefix receipts retain refusals and never certify live replay or search."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.mcts_eval.policy_opponent_roster import RosterError, load_frozen_roster
from pokezero.mcts_eval.policy_opponent_source_prefixes import qualify_source_prefixes
from pokezero.public_decision_corpus import (
    PublicActionIdentifier, PublicActorObservation, PublicDecisionRecord,
    PublicObservation, PublicResolvedActionRound, public_decision_id,
)

ROOT = Path(__file__).resolve().parents[1]
ROSTER_SHA = "fa76bc9966bba48f0dd474a2450b4c362e49ac1452db18c39b084a256d30ff7a"


def fixture(base):
    roster = json.loads(json.dumps(load_frozen_roster(
        ROOT / "docs/paper_policy_opponent_roster_20261004.json", expected_sha256=ROSTER_SHA)))
    mask = tuple(index == 0 for index in range(ACTION_COUNT))
    observation = PublicObservation("test", (), (), (), (), mask,
        {"action_candidates": [{"action_index": 0, "kind": "move", "move_id": "tackle", "legal": True}]})
    move = PublicActionIdentifier(kind="move", move_id="tackle")
    unresolved = PublicActionIdentifier(kind="event", event_id="unresolved-public-event")
    identities = {(row["seed"], row["seat"], row["turn_index"])
                  for row in roster["profile_roots"] + roster["prefix_witnesses"]}
    bindings = {}
    for seed, seat, turn in sorted(identities):
        rounds = [PublicResolvedActionRound(index, {"p1": move, "p2": move}) for index in range(turn)]
        if turn == 9 and seed == 2026100100 and seat == "p1":
            rounds[0] = PublicResolvedActionRound(0, {"p1": unresolved, "p2": move})
        if turn == 9 and seed == 2026100104 and seat == "p2":
            rounds[0] = PublicResolvedActionRound(0, {"p1": unresolved, "p2": move})
        record = PublicDecisionRecord("pending", f"mcts-h2h-{seed}-{seat}", seed,
            "gen3randombattle", seat, turn, 0, observation,
            tuple(PublicActorObservation(index, observation) for index in range(turn)),
            mask, tuple(rounds), {})
        record = replace(record, decision_id=public_decision_id(record))
        relative = (f"shards/s0/public-decision-records/seed-{seed}-{seat}/"
                    f"turn-{turn:03d}-{record.decision_id}.json")
        raw = json.dumps({"schema_version": "pokezero.mcts-guided-vs-raw-public-decision.v1",
                          "record": record.to_dict()}).encode()
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        bindings[(seed, seat, turn)] = dict(decision_id=record.decision_id,
            source_relative_path=relative, source_file_sha256=hashlib.sha256(raw).hexdigest())
    for row in roster["profile_roots"] + roster["prefix_witnesses"]:
        row.update(bindings[(row["seed"], row["seat"], row["turn_index"])])
    roster["proposed_continuation_root_ids"] = [r["decision_id"] for r in roster["profile_roots"] if r["turn_index"] == 9]
    terminal = json.dumps(dict(status="COMPLETE", games=50, pairs=25)).encode()
    (base / "shards/s0/COMPLETE.json").write_bytes(terminal)
    roster["source_terminal_sha256"] = hashlib.sha256(terminal).hexdigest()
    path = base / "roster.json"
    raw = json.dumps(roster).encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), roster


class FrozenSourcePrefixPreflightTests(unittest.TestCase):
    def test_all_roots_survive_one_refusal_and_source_owned_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path, digest, roster = fixture(base)
            result = qualify_source_prefixes(path, expected_roster_sha256=digest, source_root=base)
            self.assertEqual([r["decision_id"] for r in result["roots"]],
                             [r["decision_id"] for r in roster["profile_roots"]])
            self.assertEqual((result["root_denominator"], result["prefix_valid"], result["refused"]), (32, 31, 1))
            self.assertEqual(result["source_files_verified"], 155)
            self.assertEqual(sum(len(r["source_owned_repairs"]) for r in result["roots"]), 1)
            refused = [r for r in result["roots"] if r["state"] == "REFUSED"]
            self.assertEqual(refused[0]["refusal"], "unresolved_public_event_for_non_source_player")
            self.assertFalse(result["live_replay_qualified"])
            self.assertFalse(result["profile_qualified"])
            self.assertEqual(result["replacement_roots"], [])
            self.assertNotIn(result["status"], ("PASS", "COMPLETE"))

    def test_changed_source_or_selection_stops_the_whole_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path, digest, roster = fixture(base)
            source = base / roster["profile_roots"][0]["source_relative_path"]
            source.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(RosterError, "source bytes drift"):
                qualify_source_prefixes(path, expected_roster_sha256=digest, source_root=base)
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(RosterError, "roster byte hash drift"):
                qualify_source_prefixes(path, expected_roster_sha256=digest, source_root=base)

    def test_noncanonical_record_cannot_be_reclassified_as_an_eligibility_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path, digest, roster = fixture(base)
            row = roster["profile_roots"][0]
            source = base / row["source_relative_path"]
            wrapper = json.loads(source.read_bytes())
            wrapper["record"]["recorded_action_index"] = 1
            raw = json.dumps(wrapper).encode()
            source.write_bytes(raw)
            for reference in roster["profile_roots"] + roster["prefix_witnesses"]:
                if reference["source_relative_path"] == row["source_relative_path"]:
                    reference["source_file_sha256"] = hashlib.sha256(raw).hexdigest()
            raw = json.dumps(roster).encode()
            path.write_bytes(raw)
            with self.assertRaisesRegex(RosterError, "invalid canonical source record"):
                qualify_source_prefixes(path, expected_roster_sha256=hashlib.sha256(raw).hexdigest(), source_root=base)
