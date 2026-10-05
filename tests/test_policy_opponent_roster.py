"""Selection integrity does not imply root replay eligibility or strength."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pokezero.mcts_eval.policy_opponent_roster import (
    RosterError, load_frozen_roster, validate_roster, verify_source_files,
)

ROSTER = Path(__file__).resolve().parents[1] / "docs/paper_policy_opponent_roster_20261004.json"
ROSTER_SHA256 = "fa76bc9966bba48f0dd474a2450b4c362e49ac1452db18c39b084a256d30ff7a"


class FrozenPolicyOpponentRosterTests(unittest.TestCase):
    def setUp(self):
        self.payload = dict(load_frozen_roster(ROSTER, expected_sha256=ROSTER_SHA256))

    def test_roster_keeps_32_roots_eight_clusters_and_16_continuation_roots(self):
        self.assertEqual(len(self.payload["profile_roots"]), 32)
        self.assertEqual(len({r["seed"] for r in self.payload["profile_roots"]}), 8)
        self.assertEqual(len(self.payload["proposed_continuation_root_ids"]), 16)
        self.assertEqual(len(self.payload["prefix_witnesses"]), 138)
        self.assertEqual(len(self.payload["uncaptured_prior_records"]), 6)

    def test_no_dropping_reordering_redrawing_or_aliasing_roots(self):
        for operation in (lambda p: p["profile_roots"].pop(),
                          lambda p: p["profile_roots"].reverse(),
                          lambda p: p["selection_rule"].update(decision_indices=[False,9]),
                          lambda p: p["selection_rule"].update(paired_seeds=list(range(2026100101,2026100109))),
                          lambda p: p["profile_roots"][1].update(decision_id=p["profile_roots"][0]["decision_id"])):
            payload = copy.deepcopy(self.payload)
            operation(payload)
            with self.assertRaises(RosterError):
                validate_roster(payload)

    def test_uncaptured_prefixes_cannot_be_hidden_and_refusals_cannot_be_redrawn(self):
        for operation in (lambda p: p.update(uncaptured_prior_records=[]),
                          lambda p: p["qualification_rules"].update(unknown_other_player_actions_refuse=False),
                          lambda p: p["qualification_rules"].update(missing_or_unsupported_root_disposition="redraw"),
                          lambda p: p["prefix_witnesses"].pop()):
            payload = copy.deepcopy(self.payload)
            operation(payload)
            with self.assertRaises(RosterError):
                validate_roster(payload)

    def test_byte_binding_is_required_even_for_a_semantically_identical_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roster.json"
            path.write_text(json.dumps(self.payload))
            with self.assertRaisesRegex(RosterError, "byte hash"):
                load_frozen_roster(path, expected_sha256=ROSTER_SHA256)
            with self.assertRaisesRegex(RosterError, "explicit frozen"):
                load_frozen_roster(path, expected_sha256=None)

    def test_source_bytes_and_identity_are_verified_before_use(self):
        payload = copy.deepcopy(self.payload)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            seen = {}
            for row in payload["profile_roots"] + payload["prefix_witnesses"]:
                wrapper = {"schema_version": "pokezero.mcts-guided-vs-raw-public-decision.v1",
                           "record": {"seed": row["seed"], "acting_player": row["seat"],
                                      "turn_index": row["turn_index"], "battle_id": row["battle_id"],
                                      "decision_id": row["decision_id"]}}
                raw = json.dumps(wrapper).encode()
                digest = hashlib.sha256(raw).hexdigest()
                row["source_file_sha256"] = digest
                path = base / row["source_relative_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                seen[row["source_relative_path"]] = path
            terminal = json.dumps({"status": "COMPLETE", "games": 50, "pairs": 25}).encode()
            (base / "shards/s0/COMPLETE.json").write_bytes(terminal)
            payload["source_terminal_sha256"] = hashlib.sha256(terminal).hexdigest()
            verify_source_files(base, payload)
            first = next(iter(seen.values()))
            first.write_bytes(first.read_bytes() + b"\n")
            with self.assertRaisesRegex(RosterError, "source bytes drift"):
                verify_source_files(base, payload)

    def test_selection_receipt_cannot_be_promoted_to_a_valid_execution(self):
        self.payload["execution_status"] = "PASS"
        with self.assertRaisesRegex(RosterError, "not execution evidence"):
            validate_roster(self.payload)
