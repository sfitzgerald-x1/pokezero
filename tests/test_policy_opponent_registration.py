"""A registered contract cannot be tuned, redrawn or mistaken for evidence."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pokezero.mcts_eval.policy_opponent_roster import load_frozen_roster
from pokezero.mcts_eval.policy_opponent_registration import (
    CHAMPION_SHA256, ROSTER_SHA256, SHOWDOWN_SHA256, canonical_sha256,
    freeze_registration, load_registration, remaining_study_seconds, validate_registration, wall_statistics,
)
from pokezero.mcts_eval.resolver import ContractError


ROOT = Path(__file__).resolve().parents[1]


def registration():
    roster = load_frozen_roster(ROOT / "docs/paper_policy_opponent_roster_20261004.json",
                              expected_sha256=ROSTER_SHA256)
    runtime = {field: "a" * 64 for field in ("source_tree_sha256", "engine_fingerprint",
        "source_receipt_sha256", "model_sha256", "tables_sha256", "observation_contract_sha256")}
    runtime.update(source_commit="b" * 40, immutable_image="registry/example@sha256:" + "c" * 64,
                   checkpoint_sha256=CHAMPION_SHA256, showdown_runtime_sha256=SHOWDOWN_SHA256)
    return roster, freeze_registration(roster, roster_sha256=ROSTER_SHA256, expected_runtime=runtime)


class ProfileRegistrationTests(unittest.TestCase):
    def test_all_original_roots_domains_and_pending_qualification_are_explicit(self):
        roster, payload = registration()
        validate_registration(payload, roster=roster)
        self.assertEqual(payload["root_denominator"], 32)
        self.assertEqual([r["decision_id"] for r in payload["roots"]],
                         [r["decision_id"] for r in roster["profile_roots"]])
        self.assertEqual(len(payload["proposed_continuation_root_ids"]), 16)
        self.assertNotIn(payload["status"], ("PASS", "COMPLETE"))
        self.assertFalse(payload["analysis"]["strength_qualified"])
        self.assertFalse(payload["semantics"]["complete_paper_reproduction"])
        self.assertEqual(len({r["decision_seed"] for r in payload["roots"]}), 32)
        self.assertTrue(all(r["decision_seed"] != r["opponent_seed"] for r in payload["roots"]))
        self.assertEqual(canonical_sha256(payload), canonical_sha256(deepcopy(payload)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registration.json"
            raw = json.dumps(payload).encode()
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            self.assertEqual(load_registration(path, expected_sha256=digest, roster=roster), payload)
            changed = deepcopy(payload)
            changed["expected_runtime"]["source_commit"] = "c" * 40
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ContractError, "byte hash drift"):
                load_registration(path, expected_sha256=digest, roster=roster)

    def test_semantic_budget_roster_and_analysis_drift_are_rejected(self):
        roster, payload = registration()
        mutations = [lambda p: p["config"].update(sims=512), lambda p: p["semantics"].update(c_puct=2),
            lambda p: p["timing"].update(deadline_ms=2000), lambda p: p["budget"].update(study_cpu_hours=97),
            lambda p: p["budget"].update(shared_with_continuations=False), lambda p: p["roots"].pop(),
            lambda p: p["roots"].reverse(), lambda p: p["roots"][0].update(decision_seed=7),
            lambda p: p["analysis"].update(strength_qualified=True), lambda p: p["arms"].reverse(),
            lambda p: p.update(extra="unregistered"), lambda p: p.update(seed=True)]
        for mutate in mutations:
            changed = deepcopy(payload)
            mutate(changed)
            with self.assertRaises(ContractError):
                validate_registration(changed, roster=roster)

    def test_runtime_identity_cannot_be_partial_tag_only_or_change_the_champion(self):
        roster, payload = registration()
        for field, value in (("immutable_image", "registry/example:latest"), ("engine_fingerprint", "A" * 64),
                             ("checkpoint_sha256", "d" * 64), ("showdown_runtime_sha256", "e" * 64)):
            runtime = deepcopy(payload["expected_runtime"])
            runtime[field] = value
            with self.assertRaises(ContractError):
                freeze_registration(roster, roster_sha256=ROSTER_SHA256, expected_runtime=runtime)
        runtime.pop("source_receipt_sha256")
        with self.assertRaises(ContractError):
            freeze_registration(roster, roster_sha256=ROSTER_SHA256, expected_runtime=runtime)

    def test_shared_clock_and_cpu_ceiling_cannot_be_reset_or_exceeded(self):
        def left(now=1001., cpu=0., reserved=2):
            return remaining_study_seconds(study_started_unix_s=1000., now_unix_s=now,
                                           prior_cpu_hours=cpu, reserved_cpus=reserved)
        self.assertEqual(left(), 7199.)
        self.assertEqual(left(now=8200.), 0.)
        self.assertEqual(left(cpu=95.5), 900.)
        self.assertEqual(left(cpu=96.), 0.)
        for now, cpu, reserved in ((999., 0., 2), (float("nan"), 0., 2), (1001., -1., 2),
                                   (1001., 0., 49), (1001., 0., True)):
            with self.assertRaises(ContractError):
                left(now, cpu, reserved)

    def test_wall_tails_use_registered_nearest_rank_and_show_empty_cells(self):
        result = wall_statistics(list(range(1, 33)))
        self.assertEqual(result, {"completed": 32, "p50": 16, "p95": 31, "p99": 32, "max": 32})
        self.assertEqual(wall_statistics([]), {"completed": 0, "p50": None, "p95": None, "p99": None, "max": None})
        for values in ([True], [-1], [float("nan")]):
            with self.assertRaises(ContractError):
                wall_statistics(values)
