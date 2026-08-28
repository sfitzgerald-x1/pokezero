"""Fail-closed coverage for the Stage-1 oracle-label corpus fence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "expert_iteration_label_corpus.py"
spec = importlib.util.spec_from_file_location("expert_iteration_label_corpus", SCRIPT)
corpus_tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = corpus_tool
spec.loader.exec_module(corpus_tool)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache(path: Path, *, targets: list[float], seeds: list[int], turns: list[int]) -> str:
    path.mkdir()
    write_json(path / "metadata.json", {
        "schema_version": corpus_tool.TRAINING_CACHE_SCHEMA,
        "example_count": len(targets),
    })
    numpy.save(path / "returns.npy", numpy.asarray(targets, dtype=numpy.float32))
    numpy.save(path / "seeds.npy", numpy.asarray(seeds, dtype=numpy.int64))
    numpy.save(path / "turn_indices.npy", numpy.asarray(turns, dtype=numpy.int32))
    return corpus_tool.training_cache_tree_sha256(path)


def fixture(tmp_path: Path) -> dict[str, Path]:
    contract = {
        "experiment_id": "oracle-test",
        "status": "REGISTERED-NOT-RUN",
        "runtime": {
            "checkpoint_sha256": "a" * 64,
            "public_source_commit": "b" * 40,
            "showdown_commit": "c" * 40,
        },
        "estimator": {
            "subject": "p1", "opponent": "raw-transformer-policy", "top_k_legal_priors": 3,
            "rollouts_per_action": 16, "paired_continuation_policy_rng_across_actions": True,
            "source_max_decision_rounds": 1024, "continuation_max_decision_rounds": 4096,
            "leaf_estimator": "vhprobe-policy-continuation-v1", "uniform_leaves_excluded": True,
            "capped_source_or_continuation_policy": "fail-shard-never-label",
        },
        "sample": {"analysis_unit": "source_game_seed", "seed_pairs": 2},
    }
    contract_path = tmp_path / "contract.json"
    write_json(contract_path, contract)
    bank = {
        "schema": corpus_tool.ROOT_BANK_SCHEMA,
        "experiment_id": "oracle-test",
        "provenance": {
            "analysis_unit": "source_game_seed", "checkpoint_sha256": "a" * 64,
            "public_source_commit": "b" * 40, "showdown_commit": "c" * 40,
            "top_k": 3, "rollouts_per_action": 16, "source_max_decision_rounds": 1024,
            "leaf_estimator": "vhprobe-policy-continuation-v1", "uniform_leaves_excluded": True,
        },
        "pairs": [{
            "seed": 4,
            "oracle_decisions": [{
                "decision_index": 9, "candidate_count": 2, "selected_action": 7,
                "selected_policy_continuation_value": 0.75,
                "opponent_actions_fixed_before_selection": {"p2": 3},
                "candidate_scores": [
                    {"action": 7, "policy_continuation_value": 0.75, "rollouts_completed": 16, "terminal_shortcut": False},
                    {"action": 8, "policy_continuation_value": 0.0, "rollouts_completed": 0, "terminal_shortcut": True},
                ],
            }],
        }, {
            "seed": 5,
            "oracle_decisions": [{
                "decision_index": 2, "candidate_count": 1, "selected_action": 1,
                "selected_policy_continuation_value": 0.5,
                "opponent_actions_fixed_before_selection": {},
                "candidate_scores": [
                    {"action": 1, "policy_continuation_value": 0.5, "rollouts_completed": 16, "terminal_shortcut": False},
                ],
            }],
        }],
    }
    bank_path = tmp_path / "bank.json"
    write_json(bank_path, bank)
    split_path = tmp_path / "splits.json"
    write_json(split_path, {
        "schema": corpus_tool.SPLIT_SCHEMA,
        "bank": {"sha256": digest(bank_path)},
        "registered_rule": {"heldout": "source_game_seed % 4 == 0", "train": "source_game_seed % 4 != 0"},
        "heldout_source_game_seeds": [4], "train_source_game_seeds": [5],
    })
    train = tmp_path / "train-cache"
    heldout = tmp_path / "heldout-cache"
    train_digest = cache(train, targets=[0.5], seeds=[5], turns=[2])
    heldout_digest = cache(heldout, targets=[0.75], seeds=[4], turns=[9])
    corpus = {
        "schema": corpus_tool.CORPUS_SCHEMA,
        "bank": {"sha256": digest(bank_path)},
        "runtime": contract["runtime"],
        "replay": {
            "subject": "p1", "opponent_policy": "raw-transformer-policy", "sampling_temperature": 1.0,
            "deterministic": False, "source_max_decision_rounds": 1024,
            "branch_rule": corpus_tool.BRANCH_RULE,
        },
        "training_caches": {
            "train": {"tree_sha256": train_digest, "example_count": 1},
            "heldout": {"tree_sha256": heldout_digest, "example_count": 1},
        },
        "records": [
            {"split": "heldout", "source_game_seed": 4, "decision_index": 9, "candidate_action": 7,
             "selected_action": 7, "fixed_opponent_actions": {"p2": 3}, "target": 0.75,
             "terminal_shortcut": False, "source_state_sha256": "d" * 64,
             "successor_observation_sha256": "e" * 64, "cache_index": 0},
            {"split": "heldout", "source_game_seed": 4, "decision_index": 9, "candidate_action": 8,
             "selected_action": 7, "fixed_opponent_actions": {"p2": 3}, "target": 0.0,
             "terminal_shortcut": True, "source_state_sha256": None,
             "successor_observation_sha256": None, "cache_index": None},
            {"split": "train", "source_game_seed": 5, "decision_index": 2, "candidate_action": 1,
             "selected_action": 1, "fixed_opponent_actions": {}, "target": 0.5,
             "terminal_shortcut": False, "source_state_sha256": "f" * 64,
             "successor_observation_sha256": "0" * 64, "cache_index": 0},
        ],
    }
    corpus_path = tmp_path / "corpus.json"
    write_json(corpus_path, corpus)
    return {"bank": bank_path, "contract": contract_path, "splits": split_path, "corpus": corpus_path, "train": train, "heldout": heldout}


def validate(paths: dict[str, Path]) -> dict[str, int | str]:
    return corpus_tool.validate(
        bank_path=paths["bank"], contract_path=paths["contract"], split_path=paths["splits"],
        corpus_path=paths["corpus"], train_cache_path=paths["train"], heldout_cache_path=paths["heldout"],
    )


class ExpertIterationLabelCorpusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_validates_complete_replay_bound_candidate_coverage(self) -> None:
        paths = fixture(self.root)
        summary = validate(paths)
        self.assertEqual(summary["bank_candidates"], 3)
        self.assertEqual(summary["terminal_shortcuts"], 1)
        self.assertEqual(summary["train_examples"], summary["heldout_examples"])
        self.assertEqual(summary["train_examples"], 1)

    def test_refuses_label_changed_after_cache_is_written(self) -> None:
        paths = fixture(self.root)
        corpus = json.loads(paths["corpus"].read_text(encoding="utf-8"))
        corpus["records"][2]["target"] = 0.25
        write_json(paths["corpus"], corpus)
        with self.assertRaisesRegex(corpus_tool.CorpusError, "target differs from bank label"):
            validate(paths)

    def test_refuses_cache_return_that_does_not_match_oracle_label(self) -> None:
        paths = fixture(self.root)
        numpy.save(paths["train"] / "returns.npy", numpy.asarray([0.25], dtype=numpy.float32))
        with self.assertRaisesRegex(corpus_tool.CorpusError, "tree digest"):
            validate(paths)

    def test_refuses_seed_only_row_without_replayed_successor_hashes(self) -> None:
        paths = fixture(self.root)
        corpus = json.loads(paths["corpus"].read_text(encoding="utf-8"))
        corpus["records"][2]["source_state_sha256"] = None
        write_json(paths["corpus"], corpus)
        with self.assertRaisesRegex(corpus_tool.CorpusError, "source_state_sha256"):
            validate(paths)

    def test_refuses_partial_candidate_coverage(self) -> None:
        paths = fixture(self.root)
        corpus = json.loads(paths["corpus"].read_text(encoding="utf-8"))
        corpus["records"].pop(1)
        write_json(paths["corpus"], corpus)
        with self.assertRaisesRegex(corpus_tool.CorpusError, "exact bank candidate set"):
            validate(paths)


if __name__ == "__main__":
    unittest.main()
