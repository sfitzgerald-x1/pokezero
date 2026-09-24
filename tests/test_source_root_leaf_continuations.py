"""Focused contract tests for the source-root leaf continuation runner."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from argparse import Namespace
from dataclasses import dataclass
from contextlib import redirect_stdout
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "source_root_leaf_continuations_test", ROOT / "scripts" / "run_source_root_leaf_continuations.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


@dataclass(frozen=True)
class _Record:
    seed: int = 1
    battle_id: str = "battle-1"
    turn_index: int = 2
    acting_player: str = "p1"
    decision_id: str = "decision-1"
    recorded_action_index: int = 3
    current_legal_action_mask: tuple[bool, ...] = (True, True, True, True)

    def to_dict(self) -> dict[str, object]:
        return {"seed": self.seed, "battle_id": self.battle_id, "turn_index": self.turn_index}


RECORD = _Record()
LEAF = {
    "arms": {
        "model_control_a": {"selection": {"root_action": "move 1"}},
        "rollout_leaf": {"selection": {"root_action": "move 2"}},
    }
}
EXPECTED_ACTIONS = {"model_leaf": 1, "rollout_leaf": 2}


def _payload(*, leaked_opponent: bool = False, capped: bool = False, source_hash: str | None = None) -> dict[str, object]:
    outcomes = []
    for label, action in (("model_leaf", 1), ("rollout_leaf", 2)):
        outcomes.append(
            {
                "action_label": label,
                "action_index": action,
                "continuation": {
                    "decision_round_count": 5,
                    "terminal_after_fixed_joint_step": False,
                    "terminal": {"winner": "p1", "turn_count": 9, "capped": capped},
                    "initial_max_continuation_decision_rounds": RUNNER.INITIAL_MAX_CONTINUATION_DECISION_ROUNDS,
                    "effective_max_continuation_decision_rounds": RUNNER.INITIAL_MAX_CONTINUATION_DECISION_ROUNDS,
                    "cap_retry": False,
                },
            }
        )
    grid: dict[str, object] = {
        "schema_version": "pokezero.sealed-root-action-grid.v2",
        "source_battle_id": RECORD.battle_id,
        "source_seed": RECORD.seed,
        "source_decision_round": RECORD.turn_index,
        "subject_player": "p1",
        "opponent_player": "p2",
        "opponent_action_held_fixed": True,
        "actions": [
            {"action_label": "model_leaf", "action_index": 1},
            {"action_label": "rollout_leaf", "action_index": 2},
        ],
        "search_evidence": {
            "model_leaf_choice": "move 1",
            "rollout_leaf_choice": "move 2",
            "selection_changed": True,
            "leaf_only_intervention": True,
        },
        "continuation_targets": [
            {
                "target": target,
                "trials": [
                    {"continuation_rng_seed": seed, "outcomes": outcomes}
                    for seed in RUNNER.CONTINUATION_RNG_SEEDS
                ],
            }
            for target in RUNNER.CONTINUATION_TARGETS
        ],
    }
    if leaked_opponent:
        grid["opponent_action"] = 3
    return {
        "schema_version": RUNNER.SCHEMA_VERSION,
        "state": "COMPLETE",
        "source": {"seed": 1, "seat": "p1", "turn_index": 2},
        "manifest_sha256": "m" * 64,
        "source_record_sha256": source_hash or RUNNER._sha256(RECORD.to_dict()),
        "leaf_complete_sha256": RUNNER._sha256(LEAF),
        "grid": grid,
    }


class ContinuationContractTest(unittest.TestCase):
    def test_completed_root_accepts_complete_paired_grid(self) -> None:
        RUNNER._validate_completed_root(
            _payload(),
            root=RUNNER.SourceRoot(1, "p1", 2),
            record=RECORD,
            leaf_payload=LEAF,
            expected_actions=EXPECTED_ACTIONS,
            manifest_sha256="m" * 64,
        )

    def test_completed_root_refuses_opponent_action_leak(self) -> None:
        with self.assertRaisesRegex(RUNNER.ContinuationError, "leaks"):
            RUNNER._validate_completed_root(
                _payload(leaked_opponent=True),
                root=RUNNER.SourceRoot(1, "p1", 2),
                record=RECORD,
                leaf_payload=LEAF,
                expected_actions=EXPECTED_ACTIONS,
                manifest_sha256="m" * 64,
            )

    def test_completed_root_refuses_capped_suffix(self) -> None:
        with self.assertRaisesRegex(RUNNER.ContinuationError, "incomplete or capped"):
            RUNNER._validate_completed_root(
                _payload(capped=True),
                root=RUNNER.SourceRoot(1, "p1", 2),
                record=RECORD,
                leaf_payload=LEAF,
                expected_actions=EXPECTED_ACTIONS,
                manifest_sha256="m" * 64,
            )

    def test_completed_root_refuses_source_record_drift(self) -> None:
        with self.assertRaisesRegex(RUNNER.ContinuationError, "source record drifted"):
            RUNNER._validate_completed_root(
                _payload(source_hash="x" * 64),
                root=RUNNER.SourceRoot(1, "p1", 2),
                record=RECORD,
                leaf_payload=LEAF,
                expected_actions=EXPECTED_ACTIONS,
                manifest_sha256="m" * 64,
            )

    def test_completed_root_refuses_missing_terminal_winner(self) -> None:
        payload = _payload()
        outcome = payload["grid"]["continuation_targets"][0]["trials"][0]["outcomes"][0]
        del outcome["continuation"]["terminal"]["winner"]
        with self.assertRaisesRegex(RUNNER.ContinuationError, "winner is invalid"):
            RUNNER._validate_completed_root(
                payload,
                root=RUNNER.SourceRoot(1, "p1", 2),
                record=RECORD,
                leaf_payload=LEAF,
                expected_actions=EXPECTED_ACTIONS,
                manifest_sha256="m" * 64,
            )

    def test_multireply_readback_requires_every_registered_reply_sample(self) -> None:
        grid = _payload()["grid"]
        grid["actions"] = [
            {"action_label": "raw_policy", "action_index": 1},
            {"action_label": "rollout_leaf", "action_index": 2},
        ]
        grid["search_evidence"] = {
            "model_leaf_choice": "move 1",
            "rollout_leaf_choice": "move 2",
            "raw_policy_action_index": 1,
            "selection_changed": True,
            "leaf_only_intervention": True,
        }
        grid["continuation_targets"] = [grid["continuation_targets"][0]]
        for trial in grid["continuation_targets"][0]["trials"]:
            for outcome, label in zip(trial["outcomes"], ("raw_policy", "rollout_leaf"), strict=True):
                outcome["action_label"] = label
        projection = {
            "candidate_aliases": {"raw_policy": "raw_policy", "model_leaf": "raw_policy", "rollout_leaf": "rollout_leaf"},
            "opponent_reply_samples": list(range(RUNNER.OPPONENT_REPLY_SAMPLE_COUNT)),
            "opponent_selector": {"action_hidden": True},
        }
        study = {
            "schema_version": RUNNER.MULTIREPLY_SCHEMA_VERSION,
            "candidate_actions": {"raw_policy": 1, "rollout_leaf": 2},
            **projection,
            "reply_samples": [
                {"opponent_reply_sample": sample, "grid": grid}
                for sample in range(RUNNER.OPPONENT_REPLY_SAMPLE_COUNT)
            ],
        }
        RUNNER._validate_multireply_payload(
            study,
            root=RUNNER.SourceRoot(1, "p1", 2),
            record=RECORD,
            expected_actions={"raw_policy": 1, "rollout_leaf": 2},
            expected_projection=projection,
            expected_search_evidence=grid["search_evidence"],
        )
        study["reply_samples"].pop()
        with self.assertRaisesRegex(RUNNER.ContinuationError, "reply schedule"):
            RUNNER._validate_multireply_payload(
                study,
                root=RUNNER.SourceRoot(1, "p1", 2),
                record=RECORD,
                expected_actions={"raw_policy": 1, "rollout_leaf": 2},
                expected_projection=projection,
                expected_search_evidence=grid["search_evidence"],
            )

    def test_multireply_readback_refuses_an_empty_cap_retry(self) -> None:
        grid = _payload()["grid"]
        grid["actions"] = [{"action_label": "raw_policy", "action_index": 1}, {"action_label": "rollout_leaf", "action_index": 2}]
        grid["search_evidence"] = {"model_leaf_choice": "move 1", "rollout_leaf_choice": "move 2", "raw_policy_action_index": 1, "selection_changed": True, "leaf_only_intervention": True}
        grid["continuation_targets"] = [grid["continuation_targets"][0]]
        for trial in grid["continuation_targets"][0]["trials"]:
            for outcome, label in zip(trial["outcomes"], ("raw_policy", "rollout_leaf"), strict=True):
                outcome["action_label"] = label
        grid["continuation_targets"][0]["trials"][0]["outcomes"][0]["continuation"]["cap_retry"] = True
        projection = {"candidate_aliases": {"raw_policy": "raw_policy", "model_leaf": "raw_policy", "rollout_leaf": "rollout_leaf"}, "opponent_reply_samples": list(range(RUNNER.OPPONENT_REPLY_SAMPLE_COUNT)), "opponent_selector": {"action_hidden": True}}
        study = {"schema_version": RUNNER.MULTIREPLY_SCHEMA_VERSION, "candidate_actions": {"raw_policy": 1, "rollout_leaf": 2}, **projection, "reply_samples": [{"opponent_reply_sample": sample, "grid": grid} for sample in range(RUNNER.OPPONENT_REPLY_SAMPLE_COUNT)]}
        with self.assertRaisesRegex(RUNNER.ContinuationError, "effective ceiling"):
            RUNNER._validate_multireply_payload(study, root=RUNNER.SourceRoot(1, "p1", 2), record=RECORD, expected_actions={"raw_policy": 1, "rollout_leaf": 2}, expected_projection=projection, expected_search_evidence=grid["search_evidence"])

    def test_raw_policy_anchor_uses_ledger_argmax_not_recorded_mcts_action(self) -> None:
        root = RUNNER.SourceRoot(RECORD.seed, RECORD.acting_player, RECORD.turn_index)
        wrapper = {
            "candidate_provenance_sha256": "c" * 64,
            "raw_provenance_sha256": "r" * 64,
        }
        ledger = {
            "schema_version": "pokezero.mcts-guided-vs-raw-branch-prior-ledger.v5",
            "seed": RECORD.seed,
            "candidate_seat": RECORD.acting_player,
            **wrapper,
            "public_decision": {
                "decision_id": RECORD.decision_id,
                "battle_id": RECORD.battle_id,
                "acting_player": RECORD.acting_player,
                "turn_index": RECORD.turn_index,
                "recorded_action_index": RECORD.recorded_action_index,
            },
            "selection": {
                "model_argmax": 0,
                "search_argmax": RECORD.recorded_action_index,
                "model_override": True,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = (
                Path(temporary)
                / "seeds"
                / f"seed-{RECORD.seed}"
                / "branch-prior-fallback-ledgers"
                / f"seed-{RECORD.seed}-{RECORD.acting_player}"
            )
            directory.mkdir(parents=True)
            (directory / f"turn-{RECORD.turn_index:03d}-{RECORD.decision_id}.json").write_text(
                json.dumps(ledger), encoding="utf-8"
            )
            anchor = RUNNER._raw_policy_anchor(
                source_root=Path(temporary), root=root, record=RECORD, wrapper=wrapper
            )
        self.assertEqual(anchor.action_index, 0)
        self.assertNotEqual(anchor.action_index, RECORD.recorded_action_index)
        self.assertEqual(anchor.selection_ledger_sha256, RUNNER._sha256(ledger))

    def test_multireply_registers_the_ledger_bound_raw_action(self) -> None:
        plan = RUNNER._RootPlan(
            root=RUNNER.SourceRoot(RECORD.seed, RECORD.acting_player, RECORD.turn_index),
            model_choice="move 2",
            rollout_choice="move 3",
            model_action=1,
            rollout_action=2,
            raw_policy_action=0,
            raw_policy_selection_sha256="a" * 64,
            histories={},
            snapshot=None,
            opponent_seat="p2",
            opponent_action=0,
            opponent_observation=None,
            source_battle_id=RECORD.battle_id,
        )
        actions, projection = RUNNER._multireply_actions(plan=plan, record=RECORD)
        self.assertEqual(actions, {"raw_policy": 0, "model_leaf": 1, "rollout_leaf": 2})
        self.assertEqual(projection["raw_policy_selection_sha256"], "a" * 64)

    def test_multireply_refuses_a_missing_raw_policy_anchor(self) -> None:
        plan = RUNNER._RootPlan(
            root=RUNNER.SourceRoot(RECORD.seed, RECORD.acting_player, RECORD.turn_index),
            model_choice="move 2",
            rollout_choice="move 3",
            model_action=1,
            rollout_action=2,
            raw_policy_action=None,
            raw_policy_selection_sha256=None,
            histories={},
            snapshot=None,
            opponent_seat="p2",
            opponent_action=0,
            opponent_observation=None,
            source_battle_id=RECORD.battle_id,
        )
        with self.assertRaisesRegex(RUNNER.ContinuationError, "anchor is absent"):
            RUNNER._multireply_actions(plan=plan, record=RECORD)

    def test_source_provenance_requires_image_baked_commit(self) -> None:
        expected = "a" * 40
        with patch.object(RUNNER, "public_repo_commit", return_value=expected):
            provenance = RUNNER._source_code_provenance(expected_commit=expected)
        self.assertEqual(provenance["commit"], expected)
        with patch.object(RUNNER, "public_repo_commit", return_value="b" * 40):
            with self.assertRaisesRegex(RUNNER.ContinuationError, "image-baked source commit"):
                RUNNER._source_code_provenance(expected_commit=expected)

    def test_source_histories_keep_each_seat_order_and_append_current(self) -> None:
        replay = type(
            "Replay",
            (),
            {
                "replay_observations": {
                    1: {"p1": "p1-1", "p2": "p2-1"},
                    0: {"p1": "p1-0"},
                }
            },
        )()
        self.assertEqual(
            RUNNER._source_histories(replay, {"p1": "p1-current", "p2": "p2-current"}),
            {"p1": ("p1-0", "p1-1", "p1-current"), "p2": ("p2-1", "p2-current")},
        )

    def test_terminal_writer_never_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "terminal.json"
            RUNNER._write_create_only_json(path, {"one": 1})
            with self.assertRaisesRegex(RUNNER.ContinuationError, "refusing to replace"):
                RUNNER._write_create_only_json(path, {"two": 2})
            self.assertEqual(json.loads(path.read_text()), {"one": 1})

    def test_terminal_writer_reuses_only_identical_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "terminal.json"
            RUNNER._write_create_only_json(path, {"one": 1})
            RUNNER._write_or_require_identical_json(path, {"one": 1})
            with self.assertRaisesRegex(RUNNER.ContinuationError, "differs"):
                RUNNER._write_or_require_identical_json(path, {"two": 2})

    def test_prepare_allows_only_a_lone_pass_for_resumed_final_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"contract": "current"}
            (root / "MANIFEST.json").write_text(json.dumps(manifest))
            (root / "PASS.json").write_text("{}")
            RUNNER._prepare(
                root,
                manifest,
                resume=True,
                require_existing=False,
                allow_complete_pass=True,
            )
            self.assertFalse((root / "RUNNING.json").exists())

    def test_prepare_refuses_terminal_pass_without_matching_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "PASS.json").write_text("{}")
            with self.assertRaisesRegex(RUNNER.ContinuationError, "manifest differs"):
                RUNNER._prepare(
                    root,
                    {"contract": "current"},
                    resume=True,
                    require_existing=False,
                    allow_complete_pass=True,
                )

    def test_prepare_refuses_nonpass_or_nonfinalizer_terminal_reentry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "NONPASS.json").write_text("{}")
            with self.assertRaisesRegex(RUNNER.ContinuationError, "terminal"):
                RUNNER._prepare(
                    root,
                    {"contract": "current"},
                    resume=True,
                    require_existing=False,
                    allow_complete_pass=True,
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "PASS.json").write_text("{}")
            with self.assertRaisesRegex(RUNNER.ContinuationError, "terminal"):
                RUNNER._prepare(
                    root,
                    {"contract": "current"},
                    resume=True,
                    require_existing=False,
                    allow_complete_pass=False,
                )

    def test_main_labels_preparation_as_nonterminal(self) -> None:
        stream = io.StringIO()
        with patch.object(RUNNER, "_parse_args", return_value=Namespace(out_root="/tmp")), patch.object(
            RUNNER, "_run", return_value={"state": "PREPARED", "root_count": 7}
        ), redirect_stdout(stream):
            self.assertEqual(RUNNER.main([]), 0)
        self.assertIn("WROTE SOURCE ROOT LEAF CONTINUATION PREPARED", stream.getvalue())
        self.assertNotIn("CONTINUATION PASS", stream.getvalue())

    def test_main_labels_terminal_pass_as_pass(self) -> None:
        stream = io.StringIO()
        with patch.object(RUNNER, "_parse_args", return_value=Namespace(out_root="/tmp")), patch.object(
            RUNNER, "_run", return_value={"state": "PASS", "root_count": 7}
        ), redirect_stdout(stream):
            self.assertEqual(RUNNER.main([]), 0)
        self.assertIn("WROTE SOURCE ROOT LEAF CONTINUATION PASS", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
