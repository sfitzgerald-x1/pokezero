from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "foulplay_choice_rank_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("foulplay_choice_rank_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FoulPlayChoiceRankProbeTest(unittest.TestCase):
    def test_pairwise_summary_reports_rank_and_top_action_changes(self) -> None:
        module = _load_module()
        states = [
            {
                "state_id": "s1",
                "context": {"turn": 4, "active": "Milotic", "opponent_active": "Azumarill"},
                "teacher_action_label": "move:Toxic",
                "checkpoints": {
                    "orig": {
                        "top_action": "move:Recover",
                        "teacher_rank": 2,
                        "ranked_actions": [
                            {"action_index": 0, "label": "move:Recover", "rank": 1, "probability": 0.7},
                            {"action_index": 1, "label": "move:Toxic", "rank": 2, "probability": 0.3},
                        ],
                    },
                    "pert": {
                        "top_action": "move:Toxic",
                        "teacher_rank": 1,
                        "ranked_actions": [
                            {"action_index": 1, "label": "move:Toxic", "rank": 1, "probability": 0.8},
                            {"action_index": 0, "label": "move:Recover", "rank": 2, "probability": 0.2},
                        ],
                    },
                },
            },
            {
                "state_id": "s2",
                "context": {"turn": 5, "active": "Milotic", "opponent_active": "Azumarill"},
                "teacher_action_label": "move:Surf",
                "checkpoints": {
                    "orig": {
                        "top_action": "move:Surf",
                        "teacher_rank": 1,
                        "ranked_actions": [
                            {"action_index": 2, "label": "move:Surf", "rank": 1, "probability": 0.6},
                            {"action_index": 0, "label": "move:Recover", "rank": 2, "probability": 0.4},
                        ],
                    },
                    "pert": {
                        "top_action": "move:Surf",
                        "teacher_rank": 1,
                        "ranked_actions": [
                            {"action_index": 2, "label": "move:Surf", "rank": 1, "probability": 0.55},
                            {"action_index": 0, "label": "move:Recover", "rank": 2, "probability": 0.45},
                        ],
                    },
                },
            },
        ]

        summary = module._pairwise_summary(states, [("orig", "pert")])

        self.assertEqual(summary[0]["top_action_disagreement_rate"], 0.5)
        self.assertEqual(summary[0]["top_action_disagreements"], 1)
        self.assertEqual(summary[0]["mean_teacher_rank_improvement"], 0.5)
        self.assertEqual(summary[0]["mean_abs_legal_rank_delta"], 0.5)
        self.assertEqual(summary[0]["changed_top_action_examples"][0]["state_id"], "s1")

    def test_specs_reject_duplicate_labels_and_unknown_pairs(self) -> None:
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "duplicate checkpoint labels"):
            module._checkpoint_specs(["/a.pt=same", "/b.pt=same"])

        with self.assertRaisesRegex(ValueError, "unknown checkpoint label"):
            module._pair_specs(["orig:missing"], {"orig"})

    def test_strip_state_objects_removes_unserializable_state_only(self) -> None:
        module = _load_module()
        sentinel = object()

        self.assertEqual(
            module._strip_state_objects([{"state": sentinel, "state_id": "s1", "x": 1}]),
            [{"state_id": "s1", "x": 1}],
        )
