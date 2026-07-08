from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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

    def test_choice_label_uses_request_move_names_and_dense_switch_slots(self) -> None:
        module = _load_module()
        team = [
            SimpleNamespace(species="Milotic", condition="100/100", active=True),
            SimpleNamespace(species="Blissey", condition="100/100", active=False),
            SimpleNamespace(species="Skarmory", condition="100/100", active=False),
            SimpleNamespace(species="Flygon", condition="100/100", active=False),
            SimpleNamespace(species="Gengar", condition="0 fnt", active=False),
            SimpleNamespace(species="Snorlax", condition="100/100", active=False),
        ]
        state = SimpleNamespace(
            self_team=team,
            request={
                "active": [
                    {
                        "moves": [
                            {"id": "surf", "move": "Surf"},
                            {"id": "recover", "move": "Recover"},
                        ]
                    }
                ]
            },
        )

        self.assertEqual(module._choice_label(state, 0), "move:Surf")
        self.assertEqual(module._choice_label(state, 4), "switch:Blissey")
        self.assertEqual(module._choice_label(state, 5), "switch:Skarmory")
        self.assertEqual(module._choice_label(state, 7), "switch:Gengar")

    def test_capture_builder_normalizes_turn_merged_states_and_decodes_teacher_choice(self) -> None:
        module = _load_module()
        state = SimpleNamespace(
            legal_action_mask=(True, False, False, False, False, False, False, False, False),
            self_team=[SimpleNamespace(species="Milotic", condition="100/100", active=True)],
            request={"active": [{"moves": [{"id": "surf", "move": "Surf"}]}]},
            self_active=SimpleNamespace(species="Milotic", condition="100/100"),
            opponent_active=SimpleNamespace(species="Azumarill", condition="100/100"),
            turn_number=1,
            weather=None,
            self_side_conditions=(),
            opponent_side_conditions=(),
        )
        decision = SimpleNamespace(room="battle-1", protocol_lines=("|turn|1",), choice="move surf")
        game = SimpleNamespace(decisions=[decision])
        normalize_kwargs = {}

        def fake_normalize(*args, **kwargs):
            normalize_kwargs.update(kwargs)
            return state

        with (
            patch.object(module, "parse_capture_transcript", return_value=[game]),
            patch.object(module, "parse_showdown_replay", return_value=object()),
            patch.object(module, "normalize_for_player", side_effect=fake_normalize),
            patch.object(module, "action_index_from_choice_string", return_value=0),
        ):
            records, stats = module._build_states_from_captures(
                [("/tmp/capA.jsonl", "FoulPlayA")],
                set_source=None,
                max_states=None,
            )

        self.assertTrue(normalize_kwargs["include_turn_merged"])
        self.assertEqual(stats["states"], 1)
        self.assertNotIn("illegal_teacher_choices", stats)
        self.assertEqual(records[0]["teacher_action_label"], "move:Surf")
        self.assertIs(records[0]["state"], state)

    def test_metric_primitives_have_known_values(self) -> None:
        module = _load_module()

        self.assertEqual(module._js_divergence({0: 1.0}, {0: 1.0}), 0.0)
        self.assertEqual(module._js_divergence({0: 1.0}, {1: 1.0}), 1.0)
        self.assertEqual(module._spearman_rank_correlation({0: 1, 1: 2}, {0: 1, 1: 2}), 1.0)
        self.assertEqual(module._spearman_rank_correlation({0: 1, 1: 2}, {0: 2, 1: 1}), -1.0)

    def test_provenance_warnings_flag_schema_and_belief_mismatch(self) -> None:
        module = _load_module()
        source = SimpleNamespace(metadata=SimpleNamespace(source_hash="hash-live"))

        warnings = module._provenance_warnings(
            [
                {
                    "label": "a",
                    "observation_schema_version": "pokezero.observation.v2.1",
                    "belief_set_source_hash": "hash-live",
                },
                {
                    "label": "b",
                    "observation_schema_version": "pokezero.observation.v2.2",
                    "belief_set_source_hash": None,
                },
            ],
            set_source=source,
        )

        self.assertTrue(any("schemas differ" in warning for warning in warnings))
        self.assertTrue(any("belief_set_source_hash values differ" in warning for warning in warnings))
        self.assertTrue(any("active source hash" in warning for warning in warnings))
