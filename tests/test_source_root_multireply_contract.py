"""Focused regressions for source-root multireply registration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("source_root_multireply_contract_test", ROOT / "scripts" / "source_root_multireply_contract.py")
assert SPEC is not None and SPEC.loader is not None
CONTRACT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTRACT
SPEC.loader.exec_module(CONTRACT)


class MultireplyContractTest(unittest.TestCase):
    def test_deduplicates_raw_model_choice_without_losing_the_alias(self) -> None:
        actions, aliases = CONTRACT.register_candidate_actions(
            {"raw_policy": 3, "model_leaf": 3, "rollout_leaf": 8}
        )
        self.assertEqual(actions, {"raw_policy": 3, "rollout_leaf": 8})
        self.assertEqual(aliases, {"raw_policy": "raw_policy", "model_leaf": "raw_policy", "rollout_leaf": "rollout_leaf"})

    def test_preserves_three_distinct_candidates(self) -> None:
        actions, aliases = CONTRACT.register_candidate_actions(
            {"raw_policy": 5, "model_leaf": 0, "rollout_leaf": 3}
        )
        self.assertEqual(actions, {"raw_policy": 5, "model_leaf": 0, "rollout_leaf": 3})
        self.assertEqual(aliases, {name: name for name in CONTRACT.ARM_LABELS})

    def test_reply_selector_schedule_is_stable_and_complete(self) -> None:
        first = [CONTRACT.opponent_reply_selector_seed("decision-1", sample) for sample in range(8)]
        self.assertEqual(first, [CONTRACT.opponent_reply_selector_seed("decision-1", sample) for sample in range(8)])
        self.assertEqual(len(set(first)), 8)
        self.assertNotEqual(first[0], CONTRACT.opponent_reply_selector_seed("decision-2", 0))

    def test_refuses_a_fake_one_action_comparison(self) -> None:
        with self.assertRaisesRegex(CONTRACT.MultireplyContractError, "at least two"):
            CONTRACT.register_candidate_actions({"raw_policy": 3, "model_leaf": 3, "rollout_leaf": 3})


if __name__ == "__main__":
    unittest.main()
