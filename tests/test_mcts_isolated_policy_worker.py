"""Regression coverage for the source-isolated worker lifecycle protocol."""

from __future__ import annotations

import runpy
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKER = runpy.run_path(str(REPO / "scripts" / "mcts_isolated_policy_worker.py"))
reset_policy = WORKER["_reset_policy"]


class IsolatedPolicyWorkerResetTest(unittest.TestCase):
    def test_reset_accepts_stateless_engine_policy(self) -> None:
        class StatelessPolicy:
            pass

        reset_policy(StatelessPolicy())

    def test_reset_forwards_to_stateful_policy(self) -> None:
        class StatefulPolicy:
            calls = 0

            def reset(self) -> None:
                self.calls += 1

        policy = StatefulPolicy()
        reset_policy(policy)
        self.assertEqual(policy.calls, 1)


if __name__ == "__main__":
    unittest.main()
