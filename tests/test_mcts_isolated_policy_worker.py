"""Regression coverage for the source-isolated worker lifecycle protocol."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
WORKER = runpy.run_path(str(REPO / "scripts" / "mcts_isolated_policy_worker.py"))
reset_policy = WORKER["_reset_policy"]
serve = WORKER["_serve"]
read_frame = WORKER["read_frame"]
write_frame = WORKER["write_frame"]
worker_error = WORKER["WorkerError"]


class IsolatedPolicyWorkerResetTest(unittest.TestCase):
    def test_reset_refuses_policy_without_lifecycle(self) -> None:
        with self.assertRaises(worker_error):
            reset_policy(object())

    def test_reset_forwards_to_stateful_policy(self) -> None:
        class StatefulPolicy:
            calls = 0

            def reset(self) -> None:
                self.calls += 1

        policy = StatefulPolicy()
        reset_policy(policy)
        self.assertEqual(policy.calls, 1)

    def test_reset_frame_invokes_lifecycle_before_acknowledging(self) -> None:
        class StatefulPolicy:
            def __init__(self) -> None:
                self.calls = 0

            def reset(self) -> None:
                self.calls += 1

        policy = StatefulPolicy()
        inbound = BytesIO()
        write_frame(inbound, {"type": "start"})
        write_frame(inbound, {"type": "reset"})
        write_frame(inbound, {"type": "close"})
        inbound.seek(0)
        outbound = BytesIO()
        fake_stdin = SimpleNamespace(buffer=inbound)
        fake_stdout = SimpleNamespace(buffer=outbound)
        with (
            patch.object(WORKER["sys"], "stdin", fake_stdin),
            patch.object(WORKER["sys"], "stdout", fake_stdout),
            patch.dict(
                serve.__globals__,
                {"_worker_start": lambda _start: (policy, object(), {}, object)},
            ),
        ):
            self.assertEqual(serve(), 0)

        outbound.seek(0)
        self.assertEqual(read_frame(outbound), {"type": "hello", "receipt": {}})
        self.assertEqual(read_frame(outbound), {"type": "reset"})
        self.assertEqual(read_frame(outbound), {"type": "close"})
        self.assertEqual(policy.calls, 1)


if __name__ == "__main__":
    unittest.main()
