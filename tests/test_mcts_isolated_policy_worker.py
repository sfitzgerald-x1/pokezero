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
    def test_reset_reconstructs_policy_without_lifecycle_and_retains_telemetry(self) -> None:
        class HistoricalPolicy:
            def __init__(self, stats: object) -> None:
                self.stats = stats

        stats = object()
        policy = HistoricalPolicy(stats)
        rebuilt = HistoricalPolicy(object())
        result, strategy = reset_policy(policy, recreate=lambda: rebuilt)
        self.assertIs(result, rebuilt)
        self.assertIs(result.stats, stats)
        self.assertEqual(strategy, "fresh_source_policy")

    def test_reset_forwards_to_stateful_policy(self) -> None:
        class StatefulPolicy:
            calls = 0

            def reset(self) -> None:
                self.calls += 1

        policy = StatefulPolicy()
        result, strategy = reset_policy(policy, recreate=lambda: self.fail("must not rebuild"))
        self.assertIs(result, policy)
        self.assertEqual(policy.calls, 1)
        self.assertEqual(strategy, "policy_method")

    def test_reset_refuses_a_historical_policy_when_reconstruction_loses_telemetry(self) -> None:
        class HistoricalPolicy:
            stats = object()

        with self.assertRaisesRegex(worker_error, "cumulative telemetry"):
            reset_policy(HistoricalPolicy(), recreate=lambda: object())

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
                {
                    "_worker_start": lambda _start: (
                        policy,
                        lambda: self.fail("must not rebuild"),
                        object(),
                        {},
                        object(),
                    )
                },
            ),
        ):
            self.assertEqual(serve(), 0)

        outbound.seek(0)
        self.assertEqual(read_frame(outbound), {"type": "hello", "receipt": {}})
        self.assertEqual(read_frame(outbound), {"type": "reset", "strategy": "policy_method"})
        self.assertEqual(read_frame(outbound), {"type": "close"})
        self.assertEqual(policy.calls, 1)

    def test_reset_frame_replaces_historical_policy_before_the_next_game(self) -> None:
        class HistoricalPolicy:
            def __init__(self, stats: object) -> None:
                self.stats = stats

        class RebuiltPolicy(HistoricalPolicy):
            def __init__(self, stats: object) -> None:
                super().__init__(stats)
                self.reset_calls = 0

            def reset(self) -> None:
                self.reset_calls += 1

        stats = object()
        original = HistoricalPolicy(stats)
        rebuilt = RebuiltPolicy(object())
        factory_calls = 0

        def recreate() -> RebuiltPolicy:
            nonlocal factory_calls
            factory_calls += 1
            return rebuilt

        inbound = BytesIO()
        write_frame(inbound, {"type": "start"})
        write_frame(inbound, {"type": "reset"})
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
                {
                    "_worker_start": lambda _start: (
                        original,
                        recreate,
                        object(),
                        {},
                        object(),
                    )
                },
            ),
        ):
            self.assertEqual(serve(), 0)

        outbound.seek(0)
        self.assertEqual(read_frame(outbound), {"type": "hello", "receipt": {}})
        self.assertEqual(
            read_frame(outbound), {"type": "reset", "strategy": "fresh_source_policy"}
        )
        self.assertEqual(read_frame(outbound), {"type": "reset", "strategy": "policy_method"})
        self.assertEqual(read_frame(outbound), {"type": "close"})
        self.assertEqual(factory_calls, 1)
        self.assertIs(rebuilt.stats, stats)
        self.assertEqual(rebuilt.reset_calls, 1)

    def test_reset_frame_refuses_historical_policy_without_telemetry(self) -> None:
        class HistoricalPolicy:
            stats = None

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
                {
                    "_worker_start": lambda _start: (
                        HistoricalPolicy(),
                        lambda: self.fail("must not rebuild without telemetry"),
                        object(),
                        {},
                        object(),
                    )
                },
            ),
        ):
            self.assertEqual(serve(), 0)

        outbound.seek(0)
        self.assertEqual(read_frame(outbound), {"type": "hello", "receipt": {}})
        self.assertEqual(
            read_frame(outbound)["type"], "error", "missing telemetry must not acknowledge reset"
        )


if __name__ == "__main__":
    unittest.main()
