"""Regression coverage for the source-isolated worker lifecycle protocol."""

from __future__ import annotations

from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.engine_search import BRANCH_PRIOR_FALLBACK_REASON_VALUES


REPO = Path(__file__).resolve().parents[1]
WORKER = runpy.run_path(str(REPO / "scripts" / "mcts_isolated_policy_worker.py"))
reset_policy = WORKER["_reset_policy"]
serve = WORKER["_serve"]
read_frame = WORKER["read_frame"]
write_frame = WORKER["write_frame"]
worker_error = WORKER["WorkerError"]
source_engine_config_payload = WORKER["source_engine_config_payload"]
stats_payload = WORKER["_stats_payload"]
write_error_diagnostic = WORKER["_write_error_diagnostic"]


class IsolatedPolicyWorkerResetTest(unittest.TestCase):
    def test_error_diagnostic_persists_only_public_boundary_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "error.json"
            annotations = SimpleNamespace(error_diagnostic_path=target)
            replay = SimpleNamespace(
                volatiles={"p1": {"perish0"}},
                public_active={"p1": SimpleNamespace(ident="p1a: Misdreavus")},
                public_events=(SimpleNamespace(raw_line="|-start|p1a: Misdreavus|perish0"),),
            )
            context = SimpleNamespace(
                battle_id="battle-1",
                decision_round_index=39,
                player_id="p1",
                requested_players=("p1", "p2"),
                public_materialization_state=SimpleNamespace(
                    replay=replay,
                    self_request={"active": [{"moves": []}]},
                ),
            )
            suffix = write_error_diagnostic(
                annotations=annotations,
                context=context,
                error=worker_error("refused"),
                receipt={
                    "commit": "a" * 40,
                    "tree_sha256": "b" * 64,
                    "engine_fingerprint": "c" * 64,
                    "policy": {"policy_id": "test-policy"},
                },
            )
            self.assertIsNone(suffix)
            payload = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(payload["error"], {"type": "WorkerError", "message": "refused"})
        self.assertEqual(payload["boundary"]["seat"], "p1")
        self.assertEqual(payload["boundary"]["volatiles"], {"p1": ["perish0"]})
        self.assertNotIn("requested_observations", payload["boundary"])

    def test_stats_payload_refuses_a_source_without_opponent_prior_application_counter(self) -> None:
        stats = SimpleNamespace(
            decisions=1,
            searched_decisions=1,
            fallback_decisions=0,
            model_evals=4,
            total_iterations=8,
            worlds_constructed=1,
            worlds_searched=1,
            prior_fallbacks=0,
            root_prior_fallbacks=0,
            branch_prior_fallbacks=0,
            branch_prior_fallback_reasons={
                name: 0 for name in BRANCH_PRIOR_FALLBACK_REASON_VALUES
            },
            decision_wall_seconds=0.25,
        )

        with self.assertRaisesRegex(worker_error, "opponent_prior_arm_decisions"):
            stats_payload(stats)

    def test_stats_payload_preserves_the_closed_order_status_ledger(self) -> None:
        stats = SimpleNamespace(
            decisions=1,
            searched_decisions=1,
            fallback_decisions=0,
            model_evals=4,
            total_iterations=8,
            worlds_constructed=1,
            worlds_searched=1,
            prior_fallbacks=1,
            root_prior_fallbacks=1,
            branch_prior_fallbacks=0,
            branch_prior_fallback_reasons={
                name: 0 for name in BRANCH_PRIOR_FALLBACK_REASON_VALUES
            },
            opponent_prior_arm_decisions=0,
            opponent_request_order_statuses={"lost_active_permutation": 1},
            opponent_request_order_root_fallback_statuses={"lost_active_permutation": 1},
            override_measured_decisions=1,
            model_override_decisions=1,
            decision_wall_seconds=0.25,
        )
        self.assertEqual(
            stats_payload(stats)["opponent_request_order_statuses"],
            {"lost_active_permutation": 1},
        )
        self.assertEqual(
            stats_payload(stats)["opponent_request_order_root_fallback_statuses"],
            {"lost_active_permutation": 1},
        )

    def test_stats_payload_completes_the_native_sparse_counter_reason_ledger(self) -> None:
        def stats_with(reasons: Counter[str], branch_fallbacks: int) -> SimpleNamespace:
            return SimpleNamespace(
                decisions=1,
                searched_decisions=1,
                fallback_decisions=0,
                model_evals=4,
                total_iterations=8,
                worlds_constructed=1,
                worlds_searched=1,
                prior_fallbacks=branch_fallbacks,
                root_prior_fallbacks=0,
                branch_prior_fallbacks=branch_fallbacks,
                branch_prior_fallback_reasons=reasons,
                opponent_prior_arm_decisions=0,
                override_measured_decisions=1,
                model_override_decisions=1,
                opponent_request_order_statuses={},
                opponent_request_order_root_fallback_statuses={},
                decision_wall_seconds=0.25,
            )

        self.assertEqual(
            stats_payload(stats_with(Counter(), 0))["branch_prior_fallback_reasons"],
            {name: 0 for name in BRANCH_PRIOR_FALLBACK_REASON_VALUES},
        )
        self.assertEqual(
            stats_payload(stats_with(Counter({"unmapped_action": 2}), 2))[
                "branch_prior_fallback_reasons"
            ],
            {
                name: 2 if name == "unmapped_action" else 0
                for name in BRANCH_PRIOR_FALLBACK_REASON_VALUES
            },
        )

    def test_stats_payload_keeps_unknown_native_counter_reason_fail_closed(self) -> None:
        stats = SimpleNamespace(
            decisions=1,
            searched_decisions=1,
            fallback_decisions=0,
            model_evals=4,
            total_iterations=8,
            worlds_constructed=1,
            worlds_searched=1,
            prior_fallbacks=1,
            root_prior_fallbacks=0,
            branch_prior_fallbacks=1,
            branch_prior_fallback_reasons=Counter({"future_reason": 1}),
            opponent_prior_arm_decisions=0,
            override_measured_decisions=1,
            model_override_decisions=1,
            opponent_request_order_statuses={},
            opponent_request_order_root_fallback_statuses={},
            decision_wall_seconds=0.25,
        )

        with self.assertRaisesRegex(worker_error, "complete and conserved"):
            stats_payload(stats)

    def test_config_projection_allows_only_explicit_disabled_diagnostics(self) -> None:
        class HistoricalEngineMctsConfig:
            __dataclass_fields__ = {"leaf_eval": object(), "search_sims": object()}

        projected, compatibility = source_engine_config_payload(
            {
                "leaf_eval": "model",
                "search_sims": 32,
                "root_selector_q": False,
                "root_selector_shadow": False,
                "model_decision_time_ms": None,
                "model_native_batch_guard_ms": 0,
            },
            HistoricalEngineMctsConfig,
        )

        self.assertEqual(projected, {"leaf_eval": "model", "search_sims": 32})
        self.assertEqual(
            compatibility,
            {
                "protocol": "disabled-diagnostic-omission.v1",
                "omitted_disabled_fields": [
                    "model_decision_time_ms", "model_native_batch_guard_ms",
                    "root_selector_q", "root_selector_shadow",
                ],
            },
        )

    def test_config_projection_refuses_enabled_or_unknown_historical_gaps(self) -> None:
        class HistoricalEngineMctsConfig:
            __dataclass_fields__ = {"leaf_eval": object()}

        with self.assertRaisesRegex(worker_error, "declared policy enables"):
            source_engine_config_payload(
                {"leaf_eval": "model", "root_selector_shadow": True},
                HistoricalEngineMctsConfig,
            )
        with self.assertRaisesRegex(worker_error, "declared policy enables"):
            source_engine_config_payload(
                {"leaf_eval": "model", "model_decision_time_ms": 1},
                HistoricalEngineMctsConfig,
            )
        with self.assertRaisesRegex(worker_error, "declared policy enables"):
            source_engine_config_payload(
                {"leaf_eval": "model", "model_native_batch_guard_ms": 1},
                HistoricalEngineMctsConfig,
            )
        with self.assertRaisesRegex(worker_error, "does not support host field"):
            source_engine_config_payload(
                {"leaf_eval": "model", "future_behavior_flag": False},
                HistoricalEngineMctsConfig,
            )

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
