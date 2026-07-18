import unittest

from pokezero.root_puct_telemetry import (
    ROOT_PUCT_DECISION_TELEMETRY_SCHEMA_VERSION,
    ROOT_PUCT_TELEMETRY_REPORT_SCHEMA_VERSION,
    root_puct_benchmark_telemetry_report,
    root_puct_decision_telemetry,
    summarize_root_puct_decision_telemetry,
)


class RootPUCTTelemetryTest(unittest.TestCase):
    def test_compact_decision_keeps_stable_taxonomy_without_raw_error(self) -> None:
        telemetry = root_puct_decision_telemetry(
            {
                "policy_family": "root-puct-search",
                "root_puct_fallback": True,
                "root_puct_fallback_category": "replay_request_mismatch",
                "root_puct_fallback_reason": "request mismatch carried private state: [secret]",
                "root_puct_total_visits": 12,
                "root_puct_effective_total_visits": 9,
                "root_puct_elapsed_seconds": 0.25,
                "policy_elapsed_seconds": 0.30,
                "root_puct_timing": {
                    "prefix_replay_seconds": 0.10,
                    "prefix_replay_count": 2,
                    "root_initial_sweep_orchestration_seconds": 0.04,
                    "root_initial_sweep_orchestration_count": 1,
                    "scenario_dispatch_orchestration_seconds": 0.03,
                    "scenario_dispatch_orchestration_count": 2,
                    "puct_search_result_residual_seconds": 0.04,
                    "puct_search_result_residual_count": 1,
                    "puct_search_unrecorded_call_seconds": 0.01,
                    "puct_search_call_count": 2,
                    "raw_outer_policy_residual_seconds": 0.02,
                    "outer_policy_residual_seconds": 0.02,
                    "value_neural_forward_seconds": 0.06,
                    "value_neural_forward_count": 3,
                    "total_seconds": 0.25,
                    "private_debug_detail": "must not persist",
                },
                "root_puct_opponent_action_skip_categories": {"replay_request_mismatch": 3},
                "root_puct_direct_materialization_rejection_categories": {
                    "observation_mismatch": 2,
                    "private error detail": 4,
                },
                "root_puct_opponent_action_replay_request_mismatch_players": {"p2": 3},
                "root_puct_opponent_action_missing_sampled_world_reason_categories": {
                    "self_team_unavailable": 2,
                    "private replay detail": 8,
                    "opponent_belief_unavailable": -1,
                },
            },
            decision_index=4,
            turn_index=7,
        )

        self.assertEqual(
            telemetry,
            {
                "schema_version": "pokezero.root_puct_decision_telemetry.v1",
                "decision_index": 4,
                "turn_index": 7,
                "outcome": "fallback",
                "fallback": True,
                "fallback_category": "replay_request_mismatch",
                "root_puct_total_visits": 12,
                "root_puct_effective_total_visits": 9,
                "root_puct_elapsed_seconds": 0.25,
                "policy_elapsed_seconds": 0.30,
                "full_decision_elapsed_seconds": 0.30,
                "timing": {
                    "prefix_replay_seconds": 0.10,
                    "prefix_replay_count": 2,
                    "root_initial_sweep_orchestration_seconds": 0.04,
                    "root_initial_sweep_orchestration_count": 1,
                    "scenario_dispatch_orchestration_seconds": 0.03,
                    "scenario_dispatch_orchestration_count": 2,
                    "puct_search_result_residual_seconds": 0.04,
                    "puct_search_result_residual_count": 1,
                    "puct_search_unrecorded_call_seconds": 0.01,
                    "puct_search_call_count": 2,
                    "raw_outer_policy_residual_seconds": 0.02,
                    "outer_policy_residual_seconds": 0.02,
                    "value_neural_forward_seconds": 0.06,
                    "value_neural_forward_count": 3,
                    "total_seconds": 0.25,
                },
                "counters": {
                    "root_puct_direct_materialization_rejection_categories": {
                        "observation_mismatch": 2
                    },
                    "root_puct_opponent_action_skip_categories": {"replay_request_mismatch": 3},
                    "root_puct_opponent_action_replay_request_mismatch_players": {"p2": 3},
                    "root_puct_opponent_action_missing_sampled_world_reason_categories": {
                        "self_team_unavailable": 2
                    },
                },
            },
        )
        self.assertNotIn("root_puct_fallback_reason", telemetry)

    def test_compact_decision_signs_force_switch_without_raw_error(self) -> None:
        telemetry = root_puct_decision_telemetry(
            {
                "policy_family": "root-puct-search",
                "root_puct_fallback": True,
                "root_puct_fallback_category": "force_switch_illegal_action",
                "root_puct_fallback_reason": (
                    "search failed: p1: action_index 1 is not legal for the current request "
                    "(request_kind=force_switch)."
                ),
            },
            decision_index=3,
            turn_index=8,
        )

        assert telemetry is not None
        self.assertEqual(telemetry["fallback_signature"], "force-switch:search:p1:move")
        self.assertNotIn("root_puct_fallback_reason", telemetry)

    def test_summary_reports_fallbacks_taxonomy_visits_and_wall_samples(self) -> None:
        report = summarize_root_puct_decision_telemetry(
            (
                {
                    "outcome": "searched",
                    "fallback": False,
                    "root_puct_total_visits": 10,
                    "root_puct_effective_total_visits": 8,
                    "root_puct_start_override_direct_materializations": 2,
                    "root_puct_start_override_replay_materializations": 1,
                    "root_puct_opponent_action_scenario_count": 2,
                    "root_puct_opponent_action_scenarios_generated": 3,
                    "root_puct_opponent_action_scenarios_skipped": 1,
                    "root_puct_elapsed_seconds": 0.20,
                    "policy_elapsed_seconds": 0.25,
                    "full_decision_elapsed_seconds": 0.25,
                    "timing": {
                        "total_seconds": 0.20,
                        "prefix_replay_seconds": 0.10,
                        "puct_search_result_residual_seconds": 0.08,
                        "puct_search_result_residual_count": 1,
                        "puct_search_unrecorded_call_seconds": 0.01,
                        "puct_search_call_count": 1,
                        "outer_policy_residual_seconds": 0.01,
                    },
                },
                {
                    "outcome": "fallback",
                    "fallback": True,
                    "fallback_category": "replay_request_mismatch",
                    "root_puct_total_visits": 2,
                    "root_puct_effective_total_visits": 1,
                    "root_puct_start_override_direct_materializations": 1,
                    "root_puct_start_override_replay_materializations": 3,
                    "root_puct_opponent_action_scenario_count": 1,
                    "root_puct_opponent_action_scenarios_generated": 3,
                    "root_puct_opponent_action_scenarios_skipped": 2,
                    "root_puct_elapsed_seconds": 0.40,
                    "policy_elapsed_seconds": 0.45,
                    "full_decision_elapsed_seconds": 0.45,
                    "timing": {
                        "total_seconds": 0.40,
                        "prefix_replay_seconds": 0.20,
                        "puct_search_result_residual_seconds": 0.09,
                        "puct_search_result_residual_count": 1,
                        "puct_search_unrecorded_call_seconds": 0.02,
                        "puct_search_call_count": 2,
                        "outer_policy_residual_seconds": 0.03,
                    },
                    "counters": {
                        "root_puct_direct_materialization_rejection_categories": {
                            "observation_mismatch": 1
                        },
                        "root_puct_opponent_action_skip_categories": {"replay_request_mismatch": 2},
                        "root_puct_opponent_action_missing_sampled_world_reason_categories": {
                            "opponent_belief_unavailable": 1
                        },
                    },
                },
            )
        )

        self.assertEqual(report["schema_version"], ROOT_PUCT_DECISION_TELEMETRY_SCHEMA_VERSION)
        self.assertEqual(report["decisions"], 2)
        self.assertEqual(report["searches"], 1)
        self.assertEqual(report["fallbacks"], 1)
        self.assertEqual(report["fallback_categories"], {"replay_request_mismatch": 1})
        self.assertEqual(report["fallback_signatures"], {})
        self.assertEqual(
            report["scenario_counts"],
            {
                "scenarios_used": 3,
                "scenarios_generated": 6,
                "scenarios_skipped": 3,
            },
        )
        self.assertEqual(report["materialization_counts"], {"direct": 3, "replay": 4})
        self.assertEqual(
            report["scenario_failure_taxonomy"],
            {
                "direct_materialization_rejection_categories": {
                    "observation_mismatch": 1
                },
                "missing_sampled_world_reason_categories": {
                    "opponent_belief_unavailable": 1
                },
                "skip_categories": {"replay_request_mismatch": 2},
            },
        )
        self.assertEqual(report["visits"]["total"], 12)
        self.assertEqual(report["visits"]["effective_total"], 9)
        self.assertEqual(report["visits"]["mean_per_search"], 12.0)
        self.assertAlmostEqual(report["visits"]["per_root_search_second"], 20.0)
        self.assertEqual(report["branch_search_wall_seconds"]["samples"], 2)
        self.assertAlmostEqual(report["branch_search_wall_seconds"]["mean"], 0.30)
        self.assertEqual(report["branch_search_wall_seconds"]["p50"], 0.20)
        self.assertEqual(report["branch_search_wall_seconds"]["p95"], 0.40)
        self.assertEqual(report["full_decision_wall_seconds"]["samples"], 2)
        self.assertAlmostEqual(report["full_decision_wall_seconds"]["mean"], 0.35)
        self.assertAlmostEqual(report["timing_totals"]["prefix_replay_seconds"], 0.30)
        self.assertAlmostEqual(
            report["timing_totals"]["puct_search_result_residual_seconds"], 0.17
        )
        self.assertEqual(report["timing_totals"]["puct_search_result_residual_count"], 2)
        self.assertAlmostEqual(
            report["timing_totals"]["puct_search_unrecorded_call_seconds"], 0.03
        )
        self.assertEqual(report["timing_totals"]["puct_search_call_count"], 3)
        self.assertAlmostEqual(report["timing_totals"]["outer_policy_residual_seconds"], 0.04)
        self.assertAlmostEqual(report["timing_totals"]["total_seconds"], 0.60)

    def test_benchmark_report_groups_records_by_root_puct_policy(self) -> None:
        payload = {
            "matchups": [
                {
                    "p1_policy_id": "root-puct-120",
                    "p2_policy_id": "random-legal",
                    "game_results": [
                        {
                            "root_puct_decision_telemetry_by_player": {
                                "p1": [
                                    {
                                        "schema_version": "pokezero.root_puct_decision_telemetry.v1",
                                        "decision_index": 0,
                                        "turn_index": 0,
                                        "outcome": "searched",
                                        "fallback": False,
                                        "root_puct_total_visits": 24,
                                        "full_decision_elapsed_seconds": 0.15,
                                        "timing": {"total_seconds": 0.12},
                                    }
                                ]
                            }
                        }
                    ],
                }
            ]
        }

        report = root_puct_benchmark_telemetry_report(payload, policy_ids=("root-puct-120",))

        self.assertEqual(report["schema_version"], ROOT_PUCT_TELEMETRY_REPORT_SCHEMA_VERSION)
        self.assertEqual(report["policies"]["root-puct-120"]["decisions"], 1)
        self.assertEqual(report["policies"]["root-puct-120"]["visits"]["per_root_search_second"], 200.0)
        with self.assertRaisesRegex(ValueError, "no Root-PUCT decision telemetry"):
            root_puct_benchmark_telemetry_report(payload, policy_ids=("missing",))

    def test_benchmark_report_rejects_unversioned_telemetry(self) -> None:
        payload = {
            "matchups": [
                {
                    "p1_policy_id": "root-puct-120",
                    "p2_policy_id": "random-legal",
                    "game_results": [
                        {
                            "root_puct_decision_telemetry_by_player": {
                                "p1": [{"outcome": "searched", "fallback": False}]
                            }
                        }
                    ],
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "incompatible .* schema"):
            root_puct_benchmark_telemetry_report(payload)

    def test_benchmark_report_rejects_incomplete_searched_telemetry(self) -> None:
        payload = {
            "matchups": [
                {
                    "p1_policy_id": "root-puct-120",
                    "p2_policy_id": "random-legal",
                    "game_results": [
                        {
                            "root_puct_decision_telemetry_by_player": {
                                "p1": [
                                    {
                                        "schema_version": "pokezero.root_puct_decision_telemetry.v1",
                                        "decision_index": 0,
                                        "turn_index": 0,
                                        "outcome": "searched",
                                        "fallback": False,
                                        "full_decision_elapsed_seconds": 0.15,
                                    }
                                ]
                            }
                        }
                    ],
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "incomplete .* missing root visit count"):
            root_puct_benchmark_telemetry_report(payload)

    def test_benchmark_report_rejects_missing_root_puct_seat_telemetry(self) -> None:
        payload = {
            "matchups": [
                {
                    "p1_policy_id": "root-puct-120",
                    "p2_policy_id": "random-legal",
                    "game_results": [{}],
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "missing Root-PUCT telemetry"):
            root_puct_benchmark_telemetry_report(payload)


if __name__ == "__main__":
    unittest.main()
