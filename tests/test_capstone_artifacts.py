from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pokezero.capstone_artifacts import (
    analyze_normalized_capstone_pairs,
    load_normalized_pair,
    normalize_controlled_foulplay_artifact,
    normalize_root_puct_play_artifact,
)


def value_leaf() -> dict[str, object]:
    return {
        "policy_checkpoint_sha256": "a" * 64,
        "value_checkpoint_sha256": "b" * 64,
        "value_calibration_source_checkpoint_sha256": "a" * 64,
        "model_config_match": True,
        "belief_set_source_hash_match": True,
        "value_calibration_transform": {"method": "isotonic"},
    }


def root_payload(
    *,
    seat: str,
    fallback: int = 0,
    include_value_leaf: bool = True,
    root_time_budget_ms: int | None = None,
) -> dict[str, object]:
    opponent_seat = "p2" if seat == "p1" else "p1"
    raw_id = "test-policy"
    root_id = f"{raw_id}+root-puct"

    def matchup(policy_id: str, *, score: float, searches: int) -> dict[str, object]:
        policy_ids = {seat: policy_id, opponent_seat: "max-damage"}
        diagnostics = (
            {
                seat: {
                    "root_puct_searches": searches,
                    "root_puct_fallbacks": fallback,
                    "root_puct_opponent_action_policies": {"checkpoint": searches},
                    "root_puct_elapsed_seconds": [0.4, 0.6],
                    **(
                        {
                            "root_puct_time_budget_checks": searches,
                            "root_puct_time_budget_exhaustions": 1,
                        }
                        if root_time_budget_ms is not None
                        else {}
                    ),
                }
            }
            if searches
            else {}
        )
        return {
            "p1_policy_id": policy_ids["p1"],
            "p2_policy_id": policy_ids["p2"],
            "p1_policy_provenance": {"weights_sha256": "a" * 64} if seat == "p1" else {},
            "p2_policy_provenance": {"weights_sha256": "a" * 64} if seat == "p2" else {},
            "game_results": [
                {
                    "seed": 101,
                    "p1_score": score if seat == "p1" else 0.0,
                    "p2_score": score if seat == "p2" else 0.0,
                    "tied": False,
                    "capped": False,
                    "opponent_legal_mask_mode": "hidden",
                    "root_puct_by_player": diagnostics,
                    "policy_elapsed_seconds_by_player": {seat: [0.4, 0.6]},
                }
            ],
        }

    payload: dict[str, object] = {
        "matchups": [
            matchup(raw_id, score=0.0, searches=0),
            matchup(root_id, score=1.0, searches=2),
        ]
    }
    if include_value_leaf:
        payload["value_leaf"] = value_leaf()
    if root_time_budget_ms is not None:
        payload["root_time_budget_ms"] = root_time_budget_ms
    return payload


def foulplay_payload(*, seat: str, fallback: int = 0) -> dict[str, object]:
    schedule = {"count": 1, "first_seed": 101, "last_seed": 101, "mode": "constant", "seeds": [101]}

    def run(*, score: float, searches: int, policy_mode: str) -> dict[str, object]:
        return {
            "complete": True,
            "pokezero_player": seat,
            "checkpoint_sha256": "a" * 64,
            "seed_start": 101,
            "foulplay_random_seed": 101,
            "foulplay_random_seed_schedule": schedule,
            "root_puct": {
                "opponent_legal_mask_mode": "hidden",
                "allow_search_fallback": False,
                "opponent_action_policies": {"checkpoint": searches} if searches else {},
            },
            "value_leaf": value_leaf() if policy_mode == "root-puct" else None,
            "game_results": [
                {
                    "seed": 101,
                    "pokezero_score": score,
                    "tied": False,
                    "capped": False,
                    "root_puct_searches": searches,
                    "root_puct_fallbacks": fallback,
                    "root_puct_opponent_action_policies": {"checkpoint": searches} if searches else {},
                    "pokezero_decision_players": [seat],
                    "pokezero_submitted_choice_players": [seat],
                    "policy_elapsed_seconds": [0.4, 0.6],
                }
            ],
        }

    return {
        "comparison_mode": "per-seed",
        "complete": True,
        "foulplay_random_seed_schedule": schedule,
        "opponent_crashes": [],
        "runs": {
            "raw": run(score=0.0, searches=0, policy_mode="raw"),
            "root_puct": run(score=1.0, searches=2, policy_mode="root-puct"),
        },
    }


class CapstoneArtifactsTest(unittest.TestCase):
    def test_normalizes_root_play_pair_with_value_leaf_and_wall_samples(self) -> None:
        pair = normalize_root_puct_play_artifact(
            root_payload(seat="p1"),
            opponent_id="max-damage",
            arm_id="value-24",
            band="a",
            seat="p1",
        )

        self.assertEqual(pair.raw.outcomes[0].score, 0.0)
        self.assertEqual(pair.candidate.outcomes[0].score, 1.0)
        self.assertEqual(pair.candidate.calibrated_value_copy, f"isotonic:{'b' * 64}")
        self.assertEqual(pair.candidate_wall_seconds, (0.4, 0.6))

    def test_root_normalization_rejects_mechanics_only_artifact(self) -> None:
        payload = root_payload(seat="p1")
        payload["strength_evidence_eligible"] = False

        with self.assertRaisesRegex(ValueError, "mechanics-only"):
            normalize_root_puct_play_artifact(
                payload,
                opponent_id="max-damage",
                arm_id="value-24",
                band="a",
                seat="p1",
            )

    def test_root_normalization_rejects_missing_value_provenance_and_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing frozen calibration provenance"):
            normalize_root_puct_play_artifact(
                root_payload(seat="p1", include_value_leaf=False),
                opponent_id="max-damage",
                arm_id="value-24",
                band="a",
                seat="p1",
            )
        with self.assertRaisesRegex(ValueError, "used a search fallback"):
            normalize_root_puct_play_artifact(
                root_payload(seat="p1", fallback=1),
                opponent_id="max-damage",
                arm_id="value-24",
                band="a",
                seat="p1",
            )

    def test_root_normalization_requires_time_budget_diagnostics_when_configured(self) -> None:
        payload = root_payload(seat="p1", root_time_budget_ms=125)
        diagnostics = payload["matchups"][1]["game_results"][0]["root_puct_by_player"]["p1"]
        del diagnostics["root_puct_time_budget_checks"]
        with self.assertRaisesRegex(ValueError, "time_budget_checks"):
            normalize_root_puct_play_artifact(
                payload,
                opponent_id="max-damage",
                arm_id="rollout-tail",
                band="a",
                seat="p1",
            )
        payload = root_payload(seat="p1", root_time_budget_ms=125)
        diagnostics = payload["matchups"][1]["game_results"][0]["root_puct_by_player"]["p1"]
        diagnostics["root_puct_time_budget_checks"] = 1
        with self.assertRaisesRegex(ValueError, "invalid time-budget diagnostics"):
            normalize_root_puct_play_artifact(
                payload,
                opponent_id="max-damage",
                arm_id="rollout-tail",
                band="a",
                seat="p1",
            )

    def test_normalizes_controlled_foulplay_with_hidden_seat_proof(self) -> None:
        pair = normalize_controlled_foulplay_artifact(
            foulplay_payload(seat="p2"),
            arm_id="value-24",
            band="a",
            seat="p2",
        )

        self.assertEqual(pair.opponent_id, "foul-play")
        self.assertEqual(pair.raw.outcomes[0].seat, "p2")
        self.assertEqual(pair.candidate_wall_seconds, (0.4, 0.6))

    def test_rejects_mismatched_foulplay_schedule_and_checkpoint_lineage(self) -> None:
        schedule_mismatch = foulplay_payload(seat="p1")
        schedule_mismatch["runs"]["root_puct"]["foulplay_random_seed_schedule"] = {
            "count": 1,
            "first_seed": 202,
            "last_seed": 202,
            "mode": "constant",
            "seeds": [202],
        }
        with self.assertRaisesRegex(ValueError, "startup seed schedules do not match"):
            normalize_controlled_foulplay_artifact(
                schedule_mismatch,
                arm_id="value-24",
                band="a",
                seat="p1",
            )

    def test_rejects_privileged_or_missing_foulplay_planner_evidence(self) -> None:
        privileged = foulplay_payload(seat="p1")
        privileged["runs"]["root_puct"]["game_results"][0]["root_puct_opponent_action_policies"] = {
            "benchmark": 2
        }
        with self.assertRaisesRegex(ValueError, "privileged opponent-action planner"):
            normalize_controlled_foulplay_artifact(
                privileged,
                arm_id="value-24",
                band="a",
                seat="p1",
            )

        missing = foulplay_payload(seat="p1")
        del missing["runs"]["root_puct"]["game_results"][0]["root_puct_opponent_action_policies"]
        with self.assertRaisesRegex(ValueError, "per-game opponent-action planner evidence is missing"):
            normalize_controlled_foulplay_artifact(
                missing,
                arm_id="value-24",
                band="a",
                seat="p1",
            )

        lineage_mismatch = foulplay_payload(seat="p1")
        lineage_mismatch["runs"]["root_puct"]["checkpoint_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint hashes do not match"):
            normalize_controlled_foulplay_artifact(
                lineage_mismatch,
                arm_id="value-24",
                band="a",
                seat="p1",
            )

    def test_root_normalization_rejects_privileged_planners_missing_diagnostics_and_lineage(self) -> None:
        privileged = root_payload(seat="p1")
        privileged["matchups"][1]["game_results"][0]["root_puct_by_player"]["p1"][
            "root_puct_opponent_action_policies"
        ] = {"benchmark": 2}
        with self.assertRaisesRegex(ValueError, "privileged opponent-action planner"):
            normalize_root_puct_play_artifact(
                privileged,
                opponent_id="max-damage",
                arm_id="value-24",
                band="a",
                seat="p1",
            )

        missing_fallback = root_payload(seat="p1")
        del missing_fallback["matchups"][1]["game_results"][0]["root_puct_by_player"]["p1"][
            "root_puct_fallbacks"
        ]
        with self.assertRaisesRegex(ValueError, "missing root_puct_fallbacks"):
            normalize_root_puct_play_artifact(
                missing_fallback,
                opponent_id="max-damage",
                arm_id="value-24",
                band="a",
                seat="p1",
            )

        lineage_mismatch = root_payload(seat="p1")
        lineage_mismatch["matchups"][0]["p1_policy_provenance"]["weights_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "raw policy hash"):
            normalize_root_puct_play_artifact(
                lineage_mismatch,
                opponent_id="max-damage",
                arm_id="value-24",
                band="a",
                seat="p1",
            )

    def test_analysis_requires_full_mirrored_roster_and_reports_wall_time(self) -> None:
        p1 = normalize_root_puct_play_artifact(
            root_payload(seat="p1"),
            opponent_id="max-damage",
            arm_id="value-24",
            band="a",
            seat="p1",
        )
        p2 = normalize_root_puct_play_artifact(
            root_payload(seat="p2"),
            opponent_id="max-damage",
            arm_id="value-24",
            band="a",
            seat="p2",
        )
        report = analyze_normalized_capstone_pairs(
            (p1, p2),
            expected_keys=(("a", 101, "p1"), ("a", 101, "p2")),
            bootstrap_replicates=100,
            bootstrap_seed=7,
        )

        row = report["primary_arms"][0]
        self.assertEqual(row["candidate_wall_seconds"], {"decision_samples": 4, "mean_seconds": 0.5, "p95_seconds": 0.6})
        self.assertEqual(row["paired_delta_vs_baseline"]["overall"]["candidate_minus_baseline_score_rate"], 1.0)

    def test_persisted_pair_is_rederived_from_hashed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            output = root / "normalized.json"
            source.write_text(json.dumps(root_payload(seat="p1")), encoding="utf-8")
            pair = normalize_root_puct_play_artifact(
                json.loads(source.read_text(encoding="utf-8")),
                opponent_id="max-damage",
                arm_id="value-24",
                band="a",
                seat="p1",
                source_path=source,
            )
            output.write_text(json.dumps(pair.to_dict()), encoding="utf-8")
            self.assertEqual(load_normalized_pair(output), pair)

            tampered = pair.to_dict()
            tampered["candidate"]["outcomes"][0]["score"] = 0.0
            output.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match source-derived evidence"):
                load_normalized_pair(output)


if __name__ == "__main__":
    unittest.main()
