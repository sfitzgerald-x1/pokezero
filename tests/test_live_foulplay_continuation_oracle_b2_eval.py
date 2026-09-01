from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "live_foulplay_continuation_oracle_b2_eval.py"


def _module():
    spec = importlib.util.spec_from_file_location("live_foulplay_continuation_oracle_b2_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(letter: str) -> str:
    return letter * 64


def _bridge_summary(module: object, *, seat: str, oracle: bool, seed: int) -> dict[str, object]:
    foulplay = "p2" if seat == "p1" else "p1"
    game: dict[str, object] = {
        "seed": seed,
        "winner": "PokeZeroBot",
        "pokezero_won": True,
        "tied": False,
        "capped": False,
        "pokezero_score": 1.0,
        "battle_id": f"b-{seat}-{seed}",
    }
    if oracle:
        game["live_continuation_oracle"] = {
            "schema_version": module.ORACLE_RECEIPT_SCHEMA_VERSION,
            "controller_only_full_state": True,
            "oracle_decision_count": 1,
            "forced_boundary_raw_decisions": 0,
            "oracle_decisions": [
                {
                    "schema_version": module.ORACLE_RECEIPT_SCHEMA_VERSION,
                    "controller": "live-foulplay-continuation-oracle",
                    "controller_status": "oracle-selected",
                    "full_state_snapshot_scope": "controller-only",
                    "source_decision_round": 1,
                    "pokezero_player": seat,
                    "foulplay_player": foulplay,
                    "source_request_sha256": {"p1": _digest("a"), "p2": _digest("b")},
                    "snapshot_request_sha256": {"p1": _digest("c"), "p2": _digest("d")},
                    "actual_foulplay_choice": "move 2",
                    "decoded_actual_foulplay_action": 1,
                    "raw_action_index": 0,
                    "selected_action_index": 1,
                    "selected_changed_raw_action": True,
                    "first_restored_joint_step": {seat: 1, foulplay: 1},
                    "candidate_count": 2,
                    "candidate_cap": 9,
                    "legal_action_indices": [0, 1],
                    "candidates": [
                        {
                            "action_index": 0,
                            "score": 0.0,
                            "continuation_decision_round_count": 1,
                            "terminal": {"winner": foulplay, "turn_count": 4, "capped": False},
                            "terminal_after_fixed_joint_step": False,
                        },
                        {
                            "action_index": 1,
                            "score": 1.0,
                            "continuation_decision_round_count": 1,
                            "terminal": {"winner": seat, "turn_count": 4, "capped": False},
                            "terminal_after_fixed_joint_step": False,
                        },
                    ],
                    "action_index": 1,
                }
            ],
        }
    provenance = {
        "bridge_schema_version": module.SOURCE_SCHEMA_VERSION,
        "bridge_source_sha256": _digest("e"),
        "format_id": "gen3randombattle",
        "capture_driver": "checkpoint",
        "belief_set_source": False,
        "max_decision_rounds": 64,
        "foulplay_search_time_ms": 1000,
        "checkpoint_path": "/checkpoint.pt",
        "checkpoint_sha256": _digest("f"),
        "showdown_root": "/showdown",
        "showdown_sim_sha256": _digest("1"),
        "foulplay_root": "/foulplay",
        "foulplay_entrypoint_sha256": _digest("2"),
        "foulplay_python": "/foulplay/python",
        "node_binary": "node",
    }
    return {
        "schema_version": module.SOURCE_SCHEMA_VERSION,
        "status": "complete",
        "complete": True,
        "games": 1,
        "completed_games": 1,
        "capped_games": 0,
        "policy_mode": "raw",
        "opponent_policy_id": "foul-play",
        "capture_driver": "checkpoint",
        "format_id": "gen3randombattle",
        "belief_set_source": False,
        "max_decision_rounds": 64,
        "checkpoint_sha256": _digest("f"),
        "pokezero_player": seat,
        "foulplay_player": foulplay,
        "live_continuation_oracle": {
            "enabled": oracle,
            "schema_version": module.ORACLE_RECEIPT_SCHEMA_VERSION,
            "candidate_cap": 9,
            "controller_only_full_state": True,
            "oracle_decisions": 1 if oracle else 0,
            "games_with_oracle_decision": 1 if oracle else 0,
            "forced_boundary_raw_decisions": 0,
        },
        "execution_integrity": {
            "retries": 0,
            "errors": 0,
            "refusal_records": 0,
            "refusal_recorder_instrument_errors": 0,
            "refusal_records_unrowed": 0,
            "forced_boundary_raw_decisions": 0,
        },
        "seed_start": seed,
        "foulplay_random_seed": seed,
        "foulplay_random_seed_schedule": {
            "count": 1,
            "first_seed": seed,
            "last_seed": seed,
            "mode": "constant",
            "seeds": [seed],
        },
        "foulplay_think": {"budget_ms_configured": 1000},
        "game_results": [game],
        "b2_provenance": provenance,
    }


def _unit_document(module: object, *, seat: str = "p1", seed: int | None = None) -> dict[str, object]:
    if seed is None:
        seed = module._ORIENTATION_REGISTRATION[seat]["seed_start"]
    registered = module._registered_unit(seat=seat, seed=seed)
    raw = _bridge_summary(module, seat=seat, oracle=False, seed=seed)
    oracle = _bridge_summary(module, seat=seat, oracle=True, seed=seed)
    return {
        "schema_version": module.SCHEMA_VERSION,
        "experiment_id": module.EXPERIMENT_ID,
        "complete": True,
        "write_protocol": module.WRITE_PROTOCOL,
        "status": "PASS",
        "pokezero_player": seat,
        "seed": seed,
        "source_files_sha256": {
            "scripts/live_foulplay_continuation_oracle_b2_eval.py": _digest("3"),
            "src/pokezero/foulplay_bridge.py": _digest("e"),
            "src/pokezero/live_foulplay_continuation.py": _digest("4"),
        },
        "registration": {
            "registered_units_per_orientation": module.REGISTERED_UNITS_PER_ORIENTATION,
            "shards_per_worker": module.SHARDS_PER_WORKER,
            "units_per_shard": module.UNITS_PER_SHARD,
            "worker_index": registered["worker_index"],
            "pokezero_player": seat,
            "foulplay_player": registered["foulplay_player"],
            "seed_start": registered["seed_start"],
            "seed": seed,
            "seed_offset": registered["seed_offset"],
            "shard_index": registered["shard_index"],
            "unit_index_in_shard": registered["unit_index_in_shard"],
            "shard_id": registered["shard_id"],
            "candidate_cap": 9,
            "arms": ["raw", "live-continuation-oracle"],
            "external_opponent": "FoulPlay",
            "checkpoint": "/checkpoint.pt",
            "checkpoint_sha256": _digest("f"),
        },
        "raw": {"treatment_policy_mode": "raw-transformer-policy", **raw},
        "oracle_continuation": {"treatment_policy_mode": "oracle-continuation", **oracle},
        "oracle_minus_raw_score": 0.0,
        "summary": {
            "seat_game_count": 1,
            "oracle_successful_games": 1,
            "oracle_minus_raw_score": 0.0,
        },
    }


class B2EvaluatorTest(unittest.TestCase):
    def test_accepts_one_registered_unit_for_each_orientation(self) -> None:
        module = _module()
        for seat in ("p1", "p2"):
            with self.subTest(seat=seat):
                module.validate_b2_document(_unit_document(module, seat=seat))

    def test_rejects_receipt_snapshot_leak_or_missing_full_legal_proof(self) -> None:
        module = _module()
        leaked = _unit_document(module)
        decision = leaked["oracle_continuation"]["game_results"][0]["live_continuation_oracle"]["oracle_decisions"][0]
        decision["bridge_snapshot"] = {"hidden": "must not persist"}
        with self.assertRaisesRegex(module.B2EvaluationError, "unexpected receipt shape"):
            module.validate_b2_document(leaked)

        missing_legal = _unit_document(module)
        decision = missing_legal["oracle_continuation"]["game_results"][0]["live_continuation_oracle"]["oracle_decisions"][0]
        del decision["legal_action_indices"]
        with self.assertRaisesRegex(module.B2EvaluationError, "unexpected receipt shape"):
            module.validate_b2_document(missing_legal)

    def test_rejects_serialized_full_state_at_either_arm_envelope(self) -> None:
        module = _module()
        for arm_name in ("raw", "oracle_continuation"):
            for field in ("snapshot", "bridge_snapshot", "full_state", "genericFullState"):
                with self.subTest(arm=arm_name, field=field):
                    leaked = _unit_document(module)
                    leaked[arm_name][field] = {"hidden": "must not persist"}
                    with self.assertRaisesRegex(module.B2EvaluationError, "serialized generic/full-state"):
                        module.validate_b2_document(leaked)

    def test_rejects_terminal_integrity_candidate_or_seed_schedule_failures(self) -> None:
        module = _module()
        capped = _unit_document(module)
        capped["raw"]["game_results"][0]["capped"] = True
        with self.assertRaisesRegex(module.B2EvaluationError, "capped"):
            module.validate_b2_document(capped)

        retried = _unit_document(module)
        retried["raw"]["execution_integrity"]["retries"] = 1
        with self.assertRaisesRegex(module.B2EvaluationError, "retries"):
            module.validate_b2_document(retried)

        malformed_selection = _unit_document(module)
        decision = malformed_selection["oracle_continuation"]["game_results"][0]["live_continuation_oracle"]["oracle_decisions"][0]
        decision["selected_action_index"] = 0
        decision["action_index"] = 0
        decision["first_restored_joint_step"]["p1"] = 0
        with self.assertRaisesRegex(module.B2EvaluationError, "stable max-score"):
            module.validate_b2_document(malformed_selection)

        mismatched_change_flag = _unit_document(module)
        decision = mismatched_change_flag["oracle_continuation"]["game_results"][0]["live_continuation_oracle"]["oracle_decisions"][0]
        decision["selected_changed_raw_action"] = False
        with self.assertRaisesRegex(module.B2EvaluationError, "selected_changed_raw_action"):
            module.validate_b2_document(mismatched_change_flag)

        wrong_schedule = _unit_document(module)
        wrong_schedule["oracle_continuation"]["foulplay_random_seed_schedule"] = {
            "count": 1,
            "first_seed": 108_000_001,
            "last_seed": 108_000_001,
            "mode": "constant",
            "seeds": [108_000_001],
        }
        with self.assertRaisesRegex(module.B2EvaluationError, "exact one-seed FoulPlay schedule"):
            module.validate_b2_document(wrong_schedule)

    def test_rejects_incomplete_source_arm_or_missing_integrity_field(self) -> None:
        module = _module()
        incomplete = _unit_document(module)
        incomplete["raw"]["games"] = 2
        with self.assertRaisesRegex(module.B2EvaluationError, "exactly one configured game"):
            module.validate_b2_document(incomplete)

        malformed_integrity = _unit_document(module)
        del malformed_integrity["raw"]["execution_integrity"]["errors"]
        with self.assertRaisesRegex(module.B2EvaluationError, "unexpected receipt shape"):
            module.validate_b2_document(malformed_integrity)

    def test_rejects_treatment_identity_that_confuses_raw_bridge_mode_with_b2_arm(self) -> None:
        module = _module()
        payload = _unit_document(module)
        payload["oracle_continuation"]["treatment_policy_mode"] = "raw-transformer-policy"
        with self.assertRaisesRegex(module.B2EvaluationError, "treatment identity"):
            module.validate_b2_document(payload)

    def test_rejects_cross_band_or_wrong_shard_ownership(self) -> None:
        module = _module()
        with self.assertRaisesRegex(module.B2EvaluationError, "outside its registered orientation band"):
            module._registered_unit(seat="p1", seed=108_000_600)

        wrong_owner = _unit_document(module)
        wrong_owner["registration"]["shard_index"] = 1
        with self.assertRaisesRegex(module.B2EvaluationError, "shard_index"):
            module.validate_b2_document(wrong_owner)

    def test_registers_two_nonoverlapping_600_unit_bands_in_24_ordered_shards(self) -> None:
        module = _module()
        self.assertEqual(module.REGISTERED_UNITS_PER_ORIENTATION, 600)
        self.assertEqual(module.SHARDS_PER_WORKER, 24)
        self.assertEqual(module.UNITS_PER_SHARD, 25)
        self.assertEqual(module._registered_unit(seat="p1", seed=108_000_024)["shard_index"], 0)
        self.assertEqual(module._registered_unit(seat="p1", seed=108_000_025)["shard_index"], 1)
        p2_last = module._registered_unit(seat="p2", seed=108_001_199)
        self.assertEqual(p2_last["worker_index"], 1)
        self.assertEqual(p2_last["shard_index"], 23)
        self.assertEqual(p2_last["unit_index_in_shard"], 24)


if __name__ == "__main__":
    unittest.main()
