from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


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


def _fixed_mcts_seed(*, base_seed: int, decision_index: int, sample_index: int) -> int:
    import hashlib

    return int.from_bytes(
        hashlib.sha256(f"{base_seed}:{decision_index}:{sample_index}".encode("ascii")).digest()[:8],
        "big",
    )


def _bridge_summary(
    module: object, *, seat: str, oracle: bool, seed: int, candidate_parallelism: int = 1,
) -> dict[str, object]:
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
                    "candidate_parallelism": candidate_parallelism,
                    "max_continuation_decision_rounds": 128,
                    "expanded_continuation_decision_rounds": 1024,
                    "legal_action_indices": [0, 1],
                    "candidates": [
                        {
                            "action_index": 0,
                            "score": 0.0,
                            "continuation_decision_round_count": 1,
                            "max_continuation_decision_rounds": 128,
                            "terminal": {"winner": foulplay, "turn_count": 4, "capped": False},
                            "terminal_after_fixed_joint_step": False,
                        },
                        {
                            "action_index": 1,
                            "score": 1.0,
                            "continuation_decision_round_count": 1,
                            "max_continuation_decision_rounds": 128,
                            "terminal": {"winner": seat, "turn_count": 4, "capped": False},
                            "terminal_after_fixed_joint_step": False,
                        },
                    ],
                    "action_index": 1,
                }
            ],
        }
    fixed_iterations = 10_001
    fixed_audit = {
        "schema": module.FIXED_MCTS_SCHEMA_VERSION,
        "decision_index": 0,
        "sample_index": 0,
        "mcts_seed": _fixed_mcts_seed(base_seed=seed, decision_index=0, sample_index=0),
        "iterations_requested": fixed_iterations,
        "total_visits": fixed_iterations,
        "threads": 1,
        "parallelism": 1,
    }
    game["opponent_think"] = [
        {
            "round": 0,
            "status": "ok",
            "fixed_mcts": {
                "schema_version": module.FIXED_MCTS_SCHEMA_VERSION,
                "audits": [fixed_audit],
            },
        }
    ]
    game["opponent_think_record_failures"] = 0
    provenance = {
        "bridge_schema_version": module.SOURCE_SCHEMA_VERSION,
        "bridge_source_sha256": _digest("e"),
        "unit_evaluator_source_sha256": _digest("3"),
        "live_continuation_source_sha256": _digest("4"),
        "format_id": "gen3randombattle",
        "capture_driver": "checkpoint",
        "belief_set_source": False,
        "max_decision_rounds": 64,
        "candidate_parallelism": candidate_parallelism,
        "max_continuation_decision_rounds": 128,
        "expanded_continuation_decision_rounds": 1024,
        "foulplay_search_time_ms": 1000,
        "foulplay_mcts_iterations": fixed_iterations,
        "foulplay_fixed_mcts_audit_schema_version": module.FIXED_MCTS_SCHEMA_VERSION,
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
            "candidate_parallelism": candidate_parallelism,
            "max_continuation_decision_rounds": 128,
            "expanded_continuation_decision_rounds": 1024,
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
        "foulplay_think": {
            "schema_version": module.FOULPLAY_THINK_SCHEMA_VERSION,
            "budget_ms_configured": 1000,
            "fixed_mcts_iterations_configured": fixed_iterations,
            "fixed_mcts_audit_required": True,
            "decisions": 1,
            "decisions_attempted": 1,
            "record_failures": 0,
            "miss_decisions": 0,
        },
        "game_results": [game],
        "b2_provenance": provenance,
    }


def _unit_document(
    module: object, *, seat: str = "p1", seed: int | None = None, registration_seed_start: int | None = None,
    arm_execution_order: str = "raw-then-oracle", raw_reproducibility_control: bool = False,
    candidate_parallelism: int = 1,
) -> dict[str, object]:
    if seed is None:
        seed = registration_seed_start if registration_seed_start is not None else module._ORIENTATION_REGISTRATION[seat]["seed_start"]
    registered = module._registered_unit(
        seat=seat, seed=seed, registration_seed_start=registration_seed_start,
    )
    raw = _bridge_summary(
        module, seat=seat, oracle=False, seed=seed, candidate_parallelism=candidate_parallelism,
    )
    oracle = _bridge_summary(
        module, seat=seat, oracle=True, seed=seed, candidate_parallelism=candidate_parallelism,
    )
    document = {
        "schema_version": module.SCHEMA_VERSION,
        "experiment_id": module.EXPERIMENT_ID,
        "complete": True,
        "write_protocol": module.WRITE_PROTOCOL,
        "status": "PASS",
        "pokezero_player": seat,
        "seed": seed,
        "execution": {
            "arm_execution_order": arm_execution_order,
            "raw_reproducibility_control": raw_reproducibility_control,
        },
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
            "candidate_parallelism": candidate_parallelism,
            "max_continuation_decision_rounds": 128,
            "expanded_continuation_decision_rounds": 1024,
            "arms": (
                ["raw", "live-continuation-oracle", "raw-reproducibility-control"]
                if raw_reproducibility_control
                else ["raw", "live-continuation-oracle"]
            ),
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
    if raw_reproducibility_control:
        control = _bridge_summary(
            module,
            seat=seat,
            oracle=False,
            seed=seed,
            candidate_parallelism=candidate_parallelism,
        )
        document["raw_reproducibility_control"] = {
            "treatment_policy_mode": "raw-transformer-policy-reproducibility-control", **control,
        }
        document["raw_reproducibility_control_minus_raw_score"] = 0.0
    return document


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
        with self.assertRaisesRegex(module.B2EvaluationError, "serialized generic/full-state"):
            module.validate_b2_document(leaked)

        missing_legal = _unit_document(module)
        decision = missing_legal["oracle_continuation"]["game_results"][0]["live_continuation_oracle"]["oracle_decisions"][0]
        del decision["legal_action_indices"]
        with self.assertRaisesRegex(module.B2EvaluationError, "unexpected receipt shape"):
            module.validate_b2_document(missing_legal)

    def test_rejects_serialized_full_state_at_any_arm_evidence_depth(self) -> None:
        module = _module()
        for arm_name in ("raw", "oracle_continuation"):
            for evidence_path, field in (
                ((), "snapshot"),
                ((), "bridge_snapshot"),
                ((), "full_state"),
                (("game_results", 0), "raw_snapshot"),
                (("game_results", 0), "snapshot_data"),
                (("game_results", 0), "state"),
                (("game_results", 0, "nested_evidence"), "genericFullState"),
            ):
                with self.subTest(arm=arm_name, field=field):
                    leaked = _unit_document(module)
                    target = leaked[arm_name]
                    for path_part in evidence_path:
                        if isinstance(path_part, int):
                            target = target[path_part]
                        else:
                            target = target.setdefault(path_part, {})
                    target[field] = {"hidden": "must not persist"}
                    with self.assertRaisesRegex(module.B2EvaluationError, "serialized generic/full-state"):
                        module.validate_b2_document(leaked)

    def test_requires_exact_source_manifest_and_binds_paths_and_hashes_to_arms(self) -> None:
        module = _module()
        missing_source = _unit_document(module)
        del missing_source["source_files_sha256"]["src/pokezero/live_foulplay_continuation.py"]
        with self.assertRaisesRegex(module.B2EvaluationError, "source_files_sha256.*unexpected receipt shape"):
            module.validate_b2_document(missing_source)

        extra_source = _unit_document(module)
        extra_source["source_files_sha256"]["src/pokezero/extra.py"] = _digest("9")
        with self.assertRaisesRegex(module.B2EvaluationError, "source_files_sha256.*unexpected receipt shape"):
            module.validate_b2_document(extra_source)

        for source_path, provenance_field in (
            ("scripts/live_foulplay_continuation_oracle_b2_eval.py", "unit_evaluator_source_sha256"),
            ("src/pokezero/foulplay_bridge.py", "bridge_source_sha256"),
            ("src/pokezero/live_foulplay_continuation.py", "live_continuation_source_sha256"),
        ):
            with self.subTest(source_path=source_path):
                mismatched_source = _unit_document(module)
                mismatched_source["source_files_sha256"][source_path] = _digest("9")
                with self.assertRaisesRegex(module.B2EvaluationError, provenance_field):
                    module.validate_b2_document(mismatched_source)

        mismatched_checkpoint_path = _unit_document(module)
        mismatched_checkpoint_path["registration"]["checkpoint"] = "/another-checkpoint.pt"
        with self.assertRaisesRegex(module.B2EvaluationError, "checkpoint registration path"):
            module.validate_b2_document(mismatched_checkpoint_path)

    def test_rejects_terminal_integrity_candidate_or_seed_schedule_failures(self) -> None:
        module = _module()
        terminal_fixed_step = _unit_document(module)
        decision = terminal_fixed_step["oracle_continuation"]["game_results"][0][
            "live_continuation_oracle"
        ]["oracle_decisions"][0]
        candidate = decision["candidates"][1]
        candidate["continuation_decision_round_count"] = 0
        candidate["terminal_after_fixed_joint_step"] = True
        module.validate_b2_document(terminal_fixed_step)

        terminal_with_continuation = _unit_document(module)
        candidate = terminal_with_continuation["oracle_continuation"]["game_results"][
            0
        ][
            "live_continuation_oracle"
        ]["oracle_decisions"][0]["candidates"][1]
        candidate["terminal_after_fixed_joint_step"] = True
        with self.assertRaisesRegex(
            module.B2EvaluationError, "zero continuation decision rounds"
        ):
            module.validate_b2_document(terminal_with_continuation)

        zero_without_terminal = _unit_document(module)
        candidate = zero_without_terminal["oracle_continuation"]["game_results"][0][
            "live_continuation_oracle"
        ]["oracle_decisions"][0]["candidates"][1]
        candidate["continuation_decision_round_count"] = 0
        with self.assertRaisesRegex(module.B2EvaluationError, "did not continue"):
            module.validate_b2_document(zero_without_terminal)

        exceeds_registered_bound = _unit_document(module)
        candidate = exceeds_registered_bound["oracle_continuation"]["game_results"][0][
            "live_continuation_oracle"
        ]["oracle_decisions"][0]["candidates"][1]
        candidate["continuation_decision_round_count"] = (
            module.MAX_CONTINUATION_DECISION_ROUNDS + 1
        )
        with self.assertRaisesRegex(module.B2EvaluationError, "eligibility bound"):
            module.validate_b2_document(exceeds_registered_bound)

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
        with self.assertRaisesRegex(module.B2EvaluationError, "safe raw-preserving"):
            module.validate_b2_document(malformed_selection)

        old_lowest_index_tie = _unit_document(module)
        decision = old_lowest_index_tie["oracle_continuation"]["game_results"][0]["live_continuation_oracle"]["oracle_decisions"][0]
        for candidate in decision["candidates"]:
            candidate["score"] = 0.5
            candidate["terminal"]["winner"] = None
        decision["raw_action_index"] = 1
        decision["selected_action_index"] = 0
        decision["action_index"] = 0
        decision["selected_changed_raw_action"] = True
        decision["first_restored_joint_step"]["p1"] = 0
        with self.assertRaisesRegex(module.B2EvaluationError, "safe raw-preserving"):
            module.validate_b2_document(old_lowest_index_tie)

        immediate_loss_tie = _unit_document(module)
        decision = immediate_loss_tie["oracle_continuation"]["game_results"][0]["live_continuation_oracle"]["oracle_decisions"][0]
        for candidate in decision["candidates"]:
            candidate["score"] = 0.0
            candidate["terminal"]["winner"] = "p2"
        decision["candidates"][1]["continuation_decision_round_count"] = 0
        decision["candidates"][1]["terminal_after_fixed_joint_step"] = True
        decision["raw_action_index"] = 1
        decision["selected_action_index"] = 1
        decision["action_index"] = 1
        decision["selected_changed_raw_action"] = False
        decision["first_restored_joint_step"]["p1"] = 1
        with self.assertRaisesRegex(module.B2EvaluationError, "safe raw-preserving"):
            module.validate_b2_document(immediate_loss_tie)

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

    def test_rejects_missing_or_tampered_fixed_mcts_receipts(self) -> None:
        module = _module()
        missing = _unit_document(module)
        del missing["raw"]["game_results"][0]["opponent_think"][0]["fixed_mcts"]
        with self.assertRaisesRegex(module.B2EvaluationError, "fixed_mcts"):
            module.validate_b2_document(missing)

        wrong_seed = _unit_document(module)
        wrong_seed["raw"]["game_results"][0]["opponent_think"][0]["fixed_mcts"]["audits"][0]["mcts_seed"] = 0
        with self.assertRaisesRegex(module.B2EvaluationError, "seed/work contract"):
            module.validate_b2_document(wrong_seed)

        missing_header = _unit_document(module)
        missing_header["raw"]["foulplay_think"]["fixed_mcts_audit_required"] = False
        with self.assertRaisesRegex(module.B2EvaluationError, "does not require"):
            module.validate_b2_document(missing_header)

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

    def test_accepts_the_exact_durable_continuation_cli_contract(self) -> None:
        module = _module()
        args = module.build_arg_parser().parse_args(
            [
                "--checkpoint", "/checkpoint.pt",
                "--showdown-root", "/showdown",
                "--foulplay-root", "/foulplay",
                "--foulplay-python", "/foulplay/.venv/bin/python",
                "--out", "/shared/unit.json",
                "--pokezero-player", "p1",
                "--foulplay-player", "p2",
                "--seed", "109000000",
                "--registration-seed-start", "109000000",
                "--foulplay-mcts-iterations", "10001",
                "--max-decision-rounds", "1024",
                "--max-continuation-decision-rounds", "128",
                "--expanded-continuation-decision-rounds", "1024",
                "--oracle-progress-dir", "/shared/oracle-progress",
            ]
        )
        self.assertEqual(args.registration_seed_start, 109_000_000)
        self.assertEqual(args.max_continuation_decision_rounds, 128)
        self.assertEqual(args.expanded_continuation_decision_rounds, 1024)
        self.assertEqual(args.foulplay_mcts_iterations, 10_001)
        self.assertEqual(args.oracle_progress_dir, Path("/shared/oracle-progress"))

    def test_accepts_an_explicit_fresh_orientation_band_without_reusing_the_legacy_band(self) -> None:
        module = _module()
        for seat, start in (("p1", 109_000_000), ("p2", 109_000_600)):
            with self.subTest(seat=seat):
                seed = start + 25
                registered = module._registered_unit(
                    seat=seat, seed=seed, registration_seed_start=start,
                )
                self.assertEqual(registered["seed_start"], start)
                self.assertEqual(registered["shard_index"], 1)
                module.validate_b2_document(
                    _unit_document(
                        module, seat=seat, seed=seed, registration_seed_start=start,
                    )
                )

    def test_requires_an_explicit_non_b2_identity_for_a_diagnostic_receipt(self) -> None:
        module = _module()
        diagnostic_id = "root-oracle-b2-r40-durable-gate-20260904"
        diagnostic = _unit_document(module, seat="p1", seed=119_000_000,
                                    registration_seed_start=119_000_000)
        diagnostic["experiment_id"] = diagnostic_id

        module.validate_experiment_document(
            diagnostic, expected_experiment_id=diagnostic_id,
        )
        with self.assertRaisesRegex(module.B2EvaluationError, "paired-unit experiment id"):
            module.validate_b2_document(diagnostic)
        with self.assertRaisesRegex(module.B2EvaluationError, "paired-unit experiment id"):
            module.validate_experiment_document(
                diagnostic, expected_experiment_id="root-oracle-b2-transfer-foulplay-20260831",
            )

    def test_candidate_parallelism_is_provenance_bound_for_diagnostic_and_b2_units(self) -> None:
        module = _module()
        diagnostic_id = "root-oracle-b2-r62-candidate-parallelism-20260905"
        diagnostic = _unit_document(
            module,
            seat="p1",
            seed=119_000_000,
            registration_seed_start=119_000_000,
            candidate_parallelism=3,
        )
        diagnostic["experiment_id"] = diagnostic_id
        module.validate_experiment_document(diagnostic, expected_experiment_id=diagnostic_id)
        registered_parallel = _unit_document(module, candidate_parallelism=3)
        module.validate_b2_document(registered_parallel)

        mismatched = _unit_document(
            module,
            seat="p1",
            seed=119_000_000,
            registration_seed_start=119_000_000,
            candidate_parallelism=3,
        )
        mismatched["experiment_id"] = diagnostic_id
        mismatched["oracle_continuation"]["live_continuation_oracle"][
            "candidate_parallelism"
        ] = 1
        with self.assertRaisesRegex(
            module.B2EvaluationError, "candidate parallelism"
        ):
            module.validate_experiment_document(mismatched, expected_experiment_id=diagnostic_id)

    def test_run_arm_forwards_candidate_parallelism_for_raw_and_oracle_arms(self) -> None:
        module = _module()
        args = SimpleNamespace(
            python="python",
            checkpoint=Path("/checkpoint.pt"),
            showdown_root=Path("/showdown"),
            foulplay_root=Path("/foulplay"),
            foulplay_python=Path("/foulplay/.venv/bin/python"),
            foulplay_search_time_ms=1_000,
            foulplay_mcts_iterations=10_001,
            max_decision_rounds=64,
            device="cpu",
            node_binary="node",
            candidate_parallelism=3,
            candidate_cap=9,
            max_continuation_decision_rounds=128,
            expanded_continuation_decision_rounds=1_024,
            oracle_progress_dir=None,
        )
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
            commands.append(command)
            return SimpleNamespace(returncode=0, stdout='{"summary": "ok"}', stderr="")

        with (
            patch.object(module.subprocess, "run", side_effect=fake_run),
            patch.object(module, "_arm_provenance", return_value={"bound": True}),
        ):
            module._run_arm(args, seed=123, seat="p1", oracle=False, label="raw")
            module._run_arm(args, seed=123, seat="p1", oracle=True, label="oracle")

        self.assertEqual(len(commands), 2)
        for command in commands:
            option_index = command.index("--live-continuation-oracle-candidate-parallelism")
            self.assertEqual(command[option_index + 1], "3")
        self.assertNotIn("--live-continuation-oracle", commands[0])
        self.assertIn("--live-continuation-oracle", commands[1])

    def test_accepts_an_explicit_diagnostic_experiment_cli_identity(self) -> None:
        module = _module()
        args = module.build_arg_parser().parse_args(
            [
                "--checkpoint", "/checkpoint.pt",
                "--showdown-root", "/showdown",
                "--foulplay-root", "/foulplay",
                "--foulplay-python", "/foulplay/.venv/bin/python",
                "--out", "/shared/unit.json",
                "--pokezero-player", "p1",
                "--foulplay-player", "p2",
                "--seed", "119000000",
                "--registration-seed-start", "119000000",
                "--foulplay-mcts-iterations", "10001",
                "--experiment-id", "root-oracle-b2-r40-durable-gate-20260904",
            ]
        )
        self.assertEqual(args.experiment_id, "root-oracle-b2-r40-durable-gate-20260904")

    def test_diagnostic_counterbalancing_and_raw_control_are_provenance_bound(self) -> None:
        module = _module()
        diagnostic_id = "root-oracle-b2-r42-efficacy-pilot-20260904"
        diagnostic = _unit_document(
            module,
            seat="p1",
            seed=119_000_000,
            registration_seed_start=119_000_000,
            arm_execution_order="raw-control-oracle",
            raw_reproducibility_control=True,
        )
        with self.assertRaisesRegex(module.B2EvaluationError, "registered B2 requires"):
            module.validate_b2_document(diagnostic)
        diagnostic["experiment_id"] = diagnostic_id
        module.validate_experiment_document(diagnostic, expected_experiment_id=diagnostic_id)

        bad_control = _unit_document(
            module,
            seat="p1",
            seed=119_000_000,
            registration_seed_start=119_000_000,
            arm_execution_order="raw-control-oracle",
            raw_reproducibility_control=True,
        )
        bad_control["experiment_id"] = diagnostic_id
        bad_control["raw_reproducibility_control"]["foulplay_random_seed"] = 7
        with self.assertRaisesRegex(module.B2EvaluationError, "seeds to its game seed"):
            module.validate_experiment_document(bad_control, expected_experiment_id=diagnostic_id)

        incoherent_execution = _unit_document(
            module,
            seat="p1",
            seed=119_000_000,
            registration_seed_start=119_000_000,
            arm_execution_order="raw-then-oracle",
            raw_reproducibility_control=True,
        )
        incoherent_execution["experiment_id"] = diagnostic_id
        with self.assertRaisesRegex(module.B2EvaluationError, "order and raw-control flag disagree"):
            module.validate_experiment_document(incoherent_execution, expected_experiment_id=diagnostic_id)

    def test_paired_unit_respects_three_arm_diagnostic_order_and_raw_control(self) -> None:
        module = _module()
        expected_calls = {
            "oracle-raw-control": [("oracle", True), ("raw", False), ("raw-control", False)],
            "oracle-control-raw": [("oracle", True), ("raw-control", False), ("raw", False)],
            "raw-oracle-control": [("raw", False), ("oracle", True), ("raw-control", False)],
            "control-oracle-raw": [("raw-control", False), ("oracle", True), ("raw", False)],
            "raw-control-oracle": [("raw", False), ("raw-control", False), ("oracle", True)],
            "control-raw-oracle": [("raw-control", False), ("raw", False), ("oracle", True)],
        }
        for order, expected in expected_calls.items():
            with self.subTest(order=order):
                args = SimpleNamespace(
                    seed=119_000_000,
                    pokezero_player="p1",
                    arm_execution_order=order,
                    include_raw_reproducibility_control=True,
                )
                calls: list[tuple[str, bool]] = []

                def fake_run_arm(_args: object, *, seed: int, seat: str, oracle: bool, label: str) -> dict[str, object]:
                    self.assertEqual((seed, seat), (119_000_000, "p1"))
                    calls.append((label, oracle))
                    return _bridge_summary(module, seat=seat, oracle=oracle, seed=seed)

                with patch.object(module, "_run_arm", side_effect=fake_run_arm):
                    unit = module._paired_unit(args)
                self.assertEqual(calls, expected)
                self.assertEqual(unit["raw_control_minus_raw_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
