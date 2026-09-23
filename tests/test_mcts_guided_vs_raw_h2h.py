"""Focused contract tests for the durable guided-MCTS-versus-raw runner."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "mcts_guided_vs_raw_h2h_test", SCRIPTS / "mcts_guided_vs_raw_h2h.py"
)
assert _SPEC is not None and _SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = RUNNER
_SPEC.loader.exec_module(RUNNER)

import mcts_mcts_h2h as DURABLE  # noqa: E402
import pokezero.engine_search as ENGINE_SEARCH  # noqa: E402
from pokezero.observation import PokeZeroObservationV0  # noqa: E402
from pokezero.public_decision_corpus import (  # noqa: E402
    PublicDecisionRecord,
    PublicObservation,
    public_decision_id,
)


_IDENTITY = {
    "checkpoint_sha256": "a" * 64,
    "source_commit": "b" * 40,
    "source_tree_sha256": "c" * 64,
    "engine_fingerprint": "d" * 64,
    "showdown_source_sha256": "e" * 64,
}


def _public_record(
    *,
    seed: int = 19,
    turn_index: int = 0,
    battle_id: str | None = None,
    recorded_action_index: int = 0,
    legal_action_mask: tuple[bool, ...] = (True, False, False, False, False, False, False, False, False),
) -> PublicDecisionRecord:
    observation = PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=legal_action_mask,
        metadata={
            "belief_view": {
                "self_slot": "p1",
                "opponent_slot": "p2",
                "self_pokemon": [],
                "opponent_pokemon": [],
            }
        },
    )
    prototype = PublicDecisionRecord(
        decision_id="pending",
        battle_id=battle_id or f"mcts-h2h-{seed}-p1",
        seed=seed,
        format_id="gen3randombattle",
        acting_player="p1",
        turn_index=turn_index,
        recorded_action_index=recorded_action_index,
        observation=PublicObservation.from_observation(observation),
        history=(),
        current_legal_action_mask=tuple(observation.legal_action_mask),
        public_resolved_action_rounds=(),
        public_belief_view=dict(observation.metadata["belief_view"]),
    )
    return PublicDecisionRecord(
        **{**prototype.__dict__, "decision_id": public_decision_id(prototype)}
    )


def _branch_prior_ledger(
    *, fallbacks: int = 0, unmapped_action_witness: dict[str, object] | None = None
) -> dict[str, object]:
    reasons = {reason: 0 for reason in RUNNER.BRANCH_PRIOR_FALLBACK_REASON_VALUES}
    reasons["unmapped_action"] = fallbacks
    ledger = {
        "schema_version": "pokezero.engine-mcts.branch-prior-fallbacks.v1",
        "native_invocations": 1,
        "belief_worlds": 1,
        "branch_prior_fallbacks": fallbacks,
        "reason_counts": reasons,
        "unclassified_branch_prior_fallbacks": 0,
        "reason_ledger_complete": True,
        "events": [
            {
                "native_invocation": 1,
                "belief_records": 1,
                "collapse_multiplicity": 1,
                "branch_prior_fallbacks": fallbacks,
                "reason_counts": reasons,
            }
        ],
    }
    if unmapped_action_witness is not None:
        ledger["unmapped_action_witness"] = unmapped_action_witness
        ledger["events"][0]["unmapped_action_witness"] = unmapped_action_witness
    return ledger


def _guided_for_record(record: PublicDecisionRecord, *, fallbacks: int = 0):
    return SimpleNamespace(
        latest_decision_address={
            "battle_id": record.battle_id,
            "round": record.turn_index,
            "seat": record.acting_player,
            "action_index": record.recorded_action_index,
            "requested_players": [record.acting_player],
        },
        latest_decision_metadata={
            "engine_mcts": {
                "override": {
                    "branch_prior_fallbacks": _branch_prior_ledger(fallbacks=fallbacks),
                    "model_argmax": record.recorded_action_index,
                    "search_argmax": record.recorded_action_index,
                    "model_override": False,
                    "unmeasured_cause": None,
                    "root_q_gap": None,
                    "root_visit_gap": None,
                    "root_gap_action_indices": [record.recorded_action_index],
                    "root_allocation": {
                        "worlds": 1,
                        "prior_authority": True,
                        "prior_cause": None,
                        "arms": [
                            {
                                "move": "tackle",
                                "action_index": record.recorded_action_index,
                                "visit_share": 1.0,
                                "q": 0.25,
                                "reported_prior": 1.0,
                                "model_prior": 1.0,
                            }
                        ],
                    },
                }
            }
        },
    )


def _terminal_model_rollout_shadow() -> dict[str, object]:
    moments = {
        "leaves": 2,
        "model_sum": 1.0,
        "rollout_sum": 1.0,
        "model_sq_sum": 0.6,
        "rollout_sq_sum": 1.0,
        "cross_sum": 0.6,
        "absolute_error_sum": 0.4,
        "squared_error_sum": 0.2,
        "concordant_pairs": 1,
        "discordant_pairs": 0,
        "tied_pairs": 0,
    }
    return {
        "value_frame": "side_one_absolute",
        "partition": "seed_ordinal_parity_v1",
        "native_invocations": 1,
        "terminal_leaf_rows": 3,
        "excluded_nonterminal_leaf_rows": 0,
        "fit": dict(moments),
        "heldout": {**moments, "leaves": 1},
        "rollouts_run": 3,
        "rollout_terminal_hits": 3,
        "rollout_cap_hits": 0,
        "rollout_dead_ends": 0,
        "rollout_trials_available": True,
        "terminal_label_available": True,
        "rollout_fallback_fraction": 0.0,
        "excluded_nonterminal_leaf_fraction": 0.0,
    }


def _guided_for_shadow_record(record: PublicDecisionRecord):
    guided = _guided_for_record(record)
    guided.latest_decision_metadata["engine_mcts"]["model_rollout_shadow"] = (
        _terminal_model_rollout_shadow()
    )
    return guided


def _public_selection(
    record: PublicDecisionRecord,
    *,
    arms: list[dict[str, object]],
    gap_actions: list[int],
    q_gap: float | None,
    visit_gap: float | None,
) -> dict[str, object]:
    return {
        "model_argmax": record.recorded_action_index,
        "search_argmax": record.recorded_action_index,
        "model_override": False,
        "unmeasured_cause": None,
        "root_q_gap": q_gap,
        "root_visit_gap": visit_gap,
        "root_gap_action_indices": gap_actions,
        "root_allocation": {
            "worlds": 1,
            "prior_authority": True,
            "prior_cause": None,
            "arms": arms,
        },
    }


class RawSpecTest(unittest.TestCase):
    def test_raw_spec_freezes_exact_no_search_selector(self) -> None:
        raw = {
            **_IDENTITY,
            "config_id": "raw-policy-v1",
            "policy_id": "raw-policy",
            "selector": dict(RUNNER.RAW_SELECTOR),
        }
        policy = RUNNER._raw_spec(raw, **_IDENTITY)
        self.assertEqual(policy.config["policy_kind"], "raw_transformer_policy")
        self.assertEqual(policy.config["selector"], RUNNER.RAW_SELECTOR)

    def test_raw_spec_rejects_selector_drift(self) -> None:
        raw = {
            **_IDENTITY,
            "config_id": "raw-policy-v1",
            "policy_id": "raw-policy",
            "selector": {**RUNNER.RAW_SELECTOR, "sampling_temperature": 0.5},
        }
        with self.assertRaisesRegex(Exception, "deterministic masked argmax"):
            RUNNER._raw_spec(raw, **_IDENTITY)


@dataclass(frozen=True)
class _FakeDecision:
    policy_id: str


class _FakeRawPolicy:
    def __init__(self) -> None:
        self.seen = []
        self.reset_calls = 0

    def select_action(self, observation, *, rng):
        self.seen.append((observation, rng))
        return _FakeDecision(policy_id="checkpoint-policy")

    def reset(self) -> None:
        self.reset_calls += 1


class RawAdapterTest(unittest.TestCase):
    def test_raw_adapter_records_only_direct_policy_work(self) -> None:
        raw = _FakeRawPolicy()
        adapter = RUNNER.DeterministicRawPolicyAdapter(raw, policy_id="raw")
        result = adapter.select_action_with_context(
            type("Context", (), {"observation": "own-observation"})(), rng="rng"
        )
        self.assertEqual(result.policy_id, "raw")
        self.assertEqual(raw.seen, [("own-observation", "rng")])
        self.assertEqual(adapter.stats.decisions, 1)
        self.assertEqual(adapter.stats.model_evals, 0)
        self.assertEqual(adapter.stats.total_iterations, 0)
        self.assertEqual(adapter.stats.prior_fallbacks, 0)
        self.assertEqual(adapter.stats.raw_forward_decisions, 1)
        self.assertGreaterEqual(adapter.stats.decision_wall_seconds, 0.0)


class StudyShapeTest(unittest.TestCase):
    def test_registered_study_requires_exact_pair_and_game_counts(self) -> None:
        manifest = {
            "study": {
                "kind": "guided_mcts_vs_raw_policy",
                "pairs": 2,
                "mirrored_games": 4,
                "failure_retry_policy": {
                    "interrupted_before_runner_terminal": "resume_same_root_with_fresh_launcher_attempt",
                    "nonzero_runner_exit": "terminal_failed_no_retry",
                    "malformed_runner_terminal": "nonbankable_no_retry",
                    "completed_game_units": "immutable_reuse_only",
                },
            }
        }
        self.assertEqual(RUNNER._validated_study(manifest, seeds=(11, 12))["pairs"], 2)
        manifest["study"]["mirrored_games"] = 3
        with self.assertRaisesRegex(Exception, "twice"):
            RUNNER._validated_study(manifest, seeds=(11, 12))


class SealedOverrideAuditContractTest(unittest.TestCase):
    @staticmethod
    def _readout(*, nested_private: bool = False) -> dict[str, object]:
        evidence: dict[str, object] = {
            "model_argmax": 1,
            "search_argmax": 2,
            "model_override": True,
            "root_q_gap": 0.25,
            "root_visit_gap": 0.5,
            "root_gap_action_indices": [2, 1],
            "root_allocation_missing_action_indices": [],
            "root_allocation": {
                "worlds": 1,
                "prior_authority": True,
                "prior_cause": None,
                "arms": [
                    {"action_index": 1, "visit_share": 0.25, "q": 0.25, "reported_prior": 0.4, "model_prior": 0.4},
                    {"action_index": 2, "visit_share": 0.75, "q": 0.5, "reported_prior": 0.6, "model_prior": 0.6},
                ],
            },
        }
        if nested_private:
            evidence["root_allocation"]["snapshot"] = "forbidden"  # type: ignore[index]
        continuation = {
            "decision_round_count": 1,
            "terminal_after_fixed_joint_step": False,
            "terminal": {"winner": "p1", "turn_count": 3, "capped": False},
        }
        return {
            "schema_version": "pokezero.mcts-sealed-override-audit.v1",
            "seed": 19,
            "battle_id": "mcts-h2h-19-p1",
            "candidate_seat": "p1",
            "decision_round_index": 7,
            "audit_status": "PAIRED",
            "audit": {
                "schema_version": "pokezero.sealed-override-pair.v1",
                "source_battle_id": "mcts-h2h-19-p1",
                "source_seed": 19,
                "source_decision_round": 7,
                "subject_player": "p1",
                "opponent_player": "p2",
                "mcts_action": 2,
                "raw_action": 1,
                "opponent_action_held_fixed": True,
                "search_evidence": evidence,
                "mcts": continuation,
                "raw": {**continuation, "terminal": {"winner": "p2", "turn_count": 4, "capped": False}},
            },
        }

    @classmethod
    def _controller_readout(cls, *, nested_private: bool = False) -> dict[str, object]:
        """Mirror the pre-binding controller shape at the sealed boundary."""

        readout = cls._readout(nested_private=nested_private)
        audit = readout["audit"]  # type: ignore[index]
        evidence = dict(audit["search_evidence"])  # type: ignore[index]
        evidence.pop("root_allocation_missing_action_indices")
        audit["search_evidence"] = evidence  # type: ignore[index]
        return readout

    @staticmethod
    def _write_source_ledger(root: Path, *, record: PublicDecisionRecord) -> None:
        """Write the public-ledger selection the controller must bind to."""

        evidence = SealedOverrideAuditContractTest._readout()["audit"]["search_evidence"]  # type: ignore[index]
        selection = {**evidence, "unmeasured_cause": None}
        DURABLE._write_immutable_json(
            RUNNER._branch_prior_ledger_path(
                root, seed=record.seed, candidate_seat="p1", record=record
            ),
            {"selection": selection, "request_boundary": {"requested_players": ["p1", "p2"]}},
        )

    def test_optional_audit_contract_requires_the_registered_raw_selector(self) -> None:
        self.assertIsNone(RUNNER._sealed_override_audit_config({}))
        manifest = {
            "sealed_override_audit": {
                "schema_version": RUNNER.SEALED_OVERRIDE_AUDIT_EVIDENCE_SCHEMA_VERSION,
                "continuation_selector": dict(RUNNER.RAW_SELECTOR),
                "max_continuation_decision_rounds": 400,
            }
        }
        self.assertEqual(
            RUNNER._sealed_override_audit_config(manifest).max_continuation_decision_rounds,
            400,
        )
        manifest["sealed_override_audit"]["continuation_selector"] = {
            **RUNNER.RAW_SELECTOR,
            "search": True,
        }
        with self.assertRaisesRegex(Exception, "registered deterministic raw selector"):
            RUNNER._sealed_override_audit_config(manifest)

    def test_root_action_audit_contract_requires_fixed_trials_and_predeclared_roots(self) -> None:
        self.assertIsNone(RUNNER._sealed_root_action_audit_config({}, seeds=(19,)))
        manifest = {
            "sealed_root_action_audit": {
                "schema_version": RUNNER.SEALED_ROOT_ACTION_AUDIT_EVIDENCE_SCHEMA_VERSION,
                "targets": [{"seed": 19, "candidate_seat": "p1", "decision_round_index": 7}],
                "continuation_rng_seeds": list(range(100, 116)),
                "max_continuation_decision_rounds": 400,
                "continuation_targets": {
                    "policy_consistent": {
                        "subject": "sampled_raw_transformer",
                        "opponent": "sampled_raw_transformer",
                    },
                    "uniform_own": {
                        "subject": "uniform_legal",
                        "opponent": "sampled_raw_transformer",
                    },
                },
            }
        }
        config = RUNNER._sealed_root_action_audit_config(manifest, seeds=(19,))
        self.assertEqual(config.continuation_rng_seeds, tuple(range(100, 116)))
        self.assertEqual(config.targets[0].decision_round_index, 7)
        manifest["sealed_root_action_audit"]["continuation_rng_seeds"] = list(range(100, 115))
        with self.assertRaisesRegex(Exception, "exactly sixteen"):
            RUNNER._sealed_root_action_audit_config(manifest, seeds=(19,))

    def test_writer_persists_only_the_controller_readout(self) -> None:
        candidate = SimpleNamespace(provenance_sha256="guided-provenance")
        incumbent = SimpleNamespace(provenance_sha256="raw-provenance")
        boundary = SimpleNamespace(seed=19, decision_round_index=7)
        record = _public_record(
            seed=19,
            turn_index=7,
            recorded_action_index=2,
            legal_action_mask=(False, True, True, False, False, False, False, False, False),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "pokezero.mcts_eval.sealed_override_audit.evaluate_measured_override_boundary",
                return_value=self._controller_readout(),
            ) as evaluate:
                pre_step_writer, public_writer = RUNNER._sealed_override_audit_writer(
                    root,
                    candidate=candidate,
                    incumbent=incumbent,
                    candidate_seat="p1",
                    env_factory=object(),
                    continuation_policy_factory_builder=object(),
                    continuation_rollout_config=object(),
                    max_continuation_decision_rounds=400,
                )
                # The sealed pre-step hook cannot write the durable sidecar
                # yet: its matching public record/ledger does not exist until
                # after ``env.step`` commits the action.
                pre_step_writer(boundary)
                self.assertFalse((root / "sealed-override-audits").exists())
                self._write_source_ledger(root, record=record)
                public_writer(record)
                pre_step_writer(boundary)
                public_writer(record)
            self.assertEqual(evaluate.call_count, 2)
            path = RUNNER._sealed_override_audit_path(
                root, seed=19, candidate_seat="p1", decision_round_index=7
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["candidate_provenance_sha256"], "guided-provenance")
            self.assertEqual(payload["readout"]["audit"]["mcts_action"], 2)
            self.assertTrue(payload["readout"]["audit"]["opponent_action_held_fixed"])
            self.assertNotIn("opponent_action", payload["readout"]["audit"])
            self.assertEqual(
                payload["readout"]["audit"]["search_evidence"][
                    "root_allocation_missing_action_indices"
                ],
                [],
            )
            self.assertEqual(path.read_text(encoding="utf-8"), path.read_text(encoding="utf-8"))

    def test_writer_rejects_nested_private_controller_data_before_immutable_write(self) -> None:
        candidate = SimpleNamespace(provenance_sha256="guided-provenance")
        incumbent = SimpleNamespace(provenance_sha256="raw-provenance")
        boundary = SimpleNamespace(seed=19, decision_round_index=7)
        record = _public_record(
            seed=19,
            turn_index=7,
            recorded_action_index=2,
            legal_action_mask=(False, True, True, False, False, False, False, False, False),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "pokezero.mcts_eval.sealed_override_audit.evaluate_measured_override_boundary",
                return_value=self._controller_readout(nested_private=True),
            ):
                pre_step_writer, public_writer = RUNNER._sealed_override_audit_writer(
                    root, candidate=candidate, incumbent=incumbent, candidate_seat="p1",
                    env_factory=object(), continuation_policy_factory_builder=object(),
                    continuation_rollout_config=object(), max_continuation_decision_rounds=400,
                )
                pre_step_writer(boundary)
                self._write_source_ledger(root, record=record)
                with self.assertRaisesRegex(Exception, "disagrees with its public source ledger"):
                    public_writer(record)
            self.assertFalse((Path(directory) / "sealed-override-audits").exists())

    def test_validator_requires_sidecar_selection_to_match_measured_public_override(self) -> None:
        candidate = SimpleNamespace(provenance_sha256="guided-provenance")
        incumbent = SimpleNamespace(provenance_sha256="raw-provenance")
        record = _public_record(
            turn_index=7,
            recorded_action_index=2,
            legal_action_mask=(False, True, True, False, False, False, False, False, False),
        )
        source_selection = RUNNER._validated_selection_evidence(
            {
                "model_argmax": 1,
                "search_argmax": 2,
                "model_override": True,
                "unmeasured_cause": None,
                "root_q_gap": 0.25,
                "root_visit_gap": 0.5,
                "root_gap_action_indices": [2, 1],
                "root_allocation": {
                    "worlds": 1,
                    "prior_authority": True,
                    "prior_cause": None,
                    "arms": [
                        {"action_index": 1, "visit_share": 0.25, "q": 0.25, "reported_prior": 0.4, "model_prior": 0.4},
                        {"action_index": 2, "visit_share": 0.75, "q": 0.5, "reported_prior": 0.6, "model_prior": 0.6},
                    ],
                },
            },
            record=record,
        )
        game = SimpleNamespace(
            seed=19,
            candidate_seat="p1",
            candidate=candidate,
            incumbent=incumbent,
            candidate_telemetry=SimpleNamespace(model_override_decisions=1),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger_path = RUNNER._branch_prior_ledger_path(
                root, seed=19, candidate_seat="p1", record=record
            )
            DURABLE._write_immutable_json(
                ledger_path,
                {
                    "selection": source_selection,
                    "request_boundary": {"requested_players": ["p1", "p2"]},
                },
            )
            sidecar = RUNNER._sealed_override_audit_path(
                root, seed=19, candidate_seat="p1", decision_round_index=7
            )
            payload = RUNNER._sealed_override_audit_payload(
                candidate=candidate,
                incumbent=incumbent,
                candidate_seat="p1",
                readout=self._readout(),
            )
            DURABLE._write_immutable_json(sidecar, payload)
            with patch.object(RUNNER, "_validate_public_decision_evidence", return_value=(record,)):
                RUNNER._validate_sealed_override_audit_evidence(root, game)
            # A measured override on a forced one-sided phase cannot hold an
            # opponent action fixed.  It still needs a canonical sidecar so
            # the game-level denominator proves that the audit did not omit
            # the event merely because no paired continuation is possible.
            inapplicable_root = root / "inapplicable"
            inapplicable_ledger = RUNNER._branch_prior_ledger_path(
                inapplicable_root, seed=19, candidate_seat="p1", record=record
            )
            DURABLE._write_immutable_json(
                inapplicable_ledger,
                {
                    "selection": source_selection,
                    "request_boundary": {"requested_players": ["p1"]},
                },
            )
            inapplicable_readout = {
                "schema_version": "pokezero.mcts-sealed-override-audit.v1",
                "seed": 19,
                "battle_id": "mcts-h2h-19-p1",
                "candidate_seat": "p1",
                "decision_round_index": 7,
                "audit_status": "INAPPLICABLE_NON_SIMULTANEOUS",
                "requested_players": ["p1"],
                "search_evidence": RUNNER._sealed_search_evidence_from_selection(source_selection),
            }
            inapplicable_sidecar = RUNNER._sealed_override_audit_path(
                inapplicable_root, seed=19, candidate_seat="p1", decision_round_index=7
            )
            DURABLE._write_immutable_json(
                inapplicable_sidecar,
                RUNNER._sealed_override_audit_payload(
                    candidate=candidate,
                    incumbent=incumbent,
                    candidate_seat="p1",
                    readout=inapplicable_readout,
                ),
            )
            with patch.object(RUNNER, "_validate_public_decision_evidence", return_value=(record,)):
                RUNNER._validate_sealed_override_audit_evidence(inapplicable_root, game)
            bad_root = root / "bad"
            bad_readout = self._readout()
            bad_audit = bad_readout["audit"]  # type: ignore[index]
            bad_evidence = bad_audit["search_evidence"]  # type: ignore[index]
            bad_audit["mcts_action"] = 1
            bad_audit["raw_action"] = 2
            bad_evidence["model_argmax"] = 2
            bad_evidence["search_argmax"] = 1
            bad_ledger_path = RUNNER._branch_prior_ledger_path(
                bad_root, seed=19, candidate_seat="p1", record=record
            )
            DURABLE._write_immutable_json(
                bad_ledger_path,
                {
                    "selection": source_selection,
                    "request_boundary": {"requested_players": ["p1", "p2"]},
                },
            )
            bad_sidecar = RUNNER._sealed_override_audit_path(
                bad_root, seed=19, candidate_seat="p1", decision_round_index=7
            )
            DURABLE._write_immutable_json(
                bad_sidecar,
                RUNNER._sealed_override_audit_payload(
                    candidate=candidate, incumbent=incumbent, candidate_seat="p1", readout=bad_readout
                ),
            )
            with patch.object(RUNNER, "_validate_public_decision_evidence", return_value=(record,)):
                with self.assertRaisesRegex(Exception, "does not bind its measured public decision"):
                    RUNNER._validate_sealed_override_audit_evidence(bad_root, game)
            downgrade_root = root / "downgrade"
            downgrade_ledger = RUNNER._branch_prior_ledger_path(
                downgrade_root, seed=19, candidate_seat="p1", record=record
            )
            DURABLE._write_immutable_json(
                downgrade_ledger,
                {
                    "selection": source_selection,
                    "request_boundary": {"requested_players": ["p1", "p2"]},
                },
            )
            downgrade_sidecar = RUNNER._sealed_override_audit_path(
                downgrade_root, seed=19, candidate_seat="p1", decision_round_index=7
            )
            DURABLE._write_immutable_json(
                downgrade_sidecar,
                RUNNER._sealed_override_audit_payload(
                    candidate=candidate,
                    incumbent=incumbent,
                    candidate_seat="p1",
                    readout=inapplicable_readout,
                ),
            )
            with patch.object(RUNNER, "_validate_public_decision_evidence", return_value=(record,)):
                with self.assertRaisesRegex(Exception, "does not bind its measured public decision boundary"):
                    RUNNER._validate_sealed_override_audit_evidence(downgrade_root, game)


class DurableLauncherHandoffTest(unittest.TestCase):
    def test_guided_runner_binds_its_own_immutable_launcher_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory) / "out"
            attempt_id = "guided-attempt"
            receipt = out_dir / "launcher-attempts" / f"{attempt_id}.json"
            receipt.parent.mkdir(parents=True)
            runner_script = Path(RUNNER.__file__).resolve()
            DURABLE._write_immutable_json(
                receipt,
                {
                    "schema_version": DURABLE.DURABLE_LAUNCHER_ATTEMPT_SCHEMA_VERSION,
                    "attempt_id": attempt_id,
                    "runner_script": str(runner_script),
                    "runner_script_sha256": DURABLE._sha256_file(runner_script),
                    "writer_lock": str(out_dir / "runner-writer.lock"),
                },
            )
            lock_path = out_dir / "runner-writer.lock"
            lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            environment = {
                DURABLE.DURABLE_LAUNCHER_ATTEMPT_ID_ENV: attempt_id,
                DURABLE.DURABLE_LAUNCHER_OUT_DIR_ENV: str(out_dir),
                DURABLE.DURABLE_LAUNCHER_ATTEMPT_RECEIPT_ENV: str(receipt),
                DURABLE.DURABLE_LAUNCHER_WRITER_LOCK_FD_ENV: str(lock_fd),
            }
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.dict(os.environ, environment, clear=False):
                    RUNNER._require_durable_launcher_handoff(
                        out_dir, runner_script=runner_script
                    )
                with patch.dict(os.environ, environment, clear=False):
                    with self.assertRaisesRegex(Exception, "does not bind this scorer"):
                        RUNNER._require_durable_launcher_handoff(
                            out_dir, runner_script=SCRIPTS / "mcts_mcts_h2h.py"
                        )
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)

    def test_guided_main_passes_its_own_script_to_handoff_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory) / "out"
            manifest = Path(directory) / "manifest.json"
            with (
                patch.object(RUNNER, "_require_durable_launcher_handoff") as handoff,
                patch.object(RUNNER, "_load_manifest", side_effect=RuntimeError("stop after handoff")),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after handoff"):
                    RUNNER.main(
                        [
                            "--checkpoint",
                            str(Path(directory) / "checkpoint.pt"),
                            "--showdown-root",
                            str(Path(directory) / "showdown"),
                            "--manifest",
                            str(manifest),
                            "--out-dir",
                            str(out_dir),
                        ]
            )
            handoff.assert_called_once_with(
                out_dir.resolve(), runner_script=Path(RUNNER.__file__)
            )


class GuidedConfigTest(unittest.TestCase):
    def test_guided_configuration_is_exact_not_a_budget_lookalike(self) -> None:
        config = dict(RUNNER.REGISTERED_ENGINE_CONFIG)
        RUNNER._require_registered_candidate_config(config)
        config["model_priors"] = False
        with self.assertRaisesRegex(Exception, "enable own model priors"):
            RUNNER._require_registered_candidate_config(config)

    def test_unlisted_engine_knob_cannot_drift(self) -> None:
        config = dict(RUNNER.REGISTERED_ENGINE_CONFIG)
        config["approximate_sleep_turns"] = False
        with self.assertRaisesRegex(Exception, "registered guided-MCTS"):
            RUNNER._require_registered_candidate_config(config)

    def test_registered_deep_cuda_protocol_is_accepted_exactly(self) -> None:
        config = dict(RUNNER.REGISTERED_DEEP_ENGINE_CONFIG)
        RUNNER._require_registered_candidate_config(config)
        config["search_sims"] = 4095
        with self.assertRaisesRegex(Exception, "registered guided-MCTS"):
            RUNNER._require_registered_candidate_config(config)

    def test_registered_deep_opponent_prior_protocol_is_accepted_exactly(self) -> None:
        config = dict(RUNNER.REGISTERED_DEEP_OPPONENT_PRIOR_ENGINE_CONFIG)
        baseline = RUNNER.REGISTERED_DEEP_ENGINE_CONFIG
        self.assertEqual(
            {
                key: value
                for key, value in config.items()
                if baseline.get(key) != value
            },
            {"use_opponent_priors": True},
            "the opponent-model ablation may differ from fixed-work deep MCTS only on opponent priors",
        )
        self.assertEqual(set(config), set(baseline))
        RUNNER._require_registered_candidate_config(config)
        config["search_sims"] = 4095
        with self.assertRaisesRegex(Exception, "registered guided-MCTS"):
            RUNNER._require_registered_candidate_config(config)
        for invalid in (1, 0):
            with self.subTest(invalid=invalid):
                config = dict(RUNNER.REGISTERED_DEEP_OPPONENT_PRIOR_ENGINE_CONFIG)
                config["use_opponent_priors"] = invalid
                with self.assertRaisesRegex(Exception, "must be a boolean"):
                    RUNNER._require_registered_candidate_config(config)

    def test_registered_selective_order_opponent_prior_protocol_is_accepted_exactly(self) -> None:
        config = dict(RUNNER.REGISTERED_DEEP_OPPONENT_PRIOR_SELECTIVE_ORDER_ENGINE_CONFIG)
        baseline = RUNNER.REGISTERED_DEEP_OPPONENT_PRIOR_ENGINE_CONFIG
        self.assertEqual(
            {
                key: value
                for key, value in config.items()
                if baseline.get(key) != value
            },
            {
                "allow_lost_active_permutation_opponent_root_fallback": True,
                "allow_lost_active_permutation_opponent_prior_omission": True,
            },
            "the selective-order replay may differ from the strict opponent-prior arm only on its named recovery rule",
        )
        self.assertEqual(set(config), set(baseline))
        RUNNER._require_registered_candidate_config(config)
        config["allow_lost_active_permutation_opponent_prior_omission"] = "only-this-status"
        with self.assertRaisesRegex(Exception, "registered guided-MCTS"):
            RUNNER._require_registered_candidate_config(config)

    def test_deep_opponent_prior_manifest_inherits_cuda_without_overriding_it(self) -> None:
        manifest_config = dict(RUNNER.REGISTERED_DEEP_OPPONENT_PRIOR_ENGINE_CONFIG)
        manifest_config.pop("model_device")
        # The durable manifest correctly does not carry this runtime-only
        # default.  Reproduce the source-bound rollout-leaf failure path:
        # runtime materialization must restore False and registration must
        # accept the resulting exact, fail-closed configuration.
        manifest_config.pop("allow_lost_active_permutation_opponent_root_fallback")
        manifest_config.pop("allow_lost_active_permutation_opponent_prior_omission")
        raw = {
            "config_id": "guided-mcts-own-priors-fullwork-deep-d6-s4096-opponent-priors",
            "policy_id": "guided-mcts-own-priors-fullwork-deep-d6-s4096",
            "checkpoint_sha256": _IDENTITY["checkpoint_sha256"],
            "source_commit": _IDENTITY["source_commit"],
            "engine_fingerprint": _IDENTITY["engine_fingerprint"],
            "engine_config": manifest_config,
        }
        policy, _ = DURABLE._runtime_spec(
            raw,
            role="guided candidate",
            checkpoint="/tmp/checkpoint",
            checkpoint_sha256=_IDENTITY["checkpoint_sha256"],
            source_commit=_IDENTITY["source_commit"],
            source_tree_sha256=_IDENTITY["source_tree_sha256"],
            engine_fingerprint=_IDENTITY["engine_fingerprint"],
            showdown_source_sha256=_IDENTITY["showdown_source_sha256"],
            model_path="/tmp/model",
            tables_path="/tmp/tables",
            device="cuda",
        )
        self.assertEqual(policy.config["model_device"], "cuda")
        self.assertIs(
            policy.config["allow_lost_active_permutation_opponent_root_fallback"],
            False,
        )
        self.assertIs(
            policy.config["allow_lost_active_permutation_opponent_prior_omission"],
            False,
        )
        self.assertNotIn("model_device", manifest_config)
        self.assertNotIn(
            "allow_lost_active_permutation_opponent_root_fallback",
            manifest_config,
        )
        self.assertNotIn(
            "allow_lost_active_permutation_opponent_prior_omission",
            manifest_config,
        )
        RUNNER._require_registered_candidate_config(policy.config)

    def test_registered_deep_rollout_leaf_ablation_is_accepted_exactly(self) -> None:
        config = dict(RUNNER.REGISTERED_DEEP_ROLLOUT_LEAF_ENGINE_CONFIG)
        baseline = RUNNER.REGISTERED_DEEP_ENGINE_CONFIG
        self.assertEqual(
            {
                key: value
                for key, value in config.items()
                if baseline.get(key) != value
            },
            {
                "rollout_leaf_eval": True,
                "rollout_threads": 12,
                "rollout_threads_cpu_budget_ack": True,
            },
            "the rollout-leaf protocol may differ from fixed-work deep MCTS only on leaf evaluation and its CPU budget acknowledgement",
        )
        self.assertEqual(set(config), set(baseline))
        RUNNER._require_registered_candidate_config(config)
        config["rollout_threads"] = 11
        with self.assertRaisesRegex(Exception, "registered guided-MCTS"):
            RUNNER._require_registered_candidate_config(config)

    def test_registered_model_leaf_shadow_is_accepted_exactly(self) -> None:
        config = dict(RUNNER.REGISTERED_DEEP_MODEL_LEAF_SHADOW_ENGINE_CONFIG)
        baseline = RUNNER.REGISTERED_DEEP_ENGINE_CONFIG
        self.assertEqual(
            {
                key: value
                for key, value in config.items()
                if baseline.get(key) != value
            },
            {
                "rollout_count": 1,
                "rollout_leaf_shadow": True,
                "rollout_max_plies": 1000,
                "rollout_threads": 12,
                "rollout_threads_cpu_budget_ack": True,
            },
        )
        RUNNER._require_registered_candidate_config(config)
        config["rollout_count"] = 2
        with self.assertRaisesRegex(Exception, "registered guided-MCTS"):
            RUNNER._require_registered_candidate_config(config)


class GuidedProgressTest(unittest.TestCase):
    def test_guided_progress_uses_its_own_schema_without_weakening_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress" / "current.json"
            payload = {"schema_version": RUNNER.PROGRESS_SCHEMA_VERSION, "event": "game_started"}
            DURABLE._write_progress_json(
                path,
                payload,
                schema_version=RUNNER.PROGRESS_SCHEMA_VERSION,
            )
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)
            with self.assertRaisesRegex(Exception, "wrong schema version"):
                DURABLE._write_progress_json(
                    path,
                    {"schema_version": DURABLE.PROGRESS_SCHEMA_VERSION, "event": "game_started"},
                    schema_version=RUNNER.PROGRESS_SCHEMA_VERSION,
                )


class PublicDecisionEvidenceTest(unittest.TestCase):
    def test_branch_prior_ledger_refuses_contradictory_event_attribution(self) -> None:
        ledger = _branch_prior_ledger(fallbacks=2)
        ledger["unclassified_branch_prior_fallbacks"] = 3
        ledger["branch_prior_fallbacks"] = 5
        ledger["reason_ledger_complete"] = False
        ledger["events"][0]["branch_prior_fallbacks"] = 5
        ledger["events"][0]["reason_counts"] = {
            **{reason: 0 for reason in RUNNER.BRANCH_PRIOR_FALLBACK_REASON_VALUES},
            "unmapped_action": 5,
        }
        with self.assertRaisesRegex(Exception, "attribution disagrees"):
            RUNNER._validated_branch_prior_ledger(ledger)

    def test_branch_prior_ledger_refuses_forged_reason_vocabulary(self) -> None:
        ledger = _branch_prior_ledger(fallbacks=2)
        ledger["reason_counts"] = {"forged_reason": 2}
        ledger["events"][0]["reason_counts"] = {"forged_reason": 2}
        with self.assertRaisesRegex(Exception, "complete native reason vocabulary"):
            RUNNER._validated_branch_prior_ledger(ledger)

    def test_branch_prior_ledger_retains_valid_unmapped_action_witness(self) -> None:
        witness = {
            "acting": {"nodes": 2, "move_arms": 1, "switch_arms": 1, "none_arms": 0},
            "opponent": {"nodes": 0, "move_arms": 0, "switch_arms": 0, "none_arms": 0},
        }
        ledger = _branch_prior_ledger(fallbacks=2, unmapped_action_witness=witness)

        self.assertEqual(
            RUNNER._validated_branch_prior_ledger(ledger)["unmapped_action_witness"],
            witness,
        )

    def test_branch_prior_ledger_refuses_unmapped_witness_that_miscounts_nodes(self) -> None:
        witness = {
            "acting": {"nodes": 1, "move_arms": 1, "switch_arms": 0, "none_arms": 0},
            "opponent": {"nodes": 0, "move_arms": 0, "switch_arms": 0, "none_arms": 0},
        }
        ledger = _branch_prior_ledger(fallbacks=2, unmapped_action_witness=witness)

        with self.assertRaisesRegex(Exception, "does not match its native invocation count"):
            RUNNER._validated_branch_prior_ledger(ledger)

    def test_selection_evidence_refuses_search_action_not_bound_to_public_record(self) -> None:
        record = _public_record()
        override = _guided_for_record(record).latest_decision_metadata["engine_mcts"]["override"]
        override = {**override, "search_argmax": record.recorded_action_index + 1}
        with self.assertRaisesRegex(Exception, "does not match its public decision"):
            RUNNER._selection_evidence_from_override(override, record=record)

    def test_selection_evidence_refuses_missing_root_q_gap_field(self) -> None:
        record = _public_record()
        override = _guided_for_record(record).latest_decision_metadata["engine_mcts"]["override"]
        override = {key: value for key, value in override.items() if key != "root_q_gap"}
        with self.assertRaisesRegex(Exception, "complete selection evidence"):
            RUNNER._selection_evidence_from_override(override, record=record)

    def test_selection_evidence_refuses_forged_unmeasured_cause(self) -> None:
        record = _public_record()
        selection = _public_selection(
            record,
            arms=[
                {
                    "action_index": record.recorded_action_index,
                    "visit_share": 1.0,
                    "q": 0.25,
                    "reported_prior": 1.0,
                    "model_prior": 1.0,
                }
            ],
            gap_actions=[record.recorded_action_index],
            q_gap=None,
            visit_gap=None,
        )
        selection["unmeasured_cause"] = "forged"
        with self.assertRaisesRegex(Exception, "does not match its root allocation cause"):
            RUNNER._validated_selection_evidence(selection, record=record)

    def test_selection_evidence_refuses_an_arm_outside_public_legal_actions(self) -> None:
        record = _public_record()
        override = _guided_for_record(record).latest_decision_metadata["engine_mcts"]["override"]
        override = {**override, "root_allocation": {**override["root_allocation"]}}
        override["root_allocation"]["arms"] = [
            {**override["root_allocation"]["arms"][0], "action_index": 8}
        ]
        with self.assertRaisesRegex(Exception, "not a public legal action"):
            RUNNER._selection_evidence_from_override(override, record=record)

    def test_forced_singleton_projects_no_choice_without_hiding_real_choice_mismatch(self) -> None:
        """A one-action request cannot carry a meaningful MCTS override claim.

        The engine can retain hidden-world arms which do not name that public
        action.  They must not terminate a multi-hour run after the rollout has
        already committed the forced public action, but the resulting sidecar
        must stay explicitly unmeasured and contain no synthetic prior or Q.
        """
        record = _public_record(
            legal_action_mask=(True, False, False, False, False, False, False, False, False)
        )
        override = _guided_for_record(record).latest_decision_metadata["engine_mcts"]["override"]
        override = {
            **override,
            "model_argmax": None,
            "model_override": None,
            "unmeasured_cause": "no_root_priors",
            "root_q_gap": None,
            "root_visit_gap": None,
            "root_gap_action_indices": [],
            "root_allocation": {
                "worlds": 1,
                "prior_authority": False,
                "prior_cause": "no_root_priors",
                "arms": [
                    {
                        **override["root_allocation"]["arms"][0],
                        "action_index": None,
                        "visit_share": 0.25,
                        "q": 0.75,
                        "reported_prior": None,
                        "model_prior": None,
                    },
                    {
                        **override["root_allocation"]["arms"][0],
                        "move": "switch hidden-world-only",
                        "action_index": 8,
                        "visit_share": 0.75,
                        "q": -0.5,
                        "reported_prior": None,
                        "model_prior": None,
                    },
                ],
            },
        }
        evidence = RUNNER._selection_evidence_from_override(override, record=record)
        self.assertEqual(evidence["unmeasured_cause"], "no_root_priors")
        self.assertEqual(evidence["model_argmax"], None)
        self.assertEqual(evidence["model_override"], None)
        self.assertEqual(evidence["root_allocation_missing_action_indices"], [])
        self.assertEqual(
            evidence["root_allocation"]["arms"],
            [
                {
                    "action_index": 0,
                    "visit_share": 1.0,
                    "q": None,
                    "reported_prior": None,
                    "model_prior": None,
                }
            ],
        )

    def test_selection_evidence_names_missing_legal_actions_without_aborting_the_game(self) -> None:
        """A partial engine root is evidence of a vocabulary seam, not a lost run.

        The selected and model actions remain bound to actual native arms, while
        the public-only sidecar makes every legal action the native root omitted
        explicit.  This is the failure shape observed in the R3 audit at a late
        seed-2026092006 decision.
        """
        record = _public_record(
            legal_action_mask=(True, True, True, False, False, False, False, False, False)
        )
        selection = _public_selection(
            record,
            arms=[
                {"action_index": 0, "visit_share": 0.6, "q": 0.2, "reported_prior": 0.4, "model_prior": 0.4},
                {"action_index": 1, "visit_share": 0.4, "q": 0.3, "reported_prior": 0.6, "model_prior": 0.6},
            ],
            gap_actions=[0, 1],
            q_gap=0.1,
            visit_gap=0.2,
        )
        selection.update({"model_argmax": 1, "model_override": True})
        evidence = RUNNER._validated_selection_evidence(selection, record=record)
        self.assertEqual(evidence["root_allocation_missing_action_indices"], [2])
        self.assertEqual(
            RUNNER._sealed_search_evidence_from_selection(evidence)[
                "root_allocation_missing_action_indices"
            ],
            [2],
        )

        forged = {**evidence, "root_allocation_missing_action_indices": []}
        with self.assertRaisesRegex(Exception, "missing-action coverage disagrees"):
            RUNNER._validated_selection_evidence(forged, record=record)

    def test_selection_evidence_rejects_gap_witness_outside_native_root(self) -> None:
        record = _public_record(
            legal_action_mask=(True, True, True, False, False, False, False, False, False)
        )
        selection = _public_selection(
            record,
            arms=[
                {"action_index": 0, "visit_share": 0.6, "q": 0.2, "reported_prior": 0.4, "model_prior": 0.4},
                {"action_index": 1, "visit_share": 0.4, "q": 0.3, "reported_prior": 0.6, "model_prior": 0.6},
            ],
            gap_actions=[2, 0],
            q_gap=0.1,
            visit_gap=0.2,
        )
        with self.assertRaisesRegex(Exception, "gap witness is absent from its root allocation"):
            RUNNER._validated_selection_evidence(selection, record=record)

    def test_selection_evidence_uses_engine_top_pair_witness_for_ties_and_zero_arms(self) -> None:
        tied = _public_record(
            legal_action_mask=(True, True, True, False, False, False, False, False, False)
        )
        tied_arms = [
            {"action_index": 0, "visit_share": 0.5, "q": 0.1, "reported_prior": 0.2, "model_prior": 0.2},
            {"action_index": 1, "visit_share": 0.5, "q": 0.4, "reported_prior": 0.3, "model_prior": 0.3},
            {"action_index": 2, "visit_share": 0.0, "q": 0.9, "reported_prior": 0.5, "model_prior": 0.5},
        ]
        tie = _public_selection(
            tied, arms=tied_arms, gap_actions=[1, 0], q_gap=0.3, visit_gap=0.0
        )
        self.assertEqual(
            RUNNER._validated_selection_evidence(tie, record=tied)["root_gap_action_indices"],
            [1, 0],
        )
        one_visited = _public_record(
            legal_action_mask=(True, True, False, False, False, False, False, False, False)
        )
        one_visited_arms = [
            {"action_index": 0, "visit_share": 1.0, "q": 0.1, "reported_prior": 0.4, "model_prior": 0.4},
            {"action_index": 1, "visit_share": 0.0, "q": 0.9, "reported_prior": 0.6, "model_prior": 0.6},
        ]
        RUNNER._validated_selection_evidence(
            _public_selection(
                one_visited, arms=one_visited_arms, gap_actions=[0], q_gap=None, visit_gap=None
            ),
            record=one_visited,
        )

    def test_selection_evidence_refuses_a_nonleading_gap_pair(self) -> None:
        record = _public_record(
            legal_action_mask=(True, True, True, False, False, False, False, False, False)
        )
        arms = [
            {"action_index": 0, "visit_share": 0.5, "q": 0.1, "reported_prior": 0.2, "model_prior": 0.2},
            {"action_index": 1, "visit_share": 0.3, "q": 0.4, "reported_prior": 0.3, "model_prior": 0.3},
            {"action_index": 2, "visit_share": 0.2, "q": 0.3, "reported_prior": 0.5, "model_prior": 0.5},
        ]
        with self.assertRaisesRegex(Exception, "does not name the leading"):
            RUNNER._validated_selection_evidence(
                _public_selection(record, arms=arms, gap_actions=[1, 2], q_gap=0.1, visit_gap=0.1),
                record=record,
            )

    def test_writer_and_validator_bind_each_guided_decision_immutably(self) -> None:
        candidate = SimpleNamespace(provenance_sha256="guided-provenance")
        incumbent = SimpleNamespace(provenance_sha256="raw-provenance")
        record = _public_record()
        game = SimpleNamespace(
            seed=record.seed,
            candidate_seat="p1",
            candidate=candidate,
            incumbent=incumbent,
            candidate_telemetry=SimpleNamespace(decisions=1),
        )
        with tempfile.TemporaryDirectory() as directory:
            writer = RUNNER._public_decision_writer(
                Path(directory),
                candidate=candidate,
                incumbent=incumbent,
                seed=record.seed,
                candidate_seat="p1",
                guided_policy=_guided_for_record(record),
            )
            writer(record)
            writer(record)  # a resumed game must accept the identical prior unit
            ledger_path = RUNNER._branch_prior_ledger_path(
                Path(directory), seed=record.seed, candidate_seat="p1", record=record
            )
            ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(
                ledger_payload["public_decision"]["decision_id"], record.decision_id
            )
            self.assertEqual(
                ledger_payload["branch_prior_fallbacks"]["branch_prior_fallbacks"], 0
            )
            self.assertEqual(ledger_payload["selection"]["search_argmax"], record.recorded_action_index)
            self.assertNotIn("move", ledger_payload["selection"]["root_allocation"]["arms"][0])
            self.assertEqual(
                RUNNER._validate_public_decision_evidence(Path(directory), game),
                (record,),
            )

    def test_writer_refuses_to_bind_one_decision_to_another_decision_metadata(self) -> None:
        candidate = SimpleNamespace(provenance_sha256="guided-provenance")
        incumbent = SimpleNamespace(provenance_sha256="raw-provenance")
        record = _public_record()
        mismatched = _guided_for_record(record)
        mismatched.latest_decision_address = {
            **mismatched.latest_decision_address,
            "round": record.turn_index + 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            writer = RUNNER._public_decision_writer(
                Path(directory),
                candidate=candidate,
                incumbent=incumbent,
                seed=record.seed,
                candidate_seat="p1",
                guided_policy=mismatched,
            )
            with self.assertRaisesRegex(Exception, "not bound to the public decision"):
                writer(record)

    def test_shadow_writer_requires_and_binds_terminal_leaf_evidence(self) -> None:
        candidate = SimpleNamespace(
            provenance_sha256="guided-provenance", config={"rollout_leaf_shadow": True}
        )
        incumbent = SimpleNamespace(provenance_sha256="raw-provenance")
        record = _public_record()
        game = SimpleNamespace(
            seed=record.seed,
            candidate_seat="p1",
            candidate=candidate,
            incumbent=incumbent,
            candidate_telemetry=SimpleNamespace(decisions=1),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            RUNNER._public_decision_writer(
                root,
                candidate=candidate,
                incumbent=incumbent,
                seed=record.seed,
                candidate_seat="p1",
                guided_policy=_guided_for_shadow_record(record),
            )(record)
            sidecar = RUNNER._model_rollout_shadow_path(
                root, seed=record.seed, candidate_seat="p1", record=record
            )
            self.assertTrue(sidecar.is_file())
            self.assertEqual(RUNNER._validate_public_decision_evidence(root, game), (record,))

    def test_shadow_writer_preserves_but_excludes_nonterminal_leaf_rows(self) -> None:
        candidate = SimpleNamespace(
            provenance_sha256="guided-provenance", config={"rollout_leaf_shadow": True}
        )
        incumbent = SimpleNamespace(provenance_sha256="raw-provenance")
        record = _public_record()
        guided = _guided_for_shadow_record(record)
        shadow = guided.latest_decision_metadata["engine_mcts"]["model_rollout_shadow"]
        shadow["rollout_terminal_hits"] = 2
        shadow["rollout_cap_hits"] = 1
        shadow["terminal_leaf_rows"] = 2
        shadow["excluded_nonterminal_leaf_rows"] = 1
        shadow["fit"]["leaves"] = 1
        shadow["heldout"]["leaves"] = 1
        shadow["rollout_fallback_fraction"] = 1 / 3
        shadow["excluded_nonterminal_leaf_fraction"] = 1 / 3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            RUNNER._public_decision_writer(
                root,
                candidate=candidate,
                incumbent=incumbent,
                seed=record.seed,
                candidate_seat="p1",
                guided_policy=guided,
            )(record)
            sidecar = RUNNER._model_rollout_shadow_path(
                root, seed=record.seed, candidate_seat="p1", record=record
            )
            written = json.loads(sidecar.read_text(encoding="utf-8"))["model_rollout_shadow"]
            self.assertEqual(written["excluded_nonterminal_leaf_rows"], 1)

    def test_shadow_validator_refuses_forged_zero_trial_derived_fields(self) -> None:
        shadow = {
            "value_frame": "side_one_absolute",
            "partition": "seed_ordinal_parity_v1",
            "native_invocations": 1,
            "terminal_leaf_rows": 0,
            "excluded_nonterminal_leaf_rows": 0,
            "fit": {field: 0 for field in ENGINE_SEARCH.MODEL_ROLLOUT_SHADOW_MOMENT_FIELDS},
            "heldout": {field: 0 for field in ENGINE_SEARCH.MODEL_ROLLOUT_SHADOW_MOMENT_FIELDS},
            "rollouts_run": 0,
            "rollout_terminal_hits": 0,
            "rollout_cap_hits": 0,
            "rollout_dead_ends": 0,
            "rollout_trials_available": True,
            "terminal_label_available": False,
            "rollout_fallback_fraction": 0.0,
            "excluded_nonterminal_leaf_fraction": None,
        }
        with self.assertRaisesRegex(RUNNER.HeadToHeadError, "derived field does not match"):
            RUNNER._validated_model_rollout_shadow(shadow)

    def test_validator_refuses_incomplete_guided_decision_evidence(self) -> None:
        candidate = SimpleNamespace(provenance_sha256="guided-provenance")
        incumbent = SimpleNamespace(provenance_sha256="raw-provenance")
        record = _public_record()
        game = SimpleNamespace(
            seed=record.seed,
            candidate_seat="p1",
            candidate=candidate,
            incumbent=incumbent,
            candidate_telemetry=SimpleNamespace(decisions=2),
        )
        with tempfile.TemporaryDirectory() as directory:
            RUNNER._public_decision_writer(
                Path(directory),
                candidate=candidate,
                incumbent=incumbent,
                seed=record.seed,
                candidate_seat="p1",
                guided_policy=_guided_for_record(record),
            )(record)
            with self.assertRaisesRegex(Exception, "equal guided decision telemetry"):
                RUNNER._validate_public_decision_evidence(Path(directory), game)

    def test_validator_refuses_record_from_another_battle_with_the_same_seed(self) -> None:
        candidate = SimpleNamespace(provenance_sha256="guided-provenance")
        incumbent = SimpleNamespace(provenance_sha256="raw-provenance")
        record = _public_record(battle_id="mcts-h2h-19-p2")
        game = SimpleNamespace(
            seed=record.seed,
            candidate_seat="p1",
            candidate=candidate,
            incumbent=incumbent,
            candidate_telemetry=SimpleNamespace(decisions=1),
        )
        with tempfile.TemporaryDirectory() as directory:
            RUNNER._public_decision_writer(
                Path(directory),
                candidate=candidate,
                incumbent=incumbent,
                seed=record.seed,
                candidate_seat="p1",
                guided_policy=_guided_for_record(record),
            )(record)
            with self.assertRaisesRegex(Exception, "completed game identity"):
                RUNNER._validate_public_decision_evidence(Path(directory), game)


class CompletedGameEvidenceTest(unittest.TestCase):
    def test_root_prior_fallback_is_not_bankable(self) -> None:
        clean = SimpleNamespace(
            fallback_decisions=0,
            root_prior_fallbacks=0,
            searched_decisions=0,
            model_evals=0,
            total_iterations=0,
            worlds_constructed=0,
            worlds_searched=0,
        )
        game = SimpleNamespace(candidate_telemetry=clean, incumbent_telemetry=clean)
        RUNNER._validate_completed_game(game)
        failed_root = SimpleNamespace(**{**clean.__dict__, "root_prior_fallbacks": 1})
        with self.assertRaisesRegex(Exception, "root policy-prior"):
            RUNNER._validate_completed_game(
                SimpleNamespace(candidate_telemetry=failed_root, incumbent_telemetry=clean)
            )

    def test_raw_search_counter_is_not_accepted_as_direct_policy_work(self) -> None:
        clean = SimpleNamespace(
            fallback_decisions=0,
            root_prior_fallbacks=0,
            searched_decisions=0,
            model_evals=0,
            total_iterations=0,
            worlds_constructed=0,
            worlds_searched=0,
        )
        raw_with_search = SimpleNamespace(**{**clean.__dict__, "searched_decisions": 1})
        with self.assertRaisesRegex(Exception, "recorded search work"):
            RUNNER._validate_completed_game(
                SimpleNamespace(candidate_telemetry=clean, incumbent_telemetry=raw_with_search)
            )

    def test_rollout_leaf_candidate_requires_realized_witness(self) -> None:
        clean = SimpleNamespace(
            fallback_decisions=0,
            root_prior_fallbacks=0,
            searched_decisions=0,
            model_evals=0,
            total_iterations=0,
            worlds_constructed=0,
            worlds_searched=0,
        )
        rollout_candidate = SimpleNamespace(config={"rollout_leaf_eval": True})
        with self.assertRaisesRegex(Exception, "realized rollout witness"):
            RUNNER._validate_completed_game(
                SimpleNamespace(
                    candidate=rollout_candidate,
                    candidate_telemetry=clean,
                    incumbent_telemetry=clean,
                )
            )

    def test_summary_requires_a_live_guided_root_witness(self) -> None:
        summary = {
            "candidate_root_prior_fallbacks": 0,
            "incumbent_root_prior_fallbacks": 0,
            "candidate_override_measured_decisions": 2,
            "incumbent_model_evals": 0,
            "incumbent_iterations": 0,
        }
        RUNNER._validate_summary_evidence(summary)
        summary["candidate_override_measured_decisions"] = 0
        with self.assertRaisesRegex(Exception, "live root model-action"):
            RUNNER._validate_summary_evidence(summary)

    def test_rollout_leaf_summary_requires_terminal_partition(self) -> None:
        summary = {
            "candidate_root_prior_fallbacks": 0,
            "incumbent_root_prior_fallbacks": 0,
            "candidate_override_measured_decisions": 2,
            "incumbent_model_evals": 0,
            "incumbent_iterations": 0,
            "candidate": {"config": {"rollout_leaf_eval": True}},
            "candidate_rollout_leaf_modes": {"rollout": 8},
            "candidate_rollouts_run": 32,
            "candidate_rollout_terminal_hits": 31,
            "candidate_rollout_cap_hits": 1,
            "candidate_rollout_dead_ends": 0,
        }
        RUNNER._validate_summary_evidence(summary)
        summary["candidate_rollout_terminal_hits"] = 30
        with self.assertRaisesRegex(Exception, "do not partition"):
            RUNNER._validate_summary_evidence(summary)

        summary["candidate_rollout_terminal_hits"] = 31
        for malformed in (32.5, True):
            summary["candidate_rollouts_run"] = malformed
            with self.assertRaisesRegex(Exception, "malformed terminal partition"):
                RUNNER._validate_summary_evidence(summary)


class SealedRootActionAuditWriterTest(unittest.TestCase):
    def test_writer_runs_only_manifest_selected_boundary_and_persists_no_private_state(self) -> None:
        config = RUNNER.SealedRootActionAuditConfig(
            targets=(RUNNER.SealedRootActionAuditTarget(19, "p1", 7),),
            continuation_rng_seeds=tuple(range(100, 116)),
            max_continuation_decision_rounds=400,
        )
        candidate = SimpleNamespace(provenance_sha256="guided-provenance")
        incumbent = SimpleNamespace(provenance_sha256="raw-provenance")
        readout = {
            "schema_version": "pokezero.sealed-root-action-audit.v1",
            "seed": 19,
            "battle_id": "mcts-h2h-19-p1",
            "candidate_seat": "p1",
            "decision_round_index": 7,
            "audit_status": "PAIRED",
            "audit": {"complete": "grid"},
        }
        boundary = SimpleNamespace(seed=19, decision_round_index=7, snapshot=object())
        record = _public_record(
            seed=19,
            turn_index=7,
            recorded_action_index=4,
            legal_action_mask=(False, True, True, False, True, False, False, True, False),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "pokezero.mcts_eval.sealed_root_action_audit.evaluate_root_action_boundary",
            return_value=readout,
        ) as evaluate:
            root = Path(directory)
            pre_step_writer, public_writer = RUNNER._sealed_root_action_audit_writer(
                root,
                candidate=candidate,
                incumbent=incumbent,
                candidate_seat="p1",
                config=config,
                env_factory=object(),
                continuation_policy_factory_builder=lambda _, __: {"policy_consistent": object()},
                continuation_rollout_config=object(),
            )
            pre_step_writer(SimpleNamespace(seed=19, decision_round_index=6))
            evaluate.assert_not_called()
            pre_step_writer(boundary)
            public_writer(record)
            path = RUNNER._sealed_root_action_audit_path(
                root, seed=19, candidate_seat="p1", decision_round_index=7
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(payload["candidate_provenance_sha256"], "guided-provenance")
        self.assertEqual(payload["readout"], readout)
        self.assertNotIn("snapshot", payload)
        self.assertNotIn("opponent_action", payload)


if __name__ == "__main__":
    unittest.main()
