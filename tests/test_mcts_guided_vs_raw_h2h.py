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
        recorded_action_index=0,
        observation=PublicObservation.from_observation(observation),
        history=(),
        current_legal_action_mask=tuple(observation.legal_action_mask),
        public_resolved_action_rounds=(),
        public_belief_view=dict(observation.metadata["belief_view"]),
    )
    return PublicDecisionRecord(
        **{**prototype.__dict__, "decision_id": public_decision_id(prototype)}
    )


def _branch_prior_ledger(*, fallbacks: int = 0) -> dict[str, object]:
    reasons = {reason: 0 for reason in RUNNER.BRANCH_PRIOR_FALLBACK_REASON_VALUES}
    reasons["unmapped_action"] = fallbacks
    return {
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


def _guided_for_record(record: PublicDecisionRecord, *, fallbacks: int = 0):
    return SimpleNamespace(
        latest_decision_address={
            "battle_id": record.battle_id,
            "round": record.turn_index,
            "seat": record.acting_player,
            "action_index": record.recorded_action_index,
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

    def test_writer_persists_only_the_controller_readout(self) -> None:
        candidate = SimpleNamespace(provenance_sha256="guided-provenance")
        incumbent = SimpleNamespace(provenance_sha256="raw-provenance")
        boundary = SimpleNamespace(seed=19, decision_round_index=7)
        readout = {
            "schema_version": "pokezero.mcts-sealed-override-audit.v1",
            "seed": 19,
            "battle_id": "mcts-h2h-19-p1",
            "candidate_seat": "p1",
            "decision_round_index": 7,
            "audit": {
                "schema_version": "pokezero.sealed-override-pair.v1",
                "mcts_action": 2,
                "raw_action": 1,
                "opponent_action_held_fixed": True,
                "mcts": {"terminal": {"winner": "p1", "capped": False}},
                "raw": {"terminal": {"winner": "p2", "capped": False}},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "pokezero.mcts_eval.sealed_override_audit.evaluate_measured_override_boundary",
                return_value=readout,
            ) as evaluate:
                writer = RUNNER._sealed_override_audit_writer(
                    Path(directory),
                    candidate=candidate,
                    incumbent=incumbent,
                    candidate_seat="p1",
                    env_factory=object(),
                    continuation_policy_factory=object(),
                    continuation_rollout_config=object(),
                    max_continuation_decision_rounds=400,
                )
                writer(boundary)
            evaluate.assert_called_once()
            path = RUNNER._sealed_override_audit_path(
                Path(directory), seed=19, candidate_seat="p1", decision_round_index=7
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["candidate_provenance_sha256"], "guided-provenance")
            self.assertEqual(payload["readout"]["audit"]["mcts_action"], 2)
            self.assertTrue(payload["readout"]["audit"]["opponent_action_held_fixed"])
            self.assertNotIn("opponent_action", payload["readout"]["audit"])
            RUNNER._validate_sealed_override_audit_evidence(
                Path(directory),
                SimpleNamespace(
                    seed=19,
                    candidate_seat="p1",
                    candidate=candidate,
                    incumbent=incumbent,
                    candidate_telemetry=SimpleNamespace(model_override_decisions=1),
                ),
            )

    def test_validator_rejects_a_private_opponent_action(self) -> None:
        candidate = SimpleNamespace(provenance_sha256="guided-provenance")
        incumbent = SimpleNamespace(provenance_sha256="raw-provenance")
        with tempfile.TemporaryDirectory() as directory:
            path = RUNNER._sealed_override_audit_path(
                Path(directory), seed=19, candidate_seat="p1", decision_round_index=7
            )
            DURABLE._write_immutable_json(
                path,
                {
                    "schema_version": RUNNER.SEALED_OVERRIDE_AUDIT_EVIDENCE_SCHEMA_VERSION,
                    "candidate_provenance_sha256": candidate.provenance_sha256,
                    "raw_provenance_sha256": incumbent.provenance_sha256,
                    "candidate_seat": "p1",
                    "readout": {
                        "schema_version": "pokezero.mcts-sealed-override-audit.v1",
                        "seed": 19,
                        "battle_id": "mcts-h2h-19-p1",
                        "candidate_seat": "p1",
                        "decision_round_index": 7,
                        "audit": {"opponent_action": 4, "opponent_action_held_fixed": True},
                    },
                },
            )
            with self.assertRaisesRegex(Exception, "forbidden private source state"):
                RUNNER._validate_sealed_override_audit_evidence(
                    Path(directory),
                    SimpleNamespace(
                        seed=19,
                        candidate_seat="p1",
                        candidate=candidate,
                        incumbent=incumbent,
                        candidate_telemetry=SimpleNamespace(model_override_decisions=1),
                    ),
                )


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


if __name__ == "__main__":
    unittest.main()
