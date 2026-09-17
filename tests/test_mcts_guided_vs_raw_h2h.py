"""Focused contract tests for the durable guided-MCTS-versus-raw runner."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


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


_IDENTITY = {
    "checkpoint_sha256": "a" * 64,
    "source_commit": "b" * 40,
    "source_tree_sha256": "c" * 64,
    "engine_fingerprint": "d" * 64,
    "showdown_source_sha256": "e" * 64,
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
        with self.assertRaisesRegex(Exception, "registered one-second"):
            RUNNER._require_registered_candidate_config(config)


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
