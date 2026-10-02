"""Focused safety checks for the source-bound fixed-work parity runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mcts_fixed_work_parity_runner_under_test",
    ROOT / "scripts" / "run_mcts_source_bound_fixed_work_parity.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class _Record:
    decision_id = "fixed-work-root"

    def to_payload(self):
        return {"decision_id": self.decision_id, "public": "frozen"}


def _arm(arm: str, *, action: str = "move 1", iterations: int = 1024) -> dict[str, object]:
    engine = {
        "leaf_eval": "model",
        "worlds_constructed": 4,
        "worlds_searched": 4,
    }
    if arm == "parallel":
        engine["world_parallelism"] = {"workers": 2, "mode": "fixed_work"}
    return {
        "arm": arm,
        "decision_id": "fixed-work-root",
        "corpus_record_sha256": runner.legacy.canonical_json_sha256(_Record().to_payload()),
        "root_action": action,
        "outer_wall_ms": 1.0,
        "fallbacks": 0,
        "prior_fallbacks": 0,
        "invalid_actions": 0,
        "total_iterations": iterations,
        "model_evals": 64,
        "engine_mcts": engine,
    }


class FixedWorkParityRunnerSafetyTest(unittest.TestCase):
    def test_parallel_worker_count_must_be_a_real_second_arm(self) -> None:
        base = [
            "--checkpoint", "/checkpoint.pt", "--expected-checkpoint-sha256", "a" * 64,
            "--showdown-root", "/showdown", "--corpus", "/corpus.jsonl",
            "--expected-corpus-sha256", "b" * 64,
            "--expected-corpus-file-sha256", "c" * 64,
            "--source-receipt", "/receipt.json", "--expected-showdown-source-sha256", "d" * 64,
            "--out-root", "/out",
        ]
        self.assertEqual(runner._parse_args(base + ["--parallel-workers", "2"]).parallel_workers, 2)
        with self.assertRaises(SystemExit):
            runner._parse_args(base + ["--parallel-workers", "1"])
        with self.assertRaises(SystemExit):
            runner._parse_args(base + ["--parallel-workers", "5"])

    def test_model_priors_are_explicit_and_default_to_legacy_contract(self) -> None:
        base = [
            "--checkpoint", "/checkpoint.pt", "--expected-checkpoint-sha256", "a" * 64,
            "--showdown-root", "/showdown", "--corpus", "/corpus.jsonl",
            "--expected-corpus-sha256", "b" * 64,
            "--expected-corpus-file-sha256", "c" * 64,
            "--source-receipt", "/receipt.json", "--expected-showdown-source-sha256", "d" * 64,
            "--out-root", "/out",
        ]
        self.assertFalse(runner._parse_args(base).model_priors)
        self.assertTrue(runner._parse_args(base + ["--model-priors"]).model_priors)
        self.assertFalse(runner._parse_args(base + ["--no-model-priors"]).model_priors)

    def test_model_priors_reaches_both_fixed_work_arms(self) -> None:
        args = runner._parse_args([
            "--checkpoint", "/checkpoint.pt", "--expected-checkpoint-sha256", "a" * 64,
            "--showdown-root", "/showdown", "--corpus", "/corpus.jsonl",
            "--expected-corpus-sha256", "b" * 64,
            "--expected-corpus-file-sha256", "c" * 64,
            "--source-receipt", "/receipt.json", "--expected-showdown-source-sha256", "d" * 64,
            "--out-root", "/out", "--model-priors",
        ])
        with mock.patch.object(runner.legacy, "_LiveEngineTimingDecider") as decider:
            runner._new_decider(object(), args, workers=1)
        self.assertTrue(decider.call_args.kwargs["model_priors"])

    def test_matching_fixed_work_arms_are_accepted(self) -> None:
        args = runner._parse_args([
            "--checkpoint", "/checkpoint.pt", "--expected-checkpoint-sha256", "a" * 64,
            "--showdown-root", "/showdown", "--corpus", "/corpus.jsonl",
            "--expected-corpus-sha256", "b" * 64,
            "--expected-corpus-file-sha256", "c" * 64,
            "--source-receipt", "/receipt.json", "--expected-showdown-source-sha256", "d" * 64,
            "--out-root", "/out", "--parallel-workers", "2",
        ])
        runner._validate_row(
            {"serial": _arm("serial"), "parallel": _arm("parallel")},
            record=_Record(),
            args=args,
        )

    def test_action_divergence_cannot_be_accepted_as_parity(self) -> None:
        args = runner._parse_args([
            "--checkpoint", "/checkpoint.pt", "--expected-checkpoint-sha256", "a" * 64,
            "--showdown-root", "/showdown", "--corpus", "/corpus.jsonl",
            "--expected-corpus-sha256", "b" * 64,
            "--expected-corpus-file-sha256", "c" * 64,
            "--source-receipt", "/receipt.json", "--expected-showdown-source-sha256", "d" * 64,
            "--out-root", "/out", "--parallel-workers", "2",
        ])
        with self.assertRaisesRegex(runner.FixedWorkParityError, "root actions diverged"):
            runner._validate_row(
                {"serial": _arm("serial"), "parallel": _arm("parallel", action="move 2")},
                record=_Record(),
                args=args,
            )

    def test_changed_work_accounting_cannot_hide_behind_action_agreement(self) -> None:
        args = runner._parse_args([
            "--checkpoint", "/checkpoint.pt", "--expected-checkpoint-sha256", "a" * 64,
            "--showdown-root", "/showdown", "--corpus", "/corpus.jsonl",
            "--expected-corpus-sha256", "b" * 64,
            "--expected-corpus-file-sha256", "c" * 64,
            "--source-receipt", "/receipt.json", "--expected-showdown-source-sha256", "d" * 64,
            "--out-root", "/out", "--parallel-workers", "2",
        ])
        with self.assertRaisesRegex(runner.FixedWorkParityError, "total_iterations differs"):
            runner._validate_row(
                {"serial": _arm("serial"), "parallel": _arm("parallel", iterations=1023)},
                record=_Record(),
                args=args,
            )
