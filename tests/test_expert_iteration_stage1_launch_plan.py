"""Coverage for the non-executing, reader-revalidated Stage-1 planner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy

from tests.test_expert_iteration_label_corpus import corpus_tool, fixture as corpus_fixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "expert_iteration_stage1_launch_plan.py"
SPEC = importlib.util.spec_from_file_location("stage1_launch_plan", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def arm(identifier: str, kind: str, *, frozen: bool, seed_start: int) -> dict[str, object]:
    layers = [] if kind == "linear" else ([256] if kind == "mlp2" else [256, 256])
    conversion = None if kind == "linear" else (43010001 if kind == "mlp2" else 43010002)
    return {
        "id": identifier,
        "head": {"kind": kind, "hidden_layers": layers, "conversion_seed": conversion},
        "freeze_non_value_parameters": frozen,
        "freeze_policy_heads": not frozen,
        "training_seeds": [seed_start, seed_start + 1, seed_start + 2],
    }


def recipe(*, source_bank_sha256: str) -> dict[str, object]:
    return {
        "schema": planner.RECIPE_SCHEMA,
        "recipe_id": planner.RECIPE_ID,
        "status": "REGISTERED-NOT-RUN",
        "source": {
            "checkpoint": {"path": "/test-data/champion.pt", "sha256": "b" * 64},
            "label_corpus": {
                "required_schema": planner.CORPUS_SCHEMA,
                "required_receipt_schema": planner.CORPUS_RECEIPT_SCHEMA,
                "model_input_hash_schema": planner.MODEL_INPUT_HASH_SCHEMA,
                "source_bank_sha256": source_bank_sha256,
                "target": "policy_continuation_value",
                "reader_verification_required": True,
            },
        },
        "engine_contract": {"optimizer": {"weight_decay": 0.01}},
        "training": {
            "batch_size": 128,
            "max_epochs": 48,
            "learning_rate_grid": [3e-5, 1e-4, 3e-4],
            "learning_rate_schedule": "constant",
            "value_loss_weight": 1.0,
            "value_ranking_loss_weight": 0.0,
            "max_grad_norm": 1.0,
            "amp": "bf16",
            "early_stopping": {"metric": "mae", "split": "heldout", "patience_epochs": 6},
        },
        "arms": [
            arm("linear-frozen", "linear", frozen=True, seed_start=100),
            arm("linear-finetuned", "linear", frozen=False, seed_start=103),
            arm("mlp2-frozen", "mlp2", frozen=True, seed_start=106),
            arm("mlp2-finetuned", "mlp2", frozen=False, seed_start=109),
            arm("mlp3-frozen", "mlp3", frozen=True, seed_start=112),
            arm("mlp3-finetuned", "mlp3", frozen=False, seed_start=115),
        ],
        "launch_gate": {"requires": ["p0.1-independent-audit-verdict-not-refuted"]},
    }


class Stage1LaunchPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = corpus_fixture(self.root)
        self.receipt_path = self.root / "corpus-receipt.json"
        write_json(
            self.receipt_path,
            corpus_tool.build_receipt(
                bank_path=self.paths["bank"],
                contract_path=self.paths["contract"],
                split_path=self.paths["splits"],
                corpus_path=self.paths["corpus"],
                train_cache_path=self.paths["train"],
                heldout_cache_path=self.paths["heldout"],
            ),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def registered_recipe(self) -> dict[str, object]:
        return recipe(source_bank_sha256=hashlib.sha256(self.paths["bank"].read_bytes()).hexdigest())

    def test_enumerates_exactly_the_registered_eighteen_runs_without_execution(self) -> None:
        plan = planner.build_plan(
            recipe=self.registered_recipe(), corpus_receipt_path=self.receipt_path, run_root=self.root / "stage1"
        )
        self.assertEqual(plan["status"], "BLOCKED_PENDING_REGISTERED_P0_GATES")
        self.assertEqual(plan["execution"], "NOT_EXECUTED_BY_THIS_TOOL")
        self.assertEqual(plan["registered_run_count"], 18)
        self.assertEqual(len(plan["head_conversions"]), 2)
        runs = plan["runs"]
        self.assertTrue(all("--keep-cache-after-read" in run["argv"] for run in runs))
        self.assertTrue(all("--value-selection-data" in run["argv"] for run in runs))
        frozen = next(run for run in runs if run["arm"] == "linear-frozen")
        finetuned = next(run for run in runs if run["arm"] == "linear-finetuned")
        self.assertIn("--freeze-non-value-parameters", frozen["argv"])
        self.assertIn("--freeze-policy-heads", finetuned["argv"])

    def test_reruns_reader_and_refuses_a_changed_cache(self) -> None:
        returns = numpy.load(self.paths["train"] / "returns.npy")
        returns[0] = 0.25
        numpy.save(self.paths["train"] / "returns.npy", returns)
        with self.assertRaisesRegex(planner.Stage1LaunchRefusal, "reader revalidation failed"):
            planner.build_plan(
                recipe=self.registered_recipe(), corpus_receipt_path=self.receipt_path, run_root=self.root / "stage1"
            )

    def test_refuses_a_receipt_that_only_claims_validation(self) -> None:
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        receipt["summary"]["bank_candidates"] += 1
        write_json(self.receipt_path, receipt)
        with self.assertRaisesRegex(planner.Stage1LaunchRefusal, "exact current output of reader revalidation"):
            planner.build_plan(
                recipe=self.registered_recipe(), corpus_receipt_path=self.receipt_path, run_root=self.root / "stage1"
            )

    def test_rejects_the_superseded_cache_input_hash_schema(self) -> None:
        registered = self.registered_recipe()
        registered["source"]["label_corpus"]["model_input_hash_schema"] = "pokezero.training-cache-model-input.v1"
        with self.assertRaisesRegex(planner.Stage1LaunchRefusal, "registered replay-bound cache receipt"):
            planner.build_plan(
                recipe=registered, corpus_receipt_path=self.receipt_path, run_root=self.root / "stage1"
            )

    def test_main_binds_the_exact_recipe_bytes_and_writes_once(self) -> None:
        recipe_path = self.root / "recipe.json"
        registered = self.registered_recipe()
        write_json(recipe_path, registered)
        expected = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
        out = self.root / "stage1-plan.json"
        with mock.patch.object(planner, "RECIPE_SHA256", expected):
            self.assertEqual(planner.main([
                "--recipe", str(recipe_path), "--corpus-receipt", str(self.receipt_path),
                "--run-root", str(self.root / "stage1"), "--out", str(out),
            ]), 0)
            self.assertEqual(planner.main([
                "--recipe", str(recipe_path), "--corpus-receipt", str(self.receipt_path),
                "--run-root", str(self.root / "different-stage1"), "--out", str(out),
            ]), 2)


if __name__ == "__main__":
    unittest.main()
