"""Fail-closed provenance checks for converted-head Phase-3 value tuning."""

from __future__ import annotations

import hashlib
import io
import json
import copy
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from pokezero import neural_cli
from pokezero.neural_policy import (
    EntityTokenTransformerPolicy,
    TransformerEpochMetrics,
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    save_transformer_checkpoint,
    torch_available,
)


CONVERTER = Path(__file__).resolve().parent.parent / "scripts" / "convert_value_head.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(hidden: int | None = None) -> TransformerPolicyConfig:
    return TransformerPolicyConfig.compact_category(
        category_vocab=tuple(range(1, 17)),
        category_oov_buckets=4,
        policy_id="phase3-value-tune-test",
        window_size=2,
        token_type_vocab_size=8,
        categorical_feature_count=1,
        numeric_feature_count=1,
        embedding_dim=16,
        transformer_layers=1,
        attention_heads=4,
        feedforward_dim=32,
        dropout=0.0,
        value_head_hidden=hidden,
    )


class FoundationValueTunePhase3Test(unittest.TestCase):
    def setUp(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.pt"
        self.converted = self.root / "converted.pt"
        source_config = _config()
        result = TransformerTrainingResult(
            model_config=source_config,
            training_config=TransformerTrainingConfig(objective="ppo", window_size=2),
            epochs=(TransformerEpochMetrics(epoch=1, examples=1, loss=0.5, policy_loss=-0.1, policy_accuracy=0.4, value_loss=0.25),),
        )
        save_transformer_checkpoint(self.source, EntityTokenTransformerPolicy(source_config), result=result)
        completed = subprocess.run(
            [
                sys.executable,
                str(CONVERTER),
                "--checkpoint",
                str(self.source),
                "--output",
                str(self.converted),
                "--value-head-hidden",
                "32",
                "--head-init-seed",
                "2718",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.train = self.root / "train.jsonl"
        self.selection = self.root / "selection.jsonl"
        self.calibration = self.root / "calibration.jsonl"
        for path, value in ((self.train, "train"), (self.selection, "selection"), (self.calibration, "calibration")):
            path.write_text(value + "\n", encoding="utf-8")

    def _args(self, **overrides: object) -> SimpleNamespace:
        values: dict[str, object] = {
            "converted_initial_checkpoint": self.converted,
            "converted_initial_checkpoint_sha256": _sha256(self.converted),
            "source_checkpoint_sha256": _sha256(self.source),
            "phase3_value_head_init_seed": 2718,
            "expected_value_head_hidden": 32,
            "phase3_training_seed": 31415,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _provenance(self) -> dict[str, object]:
        return dict(
            neural_cli._foundation_value_tune_phase3_provenance(
                self._args(),
                source_checkpoint=self.source,
                train_paths=[self.train],
                selection_paths=[self.selection],
                calibration_paths=[self.calibration],
            )
        )

    def _recipe(self, out_dir: Path) -> dict[str, object]:
        manifest_path = self.root / "manifest.json"
        summary_path = self.root / "neural-foundation-run-summary.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "latest_accepted_checkpoint_path": str(self.source),
                    "iterations": [
                        {
                            "iteration": 1,
                            "checkpoint_path": str(self.source),
                            "training_rollout_paths": [str(self.train)],
                            "value_selection_training_rollout_paths": [str(self.selection)],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": "pokezero.neural_foundation_run_summary.v1",
                    "status": "passed",
                    "recipe": {"run_dir": str(self.root), "manifest_path": str(manifest_path)},
                }
            ),
            encoding="utf-8",
        )
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            exit_code = neural_cli.main(
                [
                    "foundation-value-tune-plan",
                    str(summary_path),
                    "--out-dir",
                    str(out_dir),
                    "--calibration-data",
                    str(self.calibration),
                    "--converted-initial-checkpoint",
                    str(self.converted),
                    "--converted-initial-checkpoint-sha256",
                    _sha256(self.converted),
                    "--source-checkpoint-sha256",
                    _sha256(self.source),
                    "--phase3-value-head-init-seed",
                    "2718",
                    "--expected-value-head-hidden",
                    "32",
                    "--phase3-training-seed",
                    "31415",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0)
        return json.loads(stdout.getvalue())

    def test_provenance_binds_a_deterministic_converted_head_and_every_input(self) -> None:
        provenance = self._provenance()

        self.assertEqual(provenance["source_checkpoint"]["sha256"], _sha256(self.source))
        self.assertEqual(provenance["converted_initial_checkpoint"]["sha256"], _sha256(self.converted))
        self.assertEqual(provenance["phase3_value_head_initialization"]["seed"], 2718)
        self.assertEqual(provenance["training_seed"], 31415)
        self.assertEqual(provenance["converted_input_frozen_trunk"]["value_head_hidden"], 32)
        self.assertEqual(
            [entry["path"] for entry in provenance["inputs"]["training"]],
            [str(self.train)],
        )

    def test_provenance_refuses_a_converted_checkpoint_that_moves_the_trunk(self) -> None:
        import torch

        altered = self.root / "trunk-moved.pt"
        payload = torch.load(self.converted, map_location="cpu", weights_only=False)
        trunk_name = next(name for name in payload["state_dict"] if not name.startswith("value_head."))
        payload["state_dict"][trunk_name].add_(1.0)
        torch.save(payload, altered)
        args = self._args(
            converted_initial_checkpoint=altered,
            converted_initial_checkpoint_sha256=_sha256(altered),
        )

        with self.assertRaisesRegex(ValueError, "changed non-value tensor"):
            neural_cli._foundation_value_tune_phase3_provenance(
                args,
                source_checkpoint=self.source,
                train_paths=[self.train],
                selection_paths=[self.selection],
                calibration_paths=[self.calibration],
            )

    def test_provenance_refuses_a_converted_checkpoint_with_changed_model_config_or_seed_stamp(self) -> None:
        import torch

        for field, value, error in (
            ("model_config.dropout", 0.75, "model_config outside value_head_hidden"),
            ("phase3_value_head_initialization.seed", 9999, "initialization seed"),
        ):
            altered = self.root / f"altered-{field.replace('.', '-')}.pt"
            payload = torch.load(self.converted, map_location="cpu", weights_only=False)
            if field == "model_config.dropout":
                payload["model_config"]["dropout"] = value
            else:
                payload["phase3_value_head_initialization"]["seed"] = value
            torch.save(payload, altered)
            with self.assertRaisesRegex(ValueError, error):
                neural_cli._foundation_value_tune_phase3_provenance(
                    self._args(
                        converted_initial_checkpoint=altered,
                        converted_initial_checkpoint_sha256=_sha256(altered),
                    ),
                    source_checkpoint=self.source,
                    train_paths=[self.train],
                    selection_paths=[self.selection],
                    calibration_paths=[self.calibration],
                )

    def test_provenance_rederives_the_seeded_head_bytes(self) -> None:
        import torch

        altered = self.root / "seed-stamp-but-wrong-head.pt"
        payload = torch.load(self.converted, map_location="cpu", weights_only=False)
        payload["state_dict"]["value_head.0.weight"].add_(1.0)
        torch.save(payload, altered)
        with self.assertRaisesRegex(ValueError, "does not match its registered seed"):
            neural_cli._foundation_value_tune_phase3_provenance(
                self._args(
                    converted_initial_checkpoint=altered,
                    converted_initial_checkpoint_sha256=_sha256(altered),
                ),
                source_checkpoint=self.source,
                train_paths=[self.train],
                selection_paths=[self.selection],
                calibration_paths=[self.calibration],
            )

    def test_phase3_seed_zero_matches_the_converter_contract(self) -> None:
        seed_zero = self.root / "converted-seed-zero.pt"
        completed = subprocess.run(
            [
                sys.executable,
                str(CONVERTER),
                "--checkpoint",
                str(self.source),
                "--output",
                str(seed_zero),
                "--value-head-hidden",
                "32",
                "--head-init-seed",
                "0",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        provenance = neural_cli._foundation_value_tune_phase3_provenance(
            self._args(
                converted_initial_checkpoint=seed_zero,
                converted_initial_checkpoint_sha256=_sha256(seed_zero),
                phase3_value_head_init_seed=0,
            ),
            source_checkpoint=self.source,
            train_paths=[self.train],
            selection_paths=[self.selection],
            calibration_paths=[self.calibration],
        )
        self.assertEqual(provenance["phase3_value_head_initialization"]["seed"], 0)

    def test_frozen_trunk_comparison_requires_exact_shape_dtype_and_signed_zero_bytes(self) -> None:
        import torch

        self.assertFalse(
            neural_cli._foundation_value_tune_tensor_bytes_identical(
                torch.tensor([1.0, 2.0], dtype=torch.float32),
                torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            )
        )
        self.assertFalse(
            neural_cli._foundation_value_tune_tensor_bytes_identical(
                torch.tensor([1.0], dtype=torch.float32),
                torch.tensor([1.0], dtype=torch.float64),
            )
        )
        self.assertFalse(
            neural_cli._foundation_value_tune_tensor_bytes_identical(
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([-0.0], dtype=torch.float32),
            )
        )

    def test_calibration_witness_refuses_empty_or_impossible_pearson_evidence(self) -> None:
        bins = [
            {
                "lower": -1.0 + (2.0 * index / 10),
                "upper": -1.0 + (2.0 * (index + 1) / 10),
                "count": 0,
                "mean_prediction": 0.0,
                "mean_return": 0.0,
                "calibration_error": 0.0,
            }
            for index in range(10)
        ]
        report = {
            "examples": 0,
            "mse": 0.0,
            "mae": 0.0,
            "bias": 0.0,
            "sign_accuracy": 0.0,
            "expected_calibration_error": 0.0,
            "pearson_correlation": None,
            "bins": bins,
            "slices": [],
        }
        with self.assertRaisesRegex(ValueError, "positive integer"):
            neural_cli._foundation_value_tune_require_calibration_report(
                report, expected_bins=10, label="empty calibration report"
            )

        report["examples"] = 1
        report["pearson_correlation"] = 0.5
        report["bins"][5]["count"] = 1
        with self.assertRaisesRegex(ValueError, "fewer than two examples"):
            neural_cli._foundation_value_tune_require_calibration_report(
                report, expected_bins=10, label="one-row calibration report"
            )

    def test_completion_requires_private_input_snapshots(self) -> None:
        recipe = {"phase3_provenance": self._provenance()}

        with self.assertRaisesRegex(ValueError, "private snapshots"):
            neural_cli._foundation_value_tune_verify_phase3_completion(recipe)

    def test_converted_head_mode_refuses_partial_provenance_arguments(self) -> None:
        args = self._args(converted_initial_checkpoint_sha256=None)

        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            neural_cli._foundation_value_tune_phase3_provenance(
                args,
                source_checkpoint=self.source,
                train_paths=[self.train],
                selection_paths=[self.selection],
                calibration_paths=[self.calibration],
            )

    def test_plan_routes_the_provenance_bound_converted_checkpoint_to_train(self) -> None:
        out_dir = self.root / "phase3-output"
        recipe = self._recipe(out_dir)
        argv = recipe["command"]["argv"]
        self.assertEqual(argv[argv.index("--initial-checkpoint") + 1], str(self.converted))
        self.assertEqual(recipe["candidate_checkpoint_path"], str(self.source))
        self.assertEqual(recipe["initial_checkpoint_path"], str(self.converted))
        self.assertEqual(recipe["phase3_provenance"]["source_checkpoint"]["sha256"], _sha256(self.source))
        self.assertEqual(argv[argv.index("--training-seed") + 1], "31415")
        self.assertEqual(argv[argv.index("--window-size") + 1], "2")
        self.assertEqual(recipe["config"]["window_size"], 2)

    def test_output_checkpoint_must_self_describe_the_selected_value_only_run(self) -> None:
        output = self.root / "misdeclared-output.pt"
        model, converted_result = neural_cli.load_transformer_checkpoint(self.converted, map_location="cpu")
        torch = neural_cli.require_torch()
        with torch.no_grad():
            model.value_head[0].weight.add_(0.125)
        metrics = TransformerEpochMetrics(
            epoch=1,
            examples=1,
            loss=0.5,
            policy_loss=0.0,
            policy_accuracy=1.0,
            value_loss=0.25,
        )
        declared_as_ppo = TransformerTrainingConfig(
            batch_size=64,
            epochs=1,
            learning_rate=1e-4,
            window_size=2,
            objective="ppo",
            random_seed=31415,
            freeze_non_value_parameters=False,
        )
        save_transformer_checkpoint(
            output,
            model,
            result=TransformerTrainingResult(
                model_config=converted_result.model_config,
                training_config=declared_as_ppo,
                epochs=(metrics,),
            ),
        )
        registered_value_only = TransformerTrainingConfig(
            batch_size=64,
            epochs=3,
            learning_rate=1e-4,
            window_size=2,
            device=str(neural_cli.resolve_torch_device(None)),
            objective="value-only",
            random_seed=31415,
            freeze_non_value_parameters=True,
        ).to_dict()

        with self.assertRaisesRegex(ValueError, "training_config"):
            neural_cli._foundation_value_tune_require_phase3_checkpoint_payload(
                output,
                training_summary={"training_config": registered_value_only, "epochs": [metrics.to_dict()]},
                selection={"payload": {"selected_epoch": 1}},
            )

        import torch

        for invalid_value in (float("nan"), float("inf")):
            payload = torch.load(output, map_location="cpu", weights_only=False)
            payload["state_dict"]["value_head.0.weight"][0, 0] = invalid_value
            torch.save(payload, output)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                neural_cli._foundation_value_tune_frozen_trunk_identity(
                    source_checkpoint=self.source,
                    compared_checkpoint=output,
                    expected_value_head_hidden=32,
                    label="value-tuned",
                )

        payload = torch.load(output, map_location="cpu", weights_only=False)
        payload["value_calibration_transform"] = {
            "method": "affine",
            "scale": 0.0,
            "bias": 0.75,
            "clip_min": -1.0,
            "clip_max": 1.0,
        }
        torch.save(payload, output)
        with self.assertRaisesRegex(ValueError, "unregistered value-calibration transform"):
            neural_cli._foundation_value_tune_require_phase3_checkpoint_payload(
                output,
                training_summary={"training_config": declared_as_ppo.to_dict(), "epochs": [metrics.to_dict()]},
                selection={"payload": {"selected_epoch": 1}},
            )

    def test_provenance_bound_mode_requires_a_new_private_output_directory(self) -> None:
        out_dir = self.root / "existing-output"
        out_dir.mkdir()

        with self.assertRaisesRegex(ValueError, "requires a new output directory"):
            neural_cli._validate_foundation_value_tune_paths(
                out_dir,
                summary_path=out_dir / "neural-foundation-value-tune-summary.json",
                require_fresh_output_dir=True,
            )

    def test_summary_path_may_not_replace_a_phase3_artifact(self) -> None:
        out_dir = self.root / "new-output"
        with self.assertRaisesRegex(ValueError, "may not collide"):
            neural_cli._validate_foundation_value_tune_paths(
                out_dir,
                summary_path=out_dir / "value-tuned-transformer-policy.pt",
                require_fresh_output_dir=True,
                artifact_paths=(out_dir / "value-tuned-transformer-policy.pt",),
            )

    def test_phase3_runner_refuses_unavailable_or_dirty_source_metadata(self) -> None:
        for metadata, error in (
            ({"available": False, "dirty": None}, "readable Git"),
            ({"available": True, "dirty": True, "head": "a" * 40}, "dirty source"),
        ):
            with patch.object(neural_cli, "collect_source_metadata", return_value=metadata):
                with self.assertRaisesRegex(ValueError, error):
                    neural_cli._foundation_value_tune_require_clean_source_metadata()

    def test_private_snapshot_execution_preserves_default_reused_input_identity(self) -> None:
        out_dir = self.root / "reused-input-output"
        recipe = copy.deepcopy(self._recipe(out_dir))
        provenance = recipe["phase3_provenance"]
        provenance["inputs"]["selection"] = copy.deepcopy(provenance["inputs"]["training"])
        provenance["inputs"]["calibration"] = copy.deepcopy(provenance["inputs"]["training"])
        recipe["command"]["argv"] = [
            str(self.train) if item in {str(self.selection), str(self.calibration)} else item
            for item in recipe["command"]["argv"]
        ]
        out_dir.mkdir()
        execution = neural_cli._foundation_value_tune_prepare_phase3_execution(recipe, out_dir=out_dir)
        argv = execution["execution_command"]["argv"]
        shared_snapshot = argv[argv.index("--data") + 1]
        self.assertEqual(argv[argv.index("--value-selection-data") + 1], shared_snapshot)
        self.assertEqual(argv[argv.index("--value-calibration-data") + 1], shared_snapshot)

    def test_publication_refuses_a_private_artifact_swapped_after_verification(self) -> None:
        out_dir = self.root / "publication-output"
        recipe = self._recipe(out_dir)
        out_dir.mkdir()
        execution = neural_cli._foundation_value_tune_prepare_phase3_execution(recipe, out_dir=out_dir)
        verified: dict[str, dict[str, object]] = {}
        for name, raw_path in execution["execution_artifacts"].items():
            path = Path(raw_path)
            path.write_text(f"verified-{name}\n", encoding="utf-8")
            verified[name] = neural_cli._foundation_value_tune_file_identity(path, label=name)
        Path(execution["execution_artifacts"]["checkpoint_path"]).write_text("swapped\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "after verification"):
            neural_cli._foundation_value_tune_publish_phase3_artifacts(
                execution,
                verification={"verified_artifacts": verified},
            )
        self.assertFalse((out_dir / "value-tuned-transformer-policy.pt").exists())

    def test_output_capture_publishes_the_verified_copy_not_a_later_staged_replacement(self) -> None:
        out_dir = self.root / "captured-publication-output"
        recipe = self._recipe(out_dir)
        out_dir.mkdir()
        execution = neural_cli._foundation_value_tune_prepare_phase3_execution(recipe, out_dir=out_dir)
        for name, raw_path in execution["execution_artifacts"].items():
            Path(raw_path).write_text(f"verified-{name}\n", encoding="utf-8")
        captured = neural_cli._foundation_value_tune_capture_phase3_outputs(execution)
        Path(execution["execution_artifacts"]["checkpoint_path"]).write_text("swapped\n", encoding="utf-8")
        verified = {
            name: neural_cli._foundation_value_tune_file_identity(Path(raw_path), label=name)
            for name, raw_path in captured["execution_artifacts"].items()
        }
        published = neural_cli._foundation_value_tune_publish_phase3_artifacts(
            captured,
            verification={"verified_artifacts": verified},
        )
        checkpoint = out_dir / "value-tuned-transformer-policy.pt"
        self.assertEqual(checkpoint.read_text(encoding="utf-8"), "verified-checkpoint_path\n")
        self.assertEqual(published["published_artifacts"]["checkpoint_path"]["sha256"], _sha256(checkpoint))

    def test_runner_uses_private_snapshots_and_publishes_only_verified_artifacts(self) -> None:
        out_dir = self.root / "runner-output"
        recipe = self._recipe(out_dir)
        clean_source = {
            "available": True,
            "repo_root": "/reviewed/pokezero",
            "branch": "main",
            "head": "a" * 40,
            "dirty": False,
        }
        mismatch_selected_metrics = False

        def fake_train(argv, **_kwargs):
            nonlocal mismatch_selected_metrics
            def after(flag: str) -> str:
                return argv[argv.index(flag) + 1]

            checkpoint = Path(after("--out"))
            selection = Path(after("--value-selection-out"))
            calibration = Path(after("--value-calibration-out"))
            train_summary = Path(after("--summary-out"))
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            selection_paths = [argv[argv.index("--value-selection-data") + 1]]
            calibration_paths = [argv[argv.index("--value-calibration-data") + 1]]
            train_paths = [argv[argv.index("--data") + 1]]
            model, converted_result = neural_cli.load_transformer_checkpoint(self.converted, map_location="cpu")
            torch = neural_cli.require_torch()
            with torch.no_grad():
                model.value_head[0].weight.add_(0.125)
            selected_epoch_metrics = TransformerEpochMetrics(
                epoch=1,
                examples=1,
                loss=0.5,
                policy_loss=0.0,
                policy_accuracy=1.0,
                value_loss=0.25,
            )
            full_training_config = TransformerTrainingConfig(
                batch_size=64,
                epochs=3,
                learning_rate=1e-4,
                window_size=converted_result.model_config.window_size,
                value_ranking_loss_weight=0.0,
                value_ranking_margin=0.0,
                max_batches=None,
                device=str(neural_cli.resolve_torch_device(None)),
                objective="value-only",
                random_seed=31415,
                freeze_non_value_parameters=True,
                shaping_weights=converted_result.model_config.reward_shaping,
            )
            selected_training_config = TransformerTrainingConfig(
                **{**full_training_config.to_dict(), "epochs": 1}
            )
            save_transformer_checkpoint(
                checkpoint,
                model,
                result=TransformerTrainingResult(
                    model_config=converted_result.model_config,
                    training_config=selected_training_config,
                    epochs=(selected_epoch_metrics,),
                ),
            )

            def calibration_report(pearson: float) -> dict[str, object]:
                pairs = ((-0.1, -0.2), (0.5, 0.4)) if pearson > 0.0 else ((-0.1, 0.4), (0.5, -0.2))

                def report_for(rows: tuple[tuple[float, float], ...]) -> dict[str, object]:
                    examples = len(rows)
                    errors = tuple(prediction - outcome for prediction, outcome in rows)
                    bins = []
                    for index in range(10):
                        matching = [row for row in rows if int(((row[0] + 1.0) / 2.0) * 10) == index]
                        lower = -1.0 + (2.0 * index / 10)
                        upper = -1.0 + (2.0 * (index + 1) / 10)
                        mean_prediction = sum(row[0] for row in matching) / len(matching) if matching else 0.0
                        mean_return = sum(row[1] for row in matching) / len(matching) if matching else 0.0
                        bins.append(
                            {
                                "lower": lower,
                                "upper": upper,
                                "count": len(matching),
                                "mean_prediction": mean_prediction,
                                "mean_return": mean_return,
                                "calibration_error": abs(mean_prediction - mean_return),
                            }
                        )
                    return {
                        "examples": examples,
                        "mse": sum(error * error for error in errors) / examples,
                        "mae": sum(abs(error) for error in errors) / examples,
                        "bias": sum(errors) / examples,
                        "sign_accuracy": sum(
                            int((prediction > 0.0) - (prediction < 0.0) == (outcome > 0.0) - (outcome < 0.0))
                            for prediction, outcome in rows
                        ) / examples,
                        "expected_calibration_error": sum(
                            (bin_payload["count"] / examples) * bin_payload["calibration_error"]
                            for bin_payload in bins
                            if bin_payload["count"]
                        ),
                        "pearson_correlation": pearson if examples > 1 else None,
                        "bins": bins,
                        "slices": [],
                    }

                payload = report_for(pairs)
                return_slices = []
                for name, predicate in (
                    ("return:positive", lambda outcome: outcome > 0.0),
                    ("return:negative", lambda outcome: outcome < 0.0),
                ):
                    rows = tuple(row for row in pairs if predicate(row[1]))
                    if rows:
                        row_payload = report_for(rows)
                        return_slices.append({
                            "name": name,
                            **{key: row_payload[key] for key in ("examples", "mse", "mae", "bias", "sign_accuracy", "expected_calibration_error", "pearson_correlation")},
                            "sign_accuracy_applicable": True,
                        })
                aggregate_slices = []
                for name in ("turn:early_0_9", "terminal:uncapped"):
                    aggregate_payload = report_for(pairs)
                    aggregate_slices.append({
                        "name": name,
                        **{key: aggregate_payload[key] for key in ("examples", "mse", "mae", "bias", "sign_accuracy", "expected_calibration_error", "pearson_correlation")},
                        "sign_accuracy_applicable": True,
                    })
                payload["slices"] = [*return_slices, *aggregate_slices]
                return payload

            selection_epochs = [
                {
                    "epoch": epoch,
                    "metric_value": metric_value,
                    "training_metrics": TransformerEpochMetrics(
                        epoch=epoch,
                        examples=1,
                        loss=(1.5 if mismatch_selected_metrics else 0.5) if epoch == 1 else 0.5 + epoch,
                        policy_loss=0.0,
                        policy_accuracy=1.0,
                        value_loss=0.25,
                    ).to_dict(),
                    "report": calibration_report(metric_value),
                }
                for epoch, metric_value in ((1, 1.0), (2, -1.0), (3, -1.0))
            ]
            selection_payload = {
                "paths": selection_paths,
                "batch_size": 128,
                "bins": 10,
                "metric": "pearson_correlation",
                "metric_direction": "max",
                "selected_epoch": 1,
                "selected_metric_value": 1.0,
                "epochs": selection_epochs,
            }
            # The historical shared pathname may be replaced while a long trainer runs.  The
            # wrapper must continue with its private descriptor-backed snapshot, not notice it
            # only through a fragile postflight rehash.
            self.train.write_text("transient-shared-path-replacement\n", encoding="utf-8")
            selection.write_text(json.dumps(selection_payload), encoding="utf-8")
            calibration.write_text(json.dumps({
                "paths": calibration_paths,
                "batch_size": 128,
                "bins": 10,
                "report": calibration_report(1.0),
            }), encoding="utf-8")
            train_summary.write_text(json.dumps({
                "schema_version": neural_cli.NEURAL_TRAIN_SUMMARY_SCHEMA_VERSION,
                "source": clean_source,
                "started_at": "2026-08-21T00:00:00Z",
                "completed_at": "2026-08-21T00:00:01Z",
                "elapsed_seconds": 1.0,
                "train_elapsed_seconds": 0.9,
                "data_paths": train_paths,
                "input_data_bytes": sum(Path(path).stat().st_size for path in train_paths),
                "refutation_cache_bytes": None,
                "checkpoint_path": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "model": {
                    "policy_id": converted_result.model_config.policy_id,
                    "window_size": converted_result.model_config.window_size,
                    "embedding_dim": converted_result.model_config.embedding_dim,
                    "transformer_layers": converted_result.model_config.transformer_layers,
                    "attention_heads": converted_result.model_config.attention_heads,
                    "feedforward_dim": converted_result.model_config.feedforward_dim,
                    "dropout": converted_result.model_config.dropout,
                    "temporal_aggregator": converted_result.model_config.temporal_aggregator,
                    "categorical_vocab_size": converted_result.model_config.categorical_vocab_size,
                    "category_oov_buckets": converted_result.model_config.category_oov_buckets,
                },
                "training_config": full_training_config.to_dict(),
                "distributed_training": {
                    "enabled": False,
                    "rank": 0,
                    "world_size": 1,
                    "local_rank": 0,
                    "backend": None,
                    "base_seed": 31415,
                    "rank_seed": 31415,
                },
                "epochs": [selected_epoch_metrics.to_dict()],
                "final_metrics": selected_epoch_metrics.to_dict(),
                "value_selection": selection_payload,
                "refutation_training": {"enabled": False, "paths": [], "max_fraction": None, "target_mode": None},
                "training_cache": neural_cli._TrainingCacheLifecycle().to_summary(),
            }), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            patch.object(neural_cli, "_foundation_value_tune_recipe", return_value=recipe),
            patch.object(neural_cli, "_foundation_value_tune_require_clean_source_metadata", return_value=clean_source),
            patch.object(neural_cli.subprocess, "run", side_effect=fake_train),
        ):
            with patch("sys.stdout", new_callable=io.StringIO):
                exit_code = neural_cli._foundation_value_tune_run(SimpleNamespace(summary_path=None))

        self.assertEqual(exit_code, 0)
        self.assertTrue((out_dir / "value-tuned-transformer-policy.pt").is_file())
        self.assertTrue((out_dir / "value-selection.json").is_file())
        self.assertTrue((out_dir / "value-calibration.json").is_file())
        self.assertTrue((out_dir / "phase3-train-summary.json").is_file())
        self.assertEqual(stat.S_IMODE((out_dir / "value-tuned-transformer-policy.pt").stat().st_mode), 0o644)
        self.assertEqual(self.train.read_text(encoding="utf-8"), "transient-shared-path-replacement\n")
        summary = json.loads((out_dir / "neural-foundation-value-tune-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["phase3_verification"]["published_artifacts"]["checkpoint_path"]["sha256"], _sha256(out_dir / "value-tuned-transformer-policy.pt"))

        # A selected report cannot describe different optimizer metrics from the saved selected
        # checkpoint. The row is otherwise complete, so this exercises the cross-artifact gate.
        mismatch_selected_metrics = True
        mismatched_out_dir = self.root / "runner-mismatched-selection-output"
        mismatched_recipe = self._recipe(mismatched_out_dir)
        with (
            patch.object(neural_cli, "_foundation_value_tune_recipe", return_value=mismatched_recipe),
            patch.object(neural_cli, "_foundation_value_tune_require_clean_source_metadata", return_value=clean_source),
            patch.object(neural_cli.subprocess, "run", side_effect=fake_train),
        ):
            with patch("sys.stdout", new_callable=io.StringIO):
                mismatch_exit_code = neural_cli._foundation_value_tune_run(SimpleNamespace(summary_path=None))
        self.assertEqual(mismatch_exit_code, 1)
        mismatch_summary = json.loads(
            (mismatched_out_dir / "neural-foundation-value-tune-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(mismatch_summary["status"], "failed")
        self.assertIn("selected value-selection metrics", mismatch_summary["error"]["message"])
        self.assertFalse((mismatched_out_dir / "value-tuned-transformer-policy.pt").exists())
