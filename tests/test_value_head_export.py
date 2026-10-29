import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero.neural_policy import torch_available
from pokezero.value_head_export import (
    build_value_head_export, load_selected_head, select_validation_head,
    sha256_file, validate_export,
)


class SelectionTests(unittest.TestCase):
    def summary(self):
        return {"schema": "pokezero.value_probe_head_refits.v1",
            "status": "THREE_OUTCOME_HEAD_REFITS_COMPLETE_ZEROS_UNRESOLVED", "model_fits": 3,
            "heads": {kind: {"selected_epoch": 1, "validation_mse": mse, "test": {"mse": test}}
                for kind, mse, test in (("linear", .8, .1), ("attention_mlp", .75, .01),
                                        ("attention_hl_gauss", .71, .9))}}

    def test_validation_selection_ignores_test_and_ties_are_fixed(self):
        summary = self.summary()
        self.assertEqual(select_validation_head(summary), "attention_hl_gauss")
        for head in summary["heads"].values():
            head["test"] = {"mse": -999}
        self.assertEqual(select_validation_head(summary), "attention_hl_gauss")
        summary["heads"]["linear"]["validation_mse"] = .71
        self.assertEqual(select_validation_head(summary), "linear")
        summary["heads"]["linear"]["validation_mse"] = float("nan")
        with self.assertRaises(ValueError):
            select_validation_head(summary)


@unittest.skipUnless(torch_available(), "torch required")
class ExportTests(unittest.TestCase):
    def setUp(self):
        import torch
        from pokezero.neural_policy import EntityTokenTransformerPolicy, TransformerPolicyConfig
        torch.set_num_threads(1)
        torch.manual_seed(15)
        self.config = TransformerPolicyConfig.compact_category(category_vocab=["kind:a", "kind:b"],
            window_size=1, embedding_dim=16, attention_heads=4, feedforward_dim=32,
            transformer_layers=1, dropout=0.0, observation_schema_version="pokezero.observation.v4")
        self.model = EntityTokenTransformerPolicy(self.config).float().eval()
        source = Path(__file__).resolve().parents[1] / "scripts/export_model.py"
        spec = importlib.util.spec_from_file_location("export_model_a4_test", source)
        self.exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.exporter)

    def test_all_heads_match_independent_oracle_after_trace_reload_and_batch_change(self):
        import torch
        from pokezero.value_precision_probe import make_value_head
        before = {name: tensor.clone() for name, tensor in self.model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            for kind in (None, "linear", "attention_mlp", "attention_hl_gauss"):
                with self.subTest(kind=kind):
                    head = None if kind is None else make_value_head(kind, 16).float().eval()
                    module = build_value_head_export(self.model, head)
                    path = Path(directory) / f"{kind}.pt"
                    self.exporter.export_torchscript(module,
                        self.exporter.make_random_inputs(self.config, 4, seed=1), path)
                    trace = torch.jit.load(str(path)).eval()
                    for batch in (1, 7):
                        inputs = self.exporter.make_random_inputs(self.config, batch, seed=batch)
                        parity = validate_export(module, trace, self.model, head, inputs)
                        self.assertLessEqual(max(parity.values()), 1e-5)
        self.assertTrue(all(torch.equal(tensor, before[name])
                            for name, tensor in self.model.state_dict().items()))

    def test_head_bytes_epoch_feature_lineage_and_contract_bound(self):
        import torch
        from pokezero.value_precision_probe import make_value_head
        contract = {"width": 16, "checkpoint_sha256": "frozen"}
        summary = SelectionTests().summary()
        summary.update(checkpoint_contract=contract, features_manifest_sha256="features")
        payload = {"schema": "pokezero.value_probe_head.v1", "kind": "attention_hl_gauss",
            "width": 16, "features_manifest_sha256": "features", "selected_epoch": 1,
            "state_dict": make_value_head("attention_hl_gauss", 16).state_dict()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, path = root / "attention_hl_gauss.pt", root / "head-refit-summary.json"
            def save():
                torch.save(payload, artifact)
                summary["heads"]["attention_hl_gauss"]["artifact_sha256"] = sha256_file(artifact)
                path.write_text(json.dumps(summary))
                return sha256_file(path)
            binding = save()
            _, receipt = load_selected_head(path, summary_sha256=binding, checkpoint_contract=contract)
            self.assertEqual(receipt["kind"], "attention_hl_gauss")
            with self.assertRaisesRegex(ValueError, "summary binding"):
                load_selected_head(path, summary_sha256="other", checkpoint_contract=contract)
            with self.assertRaisesRegex(ValueError, "checkpoint contract"):
                load_selected_head(path, summary_sha256=binding, checkpoint_contract={"width": 32})
            for field, value in (("features_manifest_sha256", "wrong"), ("selected_epoch", 2)):
                original = payload[field]; payload[field] = value; binding = save()
                with self.assertRaisesRegex(ValueError, "provenance"):
                    load_selected_head(path, summary_sha256=binding, checkpoint_contract=contract)
                payload[field] = original
            binding = save()
            artifact.write_bytes(artifact.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "head bytes"):
                load_selected_head(path, summary_sha256=binding, checkpoint_contract=contract)

    def test_complete_export_pair_manifest_and_refuses_resume(self):
        import sys
        import torch
        from pokezero.neural_policy import (TransformerTrainingConfig, TransformerTrainingResult,
                                            save_transformer_checkpoint)
        from pokezero.value_precision_probe import checkpoint_contract, make_value_head
        scripts = str(Path(__file__).resolve().parents[1] / "scripts")
        sys.path.insert(0, scripts)
        try:
            from export_value_probe_search import export_pair
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint, summary_path = root / "champion.pt", root / "head-refit-summary.json"
                save_transformer_checkpoint(checkpoint, self.model, result=TransformerTrainingResult(
                    model_config=self.config, training_config=TransformerTrainingConfig(), epochs=(),
                    belief_set_source_hash="source"))
                contract = checkpoint_contract(checkpoint)
                artifact = root / "attention_hl_gauss.pt"
                torch.save({"schema": "pokezero.value_probe_head.v1", "kind": "attention_hl_gauss",
                    "width": 16, "selected_epoch": 1, "features_manifest_sha256": "features",
                    "state_dict": make_value_head("attention_hl_gauss", 16).state_dict()}, artifact)
                summary = SelectionTests().summary()
                summary.update(checkpoint_contract=contract, features_manifest_sha256="features")
                summary["heads"]["attention_hl_gauss"]["artifact_sha256"] = sha256_file(artifact)
                summary_path.write_text(json.dumps(summary))
                options = dict(checkpoint_sha256=sha256_file(checkpoint),
                               summary_sha256=sha256_file(summary_path))
                manifest = export_pair(checkpoint, summary_path, root / "output", **options)
                self.assertEqual(manifest["selection"]["kind"], "attention_hl_gauss")
                self.assertEqual(manifest["precision"], "float32")
                self.assertEqual(manifest["new_games"], 0)
                self.assertEqual(set(manifest["formats"]), {"champion", "refit"})
                self.assertEqual(manifest["formats"]["refit"]["parity_batches"], [1, 7, 64])
                with self.assertRaisesRegex(ValueError, "fresh export"):
                    export_pair(checkpoint, summary_path, root / "output", **options)
        finally:
            sys.path.remove(scripts)

    def test_snapshot_swap_and_missing_value_head_refuse(self):
        import io
        import sys
        import torch
        from pokezero.neural_policy import (TransformerTrainingConfig, TransformerTrainingResult,
                                            save_transformer_checkpoint)
        from pokezero.value_precision_probe import checkpoint_contract, make_value_head
        scripts = str(Path(__file__).resolve().parents[1] / "scripts")
        sys.path.insert(0, scripts)
        try:
            import export_value_probe_search as exporter
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint, summary_path = root / "champion.pt", root / "head-refit-summary.json"
                save_transformer_checkpoint(checkpoint, self.model, result=TransformerTrainingResult(
                    model_config=self.config, training_config=TransformerTrainingConfig(), epochs=(),
                    belief_set_source_hash="source"))
                original = checkpoint.read_bytes()
                payload = torch.load(io.BytesIO(original), weights_only=True)
                payload["state_dict"]["policy_head.bias"] += 3
                changed = io.BytesIO(); torch.save(payload, changed)
                head_path = root / "attention_hl_gauss.pt"
                torch.save({"schema": "pokezero.value_probe_head.v1", "kind": "attention_hl_gauss",
                    "width": 16, "selected_epoch": 1, "features_manifest_sha256": "features",
                    "state_dict": make_value_head("attention_hl_gauss", 16).state_dict()}, head_path)
                summary = SelectionTests().summary()
                summary.update(checkpoint_contract=checkpoint_contract(checkpoint), features_manifest_sha256="features")
                summary["heads"]["attention_hl_gauss"]["artifact_sha256"] = sha256_file(head_path)
                summary_path.write_text(json.dumps(summary))
                options = dict(checkpoint_sha256=sha256_file(checkpoint), summary_sha256=sha256_file(summary_path))
                # Contract sees A, snapshot sees B, and final external path would
                # see A again. The snapshot SHA must stop before deserialization.
                read_bytes = Path.read_bytes
                calls = 0
                def aba(path):
                    nonlocal calls
                    if path == checkpoint:
                        calls += 1
                        if calls == 2:
                            return changed.getvalue()
                    return read_bytes(path)
                with patch.object(Path, "read_bytes", aba), self.assertRaisesRegex(ValueError, "snapshot SHA"):
                    exporter.export_pair(checkpoint, summary_path, root / "aba", **options)
                self.assertFalse((root / "aba").exists())
                # If the external path changes only as the torch loader is called,
                # immutable bytes remain A; B cannot enter the production model.
                load = torch.load
                def loader(source, *args, **kwargs):
                    if isinstance(source, io.BytesIO) and source.getvalue() == original:
                        checkpoint.write_bytes(changed.getvalue())
                        try:
                            return load(source, *args, **kwargs)
                        finally:
                            checkpoint.write_bytes(original)
                    return load(source, *args, **kwargs)
                import numpy as np
                real_path = root / "real-inputs.npz"
                real_tensors = self.exporter.make_random_inputs(self.config, 20, seed=21)
                with real_path.open("wb") as stream:
                    np.savez(stream, **{name: tensor.numpy() for name, tensor
                                       in zip(self.exporter.INPUT_NAMES, real_tensors)})
                real_original = real_path.read_bytes()
                real_sha = sha256_file(real_path)
                np_load = np.load
                snapshot_seen = []
                def load_inputs(source, *args, **kwargs):
                    self.assertIsInstance(source, io.BytesIO)
                    self.assertEqual(source.getvalue(), real_original)
                    snapshot_seen.append(True)
                    real_path.write_bytes(b"path swapped after snapshot")
                    try:
                        return np_load(source, *args, **kwargs)
                    finally:
                        real_path.write_bytes(real_original)
                with patch.object(torch, "load", loader), patch.object(np, "load", load_inputs):
                    manifest = exporter.export_pair(checkpoint, summary_path, root / "immutable",
                        real_inputs=real_path, **options)
                self.assertEqual(snapshot_seen, [True])
                self.assertEqual(manifest["real_saved_inputs"]["sha256"], real_sha)
                trace = torch.jit.load(manifest["formats"]["champion"]["path"])
                inputs = self.exporter.make_random_inputs(self.config, 1, seed=11)
                with torch.no_grad():
                    reference = self.model(**dict(zip(self.exporter.INPUT_NAMES, inputs)))
                    torch.testing.assert_close(trace(*inputs)[0], reference.policy_logits, rtol=0, atol=1e-5)
                # A checksum-consistent checkpoint with missing value parameters
                # must fail, rather than export randomly initialized fallback.
                payload = load(io.BytesIO(original), weights_only=True)
                del payload["state_dict"]["value_head.weight"]
                torch.save(payload, checkpoint)
                summary["checkpoint_contract"] = checkpoint_contract(checkpoint)
                summary_path.write_text(json.dumps(summary))
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    exporter.export_pair(checkpoint, summary_path, root / "missing",
                        checkpoint_sha256=sha256_file(checkpoint), summary_sha256=sha256_file(summary_path))
                self.assertFalse((root / "missing").exists())
        finally:
            sys.path.remove(scripts)

    def test_reject_half_parameters(self):
        with self.assertRaisesRegex(ValueError, "FP32"):
            build_value_head_export(self.model.half(), None)


if __name__ == "__main__":
    unittest.main()
