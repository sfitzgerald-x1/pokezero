"""Opening admission and isolated export contracts, no battles or model loads."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.lattice import materialize_search_artifacts
from pokezero.mcts_eval.resolver import ContractError
from pokezero.mcts_eval.search_over_raw import (
    ENGINEERING_EXCLUDED_SEEDS, SearchConfiguration, phase_a_contract)

SPEC = importlib.util.spec_from_file_location("qualify_search_over_raw_opening",
    Path(__file__).resolve().parents[1] / "scripts" / "qualify_search_over_raw_opening.py")
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


def registration():
    return dict(schema="pokezero.search-over-raw.opening-benchmark.v1",
        fixture_seed=DRIVER.FIXTURE_SEED, seeds=list(ENGINEERING_EXCLUDED_SEEDS),
        candidate_seat="p1", retry_authorized=False, phase_a_admission=False,
        scientific_strength_evidence=False,
        configurations=[asdict(SearchConfiguration(a, seconds=1., workers=20 if a == "reference" else 1))
            for a in DRIVER.ARMS])


class OpeningAdmissionTests(unittest.TestCase):
    def test_claim_is_durable_and_cannot_retry_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = DRIVER.claim_attempt(root, registration())
            self.assertEqual(tuple(c.arm for c in cfg), DRIVER.ARMS)
            self.assertEqual(json.loads((root / "attempt.json").read_text())["status"], "CLAIMED_BEFORE_RUNTIME")
            before = (root / "attempt.json").read_bytes()
            with self.assertRaises(FileExistsError):
                DRIVER.claim_attempt(root, registration())
            self.assertEqual(before, (root / "attempt.json").read_bytes())

    def test_scientific_or_retry_authorization_rejected_before_claim(self):
        for key in ("retry_authorized", "phase_a_admission", "scientific_strength_evidence"):
            with tempfile.TemporaryDirectory() as directory:
                r = registration()
                r[key] = True
                with self.assertRaisesRegex(ValueError, "scientific outcomes"):
                    DRIVER.claim_attempt(Path(directory), r)
                self.assertFalse((Path(directory) / "attempt.json").exists())

    def test_seed_roster_cannot_redraw(self):
        with tempfile.TemporaryDirectory() as directory:
            r = registration()
            r["seeds"].append(42)
            with self.assertRaisesRegex(ValueError, "contract changed"):
                DRIVER.claim_attempt(Path(directory), r)

    def test_reference_worker_count_and_oracle_cannot_change(self):
        for key, value in (("workers", 1), ("belief", "oracle"), ("leaf", "hp_fraction")):
            with tempfile.TemporaryDirectory() as directory:
                r = registration()
                r["configurations"][-1][key] = value
                with self.assertRaisesRegex(ValueError, "arm/resource drift"):
                    DRIVER.claim_attempt(Path(directory), r)

    def test_input_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            artifact.write_bytes(b"changed")
            r = dict(source_root=directory, source_commit="commit", showdown_root=directory,
                showdown_commit="showdown", input_sha256={str(artifact): hashlib.sha256(b"original").hexdigest()})
            with patch.object(DRIVER, "clean_identity") as clean:
                with self.assertRaisesRegex(ValueError, "input drift"):
                    DRIVER.verify_inputs(r)
                self.assertEqual(clean.call_count, 2)

    def test_commit_and_dirty_worktree_are_rejected(self):
        with patch.object(DRIVER, "git", return_value="other"):
            with self.assertRaisesRegex(ValueError, "commit drift"):
                DRIVER.clean_identity("fixture", "expected")
        with patch.object(DRIVER, "git", side_effect=["expected", " M owned.py"]):
            with self.assertRaisesRegex(ValueError, "must be clean"):
                DRIVER.clean_identity("fixture", "expected")

    def test_known_engineering_seeds_are_excluded_without_caller_inventory(self):
        configs = [SearchConfiguration(a, workers=20 if a == "reference" else 1) for a in DRIVER.ARMS]
        with patch("pokezero.mcts_eval.search_over_raw.rng_seed",
                   side_effect=[*ENGINEERING_EXCLUDED_SEEDS, *range(64)]):
            contract = phase_a_contract("12345678-1234-4234-8234-123456789abc",
                excluded_seeds=[], configurations=configs)
        seeds = {s for panel in contract["panels"].values() for s in panel["seeds"]}
        self.assertEqual(seeds, set(range(64)))
        self.assertEqual(contract["excluded_seeds"], sorted(ENGINEERING_EXCLUDED_SEEDS))


class IsolatedSearchArtifactTests(unittest.TestCase):
    def contract(self, root):
        model, tables = root / "model.pt", root / "tables.json"
        model.write_bytes(b"fixture model")
        tables.write_bytes(b"fixture tables")
        return SimpleNamespace(model_path=str(model), tables_path=str(tables),
            model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            tables_sha256=hashlib.sha256(tables.read_bytes()).hexdigest())

    def test_explicit_exports_are_validated_without_shared_cache_or_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.contract(Path(directory))
            with patch("pokezero.mcts_eval.resolver.validate_encoder_tables") as validate, \
                    patch("subprocess.run") as launch:
                result = materialize_search_artifacts(c, showdown_root="fixture")
                validate.assert_called_once_with(c, c.tables_path)
                launch.assert_not_called()
                self.assertEqual(result, dict(model_path=c.model_path, tables_path=c.tables_path))

    def test_partial_explicit_exports_are_not_silently_materialized(self):
        c = SimpleNamespace(model_path="fixture", tables_path=None)
        with self.assertRaisesRegex(ContractError, "both model and tables"):
            materialize_search_artifacts(c, showdown_root="fixture")

    def test_changed_explicit_export_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.contract(Path(directory))
            Path(c.model_path).write_bytes(b"drift")
            with self.assertRaisesRegex(ContractError, "digest drift"):
                materialize_search_artifacts(c, showdown_root="fixture")


if __name__ == "__main__":
    unittest.main()
