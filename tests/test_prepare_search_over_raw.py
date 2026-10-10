import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from pokezero.mcts_eval.search_over_raw import ENGINEERING_EXCLUDED_SEEDS


SPEC = importlib.util.spec_from_file_location("prepare_search_over_raw",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_search_over_raw.py")
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class PreparationTest(unittest.TestCase):
    def test_create_only_preparation_binds_inputs_and_excludes_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration = root / "historical.json"
            registration.write_text(json.dumps(dict(seeds=[1, 2])))
            plan = root / "plan.md"
            plan.write_text("test fixture")
            out = root / "new"
            result = DRIVER.prepare([registration], plan, out)
            self.assertFalse(result["execution_ready"])
            self.assertEqual(len(result["configurations"]), 37)
            self.assertEqual(result["excluded_seeds"], sorted({1, 2, *ENGINEERING_EXCLUDED_SEEDS}))
            self.assertEqual(len(result["input_hashes"]), 2)
            self.assertEqual(json.loads((out / "contract.json").read_text()), result)
            with self.assertRaises(FileExistsError):
                DRIVER.prepare([registration], plan, out)

    def test_malformed_exposure_cannot_prepare_a_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration = root / "bad.json"
            registration.write_text(json.dumps(dict(seeds=[True])))
            plan = root / "plan.md"
            plan.write_text("fixture")
            with self.assertRaisesRegex(ValueError, "seeds roster"):
                DRIVER.prepare([registration], plan, root / "new")
            self.assertFalse((root / "new").exists())


if __name__ == "__main__":
    unittest.main()
