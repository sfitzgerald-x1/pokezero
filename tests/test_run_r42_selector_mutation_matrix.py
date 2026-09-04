from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_r42_selector_mutation_matrix.py"


def _module():
    spec = importlib.util.spec_from_file_location("r42_mutation_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class R42SelectorMutationMatrixTest(unittest.TestCase):
    def test_accepts_an_explicit_receipt_bound_commit_without_git_metadata(self) -> None:
        module = _module()
        self.assertEqual(module._source_commit("a" * 40), "a" * 40)
        with self.assertRaisesRegex(module.MutationError, "--source-commit"):
            module._source_commit("not-a-commit")

    def test_targeted_mutations_are_killed_against_the_mutated_src_copy(self) -> None:
        module = _module()
        result = module.run(source_commit="a" * 40)
        self.assertTrue(result["all_killed"])
        self.assertEqual(
            {item["name"] for item in result["mutations"]},
            {item.name for item in module.MUTATIONS},
        )
        self.assertTrue(all(item["status"] == "KILLED" for item in result["mutations"]))
        self.assertTrue(all(item["exit_code"] != 0 for item in result["mutations"]))


if __name__ == "__main__":
    unittest.main()
