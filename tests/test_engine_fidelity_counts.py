"""Static fidelity-count guards must include inherited class selectors."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_engine_fidelity_unittest_counts.py"
spec = importlib.util.spec_from_file_location("engine_fidelity_counts", SCRIPT)
counts = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = counts
spec.loader.exec_module(counts)


class FidelityCountTests(unittest.TestCase):
    def count(self, target, source, fixture="class Fixture:\n    pass\n"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_example.py").write_text(source)
            (root / "test_fixture.py").write_text(fixture)
            with patch.object(counts, "TESTS", root):
                return counts._test_count(target)

    def test_class_selector_includes_inherited_methods_and_deduplicates_overrides(self):
        source = """import unittest
class Base(unittest.TestCase):
    def test_one(self): pass
    def test_two(self): pass
class Child(Base):
    def test_one(self): pass
    def test_three(self): pass
"""
        self.assertEqual(self.count("tests.test_example.Child", source), 3)
        self.assertEqual(self.count("tests.test_example.Child.test_two", source), 1)

    def test_imported_fixture_methods_are_not_silently_ignored(self):
        source = """import unittest
from test_fixture import Fixture as ImportedFixture
class Child(ImportedFixture, unittest.TestCase):
    def test_local(self): pass
"""
        fixture = "class Fixture:\n    def test_inherited(self): pass\n"
        self.assertEqual(self.count("tests.test_example.Child", source, fixture), 2)

    def test_wrong_class_cannot_select_another_classes_method(self):
        source = """import unittest
class First(unittest.TestCase):
    def test_one(self): pass
class Second(unittest.TestCase):
    def test_two(self): pass
"""
        with self.assertRaisesRegex(ValueError, "not a unittest test method"):
            self.count("tests.test_example.First.test_two", source)

    def test_unknown_base_refuses_instead_of_undercounting(self):
        with self.assertRaisesRegex(ValueError, "unsupported test-class base"):
            self.count("tests.test_example.Child", "class Child(Unknown):\n    def test_one(self): pass\n")

    def test_dynamic_member_and_cycles_refuse(self):
        with self.assertRaisesRegex(ValueError, "dynamic test member"):
            self.count("tests.test_example.Child", "class Child:\n    test_alias = other\n")
        with self.assertRaisesRegex(ValueError, "cyclic"):
            self.count("tests.test_example.Child", "class Base(Child): pass\nclass Child(Base): pass\n")

    def test_empty_or_missing_class_refuses(self):
        for target in ("tests.test_example.Child", "tests.test_example.Missing"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.count(target, "class Child:\n    pass\n")

    def test_legacy_workflow_scanner_counts_mixed_module_and_class_selection(self):
        from tests.test_unreachable_readjudication import EveryWorkflowTestCountGuardMatchesItsModuleTests as scanner
        targets = (
            "tests.test_policy_opponent_engine", "tests.test_policy_opponent",
            "tests.test_policy_opponent_view", "tests.test_policy_opponent_request",
            "tests.test_policy_opponent_native.NativePolicyOpponentSearchTest",
            "tests.test_policy_opponent_native.NativeCanonicalOwnPolicyTest",
        )
        self.assertEqual(sum(scanner._target_methods(target) for target in targets), 53)

    def test_legacy_workflow_scanner_refuses_invalid_class_method_selection(self):
        from tests.test_unreachable_readjudication import EveryWorkflowTestCountGuardMatchesItsModuleTests as scanner
        with self.assertRaisesRegex(ValueError, "not a unittest test method"):
            scanner._target_methods("tests.test_engine_fidelity_counts.FidelityCountTests.test_missing")


if __name__ == "__main__":
    unittest.main()
