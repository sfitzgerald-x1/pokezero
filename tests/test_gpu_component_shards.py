"""Component shards retain discovery blocks and fail closed."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from scripts.run_gpu_component_shard import partition_suite, run_suite


class ComponentShardTests(unittest.TestCase):
    def test_four_shards_cover_every_block_once_and_keep_custom_suites(self):
        called = []
        class Pass(unittest.TestCase):
            def runTest(self):
                pass
        class CustomSuite(unittest.TestSuite):
            def run(self, result, debug=False):
                called.append("custom")
                return super().run(result, debug)
        blocks = [CustomSuite([Pass(), Pass()])] + [
            unittest.TestSuite([Pass(), Pass()]) for _ in range(7)]
        original = unittest.TestSuite(blocks)
        parts = [partition_suite(original, i, 4) for i in range(4)]
        indices = [index for _, selected, _ in parts for index in selected]
        self.assertEqual(sorted(indices), list(range(8)))
        self.assertEqual(len(set(indices)), 8)
        self.assertEqual([suite.countTestCases() for suite, _, _ in parts], [4]*4)
        self.assertIs(next(iter(parts[0][0])), blocks[0])
        for suite, _, total in parts:
            self.assertEqual(total, 8)
            self.assertEqual(run_suite(suite, stream=io.StringIO()), 0)
        self.assertEqual(called, ["custom"])

    def test_original_discovery_retains_nested_packages_and_package_tests(self):
        source = "import unittest\nclass Case(unittest.TestCase):\n def test_ok(self):pass\n"
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules), \
                mock.patch.object(sys, "path", sys.path.copy()):
            root = Path(tmp)
            (root / "test_gpu_engine_shard_fixture_flat.py").write_text(source)
            package = root / "shard_fixture_package"
            package.mkdir()
            (package / "__init__.py").write_text(source)
            (package / "test_gpu_engine_shard_fixture_nested.py").write_text(source)
            discovered = unittest.TestLoader().discover(str(root), "test_gpu_engine*.py")
            self.assertEqual(discovered.countTestCases(), 3)
            parts = [partition_suite(discovered, i, 2)[0] for i in range(2)]
            self.assertEqual(sum(suite.countTestCases() for suite in parts), 3)
            for suite in parts:
                self.assertEqual(run_suite(suite, stream=io.StringIO()), 0)

    def test_invalid_partition_and_empty_discovery_fail(self):
        suite = unittest.TestSuite([unittest.TestSuite()])
        for index, count in ((0, 0), (0, True), (True, 4), (-1, 4), (4, 4)):
            with self.subTest(index=index, count=count), self.assertRaises(ValueError):
                partition_suite(suite, index, count)
        with self.assertRaisesRegex(ValueError, "empty"):
            partition_suite(unittest.TestSuite(), 0, 4)

    def test_runner_rejects_failure_skip_and_zero_tests(self):
        class Pass(unittest.TestCase):
            def runTest(self):
                pass
        class Fail(unittest.TestCase):
            def runTest(self):
                self.fail("deliberate control")
        class Skip(unittest.TestCase):
            @unittest.skip("deliberate control")
            def runTest(self):
                pass
        for cases, expected in (([Pass()], 0), ([Fail()], 1), ([Skip()], 1), ([], 1)):
            with self.subTest(expected=expected, cases=len(cases)):
                self.assertEqual(run_suite(unittest.TestSuite(cases), stream=io.StringIO()), expected)
