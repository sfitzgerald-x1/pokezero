"""Component shards retain discovery blocks and fail closed."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from scripts.run_gpu_component_shard import partition_suite, run_suite, _valid_receipt, _digest


ROOT = Path(__file__).resolve().parents[1]


class ComponentShardTests(unittest.TestCase):
    def isolated(self, root, index=0, count=1):
        # Actual fresh parent AND child interpreters, no mocked process runner
        # and no real component discovery/JAX imports in these tiny fixtures.
        code = """
import sys, unittest
from scripts.run_gpu_component_shard import run_process_blocks
suite = unittest.TestLoader().discover(sys.argv[1], 'test_gpu_engine*.py')
raise SystemExit(run_process_blocks(suite, int(sys.argv[2]), int(sys.argv[3]), start_dir=sys.argv[1]))
"""
        environment = dict(os.environ)
        environment['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(ROOT), environment.get('PYTHONPATH', '')])
        result = subprocess.run([sys.executable, '-c', code, str(root), str(index), str(count)],
            cwd=ROOT, env=environment, text=True, capture_output=True, timeout=30)
        summaries = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        terminal = [row for row in summaries if row.get('process_blocks_terminal')]
        self.assertEqual(len(terminal), 1, (result.returncode, result.stdout, result.stderr))
        return result, terminal[0]

    def fixture(self, root):
        log = root / 'events.jsonl'
        emit = f"""
import json, os, sys, unittest
from pathlib import Path
def emit(label):
 with Path({str(log)!r}).open('a') as stream:
  stream.write(json.dumps(dict(label=label, pid=os.getpid(), exe=sys.executable))+'\\n')
"""
        (root / 'test_gpu_engine_alpha.py').write_text(emit + """
class Case(unittest.TestCase):
 def test_a(self):emit('alpha_a')
 def test_b(self):emit('alpha_b')
class Reverse(unittest.TestSuite):
 def run(self, result, debug=False):
  emit('custom_reverse')
  self._tests.reverse()
  return super().run(result, debug)
def load_tests(loader, tests, pattern):
 return Reverse([Case('test_a'), Case('test_b')])
""")
        (root / 'test_gpu_engine_beta.py').write_text(emit + """
class Case(unittest.TestCase):
 def test_ok(self):emit('beta')
""")
        package = root / 'zz_shard_package'
        package.mkdir()
        (package / '__init__.py').write_text(emit + """
class Case(unittest.TestCase):
 def test_package(self):emit('package')
class Custom(unittest.TestSuite):
 def run(self, result, debug=False):
  emit('custom_package')
  return super().run(result, debug)
def load_tests(loader, tests, pattern):
 from . import test_gpu_engine_nested
 return Custom([tests, loader.loadTestsFromModule(test_gpu_engine_nested)])
""")
        (package / 'test_gpu_engine_nested.py').write_text("""
import unittest
from . import emit
class Case(unittest.TestCase):
 def test_nested(self):emit('nested')
""")
        return log

    def test_real_process_blocks_keep_custom_run_load_tests_and_nested_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); log = self.fixture(root)
            result, terminal = self.isolated(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(terminal['passed'])
            self.assertEqual(terminal['selected_blocks'], [0, 1, 2])
            self.assertEqual(terminal['expected_test_cases'], 5)
            self.assertEqual(terminal['tests_run'], 5)
            self.assertEqual([row['tests_run'] for row in terminal['blocks']], [2, 1, 2])
            events = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([e['label'] for e in events],
                ['custom_reverse', 'alpha_b', 'alpha_a', 'beta', 'custom_package', 'package', 'nested'])
            self.assertEqual(len({e['pid'] for e in events}), 3)
            self.assertTrue(all(e['pid'] != os.getpid() and e['exe'] == sys.executable for e in events))
            self.assertEqual(len({events[i]['pid'] for i in (0, 1, 2)}), 1)
            self.assertEqual(len({events[i]['pid'] for i in (4, 5, 6)}), 1)

    def test_real_process_selection_preserves_original_top_level_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); log = self.fixture(root)
            result, terminal = self.isolated(root, 0, 2)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(terminal['selected_blocks'], [0, 2])
            self.assertEqual(terminal['tests_run'], 4)
            self.assertNotIn('beta', [json.loads(line)['label'] for line in log.read_text().splitlines()])

    def test_real_case_lookup_helper_named_id_does_not_override_discovered_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'test_gpu_engine_control.py').write_text(textwrap.dedent('''
                import unittest
                class Case(unittest.TestCase):
                 def id(self, name):return {'fixture': 7}[name]
                 def test_a(self):self.assertEqual(self.id('fixture'),7)
                 def test_b(self):pass
            '''))
            result,terminal=self.isolated(root)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue(terminal['passed'])
            self.assertEqual(terminal['expected_test_cases'],2)
            self.assertEqual(terminal['tests_run'],2)

    def test_real_process_skip_failure_empty_and_abrupt_exits_fail_closed(self):
        for label, body in [('skip', "self.skipTest('deliberate control')"),
                ('failure', "self.fail('deliberate control')"), ('exit_zero', 'os._exit(0)'),
                ('crash', 'os._exit(7)'), ('empty', 'pass')]:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = f"import os, unittest\nclass Case(unittest.TestCase):\n def test_control(self):{body}\n"
                if label == 'empty':
                    source += 'def load_tests(loader, tests, pattern):return unittest.TestSuite()\n'
                (root / 'test_gpu_engine_control.py').write_text(source)
                result, terminal = self.isolated(root)
                self.assertEqual(result.returncode, 1)
                self.assertFalse(terminal['passed'])
                self.assertFalse(terminal['blocks'][0]['valid_terminal_receipt'])
                if label in ('exit_zero', 'crash'):
                    self.assertIsNone(terminal['tests_run'])
                if label == 'skip':self.assertEqual(terminal['skipped'], 1)
                if label == 'failure':self.assertEqual(terminal['failures'], 1)

    def test_real_custom_run_dropping_or_repeating_cases_cannot_false_pass(self):
        for body, actual in [('self._tests[0](result)', 1),
                ('self._tests[0](result); self._tests[0](result)', 2)]:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'test_gpu_engine_control.py').write_text(textwrap.dedent(f"""
                    import unittest
                    class Case(unittest.TestCase):
                     def test_a(self):pass
                     def test_b(self):pass
                    class Custom(unittest.TestSuite):
                     def run(self, result, debug=False):
                      {body}
                      return result
                    def load_tests(loader, tests, pattern):return Custom([Case('test_a'), Case('test_b')])
                """))
                result, terminal = self.isolated(root)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(terminal['expected_test_cases'], 2)
                self.assertEqual(terminal['tests_run'], actual)
                self.assertFalse(terminal['passed'])

    def test_real_child_rediscovery_identity_or_count_drift_refuses_before_running(self):
        for choice in ("[Case('test_b')]", "[Case('test_a'), Case('test_b')]"):
            with self.subTest(choice=choice), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'test_gpu_engine_control.py').write_text(textwrap.dedent(f"""
                    import sys, unittest
                    class Case(unittest.TestCase):
                     def test_a(self):pass
                     def test_b(self):pass
                    def load_tests(loader, tests, pattern):
                     cases = {choice} if '--_child-plan' in sys.argv else [Case('test_a')]
                     return unittest.TestSuite(cases)
                """))
                result, terminal = self.isolated(root)
                self.assertEqual(result.returncode, 1)
                self.assertFalse(terminal['passed'])
                self.assertIsNone(terminal['tests_run'])
                self.assertIn('rediscovery manifest/index/count differs', result.stderr)

    def test_terminal_receipt_requires_exact_plan_index_counts_and_observed_cases(self):
        manifest = dict(kind='case', type='fixture.Case', count=1, id='fixture.Case.test_ok')
        plan = dict(manifest=[manifest])
        good = dict(schema='component-block-terminal.v1', plan_sha256=_digest(plan), block_index=0,
            block_manifest=manifest, declared_count=1, tests_run=1, observed_ids=[manifest['id']],
            failures=0, errors=0, skipped=0, expected_failures=0, unexpected_successes=0, successful=True)
        self.assertTrue(_valid_receipt(good, plan, 0))
        for key, value in [('plan_sha256', 'wrong'), ('block_index', True), ('block_index', 1),
                ('declared_count', True), ('tests_run', 0), ('tests_run', 2), ('tests_run', True),
                ('observed_ids', []), ('observed_ids', [manifest['id']] * 2), ('skipped', 1),
                ('failures', 1), ('errors', 1), ('unexpected_successes', 1), ('successful', False)]:
            with self.subTest(key=key, value=value):
                self.assertFalse(_valid_receipt(dict(good, **{key:value}), plan, 0))
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
