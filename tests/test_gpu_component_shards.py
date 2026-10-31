"""Component shards retain discovery blocks and fail closed."""
import ast
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

from scripts.run_gpu_component_shard import (
    partition_suite, run_suite, run_process_blocks, _valid_receipt, _digest, _resource_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


class ComponentShardTests(unittest.TestCase):
    def test_heavy_runtime_cache_boundaries_keep_class_inputs_and_failed_cases(self):
        # Exercise the real fixture hook through unittest without importing JAX,
        # constructing a source session, or executing the scientific test body.
        fixtures = (
            ('test_gpu_engine_execution_runtime_plumbing.py', 'ExecutionRuntimePlumbingTests'),
            ('test_gpu_engine_status_residual_runtime.py', 'StatusResidualRuntimeTests'),
            ('test_gpu_engine_status_kernel_owner_drain.py', 'StatusKernelOwnerDrainTests'),
        )
        for filename, name in fixtures:
            with self.subTest(fixture=name):
                path = ROOT / 'tests' / filename
                source = ast.parse(path.read_text())
                case = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == name)
                hook = next(node for node in case.body if isinstance(node, ast.FunctionDef) and node.name == 'setUp')
                events = []
                namespace = {
                    'jax': mock.Mock(clear_caches=lambda: events.append('clear')),
                    'gc': mock.Mock(collect=lambda: events.append('collect')),
                }
                exec(compile(ast.Module(body=[hook], type_ignores=[]), str(path), 'exec'), namespace)
                shared_input = object()
                class Synthetic(unittest.TestCase):
                    setUp = namespace['setUp']
                    @classmethod
                    def setUpClass(cls):
                        events.append('class_setup')
                        cls.prepared = shared_input
                    def test_first(self):
                        self.assertIs(self.prepared, shared_input)
                        events.append('first')
                        self.fail('intentional control')
                    def test_second(self):
                        self.assertIs(self.prepared, shared_input)
                        events.append('second')
                result = unittest.TestResult()
                unittest.TestSuite([Synthetic('test_first'), Synthetic('test_second')]).run(result)
                self.assertEqual(events, ['class_setup', 'clear', 'collect', 'first', 'clear', 'collect', 'second'])
                self.assertEqual(result.testsRun, 2)
                self.assertEqual(len(result.failures), 1)
                self.assertEqual(result.errors, [])
                self.assertEqual(result.skipped, [])

    def isolated(self, root, index=0, count=1, resource_interval_seconds=0):
        # Actual fresh parent AND child interpreters, no mocked process runner
        # and no real component discovery/JAX imports in these tiny fixtures.
        code = """
import sys, unittest
from scripts.run_gpu_component_shard import run_process_blocks
suite = unittest.TestLoader().discover(sys.argv[1], 'test_gpu_engine*.py')
raise SystemExit(run_process_blocks(suite, int(sys.argv[2]), int(sys.argv[3]), start_dir=sys.argv[1],
    resource_interval_seconds=float(sys.argv[4])))
"""
        environment = dict(os.environ)
        environment['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(ROOT), environment.get('PYTHONPATH', '')])
        result = subprocess.run([sys.executable, '-c', code, str(root), str(index), str(count), str(resource_interval_seconds)],
            cwd=ROOT, env=environment, text=True, capture_output=True, timeout=30)
        summaries = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        terminal = [row for row in summaries if row.get('process_blocks_terminal')]
        self.assertEqual(len(terminal), 1, (result.returncode, result.stdout, result.stderr))
        return result, terminal[0]

    def test_resource_samples_preserve_real_success_skip_and_crash_dispositions(self):
        for body, expected, child_exit in [('pass', 0, 0), ("self.skipTest('control')", 1, 1),
                ('os._exit(7)', 1, 7), ('os.kill(os.getpid(), signal.SIGTERM)', 1, -15)]:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'test_gpu_engine_sample.py').write_text(
                    'import os, signal, time, unittest\nclass Case(unittest.TestCase):\n'
                    f' def test_sample(self):\n  time.sleep(0.12)\n  {body}\n')
                result, terminal = self.isolated(root, resource_interval_seconds=0.02)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(terminal['passed'], expected == 0)
                self.assertEqual(terminal['blocks'][0]['returncode'], child_exit)
                samples = [json.loads(line) for line in result.stdout.splitlines()
                    if line.startswith('{') and json.loads(line).get('resource_snapshot')]
                self.assertGreaterEqual(len(samples), 3)
                self.assertTrue(all(sample['process_block'] == 0 for sample in samples))
                self.assertTrue(all(sample['monotonic_ns'] > 0 for sample in samples))
                if child_exit in (7, -15):
                    self.assertIsNone(terminal['tests_run'])
                else:
                    self.assertEqual(terminal['tests_run'], 1)

    def test_linux_resource_samples_only_emit_allowed_numeric_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); proc = root / 'proc'; groups = root / 'cgroups'
            proc.mkdir(); groups.mkdir()
            (proc / 'meminfo').write_text('MemTotal: 1000 kB\nMemAvailable: 700 kB\nPrivate: 999 kB\n')
            (proc / 'vmstat').write_text('oom_kill 4\nprivate 999\n')
            for pid in (101, 102):
                folder = proc / str(pid); (folder / 'task' / str(pid)).mkdir(parents=True)
                stat = ['S'] + ['0'] * 21
                stat[1] = '101' if pid == 102 else '1'
                stat[11], stat[12], stat[19] = '123', '45', '999'
                (folder / 'stat').write_text(f'{pid} (PRIVATE_PROCESS_NAME) ' + ' '.join(stat))
                (folder / 'status').write_text('Name: PRIVATE_NAME\nVmRSS: 123 kB\nVmHWM: 234 kB\n'
                    'VmSize: 345 kB\nThreads: 6\nPrivate: 999\n')
                (folder / 'cmdline').write_text('PRIVATE_ARGUMENT')
                (folder / 'environ').write_text('PRIVATE_ENVIRONMENT')
                (folder / 'task' / str(pid) / 'children').write_text('102' if pid == 101 else '')
            (proc / '101' / 'cgroup').write_text('0::/PRIVATE_CGROUP_PATH\n')
            group = groups / 'PRIVATE_CGROUP_PATH'; group.mkdir()
            for name, value in [('memory.current', '500'), ('memory.peak', '900'),
                    ('memory.max', 'max'), ('memory.swap.current', '20'),
                    ('memory.events', 'oom 1\noom_kill 2\nprivate 999\n')]:
                (group / name).write_text(value)
            read_text = Path.read_text
            def checked_read(path, *args, **kwargs):
                self.assertNotIn(path.name, ('cmdline', 'environ'))
                return read_text(path, *args, **kwargs)
            with mock.patch.object(Path, 'read_text', autospec=True, side_effect=checked_read):
                sample = _resource_snapshot(101, 8, proc=proc, cgroups=groups)
            self.assertNotIn('PRIVATE', json.dumps(sample))
            self.assertEqual(sample['vm'], dict(total_bytes=1024000, available_bytes=716800, oom_kill=4))
            self.assertEqual(sample['cgroup'], dict(current_bytes=500, peak_bytes=900, swap_bytes=20, oom=1, oom_kill=2))
            self.assertEqual([row['pid'] for row in sample['processes']], [101, 102])
            for row in sample['processes']:
                self.assertEqual(row['user_ticks'], 123)
                self.assertEqual(row['system_ticks'], 45)
                self.assertEqual(row['start_ticks'], 999)
                self.assertEqual(row['rss_bytes'], 123 * 1024)
                self.assertEqual(row['peak_rss_bytes'], 234 * 1024)
                self.assertEqual(row['virtual_bytes'], 345 * 1024)
                self.assertEqual(row['threads'], 6)
                self.assertTrue(all(type(value) is int for value in row.values()))
            # Disappearing processes or unavailable Linux/cgroup counters do
            # not affect the block's assertions or exit status.
            missing = _resource_snapshot(404, 8, proc=root / 'missing', cgroups=groups)
            self.assertEqual((missing['processes'], missing['vm'], missing['cgroup']), ([], {}, {}))
            (proc / '101' / 'cgroup').write_text('0::/../../PRIVATE_ESCAPE\n')
            self.assertEqual(_resource_snapshot(101, 8, proc=proc, cgroups=groups)['cgroup'], {})

    def test_resource_sampling_interval_must_be_finite_and_nonnegative(self):
        for interval in (-1, float('nan'), float('inf')):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                run_process_blocks(None, 0, 1, start_dir='.', resource_interval_seconds=interval)

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
            result, terminal = self.isolated(root, resource_interval_seconds=0.02)
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
