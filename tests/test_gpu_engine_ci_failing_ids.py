"""scripts/ci_failing_ids.py on saved log text (no network, no test execution)."""
from argparse import Namespace
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ci_failing_ids', ROOT / 'scripts/ci_failing_ids.py')
ci = importlib.util.module_from_spec(spec);spec.loader.exec_module(ci)

T = '2026-11-09T13:26:50.1Z '
LOG = '\n'.join(f'{job}\tRun\t{T}{text}' for job, text in [
    ('shard 0', '{"shard": 0, "count": 2, "discovery_blocks": 4, "selected_blocks": [0, 2], "test_cases": 5}'),
    ('shard 0', '{"process_block": 0, "case_index": null, "declared_count": 3}'),
    ('shard 0', 'test_a (test_x.A.test_a) ... ok'),
    ('shard 0', 'test_b (test_x.A.test_b) ... '),
    ('shard 0', 'some interleaved output'),
    ('shard 0', 'setUpClass (test_y.B) ... ERROR'),
    ('shard 0', 'FAIL: test_b (test_x.A.test_b)'),
    ('shard 0', 'ERROR: setUpClass (test_y.B)'),
    ('shard 0', 'FAIL: test_c (test_x.A.test_c) (case=1)'),
    ('shard 0', 'Ran 3 tests in 0.1s'),
    ('shard 0', '{"process_block": 2, "case_index": null, "declared_count": 2}'),
    ('shard 0', 'unittest.case.FunctionTestCase (test_fn) ... ok'),
    ('shard 0', 'test_q (test_x.Q.test_q) ... ok'),
    ('shard 0', 'Ran 2 tests in 0.1s'),
    ('shard 0', '##[error]Process completed with exit code 1.'),
    # A killed job: its block never printed Ran, so FAIL: headers never came.
    ('shard 1', '{"shard": 1, "count": 2, "discovery_blocks": 4, "selected_blocks": [1, 3], "test_cases": 4}'),
    ('shard 1', '{"process_block": 1, "case_index": null, "declared_count": 3}'),
    ('shard 1', 'test_d (test_w.D.test_d) ... FAIL'),
    ('shard 1', 'test_e (test_w.D.test_e) ... ok'),
    ('shard 1', 'test_f (test_w.D.test_f) ... '),
    ('shard 1', '##[error]Process completed with exit code 143.'),
    ('shard 1', '##[error]The runner has received a shutdown signal.'),
])


class ParseTests(unittest.TestCase):
    def test_settlement_needs_ran_or_an_inline_result(self):
        jobs = ci.parse_log(LOG.splitlines(), {'test_fn': 'test_fx'})
        zero, one = jobs['shard 0'], jobs['shard 1']
        self.assertEqual(zero['settled'], {'test_x.A.test_a': 'pass', 'test_x.A.test_b': 'fail', 'test_y.B.setUpClass': 'fail',
            'test_x.A.test_c': 'fail', 'test_fx.test_fn': 'pass', 'test_x.Q.test_q': 'pass'})
        self.assertEqual(zero['unsettled'], set());self.assertEqual(zero['blocks_finished'], {0, 2})
        # The inline FAIL in the killed block is a failure, not a pass; test_f is unknown.
        self.assertEqual(one['settled'], {'test_w.D.test_d': 'fail', 'test_w.D.test_e': 'pass'})
        self.assertEqual(one['unsettled'], {'test_w.D.test_f'});self.assertTrue(one['killed'])
        meta = dict(jobs=[dict(name='shard 0', conclusion='failure'), dict(name='shard 1', conclusion='failure')])
        summary = ci.summarize('1', meta, jobs)
        self.assertEqual(summary['incomplete_jobs'], ['shard 1']);self.assertEqual(summary['jobs']['shard 1']['unfinished_blocks'], [1, 3])
        self.assertEqual(ci.test_id('unittest.case.FunctionTestCase', 'test_g', None), 'FunctionTestCase.test_g')


def run_main(*argv):
    return ci.main([str(a) for a in argv])


class ExtractDiffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='ci-ids-test-'))
        tree = self.tmp / 'tree' / 'tests';tree.mkdir(parents=True)
        for name in ('test_x', 'test_y', 'test_w', 'test_v'):(tree / f'{name}.py').write_text('')
        (tree / 'test_fx.py').write_text('import unittest\nunittest.FunctionTestCase\ndef test_fn():\n    pass\n')
        import subprocess
        subprocess.run(['git', 'init', '-q', str(self.tmp / 'tree')], check=True)
        subprocess.run(['git', '-C', str(self.tmp / 'tree'), '-c', 'core.hooksPath=/dev/null', '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q',
                        '--allow-empty', '-m', 'x'], check=True)
        (self.tmp / 'a.log').write_text(LOG)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def extract(self, *extra, logs=('a.log',), heads=('h1',)):
        args = ['extract', *[str(i) for i in range(len(logs))], '--out', self.tmp / 'base.txt', '--tree', self.tmp / 'tree',
                '--pattern', 'test_*.py']
        for log, head in zip(logs, heads):args += ['--log', self.tmp / log, '--log-head', head]
        return run_main(*args, *extra), json.loads((self.tmp / 'base.json').read_text())

    def test_extract_reports_coverage_by_module(self):
        code, side = self.extract()
        self.assertEqual(code, 0)
        self.assertEqual((self.tmp / 'base.txt').read_text().split(),
                         ['test_w.D.test_d', 'test_x.A.test_b', 'test_x.A.test_c', 'test_y.B.setUpClass'])
        self.assertEqual(side['coverage']['modules_never_reached'], ['test_v'])
        self.assertEqual(side['coverage']['modules_with_unsettled_tests'], ['test_w'])
        self.assertEqual(side['provenance']['test_w.D.test_d']['head'], 'h1')
        run_main('supplement', '--sidecar', self.tmp / 'base.json', '--out', self.tmp / 'mods.txt', '--force', 'test_x')
        self.assertEqual((self.tmp / 'mods.txt').read_text().split(), ['test_v', 'test_w', 'test_x'])

    def test_cross_head_fill_is_refused_unless_allowed(self):
        (self.tmp / 'b.log').write_text(f'shard 9\tRun\t{T}test_f (test_w.D.test_f) ... FAIL\n')
        with self.assertRaises(SystemExit):self.extract(logs=('a.log', 'b.log'), heads=('h1', 'h2'))
        code, side = self.extract('--allow-cross-head', logs=('a.log', 'b.log'), heads=('h1', 'h2'))
        self.assertEqual(side['provenance']['test_w.D.test_f']['head'], 'h2');self.assertTrue(side['cross_head'])

    def test_pod_results_fill_only_unsettled_tests_and_diff_states(self):
        pod = dict(schema=ci.SCHEMA + '.pod', tree_commit='c' * 40, modules=['test_v', 'test_w'], unsettled=[],
                   settled={'test_w.D.test_f': 'pass', 'test_w.D.test_d': 'pass', 'test_v.V.test_v': 'fail'}, failing=[])
        (self.tmp / 'pod.json').write_text(json.dumps(pod))
        code, side = self.extract('--pod', self.tmp / 'pod.json')
        self.assertEqual(side['provenance']['test_v.V.test_v']['source'], 'pod')
        self.assertEqual(side['provenance']['test_w.D.test_d']['source'], 'ci')   # CI settled it first
        self.assertEqual(side['coverage']['modules_never_reached'], []);self.assertEqual(side['unsettled_tests'], [])
        code, side2 = self.extract('--pod', self.tmp / 'pod.json', '--pod-module', 'test_w')    # test_w changed since CI
        self.assertEqual(side2['provenance'].get('test_w.D.test_d'), None)                      # the pod's pass wins
        self.assertEqual(side2['settled_outcomes']['test_w.D.test_d'], 'pass')
        code, side = self.extract('--pod', self.tmp / 'pod.json')
        base = self.tmp / 'base.txt'
        (self.tmp / 'pr.log').write_text(LOG)
        # Same logs as the PR: the killed job leaves test_f unsettled and test_v unreached -> INCOMPLETE.
        code = run_main('diff', '--baseline', base, '--pr-run', '1', '--log', self.tmp / 'pr.log', '--log-head', 'h3',
                        '--tree', self.tmp / 'tree', '--pattern', 'test_*.py')
        self.assertEqual(code, 2)
        code = run_main('diff', '--baseline', base, '--pr-run', '1', '--log', self.tmp / 'pr.log', '--log-head', 'h3',
                        '--pr-pod', self.tmp / 'pod.json', '--tree', self.tmp / 'tree', '--pattern', 'test_*.py')
        self.assertEqual(code, 0)
        (self.tmp / 'new.txt').write_text('test_x.A.test_a\n')                   # settled pass in the baseline
        self.assertEqual(run_main('diff', '--baseline', base, '--pr-ids', self.tmp / 'new.txt'), 1)
        (self.tmp / 'unknown.txt').write_text('test_q.Q.test_q\n')               # never settled in the baseline
        self.assertEqual(run_main('diff', '--baseline', base, '--pr-ids', self.tmp / 'unknown.txt'), 2)


if __name__ == '__main__':
    unittest.main()
