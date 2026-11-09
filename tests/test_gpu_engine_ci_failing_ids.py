"""scripts/ci_failing_ids.py on saved log text (no network)."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ci_failing_ids', ROOT / 'scripts/ci_failing_ids.py')
ci = importlib.util.module_from_spec(spec);spec.loader.exec_module(ci)

LOG = '\n'.join([
    'shard 0\tRun\t2026-11-09T13:26:50.1Z test_a (test_x.A.test_a) ... ok',
    'shard 0\tRun\t2026-11-09T13:26:50.2Z test_b (test_x.A.test_b) ... FAIL',
    'shard 0\tRun\t2026-11-09T13:26:50.3Z setUpClass (test_y.B) ... ERROR',
    'shard 0\tRun\t2026-11-09T13:26:50.4Z FAIL: test_b (test_x.A.test_b)',
    'shard 0\tRun\t2026-11-09T13:26:50.5Z ERROR: setUpClass (test_y.B)',
    'shard 0\tRun\t2026-11-09T13:26:50.6Z FAIL: test_c (test_x.A.test_c) (case=1)',
    'shard 0\tRun\t2026-11-09T13:26:50.7Z ERROR: test_z (unittest.loader._FailedTest.test_z)',
    'shard 0\tRun\t2026-11-09T14:00:00.0Z ##[error]Process completed with exit code 1.',
    'shard 1\tRun\t2026-11-09T13:30:00.0Z test_d (test_w.D.test_d)',
    'shard 1\tRun\t2026-11-09T14:02:47.7Z ##[error]Process completed with exit code 143.',
    'shard 1\tRun\t2026-11-09T14:02:47.8Z ##[error]The runner has received a shutdown signal.',
])


class ParseTests(unittest.TestCase):
    def test_ids_starts_and_killed_jobs(self):
        jobs = ci.parse_log(LOG.splitlines())
        self.assertEqual(jobs['shard 0']['failing'], {'test_x.A.test_b', 'test_y.B.setUpClass', 'test_x.A.test_c',
                                                       'unittest.loader._FailedTest.test_z'})
        self.assertEqual(jobs['shard 0']['started'], {'test_x.A.test_a', 'test_x.A.test_b', 'test_y.B.setUpClass'})
        self.assertFalse(jobs['shard 0']['killed']);self.assertEqual(jobs['shard 0']['exit_codes'], {1})
        self.assertTrue(jobs['shard 1']['killed']);self.assertEqual(jobs['shard 1']['started'], {'test_w.D.test_d'})
        meta = dict(jobs=[dict(name='shard 0', conclusion='failure'), dict(name='shard 1', conclusion='failure')])
        self.assertEqual(ci.summarize('1', meta, jobs)['incomplete_jobs'], ['shard 1'])

    def test_later_runs_only_fill_tests_the_killed_job_never_settled(self):
        other = '\n'.join([
            'shard 9\tRun\t2026-11-09T13:00:00.0Z test_d (test_w.D.test_d) ... FAIL',      # shard 1 started it last: unsettled there
            'shard 9\tRun\t2026-11-09T13:00:01.0Z test_e (test_w.D.test_e) ... ERROR',     # never reached by the killed job
            'shard 9\tRun\t2026-11-09T13:00:02.0Z test_b (test_x.A.test_b) ... ok',        # settled (failing) by the first run
            'shard 9\tRun\t2026-11-09T13:00:03.0Z FAIL: test_d (test_w.D.test_d)',
            'shard 9\tRun\t2026-11-09T13:00:03.0Z ERROR: test_e (test_w.D.test_e)',
        ])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp);(tmp / 'a.log').write_text(LOG);(tmp / 'b.log').write_text(other)
            ci.main(['extract', '1', '2', '--log', str(tmp / 'a.log'), '--log', str(tmp / 'b.log'), '--out', str(tmp / 'x.txt')])
            ids = set((tmp / 'x.txt').read_text().split())
            self.assertIn('test_x.A.test_b', ids)
            self.assertIn('test_w.D.test_d', ids);self.assertIn('test_w.D.test_e', ids)
            sidecar = json.loads((tmp / 'x.json').read_text())
            self.assertEqual(sidecar['failing_from_later_runs'], ['test_w.D.test_d', 'test_w.D.test_e'])
            self.assertEqual(sidecar['unsettled_tests'], [])

    def test_extract_and_diff_on_saved_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp);(tmp / 'run.log').write_text(LOG)
            self.assertEqual(ci.main(['extract', '7', '--log', str(tmp / 'run.log'), '--out', str(tmp / 'base.txt'), '--shard-prefix', 'shard']), 0)
            ids = (tmp / 'base.txt').read_text().split()
            self.assertEqual(len(ids), 4);self.assertEqual(ids, sorted(ids))
            sidecar = json.loads((tmp / 'base.json').read_text())
            self.assertEqual(sidecar['shards_never_complete'], ['shard 1'])
            (tmp / 'pr.txt').write_text('test_x.A.test_b\ntest_new.N.test_n\n')
            self.assertEqual(ci.main(['diff', '--baseline', str(tmp / 'base.txt'), '--pr-ids', str(tmp / 'pr.txt')]), 1)
            (tmp / 'pr.txt').write_text('test_x.A.test_b\n')
            self.assertEqual(ci.main(['diff', '--baseline', str(tmp / 'base.txt'), '--pr-ids', str(tmp / 'pr.txt')]), 0)


if __name__ == '__main__':
    unittest.main()
