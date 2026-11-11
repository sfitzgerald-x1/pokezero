"""CI observer checks; these do not enter component discovery or import JAX."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from scripts.run_component_memory_check import run, snapshot


class ComponentMemoryCheckTests(unittest.TestCase):
    def test_sampler_sums_descendants_once_and_keeps_kernel_peak_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for pid, rss, hwm, children in ((11, 10, 15, '12'), (12, 20, 25, '11')):
                directory = root / 'proc' / str(pid)
                (directory/'task'/str(pid)).mkdir(parents=True)
                (directory/'status').write_text(f'VmRSS: {rss} kB\nVmHWM: {hwm} kB\n')
                (directory/'task'/str(pid)/'children').write_text(children)
            (root/'cgroup').mkdir()
            (root/'cgroup/memory.peak').write_text('99999')
            row = snapshot(11, proc=root/'proc', cgroup=root/'cgroup')
            self.assertEqual(row['tree_rss_bytes'], 30*1024)
            self.assertEqual(row['process_peak_rss_bytes'], 25*1024)
            self.assertEqual(row['cgroup_memory_peak_bytes'], 99999)

    def observed(self, child_code, rss, limit):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch('scripts.run_component_memory_check.snapshot', return_value=dict(
                    tree_rss_bytes=rss, process_peak_rss_bytes=rss,
                    cgroup_memory_peak_bytes=99)), mock.patch(
                    'scripts.run_component_memory_check.resource.getrusage',
                    return_value=mock.Mock(ru_maxrss=0)):
                code = run([sys.executable, '-c', f'raise SystemExit({child_code})'],
                    output=root/'data', label='synthetic', summary=root/'summary.md',
                    limit_bytes=limit, interval=.01)
            rows = [json.loads(line) for line in (root/'data/samples.jsonl').read_text().splitlines()]
            report = json.loads((root/'data/report.json').read_text())
            return code, report, rows, (root/'summary.md').read_text()

    def test_success_reports_peak_limit_wall_and_durable_terminal(self):
        code, report, rows, summary = self.observed(0, 75, 100)
        self.assertEqual(code, 0)
        self.assertTrue(report['memory_pass'])
        self.assertEqual(report['peak_rss_bytes'], 75)
        self.assertEqual(report['bar_bytes'], 75)
        self.assertEqual(rows[-1]['child_exit_code'], 0)
        self.assertTrue(rows[-1]['terminal'])
        self.assertGreater(report['wall_seconds'], 0)
        self.assertIn('PASS', summary)

    def test_above_bar_fails_without_changing_child_result(self):
        code, report, rows, summary = self.observed(0, 76, 100)
        self.assertEqual(code, 1)
        self.assertEqual(report['child_exit_code'], 0)
        self.assertFalse(report['memory_pass'])
        self.assertIn('FAIL', summary)

    def test_child_failure_cannot_be_hidden_by_low_memory(self):
        code, report, _, _ = self.observed(13, 1, 100)
        self.assertEqual(code, 13)
        self.assertEqual(report['child_exit_code'], 13)
        self.assertTrue(report['memory_pass'])

    def test_invalid_interval_does_not_launch_child(self):
        with mock.patch('subprocess.Popen') as launch:
            with self.assertRaises(ValueError):
                run(['unused'], output='unused', label='unused', limit_bytes=100, interval=0)
            launch.assert_not_called()
