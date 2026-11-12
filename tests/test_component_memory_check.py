"""CI observer checks; these do not enter component discovery or import JAX."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from scripts.run_component_memory_check import (CGROUP_SOURCE, PF_FORKNOEXEC, PSS_SOURCE,
    own_cgroup, run, snapshot)


def fake_process(root, pid, *, rss, hwm, pss=0, flags=0, tasks=None):
    """A /proc/<pid> with status, stat, smaps_rollup and per-task children files."""
    directory = root / 'proc' / str(pid)
    directory.mkdir(parents=True)
    (directory/'status').write_text(f'VmRSS: {rss} kB\nVmHWM: {hwm} kB\n')
    (directory/'stat').write_text(f'{pid} (py thon) S 1 1 1 0 -1 {flags} 0 0 0 0 1 1\n')
    (directory/'smaps_rollup').write_text(f'Rss: {rss} kB\nPss: {pss} kB\n')
    for task, children in (tasks or {pid: ''}).items():
        (directory/'task'/str(task)).mkdir(parents=True)
        (directory/'task'/str(task)/'children').write_text(children)


class ComponentMemoryCheckTests(unittest.TestCase):
    def tree(self, root, observer, *, child_flags=0, grandchild_on_thread=True):
        # observer -> 11 -> 12 (started from 11's second thread) -> 13 (forked, not exec'd)
        fake_process(root, observer, rss=1, hwm=1, pss=1, tasks={observer: '11'})
        fake_process(root, 11, rss=10, hwm=15, pss=8,
                     tasks={11: '' if grandchild_on_thread else '12', 99: '12' if grandchild_on_thread else ''})
        fake_process(root, 12, rss=20, hwm=25, pss=20, tasks={12: '13'})
        fake_process(root, 13, rss=20, hwm=20, pss=20, flags=child_flags)

    def test_pss_fallback_counts_a_vfork_child_once_and_a_forked_child_by_its_pss(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch('os.getpid', return_value=7):
            root = Path(folder)
            self.tree(root, 7, child_flags=PF_FORKNOEXEC)
            calls = []
            def shares(first, second):
                calls.append((first, second))
                return True
            row = snapshot(11, proc=root/'proc', shares=shares)
            # Summed RSS (not gating) counts the vfork child's shared pages again.
            self.assertEqual(row['tree_rss_bytes'], 51*1024)
            self.assertEqual(row['process_peak_rss_bytes'], 25*1024)
            # Child 12 was found through thread 99; 13 shares 12's mm and is counted once.
            self.assertEqual(row['memory_bytes'], 29*1024)
            self.assertEqual(row['tree_pss_bytes'], 29*1024)
            self.assertEqual(calls, [(12, 13)])
            # A forked child with its own address space keeps its PSS.
            row = snapshot(11, proc=root/'proc', shares=lambda first, second: False)
            self.assertEqual(row['memory_bytes'], 49*1024)

    def test_exec_d_children_are_never_compared_and_all_threads_are_walked(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch('os.getpid', return_value=7):
            root = Path(folder)
            self.tree(root, 7, child_flags=0)
            shares = mock.Mock(return_value=True)
            row = snapshot(11, proc=root/'proc', shares=shares)
            shares.assert_not_called()
            self.assertEqual(row['memory_bytes'], 49*1024)
            self.assertEqual(row['tree_rss_bytes'], 51*1024)

    def test_cgroup_gates_on_current_less_file_cache_and_pss_is_optional(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch('os.getpid', return_value=7):
            root = Path(folder)
            self.tree(root, 7, child_flags=PF_FORKNOEXEC)
            group = root/'cgroup'/'job'
            group.mkdir(parents=True)
            (group/'memory.current').write_text('1000\n')
            (group/'memory.stat').write_text('anon 600\nfile 400\nactive_file 150\ninactive_file 200\nshmem 50\n')
            row = snapshot(11, proc=root/'proc', group=group, shares=lambda *_: True)
            self.assertEqual(row['memory_bytes'], 650)
            self.assertEqual((row['cgroup_current_bytes'], row['cgroup_file_cache_bytes']), (1000, 350))
            self.assertNotIn('tree_pss_bytes', row)
            self.assertEqual(row['tree_rss_bytes'], 51*1024)
            row = snapshot(11, proc=root/'proc', group=group, pss=True, shares=lambda *_: True)
            self.assertEqual((row['memory_bytes'], row['tree_pss_bytes']), (650, 29*1024))
            # A counter that disappears falls back to the larger summed RSS, never to zero.
            (group/'memory.stat').unlink()
            self.assertEqual(snapshot(11, proc=root/'proc', group=group)['memory_bytes'], 51*1024)

    def test_own_cgroup_needs_v2_counters_inside_the_mount(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'proc'/'self').mkdir(parents=True)
            (root/'cgroup'/'system.slice'/'runner').mkdir(parents=True)
            (root/'proc'/'self'/'cgroup').write_text('0::/system.slice/runner\n')
            self.assertIsNone(own_cgroup(proc=root/'proc', cgroup=root/'cgroup'))
            for name in ('memory.current', 'memory.stat'):
                (root/'cgroup'/'system.slice'/'runner'/name).write_text('0\n')
            self.assertEqual(own_cgroup(proc=root/'proc', cgroup=root/'cgroup'),
                             (root/'cgroup'/'system.slice'/'runner').resolve())
            (root/'proc'/'self'/'cgroup').write_text('0::/../../elsewhere\n')
            self.assertIsNone(own_cgroup(proc=root/'proc', cgroup=root/'cgroup'))
            (root/'proc'/'self'/'cgroup').write_text('12:memory:/legacy\n')
            self.assertIsNone(own_cgroup(proc=root/'proc', cgroup=root/'cgroup'))

    def observed(self, child_code, memory, limit, *, rss=None, hwm=None, group=None):
        rss = memory if rss is None else rss
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch('scripts.run_component_memory_check.snapshot', return_value=dict(
                    tree_rss_bytes=rss, process_peak_rss_bytes=memory if hwm is None else hwm,
                    memory_bytes=memory)), mock.patch(
                    'scripts.run_component_memory_check.own_cgroup', return_value=group), mock.patch(
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
        self.assertEqual((report['peak_bytes'], report['peak_rss_bytes']), (75, 75))
        self.assertEqual(report['bar_bytes'], 75)
        self.assertEqual(report['memory_source'], PSS_SOURCE)
        self.assertEqual(rows[-1]['child_exit_code'], 0)
        self.assertTrue(rows[-1]['terminal'])
        self.assertGreater(report['wall_seconds'], 0)
        self.assertIn('PASS', summary)

    def test_fork_double_count_in_summed_rss_does_not_fail_the_gate(self):
        code, report, rows, summary = self.observed(0, 70, 100, rss=140, hwm=70, group=Path('/unused'))
        self.assertEqual(code, 0)
        self.assertTrue(report['memory_pass'])
        self.assertEqual((report['peak_bytes'], report['peak_rss_bytes']), (70, 140))
        self.assertEqual(report['memory_source'], CGROUP_SOURCE)
        self.assertEqual(rows[0]['tree_rss_bytes'], 140)

    def test_above_bar_fails_without_changing_child_result(self):
        code, report, rows, summary = self.observed(0, 76, 100)
        self.assertEqual(code, 1)
        self.assertEqual(report['child_exit_code'], 0)
        self.assertFalse(report['memory_pass'])
        self.assertIn('FAIL', summary)

    def test_single_process_high_water_mark_between_samples_still_fails(self):
        code, report, _, _ = self.observed(0, 50, 100, hwm=80)
        self.assertEqual(code, 1)
        self.assertEqual(report['peak_bytes'], 80)
        self.assertFalse(report['memory_pass'])

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
