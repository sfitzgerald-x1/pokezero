"""CI observer checks; these do not enter component discovery or import JAX."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import os
import subprocess
import threading
import time

from scripts.run_component_memory_check import (CGROUP_SOURCE, PF_FORKNOEXEC, PSS_SOURCE, REQUIRED_STAT,
    cgroup_memory, effective_limit, own_cgroup, run, shares_address_space, snapshot)


def fake_process(root, pid, *, rss, hwm, pss=0, flags=0, tasks=None, file=0):
    """A /proc/<pid> with status, stat, smaps_rollup and per-task children files."""
    directory = root / 'proc' / str(pid)
    directory.mkdir(parents=True)
    (directory/'status').write_text(f'VmRSS: {rss} kB\nVmHWM: {hwm} kB\nRssAnon: {rss - file} kB\n'
                                    f'RssFile: {file} kB\nRssShmem: 0 kB\n')
    (directory/'stat').write_text(f'{pid} (py thon) S 1 1 1 0 -1 {flags} 0 0 0 0 1 1\n')
    (directory/'smaps_rollup').write_text(f'Rss: {rss} kB\nPss: {pss} kB\n')
    for task, children in (tasks or {pid: ''}).items():
        (directory/'task'/str(task)).mkdir(parents=True)
        (directory/'task'/str(task)/'children').write_text(children)


class ComponentMemoryCheckTests(unittest.TestCase):
    def setUp(self):
        # Exact arithmetic below; the stale-stat allowance has its own test.
        patcher = mock.patch('scripts.run_component_memory_check.STAT_ERROR_BYTES', 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tree(self, root, observer, *, child_flags=0, grandchild_on_thread=True):
        # observer -> 11 -> 12 (started from 11's second thread) -> 13 (forked, not exec'd)
        fake_process(root, observer, rss=1, hwm=1, pss=1, tasks={observer: '11'})
        fake_process(root, 11, rss=10, hwm=15, pss=8, file=4,
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
            self.assertEqual((row['tree_rss_anon_bytes'], row['tree_rss_file_bytes'], row['tree_rss_shmem_bytes']),
                             (47*1024, 4*1024, 0))
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
            (group/'memory.stat').write_text('anon 600\nfile 400\nkernel 30\nshmem 50\nfile_mapped 40\n'
                                             'file_dirty 20\nfile_writeback 10\nactive_file 150\ninactive_file 200\n'
                                             'pgfault 7\n')
            row = snapshot(11, proc=root/'proc', group=group, shares=lambda *_: True)
            # Only clean, unmapped file cache is dropped: 150 + 200 - 20 dirty - 10 writeback - 40 mapped.
            self.assertEqual(row['memory_bytes'], 720)
            self.assertEqual((row['cgroup_current_bytes'], row['cgroup_clean_cache_bytes']), (1000, 280))
            self.assertEqual(row['cgroup_stat'], dict(anon=600, file=400, shmem=50, kernel=30, file_mapped=40,
                file_dirty=20, file_writeback=10, active_file=150, inactive_file=200))
            self.assertNotIn('tree_pss_bytes', row)
            self.assertEqual(row['tree_rss_bytes'], 51*1024)
            row = snapshot(11, proc=root/'proc', group=group, pss=True, shares=lambda *_: True)
            self.assertEqual((row['memory_bytes'], row['tree_pss_bytes']), (720, 29*1024))
            # Of the readings before and after the /proc walk, the higher one gates.
            real = __import__('scripts.run_component_memory_check', fromlist=['cgroup_memory']).cgroup_memory
            readings = iter([dict(real(group), cgroup_memory_bytes=900), dict(real(group), cgroup_memory_bytes=700)])
            with mock.patch('scripts.run_component_memory_check.cgroup_memory', side_effect=lambda *_: next(readings)):
                self.assertEqual(snapshot(11, proc=root/'proc', group=group)['memory_bytes'], 900)
            readings = iter([dict(real(group), cgroup_memory_bytes=700), dict(real(group), cgroup_memory_bytes=900)])
            with mock.patch('scripts.run_component_memory_check.cgroup_memory', side_effect=lambda *_: next(readings)):
                self.assertEqual(snapshot(11, proc=root/'proc', group=group)['memory_bytes'], 900)
            # A counter that disappears falls back to the larger summed RSS, never to zero.
            (group/'memory.stat').unlink()
            self.assertEqual(snapshot(11, proc=root/'proc', group=group)['memory_bytes'], 51*1024)

    def test_unreadable_pss_keeps_the_subtree_and_counts_that_process_rss(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch('os.getpid', return_value=7):
            root = Path(folder)
            self.tree(root, 7, child_flags=PF_FORKNOEXEC)
            (root/'proc'/'11'/'smaps_rollup').unlink()
            row = snapshot(11, proc=root/'proc', shares=lambda *_: True)
            # 11 counts its RSS (10) instead of PSS (8); 12 and 13 are still walked.
            self.assertEqual((row['memory_bytes'], row['pss_errors']), (31*1024, 1))
            self.assertEqual(row['process_peak_rss_bytes'], 25*1024)
            self.assertEqual(row['tree_rss_bytes'], 51*1024)
            # A zombie (no VmRSS) is neither an error nor memory.
            (root/'proc'/'13'/'status').write_text('State:\tZ (zombie)\nVmHWM: 0 kB\n')
            (root/'proc'/'13'/'smaps_rollup').unlink()
            row = snapshot(11, proc=root/'proc', shares=lambda *_: False)
            self.assertEqual((row['memory_bytes'], row['pss_errors']), (31*1024, 1))

    def cgroup(self, root, stat, *, current=1000, swap=None):
        group = root/'cgroup'/'job'
        group.mkdir(parents=True, exist_ok=True)
        (group/'memory.current').write_text(f'{current}\n')
        (group/'memory.stat').write_text(''.join(f'{k} {v}\n' for k, v in stat.items()))
        if swap is not None:
            (group/'memory.swap.current').write_text(f'{swap}\n')
        return group

    def test_stale_memory_stat_allowance_only_raises_the_gate(self):
        stat = dict(file=400, shmem=50, file_mapped=40, file_dirty=20, file_writeback=10, active_file=150,
                    inactive_file=200)
        with tempfile.TemporaryDirectory() as folder:
            group = self.cgroup(Path(folder), stat)
            with mock.patch('scripts.run_component_memory_check.STAT_ERROR_BYTES', 100):
                self.assertEqual(cgroup_memory(group)['cgroup_memory_bytes'], 820)
            with mock.patch('scripts.run_component_memory_check.STAT_ERROR_BYTES', 10**6):
                self.assertEqual(cgroup_memory(group)['cgroup_memory_bytes'], 1000)
            # A stale swapcached is covered too, even when no clean cache is left to absorb it.
            none_clean = dict(stat, active_file=40, inactive_file=30, swapcached=150)
            self.cgroup(Path(folder), none_clean, swap=300)
            with mock.patch('scripts.run_component_memory_check.STAT_ERROR_BYTES', 100):
                self.assertEqual(cgroup_memory(group, swap_baseline=100)['cgroup_memory_bytes'], 1000 + 200 - 50)
        with mock.patch('platform.release', return_value='6.5.0-1025-azure'), \
                mock.patch('os.sysconf', side_effect=lambda name: {'SC_NPROCESSORS_CONF': 4, 'SC_PAGE_SIZE': 4096}[name]):
            from scripts.run_component_memory_check import _stat_error_bytes
            self.assertEqual(_stat_error_bytes(), 3 * 64 * 4 * 4096)
        with mock.patch('platform.release', return_value='6.11.0-1018-azure'), \
                mock.patch('os.sysconf', side_effect=lambda name: {'SC_NPROCESSORS_CONF': 4, 'SC_PAGE_SIZE': 4096}[name]):
            self.assertEqual(_stat_error_bytes(), 2 * 64 * 4 * 4096)
        for release, factor in (('6', 3), ('6.8-rc1', 3), ('6.14.0-1', 2), ('5.15.0-1', 3)):
            with mock.patch('platform.release', return_value=release):
                from scripts.run_component_memory_check import _stat_flush_factor
                self.assertEqual(_stat_flush_factor(), factor)

    def test_lazily_freed_anon_on_the_file_lru_is_not_dropped_as_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            group = self.cgroup(Path(folder), dict(file=100, shmem=0, file_mapped=0, file_dirty=0, file_writeback=0,
                                                   active_file=50, inactive_file=500))
            # 550 on the file LRU, but only 100 is page cache: MADV_FREE anon pages stay counted.
            self.assertEqual(cgroup_memory(group)['cgroup_memory_bytes'], 900)

    def test_swap_growth_since_the_start_counts_and_missing_stat_keys_refuse(self):
        stat = dict(file=400, shmem=50, file_mapped=40, file_dirty=20, file_writeback=10, active_file=150,
                    inactive_file=200)
        with tempfile.TemporaryDirectory() as folder:
            group = self.cgroup(Path(folder), stat, swap=300)
            row = cgroup_memory(group, swap_baseline=100)
            self.assertEqual((row['cgroup_memory_bytes'], row['cgroup_swap_growth_bytes']), (720 + 200, 200))
            self.assertEqual(cgroup_memory(group, swap_baseline=500)['cgroup_memory_bytes'], 720)
            # Pages read back from swap keep their slot and are already in memory.current.
            self.cgroup(Path(folder), dict(stat, swapcached=150), swap=300)
            self.assertEqual(cgroup_memory(group, swap_baseline=100)['cgroup_memory_bytes'], 720 + 50)
            for name in REQUIRED_STAT:
                self.cgroup(Path(folder), {k: v for k, v in stat.items() if k != name})
                with self.assertRaises(ValueError):
                    cgroup_memory(group)
        with tempfile.TemporaryDirectory() as folder, mock.patch('os.getpid', return_value=7):
            root = Path(folder)
            self.tree(root, 7)
            group = self.cgroup(root, {k: v for k, v in stat.items() if k != 'file_mapped'})
            self.assertEqual(snapshot(11, proc=root/'proc', group=group)['memory_bytes'], 51*1024)

    def test_limit_reads_the_gated_cgroup_and_its_ancestors(self):
        with tempfile.TemporaryDirectory() as folder:
            mount = Path(folder).resolve()/'cgroup'
            group = mount/'system.slice'/'runner'
            group.mkdir(parents=True)
            (mount/'memory.max').write_text('max\n')
            (mount/'system.slice'/'memory.max').write_text('1000\n')
            (group/'memory.max').write_text('max\n')
            self.assertEqual(effective_limit(group, cgroup=mount), 1000)
            self.assertGreater(effective_limit(None, cgroup=mount), 1000)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'real /proc, kcmp and cgroup v2 are Linux-only')
    def test_linux_kernel_interfaces(self):
        self.assertTrue(shares_address_space(os.getpid(), os.getpid()))
        tids, done = [], threading.Event()
        worker = threading.Thread(target=lambda: (tids.append(threading.get_native_id()), done.wait(5)))
        worker.start()
        try:
            while not tids:
                time.sleep(.01)
            self.assertTrue(shares_address_space(os.getpid(), tids[0]))
        finally:
            done.set(); worker.join()
        with subprocess.Popen(['sleep', '5']) as other:
            try:
                self.assertFalse(shares_address_space(os.getpid(), other.pid))
                row = snapshot(other.pid, pss=True)
                self.assertGreater(row['tree_pss_bytes'], 0)
                self.assertEqual(row['pss_errors'], 0)
            finally:
                other.kill()
        group = own_cgroup()
        if group is not None:
            row = cgroup_memory(group)
            self.assertTrue(set(REQUIRED_STAT) <= set(row['cgroup_stat']))
            self.assertGreater(row['cgroup_memory_bytes'], 0)

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
        row = dict(tree_rss_bytes=rss, process_peak_rss_bytes=memory if hwm is None else hwm, memory_bytes=memory)
        if group is not None:
            row.update(cgroup_current_bytes=memory + 9, cgroup_stat=dict(anon=memory - 3, kernel=3, file_dirty=0))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch('scripts.run_component_memory_check.snapshot', return_value=row), mock.patch(
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
        self.assertIsNone(report['peak_memory_stat'])
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
        self.assertEqual(report['cgroup_fallback_samples'], 0)
        self.assertIn(report['stat_flush_factor'], (2, 3))
        self.assertNotIn('kernel', report)
        stat = report['peak_memory_stat']
        self.assertEqual({k: stat[k] for k in ('memory_bytes', 'current_bytes', 'anon', 'kernel', 'file_dirty')},
                         dict(memory_bytes=70, current_bytes=79, anon=67, kernel=3, file_dirty=0))
        self.assertIn('memory.stat at the highest sampled value', summary)

    def test_cgroup_fallback_samples_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch('scripts.run_component_memory_check.snapshot', return_value=dict(
                    tree_rss_bytes=50, process_peak_rss_bytes=50, memory_bytes=50, cgroup_fallback=True)), \
                    mock.patch('scripts.run_component_memory_check.own_cgroup', return_value=Path('/unused')), \
                    mock.patch('scripts.run_component_memory_check.resource.getrusage',
                               return_value=mock.Mock(ru_maxrss=0)):
                run([sys.executable, '-c', 'pass'], output=root/'data', label='synthetic',
                    summary=root/'summary.md', limit_bytes=100, interval=.01)
            report = json.loads((root/'data/report.json').read_text())
            self.assertGreaterEqual(report['cgroup_fallback_samples'], 1)
            self.assertIn('gated on summed RSS', (root/'summary.md').read_text())

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
