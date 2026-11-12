"""Observe an unchanged CPU check and enforce its declared memory headroom.

The gate counts each resident page once, as the kernel does when it reclaims
or kills. When the observer's cgroup v2 exposes memory.current and memory.stat,
each sample is memory.current less the clean, unmapped file cache the kernel can
drop without I/O (see cgroup_memory); otherwise it is the sum of PSS over the
observed process tree,
where a child that still shares its parent's address space (vfork before exec)
is counted once. The observed process and kernel child high-water marks still
catch a single process's peak between samples.

The summed process-tree RSS of earlier runs is still recorded for continuity.
It no longer gates: a forked or vforked child reports its parent's shared pages
as its own resident set until it execs, so one sample can read twice the memory
in use.

The optional output directory may be placed on durable storage by private
runner configuration. Samples are flushed throughout the run, including before
an OOM. No test selection, compiler environment or child timeout is changed.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

PF_FORKNOEXEC = 0x40  # include/linux/sched.h: forked but has not exec'd
KCMP_VM = 1           # include/uapi/linux/kcmp.h
SYS_KCMP = {'x86_64': 312, 'aarch64': 272}
CGROUP_SOURCE = 'cgroup v2 memory.current less clean unmapped file cache'
PSS_SOURCE = 'process-tree PSS'


def effective_limit():
    limits = []
    for path in (Path('/sys/fs/cgroup/memory.max'),
                 Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')):
        if path.exists() and (value := path.read_text().strip()).isdigit():
            limits.append(int(value))
    physical = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    return min([physical] + limits)


def own_cgroup(*, proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    """The observer's cgroup v2 directory, if it exposes the counters the gate reads."""
    try:
        entries = (proc / 'self' / 'cgroup').read_text().splitlines()
        relative = next(line[3:] for line in entries if line.startswith('0::'))
        group = (cgroup / relative.lstrip('/')).resolve()
    except (OSError, StopIteration):
        return None
    if not group.is_relative_to(cgroup.resolve()):
        return None
    if all((group / name).is_file() for name in ('memory.current', 'memory.stat')):
        return group
    return None


STAT_FIELDS = ('anon', 'file', 'shmem', 'kernel', 'file_mapped', 'file_dirty', 'file_writeback',
               'active_file', 'inactive_file')


def cgroup_memory(group):
    """memory.current without the clean, unmapped page cache the kernel drops before an OOM kill.

    The file LRU (active_file + inactive_file) is subtracted, except:
    - dirty and writeback file pages, which cannot be dropped without I/O;
    - mapped file pages (shared libraries, mmapped data), the running processes'
      working set, which every process's RSS also counts; here each counts once.
    A page that is both dirty and mapped is in file_dirty and in file_mapped, so
    the reclaimable term can go below zero; it is clamped at 0, which keeps the
    gate at or below memory.current. shmem and tmpfs pages are swap-backed and
    sit on the anon LRU, so they are not in active_file/inactive_file and stay
    counted (memory.stat's 'file' total does include them). Anon and kernel
    memory are counted as charged.
    """
    current = int((group / 'memory.current').read_text())
    stat = {name: int(value) for name, value in (line.split() for line in
            (group / 'memory.stat').read_text().splitlines() if len(line.split()) == 2)}
    clean = stat.get('active_file', 0) + stat.get('inactive_file', 0) - stat.get('file_dirty', 0) \
        - stat.get('file_writeback', 0) - stat.get('file_mapped', 0)
    return dict(cgroup_memory_bytes=max(current - max(clean, 0), 0), cgroup_current_bytes=current,
                cgroup_clean_cache_bytes=max(clean, 0),
                cgroup_stat={name: stat[name] for name in STAT_FIELDS if name in stat})


_libc = None


def shares_address_space(first, second):
    """True only when kcmp says the two processes share one mm; any error counts both."""
    global _libc
    number = SYS_KCMP.get(platform.machine())
    if number is None or not sys.platform.startswith('linux'):
        return False
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
    return _libc.syscall(number, first, second, KCMP_VM, 0, 0) == 0


def snapshot(pid, *, proc=Path('/proc'), group=None, pss=None,
             shares=shares_address_space):
    """One sample of the observer, the child and every descendant.

    group: the cgroup that gates (None for the PSS fallback). pss: also sum PSS
    when the cgroup gates (diagnostic only; it walks every page table).
    """
    pss = group is None if pss is None else pss
    pending, seen = [(os.getpid(), None), (pid, os.getpid())], set()
    rss = process_peak = pss_total = anon = file = shmem = 0
    while pending:
        current, parent = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        directory = proc / str(current)
        try:
            fields = dict(line.split(':', 1) for line in
                          (directory / 'status').read_text().splitlines() if ':' in line)
            rss += int(fields.get('VmRSS', '0').split()[0]) * 1024
            anon += int(fields.get('RssAnon', '0').split()[0]) * 1024
            file += int(fields.get('RssFile', '0').split()[0]) * 1024
            shmem += int(fields.get('RssShmem', '0').split()[0]) * 1024
            process_peak = max(process_peak, int(fields.get('VmHWM', '0').split()[0]) * 1024)
            if pss:
                stat = (directory / 'stat').read_text()
                flags = int(stat[stat.rindex(')') + 2:].split()[6])
                if not (parent is not None and flags & PF_FORKNOEXEC and shares(parent, current)):
                    rollup = dict(line.split(':', 1) for line in
                                  (directory / 'smaps_rollup').read_text().splitlines() if ':' in line)
                    pss_total += int(rollup.get('Pss', '0').split()[0]) * 1024
            # Any thread can start a child; every task's children are descendants.
            for task in (directory / 'task').iterdir():
                try:
                    pending.extend((int(value), current) for value in (task / 'children').read_text().split())
                except OSError:
                    continue
        except (OSError, ValueError, IndexError):
            # An interpreter can exit between /proc reads. Kernel child HWM is
            # also collected at normal termination to retain short-lived peaks.
            continue
    # The RSS split attributes any difference from the cgroup value (diagnostic).
    row = dict(tree_rss_bytes=rss, process_peak_rss_bytes=process_peak, tree_rss_anon_bytes=anon,
               tree_rss_file_bytes=file, tree_rss_shmem_bytes=shmem)
    if pss:
        row['tree_pss_bytes'] = pss_total
    if group is None:
        row['memory_bytes'] = pss_total
    else:
        try:
            row.update(cgroup_memory(group))
            row['memory_bytes'] = row['cgroup_memory_bytes']
        except (OSError, ValueError):
            # A missing counter never lowers the gate: use the larger RSS sum.
            row['memory_bytes'] = rss
    return row


def cgroup_peak(group):
    try:
        return int((group / 'memory.peak').read_text())
    except (OSError, ValueError, TypeError):
        return None


def run(command, *, output, label, summary=None, limit_bytes=None, interval=.2, record_pss=False):
    limit = effective_limit() if limit_bytes is None else limit_bytes
    if limit <= 0 or not 0 < interval <= 1:
        raise ValueError('positive runner limit and sampling interval <=1s required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    group = own_cgroup()
    source = PSS_SOURCE if group is None else CGROUP_SOURCE
    peak = legacy = sampled = 0
    breakdown = None
    started = time.monotonic()
    # Exclusive creation prevents merging two attempts' evidence.
    with (output / 'samples.jsonl').open('x') as samples:
        with subprocess.Popen(command) as child:
            while True:
                row = snapshot(child.pid, group=group, pss=record_pss or group is None)
                peak = max(peak, row['memory_bytes'], row['process_peak_rss_bytes'])
                legacy = max(legacy, row['tree_rss_bytes'], row['process_peak_rss_bytes'])
                if row['memory_bytes'] > sampled:
                    sampled = row['memory_bytes']
                    # memory.stat at the highest sampled gate value (cgroup source only).
                    breakdown = None if 'cgroup_stat' not in row else dict(
                        elapsed_seconds=round(time.monotonic()-started, 3), memory_bytes=row['memory_bytes'],
                        current_bytes=row['cgroup_current_bytes'], **row['cgroup_stat'])
                samples.write(json.dumps(dict(monotonic_ns=time.monotonic_ns(),
                    elapsed_seconds=time.monotonic()-started, peak_bytes=peak,
                    peak_rss_bytes=legacy, runner_limit_bytes=limit, **row))+'\n')
                samples.flush()
                os.fsync(samples.fileno())
                try:
                    code = child.wait(timeout=interval)
                    break
                except subprocess.TimeoutExpired:
                    pass
        child_hwm = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * (
            1 if sys.platform == 'darwin' else 1024)
        peak, legacy = max(peak, child_hwm), max(legacy, child_hwm)
        report = dict(label=label, child_exit_code=code, peak_bytes=peak,
            peak_method=f'max(sampled {source}, observed process HWM, kernel child HWM)',
            memory_source=source, peak_rss_bytes=legacy,
            peak_rss_method='not gating: max(sampled process-tree RSS sum, observed process HWM, kernel child HWM)',
            peak_memory_stat=breakdown,
            sample_interval_seconds=interval, cgroup_memory_peak_bytes=cgroup_peak(group),
            runner_limit_bytes=limit, bar_bytes=limit * 3 // 4,
            memory_pass=0 < peak <= limit * 3 // 4,
            wall_seconds=time.monotonic()-started)
        samples.write(json.dumps(dict(terminal=True, **report))+'\n')
        samples.flush()
        os.fsync(samples.fileno())
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('CPU_MEMORY_REPORT '+json.dumps(report), flush=True)
    if summary is not None:
        with Path(summary).open('a') as stream:
            stream.write('\n### CPU memory: '+label+'\n\n')
            stream.write('| Peak (GiB) | Summed RSS, not gating (GiB) | Runner limit (GiB) | 75% bar (GiB) | Wall (s) | Child exit | Memory |\n')
            stream.write('|---:|---:|---:|---:|---:|---:|:---|\n')
            stream.write(f"| {peak/2**30:.3f} | {legacy/2**30:.3f} | {limit/2**30:.3f} | {report['bar_bytes']/2**30:.3f} | "
                         f"{report['wall_seconds']:.3f} | {code} | {'PASS' if report['memory_pass'] else 'FAIL'} |\n")
            stream.write(f'\nPeak: {source}, sampled every {interval*1000:g} ms, plus kernel high-water marks. '
                         'The summed process-tree RSS counts a forked child\'s shared pages twice until it execs.\n')
            if breakdown:
                stream.write('\nmemory.stat at the highest sampled value (GiB): ' + ', '.join(
                    f'{name} {breakdown[name]/2**30:.3f}' for name in ('memory_bytes', 'current_bytes') + STAT_FIELDS
                    if name in breakdown) + '\n')
    if code:
        return code if code > 0 else 128-code
    return int(not report['memory_pass'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--summary')
    parser.add_argument('--interval', type=float, default=.2, help='sampling interval in seconds (0, 1]')
    parser.add_argument('--record-pss', action='store_true',
                        help='also record process-tree PSS when the cgroup gates (diagnostic)')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('an unchanged child command is required')
    return run(command, output=args.output, label=args.label, summary=args.summary,
               interval=args.interval, record_pss=args.record_pss)


if __name__ == '__main__':
    raise SystemExit(main())
