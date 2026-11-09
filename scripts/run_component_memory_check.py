"""Observe an unchanged CPU check and enforce its declared memory headroom.

The optional output directory may be placed on durable storage by private
runner configuration. Samples are flushed throughout the run, including before
an OOM. No test selection, compiler environment or child timeout is changed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def effective_limit():
    limits = []
    for path in (Path('/sys/fs/cgroup/memory.max'),
                 Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')):
        if path.exists() and (value := path.read_text().strip()).isdigit():
            limits.append(int(value))
    physical = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    return min([physical] + limits)


def snapshot(pid, *, proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    pending, seen, rss, process_peak = [os.getpid(), pid], set(), 0, 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        directory = proc / str(current)
        try:
            fields = dict(line.split(':', 1) for line in
                          (directory / 'status').read_text().splitlines() if ':' in line)
            rss += int(fields.get('VmRSS', '0').split()[0]) * 1024
            process_peak = max(process_peak, int(fields.get('VmHWM', '0').split()[0]) * 1024)
            children = directory / 'task' / str(current) / 'children'
            pending.extend(int(value) for value in children.read_text().split())
        except (OSError, ValueError):
            # An interpreter can exit between /proc reads. Kernel child HWM is
            # also collected at normal termination to retain short-lived peaks.
            continue
    cgroup_peak = None
    try:
        cgroup_peak = int((cgroup / 'memory.peak').read_text())
    except (OSError, ValueError):
        pass
    return dict(tree_rss_bytes=rss, process_peak_rss_bytes=process_peak,
                cgroup_memory_peak_bytes=cgroup_peak)


def run(command, *, output, label, summary=None, limit_bytes=None, interval=.2):
    limit = effective_limit() if limit_bytes is None else limit_bytes
    if limit <= 0 or not 0 < interval <= 1:
        raise ValueError('positive runner limit and sampling interval <=1s required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    peak = cgroup_peak = 0
    started = time.monotonic()
    # Exclusive creation prevents merging two attempts' evidence.
    with (output / 'samples.jsonl').open('x') as samples:
        with subprocess.Popen(command) as child:
            while True:
                row = snapshot(child.pid)
                peak = max(peak, row['tree_rss_bytes'], row['process_peak_rss_bytes'])
                cgroup_peak = max(cgroup_peak, row['cgroup_memory_peak_bytes'] or 0)
                samples.write(json.dumps(dict(monotonic_ns=time.monotonic_ns(),
                    elapsed_seconds=time.monotonic()-started,
                    peak_rss_bytes=peak, runner_limit_bytes=limit, **row))+'\n')
                samples.flush()
                os.fsync(samples.fileno())
                try:
                    code = child.wait(timeout=interval)
                    break
                except subprocess.TimeoutExpired:
                    pass
        child_hwm = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        peak = max(peak, int(child_hwm) * (1 if sys.platform == 'darwin' else 1024))
        report = dict(label=label, child_exit_code=code, peak_rss_bytes=peak,
            peak_rss_method='max(sampled process-tree RSS, observed process HWM, kernel child HWM)',
            sample_interval_seconds=interval, cgroup_memory_peak_bytes=cgroup_peak,
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
            stream.write('| Peak RSS (GiB) | Runner limit (GiB) | 75% bar (GiB) | Wall (s) | Child exit | Memory |\n')
            stream.write('|---:|---:|---:|---:|---:|:---|\n')
            stream.write(f"| {peak/2**30:.3f} | {limit/2**30:.3f} | {report['bar_bytes']/2**30:.3f} | "
                         f"{report['wall_seconds']:.3f} | {code} | {'PASS' if report['memory_pass'] else 'FAIL'} |\n")
            stream.write(f'\nRSS uses {interval*1000:g} ms sampling plus kernel high-water marks; cgroup memory (including cache) is reported separately in the artifact.\n')
    if code:
        return code if code > 0 else 128-code
    return int(not report['memory_pass'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--summary')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('an unchanged child command is required')
    return run(command, output=args.output, label=args.label, summary=args.summary)


if __name__ == '__main__':
    raise SystemExit(main())
