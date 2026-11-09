"""Failing unittest IDs from GitHub Actions runs, and a PR-vs-baseline diff.

`extract` reads one or more runs (`gh run view RUN --log-failed`, or saved
logs) and writes the failing test IDs, one per line, plus a JSON sidecar with
each job's state: completed, or incomplete (killed with exit 143, runner
shutdown, cancelled or timed out). Runs are listed in precedence order: a test
takes its outcome from the first run that settled it, so later runs only fill
tests that a killed job of an earlier run never reached.

`diff` compares a PR's failing IDs (a run or a saved list) with a baseline
list: new failing IDs fail the gate; fixed IDs are reported.

IDs use unittest's dotted form, module.Class.test; class and module fixtures
become module.Class.setUpClass and module.setUpModule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

RESULT = re.compile(r'^(FAIL|ERROR|UNEXPECTED SUCCESS): (\S+) \(([^\s()]+)\)')
STARTED = re.compile(r'^(\w+) \(([\w.]+)\)(?: \.\.\.|$)')
TIMESTAMP = re.compile(r'^\d{4}-\d\d-\d\dT[\d:.]+Z ?')
KILLED = (re.compile(r'Process completed with exit code 143'), re.compile(r'runner has received a shutdown signal'),
          re.compile(r'The operation was canceled'), re.compile(r'has exceeded the maximum execution time'))


def test_id(name, target):
    """unittest's `name (target)` -> dotted ID; a `tests.` package prefix (root-level
    discovery) is dropped so every workflow names a test module the same way."""
    if target.startswith('tests.'):target = target[len('tests.'):]
    if target.endswith('.' + name):return target
    return f'{target}.{name}'


def split_line(line):
    """`gh run view --log` line -> (job, text); bare lines (saved plain logs) have job ''."""
    parts = line.rstrip('\n').split('\t', 2)
    if len(parts) == 3:job, _, text = parts
    else:job, text = '', parts[-1]
    return job, TIMESTAMP.sub('', text, count=1)


def _empty():
    return dict(failing=set(), started=set(), started_order=[], killed=False, exit_codes=set(), discovery_blocks=None,
                selected_blocks=[], blocks_finished=set(), current_block=None)


def parse_log(lines):
    """Per job: failing IDs, started IDs, whether the job was killed and, for the
    component shard runner, which discovery blocks were selected and finished."""
    jobs = {}
    for line in lines:
        job, text = split_line(line)
        record = jobs.setdefault(job, _empty())
        if text.startswith('{"shard":') or text.startswith('{"process_block":'):
            try:data = json.loads(text)
            except ValueError:data = None
            if isinstance(data, dict) and 'selected_blocks' in data:
                record['discovery_blocks'] = data.get('discovery_blocks');record['selected_blocks'] = list(data['selected_blocks'])
            elif isinstance(data, dict) and 'process_block' in data and 'declared_count' in data:
                record['current_block'] = data['process_block']
        if re.match(r'Ran \d+ tests? in ', text) and record['current_block'] is not None:
            record['blocks_finished'].add(record['current_block'])
        match = RESULT.match(text)
        if match:record['failing'].add(test_id(match.group(2), match.group(3)))
        match = STARTED.match(text)
        if match:
            test = test_id(match.group(1), match.group(2))
            if test not in record['started']:record['started'].add(test);record['started_order'].append(test)
        code = re.search(r'Process completed with exit code (\d+)', text)
        if code:record['exit_codes'].add(int(code.group(1)))
        if any(p.search(text) for p in KILLED):record['killed'] = True
    return jobs


def gh(*args, repo=None):
    command = ['gh', *args] + (['--repo', repo] if repo else [])
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def run_jobs(run, repo=None):
    data = json.loads(gh('run', 'view', str(run), '--json', 'headSha,headBranch,workflowName,conclusion,jobs', repo=repo))
    return data


def load_run(run, *, repo=None, log_dir=None):
    """(metadata, parsed jobs) for one run, caching the failed-job log under log_dir."""
    cached = Path(log_dir) / f'run-{run}.log' if log_dir else None
    if cached and cached.exists():text = cached.read_text()
    else:
        text = gh('run', 'view', str(run), '--log-failed', repo=repo)
        if cached:cached.parent.mkdir(parents=True, exist_ok=True);cached.write_text(text)
    meta = run_jobs(run, repo=repo)
    return meta, parse_log(text.splitlines())


def summarize(run, meta, jobs):
    states = {}
    for job in meta.get('jobs', []):
        name = job['name'];parsed = jobs.get(name, _empty())
        incomplete = parsed['killed'] or job.get('conclusion') in ('cancelled', 'timed_out') or 143 in parsed['exit_codes']
        unfinished = sorted(set(parsed['selected_blocks']) - parsed['blocks_finished'])
        states[name] = dict(conclusion=job.get('conclusion'), incomplete=incomplete, exit_codes=sorted(parsed['exit_codes']),
            failing=len(parsed['failing']), started=len(parsed['started']), discovery_blocks=parsed['discovery_blocks'],
            selected_blocks=len(parsed['selected_blocks']), unfinished_blocks=unfinished if incomplete else [])
    return dict(run=run, head_sha=meta.get('headSha'), branch=meta.get('headBranch'), workflow=meta.get('workflowName'),
        conclusion=meta.get('conclusion'), jobs=states,
        incomplete_jobs=sorted(n for n, s in states.items() if s['incomplete']))


def decided(parsed):
    """Tests whose outcome this job settled: every started test, except the last one
    a killed job started (it may not have finished)."""
    tests = set(parsed['started'])
    if parsed['killed'] and parsed['started_order']:tests.discard(parsed['started_order'][-1])
    return tests | parsed['failing']


def extract(args):
    """Precedence by run order: each test's outcome comes from the first listed run
    that settled it, so later runs only fill tests the earlier ones never reached."""
    runs, parsed_runs = [], []
    if args.log and len(args.log) != len(args.runs):raise SystemExit('one --log per run')
    for index, run in enumerate(args.runs):
        if args.log:
            jobs = parse_log(Path(args.log[index]).read_text().splitlines());meta = dict(jobs=[dict(name=n) for n in jobs])
        else:
            meta, jobs = load_run(run, repo=args.repo, log_dir=args.log_dir)
        runs.append(summarize(run, meta, jobs));parsed_runs.append(jobs)
    source, failing, unsettled = {}, set(), set()
    for summary, jobs in zip(runs, parsed_runs):
        for name, parsed in jobs.items():
            for test in sorted(decided(parsed)):
                if test in source:continue
                source[test] = summary['run']
                if test in parsed['failing']:failing.add(test)
            if parsed['killed'] and parsed['started_order']:unsettled.add(parsed['started_order'][-1])
    unsettled -= set(source)
    out = Path(args.out);out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(f'{x}\n' for x in sorted(failing)))
    covered = {}
    for summary in runs:
        for name, state in summary['jobs'].items():
            covered.setdefault(name, []).append(dict(run=summary['run'], complete=not state['incomplete']))
    never_complete = sorted(n for n, v in covered.items() if not any(x['complete'] for x in v) and n.startswith(args.shard_prefix))
    filled = {}
    for test, run in source.items():
        if run != runs[0]['run']:filled[run] = filled.get(run, 0) + 1
    sidecar = dict(schema='pokezero.ci_failing_ids.v1', runs=runs, failing=len(failing), settled_tests=len(source),
        settled_by_later_runs=filled, failing_from_later_runs=sorted(t for t in failing if source[t] != runs[0]['run']),
        unsettled_tests=sorted(unsettled), shards_never_complete=never_complete,
        note='Each test takes its outcome from the first listed run that settled it (started it and, in a killed job, '
             'was not the last test started). Tests in a killed job that no listed run reached are absent: their '
             'outcome is unknown. Discovery blocks differ between heads, so coverage is matched by test ID.')
    out.with_suffix('.json').write_text(json.dumps(sidecar, indent=1) + '\n')
    print(json.dumps(dict(failing=len(failing), settled=len(source), settled_by_later_runs=filled, unsettled=len(unsettled),
        runs=[dict(run=r['run'], head=r['head_sha'], incomplete=r['incomplete_jobs']) for r in runs],
        shards_never_complete=never_complete)))
    return 0


def read_ids(path):
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith('#')}


def diff(args):
    baseline = read_ids(args.baseline)
    if args.pr_ids:pr = read_ids(args.pr_ids);incomplete = []
    else:
        meta, jobs = load_run(args.pr_run, repo=args.repo, log_dir=args.log_dir)
        pr = set().union(*(j['failing'] for j in jobs.values())) if jobs else set()
        incomplete = summarize(args.pr_run, meta, jobs)['incomplete_jobs']
    new, fixed = sorted(pr - baseline), sorted(baseline - pr)
    report = dict(new_failing=new, fixed=fixed, pr_failing=len(pr), baseline_failing=len(baseline), pr_incomplete_jobs=incomplete,
        passed=not new)
    print(json.dumps(report, indent=1))
    if incomplete:print(f'warning: incomplete PR jobs {incomplete}; their missing failures are unknown', file=sys.stderr)
    return 0 if not new else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', default=None, help='OWNER/REPO (default: the current checkout)')
    parser.add_argument('--log-dir', default=None, help='cache directory for downloaded logs')
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract', help='failing IDs of one or more runs (union)')
    e.add_argument('runs', nargs='+')
    e.add_argument('--out', required=True, help='ID list; a .json sidecar is written beside it')
    e.add_argument('--log', action='append', default=None, help='parse saved logs instead of downloading (one per run, in order)')
    e.add_argument('--shard-prefix', default='CPU component shard')
    d = sub.add_parser('diff', help='new failing IDs of a PR against a baseline list')
    d.add_argument('--baseline', required=True)
    group = d.add_mutually_exclusive_group(required=True)
    group.add_argument('--pr-run')
    group.add_argument('--pr-ids')
    args = parser.parse_args(argv)
    return extract(args) if args.command == 'extract' else diff(args)


if __name__ == '__main__':
    sys.exit(main())
