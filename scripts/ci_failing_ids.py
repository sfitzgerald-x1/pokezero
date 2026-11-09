"""Failing unittest IDs: CI runs plus a pod-side supplement, and a PR-vs-baseline diff.

`extract` reads GitHub Actions runs (`gh run view RUN --log-failed`, or cached
logs) and settles each test: a test is settled only when the unittest run that
started it reached `Ran N tests` (its FAIL:/ERROR: headers print just before
that line), or when its own `... ok|FAIL|ERROR` result is printed inline. A test
a killed job (exit 143, runner shutdown, cancel, timeout) left running or never
reached stays unsettled. Runs must share one head SHA unless --allow-cross-head
is given; every failing ID records its provenance (run, head, job). With
--tree/--tree-ref the expected test modules are listed from the tree, so
coverage is reported by module and test ID: modules never reached, modules
with unsettled tests. Pod supplement results (--pod) fill only tests the CI
runs did not settle, with provenance `pod`.

`pod-run` is that supplement: it runs a list of test modules on a given tree
through the CI entry point (scripts/run_gpu_component_shard.run_process_blocks:
one fresh, rediscovered process per block, the CI case-isolation classes) with
the CI environment variables, and writes the settled results in the same form.
The pod is not the CI runner (CPU, platform, timing), and the result records it.

`diff` compares a PR (runs and/or pod results) with a baseline: new failing IDs
fail the gate (exit 1); a PR whose jobs or expected modules are not all settled
is INCOMPLETE (exit 2), never a pass. Exit 0 only for a complete PR with no new
failing ID.

IDs use unittest's dotted form, module.Class.test; class and module fixtures
become module.Class.setUpClass and module.setUpModule; a FunctionTestCase is
module.function, the module found from the tree when one is given.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

SCHEMA = 'pokezero.ci_failing_ids.v2'
RESULT = re.compile(r'^(FAIL|ERROR|UNEXPECTED SUCCESS): (\S+) \(([^\s()]+)\)')
STARTED = re.compile(r'^(\w+|unittest\.case\.FunctionTestCase) \(([\w.]+)\)(?: \.\.\.(?: (.*))?)?$')
RAN = re.compile(r'^Ran \d+ tests? in ')
TIMESTAMP = re.compile(r'^\d{4}-\d\d-\d\dT[\d:.]+Z ?')
KILLED = (re.compile(r'Process completed with exit code 143'), re.compile(r'runner has received a shutdown signal'),
          re.compile(r'The operation was canceled'), re.compile(r'has exceeded the maximum execution time'),
          re.compile(r'POD_RUN_TIMEOUT'))
FAILED_WORDS = ('FAIL', 'ERROR', 'unexpected success')
PASSED_WORDS = ('ok', 'skipped', 'expected failure')
DEFAULT_PATTERNS = ('test_gpu_engine*.py',)
# The component workflow's isolated classes (.github/workflows/gpu-engine-components.yml).
CI_ISOLATE = ('test_gpu_engine_execution_runtime_plumbing.ExecutionRuntimePlumbingTests',
              'test_gpu_engine_status_kernel_owner_drain.StatusKernelOwnerDrainTests')


def strip_package(target):
    return target[len('tests.'):] if target.startswith('tests.') else target


def test_id(name, target, function_modules=None):
    """unittest's `name (target)` -> dotted ID; a `tests.` package prefix (root-level
    discovery) is dropped so every workflow names a test module the same way."""
    if name == 'unittest.case.FunctionTestCase':
        module = (function_modules or {}).get(target)
        return f'{module}.{target}' if module else f'FunctionTestCase.{target}'
    target = strip_package(target)
    if target.endswith('.' + name):return target
    return f'{target}.{name}'


def module_of(test):
    return test.split('.', 1)[0]


def split_line(line, plain=False):
    """`gh run view --log` line -> (job, text); plain logs (pod runs) have job ''."""
    if plain:return '', TIMESTAMP.sub('', line.rstrip('\n'), count=1)
    parts = line.rstrip('\n').split('\t', 2)
    if len(parts) == 3:job, _, text = parts
    else:job, text = '', parts[-1]
    return job, TIMESTAMP.sub('', text, count=1)


def _empty():
    return dict(settled={}, unsettled=set(), started=set(), segment=[], segment_failing=set(), inline={},
                killed=False, exit_codes=set(), discovery_blocks=None, selected_blocks=[], blocks_finished=set(),
                current_block=None, checkout=None)


def _close_segment(record, finished):
    """Settle the tests of one unittest run: all of them when it printed `Ran`, else
    only those whose own result was printed inline."""
    for test in record['segment']:
        word = record['inline'].get(test)
        if finished:
            failed = test in record['segment_failing'] or (word is not None and word.startswith(FAILED_WORDS))
            record['settled'][test] = 'fail' if failed else 'pass'
        elif word is not None and word.startswith(FAILED_WORDS):record['settled'][test] = 'fail'
        elif word is not None and word.startswith(PASSED_WORDS):record['settled'][test] = 'pass'
        else:record['unsettled'].add(test)
    if finished:
        for test in record['segment_failing'] - set(record['segment']):record['settled'][test] = 'fail'
    record['segment'] = [];record['segment_failing'] = set()


def parse_log(lines, function_modules=None, *, plain=False):
    """Per job: settled tests (pass/fail), unsettled started tests, kill state, and the
    component runner's selected/finished discovery blocks."""
    jobs = {}
    for line in lines:
        job, text = split_line(line, plain)
        record = jobs.setdefault(job, _empty())
        if text.startswith('{"shard":') or text.startswith('{"process_block":'):
            try:data = json.loads(text)
            except ValueError:data = None
            if isinstance(data, dict) and 'selected_blocks' in data:
                record['discovery_blocks'] = data.get('discovery_blocks');record['selected_blocks'] = list(data['selected_blocks'])
            elif isinstance(data, dict) and 'process_block' in data and 'declared_count' in data:
                _close_segment(record, finished=False);record['current_block'] = data['process_block']
            continue
        if RAN.match(text):
            _close_segment(record, finished=True)
            if record['current_block'] is not None:record['blocks_finished'].add(record['current_block'])
            continue
        match = RESULT.match(text)
        if match:
            record['segment_failing'].add(test_id(match.group(2), match.group(3), function_modules));continue
        match = STARTED.match(text)
        if match:
            test = test_id(match.group(1), match.group(2), function_modules)
            if test not in record['started']:record['started'].add(test);record['segment'].append(test)
            if match.group(3):record['inline'][test] = match.group(3).strip()
            continue
        if record['checkout'] is None and text.startswith('HEAD is now at '):record['checkout'] = text[len('HEAD is now at '):][:160]
        code = re.search(r'Process completed with exit code (\d+)', text)
        if code:record['exit_codes'].add(int(code.group(1)))
        if any(p.search(text) for p in KILLED):record['killed'] = True
    for record in jobs.values():
        _close_segment(record, finished=False)
        record['unsettled'] -= set(record['settled'])
    return jobs


# ---------------------------------------------------------------------------
# Trees: expected modules and FunctionTestCase owners.

class Tree:
    """A test tree from a directory or a git ref of the current checkout."""
    def __init__(self, path=None, ref=None, patterns=DEFAULT_PATTERNS):
        self.path, self.ref, self.patterns = path, ref, tuple(patterns)
        if path:
            self.files = sorted(p.name for p in (Path(path) / 'tests').glob('*.py'))
            self.commit = _git(['-C', str(path), 'rev-parse', 'HEAD']).strip()
        else:
            names = _git(['ls-tree', '--name-only', f'{ref}:tests']).split()
            self.files = sorted(n for n in names if n.endswith('.py'));self.commit = _git(['rev-parse', ref]).strip()

    def read(self, name):
        if self.path:return (Path(self.path) / 'tests' / name).read_text()
        return _git(['show', f'{self.ref}:tests/{name}'])

    def modules(self):
        return sorted(f[:-3] for f in self.files if any(fnmatch.fnmatch(f, p) for p in self.patterns))

    def function_modules(self):
        """function name -> module, for modules that build FunctionTestCase suites."""
        owners = {}
        for name in self.files:
            if not name.startswith('test_'):continue
            text = self.read(name)
            if 'FunctionTestCase' not in text:continue
            for function in re.findall(r'^def (test_\w+)\(', text, re.M):owners.setdefault(function, name[:-3])
        return owners


def _git(args):
    return subprocess.run(['git', *args], check=True, capture_output=True, text=True).stdout


def tree_from(args):
    if getattr(args, 'tree', None):return Tree(path=args.tree, patterns=args.pattern or DEFAULT_PATTERNS)
    if getattr(args, 'tree_ref', None):return Tree(ref=args.tree_ref, patterns=args.pattern or DEFAULT_PATTERNS)
    return None


# ---------------------------------------------------------------------------
# Runs.

def gh(*args, repo=None):
    command = ['gh', *args] + (['--repo', repo] if repo else [])
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def load_run(run, *, repo=None, log_dir=None, function_modules=None):
    """(metadata, parsed jobs); the failed-job log and the job metadata are cached under log_dir."""
    cached = Path(log_dir) / f'run-{run}.log' if log_dir else None
    meta_path = Path(log_dir) / f'run-{run}.json' if log_dir else None
    if cached and cached.exists():text = cached.read_text()
    else:
        text = gh('run', 'view', str(run), '--log-failed', repo=repo)
        if cached:cached.parent.mkdir(parents=True, exist_ok=True);cached.write_text(text)
    if meta_path and meta_path.exists():meta = json.loads(meta_path.read_text())
    else:
        meta = json.loads(gh('run', 'view', str(run), '--json', 'headSha,headBranch,workflowName,conclusion,jobs', repo=repo))
        if meta_path:meta_path.write_text(json.dumps(meta))
    return meta, parse_log(text.splitlines(), function_modules)


def summarize(run, meta, jobs):
    states = {}
    names = [job['name'] for job in meta.get('jobs', [])] or list(jobs)
    conclusions = {job['name']: job.get('conclusion') for job in meta.get('jobs', [])}
    for name in names:
        parsed = jobs.get(name, _empty())
        incomplete = (parsed['killed'] or conclusions.get(name) in ('cancelled', 'timed_out') or 143 in parsed['exit_codes']
                      or bool(parsed['unsettled']))
        states[name] = dict(conclusion=conclusions.get(name), incomplete=incomplete, exit_codes=sorted(parsed['exit_codes']),
            settled=len(parsed['settled']), failing=sum(v == 'fail' for v in parsed['settled'].values()),
            unsettled=sorted(parsed['unsettled']), discovery_blocks=parsed['discovery_blocks'],
            unfinished_blocks=sorted(set(parsed['selected_blocks']) - parsed['blocks_finished']))
    # pull_request runs test the PR's merge ref (head merged into the base tip), not the head itself.
    checkouts = sorted({j['checkout'] for j in jobs.values() if j.get('checkout')})
    return dict(run=str(run), head_sha=meta.get('headSha'), branch=meta.get('headBranch'), workflow=meta.get('workflowName'),
        conclusion=meta.get('conclusion'), tested_checkout=checkouts, jobs=states,
        incomplete_jobs=sorted(n for n, s in states.items() if s['incomplete']))


def collect(runs, *, args, function_modules):
    """Load runs; enforce one head unless allowed. Returns (summaries, parsed per run)."""
    summaries, parsed = [], []
    logs = getattr(args, 'log', None)
    if logs and len(logs) != len(runs):raise SystemExit('one --log per run')
    for index, run in enumerate(runs):
        if logs:
            jobs = parse_log(Path(logs[index]).read_text().splitlines(), function_modules)
            meta = dict(jobs=[dict(name=n) for n in jobs], headSha=(args.log_head or [None] * len(runs))[index])
        else:
            meta, jobs = load_run(run, repo=args.repo, log_dir=args.log_dir, function_modules=function_modules)
        summaries.append(summarize(run, meta, jobs));parsed.append(jobs)
    heads = sorted({s['head_sha'] for s in summaries if s['head_sha']})
    if len(heads) > 1 and not getattr(args, 'allow_cross_head', False):
        raise SystemExit(f'runs span head SHAs {heads}; pass --allow-cross-head to fill across heads (recorded per ID)')
    return summaries, parsed


def settle(summaries, parsed, pods=()):
    """test -> (outcome, provenance): first CI run (in order) that settled it, then pod results."""
    outcome = {}
    for summary, jobs in zip(summaries, parsed):
        for name, record in jobs.items():
            for test, value in sorted(record['settled'].items()):
                if test not in outcome:
                    outcome[test] = (value, dict(source='ci', run=summary['run'], head=summary['head_sha'], job=name))
    for pod in pods:
        for test, value in sorted(pod['settled'].items()):
            if test not in outcome:
                outcome[test] = (value, dict(source='pod', tree=pod['tree_commit'], module=module_of(test)))
    unsettled = set()
    for jobs in parsed:
        for record in jobs.values():unsettled |= record['unsettled']
    for pod in pods:unsettled |= set(pod['unsettled'])
    return outcome, unsettled - set(outcome)


def coverage(tree, outcome, unsettled):
    if tree is None:return None
    expected = tree.modules();reached = {}
    for test in outcome:reached.setdefault(module_of(test), 0);reached[module_of(test)] += 1
    partial = sorted({module_of(t) for t in unsettled} & set(expected))
    return dict(tree_commit=tree.commit, patterns=list(tree.patterns), expected_modules=len(expected),
        modules_never_reached=sorted(m for m in expected if m not in reached), modules_with_unsettled_tests=partial,
        unsettled_tests=sorted(t for t in unsettled if module_of(t) in set(expected)))


def read_pod(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema') != SCHEMA + '.pod':raise SystemExit(f'{path}: not a pod-run result')
    return data


def extract(args):
    tree = tree_from(args);function_modules = tree.function_modules() if tree else None
    summaries, parsed = collect(args.runs, args=args, function_modules=function_modules)
    pods = [read_pod(p) for p in args.pod or ()]
    outcome, unsettled = settle(summaries, parsed, pods)
    failing = sorted(t for t, (v, _) in outcome.items() if v == 'fail')
    cover = coverage(tree, outcome, unsettled)
    out = Path(args.out);out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(f'{x}\n' for x in failing))
    by_source = {}
    for test in failing:
        src = outcome[test][1];key = src['source'] + ':' + (src.get('head') or src.get('tree') or '')[:8]
        by_source[key] = by_source.get(key, 0) + 1
    sidecar = dict(schema=SCHEMA, runs=summaries, pods=[{k: v for k, v in p.items() if k not in ('settled',)} for p in pods],
        failing=len(failing), settled_tests=len(outcome), failing_by_source=by_source,
        provenance={t: outcome[t][1] for t in failing}, settled_outcomes={t: v for t, (v, _) in sorted(outcome.items())},
        unsettled_tests=sorted(unsettled), coverage=cover, cross_head=bool(getattr(args, 'allow_cross_head', False)),
        note='A test is settled when its unittest run printed `Ran N tests` or its own result was printed inline; '
             'CI runs take precedence in the listed order and pod results only fill tests CI did not settle.')
    out.with_suffix('.json').write_text(json.dumps(sidecar, indent=1) + '\n')
    print(json.dumps(dict(failing=len(failing), settled=len(outcome), failing_by_source=by_source, unsettled=len(unsettled),
        modules_never_reached=None if cover is None else len(cover['modules_never_reached']),
        modules_with_unsettled_tests=None if cover is None else len(cover['modules_with_unsettled_tests']),
        runs=[dict(run=r['run'], head=r['head_sha'], incomplete=r['incomplete_jobs']) for r in summaries])))
    return 0


def supplement_modules(args):
    """Modules a pod supplement must run: never reached or with unsettled tests (from a sidecar), plus --force."""
    sidecar = json.loads(Path(args.sidecar).read_text());cover = sidecar.get('coverage')
    if not cover:raise SystemExit('the sidecar has no coverage; extract with --tree or --tree-ref')
    modules = sorted(set(cover['modules_never_reached']) | set(cover['modules_with_unsettled_tests']) | set(args.force or ()))
    Path(args.out).write_text(''.join(f'{m}\n' for m in modules))
    print(json.dumps(dict(modules=len(modules), never_reached=len(cover['modules_never_reached']),
        with_unsettled=len(cover['modules_with_unsettled_tests']), forced=sorted(args.force or ()))))
    return 0


# ---------------------------------------------------------------------------
# Pod supplement.

DRIVER = r'''
import importlib.util, json, sys, unittest
from pathlib import Path
tree, module, isolate = Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
spec = importlib.util.spec_from_file_location('run_gpu_component_shard', tree / 'scripts/run_gpu_component_shard.py')
runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
sys.path.insert(0, str(tree))
pattern = module + '.py'
discovered = unittest.defaultTestLoader.discover(start_dir=str(tree / 'tests'), pattern=pattern)
blocks = len(list(discovered))
print(json.dumps({"shard": 0, "count": 1, "discovery_blocks": blocks, "selected_blocks": list(range(blocks)),
    "test_cases": discovered.countTestCases(), "pod_module": module}), flush=True)
sys.exit(runner.run_process_blocks(discovered, 0, 1, start_dir=tree / 'tests', pattern=pattern,
    isolate_standard_classes=[c for c in isolate if c.split('.')[0] == module]))
'''


def pod_environment(args, tree):
    env = dict(os.environ)
    env.update(JAX_PLATFORMS='cpu', PYTHONPATH=f'{tree}/src:{tree}', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        XLA_FLAGS='--xla_cpu_parallel_codegen_split_count=1 --xla_backend_optimization_level=3',
        POKEZERO_GPU_TEAM_POOL=str(args.team_pool), POKEZERO_SHOWDOWN_ROOT=str(args.showdown_root), PYTHONUNBUFFERED='1')
    return env


def pod_run(args):
    tree = Path(args.tree).resolve();out = Path(args.out);logs = Path(args.log_dir or out.with_suffix('.logs'))
    if out.exists():raise SystemExit(f'{out} exists')
    logs.mkdir(parents=True, exist_ok=False)
    modules = list(args.modules or [])
    if args.modules_file:modules += [m.strip() for m in Path(args.modules_file).read_text().split() if m.strip()]
    modules = sorted(set(modules))
    if not modules:raise SystemExit('no modules')
    missing = [m for m in modules if not (tree / 'tests' / f'{m}.py').exists()]
    if missing:raise SystemExit(f'modules not in the tree: {missing}')
    commit = _git(['-C', str(tree), 'rev-parse', 'HEAD']).strip()
    dirty = bool(_git(['-C', str(tree), 'status', '--porcelain', '--', 'gpu_engine', 'tests', 'src', 'scripts']).strip())
    if dirty and not args.allow_dirty:raise SystemExit('the tree is dirty; commit it or pass --allow-dirty')
    function_modules = Tree(path=tree).function_modules();env = pod_environment(args, tree)

    def one(module):
        begin = time.perf_counter();path = logs / f'{module}.log'
        with path.open('w') as stream:
            process = subprocess.Popen([sys.executable, '-u', '-c', DRIVER, str(tree), module, json.dumps(CI_ISOLATE)],
                cwd=tree, env=env, stdout=stream, stderr=subprocess.STDOUT)
            try:code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                process.kill();process.wait();code = None
                stream.write('POD_RUN_TIMEOUT\n')
        parsed = parse_log(path.read_text().splitlines(), function_modules, plain=True).get('', _empty())
        return module, dict(returncode=code, timed_out=code is None, seconds=round(time.perf_counter() - begin, 1),
            settled=parsed['settled'], unsettled=sorted(parsed['unsettled']), blocks=len(parsed['selected_blocks']),
            blocks_finished=len(parsed['blocks_finished']))

    results = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for module, result in pool.map(one, modules):
            results[module] = result
            print(json.dumps(dict(module=module, returncode=result['returncode'], seconds=result['seconds'],
                settled=len(result['settled']), failing=sum(v == 'fail' for v in result['settled'].values()),
                unsettled=len(result['unsettled']))), flush=True)
    settled = {t: v for r in results.values() for t, v in r['settled'].items()}
    unsettled = sorted({t for r in results.values() for t in r['unsettled']} - set(settled))
    versions = subprocess.run([sys.executable, '-c', 'import jax,numpy;print(jax.__version__,numpy.__version__)'],
        capture_output=True, text=True, env=env).stdout.split()
    node = subprocess.run(['node', '--version'], capture_output=True, text=True).stdout.strip()
    data = dict(schema=SCHEMA + '.pod', tree=str(tree), tree_commit=commit, tree_dirty=dirty, modules=modules,
        entry_point='scripts/run_gpu_component_shard.run_process_blocks (fresh process per block, CI isolation classes)',
        environment=dict(python=platform.python_version(), machine=platform.machine(), jax_numpy=versions, node=node,
            team_pool=str(args.team_pool), showdown_root=str(args.showdown_root), jobs=args.jobs, timeout=args.timeout,
            note='pod, not the CI runner: CPU model, core count and timing differ'),
        results={m: {k: v for k, v in r.items() if k != 'settled'} for m, r in results.items()},
        settled=settled, unsettled=unsettled, failing=sorted(t for t, v in settled.items() if v == 'fail'))
    out.write_text(json.dumps(data, indent=1) + '\n')
    out.with_suffix('.txt').write_text(''.join(f'{t}\n' for t in data['failing']))
    print(json.dumps(dict(modules=len(modules), settled=len(settled), failing=len(data['failing']), unsettled=len(unsettled),
        timed_out=sorted(m for m, r in results.items() if r['timed_out']))))
    return 0


# ---------------------------------------------------------------------------
# Diff.

def read_ids(path):
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith('#')}


def diff(args):
    baseline = read_ids(args.baseline)
    sidecar_path = Path(args.baseline).with_suffix('.json')
    base_settled = json.loads(sidecar_path.read_text()).get('settled_outcomes') if sidecar_path.exists() else None
    tree = tree_from(args);function_modules = tree.function_modules() if tree else None
    summaries, parsed = collect(args.pr_run or [], args=args, function_modules=function_modules) if args.pr_run else ([], [])
    pods = [read_pod(p) for p in args.pr_pod or ()]
    outcome, unsettled = settle(summaries, parsed, pods)
    for test in sorted(read_ids(args.pr_ids) if args.pr_ids else ()):
        outcome.setdefault(test, ('fail', dict(source='ids')))
    pr_failing = {t for t, (v, _) in outcome.items() if v == 'fail'}
    new = sorted(t for t in pr_failing - baseline if base_settled is None or t in base_settled)
    unknown_base = sorted(t for t in pr_failing - baseline if base_settled is not None and t not in base_settled)
    fixed = sorted(t for t in baseline - pr_failing if t in outcome)
    cover = coverage(tree, outcome, unsettled)
    incomplete_jobs = [f"{s['run']}:{name}" for s in summaries for name in s['incomplete_jobs']]
    unsettled_modules = sorted({module_of(t) for t in unsettled}
                               | set((cover or {}).get('modules_never_reached', ())))
    complete = not unsettled_modules and not unknown_base and (args.pr_ids is not None or bool(summaries or pods))
    status = 'NEW_FAILURES' if new else ('PASSED' if complete else 'INCOMPLETE')
    report = dict(status=status, new_failing=new, new_failing_with_unsettled_baseline=unknown_base, fixed=fixed,
        pr_failing=len(pr_failing), baseline_failing=len(baseline), pr_incomplete_jobs=incomplete_jobs,
        unsettled_modules=unsettled_modules, unsettled_tests=sorted(unsettled),
        provenance={t: outcome[t][1] for t in sorted(pr_failing - baseline)}, passed=status == 'PASSED')
    print(json.dumps(report, indent=1))
    return {'PASSED': 0, 'NEW_FAILURES': 1, 'INCOMPLETE': 2}[status]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', default=None, help='OWNER/REPO (default: the current checkout)')
    parser.add_argument('--log-dir', default=None, help='cache directory for downloaded logs and job metadata')
    sub = parser.add_subparsers(dest='command', required=True)

    def tree_args(p):
        p.add_argument('--tree', default=None, help='checkout whose tests/ lists the expected modules')
        p.add_argument('--tree-ref', default=None, help='git ref of this checkout instead of --tree')
        p.add_argument('--pattern', action='append', default=None, help=f'module file pattern (default {DEFAULT_PATTERNS})')
        p.add_argument('--allow-cross-head', action='store_true', help='fill tests across runs of different head SHAs')

    e = sub.add_parser('extract', help='settled failing IDs of runs (+ pod results), with coverage')
    e.add_argument('runs', nargs='+')
    e.add_argument('--out', required=True, help='ID list; a .json sidecar is written beside it')
    e.add_argument('--log', action='append', default=None, help='parse saved logs instead of downloading (one per run, in order)')
    e.add_argument('--log-head', action='append', default=None, help='head SHA of each saved --log')
    e.add_argument('--pod', action='append', default=None, help='pod-run result JSON to fill unsettled tests')
    tree_args(e)
    s = sub.add_parser('supplement', help='module list for a pod supplement from an extract sidecar')
    s.add_argument('--sidecar', required=True);s.add_argument('--out', required=True)
    s.add_argument('--force', action='append', default=None, help='also run this module (e.g. changed since the CI head)')
    r = sub.add_parser('pod-run', help='run test modules on this machine through the CI entry point')
    r.add_argument('--tree', required=True);r.add_argument('--out', required=True, help='result JSON (failing IDs beside it as .txt)')
    r.add_argument('--modules', nargs='*');r.add_argument('--modules-file')
    r.add_argument('--team-pool', required=True, help='the CI fixture pool (team_pool --teams 1024 --seed-start 650200000)')
    r.add_argument('--showdown-root', required=True)
    r.add_argument('--jobs', type=int, default=8);r.add_argument('--timeout', type=float, default=3 * 3600)
    r.add_argument('--allow-dirty', action='store_true')
    d = sub.add_parser('diff', help='new failing IDs of a PR against a baseline')
    d.add_argument('--baseline', required=True)
    d.add_argument('--pr-run', action='append', default=None)
    d.add_argument('--pr-pod', action='append', default=None)
    d.add_argument('--pr-ids', default=None)
    d.add_argument('--log', action='append', default=None, help='saved logs for the --pr-run runs')
    d.add_argument('--log-head', action='append', default=None)
    tree_args(d)
    args = parser.parse_args(argv)
    if args.command == 'diff' and not (args.pr_run or args.pr_pod or args.pr_ids):parser.error('give --pr-run, --pr-pod or --pr-ids')
    commands = {'extract': extract, 'supplement': supplement_modules, 'pod-run': pod_run, 'diff': diff}
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
