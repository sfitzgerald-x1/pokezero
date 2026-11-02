"""Shard the original component discovery suite without splitting its blocks."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


def partition_suite(suite, index: int, count: int):
    if type(count) is not int or count <= 0:
        raise ValueError("positive integer shard count required")
    if type(index) is not int or not 0 <= index < count:
        raise ValueError("shard index must be within the declared count")
    # Keep complete discovery blocks, including package load_tests suites and
    # their custom run behavior, rather than flattening individual test cases.
    blocks = list(suite)
    indices = list(range(index, len(blocks), count))
    if not indices:
        raise ValueError("empty component shard")
    return unittest.TestSuite(blocks[offset] for offset in indices), indices, len(blocks)


def run_suite(suite, *, stream=None) -> int:
    result = unittest.TextTestRunner(verbosity=2, stream=stream).run(suite)
    return int(not result.wasSuccessful() or bool(result.skipped) or result.testsRun == 0)


def _case_identity(test):
    # Component fixtures legitimately use id(name) as a lookup helper. Do not
    # invoke that unrelated override when identifying the discovered method.
    if isinstance(test, unittest.FunctionTestCase):
        return unittest.FunctionTestCase.id(test)
    return unittest.TestCase.id(test)


def _manifest(test):
    """Describe discovery without replacing any suite's original run method."""
    count = test.countTestCases()
    if type(count) is not int or count < 0:
        raise ValueError("invalid discovery test count")
    name = type(test).__module__ + "." + type(test).__qualname__
    if isinstance(test, unittest.TestCase):
        identity = _case_identity(test)
        if count != 1 or type(identity) is not str or not identity:
            raise ValueError("case requires one stable discovery identity")
        return dict(kind="case", type=name, count=1, id=identity)
    if not isinstance(test, unittest.BaseTestSuite):
        raise ValueError("isolated mode requires inspectable unittest discovery blocks")
    children = [_manifest(child) for child in test]
    if count != sum(child["count"] for child in children):
        raise ValueError("suite discovery count differs from its declared cases")
    return dict(kind="suite", type=name, count=count, children=children)


def _identities(node):
    if node["kind"] == "case":
        return [node["id"]]
    return [identity for child in node["children"] for identity in _identities(child)]


def _discover(spec):
    return unittest.TestLoader().discover(**spec)


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _numeric_fields(path, allowed, *, scale=1):
    """Read only named numeric counters; never emit arbitrary file content."""
    try:
        lines = path.read_text(errors='replace').splitlines()
    except OSError:
        return {}
    fields = {}
    for line in lines:
        parts = line.replace(':', ' ').split()
        if len(parts) >= 2 and parts[0] in allowed and parts[1].isdigit():
            fields[allowed[parts[0]]] = int(parts[1]) * scale
    return fields


def _resource_snapshot(pid, block_index, *, proc=Path('/proc'), cgroups=Path('/sys/fs/cgroup')):
    """Best-effort Linux counters only: no environment, command lines or names."""
    snapshot = dict(resource_snapshot=True, process_block=block_index,
        monotonic_ns=time.monotonic_ns(), clock_ticks_per_second=os.sysconf('SC_CLK_TCK'), processes=[])
    snapshot['vm'] = _numeric_fields(proc / 'meminfo', {
        'MemTotal': 'total_bytes', 'MemAvailable': 'available_bytes',
        'SwapTotal': 'swap_total_bytes', 'SwapFree': 'swap_free_bytes'}, scale=1024)
    snapshot['vm'].update(_numeric_fields(proc / 'vmstat', {'oom_kill': 'oom_kill'}))
    pending = [os.getpid(), pid]
    seen = set()
    # Include the actual block interpreter and its subprocesses, with bounded
    # sampling overhead. /proc stat's command-name field is never retained.
    while pending and len(seen) < 256:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        directory = proc / str(current)
        try:
            stat = (directory / 'stat').read_text(errors='replace')
            fields = stat[stat.rindex(')') + 2:].split()
            counters = dict(pid=current, parent_pid=int(fields[1]), user_ticks=int(fields[11]),
                system_ticks=int(fields[12]), start_ticks=int(fields[19]))
        except (OSError, ValueError, IndexError):
            continue
        counters.update(_numeric_fields(directory / 'status', {
            'VmRSS': 'rss_bytes', 'VmHWM': 'peak_rss_bytes', 'VmSize': 'virtual_bytes'}, scale=1024))
        counters.update(_numeric_fields(directory / 'status', {'Threads': 'threads'}))
        snapshot['processes'].append(counters)
        try:
            children = (directory / 'task' / str(current) / 'children').read_text().split()
            pending.extend(int(child) for child in children if child.isdigit())
        except OSError:
            pass
    snapshot['processes'].sort(key=lambda row: row['pid'])
    snapshot['cgroup'] = {}
    # Collect cgroup v2 when available. An unavailable counter stays absent,
    # rather than turning diagnostic collection into a test failure.
    for member in (pid, os.getpid()):
        try:
            entries = (proc / str(member) / 'cgroup').read_text(errors='replace').splitlines()
            relative = next(line[3:] for line in entries if line.startswith('0::'))
            group = (cgroups / relative.lstrip('/')).resolve()
            if not group.is_relative_to(cgroups.resolve()):
                continue
        except (OSError, StopIteration):
            continue
        for filename, key in (('memory.current', 'current_bytes'), ('memory.peak', 'peak_bytes'),
                ('memory.max', 'limit_bytes'), ('memory.swap.current', 'swap_bytes')):
            try:
                value = (group / filename).read_text().strip()
            except OSError:
                continue
            if value.isdigit():
                snapshot['cgroup'][key] = int(value)
        snapshot['cgroup'].update(_numeric_fields(group / 'memory.events',
            {key: key for key in ('low', 'high', 'max', 'oom', 'oom_kill', 'oom_group_kill')}))
        break
    return snapshot


def _run_block(command, environment, *, resource_interval_seconds, block_index, stream):
    if not resource_interval_seconds:
        return subprocess.run(command, env=environment, check=False).returncode
    # Fresh exec and normal exit status remain unchanged. Timed waits only
    # sample counters; they never terminate, retry, skip or time-limit a block.
    with subprocess.Popen(command, env=environment) as process:
        while True:
            print(json.dumps(_resource_snapshot(process.pid, block_index)), file=stream, flush=True)
            try:
                returncode = process.wait(timeout=resource_interval_seconds)
            except subprocess.TimeoutExpired:
                continue
            print(json.dumps(_resource_snapshot(process.pid, block_index)), file=stream, flush=True)
            return returncode


class _ObservedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed_ids = []

    def startTest(self, test):
        self.observed_ids.append(_case_identity(test))
        super().startTest(test)


def _case_manifests(node):
    if node['kind'] == 'case':
        return [node]
    return [case for child in node['children'] for case in _case_manifests(child)]


def _isolation_units(blocks, indices, classes):
    """Split explicitly named single-class standard suites, never custom runs.

    Module/class fixtures run afresh around each selected case. This policy is
    opt-in for classes whose cases do not require cross-case fixture state.
    """
    if not classes:
        return [dict(block_index=index, case_index=None) for index in indices], []
    units, decisions = [], []
    for index in indices:
        cases, standard = [], True
        def inspect(node):
            nonlocal standard
            if isinstance(node, unittest.TestCase):
                cases.append(node)
            elif (type(node) is unittest.TestSuite
                    and getattr(node.run, '__func__', None) is unittest.TestSuite.run
                    and not (set(vars(node)) - {'_tests', '_removed_tests'})):
                for child in node:
                    inspect(child)
            else:
                # Even another __iter__ call could alter a custom suite. Leave
                # its subtree entirely to the original discovery/run path.
                standard = False
        inspect(blocks[index])
        names = {type(case).__module__ + '.' + type(case).__qualname__ for case in cases}
        requested = bool(names.intersection(classes)) or not standard
        reason = 'not_requested'
        if requested:
            reason = 'custom_or_mixed_suite'
            if standard and len(names) == 1:
                reason = 'custom_case_run'
                if all(getattr(case.run, '__func__', None) is unittest.TestCase.run
                        and type(case).__call__ is unittest.TestCase.__call__ for case in cases):
                    reason = 'load_tests'
                    module = type(cases[0]).__module__.split('.')
                    if not any(getattr(sys.modules.get('.'.join(module[:end])), 'load_tests', None) is not None
                            for end in range(1, len(module) + 1)):
                        reason = 'fresh_standard_case'
        if reason == 'fresh_standard_case':
            units.extend(dict(block_index=index, case_index=offset) for offset in range(len(cases)))
        else:
            units.append(dict(block_index=index, case_index=None))
        if requested:
            decisions.append(dict(block_index=index, case_isolation=reason == 'fresh_standard_case', reason=reason))
    return units, decisions


def _one_case_suite(block, case_index):
    """Retain original case objects and standard-suite nesting, never ID reload."""
    offset = 0
    def retain(node):
        nonlocal offset
        if isinstance(node, unittest.TestCase):
            selected = offset == case_index
            offset += 1
            return node if selected else None
        children = [selected for child in node if (selected := retain(child)) is not None]
        return unittest.TestSuite(children) if children else None
    return retain(block)


def _unit_manifest(plan, block_index, case_index):
    block = plan['manifest'][block_index]
    return block if case_index is None else _case_manifests(block)[case_index]


def _valid_receipt(receipt, plan, block_index, case_index=None):
    if case_index is not None and (type(case_index) is not int
            or not 0 <= case_index < len(_case_manifests(plan['manifest'][block_index]))
            or dict(block_index=block_index, case_index=case_index) not in plan.get('execution_units', [])):
        return False
    expected = _unit_manifest(plan, block_index, case_index)
    integers = ("tests_run", "failures", "errors", "skipped", "expected_failures", "unexpected_successes")
    return (type(receipt) is dict and receipt.get("schema") == "component-block-terminal.v1"
        and receipt.get("plan_sha256") == _digest(plan)
        and type(receipt.get("block_index")) is int and receipt["block_index"] == block_index
        and receipt.get("case_index") == case_index
        and (case_index is None or type(receipt.get("case_index")) is int)
        and receipt.get("block_manifest") == plan['manifest'][block_index]
        and type(receipt.get("declared_count")) is int and receipt["declared_count"] == expected["count"]
        and all(type(receipt.get(key)) is int and receipt[key] >= 0 for key in integers)
        and receipt["tests_run"] == expected["count"] > 0
        and type(receipt.get("observed_ids")) is list
        and all(type(value) is str for value in receipt["observed_ids"])
        and Counter(receipt["observed_ids"]) == Counter(_identities(expected))
        and receipt.get("successful") is True
        and receipt["failures"] == receipt["errors"] == receipt["skipped"] == receipt["unexpected_successes"] == 0)


def _child(plan_path, plan_sha256, block_index, receipt_path, case_index=None):
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("schema") != "component-block-plan.v1" or _digest(plan) != plan_sha256:
        raise ValueError("child discovery plan changed")
    discovered = _discover(plan["discovery"])
    blocks = list(discovered)
    _, indices, total = partition_suite(discovered, plan["index"], plan["count"])
    if ([_manifest(block) for block in blocks] != plan["manifest"]
            or indices != plan["selected_blocks"] or total != len(plan["manifest"])
            or type(block_index) is not int or block_index not in indices):
        raise ValueError("child rediscovery manifest/index/count differs")
    units, decisions = _isolation_units(blocks, indices, plan['isolate_standard_classes'])
    if (units != plan['execution_units'] or decisions != plan['isolation_decisions']
            or dict(block_index=block_index, case_index=case_index) not in units):
        raise ValueError('child case isolation policy differs from discovery plan')
    # Custom suites/load_tests retain their entire original block. Only an
    # explicitly opted-in plain suite gets one original case with fresh normal
    # module/class/setup/cleanup lifecycle in this process.
    block = blocks[block_index] if case_index is None else _one_case_suite(blocks[block_index], case_index)
    result = unittest.TextTestRunner(verbosity=2, resultclass=_ObservedResult).run(
        unittest.TestSuite([block]))
    receipt = dict(schema="component-block-terminal.v1", plan_sha256=plan_sha256,
        block_index=block_index, case_index=case_index, block_manifest=plan["manifest"][block_index],
        declared_count=_unit_manifest(plan, block_index, case_index)['count'], tests_run=result.testsRun,
        observed_ids=result.observed_ids, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), expected_failures=len(result.expectedFailures),
        unexpected_successes=len(result.unexpectedSuccesses), successful=result.wasSuccessful())
    # Only the completed runner writes this receipt. exit0/crash without it
    # cannot pass the parent, nor can a custom run dropping/repeating cases.
    target = Path(receipt_path)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    temporary.replace(target)
    return int(not _valid_receipt(receipt, plan, block_index, case_index))


def run_process_blocks(suite, index, count, *, start_dir, pattern="test_gpu_engine*.py",
                       top_level_dir=None, stream=None, resource_interval_seconds=0,
                       isolate_standard_classes=()):
    """Optional sequential fresh-exec blocks; default whole-suite API unchanged.

    Discovery is repeated and verified in each fresh sys.executable process.
    Suite run order stays owned by the original custom run method; observed
    case multiplicities/counts must match discovery exactly. Noninspectable or
    nondeterministic discovery fails closed in this optional mode.
    """
    if not math.isfinite(resource_interval_seconds) or resource_interval_seconds < 0:
        raise ValueError("resource interval must be finite and nonnegative")
    if (type(isolate_standard_classes) not in (tuple, list)
            or any(type(name) is not str or not name for name in isolate_standard_classes)):
        raise ValueError('case isolation requires explicit class names')
    classes = sorted(set(isolate_standard_classes))
    stream = sys.stdout if stream is None else stream
    _, indices, _ = partition_suite(suite, index, count)
    spec = dict(start_dir=str(Path(start_dir).resolve()), pattern=pattern)
    if top_level_dir is not None:
        spec["top_level_dir"] = str(Path(top_level_dir).resolve())
    blocks = list(suite)
    units, decisions = _isolation_units(blocks, indices, classes)
    plan = dict(schema="component-block-plan.v1", discovery=spec, index=index, count=count,
        selected_blocks=indices, manifest=[_manifest(block) for block in blocks],
        isolate_standard_classes=classes, execution_units=units, isolation_decisions=decisions)
    print(json.dumps(dict(case_isolation_decisions=decisions)), file=stream, flush=True)
    environment = dict(os.environ)
    # A fresh interpreter must exercise THIS checkout, not another editable
    # install. Keep existing caller search paths after these explicit roots.
    roots = [str(ROOT / "src"), str(ROOT)]
    if environment.get("PYTHONPATH"):
        roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    results = []
    with tempfile.TemporaryDirectory(prefix="component-blocks-") as directory:
        folder = Path(directory)
        plan_path = folder / "plan.json"
        plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n")
        for unit in units:
            block_index, case_index = unit['block_index'], unit['case_index']
            target = folder / f"block-{block_index}-case-{case_index}.json"
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                "--_child-plan", str(plan_path), "--_plan-sha256", _digest(plan),
                "--_block-index", str(block_index), "--_receipt", str(target)]
            if case_index is not None:
                command.extend(['--_case-index', str(case_index)])
            print(json.dumps(dict(process_block=block_index, case_index=case_index,
                declared_count=_unit_manifest(plan, block_index, case_index)['count'])), file=stream, flush=True)
            # No os.fork, multiprocessing fork context or preexec_fn after JAX.
            # The child starts a fresh interpreter and independently discovers.
            try:
                returncode = _run_block(command, environment,
                    resource_interval_seconds=resource_interval_seconds,
                    block_index=block_index, stream=stream)
            except OSError:
                returncode = None
            try:
                receipt = json.loads(target.read_text())
                valid = _valid_receipt(receipt, plan, block_index, case_index)
            except (OSError, ValueError, TypeError, KeyError):
                receipt, valid = None, False
            counts = {key: receipt.get(key) if type(receipt) is dict else None
                for key in ("tests_run", "skipped", "failures", "errors")}
            results.append(dict(block_index=block_index, case_index=case_index, returncode=returncode,
                valid_terminal_receipt=valid, **counts))
    passed = (len(results) == len(units)
        and [dict(block_index=result['block_index'], case_index=result['case_index']) for result in results] == units
        and all(result["returncode"] == 0 and result["valid_terminal_receipt"] for result in results))
    totals = {key: sum(result[key] for result in results)
        if all(type(result[key]) is int and result[key] >= 0 for result in results) else None
        for key in ("tests_run", "skipped", "failures", "errors")}
    print(json.dumps(dict(process_blocks_terminal=True, passed=passed, selected_blocks=indices,
        execution_units=units, expected_test_cases=sum(plan["manifest"][i]["count"] for i in indices),
        blocks=results, **totals)),
        file=stream, flush=True)
    return int(not passed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--process-blocks", action="store_true",
        help="run each original discovery block in a verified fresh Python process")
    parser.add_argument("--resource-interval-seconds", type=float, default=0,
        help="sample numeric Linux memory/CPU counters during process blocks (0 disables)")
    parser.add_argument('--isolate-standard-class', action='append', default=[],
        help='explicit class whose plain suite cases each get fresh module/class fixtures')
    parser.add_argument("--_child-plan", help=argparse.SUPPRESS)
    parser.add_argument("--_plan-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--_block-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument('--_case-index', type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_receipt", help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args._child_plan is not None:
        if args._plan_sha256 is None or args._block_index is None or args._receipt is None:
            parser.error("complete private child invocation required")
        return _child(args._child_plan, args._plan_sha256, args._block_index, args._receipt, args._case_index)
    if args.index is None or args.count is None:
        parser.error("--index and --count are required")
    discovered = unittest.defaultTestLoader.discover(
        start_dir=str(ROOT / "tests"), pattern="test_gpu_engine*.py"
    )
    selected, indices, total = partition_suite(discovered, args.index, args.count)
    print(json.dumps({"shard": args.index, "count": args.count,
        "discovery_blocks": total, "selected_blocks": indices,
        "test_cases": selected.countTestCases()}), flush=True)
    if args.process_blocks:
        return run_process_blocks(discovered, args.index, args.count, start_dir=ROOT / "tests",
            resource_interval_seconds=args.resource_interval_seconds,
            isolate_standard_classes=args.isolate_standard_class)
    return run_suite(selected)


if __name__ == "__main__":
    raise SystemExit(main())
