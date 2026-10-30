"""Shard the original component discovery suite without splitting its blocks."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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


class _ObservedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed_ids = []

    def startTest(self, test):
        self.observed_ids.append(_case_identity(test))
        super().startTest(test)


def _valid_receipt(receipt, plan, block_index):
    expected = plan["manifest"][block_index]
    integers = ("tests_run", "failures", "errors", "skipped", "expected_failures", "unexpected_successes")
    return (type(receipt) is dict and receipt.get("schema") == "component-block-terminal.v1"
        and receipt.get("plan_sha256") == _digest(plan)
        and type(receipt.get("block_index")) is int and receipt["block_index"] == block_index
        and receipt.get("block_manifest") == expected
        and type(receipt.get("declared_count")) is int and receipt["declared_count"] == expected["count"]
        and all(type(receipt.get(key)) is int and receipt[key] >= 0 for key in integers)
        and receipt["tests_run"] == expected["count"] > 0
        and type(receipt.get("observed_ids")) is list
        and all(type(value) is str for value in receipt["observed_ids"])
        and Counter(receipt["observed_ids"]) == Counter(_identities(expected))
        and receipt.get("successful") is True
        and receipt["failures"] == receipt["errors"] == receipt["skipped"] == receipt["unexpected_successes"] == 0)


def _child(plan_path, plan_sha256, block_index, receipt_path):
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
    # Wrap the original block intact: package load_tests/custom run/nesting
    # remain executable behavior. Never reconstruct a suite from flattened IDs.
    result = unittest.TextTestRunner(verbosity=2, resultclass=_ObservedResult).run(
        unittest.TestSuite([blocks[block_index]]))
    receipt = dict(schema="component-block-terminal.v1", plan_sha256=plan_sha256,
        block_index=block_index, block_manifest=plan["manifest"][block_index],
        declared_count=plan["manifest"][block_index]["count"], tests_run=result.testsRun,
        observed_ids=result.observed_ids, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), expected_failures=len(result.expectedFailures),
        unexpected_successes=len(result.unexpectedSuccesses), successful=result.wasSuccessful())
    # Only the completed runner writes this receipt. exit0/crash without it
    # cannot pass the parent, nor can a custom run dropping/repeating cases.
    target = Path(receipt_path)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    temporary.replace(target)
    return int(not _valid_receipt(receipt, plan, block_index))


def run_process_blocks(suite, index, count, *, start_dir, pattern="test_gpu_engine*.py",
                       top_level_dir=None, stream=None):
    """Optional sequential fresh-exec blocks; default whole-suite API unchanged.

    Discovery is repeated and verified in each fresh sys.executable process.
    Suite run order stays owned by the original custom run method; observed
    case multiplicities/counts must match discovery exactly. Noninspectable or
    nondeterministic discovery fails closed in this optional mode.
    """
    stream = sys.stdout if stream is None else stream
    _, indices, _ = partition_suite(suite, index, count)
    spec = dict(start_dir=str(Path(start_dir).resolve()), pattern=pattern)
    if top_level_dir is not None:
        spec["top_level_dir"] = str(Path(top_level_dir).resolve())
    plan = dict(schema="component-block-plan.v1", discovery=spec, index=index, count=count,
        selected_blocks=indices, manifest=[_manifest(block) for block in suite])
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
        for block_index in indices:
            target = folder / f"block-{block_index}.json"
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                "--_child-plan", str(plan_path), "--_plan-sha256", _digest(plan),
                "--_block-index", str(block_index), "--_receipt", str(target)]
            print(json.dumps(dict(process_block=block_index,
                declared_count=plan["manifest"][block_index]["count"])), file=stream, flush=True)
            # No os.fork, multiprocessing fork context or preexec_fn after JAX.
            # The child starts a fresh interpreter and independently discovers.
            try:
                returncode = subprocess.run(command, env=environment, check=False).returncode
            except OSError:
                returncode = None
            try:
                receipt = json.loads(target.read_text())
                valid = _valid_receipt(receipt, plan, block_index)
            except (OSError, ValueError, TypeError, KeyError):
                receipt, valid = None, False
            counts = {key: receipt.get(key) if type(receipt) is dict else None
                for key in ("tests_run", "skipped", "failures", "errors")}
            results.append(dict(block_index=block_index, returncode=returncode,
                valid_terminal_receipt=valid, **counts))
    passed = (len(results) == len(indices)
        and [result["block_index"] for result in results] == indices
        and all(result["returncode"] == 0 and result["valid_terminal_receipt"] for result in results))
    totals = {key: sum(result[key] for result in results)
        if all(type(result[key]) is int and result[key] >= 0 for result in results) else None
        for key in ("tests_run", "skipped", "failures", "errors")}
    print(json.dumps(dict(process_blocks_terminal=True, passed=passed, selected_blocks=indices,
        expected_test_cases=sum(plan["manifest"][i]["count"] for i in indices), blocks=results, **totals)),
        file=stream, flush=True)
    return int(not passed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--process-blocks", action="store_true",
        help="run each original discovery block in a verified fresh Python process")
    parser.add_argument("--_child-plan", help=argparse.SUPPRESS)
    parser.add_argument("--_plan-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--_block-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_receipt", help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args._child_plan is not None:
        if args._plan_sha256 is None or args._block_index is None or args._receipt is None:
            parser.error("complete private child invocation required")
        return _child(args._child_plan, args._plan_sha256, args._block_index, args._receipt)
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
        return run_process_blocks(discovered, args.index, args.count, start_dir=ROOT / "tests")
    return run_suite(selected)


if __name__ == "__main__":
    raise SystemExit(main())
