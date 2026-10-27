"""Shard the original component discovery suite without splitting its blocks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    discovered = unittest.defaultTestLoader.discover(
        start_dir=str(ROOT / "tests"), pattern="test_gpu_engine*.py"
    )
    selected, indices, total = partition_suite(discovered, args.index, args.count)
    print(json.dumps({"shard": args.index, "count": args.count,
        "discovery_blocks": total, "selected_blocks": indices,
        "test_cases": selected.countTestCases()}), flush=True)
    return run_suite(selected)


if __name__ == "__main__":
    raise SystemExit(main())
