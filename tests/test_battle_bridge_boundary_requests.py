"""Pins for preserving a streamed decision boundary in generic snapshots."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "battle_bridge_boundary_requests.mjs"


def _select(stream: object, simulator: object) -> object:
    program = """
const { snapshotBoundaryRequests } = await import(process.argv[1]);
const [stream, simulator] = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(snapshotBoundaryRequests(stream, simulator)));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", program, str(HELPER), json.dumps([stream, simulator])],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return json.loads(completed.stdout)


class SnapshotBoundaryRequestsTest(unittest.TestCase):
    def test_empty_direct_requests_preserve_streamed_actionable_boundary(self) -> None:
        streamed = {"p1": {"side": {"id": "p1"}, "active": [{}]}}

        self.assertEqual(_select(streamed, {}), streamed)

    def test_nonempty_direct_requests_replace_prior_stream_boundary(self) -> None:
        streamed = {
            "p1": {"side": {"id": "p1"}, "active": [{}]},
            "p2": {"side": {"id": "p2"}, "active": [{}]},
        }
        direct = {"p2": {"side": {"id": "p2"}, "forceSwitch": [True]}}

        self.assertEqual(_select(streamed, direct), direct)

    def test_empty_sources_remain_empty(self) -> None:
        self.assertEqual(_select({}, {}), {})
