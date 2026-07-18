"""Tests for scripts/trait_inventory.py.

Two concerns. The script binds its storage paths only from caller-provided environment
configuration, never repo defaults. And lineage resolution handles forks correctly: a fork starts
at its fork point and has no history below it, so two failure modes must stay fixed — the milestone
grid inventing pre-fork checkpoints, and the G0 gate failing a fork for lacking a 500k it could
never have.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trait_inventory.py"


class TraitInventoryTest(unittest.TestCase):
    def run_inventory(self, environment: dict[str, str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT)],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
            cwd=cwd,
        )

    def test_requires_explicit_storage_configuration(self) -> None:
        environment = dict(os.environ)
        environment.pop("POKEZERO_SHARED_ROOT", None)
        environment.pop("POKEZERO_TRAIT_INVENTORY_OUT", None)

        result = self.run_inventory(environment)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("set POKEZERO_SHARED_ROOT", result.stderr)

    def test_requires_an_explicit_inventory_output_path(self) -> None:
        environment = dict(os.environ)
        environment["POKEZERO_SHARED_ROOT"] = "."
        environment.pop("POKEZERO_TRAIT_INVENTORY_OUT", None)

        result = self.run_inventory(environment)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("set POKEZERO_TRAIT_INVENTORY_OUT", result.stderr)

    def test_writes_a_filename_only_configured_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = dict(os.environ)
            environment["POKEZERO_SHARED_ROOT"] = str(root)
            environment["POKEZERO_TRAIT_INVENTORY_OUT"] = "inventory.json"

            result = self.run_inventory(environment, cwd=root)

            self.assertEqual(result.returncode, 0, result.stderr)
            inventory = json.loads((root / "inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(inventory["schema"], "pokezero.trait_inventory.v1")


def _load():
    spec = importlib.util.spec_from_file_location("trait_inventory", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TI = _load()


def leg(run, offset, max_iter, gpi=1600):
    return {"run": run, "offset": offset, "games_per_iteration": gpi,
            "retained_iterations": list(range(1, max_iter + 1)), "max_iter": max_iter,
            "terminal_games": offset + max_iter * gpi}


class MilestoneGrid(unittest.TestCase):
    def test_continuation_legs_cover_the_whole_axis(self):
        # a normal lineage: legs chain from 0, so every 100k milestone is covered
        legs = [leg("a", 0, 320), leg("b", 512000, 300)]   # 0..512k, 512k..992k
        grid, frontier = TI.milestone_map(legs)
        self.assertEqual([g["milestone"] for g in grid], [100_000 * i for i in range(1, 10)])
        self.assertEqual(frontier, 992_000)

    def test_fork_gets_no_pre_fork_milestones(self):
        # a fork starting at 2M must NOT manufacture milestones for 100k..2M. The old fallback
        # picked the nearest leg and clamped want_iter to 1, inventing ~20 bogus checkpoints.
        legs = [leg("fork", 2_000_000, 200)]               # 2.0M .. 2.32M
        grid, frontier = TI.milestone_map(legs)
        self.assertTrue(all(g["milestone"] > 2_000_000 for g in grid),
                        "fork must not claim milestones below its fork point")
        self.assertEqual([g["milestone"] for g in grid], [2_100_000, 2_200_000, 2_300_000])
        self.assertEqual(frontier, 2_320_000)

    def test_fork_too_young_has_empty_grid(self):
        # forked at 2M with only 34 iterations (~54k games) — has not reached 2.1M yet
        legs = [leg("fork", 2_000_000, 34)]
        grid, _ = TI.milestone_map(legs)
        self.assertEqual(grid, [], "a fork below its first milestone should yield no points")


class ForkIsSeparateFromParent(unittest.TestCase):
    def test_fork_pattern_does_not_match_parent_and_vice_versa(self):
        import re
        parent_pat = TI.LINEAGES["v22-lr3m"][0]
        fork_pat = TI.LINEAGES["v22-flat2m"][0]
        fork_run, parent_run = "emeta-v2-2-flat2m-belief", "emeta-v2-2-lr3m-3m-belief"
        # the fork must never be swept into the parent lineage as if it were a continuation leg
        self.assertIsNone(re.match(parent_pat, fork_run))
        self.assertIsNotNone(re.match(fork_pat, fork_run))
        self.assertIsNone(re.match(fork_pat, parent_run))
        self.assertIsNotNone(re.match(parent_pat, parent_run))


if __name__ == "__main__":
    unittest.main()
