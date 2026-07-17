"""Tests for lineage resolution in scripts/trait_inventory.py.

Forked lineages are the sharp edge here: a fork starts at its fork point and has no history
below it. Two failure modes must stay fixed — the milestone grid inventing pre-fork checkpoints,
and the G0 gate failing a fork for lacking a 500k it could never have.
"""
import importlib.util
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trait_inventory.py"


def _load():
    spec = importlib.util.spec_from_file_location("trait_inventory", _SCRIPT)
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
