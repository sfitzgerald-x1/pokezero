"""Gates for the §8 acceptance merge/report path.

The acceptance run exists to settle a claim that a *pooled* number already hid
once (docs/mcts_degradation_findings.md §11), so the reporting path has three
jobs and each is pinned here:

* score the two seats SEPARATELY;
* refuse to score an incomplete mirrored pair (fail-closed, in-house rule from
  ``mcts_eval.scoring.pair_scores``) while still naming it;
* refuse to merge shards produced by two different engine builds.

Runs on shard fixtures — no cluster, no checkpoint, no torch.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "scripts" / "mcts_acceptance_report.py"
FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


def shard(
    path: Path,
    *,
    arm: str,
    config_id: str,
    pair_start: int,
    pairs: int,
    p1_outcome: str,
    p2_outcome: str,
    fingerprint: str = FINGERPRINT_A,
    drop: tuple[int, str] | None = None,
) -> Path:
    rows = []
    for index in range(pairs):
        seed = pair_start + index
        for seat, outcome in (("p1", p1_outcome), ("p2", p2_outcome)):
            if drop == (seed, seat):
                continue
            rows.append(
                {
                    "config_id": config_id,
                    "seed": seed,
                    "seat": seat,
                    "outcome": outcome,
                    "turns": 40,
                    "provenance_sha256": f"prov-{arm}",
                    "opponent_crashed": False,
                }
            )
    path.write_text(
        json.dumps(
            {
                "schema_version": "pokezero.mcts-acceptance-shard.v1",
                "arm": arm,
                "config_id": config_id,
                "checkpoint": "ckpt",
                "engine_fingerprint": fingerprint,
                "provenance_sha256": f"prov-{arm}",
                "pair_start": pair_start,
                "pairs": pairs,
                "games": len(rows),
                "total_decisions": 1,
                "fallback_decisions": 0,
                "fallback_rate": 0.0,
                "fallback_reasons": {},
                "world_failure_reasons": {},
                "search_wall_per_decision": 0.0,
                "wall_s": 1.0,
                "results": rows,
                "per_game": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def run_report(*paths: Path, extra: list[str] | None = None):
    return subprocess.run(
        [sys.executable, str(REPORT), *[str(p) for p in paths], *(extra or [])],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


class AcceptanceReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_seats_are_reported_separately(self) -> None:
        """A seat-split defect must be visible even when the pool looks fine.

        p1 wins every game and p2 loses every game: the pooled pair mean is
        exactly 0.500 — the same number a healthy arm would show — while the
        seats are 1.000 and 0.000. This is §11's failure mode in miniature.
        """
        shard(
            self.tmp / "s0.json",
            arm="search",
            config_id="d4-s1024-b64-w4",
            pair_start=7800000,
            pairs=20,
            p1_outcome="win",
            p2_outcome="loss",
        )
        result = run_report(self.tmp / "s0.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("p1 seat  n=  20  score=1.000", result.stdout)
        self.assertIn("p2 seat  n=  20  score=0.000", result.stdout)
        self.assertIn("pooled pair mean  0.500", result.stdout)

    def test_incomplete_pair_is_named_and_never_scored(self) -> None:
        shard(
            self.tmp / "s0.json",
            arm="search",
            config_id="d4-s1024-b64-w4",
            pair_start=7800000,
            pairs=10,
            p1_outcome="win",
            p2_outcome="win",
            drop=(7800004, "p2"),
        )
        result = run_report(self.tmp / "s0.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("complete pairs     : 9", result.stdout)
        self.assertIn("INCOMPLETE: [7800004]", result.stdout)
        # The surviving p1 game of the broken pair must not inflate the seat n.
        self.assertIn("p1 seat  n=   9", result.stdout)

    def test_shards_from_two_builds_are_refused(self) -> None:
        shard(
            self.tmp / "a.json",
            arm="search",
            config_id="d4-s1024-b64-w4",
            pair_start=7800000,
            pairs=5,
            p1_outcome="win",
            p2_outcome="loss",
        )
        shard(
            self.tmp / "b.json",
            arm="search",
            config_id="d4-s1024-b64-w4",
            pair_start=7800010,
            pairs=5,
            p1_outcome="win",
            p2_outcome="loss",
            fingerprint=FINGERPRINT_B,
        )
        result = run_report(self.tmp / "a.json", self.tmp / "b.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mixes 2 engine builds", result.stdout + result.stderr)

    def test_expected_fingerprint_is_enforced(self) -> None:
        shard(
            self.tmp / "a.json",
            arm="search",
            config_id="d4-s1024-b64-w4",
            pair_start=7800000,
            pairs=5,
            p1_outcome="win",
            p2_outcome="loss",
        )
        result = run_report(
            self.tmp / "a.json", extra=["--expect-fingerprint", FINGERPRINT_B]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("staged config expects", result.stdout + result.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
