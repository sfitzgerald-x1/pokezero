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


class ModelPathDepthInstrumentationTest(unittest.TestCase):
    """The model path must accumulate reached depth, like the hp_fraction path.

    Without this a depth ladder cannot be interpreted: "depth does not help" and
    "the simulation budget never let the tree reach the cap" produce the same flat
    ladder, and only the reached-depth histogram separates them.
    """

    def test_model_path_accumulates_reached_depth(self) -> None:
        import re

        source = (REPO_ROOT / "src" / "pokezero" / "engine_search.py").read_text()
        # The two accumulation sites must both exist: the hp_fraction path and
        # the model path. Locate them by their neighbouring model-only counter.
        self.assertIn("self.stats.model_evals += int(report[\"model_evals\"])", source)
        model_block = source.split("self.stats.model_evals += int(report[\"model_evals\"])")[1][:900]
        for field in (
            "depth_reached_samples",
            "depth_reached_sum",
            "depth_reached_max",
            "depth_reached_histogram",
        ):
            self.assertIn(
                field,
                model_block,
                f"model path does not accumulate {field}; a depth ladder run on "
                "leaf_eval='model' would carry no reached-depth evidence",
            )
        self.assertEqual(
            len(re.findall(r"depth_reached_histogram\[reached\] \+= 1", source)),
            2,
            "expected exactly two accumulation sites: hp_fraction and model",
        )

    def test_runner_emits_the_policy_stats_payload(self) -> None:
        source = (REPO_ROOT / "scripts" / "mcts_acceptance_h2h.py").read_text()
        self.assertIn("policy_stats", source)
        self.assertIn("to_payload()", source)


class DepthPayloadDisambiguationTest(unittest.TestCase):
    """"No data" and "the cap never bound" must not be the same bytes on disk.

    A depth ladder is read straight off these counters. If an absent histogram
    and a histogram that stayed empty serialize identically, a shard that
    silently reported nothing is indistinguishable from one proving the cap was
    never reached -- and the second is a finding while the first is a bug.
    """

    def _module(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "h2h", REPO_ROOT / "scripts" / "mcts_acceptance_h2h.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_every_shape_carries_the_depth_keys(self) -> None:
        m = self._module()

        class NoPayload:
            pass

        class WithData:
            def to_payload(self):
                return {
                    "depth_reached_samples": 3,
                    "depth_reached_max": 6,
                    "depth_reached_histogram": {"2": 1, "6": 2},
                }

        for stats in (NoPayload(), WithData()):
            payload = m._policy_stats_payload(stats)
            for key in (
                "search_ran",
                "depth_reached_samples",
                "depth_reached_max",
                "depth_reached_histogram",
            ):
                self.assertIn(key, payload, f"{type(stats).__name__} is missing {key}")

    def test_control_arm_states_its_own_emptiness(self) -> None:
        """The control must say search_ran=False, not emit a bare {}."""
        source = (REPO_ROOT / "scripts" / "mcts_acceptance_h2h.py").read_text()
        self.assertIn('"search_ran": False', source)
        # and the real data path must still be able to report a bound cap
        m = self._module()

        class Bound:
            def to_payload(self):
                return {"depth_reached_samples": 10, "depth_reached_histogram": {"6": 10}}

        payload = m._policy_stats_payload(Bound())
        self.assertTrue(payload["search_ran"])
        self.assertEqual(payload["depth_reached_histogram"], {"6": 10})
