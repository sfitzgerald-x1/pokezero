"""Gates for the FoulPlay power-config merger.

This script computes the campaign's ONLY deliverable: the paired delta
(search − raw) per cell. Everything pinned here is a way of producing a
plausible number instead of an error:

* the delta must be the mean of PER-PAIR differences, not the difference of two
  arms' means -- those coincide only when both arms cover the same pairs;
* a cell must pair against its OWN checkpoint's raw arm;
* a cell over the latency cap must not be adoptable, however large its delta;
* two builds, or two shards disagreeing about one game, must refuse to merge.

Pure fixtures -- no cluster, no checkpoint, no crate.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "foulplay_power_report_test", REPO_ROOT / "scripts" / "foulplay_power_report.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_R = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_R)

FP = "f" * 64


def shard(config_id, arm, checkpoint, scores, *, gate=4.0, fingerprint=FP):
    """`scores` maps (seed, seat) -> score."""
    return {
        "schema_version": "pokezero.foulplay-paired-shard.v1",
        "arm": arm,
        "config_id": config_id,
        "checkpoint": checkpoint,
        "engine_fingerprint": fingerprint,
        "provenance_sha256": "a" * 64,
        "seed_start": 0,
        "pairs": len(scores),
        "opponent_priors": False,
        "per_seat": {
            seat: {
                "search_wall_per_searched_decision": gate,
                "wall_per_decision_p95": None if gate is None else gate * 1.5,
                "wall_per_decision_mean": None if gate is None else gate * 0.9,
                "fallback_rate": 0.008,
                "depth_reached_mean": 3.1,
                "world_failure_reasons": {},
            }
            for seat in ("p1", "p2")
        },
        "rows": [
            {"seed": s, "seat": seat, "won": v > 0, "tied": False, "capped": False,
             "score": v}
            for (s, seat), v in sorted(scores.items())
        ],
    }


def run(shards, **kw):
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, sh in enumerate(shards):
            p = Path(d) / f"s{i}.json"
            p.write_text(json.dumps(sh), encoding="utf-8")
            paths.append(str(p))
        argv = list(paths)
        # Fixtures are deliberately small; the section 8 minimum (400) is the
        # production default and is exercised by MinimumPairsTest below.
        kw.setdefault("min_pairs", 1)
        for k, v in kw.items():
            argv += [f"--{k.replace('_','-')}", str(v)]
        out = Path(d) / "report.json"
        argv += ["--out", str(out)]
        import contextlib, io

        with contextlib.redirect_stdout(io.StringIO()):
            _R.main(argv)
        return json.loads(out.read_text(encoding="utf-8"))


class PairedDeltaTest(unittest.TestCase):
    def test_delta_is_per_pair_not_difference_of_means(self) -> None:
        """The two estimators must DIVERGE here, or this pins nothing.

        Round 3 caught the earlier fixture: it gave paired == diff-of-means ==
        1.0, so it could not tell the estimators apart while its docstring
        claimed it did. This one is built so they genuinely disagree.

        Shared pairs: search and raw tie (both 0.0), so the PAIRED delta is 0.
        Unshared: search has an extra win, raw an extra loss, which drags the
        two arms' MEANS apart. Difference-of-means would report +0.333; the
        paired estimator must report 0.0.
        """
        search = {(1, "p1"): 0.0, (2, "p1"): 0.0, (3, "p1"): 1.0}
        raw = {(1, "p1"): 0.0, (2, "p1"): 0.0, (4, "p1"): 0.0}
        # Sanity: the two estimators really do differ on this fixture.
        diff_of_means = (sum(search.values()) / len(search)) - (sum(raw.values()) / len(raw))
        self.assertAlmostEqual(diff_of_means, 1 / 3)

        rep = run([shard("cell@k0", "search", "/c/k0.pt", search),
                   shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None)])
        cell = rep["cells"]["cell@k0"]
        self.assertEqual(cell["pairs"], 2)
        self.assertAlmostEqual(cell["paired_delta"]["point"], 0.0)
        self.assertNotAlmostEqual(cell["paired_delta"]["point"], diff_of_means)
        self.assertEqual(cell["dropped_unpaired"], {"search_only": 1, "raw_only": 1})

    def test_unpaired_rows_are_dropped_and_counted_never_half_scored(self) -> None:
        search = {(i, "p1"): 1.0 for i in range(10)}
        raw = {(i, "p1"): 0.0 for i in range(5)}
        cell = run([shard("cell@k0", "search", "/c/k0.pt", search),
                    shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None)])["cells"]["cell@k0"]
        self.assertEqual(cell["pairs"], 5)
        self.assertEqual(cell["dropped_unpaired"]["search_only"], 5)

    def test_mcnemar_counts_discordant_pairs(self) -> None:
        search = {(0, "p1"): 1.0, (1, "p1"): 1.0, (2, "p1"): 0.0, (3, "p1"): 1.0}
        raw = {(0, "p1"): 0.0, (1, "p1"): 1.0, (2, "p1"): 1.0, (3, "p1"): 0.0}
        cell = run([shard("cell@k0", "search", "/c/k0.pt", search),
                    shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None)])["cells"]["cell@k0"]
        # pairs 0 and 3 favour search, pair 2 favours raw, pair 1 is concordant.
        self.assertEqual(cell["mcnemar"]["search_better"], 2)
        self.assertEqual(cell["mcnemar"]["raw_better"], 1)
        self.assertEqual(cell["mcnemar"]["discordant"], 3)


class RawArmMatchingTest(unittest.TestCase):
    def test_each_cell_pairs_against_its_own_checkpoints_raw_arm(self) -> None:
        # The collision that motivated checkpoint-qualified config_ids: cells A
        # and G run the same search config on different checkpoints, and the
        # raw arm is the denominator of both deltas.
        a = {(i, "p1"): 1.0 for i in range(6)}
        g = {(i, "p1"): 1.0 for i in range(6)}
        r0 = {(i, "p1"): 0.0 for i in range(6)}
        r1 = {(i, "p1"): 1.0 for i in range(6)}
        rep = run([
            shard("d4-s1024-b64-w4@k0", "search", "/c/k0.pt", a),
            shard("d4-s1024-b64-w4@k1", "search", "/c/k1.pt", g),
            shard("raw@k0", "raw", "/c/k0.pt", r0, gate=None),
            shard("raw@k1", "raw", "/c/k1.pt", r1, gate=None),
        ])
        self.assertEqual(rep["cells"]["d4-s1024-b64-w4@k0"]["raw_arm"], "raw@k0")
        self.assertEqual(rep["cells"]["d4-s1024-b64-w4@k1"]["raw_arm"], "raw@k1")
        # k0's raw lost everything, k1's won everything -- deltas must differ.
        self.assertAlmostEqual(rep["cells"]["d4-s1024-b64-w4@k0"]["paired_delta"]["point"], 1.0)
        self.assertAlmostEqual(rep["cells"]["d4-s1024-b64-w4@k1"]["paired_delta"]["point"], 0.0)


class LatencyGateTest(unittest.TestCase):
    def _two_cells(self, slow_gate):
        fast = {(i, "p1"): 1.0 if i % 3 else 0.0 for i in range(30)}
        slow = {(i, "p1"): 1.0 for i in range(30)}
        raw = {(i, "p1"): 0.0 for i in range(30)}
        return run([
            shard("fast@k0", "search", "/c/k0.pt", fast, gate=4.0),
            shard("slow@k0", "search", "/c/k0.pt", slow, gate=slow_gate),
            shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None),
        ], anchor="fast@k0")

    def test_cell_over_the_cap_is_rejected_despite_the_largest_delta(self) -> None:
        rep = self._two_cells(22.5)
        self.assertGreater(rep["cells"]["slow@k0"]["paired_delta"]["point"],
                           rep["cells"]["fast@k0"]["paired_delta"]["point"])
        self.assertIn("REJECTED", rep["cells"]["slow@k0"]["cap"])
        self.assertNotIn("slow@k0", rep["ranking_eligible"])
        self.assertNotEqual(rep["winner"], "slow@k0")

    def test_same_cell_under_the_cap_is_eligible(self) -> None:
        rep = self._two_cells(19.0)
        self.assertIn("PASS", rep["cells"]["slow@k0"]["cap"])
        self.assertIn("slow@k0", rep["ranking_eligible"])

    def test_missing_gate_field_is_unevaluable_not_a_pass(self) -> None:
        # A fully-fallen-back cell emits no search_wall_per_searched_decision.
        # Reading that as a pass would adopt a config whose latency is unknown.
        search = {(i, "p1"): 1.0 for i in range(5)}
        raw = {(i, "p1"): 0.0 for i in range(5)}
        cell = run([shard("c@k0", "search", "/c/k0.pt", search, gate=None),
                    shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None)])["cells"]["c@k0"]
        self.assertIn("UNEVALUABLE", cell["cap"])


class FailClosedTest(unittest.TestCase):
    def test_two_build_eras_refuse_to_merge(self) -> None:
        s = {(0, "p1"): 1.0}
        with self.assertRaises(SystemExit) as caught:
            run([shard("c@k0", "search", "/c/k0.pt", s),
                 shard("raw@k0", "raw", "/c/k0.pt", s, gate=None, fingerprint="b" * 64)])
        self.assertIn("build", str(caught.exception).lower())

    def test_expect_fingerprint_mismatch_refuses(self) -> None:
        s = {(0, "p1"): 1.0}
        with self.assertRaises(SystemExit):
            run([shard("c@k0", "search", "/c/k0.pt", s),
                 shard("raw@k0", "raw", "/c/k0.pt", s, gate=None)],
                expect_fingerprint="c" * 64)

    def test_conflicting_scores_for_one_game_are_terminal(self) -> None:
        a = shard("c@k0", "search", "/c/k0.pt", {(0, "p1"): 1.0})
        b = shard("c@k0", "search", "/c/k0.pt", {(0, "p1"): 0.0})
        with self.assertRaises(SystemExit) as caught:
            run([a, b, shard("raw@k0", "raw", "/c/k0.pt", {(0, "p1"): 0.0}, gate=None)])
        self.assertIn("conflicting", str(caught.exception).lower())

    def test_no_raw_arm_is_terminal(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            run([shard("c@k0", "search", "/c/k0.pt", {(0, "p1"): 1.0})])
        self.assertIn("raw", str(caught.exception).lower())


class SeatHealthTest(unittest.TestCase):
    def test_disjoint_per_seat_deltas_flag_stop_and_investigate(self) -> None:
        # #937 bug class: search helps one seat and hurts the other.
        search = {}
        raw = {}
        for i in range(40):
            search[(i, "p1")] = 1.0
            raw[(i, "p1")] = 0.0
            search[(i, "p2")] = 0.0
            raw[(i, "p2")] = 1.0
        cell = run([shard("c@k0", "search", "/c/k0.pt", search),
                    shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None)])["cells"]["c@k0"]
        self.assertIn("seat_health", cell)
        self.assertIn("STOP-AND-INVESTIGATE", cell["seat_health"])

    def test_symmetric_seats_are_not_flagged(self) -> None:
        search, raw = {}, {}
        for i in range(40):
            for seat in ("p1", "p2"):
                search[(i, seat)] = 1.0 if i % 2 else 0.0
                raw[(i, seat)] = 0.0
        cell = run([shard("c@k0", "search", "/c/k0.pt", search),
                    shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None)])["cells"]["c@k0"]
        self.assertNotIn("seat_health", cell)


class WilsonTest(unittest.TestCase):
    def test_wilson_brackets_the_point_estimate(self) -> None:
        low, high = _R.wilson(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_wilson_of_empty_sample_is_degenerate_not_a_crash(self) -> None:
        self.assertEqual(_R.wilson(0, 0), (0.0, 0.0))


class AdoptionRuleTest(unittest.TestCase):
    """Section 9 Phase 2 (iii): a CI on the IMPROVEMENT, not delta-vs-point."""

    def _pair(self, cand_extra_wins):
        # Anchor and candidate tie on most pairs; candidate strictly dominates
        # on `cand_extra_wins`. Both share the raw arm, so the improvement is
        # very tightly estimated even though each delta individually is not.
        n = 60
        anchor, cand, raw = {}, {}, {}
        for i in range(n):
            k = (i, "p1")
            raw[k] = 0.0
            anchor[k] = 1.0 if i % 2 else 0.0
            cand[k] = anchor[k]
        for i in range(cand_extra_wins):
            cand[(i * 2, "p1")] = 1.0
        return [
            shard("anchor@k0", "search", "/c/k0.pt", anchor),
            shard("cand@k0", "search", "/c/k0.pt", cand),
            shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None),
        ]

    def test_candidate_adopted_when_the_improvement_ci_excludes_zero(self) -> None:
        rep = run(self._pair(6), anchor="anchor@k0")
        imp = rep["cells"]["cand@k0"]["improvement_over_anchor"]
        self.assertGreater(imp["low"], 0.0)
        self.assertEqual(rep["winner"], "cand@k0")

    def test_anchor_retained_when_the_improvement_ci_includes_zero(self) -> None:
        # One extra win over 60 pairs: a positive point estimate whose CI
        # straddles 0. The old rule (candidate.low > anchor.point) could adopt
        # here; the plan's rule must not.
        rep = run(self._pair(1), anchor="anchor@k0")
        imp = rep["cells"]["cand@k0"]["improvement_over_anchor"]
        self.assertGreater(imp["point"], 0.0)
        self.assertLessEqual(imp["low"], 0.0)
        self.assertEqual(rep["winner"], "anchor@k0")
        self.assertIn("includes 0", rep["adoption_rule"])

    def test_a_mistyped_anchor_is_fatal_not_silently_ignored(self) -> None:
        # Cell ids are checkpoint-qualified, so typos are easy; falling back to
        # largest-delta-wins would disable the adoption rule with no diagnostic.
        with self.assertRaises(SystemExit) as caught:
            run(self._pair(6), anchor="anchorTYPO@k0")
        self.assertIn("not among the shards", str(caught.exception))

    def test_an_unscoreable_anchor_is_fatal(self) -> None:
        # An anchor with no shared pairs made the old comparison silently
        # degrade to "delta > 0".
        anchor = {(900, "p1"): 1.0}
        cand = {(i, "p1"): 1.0 for i in range(10)}
        raw = {(i, "p1"): 0.0 for i in range(10)}
        with self.assertRaises(SystemExit) as caught:
            run([shard("anchor@k0", "search", "/c/k0.pt", anchor),
                 shard("cand@k0", "search", "/c/k0.pt", cand),
                 shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None)],
                anchor="anchor@k0")
        self.assertIn("anchor", str(caught.exception).lower())


class HealthGateTest(unittest.TestCase):
    def test_fallback_above_the_limit_makes_a_cell_ineligible(self) -> None:
        search = {(i, "p1"): 1.0 for i in range(10)}
        raw = {(i, "p1"): 0.0 for i in range(10)}
        sh = shard("c@k0", "search", "/c/k0.pt", search)
        for seat in sh["per_seat"].values():
            seat["fallback_rate"] = 0.05
        rep = run([sh, shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None)])
        self.assertNotIn("c@k0", rep["ranking_eligible"])
        self.assertTrue(any("fallback" in r for r in rep["cells"]["c@k0"]["ineligible_because"]))

    def test_short_cell_is_ineligible_at_the_section_8_minimum(self) -> None:
        search = {(i, "p1"): 1.0 for i in range(10)}
        raw = {(i, "p1"): 0.0 for i in range(10)}
        rep = run([shard("c@k0", "search", "/c/k0.pt", search),
                   shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None)], min_pairs=400)
        self.assertNotIn("c@k0", rep["ranking_eligible"])
        self.assertTrue(
            any("minimum" in r for r in rep["cells"]["c@k0"]["ineligible_because"])
        )


if __name__ == "__main__":
    unittest.main()
