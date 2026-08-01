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
        self.assertIn("excludes 0", rep["adoption_rule"])  # phrased as "no cell ... excludes 0"

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


class CrossCheckpointImprovementTest(unittest.TestCase):
    """Cell G (k1/R1) vs anchor A (k0/R0) -- the raw arms do NOT cancel."""

    def test_improvement_subtracts_each_cells_own_raw_arm(self) -> None:
        # Anchor: k0 search ties its raw arm -> delta 0. Candidate: k1 search
        # beats its raw arm slightly -> delta small. `candidate - anchor`
        # ignores that k1's raw arm is strong and reports a huge improvement.
        n = 40
        a = {(i, "p1"): 1.0 for i in range(n)}
        r0 = {(i, "p1"): 1.0 for i in range(n)}          # anchor delta 0.0
        g = {(i, "p1"): 1.0 if i < 21 else 0.0 for i in range(n)}
        r1 = {(i, "p1"): 1.0 if i < 20 else 0.0 for i in range(n)}  # G delta +0.025
        rep = run([
            shard("anchor@k0", "search", "/c/k0.pt", a),
            shard("raw@k0", "raw", "/c/k0.pt", r0, gate=None),
            shard("cand@k1", "search", "/c/k1.pt", g),
            shard("raw@k1", "raw", "/c/k1.pt", r1, gate=None),
        ], anchor="anchor@k0")
        anchor_d = rep["cells"]["anchor@k0"]["paired_delta"]["point"]
        cand_d = rep["cells"]["cand@k1"]["paired_delta"]["point"]
        imp = rep["cells"]["cand@k1"]["improvement_over_anchor"]
        self.assertAlmostEqual(anchor_d, 0.0)
        self.assertAlmostEqual(cand_d, 0.025)
        # The improvement must equal the DIFFERENCE OF DELTAS, not
        # candidate-minus-anchor (which would be about -0.475 here).
        self.assertAlmostEqual(imp["point"], cand_d - anchor_d, places=6)


class FilterThenRankTest(unittest.TestCase):
    def test_runner_up_is_adopted_when_the_leader_fails_the_improvement(self) -> None:
        """Section 9 Phase 2 (iii) is a per-cell FILTER, not a leader-only test.

        Round 5 caught the previous fixture inverted: the cell it called
        "second" actually had the LARGER delta, so it WAS ranked[0] and the
        test passed under a leader-only implementation. This one asserts the
        ordering it depends on before asserting the outcome.
        """
        n = 60
        raw = {(i, "p1"): 0.0 for i in range(n)}
        anchor = {(i, "p1"): 1.0 if i < 30 else 0.0 for i in range(n)}
        # Leader: biggest delta, but its wins and losses against the anchor are
        # mixed, so the improvement CI straddles 0.
        leader = dict(anchor)
        for i in range(30, 48):
            leader[(i, "p1")] = 1.0
        for i in range(0, 10):
            leader[(i, "p1")] = 0.0
        # Runner-up: smaller delta, strictly dominates the anchor.
        runner_up = dict(anchor)
        for i in range(30, 36):
            runner_up[(i, "p1")] = 1.0
        rep = run([
            shard("anchor@k0", "search", "/c/k0.pt", anchor),
            shard("leader@k0", "search", "/c/k0.pt", leader),
            shard("runnerup@k0", "search", "/c/k0.pt", runner_up),
            shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None),
        ], anchor="anchor@k0")
        lead_d = rep["cells"]["leader@k0"]["paired_delta"]["point"]
        run_d = rep["cells"]["runnerup@k0"]["paired_delta"]["point"]
        # The fixture only means anything if the leader really does lead.
        self.assertGreater(lead_d, run_d, "fixture inverted: leader must rank first")
        self.assertEqual(rep["ranking_eligible"][0], "leader@k0")
        self.assertLessEqual(
            rep["cells"]["leader@k0"]["improvement_over_anchor"]["low"], 0.0
        )
        self.assertGreater(
            rep["cells"]["runnerup@k0"]["improvement_over_anchor"]["low"], 0.0
        )
        self.assertEqual(rep["winner"], "runnerup@k0")


class IneligibleAnchorTest(unittest.TestCase):
    """Round 5 finding, and round 6 found the first version of this ungated.

    The earlier fixture had NO eligible cells, so the report exited on the
    "nothing passed" path and never reached the anchor-eligibility branch --
    mutating that branch to `elif True:` left the suite green. These fixtures
    keep a live eligible cell so the branch is actually executed.
    """

    def _report(self, anchor_gate):
        n = 30
        raw = {(i, "p1"): 0.0 for i in range(n)}
        # Anchor is strong but (optionally) over the cap.
        anchor = {(i, "p1"): 1.0 for i in range(n)}
        # A genuinely eligible cell that does NOT beat the anchor, so the
        # fallback branch is the one under test.
        weak = {(i, "p1"): 1.0 if i < 5 else 0.0 for i in range(n)}
        return run([
            shard("anchor@k0", "search", "/c/k0.pt", anchor, gate=anchor_gate),
            shard("weak@k0", "search", "/c/k0.pt", weak, gate=4.0),
            shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None),
        ], anchor="anchor@k0")

    def test_over_cap_anchor_is_not_adopted_even_with_an_eligible_cell_present(self) -> None:
        rep = self._report(anchor_gate=31.0)
        # The branch is genuinely reached: something IS eligible.
        self.assertIn("weak@k0", rep["ranking_eligible"])
        self.assertIn("REJECTED", rep["cells"]["anchor@k0"]["cap"])
        self.assertIsNone(rep["winner"], "a cap-rejected anchor must not be adopted")
        self.assertIn("NO ADOPTION", rep["adoption_rule"])
        self.assertIn("ineligible", rep["adoption_rule"])

    def test_healthy_anchor_is_still_adopted_on_the_same_path(self) -> None:
        # The guard must not reject a healthy anchor, or it is not a guard but
        # a blanket refusal.
        rep = self._report(anchor_gate=4.0)
        self.assertIn("weak@k0", rep["ranking_eligible"])
        self.assertEqual(rep["winner"], "anchor@k0")


class ImprovementOverlapTest(unittest.TestCase):
    """Round 6: the overlap floor was present but ungated.

    A cell can clear --min-pairs on its OWN delta while overlapping the anchor
    on far fewer pairs. The improvement CI is computed over the OVERLAP, so a
    thin overlap yields a spuriously tight interval beside a healthy `pairs`.
    """

    def _report(self, min_pairs):
        # anchor covers 0..39. thin covers 0..4 plus a disjoint band 200..239,
        # so thin's OWN pairs (vs raw) is 45 -- comfortably over any floor --
        # while its OVERLAP with the anchor is only 5.
        # Anchor LOSES the five shared seeds, so the improvement over the
        # overlap is decisive (+1.0) -- that is what makes a thin overlap
        # dangerous rather than merely noisy.
        anchor = {(i, "p1"): 0.0 if i < 5 else (1.0 if i < 25 else 0.0) for i in range(40)}
        thin = {(i, "p1"): 1.0 for i in range(5)}
        thin.update({(i, "p1"): 1.0 for i in range(200, 240)})
        raw = {(i, "p1"): 0.0 for i in range(40)}
        raw.update({(i, "p1"): 0.0 for i in range(200, 240)})
        return run([
            shard("anchor@k0", "search", "/c/k0.pt", anchor),
            shard("thin@k0", "search", "/c/k0.pt", thin),
            shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None),
        ], anchor="anchor@k0", min_pairs=min_pairs)

    def test_thin_overlap_is_recorded_and_blocks_adoption(self) -> None:
        rep = self._report(min_pairs=20)
        imp = rep["cells"]["thin@k0"]["improvement_over_anchor"]
        # The overlap n must be REPORTED, not just used -- a reader comparing
        # the CI to the cell's own `pairs` would otherwise be misled.
        self.assertIn("pairs", imp)
        # The cell's own n is healthy; only the OVERLAP is thin. That gap is
        # the whole point -- reporting one while gating on the other is how a
        # 5-pair CI ends up beside a 45-pair cell.
        self.assertEqual(imp["pairs"], 5)
        self.assertEqual(rep["cells"]["thin@k0"]["pairs"], 45)
        self.assertNotEqual(rep["winner"], "thin@k0")
        self.assertTrue(
            any("overlap" in r for r in rep["cells"]["thin@k0"]["ineligible_because"])
        )

    def test_same_cell_is_adoptable_once_the_floor_admits_the_overlap(self) -> None:
        rep = self._report(min_pairs=3)
        self.assertEqual(rep["winner"], "thin@k0")


class DepthRuleTest(unittest.TestCase):
    """The section 5 non-starvation rule, and that it reports honestly."""

    CAMPAIGN = {
        "checkpoints": {"k0": {"path": "/store/k0/transformer-policy.pt"}},
        "cells": [
            {"cell_id": "H", "arm": "search", "checkpoint": "k0",
             "depth": 4, "sims": 2048, "batch": 64, "worlds": 4},
            {"cell_id": "I", "arm": "search", "checkpoint": "k0",
             "depth": 6, "sims": 2048, "batch": 64, "worlds": 4,
             "reads_against": "H"},
        ],
    }

    def _run(self, depth_i, depth_h=3.1):
        n = 20
        raw = {(i, "p1"): 0.0 for i in range(n)}
        h = {(i, "p1"): 1.0 if i < 10 else 0.0 for i in range(n)}
        i_ = {(i, "p1"): 1.0 for i in range(n)}
        sh_h = shard("d4-s2048-b64-w4@k0", "search", "/c/k0.pt", h)
        sh_i = shard("d6-s2048-b64-w4@k0", "search", "/c/k0.pt", i_)
        for seat in sh_h["per_seat"].values():
            seat["depth_reached_mean"] = depth_h
        for seat in sh_i["per_seat"].values():
            seat["depth_reached_mean"] = depth_i
        with tempfile.TemporaryDirectory() as d:
            cpath = Path(d) / "campaign.json"
            cpath.write_text(json.dumps(self.CAMPAIGN), encoding="utf-8")
            return run([sh_h, sh_i,
                        shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None)],
                       campaign=str(cpath))

    def test_rule_actually_matches_the_shards_cell_ids(self) -> None:
        # The regression: config_ids are tagged with the campaign KEY (k0) while
        # the rule derived them from the checkpoint PATH stem
        # (transformer-policy), so nothing matched, the rule never fired, and
        # the report still claimed it had been applied.
        rep = self._run(depth_i=2.5)
        self.assertIn("d6-s2048-b64-w4@k0", rep["depth_rule_applied"])
        self.assertEqual(rep["depth_rule_unmatched"], [])

    def test_starved_depth_cell_is_excluded_despite_the_largest_delta(self) -> None:
        rep = self._run(depth_i=2.5)
        cell = rep["cells"]["d6-s2048-b64-w4@k0"]
        self.assertGreater(cell["paired_delta"]["point"],
                           rep["cells"]["d4-s2048-b64-w4@k0"]["paired_delta"]["point"])
        self.assertTrue(any("BUDGET-STARVED" in r for r in cell["ineligible_because"]))
        self.assertNotIn("d6-s2048-b64-w4@k0", rep["ranking_eligible"])

    def test_depth_cell_that_out_reaches_its_reference_is_eligible(self) -> None:
        rep = self._run(depth_i=4.0)
        cell = rep["cells"]["d6-s2048-b64-w4@k0"]
        self.assertEqual(cell["ineligible_because"], [])
        self.assertIn("d6-s2048-b64-w4@k0", rep["ranking_eligible"])


if __name__ == "__main__":
    unittest.main()
