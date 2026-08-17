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

# The producer's think-block schema, read from the module under test rather than retyped:
# a fixture pinned to a stale literal would pool as `think_schema_mismatch` and every
# eligibility assertion below would fail for the wrong reason.
import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))
from pokezero.foulplay_bridge import FOULPLAY_THINK_SCHEMA_VERSION  # noqa: E402

#: Sentinel distinguishing "caller said nothing about the think block" (healthy default)
#: from "caller said there is no block", which is a refusal the gate must produce.
_HEALTHY = object()


def think(
    rate=380_000.0,
    decisions=120,
    *,
    strata=None,
    attempted=None,
    record_failures=0,
    observable=True,
    start_method="fork",
    schema_version=None,
):
    """One seat's `foulplay_think` run header, in the producer's shape.

    Deliberately built from RATES AND COUNTS, never from a pre-supplied verdict: the gate's
    inputs are what a fixture may set, and `status`, the ratios and the coverage are what the
    gate has to derive. A fixture that handed over `iterations_coverage` directly could not
    see the coverage division being deleted.
    """
    strata = strata or {"2x1000ms": (rate, decisions)}
    attempted = decisions + record_failures if attempted is None else attempted
    measured = sum(n for _, n in strata.values())
    return {
        "schema_version": schema_version or FOULPLAY_THINK_SCHEMA_VERSION,
        "entries_key": "opponent_think",
        "decisions": decisions,
        "decisions_attempted": attempted,
        "record_failures": record_failures,
        "iterations_measured_decisions": measured,
        "total_iterations": int(
            sum(r * 2.0 * n for r, n in strata.values())
        ),
        "mean_iterations_per_budget_second": (
            sum(r * n for r, n in strata.values()) / measured if measured else None
        ),
        "iterations_coverage": (measured / attempted) if attempted else None,
        "iterations_observable": observable,
        "opponent_start_method": start_method,
        "by_stratum": {
            name: {
                "iterations_measured_decisions": n,
                "mean_iterations_per_budget_second": r,
            }
            for name, (r, n) in sorted(strata.items())
        },
        "miss_decisions": max(0, decisions - measured),
        "miss_reasons": {},
    }


def shard(config_id, arm, checkpoint, scores, *, gate=4.0, fingerprint=FP,
          opponent_engine_mcts=None, foulplay_think=_HEALTHY):
    """`scores` maps (seed, seat) -> score.

    `foulplay_think` defaults to a HEALTHY, matched reading -- 380,000 realized opponent
    visits per granted budget-second on 120 measured decisions, all in one stratum, full
    coverage, `fork`. Both arms getting the same one is the negative control for the
    cross-arm contention gate: every eligibility assertion in this file is also an assertion
    that the gate does not fire on matched arms. Pass `foulplay_think=None` for a shard whose
    producer never measured it.
    """
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
                "opponent_engine_mcts": opponent_engine_mcts,
                "foulplay_think": (
                    think() if foulplay_think is _HEALTHY else foulplay_think
                ),
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

    def test_the_gate_prefers_the_per_decision_wall_on_a_dynamic_cell(self) -> None:
        # `search_wall_per_searched_decision` is per-RUNG on a ladder cell:
        # `searched_decisions` is charged once per `_search_model` call and a ladder
        # calls it once per rung (measured 2,224 rungs against 1,062 decisions). So
        # gating on it lets a cell at ~2.1 rungs/decision report 5.7s when its true
        # per-decision wall is 12s, and the 20s/turn cap silently stops gating on
        # exactly the cells this feature exists to produce. Found in review.
        entry = {"per_seat": [{"p1": {
            "search_wall_per_searched_decision": 5.71,
            "search_wall_per_ladder_decision": 12.0,
            "ladder_rungs_per_decision": 2.1,
            "wall_per_decision_mean": 12.4,
            "wall_per_decision_p95": 18.0,
        }}]}
        lat = _R.latency_of(entry)
        self.assertEqual(lat["search_wall_per_searched_decision_mean"], 12.0)
        self.assertEqual(lat["gate_denominator"], "per_ladder_decision")
        self.assertAlmostEqual(lat["ladder_rungs_per_decision_mean"], 2.1)

    def test_a_fixed_cell_still_gates_on_the_field_it_always_did(self) -> None:
        # No ladder field means a fixed cell, where the gate field IS per-decision.
        # Every banked cell must keep reading exactly as it did.
        entry = {"per_seat": [{"p1": {"search_wall_per_searched_decision": 12.51}}]}
        lat = _R.latency_of(entry)
        self.assertEqual(lat["search_wall_per_searched_decision_mean"], 12.51)
        self.assertEqual(lat["gate_denominator"], "per_searched_decision_PER_RUNG")
        self.assertIsNone(lat["ladder_rungs_per_decision_mean"])

    def test_mixing_the_two_denominators_is_labelled_not_averaged(self) -> None:
        # Should be impossible -- assert_single_build refuses cross-fingerprint
        # merges -- but averaging a per-rung figure with a per-decision one produces
        # a number that is neither, so if it ever happens the reader must see it.
        entry = {"per_seat": [
            {"p1": {"search_wall_per_ladder_decision": 12.0}},
            {"p1": {"search_wall_per_searched_decision": 6.0}},
        ]}
        self.assertEqual(_R.latency_of(entry)["gate_denominator"],
                         "MIXED - do not compare")

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


class ArmWitnessTest(unittest.TestCase):
    """The merger must SAY which arm a cell was, for the two axes cell ids cannot.

    `+oracle-belief` is in config_id, so its risk is the opposite one: a shard
    written by an older driver would carry the flag in its body and NOT in its id,
    pooling the oracle arm into its own sampled control. Override telemetry is not
    in config_id at all by design, so the merger has to report the split or an
    override rate read off the cell's pair count is wrong by the telemetry-off
    share.
    """

    def _cell(self, report, cid):
        return report["cells"][cid]

    def test_the_oracle_arm_is_witnessed_and_the_sampled_twin_is_not(self) -> None:
        scores = {(0, "p1"): 1.0, (1, "p1"): 0.0}
        oracle = shard("d8-s1024-b64-w1+oracle-belief@k0", "search", "k0", scores)
        oracle["oracle_belief"] = True
        report = run([
            oracle,
            shard("d8-s1024-b64-w1@k0", "search", "k0", scores),
            shard("raw@k0", "raw", "k0", scores),
        ])
        self.assertIs(
            self._cell(report, "d8-s1024-b64-w1+oracle-belief@k0")["oracle_belief"],
            True,
        )
        self.assertIs(
            self._cell(report, "d8-s1024-b64-w1@k0")["oracle_belief"], False
        )

    def test_pooling_the_two_beliefs_into_one_cell_is_terminal(self) -> None:
        # An older driver, or a hand-edited shard, is the only way to reach this.
        # It must not merge: pooled, §4a's centerpiece figure is the average of
        # the two arms it contrasts.
        mixed = shard("d8-s1024-b64-w1@k0", "search", "k0", {(0, "p1"): 1.0})
        mixed["oracle_belief"] = True
        with self.assertRaises(SystemExit) as caught:
            run([
                shard("d8-s1024-b64-w1@k0", "search", "k0", {(1, "p1"): 0.0}),
                mixed,
                shard("raw@k0", "raw", "k0", {(0, "p1"): 0.5, (1, "p1"): 0.5}),
            ])
        self.assertIn("oracle-belief", str(caught.exception))

    def test_the_telemetry_split_inside_one_cell_is_counted(self) -> None:
        # Pooling here is CORRECT -- same search -- so the count is the only way a
        # reader learns the override rate's denominator covers half the games.
        on = shard("d8-s1024-b64-w1@k0", "search", "k0", {(0, "p1"): 1.0})
        on["override_telemetry"] = True
        off = shard("d8-s1024-b64-w1@k0", "search", "k0", {(1, "p1"): 0.0})
        report = run([
            on, off, shard("raw@k0", "raw", "k0", {(0, "p1"): 0.5, (1, "p1"): 0.5}),
        ])
        cell = self._cell(report, "d8-s1024-b64-w1@k0")
        self.assertEqual(cell["override_telemetry_shards"], {"on": 1, "off": 1})
        # And the pooling itself still happened: both seeds are in the cell.
        self.assertEqual(cell["pairs"], 2)




class OpponentHealthGateTest(unittest.TestCase):
    """A head-to-head cell is only as clean as its OPPONENT seat.

    Every engine health figure describes the pokezero seat. In a budget comparison the
    opponent is half the experiment, so an opponent falling back produces a flat result
    that reads as "the two budgets are equivalent" -- a contaminated cell presenting as a
    tie, which is worse than presenting as a fault.
    """

    def _health(self, **seat):
        import importlib.util as u
        from pathlib import Path
        sp = u.spec_from_file_location(
            "pr", Path(__file__).resolve().parents[1] / "scripts" / "foulplay_power_report.py")
        m = u.module_from_spec(sp); sp.loader.exec_module(m)
        return m, m.health_of({"per_seat": [{"p1": seat}]})

    def test_a_clean_opponent_reports_a_rate(self) -> None:
        _, h = self._health(fallback_rate=0.0,
                            opponent_engine_mcts={"fallback_rate": 0.01})
        self.assertEqual(h["opponent_fallback_rate"], 0.01)

    def test_no_neural_opponent_reports_None_not_zero(self) -> None:
        """Absence must not read as a clean opponent. A vs-foul-play cell has no opponent
        engine at all, and 0.0 would assert health that was never measured."""
        _, h = self._health(fallback_rate=0.0)
        self.assertIsNone(h["opponent_fallback_rate"])

    def test_a_falling_back_opponent_makes_the_cell_ineligible(self) -> None:
        """Drives the REAL report, not a restatement of the threshold.

        The first version of this test asserted only that 0.75 > FALLBACK_LIMIT, which
        stays green if the gate is deleted outright. That is the same "does not traverse
        production" weakness that let a blocker through earlier in this PR: a config_id
        fragment was tested on the helper while the production caller never passed it.
        """
        sick = shard("d3-s2048-b16-w1+vs-engine-mcts-d6-s16384@k0", "search", "k0",
                     {(0, "p1"): 1.0, (1, "p1"): 0.0},
                     opponent_engine_mcts={"fallback_rate": 0.75, "decisions": 100,
                                           "policy_mode": "engine-mcts"})
        rep = run([sick, shard("raw@k0", "raw", "k0",
                               {(0, "p1"): 1.0, (1, "p1"): 0.0}, gate=None)])
        cell = rep["cells"]["d3-s2048-b16-w1+vs-engine-mcts-d6-s16384@k0"]
        self.assertAlmostEqual(cell["health"]["opponent_fallback_rate"], 0.75)
        self.assertIn("OPPONENT fallback", json.dumps(cell.get("ineligible_because")),
                      "an opponent over the limit must disqualify the cell")

    def test_a_healthy_opponent_does_not_disqualify(self) -> None:
        ok = shard("d3-s2048-b16-w1+vs-engine-mcts-d6-s16384@k0", "search", "k0",
                   {(0, "p1"): 1.0, (1, "p1"): 0.0},
                   opponent_engine_mcts={"fallback_rate": 0.005, "decisions": 100,
                                         "policy_mode": "engine-mcts"})
        rep = run([ok, shard("raw@k0", "raw", "k0",
                             {(0, "p1"): 1.0, (1, "p1"): 0.0}, gate=None)])
        cell = rep["cells"]["d3-s2048-b16-w1+vs-engine-mcts-d6-s16384@k0"]
        self.assertAlmostEqual(cell["health"]["opponent_fallback_rate"], 0.005)
        self.assertNotIn("OPPONENT fallback", json.dumps(cell.get("ineligible_because")))


class CrossArmContentionGateTest(unittest.TestCase):
    """THE CALLER for the cross-arm contention comparison, which is what this report is.

    The opponent-think instrument lands per shard and the merged shard runs it p1-against-p2
    WITHIN one arm; within-arm symmetry cannot see a between-arm difference, because both
    seats of the hungry arm are equally starved. This report is the first place a search
    arm's shards and its raw arm's shards are both in hand.

    The gate's semantics here are the cap's: anything but `ok` makes the cell UNSCORED, not a
    null. The unit-level behaviour of the gate itself is pinned in
    `tests/test_cross_arm_contention.py`; what these tests pin is that the report RUNS it, on
    the right pair of arms, and that a refusal reaches eligibility and adoption.
    """

    def _cells(self, search_think, raw_think=None):
        """A starved-vs-matched contrast where the SUSPECT cell has the largest delta.

        Deliberate: if the gate were merely a ranking tiebreak, the suspect cell would still
        be adopted. Eligibility is what has to exclude it.

        The anchor cell `clean@k0` is given the SAME reading as the raw arm, not the module
        default: one raw arm serves both search cells, so a raw arm moved for the suspect's sake
        would also make the anchor look starved -- and a contention-refused anchor now blocks
        adoption outright, which would mask what these tests are trying to see.
        """
        suspect = {(i, "p1"): 1.0 for i in range(30)}
        clean = {(i, "p1"): 1.0 if i % 3 else 0.0 for i in range(30)}
        raw = {(i, "p1"): 0.0 for i in range(30)}
        return run([
            shard("suspect@k0", "search", "/c/k0.pt", suspect, foulplay_think=search_think),
            shard("clean@k0", "search", "/c/k0.pt", clean,
                  **({} if raw_think is None else {"foulplay_think": raw_think})),
            shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None,
                  **({} if raw_think is None else {"foulplay_think": raw_think})),
        ], anchor="clean@k0")

    def test_a_starved_search_arm_is_ineligible_and_not_adopted(self) -> None:
        """The confound, at the size the instrument has already measured: 3.8x.

        The starved cell won every game it played. Without this gate it is the winner.
        """
        rep = self._cells(think(224_200.0 / 3.8, 200), think(224_200.0, 200))
        suspect = rep["cells"]["suspect@k0"]
        self.assertEqual(suspect["contention"]["status"], "refused")
        self.assertIn(
            "cross_arm_rate_ratio_exceeds_threshold",
            suspect["contention"]["refusal_reasons"],
        )
        self.assertGreater(suspect["paired_delta"]["point"],
                           rep["cells"]["clean@k0"]["paired_delta"]["point"])
        self.assertIn("CONTENTION-CONFOUNDED", json.dumps(suspect["ineligible_because"]))
        self.assertNotIn("suspect@k0", rep["ranking_eligible"])
        self.assertNotEqual(rep["winner"], "suspect@k0")

    def test_matched_arms_leave_the_same_cell_eligible(self) -> None:
        """The negative control on the same fixture: the gate must not fire on matched arms.

        Same shards, same delta; only the opponent's realized rate on the search arm moves.
        Every other test in this file is also this control, since `shard()` defaults to a
        matched reading -- but the pair with the test above is what shows the gate is reading
        the rate and not the shard.
        """
        rep = self._cells(think(224_200.0, 200), think(224_200.0, 200))
        suspect = rep["cells"]["suspect@k0"]
        self.assertEqual(suspect["contention"]["status"], "ok")
        self.assertNotIn("CONTENTION-CONFOUNDED", json.dumps(suspect["ineligible_because"]))
        self.assertIn("suspect@k0", rep["ranking_eligible"])
        self.assertEqual(rep["winner"], "suspect@k0")

    def test_a_measured_matched_pair_of_real_searches_stays_eligible(self) -> None:
        """The real numbers, not a round fixture.

        Two uncontended passes of 24 real foul-play searches read 367,062 and 396,750 visits
        per granted budget-second (`crossarm-contention-dispersion.py`, 48/48 measured, zero
        misses). A gate that refuses THAT has manufactured the confound it exists to detect.
        """
        rep = self._cells(think(367_062.5, 24), think(396_750.0, 24))
        suspect = rep["cells"]["suspect@k0"]
        self.assertEqual(suspect["contention"]["refusal_reasons"], [])
        self.assertAlmostEqual(
            suspect["contention"]["worst_stratum"]["fold_ratio"], 1.0809, places=4
        )
        self.assertIn("suspect@k0", rep["ranking_eligible"])

    def test_a_seat_whose_producer_never_measured_makes_the_cell_ineligible(self) -> None:
        """A shard with no coverage must refuse, not read as "no contention detected".

        `foulplay_think=None` is what `seat_block` lifts from a producer that predates the
        instrument. Nothing else about the cell changes.
        """
        rep = self._cells(None)
        suspect = rep["cells"]["suspect@k0"]
        self.assertIn(
            "search:think_block_absent_in_pool", suspect["contention"]["refusal_reasons"]
        )
        self.assertIn("CONTENTION-CONFOUNDED", json.dumps(suspect["ineligible_because"]))
        self.assertNotIn("suspect@k0", rep["ranking_eligible"])

    def test_a_spawn_shard_makes_the_cell_ineligible(self) -> None:
        """`iterations` is only observable under `fork`; `spawn` emits nothing at all."""
        rep = self._cells(think(224_200.0, 200, observable=False, start_method="spawn"),
                          think(224_200.0, 200))
        self.assertIn(
            "search:start_method_cannot_emit_iterations",
            rep["cells"]["suspect@k0"]["contention"]["refusal_reasons"],
        )
        self.assertNotIn("suspect@k0", rep["ranking_eligible"])

    def test_a_refused_comparison_leaves_no_ratio_in_the_artifact(self) -> None:
        """The withholding rule, checked on the serialized report.

        A refused comparison's reassuring 1.0 must not exist in the file, because the file
        outlives the sentence next to it. Checked by searching the JSON text, not the object,
        so a nested copy cannot hide.
        """
        rep = self._cells(
            think(decisions=510, strata={"8x500ms": (59_000.0, 500),
                                         "2x1000ms": (380_000.0, 10)}),
            think(decisions=510, strata={"4x750ms": (224_200.0, 500),
                                         "2x1000ms": (380_000.0, 10)}),
        )
        suspect = rep["cells"]["suspect@k0"]
        self.assertEqual(suspect["contention"]["status"], "refused")
        self.assertIn("rates_withheld_because", suspect["contention"])
        self.assertNotIn("ratio_lean_over_hungry", json.dumps(suspect["contention"]))
        self.assertNotIn("worst_stratum", suspect["contention"])

    def test_each_cell_is_compared_against_its_own_checkpoints_raw_arm(self) -> None:
        """A mis-joined raw arm would certify the wrong pair.

        k0's raw arm is starved and k1's is not, on identical search arms. Exactly one cell
        may refuse; comparing an arm against itself, or against whichever raw arm came first,
        gets this wrong in one direction or the other.
        """
        scores = {(i, "p1"): 1.0 for i in range(6)}
        raw = {(i, "p1"): 0.0 for i in range(6)}
        rep = run([
            shard("d4-s1024-b64-w4@k0", "search", "/c/k0.pt", scores),
            shard("d4-s1024-b64-w4@k1", "search", "/c/k1.pt", scores),
            shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None,
                  foulplay_think=think(380_000.0 * 3.8, 200)),
            shard("raw@k1", "raw", "/c/k1.pt", raw, gate=None),
        ])
        self.assertEqual(
            rep["cells"]["d4-s1024-b64-w4@k0"]["contention"]["status"], "refused"
        )
        self.assertEqual(rep["cells"]["d4-s1024-b64-w4@k1"]["contention"]["status"], "ok")

    def test_a_cell_spanning_two_shards_pools_every_shards_seats(self) -> None:
        """A campaign shards a cell by seed band, so an arm is several shards' worth of seats.

        The failing input is a cell whose FIRST shard is healthy and whose SECOND is starved:
        pooling only the first (or only the last) reads `ok` on half the arm's evidence. Both
        shards carry the same `config_id`, which is what makes them one cell.
        """
        first = {(i, "p1"): 1.0 for i in range(10)}
        second = {(i, "p1"): 1.0 for i in range(10, 20)}
        raw = {(i, "p1"): 0.0 for i in range(20)}
        rep = run([
            shard("cell@k0", "search", "/c/k0.pt", first),
            # Same decision count as the healthy default, so the pooled mean is a clean
            # average of the two shards and the arithmetic below is checkable by hand.
            shard("cell@k0", "search", "/c/k0.pt", second,
                  foulplay_think=think(380_000.0 / 3.8, 120)),
            shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None),
        ])
        cell = rep["cells"]["cell@k0"]
        self.assertEqual(cell["pairs"], 20, "both shards' rows must be in the cell")
        self.assertEqual(cell["contention"]["status"], "refused")
        self.assertIn(
            "cross_arm_rate_ratio_exceeds_threshold", cell["contention"]["refusal_reasons"]
        )
        # 2 healthy seats at 380,000 and 2 starved at 100,000 pool to 240,000 against the raw
        # arm's 380,000: a 1.58 fold. Half an arm's evidence is worth exactly half.
        self.assertAlmostEqual(
            cell["contention"]["worst_stratum"]["fold_ratio"], 1.5833, places=3
        )

    def test_the_same_cell_is_eligible_when_both_its_shards_are_healthy(self) -> None:
        """The control for the test above: only the second shard's rate moved."""
        first = {(i, "p1"): 1.0 for i in range(10)}
        second = {(i, "p1"): 1.0 for i in range(10, 20)}
        raw = {(i, "p1"): 0.0 for i in range(20)}
        rep = run([
            shard("cell@k0", "search", "/c/k0.pt", first),
            shard("cell@k0", "search", "/c/k0.pt", second),
            shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None),
        ])
        cell = rep["cells"]["cell@k0"]
        self.assertEqual(cell["pairs"], 20)
        self.assertEqual(cell["contention"]["status"], "ok")
        self.assertIn("cell@k0", rep["ranking_eligible"])

    def test_a_contention_refused_anchor_blocks_adoption_entirely(self) -> None:
        """Making the anchor INELIGIBLE is not enough; it is still the comparator.

        The adopted quantity is the paired improvement `candidate_delta - anchor_delta`, so a
        confounded anchor puts its confounded delta inside the number that gets adopted -- and
        the adoption string said only "largest eligible delta whose improvement over anchor@k0
        excludes 0". Found by independent review. Here only the ANCHOR's opponent is starved.
        """
        anchor_scores = {(i, "p1"): 1.0 if i % 2 else 0.0 for i in range(30)}
        chal = {(i, "p1"): 1.0 for i in range(30)}
        raw = {(i, "p1"): 0.0 for i in range(30)}
        rep = run([
            shard("anchor@k0", "search", "/c/k0.pt", anchor_scores,
                  foulplay_think=think(380_000.0 / 3.8, 200)),
            shard("chal@k0", "search", "/c/k0.pt", chal),
            shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None),
        ], anchor="anchor@k0")
        self.assertEqual(rep["cells"]["anchor@k0"]["contention"]["status"], "refused")
        self.assertEqual(rep["cells"]["chal@k0"]["contention"]["status"], "ok")
        self.assertIsNone(rep["winner"])
        self.assertIn("NO ADOPTION", rep["adoption_rule"])
        self.assertIn("CONTENTION-CONFOUNDED", rep["adoption_rule"])
        # And the improvement over the confounded anchor is not computed at all.
        self.assertNotIn("improvement_over_anchor", rep["cells"]["chal@k0"])

    def test_a_clean_anchor_still_adopts_on_the_same_path(self) -> None:
        """The control: only the anchor's opponent reading moved in the test above."""
        anchor_scores = {(i, "p1"): 1.0 if i % 2 else 0.0 for i in range(30)}
        chal = {(i, "p1"): 1.0 for i in range(30)}
        raw = {(i, "p1"): 0.0 for i in range(30)}
        rep = run([
            shard("anchor@k0", "search", "/c/k0.pt", anchor_scores),
            shard("chal@k0", "search", "/c/k0.pt", chal),
            shard("raw@k0", "raw", "/c/k0.pt", raw, gate=None),
        ], anchor="anchor@k0")
        self.assertEqual(rep["winner"], "chal@k0")
        self.assertIn("improvement_over_anchor", rep["cells"]["chal@k0"])

    def test_the_report_records_the_preregistered_threshold(self) -> None:
        """The threshold has to be IN the artifact, and it is not a CLI flag.

        A threshold that can be loosened at report time is not preregistered, and the
        direction it would be loosened in is known in advance.
        """
        rep = self._cells(think(224_200.0, 200), think(224_200.0, 200))
        self.assertEqual(rep["contention_gate"]["max_fold_ratio"], 1.25)
        self.assertEqual(rep["contention_gate"]["measured_decision_log_sd"], 0.0529)
        self.assertEqual(rep["contention_gate"]["measured_run_log_sd"], 0.0516)
        # And no CLI flag can move it: `main` accepts no contention argument at all.
        with self.assertRaises(SystemExit):
            run([shard("x@k0", "search", "/c/k0.pt", {(0, "p1"): 1.0}),
                 shard("raw@k0", "raw", "/c/k0.pt", {(0, "p1"): 0.0}, gate=None)],
                contention_fold_ratio=99.0)
        self.assertEqual(
            rep["cells"]["suspect@k0"]["contention"]["max_fold_ratio"],
            rep["contention_gate"]["max_fold_ratio"],
        )

    def test_the_headline_a_winner_reader_sees_is_the_shortfall_and_its_scope(self) -> None:
        """`contention: ok` must not read as "the strength comparison is clean" in three months.

        What the gate bounds is opponent THROUGHPUT; the campaign's deliverable is a win-rate
        delta, and nothing in this repo converts between them. This asserts the CLAIM a reader of
        `winner` sees, which is the durable artifact.

        An earlier version of this test asserted a keyed `tracked_follow_ups` entry instead. That
        was withdrawn on review: it tested the project's bookkeeping rather than the report's
        behaviour, and would have gone stale the moment the item was closed anywhere else. The
        prose is what survives, so the prose is what is pinned.
        """
        rep = self._cells(think(224_200.0, 200), think(224_200.0, 200))
        note = rep["winner_note"]
        # The number is the SHORTFALL, because readers subtract. Not the 1.3158 fold.
        self.assertIn("shortfall is at most 24.0%", note)
        self.assertIn("SHORTFALL is at most 24.0%", rep["contention_gate"]["pass_bounds"])
        # And the scope limit travels with it, including the measurement that would close it.
        self.assertIn("not in win-rate units", note)
        self.assertIn("unknown, not small", note)
        self.assertIn("NOT clearance", note)
        self.assertIn("has not been run", note)
        # No bookkeeping field: withdrawn deliberately, not forgotten.
        self.assertNotIn("tracked_follow_ups", rep)

    def test_the_gate_note_claims_no_error_rate_and_no_five_digit_floor(self) -> None:
        """Two retracted claims, kept out of the artifact that outlives the discussion.

        The preregistration note used to say a 3-sigma bound and left a reader to infer a rate.
        The run term has 5 degrees of freedom, so the note now carries the df, the chi-square
        upper bound and the <18% the data actually support -- and does NOT carry "1 in 300" or
        "the tightest threshold this instrument can support".
        """
        rep = self._cells(think(224_200.0, 200), think(224_200.0, 200))
        note = rep["contention_gate"]["note"]
        self.assertIn("5 degrees of freedom", note)
        self.assertIn("<18%", note)
        self.assertIn("1.58", note)
        self.assertIn("1.47-1.49", note)
        for struck in ("1 in 300", "once in 300", "tightest threshold", "1.2448"):
            self.assertNotIn(struck, note, f"retracted claim back in the artifact: {struck}")
        # BOTH conventions labelled, because one sentence's sign flips with each.
        self.assertIn("p=0.0027", note)
        self.assertIn("quote the p", note)
        self.assertIn("n=24", note)
        self.assertIn("quote the n", note)
        # The scope qualifier carries its MAGNITUDE, not only its mechanism.
        scope = rep["contention_gate"]["scope"]
        self.assertIn("UNDERSTATES", scope)
        self.assertIn("7.3%", scope)
        self.assertIn("0.9%", scope)
        self.assertIn("1.126", scope)
        # The constant itself did not move.
        self.assertEqual(rep["contention_gate"]["max_fold_ratio"], 1.25)

    def test_the_artifact_states_the_resolution_rule_and_its_two_diagnoses(self) -> None:
        """The disposition of an unresolvable stratum, in the artifact a reader keeps.

        Two independent fixes of the same dead band chose different dispositions -- refuse, or
        exclude -- and the reader of a report cannot see which one ran. So the artifact says: the
        floor, that it EXCLUDES rather than refuses, and that a coverage refusal names which of
        the two shortfalls it was. Without the last part an exclusion is reported as a coverage
        problem, which is the objection the refusing branch raised and it was a fair one.
        """
        rep = self._cells(think(224_200.0, 200), think(224_200.0, 200))
        gate = rep["contention_gate"]
        self.assertEqual(gate["resolving_stratum_decisions"], 27)
        rule = gate["resolution_rule"]
        self.assertIn("EXCLUDED", rule)
        self.assertIn("strata_excluded_for_resolution", rule)
        self.assertIn("cross_arm_share_excluded_for_resolution", rule)
        self.assertIn("cross_arm_strata_excluded_for_resolution_cover_too_little", rule)
        self.assertIn("cross_arm_compared_strata_cover_too_little", rule)
        self.assertIn("neither is contention", rule)
        # The 27-against-24 reconciliation, with the reason and not just the number.
        self.assertIn("27 and not the calibration's", rule)
        self.assertIn("1.249996", rule)
        self.assertIn("[0.051426, 0.051475]", rule)
        # And the two inert floors are named as inert.
        self.assertIn("inert at this", rule)
        # The note's floors say which SD they came from, because 1.5521/1.6692/1.7312 do not
        # come from the run component and a reader who meets them elsewhere needs to place them.
        note = gate["note"]
        self.assertIn("run COMPONENT 0.0516", note)
        self.assertIn("0.052745", note)
        for pass_mean_figure in ("1.5080", "1.5521", "1.6692", "1.7312"):
            self.assertIn(pass_mean_figure, note)
        self.assertIn("ONE-SIDED 95%", note)
        self.assertIn("1.580", note)
        self.assertIn("1.711", note)

    def test_every_scored_cell_carries_a_contention_verdict(self) -> None:
        """Deleting the call must not leave this file green.

        With `shard()` defaulting to a matched reading, a missing gate would pass every
        eligibility assertion in this file. This is the test that reads False on the deletion.
        """
        rep = self._cells(think(224_200.0, 200), think(224_200.0, 200))
        for cid, cell in rep["cells"].items():
            self.assertIn("contention", cell, f"{cid} has no contention verdict")
            self.assertIn(cell["contention"]["status"], ("ok", "refused"))
            self.assertEqual(cell["contention"]["hungry_arm"], "search")


if __name__ == "__main__":
    unittest.main()
