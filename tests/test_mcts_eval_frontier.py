"""Frontier join + parity language (plan deliverable 8, section 3).

The load-bearing properties: a configuration with no strength read must never be
treated as weak, domination must require being at least as good on BOTH axes,
and a 20-game screen must never claim parity.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pokezero.mcts_eval.frontier import (
    FrontierRow,
    build_frontier,
    mark_frontier,
    parity_label,
    render_markdown,
    wilson_interval,
)


def _row(config_id: str, mean_s: float, win_rate: float | None, **kw) -> FrontierRow:
    values = dict(
        config_id=config_id, depth=4, sims=1024, batch=64, worlds=4,
        mean_s=mean_s, p95_s=mean_s * 1.2, max_s=mean_s * 1.5,
        encode_share=0.8, model_share=0.18, gate_pass=True,
        games=20 if win_rate is not None else None,
        wins=int(round(win_rate * 20)) if win_rate is not None else None,
        win_rate=win_rate,
    )
    values.update(kw)
    return FrontierRow(**values)


class WilsonTest(unittest.TestCase):
    def test_interval_contains_point_and_stays_in_range(self) -> None:
        lo, hi = wilson_interval(10, 20)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_extremes_do_not_escape_bounds(self) -> None:
        for wins in (0, 20):
            lo, hi = wilson_interval(wins, 20)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)

    def test_more_games_narrows_the_interval(self) -> None:
        narrow = wilson_interval(50, 100)
        wide = wilson_interval(10, 20)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_zero_games_is_maximally_uncertain(self) -> None:
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))


class ParityLanguageTest(unittest.TestCase):
    def test_never_claims_parity_achieved(self) -> None:
        labels = {
            parity_label(*wilson_interval(w, 20), w / 20) for w in range(21)
        }
        self.assertFalse(any("achieved" in label for label in labels))

    def test_clearly_below_requires_upper_bound_under_half(self) -> None:
        lo, hi = wilson_interval(2, 20)
        self.assertEqual(parity_label(lo, hi, 0.1), "clearly below parity")

    def test_even_split_is_parity_compatible(self) -> None:
        lo, hi = wilson_interval(10, 20)
        self.assertIn("parity-compatible", parity_label(lo, hi, 0.5))

    def test_screening_sweep_of_ten_of_twenty_is_not_above_parity(self) -> None:
        # 10/20 must not read as an improvement — the plan's central caution.
        lo, hi = wilson_interval(10, 20)
        self.assertNotEqual(parity_label(lo, hi, 0.5), "directionally above parity")


class FrontierMarkingTest(unittest.TestCase):
    def test_dominated_row_is_off_frontier(self) -> None:
        rows = mark_frontier([
            _row("fast-strong", 1.0, 0.60),
            _row("slow-weak", 5.0, 0.40),
        ])
        by_id = {r.config_id: r for r in rows}
        self.assertTrue(by_id["fast-strong"].on_frontier)
        self.assertFalse(by_id["slow-weak"].on_frontier)

    def test_slower_but_stronger_stays_on_frontier(self) -> None:
        rows = mark_frontier([
            _row("fast-weak", 1.0, 0.45),
            _row("slow-strong", 5.0, 0.70),
        ])
        self.assertTrue(all(r.on_frontier for r in rows))

    def test_unmeasured_row_is_not_treated_as_weak(self) -> None:
        # A config with no strength read must not be dominated into 'off frontier',
        # and must not dominate anything either.
        rows = mark_frontier([_row("measured", 2.0, 0.55), _row("untested", 1.0, None)])
        by_id = {r.config_id: r for r in rows}
        self.assertFalse(by_id["untested"].on_frontier)
        self.assertTrue(by_id["measured"].on_frontier)

    def test_gate_failing_row_cannot_dominate(self) -> None:
        rows = mark_frontier([
            _row("gate-fail", 0.5, 0.99, gate_pass=False),
            _row("eligible", 2.0, 0.55),
        ])
        by_id = {r.config_id: r for r in rows}
        self.assertTrue(by_id["eligible"].on_frontier)
        self.assertFalse(by_id["gate-fail"].on_frontier)


class BuildFrontierTest(unittest.TestCase):
    def test_joins_timing_and_strength_on_config_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            t, s = Path(temp) / "t", Path(temp) / "s"
            t.mkdir(); s.mkdir()
            (t / "timing-d4-s1024-b64-w4.json").write_text(json.dumps({
                "config": "d4-s1024-b64-w4", "depth": 4, "sims": 1024, "batch": 64,
                "worlds": 4, "n": 6, "mean_s": 2.19, "p95_s": 2.38, "max_s": 2.9,
                "encode_s": 8.0, "model_s": 1.8, "tree_s": 0.2, "gate_pass_15s": True,
            }))
            (t / "timing-d10-s8192-b64-w4.json").write_text(json.dumps({
                "config": "d10-s8192-b64-w4", "depth": 10, "sims": 8192, "batch": 64,
                "worlds": 4, "n": 6, "mean_s": 19.6, "p95_s": 20.7, "max_s": 22.0,
                "encode_s": 80.0, "model_s": 17.0, "tree_s": 2.0, "gate_pass_15s": False,
            }))
            (s / "strength-d4-s1024-b64-w4.json").write_text(json.dumps({
                "config": "d4-s1024-b64-w4", "games": 20, "wins": 11, "win_rate": 0.55,
            }))
            rows = build_frontier(t, s)
            by_id = {r.config_id: r for r in rows}
            self.assertEqual(by_id["d4-s1024-b64-w4"].wins, 11)
            self.assertIsNotNone(by_id["d4-s1024-b64-w4"].parity_label)
            # timing-only row survives the join without a fabricated strength value
            self.assertIsNone(by_id["d10-s8192-b64-w4"].win_rate)
            self.assertFalse(by_id["d10-s8192-b64-w4"].gate_pass)

    def test_errored_strength_row_is_ignored_not_scored_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            t, s = Path(temp) / "t", Path(temp) / "s"
            t.mkdir(); s.mkdir()
            (t / "timing-d4-s512-b64-w4.json").write_text(json.dumps({
                "config": "d4-s512-b64-w4", "depth": 4, "sims": 512, "batch": 64,
                "worlds": 4, "n": 6, "mean_s": 1.37, "p95_s": 1.5, "max_s": 1.8,
                "encode_s": 5.0, "model_s": 1.5, "tree_s": 0.1, "gate_pass_15s": True,
            }))
            (s / "strength-d4-s512-b64-w4.json").write_text(json.dumps({
                "config": "d4-s512-b64-w4", "error": "TypeError: boom",
            }))
            rows = build_frontier(t, s)
            self.assertIsNone(rows[0].win_rate)  # a crash is not a 0% win rate

    def test_markdown_renders_every_row(self) -> None:
        text = render_markdown([_row("a", 1.0, 0.6), _row("b", 2.0, None)])
        self.assertIn("| a |", text)
        self.assertIn("| b |", text)
        self.assertIn("—", text)  # unmeasured strength shown as absent


if __name__ == "__main__":
    unittest.main()
