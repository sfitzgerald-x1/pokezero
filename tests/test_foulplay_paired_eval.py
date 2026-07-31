"""Gates for the seed-paired FoulPlay driver.

The driver's job is to make the paired delta (search - raw) scoreable, so the
things pinned here are the ones whose failure would produce a plausible NUMBER
rather than an error:

* the cell identity a shard is merged under (`config_id`);
* the seed join, which must fail rather than mis-align;
* the score key, which must fail rather than read every game as a loss;
* the opponent-priors label, which must match what the bridge actually ran;
* the opponent definition (FoulPlay's own search budget) and the thread pin,
  both of which silently change opponent strength if they drift.

Pure functions only -- no bridge subprocess, no checkpoint, no cluster.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "foulplay_paired_eval_test", REPO_ROOT / "scripts" / "foulplay_paired_eval.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_DRIVER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DRIVER)


def args(**overrides) -> argparse.Namespace:
    base = dict(
        checkpoint="/tmp/ckpt.pt",
        showdown_root="/tmp/showdown",
        device="cuda",
        arm="search",
        seed_start=7800000,
        pairs=200,
        depth=4,
        sims=1024,
        batch=64,
        worlds=4,
        opponent_priors=False,
        engine_model_path=None,
        engine_tables_path=None,
        out="/tmp/shard.json",
        skip_build_check=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def game(seed: int, *, score: float = 1.0, won: bool = True) -> dict:
    return {
        "seed": seed,
        "pokezero_won": won,
        "pokezero_score": score,
        "tied": False,
        "capped": False,
    }


class ConfigIdTest(unittest.TestCase):
    def test_search_cell_id_matches_the_campaign_grid(self) -> None:
        self.assertEqual(_DRIVER.config_id_for(args()), "d4-s1024-b64-w4")

    def test_depth_ladder_cells_are_distinguishable(self) -> None:
        # The Tier-2 ladder differs from Tier 1 only in depth/sims; if those did
        # not reach config_id, a d6 shard would merge into the d4 cell.
        self.assertEqual(
            _DRIVER.config_id_for(args(depth=6, sims=4096)), "d6-s4096-b64-w4"
        )

    def test_raw_arm_is_search_config_independent(self) -> None:
        # One raw arm per checkpoint pairs with every search cell, so its id must
        # NOT carry search axes -- otherwise it cannot be reused.
        self.assertEqual(_DRIVER.config_id_for(args(arm="raw")), "raw")
        self.assertEqual(
            _DRIVER.config_id_for(args(arm="raw", depth=6, worlds=16)), "raw"
        )


class OpponentDefinitionTest(unittest.TestCase):
    def test_foulplay_budget_is_pinned_across_every_arm(self) -> None:
        # Part of the opponent definition, not a tuning knob: an arm that faced a
        # different FoulPlay budget is measuring a different opponent.
        self.assertEqual(_DRIVER.FOULPLAY_SEARCH_TIME_MS, 1000)
        for arm in ("search", "raw"):
            with self.subTest(arm=arm):
                argv = _DRIVER.bridge_argv(args(arm=arm), seat="p1")
                self.assertIn("--search-time-ms", argv)
                self.assertEqual(argv[argv.index("--search-time-ms") + 1], "1000")

    def test_thread_pin_is_the_full_family_set(self) -> None:
        # The July-30 jobs used OMP/MKL=2; unpinned BLAS in a CPU-capped pod
        # weakens the FoulPlay side specifically.
        self.assertEqual(
            set(_DRIVER.THREAD_PIN_ENV),
            {
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "POKEZERO_TORCH_NUM_THREADS", "POKEZERO_TORCH_NUM_INTEROP_THREADS",
                "OMP_DYNAMIC",
            },
        )
        self.assertEqual(_DRIVER.THREAD_PIN_ENV["OMP_NUM_THREADS"], "1")


class BridgeArgvTest(unittest.TestCase):
    def test_search_arm_passes_every_engine_axis(self) -> None:
        argv = _DRIVER.bridge_argv(args(depth=6, sims=2048, worlds=8), seat="p2")
        self.assertEqual(argv[argv.index("--policy-mode") + 1], "engine-mcts")
        self.assertEqual(argv[argv.index("--engine-depth") + 1], "6")
        self.assertEqual(argv[argv.index("--engine-sims") + 1], "2048")
        self.assertEqual(argv[argv.index("--engine-worlds") + 1], "8")
        self.assertEqual(argv[argv.index("--pokezero-player") + 1], "p2")

    def test_raw_arm_passes_no_engine_axes(self) -> None:
        argv = _DRIVER.bridge_argv(args(arm="raw"), seat="p1")
        self.assertEqual(argv[argv.index("--policy-mode") + 1], "raw")
        for flag in ("--engine-depth", "--engine-sims", "--engine-worlds"):
            self.assertNotIn(flag, argv)

    def test_both_seats_share_one_seed_band(self) -> None:
        # Within-seed pairing: the seats must face the SAME seeds, or the seat
        # comparison silently becomes a team comparison.
        p1 = _DRIVER.bridge_argv(args(), seat="p1")
        p2 = _DRIVER.bridge_argv(args(), seat="p2")
        self.assertEqual(
            p1[p1.index("--seed-start") + 1], p2[p2.index("--seed-start") + 1]
        )
        self.assertEqual(p1[p1.index("--games") + 1], p2[p2.index("--games") + 1])


class SeedJoinTest(unittest.TestCase):
    def test_rows_are_keyed_by_seed_not_position(self) -> None:
        summary = {"game_results": [game(7800002), game(7800000), game(7800001)]}
        rows = _DRIVER.per_seed_outcomes(summary, "p1")
        self.assertEqual(sorted(rows), [7800000, 7800001, 7800002])
        self.assertEqual(rows[7800002]["seed"], 7800002)

    def test_missing_score_key_is_fatal_not_a_zero(self) -> None:
        # The regression that matters: defaulting this to 0.0 would read as a
        # perfect-loss arm, i.e. a huge and entirely fake paired delta.
        broken = {"game_results": [{"seed": 1, "pokezero_won": True}]}
        with self.assertRaises(SystemExit) as caught:
            _DRIVER.per_seed_outcomes(broken, "p1")
        self.assertIn("pokezero_score", str(caught.exception))

    def test_scores_are_read_through_verbatim(self) -> None:
        summary = {"game_results": [game(1, score=0.5, won=False)]}
        self.assertEqual(_DRIVER.per_seed_outcomes(summary, "p1")[1]["score"], 0.5)


class SeatBlockTest(unittest.TestCase):
    def test_latency_gate_field_is_surfaced_separately_from_policy_timing(self) -> None:
        summary = {
            "completed_games": 200,
            "engine_mcts": {
                "search_wall_per_searched_decision": 4.2,
                "fallback_rate": 0.008,
                "policy_stats": {"depth_reached_mean": 3.1, "depth_reached_max": 4},
            },
            "policy_timing": {"average_elapsed_seconds": 3.8, "p95_elapsed_seconds": 6.2},
        }
        block = _DRIVER.seat_block(summary, "p1")
        # Both walls present and NOT conflated: the gate reads the first.
        self.assertEqual(block["search_wall_per_searched_decision"], 4.2)
        self.assertEqual(block["wall_per_decision_mean"], 3.8)
        self.assertEqual(block["wall_per_decision_p95"], 6.2)
        self.assertEqual(block["depth_reached_mean"], 3.1)

    def test_raw_arm_block_survives_absent_engine_telemetry(self) -> None:
        block = _DRIVER.seat_block({"completed_games": 200, "wins": 90}, "p2")
        self.assertIsNone(block["search_wall_per_searched_decision"])
        self.assertIsNone(block["depth_reached_mean"])
        self.assertEqual(block["games"], 200)


class OpponentPriorsLabelTest(unittest.TestCase):
    def test_flag_reaches_the_bridge_when_the_cell_claims_it(self) -> None:
        # The label and the behaviour must not drift apart: a shard tagged
        # '+opp-priors' whose bridge ran uniform opponent priors is worse than
        # a missing cell, because cells B and E are read against that label.
        argv = _DRIVER.bridge_argv(args(opponent_priors=True), seat="p1")
        self.assertIn("--engine-opponent-priors", argv)

    def test_flag_is_absent_by_default(self) -> None:
        argv = _DRIVER.bridge_argv(args(), seat="p1")
        self.assertNotIn("--engine-opponent-priors", argv)

    def test_raw_arm_never_carries_the_flag(self) -> None:
        # Raw is the pairing partner and is search-config-independent; an
        # opponent-priors raw arm would not be reusable across cells.
        argv = _DRIVER.bridge_argv(args(arm="raw", opponent_priors=True), seat="p1")
        self.assertNotIn("--engine-opponent-priors", argv)


if __name__ == "__main__":
    unittest.main()
