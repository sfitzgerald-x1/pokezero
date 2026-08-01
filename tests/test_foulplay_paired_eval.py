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
        foulplay_root=None,
        foulplay_python=None,
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
        # Checkpoint-qualified: cells A (k0) and G (k1) run the SAME search
        # config, so an unqualified id would merge them -- and cell G's entire
        # job is the checkpoint contrast.
        self.assertEqual(_DRIVER.config_id_for(args()), "d4-s1024-b64-w4@ckpt")

    def test_depth_ladder_cells_are_distinguishable(self) -> None:
        # The Tier-2 ladder differs from Tier 1 only in depth/sims; if those did
        # not reach config_id, a d6 shard would merge into the d4 cell.
        self.assertEqual(
            _DRIVER.config_id_for(args(depth=6, sims=4096)), "d6-s4096-b64-w4@ckpt"
        )

    def test_raw_arm_is_search_config_independent_but_checkpoint_qualified(self) -> None:
        # One raw arm PER CHECKPOINT pairs with every search cell on that
        # checkpoint, so its id must not carry search axes (or it cannot be
        # reused) but must carry the checkpoint (or R0 and R1 pool, and the raw
        # arm is the denominator of every paired delta).
        self.assertEqual(_DRIVER.config_id_for(args(arm="raw")), "raw@ckpt")
        self.assertEqual(
            _DRIVER.config_id_for(args(arm="raw", depth=6, worlds=16)), "raw@ckpt"
        )
        self.assertNotEqual(
            _DRIVER.config_id_for(args(arm="raw", checkpoint="/c/k0.pt")),
            _DRIVER.config_id_for(args(arm="raw", checkpoint="/c/k1.pt")),
        )


class FoulplayPathTest(unittest.TestCase):
    """The driver must be able to point the bridge at a relocated foul-play.

    The bridge defaults to a repo-relative checkout and has no environment
    fallback, so a deployment that ships foul-play elsewhere cannot reach it. A
    whole campaign probe died on that: every shard raised FileNotFoundError for
    .../third_party/foul-play/.venv/bin/python while the image had it at
    /opt/foul-play.
    """

    def test_paths_are_forwarded_when_given(self) -> None:
        argv = _DRIVER.bridge_argv(
            args(foulplay_root="/opt/foul-play",
                 foulplay_python="/opt/foul-play/.venv/bin/python"),
            seat="p1",
        )
        self.assertEqual(argv[argv.index("--foulplay-root") + 1], "/opt/foul-play")
        self.assertEqual(
            argv[argv.index("--foulplay-python") + 1], "/opt/foul-play/.venv/bin/python"
        )

    def test_omitted_paths_leave_the_bridge_default_alone(self) -> None:
        # Passing an empty value would override the bridge default with nothing.
        argv = _DRIVER.bridge_argv(args(), seat="p1")
        self.assertNotIn("--foulplay-root", argv)
        self.assertNotIn("--foulplay-python", argv)

    def test_paths_are_forwarded_on_both_arms_and_seats(self) -> None:
        # The raw arm drives the same opponent, so it needs the same path.
        for arm in ("search", "raw"):
            for seat in ("p1", "p2"):
                with self.subTest(arm=arm, seat=seat):
                    argv = _DRIVER.bridge_argv(
                        args(arm=arm, foulplay_root="/opt/foul-play"), seat=seat
                    )
                    self.assertIn("--foulplay-root", argv)


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


class OpponentPriorsRefusalTest(unittest.TestCase):
    """Cells B/E are refused until the opponent map's ordering is verified.

    The switch-ordering defect is fixed (the request order is computed from the
    opponent's switch history and passed through ctx), but four prior attempts
    each looked correct under their own tests, the fix has not cleared
    independent review, and nothing has run against a real checkpoint. A wrong
    mapping does not fail -- it reports a confident paired delta from permuted
    priors, and the campaign reads that as "opponent priors do not help".
    """

    def test_opponent_priors_are_refused_not_silently_run(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            _DRIVER.main([
                "--checkpoint", "/tmp/c.pt", "--showdown-root", "/tmp/s",
                "--arm", "search", "--seed-start", "1", "--pairs", "2",
                "--out", "/tmp/o.json", "--opponent-priors", "--skip-build-check",
            ])
        message = str(caught.exception)
        self.assertIn("REFUSED", message)
        # The refusal must say what would lift it, not merely that it refused.
        self.assertIn("review", message)

    def test_the_default_path_is_unaffected(self) -> None:
        # The refusal must not touch the nine cells that do not use the flag.
        argv = _DRIVER.bridge_argv(args(), seat="p1")
        self.assertNotIn("--engine-opponent-priors", argv)
        self.assertEqual(_DRIVER.config_id_for(args()), "d4-s1024-b64-w4@ckpt")

    def test_raw_arm_never_carries_the_flag(self) -> None:
        argv = _DRIVER.bridge_argv(args(arm="raw", opponent_priors=True), seat="p1")
        self.assertNotIn("--engine-opponent-priors", argv)


class NoDuplicateDefinitionsTest(unittest.TestCase):
    """Catches an editing accident that Python silently tolerates.

    A bad amend spliced a second copy of this module's helpers and `main` into
    the file. Python keeps the LAST definition, so the module still imported and
    every test passed -- while the surviving `main` was the stale one and the
    corrected code sat above it as dead lines. The commit message reported the
    fix as landed.

    There is no linter configured in this repo and no lint job in CI, so an
    F811 would not be reported by anything. This asserts it directly, for the
    two campaign scripts whose `main` is the actual entry point the launcher
    shells out to.
    """

    def _duplicate_defs(self, relative):
        import ast
        from collections import Counter

        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
        names = [
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        return {name: n for name, n in Counter(names).items() if n > 1}

    def test_paired_eval_defines_each_top_level_name_once(self) -> None:
        self.assertEqual(self._duplicate_defs("scripts/foulplay_paired_eval.py"), {})

    def test_power_report_defines_each_top_level_name_once(self) -> None:
        self.assertEqual(self._duplicate_defs("scripts/foulplay_power_report.py"), {})

    def test_the_refusal_comment_agrees_with_the_refusal(self) -> None:
        # The spliced copy left a comment saying the ordering defect was live
        # two lines above a SystemExit saying it was fixed. Whichever way that
        # is resolved, they must not contradict each other.
        source = (REPO_ROOT / "scripts" / "foulplay_paired_eval.py").read_text(
            encoding="utf-8"
        )
        body = source[source.index("def main("):]
        guard = body[: body.index("if args.opponent_priors:")]
        self.assertNotIn("second switch onward", guard,
                         "stale round-4 rationale is back in the live guard")
        self.assertIn("CRATE-SIDE GATHER", guard)


if __name__ == "__main__":
    unittest.main()
