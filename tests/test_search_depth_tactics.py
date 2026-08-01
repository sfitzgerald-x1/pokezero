"""Constructed-tactics depth suite: depth must WORK in the crate search tree.

Why this exists (docs/mcts_depth_tactics_findings.md): the n=400 dual grid
(docs/mcts_dual_grid_findings.md) measured FLAT strength-vs-depth ladders, but
at s1024 the sim budget starves the depth cap, so those ladders cannot say
whether depth works at all. This suite removes the game-distribution confound
with constructed positions carrying a PROVEN forced win that requires seeing
N plies ahead (proof = exhaustive enumeration over the engine's own
``generate_instructions`` branches; see ``scripts/depth_tactics_probe.py``):

- below the position's needed depth, the searcher must take the locally
  attractive trap arm (there is nothing in the tree to distinguish the win);
- at or above it, the forced line ends in exact terminal branches inside the
  horizon and the searcher must take the winning arm.

The suite pins the CONTROL arm (``puct_search_multi``, HP-fraction leaf, the
instrument docs/mcts_handcrafted_leaf_depth_findings.md validated): identical
tree/backup/depth mechanics to model mode with the leaf swapped. The model
leaf itself is exercised by the probe script against a real checkpoint (probe
artifacts under docs/audit_artifacts/depth-tactics-20260729/) — a checkpoint
is not a test dependency here.

Requires a Pokemon Showdown checkout + node (position materialization goes
through the real env boundary) and the gen3-patched poke-engine wheel;
skips otherwise, matching the other engine-backed suites.
"""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

try:
    import poke_engine
except ModuleNotFoundError:  # pragma: no cover
    poke_engine = None

try:
    import pokezero_search
except ModuleNotFoundError:  # pragma: no cover
    pokezero_search = None

# Slimmer than the probe's s4096: these trees are tiny (2-3 arms, single
# opponent action) and saturate long before this budget; the probe's full
# grid re-ran the suite at s4096 with identical argmaxes everywhere.
SIMS = 1024
SEED = 7


def _showdown_ready() -> bool:
    try:
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

        root = Path(DEFAULT_SHOWDOWN_ROOT)
    except Exception:  # pragma: no cover
        return False
    return (root / "dist" / "sim" / "index.js").exists() and shutil.which("node") is not None


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
@unittest.skipIf(pokezero_search is None, "pokezero_search crate not installed")
@unittest.skipUnless(_showdown_ready(), "Pokemon Showdown checkout or node unavailable")
class DepthTacticsSuite(unittest.TestCase):
    """Materialize every position once; solver facts + control-arm argmaxes."""

    @classmethod
    def setUpClass(cls) -> None:
        import depth_tactics_probe as probe
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy
        from pokezero.local_showdown import (
            DEFAULT_SHOWDOWN_ROOT,
            LocalShowdownConfig,
            LocalShowdownEnv,
        )
        from pokezero.randbat import load_gen3_randbat_source_cached

        cls.probe = probe
        cls.dex = load_showdown_dex_cached(str(DEFAULT_SHOWDOWN_ROOT))
        cls.env = LocalShowdownEnv(
            LocalShowdownConfig(showdown_root=str(DEFAULT_SHOWDOWN_ROOT), set_belief_source=True)
        )
        policy = EngineMctsPolicy(
            dex=cls.dex,
            set_source=load_gen3_randbat_source_cached(str(DEFAULT_SHOWDOWN_ROOT)),
            config=EngineMctsConfig(
                worlds=1, search_time_ms=10, leaf_eval="hp_fraction_crate",
                search_sims=SIMS, search_depth=2,
            ),
        )
        cls.states = {}
        for spec in probe.POSITIONS:
            context, override = probe.materialize_context(
                cls.env, spec, cls.dex, battle_id=f"suite-{spec.name}"
            )
            _world, state = probe.build_world_state(policy, context, override)
            cls.states[spec.name] = (spec, state)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "env"):
            cls.env.close()

    # ------------------------------------------------------------------
    # Design facts: the positions really are what they claim to be.
    # ------------------------------------------------------------------

    def test_win_moves_are_forced_wins_on_every_branch(self) -> None:
        """The winning arm reaches a WIN terminal on every enumerated branch
        within the proof horizon (control position: on every branch but Hyper
        Beam's 10% miss — the strongest claim its accuracy admits)."""
        for name, (spec, state) in self.states.items():
            with self.subTest(position=name):
                bounds = self.probe.solve_win_bounds(poke_engine, state, spec.proof_horizon)
                lower = bounds[spec.win_move]["p_win_lower"]
                floor = 0.9 if name == "immediate-ko-control" else 1.0
                self.assertGreaterEqual(
                    lower, floor,
                    f"{name}: {spec.win_move} p_win_lower {lower} < {floor}",
                )

    def test_trap_moves_win_at_most_the_crit_lottery(self) -> None:
        """The trap arm's achievable win probability is bounded by the crit
        lotteries the engine's own KO-split branching admits.

        perish-clock used to assert a strict zero here, justified as "its trap
        can never reach crit range". C27 showed that claim was overclaimed, and
        the correction is a proof erratum rather than a regression: the trap
        wins on FOUR consecutive crits, because three average crits leave
        Blissey on 164 and a fourth crit rolling 164-165 out of its 140-165
        range is lethal. The pre-C27 engine collapsed the crit arm to its
        average roll (152, four of which total 608 < 620), so it could not
        represent that line at all. The zero measured the missing branch, not
        the position. See scripts/depth_tactics_probe.py for the arithmetic.
        """
        for name, (spec, state) in self.states.items():
            if name == "immediate-ko-control":
                continue  # its "trap" is merely inferior, not a forced loss
            with self.subTest(position=name):
                bounds = self.probe.solve_win_bounds(poke_engine, state, spec.proof_horizon)
                upper = bounds[spec.trap_move]["p_win_upper"]
                self.assertLessEqual(
                    upper, 0.18,
                    f"{name}: trap {spec.trap_move} p_win_upper {upper}",
                )
                if name == "perish-clock":
                    # Bounded by four consecutive crits; strictly positive
                    # because that line is real and now representable.
                    self.assertGreater(
                        upper, 0.0,
                        "the four-crit line is reachable; a zero here means the "
                        "crit kill-split regressed",
                    )
                    self.assertLessEqual(
                        upper, (1.0 / 16.0) ** 4,
                        f"perish trap {upper} exceeds the four-crit lottery",
                    )

    def test_exact_horizon_argmax_flips_at_the_designed_depth(self) -> None:
        """The infinite-sample HP-leaf expectimax argmax flips to the winning
        move exactly at ``expected_needed_depth`` and stays there."""
        for name, (spec, state) in self.states.items():
            with self.subTest(position=name):
                table = self.probe.horizon_argmax_table(
                    poke_engine, state, spec.proof_horizon
                )
                flip = next(
                    (row["horizon"] for row in table if row["argmax"] == spec.win_move),
                    None,
                )
                self.assertEqual(flip, spec.expected_needed_depth, f"{name}: {table}")
                for row in table:
                    if row["horizon"] >= spec.expected_needed_depth:
                        self.assertEqual(row["argmax"], spec.win_move, f"{name}: {row}")
                    else:
                        self.assertNotEqual(row["argmax"], spec.win_move, f"{name}: {row}")

    # ------------------------------------------------------------------
    # The real search tree (control leaf): depth converts into the answer.
    # ------------------------------------------------------------------

    def _chosen(self, state, depth: int) -> str:
        import json

        report = json.loads(
            pokezero_search.puct_search_multi(
                state.to_string(), SIMS, max_depth=depth,
                c_puct=1.4, seed=SEED, deep_ko_split=True,
            )
        )
        return report["side_one"][0]["move"]

    def test_search_below_needed_depth_takes_the_trap(self) -> None:
        for name, (spec, state) in self.states.items():
            for depth in (d for d in (1, 2, 4) if d < spec.expected_needed_depth):
                with self.subTest(position=name, depth=depth):
                    self.assertNotEqual(
                        self._chosen(state, depth), spec.win_move,
                        f"{name}: d{depth} should not be able to see the win",
                    )

    def test_search_at_needed_depth_finds_the_forced_win(self) -> None:
        for name, (spec, state) in self.states.items():
            for depth in (d for d in (1, 2, 4, 6) if d >= spec.expected_needed_depth):
                with self.subTest(position=name, depth=depth):
                    self.assertEqual(
                        self._chosen(state, depth), spec.win_move,
                        f"{name}: d{depth} must find the forced win",
                    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
