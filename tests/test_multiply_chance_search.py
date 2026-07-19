"""Correctness gates for the multi-ply decision/chance search in the native
pokezero-search crate (rust/pokezero-search/src/tree.rs).

Mirrors tests/test_pokezero_search_crate.py conventions: every test skips
unless the built native module imports, so the default suite stays green on
checkouts that never built the crate. The Rust-side twin gates (analytic
expectation, probability conservation, determinism, deep-KO toggle) live in
the crate's `cargo test` suite; this file exercises the Python surface and
the regression against the one-ply core.

Chance contract under test (docs/test_time_search_plan_v3.md, search-tree
contract): chance nodes carry the engine's exact enumerated branch
probabilities and backup is exact expectation — for the analytic fixture
(gen3 toxic: 85% hit, 6 residual damage on a 100-max-HP Chansey; HpFraction
leaf eval) the root edge value is hand-computable:

    hit  = 0.5 + 0.5 * (1 - 94/100) = 0.53
    miss = 0.5
    E    = 0.85 * 0.53 + 0.15 * 0.5 = 0.5255
"""

from __future__ import annotations

import json
import unittest

try:  # pragma: no cover - exercised only when the native crate is built
    import pokezero_search
except ImportError:  # pragma: no cover
    pokezero_search = None  # type: ignore[assignment]

ITERATIONS = 2_000


def _build_state(side_one_moves, side_two_moves, *, s1_speed=200, s2_hp=100):
    from pokezero.poke_engine_adapter import (
        BattleSpec,
        MoveSpec,
        PokemonSpec,
        SideSpec,
        build_poke_engine_state,
    )

    def mon(species, moves, *, hp=100, maxhp=100, speed=100):
        return PokemonSpec(
            id=species,
            level=100,
            types=("normal",),
            hp=hp,
            maxhp=maxhp,
            attack=100,
            defense=100,
            special_attack=100,
            special_defense=100,
            speed=speed,
            status="none",
            moves=tuple(MoveSpec(id=m, pp=32) for m in moves),
        )

    spec = BattleSpec(
        side_one=SideSpec(pokemon=(mon("rattata", side_one_moves, speed=s1_speed),)),
        side_two=SideSpec(pokemon=(mon("chansey", side_two_moves, hp=s2_hp),)),
    )
    return build_poke_engine_state(spec).to_string()


def _q(report: dict, side: str, move: str) -> float:
    for entry in report[side]:
        if entry["move"] == move:
            return entry["q"]
    raise AssertionError(f"move {move!r} not among {side} arms: {report[side]}")


@unittest.skipIf(pokezero_search is None, "pokezero_search native module not built")
class MultiPlyChanceSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.analytic = _build_state(("toxic", "seismictoss"), ("splash",))
            self.depth_benefit = _build_state(("splash", "seismictoss"), ("splash",))
            self.straddle = _build_state(
                ("splash", "tackle"), ("splash", "tackle"), s2_hp=50
            )
        except Exception as exc:  # engine binding missing/broken
            self.skipTest(f"poke_engine fixture unavailable: {exc}")

    def search(self, state, **kwargs):
        defaults = dict(max_depth=2, c_puct=1.4, seed=0, deep_ko_split=True)
        defaults.update(kwargs)
        return json.loads(
            pokezero_search.puct_search_multi(state, ITERATIONS, **defaults)
        )

    # (a) analytic fixture: exact expectation over enumerated outcomes.
    def test_analytic_expectation_depth1(self) -> None:
        report = self.search(self.analytic, max_depth=1)
        hit = 0.5 + 0.5 * (1.0 - 94.0 / 100.0)
        expected = 0.85 * hit + 0.15 * 0.5  # = 0.5255
        self.assertAlmostEqual(_q(report, "side_one", "toxic"), expected, places=4)
        # Guaranteed KO: a single terminal branch of exact value 1.
        self.assertAlmostEqual(_q(report, "side_one", "seismictoss"), 1.0, places=6)
        self.assertEqual(report["side_one"][0]["move"], "seismictoss")
        self.assertEqual(report["decision_nodes"], 1)  # depth 1 never descends

    # (b) depth=1 reproduces the one-ply core's decision on the standard
    # fixture (same option surface, same argmax; the intended semantic change
    # — exact-expectation instead of sampled-branch backup — must not move
    # the decision).
    def test_depth1_matches_oneply_core(self) -> None:
        from pokezero.poke_engine_adapter import (
            build_poke_engine_state,
            minimal_gen3_fixture,
        )

        state = build_poke_engine_state(minimal_gen3_fixture()).to_string()
        one_ply = json.loads(pokezero_search.puct_search(state, ITERATIONS, 1.4, 0))
        multi = self.search(state, max_depth=1)
        for side in ("side_one", "side_two"):
            self.assertEqual(
                {e["move"] for e in one_ply[side]},
                {e["move"] for e in multi[side]},
            )
            # Report entries are visit-sorted: [0] is the argmax.
            self.assertEqual(one_ply[side][0]["move"], multi[side][0]["move"])
            self.assertEqual(
                sum(e["visits"] for e in multi[side]), ITERATIONS
            )
            for entry in multi[side]:
                self.assertGreaterEqual(entry["q"], 0.0)
                self.assertLessEqual(entry["q"], 1.0)

    # (c) determinism under a fixed seed.
    def test_deterministic_for_seed(self) -> None:
        first = self.search(self.straddle, max_depth=3, seed=17)
        second = self.search(self.straddle, max_depth=3, seed=17)
        self.assertEqual(first["side_one"], second["side_one"])
        self.assertEqual(first["side_two"], second["side_two"])
        self.assertEqual(first["chance_nodes"], second["chance_nodes"])
        self.assertEqual(first["decision_nodes"], second["decision_nodes"])

    # Depth benefit: a win exactly one ply past the root lifts the passive
    # arm at depth 2; depth 1 cannot see it.
    def test_depth2_sees_one_ply_ahead(self) -> None:
        shallow = self.search(self.depth_benefit, max_depth=1)
        self.assertAlmostEqual(_q(shallow, "side_one", "splash"), 0.5, places=4)
        deep = self.search(self.depth_benefit, max_depth=2)
        self.assertGreater(_q(deep, "side_one", "splash"), 0.8)

    # Deep KO-threshold splits past the engine's plies-1-2 damage horizon.
    def test_deep_ko_split_toggle(self) -> None:
        with_split = self.search(self.straddle, max_depth=3, deep_ko_split=True)
        self.assertGreater(with_split["deep_ko_triggers"], 0)
        without = self.search(self.straddle, max_depth=3, deep_ko_split=False)
        self.assertEqual(without["deep_ko_triggers"], 0)

    def test_report_counters_consistent(self) -> None:
        report = self.search(self.straddle, max_depth=3)
        self.assertEqual(report["iterations"], ITERATIONS)
        self.assertEqual(report["search"], "multi_ply")
        self.assertEqual(report["evaluator"], "hp_fraction")
        self.assertEqual(report["chance_nodes"], report["expansions"])
        self.assertLess(report["max_depth_reached"], report["max_depth"])
        self.assertGreater(report["leaf_evals"], 0)
        self.assertGreaterEqual(report["root_value"], 0.0)
        self.assertLessEqual(report["root_value"], 1.0)

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            pokezero_search.puct_search_multi(self.analytic, 0)
        with self.assertRaises(ValueError):
            pokezero_search.puct_search_multi(self.analytic, 10, max_depth=0)
        with self.assertRaises(ValueError):
            pokezero_search.puct_search_multi("garbage", 10)


if __name__ == "__main__":
    unittest.main()
