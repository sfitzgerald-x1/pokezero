"""Smoke tests for the native pokezero-search crate (rust/pokezero-search).

The crate is an optional native extension: built via maturin from vendored
poke-engine source (scripts/vendor_poke_engine_src.sh). Every test here skips
unless the built module imports, so the default Python test suite stays green
on checkouts that never built the crate.
"""

from __future__ import annotations

import json
import unittest

try:  # pragma: no cover - exercised only when the native crate is built
    import pokezero_search
except ImportError:  # pragma: no cover
    pokezero_search = None  # type: ignore[assignment]

BENCH_ITERATIONS = 20_000
PUCT_ITERATIONS = 5_000


def _fixture_state_str() -> str:
    from pokezero.poke_engine_adapter import (
        build_poke_engine_state,
        minimal_gen3_fixture,
    )

    return build_poke_engine_state(minimal_gen3_fixture()).to_string()


@unittest.skipIf(pokezero_search is None, "pokezero_search native module not built")
class PokezeroSearchCrateSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.state_str = _fixture_state_str()
        except Exception as exc:  # engine binding missing/broken
            self.skipTest(f"poke_engine fixture unavailable: {exc}")

    def test_module_metadata(self) -> None:
        self.assertTrue(pokezero_search.__version__)
        self.assertIn("gen3", pokezero_search.ENGINE_FEATURES)

    def test_bench_apply_reverse_returns_positive_rate(self) -> None:
        rate = pokezero_search.bench_apply_reverse(
            self.state_str, "ember", "watergun", BENCH_ITERATIONS
        )
        self.assertGreater(rate, 0.0)
        # Regime check, deliberately loose: the native loop must clear the
        # Python-FFI ceiling (~33-46k steps/s per docs/engine_search_poc.md)
        # by a wide margin on the minimal fixture. Measured ~0.5M steps/s on
        # an M-series laptop; 100k leaves room for slow CI hardware.
        self.assertGreater(rate, 100_000.0)

    def test_bench_apply_reverse_branching_toggle(self) -> None:
        branching = pokezero_search.bench_apply_reverse(
            self.state_str, "ember", "watergun", BENCH_ITERATIONS, branch_on_damage=True
        )
        non_branching = pokezero_search.bench_apply_reverse(
            self.state_str, "ember", "watergun", BENCH_ITERATIONS, branch_on_damage=False
        )
        self.assertGreater(branching, 0.0)
        self.assertGreater(non_branching, 0.0)

    def test_bench_apply_reverse_rejects_bad_move(self) -> None:
        with self.assertRaises(ValueError):
            pokezero_search.bench_apply_reverse(
                self.state_str, "notamove", "watergun", 10
            )

    def test_puct_search_visit_report(self) -> None:
        report = json.loads(
            pokezero_search.puct_search(self.state_str, PUCT_ITERATIONS)
        )
        self.assertEqual(report["iterations"], PUCT_ITERATIONS)
        self.assertEqual(report["evaluator"], "hp_fraction")
        for side in ("side_one", "side_two"):
            entries = report[side]
            self.assertTrue(entries, f"{side} has no root options")
            self.assertEqual(
                sum(entry["visits"] for entry in entries), PUCT_ITERATIONS
            )
            for entry in entries:
                self.assertIn("move", entry)
                self.assertGreaterEqual(entry["q"], 0.0)
                self.assertLessEqual(entry["q"], 1.0)
        # The fixture's root moves must be visible as root options.
        side_one_moves = {entry["move"] for entry in report["side_one"]}
        self.assertIn("ember", side_one_moves)

    def test_puct_search_deterministic_for_seed(self) -> None:
        # Timing fields differ between runs; the search result must not.
        first = json.loads(pokezero_search.puct_search(self.state_str, 2_000, seed=7))
        second = json.loads(pokezero_search.puct_search(self.state_str, 2_000, seed=7))
        self.assertEqual(first["side_one"], second["side_one"])
        self.assertEqual(first["side_two"], second["side_two"])

class MalformedStateTests(unittest.TestCase):
    def test_malformed_state_raises_value_error_not_panic(self) -> None:
        try:
            import pokezero_search as crate
        except ImportError:
            self.skipTest("pokezero_search crate not built")
        for bad in ("", "garbage", "NONE,100"):
            with self.assertRaises(ValueError):
                crate.bench_apply_reverse(bad, "tackle", "tackle", 10)
            with self.assertRaises(ValueError):
                crate.puct_search(bad, 10)


if __name__ == "__main__":
    unittest.main()
