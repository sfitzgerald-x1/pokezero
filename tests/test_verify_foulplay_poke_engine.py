"""Unit pins for the image-time FoulPlay poke-engine capability check."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_foulplay_poke_engine", ROOT / "scripts" / "verify_foulplay_poke_engine.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Record:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _Engine:
    Pokemon = _Record
    Move = _Record
    Side = _Record
    State = _Record

    def __init__(self, *, visits: int) -> None:
        self.visits = visits
        self.calls: list[tuple[object, int, int, int]] = []

    def monte_carlo_tree_search(
        self, state: object, duration_ms: int, iterations: int, threads: int
    ) -> object:
        self.calls.append((state, duration_ms, iterations, threads))
        return SimpleNamespace(
            total_visits=self.visits,
            side_one=("sleeptalk",),
            side_two=("tackle",),
        )


class FoulPlayPokeEngineVerifierTest(unittest.TestCase):
    def test_fixture_reaches_the_exhausted_ordinary_sleep_counter(self) -> None:
        engine = _Engine(visits=128)
        MODULE.verify(engine, iterations=128)

        state, duration_ms, iterations, threads = engine.calls[0]
        self.assertEqual((duration_ms, iterations, threads), (1, 128, 1))
        sleeper = state.side_one.pokemon[0]
        self.assertEqual((sleeper.status, sleeper.rest_turns, sleeper.sleep_turns), ("sleep", 0, 5))
        self.assertEqual([move.id for move in sleeper.moves], ["sleeptalk", "rest"])

    def test_empty_search_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not complete any visits"):
            MODULE.verify(_Engine(visits=0), iterations=128)

    def test_invalid_iteration_count_is_refused_before_search(self) -> None:
        engine = _Engine(visits=0)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    MODULE.verify(engine, iterations=value)
        self.assertEqual(engine.calls, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
