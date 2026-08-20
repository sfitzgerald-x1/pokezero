#!/usr/bin/env python3
"""Fail closed unless FoulPlay's own poke-engine wheel carries the Gen 3 patch stack.

The image has two independent ``poke_engine`` consumers: PokeZero's main
interpreter and FoulPlay's private venv.  Importing the latter is not enough:
the upstream 0.0.47 wheel imports successfully but its chance sampler panics
when a reconstructed sleeper's ordinary sleep counter is past the Gen 3
four-attempt bound.  This intentionally small fixture reaches that exact
chance arm through the installed Python binding.
"""

from __future__ import annotations

import argparse
from typing import Any


def exhausted_sleep_state(engine: Any) -> Any:
    """Build a legal state whose next MCTS expansion prices sleep attempt five."""

    fainted = engine.Pokemon(id="pikachu", level=1, hp=0)
    sleeper = engine.Pokemon(
        id="snorlax",
        level=80,
        types=("normal", "typeless"),
        hp=300,
        maxhp=300,
        ability="innerfocus",
        item="leftovers",
        attack=180,
        defense=180,
        special_attack=180,
        special_defense=180,
        speed=150,
        status="sleep",
        rest_turns=0,
        sleep_turns=5,
        moves=[engine.Move(id="sleeptalk", pp=16), engine.Move(id="rest", pp=16)],
    )
    opponent = engine.Pokemon(
        id="blissey",
        level=80,
        types=("normal", "typeless"),
        hp=300,
        maxhp=300,
        ability="innerfocus",
        item="leftovers",
        attack=180,
        defense=180,
        special_attack=180,
        special_defense=180,
        speed=100,
        moves=[engine.Move(id="tackle", pp=16), engine.Move(id="rest", pp=16)],
    )
    return engine.State(
        side_one=engine.Side(active_index="0", pokemon=[sleeper] + [fainted] * 5),
        side_two=engine.Side(active_index="0", pokemon=[opponent] + [fainted] * 5),
        weather="none",
        terrain="none",
        trick_room=False,
    )


def verify(engine: Any, *, iterations: int) -> None:
    """Exercise the binding and reject an incomplete rather than a crashed search."""

    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    result = engine.monte_carlo_tree_search(
        exhausted_sleep_state(engine), duration_ms=1, iterations=iterations, threads=1
    )
    # The Python binding's shipped MCTS applies its own 1,000-visit floor, so
    # ``total_visits`` need not equal the requested small smoke count.  A
    # positive total proves it reached the chance sampler rather than merely
    # constructing a result wrapper.
    if (
        not isinstance(result.total_visits, int)
        or isinstance(result.total_visits, bool)
        or result.total_visits <= 0
    ):
        raise RuntimeError(
            "FoulPlay poke-engine MCTS did not complete any visits: "
            f"got {result.total_visits!r}"
        )
    if not result.side_one or not result.side_two:
        raise RuntimeError("FoulPlay poke-engine MCTS returned an empty side result")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=128)
    args = parser.parse_args(argv)

    import poke_engine

    verify(poke_engine, iterations=args.iterations)
    print(f"FOULPLAY_PATCHED_POKE_ENGINE_OK iterations={args.iterations}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the image build
    raise SystemExit(main())
