"""Fold advance() throughput — rust (pokezero_search.FoldState) vs the Python
reference (pokezero.transitions_fold.FoldState) — over a corpus's real
boundary cases (track B perf note).

Per boundary the corpus supplies exactly what a search-time advance consumes:
the previous same-seat fold state and the inter-decision event slice. Each
measured operation is the per-chance-outcome unit from the search-tree
contract — "each chance-child advances its OWN copy of the fold state":

- ``python advance()``: the pure reference API (clone + fold + products build);
- ``rust clone+advance``: ``clone_state()`` + ``advance_in_place(slice)``
  (native fold, no products) — the in-search shape once products are consumed
  natively in the crate;
- ``rust clone+advance+products``: adds ``products_payload()`` (products built
  as PYTHON objects — an upper bound on native products cost, paid here only
  because the boundary crossing materializes them for Python).

Tier-2 overlay application is deliberately excluded: annotations come from the
live trackers at REAL boundaries only, never per simulated chance-outcome.

Usage:

    PYTHONPATH=src python scripts/bench_fold_advance.py \
        --corpus corpus/golden-v2 [--passes 5]

Each pass runs every boundary once; the best (minimum) pass is reported.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pokezero.golden_corpus import GOLDEN_CORPUS_SCHEMA_VERSION  # noqa: E402
from pokezero.golden_corpus_fold import iter_fold_records  # noqa: E402
from pokezero.transitions_fold import FoldState  # noqa: E402


def load_cases(corpus_dir: Path) -> list[tuple[dict[str, Any] | None, str, list[str]]]:
    """(previous same-seat fold-state payload | None, seat, event slice) per boundary."""

    cases: list[tuple[dict[str, Any] | None, str, list[str]]] = []
    previous_by_seat: dict[str, dict[str, Any]] = {}
    current_battle: str | None = None
    for record in iter_fold_records(
        corpus_dir, expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION
    ):
        battle_id = str(record["battle_id"])
        seat = str(record["player_id"])
        if battle_id != current_battle:
            current_battle = battle_id
            previous_by_seat = {}
        previous = previous_by_seat.get(seat)
        cases.append(
            (
                previous["fold_state"] if previous is not None else None,
                seat,
                [str(line) for line in record["event_slice"]],
            )
        )
        previous_by_seat[seat] = dict(record)
    return cases


def best_pass_seconds(run_once: Callable[[], None], passes: int) -> float:
    best = float("inf")
    for _ in range(passes):
        started = time.perf_counter()
        run_once()
        best = min(best, time.perf_counter() - started)
    return best


def report_line(label: str, seconds: float, boundaries: int) -> None:
    per_boundary_us = seconds / boundaries * 1e6
    print(
        f"{label:34s} {seconds * 1e3:9.1f} ms total   "
        f"{per_boundary_us:8.1f} us/boundary   {boundaries / seconds:10.0f} boundaries/s"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=5)
    args = parser.parse_args(argv)

    cases = load_cases(args.corpus)
    boundaries = len(cases)
    total_lines = sum(len(slice_) for _, _, slice_ in cases)
    print(
        f"corpus={args.corpus}  boundaries={boundaries}  "
        f"mean slice={total_lines / boundaries:.1f} lines  passes={args.passes}"
    )

    # Preload states once (payload decode excluded from the measured op).
    python_states: list[FoldState] = []
    slices: list[list[str]] = []
    for previous_payload, seat, slice_ in cases:
        if previous_payload is None:
            python_states.append(FoldState.initial(perspective_slot=seat))
        else:
            python_states.append(FoldState.from_payload(previous_payload))
        slices.append(slice_)

    def python_advance() -> None:
        for state, slice_ in zip(python_states, slices):
            state.advance(slice_)

    python_seconds = best_pass_seconds(python_advance, args.passes)
    report_line("python advance()", python_seconds, boundaries)

    try:
        import pokezero_search
    except (ImportError, OSError):  # pragma: no cover - environment guard
        print("pokezero_search not importable; rust measurements skipped.")
        return 1
    if not hasattr(pokezero_search, "FoldState"):
        print("installed pokezero_search has no FoldState; rust measurements skipped.")
        return 1

    rust_states = []
    for previous_payload, seat, _slice in cases:
        if previous_payload is None:
            rust_states.append(pokezero_search.FoldState.initial(seat))
        else:
            rust_states.append(pokezero_search.FoldState.from_payload(previous_payload))

    def rust_advance() -> None:
        for state, slice_ in zip(rust_states, slices):
            child = state.clone_state()
            child.advance_in_place(slice_)

    def rust_advance_products() -> None:
        for state, slice_ in zip(rust_states, slices):
            child = state.clone_state()
            child.advance_in_place(slice_)
            child.products_payload()

    rust_seconds = best_pass_seconds(rust_advance, args.passes)
    rust_products_seconds = best_pass_seconds(rust_advance_products, args.passes)
    report_line("rust clone+advance", rust_seconds, boundaries)
    report_line("rust clone+advance+products(py)", rust_products_seconds, boundaries)
    print(
        f"speedup vs python advance(): {python_seconds / rust_seconds:.1f}x (fold only), "
        f"{python_seconds / rust_products_seconds:.1f}x (incl. Python-object products)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
