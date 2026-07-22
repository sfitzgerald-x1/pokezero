"""History-slot census — measure how many of the 128 transition-history slots a
decision actually fills, over a corpus of encoded observations.

This is the "usage-vs-fill" denominator for the history-truncation probe
(docs/history_truncation_probe_plan.md, caveat a): the plan's original census ran
on corpus games (near-random play); this tool re-runs it on any encoded-observation
source, so a trained-policy cache or a fresh trained-policy trajectory can be
measured the same way.

Fill is read straight from the attention mask: a transition slot is "used" when its
attention bit is set (True = attend), which the encoder sets iff the slot carries a
real turn/action token. No model, Showdown, or GPU is required — this only reads the
encoded arrays.

Sources (repeatable, mixed freely):
  --cache DIR    a training-cache directory (or a run's cache/ root); every
                 attention_mask.npy at any depth is scanned. Compact array-backed
                 caches store a prepended all-zero pad row (index 0) which is
                 skipped automatically (any all-False row is treated as padding).
  --npy FILE     a single attention_mask.npy (shape [..., token_count]).

Usage:
  python scripts/history_slot_census.py \
      --cache /shared/scott-experiment/<run>/cache \
      --out runs/history-slot-census-<date>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

try:  # Prefer the real layout constants; fall back to the documented defaults.
    from pokezero.showdown import TRANSITION_TOKEN_OFFSET
    from pokezero.observation import TRANSITION_TOKEN_COUNT
except Exception:  # pragma: no cover - keeps the census runnable without the package
    TRANSITION_TOKEN_OFFSET = 23
    TRANSITION_TOKEN_COUNT = 128


def _require_numpy():
    try:
        import numpy
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("history_slot_census requires numpy: pip install numpy") from exc
    return numpy


def _attention_mask_files(cache_dirs: Iterable[Path], npy_files: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in cache_dirs:
        if not directory.exists():
            raise SystemExit(f"cache dir does not exist: {directory}")
        files.extend(sorted(directory.rglob("attention_mask.npy")))
    for npy in npy_files:
        if not npy.exists():
            raise SystemExit(f"npy file does not exist: {npy}")
        files.append(npy)
    return files


def transition_fill_counts(attention_mask, numpy):
    """Per-row count of filled transition slots, dropping all-False padding rows.

    ``attention_mask`` is (..., token_count) bool; it is flattened to (rows, token_count).
    A row with no attended tokens at all is treated as a pad/placeholder and excluded.
    """
    array = numpy.asarray(attention_mask).astype(bool)
    array = array.reshape(-1, array.shape[-1])
    non_padding = array.any(axis=1)
    array = array[non_padding]
    start = TRANSITION_TOKEN_OFFSET
    stop = TRANSITION_TOKEN_OFFSET + TRANSITION_TOKEN_COUNT
    return array[:, start:stop].sum(axis=1)


def _summary(counts, numpy) -> dict:
    if counts.size == 0:
        raise SystemExit("no non-padding rows found — is this an encoded-observation source?")
    counts = counts.astype(numpy.int64)
    return {
        "n": int(counts.size),
        "transition_slots": int(TRANSITION_TOKEN_COUNT),
        "transition_offset": int(TRANSITION_TOKEN_OFFSET),
        "mean": round(float(counts.mean()), 2),
        "median": float(numpy.median(counts)),
        "p90": float(numpy.percentile(counts, 90)),
        "p99": float(numpy.percentile(counts, 99)),
        "max": int(counts.max()),
        "pct_gt_16": round(100.0 * float((counts > 16).mean()), 1),
        "pct_gt_32": round(100.0 * float((counts > 32).mean()), 1),
        "pct_gt_64": round(100.0 * float((counts > 64).mean()), 1),
    }


def _print_table(summary: dict) -> None:
    print(f"history-slot census (n={summary['n']:,}, of {summary['transition_slots']} slots)")
    print(f"  mean / median      : {summary['mean']} / {summary['median']}")
    print(f"  p90 / p99 / max    : {summary['p90']} / {summary['p99']} / {summary['max']}")
    print(f"  decisions >16 / >32 / >64 : "
          f"{summary['pct_gt_16']}% / {summary['pct_gt_32']}% / {summary['pct_gt_64']}%")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, action="append", default=[],
                        help="Training-cache dir (or run cache/ root); recurses for attention_mask.npy. Repeatable.")
    parser.add_argument("--npy", type=Path, action="append", default=[],
                        help="A single attention_mask.npy file. Repeatable.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Optional cap on total rows scanned (for a quick sample).")
    args = parser.parse_args(argv)

    numpy = _require_numpy()
    files = _attention_mask_files(args.cache, args.npy)
    if not files:
        raise SystemExit("no attention_mask.npy sources found (pass --cache or --npy).")

    chunks = []
    total = 0
    for path in files:
        array = numpy.load(path, mmap_mode="r", allow_pickle=False)
        counts = transition_fill_counts(array, numpy)
        chunks.append(counts)
        total += int(counts.size)
        if args.max_rows is not None and total >= args.max_rows:
            break
    counts = numpy.concatenate(chunks) if chunks else numpy.asarray([], dtype=numpy.int64)
    if args.max_rows is not None:
        counts = counts[: args.max_rows]

    summary = _summary(counts, numpy)
    summary["sources"] = [str(path) for path in files]
    _print_table(summary)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"history_slot_census: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
