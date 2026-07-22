"""Golden-corpus bit-exactness harness for schema-bound native encoders (track B).

Streams golden corpus rows, re-encodes each row's SANCTIONED input surface
through a pluggable backend, and diffs all five observation arrays
bit-exactly against the stored golden arrays. Emits:

- per-array exact-row counts;
- a per-token-block coverage table (field / self team / opponent team /
  action candidates / stats / transition) with exact-cell fractions for the
  2-D arrays — the honest coverage table for the track B PR;
- for mismatches, a per-(token, column) breakdown with got-vs-want examples
  (capped at --examples).

Backends (scripts/golden_encoder_backends.py):

- ``python-reference`` — the production Python encoder re-encoding from the
  stored row inputs. Run this FIRST: it validates the harness itself and
  measures exactly which golden columns the per-row corpus surface can
  reproduce (its residual mismatches are corpus input gaps, not bugs).
- ``rust`` — pokezero_search.encode_decision from rust/pokezero-search,
  fed the tables artifact from scripts/export_encoder_tables.py.

Usage:

    PYTHONPATH=src python scripts/validate_rust_encoder.py \
        --corpus corpus/golden-v1 --backend python-reference \
        [--rows N] [--examples 20] [--json report.json] \
        [--showdown-root <built-showdown>] [--tables encoder_tables.json]

Exit code: 0 when every row of every array matched bit-exactly, 1 otherwise
(2 for setup errors) — so CI can gate on full bit-exactness while humans
read the partial-coverage report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pokezero.golden_corpus import GOLDEN_ARRAY_FIELDS, load_golden_corpus  # noqa: E402

from golden_encoder_backends import (  # noqa: E402
    ARRAY_NAMES,
    PythonReferenceBackend,
    RustBackend,
    RustFoldBackend,
    row_inputs_from_decision_row,
)

# Token blocks of the v2.2 layout (docs/observation_v22_tokens.svg): the
# coverage table reports each block separately so partial encoders report
# honest per-block state.
TOKEN_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("field[0]", 0, 1),
    ("self_team[1-6]", 1, 7),
    ("opponent_team[7-12]", 7, 13),
    ("action_candidates[13-21]", 13, 22),
    ("stats[22]", 22, 23),
    ("transition[23-150]", 23, 151),
)


def _bitwise_equal_mask(got: numpy.ndarray, want: numpy.ndarray) -> numpy.ndarray:
    """Elementwise bit-equality (floats compared on their raw bit patterns)."""

    if got.shape != want.shape:
        raise ValueError(f"shape mismatch: got {got.shape}, want {want.shape}.")
    if got.dtype != want.dtype:
        raise ValueError(f"dtype mismatch: got {got.dtype}, want {want.dtype}.")
    if got.dtype.kind == "f":
        width = got.dtype.itemsize
        unsigned = numpy.dtype(f"<u{width}")
        return got.view(unsigned) == want.view(unsigned)
    return got == want


@dataclass
class ArrayReport:
    """Aggregated bit-exactness for one array across all diffed rows."""

    name: str
    rows_total: int = 0
    rows_exact: int = 0
    cells_total: int = 0
    cells_exact: int = 0
    # (token, column) -> mismatching row count; token=-1 for 1-D arrays.
    mismatch_positions: Counter = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def block_coverage(self) -> dict[str, dict[str, int]]:
        """Exact-cell counts per token block (2-D and per-token arrays only)."""

        if self.name == "legal_action_mask":
            return {}
        coverage: dict[str, dict[str, int]] = {}
        for label, start, stop in TOKEN_BLOCKS:
            mismatched = sum(
                count
                for (token, _column), count in self.mismatch_positions.items()
                if start <= token < stop
            )
            width = self.cells_total // 151 // max(1, self.rows_total) if self.rows_total else 0
            total = (stop - start) * width * self.rows_total
            coverage[label] = {
                "cells_total": total,
                "cells_exact": total - mismatched,
            }
        return coverage


def diff_row(
    row_index: int,
    got: Mapping[str, numpy.ndarray],
    want: Mapping[str, numpy.ndarray],
    reports: Mapping[str, ArrayReport],
    *,
    max_examples: int,
    context: Mapping[str, Any] | None = None,
) -> bool:
    """Diff one row's five arrays bit-exactly into ``reports``; True if all matched."""

    all_exact = True
    for name in ARRAY_NAMES:
        report = reports[name]
        got_array = numpy.ascontiguousarray(got[name])
        want_array = numpy.ascontiguousarray(want[name])
        equal = _bitwise_equal_mask(got_array, want_array)
        report.rows_total += 1
        report.cells_total += int(equal.size)
        exact_cells = int(equal.sum())
        report.cells_exact += exact_cells
        if exact_cells == equal.size:
            report.rows_exact += 1
            continue
        all_exact = False
        mismatch_indices = numpy.argwhere(~equal)
        for position in mismatch_indices:
            if want_array.ndim == 2:
                token, column = int(position[0]), int(position[1])
            else:
                token, column = int(position[0]), -1
                if name == "legal_action_mask":
                    token, column = -1, int(position[0])
            report.mismatch_positions[(token, column)] += 1
            if len(report.examples) < max_examples:
                index = tuple(int(v) for v in position)
                report.examples.append(
                    {
                        "row": row_index,
                        "token": token,
                        "column": column,
                        "got": got_array[index].item(),
                        "want": want_array[index].item(),
                        **(dict(context) if context else {}),
                    }
                )
    return all_exact


def build_report(reports: Mapping[str, ArrayReport], *, rows: int, backend: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": backend,
        "rows": rows,
        "arrays": {},
        "all_exact": all(r.rows_exact == r.rows_total for r in reports.values()),
    }
    for name, report in reports.items():
        top_positions = [
            {
                "token": token,
                "column": column,
                "mismatch_rows": count,
            }
            for (token, column), count in sorted(
                report.mismatch_positions.items(), key=lambda item: (-item[1], item[0])
            )[:40]
        ]
        payload["arrays"][name] = {
            "rows_exact": report.rows_exact,
            "rows_total": report.rows_total,
            "cells_exact": report.cells_exact,
            "cells_total": report.cells_total,
            "block_coverage": report.block_coverage(),
            "distinct_mismatch_positions": len(report.mismatch_positions),
            "top_mismatch_positions": top_positions,
            "examples": report.examples,
        }
    return payload


def render_text_report(payload: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        f"golden-corpus bit-exactness — backend={payload['backend']} rows={payload['rows']}"
    )
    lines.append("")
    lines.append(f"{'array':22} {'exact rows':>14} {'exact cells':>22}")
    for name, entry in payload["arrays"].items():
        rows = f"{entry['rows_exact']}/{entry['rows_total']}"
        cells = f"{entry['cells_exact']}/{entry['cells_total']}"
        lines.append(f"{name:22} {rows:>14} {cells:>22}")
    lines.append("")
    for name, entry in payload["arrays"].items():
        coverage = entry.get("block_coverage") or {}
        if not coverage or entry["rows_exact"] == entry["rows_total"]:
            continue
        lines.append(f"[{name}] per-block exact cells:")
        for label, cells in coverage.items():
            total = cells["cells_total"]
            exact = cells["cells_exact"]
            marker = "OK " if exact == total else "MIS"
            lines.append(f"  {marker} {label:26} {exact}/{total}")
        positions = entry.get("top_mismatch_positions") or []
        if positions:
            lines.append(f"[{name}] top mismatching (token, column) positions:")
            for item in positions[:12]:
                lines.append(
                    f"    token {item['token']:>3} column {item['column']:>3} — "
                    f"{item['mismatch_rows']} rows"
                )
        examples = entry.get("examples") or []
        if examples:
            lines.append(f"[{name}] examples (capped):")
            for example in examples[:8]:
                lines.append(
                    f"    row {example['row']} token {example['token']} column "
                    f"{example['column']}: got {example['got']!r} want {example['want']!r}"
                )
        lines.append("")
    lines.append("ALL EXACT" if payload["all_exact"] else "MISMATCHES PRESENT")
    return "\n".join(lines)


def _golden_arrays_dict(row: Any) -> dict[str, numpy.ndarray]:
    return {
        name: numpy.ascontiguousarray(getattr(row.arrays, name), dtype=dtype)
        for name, dtype, _ in GOLDEN_ARRAY_FIELDS
    }


def _default_showdown_root() -> str | None:
    return os.environ.get("POKEZERO_SHOWDOWN_ROOT")


def _compare_backends(
    corpus: Any,
    header: Mapping[str, Any],
    *,
    showdown_root: Path,
    tables_json: str,
    row_limit: int | None,
    example_cap: int,
    json_out: Path | None,
) -> int:
    """Direct rust-vs-python-reference byte diff over every row and array.

    This is the machine-checkable form of the cross-backend byte-identity
    claim: unlike the golden diff (which cannot exit 0 on a v1 corpus — the
    history cells are unreproducible by design), full parity here exits 0.
    """

    reference = PythonReferenceBackend(showdown_root=showdown_root, header=header)
    rust = RustBackend(tables_json=tables_json, header=header)
    reports = {name: ArrayReport(name=name) for name in ARRAY_NAMES}
    rows = corpus.decision_rows
    if row_limit is not None:
        rows = rows[:row_limit]
    for row_index, row in enumerate(rows):
        inputs = row_inputs_from_decision_row(row)
        diff_row(
            row_index,
            rust.encode(inputs),
            reference.encode(inputs),
            reports,
            max_examples=example_cap,
            context={
                "battle_seed": row.battle_seed,
                "player_id": row.player_id,
                "decision_round_index": row.decision_round_index,
            },
        )
    payload = build_report(reports, rows=len(rows), backend="rust-vs-python-reference")
    print(render_text_report(payload))
    if json_out is not None:
        json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if payload["all_exact"] else 1


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--corpus",
        type=Path,
        default=REPO_ROOT / "tests" / "data" / "golden_corpus_sample",
        help="Golden corpus directory (default: the committed 5-row sample).",
    )
    parser.add_argument(
        "--backend",
        choices=("python-reference", "rust", "rust-fold", "compare-backends"),
        required=True,
        help="Backend to diff against golden, or compare-backends for a direct "
        "rust-vs-python-reference byte diff (exit 0 on full parity — the "
        "reproducible form of the cross-backend byte-identity claim). "
        "rust-fold additionally feeds each row's recorded fold state (schema v2 "
        "sidecar) through the native in-crate product consumption — the FULL "
        "observation surface, expected ALL EXACT against golden.",
    )
    parser.add_argument("--rows", type=int, default=None, help="Only diff the first N rows.")
    parser.add_argument("--examples", type=int, default=20, help="Example cap per array.")
    parser.add_argument("--json", type=Path, default=None, help="Also write the JSON report here.")
    parser.add_argument(
        "--showdown-root",
        type=Path,
        default=_default_showdown_root(),
        help="Built Showdown checkout (python-reference backend; env POKEZERO_SHOWDOWN_ROOT).",
    )
    parser.add_argument(
        "--tables",
        type=Path,
        default=None,
        help="Encoder tables JSON from scripts/export_encoder_tables.py (rust backend).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    corpus = load_golden_corpus(args.corpus)
    header = corpus.header

    if args.backend == "compare-backends":
        if args.showdown_root is None or args.tables is None:
            print("compare-backends requires --showdown-root and --tables", file=sys.stderr)
            return 2
        return _compare_backends(
            corpus,
            header,
            showdown_root=args.showdown_root,
            tables_json=args.tables.read_text(encoding="utf-8"),
            row_limit=args.rows,
            example_cap=args.examples,
            json_out=args.json,
        )
    fold_states: dict[int, Any] = {}
    if args.backend == "python-reference":
        if args.showdown_root is None:
            print("--showdown-root (or POKEZERO_SHOWDOWN_ROOT) is required", file=sys.stderr)
            return 2
        backend = PythonReferenceBackend(showdown_root=args.showdown_root, header=header)
    elif args.backend == "rust-fold":
        if args.tables is None:
            print("--tables is required for the rust-fold backend", file=sys.stderr)
            return 2
        backend = RustFoldBackend(
            tables_json=args.tables.read_text(encoding="utf-8"), header=header
        )
        from pokezero.golden_corpus import GOLDEN_CORPUS_SCHEMA_VERSION  # noqa: E402
        from pokezero.golden_corpus_fold import iter_fold_records  # noqa: E402

        for record in iter_fold_records(
            args.corpus, expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION
        ):
            fold_states[int(record["array_row_index"])] = record["fold_state"]
    else:
        if args.tables is None:
            print("--tables is required for the rust backend", file=sys.stderr)
            return 2
        backend = RustBackend(tables_json=args.tables.read_text(encoding="utf-8"), header=header)

    reports = {name: ArrayReport(name=name) for name in ARRAY_NAMES}
    rows = corpus.decision_rows
    if args.rows is not None:
        rows = rows[: args.rows]
    for row_index, row in enumerate(rows):
        if args.backend == "rust-fold":
            fold_state = fold_states.get(row_index)
            if fold_state is None:
                print(f"no fold record for array row {row_index}", file=sys.stderr)
                return 2
            got = backend.encode_with_fold(row_inputs_from_decision_row(row), fold_state)
        else:
            got = backend.encode(row_inputs_from_decision_row(row))
        want = _golden_arrays_dict(row)
        diff_row(
            row_index,
            got,
            want,
            reports,
            max_examples=args.examples,
            context={
                "battle_seed": row.battle_seed,
                "player_id": row.player_id,
                "decision_round_index": row.decision_round_index,
            },
        )

    payload = build_report(reports, rows=len(rows), backend=backend.name)
    print(render_text_report(payload))
    if args.json is not None:
        args.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if payload["all_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
