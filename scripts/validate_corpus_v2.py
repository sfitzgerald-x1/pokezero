"""Schema-v2 row-pair validation harness for the golden corpus (track B).

Streams the corpus's fold sidecar (``fold.jsonl.gz``) and validates the
fold-state ADVANCE contract row-pair by row-pair, per (game, seat) chain:

- chain start: ``backend.start(...)`` advanced over row 0's event slice (+ its
  annotation overlay) must reproduce row 0's recorded fold state AND products;
- every consecutive same-seat pair: ``backend.load(row_n.fold_state)`` advanced
  over row_{n+1}'s slice (+ overlay) must reproduce row_{n+1}'s recorded state
  and products — each pair independently, exactly the transition a search-time
  advance performs.

Comparison is canonical-JSON byte equality on both payloads (dataclass equality
is implied: the payload codecs are total over the dataclasses).

Backends (the seam the Rust ``advance()`` port plugs into, mirroring
``scripts/validate_rust_encoder.py``):

- ``python-reference`` — the production incremental fold
  (``pokezero.transitions_fold.FoldState``). Run this FIRST: it validates the
  corpus itself (and re-proves fold closure over the recorded slices).
- a future ``rust`` backend implements the same three-method protocol
  (``pokezero.golden_corpus_fold.FoldBackend``: start / load / step over
  payload dicts) and registers in ``BACKENDS`` below — no harness changes.

Usage:

    PYTHONPATH=src python scripts/validate_corpus_v2.py \
        --corpus corpus/golden-v2 [--backend python-reference] \
        [--verify-corpus] [--json report.json]

Exit code: 0 when every boundary of every chain validated byte-exactly,
1 otherwise (2 for setup errors).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pokezero.golden_corpus import GOLDEN_CORPUS_SCHEMA_VERSION, verify_golden_corpus  # noqa: E402
from pokezero.golden_corpus_fold import (  # noqa: E402
    FoldChainValidationReport,
    PythonReferenceFoldBackend,
    validate_fold_chains,
)

from golden_fold_backends import CompareFoldBackend, RustFoldBackend  # noqa: E402

# Backend registry: name -> zero-argument factory. ``rust`` is the native
# crate's advance (pokezero_search.FoldState, rust/pokezero-search src/fold.rs);
# ``compare-backends`` runs rust and python-reference side by side and prints
# JSON-path locators for any divergence between the two (returning the rust
# outputs, so the corpus comparison stays the rust gate).
BACKENDS = {
    "python-reference": PythonReferenceFoldBackend,
    "rust": RustFoldBackend,
    "compare-backends": CompareFoldBackend,
}


def render_text_report(report: FoldChainValidationReport, *, corpus: str, seconds: float) -> str:
    lines = [
        f"schema-v2 fold row-pair validation — backend={report.backend} corpus={corpus}",
        "",
        f"games                {report.games}",
        f"chains (game x seat) {report.chains}",
        f"boundaries validated {report.rows_validated} "
        f"(chain starts: {report.initial_validations}, row pairs: {report.pair_validations})",
        f"row split            random={report.random_rows} scenario={report.scenario_rows}",
        f"wall time            {seconds:.1f}s",
        "",
    ]
    if report.ok:
        lines.append("ALL BOUNDARIES VALIDATED (state + products byte-exact)")
    else:
        lines.append(f"MISMATCHES PRESENT: {report.mismatch_total}")
        for mismatch in report.mismatches:
            lines.append(
                f"  {mismatch.battle_id} {mismatch.player_id} chain#{mismatch.chain_index} "
                f"[{mismatch.surface}] {mismatch.detail}"
            )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        required=True,
        help="Golden corpus directory with a fold sidecar (repeatable).",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default="python-reference",
        help="Advance implementation under test (default: python-reference).",
    )
    parser.add_argument(
        "--verify-corpus",
        action="store_true",
        help="Run full corpus verification (file hashes, row hashes, fold links) first.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Also write the JSON report(s) here.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    backend_factory = BACKENDS[args.backend]
    payloads = []
    all_ok = True
    for corpus_dir in args.corpus:
        if not (corpus_dir / "fold.jsonl.gz").exists():
            print(f"{corpus_dir}: no fold sidecar (schema v2 required)", file=sys.stderr)
            return 2
        if args.verify_corpus:
            verification = verify_golden_corpus(corpus_dir)
            print(
                f"{corpus_dir}: corpus verified — {verification.games} games, "
                f"{verification.decisions} decisions, {verification.fold_rows} fold rows"
            )
        started = time.perf_counter()
        report = validate_fold_chains(
            corpus_dir,
            backend_factory(),
            expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION,
        )
        seconds = time.perf_counter() - started
        print(render_text_report(report, corpus=str(corpus_dir), seconds=seconds))
        print()
        payload = report.to_json_dict()
        payload["corpus"] = str(corpus_dir)
        payload["seconds"] = round(seconds, 3)
        payloads.append(payload)
        all_ok = all_ok and report.ok

    if args.json is not None:
        args.json.write_text(
            json.dumps(payloads, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
