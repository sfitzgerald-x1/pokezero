"""Tests for the golden-corpus bit-exactness harness (track B).

Two layers:

- Unit tests for the harness diff logic (bitwise float comparison, per-array
  row/cell accounting, per-token-block attribution, example capping) on
  synthetic arrays — no Showdown checkout required.
- A gated end-to-end test running the committed 5-row golden sample through
  the available encoder backends (python-reference needs node + a built
  Showdown checkout; rust additionally needs the pokezero_search wheel with
  encode_decision). Both backends must sit exactly at the documented
  stored-surface ceiling: every mismatch confined to the transition block
  (23..150) and the history-derived columns (tendency triple 63..65, pinned
  Tier-2 138/139, stats-token counters 92..104) — and the two backends must
  agree with each other byte-for-byte.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

try:
    import numpy
except ModuleNotFoundError:  # pragma: no cover - environment guard
    numpy = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
SAMPLE_DIR = REPO_ROOT / "tests" / "data" / "golden_corpus_sample"
DEFAULT_SHOWDOWN_ROOT = Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown")

# History-derived numeric columns (not reconstructable from the stored per-row
# surface; see docs/golden_corpus_notes.md + the track B phase 1 finding).
HISTORY_NUMERIC_COLUMNS = frozenset({63, 64, 65, 138, 139}) | frozenset(range(92, 105))
TRANSITION_TOKEN_START = 23


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the harness defines dataclasses, whose field
    # processing resolves the defining module through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _showdown_root() -> Path:
    return Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)


def _live_showdown_available() -> bool:
    root = _showdown_root()
    if not (root / "dist" / "sim" / "index.js").exists():
        return False
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True, timeout=10)
    except Exception:
        return False
    return True


def _rust_encoder_available() -> bool:
    try:
        import pokezero_search
    except ModuleNotFoundError:
        return False
    return hasattr(pokezero_search, "encode_decision")


@unittest.skipIf(numpy is None, "requires numpy")
class DiffLogicTest(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = _load_script("validate_rust_encoder")

    def _arrays(self):
        return {
            "categorical_ids": numpy.zeros((151, 51), dtype="<i4"),
            "numeric_features": numpy.zeros((151, 155), dtype="<f8"),
            "token_type_ids": numpy.zeros(151, dtype="<i2"),
            "attention_mask": numpy.zeros(151, dtype="|b1"),
            "legal_action_mask": numpy.zeros(9, dtype="|b1"),
        }

    def test_exact_row_counts_all_arrays(self) -> None:
        reports = {name: self.harness.ArrayReport(name=name) for name in self.harness.ARRAY_NAMES}
        want = self._arrays()
        got = {name: array.copy() for name, array in want.items()}
        self.assertTrue(self.harness.diff_row(0, got, want, reports, max_examples=5))
        for report in reports.values():
            self.assertEqual(report.rows_exact, 1)
            self.assertEqual(report.rows_total, 1)
            self.assertEqual(report.cells_exact, report.cells_total)

    def test_one_ulp_float_difference_is_a_mismatch(self) -> None:
        reports = {name: self.harness.ArrayReport(name=name) for name in self.harness.ARRAY_NAMES}
        want = self._arrays()
        got = {name: array.copy() for name, array in want.items()}
        want["numeric_features"][11, 6] = 0.09090909090909091
        got["numeric_features"][11, 6] = numpy.nextafter(0.09090909090909091, 1.0)
        self.assertFalse(self.harness.diff_row(0, got, want, reports, max_examples=5))
        report = reports["numeric_features"]
        self.assertEqual(report.rows_exact, 0)
        self.assertEqual(report.cells_total - report.cells_exact, 1)
        self.assertEqual(list(report.mismatch_positions), [(11, 6)])
        example = report.examples[0]
        self.assertEqual((example["token"], example["column"]), (11, 6))
        self.assertNotEqual(example["got"], example["want"])

    def test_negative_zero_differs_from_positive_zero(self) -> None:
        # Bit-exactness means -0.0 != 0.0; a tolerance-based diff would hide it.
        reports = {name: self.harness.ArrayReport(name=name) for name in self.harness.ARRAY_NAMES}
        want = self._arrays()
        got = {name: array.copy() for name, array in want.items()}
        got["numeric_features"][0, 0] = -0.0
        self.assertFalse(self.harness.diff_row(0, got, want, reports, max_examples=5))
        self.assertEqual(list(reports["numeric_features"].mismatch_positions), [(0, 0)])

    def test_block_attribution_and_example_cap(self) -> None:
        reports = {name: self.harness.ArrayReport(name=name) for name in self.harness.ARRAY_NAMES}
        want = self._arrays()
        got = {name: array.copy() for name, array in want.items()}
        got["categorical_ids"][13, 0] = 7  # action-candidate block
        got["categorical_ids"][23, 1] = 9  # transition block
        got["categorical_ids"][23, 2] = 9
        self.harness.diff_row(0, got, want, reports, max_examples=2)
        report = reports["categorical_ids"]
        coverage = report.block_coverage()
        action = coverage["action_candidates[13-21]"]
        self.assertEqual(action["cells_total"] - action["cells_exact"], 1)
        transition = coverage["transition[23-150]"]
        self.assertEqual(transition["cells_total"] - transition["cells_exact"], 2)
        self.assertEqual(len(report.examples), 2)  # capped

    def test_report_all_exact_flag(self) -> None:
        reports = {name: self.harness.ArrayReport(name=name) for name in self.harness.ARRAY_NAMES}
        want = self._arrays()
        got = {name: array.copy() for name, array in want.items()}
        self.harness.diff_row(0, got, want, reports, max_examples=5)
        payload = self.harness.build_report(reports, rows=1, backend="test")
        self.assertTrue(payload["all_exact"])
        got["attention_mask"][23] = True
        self.harness.diff_row(1, got, want, reports, max_examples=5)
        payload = self.harness.build_report(reports, rows=2, backend="test")
        self.assertFalse(payload["all_exact"])
        rendered = self.harness.render_text_report(payload)
        self.assertIn("MISMATCHES PRESENT", rendered)


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipIf(not SAMPLE_DIR.exists(), "committed golden corpus sample not present")
@unittest.skipIf(not _live_showdown_available(), "requires node and built Pokemon Showdown checkout")
class GoldenSampleBackendTest(unittest.TestCase):
    """Both backends on the committed 5-row sample: every mismatch must be
    history-derived, and the backends must agree byte-for-byte."""

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(SCRIPTS))
        from pokezero.golden_corpus import load_golden_corpus

        cls.backends_module = _load_script("golden_encoder_backends")
        cls.corpus = load_golden_corpus(SAMPLE_DIR)

    def _assert_at_stored_surface_ceiling(self, backend) -> None:
        from pokezero.golden_corpus import GOLDEN_ARRAY_FIELDS

        for row in self.corpus.decision_rows:
            got = backend.encode(self.backends_module.row_inputs_from_decision_row(row))
            for name, dtype, _rank in GOLDEN_ARRAY_FIELDS:
                want = numpy.ascontiguousarray(getattr(row.arrays, name), dtype=dtype)
                have = numpy.ascontiguousarray(got[name])
                if name in ("token_type_ids", "legal_action_mask"):
                    self.assertTrue(numpy.array_equal(have, want), name)
                    continue
                if name == "attention_mask":
                    self.assertTrue(
                        numpy.array_equal(have[:TRANSITION_TOKEN_START], want[:TRANSITION_TOKEN_START])
                    )
                    continue
                if want.dtype.kind == "f":
                    unequal = have.view("<u8") != want.view("<u8")
                else:
                    unequal = have != want
                for token, column in numpy.argwhere(unequal):
                    token, column = int(token), int(column)
                    if token >= TRANSITION_TOKEN_START:
                        continue
                    self.assertEqual(name, "numeric_features", (name, token, column))
                    self.assertIn(column, HISTORY_NUMERIC_COLUMNS, (token, column))

    def test_python_reference_at_ceiling(self) -> None:
        backend = self.backends_module.PythonReferenceBackend(
            showdown_root=_showdown_root(), header=self.corpus.header
        )
        self._assert_at_stored_surface_ceiling(backend)

    @unittest.skipIf(not _rust_encoder_available(), "pokezero_search.encode_decision not installed")
    def test_rust_matches_python_reference_byte_for_byte(self) -> None:
        exporter = _load_script("export_encoder_tables")
        tables_json = json.dumps(
            exporter.build_tables(str(_showdown_root())),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        rust = self.backends_module.RustBackend(tables_json=tables_json, header=self.corpus.header)
        reference = self.backends_module.PythonReferenceBackend(
            showdown_root=_showdown_root(), header=self.corpus.header
        )
        self._assert_at_stored_surface_ceiling(rust)
        for row in self.corpus.decision_rows:
            inputs = self.backends_module.row_inputs_from_decision_row(row)
            got_rust = rust.encode(inputs)
            got_reference = reference.encode(inputs)
            for name in self.backends_module.ARRAY_NAMES:
                self.assertEqual(
                    numpy.ascontiguousarray(got_rust[name]).tobytes(),
                    numpy.ascontiguousarray(got_reference[name]).tobytes(),
                    name,
                )


if __name__ == "__main__":
    unittest.main()
