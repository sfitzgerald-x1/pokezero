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

import pokezero.showdown as showdown_module
from _showdown_root import requires_showdown, showdown_root_str

try:
    import numpy
except ModuleNotFoundError:  # pragma: no cover - environment guard
    numpy = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
SAMPLE_DIR = REPO_ROOT / "tests" / "data" / "golden_corpus_sample"
DEFAULT_SHOWDOWN_ROOT = Path(showdown_root_str())

# History-derived numeric columns: not reconstructable from the stored per-row surface, so the
# reference backend is allowed to differ from the recorded arrays there and only there
# (docs/golden_corpus_notes.md + the track B phase 1 finding).
#
# Named, not indexed. This was a frozenset of raw v2.2 column indices --
# `{63, 64, 65, 138, 139} | range(92, 105)` -- which silently stopped meaning anything the moment
# the sample was regenerated at v4: column 52 there is NUMERIC_MON_TURNS_ACTIVE_TOTAL, a genuine
# history column, and the test failed as if parity had broken. Six of those indices had no name in
# v2.2 at all, and two (NUMERIC_TIER2_CB_PINNED, NUMERIC_TIER2_INVESTMENT_PINNED) do not exist in
# v4 -- resolved leniently below so the set stays correct across schema generations rather than
# needing a hand edit at each one.
HISTORY_NUMERIC_COLUMN_NAMES = (
    "NUMERIC_MON_SWITCHED_BEFORE_ATTACK",
    "NUMERIC_MON_STAYED_AND_ATTACKED",
    "NUMERIC_MON_TURNS_ACTIVE_TOTAL",
    "NUMERIC_TIER2_CB_PINNED",
    "NUMERIC_TIER2_INVESTMENT_PINNED",
    "NUMERIC_STAT_OPP_SWITCH_COUNT",
    "NUMERIC_STAT_OPP_DECISION_OPPORTUNITIES",
    "NUMERIC_STAT_BLOCKED_ON_OUR_ATTACK",
    "NUMERIC_STAT_PURSUIT_INTERCEPT_PREDICT",
    "NUMERIC_STAT_MY_SWITCH_TURNS",
    "NUMERIC_STAT_WEATHER_REVEAL_OFFSET",
)


# `NUMERIC_STAT_WEATHER_REVEAL_OFFSET` names the START of a block, not a column: per weather in
# `_WEATHER_REVEAL_ORDER` it is a (set-this-game, turn-fraction) pair (`showdown.py:232-234`).
# Only the offset itself appears in the exported layout, so resolving names alone recovered 1 of
# them where the original hardcoded `range(92, 105)` covered the lot.
#
# The block is NOT a fixed 8. v3 explicitly drops the last weather pair --
# `NUMERIC_STAT_WEATHER_REVEAL_OFFSET + 6` and `+ 7` are in `V3_DROPPED_LEGACY_NUMERIC_INDICES`
# (`showdown.py:789-790`), and v4 inherits that -- so the live block is 8 columns at v2.2 and 6 at
# v3/v4. An earlier version of this code used a fixed span of 8 clipped by the array width, which
# gave the right answer at v4 only because the block happens to sit at the end of that layout, and
# the WRONG answer at v3: it pulled physical 121/122 (`NUMERIC_TT_DAMAGE_FRACTION`,
# `NUMERIC_TT_N_HITS`) into the allowance. Those are transition numerics, and the first is in
# `V3_REWRITTEN_LEGACY_NUMERIC_INDICES` -- a column whose v3 semantics deliberately changed, so
# exactly the wrong thing to silently exempt from a parity comparison.
#
# Resolved through the schema projection instead, which answers None for precisely the dropped
# indices and needs no width heuristic.
_BLOCK_LEGACY_SPANS = {"NUMERIC_STAT_WEATHER_REVEAL_OFFSET": 8}


def _history_numeric_columns(layout) -> frozenset:
    """History-column NAMES to physical indices, block offsets expanded via the schema projection."""
    from pokezero.showdown import numeric_index_if_present_for_schema

    numeric_columns = layout["numeric_columns"]
    schema_version = layout["schema_version"]
    resolved: set[int] = set()
    for name in HISTORY_NUMERIC_COLUMN_NAMES:
        if name not in numeric_columns:
            continue
        span = _BLOCK_LEGACY_SPANS.get(name)
        if span is None:
            resolved.add(numeric_columns[name])
            continue
        # Block: walk the LEGACY indices and let the projection drop the ones this schema omits.
        legacy_start = getattr(showdown_module, name)
        for offset in range(span):
            physical = numeric_index_if_present_for_schema(schema_version, legacy_start + offset)
            if physical is not None:
                resolved.add(physical)
    return frozenset(resolved)


def _transition_token_start(layout) -> int:
    """First transition token, from the layout rather than a pinned 23."""
    return int(layout["token_offsets"]["transition"])


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


def _sample_schema_version(corpus) -> str:
    """The observation schema the committed sample was encoded at, from its own header.

    Reading it rather than hardcoding is what keeps these tests pinned to PARITY instead of to a
    particular schema generation: the sample is regenerated each time the encode contract moves
    (v2.2 -> v4 most recently), and every hardcoded default turned that regeneration into a wall
    of failures that look like parity breaks and are not.
    """
    return str(corpus.header["observation"]["schema_version"])


class HistoryColumnResolutionTest(unittest.TestCase):
    """The history allowance, resolved at every schema — not just the sample's.

    `_history_numeric_columns` is only ever called with the committed corpus's schema, which is
    v4, so the v2.2 and v3 expected counts in `_assert_at_stored_surface_ceiling` were correct but
    never executed. Unexercised constants decay silently, and this file has already shipped one
    allowance that was wrong at v3 while green at v4 — the weather block, which v3 and v4 truncate
    from 8 columns to 6 (`showdown.py:789-790`).

    Needs no corpus and no wheel; just the exported layout.
    """

    SCHEMAS = {
        "pokezero.observation.v2.2": 18,
        "pokezero.observation.v3": 16,
        "pokezero.observation.v4": 14,
    }

    @requires_showdown("resolves the exported layout")
    def test_every_schema_resolves_the_expected_history_columns(self) -> None:
        exporter = _load_script("export_encoder_tables")
        for schema_version, expected in self.SCHEMAS.items():
            with self.subTest(schema=schema_version):
                layout = exporter.build_tables(
                    showdown_root_str(), observation_schema_version=schema_version
                )["layout"]
                self.assertEqual(len(_history_numeric_columns(layout)), expected)

    @requires_showdown("resolves the exported layout")
    def test_v3_does_not_absorb_the_transition_numerics(self) -> None:
        """The specific regression: a fixed span of 8 pulled v3's 121/122 into the allowance.

        Those are NUMERIC_TT_DAMAGE_FRACTION and NUMERIC_TT_N_HITS — transition numerics, and the
        first is in `V3_REWRITTEN_LEGACY_NUMERIC_INDICES`, a column whose v3 semantics deliberately
        changed. Exempting it from a parity comparison is the worst case, so it gets its own test
        rather than riding on a count.
        """
        exporter = _load_script("export_encoder_tables")
        layout = exporter.build_tables(
            showdown_root_str(), observation_schema_version="pokezero.observation.v3"
        )["layout"]
        columns = _history_numeric_columns(layout)
        for name in ("NUMERIC_TT_DAMAGE_FRACTION", "NUMERIC_TT_N_HITS"):
            index = layout["numeric_columns"].get(name)
            self.assertIsNotNone(index, f"{name} vanished from the v3 layout")
            self.assertNotIn(index, columns, f"{name} is in the history allowance")

    def test_every_block_offset_name_declares_its_span(self) -> None:
        """A future `*_OFFSET` added to the name list without a span contributes 1, silently."""
        missing = [
            name
            for name in HISTORY_NUMERIC_COLUMN_NAMES
            if name.endswith("_OFFSET") and name not in _BLOCK_LEGACY_SPANS
        ]
        self.assertEqual(
            missing, [], "block-offset names must declare a span in _BLOCK_LEGACY_SPANS"
        )


@unittest.skipIf(numpy is None, "requires numpy")
class DiffLogicTest(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = _load_script("validate_rust_encoder")

    def _arrays(self, token_count: int = 151):
        return {
            "categorical_ids": numpy.zeros((token_count, 51), dtype="<i4"),
            "numeric_features": numpy.zeros((token_count, 155), dtype="<f8"),
            "token_type_ids": numpy.zeros(token_count, dtype="<i2"),
            "attention_mask": numpy.zeros(token_count, dtype="|b1"),
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

    def test_v3_block_attribution_uses_the_87_row_schema(self) -> None:
        reports = {name: self.harness.ArrayReport(name=name) for name in self.harness.ARRAY_NAMES}
        want = self._arrays(token_count=87)
        got = {name: array.copy() for name, array in want.items()}
        got["categorical_ids"][86, 0] = 7

        self.harness.diff_row(0, got, want, reports, max_examples=1)

        transition = reports["categorical_ids"].block_coverage()["transition[23-86]"]
        self.assertEqual(transition["cells_total"], 64 * 51)
        self.assertEqual(transition["cells_total"] - transition["cells_exact"], 1)

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

        exporter = _load_script("export_encoder_tables")
        layout = exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=_sample_schema_version(self.corpus),
        )["layout"]
        history_columns = _history_numeric_columns(layout)
        transition_start = _transition_token_start(layout)
        # An EXPECTED COUNT, not a truthiness check. The previous `assertTrue(history_columns)`
        # was satisfied by 9 of 11 names resolving, which is how the weather-reveal block silently
        # shrank from 8 columns to 1 (see `_history_numeric_columns`). A count makes any future
        # shrink loud at the point it happens rather than at the next regeneration.
        expected = {
            "pokezero.observation.v2.2": 18,
            "pokezero.observation.v3": 16,
            "pokezero.observation.v4": 14,
        }.get(_sample_schema_version(self.corpus))
        self.assertIsNotNone(
            expected,
            "unknown schema in the committed sample; add its expected history-column count",
        )
        self.assertEqual(
            len(history_columns),
            expected,
            "the history-column allowance changed size; a SHRINK turns legitimate history "
            "divergence into a false parity failure, a GROWTH hides a real one",
        )

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
                        numpy.array_equal(have[:transition_start], want[:transition_start])
                    )
                    continue
                if want.dtype.kind == "f":
                    unequal = have.view("<u8") != want.view("<u8")
                else:
                    unequal = have != want
                for token, column in numpy.argwhere(unequal):
                    token, column = int(token), int(column)
                    if token >= transition_start:
                        continue
                    self.assertEqual(name, "numeric_features", (name, token, column))
                    self.assertIn(column, history_columns, (token, column))

    def test_python_reference_at_ceiling(self) -> None:
        backend = self.backends_module.PythonReferenceBackend(
            showdown_root=_showdown_root(), header=self.corpus.header
        )
        self._assert_at_stored_surface_ceiling(backend)

    @unittest.skipIf(not _rust_encoder_available(), "pokezero_search.encode_decision not installed")
    def test_rust_matches_python_reference_byte_for_byte(self) -> None:
        exporter = _load_script("export_encoder_tables")
        # Build tables at the SCHEMA THE CORPUS WAS WRITTEN AT, not `build_tables`'s v2.2 default.
        # The committed sample is regenerated whenever the schema moves; hardcoding the default
        # made these tests fail on the regeneration rather than on a real parity break.
        tables_json = json.dumps(
            exporter.build_tables(
                str(_showdown_root()),
                observation_schema_version=_sample_schema_version(self.corpus),
            ),
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


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipIf(not SAMPLE_DIR.exists(), "committed golden corpus sample not present")
@unittest.skipIf(not _live_showdown_available(), "requires a built local Showdown checkout")
@unittest.skipIf(not _rust_encoder_available(), "pokezero_search.encode_decision not installed")
class OovParityTests(unittest.TestCase):
    """The blake2b OOV bucket path is dead code under the closed corpus vocab —
    force it live (review finding) by mutating a row's species to unknown
    strings and requiring rust/python-reference byte identity end to end."""

    def test_oov_species_hash_identically_through_full_encode(self) -> None:
        import copy
        import json as _json

        from pokezero.golden_corpus import load_golden_corpus

        backends_module = _load_script("golden_encoder_backends")
        exporter = _load_script("export_encoder_tables")
        corpus = load_golden_corpus(SAMPLE_DIR)
        tables_json = _json.dumps(
            exporter.build_tables(
                str(_showdown_root()),
                observation_schema_version=_sample_schema_version(corpus),
            ),
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        )
        rust = backends_module.RustBackend(tables_json=tables_json, header=corpus.header)
        reference = backends_module.PythonReferenceBackend(
            showdown_root=_showdown_root(), header=corpus.header
        )
        row = corpus.decision_rows[0]
        inputs = backends_module.row_inputs_from_decision_row(row)
        for oov_name in ("Xyzzymon", "MissingNo", "Glitchagon"):
            mutated = copy.deepcopy(inputs)
            payload = mutated["public_materialization"]
            opponent = "p2" if mutated.get("player_id") == "p1" else "p1"
            rows_list = payload["sides"][opponent]["pokemon"]
            if not rows_list:
                self.skipTest("sample row has no revealed opponent")
            rows_list[0]["species"] = oov_name
            got_rust = rust.encode(mutated)
            got_reference = reference.encode(mutated)
            for name in backends_module.ARRAY_NAMES:
                self.assertEqual(
                    numpy.ascontiguousarray(got_rust[name]).tobytes(),
                    numpy.ascontiguousarray(got_reference[name]).tobytes(),
                    (oov_name, name),
                )


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipIf(not SAMPLE_DIR.exists(), "committed golden corpus sample not present")
@unittest.skipIf(not _live_showdown_available(), "requires a built local Showdown checkout")
@unittest.skipIf(not _rust_encoder_available(), "pokezero_search.encode_decision not installed")
class CompareBackendsCliTests(unittest.TestCase):
    """The reproducible form of the cross-backend byte-identity claim: the
    compare-backends CLI mode must exit 0 on the committed sample."""

    def test_compare_backends_exits_zero_on_sample(self) -> None:
        import json as _json
        import tempfile

        exporter = _load_script("export_encoder_tables")
        cli = _load_script("validate_rust_encoder")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            from pokezero.golden_corpus import load_golden_corpus as _load_corpus

            handle.write(_json.dumps(
                exporter.build_tables(
                    str(_showdown_root()),
                    observation_schema_version=_sample_schema_version(_load_corpus(SAMPLE_DIR)),
                ),
                sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            ))
            tables_path = handle.name
        exit_code = cli.main([
            "--backend", "compare-backends",
            "--corpus", str(SAMPLE_DIR),
            "--showdown-root", str(_showdown_root()),
            "--tables", tables_path,
        ])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
