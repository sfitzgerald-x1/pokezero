"""Tests for the schema-v2 fold surface of the golden corpus (track B).

Three layers, none requiring a Showdown checkout:

- Synthetic corpora: fold rows built by advancing the production FoldState
  over hand-built protocol slices, written through the real writer, then
  verified (links, hashes, chain contiguity) and chain-validated through the
  row-pair harness — including a corrupted-state corpus that MUST fail.
- Contract details: |t:| rejection, mixed fold/no-fold corpora, misaligned
  fold surfaces, payload determinism, sampling.
- The committed 5-row sample: full row-pair validation of its fold chains via
  the python-reference backend — the permanent no-Showdown regression net for
  ``FoldState.advance`` against recorded production state.

The live end-to-end path (generation -> sidecar -> validation over real games)
is exercised by ``GoldenCorpusLiveSmokeTest`` in ``tests/test_golden_corpus.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    numpy = None

from pokezero.golden_corpus import (
    GOLDEN_CORPUS_SCHEMA_VERSION,
    FOLD_ROWS_FILENAME,
    GoldenGame,
    sample_golden_corpus,
    verify_golden_corpus,
    write_golden_corpus,
)
from pokezero.golden_corpus_fold import (
    GoldenFoldRow,
    PythonReferenceFoldBackend,
    fold_products_to_payload,
    fold_row_from_record,
    iter_fold_records,
    validate_fold_chains,
)
from pokezero.transitions_fold import FoldState

TESTS_DATA_DIR = Path(__file__).parent / "data"
COMMITTED_SAMPLE_DIR = TESTS_DATA_DIR / "golden_corpus_sample"
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _rust_fold_available() -> bool:
    """True when the installed pokezero_search wheel carries the fold port
    (skip-if-absent, like the encoder gate in test_validate_rust_encoder)."""

    try:
        import pokezero_search
    except (ImportError, OSError):  # pragma: no cover - environment guard
        return False
    return hasattr(pokezero_search, "FoldState")

# A small two-boundary protocol log: the lead slice (what both seats' chains
# see at their first decision) and one played turn (the second decision's
# inter-boundary slice). A |t:| line rides along to prove filtering upstream.
_LEAD_SLICE = (
    "|player|p1|Alice|1",
    "|player|p2|Bob|2",
    "|switch|p1a: Tyranitar|Tyranitar, L74, M|100/100",
    "|switch|p2a: Starmie|Starmie, L77|100/100",
    "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar",
    "|turn|1",
)
_TURN_SLICE = (
    "|move|p2a: Starmie|Hydro Pump|p1a: Tyranitar",
    "|-supereffective|p1a: Tyranitar",
    "|-damage|p1a: Tyranitar|20/100",
    "|move|p1a: Tyranitar|Crunch|p2a: Starmie",
    "|-supereffective|p2a: Starmie",
    "|-damage|p2a: Starmie|45/100",
    "|",
    "|-weather|Sandstorm|[upkeep]",
    "|-damage|p2a: Starmie|39/100|[from] Sandstorm",
    "|upkeep",
    "|turn|2",
)


def _synthetic_fold_rows(slices_by_row):
    """Build fold rows exactly as the generator does: per-seat chains advanced
    over the inter-boundary slices, states/products exported at each boundary."""

    states: dict[str, FoldState] = {}
    chain_counters = {"p1": 0, "p2": 0}
    rows = []
    for seat, slice_ in slices_by_row:
        state = states.get(seat) or FoldState.initial(perspective_slot=seat)
        state, _ = state.advance(slice_)
        states[seat] = state
        rows.append(
            GoldenFoldRow(
                player_id=seat,
                chain_index=chain_counters[seat],
                event_slice=tuple(slice_),
                annotation_overlay={},
                fold_state=state.to_payload(),
                products=fold_products_to_payload(state.products()),
            )
        )
        chain_counters[seat] += 1
    return tuple(rows)


def _default_fold_rows():
    return _synthetic_fold_rows(
        [("p1", _LEAD_SLICE), ("p2", _LEAD_SLICE), ("p1", _TURN_SLICE), ("p2", _TURN_SLICE)]
    )


@unittest.skipIf(numpy is None, "requires numpy")
class FoldSidecarRoundTripTest(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_golden_corpus import _synthetic_game

        self._synthetic_game = _synthetic_game
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out_dir = Path(self._tmp.name) / "corpus"

    def _game_with_fold(self, *, fold_rows=None):
        game = self._synthetic_game(0, rows=4)
        return GoldenGame(
            record=game.record,
            rows=game.rows,
            fold_rows=_default_fold_rows() if fold_rows is None else fold_rows,
        )

    def test_write_verify_and_validate_chains(self) -> None:
        manifest = write_golden_corpus(
            self.out_dir,
            header={"generator": {"synthetic": True}},
            games=[self._game_with_fold()],
        )
        self.assertEqual(manifest["counts"]["fold_rows"], 4)
        self.assertIn(FOLD_ROWS_FILENAME, manifest["files"])
        self.assertGreater(
            manifest["files"][FOLD_ROWS_FILENAME]["uncompressed_bytes"],
            manifest["files"][FOLD_ROWS_FILENAME]["bytes"],
            "gzip sidecar should be smaller than its uncompressed content",
        )

        verification = verify_golden_corpus(self.out_dir)
        self.assertEqual(verification.fold_rows, 4)

        report = validate_fold_chains(
            self.out_dir,
            PythonReferenceFoldBackend(),
            expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION,
        )
        self.assertTrue(report.ok, report.mismatches)
        self.assertEqual(report.games, 1)
        self.assertEqual(report.chains, 2)
        self.assertEqual(report.rows_validated, 4)
        self.assertEqual(report.initial_validations, 2)
        self.assertEqual(report.pair_validations, 2)

    def test_corrupted_recorded_state_fails_row_pair_validation(self) -> None:
        fold_rows = list(_default_fold_rows())
        corrupted_state = dict(fold_rows[2].fold_state)
        corrupted_state["turn_number"] = 99  # a plausible-looking but wrong record
        fold_rows[2] = GoldenFoldRow(
            player_id=fold_rows[2].player_id,
            chain_index=fold_rows[2].chain_index,
            event_slice=fold_rows[2].event_slice,
            annotation_overlay=fold_rows[2].annotation_overlay,
            fold_state=corrupted_state,
            products=fold_rows[2].products,
        )
        write_golden_corpus(
            self.out_dir,
            header={"generator": {"synthetic": True}},
            games=[self._game_with_fold(fold_rows=tuple(fold_rows))],
        )
        # Integrity verification still passes: the record is self-consistent...
        verify_golden_corpus(self.out_dir)
        # ...but the advance contract catches it.
        report = validate_fold_chains(
            self.out_dir,
            PythonReferenceFoldBackend(),
            expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION,
        )
        self.assertFalse(report.ok)
        self.assertTrue(any(m.surface == "fold_state" for m in report.mismatches))
        self.assertTrue(
            any("turn_number" in m.detail for m in report.mismatches),
            [m.detail for m in report.mismatches],
        )

    def test_mixed_fold_presence_is_refused(self) -> None:
        with_fold = self._game_with_fold()
        without_fold = self._synthetic_game(1, rows=3)
        with self.assertRaisesRegex(ValueError, "every game carries the fold surface"):
            write_golden_corpus(
                self.out_dir,
                header={"generator": {"synthetic": True}},
                games=[with_fold, without_fold],
            )

    def test_misaligned_fold_rows_are_refused(self) -> None:
        game = self._synthetic_game(0, rows=4)
        with self.assertRaisesRegex(ValueError, "parallel"):
            GoldenGame(record=game.record, rows=game.rows, fold_rows=_default_fold_rows()[:2])

    def test_seat_mismatch_is_refused_at_write(self) -> None:
        fold_rows = _synthetic_fold_rows(
            # p2 first: disagrees with the golden rows' p1,p2,p1,p2 seat order.
            [("p2", _LEAD_SLICE), ("p1", _LEAD_SLICE), ("p2", _TURN_SLICE), ("p1", _TURN_SLICE)]
        )
        with self.assertRaisesRegex(ValueError, "does not match its decision row"):
            write_golden_corpus(
                self.out_dir,
                header={"generator": {"synthetic": True}},
                games=[self._game_with_fold(fold_rows=fold_rows)],
            )

    def test_wall_clock_lines_are_rejected_in_slices(self) -> None:
        with self.assertRaisesRegex(ValueError, "wall-clock"):
            GoldenFoldRow(
                player_id="p1",
                chain_index=0,
                event_slice=("|t:|1700000001",),
                annotation_overlay={},
                fold_state=FoldState.initial(perspective_slot="p1").to_payload(),
                products=fold_products_to_payload(
                    FoldState.initial(perspective_slot="p1").products()
                ),
            )

    def test_sampling_carries_the_fold_surface(self) -> None:
        write_golden_corpus(
            self.out_dir,
            header={"generator": {"synthetic": True}},
            games=[self._game_with_fold()],
        )
        sample_dir = Path(self._tmp.name) / "sample"
        manifest = sample_golden_corpus(self.out_dir, sample_dir, max_decisions=2)
        self.assertEqual(manifest["counts"]["fold_rows"], 2)
        verification = verify_golden_corpus(sample_dir)
        self.assertEqual(verification.fold_rows, 2)
        report = validate_fold_chains(
            sample_dir,
            PythonReferenceFoldBackend(),
            expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION,
        )
        self.assertTrue(report.ok, report.mismatches)
        self.assertEqual(report.rows_validated, 2)
        # The sampled records must be byte-identical to the source records
        # (same states, same slices, same links).
        source = [
            record
            for record in iter_fold_records(
                self.out_dir, expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION
            )
        ][:2]
        sampled = list(
            iter_fold_records(sample_dir, expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION)
        )
        self.assertEqual(
            [json.dumps(r, sort_keys=True) for r in sampled],
            [json.dumps(r, sort_keys=True) for r in source],
        )

    def test_fold_row_round_trips_through_its_record(self) -> None:
        write_golden_corpus(
            self.out_dir,
            header={"generator": {"synthetic": True}},
            games=[self._game_with_fold()],
        )
        records = list(
            iter_fold_records(self.out_dir, expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION)
        )
        rebuilt = [fold_row_from_record(record) for record in records]
        for original, back in zip(_default_fold_rows(), rebuilt):
            self.assertEqual(back.player_id, original.player_id)
            self.assertEqual(back.chain_index, original.chain_index)
            self.assertEqual(back.event_slice, original.event_slice)
            self.assertEqual(
                json.dumps(back.fold_state, sort_keys=True),
                json.dumps(original.fold_state, sort_keys=True),
            )
            self.assertEqual(
                json.dumps(back.products, sort_keys=True),
                json.dumps(original.products, sort_keys=True),
            )


class FoldProductsPayloadTest(unittest.TestCase):
    def test_products_payload_is_deterministic_and_json_safe(self) -> None:
        state = FoldState.initial(perspective_slot="p1")
        state, products = state.advance(_LEAD_SLICE + _TURN_SLICE)
        payload_a = fold_products_to_payload(products)
        payload_b = fold_products_to_payload(state.products())
        canonical_a = json.dumps(payload_a, sort_keys=True, allow_nan=False)
        canonical_b = json.dumps(payload_b, sort_keys=True, allow_nan=False)
        self.assertEqual(canonical_a, canonical_b)
        decoded = json.loads(canonical_a)
        self.assertEqual(decoded["transition_token_total"], products.transition_token_total)
        self.assertEqual(decoded["turn_merged_total"], products.turn_merged_total)
        self.assertEqual(
            decoded["tendency_stats"]["perspective_slot"], products.tendency_stats.perspective_slot
        )


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipIf(
    not (COMMITTED_SAMPLE_DIR / FOLD_ROWS_FILENAME).exists(),
    "committed golden corpus sample lacks a fold sidecar",
)
class CommittedSampleFoldChainTest(unittest.TestCase):
    """Permanent regression net: FoldState.advance must keep reproducing the
    committed sample's recorded production fold states — no Showdown needed."""

    def test_committed_sample_chains_validate(self) -> None:
        report = validate_fold_chains(
            COMMITTED_SAMPLE_DIR,
            PythonReferenceFoldBackend(),
            expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION,
        )
        self.assertTrue(report.ok, report.mismatches)
        self.assertEqual(report.rows_validated, 5)
        self.assertEqual(report.chains, 2)
        self.assertGreater(report.pair_validations, 0)


@unittest.skipIf(not _rust_fold_available(), "pokezero_search.FoldState not installed")
class RustCommittedSampleFoldChainTest(unittest.TestCase):
    """The Rust advance() port (rust/pokezero-search src/fold.rs) must keep
    reproducing the committed sample's recorded production fold states and
    products byte-exactly — the no-Showdown regression net for the native
    backend, gated on the wheel being importable."""

    def test_committed_sample_chains_validate_rust(self) -> None:
        sys.path.insert(0, str(SCRIPTS_DIR))
        try:
            from golden_fold_backends import RustFoldBackend
        finally:
            sys.path.remove(str(SCRIPTS_DIR))
        report = validate_fold_chains(
            COMMITTED_SAMPLE_DIR,
            RustFoldBackend(),
            expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION,
        )
        self.assertTrue(report.ok, report.mismatches)
        self.assertEqual(report.rows_validated, 5)
        self.assertEqual(report.chains, 2)
        self.assertGreater(report.pair_validations, 0)


if __name__ == "__main__":
    unittest.main()
