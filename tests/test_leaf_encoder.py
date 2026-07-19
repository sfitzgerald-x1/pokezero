"""Tests for the leaf observation path (engine-swap capstone).

Three layers:

- ``RustFoldFullSurfaceTest``: the committed 5-row sample re-encoded through
  ``NativeEncoder.encode_with_fold`` (native in-crate fold-product
  consumption) must reproduce ALL five golden arrays byte-exactly — the
  permanent regression net for the history-cell port (transition rows,
  tendency counters, pinned Tier-2, attention extent).
- ``LeafRootParityCommittedSampleTest``: the committed sample driven through
  the FULL leaf path at depth 0 — world from the recorded payload + true
  teams, ``LeafEncoder.encode_leaf`` on the untouched root state — must be
  byte-exact too (the root-parity gate's no-Showdown-checkout core; the full
  gate is ``scripts/leaf_root_parity.py`` over the golden corpora).
- ``HpPercentGridTest``: the /100 reconciliation contract — a fold advanced
  over percent-based lines produces damage fractions on the /100 grid the
  live ladder stream uses, while exact-base lines land on the true-HP grid
  (the local harness's omniscient regime).

Skip policy: wheel-dependent tests skip when the installed ``pokezero_search``
lacks the new surfaces; tables/showdown-dependent tests skip when their
inputs are absent (same pattern as the other crate gates).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

try:
    import numpy
except ModuleNotFoundError:  # pragma: no cover
    numpy = None

try:
    import pokezero_search
except ModuleNotFoundError:  # pragma: no cover
    pokezero_search = None

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_SAMPLE_DIR = Path(__file__).parent / "data" / "golden_corpus_sample"
SCRIPTS_DIR = REPO_ROOT / "scripts"

from pokezero.golden_corpus import (  # noqa: E402
    GOLDEN_ARRAY_FIELDS,
    GOLDEN_CORPUS_SCHEMA_VERSION,
    load_golden_corpus,
)
from pokezero.golden_corpus_fold import iter_fold_records  # noqa: E402


def _wheel_has(name: str) -> bool:
    return pokezero_search is not None and hasattr(pokezero_search, name)


def _tables_json() -> str | None:
    """The encoder tables artifact: prefer a local build, else regenerate."""

    local = REPO_ROOT / "corpus" / "encoder_tables.json"
    if local.exists():
        return local.read_text(encoding="utf-8")
    try:
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

        if not Path(DEFAULT_SHOWDOWN_ROOT).exists():
            return None
        sys.path.insert(0, str(SCRIPTS_DIR))
        from export_encoder_tables import build_tables  # noqa: E402

        return json.dumps(
            build_tables(str(DEFAULT_SHOWDOWN_ROOT)),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except Exception:  # pragma: no cover - environment-dependent
        return None


def _sample_rows_with_folds():
    corpus = load_golden_corpus(COMMITTED_SAMPLE_DIR)
    fold_states = {}
    for record in iter_fold_records(
        COMMITTED_SAMPLE_DIR, expected_schema_version=GOLDEN_CORPUS_SCHEMA_VERSION
    ):
        fold_states[int(record["array_row_index"])] = record["fold_state"]
    return corpus, fold_states


def _row_inputs(row) -> str:
    sys.path.insert(0, str(SCRIPTS_DIR))
    from golden_encoder_backends import row_inputs_from_decision_row  # noqa: E402

    return json.dumps(row_inputs_from_decision_row(row), sort_keys=True)


def _assert_arrays_exact(test: unittest.TestCase, buffers, row, label: str) -> None:
    for name, dtype, _ in GOLDEN_ARRAY_FIELDS:
        want = numpy.ascontiguousarray(getattr(row.arrays, name), dtype=dtype)
        got = numpy.frombuffer(buffers[name], dtype=dtype).reshape(want.shape)
        if want.dtype.kind == "f":
            unsigned = numpy.dtype(f"<u{want.dtype.itemsize}")
            equal = bool((got.view(unsigned) == want.view(unsigned)).all())
        else:
            equal = bool((got == want).all())
        test.assertTrue(equal, f"{label}: array {name} diverged from golden")


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipUnless(_wheel_has("NativeEncoder"), "wheel lacks NativeEncoder")
class RustFoldFullSurfaceTest(unittest.TestCase):
    """Committed sample, full observation surface via native fold products."""

    def test_committed_sample_full_surface_exact(self) -> None:
        tables_json = _tables_json()
        if tables_json is None:
            self.skipTest("no encoder tables artifact and no Showdown checkout")
        corpus, fold_states = _sample_rows_with_folds()
        encoder = pokezero_search.NativeEncoder(tables_json)
        for index, row in enumerate(corpus.decision_rows):
            fold = pokezero_search.FoldState.from_payload(fold_states[index])
            buffers = encoder.encode_with_fold(_row_inputs(row), fold)
            _assert_arrays_exact(self, buffers, row, f"sample row {index}")


@unittest.skipIf(numpy is None, "requires numpy")
@unittest.skipUnless(_wheel_has("LeafEncoder"), "wheel lacks LeafEncoder")
class LeafRootParityCommittedSampleTest(unittest.TestCase):
    """Depth-0 leaf-path parity over the committed sample (world + engine)."""

    def test_committed_sample_leaf_path_exact(self) -> None:
        tables_json = _tables_json()
        if tables_json is None:
            self.skipTest("no encoder tables artifact and no Showdown checkout")
        try:
            from pokezero.dex import load_showdown_dex_cached
            from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

            if not Path(DEFAULT_SHOWDOWN_ROOT).exists():
                self.skipTest("no Showdown checkout (dex required)")
            dex = load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT)
        except Exception as error:  # pragma: no cover
            self.skipTest(f"dex unavailable: {error}")
        from pokezero.env import BattleStartOverride
        from pokezero.engine_world import (
            EngineWorldUnsupported,
            battle_spec_from_payload,
        )
        from pokezero.poke_engine_adapter import build_poke_engine_state

        corpus, fold_states = _sample_rows_with_folds()
        games = {game.record.battle_id: game for game in corpus.games}
        driven = 0
        for index, row in enumerate(corpus.decision_rows):
            game = games[row.battle_id]
            packed = {
                slot: (game.record.true_teams.get(slot) or {}).get("packed")
                for slot in ("p1", "p2")
            }
            if not packed["p1"] or not packed["p2"]:
                continue
            try:
                world = battle_spec_from_payload(
                    row.public_materialization,
                    BattleStartOverride(player_teams=packed),
                    dex=dex,
                    approximate_sleep_turns=True,
                    approximate_substitute_health=True,
                )
                state = build_poke_engine_state(world.spec)
            except EngineWorldUnsupported:
                continue
            ctx = json.dumps(
                {
                    "p1": list(world.party_species["p1"]),
                    "p2": list(world.party_species["p2"]),
                    "turn": int(row.public_materialization.get("turn") or 0),
                }
            )
            state_str = state.to_string()
            encoder = pokezero_search.LeafEncoder(
                tables_json, _row_inputs(row), ctx, state_str
            )
            fold = pokezero_search.FoldState.from_payload(fold_states[index])
            buffers = encoder.encode_leaf(
                state_str, fold, int(row.observation_metadata.get("turn_number") or 0)
            )
            _assert_arrays_exact(self, buffers, row, f"sample row {index}")
            driven += 1
        self.assertGreater(driven, 0, "no committed-sample row could be driven")


@unittest.skipUnless(_wheel_has("FoldState"), "wheel lacks FoldState")
class HpPercentGridTest(unittest.TestCase):
    """The /100 base decision: percent-rendered lines put fold damage
    fractions on the ladder stream's /100 grid; exact lines on the true grid.

    (The local harness feeds the OMNISCIENT stream — exact HP for both sides
    — so the mapper's default exact rendering already matches the root fold's
    base in the training/eval domain; the percent base exists for ladder
    deployments, gated by ``EventContext.hp_percent``.)
    """

    LEAD = [
        "|switch|p1a: Rattata|Rattata, L88|100/100",
        "|switch|p2a: Chansey|Chansey, L80|100/100",
        "|turn|1",
    ]

    @staticmethod
    def _damage_fraction(lines):
        fold = pokezero_search.FoldState.initial("p1")
        fold.advance_in_place(HpPercentGridTest.LEAD + lines)
        products = fold.products_payload()
        tokens = [
            t
            for t in products["transition_tokens"]
            if t["kind"] == "move" and t["damage_fraction"] > 0
        ]
        assert tokens, "expected a damaging move token"
        return tokens[-1]["damage_fraction"]

    def test_percent_lines_land_on_the_percent_grid(self) -> None:
        # Ladder regime: opponent shown as ceil(100*468/641) = 74 -> /100.
        fraction = self._damage_fraction(
            [
                "|move|p1a: Rattata|Tackle|p2a: Chansey",
                "|-damage|p2a: Chansey|74/100",
                "|upkeep",
                "|turn|2",
            ]
        )
        self.assertAlmostEqual(fraction, 1.0 - 74 / 100, places=12)
        self.assertAlmostEqual(fraction * 100, round(fraction * 100), places=9)

    def test_exact_lines_land_on_the_true_grid(self) -> None:
        # Local (omniscient) regime: the same hit on the true 641 base.
        fraction = self._damage_fraction(
            [
                "|move|p1a: Rattata|Tackle|p2a: Chansey",
                "|-damage|p2a: Chansey|468/641",
                "|upkeep",
                "|turn|2",
            ]
        )
        self.assertAlmostEqual(fraction, 1.0 - 468 / 641, places=12)
        # Not representable on the /100 grid: the two regimes are distinct,
        # which is why the render base must match the root stream's base.
        self.assertNotAlmostEqual(fraction * 100, round(fraction * 100), places=9)


if __name__ == "__main__":
    unittest.main()
