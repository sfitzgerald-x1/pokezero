"""Bit-exact Python/Rust parity for the complete V4 (history-free k0 pack) surface.

The V3 twin (``test_rust_encoder_v3.py``) proves the turn-merged surface. This proves the
opposite shape: no transition region at all, plus the feature-pack columns the native encoder
reads off the observation metadata.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import unittest

try:
    import numpy
    import pokezero_search
except (ImportError, OSError):  # pragma: no cover - optional native gate
    numpy = None
    pokezero_search = None

from pokezero.golden_corpus import load_golden_corpus
from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V3, OBSERVATION_SCHEMA_VERSION_V4
from pokezero.showdown import observation_from_player_state

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
SAMPLE_DIR = REPO_ROOT / "tests" / "data" / "golden_corpus_sample"
DEFAULT_SHOWDOWN_ROOT = Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown")


def _showdown_root() -> Path:
    return Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)


def _available() -> bool:
    root = _showdown_root()
    return (
        numpy is not None
        and pokezero_search is not None
        and hasattr(pokezero_search, "NativeEncoder")
        and (root / "data" / "random-battles" / "gen3" / "sets.json").exists()
    )


@unittest.skipUnless(_available(), "requires numpy, the native crate, and a Showdown checkout")
class RustEncoderV4ParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(SCRIPTS))
        import export_encoder_tables
        import golden_encoder_backends

        cls.exporter = export_encoder_tables
        cls.backends = golden_encoder_backends
        cls.corpus = load_golden_corpus(SAMPLE_DIR)

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(str(SCRIPTS))

    def _fixture(self):
        header = copy.deepcopy(self.corpus.header)
        header["observation"].update(
            {
                "schema_version": OBSERVATION_SCHEMA_VERSION_V4,
                "token_count": 23,
                "categorical_feature_count": 41,
                "numeric_feature_count": 134,
            }
        )
        inputs = self.backends.row_inputs_from_decision_row(self.corpus.decision_rows[0])
        inputs["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION_V4
        metadata = inputs["observation_metadata"]
        # Every pack surface, on both seats, so a column the Rust port forgot cannot pass.
        metadata.update(
            {
                "self_must_recharge": True,
                "opponent_must_recharge": True,
                "self_truant_loaf": True,
                "opponent_truant_loaf": True,
                "self_choice_locked": True,
                "opponent_choice_locked": True,
                "self_item_swapped": True,
                "opponent_item_swapped": True,
                "self_last_damage_dealt": 0.4,
                "self_last_damage_taken": 0.2,
                "opponent_last_damage_dealt": 0.25,
                "opponent_last_damage_taken": 0.5,
                "self_last_used_move": "bodyslam",
                "opponent_last_used_move": "switch",
                "opponent_arrived_by_baton_pass": True,
                "self_traced_ability": "levitate",
                "opponent_traced_ability": "intimidate",
                # RAW Part-B ledgers. The settled column values are derived from these
                # below by the very function the encoder uses, so the fixture cannot assert a
                # number the production path would not produce.
                "self_hazard_damage_suffered": 0.36,
                "opponent_hazard_damage_suffered": 0.75,
                "self_items_removed": 1,
                "opponent_items_removed": 2,
                "opponent_matchup_switch_evidence": {},
            }
        )
        metadata["self_side_condition_counts"] = {"spikes": 2}
        metadata["opponent_side_condition_counts"] = {"spikes": 3}
        return header, inputs, metadata

    def _publish_credit_values(self, inputs, state, dex):
        """Mirror production: the encoder derives Part B once, metadata carries the result."""
        from pokezero.showdown import field_credit_values

        inputs["observation_metadata"].update(field_credit_values(state, dex=dex))

    def test_complete_v4_surface_matches_byte_for_byte(self) -> None:
        header, inputs, metadata = self._fixture()
        spec, masks = self.backends.observation_contract_from_header(header)
        self.assertEqual(spec.transition_token_count, 0)

        reference = self.backends.PythonReferenceBackend(
            showdown_root=_showdown_root(), header=header
        )
        state = self.backends.state_from_row_inputs(inputs)
        self._publish_credit_values(inputs, state, reference._dex)
        state = self.backends.state_from_row_inputs(inputs)
        observation = observation_from_player_state(
            state,
            category_vocab=reference._vocab,
            spec=spec,
            dex=reference._dex,
            feature_masks=masks,
        )
        want = self.backends.arrays_dict_from_observation_arrays(
            self.backends.GoldenObservationArrays.from_observation(observation)
        )
        self.assertEqual(want["numeric_features"].shape, (23, 134))
        self.assertEqual(want["categorical_ids"].shape, (23, 41))

        tables = self.exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4,
        )
        # The fixture must actually light up every pack column, or parity is vacuous.
        numeric_columns = tables["layout"]["numeric_columns"]
        for column_name in (
            "NUMERIC_TRUANT_LOAF",
            "NUMERIC_LAST_DAMAGE_DEALT",
            "NUMERIC_LAST_DAMAGE_TAKEN",
            "NUMERIC_CHOICE_LOCKED",
            "NUMERIC_ITEM_SWAPPED",
            "NUMERIC_SELF_HAZARD_CREDIT",
            "NUMERIC_OPP_HAZARD_CREDIT",
            "NUMERIC_SELF_HAZARD_EXPECTED",
            "NUMERIC_SELF_ITEMS_REMOVED_CREDIT",
            "NUMERIC_OPP_ITEMS_REMOVED_CREDIT",
        ):
            self.assertTrue(
                numpy.any(want["numeric_features"][:, numeric_columns[column_name]]),
                f"V4 parity fixture did not exercise {column_name}",
            )

        tables_json = json.dumps(
            tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        rust = self.backends.RustBackend(tables_json=tables_json, header=header)
        got = rust.encode(inputs)
        for name in self.backends.ARRAY_NAMES:
            self.assertEqual(
                numpy.ascontiguousarray(got[name]).tobytes(),
                numpy.ascontiguousarray(want[name]).tobytes(),
                name,
            )

    def test_the_pack_categoricals_match_too(self) -> None:
        header, inputs, metadata = self._fixture()
        tables = self.exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4,
        )
        index = tables["vocab"]["index"]
        cat_columns = tables["layout"]["categorical_columns"]
        offsets = tables["layout"]["token_offsets"]
        tables_json = json.dumps(
            tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        rust = self.backends.RustBackend(tables_json=tables_json, header=header)
        cats = numpy.ascontiguousarray(rust.encode(inputs)["categorical_ids"])
        self_active = next(
            i for i, mon in enumerate(metadata["self_team"]) if mon["active"]
        )
        opp_active = next(
            i for i, mon in enumerate(metadata["opponent_team"]) if mon["active"]
        )
        self_token = offsets["self_pokemon"] + self_active
        opp_token = offsets["opponent_pokemon"] + opp_active
        # A2: a real move on our side, the BATON PASS sentinel on theirs.
        self.assertEqual(
            cats[self_token, cat_columns["CATEGORY_LAST_USED_MOVE"]], index["move:bodyslam"]
        )
        self.assertEqual(
            cats[opp_token, cat_columns["CATEGORY_LAST_USED_MOVE"]],
            index["lastmove:batonpass"],
        )
        # A4 and A1: the current Trace copy, and mustrecharge in the volatile bag.
        self.assertEqual(
            cats[self_token, cat_columns["CATEGORY_TRACED_ABILITY"]], index["ability:levitate"]
        )
        volatile_offset = cat_columns["CATEGORY_VOLATILE_OFFSET"]
        bag = set(cats[opp_token, volatile_offset : volatile_offset + 6].tolist())
        self.assertIn(index["volatile:mustrecharge"], bag)

    def test_a_v3_row_is_refused_against_v4_tables(self) -> None:
        header, inputs, _ = self._fixture()
        tables = self.exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V4,
        )
        tables_json = json.dumps(
            tables, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        wrong = copy.deepcopy(inputs)
        wrong["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION_V3
        rust = self.backends.RustBackend(tables_json=tables_json, header=header)
        with self.assertRaisesRegex(ValueError, "does not match encoder-table layout"):
            rust.encode(wrong)


if __name__ == "__main__":
    unittest.main()
