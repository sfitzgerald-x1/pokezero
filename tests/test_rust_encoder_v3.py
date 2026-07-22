"""Bit-exact Python/Rust parity for the complete V3 observation surface."""

from __future__ import annotations

import copy
from dataclasses import replace
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
from pokezero.observation import (
    OBSERVATION_SCHEMA_VERSION_V2_2,
    OBSERVATION_SCHEMA_VERSION_V3,
)
from pokezero.showdown import observation_from_player_state
from pokezero.transitions_fold import FoldState


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
SAMPLE_DIR = REPO_ROOT / "tests" / "data" / "golden_corpus_sample"
DEFAULT_SHOWDOWN_ROOT = Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown")


def _showdown_root() -> Path:
    return Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)


def _available() -> bool:
    root = _showdown_root()
    return bool(
        numpy is not None
        and pokezero_search is not None
        and hasattr(pokezero_search, "NativeEncoder")
        and (root / "dist" / "sim" / "index.js").exists()
        and SAMPLE_DIR.exists()
    )


@unittest.skipUnless(_available(), "requires numpy, native encoder, and built Showdown")
class RustEncoderV3ParityTest(unittest.TestCase):
    PROTOCOL_LINES = (
        "|switch|p1a: Snorlax|Snorlax, L91, M|100/100",
        "|switch|p2a: Machamp|Machamp, L82, F|100/100",
        "|turn|1",
        "|move|p1a: Snorlax|Body Slam|p2a: Machamp",
        "|-damage|p2a: Machamp|70/100",
        "|-activate|p2a: Machamp|confusion",
        "|-damage|p2a: Machamp|60/100",
        "|turn|2",
        "|move|p1a: Snorlax|Body Slam|p2a: Machamp",
        "|-damage|p2a: Machamp|30/100",
        "|move|p2a: Machamp|Toxic|p1a: Snorlax",
        "|-fail|p1a: Snorlax",
        "|turn|3",
        "|move|p1a: Snorlax|Toxic|p2a: Machamp",
        "|-fail|p2a: Machamp",
        "|move|p2a: Machamp|Cross Chop|p1a: Snorlax",
        "|-damage|p1a: Snorlax|55/100",
        "|turn|4",
    )

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

    def test_complete_v3_surface_matches_byte_for_byte(self) -> None:
        header = copy.deepcopy(self.corpus.header)
        header["observation"].update(
            {
                "schema_version": OBSERVATION_SCHEMA_VERSION_V3,
                "token_count": 151,
                "categorical_feature_count": 51,
                "numeric_feature_count": 155,
            }
        )
        inputs = self.backends.row_inputs_from_decision_row(self.corpus.decision_rows[0])
        inputs["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION_V3
        metadata = inputs["observation_metadata"]
        metadata.update(
            {
                "self_sleep_clause_blocks": True,
                "opponent_sleep_clause_blocks": True,
                "self_wish_turns": 2,
                "opponent_wish_turns": 1,
                "self_stall_counter": 2,
                "opponent_stall_counter": 3,
                "self_confusion_elapsed": 2,
                "opponent_confusion_elapsed": 4,
                "self_encore_elapsed": 3,
                "opponent_encore_elapsed": 5,
                "self_wrap_trap_elapsed": 2,
                "opponent_wrap_trap_elapsed": 4,
                "self_meanlook_trap": True,
                "opponent_meanlook_trap": True,
            }
        )
        for index, mon in enumerate(metadata["self_team"]):
            mon["details"] = f"{mon['species']}, L80, {'M' if index % 2 == 0 else 'F'}"
        for index, mon in enumerate(metadata["opponent_team"]):
            mon["details"] = f"{mon['species']}, L80, {'F' if index % 2 == 0 else 'M'}"
        next(mon for mon in metadata["self_team"] if mon["active"])[
            "live_type_source"
        ] = "type:Fire"

        spec, masks = self.backends.observation_contract_from_header(header)
        reference = self.backends.PythonReferenceBackend(
            showdown_root=_showdown_root(), header=header
        )
        state = self.backends.state_from_row_inputs(inputs)
        fold, products = FoldState.initial(
            perspective_slot=metadata["showdown_slot"]
        ).advance(self.PROTOCOL_LINES)
        state = replace(
            state,
            transition_tokens=products.transition_tokens,
            turn_merged_tokens=products.turn_merged_tokens,
            tendency_stats=products.tendency_stats,
        )
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

        tables = self.exporter.build_tables(
            str(_showdown_root()),
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V3,
        )
        numeric_columns = tables["layout"]["numeric_columns"]
        for column_name in (
            "NUMERIC_TT_FAIL",
            "NUMERIC_TM2_FAIL",
            "NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF",
            "NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP",
            "NUMERIC_STALL_COUNTER",
            "NUMERIC_CONFUSION_TURNS",
            "NUMERIC_ENCORE_TURNS",
            "NUMERIC_WRAP_TRAP_TURNS",
            "NUMERIC_GENDER_MALE",
            "NUMERIC_GENDER_FEMALE",
            "NUMERIC_MEANLOOK_TRAP",
            "NUMERIC_SELF_WISH_TURNS",
            "NUMERIC_OPP_WISH_TURNS",
            "NUMERIC_TT_CONFUSION_SELFHIT",
        ):
            self.assertTrue(
                numpy.any(want["numeric_features"][:, numeric_columns[column_name]]),
                f"V3 parity fixture did not exercise {column_name}",
            )
        active_self_slot = next(
            index for index, mon in enumerate(metadata["self_team"]) if mon["active"]
        )
        active_self_token = (
            tables["layout"]["token_offsets"]["self_pokemon"] + active_self_slot
        )
        type_column = tables["layout"]["categorical_columns"]["CATEGORY_TYPE_1"]
        self.assertEqual(
            want["categorical_ids"][active_self_token, type_column],
            tables["vocab"]["index"]["type:fire"],
        )

        tables_json = json.dumps(
            tables,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        wrong_schema_inputs = copy.deepcopy(inputs)
        wrong_schema_inputs["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION_V2_2
        with self.assertRaisesRegex(ValueError, "does not match encoder-table layout"):
            pokezero_search.NativeEncoder(tables_json).encode(
                json.dumps(wrong_schema_inputs, sort_keys=True)
            )
        rust = self.backends.RustFoldBackend(tables_json=tables_json, header=header)
        got = rust.encode_with_fold(inputs, fold.to_payload())

        for name in self.backends.ARRAY_NAMES:
            self.assertEqual(
                numpy.ascontiguousarray(got[name]).tobytes(),
                numpy.ascontiguousarray(want[name]).tobytes(),
                name,
            )

        # The real protocol places the confusion correction on the first mover in
        # practice. Exercise the mirrored second-sub-block encoder defensively by
        # round-tripping a valid fold payload with that additive marker moved there.
        second_payload = copy.deepcopy(fold.to_payload())
        second = second_payload["merged_done"][2]["token"]["second"]
        second["damage_fraction"] = 0.1
        second["confusion_selfhit"] = True
        second["confusion_selfhit_fraction"] = 0.1
        second_fold = FoldState.from_payload(second_payload)
        second_products = second_fold.products()
        second_state = replace(
            state,
            transition_tokens=second_products.transition_tokens,
            turn_merged_tokens=second_products.turn_merged_tokens,
            tendency_stats=second_products.tendency_stats,
        )
        second_observation = observation_from_player_state(
            second_state,
            category_vocab=reference._vocab,
            spec=spec,
            dex=reference._dex,
            feature_masks=masks,
        )
        second_want = self.backends.arrays_dict_from_observation_arrays(
            self.backends.GoldenObservationArrays.from_observation(second_observation)
        )
        second_got = rust.encode_with_fold(inputs, second_payload)
        for name in self.backends.ARRAY_NAMES:
            self.assertEqual(
                numpy.ascontiguousarray(second_got[name]).tobytes(),
                numpy.ascontiguousarray(second_want[name]).tobytes(),
                f"second-sub-block {name}",
            )


if __name__ == "__main__":
    unittest.main()
