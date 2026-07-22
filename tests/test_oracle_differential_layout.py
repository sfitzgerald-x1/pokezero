"""Schema-offset checks for the standalone oracle differential."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from pokezero.observation import PokeZeroObservationV0
from pokezero.showdown import (
    NUMERIC_SELF_SCREENS,
    NUMERIC_TOXIC_STAGE,
    V3_REPLAY_OBSERVATION_SPEC,
    numeric_index_for_schema,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "oracle_differential.py"


def _load_oracle():
    spec = importlib.util.spec_from_file_location("oracle_differential_test", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib invariant
        raise RuntimeError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OracleDifferentialLayoutTest(unittest.TestCase):
    def test_semantic_row_maps_reordered_and_dropped_v3_columns(self) -> None:
        module = _load_oracle()
        spec = V3_REPLAY_OBSERVATION_SPEC
        numeric = [[0.0] * spec.numeric_feature_count for _ in range(spec.token_count)]
        physical = numeric_index_for_schema(spec.schema_version, NUMERIC_TOXIC_STAGE)
        numeric[1][physical] = 0.4
        numeric[1][NUMERIC_TOXIC_STAGE] = 0.8
        observation = PokeZeroObservationV0(
            categorical_ids=tuple(
                (0,) * spec.categorical_feature_count for _ in range(spec.token_count)
            ),
            numeric_features=tuple(tuple(row) for row in numeric),
            token_type_ids=(0,) * spec.token_count,
            attention_mask=(False,) * spec.token_count,
            legal_action_mask=(False,) * 9,
            schema_version=spec.schema_version,
        )

        self.assertEqual(
            module._SchemaNumericRow(observation, 1)[NUMERIC_TOXIC_STAGE], 0.4
        )
        self.assertIsNone(module._numeric_if_present(observation, 0, NUMERIC_SELF_SCREENS))


if __name__ == "__main__":
    unittest.main()
