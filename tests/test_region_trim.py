"""Region-trim plumbing: config field, spec threading, and the converter's
fail-closed config transform (all torch-free; the torch-side parity gate and
benchmark live in scripts/convert_region_trim.py and run where checkpoints do).
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokezero.neural_policy import (  # noqa: E402
    TransformerPolicyConfig,
    observation_spec_from_model_config,
)
from pokezero.observation import (  # noqa: E402
    OBSERVATION_SCHEMA_VERSION_V2_2,
    OBSERVATION_SCHEMA_VERSION_V3,
    TRANSITION_TOKEN_COUNT,
    V3_TRANSITION_TOKEN_COUNT,
)

_spec = importlib.util.spec_from_file_location(
    "convert_region_trim", ROOT / "scripts" / "convert_region_trim.py"
)
convert_region_trim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_spec and convert_region_trim)
trimmed_model_config_dict = convert_region_trim.trimmed_model_config_dict


def _config(**kwargs):
    """Minimal buildable config (category_vocab is mandatory)."""
    return TransformerPolicyConfig.compact_category(category_vocab=("a", "b"), **kwargs)


class RegionFieldTests(unittest.TestCase):
    def test_sentinel_resolves_to_the_schema_region(self) -> None:
        # v2.2 default construction: region resolves to the full 128-row canvas.
        config = _config()
        self.assertEqual(config.transition_token_count, TRANSITION_TOKEN_COUNT)

    def test_sentinel_is_schema_aware_for_v3(self) -> None:
        base = _config()
        v3_spec = observation_spec_from_model_config(base)  # anchor import use
        del v3_spec
        config = _config(
            observation_schema_version=OBSERVATION_SCHEMA_VERSION_V3,
            token_count=87,
            transition_token_budget=V3_TRANSITION_TOKEN_COUNT,
        )
        self.assertEqual(config.transition_token_count, V3_TRANSITION_TOKEN_COUNT)
        self.assertEqual(config.token_count, 87)

    def test_payloads_without_the_field_parse_unchanged(self) -> None:
        # The from_dict default is the stamped schema's region — every existing
        # artifact (which lacks the field) resolves exactly as before.
        payload = _config().to_dict()
        payload.pop("transition_token_count")
        config = TransformerPolicyConfig.from_dict(payload)
        self.assertEqual(config.transition_token_count, TRANSITION_TOKEN_COUNT)
        self.assertEqual(config.token_count, config.token_count)

    def test_round_trip_preserves_a_trimmed_region(self) -> None:
        trimmed = _config(
            token_count=39, transition_token_count=16, transition_token_budget=16
        )
        self.assertEqual(
            TransformerPolicyConfig.from_dict(trimmed.to_dict()).transition_token_count, 16
        )

    def test_budget_above_region_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "transition_token_budget"):
            _config(
                token_count=39, transition_token_count=16, transition_token_budget=32
            )

    def test_region_above_schema_capacity_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "transition_token_count"):
            _config(
                token_count=23 + 256, transition_token_count=256
            )

    def test_token_count_region_mismatch_fails_loudly(self) -> None:
        # The plan's hand-edited-payload guard: trimming one field without the other.
        with self.assertRaisesRegex(ValueError, "fixed prefix"):
            _config(
                token_count=151, transition_token_count=16, transition_token_budget=16
            )

    def test_spec_threading_narrows_the_encode_spec(self) -> None:
        config = _config(
            token_count=39, transition_token_count=16, transition_token_budget=16
        )
        spec = observation_spec_from_model_config(config)
        self.assertEqual(spec.transition_token_count, 16)
        self.assertEqual(spec.token_count, 39)
        self.assertEqual(spec.schema_version, OBSERVATION_SCHEMA_VERSION_V2_2)

    def test_spec_threading_is_inert_for_existing_configs(self) -> None:
        spec = observation_spec_from_model_config(_config())
        self.assertEqual(spec.transition_token_count, TRANSITION_TOKEN_COUNT)
        self.assertEqual(spec.token_count, 151)


class ConverterTransformTests(unittest.TestCase):
    def _payload(self, *, budget: int = 16) -> dict:
        return _config(transition_token_budget=budget).to_dict()

    def test_trim_to_budget_produces_the_39_token_config(self) -> None:
        trimmed = trimmed_model_config_dict(self._payload(budget=16), 16)
        self.assertEqual(trimmed["transition_token_count"], 16)
        self.assertEqual(trimmed["token_count"], 39)
        config = TransformerPolicyConfig.from_dict(trimmed)
        self.assertEqual(config.token_count, 39)

    def test_fail_closed_when_budget_exceeds_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "fail closed"):
            trimmed_model_config_dict(self._payload(budget=32), 16)

    def test_fail_closed_when_target_exceeds_current_region(self) -> None:
        already = trimmed_model_config_dict(self._payload(budget=16), 16)
        with self.assertRaisesRegex(ValueError, "fail closed"):
            trimmed_model_config_dict(already, 64)

    def test_trim_only_touches_the_two_region_fields(self) -> None:
        payload = self._payload(budget=16)
        trimmed = trimmed_model_config_dict(payload, 16)
        changed = {k for k in payload if payload[k] != trimmed[k]}
        self.assertEqual(changed, {"transition_token_count", "token_count"})


if __name__ == "__main__":
    unittest.main()
