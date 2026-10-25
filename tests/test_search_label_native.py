"""Real model-wheel flag equivalence on committed mechanics fixtures, not games."""
import json
import unittest

from test_model_priors_search import _EncodedSearchFixture, _crate_ready
from test_rollout_model_priors import TIMING_FIELDS

try:
    import pokezero_search
except ImportError:
    pokezero_search = None


@unittest.skipUnless(_crate_ready, "requires the rebuilt model-feature native wheel")
class SearchLabelNativeIntegration(_EncodedSearchFixture, unittest.TestCase):
    def test_optional_root_statistics_preserve_search_on_both_native_sides(self):
        self.assertEqual(pokezero_search.TRAINING_ROOT_STATS_VERSION, 1)
        self.assertEqual(set(self.perspective_positions), {"side_one", "side_two"})
        for side, position in self.perspective_positions.items():
            reports = []
            for flag in (None, False, True):
                with self.subTest(side=side, flag=flag):
                    fold = pokezero_search.FoldState.from_payload(position["fold_state"])
                    kwargs = {} if flag is None else {"training_root_stats": flag}
                    report = json.loads(self.native.search_batched_multi_encoded(
                        position["state_str"], 16, 1, self.tables_json,
                        position["row_inputs"], position["ctx"], fold,
                        1, 1.4, 731, True, True, 0, side == "side_one", True,
                        **kwargs,
                    ))
                    reports.append(report)
            omitted, disabled, enabled = reports
            self.assertNotIn("training_root_stats", omitted)
            self.assertNotIn("training_root_stats", disabled)
            unchanged = lambda r: {k: v for k, v in r.items()
                                   if k not in TIMING_FIELDS and k != "training_root_stats"}
            self.assertEqual(unchanged(omitted), unchanged(disabled))
            self.assertEqual(unchanged(omitted), unchanged(enabled))
            self.assertEqual(set(enabled), set(omitted) | {"training_root_stats"})
            # The entire ordered arm report is unchanged, including tie order.
            self.assertEqual(max(omitted[side], key=lambda x: x["visits"])["move"],
                             max(enabled[side], key=lambda x: x["visits"])["move"])
            stats = enabled["training_root_stats"]
            self.assertEqual(type(stats["visits"]), int)
            self.assertEqual(stats["visits"], enabled["iterations"])
            self.assertEqual(stats["visits"], sum(x["visits"] for x in enabled[side]))
            mean = stats["side_one_value_total"] / stats["visits"]
            self.assertAlmostEqual(mean, enabled["root_value"], delta=1e-6)
            # A second check against independently reported arm Qs, whose
            # six-decimal rounding bounds the precision of this oracle.
            arm_mean = sum(x["visits"] * x["q"] for x in enabled["side_one"]) / stats["visits"]
            self.assertAlmostEqual(mean, arm_mean, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
