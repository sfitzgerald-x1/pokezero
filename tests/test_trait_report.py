"""Guards for the report's conditional-rate math (scripts/trait_report.py)."""
import importlib.util
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trait_report.py"


def _load():
    spec = importlib.util.spec_from_file_location("trait_report", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TR = _load()


class ConditionalRates(unittest.TestCase):
    def _row(self, extras, cat_totals):
        return {"move_category_extras": extras,
                "move_categories": {k: {"total_uses": v} for k, v in cat_totals.items()}}

    def test_rate_cannot_exceed_100_when_move_used_but_not_carried(self):
        # The real failure: extras count every use, but move_categories.total_uses is gated on the
        # seat carrying the move. A Solar Beam used via Metronome/Mimic lifts sun without total,
        # which previously yielded 100.4%. Ungated pairs make that impossible.
        r = self._row({"cat_solarbeam_sun": 272, "cat_solarbeam_nosun": 2},
                      {"cat_solarbeam": 271})           # gated denominator is SMALLER than sun
        pct = TR._fracpct2(r, "cat_solarbeam_sun", "cat_solarbeam_sun", "cat_solarbeam_nosun")
        self.assertLessEqual(pct, 100.0)
        self.assertAlmostEqual(pct, 100 * 272 / 274, places=6)

    def test_phaze_justified_uses_ungated_pair(self):
        r = self._row({"cat_phaze_justified": 46, "cat_phaze_neutral": 22}, {"cat_phaze": 60})
        pct = TR._fracpct2(r, "cat_phaze_justified", "cat_phaze_justified", "cat_phaze_neutral")
        self.assertAlmostEqual(pct, 100 * 46 / 68, places=6)
        self.assertLessEqual(pct, 100.0)

    def test_zero_denominator_is_none_not_a_crash(self):
        r = self._row({"cat_solarbeam_sun": 0, "cat_solarbeam_nosun": 0}, {"cat_solarbeam": 0})
        self.assertIsNone(TR._fracpct2(r, "cat_solarbeam_sun", "cat_solarbeam_sun", "cat_solarbeam_nosun"))


if __name__ == "__main__":
    unittest.main()
