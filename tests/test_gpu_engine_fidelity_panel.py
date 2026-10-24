"""Authored accounting tests, not fidelity transitions or pool credit."""
from fractions import Fraction
from types import SimpleNamespace
import unittest

from gpu_engine import fidelity_panel as panel
from gpu_engine import collect_fidelity_panel as collector
from gpu_engine import future_state as mapping
from gpu_engine.finite_law import CardinalityRefusal


def future(value):
    return mapping.CanonicalFuture((("field", value),), None)


class FixedPlan:
    cardinality = 2
    def bounded_cases(self):
        return iter((SimpleNamespace(mass=Fraction(1, 2)),) * 2)


class PanelTests(unittest.TestCase):
    def compare(self, execute=lambda rows: (future(1),) * len(rows),
                native=lambda: [(1., future(1))], plan=None):
        return panel.compare_slot(plan or FixedPlan(), execute, native,
            batch_size=1, mass_tolerance=1e-5, probability_tolerance=1e-5)

    def test_complete_agreement_and_raw_representatives(self):
        result = self.compare()
        self.assertEqual(result["status"], "AGREE")
        self.assertEqual(result["exact_cases"], 2)
        self.assertEqual(result["outcomes"][0]["jax_probability"], "1")
        self.assertEqual(set(result["outcomes"][0]["raw_representatives"]), {"jax", "native"})

    def test_support_difference_is_not_excused_by_tolerance(self):
        result = self.compare(native=lambda: [(1e-7, future(2)), (1.-1e-7, future(1))])
        self.assertEqual(result["status"], "DIVERGE")
        self.assertFalse(result["support_equal"])
        self.assertLess(result["max_probability_difference"], 1e-5)

    def test_hashable_refusal_is_whole_slot_unknown(self):
        refused = mapping.Refused((mapping.Gap("pending", "SUSPENDED", "raw retained"),))
        result = self.compare(execute=lambda rows: (refused,))
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["unknown_mass"], "1")
        self.assertNotIn("outcomes", result)

    def test_late_refusal_never_accepts_prefix(self):
        count = 0
        def execute(rows):
            nonlocal count
            count += 1
            if count == 2:
                raise panel.WholeSlotUnknown("UNSUPPORTED", "second case")
            return (future(1),)
        result = self.compare(execute=execute)
        self.assertEqual(result["completed_cases"], 1)
        self.assertEqual(result["identified_mass"], "0")
        self.assertFalse(result["partial_laws_accepted"])

    def test_cap_admission_precedes_both_engines(self):
        class OverCap(FixedPlan):
            def bounded_cases(self):
                raise CardinalityRefusal(8, 4)
        def forbidden(*args):
            self.fail("transition callback called before admission")
        result = self.compare(forbidden, forbidden, OverCap())
        self.assertEqual(result["category"], "CASE_CAP")
        self.assertEqual(result["completed_cases"], 0)

    def test_input_refusal_does_not_execute(self):
        refusal = mapping.Refused((mapping.Gap("input", "UNSUPPORTED", "input"),))
        result = self.compare(plan=refusal)
        self.assertEqual(result["category"], "INPUT_OR_PLAN_REFUSED")

    def test_bad_native_mass_is_not_renormalized(self):
        result = self.compare(native=lambda: [(.7, future(1))])
        self.assertEqual(result["category"], "INVALID_NATIVE_MASS")
        self.assertNotIn("outcomes", result)

    def test_zero_native_branch_is_counted_without_mapping(self):
        result = self.compare(native=lambda: [(0., None), (1., future(1))])
        self.assertEqual(result["native_zero_branches"], 1)

    def test_executor_dropping_cases_is_fatal(self):
        with self.assertRaisesRegex(RuntimeError, "dropped"):
            self.compare(execute=lambda rows: ())

    def test_unexpected_runtime_failure_is_not_an_expected_refusal(self):
        def broken(rows):
            raise RuntimeError("kernel failed")
        with self.assertRaisesRegex(RuntimeError, "kernel failed"):
            self.compare(execute=broken)

    def test_nonfinite_native_probability_unknown(self):
        result = self.compare(native=lambda: [(float("nan"), future(1))])
        self.assertEqual(result["category"], "INVALID_NATIVE_PROBABILITY")

    def test_fixed_thousands_roster_has_no_team_reuse_or_redraw(self):
        rows = collector.source_rows(4096, 170000, 5)
        self.assertEqual(len(rows), 4096)
        self.assertEqual(rows[-1]["teams"], [178190, 178191])
        self.assertEqual(len({team for row in rows for team in row["teams"]}), 8192)
        self.assertTrue(all(row["prefix_requests"] == i % 5 for i, row in enumerate(rows)))

    def test_rational_certificate_serialization_is_explicit(self):
        self.assertEqual(collector.encoded(dict(threshold=Fraction(95, 3))), b'{"threshold":"95/3"}')

    def test_request_labels_do_not_credit_ignored_zero_or_missing_action(self):
        raw = dict(active=[0,0], moves=[[[1,0,0,0]], [[1,0,0,0]]], charge_move=[[0],[0]])
        labels = collector.request_labels(raw, [0,-1], [False,True], [[True]*4]*2, ["none","tackle"])
        self.assertEqual([x["kind"] for x in labels], ["IGNORED","MISSING_LEGAL_ACTION"])
        self.assertTrue(all(x["effective_move"] is None for x in labels))

    def test_request_labels_distinguish_struggle_charge_and_selected_slot(self):
        raw = dict(active=[0,0], moves=[[[1,0,0,0]], [[1,0,0,0]]], charge_move=[[0],[2]])
        labels = collector.request_labels(raw, [0,0], [True,True], [[False]*4,[True]*4],
                                          ["none","tackle","solarbeam"])
        self.assertEqual(labels[0], dict(kind="SYNTHETIC_STRUGGLE",selected_move=None,effective_move="struggle"))
        self.assertEqual(labels[1], dict(kind="CHARGE_RELEASE",selected_move="tackle",effective_move="solarbeam"))


if __name__ == "__main__":
    unittest.main()
