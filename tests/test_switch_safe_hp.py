"""Pins for the switch-safe HP helper.

These encode the bug the helper exists to prevent: a per-slot HP subtraction
that silently returns a plausible number when the two readings belong to
different Pokemon.
"""

from __future__ import annotations

import unittest

from scripts.switch_safe_hp import comparable_slots, slot_hp_comparable, slot_hp_delta

STATIC = {
    "active_changed": {"p1": False, "p2": False},
    "pre_features": {"p1_hp": 100, "p2_hp": 200},
    "observed": {"p1_hp": 80, "p2_hp": 200},
}
# s18000268/37: p2 switched to Cradily, so 184 -> 239 spans two Pokemon.
SWITCHED = {
    "active_changed": {"p1": False, "p2": True},
    "pre_features": {"p1_hp": 244, "p2_hp": 184},
    "observed": {"p1_hp": 229, "p2_hp": 239},
}


class SwitchSafeHpTests(unittest.TestCase):
    def test_static_slot_is_comparable(self) -> None:
        self.assertTrue(slot_hp_comparable(STATIC, "p1"))
        self.assertEqual(slot_hp_delta(STATIC, "p1"), -20)

    def test_switched_slot_is_not_comparable(self) -> None:
        self.assertFalse(slot_hp_comparable(SWITCHED, "p2"))

    def test_switched_slot_returns_None_not_a_number(self) -> None:
        """The bug was a plausible number, not an exception. 239-184=+55 must
        never be produced: those readings are Cradily and its predecessor."""

        self.assertIsNone(slot_hp_delta(SWITCHED, "p2"))

    def test_the_unswitched_slot_of_a_switched_boundary_still_works(self) -> None:
        self.assertEqual(slot_hp_delta(SWITCHED, "p1"), -15)

    def test_comparable_slots_filters(self) -> None:
        self.assertEqual(comparable_slots(STATIC), ("p1", "p2"))
        self.assertEqual(comparable_slots(SWITCHED), ("p1",))


class FailClosedTests(unittest.TestCase):
    """A row that does not say the slot held still is not comparable.

    Review finding: the first version read a missing key as "did not change",
    so the helper produced the exact +55 cross-Pokemon delta on s18000268/37
    that the module exists to prevent.
    """

    ROW = {
        "pre_features": {"p1_hp": 244, "p2_hp": 184},
        "observed": {"p1_hp": 229, "p2_hp": 239},
    }

    def test_a_row_with_no_active_changed_map_is_not_comparable(self) -> None:
        self.assertFalse(slot_hp_comparable(self.ROW, "p2"))
        self.assertIsNone(slot_hp_delta(self.ROW, "p2"))
        self.assertEqual(comparable_slots(self.ROW), ())

    def test_a_partial_active_changed_map_does_not_cover_the_other_slot(self) -> None:
        row = dict(self.ROW, active_changed={"p1": False})
        self.assertTrue(slot_hp_comparable(row, "p1"))
        self.assertFalse(slot_hp_comparable(row, "p2"))
        self.assertIsNone(slot_hp_delta(row, "p2"))
        self.assertEqual(comparable_slots(row), ("p1",))

    def test_a_non_mapping_active_changed_is_not_comparable(self) -> None:
        for bad in (None, [], "p2", 0):
            with self.subTest(active_changed=bad):
                row = dict(self.ROW, active_changed=bad)
                self.assertFalse(slot_hp_comparable(row, "p2"))

    def test_the_fully_specified_unswitched_row_is_still_comparable(self) -> None:
        row = dict(self.ROW, active_changed={"p1": False, "p2": False})
        self.assertEqual(comparable_slots(row), ("p1", "p2"))
        self.assertEqual(slot_hp_delta(row, "p2"), 55)
