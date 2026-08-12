"""`schema_with()` -- select a schema by PROPERTY, never by "whatever is currently default".

The selector exists because reading the global default where a property is meant is the defect
class this whole effort is about: a test wanting a history-window budget needs a schema WITH a
transition region, does not care which, and must not silently become a v4 test (no region at all)
the day the default rotates.

These tests were absent when the selector first landed, which is its own small version of the same
problem -- a remedy advertised in a gate's error message with nothing pinning its behaviour.
"""
from __future__ import annotations

import re
import unittest

from pokezero.observation import (
    FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
    GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS,
    OBSERVATION_SCHEMA_VERSION_V2_2,
    OBSERVATION_SCHEMA_VERSION_V3,
    OBSERVATION_SCHEMA_VERSION_V4,
    REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA,
    SUPPORTED_OBSERVATION_SCHEMA_VERSIONS,
    TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
    schema_with,
)
from pokezero.showdown import observation_spec_for_schema

_PROPS = ("transition_region", "turn_merged", "grouped_layout", "feature_pack")


def _has(version: str) -> dict[str, bool]:
    return {
        "transition_region": REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA[version] > 0,
        "turn_merged": version in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
        "grouped_layout": version in GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS,
        "feature_pack": version in FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
    }


class SchemaWithTest(unittest.TestCase):
    def test_every_combination_matches_an_independent_newest_first_oracle(self) -> None:
        """All 3^4 = 81 combinations, against an oracle built here rather than from the impl."""
        checked = raised = 0
        for t in (None, True, False):
            for m in (None, True, False):
                for g in (None, True, False):
                    for f in (None, True, False):
                        want = {"transition_region": t, "turn_merged": m,
                                "grouped_layout": g, "feature_pack": f}
                        asked = {k: v for k, v in want.items() if v is not None}
                        expect = [
                            v for v in reversed(SUPPORTED_OBSERVATION_SCHEMA_VERSIONS)
                            if all(_has(v)[k] is val for k, val in asked.items())
                        ]
                        with self.subTest(**want):
                            if not asked:
                                with self.assertRaises(ValueError):
                                    schema_with(**want)
                                raised += 1
                            elif expect:
                                self.assertEqual(schema_with(**want), expect[0])
                                checked += 1
                            else:
                                with self.assertRaises(ValueError):
                                    schema_with(**want)
                                raised += 1
        # Non-vacuity: both arms must be exercised, or a selector that always raised would pass.
        self.assertGreater(checked, 0, "no combination resolved; the oracle is broken")
        self.assertGreater(raised, 0, "no combination raised; unsatisfiable sets are unchecked")
        self.assertEqual(checked + raised, 81)

    def test_it_refuses_to_be_a_spelling_of_the_current_default(self) -> None:
        """No properties must RAISE, not return the default -- that is the whole point."""
        with self.assertRaisesRegex(ValueError, "at least one property"):
            schema_with()

    def test_it_raises_on_an_unsatisfiable_set_rather_than_falling_back(self) -> None:
        # v4 is feature-packed and NOT turn-merged; nothing is both.
        with self.assertRaisesRegex(ValueError, "no supported observation schema"):
            schema_with(turn_merged=True, feature_pack=True)

    def test_the_returned_schema_actually_has_the_requested_properties(self) -> None:
        """The property that makes this safer than the default: the answer always satisfies."""
        for prop in _PROPS:
            for value in (True, False):
                with self.subTest(prop=prop, value=value):
                    try:
                        got = schema_with(**{prop: value})
                    except ValueError:
                        continue
                    self.assertIs(_has(got)[prop], value)

    def test_newest_first_is_deliberate_and_pinned(self) -> None:
        """Documented behaviour, pinned so a change to iteration order is a decision.

        `turn_merged=True` returns v3, NOT the v2.2 that held the default when the selector
        landed -- so mechanically converting a v2.2 test to `schema_with(turn_merged=True)`
        CHANGES which schema it runs under. That is a real hazard and the reason a caller whose
        subject is one version should name that version instead.
        """
        self.assertEqual(schema_with(turn_merged=True), OBSERVATION_SCHEMA_VERSION_V3)
        self.assertNotEqual(schema_with(turn_merged=True), OBSERVATION_SCHEMA_VERSION_V2_2)
        self.assertEqual(schema_with(feature_pack=True), OBSERVATION_SCHEMA_VERSION_V4)

    def test_the_supported_tuple_is_ascending_so_reversed_really_means_newest(self) -> None:
        """The whole "newest" contract rests on SUPPORTED being in ascending order.

        Nothing pinned it. Reordering the tuple left every selector test green, because the
        oracle in this file iterates the same `reversed(SUPPORTED_...)` and moves WITH the
        mutation -- a self-referential oracle. A chronologically-appended v2.3 would silently
        re-aim every property-selected caller, which is the class of silent re-aiming
        `schema_with` exists to remove.
        """
        def _key(version: str) -> tuple[int, ...]:
            tail = version.rsplit(".", 1)[-1] if version.count(".") > 2 else version.split(".")[-1]
            digits = re.findall(r"\d+", version.removeprefix("pokezero.observation.v"))
            return tuple(int(d) for d in digits) or (0,)

        keys = [_key(v) for v in SUPPORTED_OBSERVATION_SCHEMA_VERSIONS]
        self.assertEqual(
            keys, sorted(keys),
            "SUPPORTED_OBSERVATION_SCHEMA_VERSIONS is not ascending, so `reversed()` no longer "
            f"means newest-first and every schema_with() answer moves: {list(SUPPORTED_OBSERVATION_SCHEMA_VERSIONS)}",
        )

    def test_a_misspelled_constraint_is_a_TypeError_not_a_silent_drop(self) -> None:
        """Keyword-only, no **kwargs: a typo must fail loudly.

        A `**_ignored` signature survives every other test here while returning a schema that
        does not satisfy the caller's actual intent -- a quieter instance of the defect this
        function replaces.
        """
        with self.assertRaises(TypeError):
            schema_with(transition_region=True, turn_mergd=False)  # type: ignore[call-arg]

    def test_the_properties_are_keyword_only(self) -> None:
        """Positional calls must be a TypeError.

        Removing the `*` marker leaves every other test here green while allowing
        `schema_with(True)`. A later inserted property would then silently re-aim every positional
        caller -- the exact silent re-aiming this function exists to remove. The docstring, the
        design doc and the PR body all advertise "keyword-only with no **kwargs"; the **kwargs half
        was pinned and this half was not.
        """
        with self.assertRaises(TypeError):
            schema_with(True)  # type: ignore[misc]

    def test_the_transition_count_table_agrees_with_the_spec_table(self) -> None:
        """`REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA` is a SECOND copy of spec data.

        `schema_with` reads it, so a schema added to the spec table but not here makes the
        selector raise `KeyError` instead of the designed `ValueError` -- a different failure with
        a different meaning. Until it is derived, this pins the two in agreement.
        """
        for version in SUPPORTED_OBSERVATION_SCHEMA_VERSIONS:
            with self.subTest(version=version):
                self.assertIn(version, REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA)
                self.assertEqual(
                    REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA[version],
                    observation_spec_for_schema(version).transition_token_count,
                )
        self.assertEqual(
            set(REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA),
            set(SUPPORTED_OBSERVATION_SCHEMA_VERSIONS),
            "the table and the supported set have diverged",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
