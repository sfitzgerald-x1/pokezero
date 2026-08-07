"""Pins for rump-branch adjudication: a withheld verdict, not a divergence.

The matcher's contract is EXISTENTIAL -- "some enumerated branch reproduces the
observation". `strict:lossy_render` drops individual branches before comparison,
and when the branch that would have matched is one of the dropped ones, the
surviving rump set makes the existential unverifiable rather than false. Judging
it anyway converts a matched boundary into a reported divergence.

Measured instance (retained state, replayed, not re-swept): row `19200131/129`
of `reports/artifacts/c141_final_holdout_sweep.json`. Its 93.75 % non-crit arm
carries `attract_empty_tail_ambiguous:paralyzed+cannot_act` and is dropped; the
surviving 6.25 % arm is the CRIT arm, whose capped-lethal recoil of -32 was
compared against the observed non-crit recoil of -18 and reported as a
`roll_scaled_component` divergence. Allowlisting only that one marker turns the
same boundary into `matched`, so there was no divergence to report.

The numbers here are synthetic. Fitting a pin to the reserved holdout row is
exactly what the reservation exists to prevent, so the shape is reproduced and
the row is not.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import engine_transition_differential as etd  # noqa: E402
from pokezero.engine_fidelity import TurnFeatures  # noqa: E402


class _FakeState:
    """Only `to_string()` is reached: the mapper is stubbed and the roll bases
    are supplied directly, so no engine call happens in these tests."""

    def to_string(self) -> str:
        return "FAKE"


class _FakeBranchRenderer:
    def __init__(self, branches):
        self._payload = json.dumps({"branches": branches})

    def branch_events(self, *_args, **_kwargs):
        return self._payload


class _FakeEngine:
    """`calculate_damage` returns the 100 % bases for (non-crit, crit)."""

    def __init__(self, bases):
        self._bases = bases

    def calculate_damage(self, *_args, **_kwargs):
        return (list(self._bases), [0, 0])


# One recoil move into a paralysed, attracted target that cannot act. p2 takes
# 50 (base 59, a legal roll) and p1 takes floor(50 / 3) = 16 recoil.
_OBSERVED_PROTOCOL = [
    "|",
    "|move|p1a: Attacker|Double-Edge|p2a: Target",
    "|-damage|p2a: Target|50/100",
    "|-damage|p1a: Attacker|184/200|[from] Recoil|[of] p2a: Target",
    "|cant|p2a: Target|Attract",
    "|",
    "|upkeep",
    "|turn|2",
]

# The arm that actually happened: non-crit, 52 damage, floor(52 / 3) = 17
# recoil -- and rendered lossy, because the engine cannot tell an Attract
# immobilisation from a paralysis one when the tail is empty.
_MAJORITY_LOSSY_ARM = {
    "percentage": 93.75,
    "lossy": ["attract_empty_tail_ambiguous:paralyzed+cannot_act"],
    "events": [
        "|",
        "|move|p1a: Attacker|Double-Edge|p2a: Target",
        "|-damage|p2a: Target|48/100",
        "|-damage|p1a: Attacker|183/200|[from] Recoil|[of] p2a: Target",
        "|",
        "|upkeep",
        "|turn|2",
    ],
}

# The arm that survives the filter: a crit, capped at the target's remaining HP,
# so its recoil is floor(100 / 3) = 33 -- nowhere near the observed 16.
_MINORITY_CRIT_ARM = {
    "percentage": 6.25,
    "lossy": [],
    "events": [
        "|",
        "|move|p1a: Attacker|Double-Edge|p2a: Target",
        "|-crit|p2a: Target",
        "|-damage|p2a: Target|0 fnt",
        "|-damage|p1a: Attacker|167/200|[from] Recoil|[of] p2a: Target",
        "|faint|p2a: Target",
        "|",
        "|upkeep",
    ],
}


def _adjudicate(branches, *, bases=(59, 118)):
    saved_search, saved_engine = etd.pokezero_search, etd.poke_engine
    etd.pokezero_search = _FakeBranchRenderer(branches)
    etd.poke_engine = _FakeEngine(bases)
    counts: Counter = Counter()
    try:
        verdict, misses, branch_count = etd.evaluate_boundary_strict(
            states=[_FakeState()],
            slot_sides={"p1": "side_one", "p2": "side_two"},
            choices={"p1": "doubleedge", "p2": "batonpass"},
            party_display={"p1": ["Attacker"], "p2": ["Target"]},
            turn=1,
            pre_features=TurnFeatures(
                p1_hp=200, p2_hp=100, p1_status="NONE", p2_status="PARALYZE",
                fainted=frozenset(), weather="NONE", side_conditions={},
            ),
            observed=TurnFeatures(
                p1_hp=184, p2_hp=50, p1_status="NONE", p2_status="PARALYZE",
                fainted=frozenset(), weather="NONE", side_conditions={},
            ),
            step_lines=_OBSERVED_PROTOCOL,
            observed_boosts={"p1": {}, "p2": {}},
            active_changed={"p1": False, "p2": False},
            counts=counts,
        )
    finally:
        etd.pokezero_search, etd.poke_engine = saved_search, saved_engine
    return verdict, misses, branch_count, counts


class RumpBranchSetWithholdsTheVerdict(unittest.TestCase):
    def test_the_dropped_majority_arm_is_the_one_that_matches(self) -> None:
        """Control for the whole finding: with the marker allowlisted -- i.e.
        with nothing dropped -- the SAME boundary matches. If this failed, the
        rump set would not be the reason for the divergence verdict."""

        saved = etd._TELEMETRY_ONLY_LOSSY_MARKERS
        etd._TELEMETRY_ONLY_LOSSY_MARKERS = frozenset(
            set(saved) | {"attract_empty_tail_ambiguous:paralyzed+cannot_act"}
        )
        try:
            verdict, _misses, _count, counts = _adjudicate(
                [_MAJORITY_LOSSY_ARM, _MINORITY_CRIT_ARM]
            )
        finally:
            etd._TELEMETRY_ONLY_LOSSY_MARKERS = saved
        self.assertEqual(verdict, "matched")
        self.assertEqual(counts["strict:lossy_render"], 0)

    def test_a_dropped_arm_withholds_the_verdict_instead_of_diverging(self) -> None:
        verdict, misses, _count, counts = _adjudicate(
            [_MAJORITY_LOSSY_ARM, _MINORITY_CRIT_ARM]
        )
        self.assertEqual(verdict, "skip_rump")
        self.assertEqual(counts["skip:rump_branch_set"], 1)
        self.assertEqual(counts["strict:diverged_on_full_branch_set"], 0)
        # The surviving mass is IN the report, not left to be recovered by
        # replay. This number was invisible in every artifact before now.
        self.assertIn("6.25 of 100 enumerated mass", misses[0])
        self.assertEqual(counts["skip:rump_branch_set_surviving_decile:0"], 1)

    def test_the_drop_is_attributed_to_its_marker(self) -> None:
        """`strict:lossy_render` alone made the marker behind a withheld row
        recoverable only by replaying the state."""

        _verdict, _misses, _count, counts = _adjudicate(
            [_MAJORITY_LOSSY_ARM, _MINORITY_CRIT_ARM]
        )
        self.assertEqual(
            counts["strict:lossy_render_marker:attract_empty_tail_ambiguous"], 1
        )

    def test_sensitivity_is_unchanged_when_nothing_is_dropped(self) -> None:
        """The change must not buy a lower divergence count by withholding
        verdicts in general. With a fully rendered branch set that fails to
        reproduce the observation, the verdict is still `diverged`."""

        rendered_majority = {**_MAJORITY_LOSSY_ARM, "lossy": []}
        # Make the majority arm disagree on a DETERMINISTIC component so no
        # roll window can rescue it: an unexplained residual on p1.
        rendered_majority["events"] = [
            "|",
            "|move|p1a: Attacker|Double-Edge|p2a: Target",
            "|-damage|p2a: Target|48/100",
            "|-damage|p1a: Attacker|183/200|[from] Recoil|[of] p2a: Target",
            "|-damage|p1a: Attacker|175/200|[from] psn",
            "|",
            "|upkeep",
            "|turn|2",
        ]
        verdict, misses, _count, counts = _adjudicate(
            [rendered_majority, _MINORITY_CRIT_ARM]
        )
        self.assertEqual(verdict, "diverged")
        self.assertEqual(counts["skip:rump_branch_set"], 0)
        self.assertEqual(counts["strict:diverged_on_full_branch_set"], 1)
        self.assertTrue(misses)

    def test_every_arm_lossy_still_takes_the_older_all_lossy_exit(self) -> None:
        """`skip_lossy` predates this change and keeps precedence: it is the
        stronger statement (nothing was rendered at all)."""

        verdict, misses, _count, counts = _adjudicate(
            [_MAJORITY_LOSSY_ARM, {**_MINORITY_CRIT_ARM, "lossy": ["some_future_marker"]}]
        )
        self.assertEqual(verdict, "skip_lossy")
        self.assertEqual(counts["skip:rump_branch_set"], 0)
        self.assertEqual(misses, ["every branch rendered lossy"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
