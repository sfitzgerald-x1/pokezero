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

    def test_a_MINORITY_drop_also_withholds_the_verdict(self) -> None:
        """The mass-blindness pin, and the one a threshold would fail.

        A mutant gated on `dropped_mass > 50.0` passes every other test in this
        file, because the measured row dropped 93.75 %. Here the dropped arm is
        6.25 % and the surviving arm is 93.75 %: still uncompared, still possibly
        the arm that matched, so still withheld. The property is "the enumeration
        was incomplete", not "most of it was".
        """

        minority_dropped = {**_MINORITY_CRIT_ARM,
                            "lossy": ["attract_empty_tail_ambiguous:paralyzed"]}
        majority_rendered = {**_MAJORITY_LOSSY_ARM, "lossy": []}
        # Make the surviving majority arm miss, so nothing reproduces the
        # observation and the exit is reached at all.
        majority_rendered["events"] = [
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
            [majority_rendered, minority_dropped]
        )
        self.assertEqual(verdict, "skip_rump")
        self.assertEqual(counts["skip:rump_branch_set"], 1)
        self.assertEqual(counts["strict:diverged_on_full_branch_set"], 0)
        self.assertIn("93.75 of 100 enumerated mass", misses[0])
        self.assertEqual(counts["skip:rump_branch_set_surviving_decile:9"], 1)

    def test_a_branch_whose_support_cannot_be_priced_is_also_a_drop(self) -> None:
        """`BranchLegalRollError` leaves a positive-mass branch uncompared just as
        the lossy filter does. It did not feed the drop accounting, so
        `diverged_on_full_branch_set` could have fired claiming 100 % of a mass it
        had not compared. Fail-closed guard: neither this counter nor
        `strict:branch_events_error` appears in ANY artifact in
        `reports/artifacts/`, so nothing live moves."""

        # `branch_event_legal_rolls` raises when `events` is not a sequence.
        unpriceable = {**_MINORITY_CRIT_ARM, "events": "not-a-list"}
        majority_rendered = {**_MAJORITY_LOSSY_ARM, "lossy": []}
        majority_rendered["events"] = [
            "|",
            "|move|p1a: Attacker|Double-Edge|p2a: Target",
            "|-damage|p2a: Target|48/100",
            "|-damage|p1a: Attacker|183/200|[from] Recoil|[of] p2a: Target",
            "|-damage|p1a: Attacker|175/200|[from] psn",
            "|",
            "|upkeep",
            "|turn|2",
        ]
        verdict, _misses, _count, counts = _adjudicate(
            [majority_rendered, unpriceable]
        )
        self.assertEqual(verdict, "skip_rump")
        self.assertEqual(counts["strict:diverged_on_full_branch_set"], 0)
        self.assertTrue(
            any(k.startswith("strict:branch_event_legal_error") for k in counts)
        )

    def test_every_arm_lossy_still_takes_the_older_all_lossy_exit(self) -> None:
        """`skip_lossy` predates this change and keeps precedence: it is the
        stronger statement (nothing was rendered at all)."""

        verdict, misses, _count, counts = _adjudicate(
            [_MAJORITY_LOSSY_ARM, {**_MINORITY_CRIT_ARM, "lossy": ["some_future_marker"]}]
        )
        self.assertEqual(verdict, "skip_lossy")
        self.assertEqual(counts["skip:rump_branch_set"], 0)
        self.assertEqual(misses, ["every branch rendered lossy"])


class WithheldRowsStayReplayable(unittest.TestCase):
    """A withheld row must carry its state, and must not disturb `repros`.

    The row that motivated the whole exit, 19200131/129, was diagnosable ONLY
    because it happened to be retained in `repros` as a divergence. On a reserved
    window, re-running its seed to recover the state IS the forbidden
    measurement, so a withheld row holding nothing but a counter key would be
    permanently undiagnosable. The first version of this patch did exactly that.
    """

    # Exactly what scripts/cert_sweep_reread.py's `reread_row` consumes to
    # rebuild the matcher inputs. Named here so a future trim of the payload
    # fails loudly instead of quietly making withheld rows unreplayable.
    REPLAY_FIELDS = frozenset({
        "engine_states", "choices", "pre_features", "observed", "protocol",
        "observed_boost_deltas", "active_changed",
    })

    def _payload(self) -> dict:
        """The withheld payload `run_game` writes, built from the same sources."""

        return {
            "kind": "verdict_withheld_rump_branch_set",
            "seed": 19000001,
            "step": 7,
            "choices": {"p1": "doubleedge", "p2": "batonpass"},
            "engine_state": "FAKE",
            "engine_states": ["FAKE"],
            "gating": "exact",
            "party_display": {"p1": ["Attacker"], "p2": ["Target"]},
            "slot_sides": {"p1": "side_one", "p2": "side_two"},
            "turn": 1,
            "pre_features": {
                "p1_hp": 200, "p2_hp": 100, "p1_status": "NONE",
                "p2_status": "PARALYZE", "fainted": [], "weather": "NONE",
                "side_conditions": {"p1": {}, "p2": {}},
            },
            "observed": {
                "p1_hp": 184, "p2_hp": 50, "p1_status": "NONE",
                "p2_status": "PARALYZE", "fainted": [], "weather": "NONE",
                "side_conditions": {"p1": {}, "p2": {}},
            },
            "observed_boost_deltas": {"p1": {}, "p2": {}},
            "active_changed": {"p1": False, "p2": False},
            "branch_count": 2,
            "withheld_misses": ["rump branch set: 6.25 of 100 ..."],
            "protocol": list(_OBSERVED_PROTOCOL),
        }

    def test_the_payload_carries_every_field_a_replay_needs(self) -> None:
        self.assertLessEqual(self.REPLAY_FIELDS, set(self._payload()))

    def test_the_payload_re_adjudicates_to_the_same_withheld_verdict(self) -> None:
        """Round trip: rebuild the matcher inputs from the payload ALONE and get
        the same verdict back. This is the property "replayable" actually means."""

        row = self._payload()
        saved_search, saved_engine = etd.pokezero_search, etd.poke_engine
        etd.pokezero_search = _FakeBranchRenderer(
            [_MAJORITY_LOSSY_ARM, _MINORITY_CRIT_ARM]
        )
        etd.poke_engine = _FakeEngine((59, 118))
        counts: Counter = Counter()
        try:
            verdict, misses, _branches = etd.evaluate_boundary_strict(
                states=[_FakeState() for _ in row["engine_states"]],
                slot_sides=row["slot_sides"],
                choices=row["choices"],
                party_display=row["party_display"],
                turn=row["turn"],
                pre_features=TurnFeatures(
                    **{**row["pre_features"],
                       "fainted": frozenset(row["pre_features"]["fainted"])}),
                observed=TurnFeatures(
                    **{**row["observed"],
                       "fainted": frozenset(row["observed"]["fainted"])}),
                step_lines=row["protocol"],
                observed_boosts=row["observed_boost_deltas"],
                active_changed=row["active_changed"],
                counts=counts,
            )
        finally:
            etd.pokezero_search, etd.poke_engine = saved_search, saved_engine
        self.assertEqual(verdict, "skip_rump")
        self.assertIn("6.25 of 100 enumerated mass", misses[0])

    def test_the_divergence_retention_contract_is_untouched(self) -> None:
        """`repros_retained == transitions_diverged` is what cert_sweep_reread.py
        reads. The withheld population is declared beside it, never inside it."""

        record = runner_module().checkpoint_record(
            seed=19000001,
            counts={
                "boundaries_measured": 3,
                "transition:matched": 2,
                "transition:diverged": 0,
                "skip:rump_branch_set": 1,
            },
            repros=[],
            seconds=0.1,
            build_check="gated",
            provenance={"source_commit": "a" * 40, "engine_fingerprint": "b" * 64},
            withheld_repros=[self._payload()],
        )
        report = runner_module().build_report(
            [record], elapsed=1.0, approximate_sleep=False, matcher="strict",
            keep_repro=25,
        )
        retention = report["repro_retention"]
        self.assertEqual(retention["repros_retained"], 0)
        self.assertEqual(retention["transitions_diverged"], 0)
        self.assertTrue(retention["repros_complete"])
        self.assertEqual(report["repros"], [])
        # ... and the withheld row is still there, with its state.
        self.assertEqual(retention["verdicts_withheld"], 1)
        self.assertEqual(retention["withheld_retained"], 1)
        self.assertTrue(retention["withheld_complete"])
        self.assertEqual(len(report["withheld_repros"]), 1)
        self.assertLessEqual(self.REPLAY_FIELDS, set(report["withheld_repros"][0]))

    def test_a_report_that_drops_the_withheld_rows_fails_certification(self) -> None:
        mod = runner_module()
        record = mod.checkpoint_record(
            seed=19000001, counts={"skip:rump_branch_set": 1}, repros=[],
            seconds=0.1, build_check="gated",
            provenance={"source_commit": "a" * 40, "engine_fingerprint": "b" * 64},
            withheld_repros=[self._payload()],
        )
        report = mod.build_report(
            [record], elapsed=1.0, approximate_sleep=False, matcher="strict",
            keep_repro=25,
        )
        self.assertEqual(mod.checkpoint_report_binding_failures([record], report), [])
        report["withheld_repros"] = []
        failures = mod.checkpoint_report_binding_failures([record], report)
        self.assertIn(
            "report withheld_repros does not match the checkpoint aggregate", failures
        )

    def test_a_pre_C142_REPORT_without_the_key_still_certifies(self) -> None:
        """The compatibility half, and a break this patch actually introduced.

        Every certification report on disk predates `withheld_repros`. Demanding
        the key outright made all of them fail
        `checkpoint_report_binding_failures` — caught by
        `tests/test_cert_sweep_readout_contract.py` and
        `tests/test_cert_execution_manifest.py`, which build report fixtures in
        the old shape. Absent-and-empty is consistent; only a DROPPED non-empty
        population is a mismatch, which the test above pins.
        """

        mod = runner_module()
        record = mod.checkpoint_record(
            seed=19000001, counts={"transition:matched": 1}, repros=[],
            seconds=0.1, build_check="gated",
            provenance={"source_commit": "a" * 40, "engine_fingerprint": "b" * 64},
        )
        report = mod.build_report(
            [record], elapsed=1.0, approximate_sleep=False, matcher="strict",
            keep_repro=25,
        )
        # Strip the report back to the pre-C142 shape.
        del report["withheld_repros"]
        del report["repro_retention"]["withheld_retained"]
        self.assertEqual(mod.checkpoint_report_binding_failures([record], report), [])

    def test_a_pre_C142_record_without_the_key_still_aggregates(self) -> None:
        """`--resume` and `--merge-from` run over the existing checkpoint
        archive, whose records predate this key. Absent reads as empty;
        MALFORMED still raises."""

        mod = runner_module()
        legacy = {
            "schema": mod.CHECKPOINT_SCHEMA,
            "build_check": "gated",
            "provenance": {"source_commit": "a" * 40, "engine_fingerprint": "b" * 64},
            "seed": 19000001,
            "seconds": 0.1,
            "counters": {"boundaries_measured": 1, "transition:matched": 1},
            "repros": [],
        }
        aggregate = mod.checkpoint_report_aggregate([legacy], keep_repro=25)
        self.assertEqual(aggregate["withheld_repros"], [])
        report = mod.build_report(
            [legacy], elapsed=1.0, approximate_sleep=False, matcher="strict",
            keep_repro=25,
        )
        self.assertEqual(report["withheld_repros"], [])
        self.assertEqual(mod.checkpoint_report_binding_failures([legacy], report), [])
        with self.assertRaises(ValueError):
            mod.checkpoint_report_aggregate(
                [{**legacy, "withheld_repros": "not-a-list"}], keep_repro=25
            )


def runner_module():
    """`engine_transition_differential` imported as a module object.

    It is already on `sys.path` as `engine_transition_differential`; this exists
    so the checkpoint-schema tests read like the ones in
    `tests/test_engine_transition_checkpoint_provenance.py`.
    """

    return etd


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
