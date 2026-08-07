"""The enumeration oracle as a standing gate on the collapsed roll partition — C138.

WHY THIS EXISTS, in one sentence: three hand-derived mass recipes in this family have
already been wrong, every one of them was mass-conserving, and no instrument this
repository owned could see any of them. ``test_masses_sum_to_one`` cannot — the
totals were correct. The transition differential cannot — it compares roll-scaled
*components*, never branch *masses*. ``tests/test_branch_mass_reconstruction.py``
gets closest, but it compares a single scalar (total KO mass) against a
reconstruction whose residual tick is read at PRE-move HP, and it has no notion of a
secondary at all — which is exactly the axis cause A8 lives on. Adversarial review
had to substitute for the instrument three separate times.

WHAT IS NEW. ``third_party/poke-engine-gen3-enumerate-damage-rolls.patch`` builds an
engine that emits one arm per distinct ``floor(max * r / 100)`` for ``r`` in
``85..=100`` at mass 1/16 and resolves lethality, secondaries and the ordered
residual phase inside ``run_move`` rather than in a mirror. It is exact where the
collapsed path is an approximation, so it is a usable oracle for what the collapsed
path is approximating.

THE FUNCTIONAL IS OUTCOME MASS. Read this before changing anything here. A correct
collapsed path **cannot** agree with enumerated truth arm-for-arm; that is what
collapsing means. The comparison is a coarsening: total probability mass per
``(defender faints?, defender's end status)`` cell. That is the functional the
spike's own A8 demonstration uses — enumerated 5.810547 % against independent truth
5.810547 %, delta 0, where the collapsed path gives 5.312500 % — and the one the
disjoint-band recipe is exact on. Do not "strengthen" this into an arm comparison.
It can never pass, and the only way back to green would be to delete the test.

THREE-WAY, SO A WRONG ORACLE CANNOT BLESS A WRONG RECIPE (c137 §4's open item).
``tests/data/collapsed_arm_mass_oracle.json`` is a PIN, produced out-of-process
against the enumerating build; :func:`reconstruct_outcome_masses` is an independent
pure-Python enumeration; the shipping engine is the thing under test. All three must
agree. The pin exists because ``ENUMERATE_DAMAGE_ROLLS`` is a ``OnceLock`` read from
the environment on first call — one process is one engine, permanently — so a test
that tried to flip the flag mid-run would silently compare the shipping engine
against itself.

WHAT THIS CANNOT SEE, stated rather than discovered later. The reconstruction shares
``calculate_damage``'s VALUE and the residual phase's VERDICT with the engine, so a
wrong damage formula or a wrong residual magnitude passes here; the pinned oracle
shares them too, being the same engine family. What is independent is everything the
partition does — the roll enumeration, the per-roll classification, the secondary
composition and the mass formula. It also says nothing about arms the functional
cannot distinguish: two recipes that move mass between arms with the same
``(faints, status)`` outcome are indistinguishable to it.

THE IMPORT IS DELIBERATELY HARD. Do not add ``try/except ImportError`` and a skip.
A gate that skips when the wheel is missing is how six of the previous era's
fixtures read PASS while asserting nothing.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import poke_engine as pe  # noqa: E402

from collapsed_arm_mass_oracle import (  # noqa: E402
    FIXTURES,
    PARTITION_SITES,
    build_state,
    outcome_masses,
    reconstruct_outcome_masses,
    _residual_kills,
)

ORACLE_PATH = ROOT / "tests" / "data" / "collapsed_arm_mass_oracle.json"

#: The engine accumulates branch percentages in f32, so an exactly-correct arm set
#: still lands a few 1e-6 away from the rational answer. This is the same tolerance
#: ``tests/test_branch_mass_reconstruction.py`` uses; it is three orders of
#: magnitude below every disagreement this file was written to catch (the smallest
#: is 1.41 points, on ``crit-straddle-sand``).
DELTA = 0.001


def _threshold(fixture, status: str) -> int | None:
    """Smallest damage that makes the residual phase lethal, or ``None``.

    Derived by probing the phase itself over the whole HP range rather than by
    reimplementing the mirror, so it is a description of the fixture rather than a
    second copy of the code under test.

    Bounded strictly below ``hp``: a damage of exactly ``hp`` kills on the HIT, and
    counting that as residual lethality made every fixture report a threshold and
    the A8 "the mirror must decline" assertion vacuously true.
    """

    for damage in range(0, fixture.hp):
        if _residual_kills(fixture, fixture.hp - damage, status):
            return damage
    return None


def _fan(maximum: int) -> list[int]:
    return sorted({maximum * roll // 100 for roll in range(85, 101)})


class CollapsedArmMassOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.oracle = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))

    def test_the_shipping_engine_agrees_with_the_enumerated_oracle_on_outcome_mass(self) -> None:
        """THE GATE. Functional: mass per (faints, end status) cell. See the header."""

        pinned = self.oracle["outcome_masses"]
        for fixture in FIXTURES:
            with self.subTest(case=fixture.label):
                truth = pinned[fixture.label]
                measured = outcome_masses(fixture)
                for cell in sorted(set(truth) | set(measured)):
                    self.assertAlmostEqual(
                        measured.get(cell, 0.0), truth.get(cell, 0.0), delta=DELTA,
                        msg=(
                            f"{fixture.label} / {cell}: collapsed engine "
                            f"{measured.get(cell, 0.0):.6f}% vs enumerated truth "
                            f"{truth.get(cell, 0.0):.6f}%. This is a mass-conserving "
                            "partition error: no conservation check and no sweep can "
                            "see it."
                        ),
                    )

    def test_the_pinned_oracle_agrees_with_an_independent_python_reconstruction(self) -> None:
        """Pins the pin. An unpinned oracle silently blesses a wrong recipe."""

        pinned = self.oracle["outcome_masses"]
        for fixture in FIXTURES:
            with self.subTest(case=fixture.label):
                truth = pinned[fixture.label]
                rebuilt = reconstruct_outcome_masses(fixture)
                for cell in sorted(set(truth) | set(rebuilt)):
                    self.assertAlmostEqual(
                        rebuilt.get(cell, 0.0), truth.get(cell, 0.0), delta=DELTA,
                        msg=(
                            f"{fixture.label} / {cell}: the committed enumerated "
                            f"artifact says {truth.get(cell, 0.0):.6f}% and an "
                            f"independent enumeration says {rebuilt.get(cell, 0.0):.6f}%. "
                            "Regenerating the artifact is NOT the fix; one of the two "
                            "enumerations is wrong."
                        ),
                    )

    def test_the_pinned_artifact_describes_these_fixtures(self) -> None:
        """A fixture edited without regenerating the artifact would compare a new
        state against an old truth and could read PASS by coincidence."""

        pinned = self.oracle["fixtures"]
        self.assertEqual(sorted(pinned), sorted(f.label for f in FIXTURES))
        for fixture in FIXTURES:
            with self.subTest(case=fixture.label):
                self.assertEqual(pinned[fixture.label], dataclasses.asdict(fixture))

    def test_masses_sum_to_one(self) -> None:
        """Free, weaker, and the only thing here that catches an in-place
        ``update_percentage`` leak rather than a misplaced roll."""

        for fixture in FIXTURES:
            with self.subTest(case=fixture.label):
                total = sum(
                    b.percentage
                    for b in pe.generate_instructions(
                        build_state(fixture), fixture.move, "splash"
                    )
                )
                self.assertAlmostEqual(total, 100.0, delta=DELTA, msg=fixture.label)

    def test_the_fixture_matrix_is_the_expected_size(self) -> None:
        """CI counts test METHODS, not fixtures: every assertion above runs under
        ``subTest`` inside one method, so six of the seven fixtures could be deleted
        with everything green. This is the guard one level down."""

        self.assertEqual(len(FIXTURES), 7)
        self.assertEqual(
            sorted(f.exercises for f in FIXTURES),
            [
                "crit-fan-residual-unbounded-ceiling",
                "crit-straddle-residual",
                "disjoint-bands-unbounded-ceiling",
                "ko-threshold-is-the-band-ceiling",
                "negative-control",
                "status-aware-threshold",
                "union-not-minimum",
            ],
        )

    def test_every_partition_site_is_covered(self) -> None:
        """The gap this closes, and why it is asserted rather than commented.

        The first version of this file had five fixtures reaching only TWO of the
        four sites the branch edits. Nothing said so: `case-a` and
        `case-b-crit-nokill` were simply absent, and the size pin froze the
        absence in place. Review found it by MUTATION -- an off-by-one in the arm
        price at each of those two sites left every fixture GREEN.

        So coverage is now a machine-checked property of the matrix. If a fifth
        partition site is ever added, this fails until a fixture reaches it.
        """

        covered = {fixture.site for fixture in FIXTURES if fixture.site in PARTITION_SITES}
        missing = sorted(set(PARTITION_SITES) - covered)
        self.assertEqual(
            missing, [],
            f"no fixture reaches {missing}; an off-by-one there would leave this "
            f"whole file GREEN. Sites reached: {sorted(covered)}",
        )
        self.assertEqual(
            sorted({f.site for f in FIXTURES}), sorted(PARTITION_SITES),
            "a fixture names a site that is not one of the four partition sites",
        )

    def test_the_matrix_is_not_vacuous(self) -> None:
        """Each fixture must actually reach the site it is named for.

        A fixture that straddles nothing asserts nothing and still reads PASS; six
        such shipped in the previous era. These checks are structural — computed
        from the fan and from the residual phase's own verdict — so they stay true
        if the partition code changes and false if the fixture drifts.
        """

        shapes = {}
        for fixture in FIXTURES:
            state = build_state(fixture)
            max_regular = pe.calculate_damage(state, fixture.move, "splash", False)[0][0]
            max_crit = pe.calculate_damage(state, fixture.move, "splash", True)[0][1]
            regular_fan, crit_fan = _fan(max_regular), _fan(max_crit)
            shapes[fixture.label] = {
                "regular_fan": (regular_fan[0], regular_fan[-1]),
                "crit_fan": (crit_fan[0], crit_fan[-1]),
                "hp": fixture.hp,
                "pre_move_threshold": _threshold(fixture, fixture.status),
                "burn_threshold": _threshold(fixture, "burn"),
                "toxic_threshold": _threshold(fixture, "toxic"),
            }

        crit = shapes["crit-straddle-sand"]
        self.assertLess(crit["regular_fan"][1], crit["hp"], f"not Case B: {crit}")
        self.assertLess(crit["crit_fan"][0], crit["hp"], f"crit fan does not straddle: {crit}")
        self.assertGreaterEqual(crit["crit_fan"][1], crit["hp"], f"crit fan cannot KO: {crit}")
        self.assertIsNotNone(crit["pre_move_threshold"])
        self.assertTrue(
            crit["crit_fan"][0] < crit["pre_move_threshold"] < crit["hp"],
            f"the residual threshold is not inside the surviving crit sub-fan: {crit}",
        )
        self.assertGreater(
            crit["pre_move_threshold"], crit["regular_fan"][1],
            f"the NON-crit fan reaches the threshold too, so this does not isolate "
            f"the crit site: {crit}",
        )

        a8 = shapes["a8-burn-secondary"]
        self.assertIsNone(
            a8["pre_move_threshold"],
            f"A8 requires the PRE-MOVE mirror to decline; it did not: {a8}",
        )
        self.assertIsNotNone(a8["burn_threshold"])
        self.assertTrue(
            a8["regular_fan"][0] < a8["burn_threshold"] <= a8["regular_fan"][1],
            f"the burn threshold is not inside the non-crit fan: {a8}",
        )

        nested = shapes["nested-thresholds"]
        self.assertIsNotNone(nested["pre_move_threshold"])
        self.assertIsNotNone(nested["burn_threshold"])
        self.assertLess(
            nested["burn_threshold"], nested["pre_move_threshold"],
            f"the two thresholds coincide, so no band structure exists: {nested}",
        )
        for name in ("burn_threshold", "pre_move_threshold"):
            self.assertTrue(
                nested["regular_fan"][0] < nested[name] < nested["regular_fan"][-1],
                f"{name} is not strictly inside the fan (the top rung would measure "
                f"the f32 comparator instead): {nested}",
            )

        union = shapes["min-would-destroy-an-arm"]
        self.assertTrue(
            union["regular_fan"][0] < union["pre_move_threshold"] <= union["regular_fan"][1],
            f"the pre-move arm this control protects does not exist: {union}",
        )
        self.assertLessEqual(
            union["burn_threshold"], union["regular_fan"][0],
            f"the status threshold must sit BELOW the fan floor -- that is what makes "
            f"min() destroy the pre-move arm while the union keeps it: {union}",
        )

        control = shapes["collapsed-fan-control"]
        self.assertIsNone(
            control["pre_move_threshold"],
            f"the negative control has a pending residual, so it is not a control: {control}",
        )
        self.assertLess(control["regular_fan"][1], control["hp"], f"{control}")

        # The two fixtures added after review found the coverage gap. Each is the
        # ONLY one reaching its site, so if either drifts off that site the site
        # goes unguarded again while every test here still reads PASS.
        case_a = shapes["case-a-nested-ko-ceiling"]
        self.assertLessEqual(
            case_a["hp"], case_a["regular_fan"][-1],
            f"the NON-crit fan must reach the defender's HP or this is not case-a "
            f"at all -- it would fall through to Case B: {case_a}",
        )
        self.assertLess(
            case_a["regular_fan"][0], case_a["hp"],
            f"the fan must STRADDLE hp, not clear it entirely: {case_a}",
        )
        self.assertIsNotNone(case_a["pre_move_threshold"])
        self.assertIsNotNone(case_a["toxic_threshold"])
        self.assertLess(
            case_a["toxic_threshold"], case_a["pre_move_threshold"],
            f"the two thresholds coincide, so no band structure exists under the "
            f"KO ceiling: {case_a}",
        )
        self.assertTrue(
            case_a["regular_fan"][0]
            < case_a["toxic_threshold"]
            < case_a["pre_move_threshold"]
            < case_a["hp"],
            f"the three-level nest is not intact -- both thresholds must sit "
            f"strictly between the fan floor and the KO threshold, which is what "
            f"makes hp the band CEILING here: {case_a}",
        )

        crit_nokill = shapes["crit-fan-cannot-kill-sand"]
        self.assertLess(
            crit_nokill["crit_fan"][-1], crit_nokill["hp"],
            f"the crit fan can kill on the hit, so this reaches the crit-STRADDLE "
            f"site instead and the fourth site stays unguarded: {crit_nokill}",
        )
        self.assertIsNotNone(crit_nokill["pre_move_threshold"])
        self.assertTrue(
            crit_nokill["crit_fan"][0]
            < crit_nokill["pre_move_threshold"]
            <= crit_nokill["crit_fan"][-1],
            f"the residual threshold is not inside the crit fan: {crit_nokill}",
        )
        self.assertGreater(
            crit_nokill["pre_move_threshold"], crit_nokill["regular_fan"][-1],
            f"the NON-crit fan reaches the threshold too, so this does not isolate "
            f"the crit site: {crit_nokill}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
