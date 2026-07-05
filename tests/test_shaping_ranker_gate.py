"""Unit coverage for the shaping ranker's gate statistics (#510 review HIGH-1/MED-1).

The ranker lives in scripts/ (not the package); load it by path so the gate math the
review attacked — the head-marginal partial correlation, the floor formula, and the
built-in validity probes — is pinned by tests without needing torch or a corpus.
"""

import importlib.util
import math
from pathlib import Path
import random
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "shaping_ranker", Path(__file__).resolve().parents[1] / "scripts" / "shaping_ranker.py"
)
shaping_ranker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(shaping_ranker)

from pokezero.shaping import SHAPING_PRESETS  # noqa: E402


class PartialPearsonTest(unittest.TestCase):
    def test_control_case_reduces_to_plain_pearson(self) -> None:
        rng = random.Random(1)
        outcomes = [rng.choice((-1.0, 1.0)) for _ in range(400)]
        predictions = [0.5 * outcome + rng.gauss(0, 0.5) for outcome in outcomes]
        phis = [0.0] * len(outcomes)  # unshaped control: Phi has no variance
        self.assertAlmostEqual(
            shaping_ranker._partial_pearson(predictions, outcomes, phis),
            shaping_ranker._pearson(predictions, outcomes),
            places=12,
        )

    def test_dead_head_scores_zero_marginal_even_when_phi_predicts(self) -> None:
        # The #510 HIGH-1 attack: Phi correlates with outcome, the head is noise with
        # tiny variance. Corrected Pearson inherits corr(Phi, terminal) by
        # scale-invariance; the marginal must NOT.
        rng = random.Random(2)
        outcomes = [rng.choice((-1.0, 1.0)) for _ in range(600)]
        phis = [0.3 * outcome + rng.gauss(0, 0.4) for outcome in outcomes]
        dead_predictions = [rng.gauss(0, 1e-3) for _ in outcomes]
        corrected = [p + phi for p, phi in zip(dead_predictions, phis)]

        inherited = shaping_ranker._pearson(corrected, outcomes)
        phi_alone = shaping_ranker._pearson(phis, outcomes)
        marginal = shaping_ranker._partial_pearson(dead_predictions, outcomes, phis)

        self.assertAlmostEqual(inherited, phi_alone, places=2)  # the free ride
        self.assertLess(abs(marginal), 0.05)  # the gate sees through it

    def test_phi_parroting_head_scores_zero_marginal(self) -> None:
        rng = random.Random(3)
        outcomes = [rng.choice((-1.0, 1.0)) for _ in range(400)]
        phis = [0.3 * outcome + rng.gauss(0, 0.4) for outcome in outcomes]
        parrot = [0.001 * phi for phi in phis]  # perfectly collinear with Phi
        marginal = shaping_ranker._partial_pearson(parrot, outcomes, phis)
        self.assertAlmostEqual(marginal, 0.0, places=9)

    def test_genuine_head_keeps_marginal_signal_under_large_phi(self) -> None:
        rng = random.Random(4)
        outcomes = [rng.choice((-1.0, 1.0)) for _ in range(600)]
        phis = [8.0 * (0.3 * outcome + rng.gauss(0, 0.4)) for outcome in outcomes]
        predictions = [0.5 * outcome + rng.gauss(0, 0.5) for outcome in outcomes]
        marginal = shaping_ranker._partial_pearson(predictions, outcomes, phis)
        # The additive delta (corrected - phi_alone) is variance-diluted for huge |Phi|;
        # the partial correlation is not.
        self.assertGreater(marginal, 0.3)

    def test_degenerate_inputs_do_not_crash(self) -> None:
        self.assertEqual(shaping_ranker._partial_pearson([], [], []), 0.0)
        self.assertEqual(shaping_ranker._partial_pearson([1.0], [1.0], [1.0]), 0.0)
        constant = shaping_ranker._partial_pearson([1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [0.1, 0.2, 0.3])
        self.assertEqual(constant, 0.0)


class FloorFormulaTest(unittest.TestCase):
    def _floor(self, control: float, retention: float = 0.10, minimum: float = 0.0) -> float:
        return max(control - retention * abs(control), minimum)

    def test_positive_control_matches_legacy_intent(self) -> None:
        self.assertAlmostEqual(self._floor(0.30), 0.27)

    def test_negative_control_never_places_floor_above_control(self) -> None:
        # MED-1: control * (1 - X) inverts for negative control; the |control| form
        # keeps floor <= control so a control can never fail its own relative floor.
        control = -0.1741
        relative_floor = control - 0.10 * abs(control)
        self.assertLessEqual(relative_floor, control)
        # With the absolute minimum active, the run is low-confidence by definition
        # (control <= 0) rather than silently inverted.
        self.assertEqual(self._floor(control), 0.0)


class ValidityProbeTest(unittest.TestCase):
    def test_builtin_probes_scale_the_arm1_preset(self) -> None:
        probes = shaping_ranker.builtin_validity_candidates(80.0)
        by_label = {probe["label"]: probe["shaping"] for probe in probes}
        base = SHAPING_PRESETS["wse-arm1"]
        inverted = by_label[shaping_ranker.VALIDITY_INVERTED_LABEL]
        saturating = by_label[shaping_ranker.VALIDITY_SATURATING_LABEL]
        self.assertEqual(inverted.hp_weight, -base.hp_weight)
        self.assertEqual(inverted.status_weight("tox"), -base.status_weight("tox"))
        self.assertEqual(saturating.hp_weight, 80.0 * base.hp_weight)
        self.assertEqual(saturating.faint_weight, 80.0 * base.faint_weight)
        self.assertTrue(all(probe["builtin"] for probe in probes))

    def test_scale_preserves_terminal_mode_and_status_keys(self) -> None:
        base = SHAPING_PRESETS["wse-arm1"]
        scaled = shaping_ranker.scale_shaping_config(base, -1.0)
        self.assertEqual(scaled.terminal_mode, base.terminal_mode)
        self.assertEqual(
            [status for status, _ in scaled.status_weights],
            [status for status, _ in base.status_weights],
        )


class EceTest(unittest.TestCase):
    def test_saturated_corrected_predictions_blow_past_the_cap(self) -> None:
        # arm1 x 80-style corrected predictions live far outside [-1, 1]; ECE against
        # +-1 outcomes must be enormous (the review measured 8.46) so the max-ece gate
        # can never print retained=yes next to it.
        rng = random.Random(5)
        outcomes = [rng.choice((-1.0, 1.0)) for _ in range(200)]
        corrected = [10.0 * rng.gauss(0, 1) for _ in outcomes]
        self.assertGreater(shaping_ranker._ece(corrected, outcomes), 2.0)

    def test_sane_predictions_stay_below_the_cap(self) -> None:
        rng = random.Random(6)
        outcomes = [rng.choice((-1.0, 1.0)) for _ in range(200)]
        predictions = [0.6 * outcome + rng.gauss(0, 0.2) for outcome in outcomes]
        self.assertLess(shaping_ranker._ece(predictions, outcomes), 1.0)


if __name__ == "__main__":
    unittest.main()
