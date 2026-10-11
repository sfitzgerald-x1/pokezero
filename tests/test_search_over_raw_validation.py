"""Synthetic protocol tests; no model, search, game or historical outcome calls."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

from pokezero.mcts_eval.search_over_raw import (
    SearchConfiguration, phase_a_contract, panel_summary, freeze_selection, validation_gate,
    _fixed_panel_betting_lcb,
)
from pokezero.mcts_eval.search_over_raw_ledger import PhaseALedger


class SealedValidationTest(unittest.TestCase):
    def setUp(self):
        configs = [SearchConfiguration("raw"), SearchConfiguration("incumbent"),
                   SearchConfiguration("reference", workers=20)]
        self.configs = {c.arm: c.identity for c in configs[1:]}
        self.contract = phase_a_contract("a267ac75-c45d-476a-8071-d29648191825",
            excluded_seeds=[], configurations=configs)
        self.contract.update(execution_ready=True, remaining_gates=[])

    def intervals(self, panel, value):
        return {r["root_id"]: (value, value) for r in self.contract["panels"][panel]["root_slots"]}

    def summaries(self, panel, value):
        return {arm: panel_summary(self.contract, panel=panel, configuration=config,
            root_intervals=self.intervals(panel, value), bootstrap_reps=1000)
            for arm, config in self.configs.items()}

    def selection(self):
        return freeze_selection(self.contract, self.configs,
            exploration_summaries=self.summaries("exploration", .125))

    def test_finite_sample_range_two_and_family_alpha(self):
        result = validation_gate(self.contract, self.selection(), self.summaries("validation", .75))
        factor = math.exp(math.log(40) / 32) - 1
        expected = (.99 * .75 - factor) / (.99 + factor)
        for arm in self.configs:
            self.assertAlmostEqual(result["arms"][arm]["lower_confidence_bound"], expected)
            self.assertEqual(result["arms"][arm]["alpha"], .025)
            self.assertAlmostEqual(result["arms"][arm]["descriptive_hoeffding_lower"],
                                   .75 - math.sqrt(2 * math.log(40) / 32))
        self.assertTrue(result["both_arms_admitted"])
        self.assertFalse(result["phase_b_authorized"])

    def test_positive_bootstrap_alone_does_not_admit(self):
        summaries = self.summaries("validation", .05)
        self.assertGreater(summaries["reference"]["bootstrap_interval"][0], 0)
        result = validation_gate(self.contract, self.selection(), summaries)
        self.assertFalse(result["both_arms_admitted"])
        self.assertLess(result["arms"]["reference"]["lower_confidence_bound"], 0)

    def test_one_positive_arm_cannot_admit_both(self):
        summaries = self.summaries("validation", .75)
        summaries["incumbent"] = self.summaries("validation", 0)["incumbent"]
        result = validation_gate(self.contract, self.selection(), summaries)
        self.assertEqual(result["arms"]["reference"]["status"], "PHASE_A_GAIN_VALIDATED")
        self.assertFalse(result["both_arms_admitted"])

    def test_missing_root_is_pointwise_conservative_not_complete_case(self):
        complete = self.summaries("validation", .75)
        root_intervals = self.intervals("validation", .75)
        root_intervals.pop(next(iter(root_intervals)))
        missing = {arm: panel_summary(self.contract, panel="validation", configuration=config,
            root_intervals=root_intervals, bootstrap_reps=1000) for arm, config in self.configs.items()}
        a = validation_gate(self.contract, self.selection(), complete)
        b = validation_gate(self.contract, self.selection(), missing)
        self.assertIsNone(missing["reference"]["bootstrap_interval"])
        self.assertTrue(b["both_arms_admitted"])
        self.assertLess(b["arms"]["reference"]["lower_confidence_bound"],
                        a["arms"]["reference"]["lower_confidence_bound"])

    def test_incomplete_exploration_can_be_frozen_without_selective_denominator(self):
        summaries = {arm: panel_summary(self.contract, panel="exploration", configuration=config,
            root_intervals={}, bootstrap_reps=1000) for arm, config in self.configs.items()}
        selection = freeze_selection(self.contract, self.configs, exploration_summaries=summaries)
        self.assertEqual(selection["configurations"], self.configs)
        self.assertFalse(selection["phase_b_authorized"])

    def test_single_arm_and_arm_swap_selection_rejected(self):
        for configs in ({"reference": self.configs["reference"]},
                       {"incumbent": self.configs["reference"], "reference": self.configs["incumbent"]}):
            with self.assertRaises(ValueError):
                freeze_selection(self.contract, configs, exploration_summaries=self.summaries("exploration", .5))

    def test_old_contract_cannot_be_readmitted(self):
        old = copy.deepcopy(self.contract)
        old["schema"] = "pokezero.search-over-raw.phase-a.v1"
        with self.assertRaisesRegex(ValueError, "old contracts"):
            freeze_selection(old, self.configs, exploration_summaries=self.summaries("exploration", .5))

    def test_alpha_or_look_changes_fail_closed(self):
        for key, value in (("family_alpha", .1), ("alpha_per_arm", .05), ("stake", .5), ("looks", "interim")):
            changed = copy.deepcopy(self.contract)
            changed["validation_inference"][key] = value
            with self.assertRaisesRegex(ValueError, "design"):
                freeze_selection(changed, self.configs, exploration_summaries=self.summaries("exploration", .5))

    def test_missing_arm_summary_rejected(self):
        summaries = self.summaries("validation", .75)
        summaries.pop("incumbent")
        with self.assertRaisesRegex(ValueError, "both"):
            validation_gate(self.contract, self.selection(), summaries)

    def test_seed_roster_and_aggregate_forgery_rejected(self):
        for change in ("order", "bounds", "aggregate", "count"):
            summaries = self.summaries("validation", .75)
            s = summaries["reference"]
            if change == "order":
                s["seed_intervals"].reverse()
            elif change == "bounds":
                s["seed_intervals"][0]["lower"] = float("nan")
            elif change == "aggregate":
                s["identification_interval"] = [1., 1.]
            else:
                s["source_seeds"] = 200
            with self.assertRaises(ValueError):
                validation_gate(self.contract, self.selection(), summaries)

    def test_equal_seed_weight_not_equal_root_weight(self):
        rows = self.intervals("validation", 0)
        seed = self.contract["panels"]["validation"]["seeds"][0]
        count = 0
        for slot in self.contract["panels"]["validation"]["root_slots"]:
            if slot["source_seed"] == seed:
                rows[slot["root_id"]] = (1., 1.)
                count += 1
        summary = panel_summary(self.contract, panel="validation", configuration=self.configs["reference"],
            root_intervals=rows, bootstrap_reps=1000)
        self.assertEqual(summary["identification_interval"], [1/32, 1/32])
        self.assertNotEqual(1/32, count/200)

    def test_all_missing_keeps_full_roster_and_cannot_validate(self):
        summaries = {arm: panel_summary(self.contract, panel="validation", configuration=config,
            root_intervals={}, bootstrap_reps=1000) for arm, config in self.configs.items()}
        result = validation_gate(self.contract, self.selection(), summaries)
        self.assertFalse(result["both_arms_admitted"])
        for arm, summary in summaries.items():
            self.assertEqual(summary["uncertain_roots"], 200)
            self.assertEqual(summary["identification_interval"], [-1., 1.])
            self.assertEqual(result["arms"][arm]["lower_confidence_bound"], -1.)

    def ledger(self, path):
        ledger = PhaseALedger.create(path, self.contract)
        for config in self.configs.values():
            ledger.record_exploration(config, self.intervals("exploration", .125))
        ledger.freeze(self.configs)
        return ledger

    def test_joint_callback_receives_detached_both_selection_and_one_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = self.ledger(Path(tmp) / "ledger")
            def collect(contract, configurations):
                self.assertEqual(configurations, self.configs)
                self.assertTrue((ledger.directory / "validation-claim.json").exists())
                configurations.clear()
                contract.clear()
                return {arm: self.intervals("validation", .75) for arm in self.configs}
            receipt = ledger.validation_once(collect)
            self.assertTrue(receipt["gate"]["both_arms_admitted"])
            with self.assertRaises(FileExistsError):
                ledger.validation_once(lambda *_: self.fail("no holdout reuse"))

    def test_missing_arm_callback_fails_once_without_partial_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = self.ledger(Path(tmp) / "ledger")
            with self.assertRaisesRegex(ValueError, "both"):
                ledger.validation_once(lambda *_: {"reference": self.intervals("validation", .75)})
            self.assertTrue((ledger.directory / "validation-failure.json").exists())
            self.assertFalse((ledger.directory / "validation-result.json").exists())
            with self.assertRaises(FileExistsError):
                ledger.validation_once(lambda *_: self.fail("no retry"))

    def test_in_memory_readiness_input_hash_and_roster_drift_never_admit(self):
        changes = (
            lambda c: c.update(execution_ready=True, remaining_gates=[]),
            lambda c: c.update(input_hashes={}),
            lambda c: c["panels"]["validation"]["seeds"].reverse(),
        )
        for index, change in enumerate(changes):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as tmp:
                # All disk contracts are unready. For the input-hash case,
                # even a bound nonexistent file must not be erased in memory.
                contract = copy.deepcopy(self.contract)
                contract.update(execution_ready=False, remaining_gates=["runtime qualification"],
                    input_hashes={str(Path(tmp) / "missing"): "0" * 64})
                ledger = PhaseALedger.create(Path(tmp) / "ledger", contract)
                change(ledger.contract)
                with self.assertRaisesRegex(ValueError, "in-memory"):
                    ledger.record_exploration(self.configs["incumbent"], {})
                with self.assertRaisesRegex(ValueError, "in-memory"):
                    ledger.freeze(self.configs)
                with self.assertRaisesRegex(ValueError, "in-memory"):
                    ledger.validation_once(lambda *_: self.fail("no callback"))
                self.assertFalse((ledger.directory / "validation-claim.json").exists())

    def test_in_memory_drift_during_callback_preserves_failed_one_shot_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = self.ledger(Path(tmp) / "ledger")
            def collect(*_):
                ledger.contract["candidate_seat"] = "p2"
                return {arm: self.intervals("validation", .75) for arm in self.configs}
            with self.assertRaisesRegex(ValueError, "in-memory"):
                ledger.validation_once(collect)
            self.assertTrue((ledger.directory / "validation-failure.json").exists())
            self.assertFalse((ledger.directory / "validation-result.json").exists())
            ledger.contract = copy.deepcopy(self.contract)
            with self.assertRaises(FileExistsError):
                ledger.validation_once(lambda *_: self.fail("failed attempt cannot be reopened"))

    def test_frozen_panel_and_configuration_forgery_is_rejected_before_create(self):
        mutations = (
            lambda c: c["panels"].update(validation=copy.deepcopy(c["panels"]["exploration"])),
            lambda c: c["panels"]["validation"]["seeds"].__setitem__(0, True),
            lambda c: c["panels"]["validation"]["seeds"].__setitem__(0, float(c["panels"]["validation"]["seeds"][0])),
            lambda c: c["panels"]["validation"]["root_slots"].pop(),
            lambda c: c["panels"]["validation"]["root_slots"][0].update(root_id="invented"),
            lambda c: c["panels"]["validation"]["root_slots"][0].update(source_seed=7),
            lambda c: c["panels"]["validation"]["root_slots"][0].update(root_slot=False),
            lambda c: c["configurations"][1].update(identity="forged"),
            lambda c: c["configurations"][1].update(deployable=False),
            lambda c: c.update(estimand="equal roots"),
            lambda c: c.update(excluded_seed_inventory_sha256="forged"),
            lambda c: c.update(historical_outcomes_pooled=0),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as tmp:
                changed = copy.deepcopy(self.contract)
                mutate(changed)
                path = Path(tmp) / "ledger"
                with self.assertRaises(ValueError):
                    PhaseALedger.create(path, changed)
                self.assertFalse(path.exists())

    def test_callback_subclass_accessors_cannot_diverge_analysis_and_evidence(self):
        class MisleadingIntervals(dict):
            def get(self, key, default=None):
                return [.75, .75]
        with tempfile.TemporaryDirectory() as tmp:
            ledger = self.ledger(Path(tmp) / "ledger")
            evidence = {arm: MisleadingIntervals(self.intervals("validation", -.75)) for arm in self.configs}
            receipt = ledger.validation_once(lambda *_: evidence)
            self.assertFalse(receipt["gate"]["both_arms_admitted"])
            for arm in self.configs:
                self.assertEqual(receipt["summaries"][arm]["identification_interval"], [-.75, -.75])
                self.assertEqual(receipt["root_intervals"][arm],
                                 {root: list(value) for root, value in self.intervals("validation", -.75).items()})
                evidence[arm].clear()
            self.assertTrue(receipt["root_intervals"]["reference"])

    def test_exploration_is_snapshotted_and_matches_its_persisted_summary(self):
        class MisleadingIntervals(dict):
            def get(self, key, default=None):
                return [.75, .75]
        with tempfile.TemporaryDirectory() as tmp:
            ledger = PhaseALedger.create(Path(tmp) / "ledger", self.contract)
            for config in self.configs.values():
                intervals = MisleadingIntervals(self.intervals("exploration", -.75))
                summary = ledger.record_exploration(config, intervals)
                self.assertEqual(summary["identification_interval"], [-.75, -.75])
                intervals.clear()
            ledger.freeze(self.configs)
            receipt = ledger.validation_once(lambda *_: {arm: self.intervals("validation", .75)
                                                         for arm in self.configs})
            self.assertTrue(receipt["gate"]["both_arms_admitted"])

    def test_terminal_receipt_prevents_reopening_even_if_claim_is_missing(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as tmp:
                ledger = self.ledger(Path(tmp) / "ledger")
                if failed:
                    with self.assertRaisesRegex(ValueError, "both"):
                        ledger.validation_once(lambda *_: {})
                else:
                    ledger.validation_once(lambda *_: {arm: self.intervals("validation", .75)
                                                       for arm in self.configs})
                (ledger.directory / "validation-claim.json").unlink()
                recovered = PhaseALedger(ledger.directory)
                with self.assertRaises(FileExistsError):
                    recovered.validation_once(lambda *_: self.fail("terminal evidence forbids a new callback"))
                self.assertFalse((ledger.directory / "validation-claim.json").exists())

    def test_small_noiseless_gain_can_pass_within_supported_panel_sizes(self):
        self.contract = phase_a_contract(self.contract["namespace"], excluded_seeds=[],
            configurations=[SearchConfiguration("raw"), SearchConfiguration("incumbent"),
                            SearchConfiguration("reference", workers=20)], seeds_per_panel=64)
        result = validation_gate(self.contract, self.selection(), self.summaries("validation", .1))
        self.assertTrue(result["both_arms_admitted"])
        self.assertLess(result["arms"]["reference"]["descriptive_hoeffding_lower"], 0)
        self.assertGreater(result["arms"]["reference"]["lower_confidence_bound"], 0)
        # Demonstrates mathematical feasibility, NOT noisy-real-world power.

    def test_exact_skewed_null_false_positive_probability_is_bounded(self):
        for n in (32, 64, 128, 200):
            probability = 0.
            for negatives in range(n + 1):
                values = [-1.] * negatives + [.25] * (n - negatives)
                if _fixed_panel_betting_lcb(values, alpha=.025, stake=.99) > 0:
                    probability += math.comb(n, negatives) * .2**negatives * .8**(n - negatives)
            self.assertLessEqual(probability, .025)

    def test_betting_boundary_and_pointwise_missing_monotonicity(self):
        lower = _fixed_panel_betting_lcb([.125] * 32, alpha=.025, stake=.99)
        factor = math.exp(math.log(40) / 32) - 1
        self.assertAlmostEqual(lower, (.99 * .125 - factor) / (.99 + factor))
        self.assertGreater(lower, 0)
        erased = _fixed_panel_betting_lcb([-1.] + [.125] * 31, alpha=.025, stake=.99)
        self.assertLess(erased, lower)
        self.assertEqual(_fixed_panel_betting_lcb([-1.] * 32, alpha=.025, stake=.99), -1.)
        for invalid in ([float("nan")], [True], [1.1], []):
            with self.assertRaises(ValueError):
                _fixed_panel_betting_lcb(invalid, alpha=.025, stake=.99)


if __name__ == "__main__":
    unittest.main()
