"""Synthetic native seam tests, not champion qualification or strength evidence.

Terminal-policy callbacks are opt-in and private-safe for both seats. A cap
refuses; a deadline removes unfinished reservations instead of backing up HP,
checkpoint values, or a draw. Random fixture weights cannot admit Phase A.
"""

import json
import time
import unittest

from test_model_priors_search import _EncodedSearchFixture, _crate_ready, pokezero_search


@unittest.skipUnless(_crate_ready, "requires freshly built native model feature")
class RawPolicyLeafNativeTest(_EncodedSearchFixture, unittest.TestCase):
    def run_search(self, callbacks=None, **options):
        position = self.position
        ctx = json.loads(position["ctx"])
        orders = []
        for index, slot in enumerate(("p1", "p2")):
            order = list(ctx[slot])
            active = position["actives"][index]
            order[0], order[active] = order[active], order[0]
            orders.append(order)
        if callbacks is None:
            callbacks = [self.certain, self.certain]
        kwargs = dict(max_depth=2, seed=5, model_priors=True, use_opponent_priors=False,
            arm_priors=True, rollout_leaf_mode="raw_policy_terminal", rollout_policy="raw_argmax",
            rollouts=1, rollout_threads=1, rollout_max_plies=1, rollout_seed=101,
            raw_policy_callbacks=callbacks, raw_policy_request_orders=orders, record_joint_actions=True)
        kwargs.update(options)
        return json.loads(self.native.search_batched_multi_encoded(position["state_str"], 8, 4,
            self.tables_json, position["row_inputs"], position["ctx"],
            pokezero_search.FoldState.from_payload(position["fold_state"]), **kwargs))

    @staticmethod
    def certain(raw):
        indices = json.loads(raw)["native_request_bundle"]["native_action_indices"]
        return [float(i == 0) for i in range(len(indices))]

    def test_cap_refuses_after_both_private_safe_providers_not_hp_fallback(self):
        payloads = []
        def callback(raw):
            payloads.append(json.loads(raw))
            return self.certain(raw)
        with self.assertRaisesRegex(ValueError, "nonterminal ply cap; no fallback value") as caught:
            self.run_search([callback, callback])
        diagnostic = json.loads(caught.exception.raw_policy_leaf_diagnostic)
        self.assertEqual(diagnostic["schema"], "raw-policy-terminal-refusal-v1")
        self.assertTrue(diagnostic["unresolved_no_fallback"])
        self.assertEqual(diagnostic["cap_refusals"], 1)
        self.assertEqual(diagnostic["terminal_leaves"], 0)
        self.assertGreater(diagnostic["unbacked_round_traversals"], 0)
        self.assertGreater(diagnostic["provider_calls"], 0)
        self.assertEqual([p["opponent_slot"] for p in payloads[:2]], ["p1", "p2"])
        for payload in payloads:
            self.assertEqual(set(payload), {"native_request_bundle", "public_branch_lines", "opponent_slot"})
            own = payload["native_request_bundle"]["request"]["side"]
            self.assertEqual(own["id"], payload["opponent_slot"])
            self.assertTrue(payload["public_branch_lines"])
            self.assertNotIn("state", payload)
            for line in payload["public_branch_lines"]:
                fields = line.split("|")
                if fields[1] in {"switch", "drag", "replace", "-damage", "-heal"}:
                    hp = fields[4 if fields[1] in {"switch", "drag", "replace"} else 3]
                    self.assertTrue(hp == "0 fnt" or hp.split()[0].endswith("/100"), line)

    def test_stochastic_or_failed_provider_never_becomes_raw_argmax(self):
        def distribution(raw):
            return [1.] * len(json.loads(raw)["native_request_bundle"]["native_action_indices"])
        with self.assertRaisesRegex(ValueError, "one-hot"):
            self.run_search([distribution, distribution])
        def broken(raw):
            raise ValueError("synthetic raw policy failure")
        with self.assertRaisesRegex(ValueError, "synthetic raw policy failure"):
            self.run_search([broken, self.certain])

    def test_deadline_cancels_entire_round_without_ghost_visits_or_joint_backups(self):
        calls = []
        def slow(raw):
            calls.append(json.loads(raw))
            time.sleep(.3)  # One deliberately slow synthetic inference.
            return self.certain(raw)
        report = self.run_search([slow, slow], time_budget_ms=200, rollout_max_plies=250)
        self.assertEqual(len(calls), 1, "fixture did not enter and cancel a policy continuation")
        self.assertTrue(report["time_budget_exhausted"])
        self.assertEqual(report["iterations"], 0)
        self.assertEqual(report["rounds"], 0)
        self.assertEqual(report["decision_nodes"], 1)
        self.assertEqual(report["chance_nodes"], 0)
        self.assertGreater(report["model_evals"], 1, "original model forwards must be retained")
        self.assertGreater(report["raw_leaf_cancelled_traversals"], 0)
        self.assertGreater(report["raw_leaf_cancelled_rows"], 0)
        self.assertEqual(report["raw_leaf_terminal"], 0)
        self.assertEqual(report["raw_leaf_provider_calls"], 1)
        self.assertEqual(report["raw_leaf_cap_fallbacks"], 0)
        self.assertEqual(report["model_leaf_override"], "raw_policy_terminal")
        self.assertEqual(report["raw_leaf_value_frame"], "side_one_absolute")
        self.assertEqual(report["raw_leaf_policy"], "both_seats_own_raw_masked_argmax")
        self.assertTrue(report["raw_leaf_model_forwards_retained"])
        from pokezero.engine_search import validate_native_raw_leaf_witness
        validate_native_raw_leaf_witness(report, cap=250, seed=101, branch_on_damage=False)
        self.assertNotIn("rollouts_run", report)
        for seat in ("side_one", "side_two"):
            self.assertEqual(sum(r["visits"] for r in report[seat]), 0)
        from pokezero.engine_search import validate_native_joint_action_witness
        witness = validate_native_joint_action_witness(report)
        self.assertEqual(witness["root_completed_traversals"], 0)
        self.assertEqual(witness["tree_completed_selections"], 0)

    def test_mode_requires_complete_explicit_opt_in_and_refuses_other_settings(self):
        cases = (
            dict(raw_policy_callbacks=None), dict(raw_policy_request_orders=None),
            dict(raw_policy_callbacks=[self.certain]), dict(raw_policy_callbacks=[0, 0]),
            dict(raw_policy_request_orders=[[], []]), dict(rollout_leaf_mode=None),
            dict(raw_policy_request_orders=[["unknown"], ["unknown"]]),
            dict(rollout_leaf_mode="hp_fraction"), dict(rollout_policy="uniform"),
            dict(rollouts=2), dict(rollout_threads=2), dict(rollout_max_plies=0),
            dict(model_priors=False), dict(use_opponent_priors=True),
        )
        for options in cases:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "raw policy terminal"):
                self.run_search(**options)


if __name__ == "__main__":
    unittest.main()
