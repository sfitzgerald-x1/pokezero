import copy
from types import SimpleNamespace
import unittest

from pokezero.mcts_eval.search_over_raw import (
    SearchConfiguration, phase_a_contract, paired_continuations, root_contrast,
    panel_summary, freeze_selection, validation_gate, digest, select_source_requests,
)


NAMESPACE = "a267ac75-c45d-476a-8071-d29648191825"


class FakeEnv:
    def __init__(self, capped=False):
        self.calls = []
        self.capped = capped
        self.step_index = 0

    def restore(self, snapshot):
        self.step_index = 0

    def terminal(self):
        return None if self.step_index == 0 else SimpleNamespace(capped=self.capped, winner="p1")

    def requested_players(self):
        return ("p1", "p2")

    def observe(self, seat):
        return seat

    def reseed_simulator_rng(self, seed):
        self.seed = seed

    def step(self, choices):
        self.calls.append((dict(choices), self.seed))
        self.step_index += 1


class SearchOverRawTest(unittest.TestCase):
    def setUp(self):
        self.configs = [SearchConfiguration("raw"), SearchConfiguration("incumbent"),
            SearchConfiguration("reference", workers=20), SearchConfiguration("reference", belief="oracle", workers=20)]
        self.contract = phase_a_contract(NAMESPACE, excluded_seeds=[7, 8], configurations=self.configs)

    def test_panels_are_fresh_disjoint_and_reproducible(self):
        a, b = (self.contract["panels"][p]["seeds"] for p in ("exploration", "validation"))
        self.assertEqual(len(a), 32)
        self.assertFalse(set(a) & set(b))
        self.assertFalse((set(a) | set(b)) & {7, 8})
        self.assertEqual(self.contract, phase_a_contract(NAMESPACE, excluded_seeds=[8, 7], configurations=self.configs))
        self.assertEqual(sum(len(p["root_slots"]) for p in self.contract["panels"].values()), 400)

    def test_exposed_collision_is_skipped_before_collection(self):
        exposed = self.contract["panels"]["exploration"]["seeds"][0]
        other = phase_a_contract(NAMESPACE, excluded_seeds=[exposed], configurations=self.configs)
        self.assertNotIn(exposed, other["panels"]["exploration"]["seeds"])

    def test_root_selection_is_catalog_order_independent_and_not_early_only(self):
        selected = select_source_requests(NAMESPACE, 9, list(range(60)), 7)
        self.assertEqual(selected, select_source_requests(NAMESPACE, 9, list(reversed(range(60))), 7))
        self.assertEqual(len(selected), 7)
        self.assertTrue(any(r > 20 for r in selected))
        self.assertEqual(len(select_source_requests(NAMESPACE, 9, [0, 1], 7)), 2)

    def test_root_selection_rejects_duplicate_or_bool_requests(self):
        for requests in ([1, 1], [True]):
            with self.assertRaises(ValueError):
                select_source_requests(NAMESPACE, 9, requests, 7)

    def test_configuration_rejects_invalid_fields(self):
        for kwargs in (dict(workers=True), dict(seconds=float("nan")), dict(leaf="unknown"), dict(belief="truth")):
            with self.assertRaises(ValueError):
                SearchConfiguration("reference", **kwargs)
        with self.assertRaises(ValueError):
            SearchConfiguration("raw", belief="oracle")

    def test_contract_rejects_small_panel_and_missing_arm(self):
        with self.assertRaises(ValueError):
            phase_a_contract(NAMESPACE, excluded_seeds=[], configurations=self.configs, seeds_per_panel=31)
        with self.assertRaises(ValueError):
            phase_a_contract(NAMESPACE, excluded_seeds=[], configurations=self.configs[:2])

    def audit(self, env, actions):
        return paired_continuations(env=env, snapshot=object(), subject="p1", actions=actions,
            evaluator=lambda _: ([0, 1], [.4, .6]), namespace=NAMESPACE, root_id="root")

    def test_same_action_reuses_receipts_without_dropping_root(self):
        env = FakeEnv()
        audit = self.audit(env, dict(raw=1, search=1))
        self.assertEqual(len(env.calls), 8)
        self.assertEqual(root_contrast(audit, "search"), (0., 0.))

    def test_opponent_and_chance_randomness_are_paired(self):
        env = FakeEnv()
        audit = self.audit(env, dict(raw=0, search=1))
        self.assertEqual([c[0]["p2"] for c in env.calls[:8]], [c[0]["p2"] for c in env.calls[8:]])
        self.assertEqual([c[1] for c in env.calls[:8]], [c[1] for c in env.calls[8:]])
        self.assertEqual(root_contrast(audit, "search"), (0., 0.))

    def test_capped_outcomes_are_uncertain_not_losses(self):
        audit = self.audit(FakeEnv(capped=True), dict(raw=0, search=1))
        self.assertEqual(root_contrast(audit, "search"), (-1., 1.))
        self.assertTrue(all(r["signed_outcome"] is None for r in audit["outcomes"]))

    def test_illegal_action_and_invalid_priors_fail(self):
        with self.assertRaisesRegex(ValueError, "illegal"):
            self.audit(FakeEnv(), dict(raw=2))
        with self.assertRaisesRegex(ValueError, "distribution"):
            paired_continuations(env=FakeEnv(), snapshot=object(), subject="p1", actions=dict(raw=0),
                evaluator=lambda _: ([0, 1], [float("nan"), .6]), namespace=NAMESPACE, root_id="root")

    def test_terminal_source_cannot_be_a_decision_root(self):
        env = FakeEnv()
        env.restore = lambda _: setattr(env, "step_index", 1)
        with self.assertRaisesRegex(ValueError, "already terminal"):
            self.audit(env, dict(raw=0, search=1))

    def test_missing_or_duplicate_continuations_fail(self):
        audit = self.audit(FakeEnv(), dict(raw=0, search=1))
        audit["outcomes"].pop()
        with self.assertRaisesRegex(ValueError, "missing"):
            root_contrast(audit, "search")
        audit = self.audit(FakeEnv(), dict(raw=0, search=1))
        audit["outcomes"].append(audit["outcomes"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            root_contrast(audit, "search")

    def summary(self, panel, config=None, missing=False):
        config = config or self.configs[2].identity
        rows = {r["root_id"]: (.125, .125) for r in self.contract["panels"][panel]["root_slots"]}
        if missing:
            rows.pop(next(iter(rows)))
        return panel_summary(self.contract, panel=panel, configuration=config,
            root_intervals=rows, bootstrap_reps=1000)

    def test_missing_source_root_keeps_denominator_and_disables_gate(self):
        summary = self.summary("validation", missing=True)
        self.assertEqual(summary["source_seeds"], 32)
        self.assertEqual(summary["root_slots"], 200)
        self.assertEqual(summary["uncertain_roots"], 1)
        self.assertIsNone(summary["bootstrap_interval"])

    def test_exploration_is_required_before_selection(self):
        with self.assertRaisesRegex(ValueError, "exploration"):
            freeze_selection(self.contract, self.configs[2].identity,
                exploration_summary=self.summary("validation"))

    def test_oracle_cannot_be_selected(self):
        config = self.configs[3].identity
        with self.assertRaisesRegex(ValueError, "oracle"):
            freeze_selection(self.contract, config, exploration_summary=self.summary("exploration", config))

    def test_validation_matches_frozen_selection_not_new_configuration(self):
        config = self.configs[2].identity
        selection = freeze_selection(self.contract, config, exploration_summary=self.summary("exploration"))
        with self.assertRaisesRegex(ValueError, "matching|match"):
            validation_gate(self.contract, selection, self.summary("validation", self.configs[1].identity))
        result = validation_gate(self.contract, selection, self.summary("validation"))
        self.assertEqual(result["status"], "PHASE_A_GAIN_VALIDATED")
        self.assertFalse(result["phase_b_authorized"])

    def test_uncertain_validation_cannot_open_phase_b(self):
        config = self.configs[2].identity
        selection = freeze_selection(self.contract, config, exploration_summary=self.summary("exploration"))
        result = validation_gate(self.contract, selection, self.summary("validation", missing=True))
        self.assertEqual(result["status"], "NO_VALIDATED_GAIN")

    def test_source_drift_invalidates_frozen_selection(self):
        config = self.configs[2].identity
        selection = freeze_selection(self.contract, config, exploration_summary=self.summary("exploration"))
        drifted = copy.deepcopy(self.contract)
        drifted["candidate_seat"] = "p2"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validation_gate(drifted, selection, self.summary("validation"))

    def test_no_snapshot_or_opponent_commitment_in_audit_receipt(self):
        audit = self.audit(FakeEnv(), dict(raw=0, search=1))
        self.assertNotIn("snapshot", audit)
        self.assertNotIn("opponent_action", audit)
        self.assertEqual(len(digest(audit)), 64)


if __name__ == "__main__":
    unittest.main()
