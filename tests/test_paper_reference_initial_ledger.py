import math
import random
from types import SimpleNamespace as S
import unittest

from pokezero.mcts_eval.paper_reference import ReferenceRefusal
from pokezero.mcts_eval.paper_reference_initial_ledger import retain_initial_anchor, initial_ledger_draw_view
from pokezero.mcts_eval.paper_reference_particles import (
    HistoryParticle, WeightedAdvance, bootstrap_population)
from pokezero.mcts_eval.paper_reference_seed_bank import draw_guided_seed, matches_hints


class InitialAnchorLedgerTest(unittest.TestCase):
    def test_eliminated_initial_particle_keeps_recomputable_complete_proposal(self):
        receipt, freed = {}, []
        hints = [dict(kind="coin", ordinal=1, denominator=2, numerator=1, desired=True)]
        def initial(i):
            proposal = {}
            seed, weight = draw_guided_seed(random.Random(i), hints, lambda: None, proposal, seed_bits=32)
            anchor = dict(materialization_seed=seed, materialization_seed_proposal=proposal)
            retain_initial_anchor(receipt, ordinal=i, anchor=anchor, importance_weight=weight)
            return HistoryParticle(object(), i, anchor, (), initial_importance_weight=weight)
        def advance(p, stage):
            child = HistoryParticle(object(), p.ancestor, p.anchor_receipt, (*p.steps, stage))
            return WeightedAdvance(child, p.initial_importance_weight)
        # After stage zero only ancestor zero survives resampling multiplicity.
        rng = S(choices=lambda candidates, weights, k: [candidates[0]] * k,
                choice=lambda candidates: candidates[0])
        final = bootstrap_population(count=2, stages=2, initial=initial, advance=advance,
            release=freed.append, rng=rng, check=lambda: None, receipt=receipt)
        self.assertTrue(all(p.ancestor == 0 for p in final))
        self.assertEqual([r["initialization_ordinal"] for r in receipt["initial_anchor_proposals"]], [0, 1])
        weights = []
        for row in receipt["initial_anchor_proposals"]:
            proposal = row["materialization_seed_proposal"]
            hits = [matches_hints(seed, proposal["hints"]) for seed in proposal["original_seeds"]]
            self.assertEqual(hits, proposal["compatible"])
            length, mass = len(hits), sum(hits)
            g = proposal["guided_fraction"]
            q = [1/length if not mass else (1-g)/length + (g/mass if hit else 0) for hit in hits]
            self.assertEqual(q, proposal["proposal_probabilities"])
            index = proposal["selected_bank_index"]
            self.assertEqual(row["materialization_seed"], proposal["original_seeds"][index])
            weight = (1/length) / q[index]
            self.assertTrue(math.isclose(weight, row["initial_anchor_importance_weight"], rel_tol=1e-14))
            weights.append(weight)
        self.assertEqual(receipt["stages"][0]["survivor_importance_weights"], weights)
        self.assertEqual(receipt["stages"][1]["survivor_importance_weights"], [1., 1.])

    def test_unguided_anchor_has_unit_weight_and_explicit_null_proposal(self):
        receipt = {}
        retain_initial_anchor(receipt, ordinal=0, anchor={"materialization_seed": 42}, importance_weight=1.)
        row = receipt["initial_anchor_proposals"][0]
        self.assertEqual(row["materialization_seed"], 42)
        self.assertIsNone(row["materialization_seed_proposal"])
        self.assertEqual(row["initial_anchor_importance_weight"], 1.)

    def test_duplicate_or_skipped_initialization_ordinal_refuses(self):
        for ordinal in (-1, 1, True):
            with self.assertRaisesRegex(ReferenceRefusal, "ordinal drift"):
                retain_initial_anchor({}, ordinal=ordinal, anchor={}, importance_weight=1.)
        receipt = {}
        retain_initial_anchor(receipt, ordinal=0, anchor={}, importance_weight=1.)
        with self.assertRaisesRegex(ReferenceRefusal, "ordinal drift"):
            retain_initial_anchor(receipt, ordinal=0, anchor={}, importance_weight=1.)

    def test_forward_receipts_retain_full_initialization_ledger_once(self):
        receipt = {}
        retain_initial_anchor(receipt, ordinal=0, anchor={}, importance_weight=1.)
        first = initial_ledger_draw_view(receipt, 1)
        later = initial_ledger_draw_view(receipt, 2)
        self.assertIn("initial_anchor_proposals", first)
        self.assertNotIn("initial_anchor_proposals", later)
        self.assertEqual(later["initial_anchor_proposals_retained_at_empirical_draw"], 1)
        self.assertIn("initial_anchor_proposals", receipt)


if __name__ == "__main__":
    unittest.main()
