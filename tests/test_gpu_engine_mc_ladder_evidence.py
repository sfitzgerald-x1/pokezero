"""mc_ladder_evidence: the champion scorer measure() takes in pi_FP's place (log-softmax over the legal mask; -inf for an illegal
or missing label). Torch only (a stand-in model; no env, engine or checkpoint): run it where torch is installed, never on a laptop."""
import math
import types
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:   # the scorer needs torch; the exporter runs in a pod
    torch = None


class Observation:
    def __init__(self, legal):
        self.legal_action_mask = np.array(legal, bool)


@unittest.skipIf(torch is None, 'torch is not installed')
class ChampionScorerTests(unittest.TestCase):
    def test_log_softmax_over_the_legal_mask(self):
        from gpu_engine import mc_ladder_evidence as M
        logits = torch.tensor([[1., 2., 3.] + [0.] * 6, [0.] * 9])
        seen = {}

        def model(**inputs):
            seen.update(inputs);return types.SimpleNamespace(policy_logits=logits[:inputs['categorical_ids'].shape[0]])

        def windows(batch, window_size, device):
            self.assertEqual((window_size, device), (1, 'cpu'));self.assertTrue(all(len(w) == 1 for w in batch))
            n = len(batch);return dict(categorical_ids=torch.zeros(n, 1), numeric_features=torch.zeros(n, 1), token_type_ids=torch.zeros(n, 1),
                                       attention_mask=torch.ones(n, 1), history_mask=torch.ones(n, 1))
        champion = types.SimpleNamespace(model=model, window=1)
        observations = [Observation([True, True, False] + [False] * 6), Observation([True] * 9)]
        with mock.patch('pokezero.neural_policy.observation_windows_to_torch', windows):
            scorer = M.champion_scorer(champion, torch)
            got = scorer(observations, [1, None])
            self.assertAlmostEqual(got[0], 2. - math.log(math.e + math.e ** 2), places=6)   # action 2's logit 3 is masked out
            self.assertEqual(got[1], -math.inf)
            got = scorer(observations, [2, 4])
            self.assertEqual(got[0], -math.inf);self.assertAlmostEqual(got[1], -math.log(9.), places=6)
        self.assertEqual(sorted(seen), ['attention_mask', 'categorical_ids', 'history_mask', 'numeric_features', 'token_type_ids'])


if __name__ == '__main__':
    unittest.main()
