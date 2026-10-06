from types import SimpleNamespace
import itertools
import unittest

from pokezero.mcts_eval.wider_search import ARMS, SEATS, analyze, exact_signflip_p, game_identity, play_game, study_seed
from pokezero.mcts_eval.paper_reference import Evaluation


def rows(seeds, reference=1, incumbent=-1):
    return [dict(identity=game_identity(seed, seat, arm), seed=seed, subject=seat, arm=arm,
        status="COMPLETE", signed_outcome=reference if arm == ARMS[1] else incumbent)
        for seed in seeds for arm in ARMS for seat in SEATS]


class WiderSearchTests(unittest.TestCase):
    def test_independent_roster_does_not_select_disagreements_or_overlap_qualification(self):
        confirmation = {study_seed(i) for i in range(64)}
        self.assertEqual(len(confirmation), 64)
        self.assertFalse(confirmation & {study_seed(i, qualification=True) for i in range(2)})

    def test_exact_cluster_p_matches_brute_force(self):
        values = [1., .5, -.25, 0., -1.]
        observed = abs(sum(values))
        brute = sum(abs(sum(a*b for a, b in zip(signs, values))) >= observed
            for signs in itertools.product((-1, 1), repeat=len(values))) / 2**len(values)
        self.assertEqual(exact_signflip_p(values), brute)

    def test_seats_not_counted_as_independent_clusters(self):
        result = analyze(list(range(6)), rows(list(range(6))))
        self.assertEqual(result["complete_seed_clusters"], 6)
        self.assertEqual(result["registered_games"], 24)
        self.assertEqual(result["exact_seed_cluster_signflip_p"], 2 / 2**6)
        # Six favorable clusters still have too-wide a distribution-free mean interval.
        self.assertFalse(result["statistically_supported_advantage"])

    def test_null_and_strong_alternative(self):
        seeds = list(range(64))
        self.assertFalse(analyze(seeds, rows(seeds, 1, 1))["statistically_supported_advantage"])
        self.assertTrue(analyze(seeds, rows(seeds))["statistically_supported_advantage"])

    def test_missing_and_refused_never_scored_or_complete_case_tested(self):
        seeds = [1, 2]
        data = rows(seeds)
        data[0].update(status="REFUSED", signed_outcome=None)
        result = analyze(seeds, data[:-1])
        self.assertEqual(result["complete_seed_clusters"], 0)
        self.assertEqual(result["mean_delta_worst_case_bounds"], [-1., 1.])
        self.assertFalse(result["inferential_test_allowed"])
        data[0]["signed_outcome"] = -1
        with self.assertRaisesRegex(ValueError, "never be scored"):
            analyze(seeds, data)

    def test_duplicate_and_extra_roster_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze([1], rows([1]) + rows([1])[:1])
        with self.assertRaisesRegex(ValueError, "extra"):
            analyze([1], rows([1, 2]))

    def test_searches_first_request_and_all_later_requests(self):
        class Env:
            boundary = 0
            def terminal(self):
                return None if self.boundary < 3 else SimpleNamespace(capped=False, winner="p1")
            def requested_players(self):
                return ("p1", "p2") if self.boundary != 1 else ("p2",)
            def observe(self, actor):
                return SimpleNamespace(actor=actor, legal_action_mask=(True, True))
            def reseed_simulator_rng(self, seed):
                pass
            def step(self, actions):
                self.boundary += 1
        calls = []
        def search(observation, boundary, seed):
            self.assertEqual(observation.actor, "p1")
            calls.append(boundary)
            return 1, {"selector": "search"}
        result = play_game(Env(), subject="p1", decision_id="independent", selector=search,
            opponent=lambda obs: ((0, 1), Evaluation((.5, .5), 0)), emit=lambda row: None,
            max_boundaries=10, wall_seconds=60)
        self.assertEqual(calls, [0, 2])
        self.assertEqual(result["status"], "COMPLETE")

    def test_refusal_does_not_trigger_raw_takeover(self):
        env = SimpleNamespace(terminal=lambda: None, requested_players=lambda: ("p1",),
            observe=lambda actor: SimpleNamespace(legal_action_mask=(True,)))
        def refuse(*args):
            raise RuntimeError("search refused")
        with self.assertRaisesRegex(RuntimeError, "search refused"):
            play_game(env, subject="p1", decision_id="test", selector=refuse, opponent=None,
                emit=lambda row: None, max_boundaries=10, wall_seconds=60)
