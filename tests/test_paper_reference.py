"""Paper algorithm contracts; these fixtures do not qualify Pokémon fidelity."""

from dataclasses import replace
import random
import unittest

from pokezero.mcts_eval.paper_reference import (
    DecisionState, Evaluation, Frame, Node, ReferenceConfig, ReferenceRefusal,
    Terminal, TrajectorySearch, signed_win_probability,
)


ROOT = DecisionState(b"root-public", ("move:a", "switch:b"), 0)


class ToyWorld:
    def __init__(self, *, root=ROOT, depth=1, value=1.0, opponent=True,
                 terminal=True, trace=None, hidden_draws=0):
        self.root, self.depth, self.value = root, depth, value
        self.opponent, self.terminal = opponent, terminal
        self.trace = trace if trace is not None else []
        self.index = 0
        self.closed = False
        self.evaluations = []
        self.hidden_draws = hidden_draws

    def frame(self):
        return self._frame(self.root)

    def _frame(self, subject):
        return Frame(subject, ("reply:a", "reply:b"), (.75, .25)) if self.opponent else Frame(subject)

    def advance(self, subject_action, opponent_action, chance_rng):
        self.index += 1
        self.trace.append((subject_action, opponent_action, chance_rng.random()))
        if self.terminal and self.index == self.depth:
            return Terminal(self.value)
        return self._frame(DecisionState(f"leaf:{self.index}".encode(), ("move:a",), 0))

    def evaluate(self, state):
        self.evaluations.append(state)
        return Evaluation(tuple(1 / len(state.actions) for _ in state.actions), self.value)

    def close(self):
        self.closed = True


def searcher(config=None):
    search = TrajectorySearch(config or ReferenceConfig(alpha=.5, beta=1))
    search.reset_battle("battle-1")
    return search


def run(search, factory, *, root=ROOT, trajectories=4, **kwargs):
    return search.search(root, battle_id="battle-1",
        evaluate_root=lambda state: Evaluation(tuple(1 / len(state.actions) for _ in state.actions), 0),
        sample_world=factory, seed=123, trajectories=trajectories, **kwargs)


class PaperTrajectoryTests(unittest.TestCase):
    def test_paper_formula_and_empty_tie_not_native_win_probability(self):
        node = Node.from_evaluation(ROOT, Evaluation((.2, .8), .5))
        self.assertEqual(node.select(ReferenceConfig(1, 1)), 0)  # sqrt(M=0)
        node.count, node.visits, node.totals = 4, [3, 1], [-1.5, .1]
        self.assertEqual(node.select(ReferenceConfig(.5, 1)), 1)
        node.count, node.visits, node.totals = 4, [2, 2], [1.0, -.8]
        self.assertEqual(node.select(ReferenceConfig(.5, 1)), 0)

    def test_signed_value_conversion_and_invalid_probability_refusal(self):
        self.assertEqual([signed_win_probability(p) for p in (0, .5, 1)], [-1, 0, 1])
        for invalid in (-.1, 1.1, True, float("nan"), float("inf")):
            with self.subTest(value=invalid), self.assertRaises(ReferenceRefusal):
                signed_win_probability(invalid)

    def test_new_world_draw_each_trajectory_and_cleanup(self):
        worlds = []
        def factory(rng):
            world = ToyWorld(value=1 if rng.random() > .5 else -1)
            worlds.append(world)
            return world
        result = run(searcher(), factory, trajectories=8)
        self.assertEqual((result.trajectories, result.world_draws, result.transitions), (8, 8, 8))
        self.assertEqual(sum(result.root_visits), 8)
        self.assertTrue(all(world.closed for world in worlds))
        self.assertEqual(len({id(world) for world in worlds}), 8)

    def test_signed_backups_stay_in_subject_perspective(self):
        search = searcher()
        run(search, lambda rng: ToyWorld(depth=3, value=-1), trajectories=4)
        self.assertGreater(len(search.nodes), 2)
        for node in search.nodes.values():
            for n, total in zip(node.visits, node.totals):
                if n:
                    self.assertEqual(total / n, -1)

    def test_first_new_leaf_stops_then_later_trajectories_deepen_past_two(self):
        worlds = []
        def factory(rng):
            world = ToyWorld(depth=4)
            worlds.append(world)
            return world
        run(searcher(), factory, trajectories=4)
        self.assertEqual([world.index for world in worlds], [1, 2, 3, 4])
        self.assertEqual([len(world.evaluations) for world in worlds], [1, 1, 1, 0])

    def test_search_persists_statistics_across_decisions(self):
        search = searcher()
        first = run(search, lambda rng: ToyWorld(), trajectories=3)
        second = run(search, lambda rng: ToyWorld(), trajectories=2)
        self.assertEqual(sum(first.root_visits), 3)
        self.assertEqual(sum(second.root_visits), 5)
        self.assertEqual(second.trajectories, 2)

    def test_final_action_is_most_visited_not_best_mean_or_prior(self):
        search = searcher(ReferenceConfig(0, 1))
        node = Node.from_evaluation(ROOT, Evaluation((.01, .99), 0))
        node.visits, node.totals, node.count = [10, 1], [0, 1], 11
        search.nodes[ROOT.key] = node
        result = run(search, lambda rng: ToyWorld(value=1), trajectories=1)
        self.assertEqual(result.action, ROOT.actions[0])
        self.assertEqual(result.root_visits, (10, 2))

    def test_opponent_is_sampled_full_policy_not_argmax_or_opponent_puct(self):
        trace = []
        run(searcher(), lambda rng: ToyWorld(trace=trace), trajectories=64)
        replies = {row[1] for row in trace}
        self.assertEqual(replies, {"reply:a", "reply:b"})

    def test_hidden_draw_consumption_does_not_change_chance_or_reply_rng(self):
        traces = [[], []]
        for consume, trace in zip((0, 100), traces):
            def factory(rng):
                for _ in range(consume):
                    rng.random()
                return ToyWorld(trace=trace)
            run(searcher(), factory, trajectories=12)
        self.assertEqual(traces[0], traces[1])

    def test_real_faint_pruning_keeps_future_nodes_not_simulated_floor(self):
        search = searcher()
        for f in (0, 1, 2):
            state = DecisionState(str(f).encode(), ("move:a",), f)
            search.nodes[state.key] = Node.from_evaluation(state, Evaluation((1.,), 0))
        root = DecisionState(b"new-real", ("move:a",), 1)
        run(search, lambda rng: ToyWorld(root=root), root=root, trajectories=1)
        self.assertNotIn(b"0", search.nodes)
        self.assertIn(b"1", search.nodes)
        self.assertIn(b"2", search.nodes)

    def test_battle_reset_erases_statistics_and_rng_ordinal(self):
        search = searcher()
        run(search, lambda rng: ToyWorld())
        search.reset_battle("battle-2")
        self.assertFalse(search.nodes)
        self.assertEqual(search._ordinal, 0)
        with self.assertRaises(ReferenceRefusal):
            run(search, lambda rng: ToyWorld())

    def test_public_key_legal_alias_refuses_and_poisoned_tree_requires_reset(self):
        search = searcher()
        run(search, lambda rng: ToyWorld())
        drift = replace(ROOT, actions=("move:other",))
        with self.assertRaisesRegex(ReferenceRefusal, "aliased"):
            run(search, lambda rng: ToyWorld(root=drift), root=drift)
        with self.assertRaisesRegex(ReferenceRefusal, "reset"):
            run(search, lambda rng: ToyWorld())

    def test_sampled_world_may_not_change_known_legality(self):
        world = ToyWorld(root=replace(ROOT, actions=("switch:hidden",)))
        with self.assertRaisesRegex(ReferenceRefusal, "player-known"):
            run(searcher(), lambda rng: world)
        self.assertTrue(world.closed)

    def test_invalid_inference_has_no_accepted_backup(self):
        search = searcher()
        world = ToyWorld(depth=2)
        world.evaluate = lambda state: Evaluation((1.,), float("nan"))
        with self.assertRaises(ReferenceRefusal):
            run(search, lambda rng: world)
        self.assertEqual(search.nodes[ROOT.key].count, 0)
        self.assertEqual(set(search.nodes), {ROOT.key})
        self.assertTrue(world.closed)

    def test_safety_cap_refuses_instead_of_pricing_depth_truncation(self):
        search = searcher(ReferenceConfig(.5, 1, max_transitions=2))
        # A previously expanded chain is needed to reach the safety limit.
        for n in (1, 2):
            state = DecisionState(f"leaf:{n}".encode(), ("move:a",), 0)
            search.nodes[state.key] = Node.from_evaluation(state, Evaluation((1.,), 0))
        with self.assertRaisesRegex(ReferenceRefusal, "safety limit"):
            run(search, lambda rng: ToyWorld(depth=8))
        self.assertEqual(search.nodes[ROOT.key].count, 0)

    def test_deadline_zero_new_work_cannot_accept_historical_visits(self):
        search = searcher()
        run(search, lambda rng: ToyWorld())
        times = iter((0., 2.))
        with self.assertRaisesRegex(ReferenceRefusal, "zero complete"):
            run(search, lambda rng: ToyWorld(), deadline_seconds=1., clock=lambda: next(times))

    def test_incomplete_trajectory_does_not_backup_and_cleanup_is_timed(self):
        search = searcher()
        now = [0.]
        worlds = []
        def factory(rng):
            world = ToyWorld(value=1)
            worlds.append(world)
            original = world.advance
            def advance(*args):
                now[0] += .6
                return original(*args)
            world.advance = advance
            return world
        result = run(search, factory, trajectories=4, deadline_seconds=1., clock=lambda: now[0])
        self.assertEqual((result.world_draws, result.trajectories, sum(result.root_visits)), (2, 1, 1))
        self.assertTrue(result.deadline_exhausted)
        self.assertAlmostEqual(result.deadline_overrun_seconds, .2)
        self.assertTrue(all(w.closed for w in worlds))

    def test_opponent_only_boundary_does_not_fabricate_subject_action(self):
        trace = []
        world = ToyWorld(trace=trace)
        def advance(subject, opponent, chance):
            trace.append((subject, opponent))
            if len(trace) == 1:
                return Frame(None, ("switch:opponent",), (1.,))
            return Terminal(1)
        world.advance = advance
        run(searcher(), lambda rng: world, trajectories=1)
        self.assertIsNone(trace[1][0])
        self.assertEqual(trace[1][1], "switch:opponent")

    def test_cleanup_failure_poisoned_even_after_terminal_backup(self):
        search = searcher()
        world = ToyWorld()
        def close():
            raise RuntimeError("cleanup failed")
        world.close = close
        with self.assertRaisesRegex(RuntimeError, "cleanup"):
            run(search, lambda rng: world)
        with self.assertRaisesRegex(ReferenceRefusal, "reset"):
            run(search, lambda rng: ToyWorld())

    def test_invalid_configuration_and_masked_probability_rows_refuse(self):
        for alpha, beta in ((1.4, 1), (0, -1), (True, 1), (float("nan"), 0)):
            with self.subTest(alpha=alpha), self.assertRaises(ReferenceRefusal):
                ReferenceConfig(alpha, beta)
        for priors in ((.4, .4), (-.1, 1.1), (float("nan"), 1), (True, 0)):
            with self.subTest(priors=priors), self.assertRaises(ReferenceRefusal):
                Node.from_evaluation(ROOT, Evaluation(priors, 0))


if __name__ == "__main__":
    unittest.main()
