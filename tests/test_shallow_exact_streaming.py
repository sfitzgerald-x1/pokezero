"""Exact scheduling/memory controls; not native parity or strength evidence."""
from dataclasses import dataclass
import math
import unittest
import weakref
from unittest.mock import patch

from pokezero.shallow_exact import ChanceBranch, ExactSearchRefused, exact_depth_two
from tests.test_shallow_exact import ToyBackend


class StreamingTests(unittest.TestCase):
    def compare(self, factory, *, batch_size=7, **options):
        results = [exact_depth_two([(i, ()) for i in range(4)], factory(),
                    batch_size=batch_size, streaming=mode, **options) for mode in (False, True)]
        left, right = results
        for name in ("row_actions", "column_actions", "root_matrix", "equilibrium", "worlds",
                     "decision_nodes", "joint_actions", "chance_branches", "leaf_evaluations",
                     "max_interior_solver_gap", "conservative_tree_gap"):
            self.assertEqual(getattr(left, name), getattr(right, name), name)
        self.assertLessEqual(right.peak_pending_leaf_positions, batch_size)
        return results

    def test_nonduplicate_worlds_and_nontrivial_interior_games(self):
        class Mixed(ToyBackend):
            def evaluate_leaves(self, states):
                result = []
                for world, history in states:
                    row, col, chance = history[-1]
                    payoff = {("a", "x"): 1, ("a", "y"): -1,
                              ("b", "x"): -1, ("b", "y"): 0}[row, col]
                    root_sign = 1 if history[0][:2] in (("a", "x"), ("b", "y")) else -1
                    result.append(root_sign * payoff * chance * 0.5 + world * 0.0625)
                return result
        for size in (1, 7, 65):
            with self.subTest(size=size):
                left, right = self.compare(Mixed, batch_size=size)
                self.assertGreater(right.max_interior_solver_gap, 0)
                self.assertEqual(right.conservative_tree_gap,
                                 right.equilibrium.gap + 2 * right.max_interior_solver_gap)

    def test_mixed_terminal_model_order_and_absolute_seat_values(self):
        class Mixed(ToyBackend):
            def outcomes(self, state, row, col):
                self.transitions += 1
                world, history = state
                return [ChanceBranch((world, history + ((row, col, i),)), probability)
                        for i, probability in enumerate((0.1, 0.2, 0.7))]
            def terminal_value(self, state):
                world, history = state
                return (0.3 if world % 2 else -0.2) if history and history[-1][2] == 1 else None
            def evaluate_leaves(self, states):
                return [(0.1 if history[-1][2] == 0 else 0.7) * (1 if world % 2 else -1)
                        for world, history in states]
        for size in (1, 2, 5, 100):
            with self.subTest(size=size):
                self.compare(Mixed, batch_size=size)

    def test_terminal_only_paths_need_no_forward(self):
        class Terminal(ToyBackend):
            def terminal_value(self, state):
                return -0.5 if state[1] else None
        _, result = self.compare(Terminal)
        self.assertEqual((result.leaf_evaluations, result.evaluation_batches,
                          result.peak_pending_leaf_positions), (0, 0, 0))

    def test_literal_four_world_nine_by_nine_single_chance_counts(self):
        class Nine(ToyBackend):
            def actions(self, state, *, root):
                return tuple(range(9)), tuple(range(9))
            def outcomes(self, state, row, col):
                world, history = state
                return [ChanceBranch((world, history + ((row, col, 0),)), 1.0)]
            def evaluate_leaves(self, states):
                return [0.0] * len(states)
        left, right = self.compare(Nine, batch_size=128)
        self.assertEqual((right.leaf_evaluations, right.joint_actions, right.chance_branches),
                         (26244, 26568, 26568))
        self.assertEqual(right.peak_pending_leaf_positions, 81)
        self.assertEqual((left.evaluation_batches, right.evaluation_batches), (206, 324))

    def test_later_cap_refuses_whole_decision_after_prior_forwards(self):
        for options, message in (({"max_leaves": 15}, "leaf cap"),
                                  ({"max_joint_actions": 3}, "joint-action cap"),
                                  ({"max_chance_branches": 5}, "materialization cap")):
            backend = ToyBackend()
            with self.subTest(options=options), self.assertRaisesRegex(ExactSearchRefused, message):
                exact_depth_two([(0, ())], backend, batch_size=1, streaming=True, **options)
            self.assertTrue(backend.batches)

    def test_complete_root_fan_validation_precedes_any_forward(self):
        class Missing(ToyBackend):
            def outcomes(self, state, row, col):
                return [ChanceBranch((0, ((row, col, 0),)), 0.9)]
        backend = Missing()
        with self.assertRaisesRegex(ExactSearchRefused, "unit mass"):
            exact_depth_two([(0, ())], backend, streaming=True)
        self.assertEqual(backend.batches, [])

    def test_later_bad_chance_fan_never_becomes_partial_action(self):
        class Missing(ToyBackend):
            def outcomes(self, state, row, col):
                if state[1] and state[1][0][:2] == ("a", "y"):
                    return [ChanceBranch((0, ()), 0.9)]
                return super().outcomes(state, row, col)
        backend = Missing()
        with self.assertRaisesRegex(ExactSearchRefused, "unit mass"):
            exact_depth_two([(0, ())], backend, batch_size=2, streaming=True)
        self.assertTrue(backend.batches)

    def test_cardinality_scale_deadline_and_solver_refusals_survive(self):
        for kind in ("cardinality", "scale", "nan"):
            class Bad(ToyBackend):
                def evaluate_leaves(self, states):
                    return [] if kind == "cardinality" else [2 if kind == "scale" else math.nan] * len(states)
            with self.subTest(kind=kind), self.assertRaises(ExactSearchRefused):
                exact_depth_two([(0, ())], Bad(), batch_size=2, streaming=True)
        clock = [0.0]
        class Slow(ToyBackend):
            def evaluate_leaves(self, states):
                clock[0] = 2
                return [0.0] * len(states)
        with patch("pokezero.shallow_exact.time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(ExactSearchRefused, "deadline"):
                exact_depth_two([(0, ())], Slow(), streaming=True, max_seconds=1)
        class Nonconvergent(ToyBackend):
            def evaluate_leaves(self, states):
                return [{("a", "x"): 1, ("a", "y"): -1,
                         ("b", "x"): -1, ("b", "y"): 0}[history[-1][:2]] for _, history in states]
        with self.assertRaisesRegex(ExactSearchRefused, "matrix solver exceeded"):
            exact_depth_two([(0, ())], Nonconvergent(), streaming=True, solver_max_iterations=1)

    def test_action_identity_admission_and_reset_still_precede_work(self):
        class Different(ToyBackend):
            def __init__(self):
                super().__init__()
                self.resets = 0
            def begin_decision(self):
                self.resets += 1
            def actions(self, state, *, root):
                self.asserted_reset = self.resets == 1
                return (("other" if state[0] else "a", "b"), ("x", "y"))
        backend = Different()
        with self.assertRaisesRegex(ExactSearchRefused, "identities differ"):
            exact_depth_two([(0, ()), (1, ())], backend, streaming=True)
        self.assertTrue(backend.asserted_reset)
        self.assertEqual((backend.resets, backend.transitions, backend.batches), (1, 0, []))

    def test_live_positions_are_fan_and_batch_bounded_not_whole_tree(self):
        @dataclass(eq=False)
        class State:
            world: int
            history: tuple
        class Tracking:
            def __init__(self):
                self.live = weakref.WeakSet()
                self.peak = 0
            def state(self, world, history):
                value = State(world, history)
                self.live.add(value)
                self.peak = max(self.peak, len(self.live))
                return value
            def terminal_value(self, state):
                return None
            def actions(self, state, *, root):
                return (0, 1, 2), (0, 1, 2)
            def outcomes(self, state, row, col):
                return [ChanceBranch(self.state(state.world, state.history + ((row, col, i),)), 1 / 3)
                        for i in range(3)]
            def evaluate_leaves(self, states):
                return [0.0] * len(states)  # Never retain a model-input state.
        peaks = []
        for mode in (False, True):
            backend = Tracking()
            worlds = [backend.state(i, ()) for i in range(2)]
            result = exact_depth_two(worlds, backend, batch_size=7, streaming=mode)
            self.assertEqual(result.leaf_evaluations, 1458)
            peaks.append(backend.peak)
            if mode:
                self.assertEqual(result.peak_pending_leaf_positions, 7)
        self.assertGreater(peaks[0], 1458)
        self.assertLessEqual(peaks[1], 7 + 3 + 3 + 2 + 8)


if __name__ == "__main__":
    unittest.main()
