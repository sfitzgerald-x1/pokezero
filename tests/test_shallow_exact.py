"""Authored mathematical fixtures only; these are not real battle measurements."""

import math
import unittest
from unittest.mock import patch

from pokezero.shallow_exact import (
    ChanceBranch, ExactSearchRefused, exact_depth_two, solve_matrix_game,
)


class MatrixTests(unittest.TestCase):
    def test_matching_pennies_returns_mixture_not_argmax(self):
        solution = solve_matrix_game([[1, -1], [-1, 1]])
        self.assertEqual(solution.row_mix, (0.5, 0.5))
        self.assertEqual(solution.column_mix, (0.5, 0.5))
        self.assertEqual(solution.value, 0)
        self.assertEqual(solution.gap, 0)

    def test_asymmetric_mixed_equilibrium(self):
        matrix = [[1, -1], [-1, 0]]
        solution = solve_matrix_game(matrix)
        self.assertLessEqual(solution.gap, 0.001)
        self.assertAlmostEqual(solution.row_mix[0], 1 / 3, delta=0.02)
        self.assertAlmostEqual(solution.column_mix[0], 1 / 3, delta=0.02)
        self.assertAlmostEqual(solution.value, -1 / 3, delta=0.02)

    def test_reported_gap_is_returned_strategies_exploitability(self):
        matrix = [[0.2, 0.7], [0.1, 0.5]]
        solution = solve_matrix_game(matrix, tolerance=0.01)
        upper = max(sum(a * p for a, p in zip(row, solution.column_mix)) for row in matrix)
        lower = min(sum(solution.row_mix[i] * matrix[i][j] for i in range(2)) for j in range(2))
        self.assertEqual(solution.upper_value, upper)
        self.assertEqual(solution.lower_value, lower)
        self.assertLessEqual(solution.gap, 0.01)

    def test_solver_budget_is_refusal_not_unqualified_solution(self):
        with self.assertRaisesRegex(ExactSearchRefused, "exceeded"):
            solve_matrix_game([[1, -1], [-1, 0]], max_iterations=1)

    def test_invalid_matrix_rejected(self):
        for matrix in ([], [[]], [[1], [1, 2]], [[math.nan]], [[math.inf]]):
            with self.subTest(matrix=matrix), self.assertRaises(ExactSearchRefused):
                solve_matrix_game(matrix)

    def test_solver_honors_absolute_deadline(self):
        with patch("pokezero.shallow_exact.time.monotonic", return_value=2.0):
            with self.assertRaisesRegex(ExactSearchRefused, "deadline"):
                solve_matrix_game([[0]], deadline=1.0)


class ToyBackend:
    """Two actions each ply, two non-collapsed chance branches per action pair."""

    def __init__(self):
        self.batches = []
        self.transitions = 0

    def terminal_value(self, state):
        return None

    def actions(self, state, *, root):
        return ("a", "b"), ("x", "y")

    def outcomes(self, state, row, col):
        self.transitions += 1
        world, history = state
        return [ChanceBranch((world, history + ((row, col, -1),)), 0.25),
                ChanceBranch((world, history + ((row, col, 1),)), 0.75)]

    def evaluate_leaves(self, states):
        self.batches.append(list(states))
        # Root matrix is matching pennies scaled by chance expectation 0.5.
        return [(1 if history[0][0:2] in (("a", "x"), ("b", "y")) else -1)
                * history[-1][2] for _, history in states]


class DepthTwoTests(unittest.TestCase):
    def test_two_complete_plies_four_worlds_and_chance_count(self):
        backend = ToyBackend()
        result = exact_depth_two([(index, ()) for index in range(4)], backend, batch_size=7)
        self.assertEqual(result.worlds, 4)
        self.assertEqual(result.decision_nodes, 4 * (1 + 4 * 2))
        self.assertEqual(result.joint_actions, 4 * (4 + 4 * 2 * 4))
        self.assertEqual(result.chance_branches, 4 * (4 * 2 + 4 * 2 * 4 * 2))
        self.assertEqual(result.leaf_evaluations, 4 * 4 * 2 * 4 * 2)
        self.assertEqual(result.evaluation_batches, math.ceil(256 / 7))
        self.assertEqual(result.root_matrix, ((0.5, -0.5), (-0.5, 0.5)))
        self.assertEqual(result.equilibrium.row_mix, (0.5, 0.5))
        self.assertEqual(result.equilibrium.gap, 0)
        self.assertEqual(result.conservative_tree_gap,
                         result.equilibrium.gap + 2 * result.max_interior_solver_gap)
        self.assertTrue(all(len(history) == 2 for batch in backend.batches for _, history in batch))

    def test_leaf_cap_never_evaluates_partial_tree(self):
        backend = ToyBackend()
        with self.assertRaisesRegex(ExactSearchRefused, "leaf cap"):
            exact_depth_two([(0, ())], backend, max_leaves=15)
        self.assertEqual(backend.batches, [])

    def test_joint_action_cap_never_evaluates_partial_tree(self):
        backend = ToyBackend()
        with self.assertRaisesRegex(ExactSearchRefused, "joint-action cap"):
            exact_depth_two([(0, ())], backend, max_joint_actions=3)
        self.assertEqual(backend.batches, [])

    def test_missing_chance_mass_refuses(self):
        class Missing(ToyBackend):
            def outcomes(self, *args):
                return [ChanceBranch((0, ()), 0.9)]
        with self.assertRaisesRegex(ExactSearchRefused, "unit mass"):
            exact_depth_two([(0, ())], Missing())

    def test_terminal_outcomes_are_not_model_evaluated(self):
        class Terminal(ToyBackend):
            def terminal_value(self, state):
                return 1.0 if state[1] else None
        result = exact_depth_two([(0, ())], Terminal())
        self.assertEqual(result.leaf_evaluations, 0)
        self.assertEqual(result.evaluation_batches, 0)
        self.assertEqual(result.root_matrix, ((1.0, 1.0), (1.0, 1.0)))

    def test_terminal_only_chance_fan_is_materialization_bounded(self):
        class TerminalFan(ToyBackend):
            def terminal_value(self, state):
                return 1.0 if state[1] else None
            def outcomes(self, state, row, col):
                for index in range(100):
                    yield ChanceBranch((0, (index,)), 0.01)
        backend = TerminalFan()
        with self.assertRaisesRegex(ExactSearchRefused, "materialization cap"):
            exact_depth_two([(0, ())], backend, max_chance_branches=3)
        self.assertEqual(backend.batches, [])

    def test_deadline_checks_after_backend_call(self):
        clock = [0.0]
        class Slow(ToyBackend):
            def evaluate_leaves(self, states):
                clock[0] = 2.0
                return [0.0] * len(states)
        with patch("pokezero.shallow_exact.time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(ExactSearchRefused, "deadline"):
                exact_depth_two([(0, ())], Slow(), max_seconds=1.0)

    def test_world_action_identity_mismatch_refuses(self):
        class Different(ToyBackend):
            def actions(self, state, *, root):
                return (("different" if state[0] else "a", "b"), ("x", "y"))
        backend = Different()
        with self.assertRaisesRegex(ExactSearchRefused, "identities differ"):
            exact_depth_two([(0, ()), (1, ())], backend)
        self.assertEqual(backend.transitions, 0)

    def test_26000_leaf_estimate_requires_single_outcome_per_joint_action(self):
        class NineActions(ToyBackend):
            def actions(self, state, *, root):
                return tuple(range(9)), tuple(range(9))
            def outcomes(self, state, row, col):
                world, history = state
                return [ChanceBranch((world, history + ((row, col, 0),)), 1.0)]
            def evaluate_leaves(self, states):
                return [0.0] * len(states)
        result = exact_depth_two([(i, ()) for i in range(4)], NineActions())
        self.assertEqual(result.leaf_evaluations, 4 * 81 * 81)
        self.assertEqual(result.joint_actions, 4 * (81 + 81 * 81))

    def test_bad_leaf_cardinality_or_scale_refuses(self):
        for bad in ([], [2.0] * 64, [math.nan] * 64):
            class Bad(ToyBackend):
                def evaluate_leaves(self, states):
                    return bad
            with self.subTest(bad=bad[:1]), self.assertRaises(ExactSearchRefused):
                exact_depth_two([(0, ())], Bad(), batch_size=64)


if __name__ == "__main__":
    unittest.main()
