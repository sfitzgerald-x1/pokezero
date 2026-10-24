"""Full-width two-ply zero-sum search over an explicit chance backend.

This is the mathematical core, not a deployable policy. The backend must carry
the complete engine state *and* observation-history context in each state. It
must enumerate every positive-probability outcome; sampled or collapsed damage
rolls do not satisfy this interface. Native integration owns that guarantee.

``exact`` describes the action/chance expansion, not the finite numerical game
solver. Regret matching returns its actual saddle-point gap and refuses when
its declared iteration budget cannot meet the tolerance. Work caps likewise
refuse the whole decision: they never silently prune actions or chance mass.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Hashable, Iterable, Protocol, Sequence


class ExactSearchRefused(ValueError):
    """A complete decision could not be computed within its explicit contract."""


@dataclass(frozen=True)
class MatrixSolution:
    row_mix: tuple[float, ...]
    column_mix: tuple[float, ...]
    value: float
    lower_value: float
    upper_value: float
    iterations: int

    @property
    def gap(self) -> float:
        return self.upper_value - self.lower_value


def _positive_mix(regrets: Sequence[float]) -> tuple[float, ...]:
    positive = tuple(max(0.0, value) for value in regrets)
    total = sum(positive)
    return (tuple(value / total for value in positive) if total > 0.0
            else (1.0 / len(positive),) * len(positive))


def solve_matrix_game(
    matrix: Sequence[Sequence[float]], *, tolerance: float = 0.001,
    max_iterations: int = 16384, deadline: float | None = None,
) -> MatrixSolution:
    """Alternating regret matching plus; row maximizes, column minimizes.

    Average strategies, not the last iterate or decoupled argmax, are returned.
    ``gap = max_i (A y)_i - min_j (x A)_j`` is directly computed from the
    returned distributions. The finite-solver error is therefore observable.
    Each row regret update is followed by a column update against the new row
    mix. Cumulative regrets are clipped at zero (RM+); averages remain uniform.
    """
    rows = tuple(tuple(float(value) for value in row) for row in matrix)
    if (not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows)
            or any(not math.isfinite(v) for row in rows for v in row)):
        raise ExactSearchRefused("matrix must be nonempty, rectangular and finite")
    if (not math.isfinite(tolerance) or tolerance <= 0
            or type(max_iterations) is not int or max_iterations <= 0
            or (deadline is not None and not math.isfinite(deadline))):
        raise ExactSearchRefused("invalid matrix-solver limits")
    n, m = len(rows), len(rows[0])
    row_regret, col_regret = [0.0] * n, [0.0] * m
    row_sum, col_sum = [0.0] * n, [0.0] * m
    for step in range(1, max_iterations + 1):
        if deadline is not None and (step == 1 or step % 32 == 0) and time.monotonic() >= deadline:
            raise ExactSearchRefused("decision deadline reached in matrix solver")
        x, y = _positive_mix(row_regret), _positive_mix(col_regret)
        row_values = [sum(a * p for a, p in zip(row, y)) for row in rows]
        current = sum(p * value for p, value in zip(x, row_values))
        for i in range(n):
            row_regret[i] = max(0.0, row_regret[i] + row_values[i] - current)
        x = _positive_mix(row_regret)
        col_values = [sum(x[i] * rows[i][j] for i in range(n)) for j in range(m)]
        current = sum(p * value for p, value in zip(y, col_values))
        for j in range(m):
            col_regret[j] = max(0.0, col_regret[j] + current - col_values[j])
            col_sum[j] += y[j]
        for i in range(n):
            row_sum[i] += x[i]
        # Gap evaluation is cheaper than encoding, but need not run every round.
        if step == 1 or step % 32 == 0 or step == max_iterations:
            avg_x = tuple(value / step for value in row_sum)
            avg_y = tuple(value / step for value in col_sum)
            upper = max(sum(a * p for a, p in zip(row, avg_y)) for row in rows)
            lower = min(sum(avg_x[i] * rows[i][j] for i in range(n)) for j in range(m))
            if upper - lower <= tolerance:
                value = sum(avg_x[i] * rows[i][j] * avg_y[j]
                            for i in range(n) for j in range(m))
                return MatrixSolution(avg_x, avg_y, value, lower, upper, step)
    raise ExactSearchRefused(
        f"matrix solver exceeded {max_iterations} rounds: gap={upper - lower:.9g}"
    )


@dataclass(frozen=True)
class ChanceBranch:
    state: Any
    probability: float


class ExactBackend(Protocol):
    """Backend values are side-one champion values on the fixed [-1, 1] scale.

    A backend may expose ``begin_decision()`` to reset exact input-value caches;
    the core calls it once before any root validation/expansion.
    """

    def terminal_value(self, state: Any) -> float | None: ...
    def actions(self, state: Any, *, root: bool) -> tuple[Sequence[Hashable], Sequence[Hashable]]: ...
    def outcomes(self, state: Any, row_action: Hashable, column_action: Hashable) -> Iterable[ChanceBranch]: ...
    def evaluate_leaves(self, states: Sequence[Any]) -> Sequence[float]: ...


@dataclass(frozen=True)
class DepthTwoResult:
    row_actions: tuple[Hashable, ...]
    column_actions: tuple[Hashable, ...]
    root_matrix: tuple[tuple[float, ...], ...]
    equilibrium: MatrixSolution
    worlds: int
    decision_nodes: int
    joint_actions: int
    chance_branches: int
    leaf_evaluations: int
    evaluation_batches: int
    max_interior_solver_gap: float
    conservative_tree_gap: float


@dataclass
class _MatrixNode:
    actions: tuple[tuple[Hashable, ...], tuple[Hashable, ...]]
    cells: list[list[list[tuple[float, Any]]]]


@dataclass(frozen=True)
class _Leaf:
    index: int


def exact_depth_two(
    worlds: Sequence[Any], backend: ExactBackend, *, batch_size: int = 256,
    max_leaves: int = 1_000_000, max_joint_actions: int = 1_000_000,
    max_chance_branches: int = 1_000_000, max_seconds: float = 60.0,
    solver_tolerance: float = 0.001, solver_max_iterations: int = 16384,
) -> DepthTwoResult:
    """Uniform-world expectation, two joint-action plies, equilibrium at each.

    Root action *identities* and order must agree across worlds. Hidden move
    indices are not silently equated with different moves. Native policy
    integration must declare how world-dependent opponent actions are coupled
    before relaxing this guard. Interior matrices are solved per outcome/world;
    that determinization assumption is explicit, not an information-set proof.
    A ply here is one backend decision boundary, including a forced replacement
    if exposed by that backend, not necessarily one ordinary battle turn. The
    collection registration must fix this choice; this core does not silently
    traverse forced choices without charging a ply.

    Deadlines are cooperative checks around backend calls and solver rounds,
    not preemption of a blocked native call. Integration must also use an owned
    subprocess hard deadline and bound allocations inside its outcome producer.
    ``conservative_tree_gap`` adds twice the largest interior gap to the root
    gap; the root gap alone certifies only the numerically backed-up matrix.
    """
    if (not worlds or any(type(value) is not int or value <= 0
                          for value in (batch_size, max_leaves, max_joint_actions, max_chance_branches))
            or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ExactSearchRefused("nonempty worlds and positive integer work limits required")
    deadline = time.monotonic() + max_seconds

    def check_deadline():
        if time.monotonic() >= deadline:
            raise ExactSearchRefused("decision deadline reached; no partial decision is valid")
    begin = getattr(backend, "begin_decision", None)
    if begin is not None:
        if not callable(begin):
            raise ExactSearchRefused("backend decision reset is not callable")
        begin()
        check_deadline()
    leaves: list[Any] = []
    counts = {"nodes": 0, "pairs": 0, "branches": 0}

    def checked_actions(state: Any, *, root: bool):
        check_deadline()
        actions = tuple(tuple(side) for side in backend.actions(state, root=root))
        check_deadline()
        if (len(actions) != 2 or any(not side or len(set(side)) != len(side) for side in actions)):
            raise ExactSearchRefused("both seats require unique nonempty action identities")
        return actions

    # Reject mismatched hidden move identities before building potentially huge
    # chance trees. A common slot number is not proof of a common action.
    if any(backend.terminal_value(state) is not None for state in worlds):
        raise ExactSearchRefused("root worlds must all be nonterminal decision states")
    root_actions = [checked_actions(state, root=True) for state in worlds]
    actions = root_actions[0]
    if any(world_actions != actions for world_actions in root_actions):
        raise ExactSearchRefused("root action identities differ across worlds")

    def checked_value(value: Any) -> float:
        value = float(value)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ExactSearchRefused("value outside finite side-one [-1, 1] scale")
        return value

    def expand(state: Any, depth: int, *, root: bool = False, prepared_actions=None) -> Any:
        check_deadline()
        terminal = backend.terminal_value(state)
        check_deadline()
        if terminal is not None:
            return checked_value(terminal)
        if depth == 0:
            if len(leaves) >= max_leaves:
                raise ExactSearchRefused("leaf cap reached; no partial decision is valid")
            leaves.append(state)
            return _Leaf(len(leaves) - 1)
        actions = prepared_actions if prepared_actions is not None else checked_actions(state, root=root)
        counts["nodes"] += 1
        cells = []
        for row_action in actions[0]:
            row = []
            for col_action in actions[1]:
                if counts["pairs"] >= max_joint_actions:
                    raise ExactSearchRefused("joint-action cap reached; no partial decision is valid")
                counts["pairs"] += 1
                check_deadline()
                branches = []
                for branch in backend.outcomes(state, row_action, col_action):
                    check_deadline()
                    if counts["branches"] >= max_chance_branches:
                        raise ExactSearchRefused("chance-branch materialization cap reached; no partial decision is valid")
                    counts["branches"] += 1
                    branches.append(branch)
                check_deadline()
                mass = sum(float(branch.probability) for branch in branches)
                if (not branches or not math.isfinite(mass) or abs(mass - 1.0) > 1e-6
                        or any(not math.isfinite(branch.probability) or branch.probability <= 0
                               for branch in branches)):
                    raise ExactSearchRefused("chance distribution must retain positive unit mass")
                # Only roundoff-level normalization; no outcome is discarded.
                row.append([(branch.probability / mass, expand(branch.state, depth - 1))
                            for branch in branches])
            cells.append(row)
        return _MatrixNode(actions, cells)

    roots = [expand(state, 2, root=True, prepared_actions=world_actions)
             for state, world_actions in zip(worlds, root_actions)]
    values: list[float] = []
    batches = 0
    for start in range(0, len(leaves), batch_size):
        check_deadline()
        batch = leaves[start:start + batch_size]
        evaluated = list(backend.evaluate_leaves(batch))
        check_deadline()
        if len(evaluated) != len(batch):
            raise ExactSearchRefused("leaf evaluator changed batch cardinality")
        values.extend(checked_value(value) for value in evaluated)
        batches += 1
    interior_gap = 0.0

    def matrix_for(node: _MatrixNode) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(sum(probability * backup(child) for probability, child in cell)
                           for cell in row) for row in node.cells)

    def backup(node: Any) -> float:
        nonlocal interior_gap
        if isinstance(node, _Leaf):
            return values[node.index]
        if isinstance(node, _MatrixNode):
            solution = solve_matrix_game(matrix_for(node), tolerance=solver_tolerance,
                                         max_iterations=solver_max_iterations, deadline=deadline)
            interior_gap = max(interior_gap, solution.gap)
            return solution.value
        return node

    root_matrices = [matrix_for(root) for root in roots]
    root_matrix = tuple(tuple(sum(matrix[i][j] for matrix in root_matrices) / len(roots)
                              for j in range(len(actions[1])))
                        for i in range(len(actions[0])))
    equilibrium = solve_matrix_game(root_matrix, tolerance=solver_tolerance,
                                    max_iterations=solver_max_iterations, deadline=deadline)
    check_deadline()
    return DepthTwoResult(actions[0], actions[1], root_matrix, equilibrium, len(roots),
                          counts["nodes"], counts["pairs"], counts["branches"],
                          len(leaves), batches, interior_gap, equilibrium.gap + 2 * interior_gap)
