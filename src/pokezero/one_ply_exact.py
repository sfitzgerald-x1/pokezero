"""October 24 P2 core: one exact-chance joint ply, then a root LP.

Not a gameplay policy or scientific registration. Root identities must match
across equally weighted worlds. Nonterminal forced replacements are leaves,
not losses or extra search plies. Backend tanh values become (v+1)/2 exactly
once; draws therefore score 1/2. These are champion-implied scores, not a
calibration claim. Exact refers to expansion, not floating-point arithmetic.

The prospective support-floor choice is NONE. Sampling does not prune or
flatten the LP mix. A caller must persist the returned sampling receipt and
bind solver/runtime versions, limits and the sampling domain before games.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time
from typing import Any, Hashable, Sequence

from .shallow_exact import ExactBackend, ExactSearchRefused, MatrixSolution


@dataclass(frozen=True)
class LPMatrixSolution(MatrixSolution):
    solver: str
    scipy_version: str
    tolerance: float
    primal_status: int
    dual_status: int
    primal_value: float
    dual_value: float


def solve_matrix_game_lp(matrix: Sequence[Sequence[float]], *,
                         tolerance: float = 1e-8,
                         deadline: float | None = None) -> LPMatrixSolution:
    """Primal/dual HiGHS dual-simplex LP; certify the RETURNED mixes.

    No RM fallback or random initialization. Degenerate games need not have a
    unique mix across solver versions; freeze the version for a collection.
    Report the best-response gap, not a claim of symbolic exactness.
    """
    rows = tuple(tuple(float(v) for v in row) for row in matrix)
    if (not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows)
            or any(not math.isfinite(v) for r in rows for v in r)
            or not math.isfinite(tolerance) or not 1e-9 <= tolerance <= 1e-3
            or (deadline is not None and not math.isfinite(deadline))):
        raise ExactSearchRefused("invalid finite matrix or LP limits")
    try:
        import scipy
        from scipy.optimize import linprog
    except ImportError as error:
        raise ExactSearchRefused("one-ply LP requires the scipy analysis extra") from error
    n, m = len(rows), len(rows[0])

    def run(c, inequalities, count):
        options = {"primal_feasibility_tolerance": tolerance / 10,
                   "dual_feasibility_tolerance": tolerance / 10}
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ExactSearchRefused("decision deadline reached before LP")
            options["time_limit"] = remaining
        result = linprog(c, A_ub=inequalities, b_ub=[0.0]*len(inequalities),
                         A_eq=[[1.0]*count+[0.0]], b_eq=[1.0],
                         bounds=[(0.0, 1.0)]*count+[(None, None)],
                         method="highs-ds", options=options)
        if deadline is not None and time.monotonic() >= deadline:
            raise ExactSearchRefused("decision deadline reached in LP")
        if not result.success or result.status != 0:
            raise ExactSearchRefused(f"LP refused: status={result.status}: {result.message}")
        values = tuple(float(v) for v in result.x)
        if (len(values) != count+1 or any(not math.isfinite(v) for v in values)
                or any(v < -tolerance or v > 1+tolerance for v in values[:-1])
                or abs(math.fsum(values[:-1])-1) > tolerance):
            raise ExactSearchRefused("LP returned an infeasible distribution")
        # Repair only floating-point bound noise, then certify this actual mix.
        clipped = tuple(max(0.0, v) for v in values[:-1])
        total = math.fsum(clipped)
        if total <= 0:
            raise ExactSearchRefused("LP returned zero mass")
        return tuple(v/total for v in clipped), values[-1], int(result.nit)

    # max v: A^T x >= v, versus min u: A y <= u.
    x, primal, ni = run([0.0]*n+[-1.0],
                        [[-rows[i][j] for i in range(n)]+[1.0] for j in range(m)], n)
    y, dual, nj = run([0.0]*m+[1.0], [list(r)+[-1.0] for r in rows], m)
    lower = min(math.fsum(x[i]*rows[i][j] for i in range(n)) for j in range(m))
    upper = max(math.fsum(a*p for a, p in zip(r, y)) for r in rows)
    if (upper-lower > tolerance or upper-lower < -tolerance
            or abs(primal-lower) > tolerance or abs(dual-upper) > tolerance):
        raise ExactSearchRefused("LP certificate exceeds returned-mix gap tolerance")
    value = math.fsum(x[i]*rows[i][j]*y[j] for i in range(n) for j in range(m))
    return LPMatrixSolution(x, y, value, lower, upper, ni+nj, "highs-ds",
                            scipy.__version__, tolerance, 0, 0, primal, dual)


@dataclass(frozen=True)
class OnePlyResult:
    row_actions: tuple[Hashable, ...]
    column_actions: tuple[Hashable, ...]
    root_matrix: tuple[tuple[float, ...], ...]
    equilibrium: LPMatrixSolution
    worlds: int
    joint_actions: int
    chance_branches: int
    leaf_evaluations: int
    evaluation_batches: int
    max_chance_mass_error: float
    support_floor: str = "NONE"


def exact_one_ply(worlds: Sequence[Any], backend: ExactBackend, *,
                  batch_size: int = 256, max_leaves: int = 100_000,
                  max_chance_branches: int = 200_000,
                  max_joint_actions: int = 10_000, max_seconds: float = 60.0,
                  solver_tolerance: float = 1e-8,
                  cross_cell_batches: bool = False) -> OnePlyResult:
    """Historical common-matrix game; root identities must match on both sides."""
    actions, matrices, work, deadline = _exact_world_matrices(
        worlds, backend, batch_size=batch_size, max_leaves=max_leaves,
        max_chance_branches=max_chance_branches, max_joint_actions=max_joint_actions,
        max_seconds=max_seconds, cross_cell_batches=cross_cell_batches)
    row_actions, column_actions = actions[0]
    matrix = tuple(tuple(math.fsum(m[i][j]/len(worlds) for m in matrices)
                         for j in range(len(column_actions)))
                   for i in range(len(row_actions)))
    equilibrium = solve_matrix_game_lp(matrix, tolerance=solver_tolerance, deadline=deadline)
    return OnePlyResult(row_actions, column_actions, matrix, equilibrium, len(worlds), **work)


def _exact_world_matrices(worlds: Sequence[Any], backend: ExactBackend, *,
                          batch_size: int, max_leaves: int,
                          max_chance_branches: int, max_joint_actions: int,
                          max_seconds: float, cross_cell_batches: bool,
                          acting_side: int | None = None):
    """Complete one-turn expectations; no successor actions are requested.

    Positive native probability mass must sum to 1 within 1e-6. The complete
    fan is normalized only for that float representation error, which is
    reported. Never drop, sample or renormalize a refused branch. Cooperative
    checks are not a hard timeout for a blocked native call: the eventual
    collection owner must supply a hard deadline and native allocation bounds.
    ``cross_cell_batches`` is explicit opt-in: retain world/cell/branch targets
    while filling batches across cells, without changing a single expectation.
    """
    if ((acting_side is not None and (type(acting_side) is not int or acting_side not in (0, 1)))
            or type(cross_cell_batches) is not bool or not worlds or any(type(v) is not int or v <= 0 for v in
            (batch_size, max_leaves, max_chance_branches, max_joint_actions))
            or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ExactSearchRefused("nonempty worlds and positive work limits required")
    deadline = time.monotonic()+max_seconds
    leaves = branches_count = batches = joint_actions = 0
    worlds_enumerated = cells_enumerated = 0
    world_index = row_action = column_action = branch_index = None
    phase = 'decision_reset'
    pending, targets = [], []
    # Additive diagnostics hook: bind this list once, never copy it per leaf
    # or batch. A deadline before evaluate_leaves must retain queued rows too.
    bind_pending = getattr(backend, 'bind_pending', None)
    if bind_pending is not None:
        if not callable(bind_pending):
            raise ExactSearchRefused('backend pending-list binding is not callable')
        bind_pending(pending)

    def refusal(message, code, *, now=None):
        error = ExactSearchRefused(message)
        error.failure_codes = [code]
        error.work = dict(phase=phase, world_ordinal=world_index,
                row_action=row_action, column_action=column_action, branch_index=branch_index,
                worlds_enumerated=worlds_enumerated, cells_enumerated=cells_enumerated,
                joint_actions_planned=joint_actions, chance_branches_checked=branches_count,
                leaves_queued=leaves-int(code == 'p2.decision.leaf_cap'),
                leaves_attempted=leaves, batches_accepted=batches, pending_leaves=len(pending),
                limits=dict(max_leaves=max_leaves, max_chance_branches=max_chance_branches,
                            max_joint_actions=max_joint_actions, batch_size=batch_size,
                            max_seconds=max_seconds),
                elapsed_seconds=(time.monotonic() if now is None else now)-deadline+max_seconds,
                count_basis='Counters advance at existing guarded checkpoints. Nonterminal '
                    'leaves_attempted includes a cap-rejected current leaf; leaves_queued excludes it. '
                    'In-flight work may be excluded.')
        return error

    def check():
        now = time.monotonic()
        if now >= deadline:
            raise refusal("decision deadline reached; no partial decision",
                          'p2.decision.deadline', now=now)

    def score(value):
        value = float(value)
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ExactSearchRefused("backend value outside side-one tanh scale")
        return (value+1)/2

    begin = getattr(backend, "begin_decision", None)
    if begin is not None:
        if not callable(begin):
            raise ExactSearchRefused("backend decision reset is not callable")
        begin()
        check()
    actions = []
    phase = 'root_actions'
    for world_index, world in enumerate(worlds):
        check()
        if backend.terminal_value(world) is not None:
            raise ExactSearchRefused("root worlds must be nonterminal")
        pair = tuple(tuple(side) for side in backend.actions(world, root=True))
        check()
        if len(pair) != 2 or any(not s or len(set(s)) != len(s) for s in pair):
            raise ExactSearchRefused("unique nonempty root action identities required")
        actions.append(pair)
    if acting_side is None:
        if any(a != actions[0] for a in actions):
            raise ExactSearchRefused("root action identities differ across worlds")
    else:
        own_actions = actions[0][acting_side]
        if any(set(a[acting_side]) != set(own_actions) for a in actions):
            raise ExactSearchRefused("acting-player action identities differ across worlds")
        # Semantic identity, not native list position, owns our common mix.
        actions = [tuple(own_actions if s == acting_side else a[s] for s in (0, 1))
                   for a in actions]
    joint_actions = sum(len(a[0])*len(a[1]) for a in actions)
    if joint_actions > max_joint_actions:
        raise refusal("joint-action cap; no partial decision", 'p2.decision.joint_action_cap')
    contributions = []
    mass_error = 0.0

    def flush():
        nonlocal batches, phase
        if not pending:
            return
        check()
        phase = 'leaf_evaluation'
        values = list(backend.evaluate_leaves(pending))
        check()
        if len(values) != len(pending):
            raise ExactSearchRefused("leaf evaluator changed cardinality")
        # Each target is its exact world/cell/branch position, including
        # terminal holes. No values, weights or histories are deduplicated.
        for (terms, position, weight), value in zip(targets, values):
            if position is None:
                terms.append(weight*score(value))  # Preserve legacy deferred-leaf order.
            else:
                terms[position] = weight*score(value)
        batches += 1
        pending.clear()
        targets.clear()

    for world_index, (world, (row_actions, column_actions)) in enumerate(zip(worlds, actions)):
        cells = [[[] for _ in column_actions] for _ in row_actions]
        contributions.append(cells)
        for i, row in enumerate(row_actions):
            for j, col in enumerate(column_actions):
                phase = 'transition_fan'
                row_action, column_action, branch_index = row, col, None
                check()
                fan = []
                for branch_index, branch in enumerate(backend.outcomes(world, row, col)):
                    check()
                    branches_count += 1
                    if branches_count > max_chance_branches:
                        raise refusal("chance-branch cap; no partial decision",
                                      'p2.decision.chance_branch_cap')
                    if not math.isfinite(branch.probability) or branch.probability <= 0:
                        raise ExactSearchRefused("positive finite chance mass required")
                    fan.append(branch)
                mass = math.fsum(b.probability for b in fan)
                if not fan or not math.isfinite(mass) or abs(mass-1) > 1e-6:
                    raise ExactSearchRefused("complete chance fan must retain unit mass")
                mass_error = max(mass_error, abs(mass-1))
                terms = []
                for branch_index, branch in enumerate(fan):
                    phase = 'leaf_terminal_check'
                    check()
                    terminal = backend.terminal_value(branch.state)
                    check()
                    weight = branch.probability/mass
                    if terminal is not None:
                        terms.append(weight*score(terminal))
                    else:
                        leaves += 1
                        if leaves > max_leaves:
                            raise refusal("leaf cap; no partial decision", 'p2.decision.leaf_cap')
                        pending.append(branch.state)
                        targets.append((terms, len(terms) if cross_cell_batches else None, weight))
                        if cross_cell_batches:
                            terms.append(None)
                        if len(pending) == batch_size:
                            flush()
                if not cross_cell_batches:
                    flush()
                cells[i][j] = terms
                cells_enumerated += 1
        worlds_enumerated += 1
    flush()
    phase = 'matrix_assembly'
    matrices = tuple(tuple(tuple(math.fsum(terms) for terms in row) for row in cells)
                     for cells in contributions)
    check()
    work = dict(joint_actions=joint_actions, chance_branches=branches_count,
                leaf_evaluations=leaves, evaluation_batches=batches,
                max_chance_mass_error=mass_error)
    return tuple(actions), matrices, work, deadline


def sample_equilibrium(result: OnePlyResult, *, game_seed: int, turn: int,
                       side: int, decision_index: int = 0) -> dict:
    """Versioned stateless draw; NONE support floor; persist the whole receipt.

    Seat and decision ordinal distinguish replacement requests in one turn.
    They are declared seed inputs, not mutable RNG state or runtime timing.
    """
    if (type(game_seed) is not int or not 0 <= game_seed < 2**64
            or any(type(v) is not int or v < 0 for v in (turn, decision_index))
            or type(side) is not int or side not in (0, 1)
            or result.support_floor != "NONE"):
        raise ExactSearchRefused("invalid fixed sampling domain or support floor")
    mix = result.equilibrium.row_mix if side == 0 else result.equilibrium.column_mix
    actions = result.row_actions if side == 0 else result.column_actions
    if (len(mix) != len(actions) or not mix or any(not math.isfinite(p) or p < 0 for p in mix)
            or abs(math.fsum(mix)-1) > 1e-10):
        raise ExactSearchRefused("invalid played equilibrium mix")
    payload = ["pokezero.one-ply-exact.sample.v1", game_seed, turn, side, decision_index]
    digest = sha256(json.dumps(payload, separators=(",", ":")).encode("ascii")).digest()
    uniform = (int.from_bytes(digest, "big") >> 203)/2**53
    cumulative = 0.0
    chosen = max(i for i, p in enumerate(mix) if p > 0)
    for i, p in enumerate(mix):
        cumulative += p
        if uniform < cumulative:
            chosen = i
            break
    return dict(sampling_domain=payload, seed_sha256=digest.hex(), uniform=uniform,
                support_floor="NONE", actions=actions, mix=mix,
                action_index=chosen, action=actions[chosen])
