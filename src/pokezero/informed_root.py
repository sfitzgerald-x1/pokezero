"""One-ply informed-adversary root minimax, not a full Bayesian equilibrium.

One request-known own mix is shared across equally weighted belief worlds.
The modeled adversary observes the sampled root state but not our realized
action draw. Each world can have different legal opponent actions. Exact
chance expansion and the unchanged leaf head remain owned by one_ply_exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time
from typing import Any, Hashable, Sequence

from .one_ply_exact import _exact_world_matrices
from .shallow_exact import ExactBackend, ExactSearchRefused


@dataclass(frozen=True)
class InformedRootSolution:
    own_mix: tuple[float, ...]
    opponent_mixes: tuple[tuple[float, ...], ...]
    value: float
    lower_value: float
    upper_value: float
    world_lower_values: tuple[float, ...]
    solver: str
    scipy_version: str
    tolerance: float
    iterations: int
    primal_value: float
    dual_value: float

    @property
    def gap(self) -> float:
        return self.upper_value-self.lower_value


def solve_informed_root_lp(matrices: Sequence[Sequence[Sequence[float]]], *,
                           tolerance: float = 1e-8,
                           deadline: float | None = None) -> InformedRootSolution:
    """Certify the returned common own mix and per-world adversary mixes.

    max mean(v_w), with B_w^T x >= v_w and sum(x)=1. The dual keeps
    sum(y_w)=1 for EACH world and minimizes max_i mean((B_w y_w)_i).
    No product-of-action-sets enumeration, shared-column aliases, fallback
    solver, unequal world weighting or optimization of independent own mixes.
    """
    try:
        tables = tuple(tuple(tuple(float(v) for v in row) for row in m) for m in matrices)
    except (TypeError, ValueError, OverflowError) as error:
        raise ExactSearchRefused("finite world payoff matrices required") from error
    if (not tables or not tables[0] or not tables[0][0]
            or any(len(m) != len(tables[0]) or not m[0]
                   or any(len(row) != len(m[0]) for row in m) for m in tables)
            or any(not math.isfinite(v) for m in tables for row in m for v in row)
            or not math.isfinite(tolerance) or not 1e-9 <= tolerance <= 1e-3
            or (deadline is not None and not math.isfinite(deadline))):
        raise ExactSearchRefused("nonempty finite same-own-row matrices and valid LP limits required")
    try:
        import scipy
        from scipy.optimize import linprog
    except ImportError as error:
        raise ExactSearchRefused("informed-root LP requires the scipy analysis extra") from error
    count, n = len(tables), len(tables[0])
    widths = tuple(len(m[0]) for m in tables)

    def run(c, inequalities, equalities, rhs, bounds):
        options = {"primal_feasibility_tolerance": tolerance/10,
                   "dual_feasibility_tolerance": tolerance/10}
        if deadline is not None:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise ExactSearchRefused("decision deadline reached before informed-root LP")
            options["time_limit"] = remaining
        result = linprog(c, A_ub=inequalities, b_ub=[0.0]*len(inequalities),
                         A_eq=equalities, b_eq=rhs, bounds=bounds,
                         method="highs-ds", options=options)
        if deadline is not None and time.monotonic() >= deadline:
            raise ExactSearchRefused("decision deadline reached in informed-root LP")
        if not result.success or result.status != 0:
            raise ExactSearchRefused(f"informed-root LP refused: status={result.status}: {result.message}")
        values = tuple(float(v) for v in result.x)
        if len(values) != len(c) or any(not math.isfinite(v) for v in values):
            raise ExactSearchRefused("informed-root LP returned invalid coordinates")
        return values, int(result.nit)

    def distribution(values):
        if (not values or any(v < -tolerance or v > 1+tolerance for v in values)
                or abs(math.fsum(values)-1) > tolerance):
            raise ExactSearchRefused("informed-root LP returned an infeasible distribution")
        clipped = tuple(max(0.0, v) for v in values)
        mass = math.fsum(clipped)
        if mass <= 0:
            raise ExactSearchRefused("informed-root LP returned zero mass")
        return tuple(v/mass for v in clipped)

    primal_constraints = [
        [-m[i][j] for i in range(n)]+[float(k == w) for k in range(count)]
        for w, m in enumerate(tables) for j in range(widths[w])]
    primal, ni = run([0.0]*n+[-1.0/count]*count, primal_constraints,
                     [[1.0]*n+[0.0]*count], [1.0],
                     [(0.0, 1.0)]*n+[(None, None)]*count)
    x = distribution(primal[:n])
    offsets = [0]
    for width in widths:
        offsets.append(offsets[-1]+width)
    columns = offsets[-1]
    dual_constraints = [[m[i][j]/count for m in tables for j in range(len(m[0]))]+[-1.0]
                        for i in range(n)]
    dual_equalities = [[float(offsets[w] <= j < offsets[w+1]) for j in range(columns)]+[0.0]
                       for w in range(count)]
    dual, nj = run([0.0]*columns+[1.0], dual_constraints, dual_equalities,
                   [1.0]*count, [(0.0, 1.0)]*columns+[(None, None)])
    ys = tuple(distribution(dual[offsets[w]:offsets[w+1]]) for w in range(count))
    world_lower = tuple(min(math.fsum(x[i]*m[i][j] for i in range(n))
                            for j in range(len(m[0]))) for m in tables)
    lower = math.fsum(world_lower)/count
    upper = max(math.fsum(m[i][j]*ys[w][j]/count
                          for w, m in enumerate(tables) for j in range(len(m[0])))
                for i in range(n))
    primal_value = math.fsum(primal[n:])/count
    dual_value = dual[-1]
    if (abs(upper-lower) > tolerance or abs(primal_value-lower) > tolerance
            or abs(dual_value-upper) > tolerance
            or any(primal[n+w] > v+tolerance for w, v in enumerate(world_lower))):
        raise ExactSearchRefused("informed-root returned-strategy gap certificate failed")
    value = math.fsum(x[i]*m[i][j]*ys[w][j]/count
                      for w, m in enumerate(tables) for i in range(n) for j in range(len(m[0])))
    return InformedRootSolution(x, ys, value, lower, upper, world_lower,
                                "highs-ds", scipy.__version__, tolerance, ni+nj,
                                primal_value, dual_value)


@dataclass(frozen=True)
class InformedOnePlyResult:
    acting_side: int
    own_actions: tuple[Hashable, ...]
    opponent_actions: tuple[tuple[Hashable, ...], ...]
    world_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    equilibrium: InformedRootSolution
    worlds: int
    joint_actions: int
    chance_branches: int
    leaf_evaluations: int
    evaluation_batches: int
    max_chance_mass_error: float
    information_model: str = "informed_adversary_root_minimax"
    support_floor: str = "NONE"


def exact_informed_one_ply(worlds: Sequence[Any], backend: ExactBackend, *,
                           acting_side: int, batch_size: int = 1024,
                           max_leaves: int = 100_000,
                           max_chance_branches: int = 200_000,
                           max_joint_actions: int = 10_000,
                           max_seconds: float = 60.0, solver_tolerance: float = 1e-8,
                           cross_cell_batches: bool = True) -> InformedOnePlyResult:
    actions, matrices, work, deadline = _exact_world_matrices(
        worlds, backend, acting_side=acting_side, batch_size=batch_size,
        max_leaves=max_leaves, max_chance_branches=max_chance_branches,
        max_joint_actions=max_joint_actions, max_seconds=max_seconds,
        cross_cell_batches=cross_cell_batches)
    # Backend values are side-one scores. Actor two minimizes them: complement
    # AND transpose before optimizing. It never plays an opponent dual mix.
    if acting_side == 1:
        matrices = tuple(tuple(tuple(1-m[i][j] for i in range(len(m)))
                               for j in range(len(m[0]))) for m in matrices)
    solution = solve_informed_root_lp(matrices, tolerance=solver_tolerance, deadline=deadline)
    return InformedOnePlyResult(acting_side, actions[0][acting_side],
                                tuple(a[1-acting_side] for a in actions),
                                matrices, solution, len(worlds), **work)


def sample_informed_equilibrium(result: InformedOnePlyResult, *, game_seed: int,
                                turn: int, decision_index: int = 0) -> dict:
    """Stateless actor-only draw, NONE support floor; persist the full receipt."""
    mix, actions, side = result.equilibrium.own_mix, result.own_actions, result.acting_side
    if (type(game_seed) is not int or not 0 <= game_seed < 2**64
            or any(type(v) is not int or v < 0 for v in (turn, decision_index))
            or type(side) is not int or side not in (0, 1) or result.support_floor != "NONE"
            or result.information_model != "informed_adversary_root_minimax"
            or len(mix) != len(actions) or not mix
            or any(not math.isfinite(p) or p < 0 for p in mix)
            or abs(math.fsum(mix)-1) > 1e-10):
        raise ExactSearchRefused("invalid informed-root sampling contract")
    payload = ["pokezero.informed-root.sample.v1", game_seed, turn, side, decision_index]
    digest = sha256(json.dumps(payload, separators=(",", ":")).encode("ascii")).digest()
    uniform = (int.from_bytes(digest, "big") >> 203)/2**53
    chosen = max(i for i, p in enumerate(mix) if p > 0)
    cumulative = 0.0
    for i, p in enumerate(mix):
        cumulative += p
        if uniform < cumulative:
            chosen = i
            break
    return dict(sampling_domain=payload, seed_sha256=digest.hex(), uniform=uniform,
                information_model=result.information_model, support_floor="NONE",
                actions=actions, mix=mix, action_index=chosen, action=actions[chosen])
