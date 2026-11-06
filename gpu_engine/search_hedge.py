"""Search-seat reductions over per-world price grids: averaging, the Bayesian zero-sum hedge, and its depth-2 form.

Pure host functions over D's surfaces; no engine, model or device call. A world is a `v5_gumbel.WorldPrices`
(depth 1, or a depth-2 backup's output): G_w[a][b] is the learner's value after own action a and opponent
reply b in posterior world w. A waiting opponent has the single no-action column 0 with unit mass. Worlds are
equally weighted posterior draws (p_w = 1/W), as in D's recipe. Any unknown required cell (None) refuses: the
caller then plays raw, exactly as D refuses a label.

- `averaging`: D's completed-Q, Q(a) = sum_w p_w sum_b pi_w(b) G_w[a][b] with pi_w the head's reply prior,
  then argmax (best response to the predicted reply).
- `bayes_hedge`: the opponent knows its own set (the world); I don't. Over my mixed strategy x on the legal
  actions, maximize sum_w p_w [lam * x'G_w pi_w + (1 - lam) * min_b (x'G_w)_b], solved as the LP
  max sum_w p_w [lam x'G_w pi_w + (1 - lam) t_w] s.t. t_w <= (x'G_w)_b for every reply b of world w.
  lam = 0 is the primary hedge; lam = 1 is averaging. The grids are NOT averaged before solving (that would
  assume the opponent doesn't know its set) and worlds are NOT solved separately (strategy fusion).
- `depth2_hedged_worlds`: each expanded turn-1 cell of a `Depth2Frame` gets the value of its turn-2 grid
  solved as a zero-sum matrix game per world; the result is the same WorldPrices pair, fed to `bayes_hedge`.
  Solving turn 2 per world lets my turn-2 strategy depend on the world: a strategy-fusion approximation that
  is a known limit of this reduction.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from pokezero import v5_gumbel as original

SAMPLE_DOMAIN = 'v2s.search-sample.v1'
PROBABILITY_FLOOR = 1e-9


@dataclass(frozen=True)
class Grid:
    """One world's legal own actions x enumerated replies, with the reply prior; float64."""
    actions: tuple
    replies: tuple
    values: np.ndarray   # [len(actions), len(replies)]
    prior: np.ndarray    # [len(replies)], sums to 1


@dataclass(frozen=True)
class Choice:
    """A reduction's mixed strategy over the nine action indices (zero off the legal set) and its value."""
    strategy: tuple
    value: float
    q: tuple | None = None


def grid(world, root_legal):
    """WorldPrices -> Grid over the legal root actions and the world's replies; None if any cell is unknown."""
    legal = original._mask(root_legal)
    actions = tuple(a for a in range(9) if legal[a])
    if world.opponent_requested:
        replies = tuple(b for b in range(9) if world.opponent_legal[b])
        prior = original._probabilities(world.opponent_logits, world.opponent_legal)
    else:
        replies, prior = (0,), (1.,) + (0.,) * 8
    values = np.empty((len(actions), len(replies)), np.float64)
    for i, a in enumerate(actions):
        for k, b in enumerate(replies):
            v = world.joint_values[a][b]
            if v is None:return None
            values[i, k] = float(v)
    return Grid(actions, replies, values, np.asarray([prior[b] for b in replies], np.float64))


def grids(worlds, root_legal):
    result = tuple(grid(w, root_legal) for w in worlds)
    if not result or any(g is None for g in result):return None
    if len({g.actions for g in result}) != 1:raise ValueError('every world prices the same legal root actions')
    return result


def _strategy(actions, x):
    x = np.where(np.asarray(x, np.float64) < PROBABILITY_FLOOR, 0., np.asarray(x, np.float64))
    total = math.fsum(x)
    if not total > 0:raise ValueError('empty mixed strategy')
    full = [0.] * 9
    for a, p in zip(actions, x):full[a] = float(p) / total
    return tuple(full)


def averaging(worlds, root_legal):
    """D's completed-Q over len(worlds) equally weighted worlds, then argmax (lowest index on ties)."""
    gs = grids(worlds, root_legal)
    if gs is None:return None
    count = len(gs);actions = gs[0].actions
    # Same summation as reduce_completed_q: fsum of prior * value / count, per action, world-major.
    q = [math.fsum(g.prior[k] * g.values[i, k] / count for g in gs for k in range(len(g.replies))) for i in range(len(actions))]
    best = max(range(len(actions)), key=lambda i: (q[i], -i))
    full = [None] * 9
    for a, v in zip(actions, q):full[a] = v
    return Choice(_strategy(actions, [1. if i == best else 0. for i in range(len(actions))]), q[best], tuple(full))


def _linprog(c, a_ub, b_ub, a_eq, b_eq, bounds):
    from scipy.optimize import linprog
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if result.status != 0:raise ValueError('hedge LP failed: ' + str(result.message))
    return result.x


def bayes_hedge(worlds, root_legal, *, lam=0.):
    """The Bayesian zero-sum solve (lam=0) or its blend with the reply-prior average (0<lam<1); None refuses."""
    if not 0. <= lam <= 1.:raise ValueError('blend weight in [0, 1] required')
    gs = grids(worlds, root_legal)
    if gs is None:return None
    n, count = len(gs[0].actions), len(gs)
    # Variables: x (n own probabilities), t (one worst-case value per world). linprog minimizes.
    weight = 1. / count
    c = np.concatenate([-lam * weight * sum(g.values @ g.prior for g in gs), np.full(count, -(1. - lam) * weight)])
    rows = []
    for w, g in enumerate(gs):
        for k in range(len(g.replies)):
            row = np.zeros(n + count);row[:n] = -g.values[:, k];row[n + w] = 1.;rows.append(row)
    a_eq = np.concatenate([np.ones(n), np.zeros(count)])[None]
    z = _linprog(c, np.asarray(rows), np.zeros(len(rows)), a_eq, [1.], [(0., 1.)] * n + [(-1., 1.)] * count)
    strategy = _strategy(gs[0].actions, z[:n])
    return Choice(strategy, hedge_value(gs, strategy, lam=lam))


def hedge_value(gs, strategy, *, lam=0.):
    """The objective at a given strategy: sum_w p_w [lam x'G_w pi_w + (1-lam) min_b (x'G_w)_b]."""
    x = np.asarray([strategy[a] for a in gs[0].actions], np.float64)
    terms = []
    for g in gs:
        payoff = x @ g.values
        terms.append((lam * float(payoff @ g.prior) + (1. - lam) * float(payoff.min())) / len(gs))
    return math.fsum(terms)


def matrix_game(values):
    """Value and row strategy of the zero-sum game max_x min_y x'My (row player maximizes); float64."""
    m = np.asarray(values, np.float64)
    rows, cols = m.shape
    lower, upper = m.min(axis=1).max(), m.max(axis=0).min()
    if lower == upper:   # pure saddle point: no LP needed
        x = np.zeros(rows);x[int(np.argmax(m.min(axis=1)))] = 1.
        return float(lower), x
    c = np.zeros(rows + 1);c[-1] = -1.
    a_ub = np.hstack([-m.T, np.ones((cols, 1))])
    a_eq = np.concatenate([np.ones(rows), [0.]])[None]
    z = _linprog(c, a_ub, np.zeros(cols), a_eq, [1.], [(0., 1.)] * rows + [(-1., 1.)])
    x = np.clip(z[:rows], 0., None);x = x / x.sum()
    return float((x @ m).min()), x


def _hedged_cell(frame, root, world, a1, r1):
    status = int(frame.t1_status[root, world, a1, r1])
    if status == 1:return float(frame.t1_value[root, world, a1, r1])
    if status != 2:return None
    t2 = frame.t2_status[root, world, a1, r1]
    own = np.flatnonzero(np.any(t2 != 0, axis=1));opp = np.flatnonzero(np.any(t2 != 0, axis=0))
    if np.any(t2 == 3) or not len(own) or not len(opp):return None
    if np.any(t2[np.ix_(own, opp)] == 0):raise ValueError('turn-2 enumeration is not a product of own and opponent sets')
    value, _ = matrix_game(np.asarray(frame.t2_value[root, world, a1, r1], np.float64)[np.ix_(own, opp)])
    return min(1., max(-1., value))


def depth2_hedged_worlds(frame, root):
    """Per world: each expanded turn-1 cell valued by its turn-2 zero-sum game; terminal cells exact.

    Mirrors depth2_backups.backup_learner_max's world surface (the enumerated turn-1 replies, the root
    opponent prior), so `bayes_hedge` and `averaging` consume it unchanged. A world with pre != 0 is None.
    """
    result = []
    for world in range(frame.pre.shape[1]):
        if int(frame.pre[root, world]) != 0:result.append(None);continue
        requested = bool(frame.root_peer_requested[root, world]);status = frame.t1_status[root, world]
        kept = tuple(bool(np.any(status[:, r] != 0)) for r in range(9)) if requested else (False,) * 9
        values = [[_hedged_cell(frame, root, world, a, r) if status[a, r] else None for r in range(9)] for a in range(9)]
        result.append(original.WorldPrices(requested, kept, tuple(map(float, frame.root_peer_logits[root, world])), values))
    return tuple(result)


DEPTH2_FIELDS = ('pre', 'root_peer_requested', 'root_peer_logits', 't1_status', 't1_value', 't2_status', 't2_value')


def depth2_hedged_root(fields):
    """depth2_hedged_worlds for one root given its per-root arrays (leading world axis): the pool's unit of work."""
    from types import SimpleNamespace
    return depth2_hedged_worlds(SimpleNamespace(**{k: np.asarray(fields[k])[None] for k in DEPTH2_FIELDS}), 0)


def search_uniform(seed, address):
    """A uniform in [0, 1) from (seed, stream, frame, owner) only: independent of the raw sampling keys."""
    payload = json.dumps([SAMPLE_DOMAIN, int(seed), int(address.stream), int(address.frame), int(address.owner)], separators=(',', ':'))
    return (int.from_bytes(hashlib.sha256(payload.encode()).digest()[:7], 'big') >> 3) / 2.**53


def sample(strategy, u):
    """Inverse CDF over action indices 0..8; the last positive-probability action absorbs rounding."""
    if not 0. <= u < 1.:raise ValueError('uniform in [0, 1) required')
    cumulative = 0.;last = None
    for a, p in enumerate(strategy):
        if p <= 0.:continue
        last = a;cumulative += p
        if u < cumulative:return a
    if last is None:raise ValueError('empty mixed strategy')
    return last
