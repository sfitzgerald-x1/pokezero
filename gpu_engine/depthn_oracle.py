"""Independent N-step flat oracle for the depth-N pricer's exactness gate.

Extends the depth-2 oracle (gpu_engine.depth2_oracle) without sharing anything with the pricer. Every node and leaf is
recomputed from its materialized root world along its own path, in ONE flat jax.vmap per ply (fixed chunks, padded
rows dropped), with only the plain Executor functions:
- turn 1 is D's joint step (the owner always acts; the opponent iff any(its view legal_action_mask), else -1), with
  executor.sample(root);
- every later step starts from the previous node's own state and trackers: masks = executor.legal(node), a seat is
  requested iff any(masks[0, seat]); each seat's actions are its ENCODER lanes (view.observation.legal_action_mask),
  dispatched through its own view; a waiting seat is dispatched -1; chance is executor.sample(node), the node's own;
- an intermediate node observes both seats (digests, encoder masks, request flags, priors and the learner value);
- a leaf observes the learner (digest and value).
Statuses and codes follow depth 2 at every ply: nodes 1 terminal / 2 expanded / 3 error with D's codes 1-4 then 5/6
(the opponent's view or prior, only while requested); leaves 1 terminal / 4 head-valued / 3 error.

Enumeration follows the contract's selection: node ids are parent * K_p + own_slot * k_opp,p + opp_slot (turn-1 cells are
a1 * 9 + r1), and a seat's slot -> action map is rank-ordered (logit descending, lower action on ties, non-finite last;
-1 padded; a waiting seat [0, -1, ...]). `prior_map` is this module's own ranking code; the gate feeds it the PRICER's
priors (which check (c) compares with the oracle's) so a bf16 near-tie between geometries cannot move the enumeration,
and takes `searched` maps as given.
"""
import math

import jax
import jax.numpy as j
import numpy as np

from . import depth2_digest
from . import depth2_oracle as d2
from .depth2_oracle import (STATUS_ERROR, STATUS_EXPANDED, STATUS_HEAD, STATUS_TERMINAL, ERROR_PEER_ENCODER,   # noqa: F401
    ERROR_PEER_PRIOR, stack, classify, child_codes, terminal_value, turn1_cells, NO_STATE, world_pre)


def _step(executor, state, trackers, owner, seat, own, peer, a, r, own_requested, peer_requested):
    actions = j.full((1, 2), -1, j.int32)
    mine = executor.dispatch(own.observation, state, seat, j.reshape(a, (1,)))
    theirs = executor.dispatch(peer.observation, state, 1 - seat, j.reshape(r, (1,)))
    actions = actions.at[:, owner].set(j.where(own_requested, mine, -1))
    actions = actions.at[:, 1 - owner].set(j.where(peer_requested, theirs, -1))
    return executor.advance(state, trackers, actions, executor.sample(state))   # this node's own draw


def _walk(executor, states, publics, owners, spec, plies, leaf):
    """One path of `plies` steps from its root; spec = [row, a1, r1, ..., a_plies, r_plies] (static plies)."""
    row = spec[0];state, public, owner = d2._gather(states, row), d2._gather(publics, row), owners[row]
    _, seat, own, peer = d2._views(executor, state, public, owner)
    own_requested, peer_requested = j.asarray(True), j.any(peer.observation.legal_action_mask[0])   # turn 1: D's joint rule
    broken = j.asarray(False)   # an earlier step on this path was invalid or terminal (never expected: the gate expands only status 2)
    for p in range(plies):
        after = _step(executor, state, public, owner, seat, own, peer, spec[1 + 2 * p], spec[2 + 2 * p], own_requested, peer_requested)
        if p < plies - 1:broken = broken | after.step.invalid_actions[0] | after.step.state.done[0]
        state, public = after.step.state, after.trackers
        if p < plies - 1 or not leaf:
            masks, _, own, peer = d2._views(executor, state, public, owner)
            own_requested, peer_requested = j.any(masks[0, owner]), j.any(masks[0, 1 - owner])
    facts = dict(invalid=after.step.invalid_actions[0], done=state.done[0], winner=state.winner[0], broken_path=broken)
    if leaf:
        masks = executor.legal(state);view = executor.observe(state, public, seat, masks[:, owner])
        facts.update(view_error=view.observation.error[0], digest=depth2_digest.digest(view.inputs))
        return facts, dict(leaf=view.inputs)
    facts.update(own_error=own.observation.error[0], peer_error=peer.observation.error[0], own_requested=own_requested,
        peer_requested=peer_requested, own_legal=own.observation.legal_action_mask[0], peer_legal=peer.observation.legal_action_mask[0],
        own_digest=depth2_digest.digest(own.inputs), peer_digest=depth2_digest.digest(peer.inputs))
    return facts, dict(own=own.inputs, peer=peer.inputs)


class DepthNOracle(d2.Oracle):
    """The depth-2 oracle (roots, turn-1 children) plus N-step nodes and leaves at any ply."""
    def __init__(self, executor, model, *, chunk=512, progress=None, raw_model=None):
        super().__init__(executor, model, chunk=chunk, progress=progress, raw_model=raw_model);self._walks = {}

    def _fn(self, plies, leaf):
        key = (plies, leaf)
        if key not in self._walks:
            ex = self.executor
            self._walks[key] = jax.jit(jax.vmap(lambda s, p, o, spec: _walk(ex, s, p, o, spec, plies, leaf), in_axes=(None, None, None, 0)))
        kind = f'ply{plies}' + ('_leaves' if leaf else '');self.seconds.setdefault(kind, 0.);self.rows.setdefault(kind, 0)
        return self._walks[key], kind

    @staticmethod
    def _flat(stack, specs, width):
        specs = np.asarray(specs, np.int64).reshape(-1, width)
        rows = np.asarray([stack.rows[(int(q), int(w))] for q, w in specs[:, :2]], np.int32)
        return specs, rows, np.concatenate([rows[:, None], specs[:, 2:]], axis=1)

    def nodes(self, stack, specs, plies, params):
        """Intermediate nodes at ply `plies` (>= 1) for specs [(q, world, a1, r1, ..., a_p, r_p)]: status/error/value as depth 2's
        children (codes 1-6), both seats' encoder masks, request flags, digests and priors (learner/opponent params)."""
        specs, rows, flat = self._flat(stack, specs, 2 + 2 * plies)
        if not len(specs):return dict(specs=specs)
        fn, kind = self._fn(plies, False);raw = (('own', 'learner', 'raw'),) if self.raw_forward is not None else ()
        facts, outputs, _ = self._map(kind, fn, stack, flat, params, (('own', 'learner'), ('peer', 'opponent')) + raw)
        own_logits, own_value, own_usable = outputs['own'];peer_logits, _, peer_usable = outputs['peer'];owner = stack.owner[rows]
        status, error = classify(facts['invalid'], facts['done'], facts['winner'], facts['own_error'], own_usable, owner, scored=STATUS_EXPANDED)
        status, error = child_codes(status, error, facts['peer_error'], facts['peer_requested'], peer_usable)
        learner_valued = (status == STATUS_EXPANDED) | np.isin(error, (ERROR_PEER_ENCODER, ERROR_PEER_PRIOR))
        value = np.where(status == STATUS_TERMINAL, terminal_value(facts['winner'], owner), np.where(learner_valued, own_value, np.nan)).astype(np.float32)
        extra = {} if not raw else dict(raw_head_value=outputs['own:raw'][1])
        return dict(specs=specs, status=status, error=error, value=value, head_value=own_value, own_usable=own_usable, peer_usable=peer_usable,
            own_logits=own_logits, peer_logits=peer_logits, **extra, **facts)

    def leaves_n(self, stack, specs, plies, params):
        """Leaves after `plies` steps: status/error/value (1 terminal, 4 head-valued, 3 error), the learner view's digest and head."""
        specs, rows, flat = self._flat(stack, specs, 2 + 2 * plies)
        if not len(specs):return dict(specs=specs)
        fn, kind = self._fn(plies, True);raw = (('leaf', 'learner', 'raw'),) if self.raw_forward is not None else ()
        facts, outputs, _ = self._map(kind, fn, stack, flat, params, (('leaf', 'learner'),) + raw)
        logits, head_value, usable = outputs['leaf'];owner = stack.owner[rows]
        status, error = classify(facts['invalid'], facts['done'], facts['winner'], facts['view_error'], usable, owner, scored=STATUS_HEAD)
        value = np.where(status == STATUS_TERMINAL, terminal_value(facts['winner'], owner), np.where(status == STATUS_HEAD, head_value, np.nan)).astype(np.float32)
        extra = {} if not raw else dict(raw_head_value=outputs['leaf:raw'][1])
        return dict(specs=specs, status=status, error=error, value=value, head_value=head_value, usable=usable, logits=logits, **extra, **facts)


def ranked(legal, logits):
    """Every legal action in rank order: logit descending, lower action on ties, a non-finite logit after every finite one."""
    score = lambda i: float(logits[i]) if math.isfinite(float(logits[i])) else -math.inf
    return sorted((i for i in range(9) if legal[i]), key=lambda i: (-score(i), i))


def prior_map(legal, logits, requested, k):
    """A seat's slot -> action map [k] under `prior`: its top-k legal actions in rank order, -1 padded; [0, -1, ...] when waiting."""
    m = np.full(k, -1, np.int32)
    if not requested:m[0] = 0;return m
    order = ranked(legal, logits)[:k];m[:len(order)] = order
    return m


def expand(parents, own_maps, opp_maps):
    """Children of expanded parents: parents [(spec, node id)], maps aligned [k_own] / [k_opp] -> [(spec, id, parent id, i, j)]
    with id = parent * K + i * k_opp + j (K = k_own * k_opp); a -1 slot is never enumerated."""
    out = []
    for (spec, node), own, opp in zip(parents, own_maps, opp_maps):
        k_opp = len(opp);K = len(own) * k_opp
        for i, a in enumerate(own):
            if a < 0:continue
            for jj, r in enumerate(opp):
                if r >= 0:out.append((tuple(spec) + (int(a), int(r)), node * K + i * k_opp + jj, node, i, jj))
    return out


def expand_arrays(specs, ids, own_maps, opp_maps):
    """expand over arrays: parents specs [P, w], ids [P], maps [P, k_own] / [P, k_opp] -> (child specs [C, w + 2], child ids,
    parent ids, own slots, opp slots), parent-major then own slot then opponent slot (expand's order)."""
    two = lambda x: np.asarray(x, np.int64) if np.ndim(x) == 2 else np.asarray(x, np.int64).reshape(len(ids), -1)
    specs, ids, own_maps, opp_maps = two(specs), np.asarray(ids, np.int64), two(own_maps), two(opp_maps)
    k_opp = opp_maps.shape[1];K = own_maps.shape[1] * k_opp
    parent, i, x = np.nonzero((own_maps[:, :, None] >= 0) & (opp_maps[:, None, :] >= 0))
    children = np.concatenate([specs[parent], own_maps[parent, i][:, None], opp_maps[parent, x][:, None]], axis=1)
    return children, ids[parent] * K + i * k_opp + x, ids[parent], i, x


def cell_id(a1, r1):
    return int(a1) * 9 + int(r1)
