"""Independent two-turn oracle for the depth-2 pricer's exactness gate.

Every turn-1 child and every turn-2 leaf is recomputed from its materialized root world with only the
plain Executor functions (legal, observe, dispatch, sample, advance), in ONE flat jax.vmap over specs,
chunked to a fixed size (a short last chunk repeats its first spec; padded rows are dropped). Nothing is
shared between specs: each leaf re-runs turn 1 from the root, so the oracle has no grid, no capsule
reuse and no batching structure for a bug to hide in.

One spec, step by step:
- gather the world's one-row state and public trackers by flat row;
- root: masks = executor.legal(state); observe the owner (masks[:, owner]) and 1-owner;
- turn 1: dispatch a1 for the owner; dispatch r1 for the opponent iff it is requested, where requested is
  any(opponent view legal_action_mask), exactly as D's joint; else -1. Advance with
  executor.sample(state), the ROOT's own chance;
- child: masks = executor.legal(child); observe both seats; a seat is requested iff any(masks[0, seat]).
  Each seat's legal actions are its ENCODER mask (view.observation.legal_action_mask): the request-order
  lanes that logits and executor.dispatch use. The engine mask orders switch lanes by physical bench
  slot and is used only for the request flag. Dispatch a2 / r2 for requested seats, -1 for a waiting seat. Advance with executor.sample(child), the
  CHILD's own chance (the engine's own key split decides turn-2 chance; a residual resume uses the draws
  the engine saved, inside advance);
- leaf: observe the owner at executor.legal(leaf).

Status and error codes follow D (device_public_teacher_batch.make_kernel):
error 1 invalid action route, 2 terminal winner outside {0,1,2}, 3 encoder error on the owner's view
(non-terminal only), 4 actor output refused (non-terminal only); terminal value +1/-1/0 from the owner's
side (winner 2 is a draw). Child status: 1 terminal, 2 expanded, 3 error. Leaf status: 1 terminal,
4 head-valued, 3 error. A non-terminal child without a D error then takes the contract's depth-2-only
codes, only while the opponent is requested at the child: 5 its child view has an encoder error, else 6
its child prior is unusable; such a child keeps D's learner value and is not expanded. A world with
no state has pre code -1 (world_pre). Heads (heads()) are a separate step on the oracle's own views:
v5_model.forward(..., compute_dtype='bfloat16'), learner params for the learner's child and leaf views,
opponent params for the opponent's root and child views.

Enumeration (turn1_cells, turn2_leaves) follows the contract (docs/v2e_depth2_pricer.md): turn 1 is D's
grid (recipe.pricing_order prefix x the opponent's legal replies, column 0 if it waits); turn 2 crosses
each expanded child's top-k legal actions per seat, ranked by the oracle's own child priors (logit
descending, lower action index on ties), row/column 0 for a waiting seat.
"""
from dataclasses import dataclass
import math
import time

import jax
import jax.numpy as j
import numpy as np

from . import depth2_digest, v5_model

STATUS_NONE, STATUS_TERMINAL, STATUS_EXPANDED, STATUS_ERROR, STATUS_HEAD = 0, 1, 2, 3, 4
ERROR_INVALID, ERROR_WINNER, ERROR_ENCODER, ERROR_ACTOR = 1, 2, 3, 4
ERROR_PEER_ENCODER, ERROR_PEER_PRIOR = 5, 6   # depth-2-only child codes (contract amendment)
NO_STATE = -1                                 # pre code of a world without a state
# D's pre-refusal input order (device_public_teacher_batch.SERIAL_INPUT_ORDER), restated literally.
INPUT_ORDER = ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask', 'history_mask')


@dataclass(frozen=True)
class Stack:
    """Available worlds of one request list, stacked along a flat row axis."""
    rows: dict                 # (request index, world) -> flat row; materialization refusals are absent
    states: object             # device pytree, leading axis = flat row (padded to a bucket), each row a one-row BattleState
    publics: object            # device pytree, leading axis = flat row, each row one world's tracker namespace
    owners: object             # device int32 [rows]
    owner: np.ndarray          # host int32 [rows]
    learner: np.ndarray        # host int32 [rows]: learner actor index into params
    opponent: np.ndarray       # host int32 [rows]: opponent actor index into params


def bucket(count):
    """Stacked-world rows: a power of two (at least 8), so calls with different world counts reuse compiled programs."""
    return max(8, 1 << (count - 1).bit_length())


def stack(requests, worlds):
    """Stack every materialized world (worlds[q] is the request's (world 0, world 1) pair).

    The row axis is padded to bucket(rows) by repeating row 0; no spec ever indexes a padding row, so it
    changes no result, only how often the oracle's programs compile."""
    requests, worlds = tuple(requests), tuple(worlds)
    if len(requests) != len(worlds):raise ValueError('one materialized world pair per request required')
    rows, states, publics, owner, learner, opponent = {}, [], [], [], [], []
    for q, (request, pair) in enumerate(zip(requests, worlds)):
        if len(pair) != 2:raise ValueError('two materialized worlds per request required')
        for wi, world in enumerate(pair):
            if world.state is None:continue
            rows[(q, wi)] = len(states)
            states.append(world.state);publics.append(world.public)
            owner.append(int(request.plans.root.address.owner))
            learner.append(int(request.learner_actor));opponent.append(int(request.opponent_actor))
    owner, learner, opponent = (np.asarray(x, np.int32) for x in (owner, learner, opponent))
    if not states:return Stack(rows, None, None, None, owner, learner, opponent)
    size = bucket(len(states));states = states + [states[0]] * (size - len(states));publics = publics + [publics[0]] * (size - len(publics))
    pack = lambda *leaves: np.stack([np.asarray(x) for x in leaves])
    states = jax.device_put(jax.tree.map(pack, *jax.device_get(states)))
    publics = jax.device_put(jax.tree.map(pack, *jax.device_get(publics)))
    padded = lambda x: np.concatenate([x, np.repeat(x[:1], size - len(x))])
    return Stack(rows, states, publics, j.asarray(padded(owner)), owner, learner, opponent)


def _gather(tree, row):
    return jax.tree.map(lambda x: x[row], tree)


def _views(executor, state, public, owner):
    masks = executor.legal(state)
    seat = j.reshape(owner, (1,)).astype(j.int32)
    own = executor.observe(state, public, seat, masks[:, owner])
    peer = executor.observe(state, public, 1 - seat, masks[:, 1 - owner])
    return masks, seat, own, peer


def _root(executor, states, publics, owners, spec):
    row = spec[0];state, public, owner = _gather(states, row), _gather(publics, row), owners[row]
    _, _, own, peer = _views(executor, state, public, owner)
    facts = dict(own_error=own.observation.error[0], own_legal=own.observation.legal_action_mask[0],
        peer_error=peer.observation.error[0], peer_legal=peer.observation.legal_action_mask[0],
        own_digest=depth2_digest.digest(own.inputs), peer_digest=depth2_digest.digest(peer.inputs))
    return facts, dict(own=own.inputs, peer=peer.inputs)


def _child_core(executor, states, publics, owners, row, a1, r1):
    state, public, owner = _gather(states, row), _gather(publics, row), owners[row]
    _, seat, own, peer = _views(executor, state, public, owner)
    # Turn 1 is D's joint step: the owner always acts; the opponent only if its view has a legal action.
    replying = j.any(peer.observation.legal_action_mask[0])
    actions = j.full((1, 2), -1, j.int32)
    actions = actions.at[:, owner].set(executor.dispatch(own.observation, state, seat, j.reshape(a1, (1,))))
    reply = executor.dispatch(peer.observation, state, 1 - seat, j.reshape(r1, (1,)))
    actions = actions.at[:, 1 - owner].set(j.where(replying, reply, -1))
    after = executor.advance(state, public, actions, executor.sample(state))
    child, trackers = after.step.state, after.trackers
    masks, _, own, peer = _views(executor, child, trackers, owner)
    own_requested, peer_requested = j.any(masks[0, owner]), j.any(masks[0, 1 - owner])
    facts = dict(invalid=after.step.invalid_actions[0], done=child.done[0], winner=child.winner[0],
        own_error=own.observation.error[0], peer_error=peer.observation.error[0],
        own_requested=own_requested, peer_requested=peer_requested,
        own_legal=own.observation.legal_action_mask[0], peer_legal=peer.observation.legal_action_mask[0],
        own_digest=depth2_digest.digest(own.inputs), peer_digest=depth2_digest.digest(peer.inputs))
    return facts, (child, trackers, own, peer, owner, seat, own_requested, peer_requested)


def _child(executor, states, publics, owners, spec):
    facts, (_, _, own, peer, *_) = _child_core(executor, states, publics, owners, spec[0], spec[1], spec[2])
    return facts, dict(own=own.inputs, peer=peer.inputs)


def _leaf(executor, states, publics, owners, spec):
    first, (child, trackers, own, peer, owner, seat, own_requested, peer_requested) = _child_core(
        executor, states, publics, owners, spec[0], spec[1], spec[2])
    actions = j.full((1, 2), -1, j.int32)
    mine = executor.dispatch(own.observation, child, seat, j.reshape(spec[3], (1,)))
    theirs = executor.dispatch(peer.observation, child, 1 - seat, j.reshape(spec[4], (1,)))
    actions = actions.at[:, owner].set(j.where(own_requested, mine, -1))
    actions = actions.at[:, 1 - owner].set(j.where(peer_requested, theirs, -1))
    # The child's own chance, never the root's capsule.
    after = executor.advance(child, trackers, actions, executor.sample(child))
    leaf, leaf_trackers = after.step.state, after.trackers
    masks = executor.legal(leaf)
    view = executor.observe(leaf, leaf_trackers, seat, masks[:, owner])
    facts = dict(child_invalid=first['invalid'], child_done=first['done'], child_own_digest=first['own_digest'],
        child_peer_digest=first['peer_digest'], invalid=after.step.invalid_actions[0], done=leaf.done[0],
        winner=leaf.winner[0], view_error=view.observation.error[0], digest=depth2_digest.digest(view.inputs))
    return facts, dict(leaf=view.inputs)


def heads(forward, params, inputs, actors):
    """v5 heads on one chunk of one-row views.

    inputs: encoder inputs with a leading spec axis over one-row views ([C, 1, ...]); actors: int [C]
    index into params. Each distinct actor runs once on the whole chunk and its rows are kept, so every
    row is evaluated by its own actor at one fixed batch geometry. Returns host (logits [C,9] f32,
    value [C] f32, usable [C] bool), usable as D: valid, finite and |value| <= 1.
    """
    flat = {name: x.reshape((x.shape[0],) + x.shape[2:]) for name, x in inputs.items()}
    actors = np.asarray(actors);count = len(actors)
    logits = np.zeros((count, 9), np.float32);values = np.zeros(count, np.float32);valid = np.zeros(count, bool)
    for actor in np.unique(actors):
        output = jax.device_get(forward(params[int(actor)], flat))
        rows = actors == actor
        logits[rows] = np.asarray(output.policy_logits, np.float32).reshape(count, 9)[rows]
        values[rows] = np.asarray(output.value, np.float32).reshape(count)[rows]
        valid[rows] = np.asarray(output.valid, bool).reshape(count)[rows]
    usable = valid & np.isfinite(logits).all(axis=-1) & np.isfinite(values) & (np.abs(values) <= 1)
    return logits, values, usable


def terminal_value(winner, owner):
    """D's terminal convention, from the owner's side: winner 2 draws, the owner's seat wins."""
    winner, owner = np.asarray(winner), np.asarray(owner)
    return np.where(winner == 2, 0., np.where(winner == owner, 1., -1.)).astype(np.float32)


def classify(invalid, done, winner, error, usable, owner, *, scored):
    """(status, error code, value) by D's error precedence; `scored` is the non-terminal success status."""
    invalid, done, winner, error, usable = (np.asarray(x) for x in (invalid, done, winner, error, usable))
    winner_ok = (winner >= 0) & (winner <= 2)
    code = np.where(invalid, ERROR_INVALID, np.where(done & ~winner_ok, ERROR_WINNER,
        np.where(~done & (error != 0), ERROR_ENCODER, np.where(~done & ~usable, ERROR_ACTOR, 0))))
    status = np.where(code != 0, STATUS_ERROR, np.where(done, STATUS_TERMINAL, scored))
    return status.astype(np.int8), code.astype(np.int8)


def world_pre(roots, count):
    """[count, 2] pre codes: each materialized world's D code, NO_STATE where a world has no state."""
    pre = np.full((count, 2), NO_STATE, np.int32)
    for (q, wi), root in roots.items():pre[q, wi] = root['pre']
    return pre


def child_codes(status, error, peer_error, peer_requested, peer_usable):
    """Depth-2-only codes on children D leaves expanded, while the opponent is requested at the child:
    5 (its view has an encoder error), else 6 (its prior is unusable). A waiting opponent's view is unused.
    D's codes 1-4 and terminal children are untouched (they take precedence)."""
    status, error = np.asarray(status), np.asarray(error)
    open_ = (status == STATUS_EXPANDED) & np.asarray(peer_requested, bool)
    error = np.where(open_ & (np.asarray(peer_error) != 0), ERROR_PEER_ENCODER,
        np.where(open_ & ~np.asarray(peer_usable, bool), ERROR_PEER_PRIOR, error))
    return np.where(error != 0, STATUS_ERROR, status).astype(np.int8), error.astype(np.int8)


def pre_code(request, root):
    """D's per-world pre-refusal code (make_kernel's pre), from the oracle's own root views."""
    expected = request.root_inputs;inputs = root['own_inputs']
    if root['own_error'] != 0:return 1
    if tuple(map(bool, root['own_legal'])) != tuple(map(bool, request.root_legal)):return 2
    if set(inputs) != set(expected):return 10
    for index, key in enumerate(k for k in INPUT_ORDER if k in expected):
        if key in inputs and not np.array_equal(np.asarray(inputs[key]), np.asarray(expected[key])):return 3 + index
    if root['peer_error'] != 0:return 8
    if root['peer_requested'] and not root['peer_usable']:return 9
    return 0


class Oracle:
    """Plain-executor two-turn evaluation; make one per executor/model."""
    def __init__(self, executor, model, *, chunk=512, progress=None, raw_model=None):
        """progress(kind=, chunk=, chunks=, rows_done=, rows=, seconds=, step_seconds=, head_seconds=) after each chunk.
        raw_model: also evaluate the learner's leaf and child views with v5_model.forward(..., raw_model, bf16), the
        head's own arm (U1's predict path), as raw_head_value."""
        if type(chunk) is not int or chunk <= 0:raise ValueError('positive fixed oracle chunk required')
        self.executor, self.model, self.chunk, self.progress = executor, model, chunk, progress
        self.chunks = []   # per-chunk timing: the first chunk of each program and world bucket includes its compile
        mapped = lambda fn: jax.jit(jax.vmap(lambda s, p, o, spec: fn(executor, s, p, o, spec), in_axes=(None, None, None, 0)))
        self._root, self._child, self._leaf = mapped(_root), mapped(_child), mapped(_leaf)
        self.forward = jax.jit(lambda params, inputs: v5_model.forward(params, inputs, model, compute_dtype='bfloat16'))
        self.raw_forward = None if raw_model is None else jax.jit(lambda params, inputs: v5_model.forward(params, inputs, raw_model, compute_dtype='bfloat16'))
        self.seconds = dict(roots=0., children=0., leaves=0.);self.rows = dict(roots=0, children=0, leaves=0, padded=0)

    def _map(self, kind, fn, stack, specs, params, evaluate, keep=()):
        """fn over specs [N, k] (column 0 = flat world row) in fixed chunks; evaluate: ((input, actor kind), ...)."""
        clock = time.perf_counter()
        specs = np.asarray(specs, np.int32).reshape(len(specs), -1);facts, outputs, kept = [], {}, {}
        total = -(-len(specs) // self.chunk)
        for index, start in enumerate(range(0, len(specs), self.chunk)):
            started = time.perf_counter()
            part = specs[start:start + self.chunk];count = len(part)
            padded = np.concatenate([part, np.repeat(part[:1], self.chunk - count, axis=0)])
            out, inputs = jax.block_until_ready(fn(stack.states, stack.publics, stack.owners, j.asarray(padded)))
            stepped = time.perf_counter()
            actors = dict(learner=stack.learner[padded[:, 0]], opponent=stack.opponent[padded[:, 0]])
            for name, actor, *raw in evaluate:
                result = heads(self.raw_forward if raw else self.forward, params, inputs[name], actors[actor])
                outputs.setdefault(name + (':raw' if raw else ''), []).append(tuple(x[:count] for x in result))
            headed = time.perf_counter()
            for name in keep:
                kept.setdefault(name, []).append(jax.tree.map(lambda x: np.asarray(x)[:count], jax.device_get(inputs[name])))
            facts.append(jax.tree.map(lambda x: np.asarray(x)[:count], jax.device_get(out)))
            self.rows['padded'] += self.chunk - count
            record = dict(kind=kind, chunk=index, chunks=total, rows_done=start + count, rows=len(specs), world_rows=len(stack.owner),
                seconds=time.perf_counter() - started, step_seconds=stepped - started, head_seconds=headed - stepped)
            self.chunks.append(record)
            if self.progress is not None:self.progress(**record)
        joined = {name: np.concatenate([f[name] for f in facts]) for name in facts[0]} if facts else {}
        heads_ = {name: tuple(np.concatenate(parts) for parts in zip(*chunks)) for name, chunks in outputs.items()}
        kept = {name: jax.tree.map(lambda *xs: np.concatenate(xs), *chunks) for name, chunks in kept.items()}
        self.seconds[kind] += time.perf_counter() - clock;self.rows[kind] += len(specs)
        return joined, heads_, kept

    def roots(self, stack, requests, params):
        """{(q, world): root facts} for every materialized world, with D's pre code and the opponent prior."""
        keys = sorted(stack.rows, key=stack.rows.get)
        if not keys:return {}
        facts, outputs, kept = self._map('roots', self._root, stack, [[stack.rows[k]] for k in keys], params,
            (('peer', 'opponent'),), keep=('own',))
        logits, _, usable = outputs['peer'];result = {}
        for i, (q, wi) in enumerate(keys):
            requested = bool(np.any(facts['peer_legal'][i]))
            root = dict(own_error=int(facts['own_error'][i]), own_legal=tuple(map(bool, facts['own_legal'][i])),
                peer_error=int(facts['peer_error'][i]), peer_legal=tuple(map(bool, facts['peer_legal'][i])),
                peer_requested=requested, peer_usable=bool(usable[i]),
                peer_logits=np.where(requested, logits[i], 0.).astype(np.float32), peer_logits_raw=logits[i],
                own_digest=facts['own_digest'][i], peer_digest=facts['peer_digest'][i],
                own_inputs={name: x[i] for name, x in kept['own'].items()})
            root['pre'] = pre_code(requests[q], root);result[(q, wi)] = root
        return result

    def children(self, stack, cells, params):
        """Turn-1 children of cells [(q, world, a1, r1)]: D's status/error/value plus both seats' views and priors."""
        cells = np.asarray(cells, np.int64).reshape(-1, 4)
        if not len(cells):return dict(cells=cells)
        rows = np.asarray([stack.rows[(int(q), int(w))] for q, w in cells[:, :2]], np.int32)
        specs = np.concatenate([rows[:, None], cells[:, 2:]], axis=1)
        raw = (('own', 'learner', 'raw'),) if self.raw_forward is not None else ()
        facts, outputs, _ = self._map('children', self._child, stack, specs, params, (('own', 'learner'), ('peer', 'opponent')) + raw)
        own_logits, own_value, own_usable = outputs['own'];peer_logits, _, peer_usable = outputs['peer']
        owner = stack.owner[rows]
        status, error = classify(facts['invalid'], facts['done'], facts['winner'], facts['own_error'], own_usable, owner,
            scored=STATUS_EXPANDED)
        status, error = child_codes(status, error, facts['peer_error'], facts['peer_requested'], peer_usable)
        learner_valued = (status == STATUS_EXPANDED) | np.isin(error, (ERROR_PEER_ENCODER, ERROR_PEER_PRIOR))
        value = np.where(status == STATUS_TERMINAL, terminal_value(facts['winner'], owner),
            np.where(learner_valued, own_value, np.nan)).astype(np.float32)
        extra = {} if not raw else dict(raw_head_value=outputs['own:raw'][1])
        return dict(cells=cells, status=status, error=error, value=value, head_value=own_value, own_usable=own_usable,
            peer_usable=peer_usable, own_logits=own_logits, peer_logits=peer_logits, **extra, **facts)

    def leaves(self, stack, specs, params):
        """Turn-2 leaves of specs [(q, world, a1, r1, a2, r2)]: status/error/value, digest and the learner head."""
        specs = np.asarray(specs, np.int64).reshape(-1, 6)
        if not len(specs):return dict(specs=specs)
        rows = np.asarray([stack.rows[(int(q), int(w))] for q, w in specs[:, :2]], np.int32)
        flat = np.concatenate([rows[:, None], specs[:, 2:]], axis=1)
        raw = (('leaf', 'learner', 'raw'),) if self.raw_forward is not None else ()
        facts, outputs, _ = self._map('leaves', self._leaf, stack, flat, params, (('leaf', 'learner'),) + raw)
        logits, head_value, usable = outputs['leaf'];owner = stack.owner[rows]
        status, error = classify(facts['invalid'], facts['done'], facts['winner'], facts['view_error'], usable, owner,
            scored=STATUS_HEAD)
        value = np.where(status == STATUS_TERMINAL, terminal_value(facts['winner'], owner),
            np.where(status == STATUS_HEAD, head_value, np.nan)).astype(np.float32)
        extra = {} if not raw else dict(raw_head_value=outputs['leaf:raw'][1])
        return dict(specs=specs, status=status, error=error, value=value, head_value=head_value, usable=usable,
            logits=logits, **extra, **facts)


def top_k(legal, logits, k):
    """The k highest-prior legal actions, ascending index order.

    Rank: logit descending, lower action index on ties; a non-finite logit ranks below every finite one.
    """
    if type(k) is not int or not 1 <= k <= 9:raise ValueError('width k in 1..9 required')
    score = lambda i: float(logits[i]) if math.isfinite(float(logits[i])) else -math.inf
    ranked = sorted((i for i in range(9) if legal[i]), key=lambda i: (-score(i), i))
    return tuple(sorted(ranked[:k]))


def turn1_cells(requests, recipe, roots, *, k1_own=9, k1_opp=9):
    """D's turn-1 grid per usable world: pricing_order prefix x the opponent's legal replies (column 0 if it waits)."""
    if type(k1_own) is not int or not 1 <= k1_own <= 9:raise ValueError('turn-1 width k1_own in 1..9 required')
    cells = []
    for q, request in enumerate(requests):
        order = recipe.pricing_order(request.plans.root.address, tuple(request.root_logits), tuple(request.root_legal))[:k1_own]
        for wi in (0, 1):
            root = roots.get((q, wi))
            if root is None or root['pre'] != 0:continue
            replies = top_k(root['peer_legal'], root['peer_logits'], k1_opp) if root['peer_requested'] else (0,)
            cells.extend((q, wi, a, r) for a in order for r in replies)
    return cells


def turn2_leaves(children, *, k_own=9, k_opp=9):
    """Each expanded child's top-k_own x top-k_opp legal pairs by the oracle's child priors (0 for a waiting seat)."""
    leaves = []
    for i, (q, wi, a1, r1) in enumerate(map(tuple, children['cells'].tolist())):
        if children['status'][i] != STATUS_EXPANDED:continue
        mine = top_k(children['own_legal'][i], children['own_logits'][i], k_own) if children['own_requested'][i] else (0,)
        theirs = top_k(children['peer_legal'][i], children['peer_logits'][i], k_opp) if children['peer_requested'][i] else (0,)
        leaves.extend((q, wi, a1, r1, a2, r2) for a2 in mine for r2 in theirs)
    return leaves


def enumerate_specs(oracle, stack, requests, recipe, params, *, k_own=9, k_opp=9, k1_own=9, k1_opp=9):
    """(roots, turn-1 cells, children, turn-2 leaf specs) by the contract's width rules, from the oracle's own priors."""
    roots = oracle.roots(stack, requests, params)
    cells = turn1_cells(requests, recipe, roots, k1_own=k1_own, k1_opp=k1_opp)
    children = oracle.children(stack, cells, params)
    leaves = turn2_leaves(children, k_own=k_own, k_opp=k_opp) if len(cells) else []
    return roots, cells, children, leaves
