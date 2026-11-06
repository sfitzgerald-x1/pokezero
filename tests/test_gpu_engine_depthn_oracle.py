"""Depth-N oracle and gate on traceable inert worlds/actors (CPU only).

The fake engine is the depth-2 gate tests' engine with every behaviour movable to a later ply: mode = shift * 100 +
behaviour, a node behaviour fires at turn 1 + shift and a leaf behaviour at turn 2 + shift, so a depth-3 search meets
terminal, invalid, encoder-error, waiting-seat and opponent-view cases at ply 2 and at the leaves. Values are integers
over a power of two, so every float is exact. The pricer stand-in writes the oracle's own surface in the DepthNFrame
layout; mutated stand-ins break selection, chance and masks at ply 3. Tests marked `real pricer` run the depth-N pricer
(gpu_engine.device_depthn_pricer, gpu_engine.depthn_backups) when they are on the path and are skipped otherwise.
"""
import collections
import contextlib
import copy
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

import jax.numpy as j
import numpy as np

from pokezero import v5_contract
from gpu_engine import d3_depthn_gate as gate, depthn_oracle as reference, depth2_backups
from gpu_engine import device_public_teacher as serial, device_public_teacher_batch as batch
from gpu_engine import device_public_worlds as plans, device_public_world_state as mapping
from tests.test_gpu_engine_device_public_teacher import facts
from tests.test_gpu_engine_depth2_oracle import (After, BASE, LCG, Leaf, Observation, PARAMS, Public, State, Step, SWITCH, forward)

SCALE = 4096
ENGINE_OF_LANE = (0, 1, 2, 3, 7, 8, 4, 5, 6)
HAVE_PRICER = all(importlib.util.find_spec(m) is not None for m in ('gpu_engine.device_depthn_pricer', 'gpu_engine.depthn_backups'))


def executor_n(*, sample=None):
    """Behaviours (b = mode % 100, shift = mode // 100; `node` = turn 1 + shift, `leaf` = turn 2 + shift):
    1/2/7 node terminal win/draw/bad winner; 3 invalid node step (1,1); 4 node encoder error; 5 non-finite learner value
    from the node on; 6 the opponent always waits; 8 node: opponent waits, owner forced to switch (4,5); 12 node: owner
    waits, opponent forced; 9 leaf-step terminal (a 0 loss, a 1 draw); 10 invalid leaf step a=2; 11 leaf encoder error;
    node, opponent's view only: 13 encoder error, 14 non-finite prior (requested), 15 non-finite prior while it waits;
    18 re-keys chance from the dispatched actions at every step. sample: replaces executor.sample (mutations)."""
    def split(mode):return mode % 100, mode // 100

    def legal(state):
        owner, (b, shift), turn = state.owner[0], split(state.mode[0]), state.turn[0]
        node = turn == 1 + shift
        own = j.where((b == 12) & node, False, j.where((b == 8) & node, j.asarray(SWITCH), j.asarray(BASE)))
        foe = j.where((b == 6) | (((b == 8) | (b == 15)) & node), False, j.where((b == 12) & node, j.asarray(SWITCH), j.asarray(BASE)))
        rows = j.where(owner == 0, j.stack([own, foe]), j.stack([foe, own]))
        return j.where(state.done[0], False, rows)[None]

    def observe(state, public, owner, mask):
        (b, shift), peer = split(state.mode), owner != state.owner
        node, leaf = public.turn == 1 + shift, public.turn == 2 + shift
        error = j.where(((b == 4) & node) | ((b == 11) & leaf) | ((b == 13) & node & peer), 1, 0)
        value = (state.value + (owner - state.owner) * 10).astype(j.float32) * (1. / SCALE)
        value = j.where(((b == 5) & (public.turn >= 1 + shift)) | (((b == 14) | (b == 15)) & node & peer), j.nan, value)
        feature = j.stack((value, public.clock + j.zeros(1), state.own_act.astype(j.float32), state.peer_act.astype(j.float32),
            public.turn.astype(j.float32)), axis=-1)
        return Leaf(Observation(mask, error), {'numeric_features': feature[:, None, None, :]}, owner)

    def advance(state, public, actions, chance):
        owner = state.owner[0];a = actions[:, owner];r = actions[:, 1 - owner];(b, shift), turn = split(state.mode), state.turn + 1
        node, leaf = turn == 1 + shift, turn == 2 + shift
        value = state.value + a * 30 + r * 20 + chance
        node_end = node & ((b == 1) | (b == 2) | (b == 7));leaf_end = leaf & (b == 9) & (a >= 0) & (a <= 1)
        winner = j.where(node_end, j.where(b == 2, 2, j.where(b == 7, 9, state.owner)), j.where(leaf_end, j.where(a == 0, 1 - state.owner, 2), state.winner))
        invalid = (node & (b == 3) & (a == 1) & (r == 1)) | (leaf & (b == 10) & (a == 2))
        seed = state.seed * j.uint32(LCG[0]) + j.uint32(LCG[1]) + j.where(b == 18, (a + 1) * 7 + (r + 1) * 13, 0).astype(j.uint32)
        after = state._replace(value=value, done=state.done | node_end | leaf_end, winner=winner, seed=seed, turn=turn, own_act=a, peer_act=r)
        return After(Step(after, invalid), Public(public.clock, public.turn + 1))

    draw = sample or (lambda state: (state.seed % 17).astype(j.int32))
    ex = serial.Executor(legal, observe, lambda obs, state, seat, action: action, draw, advance)
    return replace(ex, joint=serial._make_joint(ex))


def permuted_n(*, engine_mask_at_turn=None):
    """executor_n with the encoder's switch lanes a permutation of the engine's (two live bench slots per requested seat).
    engine_mask_at_turn=T: a mutation whose views at turn T carry the ENGINE mask as the encoder mask (wrong lanes)."""
    base = executor_n();order = j.asarray(ENGINE_OF_LANE);bench = j.asarray((False,) * 4 + (True, True) + (False,) * 3)
    def legal(state):
        mask = base.legal(state);return mask | (j.any(mask, axis=-1, keepdims=True) & bench)
    def observe(state, public, owner, mask):
        lanes = mask[:, order]
        if engine_mask_at_turn is not None:lanes = j.where(public.turn == engine_mask_at_turn, mask, lanes)
        return base.observe(state, public, owner, lanes)
    def advance(state, public, actions, chance):
        mask = legal(state)[0];acts = actions[0]
        wrong = j.any(j.any(mask, axis=-1) & ((acts < 0) | ~mask[j.arange(2), j.clip(acts, 0, 8)]))
        after = base.advance(state, public, actions, chance)
        return after._replace(step=after.step._replace(invalid_actions=after.step.invalid_actions | wrong))
    ex = serial.Executor(legal, observe, lambda observation, state, seat, action: order[action], base.sample, advance)
    return replace(ex, joint=serial._make_joint(ex))


def world(mode, owner, seed, clock=3.):
    state = State(j.asarray([200], j.int32), j.asarray([False]), j.asarray([0], j.int32), j.asarray([mode], j.int32),
        j.asarray([seed % 2**32], j.uint32), j.asarray([owner], j.int32), j.asarray([0], j.int32), j.zeros(1, j.int32), j.zeros(1, j.int32))
    return state, Public(j.asarray(clock, j.float32), j.asarray([0], j.int32))


# ---------------------------------------------------------------- the pricer stand-in: the oracle's surface in DepthNFrame layout

def node_layout(M, k_own, k_opp):
    """The dense layout's parent and parent slots of every ply-p node id."""
    ids = np.arange(M);K = k_own * k_opp
    return ids // K, (ids % K) // k_opp, ids % k_opp


def frame_from_n(count, data, plan):
    from tests.test_gpu_engine_depth2_oracle import frame_from
    t1 = frame_from(count, dict(roots=data['roots'], children=data['children'], leaves=dict(specs=np.zeros((0, 6), np.int64))))
    f = {k: v for k, v in t1.items() if not k.startswith('t2_')};R = count;N = plan['depth']
    f['t1_own_actions'] = np.full((R, 2, 9, 9, plan['k_own'][0]), -1, np.int32);f['t1_opp_actions'] = np.full((R, 2, 9, 9, plan['k_opp'][0]), -1, np.int32)
    for (p, q, w, node), (own, opp) in data['maps'].items():
        if p != 2:continue
        a1, r1 = divmod(node, 9);f['t1_own_actions'][q, w, a1, r1] = own;f['t1_opp_actions'][q, w, a1, r1] = opp
    nodes = []
    for p in range(2, N):
        M = plan['counts'][p - 1];k_own, k_opp = plan['k_own'][p - 2], plan['k_opp'][p - 2];records = data['levels'][p]
        parent, own_slot, opp_slot = node_layout(M, k_own, k_opp)
        n = dict(ply=p, parent=parent, parent_own_slot=own_slot, parent_opp_slot=opp_slot, status=np.zeros((R, 2, M), np.int8),
            value=np.full((R, 2, M), np.nan, np.float32), error=np.zeros((R, 2, M), np.int8), own_requested=np.zeros((R, 2, M), bool),
            opp_requested=np.zeros((R, 2, M), bool), own_legal=np.zeros((R, 2, M, 9), bool), opp_legal=np.zeros((R, 2, M, 9), bool),
            own_logits=np.zeros((R, 2, M, 9), np.float32), opp_logits=np.zeros((R, 2, M, 9), np.float32),
            own_actions=np.full((R, 2, M, plan['k_own'][p - 1]), -1, np.int32), opp_actions=np.full((R, 2, M, plan['k_opp'][p - 1]), -1, np.int32),
            digest_own=np.zeros((R, 2, M, 2), np.uint32), digest_opp=np.zeros((R, 2, M, 2), np.uint32))
        for i, node in enumerate(data['ids'][p]):
            q, w = map(int, records['specs'][i][:2]);at = (q, w, node)
            for name in ('status', 'error', 'value'):n[name][at] = records[name][i]
            for seat, rec in (('own', 'own'), ('opp', 'peer')):
                n[f'{seat}_requested'][at] = records[rec + '_requested'][i];n[f'{seat}_legal'][at] = records[rec + '_legal'][i]
                n[f'{seat}_logits'][at] = records[rec + '_logits'][i] if records[rec + '_requested'][i] else 0.
                n[f'digest_{seat}'][at] = records[rec + '_digest'][i]
            if records['status'][i] == 2:n['own_actions'][at], n['opp_actions'][at] = data['maps'][(p + 1, q, w, node)]
        nodes.append(SimpleNamespace(**n))
    M = plan['counts'][-1];shape = (R, 2, M, plan['k_own'][-1], plan['k_opp'][-1])
    f.update(leaf_status=np.zeros(shape, np.int8), leaf_value=np.full(shape, np.nan, np.float32), leaf_error=np.zeros(shape, np.int8),
        leaf_digest=np.zeros(shape + (2,), np.uint32))
    leaves = data['leaves']
    for i, index in enumerate(leaves['index'].tolist()):
        at = tuple(index)
        for name in ('status', 'error', 'value', 'digest'):f['leaf_' + name][at] = leaves[name][i]
    f.update(nodes=tuple(nodes), depth=N, k_own=plan['k_own'], k_opp=plan['k_opp'], world_reports=(),
        timing=dict(fake=True, k1_own=plan['k1_own'], device_seconds=0., plies=()))
    return SimpleNamespace(**f)


def fake_searched_widths(k_own, k_opp):
    return k_own + 2, k_opp + 2


class OracleNPricer:
    """DepthNPricer's constructor and price() over the oracle's own surface; `executor`/`reference` overrides make mutants.
    As the pricer, under `searched` the ply-2 k's are searched_maps' and the slot widths are derived (fake_searched_widths)."""
    def __init__(self, executor, model, *, capacity=32, depth, k_own, k_opp, k1_own=9, k1_opp=9, block_lanes=2187, selection='prior',
                 digest=True, actor_capacity=3, fixed_capacity=False, reference_module=None):
        self.oracle = reference.DepthNOracle(executor, model, chunk=64);self.capacity, self.block_lanes, self.selection = capacity, block_lanes, selection
        shape = gate.plan_of(depth, k_own, k_opp);self.k_own, self.k_opp, self.searched_k = shape['k_own'], shape['k_opp'], None
        if selection == 'searched':
            self.searched_k = (self.k_own[0], self.k_opp[0]);w_own, w_opp = fake_searched_widths(*self.searched_k)
            self.k_own, self.k_opp = (w_own,) + self.k_own[1:], (w_opp,) + self.k_opp[1:]
        self.plan = gate.plan_of(depth, self.k_own, self.k_opp, selection, k1_own=k1_own, k1_opp=k1_opp);self.reference = reference_module or reference

    def price(self, requests, materializer, recipe, params, *, selection_maps=None):
        if (self.selection == 'searched') != (selection_maps is not None):raise ValueError('searched selection needs selection_maps')
        stack = reference.stack(requests, batch._numpy_worlds(requests, materializer, recipe))
        data = gate.oracle_levels(oracle=self.oracle, stack=stack, requests=requests, recipe=recipe, params=params, frame=None, plan=self.plan,
            selection_maps=selection_maps, reference=self.reference)
        frame = frame_from_n(len(requests), data, self.plan)
        frame.timing.update(searched_k=self.searched_k, searched_widths=None if self.searched_k is None else (self.k_own[0], self.k_opp[0]))
        return frame


def fake_as_depth2(frame):
    """as_depth2_frame for the stand-in (N=2): leaves land at [a1][r1][a2][r2] through the turn-1 cells' maps."""
    if frame.depth != 2:raise ValueError('N=2 only')
    R = frame.pre.shape[0];shape = (R, 2, 9, 9, 9, 9)
    out = {k: v for k, v in vars(frame).items() if k.startswith(('t1_', 'root_')) or k == 'pre'}
    out.update(t2_status=np.zeros(shape, np.int8), t2_value=np.zeros(shape, np.float32), t2_error=np.zeros(shape, np.int8),
        t2_digest=np.zeros(shape + (2,), np.uint32), world_reports=(), timing=frame.timing)
    for q, w, a1, r1 in np.argwhere(np.asarray(frame.t1_status) != 0).tolist():
        for i, a2 in enumerate(frame.t1_own_actions[q, w, a1, r1]):
            for x, r2 in enumerate(frame.t1_opp_actions[q, w, a1, r1]):
                if a2 < 0 or r2 < 0:continue
                for name in ('status', 'value', 'error', 'digest'):out['t2_' + name][q, w, a1, r1, a2, r2] = getattr(frame, 'leaf_' + name)[q, w, a1 * 9 + r1, i, x]
    return SimpleNamespace(**out)


def fake_searched_maps(frame, k_own, k_opp):
    """A stand-in `searched` rule and a per-root flag: each requested seat keeps its encoder-legal actions in DESCENDING action
    order, up to k+1 of them; a waiting seat [0, -1, ...]. The learner's maps are fake_searched_widths wide (k + 2), the
    opponent's one slot NARROWER (k + 1): a narrower map pads to the pricer's width."""
    R = frame.pre.shape[0];own = np.full((R, 2, 9, 9, k_own + 2), -1, np.int32);opp = np.full((R, 2, 9, 9, k_opp + 1), -1, np.int32)
    for q, w, a1, r1 in np.argwhere(np.asarray(frame.t1_status) == 2).tolist():
        for out, k, requested, legal in ((own, k_own, frame.t1_own_requested, frame.t1_own_legal), (opp, k_opp, frame.t1_peer_requested, frame.t1_peer_legal)):
            if not requested[q, w, a1, r1]:out[q, w, a1, r1, 0] = 0;continue
            kept = sorted(np.flatnonzero(legal[q, w, a1, r1]).tolist(), reverse=True)[:k + 1];out[q, w, a1, r1, :len(kept)] = kept
    return own, opp, np.zeros(R, bool)


FAKE_MODULE = SimpleNamespace(DepthNPricer=OracleNPricer, as_depth2_frame=fake_as_depth2, searched_maps=fake_searched_maps)


class FakeBackupsN:
    """depthn_backups' call shapes (the real module lands with the pricer): recursive learner-max over the enumerated slots
    (masked softmax of each seat's prior over its map), terminals exact, any error refuses; a root action's Q is the mean
    over usable worlds and enumerated replies; the target is softmax(root logits + Q) over legal actions."""
    @staticmethod
    def _node(frame, q, w, ply, node):
        if ply == 1:
            a, r = divmod(int(node), 9);at = lambda x: np.asarray(x)[q, w, a, r]
            return (int(at(frame.t1_status)), float(at(frame.t1_value)), at(frame.t1_own_actions), at(frame.t1_opp_actions), at(frame.t1_peer_logits))
        level = frame.nodes[ply - 2];at = lambda x: np.asarray(x)[q, w, node]
        return int(at(level.status)), float(at(level.value)), at(level.own_actions), at(level.opp_actions), at(level.opp_logits)

    @classmethod
    def node_value(cls, frame, q, w, ply, node, rule=None):
        status, value, own, opp, opp_logits = cls._node(frame, q, w, ply, node)
        if status == 1:return value
        if status != 2:return None
        k_opp = len(opp);K = len(own) * k_opp;rows = []
        live = [x for x in range(k_opp) if opp[x] >= 0];logits = np.asarray([float(opp_logits[opp[x]]) for x in live]);pi = np.exp(logits - logits.max());pi /= pi.sum()
        for i in (i for i in range(len(own)) if own[i] >= 0):
            values = []
            for x in live:
                if ply + 1 == frame.depth:
                    s = int(frame.leaf_status[q, w, node, i, x]);v = float(frame.leaf_value[q, w, node, i, x]) if s in (1, 4) else None
                else:v = cls.node_value(frame, q, w, ply + 1, node * K + i * k_opp + x)
                if v is None:return None
                values.append(v)
            rows.append(float(np.dot(pi, values)))
        return max(rows) if rows else None

    @classmethod
    def learner_max_root(cls, frame, q, logits, legal):
        if (np.asarray(frame.pre[q]) != 0).any():return SimpleNamespace(status='REFUSED', improvement=None)
        values = []
        for a1 in range(9):
            cells = [cls.node_value(frame, q, w, 1, a1 * 9 + r1) for w in (0, 1) for r1 in range(9) if frame.t1_status[q, w, a1, r1]] if legal[a1] else []
            if legal[a1] and (not cells or any(c is None for c in cells)):return SimpleNamespace(status='REFUSED', improvement=None)
            values.append(float(np.mean(cells)) if legal[a1] else None)
        scores = np.where(legal, np.asarray(logits) + np.asarray([v or 0. for v in values]), -np.inf);target = np.exp(scores - scores.max())
        return SimpleNamespace(status='LABELED', improvement=SimpleNamespace(completed_q=tuple(values), policy_target=tuple(target / target.sum())))


# ---------------------------------------------------------------- fixture

class FixtureN:
    """Facts-based requests over the fake engine, the real FramePricer and the gate's own oracle, shared per class."""
    MODES = ((0, 0), (18, 1), (101, 0), (102, 1), (103, 0), (104, 1), (105, 0), (0, 1), (108, 0), (109, 1), (110, 0), (111, 1), (112, 0),
        (113, 1), (114, 0), (115, 1), (9, 0), (6, 1), (13, 0))
    MISSING = {(3, 1)}
    PRE_REFUSED = 7
    make = staticmethod(executor_n)

    @classmethod
    def setUpClass(cls):
        _, cls.root, _, cls.tables, cls.recipe = facts();cls.ex = cls.make();cls.requests = cls.make_requests()
        cls.oracle = reference.DepthNOracle(cls.ex, object(), chunk=64);cls.cache = {}

    @classmethod
    def make_requests(cls):
        requests = []
        for i, (mode, owner) in enumerate(cls.MODES):
            stream = 100 + i
            while not cls.recipe.selected(v5_contract.DecisionAddress(stream, 4 + i, owner)):stream += 100
            root = replace(cls.root, address=v5_contract.DecisionAddress(stream, 4 + i, owner))
            starts = plans.plan_starts(root, cls.tables.source, cls.recipe)
            state, public = world(mode, owner, starts.starts[0].seed, float(stream % 79))
            view = cls.ex.observe(state, public, j.asarray([owner]), cls.ex.legal(state)[:, owner]);inputs = view.inputs
            if i == cls.PRE_REFUSED:inputs = {'numeric_features': inputs['numeric_features'] + 1}
            logits = tuple(float(x) for x in (0., .5, .25) + (0.,) * 6)
            requests.append(batch.Request(starts, logits, tuple(map(bool, np.asarray(view.observation.legal_action_mask)[0])), inputs, i % 2, 1 - i % 2))
        return tuple(requests)

    def materialize(self, root, start, tables, recipe, *, keys=None):
        index = root.address.frame - 4;mode, owner = self.MODES[index]
        if (index, start.world) in self.MISSING:return mapping.World(start.world, None, None, ('inert missing',), start.seed, False)
        state, public = world(mode, owner, start.seed, float(root.address.stream % 79))
        return mapping.World(start.world, state, public, (), start.seed, False)

    @contextlib.contextmanager
    def engine(self):
        with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=forward):
            yield

    def frame_pricer(self):
        return batch.FramePricer(self.ex, object(), capacity=32, block_rows=32, actor_capacity=3, vectorized=True, host_prep=True)

    def pricers(self, *, module=FAKE_MODULE, depth=3, k_own=2, k_opp=2, selection='prior', lanes=(2187,), executor=None, k1_own=9, k1_opp=9):
        return gate.DepthNPricers(module, executor or self.ex, object(), depth=depth, k_own=k_own, k_opp=k_opp, selection=selection, capacity=32,
            actor_capacity=3, lanes=lanes, k1_own=k1_own, k1_opp=k1_opp)

    def call(self, pricers, *, sample=None, decision=None, backups=FakeBackupsN, save=None, depth2=True, requests=None):
        module = pricers.module
        depth2_pricer = gate.depth2_pass_pricer(module, self.ex, object(), capacity=32, actor_capacity=3, k1_own=pricers.k1_own, k1_opp=pricers.k1_opp) if depth2 else None
        with self.engine():
            return gate.run_call_n(requests=requests or self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS, pricers=pricers,
                depth2_pricer=depth2_pricer, frame_pricer=self.frame_pricer(), oracle=self.oracle, sample=sample, block=0, context=dict(block=0, call=0),
                decision=decision, backups=backups, save=save)

    def cached(self, key, make):
        if key not in self.cache:self.cache[key] = make()
        return self.cache[key]

    @staticmethod
    def judged(*evaluations):
        result = gate.summarize_run(list(evaluations), (2187,));return result, gate.gate_status_n(result)


def failing(result):
    return {k: v.get('mismatched') or v.get('reason') for k, v in result['checks'].items() if not v['passed']}


# ---------------------------------------------------------------- the oracle

class OracleNTests(FixtureN, unittest.TestCase):
    def test_two_ply_walk_equals_the_depth2_oracle(self):
        from gpu_engine import depth2_oracle as d2
        with self.engine():
            stack = reference.stack(self.requests, batch._numpy_worlds(self.requests, self.tables, self.recipe))
            two = d2.Oracle(self.ex, object(), chunk=64);roots = two.roots(stack, self.requests, PARAMS)
            cells = d2.turn1_cells(self.requests, self.recipe, roots);children = two.children(stack, cells, PARAMS)
            nodes = self.oracle.nodes(stack, cells, 1, PARAMS)
            for name in ('status', 'error', 'value', 'own_digest', 'peer_digest', 'own_legal', 'peer_legal', 'own_requested', 'peer_requested', 'own_logits'):
                np.testing.assert_array_equal(nodes[name], children[name], err_msg=name)
            specs = d2.turn2_leaves(children, k_own=2, k_opp=2);leaves = two.leaves(stack, specs, PARAMS);walked = self.oracle.leaves_n(stack, specs, 2, PARAMS)
            for name in ('status', 'error', 'value', 'digest', 'head_value'):np.testing.assert_array_equal(walked[name], leaves[name], err_msg=name)
            self.assertFalse(walked['broken_path'].any())

    def test_every_step_draws_the_nodes_own_chance(self):
        """Mode 18 re-keys chance from each step's actions: a ply-3 leaf's value is 200 + sum over steps of
        a*30 + r*20 + sample(that step's node)."""
        q = 1;mode, owner = self.MODES[q];assert mode == 18
        with self.engine():
            stack = reference.stack(self.requests, batch._numpy_worlds(self.requests, self.tables, self.recipe))
            seed0 = self.requests[q].plans.starts[0].seed
            specs = [(q, 0, a1, r1, a2, r2, a3, r3) for a1, r1, a2, r2, a3, r3 in ((0, 1, 2, 0, 1, 1), (2, 2, 1, 0, 0, 2), (1, 0, 0, 1, 2, 2))]
            leaves = self.oracle.leaves_n(stack, specs, 3, PARAMS)
        for spec, value in zip(specs, leaves['head_value']):
            seed, total = seed0 % 2**32, 200
            for a, r in zip(spec[2::2], spec[3::2]):
                total += a * 30 + r * 20 + seed % 17;seed = (seed * LCG[0] + LCG[1] + (a + 1) * 7 + (r + 1) * 13) % 2**32
            offset = np.float32(PARAMS[self.requests[q].learner_actor]['offset'])
            self.assertEqual(float(value), float(np.float32(np.float32(total) / np.float32(SCALE)) + offset), spec)

    def test_ranking_and_expansion(self):
        legal, logits = (True, True, True, False) + (False,) * 5, (0., 1., 1., 9.) + (0.,) * 5
        self.assertEqual(reference.ranked(legal, logits), [1, 2, 0])
        self.assertEqual(reference.ranked(legal, (np.nan, 1., -np.inf) + (0.,) * 6), [1, 0, 2])   # non-finite last, lower action first
        self.assertEqual(reference.prior_map(legal, logits, True, 2).tolist(), [1, 2])
        self.assertEqual(reference.prior_map(legal, logits, True, 5).tolist(), [1, 2, 0, -1, -1])
        self.assertEqual(reference.prior_map(legal, logits, False, 3).tolist(), [0, -1, -1])
        children = reference.expand([((0, 1, 2, 3), 7)], [np.asarray([4, -1, 5])], [np.asarray([0, -1])])
        self.assertEqual([(c[1], c[3], c[4]) for c in children], [(7 * 6 + 0, 0, 0), (7 * 6 + 4, 2, 0)])
        self.assertEqual(children[1][0], (0, 1, 2, 3, 5, 0))
        rng = np.random.default_rng(3);specs = rng.integers(0, 9, (6, 4));ids = rng.integers(0, 81, 6)
        own, opp = np.where(rng.random((6, 3)) < .3, -1, rng.integers(0, 9, (6, 3))), np.where(rng.random((6, 2)) < .3, -1, rng.integers(0, 9, (6, 2)))
        listed = reference.expand(list(zip(map(tuple, specs.tolist()), ids.tolist())), own, opp);arrays = reference.expand_arrays(specs, ids, own, opp)
        self.assertEqual([c[0] for c in listed], list(map(tuple, arrays[0].tolist())));self.assertEqual([c[1:] for c in listed], list(zip(*(a.tolist() for a in arrays[1:]))))
        self.assertEqual(reference.expand_arrays(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 2)), np.zeros((0, 2)))[0].shape, (0, 6))

    def test_ply2_statuses_and_codes_follow_depth2_at_every_ply(self):
        surface = self.cached('prior', lambda: self.call(self.pricers()))[0]
        levels = surface['oracle']   # the oracle's own consistency
        self.assertTrue(levels['passed'])
        frame_status = None
        with self.engine():
            stack = reference.stack(self.requests, batch._numpy_worlds(self.requests, self.tables, self.recipe))
            data = gate.oracle_levels(oracle=self.oracle, stack=stack, requests=self.requests, recipe=self.recipe, params=PARAMS, frame=frame_status,
                plan=gate.plan_of(3, 2, 2))
        ply2 = data['levels'][2];by_root = collections.defaultdict(set)
        for spec, status, error in zip(ply2['specs'].tolist(), ply2['status'], ply2['error']):by_root[spec[0]].add((int(status), int(error)))
        self.assertIn((1, 0), by_root[2]);self.assertIn((1, 0), by_root[3])   # ply-2 terminal win / draw
        self.assertIn((3, 1), by_root[4]);self.assertIn((3, 3), by_root[5]);self.assertIn((3, 4), by_root[6])   # invalid, encoder, head
        self.assertIn((3, 5), by_root[13]);self.assertIn((3, 6), by_root[14]);self.assertEqual(by_root[15], {(2, 0)})   # 5/6 only while requested
        self.assertIn((1, 0), by_root[16])   # an action-dependent ply-2 terminal
        leaf = collections.defaultdict(set)
        for index, status, error in zip(data['leaves']['index'].tolist(), data['leaves']['status'], data['leaves']['error']):leaf[index[0]].add((int(status), int(error)))
        self.assertIn((1, 0), leaf[9]);self.assertIn((3, 1), leaf[10]);self.assertIn((3, 3), leaf[11]);self.assertIn((4, 0), leaf[0])
        waiting = [(m, ply2['own_requested'][i], ply2['peer_requested'][i]) for i, m in enumerate(ply2['specs'][:, 0]) if m in (8, 12, 17)]
        self.assertTrue(all(not peer for m, own, peer in waiting if m in (8, 17)) and all(not own for m, own, _ in waiting if m == 12))
        maps = [data['maps'][(3, q, w, n)] for (p, q, w, n) in data['maps'] if p == 3 and q == 12]
        self.assertTrue(maps and all(own.tolist() == [0, -1] for own, _ in maps))   # the waiting owner at ply 2: [0, -1]


# ---------------------------------------------------------------- the gate on the stand-in surface

class GateNTests(FixtureN, unittest.TestCase):
    def prior(self):
        return self.cached('prior', lambda: self.call(self.pricers()))

    def test_the_oracles_own_surface_passes_every_check(self):
        evaluation, summary = self.prior();result, (status, reasons) = self.judged(evaluation)
        self.assertEqual(status, 'PASS', (failing(result), reasons))
        a = result['checks']['a']['compared']
        for name in ('ply2_maps', 'ply3_maps', 'ply2_status', 'ply2_digest_own', 'ply2_digest_opp', 'ply2_own_legal', 'ply2_parent', 'leaf_status',
                'leaf_digest', 'leaf_value_terminal', 'ply2_value_terminal', 't1_digest_own', 'pre'):
            self.assertGreater(a.get(name, 0), 0, name)
        c = result['checks']['c']['quantities']
        for name in ('ply2_value_head', 'ply2_own_logits', 'ply2_opp_logits', 'leaf_value', 't1_value_head', 'depth1_noise_floor'):
            self.assertGreater(c[name]['compared'], 0, name);self.assertEqual(c[name]['max_abs_difference'], 0., name)
        self.assertGreater(result['checks']['d']['compared']['leaf_enumerated'], 0);self.assertGreater(result['checks']['d']['compared']['ply2_enumerated'], 0)
        e = result['checks']['e'];self.assertGreater(e['roots_compared'], 0);self.assertEqual(e['argmax_q_differs'], 0)
        self.assertGreater(summary['refusals']['depth1']['refused_count'], 0);self.assertIn('depth2_full_width', summary['refusals'])
        self.assertGreater(summary['refusals']['depthn']['refused_count'], summary['refusals']['depth1']['refused_count'])   # ply-2/leaf errors refuse too
        self.assertGreater(len(result['tolerance']['table']), 0)

    def test_sampled_leaves_keep_every_terminal_and_error_leaf(self):
        evaluation, summary = self.call(self.pricers(), sample=1)
        result = gate.summarize_run([evaluation], (2187,));status, _ = gate.gate_status_n(result, sampled=True)
        self.assertEqual(status, 'PASS_SAMPLED', failing(result));self.assertNotIn('e', evaluation)
        self.assertLess(summary['leaves_checked'], summary['leaves_enumerated']);self.assertGreater(summary['forced_leaves'], 0)
        full = self.prior()[1]
        self.assertEqual(summary['leaves_enumerated'], full['leaves_enumerated'])

    def test_turn1_widths(self):
        evaluation, summary = self.call(self.pricers(k1_own=3, k1_opp=2))
        result, (status, reasons) = self.judged(evaluation)
        self.assertEqual(status, 'PASS', (failing(result), reasons));self.assertNotIn('t1_enumeration_vs_d', result['checks']['b']['compared'])
        self.assertLess(summary['refusals']['depthn']['refused_count'], 19)
        evaluation, summary = self.call(self.pricers(k1_own=2, k1_opp=2))
        result, (status, reasons) = self.judged(evaluation)   # k1_own = 2 prunes a legal root action of every root: all refuse
        self.assertEqual((status, summary['refusals']['depthn']['refused_count']), ('INCOMPLETE', 19));self.assertIn('vacuous', reasons[0])

    def mutant(self, **overrides):
        """A run whose pricer is a mutant stand-in (the gate's oracle and D3 stay correct)."""
        class Module:
            DepthNPricer = staticmethod(lambda executor, model, **kw: OracleNPricer(overrides.get('executor', executor), model,
                **dict(kw, **{k: v for k, v in overrides.items() if k != 'executor'})))
            as_depth2_frame = staticmethod(fake_as_depth2);searched_maps = staticmethod(fake_searched_maps)
        return Module

    def test_mutant_ply3_selection_is_caught(self):
        """Ply-3 ties broken toward the higher action (ply 2 untouched: its width differs): ply3_maps and the leaf digests fail."""
        def wrong(legal, logits, requested, k):
            if k != 3 or not requested:return reference.prior_map(legal, logits, requested, k)
            m = np.full(k, -1, np.int32);order = sorted((i for i in range(9) if legal[i]), key=lambda i: (-float(logits[i]), -i))[:k];m[:len(order)] = order
            return m
        bad = SimpleNamespace(**{k: getattr(reference, k) for k in dir(reference) if not k.startswith('__')});bad.prior_map = wrong
        evaluation, _ = self.call(self.pricers(module=self.mutant(reference_module=bad), k_own=(2, 3)), depth2=False)
        result, (status, _) = self.judged(evaluation);mismatched = result['checks']['a']['mismatched']
        self.assertEqual(status, 'FAIL');self.assertGreater(mismatched['ply3_maps'], 0);self.assertEqual(mismatched['ply2_maps'], 0)
        self.assertGreater(mismatched['leaf_digest'], 0)

    def test_mutant_ply3_chance_is_caught(self):
        """The ply-3 step drawn from a constant instead of the ply-2 node's own sample: leaf digests and values fail."""
        bad = executor_n(sample=lambda state: j.where(state.turn == 2, 0, state.seed % 17).astype(j.int32))
        evaluation, _ = self.call(self.pricers(module=self.mutant(executor=bad)), depth2=False)
        result, (status, _) = self.judged(evaluation)
        self.assertEqual(status, 'FAIL');self.assertGreater(result['checks']['a']['mismatched']['leaf_digest'], 0)
        self.assertEqual(result['checks']['a']['mismatched']['ply2_digest_own'], 0)
        self.assertFalse(result['checks']['c']['quantities']['leaf_value']['passed'])

    def test_mutant_ply3_masks_are_caught(self):
        """On permuted encoder lanes, a pricer whose ply-2 views carry the engine mask: ply-2 masks, ply-3 maps and leaves fail."""
        class Permuted(FixtureN):
            make = staticmethod(permuted_n)
        Permuted.setUpClass();fixture = Permuted()
        clean, _ = fixture.call(fixture.pricers(), depth2=False)
        self.assertEqual(self.judged(clean)[1][0], 'PASS', failing(self.judged(clean)[0]))
        evaluation, _ = fixture.call(fixture.pricers(module=self.mutant(executor=permuted_n(engine_mask_at_turn=2))), depth2=False)
        result, (status, _) = self.judged(evaluation);mismatched = result['checks']['a']['mismatched']
        self.assertEqual(status, 'FAIL');self.assertGreater(mismatched['ply2_own_legal'], 0);self.assertGreater(mismatched['ply3_maps'], 0)
        self.assertEqual(mismatched['t1_own_legal'], 0)

    def test_each_kind_of_frame_mismatch_is_caught(self):
        evaluation, _ = self.prior()
        frame = self.cached('prior_frame', lambda: self._frame())
        level = frame.nodes[0];expanded = np.argwhere(np.asarray(level.status) == 2)[0];at = tuple(expanded)
        leaf = tuple(np.argwhere(np.asarray(frame.leaf_status) == 4)[0])
        cases = dict(
            ply2_status=lambda f: f.nodes[0].status.__setitem__(at, 1), ply2_error=lambda f: f.nodes[0].error.__setitem__(at, 4),
            ply2_digest_opp=lambda f: f.nodes[0].digest_opp.__setitem__(at + (0,), 7), ply2_opp_requested=lambda f: f.nodes[0].opp_requested.__setitem__(at, False),
            ply2_parent=lambda f: f.nodes[0].parent.__setitem__(5, 3), leaf_error=lambda f: f.leaf_error.__setitem__(leaf, 3),
            leaf_digest=lambda f: f.leaf_digest.__setitem__(leaf + (1,), 9), ply3_maps=lambda f: f.nodes[0].own_actions.__setitem__(at + (0,), 0))
        for name, change in cases.items():
            mutated = copy.deepcopy(frame);change(mutated)
            data = self._data(mutated)
            with self.subTest(name):
                ev = gate.evaluate_n(mutated, self._d(), self.requests, data, context={})
                self.assertGreater(ev['a']['mismatched'].get(name, 0) + ev['d']['mismatched'].get(name, 0), 0, ev['a']['mismatched'])
        value = copy.deepcopy(frame);value.leaf_value[leaf] += 1.;ev = gate.evaluate_n(value, self._d(), self.requests, self._data(value), context={})
        self.assertFalse(gate.summarize_n([ev])['checks']['c']['passed'])
        logits = copy.deepcopy(frame);logits.nodes[0].opp_logits[at + (0,)] += .5;ev = gate.evaluate_n(logits, self._d(), self.requests, self._data(logits), context={})
        self.assertFalse(gate.summarize_n([ev])['checks']['c']['quantities']['ply2_opp_logits']['passed'])
        extra = copy.deepcopy(frame);extra.leaf_status[tuple(np.argwhere(np.asarray(frame.leaf_status) == 0)[0])] = 4
        ev = gate.evaluate_n(extra, self._d(), self.requests, self._data(extra), context={});self.assertGreater(ev['d']['mismatched']['leaf_extra_in_pricer'], 0)

    def _frame(self):
        with self.engine():return OracleNPricer(self.ex, object(), depth=3, k_own=2, k_opp=2).price(self.requests, self.tables, self.recipe, PARAMS)

    def _d(self):
        if 'd' not in self.cache:
            with self.engine():self.cache['d'] = self.frame_pricer().price(self.requests, self.tables, self.recipe, PARAMS)
        return self.cache['d']

    def _data(self, frame):
        with self.engine():
            stack = reference.stack(self.requests, batch._numpy_worlds(self.requests, self.tables, self.recipe))
            return gate.oracle_levels(oracle=self.oracle, stack=stack, requests=self.requests, recipe=self.recipe, params=PARAMS, frame=frame, plan=gate.plan_of(3, 2, 2))

    def test_wrong_geometry_fails_closed(self):
        frame = copy.deepcopy(self.cached('prior_frame', lambda: self._frame()));frame.nodes = ()
        ev = gate.evaluate_n(frame, self._d(), self.requests, self._data(self.cached('prior_frame', lambda: self._frame())), context={})
        self.assertTrue(ev['geometry']);self.assertFalse(gate.summarize_n([ev])['passed'])

    def test_block_invariance_over_block_lanes(self):
        evaluation, summary = self.call(self.pricers(lanes=gate.INVARIANCE_LANES))
        result, (status, _) = self.judged(evaluation)
        self.assertEqual(status, 'PASS', failing(result));inv = result['checks']['block_invariance']
        self.assertTrue(inv['discrete_equal']);self.assertEqual(inv['block_lanes'], [2187]);self.assertGreater(inv['arrays_compared'], 20)
        a, b = self._frame(), self._frame();b.nodes[0].own_actions[0, 0, 0, 0] = 2
        self.assertFalse(gate.g2.block_invariance(gate.frame_arrays(a), gate.frame_arrays(b))['passed'])

    def test_searched_pricers_take_ks_and_the_maps_pad_to_the_derived_widths(self):
        pricers = self.pricers(selection='searched');evaluation, summary = self.call(pricers)
        result, (status, _) = self.judged(evaluation)
        self.assertEqual(status, 'PASS', failing(result))
        self.assertEqual([(p.searched_k, p.k_own, p.k_opp) for p in pricers.instances], [((2, 2), (4, 2), (4, 2))])   # built with k's
        self.assertEqual(summary['plan']['k_own'], (4, 2));self.assertEqual(summary['plan']['k_opp'], (4, 2))   # the engine's derived widths
        self.assertEqual(summary['searched'], dict(searched_k=[2, 2], searched_widths=[4, 4]))
        self.assertEqual(summary['searched_maps']['widths'], dict(own=4, opp=3));self.assertEqual(summary['searched_maps']['extras'], [dict(shape=[19], dtype='bool', set=0)])
        self.assertEqual(pricers.options()['searched_k'], [2, 2]);self.assertEqual(pricers.options()['slot_widths'], dict(k_own=[4, 2], k_opp=[4, 2]))
        self.assertGreater(result['checks']['a']['compared']['searched_maps_own'], 0)
        self.assertEqual(summary['timing'].keys() >= {'searched_maps_seconds'}, True)
        def wider(frame, k_own, k_opp):
            own, opp, flag = fake_searched_maps(frame, k_own, k_opp);return np.concatenate([own, own[..., :1] * 0 - 1], axis=-1), opp, flag
        with self.assertRaises(ValueError):self.call(self.pricers(module=SimpleNamespace(DepthNPricer=OracleNPricer, as_depth2_frame=fake_as_depth2, searched_maps=wider), selection='searched'))
        def duplicate(frame, k_own, k_opp):
            own, opp, flag = fake_searched_maps(frame, k_own, k_opp);cell = tuple(np.argwhere(np.asarray(frame.t1_status) == 2)[0]);own[cell][1] = own[cell][0]
            return own, opp, flag
        module = SimpleNamespace(DepthNPricer=OracleNPricer, as_depth2_frame=fake_as_depth2, searched_maps=duplicate)
        evaluation, _ = self.call(self.pricers(module=module, selection='searched'))
        self.assertGreater(evaluation['a']['mismatched']['searched_maps_own'], 0)

    def test_refusals_at_depth_n_beside_depth1_and_depth2(self):
        _, summary = self.prior();refusals = summary['refusals']
        own = refusals.get('depthn_gate_count', refusals['depthn'])
        self.assertEqual(own['refused_count'], int(gate.depthn_refused(self._frame(), [r.root_legal for r in self.requests]).sum()))
        for root in (4, 5, 6, 10, 11, 13, 14):self.assertIn(root, own['refused_roots'])   # ply-2 or leaf errors only
        for root in (0, 1, 15, 17):self.assertNotIn(root, own['refused_roots'])
        self.assertNotIn(4, refusals['depth1']['refused_roots']);self.assertIn(self.PRE_REFUSED, refusals['depth1']['refused_roots'])
        with patch.dict(sys.modules, {'gpu_engine.depthn_backups': None}):
            report = gate.refusal_report_n(self._frame())
        self.assertEqual(report['depthn']['source'], 'gate count of the frame');self.assertRegex(report['refusal_summary'], 'ImportError|ModuleNotFoundError')

    def test_decision_flip_fails_and_missing_backups_fail_closed(self):
        evaluation, _ = self.prior();frame = self._frame();data = self._data(frame)
        swapped, counts = gate.oracle_heads_frame_n(frame, data);self.assertEqual(counts['unmatched_head_leaves'], 0);self.assertGreater(counts['leaf_value'], 0)
        root = 0;labeled = FakeBackupsN.learner_max_root(frame, root, self.requests[root].root_logits, self.requests[root].root_legal)
        q = labeled.improvement.completed_q;best = gate.g2._argmax(q, self.requests[root].root_legal)
        lowered = copy.deepcopy(frame)
        cells = np.argwhere(np.asarray(frame.leaf_status[root]) == 4);mask = np.zeros(frame.leaf_status.shape, bool)
        for w, parent, i, x in cells.tolist():
            if parent // 4 // 9 == best:mask[root, w, parent, i, x] = True   # ply-2 node -> turn-1 cell -> a1 (K = 2 x 2)
        lowered.leaf_value[mask] -= 1.
        result = gate.decide_roots(lowered, swapped, self.requests, FakeBackupsN, q_noise_bound=0.)
        self.assertFalse(result['passed']);self.assertIn(root, result['differing_roots'])
        with patch.dict(sys.modules, {'gpu_engine.depthn_backups': None}):
            self.assertFalse(gate.decision_check_n(frame, data, self.requests)['passed'])

    def test_saved_frames_round_trip_and_feed_the_diagnostics(self):
        directory = Path(tempfile.mkdtemp(prefix='depthn-gate-test-'))
        for selection, k in (('prior', 2), ('searched', 2), ('prior', 3)):
            with self.subTest(selection=selection, k=k):
                self.call(self.pricers(selection=selection, k_own=k, k_opp=k), save=(directory, 'frames-block0-call0'), decision=False)
        names = sorted(p.name for p in directory.iterdir())
        self.assertEqual(names, ['frames-block0-call0-depth2.npz', 'frames-block0-call0-prior-k2.npz', 'frames-block0-call0-prior-k3.npz',
            'frames-block0-call0-searched-k2.npz'])
        frame, requests, meta = gate.load_frame(directory / 'frames-block0-call0-prior-k2.npz');original = self._frame()
        for name, value in gate.frame_arrays(original).items():np.testing.assert_array_equal(gate.frame_arrays(frame)[name], value, err_msg=name)
        self.assertEqual(meta['depth'], 3);self.assertEqual([r.plans.root.address.stream for r in requests], [r.plans.root.address.stream for r in self.requests])
        self.assertEqual(FakeBackupsN.node_value(frame, 0, 0, 1, 0), FakeBackupsN.node_value(original, 0, 0, 1, 0))
        report = gate.offline_diagnostics([directory], backups=FakeBackupsN, backups2=depth2_backups)
        self.assertEqual(report['calls'], 1);self.assertEqual(set(report['change_rate']), {'prior-k2', 'prior-k3', 'searched-k2'})
        change = report['change_rate']['prior-k2'];self.assertGreater(change['labeled_both'], 0);self.assertIsNotNone(change['rate'])
        pvs = report['prior_vs_searched']['k2'];self.assertGreater(pvs['cells'], 0);self.assertGreater(pvs['set_differs'], 0)   # descending vs prior rank
        pruned = report['pruned_errors']['prior-k2'];self.assertEqual(pruned['roots'], 19);self.assertGreater(pruned['with_depth2_errors'], 0)
        self.assertEqual(pruned['roots'], sum(pruned.get(k, 0) for k in ('pre_or_materialization', 'no_depth2_errors', 'with_depth2_errors')))
        self.assertEqual(pruned['with_depth2_errors'], pruned.get('errors_kept', 0) + pruned.get('only_pruned_errors', 0))
        self.assertIn('definitions', report)

    def test_pruned_error_share_is_exact_on_a_synthetic_case(self):
        d2 = SimpleNamespace(pre=np.zeros((3, 2), np.int32), t1_status=np.zeros((3, 2, 9, 9), np.int8), t2_status=np.zeros((3, 2, 9, 9, 9, 9), np.int8))
        d3 = SimpleNamespace(t1_own_actions=np.full((3, 2, 9, 9, 2), -1, np.int32), t1_opp_actions=np.full((3, 2, 9, 9, 2), -1, np.int32))
        d3.t1_own_actions[:, :, 0, 0] = (1, 2);d3.t1_opp_actions[:, :, 0, 0] = (0, 1)
        d2.t2_status[0, 0, 0, 0, 0, 0] = 3                       # root 0: a2 = 0 is pruned
        d2.t2_status[1, 0, 0, 0, 1, 1] = 3                       # root 1: (1, 1) is kept
        d2.t1_status[2, 1, 3, 3] = 3                             # root 2: a turn-1 error is never pruned
        counts = gate.pruned_error_counts(d2, d3)
        self.assertEqual((counts['only_pruned_errors'], counts['errors_kept'], counts['with_depth2_errors']), (1, 2, 3))

    def test_n2_compare_is_exact_on_identical_surfaces_and_catches_each_kind(self):
        frame = fake_as_depth2(self.cached('n2', lambda: self._n2_frame()))
        found, floats = gate.n2_compare(frame, frame);self.assertTrue(found['passed']);self.assertGreater(found['compared']['t2_status'], 0)
        self.assertTrue(all(m['max_abs_difference'] == 0. for m in floats.values()));self.assertGreater(floats['t2_value']['compared'], 0)
        cell = tuple(np.argwhere(np.asarray(frame.t2_status) == 4)[0])
        for name, change in (('t2_status', lambda f: f.t2_status.__setitem__(cell, 1)), ('t2_digest', lambda f: f.t2_digest.__setitem__(cell + (0,), 1)),
                ('t1_peer_legal', lambda f: f.t1_peer_legal.__setitem__((0, 0, 0, 0, 8), True))):
            other = copy.deepcopy(frame);change(other)
            with self.subTest(name):self.assertGreater(gate.n2_compare(frame, other)[0]['mismatched'][name], 0)
        other = copy.deepcopy(frame);other.t2_value[cell] += .25
        self.assertEqual(gate.n2_compare(frame, other)[1]['t2_value']['max_abs_difference'], .25)

    def _n2_frame(self):
        with self.engine():return OracleNPricer(self.ex, object(), depth=2, k_own=9, k_opp=9).price(self.requests, self.tables, self.recipe, PARAMS)


# ---------------------------------------------------------------- the real depth-N pricer (when on the path)

@unittest.skipUnless(HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
class RealPricerTests(FixtureN, unittest.TestCase):
    def module(self):
        return importlib.import_module('gpu_engine.device_depthn_pricer')

    def real(self, **kw):
        return self.pricers(module=self.module(), **kw)

    def test_depth3_prior_passes_every_check(self):
        evaluation, summary = self.call(self.real(lanes=(8, 16)), backups=None)
        result, (status, reasons) = self.judged(evaluation)
        self.assertEqual(status, 'PASS', (failing(result), reasons, result['checks'].get('a', {}).get('first_mismatches')))
        self.assertTrue(result['checks']['block_invariance']['discrete_equal'])
        self.assertTrue(summary['refusals']['depthn_sources_agree'], summary['refusals'])
        self.assertEqual(summary['refusals']['depthn']['source'], 'depthn_backups.refusal_summary')

    def test_depth3_searched_passes_every_check(self):
        evaluation, summary = self.call(self.real(selection='searched'), backups=None)
        result, (status, reasons) = self.judged(evaluation)
        self.assertEqual(status, 'PASS', (failing(result), reasons, result['checks'].get('a', {}).get('first_mismatches')))
        searched_widths = self.module().searched_widths(2, 2)
        self.assertEqual((summary['plan']['k_own'][0], summary['plan']['k_opp'][0]), searched_widths)
        self.assertEqual(summary['searched'], dict(searched_k=[2, 2], searched_widths=list(searched_widths)))
        self.assertTrue(summary['searched_maps']['widths']['own'] <= searched_widths[0] and summary['searched_maps']['widths']['opp'] <= searched_widths[1])

    def test_depth4_and_mixed_widths_pass(self):
        for depth, k_own, k_opp, k1 in ((4, 2, 2, (9, 9)), (3, (3, 2), (1, 3), (9, 9)), (3, 2, 2, (3, 2))):
            with self.subTest(depth=depth, k_own=k_own, k_opp=k_opp, k1=k1):
                evaluation, summary = self.call(self.real(depth=depth, k_own=k_own, k_opp=k_opp, k1_own=k1[0], k1_opp=k1[1]), backups=None)
                self.assertTrue(summary['refusals']['depthn_sources_agree'], summary['refusals'])
                result, (status, reasons) = self.judged(evaluation)
                self.assertEqual(status, 'PASS', (failing(result), reasons, result['checks'].get('a', {}).get('first_mismatches')))

    def test_n2_equivalence_with_the_depth2_kernel(self):
        from gpu_engine.device_depth2_pricer import Depth2Pricer
        module = self.module();configs = []
        for name, k in (('full_width', 9), ('top2', 2)):
            configs.append((name, module.DepthNPricer(self.ex, object(), capacity=32, depth=2, k_own=k, k_opp=k, digest=True, actor_capacity=3),
                Depth2Pricer(self.ex, object(), capacity=32, k_own=k, k_opp=k, digest=True, actor_capacity=3)))
        with self.engine():
            results = gate.n2_equivalence_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS, configs=configs,
                frame_pricer=self.frame_pricer(), oracle=self.oracle, context=dict(block=0, call=0), module=module)
        report = gate.summarize_n2({name: [result] for name, result in results.items()})
        for name, r in report.items():
            self.assertTrue(r['passed'], (name, r['discrete']['mismatched'], {k: v['max_abs_difference'] for k, v in r['floats'].items()}, r['e'].get('reasons')))
            self.assertGreater(r['e']['roots_compared'], 0)

    def test_live_diagnostics_from_real_frames(self):
        module = self.module();backups = importlib.import_module('gpu_engine.depthn_backups')
        with self.engine():
            depth2 = module.as_depth2_frame(gate.depth2_pass_pricer(module, self.ex, object(), capacity=32, actor_capacity=3).price(
                self.requests, self.tables, self.recipe, PARAMS))
            frames = {}
            for mode in gate.MODES:
                pricers = self.real(selection=mode);maps, first, _ = pricers.maps(depth2);_, (pricer,) = pricers.get()
                frames[(mode, 2)] = pricer.price(self.requests, self.tables, self.recipe, PARAMS, **({} if maps is None else dict(selection_maps=maps)))
        report = gate.merge_diagnostics([gate.diagnostics_call(depth2, frames, self.requests, backups=backups, backups2=depth2_backups)])
        self.assertEqual(set(report['change_rate']), {'prior-k2', 'searched-k2'});self.assertGreater(report['prior_vs_searched']['k2']['cells'], 0)
        for mode in gate.MODES:   # depthn_backups' own learner_max_root agrees with the gate's turn-2 choices' building blocks
            choices = gate.turn2_choices(frames[(mode, 2)], backups);self.assertTrue(choices)


    def test_throughput_callers_run_the_real_pricer(self):
        from gpu_engine import d3_depthn_throughput as tool
        args = tool.parse(['--d3-root', 'r', '--chunk', '0', '--out', 'o', '--capacity', '32', '--ks', '2', '--worlds', '2', '--repeats', '1', '--block-lanes', '16'])
        ctx = SimpleNamespace(provider=SimpleNamespace(executor=self.ex, materializer=self.tables), plan=SimpleNamespace(model=object()), params=PARAMS,
            planned=[(0, self.recipe, list(self.requests), contextlib.nullcontext)], actor_capacity=3)
        with self.engine(), patch.object(tool.t2, 'memory_stats', lambda: {}):
            sections = {c: tool.measure_config(c, args, ctx, None) for c in args.configs}
        searched = sections['depth3-searched-k2']['blocks'][0]['calls'][1]
        self.assertEqual(searched['maps_widths'], [int(x) for x in searched['maps_widths']]);self.assertGreater(searched['leaves_evaluated'], 0)
        for name, section in sections.items():self.assertIsNotNone(section['derived']['2']['device_ms_per_root'], name)

    def test_the_gate_catches_a_real_pricer_on_a_broken_ply3(self):
        """The real pricer driven by mutated engines (the gate's oracle and D3 keep the true one): the ply-3 step's chance
        drawn from a constant, and (on permuted lanes) the ply-2 views carrying the engine mask."""
        bad = executor_n(sample=lambda state: j.where(state.turn == 2, 0, state.seed % 17).astype(j.int32))
        evaluation, _ = self.call(self.real(executor=bad), backups=None, depth2=False)
        result, (status, _) = self.judged(evaluation);mismatched = result['checks']['a']['mismatched']
        self.assertEqual(status, 'FAIL');self.assertGreater(mismatched['leaf_digest'], 0);self.assertEqual(mismatched['ply2_digest_own'], 0)
        class Permuted(FixtureN):
            make = staticmethod(permuted_n)
        Permuted.setUpClass();fixture = Permuted()
        clean, _ = fixture.call(fixture.pricers(module=self.module()), backups=None, depth2=False)
        self.assertEqual(self.judged(clean)[1][0], 'PASS', failing(self.judged(clean)[0]))
        evaluation, _ = fixture.call(fixture.pricers(module=self.module(), executor=permuted_n(engine_mask_at_turn=2)), backups=None, depth2=False)
        result, (status, _) = self.judged(evaluation);mismatched = result['checks']['a']['mismatched']
        self.assertEqual(status, 'FAIL');self.assertGreater(mismatched['ply2_own_legal'], 0);self.assertEqual(mismatched['t1_own_legal'], 0)


# ---------------------------------------------------------------- the command line

class CommandLineRunTests(FixtureN, unittest.TestCase):
    """main() end to end with the roots loader replaced by the fixture's requests (no bank): the gate, the saved frames and
    both diagnostics paths, and --n2-equivalence against the real depth-2 kernel (the stand-in or the real depth-N pricer)."""
    def roots(self):
        provider = SimpleNamespace(executor=self.ex, materializer=self.tables, recipe=self.recipe, source=self.tables.source)
        loaded = SimpleNamespace(provider=provider, plan=SimpleNamespace(model=object(), historical_window=2), frame_pricer=self.frame_pricer(), source=dict(fixture=True))
        return SimpleNamespace(loaded=loaded, head=None, params=PARAMS, groups=[list(self.requests)], root_values=[.1, .2], evaluator=dict(override=False), actor_capacity=3)

    def main(self, *argv, real=False, out=None):
        out = out or Path(tempfile.mkdtemp(prefix='depthn-cli-')) / 'out';find = importlib.util.find_spec
        names = ('gpu_engine.device_depthn_pricer', 'gpu_engine.depthn_backups', 'gpu_engine.depth2_backups', 'gpu_engine.device_depth2_pricer')
        modules = {} if real else {'gpu_engine.depthn_backups': FakeBackupsN}
        with self.engine(), patch.object(gate, 'load_roots', lambda args: self.roots()), patch.object(gate, 'log', lambda *a, **k: None), \
                patch.object(gate, 'pricer_module', (lambda: importlib.import_module('gpu_engine.device_depthn_pricer')) if real else (lambda: FAKE_MODULE)), \
                patch.object(importlib.util, 'find_spec', side_effect=lambda name, *a: object() if name in names and not real else find(name, *a)), \
                patch.dict(sys.modules, modules):
            code = gate.main(['--out', str(out), '--d3-root', 'x', '--chunk', '0', '--k-own', '2', '--k-opp', '2', '--capacity', '32', *argv])
        return code, out

    def test_gate_save_and_offline_diagnostics(self):
        code, prior = self.main('--save-arrays')
        report = json.loads((prior / 'gate.json').read_text())
        self.assertEqual((code, report['status']), (0, 'PASS'), report.get('status_reasons'));self.assertEqual(list(report)[:2], ['evaluator', 'value_scale'])
        self.assertTrue((prior / 'frames-block0-call0-prior-k2.npz').exists() and (prior / 'frames-block0-call0-depth2.npz').exists())
        self.assertIn('depth2_full_width', report['refusals']);self.assertGreater(report['leaves']['checked'], 0)
        code, searched = self.main('--save-arrays', '--selection', 'searched', '--sample-leaves', '2')
        report = json.loads((searched / 'gate.json').read_text());self.assertEqual((code, report['status']), (0, 'PASS_SAMPLED'))
        self.assertEqual(report['searched_maps'][0]['widths'], dict(own=4, opp=3))
        out = Path(tempfile.mkdtemp(prefix='depthn-cli-')) / 'diag'
        with patch.object(gate, 'log', lambda *a, **k: None), patch.dict(sys.modules, {'gpu_engine.depthn_backups': FakeBackupsN}):
            code = gate.main(['--out', str(out), '--diagnostics-from', str(prior), str(searched)])
        diagnostics = json.loads((out / 'diagnostics.json').read_text())
        self.assertEqual((code, diagnostics['calls']), (0, 1));self.assertEqual(set(diagnostics['change_rate']), {'prior-k2', 'searched-k2'})
        self.assertEqual(len(diagnostics['duplicates']), 1)   # both runs saved the depth-2 pass; the first wins
        self.assertGreater(diagnostics['prior_vs_searched']['k2']['cells'], 0)

    def test_live_diagnostics(self):
        code, out = self.main('--depth3-diagnostics', '--diagnostic-ks', '2,3')
        report = json.loads((out / 'diagnostics.json').read_text())
        self.assertEqual(code, 0);self.assertEqual(set(report['change_rate']), {'prior-k2', 'prior-k3', 'searched-k2', 'searched-k3'})
        self.assertEqual(set(report['prior_vs_searched']), {'k2', 'k3'});self.assertEqual(report['widths']['searched-k3'], ['{"k_own": [5, 3], "k_opp": [5, 3]}'])

    def test_n2_equivalence_against_the_depth2_kernel(self):
        code, out = self.main('--n2-equivalence', '--n2-k', '2')
        report = json.loads((out / 'n2-equivalence.json').read_text())
        self.assertEqual((code, report['status']), (0, 'PASS'), {n: (c['discrete']['mismatched'], c['e'].get('reasons')) for n, c in report['configs'].items()})
        self.assertEqual(set(report['configs']), {'full_width', 'top2'})
        for config in report['configs'].values():
            self.assertGreater(config['discrete']['compared']['t2_status'], 0);self.assertGreater(config['e']['roots_compared'], 0)

    @unittest.skipUnless(HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
    def test_real_pricer_gate_and_n2_equivalence(self):
        code, out = self.main('--block-invariance', '--selection', 'searched', real=True)
        report = json.loads((out / 'gate.json').read_text());self.assertEqual((code, report['status']), (0, 'PASS'), report.get('status_reasons'))
        self.assertTrue(report['checks']['block_invariance']['passed']);self.assertEqual(report['checks']['block_invariance']['block_lanes'], [729, 2187])
        code, out = self.main('--n2-equivalence', '--n2-k', '2', real=True)
        self.assertEqual(code, 0, json.loads((out / 'n2-equivalence.json').read_text())['configs'])


class CommandLineTests(unittest.TestCase):
    def test_arguments_are_validated_and_missing_modules_fail_before_loading(self):
        out = Path(tempfile.gettempdir()) / f'depthn-gate-never-created-{uuid.uuid4().hex}'
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (['--out', str(out)], ['--out', str(out), '--saved-roots', 'a'], ['--out', str(out), '--d3-root', 'b'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--k-own', '10'], ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--k-own', '2,2,2'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--depth', '2', '--selection', 'searched'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--n2-equivalence', '--depth3-diagnostics'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--block-invariance', '--n2-equivalence'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--raw-head-check'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--sample-leaves', '0'],
                    ['--out', str(out), '--diagnostics-from', 'x', '--d3-root', 'b']):
                with self.assertRaises(SystemExit) as raised:gate.main(argv)
                self.assertEqual(raised.exception.code, 2, argv)
        self.assertEqual(gate.parse(['--out', 'o', '--d3-root', 'b', '--chunk', '0', '--depth', '4', '--k-own', '3,2,1'])[1].k_own_ply, (3, 2, 1))
        missing = lambda name: None if name == 'gpu_engine.device_depthn_pricer' else object()
        with patch.object(importlib.util, 'find_spec', side_effect=missing), self.assertRaises(SystemExit) as raised:
            gate.main(['--out', str(out), '--d3-root', 'b', '--chunk', '0'])
        self.assertIn('device_depthn_pricer', str(raised.exception.code));self.assertFalse(out.exists())

    def test_offline_diagnostics_cli_on_an_empty_directory(self):
        empty = Path(tempfile.mkdtemp(prefix='depthn-diag-empty-'));out = Path(tempfile.gettempdir()) / f'depthn-diag-{uuid.uuid4().hex}'
        with patch.object(gate, 'log', lambda *a, **k: None):code = gate.main(['--out', str(out), '--diagnostics-from', str(empty)])
        self.assertEqual(code, 1);report = json.loads((out / 'diagnostics.json').read_text());self.assertEqual(report['calls'], 0)

    def test_frame_file_names(self):
        m = gate.FRAME_FILE.match('frames-block2-call13-searched-k4.npz');self.assertEqual((m['block'], m['call'], m['mode'], m['k']), ('2', '13', 'searched', '4'))
        self.assertEqual(gate.FRAME_FILE.match('frames-block0-call0-depth2.npz')['tag'], 'depth2');self.assertIsNone(gate.FRAME_FILE.match('arrays-block0-call0.npz'))


if __name__ == '__main__':unittest.main()
