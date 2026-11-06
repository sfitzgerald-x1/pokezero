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
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
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
# Mode 22's non-additive (a, r) term (x 100): row 0 has the best q and its minimum at r = 1, row 1 at r = 2, row 2 at r = 1, so
# the refutations of two kept learner moves are two different replies and the lowest-threat fill is only one of them.
INTERACTION = j.zeros((9, 9), j.int32).at[:3, :3].set(j.asarray(((9, 5, 8), (3, 4, 1), (2, 1, 4)), j.int32))
ENGINE_OF_LANE = (0, 1, 2, 3, 7, 8, 4, 5, 6)
HAVE_PRICER = all(importlib.util.find_spec(m) is not None for m in ('gpu_engine.device_depthn_pricer', 'gpu_engine.depthn_backups'))


def executor_n(*, sample=None):
    """Behaviours (b = mode % 100, shift = mode // 100; `node` = turn 1 + shift, `leaf` = turn 2 + shift):
    1/2/7 node terminal win/draw/bad winner; 3 invalid node step (1,1); 4 node encoder error; 5 non-finite learner value
    from the node on; 6 the opponent always waits; 8 node: opponent waits, owner forced to switch (4,5); 12 node: owner
    waits, opponent forced; 9 leaf-step terminal (a 0 loss, a 1 draw); 10 invalid leaf step a=2; 11 leaf encoder error;
    node, opponent's view only: 13 encoder error, 14 non-finite prior (requested), 15 non-finite prior while it waits,
    19 encoder error while it waits; 20 node terminal win with the learner's view in encoder error; 21 leaf-step terminal win
    with the learner's view in encoder error; 18 re-keys chance from the dispatched actions at every step; 22 adds a
    non-additive (a, r) term at every step, so a learner move's refutation is not the reply with the lowest threat over the
    kept moves. sample: replaces executor.sample (mutations)."""
    def split(mode):return mode % 100, mode // 100

    def legal(state):
        owner, (b, shift), turn = state.owner[0], split(state.mode[0]), state.turn[0]
        node = turn == 1 + shift
        own = j.where((b == 12) & node, False, j.where((b == 8) & node, j.asarray(SWITCH), j.asarray(BASE)))
        foe = j.where((b == 6) | (((b == 8) | (b == 15) | (b == 19)) & node), False, j.where((b == 12) & node, j.asarray(SWITCH), j.asarray(BASE)))
        rows = j.where(owner == 0, j.stack([own, foe]), j.stack([foe, own]))
        return j.where(state.done[0], False, rows)[None]

    def observe(state, public, owner, mask):
        (b, shift), peer = split(state.mode), owner != state.owner
        node, leaf = public.turn == 1 + shift, public.turn == 2 + shift
        error = j.where(((b == 4) & node) | ((b == 11) & leaf) | (((b == 13) | (b == 19)) & node & peer) | ((b == 20) & node & ~peer) | ((b == 21) & leaf & ~peer), 1, 0)
        value = (state.value + (owner - state.owner) * 10).astype(j.float32) * (1. / SCALE)
        value = j.where(((b == 5) & (public.turn >= 1 + shift)) | (((b == 14) | (b == 15)) & node & peer), j.nan, value)
        feature = j.stack((value, public.clock + j.zeros(1), state.own_act.astype(j.float32), state.peer_act.astype(j.float32),
            public.turn.astype(j.float32)), axis=-1)
        return Leaf(Observation(mask, error), {'numeric_features': feature[:, None, None, :]}, owner)

    def advance(state, public, actions, chance):
        owner = state.owner[0];a = actions[:, owner];r = actions[:, 1 - owner];(b, shift), turn = split(state.mode), state.turn + 1
        node, leaf = turn == 1 + shift, turn == 2 + shift
        value = state.value + a * 30 + r * 20 + chance + j.where(b == 22, INTERACTION[j.clip(a, 0, 8), j.clip(r, 0, 8)] * 100, 0)
        node_end = node & ((b == 1) | (b == 2) | (b == 7) | (b == 20));leaf_end = (leaf & (b == 9) & (a >= 0) & (a <= 1)) | (leaf & (b == 21))
        winner = j.where(node_end, j.where(b == 2, 2, j.where(b == 7, 9, state.owner)),
            j.where(leaf_end, j.where(b == 21, state.owner, j.where(a == 0, 1 - state.owner, 2)), state.winner))
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


def forward_n(params, inputs, model=None, **kw):
    """The depth-2 tests' fake head with NODE-DEPENDENT priors: lane (own_act + 2 * peer_act + turn) mod 9 of the prior gains
    0.5 (exact in float32), so a prior read from the wrong node, ply or seat shows; ties elsewhere stay ties."""
    out = forward(params, inputs, model, **kw);x = inputs['numeric_features'][:, 0, 0]
    lane = j.mod(x[:, 2] + 2. * x[:, 3] + x[:, 4], 9.)
    return out._replace(policy_logits=out.policy_logits + .5 * (j.arange(9, dtype=j.float32)[None, :] == lane[:, None]))


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


def fake_searched_widths(k_own, k_opp, *, extra=0):
    """The stand-in's ply-2 slot widths under `searched`: the documented searched_widths (extra > 0 makes a mutant)."""
    w_own, w_opp = gate.searched_widths_reference(k_own, k_opp);return w_own + extra, w_opp + extra


class OracleNPricer:
    """DepthNPricer's constructor and price() over the oracle's own surface; `executor`/`reference` overrides make mutants.
    As the pricer, under `searched` the ply-2 k's are searched_maps' and the slot widths are derived (fake_searched_widths)."""
    def __init__(self, executor, model, *, capacity=32, depth, k_own, k_opp, k1_own=9, k1_opp=9, block_lanes=2187, selection='prior',
                 digest=True, actor_capacity=3, fixed_capacity=False, reference_module=None, width_extra=0):
        self.oracle = reference.DepthNOracle(executor, model, chunk=64);self.capacity, self.block_lanes, self.selection = capacity, block_lanes, selection
        shape = gate.plan_of(depth, k_own, k_opp);self.k_own, self.k_opp, self.searched_k = shape['k_own'], shape['k_opp'], None
        if selection == 'searched':
            self.searched_k = (self.k_own[0], self.k_opp[0]);w_own, w_opp = fake_searched_widths(*self.searched_k, extra=width_extra)
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


def rule_maps(frame, k_own, k_opp, *, top1=True, errors=True, refutations=True, fill=True):
    """A second, test-side implementation of the documented `searched` rule (the stand-in pricer's), with switches that make
    the reviewer's mutants: no learner prior top-1, errored pairs not kept reachable, no refutations, no fill."""
    w_own, w_opp = gate.searched_widths_reference(k_own, k_opp);R = frame.pre.shape[0]
    own = np.full((R, 2, 9, 9, w_own), -1, np.int32);opp = np.full((R, 2, 9, 9, w_opp), -1, np.int32)
    first = lambda legal, logits: sorted((a for a in range(9) if legal[a]), key=lambda a: (-float(logits[a]), a))[0]
    unique = lambda xs: list(dict.fromkeys(int(x) for x in xs))
    for c in map(tuple, np.argwhere(np.asarray(frame.t1_status) == 2).tolist()):
        st, v = frame.t2_status[c], np.asarray(frame.t2_value[c], np.float64);ok = (st == 1) | (st == 4)
        bad = [tuple(x) for x in np.argwhere(st == 3).tolist()] if errors else []
        rows = [a for a in range(9) if (st[a] != 0).any() and ok[a].any()];cols = [r for r in range(9) if (st[:, r] != 0).any()]
        def q(a):   # pi_opp over ALL enumerated replies, renormalized over the row's valid ones (the documented formula)
            top = max(float(frame.t1_peer_logits[c][r]) for r in cols);x = {r: math.exp(float(frame.t1_peer_logits[c][r]) - top) for r in cols}
            total = math.fsum(x.values());live = [r for r in cols if ok[a, r]]
            return math.fsum(x[r] / total * v[a, r] for r in live) / math.fsum(x[r] / total for r in live)
        kept = unique(sorted(rows, key=lambda a: (-q(a), a))[:k_own] + ([first(frame.t1_own_legal[c], frame.t1_own_logits[c])] if top1 else [])) \
            if frame.t1_own_requested[c] else [0]
        learner = unique([a for a, _ in bad] + kept)[:w_own]   # errored rows in front, outside the k count; refutations use `kept`
        if frame.t1_peer_requested[c]:
            kept = [a for a in kept if ok[a].any()]
            ranked = unique([first(frame.t1_peer_legal[c], frame.t1_peer_logits[c])] + ([min((r for r in cols if ok[a, r]), key=lambda r: (v[a, r], r)) for a in kept] if refutations else []))
            threat = {r: min(v[a, r] for a in kept if ok[a, r]) for r in cols if any(ok[a, r] for a in kept)}
            for r in sorted(threat, key=lambda r: (threat[r], r)) if fill else ():
                if len(ranked) < k_opp + 1 and r not in ranked:ranked.append(r)
            opponent = unique([r for _, r in bad] + ranked)[:w_opp]
        else:opponent = [0]
        own[c][:len(learner)] = learner;opp[c][:len(opponent)] = opponent
    return own, opp, np.asarray([bool((frame.pre[q] != 0).any() or (frame.t1_status[q] == 3).any() or (frame.t2_status[q] == 3).any()) for q in range(R)])


def fake_searched_maps(frame, k_own, k_opp):
    return rule_maps(frame, k_own, k_opp)


FAKE_MODULE = SimpleNamespace(DepthNPricer=OracleNPricer, as_depth2_frame=fake_as_depth2, searched_maps=fake_searched_maps)


# depthn_backups' call shapes while the real module lands with the pricer: its node_value (masked softmax of each seat's prior over its
# map, fsum rows, strict max, clip, terminals exact, any error refuses, the `rule` hook) with the child-read lines the gate's backup-path
# mutants patch (gate.BACKUP_PATCH) kept verbatim, so --mutation-test runs on this tree too; a root action's Q is the mean over usable
# worlds and enumerated replies, the target softmax(root logits + Q) over legal actions.
FAKE_BACKUPS_SOURCE = '''
import math
from types import SimpleNamespace

import numpy as np


def _softmax(logits,actions):
    top=max(float(logits[a])for a in actions)
    weights=[math.exp(float(logits[a])-top)for a in actions]
    total=math.fsum(weights)
    return [w/total for w in weights]


def learner_max(grid,pi_own,pi_opp):
    best=None
    for row in grid:
        q=math.fsum(p*v for p,v in zip(pi_opp,row))
        if best is None or q>best:best=q
    return best


def _node(frame,root,world,ply,node):
    if ply==1:
        a,r=divmod(int(node),9);at=lambda x:x[root,world,a,r]
        return (int(at(frame.t1_status)),at(frame.t1_value),at(frame.t1_own_actions),at(frame.t1_opp_actions),at(frame.t1_own_logits),at(frame.t1_peer_logits))
    p=frame.nodes[ply-2];at=lambda x:x[root,world,node]
    return int(at(p.status)),at(p.value),at(p.own_actions),at(p.opp_actions),at(p.own_logits),at(p.opp_logits)


def node_value(frame,root,world,ply,node,rule=learner_max):
    status,value,own_actions,opp_actions,own_logits,opp_logits=_node(frame,root,world,ply,node)
    if status==1:return float(value)
    if status!=2:return None
    k_own,k_opp=frame.k_own[ply-1],frame.k_opp[ply-1];K=k_own*k_opp;leaf=ply+1==frame.depth
    own_slots=[i for i in range(k_own)if own_actions[i]>=0];opp_slots=[i for i in range(k_opp)if opp_actions[i]>=0]
    if not own_slots or not opp_slots:return None
    grid=[]
    for i in own_slots:
        row=[]
        for x in opp_slots:
            if leaf:
                s=int(frame.leaf_status[root,world,node,i,x])
                v=float(frame.leaf_value[root,world,node,i,x])if s in(1,4)else None
            else:v=node_value(frame,root,world,ply+1,node*K+i*k_opp+x,rule)
            if v is None:return None
            row.append(v)
        grid.append(row)
    pi_own=_softmax(own_logits,[int(own_actions[i])for i in own_slots]);pi_opp=_softmax(opp_logits,[int(opp_actions[x])for x in opp_slots])
    return min(1.,max(-1.,rule(grid,pi_own,pi_opp)))


def learner_max_root(frame,q,logits,legal):
    if (np.asarray(frame.pre[q])!=0).any():return SimpleNamespace(status='REFUSED',improvement=None)
    values=[]
    for a1 in range(9):
        cells=[node_value(frame,q,w,1,a1*9+r1)for w in(0,1)for r1 in range(9)if frame.t1_status[q,w,a1,r1]]if legal[a1]else[]
        if legal[a1]and(not cells or any(c is None for c in cells)):return SimpleNamespace(status='REFUSED',improvement=None)
        values.append(float(np.mean(cells))if legal[a1]else None)
    scores=np.where(legal,np.asarray(logits)+np.asarray([v or 0. for v in values]),-np.inf);target=np.exp(scores-scores.max())
    return SimpleNamespace(status='LABELED',improvement=SimpleNamespace(completed_q=tuple(values),policy_target=tuple(target/target.sum())))
'''


def fake_backups(name='fake_depthn_backups'):
    module = types.ModuleType(name);exec(compile(FAKE_BACKUPS_SOURCE, f'<{name}>', 'exec'), module.__dict__);return module


FakeBackupsN = fake_backups()


# ---------------------------------------------------------------- fixture

class FixtureN:
    """Facts-based requests over the fake engine, the real FramePricer and the gate's own oracle, shared per class."""
    MODES = ((0, 0), (18, 1), (101, 0), (102, 1), (103, 0), (104, 1), (105, 0), (0, 1), (108, 0), (109, 1), (110, 0), (111, 1), (112, 0),
        (113, 1), (114, 0), (115, 1), (9, 0), (6, 1), (13, 0), (12, 1), (19, 0), (119, 1), (20, 0), (120, 1), (121, 0), (22, 1), (122, 0))
    ROOTS = len(MODES)
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
        with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=forward_n):
            yield

    def frame_pricer(self):
        return batch.FramePricer(self.ex, object(), capacity=32, block_rows=32, actor_capacity=3, vectorized=True, host_prep=True)

    def pricers(self, *, module=FAKE_MODULE, depth=3, k_own=2, k_opp=2, selection='prior', lanes=(2187,), executor=None, k1_own=9, k1_opp=9):
        return gate.DepthNPricers(module, executor or self.ex, object(), depth=depth, k_own=k_own, k_opp=k_opp, selection=selection, capacity=32,
            actor_capacity=3, lanes=lanes, k1_own=k1_own, k1_opp=k1_opp)

    def call(self, pricers, *, sample=None, decision=None, backups=FakeBackupsN, save=None, depth2=True, requests=None):
        module = pricers.module
        depth2_pricer = gate.depth2_pass_pricer(module, self.ex, object(), capacity=32, actor_capacity=3, k1_own=pricers.k1_own, k1_opp=pricers.k1_opp,
            digest=True) if depth2 else None   # as the gate prices it
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
        self.assertLess(summary['refusals']['depthn']['refused_count'], self.ROOTS)
        evaluation, summary = self.call(self.pricers(k1_own=2, k1_opp=2))
        result, (status, reasons) = self.judged(evaluation)   # k1_own = 2 prunes a legal root action of every root: all refuse
        self.assertEqual((status, summary['refusals']['depthn']['refused_count']), ('INCOMPLETE', self.ROOTS));self.assertIn('vacuous', reasons[0])

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

    def test_searched_pricers_take_ks_and_derive_the_documented_widths(self):
        pricers = self.pricers(selection='searched');evaluation, summary = self.call(pricers)
        result, (status, _) = self.judged(evaluation)
        self.assertEqual(status, 'PASS', failing(result))
        self.assertEqual([(p.searched_k, p.k_own, p.k_opp) for p in pricers.instances], [((2, 2), (3, 2), (4, 2))])   # built with k's
        self.assertEqual(summary['plan']['k_own'], (3, 2));self.assertEqual(summary['plan']['k_opp'], (4, 2))   # searched_widths_reference(2, 2)
        self.assertEqual(summary['searched'], dict(searched_k=[2, 2], searched_widths=[3, 4]))
        flagged = int(evaluation['searched_maps']['extras'][0]['set']);self.assertGreater(flagged, 0)
        self.assertEqual(summary['searched_maps']['widths'], dict(own=3, opp=4));self.assertEqual(summary['searched_maps']['rule_mismatches'],
            dict(searched_maps_widths=0, searched_maps_own_rule=0, searched_maps_opp_rule=0, searched_depth2_refused=0))
        self.assertEqual(pricers.options()['searched_k'], [2, 2]);self.assertEqual(pricers.options()['slot_widths']['k_own'], [3, 2])
        self.assertEqual(result['checks']['a']['mismatched']['engine_widths'], 0)
        self.assertEqual(gate.pad_map([4, 1], 4).tolist(), [4, 1, -1, -1])
        with self.assertRaises(ValueError):gate.pad_map([1, 2, 3], 2)
        self.assertGreater(result['checks']['a']['compared']['searched_maps_own_rule'], 0)
        self.assertGreater(result['checks']['a']['compared']['searched_maps_own'], 0)
        self.assertEqual(summary['timing'].keys() >= {'searched_maps_seconds'}, True)
        def wider(frame, k_own, k_opp):
            own, opp, flag = fake_searched_maps(frame, k_own, k_opp);return np.concatenate([own, np.full(own.shape[:-1] + (2,), -1, np.int32)], axis=-1), opp, flag
        with self.assertRaises(ValueError):self.call(self.pricers(module=SimpleNamespace(DepthNPricer=OracleNPricer, as_depth2_frame=fake_as_depth2, searched_maps=wider), selection='searched'))
        def duplicate(frame, k_own, k_opp):
            own, opp, flag = fake_searched_maps(frame, k_own, k_opp);cell = tuple(np.argwhere(np.asarray(frame.t1_status) == 2)[0]);own[cell][1] = own[cell][0]
            return own, opp, flag
        module = SimpleNamespace(DepthNPricer=OracleNPricer, as_depth2_frame=fake_as_depth2, searched_maps=duplicate)
        evaluation, _ = self.call(self.pricers(module=module, selection='searched'))
        self.assertGreater(evaluation['a']['mismatched']['searched_maps_own'], 0)

    def test_searched_rule_mutants_are_caught(self):
        """The reviewer's searched_maps mutants (the stand-in pricer's own rule with a part removed) fail (a) EXACT against the
        gate's re-derivation; the intact rule passes. Mode 22's non-additive values separate refutations from the fill."""
        for k in (1, 2):
            with self.subTest(k=k):
                evaluation, _ = self.call(self.pricers(selection='searched', k_own=k, k_opp=k), decision=False)
                self.assertTrue(evaluation['a']['passed'], evaluation['a']['first_mismatches'][:3])
        for name, switch, k in (('no_prior_top1', dict(top1=False), 1), ('errors_not_kept', dict(errors=False), 2), ('no_refutations', dict(refutations=False), 1),
                ('no_fill', dict(fill=False), 2)):
            with self.subTest(name):
                module = SimpleNamespace(DepthNPricer=OracleNPricer, as_depth2_frame=fake_as_depth2, searched_maps=lambda f, ko, kp, sw=switch: rule_maps(f, ko, kp, **sw))
                evaluation, _ = self.call(self.pricers(module=module, selection='searched', k_own=k, k_opp=k), decision=False)
                mismatched = evaluation['a']['mismatched'];self.assertGreater(mismatched['searched_maps_own_rule'] + mismatched['searched_maps_opp_rule'], 0, name)
                self.assertEqual(mismatched.get('ply2_maps', 0), 0)   # the pricer expanded the maps it was given: only the rule check sees it

    def test_width_clamping_engines_fail(self):
        """Engines that price a k = 3 request at k = 2: honestly reporting 2, reporting the requested 3 over width-2 arrays,
        padding width-3 arrays with only 2 live slots, and a searched engine deriving the wrong ply-2 widths: every one FAILs."""
        def clamp(legal, logits, requested, k):
            m = np.full(k, -1, np.int32);m[:min(k, 2)] = reference.prior_map(legal, logits, requested, min(k, 2));return m
        padded = SimpleNamespace(**{k: getattr(reference, k) for k in dir(reference) if not k.startswith('__')});padded.prior_map = clamp
        def factory(kind):
            def make(executor, model, **kw):
                if kind == 'padded':return OracleNPricer(executor, model, **dict(kw, reference_module=padded))
                if kind == 'searched':return OracleNPricer(executor, model, **dict(kw, width_extra=1))
                engine = OracleNPricer(executor, model, **dict(kw, k_own=2, k_opp=2))
                if kind == 'lying':engine.k_own, engine.k_opp = gate.plan_of(kw['depth'], kw['k_own'], kw['k_opp'])['k_own'], gate.plan_of(kw['depth'], kw['k_own'], kw['k_opp'])['k_opp']
                return engine
            return SimpleNamespace(DepthNPricer=make, as_depth2_frame=fake_as_depth2, searched_maps=fake_searched_maps)
        for kind, selection in (('honest', 'prior'), ('lying', 'prior'), ('padded', 'prior'), ('searched', 'searched')):
            with self.subTest(kind):
                evaluation, summary = self.call(self.pricers(module=factory(kind), k_own=3, k_opp=3, selection=selection), decision=False)
                result, (status, _) = self.judged(evaluation);self.assertEqual(status, 'FAIL', kind)
                if kind in ('honest', 'searched'):self.assertGreater(evaluation['a']['mismatched']['engine_widths'], 0)
                if kind in ('honest', 'lying', 'searched'):self.assertTrue(evaluation['geometry'], kind)
                if kind == 'padded':self.assertGreater(evaluation['a']['mismatched']['ply2_maps'], 0);self.assertGreater(evaluation['d']['mismatched']['leaf_missing_from_pricer'], 0)

    def test_the_depth2_pass_is_rechecked_where_cheap(self):
        _, summary = self.prior();counts = summary['depth2_pass']
        self.assertGreater(counts['rerun'], counts['forced']);self.assertGreater(counts['forced'], 0);self.assertGreater(counts['enumerated'], counts['rerun'])
        def broken(frame):
            out = fake_as_depth2(frame);cell = tuple(np.argwhere(np.asarray(out.t2_status) == 1)[0]);out.t2_value[cell] = -out.t2_value[cell] + .5
            extra = tuple(np.argwhere(np.asarray(out.t2_status) == 0)[0]);out.t2_status[extra] = 4;return out
        module = SimpleNamespace(DepthNPricer=OracleNPricer, as_depth2_frame=broken, searched_maps=fake_searched_maps)
        evaluation, _ = self.call(self.pricers(module=module), decision=False);mismatched = evaluation['a']['mismatched']
        self.assertGreater(mismatched['depth2_pass_t2_value_terminal'], 0);self.assertGreater(mismatched['depth2_pass_leaf_extra'], 0)
        self.assertEqual(mismatched['depth2_pass_t1_status'], 0)

    def test_fixture_gaps_waiting_view_errors_terminal_errors_and_node_priors(self):
        _, summary = self.prior()
        with self.engine():
            stack = reference.stack(self.requests, batch._numpy_worlds(self.requests, self.tables, self.recipe))
            data = gate.oracle_levels(oracle=self.oracle, stack=stack, requests=self.requests, recipe=self.recipe, params=PARAMS, frame=None, plan=gate.plan_of(3, 2, 2))
        root = {mode: i for i, (mode, _) in enumerate(self.MODES)};children = data['children']
        rows = lambda q: [i for i, c in enumerate(children['cells'].tolist()) if c[0] == q]
        waiting = rows(root[12]);self.assertTrue(waiting and all(not children['own_requested'][i] for i in waiting if children['status'][i] == 2))   # learner waits at turn 1
        self.assertTrue(all(data['maps'][(2, c[0], c[1], c[2] * 9 + c[3])][0].tolist() == [0, -1] for c in children['cells'][waiting].tolist() if (2, c[0], c[1], c[2] * 9 + c[3]) in data['maps']))
        self.assertEqual({int(children['status'][i]) for i in rows(root[19])}, {2})   # the waiting opponent's view error never refuses
        self.assertEqual({(int(children['status'][i]), int(children['error'][i])) for i in rows(root[20])}, {(1, 0)})   # terminal before encoder error
        ply2 = data['levels'][2];at = lambda q: [i for i, spec in enumerate(ply2['specs'].tolist()) if spec[0] == q]
        self.assertEqual({int(ply2['status'][i]) for i in at(root[119])}, {2});self.assertEqual({(int(ply2['status'][i]), int(ply2['error'][i])) for i in at(root[120])}, {(1, 0)})
        leaves = data['leaves'];self.assertEqual({(int(s), int(e)) for (q, *_), s, e in zip(leaves['index'].tolist(), leaves['status'], leaves['error']) if q == root[121]}, {(1, 0)})
        # node-dependent priors: siblings' priors differ, so a prior read from the wrong node fails (c)
        frame = self._frame();level = frame.nodes[0];expanded = np.argwhere(np.asarray(level.status) == 2)
        lanes = lambda x: np.asarray(level.own_legal[x], bool) & bool(level.own_requested[x])
        pairs = [(tuple(x), tuple(y)) for x, y in zip(expanded[:-1], expanded[1:]) if tuple(x[:2]) == tuple(y[:2])
            and not np.array_equal(np.asarray(level.own_logits[tuple(x)])[lanes(tuple(x))], np.asarray(level.own_logits[tuple(y)])[lanes(tuple(x))])]
        self.assertTrue(pairs);wrong = copy.deepcopy(frame);wrong.nodes[0].own_logits[pairs[0][0]] = level.own_logits[pairs[0][1]]
        ev = gate.evaluate_n(wrong, self._d(), self.requests, self._data(wrong), context={})
        self.assertFalse(gate.summarize_n([ev])['checks']['c']['quantities']['ply2_own_logits']['passed'])

    def test_topk_boundary_per_ply(self):
        _, summary = self.prior();merged = gate.merge_topk_n([summary['topk']])
        self.assertEqual({k for k in merged if k.startswith('ply')}, {'ply2', 'ply3'})
        for ply in ('ply2', 'ply3'):
            self.assertGreater(merged[ply]['own']['selections'], 0);self.assertEqual(merged[ply]['own']['oracle_prior_set_differs'], 0)   # the same fake heads
            self.assertIsNotNone(merged[ply]['own']['gap']['p50'])
        self.assertIsNone(gate.merge_topk_n([None, {}]))

    def test_node_spreads_and_guard_spread(self):
        status, value = np.zeros((1, 2, 81, 2, 2), np.int8), np.zeros((1, 2, 81, 2, 2), np.float32)
        status[0, 0, 0] = 4;value[0, 0, 0] = ((.5, .1), (.2, .2))           # rows .3 and .2 under a uniform opponent: spread .1
        status[0, 0, 1] = 4;value[0, 0, 1] = ((.4, .4), (.4, .4))           # identical rows: spread 0
        status[0, 0, 2, 0] = 4                                              # one own slot: skipped
        status[0, 1, 3] = 4;status[0, 1, 3, 1, 1] = 3                       # an errored leaf: skipped
        opp = np.full((1, 2, 9, 9, 2), -1, np.int32);opp[0, :, 0, :3] = (0, 1)
        frame = SimpleNamespace(depth=2, leaf_status=status, leaf_value=value, t1_opp_actions=opp, t1_peer_logits=np.zeros((1, 2, 9, 9, 9), np.float32))
        spreads = gate.node_spreads(frame);self.assertEqual(len(spreads), 2);self.assertAlmostEqual(sorted(spreads)[1], .1, places=6);self.assertEqual(sorted(spreads)[0], 0.)
        p10, info = gate.guard_spread([0., .1, .2, 0.]);self.assertAlmostEqual(p10, .11);self.assertEqual((info['nonzero'], info['zero_share']), (2, .5))
        p10, info = gate.guard_spread([0., 0.]);self.assertIsNone(p10)
        rule = gate.guard_note(gate.g2.tolerance_rule(None, p10), info);self.assertIn('every one of the 2', rule['guard']);self.assertIn('NON-ZERO', rule['guard_spread_rule'])
        self.assertIn('no expanded last-ply parent', gate.guard_note(gate.g2.tolerance_rule(None, None), gate.guard_spread([])[1])['guard'])

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
        pvs = report['prior_vs_searched']['k2'];self.assertGreater(pvs['cells'], 0);self.assertIn('prior_set_not_subset_of_searched', pvs)
        self.assertIn('prior_set_not_subset_of_searched_share', pvs)
        pruned = report['pruned_errors']['prior-k2'];self.assertEqual(pruned['roots'], self.ROOTS);self.assertGreater(pruned['with_depth2_errors'], 0)
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

    def test_n2_status_incomplete_and_world_reports_gated(self):
        frame = fake_as_depth2(self.cached('n2', lambda: self._n2_frame()));found, floats = gate.n2_compare(frame, frame)
        result = lambda found, e: dict(a=found, floats=floats, c={}, e=e, spreads=[.1])
        e = gate.decide_roots(frame, frame, self.requests, depth2_backups)
        self.assertEqual(gate.summarize_n2({'x': [result(found, e)]})['x']['status'], 'PASS')
        empty = dict(e, roots_compared=0, gaps=[]);self.assertEqual(gate.summarize_n2({'x': [result(found, empty)]})['x']['status'], 'INCOMPLETE')
        other = copy.deepcopy(frame);other.world_reports = ((dict(reason='x'), dict()),);found, _ = gate.n2_compare(frame, other)
        report = gate.summarize_n2({'x': [result(found, e)]})['x'];self.assertEqual(report['status'], 'FAIL');self.assertEqual(report['world_reports_first_difference']['root'], 0)

    def test_turn2_choices_break_ties_by_the_lower_action(self):
        frame = SimpleNamespace(t1_status=np.zeros((1, 2, 9, 9), np.int8), t1_own_actions=np.full((1, 2, 9, 9, 2), -1, np.int32),
            t1_opp_actions=np.full((1, 2, 9, 9, 1), -1, np.int32), t1_peer_logits=np.zeros((1, 2, 9, 9, 9), np.float32))
        frame.t1_status[0, 0, 0, 0] = 2;frame.t1_own_actions[0, 0, 0, 0] = (2, 1);frame.t1_opp_actions[0, 0, 0, 0] = (0,)
        self.assertEqual(gate.turn2_choices(frame, SimpleNamespace(node_value=lambda *a: .25)), {(0, 0, 0, 0): 1})   # slot 1 holds the lower action

    def _n2_frame(self):
        with self.engine():return OracleNPricer(self.ex, object(), depth=2, k_own=9, k_opp=9).price(self.requests, self.tables, self.recipe, PARAMS)


# ---------------------------------------------------------------- GUARD_DEPTHN_RULE and --reverdict

FLOOR, SPREAD = .004, .02   # value bound 2 x FLOOR = .008 and guard 0.25 x SPREAD = .005: noise .006 lies between them
# Optional real gate.json files (os.pathsep-separated), each expected to PASS under the current rules: unset, only the synthetic cases run.
SAVED_REPORTS = [Path(x) for x in os.environ.get('V2E_DEPTHN_SAVED_GATE_REPORTS', '').split(os.pathsep) if x]


def saved_gate(depth=3, ply2_noise=.004, t1_logits=.05, ply2_logits=None, single_actor=True):
    """A minimal gate.json in the format before GUARD_DEPTHN_RULE and DEEPER_LOGIT_RULE: the leaf values inside the value bound (.008)
    but beyond the guard (.005), every digest compared and matched, no guard_mode or per-position evidence; ply2_logits: a ply-2 prior
    max |diff| judged against LOGIT_BOUND, with its first_beyond rows; single_actor: the head override and actor_capacity 1 (else the
    bank actor with actor_capacity 3, which cannot establish one parameter set)."""
    def q(compared, noise, guard=True, bound=.008):
        within = noise < .005
        return dict(compared=compared, nonfinite=0, max_abs_difference=noise, bound=bound, within_bound=noise <= bound, ratio_to_bound=noise / bound,
            beyond_bound=int(noise > bound), first_beyond=[dict(cell=[1, 0, 7, 7], abs_difference=noise)] if noise > bound else [],
            **(dict(guard_bound=.005, within_guard=within, ratio_to_guard=noise / .005) if guard else {}), passed=noise <= bound and (within or not guard), gates='c')
    quantities = dict(t1_value_head=q(4, .004), t1_own_logits=q(30, t1_logits, guard=False, bound=.1), depth1_noise_floor=dict(compared=4, max_abs_difference=FLOOR, passed=None),
        ply2_value_head=q(6, ply2_noise), leaf_value=q(10, .006), depth2_pass_t2_value=q(3, .001), depth2_pass_t1_own_logits=q(30, 0., guard=False, bound=.1))
    if ply2_logits is not None:quantities['ply2_own_logits'] = q(40, ply2_logits, guard=False, bound=.1)
    a = dict(passed=True, compared=dict(leaf_status=12, leaf_digest=10, ply2_digest_own=6), mismatched=dict(leaf_status=0, leaf_digest=0, ply2_digest_own=0), first_mismatches=[])
    table = [dict(check='c', quantity=name, max_abs_difference=m['max_abs_difference'], ratio_to_bound=m['ratio_to_bound'], ratio_to_guard=m.get('ratio_to_guard'),
        guard_bound=m.get('guard_bound'), passed=m['passed']) for name, m in quantities.items() if name != 'depth1_noise_floor']
    return dict(schema=gate.SCHEMA, mode='SAVED_ROOTS', status='FAIL', passed=False, status_reasons=[], sample_leaves=None, evaluator=dict(override=single_actor),
        pricer_options=dict(depth=depth, actor_capacity=1 if single_actor else 3),
        tolerance=dict(value_bound=.008, guard_bound=.005, table=table), checks=dict(a=a, c=dict(passed=False, quantities=quantities), e=dict(passed=True, roots_compared=2),
            f=saved_choices()))


def saved_choices(differing=()):
    """A saved (f) section as the live run writes it (value bound .008): `differing` flips as (gap_pricer, gap_oracle) pairs."""
    flips = [dict(block=0, call=0, root=1, world=0, ply=2, node=7 + i, address=[1, 2, 0], kind='learner', own_action=None, actions=[1, 2], gap_pricer=gp, gap_oracle=go)
        for i, (gp, go) in enumerate(differing)]
    return dict(passed=True, status='COMPARED', nodes=5, nodes_by_ply={'1': 2, '2': 3}, comparisons=12, learner_comparisons=5, opponent_comparisons=7, not_comparable=0,
        both_unpriced=0, one_sided=0, one_sided_examples=[], problems=0, problem_examples=[], differing=flips, differing_complete=True, max_abs_delta_row_q=.003,
        max_abs_delta_child_value=.006, sampled=False, value_bound=.008, near_ties=2, near_tie_flips=0, non_exempt_mismatches=0, rule=gate.CHOICE_RULE)


def reverdicted(report, *, name='gate.json'):
    """(exit code, gate-reverdict.json, log lines) of --reverdict on `report` written to a new directory (a dict) or copied (a path)."""
    directory = Path(tempfile.mkdtemp(prefix='depthn-reverdict-'));path = directory / name;lines = []
    if isinstance(report, dict):path.write_text(json.dumps(report))
    else:shutil.copyfile(report, path)
    with patch.object(gate, 'log', lambda message, **values: lines.append(dict(values, message=message))):code = gate.main(['--reverdict', str(path)])
    return code, json.loads((directory / gate.REVERDICT_FILE).read_text()), lines, path


class GuardDepthNTests(FixtureN, unittest.TestCase):
    """GUARD_DEPTHN_RULE and DEEPER_LOGIT_RULE on the stand-in depth-3 surface with planted noise, judged under a synthetic depth-1
    floor and spreads (value bound .008, guard .005), the N=2 equivalence path unchanged, and --reverdict on saved reports."""
    _frame, _d, _data, _n2_frame = GateNTests._frame, GateNTests._d, GateNTests._data, GateNTests._n2_frame

    def evaluation(self, *, leaf=0., node=0., t1=0., ply_logits=0., t1_logits=0., t1_peer_logits=0., change=None, data_change=None, edit=None, floor=FLOOR,
                   spread=SPREAD, actors=1):
        """One call's evaluation of the stand-in frame with `leaf` added to head-valued leaves, `node` to expanded ply-2 nodes, `t1` to
        expanded turn-1 cells, `ply_logits` to every lane of the ply-2 nodes' own prior and `t1_logits` / `t1_peer_logits` to the turn-1
        cells' own / opponent prior (a constant shift: rankings unchanged), then change(frame) and data_change(oracle data, frame);
        `actors` stands for run_call_n's len(params) (the fixture's PARAMS has two), then edit(evaluation)."""
        base = self.cached('frame', self._frame);data = self.cached('data', lambda: self._data(base));frame = copy.deepcopy(base);level = frame.nodes[0]
        frame.leaf_value[np.asarray(frame.leaf_status) == 4] += np.float32(leaf);level.value[np.asarray(level.status) == 2] += np.float32(node)
        frame.t1_value[np.asarray(frame.t1_status) == 2] += np.float32(t1)
        level.own_logits[np.asarray(level.status) != 0] += np.float32(ply_logits);frame.t1_own_logits[np.asarray(frame.t1_status) != 0] += np.float32(t1_logits)
        frame.t1_peer_logits[np.asarray(frame.t1_status) != 0] += np.float32(t1_peer_logits)
        if change is not None:change(frame)
        if data_change is not None:data = copy.deepcopy(data);data_change(data, frame)
        ev = gate.evaluate_n(frame, self._d(), self.requests, data, context={})
        ev['c']['depth1_noise_floor'] = gate.g2._numeric([[0]], [floor], [0.]);ev['spreads'] = [spread] * 10;ev['actors'] = actors
        if edit is not None:edit(ev)
        return ev

    def judge(self, **kw):
        """(result, status, (c) quantities) of one evaluation(**kw) through summarize_run."""
        result, (status, _) = self.judged(self.evaluation(**kw));return result, status, result['checks']['c']['quantities']

    def test_noise_between_guard_and_bound_past_turn1_is_digest_certified(self):
        result, status, c = self.judge(leaf=.006, node=.006)
        self.assertEqual(status, 'PASS', failing(result));self.assertEqual(result['tolerance']['value_bound'], 2 * float(np.float32(FLOOR)))
        for name in ('leaf_value', 'ply2_value_head'):
            m = c[name];self.assertEqual((m['within_bound'], m['within_guard'], m['passed'], m['guard_mode']), (True, False, True, 'digest_certified'), name)
            self.assertGreater(m['beyond_guard'], 0);self.assertGreater(m['ratio_to_guard'], 1.);self.assertEqual(m['guard_evidence']['value_positions_outside_digest'], 0)
            self.assertGreaterEqual(m['guard_evidence']['digest_compared'], m['compared']);self.assertEqual(m['guard_evidence']['digest_mismatched'], 0)
        self.assertEqual(c['leaf_value']['beyond_guard'], c['leaf_value']['compared']);self.assertEqual(c['leaf_value']['guard_evidence']['digest'], 'leaf_digest')
        self.assertEqual(c['ply2_value_head']['guard_evidence']['digest'], 'ply2_digest_own')
        self.assertEqual((c['leaf_value']['guard_evidence']['evaluator_actors'], c['leaf_value']['guard_evidence']['evaluator_actors_source']), (1, gate.ACTORS_LIVE))
        self.assertEqual(result['tolerance']['evaluator_actors'], dict(count=1, source=gate.ACTORS_LIVE))
        self.assertEqual((c['t1_value_head']['guard_mode'], c['t1_value_head']['beyond_guard']), ('gated', 0));self.assertNotIn('guard_mode', c['ply2_own_logits'])
        tolerance = result['tolerance'];self.assertEqual(tolerance['guard_depthn_rule'], gate.GUARD_DEPTHN_RULE)
        self.assertEqual({k: v for k, v in tolerance['guard_modes'].items() if v == 'digest_certified'}, dict(leaf_value='digest_certified', ply2_value_head='digest_certified'))
        row = next(r for r in tolerance['table'] if r['quantity'] == 'leaf_value');self.assertEqual((row['passed'], row['guard_mode']), (True, 'digest_certified'))
        entry = next(r for r in gate.tolerance_line_n(tolerance)['checks'] if r['quantity'] == 'leaf_value')
        self.assertEqual((entry['passed'], entry['guard_mode'], entry['beyond_guard']), (True, 'digest_certified', c['leaf_value']['compared']))

    def test_a_digest_mismatch_or_incomplete_coverage_keeps_the_guard_gated(self):
        def flip(frame):
            at = tuple(np.argwhere(np.asarray(frame.leaf_status) == 4)[0]);frame.leaf_digest[at + (0,)] ^= np.uint32(1)
        def fewer(ev):ev['a']['compared']['leaf_digest'] = ev['c']['leaf_value']['compared'] - 1
        cases = dict(leaf_digest_mismatch=(dict(change=flip), 'leaf_value'), fewer_leaf_digests_than_values=(dict(edit=fewer), 'leaf_value'),
            leaf_value_outside_the_digest_mask=(dict(edit=lambda ev: ev['digest_cover'].__setitem__('leaf_value', 1)), 'leaf_value'),
            node_value_outside_the_digest_mask=(dict(edit=lambda ev: ev['digest_cover'].__setitem__('ply2_value_head', 1)), 'ply2_value_head'),
            node_coverage_not_counted=(dict(edit=lambda ev: ev['digest_cover'].pop('ply2_value_head')), 'ply2_value_head'),
            depth2_frame=(dict(edit=lambda ev: ev.__setitem__('depth', 2)), 'leaf_value'),
            two_evaluator_actors=(dict(actors=2), 'leaf_value'), evaluator_actors_not_recorded=(dict(edit=lambda ev: ev.pop('actors')), 'ply2_value_head'))
        for case, (kw, name) in cases.items():
            with self.subTest(case):
                result, status, c = self.judge(leaf=.006, node=.006, **kw);m = c[name]
                self.assertEqual(status, 'FAIL');self.assertEqual((m['guard_mode'], m['within_bound'], m['passed']), ('gated', True, False))
                if 'actors' in case:self.assertEqual(m['guard_evidence']['evaluator_actors'], 2 if case == 'two_evaluator_actors' else None)
                self.assertFalse(result['checks']['c']['passed'])
                if case != 'leaf_digest_mismatch':self.assertEqual(set(failing(result)), {'c'})   # the gated guard alone fails the run
                else:self.assertEqual(result['checks']['a']['mismatched']['leaf_digest'], 1)

    def test_digest_cover_is_summed_across_calls(self):
        """Two calls through summarize_run: the per-position counts add up (keeping one call's count would certify (1, 0) or
        (0, 1)), and a call without a count makes the merged count unknown: gated for nodes, by construction for leaves."""
        one = self.evaluation(leaf=.006, node=.006)
        def run(leaf, node):
            calls = [copy.deepcopy(one), copy.deepcopy(one)]
            for ev, x, y in zip(calls, leaf, node):
                for name, n in (('leaf_value', x), ('ply2_value_head', y)):
                    if n is None:ev['digest_cover'].pop(name)
                    else:ev['digest_cover'][name] = n
            result, (status, _) = self.judged(*calls);c = result['checks']['c']['quantities']
            return status, {name: (c[name]['guard_mode'], c[name]['guard_evidence']['value_positions_outside_digest']) for name in ('leaf_value', 'ply2_value_head')}
        self.assertEqual(run((0, 0), (0, 0)), ('PASS', dict(leaf_value=('digest_certified', 0), ply2_value_head=('digest_certified', 0))))
        for leaf, node, want in (((1, 0), (0, 1), (1, 1)), ((0, 1), (1, 2), (1, 3))):
            with self.subTest(leaf=leaf, node=node):
                self.assertEqual(run(leaf, node), ('FAIL', dict(leaf_value=('gated', want[0]), ply2_value_head=('gated', want[1]))))
        self.assertEqual(run((None, 0), (0, None)), ('FAIL', dict(leaf_value=('digest_certified', None), ply2_value_head=('gated', None))))
        self.assertEqual(gate.digest_cover_merged([dict(c=dict(x={}), digest_cover=dict(x=2)), dict(c=dict(x={}), digest_cover=dict(x=3))]), dict(x=5))

    def test_a_node_valued_outside_the_digest_mask_keeps_the_guard_gated(self):
        """An oracle ply-2 node record with code 5 on a TERMINAL node (status 1), matched by the pricer: (c) values it (codes 5/6),
        (a)'s _viewed_own does not digest it (codes 5/6 count only under status 3; under status 2 the node is inside the mask anyway),
        so its per-position count is 1 and the ply-2 guard stays gated."""
        def plant(data, frame):
            records = data['levels'][2];i = int(np.flatnonzero(records['status'] == 1)[0]);q, w = map(int, records['specs'][i][:2]);node = int(data['ids'][2][i])
            records['error'][i] = 5;frame.nodes[0].error[q, w, node] = 5;records['head_value'][i] = frame.nodes[0].value[q, w, node]
        result, status, c = self.judge(leaf=.006, node=.006, data_change=plant);m = c['ply2_value_head']
        self.assertEqual(status, 'FAIL');self.assertEqual(set(failing(result)), {'c'})
        self.assertEqual((m['guard_mode'], m['guard_evidence']['value_positions_outside_digest'], m['passed']), ('gated', 1, False))
        self.assertEqual(c['leaf_value']['guard_mode'], 'digest_certified')

    def test_the_turn1_guard_stays_gated(self):
        result, status, c = self.judge(t1=.006);m = c['t1_value_head']
        self.assertEqual(status, 'FAIL');self.assertEqual((m['guard_mode'], m['guard_evidence'], m['within_bound'], m['passed']), ('gated', None, True, False))
        self.assertGreater(m['beyond_guard'], 0);self.assertFalse(result['checks']['c']['passed'])

    def test_a_planted_sibling_misplacement_beyond_the_bound_fails(self):
        """Two sibling leaves' values swapped (digests intact, as a value written to the wrong own slot): the guard is
        digest-certified, but the swap moves each by more than the value bound."""
        frame = self.cached('frame', self._frame);status, value = np.asarray(frame.leaf_status), np.asarray(frame.leaf_value, np.float64)
        both = (status[..., 0, :] == 4) & (status[..., 1, :] == 4) & np.isfinite(value[..., 0, :]) & np.isfinite(value[..., 1, :])
        gap = np.where(both, np.abs(value[..., 0, :] - value[..., 1, :]), 0.);q, w, node, x = np.unravel_index(int(np.argmax(gap)), gap.shape);d = float(gap.max())
        self.assertGreater(d, 0.)
        def swap(f):f.leaf_value[q, w, node, :, x] = f.leaf_value[q, w, node, ::-1, x].copy()
        result, status, c = self.judge(change=swap, floor=d / 4, spread=d);m = c['leaf_value']   # value bound d / 2, guard d / 4
        self.assertEqual(status, 'FAIL');self.assertTrue(result['checks']['a']['passed'])
        self.assertEqual((m['guard_mode'], m['within_bound'], m['beyond_bound'], m['passed']), ('digest_certified', False, 2, False))

    def test_n2_equivalence_keeps_the_guard_gated(self):
        frame = fake_as_depth2(self.cached('n2', self._n2_frame));other = copy.deepcopy(frame);other.t2_value[np.asarray(other.t2_status) == 4] += np.float32(.006)
        other.t1_own_logits[np.asarray(other.t1_status) != 0] += np.float32(.12)   # beyond 0.1, and no deeper bound at N=2
        found, floats = gate.n2_compare(frame, other);e = gate.decide_roots(frame, frame, self.requests, depth2_backups)
        t1_floor = gate.g2._numeric([[0]], [.07], [0.])   # a turn-1 logit floor that would give a depth >= 3 frame's deeper priors .14
        call = dict(a=found, floats=floats, c=dict(depth1_noise_floor=gate.g2._numeric([[0]], [FLOOR], [0.]), t1_own_logits=t1_floor), e=e, spreads=[SPREAD] * 10)
        report = gate.summarize_n2({'x': [call]})['x'];m = report['floats']['t2_value']
        self.assertEqual(report['status'], 'FAIL');self.assertEqual((m['within_bound'], m['within_guard'], m['passed']), (True, False, False))
        self.assertNotIn('guard_mode', m);self.assertNotIn('guard_depthn_rule', report['tolerance'])
        m = report['floats']['t1_own_logits'];self.assertEqual((m['bound'], m['within_bound']), (gate.g2.LOGIT_BOUND, False));self.assertNotIn('bound_rule', m)
        self.assertNotIn('deeper_logit_bound', report['tolerance'])

    def test_deeper_logits_take_twice_the_turn1_logit_floor(self):
        """Turn-1 own priors .07 off (the floor; 0.1 still bounds them): ply-2 priors .12 off pass under max(0.1, .14) and .15 off
        fail; turn-1, root and depth2_pass priors keep 0.1."""
        result, status, c = self.judge(ply_logits=.12, t1_logits=.07);tolerance = result['tolerance'];m = c['ply2_own_logits']
        self.assertEqual(status, 'PASS', failing(result));self.assertAlmostEqual(tolerance['turn1_logit_floor'], .07, places=5)
        self.assertAlmostEqual(tolerance['deeper_logit_bound'], .14, places=5);self.assertIn('2 x turn1_logit_floor', tolerance['deeper_logit_bound_source'])
        self.assertEqual(tolerance['deeper_logit_rule'], gate.DEEPER_LOGIT_RULE);self.assertEqual(tolerance['deeper_logit_quantities'], ['ply2_own_logits', 'ply2_opp_logits'])
        self.assertEqual((m['bound'], m['bound_rule'], m['within_bound'], m['beyond_bound'], m['passed']), (tolerance['deeper_logit_bound'], 'deeper_logit_bound', True, 0, True))
        self.assertGreater(m['max_abs_difference'], gate.g2.LOGIT_BOUND);self.assertEqual(c['ply2_opp_logits']['bound'], tolerance['deeper_logit_bound'])
        for name in ('t1_own_logits', 't1_peer_logits', 'root_peer_logits'):self.assertEqual((c[name]['bound'], c[name]['bound_rule']), (gate.g2.LOGIT_BOUND, 'logit_bound'), name)
        row = next(r for r in tolerance['table'] if r['quantity'] == 'ply2_own_logits');self.assertEqual((row['bound'], row['ratio_to_bound'], row['passed']), (m['bound'], m['ratio_to_bound'], True))
        line = gate.tolerance_line_n(tolerance);self.assertEqual((line['turn1_logit_floor'], line['deeper_logit_bound']), (tolerance['turn1_logit_floor'], m['bound']))
        self.assertEqual(next(r for r in line['checks'] if r['quantity'] == 'ply2_own_logits')['bound'], m['bound'])
        result, status, c = self.judge(ply_logits=.15, t1_logits=.07);m = c['ply2_own_logits']
        self.assertEqual(status, 'FAIL');self.assertEqual(set(failing(result)), {'c'})
        self.assertEqual((m['within_bound'], m['beyond_bound'], m['passed']), (False, m['compared'], False));self.assertTrue(m['first_beyond'])

    def test_the_turn1_peer_logits_can_set_the_floor(self):
        """Own priors .05 off, opponent priors .07 off at turn 1 (the real searched run's shape: the peer max is the larger): the floor is
        the peer's .07, so ply-2 priors .12 off pass under .14 (an own-only floor would give 0.1 and fail them)."""
        result, status, c = self.judge(ply_logits=.12, t1_logits=.05, t1_peer_logits=.07);tolerance = result['tolerance']
        self.assertEqual(status, 'PASS', failing(result));self.assertGreater(c['t1_peer_logits']['max_abs_difference'], c['t1_own_logits']['max_abs_difference'])
        self.assertEqual(tolerance['turn1_logit_floor'], c['t1_peer_logits']['max_abs_difference']);self.assertIn('t1_peer_logits', tolerance['deeper_logit_bound_source'])
        self.assertAlmostEqual(tolerance['deeper_logit_bound'], .14, places=5);self.assertEqual((c['ply2_own_logits']['bound'], c['ply2_own_logits']['passed']), (tolerance['deeper_logit_bound'], True))

    def test_a_nonfinite_deeper_logit_fails_under_the_deeper_bound(self):
        """One pricer ply-2 own-prior lane NaN against a finite oracle lane (every finite difference 0, the bound .14): non-finite, so
        outside the bound, live and on --reverdict alike."""
        def nan(frame):
            level = frame.nodes[0];at = tuple(np.argwhere((np.asarray(level.status) == 2) & np.asarray(level.own_requested, bool))[0])
            level.own_logits[at + (int(np.flatnonzero(np.asarray(level.own_legal[at], bool))[0]),)] = np.nan
        result, status, c = self.judge(t1_logits=.07, change=nan);m = c['ply2_own_logits']
        self.assertAlmostEqual(result['tolerance']['deeper_logit_bound'], .14, places=5);self.assertEqual((m['nonfinite'], m['max_abs_difference'], m['bound']), (1, 0., result['tolerance']['deeper_logit_bound']))
        self.assertEqual((status, m['within_bound'], m['beyond_bound'], m['ratio_to_bound'], m['passed']), ('FAIL', False, 1, None, False));self.assertEqual(set(failing(result)), {'c'})
        report = saved_gate(t1_logits=.07, ply2_logits=.05)   # as finalize_numeric saves it: the non-finite difference beyond, as 'inf'
        report['checks']['c']['quantities']['ply2_own_logits'].update(nonfinite=1, within_bound=False, beyond_bound=1, ratio_to_bound=None, passed=False,
            first_beyond=[dict(cell=[1, 0, 7, 7], abs_difference='inf')])
        code, out, _, _ = reverdicted(report);m = out['checks']['c']['quantities']['ply2_own_logits']
        self.assertEqual((code, out['status'], m['within_bound'], m['beyond_bound'], m['passed']), (1, 'FAIL', False, 1, False));self.assertAlmostEqual(m['bound'], .14)

    def test_turn1_logits_never_raise_their_own_bound(self):
        result, status, c = self.judge(t1_logits=.12);m = c['t1_own_logits']
        self.assertEqual(status, 'FAIL');self.assertEqual(set(failing(result)), {'c'})
        self.assertEqual((m['bound'], m['within_bound'], m['passed']), (gate.g2.LOGIT_BOUND, False, False))
        self.assertAlmostEqual(result['tolerance']['deeper_logit_bound'], .24, places=5);self.assertTrue(c['ply2_own_logits']['passed'])

    def test_the_deeper_logit_bound_falls_back_to_the_logit_bound(self):
        """No turn-1 logit compared, or a non-finite turn-1 difference: ply-2 priors .12 off are judged against 0.1."""
        def unmeasured(ev):
            for name in gate.TURN1_LOGITS:ev['c'].pop(name)
        def nonfinite(frame):
            at = tuple(np.argwhere((np.asarray(frame.t1_status) == 2) & np.asarray(frame.t1_own_requested, bool))[0])
            frame.t1_own_logits[at + (int(np.flatnonzero(np.asarray(frame.t1_own_legal[at], bool))[0]),)] = np.nan
        for case, kw, source in (('unmeasured', dict(edit=unmeasured), 'no turn-1 logit'), ('nonfinite', dict(change=nonfinite, t1_logits=.07), 'non-finite')):
            with self.subTest(case):
                result, status, c = self.judge(ply_logits=.12, **kw);tolerance = result['tolerance'];m = c['ply2_own_logits']
                self.assertEqual((tolerance['turn1_logit_floor'], tolerance['deeper_logit_bound']), (None, gate.g2.LOGIT_BOUND));self.assertIn(source, tolerance['deeper_logit_bound_source'])
                self.assertEqual((m['bound'], m['within_bound'], m['passed'], status), (gate.g2.LOGIT_BOUND, False, False, 'FAIL'))

    def test_reverdict_applies_the_rule_to_a_saved_report(self):
        code, out, lines, path = reverdicted(saved_gate())
        self.assertEqual((code, out['status'], out['passed'], out['status_reasons']), (0, 'PASS', True, []))
        self.assertEqual((out['reverdict']['source_status'], out['reverdict']['source_failing_checks'], out['reverdict']['failing_checks']), ('FAIL', ['c'], []))
        c = out['checks']['c']['quantities'];self.assertTrue(out['checks']['c']['passed'])
        self.assertEqual((c['leaf_value']['guard_mode'], c['leaf_value']['passed']), ('digest_certified', True))
        self.assertEqual(c['leaf_value']['guard_evidence']['subset'].split(':')[0], 'by construction')
        self.assertEqual(c['leaf_value']['guard_evidence']['evaluator_actors'], 1);self.assertIn('evaluator.override', c['leaf_value']['guard_evidence']['evaluator_actors_source'])
        self.assertEqual(c['ply2_value_head']['guard_mode'], 'gated')   # nodes need the per-position count, which this report predates
        self.assertEqual((c['t1_value_head']['guard_mode'], c['depth2_pass_t2_value']['guard_mode']), ('gated', 'gated'))
        self.assertEqual(out['tolerance']['guard_depthn_rule'], gate.GUARD_DEPTHN_RULE)
        self.assertEqual(next(r for r in out['tolerance']['table'] if r['quantity'] == 'leaf_value')['guard_mode'], 'digest_certified')
        line = next(x for x in lines if x['message'] == 'REVERDICT');self.assertEqual((line['old_status'], line['new_status'], line['evaluator_actors']['count']), ('FAIL', 'PASS', 1))
        self.assertEqual(line['changed_quantities'], [dict(quantity='leaf_value', passed_before=False, passed=True, guard_mode='digest_certified')])
        leaf = next(x for x in lines if x['message'] == 'REVERDICT_QUANTITY' and x['quantity'] == 'leaf_value')
        self.assertEqual((leaf['guard_mode'], leaf['within_guard'], leaf['passed_before'], leaf['passed']), ('digest_certified', False, False, True))
        before = (path.parent / gate.REVERDICT_FILE).read_bytes()
        with self.assertRaises(SystemExit) as raised:gate.main(['--reverdict', str(path)])
        self.assertIn('never overwritten', str(raised.exception.code));self.assertEqual((path.parent / gate.REVERDICT_FILE).read_bytes(), before)
        for case, report in (('depth2', saved_gate(depth=2)), ('uncounted_node', saved_gate(ply2_noise=.006)), ('bank_actors', saved_gate(single_actor=False))):
            with self.subTest(case):   # each keeps a guard gated
                code, out, _, _ = reverdicted(report);self.assertEqual((code, out['status'], out['passed']), (1, 'FAIL', False))
        m = out['checks']['c']['quantities']['leaf_value'];self.assertEqual((m['guard_mode'], m['guard_evidence']['evaluator_actors']), ('gated', None))
        self.assertIn('not established', m['guard_evidence']['evaluator_actors_source'])
        sampled = dict(saved_gate(), sample_leaves=2);sampled['checks'].pop('e');code, out, _, _ = reverdicted(sampled)
        self.assertEqual((code, out['status']), (0, 'PASS_SAMPLED'))

    def test_reverdict_refuses_incomplete_reports(self):
        def drop(*keys):
            report = saved_gate();node = report
            for key in keys[:-1]:node = node[key]
            node.pop(keys[-1]);return report
        cases = dict(within_guard=drop('checks', 'c', 'quantities', 'leaf_value', 'within_guard'), within_bound=drop('checks', 'c', 'quantities', 'ply2_value_head', 'within_bound'),
            digest_counts=drop('checks', 'a', 'compared'), depth=drop('pricer_options', 'depth'), sample_leaves=drop('sample_leaves'), quantities=drop('checks', 'c', 'quantities'),
            n2_report=dict(saved_gate(), mode='N2_EQUIVALENCE'), schema=dict(saved_gate(), schema='pokezero.depth2-gate.v1'), choices=drop('checks', 'f'),
            choice_flips=drop('checks', 'f', 'differing'), value_bound=drop('tolerance', 'value_bound'))
        for case, report in cases.items():
            with self.subTest(case):
                directory = Path(tempfile.mkdtemp(prefix='depthn-reverdict-'));path = directory / 'gate.json';path.write_text(json.dumps(report))
                with patch.object(gate, 'log', lambda *a, **k: None), self.assertRaises(SystemExit) as raised:gate.main(['--reverdict', str(path)])
                self.assertIn('lacks what the verdict needs', str(raised.exception.code));self.assertFalse((directory / gate.REVERDICT_FILE).exists())
        with self.assertRaises(SystemExit) as raised:gate.main(['--reverdict', str(Path(tempfile.mkdtemp()) / 'absent.json')])
        self.assertIn('cannot read', str(raised.exception.code))
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (['--reverdict', 'g.json', '--out', 'o'], ['--reverdict', 'g.json', '--d3-root', 'b', '--chunk', '0'], ['--d3-root', 'b', '--chunk', '0'],
                    ['--reverdict', 'g.json', '--depth', '4'], ['--reverdict', 'g.json', '--sample-leaves', '2'], ['--reverdict', 'g.json', '--block-invariance'],
                    ['--reverdict', 'g.json', '--selection', 'searched'], ['--reverdict', 'g.json', '--head-archive', 'h'], ['--reverdict', 'g.json', '--mutation-test']):
                with self.assertRaises(SystemExit) as raised:gate.main(argv)
                self.assertEqual(raised.exception.code, 2, argv)

    def test_reverdict_applies_the_deeper_logit_bound(self):
        code, out, lines, _ = reverdicted(saved_gate(t1_logits=.06, ply2_logits=.11));c = out['checks']['c']['quantities'];m = c['ply2_own_logits']
        self.assertEqual((code, out['status']), (0, 'PASS'));self.assertAlmostEqual(m['bound'], .12);self.assertAlmostEqual(out['tolerance']['deeper_logit_bound'], .12)
        self.assertEqual((m['bound_rule'], m['within_bound'], m['beyond_bound'], m['first_beyond'], m['passed']), ('deeper_logit_bound', True, 0, [], True))
        self.assertEqual({x['quantity'] for x in out['reverdict']['changed_quantities']}, {'leaf_value', 'ply2_own_logits'})
        self.assertEqual((c['t1_own_logits']['bound'], c['depth2_pass_t1_own_logits']['bound'], c['depth2_pass_t1_own_logits']['bound_rule']), (.1, .1, 'logit_bound'))
        self.assertEqual(next(r for r in out['tolerance']['table'] if r['quantity'] == 'ply2_own_logits')['bound'], m['bound'])
        line = next(x for x in lines if x['message'] == 'REVERDICT_QUANTITY' and x['quantity'] == 'ply2_own_logits')
        self.assertEqual((line['bound_rule'], line['passed_before'], line['passed']), ('deeper_logit_bound', False, True))
        self.assertAlmostEqual(next(x for x in lines if x['message'] == 'REVERDICT')['deeper_logit_bound'], .12)
        for t1, ply2, bound in ((.05, .11, .1), (.06, .13, .12)):   # 2 x .05 leaves no headroom over 0.1; .13 exceeds 2 x .06
            with self.subTest(t1=t1, ply2=ply2):
                code, out, _, _ = reverdicted(saved_gate(t1_logits=t1, ply2_logits=ply2));m = out['checks']['c']['quantities']['ply2_own_logits']
                self.assertEqual((code, out['status'], m['within_bound'], m['beyond_bound'], m['passed']), (1, 'FAIL', False, 1, False));self.assertAlmostEqual(m['bound'], bound)

    def test_reverdict_on_saved_reports(self):
        """A synthetic report with a raw-head arm, then each V2E_DEPTHN_SAVED_GATE_REPORTS file (unset: none, never a skip): PASS,
        every other check's verdict as saved, and only what the current rules relax changed (a digest-certified guard on a single
        evaluator actor, or a deeper prior under 2 x the turn-1 logit floor); a report from before (f) is refused as missing checks.f."""
        report = saved_gate();report['checks']['c']['quantities']['raw_head_leaf_value'] = dict(report['checks']['c']['quantities']['leaf_value'])
        code, out, _, _ = reverdicted(report);self.assertEqual((code, out['status']), (0, 'PASS'))
        self.assertEqual([x['quantity'] for x in out['reverdict']['changed_quantities']], ['leaf_value', 'raw_head_leaf_value'])
        for path in SAVED_REPORTS:
            with self.subTest(path.name):
                source = json.loads(path.read_text())
                if 'f' not in (source.get('checks') or {}):
                    with self.assertRaises(SystemExit) as raised:reverdicted(path, name=path.name)
                    self.assertIn('checks.f', str(raised.exception.code));continue
                code, out, _, _ = reverdicted(path, name=path.name);c, tolerance = out['checks']['c']['quantities'], out['tolerance']
                self.assertEqual((code, out['status'] in ('PASS', 'PASS_SAMPLED')), (0, True), out['reverdict'])
                self.assertEqual({k: v['passed'] for k, v in out['checks'].items() if k != 'c'}, {k: v['passed'] for k, v in source['checks'].items() if k != 'c'})
                for change in out['reverdict']['changed_quantities']:
                    m = c[change['quantity']];self.assertTrue(change['passed'] and (m.get('guard_mode') == 'digest_certified' or m.get('bound_rule') == 'deeper_logit_bound'), change)
                    if m.get('guard_mode') == 'digest_certified':self.assertEqual(m['guard_evidence']['evaluator_actors'], 1)
                measured = [c[n] for n in gate.TURN1_LOGITS if (c.get(n) or {}).get('compared')]
                if measured and not any(m['nonfinite'] for m in measured):self.assertEqual(tolerance['turn1_logit_floor'], max(m['max_abs_difference'] for m in measured))
                for name in tolerance['deeper_logit_quantities']:self.assertEqual(c[name]['bound'], tolerance['deeper_logit_bound'], name)
                self.assertEqual([c[n]['bound'] for n in gate.TURN1_LOGITS if n in c], [gate.g2.LOGIT_BOUND] * len([n for n in gate.TURN1_LOGITS if n in c]))


# ---------------------------------------------------------------- (f) interior-node choices and the mutation kill test

class ChoicesTests(FixtureN, unittest.TestCase):
    """(f) on the stand-in surface (the oracle's own values: the value bound is 0 unless a synthetic floor is set), the gate's independent
    recursion against the backups, the per-child grid gate (misreads inside the backup, the grid captured as passed, any rule), the row-q
    clause, rule 1's per-node exemption and the bound's edge, one-sided pricing, sampled runs, --reverdict and the header; then the eight
    mutants on the fake backups' source at the run's value bound and both sweeps' survivor accounting."""
    _d = GateNTests._d

    def prior(self):
        return self.cached('prior', lambda: self.call(self.pricers()))

    def frame_data(self, depth=3, k=2):
        """The stand-in's depth-N frame (k_own = k_opp = k) and the gate's oracle data over it."""
        def make():
            with self.engine():
                frame = OracleNPricer(self.ex, object(), depth=depth, k_own=k, k_opp=k).price(self.requests, self.tables, self.recipe, PARAMS)
                stack = reference.stack(self.requests, batch._numpy_worlds(self.requests, self.tables, self.recipe))
                return frame, gate.oracle_levels(oracle=self.oracle, stack=stack, requests=self.requests, recipe=self.recipe, params=PARAMS, frame=frame,
                    plan=gate.plan_of(depth, k, k))
        return self.cached(('frame_data', depth, k), make)

    def judge(self, frame, data, *, backups=FakeBackupsN, floor=None):
        """(result, status, (f)) of every check on (frame, data) through summarize_run; floor: a synthetic depth-1 floor (value bound 2 x floor)
        with the evaluator actors 1 (GUARD_DEPTHN_RULE certifies past turn 1)."""
        ev = gate.evaluate_n(frame, self._d(), self.requests, data, context={}, backups=backups, decision=True, choices=True)
        if floor is not None:ev['c']['depth1_noise_floor'] = gate.g2._numeric([[0]], [floor], [0.]);ev['spreads'] = [SPREAD] * 10;ev['actors'] = 1
        result, (status, _) = self.judged(ev);return result, status, result['checks']['f']

    def test_choices_pass_on_the_clean_surface(self):
        evaluation, summary = self.prior();result, (status, reasons) = self.judged(evaluation);f = result['checks']['f']
        self.assertEqual(status, 'PASS', (failing(result), reasons));self.assertEqual(list(result['checks'])[-2:], ['e', 'f'])
        self.assertEqual((f['passed'], f['status'], f['non_exempt_mismatches'], f['one_sided'], f['problems'], f['not_comparable']), (True, 'COMPARED', 0, 0, 0, 0))
        self.assertEqual(set(f['nodes_by_ply']), {'1', '2'});self.assertEqual(f['nodes'], f['learner_comparisons']);self.assertGreater(f['opponent_comparisons'], f['nodes'])
        self.assertEqual((f['max_abs_delta_row_q'], f['max_abs_delta_child_value'], f['value_bound'], f['row_q_within_bound']), (0., 0., 0., True))
        self.assertGreater(f['near_ties'], 0);self.assertEqual(f['near_tie_flips'], 0)   # exact ties among the fake's integer values (bound 0), none flips
        self.assertEqual((result['tolerance']['f_near_ties'], summary['choices']['comparisons']), (f['near_ties'], f['comparisons']))
        for depth, selection in ((4, 'prior'), (3, 'searched')):
            with self.subTest(depth=depth, selection=selection):
                ev, _ = self.call(self.pricers(depth=depth, selection=selection), decision=False);result, (status, _) = self.judged(ev)
                self.assertEqual(status, 'PASS', failing(result));self.assertEqual(set(result['checks']['f']['nodes_by_ply']), {str(p) for p in range(1, depth)})

    def test_the_oracle_recursion_equals_the_backups_and_reads_no_pricer_value(self):
        """OracleTree's values equal node_value's on every interior node (depths 3 and 4) and its choices equal pricer_record's; built over a
        frame whose values and statuses are scrambled (logits kept), it returns the same: it reads only the pricer's priors."""
        for depth in (3, 4):
            with self.subTest(depth=depth):
                frame, data = self.frame_data(depth);tree = gate.OracleTree(frame, data);nodes = gate.interior_nodes(frame, data);priced = 0
                scrambled = copy.deepcopy(frame);scrambled.leaf_value[...] = 7.;scrambled.leaf_status[...] = 3;scrambled.t1_status[...] = 0
                for level in scrambled.nodes:level.value[...] = -7.;level.status[...] = 0
                blind = gate.OracleTree(scrambled, data)
                for q, w, ply, node in nodes:
                    mine, (value, record) = FakeBackupsN.node_value(frame, q, w, ply, node), tree.solve(q, w, ply, node)
                    self.assertEqual(mine, value, (q, w, ply, node));self.assertEqual(blind.solve(q, w, ply, node)[0], value);priced += mine is not None
                    if record is not None:
                        pricer = gate.pricer_record(frame, FakeBackupsN, q, w, ply, node)
                        self.assertEqual([pricer[k] for k in ('own', 'opp', 'learner', 'argmin', 'q', 'gap', 'row_gaps', 'value')],
                            [record[k] for k in ('own', 'opp', 'learner', 'argmin', 'q', 'gap', 'row_gaps')] + [value])
                self.assertGreater(priced, 200);self.assertEqual({k[2] for k in nodes}, set(range(1, depth)))

    def plant(self, oracle_rows, pricer_rows):
        """(frame, data, P): copies with the four leaves of the first fully head-valued two-by-two ply-2 node P set to base + row offset, the
        oracle's (head values) and the pricer's (leaf_value) separately; digests untouched."""
        frame, data = self.frame_data();level = frame.nodes[0];status = np.asarray(frame.leaf_status)
        P = next((q, w, n) for q, w, n in np.argwhere(np.asarray(level.status) == 2).tolist() if (status[q, w, n] == 4).all()
            and (np.asarray(level.own_actions[q, w, n]) >= 0).all() and (np.asarray(level.opp_actions[q, w, n]) >= 0).all())
        frame, data = copy.deepcopy(frame), copy.deepcopy(data);base = float(frame.leaf_value[P + (0, 0)]);leaves = data['leaves']
        for i, key in enumerate(leaves['index'].tolist()):
            if tuple(key[:3]) == P:leaves['head_value'][i] = leaves['value'][i] = np.float32(base + oracle_rows[key[3]])
        for own in (0, 1):frame.leaf_value[P + (own,)] = np.float32(base + pricer_rows[own])
        return frame, data, P

    def test_near_tie_flips_are_exempt_and_reported_and_clear_flips_fail(self):
        """P's rows tie in the oracle and the pricer's row 1 leads by FLOOR (inside the value bound 2 x FLOOR): the learner flip is a near tie,
        reported with both gaps, and the run passes. Rows .05 apart the other way on each side: both gaps exceed the bound, (f) fails."""
        frame, data, P = self.plant((0., 0.), (0., FLOOR));result, status, f = self.judge(frame, data, floor=FLOOR)
        self.assertEqual(status, 'PASS', failing(result));self.assertEqual((f['passed'], f['non_exempt_mismatches']), (True, 0));self.assertGreaterEqual(f['near_tie_flips'], 1)
        flip = next(r for r in f['near_tie_flip_examples'] if (r['root'], r['world'], r['node']) == P and r['ply'] == 2);self.assertEqual((flip['kind'], flip['gap_oracle']), ('learner', 0.))
        self.assertAlmostEqual(flip['gap_pricer'], FLOOR, places=6);self.assertEqual(flip['actions'][::-1], [int(a) for a in frame.nodes[0].own_actions[P][:2]])
        header = gate.header_block(result['checks'])['near_ties']['f'];self.assertEqual((header['near_ties'], header['near_tie_flips']), (f['near_ties'], f['near_tie_flips']))
        frame, data, P = self.plant((.05, 0.), (0., .05));result, status, f = self.judge(frame, data, floor=FLOOR)
        self.assertEqual(status, 'FAIL');self.assertIn('f', failing(result));self.assertFalse(f['passed']);self.assertGreaterEqual(f['non_exempt_mismatches'], 1)
        first = next(r for r in f['first_mismatches'] if (r['root'], r['world'], r['node']) == P and r['ply'] == 2)
        self.assertEqual(first['kind'], 'learner');self.assertTrue(min(first['gap_pricer'], first['gap_oracle']) > f['value_bound'])
        self.assertTrue(any('flip with a gap above the value bound in both frames' in r for r in f['reasons']), f['reasons']);self.assertFalse(f['grid_within_bound'])

    def test_a_node_priced_on_one_side_only_fails(self):
        frame, data = self.frame_data()
        key = next(k for k in gate.interior_nodes(frame, data) if k[2] == 2 and FakeBackupsN.node_value(frame, *k) is not None)
        def node_value(f, q, w, ply, node, rule=FakeBackupsN.learner_max):
            return None if (q, w, ply, node) == key else FakeBackupsN.node_value(f, q, w, ply, node, rule)
        backups = SimpleNamespace(node_value=node_value, learner_max=FakeBackupsN.learner_max, learner_max_root=FakeBackupsN.learner_max_root)
        result, status, f = self.judge(frame, data, backups=backups)
        self.assertEqual(status, 'FAIL');self.assertEqual(set(failing(result)), {'f'});self.assertEqual(f['one_sided'], 1)
        self.assertEqual({k: f['one_sided_examples'][0][k] for k in ('root', 'world', 'ply', 'node', 'priced_by')}, dict(root=key[0], world=key[1], ply=2, node=key[3], priced_by='oracle'))
        self.assertIn('priced on one side only', ' '.join(f['reasons']))
        record = gate.pricer_record(frame, FakeBackupsN, *key);self.assertFalse(gate.node_fails(record, gate.OracleTree(frame, data).solve(*key)[1], 0.))
        self.assertEqual(gate.compare_records(dict(record, finite=False), record)[:2], ([], ['non-finite row q or child value (pricer)']))   # never a NaN gap
        self.assertTrue(gate.node_fails(dict(record, finite=False), record, 1.) and gate.node_fails(record, None, 1.) and not gate.node_fails(None, None, 0.))
        self.assertIn('own actions differ', gate.compare_records(dict(record, own=[8] * len(record['own'])), record)[1][0])
        with patch.dict(sys.modules, {'gpu_engine.depthn_backups': None}):self.assertIn('fail closed', gate.choice_check_n(frame, data, self.requests)['reason'])
        self.assertEqual(gate.merge_choices([dict(passed=False, reason='x')], value_bound=0.)['status'], 'NOT_RUN')

    def test_sampled_runs_compare_only_fully_checked_nodes(self):
        full = self.judged(self.prior()[0])[0]['checks']['f']
        ev, _ = self.call(self.pricers(), sample=1, decision=False);result = gate.summarize_run([ev], (2187,));f = result['checks']['f']
        self.assertEqual(gate.gate_status_n(result, sampled=True)[0], 'PASS_SAMPLED', failing(result));self.assertTrue(f['sampled']);self.assertGreater(f['not_comparable'], 0)
        self.assertLess(f['comparisons'], full['comparisons']);self.assertEqual(f['status'], 'COMPARED' if f['comparisons'] else 'SKIPPED_SAMPLED')
        ev, _ = self.call(self.pricers(), sample=10 ** 6, decision=False);result = gate.summarize_run([ev], (2187,));f = result['checks']['f']   # every leaf drawn
        self.assertEqual((gate.gate_status_n(result, sampled=True)[0], f['comparisons'], f['not_comparable']), ('PASS_SAMPLED', full['comparisons'], 0))
        empty = gate.choice_verdict(dict(saved_choices(), comparisons=0, nodes=0, not_comparable=4), .008, sampled=True)
        self.assertEqual((empty['status'], empty['passed'], gate.gate_status_n(dict(checks=dict(f=empty)), sampled=True)[0]), ('SKIPPED_SAMPLED', True, 'PASS_SAMPLED'))
        vacuous = gate.choice_verdict(dict(saved_choices(), comparisons=0, nodes=0), .008)
        self.assertEqual(gate.gate_status_n(dict(checks=dict(f=vacuous))), ('INCOMPLETE', ['vacuous: (f) compared no interior node']))
        self.assertFalse(gate.choice_verdict(dict(saved_choices(), not_comparable=3), .008)['passed'])   # unsampled: the oracle must value every node

    def test_reverdict_carries_f_through_the_same_verdict(self):
        code, out, lines, _ = reverdicted(saved_gate());f = out['checks']['f']
        self.assertEqual((code, out['status'], f['passed'], f['near_ties'], f['non_exempt_mismatches']), (0, 'PASS', True, 2, 0))
        self.assertEqual(out['header']['near_ties']['f']['near_ties'], 2);self.assertEqual(next(x for x in lines if x['message'] == 'REVERDICT')['header'], out['header'])
        self.assertEqual(list(out)[:2], ['evaluator', 'header'])
        report = saved_gate();report['checks']['f'] = saved_choices([(.001, .02)])   # a near-tie flip: reported, not gated
        code, out, _, _ = reverdicted(report);self.assertEqual((code, out['checks']['f']['near_tie_flips'], out['checks']['f']['passed']), (0, 1, True))
        report = saved_gate();report['checks']['f'] = saved_choices([(.02, .03)])   # saved as passed: the verdict function fails it
        code, out, _, _ = reverdicted(report);f = out['checks']['f']
        self.assertEqual((code, out['status'], f['passed'], f['non_exempt_mismatches']), (1, 'FAIL', False, 1));self.assertIn('f', out['reverdict']['failing_checks'])
        self.assertIn(dict(quantity='f', passed_before=True, passed=False), out['reverdict']['changed_quantities'])
        report = saved_gate();report['checks']['f'] = dict(saved_choices([(.02, .03)]), differing_complete=False, non_exempt_mismatches=1, passed=False)
        code, out, _, _ = reverdicted(report);self.assertEqual((code, out['checks']['f']['non_exempt_mismatches']), (1, 1))   # truncated list: kept under the same bound

    def test_header_block(self):
        e, f = dict(near_tie_roots=3, near_tie_flips=1, q_noise_bound=.008), dict(near_ties=7, near_tie_flips=2, status='COMPARED', value_bound=.008)
        block = gate.header_block(dict(e=dict(e, passed=True), f=dict(f, passed=True)));self.assertEqual(block['near_ties'], dict(e=e, f=f))
        rules = block['rules_changed_after_failure'];self.assertEqual(len(rules), 4)
        self.assertEqual([r['triggering_run'] for r in rules], ['d2-gate40-b0-r1', None, 'dn-gate40-prior3-r3', 'dn-gate40-searched3-r3'])
        self.assertTrue(all(set(r) == {'name', 'summary', 'triggering_run', 'trigger', 'approval'} for r in rules))
        self.assertEqual([r['approval'] for r in rules], ['confirmed by the lead 2026-11-06'] * 2 + ['per-child grid gate replaces the guard at plies >= 2: '
            'signed off by the lead 2026-11-06', 'confirmed by the lead 2026-11-06'])
        self.assertIn('not a run', rules[1]['trigger']);self.assertIn('per-child grid gate', rules[2]['summary'])
        self.assertTrue(all(s in rules[0]['trigger'] for s in ('d2-gate40-b0-r1', '60a52442', 'root 15', '1.2e-3 apart in the pricer frame', '2.3e-5 apart in the oracle-heads frame', '4f88727e')))
        self.assertIn('GUARD_DEPTHN_RULE', rules[2]['name']);self.assertIn('DEEPER_LOGIT_RULE', rules[3]['name']);self.assertEqual(gate.header_block()['near_ties'], dict(e=None, f=None))
        self.assertEqual(gate.header_block(configs=dict(top2=dict(e=e)))['near_ties'], dict(top2=dict(e=e)));self.assertIn('per-child grid gate', gate.GUARD_DEPTHN_RULE)

    # ---- the per-child grid gate, the row-q clause and the bound's edge (the review's findings 1, 2 and 4)

    def find(self, k, want):
        """(frame, data, P, record, (row slot, x, y)) for the first ply-2 node P of the k x k stand-in and a pair of opponent slots x, y in one row
        that want(record, row, x, y) accepts (row: the record's row index; x, y: its column indices)."""
        frame, data = self.frame_data(k=k)
        for key in gate._parents(frame):
            record = gate.pricer_record(frame, FakeBackupsN, *key)
            if record is None or 'shape' in record:continue
            for row in range(len(record['own'])):
                for x, y in itertools.combinations(range(len(record['opp'])), 2):
                    if want(record, row, x, y):return frame, data, key, record, (row, x, y)
        self.fail('no such node on the stand-in')

    def swap_in_row(self, key, record, row, x, y):
        """mutant_backups (the fake's source) whose node_value at P reads row `row`'s opponent slots x and y swapped: a misread INSIDE the backup."""
        i, sx, sy = record['own_slots'][row], record['opp_slots'][x], record['opp_slots'][y]
        return gate.mutant_backups([(key + (i, sx), key + (i, sy))], source=FAKE_BACKUPS_SOURCE)

    def test_a_non_pivotal_misread_fails_on_the_per_child_grid_gate_alone(self):
        """Two opponent columns of one row with EQUAL priors, neither the row's argmin, values more than the bound apart, read swapped inside the
        backup: no row q moves (the same products, fsum exact), no choice moves, the frame is clean; only the per-child grid gate fails."""
        floor = 1e-4;bound = 2 * float(np.float32(floor))
        def want(r, row, x, y):
            g = r['grid'][row];low = r['argmin'][row]
            return r['pi_opp'][x] == r['pi_opp'][y] and low not in (x, y) and abs(g[x] - g[y]) > 2 * bound
        frame, data, key, record, (row, x, y) = self.find(3, want)
        result, status, f = self.judge(frame, data, backups=self.swap_in_row(key, record, row, x, y), floor=floor)
        self.assertEqual(status, 'FAIL');self.assertEqual(set(failing(result)), {'f'}, failing(result))
        self.assertEqual((f['grid_within_bound'], f['row_q_within_bound'], f['max_abs_delta_row_q'], f['non_exempt_mismatches'], f['near_tie_flips']), (False, True, 0., 0, 0))
        self.assertEqual(f['nodes_grid_beyond_bound'], 1);self.assertAlmostEqual(f['max_abs_delta_child_value'], abs(record['grid'][row][x] - record['grid'][row][y]))
        self.assertEqual({k: f['first_grid_mismatches'][0][k] for k in ('root', 'world', 'ply', 'node')}, dict(root=key[0], world=key[1], ply=2, node=key[3]))
        self.assertTrue(any('per-child grid gate' in r for r in f['reasons']))
        self.assertTrue(self.judge(frame, data, floor=floor)[2]['passed'])   # the clean backup passes

    def test_a_row_q_misread_without_a_flip_fails_on_the_row_q_clause(self):
        """A non-learner row's two non-argmin columns with unequal priors read swapped: that row's q moves beyond the bound, no choice flips;
        the row-q clause fails (beside the grid gate), on a row other than the learner's."""
        floor = 1e-4;bound = 2 * float(np.float32(floor))
        def want(r, row, x, y):
            g = r['grid'][row];dq = abs((r['pi_opp'][x] - r['pi_opp'][y]) * (g[y] - g[x]));moved = r['q'][row] + (r['pi_opp'][x] - r['pi_opp'][y]) * (g[y] - g[x])
            return row != r['learner'] and r['argmin'][row] not in (x, y) and dq > 2 * bound and moved < r['q'][r['learner']] - 2 * bound and r['gap'] > 2 * bound
        frame, data, key, record, (row, x, y) = self.find(3, want)
        result, status, f = self.judge(frame, data, backups=self.swap_in_row(key, record, row, x, y), floor=floor)
        self.assertEqual(status, 'FAIL');self.assertEqual(set(failing(result)), {'f'}, failing(result))
        self.assertEqual((f['row_q_within_bound'], f['non_exempt_mismatches'], f['near_tie_flips']), (False, 0, 0));self.assertEqual(f['nodes_row_q_beyond_bound'], 1)
        self.assertTrue(any(r.startswith('max |row q diff|') for r in f['reasons']));self.assertGreater(f['max_abs_delta_row_q'], bound)

    def test_the_grid_is_captured_as_passed_to_the_rule(self):
        """A backup that permutes P's grid only where it hands it to the rule (every leaf read and the frame correct): the recorder holds the grid
        as passed, so the per-child grid gate fails at P; a gate re-gathering P's leaves from the frame would see nothing."""
        frame, data, key, record, _ = self.find(2, lambda r, row, x, y: r['grid'][0][0] != r['grid'][0][1])
        module = gate.mutant_backups([], source=FAKE_BACKUPS_SOURCE)
        module._AT_RULE = lambda grid, root, world, ply, node: [row[::-1] for row in grid] if (root, world, ply, node) == key else grid
        result, status, f = self.judge(frame, data, backups=module)
        self.assertEqual(status, 'FAIL');self.assertEqual(set(failing(result)), {'f'});self.assertFalse(f['grid_within_bound'])
        self.assertIn(key, [(r['root'], r['world'], r['ply'], r['node']) for r in f['first_grid_mismatches']])
        np.testing.assert_array_equal(np.asarray(frame.leaf_value)[key[:2] + key[3:]][:len(record['own']), :len(record['opp'])], np.asarray(record['grid'], np.float32))

    def test_any_rule_passes_through_the_recorder(self):
        """A stub min-max rule (not learner-max) on both sides: the recorder returns its value unchanged, the clean grids certify, and a misread
        inside the backup fails the per-child grid gate under it too (the hedge plugs in the same way)."""
        minmax = lambda grid, pi_own, pi_opp: max(min(row) for row in grid)
        frame, data = self.frame_data(k=3);tree = gate.OracleTree(frame, data, rule=minmax)
        clean = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=FakeBackupsN, rule=minmax, tree=tree), 0.)
        self.assertTrue(clean['passed'], clean['reasons']);self.assertGreater(clean['nodes'], 100);self.assertEqual(clean['max_abs_delta_child_value'], 0.)
        moved = [k for k in gate.interior_nodes(frame, data) if FakeBackupsN.node_value(frame, *k) not in (None, FakeBackupsN.node_value(frame, *k, rule=minmax))]
        self.assertTrue(moved);key = moved[0]
        self.assertEqual(gate.pricer_record(frame, FakeBackupsN, *key, rule=minmax)['value'], FakeBackupsN.node_value(frame, *key, rule=minmax))
        self.assertEqual(tree.solve(*key)[0], FakeBackupsN.node_value(frame, *key, rule=minmax))
        _, _, P, record, (row, x, y) = self.find(3, lambda r, row, x, y: abs(r['grid'][row][x] - r['grid'][row][y]) > 0)
        bad = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=self.swap_in_row(P, record, row, x, y), rule=minmax, tree=tree), 0.)
        self.assertFalse(bad['grid_within_bound']);self.assertFalse(bad['passed'])

    def test_a_gap_or_difference_at_the_bound_is_within_it(self):
        """Exactly at the bound (and a rounding's REL_EPS above) a gap is a near tie and a row q or grid difference passes; 1e-9 above, not."""
        for scale, within in ((1., True), (1. + 1e-13, True), (1. + 1e-9, False)):
            with self.subTest(scale=scale):
                at = .008 * scale;f = gate.choice_verdict(saved_choices([(at, .03)]), .008)
                self.assertEqual((f['passed'], f['near_tie_flips'], f['non_exempt_mismatches']), (within, int(within), int(not within)))
                for key in ('max_abs_delta_row_q', 'max_abs_delta_child_value'):
                    self.assertEqual(gate.choice_verdict(dict(saved_choices(), **{key: at}), .008)['passed'], within, key)
                self.assertEqual(gate._within(at, .008), within)
        self.assertFalse(gate._within(1e-300, 0.));self.assertTrue(gate._within(0., 0.))

    def test_tie_breaks_keep_the_lower_slot(self):
        solved = gate._solve([[.5, .5, .2], [.3, .3, .3]], [.25, .25, .5], [0, 2], [0, 1, 2], [4, 6], [1, 2, 3])
        self.assertEqual((solved['argmin'], solved['learner'], solved['row_gaps']), ([2, 0], 0, [.3, 0.]))
        self.assertEqual(gate._solve([[.3, .3], [.3, .3]], [.5, .5], [0, 1], [0, 1], [4, 6], [1, 2])['learner'], 0)

    def test_an_errored_descendant_decides_none_over_a_sampled_out_leaf(self):
        frame, data = self.frame_data();leaves = data['leaves'];status = np.asarray(frame.leaf_status)
        key = next(k for k in gate._parents(frame) if (status[k[:2] + k[3:]] == 4).sum() >= 2 and gate.OracleTree(frame, data).solve(*k)[0] is not None)
        rows = [i for i, index in enumerate(leaves['index'].tolist()) if (index[0], index[1], index[2]) == (key[0], key[1], key[3])]
        def tree(drop, error):
            changed = copy.deepcopy(data);mask = np.ones(len(leaves['specs']), bool);mask[drop] = False
            if error is not None:changed['leaves']['status'][error] = 3
            changed['leaves'] = {k: (np.asarray(v)[mask] if k in ('specs', 'index', 'status', 'value', 'head_value') else v) for k, v in changed['leaves'].items()}
            return gate.OracleTree(frame, changed)
        self.assertIs(tree([rows[0]], None).solve(*key)[0], gate.MISSING);self.assertIs(tree([rows[0]], rows[1]).solve(*key)[0], None)
        self.assertIs(tree([rows[0]], rows[1]).solve(*gate._up(frame, key))[0], None)

    # ---- the mutation kill test

    def mutated(self, value_bound=None):
        """The prior k = 3 call with --mutation-test (the fake backups' source) judged at value_bound (default: the call's own run bound), cached:
        (evaluation, mutation report, log lines)."""
        def make():
            pricers = self.pricers(k_own=3, k_opp=3);lines = []
            depth2_pricer = gate.depth2_pass_pricer(FAKE_MODULE, self.ex, object(), capacity=32, actor_capacity=3, digest=True)
            with self.engine():
                ev, _ = gate.run_call_n(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS, pricers=pricers, depth2_pricer=depth2_pricer,
                    frame_pricer=self.frame_pricer(), oracle=self.oracle, sample=None, block=0, context=dict(block=0, call=0), backups=FakeBackupsN,
                    mutation=dict(source=FAKE_BACKUPS_SOURCE))
            bound = gate.summarize_run([ev], (2187,))['tolerance']['value_bound'] if value_bound is None else value_bound
            (report,) = gate.mutation_tests([ev], value_bound=bound, backups=FakeBackupsN, progress=lambda message, **k: lines.append(dict(k, message=message)))
            return ev, report, lines
        return self.cached(('mutated', value_bound), make)

    def test_every_mutant_is_killed_on_the_fakes(self):
        ev, m, lines = self.mutated()
        self.assertTrue(m['passed'], {n: (r.get('caught_by'), r.get('reason')) for n, r in m['mutants'].items()});self.assertNotIn(gate.MUTATION_INPUTS, ev)
        self.assertEqual((m['clean']['status'], m['clean_unchanged'], m['survivors'], m['material_survivors']), ('PASS', True, [], dict(scatter=0, backup=0)))
        self.assertEqual(self.judged(ev)[1][0], 'PASS');self.assertEqual(list(m['mutants']), [f'{p}-{k}-{v}' for p, k, v in gate.MUTANTS])
        for name, r in m['mutants'].items():
            with self.subTest(name):
                self.assertEqual((r['exercised'], r['killed'], r['status'], r['guard_only'], r['value_bound']), (True, True, 'FAIL', [], m['value_bound']), r)
                if r['path'] == 'backup':self.assertEqual(r['caught_by'], ['f']);self.assertFalse(r['caught']['f']['grid_within_bound'])   # the frame is clean
                else:self.assertIn('c', r['caught_by']);self.assertEqual(set(r['guard_modes_past_turn1'].values()), {'digest_certified'})   # guard report-only
                if r['variant'] == 'single':
                    pair = r['pair'];self.assertTrue(pair['choice_reversed']);self.assertGreater(pair['abs_value_difference'], 2 * m['value_bound'])
                    self.assertGreater(min(pair['clean_gap_pricer'], pair['clean_gap_oracle']), m['value_bound']);self.assertGreater(r['caught']['f']['non_exempt_mismatches'], 0)
                    self.assertEqual(r['pairs'], 1);self.assertIn('reverses', pair['why'])
                else:self.assertGreater(r['pairs'], 10)
        cousin = m['mutants']['backup-cousin-single']['pair'];self.assertEqual((cousin['same_root_world'], cousin['fell_back_to_another_root_world']), (True, False))
        self.assertEqual(cousin['parent2'][:2], cousin['parent'][:2]);self.assertNotEqual(cousin['parent2'], cousin['parent'])
        for kind, variant in itertools.product(('sibling', 'cousin'), ('single', 'systematic')):   # both paths misplace the same pairs: (f) sees the same
            b, s = (m['mutants'][f'{path}-{kind}-{variant}']['caught']['f'] for path in ('backup', 'scatter'))
            self.assertEqual((b['non_exempt_mismatches'], b['max_abs_delta_row_q'], b['max_abs_delta_child_value']), (s['non_exempt_mismatches'], s['max_abs_delta_row_q'], s['max_abs_delta_child_value']))
        for path in gate.SWEEP_PATHS:
            sweep = m['sweep'][path];self.assertEqual((sweep['material_survivors'], sweep['claim_holds']), (0, True), sweep['first_material_survivors'][:2])
            self.assertEqual(sum(sweep['swaps'].values()), sum(m['sweep']['pairs']['drawn'].values()));self.assertGreater(sum(sweep['swaps'].values()), 5000)
        self.assertEqual(m['sweep']['scatter']['swaps'], m['sweep']['backup']['swaps'])
        messages = [x['message'] for x in lines];self.assertEqual(messages.count('MUTATION_TEST_MUTANT'), 8);self.assertEqual(messages.count('MUTATION_TEST_SWEEP'), 2)
        self.assertIn('MUTATION_TEST', messages);self.assertGreater(m['seconds'], 0.)

    def test_mutants_are_judged_at_the_runs_value_bound(self):
        """A run whose value bound (.004 here, another call's floor) exceeds this call's own (0): the clean call and every mutant are judged at it."""
        ev, m, _ = self.mutated(value_bound=.004)
        self.assertEqual((m['value_bound'], m['clean']['status']), (.004, 'PASS'));self.assertIn('the run', m['value_bound_source'])
        self.assertEqual({r['value_bound'] for r in m['mutants'].values()}, {.004});self.assertTrue(all(r['killed'] for r in m['mutants'].values()), m['survivors'])
        self.assertEqual(m['sweep']['pairs']['value_bound'], .004);self.assertEqual(m['material_survivors'], dict(scatter=0, backup=0))
        verdict, result = gate.mutant_verdict(dict(ev, actors=1), lanes=(2187,), value_bound=.004);self.assertEqual(result['tolerance']['value_bound'], .004)
        self.assertIs(gate.at_bound(ev, None), ev)

    def test_an_unpatchable_source_leaves_the_backup_mutants_unexercised(self):
        frame, data = self.frame_data()
        clean = gate.evaluate_n(frame, self._d(), self.requests, data, context={}, backups=FakeBackupsN, decision=True, choices=True)
        out = gate.mutation_test_call(frame, data, self.requests, self._d(), clean, backups=FakeBackupsN, source='def node_value():pass\n', sweep=20)
        self.assertFalse(out['passed']);self.assertEqual(sorted(out['survivors']), sorted(n for n in out['mutants'] if n.startswith('backup')))
        self.assertIn('occurs 0 times', out['mutants']['backup-sibling-single']['reason']);self.assertTrue(all(r['killed'] for n, r in out['mutants'].items() if n.startswith('scatter')))
        with self.assertRaises(gate.MutantPatchError):gate.mutant_backups([], source=FAKE_BACKUPS_SOURCE.replace('            if leaf:\n', '            if leaf :\n'))

    def test_sweep_survivor_accounting(self):
        """Bound 0: no material survivor on either path; scatter survivors are the equal-valued pairs, backup survivors those or pairs under refused
        parents. A bound above some differences, every sibling pair drawn: scatter survivors are exactly the pairs within it, backup survivors those
        plus the pairs beyond it under a refused parent, none material. The oracle holding a pair swapped (a broken clean state the swap repairs):
        a material survivor, the claim fails."""
        frame, data = self.frame_data();tree = gate.OracleTree(frame, data);count = dict(sibling=10 ** 7, cousin=2000)
        out = gate.swap_sweep(frame, data, FakeBackupsN, tree, 0., count=count, source=FAKE_BACKUPS_SOURCE)
        self.assertEqual(out['pairs']['drawn'], dict(sibling=out['pairs']['population']['sibling'], cousin=2000))
        for path in gate.SWEEP_PATHS:
            s = out[path]
            with self.subTest(path):
                self.assertEqual(sum(s['caught'].values()) + s['survivors'], sum(s['swaps'].values()));self.assertEqual(sum(s['survived'].values()), s['survivors'])
                self.assertEqual(len(s['survivor_rows']), s['survivors']);self.assertEqual((s['material_survivors'], s['claim_holds']), (0, True))
                self.assertTrue(all(r['abs_value_difference'] == 0. or not r['parent_priced'] for r in s['survivor_rows']))
        self.assertEqual(out['scatter']['survivors_parents_refused'] + sum(r['abs_value_difference'] == 0. for r in out['scatter']['survivor_rows'] if r['parent_priced']), out['scatter']['survivors'])
        self.assertGreater(out['backup']['survivors_parents_refused'], 0)
        self.assertEqual(gate.swap_sweep(frame, data, FakeBackupsN, tree, 0., count=count, source=FAKE_BACKUPS_SOURCE)['backup']['caught'], out['backup']['caught'])
        leaves = data['leaves'];index = leaves['index'].tolist();values = np.asarray(frame.leaf_value, np.float64)[tuple(leaves['index'].T)]
        scored = (np.asarray(leaves['status']) == 4) & (np.asarray(frame.leaf_status)[tuple(leaves['index'].T)] == 4);bound = .01
        groups = collections.defaultdict(list)
        for i in np.flatnonzero(scored).tolist():groups[tuple(index[i][:3])].append(values[i])
        priced = {key: FakeBackupsN.node_value(frame, key[0], key[1], 2, key[2]) is not None for key in groups}
        within = sum(abs(a - b) <= bound for g in groups.values() for a, b in itertools.combinations(g, 2))
        refused_beyond = sum(abs(a - b) > bound for key, g in groups.items() if not priced[key] for a, b in itertools.combinations(g, 2))
        wide = gate.swap_sweep(frame, data, FakeBackupsN, tree, bound, count=dict(sibling=10 ** 7), source=FAKE_BACKUPS_SOURCE)
        self.assertEqual(wide['scatter']['survived'].get('sibling', 0), within);self.assertGreater(within, out['scatter']['survived'].get('sibling', 0))
        self.assertEqual(wide['backup']['survived'].get('sibling', 0), within + refused_beyond)
        for path in gate.SWEEP_PATHS:
            self.assertTrue(all(r['abs_value_difference'] <= bound for r in wide[path]['survivor_rows'] if r['parent_priced']));self.assertEqual(wide[path]['material_survivors'], 0)
        planted = copy.deepcopy(data);head = planted['leaves']['head_value']
        i, j = next((i, j) for g in [[k for k in np.flatnonzero(scored).tolist() if tuple(index[k][:3]) == key] for key in groups if priced[key]]
            for i, j in itertools.combinations(g, 2) if head[i] != head[j])
        head[i], head[j] = head[j], head[i]
        bad = gate.swap_sweep(frame, planted, FakeBackupsN, gate.OracleTree(frame, planted), 0., count=dict(sibling=10 ** 7), paths=('scatter',))['scatter']
        self.assertFalse(bad['claim_holds']);self.assertIn(sorted([index[i], index[j]]), [sorted([r['a'], r['b']]) for r in bad['first_material_survivors']])


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

    def test_searched_on_permuted_lanes_with_errored_rows(self):
        """The permuted-lanes fixture: errored turn-2 pairs sit in rows that also have valid leaves, so the rule's ordering of
        refutations (over the kept moves, before the errored rows) is exercised against the real searched_maps."""
        class Permuted(FixtureN):
            make = staticmethod(permuted_n)
        Permuted.setUpClass();fixture = Permuted()
        for k in (1, 2, 3):
            with self.subTest(k=k):
                evaluation, summary = fixture.call(fixture.pricers(module=self.module(), selection='searched', k_own=k, k_opp=k), backups=None)
                result, (status, reasons) = self.judged(evaluation)
                self.assertEqual(status, 'PASS', (failing(result), reasons, [m for m in result['checks']['a']['first_mismatches'] if 'searched' in m['field']][:3]))
                self.assertGreater(result['checks']['a']['compared']['searched_maps_opp_rule'], 0)

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


    def test_choices_and_the_mutation_test_on_the_real_pricer(self):
        """The real pricer at depth 3 (k = 3) with --mutation-test on depthn_backups' own SOURCE: (f) passes clean, all eight mutants are killed
        (the backup path by (f) alone), neither sweep has a material survivor; a grid permuted only where depthn_backups hands it to the rule
        fails (f)'s per-child grid gate; the gate's OracleTree equals depthn_backups.node_value on every interior node of the real frame (both
        read the oracle's own values here: the real frame matches them exactly on this engine)."""
        backups = importlib.import_module('gpu_engine.depthn_backups');pricers = self.real(k_own=3, k_opp=3);module = self.module()
        depth2_pricer = gate.depth2_pass_pricer(module, self.ex, object(), capacity=32, actor_capacity=3, digest=True)
        with self.engine():
            ev, _ = gate.run_call_n(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS, pricers=pricers, depth2_pricer=depth2_pricer,
                frame_pricer=self.frame_pricer(), oracle=self.oracle, sample=None, block=0, context=dict(block=0, call=0), mutation={})
        inputs = dict(ev[gate.MUTATION_INPUTS]);result, (status, reasons) = self.judged(ev)
        (m,) = gate.mutation_tests([ev], value_bound=result['tolerance']['value_bound'])
        self.assertEqual(status, 'PASS', (failing(result), reasons));self.assertGreater(result['checks']['f']['comparisons'], 0)
        self.assertTrue(m['passed'], {n: (r.get('caught_by'), r.get('reason')) for n, r in m['mutants'].items()})
        self.assertEqual({n: r['caught_by'] for n, r in m['mutants'].items() if r['path'] == 'backup'}, {f'backup-{k}-{v}': ['f'] for k in ('sibling', 'cousin') for v in ('single', 'systematic')})
        self.assertEqual(m['material_survivors'], dict(scatter=0, backup=0))
        frame, data = inputs['frame'], inputs['data'];key = next(k for k in gate._parents(frame) if gate.pricer_record(frame, backups, *k) is not None)
        permuted = gate.mutant_backups([]);permuted._AT_RULE = lambda grid, root, world, ply, node: [row[::-1] for row in grid] if (root, world, ply, node) == key else grid
        f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=permuted), result['tolerance']['value_bound'])
        self.assertFalse(f['grid_within_bound']);self.assertIn(key, [(r['root'], r['world'], r['ply'], r['node']) for r in f['first_grid_mismatches']])
        tree = gate.OracleTree(frame, data);nodes = gate.interior_nodes(frame, data)
        for q, w, ply, node in nodes:self.assertEqual(backups.node_value(frame, q, w, ply, node), tree.solve(q, w, ply, node)[0], (q, w, ply, node))
        self.assertGreater(sum(tree.solve(*k)[0] is not None for k in nodes), 200)

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


# ---------------------------------------------------------------- the `searched` rule on random full-width depth-2 frames

def random_depth2_frame(rng, roots=4):
    """A random FULL-WIDTH depth-2 surface: expanded, terminal and errored turn-1 cells; requested or waiting seats with random
    encoder-legal sets; every expanded cell enumerates its full legal product with valid (head/terminal) and errored leaves;
    values and logits from small sets, so ties are common; some worlds without a state."""
    R = roots;f = dict(pre=np.zeros((R, 2), np.int32), root_peer_requested=np.ones((R, 2), bool), root_peer_legal=np.ones((R, 2, 9), bool),
        root_peer_logits=np.zeros((R, 2, 9), np.float32), t1_status=np.zeros((R, 2, 9, 9), np.int8), t1_value=np.zeros((R, 2, 9, 9), np.float32),
        t1_error=np.zeros((R, 2, 9, 9), np.int8), t2_status=np.zeros((R, 2, 9, 9, 9, 9), np.int8), t2_value=np.zeros((R, 2, 9, 9, 9, 9), np.float32),
        t2_error=np.zeros((R, 2, 9, 9, 9, 9), np.int8), t1_digest_own=None, t1_digest_peer=None, t2_digest=None, world_reports=(), timing={})
    for seat in ('own', 'peer'):
        f[f't1_{seat}_requested'] = np.zeros((R, 2, 9, 9), bool);f[f't1_{seat}_legal'] = np.zeros((R, 2, 9, 9, 9), bool)
        f[f't1_{seat}_logits'] = np.zeros((R, 2, 9, 9, 9), np.float32)
    values, logits = np.asarray((-.5, 0., .25, .5, 1.), np.float32), np.asarray((0., .5, 1.), np.float32)
    for q in range(R):
        for w in (0, 1):
            if rng.random() < .1:f['pre'][q, w] = -1;continue
            for a1 in range(9):
                for r1 in range(9):
                    if rng.random() < .6:continue
                    c = (q, w, a1, r1);kind = rng.random()
                    if kind < .15:f['t1_status'][c] = 1;f['t1_value'][c] = rng.choice((-1., 0., 1.));continue
                    if kind < .25:f['t1_status'][c] = 3;f['t1_error'][c] = rng.integers(1, 7);continue
                    f['t1_status'][c] = 2;f['t1_value'][c] = rng.choice(values);errored = rng.random() < .35
                    for seat in ('own', 'peer'):
                        if rng.random() < .1:continue   # a waiting seat: enumerated as action 0 only
                        f[f't1_{seat}_requested'][c] = True;lanes = rng.choice(9, size=rng.integers(1, 10), replace=False)
                        f[f't1_{seat}_legal'][c][lanes] = True;f[f't1_{seat}_logits'][c] = rng.choice(logits, size=9)
                    rows = np.flatnonzero(f['t1_own_legal'][c]) if f['t1_own_requested'][c] else [0]
                    cols = np.flatnonzero(f['t1_peer_legal'][c]) if f['t1_peer_requested'][c] else [0]
                    for a2 in rows:
                        for r2 in cols:
                            leaf = c + (a2, r2);x = rng.random()
                            if errored and x < .2:f['t2_status'][leaf] = 3;f['t2_error'][leaf] = rng.integers(1, 5)
                            elif x < .3:f['t2_status'][leaf] = 1;f['t2_value'][leaf] = rng.choice((-1., 0., 1.))
                            else:f['t2_status'][leaf] = 4;f['t2_value'][leaf] = rng.choice(values)
    return SimpleNamespace(**f)


def rule_mismatches(frame, own, opp, k_own, k_opp, extras=()):
    found = gate.check_searched_rule(frame, own, opp, k_own, k_opp, gate.Findings({}), extras).report()
    return {k: v for k, v in found['mismatched'].items() if v}, found['compared'], found['first_mismatches'][:2]


class SearchedRuleFuzzTests(unittest.TestCase):
    """The gate's re-derivation (searched_maps_reference) against a second implementation on random full-width frames, and
    against the pricer's public searched_maps when it is on the path: EXACT on every enumerated cell, errored cells included."""
    FRAMES = 40

    def frames(self):
        rng = np.random.default_rng(20261106)
        for i in range(self.FRAMES):
            k_own, k_opp = (int(x) for x in rng.integers(1, 10, size=2))
            yield i, random_depth2_frame(rng), k_own, k_opp

    def test_reference_equals_the_test_side_rule(self):
        cells = errored = 0
        for i, frame, k_own, k_opp in self.frames():
            own, opp, flag = rule_maps(frame, k_own, k_opp);bad, compared, first = rule_mismatches(frame, own, opp, k_own, k_opp, (flag,))
            self.assertEqual(bad, {}, (i, k_own, k_opp, first));cells += compared['searched_maps_own_rule']
            errored += int(((np.asarray(frame.t2_status) == 3).any(axis=(-1, -2)) & (np.asarray(frame.t1_status) == 2)).sum())
        self.assertGreater(cells, 2000);self.assertGreater(errored, 400)

    @unittest.skipUnless(HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
    def test_reference_equals_the_pricers_searched_maps(self):
        module = importlib.import_module('gpu_engine.device_depthn_pricer');cells = errored_cells = 0
        for i, frame, k_own, k_opp in self.frames():
            own, opp, *extras = module.searched_maps(frame, k_own, k_opp)
            self.assertEqual((own.shape[-1], opp.shape[-1]), gate.searched_widths_reference(k_own, k_opp))
            self.assertEqual(module.searched_widths(k_own, k_opp), gate.searched_widths_reference(k_own, k_opp))
            bad, compared, first = rule_mismatches(frame, own, opp, k_own, k_opp, tuple(extras))
            self.assertEqual(bad, {}, (i, k_own, k_opp, first));cells += compared['searched_maps_own_rule']
            errored_cells += int(((np.asarray(frame.t2_status) == 3).any(axis=(-1, -2)) & (np.asarray(frame.t1_status) == 2)).sum())
        self.assertGreater(cells, 2000);self.assertGreater(errored_cells, 400)


# ---------------------------------------------------------------- the command line

class CommandLineRunTests(FixtureN, unittest.TestCase):
    """main() end to end with the roots loader replaced by the fixture's requests (no bank): the gate, the saved frames and
    both diagnostics paths, and --n2-equivalence against the real depth-2 kernel (the stand-in or the real depth-N pricer)."""
    def roots(self):
        provider = SimpleNamespace(executor=self.ex, materializer=self.tables, recipe=self.recipe, source=self.tables.source)
        loaded = SimpleNamespace(provider=provider, plan=SimpleNamespace(model=object(), historical_window=2), frame_pricer=self.frame_pricer(), source=dict(fixture=True))
        return SimpleNamespace(loaded=loaded, head=None, params=PARAMS, groups=[list(self.requests)], root_values=[.1, .2], evaluator=dict(override=False), actor_capacity=3)

    def main(self, *argv, real=False, out=None, lines=None, source=FAKE_BACKUPS_SOURCE):
        """main() with the fixture's roots; lines: a list collecting the log lines; source: the backup-path mutants' depthn_backups
        source (the fake's unless real)."""
        out = out or Path(tempfile.mkdtemp(prefix='depthn-cli-')) / 'out';find = importlib.util.find_spec
        names = ('gpu_engine.device_depthn_pricer', 'gpu_engine.depthn_backups', 'gpu_engine.depth2_backups', 'gpu_engine.device_depth2_pricer')
        modules = {} if real else {'gpu_engine.depthn_backups': FakeBackupsN};log = (lambda m, **k: lines.append(dict(k, message=m))) if lines is not None else (lambda *a, **k: None)
        with self.engine(), patch.object(gate, 'load_roots', lambda args: self.roots()), patch.object(gate, 'log', log), \
                patch.object(gate, 'pricer_module', (lambda: importlib.import_module('gpu_engine.device_depthn_pricer')) if real else (lambda: FAKE_MODULE)), \
                patch.object(importlib.util, 'find_spec', side_effect=lambda name, *a: object() if name in names and not real else find(name, *a)), \
                patch.object(gate, 'backups_source', gate.backups_source if real else (lambda: source)), patch.dict(sys.modules, modules):
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
        self.assertEqual(report['searched_maps'][0]['widths'], dict(own=3, opp=4))
        self.assertIn('trusted via --n2-equivalence', report['depth2_pass']['trust']);self.assertIn('EXACT', report['searched_rule'])
        out = Path(tempfile.mkdtemp(prefix='depthn-cli-')) / 'diag'
        with patch.object(gate, 'log', lambda *a, **k: None), patch.dict(sys.modules, {'gpu_engine.depthn_backups': FakeBackupsN}):
            code = gate.main(['--out', str(out), '--diagnostics-from', str(prior), str(searched)])
        diagnostics = json.loads((out / 'diagnostics.json').read_text())
        self.assertEqual((code, diagnostics['calls']), (0, 1));self.assertEqual(set(diagnostics['change_rate']), {'prior-k2', 'searched-k2'})
        self.assertEqual(len(diagnostics['duplicates']), 1)   # both runs saved the depth-2 pass; the first wins
        self.assertGreater(diagnostics['prior_vs_searched']['k2']['cells'], 0)

    def test_live_guard_modes_and_reverdict(self):
        """The live gate records each (c) quantity's guard_mode with its per-position digest and actor evidence (the fixture prices
        with two parameter sets, so every guard stays gated); --reverdict reproduces the verdict on a full prior run (PASS) and a
        sampled searched run (PASS_SAMPLED), both bank-chunk mode."""
        for argv, want in (((), 'PASS'), (('--selection', 'searched', '--sample-leaves', '2'), 'PASS_SAMPLED')):
            with self.subTest(argv=argv):
                code, out = self.main(*argv);report = json.loads((out / 'gate.json').read_text());c = report['checks']['c']['quantities']
                self.assertEqual((code, report['status'], report['mode']), (0, want, 'BANK_CHUNK'), report.get('status_reasons'))
                self.assertEqual((c['leaf_value']['guard_mode'], c['ply2_value_head']['guard_mode'], c['t1_value_head']['guard_mode']), ('gated',) * 3)   # two actors
                self.assertEqual(report['tolerance']['evaluator_actors'], dict(count=len(PARAMS), source=gate.ACTORS_LIVE));self.assertEqual(len(PARAMS), 2)
                evidence = c['leaf_value']['guard_evidence'];self.assertEqual((evidence['value_positions_outside_digest'], evidence['evaluator_actors']), (0, 2))
                self.assertEqual(c['leaf_value']['beyond_guard'], 0)
                self.assertEqual(report['tolerance']['guard_depthn_rule'], gate.GUARD_DEPTHN_RULE)
                with patch.object(gate, 'log', lambda *a, **k: None):self.assertEqual(gate.main(['--reverdict', str(out / 'gate.json')]), 0)
                again = json.loads((out / gate.REVERDICT_FILE).read_text());self.assertEqual((again['status'], again['reverdict']['changed_quantities']), (want, []))
                verdicts = lambda r: {k: (m.get('passed'), m.get('bound'), m.get('guard_mode'), m.get('beyond_guard')) for k, m in r['checks']['c']['quantities'].items()}
                self.assertEqual(verdicts(again), verdicts(report))
                choices = lambda r: {k: r['checks']['f'].get(k) for k in ('passed', 'status', 'comparisons', 'near_ties', 'near_tie_flips', 'non_exempt_mismatches')}
                self.assertEqual(choices(again), choices(report));self.assertEqual(again['header'], report['header'])
                self.assertEqual(report['header']['near_ties']['f']['status'], 'COMPARED' if want == 'PASS' else report['checks']['f']['status'])

    def test_live_mutation_test_headers_and_offline_rerun(self):
        """--mutation-test --save-oracle: gate.json PASS with the header (also on DEPTHN_GATE_COMPLETE), mutation-test.json PASS with all eight
        killed and the MUTATION_TEST lines, the oracle data saved beside the frames, and --mutation-from on them reproducing every verdict."""
        lines = [];code, out = self.main('--mutation-test', '--save-oracle', lines=lines)
        report, mutation = (json.loads((out / name).read_text()) for name in ('gate.json', gate.MUTATION_FILE));call = mutation['calls'][0]
        self.assertEqual((code, report['status'], mutation['status']), (0, 'PASS', 'PASS'), (report.get('status_reasons'), call.get('survivors')))
        self.assertTrue((out / 'frames-block0-call0-oracle.npz').exists() and (out / 'frames-block0-call0-prior-k2.npz').exists())   # implies --save-arrays
        self.assertEqual({n: r['killed'] for n, r in call['mutants'].items()}, {f'{p}-{k}-{v}': True for p, k, v in gate.MUTANTS})
        self.assertGreater(sum(call['sweep']['pairs']['drawn'].values()), 0);self.assertEqual((call['material_survivors'], call['clean_unchanged']), (dict(scatter=0, backup=0), True))
        self.assertEqual((mutation['value_bound'], call['value_bound']), (report['tolerance']['value_bound'],) * 2)   # the run's bound
        messages = [x['message'] for x in lines];self.assertEqual((messages.count('MUTATION_TEST_MUTANT'), messages.count('MUTATION_TEST_SWEEP')), (8, 2))
        self.assertTrue({'MUTATION_TEST', 'MUTATION_TEST_COMPLETE'} <= set(messages))
        header = report['header'];self.assertEqual(list(report)[:4], ['evaluator', 'value_scale', 'source', 'header'])
        self.assertEqual(next(x for x in lines if x['message'] == 'DEPTHN_GATE_COMPLETE')['header'], header);self.assertEqual(mutation['header'], header)
        self.assertEqual((header['near_ties']['f']['near_ties'], header['near_ties']['f']['near_tie_flips']), (report['checks']['f']['near_ties'], report['checks']['f']['near_tie_flips']))
        self.assertEqual(header['near_ties']['e']['near_tie_roots'], report['checks']['e']['near_tie_roots']);self.assertEqual(len(header['rules_changed_after_failure']), 4)
        offline = Path(tempfile.mkdtemp(prefix='depthn-cli-')) / 'offline'
        with patch.object(gate, 'log', lambda *a, **k: None), patch.object(gate, 'backups_source', lambda: FAKE_BACKUPS_SOURCE), patch.dict(sys.modules, {'gpu_engine.depthn_backups': FakeBackupsN}):
            code = gate.main(['--out', str(offline), '--mutation-from', str(out)])
        again = json.loads((offline / gate.MUTATION_FILE).read_text());self.assertEqual((code, again['status'], len(again['calls'])), (0, 'PASS', 1))
        again = again['calls'][0];self.assertEqual({n: (r['killed'], r['caught_by']) for n, r in again['mutants'].items()}, {n: (r['killed'], r['caught_by']) for n, r in call['mutants'].items()})
        for path in gate.SWEEP_PATHS:self.assertEqual((again['sweep'][path]['caught'], again['sweep'][path]['survivors']), (call['sweep'][path]['caught'], call['sweep'][path]['survivors']))
        self.assertEqual(again['value_bound'], call['value_bound']);self.assertEqual(json.loads((offline / gate.MUTATION_FILE).read_text())['header']['near_ties'], header['near_ties'])

    def test_a_surviving_mutant_fails_the_process(self):
        """A depthn_backups source the backup-path patch cannot apply to: those four mutants are never exercised, so they survive; the clean gate
        still passes and the process exits 1."""
        code, out = self.main('--mutation-test', source='')
        report, mutation = (json.loads((out / name).read_text()) for name in ('gate.json', gate.MUTATION_FILE))
        self.assertEqual((code, report['status'], mutation['status']), (1, 'PASS', 'FAIL'))
        self.assertEqual(sorted(mutation['survivors']['block0-call0']), sorted(f'backup-{k}-{v}' for k in ('sibling', 'cousin') for v in ('single', 'systematic')))

    def test_live_diagnostics(self):
        code, out = self.main('--depth3-diagnostics', '--diagnostic-ks', '2,3')
        report = json.loads((out / 'diagnostics.json').read_text())
        self.assertEqual(code, 0);self.assertEqual(set(report['change_rate']), {'prior-k2', 'prior-k3', 'searched-k2', 'searched-k3'})
        self.assertEqual(set(report['prior_vs_searched']), {'k2', 'k3'});self.assertEqual(report['widths']['searched-k3'], ['{"k_own": [4, 3], "k_opp": [5, 3]}'])

    def test_live_diagnostics_exit_1_without_a_call(self):
        roots = self.roots;self.roots = lambda: dict.__setitem__(vars(r := roots()), 'groups', []) or r
        try:code, out = self.main('--depth3-diagnostics', '--diagnostic-ks', '2')
        finally:self.roots = roots
        self.assertEqual(code, 1);self.assertEqual(json.loads((out / 'diagnostics.json').read_text())['calls'], 0)

    def test_n2_equivalence_against_the_depth2_kernel(self):
        """The stand-in N=2 frame equals the real depth-2 kernel's on every array and decision; it carries no D world_reports, so
        the gated report comparison alone fails it (the real pricer's run below passes in full)."""
        code, out = self.main('--n2-equivalence', '--n2-k', '2')
        report = json.loads((out / 'n2-equivalence.json').read_text())
        self.assertEqual((code, report['status']), (1, 'FAIL'));self.assertEqual(set(report['configs']), {'full_width', 'top2'})
        self.assertEqual(set(report['header']['near_ties']), {'full_width', 'top2'});self.assertEqual(len(report['header']['rules_changed_after_failure']), 4)
        self.assertEqual(report['header']['near_ties']['top2']['e']['near_tie_roots'], report['configs']['top2']['e']['near_tie_roots'])
        for config in report['configs'].values():
            self.assertTrue(config['discrete']['passed'], config['discrete']['mismatched']);self.assertTrue(config['e']['passed'], config['e'].get('reasons'))
            self.assertTrue(all(m['passed'] for m in config['floats'].values()));self.assertFalse(config['world_reports_equal'])
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
                    ['--out', str(out), '--diagnostics-from', 'x', '--d3-root', 'b'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--mutation-test', '--sample-leaves', '2'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--mutation-test', '--depth', '2'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--mutation-test', '--n2-equivalence'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--save-oracle', '--depth3-diagnostics'],
                    ['--out', str(out), '--mutation-from', 'x', '--d3-root', 'b'], ['--out', str(out), '--mutation-from', 'x', '--diagnostics-from', 'y'],
                    ['--out', str(out), '--mutation-from', 'x', '--mutation-test']):
                with self.assertRaises(SystemExit) as raised:gate.main(argv)
                self.assertEqual(raised.exception.code, 2, argv)
        self.assertEqual(gate.parse(['--out', 'o', '--d3-root', 'b', '--chunk', '0', '--depth', '4', '--k-own', '3,2,1'])[1].k_own_ply, (3, 2, 1))
        self.assertTrue(gate.parse(['--out', 'o', '--d3-root', 'b', '--chunk', '0', '--save-oracle'])[1].save_arrays)   # the oracle data sits beside the frames
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
