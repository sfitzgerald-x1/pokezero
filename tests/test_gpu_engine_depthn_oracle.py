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
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

import jax
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
        tolerance=dict(rule=gate.g2.TOLERANCE_RULE, depth1_noise_floor=FLOOR, value_bound=.008, logit_bound=.1, guard_bound=.005, table=table), checks=dict(a=a, c=dict(passed=False, quantities=quantities), e=dict(passed=True, roots_compared=2),
            f=saved_choices()))


def saved_choices(differing=()):
    """A saved (f) section as the live run writes it (value bound .008): `differing` flips as (gap_pricer, gap_oracle) pairs."""
    flips = [dict(block=0, call=0, root=1, world=0, ply=2, node=7 + i, address=[1, 2, 0], kind='learner', own_action=None, actions=[1, 2], gap_pricer=gp, gap_oracle=go)
        for i, (gp, go) in enumerate(differing)]
    return dict(passed=True, status='COMPARED', nodes=5, nodes_by_ply={'1': 2, '2': 3}, comparisons=12, learner_comparisons=5, opponent_comparisons=7, not_comparable=0,
        both_unpriced=0, one_sided=0, one_sided_examples=[], problems=0, problem_examples=[], differing=flips, differing_complete=True, max_abs_delta_row_q=.003,
        max_abs_delta_child_value=.006, sampled=False, value_bound=.008, near_ties=2, near_tie_flips=0, non_exempt_mismatches=0, rule=gate.CHOICE_RULE,
        read_identity=dict(passed=True, nodes=5, entries=20, mismatched_nodes=0, one_sided=0, not_comparable=0, both_unpriced=0, interned_collisions=0, injective=True,
            first_mismatches=[], reasons=[]), priors=dict(passed=True, nodes=5, mismatched_nodes=0, one_sided=0, first_mismatches=[]))


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
            choice_flips=drop('checks', 'f', 'differing'), value_bound=drop('tolerance', 'value_bound'), read_identity=drop('checks', 'f', 'read_identity'),
            priors=drop('checks', 'f', 'priors'), rule=drop('tolerance', 'rule'), logit_bound=drop('tolerance', 'logit_bound'),
            depth1_noise_floor=drop('tolerance', 'depth1_noise_floor'), value_bound_of_a_value=drop('checks', 'c', 'quantities', 'leaf_value', 'bound'))
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
    clause, rule 1's per-node exemption and the bound's edge, one-sided pricing, sampled runs, --reverdict and the header; then the ten
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
        report = saved_gate();report['checks']['f']['read_identity'].update(passed=False, mismatched_nodes=2)   # exact: whatever the bound
        code, out, _, _ = reverdicted(report);self.assertEqual((code, out['checks']['f']['passed']), (1, False));self.assertIn('read identity', out['checks']['f']['reasons'][0])
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
        self.assertIn('not a run', rules[1]['trigger']);self.assertIn('per-child grid gate', rules[2]['summary']);self.assertIn('read identity', rules[2]['summary'])
        self.assertIn('review findings, not a run', rules[2]['trigger']);self.assertIn('kill-test sweep', rules[2]['trigger'])
        self.assertTrue(all(s in rules[0]['trigger'] for s in ('d2-gate40-b0-r1', '60a52442', 'root 15', '1.2e-3 apart in the pricer frame', '2.3e-5 apart in the oracle-heads frame', '4f88727e')))
        self.assertIn('GUARD_DEPTHN_RULE', rules[2]['name']);self.assertIn('DEEPER_LOGIT_RULE', rules[3]['name']);self.assertEqual(gate.header_block()['near_ties'], dict(e=None, f=None))
        self.assertEqual(gate.header_block(configs=dict(top2=dict(e=e)))['near_ties'], dict(top2=dict(e=e)));self.assertTrue(all(x in gate.GUARD_DEPTHN_RULE for x in ('read identity', 'per-child value gate', 'any rule')))

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
        layout = gate.tagged_frame(frame)[1];identity = lambda drop, error: gate.IdentityTree(tree(drop, error), layout, gate.Intern())   # read identity: the same order
        self.assertIs(identity([rows[0]], None).solve(*key)[0], gate.MISSING);self.assertIs(identity([rows[0]], rows[1]).solve(*key)[0], None)
        self.assertIs(identity([rows[0]], rows[1]).solve(*gate._up(frame, key))[0], None)
        sampled_out = tree([rows[0]], None)   # a leaf the oracle did not compute: unsampled, read identity fails closed; sampled, it skips the node
        hidden = gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN, tree=sampled_out)
        self.assertEqual((hidden['passed'], hidden['mismatched_nodes']), (False, 0));self.assertGreater(hidden['not_comparable'], 0)
        self.assertTrue(any('could not name' in r for r in hidden['reasons']))
        self.assertTrue(gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN, tree=sampled_out, sampled=True)['passed'])
        clean = gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN);self.assertEqual((clean['passed'], clean['not_comparable']), (True, 0))

    # ---- read identity (the re-review's C2), rules that edit their grid or use the priors, matching by action

    def interior_pair(self, depth, k, ply, want):
        """(frame, data, parent key, slot pair s, slot pair t) for the first priced parent at `ply` of the depth/k stand-in with two EXPANDED
        children (interior: ply + 1 < depth) that want(frame, key, a, b) accepts (a, b: the children's node ids at ply + 1)."""
        frame, data = self.frame_data(depth=depth, k=k);keys = gate._parents(frame) if ply >= 2 else [(q, w, 1, a1 * 9 + r1) for q, w, a1, r1 in
            np.argwhere(np.asarray(frame.t1_status) == 2).tolist()]
        k_own, k_opp = int(frame.k_own[ply - 1]), int(frame.k_opp[ply - 1]);level = frame.nodes[ply - 1]
        for key in keys:
            if key[2] != ply:continue
            record = gate.pricer_record(frame, FakeBackupsN, *key)
            if record is None or 'shape' in record:continue
            for s, t in itertools.combinations([(i, x) for i in record['own_slots'] for x in record['opp_slots']], 2):
                a, b = (key[3] * k_own * k_opp + u[0] * k_opp + u[1] for u in (s, t))
                if all(int(level.status[key[0], key[1], n]) == 2 for n in (a, b)) and want(frame, key, a, b):return frame, data, key, s, t
        self.fail('no such pair on the stand-in')

    def test_read_identity_passes_clean_frames_at_depths_3_and_4(self):
        for depth, k in ((3, 2), (3, 3), (4, 2)):
            with self.subTest(depth=depth, k=k):
                frame, data = self.frame_data(depth=depth, k=k);ri = gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN)
                self.assertEqual((ri['passed'], ri['mismatched_nodes'], ri['one_sided']), (True, 0, 0));self.assertGreater(ri['nodes'], 1000)
                self.assertEqual(ri['tags']['interned'], ri['nodes']);self.assertGreater(ri['entries'], 3 * ri['nodes'])
        f = self.judged(self.prior()[0])[0]['checks']['f'];self.assertTrue(f['read_identity']['passed']);self.assertGreater(f['read_identity']['nodes'], 1000)

    def test_read_identity_catches_an_interior_misread_the_value_gates_pass(self):
        """The re-review's test_2: depth 4, k = 3, a ply-2 parent reading two interior (ply-3) children swapped, picked where their learner-max
        values are closest relative to a min-max rule's. At a bound equal to the learner-max difference every value gate passes; read identity
        fails at the parent under the live learner-max, naming the two subtrees each entry read."""
        minmax = lambda grid, pi_own, pi_opp: max(min(row) for row in grid);B = FakeBackupsN;best = []
        def want(frame, key, a, b):
            la, lb, ma, mb = (B.node_value(frame, key[0], key[1], 3, n, **kw) for kw in ({}, {'rule': minmax}) for n in (a, b))
            if None in (la, lb, ma, mb) or ma == mb:return False
            best.append((abs(ma - mb) / max(abs(la - lb), 1e-12), key, a, b, abs(la - lb)));return False
        with self.assertRaises(AssertionError):self.interior_pair(4, 3, 2, want)   # scores every pair, accepts none
        _, key, a, b, bound = max(best, key=lambda r: r[0]);frame, data = self.frame_data(depth=4, k=3);K = int(frame.k_own[1]) * int(frame.k_opp[1])
        s, t = (((n - key[3] * K) // int(frame.k_opp[1]), (n - key[3] * K) % int(frame.k_opp[1])) for n in (a, b))
        f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=gate.mutant_backups([(key + s, key + t)], source=FAKE_BACKUPS_SOURCE)), bound)
        self.assertEqual((f['grid_within_bound'], f['row_q_within_bound'], f['non_exempt_mismatches'], f['read_identity']['passed'], f['passed']), (True, True, 0, False, False))
        first = f['read_identity']['first_mismatches'][0];self.assertEqual((first['root'], first['world'], first['ply'], first['node']), key)
        reads = {tuple(e['actions']): (e['read'], e['expected']) for e in first['entries']};self.assertEqual(len(reads), 2)
        (r1, e1), (r2, e2) = reads.values();self.assertEqual((r1, r2), (e2, e1))   # each entry read the other child's subtree
        self.assertTrue(all(r[0][:2] == ['node', key[0]] and r[0][3] == 3 and r[1][0] == 'node' for r in reads.values()))   # interned tags named as oracle nodes
        self.assertTrue(first['origin']);self.assertTrue(any('read identity' in r for r in f['reasons']))
        self.assertEqual((f['read_identity']['injective'], f['read_identity']['interned_collisions']), (True, 0))   # a swap repeats no read

    def test_read_identity_catches_a_turn1_misread_under_min_max(self):
        """Depth 3: a turn-1 cell reading two interior (ply-2) children swapped whose min-max values are EQUAL (subtrees differ): under the
        min-max rule every value gate passes; read identity fails at the cell."""
        minmax = lambda grid, pi_own, pi_opp: max(min(row) for row in grid);B = FakeBackupsN
        frame, data, key, s, t = self.interior_pair(3, 3, 1, lambda frame, key, a, b: B.node_value(frame, key[0], key[1], 2, a, rule=minmax) is not None
            and B.node_value(frame, key[0], key[1], 2, a, rule=minmax) == B.node_value(frame, key[0], key[1], 2, b, rule=minmax))
        module = gate.mutant_backups([(key + s, key + t)], source=FAKE_BACKUPS_SOURCE)
        f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=module, rule=minmax, tree=gate.OracleTree(frame, data, rule=minmax)), 0.)
        self.assertEqual((f['grid_within_bound'], f['row_q_within_bound'], f['read_identity']['passed'], f['passed']), (True, True, False, False))
        self.assertIn(key, [(r['root'], r['world'], r['ply'], r['node']) for r in f['read_identity']['first_mismatches']])

    def test_read_identity_guards_injectivity_and_one_to_one(self):
        """INJECTIVITY: a tagged copy (wrongly) cast to float32 collapses the tags: the distinct-count guard fails, gated, not a pass. INTERIOR
        IDENTITY: a backup handing a second node the first node's grid gives two nodes one interned tag: the one-to-one guard fails."""
        frame, data = self.frame_data();clean = gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN)
        self.assertEqual((clean['injective'], clean['interned_collisions'], clean['distinct_captured_tags']), (True, 0, clean['distinct_named_reads']))
        real = gate.tagged_frame
        def float32(frame, **kw):
            tagged, layout = real(frame, **kw)
            return gate._copy(tagged, leaf_value=np.asarray(tagged.leaf_value, np.float32), nodes=tuple(gate._copy(level, value=np.asarray(level.value, np.float32))
                for level in tagged.nodes)), layout
        with patch.object(gate, 'tagged_frame', float32):ri = gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN)
        self.assertEqual((ri['passed'], ri['injective']), (False, False));self.assertLess(ri['distinct_captured_tags'], ri['distinct_named_reads'])
        self.assertTrue(any('injectivity' in r for r in ri['reasons']))
        f = gate.choice_verdict(dict(saved_choices(), read_identity=ri), .008);self.assertFalse(f['passed']);self.assertTrue(any('injectivity' in r for r in f['reasons']))
        first, second = [k for k in gate._parents(frame) if gate.pricer_record(frame, FakeBackupsN, *k) is not None][:2];held = {}
        def hand_over(grid, root, world, ply, node):
            if (root, world, ply, node) == first:held['grid'] = grid
            return held['grid'] if (root, world, ply, node) == second and 'grid' in held else grid
        module = gate.mutant_backups([], source=FAKE_BACKUPS_SOURCE);module._AT_RULE = hand_over
        ri = gate.read_identity_n(frame, data, self.requests, backups=module)
        self.assertFalse(ri['passed']);self.assertGreaterEqual(ri['interned_collisions'], 1);self.assertIn([list(first), list(second)], [c['nodes'] for c in ri['first_collisions']])
        self.assertTrue(any('one-to-one' in r for r in ri['reasons']))


    def test_tags_are_exact_and_overflow_fails_closed(self):
        from fractions import Fraction
        for k in (0, 1, 12345, gate.TAG_CAPACITY - 1):
            tag = gate.position_tag(k);self.assertEqual(Fraction(tag), Fraction(-1) + Fraction(k + 1, 2 ** 42));self.assertTrue(-1. < tag < 0.)
            tag = gate.interned_tag(k);self.assertEqual(Fraction(tag), Fraction(k + 1, 2 ** 42));self.assertTrue(0. < tag < 1.)
        frame, data = self.frame_data(depth=4, k=2);tagged, layout = gate.tagged_frame(frame)
        arrays = [np.asarray(tagged.leaf_value)] + [np.asarray(level.value) for level in tagged.nodes]
        self.assertTrue(all(a.dtype == np.float64 for a in arrays));flat = np.concatenate([a.reshape(-1) for a in arrays])
        self.assertEqual(len(np.unique(flat)), len(flat));self.assertTrue(((flat > -1.) & (flat < 0.)).all())
        self.assertEqual(gate.decode_tag(float(tagged.leaf_value[3, 1, 5, 1, 0]), layout), ['leaf', 3, 1, 5, 1, 0])
        self.assertEqual(gate.decode_tag(float(tagged.nodes[1].value[2, 0, 17]), layout), ['ply3', 2, 0, 17]);self.assertEqual(gate.decode_tag(gate.interned_tag(4), layout), ['grid', 4])
        self.assertIs(tagged.leaf_status, frame.leaf_status);self.assertIs(tagged.nodes[0].own_actions, frame.nodes[0].own_actions)   # statuses and maps untouched
        positions = len(flat);ri = gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN, capacity=positions - 1)
        self.assertEqual(ri['passed'], False);self.assertIn('fails closed', ri['reason']);self.assertTrue(gate.read_identity_n(frame, data, self.requests, backups=FakeBackupsN,
            capacity=positions)['passed'])
        intern = gate.Intern(capacity=2);intern.tag([[.1]]);intern.tag([[.2]]);self.assertEqual(intern.tag([[.1]]), gate.interned_tag(0))
        with self.assertRaises(gate.TagOverflow):intern.tag([[.3]])
        f = gate.choice_verdict(dict(saved_choices(), read_identity=dict(ri)), .008);self.assertFalse(f['passed']);self.assertIn('fails closed', f['reasons'][0])

    def test_an_in_place_sorting_rule_neither_false_fails_nor_blinds(self):
        """The re-review's rule that sorts its grid's rows IN PLACE before solving: the clean backup passes (the recorder snapshots before the rule,
        the oracle hands the rule its own copy) and a within-row misread fails."""
        def sorting(grid, pi_own, pi_opp):
            for row in grid:row.sort()
            return max(sum(row) / len(row) for row in grid)
        frame, data = self.frame_data(k=3);tree = gate.OracleTree(frame, data, rule=sorting)
        clean = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=FakeBackupsN, rule=sorting, tree=tree), 0.)
        self.assertTrue(clean['passed'], clean['reasons']);self.assertEqual(clean['max_abs_delta_child_value'], 0.)
        _, _, key, record, (row, x, y) = self.find(3, lambda r, row, x, y: r['grid'][row][x] != r['grid'][row][y])
        bad = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=self.swap_in_row(key, record, row, x, y), rule=sorting, tree=tree), 0.)
        self.assertFalse(bad['passed']);self.assertFalse(bad['grid_within_bound']);self.assertFalse(bad['read_identity']['passed'])

    def test_a_prior_using_rule_solves_alike_on_both_sides(self):
        """A rule that weighs rows by pi_own and columns by pi_opp (depthn_backups' diagnostic average): both sides hand it the same priors in the
        same roles, so the clean grids certify; the priors' roles matter on this surface (swapping them moves some node's value)."""
        average = lambda grid, pi_own, pi_opp: math.fsum(p * math.fsum(q * v for q, v in zip(pi_opp, row)) for p, row in zip(pi_own, grid))
        swapped = lambda grid, pi_own, pi_opp: average(grid, pi_opp, pi_own)
        frame, data = self.frame_data(k=3)
        clean = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=FakeBackupsN, rule=average, tree=gate.OracleTree(frame, data, rule=average)), 0.)
        self.assertTrue(clean['passed'], clean['reasons']);self.assertGreater(clean['nodes'], 1000)
        self.assertTrue(any(FakeBackupsN.node_value(frame, *k, rule=average) != FakeBackupsN.node_value(frame, *k, rule=swapped)
            for k in gate.interior_nodes(frame, data) if FakeBackupsN.node_value(frame, *k) is not None))

    def test_records_are_matched_by_action_not_slot(self):
        """The oracle's record with its opponent columns in the reverse slot order (the same actions): no grid, row q or choice difference."""
        frame, data = self.frame_data(k=3);key = next(k for k in gate._parents(frame) if (lambda r: r is not None and len(r['opp']) == 3 and len(set(r['grid'][0])) == 3)(
            gate.pricer_record(frame, FakeBackupsN, *k)))
        mine = gate.pricer_record(frame, FakeBackupsN, *key)
        theirs = gate._solve([row[::-1] for row in mine['grid']], mine['pi_opp'][::-1], mine['own_slots'], mine['opp_slots'][::-1], mine['own'], mine['opp'][::-1])
        comparisons, problems, dq, dv = gate.compare_records(mine, theirs)
        self.assertEqual((problems, dq, dv), ([], 0., 0.));self.assertTrue(all(got == want for _, _, got, want, _, _ in comparisons))
        theirs['pi_own'] = mine['pi_own'];self.assertEqual(gate.compare_priors(mine, theirs), [])   # priors by action too (the opponent's reversed)
        self.assertNotEqual(gate.compare_priors(mine, dict(theirs, pi_opp=mine['pi_opp'])), [])

    # ---- the exact prior check (the re-review of 27adf5a7), merged gating and diagnostics

    def wrong_priors(self, new):
        """The fake backups' source with node_value's pi_own replaced by `new` (the re-review's probes)."""
        line = 'pi_own=_softmax(own_logits,[int(own_actions[i])for i in own_slots])';self.assertEqual(FAKE_BACKUPS_SOURCE.count(line), 1)
        module = types.ModuleType('wrong_priors');exec(compile(FAKE_BACKUPS_SOURCE.replace(line, new), '<wrong priors>', 'exec'), module.__dict__);return module

    def test_the_priors_handed_to_the_rule_are_exact(self):
        """Clean frames pass the exact prior check at depths 2 to 4 (k 1 to 3: single-slot maps; the fixture's waiting seats); a backup handing its
        rule a uniform pi_own, or one built from the opponent's logits, fails it alone (grid, row q and read identity pass: learner-max ignores
        pi_own), and under depthn_backups' diagnostic average rule it moves values."""
        for depth, k in ((2, 3), (3, 1), (3, 3), (4, 2)):
            with self.subTest(depth=depth, k=k):
                ev, _ = self.call(self.pricers(depth=depth, k_own=k, k_opp=k), decision=False);result = self.judged(ev)[0];f = result['checks']['f']
                self.assertEqual(self.judged(ev)[1][0], 'PASS', failing(result));self.assertEqual((f['priors']['passed'], f['priors']['mismatched_nodes']), (True, 0))
                self.assertGreater(f['priors']['nodes'], 100)
        frame, data = self.frame_data(k=3);records = [gate.pricer_record(frame, FakeBackupsN, *key) for key in gate.interior_nodes(frame, data)]
        self.assertTrue(any(r and 'shape' not in r and len(r['own']) == 1 for r in records));self.assertTrue(any(r and 'shape' not in r and len(r['opp']) == 1 for r in records))
        average = lambda grid, pi_own, pi_opp: math.fsum(p * math.fsum(q * v for q, v in zip(pi_opp, row)) for p, row in zip(pi_own, grid))
        for name, new in (('uniform', 'pi_own=[1./len(own_slots)]*len(own_slots)'), ('opponent logits', 'pi_own=_softmax(opp_logits,[int(own_actions[i])for i in own_slots])')):
            with self.subTest(name):
                module = self.wrong_priors(new);f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=module), 0.)
                self.assertEqual((f['passed'], f['priors']['passed'], f['grid_within_bound'], f['row_q_within_bound'], f['read_identity']['passed']), (False, False, True, True, True))
                self.assertGreater(f['priors']['mismatched_nodes'], 100);self.assertEqual(f['priors']['first_mismatches'][0]['seats'][0]['seat'], 'own')
                avg = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=module, rule=average, tree=gate.OracleTree(frame, data, rule=average)), 0.)
                self.assertFalse(avg['grid_within_bound']);self.assertFalse(avg['priors']['passed'])
        mine = next(r for r in records if r and 'shape' not in r and len(r['own']) > 1)
        self.assertEqual(gate.compare_priors(dict(mine, pi_own=None), mine)[0]['pricer'], None)   # a missing prior is a mismatch
        self.assertEqual(gate.compare_priors(mine, dict(mine)), [])
        line = 'pi_opp=_softmax(opp_logits,[int(opp_actions[x])for x in opp_slots])'   # the opponent's seat is checked as exactly (the value gates aside)
        self.assertEqual(FAKE_BACKUPS_SOURCE.count(line), 1);module = types.ModuleType('wrong_opp')
        exec(compile(FAKE_BACKUPS_SOURCE.replace(line, 'pi_opp=[1./len(opp_slots)]*len(opp_slots)'), '<wrong pi_opp>', 'exec'), module.__dict__)
        f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=module), 0.)
        self.assertFalse(f['priors']['passed']);self.assertEqual({x['seat'] for m in f['priors']['first_mismatches'] for x in m['seats']}, {'opp'})
        ulp = self.wrong_priors('pi_own=[math.nextafter(p,2.)for p in _softmax(own_logits,[int(own_actions[i])for i in own_slots])]')   # one ulp: exact, no tolerance
        f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=ulp), 0.);self.assertFalse(f['priors']['passed'])
        self.assertEqual((f['grid_within_bound'], f['read_identity']['passed']), (True, True))

    def test_priors_are_checked_at_turn1_cells_and_at_leaf_parents(self):
        """A backup handing a uniform pi_own ONLY at the turn-1 cells, or ONLY at the parents of leaves (depthn_backups' _PRIORS hook): learner-max
        values, the grid and read identity are unchanged, and the exact prior check fails at exactly those plies."""
        frame, data = self.frame_data(k=3);uniform = lambda pi: [1. / len(pi)] * len(pi)
        for name, at in (('turn-1 cells', lambda ply: ply == 1), ('leaf parents', lambda ply: ply + 1 == int(frame.depth))):
            with self.subTest(name):
                module = gate.mutant_backups([], source=FAKE_BACKUPS_SOURCE);module._PRIORS = lambda pi, root, world, ply, node, at=at: uniform(pi) if at(ply) else pi
                f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=module), 0.)
                self.assertEqual((f['passed'], f['priors']['passed'], f['grid_within_bound'], f['row_q_within_bound'], f['read_identity']['passed']), (False, False, True, True, True))
                self.assertGreater(f['priors']['mismatched_nodes'], 100);self.assertTrue(f['priors']['first_mismatches'])
                self.assertTrue(all(at(m['ply']) for m in f['priors']['first_mismatches']), [m['ply'] for m in f['priors']['first_mismatches']])

    def test_the_expected_prior_is_the_gates_own_softmax(self):
        """A gpu_engine.depthn_backups whose _softmax is wrong (twice the logit scale) installed where the gate imports the backups: the backup
        built from it hands wrong priors and the exact prior check FAILS, the gate's expectation being its own prior_softmax, never depthn_backups';
        the correct backup passed beside it still passes."""
        line = 'weights=[math.exp(float(logits[a])-top)for a in actions]';self.assertEqual(FAKE_BACKUPS_SOURCE.count(line), 1)
        wrong = types.ModuleType('gpu_engine.depthn_backups')
        exec(compile(FAKE_BACKUPS_SOURCE.replace(line, 'weights=[math.exp(2.*(float(logits[a])-top))for a in actions]'), '<wrong softmax>', 'exec'), wrong.__dict__)
        frame, data = self.frame_data(k=3)
        with patch.dict(sys.modules, {'gpu_engine.depthn_backups': wrong}):
            bad = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests), 0.)   # backups=None: the installed module
            good = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=FakeBackupsN), 0.)
        self.assertEqual((bad['passed'], bad['priors']['passed']), (False, False));self.assertGreater(bad['priors']['mismatched_nodes'], 100)
        self.assertTrue(good['passed'], good['reasons']);self.assertEqual(good['priors']['mismatched_nodes'], 0)
        self.assertEqual(gate.prior_softmax([0., .5, 1.], [0, 2, 1]), FakeBackupsN._softmax([0., .5, 1.], [0, 2, 1]))
        self.assertNotEqual(gate.prior_softmax([0., .5, 1.], [0, 2, 1]), wrong._softmax([0., .5, 1.], [0, 2, 1]))

    def test_a_rule_editing_its_priors_in_place_changes_nothing_recorded(self):
        """A rule that edits pi_own and pi_opp in place after solving (reversed, zeroed): the recorder copied both before the rule ran and the
        oracle hands the rule copies, so the correct backup passes the prior check and every gate, and a misread still fails."""
        def editing(grid, pi_own, pi_opp):
            value = max(math.fsum(p * v for p, v in zip(pi_opp, row)) for row in grid)   # learner-max on the priors as handed
            pi_own.reverse();pi_opp[:] = [0.] * len(pi_opp);return value
        frame, data = self.frame_data(k=3);tree = gate.OracleTree(frame, data, rule=editing)
        clean = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=FakeBackupsN, rule=editing, tree=tree), 0.)
        self.assertTrue(clean['passed'], clean['reasons']);self.assertEqual((clean['priors']['passed'], clean['priors']['mismatched_nodes']), (True, 0))
        self.assertGreater(clean['priors']['nodes'], 1000)
        _, _, key, record, (row, x, y) = self.find(3, lambda r, row, x, y: r['grid'][row][x] != r['grid'][row][y])
        bad = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=self.swap_in_row(key, record, row, x, y), rule=editing, tree=tree), 0.)
        self.assertFalse(bad['passed']);self.assertTrue(bad['priors']['passed'])   # a misread, not a prior: the priors handed stay right


    def test_read_identity_and_priors_are_gated_on_their_counts_too(self):
        """Defence in depth: a merged read identity ANDs its calls; (f) fails on a missing or null read identity or priors (a saved null fails
        --reverdict), and on mismatched_nodes, interned_collisions or a false injective flag even when `passed` says otherwise."""
        ok, bad = dict(saved_choices()['read_identity']), dict(saved_choices()['read_identity'], passed=False, mismatched_nodes=1)
        self.assertFalse(gate.merge_read_identity([ok, bad])['passed']);self.assertTrue(gate.merge_read_identity([ok, ok])['passed'])
        two = [dict(saved_choices(), read_identity=ok, _gaps=np.zeros(0)), dict(saved_choices(), read_identity=bad, _gaps=np.zeros(0))]
        self.assertFalse(gate.merge_choices(two, value_bound=.008)['passed'])
        self.assertFalse(gate.merge_choices([dict(two[0], priors=dict(two[0]['priors'], passed=False)), two[0]], value_bound=.008)['passed'])
        for name, change in (('missing read identity', dict(read_identity=None)), ('read identity a list', dict(read_identity=[])), ('missing priors', dict(priors=None)),
                ('mismatched nodes', dict(read_identity=dict(ok, mismatched_nodes=3))), ('collisions', dict(read_identity=dict(ok, interned_collisions=1))),
                ('not injective', dict(read_identity=dict(ok, injective=False))), ('injective unrecorded', dict(read_identity={k: v for k, v in ok.items() if k != 'injective'})),
                ('prior mismatches', dict(priors=dict(saved_choices()['priors'], mismatched_nodes=2))), ('prior one-sided', dict(priors=dict(saved_choices()['priors'], one_sided=1)))):
            with self.subTest(name):self.assertFalse(gate.choice_verdict(dict(saved_choices(), **change), .008)['passed'])
        self.assertIn('no read identity recorded', gate.choice_verdict(dict(saved_choices(), read_identity=None), .008)['reasons'])   # named, not only failed
        for key in ('read_identity', 'priors'):
            with self.subTest(saved_null=key):
                report = saved_gate();report['checks']['f'][key] = None;code, out, _, _ = reverdicted(report)
                self.assertEqual((code, out['status'], out['checks']['f']['passed']), (1, 'FAIL', False))

    def test_read_identity_names_nodes_and_marks_the_origin(self):
        """A ply-2 parent under a priced cell reads two leaves swapped: the parent is the origin (its leaves named), the cell above it is not (its
        entry read a grid no oracle node has, where the parent was expected)."""
        frame, data = self.frame_data();tree = gate.OracleTree(frame, data)
        key = next(k for k in gate._parents(frame) if gate.pricer_record(frame, FakeBackupsN, *k) is not None and tree.solve(*gate._up(frame, k))[0] is not None
            and len(set(map(tuple, np.asarray(frame.leaf_value)[k[:2] + k[3:]].tolist()))) > 1)
        module = gate.mutant_backups([(key + (0, 0), key + (1, 0))], source=FAKE_BACKUPS_SOURCE)
        ri = gate.read_identity_n(frame, data, self.requests, backups=module);first = ri['first_mismatches']
        self.assertEqual((ri['mismatched_nodes'], ri['origins']), (2, 1))
        self.assertEqual([(m['origin'], (m['root'], m['world'], m['ply'], m['node'])) for m in first], [(True, key), (False, gate._up(frame, key))])
        self.assertTrue(all(e['read'][0] == 'leaf' and e['expected'][0] == 'leaf' for e in first[0]['entries']))
        (cell,) = first[1]['entries'];self.assertEqual((cell['read'][0], cell['read'][-1], cell['expected']), ('grid', 'no oracle node', ['node'] + list(key)))


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
        self.assertEqual(len(m['mutants']), 10);self.assertEqual(m['mutants']['backup-turn1-systematic']['caught']['f']['read_identity']['passed'], False)
        priors = m['mutants']['backup-priors-systematic']['caught']['f']   # under learner-max no value moves: the exact prior check alone
        self.assertEqual((priors['priors']['passed'], priors['grid_within_bound'], priors['row_q_within_bound'], priors['read_identity']['passed']), (False, True, True, True))
        self.assertGreater(priors['priors']['mismatched_nodes'], 100)
        for name, r in m['mutants'].items():
            with self.subTest(name):
                self.assertEqual((r['exercised'], r['killed'], r['status'], r['guard_only'], r['value_bound']), (True, True, 'FAIL', [], m['value_bound']), r)
                if r['path'] == 'backup' and r['kind'] != 'priors':   # the frame is clean: only (f) sees a misread, by read identity and the grid gate
                    self.assertEqual(r['caught_by'], ['f']);self.assertFalse(r['caught']['f']['grid_within_bound']);self.assertFalse(r['caught']['f']['read_identity']['passed'])
                elif r['path'] == 'backup':self.assertEqual(r['caught_by'], ['f'])
                else:   # the reads are right (read identity passes); the values moved: (c), the guard report-only
                    self.assertIn('c', r['caught_by']);self.assertEqual(set(r['guard_modes_past_turn1'].values()), {'digest_certified'})
                    self.assertTrue((r['caught'].get('f') or {}).get('read_identity', dict(passed=True))['passed'])
                if 'f' in r['caught']:self.assertEqual(r['caught']['f']['read_identity']['interned_collisions'], 0)
                if r['variant'] == 'single':
                    pair = r['pair'];self.assertTrue(pair['choice_reversed']);self.assertGreater(pair['abs_value_difference'], 2 * m['value_bound'])
                    self.assertGreater(min(pair['clean_gap_pricer'], pair['clean_gap_oracle']), m['value_bound']);self.assertGreater(r['caught']['f']['non_exempt_mismatches'], 0)
                    self.assertEqual(r['pairs'], 1);self.assertIn('reverses', pair['why'])
                else:self.assertGreater(r['pairs'], 10) if r['kind'] != 'priors' else self.assertEqual((r['pairs'], r['exercised']), (0, True))
        cousin = m['mutants']['backup-cousin-single']['pair'];self.assertEqual((cousin['same_root_world'], cousin['fell_back_to_another_root_world']), (True, False))
        self.assertEqual(cousin['parent2'][:2], cousin['parent'][:2]);self.assertNotEqual(cousin['parent2'], cousin['parent'])
        for kind, variant in itertools.product(('sibling', 'cousin'), ('single', 'systematic')):   # both paths misplace the same pairs: (f) sees the same
            b, s = (m['mutants'][f'{path}-{kind}-{variant}']['caught']['f'] for path in ('backup', 'scatter'))
            self.assertEqual((b['non_exempt_mismatches'], b['max_abs_delta_row_q'], b['max_abs_delta_child_value']), (s['non_exempt_mismatches'], s['max_abs_delta_row_q'], s['max_abs_delta_child_value']))
        for path in gate.SWEEP_PATHS:
            sweep = m['sweep'][path];self.assertEqual((sweep['material_survivors'], sweep['claim_holds']), (0, True), sweep['first_material_survivors'][:2])
            self.assertEqual(sum(sweep['swaps'].values()), sum(m['sweep']['pairs']['drawn'].values()));self.assertGreater(sum(sweep['swaps'].values()), 5000)
        self.assertEqual(m['sweep']['scatter']['swaps'], m['sweep']['backup']['swaps'])
        messages = [x['message'] for x in lines];self.assertEqual(messages.count('MUTATION_TEST_MUTANT'), 10);self.assertEqual(messages.count('MUTATION_TEST_SWEEP'), 2)
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
        head[i], head[j] = head[j], head[i];d = abs(float(head[i]) - float(head[j]))
        for planted_bound in (0., d / 3):   # material at 2 x bound: d / 3 puts the pair between 2 x and 4 x the bound
            bad = gate.swap_sweep(frame, planted, FakeBackupsN, gate.OracleTree(frame, planted), planted_bound, count=dict(sibling=10 ** 7), paths=('scatter',))['scatter']
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
        """The real pricer at depth 3 (k = 3) with --mutation-test on depthn_backups' own SOURCE: (f) passes clean, all ten mutants are killed
        (the backup path by (f) alone, read identity included), neither sweep has a material survivor; read identity passes on the clean real
        frame; a grid permuted only where depthn_backups hands it to the rule fails (f)'s per-child grid gate and read identity; the gate's
        OracleTree equals depthn_backups.node_value on every interior node of the real frame (both
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
        self.assertEqual({n: r['caught_by'] for n, r in m['mutants'].items() if r['path'] == 'backup'}, {f'{p}-{k}-{v}': ['f'] for p, k, v in gate.MUTANTS if p == 'backup'})
        self.assertTrue(all(not r['caught']['f']['read_identity']['passed'] for r in m['mutants'].values() if r['path'] == 'backup' and r['kind'] != 'priors'))
        self.assertFalse(m['mutants']['backup-priors-systematic']['caught']['f']['priors']['passed'])
        self.assertEqual(result['checks']['f']['priors']['passed'], True);self.assertGreater(result['checks']['f']['priors']['nodes'], 200)
        self.assertEqual(m['material_survivors'], dict(scatter=0, backup=0));identity = result['checks']['f']['read_identity']
        self.assertEqual((identity['passed'], identity['injective'], identity['interned_collisions']), (True, True, 0));self.assertGreater(identity['nodes'], 200)
        frame, data = inputs['frame'], inputs['data'];key = next(k for k in gate._parents(frame) if gate.pricer_record(frame, backups, *k) is not None)
        permuted = gate.mutant_backups([]);permuted._AT_RULE = lambda grid, root, world, ply, node: [row[::-1] for row in grid] if (root, world, ply, node) == key else grid
        f = gate.choice_verdict(gate.choice_check_n(frame, data, self.requests, backups=permuted), result['tolerance']['value_bound'])
        self.assertFalse(f['grid_within_bound']);self.assertIn(key, [(r['root'], r['world'], r['ply'], r['node']) for r in f['first_grid_mismatches']])
        self.assertFalse(f['read_identity']['passed']);self.assertIn(key, [(r['root'], r['world'], r['ply'], r['node']) for r in f['read_identity']['first_mismatches']])
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

    def main(self, *argv, real=False, out=None, lines=None, source=FAKE_BACKUPS_SOURCE, module=None):
        """main() with the fixture's roots; lines: a list collecting the log lines; source: the backup-path mutants' depthn_backups
        source (the fake's unless real); module: the pricer module main() gets (default: the real one if real, else the stand-in)."""
        out = out or Path(tempfile.mkdtemp(prefix='depthn-cli-')) / 'out';find = importlib.util.find_spec
        names = ('gpu_engine.device_depthn_pricer', 'gpu_engine.depthn_backups', 'gpu_engine.depth2_backups', 'gpu_engine.device_depth2_pricer')
        modules = {} if real else {'gpu_engine.depthn_backups': FakeBackupsN};log = (lambda m, **k: lines.append(dict(k, message=m))) if lines is not None else (lambda *a, **k: None)
        chosen = (lambda: module) if module is not None else (lambda: importlib.import_module('gpu_engine.device_depthn_pricer')) if real else (lambda: FAKE_MODULE)
        with self.engine(), patch.object(gate, 'load_roots', lambda args: self.roots()), patch.object(gate, 'log', log), \
                patch.object(gate, 'pricer_module', chosen), \
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
        """--mutation-test --save-oracle: gate.json PASS with the header (also on DEPTHN_GATE_COMPLETE), mutation-test.json PASS with all ten
        killed and the MUTATION_TEST lines, the oracle data saved beside the frames, and --mutation-from on them reproducing every verdict."""
        lines = [];code, out = self.main('--mutation-test', '--save-oracle', lines=lines)
        report, mutation = (json.loads((out / name).read_text()) for name in ('gate.json', gate.MUTATION_FILE));call = mutation['calls'][0]
        self.assertEqual((code, report['status'], mutation['status']), (0, 'PASS', 'PASS'), (report.get('status_reasons'), call.get('survivors')))
        self.assertTrue((out / 'frames-block0-call0-oracle.npz').exists() and (out / 'frames-block0-call0-prior-k2.npz').exists())   # implies --save-arrays
        self.assertEqual({n: r['killed'] for n, r in call['mutants'].items()}, {f'{p}-{k}-{v}': True for p, k, v in gate.MUTANTS});self.assertEqual(len(call['mutants']), 10)
        self.assertGreater(sum(call['sweep']['pairs']['drawn'].values()), 0);self.assertEqual((call['material_survivors'], call['clean_unchanged']), (dict(scatter=0, backup=0), True))
        self.assertEqual((mutation['value_bound'], call['value_bound']), (report['tolerance']['value_bound'],) * 2)   # the run's bound
        messages = [x['message'] for x in lines];self.assertEqual((messages.count('MUTATION_TEST_MUTANT'), messages.count('MUTATION_TEST_SWEEP')), (10, 2))
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
        self.assertEqual(sorted(mutation['survivors']['block0-call0']), sorted(f'{p}-{k}-{v}' for p, k, v in gate.MUTANTS if p == 'backup'))

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


# ---------------------------------------------------------------- --fixed-shapes

def recording(base):
    """(module, built): base's pricer surface with DepthNPricer wrapped to record every constructor's keyword arguments."""
    built = []
    def make(executor, model, **kw):
        built.append(dict(kw));return base.DepthNPricer(executor, model, **kw)
    return SimpleNamespace(DepthNPricer=make, as_depth2_frame=base.as_depth2_frame, searched_maps=base.searched_maps), built


def saved_arrays(path):
    """A saved .npz's arrays without its meta (timing and compile records differ between runs)."""
    with np.load(path, allow_pickle=False) as saved:return {k: saved[k] for k in saved.files if k != 'meta'}


class FixedShapesTests(FixtureN, unittest.TestCase):
    """--fixed-shapes on the fixture split into two calls: --capacity 16 over the first 19 roots is 16 requests (31 world rows) then 3 (6).
    With the flag every DepthNPricer is built with fixed_capacity=True and both oracle stacks have 2 x 16 = 32 rows; without it the stacks
    bucket to 32 and 8 rows and nothing is built, logged or written differently. The verdict and every exact field are the same."""
    USED = 19
    main = CommandLineRunTests.main

    def roots(self):
        roots = CommandLineRunTests.roots(self);roots.groups = [list(self.requests[:self.USED])];return roots

    @contextlib.contextmanager
    def stacks(self):
        """Every oracle stack the gate builds (run_call_n, n2_equivalence_call; not the stand-in pricer's own): its size argument, world
        rows and the leading dimension of every stacked array."""
        original, seen = reference.stack, []
        def stack(requests, worlds, **kw):
            s = original(requests, worlds, **kw)
            if sys._getframe(1).f_code.co_name in ('run_call_n', 'n2_equivalence_call'):
                seen.append(dict(size=kw.get('size'), rows=len(s.rows), shape={int(x.shape[0]) for x in jax.tree.leaves((s.states, s.publics, s.owners))}))
            return s
        with patch.object(reference, 'stack', stack):yield seen

    def test_the_flag_reaches_every_pricer_and_every_oracle_stack(self):
        for flag in (True, False):
            with self.subTest(fixed_shapes=flag):
                module, built = recording(FAKE_MODULE);lines = []
                with self.stacks() as seen:code, out = self.main('--capacity', '16', *(('--fixed-shapes',) if flag else ()), lines=lines, module=module)
                report = json.loads((out / 'gate.json').read_text());start = next(x for x in lines if x['message'] == 'GATE_START')
                self.assertEqual((code, report['status'], report['split_capacity'], len(report['calls'])), (0, 'PASS', 16, 2), report.get('status_reasons'))
                self.assertEqual([(s['rows'], s['shape']) for s in seen], [(31, {32}), (6, {32 if flag else 8})])
                self.assertEqual(len(built), 2)   # the depth-N pricer and the full-width depth-2 pass
                if flag:
                    self.assertTrue(all(kw.get('fixed_capacity') is True for kw in built), built);self.assertEqual([s['size'] for s in seen], [32, 32])
                    want = dict(fixed_shapes=True, pricer_fixed_capacity=True, oracle_stack_rows=32, split_capacity=16, rule=gate.FIXED_SHAPES)
                    self.assertEqual((report['harness'], start['harness']), (want, want));self.assertIs(report['pricer_options']['fixed_capacity'], True)
                else:
                    self.assertFalse(any('fixed_capacity' in kw for kw in built), built);self.assertEqual([s['size'] for s in seen], [None, None])
                    self.assertNotIn('harness', report);self.assertNotIn('harness', start);self.assertNotIn('fixed_capacity', report['pricer_options'])

    def test_n2_equivalence_refuses_and_diagnostics_take_the_flag(self):
        """--n2-equivalence refuses the flag (exit 2, nothing created: its Depth2Pricer keeps per-call buckets, so the compared pricers would
        price turn 1 at different batch sizes); --depth3-diagnostics: every DepthNPricer fixed, no oracle (oracle_stack_rows None)."""
        out = str(Path(tempfile.gettempdir()) / f'depthn-gate-never-created-{uuid.uuid4().hex}')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            gate.main(['--out', out, '--d3-root', 'b', '--chunk', '0', '--n2-equivalence', '--fixed-shapes'])
        self.assertEqual(raised.exception.code, 2);self.assertFalse(Path(out).exists())
        module, built = recording(FAKE_MODULE);lines = []
        code, out = self.main('--depth3-diagnostics', '--diagnostic-ks', '2', '--fixed-shapes', lines=lines, module=module)
        report = json.loads((out / 'diagnostics.json').read_text());start = next(x for x in lines if x['message'] == 'DIAGNOSTICS_START')
        self.assertEqual((code, len(built)), (0, 3));self.assertTrue(all(kw.get('fixed_capacity') is True for kw in built), built)
        self.assertEqual((report['harness']['oracle_stack_rows'], start['harness']), (None, report['harness']))
        self.assertTrue(all(o['fixed_capacity'] for o in report['pricer_options'].values()))

    def test_offline_modes_refuse_the_flag(self):
        out = str(Path(tempfile.gettempdir()) / f'depthn-gate-never-created-{uuid.uuid4().hex}')
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (['--reverdict', 'gate.json', '--fixed-shapes'], ['--out', out, '--mutation-from', 'x', '--fixed-shapes'],
                    ['--out', out, '--diagnostics-from', 'x', '--fixed-shapes']):
                with self.assertRaises(SystemExit) as raised:gate.main(argv)
                self.assertEqual(raised.exception.code, 2, argv)
        self.assertFalse(Path(out).exists());self.assertFalse(gate.parse(['--out', 'o', '--d3-root', 'b', '--chunk', '0'])[1].fixed_shapes)

    def test_the_verdict_and_every_exact_field_are_unchanged(self):
        """The same two-call run with and without the flag (the stand-in, and the real pricer when it is on the tree): the same status and
        checks ((a)-(f), (f)'s read identity and priors included), tolerance, header, refusals and per-call results; every saved frame and
        oracle array (digests, statuses, maps, values) bit-equal; --reverdict and --mutation-from reproduce on both, a report without
        `harness` (the no-flag run's) read as before."""
        for real in (False, True) if HAVE_PRICER else (False,):
            with self.subTest(real=real):
                runs = {flag: self.main('--capacity', '16', '--save-oracle', *(('--fixed-shapes',) if flag else ()), real=real) for flag in (False, True)}
                self.assertEqual([code for code, _ in runs.values()], [0, 0])
                plain, fixed = (json.loads((runs[flag][1] / 'gate.json').read_text()) for flag in (False, True))
                self.assertEqual((plain['status'], set(fixed) - set(plain), set(plain) - set(fixed)), ('PASS', {'harness'}, set()))
                for key in ('status', 'passed', 'status_reasons', 'checks', 'decision', 'tolerance', 'header', 'leaves', 'refusals', 'topk_boundary',
                        'depth2_pass', 'split_capacity', 'searched_maps'):
                    self.assertEqual(fixed[key], plain[key], key)
                self.assertEqual({k: v for k, v in fixed['pricer_options'].items() if k != 'fixed_capacity'}, plain['pricer_options'])
                for a, b in zip(fixed['calls'], plain['calls']):
                    for key in ('roots', 'worlds', 'plan', 'refusals', 'nodes_checked', 'leaves_enumerated', 'leaves_checked', 'forced_leaves', 'passed',
                            'choices', 'topk', 'depth2_pass'):
                        self.assertEqual(a[key], b[key], key)
                f = fixed['checks']['f'];self.assertGreater(f['comparisons'], 0);self.assertTrue(f['read_identity']['passed'] and f['priors']['passed'])
                names = sorted(p.name for p in runs[False][1].glob('*.npz'));self.assertEqual(names, sorted(p.name for p in runs[True][1].glob('*.npz')))
                self.assertEqual(len(names), 6)   # per call: the depth-N frame, the depth-2 pass and the oracle data
                for name in names:
                    a, b = saved_arrays(runs[True][1] / name), saved_arrays(runs[False][1] / name);self.assertEqual(set(a), set(b), name)
                    for key in a:self.assertEqual((a[key].dtype, a[key].shape, a[key].tobytes()), (b[key].dtype, b[key].shape, b[key].tobytes()), (name, key))
                again = {}
                for flag, (_, out) in runs.items():
                    with patch.object(gate, 'log', lambda *a, **k: None):self.assertEqual(gate.main(['--reverdict', str(out / 'gate.json')]), 0)
                    again[flag] = json.loads((out / gate.REVERDICT_FILE).read_text())
                self.assertEqual((again[True]['harness'], 'harness' in again[False]), (fixed['harness'], False))
                self.assertEqual({k: v for k, v in again[True]['reverdict'].items() if k != 'source'}, {k: v for k, v in again[False]['reverdict'].items() if k != 'source'})
                if real:continue   # the offline mutation test is the same code on either frame; once on the stand-in
                mutation = {}
                for flag, (_, out) in runs.items():
                    offline = Path(tempfile.mkdtemp(prefix='depthn-fixed-')) / 'offline'
                    with patch.object(gate, 'log', lambda *a, **k: None), patch.object(gate, 'backups_source', lambda: FAKE_BACKUPS_SOURCE), \
                            patch.dict(sys.modules, {'gpu_engine.depthn_backups': FakeBackupsN}):
                        self.assertEqual(gate.main(['--out', str(offline), '--mutation-from', str(out)]), 0)
                    mutation[flag] = json.loads((offline / gate.MUTATION_FILE).read_text())
                verdicts = lambda m: [(c['passed'], {n: (r['killed'], r['caught_by']) for n, r in c['mutants'].items()}, c['material_survivors']) for c in m['calls']]
                self.assertEqual(verdicts(mutation[True]), verdicts(mutation[False]));self.assertEqual(len(mutation[True]['calls']), 2)

    @unittest.skipUnless(HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
    def test_two_calls_compile_each_program_once(self):
        """run_call_n on 16 requests then 3 with the real pricer. Fixed shapes: both oracle stacks are 32 rows, the second call compiles
        no pricer program (the depth-N pricer's and the depth-2 pass's program keys are the same after each call; turn 1 at capacity 16)
        and every oracle program holds one compiled shape. The control (no flag): stacks of 32 and 8 rows, turn 1 recompiled at
        capacity 4 and the oracle's programs compiled for both shapes."""
        module = importlib.import_module('gpu_engine.device_depthn_pricer')
        for fixed in (True, False):
            with self.subTest(fixed_shapes=fixed):
                pricers = gate.DepthNPricers(module, self.ex, object(), depth=3, k_own=2, k_opp=2, selection='prior', capacity=16, actor_capacity=3,
                    fixed_capacity=fixed)
                depth2_pricer = gate.depth2_pass_pricer(module, self.ex, object(), capacity=16, actor_capacity=3, digest=True, fixed_capacity=fixed)
                oracle = reference.DepthNOracle(self.ex, object(), chunk=64);keys, compiled = [], []
                with self.stacks() as seen, self.engine():
                    for call, requests in enumerate((self.requests[:16], self.requests[16:19])):
                        evaluation, _ = gate.run_call_n(requests=requests, recipe=self.recipe, materializer=self.tables, params=PARAMS, pricers=pricers,
                            depth2_pricer=depth2_pricer, frame_pricer=self.frame_pricer(), oracle=oracle, sample=None, block=0, context=dict(block=0, call=call),
                            stack_rows=gate.fixed_shapes_harness(16)['oracle_stack_rows'] if fixed else None)
                        self.assertEqual(self.judged(evaluation)[1][0], 'PASS', failing(self.judged(evaluation)[0]))
                        engines = (pricers.instances[-1], depth2_pricer)
                        keys.append([sorted(map(repr, e.programs)) for e in engines]);compiled.append([e.last_timing['compiled'] for e in engines])
                sizes = {name: f._cache_size() for name, f in [('roots', oracle._root), ('children', oracle._child)] + [(str(k), f) for k, f in oracle._walks.items()]}
                p1 = [[k for k in c[0] if k[0] == 'P1'] for c in compiled]
                if fixed:
                    self.assertEqual([(s['size'], s['shape']) for s in seen], [(32, {32}), (32, {32})])
                    self.assertEqual(keys[0], keys[1]);self.assertEqual(compiled[1], [(), ()]);self.assertEqual([k[1] for k in p1[0]], [16])
                    self.assertEqual(sizes, dict.fromkeys(sizes, 1));self.assertGreaterEqual(len(sizes), 4)
                else:
                    self.assertEqual([(s['size'], s['shape']) for s in seen], [(None, {32}), (None, {8})])
                    self.assertNotEqual(keys[0], keys[1]);self.assertEqual(([k[1] for k in p1[0]], [k[1] for k in p1[1]]), ([16], [4]))
                    self.assertEqual(sizes, dict.fromkeys(sizes, 2))


# ---------------------------------------------------------------- --declared-bounds

DECLARED_ARCHIVE, DECLARED_CHECKPOINT = '7' * 64, '8' * 64
HEAD_FLAGS = ('--head-archive', '/shared/x/head.npz', '--head-archive-sha256', DECLARED_ARCHIVE, '--head-checkpoint-sha256', DECLARED_CHECKPOINT)
CHAMPION_FLAGS = ('--champion-archive', '/shared/x/champion.npz', '--champion-archive-sha256', DECLARED_ARCHIVE, '--champion-checkpoint-sha256', DECLARED_CHECKPOINT)


def declaration(**changes):
    """A valid declaration for the champion CHAMPION_FLAGS name (DECLARED_ARCHIVE, bfloat16); each change replaces a top-level key (None drops it)."""
    content = dict(schema=gate.DECLARED_SCHEMA, evaluator=dict(kind='champion', archive_sha256=DECLARED_ARCHIVE, precision='bfloat16'),
        bounds=dict(value_bound=.008, turn1_logit_bound=.1, deeper_logit_rule=dict(factor=2)), derivation='2 x the bf16-vs-f32 spread, as B1 (test)',
        declared_at='2026-11-01T00:00:00Z', source_measurement=dict(path='/shared/measure/spread.json', sha256='9' * 64), declared_by='test')
    for key, value in changes.items():
        if value is None:content.pop(key)
        else:content[key] = value
    return content


def bounds(**values):
    """A declaration's bounds: value .008, turn-1 logits .1 and the deeper rule's factor 2 unless given (None drops a key)."""
    out = dict(dict(value_bound=.008, turn1_logit_bound=.1, deeper_logit_rule=dict(factor=2)), **values)
    return {k: v for k, v in out.items() if v is not None}


def depth2_pass_priors(a, b):
    """depth2_pass_check's prior quantities with frame `a` standing in for the full-width depth-2 pass and `b` the depth-N frame: each
    requested seat's legal lanes at the valued (own) / expanded (peer) turn-1 cells and the root opponent's legal lanes at usable worlds."""
    t1s, t1e = np.asarray(b.t1_status), np.asarray(b.t1_error);valued = (t1s == 2) | ((t1s == 3) & np.isin(t1e, (5, 6)));out = {}
    for seat, scope in (('own', valued), ('peer', t1s == 2)):
        lanes = (scope & np.asarray(getattr(b, f't1_{seat}_requested'), bool))[..., None] & np.asarray(getattr(b, f't1_{seat}_legal'), bool)
        out[f'depth2_pass_t1_{seat}_logits'] = gate.g2._numeric(np.argwhere(lanes), np.asarray(getattr(a, f't1_{seat}_logits'))[lanes], np.asarray(getattr(b, f't1_{seat}_logits'))[lanes])
    lanes = ((np.asarray(b.pre) == 0) & np.asarray(b.root_peer_requested, bool))[..., None] & np.asarray(b.root_peer_legal, bool)
    out['depth2_pass_root_peer_logits'] = gate.g2._numeric(np.argwhere(lanes), np.asarray(a.root_peer_logits)[lanes], np.asarray(b.root_peer_logits)[lanes])
    return out


ROOT_SITES = (('b', 'root_peer_logits'), ('c', 'root_peer_logits'), ('c', 'depth2_pass_root_peer_logits'), ('block_invariance', 'root_peer_logits'))
TURN1_SITES = (('c', 't1_own_logits'), ('c', 't1_peer_logits'), ('c', 'depth2_pass_t1_own_logits'), ('c', 'depth2_pass_t1_peer_logits'),
    ('block_invariance', 't1_own_logits'), ('block_invariance', 't1_peer_logits'))
SITE_KEYS = dict(b='values', c='quantities', block_invariance='floats')


def site(result, check, name):
    """A prior site's record in a summarized result or report: (b) values, (c) quantities, block invariance floats."""
    return result['checks'][check][SITE_KEYS[check]][name]


def written(content=None, *, raw=None):
    """(path, sha256) of `content` as JSON, or `raw` bytes verbatim, in a new file outside any repository."""
    path = Path(tempfile.mkdtemp(prefix='depthn-declared-')) / 'declared.json';raw = raw if raw is not None else json.dumps(content, indent=1).encode()
    path.write_bytes(raw);return path, hashlib.sha256(raw).hexdigest()


def git(root, *args):
    """git in a scratch repository (no hooks, no signing, a fixed author); its stdout."""
    return subprocess.run(['git', '-C', str(root), '-c', 'core.hooksPath=/dev/null', '-c', 'commit.gpgsign=false', '-c', 'user.name=declared-bounds-test',
        '-c', 'user.email=declared-bounds-test@example.com', *args], check=True, capture_output=True, text=True).stdout.strip()


def committed(content=None, *, raw=None, name='champion.json'):
    """Record(path, sha256, commit, root): `content` as JSON (or `raw` bytes) committed under DECLARED_SUBDIR of a new scratch repository (the
    reviewed-PR record a declaration must have)."""
    root = Path(tempfile.mkdtemp(prefix='depthn-declared-repo-'));git(root, 'init', '-q');path = root / gate.DECLARED_SUBDIR / name
    path.parent.mkdir(parents=True);raw = raw if raw is not None else json.dumps(content, indent=1).encode();path.write_bytes(raw)
    git(root, 'add', '.');git(root, 'commit', '-q', '-m', 'declare bounds')
    return SimpleNamespace(path=path, sha256=hashlib.sha256(raw).hexdigest(), commit=git(root, 'rev-parse', 'HEAD'), root=root)


def load(record, **changes):
    """gate.load_declared on a committed record (changes: path, sha256, commit, root overrides)."""
    r = dict(vars(record), **changes);return gate.load_declared(r['path'], r['sha256'], r['commit'], root=r['root'])


def declared(**changes):
    return load(committed(declaration(**changes)))


class DeclaredBoundsTests(FixtureN, unittest.TestCase):
    """--declared-bounds: the file fails closed; the verdict takes the declared numbers (a run beyond B1's bounds passes looser declared ones
    and one inside them fails tighter ones; declaring the frozen numbers reproduces the frozen verdict); the near-tie Q bound is (e)'s alone;
    the flags are champion only (the command-line round trip runs on the champion: test_gpu_engine_champion_evaluator.ChampionCommandLineTests);
    without the flag nothing changes (the head override's roots here stand for B1's path)."""
    _frame, _d, _data, _n2_frame = GateNTests._frame, GateNTests._d, GateNTests._data, GateNTests._n2_frame
    main = CommandLineRunTests.main

    def roots(self):
        """The CLI fixture's roots under the head override's evaluator header (kind v2, DECLARED_ARCHIVE): what load_roots resolves for HEAD_FLAGS."""
        roots = CommandLineRunTests.roots(self);roots.evaluator = dict(override=True, kind='v2', identity=['v2'], precision='bfloat16',
            archive_sha256=DECLARED_ARCHIVE, checkpoint_sha256=DECLARED_CHECKPOINT, arm_checked=True, dropped_roots=[]);return roots

    def evaluated(self, *, leaf=0., t1_logits=0., ply_logits=0., root_logits=0., twin=False):
        """One call of the stand-in depth-3 surface with `leaf` added to head-valued leaves, `t1_logits` / `ply_logits` to the turn-1 cells' /
        ply-2 nodes' own priors, `root_logits` to every lane of the root opponent's prior, (e) and (f) through the fake backups, a synthetic
        depth-1 floor FLOOR (frozen value bound .008) and one actor. twin: the unshifted frame stands in for the full-width depth-2 pass
        (depth2_pass_priors) and for a block-invariance twin, so every prior site compares the shifts."""
        base = self.cached('frame', self._frame);data = self.cached('data', lambda: self._data(base));frame = copy.deepcopy(base);level = frame.nodes[0]
        frame.leaf_value[np.asarray(frame.leaf_status) == 4] += np.float32(leaf);frame.t1_own_logits[np.asarray(frame.t1_status) != 0] += np.float32(t1_logits)
        level.own_logits[np.asarray(level.status) != 0] += np.float32(ply_logits)
        if root_logits:frame.root_peer_logits[...] += np.float32(root_logits)
        ev = gate.evaluate_n(frame, self._d(), self.requests, data, context={}, backups=FakeBackupsN, decision=True, choices=True)
        ev['c']['depth1_noise_floor'] = gate.g2._numeric([[0]], [FLOOR], [0.]);ev['spreads'] = [SPREAD] * 10;ev['actors'] = 1
        if twin:ev['c'].update(depth2_pass_priors(base, frame));ev['invariance'] = [gate.g2.block_invariance(gate.frame_arrays(base), gate.frame_arrays(frame))]
        return ev

    def judge(self, ev, declared=None):
        result = gate.summarize_run([copy.deepcopy(ev)], (2187,), declared=declared);return result, gate.gate_status_n(result)[0], result['checks']['c']['quantities']

    def test_the_declaration_parses_and_fails_closed(self):
        d = declared();self.assertEqual((d['content'], d['path']), (declaration(), d['path']));self.assertEqual(len(d['sha256']), 64)
        self.assertEqual(gate.declared_numbers(d), dict(value_bound=.008, turn1_logit_bound=.1, deeper_logit_bound=None, deeper_logit_factor=2., near_tie_q_bound=None))
        self.assertEqual(gate.declared_numbers(declared(bounds=bounds(deeper_logit_rule=None, deeper_logit_bound=.3, near_tie_q_bound=.02)))['deeper_logit_bound'], .3)
        refused = {f'missing {k}': (declaration(**{k: None}), f'declaration.{k}: missing') for k in gate.DECLARED_FIELDS}
        refused.update(
            unknown_key=(dict(declaration(), comment='x'), 'declaration.comment: unknown key'), misspelt_bound=(declaration(bounds=dict(bounds(), value_bnd=1)), 'unknown key'),
            no_value_bound=(declaration(bounds=bounds(value_bound=None)), 'bounds.value_bound: missing'),
            no_turn1_logit_bound=(declaration(bounds=bounds(turn1_logit_bound=None)), 'bounds.turn1_logit_bound: missing'),
            no_deeper=(declaration(bounds=bounds(deeper_logit_rule=None)), 'exactly one'), both_deeper=(declaration(bounds=bounds(deeper_logit_bound=.2)), 'exactly one'),
            zero=(declaration(bounds=bounds(value_bound=0)), 'finite number > 0'), boolean=(declaration(bounds=bounds(turn1_logit_bound=True)), 'finite number > 0'),
            negative_factor=(declaration(bounds=bounds(deeper_logit_rule=dict(factor=-2))), 'factor: a finite number > 0'),
            text_bound=(declaration(bounds=bounds(near_tie_q_bound='.01')), 'near_tie_q_bound: a finite number > 0'),
            schema=(declaration(schema='v0'), 'schema'), kind=(declaration(evaluator=dict(declaration()['evaluator'], kind='v2')), 'evaluator.kind'),
            alias_precision=(declaration(evaluator=dict(declaration()['evaluator'], precision='bf16')), 'evaluator.precision'),
            archive=(declaration(evaluator=dict(declaration()['evaluator'], archive_sha256='7' * 63)), 'evaluator.archive_sha256'),
            naive_time=(declaration(declared_at='2026-11-01T00:00:00'), 'declared_at'), other_zone=(declaration(declared_at='2026-11-01T00:00:00+01:00'), 'declared_at'),
            empty_derivation=(declaration(derivation='  '), 'derivation: non-empty'), empty_author=(declaration(declared_by=''), 'declared_by: non-empty'),
            measurement=(declaration(source_measurement=dict(path='/shared/m.json')), 'source_measurement.sha256: missing'),
            measurement_sha=(declaration(source_measurement=dict(path='/shared/m.json', sha256='x')), 'source_measurement.sha256: a 64-hex'),
            future=(declaration(declared_at='2999-01-01T00:00:00Z'), 'after the run\'s start'), not_an_object=([1], 'a JSON object'))
        for case, (content, message) in refused.items():
            with self.subTest(case):
                with self.assertRaises(gate.DeclaredBoundsError) as raised:load(committed(content))
                self.assertIn(message, str(raised.exception))
        record = committed(declaration());text = json.dumps(declaration())
        for case, (call, message) in dict(sha_mismatch=(lambda: load(record, sha256='0' * 64), 'differs from --declared-bounds-sha256'),
                absent=(lambda: load(record, path=record.path.with_name('absent.json')), 'cannot read'),
                duplicate=(lambda: load(committed(raw=text.replace('{', '{"derivation": "a", ', 1).encode())), 'duplicate'),
                nan=(lambda: load(committed(raw=text.replace('0.008', 'NaN').encode())), 'NaN is not a JSON number'),
                not_json=(lambda: load(committed(raw=b'{"schema": ')), 'not a JSON declaration'),
                not_utf8=(lambda: load(committed(raw=b'\xff\xfe{}')), 'not UTF-8')).items():
            with self.subTest(case):
                with self.assertRaises(gate.DeclaredBoundsError) as raised:call()
                self.assertIn(message, str(raised.exception))

    def test_the_declaration_is_a_committed_record(self):
        """The file must lie under DECLARED_SUBDIR of the checkout; its git blob sha1 (git hash-object) and the commit are recorded and, in a git
        work tree, the commit must exist (a made-up one is refused), hold exactly this blob there and be an ancestor of HEAD, with the file tracked
        and unmodified at HEAD; "not verifiable here" only without git, without a work tree of the checkout, or in a shallow clone lacking the
        commit. Every case of the re-review's prereg.py, then the refusals beyond it, then the source measurement."""
        record = committed(declaration());d = load(record)
        self.assertEqual((d['repo_path'], d['commit'], d['git_verification']['status']), (gate.DECLARED_SUBDIR + '/champion.json', record.commit, 'verified'))
        self.assertIn('ancestor of HEAD', d['git_verification']['detail']);self.assertIn('tracked and unmodified at HEAD', d['git_verification']['detail'])
        self.assertEqual(d['git_blob_sha1'], git(record.root, 'hash-object', str(record.path)));self.assertEqual(d['source_measurement_verification']['status'], 'not verifiable here')
        raw = json.dumps(declaration(), indent=1).encode();sha = hashlib.sha256(raw).hexdigest();made_up = 'e' * 40
        edited = json.dumps(declaration(bounds=bounds(value_bound=.5)), indent=1).encode()
        def repo():
            root = Path(tempfile.mkdtemp(prefix='depthn-declared-repo-'));git(root, 'init', '-q');(root / gate.DECLARED_SUBDIR).mkdir(parents=True)
            (root / 'README').write_text('x');git(root, 'add', '.');git(root, 'commit', '-q', '-m', 'base');return root
        def place(root, data=raw, name='champion.json'):
            path = root / gate.DECLARED_SUBDIR / name;path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(data);return path
        def commit(root, message='declare'):
            git(root, 'add', '-A');git(root, 'commit', '-q', '-m', message);return git(root, 'rev-parse', 'HEAD')
        r1 = repo();p1 = place(r1)   # prereg 1, 2, 7: never committed, in a full repository
        r3 = repo();p3 = place(r3);c3 = commit(r3);p3.write_bytes(edited)   # prereg 3, 3b: committed, then edited in the work tree
        r4 = repo();git(r4, 'checkout', '-q', '-b', 'after-the-failure');p4 = place(r4);c4 = commit(r4)   # prereg 4: on a side branch, HEAD there
        r5 = Path(tempfile.mkdtemp(prefix='depthn-nogit-'));(r5 / gate.DECLARED_SUBDIR).mkdir(parents=True);p5 = place(r5)   # prereg 5: no git on PATH
        r6 = repo();outside = Path(tempfile.mkdtemp(prefix='depthn-outside-')) / 'champion.json';outside.write_bytes(raw)
        link = r6 / gate.DECLARED_SUBDIR / 'link.json';link.symlink_to(outside)   # prereg 6: a symlink under the directory to outside the checkout
        outer = repo();inner = outer / 'checkout';(inner / gate.DECLARED_SUBDIR).mkdir(parents=True);p8 = place(inner);c8 = commit(outer)   # prereg 8: nested
        rm = repo();pm = place(rm);cm = commit(rm);pm.write_bytes(edited);commit(rm, 'revise');pm.write_bytes(raw)   # the commit's bytes, HEAD's another revision
        ru = repo();pu = place(ru);cu = commit(ru);git(ru, 'rm', '-q', '--cached', str(pu.relative_to(ru)));git(ru, 'commit', '-q', '-m', 'untrack')   # untracked at HEAD, left in the work tree
        rs = repo();trunk = git(rs, 'rev-parse', '--abbrev-ref', 'HEAD');git(rs, 'checkout', '-q', '-b', 'side');ps = place(rs);cs = commit(rs, 'side')
        git(rs, 'checkout', '-q', trunk);place(rs);commit(rs, 'trunk')   # the same file on HEAD's line, the given commit off it
        origin = repo();po = place(origin);co = commit(origin);(origin / 'later').write_text('y');commit(origin, 'later')
        clone = Path(tempfile.mkdtemp(prefix='depthn-shallow-')) / 'clone';git(origin, 'clone', '-q', '--depth', '1', f'file://{origin}', str(clone))   # lacks co
        refused = dict(
            prereg_1_untracked_made_up_commit=(lambda: gate.load_declared(p1, sha, made_up, root=r1), 'absent: commit ' + made_up),
            prereg_2_untracked_real_head=(lambda: gate.load_declared(p1, sha, git(r1, 'rev-parse', 'HEAD'), root=r1), 'absent: commit'),
            prereg_3_edited_real_commit=(lambda: gate.load_declared(p3, hashlib.sha256(edited).hexdigest(), c3, root=r3), 'mismatch'),
            prereg_3b_edited_made_up_commit=(lambda: gate.load_declared(p3, hashlib.sha256(edited).hexdigest(), made_up, root=r3), 'absent: commit ' + made_up),
            prereg_6_symlink_to_outside=(lambda: gate.load_declared(link, sha, made_up, root=r6), f'committed under {gate.DECLARED_SUBDIR}/'),
            prereg_7_dotdot_made_up_commit=(lambda: gate.load_declared(r1 / gate.DECLARED_SUBDIR / '..' / 'declared_bounds' / 'champion.json', sha, made_up, root=r1),
                'absent: commit ' + made_up),
            made_up_commit_for_a_committed_file=(lambda: load(record, commit=made_up), 'absent: commit ' + made_up),
            modified_against_head=(lambda: gate.load_declared(pm, sha, cm, root=rm), 'modified'),
            untracked_at_head=(lambda: gate.load_declared(pu, sha, cu, root=ru), 'untracked'),
            commit_off_heads_line=(lambda: gate.load_declared(ps, sha, cs, root=rs), 'not an ancestor of HEAD'),
            short_commit=(lambda: load(record, commit=record.commit[:12]), 'full lowercase hex commit id'),
            outside=(lambda: gate.load_declared(*written(declaration()), record.commit, root=record.root), f'committed under {gate.DECLARED_SUBDIR}/'))
        for case, (call, message) in refused.items():
            with self.subTest(case):
                with self.assertRaises(gate.DeclaredBoundsError) as raised:call()
                self.assertIn(message, str(raised.exception))
        with patch.dict(os.environ, {'PATH': '/nonexistent'}):no_git = gate.load_declared(p5, sha, made_up, root=r5)
        accepted = dict(prereg_4_side_branch_head=(gate.load_declared(p4, sha, c4, root=r4), 'verified', 'ancestor of HEAD'),   # the integration branch is the audit's
            prereg_5_no_git_on_path=(no_git, 'not verifiable here', 'git is not available'),
            prereg_8_nested_checkout=(gate.load_declared(p8, sha, c8, root=inner), 'not verifiable here', 'is not the top of a git work tree'),
            no_repository=(gate.load_declared(p5, sha, made_up, root=r5), 'not verifiable here', 'is not the top of a git work tree'),
            shallow_clone_lacking_the_commit=(gate.load_declared(clone / gate.DECLARED_SUBDIR / 'champion.json', sha, co, root=clone), 'not verifiable here', 'shallow'))
        for case, (result, status, detail) in accepted.items():
            with self.subTest(case):self.assertEqual(result['git_verification']['status'], status);self.assertIn(detail, result['git_verification']['detail'])
        self.assertTrue(gate.provisional_reasons(no_git['git_verification'], no_git['source_measurement_verification'])[0].startswith('provisional: git_verification not verifiable here'))
        elsewhere = record.root / 'gpu_engine' / 'champion.json';elsewhere.write_bytes(record.path.read_bytes());inside = commit(record.root, 'the same file outside the directory')
        with self.assertRaises(gate.DeclaredBoundsError) as raised:load(record, path=elsewhere, commit=inside)
        self.assertIn(f'committed under {gate.DECLARED_SUBDIR}/', str(raised.exception))
        measurement = Path(tempfile.mkdtemp(prefix='depthn-measure-')) / 'spread.json';measurement.write_bytes(b'{"spread": 0.003}\n')
        good = dict(path=str(measurement), sha256=hashlib.sha256(measurement.read_bytes()).hexdigest())
        self.assertEqual(declared(source_measurement=good)['source_measurement_verification']['status'], 'verified')
        with self.assertRaises(gate.DeclaredBoundsError) as raised:declared(source_measurement=dict(good, sha256='9' * 64))
        self.assertIn('source_measurement', str(raised.exception))

    def test_the_declaration_binds_one_evaluator(self):
        run = dict(override=True, kind='champion', archive_sha256=DECLARED_ARCHIVE, precision='bfloat16', checkpoint_sha256=DECLARED_CHECKPOINT)
        self.assertEqual(gate.evaluator_mismatches(declaration(), run), [])
        for key, value in (('kind', 'v2'), ('archive_sha256', '6' * 64), ('precision', 'float32')):
            with self.subTest(key):
                mismatched = gate.evaluator_mismatches(declaration(), dict(run, **{key: value}))
                self.assertEqual(len(mismatched), 1);self.assertIn(f'evaluator.{key}', mismatched[0])
        pinned = declaration(evaluator=dict(declaration()['evaluator'], checkpoint_sha256=DECLARED_CHECKPOINT))
        self.assertEqual(gate.evaluator_mismatches(pinned, run), []);self.assertIn('checkpoint_sha256', gate.evaluator_mismatches(pinned, dict(run, checkpoint_sha256='5' * 64))[0])
        self.assertIn('no champion evaluator', gate.evaluator_mismatches(declaration(), dict(run, override=False))[0])
        base = ['--out', 'o', '--d3-root', 'b', '--chunk', '0', '--declared-bounds', 'd.json', '--declared-bounds-sha256', 'a' * 64, '--declared-bounds-commit', 'b' * 40]
        self.assertEqual(gate.args_identity(gate.parse(base + list(CHAMPION_FLAGS))[1]), run)
        self.assertEqual(gate.args_identity(gate.parse(base + list(CHAMPION_FLAGS) + ['--champion-precision', 'f32'])[1]), dict(run, precision='float32'))
        for flags in ((), HEAD_FLAGS):self.assertEqual(gate.args_identity(gate.parse(base[:6] + list(flags))[1])['override'], False)   # no champion: never applies

    def test_flags_are_refused_off_the_champion_and_in_incompatible_modes(self):
        out = str(Path(tempfile.gettempdir()) / f'depthn-gate-never-created-{uuid.uuid4().hex}')
        flags = ['--declared-bounds', 'd.json', '--declared-bounds-sha256', 'a' * 64, '--declared-bounds-commit', 'b' * 40]
        live = ['--out', out, '--d3-root', 'b', '--chunk', '0']
        cases = dict(alone=(live + flags[:2] + list(CHAMPION_FLAGS), 'go together'), sha_alone=(live + flags[2:4] + list(CHAMPION_FLAGS), 'go together'),
            no_commit=(live + flags[:4] + list(CHAMPION_FLAGS), 'go together'), bad_commit=(live + flags[:5] + ['B' * 40] + list(CHAMPION_FLAGS), 'full lowercase hex'),
            bad_sha=(live + flags[:3] + ['A' * 64] + flags[4:] + list(CHAMPION_FLAGS), '64-hex'), bank_actors=(live + flags, 'needs --champion-archive'),
            head_override=(live + flags + list(HEAD_FLAGS), 'champion only'),
            diagnostics=(live + flags + list(CHAMPION_FLAGS) + ['--depth3-diagnostics'], '--depth3-diagnostics'),
            diagnostics_from=(['--out', out, '--diagnostics-from', 'x'] + flags + list(CHAMPION_FLAGS), 'live gate run'),
            mutation_from=(['--out', out, '--mutation-from', 'x'] + flags, 'live gate run'), reverdict=(['--reverdict', 'gate.json'] + flags, 'takes no other option'),
            reverdict_bad_commit=(['--reverdict', 'gate.json', '--declared-bounds-commit', 'abc'], 'full lowercase hex'))
        for case, (argv, message) in cases.items():
            with self.subTest(case):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:gate.main(argv)
                self.assertEqual(raised.exception.code, 2);self.assertIn(message, stderr.getvalue())
        for extra in ((), ('--n2-equivalence',), ('--mutation-test',), ('--n2-equivalence', '--block-invariance')):
            self.assertEqual(gate.parse(live + flags + list(CHAMPION_FLAGS) + list(extra))[1].declared_bounds, 'd.json')
        self.assertEqual(gate.parse(['--reverdict', 'gate.json', '--declared-bounds-commit', 'b' * 40, '--declared-bounds-sha256', 'a' * 64])[1].reverdict, 'gate.json')
        record = committed(declaration())   # main refuses before anything is created or loaded: the file's sha256, its commit, then the flags' evaluator
        for argv, message in (([record.sha256[:-1] + '0', record.commit], 'differs from --declared-bounds-sha256'), ([record.sha256, record.commit], 'evaluator.archive_sha256')):
            head = list(CHAMPION_FLAGS);head[3] = '6' * 64
            with patch.object(gate, 'load_roots', side_effect=AssertionError('loaded')), patch.object(gate, 'log', lambda *a, **k: None), patch.object(gate, 'REPO_ROOT', record.root), \
                    self.assertRaises(SystemExit) as raised:gate.main(live + ['--declared-bounds', str(record.path), '--declared-bounds-sha256', argv[0], '--declared-bounds-commit', argv[1]] + head)
            self.assertIn(message, str(raised.exception.code));self.assertIn('fail closed', str(raised.exception.code));self.assertFalse(Path(out).exists())

    def test_the_verdict_takes_the_declared_bounds(self):
        """Turn-1 priors .12 off fail B1's 0.1 and pass a declared .15; leaves .006 off pass B1's .008 and fail a declared .005; ply-2 priors .12 off
        (turn 1 .07) pass the frozen deeper .14 and fail a declared .11 or a declared rule of factor 1.5 (.105), and pass factor 2 (.14). The
        measured floors and the frozen rule's numbers are reported beside the declared ones."""
        cases = (('turn1_logits', dict(t1_logits=.12), bounds(turn1_logit_bound=.15), ('FAIL', 'PASS'), 't1_own_logits', .15, 'declared_turn1_logit_bound'),
            ('leaf_values', dict(leaf=.006), bounds(value_bound=.005), ('PASS', 'FAIL'), 'leaf_value', .005, 'declared_value_bound'),
            ('deeper_bound', dict(t1_logits=.07, ply_logits=.12), bounds(deeper_logit_rule=None, deeper_logit_bound=.11), ('PASS', 'FAIL'), 'ply2_own_logits', .11,
                'declared_deeper_logit_bound'),
            ('deeper_rule_tighter', dict(t1_logits=.07, ply_logits=.12), bounds(deeper_logit_rule=dict(factor=1.5)), ('PASS', 'FAIL'), 'ply2_own_logits', None,
                'declared_deeper_logit_bound'),
            ('deeper_rule_frozen_factor', dict(t1_logits=.07, ply_logits=.12), bounds(), ('PASS', 'PASS'), 'ply2_own_logits', None, 'declared_deeper_logit_bound'))
        for case, noise, numbers, (frozen, wanted), name, bound, label in cases:
            with self.subTest(case):
                ev = self.evaluated(**noise);d = declared(bounds=numbers);result, status, c = self.judge(ev)
                self.assertEqual(status, frozen, failing(result))
                result, status, c = self.judge(ev, d);m = c[name];tolerance = result['tolerance']
                self.assertEqual(status, wanted, failing(result));self.assertEqual((m['bound_rule'], m['passed']), (label, wanted == 'PASS'))
                if wanted == 'FAIL':self.assertTrue('c' in failing(result) and set(failing(result)) <= {'c', 'e', 'f'}, failing(result))   # (e)/(f) share the value bound
                if bound is not None:self.assertAlmostEqual(m['bound'], bound, places=12)
                self.assertEqual((tolerance['value_bound'], tolerance['logit_bound']), (numbers['value_bound'], numbers['turn1_logit_bound']))
                section = tolerance['declared_bounds'];self.assertEqual(section['sha256'], d['sha256'])
                self.assertEqual(section['frozen_rule_would_give']['value_bound'], 2 * float(np.float32(FLOOR)));self.assertEqual(section['frozen_rule_would_give']['logit_bound'], .1)
                self.assertEqual(section['measured']['depth1_noise_floor'], float(np.float32(FLOOR)));self.assertEqual(section['measured']['turn1_logit_floor'], tolerance['turn1_logit_floor'])
                self.assertEqual(tolerance['frozen_rule']['rule'], gate.g2.TOLERANCE_RULE);self.assertEqual(tolerance['rule'], gate.DECLARED_TOLERANCE_RULE)
                if case == 'deeper_rule_tighter':self.assertAlmostEqual(m['bound'], 1.5 * tolerance['turn1_logit_floor']);self.assertIn('1.5 x turn1_logit_floor', tolerance['deeper_logit_bound_source'])
                for other, q in c.items():
                    if other not in gate.g2.FLOORS:self.assertTrue(q['bound_rule'].startswith('declared_'), other)
                self.assertEqual(result['checks']['f']['value_bound'], numbers['value_bound'])
                self.assertEqual(gate.tolerance_line_n(tolerance)['declared_bounds_sha256'], d['sha256'])

    def test_declaring_the_frozen_numbers_reproduces_the_frozen_verdict(self):
        """value_bound = 2 x the run's floor, turn1_logit_bound = 0.1, deeper_logit_rule factor 2: every check's verdict, every (c) quantity's
        bound and counts and (e)'s and (f)'s numbers are the frozen rule's: only the numbers are declared, and the one rule a declaration brings,
        GUARD_DECLARED_RULE, changes no verdict here, only the turn-1 values' guard modes (gated -> digest_certified: the stand-in's digests match
        and it prices with one parameter set)."""
        d = declared(bounds=bounds(value_bound=2 * float(np.float32(FLOOR))))
        keep = ('passed', 'bound', 'within_bound', 'beyond_bound', 'ratio_to_bound', 'within_guard', 'beyond_guard', 'first_beyond')
        for noise in (dict(), dict(leaf=.006), dict(leaf=.009), dict(t1_logits=.07, ply_logits=.12), dict(t1_logits=.12), dict(t1_logits=.07, ply_logits=.15)):
            with self.subTest(**noise):
                ev = self.evaluated(**noise);(frozen, status, c), (mine, status2, c2) = self.judge(ev), self.judge(ev, d)
                self.assertEqual(status2, status);self.assertEqual({k: v['passed'] for k, v in mine['checks'].items()}, {k: v['passed'] for k, v in frozen['checks'].items()})
                self.assertEqual({n: {k: m.get(k) for k in keep} for n, m in c2.items()}, {n: {k: m.get(k) for k in keep} for n, m in c.items()})
                self.assertEqual({n: (c[n].get('guard_mode'), m.get('guard_mode')) for n, m in c2.items() if m.get('guard_mode') != c[n].get('guard_mode')},
                    {'t1_value_head': ('gated', 'digest_certified')})
                for key in ('value_bound', 'logit_bound', 'guard_bound', 'q_noise_bound', 'deeper_logit_bound', 'turn1_logit_floor', 'depth1_noise_floor'):
                    self.assertEqual(mine['tolerance'][key], frozen['tolerance'][key], key)
                for check, keys in (('e', ('q_noise_bound', 'near_tie_roots', 'near_tie_flips', 'max_abs_delta_q')), ('f', ('value_bound', 'near_ties', 'near_tie_flips',
                        'nodes_grid_beyond_bound', 'nodes_row_q_beyond_bound'))):
                    self.assertEqual({k: mine['checks'][check][k] for k in keys}, {k: frozen['checks'][check][k] for k in keys}, check)

    def test_the_near_tie_q_bound_is_e_alone(self):
        """Leaves .006 off: (e)'s max |dQ| lies above a declared near-tie Q bound .004 (e fails), while (f) keeps the value bound .008 (passes);
        undeclared, the near-tie Q bound is the value bound."""
        ev = self.evaluated(leaf=.006)
        result, status, _ = self.judge(ev, declared(bounds=bounds(near_tie_q_bound=.004)));e, f = result['checks']['e'], result['checks']['f']
        self.assertEqual((status, set(failing(result))), ('FAIL', {'e'}));self.assertGreater(e['max_abs_delta_q'], .004)
        self.assertEqual((e['q_noise_bound'], result['tolerance']['q_noise_bound'], f['value_bound'], f['passed']), (.004, .004, .008, True))
        self.assertEqual(result['tolerance']['declared_bounds']['applied']['q_noise_bound'], .004);self.assertIn('near_tie_q_bound', result['tolerance']['q_noise_rule'])
        result, status, _ = self.judge(ev, declared());self.assertEqual((status, result['checks']['e']['q_noise_bound']), ('PASS', .008))
        self.assertIn('(= the value bound)', result['tolerance']['q_noise_rule'])
        self.assertEqual(result['tolerance']['declared_bounds']['applied']['q_noise_bound'], .008)

    def test_n2_equivalence_takes_the_declared_bounds(self):
        """summarize_n2 under a declaration: priors .12 off fail 0.1 and pass a declared .15; the near-tie Q bound is (e)'s q_noise_bound."""
        frame = fake_as_depth2(self.cached('n2', self._n2_frame));other = copy.deepcopy(frame);other.t1_own_logits[np.asarray(other.t1_status) != 0] += np.float32(.12)
        found, floats = gate.n2_compare(frame, other);e = gate.decide_roots(frame, frame, self.requests, depth2_backups)
        call = dict(a=found, floats=floats, c=dict(depth1_noise_floor=gate.g2._numeric([[0]], [FLOOR], [0.])), e=e, spreads=[SPREAD] * 10)
        self.assertEqual(gate.summarize_n2({'x': [call]})['x']['floats']['t1_own_logits']['passed'], False)
        report = gate.summarize_n2({'x': [copy.deepcopy(call)]}, declared=declared(bounds=bounds(turn1_logit_bound=.15, near_tie_q_bound=.01)))['x']
        self.assertEqual((report['floats']['t1_own_logits']['bound'], report['floats']['t1_own_logits']['passed']), (.15, True))
        self.assertEqual((report['tolerance']['value_bound'], report['tolerance']['q_noise_bound'], report['e']['q_noise_bound']), (.008, .01, .01))
        self.assertEqual(report['tolerance']['frozen_rule']['logit_bound'], .1);self.assertIn('declared_bounds', report['tolerance'])

    def test_the_root_logit_bound_parses_and_fails_closed(self):
        """bounds.root_logit_bound is optional: declared, declared_numbers carries it; absent, declared_numbers has no such key (the root keeps
        turn1_logit_bound). 0, a negative, NaN, inf, a bool and a text are refused as every bound is; NaN / Infinity in the file are not JSON
        numbers; an unknown key is still refused."""
        self.assertEqual(gate.declared_numbers(declared(bounds=bounds(root_logit_bound=.05)))['root_logit_bound'], .05)
        self.assertNotIn('root_logit_bound', gate.declared_numbers(declared()))
        for value in (.05, 1, 1e-9):self.assertEqual(gate.declared_problems(declaration(bounds=bounds(root_logit_bound=value))), [], value)
        for case, value in dict(zero=0, zero_float=0., negative=-.05, nan=float('nan'), inf=float('inf'), boolean=True, false=False, text='.05').items():
            with self.subTest(case):
                self.assertEqual(gate.declared_problems(declaration(bounds=bounds(root_logit_bound=value))), ['bounds.root_logit_bound: a finite number > 0'])
                if case not in ('nan', 'inf'):   # json.dumps would write them as NaN / Infinity: the file case below
                    with self.assertRaises(gate.DeclaredBoundsError) as raised:load(committed(declaration(bounds=bounds(root_logit_bound=value))))
                    self.assertIn('bounds.root_logit_bound: a finite number > 0', str(raised.exception))
        for literal in ('NaN', 'Infinity', '-Infinity'):
            raw = json.dumps(declaration(bounds=bounds(root_logit_bound=.05))).replace('0.05', literal).encode()
            with self.subTest(literal), self.assertRaises(gate.DeclaredBoundsError) as raised:load(committed(raw=raw))
            self.assertIn(f'{literal} is not a JSON number', str(raised.exception))
        self.assertIn('bounds.root_bound: unknown key', gate.declared_problems(declaration(bounds=bounds(root_bound=.05))))

    def test_the_root_logit_bound_reaches_every_root_site_and_no_turn1_site(self):
        """Turn-1 own priors and the root opponent's prior .1 off at every prior site ((b) against D3, (c) against the oracle, depth2_pass and
        block invariance): turn1_logit_bound .15 passes them all, .05 fails them all. A declared root_logit_bound .05 beside turn1_logit_bound
        .15 fails every root site and nothing else: each root record is the one turn1_logit_bound .05 gives (bound_rule
        declared_root_logit_bound aside), every other record, (e) and (f) are the ones without it. Swapped (turn1_logit_bound .05,
        root_logit_bound .15): every turn-1 site fails as under .05 and every root site passes as under .15. The tolerance, its applied
        section, its log line and the table record root_logit_bound; without one nothing names it."""
        ev = self.evaluated(t1_logits=.1, root_logits=.1, twin=True)
        judged = lambda **numbers: self.judge(ev, declared(bounds=bounds(**numbers)))[:2]
        (loose, _), (tight, _) = judged(turn1_logit_bound=.15), judged(turn1_logit_bound=.05)
        (root_tight, root_tight_status), (root_loose, root_loose_status) = judged(turn1_logit_bound=.15, root_logit_bound=.05), judged(turn1_logit_bound=.05, root_logit_bound=.15)
        plain = lambda m: {k: v for k, v in m.items() if k != 'bound_rule'};shifted = ROOT_SITES + tuple(s for s in TURN1_SITES if 'own' in s[1])
        for check, name in ROOT_SITES + TURN1_SITES:   # the shifted sites compare the shift and fail .05; the turn-1 peer prior is not shifted
            m = site(loose, check, name);self.assertGreater(m['compared'], 0, (check, name))
            if (check, name) in shifted:self.assertAlmostEqual(m['max_abs_difference'], .1, delta=1e-3, msg=(check, name))
            self.assertEqual((m['passed'], site(tight, check, name)['passed']), (True, (check, name) not in shifted), (check, name))
        self.assertEqual(failing(loose), {});self.assertEqual(set(failing(tight)), {'b', 'c', 'block_invariance'})
        for result, wanted, other, status in ((root_tight, .05, loose, root_tight_status), (root_loose, .15, tight, root_loose_status)):
            for check, name in ROOT_SITES:   # the root at root_logit_bound, exactly as the turn-1 bound of that number gives it
                m = site(result, check, name);self.assertEqual((m['bound'], m['bound_rule'], m['passed']), (wanted, 'declared_root_logit_bound', wanted > .1), (check, name))
                self.assertEqual(plain(m), plain(site(tight if wanted == .05 else loose, check, name)), (check, name))
            for check, name in TURN1_SITES:   # every turn-1 prior as without root_logit_bound
                self.assertEqual(site(result, check, name), site(other, check, name), (check, name))
            for check in ('b', 'c', 'block_invariance'):   # every other quantity untouched
                self.assertEqual({n: m for n, m in result['checks'][check][SITE_KEYS[check]].items() if (check, n) not in ROOT_SITES},
                    {n: m for n, m in other['checks'][check][SITE_KEYS[check]].items() if (check, n) not in ROOT_SITES}, check)
            self.assertEqual((result['checks']['e'], result['checks']['f']), (other['checks']['e'], other['checks']['f']))
            tolerance = result['tolerance'];self.assertEqual((tolerance['root_logit_bound'], tolerance['declared_bounds']['applied']['root_logit_bound']), (wanted, wanted))
            self.assertEqual(gate.tolerance_line_n(tolerance)['root_logit_bound'], wanted);self.assertEqual(tolerance['logit_bound'], .15 if wanted == .05 else .05)
            rows = {(r['check'], r['quantity']): r for r in tolerance['table']}
            for key in ROOT_SITES:self.assertEqual((rows[key]['bound'], rows[key]['passed']), (wanted, wanted > .1), key)
        self.assertEqual((root_tight_status, set(failing(root_tight))), ('FAIL', {'b', 'c', 'block_invariance'}))   # the root sites alone fail
        self.assertEqual((root_loose_status, set(failing(root_loose))), ('FAIL', {'c', 'block_invariance'}))   # (b) compares no turn-1 prior
        for result in (loose, tight):
            self.assertNotIn('root_logit_bound', result['tolerance']);self.assertNotIn('root_logit_bound', result['tolerance']['declared_bounds']['applied'])
            self.assertNotIn('root_logit_bound', gate.tolerance_line_n(result['tolerance']));self.assertNotIn('declared_root_logit_bound', json.dumps(result, default=str))
            for check, name in ROOT_SITES:self.assertEqual(site(result, check, name)['bound'], result['tolerance']['logit_bound'], (check, name))

    def test_a_root_logit_bound_equal_to_turn1_logit_bound_is_the_declaration_without_it(self):
        """root_logit_bound declared equal to turn1_logit_bound: the summarized result is the one without it (every check, record, row and
        number) once its own records are removed: tolerance.root_logit_bound and applied.root_logit_bound, and the declared_root_logit_bound
        label (declared_turn1_logit_bound on (c), absent on (b) and block invariance)."""
        def unrooted(value, where=()):
            if isinstance(value, dict):
                out = {k: unrooted(v, where + (k,)) for k, v in value.items() if k != 'root_logit_bound'}
                if out.get('bound_rule') == 'declared_root_logit_bound':
                    if 'c' in where:out['bound_rule'] = 'declared_turn1_logit_bound'
                    else:out.pop('bound_rule')
                return out
            return [unrooted(v, where) for v in value] if isinstance(value, list) else value
        for noise in (dict(t1_logits=.1, root_logits=.1, twin=True), dict(root_logits=.12, twin=True), dict(leaf=.006, root_logits=.05, twin=True)):
            with self.subTest(**noise):
                ev = self.evaluated(**noise);(without, status, _), (rooted, status2, _) = self.judge(ev, declared(bounds=bounds(turn1_logit_bound=.1))), \
                    self.judge(ev, declared(bounds=bounds(turn1_logit_bound=.1, root_logit_bound=.1)))
                self.assertEqual(status2, status);self.assertEqual(rooted['tolerance']['root_logit_bound'], .1)
                self.assertNotEqual(rooted['tolerance']['declared_bounds']['sha256'], without['tolerance']['declared_bounds']['sha256'])
                for result in (without, rooted):[result['tolerance']['declared_bounds'].pop(k) for k in ('sha256', 'path')];result['tolerance'].pop('value_bound_source')   # the files differ
                self.assertEqual(unrooted(rooted), without);self.assertEqual(unrooted(gate.tolerance_line_n(rooted['tolerance'])), gate.tolerance_line_n(without['tolerance']))

    def test_n2_equivalence_takes_the_declared_root_logit_bound(self):
        """summarize_n2: turn-1 own priors and the root prior .1 off (floats and child-block invariance). root_logit_bound .05 beside
        turn1_logit_bound .15 fails the root alone; swapped, the turn-1 prior alone fails; each record as the turn-1 bound of its number gives it."""
        frame = fake_as_depth2(self.cached('n2', self._n2_frame));other = copy.deepcopy(frame)
        other.t1_own_logits[np.asarray(other.t1_status) != 0] += np.float32(.1);other.root_peer_logits[...] += np.float32(.1)
        found, floats = gate.n2_compare(frame, other);e = gate.decide_roots(frame, frame, self.requests, depth2_backups)
        call = dict(a=found, floats=floats, c=dict(depth1_noise_floor=gate.g2._numeric([[0]], [FLOOR], [0.])), e=e, spreads=[SPREAD] * 10,
            invariance=[gate.g2.block_invariance(frame, other)])
        judged = lambda **numbers: gate.summarize_n2({'x': [copy.deepcopy(call)]}, declared=declared(bounds=bounds(**numbers)))['x']
        loose, tight = judged(turn1_logit_bound=.15), judged(turn1_logit_bound=.05)
        self.assertEqual((loose['floats']['root_peer_logits']['passed'], loose['floats']['t1_own_logits']['passed'], loose['block_invariance']['passed']), (True, True, True))
        self.assertGreater(loose['floats']['root_peer_logits']['compared'], 0);self.assertNotIn('declared_root_logit_bound', json.dumps(loose, default=str))
        plain = lambda m: {k: v for k, v in m.items() if k != 'bound_rule'}
        for numbers, wanted, other_report in ((dict(turn1_logit_bound=.15, root_logit_bound=.05), .05, loose), (dict(turn1_logit_bound=.05, root_logit_bound=.15), .15, tight)):
            report = judged(**numbers)
            for section in ('floats', 'block_invariance'):
                part = report[section] if section == 'floats' else report[section]['floats'];was = other_report[section] if section == 'floats' else other_report[section]['floats']
                m = part['root_peer_logits'];self.assertEqual((m['bound'], m['bound_rule'], m['passed']), (wanted, 'declared_root_logit_bound', wanted > .1), section)
                self.assertEqual(plain(m), plain((tight if wanted == .05 else loose)[section]['root_peer_logits'] if section == 'floats' else
                    (tight if wanted == .05 else loose)[section]['floats']['root_peer_logits']), section)
                self.assertEqual({n: x for n, x in part.items() if n != 'root_peer_logits'}, {n: x for n, x in was.items() if n != 'root_peer_logits'}, section)
            self.assertEqual((report['status'], report['passed'], report['block_invariance']['passed']), ('FAIL', False, False))
            self.assertEqual((report['tolerance']['root_logit_bound'], report['tolerance']['logit_bound']), (wanted, numbers['turn1_logit_bound']))

    def test_a_root_only_shift_passes_under_its_root_logit_bound(self):
        """Only the root opponent's prior .1 off. turn1_logit_bound .05 alone fails every root site and nothing else (gate: (b), (c),
        depth2_pass, block invariance; --n2-equivalence: the floats and child-block invariance). root_logit_bound .15 beside it passes: (b)'s
        and block invariance's passed are recomputed from their records, not kept from the turn-1 verdict g2.summarize / merge_invariance gave
        them, in the gate's verdict, in mutant_verdict and in summarize_n2. n2's turn-1 heads against the oracle (reported, not gated) bound
        the root at root_logit_bound and the turn-1 priors at turn1_logit_bound."""
        tight, rooted = declared(bounds=bounds(turn1_logit_bound=.05)), declared(bounds=bounds(turn1_logit_bound=.05, root_logit_bound=.15))
        ev = self.evaluated(root_logits=.1, twin=True)
        result, status, _ = self.judge(ev, tight)
        self.assertEqual((status, set(failing(result))), ('FAIL', {'b', 'c', 'block_invariance'}))
        for check, name in ROOT_SITES:self.assertEqual((site(result, check, name)['bound'], site(result, check, name)['passed']), (.05, False), (check, name))
        self.assertEqual({(check, n) for check in SITE_KEYS for n, m in result['checks'][check][SITE_KEYS[check]].items() if m.get('passed') is False}, set(ROOT_SITES))
        result, status, _ = self.judge(ev, rooted)
        self.assertEqual((status, failing(result), result['passed']), ('PASS', {}, True))
        self.assertEqual((result['checks']['b']['passed'], result['checks']['block_invariance']['passed']), (True, True))
        for check, name in ROOT_SITES:self.assertEqual((site(result, check, name)['bound'], site(result, check, name)['passed']), (.15, True), (check, name))
        verdict, judged = gate.mutant_verdict(copy.deepcopy(ev), lanes=(2187,), declared=rooted)
        self.assertEqual((verdict['status'], verdict['killed'], verdict['caught_by'], verdict['root_logit_bound']), ('PASS', False, [], .15))
        self.assertEqual((judged['checks']['b']['passed'], judged['checks']['block_invariance']['passed']), (True, True))
        verdict, _ = gate.mutant_verdict(copy.deepcopy(ev), lanes=(2187,), declared=tight)
        self.assertEqual((verdict['status'], verdict['caught_by']), ('FAIL', ['b', 'block_invariance', 'c']));self.assertNotIn('root_logit_bound', verdict)
        frame = fake_as_depth2(self.cached('n2', self._n2_frame));other = copy.deepcopy(frame);other.root_peer_logits[...] += np.float32(.1)
        found, floats = gate.n2_compare(frame, other);e = gate.decide_roots(frame, frame, self.requests, depth2_backups)
        heads = dict(depth1_noise_floor=gate.g2._numeric([[0]], [FLOOR], [0.]), **copy.deepcopy({k: floats[k] for k in ('root_peer_logits', 't1_own_logits', 't1_peer_logits')}))
        call = dict(a=found, floats=floats, c=heads, e=e, spreads=[SPREAD] * 10, invariance=[gate.g2.block_invariance(frame, other)])   # heads: check_heads' stand-ins
        self.assertGreater(floats['root_peer_logits']['compared'], 0);self.assertAlmostEqual(floats['root_peer_logits']['max_abs_difference'], .1, delta=1e-3)
        report = gate.summarize_n2({'x': [copy.deepcopy(call)]}, declared=tight)['x']
        self.assertEqual((report['passed'], report['floats']['root_peer_logits']['passed'], report['block_invariance']['passed']), (False, False, False))
        self.assertEqual([n for n, m in report['floats'].items() if not m['passed']] + [n for n, m in report['block_invariance']['floats'].items() if not m['passed']],
            ['root_peer_logits', 'root_peer_logits'])
        self.assertEqual(report['turn1_heads_vs_oracle']['quantities']['root_peer_logits']['bound'], .05)
        report = gate.summarize_n2({'x': [copy.deepcopy(call)]}, declared=rooted)['x']
        self.assertEqual((report['status'], report['passed'], report['block_invariance']['passed'], report['block_invariance']['discrete_equal']), ('PASS', True, True, True))
        self.assertEqual({k: report['floats']['root_peer_logits'][k] for k in ('bound', 'bound_rule', 'passed')}, dict(bound=.15, bound_rule='declared_root_logit_bound', passed=True))
        quantities = report['turn1_heads_vs_oracle']['quantities']
        self.assertEqual({k: quantities['root_peer_logits'][k] for k in ('bound', 'bound_rule', 'within_bound')}, dict(bound=.15, bound_rule='declared_root_logit_bound', within_bound=True))
        for name in ('t1_own_logits', 't1_peer_logits'):self.assertEqual((quantities[name]['bound'], 'bound_rule' in quantities[name]), (.05, False), name)

    def depth2_evaluated(self, *, values=0., terminal=0., twin=False, corrupt_leaf_digest=False, actors=1, twin_terminal=0., twin_outside_digest=False):
        """One call of the stand-in at depth 2 (k 2): `values` added to the head-valued leaves and the expanded turn-1 cells (so (b)'s and (c)'s
        turn-1 values and the leaf values move together), `terminal` to the terminal turn-1 cells (outside every digest mask), the synthetic
        floor FLOOR and spreads SPREAD (value bound .008, guard .005) and `actors` parameter sets. corrupt_leaf_digest: the pricer's digest of one
        scored leaf one bit off. twin: the unshifted frame as the full-width depth-2 pass (its depth2_pass_t1_value, which no digest certifies) and
        as a block-invariance twin (with its invariance_cover, as run_call_n records it); twin_terminal is added to that twin's terminal turn-1
        values and twin_outside_digest sets one word of its leaf digest at a leaf no one enumerated (outside the digest mask) one bit off."""
        def priced():
            with self.engine():return OracleNPricer(self.ex, object(), depth=2, k_own=2, k_opp=2).price(self.requests, self.tables, self.recipe, PARAMS)
        def enumerated():
            with self.engine():
                stack = reference.stack(self.requests, batch._numpy_worlds(self.requests, self.tables, self.recipe))
                return gate.oracle_levels(oracle=self.oracle, stack=stack, requests=self.requests, recipe=self.recipe, params=PARAMS, frame=base, plan=gate.plan_of(2, 2, 2))
        base = self.cached('frame2', priced);data = self.cached('data2', enumerated);frame = copy.deepcopy(base);t1s = np.asarray(frame.t1_status)
        frame.leaf_value[np.asarray(frame.leaf_status) == 4] += np.float32(values);frame.t1_value[t1s == 2] += np.float32(values)
        if terminal:frame.t1_value[t1s == 1] += np.float32(terminal)
        if corrupt_leaf_digest:
            index = data['leaves']['index'][np.flatnonzero(gate._leaf_scored(frame, data['leaves']))[0]];frame.leaf_digest[tuple(index)][0] ^= np.uint32(1)
        ev = gate.evaluate_n(frame, self._d(), self.requests, data, context={}, backups=FakeBackupsN, decision=True, choices=True)
        ev['c']['depth1_noise_floor'] = gate.g2._numeric([[0]], [FLOOR], [0.]);ev['spreads'] = [SPREAD] * 10;ev['actors'] = actors
        if twin:
            valued = (t1s == 2) | ((t1s == 3) & np.isin(np.asarray(frame.t1_error), (5, 6)))   # depth2_pass_check's t1_value cells
            ev['c']['depth2_pass_t1_value'] = gate.g2._numeric(np.argwhere(valued), np.asarray(base.t1_value)[valued], np.asarray(frame.t1_value)[valued])
            other = copy.deepcopy(base) if twin_terminal or twin_outside_digest else base
            if twin_terminal:other.t1_value[np.asarray(other.t1_status) == 1] += np.float32(twin_terminal)
            if twin_outside_digest:other.leaf_digest[tuple(np.argwhere(np.asarray(other.leaf_status) == 0)[0])][0] ^= np.uint32(1)
            ev['invariance'] = [gate.g2.block_invariance(gate.frame_arrays(other), gate.frame_arrays(frame))]
            ev['invariance_cover'] = [gate.invariance_cover(gate.frame_arrays(other), gate.frame_arrays(frame))]
        return ev

    def test_the_declared_guard_is_digest_certified_at_depth_2(self):
        """GUARD_DECLARED_RULE at depth 2: (b)'s and (c)'s turn-1 values and the leaf values .006 off, beyond the guard (.005) and within the value
        bound (.008), every digest matching, one parameter set. Frozen: FAIL on the guard ((b), (c)). Declared: PASS, each guarded quantity
        digest_certified with its digest evidence ((b)'s terminal cells outside t1_digest_own's mask, bit-equal), the guard still computed and
        reported (within_guard False, ratio_to_guard > 1), the table, the TOLERANCE line and tolerance.guard_declared recording the modes;
        mutant_verdict judges under the same rule. Two parameter sets, or terminal cells (outside every digest mask) that are not bit-equal, keep
        the guard gated ((a) fails those exactly too)."""
        d = declared();ev = self.depth2_evaluated(values=.006)
        frozen, status, c = self.judge(ev)
        self.assertEqual((status, set(failing(frozen))), ('FAIL', {'b', 'c'}))
        for m in (frozen['checks']['b']['values']['t1_value'], c['t1_value_head'], c['leaf_value']):
            self.assertEqual((m['within_bound'], m['within_guard'], m['passed'], m.get('guard_mode', 'gated'), m.get('guard_evidence')), (True, False, False, 'gated', None))
        self.assertNotIn('guard_declared', frozen['tolerance']);self.assertNotIn('guard_mode', frozen['checks']['b']['values']['t1_value'])
        result, status, c = self.judge(ev, d);b = result['checks']['b']['values']['t1_value']
        self.assertEqual((status, failing(result)), ('PASS', {}))
        for name, m, digest in (('b.t1_value', b, 't1_digest_own'), ('c.t1_value_head', c['t1_value_head'], 't1_digest_own'), ('c.leaf_value', c['leaf_value'], 'leaf_digest')):
            e = m['guard_evidence']
            self.assertEqual((m['guard_mode'], m['passed'], m['within_bound'], m['within_guard'], m['guard_bound'], e['digest'], e['digest_mismatched'], e['evaluator_actors']),
                ('digest_certified', True, True, False, .005, digest, 0, 1), name)
            self.assertGreater(m['ratio_to_guard'], 1, name);self.assertGreaterEqual(e['digest_compared'], m['compared'] - (e['value_positions_outside_digest'] or 0), name)
        self.assertGreater(b['guard_evidence']['value_positions_outside_digest'], 0);self.assertEqual(b['guard_evidence']['value_positions_outside_digest_inexact'], 0)
        self.assertEqual(c['leaf_value']['guard_evidence']['subset'], gate.LEAF_CONSTRUCTION);self.assertGreater(c['leaf_value']['beyond_guard'], 0)
        record = result['tolerance']['guard_declared'];self.assertEqual(record['rule'], gate.GUARD_DECLARED_RULE)
        self.assertEqual((record['modes'], record['gated_no_certifying_digest']), (dict.fromkeys(('b.t1_value', 'c.t1_value_head', 'c.leaf_value'), 'digest_certified'), []))
        self.assertIn('before any champion result', gate.GUARD_DECLARED_RULE);self.assertIn('GUARD_DECLARED_RULE', gate.DECLARED_RULE + gate.DECLARED_TOLERANCE_RULE)
        rows = {(r['check'], r['quantity']): r for r in result['tolerance']['table']}
        self.assertEqual((rows[('b', 't1_value')]['guard_mode'], rows[('b', 't1_value')]['passed']), ('digest_certified', True))
        line = {(x['check'], x['quantity']): x for x in gate.tolerance_line_n(result['tolerance'])['checks']}
        self.assertEqual({k: line[k]['guard_mode'] for k in (('b', 't1_value'), ('c', 't1_value_head'), ('c', 'leaf_value'))}, dict.fromkeys((('b', 't1_value'), ('c', 't1_value_head'), ('c', 'leaf_value')), 'digest_certified'))
        self.assertEqual(gate.mutant_verdict(copy.deepcopy(ev), lanes=(2187,), declared=d)[0]['status'], 'PASS')   # the mutation test's verdicts, same rule
        self.assertEqual(gate.mutant_verdict(copy.deepcopy(ev), lanes=(2187,))[0]['status'], 'FAIL')
        result, status, c = self.judge(self.depth2_evaluated(values=.006, actors=2), d)   # a digest certifies inputs, not which parameters priced them
        self.assertEqual((status, set(failing(result))), ('FAIL', {'b', 'c'}));self.assertEqual(set(result['tolerance']['guard_declared']['modes'].values()), {'gated'})
        self.assertEqual(c['leaf_value']['guard_evidence']['evaluator_actors'], 2)
        result, status, _ = self.judge(self.depth2_evaluated(values=.006, terminal=1e-6), d);b = result['checks']['b']['values']['t1_value']   # outside the mask, not exact
        self.assertEqual((status, set(failing(result)), b['guard_mode'], b['within_bound']), ('FAIL', {'a', 'b'}, 'gated', True))   # (a): terminal values exact
        self.assertEqual({k for k, v in result['checks']['a']['mismatched'].items() if v}, {'t1_value_terminal'})
        self.assertEqual(b['guard_evidence']['value_positions_outside_digest_inexact'], b['guard_evidence']['value_positions_outside_digest'])
        recorded, derived = self.rederived(result, d);self.assertEqual((derived, recorded['b.t1_value']), (recorded, 'gated'))   # --reverdict agrees

    def rederived(self, result, d):
        """(the guard modes a summarized declared depth-2 result records, those --reverdict re-derives once it is written as gate.json is):
        reverdict's own steps, declared_guards from the saved evidence and apply_rules_depthn without covers."""
        saved = json.loads(json.dumps(result, allow_nan=False, default=str));recorded = gate.guard_record(saved['checks'])['modes']
        actors = saved['tolerance']['evaluator_actors'];gate.declared_guards(saved, None, actors);gate.apply_rules_depthn(saved, 2, actors=actors, declared=d)
        return recorded, saved['tolerance']['guard_declared']['modes']

    def test_a_twin_terminal_value_that_is_not_exact_gates_block_invariance(self):
        """Block invariance's t1_value with the other child block's terminal turn-1 values .006 off: terminal cells lie outside t1_digest_own's
        mask, so they must be bit-equal; they are not, the guard is gated and block invariance FAILs on the guard alone (within the value
        bound), while leaf_value stays certified; --reverdict's re-derivation from the written report agrees."""
        d = declared();result, status, _ = self.judge(self.depth2_evaluated(twin=True, twin_terminal=.006), d)
        inv = result['checks']['block_invariance'];m = inv['floats']['t1_value'];e = m['guard_evidence']
        self.assertEqual((status, set(failing(result)), inv['discrete_equal']), ('FAIL', {'block_invariance'}, True))
        self.assertEqual((m['guard_mode'], m['within_bound'], m['within_guard'], m['passed'], e['digest_mismatched']), ('gated', True, False, False, 0))
        self.assertGreater(e['value_positions_outside_digest_inexact'], 0);self.assertEqual(inv['floats']['leaf_value']['guard_mode'], 'digest_certified')
        recorded, derived = self.rederived(result, d);self.assertEqual(derived, recorded);self.assertEqual(recorded['block_invariance.t1_value'], 'gated')

    def test_a_digest_differing_only_outside_the_mask_gates_the_guard(self):
        """Block invariance with the twin's leaf digest one bit off at a leaf no one enumerated (outside the mask) and the leaf values .006 apart:
        the certifying check, block_invariance's comparison of the whole digest array, has a mismatch, so leaf_value's guard is gated, live and as
        --reverdict re-derives it from the written report (its differing count); t1_value, whose digests all match, stays certified."""
        d = declared();result, status, _ = self.judge(self.depth2_evaluated(values=.006, twin=True, twin_outside_digest=True), d)
        inv = result['checks']['block_invariance'];m = inv['floats']['leaf_value']
        self.assertEqual((status, inv['passed'], inv['discrete_equal'], inv['differing']['leaf_digest']['count']), ('FAIL', False, False, 1))
        self.assertEqual((m['guard_mode'], m['guard_evidence']['digest_mismatched'], m['within_bound'], m['passed']), ('gated', 1, True, False))
        self.assertEqual(inv['floats']['t1_value']['guard_mode'], 'digest_certified')
        recorded, derived = self.rederived(result, d);self.assertEqual(derived, recorded);self.assertEqual(recorded['block_invariance.leaf_value'], 'gated')

    def test_a_corrupted_leaf_digest_gates_the_leaf_guard(self):
        """The same declared depth-2 call with one scored leaf's digest one bit off: (a) fails on it, the leaf values' guard is gated there
        (leaf_value fails on the guard alone), the turn-1 values stay digest-certified, and the run FAILs."""
        result, status, c = self.judge(self.depth2_evaluated(values=.006, corrupt_leaf_digest=True), declared())
        self.assertEqual((status, set(failing(result)), result['checks']['a']['mismatched']['leaf_digest']), ('FAIL', {'a', 'c'}, 1))
        m = c['leaf_value'];self.assertEqual((m['guard_mode'], m['guard_evidence']['digest_mismatched'], m['within_bound'], m['passed']), ('gated', 1, True, False))
        self.assertEqual((c['t1_value_head']['guard_mode'], result['checks']['b']['values']['t1_value']['guard_mode']), ('digest_certified', 'digest_certified'))
        self.assertEqual(result['tolerance']['guard_declared']['modes']['c.leaf_value'], 'gated')

    def test_a_guarded_value_without_a_certifying_digest_stays_gated(self):
        """With the full-width depth-2 pass and a block-invariance twin: block invariance's t1_value and leaf_value are digest-certified by the
        two frames' digest arrays on (a)'s masks (every value outside them bit-equal), while depth2_pass_t1_value, which no digest check
        certifies, stays gated (guard_evidence.reason, tolerance.guard_declared.gated_no_certifying_digest) and fails (c) on the guard alone."""
        result, status, c = self.judge(self.depth2_evaluated(values=.006, twin=True), declared())
        m = c['depth2_pass_t1_value'];self.assertEqual((status, set(failing(result))), ('FAIL', {'c'}))
        self.assertEqual((m['guard_mode'], m['guard_evidence']['reason'], m['guard_evidence']['digest'], m['within_bound'], m['within_guard'], m['passed']),
            ('gated', gate.GUARD_NO_DIGEST, None, True, False, False))
        self.assertEqual([n for n, q in c.items() if n not in gate.g2.FLOORS and not q['passed']], ['depth2_pass_t1_value'])
        self.assertEqual(result['tolerance']['guard_declared']['gated_no_certifying_digest'], ['c.depth2_pass_t1_value'])
        inv = result['checks']['block_invariance'];self.assertTrue(inv['passed'])
        for name, digest in (('t1_value', 't1_digest_own'), ('leaf_value', 'leaf_digest')):
            q = inv['floats'][name];e = q['guard_evidence']
            self.assertEqual((q['guard_mode'], q['within_guard'], e['digest'], e['digest_mismatched'], e['value_positions_outside_digest_inexact']), ('digest_certified', False, digest, 0, 0), name)
            self.assertEqual(result['tolerance']['guard_declared']['modes'][f'block_invariance.{name}'], 'digest_certified')
        frozen, status, _ = self.judge(self.depth2_evaluated(values=.006, twin=True))
        self.assertEqual((status, frozen['checks']['block_invariance']['passed']), ('FAIL', False))

    def test_n2_equivalence_takes_the_declared_guard(self):
        """summarize_n2 under a declaration: the Depth2Pricer frame's expanded turn-1 values and head-valued leaves .006 off the DepthNPricer's
        (beyond the guard, within the value bound): the floats and child-block invariance are digest-certified by n2_compare's t1_digest_own /
        t2_digest and the frames' digest arrays (PASS; frozen: FAIL on the guard); one head-valued leaf's t2_digest one bit off gates t2_value;
        two parameter sets gate every guard."""
        frame = fake_as_depth2(self.cached('n2', self._n2_frame))
        def call(corrupt=False, actors=1):
            other = copy.deepcopy(frame);t1s, t2s = np.asarray(other.t1_status), np.asarray(other.t2_status)
            other.t1_value[t1s == 2] += np.float32(.006);other.t2_value[t2s == 4] += np.float32(.006)
            if corrupt:other.t2_digest[tuple(np.argwhere(t2s == 4)[0])][0] ^= np.uint32(1)
            found, floats = gate.n2_compare(frame, other);e = gate.decide_roots(frame, frame, self.requests, depth2_backups)
            return dict(a=found, floats=floats, c=dict(depth1_noise_floor=gate.g2._numeric([[0]], [FLOOR], [0.])), e=e, spreads=[SPREAD] * 10, cover=gate.n2_cover(frame, other),
                actors=actors, invariance=[gate.g2.block_invariance(frame, other)], invariance_cover=[gate.invariance_cover(frame, other)])
        frozen = gate.summarize_n2({'x': [call()]})['x']
        self.assertEqual((frozen['status'], frozen['floats']['t1_value']['passed'], frozen['floats']['t2_value']['passed'], frozen['block_invariance']['passed']), ('FAIL', False, False, False))
        self.assertNotIn('guard_declared', frozen['tolerance']);self.assertNotIn('guard_mode', frozen['floats']['t1_value'])
        report = gate.summarize_n2({'x': [call()]}, declared=declared())['x'];record = report['tolerance']['guard_declared']
        self.assertEqual((report['status'], report['passed'], report['block_invariance']['passed']), ('PASS', True, True))
        self.assertEqual(record['modes'], dict.fromkeys(('floats.t1_value', 'floats.t2_value', 'block_invariance.t1_value', 'block_invariance.t2_value'), 'digest_certified'))
        for name, digest in (('t1_value', 't1_digest_own'), ('t2_value', 't2_digest')):
            m = report['floats'][name];self.assertEqual((m['within_guard'], m['guard_evidence']['digest'], m['guard_evidence']['value_positions_outside_digest']), (False, digest, 0), name)
        report = gate.summarize_n2({'x': [call(corrupt=True)]}, declared=declared())['x']
        self.assertEqual((report['status'], report['discrete']['mismatched']['t2_digest'], report['floats']['t2_value']['guard_mode'], report['floats']['t1_value']['guard_mode']),
            ('FAIL', 1, 'gated', 'digest_certified'))
        self.assertEqual((report['block_invariance']['floats']['t2_value']['guard_mode'], report['block_invariance']['floats']['t2_value']['guard_evidence']['digest_mismatched']), ('gated', 1))
        report = gate.summarize_n2({'x': [call(actors=2)]}, declared=declared())['x']   # two parameter sets: a digest certifies inputs, not which parameters priced them
        self.assertEqual((report['status'], set(report['tolerance']['guard_declared']['modes'].values()), report['floats']['t1_value']['guard_evidence']['evaluator_actors']),
            ('FAIL', {'gated'}, 2))

    def test_the_topk_root_gap_count_takes_the_root_logit_bound(self):
        """merge_topk_n: the root selection's gaps are counted below a given root_logit_bound (recorded beside the plies), every other ply's
        below logit_bound; root_logit_bound None is the call without it."""
        reports = [{'root': {'opp': dict(gaps=[.05, .12, .3], oracle_prior_set_differs=1)}, 'ply2': {'own': dict(gaps=[.05, .12, .3], oracle_prior_set_differs=0)}}]
        plain = gate.merge_topk_n(reports, logit_bound=.15)
        self.assertEqual((plain['root']['opp']['gap_below_logit_bound'], plain['ply2']['own']['gap_below_logit_bound']), (2, 2));self.assertNotIn('root_logit_bound', plain)
        self.assertEqual(gate.merge_topk_n(reports, logit_bound=.15, root_logit_bound=None), plain)
        rooted = gate.merge_topk_n(reports, logit_bound=.15, root_logit_bound=.1)
        self.assertEqual((rooted['root']['opp']['gap_below_logit_bound'], rooted['ply2'], rooted['root_logit_bound']), (1, plain['ply2'], .1))
        self.assertEqual({k: v for k, v in rooted.items() if k not in ('root', 'root_logit_bound')}, {k: v for k, v in plain.items() if k != 'root'})

    def test_reverdict_refuses_traces_of_a_declaration_and_numbers_off_the_frozen_rule(self):
        """A report without declared_bounds sections is re-judged only if nothing in it shows a declaration (the declared tolerance texts and
        sources, a frozen_rule record, a declared bound_rule, a reverdict of one) and its numbers are the frozen rule's: tolerance.rule, logit_bound
        LOGIT_BOUND, value_bound 2 x depth1_noise_floor (1e-4 without a floor), and every kept (c) and (b) bound its class's; a pre-registration
        record given for it is refused. The unchanged synthetic report re-judges as before."""
        def edited(change):
            report = saved_gate();change(report);return report
        quantity = lambda name, **values: lambda r: r['checks']['c']['quantities'][name].update(values)
        cases = dict(
            declared_rule_text=(edited(lambda r: r['tolerance'].__setitem__('rule', gate.DECLARED_TOLERANCE_RULE)), 'left traces'),
            frozen_rule_record=(edited(lambda r: r['tolerance'].__setitem__('frozen_rule', dict(value_bound=.008))), 'left traces'),
            declared_source=(edited(lambda r: r['tolerance'].__setitem__('value_bound_source', 'declared value_bound (--declared-bounds sha256 ...)')), 'left traces'),
            declared_bound_rule=(edited(quantity('leaf_value', bound_rule='declared_value_bound')), 'left traces'),
            reverdict_of_one=(edited(lambda r: r.__setitem__('reverdict', dict(declared_bounds_sha256='a' * 64))), 'left traces'),
            other_rule_text=(edited(lambda r: r['tolerance'].__setitem__('rule', 'values: something else')), 'not the frozen TOLERANCE_RULE'),
            logit_bound=(edited(lambda r: r['tolerance'].__setitem__('logit_bound', .15)), 'is not LOGIT_BOUND'),
            value_bound=(edited(lambda r: r['tolerance'].update(value_bound=.05)), "not the frozen rule's"),
            no_floor=(edited(lambda r: r['tolerance'].__setitem__('depth1_noise_floor', None)), "not the frozen rule's 0.0001"),
            value_quantity=(edited(quantity('leaf_value', bound=.05)), 'checks.c.leaf_value.bound 0.05'),
            turn1_prior=(edited(quantity('t1_own_logits', bound=.15)), 'checks.c.t1_own_logits.bound 0.15'),
            b_value=(edited(lambda r: r['checks'].__setitem__('b', dict(passed=True, values=dict(t1_value=dict(bound=.05))))), 'checks.b.t1_value.bound 0.05'))
        for case, (report, message) in cases.items():
            with self.subTest(case):
                directory = Path(tempfile.mkdtemp(prefix='depthn-reverdict-'));(directory / 'gate.json').write_text(json.dumps(report))
                with patch.object(gate, 'log', lambda *a, **k: None), self.assertRaises(SystemExit) as raised:gate.main(['--reverdict', str(directory / 'gate.json')])
                self.assertIn(message, str(raised.exception.code));self.assertIn('fail closed', str(raised.exception.code));self.assertFalse((directory / gate.REVERDICT_FILE).exists())
        directory = Path(tempfile.mkdtemp(prefix='depthn-reverdict-'));(directory / 'gate.json').write_text(json.dumps(saved_gate()))
        with patch.object(gate, 'log', lambda *a, **k: None), self.assertRaises(SystemExit) as raised:
            gate.main(['--reverdict', str(directory / 'gate.json'), '--declared-bounds-commit', 'b' * 40])
        self.assertIn('records no declaration', str(raised.exception.code))
        code, out, _, _ = reverdicted(saved_gate());self.assertEqual((code, out['status']), (0, 'PASS'));self.assertNotIn('declared_bounds_sha256', out['reverdict'])

    def test_reverdict_refuses_saved_verdicts_that_disagree_with_their_numbers(self):
        """--reverdict recomputes each (c) quantity's within_bound (no non-finite difference, max |diff| <= bound) and within_guard (max |diff| <
        guard) from its saved numbers, the live comparison, and refuses any disagreement: a max |diff| raised past the bound, a non-finite count,
        a verdict flipped either way, a guard verdict flipped. A consistent report re-judges as before."""
        def edited(name, **values):
            report = saved_gate();report['checks']['c']['quantities'][name].update(values);return report
        cases = dict(raised_max=(edited('leaf_value', max_abs_difference=1.), 'leaf_value: within_bound True'),
            nonfinite=(edited('t1_value_head', nonfinite=1), 't1_value_head: within_bound True'),
            flipped_false=(edited('ply2_value_head', within_bound=False), 'ply2_value_head: within_bound False'),
            guard_flipped=(edited('leaf_value', within_guard=True), 'leaf_value: within_guard True'))
        for case, (report, message) in cases.items():
            with self.subTest(case):
                directory = Path(tempfile.mkdtemp(prefix='depthn-reverdict-'));(directory / 'gate.json').write_text(json.dumps(report))
                with patch.object(gate, 'log', lambda *a, **k: None), self.assertRaises(SystemExit) as raised:gate.main(['--reverdict', str(directory / 'gate.json')])
                self.assertIn(message, str(raised.exception.code));self.assertIn('disagree with their own numbers', str(raised.exception.code))
                self.assertFalse((directory / gate.REVERDICT_FILE).exists())
        self.assertEqual(gate.verdict_inconsistencies(saved_gate()['checks']['c']['quantities']), [])

    def test_off_the_champion_path_depth2_and_n2_are_unchanged(self):
        """B1's path (the head override here): a depth-2 full-width gate gates only the depth-N leaf walk's path in oracle consistency and
        carries no merged_depth2_parity; --n2-equivalence --block-invariance stays refused off the champion path (the champion takes it)."""
        code, out = self.main('--depth', '2', '--k-own', '9', '--k-opp', '9', *HEAD_FLAGS);report = json.loads((out / 'gate.json').read_text())
        self.assertEqual((code, report['status']), (0, 'PASS'), report.get('status_reasons'))
        self.assertEqual(set(report['checks']['oracle_consistency']['compared']), {'leaf_path'});self.assertNotIn('merged_depth2_parity', report['header'])
        live = ['--out', 'o', '--d3-root', 'b', '--chunk', '0', '--n2-equivalence', '--block-invariance']
        champion = ['--champion-archive', 'c.npz', '--champion-archive-sha256', 'e' * 64, '--champion-checkpoint-sha256', 'c' * 64]
        for argv in (live, live + list(HEAD_FLAGS), live[:-2] + ['--depth3-diagnostics', '--block-invariance'] + champion):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:gate.parse(argv)
            self.assertEqual(raised.exception.code, 2);self.assertIn('--block-invariance belongs to the gate', stderr.getvalue())
        self.assertTrue(gate.parse(live + champion)[1].block_invariance)

    def test_without_the_flag_nothing_changes(self):
        """No declaration: no declared key in any report section, log line or saved frame; header_block keeps its three keys; declared_rule and
        declared_kwargs are the identity and the empty call. (The byte-for-byte check against the pre-change gate is the probe in the report.)"""
        lines = [];code, out = self.main('--save-arrays', lines=lines);text = (out / 'gate.json').read_text();report = json.loads(text)
        self.assertEqual((code, report['status']), (0, 'PASS'));self.assertNotIn('declared', text);self.assertNotIn('frozen_rule', report['tolerance'])
        self.assertEqual(list(report['header']), ['near_ties', 'rules_changed_after_failure', 'frozen_rule_set'])
        self.assertFalse(any('declared' in json.dumps(x, default=str) for x in lines));self.assertNotIn('DECLARED_BOUNDS', [x['message'] for x in lines])
        _, _, meta = gate.load_frame(out / 'frames-block0-call0-prior-k2.npz');self.assertNotIn('declared_bounds_sha256', meta['evaluator'])
        rule = gate.g2.tolerance_rule(None, None);self.assertIs(gate.declared_rule(rule, None), rule);self.assertEqual(gate.declared_kwargs(None), {})
        self.assertIsNone(gate.parse(['--out', 'o', '--d3-root', 'b', '--chunk', '0'])[1].declared_bounds)


# ---------------------------------------------------------------- --reverdict on a saved n2-equivalence.json

N2_CHAMPION = dict(override=True, kind='champion', archive_sha256=DECLARED_ARCHIVE, precision='bfloat16', checkpoint_sha256=DECLARED_CHECKPOINT)   # CHAMPION_FLAGS' header


def n2_saved(per_config, *, declared=None, evaluator=None):
    """An n2-equivalence.json as run_n2 writes it (its keys, header and statuses) from summarize_n2 on `per_config` ({config: [call]}) under
    `declared`, JSON round-tripped."""
    configs = gate.summarize_n2(per_config, declared=declared);status = gate.n2_run_status(configs)
    section = None if declared is None else gate.declared_section(declared, evaluator)
    report = dict(evaluator=evaluator or dict(override=False), value_scale={}, source=dict(fixture=True), header=gate.header_block(configs=configs, declared=section),
        schema=gate.SCHEMA, mode=gate.N2_MODE, status=status, passed=status == 'PASS', configs=configs, options={}, split_capacity=32, block_seeds=[0],
        **({} if declared is None else dict(status_reasons=gate.provisional_reasons(declared['git_verification'], declared['source_measurement_verification']))),
        rule='per config (run_n2)')
    return json.loads(json.dumps(report, allow_nan=False, default=str))


def n2_reverdicted(report, *extra, repo=None):
    """(exit code or the SystemExit text, n2-equivalence-reverdict.json or None, log lines, path) of --reverdict (and `extra`) on `report`
    written as n2-equivalence.json to a new directory, `repo` (default: this checkout) the checkout git verifies against."""
    directory = Path(tempfile.mkdtemp(prefix='depthn-n2-reverdict-'));path = directory / gate.N2_FILE;path.write_text(json.dumps(report));lines = []
    with patch.object(gate, 'log', lambda message, **values: lines.append(dict(values, message=message))), patch.object(gate, 'REPO_ROOT', repo or gate.REPO_ROOT):
        try:code = gate.main(['--reverdict', str(path), *extra])
        except SystemExit as raised:return str(raised.code), None, lines, path
    return code, json.loads((directory / gate.N2_REVERDICT_FILE).read_text()), lines, path


def measured_declaration(**changes):
    """declaration() for the champion N2_CHAMPION names (its checkpoint pinned) with a readable source measurement, changes applied."""
    measurement = Path(tempfile.mkdtemp(prefix='depthn-measure-')) / 'spread.json';measurement.write_bytes(b'{"spread": 0.003}\n')
    return declaration(**dict(dict(evaluator=dict(declaration()['evaluator'], checkpoint_sha256=DECLARED_CHECKPOINT),
        source_measurement=dict(path=str(measurement), sha256=hashlib.sha256(measurement.read_bytes()).hexdigest())), **changes))


class N2ReverdictTests(FixtureN, unittest.TestCase):
    """--reverdict on a saved n2-equivalence.json (reverdict_n2): the audit of a declared report written where git could not verify its record
    (provisional) clears it on the work tree holding the commit; every saved verdict is re-judged from its own numbers and a disagreement, a
    bound off the declaration or the live path's, or a guard mode the saved evidence does not support is refused; the saved statuses are
    re-derived (a FAIL stays FAIL, an inconsistent one is corrected and recorded, as the gate path re-derives its status); a report without a
    declaration is judged under the frozen rule from each config's own floor. The gate path's reverdict tests above are unchanged."""
    _n2_frame = GateNTests._n2_frame
    main, roots = CommandLineRunTests.main, CommandLineRunTests.roots

    def call(self, *, values=.006, invariance=True):
        """One --n2-equivalence call's results as n2_equivalence_call records them: the stand-in N=2 frame against a copy whose expanded turn-1
        values and head-valued leaves are `values` off (.006: beyond the guard .005, within the value bound .008), (e) and the merged (e)
        through depth2_backups, the synthetic floor FLOOR and spreads SPREAD, one parameter set, the per-position covers and child-block
        invariance against the copy."""
        frame = fake_as_depth2(self.cached('n2', self._n2_frame));other = copy.deepcopy(frame);t1s, t2s = np.asarray(other.t1_status), np.asarray(other.t2_status)
        other.t1_value[t1s == 2] += np.float32(values);other.t2_value[t2s == 4] += np.float32(values)
        found, floats = gate.n2_compare(frame, other);e = gate.decide_roots(frame, frame, self.requests, depth2_backups)
        return dict(a=found, floats=floats, c=dict(depth1_noise_floor=gate.g2._numeric([[0]], [FLOOR], [0.])), e=e, e_merged=copy.deepcopy(e), spreads=[SPREAD] * 10,
            cover=gate.n2_cover(frame, other), actors=1, **(dict(invariance=[gate.g2.block_invariance(frame, other)], invariance_cover=[gate.invariance_cover(frame, other)])
                if invariance else {}))

    def declared_report(self, *, numbers=None, **call):
        """(report, record, audit arguments) of a declared run on the champion as a pod without git writes it: the declaration (default: value
        .008, turn-1 logits .1, near-tie Q .009, root logits .05) committed in a scratch repository with a readable source measurement, and
        git_verification 'not verifiable here' in the run's record (provisional)."""
        record = committed(measured_declaration(bounds=numbers or bounds(near_tie_q_bound=.009, root_logit_bound=.05)))
        d = dict(load(record), git_verification=dict(status='not verifiable here', detail='/work/gate is not the top of a git work tree'))
        report = n2_saved({'full_width': [self.call(**call)], 'top2': [self.call(**call)]}, declared=d, evaluator=N2_CHAMPION)
        return report, record, ('--declared-bounds-commit', record.commit, '--declared-bounds-sha256', record.sha256)

    def test_a_declared_report_audits_on_the_work_tree_holding_its_commit(self):
        """The provisional PASS (guards digest-certified, the root prior at its declared bound, (e)'s q_noise_bound the near-tie Q bound): the
        audit re-judges it PASS with git and the source measurement verified and no reason left, every config and the header as saved, written
        beside it and never overwritten; on a checkout without git --reverdict alone keeps the reason and the audit refuses."""
        report, record, audit = self.declared_report();top = report['configs']['top2']
        self.assertEqual((report['status'], [c['status'] for c in report['configs'].values()]), ('PASS', ['PASS', 'PASS']))
        self.assertEqual(len(report['status_reasons']), 1);self.assertTrue(report['status_reasons'][0].startswith('provisional: git_verification not verifiable here'))
        self.assertEqual(top['tolerance']['guard_declared']['modes'], dict.fromkeys(('floats.t1_value', 'floats.t2_value', 'block_invariance.t1_value',
            'block_invariance.t2_value'), 'digest_certified'))
        self.assertEqual((top['floats']['root_peer_logits']['bound'], top['floats']['root_peer_logits']['bound_rule'], top['e']['q_noise_bound'], top['e_merged_depth2']['q_noise_bound']),
            (.05, 'declared_root_logit_bound', .009, .009))
        code, out, lines, path = n2_reverdicted(report, *audit, repo=record.root);rv = out['reverdict']
        self.assertEqual((code, out['status'], out['passed'], out['status_reasons']), (0, 'PASS', True, []))
        self.assertEqual((rv['source_status'], rv['source_status_reasons'], rv['configs'], rv['changed']), ('PASS', report['status_reasons'], dict(full_width='PASS', top2='PASS'), []))
        self.assertEqual((rv['declared_bounds_sha256'], rv['declared_bounds_commit'], rv['declared_bounds_reverified']['git']['status'],
            rv['declared_bounds_reverified']['source_measurement']['status']), (record.sha256, record.commit, 'verified', 'verified'))
        self.assertIn('nothing was re-priced', rv['note']);self.assertEqual(rv['rule'], gate.N2_REVERDICT_RULE)
        self.assertEqual((out['configs'], out['header']), (report['configs'], report['header']))   # every saved verdict re-derived as saved
        self.assertEqual(list(out)[:4], ['evaluator', 'value_scale', 'source', 'header'])
        line = next(x for x in lines if x['message'] == 'REVERDICT')
        self.assertEqual((line['old_status'], line['new_status'], line['status_reasons'], line['changed'], line['out']), ('PASS', 'PASS', [], [], str(path.with_name(gate.N2_REVERDICT_FILE))))
        before = path.with_name(gate.N2_REVERDICT_FILE).read_bytes()
        with patch.object(gate, 'log', lambda *a, **k: None), patch.object(gate, 'REPO_ROOT', record.root), self.assertRaises(SystemExit) as raised:
            gate.main(['--reverdict', str(path), *audit])
        self.assertIn('never overwritten', str(raised.exception.code));self.assertEqual(path.with_name(gate.N2_REVERDICT_FILE).read_bytes(), before)
        exported = Path(tempfile.mkdtemp(prefix='depthn-exported-'))   # a checkout that is not a git work tree
        code, out, _, _ = n2_reverdicted(report, repo=exported)
        self.assertEqual((code, out['status'], len(out['status_reasons'])), (0, 'PASS', 1));self.assertTrue(out['status_reasons'][0].startswith('provisional: git_verification'))
        text, out, _, path = n2_reverdicted(report, *audit, repo=exported)
        self.assertIsNone(out);self.assertIn('requires git_verification verified', text);self.assertFalse(path.with_name(gate.N2_REVERDICT_FILE).exists())

    def test_the_audit_refuses_what_the_saved_numbers_or_the_record_do_not_support(self):
        """Each edit of the declared report (or a commit / sha256 given that is not the recorded one) is refused and nothing is written: the
        declaration's text and sections, a config's tolerance off the declaration, a bound off the live path's, a verdict off its own numbers,
        a guard mode the saved digest findings or per-position evidence do not support, an (e) whose passed its numbers contradict; a field the
        verdict needs, missing."""
        report, record, audit = self.declared_report();h = lambda r: r['header']['declared_bounds'];top = lambda r: r['configs']['top2']
        def stripped(r):
            r['header'].pop('declared_bounds')
            for c in r['configs'].values():c['tolerance'].pop('declared_bounds')
        def evidence(section, **values):
            return lambda r: [m['guard_evidence'].update(values) for m in top(r)[section].values() if 'guard_evidence' in m]
        cases = dict(
            other_commit=(None, ('--declared-bounds-commit', 'b' * 40), 'is not the recorded'), other_sha=(None, ('--declared-bounds-sha256', '0' * 64), 'is not the recorded'),
            content_text=(lambda r: h(r).__setitem__('content_text', h(r)['content_text'].replace('0.008', '0.5')), audit, 'does not hash'),
            header=(lambda r: r['header'].pop('declared_bounds'), audit, 'configs.top2.tolerance.declared_bounds without header.declared_bounds'),
            stripped=(stripped, (), 'left traces'), evaluator=(lambda r: r['evaluator'].__setitem__('archive_sha256', '6' * 64), audit, 'evaluator.archive_sha256'),
            config_value_bound=(lambda r: top(r)['tolerance'].__setitem__('value_bound', .5), audit, 'configs.top2.tolerance value_bound / logit_bound [0.5, 0.1]'),
            config_root_bound=(lambda r: top(r)['tolerance'].pop('root_logit_bound'), audit, 'configs.top2.tolerance.root_logit_bound None'),
            config_sha=(lambda r: top(r)['tolerance']['declared_bounds'].__setitem__('sha256', '0' * 64), audit, 'configs.top2.tolerance.declared_bounds does not name'),
            q_noise_bound=(lambda r: top(r)['tolerance'].__setitem__('q_noise_bound', .008), audit, 'configs.top2.tolerance.q_noise_bound 0.008 is not 0.009'),
            e_q_noise_bound=(lambda r: top(r)['e_merged_depth2'].__setitem__('q_noise_bound', .008), audit, 'configs.top2.e_merged_depth2.q_noise_bound'),
            float_bound=(lambda r: top(r)['floats']['t1_own_logits'].__setitem__('bound', .2), audit, "configs.top2.floats.t1_own_logits: bound, guard_bound, bound_rule"),
            root_rule=(lambda r: top(r)['block_invariance']['floats']['root_peer_logits'].pop('bound_rule'), audit, "are not the live path's [0.05, None, 'declared_root_logit_bound']"),
            guard_bound=(lambda r: top(r)['floats']['t2_value'].__setitem__('guard_bound', .007), audit, 'configs.top2.floats.t2_value: bound, guard_bound'),
            within_bound=(lambda r: top(r)['floats']['t2_value'].__setitem__('within_bound', False), audit, 'disagree with their own numbers'),
            within_guard=(lambda r: top(r)['floats']['t1_value'].__setitem__('within_guard', True), audit, 'disagree with their own numbers'),
            raised_max=(lambda r: top(r)['block_invariance']['floats']['t2_value'].__setitem__('max_abs_difference', .5), audit, 'disagree with their own numbers'),
            float_passed=(lambda r: top(r)['floats']['t1_own_logits'].__setitem__('passed', False), audit, 'configs.top2.floats.t1_own_logits: passed False, re-derived True'),
            digest_mismatch=(lambda r: top(r)['discrete']['mismatched'].__setitem__('t2_digest', 1), audit, "configs.top2.floats.t2_value: saved 'digest_certified', re-derived 'gated'"),
            inexact_outside=(lambda r: top(r)['floats']['t1_value']['guard_evidence'].__setitem__('value_positions_outside_digest_inexact', 1), audit, 'saved guard modes differ'),
            invariance_digest=(lambda r: top(r)['block_invariance']['floats']['t2_value']['guard_evidence'].__setitem__('digest_mismatched', 1), audit, 'saved guard modes differ'),
            two_actors=(evidence('floats', evaluator_actors=2), audit, 'saved guard modes differ'), mode_gated=(lambda r: top(r)['floats']['t2_value'].__setitem__('guard_mode', 'gated'), audit,
                'saved guard modes differ'),
            record_gated=(lambda r: top(r)['tolerance']['guard_declared']['modes'].__setitem__('floats.t2_value', 'gated'), audit, 'configs.top2.tolerance.guard_declared.modes'),
            e_passed=(lambda r: top(r)['e'].__setitem__('max_abs_delta_q', .02), audit, 'configs.top2.e: passed True, q_noise_within_bound True; re-derived False, False'),
            merged_passed=(lambda r: top(r)['e_merged_depth2'].__setitem__('argmax_q_differs_non_tie', 1), audit, 'configs.top2.e_merged_depth2: passed True'))
        for case, (change, extra, message) in cases.items():
            with self.subTest(case):
                r = copy.deepcopy(report)
                if change is not None:change(r)
                text, out, _, path = n2_reverdicted(r, *extra, repo=record.root)
                self.assertIsNone(out, case);self.assertIn(message, text);self.assertIn('fail closed', text);self.assertFalse(path.with_name(gate.N2_REVERDICT_FILE).exists())
        def drop(*keys):
            def change(r):
                node = r
                for key in keys[:-1]:node = node[key]
                node.pop(keys[-1])
            return change
        missing = dict(within_bound=(drop('configs', 'top2', 'floats', 't2_value', 'within_bound'), 'configs.top2.floats.t2_value.within_bound'),
            within_guard=(drop('configs', 'top2', 'block_invariance', 'floats', 't1_value', 'within_guard'), 'configs.top2.block_invariance.floats.t1_value.within_guard'),
            max_dq=(drop('configs', 'full_width', 'e', 'max_abs_delta_q'), 'configs.full_width.e.max_abs_delta_q (a number)'),
            q_noise_bound=(drop('configs', 'top2', 'tolerance', 'q_noise_bound'), 'configs.top2.tolerance.q_noise_bound (a number)'),
            mismatched=(drop('configs', 'top2', 'discrete', 'mismatched'), 'configs.top2.discrete.mismatched (a JSON object)'),
            discrete_equal=(drop('configs', 'top2', 'block_invariance', 'discrete_equal'), 'configs.top2.block_invariance.discrete_equal'),
            world_reports=(lambda r: top(r).__setitem__('world_reports_equal', 'yes'), 'configs.top2.world_reports_equal (true or false)'),
            digest_count=(lambda r: top(r)['discrete']['compared'].__setitem__('t1_digest_own', '9'), 'configs.top2.discrete.compared (counts)'),
            evidence_count=(lambda r: top(r)['floats']['t1_value']['guard_evidence'].__setitem__('value_positions_outside_digest', 'none'),
                'configs.top2.floats.t1_value.guard_evidence.value_positions_outside_digest (an integer or null)'),
            differing=(lambda r: top(r)['block_invariance'].__setitem__('differing', dict(t2_digest=dict(count='1'))), 'configs.top2.block_invariance.differing'),
            not_an_object=(lambda r: r['configs'].__setitem__('top2', [1]), 'configs.top2 (a JSON object)'),
            configs=(drop('configs'), 'configs'), schema=(lambda r: r.__setitem__('schema', 'v0'), 'schema'))
        for case, (change, message) in missing.items():
            with self.subTest(case):
                r = copy.deepcopy(report);change(r);text, out, _, path = n2_reverdicted(r, *audit, repo=record.root)
                self.assertIsNone(out, case);self.assertIn('lacks what the verdict needs', text);self.assertIn(message, text)
                self.assertFalse(path.with_name(gate.N2_REVERDICT_FILE).exists())

    def test_the_saved_statuses_are_rederived(self):
        """A saved FAIL (values .012 off, beyond the value bound) stays FAIL with nothing changed. A saved status that its own parts contradict is
        re-derived from them (n2_status, n2_run_status) as the gate path re-derives its status, the saved one kept as source_status and each
        differing field listed in `changed`: a config saved FAIL whose parts all pass is PASS, a mismatch count under a saved PASS fails the
        config (discrete.passed re-derived), an (e) that compared no root makes it INCOMPLETE."""
        failing, record, audit = self.declared_report(values=.012);top = failing['configs']['top2']
        self.assertEqual((failing['status'], top['status'], top['floats']['t2_value']['within_bound'], top['floats']['t2_value']['guard_mode']), ('FAIL', 'FAIL', False, 'digest_certified'))
        code, out, _, _ = n2_reverdicted(failing, *audit, repo=record.root)
        self.assertEqual((code, out['status'], out['reverdict']['configs'], out['reverdict']['changed'], out['status_reasons']), (1, 'FAIL', dict(full_width='FAIL', top2='FAIL'), [], []))
        self.assertEqual(out['configs'], failing['configs'])
        report, record, audit = self.declared_report()
        def edited(change):
            r = copy.deepcopy(report);change(r);return n2_reverdicted(r, *audit, repo=record.root)
        code, out, lines, _ = edited(lambda r: (r['configs']['top2'].update(status='FAIL', passed=False), r.update(status='FAIL', passed=False)))
        self.assertEqual((code, out['status'], out['configs']['top2']['status'], out['reverdict']['source_status'], out['reverdict']['source_configs']['top2']),
            (0, 'PASS', 'PASS', 'FAIL', 'FAIL'))
        self.assertEqual(out['reverdict']['changed'], [dict(config='top2', field='status', before='FAIL', after='PASS'), dict(config='top2', field='passed', before=False, after=True)])
        self.assertEqual(next(x for x in lines if x['message'] == 'REVERDICT')['changed'], out['reverdict']['changed'])
        code, out, _, _ = edited(lambda r: r['configs']['full_width']['discrete']['mismatched'].__setitem__('t1_status', 1))
        self.assertEqual((code, out['status'], out['reverdict']['configs'], out['configs']['full_width']['discrete']['passed']), (1, 'FAIL', dict(full_width='FAIL', top2='PASS'), False))
        self.assertEqual([(c['field'], c['before'], c['after']) for c in out['reverdict']['changed']],
            [('status', 'PASS', 'FAIL'), ('passed', True, False), ('discrete.passed', True, False)])
        code, out, _, _ = edited(lambda r: r['configs']['top2']['e'].__setitem__('roots_compared', 0))
        self.assertEqual((code, out['status'], out['configs']['top2']['status'], out['configs']['top2']['status_reasons']), (1, 'INCOMPLETE', 'INCOMPLETE', ['(e) compared no root']))

    def test_a_report_without_a_declaration_is_judged_under_the_frozen_rule(self):
        """No declaration: each config's numbers must be the frozen rule's from its own floor (value bound 2 x FLOOR, LOGIT_BOUND, q_noise_bound
        the value bound, the floats' bounds their class's) and no trace of a declaration may remain; then it is re-judged as saved (PASS within
        the guard, FAIL beyond it: the frozen rule gates the guard); a pre-registration record given for it is refused."""
        report = n2_saved({'full_width': [self.call(values=.002)], 'top2': [self.call(values=.002)]})
        self.assertEqual((report['status'], report['configs']['top2']['tolerance']['value_bound'], 'status_reasons' in report), ('PASS', 2 * float(np.float32(FLOOR)), False))
        code, out, _, _ = n2_reverdicted(report)
        self.assertEqual((code, out['status'], out['status_reasons'], out['reverdict']['changed'], out['configs']), (0, 'PASS', [], [], report['configs']))
        self.assertFalse(any(k.startswith('declared') for k in out['reverdict']));self.assertNotIn('declared_bounds', out['header'])
        frozen = n2_saved({'x': [self.call()]});code, out, _, _ = n2_reverdicted(frozen)   # .006: beyond the guard, which the frozen rule gates
        self.assertEqual((code, out['status'], {q for q, m in out['configs']['x']['floats'].items() if not m['passed']}), (1, 'FAIL', {'t1_value', 't2_value'}))
        text, out, _, _ = n2_reverdicted(report, '--declared-bounds-commit', 'b' * 40);self.assertIsNone(out);self.assertIn('records no declaration', text)
        top = lambda r: r['configs']['top2']
        cases = dict(value_bound=(lambda r: top(r)['tolerance'].__setitem__('value_bound', .05), "configs.top2.tolerance.value_bound 0.05 is not the frozen rule's"),
            no_floor=(lambda r: top(r)['tolerance'].__setitem__('depth1_noise_floor', None), "not the frozen rule's 0.0001"),
            logit_bound=(lambda r: top(r)['tolerance'].__setitem__('logit_bound', .15), 'configs.top2.tolerance.logit_bound 0.15 is not LOGIT_BOUND'),
            rule_text=(lambda r: top(r)['tolerance'].__setitem__('rule', gate.DECLARED_TOLERANCE_RULE), 'left traces'),
            frozen_record=(lambda r: top(r)['tolerance'].__setitem__('frozen_rule', {}), 'configs.top2.tolerance.frozen_rule'),
            bound_rule=(lambda r: top(r)['floats']['root_peer_logits'].__setitem__('bound_rule', 'declared_root_logit_bound'), 'configs.top2.floats.root_peer_logits.bound_rule'),
            guard_mode=(lambda r: top(r)['block_invariance']['floats']['t1_value'].__setitem__('guard_mode', 'digest_certified'), 'block_invariance.floats.t1_value.guard_mode'),
            reverdict_of_one=(lambda r: r.__setitem__('reverdict', dict(declared_bounds_sha256='a' * 64)), 'reverdict.declared_bounds_sha256'),
            q_noise_bound=(lambda r: top(r)['tolerance'].__setitem__('q_noise_bound', .009), 'configs.top2.tolerance.q_noise_bound 0.009 is not'),
            float_bound=(lambda r: top(r)['floats']['t1_own_logits'].__setitem__('bound', .15), "live path's [0.1, None, None]"),
            invariance_bound=(lambda r: top(r)['block_invariance']['floats']['t2_value'].__setitem__('bound', .05), 'configs.top2.block_invariance.floats.t2_value'))
        for case, (change, message) in cases.items():
            with self.subTest(case):
                r = copy.deepcopy(report);change(r);text, out, _, path = n2_reverdicted(r)
                self.assertIsNone(out, case);self.assertIn(message, text);self.assertIn('fail closed', text);self.assertFalse(path.with_name(gate.N2_REVERDICT_FILE).exists())

    def test_live_n2_reports_reverdict_as_written(self):
        """run_n2's own n2-equivalence.json through main(): without a declaration (the stand-in, which carries no world_reports, FAILs) and
        under a declaration written where git could not verify it (a checkout without git; the real pricer when on the tree: PASS); --reverdict
        reproduces every config, the latter's audit on the work tree holding the commit clearing its provisional reason."""
        code, out = self.main('--n2-equivalence', '--n2-k', '2');report = json.loads((out / gate.N2_FILE).read_text())
        self.assertEqual((code, report['status']), (1, 'FAIL'))
        with patch.object(gate, 'log', lambda *a, **k: None):code = gate.main(['--reverdict', str(out / gate.N2_FILE)])
        again = json.loads((out / gate.N2_REVERDICT_FILE).read_text())
        self.assertEqual((code, again['status'], again['reverdict']['changed'], again['configs']), (1, 'FAIL', [], report['configs']))
        record = committed(measured_declaration());exported = Path(tempfile.mkdtemp(prefix='depthn-exported-'))
        target = exported / gate.DECLARED_SUBDIR / 'champion.json';target.parent.mkdir(parents=True);target.write_bytes(record.path.read_bytes())
        roots = self.roots;self.roots = lambda: setattr((r := roots()), 'evaluator', dict(N2_CHAMPION)) or r   # what load_roots resolves for CHAMPION_FLAGS
        try:
            with patch.object(gate, 'REPO_ROOT', exported):
                code, out = self.main('--n2-equivalence', '--n2-k', '2', *CHAMPION_FLAGS, '--declared-bounds', str(target), '--declared-bounds-sha256', record.sha256,
                    '--declared-bounds-commit', record.commit, real=HAVE_PRICER)
        finally:self.roots = roots
        report = json.loads((out / gate.N2_FILE).read_text());want = 'PASS' if HAVE_PRICER else 'FAIL'
        self.assertEqual((code, report['status'], report['header']['declared_bounds']['git_verification']['status']), (int(want != 'PASS'), want, 'not verifiable here'))
        self.assertTrue(report['status_reasons'][0].startswith('provisional: git_verification'))
        self.assertEqual({c['tolerance']['guard_declared']['modes']['floats.t2_value'] for c in report['configs'].values()}, {'gated'})   # the fixture prices with two parameter sets
        code, again, _, _ = n2_reverdicted(report, '--declared-bounds-commit', record.commit, '--declared-bounds-sha256', record.sha256, repo=record.root)
        self.assertEqual((code, again['status'], again['status_reasons'], again['reverdict']['changed'], again['configs']), (int(want != 'PASS'), want, [], [], report['configs']))
        self.assertEqual(again['reverdict']['declared_bounds_reverified']['git']['status'], 'verified')


# ---------------------------------------------------------------- the frozen default paths, byte for byte

REFERENCE_DROP = ('costs', 'oracle_cost', 'timing', 'pricer_timing', 'plies_work', 'compiles', 'compiled')   # timing and compile records
NOISY_VALUE, NOISY_LOGIT = 2. ** -16, 2. ** -12   # the oracle's head offsets: non-zero depth-1, root-logit and turn-1 logit floors


def reference_normalized(value, masks):
    """A report or log line as compared: REFERENCE_DROP keys and keys ending in 'seconds' dropped, each run directory in `masks` masked."""
    if isinstance(value, dict):return {k: reference_normalized(v, masks) for k, v in value.items() if k not in REFERENCE_DROP and not k.endswith('seconds')}
    if isinstance(value, list):return [reference_normalized(v, masks) for v in value]
    if isinstance(value, str):
        for mask in masks:value = value.replace(mask, '<ROOT>')
    return value


@contextlib.contextmanager
def noisy_oracle_heads():
    """depth2_oracle.heads (both gates' oracles evaluate through it, no pricer does) with NOISY_LOGIT added to every logit and NOISY_VALUE to
    every value: exact dyadic offsets, so the floors and every bound derived from them are non-zero and deterministic."""
    from gpu_engine import depth2_oracle
    real = depth2_oracle.heads
    def heads(forward, params, inputs, actors):
        logits, values, usable = real(forward, params, inputs, actors);return logits + np.float32(NOISY_LOGIT), values + np.float32(NOISY_VALUE), usable
    with patch.object(depth2_oracle, 'heads', heads):yield


def reference_runs(test, root):
    """{run: {file: sha256 of its normalized canonical JSON}} of the default paths on test's fixture under noisy_oracle_heads: d3_depthn_gate on
    the bank actors (the stand-in at depth 3 with --mutation-test, the real pricer at depth 2 full width with --block-invariance and under
    --n2-equivalence, --reverdict of the first) and d3_depth2_gate (full width with --block-invariance, and top-3). Writes each normalized file
    under root/normalized for a look at a mismatch."""
    from gpu_engine import d3_depth2_gate as g2
    out, digests = {}, {}
    def record(name, directory, lines, files):
        masks = [str(directory), str(Path(directory).resolve())];target = root / 'normalized' / name;target.mkdir(parents=True);digests[name] = {}
        for file, value in [('log.json', lines)] + [(f, json.loads((Path(directory) / f).read_text())) for f in files]:
            text = json.dumps(reference_normalized(value, masks), indent=1, default=str) + '\n';(target / file).write_text(text)
            digests[name][file] = hashlib.sha256(text.encode()).hexdigest()
    def depth2_main(*argv, directory, lines):
        provider = SimpleNamespace(executor=test.ex, materializer=test.tables, recipe=test.recipe, source=test.tables.source)
        model = type('Model', (), dict(arm='D'))()   # hashable, as object(), with the arm the evaluator header names
        # The last 20 requests: the merged gate takes its guard's p10 over EVERY cell_a2_spread, and over all 27 that p10 is 0, where its
        # finalize_numeric divides by the zero guard (it raises; the depth-N gate's p10 excludes zero spreads). Over these 20 it is not.
        loaded = SimpleNamespace(provider=provider, plan=SimpleNamespace(model=model, historical_window=2), params=PARAMS, groups=[list(test.requests[-20:])],
            frame_pricer=FixtureN.frame_pricer(test), source=dict(fixture=True), labels=SimpleNamespace(sha256='l' * 64), category_sha256='k' * 64,
            snapshot=dict(iteration=-1, archive='run/actor--1.npz', archive_sha256='4' * 64, checkpoint='c.pt', checkpoint_sha256='3' * 64, model_sha256='5' * 64))
        with test.engine(), patch.object(g2, 'bank_chunk_requests', lambda args: loaded), patch.object(g2, 'log', lambda m, **k: lines.append(dict(k, message=m))):
            return g2.main(['--out', str(directory), '--d3-root', 'x', '--chunk', '0', '--depth2-capacity', '32', '--oracle-chunk', '64', *argv])
    with noisy_oracle_heads():
        for name, argv, real, files in (('depthn_standin_prior3_mutation', ('--mutation-test',), False, ('gate.json', gate.MUTATION_FILE)),
                ('depthn_real_depth2_full_invariance', ('--depth', '2', '--k-own', '9', '--k-opp', '9', '--block-invariance'), True, ('gate.json',)),
                ('depthn_real_n2', ('--n2-equivalence', '--n2-k', '2'), True, ('n2-equivalence.json',))):
            lines = [];directory = root / 'runs' / name;code, _ = CommandLineRunTests.main(test, *argv, real=real, out=directory, lines=lines)
            lines.append(dict(message='EXIT', code=code));record(name, directory, lines, files)
        lines = [];directory = root / 'runs' / 'depthn_standin_prior3_mutation'
        with patch.object(gate, 'log', lambda m, **k: lines.append(dict(k, message=m))):lines.append(dict(message='EXIT', code=gate.main(['--reverdict', str(directory / 'gate.json')])))
        record('depthn_reverdict', directory, lines, (gate.REVERDICT_FILE,))
        for name, argv in (('depth2_full_invariance', ('--block-invariance',)), ('depth2_top3', ('--k-own', '3', '--k-opp', '3'))):
            lines = [];directory = root / 'runs' / name;lines.append(dict(message='EXIT', code=depth2_main(*argv, directory=directory, lines=lines)))
            record(name, directory, lines, ('gate.json',))
    return digests


class DefaultPathReferenceTests(FixtureN, unittest.TestCase):
    """The frozen default paths byte for byte: reference_runs' digests (reports, mutation-test.json, gate-reverdict.json and every log line,
    timing and run directories aside) must equal REFERENCE, recorded by the same harness on 1a3b2dd3, the commit before --declared-bounds, on
    the gpu_engine of that commit. The oracle's heads are offset so the floors are non-zero (a doubled q_noise default, an extra field or text
    on the default path changes a digest). A deliberate change to a default path updates REFERENCE in its own reviewed commit."""
    roots = CommandLineRunTests.roots   # the bank actors (no override): the default path
    REFERENCE = {   # reference_runs on 1a3b2dd3 (the gpu_engine of that commit, this harness)
        'depthn_standin_prior3_mutation': {'log.json': '526ce50e796a2085301ea23ca6f81d781f6fff5d535d4079bc8ae8709acaff26',
            'gate.json': '9b4071a272a1f2a637786551d7b7e232aa606b9f4b188bc9a15f5532a704c52b',
            'mutation-test.json': 'c69beb3648e6bc8f41c2e7831f4e3805aa7a98dc6376eefc5b26d5a778c350f5'},
        'depthn_real_depth2_full_invariance': {'log.json': '8f3d68285b062c1e219e018e573347ab556d95d23ffd897ad9799664019f6664',
            'gate.json': '566c88b1dfb818d39103f2a2cf5138e137b18e7930f91d7e06bb22432b9c44ce'},
        'depthn_real_n2': {'log.json': '28e426902c025f562907e0c6cd1223c3a2cb1be09a906cde50317004c66a450a',
            'n2-equivalence.json': '0e6c7c3aa4887cd36a30f54d2ace860389a2063e65f88de76d987ee93e7e7493'},
        'depthn_reverdict': {'log.json': 'eeafbaf75f8320e18af0a324ba0540d960cb9717a8710fd0eb72e6e9d8351efa',
            'gate-reverdict.json': 'daaf0ac7f4b689d58a1f2e4827fedc1664bff8cbb44d2b07fde515ca2e5d4e38'},
        'depth2_full_invariance': {'log.json': '9197a81e17c9537e4c9452ae4a23c799aa64ac45f6429f5ac9934e0e884007c1',
            'gate.json': '5807bd64727ba7490eae9adcd3e3b468f73bd177d54a928ac1e56721e43e3a19'},
        'depth2_top3': {'log.json': 'db15ece5fcce98e5decab35a5a4b42972671f7eb0d6ae89e2dacfe125688ed06',
            'gate.json': '1ff2a5298a1e3809a25d17cd22e85b05e11564c56cabf680a095baf43a03cc2b'},
    }

    def test_default_paths_match_the_reference(self):
        root = Path(tempfile.mkdtemp(prefix='gate-default-reference-'));got = reference_runs(self, root)
        for name in sorted(set(got) | set(self.REFERENCE)):
            with self.subTest(name):self.assertEqual(got.get(name), self.REFERENCE.get(name), f'normalized outputs under {root / "normalized" / name}')
        tolerance = lambda run, f='gate.json': json.loads((root / 'normalized' / run / f).read_text())['tolerance']   # the floors it rests on are non-zero
        self.assertGreater(tolerance('depthn_standin_prior3_mutation')['depth1_noise_floor'], 0.);self.assertGreater(tolerance('depth2_full_invariance')['q_noise_bound'], 0.)
        self.assertGreater(tolerance('depthn_real_depth2_full_invariance')['turn1_logit_floor'], 0.)

    def test_the_topk_gap_count_is_strict(self):
        """merge_topk_n counts gaps strictly below the logit bound: a gap of exactly LOGIT_BOUND is not below it."""
        topk = gate.merge_topk_n([{'ply2': {'own': dict(gaps=[gate.g2.LOGIT_BOUND, .05, .2], oracle_prior_set_differs=0)}}])
        self.assertEqual(topk['ply2']['own']['gap_below_logit_bound'], 1)


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
