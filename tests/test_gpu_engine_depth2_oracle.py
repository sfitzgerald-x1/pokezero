"""Depth-2 oracle and gate on traceable inert worlds/actors (CPU only).

The fake engine records each step's dispatched actions in the state, draws chance from a seed that
the engine re-keys on every step independently of the actions (as the real key split), and puts the
value, the last actions and the public turn into the encoder features, so heads and digests expose
what the oracle dispatched and which chance it used. Its value is an integer (1/SCALE units) scaled by a
power of two, so every float is exact and no compile context (fusion, FMA) can move a bit.
"""
from dataclasses import replace
import collections
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import NamedTuple
import unittest
import uuid
from unittest.mock import patch

import jax.numpy as j
import numpy as np

from pokezero import v5_contract, v5_gumbel
from gpu_engine import d3_depth2_gate as gate, depth2_oracle as reference
from gpu_engine import device_public_teacher as serial, device_public_teacher_batch as batch
from gpu_engine import device_public_worlds as plans, device_public_world_state as mapping
from tests.test_gpu_engine_device_public_teacher import facts

BASE = (True, True, True) + (False,) * 6
SWITCH = (False,) * 4 + (True, True) + (False,) * 3
LCG = (1103515245, 12345)
SCALE = 1024


def units(n):
    return float(np.float32(n) / np.float32(SCALE))


class State(NamedTuple):
    value: object
    done: object
    winner: object
    mode: object
    seed: object
    owner: object
    turn: object
    own_act: object
    peer_act: object


class Public(NamedTuple):
    clock: object
    turn: object


class Observation(NamedTuple):
    legal_action_mask: object
    error: object


class Leaf(NamedTuple):
    observation: object
    inputs: object
    observer_sides: object


class Step(NamedTuple):
    state: object
    invalid_actions: object


class After(NamedTuple):
    step: object
    trackers: object


class Prediction(NamedTuple):
    policy_logits: object
    value: object
    valid: object


def forward(params, inputs, model=None, **kw):
    x = inputs['numeric_features'][:, 0, 0]
    return Prediction(j.broadcast_to(params['logits'], (len(x), 9)).astype(j.float32), x[:, 0] + params['offset'], j.ones(len(x), bool))


PARAMS = ({'offset': j.asarray(0., j.float32), 'logits': j.asarray([0, 1, 1, .5, .5, .25, 0, 0, 0], j.float32)},
    {'offset': j.asarray(.07, j.float32), 'logits': -j.arange(9, dtype=j.float32) / 9})


def executor():
    """Modes: 1/2/7 child terminal win/draw/bad winner; 3 invalid child (1,1); 4 child encoder error; 5 nonfinite
    child value; 6 opponent always waits; 8 child: opponent waits, owner forced to switch (4,5); 12 child: owner
    waits, opponent forced; 9 leaf terminal (a2 0 loss, a2 1 draw); 10 invalid leaf a2=2; 11 leaf encoder error;
    child, opponent's view only: 13 encoder error, 14 non-finite prior (requested), 15 non-finite prior while it waits,
    16 both an encoder error and a non-finite prior."""
    def legal(state):
        owner, mode, turn = state.owner[0], state.mode[0], state.turn[0]
        child = turn == 1
        own = j.where((mode == 12) & child, False, j.where((mode == 8) & child, j.asarray(SWITCH), j.asarray(BASE)))
        foe = j.where((mode == 6) | (((mode == 8) | (mode == 15)) & child), False, j.where((mode == 12) & child, j.asarray(SWITCH), j.asarray(BASE)))
        rows = j.where(owner == 0, j.stack([own, foe]), j.stack([foe, own]))
        return j.where(state.done[0], False, rows)[None]

    def observe(state, public, owner, mask):
        if public.clock.ndim != 0:raise ValueError('one-world namespace was flattened')
        mode, child, peer = state.mode, public.turn == 1, owner != state.owner
        error = j.where(((mode == 4) & child) | ((mode == 11) & (public.turn == 2)) | (((mode == 13) | (mode == 16)) & child & peer), 1, 0)
        value = (state.value + (owner - state.owner) * 10).astype(j.float32) * (1. / SCALE)
        value = j.where(((mode == 5) & (public.turn >= 1)) | (((mode == 14) | (mode == 15) | (mode == 16)) & child & peer), j.nan, value)
        feature = j.stack((value, public.clock + j.zeros(1), state.own_act.astype(j.float32),
            state.peer_act.astype(j.float32), public.turn.astype(j.float32)), axis=-1)
        return Leaf(Observation(mask, error), {'numeric_features': feature[:, None, None, :]}, owner)

    def advance(state, public, actions, chance):
        owner = state.owner[0];a = actions[:, owner];r = actions[:, 1 - owner];mode = state.mode;turn = state.turn + 1
        value = state.value + a * 30 + r * 20 + chance
        child, leaf = turn == 1, turn == 2
        child_end = child & ((mode == 1) | (mode == 2) | (mode == 7))
        leaf_end = leaf & (mode == 9) & (a >= 0) & (a <= 1)
        winner = j.where(child_end, j.where(mode == 2, 2, j.where(mode == 7, 9, state.owner)),
            j.where(leaf_end, j.where(a == 0, 1 - state.owner, 2), state.winner))
        invalid = (child & (mode == 3) & (a == 1) & (r == 1)) | (leaf & (mode == 10) & (a == 2))
        seed = state.seed * j.uint32(LCG[0]) + j.uint32(LCG[1])   # re-keyed every step, whatever the actions
        after = state._replace(value=value, done=state.done | child_end | leaf_end, winner=winner, seed=seed, turn=turn, own_act=a, peer_act=r)
        return After(Step(after, invalid), Public(public.clock, public.turn + 1))

    ex = serial.Executor(legal, observe, lambda obs, state, seat, action: action, lambda state: (state.seed % 17).astype(j.int32), advance)
    return replace(ex, joint=serial._make_joint(ex))


def world(mode, owner, seed, clock=3.):
    state = State(j.asarray([200], j.int32), j.asarray([False]), j.asarray([0], j.int32), j.asarray([mode], j.int32),
        j.asarray([seed % 2**32], j.uint32), j.asarray([owner], j.int32), j.asarray([0], j.int32), j.zeros(1, j.int32), j.zeros(1, j.int32))
    return state, Public(j.asarray(clock, j.float32), j.asarray([0], j.int32))


def chance(seed):
    return seed % 17


def rekey(seed):
    return (seed * LCG[0] + LCG[1]) % 2**32


def simple_requests(ex, specs):
    """Duck-typed requests and world pairs: specs = ((mode, owner, (seed0, seed1)), ...)."""
    requests, worlds = [], []
    for i, (mode, owner, seeds) in enumerate(specs):
        state, public = world(mode, owner, seeds[0])
        inputs = ex.observe(state, public, j.asarray([owner]), ex.legal(state)[:, owner]).inputs
        address = v5_contract.DecisionAddress(100 + i, 4, owner)
        requests.append(SimpleNamespace(plans=SimpleNamespace(root=SimpleNamespace(address=address)), root_logits=(0.,) * 9,
            root_legal=BASE, root_inputs=inputs, learner_actor=0, opponent_actor=1))
        worlds.append(tuple(SimpleNamespace(state=s, public=p) for s, p in (world(mode, owner, x) for x in seeds)))
    return requests, worlds


class OracleTests(unittest.TestCase):
    recipe = v5_gumbel.Recipe(7)

    def run_oracle(self, specs, *, chunk=8, **widths):
        ex = executor();requests, worlds = simple_requests(ex, specs)
        with patch.object(batch.v5_model, 'forward', side_effect=forward):
            oracle = reference.Oracle(ex, object(), chunk=chunk);stack = reference.stack(requests, worlds)
            roots, cells, children, leaves = reference.enumerate_specs(oracle, stack, requests, self.recipe, PARAMS, **widths)
            return roots, cells, children, leaves, oracle.leaves(stack, leaves, PARAMS)

    @staticmethod
    def find(table, key, cell):
        rows = [i for i, c in enumerate(table[key].tolist()) if tuple(c) == tuple(cell)]
        assert len(rows) == 1, (cell, rows);return rows[0]

    def test_turn2_chance_comes_from_the_child_not_the_root(self):
        seed = next(s for s in range(1000, 2000) if chance(s) != chance(rekey(s)) and chance(s + 1) != chance(rekey(s + 1)))
        roots, _, children, enumerated, leaves = self.run_oracle(((0, 0, (seed, seed + 1)),))
        self.assertEqual(roots[(0, 0)]['pre'], 0)
        row = self.find(children, 'cells', (0, 0, 1, 2))
        v1 = 200 + 1 * 30 + 2 * 20 + chance(seed)
        self.assertEqual(float(children['head_value'][row]), units(v1))
        row = self.find(leaves, 'specs', (0, 0, 1, 2, 2, 1))
        self.assertEqual(int(leaves['status'][row]), reference.STATUS_HEAD)
        self.assertEqual(float(leaves['head_value'][row]), units(v1 + 2 * 30 + 1 * 20 + chance(rekey(seed))))
        self.assertNotEqual(float(leaves['head_value'][row]), units(v1 + 2 * 30 + 1 * 20 + chance(seed)))   # not the root capsule
        # Every turn-1 child of a world re-keys identically: one shared turn-2 capsule (common random numbers).
        for a1 in range(3):
            row = self.find(leaves, 'specs', (0, 0, a1, 0, 0, 0))
            self.assertEqual(float(leaves['head_value'][row]), units(200 + a1 * 30 + chance(seed) + chance(rekey(seed))))
        # World 1 uses its own seed.
        row = self.find(leaves, 'specs', (0, 1, 0, 0, 0, 0))
        self.assertEqual(float(leaves['head_value'][row]), units(200 + chance(seed + 1) + chance(rekey(seed + 1))))

    def test_waiting_seat_is_dispatched_minus_one_and_enumerated_as_action_zero(self):
        seed = 1234
        _, cells, children, enumerated, leaves = self.run_oracle(((6, 1, (seed, seed)), (8, 1, (seed, seed)), (12, 0, (seed, seed))))
        # Root: the opponent waits, so turn 1 has column 0 only and the reply dispatched is -1.
        self.assertEqual({c[3] for c in cells if c[0] == 0}, {0})
        row = self.find(children, 'cells', (0, 0, 2, 0))
        self.assertEqual(float(children['head_value'][row]), units(200 + 2 * 30 - 20 + chance(seed)))
        self.assertFalse(children['peer_requested'][row])
        # Child: the opponent waits, the owner must switch; leaves are (4|5) x column 0, reply dispatched -1.
        row = self.find(children, 'cells', (1, 0, 0, 1))
        self.assertTrue(children['own_requested'][row]);self.assertFalse(children['peer_requested'][row])
        self.assertEqual(tuple(children['own_legal'][row]), SWITCH)
        self.assertEqual({l[4:] for l in enumerated if l[:4] == (1, 0, 0, 1)}, {(4, 0), (5, 0)})
        v1 = 200 + 1 * 20 + chance(seed)
        row = self.find(leaves, 'specs', (1, 0, 0, 1, 5, 0))
        self.assertEqual(float(leaves['head_value'][row]), units(v1 + 5 * 30 - 20 + chance(rekey(seed))))
        # Child: the owner waits; leaves are row 0 x (4|5), own action dispatched -1.
        self.assertEqual({l[4:] for l in enumerated if l[:4] == (2, 0, 1, 1)}, {(0, 4), (0, 5)})
        v1 = 200 + 30 + 20 + chance(seed)
        row = self.find(leaves, 'specs', (2, 0, 1, 1, 0, 4))
        self.assertEqual(float(leaves['head_value'][row]), units(v1 - 30 + 4 * 20 + chance(rekey(seed))))

    def test_terminal_values_and_error_codes_follow_d(self):
        modes = ((1, 0), (1, 1), (2, 0), (7, 1), (3, 0), (4, 1), (5, 0), (9, 0), (9, 1), (10, 1), (11, 0))
        _, _, children, _, leaves = self.run_oracle(tuple((mode, owner, (50 + i, 60 + i)) for i, (mode, owner) in enumerate(modes)))
        def child(q, cell=(0, 0)):
            row = self.find(children, 'cells', (q, 0) + cell)
            return int(children['status'][row]), int(children['error'][row]), float(children['value'][row])
        self.assertEqual(child(0), (1, 0, 1.));self.assertEqual(child(1), (1, 0, 1.))   # the owner's seat wins, either owner
        self.assertEqual(child(2), (1, 0, 0.))
        self.assertEqual(child(3)[:2], (3, 2))
        self.assertEqual(child(4, (1, 1))[:2], (3, 1));self.assertEqual(child(4, (1, 0))[:2], (2, 0))
        self.assertEqual(child(5)[:2], (3, 3));self.assertEqual(child(6)[:2], (3, 4))
        expanding = {s[0] for s in leaves['specs'].tolist()}
        self.assertFalse(expanding & {0, 1, 2, 3, 5, 6})   # terminal/error children expand nothing
        self.assertNotIn((4, 0, 1, 1), {tuple(s[:4]) for s in leaves['specs'].tolist()});self.assertIn(4, expanding)
        def leaf(q, a2, r2=0):
            row = self.find(leaves, 'specs', (q, 0, 0, 0, a2, r2))
            return int(leaves['status'][row]), int(leaves['error'][row]), float(leaves['value'][row])
        self.assertEqual(leaf(7, 0), (1, 0, -1.));self.assertEqual(leaf(8, 0), (1, 0, -1.))   # the opponent's seat wins
        self.assertEqual(leaf(7, 1), (1, 0, 0.));self.assertEqual(leaf(7, 2)[:2], (4, 0))
        self.assertEqual(leaf(9, 2)[:2], (3, 1));self.assertEqual(leaf(9, 1)[:2], (4, 0))
        self.assertEqual(leaf(10, 1)[:2], (3, 3))

    def test_chunk_size_does_not_change_any_result(self):
        specs = tuple((mode, i % 2, (70 + i, 90 + i)) for i, mode in enumerate((0, 3, 8, 9, 11, 12)))
        results = [self.run_oracle(specs, chunk=chunk) for chunk in (1, 3, 8, 512)]
        first = results[0]
        for other in results[1:]:
            self.assertEqual(first[1], other[1]);self.assertEqual(first[3], other[3])
            for a, b in ((first[2], other[2]), (first[4], other[4])):
                self.assertEqual(set(a), set(b))
                for key in a:np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        self.assertGreater(len(first[3]), 100)

    def test_top_k_ranks_logits_then_lower_index(self):
        logits = np.asarray([0, 1, 1, .5, 9, 9, np.nan, 0, 0], np.float32);legal = (True,) * 4 + (False, False, True, False, False)
        self.assertEqual(reference.top_k(legal, logits, 1), (1,))
        self.assertEqual(reference.top_k(legal, logits, 2), (1, 2))
        self.assertEqual(reference.top_k(legal, logits, 3), (1, 2, 3))
        self.assertEqual(reference.top_k(legal, logits, 4), (0, 1, 2, 3))
        self.assertEqual(reference.top_k(legal, logits, 9), (0, 1, 2, 3, 6))   # illegal lanes never, non-finite last
        tied = np.zeros(9, np.float32)
        self.assertEqual(reference.top_k(BASE, tied, 2), (0, 1))
        for k in (0, 10, 1.):
            with self.assertRaises(ValueError):reference.top_k(BASE, tied, k)

    def test_turn2_width_uses_the_oracles_child_priors(self):
        # Learner child prior [0,1,1,...] ties lanes 1 and 2; opponent prior -arange/9 prefers lane 0.
        _, _, children, full, _ = self.run_oracle(((0, 0, (5, 6)),))
        _, _, _, narrow, _ = self.run_oracle(((0, 0, (5, 6)),), k_own=1, k_opp=1)
        _, _, _, two, _ = self.run_oracle(((0, 0, (5, 6)),), k_own=2, k_opp=2)
        expanded = int(np.count_nonzero(children['status'] == 2))
        self.assertEqual(len(full), expanded * 9);self.assertEqual(len(narrow), expanded);self.assertEqual(len(two), expanded * 4)
        self.assertEqual({l[4:] for l in narrow}, {(1, 0)});self.assertEqual({l[4:] for l in two}, {(1, 0), (1, 1), (2, 0), (2, 1)})
        synthetic = dict(cells=np.asarray([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]), status=np.asarray([2, 2, 1]),
            own_requested=np.asarray([True, False, True]), peer_requested=np.asarray([True, True, True]),
            own_legal=np.asarray([BASE] * 3), peer_legal=np.asarray([BASE] * 3),
            own_logits=np.asarray([[3, 3, 3] + [0] * 6] * 3, np.float32), peer_logits=np.asarray([[0, 2, 2] + [0] * 6] * 3, np.float32))
        self.assertEqual(reference.turn2_leaves(synthetic, k_own=2, k_opp=1), [(0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 1, 1), (0, 0, 1, 0, 0, 1)])

    def test_turn1_width_prefixes_pricing_order_and_ranks_root_replies(self):
        roots, cells, *_ = self.run_oracle(((0, 0, (5, 6)),), k1_own=2, k1_opp=1)
        request = simple_requests(executor(), ((0, 0, (5, 6)),))[0][0]
        order = self.recipe.pricing_order(request.plans.root.address, request.root_logits, request.root_legal)
        self.assertEqual(cells, [(0, w, a, 0) for w in (0, 1) for a in order[:2]])

    def test_pre_refusal_codes_follow_d(self):
        ex = executor();requests, worlds = simple_requests(ex, ((0, 0, (1, 2)), (0, 1, (3, 4)), (0, 0, (5, 6))))
        requests[0].root_inputs = {'numeric_features': requests[0].root_inputs['numeric_features'] + 1}
        requests[1].root_legal = SWITCH
        worlds[2] = (worlds[2][0], SimpleNamespace(state=None, public=None))
        with patch.object(batch.v5_model, 'forward', side_effect=forward):
            oracle = reference.Oracle(ex, object(), chunk=4);stack = reference.stack(requests, worlds)
            roots, cells, *_ = reference.enumerate_specs(oracle, stack, requests, self.recipe, PARAMS)
        self.assertEqual({k: v['pre'] for k, v in roots.items()}, {(0, 0): 3, (0, 1): 3, (1, 0): 2, (1, 1): 2, (2, 0): 0})
        self.assertEqual({c[:2] for c in cells}, {(2, 0)})
        self.assertEqual(reference.world_pre(roots, 3).tolist(), [[3, 3], [2, 2], [0, reference.NO_STATE]])

    def test_opponent_child_view_codes_5_and_6_after_ds_codes(self):
        # 13: opponent view encoder error -> 5; 14: requested opponent prior non-finite -> 6; 15: same prior, but the
        # opponent waits -> expanded; 4: both views error -> D's 3 first; 5: both heads non-finite -> D's 4 first;
        # 16: the opponent's view has both an encoder error and a non-finite prior -> 5 first.
        modes = ((13, 0), (14, 1), (15, 0), (4, 1), (5, 0), (16, 1))
        _, _, children, enumerated, leaves = self.run_oracle(tuple((mode, owner, (30 + i, 40 + i)) for i, (mode, owner) in enumerate(modes)))
        by_request = lambda q, key: {int(x) for x, c in zip(children[key], children['cells']) if c[0] == q}
        self.assertEqual((by_request(0, 'status'), by_request(0, 'error')), ({3}, {5}))
        self.assertEqual((by_request(1, 'status'), by_request(1, 'error')), ({3}, {6}))
        self.assertEqual((by_request(2, 'status'), by_request(2, 'error')), ({2}, {0}))
        self.assertEqual((by_request(3, 'status'), by_request(3, 'error')), ({3}, {3}))
        self.assertEqual((by_request(4, 'status'), by_request(4, 'error')), ({3}, {4}))
        self.assertEqual((by_request(5, 'status'), by_request(5, 'error')), ({3}, {5}))
        for q in (0, 1, 5):   # codes 5/6 keep D's learner value: the learner head at the child
            rows = [i for i, c in enumerate(children['cells']) if c[0] == q]
            np.testing.assert_array_equal(children['value'][rows], children['head_value'][rows])
            self.assertTrue(np.isfinite(children['value'][rows]).all())
        rows = [i for i, c in enumerate(children['cells']) if c[0] in (3, 4)]
        self.assertTrue(np.isnan(children['value'][rows]).all())
        self.assertEqual({l[0] for l in enumerated}, {2});self.assertEqual({l[5] for l in enumerated}, {0})   # 5/6 never expand


class OraclePricer:
    """Writes the oracle's own surface in the contract's layout; a correct gate must pass it."""
    def __init__(self, ex, widths):
        self.oracle = reference.Oracle(ex, object(), chunk=16);self.widths = widths

    def price(self, requests, materializer, recipe, params):
        stack = reference.stack(requests, batch._numpy_worlds(requests, materializer, recipe))
        roots, _, children, leaves = reference.enumerate_specs(self.oracle, stack, requests, recipe, params, **self.widths)
        return frame_from(len(requests), dict(roots=roots, children=children, leaves=self.oracle.leaves(stack, leaves, params)))


def frame_from(count, data):
    R = count;f = dict(pre=np.full((R, 2), reference.NO_STATE, np.int32), root_peer_requested=np.zeros((R, 2), bool),
        root_peer_legal=np.zeros((R, 2, 9), bool), root_peer_logits=np.zeros((R, 2, 9), np.float32),
        t1_status=np.zeros((R, 2, 9, 9), np.int8), t1_value=np.full((R, 2, 9, 9), np.nan, np.float32), t1_error=np.zeros((R, 2, 9, 9), np.int8),
        t2_status=np.zeros((R, 2, 9, 9, 9, 9), np.int8), t2_value=np.full((R, 2, 9, 9, 9, 9), np.nan, np.float32),
        t2_error=np.zeros((R, 2, 9, 9, 9, 9), np.int8), t2_digest=np.zeros((R, 2, 9, 9, 9, 9, 2), np.uint32),
        t1_digest_own=np.zeros((R, 2, 9, 9, 2), np.uint32), t1_digest_peer=np.zeros((R, 2, 9, 9, 2), np.uint32), timing=dict(fake=True))
    for seat in ('own', 'peer'):
        f[f't1_{seat}_requested'] = np.zeros((R, 2, 9, 9), bool);f[f't1_{seat}_legal'] = np.zeros((R, 2, 9, 9, 9), bool)
        f[f't1_{seat}_logits'] = np.zeros((R, 2, 9, 9, 9), np.float32)
    for (q, wi), root in data['roots'].items():
        f['pre'][q, wi] = root['pre']
        if root['pre'] == 0:
            f['root_peer_requested'][q, wi] = root['peer_requested'];f['root_peer_legal'][q, wi] = root['peer_legal']
            f['root_peer_logits'][q, wi] = root['peer_logits']
    children = data['children']
    for i, cell in enumerate(children['cells'].tolist()):
        at = tuple(cell)
        for name in ('status', 'error', 'value'):f['t1_' + name][at] = children[name][i]
        for seat in ('own', 'peer'):
            for name in ('requested', 'legal', 'logits'):f[f't1_{seat}_{name}'][at] = children[f'{seat}_{name}'][i]
            f['t1_digest_' + seat][at] = children[seat + '_digest'][i]
    leaves = data['leaves']
    for i, spec in enumerate(leaves['specs'].tolist()):
        at = tuple(spec)
        for name in ('status', 'error', 'value', 'digest'):f['t2_' + name][at] = leaves[name][i]
    return f


class GateTests(unittest.TestCase):
    WIDTHS = dict(k_own=9, k_opp=9, k1_own=9, k1_opp=9)
    MODES = ((0, 0), (9, 1), (8, 0), (3, 1), (4, 0), (11, 1), (12, 0), (0, 1), (1, 0), (13, 1), (14, 0), (15, 1))
    MISSING = {(6, 1)}
    PRE_REFUSED = 7

    @classmethod
    def setUpClass(cls):
        _, cls.root, _, cls.tables, cls.recipe = facts();cls.ex = executor();cls.requests = cls.make_requests()

    @classmethod
    def make_requests(cls):
        requests = []
        for i, (mode, owner) in enumerate(cls.MODES):
            stream = 100 + i
            while not cls.recipe.selected(v5_contract.DecisionAddress(stream, 4 + i, owner)):stream += 100
            root = replace(cls.root, address=v5_contract.DecisionAddress(stream, 4 + i, owner))
            starts = plans.plan_starts(root, cls.tables.source, cls.recipe)
            state, public = world(mode, owner, starts.starts[0].seed, float(stream % 79))
            inputs = cls.ex.observe(state, public, j.asarray([owner]), cls.ex.legal(state)[:, owner]).inputs
            if i == cls.PRE_REFUSED:inputs = {'numeric_features': inputs['numeric_features'] + 1}   # D pre-refuses this root
            requests.append(batch.Request(starts, (0.,) * 9, BASE, inputs, i % 2, 1 - i % 2))
        return tuple(requests)

    def materialize(self, root, start, tables, recipe, *, keys=None):
        index = root.address.frame - 4;mode, owner = self.MODES[index]
        if (index, start.world) in self.MISSING:return mapping.World(start.world, None, None, ('inert missing',), start.seed, False)
        state, public = world(mode, owner, start.seed, float(root.address.stream % 79))
        return mapping.World(start.world, state, public, (), start.seed, False)

    def price(self, *, sample=None, widths=None, pricer=None):
        widths = widths or self.WIDTHS
        frame_pricer = batch.FramePricer(self.ex, object(), capacity=16, block_rows=16, actor_capacity=3, vectorized=True, host_prep=True)
        with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=forward):
            return gate.price_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS,
                pricer=pricer or OraclePricer(self.ex, widths), frame_pricer=frame_pricer, oracle=reference.Oracle(self.ex, object(), chunk=32),
                widths=widths, sample=sample, block=0)

    def priced(self):
        if not hasattr(type(self), '_priced'):type(self)._priced = self.price()
        return type(self)._priced

    def evaluate(self, frame, *, data=None, full_turn1=True):
        _, d_results, clean, _ = self.priced()
        return gate.summarize([gate.evaluate(frame, d_results, self.requests, data or clean, context=dict(block=0, call=0), full_turn1=full_turn1)])

    def mutated(self, change):
        frame = copy.deepcopy(self.priced()[0]);change(frame, self.priced()[2]);return frame

    def test_the_oracles_own_surface_passes_every_check(self):
        frame, d_results, data, _ = self.priced()
        result = self.evaluate(frame);checks = result['checks']
        self.assertTrue(result['passed'], {k: v.get('first_mismatches') for k, v in checks.items()})
        self.assertEqual(checks['c']['quantities']['t2_value']['exact_equal_fraction'], 1.)
        statuses = collections.Counter(r.status for r in d_results)
        self.assertEqual(statuses, {'LABELED': 8, 'REFUSED': 4})   # D refuses invalid/encoder/missing/pre roots
        for name in ('t1_status', 't2_status', 't2_digest', 't1_digest_peer', 't1_value_terminal', 't2_value_terminal'):
            self.assertGreater(checks['a']['compared'][name], 0, name)
        for name in ('t1_value_vs_d', 't1_status_vs_d', 't1_depth2_code_status', 't1_error_vs_d', 'root_peer_logits_vs_d', 'pre_vs_d',
                'd_refused_world_unenumerated', 't1_enumeration_vs_d'):
            self.assertGreater(checks['b']['compared'][name], 0, name)
        self.assertEqual(set(np.unique(data['leaves']['status'])), {1, 3, 4})
        self.assertEqual(set(np.unique(data['children']['status'])), {1, 2, 3})

    def test_each_kind_of_mismatch_is_caught(self):
        frame, _, data, _ = self.priced();leaves, children = data['leaves'], data['children']
        head = tuple(leaves['specs'][list(leaves['status']).index(4)]);terminal = tuple(leaves['specs'][list(leaves['status']).index(1)])
        expanded = tuple(children['cells'][list(children['status']).index(2)]);error = tuple(children['cells'][list(children['status']).index(3)])
        unexpanded = expanded + (8, 8)
        other_error = 4 if children['error'][list(children['status']).index(3)] != 4 else 1
        code5 = tuple(children['cells'][list(children['error']).index(5)]);code6 = tuple(children['cells'][list(children['error']).index(6)])
        def setter(name, at, value):
            def change(f, _):f[name][at] = value
            return change
        def bump(name, at, delta):
            def change(f, _):f[name][at] = np.float32(f[name][at] + delta) if delta else np.nextafter(f[name][at], np.float32(2))
            return change
        def flip(name, at):
            def change(f, _):f[name][at] = ~f[name][at] if f[name].dtype == bool else f[name][at] ^ 1
            return change
        cases = (
            ('t2_status', setter('t2_status', head, 1), 'a'), ('t2_error', setter('t2_error', head, 3), 'a'),
            ('t2_value_terminal', setter('t2_value', terminal, -frame['t2_value'][terminal] + .5), 'a'),
            ('t2_digest', flip('t2_digest', head + (0,)), 'a'), ('t1_digest_own', flip('t1_digest_own', expanded + (1,)), 'a'),
            ('t1_digest_peer', flip('t1_digest_peer', expanded + (0,)), 'a'), ('t1_own_legal', flip('t1_own_legal', expanded + (7,)), 'a'),
            ('t1_peer_requested', flip('t1_peer_requested', expanded), 'a'), ('t1_status', setter('t1_status', expanded, 1), 'a'),
            ('t1_error', setter('t1_error', error, other_error), 'a'), ('pre', setter('pre', (0, 0), 2), 'a'),
            ('root_peer_legal', flip('root_peer_legal', (0, 0, 1)), 'a'),
            ('t1_error_vs_d', setter('t1_error', error, other_error), 'b'), ('pre_vs_d', setter('pre', (0, 0), 2), 'b'),
            ('t1_value_vs_d', bump('t1_value', expanded, 0), 'b'), ('root_peer_logits_vs_d', bump('root_peer_logits', (0, 0, 3), 0), 'b'),
            ('t1_own_logits', bump('t1_own_logits', expanded + (1,), 1e-3), 'b'),
            ('t2_value', bump('t2_value', head, 1e-3), 'c'),
            ('t2_extra_in_pricer', setter('t2_status', unexpanded, 4), 'd'), ('t2_missing_from_pricer', setter('t2_status', head, 0), 'd'),
            ('t1_extra_in_pricer', setter('t1_status', (0, 0, 8, 8), 2), 'd'), ('t1_enumeration_vs_d', setter('t1_status', (0, 0, 8, 8), 2), 'b'),
            ('t1_extra_in_pricer', setter('t1_status', (6, 1, 0, 0), 2), 'd'), ('d_refused_world_unenumerated', setter('t1_status', (6, 1, 0, 0), 2), 'b'),
            ('t1_error', setter('t1_error', code5, 6), 'a'), ('t1_error', setter('t1_error', code6, 0), 'a'),
            ('t1_depth2_code_status', setter('t1_status', code5, 2), 'b'), ('t1_status', setter('t1_status', code6, 2), 'a'),
            ('t1_value_vs_d', bump('t1_value', code6, 0), 'b'), ('t1_value_vs_d', bump('t1_value', code5, 0), 'b'),
            ('t1_digest_peer', flip('t1_digest_peer', code5 + (1,)), 'a'),
            ('pre', setter('pre', (6, 1), 0), 'a'), ('pre_vs_d', setter('pre', (6, 1), 0), 'b'),
        )
        for name, change, check in cases:
            with self.subTest(name):
                result = self.evaluate(self.mutated(change));found = result['checks'][check]
                self.assertFalse(result['passed']);self.assertFalse(found['passed'])
                if check == 'c':self.assertFalse(found['quantities'][name]['within_tolerance'])
                elif name == 't1_own_logits':self.assertFalse(found['t1_own_logits']['within_tolerance'])
                else:
                    self.assertGreater(found['mismatched'][name], 0, found['mismatched'])
                    self.assertIn(name, {m['field'] for m in found['first_mismatches']})
                    self.assertTrue(all(m['block'] == 0 for m in found['first_mismatches']))

    def test_head_differences_within_tolerance_pass_and_are_reported(self):
        _, _, data, _ = self.priced();leaves, children = data['leaves'], data['children']
        heads = [tuple(s) for s, x in zip(leaves['specs'], leaves['status']) if x == 4]
        expanded = tuple(children['cells'][list(children['status']).index(2)])
        def change(f, _):
            for at in heads[:3]:f['t2_value'][at] = np.nextafter(f['t2_value'][at], np.float32(2))
            f['t1_own_logits'][expanded + (1,)] = np.nextafter(f['t1_own_logits'][expanded + (1,)], np.float32(2))
        result = self.evaluate(self.mutated(change));c = result['checks']['c']['quantities']
        self.assertTrue(result['passed'])
        self.assertEqual(c['t2_value']['compared'] - c['t2_value']['exact_equal'], 3)
        self.assertGreater(c['t2_value']['max_abs_difference'], 0);self.assertLessEqual(c['t2_value']['max_abs_difference'], gate.HEAD_TOLERANCE)
        self.assertLess(result['checks']['b']['t1_own_logits']['exact_equal_fraction'], 1)

    def test_missing_field_or_wrong_geometry_fails_closed(self):
        frame = dict(self.priced()[0]);frame.pop('t2_digest')
        result = self.evaluate(frame)
        self.assertFalse(result['passed']);self.assertIn('t2_digest:missing', result['checks']['a']['first_mismatches'][0]['problems'])
        frame = dict(self.priced()[0]);frame['t1_status'] = frame['t1_status'][:, :1]
        self.assertFalse(self.evaluate(frame)['passed'])

    def test_first_mismatches_are_capped(self):
        def change(f, data):
            for spec in data['leaves']['specs'].tolist():f['t2_digest'][tuple(spec)] ^= 1
        a = self.evaluate(self.mutated(change))['checks']['a']
        self.assertGreater(a['mismatched']['t2_digest'], gate.FIRST);self.assertEqual(len(a['first_mismatches']), gate.FIRST)

    def test_sampled_leaves_are_fixed_per_world_and_enumeration_still_covers_all(self):
        frame, d_results, data, _ = self.price(sample=2)
        counts = {}
        for spec in data['leaves']['specs'].tolist():counts[tuple(spec[:2])] = counts.get(tuple(spec[:2]), 0) + 1
        self.assertTrue(counts);self.assertLessEqual(max(counts.values()), 2)
        self.assertEqual(data['leaves']['specs'].tolist(), self.price(sample=2)[2]['leaves']['specs'].tolist())
        self.assertGreater(len(data['enumerated']), len(data['leaves']['specs']))
        result = gate.summarize([gate.evaluate(frame, d_results, self.requests, data)])
        self.assertTrue(result['passed']);self.assertEqual(result['checks']['d']['compared']['t2_enumerated'], len(data['enumerated']))

    def test_top_k_widths_end_to_end(self):
        widths = dict(k_own=1, k_opp=2, k1_own=2, k1_opp=1)
        frame, d_results, data, _ = self.price(widths=widths)
        self.assertTrue(gate.summarize([gate.evaluate(frame, d_results, self.requests, data, full_turn1=False)])['passed'])
        full = self.price(pricer=OraclePricer(self.ex, self.WIDTHS), widths=widths)
        result = gate.summarize([gate.evaluate(full[0], full[1], self.requests, full[2], full_turn1=False)])
        self.assertFalse(result['passed']);self.assertGreater(result['checks']['d']['mismatched']['t2_extra_in_pricer'], 0)

    def test_world_block_seeds_extend_the_declared_recipe(self):
        seeds = gate.world_block_seeds(1779000000, 4)
        self.assertEqual(seeds[0], 1779000000);self.assertEqual(len(set(seeds)), 4);self.assertEqual(seeds[:2], gate.world_block_seeds(1779000000, 2))

    def test_run_call_summarizes_one_call(self):
        frame_pricer = batch.FramePricer(self.ex, object(), capacity=16, block_rows=16, actor_capacity=3, vectorized=True, host_prep=True)
        with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=forward):
            evaluation, summary = gate.run_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS,
                pricer=OraclePricer(self.ex, self.WIDTHS), frame_pricer=frame_pricer, oracle=reference.Oracle(self.ex, object(), chunk=32),
                widths=self.WIDTHS, sample=None, block=0, context=dict(block=0, call=0))
        self.assertTrue(all(summary['passed'].values()), summary)
        self.assertEqual((summary['roots'], summary['worlds']), (len(self.MODES), 2 * len(self.MODES) - len(self.MISSING)))
        self.assertEqual(summary['pricer_timing'], dict(fake=True))
        self.assertEqual(summary['refusals']['depth2']['refused_count'], 7)

    def test_refusal_shares_and_error_counts_by_code(self):
        frame, d_results, data, _ = self.priced()
        report = gate.refusal_report(frame, d_results, data)
        # Depth 2 refuses invalid/encoder (3, 4), a turn-2 encoder error (5), the missing and pre worlds (6, 7)
        # and codes 5/6 (9, 10); depth 1 only 3, 4, 6, 7. Mode 15's unusable prior is not requested: no refusal.
        self.assertEqual((report['depth2']['root_count'], report['depth2']['refused_count']), (12, 7))
        self.assertEqual(report['depth2']['refused_roots'], [3, 4, 5, 6, 7, 9, 10])
        self.assertEqual((report['depth1']['root_count'], report['depth1']['refused_count'], report['depth1']['refused_roots']), (12, 4, [3, 4, 6, 7]))
        self.assertEqual(report['turn1_errors_by_code'], {'1': 2, '2': 0, '3': 18, '4': 0, '5': 18, '6': 18})
        self.assertEqual(report['turn2_errors_by_code'], {'1': 0, '2': 0, '3': 162, '4': 0})
        self.assertEqual(report['oracle_turn1_errors_by_code'], report['turn1_errors_by_code'])
        self.assertEqual(report['oracle_turn2_errors_by_code_checked_leaves'], report['turn2_errors_by_code'])
        self.assertTrue(report['refusal_summary'].startswith('not importable'))
        merged = gate.merge_refusals([report, report])
        self.assertEqual((merged['depth2']['root_count'], merged['depth2']['refused_count']), (24, 14))
        self.assertEqual((merged['depth2']['refused_share'], merged['depth1']['refused_share']), (7 / 12, 4 / 12))
        self.assertEqual(merged['turn1_errors_by_code']['5'], 36);self.assertNotIn('refused_roots', merged['depth2'])

    def test_refusal_summary_is_preferred_when_importable(self):
        frame, d_results, data, _ = self.priced();seen = []
        def refusal_summary(view):
            seen.append(view.pre.shape)   # the contract's attribute-style frame
            return dict(roots=tuple(dict(root=r, refused=r in (3, 4, 5, 6, 7, 9, 10)) for r in range(12)), root_count=12, refused_count=7,
                depth1_refused_count=5, depth2_only_refused_count=2, refused_by_cause=dict(turn1=2, turn1_depth2_only=2),
                turn1_errors_by_code={5: 18}, turn2_errors_by_code={3: 162})
        with patch.dict(sys.modules, {'gpu_engine.depth2_backups': SimpleNamespace(refusal_summary=refusal_summary)}):
            report = gate.refusal_report(frame, d_results, data)
        self.assertEqual(seen, [(12, 2)]);self.assertEqual(report['depth2']['source'], 'depth2_backups.refusal_summary')
        self.assertEqual(report['depth2']['turn1_errors_by_code'], {'5': 18});self.assertTrue(report['depth2_sources_agree'])
        self.assertEqual(report['depth2']['refused_roots'], [3, 4, 5, 6, 7, 9, 10])
        self.assertEqual(gate.merge_refusals([report])['depth2']['refused_share'], 7 / 12)
        with patch.dict(sys.modules, {'gpu_engine.depth2_backups': SimpleNamespace(refusal_summary=lambda view: dict(refusal_summary(view), refused_count=6))}):
            self.assertFalse(gate.refusal_report(frame, d_results, data)['depth2_sources_agree'])


class PricerBindingTests(unittest.TestCase):
    """The pricer's constructor is not on this branch: bind it by signature, and fail closed."""
    WIDTHS = dict(k_own=9, k_opp=9, k1_own=9, k1_opp=9)

    def make(self, cls, **widths):
        module = SimpleNamespace(Depth2Pricer=cls)
        with patch.dict(sys.modules, {'gpu_engine.device_depth2_pricer': module}):
            return gate.make_pricer('executor', 'model', widths=dict(self.WIDTHS, **widths), actor_capacity=33)

    def test_options_follow_the_constructor_signature(self):
        class Strict:
            def __init__(self, executor, model, *, digest=False, k_own=9, k_opp=9):self.options = dict(digest=digest, k_own=k_own, k_opp=k_opp)
        pricer, options = self.make(Strict, k_own=3)
        self.assertEqual(options, dict(digest=True, k_own=3, k_opp=9));self.assertEqual(pricer.options, options)
        class Loose:
            def __init__(self, executor, model, **options):self.options = options
        pricer, options = self.make(Loose, k1_opp=2)
        self.assertEqual(pricer.options, dict(digest=True, actor_capacity=33, k_own=9, k_opp=9, k1_own=9, k1_opp=2))

    def test_missing_digest_or_unsupported_width_refuses(self):
        class NoDigest:
            def __init__(self, executor, model, *, k_own=9, k_opp=9):pass
        class FullTurn1:
            def __init__(self, executor, model, *, digest=False, k_own=9, k_opp=9):pass
        with self.assertRaises(SystemExit):self.make(NoDigest)
        with self.assertRaises(SystemExit):self.make(FullTurn1, k1_own=2)

    def test_cli_validates_arguments_and_fails_before_loading_without_the_pricer(self):
        out = Path(tempfile.gettempdir()) / f'depth2-gate-never-created-{uuid.uuid4().hex}'
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (['--out', str(out)], ['--out', str(out), '--saved-roots', 'a', '--d3-root', 'b', '--chunk', '0'],
                    ['--out', str(out), '--saved-roots', 'a'], ['--out', str(out), '--d3-root', 'b'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--k-own', '10']):
                with self.assertRaises(SystemExit) as raised:gate.main(argv)
                self.assertEqual(raised.exception.code, 2, argv)
        with patch.object(importlib.util, 'find_spec', return_value=None), self.assertRaises(SystemExit) as raised:
            gate.main(['--out', str(out), '--d3-root', 'b', '--chunk', '0'])
        self.assertIn('device_depth2_pricer', str(raised.exception.code));self.assertFalse(out.exists())


if __name__ == '__main__':unittest.main()
