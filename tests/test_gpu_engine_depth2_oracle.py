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
import dataclasses
import copy
import importlib.util
import io
import json
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
    16 both an encoder error and a non-finite prior, 19 an encoder error while the opponent waits;
    18 re-keys chance from the dispatched actions, so children of one world draw different turn-2 chance."""
    def legal(state):
        owner, mode, turn = state.owner[0], state.mode[0], state.turn[0]
        child = turn == 1
        own = j.where((mode == 12) & child, False, j.where((mode == 8) & child, j.asarray(SWITCH), j.asarray(BASE)))
        foe = j.where((mode == 6) | (((mode == 8) | (mode == 15) | (mode == 19)) & child), False, j.where((mode == 12) & child, j.asarray(SWITCH), j.asarray(BASE)))
        rows = j.where(owner == 0, j.stack([own, foe]), j.stack([foe, own]))
        return j.where(state.done[0], False, rows)[None]

    def observe(state, public, owner, mask):
        if public.clock.ndim != 0:raise ValueError('one-world namespace was flattened')
        mode, child, peer = state.mode, public.turn == 1, owner != state.owner
        error = j.where(((mode == 4) & child) | ((mode == 11) & (public.turn == 2)) | (((mode == 13) | (mode == 16) | (mode == 19)) & child & peer), 1, 0)
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
        seed = state.seed * j.uint32(LCG[0]) + j.uint32(LCG[1])   # re-keyed every step, whatever the actions (mode 18 aside)
        seed = seed + j.where(mode == 18, (a + 1) * 7 + (r + 1) * 13, 0).astype(j.uint32)
        after = state._replace(value=value, done=state.done | child_end | leaf_end, winner=winner, seed=seed, turn=turn, own_act=a, peer_act=r)
        return After(Step(after, invalid), Public(public.clock, public.turn + 1))

    ex = serial.Executor(legal, observe, lambda obs, state, seat, action: action, lambda state: (state.seed % 17).astype(j.int32), advance)
    return replace(ex, joint=serial._make_joint(ex))


ENGINE_OF_LANE = (0, 1, 2, 3, 7, 8, 4, 5, 6)   # encoder request lane -> engine lane (switch lanes reordered)


def permuted_executor():
    """The encoder's switch lanes are a permutation of the engine's (as once a side has switched): observe
    reorders the engine mask into request lanes, dispatch maps a lane back to its engine lane, and advance
    refuses (invalid) any requested seat's engine action that the engine mask does not allow. Every requested
    seat also has two live bench slots, engine lanes 4 and 5 = encoder lanes 6 and 7."""
    base = executor();order = j.asarray(ENGINE_OF_LANE);bench = j.asarray((False,) * 4 + (True, True) + (False,) * 3)
    def legal(state):
        mask = base.legal(state)
        return mask | (j.any(mask, axis=-1, keepdims=True) & bench)
    def advance(state, public, actions, chance):
        mask = legal(state)[0];acts = actions[0]
        wrong = j.any(j.any(mask, axis=-1) & ((acts < 0) | ~mask[j.arange(2), j.clip(acts, 0, 8)]))
        after = base.advance(state, public, actions, chance)
        return after._replace(step=after.step._replace(invalid_actions=after.step.invalid_actions | wrong))
    ex = serial.Executor(legal, lambda state, public, owner, mask: base.observe(state, public, owner, mask[:, order]),
        lambda observation, state, seat, action: order[action], base.sample, advance)
    return replace(ex, joint=serial._make_joint(ex))


def world(mode, owner, seed, clock=3.):
    state = State(j.asarray([200], j.int32), j.asarray([False]), j.asarray([0], j.int32), j.asarray([mode], j.int32),
        j.asarray([seed % 2**32], j.uint32), j.asarray([owner], j.int32), j.asarray([0], j.int32), j.zeros(1, j.int32), j.zeros(1, j.int32))
    return state, Public(j.asarray(clock, j.float32), j.asarray([0], j.int32))


def chance(seed):
    return seed % 17


def rekey(seed, actions=None):
    extra = 0 if actions is None else (actions[0] + 1) * 7 + (actions[1] + 1) * 13
    return (seed * LCG[0] + LCG[1] + extra) % 2**32


def simple_requests(ex, specs):
    """Duck-typed requests and world pairs: specs = ((mode, owner, (seed0, seed1)), ...)."""
    requests, worlds = [], []
    for i, (mode, owner, seeds) in enumerate(specs):
        state, public = world(mode, owner, seeds[0])
        view = ex.observe(state, public, j.asarray([owner]), ex.legal(state)[:, owner])
        address = v5_contract.DecisionAddress(100 + i, 4, owner)
        requests.append(SimpleNamespace(plans=SimpleNamespace(root=SimpleNamespace(address=address)), root_logits=(0.,) * 9,
            root_legal=tuple(map(bool, np.asarray(view.observation.legal_action_mask)[0])), root_inputs=view.inputs,
            learner_actor=0, opponent_actor=1))
        worlds.append(tuple(SimpleNamespace(state=s, public=p) for s, p in (world(mode, owner, x) for x in seeds)))
    return requests, worlds


class OracleTests(unittest.TestCase):
    recipe = v5_gumbel.Recipe(7)

    def run_oracle(self, specs, *, chunk=8, make=executor, **widths):
        ex = make();requests, worlds = simple_requests(ex, specs)
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

    def test_each_child_draws_its_own_turn2_chance(self):
        seed = 4321;_, _, children, enumerated, leaves = self.run_oracle(((18, 0, (seed, seed + 7)),))
        drawn = {chance(rekey(seed, c[2:])) for c in children['cells'].tolist() if c[1] == 0}
        self.assertGreater(len(drawn), 3)   # the children of one world really draw different chance here
        for row, (q, w, a1, r1, a2, r2) in enumerate(leaves['specs'].tolist()):
            base = seed + 7 * w
            expected = 200 + a1 * 30 + r1 * 20 + chance(base) + a2 * 30 + r2 * 20 + chance(rekey(base, (a1, r1)))
            self.assertEqual(float(leaves['head_value'][row]), units(expected), (w, a1, r1, a2, r2))
        self.assertEqual(len(leaves['specs']), 2 * 9 * 9)

    def test_child_masks_are_encoder_lanes_and_dispatch_inverts_them(self):
        encoder = (True, True, True, False, False, False, True, True, False)   # engine lanes 0,1,2,4,5
        roots, cells, children, enumerated, leaves = self.run_oracle(((0, 0, (5, 6)), (8, 1, (7, 8)), (12, 0, (9, 10))), make=permuted_executor)
        ex = permuted_executor();state, _ = world(0, 0, 5)
        self.assertNotEqual(tuple(map(bool, np.asarray(ex.legal(state))[0, 0])), encoder)   # engine and encoder masks differ
        self.assertEqual(roots[(0, 0)]['peer_legal'], encoder)
        self.assertEqual({c[2] for c in cells}, {0, 1, 2, 6, 7});self.assertEqual({c[3] for c in cells}, {0, 1, 2, 6, 7})
        for i, (q, w, a1, r1) in enumerate(children['cells'].tolist()):
            own, peer = tuple(map(bool, children['own_legal'][i])), tuple(map(bool, children['peer_legal'][i]))
            switch = tuple(i in (6, 7) for i in range(9))
            self.assertEqual((own, peer), {0: (encoder, encoder), 1: (switch, (False,) * 9), 2: ((False,) * 9, switch)}[q], (q, a1, r1))
        self.assertTrue((children['status'] == 2).all(), set(children['error'].tolist()))
        self.assertTrue((leaves['status'] == 4).all(), set(leaves['error'].tolist()))   # no invalid action anywhere
        self.assertEqual({l[4:] for l in enumerated if l[0] == 1}, {(6, 0), (7, 0)});self.assertEqual({l[4:] for l in enumerated if l[0] == 2}, {(0, 6), (0, 7)})
        self.assertEqual({l[4] for l in enumerated if l[0] == 0}, {0, 1, 2, 6, 7})
        # The leaf really stepped the engine lane the encoder lane maps to (forced switch to lane 7 = engine slot 5).
        row = self.find(leaves, 'specs', (1, 0, 0, 1, 7, 0))
        self.assertEqual(float(leaves['head_value'][row]), units(200 + 20 + chance(7) + 5 * 30 - 20 + chance(rekey(7))))

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

    def test_stacked_worlds_are_bucketed_and_chunks_are_timed(self):
        ex = executor();requests, worlds = simple_requests(ex, tuple((0, 0, (i, i + 50)) for i in range(5)))
        stack = reference.stack(requests, worlds)
        self.assertEqual(len(stack.rows), 10);self.assertEqual(stack.states.value.shape[0], 16);self.assertEqual(len(stack.owner), 10)
        self.assertEqual([reference.bucket(n) for n in (1, 8, 9, 100)], [8, 8, 16, 128])
        with patch.object(batch.v5_model, 'forward', side_effect=forward):
            oracle = reference.Oracle(ex, object(), chunk=64);reference.enumerate_specs(oracle, stack, requests, self.recipe, PARAMS)
        kinds = [c['kind'] for c in oracle.chunks];self.assertEqual(kinds[0], 'roots');self.assertIn('children', kinds)
        self.assertTrue(all(c['seconds'] >= c['step_seconds'] + c['head_seconds'] - 1e-9 for c in oracle.chunks))
        self.assertEqual(oracle.chunks[-1]['rows_done'], oracle.chunks[-1]['rows'])

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
        # 16: the opponent's view has both an encoder error and a non-finite prior -> 5 first; 19: an encoder
        # error in a WAITING opponent's view -> expanded (its view is unused).
        modes = ((13, 0), (14, 1), (15, 0), (4, 1), (5, 0), (16, 1), (19, 0))
        _, _, children, enumerated, leaves = self.run_oracle(tuple((mode, owner, (30 + i, 40 + i)) for i, (mode, owner) in enumerate(modes)))
        by_request = lambda q, key: {int(x) for x, c in zip(children[key], children['cells']) if c[0] == q}
        self.assertEqual((by_request(0, 'status'), by_request(0, 'error')), ({3}, {5}))
        self.assertEqual((by_request(1, 'status'), by_request(1, 'error')), ({3}, {6}))
        self.assertEqual((by_request(2, 'status'), by_request(2, 'error')), ({2}, {0}))
        self.assertEqual((by_request(3, 'status'), by_request(3, 'error')), ({3}, {3}))
        self.assertEqual((by_request(4, 'status'), by_request(4, 'error')), ({3}, {4}))
        self.assertEqual((by_request(5, 'status'), by_request(5, 'error')), ({3}, {5}))
        self.assertEqual((by_request(6, 'status'), by_request(6, 'error')), ({2}, {0}))
        for q in (0, 1, 5):   # codes 5/6 keep D's learner value: the learner head at the child
            rows = [i for i, c in enumerate(children['cells']) if c[0] == q]
            np.testing.assert_array_equal(children['value'][rows], children['head_value'][rows])
            self.assertTrue(np.isfinite(children['value'][rows]).all())
        rows = [i for i, c in enumerate(children['cells']) if c[0] in (3, 4)]
        self.assertTrue(np.isnan(children['value'][rows]).all())
        self.assertEqual({l[0] for l in enumerated}, {2, 6});self.assertEqual({l[5] for l in enumerated}, {0})   # 5/6 never expand


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
            if not children[f'{seat}_requested'][i]:f[f't1_{seat}_logits'][at] = 0.   # a waiting seat's prior is unused; zeroed, as D
            f['t1_digest_' + seat][at] = children[seat + '_digest'][i]
    leaves = data['leaves']
    for i, spec in enumerate(leaves['specs'].tolist()):
        at = tuple(spec)
        for name in ('status', 'error', 'value', 'digest'):f['t2_' + name][at] = leaves[name][i]
    return f


class GateFixture:
    """The facts-based requests, fake engine, real FramePricer and the oracle's own surface, priced once per class."""
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



class GateTests(GateFixture, unittest.TestCase):
    def test_the_oracles_own_surface_passes_every_check(self):
        frame, d_results, data, _ = self.priced()
        result = self.evaluate(frame);checks = result['checks']
        self.assertTrue(result['passed'], {k: v.get('first_mismatches') for k, v in checks.items()})
        self.assertEqual(checks['c']['quantities']['t2_value']['exact_equal_fraction'], 1.)
        statuses = collections.Counter(r.status for r in d_results)
        self.assertEqual(statuses, {'LABELED': 8, 'REFUSED': 4})   # D refuses invalid/encoder/missing/pre roots
        for name in ('t1_status', 't2_status', 't2_digest', 't1_digest_peer', 't1_value_terminal', 't2_value_terminal'):
            self.assertGreater(checks['a']['compared'][name], 0, name)
        for name in ('t1_status_vs_d', 't1_depth2_code_status', 't1_error_vs_d', 'pre_vs_d', 'root_peer_legal_vs_d',
                'd_refused_world_unenumerated', 't1_enumeration_vs_d'):
            self.assertGreater(checks['b']['compared'][name], 0, name)
        for name in ('t1_value', 'root_peer_logits'):
            self.assertGreater(checks['b']['values'][name]['compared'], 0);self.assertEqual(checks['b']['values'][name]['exact_equal_fraction'], 1.)
        self.assertIn('not bitwise', checks['b']['reference'])
        rule = result['tolerance']
        self.assertEqual((rule['depth1_noise_floor'], rule['value_bound'], rule['logit_bound']), (0., 0., .1))   # the fake heads are exact
        self.assertEqual((checks['b']['values']['t1_value']['bound'], checks['b']['values']['root_peer_logits']['bound']), (0., .1))
        floor = checks['c']['quantities']['depth1_noise_floor']
        self.assertGreater(floor['compared'], 0);self.assertEqual(floor['max_abs_difference'], 0.)   # the fake heads are exact
        self.assertIsNone(checks['c']['t2_value_to_floor_ratio']);self.assertEqual(set(checks['c']['gated']), set(gate.GATED) - {'raw_head_t2_value', 'raw_head_t1_value'})
        self.assertEqual({k: v['gates'] for k, v in checks['c']['quantities'].items()}['depth1_noise_floor'], None)
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
            ('t1_value', bump('t1_value', expanded, 2e-4), 'b_values'),   # the fake's floor is 0: the value bound is 0
            ('root_peer_logits', bump('root_peer_logits', (0, 0, 3), .2), 'b_values'),
            ('t1_value_head', bump('t1_value', expanded, 2e-4), 'c'), ('t1_peer_logits', bump('t1_peer_logits', expanded + (0,), .2), 'c'),
            ('t1_own_logits', bump('t1_own_logits', expanded + (1,), .2), 'c'),
            ('t2_value', bump('t2_value', head, .02), 'c'), ('t2_value', setter('t2_value', head, np.nan), 'c'),
            ('t2_extra_in_pricer', setter('t2_status', unexpanded, 4), 'd'), ('t2_missing_from_pricer', setter('t2_status', head, 0), 'd'),
            ('t1_extra_in_pricer', setter('t1_status', (0, 0, 8, 8), 2), 'd'), ('t1_enumeration_vs_d', setter('t1_status', (0, 0, 8, 8), 2), 'b'),
            ('t1_extra_in_pricer', setter('t1_status', (6, 1, 0, 0), 2), 'd'), ('d_refused_world_unenumerated', setter('t1_status', (6, 1, 0, 0), 2), 'b'),
            ('t1_error', setter('t1_error', code5, 6), 'a'), ('t1_error', setter('t1_error', code6, 0), 'a'),
            ('t1_depth2_code_status', setter('t1_status', code5, 2), 'b'), ('t1_status', setter('t1_status', code6, 2), 'a'),
            ('t1_value', bump('t1_value', code6, 2e-4), 'b_values'), ('t1_value', bump('t1_value', code5, 2e-4), 'b_values'),
            ('t1_digest_peer', flip('t1_digest_peer', code5 + (1,)), 'a'),
            ('pre', setter('pre', (6, 1), 0), 'a'), ('pre_vs_d', setter('pre', (6, 1), 0), 'b'),
        )
        for name, change, check in cases:
            with self.subTest(name):
                result = self.evaluate(self.mutated(change));found = result['checks'][check.removesuffix('_values')]
                self.assertFalse(result['passed']);self.assertFalse(found['passed'])
                if check == 'c':self.assertFalse(found['quantities'][name]['within_bound'])
                elif check == 'b_values':self.assertFalse(found['values'][name]['within_bound']);self.assertTrue(found['values'][name]['first_beyond'])
                else:
                    self.assertGreater(found['mismatched'][name], 0, found['mismatched'])
                    self.assertIn(name, {m['field'] for m in found['first_mismatches']})
                    self.assertTrue(all(m['block'] == 0 for m in found['first_mismatches']))

    def test_root_logit_floor_is_reported_not_gated(self):
        c = self.evaluate(self.priced()[0])['checks']['c']
        self.assertEqual(c['root_logit_floor'], 0.);self.assertIsNone(c['quantities']['root_logit_floor']['gates'])
        self.assertNotIn('root_logit_floor', c['gated']);self.assertNotIn('depth1_noise_floor', c['gated'])

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
        counts = {};forced = set(map(tuple, np.argwhere((frame['t2_status'] == 1) | (frame['t2_status'] == 3)).tolist()))
        for spec in data['leaves']['specs'].tolist():
            if tuple(spec) not in forced:counts[tuple(spec[:2])] = counts.get(tuple(spec[:2]), 0) + 1
        self.assertTrue(counts);self.assertLessEqual(max(counts.values()), 2)   # plus every terminal/error leaf the pricer reports
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
        with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=forward), \
                patch.dict(sys.modules, {'gpu_engine.depth2_backups': FakeBackups}):
            evaluation, summary = gate.run_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS,
                pricer=OraclePricer(self.ex, self.WIDTHS), frame_pricer=frame_pricer, oracle=reference.Oracle(self.ex, object(), chunk=32),
                widths=self.WIDTHS, sample=None, block=0, context=dict(block=0, call=0))
        self.assertTrue(all(summary['passed'].values()), summary);self.assertIn('e', summary['passed'])
        costs = gate.merge_costs([summary])
        self.assertGreater(costs['frame_pricer']['lower_and_compile_seconds'], 0);self.assertGreaterEqual(costs['frame_pricer']['device_seconds'], 0)
        self.assertIsNone(costs['depth2_pricer']['device_seconds'])   # the fake pricer's timing reports none
        self.assertGreater(costs['oracle']['chunks'], 0);self.assertGreater(costs['oracle']['chunk_device_seconds'], 0)
        self.assertEqual((summary['roots'], summary['worlds']), (len(self.MODES), 2 * len(self.MODES) - len(self.MISSING)))
        self.assertEqual(summary['pricer_timing'], dict(fake=True))
        self.assertEqual(summary['refusals']['depth2']['refused_count'], 7)

    def test_block_invariance_requires_bit_equal_arrays(self):
        frame, _, data, _ = self.priced();leaves = data['leaves']
        head = tuple(leaves['specs'][list(leaves['status']).index(4)])
        same = gate.block_invariance(frame, copy.deepcopy(frame))
        self.assertTrue(same['passed']);self.assertEqual(same['arrays_compared'], len(gate.SHAPES))
        rule = gate.tolerance_rule(dict(compared=10, nonfinite=0, max_abs_difference=1e-4), None)   # value bound 2e-4
        def changed(change):
            other = copy.deepcopy(frame);change(other);return gate.block_invariance(frame, other, context=dict(block=0, call=0))
        judge = lambda report: gate.merge_invariance([report], rule=rule)
        def ulp(f):f['t2_value'][head] = np.nextafter(f['t2_value'][head], np.float32(2))
        def payload(f):f['t2_value'][(0, 0, 8, 8, 8, 8)] = np.uint32(0x7fc00001).view(np.float32)   # NaN == NaN numerically
        def dtype(f):f['t1_status'] = f['t1_status'].astype(np.int32)
        def missing(f):f.pop('t2_digest')
        result = judge(changed(ulp))   # noise within 2 x floor passes, and is reported
        self.assertTrue(result['passed']);self.assertEqual(result['floats']['t2_value']['compared'] - result['floats']['t2_value']['exact_equal'], 1)
        def far(f):f['t2_value'][head] = np.float32(f['t2_value'][head] + 3e-4)
        result = judge(changed(far))
        self.assertFalse(result['passed']);self.assertTrue(result['discrete_equal']);self.assertEqual(result['floats']['t2_value']['beyond_bound'], 1)
        self.assertEqual(result['floats']['t2_value']['first_beyond'][0]['cell'], list(head));self.assertEqual(result['floats']['t2_value']['first_beyond'][0]['block'], 0)
        self.assertEqual(judge(changed(payload))['floats']['t2_value']['beyond_bound'], 1)   # a NaN only on one side is never within a bound
        def logit(delta):
            def change(f):f['t1_own_logits'][head[:4] + (1,)] += np.float32(delta)
            return change
        self.assertTrue(judge(changed(logit(.05)))['passed']);self.assertFalse(judge(changed(logit(.2)))['passed'])
        def status(f):f['t2_status'][head] = 1
        self.assertEqual(changed(status)['differing']['t2_status']['count'], 1);self.assertFalse(judge(changed(status))['passed'])   # discrete stays exact
        self.assertIn('dtype/shape', changed(dtype)['differing']['t1_status']['problem'])
        self.assertIn('missing', changed(missing)['differing']['t2_digest']['problem'])
        reports = dict(frame, world_reports=[[dict(seed=1)]]), dict(frame, world_reports=[[dict(seed=2)]])
        result = gate.block_invariance(*reports);self.assertTrue(result['passed']);self.assertEqual(result['nonarray_equal'], dict(world_reports=False))
        guarded = gate.merge_invariance([changed(ulp)], rule=gate.tolerance_rule(dict(compared=10, nonfinite=0, max_abs_difference=1e-4), 1e-9))
        self.assertFalse(guarded['passed']);self.assertFalse(guarded['floats']['t2_value']['within_guard'])   # the guard bounds value fields too

    def test_run_call_with_block_invariance_gates_the_frames(self):
        class Nudged(OraclePricer):
            def price(self, *args):
                frame = super().price(*args);at = tuple(np.argwhere(frame['t2_status'] == 4)[0])
                frame['t2_value'][at] = np.float32(frame['t2_value'][at] + 1e-3);return frame
        for other, expected in ((OraclePricer(self.ex, self.WIDTHS), True), (Nudged(self.ex, self.WIDTHS), False)):
            frame_pricer = batch.FramePricer(self.ex, object(), capacity=16, block_rows=16, actor_capacity=3, vectorized=True, host_prep=True)
            with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=forward):
                evaluation, summary = gate.run_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS,
                    pricer=OraclePricer(self.ex, self.WIDTHS), frame_pricer=frame_pricer, oracle=reference.Oracle(self.ex, object(), chunk=32),
                    widths=self.WIDTHS, sample=2, block=0, context=dict(block=0, call=0), invariance=[other])
            result = gate.summarize([evaluation])
            self.assertTrue(summary['passed']['block_invariance_discrete']);self.assertEqual(result['checks']['block_invariance']['passed'], expected)
            self.assertEqual(result['passed'], expected);self.assertIn('invariance_pricer_0_seconds', summary['timing'])
            self.assertTrue(result['checks']['a']['passed'])   # the checked (second) frame is the clean one

    def test_progress_lines_bracket_every_stage_of_a_call(self):
        lines = [];frame_pricer = batch.FramePricer(self.ex, object(), capacity=16, block_rows=16, actor_capacity=3, vectorized=True, host_prep=True)
        oracle = reference.Oracle(self.ex, object(), chunk=32, progress=gate.oracle_chunk_progress(lambda m, **v: lines.append((m, v)), interval=1e9))
        with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=forward), \
                patch.dict(sys.modules, {'gpu_engine.depth2_backups': FakeBackups}):
            gate.run_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS, pricer=OraclePricer(self.ex, self.WIDTHS),
                frame_pricer=frame_pricer, oracle=oracle, widths=self.WIDTHS, sample=None, block=0, context=dict(block=0, call=3),
                invariance=[OraclePricer(self.ex, self.WIDTHS)], progress=lambda m, **v: lines.append((m, v)))
        stages = [(m, v['stage']) for m, v in lines if m.startswith('STAGE')]
        expected = ('depth2_pricer_invariance_0', 'depth2_pricer', 'frame_pricer', 'oracle_worlds', 'oracle_roots', 'oracle_children', 'oracle_leaves',
            'check_a', 'check_b', 'check_c', 'check_d', 'check_oracle_consistency', 'check_block_invariance', 'refusals', 'check_e_decision', 'values')
        self.assertEqual(stages, [(m, name) for name in expected for m in ('STAGE_START', 'STAGE_END')])
        ends = {v['stage']: v for m, v in lines if m == 'STAGE_END'}
        self.assertTrue(all(v['seconds'] >= 0 and v['call'] == 3 and v['block'] == 0 for v in ends.values()))
        self.assertTrue(all(ends[name]['roots'] == len(self.MODES) for name in expected[:7]))
        self.assertGreater(ends['frame_pricer']['compile_seconds'], 0);self.assertIn('device_seconds', ends['depth2_pricer'])
        chunks = [v for m, v in lines if m == 'ORACLE_CHUNK']
        for kind in ('roots', 'children', 'leaves'):   # first and last chunk of every pass, with a running count
            mine = [c for c in chunks if c['kind'] == kind]
            self.assertEqual([c['chunk'] for c in mine], sorted({0, mine[0]['chunks'] - 1}))
            self.assertEqual(mine[-1]['rows_done'], mine[-1]['rows']);self.assertGreater(mine[0]['seconds'], 0)
        self.assertEqual(len(oracle.chunks), sum(c['chunks'] for c in chunks if c['chunk'] == 0))

    def test_oracle_chunk_progress_is_throttled(self):
        times = iter([0., 1., 2., 11., 12., 13.]);lines = []
        progress = gate.oracle_chunk_progress(lambda m, **v: lines.append(v['chunk']), interval=10., clock=lambda: next(times))
        for chunk in range(6):progress(kind='leaves', chunk=chunk, chunks=6, rows_done=chunk, rows=6)
        self.assertEqual(lines, [0, 3, 5])

    def test_refusal_shares_and_error_counts_by_code(self):
        frame, d_results, data, _ = self.priced()
        with patch.dict(sys.modules, {'gpu_engine.depth2_backups': None}):   # the gate's own count (the module may exist)
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


class FakeBackups:
    """A small learner-max with depth2_backups' call shapes (the real module lands with the pricer): an expanded
    cell is max over enumerated a2 of the opponent-prior average of t2_value; a root action's Q is the mean over
    usable worlds and enumerated replies; the target is softmax(root logits + Q) over legal actions."""
    @staticmethod
    def _cell(frame, q, w, a1, r1):
        status = int(frame.t1_status[q, w, a1, r1])
        if status == 1:return float(frame.t1_value[q, w, a1, r1]), -1
        t2 = frame.t2_status[q, w, a1, r1]
        if status != 2 or (t2 == 3).any():return None, -1
        own, opp = (t2 != 0).any(axis=1), (t2 != 0).any(axis=0)
        logits = np.where(opp, np.asarray(frame.t1_peer_logits[q, w, a1, r1], np.float64), -np.inf);prior = np.exp(logits - logits.max());prior /= prior.sum()
        rows = [float(np.sum(prior[opp] * np.asarray(frame.t2_value[q, w, a1, r1, a2], np.float64)[opp])) if own[a2] else None for a2 in range(9)]
        best = gate._argmax(rows, own);return rows[best], best

    @classmethod
    def learner_max_choice(cls, frame, q, w):
        return np.asarray([[cls._cell(frame, q, w, a1, r1)[1] if frame.t1_status[q, w, a1, r1] == 2 else -1 for r1 in range(9)] for a1 in range(9)])

    @classmethod
    def learner_max_root(cls, frame, q, logits, legal):
        if (frame.pre[q] != 0).any():return SimpleNamespace(status='REFUSED', improvement=None)
        values = []
        for a1 in range(9):
            cells = [cls._cell(frame, q, w, a1, r1)[0] for w in (0, 1) for r1 in range(9) if frame.t1_status[q, w, a1, r1]] if legal[a1] else []
            if legal[a1] and (not cells or any(c is None for c in cells)):return SimpleNamespace(status='REFUSED', improvement=None)
            values.append(float(np.mean(cells)) if legal[a1] else None)
        scores = np.where(legal, np.asarray(logits) + np.asarray([v or 0. for v in values]), -np.inf);target = np.exp(scores - scores.max())
        return SimpleNamespace(status='LABELED', improvement=SimpleNamespace(completed_q=tuple(values), policy_target=tuple(target / target.sum())))


class DecisionTests(GateFixture, unittest.TestCase):
    """Check (e): learner-max decisions on the pricer frame vs the oracle-heads copy."""
    def test_oracle_heads_frame_replaces_only_head_outputs(self):
        frame, _, data, _ = self.priced();frame = copy.deepcopy(frame);frame['t2_value'][frame['t2_status'] == 4] += np.float32(.25)
        frame['t1_own_logits'][frame['t1_status'] == 2] += np.float32(1.);before = copy.deepcopy(frame)
        swapped, counts = gate.oracle_heads_frame(frame, data)
        self.assertEqual(counts['unmatched_head_leaves'], 0);self.assertEqual(counts['t2_value'], int(np.count_nonzero(frame['t2_status'] == 4)))
        leaves = data['leaves'];head = leaves['status'] == 4
        np.testing.assert_array_equal(swapped.t2_value[tuple(leaves['specs'][head].T)], leaves['head_value'][head])
        children = data['children'];expanded = children['status'] == 2
        lanes = expanded[:, None] & children['own_requested'][:, None] & children['own_legal'];rows, lane = np.nonzero(lanes)
        cells = tuple(children['cells'][rows].T) + (lane,)
        np.testing.assert_array_equal(swapped.t1_own_logits[cells], children['own_logits'][rows, lane])   # the lanes (c) checks
        other = expanded[:, None] & ~lanes;rows, lane = np.nonzero(other);cells = tuple(children['cells'][rows].T) + (lane,)
        self.assertTrue(len(rows));np.testing.assert_array_equal(swapped.t1_own_logits[cells], frame['t1_own_logits'][cells])   # the rest stay the pricer's
        terminal = frame['t2_status'] == 1;np.testing.assert_array_equal(swapped.t2_value[terminal], frame['t2_value'][terminal])
        for name in ('t1_status', 't2_status', 't1_value', 't1_error', 'pre'):np.testing.assert_array_equal(getattr(swapped, name), frame[name])
        for name in before:
            if isinstance(before[name], np.ndarray):np.testing.assert_array_equal(frame[name], before[name])   # the input is not modified
        frame['t2_status'][(0, 0, 8, 8, 8, 8)] = 4   # a head-valued leaf the oracle never evaluated
        self.assertEqual(gate.oracle_heads_frame(frame, data)[1]['unmatched_head_leaves'], 1)
        self.assertFalse(gate.decision_check(frame, data, self.requests, backups=FakeBackups)['passed'])

    def test_same_heads_give_the_same_decisions(self):
        frame, _, data, _ = self.priced()
        result = gate.decision_check(frame, data, self.requests, context=dict(block=0, call=0), backups=FakeBackups)
        self.assertTrue(result['passed'], result);self.assertEqual(result['roots_compared'], 5);self.assertEqual(result['both_refused'], 7)
        self.assertEqual((result['argmax_q_differs'], result['argmax_target_differs'], result['cell_choice_differs'], result['status_differs']), (0, 0, 0, 0))
        self.assertEqual((result['max_abs_delta_q'], result['max_abs_delta_target']), (0., 0.));self.assertGreater(result['cells_compared'], 0)
        self.assertEqual(result['nonfinite_roots'], 0)
        class Nan(FakeBackups):
            @classmethod
            def learner_max_root(cls, f, q, logits, legal):
                out = super().learner_max_root(f, q, logits, legal)
                if q != 0 or out.improvement is None:return out
                return SimpleNamespace(status=out.status, improvement=SimpleNamespace(completed_q=(np.nan,) + out.improvement.completed_q[1:],
                    policy_target=out.improvement.policy_target))
        result = gate.decision_check(frame, data, self.requests, backups=Nan)
        self.assertFalse(result['passed']);self.assertEqual(result['nonfinite_roots'], 1)

    def test_a_flipped_root_decision_fails_and_lists_the_root(self):
        frame, _, data, _ = self.priced();clean = FakeBackups.learner_max_root(SimpleNamespace(**frame), 0, self.requests[0].root_logits, BASE)
        best = gate._argmax(clean.improvement.completed_q, BASE);other = next(a for a in range(3) if a != best)
        noisy = copy.deepcopy(frame);head = noisy['t2_status'][0, :, other] == 4;noisy['t2_value'][0, :, other][head] = 1.
        result = gate.merge_decisions([gate.decision_check(noisy, data, self.requests, context=dict(block=0, call=0), backups=FakeBackups)])
        self.assertFalse(result['passed']);self.assertEqual((result['argmax_q_differs'], result['argmax_target_differs']), (1, 1))
        self.assertGreater(result['cell_choice_differs'], 0);self.assertGreater(result['max_abs_delta_q'], .1)
        first = result['first_differing_roots'][0]
        self.assertEqual((first['root'], first['argmax_q'], first['block']), (0, [other, best], 0));self.assertEqual(len(first['completed_q_pricer']), 9)

    def test_cell_differences_are_reported_not_gated_and_status_flips_gate(self):
        frame, _, data, _ = self.priced()
        class Cells(FakeBackups):
            @classmethod
            def learner_max_choice(cls, f, q, w):
                choice = super().learner_max_choice(f, q, w)
                return np.where(choice >= 0, 8 - choice, choice) if f.t2_value is not frame['t2_value'] else choice
        result = gate.decision_check(frame, data, self.requests, backups=Cells)
        self.assertTrue(result['passed']);self.assertGreater(result['cell_choice_differs'], 0)
        class Refuses(FakeBackups):
            @classmethod
            def learner_max_root(cls, f, q, logits, legal):
                return SimpleNamespace(status='REFUSED', improvement=None) if q == 0 and f.t2_value is not frame['t2_value'] else super().learner_max_root(f, q, logits, legal)
        result = gate.decision_check(frame, data, self.requests, backups=Refuses)
        self.assertFalse(result['passed']);self.assertEqual((result['status_differs'], result['argmax_q_differs'], result['roots_compared']), (1, 1, 4))

    def test_missing_backups_fail_closed(self):
        frame, _, data, _ = self.priced()
        with patch.dict(sys.modules, {'gpu_engine.depth2_backups': None}):result = gate.decision_check(frame, data, self.requests)
        self.assertFalse(result['passed']);self.assertIn('fail closed', result['reason'])
        merged = gate.merge_decisions([result]);self.assertFalse(merged['passed']);self.assertTrue(merged['reasons'])

    def test_run_call_gates_e_and_saves_arrays(self):
        frame_pricer = batch.FramePricer(self.ex, object(), capacity=16, block_rows=16, actor_capacity=3, vectorized=True, host_prep=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'arrays-block0-call0.npz'
            with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=forward), \
                    patch.dict(sys.modules, {'gpu_engine.depth2_backups': FakeBackups}):
                evaluation, summary = gate.run_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS,
                    pricer=OraclePricer(self.ex, self.WIDTHS), frame_pricer=frame_pricer, oracle=reference.Oracle(self.ex, object(), chunk=32),
                    widths=self.WIDTHS, sample=None, block=0, context=dict(block=0, call=0), save=path)
                sampled, _ = gate.run_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS,
                    pricer=OraclePricer(self.ex, self.WIDTHS), frame_pricer=frame_pricer, oracle=reference.Oracle(self.ex, object(), chunk=32),
                    widths=self.WIDTHS, sample=2, block=0, context=dict(block=0, call=1))
            self.assertTrue(summary['passed']['e']);self.assertNotIn('e', sampled)   # (e) needs every leaf
            result = gate.summarize([evaluation, sampled]);self.assertTrue(result['passed']);self.assertEqual(result['checks']['e']['roots'], 12)
            with np.load(path) as saved:
                self.assertEqual(saved['frame_t2_value'].shape, (12, 2, 9, 9, 9, 9));self.assertEqual(saved['request_root_logits'].shape, (12, 9))
                self.assertEqual(len(saved['oracle_leaf_head_value']), len(saved['oracle_leaf_specs']))
                self.assertNotIn('frame_timing', saved.files);self.assertEqual(summary['saved_arrays'], str(path))
                # Offline: the saved arrays alone reproduce (e).
                fields = {name[6:]: saved[name] for name in saved.files if name.startswith('frame_')}
                data = dict(leaves=dict(specs=saved['oracle_leaf_specs'], status=saved['oracle_leaf_status'], head_value=saved['oracle_leaf_head_value']),
                    children={name: saved['oracle_child_' + name] for name in ('cells', 'status', 'own_logits', 'peer_logits', 'own_requested',
                        'own_legal', 'peer_requested', 'peer_legal')})
                self.assertTrue(gate.decision_check(fields, data, self.requests, backups=FakeBackups)['passed'])


def blank_frame(count):
    """A contract-shaped frame with nothing enumerated and every world usable."""
    frame = frame_from(count, dict(roots={}, children=dict(cells=np.zeros((0, 4), np.int64)), leaves=dict(specs=np.zeros((0, 6), np.int64))))
    frame['pre'][:] = 0;return frame


def plain_request(frame_index, legal, logits=(0.,) * 9):
    return SimpleNamespace(root_logits=tuple(logits), root_legal=tuple(legal), root_inputs={'numeric_features': None},
        plans=SimpleNamespace(root=SimpleNamespace(address=SimpleNamespace(stream=7, frame=frame_index, owner=0))))


V1_KEYS = ('oracle_leaf_specs', 'oracle_leaf_status', 'oracle_leaf_head_value', 'oracle_child_cells', 'oracle_child_status', 'oracle_child_head_value',
    'oracle_child_own_logits', 'oracle_child_peer_logits', 'request_root_logits', 'request_root_legal', 'request_address')


class AnalysisTests(GateFixture, unittest.TestCase):
    """The values section, sibling spread, pruning report, saved arrays, --recheck and --pricer-only."""
    def run_saved(self, directory, *, invariance=True):
        frame_pricer = batch.FramePricer(self.ex, object(), capacity=16, block_rows=16, actor_capacity=3, vectorized=True, host_prep=True)
        with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=forward), \
                patch.dict(sys.modules, {'gpu_engine.depth2_backups': FakeBackups}):
            return gate.run_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS,
                pricer=OraclePricer(self.ex, self.WIDTHS), frame_pricer=frame_pricer, oracle=reference.Oracle(self.ex, object(), chunk=32),
                widths=self.WIDTHS, sample=None, block=0, context=dict(block=0, call=0),
                invariance=[OraclePricer(self.ex, self.WIDTHS)] if invariance else None, save=Path(directory) / 'arrays-block0-call0.npz')

    def test_values_by_frame_band(self):
        frame, _, data, _ = self.priced();result = gate.value_samples(frame, self.requests, backups=FakeBackups, context=dict(block=0, call=0))
        section = gate.values_section([(result, dict(differing_roots=[]))], oracle_checked=True)
        totals, bands = section['totals'], section['by_band']
        self.assertEqual(totals['abs_t2_value']['count'], int(np.count_nonzero(frame['t2_status'] == 4)))
        self.assertEqual(totals['abs_t1_value']['count'], int(np.count_nonzero(frame['t1_status'] == 2)))
        self.assertEqual(sum(bands[b]['abs_t2_value']['count'] for b in bands), totals['abs_t2_value']['count'])
        self.assertEqual(bands['0-4']['abs_t1_value']['count'], int(np.count_nonzero(frame['t1_status'][0] == 2)))   # request 0 is frame 4
        self.assertEqual(bands['40+']['abs_t2_value'], dict(count=0));self.assertEqual(set(bands), {'0-4', '5-9', '10-19', '20-39', '40+'})
        self.assertLessEqual(totals['abs_t2_value']['p1'], totals['abs_t2_value']['p99'])
        cells = [c for c in map(tuple, np.argwhere(frame['t1_status'] == 2).tolist())
            if not (frame['t2_status'][c] == 3).any() and np.count_nonzero((frame['t2_status'][c] != 0).any(axis=1)) >= 2]
        self.assertEqual(totals['cell_a2_spread']['count'], len(cells))
        for c in cells[:5]:   # the learner-max value of a cell is the max of its a2 rows, as the reference computes it
            rows = gate.cell_rows(frame['t2_status'][c], frame['t2_value'][c], frame['t1_peer_logits'][c])
            self.assertAlmostEqual(np.nanmax(rows), FakeBackups._cell(SimpleNamespace(**frame), *c)[0], places=12)
        self.assertEqual(totals['root_q_spread']['count'], 5);self.assertEqual(section['near_ties']['labeled_roots'], 5)
        self.assertTrue(section['oracle_checked']);self.assertEqual(section['source'], 'PRICER_FRAME')
        roots = [dict(root=0, band='0-4', top_two_gap=5e-5, address=[1, 2, 0]), dict(root=1, band='5-9', top_two_gap=.01, address=[1, 6, 0]),
            dict(root=2, band='5-9', top_two_gap=1e-4, address=[1, 7, 0])]
        near = gate.values_section([(dict(samples={}, roots=roots, unavailable=None), dict(differing_roots=[0]))], oracle_checked=True)['near_ties']
        self.assertEqual((near['near_tie_roots'], near['share'], near['by_band']['0-4'], near['by_band']['5-9']), (2, 2 / 3, 1, 1))
        self.assertFalse(near['e_exact_on_all']);self.assertEqual([r['root'] for r in near['e_differs_on']], [0])
        unchecked = gate.values_section([(dict(samples={}, roots=roots, unavailable=None), None)], oracle_checked=False)
        self.assertIsNone(unchecked['near_ties']['e_exact_on_all']);self.assertIn('NOT oracle-checked', unchecked['scope'])
        with patch.dict(sys.modules, {'gpu_engine.depth2_backups': None}):
            missing = gate.value_samples(frame, self.requests)
        self.assertIn('depth2_backups', missing['unavailable']);self.assertEqual(missing['roots'], [])

    def test_sibling_spread(self):
        frame = blank_frame(1)
        for a1, values in ((0, (0., 1e-5, 1e-3)), (1, (0., 1e-6, 2e-6))):
            frame['t1_status'][0, 0, a1, 0] = 2
            for a2, v in enumerate(values):frame['t2_status'][0, 0, a1, 0, a2, 0] = 4;frame['t2_value'][0, 0, a1, 0, a2, 0] = v
        frame['t2_status'][0, 0, 1, 0, 3, 0] = 1;frame['t2_value'][0, 0, 1, 0, 3, 0] = 1.   # a terminal sibling is not head-valued
        result = gate.merge_sibling_spread([gate.sibling_spread(frame, context=dict(call=0))], oracle_checked=False)
        pairs = [np.abs(np.subtract.outer(v, v))[np.triu_indices(3, 1)] for v in (np.float32([0., 1e-5, 1e-3]), np.float32([0., 1e-6, 2e-6]))]
        self.assertEqual(len(result['worlds']), 1);world = result['worlds'][0]
        self.assertAlmostEqual(world['median'], float(np.median(np.concatenate(pairs))), places=12);self.assertEqual(world['pairs'], 6)
        self.assertEqual((result['children'], result['children_p10_below_threshold'], result['share_children_p10_below_threshold']), (2, 1, .5))

    def test_pruning_ranks_the_chosen_a2_by_the_child_prior(self):
        frame = blank_frame(1);legal = (True, True, True) + (False,) * 6
        for r1, values in ((0, (.5, .1, .2)), (1, (.1, .9, .2))):   # learner-max picks a2 0 (prior rank 2), then a2 1 (rank 0)
            at = (0, 0, 1, r1);frame['t1_status'][at] = 2;frame['t1_own_requested'][at] = True;frame['t1_own_legal'][at] = legal
            frame['t1_own_logits'][at][:3] = (0., 2., 1.)
            for a2, v in enumerate(values):frame['t2_status'][at + (a2, 0)] = 4;frame['t2_value'][at + (a2, 0)] = v
        request = plain_request(3, (False, True) + (False,) * 7)
        report = gate.pruning_report([(frame, [request], dict(call=0))], FakeBackups)
        self.assertTrue(report['full_width']);self.assertEqual(report['chosen_a2_rank_histogram'][:3], [1, 0, 1])
        self.assertEqual((report['per_cell']['total'], report['per_cell']['outside_top2'], report['per_cell']['outside_top3']), (2, 1, 0))
        self.assertEqual(report['per_cell']['share_outside_top2'], .5);self.assertEqual(report['under_chosen_root_action']['total'], 2)
        self.assertEqual((report['roots']['labeled'], report['roots']['any_outside_top2'], report['roots']['any_outside_top3']), (1, 1, 0))
        frame['t2_status'][0, 0, 1, 1, 2, 0] = 0   # a pruned learner action: not a full-width frame
        report = gate.pruning_report([(frame, [request], dict(call=0))], FakeBackups)
        self.assertFalse(report['full_width']);self.assertEqual(report['not_full_width_cells'][0]['cell'], [0, 0, 1, 1])

    def test_saved_arrays_recheck_every_check_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation, summary = self.run_saved(directory)
            saved = gate.load_arrays(Path(directory) / 'arrays-block0-call0.npz')
            self.assertEqual(len(saved['invariance']), 1);self.assertIsNotNone(saved['d_results']);self.assertEqual(saved['notes'], [])
            self.assertEqual(saved['meta']['schema'], gate.ARRAYS_SCHEMA)
            self.assertEqual(saved['frame']['timing'], dict(fake=True))   # restored for refusal_summary
            report = gate.recheck(directory, backups=FakeBackups, progress=None)
            self.assertEqual(report['status'], 'PASS', report['unavailable']);self.assertEqual(report['unavailable'], {})
            self.assertEqual(set(report['checks']), {'a', 'b', 'c', 'd', 'oracle_consistency', 'e', 'block_invariance'});self.assertEqual(report['substitutions'], [])
            live = gate.summarize([evaluation])
            for check in ('a', 'b', 'd', 'oracle_consistency'):self.assertEqual(report['checks'][check]['compared'], live['checks'][check]['compared'], check)
            self.assertEqual(report['checks']['e']['roots_compared'], live['checks']['e']['roots_compared'])
            self.assertEqual(report['refusals']['depth1']['refused_count'], 4);self.assertTrue(report['values']['oracle_checked'])
            # A tampered pricer surface fails the offline recheck.
            with np.load(Path(directory) / 'arrays-block0-call0.npz') as z:arrays = {k: z[k] for k in z.files}
        with tempfile.TemporaryDirectory() as tampered:
            changed = dict(arrays);changed['frame_t2_status'] = arrays['frame_t2_status'].copy()
            changed['frame_t2_status'][tuple(np.argwhere(arrays['frame_t2_status'] == 4)[0])] = 1
            np.savez_compressed(Path(tampered) / 'arrays-block0-call0.npz', **changed)
            report = gate.recheck(tampered, backups=FakeBackups, progress=None)
            self.assertEqual(report['status'], 'FAIL');self.assertFalse(report['checks']['a']['passed'])
        with tempfile.TemporaryDirectory() as old:   # the arrays 5ea0552b's --save-arrays wrote
            v1 = {k: v for k, v in arrays.items() if k.startswith('frame_') or k in V1_KEYS}
            np.savez_compressed(Path(old) / 'arrays-block0-call0.npz', **v1)
            report = gate.recheck(old, backups=FakeBackups, progress=None)
            self.assertEqual(report['status'], 'INCOMPLETE');self.assertEqual(set(report['checks']), {'c', 'd', 'e'})
            self.assertIn('oracle_root_*', report['unavailable']['a']);self.assertIn('oracle_child_error', report['unavailable']['a'])
            self.assertEqual(report['unavailable']['b'], ['d_* (FramePricer outputs)']);self.assertIn('oracle_leaf_child_invalid', report['unavailable']['oracle'])
            self.assertEqual(report['unavailable']['block_invariance'], ['invariance0_frame_* (the other child-block frame)'])
            self.assertTrue(any('taken from the pricer frame' in x for x in report['substitutions']));self.assertTrue(any('v1 arrays' in x for x in report['notes']))
            self.assertTrue(report['checks']['e']['passed']);self.assertNotIn('depth1', report['refusals'])
            with patch.dict(sys.modules, {'gpu_engine.depth2_backups': SimpleNamespace(refusal_summary=lambda f: dict(root_count=len(f.pre), refused_count=7,
                    k1=f.timing.get('k1_own', 9)))}):   # the real refusal_summary reads frame.timing's k1 widths
                self.assertEqual(gate.recheck(old, backups=FakeBackups)['refusals']['depth2']['source'], 'depth2_backups.refusal_summary')
            pruning = gate.offline_pruning(old, backups=FakeBackups);self.assertTrue(pruning['full_width']);self.assertGreater(pruning['per_cell']['total'], 0)

    def test_offline_cli_writes_recheck_and_pruning(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_saved(directory, invariance=False)
            with patch.dict(sys.modules, {'gpu_engine.depth2_backups': FakeBackups}), patch.object(gate, 'log', lambda *a, **k: None):
                status = gate.main(['--out', str(Path(directory) / 'offline'), '--recheck', directory, '--pruning-report', directory])
            report = json.loads((Path(directory) / 'offline' / 'recheck.json').read_text())
            self.assertEqual((status, report['status'], report['unavailable']), (0, 'PASS', {}))
            self.assertEqual(report['skipped'], {'block_invariance': 'the run did not request --block-invariance'})
            self.assertTrue(json.loads((Path(directory) / 'offline' / 'pruning.json').read_text())['full_width'])

    def test_pricer_only_writes_values_refusals_and_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(save_arrays=True, worlds_blocks=1, max_roots=None)
            provider = SimpleNamespace(materializer=self.tables, source=self.tables.source, recipe=self.recipe)
            with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=forward), \
                    patch.dict(sys.modules, {'gpu_engine.depth2_backups': FakeBackups}), patch.object(gate, 'log', lambda *a, **k: None):
                status = gate.pricer_only(args, provider=provider, params=PARAMS, groups=[list(self.requests)], pricer=OraclePricer(self.ex, self.WIDTHS),
                    options={}, seeds=(self.recipe.seed,), out=Path(directory), source={}, widths=self.WIDTHS)
            report = json.loads((Path(directory) / 'gate.json').read_text())
            self.assertEqual((status, report['status'], report['passed']), (0, 'PRICER_ONLY', False));self.assertIn('digest=True', report['digest_note'])
            self.assertFalse(report['values']['oracle_checked']);self.assertEqual(report['values']['near_ties']['labeled_roots'], 5)
            self.assertEqual(report['refusals']['depth2']['refused_count'], 7);self.assertNotIn('checks', report)
            with np.load(Path(directory) / 'arrays-block0-call0.npz') as z:
                self.assertIn('frame_t2_value', z.files);self.assertIn('request_root_logits', z.files)
                self.assertFalse(any(k.startswith(('oracle_', 'd_')) for k in z.files))
            self.assertEqual(gate.recheck(directory, backups=FakeBackups, progress=None)['status'], 'INCOMPLETE')


@dataclasses.dataclass(frozen=True)
class FakeConfig:
    """Stands in for V5ModelConfig: replace(arm=, replicate=) and metadata() are all load_head uses."""
    arm: str = 'B'
    replicate: int = 1
    width: int = 8

    def metadata(self):return dict(arm=self.arm, replicate=self.replicate, width=self.width)


class HeadOverrideTests(GateFixture, unittest.TestCase):
    """--head-archive: U1's load and arm checks, the head's root logits, one actor, the evaluator header, --raw-head-check."""
    PLAN = SimpleNamespace(model=FakeConfig(arm='D', replicate=0), historical_window=32)
    LABELS = SimpleNamespace(sha256='a' * 64)
    HEAD = dict(head_archive='/shared/x/actor-99.npz', head_archive_sha256='1' * 64, head_checkpoint_sha256='2' * 64,
        head_category_sha256=None, head_model_sha256=None, value_reference='B1 iter~95: p50 0.57, p90 0.97')

    def load(self, config=FakeConfig(), **flags):
        args = SimpleNamespace(**dict(self.HEAD, **flags))
        with patch.object(gate_v5_model(), 'load', return_value=('PARAMS', config, dict(parameter_count=7))) as load:
            result = gate.load_head(args, plan=self.PLAN, labels=self.LABELS, category_sha256='c' * 64)
        return result, load

    def test_load_head_binds_the_bank_namespaces_and_the_d_geometry(self):
        (params, config, meta, model_sha), load = self.load()
        self.assertEqual((params, config.arm), ('PARAMS', 'B'))
        self.assertEqual(load.call_args.kwargs, dict(expected_archive_sha256='1' * 64, expected_checkpoint_sha256='2' * 64,
            expected_category_vocabulary_sha256='c' * 64, expected_label_vocabulary_sha256='a' * 64, enable_v5=True))
        self.assertEqual(load.call_args.args, ('/shared/x/actor-99.npz',))
        from pokezero import v5_run as runs
        self.assertEqual(model_sha, runs._sha(FakeConfig().metadata()))
        self.load(head_model_sha256=model_sha);self.load(head_category_sha256='c' * 64)   # matching optional checks pass

    def test_load_head_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'geometry'):self.load(config=FakeConfig(width=9))
        with self.assertRaisesRegex(ValueError, 'category'):self.load(head_category_sha256='d' * 64)
        with self.assertRaisesRegex(ValueError, 'model sha256'):self.load(head_model_sha256='e' * 64)
        args = SimpleNamespace(**self.HEAD)
        with patch.object(gate_v5_model(), 'load', side_effect=ValueError('caller-trusted v5 actor archive SHA256 differs')), \
                self.assertRaisesRegex(ValueError, 'SHA256 differs'):
            gate.load_head(args, plan=self.PLAN, labels=self.LABELS, category_sha256='c' * 64)

    def test_bind_roots_checks_arms_and_moves_every_request_to_the_head(self):
        head = {'offset': j.asarray(.05, j.float32), 'logits': j.arange(9, dtype=j.float32) / 4}
        mixed = [list(self.requests[:4])]
        self.assertEqual({(r.learner_actor, r.opponent_actor) for r in mixed[0]}, {(0, 1), (1, 0)})
        with patch.object(batch.v5_model, 'forward', side_effect=forward):
            groups, values, dropped, checked = gate.bind_roots(mixed, (head,), self.PLAN, head_config=FakeConfig())
            plain, plain_values, _, plain_checked = gate.bind_roots(mixed, PARAMS, self.PLAN)
        self.assertTrue(checked);self.assertEqual(dropped, [])
        self.assertEqual({(r.learner_actor, r.opponent_actor) for r in groups[0]}, {(0, 0)})
        self.assertEqual(groups[0][0].root_logits, tuple(map(float, np.arange(9, dtype=np.float32) / 4)))   # the head's root prior, as U1
        self.assertAlmostEqual(values[0], abs(float(np.asarray(mixed[0][0].root_inputs['numeric_features']).reshape(-1)[0]) + .05), places=6)
        self.assertFalse(plain_checked);self.assertEqual([r for g in plain for r in g], mixed[0])   # no override: requests untouched
        self.assertAlmostEqual(plain_values[1] - plain_values[0], .07, places=5)   # each learner's own params (offset .07 vs 0)
        def other_arm(params, inputs, model=None, **kw):
            out = forward(params, inputs);return out._replace(value=out.value + (1e-3 if isinstance(model, FakeConfig) and model.arm == 'B' else 0.))
        with patch.object(batch.v5_model, 'forward', side_effect=other_arm), self.assertRaisesRegex(ValueError, 'D-arm and head-arm'):
            gate.bind_roots(mixed, (head,), self.PLAN, head_config=FakeConfig())
        nan = {'offset': j.asarray(np.nan, j.float32), 'logits': j.zeros(9, j.float32)}   # a NaN on the first root: not "arms differ"
        with patch.object(batch.v5_model, 'forward', side_effect=forward), self.assertRaisesRegex(ValueError, 'invalid or non-finite'):
            gate.bind_roots(mixed, (nan,), self.PLAN, head_config=FakeConfig())
        bad = {'offset': j.asarray(5., j.float32), 'logits': j.zeros(9, j.float32)}   # |v| > 1: an invalid head output
        first_ok = [[mixed[0][0]] + list(mixed[0][1:])]
        def later_bad(params, inputs, model=None, **kw):   # the first root is valid; the others are dropped
            out = forward(params, inputs);return out._replace(value=out.value + 5. * (inputs['numeric_features'][:, 0, 0, 1] != first_clock))
        first_clock = float(np.asarray(mixed[0][0].root_inputs['numeric_features']).reshape(-1)[1])
        with patch.object(batch.v5_model, 'forward', side_effect=later_bad):
            groups, values, dropped, _ = gate.bind_roots(first_ok, (head,), self.PLAN, head_config=FakeConfig())
        self.assertEqual((len(groups[0]), len(dropped), len(values)), (1, 3, 1))
        refusals = gate.with_dropped(dict(depth2=dict(refused_count=0)), dropped)
        self.assertEqual(refusals['dropped_roots']['count'], 3);self.assertEqual(refusals['depth2'], dict(refused_count=0))

    def test_evaluator_header(self):
        loaded = SimpleNamespace(plan=self.PLAN, labels=self.LABELS, category_sha256='c' * 64, snapshot=dict(iteration=-1, checkpoint='untrained-smoke.pt',
            checkpoint_sha256='3' * 64, archive='run/actor--1.npz', archive_sha256='4' * 64, model_sha256='5' * 64))
        args = SimpleNamespace(**self.HEAD)
        own = gate.evaluator_header(args, loaded, None)
        self.assertEqual((own['override'], own['iteration'], own['checkpoint'], own['head_arm'], own['pricer_arm']), (False, -1, 'untrained-smoke.pt', 'D', 'D'))
        self.assertIn('UNTRAINED', own['warning'])
        head = gate.evaluator_header(args, loaded, ('PARAMS', FakeConfig(), dict(parameter_count=7), 'f' * 64))
        self.assertEqual((head['override'], head['iteration'], head['archive'], head['archive_sha256'], head['checkpoint_sha256']),
            (True, None, '/shared/x/actor-99.npz', '1' * 64, '2' * 64))
        self.assertEqual((head['head_arm'], head['pricer_arm'], head['model_sha256'], head['parameter_count']), ('B', 'D', 'f' * 64, 7))
        self.assertNotIn('warning', head)
        scale = gate.value_scale([.5, .6, .7], dict(count=3, p50=.4, p90=.9, p1=0.), args.value_reference)
        self.assertEqual(scale['reference'], 'B1 iter~95: p50 0.57, p90 0.97');self.assertEqual(scale['root_abs_value']['p50'], .6)
        self.assertEqual(scale['t1_abs_value_expanded'], dict(count=3, p50=.4, p90=.9))

    def test_raw_head_check_gates_the_pricer_against_the_head_arm(self):
        def run(shift):
            def arms(params, inputs, model=None, **kw):
                out = forward(params, inputs);return out._replace(value=out.value + (shift if model == 'HEAD_ARM' else 0.))
            frame_pricer = batch.FramePricer(self.ex, object(), capacity=16, block_rows=16, actor_capacity=3, vectorized=True, host_prep=True)
            with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=arms), \
                    patch.dict(sys.modules, {'gpu_engine.depth2_backups': FakeBackups}):
                evaluation, _ = gate.run_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS,
                    pricer=OraclePricer(self.ex, self.WIDTHS), frame_pricer=frame_pricer, widths=self.WIDTHS, sample=None, block=0, context=dict(block=0, call=0),
                    oracle=reference.Oracle(self.ex, object(), chunk=32, raw_model='HEAD_ARM'))
            return gate.summarize([evaluation])['checks']['c']
        same, shifted = run(0.), run(1e-3)
        for name in ('raw_head_t2_value', 'raw_head_t1_value'):
            self.assertTrue(same['quantities'][name]['passed']);self.assertEqual(same['quantities'][name]['exact_equal_fraction'], 1.)
            self.assertGreater(same['quantities'][name]['compared'], 0);self.assertEqual(same['quantities'][name]['gates'], 'c')
            self.assertFalse(shifted['quantities'][name]['passed'])
        self.assertTrue(same['passed']);self.assertFalse(shifted['passed'])

    def test_pricer_only_header_comes_first(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(save_arrays=False, worlds_blocks=1, max_roots=None, value_reference='ref')
            provider = SimpleNamespace(materializer=self.tables, source=self.tables.source, recipe=self.recipe)
            with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=forward), \
                    patch.dict(sys.modules, {'gpu_engine.depth2_backups': FakeBackups}), patch.object(gate, 'log', lambda *a, **k: None):
                gate.pricer_only(args, provider=provider, params=PARAMS, groups=[list(self.requests)], pricer=OraclePricer(self.ex, self.WIDTHS),
                    options={}, seeds=(self.recipe.seed,), out=Path(directory), source={}, widths=self.WIDTHS, evaluator=dict(override=True, iteration=None),
                    root_values=[.25, .5])
            report = json.loads((Path(directory) / 'gate.json').read_text())
        self.assertEqual(list(report)[:2], ['evaluator', 'value_scale']);self.assertEqual(report['value_scale']['reference'], 'ref')
        self.assertEqual(report['refusals']['dropped_roots']['count'], 0)
        self.assertEqual(report['value_scale']['root_abs_value']['count'], 2);self.assertGreater(report['value_scale']['t1_abs_value_expanded']['count'], 0)


def nan_on_invalid_forward(params, inputs, model=None, **kw):
    """The fake forward, but as v5_model.forward: a row with non-finite numeric inputs is invalid and all its outputs are NaN."""
    x = inputs['numeric_features'][:, 0, 0];valid = j.all(j.isfinite(x), axis=-1)
    logits = j.where(valid[:, None], j.broadcast_to(params['logits'], (len(x), 9)).astype(j.float32), j.nan)
    return Prediction(logits, j.where(valid, x[:, 0] + params['offset'], j.nan), valid)


def geometry_noise_forward(params, inputs, model=None, **kw):
    """Cross-geometry noise far below LOGIT_TOLERANCE: +1e-3 on lane 2 at odd batch sizes only (lanes 1 and 2 tie otherwise)."""
    x = inputs['numeric_features'][:, 0, 0];n = len(x);bump = (1e-3 if n % 2 else 0.) * (j.arange(9) == 2)
    return Prediction(j.broadcast_to(params['logits'] + bump, (n, 9)).astype(j.float32), x[:, 0] + params['offset'], j.ones(n, bool))


class RelativeToleranceTests(GateFixture, unittest.TestCase):
    """Evaluator-relative bounds: values <= 2 x the run's depth-1 noise floor (1e-4 if unmeasurable), logits <= 0.1, and
    the misplacement guard (every value max |diff| < 0.25 x the p10 cell_a2_spread)."""
    def shifted(self, floor):
        """D3's view of one expanded cell moved by `floor`: the run's measured depth-1 noise floor becomes `floor`."""
        frame, d_results, data, _ = self.priced();children = data['children']
        cell = next(tuple(c) for c, x in zip(children['cells'].tolist(), children['status']) if x == 2 and c[1] == 0)
        shifted = copy.deepcopy(d_results);report = shifted[cell[0]].world_reports[cell[1]]
        report['joint_values'][cell[2]][cell[3]] = float(np.float32(report['joint_values'][cell[2]][cell[3]] + floor))
        return frame, shifted, data, cell

    def judged(self, frame, d_results, data, *, spread_p10=None, drop_floor=False):
        evaluation = gate.evaluate(frame, d_results, self.requests, data, context=dict(block=0, call=0))
        if drop_floor:evaluation['c'].pop('depth1_noise_floor')
        return gate.summarize([evaluation], spread_p10=spread_p10)

    def noisy(self, frame, data, delta, *, logit=0.):
        heads = [tuple(s) for s, x in zip(data['leaves']['specs'].tolist(), data['leaves']['status']) if x == 4][:3]
        noisy = copy.deepcopy(frame)
        for at in heads:noisy['t2_value'][at] = np.float32(noisy['t2_value'][at] + delta)
        if logit:
            cell = next(tuple(c) for c, x in zip(data['children']['cells'].tolist(), data['children']['status']) if x == 2 and c[0] == 0)
            noisy['t1_own_logits'][cell + (1,)] = np.float32(noisy['t1_own_logits'][cell + (1,)] + logit)
        return noisy

    def test_values_pass_within_twice_the_floor_and_fail_beyond(self):
        frame, d_results, data, cell = self.shifted(1e-3)
        result = self.judged(self.noisy(frame, data, 1.5e-3), d_results, data);rule = result['tolerance']
        self.assertAlmostEqual(rule['depth1_noise_floor'], 1e-3, delta=1e-6);self.assertAlmostEqual(rule['value_bound'], 2e-3, delta=2e-6)
        self.assertEqual(rule['value_bound_source'], '2 x depth1_noise_floor')
        self.assertTrue(result['passed'], {k: v['passed'] for k, v in result['checks'].items()})
        t2 = result['checks']['c']['quantities']['t2_value']
        self.assertAlmostEqual(t2['max_abs_difference'], 1.5e-3, delta=1e-6);self.assertAlmostEqual(t2['ratio_to_bound'], .75, delta=1e-2)
        b = result['checks']['b']['values']['t1_value'];self.assertAlmostEqual(b['max_abs_difference'], 1e-3, delta=1e-6)   # D's own shift, within 2 x floor
        self.assertAlmostEqual(result['checks']['c']['t2_value_to_floor_ratio'], 1.5, delta=1e-2)
        result = self.judged(self.noisy(frame, data, 2.5e-3), d_results, data)
        self.assertFalse(result['passed']);t2 = result['checks']['c']['quantities']['t2_value']
        self.assertFalse(t2['within_bound']);self.assertEqual(t2['beyond_bound'], 3);self.assertEqual(len(t2['first_beyond']), 3)

    def test_logits_are_bounded_at_one_tenth(self):
        frame, d_results, data, _ = self.shifted(1e-3)
        self.assertTrue(self.judged(self.noisy(frame, data, 0., logit=.05), d_results, data)['passed'])
        result = self.judged(self.noisy(frame, data, 0., logit=.15), d_results, data)
        self.assertFalse(result['passed']);self.assertFalse(result['checks']['c']['quantities']['t1_own_logits']['within_bound'])
        self.assertEqual(result['checks']['c']['quantities']['t1_own_logits']['bound'], .1)

    def test_the_misplacement_guard(self):
        frame, d_results, data, _ = self.shifted(1e-2)   # a loose value bound (2e-2), so only the guard decides
        noisy = self.noisy(frame, data, 5e-3)
        guarded = self.judged(noisy, d_results, data, spread_p10=1e-2)   # guard 2.5e-3 < 5e-3
        self.assertFalse(guarded['passed']);t2 = guarded['checks']['c']['quantities']['t2_value']
        self.assertTrue(t2['within_bound']);self.assertFalse(t2['within_guard']);self.assertAlmostEqual(t2['ratio_to_guard'], 2., delta=1e-2)
        self.assertEqual((guarded['tolerance']['guard_bound'], guarded['tolerance']['guard']), (2.5e-3, 'applied'))
        self.assertTrue(self.judged(noisy, d_results, data, spread_p10=.1)['passed'])   # guard 2.5e-2
        skipped = self.judged(noisy, d_results, data, spread_p10=None)
        self.assertTrue(skipped['passed']);self.assertIn('skipped', skipped['tolerance']['guard']);self.assertNotIn('within_guard', skipped['checks']['c']['quantities']['t2_value'])
        self.assertNotIn('guard_bound', guarded['checks']['c']['quantities']['t1_own_logits'])   # the guard bounds values, not logits
        # The live gate takes the guard's spread from the same frames' values section.
        values = gate.values_section([(gate.value_samples(frame, self.requests, backups=FakeBackups), None)], oracle_checked=True)
        self.assertGreater(values['totals']['cell_a2_spread']['p10'], 0)

    def test_fallback_when_the_floor_cannot_be_measured(self):
        frame, d_results, data, _ = self.priced()
        result = self.judged(self.noisy(frame, data, 5e-5), d_results, data, drop_floor=True);rule = result['tolerance']
        self.assertIsNone(rule['depth1_noise_floor']);self.assertEqual(rule['value_bound'], 1e-4);self.assertIn('fallback 1e-4', rule['value_bound_source'])
        self.assertTrue(result['passed'])
        self.assertFalse(self.judged(self.noisy(frame, data, 2e-4), d_results, data, drop_floor=True)['passed'])
        for floor in (dict(compared=0, nonfinite=0, max_abs_difference=0.), dict(compared=5, nonfinite=1, max_abs_difference=.1)):
            self.assertEqual(gate.tolerance_rule(floor, None)['value_bound'], 1e-4)
        self.assertEqual(gate.tolerance_rule(dict(compared=5, nonfinite=0, max_abs_difference=0.), None)['value_bound'], 0.)   # a measured zero floor is strict

    def test_the_header_gives_the_rule_and_every_ratio(self):
        frame, d_results, data, _ = self.shifted(1e-3)
        result = self.judged(self.noisy(frame, data, 1.5e-3), d_results, data, spread_p10=.1)
        line = gate.tolerance_line(result['tolerance'])
        for key in ('depth1_noise_floor', 'value_bound', 'logit_bound', 'p10_cell_a2_spread', 'guard_bound'):self.assertIsNotNone(line[key], key)
        rows = {(r['check'], r['quantity']): r for r in line['checks']}
        self.assertIn(('b', 't1_value'), rows);self.assertIn(('c', 't2_value'), rows);self.assertIn(('c', 't1_own_logits'), rows)
        self.assertAlmostEqual(rows[('c', 't2_value')]['ratio_to_bound'], .75, delta=1e-2);self.assertAlmostEqual(rows[('c', 't2_value')]['ratio_to_guard'], .06, delta=1e-3)
        self.assertIn('2 x depth1_noise_floor', result['tolerance']['rule']);json.dumps(result, allow_nan=False, default=str)


class ReviewTests(GateFixture, unittest.TestCase):
    """The independent review's findings at 41e45122, each pinned."""
    def price_with(self, *, forward_=forward, widths=None, sample=None, chunk=32, pricer_chunk=16):
        widths = widths or self.WIDTHS;pricer = OraclePricer(self.ex, widths);pricer.oracle = reference.Oracle(self.ex, object(), chunk=pricer_chunk)
        frame_pricer = batch.FramePricer(self.ex, object(), capacity=16, block_rows=16, actor_capacity=3, vectorized=True, host_prep=True)
        with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=forward_):
            return gate.price_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS, pricer=pricer,
                frame_pricer=frame_pricer, oracle=reference.Oracle(self.ex, object(), chunk=chunk), widths=widths, sample=sample, block=0)

    def test_e_copies_only_the_lanes_c_checks(self):
        # Mode 15: the opponent waits at every child and its unused child view is invalid, so its raw prior is NaN.
        frame, d_results, data, _ = self.price_with(forward_=nan_on_invalid_forward)
        waiting = [i for i, c in enumerate(data['children']['cells'].tolist()) if c[0] == 11 and data['children']['status'][i] == 2]
        self.assertTrue(waiting);self.assertTrue(np.isnan(data['children']['peer_logits'][waiting]).all())
        swapped, counts = gate.oracle_heads_frame(frame, data)
        cells = tuple(data['children']['cells'][waiting].T)
        np.testing.assert_array_equal(swapped.t1_peer_logits[cells], frame['t1_peer_logits'][cells])   # the pricer's lanes stay
        result = gate.decision_check(frame, data, self.requests, backups=FakeBackups)
        self.assertTrue(result['passed'], result['first_differing_roots']);self.assertEqual(result['nonfinite_roots'], 0)
        self.assertGreater(counts['root_logits'], 0)

    def test_e_swaps_the_root_opponent_prior(self):
        frame, _, data, _ = self.priced();frame = copy.deepcopy(frame)
        legal = np.asarray(data['roots'][(0, 0)]['peer_legal'], bool);frame['root_peer_logits'][0, 0, legal] += np.float32(.5)
        swapped, _ = gate.oracle_heads_frame(frame, data)
        np.testing.assert_array_equal(swapped.root_peer_logits[0, 0, legal], np.asarray(data['roots'][(0, 0)]['peer_logits_raw'], np.float32)[legal])
        np.testing.assert_array_equal(swapped.root_peer_logits[0, 0, ~legal], frame['root_peer_logits'][0, 0, ~legal])

    def test_vacuous_runs_are_incomplete_not_pass(self):
        frame, d_results, data, _ = self.priced()
        clean = gate.summarize([gate.evaluate(frame, d_results, self.requests, data)])
        self.assertEqual(gate.gate_status(clean)[0], 'PASS');self.assertEqual(gate.gate_status(clean, sampled=True)[0], 'PASS_SAMPLED')
        refused = copy.deepcopy(frame);refused['pre'] = np.where(frame['pre'] < 0, frame['pre'], 1).astype(np.int32)   # world (6,1) has no state
        empty = dict(roots={k: dict(v, pre=1) for k, v in data['roots'].items()}, children=dict(cells=np.zeros((0, 4), np.int64)),
            leaves=dict(specs=np.zeros((0, 6), np.int64)), enumerated=[])
        for name in ('t1_status', 't2_status'):refused[name][:] = 0
        refused_d = [SimpleNamespace(status='REFUSED', world_reports=tuple(dict(joint_values=None, reason=w['reason']
            if (w.get('reason') or '').startswith('physical-materialization') else 'ValueError: current direct-encoder input error')
            for w in result.world_reports)) for result in d_results]
        result = gate.summarize([dict(gate.evaluate(refused, refused_d, self.requests, empty), e=gate.decision_check(refused, empty, self.requests, backups=FakeBackups))])
        status, reasons = gate.gate_status(result)
        self.assertEqual(status, 'INCOMPLETE', result['checks']);self.assertTrue(any('no turn-2 leaf' in r for r in reasons))
        self.assertTrue(any('(e) compared no root' in r for r in reasons))

    def test_top_k_ranks_by_the_pricers_checked_priors(self):
        widths = dict(k_own=1, k_opp=9, k1_own=9, k1_opp=9)
        frame, d_results, data, _ = self.price_with(forward_=geometry_noise_forward, widths=widths, chunk=33, pricer_chunk=16)
        result = gate.summarize([gate.evaluate(frame, d_results, self.requests, data, full_turn1=True)])
        self.assertTrue(result['passed'], {k: v.get('first_mismatches') for k, v in result['checks'].items()})
        self.assertLess(result['checks']['c']['quantities']['t1_own_logits']['max_abs_difference'], gate.LOGIT_BOUND)
        topk = gate.merge_topk_boundary([data['topk']])
        self.assertEqual(topk['ranked_by'], ['PRICER_PRIORS']);self.assertGreater(topk['oracle_prior_set_differs']['own'], 0)
        self.assertGreater(topk['gap_below_logit_bound']['own'], 0);self.assertGreater(topk['selections']['own'], 0)
        self.assertEqual(topk['selections']['peer'], 0)
        self.assertIsNone(gate.topk_boundary(data['children'], data['roots'], self.WIDTHS, None, None, reference))

    def test_leaf_counts_report_the_forced_leaves(self):
        calls = [dict(turn2_leaves=100, forced_leaves=7, checked_leaves=19), dict(turn2_leaves=50, forced_leaves=0, checked_leaves=4)]
        counts = gate.leaf_counts(calls, 2)
        self.assertEqual((counts['sample_leaves'], counts['enumerated'], counts['forced_terminal_or_error'], counts['checked']), (2, 150, 7, 23))
        self.assertIn('every enumerated leaf', gate.leaf_counts(calls, None)['scope'])

    def test_samples_carry_every_pricer_terminal_and_error_leaf(self):
        frame, _, data, _ = self.price_with(sample=1)
        checked = set(map(tuple, data['leaves']['specs'].tolist()));t2 = frame['t2_status']
        reported = set(map(tuple, np.argwhere((t2 == 1) | (t2 == 3)).tolist()))
        self.assertTrue(reported);self.assertLessEqual(reported, checked);self.assertEqual(data['forced_leaves'], len(reported))

    def test_sampled_recheck_skips_e_and_passes_sampled(self):
        frame_pricer = batch.FramePricer(self.ex, object(), capacity=16, block_rows=16, actor_capacity=3, vectorized=True, host_prep=True)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=forward):
                gate.run_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS, pricer=OraclePricer(self.ex, self.WIDTHS),
                    frame_pricer=frame_pricer, oracle=reference.Oracle(self.ex, object(), chunk=32), widths=self.WIDTHS, sample=2, block=0,
                    context=dict(block=0, call=0), save=Path(directory) / 'arrays-block0-call0.npz')
            report = gate.recheck(directory, backups=FakeBackups)
        self.assertEqual(report['status'], 'PASS_SAMPLED', report['status_reasons']);self.assertNotIn('e', report['checks'])
        self.assertIn('SKIPPED_SAMPLED', report['skipped']['e']);self.assertNotIn('e', report['unavailable'])

    def test_refusals_are_exact_under_k1_pruning(self):
        frame = blank_frame(2);legal = (True, True, True) + (False,) * 6
        for q in (0, 1):
            for a1 in (0, 1):frame['t1_status'][q, :, a1, 0] = 1   # root action 2 is never enumerated (k1_own = 2)
        self.assertEqual(gate.depth2_refused(frame).tolist(), [False, False])
        self.assertEqual(gate.depth2_refused(frame, [legal, (True, True) + (False,) * 7]).tolist(), [True, False])
        seen = {}
        def refusal_summary(view, root_legal=None):
            seen['root_legal'] = root_legal;return dict(roots=(), root_count=2, refused_count=1)
        with patch.dict(sys.modules, {'gpu_engine.depth2_backups': SimpleNamespace(refusal_summary=refusal_summary)}):
            report = gate.refusal_report(frame, root_legal=[legal, legal])
        self.assertEqual(seen['root_legal'].tolist(), [list(legal)] * 2);self.assertEqual(report['depth2_gate_count']['refused_count'], 2)
        with patch.dict(sys.modules, {'gpu_engine.depth2_backups': SimpleNamespace(refusal_summary=lambda view: dict(roots=(), root_count=2, refused_count=2))}):
            self.assertEqual(gate.refusal_report(frame, root_legal=[legal, legal])['depth2']['refused_count'], 2)   # an older signature still works

    def test_b_flags_a_depth2_code_where_d_errored(self):
        frame, _, data, _ = self.priced();frame = copy.deepcopy(frame)
        error = tuple(data['children']['cells'][list(data['children']['error']).index(1)])   # mode 3: D's invalid action
        frame['t1_error'][error] = 5
        b = self.evaluate(frame)['checks']['b']
        self.assertFalse(b['passed']);self.assertGreater(b['mismatched']['t1_error_vs_d'], 0)

    def test_topk_error_share_over_two_array_sets(self):
        legal = (True, True, True) + (False,) * 6;full, topk = blank_frame(3), blank_frame(3)
        for f in (full, topk):f['t1_status'][:, 0, :3, 0] = 2
        full['t2_status'][0, 0, 0, 0, 2, 0] = 3                               # root 0: its only error is pruned at top-k
        full['t2_status'][1, 0, 1, 0, 0, 0] = 3;topk['t2_status'][1, 0, 1, 0, 0, 0] = 3   # root 1: the error survives top-k
        topk['pre'][2] = 0                                                  # root 2: no errors at all
        requests = [plain_request(4 + q, legal) for q in range(3)]
        for q, r in enumerate(requests):r.plans.root.address.stream = 100 + q
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            gate.save_arrays(Path(a) / 'arrays-block0-call0.npz', full, requests, meta=dict(widths=dict(k_own=9, k_opp=9, k1_own=9, k1_opp=9)))
            gate.save_arrays(Path(b) / 'arrays-block0-call0.npz', topk, requests, meta=dict(widths=dict(k_own=2, k_opp=9, k1_own=9, k1_opp=9)))
            report = gate.topk_error_share(a, b)
            with self.assertRaises(SystemExit):gate.topk_error_share(b, a)   # the first set must be full width
        self.assertEqual(report['counts'], dict(matched=3, with_full_width_errors=2, only_pruned_errors=1, errors_kept_by_topk=1, no_full_width_errors=1,
            topk_errors_not_at_full_width=0))
        self.assertEqual((report['share_of_roots_with_full_width_errors'], report['share_of_matched_roots']), (.5, 1 / 3))
        self.assertEqual(report['examples'][0]['address'], [100, 4, 0])

    def test_groups_split_at_the_smallest_pricer_capacity(self):
        self.assertEqual(gate.call_capacity(SimpleNamespace(capacity=4096), [SimpleNamespace(capacity=2048), SimpleNamespace()]), 2048)
        self.assertEqual([len(g) for g in gate.split_groups([list(range(5)), list(range(2))], 2)], [2, 2, 1, 2])


def gate_v5_model():
    from gpu_engine import v5_model
    return v5_model


class PricerBindingTests(unittest.TestCase):
    """The pricer's constructor is not on this branch: bind it by signature, and fail closed."""
    WIDTHS = dict(k_own=9, k_opp=9, k1_own=9, k1_opp=9)

    def make(self, cls, child_block=None, **widths):
        module = SimpleNamespace(Depth2Pricer=cls)
        with patch.dict(sys.modules, {'gpu_engine.device_depth2_pricer': module}):
            return gate.make_pricer('executor', 'model', widths=dict(self.WIDTHS, **widths), actor_capacity=33, child_block=child_block)

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
        with self.assertRaises(SystemExit):self.make(FullTurn1, child_block=27)
        class Blocked:
            def __init__(self, executor, model, *, digest=False, child_block=81):self.child_block = child_block
        pricer, options = self.make(Blocked, child_block=27)
        self.assertEqual((pricer.child_block, options), (27, dict(digest=True, child_block=27)))
        self.assertEqual(self.make(Blocked)[0].child_block, 81)

    def test_cli_validates_arguments_and_fails_before_loading_without_the_pricer(self):
        out = Path(tempfile.gettempdir()) / f'depth2-gate-never-created-{uuid.uuid4().hex}'
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (['--out', str(out)], ['--out', str(out), '--saved-roots', 'a', '--d3-root', 'b', '--chunk', '0'],
                    ['--out', str(out), '--saved-roots', 'a'], ['--out', str(out), '--d3-root', 'b'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--k-own', '10'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--block-invariance', '--child-block', '27'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--child-block', '0'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--pricer-only', '--block-invariance'],
                    ['--out', str(out), '--recheck', 'arrays', '--d3-root', 'b'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--head-archive', 'a.npz'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--raw-head-check'],
                    ['--out', str(out), '--recheck', 'arrays', '--head-archive', 'a', '--head-archive-sha256', 'b', '--head-checkpoint-sha256', 'c'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--head-model-sha256', 'x'],
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--sample-leaves', '0'],
                    ['--out', str(out), '--topk-errors', 'full', 'topk', '--d3-root', 'b']):
                with self.assertRaises(SystemExit) as raised:gate.main(argv)
                self.assertEqual(raised.exception.code, 2, argv)
        with patch.object(importlib.util, 'find_spec', return_value=None), self.assertRaises(SystemExit) as raised:
            gate.main(['--out', str(out), '--d3-root', 'b', '--chunk', '0'])
        self.assertIn('device_depth2_pricer', str(raised.exception.code));self.assertFalse(out.exists())
        missing = lambda name: None if name == 'gpu_engine.depth2_backups' else object()
        with patch.object(importlib.util, 'find_spec', side_effect=missing), self.assertRaises(SystemExit) as raised:
            gate.main(['--out', str(out), '--d3-root', 'b', '--chunk', '0'])
        self.assertIn('depth2_backups', str(raised.exception.code));self.assertFalse(out.exists())


if __name__ == '__main__':unittest.main()
