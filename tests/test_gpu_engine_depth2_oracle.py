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
                'd_refused_world_unenumerated', 't1_enumeration_vs_d', 't1_value_vs_d_beyond_tolerance', 'root_peer_logits_vs_d_beyond_tolerance'):
            self.assertGreater(checks['b']['compared'][name], 0, name)
        for name in ('t1_value', 'root_peer_logits'):
            self.assertGreater(checks['b']['values'][name]['compared'], 0);self.assertEqual(checks['b']['values'][name]['exact_equal_fraction'], 1.)
        self.assertIn('not bitwise', checks['b']['reference'])
        self.assertEqual(checks['c']['tolerance'], .01);self.assertIn('device_d_frame_parity.py:94-95', checks['c']['tolerance_source'])
        floor = checks['c']['quantities']['depth1_noise_floor']
        self.assertGreater(floor['compared'], 0);self.assertEqual(floor['max_abs_difference'], 0.)   # the fake heads are exact
        self.assertIsNone(checks['c']['t2_value_to_floor_ratio']);self.assertEqual(checks['c']['gated'], ['t2_value', 't1_own_logits'])
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
            ('t1_value_vs_d_beyond_tolerance', bump('t1_value', expanded, 2e-5), 'b'),
            ('root_peer_logits_vs_d_beyond_tolerance', bump('root_peer_logits', (0, 0, 3), 2e-5), 'b'),
            ('t1_own_logits', bump('t1_own_logits', expanded + (1,), .02), 'c'),
            ('t2_value', bump('t2_value', head, .02), 'c'), ('t2_value', setter('t2_value', head, np.nan), 'c'),
            ('t2_extra_in_pricer', setter('t2_status', unexpanded, 4), 'd'), ('t2_missing_from_pricer', setter('t2_status', head, 0), 'd'),
            ('t1_extra_in_pricer', setter('t1_status', (0, 0, 8, 8), 2), 'd'), ('t1_enumeration_vs_d', setter('t1_status', (0, 0, 8, 8), 2), 'b'),
            ('t1_extra_in_pricer', setter('t1_status', (6, 1, 0, 0), 2), 'd'), ('d_refused_world_unenumerated', setter('t1_status', (6, 1, 0, 0), 2), 'b'),
            ('t1_error', setter('t1_error', code5, 6), 'a'), ('t1_error', setter('t1_error', code6, 0), 'a'),
            ('t1_depth2_code_status', setter('t1_status', code5, 2), 'b'), ('t1_status', setter('t1_status', code6, 2), 'a'),
            ('t1_value_vs_d_beyond_tolerance', bump('t1_value', code6, 2e-5), 'b'), ('t1_value_vs_d_beyond_tolerance', bump('t1_value', code5, 2e-5), 'b'),
            ('t1_digest_peer', flip('t1_digest_peer', code5 + (1,)), 'a'),
            ('pre', setter('pre', (6, 1), 0), 'a'), ('pre_vs_d', setter('pre', (6, 1), 0), 'b'),
        )
        for name, change, check in cases:
            with self.subTest(name):
                result = self.evaluate(self.mutated(change));found = result['checks'][check]
                self.assertFalse(result['passed']);self.assertFalse(found['passed'])
                if check == 'c':self.assertFalse(found['quantities'][name]['within_tolerance'])
                else:
                    self.assertGreater(found['mismatched'][name], 0, found['mismatched'])
                    self.assertIn(name, {m['field'] for m in found['first_mismatches']})
                    self.assertTrue(all(m['block'] == 0 for m in found['first_mismatches']))

    def test_float_differences_within_tolerance_pass_and_are_reported(self):
        _, _, data, _ = self.priced();leaves, children = data['leaves'], data['children']
        heads = [tuple(s) for s, x in zip(leaves['specs'], leaves['status']) if x == 4]
        valued = [tuple(c) for c, x in zip(children['cells'], children['status']) if x == 2]
        def change(f, _):
            for at in heads[:3]:f['t2_value'][at] = np.float32(f['t2_value'][at] + 3e-3)   # bf16 cross-geometry noise
            f['t1_own_logits'][valued[0] + (1,)] = np.float32(f['t1_own_logits'][valued[0] + (1,)] + 5e-3)
            f['t1_value'][valued[1]] = np.float32(f['t1_value'][valued[1]] + 3e-6)            # within (b)'s 1e-5
            f['root_peer_logits'][0, 0, 2] = np.nextafter(f['root_peer_logits'][0, 0, 2], np.float32(2))
        result = self.evaluate(self.mutated(change));checks = result['checks'];c = checks['c']['quantities']
        self.assertTrue(result['passed'], {k: v.get('first_mismatches') for k, v in checks.items()})
        self.assertEqual(c['t2_value']['compared'] - c['t2_value']['exact_equal'], 3)
        self.assertAlmostEqual(c['t2_value']['max_abs_difference'], 3e-3, delta=1e-6);self.assertLess(c['t1_own_logits']['exact_equal_fraction'], 1)
        values = checks['b']['values']['t1_value']
        self.assertEqual(values['compared'] - values['exact_equal'], 1);self.assertAlmostEqual(values['max_abs_difference'], 3e-6, delta=1e-7)
        self.assertEqual(values['first_inexact'][0]['cell'], list(valued[1]));self.assertEqual(values['first_beyond'], [])
        self.assertEqual(checks['b']['values']['root_peer_logits']['compared'] - checks['b']['values']['root_peer_logits']['exact_equal'], 1)
        # The floor is depth 1's own distance to the oracle heads: shifting D's view of t1 moves it, and the ratio follows.
        frame, d_results, clean, _ = self.priced()
        shifted = copy.deepcopy(d_results);report = shifted[0].world_reports[0];(a, r) = valued[0][2:]
        report['joint_values'][a][r] = float(np.float32(report['joint_values'][a][r] + 1e-3))
        noisy = self.mutated(change)
        noisy_result = gate.summarize([gate.evaluate(noisy, shifted, self.requests, clean)])['checks']
        self.assertAlmostEqual(noisy_result['c']['depth1_noise_floor'], 1e-3, delta=1e-6)
        self.assertAlmostEqual(noisy_result['c']['t2_value_to_floor_ratio'], 3., delta=1e-2)
        self.assertFalse(noisy_result['b']['passed'])   # and that 1e-3 is beyond (b)'s 1e-5 against D

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
        def changed(change):
            other = copy.deepcopy(frame);change(other);return gate.block_invariance(frame, other, context=dict(block=0, call=0))
        def ulp(f):f['t2_value'][head] = np.nextafter(f['t2_value'][head], np.float32(2))
        def payload(f):f['t2_value'][(0, 0, 8, 8, 8, 8)] = np.uint32(0x7fc00001).view(np.float32)   # NaN == NaN numerically
        def dtype(f):f['t1_status'] = f['t1_status'].astype(np.int32)
        def missing(f):f.pop('t2_digest')
        result = changed(ulp)
        self.assertFalse(result['passed']);self.assertEqual(result['differing'], {'t2_value': dict(count=1, size=frame['t2_value'].size)})
        self.assertEqual(result['first_differences'][0]['cell'], list(head));self.assertEqual(result['first_differences'][0]['block'], 0)
        self.assertEqual(changed(payload)['differing']['t2_value']['count'], 1)
        self.assertIn('dtype/shape', changed(dtype)['differing']['t1_status']['problem'])
        self.assertIn('missing', changed(missing)['differing']['t2_digest']['problem'])
        reports = dict(frame, world_reports=[[dict(seed=1)]]), dict(frame, world_reports=[[dict(seed=2)]])
        result = gate.block_invariance(*reports);self.assertTrue(result['passed']);self.assertEqual(result['nonarray_equal'], dict(world_reports=False))
        merged = gate.merge_invariance([changed(ulp), same]);self.assertFalse(merged['passed']);self.assertEqual(merged['differing']['t2_value']['count'], 1)

    def test_run_call_with_block_invariance_gates_the_frames(self):
        class Nudged(OraclePricer):
            def price(self, *args):
                frame = super().price(*args);at = tuple(np.argwhere(frame['t2_status'] == 4)[0])
                frame['t2_value'][at] = np.nextafter(frame['t2_value'][at], np.float32(2));return frame
        for other, expected in ((OraclePricer(self.ex, self.WIDTHS), True), (Nudged(self.ex, self.WIDTHS), False)):
            frame_pricer = batch.FramePricer(self.ex, object(), capacity=16, block_rows=16, actor_capacity=3, vectorized=True, host_prep=True)
            with patch.object(mapping, 'materialize_host', side_effect=self.materialize), patch.object(batch.v5_model, 'forward', side_effect=forward):
                evaluation, summary = gate.run_call(requests=self.requests, recipe=self.recipe, materializer=self.tables, params=PARAMS,
                    pricer=OraclePricer(self.ex, self.WIDTHS), frame_pricer=frame_pricer, oracle=reference.Oracle(self.ex, object(), chunk=32),
                    widths=self.WIDTHS, sample=0, block=0, context=dict(block=0, call=0), invariance=[other])
            result = gate.summarize([evaluation])
            self.assertEqual(summary['passed']['block_invariance'], expected);self.assertEqual(result['checks']['block_invariance']['passed'], expected)
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
            'check_a', 'check_b', 'check_c', 'check_d', 'check_oracle_consistency', 'check_block_invariance', 'refusals', 'check_e_decision')
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
        logits = np.where(opp, frame.t1_peer_logits[q, w, a1, r1], -np.inf);prior = np.exp(logits - logits.max());prior /= prior.sum()
        rows = [float(np.sum(prior[opp] * frame.t2_value[q, w, a1, r1, a2][opp])) if own[a2] else None for a2 in range(9)]
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
        np.testing.assert_array_equal(swapped.t1_own_logits[tuple(children['cells'][expanded].T)], children['own_logits'][expanded])
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
                    children=dict(cells=saved['oracle_child_cells'], status=saved['oracle_child_status'], own_logits=saved['oracle_child_own_logits'],
                        peer_logits=saved['oracle_child_peer_logits']))
                self.assertTrue(gate.decision_check(fields, data, self.requests, backups=FakeBackups)['passed'])


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
                    ['--out', str(out), '--d3-root', 'b', '--chunk', '0', '--child-block', '0']):
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
