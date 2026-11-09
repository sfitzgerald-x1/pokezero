"""CPU tests of scripts/depthn_rr_reprice.py: no GPU, no pricer. The live search object is the real DepthNLiveSearch (its own
process_many, _batch and _search) over a fake constructor and fake pricing (build, price and the backups patched), so the
wrappers are exercised at the live call sites; decide and prepare_roots are the real functions underneath."""
from collections import OrderedDict
from contextlib import ExitStack, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from typing import NamedTuple
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from scripts import depthn_rr_reprice as R
from tests import test_depthn_live_bench as BT   # its own-request fakes: one or nine legal actions
from gpu_engine import constructor_worlds as cw, depthn_backups as backups, depthn_live_search as live
from gpu_engine.device_public_world_state import World
from gpu_engine.phase_d_live_worlds import Refusal
from gpu_engine.state import empty_state
from pokezero import phase_d_owning_capture as owning, v5_gumbel
from pokezero.phase_d_search_server import SCHEMA

ORIGINAL_DECIDE, ORIGINAL_PREPARE = live.decide, cw.prepare_roots
VOCABULARY = dict(species=['', 'Pikachu', 'Mr. Mime', 'Snorlax', "Farfetch’d", 'Ho-Oh', 'Gengar'],
    moves=['', 'Thunderbolt', 'Hidden Power Fire', 'hiddenpowerwater70', 'Body Slam', 'Solar Beam', 'Hidden Power', 'Return 102'],
    abilities=['', 'Static', 'Pressure', 'Thick Fat', 'Trace'], items=['', 'Leftovers', 'Choice Band'], types=['', 'Fire', 'Water', 'Ice'])
LEGAL = (True, True, False, True, True, True, False, False, False)   # the root's own legal lanes
LOGITS = (.5, -.2, 0., 1.1, -.7, .3, 0., 0., 0.)
KEPT = (True, False, True, False, True, False, False, True, False)   # the opponent's lanes turn 1 enumerates


def prices(w, kept=KEPT):
    """World w's WorldPrices: values with long binary expansions, None off the legal x kept product."""
    grid = [[(((a * 9 + b) * 7 + w * 3) % 23) / 23 - .5 + 1e-3 / 3 if LEGAL[a] and kept[b] else None for b in range(9)] for a in range(9)]
    return v5_gumbel.WorldPrices(any(kept), kept, tuple(.1 * b - .3 + w / 300 for b in range(9)), grid)


def battle_state(*, active=2, charge=0):
    """A real BattleState: the opponent (p2) has species 1..6, its active `active` knows Thunderbolt, two Hidden Powers and an
    empty slot; slots 1 and 3 fainted."""
    s = empty_state(1)
    moves = s.moves.at[0, 1, active].set(jnp.asarray([1, 2, 3, 0]))
    return s._replace(species=s.species.at[0, 1].set(jnp.arange(1, 7)), moves=moves, pp=s.pp.at[0, 1, active].set(jnp.asarray([5, 5, 5, 0])),
        hp=s.hp.at[0, 1].set(jnp.asarray([100, 0, 50, 0, 1, 2])), max_hp=s.max_hp.at[0, 1].set(jnp.full(6, 100)),
        active=s.active.at[0, 1].set(active), charge_move=s.charge_move.at[0, 1, active].set(charge))


STATE = battle_state()


def world(i, state=STATE):
    return World(i, state, dict(w=jnp.asarray(i, jnp.int32)), (), 1000 + i, False)


class FakeConstructor:
    count = 16

    def __init__(self, refuse=()):
        self.sessions, self.seen, self.metadata, self.refuse, self.calls = {}, set(), dict(vocabulary=VOCABULARY), set(refuse), []

    def prepare(self, request, index):
        self.calls.append(request['decision_index'])
        if request['decision_index'] in self.refuse:raise Refusal('TEST_REFUSAL', 'no worlds')
        made = SimpleNamespace(worlds=tuple(world(i) for i in range(16)), seeds=tuple(500 + i for i in range(16)), requested_worlds=16,
            fully_constructed=True, mapper_diagnostics=())
        return SimpleNamespace(address=SimpleNamespace(stream=1, frame=request['decision_index'], owner=0)), made, LEGAL


def fake_build(constructor, prepared, pricer, params, *, binding, recipe, only_complete):
    roots, requests = [], []
    for item in prepared:
        if type(item) is cw.RootRefusal:
            roots.append(dict(request_id=item.request_id, status='REFUSED', reason='constructor-root:' + item.kind));continue
        rows = tuple(range(len(requests), len(requests) + 8));requests += rows
        roots.append(dict(request_id=item.request_id, status='PRICED', reason=None, rows=rows, root_logits=LOGITS, root_legal=item.legal,
            root_view_source='b2', b2_view=dict(equal=True, keys=dict(numeric_features=True), numeric_max_abs_diff=0.)))
    return SimpleNamespace(requests=tuple(requests), roots=tuple(roots))


def pricing(stack, *, still_clock=False):
    """Fake pricing under the real _search: build, price and the backups (each row r is the block of worlds 2r, 2r+1). The fake
    worlds' carries are not the device tracker carry: the real prepare_roots keeps them as built (no carry completion)."""
    stack.enter_context(patch.object(cw, 'constructor_binding', return_value='binding'))
    stack.enter_context(patch.object(cw, 'world_completer', return_value=lambda world, owner, **kw: world))
    stack.enter_context(patch.object(live, 'classify', return_value=([], 'protocol_facts')))
    stack.enter_context(patch.object(cw, 'build', side_effect=fake_build))
    stack.enter_context(patch.object(cw, 'price', return_value=(SimpleNamespace(), None)))
    stack.enter_context(patch.object(backups, 'refusal_summary', return_value=dict(roots=[dict(cause=None)] * 64)))
    stack.enter_context(patch.object(backups, 'backup', side_effect=lambda frame, r, rule: (prices(2 * r), prices(2 * r + 1))))
    if still_clock:   # every stage timing 0, so whole responses compare
        for module in (live, cw):stack.enter_context(patch.object(module, 'time', SimpleNamespace(perf_counter=lambda: 0.)))


def fake_search(constructor):
    """The real DepthNLiveSearch, its attributes set directly (no pricer or champion), warmed (a carry reference)."""
    s = object.__new__(live.DepthNLiveSearch)
    s.W, s.K, s.rung, s.buckets, s.params = 16, 8, 'DN2W16', (1,), (jnp.float32(1.),)
    s.pricers = {1: SimpleNamespace(executor=PeerExecutor(), forward=FakeEvaluator())}   # the views' executor and the champion: fakes
    s.binding, s.recipe, s.tree_commit, s.min_price_seconds, s.max_requests, s.session_cap = 'binding', 'recipe', 'replay', 1., 1, 256
    s.archive_sha256, s.device, s.clock, s.battles, s.evicted, s.warm, s.refuse_classes = 'a' * 64, 'cpu', lambda: 0., OrderedDict(), 0, None, ()
    s.warming, s.pricer_identity, s.constructor = False, dict(engine='fake'), constructor
    s.carry = dict(structure=live.carry_structure(world(0)), signature='sig', provisional=False, root={})
    s.warmup = lambda requests: dict(buckets={1: dict(cold_seconds=1., warm_seconds=.5)}, programs=3, seen=len(requests))
    return s


def body(i, single=False):
    return dict(BT.live_request(i, single), schema=SCHEMA, request_id=f'r{i}', rung='DN2W16', budget_seconds=600.)


def fake_peer(state, public, owner):
    """The opponent's lanes: KEPT, except world 5, whose last lane differs (a kept/legal disagreement)."""
    legal = list(KEPT)
    if int(np.asarray(public['w'])) == 5:legal[8] = True
    return dict(legal=legal, move_targets=[0, 1, 2, 3], switch_targets=[0, 1, 3, 4, 5], error=0, usable=[True, True, True, False])


class Out(NamedTuple):
    policy_logits: object
    value: object
    value_logits: object
    valid: object


class FakeEvaluator:
    """A ChampionEvaluator stand-in, jittable: policy logits = the params x token 0's first nine numerics of the inputs it is
    handed (exact for params 1); its view is ChampionEvaluator.view itself (the adapter's 23-token slice)."""
    identity = ('champion', 'e' * 64, 'bfloat16')

    def __call__(self, p, x):
        logits = x['numeric_features'][:, 0, 0, :9] * p;n = logits.shape[0]
        return Out(logits, jnp.zeros(n, jnp.float32), None, jnp.ones(n, bool))

    def view(self, inputs):
        from gpu_engine.champion_evaluator import ChampionEvaluator
        return ChampionEvaluator.view(self, inputs)


class PeerExecutor:
    """The views' executor: the peer seat's (seat 1's) inputs carry world w's recorded root_peer_logits (prices(w), as f32) in
    token 0's first nine numerics, the owner's seat 100 more (so the peer inputs are told apart); worlds in `off` are one f32
    ulp off on lane 3."""
    def __init__(self, off=()):
        self.off = off

    def legal(self, state):return jnp.asarray([[[True] * 9, list(KEPT)]])   # the peer seat's (seat 1's) mask is KEPT

    def observe(self, state, public, seat, mask):
        w, s = int(np.asarray(public['w'])) if 'w' in public else 0, int(np.asarray(seat)[0])
        values = np.asarray(prices(w).opponent_logits, np.float32) + np.float32(100 if s == 0 else 0)
        if w in self.off and s == 1:values[3] = np.nextafter(values[3], np.float32(9))
        numeric = np.zeros((1, 1, 29, 132), np.float32);numeric[0, 0, 0, :9] = values
        return SimpleNamespace(observation=SimpleNamespace(legal_action_mask=mask), inputs=dict(numeric_features=numeric, categorical_ids=np.zeros((1, 1, 29, 41), np.int32),
            token_type_ids=np.zeros((1, 1, 29), np.int32), attention_mask=np.ones((1, 1, 29), bool), history_mask=np.ones((1, 1), bool)))


def expected_decision():
    return ORIGINAL_DECIDE([(prices(2 * k), prices(2 * k + 1)) for k in range(8)], LOGITS, LEGAL)


class CaptureTests(unittest.TestCase):
    def test_call_sites_resolve_through_the_modules_at_call_time(self):
        self.assertTrue(R.check_call_sites())

        def bound(self, todo):   # a search holding its own reference to decide would escape the wrapper
            decide = live.decide
            return decide(None, None, None), cw.prepare_roots(None, todo)
        with self.assertRaises(RuntimeError):R.check_call_sites(bound)

    def test_wrappers_capture_restore_and_leave_responses_identical(self):
        requests = [body(0), body(1), body(2)]
        with ExitStack() as stack:
            pricing(stack, still_clock=True)
            plain = fake_search(FakeConstructor(refuse={1}))
            without = [plain.process_many([r])[0] for r in requests]
            wrapped, captures = fake_search(FakeConstructor(refuse={1})), []
            within = []
            for r in requests:
                capture = R.Capture()
                with capture.installed():
                    self.assertIsNot(live.decide, ORIGINAL_DECIDE);self.assertIsNot(cw.prepare_roots, ORIGINAL_PREPARE)
                    within.append(wrapped.process_many([r])[0])
                captures.append(capture)
                self.assertIs(live.decide, ORIGINAL_DECIDE);self.assertIs(cw.prepare_roots, ORIGINAL_PREPARE)
        self.assertEqual([r['status'] for r in without], ['COMPLETE', 'REFUSED', 'COMPLETE'])
        self.assertEqual(without[1]['cause'], 'constructor_refused')
        self.assertEqual(within, without)
        self.assertEqual(json.dumps(within, sort_keys=True), json.dumps(without, sort_keys=True))   # byte for byte, timings included
        self.assertEqual(without[0]['q_values'], expected_decision()['q_values'])
        self.assertEqual([len(c.decided) for c in captures], [1, 0, 1]);self.assertEqual([len(c.prepared) for c in captures], [1, 1, 1])
        got = captures[0].decided[0]
        self.assertEqual([w for pair in got.blocks for w in pair], [prices(w) for w in range(16)])   # block order: world 2k+j
        self.assertEqual((tuple(got.root_logits), tuple(got.root_legal)), (LOGITS, LEGAL))
        self.assertIs(type(captures[1].prepared[0][0]), cw.RootRefusal)

    def test_wrappers_restore_when_the_call_raises(self):
        capture = R.Capture()
        with self.assertRaises(KeyError):
            with capture.installed():raise KeyError('boom')
        self.assertIs(live.decide, ORIGINAL_DECIDE);self.assertIs(cw.prepare_roots, ORIGINAL_PREPARE)

    def test_prices_encode_none_as_null_and_floats_exactly(self):
        p = prices(3);fields = R.prices_fields(p);text = json.dumps(fields, allow_nan=False)
        back = json.loads(text)
        self.assertIn('null', text);self.assertEqual(back['grid'][2], [None] * 9)   # an illegal root lane
        self.assertEqual(back['grid'], [list(row) for row in p.joint_values])   # every float exact after the round trip
        self.assertEqual((back['kept'], back['root_peer_logits'], back['opponent_requested']), (list(KEPT), list(p.opponent_logits), True))
        self.assertEqual(R.prices_fields(None)['priced'], False)
        self.assertIn('NaN', R.encode_row(dict(x=float('nan'))));self.assertTrue(json.loads(R.encode_row(dict(x=float('nan'))))['nonfinite'])


class FactsTests(unittest.TestCase):
    def test_opponent_facts_through_the_constructor_tables(self):
        facts = R.opponent_facts(STATE, 0, VOCABULARY)
        self.assertEqual((facts['side'], facts['active_slot'], facts['active_species']), ('p2', 2, 'snorlax'))   # slot 2 holds species id 3
        self.assertEqual(facts['active_moves'], ['thunderbolt', 'hiddenpower', 'hiddenpower', None])   # Hidden Power collapsed; empty slot
        self.assertEqual(facts['active_move_engine_ids'], [1, 2, 3, 0]);self.assertIsNone(facts['charge_move'])
        self.assertEqual([m['species'] for m in facts['party']], ['pikachu', 'mrmime', 'snorlax', 'farfetchd', 'hooh', 'gengar'])
        self.assertEqual([m['fainted'] for m in facts['party']], [False, True, False, True, False, False])
        self.assertEqual(R.opponent_facts(battle_state(active=2, charge=5), 0, VOCABULARY)['charge_move'], 'solarbeam')
        self.assertEqual(R.opponent_facts(STATE, 1, VOCABULARY)['side'], 'p1')

    def test_opponent_party_sets_in_party_and_request_order(self):
        """opponent.party[]: each member's set in party (slot) order, `order` its position in the peer's request order (the
        completed carry's own_order[peer]); request_order lists the slots in that order."""
        s = STATE;slot = lambda field, values: getattr(s, field).at[0, 1].set(jnp.asarray(values, jnp.int32))
        s = s._replace(ability=slot('ability', [1, 2, 4, 0, 3, 3]), base_ability=slot('base_ability', [1, 2, 4, 0, 3, 2]),
            item=slot('item', [1, 0, 2, 0, 1, 9]), moves=s.moves.at[0, 1, 4].set(jnp.asarray([4, 5, 0, 0])).at[0, 1, 5].set(jnp.asarray([6, 7, 6, 0])),
            hidden_power_type=slot('hidden_power_type', [0, 0, 1, 0, 0, 3]), hidden_power_power=slot('hidden_power_power', [0, 0, 70, 0, 0, 70]))
        order = np.tile(np.arange(6, dtype=np.int32), (1, 2, 1));order[0, 1] = [2, 0, 1, 3, 4, 5]   # the peer's active (slot 2) leads
        facts = R.opponent_facts(s, 0, VOCABULARY, SimpleNamespace(own_order=order))
        party = facts['party'];self.assertEqual(facts['request_order'], [2, 0, 1, 3, 4, 5])
        self.assertEqual([m['order'] for m in party], [1, 2, 0, 3, 4, 5])
        self.assertEqual([m['ability'] for m in party], ['static', 'pressure', 'trace', None, 'thickfat', 'thickfat'])
        self.assertEqual(party[5]['base_ability'], 'pressure')   # the engine's original ability beside the current one
        self.assertEqual([m['item'] for m in party], ['leftovers', None, 'choiceband', None, 'leftovers', None])   # 9: outside the table
        # Hidden Power keeps its type, as the request spells it: the engine id's own typed name (power suffix kept), or a typeless
        # engine entry typed from the world's hidden_power_type / hidden_power_power; the active's facts stay collapsed.
        self.assertEqual((party[2]['moves'], party[4]['moves'], party[4]['move_engine_ids']),
            (['thunderbolt', 'hiddenpowerfire', 'hiddenpowerwater70', None], ['bodyslam', 'solarbeam', None, None], [4, 5, 0, 0]))
        self.assertEqual(party[5]['moves'], ['hiddenpowerice70', 'return', 'hiddenpowerice70', None])
        self.assertEqual(facts['active_moves'], ['thunderbolt', 'hiddenpower', 'hiddenpower', None])
        untyped = R.opponent_facts(s._replace(hidden_power_type=slot('hidden_power_type', [0] * 6)), 0, VOCABULARY)
        self.assertEqual(untyped['party'][5]['moves'][0], 'hiddenpower')   # no type in the world either: the typeless id
        self.assertEqual([(m['slot'], m['fainted'], m['hp'], m['max_hp']) for m in party][:2], [(0, False, 100, 100), (1, True, 0, 100)])
        bare = R.opponent_facts(s, 0, VOCABULARY, dict(w=0))   # a carry without own_order: no request order
        self.assertEqual((bare['request_order'], {m['order'] for m in bare['party']}), (None, {None}))
        self.assertEqual(R.opponent_facts(s, 0, VOCABULARY, SimpleNamespace(own_order=order))['party'][0]['species'], 'pikachu')

    def test_peer_lanes_take_the_views_second_output(self):
        class Executor:   # the pricer's views call legal() then observe() per seat; each seat's observation is told apart
            def legal(self, state):return jnp.asarray([[[True] * 9, [True, False, True, False, True, True, False, False, False]]])

            def observe(self, state, public, seat, mask):
                s = int(np.asarray(seat)[0])
                return SimpleNamespace(inputs={}, observation=SimpleNamespace(legal_action_mask=mask, move_targets=jnp.asarray([[s, 1, 2, 3]]),
                    switch_targets=jnp.asarray([[0, 1, 3, 4, 5 - s]]), error=jnp.asarray([7 * s])))
        lanes = R.PeerLanes(Executor())(STATE, {}, 0)
        self.assertEqual(lanes['legal'], [True, False, True, False, True, True, False, False, False])   # seat 1's engine mask
        self.assertEqual((lanes['move_targets'], lanes['switch_targets'], lanes['error']), ([1, 1, 2, 3], [0, 1, 3, 4, 4], 7))
        self.assertEqual(lanes['usable'], [True, True, True, False])   # usable_moves on the opponent's active: pp 0 in the empty slot

    def test_lane_labels_follow_the_encoder_lane_space(self):
        facts = R.opponent_facts(STATE, 0, VOCABULARY)
        peer = dict(legal=[True, True, False, False, True, False, True, False, True], move_targets=[0, 1, 2, 3], switch_targets=[0, 1, 3, 4, 5],
            usable=[True, True, True, False])
        self.assertEqual(R.lane_labels(facts, peer), ['move:thunderbolt', 'move:hiddenpower', None, None, 'switch:pikachu', None,
            'switch:farfetchd', None, 'switch:gengar'])
        self.assertEqual(R.lane_labels(facts, dict(peer, usable=[False] * 4))[:2], ['move:struggle', 'move:hiddenpower'])
        charged = R.opponent_facts(battle_state(charge=5), 0, VOCABULARY)
        self.assertEqual(R.lane_labels(charged, dict(peer, legal=[True] + [False] * 8))[0], 'move:solarbeam')


class OwnMovesTests(unittest.TestCase):
    def test_decision_rows_carry_the_own_moves_and_recharge_lanes_of_a_decided_root(self):
        """root.own_moves / root.own_recharge_lanes: the live actor's own_moves and recharge_lane on the request's own_request_line,
        so a scorer can apply the actor's own-recharge guard to every operator; every row has a root (decided false when it never
        reached decide), with opponent_recharge from the lines and the refusal cause."""
        raw = json.loads(BT.own_request(False)[len('|request|'):])
        raw['active'][0]['moves'] = [dict(id=m, move=m, pp=5, maxpp=5) for m in ('bodyslam', 'hyperbeam', 'earthquake')]
        line = '|request|' + json.dumps(raw)
        row = dict(battle_id='b', owner='p1', stream_id=1, decision_index=0, generation=1, prefix_sha256='0' * 64, turn=1, request_count=1)
        request = dict(row, lines=['|turn|1', line], own_request_line=line)
        response = dict(status='COMPLETE', action_index=1, q_values=[.1] * 9, cause=None, request_id='r')
        detail = dict(root=dict(root_logits=list(LOGITS), root_legal=list(LEGAL)), prepared=None, worlds=[], notes=[], kept_disagreements=0)
        found = dict(source='search', search_action_index=1, search_q_values=[.1] * 9, greedy_action_index=0)
        out = R.decision_row(dict(ordinal=1), row, request, response, .1, found, detail, shard=0, capture_seconds=0.)
        self.assertEqual((out['root']['own_moves'], out['root']['own_recharge_lanes']), (['bodyslam', 'hyperbeam', 'earthquake', None],
            [False, True, False, False]))
        self.assertEqual({k: out['root'][k] for k in ('decided', 'root_logits', 'opponent_recharge', 'refusal_cause')},
            dict(decided=True, root_logits=list(LOGITS), opponent_recharge=False, refusal_cause=None))
        with patch.object(R.actor, 'RECHARGE_MOVES', frozenset({'earthquake'})):   # the live guards' one set (depthn_effect_guard), read at call time
            moved = R.decision_row(dict(ordinal=1), row, request, response, .1, found, detail, shard=0, capture_seconds=0.)
        self.assertEqual(moved['root']['own_recharge_lanes'], [False, False, True, False])
        self.assertEqual(detail['root'], dict(root_logits=list(LOGITS), root_legal=list(LEGAL)))   # the capture is not mutated
        from pokezero.depthn_live_actor import recharge_guarded   # the scorer's guard on this row: the live search went INTO Hyper Beam
        self.assertTrue(recharge_guarded(out['root']['own_moves'], found['search_action_index'], found['greedy_action_index']))
        unread = R.decision_row(dict(ordinal=1), row, dict(request, own_request_line=None), response, .1, found, detail, shard=0, capture_seconds=0.)
        self.assertEqual((unread['root']['own_moves'], unread['root']['own_recharge_lanes']), (None, None))
        # The opponent's -mustrecharge on the lines; the replay refused the root in prepare_roots (a RootRefusal of that kind).
        beam = dict(request, lines=['|turn|1', '|move|p2a: Snorlax|Hyper Beam|p1a: Snorlax', '|-mustrecharge|p2a: Snorlax', line])
        refusal = dict(kind='RootRefusal', refusal_kind='OPPONENT_RECHARGE_UNREPRESENTED', refusal_detail='d', seconds=0.)
        refused = R.decision_row(dict(ordinal=1), row, beam, dict(response, status='REFUSED', cause='constructor_refused'), .1, found,
            dict(detail, root=None, prepared=refusal), shard=0, capture_seconds=0.)
        self.assertEqual({k: refused['root'][k] for k in ('decided', 'root_logits', 'root_legal', 'opponent_recharge', 'refusal_cause')},
            dict(decided=False, root_logits=None, root_legal=None, opponent_recharge=True, refusal_cause='OPPONENT_RECHARGE_UNREPRESENTED'))
        self.assertEqual(refused['root']['own_recharge_lanes'], [False, True, False, False]);self.assertTrue(refused['capture_ok'])
        other = R.decision_row(dict(ordinal=1), row, request, dict(response, status='REFUSED', cause='depthn_refused'), .1, found,
            dict(detail, root=None), shard=0, capture_seconds=0.)
        self.assertEqual((other['root']['refusal_cause'], other['root']['opponent_recharge'], other['peer_inputs_check']), ('depthn_refused', False, None))
        # private_reveals: the response's own record (constructor_worlds.private_reveal_record), None when the response has none
        record = dict(enabled=True, found=[dict(species='zapdos', ability='pressure')], applicable=[dict(species='zapdos', ability='pressure', slot=1)],
            redacted=[dict(species='zapdos', ability='pressure', slot=1)], unmatched=[])
        self.assertEqual(R.decision_row(dict(ordinal=1), row, request, dict(response, private_reveals=record), .1, found, detail, shard=0,
            capture_seconds=0.)['private_reveals'], record)
        self.assertIsNone(out['private_reveals'])


class JoinTests(unittest.TestCase):
    def test_q_equal_is_exact_after_a_json_round_trip(self):
        q = [.1 + .2, None, -1 / 3, 1., None, None, None, None, None]
        live_q = json.loads(json.dumps(q))
        self.assertTrue(R.q_equal(q, live_q));self.assertTrue(R.q_equal(tuple(q), live_q))
        self.assertFalse(R.q_equal([np.nextafter(q[0], 1.)] + q[1:], live_q))   # one ulp apart
        self.assertFalse(R.q_equal(q, [0.] + live_q[1:]));self.assertFalse(R.q_equal(q, live_q[:-1]))
        self.assertFalse(R.q_equal([.5, None], [.5, 0.]));self.assertFalse(R.q_equal([1.], [True]))   # None is not 0; a bool is not a q
        self.assertTrue(R.q_equal([1.], [1]));self.assertTrue(R.q_equal(None, None));self.assertFalse(R.q_equal(None, live_q))

    def test_live_join_and_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 't.jsonl'
            rows = [dict(decision_index=0, source='search', search=dict(status='COMPLETE', action_index=3, q_values=[.1] * 9, tree_commit='live'),
                greedy=dict(action_index=1), request_kind='move', turn=1, played_action_index=3, prefix_sha256='p0'),
                dict(decision_index=1, source='single_legal', search=None, greedy=dict(action_index=4), request_kind='forced', turn=None,
                    played_action_index=4, prefix_sha256='p1'), dict(decision_index=1, source='single_legal')]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows) + '\n')
            index, duplicated = R.live_index(path)
            self.assertEqual((sorted(index), duplicated), ([0, 1], [1]));self.assertEqual(R.live_index(Path(tmp) / 'none.jsonl'), ({}, []))
        found = R.live_fields(rows[0]);single = R.live_fields(rows[1])
        self.assertEqual((found['search_action_index'], found['greedy_action_index'], found['tree_commit'], found['played_action_index']), (3, 1, 'live', 3))
        self.assertEqual((single['search_q_values'], single['search_status'], single['request_kind']), (None, None, 'forced'))
        complete = dict(status='COMPLETE', action_index=3, q_values=[.1] * 9)
        self.assertEqual(R.parity(complete, found), dict(live_found=True, live_searched=True, replay_complete=True, status_match=True, action_equal=True,
            q_equal=True))
        p = R.parity(dict(complete, action_index=2, q_values=[.1] * 8 + [None]), found)
        self.assertEqual((p['action_equal'], p['q_equal']), (False, False))
        p = R.parity(dict(status='REFUSED', action_index=None, q_values=None), found)
        self.assertEqual((p['status_match'], p['action_equal'], p['q_equal']), (False, None, None))   # parity only where both searched
        p = R.parity(complete, dict(found, source='greedy_fallback'))
        self.assertEqual((p['live_searched'], p['status_match'], p['q_equal']), (False, False, None))
        self.assertEqual(R.parity(complete, None)['live_found'], False)

    def test_single_legal_decisions_are_never_sent_as_the_bench(self):
        for single in (True, False):
            request = BT.live_request(0, single)
            self.assertEqual(R.sent(request), not (sum(R.bench.legal_mask(request) or ()) == 1));self.assertEqual(R.sent(request), not single)
        self.assertTrue(R.sent(dict(BT.address_request(0))))   # an unreadable request is sent, as the bench sends it


def write_capture(path, singles, *, owner='p1', battle_id='battle-1001', stream_id=5):
    """A valid owner capture (phase_d_owning_capture's chain) with one decision per entry of `singles` (one legal action or nine)."""
    blobs, state = [], dict(prior='0' * 64, lines=0, requests=0)
    prefix = hashlib.sha256()

    def emit(**fields):
        blob = owning.encoded(dict(schema=owning.SCHEMA, sequence=len(blobs), previous_sha256=state['prior'], **fields))
        blobs.append(blob);state['prior'] = hashlib.sha256(blob).hexdigest()
    emit(event='BEGIN', cell=dict(ordinal=1001), declaration_sha256='d' * 64, namespace='ns', battle_id=battle_id, owner=owner, seed=7, stream_id=stream_id)
    for i, single in enumerate(singles):
        line = BT.own_request(single);delta = [f'|turn|{i + 1}', line]
        for x in delta:prefix.update(x.encode() + b'\n')
        emit(event='WIRE', generation=i + 1, start=state['lines'], end=state['lines'] + 2, prefix_sha256=prefix.hexdigest(), lines=delta)
        state['lines'] += 2;state['requests'] += 1
        emit(event='ROOT', decision_index=i, turn=i + 1, generation=i + 1, request_count=state['requests'], prefix_sha256=prefix.hexdigest(),
            own_request_line=line)
    path.write_bytes(b''.join(b + b'\n' for b in blobs))
    return {r['decision_index']: r for r in owning.read_capture(path)}


def write_run(root, cells):
    """results/<ordinal>.json (and an owner capture file) per (ordinal, arm, seed, seat, status)."""
    for d in ('results', 'owning-wire', 'telemetry'):(root / d).mkdir(parents=True, exist_ok=True)
    for ordinal, arm, seed, seat, status in cells:
        (root / 'results' / f'{ordinal}.json').write_text(json.dumps(dict(status=status, cell=dict(ordinal=ordinal, arm=arm, seed=seed, seat=seat,
            pair_ordinal=ordinal // 3))))
        (root / 'owning-wire' / f'{ordinal}.jsonl').write_text('')


class CellTests(unittest.TestCase):
    def run_directory(self, tmp):
        root = Path(tmp) / 'run'
        cells = [(3 * p + a, ('RAW', 'DN2W8', 'DN2W16')[a], 100 + p // 2, p % 2, 'COMPLETE') for p in range(40) for a in (0, 2)]
        write_run(root, cells + [(3 * 40 + 2, 'DN2W16', 120, 0, 'UNKNOWN')])
        (root / 'owning-wire' / '5.jsonl').unlink()   # cell 5 (DN2W16) has no capture
        return root

    def args(self, root, **kw):
        return SimpleNamespace(**dict(dict(run_directory=str(root), arm='DN2W16', cells=None, sample=None, sample_seed=None, shard=0, shards=1), **kw))

    def test_sample_is_stratified_reproducible_and_sharded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.run_directory(tmp)
            pool = R.arm_cells(root, 'DN2W16')
            self.assertEqual(len(pool), 41);self.assertTrue(all(c['arm'] == 'DN2W16' for c in pool))
            picked = R.select_cells(self.args(root, sample=8, sample_seed=1))
            again = R.select_cells(self.args(root, sample=8, sample_seed=1));other = R.select_cells(self.args(root, sample=8, sample_seed=2))
            self.assertEqual(picked, again);self.assertNotEqual(picked, other)
            strata = lambda cells: [sum((c['seed'] % 2, c['seat']) == s for c in cells) for s in R.STRATA]
            self.assertEqual(strata(picked), [2, 2, 2, 2]);self.assertEqual([c['ordinal'] for c in picked], sorted(c['ordinal'] for c in picked))
            self.assertEqual(strata(R.select_cells(self.args(root, sample=10, sample_seed=1))), [3, 3, 2, 2])   # the remainder to the first strata
            everything = R.select_cells(self.args(root, sample=36, sample_seed=4))
            self.assertNotIn(122, [c['ordinal'] for c in everything]);self.assertNotIn(5, [c['ordinal'] for c in everything])   # UNKNOWN; no capture
            shards = [R.select_cells(self.args(root, sample=36, sample_seed=4, shard=i, shards=3)) for i in range(3)]
            self.assertEqual(sorted(c['ordinal'] for s in shards for c in s), [c['ordinal'] for c in everything])   # a partition
            self.assertEqual([len(s) for s in shards], [12, 12, 12])
            with self.assertRaises(ValueError):R.select_cells(self.args(root, sample=44, sample_seed=1))   # a stratum holds at most 10
            self.assertEqual([c['ordinal'] for c in R.select_cells(self.args(root, cells=[8, 2, 8]))], [2, 8])
            self.assertEqual([c['ordinal'] for c in R.select_cells(self.args(root, cells=[2, 8, 11], shard=1, shards=2))], [8])
            with self.assertRaises(ValueError):R.select_cells(self.args(root, cells=[3]))   # a RAW cell
            with self.assertRaises(ValueError):R.select_cells(self.args(root, cells=[5]))   # no capture
            with self.assertRaises(OSError):R.select_cells(self.args(root, cells=[1]))   # no result

    def test_sample_rank_is_a_fixed_hash(self):
        self.assertEqual(R.sample_rank(1, 2), hashlib.sha256(json.dumps([R.SAMPLE_DOMAIN, 1, 2], separators=(',', ':')).encode()).hexdigest())


class GuardTests(unittest.TestCase):
    def runner(self, override=None):
        answers = {('rev-parse', 'HEAD'): 'h' * 40, ('status', '--porcelain', '--untracked-files=all', '--', *R.LIVE_TREES): ''}
        for path, tree in R.LIVE_TREES.items():answers[('rev-parse', f'HEAD:{path}')] = answers[('rev-parse', f'{R.LIVE_COMMIT}:{path}')] = tree
        answers.update(override or {})

        def run(root, *arguments):
            value = answers[arguments]
            if value is None:raise subprocess.CalledProcessError(128, 'git')
            return value
        return run

    def test_the_guard_refuses_a_mismatched_tree(self):
        self.assertEqual(R.tree_guard(run=self.runner())['trees'], R.LIVE_TREES)
        with self.assertRaisesRegex(ValueError, 'HEAD:gpu_engine'):R.tree_guard(run=self.runner({('rev-parse', 'HEAD:gpu_engine'): 'f' * 40}))
        with self.assertRaisesRegex(ValueError, 'HEAD:src'):R.tree_guard(run=self.runner({('rev-parse', 'HEAD:src'): None}))
        with self.assertRaisesRegex(ValueError, 'pin'):R.tree_guard(run=self.runner({('rev-parse', f'{R.LIVE_COMMIT}:src'): 'e' * 40}))
        with self.assertRaisesRegex(ValueError, 'tree-commit'):R.tree_guard(tree_commit='x' * 40, run=self.runner())
        self.assertEqual(R.tree_guard(tree_commit='h' * 40, run=self.runner())['head'], 'h' * 40)
        status = ('status', '--porcelain', '--untracked-files=all', '--', *R.LIVE_TREES)
        with self.assertRaisesRegex(ValueError, 'working tree'):R.tree_guard(run=self.runner({status: 'M gpu_engine/state.py'}))
        with self.assertRaisesRegex(ValueError, 'working tree'):R.tree_guard(run=self.runner({status: '?? src/pokezero/extra.py'}))
        self.assertTrue(R.tree_guard(run=self.runner({status: '?? src/notes.txt'})))   # an untracked non-Python file is no code
        shallow = {('rev-parse', f'{R.LIVE_COMMIT}:{p}'): None for p in R.LIVE_TREES}   # a clone without the live commit: the pins alone
        self.assertFalse(R.tree_guard(run=self.runner(shallow))['live_commit_present'])
        with self.assertRaisesRegex(ValueError, 'git checkout'):R.tree_guard(run=self.runner({('rev-parse', 'HEAD'): None}))

    def accept_runner(self, override=None, *, head='a' * 40, trees=None):
        """git answers for --accept-tree: HEAD `head`, its trees `trees` (default: gpu_engine modified, the rest the live pins)."""
        trees = dict(R.LIVE_TREES, gpu_engine='9' * 40) if trees is None else trees
        answers = {('rev-parse', 'HEAD'): head, ('rev-parse', '--verify', f'{head}^{{commit}}'): head, ('rev-parse', '--verify', 'HEAD^{commit}'): head,
            ('rev-parse', '--verify', 'b' * 40 + '^{commit}'): 'b' * 40, ('rev-parse', '--verify', 'nope^{commit}'): None,
            ('status', '--porcelain', '--untracked-files=all', '--', *R.ACCEPT_PATHS): ''}
        for path, tree in trees.items():answers[('rev-parse', f'{head}:{path}')] = answers[('rev-parse', f'HEAD:{path}')] = tree
        for path, tree in R.LIVE_TREES.items():answers[('rev-parse', f'{R.LIVE_COMMIT}:{path}')] = tree
        answers.update(override or {})

        def run(root, *arguments):
            value = answers[arguments]
            if value is None:raise subprocess.CalledProcessError(128, 'git')
            return value
        return run

    def test_accept_tree_replays_a_modified_clean_head_and_refuses_anything_else(self):
        head = 'a' * 40;run = self.accept_runner()
        with self.assertRaisesRegex(ValueError, 'HEAD:gpu_engine'):R.tree_guard(run=run)   # the default guard is unchanged: refused
        for commit in (head, 'HEAD'):
            got = R.tree_guard(accept_tree=commit, run=run)
            self.assertEqual((got['head'], got['accepted_tree'], got['modified_tree'], got['note'], got['live_trees']), (head, head, True, R.MODIFIED_TREE, R.LIVE_TREES))
            self.assertEqual((got['trees']['gpu_engine'], got['trees']['src'], got['differs_from_live']), ('9' * 40, R.LIVE_TREES['src'], ['gpu_engine']))
        with self.assertRaisesRegex(ValueError, 'is not HEAD'):R.tree_guard(accept_tree='b' * 40, run=run)   # a commit, not HEAD
        with self.assertRaisesRegex(ValueError, 'not a commit'):R.tree_guard(accept_tree='nope', run=run)
        with self.assertRaisesRegex(ValueError, 'needs a commit'):R.tree_guard(accept_tree='  ', run=run)
        status = ('status', '--porcelain', '--untracked-files=all', '--', *R.ACCEPT_PATHS)
        for dirty in ('M gpu_engine/constructor_worlds.py', ' M scripts/depthn_rr_reprice.py', '?? scripts/extra.py'):   # tracked change, untracked .py
            with self.subTest(dirty=dirty), self.assertRaisesRegex(ValueError, 'clean HEAD'):R.tree_guard(accept_tree=head, run=self.accept_runner({status: dirty}))
        self.assertTrue(R.tree_guard(accept_tree=head, run=self.accept_runner({status: '?? scripts/notes.txt'})))   # no code
        with self.assertRaisesRegex(ValueError, 'tree-commit'):R.tree_guard(tree_commit='c' * 40, accept_tree=head, run=run)
        with self.assertRaisesRegex(ValueError, 'has no src'):R.tree_guard(accept_tree=head, run=self.accept_runner({('rev-parse', f'{head}:src'): None}))
        default = R.tree_guard(run=self.runner())   # the live tree: no accept keys, the live pins
        self.assertEqual(set(default), {'head', 'live_commit', 'live_commit_present', 'trees'})

    def test_accept_tree_option(self):
        args = R.parser().parse_args(BASE + CHAMPION + ['--run-directory', 'r', '--out', 'o', '--cells', '1001'])
        self.assertIsNone(args.accept_tree)
        args = R.parser().parse_args(BASE + CHAMPION + ['--run-directory', 'r', '--out', 'o', '--cells', '1001', '--accept-tree', 'abc'])
        self.assertEqual(args.accept_tree, 'abc')

    @unittest.skipUnless(subprocess.run(['git', '-C', str(R.ROOT), 'rev-parse', R.LIVE_COMMIT], capture_output=True).returncode == 0,
        'needs a checkout holding the live commit')
    def test_the_pins_are_the_live_commits_trees(self):
        for path, tree in R.LIVE_TREES.items():self.assertEqual(R.git(R.ROOT, 'rev-parse', f'{R.LIVE_COMMIT}:{path}'), tree)

    def test_identity_check(self):
        good = dict(R.LIVE_IDENTITY, rung='DN2W16', pricer=dict(buckets=[1], W=16))
        self.assertEqual(R.identity_problems(good, 'DN2W16'), [])
        bad = R.identity_problems(dict(good, recipe_seed=1, session_cap=8, pricer=dict(buckets=[1, 2], W=8)), 'DN2W16')
        self.assertEqual(len(bad), 4);self.assertEqual(len(R.identity_problems(good, 'DN2W8')), 2)   # the rung and W


BASE = ['--showdown-root', 's', '--node-binary', 'n', '--metadata', 'm', '--expected-metadata-sha256', 'x', '--vocabulary', 'v',
    '--expected-vocabulary-sha256', 'y', '--warmup-requests', 'w.jsonl', '--arm', 'DN2W16', '--rung', 'DN2W16']
CHAMPION = ['--champion-archive', 'a', '--champion-archive-sha256', 'c' * 64, '--champion-checkpoint-sha256', 'd' * 64]


class MainTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'JAX_DEFAULT_MATMUL_PRECISION': 'highest'});environment.start();self.addCleanup(environment.stop)

    def test_options(self):
        run = ['--run-directory', 'r', '--out', 'o', '--cells', '1001']
        with patch.object(R.server.evaluators, 'check_arguments'):
            args = R.options(BASE + CHAMPION + run)
            self.assertEqual((args.buckets, args.max_requests, args.saved_root_admission, args.cells, args.port), ((1,), 1, False, [1001], 0))
            refused = (['--roster', 'r.json', 'z'], ['--sessions', '2'], ['--buckets', '1,2'], ['--max-requests', '2'], ['--saved-root-admission'],
                ['--rung', 'DN2W8'], ['--shard', '2', '--shards', '2'], ['--max-decisions', '0'])
            for extra in refused:
                with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):R.options(BASE + CHAMPION + run + extra)
            for argv in (BASE + run, [x for x in BASE + CHAMPION + run if x not in ('--warmup-requests', 'w.jsonl')],
                    BASE + CHAMPION + ['--run-directory', 'r', '--out', 'o', '--sample', '8'], BASE + CHAMPION + run + ['--sample', '8', '--sample-seed', '1'],
                    BASE[:-2] + CHAMPION + run):   # no champion; no warmup file; no sample seed; both selections; the server's default rung
                with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):R.options(argv)
            self.assertEqual(R.options(BASE + CHAMPION + ['--run-directory', 'r', '--out', 'o', '--sample', '8', '--sample-seed', '3']).sample_seed, 3)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):R.options(BASE + CHAMPION + run + ['--reply-archive', 'p.npz'])   # dropped

    def replay(self, tmp, *extra, header=None, guard=None, peer_off=()):
        """main over a run directory with cell 1001: decisions 0-4, decision 1 single-legal, decision 4 refused by the constructor."""
        root = Path(tmp) / 'run';write_run(root, [(1001, 'DN2W16', 7, 1, 'COMPLETE')])
        roots = write_capture(root / 'owning-wire' / '1001.jsonl', [False, True, False, False, False])
        want = expected_decision();q = json.loads(json.dumps(want['q_values']))
        lane = next(a for a in range(9) if q[a] is not None);off = list(q);off[lane] = float(np.nextafter(q[lane], 2.))
        search = lambda q_values, source='search', cause=None, status='COMPLETE': dict(status=status, cause=cause, action_index=want['action_index'],
            q_values=q_values, tree_commit=R.LIVE_COMMIT, server_id='gpu:7001')
        live_rows = [dict(decision_index=0, source='search', search=search(q), request_kind='move'),
            dict(decision_index=1, source='single_legal', search=None, request_kind='forced'),
            dict(decision_index=2, source='search', search=search(off), request_kind='move'),   # one ulp off on one lane
            dict(decision_index=3, source='greedy_fallback', cause='timeout', search=dict(status='REFUSED', cause='timeout', client_refused=True),
                request_kind='move'),
            dict(decision_index=4, source='search', search=search(q), request_kind='switch')]
        with (root / 'telemetry' / '1001.jsonl').open('w') as f:
            for r in live_rows:f.write(json.dumps(dict(r, prefix_sha256=roots[r['decision_index']]['prefix_sha256'], greedy=dict(action_index=0), turn=1,
                played_action_index=0)) + '\n')
        args = R.parser().parse_args(BASE + CHAMPION + ['--run-directory', str(root), '--out', str(Path(tmp) / 'out'), '--cells', '1001', *extra])
        args.port, args.max_requests = 0, 1
        constructor = FakeConstructor(refuse={4});search_object = fake_search(constructor);warmed = []
        search_object.pricers[1].executor.off = peer_off   # these worlds' peer inputs are one ulp off their recorded logits
        search_object.warmup = lambda requests: warmed.append(list(requests)) or dict(buckets={1: dict(cold_seconds=1.)}, programs=3)
        header = dict(R.LIVE_IDENTITY, rung='DN2W16', pricer=dict(buckets=[1], W=16), tree_commit='t') if header is None else header
        with ExitStack() as stack:
            pricing(stack)
            stack.enter_context(patch.object(R, 'options', return_value=args))
            self.guard = stack.enter_context(patch.object(R, 'tree_guard', return_value=dict(head='h') if guard is None else guard))
            stack.enter_context(patch.object(R.server, 'read_requests', return_value=[dict(saved=1)]))
            stack.enter_context(patch.object(R.server, 'build', return_value=search_object))
            stack.enter_context(patch.object(R.server, 'header', return_value=header))
            stack.enter_context(patch.object(R, 'PeerLanes', return_value=fake_peer))
            stack.enter_context(redirect_stderr(io.StringIO()))
            with patch('sys.stdout', new_callable=io.StringIO) as stdout:code = R.main([])
        out = Path(tmp) / 'out'
        rows = [json.loads(x) for x in (out / 'rows-0.jsonl').read_text().splitlines()]
        with np.load(out / 'peer-inputs-0.npz') as sidecar:self.sidecar = {k: sidecar[k] for k in sidecar.files}
        return code, rows, json.loads((out / 'summary-0.json').read_text()), json.loads((out / 'header-0.json').read_text()), constructor, warmed, stdout.getvalue(), args

    def test_main_with_accept_tree_records_a_modified_tree_replay(self):
        accepted = dict(head='h', trees={}, accepted_tree='h', modified_tree=True, note=R.MODIFIED_TREE)
        with tempfile.TemporaryDirectory() as tmp:
            code, rows, summary, header, *_ = self.replay(tmp, '--accept-tree', 'h', guard=accepted)
        self.assertEqual(code, 0);self.guard.assert_called_once_with(tree_commit=None, accept_tree='h')
        self.assertEqual((header['tree_guard'], header['replay_against']), (accepted, R.MODIFIED_TREE))
        self.assertTrue(all('parity' in r for r in rows))   # parity against live is still recorded
        with tempfile.TemporaryDirectory() as tmp:header = self.replay(tmp)[3]
        self.assertNotIn('replay_against', header);self.guard.assert_called_once_with(tree_commit=None, accept_tree=None)   # the default: unchanged

    def test_main_replays_records_and_joins(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, rows, summary, header, constructor, warmed, stdout, args = self.replay(tmp)
            with patch.object(R, 'options', return_value=args), patch.object(R, 'tree_guard', return_value={}), self.assertRaises(FileExistsError):
                R.main([])   # never into an existing directory
        self.assertEqual(code, 0)
        self.assertIs(live.decide, ORIGINAL_DECIDE);self.assertIs(cw.prepare_roots, ORIGINAL_PREPARE)
        self.assertEqual(warmed, [[dict(saved=1)]])   # the server's warmup file, once
        self.assertEqual(constructor.calls, [0, 2, 3, 4])   # the single-legal decision 1 is never sent
        self.assertEqual([r['decision_index'] for r in rows], [0, 2, 3, 4]);self.assertEqual(len(stdout.splitlines()), 5)   # progress + done
        first = rows[0];want = expected_decision()
        self.assertEqual((first['ordinal'], first['battle_id'], first['owner'], first['stream_id'], first['request_count']), (1001, 'battle-1001', 'p1', 5, 1))
        self.assertEqual((rows[1]['request_count'], first['seat'], first['seed']), (3, 1, 7))
        self.assertEqual((first['replay']['root_view_source'], first['replay']['b2_view']),   # the lead's validation: world 0's view = B2's
            ('b2', dict(equal=True, keys=dict(numeric_features=True), numeric_max_abs_diff=0., error=None)))
        self.assertEqual((rows[-1]['replay']['root_view_source'], rows[-1]['replay']['b2_view']), (None, None))   # constructor-refused: never built
        self.assertEqual((first['replay']['status'], first['replay']['action_index'], first['replay']['q_values']), ('COMPLETE', want['action_index'],
            json.loads(json.dumps(want['q_values']))))
        self.assertEqual(first['root'], dict(decided=True, root_logits=list(LOGITS), root_legal=list(LEGAL), own_moves=['m0', 'm1', 'm2', 'm3'],
            own_recharge_lanes=[False] * 4, opponent_recharge=False, refusal_cause=None))   # the own request's lanes 0-3 (BT.own_request: m0..m3)
        self.assertEqual(len(first['worlds']), 16)
        for w, row in enumerate(first['worlds']):
            self.assertEqual((row['world'], row['index'], row['seed'], row['kernel_seed'], row['constructed']), (w, w, 500 + w, 1000 + w, True))
            self.assertEqual(row['grid'], [list(r) for r in prices(w).joint_values])
            self.assertEqual((row['kept'], row['root_peer_logits']), (list(KEPT), list(prices(w).opponent_logits)))
            self.assertEqual(row['kept_matches_peer_legal'], w != 5);self.assertTrue(row['requested_matches_peer'])
        self.assertEqual(first['worlds'][0]['opponent']['active_species'], 'snorlax')
        self.assertEqual(first['worlds'][0]['peer']['lanes'], ['move:thunderbolt', None, 'move:hiddenpower', None, 'switch:pikachu', None, None,
            'switch:hooh', None])   # lane 7 is switch_targets[3]: party slot 4
        self.assertEqual((first['kept_disagreements'], first['prepared']['kind'], first['capture_ok'], first['notes']), (1, 'Prepared', True, []))
        self.assertEqual([r['parity']['action_equal'] for r in rows], [True, True, None, None])
        self.assertEqual([r['parity']['q_equal'] for r in rows], [True, False, None, None])
        self.assertEqual([r['parity']['status_match'] for r in rows], [True, True, False, False])
        last = rows[3]
        self.assertEqual((last['replay']['status'], last['replay']['cause'], last['prepared']['kind'], last['worlds'], last['root']['decided'],
            last['root']['refusal_cause']), ('REFUSED', 'constructor_refused', 'RootRefusal', [], False, 'TEST_REFUSAL'))
        self.assertEqual((header['peer_inputs'], header['recharge_moves']), (dict(path='peer-inputs-0.npz', schema=R.PEER_INPUTS_SCHEMA),
            ['blastburn', 'frenzyplant', 'hydrocannon', 'hyperbeam']))
        self.assertEqual((summary['peer_inputs_check']['passed'], last['peer_inputs_check']), (True, None))
        self.assertEqual((last['live']['source'], last['live']['request_kind'], last['live_problem']), ('search', 'switch', None))
        self.assertEqual((summary['decisions_sent'], summary['complete'], summary['live_searched'], summary['skipped_single_legal']), (4, 3, 3, 1))
        self.assertEqual({k: summary['parity'][k] for k in ('both', 'action_equal', 'q_equal')}, dict(both=2, action_equal=2, q_equal=1))
        self.assertEqual(summary['parity']['q_mismatches'], [dict(ordinal=1001, decision_index=2)])
        self.assertEqual([(m['decision_index'], m['live_cause'], m['replay_status']) for m in summary['status_mismatches']],
            [(3, 'timeout', 'COMPLETE'), (4, None, 'REFUSED')])
        self.assertEqual((summary['kept_legal_disagreements'], summary['skip_disagreements'], summary['live_problems'], summary['capture_anomalies']),
            (3, [], [], []))
        self.assertEqual((summary['live_tree_commits'], summary['interrupted'], summary['max_decisions_reached']), ({R.LIVE_COMMIT: 3}, False, False))
        self.assertEqual((header['cells'][0]['ordinal'], header['tree_guard'], header['warmup']['programs']), (1001, dict(head='h'), 3))

    def test_peer_inputs_sidecar_and_the_champion_self_check(self):
        """peer-inputs-<shard>.npz: per decided world the champion adapter's 23-token view of that world's depth-0 peer inputs
        (the views' fourth output) and the peer legal mask, keyed by (ordinal, decision_index, world); the champion on the same
        stacked inputs equals the recorded root_peer_logits bit for bit."""
        with tempfile.TemporaryDirectory() as tmp:
            code, rows, summary, header, *_ = self.replay(tmp)
        x = self.sidecar;self.assertEqual(code, 0)
        self.assertEqual(sorted(x), ['attention_mask', 'categorical_ids', 'decision_index', 'header', 'history_mask', 'legal', 'numeric_features',
            'opponent_requested', 'ordinal', 'root_peer_logits', 'self_check', 'token_type_ids', 'world'])
        shapes = {k: (v.shape, str(v.dtype)) for k, v in x.items() if k != 'header'}
        self.assertEqual(shapes, dict(numeric_features=((48, 23, 132), 'float32'), categorical_ids=((48, 23, 41), 'int32'), token_type_ids=((48, 23), 'int32'),
            attention_mask=((48, 23), 'bool'), history_mask=((48, 1), 'bool'), legal=((48, 9), 'bool'), ordinal=((48,), 'int64'),
            decision_index=((48,), 'int64'), world=((48,), 'int32'), root_peer_logits=((48, 9), 'float32'), opponent_requested=((48,), 'bool'),
            self_check=((48,), 'int8')))
        self.assertEqual(list(zip(x['ordinal'].tolist(), x['decision_index'].tolist(), x['world'].tolist())),
            [(1001, d, w) for d in (0, 2, 3) for w in range(16)])   # the three decided roots; the refused one has no worlds
        for k in range(48):
            w = int(x['world'][k]);want = np.asarray(prices(w).opponent_logits, np.float32)
            np.testing.assert_array_equal(x['numeric_features'][k, 0, :9], want)   # the PEER seat's inputs (the owner's are 100 more)
            np.testing.assert_array_equal(x['root_peer_logits'][k], want)
        self.assertTrue((x['legal'] == np.asarray(KEPT)).all() and x['self_check'].tolist() == [1] * 48 and x['opponent_requested'].all())
        document = json.loads(str(x['header']))
        self.assertEqual({k: document[k] for k in ('schema', 'worlds', 'shard', 'champion_identity', 'key', 'path')},
            dict(schema=R.PEER_INPUTS_SCHEMA, worlds=48, shard=0, champion_identity=['champion', 'e' * 64, 'bfloat16'],
                key=['ordinal', 'decision_index', 'world'], path='peer-inputs-0.npz'))
        self.assertEqual((document['tree_guard'], document['arrays']['categorical_ids']), (dict(head='h'), dict(shape=[48, 23, 41], dtype='int32')))
        first = rows[0]
        self.assertEqual([(w['peer_inputs_index'], w['peer_self_check'], w['peer_self_check_max_abs_diff']) for w in first['worlds']],
            [(w, True, 0.) for w in range(16)])
        self.assertEqual(first['peer_inputs_check'], dict(checked=16, equal=16, failed_worlds=[], written=16, applicable=True))
        self.assertEqual(summary['peer_inputs_check'], dict(decided_roots=3, roots_checked=3, roots_not_applicable=0, worlds_written=48,
            worlds_checked=48, worlds_equal=48, failures=[], unchecked_roots=[], passed=True))
        with tempfile.TemporaryDirectory() as tmp:   # one world's peer inputs one ulp off its recorded logits: the check fails
            _, rows, summary, *_ = self.replay(tmp, peer_off=(3,))
        self.assertEqual([w['peer_self_check'] for w in rows[0]['worlds']], [w != 3 for w in range(16)])
        self.assertGreater(rows[0]['worlds'][3]['peer_self_check_max_abs_diff'], 0.)
        self.assertEqual(self.sidecar['self_check'].tolist(), [int(w != 3) for _ in range(3) for w in range(16)])
        c = summary['peer_inputs_check'];self.assertEqual((c['worlds_checked'], c['worlds_equal'], c['passed']), (48, 45, False))
        self.assertEqual(c['failures'], [dict(ordinal=1001, decision_index=d, worlds=[3]) for d in (0, 2, 3)])

    def test_a_failed_peer_capture_fails_the_check_and_keeps_the_sidecar_aligned(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(FakeEvaluator, 'view', side_effect=ValueError('slice')):
            _, rows, summary, *_ = self.replay(tmp)
        self.assertEqual(rows[0]['peer_inputs_check'], dict(checked=16, equal=0, failed_worlds=list(range(16)), written=0, applicable=True))
        self.assertIn('peer inputs: ValueError: slice', rows[0]['notes']);self.assertFalse(summary['peer_inputs_check']['passed'])
        self.assertEqual(self.sidecar['numeric_features'].shape, (0, 23, 132))   # nothing kept, every column still aligned

    def test_a_checkable_world_whose_capture_raised_fails_and_undecided_roots_are_never_captured(self):
        observe = PeerExecutor.observe
        def broken(executor, state, public, seat, mask):
            if int(np.asarray(public['w'])) == 5 and int(np.asarray(seat)[0]) == 1:raise ValueError('peer view')
            return observe(executor, state, public, seat, mask)
        with tempfile.TemporaryDirectory() as tmp, patch.object(PeerExecutor, 'observe', broken):
            _, rows, summary, *_ = self.replay(tmp)
        first = rows[0]
        self.assertEqual([w['peer_self_check'] for w in first['worlds']], [w != 5 for w in range(16)])   # failed, not skipped
        self.assertIsNone(first['worlds'][5]['peer_inputs_index']);self.assertIn('world 5: ValueError: peer view', first['notes'])
        self.assertEqual(first['peer_inputs_check'], dict(checked=16, equal=15, failed_worlds=[5], written=15, applicable=True))
        self.assertFalse(summary['peer_inputs_check']['passed'])
        # A prepared root that never reached decide: its worlds are described, never captured or kept.
        class Counting:
            calls = 0
            def capture(self, *a):Counting.calls += 1
            def annotate(self, *a):raise AssertionError('annotated an undecided root')
        detached, made, legal = FakeConstructor().prepare(dict(decision_index=0), 0)
        prepared = SimpleNamespace(prepared=[(cw.Prepared('r', detached, made, legal, 0., 'binding'),)], decided=[])
        detail = R.describe(prepared, fake_peer, VOCABULARY, Counting(), (1, 0))
        self.assertEqual((Counting.calls, detail['root'], len(detail['worlds'])), (0, None, 16))
        self.assertTrue(all('peer_self_check' not in w for w in detail['worlds']))

    def test_peer_inputs_summary_fails_closed(self):
        complete = lambda i, check: dict(ordinal=1, decision_index=i, replay=dict(status='COMPLETE'), peer_inputs_check=check)
        ok = dict(checked=2, equal=2, failed_worlds=[], written=2)
        self.assertTrue(R.peer_inputs_summary([complete(0, ok), dict(complete(1, None), replay=dict(status='REFUSED'))])['passed'])
        self.assertFalse(R.peer_inputs_summary([complete(0, ok), complete(1, None)])['passed'])   # a decided root left unchecked
        self.assertFalse(R.peer_inputs_summary([complete(0, dict(ok, checked=0, equal=0))])['passed'])   # nothing checked
        # a decided root with no priced opponent-requested world: not applicable, counted, never failing (the definitive run's 362)
        na = R.peer_inputs_summary([complete(0, ok), complete(1, dict(ok, checked=0, equal=0, applicable=False))])
        self.assertEqual((na['roots_checked'], na['roots_not_applicable'], na['unchecked_roots'], na['passed']), (1, 1, [], True))
        self.assertFalse(R.peer_inputs_summary([complete(0, dict(ok, checked=0, equal=0, applicable=False))])['passed'])   # still nothing checked
        broken = R.peer_inputs_summary([complete(0, ok), complete(1, dict(ok, checked=0, equal=0, applicable=False, failed_worlds=[2]))])
        self.assertFalse(broken['passed'])   # a failure is never not applicable
        none = R.peer_inputs_summary([complete(0, ok), complete(1, dict(ok, checked=0, equal=0))])   # a decided root with no checkable world
        self.assertEqual((none['roots_checked'], none['unchecked_roots'], none['passed']), (1, [dict(ordinal=1, decision_index=1)], False))
        self.assertFalse(R.peer_inputs_summary([])['passed'])

    def test_max_decisions_stops_a_smoke_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, rows, summary, *_ = self.replay(tmp, '--max-decisions', '2')
        self.assertEqual((code, [r['decision_index'] for r in rows], summary['max_decisions_reached']), (0, [0, 2], True))

    def test_a_search_unlike_the_live_servers_is_refused_before_warmup(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(SystemExit, 'session_cap'):
            self.replay(tmp, header=dict(R.LIVE_IDENTITY, rung='DN2W16', session_cap=64, pricer=dict(buckets=[1], W=16)))


if __name__ == '__main__':
    unittest.main()
