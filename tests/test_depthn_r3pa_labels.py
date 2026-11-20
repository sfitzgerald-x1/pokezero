"""R3pa's live lane labels (gpu_engine.depthn_r3pa_labels) against the offline labels: depthn_rr_reprice.opponent_facts, PeerLanes
and lane_labels, then mc_ladder_backline.labels / seen_species, on real BattleStates with a views stub (CPU only, light)."""
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from scripts import depthn_rr_reprice as R
from gpu_engine import depthn_r3pa_labels as L
from gpu_engine import fp_reply_evidence as ev
from gpu_engine import mc_ladder_backline as B
from gpu_engine.state import empty_state
from pokezero import v5_gumbel

VOCABULARY = dict(species=['', 'Pikachu', 'Mr. Mime', 'Snorlax', "Farfetch’d", 'Ho-Oh', 'Gengar', 'Unown-A', 'unownb', 'Unown', 'Starmie', '-'],
    moves=['', 'Thunderbolt', 'Hidden Power Fire', 'hiddenpowerwater70', 'Body Slam', 'Solar Beam', 'Hidden Power', 'Return 102', 'Struggle', 'Surf'],
    abilities=['', 'Static', 'Pressure'], items=['', 'Leftovers'], types=['', 'Fire', 'Water'])
EMPTY = len(VOCABULARY['species']) - 1   # '-': a species that normalizes to '' (a member without a species)
W = 4


class Executor:
    """depth2._make_views' executor: legal() then observe() per seat; seat s's observation reads the public stub's row s, so a
    seat mix-up changes the lanes. Raises when `armed` is false (nothing may run)."""
    def __init__(self, armed=True):self.armed = armed

    def legal(self, state):
        if not self.armed:raise AssertionError('the executor ran')
        return jnp.zeros((1, 2, 9), bool)

    def observe(self, state, public, seat, mask):
        if not self.armed:raise AssertionError('the executor ran')
        s = jnp.asarray(seat)[0];row = lambda k: jnp.asarray(public[k])[0, s][None]
        return SimpleNamespace(inputs={}, observation=SimpleNamespace(legal_action_mask=row('legal'), move_targets=row('move_targets'),
            switch_targets=row('switch_targets'), error=jnp.zeros((1,), jnp.int32)))


def world_state(rng, foe, core):
    """A BattleState whose side `foe` has the core species plus random ones, a random active with random moves, PP and charge."""
    s = empty_state(1);species = list(core) + [int(x) for x in rng.integers(1, EMPTY, 6 - len(core))];rng.shuffle(species)
    if rng.random() < .04:species[int(rng.integers(6))] = 0 if rng.random() < .5 else EMPTY
    active = -1 if rng.random() < .1 else int(rng.integers(6))   # -1: an active outside the party (its move lanes unlabelled)
    moves = [int(x) for x in rng.integers(0, len(VOCABULARY['moves']), 4)]
    pp = [0] * 4 if rng.random() < .15 else [int(x) for x in rng.integers(0, 3, 4)]
    charge = 0 if rng.random() < .8 else (moves[int(rng.integers(4))] if rng.random() < .5 else int(rng.integers(1, len(VOCABULARY['moves']))))
    other = 1 - foe
    return s._replace(species=s.species.at[0, foe].set(jnp.asarray(species)).at[0, other].set(jnp.arange(1, 7)),
        moves=s.moves.at[0, foe, active].set(jnp.asarray(moves)), pp=s.pp.at[0, foe, active].set(jnp.asarray(pp)),
        hp=s.hp.at[0, foe].set(jnp.full(6, 50)), max_hp=s.max_hp.at[0, foe].set(jnp.full(6, 100)),
        active=s.active.at[0, foe].set(active), charge_move=s.charge_move.at[0, foe, active].set(charge))


def world_public(rng):
    return dict(legal=rng.random((1, 2, 9)) < .6, move_targets=rng.integers(-1, 5, (1, 2, 4)).astype(np.int32),   # -1 and 4: outside the active's moves
        switch_targets=rng.integers(-1, 7, (1, 2, 5)).astype(np.int32))


def offline(worlds, owner, peer_lanes):
    """The offline labels of a root's worlds (the re-pricer's capture with `peer_lanes`, a PeerLanes, then the backline) and their
    peer legal masks, or None (world_sets_missing)."""
    rows = []
    for w in worlds:
        state, public = jax.device_get((w.state, w.public))
        facts = R.opponent_facts(state, owner, VOCABULARY, public);peer = peer_lanes(state, public, owner)
        peer['lanes'] = R.lane_labels(facts, peer);rows.append(dict(opponent=facts, peer=peer))
    seen = B.seen_species(rows)
    return None if seen is None else ([B.labels(r, seen) for r in rows], [r['peer']['legal'] for r in rows])


class ParityTests(unittest.TestCase):
    def test_random_roots_label_as_the_offline_path(self):
        rng = np.random.default_rng(20261115);executor = Executor();labels = L.LaneLabels(executor, VOCABULARY);peer_lanes = R.PeerLanes(executor)
        compared = missing = 0;kinds = set();unlabelled = dict(move=0, switch=0)   # legal lanes without a label
        for _ in range(120):
            owner = int(rng.integers(2));core = [int(x) for x in rng.integers(1, EMPTY, int(rng.integers(0, 4)))]
            worlds = [SimpleNamespace(state=world_state(rng, 1 - owner, core), public=world_public(rng)) for _ in range(W)]
            want = offline(worlds, owner, peer_lanes);views = labels.views(worlds, owner)
            if want is None:
                with self.assertRaises(L.WorldSetsMissing):L.lane_labels_live(views, labels.names)
                missing += 1;continue
            got = L.lane_labels_live(views, labels.names)
            self.assertEqual(got, want[0]);self.assertEqual(views.legal.tolist(), want[1])
            for w in range(W):
                for b in range(9):unlabelled['move' if b < 4 else 'switch'] += bool(views.legal[w, b]) and got[w][b] is None
            compared += W;kinds |= {x if x is None or x == L.UNSEEN else x.split(':')[0] + (':struggle' if x == 'move:struggle' else '') for row in got for x in row}
        self.assertGreater(compared, 300);self.assertGreater(missing, 0)
        self.assertTrue(all(unlabelled.values()), unlabelled)   # both None paths (target outside, active outside) compared
        self.assertEqual(kinds, {None, 'move', 'move:struggle', 'switch', L.UNSEEN})   # every label kind was exercised
        self.assertEqual(labels.traces, {0: 1, 1: 1})   # one trace per seat at this world count

    def test_species_key_is_fp_reply_evidence_s(self):
        for name in VOCABULARY['species'] + ['Unown-?', 'UNOWN!', 'None', '', 'Nidoran-F', 'Porygon2']:
            self.assertEqual(L.species_key(name), ev.species_key(name))


class MapperTests(unittest.TestCase):
    names = L.Names(VOCABULARY)

    def views(self, kind, lane_id, party):
        kind, lane_id = np.asarray(kind, np.int32), np.asarray(lane_id, np.int32)
        return L.LaneViews(kind != L.NONE, kind, lane_id, np.asarray(party, np.int32))

    def test_labels(self):
        M, S, X, N = L.MOVE, L.SWITCH, L.STRUGGLE, L.NONE
        party = [[1, 2, 3, 7, 4, 5], [1, 2, 3, 8, 6, 10]]   # seen: pikachu, mrmime, snorlax, unown (Unown-A and unownb)
        got = L.lane_labels_live(self.views([[X, M, M, N, S, S, S, S, N], [M, M, M, M, S, S, S, S, S]],
            [[0, 2, 7, 0, 1, 7, 4, 2, 0], [0, 3, 9, 1, 2, 8, 6, 10, 3]], party), self.names)
        self.assertEqual(got, [['move:struggle', 'move:hiddenpower', 'move:return', None, 'switch:pikachu', 'switch:unown', 'switch:?', 'switch:mrmime',
            None], ['move:None', 'move:hiddenpower', 'move:surf', 'move:thunderbolt', 'switch:mrmime', 'switch:unown', 'switch:?', 'switch:?',
            'switch:snorlax']])

    def test_refusals(self):
        with self.assertRaises(L.WorldSetsMissing):L.lane_labels_live(self.views([[L.NONE] * 9], [[0] * 9], [[1, 2, 3, 4, 5, 0]]), self.names)
        with self.assertRaises(L.WorldSetsMissing):L.lane_labels_live(self.views([[L.NONE] * 9], [[0] * 9], [[1, 2, 3, 4, 5, EMPTY]]), self.names)
        with self.assertRaises(ValueError):L.lane_labels_live(self.views([[7] + [L.NONE] * 8], [[0] * 9], [[1, 2, 3, 4, 5, 6]]), self.names)
        with self.assertRaises(IndexError):L.lane_labels_live(self.views([[L.MOVE] + [L.NONE] * 8], [[99] + [0] * 8], [[1, 2, 3, 4, 5, 6]]), self.names)
        self.assertEqual(L.WorldSetsMissing('x').reason, 'world_sets_missing')

    def test_peer_legal_mismatch(self):
        legal = np.asarray([[True, False] + [False] * 7, [True] * 9, [True] + [False] * 8], bool)   # world 2 waits: never compared
        views = L.LaneViews(legal, np.zeros((3, 9), np.int32), np.zeros((3, 9), np.int32), np.ones((3, 6), np.int32))
        prices = lambda kept, asked=True: v5_gumbel.WorldPrices(asked, kept, (0.,) * 9, ((0.,) * 9,) * 9)
        self.assertEqual(L.peer_legal_mismatch([prices(tuple(bool(x) for x in legal[0])), prices((True,) * 8 + (False,)), prices((False,) * 9, False)], views), [1])
        self.assertEqual(L.peer_legal_mismatch([None, prices((True,) * 9), None], views), [])
        with self.assertRaises(ValueError):L.peer_legal_mismatch([None, None], views)


class FlagOffTests(unittest.TestCase):
    def test_construction_runs_traces_and_compiles_nothing(self):
        """Built (as the hook builds it only when R3pa is on) it compiles nothing until views() or warm(); views() then traces each
        seat once per world count, and a second root reuses the program."""
        events = []
        jax.monitoring.register_event_duration_secs_listener(lambda name, *a, **k: events.append(name) if 'compile' in name else None)
        labels = L.LaneLabels(Executor(armed=False), VOCABULARY)
        self.assertEqual((labels.programs, labels.traces, events), ({}, {0: 0, 1: 0}, []))
        rng = np.random.default_rng(3);labels.executor.armed = True
        worlds = [SimpleNamespace(state=world_state(rng, 1, [1, 2]), public=world_public(rng)) for _ in range(2)]
        labels.views(worlds, 0);self.assertEqual(labels.traces, {0: 1, 1: 0});self.assertTrue(events)
        before = len(events);labels.views(worlds, 0);self.assertEqual((labels.traces, len(events)), ({0: 1, 1: 0}, before))
        self.assertEqual(set(labels.warm(worlds)), {0, 1});self.assertEqual(labels.traces, {0: 1, 1: 1})

    def test_an_unconstructed_world_is_refused(self):
        with self.assertRaises(ValueError):L.LaneLabels(Executor(), VOCABULARY).views([SimpleNamespace(state=None, public=None)], 0)


if __name__ == '__main__':
    unittest.main()
