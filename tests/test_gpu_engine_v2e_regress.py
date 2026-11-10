"""The regression replayer on synthetic shards: digests, migration, attribution and its special rules."""
from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from typing import NamedTuple
import unittest
from unittest import mock

import numpy as np

HAS_JAX = importlib.util.find_spec('jax') is not None
VOCAB = dict(moves=['_none', 'curse', 'tackle', 'sleeptalk', 'spore', 'struggle', 'encore', 'magiccoat', 'batonpass'],
             abilities=['_none', 'insomnia', 'pressure', 'effectspore'], items=['_none', 'leftovers', 'whiteherb'])
CURSE, TACKLE, SLEEPTALK, SPORE, STRUGGLE, ENCORE, MAGICCOAT, BATONPASS = range(1, 9)
CALLABLE = np.array([False, True, True, False, True, False, True, True, True])


class Trackers(NamedTuple):
    physical_to_public: object
    active_slot: object
    public_volatiles: object
    mon_features: object


class TrackersV2(NamedTuple):                 # a later tree: one appended leaf
    physical_to_public: object
    active_slot: object
    public_volatiles: object
    mon_features: object
    public_order: object


def lanes_state(lanes, seed):
    import jax.numpy as j
    from gpu_engine.state import empty_state
    rng = np.random.default_rng(seed)
    moves = np.full((lanes, 2, 6, 4), TACKLE, np.int32);moves[: lanes // 2, 0, 0, 0] = CURSE
    return empty_state(lanes)._replace(hp=j.asarray(rng.integers(50, 100, (lanes, 2, 6)), j.int32), max_hp=j.full((lanes, 2, 6), 100, j.int32),
        moves=j.asarray(moves), pp=j.full((lanes, 2, 6, 4), 10, j.int32), done=j.asarray(np.arange(lanes) == lanes - 1),
        turn=j.asarray(rng.integers(0, 9, lanes), j.int32))


def trackers(lanes, cls=Trackers):
    import jax.numpy as j
    extra = dict(public_order=j.broadcast_to(j.arange(6, dtype=j.int32), (lanes, 2, 6))) if cls is TrackersV2 else {}
    return cls(physical_to_public=j.broadcast_to(j.arange(6, dtype=j.int32), (lanes, 2, 6)), active_slot=j.zeros((lanes, 2), j.int32),
        public_volatiles=j.zeros((lanes, 2, 6, 3), j.bool_), mon_features={'meanlook_trap': j.zeros((lanes, 2, 6), j.int32)}, **extra)


def result(state, out_trackers):
    import jax.numpy as j
    from gpu_engine.engine_device_public_hooks import PublicStepResult
    from gpu_engine.transition import StepResult
    b = state.hp.shape[0]
    return PublicStepResult(StepResult(state, j.zeros(b, j.bool_), j.zeros(b, j.bool_)), out_trackers)


class Ctx(NamedTuple):
    trackers: object


def base_step(state, actions, t):
    """Deterministic stand-in: actives lose 3 HP per foe move action, lane -2 faints side 1's
    lead, and a seat-0 switch request sets that seat's public active slot to turn % 6."""
    import jax.numpy as j
    rows = j.arange(state.hp.shape[0])
    hit = (actions < 4).astype(j.int32) * 3
    hp = state.hp.at[rows, 0, state.active[:, 0]].add(-hit[:, 1]).at[rows, 1, state.active[:, 1]].add(-hit[:, 0])
    hp = hp.at[rows[-2], 1, 0].set(0)
    switch = actions[:, 0] >= 4
    t = t._replace(active_slot=t.active_slot.at[:, 0].set(j.where(switch, state.turn % 6, t.active_slot[:, 0])))
    return result(state._replace(hp=hp, turn=state.turn + 1), t)


def curse_users(state, actions):
    import jax.numpy as j
    rows = j.arange(state.hp.shape[0]);slot = j.clip(actions[:, 0], 0, 3)
    return (actions[:, 0] < 4) & (state.moves[rows, 0, state.active[:, 0], slot] == CURSE)


def fake_install(c, state, side, use, *, initial=False):
    """PR0's rule on a fake carry: the installed public slot moves to the front."""
    import jax.numpy as j
    t = c.trackers;rows = j.arange(state.hp.shape[0]);s = j.broadcast_to(j.asarray(side), rows.shape)
    slot = t.active_slot[rows, s];current = t.public_order[rows, s]
    position = j.argmax(current == slot[:, None], axis=1)
    swapped = current.at[rows, position].set(current[:, 0]).at[:, 0].set(slot)
    keep = j.asarray(use) & (not initial)
    return c._replace(trackers=t._replace(public_order=t.public_order.at[rows, s].set(j.where(keep[:, None], swapped, current))))


def pre_arrays(L):
    z = lambda *s: np.zeros(s, np.int32)
    s = {k: z(L, 2, 6) for k in ('hp', 'status', 'charge_move', 'encore_move', 'encore_counter', 'ability', 'item', 'confusion_counter')}
    s.update(hp=np.full((L, 2, 6), 50), rest_origin=np.zeros((L, 2, 6), bool), volatile=np.zeros((L, 2, 6), np.uint32), active=z(L, 2),
             pending_residual=np.zeros(L, bool), queued_actor=np.full(L, -1), queued_actions=z(L, 2), last_action_kind=z(L, 2),
             last_action_index=np.full((L, 2), -1), forced_switch=np.zeros((L, 2), bool), moves=np.full((L, 2, 6, 4), TACKLE))
    return s


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class DigestAndRuleTests(unittest.TestCase):
    def test_lane_digest_sees_every_single_element_change(self):
        import jax.numpy as j
        from gpu_engine.v2e_regress import lane_digest
        rng = np.random.default_rng(1)
        for dtype in (np.int32, np.uint32, np.float32, np.int16, np.uint8, np.bool_):
            with self.subTest(dtype=dtype.__name__):
                x = (rng.integers(0, 2, (4, 3, 5)) if dtype is np.bool_ else rng.integers(0, 100, (4, 3, 5))).astype(dtype)
                base = np.asarray(lane_digest(j.asarray(x), 4))
                self.assertEqual(base.shape, (4, 2))
                for lane in range(4):
                    y = x.copy();flat = y[lane].reshape(-1);flat[7] = (not flat[7]) if dtype is np.bool_ else flat[7] + 1
                    changed = (np.asarray(lane_digest(j.asarray(y), 4)) != base).any(-1)
                    self.assertEqual(changed.tolist(), [i == lane for i in range(4)])
        self.assertEqual(np.asarray(lane_digest(j.asarray(np.int32(7)), 3)).shape, (3, 2))   # lane-global leaf

    def test_side_digests_localize_a_change_to_its_side(self):
        import jax.numpy as j
        from gpu_engine.v2e_regress import side_digest
        x = np.arange(4 * 2 * 6, dtype=np.int32).reshape(4, 2, 6)
        base = np.asarray(side_digest('state.hp', j.asarray(x), 4))
        y = x.copy();y[2, 1, 3] += 1
        changed = (np.asarray(side_digest('state.hp', j.asarray(y), 4)) != base).any(-1)
        self.assertEqual(changed.tolist(), [[False, False]] * 2 + [[False, True]] + [[False, False]])
        keys = np.asarray(side_digest('state.keys', j.asarray(x[:, :, 0]), 4))           # not a side axis
        self.assertTrue((keys[:, 0] == keys[:, 1]).all())

    def test_migration_by_path_and_declared_defaults(self):
        from gpu_engine.v2e_regress import MigrationError, migrate
        template = {'state.a': np.zeros((1, 2), np.int32), 'state.new': np.full((1, 6), -1, np.int32), 'actions': np.zeros((1, 2), np.int32),
                    'trackers.public_order': np.zeros((1, 2, 6), np.int32), "trackers.mon_features['wrap_trap_turns']": np.ones((1, 2, 6), np.float32)}
        leaves = [np.ones((3, 2), np.int32), np.zeros((3, 2), np.int32)]
        defaults = {'state.new': lambda n: np.full((n, 6), -1, np.int32)}
        named, filled = migrate(['state.a', 'actions'], leaves, template, defaults=defaults)
        self.assertEqual(sorted(filled), ['state.new', "trackers.mon_features['wrap_trap_turns']", 'trackers.public_order'])
        self.assertTrue((named['state.new'] == -1).all())
        self.assertEqual(named['trackers.public_order'][2, 1].tolist(), list(range(6)))       # declared identity, not lane 0
        self.assertTrue((named["trackers.mon_features['wrap_trap_turns']"] == 0).all())
        with self.assertRaises(MigrationError):migrate(['state.a', 'actions'], [leaves[0].astype(np.int16), leaves[1]], template, defaults=defaults)
        with self.assertRaises(MigrationError):migrate(['state.a', 'actions'], [np.ones((3, 3), np.int32), leaves[1]], template, defaults=defaults)
        with self.assertRaises(MigrationError):migrate(['state.a', 'actions', 'state.gone'], leaves + [leaves[0]], template, defaults=defaults)
        with self.assertRaises(MigrationError):    # a default of the wrong shape
            migrate(['state.a', 'actions'], leaves, template, defaults={'state.new': lambda n: np.zeros((n, 5), np.int32)})
        with self.assertRaises(MigrationError):    # an undeclared new tracker leaf
            migrate(['state.a', 'actions'], leaves, dict(template, **{'trackers.mystery': np.zeros((1, 2), np.int32)}), defaults=defaults)

    def test_public_order_reference_replay(self):
        from gpu_engine.v2e_regress import replay_public_order
        order = np.broadcast_to(np.arange(6), (2, 2, 6)).copy()
        calls = [(np.array([0, 0]), np.array([s, s]), np.array([True, False])) for s in (0, 1, 2, 0)]
        out = replay_public_order(order, calls)
        self.assertEqual(out[0, 0].tolist(), [0, 2, 1, 3, 4, 5])
        self.assertEqual(out[1, 0].tolist(), list(range(6)))           # inactive row unchanged
        self.assertEqual(out[0, 1].tolist(), list(range(6)))           # other seat untouched

    def test_faint_clear_allows_only_the_fainted_mon_and_the_actives_meanlook(self):
        from gpu_engine.v2e_regress import faint_clear_ok
        fainted = np.zeros((2, 6), bool);fainted[1, 0] = True
        p2p = np.broadcast_to(np.arange(6), (2, 6));active_slot = np.array([3, 0])
        ml = "trackers.mon_features['meanlook_trap']"
        base = {'trackers.public_volatiles': np.ones((2, 6, 3), bool), ml: np.ones((2, 6), np.int32)}
        post = {k: v.copy() for k, v in base.items()}
        post['trackers.public_volatiles'][1, 0] = False;post[ml][0, 3] = 0;post[ml][1, 0] = 0
        self.assertTrue(faint_clear_ok(list(post), post, base, fainted, p2p, active_slot))
        bench = {k: v.copy() for k, v in post.items()};bench[ml][0, 4] = 0          # a benched mon's meanlook entry
        self.assertFalse(faint_clear_ok(list(bench), bench, base, fainted, p2p, active_slot))
        post['trackers.public_volatiles'][1, 1] = False                             # another mon
        self.assertFalse(faint_clear_ok(list(post), post, base, fainted, p2p, active_slot))
        self.assertFalse(faint_clear_ok(['trackers.other'], {'trackers.other': np.zeros(1)}, {'trackers.other': np.ones(1)}, fainted, p2p, active_slot))

    def test_involvement_resolves_the_executed_move(self):
        from gpu_engine.v2e_regress import Pre, sleep_clause, sleep_immunity, destiny_bond_bit, white_herb
        L = 9;s = pre_arrays(L)
        s['moves'][0, 0, 0, 1] = CURSE                                  # lane 0: slot 1 chosen
        s['moves'][1, 0, 0, 2] = CURSE;s['encore_move'][1, 0, 0] = CURSE;s['encore_counter'][1, 0, 0] = 2   # lane 1: standing Encore
        s['charge_move'][2, 1, 0] = CURSE                               # lane 2: charge lock (side 1)
        s['moves'][3, 0, 0] = [SLEEPTALK, CURSE, TACKLE, 0]               # lane 3: Sleep Talk may call Curse
        s['pending_residual'][4] = True;s['queued_actor'][4] = 1;s['queued_actions'][4, 1] = 3;s['moves'][4, 1, 0, 3] = CURSE  # resume
        # lane 6: the foe's Encore this turn forces side 0's last executed slot (Curse in slot 2).
        s['moves'][6, 0, 0, 2] = CURSE;s['last_action_kind'][6, 0] = 2;s['last_action_index'][6, 0] = 2;s['moves'][6, 1, 0, 0] = ENCORE
        # lane 7: side 1's Magic Coat reflects side 0's Spore.
        s['moves'][7, 0, 0, 0] = SPORE;s['moves'][7, 1, 0, 0] = MAGICCOAT
        actions = np.zeros((L, 2), np.int32);actions[0, 0] = 1;actions[4, 1] = 5   # lane 4's replacement action is ignored
        named = {f'state.{k}': v for k, v in s.items()};named['actions'] = actions
        p = Pre(named, VOCAB, CALLABLE, np.zeros((L, 2), bool))
        self.assertEqual(p.uses('curse').tolist(), [True, True, True, True, True, False, True, False, False])
        self.assertEqual(p.side_uses('curse')[2].tolist(), [False, True])
        self.assertEqual(p.side_uses('spore')[7].tolist(), [True, True])
        lane6 = dict(s, last_action_kind=s['last_action_kind'].copy());lane6['last_action_kind'][6, 0] = 1   # last action a switch
        self.assertFalse(Pre(dict({f'state.{k}': v for k, v in lane6.items()}, actions=actions), VOCAB, CALLABLE, np.zeros((L, 2), bool)).uses('curse')[6])
        struggle = np.zeros((L, 2), bool);struggle[5, 1] = True
        self.assertTrue(Pre(named, VOCAB, CALLABLE, struggle).uses('struggle')[5])
        # Sleep Clause: lane 0 side 1 has a benched non-Rest sleeper and side 0 uses Spore.
        named['state.moves'] = np.full((L, 2, 6, 4), SPORE);named['state.status'][0, 1, 3] = 5;named['state.status'][1, 1, 3] = 5
        named['state.rest_origin'][1, 1, 3] = True;named['state.status'][2, 1, 0] = 5   # Rest sleeper; the target itself
        p = Pre(named, VOCAB, CALLABLE, np.zeros((L, 2), bool))
        self.assertEqual(sleep_clause(p).any(1).tolist()[:3], [True, False, False])
        self.assertEqual(sleep_clause(p)[0].tolist(), [True, False])
        named['state.ability'][3, 1, 0] = 1
        self.assertTrue(sleep_immunity(Pre(named, VOCAB, CALLABLE, np.zeros((L, 2), bool)))[3, 0])
        named['state.volatile'][5, 0, 2] = 64
        self.assertEqual(destiny_bond_bit(Pre(named, VOCAB, CALLABLE, np.zeros((L, 2), bool))).any(1).tolist(), [False] * 5 + [True] + [False] * 3)
        # An Effect Spore foe can sleep the attacker while the attacker's side holds another sleeper.
        named['state.ability'][8, 1, 0] = 3;named['state.status'][8, 0, 4] = 5;named['state.moves'][8] = TACKLE
        self.assertEqual(sleep_clause(Pre(named, VOCAB, CALLABLE, np.zeros((L, 2), bool)))[8].tolist(), [True, False])
        # White Herb: only a holder that is active or may come in (here: Baton Pass).
        named['state.item'][8, 0, 3] = 2
        self.assertFalse(white_herb(Pre(named, VOCAB, CALLABLE, np.zeros((L, 2), bool)))[8, 0])
        named['state.moves'][8, 0, 0, 0] = BATONPASS
        self.assertTrue(white_herb(Pre(named, VOCAB, CALLABLE, np.zeros((L, 2), bool)))[8, 0])

    def test_pr0_plug_in_predicates_allowances_and_faint_split(self):
        from gpu_engine.v2e_regress import Pre, PLUGINS, ALLOWED, faint_rule, lane_verdict
        ROAR = len(VOCAB['moves']);INTIM, SAND, COLOR = 4, 5, 6
        vocab = dict(VOCAB, moves=VOCAB['moves'] + ['roar'], abilities=VOCAB['abilities'] + ['intimidate', 'sandstream', 'colorchange'])
        L = 8;s = pre_arrays(L);actions = np.zeros((L, 2), np.int32)
        s['types'] = np.ones((L, 2, 6, 2), np.int32);s['base_types'] = np.ones((L, 2, 6, 2), np.int32)
        s['ability'][0, 1, 3] = INTIM;actions[0, 1] = 6                   # lane 0: switches to the holder (bench index 2)
        s['ability'][1, 1, 3] = INTIM                                      # lane 1: a benched holder, no switch
        s['ability'][7, 1, 3] = INTIM;actions[7, 1] = 7                   # lane 7: switches to a non-holder (physical 4)
        s['ability'][2, 0, 2] = SAND;s['forced_switch'][2, 0] = True       # lane 2: a replacement may bring Sand Stream
        s['ability'][3, 1, 4] = INTIM;s['moves'][3, 0, 0, 0] = ROAR        # lane 3: the foe's Roar may drag it in
        s['types'][4, 0, 0] = [5, 0];actions[4, 0] = 4                     # lane 4: a Color Changed active leaves
        s['ability'][5, 1, 0] = COLOR;actions[5, 1] = 5                    # lane 5: a Color Change active leaves (Pursuit)
        s['types'][6, 0, 0] = [5, 0]                                       # lane 6: the overlay stays in
        named = {f'state.{k}': v for k, v in s.items()};named['actions'] = actions
        p = Pre(named, vocab, np.append(CALLABLE, False), np.zeros((L, 2), bool))
        pr0 = PLUGINS['PR0'];no = [False, False]
        self.assertEqual(pr0['intimidate_reveal'](p).tolist(), [[False, True], no, no, [False, True], no, no, no, no])
        self.assertEqual(pr0['weather_reveal'](p).tolist(), [no, no, [True, False], no, no, no, no, no])
        self.assertEqual(pr0['type_revert'](p).tolist(), [no, no, no, no, [True, False], [False, True], no, no])
        self.assertTrue(all(e in ALLOWED for e in pr0))
        self.assertEqual(lane_verdict({('trackers.shown_ability', 1), ('trackers.belief_mask', 1), ('trackers.visible_belief.pin', 1)},
                                      {'intimidate_reveal': [1]}, None)[0], [])
        self.assertEqual(lane_verdict({('trackers.shown_ability', 0), ('state.boosts', 1)}, {'weather_reveal': [1]}, None)[0],
                         [('state.boosts', 1), ('trackers.shown_ability', 0)])
        self.assertEqual(lane_verdict({('trackers.shown_types', 0)}, {'type_revert': [0]}, None)[0], [])
        self.assertEqual(lane_verdict({('trackers.shown_types', 1)}, {'type_revert': [0]}, None)[0], [('trackers.shown_types', 1)])
        # A faint lane-step: the rule removes the cleared leaves; another change still needs an effect.
        vol = 'trackers.public_volatiles';fainted = np.zeros((2, 6), bool);fainted[1, 0] = True
        base = np.zeros((2, 6, 3), bool);base[1, 0, 1] = True;post = np.zeros_like(base)
        entries = {(vol, 1), ('trackers.shown_types', 1)};p2p = np.tile(np.arange(6), (2, 1));slots = np.zeros(2, np.int64)
        self.assertEqual(faint_rule(entries, {vol: post}, {vol: base}, fainted, p2p, slots), ({('trackers.shown_types', 1)}, [vol]))
        wrong = post.copy();wrong[0, 2, 0] = True
        self.assertEqual(faint_rule(entries, {vol: wrong}, {vol: base}, fainted, p2p, slots), (entries, []))

    def test_pr0_meanlook_arrival_predicate_reads_the_tracker_pre_value(self):
        from gpu_engine.v2e_regress import Pre, PLUGINS, lane_verdict, MEANLOOK
        L = 6;s = pre_arrays(L);actions = np.zeros((L, 2), np.int32);MEAN = len(VOCAB['moves'])
        s['trap_source'] = np.full((L, 2, 6), -1, np.int32);flag = np.zeros((L, 2, 6), np.int32)
        flag[5, 1, 4] = 1;actions[5, 0] = 5                                 # lane 5: a flag on a bench entry only
        flag[0, 1, 0] = 1;actions[0, 0] = 5                                 # lane 0: a stale public flag, a switch
        flag[1, 1, 0] = 1                                                   # lane 1: a flag, no arrival
        s['trap_source'][2, 1, 0] = 0;s['forced_switch'][2, 0] = True       # lane 2: an engine link, a replacement
        actions[3, 1] = 6                                                   # lane 3: an arrival, no trap anywhere
        s['moves'][4, 0, 0, 0] = MEAN;actions[4, 1] = 4                     # lane 4: Mean Look this step, a switch
        named = {f'state.{k}': v for k, v in s.items()};named['actions'] = actions;named[MEANLOOK] = flag
        named['trackers.active_slot'] = np.zeros((L, 2), np.int32)
        p = Pre(named, dict(VOCAB, moves=VOCAB['moves'] + ['meanlook']), np.append(CALLABLE, False), np.zeros((L, 2), bool))
        both, no = [True, True], [False, False]
        self.assertEqual(PLUGINS['PR0']['meanlook_arrival'](p).tolist(), [both, no, both, no, both, no])
        self.assertEqual(lane_verdict({(MEANLOOK, 0), (MEANLOOK, 1)}, {'meanlook_arrival': [0, 1]}, None)[0], [])
        self.assertEqual(lane_verdict({('trackers.shown_ability', 1)}, {'meanlook_arrival': [0, 1]}, None)[0],
                         [('trackers.shown_ability', 1)])

    def test_allowed_fields_are_side_aware(self):
        from gpu_engine.v2e_regress import lane_verdict
        effects = {'curse': [0]}
        ok = lane_verdict({('state.boosts', 0), ('state.hp', 1), ('trackers.boosts', 0)}, effects, None)[0]
        self.assertEqual(ok, [])
        bad = lane_verdict({('state.boosts', 1), ('state.turn', None)}, effects, None)[0]
        self.assertEqual(bad, [('state.boosts', 1), ('state.turn', None)])
        self.assertEqual(lane_verdict({('state.boosts', 1)}, effects, None, waive=(r'^state\.boosts$',))[:2], ([], [('state.boosts', 1)]))
        self.assertEqual(lane_verdict({('state.pp', 1), ('state.turn', None)}, {'sleep_clause': [0]}, None)[0], [])
        self.assertEqual(lane_verdict({('state.boosts', 1)}, {'charm': [0]}, None)[0], [])
        self.assertEqual(lane_verdict({('state.boosts', 0)}, {'charm': [0]}, None)[0], [('state.boosts', 0)])

    def test_ko_cascade_accepts_only_ko_consequences(self):
        from gpu_engine.v2e_regress import lane_verdict
        effects = {'curse': [0]}
        # Side 0 had a mon (un)faint; only side-0 mon 3's per-mon state may differ.
        cascade = dict(ko_sides=np.array([True, False]), mon_ok=lambda leaf, side: side == 0)
        entries = {('state.hp', 0), ('state.forced_switch', 0), ('state.done', None), ('state.status', 0), ('trackers.fainted', 0),
                   ('trackers.hp_percent', 1), ("trackers.mon_features['turns_active']", 0)}
        unexpected, _, by_ko = lane_verdict(entries, effects, None, cascade=cascade)
        self.assertEqual(unexpected, [])
        self.assertIn(('state.forced_switch', 0), by_ko);self.assertIn(('state.status', 0), by_ko)
        # The reviewer's case: the foe's status (side 1, no faint there) and a side-less turn change stay unexpected.
        unexpected, _, _ = lane_verdict({('state.status', 1), ('state.turn', None), ('state.hp', 0)}, effects, None, cascade=cascade)
        self.assertEqual(unexpected, [('state.status', 1), ('state.turn', None)])
        refused = dict(cascade, mon_ok=lambda leaf, side: False)                # a non-fainted mon's status on side 0
        self.assertEqual(lane_verdict({('state.status', 0)}, effects, None, cascade=refused)[0], [('state.status', 0)])


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class TwoPassTests(unittest.TestCase):
    """base -> patched end to end on three synthetic shards, without a V2 runtime."""
    LANES = 8

    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine.v2e_capture import leaf_paths
        cls.tmp = Path(tempfile.mkdtemp(prefix='v2e-regress-test-'))
        shards = cls.tmp / 'cap' / 'shards';shards.mkdir(parents=True)
        first = lambda tree: jax.tree.map(lambda x: x[:1], tree)
        state0 = lanes_state(cls.LANES, 0)
        cls.template = (first(state0), j.zeros((1, 2), j.int32), first(trackers(cls.LANES)))
        cls.template_v2 = (cls.template[0], cls.template[1], first(trackers(cls.LANES, TrackersV2)))
        for k in range(3):
            lanes = (cls.LANES, cls.LANES // 2, cls.LANES)[k]
            tree = (lanes_state(lanes, k), j.asarray(np.random.default_rng(k).integers(0, 9, (lanes, 2)), j.int32), trackers(lanes))
            np.savez_compressed(shards / f'shard-{k:06d}.npz', tick=np.int64(2 * k), level=np.int64(0),
                **{f'l{i:03d}': np.asarray(x) for i, x in enumerate(jax.tree.leaves(tree))})
        (cls.tmp / 'cap' / 'capture.json').write_text(json.dumps(dict(shards=3, paths=leaf_paths(cls.template))))
        cls.context = dict(prepared=SimpleNamespace(engine_tables=SimpleNamespace(sleep_talk=SimpleNamespace(
            callable=CALLABLE))), pool=SimpleNamespace(metadata=lambda: dict(vocabulary=VOCAB)))
        from gpu_engine import v2e_regress
        cls.base = cls.tmp / 'base'
        v2e_regress.base_pass(cls.args(out=cls.base, pad_lanes=cls.LANES), step_override=base_step,
            prepared=(cls.context, cls.template))
        cls.manifest = json.loads((cls.base / 'manifest.json').read_text())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    @classmethod
    def args(cls, **extra):
        values = dict(config=None, reference_run=None, shards=str(cls.tmp / 'cap' / 'shards'), limit=0, arm='B', allow_dirty=True,
                      integration=None, allow_base_mismatch=True, waive=None)
        values.update(extra);return Namespace(**values)

    def patched(self, name, step, *, pr='A1', template=None, faint_clear=False, **extra):
        from gpu_engine import v2e_regress
        out = self.tmp / name
        code = v2e_regress.patched_pass(self.args(out=out, base=str(self.base), pr=pr, faint_clear=faint_clear, **extra),
            step_override=step, prepared=(self.context, template or self.template))
        return code, json.loads((out / 'report.json').read_text()), [json.loads(x) for x in (out / 'changed.jsonl').read_text().splitlines()]

    def test_base_pass_manifest(self):
        m = self.manifest
        self.assertEqual(m['lane_steps'], 2 * self.LANES + self.LANES // 2);self.assertEqual(m['live_lane_steps'], m['lane_steps'] - 3)
        self.assertIn('state.hp', m['output_names']);self.assertIn('trackers.public_volatiles', m['faint_leaves'])
        self.assertEqual(dict(zip(m['output_names'], m['side_axis']))['state.turn'], False)
        digest = np.load(self.base / 'digest-shard-000001.npz')
        self.assertEqual(digest['digests'].shape, (self.LANES // 2, len(m['output_names']), 2, 2))
        self.assertFalse(m['partial'])

    def test_base_against_base_is_unchanged(self):
        code, report, listing = self.patched('same', base_step)
        self.assertEqual(code, 0);self.assertEqual(listing, [])
        c = report['counts']
        self.assertEqual((c['changed_and_involved'], c['changed_but_uninvolved'], c['done_skipped']), (0, 0, 3))
        self.assertEqual(c['unchanged'], c['lane_steps'] - 3)
        self.assertGreater(report['by_effect']['curse']['involved_unchanged'], 0)
        self.assertFalse(report['partial']);self.assertTrue(report['allow_base_mismatch'])

    def test_perturbed_engine_is_caught_as_changed_but_uninvolved(self):
        def perturbed(state, actions, t):
            out = base_step(state, actions, t)
            s = out.step.state
            return result(s._replace(hp=s.hp.at[:, 1, 5].add((state.turn % 3 == 0).astype(s.hp.dtype))), out.trackers)
        code, report, listing = self.patched('perturbed', perturbed)
        self.assertEqual(code, 1);self.assertFalse(report['passed'])
        self.assertGreater(report['counts']['changed_but_uninvolved'], 0)
        self.assertEqual(report['uninvolved_leaves'].get('state.hp'), report['counts']['changed_but_uninvolved'])
        self.assertTrue(all(x['leaves'] == ['state.hp'] for x in listing))

    def test_change_on_involved_lanes_is_attributed_and_field_checked(self):
        def curse_boosts(state, actions, t, *, side=0):
            out = base_step(state, actions, t);s = out.step.state
            uses = curse_users(state, actions).astype(s.boosts.dtype)
            return result(s._replace(boosts=s.boosts.at[:, side, 0, 0].add(uses)), out.trackers)
        code, report, listing = self.patched('involved', curse_boosts)
        self.assertEqual(code, 0, report['counts']);self.assertEqual(report['counts']['changed_but_uninvolved'], 0)
        self.assertGreater(report['by_effect']['curse']['changed'], 0)
        self.assertEqual(set(report['by_effect']['curse']['changed_fields']), {'state.boosts@user'})
        self.assertTrue(all(x['involved'] == {'curse': [0]} and x['kind'] == 'involved' for x in listing))
        # The same lanes changing the target's boosts: involved, but in a field Curse may not touch.
        code, report, listing = self.patched('unexpected', lambda s, a, t: curse_boosts(s, a, t, side=1))
        self.assertEqual(code, 1);self.assertGreater(report['counts']['changed_involved_unexpected_fields'], 0)
        self.assertEqual(set(report['by_effect']['curse']['unexpected_fields']), {'state.boosts@target'})
        code, report, _ = self.patched('waived', lambda s, a, t: curse_boosts(s, a, t, side=1), waive=[r'^state\.boosts$'])
        self.assertEqual(code, 0);self.assertGreater(report['counts']['waived'], 0)
        code, report, _ = self.patched('involved-none', curse_boosts, pr='none')
        self.assertEqual(code, 1);self.assertGreater(report['counts']['changed_but_uninvolved'], 0)

    def test_ko_cascade_is_limited_to_ko_consequences(self):
        import jax.numpy as j
        def curse_ko(state, actions, t, *, plant=False):
            out = base_step(state, actions, t);s = out.step.state
            uses = curse_users(state, actions)
            hp = s.hp.at[:, 0, 3].set(j.where(uses, 0, s.hp[:, 0, 3]))   # a bench KO the base never had (Curse side)
            status = s.status.at[:, 0, 3].set(j.where(uses, 1, s.status[:, 0, 3]))   # the fainted mon's own state
            if plant:status = status.at[:, 1, 0].set(j.where(uses, 1, status[:, 1, 0]))   # the foe paralysed: not a KO consequence
            return result(s._replace(hp=hp, status=status), out.trackers)
        code, report, listing = self.patched('cascade', curse_ko)
        self.assertEqual(code, 0, report['counts']);self.assertGreater(report['counts']['ko_cascade'], 0)
        self.assertTrue(all(x['ko_cascade'] for x in listing))
        # A2 review L1/L2: phase shifts are counted and flagged per lane; link leaves use the base arrays.
        self.assertEqual(report['link_rule'], 'per-mon');self.assertIn('state.last_move', self.manifest['ko_leaves'])
        self.assertEqual(report['counts']['phase_shift'], sum(x['phase_shift'] for x in listing))
        self.assertEqual(set(report['by_effect']['curse']['ko_cascade_non_hp_fields']), {'state.status@user'})
        code, report, listing = self.patched('cascade-plant', lambda s, a, t: curse_ko(s, a, t, plant=True))
        self.assertEqual(code, 1);self.assertGreater(report['counts']['changed_involved_unexpected_fields'], 0)
        self.assertEqual(set(report['by_effect']['curse']['unexpected_fields']), {'state.status@target'})

    def test_partial_runs_and_plugin_supersets_are_marked(self):
        code, report, _ = self.patched('partial', base_step, pr='A1,A2', limit=1)
        self.assertEqual(code, 2);self.assertTrue(report['partial']);self.assertIsNone(report['passed'])
        self.assertTrue(report['warnings']);self.assertEqual(report['plug_ins'], ['A1', 'A2'])
        with self.assertRaises(SystemExit):self.patched('undeclared', base_step, pr='R')    # the PR's own plug-in must declare
        from gpu_engine import v2e_regress
        with self.assertRaises(SystemExit):         # a base pass that is not this PR's merge base
            v2e_regress.patched_pass(self.args(out=self.tmp / 'nobase', base=str(self.base), pr='A1', faint_clear=False,
                allow_base_mismatch=False), step_override=base_step, prepared=(self.context, self.template))

    def test_new_tracker_leaf_is_migrated_and_checked_by_the_public_order_replay(self):
        from gpu_engine import engine_device_public_hooks as hooks

        def step(state, actions, t, *, wrong=False):
            out = base_step(state, actions, t)
            c = hooks.install(Ctx(out.trackers), out.step.state, 0, actions[:, 0] >= 4)
            new = c.trackers
            if wrong:new = new._replace(public_order=new.public_order.at[:, 1, 5].set(0))
            return result(out.step.state, new)

        with mock.patch.object(hooks, 'install', fake_install):
            code, report, _ = self.patched('order', step, template=self.template_v2)
            self.assertEqual(code, 0, report);self.assertTrue(report['public_order_rule'])
            self.assertIn('trackers.public_order', report['migrated_new_inputs'])
            self.assertEqual(report['counts']['changed_but_uninvolved'], 0)
            code, report, _ = self.patched('order-wrong', lambda s, a, t: step(s, a, t, wrong=True), template=self.template_v2)
            self.assertEqual(code, 1);self.assertIn('trackers.public_order', report['uninvolved_leaves'])

    def test_base_pass_migrates_a_capture_that_predates_a_tracker_leaf(self):
        from gpu_engine import v2e_regress
        base = self.tmp / 'base-v2'
        code = v2e_regress.base_pass(self.args(out=base, pad_lanes=self.LANES), step_override=base_step,
            prepared=(self.context, self.template_v2))
        self.assertEqual(code, 0)
        m = json.loads((base / 'manifest.json').read_text())
        self.assertEqual(m['migrated_new_inputs'], ['trackers.public_order'])
        self.assertIn('trackers.public_order', m['input_names']);self.assertIn('trackers.public_order', m['output_names'])
        self.assertEqual(self.manifest['migrated_new_inputs'], [])
        # A later PR's patched pass on that base compares public_order bitwise, not by replay.
        def run(name, step):
            out = self.tmp / name
            code = v2e_regress.patched_pass(self.args(out=out, base=str(base), pr='A1', faint_clear=False),
                step_override=step, prepared=(self.context, self.template_v2))
            return code, json.loads((out / 'report.json').read_text())
        code, report = run('on-base-v2', base_step)
        self.assertEqual(code, 0, report['counts']);self.assertFalse(report['public_order_rule'])
        self.assertEqual(report['counts']['changed_but_uninvolved'], 0)
        def moved(state, actions, t):
            out = base_step(state, actions, t)
            return result(out.step.state, out.trackers._replace(public_order=out.trackers.public_order.at[:, 1, 5].set(0)))
        code, report = run('on-base-v2-moved', moved)
        self.assertEqual(code, 1);self.assertIn('trackers.public_order', report['uninvolved_leaves'])


if __name__ == '__main__':
    unittest.main()
