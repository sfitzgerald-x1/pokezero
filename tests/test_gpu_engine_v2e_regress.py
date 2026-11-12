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
    s = empty_state(lanes)
    # Side 1's lead is Choice-locked into Tackle before every step (#1917 review): any holder-based allowance (a locked
    # or Choice-holding active, under any name) then involves every lane, which the clear_volatile_rule test catches.
    return s._replace(hp=j.asarray(rng.integers(50, 100, (lanes, 2, 6)), j.int32), max_hp=j.full((lanes, 2, 6), 100, j.int32),
        moves=j.asarray(moves), pp=j.full((lanes, 2, 6, 4), 10, j.int32), done=j.asarray(np.arange(lanes) == lanes - 1),
        turn=j.asarray(rng.integers(0, 9, lanes), j.int32), choice_move=s.choice_move.at[:, 1, 0].set(TACKLE))


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
        # The saved effect block (M1 GPU pass): a cascade side accepts it only where this tree's block is the derivation.
        derived = dict(cascade, effect_ok=np.array([True, False]))
        unexpected, _, by_ko = lane_verdict({('state.effect_draws', 0), ('state.effect_draws', 1)}, effects, None, cascade=derived)
        self.assertEqual((unexpected, by_ko), ([('state.effect_draws', 1)], [('state.effect_draws', 0)]))
        self.assertEqual(lane_verdict({('state.effect_draws', 0)}, effects, None, cascade=cascade)[0], [('state.effect_draws', 0)])
        self.assertEqual(lane_verdict({('state.effect_draws', 0)}, effects, None)[0], [('state.effect_draws', 0)])
        # A moved KO's Color Change retype (M1 GPU pass): accepted only where retype_ok says the side's types differ only
        # at a holder whose KO moved (M1 review: per mon, from the base's stored arrays).
        retyped = dict(cascade, retype_ok=lambda leaf, side: side == 1)
        unexpected, _, by_ko = lane_verdict({('state.types', 0), ('state.types', 1), ('trackers.shown_types', 1)}, effects, None,
                                            cascade=retyped)
        self.assertEqual(unexpected, [('state.types', 0)])
        self.assertEqual(sorted(by_ko), [('state.types', 1), ('trackers.shown_types', 1)])
        self.assertEqual(lane_verdict({('state.types', 1), ('trackers.shown_types', 1)}, effects, None, cascade=cascade)[0],
                         [('state.types', 1), ('trackers.shown_types', 1)])


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
        self.assertEqual(report['counts']['terminal'], sum(x['terminal'] for x in listing))
        self.assertEqual(set(report['by_effect']['curse']['ko_cascade_non_hp_fields']), {'state.status@user'})
        code, report, listing = self.patched('cascade-plant', lambda s, a, t: curse_ko(s, a, t, plant=True))
        self.assertEqual(code, 1);self.assertGreater(report['counts']['changed_involved_unexpected_fields'], 0)
        self.assertEqual(set(report['by_effect']['curse']['unexpected_fields']), {'state.status@target'})

    def test_clear_volatile_rule_explains_only_a_fainted_mons_zeroed_lock(self):
        """B: a faint's clearVolatile zeroes the fainted mon's Choice lock and confusion: a special rule with --pr B,
        not involvement. Against a base that locks and confuses side 1's lead, zeroing them on the lead that faints
        this step (base_step's lane -2) is rule-explained; zeroing them on a living lead stays uninvolved; without
        the B plug-in the rule is off."""
        import jax.numpy as j
        from gpu_engine import v2e_regress
        def locked(state, actions, t):
            out = base_step(state, actions, t);s = out.step.state
            return result(s._replace(choice_move=s.choice_move.at[:, 1, 0].set(TACKLE),
                confusion_counter=s.confusion_counter.at[:, 1, 0].set(3)), out.trackers)
        base = self.tmp / 'base-locked'
        v2e_regress.base_pass(self.args(out=base, pad_lanes=self.LANES), step_override=locked, prepared=(self.context, self.template))
        def cleared(state, actions, t, *, everywhere=False):
            out = locked(state, actions, t);s = out.step.state
            gone = ((state.hp[:, 1, 0] > 0) & (s.hp[:, 1, 0] <= 0)) | everywhere
            return result(s._replace(choice_move=s.choice_move.at[:, 1, 0].set(j.where(gone, 0, s.choice_move[:, 1, 0])),
                confusion_counter=s.confusion_counter.at[:, 1, 0].set(j.where(gone, 0, s.confusion_counter[:, 1, 0]))), out.trackers)
        def run(name, step, pr):
            out = self.tmp / name
            code = v2e_regress.patched_pass(self.args(out=out, base=str(base), pr=pr, faint_clear=False), step_override=step,
                prepared=(self.context, self.template))
            return code, json.loads((out / 'report.json').read_text()), [json.loads(x) for x in (out / 'changed.jsonl').read_text().splitlines()]
        code, report, listing = run('clear-fainted', cleared, 'B')
        c = report['counts'];fainted = c['clear_volatile_leaves']
        self.assertEqual(code, 0, c);self.assertTrue(report['clear_volatile_rule']);self.assertGreater(fainted, 0)
        self.assertEqual((c['rule_explained'], c['changed_but_uninvolved'], c['changed_and_involved']), (fainted, 0, 0))
        self.assertTrue(all(x['kind'] == 'rule' and x['clear_volatile_rule'] == ['state.choice_move@1', 'state.confusion_counter@1']
                            for x in listing))
        code, report, _ = run('clear-fainted-a1', cleared, 'A1')                       # the rule belongs to B
        self.assertEqual(code, 1);self.assertFalse(report['clear_volatile_rule'])
        self.assertEqual(report['counts']['changed_but_uninvolved'], fainted)
        code, report, listing = run('clear-living', lambda s, a, t: cleared(s, a, t, everywhere=True), 'B')
        c = report['counts'];living = c['lane_steps'] - c['done_skipped'] - fainted
        self.assertEqual(code, 1);self.assertEqual((c['clear_volatile_leaves'], c['changed_but_uninvolved']), (fainted, living))
        self.assertEqual(report['uninvolved_leaves'], {'state.choice_move': living, 'state.confusion_counter': living})
        # Every pre-state has side 1's lead locked (lanes_state), so a holder-based involvement (the old choice_holder
        # predicate with choice_faint's allowance, removed in 4aa184bb) involves every living lane and hides a planted
        # clear of the living lead's lock: re-added, that run passes, so this test's uninvolved count above would be 0.
        def lock_only(state, actions, t):
            out = locked(state, actions, t);s = out.step.state
            return result(s._replace(choice_move=s.choice_move.at[:, 1, 0].set(0)), out.trackers)
        code, report, _ = run('clear-lock-living', lock_only, 'B')
        c = report['counts']
        self.assertEqual(code, 1);self.assertEqual(c['changed_but_uninvolved'], living)
        def choice_holder(p):
            items = p.vocab['items'];ids = [items.index(n) for n in ('choiceband', 'choicespecs', 'choicescarf') if n in items]
            return (p.active('choice_move') > 0) | np.isin(p.active('item'), ids)
        with mock.patch.dict(v2e_regress.PLUGINS, {'B': dict(v2e_regress.PLUGINS['B'], choice_faint=choice_holder)}), \
                mock.patch.dict(v2e_regress.ALLOWED, {'choice_faint': dict(user=('choice',), target=(), glob=())}):
            code, report, _ = run('clear-lock-living-oldpred', lock_only, 'B')
        c = report['counts']
        self.assertEqual(code, 0, c);self.assertEqual(c['changed_but_uninvolved'], 0)        # hidden: the reason it is gone

    def test_partial_runs_and_plugin_supersets_are_marked(self):
        code, report, _ = self.patched('partial', base_step, pr='A1,A2', limit=1)
        self.assertEqual(code, 2);self.assertTrue(report['partial']);self.assertIsNone(report['passed'])
        self.assertTrue(report['warnings']);self.assertEqual(report['plug_ins'], ['A1', 'A2'])
        with self.assertRaises(SystemExit):self.patched('undeclared', base_step, pr='C')   # the PR's own plug-in must declare
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



def clocked(step, ran):
    """A stand-in step that runs the residual weather clock with `ran(state, actions)` as its eligibility,
    exactly where transition.step_with_chance reads residual_ready (the base pass records it)."""
    def run(state, actions, t):
        from gpu_engine import weather
        weather.residual_clock(state.weather, state.weather_counter, eligible=ran(state, actions))
        return step(state, actions, t)
    return run


def pending_step(state, actions, t):
    """base_step whose even-turn lanes defer their residual (a suspended turn)."""
    out = base_step(state, actions, t)
    return result(out.step.state._replace(pending_residual=(state.turn % 2 == 0) & ~state.done), out.trackers)


def even(state, actions):
    return state.turn % 2 == 0


def saved_block(state, post):
    """The effect block the design saves on a deferring step (transition.effect_chance of the split key), else 0."""
    import jax
    import jax.numpy as j
    from gpu_engine import transition
    block = transition.effect_chance(jax.vmap(lambda k: jax.random.split(k, 2))(state.keys)[:, 1])
    return j.where(post.pending_residual[:, None, None], block, 0)


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class EffectAndResidualRuleTests(unittest.TestCase):
    """R: a capture predating state.effect_draws, the saved effect block checked against the design derivation;
    A2 review L1: the base pass records whether the residual ran, and a terminal step widens only where it differs."""
    LANES = 8

    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine.state import BattleState
        from gpu_engine.v2e_capture import leaf_paths
        cls.tmp = Path(tempfile.mkdtemp(prefix='v2e-regress-r-test-'))
        legacy_type = NamedTuple('LegacyState', [(n, object) for n in BattleState._fields if n != 'effect_draws'])
        legacy = lambda s: legacy_type(**{n: getattr(s, n) for n in legacy_type._fields})
        shards = cls.tmp / 'cap' / 'shards';shards.mkdir(parents=True)
        first = lambda tree: jax.tree.map(lambda x: x[:1], tree)
        cls.old = (first(legacy(lanes_state(cls.LANES, 0))), j.zeros((1, 2), j.int32), first(trackers(cls.LANES)))
        cls.new = (first(lanes_state(cls.LANES, 0)), j.zeros((1, 2), j.int32), first(trackers(cls.LANES)))
        for k in range(2):
            tree = (legacy(lanes_state(cls.LANES, k)), j.asarray(np.random.default_rng(k).integers(0, 9, (cls.LANES, 2)), j.int32),
                    trackers(cls.LANES))
            np.savez_compressed(shards / f'shard-{k:06d}.npz', tick=np.int64(k), level=np.int64(0),
                **{f'l{i:03d}': np.asarray(x) for i, x in enumerate(jax.tree.leaves(tree))})
        (cls.tmp / 'cap' / 'capture.json').write_text(json.dumps(dict(shards=2, paths=leaf_paths(cls.old))))
        cls.context = dict(prepared=SimpleNamespace(engine_tables=SimpleNamespace(sleep_talk=SimpleNamespace(
            callable=CALLABLE))), pool=SimpleNamespace(metadata=lambda: dict(vocabulary=VOCAB)))
        from gpu_engine import v2e_regress
        cls.base = cls.tmp / 'base'
        code = v2e_regress.base_pass(TwoPassTests.args.__func__(cls, out=cls.base, pad_lanes=cls.LANES),
            step_override=clocked(pending_step, even), prepared=(cls.context, cls.old))
        assert code == 0
        cls.manifest = json.loads((cls.base / 'manifest.json').read_text())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def patched(self, name, step, pr='A1'):
        from gpu_engine import v2e_regress
        out = self.tmp / name
        code = v2e_regress.patched_pass(TwoPassTests.args.__func__(type(self), out=out, base=str(self.base), pr=pr, faint_clear=False),
            step_override=step, prepared=(self.context, self.new))
        return code, json.loads((out / 'report.json').read_text()), [json.loads(x) for x in (out / 'changed.jsonl').read_text().splitlines()]

    def inputs(self):
        import jax.numpy as j
        state = lanes_state(self.LANES, 0)
        return state, j.zeros((self.LANES, 2), j.int32), trackers(self.LANES)

    def test_base_pass_records_the_residual_and_the_actives(self):
        self.assertTrue(self.manifest['residual_ran_recorded']);self.assertTrue(self.manifest['post_active_recorded'])
        digest = np.load(self.base / 'digest-shard-000000.npz')
        state = lanes_state(self.LANES, 0)
        self.assertEqual(digest['residual_ran'].tolist(), (np.asarray(state.turn) % 2 == 0).tolist())
        self.assertEqual(digest['post_active'].shape, (self.LANES, 2))
        self.assertNotIn('state.effect_draws', self.manifest['output_names'])

    def test_saved_effect_block_follows_the_design_derivation(self):
        import jax
        import jax.numpy as j
        from gpu_engine import transition

        def saving(state, actions, t, *, wrong=False):
            out = pending_step(state, actions, t);s = out.step.state
            block = transition.effect_chance(jax.vmap(lambda k: jax.random.split(k, 2))(state.keys)[:, 1])
            block = (block + 1) % 2 if wrong else block
            return result(s._replace(effect_draws=j.where(s.pending_residual[:, None, None], block, 0)), out.trackers)
        np.testing.assert_array_equal(np.asarray(saving(*self.inputs())[0].state.effect_draws),
            np.asarray(saved_block(self.inputs()[0], saving(*self.inputs())[0].state)))
        code, report, listing = self.patched('effect', clocked(saving, even))
        self.assertEqual(code, 0, report['counts']);self.assertTrue(report['effect_rule'])
        self.assertIn('state.effect_draws', report['migrated_new_inputs'])
        self.assertGreater(report['counts']['effect_draws_saved'], 0);self.assertEqual(listing, [])
        code, report, _ = self.patched('effect-wrong', clocked(lambda s, a, t: saving(s, a, t, wrong=True), even))
        self.assertEqual(code, 1);self.assertIn('state.effect_draws', report['uninvolved_leaves'])
        code, report, _ = self.patched('effect-unsaved', clocked(pending_step, even))   # deferred lanes left at zero
        self.assertEqual(code, 1);self.assertIn('state.effect_draws', report['uninvolved_leaves'])

    def test_a_phase_shift_changes_the_saved_effect_block_only_by_the_derivation(self):
        """M1 GPU pass (2 lanes of 1.1M): once the base pass has state.effect_draws it is compared bitwise, and a
        phase shift (the deferral moved) saves or drops the block. Accepted per side only where this tree's block
        is the design derivation (effect_ok); a wrong block on the same lanes stays unexpected."""
        import jax.numpy as j
        from gpu_engine import v2e_regress

        def saving(state, actions, t, *, shift=False, wrong=False):
            out = pending_step(state, actions, t);s = out.step.state
            uses = curse_users(state, actions)
            if shift:s = s._replace(pending_residual=s.pending_residual ^ uses)      # Curse lanes move the deferral
            block = saved_block(state, s)
            if wrong:block = j.where(uses[:, None, None], (block + 1) % 2, block)
            return result(s._replace(effect_draws=block), out.trackers)
        args = lambda **extra: TwoPassTests.args.__func__(type(self), **extra)
        base = self.tmp / 'base-with-effect'
        code = v2e_regress.base_pass(args(out=base, pad_lanes=self.LANES), step_override=clocked(saving, even),
            prepared=(self.context, self.new))
        self.assertEqual(code, 0)
        self.assertIn('state.effect_draws', json.loads((base / 'manifest.json').read_text())['output_names'])

        def run(name, **kind):
            out = self.tmp / name
            code = v2e_regress.patched_pass(args(out=out, base=str(base), pr='A1', faint_clear=False),
                step_override=clocked(lambda s, a, t: saving(s, a, t, **kind), even), prepared=(self.context, self.new))
            return code, json.loads((out / 'report.json').read_text()), [json.loads(x) for x in (out / 'changed.jsonl').read_text().splitlines()]
        code, report, listing = run('effect-shift', shift=True)
        self.assertEqual(code, 0, report['counts']);self.assertFalse(report['effect_rule'])
        shifted = [x for x in listing if x['phase_shift']]
        self.assertTrue(shifted);self.assertEqual(len(shifted), len(listing))
        self.assertTrue(all({'state.effect_draws@0', 'state.effect_draws@1'} <= set(x['ko_consequences']) for x in shifted))
        code, report, _ = run('effect-shift-wrong', shift=True, wrong=True)
        self.assertEqual(code, 1)
        self.assertEqual(set(report['by_effect']['curse']['unexpected_fields']), {'state.effect_draws@user', 'state.effect_draws@target'})

    def test_a_moved_ko_retypes_only_a_color_change_holder(self):
        """M1 GPU pass (shard 176 lane 1687: a Trick-given Choice Band KOs Kecleon in the action phase, so Color Change
        never retypes it; the base retyped it and it fainted at the residual). A cascade lane accepts a changed type only
        on a side whose Color Change holder fainted this step here or (un)fainted relative to the base."""
        import jax
        import jax.numpy as j
        from gpu_engine import v2e_regress
        from gpu_engine.v2e_capture import leaf_paths
        vocab = dict(VOCAB, abilities=VOCAB['abilities'] + ['colorchange'])
        cap = self.tmp / 'cap-cc';shards = cap / 'shards';shards.mkdir(parents=True)
        for k in range(2):
            s = lanes_state(self.LANES, k)
            s = s._replace(ability=s.ability.at[:, 0, 3].set(len(vocab['abilities']) - 1))   # side 0's bench mon 3 holds it
            tree = (s, j.asarray(np.random.default_rng(k).integers(0, 9, (self.LANES, 2)), j.int32), trackers(self.LANES))
            np.savez_compressed(shards / f'shard-{k:06d}.npz', tick=np.int64(k), level=np.int64(0),
                **{f'l{i:03d}': np.asarray(x) for i, x in enumerate(jax.tree.leaves(tree))})
        (cap / 'capture.json').write_text(json.dumps(dict(shards=2, paths=leaf_paths(self.new))))
        context = dict(self.context, pool=SimpleNamespace(metadata=lambda: dict(vocabulary=vocab)))
        args = lambda **extra: TwoPassTests.args.__func__(type(self), shards=str(shards), **extra)

        def moved(state, actions, t, *, side=0, mon=3):
            out = base_step(state, actions, t);s = out.step.state
            uses = curse_users(state, actions)
            hp = s.hp.at[:, side, 3].set(j.where(uses, 0, s.hp[:, side, 3]))                # a bench KO the base never had
            types = s.types.at[:, side, mon, 0].set(j.where(uses, 9, s.types[:, side, mon, 0]))  # and a type differs
            return result(s._replace(hp=hp, types=types), out.trackers)
        base = self.tmp / 'base-cc'
        code = v2e_regress.base_pass(args(out=base, pad_lanes=self.LANES), step_override=clocked(base_step, even),
            prepared=(context, self.new))
        self.assertEqual(code, 0)

        def run(name, side, mon=3):
            out = self.tmp / name
            code = v2e_regress.patched_pass(args(out=out, base=str(base), pr='A1', faint_clear=False),
                step_override=clocked(lambda s, a, t: moved(s, a, t, side=side, mon=mon), even), prepared=(context, self.new))
            return code, json.loads((out / 'report.json').read_text()), [json.loads(x) for x in (out / 'changed.jsonl').read_text().splitlines()]
        code, report, listing = run('retype-holder', 0)
        self.assertEqual(code, 0, report['counts'])
        self.assertTrue(listing);self.assertTrue(all('state.types@0' in x['ko_consequences'] for x in listing))
        code, report, _ = run('retype-no-holder', 1)                                # side 1's mon 3 holds no Color Change
        self.assertEqual(code, 1);self.assertIn('state.types@target', report['by_effect']['curse']['unexpected_fields'])
        # M1 review: the holder's KO moved, but the type changed on another mon of its side (mon 4): per mon, flagged.
        self.assertIn('state.types', json.loads((base / 'manifest.json').read_text())['witness_leaves'])
        code, report, _ = run('retype-other-mon', 0, mon=4)
        self.assertEqual(code, 1);self.assertIn('state.types@user', report['by_effect']['curse']['unexpected_fields'])

    def test_terminal_steps_widen_only_where_the_residual_ran_in_one_tree(self):
        import jax.numpy as j

        def ending(state, actions, t, *, item=False):
            out = pending_step(state, actions, t);s = out.step.state
            uses = curse_users(state, actions)
            s = s._replace(done=s.done | uses,
                truant_loaf=s.truant_loaf.at[:, 1, 0].set(j.where(uses, ~s.truant_loaf[:, 1, 0], s.truant_loaf[:, 1, 0])))
            if item:s = s._replace(item=s.item.at[:, 1, 0].set(j.where(uses, 2, s.item[:, 1, 0])))
            return result(s._replace(effect_draws=saved_block(state, s)), out.trackers)
        code, report, listing = self.patched('terminal-same', clocked(ending, even))
        curse = [x for x in listing if 'curse' in x['involved']]
        self.assertTrue(curse);self.assertTrue(all(x['terminal'] and not x['terminal_residual'] for x in curse))
        self.assertTrue(all(not x['unexpected'] for x in curse), [x['unexpected'] for x in curse])
        self.assertEqual(report['residual_rule'], 'per-lane');self.assertEqual(report['counts']['terminal_residual'], 0)
        code, report, listing = self.patched('terminal-same-item', clocked(lambda s, a, t: ending(s, a, t, item=True), even))
        curse = [x for x in listing if 'curse' in x['involved']]
        self.assertTrue(all('state.item@1' in x['unexpected'] for x in curse))
        odd = lambda s, a: ~even(s, a)
        code, report, listing = self.patched('terminal-shift', clocked(lambda s, a, t: ending(s, a, t, item=True), odd))
        curse = [x for x in listing if 'curse' in x['involved']]
        self.assertTrue(all(x['terminal_residual'] and not x['unexpected'] for x in curse))
        self.assertEqual(report['counts']['terminal_residual'], len(curse))


# M1 review: planted bugs over one synthetic capture with Hyper Beam and Trick lanes.
M1_VOCAB = dict(moves=['_none', 'hyperbeam', 'trick', 'tackle', 'splash'], abilities=['_none'],
                items=['_none', 'leftovers', 'lumberry', 'liechiberry', 'choiceband'])
HB, TRICK = 1, 2
LEFTOVERS, LUM, LIECHI, BAND = 1, 2, 3, 4
M1_LANES = 8           # lanes 0-1 Hyper Beam hits, 2-3 misses; 4 Liechi to the Tricker, 5 Lum to a confused foe, 6 a plain swap


def m1_trackers(lanes):
    import jax.numpy as j
    from gpu_engine.public_active import NAMES
    return Trackers(physical_to_public=j.broadcast_to(j.arange(6, dtype=j.int32), (lanes, 2, 6)), active_slot=j.zeros((lanes, 2), j.int32),
        public_volatiles=j.zeros((lanes, 2, 6, len(NAMES)), j.bool_).at[5, 1, 0, NAMES.index('confusion')].set(True),
        mon_features={'meanlook_trap': j.zeros((lanes, 2, 6), j.int32)})


def m1_state(seed):
    import jax.numpy as j
    s = lanes_state(M1_LANES, seed)
    moves = np.full((M1_LANES, 2, 6, 4), 3, np.int32);moves[:, 0, 0] = [HB, TRICK, 3, 3]
    item = np.zeros((M1_LANES, 2, 6), np.int32);item[:, 0, 0] = LEFTOVERS;item[:, 1, 0] = LEFTOVERS
    item[4, 1, 0] = LIECHI;item[5, 0, 0] = LUM;item[6, 1, 0] = BAND
    confusion = np.zeros((M1_LANES, 2, 6), np.int32);confusion[5, 1, 0] = 3
    return s._replace(moves=j.asarray(moves), item=j.asarray(item), confusion_counter=j.asarray(confusion),
                      choice_move=s.choice_move.at[:, 1, 0].set(0))


def m1_actions():
    return np.array([[0, 0]] * 4 + [[1, 0]] * 3 + [[0, 0]], np.int32)


def m1_step(state, actions, t, *, recharge=True, miss_recharge=False, bag=True, trick=True, wide=False):
    """A stand-in engine: side 1's Tackle takes 3 HP from side 0's active; side 0's Hyper Beam (lanes 0-3) hits for 10 on
    lanes 0-1 (the device move and damage hooks see it; on lane 1 the foe made a Substitute earlier in the step and the
    hit only lowers it, which the step's pre/post state cannot show) and misses on 2-3; recharge sets the counter and
    the mustrecharge bag bit; trick swaps the actives' items: lane 4's Tricker eats the Liechi (+1 Atk), lane 5's
    confused foe eats the Lum (confusion and its bag bit end). The base pass is recharge=False, trick=False."""
    import jax.numpy as j
    from gpu_engine import engine_device_public_hooks as hooks
    from gpu_engine.public_active import NAMES
    lanes = state.hp.shape[0];rows = j.arange(lanes)
    s = state;hp = s.hp.at[:, 0, 0].add(-3)
    beam = (actions[:, 0] == 0) & (s.moves[:, 0, 0, 0] == HB) & ~s.done
    hit = beam & (rows < 2);behind = hit & (rows == 1)
    s = s._replace(substitute_hp=s.substitute_hp.at[:, 1, 0].set(j.where(rows == 1, 50, s.substitute_hp[:, 1, 0])))
    hooks.move(None, s, j.zeros(lanes, j.int32), s.moves[:, 0, 0, 0], beam)
    hp = hp.at[:, 1, 0].add(j.where(hit & ~behind, -10, 0))
    s = s._replace(substitute_hp=s.substitute_hp.at[:, 1, 0].add(j.where(behind, -10, 0)))
    dealt = j.where(hit & ~behind, 10, 0)
    hooks.damage(None, s, j.zeros(lanes, j.int32), dealt, dealt, beam)
    v = t.public_volatiles;counter = s.recharge_counter
    set_ = hit | (beam & miss_recharge)
    if recharge:
        counter = counter.at[:, 0, 0].set(j.where(set_, 1, counter[:, 0, 0]))
        if bag:v = v.at[:, 0, 0, NAMES.index('mustrecharge')].set(v[:, 0, 0, NAMES.index('mustrecharge')] | set_)
    item, boosts, confusion = s.item, s.boosts, s.confusion_counter
    if trick:
        swap = (actions[:, 0] == 1) & ~s.done
        own, foe = item[:, 0, 0], item[:, 1, 0]
        item = item.at[:, 0, 0].set(j.where(swap, foe, own)).at[:, 1, 0].set(j.where(swap, own, foe))
        liechi = swap & (item[:, 0, 0] == LIECHI)
        item = item.at[:, 0, 0].set(j.where(liechi, 0, item[:, 0, 0]))
        boosts = boosts.at[:, 0, 0, 0].add(j.where(liechi, 1, 0))
        lum = swap & (item[:, 1, 0] == LUM) & (confusion[:, 1, 0] > 0)
        item = item.at[:, 1, 0].set(j.where(lum, 0, item[:, 1, 0]))
        confusion = confusion.at[:, 1, 0].set(j.where(lum, 0, confusion[:, 1, 0]))
        v = v.at[:, 1, 0, NAMES.index('confusion')].set(v[:, 1, 0, NAMES.index('confusion')] & ~lum)
        if wide:   # the reviewer's TWIDE: the target also loses an Attack stage, the Tricker's bag shows Destiny Bond
            boosts = boosts.at[:, 1, 0, 0].add(j.where(swap, -1, 0))
            v = v.at[:, 0, 0, NAMES.index('destinybond')].set(v[:, 0, 0, NAMES.index('destinybond')] | swap)
    s = s._replace(hp=hp, recharge_counter=counter, item=item, boosts=boosts, confusion_counter=confusion, turn=s.turn + 1)
    return result(s, t._replace(public_volatiles=v))


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class M1PlantedBugTests(unittest.TestCase):
    """M1 review: the reviewer's planted bugs (RMISS, RNOBAG, TWIDE) must fail the gate; the correct stand-in passes."""

    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import engine_device_public_hooks as hooks
        from gpu_engine.v2e_capture import leaf_paths
        cls.tmp = Path(tempfile.mkdtemp(prefix='v2e-regress-m1-test-'))
        shards = cls.tmp / 'cap' / 'shards';shards.mkdir(parents=True)
        first = lambda tree: jax.tree.map(lambda x: x[:1], tree)
        cls.template = (first(m1_state(0)), j.zeros((1, 2), j.int32), first(m1_trackers(M1_LANES)))
        for k in range(2):
            tree = (m1_state(k), j.asarray(m1_actions()), m1_trackers(M1_LANES))
            np.savez_compressed(shards / f'shard-{k:06d}.npz', tick=np.int64(k), level=np.int64(0),
                **{f'l{i:03d}': np.asarray(x) for i, x in enumerate(jax.tree.leaves(tree))})
        (cls.tmp / 'cap' / 'capture.json').write_text(json.dumps(dict(shards=2, paths=leaf_paths(cls.template))))
        cls.context = dict(prepared=SimpleNamespace(engine_tables=SimpleNamespace(sleep_talk=SimpleNamespace(
            callable=np.array([False, True, True, True, True])))), pool=SimpleNamespace(metadata=lambda: dict(vocabulary=M1_VOCAB)))
        cls.stubs = [mock.patch.object(hooks, 'damage', lambda c, state, side, dealt, taken, use: c),
                     mock.patch.object(hooks, 'move', lambda c, *rest, **kw: c)]
        for stub in cls.stubs:stub.start()
        from gpu_engine import v2e_regress
        cls.base = cls.tmp / 'base'
        code = v2e_regress.base_pass(cls.args(out=cls.base, pad_lanes=M1_LANES),
            step_override=clocked(lambda s, a, t: m1_step(s, a, t, recharge=False, trick=False), cls.ran), prepared=(cls.context, cls.template))
        assert code == 0

    @classmethod
    def tearDownClass(cls):
        for stub in cls.stubs:stub.stop()
        shutil.rmtree(cls.tmp)

    @staticmethod
    def ran(state, actions):
        import jax.numpy as j
        return j.ones(state.hp.shape[0], j.bool_)

    @classmethod
    def args(cls, **extra):
        values = dict(config=None, reference_run=None, shards=str(cls.tmp / 'cap' / 'shards'), limit=0, arm='B', allow_dirty=True,
                      integration=None, allow_base_mismatch=True, waive=None)
        values.update(extra);return Namespace(**values)

    def patched(self, name, **kind):
        from gpu_engine import v2e_regress
        out = self.tmp / name
        code = v2e_regress.patched_pass(self.args(out=out, base=str(self.base), pr='M1', faint_clear=False),
            step_override=clocked(lambda s, a, t: m1_step(s, a, t, **kind), self.ran), prepared=(self.context, self.template))
        return code, json.loads((out / 'report.json').read_text()), [json.loads(x) for x in (out / 'changed.jsonl').read_text().splitlines()]

    def test_correct_stand_in_passes_by_post_state_witnesses(self):
        manifest = json.loads((self.base / 'manifest.json').read_text())
        self.assertIn('trackers.public_volatiles', manifest['witness_leaves']);self.assertIn('state.substitute_hp', manifest['witness_leaves'])
        code, report, listing = self.patched('correct')
        c = report['counts']
        self.assertEqual(code, 0, c);self.assertEqual((c['changed_but_uninvolved'], c['changed_involved_unexpected_fields']), (0, 0))
        self.assertEqual(c['recharge_leaves'], 4)                      # 2 hit lanes (one behind a fresh Substitute) x 2 shards
        self.assertEqual(c['trick_witness_leaves'], 4)                 # Liechi and Lum lanes x 2 shards
        self.assertEqual(c['bag_leaves'], 6)                           # the 4 recharge bits and the 2 Lum confusion clears
        by_lane = {(x['shard'], x['lane']): x for x in listing}
        self.assertEqual(by_lane[('shard-000000.npz', 4)]['trick_rule'], ['state.boosts@0'])
        self.assertEqual(by_lane[('shard-000000.npz', 5)]['trick_rule'], ['state.confusion_counter@1'])
        self.assertEqual(by_lane[('shard-000000.npz', 5)]['bag_rule'], ['trackers.public_volatiles@1'])

    def test_rmiss_recharge_on_a_miss_is_flagged(self):
        """RMISS: the counter (and its bag bit) set whenever Hyper Beam acts, a miss included: no hit evidence there."""
        code, report, listing = self.patched('rmiss', miss_recharge=True)
        self.assertEqual(code, 1)
        flagged = {x['lane'] for x in listing if x['kind'] == 'unexpected'}
        self.assertEqual(flagged, {2, 3})
        self.assertEqual(set(report['by_effect']['hyperbeam']['unexpected_fields']),
                         {'state.recharge_counter@user', 'trackers.public_volatiles@user'})

    def test_rnobag_counter_without_its_bag_bit_is_flagged(self):
        """RNOBAG: the device never publishes -mustrecharge: the counter alone is not explained."""
        code, report, listing = self.patched('rnobag', bag=False)
        self.assertEqual(code, 1)
        self.assertEqual({x['lane'] for x in listing if x['kind'] == 'unexpected'}, {0, 1})
        self.assertEqual(set(report['by_effect']['hyperbeam']['unexpected_fields']), {'state.recharge_counter@user'})

    def test_twide_extra_stage_and_bag_bit_are_flagged(self):
        """TWIDE: a successful Trick also drops the target's Attack and shows Destiny Bond on the Tricker's bag."""
        code, report, listing = self.patched('twide', wide=True)
        self.assertEqual(code, 1)
        fields = report['by_effect']['trick']['unexpected_fields']
        self.assertIn('trackers.public_volatiles@user', fields)     # Destiny Bond: not a cell a swap can move
        self.assertIn('state.boosts@target', fields)                # no pinch berry the swap moved explains it (lanes 5, 6)
        self.assertEqual({x['lane'] for x in listing if x['kind'] == 'unexpected'}, {4, 5, 6})


# M2 (standing planted-mutant gate, lead 2026-11-11): one synthetic capture with Perish Song, Yawn and Wrap lanes.
M2_VOCAB = dict(moves=['_none', 'perishsong', 'yawn', 'wrap', 'tackle'], abilities=['_none', 'soundproof', 'insomnia'],
                items=['_none', 'leftovers'])
M2_PERISH, M2_YAWN, M2_WRAP, M2_TACKLE = 1, 2, 3, 4
M2_LANES = 10          # 0 Perish Song acts, 1 it is selected but the user cannot act; 2 Yawn lands, 3 into Protect; 4 Wrap hits, 5 misses;
                       # 6 side 1's active is drowsy before the step (its Yawn sleep lands at this residual); 7 Wrap hits, then the
                       # trapper is KO'd later in the turn (both trees), so the trap releases silently; 8 Wrap hits and ticks, then
                       # the slower trapper faints at its own residual after the holder's handler (both trees): the trap stands; 9 done


class TrackersM2(NamedTuple):
    physical_to_public: object
    active_slot: object
    public_volatiles: object
    mon_features: object


def m2_trackers(lanes):
    import jax.numpy as j
    from gpu_engine.public_active import NAMES
    z = j.zeros((lanes, 2, 6), j.int32)
    return TrackersM2(physical_to_public=j.broadcast_to(j.arange(6, dtype=j.int32), (lanes, 2, 6)), active_slot=j.zeros((lanes, 2), j.int32),
        public_volatiles=j.zeros((lanes, 2, 6, len(NAMES)), j.bool_), mon_features={'meanlook_trap': z, 'wrap_trap_turns': z})


def m2_state(seed):
    import jax.numpy as j
    s = lanes_state(M2_LANES, seed)
    moves = np.full((M2_LANES, 2, 6, 4), M2_TACKLE, np.int32)
    moves[:, 0, 0, 0] = [M2_PERISH, M2_PERISH, M2_YAWN, M2_YAWN, M2_WRAP, M2_WRAP, M2_TACKLE, M2_WRAP, M2_WRAP, M2_TACKLE]
    return s._replace(moves=j.asarray(moves), choice_move=s.choice_move.at[:, 1, 0].set(0), item=s.item.at[:, 1, 0].set(1),
                      yawn_counter=s.yawn_counter.at[6, 1, 0].set(1), active_turns=j.ones_like(s.active_turns))


def m2_step(state, actions, t, *, perish_fail=False, perish_wide=False, perish_bag=True, yawn_protect=False, yawn_wide=False,
            yawn_bag=True, wrap_miss=False, wrap_wide=False, wrap_bag=True, wrap_clock=True, perish_bench=False, yawn_bench=False,
            wrap_bench=False, trap_bench=False, sleep_bench=False, tick_div=16, no_release=False, m2=True):
    """A stand-in engine with M2's writes: side 0's move line (the move hook), its hit (the damage hook) and the foe's Protect
    (port_protect) are reported as the engine reports them, then the residual has run (counters after one tick: Perish 3,
    Yawn 1, Wrap 3+draws[0][7]-1 with source 0 and its tick max(1, maxhp//16) on the trapped active; lane 6's drowsy active
    falls asleep; lane 7's trap released, its trapper fainted; the holder's handler cutpoint, the HP line transition emits
    after it, reports the trapper there, before lane 8's late trapper faint) and turn_end is ordinary. The base pass is m2=False. Keyword
    flags plant the reviewer-style bugs: an effect on a failure, a wrong extra field, a removed bag or clock producer, the
    effect on a benched mon (side 1's party index 1) instead of the active, a tick of the wrong size (RW7: tick_div=8) and
    a trap kept and ticked for a fainted trapper (RW8: no_release)."""
    import jax.numpy as j
    from gpu_engine import engine_device_public_hooks as hooks
    from gpu_engine.public_active import NAMES
    from gpu_engine.transition import sample_chance_inputs
    from gpu_engine.mechanics import SLP
    lanes = state.hp.shape[0];rows = j.arange(lanes);s = state
    live = ~s.done;z = j.zeros(lanes, j.int32)
    perish = live & (rows < 2);acting = live & (rows != 1)                 # lane 1: Perish Song selected, the user cannot act
    yawn = live & (rows >= 2) & (rows < 4);protected = live & (rows == 3)
    wrap = live & (((rows >= 4) & (rows < 6)) | (rows == 7) | (rows == 8));hit = live & ((rows == 4) | (rows == 7) | (rows == 8))
    move = s.moves[:, 0, 0, 0]
    hooks.port_protect(None, z, acting, j.ones(lanes, j.bool_), protected, protected)
    hooks.move(None, s, z, move, acting)
    dealt = j.where(hit, 10, 0)
    s = s._replace(hp=s.hp.at[:, 1, 0].add(-dealt))
    hooks.damage(None, s, z, dealt, dealt, acting & (move == M2_WRAP))
    s = s._replace(hp=s.hp.at[:, 0, 0].set(j.where(live & (rows == 7), 0, s.hp[:, 0, 0])))   # lane 7: the trapper is KO'd
    v = t.public_volatiles;clock = t.mon_features['wrap_trap_turns']
    def setbag(v, side, name, mask, mon=0):
        k = NAMES.index(name);return v.at[:, side, mon, k].set(v[:, side, mon, k] | mask)
    if m2:
        lands = perish & (acting | perish_fail)
        foe = 1 if perish_bench else 0                                      # perish_bench: the foe's counter on its bench
        s = s._replace(perish_counter=s.perish_counter.at[:, 0, 0].set(j.where(lands, 3, s.perish_counter[:, 0, 0]))
                       .at[:, 1, foe].set(j.where(lands, 3, s.perish_counter[:, 1, foe])))
        if perish_bag:
            v = setbag(v, 0, 'perish3', lands);v = setbag(v, 1, 'perish3', lands, foe)
        if perish_wide:s = s._replace(boosts=s.boosts.at[:, 1, 0, 0].add(j.where(lands, -1, 0)))
        drowsy = yawn & (~protected | yawn_protect)
        mon = 1 if yawn_bench else 0                                        # yawn_bench: the target's bench mon drowses
        s = s._replace(yawn_counter=s.yawn_counter.at[:, 1, mon].set(j.where(drowsy, 1, s.yawn_counter[:, 1, mon])))
        if yawn_bag:v = setbag(v, 1, 'yawn', drowsy, mon)
        # Lane 6: the pre-step drowsy active's countdown ends and it falls asleep (sleep_bench: a benched mon does).
        sleeps = live & (rows == 6);mon = 1 if sleep_bench else 0
        s = s._replace(yawn_counter=s.yawn_counter.at[:, 1, 0].set(j.where(sleeps, 0, s.yawn_counter[:, 1, 0])),
                       status=s.status.at[:, 1, mon].set(j.where(sleeps, SLP, s.status[:, 1, mon])),
                       status_counter=s.status_counter.at[:, 1, mon].set(j.where(sleeps, 2, s.status_counter[:, 1, mon])))
        if yawn_wide:s = s._replace(item=s.item.at[:, 1, 0].set(j.where(drowsy, 0, s.item[:, 1, 0])))
        trapped = wrap & (hit | wrap_miss) & ((rows != 7) | no_release)     # lane 7: a fainted trapper releases it
        d = 3 + sample_chance_inputs(state).draws[:, 0, 7];mon = 1 if trap_bench else 0   # trap_bench: the bench is trapped
        tick = j.maximum(1, s.max_hp[:, 1, 0] // tick_div);at = 1 if wrap_bench else 0     # wrap_bench: the tick on the bench
        s = s._replace(hp=s.hp.at[:, 1, at].set(j.where(trapped, j.maximum(0, s.hp[:, 1, at] - tick), s.hp[:, 1, at])))
        s = s._replace(partial_trap_counter=s.partial_trap_counter.at[:, 1, mon].set(j.where(trapped, d - 1, 0)),
                       partial_trap_source=s.partial_trap_source.at[:, 1, mon].set(j.where(trapped, 0, -1)))
        if wrap_bag:v = setbag(v, 1, 'partiallytrapped', trapped, mon)
        if wrap_clock:clock = clock.at[:, 1, mon].set(j.where(trapped, 1, clock[:, 1, mon]))
        # The holder's partiallytrapped handler and the HP-line cutpoint after it (the names transition uses there).
        _wrap_holder = trapped;_wrap_tick = trapped
        hooks.hp_line(None, s, j.ones(lanes, j.int32), _wrap_tick, source_class=2)
        if wrap_wide:s = s._replace(boosts=s.boosts.at[:, 1, 0, 1].add(j.where(trapped, -1, 0)))
    s = s._replace(hp=s.hp.at[:, 0, 0].set(j.where(live & (rows == 8), 0, s.hp[:, 0, 0])))   # lane 8: a late trapper faint
    s = s._replace(turn=s.turn + 1)
    hooks.turn_end(None, state, s, live, upkeep=None, leftovers=None)
    return result(s, t._replace(public_volatiles=v, mon_features={**t.mon_features, 'wrap_trap_turns': clock}))


class M2PlantedBugTests(unittest.TestCase):
    """M2: the effect on a failure, a wrong extra field, a removed bag or clock producer and the effect on a benched mon must
    each fail the gate; the correct stand-in passes by m2_rule's post-state witnesses (the standing planted-mutant gate)."""

    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import engine_device_public_hooks as hooks
        from gpu_engine.v2e_capture import leaf_paths
        cls.tmp = Path(tempfile.mkdtemp(prefix='v2e-regress-m2-test-'))
        shards = cls.tmp / 'cap' / 'shards';shards.mkdir(parents=True)
        first = lambda tree: jax.tree.map(lambda x: x[:1], tree)
        cls.template = (first(m2_state(0)), j.zeros((1, 2), j.int32), first(m2_trackers(M2_LANES)))
        for k in range(2):
            tree = (m2_state(k), j.zeros((M2_LANES, 2), j.int32), m2_trackers(M2_LANES))
            np.savez_compressed(shards / f'shard-{k:06d}.npz', tick=np.int64(k), level=np.int64(0),
                **{f'l{i:03d}': np.asarray(x) for i, x in enumerate(jax.tree.leaves(tree))})
        (cls.tmp / 'cap' / 'capture.json').write_text(json.dumps(dict(shards=2, paths=leaf_paths(cls.template))))
        cls.context = dict(prepared=SimpleNamespace(engine_tables=SimpleNamespace(sleep_talk=SimpleNamespace(
            callable=np.array([False, True, True, True, True])))), pool=SimpleNamespace(metadata=lambda: dict(vocabulary=M2_VOCAB)))
        cls.stubs = [mock.patch.object(hooks, 'damage', lambda c, state, side, dealt, taken, use: c),
                     mock.patch.object(hooks, 'move', lambda c, *rest, **kw: c),
                     mock.patch.object(hooks, 'port_protect', lambda c, *rest: c),
                     mock.patch.object(hooks, 'turn_end', lambda c, *rest, **kw: c),
                     mock.patch.object(hooks, 'hp_line', lambda c, *rest, **kw: c)]
        for stub in cls.stubs:stub.start()
        from gpu_engine import v2e_regress
        cls.base = cls.tmp / 'base'
        code = v2e_regress.base_pass(cls.args(out=cls.base, pad_lanes=M2_LANES),
            step_override=clocked(lambda s, a, t: m2_step(s, a, t, m2=False), cls.ran), prepared=(cls.context, cls.template))
        assert code == 0

    @classmethod
    def tearDownClass(cls):
        for stub in cls.stubs:stub.stop()
        shutil.rmtree(cls.tmp)

    ran = staticmethod(M1PlantedBugTests.ran)
    args = classmethod(M1PlantedBugTests.args.__func__)

    def patched(self, name, **kind):
        from gpu_engine import v2e_regress
        out = self.tmp / name
        code = v2e_regress.patched_pass(self.args(out=out, base=str(self.base), pr='M2', faint_clear=False),
            step_override=clocked(lambda s, a, t: m2_step(s, a, t, **kind), self.ran), prepared=(self.context, self.template))
        return code, json.loads((out / 'report.json').read_text()), [json.loads(x) for x in (out / 'changed.jsonl').read_text().splitlines()]

    def flagged(self, listing):
        return {(x['shard'][-10:-4], x['lane']) for x in listing if x['kind'] == 'unexpected'}

    def test_correct_stand_in_passes_by_post_state_witnesses(self):
        code, report, listing = self.patched('correct')
        c = report['counts']
        self.assertEqual(code, 0, c);self.assertEqual((c['changed_but_uninvolved'], c['changed_involved_unexpected_fields']), (0, 0))
        self.assertEqual(c['m2_leaves'], 8)                               # Perish, Yawn, two Wrap lanes x 2 shards
        self.assertEqual((c['m2_wrong_mon'], c['m2_forced']), (0, 0))
        self.assertEqual({x['lane'] for x in listing if 'drowsy' in x['involved']}, {6})   # the drowsy sleep, allowed per mon
        self.assertEqual(c["bag_leaves"], 8)
        by_lane = {(x['shard'], x['lane']): x for x in listing}
        self.assertEqual(sorted(by_lane[('shard-000000.npz', 4)]['m2_rule']), sorted(
            ["state.hp@1", "state.partial_trap_counter@1", "state.partial_trap_source@1", "trackers.mon_features['wrap_trap_turns']@1"]))
        self.assertNotIn(('shard-000000.npz', 7), by_lane)                 # the released trap: nothing differs from the base
        # Lane 8: the trapper fainted after the holder's handler; the cutpoint shows it held there, so the tick stands.
        self.assertEqual(sorted(by_lane[('shard-000000.npz', 8)]['m2_rule']), sorted(by_lane[('shard-000000.npz', 4)]['m2_rule']))

    def assert_flagged(self, kind, lanes, fields, effect):
        code, report, listing = self.patched(kind[0], **kind[1])
        self.assertEqual(code, 1)
        self.assertEqual({lane for _, lane in self.flagged(listing)}, lanes)
        self.assertTrue(set(fields) <= set(report['by_effect'][effect]['unexpected_fields']), report['by_effect'][effect]['unexpected_fields'])

    def test_effects_on_a_failure_are_flagged(self):
        """P1: Perish Song lands while its user cannot act (no move line); Y1: Yawn through Protect; W1: Wrap on a miss."""
        self.assert_flagged(('p1', dict(perish_fail=True)), {1}, ['state.perish_counter@user', 'state.perish_counter@target'], 'perishsong')
        self.assert_flagged(('y1', dict(yawn_protect=True)), {3}, ['state.yawn_counter@target'], 'yawn')
        self.assert_flagged(('w1', dict(wrap_miss=True)), {5}, ['state.partial_trap_counter@target'], 'wrap')

    def test_wrong_fields_are_flagged(self):
        """P2: Perish Song also drops the foe's Attack; Y2: Yawn also takes the target's item; W2: Wrap also drops its Defense."""
        self.assert_flagged(('p2', dict(perish_wide=True)), {0}, ['state.boosts@target'], 'perishsong')
        self.assert_flagged(('y2', dict(yawn_wide=True)), {2}, ['state.item@target'], 'yawn')
        self.assert_flagged(('w2', dict(wrap_wide=True)), {4}, ['state.boosts@target'], 'wrap')

    def test_removed_device_producers_are_flagged(self):
        """P3/Y3/W3: the perish, yawn or partiallytrapped bag producer removed; W4: the WRAP_TRAP_TURNS clock removed."""
        self.assert_flagged(('p3', dict(perish_bag=False)), {0}, ['state.perish_counter@user'], 'perishsong')
        self.assert_flagged(('y3', dict(yawn_bag=False)), {2}, ['state.yawn_counter@target'], 'yawn')
        self.assert_flagged(('w3', dict(wrap_bag=False)), {4}, ['state.partial_trap_counter@target'], 'wrap')
        self.assert_flagged(('w4', dict(wrap_clock=False)), {4}, ['state.partial_trap_counter@target'], 'wrap')

    def test_effects_on_a_benched_mon_are_flagged(self):
        """Wrong mon (M1 re-review's benched-mon lesson): P4 Perish Song's foe counter, Y4 Yawn's drowsiness, W5 Wrap's
        damage, W6 Wrap's trap and Y5 a Yawn sleep land on side 1's bench mon (party index 1) instead of its active; m2_rule
        (the counters) or the wrong-mon check (HP, status, which their families would allow) flags each."""
        self.assert_flagged(('p4', dict(perish_bench=True)), {0}, ['state.perish_counter@target'], 'perishsong')
        self.assert_flagged(('y4', dict(yawn_bench=True)), {2}, ['state.yawn_counter@target'], 'yawn')
        self.assert_flagged(('w6', dict(trap_bench=True)), {4}, ['state.partial_trap_counter@target'], 'wrap')
        code, report, listing = self.patched('w5', wrap_bench=True)
        self.assertEqual(code, 1);self.assertEqual({lane for _, lane in self.flagged(listing)}, {4})
        self.assertIn('state.hp@target', report['by_effect']['wrap']['unexpected_fields'])
        self.assertEqual([x['m2_wrong_mon'] for x in listing if x['lane'] == 4], [['state.hp@1']] * 2)
        self.assertEqual([x['m2_forced'] for x in listing if x['lane'] == 4], [['state.hp@1']] * 2)
        code, report, listing = self.patched('y5', sleep_bench=True)
        self.assertEqual(code, 1);self.assertEqual({lane for _, lane in self.flagged(listing)}, {6})
        self.assertTrue({'state.status@user', 'state.status_counter@user'} <= set(report['by_effect']['drowsy']['unexpected_fields']))
        self.assertEqual(report['counts']['m2_wrong_mon'], 2)

    def test_wrong_tick_and_kept_trap_are_flagged(self):
        """M2 review: RW7, a tick of maxhp//8, is caught by the HP digest witness alone (the trap leaves and the wrong-mon
        check pass; the HP is a `must` entry, never an allowance); RW8, a trap kept and ticked after its trapper fainted, by
        the trapper-held check (and the unexplained tick)."""
        code, report, listing = self.patched('rw7', tick_div=8)
        self.assertEqual(code, 1);self.assertEqual({lane for _, lane in self.flagged(listing)}, {4})
        self.assertEqual(report['by_effect']['wrap']['unexpected_fields'], {'state.hp@target': 2})
        rows = [x for x in listing if x['lane'] == 4]
        self.assertEqual([(x['m2_forced'], x['m2_wrong_mon']) for x in rows], [(['state.hp@1'], [])] * 2)
        self.assertEqual(sorted(rows[0]['m2_rule']), sorted(["state.partial_trap_counter@1", "state.partial_trap_source@1",
                                                             "trackers.mon_features['wrap_trap_turns']@1"]))
        code, report, listing = self.patched('rw8', no_release=True)
        self.assertEqual(code, 1);self.assertEqual({lane for _, lane in self.flagged(listing)}, {7})
        self.assertTrue({'state.partial_trap_counter@target', 'state.hp@target'} <= set(report['by_effect']['wrap']['unexpected_fields']))


if __name__ == '__main__':
    unittest.main()
