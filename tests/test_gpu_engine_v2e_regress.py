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
VOCAB = dict(moves=['_none', 'curse', 'tackle', 'sleeptalk', 'spore', 'struggle'], abilities=['_none', 'insomnia', 'pressure'],
             items=['_none', 'leftovers', 'whiteherb'])
CURSE, TACKLE, SLEEPTALK, SPORE = 1, 2, 3, 4


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


def fake_install(c, state, side, use, *, initial=False):
    """PR0's rule on a fake carry: the installed public slot moves to the front."""
    import jax.numpy as j
    t = c.trackers;rows = j.arange(state.hp.shape[0]);s = j.broadcast_to(j.asarray(side), rows.shape)
    slot = t.active_slot[rows, s];current = t.public_order[rows, s]
    position = j.argmax(current == slot[:, None], axis=1)
    swapped = current.at[rows, position].set(current[:, 0]).at[:, 0].set(slot)
    keep = j.asarray(use) & (not initial)
    return c._replace(trackers=t._replace(public_order=t.public_order.at[rows, s].set(j.where(keep[:, None], swapped, current))))


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

    def test_migration_by_path(self):
        from gpu_engine.v2e_regress import MigrationError, migrate
        template = {'state.a': np.zeros((1, 2), np.int32), 'state.new': np.full((1, 6), -1, np.int32), 'actions': np.zeros((1, 2), np.int32)}
        leaves = [np.ones((3, 2), np.int32), np.zeros((3, 2), np.int32)]
        named, filled = migrate(['state.a', 'actions'], leaves, template)
        self.assertEqual(filled, ['state.new']);self.assertEqual(named['state.new'].shape, (3, 6))
        self.assertTrue((named['state.new'] == -1).all())
        named, _ = migrate(['state.a', 'actions'], leaves, template, defaults={'state.new': lambda n: np.full((n, 6), 5)})
        self.assertTrue((named['state.new'] == 5).all())
        with self.assertRaises(MigrationError):migrate(['state.a', 'actions'], [leaves[0].astype(np.int16), leaves[1]], template)
        with self.assertRaises(MigrationError):migrate(['state.a', 'actions'], [np.ones((3, 3), np.int32), leaves[1]], template)
        with self.assertRaises(MigrationError):migrate(['state.a', 'actions', 'state.gone'], leaves + [leaves[0]], template)

    def test_public_order_reference_replay(self):
        from gpu_engine.v2e_regress import replay_public_order
        order = np.broadcast_to(np.arange(6), (2, 2, 6)).copy()
        calls = [(np.array([0, 0]), np.array([s, s]), np.array([True, False])) for s in (0, 1, 2, 0)]
        out = replay_public_order(order, calls)
        self.assertEqual(out[0, 0].tolist(), [0, 2, 1, 3, 4, 5])
        self.assertEqual(out[1, 0].tolist(), list(range(6)))           # inactive row unchanged
        self.assertEqual(out[0, 1].tolist(), list(range(6)))           # other seat untouched

    def test_faint_clear_allows_only_the_fainted_mons_cleared_entries(self):
        from gpu_engine.v2e_regress import faint_clear_ok
        fainted = np.zeros((2, 6), bool);fainted[1, 0] = True
        p2p = np.broadcast_to(np.arange(6), (2, 6))
        base = {'trackers.public_volatiles': np.ones((2, 6, 3), bool), "trackers.mon_features['meanlook_trap']": np.ones((2, 6), np.int32)}
        post = {k: v.copy() for k, v in base.items()}
        post['trackers.public_volatiles'][1, 0] = False;post["trackers.mon_features['meanlook_trap']"][0, 3] = 0
        self.assertTrue(faint_clear_ok(list(post), post, base, fainted, p2p))
        post['trackers.public_volatiles'][1, 1] = False                 # another mon
        self.assertFalse(faint_clear_ok(list(post), post, base, fainted, p2p))
        self.assertFalse(faint_clear_ok(['trackers.other'], {'trackers.other': np.zeros(1)}, {'trackers.other': np.ones(1)}, fainted, p2p))

    def test_involvement_resolves_the_executed_move(self):
        from gpu_engine.v2e_regress import Pre, sleep_clause, sleep_immunity, destiny_bond_bit
        L = 6;z = lambda *s: np.zeros(s, np.int32)
        s = {k: z(L, 2, 6) for k in ('hp', 'status', 'charge_move', 'encore_move', 'encore_counter', 'ability', 'item', 'confusion_counter')}
        s.update(hp=np.full((L, 2, 6), 50), rest_origin=np.zeros((L, 2, 6), bool), volatile=np.zeros((L, 2, 6), np.uint32), active=z(L, 2),
                 pending_residual=np.zeros(L, bool), queued_actor=np.full(L, -1), queued_actions=z(L, 2),
                 moves=np.full((L, 2, 6, 4), TACKLE))
        s['moves'][0, 0, 0, 1] = CURSE                                  # lane 0: slot 1 chosen
        s['moves'][1, 0, 0, 2] = CURSE;s['encore_move'][1, 0, 0] = CURSE;s['encore_counter'][1, 0, 0] = 2   # lane 1: Encore override
        s['charge_move'][2, 1, 0] = CURSE                               # lane 2: charge lock
        s['moves'][3, 0, 0] = [SLEEPTALK, CURSE, TACKLE, 0]               # lane 3: Sleep Talk may call Curse
        s['pending_residual'][4] = True;s['queued_actor'][4] = 1;s['queued_actions'][4, 1] = 3;s['moves'][4, 1, 0, 3] = CURSE  # resume
        actions = np.zeros((L, 2), np.int32);actions[0, 0] = 1;actions[4, 1] = 5   # lane 4's replacement action is ignored
        named = {f'state.{k}': v for k, v in s.items()};named['actions'] = actions
        callable_moves = np.array([False, True, True, False, True, False])
        p = Pre(named, VOCAB, callable_moves, np.zeros((L, 2), bool))
        self.assertEqual(p.uses('curse').tolist(), [True, True, True, True, True, False])
        struggle = np.zeros((L, 2), bool);struggle[5, 1] = True
        self.assertTrue(Pre(named, VOCAB, callable_moves, struggle).uses('struggle')[5])
        # Sleep Clause: lane 0 side 1 has a benched non-Rest sleeper and side 0 uses Spore.
        named['state.moves'] = np.full((L, 2, 6, 4), SPORE);named['state.status'][0, 1, 3] = 5;named['state.status'][1, 1, 3] = 5
        named['state.rest_origin'][1, 1, 3] = True;named['state.status'][2, 1, 0] = 5   # Rest sleeper; the target itself
        p = Pre(named, VOCAB, callable_moves, np.zeros((L, 2), bool))
        self.assertEqual(sleep_clause(p).tolist()[:3], [True, False, False])
        named['state.ability'][3, 1, 0] = 1
        self.assertTrue(sleep_immunity(Pre(named, VOCAB, callable_moves, np.zeros((L, 2), bool)))[3])
        named['state.volatile'][5, 0, 2] = 64
        self.assertEqual(destiny_bond_bit(Pre(named, VOCAB, callable_moves, np.zeros((L, 2), bool))).tolist(), [False] * 5 + [True])


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
            callable=np.array([False, True, True, False, True, False])))), pool=SimpleNamespace(metadata=lambda: dict(vocabulary=VOCAB)))
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
        values = dict(config=None, reference_run=None, shards=str(cls.tmp / 'cap' / 'shards'), limit=0, arm='B')
        values.update(extra);return Namespace(**values)

    def patched(self, name, step, *, pr='A1', template=None, faint_clear=False):
        from gpu_engine import v2e_regress
        out = self.tmp / name
        code = v2e_regress.patched_pass(self.args(out=out, base=str(self.base), pr=pr, faint_clear=faint_clear), step_override=step,
            prepared=(self.context, template or self.template))
        return code, json.loads((out / 'report.json').read_text()), [json.loads(x) for x in (out / 'changed.jsonl').read_text().splitlines()]

    def test_base_pass_manifest(self):
        m = self.manifest
        self.assertEqual(m['lane_steps'], 2 * self.LANES + self.LANES // 2);self.assertEqual(m['live_lane_steps'], m['lane_steps'] - 3)
        self.assertIn('state.hp', m['output_names']);self.assertIn('trackers.public_volatiles', m['faint_leaves'])
        digest = np.load(self.base / 'digest-shard-000001.npz')
        self.assertEqual(digest['digests'].shape, (self.LANES // 2, len(m['output_names']), 2))

    def test_base_against_base_is_unchanged(self):
        code, report, listing = self.patched('same', base_step)
        self.assertEqual(code, 0);self.assertEqual(listing, [])
        c = report['counts']
        self.assertEqual((c['changed_and_involved'], c['changed_but_uninvolved'], c['done_skipped']), (0, 0, 3))
        self.assertEqual(c['unchanged'], c['lane_steps'] - 3)
        self.assertGreater(report['by_effect']['curse']['involved_unchanged'], 0)

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

    def test_change_on_involved_lanes_is_attributed(self):
        import jax.numpy as j
        def curse_only(state, actions, t):
            out = base_step(state, actions, t)
            s = out.step.state;rows = j.arange(s.hp.shape[0])
            slot = j.clip(actions[:, 0], 0, 3)
            uses = (actions[:, 0] < 4) & (state.moves[rows, 0, state.active[:, 0], slot] == CURSE)
            return result(s._replace(turn=s.turn + uses.astype(s.turn.dtype)), out.trackers)
        code, report, listing = self.patched('involved', curse_only)
        self.assertEqual(code, 0);self.assertEqual(report['counts']['changed_but_uninvolved'], 0)
        self.assertGreater(report['by_effect']['curse']['changed'], 0)
        self.assertTrue(all('curse' in x['involved'] and x['kind'] == 'involved' for x in listing))
        code, report, _ = self.patched('involved-none', curse_only, pr='none')
        self.assertEqual(code, 1);self.assertGreater(report['counts']['changed_but_uninvolved'], 0)

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

if __name__ == '__main__':
    unittest.main()
