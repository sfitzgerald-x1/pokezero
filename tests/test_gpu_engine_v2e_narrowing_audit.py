"""Strict pin attribution on synthetic captures, including planted writes hidden by general cascade rules."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import NamedTuple
import unittest
from unittest import mock

import numpy as np

from gpu_engine import v2e_narrowing_audit as audit


class Pins(NamedTuple):
    pin: object
    pin_active: object
    visible_pin: object
    visible_pin_active: object
    conflicts: object
    visible_hard: object


class Trackers(NamedTuple):
    physical_to_public: object
    active_slot: object
    public_volatiles: object
    mon_features: object
    species: object
    present: object
    belief_mask: object
    visible_belief: object


def trackers(lanes):
    import jax.numpy as j
    words = j.zeros((lanes, 2, 6, 2), j.uint32);flags = j.zeros((lanes, 2, 6), bool)
    return Trackers(j.broadcast_to(j.arange(6), (lanes, 2, 6)), j.zeros((lanes, 2), j.int32),
        j.zeros((lanes, 2, 6, 3), bool), {}, j.ones((lanes, 2, 6), j.int32), ~flags, words + 3,
        Pins(words, flags, words, flags, j.zeros_like(flags, j.int32), words + 3))


def producer(kind):
    def run(t, tables, state, side, move, caller_move, critical, sub, endured, dealt, hp_before, hp_after, num, den, use, **kw):
        import jax.numpy as j
        lanes = state.hp.shape[0];r = j.arange(lanes);s = j.broadcast_to(j.asarray(side), (lanes,))
        slot = j.clip(t.physical_to_public[r, s, state.active[r, s]], 0, 5)
        if kind == 'wrong_side':s = 1 - s
        if kind == 'wrong_slot':slot = (slot + 1) % 6
        vb = t.visible_belief
        words = vb.pin.at[r, s, slot, 0].set(vb.pin[r, s, slot, 0] ^ j.uint32(1))
        vb = vb._replace(pin=words, visible_pin=words, pin_active=vb.pin_active.at[r, s, slot].set(True),
                         visible_pin_active=vb.visible_pin_active.at[r, s, slot].set(True))
        if kind == 'visible_hard':vb = vb._replace(visible_hard=vb.visible_hard.at[r, s, slot, 0].set(1))
        refused = kind == 'refused' or (kind in ('late_refusal', 'late_hook_refusal') and kw.get('pursuit', False))
        out = t if kind == 'late_hook_refusal' and refused else t._replace(visible_belief=vb)
        return out, SimpleNamespace(reason=j.full(lanes, int(refused), j.int32),
            fit=j.full((lanes, 2), 0 if kind == 'empty_fit' else 1, j.uint32))
    return run


def step(kind):
    def run(state, actions, t):
        import jax.numpy as j
        from gpu_engine import device_set_posterior as dsp, engine_device_public_hooks as hooks
        from test_gpu_engine_v2e_regress import result
        if kind == 'off':return result(state, t)
        lanes = state.hp.shape[0];zero = j.zeros(lanes, j.int32);false = zero.astype(bool);use = ~false
        if kind == 'no_use':use = false
        tables = SimpleNamespace(source_species=j.asarray([0, 1], j.int32), posterior=True)
        c = hooks.Context(t, tables)
        args = (state, 0, zero, zero, false, false, false, zero + 1, zero + 2, zero + 1, zero + 1, zero + 1, use)
        if kind == 'no_hook':
            c.trackers = producer('correct')(c.trackers, tables, *args)[0]
        else:hooks.set_posterior_strike(c, *args)
        if kind in ('multi', 'late_refusal', 'late_hook_refusal'):
            second = (state, 1 if kind == 'multi' else 0, *args[2:])
            hooks.set_posterior_strike(c, *second, pursuit=True)
        if kind == 'hp_terminal':state = state._replace(hp=state.hp.at[:, 1, 0].set(0), done=j.ones(lanes, bool))
        return result(state, c.trackers)
    return run


class StrictPassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import v2e_regress as regress
        from gpu_engine.v2e_capture import leaf_paths
        from test_gpu_engine_v2e_regress import lanes_state, CALLABLE, VOCAB
        cls.temp = tempfile.TemporaryDirectory(prefix='narrowing-audit-');cls.root = Path(cls.temp.name)
        shards = cls.root / 'capture' / 'shards';shards.mkdir(parents=True)
        st = lanes_state(2, 0)._replace(done=j.zeros(2, bool))
        tree = (st, j.zeros((2, 2), j.int32), trackers(2))
        cls.tree = tree
        cls.template = jax.tree.map(lambda x: x[:1], tree)
        np.savez_compressed(shards / 'shard-000000.npz', tick=np.int64(0), level=np.int64(0),
            **{f'l{i:03d}': np.asarray(x) for i, x in enumerate(jax.tree.leaves(tree))})
        (shards.parent / 'capture.json').write_text(json.dumps(dict(shards=1, paths=leaf_paths(cls.template))))
        cls.context = dict(prepared=SimpleNamespace(engine_tables=SimpleNamespace(sleep_talk=SimpleNamespace(callable=CALLABLE))),
                           pool=SimpleNamespace(metadata=lambda: dict(vocabulary=VOCAB)))
        cls.base = cls.root / 'base'
        cls.prepared = (cls.context, cls.template)
        regress.base_pass(cls.args(out=cls.base, pad_lanes=2), step_override=step('off'), prepared=cls.prepared)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    @classmethod
    def args(cls, **extra):
        values = dict(config=None, reference_run=None, shards=str(cls.root / 'capture' / 'shards'), limit=0, arm='B',
            allow_dirty=True, integration=None, allow_base_mismatch=True, waive=None, faint_clear=False)
        values.update(extra);return Namespace(**values)

    def run_pass(self, kind, pr='NAR'):
        from gpu_engine import v2e_regress as regress, device_set_posterior as dsp, engine_device_public_hooks as hooks
        dest = self.root / (self._testMethodName + '-' + kind + '-' + pr)
        base, prepared, extra = self.base, self.prepared, {}
        if kind in ('invalid_mapping', 'absent'):
            # The malformed member is in BOTH captured inputs: only the planted pin writes may differ.
            import jax
            from gpu_engine.v2e_capture import leaf_paths
            t = self.tree[2]
            t = (t._replace(present=t.present.at[:, 0, 0].set(False)) if kind == 'absent' else
                 t._replace(physical_to_public=t.physical_to_public.at[:, 0, 0].set(-1)))
            tree = (*self.tree[:2], t);template = jax.tree.map(lambda x: x[:1], tree)
            shards = self.root / (kind + '-capture') / 'shards';shards.mkdir(parents=True)
            np.savez_compressed(shards / 'shard-000000.npz', tick=np.int64(0), level=np.int64(0),
                **{f'l{i:03d}': np.asarray(x) for i, x in enumerate(jax.tree.leaves(tree))})
            (shards.parent / 'capture.json').write_text(json.dumps(dict(shards=1, paths=leaf_paths(template))))
            base = self.root / (kind + '-base');prepared = (self.context, template);extra['shards'] = str(shards)
            regress.base_pass(self.args(out=base, pad_lanes=2, **extra), step_override=step('off'), prepared=prepared)
        hook = hooks.set_posterior_strike
        def outer(c, *args, **kwargs):
            out = hook(c, *args, **kwargs)
            if kind == 'late_hook_refusal' and kwargs.get('pursuit', False):
                out._replace(trackers=producer('correct')(out.trackers, out.belief_tables, *args, **kwargs)[0])
            return out
        with mock.patch.object(dsp, 'in_tree', producer(kind)) as original, mock.patch.object(hooks, 'set_posterior_strike', outer):
            code = regress.patched_pass(self.args(out=dest, base=str(base), pr=pr, **extra),
                                        step_override=step(kind), prepared=prepared)
            self.assertIs(dsp.in_tree, original)  # the trace-time wrapper must always restore its target
            self.assertIs(hooks.set_posterior_strike, outer)
        return code, json.loads((dest / 'report.json').read_text()), [json.loads(x) for x in (dest / 'changed.jsonl').read_text().splitlines()]

    def test_off_none_is_unchanged(self):
        code, report, rows = self.run_pass('off', 'none')
        self.assertEqual(code, 0);self.assertEqual(rows, []);self.assertEqual(report['counts']['unchanged'], 2)

    def test_same_forced_on_output_fails_without_plugin_and_passes_with_it(self):
        code, report, rows = self.run_pass('multi', 'none')
        self.assertEqual(code, 1);self.assertEqual(report['counts']['changed_but_uninvolved'], 2)
        self.assertTrue(all(set(r['leaves']) <= set(audit.PIN_LEAVES) for r in rows))
        code, report, rows = self.run_pass('multi')
        self.assertEqual(code, 0);self.assertTrue(report['narrowing_rules'])
        self.assertTrue(all(r['admitted_slots'] == [[0, 0], [1, 0]] for r in rows))

    def test_wrong_subject_and_absent_evidence_fail(self):
        for kind in ('wrong_side', 'wrong_slot', 'no_hook', 'refused', 'empty_fit', 'no_use', 'invalid_mapping', 'absent'):
            with self.subTest(kind=kind):
                code, report, rows = self.run_pass(kind)
                self.assertEqual(code, 1);self.assertEqual(report['failures'], 2)
                self.assertTrue(all(r['unexpected'] for r in rows))
                if kind in ('invalid_mapping', 'absent'):
                    self.assertEqual(report['counts']['narrowing_bad'], 2)
                    self.assertTrue(all(not r['admitted_slots'] for r in rows))
                    self.assertTrue(all(set(r['leaves']) <= set(audit.PIN_LEAVES) for r in rows))

    def test_refused_call_cannot_borrow_prior_admission(self):
        for kind in ('late_refusal', 'late_hook_refusal'):
            code, report, rows = self.run_pass(kind)
            self.assertEqual(code, 1);self.assertEqual(report['counts']['narrowing_bad'], 2)
            self.assertTrue(all(r['admitted_slots'] == [[0, 0]] and r['narrowing_bad'] for r in rows))

    def test_forbidden_fields_fail_even_on_terminal_and_admitted_lanes(self):
        for kind, leaf in (('hp_terminal', 'state.hp'), ('visible_hard', 'trackers.visible_belief.visible_hard')):
            code, report, rows = self.run_pass(kind)
            self.assertEqual(code, 1);self.assertEqual(report['failures'], 2)
            self.assertTrue(all(any(s.startswith(leaf + '@') for s in r['unexpected']) for r in rows))

    def test_no_broadening_options(self):
        for plugins, extra in ((['NAR', 'D'], {}), (['none', 'NAR'], {}), (['NAR'], {'waive': ['.*']}),
                               (['NAR'], {'faint_clear': True})):
            with self.assertRaises(ValueError):audit.validate_options(plugins, **extra)
        audit.validate_options(['NAR']);audit.validate_options(['D', 'WH'], waive=['x'])


if __name__ == '__main__':unittest.main()
