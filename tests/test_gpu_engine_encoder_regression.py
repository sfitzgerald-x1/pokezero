"""Fail-closed admission of numeric regression witnesses, without an engine run."""
import ast
import copy
from collections import namedtuple
import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from gpu_engine.v2e_regress import ENCODER_SHAPES, ENCODER_WITNESS_VERSION, encoder_rule
from gpu_engine import encoder_regression_witness as ew
from gpu_engine import encoder_regression_trace as et


def evidence():
    expected = {n: np.full(shape, np.float32(0.25)) for n, shape in ENCODER_SHAPES.items()}
    digests = {n: np.array([[1, 2], [3, 4]], np.uint32) for n in ENCODER_SHAPES}
    return dict(version=ENCODER_WITNESS_VERSION, events_complete=True, arrays_valid=True, outside_equal=True,
                actual=copy.deepcopy(expected), expected=expected,
                legacy=copy.deepcopy(digests), base=digests)


def fold_pre():
    pre = {n: np.zeros((1,) + shape, np.float32) for n, shape in ENCODER_SHAPES.items()}
    mapping = np.full((1, 2, 6), -1, np.int32)
    mapping[0, :, 0] = [2, 4]
    pre.update({ew.HISTORY_HP: np.full((1, 2), 100, np.int32),
                ew.HISTORY_MAXIMUM: np.full((1, 2), 100, np.int32),
                ew.MAPPING: mapping, ew.NEXT_SLOT: np.array([[3, 5]], np.int32),
                ew.TURN: np.array([1], np.int32)})
    return pre


def tape(*events):
    schema, payloads = [], []
    for seq, (kind, parent, values) in enumerate(events):
        fields = ew.EVENT_FIELDS[kind]
        payloads.append({n: np.asarray([values[n]], dtype=d) for n, (d, _) in fields.items()})
        schema.append(dict(seq=seq, kind=kind, parent=parent, guards=ew.EXPECTED_GUARDS[kind].copy(),
                           fields=tuple((n, d, s) for n, (d, s) in fields.items())))
    return schema, payloads


def observation(hp=(99, 100), previous=(100, 100), *, use=True, record=True):
    return dict(use=use, record_damage=record, hp=hp, maximum=[100, 100],
                history_hp=previous, history_maximum=[100, 100])


class EncoderEventFoldTests(unittest.TestCase):
    def test_ratio_residual_signs_and_hp_domain(self):
        positive, negative = ew.ratio_pair(1, 100), ew.ratio_pair(10, 100)
        self.assertGreater(positive[1], 0)
        self.assertLess(negative[1], 0)
        self.assertEqual(positive[0], np.float32(0.01))
        self.assertEqual(negative[0], np.float32(0.1))
        self.assertEqual(ew.percent(713, 714), 99)
        self.assertEqual(ew.percent(714, 714), 100)
        for n, d in ((-1, 100), (1, -1), (1, 715)):
            with self.assertRaises(ValueError):ew.ratio_pair(n, d)

    def test_observe_zero_increment_normalizes_even_when_use_is_false(self):
        pre = fold_pre();pre[ew.TAKEN][0, :, :, :] = [0.5, 0.25]
        saved = copy.deepcopy(pre)
        schema, events = tape(('OBSERVE', None, observation(use=False)))
        folded = ew.fold_encoder(pre, schema, events)
        self.assertEqual(folded.errors, ())
        np.testing.assert_array_equal(folded.expected[ew.TAKEN][0], np.tile([0.75, 0], (2, 2, 1)))
        for n in pre:np.testing.assert_array_equal(pre[n], saved[n])

    def test_false_record_updates_history_without_recording_loss(self):
        pre = fold_pre()
        schema, events = tape(('OBSERVE', None, observation(record=False)),
                              ('OBSERVE', None, observation((98, 100), (99, 100))))
        folded = ew.fold_encoder(pre, schema, events)
        self.assertEqual(folded.errors, ())
        self.assertEqual(folded.expected[ew.TAKEN][0, 0, 1].sum(dtype=np.float32), np.float32(0.01))
        self.assertEqual(folded.event_counts, {'OBSERVE': 2})

    def test_history_snapshots_are_checks_not_new_seeds(self):
        pre = fold_pre()
        schema, events = tape(('OBSERVE', None, observation(previous=(99, 100))))
        folded = ew.fold_encoder(pre, schema, events)
        self.assertEqual(len(folded.errors), 1)
        self.assertEqual(folded.errors[0]['lane'], 0)
        self.assertIn('continuity', folded.errors[0]['detail'])

    def test_direct_uses_taken_operand_and_victim_public_slot(self):
        pre = fold_pre();shown = np.zeros((2, 6), np.int32);shown[0, 2] = 100
        schema, events = tape(('DIRECT', None, dict(side=1, use=True, dealt=77, taken=1,
            active=[0, 0], hp=99, maximum=100, hp_percent=shown, mapping=pre[ew.MAPPING][0])))
        folded = ew.fold_encoder(pre, schema, events)
        self.assertEqual(folded.errors, ())
        np.testing.assert_array_equal(folded.expected[ew.DEALT][0, 0], 0)
        np.testing.assert_array_equal(folded.expected[ew.DEALT][0, 1].sum(axis=-1, dtype=np.float32),
                                      np.array([0.01, 0.01], np.float32))
        np.testing.assert_array_equal(folded.expected[ew.TAKEN], pre[ew.TAKEN])

    def test_spikes_mask_preserves_pair_and_arrival_preserves_hazards(self):
        pre = fold_pre();pre[ew.HAZARDS][0, 0] = [0.5, 0.25]
        mapping = pre[ew.MAPPING][0].copy()
        schema, events = tape(
            ('SPIKES', None, dict(side=0, use=False, physical=0, hp=100, maximum=100, new_hp=90)),
            ('ARRIVE', None, dict(side=0, use=True, initial=False, active=[0, 0], hp=100, maximum=100,
                                   mapping=mapping, next_slot=pre[ew.NEXT_SLOT][0])))
        folded = ew.fold_encoder(pre, schema, events)
        self.assertEqual(folded.errors, ())
        np.testing.assert_array_equal(folded.expected[ew.HAZARDS], pre[ew.HAZARDS])

    def test_spikes_updates_only_its_side(self):
        pre = fold_pre()
        schema, events = tape(('SPIKES', None, dict(side=1, use=True, physical=0, hp=100, maximum=100, new_hp=90)))
        folded = ew.fold_encoder(pre, schema, events)
        self.assertEqual(folded.errors, ())
        np.testing.assert_array_equal(folded.expected[ew.HAZARDS][0, 0], 0)
        np.testing.assert_array_equal(folded.expected[ew.HAZARDS][0, 1].sum(axis=-1, dtype=np.float32),
                                      np.array([0.1, 0.1], np.float32))

    def test_fresh_slot_then_rearrival_clear_only_entrant_slot(self):
        pre = fold_pre()
        for n in ENCODER_SHAPES:pre[n].fill(np.float32(0.25))
        old = pre[ew.MAPPING][0].copy();new = old.copy();new[0, 3] = 3
        schema, events = tape(
            ('ARRIVE', None, dict(side=0, use=True, initial=False, active=[3, 0], hp=100, maximum=100,
                                   mapping=old, next_slot=[3, 5])),
            ('ARRIVE', None, dict(side=0, use=True, initial=False, active=[3, 0], hp=90, maximum=100,
                                   mapping=new, next_slot=[4, 5])))
        folded = ew.fold_encoder(pre, schema, events)
        self.assertEqual(folded.errors, ())
        for n, _, _ in ew.SETTLED:
            expected = pre[n].copy();expected[0, 0, 3] = 0
            np.testing.assert_array_equal(folded.expected[n], expected)
        np.testing.assert_array_equal(folded.expected[ew.TAKEN][0, 0], 0)
        np.testing.assert_array_equal(folded.expected[ew.TAKEN][0, 1], pre[ew.TAKEN][0, 1])
        np.testing.assert_array_equal(folded.expected[ew.HAZARDS], pre[ew.HAZARDS])

    def boundary_tape(self):
        pre = fold_pre();mapping = pre[ew.MAPPING][0]
        boundary = dict(before_done=False, after_done=False, pending_residual=True, forced_switch=[False, False],
                        after_turn=1, incoming_turn=1, ordinary=False)
        completed = dict(boundary, pending_residual=False)
        schema, events = tape(
            ('BOUNDARY', None, boundary),
            ('OBSERVE', 0, observation((90, 90))),
            ('TURN', 0, dict(use=False, number=2, active=[0, 0], mapping=mapping)),
            ('BOUNDARY', None, completed),
            ('OBSERVE', 3, observation((90, 90), (90, 90))),
            ('TURN', 3, dict(use=True, number=2, active=[0, 0], mapping=mapping)))
        return pre, schema, events

    def test_deferred_boundary_then_replacement_settlement_without_ordinary(self):
        pre, schema, events = self.boundary_tape()
        folded = ew.fold_encoder(pre, schema, events)
        self.assertEqual(folded.errors, ())
        for n in (ew.TAKEN, ew.DEALT):np.testing.assert_array_equal(folded.expected[n], 0)
        for n in ("trackers.mon_features['last_damage_taken']", 'trackers.damage_exact_taken'):
            expected = np.zeros((1, 2, 6), np.float32);expected[0, 0, 2] = expected[0, 1, 4] = np.float32(0.1)
            np.testing.assert_array_equal(folded.expected[n], expected)

    def test_wrong_turn_mask_and_incomplete_guards_fail_closed(self):
        pre, schema, events = self.boundary_tape();events[-1]['use'][0] = False
        self.assertIn('mask/number', ew.fold_encoder(pre, schema, events).errors[0]['detail'])
        pre, schema, events = self.boundary_tape();schema[-1]['guards']['settle'] = 0
        result = ew.fold_encoder(pre, schema, events)
        self.assertIsNone(result.errors[0]['lane'])
        self.assertIn('guards', result.errors[0]['detail'])

    def test_full_array_validation_finds_unlisted_bench_corruption(self):
        pre = fold_pre();folded = ew.fold_encoder(pre, (), ())
        actual = copy.deepcopy(folded.expected);name = 'trackers.damage_exact_dealt'
        actual[name][0, 1, 5] = np.float32(0.5)
        errors = ew.encoder_array_errors(actual, folded)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['coordinate'], (1, 5))
        self.assertEqual(errors[0]['actual_bits'], '3f000000')
        self.assertEqual(errors[0]['expected_bits'], '00000000')

    def test_full_array_validation_rejects_missing_arrays_and_signed_zero(self):
        pre = fold_pre();folded = ew.fold_encoder(pre, (), ())
        actual = copy.deepcopy(folded.expected);del actual[ew.HAZARDS]
        self.assertIsNone(ew.encoder_array_errors(actual, folded)[0]['lane'])
        actual = copy.deepcopy(folded.expected);actual[ew.HAZARDS][0, 1, 0, 1] = np.float32(-0.0)
        self.assertEqual(ew.encoder_array_errors(actual, folded)[0]['coordinate'], (1, 0, 1))

    def witness_inputs(self):
        pre = fold_pre();schema, events = tape(('OBSERVE', None, observation()))
        folded = ew.fold_encoder(pre, schema, events)
        baseline = {n: np.zeros((1, 2, 2), np.uint32) for n in ENCODER_SHAPES}
        trace = dict(schema=schema, events=events, actual=copy.deepcopy(folded.expected),
                     legacy=copy.deepcopy(baseline), outside_equal=np.ones(1, bool), event_inputs_equal=np.ones(1, bool))
        return pre, trace, baseline

    def test_witness_builder_explains_only_with_both_obligations(self):
        pre, trace, base = self.witness_inputs()
        receipt = ew.make_encoder_witness(pre, trace, base)
        self.assertEqual(receipt['errors'], [])
        entry = (ew.TAKEN, 0)
        self.assertEqual(encoder_rule({entry}, receipt['lanes'][0]), (set(), [entry], []))
        trace['legacy'][ew.TAKEN][0, 0, 0] = np.uint32(1)
        receipt = ew.make_encoder_witness(pre, trace, base)
        self.assertEqual(encoder_rule({entry}, receipt['lanes'][0]), ({entry}, [], [entry]))

    def test_witness_builder_retains_error_without_any_changed_entries(self):
        pre, trace, base = self.witness_inputs()
        trace['actual']['trackers.damage_exact_taken'][0, 1, 5] = np.float32(0.5)
        receipt = ew.make_encoder_witness(pre, trace, base)
        self.assertEqual(receipt['errors'][0]['coordinate'], (1, 5))
        self.assertIs(receipt['lanes'][0]['arrays_valid'], False)
        self.assertTrue(receipt['lanes'][0]['errors'])

    def test_witness_builder_rejects_outside_and_raw_operand_mismatches(self):
        for flag in ('outside_equal', 'event_inputs_equal'):
            with self.subTest(flag=flag):
                pre, trace, base = self.witness_inputs();trace[flag][0] = False
                receipt = ew.make_encoder_witness(pre, trace, base)
                self.assertTrue(receipt['lanes'][0]['errors'])
                self.assertIs(receipt['lanes'][0]['events_complete'], False)


class EncoderRegressionRuleTests(unittest.TestCase):
    def test_both_seats_need_corrected_arrays_and_baseline_shadow(self):
        entries = {(n, s) for n in ENCODER_SHAPES for s in (0, 1)}
        remaining, explained, forced = encoder_rule(entries, evidence())
        self.assertEqual(remaining, set())
        self.assertEqual(set(explained), entries)
        self.assertEqual(forced, [])
        self.assertEqual(len(entries), 14)

    def test_unrelated_leaf_has_no_encoder_allowance(self):
        n = 'trackers.damage_history.taken'
        unrelated = ('state.hp', 0)
        remaining, explained, forced = encoder_rule({(n, 0), unrelated}, evidence())
        self.assertEqual(remaining, {unrelated})
        self.assertEqual(explained, [(n, 0)])
        self.assertEqual(forced, [])

    def test_low_limb_corruption_cannot_hide_behind_same_readout(self):
        w = evidence();n = 'trackers.damage_history.taken'
        w['actual'][n][0, 0] = [np.float32(0.5), np.float32(2**-30)]
        w['expected'][n][0, 0] = [np.float32(0.5), np.float32(0)]
        self.assertEqual(np.sum(w['actual'][n][0, 0], dtype=np.float32),
                         np.sum(w['expected'][n][0, 0], dtype=np.float32))
        self.assert_forced(w, (n, 0))

    def test_wrong_slot_and_untouched_cell_are_checked(self):
        n = 'trackers.damage_exact_taken'
        for slot in (0, 5):
            with self.subTest(slot=slot):
                w = evidence();w['actual'][n][1, slot] = np.float32(0.5)
                self.assert_forced(w, (n, 1))

    def test_wrong_side_cannot_use_other_seats_witness(self):
        w = evidence();n = 'trackers.damage_exact_dealt'
        w['actual'][n][1, 0] = np.float32(0.5)
        remaining, explained, forced = encoder_rule({(n, 0), (n, 1)}, w)
        self.assertEqual(remaining, {(n, 1)})
        self.assertEqual(explained, [(n, 0)])
        self.assertEqual(forced, [(n, 1)])

    def test_shadow_mismatch_is_forced_even_when_corrected_array_matches(self):
        w = evidence();n = 'trackers.public_port.hazards'
        w['legacy'][n][0, 0] ^= np.uint32(1)
        self.assert_forced(w, (n, 0))

    def test_missing_or_incomplete_evidence_is_forced(self):
        n = 'trackers.damage_history.dealt'
        for key in ('version', 'events_complete', 'arrays_valid', 'outside_equal', 'actual', 'expected', 'legacy', 'base'):
            with self.subTest(key=key):
                w = evidence();del w[key]
                self.assert_forced(w, (n, 0))
        for key in ('events_complete', 'arrays_valid', 'outside_equal'):
            with self.subTest(key=key):
                w = evidence();w[key] = False
                self.assert_forced(w, (n, 0))

    def test_wrong_shape_dtype_signed_zero_and_nonfinite_are_forced(self):
        n = 'trackers.damage_exact_taken'
        for defect in ('shape', 'dtype', 'signed_zero', 'nonfinite'):
            with self.subTest(defect=defect):
                w = evidence()
                if defect == 'shape':w['actual'][n] = w['actual'][n][:, :1]
                elif defect == 'dtype':w['actual'][n] = w['actual'][n].astype(np.float64)
                elif defect == 'signed_zero':
                    w['actual'][n][0, 0] = np.float32(-0.0)
                    w['expected'][n][0, 0] = np.float32(0.0)
                else:
                    w['actual'][n][0, 0] = np.float32(np.inf)
                    w['expected'][n][0, 0] = np.float32(np.inf)
                self.assert_forced(w, (n, 0))

    def assert_forced(self, w, entry):
        remaining, explained, forced = encoder_rule({entry}, w)
        self.assertEqual(remaining, {entry})
        self.assertEqual(explained, [])
        self.assertEqual(forced, [entry])


class EncoderLegacySourceTests(unittest.TestCase):
    def test_frozen_bodies_retain_baseline_ast_identity(self):
        identity = et.legacy_source_identity()
        self.assertEqual(identity['commit'], '24f39abaa478fd11d27188353ccb0fe6a3f471f0')
        self.assertEqual(set(identity['bodies']), {'_add', 'observe', 'direct', 'damage', 'spikes'})

    def test_ast_identity_ignores_only_empty_new_type_parameter_field(self):
        original = ast.parse('def f(x): return x + 1').body[0]
        later = copy.deepcopy(original)
        if 'type_params' not in later._fields:later._fields += ('type_params',)
        later.type_params = []
        self.assertEqual(et._ast_identity(original), et._ast_identity(later))
        changed = copy.deepcopy(later);changed.body[0].value.right.value = 2
        self.assertNotEqual(et._ast_identity(original), et._ast_identity(changed))
        later.type_params = [ast.Name(id='T', ctx=ast.Load())]
        self.assertNotEqual(et._ast_identity(original), et._ast_identity(later))


@unittest.skipUnless(importlib.util.find_spec('jax') is not None, 'requires JAX')
class EncoderTraceTests(unittest.TestCase):
    def inputs(self):
        import jax.numpy as j
        from gpu_engine.engine_device_damage_history import History
        pre = {n: j.asarray(a) for n, a in fold_pre().items()}
        t = SimpleNamespace(damage_history=History(pre[ew.HISTORY_HP], pre[ew.HISTORY_MAXIMUM], pre[ew.TAKEN], pre[ew.DEALT]),
            public_port=SimpleNamespace(hazards=pre[ew.HAZARDS]), physical_to_public=pre[ew.MAPPING],
            next_slot=pre[ew.NEXT_SLOT], turn=pre[ew.TURN], damage_exact_taken=pre['trackers.damage_exact_taken'],
            damage_exact_dealt=pre['trackers.damage_exact_dealt'], hp_percent=j.full((1, 2, 6), 100, j.int32),
            mon_features={name: pre[f"trackers.mon_features['{name}']"] for name in ('last_damage_taken', 'last_damage_dealt')})
        hp = j.full((1, 2, 6), 100, j.int32).at[0, 0, 0].set(99)
        state = SimpleNamespace(hp=hp, max_hp=j.full_like(hp, 100), active=j.zeros((1, 2), j.int32))
        return state, j.zeros((1, 2), j.int32), t

    def attributes(self):
        from gpu_engine import engine_device_damage_history as h, engine_device_public_hooks as hooks
        from gpu_engine import engine_device_public_port as port
        return [(m, n, getattr(m, n)) for m, names in (
            (h, ('arrive', 'observe', 'direct', 'settle')),
            (hooks, ('install', 'observe', 'damage', 'public_turn', 'turn_end', 'feature', '_write')),
            (port, ('spikes',))) for n in names]

    def assert_restored(self, saved):
        for module, name, original in saved:self.assertIs(getattr(module, name), original)

    def test_recorder_restores_all_attributes_after_exception(self):
        state, actions, t = self.inputs();saved = self.attributes()
        def broken(*args):raise RuntimeError('planted trace failure')
        with self.assertRaisesRegex(RuntimeError, 'planted trace failure'):
            et.trace_encoder_call(broken, state, actions, t, mode='corrected')
        self.assert_restored(saved)

    def test_unknown_writer_fails_and_restores(self):
        from gpu_engine import engine_device_damage_history as h
        import jax.numpy as j
        state, actions, t = self.inputs();saved = self.attributes()
        def broken(st, acts, trackers):return h.settle(trackers.damage_history, j.ones(1, j.bool_))
        with self.assertRaisesRegex(ValueError, 'outside its recorded hook'):
            et.trace_encoder_call(broken, state, actions, t, mode='corrected')
        self.assert_restored(saved)

    def test_arithmetic_context_restores_after_exception(self):
        saved = self.attributes()
        with self.assertRaisesRegex(RuntimeError, 'planted compile failure'):
            with et.encoder_arithmetic('legacy'):
                raise RuntimeError('planted compile failure')
        self.assert_restored(saved)

    def test_corrected_legacy_corrected_compilations_are_isolated(self):
        import jax
        import jax.numpy as j
        from gpu_engine import engine_device_damage_history as h, engine_device_trackers as trackers
        state, _, t = self.inputs();programs = [];saved = self.attributes()
        for mode in ('corrected', 'legacy', 'corrected'):
            with et.encoder_arithmetic(mode):
                def fresh(value):return h.observe(value, state, j.ones(1, j.bool_), trackers._percent)
                programs.append(jax.jit(fresh).lower(t.damage_history).compile())
            self.assert_restored(saved)
        values = [np.asarray(p(t.damage_history).taken) for p in programs]
        np.testing.assert_array_equal(values[0].view(np.uint32), values[2].view(np.uint32))
        self.assertGreater(values[0][0, 0, 0, 1], 0)
        self.assertEqual(values[1][0, 0, 0, 1], 0)

    def test_recorded_observation_folds_without_post_value_seeding(self):
        import jax
        import jax.numpy as j
        from gpu_engine import engine_device_damage_history as h, engine_device_public_hooks as hooks
        from gpu_engine import engine_device_trackers as trackers
        state, actions, t = self.inputs();saved = self.attributes()
        def observe(c, st, use, *, record_damage=True, rest_side=None, rest_mask=False):
            new = SimpleNamespace(**vars(c.trackers))
            new.damage_history = h.observe(c.trackers.damage_history, st, use, trackers._percent, record_damage=record_damage)
            c.trackers = new
            return c
        def step(st, acts, inp):return hooks.observe(SimpleNamespace(trackers=inp), st, j.ones(1, j.bool_))
        with mock.patch.object(hooks, 'observe', observe):
            build = et.trace_encoder_call(step, state, actions, t, mode='corrected')
        folded = ew.fold_encoder(jax.device_get(build.pre), build.schema, jax.device_get(build.events))
        actual = {n: a for n, a in jax.device_get(et.encoder_pre(build.out.trackers)).items() if n in ENCODER_SHAPES}
        self.assertEqual(folded.errors, ())
        self.assertEqual(ew.encoder_array_errors(actual, folded), ())
        self.assertEqual(len(build.schema), 1)
        self.assertEqual(build.schema[0]['guards'], {'observe': 1})
        self.assert_restored(saved)

    def test_compiled_comparison_sees_scalar_signed_zero_for_every_lane(self):
        import jax
        import jax.numpy as j
        def build(mode):
            def program(leaves):
                x = leaves[0];lanes = x.shape[0]
                outputs = {n: j.zeros((lanes,) + shape, j.float32) for n, shape in ENCODER_SHAPES.items()}
                outputs['global_scalar'] = j.float32(-0.0 if mode == 'legacy' else 0.0)
                return dict(digests=j.full((lanes, 1, 2, 2), int(mode == 'legacy'), j.uint32),
                            _encoder=dict(outputs=outputs, pre={'operand': x}, events=({'use': x > 0},)))
            return program
        leaves = [j.ones(4, j.int32)]
        programs = et.compile_encoder_programs(build, leaves, source_identity=et.encoder_source_identity())
        result = jax.device_get(programs.execute(leaves))
        np.testing.assert_array_equal(result['encoder']['outside_equal'], np.zeros(4, bool))
        np.testing.assert_array_equal(result['encoder']['event_inputs_equal'], np.ones(4, bool))
        np.testing.assert_array_equal(result['encoder']['legacy_digests'], np.ones((4, 1, 2, 2), np.uint32))
        self.assertNotIn('_encoder', result)
        self.assertNotIn('global_scalar', result['encoder']['actual'])

    def test_stepper_encoder_programs_return_valid_independent_fold(self):
        import jax
        import jax.numpy as j
        from gpu_engine import engine_device_damage_history as h, engine_device_public_hooks as hooks
        from gpu_engine import engine_device_trackers as trackers
        from gpu_engine.v2e_regress import Stepper
        state, actions, inp = self.inputs();saved = self.attributes()
        State = namedtuple('EncoderState', 'hp max_hp active done')
        state = State(state.hp, state.max_hp, state.active, j.zeros(1, j.bool_))
        fields = dict(vars(inp));fields['public_port'] = namedtuple('EncoderPort', 'hazards')(inp.public_port.hazards)
        fields['active_slot'] = j.array([[2, 4]], j.int32)
        Trackers = namedtuple('EncoderTrackers', fields)
        inp = Trackers(**fields)
        Step = namedtuple('EncoderStep', 'state invalid_actions placeholder_moves')
        Result = namedtuple('EncoderResult', 'step trackers')
        def observe(c, st, use, *, record_damage=True, rest_side=None, rest_mask=False):
            history = h.observe(c.trackers.damage_history, st, use, trackers._percent, record_damage=record_damage)
            return c._replace(trackers=c.trackers._replace(damage_history=history))
        def step(st, acts, t):
            c = hooks.observe(hooks.Context(t), st, j.ones(1, j.bool_))
            return Result(Step(st, j.zeros(1, j.bool_), j.zeros(1, j.bool_)), c.trackers)
        tree = (state, actions, inp)
        with mock.patch.object(hooks, 'observe', observe):
            stepper = Stepper(step, jax.tree.structure(tree), encoder=True)
            result = stepper(jax.tree.leaves(tree))
        trace = result['encoder'];folded = ew.fold_encoder(trace['pre'], trace['schema'], trace['events'])
        self.assertEqual(ew.encoder_array_errors(trace['actual'], folded), ())
        np.testing.assert_array_equal(trace['outside_equal'], [True])
        np.testing.assert_array_equal(trace['event_inputs_equal'], [True])
        self.assertEqual(len(trace['schema']), 1)
        self.assert_restored(saved)


if __name__ == '__main__':
    unittest.main()
