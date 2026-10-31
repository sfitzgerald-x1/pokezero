"""Benchmark option/timing contracts; no checkpoint, model forward or game."""
from contextlib import ExitStack
from collections import namedtuple
from dataclasses import dataclass
import gc
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import benchmark_retained_games as bench, champion, precision_benchmark
from gpu_engine import engine_retained_encoder_driver as driver
import test_gpu_engine_benchmark_retained_games as fixtures


@dataclass(eq=False)
class PreparedStub:
    binding: str = 'synthetic-prepared'


class BenchmarkSourceWitnessTests(unittest.TestCase):
    def preparation(self, **options):
        """Replace every archive/device boundary; only preparation is exercised."""
        request, own, prepared = object(), object(), PreparedStub()
        config, sentinel = object(), object()
        metadata = dict(weights_archive_sha256='a' * 64, checkpoint_sha256='b' * 64)
        order = []
        with ExitStack() as stack:
            stack.enter_context(patch.object(bench.engine_own_facts, '_checked', return_value={
                'request_compiled': request, 'catalog': NS(checkpoint_sha256='b' * 64)}))
            digest = stack.enter_context(patch.object(precision_benchmark, 'digest',
                side_effect=lambda _: order.append('digest') or 'a' * 64))
            stack.enter_context(patch.object(bench, 'sole_gpu', return_value=jax.devices('cpu')[0]))
            load = stack.enter_context(patch.object(champion, 'load',
                side_effect=lambda _: order.append('load') or (j.array(1), config, metadata)))
            stack.enter_context(patch.object(champion, 'Config', return_value=config))
            prepare = stack.enter_context(patch.object(driver, 'prepare',
                side_effect=lambda *a, **kw: order.append('prepare') or prepared))
            bind = stack.enter_context(patch.object(bench, '_bind', return_value=sentinel))
            actual = bench.prepare(request, own, '/never-open-an-archive',
                expected_weights_sha256='a' * 64, expected_checkpoint_sha256='b' * 64,
                enable_unqualified=True, **options)
        self.assertIs(actual, sentinel)
        self.assertIs(bind.call_args.args[0], prepared)
        self.assertEqual(prepare.call_args.args, (request, own))
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual((digest.call_count, load.call_count, bind.call_count), (1, 1, 1))
        return prepare.call_args.kwargs, order

    def test_default_preparation_remains_closed_and_preserves_order(self):
        kwargs, order = self.preparation()
        self.assertEqual(kwargs, dict(public_mode='retained-full', fake_forward=bench._noop_fake))
        self.assertEqual(order, ['digest', 'load', 'prepare'])

    def test_original_helpers_modes_and_deliberate_flags_forward_once(self):
        leftovers, status, direct = object(), object(), object()
        for mode, kernel, accounting, risks in (
                ('closed', 'diagnostic', False, False),
                ('closed', 'kernel-event', True, False),
                ('source-witness', 'kernel-event', True, True)):
            with self.subTest(mode=mode, kernel=kernel):
                kwargs, order = self.preparation(leftovers_compiled=leftovers,
                    leftovers_mode='retained', status_residual_compiled=status,
                    status_residual_mode=kernel, direct_hit_compiled=direct,
                    direct_hit_mode=kernel, frame_input_mode=mode)
                self.assertIs(kwargs['leftovers_compiled'], leftovers)
                self.assertIs(kwargs['status_residual_compiled'], status)
                self.assertIs(kwargs['direct_hit_compiled'], direct)
                self.assertEqual((kwargs['leftovers_mode'], kwargs['status_residual_mode'],
                    kwargs['direct_hit_mode'], kwargs['frame_input_mode']), ('retained', kernel, kernel, mode))
                self.assertIs(kwargs['account_source_sites'], accounting)
                self.assertIs(kwargs['track_kernel_risks'], risks)
                self.assertEqual(order, ['prepare', 'digest', 'load'])

    def test_opt_in_and_original_driver_rejections_precede_archive_access(self):
        with patch.object(champion, 'load') as load, patch.object(precision_benchmark, 'digest') as digest:
            with self.assertRaisesRegex(ValueError, 'enablement'):
                bench.prepare(None, None, '/never-open', expected_weights_sha256='a' * 64,
                    expected_checkpoint_sha256='b' * 64, frame_input_mode='source-witness')
            request, own = object(), object()
            with patch.object(bench.engine_own_facts, '_checked', return_value={
                    'request_compiled': request, 'catalog': NS(checkpoint_sha256='b' * 64)}):
                # Use the actual driver for its early dependency/mode checks.
                for options, reason in ((dict(frame_input_mode='source-witness'), 'original Leftovers'),
                        (dict(frame_input_mode='claimed'), 'frame input mode'),
                        (dict(direct_hit_mode='kernel-event'), 'original helper'),
                        (dict(status_residual_mode='kernel-event'), 'original helper')):
                    with self.subTest(options=options), self.assertRaisesRegex(ValueError, reason):
                        bench.prepare(request, own, '/never-open', expected_weights_sha256='a' * 64,
                            expected_checkpoint_sha256='b' * 64, enable_unqualified=True, **options)
            load.assert_not_called()
            digest.assert_not_called()

    def test_runtime_retains_prepared_only_for_runtime_lifetime(self):
        config = champion.Config(width=8, heads=2, feedforward=16, layers=1)
        params, digest = precision_benchmark.random_parameters(config, 73)
        prepared = PreparedStub()
        origin = weakref.ref(prepared)
        runtime = bench._bind(prepared, params, config, {'tiny_synthetic_sha256': digest})
        live = weakref.ref(runtime)
        del prepared
        gc.collect()
        self.assertIsNotNone(origin())
        with patch.object(driver, '_checked', return_value={}) as checked:
            self.assertIs(bench._checked(runtime)['prepared'], origin())
            checked.assert_called_once_with(origin())
        del checked, runtime
        gc.collect()
        self.assertIsNone(live())
        self.assertIsNone(origin())

    def test_current_session_witness_guard_precedes_downstream_inference(self):
        first, config = fixtures.result(batch=1)
        session = NS(state=first.state, result=first)
        runtime = bench.Runtime('synthetic-runtime')
        bench._RUNTIMES[runtime] = dict(binding=runtime.binding,
            prepared=PreparedStub(), config=config)
        prepared_record = {'frame_input_mode': 'source-witness'}
        calls = []
        def guard(actual_record, *, session=None):
            self.assertIs(actual_record, prepared_record)
            calls.append(session)
            if session is not None:
                raise ValueError('original current session drift')
        with ExitStack() as stack:
            stack.enter_context(patch.object(driver, '_checked', return_value=prepared_record))
            stack.enter_context(patch.object(driver, '_checked_witness', side_effect=guard))
            stack.enter_context(patch.object(driver, 'construct', return_value=session))
            stack.enter_context(patch.object(bench.source_provenance, '_pool', return_value={}))
            stack.enter_context(patch.object(bench, '_source_snapshot', return_value={}))
            network = stack.enter_context(patch.object(bench, '_network'))
            advance = stack.enter_context(patch.object(driver, 'advance'))
            pool = NS(arrays={'species': np.zeros((2, 6), np.int32)})
            with self.assertRaisesRegex(ValueError, 'session drift'):
                bench.measure(runtime, pool, np.array([[0, 1]]),
                    stream_ids=np.array([101], np.int32), max_requests=1, enable_unqualified=True)
            network.assert_not_called()
            advance.assert_not_called()
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0])
        self.assertIs(calls[1], session)

    def measurement(self, *, mode, timing_names, closed_owner=False, evidence=None):
        reasons = np.array([[driver.PUBLIC_COVERAGE, 0], [0, 0], [0, 0]], np.int32)
        if closed_owner:
            reasons[:] = [0, driver.PUBLIC_COVERAGE]
        first, config = fixtures.result(reasons=reasons)
        if evidence is not None:
            first.state.core.frame_input_report, first.state.core.risks = evidence
        session = NS(state=first.state, result=first)
        runtime, prepared = NS(binding='synthetic-runtime'), PreparedStub()
        record = dict(prepared=prepared, config=config, metadata={'synthetic': True})
        pool = NS(arrays={'species': np.zeros((6, 6), np.int32)})
        def network(_, args):
            rows = len(args[0])
            return champion.Output(j.zeros((rows, 9)), j.zeros(rows), j.zeros((rows, 9))), 0., 0.
        def advance(actual_session, actions, *, active_rows, timing_callback):
            self.assertIs(actual_session, session)
            np.testing.assert_array_equal(active_rows, [False, True, True])
            self.assertEqual(actions.shape, (3, 2))
            next_result, _ = fixtures.result(frames=[0, 1, 1], done=[False, True, False],
                reasons=np.array([[driver.PUBLIC_COVERAGE, 0], [driver.ENCODER, driver.ENCODER], [0, 0]]))
            if evidence is not None:
                next_result.state.core.frame_input_report, next_result.state.core.risks = evidence
            session.state, session.result = next_result.state, next_result
            session.diagnostics = NS(active_rows=active_rows, executed_rows=active_rows,
                invalid_actions=j.zeros(3, bool), placeholder_moves=j.zeros(3, j.int32))
            for name in timing_names:
                timing_callback(name, 1.)
            return next_result
        with ExitStack() as stack:
            stack.enter_context(patch.object(bench, '_checked', return_value=record))
            stack.enter_context(patch.object(driver, '_checked', return_value={'frame_input_mode': mode}))
            stack.enter_context(patch.object(driver, 'construct', return_value=session))
            stack.enter_context(patch.object(bench.source_provenance, '_pool', return_value={'synthetic': True}))
            stack.enter_context(patch.object(bench, '_source_snapshot', return_value={'commit': 'synthetic'}))
            model = stack.enter_context(patch.object(bench, '_network', side_effect=network))
            advancing = stack.enter_context(patch.object(driver, 'advance', side_effect=advance))
            stack.enter_context(patch.object(driver, 'latest_diagnostics', side_effect=lambda _: session.diagnostics))
            measured = bench.measure(runtime, pool, np.arange(6).reshape(3, 2),
                stream_ids=np.arange(101, 104, dtype=np.int32), max_requests=1, enable_unqualified=True)
        return measured, model, advancing

    def test_pair_timing_preserves_finite_roster_and_unknowns_without_dual_folds(self):
        phases = ('engine', 'requests', 'public_pair', 'encode_p1', 'encode_p2', 'admission_p1', 'admission_p2')
        measured, model, advancing = self.measurement(mode='source-witness', timing_names=phases)
        self.assertEqual((measured['original_denominator'], measured['completed_games'],
            measured['refused_games'], measured['censored_games'], measured['unknown_games']), (3, 1, 1, 1, 2))
        self.assertEqual([row['stream_id'] for row in measured['cohort']], [101, 102, 103])
        self.assertEqual([row['executed_steps'] for row in measured['cohort']], [0, 1, 1])
        self.assertEqual(measured['model_owner_rows'], 4)
        self.assertEqual((model.call_count, advancing.call_count), (1, 1))
        self.assertEqual(set(measured['driver_components']), set(phases))
        self.assertEqual(measured['retained_encoder']['samples_ms'], [5.])
        self.assertFalse(any(measured[x] for x in
            ('full_B4_qualified', 'fidelity_qualified', 'strength_qualified')))

    def test_pair_mode_does_not_bypass_second_owner_coverage(self):
        measured, model, advancing = self.measurement(mode='source-witness', timing_names=(), closed_owner=True)
        self.assertEqual((measured['completed_games'], measured['refused_games'], measured['unknown_games']), (0, 3, 3))
        model.assert_not_called()
        advancing.assert_not_called()

    def test_final_raw_evidence_is_literal_and_does_not_change_roster_accounting(self):
        packet=namedtuple('Packet','refusal raw_words')(
            j.array([3,0,0],j.int32),j.arange(12,dtype=j.int32).reshape(3,2,2))
        risk=namedtuple('Risk','binding risk')(j.arange(8,dtype=j.uint32),j.array([7,0,0],j.int32))
        phases=('engine','requests','public_pair','encode_p1','encode_p2','admission_p1','admission_p2')
        measured,_,_=self.measurement(mode='source-witness',timing_names=phases,evidence=(packet,risk))
        self.assertEqual(measured['final_frame_input_witness'],
            {name:np.asarray(value).tolist() for name,value in packet._asdict().items()})
        self.assertEqual(measured['final_kernel_risks'],
            {name:np.asarray(value).tolist() for name,value in risk._asdict().items()})
        self.assertEqual((measured['original_denominator'],measured['completed_games'],
            measured['refused_games'],measured['censored_games'],measured['unknown_games']),(3,1,1,1,2))

    def test_timing_layout_must_match_original_prepared_mode(self):
        common = ('engine', 'requests', 'encode_p1', 'encode_p2', 'admission_p1', 'admission_p2')
        for mode, public in (('source-witness', ('public_p1', 'public_p2')),
                ('closed', ('public_pair',)), ('source-witness', ('public_pair', 'public_p1', 'public_p2'))):
            with self.subTest(mode=mode, public=public), self.assertRaisesRegex(RuntimeError, 'timing phases'):
                self.measurement(mode=mode, timing_names=common + public)


if __name__ == '__main__':
    unittest.main()
