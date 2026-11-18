"""Changed-behavior guards for the complete joint handoff budget."""
import gc
import json
import random
import threading
import unittest
from unittest.mock import patch

import pokezero_search as native
from pokezero.joint_budget import defer_joint_gc
from pokezero.joint_generator import JointGenerator
from pokezero.joint_native import NativeJointPosterior, draw_with_stage1_fallback, own_team_terminal_weights
from pokezero.joint_serving import JointPlan, JointSession
from tests.test_depthn_joint_generator import fixture, brute


class GCDeferral(unittest.TestCase):
    def setUp(self): self.enabled = gc.isenabled()
    def tearDown(self): (gc.enable if self.enabled else gc.disable)()

    def test_nested_exception_restores_both_initial_states_without_collecting(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch.object(gc, 'collect') as collect:
                (gc.enable if enabled else gc.disable)()
                with self.assertRaisesRegex(RuntimeError, 'failure'):
                    with defer_joint_gc():
                        self.assertFalse(gc.isenabled())
                        with defer_joint_gc(): self.assertFalse(gc.isenabled())
                        raise RuntimeError('failure')
                self.assertEqual(gc.isenabled(), enabled)
                collect.assert_not_called()

    def test_overlapping_users_restore_only_after_last_exit_in_either_order(self):
        for first_exit in (0, 1):
            with self.subTest(first_exit=first_exit):
                gc.enable()
                entered = [threading.Event(), threading.Event()]
                leave = [threading.Event(), threading.Event()]
                exited = [threading.Event(), threading.Event()]
                errors = []
                def worker(i):
                    try:
                        with defer_joint_gc():
                            entered[i].set()
                            if not leave[i].wait(5): raise RuntimeError('test synchronization timeout')
                    except Exception as error: errors.append(error)
                    finally: exited[i].set()
                threads = [threading.Thread(target=worker,args=(i,)) for i in range(2)]
                try:
                    for i,t in enumerate(threads): t.start(); self.assertTrue(entered[i].wait(5))
                    leave[first_exit].set(); self.assertTrue(exited[first_exit].wait(5))
                    self.assertFalse(gc.isenabled())
                    leave[1-first_exit].set(); self.assertTrue(exited[1-first_exit].wait(5))
                finally:
                    for event in leave: event.set()
                    for t in threads:
                        if t.ident is not None: t.join(5)
                self.assertFalse(errors)
                self.assertTrue(gc.isenabled())


class SharedDeadline(unittest.TestCase):
    def test_expansion_cutoff_returns_enclosing_integer_bounds(self):
        for flag in (False, True):
            parts = fixture(slots=2,rich=True)
            posterior = NativeJointPosterior(*parts,evidence={'a':(1,0)},initial_wobbuffet=flag)
            clock = native.JointDeadline(1)
            posterior.native.bind_deadline(clock,0,.5)
            result = json.loads(posterior.native.solve(.5,64))
            reference = JointGenerator(*parts,evidence={'a':(1,0)},initial_wobbuffet=flag)
            mass = sum(brute(*parts,evidence={'a':(1,0)},initial_wobbuffet=flag).values()) * reference.quantum**reference.slots
            self.assertLessEqual(int(result['lower']), mass)
            self.assertGreaterEqual(int(result['upper']), mass)
            self.assertTrue(result['cut'])

    def test_cancelled_or_hard_aborted_attempt_cannot_emit_or_revive(self):
        posterior = NativeJointPosterior(*fixture(slots=2),evidence={'a':(1,0)})
        clock = native.JointDeadline(1)
        # Binding an already-expired phase aborts immediately and returns no certificate.
        with self.assertRaises(TimeoutError): posterior.native.bind_deadline(clock,0,0)
        for operation in (lambda:posterior.native.solve(.1,64),lambda:posterior.native.draw_paths(1,2),
                posterior.native.revealed_context_masses,
                lambda:posterior.native.bind_deadline(native.JointDeadline(1),.1,.2)):
            with self.assertRaises(TimeoutError): operation()

    def test_unexpired_clock_preserves_exact_counts_and_frozen_exports(self):
        parts = fixture(slots=2,rich=True)
        deadline = native.JointDeadline(.25)
        posterior = NativeJointPosterior(*parts,evidence={'a':(1,0)},deadline=deadline)
        result = posterior.solve(seconds=.1)
        reference = JointGenerator(*parts,evidence={'a':(1,0)})
        expected = sum(brute(*parts,evidence={'a':(1,0)}).values()) * reference.quantum**reference.slots
        self.assertEqual(int(result['lower']),expected)
        self.assertEqual(result['lower'],result['upper'])
        before = posterior.revealed_marginal_weights()
        posterior.draw_many(8,random.Random(4))
        self.assertEqual(posterior.revealed_marginal_weights(),before)
        deadline.cancel()
        with self.assertRaises(TimeoutError): posterior.feature_payload()
        with self.assertRaises(TimeoutError): posterior.draw_many(1,random.Random(4))

    def test_cancelled_reference_cache_and_both_arm_conditioning_do_not_return_zero(self):
        parts = fixture()
        clock = native.JointDeadline(1)
        reference = JointGenerator(*parts,deadline=clock)
        reference.normalizer(); clock.cancel()
        with self.assertRaises(TimeoutError): reference.normalizer()
        clock = native.JointDeadline(1)
        original = JointGenerator.normalizer
        flags = []
        def first_arm(generator):
            flags.append(generator.initial_wobbuffet)
            result = original(generator)
            clock.cancel()
            return result
        with patch.object(JointGenerator,'normalizer',first_arm), self.assertRaises(TimeoutError):
            own_team_terminal_weights(*parts,(('a',1),('b',1),('c',0)),deadline=clock)
        self.assertEqual(flags,[False])

    def test_expiry_after_rng_consumption_restores_exact_stage1_draws(self):
        parts = fixture(slots=2)
        clock = native.JointDeadline(.25)
        original = NativeJointPosterior.draw_many
        calls = []
        def consume_then_expire(posterior,count,rng):
            result = original(posterior,count,rng); clock.cancel(); return result
        def stage1(count,rng):
            calls.append(count); return tuple(rng.getrandbits(64) for _ in range(count))
        expected_rng = random.Random(73)
        expected = tuple(expected_rng.getrandbits(64) for _ in range(32))
        with patch('pokezero.joint_native.validate_pinned_catalog'), patch.object(NativeJointPosterior,'draw_many',consume_then_expire):
            result = draw_with_stage1_fallback(*parts,count=32,rng=random.Random(73),stage1_sampler=stage1,
                battle_context=dict(owner='p1',own_initial_team=(('a',0),('b',0)),producer='independent-teams'),
                deadline=clock,export_posterior=True,evidence={'a':(1,0)})
        self.assertEqual(result.mode,'stage1_fallback')
        self.assertEqual(result.teams,expected)
        self.assertIsNone(result.features)
        self.assertEqual(calls,[32])

    def test_late_finalization_selects_complete_fallback_once(self):
        fallback = JointPlan(None,{'stage1':'restricted'}, {'payload':'stage1'},
            dict(mode='stage1_fallback',fallback=True,reason=None,seconds=0.,over_budget=False),'source','game')
        candidate = JointPlan((('joint',),),{}, {'payload':'joint'},
            dict(mode='exact',fallback=False,reason=None,seconds=0.,over_budget=False),'source','game',fallback)
        clock = native.JointDeadline(0)
        selected = JointSession.finish_handoff(candidate,clock)
        self.assertIs(selected,fallback)
        self.assertEqual(selected.posts,{'stage1':'restricted'})
        self.assertEqual(selected.feature_payload,{'payload':'stage1'})
        self.assertTrue(selected.diagnostic['over_budget'])
        self.assertIs(JointSession.finish_handoff(selected,clock),fallback)

    def test_sampler_failure_still_restores_gc(self):
        enabled = gc.isenabled()
        try:
            gc.enable()
            with self.assertRaisesRegex(RuntimeError,'sampler failure'):
                with defer_joint_gc():
                    draw_with_stage1_fallback(*fixture(slots=2),count=1,rng=random.Random(1),
                        stage1_sampler=lambda count,rng:(_ for _ in()).throw(RuntimeError('sampler failure')),
                        battle_context=dict(owner='p1',own_initial_team=(('a',0),('b',0)),producer='independent-teams'),
                        deadline=native.JointDeadline(0))
            self.assertTrue(gc.isenabled())
        finally:
            (gc.enable if enabled else gc.disable)()
