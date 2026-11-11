"""Mapped original one-row arithmetic controls; fake engine/actor, no gate claim."""
from dataclasses import replace
from types import SimpleNamespace as NS
from typing import NamedTuple
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import device_public_teacher as t, device_public_world_state as mapping
from gpu_engine.v5_model import Output
from tests.test_gpu_engine_device_public_teacher import facts


class FakeObservation(NamedTuple):
    legal_action_mask: object
    error: object


class JointBatchTests(unittest.TestCase):
    def executor(self, owner, *, peer_requested=True, terminal=False,
                 invalid=None, encoder_error=None, model_error=None, actions=2):
        legal = j.arange(9) < actions

        def masks(state):
            result = j.broadcast_to(legal, (1, 2, 9)) & ~state.done[:, None, None]
            if not peer_requested:
                result = result.at[:, 1-owner].set(False)
            return result

        def observe(state, public, seat, mask):
            who = seat[0]
            # The fake engine encodes its pair in HP solely for these controls.
            code = state.hp[:, who, 0]
            a, r = (code-50)//10, (code-50)%10
            error = j.zeros((1,), j.int32) if encoder_error is None else \
                ((a == encoder_error[0]) & (r == encoder_error[1])).astype(j.int32)
            return NS(observation=FakeObservation(mask, error),
                inputs=dict(hp=code.astype(j.float32), turn=public.turn, terminal=state.done),
                observer_sides=seat)

        def advance(state, public, actions, chance):
            # Nonidentity request mapping must survive the batch composition.
            a = actions[:, owner]-1
            r = j.where(actions[:, 1-owner] < 0, 0, actions[:, 1-owner]-1)
            hp = 50+10*a+r
            done = (a == 1) if terminal else j.zeros((1,), bool)
            winner = j.where(r == 1, 2, owner).astype(j.int32)
            state = state._replace(hp=state.hp.at[:, owner, 0].set(hp), done=done,
                winner=winner, turn=state.turn+1)
            bad = j.zeros((1,), bool) if invalid is None else \
                (a == invalid[0]) & (r == invalid[1])
            return NS(step=NS(state=state, invalid_actions=bad),
                trackers=public._replace(turn=public.turn+chance[0]))

        def actor(inputs):
            hp = inputs['hp']
            bad = inputs['terminal']
            if model_error is not None:
                bad = bad | (hp == 50+10*model_error[0]+model_error[1])
            values = j.where(bad, j.nan, hp/512)
            return Output(j.broadcast_to(j.arange(9, dtype=j.float32), (1, 9)),
                values, None, ~bad)

        serial = t.Executor(masks, observe, lambda obs, state, seat, action: action+1,
            lambda state: j.asarray([1], j.int32), advance)
        batch = replace(serial, joint=t._make_joint(serial))
        return serial, batch, actor, jax.jit(jax.vmap(actor))

    def compare(self, owner=0, **options):
        context, root, starts, tables, recipe = facts(owner)
        serial, batch, actor, batched_actor = self.executor(owner, **options)
        world = mapping.materialize(root, starts.starts[0], tables, recipe)
        own = t._view(serial, world.state, world.public, owner)
        kwargs = dict(root_logits=(0.,)*9,
            root_legal=tuple(map(bool, np.asarray(own.observation.legal_action_mask)[0])),
            root_inputs=own.inputs, opponent_forward=actor, learner_forward=actor,
            enable_device_teacher=True)
        before = tuple(np.array(x) for x in jax.tree.leaves(context))
        reference = t.price(starts, tables, serial, recipe, **kwargs)
        result = t.price(starts, tables, batch, recipe, learner_batch_forward=batched_actor, **kwargs)
        self.assertEqual(reference.status, result.status)
        self.assertEqual(reference.reduction, result.reduction)
        self.assertEqual(reference.missing_worlds, result.missing_worlds)
        for a, b in zip(reference.world_reports, result.world_reports):
            for key in ('joint_values', 'required_pairs', 'completed_pairs', 'unknown_pairs', 'reply_logits', 'reply_legal'):
                self.assertEqual(a[key], b[key], key)
        for old, new in zip(before, jax.tree.leaves(context)):
            np.testing.assert_array_equal(old, new)
        self.assertEqual(result.actual_teacher_calls['chance'], 2)
        self.assertEqual(result.actual_teacher_calls['transitions'], 162)
        self.assertEqual(result.actual_teacher_calls['public_leaf'], 162)
        self.assertEqual(result.actual_teacher_calls['batch_invocations'], 2)
        return result, reference

    def test_both_owners_all_pair_values_reply_mass_and_immutable_root(self):
        for owner in (0, 1):
            result, reference = self.compare(owner)
            self.assertEqual(result.status, 'LABELED')
            self.assertEqual(result.reduction.required_pairs, 8)
            self.assertEqual(result.actual_teacher_calls['padding_transitions'], 154)
            self.assertEqual(reference.actual_teacher_calls['public_leaf'], 8)

    def test_all81_pairs_and_no_reply_request_keep_literal_denominators(self):
        result, _ = self.compare(actions=9)
        self.assertEqual(result.reduction.required_pairs, 162)
        self.assertEqual(result.actual_teacher_calls['padding_transitions'], 0)
        result, _ = self.compare(peer_requested=False)
        self.assertEqual(result.reduction.required_pairs, 4)
        self.assertEqual(result.actual_teacher_calls['reply_policy'], 0)
        self.assertEqual(result.actual_teacher_calls['padding_forward_rows'], 158)

    def test_real_terminal_scores_ignore_computed_invalid_terminal_predictions(self):
        for owner in (0, 1):
            result, reference = self.compare(owner, terminal=True)
            self.assertEqual(result.status, 'LABELED')
            for world in result.world_reports:
                self.assertEqual(world['joint_values'][1][:2], [1., 0.])
            self.assertEqual(result.actual_teacher_calls['terminal_forward_rows'], 4)
            self.assertEqual(reference.actual_teacher_calls['public_leaf'], 4)

    def test_each_required_error_stays_unknown_without_mass_renormalization(self):
        for key in ('invalid', 'encoder_error', 'model_error'):
            result, _ = self.compare(**{key: (0, 1)})
            self.assertEqual(result.status, 'REFUSED')
            self.assertEqual(result.reduction.unknown_pairs, 2)
            self.assertGreater(result.reduction.unknown_reply_mass, .5)
            for world in result.world_reports:
                self.assertIsNone(world['joint_values'][0][1])
                self.assertEqual(world['completed_pairs'], 3)

    def test_bad_batch_callback_shape_refuses_without_serial_fallback(self):
        _, root, starts, tables, recipe = facts()
        serial, batch, actor, batched = self.executor(0)
        world = mapping.materialize(root, starts.starts[0], tables, recipe)
        own = t._view(serial, world.state, world.public, 0)
        def wrong(inputs):
            out = batched(inputs)
            return out._replace(value=out.value[:, 0])
        result = t.price(starts, tables, batch, recipe, root_logits=(0.,)*9,
            root_legal=(True, True, *([False]*7)), root_inputs=own.inputs,
            opponent_forward=actor, learner_forward=actor,
            learner_batch_forward=wrong, enable_device_teacher=True)
        self.assertEqual(result.status, 'REFUSED')
        self.assertIsNone(result.reduction)
        self.assertEqual(result.missing_worlds, 2)
        self.assertEqual(result.actual_teacher_calls['batch_invocations'], 2)
        self.assertTrue(all('geometry' in x['reason'] for x in result.world_reports))

    def test_factory_batching_requires_a_literal_bool_and_stays_default_off(self):
        with self.assertRaisesRegex(ValueError, 'batching bool'):
            t.make_executor(None, None, enable_device_teacher=True, batch_joint=1)
        serial = t.make_executor(None, NS(belief=None), enable_device_teacher=True)
        self.assertIsNone(serial.joint)
        batch = t.make_executor(None, NS(belief=None), enable_device_teacher=True, batch_joint=True)
        self.assertTrue(callable(batch.joint))


if __name__ == '__main__':
    unittest.main()
