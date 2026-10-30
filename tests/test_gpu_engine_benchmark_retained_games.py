"""Finite-cohort B4 producer contracts; synthetic tensors/tiny random weights only."""
from contextlib import ExitStack
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import benchmark_retained_games as bench, champion, precision_benchmark
from gpu_engine import engine_retained_encoder_driver as driver
from gpu_engine import engine_private_actions as actions, engine_request_order as order
from gpu_engine import private_actions, private_requests


def result(batch=3, *, reasons=None, waiting=(), forced=(), frames=None, done=None, permutation=None):
    """Literal owning packets; no engine construction, checkpoint or root data."""
    config = champion.Config(width=8, heads=2, feedforward=16, layers=1)
    streams = j.arange(101, 101 + batch, dtype=j.int32)
    binding = j.zeros((batch, 8), j.uint32)
    reports, requests, ledgers, observers = [], [], [], []
    for side in (0, 1):
        owners = j.full(batch, side, j.int32)
        ledger = order.initialize_order(streams, owners, j.full(batch, 6, j.int32), engine_binding=binding)
        if permutation is not None:
            ledger = ledger._replace(request_to_physical=j.asarray(np.broadcast_to(permutation, (batch, 6))))
        boundary = order.RequestBoundary(streams, owners, binding, j.zeros(batch, j.int32),
            ledger.request_to_physical[:, 0])
        wait = j.asarray([side in waiting for _ in range(batch)])
        force = j.asarray([side in forced for _ in range(batch)])
        request = private_requests.ActionRequest(owners, j.ones(batch, bool), j.zeros(batch, j.int32),
            j.where(force, private_requests.RequestKind.FORCE_SWITCH, private_requests.RequestKind.MOVE).astype(j.int32),
            wait, force, j.zeros(batch, bool), j.full(batch, 6, j.int32),
            j.broadcast_to(j.array([True, False, False, False, False, False]), (batch, 6)),
            j.zeros((batch, 6), bool), j.ones((batch, 4), bool), j.zeros((batch, 4), bool),
            j.ones((batch, 4), j.float32), j.ones((batch, 4), j.float32), j.ones((batch, 4), bool))
        inputs = private_requests.action_inputs(request, observer_side=owners)
        af = NS(request=request)
        actual = actions.Result(af, private_actions.ActionFeatures(None, None, inputs.legal, j.ones(batch, bool)),
            j.broadcast_to(j.arange(4, dtype=j.int32), (batch, 4)), j.ones((batch, 9), bool), boundary,
            j.ones(batch, bool))
        why = j.asarray(reasons[:, side] if reasons is not None else np.zeros(batch), j.int32)
        ok = why == 0
        obs = NS(numeric_features=j.zeros((batch, 23, 132), j.float32),
            categorical_ids=j.zeros((batch, 23, 41), j.int32),
            token_type_ids=j.zeros((batch, 23), j.int32), attention_mask=j.ones((batch, 23), bool),
            legal_action_mask=inputs.legal, switch_targets=inputs.switch_targets, admitted=ok)
        reports.append(NS(encoded=NS(observation=obs), reason=why, policy_admitted=ok,
            full_line_consumers_complete=j.ones(batch, bool),
            basic_error=j.zeros(batch, j.int32), retained_error=j.zeros(batch, j.int32)))
        requests.append(NS(actions=actual)); ledgers.append(ledger)
        observers.append(NS(owner=owners, retained_healthy=j.ones(batch, bool)))
    core = NS(cursor=NS(stream_id=streams), frame_number=j.asarray(frames if frames is not None else np.zeros(batch), j.int32),
        engine=NS(done=j.asarray(done if done is not None else np.zeros(batch), bool), turn=j.ones(batch, j.int32)),
        requests=tuple(requests), orders=tuple(ledgers), admitted=j.ones(batch, bool))
    return driver.Result(NS(core=core, observers=tuple(observers)), tuple(reports)), config


class RetainedBenchmarkTests(unittest.TestCase):
    def runtime(self, config):
        params, digest = precision_benchmark.random_parameters(config, 73)
        return bench._bind(NS(binding='test-only-driver'), params, config, {'tiny_synthetic_sha256': digest})

    def measure(self, first, config, *, advance=None, network=None, cap=2):
        runtime = self.runtime(config)
        session = NS(state=first.state, result=first)
        batch = len(first.state.core.frame_number)
        pool = NS(arrays={'species': np.zeros((2 * batch, 6), np.int32)})
        with ExitStack() as stack:
            stack.enter_context(patch.object(driver, '_checked', return_value={}))
            stack.enter_context(patch.object(driver, 'construct', return_value=session))
            stack.enter_context(patch.object(bench.source_provenance, '_pool', return_value={'literal': True}))
            stack.enter_context(patch.object(bench, '_source_snapshot', return_value={'commit': 'synthetic'}))
            model = stack.enter_context(patch.object(bench, '_network', side_effect=network))
            advancing = stack.enter_context(patch.object(driver, 'advance', side_effect=advance))
            # Root owns this new shared-driver API; source tests pin its contract.
            stack.enter_context(patch.object(driver, 'latest_diagnostics', create=True,
                side_effect=lambda _: session.diagnostics))
            measured = bench.measure(runtime, pool, np.arange(2 * batch).reshape(batch, 2),
                stream_ids=np.arange(101, 101 + batch, dtype=np.int32),
                max_requests=cap, enable_unqualified=True)
        return measured, model, advancing

    def test_opt_in_refuses_before_checkpoint_or_constructor_access(self):
        with patch.object(champion, 'load') as load, patch.object(driver, 'construct') as construct:
            with self.assertRaisesRegex(ValueError, 'enablement'):
                bench.prepare(None, None, '/no/checkpoint', expected_weights_sha256='a'*64,
                    expected_checkpoint_sha256='b'*64)
            with self.assertRaisesRegex(ValueError, 'enablement'):
                bench.measure(None, None, None, stream_ids=None)
            load.assert_not_called(); construct.assert_not_called()

    def test_closed_admission_has_zero_model_calls_steps_or_completions(self):
        first, config = result(reasons=np.full((3, 2), driver.PUBLIC_COVERAGE))
        measured, model, advance = self.measure(first, config)
        self.assertEqual((measured['completed_games'], measured['refused_games'], measured['unknown_games']), (0, 3, 3))
        self.assertEqual(measured['original_denominator'], 3)
        self.assertIsNone(measured['games_per_second'])
        self.assertIsNone(measured['retained_encoder']['mean_ms'])
        self.assertIsNone(measured['encoder_fraction_of_instrumented_full_step'])
        model.assert_not_called(); advance.assert_not_called()
        self.assertFalse(measured['full_B4_qualified'])

    def test_tiny_real_champion_bf16_all_heads_and_shape_cache(self):
        first, config = result(batch=1)
        runtime = self.runtime(config)
        record = bench._RUNTIMES[runtime]
        eligible, failed, _ = bench._admission(first, np.ones(1, bool), config)
        self.assertFalse(failed.any())
        routed, failed, cost = bench._decide(record, first, eligible, 123)
        self.assertFalse(failed.any())
        self.assertEqual(routed.shape, (1, 2))
        self.assertEqual(cost['model_rows'], 2)
        self.assertEqual(set(record['networks']), {2})
        _, _, again = bench._decide(record, first, eligible, 123)
        self.assertEqual(again['compile_ms'], 0)

    def test_compacts_owners_waits_and_uses_existing_permuted_force_router(self):
        first, config = result(batch=2, waiting=(1,), forced=(0,), permutation=np.array([2, 0, 1, 3, 4, 5], np.int32))
        runtime = self.runtime(config); record = bench._RUNTIMES[runtime]
        eligible, failed, _ = bench._admission(first, np.ones(2, bool), config)
        self.assertEqual(eligible.tolist(), [[True, False], [True, False]])
        def model(_, args):
            self.assertEqual(args[0].shape[0], 2)
            logits = j.zeros((2, 9)).at[:, 4].set(1000)
            return champion.Output(logits, j.zeros(2), j.zeros((2, 9))), 2., 3.
        with patch.object(bench, '_network', side_effect=model):
            routed, failed, cost = bench._decide(record, first, eligible, 42)
        self.assertEqual(routed.tolist(), [[4, 0], [4, 0]])
        self.assertFalse(failed.any())
        self.assertEqual(cost['model_rows'], 2)

    def test_sampling_keys_independent_of_survivor_order_and_owner(self):
        keys = bench._keys(9, np.array([101, 102, 101]), np.array([4, 4, 4]), np.array([0, 0, 1]))
        alone = bench._keys(9, np.array([102]), np.array([4]), np.array([0]))
        np.testing.assert_array_equal(keys[1], alone[0])
        self.assertFalse(np.array_equal(keys[0], keys[2]))
        self.assertFalse(np.array_equal(keys[0], bench._keys(9, [101], [5], [0])[0]))

    def test_refusal_isolates_original_row_and_terminal_counts_only_after_advance(self):
        first, config = result(reasons=np.array([[256, 0], [0, 0], [0, 0]]))
        calls = []
        def model(_, args):
            count = len(args[0]); calls.append(count)
            return champion.Output(j.zeros((count, 9)), j.zeros(count), j.zeros((count, 9))), 11., 12.
        def advance(session, engine_actions, *, active_rows, timing_callback):
            mask = np.asarray(active_rows)
            self.assertEqual(mask.tolist(), [False, True, True])
            self.assertEqual(engine_actions.shape, (3, 2))
            next_result, _ = result(frames=mask.astype(np.int32), done=[False, True, False],
                reasons=np.array([[256, 0], [256, 256], [0, 0]]))
            session.state, session.result = next_result.state, next_result
            session.diagnostics = NS(active_rows=active_rows, executed_rows=active_rows,
                invalid_actions=j.zeros(3, bool), placeholder_moves=j.ones(3, j.int32))
            for name in ('engine', 'requests', 'public_p1', 'public_p2', 'encode_p1', 'encode_p2', 'admission_p1', 'admission_p2'):
                timing_callback(name, .001)
            return next_result
        measured, model_spy, advance_spy = self.measure(first, config, network=model, advance=advance, cap=1)
        self.assertEqual(calls, [4])
        self.assertEqual((measured['completed_games'], measured['refused_games'], measured['censored_games']), (1, 1, 1))
        self.assertEqual(measured['unknown_games'], 2)
        self.assertEqual([x['status'] for x in measured['cohort']], ['UNKNOWN_REFUSED', 'COMPLETED', 'UNKNOWN_CENSORED'])
        self.assertEqual(measured['cold_compile']['samples_ms'], [11.])
        self.assertEqual(measured['retained_encoder']['samples_ms'], [.006])
        self.assertEqual(advance_spy.call_count, 1)

    def test_nonfinite_opponent_head_freezes_only_its_physical_row(self):
        first, config = result(batch=2)
        runtime = self.runtime(config)
        def model(_, args):
            return champion.Output(j.zeros((4, 9)), j.zeros(4), j.zeros((4, 9)).at[2, 0].set(j.nan)), 0., 1.
        with patch.object(bench, '_network', side_effect=model):
            _, failed, costs = bench._decide(bench._RUNTIMES[runtime], first, np.ones((2, 2), bool), 7)
        self.assertEqual(failed.tolist(), [False, True])
        self.assertEqual(costs['nonfinite_owner_rows'], 1)

    def test_invalid_engine_terminal_is_unknown_not_completed(self):
        first, config = result(batch=1)
        def model(_, args):
            return champion.Output(j.zeros((2, 9)), j.zeros(2), j.zeros((2, 9))), 0., 1.
        def advance(session, actions, *, active_rows, timing_callback):
            out, _ = result(batch=1, frames=[1], done=[True])
            session.state, session.result = out.state, out
            session.diagnostics = NS(active_rows=active_rows, executed_rows=active_rows,
                invalid_actions=j.ones(1, bool), placeholder_moves=j.zeros(1, j.int32))
            for name in ('engine', 'requests', 'public_p1', 'public_p2', 'encode_p1', 'encode_p2', 'admission_p1', 'admission_p2'):
                timing_callback(name, .001)
            return out
        measured, _, _ = self.measure(first, config, network=model, advance=advance, cap=1)
        self.assertEqual(measured['completed_games'], 0)
        self.assertEqual(measured['refused_games'], 1)
        self.assertTrue(measured['cohort'][0]['refusal']['invalid_action'])

    def test_stale_or_wrong_legal_observation_refuses_before_model(self):
        first, config = result(batch=1)
        first.owners[0].encoded.observation.legal_action_mask = j.zeros((1, 9), bool)
        eligible, failed, _ = bench._admission(first, np.ones(1, bool), config)
        self.assertTrue(failed[0]); self.assertFalse(eligible.any())

    def test_model_does_not_accept_zero_eligible_rows(self):
        first, config = result(batch=1)
        runtime = self.runtime(config)
        with patch.object(bench, '_network') as model, self.assertRaisesRegex(ValueError, 'forbidden'):
            bench._decide(bench._RUNTIMES[runtime], first, np.zeros((1, 2), bool), 7)
        model.assert_not_called()

    def test_completed_rows_never_forward_again_and_survivors_keep_original_positions(self):
        first, config = result(reasons=np.array([[256, 0], [0, 0], [0, 0]]))
        calls = []
        def model(_, args):
            count = len(args[0]); calls.append(count)
            return champion.Output(j.zeros((count, 9)), j.zeros(count), j.zeros((count, 9))), 0., 1.
        def advance(session, engine_actions, *, active_rows, timing_callback):
            turn = len(calls)
            self.assertEqual(np.asarray(active_rows).tolist(), [False, turn == 1, True])
            frames = [0, 1, turn]
            out, _ = result(frames=frames, done=[False, True, turn == 2],
                reasons=np.array([[256, 0], [256, 256], [0, 0]]))
            session.state, session.result = out.state, out
            session.diagnostics = NS(active_rows=active_rows, executed_rows=active_rows,
                invalid_actions=j.zeros(3, bool), placeholder_moves=j.ones(3, j.int32))
            for name in ('engine', 'requests', 'public_p1', 'public_p2', 'encode_p1', 'encode_p2', 'admission_p1', 'admission_p2'):
                timing_callback(name, .001)
            return out
        measured, _, _ = self.measure(first, config, network=model, advance=advance, cap=3)
        self.assertEqual(calls, [4, 2])
        self.assertEqual((measured['completed_games'], measured['refused_games'], measured['censored_games']), (2, 1, 0))
        self.assertEqual([x['engine_placeholder_moves'] for x in measured['cohort']], [0, 1, 2])

    def test_no_requested_owner_is_unknown_even_with_admitted_encoders(self):
        first, config = result(batch=1, waiting=(0, 1))
        eligible, failed, _ = bench._admission(first, np.ones(1, bool), config)
        self.assertTrue(failed[0]); self.assertFalse(eligible.any())

    def test_initial_done_is_not_a_new_completed_game(self):
        first, config = result(batch=1, done=[True])
        measured, model, advance = self.measure(first, config)
        self.assertEqual((measured['completed_games'], measured['refused_games']), (0, 1))
        model.assert_not_called(); advance.assert_not_called()

    def test_requested_but_unexecuted_row_is_unknown_while_other_row_completes(self):
        first, config = result(batch=2)
        def model(_, args):
            return champion.Output(j.zeros((4, 9)), j.zeros(4), j.zeros((4, 9))), 0., 1.
        def advance(session, actions, *, active_rows, timing_callback):
            self.assertEqual(np.asarray(active_rows).tolist(), [True, True])
            # The first row stays frozen. Its terminal bit cannot certify a
            # completion even if a malformed/unknown fixture presents one.
            out, _ = result(batch=2, frames=[0, 1], done=[True, True])
            session.state, session.result = out.state, out
            session.diagnostics = NS(active_rows=active_rows, executed_rows=j.array([False, True]),
                invalid_actions=j.zeros(2, bool), placeholder_moves=j.array([7, 1], j.int32))
            for name in ('engine', 'requests', 'public_p1', 'public_p2', 'encode_p1', 'encode_p2', 'admission_p1', 'admission_p2'):
                timing_callback(name, .001)
            return out
        measured, _, _ = self.measure(first, config, network=model, advance=advance, cap=1)
        self.assertEqual((measured['completed_games'], measured['refused_games']), (1, 1))
        self.assertEqual(measured['executed_game_steps'], 1)
        self.assertFalse(measured['cohort'][0]['refusal']['executed'])
        self.assertEqual([x['executed_steps'] for x in measured['cohort']], [0, 1])
        self.assertEqual(measured['engine_placeholder_moves'], 1)

    def test_terminal_public_frame_refusal_prevents_completion(self):
        first, config = result(batch=2)
        def model(_, args):
            return champion.Output(j.zeros((4, 9)), j.zeros(4), j.zeros((4, 9))), 0., 1.
        def advance(session, actions, *, active_rows, timing_callback):
            # Both engine transitions terminate; only row 1's public frame is
            # healthy. Missing next owning requests alone are expected at done.
            out, _ = result(batch=2, frames=[1, 1], done=[True, True], reasons=np.full((2, 2), driver.REQUEST))
            out.owners[0].retained_error = j.array([1, 0], j.int32)
            for owner in out.owners:
                owner.full_line_consumers_complete = j.zeros(2, bool)
            session.state, session.result = out.state, out
            session.diagnostics = NS(active_rows=active_rows, executed_rows=active_rows,
                invalid_actions=j.zeros(2, bool), placeholder_moves=j.zeros(2, j.int32))
            for name in ('engine', 'requests', 'public_p1', 'public_p2', 'encode_p1', 'encode_p2', 'admission_p1', 'admission_p2'):
                timing_callback(name, .001)
            return out
        measured, _, _ = self.measure(first, config, network=model, advance=advance, cap=1)
        self.assertEqual((measured['completed_games'], measured['refused_games']), (1, 1))
        self.assertFalse(measured['cohort'][0]['refusal']['public_frame_admitted'])
        self.assertEqual(measured['cohort'][1]['final_owner_full_line_consumers_complete'], [False, False])


if __name__ == '__main__':
    unittest.main()
