"""Reference leaf isolation, terminal laws, deadline and restoration contracts."""
from copy import deepcopy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.paper_reference import (
    Evaluation, LeafDeadlineExceeded, ReferenceConfig, ReferenceRefusal, TrajectorySearch)
from pokezero.mcts_eval.paper_reference_parallel import PreparedDecision
from pokezero.mcts_eval.paper_reference_showdown import ShowdownTrajectoryWorld, decision_state
from pokezero.mcts_eval.search_over_raw_leaves import (
    ReferenceLeafWorld, ReferenceLeafWorkerFactory, _ReferenceLeafRuntime, hp_value, validate_leaf_work, ROLLOUT_CAP)
from tests.test_paper_reference_showdown import observation
from tests.test_paper_reference import ROOT, ToyWorld


def evaluator(obs):
    return (0, 1), Evaluation((.4, .6), .3)


class LeafEnv:
    _search_snapshot_permitted = True

    def __init__(self, rounds=2, capped=False, winner="p1"):
        self.index, self.rounds, self.capped = 0, rounds, capped
        self.winner = winner
        self.actions, self.seeds, self.released = [], [], []

    def observe(self, seat):
        own = deepcopy(observation(seat))
        own.numeric_features = ((float(self.index),),)
        return own

    def requested_players(self):
        return ("p1", "p2") if self.index < self.rounds else ()

    def terminal(self):
        return SimpleNamespace(winner=self.winner, capped=self.capped) if self.index >= self.rounds else None

    def snapshot(self):
        return SimpleNamespace(bridge_snapshot={"battle": {"sides": [
            {"pokemon": [{"hp": 30, "maxhp": 100}, {"hp": 90, "maxhp": 100}]},
            {"pokemon": [{"hp": 80, "maxhp": 100}, {"hp": 20, "maxhp": 100}]}]}})

    def snapshot_for_search(self):
        return self.index

    def restore_search_snapshot(self, snapshot):
        self.index = snapshot

    def release_search_snapshot(self, snapshot):
        self.released.append(snapshot)
        return True

    def reseed_simulator_rng(self, seed):
        self.seeds.append(seed)

    def step(self, actions):
        self.actions.append(dict(actions))
        self.index += 1


class ReferenceLeafTests(unittest.TestCase):
    def world(self, env=None, leaf="raw_rollout", **kwargs):
        env = env or LeafEnv()
        underlying = ShowdownTrajectoryWorld(env, subject="p1", evaluator=evaluator, release=lambda: None)
        receipts = []
        world = ReferenceLeafWorld(underlying, leaf=leaf, deadline=lambda: None, receipts=receipts, **kwargs)
        self.addCleanup(world.close)
        state = world.frame().subject
        return env, world, state, receipts

    def test_hp_is_sum_weighted_and_subject_signed(self):
        snapshot = LeafEnv().snapshot()
        self.assertAlmostEqual(hp_value(snapshot, "p1"), .1)
        self.assertAlmostEqual(hp_value(snapshot, "p2"), -.1)

    def test_hp_rejects_invalid_or_incomplete_world_ledger(self):
        snapshot = LeafEnv().snapshot()
        snapshot.bridge_snapshot["battle"]["sides"][0]["pokemon"][0]["hp"] = float("nan")
        with self.assertRaisesRegex(ValueError, "HP ledger"):
            hp_value(snapshot, "p1")

    def test_hp_changes_only_value_not_champion_priors(self):
        env, world, state, receipts = self.world(leaf="hp_fraction")
        result = world.evaluate(state)
        self.assertEqual(result.priors, (.4, .6))
        self.assertAlmostEqual(result.value, .1)
        self.assertEqual(env.index, 0)
        self.assertEqual(receipts[0]["model_signed_value"], .3)

    def test_raw_leaf_uses_argmax_for_both_seats_and_restores(self):
        env, world, state, receipts = self.world()
        result = world.evaluate(state)
        self.assertEqual(result, Evaluation((.4, .6), 1.))
        self.assertEqual(env.actions, [{"p1": 1, "p2": 1}]*2)
        self.assertEqual(env.index, 0)
        self.assertEqual(env.released, [0])
        self.assertEqual(receipts[0]["boundaries"], 2)
        self.assertEqual(receipts[0]["continuation_policy"], "raw_argmax_both_seats")

    def test_raw_rollout_seed_is_deterministic_and_separate(self):
        env, world, state, _ = self.world(rollout_seed=9)
        world.evaluate(state)
        first = list(env.seeds)
        world.evaluate(state)
        self.assertEqual(env.seeds[2:], first)
        other, alternative, other_state, _ = self.world(rollout_seed=10)
        alternative.evaluate(other_state)
        self.assertNotEqual(other.seeds, first)

    def test_draw_and_opponent_win_are_actual_signed_terminals(self):
        for winner, expected in ((None, 0.), ("p2", -1.)):
            env, world, state, receipts = self.world(LeafEnv(winner=winner))
            self.assertEqual(world.evaluate(state).value, expected)
            self.assertEqual(receipts[0]["alternative_signed_value"], expected)
            self.assertEqual(env.index, 0)

    def test_second_seat_leaf_uses_second_seat_perspective(self):
        env, receipts = LeafEnv(), []
        underlying = ShowdownTrajectoryWorld(env, subject="p2", evaluator=evaluator, release=lambda: None)
        world = ReferenceLeafWorld(underlying, leaf="raw_rollout", deadline=lambda: None, receipts=receipts)
        self.addCleanup(world.close)
        self.assertEqual(world.evaluate(world.frame().subject).value, -1.)

    def test_cap_is_a_refusal_not_an_hp_blend(self):
        env, world, state, receipts = self.world(LeafEnv(rounds=3), cap=1)
        with self.assertRaisesRegex(ReferenceRefusal, "no fallback"):
            world.evaluate(state)
        self.assertEqual(env.index, 0)
        self.assertEqual(receipts[0]["status"], "REFUSED")
        self.assertIsNone(receipts[0]["alternative_signed_value"])

    def test_capped_simulator_terminal_is_refused(self):
        env, world, state, receipts = self.world(LeafEnv(capped=True))
        with self.assertRaisesRegex(ReferenceRefusal, "not a terminal"):
            world.evaluate(state)
        self.assertEqual(env.index, 0)
        self.assertEqual(receipts[0]["status"], "REFUSED")

    def test_deadline_cancel_has_no_value_and_restores(self):
        env, world, state, receipts = self.world()
        world.deadline = lambda: 1.
        with patch("pokezero.mcts_eval.search_over_raw_leaves.time.perf_counter", return_value=2.):
            with self.assertRaises(LeafDeadlineExceeded):
                world.evaluate(state)
        self.assertEqual(env.index, 0)
        self.assertEqual(env.released, [0])
        self.assertEqual(receipts[0]["status"], "DEADLINE_CANCELLED")
        self.assertFalse(receipts[0]["backed_up"])

    def test_world_cannot_be_live_source_or_uniform_rollout(self):
        with self.assertRaisesRegex(ValueError, "owned hypothetical"):
            ReferenceLeafWorld(object(), leaf="raw_rollout", deadline=lambda: None, receipts=[])
        underlying = ShowdownTrajectoryWorld(LeafEnv(), subject="p1", evaluator=evaluator, release=lambda: None)
        with self.assertRaisesRegex(ValueError, "invalid leaf"):
            ReferenceLeafWorld(underlying, leaf="uniform", deadline=lambda: None, receipts=[])

    def test_snapshot_is_released_even_if_restore_fails(self):
        env, world, state, _ = self.world()
        env.restore_search_snapshot = lambda _: (_ for _ in ()).throw(RuntimeError("restore failed"))
        with self.assertRaisesRegex(RuntimeError, "restore failed"):
            world.evaluate(state)
        self.assertEqual(env.released, [0])

    def test_kernel_cancels_leaf_without_backing_up_partial_trajectory(self):
        now = [0.]
        search = TrajectorySearch(ReferenceConfig(.5, 1.))
        search.reset_battle("leaf-cancel")
        world = ToyWorld(depth=2, terminal=False)

        def cancelled(_):
            now[0] = 2.
            raise LeafDeadlineExceeded()

        world.evaluate = cancelled
        result = search.search_batch(ROOT, battle_id="leaf-cancel", evaluate_root=lambda _: Evaluation((.5,.5),0.),
            sample_world=lambda rng: world, seed=1, trajectories=1, deadline_at=1., clock=lambda: now[0])
        self.assertEqual(result.trajectories, 0)
        self.assertEqual(search.nodes[ROOT.key].visits, [0, 0])
        self.assertTrue(result.deadline_exhausted)
        self.assertTrue(world.closed)

    def test_kernel_refuses_fabricated_leaf_deadline(self):
        search = TrajectorySearch(ReferenceConfig(.5, 1.))
        search.reset_battle("leaf-invalid")
        world = ToyWorld(depth=2, terminal=False)
        world.evaluate = lambda _: (_ for _ in ()).throw(LeafDeadlineExceeded())
        with self.assertRaisesRegex(ReferenceRefusal, "before the search deadline"):
            search.search_batch(ROOT, battle_id="leaf-invalid", evaluate_root=lambda _: Evaluation((.5,.5),0.),
                sample_world=lambda rng: world, seed=1, trajectories=1, deadline_at=1., clock=lambda: 0.)

    def test_worker_wrapper_keeps_root_and_hidden_rng_schedule(self):
        env = LeafEnv()
        world = ShowdownTrajectoryWorld(env, subject="p1", evaluator=evaluator, release=lambda: None)
        root = world.frame().subject
        original = PreparedDecision(root, lambda _: Evaluation((.4,.6),.3), lambda rng: world,
            lambda: {"draws": []})
        base = SimpleNamespace(prepare=lambda request: original, close=lambda: None)
        runtime = _ReferenceLeafRuntime(base, "raw_rollout", ROLLOUT_CAP)
        prepared = runtime.prepare(object())
        rng = random.Random(123)
        before = rng.getstate()
        wrapped = prepared.sample_world(rng)
        self.assertEqual(rng.getstate(), before)
        self.assertIs(prepared.evaluate_root, original.evaluate_root)
        wrapped.evaluate(root)
        self.assertEqual(len(prepared.evidence()["leaf_ablation"]["evaluations"]), 1)
        self.assertEqual(prepared.evidence()["leaf_ablation"]["evaluations"], [])
        wrapped.close()

    def test_undeclared_worker_leaf_or_cap_is_rejected(self):
        from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
        base = ShowdownWorkerFactory("weights", "a"*64, "showdown", "source")
        for leaf, cap in (("model", ROLLOUT_CAP), ("uniform", ROLLOUT_CAP), ("raw_rollout", 3)):
            with self.assertRaisesRegex(ValueError, "unregistered"):
                ReferenceLeafWorkerFactory(base, leaf, cap)

    def measured(self, leaf="raw_rollout"):
        from tests.test_search_over_raw_adapters import work
        measured = work()
        _, world, state, rows = self.world(leaf=leaf)
        world.evaluate(state)
        measured.worker_receipts[0]["evidence"]["leaf_ablation"] = dict(
            schema="pokezero.search-over-raw.leaf-batch.v1", leaf=leaf, rollout_cap=ROLLOUT_CAP,
            root_priors="unchanged_champion", root_value_used_for_backup=False, evaluations=rows)
        return measured

    def test_leaf_receipt_validates_real_terminal_or_hp_value(self):
        for leaf in ("hp_fraction", "raw_rollout"):
            validate_leaf_work(self.measured(leaf), leaf)

    def test_leaf_receipt_rejects_model_default_uniform_or_hp_fallback(self):
        measured = self.measured()
        row = measured.worker_receipts[0]["evidence"]["leaf_ablation"]["evaluations"][0]
        for key, value in (("alternative_signed_value", .1), ("continuation_policy", "uniform"),
                ("maximum_boundaries", 3), ("champion_priors_unchanged", False),
                ("model_signed_value", float("nan")), ("elapsed_seconds", -1.), ("boundaries", 251)):
            changed = deepcopy(measured)
            changed.worker_receipts[0]["evidence"]["leaf_ablation"]["evaluations"][0][key] = value
            with self.assertRaises(ValueError):
                validate_leaf_work(changed, "raw_rollout")
        row["status"] = "REFUSED"
        with self.assertRaisesRegex(ValueError, "unjustified"):
            validate_leaf_work(measured, "raw_rollout")

    def test_leaf_cancellation_requires_finite_expired_clock_and_zero_backup(self):
        from dataclasses import replace
        measured = self.measured()
        receipt = measured.worker_receipts[0]
        receipt["batch"] = replace(receipt["batch"], deadline_exhausted=True)
        row = receipt["evidence"]["leaf_ablation"]["evaluations"][0]
        row.update(status="DEADLINE_CANCELLED", alternative_signed_value=None, backed_up=False,
            checked_at=2., deadline_at=1.)
        validate_leaf_work(measured, "raw_rollout")
        for key, value in (("checked_at", .5), ("checked_at", float("inf")),
                ("deadline_at", float("nan")), ("backed_up", True), ("alternative_signed_value", 0.)):
            changed = deepcopy(measured)
            changed.worker_receipts[0]["evidence"]["leaf_ablation"]["evaluations"][0][key] = value
            with self.assertRaisesRegex(ValueError, "unjustified"):
                validate_leaf_work(changed, "raw_rollout")


if __name__ == "__main__":
    unittest.main()
