"""Plain paired evaluation must bind both actors to the selected encoding."""

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gpu_engine import v2_head_to_head as h2h
from pokezero import v5_run
from test_v5_run import plan


class HeadToHeadEncodingTests(unittest.TestCase):
    def evaluate(self, enabled, metadata, *, catalog=None, bad_model=False, error=None):
        original = plan()
        # The paired runner assigns swapped games to the lower stream namespace.
        iterations = tuple(tuple(replace(batch, games=tuple(
            replace(game, stream_id=game.stream_id + h2h.SWAP_OFFSET) for game in batch.games))
            for batch in iteration) for iteration in original.iterations)
        recipe = replace(original, iterations=iterations, self_moves=enabled)
        configs = (recipe.model, replace(recipe.model, replicate=1))
        actors = [v5_run.Snapshot(0, 'actor.pt', 'a' * 64, 'actor.npz', digest * 64,
                  v5_run._sha(config.metadata())) for digest, config in zip('bc', configs)]
        if bad_model:
            actors[1] = replace(actors[1], model_sha256='d' * 64)
        tables = object()
        prepared = SimpleNamespace(engine_tables=object(), observation_tables=tables, source_sha256='e' * 64,
            catalog=SimpleNamespace(masks=SimpleNamespace(self_moves=enabled if catalog is None else catalog)))
        context = dict(plan=recipe, prepared=prepared, pool=object(), gender=None)
        weights = (object(), object())
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch('gpu_engine.v2_fused.prepare', return_value=context))
            snapshots = stack.enter_context(patch.object(h2h, 'snapshot', side_effect=actors))
            stack.enter_context(patch('gpu_engine.v5_model.load', side_effect=[
                (weight, config, meta) for weight, config, meta in zip(weights, configs, metadata)]))
            backend = stack.enter_context(patch('gpu_engine.v5_device_backend.RuntimeBackend'))
            stack.enter_context(patch('gpu_engine.v5_device_rollout.Bindings'))
            stack.enter_context(patch('gpu_engine.v5_device_rollout.actor_bank_from_assignments'))
            stack.enter_context(patch('gpu_engine.v2_collector.roster_from_blocks'))
            stack.enter_context(patch('jax.block_until_ready', side_effect=lambda value: value))
            collector = stack.enter_context(patch('gpu_engine.v2_collector.RefillCollector'))
            collector.return_value.run.return_value = SimpleNamespace(stats=dict(occupancy=1., kernel_seconds=1.))
            streams = [g.stream_id for g in recipe.iterations[0][0].games]
            stack.enter_context(patch('gpu_engine.v2_collector.game_records', return_value=dict(
                streams=streams + [s - h2h.SWAP_OFFSET for s in streams], status=[1] * 4, winner=[0, 1, 1, 0])))
            stack.enter_context(patch.object(h2h, 'log'))
            args = ['--config', 'unit.json', '--out', directory, '--iteration', '0',
                    '--a-run', 'a', '--a-iteration', '0', '--b-run', 'b', '--b-iteration', '0']
            if error:
                with self.assertRaisesRegex(ValueError, error):
                    h2h.main(args)
                backend.assert_not_called()
                collector.assert_not_called()
                if catalog is not None:
                    snapshots.assert_not_called()
                return
            self.assertEqual(h2h.main(args), 0)
            backend.return_value.check_plan.assert_called_once_with(recipe)
            self.assertIs(collector.call_args.args[1], tables)
            collector.return_value.run.assert_called_once_with(weights)
            report = json.loads((Path(directory) / 'head-to-head.json').read_text())
            self.assertEqual(report['a_paired_score'], 0.5)
            if enabled:
                self.assertIs(report['self_moves'], True)
            else:
                self.assertNotIn('self_moves', report)

    def test_matched_on_and_legacy_off_reach_the_paired_collector(self):
        self.evaluate(True, ({'self_moves': True}, {'self_moves': True}))
        self.evaluate(False, ({}, {'self_moves': False}))

    def test_either_actor_may_not_use_a_different_encoding(self):
        for a, b in (({}, {'self_moves': True}), ({'self_moves': True}, {})):
            self.evaluate(True, (a, b), error='actor archive self_moves differs')
        self.evaluate(False, ({}, {'self_moves': True}), error='actor archive self_moves differs')

    def test_catalog_must_match_before_loading_either_actor(self):
        self.evaluate(True, ({}, {}), catalog=False, error='prepared catalog self_moves differs')

    def test_snapshot_model_digest_matches_the_actor_not_the_evaluation_replicate(self):
        self.evaluate(False, ({}, {}), bad_model=True, error='actor snapshot model digest differs')


if __name__ == '__main__':
    unittest.main()
