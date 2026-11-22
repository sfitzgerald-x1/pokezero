"""Search H2H binds both seats and the search pricer to the evaluation encoding."""

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import NamedTuple
import unittest
from unittest.mock import Mock, patch

import numpy as np

from gpu_engine import v2_search_head_to_head as search
from gpu_engine.v5_device_rollout import COMPLETE
from pokezero import v5_run
from test_v5_run import plan


class Finished(NamedTuple):
    streams: object
    status: object
    winner: object
    terminal_frame: object


class SearchHeadToHeadEncodingTests(unittest.TestCase):
    def evaluate(self, enabled, metadata, *, catalog=None, tables=None, error=None):
        original = plan()
        # Swapped games use the lower stream namespace, as in the paired runner.
        iterations = tuple(tuple(replace(batch, games=tuple(
            replace(game, stream_id=game.stream_id + search.SWAP_OFFSET) for game in batch.games))
            for batch in iteration) for iteration in original.iterations)
        recipe = replace(original, iterations=iterations, self_moves=enabled)
        configs = (recipe.model, replace(recipe.model, replicate=1), recipe.model)
        actors = [v5_run.Snapshot(0, 'actor.pt', 'a' * 64, 'actor.npz', digest * 64,
                  v5_run._sha(config.metadata())) for digest, config in zip('bcd', configs)]
        table_enabled = enabled if tables is None else tables
        observation = SimpleNamespace(belief=SimpleNamespace(self_moves=object() if table_enabled else None))
        prepared = SimpleNamespace(engine_tables=object(), observation_tables=observation,
            catalog=SimpleNamespace(masks=SimpleNamespace(self_moves=enabled if catalog is None else catalog)))
        context = dict(plan=recipe, prepared=prepared, pool=object(), gender=None)
        weights = tuple(object() for _ in metadata)

        def construct(batch):
            count = len(batch.games)
            return Finished(np.asarray([game.stream_id for game in batch.games]), np.full(count, COMPLETE),
                            np.zeros(count, np.int32), np.ones(count, np.int32))

        def decide(carry, params, a_side):
            fields = ('requested', 'legal', 'raw_logits', 'actions', 'input_valid', 'turn', 'values')
            return dict.fromkeys(fields) | {'live': np.zeros(len(carry.streams), bool)}

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            cells = Path(directory) / 'cells.json'
            cells.write_text(json.dumps([
                dict(label=f'cell-{i}', b_run=f'b-{i}', b_iteration=0, reduction='average')
                for i in range(1, len(metadata))]))
            stack.enter_context(patch('gpu_engine.v2_fused.prepare', return_value=context))
            snapshots = stack.enter_context(patch.object(search, 'snapshot', side_effect=actors))
            stack.enter_context(patch('gpu_engine.v5_model.load', side_effect=[
                (weight, config, meta) for weight, config, meta in zip(weights, configs, metadata)]))
            backend = stack.enter_context(patch('gpu_engine.v5_device_backend.RuntimeBackend'))
            backend.return_value._construct.side_effect = construct
            step = stack.enter_context(patch.object(search, 'make_step', return_value=(decide, Mock())))
            pricer = stack.enter_context(patch.object(search, 'Search'))
            stack.enter_context(patch.object(search, 'log'))
            args = ['--config', 'unit.json', '--out', directory, '--iteration', '0',
                    '--a-run', 'a', '--a-iteration', '0', '--cells', str(cells)]
            if error:
                with self.assertRaisesRegex(ValueError, error):
                    search.main(args)
                backend.assert_not_called()
                step.assert_not_called()
                pricer.assert_not_called()
                self.assertFalse((Path(directory) / 'cell-1.json').exists())
                if catalog is not None or tables is not None:
                    snapshots.assert_not_called()
                return
            self.assertEqual(search.main(args), 0)
            backend.return_value.check_plan.assert_called_once_with(recipe)
            self.assertIs(step.call_args.args[1], observation)
            self.assertEqual(pricer.call_args.args, (prepared, recipe, weights[0]))
            report = json.loads((Path(directory) / 'cell-1.json').read_text())
            self.assertEqual(report['a_paired_score'], 0.5)
            if enabled:
                self.assertIs(report['self_moves'], True)
            else:
                self.assertNotIn('self_moves', report)

    def test_matched_on_and_legacy_off_reach_root_and_pricer(self):
        self.evaluate(True, ({'self_moves': True}, {'self_moves': True}))
        self.evaluate(False, ({}, {'self_moves': False}))

    def test_mixed_actors_refuse_before_playing_any_cell(self):
        enabled = {'self_moves': True}
        self.evaluate(True, ({}, enabled), error='actor archive self_moves differs')
        self.evaluate(True, (enabled, enabled, {}), error='actor archive self_moves differs')
        self.evaluate(False, ({}, enabled), error='actor archive self_moves differs')

    def test_mismatched_catalog_or_tables_refuse_before_loading_actors(self):
        self.evaluate(True, ({}, {}), catalog=False, error='prepared catalog self_moves differs')
        self.evaluate(True, ({}, {}), tables=False, error='prepared observation tables self_moves differs')
        self.evaluate(False, ({}, {}), tables=True, error='prepared observation tables self_moves differs')
