"""Exercise sealed roster addressing and partial final input chunks."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import fresh200_fixed_prefix as prefix


class FrozenPrefixInputsTest(unittest.TestCase):
    def test_bind_original_ordered_roster_and_owner_counts(self):
        selected = [dict(root_index=index, frame=8, original_ordinal=index, owner=int(index >= 107))
                    for index in range(200)]
        replay = dict(status='DEVICE_REPLAYED_ONLY', passed=False, roots=200, owner_counts=[107, 93],
                      completed_steps=64, observation_errors=0, selected=deepcopy(selected))
        self.assertEqual(len(prefix._ordered_roster({'selected': selected}, replay)), 64)
        replay['selected'][0], replay['selected'][1] = replay['selected'][1], replay['selected'][0]
        with self.assertRaisesRegex(ValueError, 'rosters differ'):
            prefix._ordered_roster({'selected': selected}, replay)
        replay['selected'] = deepcopy(selected)
        replay['selected'][0]['owner'] = False
        with self.assertRaisesRegex(ValueError, 'rosters differ'):
            prefix._ordered_roster({'selected': selected}, replay)

    def test_selected_endpoint_defines_exact_native_command_extent(self):
        schedule = prefix.prefix_schedule(9)
        self.assertEqual(len(schedule), 22)
        self.assertEqual([command.frame for command in schedule if command.phase == 'choices'], list(range(9)))
        self.assertEqual([command.frame for command in schedule if command.phase == 'snapshot'], list(range(10)))
        self.assertEqual([command.phase for command in schedule[-2:]], ['end', 'close'])
        for bad in (7, 64, True, 9.0):
            with self.subTest(endpoint=bad), self.assertRaisesRegex(ValueError, 'outside the captured horizon'):
                prefix.prefix_schedule(bad)

    def test_select_frozen_root_and_read_only_rows_before_its_endpoint(self):
        game = dict(original_ordinal=5, stream_id=123, battle_id='sealed', seed=234, team_indices=[5, 6])
        addresses = [dict(game, row=1, step=step, frame=step, live=True, requested=[True, True])
                     for step in range(10)]
        selected = dict(addresses[-1], root_index=1, owner=0)
        chunks = [dict(start_step=start, steps=4, link=f'link-{start}', arrays=f'arrays-{start}',
                       replay_inputs=f'inputs-{start}') for start in (0, 4, 8)]
        links = {chunk['link']: dict(parents=[previous, chunk['arrays'], chunk['replay_inputs']],
                    payload=dict(start_step=chunk['start_step'], steps=4))
                 for chunk, previous in zip(chunks, ('construction', 'link-0', 'link-4'))}
        capture = dict(selected=[None, selected, *([None] * 198)], addresses=addresses, chunks=chunks)
        chosen, chosen_game = prefix._selected_game(capture, {'games': [None, game]}, 2, 1)
        self.assertEqual((chosen, chosen_game), (selected, game))

        reads = []

        class ReplayInputs:
            def __init__(self, root, reference, storage, *, source, kind, start_step, steps):
                self.reference, self.start_step = reference, start_step

            def identity(self):
                return self.reference

            def replay_inputs(self, relative, row):
                frame = self.start_step + relative
                reads.append((frame, row))
                return dict(streams=123, frames=frame, status=0, requested=np.asarray([True, True]),
                            actions=np.asarray([1, 2]), chance={name: np.asarray(0) for name in prefix.CHANCE_FIELDS})

        with patch.object(prefix, 'SealedInputs', ReplayInputs), patch.object(
                prefix, '_manifest', side_effect=lambda root, reference, source: links[reference]):
            steps, sources = prefix._read_steps(None, None, None, game, 1, 'construction', links['link-0'],
                                                selected, bind_consumed=True)
        self.assertEqual(reads, [(step, 1) for step in range(9)])
        self.assertEqual([step.frame for step in steps], list(range(9)))
        self.assertEqual(sources, [
            dict(identity='inputs-0', consumed_start=0, consumed_steps=4),
            dict(identity='inputs-4', consumed_start=4, consumed_steps=4),
            dict(identity='inputs-8', consumed_start=8, consumed_steps=1),
        ])
