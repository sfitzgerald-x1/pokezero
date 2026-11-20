"""Selection/RNG-coordinate and full-bit guards; no model or game execution."""
from collections import namedtuple
from dataclasses import dataclass
import copy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from gpu_engine import fresh200_capture as capture


@dataclass(frozen=True)
class Batch:
    games: tuple
    constructor_seed: int
    sampling_seed: int
    gender_seed: int | None = None
    gender_domain: int | None = None
    gender_root_ids: tuple = ()


class CaptureGuardsTests(unittest.TestCase):
    def addresses(self):
        return [dict(frame=frame, original_ordinal=row, row=row, stream_id=1000+row,
            live=True, requested=[True, True], observation_errors=[0, 0])
            for frame in (7, 8, 9) for row in range(256)]

    def test_exact_both_owner_roster_distinct_roots_and_order(self):
        rows = self.addresses()
        selected = capture.select_roots(rows)
        self.assertEqual(len(selected), 200)
        self.assertEqual([sum(row['owner'] == owner for row in selected) for owner in (0, 1)], [107, 93])
        self.assertEqual(len({(row['frame'], row['original_ordinal']) for row in selected}), 200)
        self.assertEqual([row['original_ordinal'] for row in selected], list(range(200)))
        self.assertTrue(all(row['frame'] == 8 for row in selected))
        self.assertEqual(selected, capture.select_roots(reversed(rows)))

    def test_error_and_comparison_outcomes_do_not_remove_selected_roots(self):
        rows = self.addresses();before = capture.select_roots(rows)
        for row in rows:
            row['observation_errors'] = [123, 456]
            row['reference_passed'] = False
            row['next_status'] = 'REFUSED'
        after = capture.select_roots(rows)
        address = lambda row: (row['frame'], row['original_ordinal'], row['owner'])
        self.assertEqual(list(map(address, before)), list(map(address, after)))
        self.assertEqual(after[0]['observation_errors'], [123, 456])

    def test_missing_owner_and_insufficient_universe_fail_without_relaxation(self):
        rows = self.addresses()
        for row in rows:row['requested'][1] = False
        with self.assertRaisesRegex(ValueError, 'fixed owner schedule'):capture.select_roots(rows)
        with self.assertRaisesRegex(ValueError, 'fixed owner schedule'):capture.select_roots(self.addresses()[:256+199])
        rows = self.addresses()
        for row in rows:row['live'] = False
        with self.assertRaisesRegex(ValueError, 'fixed owner schedule'):capture.select_roots(rows)

    def test_duplicate_physical_root_and_bad_owner_flags_fail(self):
        rows = self.addresses()
        with self.assertRaisesRegex(ValueError, 'duplicate'):capture.select_roots(rows+[copy.deepcopy(rows[0])])
        rows[0]['requested'] = [1, True]
        with self.assertRaisesRegex(ValueError, 'malformed'):capture.select_roots(rows)

    def test_leading_batch_preserves_original_seed_gender_and_game_coordinates(self):
        games = tuple(SimpleNamespace(stream_id=1000+i) for i in range(512))
        original = Batch(games, 23, 42, 53, 71, tuple(range(10000, 10512)))
        plan = SimpleNamespace(iterations=((original,),), max_requests=4096)
        sliced = capture.leading_batch(plan)
        self.assertEqual(sliced.games, original.games[:256])
        self.assertIs(sliced.games[0], original.games[0])
        self.assertEqual((sliced.constructor_seed, sliced.sampling_seed, sliced.gender_seed, sliced.gender_domain), (23, 42, 53, 71))
        self.assertEqual(sliced.gender_root_ids, original.gender_root_ids[:256])
        self.assertEqual(len(original.games), 512)
        plan.iterations = ((Batch(games[:255], 23, 42),),)
        with self.assertRaisesRegex(ValueError, 'fewer than 256'):capture.leading_batch(plan)

    def test_assignments_require_recorded_full_bank_and_keep_original_ordinals(self):
        games = tuple(SimpleNamespace(stream_id=1000+i) for i in range(512))
        original = Batch(games, 23, 42)
        plan = SimpleNamespace(iterations=((original,),), games=(SimpleNamespace(stream_id=999),)+games,
            max_requests=4096)
        def assignment(plan, game, ordinal, snapshot, past):
            return dict(stream_id=game.stream_id, ordinal=ordinal, archive=snapshot)
        runs = SimpleNamespace(assignment=assignment, _json=json.dumps)
        expected = [assignment(plan, game, i+1, 'archive', ()) for i, game in enumerate(games)]
        events = [dict(kind='BATCH_STARTED', payload=dict(streams=[game.stream_id for game in games], assignments=expected))]
        rows, ordinals = capture.original_assignments(plan, capture.leading_batch(plan), 'archive', events, runs)
        self.assertEqual(rows, expected[:256]);self.assertEqual(ordinals, list(range(1, 257)))
        events[0]['payload']['assignments'][4]['archive'] = 'substituted'
        with self.assertRaisesRegex(ValueError, 'assignments differ'):
            capture.original_assignments(plan, capture.leading_batch(plan), 'archive', events, runs)
        with self.assertRaisesRegex(ValueError, 'one original'):
            capture.original_assignments(plan, capture.leading_batch(plan), 'archive', [], runs)

    def test_complete_tree_guard_catches_history_none_dtype_shape_and_one_bit(self):
        State = namedtuple('State', 'hp history optional')
        old = State(np.array([714], np.int32), {'taken': np.array([0, 0x80000000], np.uint32).view(np.float32)}, None)
        capture.assert_tree_bits(old, copy.deepcopy(old))
        changed = copy.deepcopy(old);changed.history['taken'].view(np.uint32)[0] = 1
        for value in (changed, old._replace(hp=old.hp.astype(np.int64)),
                old._replace(hp=old.hp.reshape(1, 1)), old._replace(optional=np.zeros(1)),
                old._replace(history={}), old._replace(history={'taken': np.zeros(2, np.float32)})):
            with self.subTest(value=repr(value)), self.assertRaisesRegex(ValueError, 'mismatch'):
                capture.assert_tree_bits(old, value)

    def test_replay_inputs_preserve_complete_trees_and_original_array_objects(self):
        State = namedtuple('State', 'hp keys optional')
        Trackers = namedtuple('Trackers', 'history optional')
        state = State(np.array([[[714]]], np.int32), np.array([[[0, 2**32-1]]], np.uint32), None)
        public = Trackers({'taken': np.array([[[0x80000000]]], np.uint32).view(np.float32)}, None)
        pre = SimpleNamespace(state=state, public=public, streams=np.array([[51]], np.int32),
            frames=np.array([[8]], np.int32), status=np.array([[0]], np.int32))
        post = SimpleNamespace(state=copy.deepcopy(state), public=copy.deepcopy(public))
        frames = SimpleNamespace(engine_actions=np.array([[[0, 1]]], np.int32), requested=np.array([[[True, False]]]))
        record = dict(pre=pre, post=post, step_state=state, frames=frames,
            chance=SimpleNamespace(effect=np.zeros((1, 1, 2, 5), np.int32)), actor_keys=np.zeros((1, 1, 2, 2), np.uint32))
        result = capture.replay_inputs(record)
        for key, expected in dict(pre_state=state, pre_public=public, step_state=state,
                post_state=post.state, post_public=post.public, actions=frames.engine_actions,
                requested=frames.requested, streams=pre.streams, frames=pre.frames,
                status=pre.status, chance=record['chance'], actor_keys=record['actor_keys']).items():
            self.assertIs(result[key], expected)
        self.assertIsNone(result['pre_state'].optional)
        self.assertEqual(result['pre_public'].history['taken'].view(np.uint32).item(), 0x80000000)


if __name__ == '__main__':unittest.main()
