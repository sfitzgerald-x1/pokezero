import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from gpu_engine import source_provenance
from gpu_engine.team_pool import FIELDS
from scripts.assemble_v2_campaign_pool import assemble, file_sha


def make_shard(path, start, stop, row_start, *, value=1):
    rows = stop - start
    arrays = {name: np.full((rows, *shape), value, dtype=dtype) for name, (shape, dtype) in FIELDS.items()}
    metadata = {
        'seed_start': start, 'completed_teams': rows, 'team_count': rows,
        'vocabulary': {'moves': ['tackle']}, 'move_frequency': [{'id': 'tackle', 'count': rows * 24}],
    }
    np.savez_compressed(
        path, **arrays, seeds=np.arange(start, stop, dtype=np.uint32),
        metadata_utf8=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
    )
    return dict(path=str(path), sha256=file_sha(path), seeds=[start, stop], row_start=row_start)


class CampaignPoolTests(unittest.TestCase):
    def test_streamed_loader_keeps_original_bytes_and_read_only_receipt(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'pool.npz'
            arrays = {
                name: np.asfortranarray(np.arange(2 * int(np.prod(shape)), dtype=dtype).reshape(2, *shape))
                for name, (shape, dtype) in FIELDS.items()
            }
            arrays['gender'] = np.arange(12, dtype=np.uint8).reshape(2, 6)
            arrays['happiness'] = np.full((2, 6), 255, dtype=np.uint8)
            metadata = {'seed_start': 200, 'completed_teams': 2}
            np.savez_compressed(
                path, **arrays,
                metadata_utf8=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
            )
            loaded = source_provenance.load_constructor_pool(path)
            receipt = source_provenance._pool(loaded, rehash=True)
            self.assertEqual(receipt['pool_sha256'], file_sha(path))
            self.assertEqual(loaded.metadata(), metadata)
            for name, actual in loaded.arrays.items():
                expected = arrays[name].tobytes()
                self.assertEqual(actual.tobytes(), expected)
                self.assertEqual(receipt['arrays'][name]['sha256'], hashlib.sha256(expected).hexdigest())
                with self.assertRaises(ValueError):
                    actual.setflags(write=True)

    def test_preserves_prefix_with_fresh_team_seed_band(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parent = make_shard(root / 'p.npz', 100, 102, 0)
            extra = make_shard(root / 'e.npz', 200, 203, 2)
            manifest = dict(parent_pool=parent, shards=[parent, extra],
                            preserved_team_rows=2, required_team_rows=5)
            target = root / 'out.npz'
            result = assemble(manifest, target)
            self.assertEqual(result['team_rows'], 5)
            self.assertEqual(result['sha256'], file_sha(target))
            with np.load(target, allow_pickle=False) as pool:
                np.testing.assert_array_equal(pool['seeds'], [100, 101, 200, 201, 202])
                metadata = json.loads(pool['metadata_utf8'].tobytes())
                self.assertEqual(metadata['move_slots'], 120)

    def test_mismatched_committed_bytes_block_publication(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parent = make_shard(root / 'p.npz', 100, 102, 0)
            changed = make_shard(root / 'c.npz', 100, 102, 0, value=2)
            manifest = dict(parent_pool=parent, shards=[changed], preserved_team_rows=2, required_team_rows=2)
            target = root / 'out.npz'
            with self.assertRaisesRegex(ValueError, 'committed constructor bytes differ'):
                assemble(manifest, target)
            self.assertFalse(target.exists())
