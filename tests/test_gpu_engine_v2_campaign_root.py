import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pokezero.v2_fused_state import REPORTS, SCHEMA, read_state
from scripts.prepare_v2_campaign_root import prepare


def write(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_doc(root, relative, value):
    return write(root, relative, json.dumps(value).encode())


def fixture(root):
    declaration = write_doc(root, 'declaration.json', {'train_both_seats': True})
    write_doc(root, 'route-stop-authority.json', {'enabled': True})
    write_doc(root, 'evaluation-stop.json', {'payload': {}, 'signature': 'bound-by-runtime'})
    reports = []
    milestones = {}
    for index in range(100):
        archive = f'artifacts/a{index}.npz'
        checkpoint = f'artifacts/c{index}.pt'
        actor = write(root, archive, str(index).encode())
        checkpoint_sha = hashlib.sha256(checkpoint.encode()).hexdigest()
        snapshot = dict(iteration=index, archive=archive, archive_sha256=actor,
                        checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha)
        report = dict(report=dict(iteration=index, snapshot=snapshot), encoding_receipt=None)
        reports.append(json.dumps(report).encode() + b'\n')
        if index in (49, 99):
            games = (index + 1) * 20_000
            write(root, checkpoint, checkpoint.encode())
            write_doc(root, checkpoint + '.encoding.json', {'checkpoint_sha256': checkpoint_sha})
            dispatch = dict(request=dict(declared_games=games, declaration_sha256=declaration,
                                         snapshot=snapshot),
                            full_checkpoint=dict(path=checkpoint, sha256=checkpoint_sha))
            milestones[str(games)] = write_doc(root, f'artifacts/cpu-panels/{games}/request.json', dispatch)
    prefix = b''.join(reports)
    write(root, REPORTS, prefix + b'UNCOMMITTED')
    state = write_doc(root, 'state.json', dict(declaration_sha256=declaration, round_ledger=dict(
        schema=SCHEMA, initializer={}, latest_snapshot=snapshot, completed_rounds=100,
        committed_bytes=len(prefix))))
    binding = dict(state_sha256=state, declaration_sha256=declaration, checkpoint_sha256=checkpoint_sha,
                   milestone_sha256s=milestones)
    return binding, prefix


class CampaignRootTests(unittest.TestCase):
    def test_copies_committed_prefix_and_milestone_artifacts_to_a_fresh_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'parent'
            dest = root.with_name('successor')
            binding, prefix = fixture(root)
            result = prepare(root, dest, binding)
            self.assertEqual(result['committed_rounds'], 100)
            self.assertEqual(read_state(dest), read_state(root))
            self.assertEqual((dest / REPORTS).read_bytes(), prefix)
            self.assertTrue((root / REPORTS).read_bytes().endswith(b'UNCOMMITTED'))
            for relative in ('artifacts/c49.pt.encoding.json', 'artifacts/c99.pt',
                             'artifacts/a0.npz', 'artifacts/cpu-panels/1000000/request.json',
                             'evaluation-stop.json'):
                self.assertEqual((dest / relative).read_bytes(), (root / relative).read_bytes())
            with self.assertRaises(FileExistsError):
                prepare(root, dest, binding)

    def test_wrong_actor_digest_cannot_publish_a_partial_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'parent'
            dest = root.with_name('successor')
            binding, _ = fixture(root)
            write(root, 'artifacts/a17.npz', b'changed')
            with self.assertRaisesRegex(ValueError, 'parent artifact differs for a17'):
                prepare(root, dest, binding)
            self.assertFalse(dest.exists())
