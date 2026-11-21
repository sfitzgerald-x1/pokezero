import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from gpu_engine import v2_distributed, v2_fused
from pokezero import v2_fused_state as state


def report(iteration, digest):
    return dict(iteration=iteration, snapshot=dict(iteration=iteration, checkpoint_sha256=digest),
                ranks=[dict(rows=100)])


class FusedStateTests(unittest.TestCase):
    def test_resume_ignores_and_replaces_uncommitted_tail(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            first = report(0, 'first')
            end = state.append_round(out, 0, first, 'proof1')
            pointer = dict(schema=state.SCHEMA, committed_bytes=end, completed_rounds=1,
                           initializer={'iteration': -1}, latest_snapshot=first['snapshot'])
            manifest = out / 'state.json'
            manifest.write_text(json.dumps({'round_ledger': pointer}))
            state.append_round(out, end, report(1, 'orphan'), 'orphan-proof')
            resumed = state.read_state(out, include_full_reports=False)
            self.assertEqual(len(resumed['reports']), 1)
            self.assertNotIn('ranks', resumed['reports'][0])
            self.assertEqual(resumed['checkpoint_encoding_receipts'], {'first': 'proof1'})

            second = report(1, 'second')
            new_end = state.append_round(out, end, second, 'proof2')
            pointer.update(committed_bytes=new_end, completed_rounds=2, latest_snapshot=second['snapshot'])
            manifest.write_text(json.dumps({'round_ledger': pointer}))
            resumed = state.read_state(out)
            self.assertEqual(resumed['reports'], [first, second])
            self.assertEqual(resumed['checkpoint_encoding_receipts'], {'first': 'proof1', 'second': 'proof2'})
            self.assertEqual((out / state.REPORTS).stat().st_size, new_end)

            with (out / state.REPORTS).open('r+b') as stream:
                stream.truncate(new_end - 1)
            with self.assertRaisesRegex(ValueError, 'truncated committed'):
                state.read_state(out)

    def test_stale_writer_cannot_truncate_a_committed_round(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            first = report(0, 'first')
            with state.commit_guard(out, 0, 0):
                end = state.append_round(out, 0, first, 'proof')
                (out / 'state.json').write_text(json.dumps(dict(round_ledger={
                    'committed_bytes': end, 'completed_rounds': 1,
                })))
            before = (out / state.REPORTS).read_bytes()
            with self.assertRaisesRegex(ValueError, 'another writer'):
                with state.commit_guard(out, 0, 0):
                    state.append_round(out, 0, report(0, 'stale'), 'stale-proof')
            self.assertEqual((out / state.REPORTS).read_bytes(), before)

    def test_launcher_and_worker_refuse_a_nested_mount_alias_before_writes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            parent, alias = root / 'parent', root / 'alias'
            parent.mkdir()
            alias.mkdir()
            original = parent / 'state.json'
            original.write_text('untouched')
            out = alias / 'nested'
            options = ['--config', 'unused', '--reference-run', 'unused', '--out', str(out),
                       '--lr-start', 'sqrt', '--lr-warmup-updates', '2', '--campaign-anneal',
                       '--cooldown-child', str(parent), '--cooldown-parent-sha256', 'b' * 64]
            identity = state._directory_identity
            with patch.object(state, '_directory_identity', side_effect=lambda path: identity(parent if path == alias else path)):
                for launch, launcher_options in (
                    (v2_fused.main, []), (v2_distributed.main, ['--ranks', '4', '--timeout-seconds', '60']),
                ):
                    with self.assertRaisesRegex(ValueError, 'distinct, non-nested'):
                        launch([*launcher_options, *options])
            self.assertFalse(out.exists())
            self.assertEqual(list(parent.iterdir()), [original])
            self.assertEqual(original.read_text(), 'untouched')

    def test_receipt_requires_parent_arguments_even_before_first_commit(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            receipt = out / state.PARENT_RECEIPT
            receipt.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'original parent arguments'):
                v2_fused.main(['--config', 'unused', '--reference-run', 'unused', '--out', str(out)])
            self.assertEqual(list(out.iterdir()), [receipt])


if __name__ == '__main__':
    unittest.main()
