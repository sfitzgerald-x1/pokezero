import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

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


if __name__ == '__main__':
    unittest.main()
