"""Complete create-only failure evidence without dropped dataclass fields."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

from wider_search_comparison import save
from pokezero.mcts_eval.paper_reference import BatchResult


@dataclass(frozen=True)
class ReceiptFixture:
    world_draws: int
    transitions: int
    counters: tuple


class EvidenceSerializationTests(unittest.TestCase):
    def test_real_failed_parallel_batch_result_is_fully_preserved(self):
        batch = BatchResult(trajectories=3, transitions=12, world_draws=3,
            elapsed_seconds=14., deadline_exhausted=True, deadline_overrun_seconds=4.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'terminal.json'
            save(path, dict(status='REFUSED', signed_outcome=None, failure_evidence=dict(
                receipts=[dict(worker=0, sequence=5, batch=batch)],
                errors=[['error', 3, 393, 'trajectory_batch', 'original refusal']])))
            result = json.loads(path.read_text())
        self.assertEqual(result['failure_evidence']['receipts'][0]['batch'], dict(
            trajectories=3, transitions=12, world_draws=3, elapsed_seconds=14.,
            deadline_exhausted=True, deadline_overrun_seconds=4.))
        self.assertEqual(result['failure_evidence']['errors'][0][-1], 'original refusal')
        self.assertEqual(result['status'], 'REFUSED')
        self.assertIsNone(result['signed_outcome'])

    def test_full_dataclass_failure_receipts_are_serialized_without_scoring(self):
        value = dict(status='REFUSED', signed_outcome=None, failure_evidence=dict(
            receipts=[dict(worker=3, batch=ReceiptFixture(2, 8, (1, 4)))]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'terminal.json'
            save(path, value)
            result = json.loads(path.read_text())
        self.assertEqual(result, dict(status='REFUSED', signed_outcome=None,
            failure_evidence=dict(receipts=[dict(worker=3,
                batch=dict(world_draws=2, transitions=8, counters=[1, 4]))])))
        self.assertEqual(value['failure_evidence']['receipts'][0]['batch'].world_draws, 2)

    def test_unsupported_evidence_leaves_no_partial_canonical_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'terminal.json'
            with self.assertRaises(TypeError):
                save(path, dict(status='REFUSED', failure_evidence=object()))
            self.assertFalse(path.exists())

    def test_existing_terminal_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'terminal.json'
            save(path, dict(status='COMPLETE'))
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save(path, dict(status='REFUSED'))
            self.assertEqual(path.read_bytes(), before)

    def test_ordinary_json_bytes_remain_unchanged(self):
        value = dict(status='COMPLETE', signed_outcome=1, rows=[dict(q=.5, n=2)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'terminal.json'
            save(path, value)
            self.assertEqual(path.read_text(), json.dumps(value, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    unittest.main()
