"""Fail-closed cap-only transfer and original-prefix recovery guards."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from pokezero.mcts_eval.wider_substitute_tail_recovery import (
    RemoveTailGuard, charge_failed_attempt, validate_failed_attempt,
    validate_resumed_workers, bind_transfer_proofs, validate_transfer,
)


class TailRecoveryTests(unittest.TestCase):
    def test_first_draw_of_every_completed_worker_is_bound_not_only_worker_zero(self):
        draw = dict(status='ROOT_VALIDATED', released=True, packed_team_sha256='team',
            materialization_seed=17, substitute_policy_conditioning=dict(max_attempts=2048, attempts=139))
        probe = dict(worker_ordinals=list(range(20)), workers=[dict(worker=i, ordinal=i,
            draws=[copy.deepcopy(draw)]) for i in range(20)])
        receipts = [dict(worker=i, evidence=dict(draws=[copy.deepcopy(draw)])) for i in range(20)]
        validate_resumed_workers(receipts, probe)
        validate_resumed_workers(receipts[:1], probe)  # No unfinished batch may be imported.
        for worker in (0, 3, 19):
            bad = copy.deepcopy(receipts)
            bad[worker]['evidence']['draws'][0]['materialization_seed'] += 1
            with self.subTest(worker=worker), self.assertRaises(RuntimeError):
                validate_resumed_workers(bad, probe)

    def test_conditional_receipt_and_release_cannot_change(self):
        expected = dict(status='ROOT_VALIDATED', released=True, packed_team_sha256='team',
            materialization_seed=17, substitute_policy_conditioning=dict(max_attempts=2048, attempts=139))
        probe = dict(worker_ordinals=[10] * 20, workers=[dict(worker=3, ordinal=10, draws=[expected])])
        for field, value in (('released', False), ('status', 'PARTIAL'), ('packed_team_sha256', 'different'),
                             ('substitute_policy_conditioning', dict(max_attempts=128, attempts=20))):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_resumed_workers([dict(worker=3, evidence=dict(draws=[dict(expected, **{field: value})]))], probe)

    def failure_fixture(self):
        prefix = {'boundary-000.json.gz': 'original'}
        old = dict(repair_retention=dict(resume=dict(identity='saved', prefix_steps=prefix)))
        row = dict(status='REFUSED', signed_outcome=None, identity='saved', arm='paper_reference',
            retained_prefix_steps=prefix, failure_evidence=dict(decision_id=391,
                battle_id='wider-search:saved', receipts=[], errors=[['error', 3, 391, 'trajectory_batch',
                    'ReferenceRefusal: Substitute public-history conditioning exhausted its explicit rejection cap',
                    'trace', dict(partial_batch_evidence=dict(draws=[dict(status='REFUSED', ordinal=0)]))]]))
        return row, old

    def test_exact_refusal_requires_no_accepted_new_work(self):
        row, old = self.failure_fixture()
        validate_failed_attempt(row, old, [])
        with self.assertRaises(RuntimeError):
            validate_failed_attempt(row, old, ['new-accepted-boundary'])
        for mutate in (lambda r: r.update(signed_outcome=-1),
                       lambda r: r['failure_evidence'].update(decision_id=392),
                       lambda r: r['failure_evidence'].update(receipts=[{}]),
                       lambda r: r['failure_evidence']['errors'][0].__setitem__(1, 4),
                       lambda r: r['failure_evidence']['errors'][0][-1]['partial_batch_evidence']['draws'][0].update(status='ROOT_VALIDATED')):
            bad = copy.deepcopy(row)
            mutate(bad)
            with self.assertRaises(RuntimeError):
                validate_failed_attempt(bad, old, [])

    def test_original_wall_charge_plus_failed_attempt_is_retained(self):
        resume = dict(elapsed_before_resume=145., decision_id=390,
            wall_clock_envelope=dict(directory_birth=1., refused_result_birth=145., margin_seconds=1.))
        with mock.patch.object(Path, 'stat', side_effect=[SimpleNamespace(st_birthtime=100.),
                                                         SimpleNamespace(st_birthtime=112.)]):
            result = charge_failed_attempt(resume, 'directory', 'terminal', 2400.)
        self.assertEqual(result['elapsed_before_resume'], 158.)
        self.assertEqual(resume['elapsed_before_resume'], 145.)
        self.assertEqual(result['decision_id'], 390)
        self.assertEqual(result['wall_clock_envelope']['directory_birth'], 1.)
        self.assertEqual(result['wall_clock_envelope']['additional_attempts'][0]['charged_seconds'], 13.)

    def test_missing_reversed_or_exhausted_wall_envelope_refuses(self):
        resume = dict(elapsed_before_resume=145., wall_clock_envelope={})
        for born, stopped, cap in ((None, 112., 2400.), (113., 112., 2400.), (100., 112., 150.)):
            with self.subTest(born=born, cap=cap), mock.patch.object(Path, 'stat', side_effect=[
                    SimpleNamespace(st_birthtime=born), SimpleNamespace(st_birthtime=stopped)]), self.assertRaises(RuntimeError):
                charge_failed_attempt(resume, 'directory', 'terminal', cap)

    def test_ast_guard_normalization_preserves_only_declared_proof_check(self):
        guard = """def run():
    if recovery.get('kind') == 'native-and-reference-substitute-tail-repair':
        from pokezero.mcts_eval.wider_substitute_tail_recovery import validate_resumed_workers
        validate_resumed_workers(measured.worker_receipts, probe)
    elif worker_zero:
        raise RuntimeError('old check')
"""
        old = ast.parse("def run():\n    if worker_zero:\n        raise RuntimeError('old check')\n")
        self.assertEqual(ast.dump(RemoveTailGuard().visit(ast.parse(guard))), ast.dump(old))
        for changed in (guard.replace('measured.worker_receipts', 'diagnostic_checkpoint'),
                        guard.replace('validate_resumed_workers(measured.worker_receipts, probe)', 'restore_failed_work()')):
            with self.assertRaises(RuntimeError):
                RemoveTailGuard().visit(ast.parse(changed))

    def test_transfer_proof_digest_is_not_optional(self):
        with self.assertRaises(RuntimeError):
            bind_transfer_proofs('unbound', sha=lambda p: 'unknown')

    def test_qualification_transfer_cannot_change_qualification_identity(self):
        with self.assertRaises(RuntimeError):
            validate_transfer(Path('wrong/READOUT.json'), {}, dict(qualification_readout='original/READOUT.json'),
                repo=Path('.'), git=None, sha=None, verify=None, bound_rows=None)


if __name__ == '__main__':
    unittest.main()
