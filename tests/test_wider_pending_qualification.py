"""Qualification recovery must retain all evidence and refuse drift."""
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from pokezero.mcts_eval.followthrough import continuation_seed
from pokezero.mcts_eval.wider_native_recovery import CONTRACT_KEYS
from pokezero.mcts_eval.wider_pending_qualification import prepare, VALID
from pokezero.mcts_eval.wider_search import ARMS, SEATS, game_identity, study_seed
from pokezero.mcts_eval.wider_search import analyze


class PendingQualificationRetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.previous, self.repair, self.current = [self.root / name for name in ('previous', 'repair', 'current')]
        for directory in (self.previous, self.repair / 'src', self.current / 'src'):
            directory.mkdir(parents=True)
        for directory in (self.repair, self.current):
            (directory / 'src/example.py').write_text('same semantic source')
        self.old = {key: 'unchanged' for key in CONTRACT_KEYS}
        self.old.update(phase='QUALIFICATION_NOT_STRENGTH', source_commit='old-commit',
            source_root=str(self.previous), seeds=[study_seed(i, qualification=True) for i in range(2)],
            registered_games=8, max_boundaries=200, per_game_wall_seconds=2400.,
            reference={'workers': 20}, input_hashes={})
        self.registration = self.previous / 'registration.json'
        self.write(self.registration, self.old)
        self.ids = [game_identity(seed, seat, arm) for seed in self.old['seeds'] for seat in SEATS for arm in ARMS]
        self.rows = []
        for identity in self.ids[:6]:
            seed = int(identity.split('-')[1])
            subject = identity.split('-')[2]
            arm = identity.split('-', 3)[3]
            directory = self.previous / identity
            directory.mkdir()
            domain = f'wider-search:{seed}:{subject}'
            witness = dict(worker_pids=list(range(20)), decision_id=17, statistics_checkpoint={},
                worker_receipts=[dict(worker=i, batch={'world_draws': 1}, evidence={'draws': [
                    dict(status='ROOT_VALIDATED', released=True)]}) for i in range(20)],
                result={'world_draws': 20})
            step = dict(boundary=0, chance_seed=continuation_seed(domain, 0, 0, 'chance'), actions={subject: 1},
                evidence={subject: dict(selector='paper_reference', seed=continuation_seed(domain, 0, 0, 'search'),
                    elapsed_seconds=1., search_evidence=witness)})
            self.write(directory / 'boundary-000.json.gz', step)
            row = dict(identity=identity, seed=seed, subject=subject, arm=arm,
                status='COMPLETE' if len(self.rows) < 5 else 'REFUSED', signed_outcome=None,
                step_hashes={'boundary-000.json.gz': self.sha(directory / 'boundary-000.json.gz')})
            if row['status'] == 'REFUSED':
                row['failure_evidence'] = dict(receipts=[], errors=[['error', 0, 18, 'public_root_preparation',
                    'ReferenceRefusal: pending committed opponent action needs a sampled-policy certificate', '',
                    {'partial_batch_evidence': None}]])
            self.write(self.previous / (identity + '.json'), row)
            self.rows.append(row)
        self.refused = self.rows[-1]
        self.refused_path = self.previous / (self.refused['identity'] + '.json')
        self.probe_path = self.root / 'probe.json'
        self.probe_observer = self.probe_path.with_suffix('.py')
        self.probe_observer.write_text('exact observer')
        self.probe = dict(status='EXACT_PENDING_ROOT_AND_TRANSPORT_VALIDATED', source_status='',
            strength_inference=False, game_resumed=False, registration_sha256=self.sha(self.registration),
            observer_sha256=self.sha(self.probe_observer), source_root=str(self.repair), source_commit='repair-commit',
            fresh_source_hashes={}, input_hashes={}, identity=self.refused['identity'], boundary=1,
            result_sha256=self.sha(self.refused_path), hidden_ordinal=1, worker_ordinals=[1] * 20)
        self.observer = self.root / 'validate_games.py'
        self.observer.write_text('base semantic observer')
        self.audit_path = self.root / 'audit.json'
        self.audit = dict(schema='pokezero.wider-search.read-only-audit.v1', source_commit='old-commit',
            registration_sha256=self.sha(self.registration), phase='QUALIFICATION_NOT_STRENGTH',
            registered_games=8, strength_inference=False, observer_sha256=self.sha(self.observer),
            audited_complete_games=5, games=[dict(identity=row['identity'], status=VALID,
                result_sha256=self.sha(self.previous / (row['identity'] + '.json'))) for row in self.rows[:5]])

    @staticmethod
    def write(path, value):
        if path.suffix == '.gz':
            path.write_bytes(gzip.compress(json.dumps(value).encode()))
        else:
            path.write_text(json.dumps(value))

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def prepare(self, current=None):
        self.write(self.probe_path, self.probe)
        self.write(self.audit_path, self.audit)
        original_stat = Path.stat
        def stat(path, *args, **kwargs):
            if path == self.previous / self.refused['identity']:
                return SimpleNamespace(st_birthtime=100.)
            if path == self.refused_path:
                return SimpleNamespace(st_birthtime=102.)
            return original_stat(path, *args, **kwargs)
        def git(command, *args, **kwargs):
            return 'repair-commit' if command == 'rev-parse' else ('src/example.py' if command == 'ls-files' else '')
        with mock.patch('pokezero.mcts_eval.wider_pending_qualification.BASE_AUDIT_HASH', self.sha(self.observer)), \
             mock.patch.object(Path, 'stat', stat):
            return prepare(self.previous, self.probe_path, self.audit_path, current or self.old,
                repo=self.current, git=git, sha=self.sha, verify=mock.Mock(), bound_rows=lambda *args: self.rows,
                step_files=lambda p: sorted(p.iterdir()), restore_reference_checkpoint=lambda p:
                    SimpleNamespace(battle_id='wider-search:' + self.refused['identity']))

    def test_retains_all_five_games_and_exact_accepted_checkpoint_prefix(self):
        result = self.prepare()
        self.assertEqual(len(result['retained_complete']), 5)
        self.assertEqual(result['resume']['start_boundary'], 1)
        self.assertEqual(result['resume']['worker_ordinals'], [1] * 20)
        self.assertEqual(result['resume']['decision_id'], 17)
        self.assertEqual(result['resume']['elapsed_before_resume'], 3.)
        self.assertIn(str(self.refused_path), result['input_hashes'])
        self.assertIn('no strength inference', result['disclosure'])

    def test_treatment_change_cannot_hide_behind_recovery(self):
        with self.assertRaisesRegex(RuntimeError, 'treatment changed'):
            self.prepare(dict(self.old, nominal_decision_seconds=999))

    def test_missing_duplicate_or_wrong_hash_audit_refuses(self):
        original = self.audit['games']
        for changed in (original[:-1], original + original[:1], [dict(original[0], result_sha256='wrong'), *original[1:]]):
            self.audit['games'] = changed
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                self.prepare()
        self.audit['games'] = original

    def test_unwitnessed_semantic_change_refuses(self):
        (self.current / 'src/example.py').write_text('changed selector')
        with self.assertRaisesRegex(RuntimeError, 'unwitnessed semantic change'):
            self.prepare()

    def test_worker_ordinal_or_failed_partial_batch_cannot_be_restored(self):
        self.probe['worker_ordinals'][1] += 1
        with self.assertRaisesRegex(RuntimeError, 'ordinals drift'):
            self.prepare()
        self.probe['worker_ordinals'][1] -= 1
        self.refused['failure_evidence']['receipts'] = ['partial refused work']
        with self.assertRaisesRegex(RuntimeError, 'before-draw certificate refusal'):
            self.prepare()

    def test_refusal_is_not_a_loss_and_probe_cannot_claim_strength(self):
        self.refused['signed_outcome'] = -1
        with self.assertRaisesRegex(RuntimeError, 'unscored pending refusal'):
            self.prepare()
        self.refused['signed_outcome'] = None
        self.probe['strength_inference'] = True
        with self.assertRaisesRegex(RuntimeError, 'clean pending witness'):
            self.prepare()


class CompletePendingQualificationTests(unittest.TestCase):
    setUpBase = PendingQualificationRetentionTests.setUp
    write = staticmethod(PendingQualificationRetentionTests.write)
    sha = staticmethod(PendingQualificationRetentionTests.sha)

    def setUp(self):
        self.setUpBase()
        self.study = self.root / 'qualification'
        self.study.mkdir()
        retained = {row['identity']: dict(path=str(self.previous / (row['identity'] + '.json')),
            registration=str(self.registration)) for row in self.rows[:5]}
        self.m = dict(self.old, source_commit='new-commit', input_hashes={},
            repair_retention=dict(kind='public-pending-qualification-repair', retained_complete=retained,
                resume={'identity': self.refused['identity'], 'prefix_steps': {}}, disclosure='explicit mixed-source qualification',
                historical_semantic_audit=str(self.audit_path)), retained_input_hashes={})
        self.write(self.study / 'registration.json', self.m)
        self.complete_rows, self.audit_records = [], []
        for identity in self.ids:
            retained_binding = retained.get(identity)
            path = Path(retained_binding['path']) if retained_binding else self.study / (identity + '.json')
            reg = self.registration if retained_binding else self.study / 'registration.json'
            directory = path.parent / identity
            directory.mkdir(exist_ok=True)
            if not (directory / 'boundary-000.json.gz').exists():
                self.write(directory / 'boundary-000.json.gz', {'boundary': 0})
            cell = dict(identity=identity, seed=int(identity.split('-')[1]), subject=identity.split('-')[2],
                arm=identity.split('-', 3)[3], status='COMPLETE', signed_outcome=0,
                registration_sha256=self.sha(reg), step_hashes={p.name: self.sha(p) for p in directory.iterdir()})
            self.write(path, cell)
            self.complete_rows.append(cell)
            self.audit_records.append(dict(identity=identity, result_sha256=self.sha(path),
                status=VALID, result_path=str(path), registration_path=str(reg),
                historical_original_source_proof_retained=bool(retained_binding),
                pending_draws_independently_bound=1 if identity == self.refused['identity'] else 0))
        self.readout_path = self.study / 'READOUT.json'
        self.value = analyze(self.m['seeds'], self.complete_rows)
        self.value.update(source_commit='new-commit', input_hashes={}, phase='QUALIFICATION_NOT_STRENGTH',
            literal_homogeneous_source=False, recovery_disclosure=self.m['repair_retention']['disclosure'],
            retained_complete_games=sorted(retained), recovered_game=self.refused['identity'],
            statistically_supported_advantage=False, inferential_test_allowed=False,
            status='QUALIFICATION_COMPLETE_NOT_STRENGTH')

    def validate(self):
        from pokezero.mcts_eval.wider_pending_qualification import validate_complete
        self.write(self.readout_path, self.value)
        self.write(self.study / 'registration.json', self.m)
        with mock.patch('pokezero.mcts_eval.wider_native_recovery.bind_qualified_source',
                        return_value={'qualified_source': 'bound'}):
            return validate_complete(self.readout_path, self.old, repo=self.current, git=mock.Mock(),
                sha=self.sha, verify=mock.Mock(), bound_rows=lambda *args: self.complete_rows)

    def test_complete_mixed_qualification_is_never_a_strength_result(self):
        result = self.validate()
        self.assertEqual(result['qualified_source_commit'], 'new-commit')
        self.assertIn('no strength inference', result['mode'])
        self.assertEqual(result['readout_sha256'], self.sha(self.readout_path))

    def test_forged_readout_or_missing_game_cannot_qualify(self):
        self.value['statistically_supported_advantage'] = True
        with self.assertRaisesRegex(RuntimeError, 'incomplete or readout differs'):
            self.validate()
        self.value['statistically_supported_advantage'] = False
        self.complete_rows.pop()
        with self.assertRaisesRegex(RuntimeError, 'incomplete or readout differs'):
            self.validate()

    def test_changed_qualification_roster_is_not_admitted(self):
        self.m['seeds'] = self.m['seeds'][:1]
        with self.assertRaisesRegex(RuntimeError, 'roster drift'):
            self.validate()

    def bind(self, *, changed=None):
        from pokezero.mcts_eval.wider_pending_qualification import bind_complete_audit
        self.write(self.audit_path, {'historical original-source proof': True})
        self.m['retained_input_hashes'][str(self.audit_path)] = self.sha(self.audit_path)
        # The current registration is rewritten only in this synthetic fixture;
        # its freshly generated result bindings must follow that fixture version.
        self.write(self.study / 'registration.json', self.m)
        for record in self.audit_records:
            if not record['historical_original_source_proof_retained']:
                path = Path(record['result_path'])
                cell = json.loads(path.read_text())
                cell['registration_sha256'] = self.sha(self.study / 'registration.json')
                self.write(path, cell)
                record['result_sha256'] = self.sha(path)
        observer = self.root / 'validate_pending_qualification_r7.py'
        observer.write_text('independent full pending audit')
        audit = dict(schema='pokezero.wider-search.pending-qualification-read-only-audit.v1',
            observer_sha256=self.sha(observer), original_observer_sha256=self.sha(self.observer),
            source_commit='new-commit', registration_sha256=self.sha(self.study / 'registration.json'),
            phase='QUALIFICATION_NOT_STRENGTH', strength_inference=False, registered_games=8,
            audited_complete_games=8, games=self.audit_records, complete_roster_valid=True,
            literal_homogeneous_source=False, retained_original_audit_sha256=self.sha(self.audit_path))
        if changed:
            changed(audit)
        path = self.root / 'full-audit.json'
        self.write(path, audit)
        with mock.patch('pokezero.mcts_eval.wider_pending_qualification.PENDING_AUDIT_HASH', self.sha(observer)), \
             mock.patch('pokezero.mcts_eval.wider_pending_qualification.BASE_AUDIT_HASH', self.sha(self.observer)):
            return bind_complete_audit(path, self.readout_path, sha=self.sha)

    def test_all_eight_independent_audit_results_and_boundaries_are_bound(self):
        result = self.bind()
        self.assertIn(str(self.study / 'registration.json'), result)
        self.assertEqual(len([p for p in result if p.endswith('boundary-000.json.gz')]), 8)

    def test_partial_duplicate_wrong_hash_or_unconditioned_pending_audit_refuses(self):
        changes = [lambda a: a.update(audited_complete_games=7),
            lambda a: a['games'].__setitem__(-1, a['games'][0]),
            lambda a: a['games'][0].update(result_sha256='wrong'),
            lambda a: next(r for r in a['games'] if r['identity'] == self.refused['identity']).update(
                pending_draws_independently_bound=0),
            lambda a: a.update(literal_homogeneous_source=True)]
        original = json.loads(json.dumps(self.audit_records))
        for change in changes:
            self.audit_records = json.loads(json.dumps(original))
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.bind(changed=change)


if __name__ == '__main__':
    unittest.main()
