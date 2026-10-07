"""Fail-closed continuation: fresh all8, accepted prefixes, no diagnostic work."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from pokezero.mcts_eval.followthrough import continuation_seed
from pokezero.mcts_eval.wider_guarded_staged_recovery import (
    IDENTITY, KIND, PARENT, QUALIFIED, PUBLIC_OBSERVER, SEMANTIC_OBSERVER,
    RemoveStagedGuard, bind_fresh_audits, bind_proofs, resume_capture, validate_resumed_workers,
)
from pokezero.mcts_eval.wider_search import ARMS, SEATS, game_identity, study_seed
from wider_search_comparison import register


def draw():
    return dict(ordinal=0, status='ROOT_VALIDATED', released=True, packed_team_sha256='team',
        materialization_seed=17, substitute_policy_conditioning=dict(schema='staged', attempts=2))


def probe_fixture():
    return dict(measurement=dict(worker_receipts=[dict(worker=i, evidence=dict(draws=[draw()]))
        for i in range(20)]))


class StagedGuardTests(unittest.TestCase):
    def test_first_draw_all20_compared_without_diagnostic_statistics(self):
        probe = probe_fixture()
        validate_resumed_workers(probe['measurement']['worker_receipts'], probe)
        for worker in (0, 3, 19):
            bad = deepcopy(probe['measurement']['worker_receipts'])
            bad[worker]['evidence']['draws'][0]['materialization_seed'] += 1
            with self.subTest(worker=worker), self.assertRaisesRegex(RuntimeError, 'original stream'):
                validate_resumed_workers(bad, probe)

    def test_first_cancellation_remains_unaccepted_not_a_false_root_comparison(self):
        cancelled = dict(ordinal=0, status='DEADLINE_CANCELLED', sampling_diagnostic=dict(
            schema='pokezero.world-sampling-deadline.v1', checked_at=11., deadline_at=10.,
            accepted_world=False, backed_up=False))
        receipts = [dict(worker=3, evidence=dict(draws=[cancelled]))]
        validate_resumed_workers(receipts, probe_fixture())
        for key, value in (('accepted_world', True), ('backed_up', True), ('checked_at', 9.),
                           ('deadline_at', float('nan')), ('schema', 'unknown')):
            bad = deepcopy(receipts)
            bad[0]['evidence']['draws'][0]['sampling_diagnostic'][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_resumed_workers(bad, probe_fixture())

    def test_refusal_unreleased_or_nonzero_first_ordinal_cannot_transfer(self):
        for key, value in (('status', 'REFUSED'), ('released', False), ('ordinal', 1),
                           ('substitute_policy_conditioning', {'schema': 'different'})):
            actual = draw()
            actual[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_resumed_workers([dict(worker=7, evidence=dict(draws=[actual]))], probe_fixture())

    def test_incomplete_original20_proof_refuses(self):
        probe = probe_fixture()
        probe['measurement']['worker_receipts'].pop()
        with self.assertRaisesRegex(RuntimeError, 'original20'):
            validate_resumed_workers([], probe)

    def test_only_explicit_fail_closed_guard_normalizes_run_ast(self):
        old = ast.parse("def run():\n    if worker_zero:\n        raise RuntimeError('old')\n")
        guarded = """def run():
    if recovery.get('kind') == 'native-and-reference-guarded-staged-repair':
        from pokezero.mcts_eval.wider_guarded_staged_recovery import validate_resumed_workers
        validate_resumed_workers(measured.worker_receipts, probe)
    elif worker_zero:
        raise RuntimeError('old')
"""
        self.assertEqual(ast.dump(old), ast.dump(RemoveStagedGuard().visit(ast.parse(guarded))))
        with self.assertRaises(RuntimeError):
            RemoveStagedGuard().visit(ast.parse(guarded.replace('measured.worker_receipts', 'diagnostic_checkpoint')))

    def test_kernel_confirmation_still_requires_explicit_qualified_path(self):
        with self.assertRaisesRegex(RuntimeError, 'fresh technical qualification'):
            register(NS(staged_substitute_conditioning=True, qualification=False))
        for qualification, staged in ((True, True), (False, False)):
            with self.subTest(qualification=qualification), self.assertRaisesRegex(RuntimeError, 'confirmation'):
                register(NS(staged_substitute_conditioning=staged, qualification=qualification,
                    guarded_staged_recover_from=Path('old')))

    def test_independent_proof_hashes_are_mandatory(self):
        with self.assertRaisesRegex(RuntimeError, 'proof drift'):
            bind_proofs('unknown', sha=lambda p: 'wrong')


class AcceptedPrefixTests(unittest.TestCase):
    def fixture(self):
        paths = {f'boundary-{i:03d}.json.gz': f'/original/boundary-{i:03d}.json.gz' for i in range(15)}
        capture = dict(status='TRUNCATED_FAILURE_FIELDS_AND_ACCEPTED_PREFIX_CAPTURED_NOT_A_TERMINAL',
            source_commit=PARENT, identity=IDENTITY, accepted_prefix_boundaries=15, next_boundary=15,
            last_accepted_decision_id=392, failed_decision_id=393,
            failure_receipts_truncated_and_unrecoverable=True, failed_partial_work_restored=False,
            strength_inference=False, accepted_prefix_steps=paths,
            accepted_prefix_hashes={p: 'hash' for p in paths}, worker_ordinals=[15] * 20,
            checkpoint_step=paths['boundary-014.json.gz'], checkpoint_sha256='hash')
        old = dict(per_game_wall_seconds=2400., repair_retention=dict(resume=dict(identity=IDENTITY,
            result_path='/original/refused.json',
            prefix_steps=dict(list(paths.items())[:13]), elapsed_before_resume=157., decision_id=390,
            wall_clock_envelope=dict(additional_attempts=[dict(charged_seconds=12.)]))))
        steps = {}
        for boundary in range(15):
            steps[f'/original/boundary-{boundary:03d}.json.gz'] = dict(boundary=boundary,
                chance_seed=continuation_seed('wider-search:474486314:p1', 0, boundary, 'chance'),
                actions={'p1': 3}, evidence=dict(p1=dict(selector='paper_reference', elapsed_seconds=10.,
                    seed=continuation_seed('wider-search:474486314:p1', 0, boundary, 'search'),
                    search_evidence=dict(decision_id=378 + boundary, statistics_checkpoint='accepted392',
                        worker_receipts=[dict(worker=i, batch=dict(world_draws=1),
                            evidence=dict(draws=[dict(status='ROOT_VALIDATED', released=True)])) for i in range(20)]))))
        return capture, old, steps

    def resume(self, capture, old, steps, births=(100., 200.)):
        with mock.patch('pokezero.mcts_eval.wider_guarded_staged_recovery.read_step',
                side_effect=lambda p: steps[str(p)]), mock.patch.object(Path, 'stat',
                side_effect=[NS(st_birthtime=t) for t in births]):
            return resume_capture(capture, old, '/historical', sha=lambda p: 'hash',
                restore_reference_checkpoint=lambda x: NS(battle_id='wider-search:' + IDENTITY))

    def test_all15_checkpoint392_ordinals_and_cumulative_wall_preserved(self):
        capture, old, steps = self.fixture()
        result = self.resume(capture, old, steps)
        self.assertEqual(result['decision_id'], 392)
        self.assertEqual(result['worker_ordinals'], [15] * 20)
        self.assertEqual(result['start_boundary'], result['prior_selections'])
        self.assertEqual(result['start_boundary'], 15)
        self.assertEqual(result['elapsed_before_resume'], 258.)
        self.assertEqual([a['charged_seconds'] for a in result['wall_clock_envelope']['additional_attempts']], [12., 101.])
        self.assertEqual(old['repair_retention']['resume']['elapsed_before_resume'], 157.)
        self.assertEqual(result['checkpoint_step'], capture['checkpoint_step'])
        self.assertEqual(len(result['prefix_steps']), 15)

    def test_diagnostic393_or_failed_rng_cannot_replace_accepted_state(self):
        for key, value in (('last_accepted_decision_id', 393), ('worker_ordinals', [16] * 20),
                           ('failed_partial_work_restored', True), ('strength_inference', True),
                           ('accepted_prefix_boundaries', 13)):
            capture, old, steps = self.fixture()
            capture[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.resume(capture, old, steps)

    def test_gap_hash_chance_or_selector_drift_refuses(self):
        for mode in ('gap', 'hash', 'chance', 'selector', 'refusal', 'original_path'):
            capture, old, steps = self.fixture()
            row = steps['/original/boundary-014.json.gz']
            if mode == 'gap':
                capture['accepted_prefix_steps'].pop('boundary-013.json.gz')
            elif mode == 'hash':
                capture['accepted_prefix_hashes']['boundary-014.json.gz'] = 'different'
            elif mode == 'chance':
                row['chance_seed'] += 1
            elif mode == 'selector':
                row['evidence']['p1']['selector'] = 'raw'
            elif mode == 'refusal':
                row['evidence']['p1']['search_evidence']['worker_receipts'][3]['evidence']['draws'][0]['status'] = 'REFUSED'
            else:
                old['repair_retention']['resume']['prefix_steps']['boundary-000.json.gz'] = '/different'
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                self.resume(capture, old, steps)

    def test_missing_reversed_or_exhausted_wall_cannot_reset_allowance(self):
        for births in ((None, 200.), (200., 100.), (100., 3000.)):
            capture, old, steps = self.fixture()
            with self.subTest(births=births), self.assertRaises(RuntimeError):
                self.resume(capture, old, steps, births=births)


class FreshEightAuditTests(unittest.TestCase):
    def fixture(self, root):
        q = root / 'READOUT.json'
        q.write_text('{}')
        seeds = [study_seed(i, qualification=True) for i in range(2)]
        m = dict(phase='QUALIFICATION_NOT_STRENGTH', source_commit=QUALIFIED, registered_games=8,
            seeds=seeds, reference=dict(staged_substitute_conditioning=True))
        (root / 'registration.json').write_text(json.dumps(m))
        records = []
        for seed in seeds:
            for seat in SEATS:
                for arm in ARMS:
                    identity = game_identity(seed, seat, arm)
                    (root / identity).mkdir()
                    (root / identity / 'boundary-000.json.gz').write_bytes(b'bound')
                    (root / (identity + '.json')).write_text(json.dumps(dict(identity=identity,
                        status='COMPLETE', registration_sha256='hash', step_hashes={'boundary-000.json.gz': 'hash'})))
                    records.append(dict(identity=identity, result_sha256='hash', status=''))
        semantic, public = root / 'semantic.json', root / 'public.json'
        for path, observer, digest, valid in ((semantic, 'validate_guarded_qualification_games_r31.py',
                SEMANTIC_OBSERVER, 'HASHES_REQUESTS_OPPONENT_POLICY_SEARCH_WITNESSES_AND_TERMINAL_REPLAY_VALID'),
                (public, 'validate_guarded_qualification_conditioning_r32.py', PUBLIC_OBSERVER,
                'PUBLIC_AND_SUBSTITUTE_CERTIFICATES_VALID')):
            (root / observer).write_text('observer')
            path.write_text(json.dumps(dict(source_commit=QUALIFIED, observer_sha256=digest,
                registration_sha256='hash', audited_complete_games=8, complete_roster_valid=True,
                strength_inference=False, full_replay_audit_sha256='hash',
                games=[dict(r, status=valid) for r in records])))
        def sha(path):
            return {'validate_guarded_qualification_games_r31.py': SEMANTIC_OBSERVER,
                'validate_guarded_qualification_conditioning_r32.py': PUBLIC_OBSERVER}.get(Path(path).name, 'hash')
        return semantic, public, q, sha

    def test_all8_fresh_audited_terminals_required(self):
        with TemporaryDirectory() as tmp:
            semantic, public, q, sha = self.fixture(Path(tmp))
            bound = bind_fresh_audits(semantic, public, q, sha=sha)
            self.assertIn(str(q), bound)
            self.assertEqual(sum(Path(p).name.startswith('seed-') and p.endswith('.json') for p in bound), 8)

    def test_partial_default_off_wrong_source_duplicate_or_refused_audit_fails(self):
        for mode in ('partial', 'off', 'source', 'duplicate', 'refused', 'public_binding'):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                semantic, public, q, sha = self.fixture(Path(tmp))
                path = q.parent / 'registration.json' if mode in ('off', 'source') else public
                data = json.loads(path.read_text())
                if mode == 'partial':
                    data['complete_roster_valid'] = False
                elif mode == 'off':
                    data['reference']['staged_substitute_conditioning'] = False
                elif mode == 'source':
                    data['source_commit'] = PARENT
                elif mode == 'duplicate':
                    data['games'][-1] = data['games'][0]
                elif mode == 'refused':
                    data['games'][-1]['status'] = 'REFUSED'
                else:
                    data['full_replay_audit_sha256'] = 'different'
                path.write_text(json.dumps(data))
                with self.assertRaises(RuntimeError):
                    bind_fresh_audits(semantic, public, q, sha=sha)


if __name__ == '__main__':
    unittest.main()
