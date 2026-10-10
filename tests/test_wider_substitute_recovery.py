"""Fail-closed retention and first-draw guards for the complete fixed study."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pokezero.mcts_eval.wider_substitute_recovery import (
    HISTORICAL_OBSERVER, VALID, add_substitute_sensitivity,
    validate_prior_audit, validate_resumed_draw, prepare, bind_conditioning_audit,
)
from pokezero.mcts_eval.wider_native_recovery import CONTRACT_KEYS
from pokezero.mcts_eval.wider_search import ARMS, SEATS, game_identity, study_seed


class SubstituteRecoveryGuardTests(unittest.TestCase):
    def test_bound_original_source_audit_requires_all32_and_no_duplicates(self):
        old = dict(source_commit='old')
        audit = dict(schema='pokezero.wider-search.mixed-faint-recovery-read-only-audit.v5',
            observer_sha256=HISTORICAL_OBSERVER, source_commit='old', registration_sha256='reg',
            registered_games=256, audited_complete_games=32, strength_inference=False,
            literal_homogeneous_source=False,
            games=[dict(identity=str(i), status=VALID) for i in range(32)])
        self.assertEqual(len(validate_prior_audit(audit, old, 'reg')), 32)
        for field, value in (('audited_complete_games', 31), ('source_commit', 'new'),
                             ('strength_inference', True), ('literal_homogeneous_source', True),
                             ('registered_games', 128), ('observer_sha256', 'unbound')):
            changed = dict(audit, **{field: value})
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_prior_audit(changed, old, 'reg')
        changed = copy.deepcopy(audit)
        changed['games'][-1]['identity'] = '0'
        with self.assertRaises(RuntimeError):
            validate_prior_audit(changed, old, 'reg')

    def test_exact_first_draw_and_conditional_certificate_must_match(self):
        draw = dict(status='ROOT_VALIDATED', released=True, packed_team_sha256='team',
            materialization_seed=17, substitute_policy_conditioning=dict(attempts=20,
                sampled_substitute_hp={'p1': 61}, live_hidden_hp_used=False))
        probe = dict(witnesses=[dict(draw=draw)])
        validate_resumed_draw([copy.deepcopy(draw)], probe)
        validate_resumed_draw([], probe)  # No unfinished worker batch is imported.
        for field, value in (('released', False), ('status', 'PARTIAL'),
                             ('packed_team_sha256', 'wrong'), ('materialization_seed', 18)):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_resumed_draw([dict(draw, **{field: value})], probe)
        changed = copy.deepcopy(draw)
        changed['substitute_policy_conditioning']['live_hidden_hp_used'] = True
        with self.assertRaises(RuntimeError):
            validate_resumed_draw([changed], probe)

    def test_partial_roster_cannot_claim_strength_and_nine_is_not_optional(self):
        result = dict(inferential_test_allowed=False, statistically_supported_advantage=False)
        registration = dict(seeds=list(range(64)), repair_retention=dict(uncertain_seed_clusters=list(range(9))))
        add_substitute_sensitivity(result, registration)
        self.assertEqual(result['mixed_repair_sensitivity']['status'], 'INCOMPLETE_NO_INFERENCE')
        self.assertFalse(result['statistically_supported_advantage'])
        registration['repair_retention']['uncertain_seed_clusters'] = list(range(5))
        with self.assertRaises(RuntimeError):
            add_substitute_sensitivity(result, registration)


class SubstituteRetentionInventoryTests(unittest.TestCase):
    def test_conditioning_audit_binds_all8_terminal_hashes_and_required_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = root / 'qualification'
            qualification.mkdir()
            digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            registration = qualification / 'registration.json'
            manifest = dict(source_commit='qualified', seeds=[study_seed(i, qualification=True) for i in range(2)])
            registration.write_text(json.dumps(manifest))
            semantic = root / 'semantic.json'
            semantic.write_text('all8 semantic fixture')
            observer = root / 'validate_substitute_conditioning_r9.py'
            observer.write_text('independent checker fixture')
            records = []
            inputs = {str(p): digest(p) for p in (registration, semantic)}
            for seed in manifest['seeds']:
                for seat in SEATS:
                    for arm in ARMS:
                        identity = game_identity(seed, seat, arm)
                        terminal = qualification / (identity + '.json')
                        terminal.write_text(json.dumps(dict(identity=identity, status='COMPLETE')))
                        inputs[str(terminal)] = digest(terminal)
                        records.append(dict(identity=identity, result_sha256=digest(terminal),
                            status='PUBLIC_AND_SUBSTITUTE_CERTIFICATES_VALID'))
            audit_path = root / 'conditioning.json'
            value = dict(schema='pokezero.wider-search.substitute-conditioning-audit.v3',
                source_commit='qualified', registration_sha256=digest(registration),
                full_replay_audit_sha256=digest(semantic), observer_sha256=digest(observer),
                audited_complete_games=8, games=records, complete_roster_valid=True,
                strength_inference=False, input_hashes=inputs)
            audit_path.write_text(json.dumps(value))
            bound = bind_conditioning_audit(audit_path, qualification / 'READOUT.json', semantic, sha=digest)
            self.assertTrue(set(inputs) <= set(bound))
            for mutation in ('terminal_hash', 'missing_terminal', 'missing_semantic', 'duplicate', 'partial'):
                changed = copy.deepcopy(value)
                if mutation == 'terminal_hash':
                    changed['games'][0]['result_sha256'] = 'wrong'
                elif mutation == 'missing_terminal':
                    changed['input_hashes'].pop(str(qualification / (records[0]['identity'] + '.json')))
                elif mutation == 'missing_semantic':
                    changed['input_hashes'].pop(str(semantic))
                elif mutation == 'duplicate':
                    changed['games'][-1] = changed['games'][0]
                else:
                    changed['complete_roster_valid'] = False
                audit_path.write_text(json.dumps(changed))
                with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                    bind_conditioning_audit(audit_path, qualification / 'READOUT.json', semantic, sha=digest)

    def test_all32_completes_all13_steps_and_all9_uncertain_seeds_survive_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous, qualification = root / 'previous', root / 'qualification'
            previous.mkdir()
            qualification.mkdir()

            def write(path, value):
                path.write_text(json.dumps(value, sort_keys=True))

            def digest(path):
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()

            seeds = [study_seed(i) for i in range(64)]
            contract = {key: 'same' for key in CONTRACT_KEYS}
            contract.update(seeds=seeds, registered_games=256)
            old = dict(contract, source_commit='old', phase='FIXED_64_SEED_CONFIRMATION',
                input_hashes={}, repair_retention=dict(kind='native-and-reference-faint-repair',
                    retained_complete={}, uncertain_seed_clusters=seeds[:5]))
            registration = previous / 'registration.json'
            write(registration, old)
            rows, records = [], []
            identities = [(seed, seat, arm) for seed in seeds[:8] for seat in SEATS for arm in ARMS]
            identities.append((seeds[8], 'p1', 'paper_reference'))
            for seed, seat, arm in identities:
                identity = game_identity(seed, seat, arm)
                folder = previous / identity
                folder.mkdir()
                refused = seed == seeds[8]
                paths = [folder / f'boundary-{i:03d}.json.gz' for i in range(13 if refused else 1)]
                for path in paths:
                    path.write_bytes(b'accepted fixture boundary')
                row = dict(identity=identity, seed=seed, subject=seat, arm=arm,
                    status='REFUSED' if refused else 'COMPLETE', signed_outcome=None if refused else 1,
                    step_hashes={p.name: digest(p) for p in paths})
                write(previous / (identity + '.json'), row)
                rows.append(row)
                if not refused:
                    records.append(dict(identity=identity, status=VALID,
                        result_sha256=digest(previous / (identity + '.json'))))
            audit_path = root / 'historical.json'
            write(audit_path, dict(schema='pokezero.wider-search.mixed-faint-recovery-read-only-audit.v5',
                observer_sha256=HISTORICAL_OBSERVER, source_commit='old', registration_sha256=digest(registration),
                registered_games=256, audited_complete_games=32, strength_inference=False,
                literal_homogeneous_source=False, games=records))
            probe_path = root / 'probe.json'
            probe_source = probe_path.with_suffix('.py')
            probe_source.write_text('exact root fixture observer')
            refused_path = previous / (rows[-1]['identity'] + '.json')
            write(probe_path, dict(status='EXACT_SUBSTITUTE_ROOT_JOINT_PUBLIC_HISTORY_POSTERIOR_VALIDATED',
                strength_inference=False, game_resumed=False, retained_source_commit='old',
                observer_sha256=digest(probe_source), source_commit='qualified', source_root=str(qualification),
                input_hashes={str(registration): digest(registration), str(refused_path): digest(refused_path)},
                identity=rows[-1]['identity'], boundary=13, worker_ordinals=[10] * 20,
                witnesses=[dict(draw=dict(status='ROOT_VALIDATED', released=True,
                    substitute_policy_conditioning=dict(live_hidden_hp_used=False,
                        live_opponent_action_used=False, attempts=20))) for _ in range(3)]))
            write(qualification / 'registration.json', dict(source_commit='qualified', source_root=str(qualification)))
            write(qualification / 'READOUT.json', {})
            current = dict(contract)
            package = 'pokezero.mcts_eval.wider_substitute_recovery.'
            historical_observer = root / 'validate_faint_recovered_games_r8.py'
            observed_resume = {}

            def restore(row, path, steps, manifest, probe, **kwargs):
                observed_resume.update(identity=row['identity'], paths=steps, probe=probe)
                return dict(worker_ordinals=[10] * 20)

            def bound_sha(path):
                return HISTORICAL_OBSERVER if Path(path) == historical_observer else digest(path)

            with mock.patch(package + 'verify_historical', return_value=old), \
                    mock.patch(package + 'validate_native_qualification', return_value={'semantic_input_hashes': {}}), \
                    mock.patch(package + 'bind_qualification_audit', return_value={}), \
                    mock.patch(package + 'bind_conditioning_audit', return_value={}), \
                    mock.patch(package + 'reference_resume', side_effect=restore):
                result = prepare(previous, probe_path, audit_path, qualification / 'READOUT.json',
                    root / 'semantic.json', root / 'conditioning.json', current,
                    repo=root, git=lambda *a, **k: '', sha=bound_sha, verify=lambda *a, **k: None,
                    bound_rows=lambda *a: rows, step_files=lambda path: sorted(path.iterdir()),
                    restore_reference_checkpoint=lambda value: value)
            self.assertEqual(len(result['retained_complete']), 32)
            self.assertEqual(result['uncertain_seed_clusters'], seeds[:9])
            self.assertEqual(observed_resume['identity'], rows[-1]['identity'])
            self.assertEqual(len(observed_resume['paths']), 13)
            self.assertEqual(observed_resume['probe']['result_sha256'], digest(refused_path))
            self.assertEqual(observed_resume['probe']['hidden_ordinal'], 10)
            self.assertTrue(all(str(p) in result['input_hashes'] for p in observed_resume['paths']))


if __name__ == '__main__':
    unittest.main()
