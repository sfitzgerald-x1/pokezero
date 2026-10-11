"""Create-only faint-boundary repair: retain every game and accepted prefix.

The whole fixed roster and all-five-historical-cluster sensitivity survive.
The repair's separately pinned source must finish all eight qualification games.
"""
import json
from pathlib import Path

from .wider_native_recovery import (
    CONTRACT_KEYS, bind_qualification_audit, uncertainty_seeds, validate_native_qualification,
    verify_historical,
)
from .wider_recovery import combined_steps, require
from .wider_search import ARMS, SEATS, game_identity, study_seed
from .wider_trapping_recovery import reference_resume

KIND = 'native-and-reference-faint-repair'
HISTORICAL_OBSERVER = 'f69c390bfbecf8387391a66a54504a083293a6f19ae288af7a0b7ce7c346f7f7'
VALID = 'HASHES_REQUESTS_OPPONENT_POLICY_SEARCH_WITNESSES_AND_TERMINAL_REPLAY_VALID'


def validate_resumed_draw(draws, probe):
    """Bind worker zero's first accepted draw to the repaired exact-root probe.

    The old failed unconditional candidate is not the conditional posterior.
    Do not demand a worker-zero batch if none completed before the deadline.
    """
    if not draws:
        return
    expected, actual = probe['witnesses'][0]['draw'], draws[0]
    require(actual['status'] == 'ROOT_VALIDATED' and actual['released'] is True
        and all(actual[key] == expected[key] for key in (
            'packed_team_sha256', 'materialization_seed', 'pending_policy_conditioning')),
        'resumed conditional draw differs from exact forced-root qualification')


def validate_prior_audit(audit, old, registration_digest):
    require(audit['schema'] == 'pokezero.wider-search.mixed-trapping-recovery-read-only-audit.v4'
        and audit['observer_sha256'] == HISTORICAL_OBSERVER
        and audit['source_commit'] == old['source_commit']
        and audit['registration_sha256'] == registration_digest
        and audit['registered_games'] == 256 and audit['audited_complete_games'] == len(audit['games']) == 19
        and audit['strength_inference'] is False and audit['literal_homogeneous_source'] is False,
        'all nineteen original-source replay proofs required')
    records = {row['identity']: row for row in audit['games']}
    require(len(records) == 19 and all(row['status'] == VALID for row in records.values()),
            'duplicate or invalid retained replay record')
    return records


def prepare(previous, probe_path, audit_path, qualification_path, qualification_audit_path,
            conditioning_audit_path, current, *, repo, git, sha, verify, bound_rows,
            step_files, restore_reference_checkpoint):
    previous, probe_path, audit_path = map(Path, (previous, probe_path, audit_path))
    registered = previous / 'registration.json'
    old = verify_historical(registered)
    require(old['phase'] == 'FIXED_64_SEED_CONFIRMATION' and old['registered_games'] == 256
        and old['seeds'] == [study_seed(i) for i in range(64)]
        and old['repair_retention']['kind'] == 'native-and-reference-trapping-repair',
        'original fixed mixed-source confirmation required')
    for key in CONTRACT_KEYS:
        require(current[key] == old[key], 'faint repair changes the fixed contract: ' + key)
    probe = json.loads(probe_path.read_text())
    qualified = json.loads((Path(qualification_path).parent / 'registration.json').read_text())
    require(probe['status'] == 'EXACT_FORCED_ENCORE_ROOT_AND_PENDING_RESIDUAL_REPLAY_VALIDATED'
        and probe['strength_inference'] is False and probe['game_resumed'] is False
        and probe['retained_source_commit'] == old['source_commit']
        and probe['registration_sha256'] == sha(registered)
        and probe['observer_sha256'] == sha(probe_path.with_suffix('.py'))
        and probe['source_commit'] == qualified['source_commit']
        and probe['source_root'] == qualified['source_root']
        and len(probe['witnesses']) == 3 and all(w['queue']['residual_queue_retained'] is True
            and w['queue']['original_actor_root_preserved'] is True
            and w['queue']['after_duration'] == w['queue']['before_duration'] - 1
            and w['draw']['status'] == 'ROOT_VALIDATED' and w['draw']['released'] is True
            for w in probe['witnesses']), 'exact forced root/residual/source qualification missing')
    inputs = {str(p): sha(p) for p in (registered, probe_path, probe_path.with_suffix('.py'), audit_path)}
    for bindings in (old['input_hashes'], old.get('retained_input_hashes', {}), probe['input_hashes']):
        for path, digest in bindings.items():
            require(sha(path) == digest, 'historical or exact-probe input drift: ' + path)
            inputs[path] = digest
    q = validate_native_qualification(qualification_path, current, repo=repo, git=git, sha=sha,
        verify=verify, bound_rows=bound_rows)
    inputs.update(q['semantic_input_hashes'])
    inputs.update(bind_qualification_audit(qualification_audit_path, qualification_path, sha=sha))
    conditional_path = Path(conditioning_audit_path)
    conditional = json.loads(conditional_path.read_text())
    qualification_roster = {game_identity(s, seat, arm) for s in qualified['seeds'] for seat in SEATS for arm in ARMS}
    require(conditional['schema'] == 'pokezero.wider-search.interrupted-qualification-audit.v2'
        and conditional['source_commit'] == qualified['source_commit']
        and conditional['registration_sha256'] == sha(Path(qualification_path).parent / 'registration.json')
        and conditional['audited_complete_games'] == len(conditional['games']) == 8
        and {r['identity'] for r in conditional['games']} == qualification_roster
        and all(r['status'] == 'PUBLIC_INTERRUPTED_CERTIFICATES_VALID' for r in conditional['games'])
        and conditional['complete_roster_valid'] is True and conditional['strength_inference'] is False
        and conditional['full_replay_audit_sha256'] == sha(qualification_audit_path)
        and conditional['observer_sha256'] == sha(conditional_path.with_name('validate_interrupted_qualification_r8.py')),
        'all-eight interrupted-transition qualification audit required')
    inputs.update(conditional['input_hashes'])
    inputs[str(conditional_path)] = sha(conditional_path)
    inputs[str(conditional_path.with_name('validate_interrupted_qualification_r8.py'))] = conditional['observer_sha256']
    for p in (Path(qualification_path), Path(qualification_path).parent / 'registration.json'):
        inputs[str(p)] = sha(p)
    observer = audit_path.with_name('validate_trapping_recovered_games.py')
    require(sha(observer) == HISTORICAL_OBSERVER, 'historical replay observer drift')
    inputs[str(observer)] = sha(observer)
    audit = json.loads(audit_path.read_text())
    records = validate_prior_audit(audit, old, sha(registered))
    rows = bound_rows(previous, old)
    expected = {game_identity(s, seat, arm) for s in old['seeds'] for seat in SEATS for arm in ARMS}
    require(len(rows) == 20 and len({row['identity'] for row in rows}) == 20
        and all(row['identity'] in expected for row in rows), 'retained inventory drift')
    complete, refused = {}, []
    for row in rows:
        inherited = old['repair_retention']['retained_complete'].get(row['identity'])
        path = Path(inherited['path']) if inherited else previous / (row['identity'] + '.json')
        registration = Path(inherited['registration']) if inherited else registered
        m = json.loads(registration.read_text())
        steps = combined_steps(path.parent, row, m, sha=sha, step_files=step_files)
        require({p.name: sha(p) for p in steps} == row['step_hashes'], 'accepted boundary inventory drift')
        inputs.update({str(p): sha(p) for p in (path, registration, *steps)})
        if row['status'] == 'COMPLETE':
            require(row['identity'] in records and records[row['identity']]['result_sha256'] == sha(path),
                    'complete result lacks its bound original replay proof')
            complete[row['identity']] = dict(path=str(path), registration=str(registration),
                step_dir=str(path.parent / row['identity']), source_commit=m['source_commit'], sha256=sha(path))
        else:
            refused.append((row, path, steps))
    require(set(complete) == set(records) and len(refused) == 1, 'preserve exactly nineteen and one unscored refusal')
    resume = reference_resume(*refused[0], old, probe, restore_reference_checkpoint=restore_reference_checkpoint, sha=sha)
    require(resume['worker_ordinals'] == probe['worker_ordinals']
        and probe['witnesses'][0]['draw']['pending_policy_conditioning']['transition_kind']
            == 'pre-upkeep-faint-replacement', 'qualified worker positions or forced-root certificate drift')
    uncertain = uncertainty_seeds(rows, old)
    require(uncertain == old['repair_retention']['uncertain_seed_clusters'] and len(uncertain) == 5,
            'do not reduce or outcome-select historical uncertainty')
    return dict(kind=KIND, retained_complete=complete, original_registration=str(registered),
        original_source_commit=old['source_commit'], audited_repair_source_root=qualified['source_root'],
        audited_repair_source_commit=qualified['source_commit'], qualified_source_commit=qualified['source_commit'],
        qualification_readout=str(qualification_path), qualification_semantic_audit=str(qualification_audit_path),
        qualification_conditioning_audit=str(conditional_path), exact_failed_draw_certificate=str(probe_path),
        historical_semantic_audit=str(audit_path), uncertain_seed_clusters=uncertain,
        resume=resume, input_hashes=inputs,
        disclosure='native/trapping/pending/faint repair chain: preserve all nineteen complete original-source games '
                   'and all 75 accepted boundaries; full 64/256 roster and all five historical sensitivity clusters; '
                   'not homogeneous-source evidence; no failed partial work or private opponent commitment restored')
