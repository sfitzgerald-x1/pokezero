"""Create-only Substitute/callback repair; preserve the full 64/256 study."""
import json
from pathlib import Path

from .wider_native_recovery import (CONTRACT_KEYS, add_sensitivity, bind_qualification_audit,
    validate_native_qualification, verify_historical, uncertainty_seeds)
from .wider_recovery import combined_steps, require
from .wider_search import ARMS, SEATS, game_identity, study_seed
from .wider_trapping_recovery import reference_resume

KIND = 'native-and-reference-substitute-repair'
HISTORICAL_OBSERVER = '6949ae4ecf46f9d373c8e1f9950fa6d8a7ddbd1631ba13e3ac809ba5dfcd4878'
VALID = 'HASHES_REQUESTS_OPPONENT_POLICY_SEARCH_WITNESSES_AND_TERMINAL_REPLAY_VALID'


def validate_resumed_draw(draws, probe):
    if not draws:
        return  # A worker-zero batch need not finish before the soft deadline.
    actual, expected = draws[0], probe['witnesses'][0]['draw']
    require(actual['status'] == 'ROOT_VALIDATED' and actual['released'] is True
        and all(actual[k] == expected[k] for k in ('packed_team_sha256',
            'materialization_seed', 'substitute_policy_conditioning')),
        'resumed Substitute draw differs from exact-root proof')


def validate_prior_audit(audit, old, registration_digest):
    require(audit['schema'] == 'pokezero.wider-search.mixed-faint-recovery-read-only-audit.v5'
        and audit['observer_sha256'] == HISTORICAL_OBSERVER
        and audit['source_commit'] == old['source_commit']
        and audit['registration_sha256'] == registration_digest
        and audit['registered_games'] == 256
        and audit['audited_complete_games'] == len(audit['games']) == 32
        and audit['strength_inference'] is False and audit['literal_homogeneous_source'] is False,
        'all32 original-source semantic proofs required')
    records = {r['identity']: r for r in audit['games']}
    require(len(records) == 32 and all(r['status'] == VALID for r in records.values()),
        'duplicate or invalid retained semantic proof')
    return records


def bind_conditioning_audit(path, qualification_path, semantic_path, *, sha):
    path, qualification_path, semantic_path = map(Path, (path, qualification_path, semantic_path))
    registration_path = qualification_path.parent / 'registration.json'
    qualified = json.loads(registration_path.read_text())
    audit = json.loads(path.read_text())
    observer = path.with_name('validate_substitute_conditioning_r9.py')
    expected = {game_identity(s, seat, arm) for s in qualified['seeds'] for seat in SEATS for arm in ARMS}
    require(audit['schema'] == 'pokezero.wider-search.substitute-conditioning-audit.v3'
        and audit['source_commit'] == qualified['source_commit']
        and audit['registration_sha256'] == sha(registration_path)
        and audit['full_replay_audit_sha256'] == sha(semantic_path)
        and audit['observer_sha256'] == sha(observer)
        and audit['audited_complete_games'] == len(audit['games']) == 8
        and {r['identity'] for r in audit['games']} == expected
        and all(r['status'] == 'PUBLIC_AND_SUBSTITUTE_CERTIFICATES_VALID' for r in audit['games'])
        and audit['complete_roster_valid'] is True and audit['strength_inference'] is False,
        'all-eight independent public/Substitute conditioning audit required')
    inputs = {str(path): sha(path), str(observer): sha(observer)}
    required_inputs = {str(registration_path): sha(registration_path), str(semantic_path): sha(semantic_path)}
    for record in audit['games']:
        terminal = qualification_path.parent / (record['identity'] + '.json')
        require(record['result_sha256'] == sha(terminal), 'conditioning terminal hash drift')
        required_inputs[str(terminal)] = sha(terminal)
    require(all(audit['input_hashes'].get(name) == digest for name, digest in required_inputs.items()),
        'conditioning audit omits required registration, semantic audit or terminal bindings')
    for name, digest in audit['input_hashes'].items():
        require(sha(name) == digest, 'conditioning audit input drift: ' + name)
        inputs[name] = digest
    return inputs


def prepare(previous, probe_path, audit_path, qualification_path, qualification_audit_path,
            conditioning_audit_path, current, *, repo, git, sha, verify, bound_rows,
            step_files, restore_reference_checkpoint):
    previous, probe_path, audit_path = map(Path, (previous, probe_path, audit_path))
    registered = previous / 'registration.json'
    old = verify_historical(registered)
    require(old['phase'] == 'FIXED_64_SEED_CONFIRMATION' and old['registered_games'] == 256
        and old['seeds'] == [study_seed(i) for i in range(64)]
        and old['repair_retention']['kind'] == 'native-and-reference-faint-repair',
        'fixed mixed-faint historical confirmation required')
    for key in CONTRACT_KEYS:
        require(current[key] == old[key], 'Substitute repair changes fixed contract: ' + key)
    probe = json.loads(probe_path.read_text())
    qualified = json.loads((Path(qualification_path).parent / 'registration.json').read_text())
    require(probe['status'] == 'EXACT_SUBSTITUTE_ROOT_JOINT_PUBLIC_HISTORY_POSTERIOR_VALIDATED'
        and probe['strength_inference'] is False and probe['game_resumed'] is False
        and probe['retained_source_commit'] == old['source_commit']
        and probe['observer_sha256'] == sha(probe_path.with_suffix('.py'))
        and probe['input_hashes'][str(registered)] == sha(registered)
        and probe['source_commit'] == qualified['source_commit']
        and probe['source_root'] == qualified['source_root']
        and len(probe['witnesses']) == 3
        and all(w['draw']['status'] == 'ROOT_VALIDATED' and w['draw']['released'] is True
            and w['draw']['substitute_policy_conditioning']['live_hidden_hp_used'] is False
            and w['draw']['substitute_policy_conditioning']['live_opponent_action_used'] is False
            and 1 <= w['draw']['substitute_policy_conditioning']['attempts'] <= 128
            for w in probe['witnesses']), 'exact conditional root/source proof missing')
    inputs = {str(p): sha(p) for p in (registered, probe_path, probe_path.with_suffix('.py'), audit_path)}
    for bindings in (old['input_hashes'], old.get('retained_input_hashes', {}), probe['input_hashes']):
        for name, digest in bindings.items():
            require(sha(name) == digest, 'historical/probe input drift: ' + name)
            inputs[name] = digest
    q = validate_native_qualification(qualification_path, current, repo=repo, git=git, sha=sha,
        verify=verify, bound_rows=bound_rows)
    inputs.update(q['semantic_input_hashes'])
    inputs.update(bind_qualification_audit(qualification_audit_path, qualification_path, sha=sha))
    inputs.update(bind_conditioning_audit(conditioning_audit_path, qualification_path,
        qualification_audit_path, sha=sha))
    for p in (Path(qualification_path), Path(qualification_path).parent / 'registration.json'):
        inputs[str(p)] = sha(p)
    observer = audit_path.with_name('validate_faint_recovered_games_r8.py')
    require(sha(observer) == HISTORICAL_OBSERVER, 'historical observer drift')
    inputs[str(observer)] = sha(observer)
    records = validate_prior_audit(json.loads(audit_path.read_text()), old, sha(registered))
    rows = bound_rows(previous, old)
    expected = {game_identity(s, seat, arm) for s in old['seeds'] for seat in SEATS for arm in ARMS}
    require(len(rows) == len({r['identity'] for r in rows}) == 33
        and all(r['identity'] in expected for r in rows), 'all33 historical terminals required')
    complete, refused = {}, []
    for row in rows:
        inherited = old['repair_retention']['retained_complete'].get(row['identity'])
        path = Path(inherited['path']) if inherited else previous / (row['identity'] + '.json')
        registration = Path(inherited['registration']) if inherited else registered
        m = json.loads(registration.read_text())
        steps = combined_steps(path.parent, row, m, sha=sha, step_files=step_files)
        require({p.name: sha(p) for p in steps} == row['step_hashes'], 'historical step inventory drift')
        inputs.update({str(p): sha(p) for p in (path, registration, *steps)})
        if row['status'] == 'COMPLETE':
            require(row['identity'] in records and records[row['identity']]['result_sha256'] == sha(path),
                'complete result lacks bound original-source replay proof')
            complete[row['identity']] = dict(path=str(path), registration=str(registration),
                step_dir=str(path.parent / row['identity']), source_commit=m['source_commit'], sha256=sha(path))
        else:
            refused.append((row, path, steps))
    require(set(complete) == set(records) and len(refused) == 1,
        'preserve all32 completes and exactly one unscored refusal')
    row, path, steps = refused[0]
    require(probe['input_hashes'][str(path)] == sha(path) and len(steps) == probe['boundary'] == 13,
        'exact refusal result/prefix binding drift')
    resume_probe = dict(probe, result_sha256=sha(path), hidden_ordinal=probe['worker_ordinals'][0])
    resume = reference_resume(row, path, steps, old, resume_probe,
        restore_reference_checkpoint=restore_reference_checkpoint, sha=sha)
    require(resume['worker_ordinals'] == probe['worker_ordinals'], 'all20 accepted draw positions drift')
    uncertain = uncertainty_seeds(rows, old)
    require(len(uncertain) == 9 and set(old['repair_retention']['uncertain_seed_clusters']) <= set(uncertain),
        'do not reduce or outcome-select historical uncertainty')
    return dict(kind=KIND, retained_complete=complete, original_registration=str(registered),
        original_source_commit=old['source_commit'], audited_repair_source_root=qualified['source_root'],
        audited_repair_source_commit=qualified['source_commit'], qualified_source_commit=qualified['source_commit'],
        qualification_readout=str(qualification_path), qualification_semantic_audit=str(qualification_audit_path),
        qualification_conditioning_audit=str(conditioning_audit_path), exact_failed_draw_certificate=str(probe_path),
        historical_semantic_audit=str(audit_path), uncertain_seed_clusters=uncertain,
        resume=resume, input_hashes=inputs,
        disclosure='all32 complete historical games and13 accepted boundaries retained across disclosed native, '
            'trapping, pending, faint, Substitute and intrinsic-callback repairs; not homogeneous source; '
            'full64/256 fixed roster and all9 outcome-blind historical sensitivity clusters; '
            'no refused work, live hidden HP or private opponent commitment restored')


def add_substitute_sensitivity(result, registration):
    require(len(registration['repair_retention']['uncertain_seed_clusters']) == 9,
        'registered nine-cluster historical sensitivity required')
    add_sensitivity(result, registration, max_uncertain=9)
    result['mixed_repair_sensitivity'] = result.pop('native_repair_sensitivity')
