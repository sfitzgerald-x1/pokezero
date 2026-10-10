"""Disclosed trapping repair after native repair; no dropped games or prefixes.

The refreshed trapping cache can affect old sampled interiors. Treat EVERY
historically touched seed as uncertain, not just the refused game. Retain the
full 64-seed/256-game roster and require fresh-source qualification first.
"""
import json
import math
from pathlib import Path

from pokezero.mcts_eval.followthrough import continuation_seed
from pokezero.mcts_eval.wider_recovery import combined_steps, read_step, require
from pokezero.mcts_eval.wider_native_recovery import (
    CONTRACT_KEYS, add_sensitivity, bind_qualification_audit,
    validate_native_qualification, verify_historical, uncertainty_seeds,
)
from pokezero.mcts_eval.wider_search import ARMS, SEATS, game_identity, study_seed

KIND = 'native-and-reference-trapping-repair'
VALID = 'HASHES_REQUESTS_OPPONENT_POLICY_SEARCH_WITNESSES_AND_TERMINAL_REPLAY_VALID'
AUDIT_SCHEMA = 'pokezero.wider-search.native-recovery-read-only-audit.v3'


def reference_resume(row, path, steps, old, probe, *, restore_reference_checkpoint, sha):
    require(row['status'] == 'REFUSED' and row['signed_outcome'] is None
            and row['arm'] == 'paper_reference' and row['identity'] == probe['identity']
            and sha(path) == probe['result_sha256'] and len(steps) == probe['boundary'],
            'only the exact unscored reference refusal may resume')
    require(0 < len(steps) < old['max_boundaries'], 'invalid accepted prefix size')
    ordinals, selections, last = [0] * old['reference']['workers'], 0, None
    for boundary, step_path in enumerate(steps):
        step = read_step(step_path)
        require(step_path.name == f'boundary-{boundary:03d}.json.gz' and step['boundary'] == boundary,
                'accepted prefix gap')
        domain = f"wider-search:{row['seed']}:{row['subject']}"
        require(step['chance_seed'] == continuation_seed(domain, 0, boundary, 'chance'), 'accepted chance drift')
        if row['subject'] not in step['actions']:
            continue
        own = step['evidence'][row['subject']]
        require(own['selector'] == 'paper_reference'
                and own['seed'] == continuation_seed(domain, 0, boundary, 'search'), 'accepted selector drift')
        selections += 1
        witness = own['search_evidence']
        require(len(set(witness['worker_pids'])) == len(ordinals), 'accepted worker process roster drift')
        counts = [0] * len(ordinals)
        for receipt in witness['worker_receipts']:
            worker = receipt['worker']
            require(type(worker) is int and 0 <= worker < len(ordinals), 'accepted worker identity drift')
            draws = receipt['evidence']['draws']
            require(len(draws) == receipt['batch']['world_draws']
                    and all(d['status'] == 'ROOT_VALIDATED' and d['released'] for d in draws),
                    'unvalidated/unreleased accepted draw')
            counts[worker] += len(draws)
        require(sum(counts) == witness['result']['world_draws'], 'accepted aggregate draw count drift')
        ordinals = [a+b for a, b in zip(ordinals, counts)]
        last = step_path, witness
    require(last is not None and ordinals[0] == probe['hidden_ordinal'], 'exact hidden draw ordinal drift')
    checkpoint = restore_reference_checkpoint(last[1]['statistics_checkpoint'])
    require(checkpoint.battle_id == 'wider-search:' + row['identity'], 'accepted Q/N/M/F identity drift')
    born = getattr((path.parent / row['identity']).stat(), 'st_birthtime', None)
    stopped = getattr(path.stat(), 'st_birthtime', None)
    require(born is not None and stopped is not None and stopped >= born, 'historical wall envelope missing')
    elapsed = stopped-born+1.
    measured = sum(read_step(p)['evidence'].get(row['subject'], {}).get('elapsed_seconds', 0.) for p in steps)
    require(math.isfinite(elapsed) and measured <= elapsed < old['per_game_wall_seconds'],
            'historical wall cap exhausted or drifted')
    return dict(identity=row['identity'], result_path=str(path), checkpoint_step=str(last[0]),
        prefix_steps={p.name: str(p) for p in steps}, start_boundary=len(steps), prior_selections=selections,
        worker_ordinals=ordinals, decision_id=last[1]['decision_id'], elapsed_before_resume=elapsed,
        wall_clock_envelope=dict(directory_birth=born, refused_result_birth=stopped, margin_seconds=1.),
        recovery='accepted Q/N/M/F plus all worker ordinals; original actions/chance; fresh private P; no refused batches')


def prepare(previous, probe_path, audit_path, qualification_path, qualification_audit_path, current, *,
            repo, git, sha, verify, bound_rows, step_files, restore_reference_checkpoint):
    previous, probe_path, audit_path = map(Path, (previous, probe_path, audit_path))
    original_path = previous / 'registration.json'
    old = verify_historical(original_path)
    require(old['phase'] == 'FIXED_64_SEED_CONFIRMATION'
            and old['seeds'] == [study_seed(i) for i in range(64)] and old['registered_games'] == 256
            and old.get('repair_retention', {}).get('kind') == 'native-semantic-repair',
            'trapping recovery requires the original fixed native-repair confirmation')
    for key in CONTRACT_KEYS:
        require(current[key] == old[key], 'repair changes fixed contract: ' + key)
    probe = json.loads(probe_path.read_text())
    require(probe['status'] == 'ROOT_VALIDATED' and probe['strength_inference'] is False
            and probe['retained_source_commit'] == old['source_commit']
            and probe['registration_sha256'] == sha(original_path)
            and probe['observer_sha256'] == sha(probe_path.with_suffix('.py')),
            'exact failed-draw repair proof missing or drifted')
    qualified = json.loads((Path(qualification_path).parent / 'registration.json').read_text())
    require(probe['fresh_source_commit'] == qualified['source_commit']
            and probe['fresh_source_root'] == qualified['source_root'], 'probe/qualification source differs')
    inputs = {str(p): sha(p) for p in (original_path, probe_path, probe_path.with_suffix('.py'), audit_path)}
    # Recursively pin old sources, steps, qualification and original audits too.
    for bindings in (old['input_hashes'], old.get('retained_input_hashes', {}),
                     probe['input_hashes'], probe['fresh_source_hashes']):
        for filename, digest in bindings.items():
            require(sha(filename) == digest, 'historical/probe input drift: ' + filename)
            inputs[filename] = digest
    q = validate_native_qualification(qualification_path, current, repo=repo, git=git, sha=sha,
                                      verify=verify, bound_rows=bound_rows)
    inputs.update(q['semantic_input_hashes'])
    inputs.update(bind_qualification_audit(qualification_audit_path, qualification_path, sha=sha))
    for p in (Path(qualification_path), Path(qualification_path).parent / 'registration.json'):
        inputs[str(p)] = sha(p)
    audit = json.loads(audit_path.read_text())
    require(audit['schema'] == AUDIT_SCHEMA and audit['source_commit'] == old['source_commit']
            and audit['registration_sha256'] == sha(original_path) and audit['strength_inference'] is False,
            'historical semantic replay audit binding drift')
    observer = audit_path.with_name('validate_native_recovered_games.py')
    require(audit['observer_sha256'] == sha(observer)
            == 'e1374cd2f34ede6fb876525ef01a8831b8b4ee9cde3f808af81b6375fcdf2ca5',
            'historical semantic observer drift')
    inputs[str(observer)] = sha(observer)
    records = {record['identity']: record for record in audit['games']}
    require(len(records) == len(audit['games']), 'duplicate historical audit record')
    rows = bound_rows(previous, old)
    expected = {game_identity(seed, seat, arm) for seed in old['seeds'] for seat in SEATS for arm in ARMS}
    require(len({r['identity'] for r in rows}) == len(rows)
            and all(r['identity'] in expected for r in rows), 'historical roster drift')
    complete, refused = {}, []
    for row in rows:
        identity = row['identity']
        inherited = old['repair_retention']['retained_complete'].get(identity)
        path = Path(inherited['path']) if inherited else previous / (identity + '.json')
        registration = Path(inherited['registration']) if inherited else original_path
        registered = json.loads(registration.read_text())
        steps = combined_steps(path.parent, row, registered, sha=sha, step_files=step_files)
        require({p.name: sha(p) for p in steps} == row['step_hashes'], 'retained decision evidence drift')
        inputs.update({str(p): sha(p) for p in (path, registration, *steps)})
        if row['status'] == 'COMPLETE':
            record = records.get(identity)
            require(record is not None and record['result_sha256'] == sha(path) and record['status'] == VALID,
                    'complete retained game lacks independent replay audit')
            complete[identity] = dict(path=str(path), registration=str(registration),
                step_dir=str(path.parent / identity), source_commit=registered['source_commit'], sha256=sha(path))
        else:
            refused.append((row, path, steps))
    require(set(records) == set(complete) and len(complete) == audit['audited_complete_games'],
            'historical audit coverage differs from retained complete roster')
    require(len(refused) == 1, 'exactly one unscored refused game required')
    resume = reference_resume(*refused[0], old, probe, restore_reference_checkpoint=restore_reference_checkpoint, sha=sha)
    uncertain = uncertainty_seeds(rows, old)
    require(set(old['repair_retention']['uncertain_seed_clusters']) <= set(uncertain)
            and 1 <= len(uncertain) <= 5, 'historical uncertainty was dropped or exceeds registered bound')
    return dict(kind=KIND, retained_complete=complete, original_registration=str(original_path),
        original_source_commit=old['source_commit'], audited_repair_source_root=qualified['source_root'],
        audited_repair_source_commit=qualified['source_commit'], exact_failed_draw_certificate=str(probe_path),
        qualification_readout=str(qualification_path), qualification_semantic_audit=str(qualification_audit_path),
        qualified_source_commit=qualified['source_commit'], historical_semantic_audit=str(audit_path),
        uncertain_seed_clusters=uncertain, resume=resume, input_hashes=inputs,
        disclosure='native plus reference trapping and public pending-policy repair: all complete historical games and accepted prefix retained, '
                   'not homogeneous-source evidence; every historically touched seed bounded at all nine paired scores')


def add_mixed_sensitivity(result, registration):
    add_sensitivity(result, registration, max_uncertain=5)
    result['mixed_repair_sensitivity'] = result.pop('native_repair_sensitivity')
