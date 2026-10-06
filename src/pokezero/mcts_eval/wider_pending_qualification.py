"""Create-only qualification recovery, never a reduced strength experiment.

Retain every completed disjoint qualification game and every accepted boundary.
The prior missing-certificate refusal is unscored. The new path conditions the
interrupted public root; historical games remain explicitly mixed-source.
"""
import json
import math
from pathlib import Path

from .followthrough import continuation_seed
from .wider_native_recovery import CONTRACT_KEYS
from .wider_recovery import read_step, require
from .wider_search import ARMS, SEATS, game_identity, study_seed

KIND = 'public-pending-qualification-repair'
BASE_AUDIT_HASH = 'df7748a6c812a913cd5314128860c1a84ea166c4357315722304664508557b7e'
VALID = 'HASHES_REQUESTS_OPPONENT_POLICY_SEARCH_WITNESSES_AND_TERMINAL_REPLAY_VALID'


def resume_prefix(row, path, steps, old, probe, *, sha, restore_reference_checkpoint):
    require(row['status'] == 'REFUSED' and row['signed_outcome'] is None
            and row['arm'] == 'paper_reference' and row['identity'] == probe['identity']
            and sha(path) == probe['result_sha256'] and len(steps) == probe['boundary'],
            'exact unscored pending refusal required')
    failure = row['failure_evidence']
    errors = failure.get('errors', [])
    require(failure['receipts'] == [] and errors and all(
        error[3] == 'public_root_preparation'
        and error[4] == 'ReferenceRefusal: pending committed opponent action needs a sampled-policy certificate'
        and error[6]['partial_batch_evidence'] is None for error in errors),
        'only before-draw certificate refusal may resume')
    ordinals, selections, last = [0] * old['reference']['workers'], 0, None
    require(0 < len(steps) < old['max_boundaries'], 'invalid accepted prefix size')
    for boundary, path_step in enumerate(steps):
        step = read_step(path_step)
        require(path_step.name == f'boundary-{boundary:03d}.json.gz' and step['boundary'] == boundary,
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
        require(len(set(witness['worker_pids'])) == len(ordinals), 'accepted worker roster drift')
        counts = [0] * len(ordinals)
        for receipt in witness['worker_receipts']:
            worker = receipt['worker']
            require(type(worker) is int and 0 <= worker < len(ordinals), 'accepted worker identity drift')
            draws = receipt['evidence']['draws']
            require(len(draws) == receipt['batch']['world_draws']
                    and all(d['status'] == 'ROOT_VALIDATED' and d['released'] for d in draws),
                    'unvalidated accepted draw')
            counts[worker] += len(draws)
        require(sum(counts) == witness['result']['world_draws'], 'accepted aggregate draw count drift')
        ordinals = [a + b for a, b in zip(ordinals, counts)]
        last = path_step, witness
    require(last is not None and ordinals == probe['worker_ordinals']
            and ordinals[0] == probe['hidden_ordinal'], 'pending witness draw ordinals drift')
    checkpoint = restore_reference_checkpoint(last[1]['statistics_checkpoint'])
    require(checkpoint.battle_id == 'wider-search:' + row['identity'], 'accepted Q/N/M/F identity drift')
    born = getattr((path.parent / row['identity']).stat(), 'st_birthtime', None)
    stopped = getattr(path.stat(), 'st_birthtime', None)
    require(born is not None and stopped is not None and stopped >= born, 'historical wall envelope missing')
    elapsed = stopped - born + 1.
    measured = sum(read_step(p)['evidence'].get(row['subject'], {}).get('elapsed_seconds', 0.) for p in steps)
    require(math.isfinite(elapsed) and measured <= elapsed < old['per_game_wall_seconds'], 'historical game cap exhausted')
    return dict(identity=row['identity'], result_path=str(path), checkpoint_step=str(last[0]),
        prefix_steps={p.name: str(p) for p in steps}, start_boundary=len(steps), prior_selections=selections,
        worker_ordinals=ordinals, decision_id=last[1]['decision_id'], elapsed_before_resume=elapsed,
        wall_clock_envelope=dict(directory_birth=born, refused_result_birth=stopped, margin_seconds=1.),
        recovery='accepted Q/N/M/F and all worker ordinals; original actions/chance; no refused batches or private P')


def prepare(previous, probe_path, audit_path, current, *, repo, git, sha, verify,
            bound_rows, step_files, restore_reference_checkpoint):
    previous, probe_path, audit_path = map(Path, (previous, probe_path, audit_path))
    registered = previous / 'registration.json'
    old = json.loads(registered.read_text())
    verify(old, source_root=Path(old['source_root']))
    require(old['phase'] == current['phase'] == 'QUALIFICATION_NOT_STRENGTH'
            and old['registered_games'] == current['registered_games'] == 8
            and old['seeds'] == [study_seed(i, qualification=True) for i in range(2)], 'qualification roster drift')
    for key in CONTRACT_KEYS:
        require(old[key] == current[key], 'qualification treatment changed: ' + key)
    probe = json.loads(probe_path.read_text())
    require(probe['status'] == 'EXACT_PENDING_ROOT_AND_TRANSPORT_VALIDATED'
            and probe['source_status'] == '' and probe['strength_inference'] is False
            and probe['game_resumed'] is False and probe['registration_sha256'] == sha(registered)
            and probe['observer_sha256'] == sha(probe_path.with_suffix('.py')), 'exact clean pending witness missing')
    repair_root = Path(probe['source_root'])
    require(git('rev-parse', 'HEAD', root=repair_root) == probe['source_commit']
            and not git('status', '--porcelain', root=repair_root), 'pinned pending repair source drift')
    inputs = {str(p): sha(p) for p in (registered, probe_path, probe_path.with_suffix('.py'), audit_path)}
    for bindings in (old['input_hashes'], probe['input_hashes'], probe['fresh_source_hashes']):
        for name, digest in bindings.items():
            require(sha(name) == digest, 'historical/witness input drift: ' + name)
            inputs[name] = digest
    # Only this registry adapter may differ from the exact observed semantic
    # source. Keep its new identity explicit, including the operational driver.
    excluded = {'scripts/wider_search_comparison.py', 'src/pokezero/mcts_eval/wider_pending_qualification.py'}
    names = set(git('ls-files', 'src', 'scripts', root=repair_root).splitlines())
    names.update(git('ls-files', 'src', 'scripts', root=repo).splitlines())
    for name in names - excluded:
        require((repair_root / name).is_file() and (repo / name).is_file()
                and sha(repair_root / name) == sha(repo / name), 'unwitnessed semantic change: ' + name)
    audit = json.loads(audit_path.read_text())
    observer = audit_path.with_name('validate_games.py')
    require(audit['schema'] == 'pokezero.wider-search.read-only-audit.v1'
            and audit['source_commit'] == old['source_commit']
            and audit['registration_sha256'] == sha(registered)
            and audit['phase'] == 'QUALIFICATION_NOT_STRENGTH' and audit['registered_games'] == 8
            and audit['strength_inference'] is False
            and audit['observer_sha256'] == sha(observer) == BASE_AUDIT_HASH,
            'historical full-trajectory qualification audit binding drift')
    inputs[str(observer)] = sha(observer)
    records = {r['identity']: r for r in audit['games']}
    require(len(records) == len(audit['games']) == audit['audited_complete_games'], 'duplicate audited game')
    rows = bound_rows(previous, old)
    expected = {game_identity(seed, seat, arm) for seed in old['seeds'] for seat in SEATS for arm in ARMS}
    require(len({r['identity'] for r in rows}) == len(rows)
            and all(r['identity'] in expected for r in rows), 'historical roster drift')
    complete, refused = {}, []
    for row in rows:
        path = previous / (row['identity'] + '.json')
        steps = step_files(previous / row['identity'])
        require({p.name: sha(p) for p in steps} == row['step_hashes'], 'historical boundary inventory drift')
        inputs.update({str(p): sha(p) for p in (path, *steps)})
        if row['status'] == 'COMPLETE':
            record = records.get(row['identity'])
            require(record is not None and record['status'] == VALID and record['result_sha256'] == sha(path),
                    'retained complete game lacks original-source full replay')
            complete[row['identity']] = dict(path=str(path), registration=str(registered),
                step_dir=str(path.parent / row['identity']), source_commit=old['source_commit'], sha256=sha(path))
        else:
            refused.append((row, path, steps))
    require(set(records) == set(complete) and len(complete) == 5 and len(refused) == 1,
            'expected all five complete games plus exactly one unscored pending refusal')
    resume = resume_prefix(*refused[0], old, probe, sha=sha, restore_reference_checkpoint=restore_reference_checkpoint)
    return dict(kind=KIND, retained_complete=complete, original_registration=str(registered),
        original_source_commit=old['source_commit'], audited_repair_source_root=str(repair_root),
        audited_repair_source_commit=probe['source_commit'], exact_failed_draw_certificate=str(probe_path),
        historical_semantic_audit=str(audit_path), resume=resume, input_hashes=inputs,
        disclosure='disjoint qualification only: five original-source COMPLETE games and 37 accepted steps '
            'retained by hash; resumed root uses public-only joint pending-policy conditioning; mixed source, '
            'no strength inference and no claim of eight homogeneous-source games')
