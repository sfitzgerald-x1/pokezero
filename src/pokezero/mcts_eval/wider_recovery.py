"""Explicit compatible-repair retention; never silently pool source identities.

Retain COMPLETE cells, replay only accepted actions, and recover the last
accepted aggregate plus each worker's draw ordinal. Refused-decision batches
never enter that state. The unchanged fixed roster and inference rule survive.
"""
import gzip
import json
import math
from pathlib import Path

from pokezero.mcts_eval.followthrough import continuation_seed
from pokezero.mcts_eval.wider_search import ARMS, SEATS, game_identity, study_seed


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read_step(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())


def combined_steps(output, cell, m, *, sha, step_files):
    local = {path.name: path for path in step_files(output/cell['identity'])}
    prefix = cell.get('retained_prefix_steps', {})
    registered = m.get('repair_retention', {}).get('resume', {})
    if prefix or cell['identity'] == registered.get('identity'):
        require(cell['identity'] == registered.get('identity')
                and prefix == registered.get('prefix_steps'), 'unregistered retained prefix')
    for name, path in prefix.items():
        require(name not in local and sha(path) == m['retained_input_hashes'].get(path),
                'retained prefix duplicate or drift')
        local[name] = Path(path)
    return [path for _, path in sorted(local.items())]


def prepare(previous, proof_path, probe_path, qualification_path, m, *,
            repo, bound_rows, git, sha, verify, restore_reference_checkpoint):
    REPO = repo
    previous, proof_path, probe_path = map(Path, (previous, proof_path, probe_path))
    old = json.loads((previous/'registration.json').read_text())
    verify(old, source_root=Path(old['source_root']))
    require(old['phase'] == 'FIXED_64_SEED_CONFIRMATION'
            and old['seeds'] == [study_seed(i) for i in range(64)]
            and old['registered_games'] == 256, 'recovery cannot change the fixed confirmation roster')
    for key in ('seeds', 'registered_games', 'checkpoint_sha256', 'showdown_commit',
                'set_source_hash', 'nominal_decision_seconds', 'per_decision_safety_seconds',
                'max_boundaries', 'per_game_wall_seconds', 'opponent', 'initial_state',
                'first_action', 'rng', 'incumbent', 'reference', 'primary_analysis'):
        require(m[key] == old[key], 'recovery treatment/analysis changed: '+key)
    proof, probe = json.loads(proof_path.read_text()), json.loads(probe_path.read_text())
    require(proof['status'] == 'ACCEPTED_ROOT_PAYLOADS_UNCHANGED'
            and proof['retained_source_commit'] == old['source_commit']
            and proof['accepted_reference_roots'] > 0,
            'missing accepted-root noninterference evidence')
    require(probe['status'] == 'ROOT_VALIDATED' and probe['strength_inference'] is False
            and probe['retained_source_commit'] == old['source_commit'], 'missing exact failed-draw qualification')
    require(proof['repair_source_sha256'] == sha(REPO/'src/pokezero/local_showdown.py'),
            'constructor differs from audited repair')
    repair_root = Path(proof['repair_root'])
    require(git('status', '--porcelain', root=repair_root) == '', 'audited repair source is dirty')
    allowed_recovery = {'scripts/wider_search_comparison.py',
        'src/pokezero/mcts_eval/paper_reference.py',
        'src/pokezero/mcts_eval/paper_reference_parallel.py',
        'src/pokezero/mcts_eval/wider_search.py'}
    for name, digest in proof['unchanged_input_hashes'].items():
        require(sha(repair_root/name) == digest, 'audited repair input drift: '+name)
        if name not in allowed_recovery:
            require(sha(REPO/name) == digest, 'undeclared current input change: '+name)
    inputs = {str(path): sha(path) for path in (proof_path, probe_path,
        proof_path.with_suffix('.py'), probe_path.with_suffix('.py'), previous/'registration.json')}
    require(proof['observer_sha256'] == inputs[str(proof_path.with_suffix('.py'))]
            and probe['observer_sha256'] == inputs[str(probe_path.with_suffix('.py'))], 'observer drift')
    complete, refused = {}, []
    indexed = {row['identity']: row for row in bound_rows(previous, old)}
    certificates = {cell['identity']: cell for cell in proof['cells'] if cell['study'] == str(previous)}
    expected = {game_identity(seed, seat, arm) for seed in m['seeds'] for seat in SEATS for arm in ARMS}
    for identity, row in indexed.items():
        require(identity in expected and identity in certificates, 'unregistered retained game')
        path, step_dir = previous/(identity+'.json'), previous/identity
        certificate = certificates[identity]
        require(certificate['result_sha256'] == sha(path)
                and certificate['registration_sha256'] == sha(previous/'registration.json'), 'certificate binding drift')
        if row['arm'] == 'paper_reference' and row['status'] == 'COMPLETE':
            require(len(certificate['accepted_reference_root_comparisons']) == row['own_decisions'],
                    'complete reference game has unaudited accepted roots')
        inputs[str(path)] = sha(path)
        paths = sorted(step_dir.iterdir())
        for step in paths:
            inputs[str(step)] = sha(step)
        binding = dict(path=str(path), registration=str(previous/'registration.json'),
            step_dir=str(step_dir), source_commit=old['source_commit'], sha256=sha(path))
        if row['status'] == 'COMPLETE':
            complete[identity] = binding
        else:
            require(row['status'] == 'REFUSED' and row['arm'] == 'paper_reference'
                    and row['signed_outcome'] is None, 'only an explicitly unscored reference refusal can recover')
            refused.append((row, paths))
    require(len(refused) == 1, 'recovery requires exactly one preserved refused cell')
    row, paths = refused[0]
    require(row['identity'] == probe['identity'] and len(paths) == probe['boundary'], 'refusal/probe identity drift')
    require(probe['result_sha256'] == sha(previous/(row['identity']+'.json')), 'failed-draw result binding drift')
    require(0 < len(paths) < m['max_boundaries'], 'invalid replay prefix size')
    ordinals, selections, last, root_comparisons = [0]*m['reference']['workers'], 0, None, []
    for boundary, path in enumerate(paths):
        step = read_step(path)
        require(path.name == f'boundary-{boundary:03d}.json.gz' and step['boundary'] == boundary,
                'retained boundary gap')
        domain = f"wider-search:{row['seed']}:{row['subject']}"
        require(step['chance_seed'] == continuation_seed(domain, 0, boundary, 'chance'), 'retained chance domain drift')
        if row['subject'] not in step['actions']:
            continue
        own = step['evidence'][row['subject']]
        require(own['selector'] == 'paper_reference'
                and own['seed'] == continuation_seed(domain, 0, boundary, 'search'), 'retained own-policy domain drift')
        selections += 1
        root_comparisons.append(boundary)
        witness = own['search_evidence']
        require(len(set(witness['worker_pids'])) == len(ordinals), 'retained worker roster drift')
        draws = 0
        for receipt in witness['worker_receipts']:
            worker = receipt['worker']
            require(type(worker) is int and 0 <= worker < len(ordinals), 'retained worker identity drift')
            batch_draws = receipt['evidence']['draws']
            require(all(draw['status'] == 'ROOT_VALIDATED' and draw['released'] for draw in batch_draws),
                    'unvalidated or unreleased accepted-prefix draw')
            require(len(batch_draws) == receipt['batch']['world_draws'], 'worker draw count drift')
            ordinals[worker] += len(batch_draws)
            draws += len(batch_draws)
        require(draws == witness['result']['world_draws'], 'decision draw count drift')
        last = (path, witness)
    require(root_comparisons == [record['boundary'] for record in
        certificates[row['identity']]['accepted_reference_root_comparisons']], 'accepted-root audit is incomplete')
    require(last is not None and ordinals[0] == probe['hidden_ordinal'], 'accepted draw position differs from exact refusal')
    snapshot = restore_reference_checkpoint(last[1]['statistics_checkpoint'])
    require(snapshot.battle_id == 'wider-search:'+row['identity'], 'accepted checkpoint battle drift')
    # Directory birth precedes reset/play; terminal birth follows the refusal.
    # Charge that entire observed span plus one second, not merely search time.
    born = getattr((previous/row['identity']).stat(), 'st_birthtime', None)
    stopped = getattr((previous/(row['identity']+'.json')).stat(), 'st_birthtime', None)
    require(born is not None and stopped is not None and stopped >= born,
            'historical wall-clock envelope unavailable; do not reset the cap')
    elapsed = stopped-born+1.
    measured = sum(read_step(path)['evidence'].get(row['subject'], {}).get('elapsed_seconds', 0.) for path in paths)
    require(math.isfinite(elapsed) and elapsed >= measured and elapsed < m['per_game_wall_seconds'],
            'historical wall-clock envelope invalid or exhausted')
    qualification_path = Path(qualification_path)
    q = json.loads((qualification_path.parent/'registration.json').read_text())
    require(q['source_commit'] == old['source_commit'] and q['input_hashes'] == old['input_hashes'],
            'retained qualification differs from originally qualified confirmation')
    for path in (qualification_path, qualification_path.parent/'registration.json'):
        inputs[str(path)] = sha(path)
    for cell in bound_rows(qualification_path.parent, q):
        require(cell['status'] == 'COMPLETE', 'retained qualification incomplete')
        path = qualification_path.parent/(cell['identity']+'.json')
        inputs[str(path)] = sha(path)
        certificate = next((record for record in proof['cells'] if record['study'] == str(qualification_path.parent)
                            and record['identity'] == cell['identity']), None)
        require(certificate is not None and certificate['result_sha256'] == sha(path),
                'qualification missing noninterference certificate')
        if cell['arm'] == 'paper_reference':
            require(len(certificate['accepted_reference_root_comparisons']) == cell['own_decisions'],
                    'qualification has unaudited accepted roots')
        for step in (qualification_path.parent/cell['identity']).iterdir():
            inputs[str(step)] = sha(step)
    return dict(retained_complete=complete,
        original_registration=str(previous/'registration.json'), original_source_commit=old['source_commit'],
        accepted_root_certificate=str(proof_path), exact_failed_draw_certificate=str(probe_path),
        audited_repair_source_root=str(repair_root), audited_repair_source_commit=git('rev-parse', 'HEAD', root=repair_root),
        qualification_readout=str(qualification_path),
        resume=dict(identity=row['identity'], result_path=str(previous/(row['identity']+'.json')),
            checkpoint_step=str(last[0]), prefix_steps={path.name: str(path) for path in paths},
            start_boundary=len(paths), prior_selections=selections, worker_ordinals=ordinals,
            decision_id=last[1]['decision_id'], elapsed_before_resume=elapsed,
            wall_clock_envelope=dict(directory_birth=born, refused_result_birth=stopped, margin_seconds=1.),
            recovery='accepted aggregate Q/N/M/F and per-worker draw positions; fresh P caches; no failed-decision batches'),
        input_hashes=inputs,
        disclosure='constructor-compatible retained cells and one accepted-prefix recovery, not a literal homogeneous-source rerun')


def add_recovery_sensitivity(result, registration, *, sha):
    """A positive claim must survive ANY score for the recovered seed cluster.

    Keep the preregistered full-roster test visible. Add a stronger bound rather
    than exclude a troublesome cluster or claim recovery was literally identical.
    All nine possible quarter-unit contrasts are considered; no outcome selects
    the sensitivity rule. Missing games still prohibit confirmatory inference.
    """
    from pokezero.mcts_eval.wider_search import exact_signflip_p
    resume = registration['repair_retention']['resume']
    historical = json.loads(Path(resume['result_path']).read_text())
    seed = historical['seed']
    result['recovery_sensitivity'] = dict(seed=seed,
        rule='all nine possible recovered-seed quarter contrasts; fixed N retained; maximum exact p and minimum bounded-mean lower confidence limit',
        status='INCOMPLETE_NO_INFERENCE', all_scores_support_advantage=False)
    if not result['inferential_test_allowed']:
        return
    contrasts = result['contrasts']
    require(len(contrasts) == len(registration['seeds'])
            and sum(row['seed'] == seed for row in contrasts) == 1, 'recovery sensitivity roster drift')
    base = [row['difference'] for row in contrasts]
    index = next(i for i, row in enumerate(contrasts) if row['seed'] == seed)
    n = len(base)
    radius = math.sqrt(2*math.log(2/.05)/n)
    possibilities = []
    for quarter in range(-4, 5):
        values = list(base)
        values[index] = quarter/4
        possibilities.append(dict(contrast=quarter/4,
            mean_win_score_delta=sum(values)/n, exact_seed_cluster_signflip_p=exact_signflip_p(values),
            bounded_mean_lower=max(-1., sum(values)/n-radius)))
    robust = all(row['exact_seed_cluster_signflip_p'] <= .05 and row['bounded_mean_lower'] > 0
                 for row in possibilities)
    result['preregistered_full_roster_advantage'] = result['statistically_supported_advantage']
    result['statistically_supported_advantage'] = result['statistically_supported_advantage'] and robust
    result['recovery_sensitivity'].update(status='COMPLETE_FIXED_N_ALL_RECOVERED_SCORES',
        all_scores_support_advantage=robust, possible_recovered_seed_contrasts=possibilities,
        maximum_exact_p=max(row['exact_seed_cluster_signflip_p'] for row in possibilities),
        minimum_bounded_mean_lower=min(row['bounded_mean_lower'] for row in possibilities),
        historical_refusal_sha256=sha(resume['result_path']))
