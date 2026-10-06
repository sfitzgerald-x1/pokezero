"""Explicit native-semantic repair, with historical-cluster worst-case bounds.

Historical games are preserved, not advertised as homogeneous repaired-source
measurements. Native changes can affect interior search before any public
Struggle request. Therefore every seed touched before the repair is uncertain,
regardless of its score. No redraw, refusal-as-loss, or outcome-selected pooling.
"""
from collections import Counter
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from pokezero.mcts_eval.followthrough import continuation_seed
from pokezero.mcts_eval.wider_recovery import read_step, require
from pokezero.mcts_eval.wider_search import ARMS, SEATS, analyze, game_identity, study_seed

KIND = 'native-semantic-repair'
OPERATIONAL_CHANGES = {'scripts/wider_search_comparison.py',
                       'src/pokezero/mcts_eval/wider_faint_recovery.py',
                       'src/pokezero/mcts_eval/wider_native_recovery.py',
                       'src/pokezero/mcts_eval/wider_trapping_recovery.py',
                       'src/pokezero/mcts_eval/wider_pending_qualification.py'}
CONTRACT_KEYS = ('seeds', 'registered_games', 'checkpoint_sha256', 'showdown_commit',
    'set_source_hash', 'nominal_decision_seconds', 'per_decision_safety_seconds',
    'max_boundaries', 'per_game_wall_seconds', 'opponent', 'initial_state',
    'first_action', 'rng', 'incumbent', 'reference', 'primary_analysis')
QUALIFICATION_OBSERVER_HASH = 'df7748a6c812a913cd5314128860c1a84ea166c4357315722304664508557b7e'


def verify_historical(manifest_path):
    """Use the historical source/package to verify its receipt, never a bypass."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest['source_root'])
    environment = dict(os.environ)
    packages = [manifest['native_package']]
    if manifest.get('native_build_receipt'):
        receipt = json.loads(Path(manifest['native_build_receipt']).read_text())
        if receipt.get('python_engine_package'):
            packages.append(str(Path(receipt['python_engine_package']).parent))
    environment['PYTHONPATH'] = os.pathsep.join((*packages, str(root/'src'), str(root/'scripts')))
    subprocess.run([sys.executable, '-c',
        'import json,sys; from pathlib import Path; from wider_search_comparison import verify; '
        'm=json.loads(Path(sys.argv[1]).read_text()); verify(m,source_root=Path(m["source_root"]))',
        str(manifest_path)], cwd=root, env=environment, check=True)
    return manifest


def bind_qualified_source(qualification, current, *, repo, git, sha):
    """Only the registry/recovery adapter may differ from the qualified source."""
    origin = Path(qualification['source_root'])
    names = set()
    for root in (origin, repo):
        names.update(git('ls-files', 'src', 'scripts', 'rust/pokezero-search',
                         'third_party', 'pyproject.toml', root=root).splitlines())
    bound = {}
    for name in sorted(names - OPERATIONAL_CHANGES - {'third_party/foul-play'}):
        require((origin/name).is_file() and (repo/name).is_file()
                and sha(origin/name) == sha(repo/name), 'unqualified semantic change: '+name)
        bound[str(origin/name)] = sha(origin/name)
    for key in CONTRACT_KEYS:
        if key not in ('seeds', 'registered_games'):
            require(current[key] == qualification[key], 'qualified treatment drift: '+key)
    require(current['native_package'] == qualification['native_package'],
            'qualification used a different native binary package')
    return bound


def validate_native_qualification(path, current, *, repo, git, sha, verify, bound_rows):
    path = Path(path)
    require(path.name == 'READOUT.json', 'qualification requires canonical READOUT.json')
    registered = json.loads((path.parent/'registration.json').read_text())
    if registered.get('repair_retention', {}).get('kind') == 'public-pending-qualification-repair':
        from .wider_pending_qualification import validate_complete
        return validate_complete(path, current, repo=repo, git=git, sha=sha, verify=verify, bound_rows=bound_rows)
    verify(registered, source_root=Path(registered['source_root']))
    require(registered['phase'] == 'QUALIFICATION_NOT_STRENGTH'
            and registered['seeds'] == [study_seed(i, qualification=True) for i in range(2)]
            and registered['registered_games'] == 8, 'disjoint qualification roster drift')
    bindings = bind_qualified_source(registered, current, repo=repo, git=git, sha=sha)
    cells = bound_rows(path.parent, registered)
    result = analyze(registered['seeds'], cells)
    result.update(source_commit=registered['source_commit'], input_hashes=registered['input_hashes'],
        phase=registered['phase'], statistically_supported_advantage=False, inferential_test_allowed=False,
        status='QUALIFICATION_COMPLETE_NOT_STRENGTH' if not result['missing_seed_clusters'] else 'QUALIFICATION_INCOMPLETE')
    require(result == json.loads(path.read_text())
            and result['status'] == 'QUALIFICATION_COMPLETE_NOT_STRENGTH',
            'new native qualification incomplete or readout differs from durable evidence')
    return dict(mode='new native qualification; only disclosed operational adapters differ',
        readout_sha256=sha(path), registration_sha256=sha(path.parent/'registration.json'),
        qualified_source_commit=registered['source_commit'], semantic_input_hashes=bindings)


def uncertainty_seeds(rows, manifest):
    touched = {row['seed'] for row in rows}
    require(touched and touched <= set(manifest['seeds']), 'historical seed roster drift')
    # Freeze uncertainty from the historical inventory, never from win/loss or
    # which positions happened to display the synthetic move publicly.
    return [seed for seed in manifest['seeds'] if seed in touched]


def bind_qualification_audit(audit_path, qualification_path, *, sha):
    """Require the completed disjoint qualification's full semantic replay."""
    require(audit_path is not None, 'new native qualification needs complete semantic replay audit')
    audit_path, study = Path(audit_path), Path(qualification_path).parent
    audit = json.loads(audit_path.read_text())
    registration_path = study/'registration.json'
    m = json.loads(registration_path.read_text())
    if m.get('repair_retention', {}).get('kind') == 'public-pending-qualification-repair':
        from .wider_pending_qualification import bind_complete_audit
        return bind_complete_audit(audit_path, qualification_path, sha=sha)
    observer = audit_path.with_name('validate_games.py')
    require(sha(observer) == QUALIFICATION_OBSERVER_HASH
            and audit['observer_sha256'] == QUALIFICATION_OBSERVER_HASH
            and audit['schema'] == 'pokezero.wider-search.read-only-audit.v1'
            and audit['source_commit'] == m['source_commit']
            and audit['registration_sha256'] == sha(registration_path)
            and audit['phase'] == 'QUALIFICATION_NOT_STRENGTH'
            and audit['registered_games'] == audit['audited_complete_games'] == 8
            and len(audit['games']) == 8 and audit['complete_roster_valid'] is True
            and audit['strength_inference'] is False, 'qualification semantic audit binding drift')
    expected = {game_identity(seed, seat, arm) for seed in m['seeds'] for seat in SEATS for arm in ARMS}
    require(len(expected) == 8 and {row['identity'] for row in audit['games']} == expected,
            'qualification semantic audit roster drift')
    inputs = {str(audit_path): sha(audit_path), str(observer): sha(observer)}
    for row in audit['games']:
        path = study/(row['identity']+'.json')
        require(row['status'] == 'HASHES_REQUESTS_OPPONENT_POLICY_SEARCH_WITNESSES_AND_TERMINAL_REPLAY_VALID'
                and row['result_sha256'] == sha(path), 'qualification semantic result drift')
        inputs[str(path)] = sha(path)
        cell = json.loads(path.read_text())
        files = sorted((study/row['identity']).iterdir())
        require(cell['identity'] == row['identity'] and cell['status'] == 'COMPLETE'
                and cell['registration_sha256'] == sha(registration_path)
                and files and all(p.is_file() for p in files)
                and {p.name: sha(p) for p in files} == cell['step_hashes'],
                'qualification semantic decision evidence drift')
        inputs.update({str(p): sha(p) for p in files})
    return inputs


def prepare(previous, certificate_path, audit_paths, qualification_path, current, *,
            repo, git, sha, verify, bound_rows, step_files, qualification_audit_path=None):
    previous, certificate_path = Path(previous), Path(certificate_path)
    original_path = previous/'registration.json'
    old = verify_historical(original_path)
    require(old['phase'] == 'FIXED_64_SEED_CONFIRMATION'
            and old['seeds'] == [study_seed(i) for i in range(64)]
            and old['registered_games'] == 256, 'fixed confirmation roster drift')
    for key in CONTRACT_KEYS:
        require(current[key] == old[key], 'repair cannot change registered contract: '+key)
    certificate = json.loads(certificate_path.read_text())
    require(certificate['status'] == 'EXACT_REFUSED_ROOT_VALID_WITH_REAL_NATIVE_STRUGGLE'
            and certificate['strength_inference'] is False, 'missing native repair certificate')
    qualified = json.loads((Path(qualification_path).parent/'registration.json').read_text())
    require(certificate['source_commit'] == qualified['source_commit'],
            'repair certificate and native qualification source differ')
    repair_root = Path(certificate['source_root'])
    require(git('rev-parse', 'HEAD', root=repair_root) == certificate['source_commit']
            and not git('status', '--porcelain', root=repair_root), 'native repair source drift')
    inputs = {str(original_path): sha(original_path), str(certificate_path): sha(certificate_path)}
    for path, digest in certificate['input_hashes'].items():
        require(sha(path) == digest, 'repair certificate input drift: '+path)
        inputs[path] = digest
    new_probe = next((json.loads(Path(path).read_text()) for path in certificate['input_hashes']
        if Path(path).name == 'probe-incumbent-struggle-repair-r2.json'), None)
    require(new_probe is not None and new_probe['validation'] == 'VALID'
            and new_probe['fresh_source_commit'] == certificate['source_commit']
            and new_probe['fresh_source_status'] == ''
            and new_probe['original_registration_sha256'] == sha(original_path), 'exact repair probe drift')
    for path, digest in new_probe['fresh_source_and_binary_hashes'].items():
        require(sha(path) == digest, 'exact probe source/binary drift: '+path)
        inputs[path] = digest
    q = validate_native_qualification(qualification_path, current, repo=repo,
        git=git, sha=sha, verify=verify, bound_rows=bound_rows)
    inputs.update(q['semantic_input_hashes'])
    inputs.update(bind_qualification_audit(qualification_audit_path, qualification_path, sha=sha))
    for path in (Path(qualification_path), Path(qualification_path).parent/'registration.json'):
        inputs[str(path)] = sha(path)
    rows = bound_rows(previous, old)
    expected = {game_identity(seed, seat, arm) for seed in old['seeds'] for seat in SEATS for arm in ARMS}
    require(len({row['identity'] for row in rows}) == len(rows)
            and all(row['identity'] in expected for row in rows), 'duplicate or unregistered historical game')
    audited = {}
    for audit_path in map(Path, audit_paths):
        audit = json.loads(audit_path.read_text())
        require(audit['schema'] == 'pokezero.wider-search.recovery-read-only-audit.v2'
                and audit['source_commit'] == old['source_commit']
                and audit['registration_sha256'] == sha(original_path)
                and audit['strength_inference'] is False, 'historical semantic audit binding drift')
        for filename, key in (('validate_recovered_games.py', 'observer_sha256'),
                              ('validate_games.py', 'original_observer_sha256')):
            observer = audit_path.with_name(filename)
            require(sha(observer) == audit[key], 'historical audit observer drift')
            inputs[str(observer)] = sha(observer)
        inputs[str(audit_path)] = sha(audit_path)
        for record in audit['games']:
            require(record['status'] == 'HASHES_REQUESTS_OPPONENT_POLICY_SEARCH_WITNESSES_AND_TERMINAL_REPLAY_VALID',
                    'historical game lacks complete semantic replay')
            if record['identity'] in audited:
                require(record == audited[record['identity']], 'conflicting historical audit records')
            audited[record['identity']] = record
    complete, refused = {}, []
    from pokezero.mcts_eval.wider_recovery import combined_steps
    for row in rows:
        identity = row['identity']
        retained = old.get('repair_retention', {}).get('retained_complete', {}).get(identity)
        path = Path(retained['path']) if retained else previous/(identity+'.json')
        registration = Path(retained['registration']) if retained else original_path
        registered = json.loads(registration.read_text())
        inputs[str(path)] = sha(path)
        inputs[str(registration)] = sha(registration)
        steps = combined_steps(path.parent, row, registered, sha=sha, step_files=step_files)
        require({p.name: sha(p) for p in steps} == row['step_hashes'], 'historical decision evidence drift')
        inputs.update({str(p): sha(p) for p in steps})
        if row['status'] == 'COMPLETE':
            require(identity in audited and audited[identity]['result_sha256'] == sha(path),
                    'complete retained game missing semantic audit')
            complete[identity] = dict(path=str(path), registration=str(registration),
                step_dir=str(path.parent/identity), source_commit=registered['source_commit'], sha256=sha(path))
        else:
            require(row['status'] == 'REFUSED' and row['signed_outcome'] is None
                    and row['arm'] == 'deep_incumbent', 'only unscored incumbent refusal may resume here')
            refused.append((row, path, steps))
    require(len(refused) == 1, 'native recovery requires exactly one refused game')
    row, path, steps = refused[0]
    require(row['identity'] == new_probe['identity'] and sha(path) == new_probe['original_result_sha256']
            and row['step_hashes'] == new_probe['prefix_hashes']
            and len(steps) == new_probe['boundary'] > 0, 'refusal does not match qualified exact prefix')
    selections = 0
    for boundary, step_path in enumerate(steps):
        step = read_step(step_path)
        require(step_path.name == f'boundary-{boundary:03d}.json.gz' and step['boundary'] == boundary,
                'accepted prefix has a gap')
        domain = f"wider-search:{row['seed']}:{row['subject']}"
        require(step['chance_seed'] == continuation_seed(domain, 0, boundary, 'chance'), 'prefix chance-domain drift')
        if row['subject'] in step['actions']:
            own = step['evidence'][row['subject']]
            require(own['selector'] == 'deep_incumbent'
                    and own['seed'] == continuation_seed(domain, 0, boundary, 'search'), 'prefix own-selector drift')
            selections += 1
    require(len(steps) < old['max_boundaries'], 'accepted prefix already exhausted boundary cap')
    born = getattr((path.parent/row['identity']).stat(), 'st_birthtime', None)
    stopped = getattr(path.stat(), 'st_birthtime', None)
    require(born is not None and stopped is not None and stopped >= born,
            'historical wall-clock envelope unavailable')
    elapsed = stopped-born+1.
    measured = sum(read_step(p)['evidence'].get(row['subject'], {}).get('elapsed_seconds', 0.) for p in steps)
    require(math.isfinite(elapsed) and measured <= elapsed < old['per_game_wall_seconds'],
            'historical wall-clock cap invalid or exhausted')
    uncertain = uncertainty_seeds(rows, old)
    require(1 <= len(uncertain) <= 4, 'bounded native recovery only supports at most four historical clusters')
    return dict(kind=KIND, retained_complete=complete, original_registration=str(original_path),
        original_source_commit=old['source_commit'], audited_repair_source_root=str(repair_root),
        audited_repair_source_commit=certificate['source_commit'], native_repair_certificate=str(certificate_path),
        exact_failed_draw_certificate=str(certificate_path), qualification_readout=str(qualification_path),
        qualification_semantic_audit=str(qualification_audit_path),
        qualified_source_commit=qualified['source_commit'], uncertain_seed_clusters=uncertain,
        resume=dict(identity=row['identity'], result_path=str(path),
            prefix_steps={p.name: str(p) for p in steps}, start_boundary=len(steps), prior_selections=selections,
            worker_ordinals=None, decision_id=None, elapsed_before_resume=elapsed,
            wall_clock_envelope=dict(directory_birth=born, refused_result_birth=stopped, margin_seconds=1.),
            recovery='replay accepted native actions/chance; fresh private policy caches; no refused action/work accepted'),
        input_hashes=inputs,
        disclosure='native semantic repair: preserve complete games and accepted prefixes, not homogeneous repaired-source evidence; bound every historically touched seed cluster at all possible paired scores')


def worst_case_statistics(differences, uncertain_indices, *, alpha=.05, max_uncertain=4):
    """Exact p and conservative CI over every possible historical score vector.

    Statistics are invariant to permutation of the uncertain contrasts. Enumerate
    their multisets (495 for four clusters) rather than 9**4 ordered assignments;
    multiplicities document complete coverage, not selected favorable outcomes.
    """
    n = len(differences)
    listed_indices = list(uncertain_indices)
    indices = set(listed_indices)
    require(type(max_uncertain) is int and 1 <= max_uncertain <= 5
            and n > 0 and 0 < alpha < 1 and 1 <= len(indices) <= max_uncertain
            and len(indices) == len(listed_indices)
            and all(type(i) is int and 0 <= i < n for i in indices), 'invalid sensitivity roster')
    quarters = []
    for value in differences:
        require(math.isfinite(value), 'nonfinite quarter contrast')
        quarter = round(value*4)
        require(math.isfinite(value) and -4 <= quarter <= 4
                and abs(quarter-value*4) < 1e-9, 'invalid quarter contrast')
        quarters.append(quarter)
    known = [v for i, v in enumerate(quarters) if i not in indices]
    baseline = Counter({0: 1})
    for value in known:
        updated = Counter()
        for total, count in baseline.items():
            updated[total+value] += count
            updated[total-value] += count
        baseline = updated
    distributions, checked, covered = {}, 0, 0
    maximum_p, minimum_lower, minimum_delta = 0., 1., 1.
    worst_p_assignment = None
    radius = math.sqrt(2*math.log(2/alpha)/n)
    for assignment in itertools.combinations_with_replacement(range(-4, 5), len(indices)):
        magnitudes = tuple(sorted(abs(v) for v in assignment))
        if magnitudes not in distributions:
            distribution = baseline
            for value in magnitudes:
                updated = Counter()
                for total, count in distribution.items():
                    updated[total+value] += count
                    updated[total-value] += count
                distribution = updated
            distributions[magnitudes] = distribution
        observed = abs(sum(known)+sum(assignment))
        p = sum(count for total, count in distributions[magnitudes].items()
                if abs(total) >= observed)/(2**n)
        delta = (sum(known)+sum(assignment))/(4*n)
        lower = max(-1., delta-radius)
        if p > maximum_p or worst_p_assignment is None:
            maximum_p, worst_p_assignment = p, assignment
        minimum_lower, minimum_delta = min(minimum_lower, lower), min(minimum_delta, delta)
        checked += 1
        multiplicity = math.factorial(len(indices))
        for count in Counter(assignment).values():
            multiplicity //= math.factorial(count)
        covered += multiplicity
    require(covered == 9**len(indices), 'counterfactual sensitivity coverage incomplete')
    return dict(maximum_exact_p=maximum_p, minimum_bounded_mean_lower=minimum_lower,
        minimum_mean_win_score_delta=minimum_delta, worst_p_assignment=list(worst_p_assignment),
        distinct_counterfactual_multisets=checked, ordered_counterfactuals_covered=covered,
        all_scores_support_advantage=maximum_p <= alpha and minimum_lower > 0)


def add_sensitivity(result, registration, *, max_uncertain=4):
    uncertain = registration['repair_retention']['uncertain_seed_clusters']
    require(len(uncertain) == len(set(uncertain))
            and set(uncertain) <= set(registration['seeds']), 'historical sensitivity seed drift')
    result['native_repair_sensitivity'] = dict(uncertain_seed_clusters=uncertain,
        status='INCOMPLETE_NO_INFERENCE', all_scores_support_advantage=False,
        rule='worst exact p and bounded-mean lower limit over all nine scores for EVERY historical seed cluster')
    if not result['inferential_test_allowed']:
        return
    rows = result['contrasts']
    require([row['seed'] for row in rows] == registration['seeds'], 'sensitivity full-roster drift')
    indices = [registration['seeds'].index(seed) for seed in uncertain]
    bounds = worst_case_statistics([row['difference'] for row in rows], indices, max_uncertain=max_uncertain)
    # The repair was diagnosed using historical games. Also show the conditional
    # fresh-source evidence, on the untouched seeds, rather than pretending all
    # 64 seeds were measured under an implementation fixed in advance. This is
    # an additional safeguard, never a substitute for the complete fixed roster.
    from pokezero.mcts_eval.wider_search import exact_signflip_p
    fresh = [row['difference'] for row in rows if row['seed'] not in uncertain]
    require(bool(fresh), 'no untouched fresh-source clusters')
    mean = sum(fresh)/len(fresh)
    lower = max(-1., mean-math.sqrt(2*math.log(40)/len(fresh)))
    p = exact_signflip_p(fresh)
    conditional = dict(seed_clusters=len(fresh), mean_win_score_delta=mean,
        exact_seed_cluster_signflip_p=p, bounded_mean_lower=lower,
        disclosure='conditional on the repair; supplementary evidence on untouched seeds, not a smaller replacement study',
        supports_advantage=p <= .05 and lower > 0)
    result['preregistered_full_roster_advantage'] = result['statistically_supported_advantage']
    result['statistically_supported_advantage'] &= bounds['all_scores_support_advantage'] and conditional['supports_advantage']
    result['native_repair_sensitivity'].update(status='COMPLETE_FIXED_N_ALL_HISTORICAL_SCORES',
        untouched_seed_conditional_evidence=conditional, **bounds)
