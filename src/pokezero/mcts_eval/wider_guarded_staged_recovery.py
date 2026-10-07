"""Fresh qualified staged repair with lossless historical-prefix retention.

No diagnostic statistics or attempted-world ordinals are restored. The 32
completed games and nine outcome-blind sensitivity clusters remain historical,
mixed-source evidence in the unchanged 64-seed / 256-game study.
"""
import ast
import json
import math
from pathlib import Path

from .followthrough import continuation_seed
from .wider_native_recovery import CONTRACT_KEYS, verify_historical
from .wider_recovery import combined_steps, read_step, require
from .wider_search import ARMS, SEATS, analyze, game_identity, study_seed
from .wider_substitute_recovery import VALID, validate_prior_audit

KIND = 'native-and-reference-guarded-staged-repair'
QUALIFIED = 'd3210f9c1d61efd33fbd3cf91acadb07cea62a95'
PARENT = '5e1e67c0ca057cbcbafc0dcc1fe64432cb597e1a'
IDENTITY = 'seed-474486314-p1-paper_reference'
SEMANTIC_OBSERVER = '07abc045ba0bc26dc35332b90c1aa87e780ffa99cf9d70a4b04c335136075739'
PUBLIC_OBSERVER = '4c7ccc12e4a2d97878603351b45c89728aa747fe9bdd9c2a1aad177bccb0d682'
PROOFS = {
    'capture_substitute_tail_failure_r10.json': '9ee6b697768248fb43a74bc360b0b0cd7535f4349dfa9e39f9c60b64104355bf',
    'audit_public_staged_dependencies_r28.json': '314d399053fafa6836734a54dcf09d9b4319b09229f3ba23a5054bff354e52b9',
    'probe_guarded_staged_original20_r29.json': '1b0ae1cba03826be3cc08a4dc3358879e63236a26e3bad181d6012dc57c79303',
    'validate_guarded_staged_original20_r30.json': '831dfa33094eaad3145d2d3cb95df0f4f42c819fc3157b354ce851e7a46cf643',
    'test_guarded_qualification_observers_r33.json': '609b0c62d90f73d658d593dd7ffbf5b86d13bdd564ded29c03fb9b8a76f8f802',
    'capture_guarded_resume_failure_r54.json': '96d9497e2ef9ba0f02d6a61d7b3782710c679b0ade80edb62fcff7d50b17912b',
    'probe_guarded_receipt_representation_r55.json': '1471ee8e49d860a49eca150f33ddc761440f5b495839a5ff08b2fbb588d56368',
}


class RemoveStagedGuard(ast.NodeTransformer):
    """Permit only an explicit fail-closed first-draw comparison in run()."""
    def visit_If(self, node):
        self.generic_visit(node)
        if ast.unparse(node.test) == "recovery.get('kind') == 'native-and-reference-guarded-staged-repair'":
            require(len(node.body) == 2 and isinstance(node.body[0], ast.ImportFrom)
                and node.body[0].module == 'pokezero.mcts_eval.wider_guarded_staged_recovery'
                and ast.unparse(node.body[1]) == 'validate_resumed_workers(measured.worker_receipts, probe)',
                'undeclared behavior in staged resume guard')
            return node.orelse
        return node


def bind_source(registered, current, *, repo, git, sha):
    origin = Path(registered['source_root'])
    require(registered['source_commit'] == QUALIFIED
        and git('rev-parse', 'HEAD', root=origin) == QUALIFIED
        and not git('status', '--porcelain', root=origin), 'qualified source is not pinned and clean')
    allowed = {'scripts/wider_search_comparison.py',
        'src/pokezero/mcts_eval/wider_guarded_staged_recovery.py', 'third_party/foul-play'}
    names = set()
    for root in (origin, repo):
        names.update(git('ls-files', 'src', 'scripts', 'rust/pokezero-search',
            'third_party', 'pyproject.toml', root=root).splitlines())
    inputs = {}
    for name in sorted(names - allowed):
        require((origin / name).is_file() and (repo / name).is_file()
            and sha(origin / name) == sha(repo / name), 'unqualified scientific source change: ' + name)
        inputs[str(origin / name)] = sha(origin / name)
    trees = [ast.parse((root / 'scripts/wider_search_comparison.py').read_text()) for root in (origin, repo)]
    functions = [{n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)} for tree in trees]
    require(functions[0].keys() == functions[1].keys(), 'undeclared driver helper')
    operational = {'verify', 'register', 'validate_qualification', 'main', 'readout'}
    for name in functions[0].keys() - operational:
        candidate = functions[1][name]
        if name == 'run':
            candidate = RemoveStagedGuard().visit(candidate)
        require(ast.dump(functions[0][name]) == ast.dump(candidate),
            'qualified driver changed beyond registry/guard: ' + name)
    for key in CONTRACT_KEYS:
        if key not in ('seeds', 'registered_games'):
            require(current[key] == registered[key], 'fresh qualification treatment drift: ' + key)
    require(current['native_package'] == registered['native_package'], 'qualified native package drift')
    return inputs


def bind_proofs(audit_dir, *, sha):
    records, inputs = {}, {}
    for name, expected in PROOFS.items():
        path = Path(audit_dir) / name
        require(sha(path) == expected, 'independent repair proof drift: ' + name)
        record = json.loads(path.read_text())
        inputs[str(path)] = expected
        inputs[str(path.with_suffix('.py'))] = sha(path.with_suffix('.py'))
        for filename, digest in record['input_hashes'].items():
            require(sha(filename) == digest, 'proof input drift: ' + filename)
            require(filename not in inputs or inputs[filename] == digest, 'conflicting proof input')
            inputs[filename] = digest
        records[name] = record
        require(record.get('observer_sha256', inputs[str(path.with_suffix('.py'))])
            == inputs[str(path.with_suffix('.py'))], 'proof observer drift: ' + name)
    replay = records['validate_guarded_staged_original20_r30.json']
    require(replay['status'] == 'ALL_ACCEPTED_ORIGINAL20_WITNESSES_REPLAYED_WITH_FULL_POLICY_RNG_AND_NEW_ROOT_BACKUPS'
        and replay['source_commit'] == QUALIFIED and len(replay['verified']) == 50
        and replay['checked_cancellations'] == 20 and replay['negative_controls'] == 3
        and replay['study_resumed'] is False and replay['strength_inference'] is False,
        'full original20 independent replay required')
    representation = records['probe_guarded_receipt_representation_r55.json']
    comparisons = representation['first_draw_comparison']
    require(representation['source_commit'] == '07e199f39d56d9620b8149fa0059958c4b0c7b29'
        and representation['raw_guard_error'] == 'resumed first accepted draw differs from exact original stream proof'
        and representation['canonical_guard_error'] is None
        and len(comparisons) == 20 and {r['worker'] for r in comparisons} == set(range(20))
        and all(r['status'] == 'ROOT_VALIDATED' and r['accepted'] is True
            and r['packed_team_equal'] is True and r['materialization_seed_equal'] is True
            and r['canonical_certificate_equal'] is True and r['raw_certificate_equal'] is False
            and {(d['path'], d['runtime_type'], d['durable_type']) for d in r['type_differences']}
                == {('/anchor_rng_state', 'tuple', 'list'), ('/anchor_rng_state/1', 'tuple', 'list')}
            for r in comparisons)
        and representation['action_played'] is False and representation['game_resumed'] is False
        and representation['diagnostic_checkpoint_or_rng_restored'] is False,
        'all20 representation-only diagnostic required; no scientific witness relaxation')
    return records, inputs


def bind_fresh_audits(semantic_path, public_path, qualification_path, *, sha):
    """No partial first-cell report or old default-off qualification may pass."""
    semantic_path, public_path, qualification_path = map(Path,
        (semantic_path, public_path, qualification_path))
    registration_path = qualification_path.parent / 'registration.json'
    m = json.loads(registration_path.read_text())
    expected = {game_identity(seed, seat, arm) for seed in m['seeds'] for seat in SEATS for arm in ARMS}
    require(m['phase'] == 'QUALIFICATION_NOT_STRENGTH' and m['source_commit'] == QUALIFIED
        and m['registered_games'] == 8 and len(expected) == 8
        and m['seeds'] == [study_seed(i, qualification=True) for i in range(2)]
        and m['reference']['staged_substitute_conditioning'] is True, 'fresh disjoint staged all8 required')
    inputs = {str(registration_path): sha(registration_path), str(qualification_path): sha(qualification_path)}
    for path, observer_name, observer_hash, valid in (
        (semantic_path, 'validate_guarded_qualification_games_r31.py', SEMANTIC_OBSERVER, VALID),
        (public_path, 'validate_guarded_qualification_conditioning_r32.py', PUBLIC_OBSERVER,
            'PUBLIC_AND_SUBSTITUTE_CERTIFICATES_VALID')):
        audit = json.loads(path.read_text())
        observer = path.with_name(observer_name)
        require(sha(observer) == observer_hash == audit['observer_sha256']
            and audit['source_commit'] == QUALIFIED and audit['registration_sha256'] == sha(registration_path)
            and audit['audited_complete_games'] == len(audit['games']) == 8
            and audit['complete_roster_valid'] is True and audit['strength_inference'] is False
            and {r['identity'] for r in audit['games']} == expected
            and all(r['status'] == valid for r in audit['games']), 'complete independent all8 audit required')
        if path == public_path:
            require(audit['full_replay_audit_sha256'] == sha(semantic_path), 'public audit semantic binding drift')
        inputs.update({str(path): sha(path), str(observer): observer_hash})
        for row in audit['games']:
            terminal = qualification_path.parent / (row['identity'] + '.json')
            cell = json.loads(terminal.read_text())
            files = sorted((qualification_path.parent / row['identity']).iterdir())
            require(row['result_sha256'] == sha(terminal) and cell['status'] == 'COMPLETE'
                and cell['registration_sha256'] == sha(registration_path)
                and files and all(p.is_file() for p in files)
                and {p.name: sha(p) for p in files} == cell['step_hashes'], 'all8 audited decision evidence drift')
            inputs.update({str(p): sha(p) for p in (terminal, *files)})
        for filename, digest in audit.get('input_hashes', {}).items():
            require(sha(filename) == digest, 'qualification audit input drift: ' + filename)
            inputs[filename] = digest
    return inputs


def validate_qualification(path, current, recovery, *, repo, git, sha, verify, bound_rows):
    path = Path(path)
    require(path.name == 'READOUT.json' and str(path) == recovery['qualification_readout'],
        'fresh canonical qualification path required')
    registered = verify_historical(path.parent / 'registration.json')
    inputs = bind_source(registered, current, repo=repo, git=git, sha=sha)
    rows = bound_rows(path.parent, registered)
    result = analyze(registered['seeds'], rows)
    result.update(source_commit=registered['source_commit'], input_hashes=registered['input_hashes'],
        phase=registered['phase'], statistically_supported_advantage=False, inferential_test_allowed=False,
        status='QUALIFICATION_COMPLETE_NOT_STRENGTH' if not result['missing_seed_clusters'] else 'QUALIFICATION_INCOMPLETE')
    require(result == json.loads(path.read_text()) and result['status'] == 'QUALIFICATION_COMPLETE_NOT_STRENGTH'
        and len(rows) == 8 and all(r['status'] == 'COMPLETE' for r in rows), 'fresh all8 durable qualification required')
    inputs.update(bind_fresh_audits(recovery['qualification_semantic_audit'],
        recovery['qualification_conditioning_audit'], path, sha=sha))
    _, proofs = bind_proofs(recovery['staged_audit_dir'], sha=sha)
    inputs.update(proofs)
    return dict(mode='fresh all8 staged qualification with independent semantic/public audits; operational adapter only',
        fresh_eight_game_qualification=True, qualified_source_commit=QUALIFIED,
        readout_sha256=sha(path), registration_sha256=sha(path.parent / 'registration.json'),
        semantic_input_hashes=inputs)


def resume_capture(capture, old, previous, *, sha, restore_reference_checkpoint):
    """Derive only accepted checkpoint392, not the unscored diagnostic393."""
    require(capture['status'] == 'TRUNCATED_FAILURE_FIELDS_AND_ACCEPTED_PREFIX_CAPTURED_NOT_A_TERMINAL'
        and capture['source_commit'] == PARENT and capture['identity'] == IDENTITY
        and capture['accepted_prefix_boundaries'] == capture['next_boundary'] == 15
        and capture['last_accepted_decision_id'] == 392 and capture['failed_decision_id'] == 393
        and capture['failure_receipts_truncated_and_unrecoverable'] is True
        and capture['failed_partial_work_restored'] is False
        and capture['strength_inference'] is False, 'exact truncated historical capture required')
    names = [f'boundary-{i:03d}.json.gz' for i in range(15)]
    require(sorted(capture['accepted_prefix_steps']) == names, 'accepted prefix gap/extra boundary')
    ordinals, selections, elapsed, last = [0] * 20, 0, 0., None
    for boundary, name in enumerate(names):
        path = Path(capture['accepted_prefix_steps'][name])
        require(path.name == name and sha(path) == capture['accepted_prefix_hashes'][name], 'accepted prefix hash drift')
        step = read_step(path)
        require(step['boundary'] == boundary
            and step['chance_seed'] == continuation_seed('wider-search:474486314:p1', 0, boundary, 'chance'),
            'accepted prefix boundary/chance domain drift')
        if 'p1' not in step['actions']:
            continue
        own = step['evidence']['p1']
        require(own['selector'] == 'paper_reference'
            and own['seed'] == continuation_seed('wider-search:474486314:p1', 0, boundary, 'search'),
            'accepted prefix selector/search domain drift')
        selections += 1
        elapsed += own['elapsed_seconds']
        last = own['search_evidence']
        for receipt in last['worker_receipts']:
            worker, draws = receipt['worker'], receipt['evidence']['draws']
            require(type(worker) is int and 0 <= worker < 20
                and len(draws) == receipt['batch']['world_draws']
                and all(d['status'] == 'ROOT_VALIDATED' and d['released'] is True for d in draws),
                'unaccepted work in historical prefix')
            ordinals[worker] += len(draws)
    checkpoint = restore_reference_checkpoint(last['statistics_checkpoint'])
    require(last['decision_id'] == 392 and checkpoint.battle_id == 'wider-search:' + IDENTITY
        and ordinals == capture['worker_ordinals']
        and capture['checkpoint_step'] == capture['accepted_prefix_steps'][names[-1]]
        and capture['checkpoint_sha256'] == sha(capture['checkpoint_step']), 'accepted checkpoint/ordinals drift')
    inherited = old['repair_retention']['resume']
    for name, path in inherited['prefix_steps'].items():
        require(capture['accepted_prefix_steps'][name] == path, 'original accepted prefix changed')
    directory, terminal = Path(previous) / IDENTITY, Path(previous) / (IDENTITY + '.json')
    born, stopped = (getattr(p.stat(), 'st_birthtime', None) for p in (directory, terminal))
    require(born is not None and stopped is not None and math.isfinite(born)
        and math.isfinite(stopped) and stopped >= born, 'historical active wall envelope unavailable')
    active = stopped - born + 1.
    charged = inherited['elapsed_before_resume'] + active
    require(math.isfinite(charged) and elapsed <= charged < old['per_game_wall_seconds'],
        'unchanged original game wall allowance exhausted')
    wall = dict(inherited['wall_clock_envelope'])
    wall['additional_attempts'] = [*wall.get('additional_attempts', []), dict(directory=str(directory),
        terminal=str(terminal), directory_birth=born, refused_result_birth=stopped,
        margin_seconds=1., charged_seconds=active, terminal_is_truncated_historical_evidence=True)]
    return dict(inherited, result_path=str(terminal), canonical_terminal_is_truncated=True,
        accepted_base_refusal_result_path=inherited['result_path'], checkpoint_step=capture['checkpoint_step'],
        prefix_steps=capture['accepted_prefix_steps'], start_boundary=15, prior_selections=selections,
        worker_ordinals=ordinals, decision_id=392, elapsed_before_resume=charged,
        wall_clock_envelope=wall,
        recovery='all15 accepted prefixes/checkpoint392 retained; truncated refusal393 not scored or restored')


def prepare(previous, audit_dir, qualification_path, semantic_path, public_path, current, *,
            repo, git, sha, verify, bound_rows, step_files, restore_reference_checkpoint):
    previous, audit_dir = Path(previous), Path(audit_dir)
    registered = previous / 'registration.json'
    old = verify_historical(registered)
    retained = old['repair_retention']
    require(old['source_commit'] == PARENT and old['phase'] == 'FIXED_64_SEED_CONFIRMATION'
        and old['seeds'] == [study_seed(i) for i in range(64)] and old['registered_games'] == 256
        and retained['kind'] == 'native-and-reference-substitute-tail-repair'
        and len(retained['retained_complete']) == 32 and len(set(retained['uncertain_seed_clusters'])) == 9,
        'exact historical full64/256 inventory required')
    for key in CONTRACT_KEYS:
        if key != 'reference':
            require(current[key] == old[key], 'original study contract drift: ' + key)
    require(all(current['reference'].get(k) == v for k, v in old['reference'].items())
        and current['reference']['staged_substitute_conditioning'] is True,
        'original reference settings must remain with disclosed staged/deadline extension')
    records, inputs = bind_proofs(audit_dir, sha=sha)
    capture = records['capture_substitute_tail_failure_r10.json']
    require(capture['input_hashes'][str(registered)] == sha(registered), 'captured registration drift')
    local = step_files(previous / IDENTITY)
    expected_local = {str(Path(p)) for p in capture['accepted_prefix_steps'].values()
        if Path(p).parent == previous / IDENTITY}
    require({str(p) for p in local} == expected_local and len(local) == 2,
        'historical tail acquired missing/extra accepted evidence')
    probe_path = audit_dir / 'probe_guarded_staged_original20_r29.json'
    probe = records[probe_path.name]
    require(probe['source_commit'] == QUALIFIED and probe['worker_ordinals'] == capture['worker_ordinals']
        and probe['restored_decision_id'] == 392 and probe['boundary'] == 15
        and probe['game_resumed'] is False and probe['action_played'] is False
        and probe['accepted_prefix_extended'] is False, 'original20 exact-root diagnostic binding required')
    require(set(retained['uncertain_seed_clusters']) ==
        {binding_seed for identity in retained['retained_complete'] for binding_seed in [int(identity.split('-')[1])]}
        | {474486314}, 'historical uncertainty inventory must be outcome-blind all9 touched seeds')
    for bindings in (old['input_hashes'], old['retained_input_hashes']):
        for filename, digest in bindings.items():
            require(sha(filename) == digest, 'historical evidence/source drift: ' + filename)
            inputs[filename] = digest
    original = verify_historical(retained['original_registration'])
    audit = json.loads(Path(retained['historical_semantic_audit']).read_text())
    audited = validate_prior_audit(audit, original, sha(retained['original_registration']))
    require(set(audited) == set(retained['retained_complete']), 'retain every audited complete game')
    for identity, binding in retained['retained_complete'].items():
        path = Path(binding['path'])
        m = json.loads(Path(binding['registration']).read_text())
        cell = json.loads(path.read_text())
        steps = combined_steps(path.parent, cell, m, sha=sha, step_files=step_files)
        require(cell['identity'] == identity and cell['status'] == 'COMPLETE'
            and binding['sha256'] == sha(path) == audited[identity]['result_sha256']
            and {p.name: sha(p) for p in steps} == cell['step_hashes'], 'retained complete semantic/step drift')
        inputs.update({str(p): sha(p) for p in (path, Path(binding['registration']), *steps)})
    result = dict(retained, kind=KIND, qualification_readout=str(qualification_path),
        qualification_semantic_audit=str(semantic_path), qualification_conditioning_audit=str(public_path),
        qualified_source_commit=QUALIFIED, audited_repair_source_commit=QUALIFIED,
        audited_repair_source_root=str(json.loads((Path(qualification_path).parent / 'registration.json').read_text())['source_root']),
        staged_audit_dir=str(audit_dir), exact_failed_draw_certificate=str(probe_path),
        previous_failed_registration=str(registered), fresh_eight_game_qualification=True,
        historical_truncated_capture=str(audit_dir / 'capture_substitute_tail_failure_r10.json'),
        disclosure=retained['disclosure'] + '; fresh disjoint all8 staged/deadline technical qualification; '
            'all32 historical COMPLETE games and15 accepted boundaries/decision392 retained; '
            'mixed source, all9 outcome-blind sensitivity clusters; diagnostic393 Q/N/RNG never restored; '
            'unsupported public programs retain joint conditioning; deadline sample-selection effects not proven away')
    q = validate_qualification(qualification_path, current, result, repo=repo, git=git, sha=sha,
        verify=verify, bound_rows=bound_rows)
    inputs.update(q['semantic_input_hashes'])
    inputs[str(registered)] = sha(registered)
    result['resume'] = resume_capture(capture, old, previous, sha=sha,
        restore_reference_checkpoint=restore_reference_checkpoint)
    failed = records['capture_guarded_resume_failure_r54.json']
    result['resume'] = charge_unplayed_guard_attempt(result['resume'], failed, sha=sha,
        wall_seconds=old['per_game_wall_seconds'])
    result['guard_representation_repair'] = dict(failed_capture=str(audit_dir / 'capture_guarded_resume_failure_r54.json'),
        diagnostic=str(audit_dir / 'probe_guarded_receipt_representation_r55.json'),
        algorithm_unchanged=True, only_normalized_fields=['anchor_rng_state outer tuple', 'anchor_rng_state state-vector tuple'])
    result['disclosure'] += '; unplayed operational guard refusal retained and wall charged; '
    result['disclosure'] += 'only RNG-state tuple/JSON-array representation normalized, every scientific field still compared'
    result['input_hashes'] = inputs
    return result


def charge_unplayed_guard_attempt(resume, capture, *, sha, wall_seconds):
    """An unplayed comparison failure cannot reset the original game allowance."""
    require(capture['status'] == 'UNPLAYED_FIRST_DECISION_GUARD_REFUSAL_CAPTURED_ALL32_AND_ACCEPTED15_PRESERVED'
        and capture['source_commit'] == '07e199f39d56d9620b8149fa0059958c4b0c7b29'
        and capture['identity'] == IDENTITY and capture['completed_games_retained'] == 32
        and capture['accepted_prefix_boundaries'] == 15 and capture['accepted_decision_id'] == 392
        and capture['failed_decision_id'] == 393 and capture['new_action_played'] is False
        and capture['new_boundary_accepted'] is False and capture['failed_statistics_or_rng_restored'] is False
        and capture['strength_inference'] is False
        and capture['original_worker_ordinals'] == resume['worker_ordinals']
        and capture['elapsed_before_failed_attempt'] == resume['elapsed_before_resume'],
        'exact unplayed guard refusal capture required')
    registration, terminal = Path(capture['registration']), Path(capture['result_path'])
    require(sha(registration) == capture['registration_sha256']
        and sha(terminal) == capture['result_sha256'], 'failed guard terminal/registration binding drift')
    directory = terminal.parent / IDENTITY
    envelope = capture['wall_clock_envelope']
    born, stopped = directory.stat().st_birthtime, terminal.stat().st_mtime
    charge = stopped - born + 1.
    require(born == envelope['start_birthtime'] and stopped == envelope['terminal_mtime']
        and envelope['slack_seconds'] == 1. and charge == envelope['charged_seconds']
        and math.isfinite(charge) and 0 < charge < 120
        and not list(directory.iterdir()), 'unplayed attempt envelope/evidence inventory changed')
    charged = resume['elapsed_before_resume'] + charge
    require(math.isfinite(charged) and charged < wall_seconds, 'original game wall allowance exhausted')
    wall = dict(resume['wall_clock_envelope'])
    wall['additional_attempts'] = [*wall.get('additional_attempts', []),
        dict(directory=str(directory), terminal=str(terminal), directory_birth=born, terminal_mtime=stopped,
            margin_seconds=1., charged_seconds=charge, new_boundary_accepted=False,
            reason='unplayed tuple-versus-JSON-array guard refusal; no failed RNG/statistics restored')]
    return dict(resume, elapsed_before_resume=charged, wall_clock_envelope=wall)


def comparable_certificate(certificate):
    """Normalize only the documented Random.getstate tuple/JSON array boundary."""
    if 'anchor_rng_state' not in certificate:
        return certificate
    state = certificate['anchor_rng_state']
    require(type(state) in (tuple, list) and len(state) == 3
        and type(state[0]) is int and state[0] == 3
        and type(state[1]) in (tuple, list) and len(state[1]) == 625
        and all(type(v) is int and 0 <= v <= 0xffffffff for v in state[1][:-1])
        and type(state[1][-1]) is int and 0 <= state[1][-1] <= 624
        and (state[2] is None or type(state[2]) is float and math.isfinite(state[2])),
        'invalid anchor RNG-state receipt; normalization cannot repair changed values')
    return dict(certificate, anchor_rng_state=[state[0], list(state[1]), state[2]])


def validate_resumed_workers(receipts, probe):
    """Bind first accepted original draw; never promote clock cancellation."""
    expected = {}
    for receipt in probe['measurement']['worker_receipts']:
        worker = receipt['worker']
        if worker not in expected and receipt['evidence']['draws']:
            expected[worker] = receipt['evidence']['draws'][0]
    require(set(expected) == set(range(20)) and all(d['status'] == 'ROOT_VALIDATED'
        and d['ordinal'] == 0 for d in expected.values()), 'original20 first-draw proof incomplete')
    seen = set()
    for receipt in receipts:
        worker, draws = receipt['worker'], receipt['evidence']['draws']
        if worker in seen or not draws:
            continue
        require(type(worker) is int and 0 <= worker < 20, 'invalid resumed worker identity')
        seen.add(worker)
        actual = draws[0]
        require(actual['ordinal'] == 0, 'first resumed attempt ordinal drift')
        if actual['status'] == 'DEADLINE_CANCELLED':
            diagnostic = actual.get('sampling_diagnostic', {})
            checked, deadline = diagnostic.get('checked_at'), diagnostic.get('deadline_at')
            require(diagnostic.get('schema') == 'pokezero.world-sampling-deadline.v1'
                and type(checked) in (int, float) and type(deadline) in (int, float)
                and math.isfinite(checked) and math.isfinite(deadline) and checked >= deadline
                and diagnostic.get('accepted_world') is False and diagnostic.get('backed_up') is False,
                'first cancellation is not accepted qualified work')
            continue
        reference = expected[worker]
        matching = actual['status'] == 'ROOT_VALIDATED' and actual['released'] is True
        matching = matching and all(actual[k] == reference[k] for k in ('packed_team_sha256', 'materialization_seed'))
        matching = matching and comparable_certificate(actual['substitute_policy_conditioning']) \
            == comparable_certificate(reference['substitute_policy_conditioning'])
        if not matching:
            error = RuntimeError('resumed first accepted draw differs from exact original stream proof')
            error.evidence = dict(worker=worker, actual_first_draw=actual, expected_first_draw=reference,
                failed_statistics_or_rng_restored=False)
            raise error
