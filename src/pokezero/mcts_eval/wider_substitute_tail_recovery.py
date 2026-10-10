"""Explicit bounded-tail transfer, never a new eight-game qualification claim.

Preserve all 32 completed games, the original accepted prefix/statistics/RNG,
the fixed 64/256 roster and all nine historical uncertainty clusters. The only
sampling change is continuing the same rejection stream up to 2048 proposals.
The isolated diagnostic decision-391 contribution is never restored.
"""
import ast
import json
import math
from pathlib import Path

from .wider_native_recovery import (CONTRACT_KEYS, bind_qualification_audit,
    validate_native_qualification, verify_historical)
from .wider_recovery import require
from .wider_search import study_seed
from .wider_substitute_recovery import bind_conditioning_audit, validate_prior_audit
from .wider_trapping_recovery import reference_resume

KIND = 'native-and-reference-substitute-tail-repair'
CANDIDATE = '0bd6cdb498fd84881f3dce6cff271f460f5f0473'
PARENT = '2280d2bacf7fc903f380a1817c59c9aaa405ec53'
QUALIFIED = '2ba73d74cd347afac372877ed6de367badc5f97b'
OPERATIONAL = {'scripts/wider_search_comparison.py',
    'src/pokezero/mcts_eval/wider_substitute_tail_recovery.py'}
PROOFS = {
    'validate_substitute_operational_boundary_r9.json':
        'e409b1a4e9c5b70f4362460fab9f675c9b0e49b02ea09a993128f4d1b5b11e5f',
    'validate_substitute_rejection_tail_probe_r10.json':
        '3b2cfcdb1ad8f394dacc5bdbc9f3f37c09d46e074fc37d2bc0b2d472297aeca5',
    'validate_substitute_tail_parallel_r10.json':
        '0c4e34fbbf9161cb3691d04a30f9f26cd389ced0bf4446e802189c07d5ad06fb',
}


class RemoveTailGuard(ast.NodeTransformer):
    """Only a fail-closed first-resumed-worker proof may differ in run()."""
    def visit_If(self, node):
        self.generic_visit(node)
        if ast.unparse(node.test) == "recovery.get('kind') == 'native-and-reference-substitute-tail-repair'":
            require(len(node.body) == 2 and isinstance(node.body[0], ast.ImportFrom)
                and node.body[0].module == 'pokezero.mcts_eval.wider_substitute_tail_recovery'
                and ast.unparse(node.body[1]) == 'validate_resumed_workers(measured.worker_receipts, probe)',
                'undeclared selector behavior in tail guard')
            return node.orelse
        return node


def bind_transfer_proofs(audit_dir, *, sha):
    """Pinned independent proofs, not self-attested scientific exceptions."""
    audit_dir, records, inputs = Path(audit_dir), {}, {}
    for name, expected in PROOFS.items():
        path = audit_dir / name
        require(sha(path) == expected, 'tail transfer proof changed: ' + name)
        record = json.loads(path.read_text())
        observer = path.with_suffix('.py')
        require(record['observer_sha256'] == sha(observer), 'tail proof observer drift')
        inputs.update({str(path): expected, str(observer): sha(observer)})
        for filename, digest in record['input_hashes'].items():
            require(sha(filename) == digest, 'tail proof input drift: ' + filename)
            require(filename not in inputs or inputs[filename] == digest, 'conflicting proof binding')
            inputs[filename] = digest
        records[name] = record
    operation = records['validate_substitute_operational_boundary_r9.json']
    require(operation['status'] == 'SCIENTIFIC_TREATMENT_IDENTICAL_EXCEPT_FAIL_CLOSED_RESUME_GUARD'
        and operation['qualified_source_commit'] == QUALIFIED
        and operation['qualified_scientific_files'] == 513
        and operation['run_ast_unchanged_after_removing_new_first_draw_guard'] is True,
        'old eight-game qualification/source bridge invalid')
    public = records['validate_substitute_rejection_tail_probe_r10.json']
    require(public['status'] == 'FIXED_200_PUBLIC_DRAWS_AND_ORIGINAL_STREAM_IDENTITIES_VALID'
        and public['source_commit'] == CANDIDATE and public['parent_commit'] == PARENT
        and public['sampler_ast_identical_after_declared_limit_and_receipt_normalization'] is True
        and public['accepted_draws'] == 200 and public['workers'] == 20
        and public['draws_per_worker'] == 10 and public['original_cap'] == 128
        and public['candidate_cap'] == 2048 and public['original_accepted_first_draws_identical'] == 19
        and public['original_failed_draw_accepts_at'] == 139
        and public['game_resumed'] is False and public['complete_qualification'] is False
        and public['strength_inference'] is False, 'all200 exact-root/source-law proof required')
    parallel = records['validate_substitute_tail_parallel_r10.json']
    require(parallel['status'] == 'ALL_PARALLEL_PUBLIC_CERTIFICATES_AND_ACCEPTED_RESUME_VALID'
        and parallel['source_commit'] == CANDIDATE
        and parallel['accepted_prefix_boundaries'] == 13 and parallel['restored_decision_id'] == 390
        and parallel['diagnostic_decision_id'] == 391 and parallel['completed_world_draws'] == 102
        and parallel['workers'] == 20 and parallel['nominal_decision_seconds'] == 10
        and parallel['substitute_rejection_cap'] == 2048 and parallel['pending_rejection_cap'] == 128
        and parallel['game_resumed'] is False and parallel['accepted_prefix_extended'] is False
        and parallel['strength_inference'] is False
        and parallel['diagnostic_statistics_must_not_be_restored'] is True,
        'real parallel decision/public-certificate gate required')
    return records, inputs


def bind_candidate_source(candidate, current, *, repo, git, sha):
    candidate = Path(candidate)
    require(git('rev-parse', 'HEAD', root=candidate) == CANDIDATE
        and not git('status', '--porcelain', root=candidate), 'audited candidate source changed')
    names = set()
    for root in (candidate, repo):
        names.update(git('ls-files', 'src', 'scripts', 'rust/pokezero-search',
            'third_party', 'pyproject.toml', root=root).splitlines())
    inputs = {}
    for name in sorted(names - OPERATIONAL - {'third_party/foul-play'}):
        require((candidate / name).is_file() and (repo / name).is_file()
            and sha(candidate / name) == sha(repo / name), 'unqualified tail source change: ' + name)
        inputs[str(candidate / name)] = sha(candidate / name)
    old, new = (ast.parse((root / 'scripts/wider_search_comparison.py').read_text())
        for root in (candidate, repo))
    runs = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run') for tree in (old, new)]
    require(ast.dump(runs[0]) == ast.dump(RemoveTailGuard().visit(runs[1])),
        'selector/replay/run changed beyond first-worker guard')
    require(current['conditioning_limits'] == dict(substitute=2048, pending=128,
        compatible_template=128), 'explicit rejection limits drift')
    return inputs


def validate_transfer(qualification_path, current, recovery, *, repo, git, sha, verify, bound_rows):
    """Revalidate the old eight games; separately certify cap-only transfer."""
    qualification_path = Path(qualification_path)
    require(str(qualification_path) == recovery['qualification_readout'], 'qualification path drift')
    registered = verify_historical(qualification_path.parent / 'registration.json')
    require(registered['source_commit'] == QUALIFIED, 'wrong inherited qualification source')
    for key in CONTRACT_KEYS:
        if key not in ('seeds', 'registered_games'):
            require(current[key] == registered[key], 'qualified treatment/analysis drift: ' + key)
    require(current['native_package'] == registered['native_package'], 'native package drift')
    old = validate_native_qualification(qualification_path, registered,
        repo=Path(registered['source_root']), git=git, sha=sha, verify=verify, bound_rows=bound_rows)
    inputs = dict(old['semantic_input_hashes'])
    inputs.update(bind_qualification_audit(recovery['qualification_semantic_audit'], qualification_path, sha=sha))
    inputs.update(bind_conditioning_audit(recovery['qualification_conditioning_audit'], qualification_path,
        recovery['qualification_semantic_audit'], sha=sha))
    proofs, bindings = bind_transfer_proofs(recovery['tail_transfer_audit_dir'], sha=sha)
    inputs.update(bindings)
    inputs.update(bind_candidate_source(recovery['audited_repair_source_root'], current,
        repo=repo, git=git, sha=sha))
    require(proofs['validate_substitute_tail_parallel_r10.json']['elapsed_seconds']
        < current['per_decision_safety_seconds'], 'parallel repair gate exceeded safety cap')
    return dict(mode='inherited all8 qualification plus explicit cap-only source-law and exact-root parallel transfer',
        fresh_eight_game_qualification=False, qualified_source_commit=QUALIFIED,
        transferred_candidate_commit=CANDIDATE, substitute_rejection_cap=2048, pending_rejection_cap=128,
        readout_sha256=sha(qualification_path), registration_sha256=sha(qualification_path.parent / 'registration.json'),
        semantic_input_hashes=inputs)


def validate_failed_attempt(row, old, local_files):
    recovery = old['repair_retention']
    failure = row.get('failure_evidence', {})
    errors = failure.get('errors', [])
    require(row['status'] == 'REFUSED' and row['signed_outcome'] is None
        and row['identity'] == recovery['resume']['identity'] and row['arm'] == 'paper_reference'
        and row['retained_prefix_steps'] == recovery['resume']['prefix_steps']
        and not local_files and failure['decision_id'] == 391
        and failure['battle_id'] == 'wider-search:' + row['identity'] and failure['receipts'] == []
        and len(errors) == 1 and errors[0][1:4] == [3, 391, 'trajectory_batch']
        and errors[0][4] == 'ReferenceRefusal: Substitute public-history conditioning exhausted its explicit rejection cap',
        'only exact unscored cap refusal with no accepted new work may transfer')
    partial = errors[0][-1]['partial_batch_evidence']['draws']
    require(len(partial) == 1 and partial[0]['status'] == 'REFUSED' and partial[0]['ordinal'] == 0,
        'failed diagnostic contribution cannot become accepted work')


def charge_failed_attempt(resume, directory, terminal, wall_seconds):
    """Keep original allowance and charge the failed active attempt as well."""
    born = getattr(Path(directory).stat(), 'st_birthtime', None)
    stopped = getattr(Path(terminal).stat(), 'st_birthtime', None)
    require(born is not None and stopped is not None and stopped >= born,
        'failed active wall-clock envelope unavailable')
    active = stopped - born + 1.
    elapsed = resume['elapsed_before_resume'] + active
    require(math.isfinite(elapsed) and 0 < elapsed < wall_seconds, 'original game wall cap exhausted')
    result = dict(resume, elapsed_before_resume=elapsed)
    result['wall_clock_envelope'] = dict(resume['wall_clock_envelope'], additional_attempts=[dict(
        directory=str(directory), terminal=str(terminal), directory_birth=born,
        refused_result_birth=stopped, margin_seconds=1., charged_seconds=active)])
    return result


def prepare(previous, audit_dir, current, *, repo, git, sha, verify, bound_rows,
            step_files, restore_reference_checkpoint):
    previous, audit_dir = Path(previous), Path(audit_dir)
    registered = previous / 'registration.json'
    old = verify_historical(registered)
    retained = old['repair_retention']
    require(old['phase'] == 'FIXED_64_SEED_CONFIRMATION' and old['registered_games'] == 256
        and old['seeds'] == [study_seed(i) for i in range(64)] and old['source_commit'] == PARENT
        and retained['kind'] == 'native-and-reference-substitute-repair'
        and len(retained['retained_complete']) == 32
        and len(set(retained['uncertain_seed_clusters'])) == 9, 'fixed R9 historical inventory required')
    for key in CONTRACT_KEYS:
        require(current[key] == old[key], 'tail repair changes original contract: ' + key)
    public_probe_path = audit_dir / 'probe_substitute_rejection_tail_200_draws_r10.json'
    public_probe = json.loads(public_probe_path.read_text())
    proof_path = audit_dir / 'validate_substitute_rejection_tail_probe_r10.json'
    require(sha(proof_path) == PROOFS[proof_path.name], 'candidate public/source proof drift')
    public_proof = json.loads(proof_path.read_text())
    candidate_paths = [p for p in public_proof['input_hashes']
        if p.endswith('/src/pokezero/mcts_eval/paper_reference_substitute.py')
        and 'rejection-tail-repair' in p]
    require(len(candidate_paths) == 1, 'candidate source binding missing or ambiguous')
    candidate_root = Path(candidate_paths[0]).parents[3]
    result = dict(retained, kind=KIND, audited_repair_source_root=str(candidate_root),
        audited_repair_source_commit=CANDIDATE, tail_transfer_audit_dir=str(audit_dir),
        previous_failed_registration=str(registered), exact_failed_draw_certificate=str(public_probe_path),
        fresh_eight_game_qualification=False,
        disclosure=retained['disclosure'] + '; explicit same-stream Substitute cap128->2048 transfer from '
            'all8 prior qualification games, all200 public draws and real parallel gate; no fresh all8 claim; '
            'pending/template caps remain128; diagnostic decision391 not restored')
    q = validate_transfer(Path(retained['qualification_readout']), current, result,
        repo=repo, git=git, sha=sha, verify=verify, bound_rows=bound_rows)
    inputs = dict(q['semantic_input_hashes'])
    inputs.update({str(registered): sha(registered), str(public_probe_path): sha(public_probe_path)})
    for bindings in (old['input_hashes'], old['retained_input_hashes']):
        for filename, digest in bindings.items():
            require(sha(filename) == digest, 'historical source/input drift: ' + filename)
            inputs[filename] = digest
    rows = bound_rows(previous, old)
    require(len(rows) == 33 and len({r['identity'] for r in rows}) == 33
        and sum(r['status'] == 'COMPLETE' for r in rows) == 32, 'all32 complete plus one refusal required')
    refusal = [r for r in rows if r['status'] != 'COMPLETE']
    require(len(refusal) == 1, 'exactly one refused game required')
    row = refusal[0]
    path = previous / (row['identity'] + '.json')
    validate_failed_attempt(row, old, step_files(previous / row['identity']))
    inputs[str(path)] = sha(path)
    original = verify_historical(retained['original_registration'])
    original_path = Path(retained['resume']['result_path'])
    original_row = json.loads(original_path.read_text())
    steps = [Path(p) for _, p in sorted(retained['resume']['prefix_steps'].items())]
    require({p.name: sha(p) for p in steps} == row['step_hashes'] == original_row['step_hashes'],
        'original accepted prefix/checkpoint drift')
    fresh_resume = reference_resume(original_row, original_path, steps, original,
        dict(identity=row['identity'], result_sha256=sha(original_path), boundary=13,
            hidden_ordinal=public_probe['worker_ordinals'][0]),
        restore_reference_checkpoint=restore_reference_checkpoint, sha=sha)
    require(fresh_resume == retained['resume'] and fresh_resume['decision_id'] == 390
        and fresh_resume['worker_ordinals'] == public_probe['worker_ordinals'],
        'accepted statistics/ordinals/original wall allowance changed')
    historical = json.loads(Path(retained['historical_semantic_audit']).read_text())
    records = validate_prior_audit(historical, original, sha(retained['original_registration']))
    require(set(records) == set(retained['retained_complete']), 'all32 original replay proofs required')
    for identity, binding in retained['retained_complete'].items():
        require(binding['sha256'] == sha(binding['path']) == records[identity]['result_sha256'],
            'retained terminal/replay drift')
    result['resume'] = charge_failed_attempt(fresh_resume, previous / row['identity'], path,
        current['per_game_wall_seconds'])
    result['input_hashes'] = inputs
    return result


def validate_resumed_workers(receipts, probe):
    expected = {(r['worker'], r['ordinal']): r['draws'][0] for r in probe['workers']}
    seen = set()
    for contribution in receipts:
        worker = contribution['worker']
        draws = contribution['evidence']['draws']
        if worker in seen or not draws:
            continue
        require(type(worker) is int and 0 <= worker < 20, 'invalid resumed worker')
        seen.add(worker)
        actual = draws[0]
        original = expected[worker, probe['worker_ordinals'][worker]]
        require(actual['status'] == 'ROOT_VALIDATED' and actual['released'] is True
            and all(actual[k] == original[k] for k in ('packed_team_sha256',
                'materialization_seed', 'substitute_policy_conditioning')),
            'resumed first worker draw differs from original stream proof')
