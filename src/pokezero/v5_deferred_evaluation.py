"""Durable CPU panel dispatch; no launch, model or game runs on import.

The GPU owner writes one immutable request and continues training. An external
CPU Job explicitly executes one original panel attempt. That Job never writes
the training journal. The controller file may run outside the frozen panel
source tree, importing that original tree's evaluator and scorer unchanged.
"""
import argparse
from dataclasses import asdict,dataclass,replace
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import stat
import time

from pokezero import v5_run as run, v5_evaluation as evaluation

SCHEMA='pokezero.v5.cpu_panel_dispatch.v1'
ATTEMPT='pokezero.v5.cpu_panel_attempt.v1'
MILESTONE = 'pokezero.v5.milestone_dispatch.v1'
STOP = 'pokezero.s0.route_stop.v1'
STOP_KEY_ENV = 'PZ_S0_ROUTE_STOP_KEY'
ASSAY_PROTOCOL = 'pokezero.v5.deferred_assays.v1'
ASSAY_DEVICES = {'external_fp_greedy': 'cpu', 'matched_h2h_showdown': 'cpu',
                 'matched_h2h_engine': 'gpu', 'calibration': 'cpu'}


def _small_document(path):
    """Read a bounded regular file once. An absent signal is normal."""
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 16384:
            raise ValueError('unexpected milestone control file')
        data = stream.read(16385)
    if len(data) > 16384:
        raise ValueError('milestone control file exceeds its bound')
    return json.loads(data), hashlib.sha256(data).hexdigest()


def _signature(payload, key):
    return hmac.new(key, b'pokezero.s0.route_stop.v1\0' + run._json(payload).encode(),
                    hashlib.sha256).hexdigest()


def _stop_key():
    value = os.environ.get(STOP_KEY_ENV, '')
    if not value:
        return None
    key = bytes.fromhex(value)
    if len(key) < 32:
        raise ValueError('S0 route stop signing key must contain at least 32 bytes')
    return key


class MilestoneDispatch:
    """The training-side interface. Call only after the round commits."""

    def __init__(self, root, policy, milestones, declaration_sha256, *, stop_key=None,
                 report_observer_error=None):
        self.root = Path(root).resolve()
        self.policy = policy
        self.milestones = milestones
        self.declaration_sha256 = declaration_sha256
        self.stop_key = None
        self.report_observer_error = report_observer_error
        self.last_error = None
        self.pending = {}
        if policy.route_declaration_sha256:
            self.stop_key = _stop_key() if stop_key is None else stop_key
            if type(self.stop_key) is not bytes or len(self.stop_key) < 32:
                raise ValueError('S0 route requires a configured stop verification key')

    def path(self, games):
        return self.root / 'artifacts' / 'cpu-panels' / str(games) / 'request.json'

    def request_missing(self, snapshot):
        return not self.path(self.milestones[snapshot.iteration]).exists()

    def publish_committed(self, snapshot, *, report_error, full_checkpoint=None):
        games = self.milestones.get(snapshot.iteration)
        # Retry at most one older publication per committed round. A failed
        # publication never turns into a failed evaluation or stalls training.
        if self.pending:
            older = next(iter(self.pending))
            if older != games:
                old_snapshot, old_checkpoint = self.pending.pop(older)
                self._publish(older, old_snapshot, old_checkpoint, report_error)
        if games is not None:
            self.pending.pop(games, None)
            return self._publish(games, snapshot, full_checkpoint, report_error)

    def _publish(self, games, snapshot, full_checkpoint, report_error):
        request = run.EvaluationRequest(games, self.declaration_sha256, self.policy.protocol_sha256,
            snapshot, self.root, self.root / snapshot.checkpoint, self.root / snapshot.archive)
        try:
            # Only serialize committed artifact references. The evaluator checks
            # their bytes and the actual scientific protocol in its own process.
            return publish_dispatch(request, None, self.declaration_sha256,
                                    route_declaration_sha256=self.policy.route_declaration_sha256,
                                    full_checkpoint=full_checkpoint)
        except (ValueError, OSError) as error:
            self.pending[games] = (snapshot, full_checkpoint)
            report_error(type(error).__name__ + ': ' + str(error))
            return None

    def stop_at_boundary(self, next_iteration):
        if self.stop_key is None:
            return None
        try:
            envelope, _ = _small_document(self.root / 'evaluation-stop.json')
            if type(envelope) is not dict or set(envelope) != {'payload', 'signature'}:
                raise ValueError('invalid stop envelope')
            payload = envelope['payload']
            if (type(envelope['signature']) is not str
                    or not hmac.compare_digest(envelope['signature'], _signature(payload, self.stop_key))):
                raise ValueError('invalid route stop signature')
            if (payload['schema'] != STOP or payload['verdict'] != 'FAIL'
                    or payload['route_schema'] != 's0.transfer.r2'
                    or payload['route_declaration_sha256'] != self.policy.route_declaration_sha256
                    or payload['declaration_sha256'] != self.declaration_sha256
                    or payload['protocol_sha256'] != self.policy.protocol_sha256):
                raise ValueError('stop scope or verdict differs')
            iteration = payload['iteration']
            games = self.milestones.get(iteration)
            if type(iteration) is not int or games is None or payload['declared_games'] != games:
                raise ValueError('stop is not for a declared milestone')
            if iteration >= next_iteration:
                return None
            request, digest = _small_document(self.path(games))
            if (digest != payload['dispatch_sha256'] or request['schema'] != MILESTONE
                    or request['request']['declared_games'] != games
                    or request['request']['snapshot']['iteration'] != iteration):
                raise ValueError('stop does not match the committed evaluation request')
            run.c._binding(payload['independent_verdict_sha256'])
            return payload
        except FileNotFoundError:
            return None
        except (ValueError, TypeError, KeyError, OSError) as error:
            problem = type(error).__name__ + ': ' + str(error)
            if self.report_observer_error is not None and self.last_error != problem:
                self.report_observer_error(problem)
            self.last_error = problem
            return None


def publish_route_failure(reference, verdict_reference, *, key=None):
    """Called by the trusted r2 recompute owner after its registered verdict."""
    if key is None:
        key = _stop_key()
    if type(key) is not bytes or len(key) < 32:
        raise ValueError('S0 route stop signing key required')
    value = reference.document()
    request = read_milestone(reference)
    verdict = verdict_reference.document()
    if (not value['route_declaration_sha256'] or verdict['schema'] != 's0.transfer.r2'
            or verdict['verdict'] != 'FAIL'
            or verdict['route_declaration_sha256'] != value['route_declaration_sha256']
            or verdict.get('validity') != 'VALID'
            or reference.sha256 not in verdict['dispatch_sha256s']):
        raise ValueError('verified failed r2 verdict for this request required')
    for field in ('raw_cells_sha256', 'primary_result_sha256', 'independent_code_sha256'):
        run.c._binding(verdict[field])
    endpoints = verdict['completion_endpoints_pp']
    boundaries = {'D': 2, 'G': 3, 'J': 3, 'Q_FP': 2, 'Q_H2H': 2}
    if set(endpoints) != set(boundaries):
        raise ValueError('all five registered S0 route contrasts required')
    robust = False
    for name, bound in boundaries.items():
        lower, upper = (endpoints[name][key] for key in ('L_min', 'U_max'))
        if (not all(type(x) in (int, float) and math.isfinite(x) for x in (lower, upper))
                or lower > upper):
            raise ValueError('invalid r2 route interval')
        robust |= upper < -bound or (name != 'D' and lower > bound)
    if not robust:
        raise ValueError('r2 FAIL requires a robust rejection under all completions')
    payload = dict(schema=STOP, verdict='FAIL', route_schema='s0.transfer.r2',
        route_declaration_sha256=value['route_declaration_sha256'], declaration_sha256=request.declaration_sha256,
        protocol_sha256=request.protocol_sha256, iteration=request.snapshot.iteration,
        declared_games=request.declared_games, dispatch_sha256=reference.sha256,
        independent_verdict_sha256=verdict_reference.sha256)
    path = request.run_directory / 'evaluation-stop.json'
    return _publish_equal(path, dict(payload=payload, signature=_signature(payload, key)))


def retained_full_checkpoints(checkpoints, milestones):
    """One full checkpoint per milestone and the latest one for recovery."""
    if not checkpoints:
        return set()
    by_iteration = {row['iteration']: row['path'] for row in checkpoints}
    return {checkpoints[-1]['path']} | {
        path for iteration, path in by_iteration.items() if iteration in milestones or iteration == -1}


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_actor(path):
    with path.open('rb') as stream:
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def retain_full_checkpoints(root, checkpoints, milestones):
    """Apply retention to owned checkpoint receipts after the durable commit."""
    keep = retained_full_checkpoints(checkpoints, milestones)
    artifacts = Path(root).resolve() / 'artifacts'
    if artifacts.is_symlink():
        raise ValueError('checkpoint retention requires an owned artifact directory')
    for row in checkpoints:
        relative = Path(row['path'])
        base = f"train-{row['iteration']}"
        if (row['path'] in keep or relative.parent != Path('artifacts') or relative.suffix != '.pt'
                or not (relative.stem == base or relative.stem.startswith(base + '-'))):
            continue
        path = artifacts / relative.name
        if path.is_symlink():
            raise ValueError('checkpoint retention requires an owned regular file')
        path.unlink(missing_ok=True)


def _publish_equal(path,value):
    blob=run._json(value).encode();path=Path(path)
    if path.exists():
        if path.read_bytes()!=blob:raise ValueError('immutable CPU panel artifact changed')
    else:
        try:run._publish(path,blob)
        except FileExistsError:
            if path.read_bytes()!=blob:raise ValueError('immutable CPU panel publication differs')
        sync_directory(path.parent)
    return evaluation.Artifact(str(path.resolve()),run._file_sha(path))


def _binding(request):
    value=dict(iteration=request.snapshot.iteration,declared_games=request.declared_games,
        declaration_sha256=request.declaration_sha256,protocol_sha256=request.protocol_sha256,
        snapshot=asdict(request.snapshot))
    return dict(value,evaluation_sha256=run._sha(value))


def _prior(request):
    rows=sorted(request.prior_completions,key=lambda r:r.cell_ordinal)
    if len({r.cell_ordinal for r in rows})!=len(rows):raise ValueError('duplicate carried CPU panel cell')
    for receipt in rows:
        run._checked_evaluation_completion(receipt)
        run._checked_completion_protocol(receipt,_binding(request))
    return [asdict(r) for r in rows]


def checked_dispatch_document(value,request):
    evaluation._keys(value,('schema','protocol','expected_run_declaration_sha256','request','prior_completions'))
    reference=evaluation.Artifact(**value['protocol']);protocol=reference.document()
    if (value['schema']!=SCHEMA or value['expected_run_declaration_sha256']!=request.declaration_sha256
            or value['request']!=evaluation._request_document(request)
            or value['prior_completions']!=_prior(request)):
        raise ValueError('original frozen CPU panel request differs')
    evaluation._check_request(request,protocol,reference,value['expected_run_declaration_sha256'])
    return value


def publish_dispatch(request,protocol,expected_run_declaration_sha256, *,
                     route_declaration_sha256=None, full_checkpoint=None):
    """Idempotent publication only; never starts a process or grants RESULT."""
    if protocol is None:
        value = dict(schema=MILESTONE, request=evaluation._request_document(request),
                     route_declaration_sha256=route_declaration_sha256,
                     full_checkpoint=full_checkpoint or dict(path=request.snapshot.checkpoint,
                                                            sha256=request.snapshot.checkpoint_sha256))
        if request.declaration_sha256 != expected_run_declaration_sha256:
            raise ValueError('milestone request differs from the committed declaration')
    else:
        value=dict(schema=SCHEMA,protocol=asdict(protocol),expected_run_declaration_sha256=expected_run_declaration_sha256,
            request=evaluation._request_document(request),prior_completions=_prior(request))
        checked_dispatch_document(value,request)
    directory=request.run_directory/'artifacts'/'cpu-panels'/str(request.declared_games)
    path = directory/'request.json'
    if protocol is None:
        if not path.exists():
            try:
                run._publish(path, run._json(dict(value, published_unix=time.time())).encode())
                sync_directory(directory)
            except FileExistsError:
                pass
        existing = json.loads(path.read_bytes())
        if {k: v for k, v in existing.items() if k != 'published_unix'} != value:
            raise ValueError('immutable milestone dispatch differs')
        reference = evaluation.Artifact(str(path.resolve()), run._file_sha(path))
    else:
        reference = _publish_equal(path, value)
    relative=str(Path(reference.path).relative_to(request.run_directory))
    return run.EvaluationDispatch(relative,reference.sha256)


@dataclass(frozen=True)
class DetachedEvaluationHook:
    protocol:evaluation.Artifact
    expected_run_declaration_sha256:str

    def __post_init__(self):
        run.c._binding(self.expected_run_declaration_sha256)
        protocol,_=evaluation._protocol(self.protocol)
        evaluation._capture_resolver(protocol).require_original_capture(protocol)

    def __call__(self,request):
        return publish_dispatch(request,self.protocol,self.expected_run_declaration_sha256)


def _restore(value):
    request=evaluation._restore_request(value['request'])
    request=replace(request,prior_completions=tuple(run.EvaluationCompletion(**r)for r in value['prior_completions']))
    checked_dispatch_document(value,request)
    return request


def _restore_milestone(value, *, staged_run_root=None):
    keys = ('schema', 'request', 'route_declaration_sha256', 'full_checkpoint')
    if 'published_unix' in value:
        keys += ('published_unix',)
        if type(value['published_unix']) not in (int, float) or not math.isfinite(value['published_unix']):
            raise ValueError('invalid original milestone publication time')
    evaluation._keys(value, keys)
    if value['schema'] != MILESTONE:
        raise ValueError('committed milestone dispatch required')
    request = evaluation._restore_request(value['request'])
    if staged_run_root is not None:
        original = request.run_directory
        if (request.archive_path != original / request.snapshot.archive
                or request.checkpoint_path != original / request.snapshot.checkpoint):
            raise ValueError('original dispatch contains a different artifact path')
        root = Path(staged_run_root).resolve()
        request = replace(request, run_directory=root, archive_path=root / request.snapshot.archive,
                          checkpoint_path=root / request.snapshot.checkpoint)
    declaration = json.loads((request.run_directory / 'declaration.json').read_bytes())
    policy = declaration['evaluation']
    if (run._sha(declaration) != request.declaration_sha256
            or policy['protocol_sha256'] != request.protocol_sha256
            or policy['execution'] != 'deferred'
            or policy['route_declaration_sha256'] != value['route_declaration_sha256']):
        raise ValueError('milestone request declaration differs')
    milestones = run.evaluation_milestones(
        (sum(len(batch['games']) for batch in iteration) for iteration in declaration['iterations']),
        policy['interval_games'])
    if milestones.get(request.snapshot.iteration) != request.declared_games:
        raise ValueError('request is outside declared milestone cadence')
    evaluation._check_request_binding(request, run._sha(declaration['model']),
                                      request.protocol_sha256, request.declaration_sha256,
                                      declaration=declaration)
    full = value['full_checkpoint']
    evaluation._keys(full, ('path', 'sha256'))
    run.c._binding(full['sha256'])
    path = (request.run_directory / full['path']).resolve()
    if not path.is_relative_to(request.run_directory) or (
            path != request.checkpoint_path or full['sha256'] != request.snapshot.checkpoint_sha256
    ) and run._file_sha(path) != full['sha256']:
        raise ValueError('original full milestone checkpoint differs')
    return request


def read_milestone(reference, *, staged_run_root=None):
    """Shared reader for separate CPU and GPU evaluator entries."""
    request = _restore_milestone(reference.document(), staged_run_root=staged_run_root)
    path = request.run_directory / 'artifacts' / 'cpu-panels' / str(request.declared_games) / 'request.json'
    if Path(reference.path).resolve() != path:
        raise ValueError('original owned milestone dispatch path required')
    return request


def execute_assay(reference, protocol, assay, *, resume=False):
    """Run a source-pinned assay adapter inside its own launcher-owned job.

    Adapters own the registered cells and the immutable per-cell attempt ledger.
    This controller records operational attempts. Only the separate independent
    r2 scorer may issue a route stop, even when an adapter succeeds.
    """
    import fcntl
    import subprocess
    import sys

    request = read_milestone(reference)
    if protocol.sha256 != request.protocol_sha256:
        raise ValueError('assay protocol differs from the committed request')
    spec = protocol.document()
    if spec['schema'] != ASSAY_PROTOCOL or assay not in ASSAY_DEVICES:
        raise ValueError('registered deferred assay required')
    cell = spec['milestones'][str(request.declared_games)][assay]
    evaluation._keys(cell, ('entry', 'input_manifest', 'deadline_seconds'))
    expected_device = ASSAY_DEVICES[assay]
    if expected_device == 'cpu':
        if os.environ.get('JAX_PLATFORMS') != 'cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('CPU assay requires CUDA disabled in the eval job')
    elif os.environ.get('JAX_PLATFORMS') != 'cuda' or os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        raise ValueError('engine assay requires the separate GPU eval job')
    entry = evaluation.Artifact(**cell['entry'])
    inputs = evaluation.Artifact(**cell['input_manifest'])
    entry.verify()
    inputs.verify()
    deadline = cell['deadline_seconds']
    if type(deadline) is not int or deadline <= 0:
        raise ValueError('finite declared assay deadline required')
    directory = Path(reference.path).parent / 'assays' / assay
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'invocation.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempts = sorted((directory / 'attempts').glob('*.json'))
        if [p.name for p in attempts] != [f'{i:06d}.json' for i in range(len(attempts))]:
            raise ValueError('assay attempt order differs')
        if attempts and not resume:
            raise ValueError('explicit evaluator resume required')
        for outcome_path in (directory / 'outcomes').glob('*.json'):
            outcome = json.loads(outcome_path.read_bytes())
            if outcome['dispatch_sha256'] != reference.sha256 or outcome['assay'] != assay:
                raise ValueError('assay outcome belongs to another request')
            if outcome['status'] == 'RESULT':
                raise ValueError('completed assay cannot replay')
        attempt = len(attempts)
        filename = f'{attempt:06d}.json'
        started = dict(dispatch_sha256=reference.sha256, protocol_sha256=protocol.sha256,
                       assay=assay, attempt=attempt, entry_sha256=entry.sha256, inputs_sha256=inputs.sha256)
        _publish_equal(directory / 'attempts' / filename, started)
        output = directory / 'results' / filename
        output.parent.mkdir(exist_ok=True)
        cmd = [sys.executable, entry.path, '--dispatch', reference.path, '--dispatch-sha256', reference.sha256,
               '--inputs', inputs.path, '--inputs-sha256', inputs.sha256, '--ledger', str(directory / 'cells'),
               '--output', str(output)]
        if resume:
            cmd.append('--resume')
        if 'progress_showdown' in spec and assay in ('external_fp_greedy', 'matched_h2h_showdown'):
            progress = evaluation.Artifact(**spec['progress_showdown']).document()
            registry = Path(progress['shared_cell_registry'])
            if not registry.is_absolute():
                raise ValueError('route overlap requires the declared shared cell store')
            cmd.extend(('--shared-cell-registry', str(registry)))
        outcome = dict(started, status='UNKNOWN')
        try:
            result = subprocess.run(cmd, timeout=deadline, check=False)
            outcome['exit_code'] = result.returncode
            if result.returncode == 0:
                metrics = json.loads(output.read_bytes())
                if metrics['dispatch_sha256'] != reference.sha256 or metrics['assay'] != assay:
                    raise ValueError('assay result identity differs')
                if metrics['status'] not in ('RESULT', 'UNKNOWN'):
                    raise ValueError('invalid assay result status')
                outcome.update(status=metrics['status'], result=dict(path=str(output), sha256=run._file_sha(output)))
            else:
                outcome['reason'] = 'assay entry exited unsuccessfully'
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
            outcome['reason'] = type(error).__name__ + ': ' + str(error)
        _publish_equal(directory / 'outcomes' / filename, outcome)
        return outcome


def _completed(directory,request):
    carried={r.cell_ordinal:r for r in request.prior_completions}
    for path in sorted((directory/'completions').glob('*.json')):
        receipt=run.EvaluationCompletion(**json.loads(path.read_bytes()))
        run._checked_evaluation_completion(receipt)
        run._checked_completion_protocol(receipt,_binding(request))
        if path.stem!=str(receipt.cell_ordinal):raise ValueError('CPU completion path differs from original ordinal')
        if receipt.cell_ordinal in carried and carried[receipt.cell_ordinal]!=receipt:
            raise ValueError('original CPU completion changed')
        carried[receipt.cell_ordinal]=receipt
    return carried


def _dispatch_inputs(reference, *, protocol=None):
    value = reference.document()
    if value['schema'] == MILESTONE:
        request = read_milestone(reference)
        if protocol is None or protocol.sha256 != request.protocol_sha256:
            raise ValueError('separate evaluator requires the declared protocol artifact')
        if value['route_declaration_sha256'] is not None:
            raise ValueError('S0 r2 requires its matched assay executor and registered roster')
    else:
        request = _restore(value)
        original = evaluation.Artifact(**value['protocol'])
        if protocol is not None and protocol != original:
            raise ValueError('legacy dispatch protocol differs')
        protocol = original
    return value, request, protocol


def execute_dispatch(reference,*,resume=False,enable_cpu_panel=False,protocol=None):
    """One original attempt on CPU; explicit resume keeps only untouched cells."""
    if enable_cpu_panel is not True:raise ValueError('explicit separate CPU panel execution required')
    import fcntl,os
    if os.environ.get('JAX_PLATFORMS')!='cpu' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('explicit CPU-only panel environment required')
    value, request, protocol = _dispatch_inputs(reference, protocol=protocol)
    directory=Path(reference.path).parent
    expected=request.run_directory/'artifacts'/'cpu-panels'/str(request.declared_games)/'request.json'
    if Path(reference.path).resolve()!=expected:raise ValueError('original owned CPU dispatch path required')
    with (directory/'invocation.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempts=sorted((directory/'attempts').glob('*.json'))
        if [p.name for p in attempts]!=[f'{i:06d}.json'for i in range(len(attempts))]:
            raise ValueError('CPU panel invocation order changed')
        if attempts and not resume:raise ValueError('explicit original CPU panel resume required')
        for path in sorted((directory/'outcomes').glob('*.json')):
            previous=json.loads(path.read_bytes())
            if previous['dispatch_sha256']!=reference.sha256:raise ValueError('CPU outcome belongs to another dispatch')
            if previous['outcome']['status']=='RESULT':raise ValueError('completed CPU panel cannot replay')
        attempt=len(attempts)
        started=dict(schema=ATTEMPT,dispatch_sha256=reference.sha256,attempt=attempt)
        _publish_equal(directory/'attempts'/f'{attempt:06d}.json',started)
        def completion(receipt):
            run._checked_evaluation_completion(receipt)
            run._checked_completion_protocol(receipt,_binding(request))
            if receipt.cell_ordinal in prior and prior[receipt.cell_ordinal]!=receipt:
                raise ValueError('completed CPU source changed on resume')
            _publish_equal(directory/'completions'/f'{receipt.cell_ordinal}.json',asdict(receipt))
            prior[receipt.cell_ordinal]=receipt
        try:
            prior = _completed(directory, request)
            actual = replace(request, progress_callback=completion, prior_completions=tuple(prior.values()))
            hook = evaluation.EvaluationHook(protocol, request.declaration_sha256)
            outcome = hook(actual)
            if type(outcome) is not run.EvaluationOutcome:
                raise ValueError('original CPU evaluator outcome required')
            # Recheck immutable inputs only in the evaluator.
            _dispatch_inputs(reference, protocol=protocol)
        except Exception as error:
            outcome = run.EvaluationOutcome('UNKNOWN', reason=type(error).__name__ + ': ' + str(error))
        result=dict(started,outcome=asdict(outcome))
        _publish_equal(directory/'outcomes'/f'{attempt:06d}.json',result)
        return outcome


def read_dispatched_result(reference, *, protocol=None):
    """Read an actual CPU outcome without writing the GPU-owned journal."""
    _, request, protocol = _dispatch_inputs(reference, protocol=protocol)
    directory=Path(reference.path).parent
    attempts=sorted((directory/'attempts').glob('*.json'))
    paths=sorted((directory/'outcomes').glob('*.json'))
    if [p.name for p in attempts]!=[f'{i:06d}.json'for i in range(len(attempts))]:
        raise ValueError('CPU panel invocation order changed')
    for index,started in enumerate(attempts):
        if json.loads(started.read_bytes())!=dict(schema=ATTEMPT,dispatch_sha256=reference.sha256,attempt=index):
            raise ValueError('original CPU invocation binding differs')
    if any(p.name not in {a.name for a in attempts}for p in paths):
        raise ValueError('CPU outcome lacks its durable invocation')
    if not attempts or not (directory/'outcomes'/attempts[-1].name).exists():
        return dict(status='PENDING',dispatch=asdict(reference))
    path=directory/'outcomes'/attempts[-1].name;document=json.loads(path.read_bytes())
    evaluation._keys(document,('schema','dispatch_sha256','attempt','outcome'))
    if (document['schema']!=ATTEMPT or document['dispatch_sha256']!=reference.sha256
            or type(document['attempt']) is not int or path.stem!=f"{document['attempt']:06d}"):
        raise ValueError('original CPU outcome binding differs')
    started=directory/'attempts'/path.name
    if json.loads(started.read_bytes())!={k:document[k]for k in ('schema','dispatch_sha256','attempt')}:
        raise ValueError('original CPU outcome lacks its durable invocation')
    outcome=run.EvaluationOutcome(**document['outcome'])
    if outcome.report is None:
        if outcome.status!='UNKNOWN':raise ValueError('CPU result requires actual report')
        return dict(status='UNKNOWN',outcome=asdict(outcome),dispatch=asdict(reference))
    target=(request.run_directory/outcome.report).resolve()
    if not target.is_relative_to(request.run_directory.resolve()):raise ValueError('CPU report outside original run')
    artifact=evaluation.Artifact(str(target),outcome.report_sha256);report=artifact.document()
    if (not {'request','status','rows','known_completed','unknown_cells','completed_only_score','score_bounds'}<=set(report)
            or report['request']!=evaluation._request_document(request) or report['status']!=outcome.status):
        raise ValueError('CPU report replaced the frozen request/outcome')
    _,panels=evaluation._protocol(protocol,parse_configs=True,checkpoint=request.checkpoint_path)
    configs=panels[request.declared_games];expected=evaluation._fixed_rows(configs)
    receipts=evaluation._completed_receipts(report['rows'],expected,configs,protocol_path=protocol.path)
    reported={r.cell_ordinal:r for r in receipts}
    for ordinal,receipt in _completed(directory,request).items():
        if reported.get(ordinal)!=receipt:
            raise ValueError('CPU report changed or erased an original durable completion')
    known=len(receipts);scores=[r['score']for r in report['rows']if r['status']=='COMPLETE'];unknown=1000-known
    if outcome.status=='RESULT' and any(r['status']in('RUNNING','NOT_ATTEMPTED')for r in report['rows']):
        raise ValueError('CPU result still contains untouched or in-flight cells')
    if (report['known_completed']!=known or report['unknown_cells']!=unknown
            or report['completed_only_score']!=(sum(scores)/known if known else None)
            or report['score_bounds']!=[sum(scores)/1000,(sum(scores)+unknown)/1000]):
        raise ValueError('full CPU panel denominator or score bounds changed')
    return dict(status=outcome.status,outcome=asdict(outcome),report=asdict(artifact),
        known_completed=known,unknown_cells=unknown,score_bounds=report['score_bounds'],dispatch=asdict(reference))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dispatch',required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--protocol');parser.add_argument('--protocol-sha256')
    parser.add_argument('--assay', choices=tuple(ASSAY_DEVICES))
    parser.add_argument('--resume',action='store_true');parser.add_argument('--enable-cpu-panel',action='store_true')
    args=parser.parse_args(argv)
    if (args.protocol is None) != (args.protocol_sha256 is None):
        parser.error('--protocol and --protocol-sha256 are required together')
    protocol = None if args.protocol is None else evaluation.Artifact(args.protocol, args.protocol_sha256)
    reference = evaluation.Artifact(args.dispatch, args.sha256)
    if args.assay is not None:
        if protocol is None or args.enable_cpu_panel:
            parser.error('registered assays require the protocol artifact and their declared job device')
        result = execute_assay(reference, protocol, args.assay, resume=args.resume)
        print(run._json(result))
        return 0 if result['status'] == 'RESULT' else 1
    outcome=execute_dispatch(reference,resume=args.resume,
                             enable_cpu_panel=args.enable_cpu_panel,protocol=protocol)
    print(run._json(asdict(outcome)))
    return 0 if outcome.status=='RESULT' else 1


if __name__=='__main__':raise SystemExit(main())
