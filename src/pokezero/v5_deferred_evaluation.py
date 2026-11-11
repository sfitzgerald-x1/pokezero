"""Durable CPU panel dispatch; no launch, model or game runs on import.

The GPU owner writes one immutable request and continues training. An external
CPU Job explicitly executes one original panel attempt. That Job never writes
the training journal. The controller file may run outside the frozen panel
source tree, importing that original tree's evaluator and scorer unchanged.
"""
import argparse
from dataclasses import asdict,dataclass,replace
import json
from pathlib import Path

from pokezero import v5_run as run, v5_evaluation as evaluation

SCHEMA='pokezero.v5.cpu_panel_dispatch.v1'
ATTEMPT='pokezero.v5.cpu_panel_attempt.v1'


def _publish_equal(path,value):
    blob=run._json(value).encode();path=Path(path)
    if path.exists():
        if path.read_bytes()!=blob:raise ValueError('immutable CPU panel artifact changed')
    else:
        try:run._publish(path,blob)
        except FileExistsError:
            if path.read_bytes()!=blob:raise ValueError('immutable CPU panel publication differs')
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


def publish_dispatch(request,protocol,expected_run_declaration_sha256):
    """Idempotent publication only; never starts a process or grants RESULT."""
    value=dict(schema=SCHEMA,protocol=asdict(protocol),expected_run_declaration_sha256=expected_run_declaration_sha256,
        request=evaluation._request_document(request),prior_completions=_prior(request))
    checked_dispatch_document(value,request)
    directory=request.run_directory/'artifacts'/'cpu-panels'/str(request.declared_games)
    reference=_publish_equal(directory/'request.json',value)
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


def execute_dispatch(reference,*,resume=False,enable_cpu_panel=False):
    """One original attempt on CPU; explicit resume keeps only untouched cells."""
    if enable_cpu_panel is not True:raise ValueError('explicit separate CPU panel execution required')
    import fcntl,os
    if os.environ.get('JAX_PLATFORMS')!='cpu' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('explicit CPU-only panel environment required')
    value=reference.document();request=_restore(value);directory=Path(reference.path).parent
    expected=request.run_directory/'artifacts'/'cpu-panels'/str(request.declared_games)/'request.json'
    if Path(reference.path)!=expected:raise ValueError('original owned CPU dispatch path required')
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
        prior=_completed(directory,request)
        def completion(receipt):
            run._checked_evaluation_completion(receipt)
            run._checked_completion_protocol(receipt,_binding(request))
            if receipt.cell_ordinal in prior and prior[receipt.cell_ordinal]!=receipt:
                raise ValueError('completed CPU source changed on resume')
            _publish_equal(directory/'completions'/f'{receipt.cell_ordinal}.json',asdict(receipt))
            prior[receipt.cell_ordinal]=receipt
        actual=replace(request,progress_callback=completion,prior_completions=tuple(prior.values()))
        hook=evaluation.EvaluationHook(evaluation.Artifact(**value['protocol']),value['expected_run_declaration_sha256'])
        outcome=hook(actual)
        if type(outcome) is not run.EvaluationOutcome:raise ValueError('original CPU evaluator outcome required')
        # Recheck the unchanged training input after the CPU worker returns.
        checked_dispatch_document(value,_restore(value))
        result=dict(started,outcome=asdict(outcome))
        _publish_equal(directory/'outcomes'/f'{attempt:06d}.json',result)
        return outcome


def read_dispatched_result(reference):
    """Read an actual CPU outcome without writing the GPU-owned journal."""
    value=reference.document();request=_restore(value);directory=Path(reference.path).parent
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
            or report['request']!=value['request'] or report['status']!=outcome.status):
        raise ValueError('CPU report replaced the frozen request/outcome')
    protocol,panels=evaluation._protocol(evaluation.Artifact(**value['protocol']),parse_configs=True,checkpoint=request.checkpoint_path)
    configs=panels[request.declared_games];expected=evaluation._fixed_rows(configs)
    receipts=evaluation._completed_receipts(report['rows'],expected,configs,protocol_path=value['protocol']['path'])
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
    parser.add_argument('--resume',action='store_true');parser.add_argument('--enable-cpu-panel',action='store_true')
    args=parser.parse_args(argv)
    outcome=execute_dispatch(evaluation.Artifact(args.dispatch,args.sha256),resume=args.resume,enable_cpu_panel=args.enable_cpu_panel)
    print(run._json(asdict(outcome)))
    return 0 if outcome.status=='RESULT' else 1


if __name__=='__main__':raise SystemExit(main())
