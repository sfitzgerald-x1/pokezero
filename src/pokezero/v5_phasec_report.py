"""Repair only a completed throughput report from immutable saved evidence.

No backend, model, engine, optimizer, replay or new profiler is invoked. The
original owner outcome and raw trace remain intact. Missing component counters
are unavailable, rather than reconstructed from inclusive phase durations.
"""
import copy
import hashlib
import json
from pathlib import Path

from . import v5_phasec_throughput as ct


def _sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def saved_events(root):
    """Read, but never append to, the original ordered/hash-linked journal."""
    events=[];previous=None
    for index,path in enumerate(sorted((root/'journal').glob('*.json'))):
        event=json.loads(path.read_bytes());digest=event.pop('sha256',None)
        if (path.name!=f'{index:09d}.json'or event['sequence']!=index
                or event['previous']!=previous or _sha(event)!=digest):
            raise ValueError('original ordered journal changed')
        event['sha256']=digest;events.append(event);previous=digest
    declaration=json.loads((root/'declaration.json').read_bytes())
    starts=[e['payload']for e in events if e['kind']=='RUN_STARTED']
    if starts!=[dict(declaration_sha256=_sha(declaration))]:
        raise ValueError('original run declaration binding differs')
    return events,declaration


def completed_smoke(events,declaration):
    records=[e for e in events if e['kind']=='SMOKE_END_TO_END']
    if len(records)!=1:raise ValueError('one original post-reload completion record required')
    event=records[0];smoke=copy.deepcopy(event['payload'])
    if (smoke['status']!='COMPLETE_DECLARATION'or smoke['declared_games']!=40000
            or smoke.get('iteration_policy_lag')!=1
            or smoke['jax_reload']!='VERIFIED_ORIGINAL_TRAINED_ARCHIVE_LOADED'):
        raise ValueError('both original cohorts and strict trained reload must already be complete')
    windows=smoke['pipeline_windows']
    if len(windows)!=1 or windows[0]['includes_interruption']:
        raise ValueError('this repair requires the original uninterrupted fixed window')
    if any(e['kind']=='CT_INTERRUPTED_BANK_RECOVERED'or
            e['kind']=='RUNTIME_PHASE'and e['payload'].get('scope')=='TRAINER_RESUME'for e in events):
        raise ValueError('numerical continuation is not a report-only repair')
    if declaration['model']['arm']!=smoke['model_arm']:
        raise ValueError('original model arm differs')
    expected={game['stream_id']for batches in declaration['iterations']for batch in batches for game in batch['games']}
    ends=[e['payload']for e in events[:event['sequence']]if e['kind']=='GAME_END']
    if (len(expected)!=40000 or len(ends)!=40000 or len({x['stream_id']for x in ends})!=40000
            or {x['stream_id']for x in ends}!=expected):
        raise ValueError('every original declared game disposition required')
    counts={key:sum(x['status']==key for x in ends)for key in smoke['dispositions']}
    if (any(x['status']not in ('COMPLETE','CAPPED')for x in ends)
            or counts!=smoke['dispositions']or any(counts.get(k,0)for k in ('RUNNING','NOT_ATTEMPTED','FAILED','REFUSED'))):
        raise ValueError('original completed/capped roster differs')
    return smoke,event


def repair_completed_report(run_root,owner_result,*,output=None):
    root=Path(run_root).resolve();owner=json.loads(Path(owner_result).read_bytes())
    if (owner.get('exit_code')not in (0,1)or owner.get('stop_reason')is not None
            or owner.get('observer_error')is not None):
        raise ValueError('terminal original owner required; no stopped or failed numerical work')
    events,declaration=saved_events(root);smoke,event=completed_smoke(events,declaration)
    for snapshot in smoke['complete_iterations']:
        for key in ('checkpoint','archive'):
            item=snapshot['snapshot'];path=(root/item[key]).resolve()
            if not path.is_relative_to(root)or _file_sha(path)!=item[key+'_sha256']:
                raise ValueError('original trained artifact changed')
    trace=root/'ct-training-trace.json';document=json.loads(trace.read_bytes())
    raw=root/'ct-result.json';counters=root/'ct-training-counters.json'
    if raw.exists():
        original=json.loads(raw.read_bytes())
        if original['original_pipeline']!=smoke:raise ValueError('original report/smoke binding differs')
        timing=copy.deepcopy(original['training']);counter_scope='ORIGINAL_REPORT_COUNTERS'
    elif counters.exists():
        value=json.loads(counters.read_bytes())
        if value['schema']!='pokezero.ct-training-counters.v1':raise ValueError('counter schema differs')
        timing=value['counters'];counter_scope='ORIGINAL_DURABLE_COMPONENT_COUNTERS'
    else:
        # Only the original completion/checkpoint counters survive. The old
        # in-memory timing dictionary cannot be recovered from phase totals.
        timing=dict(updates=0,prior_durable_updates=smoke['steps'],training_datasets=[],
            cache_prepare_seconds=None,batch_slice_seconds=None,tensor_conversion_seconds=None,
            update_wall_seconds=None,data_seconds_per_update=None,
            data_scope='UNAVAILABLE_ORIGINAL_COMPONENT_COUNTERS_NOT_PERSISTED')
        counter_scope='UNAVAILABLE_ORIGINAL_COMPONENT_COUNTERS_NOT_PERSISTED'
    stopped=any(x['action']=='DEVICE_STOPPED'for x in timing.get('profiler_activity',[]))
    attribution=ct.learner_trace_categories(document,capture_stopped=stopped)
    available=attribution['per_update_mean_available']
    timing.update(gpu_trace_path=str(trace),gpu_trace_sha256=_file_sha(trace),
        concurrent_gpu_attribution=attribution,gpu_sample_categories=attribution['learner'],
        gpu_profile_attempted_update_ordinals=list(ct.PROFILE_UPDATES),
        gpu_sample_updates=len(ct.PROFILE_UPDATES)if available else None,
        gpu_sample_update_ordinals=list(ct.PROFILE_UPDATES)if available else None,
        gpu_kernel_seconds_per_sampled_update=(attribution['learner']['kernel']['seconds']/len(ct.PROFILE_UPDATES)if available else None),
        gpu_scope=attribution['scope'])
    if counter_scope!='UNAVAILABLE_ORIGINAL_COMPONENT_COUNTERS_NOT_PERSISTED':
        timing['data_seconds_per_update']=((timing['batch_slice_seconds']+timing['tensor_conversion_seconds'])/timing['updates']if timing['updates']else None)
    report=ct.gate_report(smoke,timing)
    banks=ct.collector_banks(events)
    report['collector_phase_walls']=ct.collector_phase_totals(banks.values(),expected_banks=20)
    report['cache_pipeline']=[dict(bank=list(key),**p['cache_pipeline'])for key,p in sorted(banks.items())if'cache_pipeline'in p]
    report['teacher_queue']=[dict(bank=list(key),**p['teacher_queue'])for key,p in sorted(banks.items())if p.get('teacher_queue')is not None]
    report['threshold_before_report_repair_caveat']=report['threshold']
    report['threshold']='RECOVERED_REPORT_ONLY_NO_AUTOMATIC_ADMISSION'
    report['report_repair']=dict(original_owner=owner,component_counter_scope=counter_scope,
        original_smoke_sequence=event['sequence'],original_smoke_sha256=event['sha256'],
        numerical_computation_already_complete=True,original_window_and_full_study_walls_unchanged=True,
        no_backend_model_forward_optimizer_game_or_profile_invoked=True,
        original_report_preserved=raw.exists(),original_raw_trace_preserved=True,
        requires_one_independent_result_review=True)
    target=Path(output)if output is not None else root/'ct-result-report-repair.json'
    if target.resolve()in (raw.resolve(),trace.resolve(),Path(owner_result).resolve()):
        raise ValueError('original evidence must not be overwritten')
    with target.open('x')as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    return report


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True);parser.add_argument('--owner-result',required=True)
    parser.add_argument('--output')
    args=parser.parse_args(argv)
    result=repair_completed_report(args.run_root,args.owner_result,output=args.output)
    print(json.dumps(dict(status=result['threshold'],rate=result['declared_games_per_second'])))
    return 0


if __name__=='__main__':raise SystemExit(main())
