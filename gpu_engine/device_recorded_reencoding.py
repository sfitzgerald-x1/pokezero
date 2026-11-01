"""Re-encode a FINAL fixed device/Native collection, without new games.

Only literal saved device constructor/step inputs are evaluated. Every numeric
state field must equal the saved output before public carry is trusted again.
Original Native agreement, unknowns, refusals, turns and reference tensors are
immutable inputs. This cannot improve their physical-agreement classification.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import os
import subprocess
from types import MappingProxyType, ModuleType

import numpy as np

from . import device_observation_collector as collector
from .device_observation_parity import AgreementParity, SHAPES

SCHEMA='pokezero.device-recorded-reencoding.v1'
FILES={'plan':'plan.json','summary':'summary.json','source_bindings':'source-bindings.json',
       'physical':'physical.jsonl','decisions':'decisions.jsonl','journal':'journal.jsonl'}


def _document(ref):return collector.checked_document(ref)


def _verify_file(ref):
    if (type(ref)is not dict or set(ref)!={'path','sha256'} or not collector._hex(ref['sha256'])
            or type(ref['path'])is not str or collector.digest(ref['path'])!=ref['sha256']):
        raise ValueError('immutable stored file hash differs')
    return Path(ref['path'])


def _records(path):
    with Path(path).open('rb')as stream:
        before=os.fstat(stream.fileno())
        for raw in stream:
            if not raw.endswith(b'\n') or not raw.strip():raise ValueError('complete JSONL records required')
            yield json.loads(raw),hashlib.sha256(raw[:-1]).hexdigest()
        after=os.fstat(stream.fileno())
        if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError('stored file changed during read')


def _owner_complete(job,pod):
    conditions=job.get('status',{}).get('conditions',[])
    if (not any(x.get('type')=='Complete'and x.get('status')=='True'for x in conditions)
            or pod.get('status',{}).get('phase')!='Succeeded'):
        raise ValueError('original source owner must be FINAL COMPLETE/Succeeded')
    states=pod.get('status',{}).get('containerStatuses',[])
    if not states or any(type(x.get('state',{}).get('terminated',{}).get('exitCode'))is not int
            or x['state']['terminated']['exitCode']!=0 for x in states):
        raise ValueError('original owner exit0 terminal evidence required')
    uid=job.get('metadata',{}).get('uid')
    if not uid or not any(x.get('kind')=='Job'and x.get('uid')==uid for x in pod.get('metadata',{}).get('ownerReferences',[])):
        raise ValueError('original source Job/Pod ownership differs')


def validate_inputs(manifest,*,expected_cells=512):
    """Final-file/provenance admission before any device evaluation."""
    if (type(manifest)is not dict or set(manifest)!={'schema','original_root','files','final_owner','runtime_plan','original_reducer'}
            or manifest['schema']!=SCHEMA or set(manifest['files'])!=set(FILES)
            or set(manifest['final_owner'])!={'job','pod'}):
        raise ValueError('complete final stored-collection manifest required')
    root=Path(manifest['original_root']).resolve()
    for key,ref in manifest['files'].items():
        path=_verify_file(ref)
        if path.resolve().parent!=root or path.name!=FILES[key]:raise ValueError('foreign stored collection file')
    _owner_complete(_document(manifest['final_owner']['job']),_document(manifest['final_owner']['pod']))
    plan=_document(manifest['files']['plan']);summary=_document(manifest['files']['summary'])
    bindings=_document(manifest['files']['source_bindings']);runtime_plan=_document(manifest['runtime_plan'])
    collector.validate_plan(plan);collector.validate_plan(runtime_plan)
    if ({k:v for k,v in plan.items()if k!='source_commit'}!={k:v for k,v in runtime_plan.items()if k!='source_commit'}):
        raise ValueError('stored dataset/config must be identical; only source revision may change')
    if (len(plan['cells'])!=expected_cells or summary.get('declared_cells')!=expected_cells
            or summary.get('source_bindings')!=bindings or bindings.get('source_commit')!=plan['source_commit']
            or summary.get('collection_schema')!=collector.SCHEMA):
        raise ValueError('original FINAL full declared cohort/source binding required')
    cells=summary.get('cells',[])
    if len(cells)!=expected_cells:raise ValueError('original final per-cell manifest missing')
    starts=[];ends=[]
    for row,_ in _records(manifest['files']['journal']['path']):
        if row.get('kind')=='START':starts.append(row)
        elif row.get('kind')=='END':ends.append(row)
        else:raise ValueError('original journal kind differs')
    if len(starts)!=expected_cells or len(ends)!=expected_cells:raise ValueError('full START/END declaration required')
    for cell,start,end,final in zip(plan['cells'],starts,ends,cells):
        if (any(start.get(k)!=v or end.get(k)!=v or final.get(k)!=v for k,v in cell.items())
                or {k:v for k,v in end.items()if k!='kind'}!=final or final.get('status')=='ATTEMPTED'):
            raise ValueError('final cell/journal/order differs')
    for key in ('plan','source_bindings','physical','decisions','journal'):
        saved=summary.get('artifact_files',{}).get(FILES[key],{})
        ref=manifest['files'][key]
        if (saved.get('sha256')!=ref['sha256'] or saved.get('bytes')!=Path(ref['path']).stat().st_size):
            raise ValueError('original summary file binding differs: '+key)
    # Original provenance belongs to its exact frozen source, not the current
    # reviewed comparator. No substituted "equivalent" implementation.
    _frozen_reducer(bindings,reference=manifest['original_reducer'])
    return root,plan,summary,bindings,runtime_plan


def _frozen_reducer(bindings,*,reference=None):
    name='gpu_engine/device_observation_parity.py'
    expected=bindings.get('source_files',{}).get(name)
    if not collector._hex(expected):raise ValueError('original reducer source identity missing')
    if reference is None:
        # Compatibility for direct helper callers: read a genuine immutable
        # Git blob, never the current file or a caller's fake legacy class.
        commit=bindings.get('source_commit')
        if not collector._hex(commit,40):raise ValueError('original Git source revision missing')
        payload=subprocess.check_output(['git','show',commit+':'+name],cwd=collector.ROOT)
        location=commit+':'+name
    else:
        path=_verify_file(reference)
        if reference['sha256']!=expected:raise ValueError('original reducer reference differs from recorded source')
        payload=path.read_bytes();location=str(path)
    if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('frozen original reducer bytes differ')
    module=ModuleType('pokezero_frozen_original_parity_'+expected);module.__file__=location
    exec(compile(payload,location,'exec'),module.__dict__)
    if module.SHAPES!=SHAPES:raise ValueError('old/new observation tensor contracts differ')
    return module.AgreementParity


def _legacy_reproduction(root,decisions_path,summary,bindings,original_reducer):
    original=original_reducer(source_bindings=bindings)
    for row,_ in _records(decisions_path):
        candidate,reference,_=_load_row(root,row)
        original.add(decision_id=row['decision_id'],turn=row['turn'],line_kinds=row['line_kinds'],
            agreement=row['agreement'],reference=reference,candidate=candidate,refusal=row['refusal'])
    report=original.report()
    expected={key:summary.get(key)for key in report}
    if collector._json(report).encode()!=collector._json(expected).encode():
        raise ValueError('original frozen legacy report canonical bytes differ')
    return report,hashlib.sha256(collector._json(report).encode()).hexdigest()


@lru_cache(maxsize=1)
def _state_layout():
    from .state import empty_state
    template=empty_state(1)  # dtype/shape storage template, never a battle constructor
    return MappingProxyType({k:MappingProxyType(dict(shape=tuple(getattr(template,k).shape[1:]),
        dtype=str(getattr(template,k).dtype)))for k in template._fields})


def decode_state(packet):
    from .finite_gameplay_comparison import decode_state as original_decode
    return original_decode(packet,_state_layout())


def decode_chance(packet):
    from .transition import ChanceInputs
    if type(packet)is not dict or set(packet)!=set(ChanceInputs._fields):raise ValueError('all saved ChanceInputs required')
    arrays=[]
    for key in ChanceInputs._fields:
        a=np.asarray(packet[key]);shape=(1,2,12)if key=='draws'else(1,)if key=='initial_tie'else(1,2)
        bound=np.array([2,100,100,100,16,16,100,4,8,2,2,2]) if key=='draws' else 100 if key=='secondary' else 2 if key in ('entry_ties','initial_tie') else 60
        if a.shape!=shape or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>=bound):
            raise ValueError('saved chance shape/range differs: '+key)
        arrays.append(a.astype(np.int32))
    return ChanceInputs(*arrays)


def decode_actions(value):
    a=np.asarray(value)
    if a.shape!=(1,2)or a.dtype.kind not in 'iu'or np.any(a<0)or np.any(a>=9):
        raise ValueError('literal original physical actions required')
    return a.astype(np.int32)


def differences(actual,expected):
    if tuple(actual._fields)!=tuple(expected._fields):return ['STATE_SCHEMA']
    return [key for key in actual._fields if getattr(actual,key).dtype!=getattr(expected,key).dtype
        or not np.array_equal(np.asarray(getattr(actual,key)),np.asarray(getattr(expected,key)))]


def _boundary(packet):
    if type(packet)is not dict or set(packet)!={'state','requested'}:
        raise ValueError('complete recorded device boundary required')
    requested=np.asarray(packet['requested'])
    if requested.shape!=(2,)or requested.dtype!=np.bool_:raise ValueError('literal requested seats required')
    return decode_state(packet['state']),requested


def _load_row(root,row):
    root=Path(root).resolve()
    path=Path(row['tensor_path'])
    if path.is_absolute()or '..'in path.parts:raise ValueError('original relative tensor path required')
    path=(root/path).resolve()
    if not path.is_relative_to(root):raise ValueError('foreign tensor path')
    if collector.digest(path)!=row['tensor_sha256']:raise ValueError('original tensor bytes changed')
    with np.load(path,allow_pickle=False)as archive:
        candidate={k[len('candidate__'):]:archive[k]for k in archive.files if k.startswith('candidate__')}
        reference={k[len('reference__'):]:archive[k]for k in archive.files if k.startswith('reference__')}
        if (set(candidate)!=set(SHAPES)or set(reference)not in (set(),set(SHAPES),set(SHAPES)|{'admitted'})
                or any(not(k.startswith('candidate__')or k.startswith('reference__'))for k in archive.files)):
            raise ValueError('original tensor namespace differs')
    return candidate,reference or None,path


def _refusal(reason,fields=None):
    return dict(source='STORED_REENCODING',kind='NUMERIC_OR_CARRY',column='ALL',reason=reason,
        differing_state_fields=[]if fields is None else fields)


class Replay:
    """One root's public continuity. All numeric targets come from saved data."""
    def __init__(self,runtime):self.runtime=runtime;self.state=None;self.public=None;self.stop=None
    def constructor(self,record):
        try:
            raw,_=_boundary(record['raw']);expected,requested=_boundary(record['initialized'])
            current=self.runtime.initialize(raw)
            bad=differences(current.state,expected)
            if not np.array_equal(self.runtime.boundary(current.state)['requested'],requested):bad=['REQUESTED_SEATS']+bad
            if bad:self.stop=_refusal('INITIALIZED_STATE_CHANGED',bad)
            else:self.state,self.public=current.state,current.trackers
        except Exception as error:self.stop=_refusal('CONSTRUCTOR_REFUSED:'+type(error).__name__+': '+str(error)[:1000])
        return self.stop

    def step(self,record):
        candidate=[None,None];error_bits=[None,None];pre_error=self.stop;post_error=None
        if self.stop is not None:return candidate,error_bits,pre_error,post_error
        try:
            before,requested=_boundary(record['before']);bad=differences(self.state,before)
            if bad:raise ValueError('PRE_STATE_CHANGED:'+','.join(bad))
            _,actual_requested,_,observations=self.runtime.observe(before,self.public)
            if not np.array_equal(np.asarray(actual_requested)[0],requested):raise ValueError('REQUESTED_SEATS_CHANGED')
            candidate=[collector._candidate(o)for o in observations]
            error_bits=[int(np.asarray(o.error)[0])for o in observations]
        except Exception as error:
            self.stop=pre_error=_refusal('PRE_VIEW_REFUSED:'+type(error).__name__+': '+str(error)[:1000])
            return candidate,error_bits,pre_error,post_error
        if 'after'not in record:
            # An original failed step has no stored numerical outcome to use.
            # Its already-recorded PRE view remains eligible under old flags.
            self.stop=post_error=_refusal('RECORDED_STEP_HAS_NO_AFTER; NO_NEW_OUTCOME_EVALUATED')
            return candidate,error_bits,pre_error,post_error
        try:
            after,requested_after=_boundary(record['after']);chance=decode_chance(record['chance']);actions=decode_actions(record['actions'])
            result=self.runtime.advance(before,actions,self.public,chance)
            bad=differences(result.step.state,after)
            if not np.array_equal(self.runtime.boundary(result.step.state)['requested'],requested_after):bad=['REQUESTED_SEATS']+bad
            if bool(np.asarray(result.step.invalid_actions)[0]):bad=['INVALID_ACTION']+bad
            if bad:self.stop=post_error=_refusal('AFTER_STATE_CHANGED',bad)
            else:self.state,self.public=result.step.state,result.trackers
        except Exception as error:self.stop=post_error=_refusal('STEP_REFUSED:'+type(error).__name__+': '+str(error)[:1000])
        return candidate,error_bits,pre_error,post_error


def reencode_records(runtime,root,plan,summary,old_bindings,physical_path,decisions_path,writer,*,original_reducer_reference=None):
    """Streaming record join; no Native object, chance draw or policy selection."""
    original_reducer=_frozen_reducer(old_bindings,reference=original_reducer_reference)
    # Complete byte-exact legacy reproduction BEFORE creating any new-mode
    # accumulator or evaluating a new candidate/device step.
    original_report,original_report_sha=_legacy_reproduction(root,decisions_path,summary,old_bindings,original_reducer)
    new=AgreementParity(source_bindings=dict(mode='STORED_REENCODING',original=old_bindings,current=runtime.bindings),
        comparison_mode='source_routing_coordinates')
    physical=iter(_records(physical_path));decisions=iter(_records(decisions_path))
    next_physical=next(physical,None);next_decision=next(decisions,None);roots=[];seen=set();rows=0
    for cell,original_final in zip(plan['cells'],summary['cells']):
        stream=cell['stream_id'];replay=Replay(runtime);constructor=False;steps=0
        writer.event(dict(kind='START_STORED_REENCODING',original_cell=original_final))
        while next_physical is not None and next_physical[0].get('stream_id')==stream:
            record,physical_sha=next_physical;kind=record.get('kind')
            if kind=='CONSTRUCTOR':
                if constructor or steps:raise ValueError('duplicate/late stored constructor')
                constructor=True;failure=replay.constructor(record)
                writer.boundary(dict(kind='REENCODE_CONSTRUCTOR',stream_id=stream,
                    original_physical_sha256=physical_sha,refusal=failure))
            elif kind=='INITIAL_NATIVE':pass  # original physical evidence retained, never reinterpreted
            elif kind=='STEP':
                if record.get('request')!=steps:raise ValueError('original STEP order differs')
                if not constructor:replay.stop=_refusal('MISSING_STORED_CONSTRUCTOR')
                candidates,bits,pre_error,post_error=replay.step(record)
                _,requested=_boundary(record['before'])
                owners=list(np.flatnonzero(requested))
                for side in owners:
                    if next_decision is None:raise ValueError('original decision missing')
                    row,_=next_decision
                    if (row.get('stream_id')!=stream or row.get('root_id')!=cell['root_id']
                            or row.get('request')!=steps or row.get('owner_side')!=int(side)
                            or row.get('physical_binding_sha256')!=physical_sha
                            or row.get('decision_id')!=f'{stream}-{steps}-{side}'
                            or row['decision_id']in seen or type(row.get('turn'))is not int
                            or row['turn']!=record['before']['state']['turn'][0]
                            or row.get('agreement')is not None and type(row['agreement'])is not bool
                            or row.get('agreement')is not record.get('pre_agreement')):
                        raise ValueError('original decision address/physical/agreement binding differs')
                    seen.add(row['decision_id']);rows+=1
                    _,reference,path=_load_row(root,row)
                    # Current PRE was already checked/encoded. A subsequent
                    # POST failure stops future carry; it cannot erase this
                    # independently recorded current PRE decision.
                    effective=row['refusal']if row['refusal']is not None else pre_error
                    metadata=dict(row,original_metadata=row,original_tensor_path=str(path),
                        original_tensor_sha256=row['tensor_sha256'],reencoding_refusal=pre_error,
                        reencoded_candidate_error=bits[side],mode='STORED_REENCODING')
                    writer.decision(metadata,candidates[side],reference)
                    new.add(decision_id=row['decision_id'],turn=row['turn'],line_kinds=row['line_kinds'],
                        agreement=row['agreement'],reference=reference,candidate=candidates[side],refusal=effective)
                    next_decision=next(decisions,None)
                writer.boundary(dict(kind='REENCODE_STEP',stream_id=stream,request=steps,
                    original_physical_sha256=physical_sha,pre_refusal=pre_error,post_refusal=post_error))
                steps+=1
            else:raise ValueError('unknown original physical record kind')
            next_physical=next(physical,None)
        roots.append(dict(stream_id=stream,stored_steps=steps,reencoding_stopped=replay.stop,
            original_cell_status=original_final['status']))
        writer.event(dict(kind='END_STORED_REENCODING',**roots[-1]))
    if next_physical is not None or next_decision is not None:raise ValueError('extra/foreign original records')
    result=new.report()
    # These denominators are frozen source inputs, not a new eligibility test.
    for key in ('attempted','agreement','state_disagreement','state_agreement_unknown'):
        if result['counts'].get(key,0)!=original_report['counts'].get(key,0):raise ValueError('immutable original denominator changed')
    result.update(mode='STORED_REENCODING',new_games=0,new_native_battles=0,model_calls=0,
        new_team_generation_calls=0,new_chance_sampling_calls=0,full_game_throughput=None,
        original_cells=summary['cells'],reencoding_cells=roots,original_counts=summary['counts'],
        original_report_recomputed_exact=True,stored_rows_retained=rows,
        original_legacy_report_sha256=original_report_sha,
        frozen_original_reducer_sha256=old_bindings['source_files']['gpu_engine/device_observation_parity.py'],
        replay_integrity_requirement_met=not any(row['reencoding_stopped']is not None for row in roots),
        replay_integrity_failed_roots=sum(row['reencoding_stopped']is not None for row in roots),
        provenance_qualification_status=('INELIGIBLE_REPLAY_INTEGRITY_FAILURE'
            if any(row['reencoding_stopped']is not None for row in roots) else 'PENDING_INDEPENDENT_REVIEW'),
        scope='Fixed original legal-policy collection; current device encoder/hooks applied to literal saved inputs only. All numeric fields must exactly match recorded states. Original physical flags/refusals unchanged; no fresh roots/outcomes, strength, Native collection or automatic B4 qualification.')
    return result


def run(manifest,*,out,enable_reencoding=False):
    if enable_reencoding is not True:raise ValueError('explicit stored reencoding enablement required')
    root,plan,summary,bindings,runtime_plan=validate_inputs(manifest)
    out=Path(out)
    if ('..'in out.parts or not any(out.is_relative_to(p)for p in (Path('/shared/scott'),Path('/store')))
            or out.exists()):raise ValueError('fresh remote-only reencoding output required')
    runtime=collector.prepare(runtime_plan)  # immutable catalogs/JITs only; no Factory.game/Native
    writer=collector.Evidence(out)
    try:
        (out/'manifest.json').write_text(collector._json(manifest)+'\n')
        result=reencode_records(runtime,root,plan,summary,bindings,manifest['files']['physical']['path'],
            manifest['files']['decisions']['path'],writer,original_reducer_reference=manifest['original_reducer'])
    finally:writer.close()
    for ref in manifest['files'].values():_verify_file(ref)
    _verify_file(manifest['original_reducer'])
    result['input_manifest']=manifest
    result['output_files']={p.name:dict(sha256=collector.digest(p),bytes=p.stat().st_size)
        for p in (out/'manifest.json',out/'decisions.jsonl',out/'physical.jsonl',out/'journal.jsonl')}
    path=out/'summary.json';path.write_text(collector._json(result)+'\n')
    return dict(mode='STORED_REENCODING',summary_path=str(path),summary_sha256=collector.digest(path),
        new_games=0,full_B4_pass=False)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--expected-manifest-sha256',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--enable-reencoding',action='store_true')
    args=parser.parse_args(argv)
    if not args.enable_reencoding:parser.error('--enable-reencoding is required before work')
    manifest=_document(dict(path=args.manifest,sha256=args.expected_manifest_sha256))
    print(collector._json(run(manifest,out=args.out,enable_reencoding=True)),flush=True)


if __name__=='__main__':main()
