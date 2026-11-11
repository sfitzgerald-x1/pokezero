"""Read-only reducer recovery, then recoverable tail/orphan quarantine.

Never constructs or advances a battle. Complete metadata supplies every
scoring fact; tensor files without metadata are retained as unknown evidence.
"""
from collections import Counter,defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import numpy as np


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def durable(path,blob):
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.publish-',delete=False) as stream:
        pending=Path(stream.name)
        stream.write(blob);stream.flush();os.fsync(stream.fileno())
    os.link(pending,path);pending.unlink()


def require_documents(root,expected):
    for name,value in expected.items():
        if (Path(root)/name).read_bytes()!=(canonical(value)+'\n').encode():
            raise ValueError('resume requires exact canonical '+name)


def _lines(path,repairs):
    """Only an unterminated final fragment can be repaired after inspection."""
    offset=0;h=hashlib.sha256()
    with path.open('rb') as stream:
        while raw:=stream.readline():
            start=offset;prefix=h.hexdigest();h.update(raw);offset+=len(raw)
            payload=raw[:-1] if raw.endswith(b'\n') else raw
            try:
                value=json.loads(payload)
                if type(value) is not dict or canonical(value).encode()!=payload:raise ValueError('noncanonical record')
            except (ValueError,UnicodeError) as error:
                if raw.endswith(b'\n'):raise ValueError('midstream/canonical corruption in '+path.name) from error
                repairs.append(dict(path=path,prefix_bytes=start,prefix_sha256=prefix,tail=raw,
                    original_sha256=h.hexdigest(),kind='UNPARSED_TRAILING_FRAGMENT'))
                break
            if not raw.endswith(b'\n'):
                repairs.append(dict(path=path,prefix_bytes=start,prefix_sha256=prefix,tail=raw,
                    original_sha256=h.hexdigest(),kind='COMPLETE_RECORD_MISSING_NEWLINE'))
            yield value,hashlib.sha256(payload).hexdigest()


def _cell_matches(row,cell):
    return canonical({k:row.get(k) for k in cell})==canonical(cell)


@dataclass
class Recovery:
    cells:list
    interrupted:list
    repairs:list
    orphans:list
    next_counter:int
    report:dict
    sequence:int=0

    def apply(self,root):
        """All source/data checks finished before any original byte changes."""
        root=Path(root);base=root/'recovery';base.mkdir(exist_ok=True)
        prior=[int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdecimal()]
        self.sequence=max(prior,default=0)+1;out=base/f'{self.sequence:06d}';out.mkdir()
        tails=[]
        for i,repair in enumerate(self.repairs):
            path=repair['path']
            if sha(path)!=repair['original_sha256']:raise ValueError('recovery input changed before tail repair')
            preserved=out/(path.name+'.tail.bin')
            durable(preserved,repair['tail'])
            tails.append({k:v for k,v in repair.items() if k not in ('path','tail')}|dict(
                file=path.name,tail_bytes=len(repair['tail']),preserved_path=str(preserved.relative_to(root))))
        orphans=[]
        if self.orphans:(out/'orphans').mkdir()
        for path in self.orphans:
            destination=out/'orphans'/path.name
            orphans.append(dict(original=str(path.relative_to(root)),quarantined=str(destination.relative_to(root)),
                bytes=path.stat().st_size,sha256=sha(path)))
        # Persist the complete reconstruction recipe BEFORE touching a byte.
        # A second interruption during recovery cannot erase unknown evidence
        # or lose the offset needed to reconstruct the original file.
        self.report.update(sequence=self.sequence,tail_repairs=tails,orphan_tensors=orphans,
            next_tensor_counter=self.next_counter)
        durable(out/'intent.json',(canonical(self.report)+'\n').encode())
        for repair in self.repairs:
            path=repair['path']
            with path.open('r+b') as stream:
                if repair['kind']=='UNPARSED_TRAILING_FRAGMENT':stream.truncate(repair['prefix_bytes'])
                else:stream.seek(0,2);stream.write(b'\n')
                stream.flush();os.fsync(stream.fileno())
        for path in self.orphans:
            destination=out/'orphans'/path.name
            path.rename(destination)
        durable(out/'report.json',(canonical(self.report)+'\n').encode())
        return dict(path=str(out/'report.json'),**self.report)


def inspect(root,plan,reducer,*,error_kinds):
    root=Path(root);repairs=[];declared={c['stream_id']:c for c in plan['cells']}
    if not (root/'rows').is_dir():raise ValueError('owning tensor directory missing')
    starts={};ends={};order=[]
    for row,_ in _lines(root/'journal.jsonl',repairs):
        stream=row.get('stream_id');cell=declared.get(stream)
        if cell is None or not _cell_matches(row,cell):raise ValueError('journal declaration/source address differs')
        if row.get('kind')=='START':
            if stream in starts or order!=[c['stream_id'] for c in plan['cells'][:len(order)]]:
                raise ValueError('repeated or unordered durable START')
            if len(order)>=len(plan['cells']) or stream!=plan['cells'][len(order)]['stream_id'] or any(s not in ends for s in order):
                raise ValueError('START must advance next not-attempted declared cell')
            starts[stream]={k:v for k,v in row.items() if k!='kind'};order.append(stream)
        elif row.get('kind')=='END':
            if stream not in starts or stream in ends:raise ValueError('END without one durable START')
            ends[stream]={k:v for k,v in row.items() if k!='kind'}
        else:raise ValueError('unexpected journal record')
    physical={};steps=Counter();step_requests=defaultdict(set)
    for row,binding in _lines(root/'physical.jsonl',repairs):
        stream=row.get('stream_id')
        if stream not in starts:raise ValueError('physical evidence has no durable START')
        if row.get('kind')=='STEP':
            request=row.get('request')
            if type(request)is not int or request<0 or request in step_requests[stream]:raise ValueError('physical STEP address differs')
            step_requests[stream].add(request)
            before=row.get('before',{});turn=before.get('state',{}).get('turn');requested=before.get('requested')
            if type(requested)is list and len(requested)==1 and type(requested[0])is list:requested=requested[0]
            if ('pre_agreement' not in row or not any(row['pre_agreement'] is value for value in (True,False,None))
                    or type(turn)is not list or len(turn)!=1 or type(turn[0])is not int
                    or type(requested)is not list or len(requested)!=2 or any(type(x)is not bool for x in requested)):
                raise ValueError('literal physical PRE eligibility/turn/request flags required')
            physical[binding]=(stream,request,row.get('pre_agreement'),turn[0],requested)
            steps[stream]+=int('after' in row and row.get('engine_error') is None)
        elif row.get('kind') not in ('CONSTRUCTOR','INITIAL_NATIVE'):raise ValueError('unexpected physical record')
    counts=Counter();errors=Counter();kinds=defaultdict(Counter);paths=set();ids=set()
    for row,_ in _lines(root/'decisions.jsonl',repairs):
        stream=row.get('stream_id');request=row.get('request');owner=row.get('owner_side');turn=row.get('turn')
        if (stream not in starts or row.get('root_id')!=declared[stream]['root_id'] or type(request)is not int or request<0
                or type(owner)is not int or owner not in (0,1) or type(turn)is not int or turn<0
                or row.get('decision_id')!=f'{stream}-{request}-{owner}' or row['decision_id'] in ids
                or row.get('agreement') is not True and row.get('agreement') is not False and row.get('agreement') is not None):
            raise ValueError('complete decision address/eligibility differs')
        binding=row.get('physical_binding_sha256')
        if binding not in physical:raise ValueError('decision physical binding missing')
        ps,pr,agreement,pre_turn,requested=physical[binding]
        if (ps,pr,agreement,pre_turn)!=(stream,request,row['agreement'],turn) or requested[owner] is not True:
            raise ValueError('decision physical address/turn/eligibility/request differs')
        relative=Path(row['tensor_path']);path=root/relative
        if relative.is_absolute() or relative.parts[:1]!=('rows',) or '..' in relative.parts or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('decision tensor must remain in owning rows directory')
        if path in paths:raise ValueError('one immutable tensor per decision required')
        blob=path.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=row['tensor_sha256']:raise ValueError('immutable decision tensor hash differs')
        records={'candidate':{},'reference':{}}
        with np.load(io.BytesIO(blob),allow_pickle=False) as archive:
            for name in archive.files:
                prefix,sep,key=name.partition('__')
                if prefix not in records or not sep:raise ValueError('unknown stored tensor namespace')
                records[prefix][key]=archive[name]
        for label,values in records.items():
            if type(row.get(label+'_present'))is not bool or (not row[label+'_present'] and values):
                raise ValueError('literal stored view presence required; never infer from absent arrays')
        error=row.get('candidate_error')
        if type(error)is not int or error<0:raise ValueError('literal candidate error bits required')
        for bit,name in enumerate(error_kinds):kinds[stream][name]+=int(bool(error&(1<<bit)))
        counts[stream]+=1;errors[stream]+=int(error!=0);paths.add(path);ids.add(row['decision_id'])
        reducer.add(decision_id=row['decision_id'],turn=turn,line_kinds=row['line_kinds'],agreement=row['agreement'],
            reference=records['reference'] if row['reference_present'] else None,
            candidate=records['candidate'] if row['candidate_present'] else None,refusal=row['refusal'])
    all_files=list((root/'rows').glob('*.npz'))
    if any(p.is_symlink() for p in all_files):raise ValueError('stored tensor symlink is not owning evidence')
    orphans=[p for p in all_files if p not in paths]
    prior={}
    for name in ('intent.json','report.json'):
        for path in (root/'recovery').glob('*/'+name):prior[path.parent.name]=(path,json.loads(path.read_text()))
    previous=[value for _,value in prior.values()]
    unknown=bool(orphans or any(r['kind']=='UNPARSED_TRAILING_FRAGMENT' for r in repairs)
        or any(r.get('unknown_view_mass') for r in previous))
    interrupted=[];missing={}
    for stream,end in ends.items():
        if any(type(end.get(k))is not int or end[k]<0 for k in ('requested_views','executed_steps','candidate_error_views')):
            raise ValueError('literal END accounting required')
        if counts[stream]>end['requested_views']:raise ValueError('more decisions than immutable END')
        gap=end['requested_views']-counts[stream]
        if gap:
            if end['status'] in ('TERMINAL','REQUEST_CAP_UNKNOWN','TURN_CAP_UNKNOWN'):raise ValueError('completed END lost decision metadata')
            missing[str(stream)]=gap;unknown=True
        if end.get('counts_unknown'):unknown=True
        elif not gap and (end['candidate_error_views']!=errors[stream] or
                {k:v for k,v in end['candidate_error_kind_views'].items() if v}!={k:v for k,v in kinds[stream].items() if v}):
            raise ValueError('immutable END error accounting differs')
        if end['status'] in ('TERMINAL','REQUEST_CAP_UNKNOWN','TURN_CAP_UNKNOWN') and end['executed_steps']!=steps[stream]:
            raise ValueError('immutable END step accounting differs')
    for stream in order:
        if stream in ends:continue
        end=dict(starts[stream],status='INTERRUPTED_UNKNOWN',executed_steps=steps[stream],requested_views=counts[stream],
            candidate_error_views=errors[stream],candidate_error_kind_views={k:v for k,v in kinds[stream].items() if v},
            final_turn=None,final_done=None,final_winner=None,final_physical_prefix=None,counts_unknown=True,
            accounting_scope='Known complete literal records only; unrecorded activity/eligibility/errors remain UNKNOWN. Never replay this cell.')
        interrupted.append(end);ends[stream]=end;unknown=True
    counters=[int(m.group(1))+1 for p in all_files if (m:=re.match(r'resume-(\d+)-',p.name))]
    counter=max([len(all_files),*counters,*(p.get('next_tensor_counter',0) for p in previous)])
    report=dict(schema='pokezero.device-collection-recovery.v1',restored_decision_rows=len(ids),
        preserved_end_cells=len(ends)-len(interrupted),interrupted_cells=[r['stream_id'] for r in interrupted],
        unparsed_trailing_fragments=sum(r['kind']=='UNPARSED_TRAILING_FRAGMENT' for r in repairs),
        orphan_tensor_files=len(orphans),missing_metadata_rows_by_ended_cell=missing,unknown_view_mass=unknown,
        prior_recoveries=[str(path.relative_to(root)) for path,_ in prior.values()],
        scope='Only complete original metadata supplies turn/eligibility/refusal/error facts. Orphan/tail counts overlap and are not a reconstructed view denominator.')
    return Recovery([ends[s] for s in order],interrupted,repairs,orphans,counter,report)


def scope_unknown(report,recovery):
    if recovery and recovery.get('unknown_view_mass'):
        report['joint_match_fraction_on_RECORDED_agreement_views']=report['joint_match_fraction_on_ALL_agreement_views']
        report['joint_match_fraction_on_ALL_agreement_views']=None
        report['joint_denominator_scope']+=' Interrupted/unparsed view mass is UNKNOWN; the exact ALL-view fraction is unavailable.'
        report['numerical_gate_only']=False
    return report
