#!/usr/bin/env python3
"""Fixed TRAIN root bank and paired full EngineMctsPolicy action screen.

No FoulPlay, new games, outcome selection, cost screen, export, or replay of old
scientific roots. Source/runtime tensors/models/results stay remote. A bank is
frozen before the first policy call. Refusals remain in its full denominator.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tarfile
import time

SALT='a4-head-action-screen-20261029-v1'
COUNT=256
FIELDS=('categorical_ids','numeric_features','token_type_ids','attention_mask','legal_action_mask')
SHAPES=((23,41),(23,132),(23,),(23,),(9,))

def require(ok,message):
    if not ok:raise ValueError(message)

def canonical(v):
    return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,v):
    with Path(p).open('xb') as f:
        f.write(canonical(v)+b'\n');f.flush();os.fsync(f.fileno())

def array_shape(v,shape,kind):
    if shape:
        return type(v) is list and len(v)==shape[0] and all(array_shape(x,shape[1:],kind) for x in v)
    if kind=='bool':return type(v) is bool
    return type(v) in ((int,) if kind=='int' else (int,float)) and math.isfinite(v)

def roles(row):
    requests=row['requests']; require(type(requests) is dict and set(requests)<= {'p1','p2'},'request roles')
    active=[]
    for seat in ('p1','p2'):
        request=requests.get(seat)
        if request is None:continue
        require(type(request) is dict and request.get('side',{}).get('id')==seat,'request ownership')
        if request.get('wait') or request.get('teamPreview'):continue
        if (type(request.get('forceSwitch')) is list and any(request['forceSwitch'])) or (type(request.get('active')) is list and request['active']):active.append(seat)
    require(row['requested_players']==active and bool(active) and set(row['observations'])==set(active)
            and set(row['actions'])==set(active),'exact actionable role coverage')
    for seat in active:
        obs=row['observations'][seat]
        for name,shape,kind in zip(FIELDS,SHAPES,('int','float','int','bool','bool')):
            require(array_shape(obs[name],shape,kind),'invalid recorded '+name)
        action=row['actions'][seat]
        require(type(action) is int and 0<=action<9 and obs['legal_action_mask'][action]
                and any(obs['attention_mask']),'committed action legality')
    return active

def select_roots(candidates,excluded,*,count=COUNT,salt=SALT):
    """Alternate seats; hash rank only immutable input identities, never outputs."""
    require(type(count) is int and count>0 and count%2==0,'even fixed root count')
    require(len({(r['root_id'],r['cost_seat']) for r in candidates})==len(candidates),'duplicate source root/seat')
    ranked={seat:sorted((r for r in candidates if r['cost_seat']==seat and r['root_id'] not in excluded),
        key=lambda r:digest([salt,'root',seat,r['battle_id'],r['root_id']])) for seat in ('p1','p2')}
    selected=[];used=set()
    for ordinal in range(count):
        seat='p1' if ordinal%2==0 else 'p2'
        row=next((r for r in ranked[seat] if r['root_id'] not in used),None)
        require(row is not None,'insufficient fixed TRAIN roots; no replacements')
        used.add(row['root_id']);seed=int(digest([salt,'search',row['root_id'],seat])[:16],16)
        selected.append(dict(row,cost_ordinal=ordinal,search_seed=seed))
    return selected

def freeze(plan,output):
    root=Path(plan['collection']);blob=(root/'collection.json').read_bytes()
    require(hashlib.sha256(blob).hexdigest()==plan['collection_sha256'],'collection SHA')
    manifest=json.loads(blob);cohort_blob=(root/'cohort.json').read_bytes()
    require(hashlib.sha256(cohort_blob).hexdigest()==plan['cohort_sha256']==manifest['cohort_sha256'],'cohort SHA')
    cohort=json.loads(cohort_blob)
    require(manifest['status']=='COMPLETE_FIXED_COHORT' and cohort['config']==manifest['config']
        and manifest['config']['champion_sha256']==plan['checkpoint_sha256'],'fixed champion cohort')
    require(len(cohort['battles'])==256 and len({b['battle_id'] for b in cohort['battles']})==256,'fixed battle roster')
    candidates=[];members={'collection.json':plan['collection_sha256'],'cohort.json':plan['cohort_sha256']}
    train=[b for b in cohort['battles'] if b['split']=='train'];require(len(train)==192,'original TRAIN split')
    for battle in train:
        rel=f"battle-{battle['index']:04d}/battle.json";receipt_blob=(root/rel).read_bytes()
        require(hashlib.sha256(receipt_blob).hexdigest()==manifest['battle_receipt_hashes'][rel],'battle receipt SHA')
        receipt=json.loads(receipt_blob)
        require(all(receipt[k]==battle[k] for k in ('index','battle_id','seed','split','split_key'))
            and receipt['contract']==plan['checkpoint_contract'],'recorded contract/identity')
        path=f"battle-{battle['index']:04d}/boundaries.jsonl.gz";compressed=(root/path).read_bytes()
        require(hashlib.sha256(compressed).hexdigest()==receipt['files']['boundaries.jsonl.gz'],'boundary gzip SHA')
        prefix=[];ended=False;rounds=set();root_ids=set()
        for line in gzip.decompress(compressed).splitlines():
            row=json.loads(line);require(not ended,'postterminal source record')
            require(row['protocol_start']==len(prefix),'prefix continuity');prefix.extend(row['protocol_delta'])
            require(row['protocol_end']==len(prefix),'prefix length')
            if row.get('event')=='end':ended=True;continue
            require(row['battle_id']==battle['battle_id'] and row['seed']==battle['seed']
                and row['decision_round_index']==len(rounds) and row['root_id'] not in root_ids,'unique chronological root')
            rounds.add(row['decision_round_index']);root_ids.add(row['root_id'])
            require(digest(prefix)==row['public_prefix_sha256'] and row['root_id']==digest([
                row['battle_id'],row['decision_round_index'],row['public_prefix_sha256'],row['requests']]),'source root/prefix SHA')
            for seat in roles(row):
                candidates.append(dict(index=battle['index'],battle_id=battle['battle_id'],seed=battle['seed'],split='train',
                    cost_seat=seat,decision_round_index=row['decision_round_index'],root_id=row['root_id'],boundary_sha256=digest(row),
                    observation_sha256=digest(row['observations'][seat]),protocol_prefix_sha256=row['public_prefix_sha256'],
                    requests_sha256=digest(row['requests']),battle_receipt_sha256=hashlib.sha256(receipt_blob).hexdigest(),
                    boundaries_file_sha256=hashlib.sha256(compressed).hexdigest(),boundaries_relative_path=path))
        require(ended,'complete original battle required')
        members[rel]=hashlib.sha256(receipt_blob).hexdigest();members[path]=hashlib.sha256(compressed).hexdigest()
    chosen=select_roots(candidates,set(plan['excluded_root_ids']))
    bank=dict(schema='pokezero.A4-head-action-root-bank.v1',selection_salt=SALT,count=COUNT,roots=chosen,
        selected_root_sha256=digest(chosen),collection_sha256=plan['collection_sha256'],cohort_sha256=plan['cohort_sha256'],
        excluded_root_ids=plan['excluded_root_ids'],selection='256 distinct root IDs; alternating128p1/128p2; immutable hash rank over192 TRAIN battles; no outcomes or replacements',
        source_members={n:s for n,s in members.items() if n in ('collection.json','cohort.json') or n.split('/')[0] in {r['boundaries_relative_path'].split('/')[0] for r in chosen}},
        candidate_root_seats=len(candidates),source_train_battles=192,new_games=0,search_calls=0)
    write(output,bank)
    if plan.get('source_bundle'):
        with tarfile.open(plan['source_bundle'],'x:gz') as archive:
            for name,blob in [('root-bank.json',Path(output).read_bytes())]+[
                    ('collection/'+name,(root/name).read_bytes()) for name in sorted(bank['source_members'])]:
                if name.startswith('collection/'):
                    require(hashlib.sha256(blob).hexdigest()==bank['source_members'][name[len('collection/'):]],'source changed before transfer')
                entry=tarfile.TarInfo(name);entry.size=len(blob);entry.mode=0o600;entry.mtime=0
                archive.addfile(entry,io.BytesIO(blob))
    print(json.dumps(dict(event='ROOT_BANK_FROZEN',count=COUNT,selected_root_sha256=bank['selected_root_sha256'],file_sha256=file_sha(output))),flush=True)
    return bank

class NativeWitness:
    """Observe exact ABI arguments and reply, then delegate once without edits/RNG."""
    def __init__(self,native,directory,canonical_native,*,capture_module=None,evidence_root=None):
        self.native=native;self.directory=Path(directory);self.canonical_native=canonical_native;self.rows=[]
        self.capture_module=capture_module;self.evidence_root=evidence_root
    def __getattr__(self,name):return getattr(self.native,name)
    def search_batched_multi_encoded(self,*args,**kwargs):
        values=list(args);values[6]=values[6].to_payload()
        blob=self.canonical_native(dict(args=values,kwargs=kwargs));number=len(self.rows)
        record=dict(input_sha256=hashlib.sha256(blob).hexdigest(),native_seed=args[9],started=False,completed=False)
        self.rows.append(record)
        input_path=self.directory/f'native-{number:02d}-input.json'
        with input_path.open('xb') as f:f.write(blob+b'\n');f.flush();os.fsync(f.fileno())
        capture=self.capture_module.current() if self.capture_module is not None else None
        if self.capture_module is not None:
            if capture is None:raise self.capture_module.CaptureFailed('screen native invocation has no decision capture')
            capture._screen_native_input=evidence_link(input_path,self.evidence_root)
            capture._screen_native_reply=None
        record['started']=True
        reply=self.native.search_batched_multi_encoded(*args,**kwargs)
        parsed=json.loads(reply);reply_path=self.directory/f'native-{number:02d}-reply.json';write(reply_path,parsed)
        if capture is not None:capture._screen_native_reply=evidence_link(reply_path,self.evidence_root)
        record.update(completed=True,reply_sha256=digest(parsed),semantic_sha256=digest({k:parsed.get(k) for k in
            ('side_one','side_two','root_value','iterations','model_evals','prior_branches','prior_fallbacks','depth_reached','capture_refusals')}),
            prior_fallbacks=parsed.get('prior_fallbacks',0),capture_refusal_codes=[c for r in parsed.get('capture_refusals',[]) for c in r['reason_codes']])
        return reply

def evidence_link(path,root):
    path=Path(path);root=Path(root).resolve()
    require(path.resolve().is_relative_to(root) and not path.is_symlink(),'evidence outside owned output')
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(file=path.resolve().relative_to(root).as_posix(),sha256=file_sha(path))

def read_link(link,root):
    root=Path(root).resolve();path=root/link['file']
    require(not Path(link['file']).is_absolute() and path.resolve().is_relative_to(root)
        and not path.is_symlink() and file_sha(path)==link['sha256'],'capture evidence link missing, foreign, or changed')
    return json.loads(path.read_bytes())

def install_native_refusal_batches(fc,output):
    """Screen-only capture transport, explicitly distinct from the base build.

    Keep the production single-error path. Native refusals reference the already
    durable full reply once, with one event-index batch instead of N full-root
    files. No callback, policy, RNG, native argument, or reply is changed.
    """
    output=Path(output).resolve();original=fc._capture_native_events
    def batch(capture,events):
        q=capture.quarantine
        try:
            require(type(events) is list,'native refusal events are not an array')
            for e in events:
                require(type(e) is dict and type(e.get('reason_codes')) is list and bool(e['reason_codes'])
                    and all(type(c) is str and c for c in e['reason_codes']),'malformed native refusal event')
            if not events:return
            entry=capture.fields.get('input_receipt');require(type(entry) is dict,'missing once-per-decision input receipt')
            root_link=dict(entry,file=(q.directory/entry['file']).relative_to(Path(output)).as_posix())
            root_input=read_link(root_link,output)
            fields=fc.snapshot(capture.fields)
            updates={k:v for k,v in fields.items() if k!='input_receipt' and (k not in root_input or v!=root_input[k])}
            payload=dict(schema='pokezero.screen-native-refusal-batch.v1',build=q.build,
                captured_unix=time.time(),event_time_semantics='one batch timestamp; no individual event times claimed',
                root_input=root_link,root=fields['root'],field_updates=updates,
                lazy_evidence={name:fc.snapshot(provider()) for name,provider in capture._lazy.items()},
                event_count=len(events),event_index=[dict(index=i,reason_codes=e['reason_codes']) for i,e in enumerate(events)],
                events_sha256=digest(events),evidence_root=str(Path(output)))
            reply_link=getattr(capture,'_screen_native_reply',None)
            input_link=getattr(capture,'_screen_native_input',None)
            if input_link is not None:read_link(input_link,output);payload['native_input']=input_link
            if reply_link is not None:
                reply=read_link(reply_link,output)
                require(reply.get('capture_refusals')==events,'native reply/event array mismatch')
                payload.update(native_reply=reply_link,events_json_pointer='/capture_refusals')
            else:
                # Native exceptions can supply witnesses without a returned reply.
                # Preserve that entire event array once, never omit its witnesses.
                payload['inline_events']=fc.snapshot(events)
            q.events+=len(events)
            for e in events:q.reason_counts.update(e['reason_codes'])
            link=q.write('native-refusal-batch',payload)
            link.update(reason_codes=sorted({c for e in events for c in e['reason_codes']}),
                event_count=len(events),event_indices=list(range(len(events))))
            q.successes+=len(events);capture.receipts.append(link)
        except BaseException as error:
            if isinstance(error,fc.CaptureFailed):raise fc._count_capture_loss(q,error)
            raise fc._count_capture_loss(q,fc.CaptureFailed('screen native refusal batch capture failed: '+str(error))) from error
    fc._capture_native_events=batch
    return original

def decision_signature(result):
    return (result['status'],result['action'],[(x['input_sha256'],x.get('semantic_sha256')) for x in result['native_calls']])

def verify_prior_native_files(directory,result):
    for i,call in enumerate(result['native_calls']):
        require(call['started'] is True and call['completed'] is True,'prior consumed call is incomplete; never replay it')
        blob=(directory/f'native-{i:02d}-input.json').read_bytes()
        reply=json.loads((directory/f'native-{i:02d}-reply.json').read_bytes())
        require(blob.endswith(b'\n') and hashlib.sha256(blob[:-1]).hexdigest()==call['input_sha256']
            and digest(reply)==call['reply_sha256'] and digest({k:reply.get(k) for k in
            ('side_one','side_two','root_value','iterations','model_evals','prior_branches','prior_fallbacks','depth_reached','capture_refusals')})==call['semantic_sha256'],
            'prior decision/native evidence mismatch')

def repeat_record(ordinal,head,original,again,*,context_version='saved-own-public-history-v2'):
    return dict(ordinal=ordinal,head=head,status='ATTEMPTED',context_version=context_version,native_calls=sum(x['started'] for x in again['native_calls']),
        both_clean_search=original['status']==again['status']=='SEARCHED',
        action_and_root_arm_repeat_equal=decision_signature(original)==decision_signature(again))

def restore_checkpoint(plan,bank):
    """Restore immutable facts only; consumed roots are never reconstructed."""
    c=plan['continuation'];blob=Path(c['checkpoint']).read_bytes()
    require(hashlib.sha256(blob).hexdigest()==c['checkpoint_sha256'],'checkpoint bytes changed')
    checkpoint=json.loads(blob);prior=Path(c['prior_output'])
    require(checkpoint['prior_output']==str(prior) and checkpoint['fixed_bank_sha256']==plan['root_bank_sha256']
        and checkpoint['completed_root_ordinals']==[0] and checkpoint['consumed_partial_root_ordinals']==[1]
        and checkpoint['continue_only_root_ordinals']==list(range(2,COUNT)),'fixed stopped-run continuation scope')
    require(checkpoint['prior_terminal_summary']['completed_root_rows']==2
        and checkpoint['prior_terminal_summary']['failure_capture_telemetry']['capture_failures']==1,'stopped run failure provenance')
    actual_slots=sorted(p.name for p in prior.glob('root-*') if p.is_dir())
    require(actual_slots==sorted(s['directory'] for s in checkpoint['slots']),'prior attempt roster changed')
    require(all(not (prior/name).is_symlink() and all(p.is_file() and not p.is_symlink() for p in (prior/name).iterdir())
        for name in actual_slots),'prior query contains a foreign link or nested attempt')
    actual_query_files={p.relative_to(prior).as_posix() for name in actual_slots for p in (prior/name).iterdir() if p.is_file()}
    require(actual_query_files=={n for n in checkpoint['files'] if n.split('/')[0] in actual_slots},'prior query file roster changed')
    for name,item in checkpoint['files'].items():
        path=prior/name
        require(not Path(name).is_absolute() and path.resolve().is_relative_to(prior.resolve())
            and not path.is_symlink() and path.stat().st_size==item['bytes'] and file_sha(path)==item['sha256'],'prior immutable evidence changed: '+name)
    require(file_sha(prior/'root-bank.json')==plan['root_bank_sha256'],'prior bank differs')
    runtime=json.loads((prior/'runtime-and-config.json').read_bytes())
    require(runtime['build']==plan['runtime_capture_build'] and runtime['search_config']==plan['search_config']
        and runtime['numeric_mode']=='FP32_MHAOFF_TF32OFF_HIGHEST'
        and runtime['models']=={n:v['sha256'] for n,v in plan['immutable_inputs'].items()},'prior runtime/model/numeric/search identity changed')
    expected_slots={f'root-000-{h}-{label}' for h in ('champion','refit') for label in ('primary','determinism')}|{
        'root-001-champion-primary','root-001-refit-primary','root-001-champion-determinism'}
    require(set(actual_slots)==expected_slots,'unexpected consumed head/repeat slots; never infer a retry')
    row=json.loads((prior/'row-000.json').read_bytes())
    require(row['ordinal']==0 and row['root_id']==bank['roots'][0]['root_id'],'completed root identity')
    repeats=[]
    for head in ('champion','refit'):
        original=json.loads((prior/f'root-000-{head}-primary/decision.json').read_bytes())
        again=json.loads((prior/f'root-000-{head}-determinism/decision.json').read_bytes())
        verify_prior_native_files(prior/f'root-000-{head}-primary',original)
        verify_prior_native_files(prior/f'root-000-{head}-determinism',again)
        require(original==row['heads'][head],'completed primary/row mismatch')
        repeats.append(repeat_record(0,head,original,again,context_version='legacy-empty-trajectory-v1'))
    row.update(context_version='legacy-empty-trajectory-v1',prior_row_sha256=checkpoint['files']['row-000.json']['sha256'],prior_evidence_root=str(prior))
    root=bank['roots'][1];interrupted=json.loads((prior/'root-001-champion-determinism/decision.json').read_bytes())
    require(interrupted['status']=='CAPTURE_FAILED','interrupted root must stay unknown')
    verify_prior_native_files(prior/'root-001-champion-determinism',interrupted)
    unknown=json.loads((prior/'row-001.json').read_bytes())
    require(unknown['ordinal']==1 and unknown['root_id']==root['root_id'],'interrupted repeat root identity')
    for head in ('champion','refit'):
        original=json.loads((prior/f'root-001-{head}-primary/decision.json').read_bytes())
        verify_prior_native_files(prior/f'root-001-{head}-primary',original)
        require(original==unknown['heads'][head] and type(original['action']) is int,'completed interrupted-root primary mismatch')
    unknown.update(continuation_status='INTERRUPTED_REPEAT_PRIMARY_COMPLETE_WITH_NATIVE_REFUSALS',
        context_version='legacy-empty-trajectory-v1',
        prior_row_sha256=checkpoint['files']['row-001.json']['sha256'],prior_evidence_root=str(prior))
    for head in ('champion','refit'):repeats.append(dict(ordinal=1,head=head,
        status='CAPTURE_FAILED_CONSUMED_ROOT_UNKNOWN' if head=='champion' else 'NOT_ATTEMPTED_CONSUMED_ROOT_UNKNOWN',
        context_version='legacy-empty-trajectory-v1',
        native_calls=sum(x['started'] for x in interrupted['native_calls']) if head=='champion' else 0,
        both_clean_search=False,action_and_root_arm_repeat_equal=None))
    return [row,unknown],repeats,checkpoint

def remaining_fixed_roots(bank,retained_rows):
    require([r['ordinal'] for r in retained_rows]==[0,1]
        and [r['root_id'] for r in retained_rows]==[r['root_id'] for r in bank['roots'][:2]],'never restart a consumed fixed root')
    return bank['roots'][2:]

def reconstruct_screen_root(config,collection,root):
    """Rebuild only original own/public pre-root history, never a simulator.

    Protocol delta at boundary N completes the public action round N-1. Own
    observations/actions are appended only for strictly earlier requests. The
    current/future committed actions, peer observations/requests, and outcomes
    never enter the trajectory supplied to either head.
    """
    require(type(root['cost_ordinal']) is int and 2<=root['cost_ordinal']<COUNT,'consumed roots must never be reconstructed')
    require(root.get('split')=='train' and root.get('cost_seat') in ('p1','p2'),'fixed TRAIN owning root required')
    from pokezero.recorded_mirror_roots import new_recorded_environment,feed_recorded_boundary
    from pokezero.trajectory import BattleTrajectory,TrajectoryStep
    from pokezero.public_action_capture import public_action_round_from_protocol_lines,append_public_action_round
    base=Path(collection).resolve(strict=True);path=(base/root['boundaries_relative_path']).resolve(strict=True)
    require(path.is_relative_to(base) and file_sha(path)==root['boundaries_file_sha256'],'frozen source boundary artifact changed')
    trajectory=BattleTrajectory(root['battle_id'],'gen3randombattle',root['seed']);env=new_recorded_environment(config,seed=root['seed']);previous=None
    try:
        with gzip.open(path,'rb') as stream:
            for line in stream:
                row=json.loads(line)
                require(row.get('event') is None and row['battle_id']==root['battle_id'] and row['seed']==root['seed'],'selected root absent before terminal or source identity changed')
                require(not any(v.startswith('|win|') or v in ('|tie','|tie|') for v in row['protocol_delta']),'terminal information before selected root')
                if previous is not None:
                    public_lines=[v for v in row['protocol_delta'] if not v.startswith('|request|')]
                    append_public_action_round(trajectory,public_action_round_from_protocol_lines(public_lines,
                        turn_index=previous['round'],requested_players=previous['requested']))
                observations=feed_recorded_boundary(env,row)
                if row['decision_round_index']==root['decision_round_index']:
                    require(digest(row)==root['boundary_sha256'] and row['root_id']==root['root_id']
                        and digest(row['requests'])==root['requests_sha256']
                        and digest(row['observations'][root['cost_seat']])==root['observation_sha256'],'frozen root or observation changed')
                    return env,observations[root['cost_seat']],trajectory
                require(row['decision_round_index']<root['decision_round_index'],'selected root missing; no replacement')
                seat=root['cost_seat']
                if seat in row['requested_players']:
                    observation=observations[seat]
                    trajectory.append(TrajectoryStep(player_id=seat,turn_index=row['decision_round_index'],observation=observation,
                        legal_action_mask=tuple(observation.legal_action_mask),action_index=row['actions'][seat]))
                previous=dict(round=row['decision_round_index'],requested=tuple(row['requested_players']))
        raise ValueError('selected root absent; no replacement')
    except BaseException:
        env.close();raise

def persist_row(output,row,*,retained=False):
    write(output/f"row-{row['ordinal']:03d}.json",row)
    compact={k:row[k] for k in ('ordinal','root_id','seat','search_seed','paired_clean_search','chosen_actions_differ')}
    compact['heads']={h:{'status':v['status'],'action':v['action'],
        'native_input_sha256':[x['input_sha256'] for x in v['native_calls']],
        'native_calls_started':sum(x['started'] for x in v['native_calls']),
        'native_prior_refusals':sum(x.get('prior_fallbacks',0) for x in v['native_calls'])} for h,v in row['heads'].items()}
    compact['row_sha256']=file_sha(output/f"row-{row['ordinal']:03d}.json")
    with (output/'actions.jsonl').open('ab') as stream:
        stream.write(canonical(compact)+b'\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(event='SCREEN_ROOT_RETAINED' if retained else 'SCREEN_ROOT_COMPLETED',ordinal=row['ordinal'],
        paired_clean=row['paired_clean_search'],diff=row['chosen_actions_differ'])),flush=True)

def summarize(rows,count=COUNT):
    require(len(rows)==count and [r['ordinal'] for r in rows]==list(range(count)),'whole frozen roster denominator')
    legacy=[r for r in rows if r.get('context_version')=='legacy-empty-trajectory-v1']
    corrected=[r for r in rows if r not in legacy]
    clean=[r for r in corrected if r['paired_clean_search']]
    different=sum(r['chosen_actions_differ'] is True for r in clean)
    unresolved=count-len(clean)
    returned=[r for r in corrected if all(type(r['heads'][h].get('action')) is int for h in ('champion','refit'))]
    legacy_returned=[r for r in legacy if all(type(r['heads'][h].get('action')) is int for h in ('champion','refit'))]
    returned_different=sum(r['heads']['champion']['action']!=r['heads']['refit']['action'] for r in returned)
    return dict(frozen_roots=count,paired_clean_searches=len(clean),chosen_actions_differ=different,
        corrected_context_roots=len(corrected),legacy_empty_context_roots=len(legacy),legacy_empty_context_ordinals=[r['ordinal'] for r in legacy],
        legacy_observed_returned_pairs=len(legacy_returned),legacy_observed_action_differences=sum(r['heads']['champion']['action']!=r['heads']['refit']['action'] for r in legacy_returned),
        legacy_context_unknown_for_corrected_metric=True,all_returned_pairs=len(returned),all_returned_chosen_actions_differ=returned_different,
        difference_fraction_among_returned_corrected_pairs=returned_different/len(returned) if returned else None,
        full_roster_returned_action_difference_lower=returned_different/count,
        full_roster_returned_action_difference_upper=(returned_different+count-len(returned))/count,
        difference_fraction_among_clean_pairs=different/len(clean) if clean else None,
        full_roster_difference_fraction_lower=different/count,full_roster_difference_fraction_upper=(different+unresolved)/count,
        unresolved_roots=unresolved,head_refusals={h:sum(r['heads'][h]['status']!='SEARCHED' for r in rows) for h in ('champion','refit')},
        paired_native_input_mismatches=sum(r.get('paired_native_input_identity') is False and
            all(r['heads'][h]['status']=='SEARCHED' for h in ('champion','refit')) for r in rows),
        full_roster_complete=True,no_foulplay_panel=True,not_strength_or_value_improvement=True,
        source_visibility='TRUSTED_SAVED_MIXED_TRANSCRIPT_NOT_CAPTURED_OWNING_WIRE',public_wire_qualified=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('freeze','run'));parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_bytes())
    if args.mode=='freeze':freeze(plan,plan['root_bank']);return 0
    return run(plan)


def prepare_capture_admission(plan,build,output):
    """Bind existing closed-run metadata/fresh R4 fixtures, never replay inputs."""
    from pokezero.failure_capture import require_capture_admission
    base=Path(plan['admission_parent']);legacy=base/plan['legacy_capture_relative_root']
    original=legacy/'gate0-inventory-r2.json'
    require(file_sha(original)==plan['legacy_inventory_sha256'],'closed inventory identity')
    inventory=json.loads(original.read_bytes());known=set(inventory['reason_codes'])
    # Ordinary library admission validates its declared, build-qualified classes.
    # Later historical classes are disclosed, not silently claimed qualified or
    # made an extra screen veto. Every actual screen refusal stays in its roster.
    history=[]
    for p in (legacy/'quarantine/engine-mcts-history').rglob('failure-*.json'):
        blob=p.read_bytes();codes=json.loads(blob)['reason_codes']
        require(type(codes) is list and bool(codes) and all(type(c) is str and c for c in codes),'invalid historical refusal metadata')
        history.append(dict(file=str(p.relative_to(legacy)),sha256=hashlib.sha256(blob).hexdigest(),reason_codes=codes))
    history.sort(key=lambda row:row['file'])
    extra=sorted({c for row in history for c in row['reason_codes']}-known)
    disclosure=dict(history_files=history,qualified_fixture_reason_codes=sorted(known),
        additional_historical_reason_codes_not_requalified=extra,all_historical_classes_qualified=not extra,
        no_historical_replay=True,scope='Declared R4 fixture admission; additional historical classes remain unqualified. Actual screen refusals are retained, not scored as clean searches.')
    write(output/'capture-history-disclosure.json',disclosure)
    inventory=copy.deepcopy(inventory)
    inventory['history_scope']='Closed Gate0-r1 declared two-class inventory only; later history is separately hash-disclosed without a qualification claim.'
    for row in inventory['source_runs']:
        for key in ('bundle','diagnostic_receipt','terminal'):
            row[key]['file']=plan['legacy_capture_relative_root']+'/'+row[key]['file']
    name=output.name+'-observed-inventory.json';write(base/name,inventory)
    fresh=Path(plan['fresh_capture_fixtures'])
    report=json.loads((fresh/'current-capture-source-verification.json').read_bytes())
    require(file_sha(fresh/'current-capture-source-verification.json')==plan['fresh_capture_verification_sha256']
        and report['build']==build,'R4 source-fixture identity')
    rows=[]
    for row in report['fixtures']:
        key='renderer' if row['reason_codes'][0].startswith('renderer.') else 'contract'
        fixture=dict(reason_codes=row['reason_codes'])
        fixture['bundle']={'file':plan['legacy_capture_relative_root']+'/'+row['original_bundle']['file'],'sha256':row['original_bundle']['sha256']}
        fixture['diagnostic_receipt']={'file':plan['legacy_capture_relative_root']+'/quarantine/gate0-r1-recovery/attempt2/'+key+'-diagnostic.json','sha256':row['original_diagnostic']['sha256']}
        for field in ('fixture','verification'):
            path=fresh/row[field]['file'];require(file_sha(path)==row[field]['sha256'],'fresh fixture bytes')
            fixture[field]={'file':str(path.relative_to(base)),'sha256':row[field]['sha256']}
        rows.append(fixture)
    admission=dict(schema='pokezero.capture-admission.v1',build_fingerprint=build['fingerprint'],
        observed_inventory={'file':name,'sha256':file_sha(base/name)},observed_reason_codes=sorted(known),fixtures=rows)
    path=base/(output.name+'-capture-admission.json');write(path,admission)
    proof=require_capture_admission(path,build=build,scientific_path='engine-mcts',capture_directory=output/'quarantine')
    os.environ['POKEZERO_CAPTURE_ADMISSION']=str(path)
    return dict(proof,history_disclosure_sha256=file_sha(output/'capture-history-disclosure.json'),
        additional_historical_reason_codes_not_requalified=extra,all_historical_classes_qualified=not extra)


def run(plan):
    # The scientific producer runs against the existing qualified library tree;
    # its own standalone source does not replace that package or its fingerprint.
    sys.path[:0]=[str(Path(plan['runtime'])/'repo/src'),str(Path(plan['runtime'])/'repo')]
    os.environ['POKEZERO_SOURCE_COMMIT']=plan['runtime_source_commit']
    import torch
    from pokezero import failure_capture as fc
    from pokezero.failure_capture import runtime_build,canonical as native_canonical,CaptureFailed
    from pokezero.engine_search import EngineMctsConfig,EngineMctsPolicy
    from pokezero.neural_policy import (load_transformer_checkpoint_payload,parse_transformer_model_config,
        category_vocab_from_model_config,feature_masks_from_model_config,observation_spec_from_model_config)
    from pokezero.local_showdown import LocalShowdownConfig,env_config_from_checkpoint_provenance,_public_materialization_payload
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.policy import PolicyContext
    from pokezero.trajectory import trajectory_to_dict
    output=Path(plan['output']);output.mkdir(exist_ok=False)
    summary={'status':'FAILED_OR_PARTIAL_NO_RETRY','fixed_roots':COUNT,'no_foulplay_panel':True};rows=[];repeats=[];quarantine=None;original_capture=None
    try:
        require(type(plan.get('continuation')) is dict,'this fixed bank requires its stopped-run checkpoint; never restart consumed roots')
        require(plan.get('history_context')=='saved-own-public-history-v2','explicit corrected pre-root history required')
        require(all(os.environ.get(k) is None for k in ('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','NVIDIA_TF32_OVERRIDE')),'precision override present')
        torch.set_num_threads(plan['cpu_threads']);torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)
        require(torch.__version__=='2.13.0+cu130' and torch.version.cuda=='13.0' and torch.cuda.device_count()==1,'qualified CUDA ABI/device count')
        require('B300' in torch.cuda.get_device_name(0) or 'GB300' in torch.cuda.get_device_name(0),'one GB300')
        build=runtime_build();require(build==plan['runtime_capture_build'],'qualified runtime source/binaries changed')
        summary.update(base_qualified_runtime_fingerprint=build['fingerprint'],scientific_producer_sha256=file_sha(__file__),
            capture_transport=plan.get('capture_transport'),capture_adapter_is_new_source=True)
        compiled=Path(plan['showdown'])/'dist';tree=hashlib.sha256();files=sorted(p for p in compiled.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in ('.js','.json'))
        for path in files:
            tree.update(path.relative_to(compiled).as_posix().encode()+b'\0');tree.update(bytes.fromhex(file_sha(path)))
        require(dict(files=len(files),sha256=tree.hexdigest())==plan['showdown_compiled_tree'],'qualified Showdown compiled bytes changed')
        os.environ['POKEZERO_QUARANTINE_DIR']=str(output/'quarantine')
        summary['capture_admission']=prepare_capture_admission(plan,build,output)
        models=output/'models';models.mkdir()
        private={}
        for name,item in plan['immutable_inputs'].items():
            blob=Path(item['path']).read_bytes();require(hashlib.sha256(blob).hexdigest()==item['sha256'],'immutable artifact drift: '+name)
            target=models/name
            with target.open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
            private[name]=target
        bank_blob=Path(plan['root_bank']).read_bytes()
        require(hashlib.sha256(bank_blob).hexdigest()==plan['root_bank_sha256'],'bank must be frozen and byte-bound before any search')
        bank=json.loads(bank_blob)
        require(bank['count']==COUNT and bank['selected_root_sha256']==digest(bank['roots'])
            and bank['collection_sha256']==plan['collection_sha256'] and bank['cohort_sha256']==plan['cohort_sha256']
            and bank['selection_salt']==SALT and bank['excluded_root_ids']==plan['excluded_root_ids'],'frozen bank identity')
        require(len({r['root_id'] for r in bank['roots']})==COUNT and [r['cost_ordinal'] for r in bank['roots']]==list(range(COUNT))
            and [r['cost_seat'] for r in bank['roots']]==['p1','p2']*(COUNT//2),'fixed distinct root/seat roster')
        write(output/'root-bank.json',bank)
        require(plan.get('capture_transport')=='native-refusal-batch-v1','screen capture adapter must be explicit')
        original_capture=install_native_refusal_batches(fc,output)
        if plan.get('continuation'):
            rows,repeats,checkpoint=restore_checkpoint(plan,bank)
            write(output/'prior-checkpoint.json',checkpoint)
            for row in rows:persist_row(output,row,retained=True)
            summary.update(continuation_checkpoint_sha256=plan['continuation']['checkpoint_sha256'],
                prior_capture_telemetry=checkpoint['prior_terminal_summary']['failure_capture_telemetry'],
                continued_only_root_ordinals=list(range(2,COUNT)),never_reexecuted_roots=[0,1])
        checkpoint=private['checkpoint.pt'];cfg=parse_transformer_model_config(load_transformer_checkpoint_payload(checkpoint))
        config=env_config_from_checkpoint_provenance(LocalShowdownConfig(showdown_root=Path(plan['showdown']),node_binary=plan['node'],set_belief_source=True),
            feature_masks_from_model_config(cfg),context='A4 fixed TRAIN chosen-action screen trusted saved source',
            required_specs=observation_spec_from_model_config(cfg),required_vocabs=category_vocab_from_model_config(cfg,Path(plan['showdown'])))
        dex=load_showdown_dex_cached(Path(plan['showdown']));source=load_gen3_randbat_source_cached(Path(plan['showdown']))
        require(source.metadata.source_hash==plan['checkpoint_contract']['belief_set_source_hash'],'belief source binding')
        policies={};handles={}
        for head in ('champion','refit'):
            engine=EngineMctsConfig(leaf_eval='model',checkpoint_path=str(checkpoint),model_path=str(private[head+'.pt']),
                tables_path=str(private['encoder_tables.json']),model_device='cuda',**plan['search_config'])
            policy=EngineMctsPolicy(dex=dex,set_source=source,config=engine,policy_id='a4-'+head,scientific_capture=True)
            quarantine=policy._quarantine
            policy.warm_model_runtime();policies[head]=policy;handles[head]=policy._native_model
        write(output/'runtime-and-config.json',dict(build=build,device=torch.cuda.get_device_name(0),
            models={k:file_sha(v) for k,v in private.items()},numeric_mode='FP32_MHAOFF_TF32OFF_HIGHEST',
            search_config=plan['search_config'],historical_config_sha256=plan['historical_config_sha256'],
            scientific_producer_sha256=file_sha(__file__),capture_transport=plan['capture_transport'],
            history_context=plan['history_context'],
            capture_adapter_scope='Screen-only native refusal transport differs from untouched qualified base package/native/model build.',
            continuation_checkpoint_sha256=summary.get('continuation_checkpoint_sha256')))

        def evaluate(root,head,base_context,run_label):
            directory=output/f"root-{root['cost_ordinal']:03d}-{head}-{run_label}";directory.mkdir()
            policy=policies[head];policy.reset();context=copy.deepcopy(base_context)
            witness=NativeWitness(handles[head],directory,native_canonical,capture_module=fc,evidence_root=output);policy._native_model=witness
            result=dict(status='REFUSED',action=None,native_calls=[],reason=None)
            try:
                decision=policy.select_action_with_context(context,rng=random.Random(root['search_seed']))
                metadata=dict(decision.metadata or {});fallback=bool(metadata.get('engine_mcts',{}).get('fallback'))
                # Production uses both a string fallback and root_puct_fallback
                # depending on call site; inspect the production policy family too.
                fallback=fallback or bool(metadata.get('engine_mcts_fallback')) or bool(metadata.get('fallback'))
                result.update(action=decision.action_index,metadata=metadata)
                result['status']='SEARCHED' if witness.rows and all(x['completed'] for x in witness.rows) and not fallback else 'FALLBACK_OR_NO_COMPLETED_SEARCH'
                if result['status']=='SEARCHED' and any(x['prior_fallbacks'] or x['capture_refusal_codes'] for x in witness.rows):
                    result['status']='SEARCHED_WITH_NATIVE_PRIOR_REFUSALS'
                require(type(decision.action_index) is int and 0<=decision.action_index<9
                    and context.observation.legal_action_mask[decision.action_index],'selected action illegal')
            except CaptureFailed as error:
                result.update(status='CAPTURE_FAILED',reason=type(error).__name__)
                raise
            except Exception as error:
                result.update(status='REFUSED',reason=type(error).__name__+': '+str(error)[:400])
            finally:
                result['native_calls']=witness.rows;write(directory/'decision.json',result);policy._native_model=handles[head]
            return result

        for root in remaining_fixed_roots(bank,rows):
            row=dict(ordinal=root['cost_ordinal'],root_id=root['root_id'],seat=root['cost_seat'],search_seed=root['search_seed'],
                paired_clean_search=False,chosen_actions_differ=None,heads={},context_version=plan['history_context'])
            env=None
            try:
                env,observation,trajectory=reconstruct_screen_root(config,plan['collection'],root)
                public=env.public_materialization_state(root['cost_seat'])
                require(not public.replay.requests,'peer requests in public materialization')
                context=PolicyContext(player_id=root['cost_seat'],decision_round_index=root['decision_round_index'],
                    battle_id=root['battle_id'],format_id='gen3randombattle',seed=root['seed'],observation=observation,
                    requested_players=tuple(env.requested_players()),trajectory=trajectory,
                    requested_legal_action_masks={root['cost_seat']:tuple(observation.legal_action_mask)},
                    requested_observations={root['cost_seat']:observation},public_materialization_state=public)
                row['public_input_sha256']=digest(_public_materialization_payload(public))
                history=fc.snapshot(trajectory_to_dict(trajectory));write(output/f"history-{root['cost_ordinal']:03d}.json",history)
                row.update(history_context_sha256=digest(history),history_own_step_count=len(trajectory.steps),
                    history_public_round_count=len(trajectory.metadata.get('public_resolved_action_rounds',[])),
                    legal_action_count=sum(observation.legal_action_mask))
                for head in ('champion','refit'):row['heads'][head]=evaluate(root,head,context,'primary')
                a,b=(row['heads'][head] for head in ('champion','refit'))
                row['paired_native_input_identity']=bool(a['native_calls']) and [x['input_sha256'] for x in a['native_calls']]==[x['input_sha256'] for x in b['native_calls']]
                row['paired_clean_search']=a['status']==b['status']=='SEARCHED' and row['paired_native_input_identity']
                if row['paired_clean_search']:row['chosen_actions_differ']=a['action']!=b['action']
                # Predeclared first8 roots only, no data-dependent repeats/tuning.
                if root['cost_ordinal']<8:
                    for head in ('champion','refit'):
                        again=evaluate(root,head,context,'determinism')
                        original=row['heads'][head]
                        repeats.append(repeat_record(root['cost_ordinal'],head,original,again))
            except CaptureFailed:raise
            except Exception as error:
                row['root_refusal']=type(error).__name__+': '+str(error)[:400]
            finally:
                if env is not None:env.close()
                for head in ('champion','refit'):row['heads'].setdefault(head,dict(status='NOT_COMPLETED_SEE_REMOTE_FAILURE_CAPTURE',action=None,native_calls=[]))
                persist_row(output,row);rows.append(row)
        repeat_keys={(r['ordinal'],r['head']) for r in repeats}
        require(len(repeat_keys)==len(repeats) and repeat_keys<={(i,h) for i in range(8) for h in ('champion','refit')},'duplicate/foreign repeat address')
        for ordinal in range(8):
            for head in ('champion','refit'):
                if (ordinal,head) not in repeat_keys:repeats.append(dict(ordinal=ordinal,head=head,status='NOT_ATTEMPTED_AFTER_ROOT_REFUSAL',context_version=plan['history_context'],native_calls=0,both_clean_search=False,action_and_root_arm_repeat_equal=None))
        repeats.sort(key=lambda r:(r['ordinal'],r['head']))
        summary.update(summarize(rows),status='COMPLETE_FIXED_ROOT_SCREEN',determinism_controls=repeats,
            all_fixed_repeats_equal=len(repeats)==16 and all(r['context_version']==plan['history_context'] and r['both_clean_search'] and r['action_and_root_arm_repeat_equal'] for r in repeats),
            fixed_repeat_denominator=16,completed_repeat_cells=sum(r.get('status')=='ATTEMPTED' for r in repeats),
            corrected_completed_repeat_cells=sum(r.get('status')=='ATTEMPTED' and r['context_version']==plan['history_context'] for r in repeats),
            legacy_completed_repeat_cells=sum(r.get('status')=='ATTEMPTED' and r['context_version']=='legacy-empty-trajectory-v1' for r in repeats),
            native_started_repeat_cells=sum(r['native_calls']>0 for r in repeats),
            selected_root_sha256=bank['selected_root_sha256'],native_search_calls_primary=sum(x['started'] for r in rows for v in r['heads'].values() for x in v['native_calls']),
            native_search_calls_repeats=sum(r['native_calls'] for r in repeats),
            full_rows_sha256=digest(rows),no_new_games=True,no_strength_claim=True)
        summary['compact_actions_sha256']=file_sha(output/'actions.jsonl')
        require(all(file_sha(private[n])==item['sha256'] for n,item in plan['immutable_inputs'].items()),'private model/table snapshot changed')
        return 0
    except BaseException as error:
        summary.update(error_type=type(error).__name__,error=str(error)[:600],full_roster_complete=False,
            unattempted_or_incomplete_roots=COUNT-len(rows),no_retry=True)
        raise
    finally:
        if original_capture is not None:fc._capture_native_events=original_capture
        if quarantine is not None:summary['failure_capture_telemetry']=quarantine.telemetry()
        summary['determinism_controls']=repeats;summary['fixed_repeat_denominator']=16
        summary['completed_root_rows']=len(rows);write(output/'summary.json',summary)
        print(json.dumps(dict(event='A4_ACTION_SCREEN_TERMINAL',**summary)),flush=True)

if __name__=='__main__':raise SystemExit(main())
