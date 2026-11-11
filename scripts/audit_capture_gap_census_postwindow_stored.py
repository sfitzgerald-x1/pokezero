"""Bounded stdlib stored-result consistency audit; no project/runtime imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

SCHEMA='pokezero.capture_gap_census.diagnostic.v1'

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def digest(value):
    require(type(value)is str and len(value)==64 and all(c in '0123456789abcdef'for c in value),'literal SHA256 required')
    return value

def number(value,label):
    require(type(value)in(int,float)and math.isfinite(value),'finite '+label+' required')
    return value

def count(value,label):
    require(type(value)is int and value>=0,'nonnegative integer '+label+' required')
    return value

def flag(value,label):
    require(type(value)is bool,'bool '+label+' required')
    return value

def close(a,b,label):
    require(math.isclose(number(a,label),number(b,label),rel_tol=1e-10,abs_tol=1e-7),'stored arithmetic differs: '+label)

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def policy_work(s,step):
    """Reviewed producer's literal pre-policy count; no postgap requalification."""
    n=count(s['policy_input_capture_valid_rows'],'pre-policy capture-valid rows')
    require(n<=count(s['active_engine_rows'],'active rows'),'pre-policy valid rows exceed active')
    skipped=flag(s['policy_work_skipped'],'policy work skipped')
    require(skipped==(step>=32 and n==0),'policy skip outside exact postwindow/no-valid boundary')
    require(s['policy_work_skip_reason']==('POST_TIMING_WINDOW_NO_ACTIVE_CAPTURE_VALID_GAME'if skipped else None),'policy skip reason differs')
    physical=count(s['bf16_physical_owner_rows'],'physical owners');admit=count(s['admitted_policy_owner_rows'],'admitted owners');shadow=count(s['shadow_discarded_owner_rows'],'discarded owners')
    encoder=count(s['shadow_actual_encoder_rows'],'actual encoder rows')
    if skipped:
        require((physical,admit,shadow,encoder)==(0,0,0,0),'skipped work gained rows')
        require('bf16_forward_ms'not in s and 'bf16_compile_ms'not in s and s['shadow_actual_encoder_ms']==[],'unexecuted work gained measured time')
        require(s['phases_ms']['original_request_rows']==0,'skipped policy gained retained driver rows')
    else:
        require(physical==4096 and admit+shadow==4096 and encoder==2048,'actual model/encoder shape differs')
    require(admit<=2*s['active_engine_rows'],'admitted/discarded geometry differs')
    require(admit<=2*n,'admitted owners exceed pre-policy capture-valid inputs')
    return skipped


def check(data,p):
    require(data.get('schema')==SCHEMA,'populated original census ledger required')
    require(p['batch']==2048 and p['census_games']==512 and p['timing_only_rows']==[512,2048],'original2048/512/1536 declaration required')
    require(data['source']['commit']==p['source_commit'],'source commit differs')
    package=data['source']['package_sha256'];require(type(package)is dict and all(type(k)is str and '/'not in k for k in package),'original package filename map required')
    require(canonical({k:digest(v)for k,v in package.items()})==p['actual_source_map_sha256'],'original package map differs')
    digest(data['pool_receipt']['pool_sha256']);require('reuse'not in p,'fresh census cannot use reused-pool declaration')
    pg=p['pool_generation'];require(pg['new_output_only']is True and pg['no_reuse']is True and pg['producer']=='gpu_engine.team_pool'and count(pg['teams'],'declared teams')==4096 and count(pg['seed_start'],'declared team seed')==p['team_seed_start'],'fresh pool declaration differs')
    require(data['constructor_seed']==p['constructor_seed']and data['sampling_seed']==p['sampling_seed'],'original seed declaration differs')
    plan=data['plan'];require(plan['batch']==2048 and plan['census_games']==512 and plan['timing_warmup']+plan['timing_steps']==32,'producer plan geometry differs')
    require(plan['fallback_seed']==p['fallback_seed']and plan['max_steps']<=2048 and plan['max_turn']<=1000,'original finite diagnostic coordinates differ')
    rows=data['cohort'];require(len(rows)==2048,'full2048 cohort required')
    require(data['original_stream_ids']==list(range(p['stream_start'],p['stream_start']+2048)),'envelope stream roster differs')
    teams=data['original_team_indices'];require(len(teams)==2048 and all(type(x)is list and len(x)==2 for x in teams),'declared teamindex geometry differs')
    require([i for pair in teams for i in pair]==list(range(4096)),'original ordered distinct pool teamindices differ')
    terminals=[]
    for i,row in enumerate(rows):
        require(row['stream_id']==p['stream_start']+i and row['role']==('census'if i<512 else'timing-only'),'original cohort order differs')
        for key in ('engine_failed','engine_terminal','capped','observation_scoring_valid'):flag(row[key],key)
        steps=count(row['executed_steps'],'row steps');require(steps<=(plan['max_steps']if i<512 else 32),'row step cap exceeded')
        gap=row['first_capture_gap'];require(row['observation_scoring_valid']==(gap is None),'capture gap regained observation validity')
        if gap is not None:require(count(gap['step'],'first gap')<plan['max_steps'],'first gap beyond declaration')
        term=row['terminal'];require((term is not None)==row['engine_terminal'],'terminal snapshot availability differs')
        if term is not None:
            require(term['done']is True and type(term['winner'])is int and term['winner']in(0,1,2),'literal terminal done/winner invalid')
            require(term['stream_id']==row['stream_id'],'terminal identity differs')
            turn=count(term['turn'],'terminal turn');require(turn<=plan['max_turn'],'terminal beyond declared turn cap')
            hp,mx=term['hp'],term['max_hp'];require(len(hp)==len(mx)==2 and all(len(x)==6 for x in hp+mx),'literal terminal12HP geometry differs')
            for h,m in zip(sum(hp,[]),sum(mx,[])):require(type(h)is int and type(m)is int and 0<=h<=m,'literal terminal HP/max invalid')
            require(not row['engine_failed']and not row['capped'],'failed/capped row gained terminal credit')
            terminals.append(dict(stream_id=row['stream_id'],role=row['role'],winner=term['winner'],turn=turn,loser_party_all_zero=None if term['winner']==2 else not any(hp[1-term['winner']])))
    roles={}
    for role,subset in [('census',rows[:512]),('timing-only',rows[512:])]:
        n=len(subset);values=dict(original_denominator=n,engine_terminal_games=sum(r['engine_terminal']for r in subset),capture_gap_games=sum(r['first_capture_gap']is not None for r in subset),engine_refused_games=sum(r['engine_failed']for r in subset),capped_games=sum(r['capped']for r in subset),unknown_terminal_games=sum(not r['engine_terminal']for r in subset),terminal_without_capture_gap=sum(r['engine_terminal']and r['observation_scoring_valid']for r in subset))
        if role=='timing-only':values={k.replace('_games','_rows'):v for k,v in values.items()};values['original_row_denominator']=values.pop('original_denominator')
        require(data['census'if role=='census'else'timing_only']==values,'cohort aggregate differs: '+role);roles[role]=values
    steps=data['per_step'];require([x['step']for x in steps]==list(range(len(steps))),'per-step order differs')
    for x in steps:
        require(x['launch_rows']==2048,'step launch differs');active=count(x['active_engine_rows'],'active rows');executed=count(x['executed_engine_rows'],'executed rows')
        require(executed<=active<=2048 and count(x['capture_valid_rows'],'valid rows')+count(x['fallback_rows'],'fallback rows')==active,'step row accounting differs')
    executed=sum(r['executed_steps']for r in rows);require(executed==sum(x['executed_engine_rows']for x in steps),'cohort/perstep executed totals differ')
    kinds=[];seen=set()
    for x in data['unsupported']:
        role=x['role'];require(role in roles,'unknown unsupported role');key='games_affected'if role=='census'else'timing_rows_affected';affected=count(x[key],key);occ=count(x['occurrences'],'occurrences')
        require(affected<= (512 if role=='census'else 1536)and affected<=occ,'unsupported aggregate denominator differs')
        identity=(role,x['domain'],x['kind']);require(identity not in seen,'duplicate unsupported entry');seen.add(identity)
        kinds.append(dict(role=role,domain=x['domain'],kind=x['kind'],occurrences=occ,**{key:affected}))
    grouped={}
    for entry in kinds:
        require(entry['domain']in('engine','capture','capture-context'),'unknown original count domain')
        source='engine'if entry['domain']=='engine'else'capture-gap';key=(entry['role'],source,entry['kind']);affected=entry['games_affected'if entry['role']=='census'else'timing_rows_affected']
        g=grouped.setdefault(key,dict(occurrences=0,affected=[]));g['occurrences']+=entry['occurrences'];g['affected'].append(affected)
    union=data['unsupported_by_source'];require(len(union)==len(grouped),'source-union entry count differs');union_seen=set()
    for entry in union:
        key=(entry['role'],entry['source'],entry['kind']);require(key in grouped and key not in union_seen,('missing/duplicate/'
            'unrecognized source-union group'));union_seen.add(key);g=grouped[key]
        require(count(entry['occurrences'],'source occurrences')==g['occurrences'],'source-union occurrences differ')
        n=count(entry['games_affected'if entry['role']=='census'else'timing_rows_affected'],'source union affected');limit=512 if entry['role']=='census'else 1536
        require(max(g['affected'])<=n<=min(limit,sum(g['affected'])),'source union unique bounds differ')
    require([(x['role'],x['source'],x['kind'])for x in union]==sorted(union_seen),'original source-union order differs')
    samples=data.get('samples',[]);phase={};scalar={};warm=[];cache_nulls={};zero_admitted=[];selected_without_driver_wall=[]
    for i,s in enumerate(samples):
        require(s['step']==i and i<len(steps),'sample/step order differs');require(s['launch_shape']==2048,'model launch differs')
        require(s['active_engine_rows']==steps[i]['active_engine_rows']and s['census_active_rows']+s['timing_only_active_rows']==s['active_engine_rows'],'sample activity differs')
        skipped=policy_work(s,i);admit=s['admitted_policy_owner_rows']
        require(s['policy_input_capture_valid_rows']>=steps[i]['capture_valid_rows'],'capture validity increased within step')
        if admit==0:zero_admitted.append(i)
        phases=s['phases_ms'];selected=count(phases['original_request_rows'],'selected original-driver rows');require(selected<=s['active_engine_rows'],'selected driver rows exceed active')
        caller_keys={'policy_wall_ms','legal_mask_fallback_wall_ms','chance_sampling_dispatch_wall_ms','advance_wall_ms','accounting_wall_ms'}
        actual_caller={k.removeprefix('census_host_overlap.')for k in phases if k.startswith('census_host_overlap.')};require(actual_caller==caller_keys,'five literal census host-wall fields required')
        for key in caller_keys:require(number(phases['census_host_overlap.'+key],key)>=0,'negative caller wall')
        driver_host={k:v for k,v in phases.items()if k.startswith('driver_host_overlap.')}
        if selected and 'driver_host_overlap.advance_wall_ms'not in driver_host:selected_without_driver_wall.append(i)
        # Driver remainder may be signed; preserve every raw span, no clipping.
        for k,v in phases.items():
            if k!='original_request_rows':phase[k]=phase.get(k,0.)+number(v,k)
        for k in ('elapsed_ms','inclusive_step_wall_ms','journal_publication_ms','bf16_compile_ms','bf16_forward_ms'):
            if k in s:scalar[k]=scalar.get(k,0.)+number(s[k],k)
        scalar['shadow_actual_encoder_ms']=scalar.get('shadow_actual_encoder_ms',0.)+sum(number(v,'encoder ms')for v in s['shadow_actual_encoder_ms'])
        deltas=s['cache_signature_delta'];require(all(v is None or type(v)is int for v in deltas.values()),'cache deltas must be literal int/null')
        complete=all(v is not None for v in deltas.values());require(s['warm_cache_observability_complete']==complete,'cache observability flag differs')
        for k,v in deltas.items():cache_nulls[k]=cache_nulls.get(k,0)+(v is None)
        expected=not skipped and i>=plan['timing_warmup']and not any(v is not None and v>0 for v in deltas.values())and not(s.get('bf16_compile_ms',0)>0 or phases.get('diagnostic_fallback_lower_compile_ms',0)>0)
        require(s['warm_candidate']==expected,'warm flag differs')
        if expected and i<32:warm.append(s)
    if samples:
        require(len(samples)==len(steps),'final timing/perstep sample counts differ')
        w=data['warm_launch_shape'];require(w['rows']==2048 and w['samples']==len(warm),'warm window count differs')
        for key in ('active_engine_rows','bf16_physical_owner_rows','admitted_policy_owner_rows'):require(w[key]==[s[key]for s in warm],'warm window row vector differs: '+key)
        require(w['original_request_rows']==[s['phases_ms']['original_request_rows']for s in warm],'warm selected driver vector differs')
        obs=[sum(s['shadow_actual_encoder_ms'])+sum(v for k,v in s['phases_ms'].items()if k.startswith(('driver_public','driver_encode','driver_admission')))for s in warm]
        require(len(obs)==len(w['observed_encoder_ms']),'warm encoder sample count differs')
        for actual,expected in zip(w['observed_encoder_ms'],obs):close(actual,expected,'warm observed encoder ms')
        if warm:close(w['partial_diagnostic_observation_wall_fraction'],sum(obs)/sum(s['elapsed_ms']for s in warm),'partial diagnostic fraction')
        else:require(w['partial_diagnostic_observation_wall_fraction']is None,'zero warm samples gained fraction')
    for key in ('full_B4_qualified','fidelity_qualified','strength_qualified'):require(data[key]is False,'diagnostic gained qualification')
    for key in ('production_fullgame_throughput','full_policy_game_rate','observation_trigger_25_percent'):require(data.get(key)is None,'diagnostic gained production claim')
    return dict(schema='pokezero.census.fresh-independent-stored-consistency.postwindow.v2',verdict='STORED_CONSISTENT_DIAGNOSTIC_ONLY',source_commit=data['source']['commit'],source_snapshot_sha256=canonical(data['source']),role_counts=roles,sum_executed_row_steps=executed,census_executed_steps=sum(r['executed_steps']for r in rows[:512]),timing_executed_steps=sum(r['executed_steps']for r in rows[512:]),per_step_records=len(steps),terminal_snapshot_count=len(terminals),terminal_snapshot_nonzero_loser_party=[x for x in terminals if x['loser_party_all_zero']is False],unsupported=kinds,unsupported_by_source=union,source_union_scope='Occurrence sums verified; unique affected bounds only. Exact original set union cannot be independently reconstructed from summary strata.',callback_counts_complete=data['callback_counts_complete'],timing=dict(samples=len(samples),postwindow_skipped_sample_steps=[s['step']for s in samples if s['policy_work_skipped']],warm_window_candidate_count=len(warm),warm_window_zero_admitted_samples=sum(s['admitted_policy_owner_rows']==0 for s in warm),zero_admitted_sample_steps=zero_admitted,selected_rows_without_driver_host_wall_steps=selected_without_driver_wall,cache_null_observations=cache_nulls,component_ms_totals=phase,scalar_ms_totals=scalar,construction_ms=data.get('construction_ms'),elapsed_seconds=data.get('elapsed_seconds'),requested_owner_rows=None),host_timing_scope=data.get('host_timing_scope'),unsupported_by_source_scope=data.get('unsupported_by_source_scope'),producer_status=data.get('status'),producer_actual_exit=data.get('actual_exit'),limits=['Full per-kind callback/raw-history recount unavailable from bounded examples; preserved table aggregates are checked for scope/arithmetic, not independently reconstructed.','Warm-candidate and cache-observability fields remain distinct. A window with zero admitted owners is shadow diagnostic graph work, not representative valid selfplay throughput.','Original selected-driver rows are counts, not milliseconds or requested-owner counts. Component and inclusive timings overlap and must not be summed as disjoint total time.','Literal engine terminal snapshots do not establish original Showdown equivalence or public/full-policy completion. B4/fidelity/strength remain false.'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--bindings',type=Path,required=True);p.add_argument('--bindings-sha256',required=True);a=p.parse_args();require(sha(a.bindings)==digest(a.bindings_sha256),'audit bindings differ');b=json.loads(a.bindings.read_bytes());loaded={};files={}
    for key in ('data','prespec','pool','runtime','pool_generation'):
        r=b[key];path=Path(r['path']);require(path.is_absolute(),'absolute original path required');actual=sha(path);require(actual==digest(r['sha256']),'original file differs: '+key);files[key]=dict(path=str(path),sha256=actual,bytes=path.stat().st_size)
        if key!='pool':require(path.stat().st_size<=128*1024*1024,'bounded JSON required');loaded[key]=json.loads(path.read_bytes())
    d,plan=loaded['data'],loaded['prespec'];require(plan['source_commit']==b['source_commit'],'external source binding differs');require(plan['prospective_bank_sha256']==digest(b['bank_sha256']),'external original bank binding differs');generation=loaded['pool_generation'];require(set(generation)=={'pool_sha256','pool_bytes','team_seed_start','teams','prospective_bank_sha256','producer','new_output','no_reuse'},'exact original pool-generation receipt required')
    require(generation['pool_sha256']==files['pool']['sha256']==d['pool_receipt']['pool_sha256']and count(generation['pool_bytes'],'pool bytes')==files['pool']['bytes'],'fresh original pool hash/size differs')
    require(generation['new_output']is True and generation['no_reuse']is True and generation['producer']=='gpu_engine.team_pool'and count(generation['teams'],'actual generated teams')==4096 and count(generation['team_seed_start'],'actual generated team seed')==plan['team_seed_start']==plan['pool_generation']['seed_start']and generation['prospective_bank_sha256']==plan['prospective_bank_sha256'],'fresh actual pool generation identity differs')
    expected_pool=plan['pool_generation']['expected_pool_sha256'];require(expected_pool is None or expected_pool==generation['pool_sha256'],'prespecified pool digest differs')
    result=check(d,plan);result['original_files']=files;result['bindings_sha256']=a.bindings_sha256
    raw=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n';require(len(raw)<=256*1024,'compact output exceeds256KiB');print(raw.decode(),end='')
if __name__=='__main__':main()
