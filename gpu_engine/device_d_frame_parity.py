"""Quarantined fixed saved-root D1 comparison to the original serial provider.

Load the bank with its ORIGINAL Prepared/namespace. The caller separately
records this batch module's execution source; never rewrite a saved namespace.
There are zero new games/outcomes. No latency gate or throughput qualification
is inferred from this comparison. The declared full iteration measures D-T.
"""
from dataclasses import asdict
from unittest.mock import patch
import math
import jax
import numpy as np
from . import device_d_cost as cost,device_public_teacher as serial
from . import device_public_worlds as worlds
from .device_public_teacher_batch import FramePricer,Request

TOLERANCE=1e-6


def differences(a,b,path='root'):
    """All literal structure/status differences; floats have a fixed tolerance."""
    if isinstance(a,dict)and isinstance(b,dict):
        if set(a)!=set(b):return [path+':keys']
        return [v for key in a for v in differences(a[key],b[key],path+'.'+str(key))]
    if isinstance(a,(tuple,list))and isinstance(b,(tuple,list)):
        if len(a)!=len(b):return [path+':length']
        return [v for i,(x,y)in enumerate(zip(a,b))for v in differences(x,y,path+f'[{i}]')]
    if type(a)in (float,int)and type(b)in (float,int):
        return []if math.isfinite(a)and math.isfinite(b)and abs(a-b)<=TOLERANCE else[path+':numeric']
    return []if type(a)is type(b)and a==b else[path+':literal']


def comparable(result):
    names=('world','posterior_seed','kernel_seed','status','reason','required_pairs','completed_pairs','unknown_pairs','joint_values','opponent_requested','reply_legal','reply_logits','errors')
    return dict(status=result.status,reason=result.reason,missing_worlds=result.missing_worlds,
        improvement=None if result.reduction is None or result.reduction.improvement is None else asdict(result.reduction.improvement),
        reduction=None if result.reduction is None else asdict(result.reduction),
        world_reports=[{k:row[k]for k in names if k in row}for row in result.world_reports],
        reply_input_approximation=result.reply_input_approximation)


def numeric_summary(a,b,path='root',*,tolerance=TOLERANCE):
    """Magnitude evidence alongside the unchanged strict difference paths."""
    totals={}
    def walk(x,y,p):
        if isinstance(x,dict)and isinstance(y,dict):
            for k in set(x)&set(y):walk(x[k],y[k],p+'.'+str(k))
        elif isinstance(x,(tuple,list))and isinstance(y,(tuple,list)):
            for i,(u,v)in enumerate(zip(x,y)):walk(u,v,p+f'[{i}]')
        elif type(x)in (int,float)and type(y)in (int,float):
            kind='reply_logits'if 'reply_logits'in p else'joint_values'if 'joint_values'in p else'root_target'if 'improvement'in p else'other_numeric'
            row=totals.setdefault(kind,dict(compared=0,above_tolerance=0,nonfinite=0,max_absolute_difference=0.))
            row['compared']+=1
            if not(math.isfinite(x)and math.isfinite(y)):row['nonfinite']+=1;return
            delta=abs(x-y);row['max_absolute_difference']=max(row['max_absolute_difference'],delta)
            row['above_tolerance']+=int(delta>tolerance)
    walk(a,b,path)
    return totals


def vectorized_comparison(a,b):
    """D3 compares real structure exactly and final label targets within0.01.

    The outcome term cancels: the final outcome-plus-root value target differs
    by half the root-value delta, for every possible same eventual outcome.
    No terminal outcome is manufactured for a saved nonterminal root.
    """
    ca,cb=comparable(a),comparable(b);errors=[]
    for key in ('status','reason','missing_worlds','reply_input_approximation'):
        errors.extend(differences(ca[key],cb[key],key))
    if len(a.world_reports)!=len(b.world_reports):errors.append('world_reports:length')
    names=('world','posterior_seed','kernel_seed','status','reason','required_pairs','completed_pairs','unknown_pairs','opponent_requested','reply_legal','errors')
    for i,(x,y)in enumerate(zip(a.world_reports,b.world_reports)):
        errors.extend(differences({k:x[k]for k in names if k in x},{k:y[k]for k in names if k in y},f'world[{i}]:branch_structure'))
        def branch_set(row):
            grid=row.get('joint_values')
            return None if grid is None else tuple(tuple(v is not None for v in line)for line in grid)
        if branch_set(x)!=branch_set(y):errors.append(f'world[{i}]:branch_set')
    metrics=dict(policy_max_absolute_difference=None,value_target_max_absolute_difference=None,
        root_value_absolute_difference=None,top_target_action_match=None)
    x=ca['improvement'];y=cb['improvement']
    if (x is None)!=(y is None):errors.append('improvement:presence')
    elif x is not None:
        for key in ('worlds','depth','leaf','replies','label_only'):
            errors.extend(differences(x[key],y[key],'improvement.'+key))
        if tuple(v is not None for v in x['completed_q'])!=tuple(v is not None for v in y['completed_q']):errors.append('improvement:branch_set')
        px=np.asarray(x['policy_target'],float);py=np.asarray(y['policy_target'],float)
        finite=px.shape==py.shape==(9,)and np.isfinite(px).all()and np.isfinite(py).all()and math.isfinite(x['root_value'])and math.isfinite(y['root_value'])
        if not finite:errors.append('targets:nonfinite_or_shape')
        else:
            policy=float(np.max(np.abs(px-py)));root=abs(x['root_value']-y['root_value']);value=.5*root
            metrics.update(policy_max_absolute_difference=policy,value_target_max_absolute_difference=value,
                root_value_absolute_difference=root,top_target_action_match=int(np.argmax(px))==int(np.argmax(py)))
            if policy>.01:errors.append('policy_target:absolute_tolerance')
            if value>.01:errors.append('value_target:absolute_tolerance')
    return errors,metrics


def run(provider,snapshot,archive,index,*,execution_source,enable_parity=False,progress_callback=None,chunk_batch=False,vectorized=False):
    if enable_parity is not True or getattr(provider,'frame_batch',False):
        raise ValueError('explicit quarantined original serial provider required')
    if type(execution_source)is not str or len(execution_source)!=40:raise ValueError('separate batch execution source required')
    if type(chunk_batch)is not bool or type(vectorized)is not bool:raise ValueError('explicit saved-root packing choices required')
    requests={};references={};actors=[];method=provider.label_chunk;price=serial.price
    def price_reference(plans,*args,**kwargs):
        result=price(plans,*args,**kwargs)
        references[plans.root.address]=result
        return result
    def capture(roots,frames,bank,params):
        labels=method(roots,frames,bank,params)
        # The original cost driver verifies exact40 selectors and typed bank
        # before entering this callback. Only owner/public roots are retained.
        from .v5_device_teacher import _requests,INPUTS
        for step,row,owner in _requests(frames,bank):
            step,row,owner=map(int,(step,row,owner))
            if not frames.trainable[step,row,owner]:continue
            context=jax.tree.map(lambda x:x[step],roots)
            root=worlds.detach_root(context,provider.source,row=row,owner=owner,enable_device_worlds=True)
            plans=worlds.plan_starts(root,provider.source,provider.recipe)
            inputs={name:getattr(frames,name)[step,row,owner][None,None]for name in INPUTS}
            inputs['history_mask']=np.ones((1,1),bool)
            requests[root.address]=Request(plans,tuple(map(float,frames.raw_policy_logits[step,row,owner])),
                tuple(map(bool,frames.legal_action_mask[step,row,owner])),inputs,
                int(frames.actor_indices[step,row,owner]),int(frames.actor_indices[step,row,1-owner]))
        if actors and any(p is not q for p,q in zip(actors[0],params)):raise ValueError('saved actor parameter identity differs')
        if not actors:actors.append(params)
        return labels
    provider.label_chunk=capture
    try:
        with patch.object(serial,'price',side_effect=price_reference):
            reference=cost.run(provider,snapshot,archive,index,enable_cost=True,progress_callback=progress_callback)
    finally:provider.label_chunk=method
    if not reference['producer_integrity']:raise ValueError('original fixed saved-root producer failed')
    declared=cost.select(provider.plan);groups={}
    for address in declared:
        if address in requests:groups.setdefault(0 if chunk_batch else address.frame,[]).append(address)
    pricer=FramePricer(provider.executor,provider.plan.model,actor_capacity=provider.plan.historical_window+1,**(dict(capacity=4096,vectorized=True)if vectorized else{}))
    results={};timings=[]
    for frame,addresses in groups.items():
        priced=pricer.price([requests[a]for a in addresses],provider.materializer,provider.recipe,actors[0])
        results.update(zip(addresses,priced));timings.append(dict(frame=frame,**pricer.last_timing))
    rows=[]
    for address,old in zip(declared,reference['records']):
        row=dict(address=asdict(address),serial_status=old['status'],status='UNAVAILABLE',differences=[])
        rows.append(row)
        if address not in results:
            row['reason']=old['reason'];continue
        if address not in references:raise ValueError('batch priced a root absent from original serial provider')
        a,b=references[address],results[address];row['batch_status']=b.status
        ca,cb=comparable(a),comparable(b)
        if vectorized:
            row['differences'],row['target_comparison']=vectorized_comparison(a,b)
        else:row['differences']=differences(ca,cb)
        row['numeric_comparison']=numeric_summary(ca,cb,tolerance=.01 if vectorized else TOLERANCE)
        # Materialization moved to CPU, not changed: all physical/public arrays
        # and original world keys must remain byte-equal, independently of Q.
        for wi,(x,y)in enumerate(zip(a.worlds,b.worlds)):
            if x.kernel_seed!=y.kernel_seed or (x.state is None)!=(y.state is None):row['differences'].append(f'world[{wi}]:binding')
            if x.state is None:continue
            for name in ('state','public'):
                lx,tx=jax.tree.flatten(getattr(x,name));ly,ty=jax.tree.flatten(getattr(y,name))
                def same_array(u,v):
                    u,v=np.asarray(u),np.asarray(v)
                    return (u.dtype==v.dtype and u.shape==v.shape and u.tobytes()==v.tobytes())if vectorized else np.array_equal(u,v,equal_nan=True)
                if tx!=ty or any(not same_array(u,v)for u,v in zip(lx,ly)):
                    row['differences'].append(f'world[{wi}]:{name}')
        row['status']='MATCH'if not row['differences']else'MISMATCH'
    totals={status:sum(x['status']==status for x in rows)for status in ('MATCH','MISMATCH','UNAVAILABLE')}
    target_rows=[r['target_comparison']for r in rows if r.get('target_comparison',{}).get('top_target_action_match')is not None]
    top_matches=sum(r['top_target_action_match']for r in target_rows)
    passes=totals['MATCH']==40 and(not vectorized or len(target_rows)==32 and top_matches>=31)
    result=dict(schema='pokezero.device-d-frame-parity.v1',status='PASS'if passes else'FAIL',
        scope='QUARANTINED_ORIGINAL_FIXED40_SAVED_ROOTS_ZERO_NEW_GAMES',tolerance=.01 if vectorized else TOLERANCE,
        batching='CHUNK'if chunk_batch else'FRAME',
        execution_source=execution_source,original_bank_namespace=index['namespace'],
        serial_snapshot=reference['snapshot'],serial_declaration_sha256=reference['declaration_sha256'],
        totals=totals,serial_dispositions=reference['summary']['dispositions'],rows=rows,batch_timings=timings,
        no_strength_or_throughput_claim=True)
    if vectorized:
        result['batching']='D3_VECTORIZED'
        result['target_summary']=dict(labeled_roots=len(target_rows),top_target_action_matches=top_matches,required_top_matches=31,
            policy_max_absolute_difference=max((r['policy_max_absolute_difference']for r in target_rows),default=None),
            value_target_max_absolute_difference=max((r['value_target_max_absolute_difference']for r in target_rows),default=None),
            root_value_max_absolute_difference=max((r['root_value_absolute_difference']for r in target_rows),default=None),
            value_scope='Same-outcome final search value target; delta=half root-value delta; original root-value delta also reported, no fabricated outcome.')
    return result
