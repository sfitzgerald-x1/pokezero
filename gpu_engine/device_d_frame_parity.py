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


def run(provider,snapshot,archive,index,*,execution_source,enable_parity=False,progress_callback=None):
    if enable_parity is not True or getattr(provider,'frame_batch',False):
        raise ValueError('explicit quarantined original serial provider required')
    if type(execution_source)is not str or len(execution_source)!=40:raise ValueError('separate batch execution source required')
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
        if address in requests:groups.setdefault(address.frame,[]).append(address)
    pricer=FramePricer(provider.executor,provider.plan.model,actor_capacity=provider.plan.historical_window+1)
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
        row['differences']=differences(comparable(a),comparable(b))
        # Materialization moved to CPU, not changed: all physical/public arrays
        # and original world keys must remain byte-equal, independently of Q.
        for wi,(x,y)in enumerate(zip(a.worlds,b.worlds)):
            if x.kernel_seed!=y.kernel_seed or (x.state is None)!=(y.state is None):row['differences'].append(f'world[{wi}]:binding')
            if x.state is None:continue
            for name in ('state','public'):
                lx,tx=jax.tree.flatten(getattr(x,name));ly,ty=jax.tree.flatten(getattr(y,name))
                if tx!=ty or any(not np.array_equal(np.asarray(u),np.asarray(v),equal_nan=True)for u,v in zip(lx,ly)):
                    row['differences'].append(f'world[{wi}]:{name}')
        row['status']='MATCH'if not row['differences']else'MISMATCH'
    totals={status:sum(x['status']==status for x in rows)for status in ('MATCH','MISMATCH','UNAVAILABLE')}
    return dict(schema='pokezero.device-d-frame-parity.v1',status='PASS'if totals['MATCH']==40 else'FAIL',
        scope='QUARANTINED_ORIGINAL_FIXED40_SAVED_ROOTS_ZERO_NEW_GAMES',tolerance=TOLERANCE,
        execution_source=execution_source,original_bank_namespace=index['namespace'],
        serial_snapshot=reference['snapshot'],serial_declaration_sha256=reference['declaration_sha256'],
        totals=totals,serial_dispositions=reference['summary']['dispositions'],rows=rows,batch_timings=timings,
        no_strength_or_throughput_claim=True)
