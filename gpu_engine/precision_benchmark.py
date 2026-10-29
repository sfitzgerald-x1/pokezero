"""A3 saved-input precision gate and forward-only candidate measurements.

Inputs are frozen Python-encoded observations, never the unfinished GPU encoder.
Batch means battle slots with two seat rows, not completed games. Candidate
weights are deterministic random weights; their timings say nothing about skill.
"""
from dataclasses import asdict,replace
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import jax
import jax.numpy as j
import numpy as np

from . import champion


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()


def policy_comparison(fp32,bf16,legal):
    """KL(p_FP32 || p_bf16); stable float64 accounting outside timed regions."""
    a,b=np.asarray(fp32,np.float64),np.asarray(bf16,np.float64)
    legal=np.asarray(legal)
    if (a.ndim!=2 or a.shape!=b.shape or a.shape!=legal.shape or a.shape[1]!=9
            or legal.dtype!=np.bool_ or not legal.any(axis=1).all()
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('finite nine-action logits and at least one legal action required')
    def one(mask):
        x,y=np.where(mask,a,-np.inf),np.where(mask,b,-np.inf)
        def logprob(z):
            z=z-z.max(axis=1,keepdims=True)
            return z-np.log(np.exp(z).sum(axis=1,keepdims=True))
        p,q=logprob(x),logprob(y)
        kl=np.sum(np.where(mask,np.exp(p)*np.where(mask,p,0)-np.exp(p)*np.where(mask,q,0),0),axis=1)
        return {'rows':len(a),'agreement_count':int((x.argmax(axis=1)==y.argmax(axis=1)).sum()),
            'agreement':float(np.mean(x.argmax(axis=1)==y.argmax(axis=1))),
            'kl_sum':float(kl.sum()),'kl_mean':float(kl.mean()),'kl_max':float(kl.max())}
    return {'raw':one(np.ones_like(legal)),'legal':one(legal)}


def random_parameters(c,seed):
    rng=np.random.default_rng(seed);params={}
    for key,shape in champion.shapes(c).items():
        if '.norm' in key and key.endswith('.weight'):a=np.ones(shape,np.float32)
        elif key.endswith('.bias'):a=np.zeros(shape,np.float32)
        else:a=rng.normal(0,1/np.sqrt(shape[-1]),shape).astype(np.float32)
        if key=='category_embedding.weight':a[0]=0
        params[key]=a
    champion.validate_weights(params,c)
    h=hashlib.sha256()
    for key,a in sorted(params.items()):
        h.update(json.dumps([key,list(a.shape),a.dtype.str],separators=(',',':')).encode());h.update(a.tobytes())
    return params,h.hexdigest()


def load_inputs(path,expected_sha,c):
    if digest(path)!=expected_sha:raise ValueError('saved input archive hash mismatch')
    with np.load(path,allow_pickle=False) as data:
        required={'numeric_features','categorical_ids','token_type_ids','attention_mask','legal_action_mask'}
        if set(data.files)!=required:raise ValueError('saved input keys differ')
        n,k,t,a,l=[np.array(data[key],copy=True) for key in
            ('numeric_features','categorical_ids','token_type_ids','attention_mask','legal_action_mask')]
    rows=len(n)
    if (n.dtype!=np.float32 or n.shape!=(rows,c.tokens,c.numeric)
            or k.dtype!=np.int32 or k.shape!=(rows,c.tokens,c.categorical)
            or t.dtype!=np.int32 or t.shape!=(rows,c.tokens)
            or a.dtype!=np.bool_ or a.shape!=(rows,c.tokens)
            or l.dtype!=np.bool_ or l.shape!=(rows,9) or rows<16384
            or not np.isfinite(n).all() or not a.any(axis=1).all() or not l.any(axis=1).all()
            or not ((k>=0)&(k<c.vocabulary)).all() or not ((t>=0)&(t<c.token_types)).all()):
        raise ValueError('invalid frozen champion input tensors')
    # Config's FF experts express these semantic token roles. Refuse another
    # observation contract rather than silently route that contract by position.
    expected=np.asarray((0,)+(1,)*6+(2,)*6+(3,)*9+(5,),np.int32)
    if not np.array_equal(t,np.broadcast_to(expected,t.shape)):
        raise ValueError('token types differ from static semantic expert grouping')
    return (n,k,t,a),l


def _timing(params,c,args,device,*,dtype,warmup,repeats):
    params=jax.device_put(params,device)
    compiled=jax.jit(lambda n,k,t,a:champion.forward(params,n,k,t,a,c,compute_dtype=dtype))
    start=time.perf_counter();compiled=compiled.lower(*args).compile();compile_seconds=time.perf_counter()-start
    for _ in range(warmup):out=jax.block_until_ready(compiled(*args))
    if any(x.devices()!={device} for x in out) or not all(np.isfinite(np.asarray(x)).all() for x in out):
        raise RuntimeError('output not finite or not solely on recorded GPU')
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();jax.block_until_ready(compiled(*args))
        samples.append((time.perf_counter_ns()-start)/1e6)
    return {'config':asdict(c),'dtype':dtype,'compile_seconds':compile_seconds,
        'mean_ms':statistics.fmean(samples),'median_ms':statistics.median(samples),
        'p95_ms':float(np.quantile(samples,.95)),'samples_ms':samples}


def measure(weights,inputs,*,weights_sha,inputs_sha,checkpoint_sha,output,warmup=10,repeats=100):
    if type(warmup)is not int or warmup<1 or type(repeats)is not int or repeats<2:
        raise ValueError('positive warmup and at least two repeats required')
    out=Path(output);out.mkdir(exist_ok=False)
    devices=jax.devices()
    if len(devices)!=1 or devices[0].platform!='gpu' or 'GB300' not in devices[0].device_kind:
        raise RuntimeError('exactly one GB300 required')
    device=devices[0]
    if digest(weights)!=weights_sha:raise ValueError('weights archive hash mismatch')
    params,c,meta=champion.load(weights)
    if meta['checkpoint_sha256']!=checkpoint_sha or c!=champion.Config():
        raise ValueError('exact default champion checkpoint/config required')
    arrays,legal=load_inputs(inputs,inputs_sha,c)
    source={p.name:digest(p) for p in (Path(champion.__file__),Path(__file__))}
    result={'schema':'pokezero.A3-bf16-forward.v1','device':device.device_kind,'jax':jax.__version__,
        'jaxlib':__import__('jaxlib').__version__,'source_sha256':source,'weights':meta,'inputs_sha256':inputs_sha,
        'input_rows':len(legal),'warmup':warmup,'repeats':repeats,
        'precision':'bf16 dot operands, FP32 accumulation/embedding/residual/norm/softmax/bias/output; FP32 HIGHEST reference',
        'timing':'host dispatch plus synchronized all3heads; encode/setup/transfer/compile/warmup/sampling excluded',
        'batch_definition':'battle slots, two independent saved observation rows per slot; not paired battle-seat records',
        'scope':'saved Python encoder inputs; random candidate weights; appended29token fields authored zeros; forward only, no B2 parity/games/strength/training',
        'rows':[]}
    parameters=jax.device_put(params,device)
    funcs={dtype:jax.jit(lambda n,k,t,a,dtype=dtype:champion.forward(parameters,n,k,t,a,c,compute_dtype=dtype))
        for dtype in ('float32','bfloat16')}
    totals={kind:{'rows':0,'agreement_count':0,'kl_sum':0.,'kl_max':0.} for kind in ('raw','legal')}
    value_error=0.
    for start in range(0,len(legal),512):
        args=jax.device_put(tuple(x[start:start+512] for x in arrays),device)
        full=jax.block_until_ready(funcs['float32'](*args));low=jax.block_until_ready(funcs['bfloat16'](*args))
        for value in (*full,*low):
            if not np.isfinite(np.asarray(value)).all():raise RuntimeError('nonfinite precision gate output')
        comparison=policy_comparison(full.policy_logits,low.policy_logits,legal[start:start+512])
        value_error=max(value_error,float(np.max(np.abs(np.asarray(full.value)-np.asarray(low.value)))))
        for kind in totals:
            for key in ('rows','agreement_count','kl_sum'):totals[kind][key]+=comparison[kind][key]
            totals[kind]['kl_max']=max(totals[kind]['kl_max'],comparison[kind]['kl_max'])
    for record in totals.values():
        record.update(agreement=record['agreement_count']/record['rows'],kl_mean=record['kl_sum']/record['rows'])
    gate={'policies':totals,'kl_direction':'FP32 || bf16, natural logarithms',
        'maximum_value_absolute_difference':value_error,'threshold':.995,
        'pass':all(x['agreement']>=.995 for x in totals.values())}
    result['champion_gate']=gate
    (out/'champion-gate.json').write_text(json.dumps(gate,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'event':'champion_precision_gate',**gate}),flush=True)
    if not gate['pass']:
        (out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        raise RuntimeError('champion bf16 argmax gate failed; timing not started')
    # Run exactly three registered candidate shapes. LayerNorm and the existing
    # mean/linear/tanh head isolate width/depth/MoE cost; v5 RMS/head cost is not
    # folded into this primary table. Repeat the table with six authored tokens.
    candidates=(('256-1024-3',256,1024,3,False),('512-2048-3-moe',512,2048,3,True),
        ('512-2048-6',512,2048,6,False))
    for tokens in (23,29):
        shaped=arrays if tokens==23 else tuple(np.concatenate((x,
            np.full((len(x),6)+x.shape[2:],True if i==3 else 6 if i==2 else 0,dtype=x.dtype)),axis=1)
            for i,x in enumerate(arrays))
        for batch in (2048,8192):
            args=jax.device_put(tuple(x[:batch*2] for x in shaped),device);jax.block_until_ready(args)
            if tokens==23:
                for dtype in ('float32','bfloat16'):
                    row=_timing(params,c,args,device,dtype=dtype,warmup=warmup,repeats=repeats)
                    row.update(model='champion',batch=batch,seat_rows=2*batch,tokens=tokens)
                    result['rows'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
            for name,width,ff,layers,moe in candidates:
                config=replace(c,width=width,feedforward=ff,layers=layers,moe=moe,tokens=tokens,expert_groups=())
                candidate,parameter_sha=random_parameters(config,20261029)
                row=_timing(candidate,config,args,device,dtype='bfloat16',warmup=warmup,repeats=repeats)
                row.update(model=name,batch=batch,seat_rows=2*batch,tokens=tokens,
                    parameter_sha256=parameter_sha,parameter_count=sum(a.size for a in candidate.values()),seed=20261029)
                result['rows'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
            jax.clear_caches()
    if digest(inputs)!=inputs_sha or digest(weights)!=weights_sha or source!={p.name:digest(p) for p in (Path(champion.__file__),Path(__file__))}:
        raise RuntimeError('bound input or source changed during measurement')
    result['nvidia_smi_after']=subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,uuid,driver_version,memory.total,memory.used','--format=csv,noheader'],text=True).strip()
    (out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('weights','inputs','weights-sha','inputs-sha','checkpoint-sha','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--warmup',type=int,default=10);parser.add_argument('--repeats',type=int,default=100)
    args=parser.parse_args();measure(args.weights,args.inputs,weights_sha=args.weights_sha,
        inputs_sha=args.inputs_sha,checkpoint_sha=args.checkpoint_sha,output=args.output,
        warmup=args.warmup,repeats=args.repeats)
