"""Inference-only functional JAX port of the champion's one-frame transformer.

Weights are exported once on the host; device forward never imports Torch or
calls Python per battle. This measures compute, not production-encoder parity,
training, value calibration, strength, or completed-game throughput.
"""
from dataclasses import asdict,dataclass
import hashlib
import io
import json
import os
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as j
import numpy as np


@dataclass(frozen=True)
class Config:
    numeric: int=132
    categorical: int=41
    tokens: int=23
    vocabulary: int=916
    token_types: int=16
    width: int=512
    layers: int=3
    heads: int=8
    feedforward: int=2048
    tanh_value: bool=True
    norm: str='layer'
    moe: bool=False
    # Semantic token roles are fixed by the observation contract. 0=Pokemon,
    # 1=action, 2=field/statistics/belief. Appended v5 belief tokens use expert2.
    expert_groups: tuple[int,...]=()
    value_head: str='linear'
    value_hidden: int=512
    value_bins: int=51

    def __post_init__(self):
        if any(not isinstance(x,int) or x<=0 for x in (self.numeric,self.categorical,self.tokens,
                self.vocabulary,self.token_types,self.width,self.layers,self.heads,self.feedforward,
                self.value_hidden,self.value_bins)):
            raise ValueError('positive integer transformer dimensions required')
        if self.width%self.heads or self.tokens<22:
            raise ValueError('head division and action tokens13..21 required')
        if self.norm not in ('layer','rms') or self.value_head not in ('linear','mlp','hl_gauss'):
            raise ValueError('unsupported norm or value head')
        if type(self.moe) is not bool or type(self.tanh_value) is not bool or self.value_bins<2:
            raise ValueError('boolean flags and at least two value bins required')
        groups=tuple(self.expert_groups) or (2,)+(0,)*12+(1,)*9+(2,)*(self.tokens-22)
        if len(groups)!=self.tokens or any(type(g) is not int or g not in (0,1,2) for g in groups):
            raise ValueError('one semantic expert group per token required')
        if self.moe and set(groups)!={0,1,2}:raise ValueError('all three expert groups required')
        object.__setattr__(self,'expert_groups',groups)


def shapes(c):
    d=c.width
    result={'category_embedding.weight':(c.vocabulary,d),'token_type_embedding.weight':(c.token_types,d),
        'history_position_embedding.weight':(1,d),'numeric_projection.weight':(d,c.numeric),
        'numeric_projection.bias':(d,),'policy_head.weight':(1,d),'policy_head.bias':(1,),
        'opponent_action_head.weight':(9,d),'opponent_action_head.bias':(9,)}
    if c.value_head=='linear':result.update({'value_head.weight':(1,d),'value_head.bias':(1,)})
    else:
        h=c.value_hidden;o=c.value_bins if c.value_head=='hl_gauss' else 1
        result.update({'value_pool.query':(d,),'value_mlp.0.weight':(h,d),'value_mlp.0.bias':(h,),
            'value_mlp.1.weight':(h,h),'value_mlp.1.bias':(h,),
            'value_output.weight':(o,h),'value_output.bias':(o,)})
    for layer in range(c.layers):
        p=f'encoder.layers.{layer}.'
        result.update({p+'self_attn.in_proj_weight':(3*d,d),p+'self_attn.in_proj_bias':(3*d,),
            p+'self_attn.out_proj.weight':(d,d),p+'self_attn.out_proj.bias':(d,),
            p+'norm1.weight':(d,),p+'norm2.weight':(d,)})
        if c.norm=='layer':result.update({p+'norm1.bias':(d,),p+'norm2.bias':(d,)})
        for expert in range(3) if c.moe else (None,):
            f=p+(f'experts.{expert}.' if expert is not None else '')
            result.update({f+'linear1.weight':(c.feedforward,d),f+'linear1.bias':(c.feedforward,),
                f+'linear2.weight':(d,c.feedforward),f+'linear2.bias':(d,)})
    return result


def validate_weights(params,c):
    expected=shapes(c)
    if set(params)!=set(expected):raise ValueError('checkpoint tensor keys differ from supported architecture')
    for name,shape in expected.items():
        a=np.asarray(params[name])
        if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise ValueError(f'invalid float32 tensor {name}: {a.shape}/{a.dtype}, expected {shape}')


def snapshot(path):
    """Bind deserialization and digest to the same immutable byte snapshot."""
    with Path(path).open('rb') as stream:
        before=os.fstat(stream.fileno());blob=stream.read();after=os.fstat(stream.fileno())
    identity=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns)
    if identity(before)!=identity(after) or len(blob)!=after.st_size:
        raise RuntimeError('checkpoint changed while reading')
    return blob,hashlib.sha256(blob).hexdigest()


def config_from_checkpoint(cfg):
    if (cfg['window_size']!=1 or cfg.get('temporal_aggregator','mean')!='mean'
            or cfg.get('value_head_hidden_layers',0) not in (0,None)
            or cfg.get('value_activation','linear') not in ('linear','tanh')):
        raise ValueError('only one-frame mean pooling and linear/tanh single-layer value heads supported')
    return Config(numeric=cfg['numeric_feature_count'],categorical=cfg['categorical_feature_count'],
        tokens=cfg['token_count'],vocabulary=cfg['categorical_vocab_size'],token_types=cfg['token_type_vocab_size'],
        width=cfg['embedding_dim'],layers=cfg['transformer_layers'],heads=cfg['attention_heads'],
        feedforward=cfg['feedforward_dim'],tanh_value=cfg.get('value_activation','linear')=='tanh')


def export(checkpoint,out):
    """CPU-only, safe Torch loader; exclusive NPZ output outside version control."""
    import torch
    blob,digest=snapshot(checkpoint)
    payload=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
    c=config_from_checkpoint(payload['model_config'])
    params={k:v.detach().cpu().numpy() for k,v in payload['state_dict'].items()}
    validate_weights(params,c)
    meta={'schema':'pokezero.gpu_champion_forward.v1','config':asdict(c),
        'checkpoint_sha256':digest,
        'parameter_count':sum(a.size for a in params.values())}
    encoded=np.frombuffer(json.dumps(meta,sort_keys=True).encode(),np.uint8)
    with Path(out).open('xb') as stream:np.savez_compressed(stream,metadata_json=encoded,**params)
    return meta


def load(path):
    blob,digest=snapshot(path)
    with np.load(io.BytesIO(blob),allow_pickle=False) as data:
        meta=json.loads(data['metadata_json'].tobytes())
        if meta['schema']!='pokezero.gpu_champion_forward.v1':raise ValueError('unsupported weight archive')
        c=Config(**meta['config']);params={k:data[k] for k in data.files if k!='metadata_json'}
    validate_weights(params,c)
    if meta['parameter_count']!=sum(a.size for a in params.values()):raise ValueError('parameter count mismatch')
    meta['weights_archive_sha256']=digest
    return {k:j.asarray(v) for k,v in params.items()},c,meta


class Output(NamedTuple):
    policy_logits: jax.Array
    value: jax.Array
    opponent_action_logits: jax.Array


def forward(params,numeric,categorical,token_types,attention,c,*,compute_dtype='float32'):
    """Eval forward with arbitrary leading batch/seat axes and explicit precision.

    At least one attended token per row is required (the placeholder always
    includes field/action tokens). Padding masks keys, not queries, just as the
    production TransformerEncoder. No dropout, history beyond1, GRU or training.
    BF16 uses bf16 dot operands with FP32 accumulations, embeddings, residuals,
    normalization, softmax, biases and outputs. FP32 retains HIGHEST precision.
    MoE groups are static semantic roles from Config, not learned routing: the
    observation contract must supply the configured role at each token index.
    """
    if compute_dtype not in ('float32','bfloat16'):raise ValueError('unsupported compute dtype')
    dtype=j.bfloat16 if compute_dtype=='bfloat16' else j.float32
    def dot(a,b):
        return j.matmul(a.astype(dtype),b.astype(dtype),precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=j.float32)
    leading=numeric.shape[:-2]
    if (numeric.shape[-2:]!=(c.tokens,c.numeric) or categorical.shape!=leading+(c.tokens,c.categorical)
            or token_types.shape!=leading+(c.tokens,) or attention.shape!=leading+(c.tokens,)):
        raise ValueError('observation tensor shapes differ from weight config')
    linear=lambda x,p:dot(x,params[p+'.weight'].T)+params[p+'.bias']
    cats=j.clip(categorical,0,c.vocabulary-1)
    types=j.clip(token_types,0,c.token_types-1)
    x=j.sum(params['category_embedding.weight'][cats],axis=-2)
    x=x+params['token_type_embedding.weight'][types]+linear(numeric.astype(j.float32),'numeric_projection')
    x=x+params['history_position_embedding.weight'][0]
    x=x.reshape((-1,c.tokens,c.width));valid=attention.reshape((-1,c.tokens)).astype(j.bool_)
    def norm(a,p):
        if c.norm=='rms':return a*jax.lax.rsqrt(j.mean(j.square(a),axis=-1,keepdims=True)+1e-5)*params[p+'.weight']
        mean=j.mean(a,axis=-1,keepdims=True)
        variance=j.mean(j.square(a-mean),axis=-1,keepdims=True)
        return (a-mean)*jax.lax.rsqrt(variance+1e-5)*params[p+'.weight']+params[p+'.bias']
    for layer in range(c.layers):
        p=f'encoder.layers.{layer}.';a=norm(x,p+'norm1')
        packed=dot(a,params[p+'self_attn.in_proj_weight'].T)+params[p+'self_attn.in_proj_bias']
        q,k,v=[part.reshape((-1,c.tokens,c.heads,c.width//c.heads)).transpose((0,2,1,3))
            for part in j.split(packed,3,axis=-1)]
        scores=dot(q,k.swapaxes(-1,-2))/j.sqrt(j.asarray(c.width//c.heads,j.float32))
        probs=jax.nn.softmax(j.where(valid[:,None,None,:],scores,-j.inf),axis=-1)
        attended=dot(probs,v).transpose((0,2,1,3)).reshape(x.shape)
        x=x+linear(attended,p+'self_attn.out_proj')
        a=norm(x,p+'norm2')
        if c.moe:
            # Partition once per expert; each token executes exactly one FFN.
            update=j.zeros_like(x)
            for expert in range(3):
                indices=np.asarray([i for i,g in enumerate(c.expert_groups) if g==expert],np.int32)
                f=p+f'experts.{expert}.'
                z=linear(jax.nn.gelu(linear(a[:,indices],f+'linear1'),approximate=False),f+'linear2')
                update=update.at[:,indices].set(z)
            x=x+update
        else:x=x+linear(jax.nn.gelu(linear(a,p+'linear1'),approximate=False),p+'linear2')
    pooled=j.sum(j.where(valid[...,None],x,0),axis=1)/j.maximum(1,j.sum(valid,axis=1,keepdims=True))
    if c.value_head=='linear':raw=linear(pooled,'value_head')[...,0]
    else:
        scores=j.sum(x*params['value_pool.query'],axis=-1)/j.sqrt(j.asarray(c.width,j.float32))
        weights=jax.nn.softmax(j.where(valid,scores,-j.inf),axis=-1)
        value_pool=j.sum(x*weights[...,None],axis=1)
        hidden=jax.nn.gelu(linear(value_pool,'value_mlp.0'),approximate=False)
        hidden=jax.nn.gelu(linear(hidden,'value_mlp.1'),approximate=False)
        logits=linear(hidden,'value_output')
        raw=(j.sum(jax.nn.softmax(logits,axis=-1)*j.linspace(-1,1,c.value_bins),axis=-1)
            if c.value_head=='hl_gauss' else logits[...,0])
    return Output(linear(x[:,13:22],'policy_head')[...,0].reshape(leading+(9,)),
        (j.tanh(raw) if c.tanh_value and c.value_head!='hl_gauss' else raw).reshape(leading),
        linear(pooled,'opponent_action_head').reshape(leading+(9,)))


def sample(output,legal,keys):
    """Temperature1 per-seat sample; terminal/no-legal rows return index−1."""
    if output.policy_logits.shape!=legal.shape or keys.shape!=legal.shape[:-1]+(2,):
        raise ValueError('policy/mask/per-seat PRNG shape mismatch')
    live=j.any(legal,axis=-1)
    logits=j.where(legal,output.policy_logits,-j.inf)
    safe=j.where(live[...,None],logits,j.zeros_like(logits))
    actions=jax.vmap(lambda k,x:jax.random.categorical(k,x))(keys.reshape((-1,2)),safe.reshape((-1,9)))
    return j.where(live,actions.reshape(live.shape),-1)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();print(json.dumps(export(args.checkpoint,args.out),sort_keys=True))
