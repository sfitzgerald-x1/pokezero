"""Inference-only functional JAX port of the champion's one-frame transformer.

Weights are exported once on the host; device forward never imports Torch or
calls Python per battle. This measures compute, not production-encoder parity,
training, value calibration, strength, or completed-game throughput.
"""
from dataclasses import asdict,dataclass
import hashlib
import json
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

    def __post_init__(self):
        if any(not isinstance(x,int) or x<=0 for x in (self.numeric,self.categorical,self.tokens,
                self.vocabulary,self.token_types,self.width,self.layers,self.heads,self.feedforward)):
            raise ValueError('positive integer transformer dimensions required')
        if self.width%self.heads or self.tokens<22:
            raise ValueError('head division and action tokens13..21 required')


def shapes(c):
    d=c.width
    result={'category_embedding.weight':(c.vocabulary,d),'token_type_embedding.weight':(c.token_types,d),
        'history_position_embedding.weight':(1,d),'numeric_projection.weight':(d,c.numeric),
        'numeric_projection.bias':(d,),'policy_head.weight':(1,d),'policy_head.bias':(1,),
        'value_head.weight':(1,d),'value_head.bias':(1,),
        'opponent_action_head.weight':(9,d),'opponent_action_head.bias':(9,)}
    for layer in range(c.layers):
        p=f'encoder.layers.{layer}.'
        result.update({p+'self_attn.in_proj_weight':(3*d,d),p+'self_attn.in_proj_bias':(3*d,),
            p+'self_attn.out_proj.weight':(d,d),p+'self_attn.out_proj.bias':(d,),
            p+'linear1.weight':(c.feedforward,d),p+'linear1.bias':(c.feedforward,),
            p+'linear2.weight':(d,c.feedforward),p+'linear2.bias':(d,),
            p+'norm1.weight':(d,),p+'norm1.bias':(d,),p+'norm2.weight':(d,),p+'norm2.bias':(d,)})
    return result


def validate_weights(params,c):
    expected=shapes(c)
    if set(params)!=set(expected):raise ValueError('checkpoint tensor keys differ from supported architecture')
    for name,shape in expected.items():
        a=np.asarray(params[name])
        if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise ValueError(f'invalid float32 tensor {name}: {a.shape}/{a.dtype}, expected {shape}')


def export(checkpoint,out):
    """CPU-only, safe Torch loader; exclusive NPZ output outside version control."""
    import torch
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=payload['model_config']
    if (cfg['window_size']!=1 or cfg.get('temporal_aggregator','mean')!='mean'
            or cfg.get('value_head_hidden_layers',0) not in (0,None)
            or cfg.get('value_activation','tanh') not in ('linear','tanh')):
        raise ValueError('only one-frame mean pooling and linear/tanh single-layer value heads supported')
    c=Config(numeric=cfg['numeric_feature_count'],categorical=cfg['categorical_feature_count'],
        tokens=cfg['token_count'],vocabulary=cfg['categorical_vocab_size'],token_types=cfg['token_type_vocab_size'],
        width=cfg['embedding_dim'],layers=cfg['transformer_layers'],heads=cfg['attention_heads'],
        feedforward=cfg['feedforward_dim'],tanh_value=cfg.get('value_activation','tanh')=='tanh')
    params={k:v.detach().cpu().numpy() for k,v in payload['state_dict'].items()}
    validate_weights(params,c)
    meta={'schema':'pokezero.gpu_champion_forward.v1','config':asdict(c),
        'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'parameter_count':sum(a.size for a in params.values())}
    encoded=np.frombuffer(json.dumps(meta,sort_keys=True).encode(),np.uint8)
    with Path(out).open('xb') as stream:np.savez_compressed(stream,metadata_json=encoded,**params)
    return meta


def load(path):
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(data['metadata_json'].tobytes())
        if meta['schema']!='pokezero.gpu_champion_forward.v1':raise ValueError('unsupported weight archive')
        c=Config(**meta['config']);params={k:data[k] for k in data.files if k!='metadata_json'}
    validate_weights(params,c)
    if meta['parameter_count']!=sum(a.size for a in params.values()):raise ValueError('parameter count mismatch')
    return {k:j.asarray(v) for k,v in params.items()},c,meta


class Output(NamedTuple):
    policy_logits: jax.Array
    value: jax.Array
    opponent_action_logits: jax.Array


def forward(params,numeric,categorical,token_types,attention,c):
    """Float32 eval-mode forward, arbitrary leading batch/seat axes.

    At least one attended token per row is required (the placeholder always
    includes field/action tokens). Padding masks keys, not queries, just as the
    production TransformerEncoder. No dropout, history beyond1, GRU or training.
    """
    leading=numeric.shape[:-2]
    if (numeric.shape[-2:]!=(c.tokens,c.numeric) or categorical.shape!=leading+(c.tokens,c.categorical)
            or token_types.shape!=leading+(c.tokens,) or attention.shape!=leading+(c.tokens,)):
        raise ValueError('observation tensor shapes differ from weight config')
    linear=lambda x,p:j.matmul(x,params[p+'.weight'].T,precision=jax.lax.Precision.HIGHEST)+params[p+'.bias']
    cats=j.clip(categorical,0,c.vocabulary-1)
    types=j.clip(token_types,0,c.token_types-1)
    x=j.sum(params['category_embedding.weight'][cats],axis=-2)
    x=x+params['token_type_embedding.weight'][types]+linear(numeric.astype(j.float32),'numeric_projection')
    x=x+params['history_position_embedding.weight'][0]
    x=x.reshape((-1,c.tokens,c.width));valid=attention.reshape((-1,c.tokens)).astype(j.bool_)
    def norm(a,p):
        mean=j.mean(a,axis=-1,keepdims=True)
        variance=j.mean(j.square(a-mean),axis=-1,keepdims=True)
        return (a-mean)*jax.lax.rsqrt(variance+1e-5)*params[p+'.weight']+params[p+'.bias']
    for layer in range(c.layers):
        p=f'encoder.layers.{layer}.';a=norm(x,p+'norm1')
        packed=j.matmul(a,params[p+'self_attn.in_proj_weight'].T,
            precision=jax.lax.Precision.HIGHEST)+params[p+'self_attn.in_proj_bias']
        q,k,v=[part.reshape((-1,c.tokens,c.heads,c.width//c.heads)).transpose((0,2,1,3))
            for part in j.split(packed,3,axis=-1)]
        scores=j.einsum('bhtd,bhsd->bhts',q,k,precision=jax.lax.Precision.HIGHEST)/j.sqrt(j.asarray(c.width//c.heads,j.float32))
        probs=jax.nn.softmax(j.where(valid[:,None,None,:],scores,-j.inf),axis=-1)
        attended=j.einsum('bhts,bhsd->bhtd',probs,v,precision=jax.lax.Precision.HIGHEST).transpose((0,2,1,3)).reshape(x.shape)
        x=x+linear(attended,p+'self_attn.out_proj')
        x=x+linear(jax.nn.gelu(linear(norm(x,p+'norm2'),p+'linear1'),approximate=False),p+'linear2')
    pooled=j.sum(j.where(valid[...,None],x,0),axis=1)/j.maximum(1,j.sum(valid,axis=1,keepdims=True))
    raw=linear(pooled,'value_head')[...,0]
    return Output(linear(x[:,13:22],'policy_head')[...,0].reshape(leading+(9,)),
        (j.tanh(raw) if c.tanh_value else raw).reshape(leading),
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
