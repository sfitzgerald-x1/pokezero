"""Compare the functional JAX port with the production Torch model on CPU.

This is numerical forward validation on authored inputs, not encoder parity,
strength or a game-rate measurement. No checkpoint or model is changed.
"""
import argparse
import json
import numpy as np
import torch

from . import champion


def check(checkpoint,archive):
    from pokezero.neural_policy import EntityTokenTransformerPolicy,TransformerPolicyConfig
    torch.set_num_threads(2)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    config=TransformerPolicyConfig.from_dict(payload['model_config'])
    reference=EntityTokenTransformerPolicy(config).eval()
    reference.load_state_dict(payload['state_dict'],strict=True)
    params,c,meta=champion.load(archive)
    import hashlib
    from pathlib import Path
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()!=meta['checkpoint_sha256']:
        raise ValueError('archive/checkpoint binding mismatch')
    rows=[]
    for seed in (701,702):
        rng=np.random.default_rng(seed)
        numeric=rng.normal(0,.1,(2,2,c.tokens,c.numeric)).astype(np.float32)
        cat=rng.integers(-2,c.vocabulary+2,(2,2,c.tokens,c.categorical),dtype=np.int32)
        types=rng.integers(-2,c.token_types+2,(2,2,c.tokens),dtype=np.int32)
        attention=rng.random((2,2,c.tokens))>.25;attention[...,0]=True
        with torch.no_grad():
            out=reference(numeric_features=torch.from_numpy(numeric.reshape(4,1,c.tokens,c.numeric)),
                categorical_ids=torch.from_numpy(cat.reshape(4,1,c.tokens,c.categorical)),
                token_type_ids=torch.from_numpy(types.reshape(4,1,c.tokens)),
                attention_mask=torch.from_numpy(attention.reshape(4,1,c.tokens)),
                history_mask=torch.ones((4,1),dtype=torch.bool))
        actual=champion.forward(params,numeric,cat,types,attention,c)
        for name,a in zip(champion.Output._fields,actual):
            expected=getattr(out,name).numpy().reshape(a.shape)
            value=np.asarray(a)
            np.testing.assert_allclose(value,expected,rtol=1e-4,atol=1e-4)
            rows.append({'seed':seed,'output':name,'maximum_absolute_error':float(np.max(np.abs(value-expected)))})
    return {'checkpoint_sha256':meta['checkpoint_sha256'],'parameter_count':meta['parameter_count'],
        'devices':[str(x) for x in champion.jax.devices()],'rtol':1e-4,'atol':1e-4,'comparisons':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--archive',required=True);a=p.parse_args()
    print(json.dumps(check(a.checkpoint,a.archive),sort_keys=True))
