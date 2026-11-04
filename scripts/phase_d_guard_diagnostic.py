#!/usr/bin/env python3
"""Collect all independently evaluable guards on the original frozen roots.

No sampled worlds, physics, evaluator, forward, or coverage admission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main(args):
    os.environ['JAX_PLATFORMS']='cpu'
    os.environ['JAX_COMPILATION_CACHE_DIR']=args.compilation_cache
    import jax
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_default_matmul_precision','highest')
    from pokezero import phase_d_owning_capture as capture,phase_d_ladder_admission as admission
    from pokezero.live_protocol_actor import ActorContract,_packet
    from pokezero.live_protocol_runtime import CompiledEncoderFunctions
    from pokezero.recorded_encoder_runner import JaxRecordedAdapter
    from pokezero.v5_contract import DecisionAddress
    from pokezero.v5_teacher_public_root import capture_root
    from gpu_engine.phase_d_protocol_constructor import load_inputs,read_bound
    from gpu_engine.phase_d_guard_diagnostic import scan,summarize
    from pokezero.phase_d_ladder_panel import write_new
    if any(d.platform!='cpu'for d in jax.devices()):raise ValueError('CPU diagnostic only')
    catalog,source,metadata,config=load_inputs(args)
    roster=read_bound(args.roster,args.expected_roster_sha256)['selected']
    if len(roster)!=200:raise ValueError('same original200 roots required')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    settings=dict(roster_sha256=args.expected_roster_sha256,source_sha256=source.source_hash,
        metadata_sha256=args.expected_metadata_sha256,seed=20261103,coverage_claim=False)
    p=root/'settings.json'
    if p.exists():
        if json.loads(p.read_bytes())!=settings:raise ValueError('diagnostic bindings differ')
    else:write_new(p,settings)
    functions=CompiledEncoderFunctions(catalog);records=[]
    for ordinal,meta in enumerate(roster):
        p=root/'records'/f'{ordinal}.json'
        if p.exists():records.append(json.loads(p.read_bytes()));continue
        claim=root/'claims'/f'{ordinal}.json'
        record=dict(ordinal=ordinal,address=admission.address(meta),band=admission.band(meta['turn']))
        try:
            if claim.exists():raise ValueError('INTERRUPTED_DIAGNOSTIC_ROOT_UNKNOWN')
            write_new(claim,dict(ordinal=ordinal,address=admission.address(meta)))
            found=next((x for x in capture.read_capture(meta['capture_path'])if x['capture_row']==meta['capture_row']),None)
            if found is None or admission.address(found)!=admission.address(meta):raise ValueError('original captured root unavailable')
            lines=tuple(found['lines']);owner=int(found['owner'][1])-1
            if hashlib.sha256(''.join(x+'\n'for x in lines).encode()).hexdigest()!=found['prefix_sha256']or lines[-1]!=found['own_request_line']:
                raise ValueError('original owning prefix/request differs')
            public=capture_root(config=config,contract=ActorContract.champion(),lines=lines,
                address=DecisionAddress(found['stream_id'],found['decision_index'],owner),battle_id=found['battle_id'],seed=20261103)
            adapter=JaxRecordedAdapter(catalog,owner_side=owner,compiled_functions=functions)
            adapter.reset(stream_id=found['stream_id'],battle_id=found['battle_id'])
            adapter.feed(tuple(x for x in lines if not x.startswith('|request|')))
            record['diagnostic']=scan(adapter,_packet(found['own_request_line'],found['owner']),public,source,metadata,lines=lines)
        except (ValueError,KeyError,TypeError,IndexError,AttributeError,OSError)as error:
            if isinstance(error,PermissionError):raise
            record['diagnostic']=dict(checks=[dict(kind='DIAGNOSTIC_ROOT_PREREQUISITE_UNAVAILABLE',status='NOT_EVALUABLE',details=dict(error=str(error)))],
                all_refusals=[],not_evaluable=['DIAGNOSTIC_ROOT_PREREQUISITE_UNAVAILABLE'],coverage_claim=False)
        write_new(p,record);records.append(record)
        print(json.dumps(dict(ordinal=ordinal,band=record['band'],refusals=record['diagnostic']['all_refusals'],unavailable=record['diagnostic']['not_evaluable'])),flush=True)
    result=dict(settings=settings,summary=summarize(records),world_budgets=[16,32],
        budget_scope='Only independent pre-sampling guards; identical at16/32. No sampled-spec guard fabricated.',coverage_claim=False)
    write_new(root/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('showdown-root','node-binary','metadata','expected-metadata-sha256','vocabulary','expected-vocabulary-sha256',
                 'roster','expected-roster-sha256','output','compilation-cache'):
        p.add_argument('--'+name,required=True)
    print(json.dumps(main(p.parse_args()),sort_keys=True,allow_nan=False))
