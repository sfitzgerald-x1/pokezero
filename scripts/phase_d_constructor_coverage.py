#!/usr/bin/env python3
"""Finite host-only construction of the fixed genuine FP roots at16/32 worlds.

Platform is selected before any JAX-dependent import. There is no archive,
model, forward, engine step, search listener or GPU timing in this program.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('showdown-root','node-binary','metadata','expected-metadata-sha256','vocabulary',
                 'expected-vocabulary-sha256','roster','expected-roster-sha256','producer-review',
                 'expected-producer-review-sha256','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--worlds',type=int,choices=(16,32),required=True)
    p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--platform',choices=('cpu',),default='cpu')
    p.add_argument('--compilation-cache')
    return p


def select_platform(args):
    if 'jax'in sys.modules:raise ValueError('select host coverage platform before importing JAX')
    os.environ['JAX_PLATFORMS']=args.platform
    if args.compilation_cache:os.environ['JAX_COMPILATION_CACHE_DIR']=args.compilation_cache


def build(args):
    import jax
    jax.config.update('jax_enable_x64',False);jax.config.update('jax_default_matmul_precision','highest')
    if any(device.platform!='cpu'for device in jax.devices()):raise ValueError('host-only constructor coverage requires CPU')
    from gpu_engine.phase_d_protocol_constructor import ProtocolConstructor,load_inputs
    catalog,source,metadata,config=load_inputs(args)
    return ProtocolConstructor(catalog=catalog,source=source,metadata=metadata,
        owning_config=config,count=args.worlds,seed=args.seed,saved_root_admission=True)


def measure(args,constructor):
    from pokezero import phase_d_owning_capture as capture,phase_d_ladder_admission as admission
    from pokezero.phase_d_search_server import SCHEMA
    from pokezero.phase_d_ladder_panel import read_bound,write_new
    from gpu_engine.phase_d_protocol_constructor import ProtocolConstructor
    if type(constructor)is not ProtocolConstructor or constructor.count!=args.worlds:
        raise ValueError('same genuine typed protocol constructor and fixed budget required')
    declaration=read_bound(args.roster,args.expected_roster_sha256)
    read_bound(args.producer_review,args.expected_producer_review_sha256)
    roster=declaration['selected']
    if len(roster)!=200:raise ValueError('same frozen200 genuine root roster required')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    settings=dict(roster_sha256=args.expected_roster_sha256,producer_review_sha256=args.expected_producer_review_sha256,
        metadata_sha256=args.expected_metadata_sha256,vocabulary_sha256=args.expected_vocabulary_sha256,
        worlds=args.worlds,seed=args.seed,platform=args.platform,constructor_only=True,
        addresses=[capture.encoded(admission.address(row)).decode()for row in roster])
    settings_path=root/'settings.json'
    if settings_path.exists():
        if settings_path.read_bytes()!=capture.encoded(settings):raise ValueError('original constructor settings changed')
    else:write_new(settings_path,settings)
    records=[]
    for ordinal,meta in enumerate(roster):
        target=root/'records'/f'{ordinal}.json';claim=root/'claims'/f'{ordinal}.json'
        if target.exists():records.append(json.loads(target.read_bytes()));continue
        record=dict(address=admission.address(meta),requested_worlds=args.worlds,constructed_worlds=0,decision_seconds=None,refusals=[])
        if claim.exists():record['refusals']=['INTERRUPTED_ORIGINAL_ATTEMPT_UNKNOWN_NO_REPLAY']
        else:
            write_new(claim,dict(ordinal=ordinal,address=admission.address(meta),settings_sha256=hashlib.sha256(capture.encoded(settings)).hexdigest()))
            try:
                found=next((row for row in capture.read_capture(meta['capture_path'])if row['capture_row']==meta['capture_row']),None)
                if found is None or admission.address(found)!=admission.address(meta):raise ValueError('same original genuine captured root unavailable')
                request=dict(schema=SCHEMA,request_id=args.expected_roster_sha256+':CONSTRUCTOR:'+str(args.worlds)+':'+str(ordinal),
                    **{key:found[key]for key in ('battle_id','owner','stream_id','decision_index','generation','prefix_sha256','lines','own_request_line')})
                result=constructor.construct(request,ordinal)
                if result.get('requested_worlds')!=args.worlds:raise ValueError('constructor used different world budget')
                record['constructed_worlds']=result['constructed_worlds'];record['result']=result
                if result['status']=='COMPLETE':record['constructor_seconds']=result['constructor_seconds']
                else:
                    reasons=[reason for row in result.get('world_refusals',())for reason in row['kinds']]
                    record['refusals']=sorted(set(reasons or[result.get('refusal')or'UNNAMED_CONSTRUCTOR_REFUSAL']))
            except (ValueError,TypeError,KeyError,IndexError,OSError)as error:
                if isinstance(error,PermissionError):raise
                record['refusals']=[type(error).__name__+': '+str(error)]
        write_new(target,record);records.append(record)
        print(json.dumps(dict(ordinal=ordinal,constructed=record['constructed_worlds'],requested=args.worlds,refusals=record['refusals'])),flush=True)
    summary=admission.summarize(roster,records,worlds=args.worlds,constructor_only=True)
    value=dict(settings=settings,summary=summary,original_roster_count=len(roster),measured_roots=len(roster),
        scope='HOST_CONSTRUCTOR_ONLY_NO_MODEL_FORWARD_SEARCH_TIME_OR_GPU_CLAIM',
        stop_rule='One declared measurement; interrupted cells remain unknown, resume only not-attempted slots. One refusal-driven fix/retest budget remains shared with GPU constructor coverage.')
    target=root/'result.json'
    if target.exists():
        if target.read_bytes()!=capture.encoded(value):raise ValueError('immutable completed constructor result differs')
    else:write_new(target,value)
    return value


if __name__=='__main__':
    args=parser().parse_args();select_platform(args)
    print(json.dumps(measure(args,build(args)),sort_keys=True,allow_nan=False))
