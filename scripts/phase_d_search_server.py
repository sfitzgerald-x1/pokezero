#!/usr/bin/env python3
"""Explicit shared-GPU Phase D listener; no games or panel allocation."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess


def read_bound(path,digest):
    blob=Path(path).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('declared startup input hash differs: '+str(path))
    return json.loads(blob)


def build(args):
    import jax
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_default_matmul_precision','highest')
    if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':
        raise ValueError('exactly one visible GPU required before evaluator startup')
    from pokezero.live_protocol_actor import ActorContract,_vocab_sha
    from pokezero.category_vocab import CategoryVocabulary
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero import v5_contract
    from gpu_engine import public_belief,v4_encoder,transition,engine_device_observation,device_public_worlds
    from gpu_engine.phase_d_search import Search
    contract=ActorContract.champion()
    revision=subprocess.check_output(['git','-C',args.showdown_root,'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',args.showdown_root,'status','--porcelain','--untracked-files=no'],text=True)
    if revision!=contract.showdown_revision or dirty:raise ValueError('original clean pinned Showdown source required')
    vocabulary=CategoryVocabulary(**read_bound(args.vocabulary,args.expected_vocabulary_sha256))
    if _vocab_sha(vocabulary)!=contract.category_vocab_sha256:raise ValueError('original9375 vocabulary required')
    metadata=read_bound(args.metadata,args.expected_metadata_sha256)
    original=load_gen3_randbat_source_cached(args.showdown_root)
    catalog=v4_encoder.compile_encoder(public_belief.compile_source(original),load_showdown_dex_cached(args.showdown_root),
        vocabulary,checkpoint_sha256=contract.checkpoint_sha256,feature_masks=dict(contract.feature_masks))
    source=device_public_worlds.compile_source(original,metadata,expected_source_hash=contract.belief_set_source_hash,
        expected_source_rows_sha256=v5_contract.source_variant_census(original)['source_rows_sha256'])
    return Search(args.archive,expected_archive_sha256=args.expected_archive_sha256,catalog=catalog,source=source,
        metadata=metadata,engine_tables=transition.tables_from_metadata(metadata),
        observation_tables=engine_device_observation.compile_tables(metadata,catalog=catalog),count=args.worlds,
        seed=args.seed,chance_cap=args.chance_cap,lanes=args.lanes,compute_dtype=args.compute_dtype,saved_root_admission=args.saved_root_admission)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('showdown-root','metadata','expected-metadata-sha256','vocabulary','expected-vocabulary-sha256','archive','expected-archive-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,required=True)
    p.add_argument('--worlds',type=int,choices=(16,32),default=16);p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--chance-cap',type=int,default=262144);p.add_argument('--lanes',type=int,default=4096)
    p.add_argument('--compute-dtype',choices=('float32','bfloat16'),default='bfloat16')
    p.add_argument('--saved-root-admission',action='store_true',help='Fold each fixed saved prefix independently; no games or root refill')
    p.add_argument('--batch-requests',type=int,default=32);p.add_argument('--batch-wait-seconds',type=float,default=.005)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    from pokezero.phase_d_search_server import Server
    search=build(args)
    asyncio.run(Server(search,max_requests=args.batch_requests,batch_wait_seconds=args.batch_wait_seconds).serve(args.host,args.port))
