#!/usr/bin/env python3
"""One explicit champion GPU listener for one fixed rung or constructor table."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess


def read_bound(path,digest):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('original bound startup input differs')
    return json.loads(raw)


def build(args):
    import jax
    jax.config.update('jax_enable_x64',False);jax.config.update('jax_default_matmul_precision','highest')
    if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':raise ValueError('ONE shared GPU required')
    from pokezero.live_protocol_actor import ActorContract,_vocab_sha
    from pokezero.category_vocab import CategoryVocabulary
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.local_showdown import LocalShowdownConfig
    from pokezero.observation import ObservationFeatureMasks
    from pokezero import v5_contract
    from gpu_engine import public_belief,v4_encoder,device_public_worlds,transition,engine_device_observation
    from gpu_engine.phase_d_ladder_search import Search
    contract=ActorContract.champion();revision=subprocess.check_output(['git','-C',args.showdown_root,'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',args.showdown_root,'status','--porcelain','--untracked-files=no'],text=True)
    if revision!=contract.showdown_revision or dirty:raise ValueError('original clean pinned Showdown required')
    vocab=CategoryVocabulary(**read_bound(args.vocabulary,args.expected_vocabulary_sha256))
    if _vocab_sha(vocab)!=contract.category_vocab_sha256:raise ValueError('original9375 vocabulary required')
    metadata=read_bound(args.metadata,args.expected_metadata_sha256)
    source=load_gen3_randbat_source_cached(args.showdown_root)
    catalog=v4_encoder.compile_encoder(public_belief.compile_source(source),load_showdown_dex_cached(args.showdown_root),
        vocab,checkpoint_sha256=contract.checkpoint_sha256,feature_masks=dict(contract.feature_masks))
    bound=device_public_worlds.compile_source(source,metadata,expected_source_hash=contract.belief_set_source_hash,
        expected_source_rows_sha256=v5_contract.source_variant_census(source)['source_rows_sha256'])
    masks=dict(contract.feature_masks);masks['opponent_tendency_stats_block']=masks.pop('stats_block')
    config=LocalShowdownConfig(showdown_root=args.showdown_root,node_binary=args.node_binary,observation_spec=v5_contract.V4_SPEC,
        category_vocab=vocab,feature_masks=ObservationFeatureMasks(**masks),set_belief_source=True)
    return Search(args.archive,expected_archive_sha256=args.expected_archive_sha256,catalog=catalog,source=bound,metadata=metadata,
        engine_tables=transition.tables_from_metadata(metadata),observation_tables=engine_device_observation.compile_tables(metadata,catalog=catalog),
        rung=args.rung,owning_config=config,seed=args.seed,chance_cap=args.branch_cap,lanes=args.lanes,
        compute_dtype=args.compute_dtype,saved_root_admission=args.saved_root_admission,
        rollout_max_requests=args.rollout_max_requests,constructor_worlds=args.constructor_worlds)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('showdown-root','node-binary','metadata','expected-metadata-sha256','vocabulary','expected-vocabulary-sha256','archive','expected-archive-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--rung',choices=('R1','R3','R4','R7','R5'),required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,required=True)
    p.add_argument('--seed',type=int,default=20261103);p.add_argument('--branch-cap',type=int,default=262144)
    p.add_argument('--lanes',type=int,default=4096);p.add_argument('--compute-dtype',choices=('float32','bfloat16'),default='bfloat16')
    p.add_argument('--saved-root-admission',action='store_true');p.add_argument('--constructor-worlds',type=int,choices=(16,32))
    p.add_argument('--rollout-max-requests',type=int,default=1024)
    p.add_argument('--batch-requests',type=int,default=32);p.add_argument('--batch-wait-seconds',type=float,default=.005)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    from pokezero.phase_d_ladder_transport import Server
    asyncio.run(Server(build(args),max_requests=args.batch_requests,batch_wait_seconds=args.batch_wait_seconds).serve(args.host,args.port))
