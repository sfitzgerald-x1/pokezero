"""Compare the entire attested trusted corpus; never claim player-wire parity.

Run on remote storage/compute. An execution owner must bind this source and
dependencies, enforce deadlines, and retain both success and failure receipts.
No simulator or new game is started. The original champion is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def snapshot_checkpoint(source,target,expected_sha256):
    """Deserialize only a private snapshot of the exact bytes hashed once."""
    blob=Path(source).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=expected_sha256:raise ValueError('champion snapshot SHA mismatch')
    with Path(target).open('xb') as stream:stream.write(blob)
    Path(target).chmod(0o444)
    return dict(sha256=expected_sha256,bytes=len(blob))


def publish_comparison(output,blob):
    """Publish a complete report atomically without replacing prior evidence."""
    target=Path(output)/'comparison.json';pending=Path(output)/'.comparison.json.pending';created=False
    try:
        with pending.open('xb') as stream:
            created=True;stream.write(blob);stream.flush();os.fsync(stream.fileno())
        # Same-directory hard link is atomic and refuses an existing final name.
        os.link(pending,target)
    finally:
        if created:pending.unlink()


def build_runtime(args):
    # CPU-only is a scoped B2 comparison, independent of GPU throughput gates.
    os.environ['JAX_PLATFORMS']='cpu'
    node=Path(args.node).resolve(strict=True)
    if node.name!='node' or not node.is_file() or not os.access(node,os.X_OK):raise ValueError('bound Node executable required')
    os.environ['PATH']=str(node.parent)+os.pathsep+os.environ.get('PATH','')
    if Path(shutil.which('node')).resolve()!=node:raise ValueError('Dex metadata Node differs from bound runtime')
    from pokezero.mirror_value_probe import load_runtime,CHAMPION_SHA
    from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
    checkpoint=snapshot_checkpoint(args.checkpoint,args.scratch_checkpoint,CHAMPION_SHA)
    model,_,vocab,_,contract=load_runtime(args.scratch_checkpoint,args.showdown,str(node),args.threads)
    if contract!=CHAMPION_CONTRACT:raise ValueError('exact recorded champion contract required')
    import jax
    from pokezero.neural_policy import require_torch
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.recorded_encoder_runner import JaxRecordedAdapter
    from gpu_engine import public_belief,v4_encoder
    torch=require_torch();torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision('highest')
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_default_matmul_precision','highest')
    if any(device.platform!='cpu' for device in jax.devices()):raise ValueError('CPU-only comparison required')
    source=load_gen3_randbat_source_cached(args.showdown)
    if source.metadata.source_hash!=contract['belief_set_source_hash']:raise ValueError('source hash differs from recorded contract')
    catalog=v4_encoder.compile_encoder(public_belief.compile_source(source),load_showdown_dex_cached(args.showdown),vocab,
        checkpoint_sha256=CHAMPION_SHA,feature_masks=contract['feature_masks'])
    adapter=JaxRecordedAdapter(catalog,model=model if args.forward else None,torch_module=torch if args.forward else None)
    receipt=dict(checkpoint=checkpoint,contract=contract,devices=[str(d) for d in jax.devices()],
        torch_version=torch.__version__,jax_version=jax.__version__,mha_fastpath=False,
        torch_float32_matmul_precision=torch.get_float32_matmul_precision(),jax_enable_x64=False,
        jax_default_matmul_precision='highest',threads=args.threads,forwards_requested=args.forward,
        source_hash=source.metadata.source_hash,node_sha256=hashlib.sha256(node.read_bytes()).hexdigest())
    return adapter,receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','scratch-checkpoint','showdown','node','collection','collection-sha256','cohort-sha256','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--mode',choices=('inputs-only','paired-forward'),required=True)
    args=parser.parse_args(argv)
    if not 1<=args.threads<=8:parser.error('bounded CPU thread count1..8 required')
    args.forward=args.mode=='paired-forward'
    output=Path(args.output);output.mkdir(exist_ok=False)
    start=time.monotonic();receipt=dict(schema='pokezero.B2-recorded-CPU-attempt.v1',status='FAILED',actual_exit=1,
        new_games=0,simulator_steps=0,public_wire_qualified=False,full_encoder_admitted=False)
    status=1
    try:
        adapter,runtime=build_runtime(args)
        receipt['runtime']=runtime
        from pokezero.recorded_encoder_collection import compare_collection
        result=compare_collection(args.collection,expected_collection_sha256=args.collection_sha256,
            expected_cohort_sha256=args.cohort_sha256,adapter=adapter,with_forward=args.forward)
        blob=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        publish_comparison(output,blob)
        receipt.update(status='COMPLETE_FIXED_TRUSTED_SOURCE_METRICS_ONLY',actual_exit=0,
            comparison_sha256=hashlib.sha256(blob).hexdigest(),comparison_bytes=len(blob),views=result['metrics']['views'])
        status=0
    except BaseException as error:
        # No partial tally or completed qualification is returned after a late
        # source/runtime failure. The original inputs remain read-only.
        receipt.update(error_type=type(error).__name__,error=str(error)[:4000])
    finally:
        receipt['seconds']=time.monotonic()-start
        with (output/'attempt.json').open('x') as stream:
            json.dump(receipt,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
        print(json.dumps(receipt,sort_keys=True,allow_nan=False),flush=True)
    return status


if __name__=='__main__':sys.exit(main())
