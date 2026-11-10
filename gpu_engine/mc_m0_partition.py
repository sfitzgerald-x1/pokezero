"""Explicit disjoint root partitions using the already-qualified M0 core.

This does not stop workers, retry roots, change the roster or declare the whole
experiment complete. The caller must freeze disjoint assignments and ensure an
index has no live owner before launching it. Original results stay intact.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from gpu_engine import mc_m0 as m0, mc_root_labeller as lab, fp_reply_data as fp


def assignment(text):
    try:indices=tuple(int(x) for x in text.split(','))
    except ValueError:raise ValueError('comma-separated root indices required') from None
    if not indices or len(set(indices))!=len(indices) or any(not 0<=i<50 for i in indices):
        raise ValueError('unique root indices in 0..49 required')
    return indices


def qualify(directory):
    directory=Path(directory);config=json.loads((directory/'config.json').read_bytes())
    manifest=json.loads((directory/'manifest.json').read_bytes())
    if m0.sha(directory/'config.json')!=manifest['config_sha256']:raise ValueError('qualification config changed')
    modules={'mc_m0.py':m0,'mc_root_labeller.py':lab,'fp_reply_data.py':fp}
    for name,module in modules.items():
        if m0.sha(module.__file__)!=manifest['source_sha256'][name]:
            raise ValueError('qualified core changed: '+name)
    smoke=json.loads((directory/'runtime-smoke.json').read_bytes())
    if len(smoke)!=2 or {r['kind'] for r in smoke}!={'self','foulplay'} or not all(r['status']=='PASS' for r in smoke):
        raise ValueError('both runtime qualifications required')
    reference=json.loads((directory/'prepared.json').read_bytes())
    if len(reference)!=50 or {r['index'] for r in reference}!=set(range(50)) or any(r['status']!='READY' for r in reference):
        raise ValueError('the same 50 READY roots required')
    return config,{r['index']:r for r in reference},manifest


def require_same_root(frozen, reconstructed, root):
    if root!=frozen['root'] or reconstructed.get('source_hashes')!=frozen.get('source_hashes'):
        raise ValueError('root or source inputs differ from frozen qualified preparation')


def validate_dispatch(plan,indices,worker,manifest):
    if plan['config_sha256']!=manifest['config_sha256'] or plan['core_sha256']!=manifest['source_sha256']:
        raise ValueError('dispatch plan is not bound to qualified inputs')
    if not 0<=worker<len(plan['groups']) or list(indices)!=plan['groups'][worker]:raise ValueError('worker assignment differs')
    flat=[i for group in plan['groups'] for i in group]
    if any(type(i) is not int or not 0<=i<50 for i in flat) or len(flat)!=len(set(flat)):
        raise ValueError('overlapping or invalid root assignments')
    retained={i for i in range(50) if all(f'{i}:{k}' in plan['reuse'] for k in (64,256))}
    if retained & set(flat) or retained | set(flat)!=set(range(50)):
        raise ValueError('dispatch must cover exactly the unfinished frozen roots')
    if plan['serial_release']['status']!='CONTROLLED_HANDOFF_AFTER_COMPLETE_ROOT':raise ValueError('serial release not certified')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--qualified-run',required=True)
    p.add_argument('--indices',required=True);p.add_argument('--out',required=True)
    p.add_argument('--plan',required=True);p.add_argument('--worker',type=int,required=True)
    args=p.parse_args(argv);indices=assignment(args.indices)
    config,reference,manifest=qualify(args.qualified_run);m0.validate(config)
    from gpu_engine.mc_m0_pool import process, validate_stage
    plan_path=Path(args.plan).resolve();plan=json.loads(plan_path.read_bytes())
    validate_dispatch(plan,indices,args.worker,manifest)
    if Path(plan['qualified_run']).resolve()!=Path(args.qualified_run).resolve():raise ValueError('qualified directory differs')
    handoff=Path(args.qualified_run)/'handoff-claim.json'
    claim=json.loads(handoff.read_bytes())
    if (m0.sha(handoff)!=plan['handoff_claim_sha256'] or Path(claim['pool_out']).resolve()!=plan_path.parent
            or claim['config_sha256']!=manifest['config_sha256']):raise ValueError('shared handoff ownership differs')
    release=plan['serial_release'];old=process(release['pid'])
    if old is not None and old['start']==release['start']:raise ValueError('serial worker still owns the roster')
    claims=Path(args.qualified_run)/'root-claims';claims.mkdir(exist_ok=True)
    for i in indices:
        m0.write(claims/f'root-{i:03d}.json',dict(pid=os.getpid(),worker=args.worker,index=i,plan_sha256=m0.sha(plan_path)))
    # Runtime checks match the original driver. This adapter changes dispatch,
    # not continuation policies, sampling, batching shape or outcome handling.
    import subprocess
    base=config['base_config']
    if subprocess.check_output(['git','-C',base['showdown_root'],'rev-parse','HEAD'],text=True).strip()!=config['showdown_git_sha']:
        raise ValueError('simulator identity differs')
    if subprocess.check_output([base['node_binary'],'--version'],text=True).strip()!=config['node_version']:
        raise ValueError('Node identity differs')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    m0.write(out/'assignment.json',dict(indices=indices,global_roots=50,qualified_run=args.qualified_run,
             config_sha256=manifest['config_sha256'],core_sha256=manifest['source_sha256'],
             adapter_sha256=m0.sha(__file__),plan_sha256=m0.sha(plan_path)))
    start,cpu0=time.perf_counter(),m0.process_cpu();completed=[];envs=[]
    try:
        import torch
        torch.set_num_threads(config['threads']);torch.set_num_interop_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)
        champion=lab.Model(config['checkpoint'],torch);peer=lab.Model(config['pi_fp'],torch,opponent=True)
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.neural_policy import category_vocab_from_model_config,feature_masks_from_model_config,observation_spec_from_model_config
        peer.views=dict(source=load_gen3_randbat_source_cached(base['showdown_root']),dex=load_showdown_dex_cached(base['showdown_root']),
            vocab=category_vocab_from_model_config(peer.result.model_config,base['showdown_root']),
            masks=feature_masks_from_model_config(peer.result.model_config),spec=observation_spec_from_model_config(peer.result.model_config))
        for _ in range(config['envs']):envs.append(lab.make_env(base,champion.result,peer.result))
        ready,audit=m0.prepare(config,champion,peer,envs[0]);m0.write(out/'prepared.json',audit)
        reconstructed={r['index']:r for r in audit}
        selected={i:(r,s) for i,r,s in ready if i in indices}
        if set(selected)!=set(indices):raise ValueError('an assigned root could not be reconstructed; no replacements')
        for i in indices:
            root,snapshot=selected[i]
            require_same_root(reference[i],reconstructed[i],root)
            for env in envs:env.reset(seed=root['seed'])
            peer_model=peer if root['kind']=='foulplay' else champion
            for batched,k in ((False,64),(True,256)):
                reused=plan['reuse'].get(f'{i}:{k}')
                if reused is not None:
                    source=Path(reused['path'])
                    if m0.sha(source)!=reused['sha256']:raise ValueError('retained stage changed')
                    validate_stage(json.loads(source.read_bytes()),reference)
                    os.link(source,out/f'root-{i:03d}-{k}.json');continue
                at,ct=time.perf_counter(),m0.process_cpu()
                print(json.dumps(dict(event='PARTITION_STAGE_START',index=i,K=k)),flush=True)
                y,decisions,statuses=lab.run_root(envs,dict(ours=champion,opp=peer_model),root,snapshot,k,batched)
                row=dict(index=i,root=root,mode='batched' if batched else 'unbatched',K=k,
                    outcomes=[[None if np.isnan(v) else float(v) for v in a] for a in y],statuses=statuses,
                    decisions=decisions.tolist(),pairs={str(n):m0.pair_stats(y,statuses,n) for n in (64,256) if n<=k},
                    wall_seconds=time.perf_counter()-at,cpu_seconds=m0.process_cpu()-ct)
                m0.write(out/f'root-{i:03d}-{k}.json',row)
            completed.append(i)
            print(json.dumps(dict(event='PARTITION_ROOT_COMPLETE',index=i,completed=len(completed),assigned=len(indices))),flush=True)
        m0.write(out/'partition-summary.json',dict(status='PARTITION_COMPLETE',assigned=indices,completed=completed,
                  global_roots=50,wall_seconds=time.perf_counter()-start,cpu_seconds=m0.process_cpu()-cpu0))
        return 0
    except Exception as error:
        m0.write(out/'failure-summary.json',dict(status='FAILED',assigned=indices,completed=completed,
                  reason=type(error).__name__+': '+str(error),wall_seconds=time.perf_counter()-start,cpu_seconds=m0.process_cpu()-cpu0))
        raise
    finally:
        for env in envs:env.close()


if __name__=='__main__':raise SystemExit(main())
