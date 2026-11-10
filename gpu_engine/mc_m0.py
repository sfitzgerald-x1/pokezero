"""Fixed-roster, real-Showdown M0. Configuration and raw outputs stay remote.

Run: python -m gpu_engine.mc_m0 --config CONFIG --out NEW_DIRECTORY
The configuration pins checkpoint/runtime identities and exactly 50 prospective
roots (25 self, 25 proxy-Foul-Play). No failed root is replaced. K64 is a nested
prefix of the batched K256 bank; unbatched K64 is a paired timing control.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import resource
import subprocess
import time

import numpy as np
from gpu_engine import mc_root_labeller as lab


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with open(path, 'x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False);f.write('\n');f.flush()


def observation_sha(observation):
    fields={name:np.asarray(getattr(observation,name)).tolist() for name in
            ('categorical_ids','numeric_features','token_type_ids','attention_mask','legal_action_mask')}
    return hashlib.sha256(json.dumps(fields,sort_keys=True,allow_nan=False).encode()).hexdigest()


def validate(config):
    roots = config['roots']
    if len(roots) != 50 or collections.Counter(r['kind'] for r in roots) != {'self':25, 'foulplay':25}:
        raise ValueError('exactly 50 roots, 25 per opponent stratum required')
    seen = set()
    for r in roots:
        if r['ours'] not in ('p1', 'p2'):raise ValueError('invalid seat')
        for k in ('seed', 'target', 'root_seed'):
            if type(r[k]) is not int or r[k] < 0:raise ValueError('nonnegative integer '+k)
        key = (r['kind'], r['seed'], r['ours'], r['target'])
        if key in seen:raise ValueError('duplicate root')
        seen.add(key)
    for k in ('checkpoint', 'pi_fp'):
        if sha(config[k]) != config[k+'_sha256']:raise ValueError(k+' digest differs')
    if type(config['envs']) is not int or not 1 <= config['envs'] <= 32:raise ValueError('1..32 envs required')
    if type(config['threads']) is not int or not 1 <= config['threads'] <= 8:raise ValueError('1..8 threads required')


def process_cpu():
    # The live descendant census includes Node bridge CPU; add already-reaped
    # children so closing an env does not erase failed preparation work.
    done = resource.getrusage(resource.RUSAGE_CHILDREN)
    return lab.process_cpu_seconds() + done.ru_utime + done.ru_stime


def prepare(config, champion, peer, env):
    from gpu_engine import fp_reply_data as fp
    prepared = [];audit = []
    for index, selected in enumerate(config['roots']):
        start, cpu0 = time.perf_counter(), process_cpu()
        try:
            if selected['kind'] == 'foulplay':
                game, reason = fp.load_game(Path(config['collection']), Path(config['collection'])/'results'/f"{selected['ordinal']}.json")
                if game is None:raise ValueError('source refused: '+reason)
                if (game['seed'],game['ours']) != (selected['seed'],selected['ours']):raise ValueError('source root identity differs')
                if game['base_config'] != config['base_config']:raise ValueError('source runtime differs')
                result = fp.replay_to_root(env, champion, game, random.Random(0), decision_index=selected['target'],
                                          expected_checkpoint_sha256=config['checkpoint_sha256'])
                snapshot = result['snapshot']
                source_hashes = {str(p.relative_to(Path(config['collection']))):sha(p) for p in (
                    Path(config['collection'])/'results'/f"{selected['ordinal']}.json",
                    Path(config['collection'])/'telemetry'/f"{selected['ordinal']}.jsonl",
                    Path(config['collection'])/'owning-wire'/f"{selected['ordinal']}.jsonl",
                    Path(config['collection'])/'requests'/f"{selected['ordinal']}.json")}
            else:
                if not lab.self_root(env, champion, selected['seed'], selected['ours'], selected['target']):
                    raise ValueError('self root unavailable or fewer than two actions')
                snapshot = env.snapshot();source_hashes = {}
            legal = lab.legal_of(env.observe(selected['ours']))
            actions = [int(x) for x in np.flatnonzero(legal)]
            if len(actions) < 2:raise ValueError('fewer than two legal actions')
            opp = 'p2' if selected['ours'] == 'p1' else 'p1'
            model = peer if selected['kind'] == 'foulplay' else champion
            replies = None
            if opp in env.requested_players():
                obs = model.observe(env, opp, selected['seed']);probs = model.probabilities([obs])[0]
                replies = [lab.sample(probs, lab.legal_of(obs), lab.draw(selected['root_seed'], k, 0, 1)) for k in range(256)]
            root = dict(selected, opp=opp, actions=actions, opp_replies=replies,
                        original_observation_sha256=observation_sha(env.observe(selected['ours'])))
            prepared.append((index, root, snapshot))
            audit.append(dict(index=index, status='READY', root=root, source_hashes=source_hashes))
        except Exception as error:
            audit.append(dict(index=index, status='REFUSED', reason=type(error).__name__+': '+str(error)))
        audit[-1].update(wall_seconds=time.perf_counter()-start, cpu_seconds=process_cpu()-cpu0)
    return prepared, audit


def pair_stats(values, statuses, k):
    """Terminal-only paired diagnostics; identification bounds include caps.

    A censored pair never enters SE/correlation as a fabricated draw or −0.25.
    These diagnostics do not establish significance under censoring.
    """
    y = np.asarray(values, float)[:, :k];st = np.asarray(statuses)[:, :k]
    if y.ndim != 2 or y.shape != st.shape or y.shape[1] != k:raise ValueError('paired bank shape')
    known = st == 'terminal'
    if not np.isin(st, ['terminal', 'simulator_cap', 'decision_cap']).all():raise ValueError('outcome status')
    if not np.isin(y[known], [-1.,0.,1.]).all() or not np.isnan(y[~known]).all():raise ValueError('terminal/censor provenance')
    out = []
    for a in range(len(y)):
        for b in range(a):
            both = known[a] & known[b];d = (y[a,both]-y[b,both])/2
            low_a = np.where(known[a], y[a], -1);hi_a = np.where(known[a], y[a], 1)
            low_b = np.where(known[b], y[b], -1);hi_b = np.where(known[b], y[b], 1)
            corr = None
            if both.sum()>1 and np.std(y[a,both])>0 and np.std(y[b,both])>0:
                corr=float(np.corrcoef(y[a,both],y[b,both])[0,1])
            se = float(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else None
            out.append(dict(a=a,b=b,complete_pairs=int(both.sum()),rho=corr,
                            terminal_pair_gap=float(d.mean()) if len(d) else None,terminal_pair_se=se,
                            gap_identification_bounds=[float(np.mean(low_a-hi_b)/2),float(np.mean(hi_a-low_b)/2)],
                            separable=(bool(abs(d.mean())>2*se) if len(d)==k and se is not None else None)))
    return out


def runtime_smoke(envs, champion, peer, ready):
    """Qualify restore and RNG semantics on both real continuation strata."""
    results=[];seen=set()
    for index,root,snapshot in ready:
        if root['kind'] in seen:continue
        seen.add(root['kind']);models=dict(ours=champion,opp=peer if root['kind']=='foulplay' else champion)
        at,ct=time.perf_counter(),process_cpu()
        reference=None
        for env in envs:
            env.reset(seed=root['seed']);env.restore(snapshot)
            if env._latest_requests != snapshot.latest_requests:raise ValueError('restored request differs')
            current={p:env.observe(p) for p in env.requested_players()}
            if observation_sha(current[root['ours']])!=root['original_observation_sha256']:
                raise ValueError('restore differs from original pre-snapshot observation')
            if reference is None:reference=current
            for p,obs in current.items():
                for name in ('categorical_ids','numeric_features','token_type_ids','attention_mask','legal_action_mask'):
                    if not np.array_equal(getattr(obs,name),getattr(reference[p],name)):
                        raise ValueError('restored observation differs: '+name)
        states=[]
        for k in (0,0,1):
            envs[0].restore(snapshot)
            seed=int(np.random.default_rng([root['root_seed'],k]).integers(1,2**31-1))
            envs[0].reseed_simulator_rng(seed)
            states.append(json.dumps(envs[0].snapshot().bridge_snapshot['battle'],sort_keys=True))
        if states[0]!=states[1] or states[0]==states[2]:raise ValueError('suffix stream repeat/distinctness failed')
        a,ad,ast=lab.run_root(envs,models,root,snapshot,4,False)
        b,bd,bst=lab.run_root(envs,models,root,snapshot,4,False)
        c,cd,cst=lab.run_root(envs,models,root,snapshot,4,True)
        if not (np.array_equal(a,b,equal_nan=True) and np.array_equal(a,c,equal_nan=True)
                and np.array_equal(ad,bd) and np.array_equal(ad,cd) and ast==bst==cst):
            raise ValueError('fixed-stream repeat or batched prefix differs')
        results.append(dict(index=index,kind=root['kind'],status='PASS',K=4,
                            wall_seconds=time.perf_counter()-at,cpu_seconds=process_cpu()-ct))
    if seen!={'self','foulplay'}:raise ValueError('both continuation strata required for runtime smoke')
    return results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    args=p.parse_args(argv);config=json.loads(Path(args.config).read_bytes());validate(config)
    base=config['base_config']
    runtime=subprocess.check_output(['git','-C',base['showdown_root'],'rev-parse','HEAD'],text=True).strip()
    node=subprocess.check_output([base['node_binary'],'--version'],text=True).strip()
    if runtime!=config['showdown_git_sha'] or node!=config['node_version']:raise ValueError('runtime identity differs')
    out=Path(args.out);out.mkdir(exist_ok=False,parents=True)
    write(out/'config.json', config)
    write(out/'manifest.json',dict(schema='mc-m0-manifest.v1', config_sha256=sha(out/'config.json'),
          source_sha256={name:sha(Path(__file__).parent/name) for name in ('mc_m0.py','mc_root_labeller.py','fp_reply_data.py')},
          nesting='batched K64 is prefix of batched K256; paired unbatched K64 control',roots=50))
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
    envs=[];start=time.perf_counter();cpu0=process_cpu();completed=0
    try:
        for _ in range(config['envs']):envs.append(lab.make_env(base,champion.result,peer.result))
        ready,audit=prepare(config,champion,peer,envs[0]);write(out/'prepared.json',audit)
        # The complete roster and failures are durable before any outcome draw.
        print(json.dumps(dict(event='PREPARED',ready=len(ready),expected=50)),flush=True)
        write(out/'runtime-smoke.json',runtime_smoke(envs,champion,peer,ready))
        print(json.dumps(dict(event='RUNTIME_SMOKE_PASS')),flush=True)
        for index,root,snapshot in ready:
            for env in envs:env.reset(seed=root['seed'])
            model=peer if root['kind']=='foulplay' else champion
            for batched,k in ((False,64),(True,256)):
                at,ct=time.perf_counter(),process_cpu()
                try:
                    y,decisions,statuses=lab.run_root(envs,dict(ours=champion,opp=model),root,snapshot,k,batched)
                    row=dict(index=index,root=root,mode='batched' if batched else 'unbatched',K=k,
                             outcomes=[[None if np.isnan(v) else float(v) for v in a] for a in y],statuses=statuses,
                             decisions=decisions.tolist(),pairs={str(n):pair_stats(y,statuses,n) for n in (64,256) if n<=k},
                             wall_seconds=time.perf_counter()-at,cpu_seconds=process_cpu()-ct)
                    write(out/f'root-{index:03d}-{k}.json',row)
                except Exception as error:
                    write(out/f'failure-{index:03d}-{k}.json',dict(reason=type(error).__name__+': '+str(error),
                          wall_seconds=time.perf_counter()-at,cpu_seconds=process_cpu()-ct))
                    raise
            completed+=1
            print(json.dumps(dict(event='ROOT_COMPLETE',index=index,completed=completed)),flush=True)
        status='COMPLETE' if completed==50 else 'SHORTFALL'
        write(out/'summary.json',dict(status=status,completed=completed,expected=50,refused=50-len(ready),
              wall_seconds=time.perf_counter()-start,cpu_seconds=process_cpu()-cpu0))
        return 0 if completed==50 else 2
    except Exception as error:
        write(out/'failure-summary.json',dict(status='FAILED',completed=completed,expected=50,
              reason=type(error).__name__+': '+str(error),wall_seconds=time.perf_counter()-start,cpu_seconds=process_cpu()-cpu0))
        raise
    finally:
        for env in envs:env.close()


if __name__=='__main__':raise SystemExit(main())
