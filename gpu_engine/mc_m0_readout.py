"""Compact readout of the frozen M0; pending roots remain in the denominator."""
import argparse
import collections
import json
from pathlib import Path
import statistics

from gpu_engine.mc_m0 import sha, write, pair_stats


def readout(directory):
    root=Path(directory);config=json.loads((root/'config.json').read_bytes())
    manifest=json.loads((root/'manifest.json').read_bytes())
    if sha(root/'config.json')!=manifest['config_sha256']:raise ValueError('changed configuration')
    prepared_path=root/'prepared.json'
    prepared=json.loads(prepared_path.read_bytes()) if prepared_path.exists() else []
    ready={r['index']:r['root'] for r in prepared if r['status']=='READY'}
    if len({r['index'] for r in prepared})!=len(prepared):raise ValueError('duplicate prepared index')
    for i,r in ready.items():
        if not 0<=i<50 or any(r.get(k)!=v for k,v in config['roots'][i].items()):raise ValueError('prepared roster differs')
    stages=collections.defaultdict(list);done=collections.defaultdict(set)
    for path in sorted(root.glob('root-*.json')):
        row=json.loads(path.read_bytes());i=row['index'];k=row['K']
        if i not in ready or row['root']!=ready[i]:raise ValueError('unbound root result')
        if k not in (64,256) or row['mode']!=('unbatched' if k==64 else 'batched'):raise ValueError('stage differs')
        if i in done[k]:raise ValueError('duplicate stage result')
        for n in (64,256):
            if n<=k and row['pairs'][str(n)]!=pair_stats(row['outcomes'],row['statuses'],n):
                raise ValueError('stored pair diagnostics differ from raw outcomes')
        done[k].add(i);stages[(row['root']['kind'],k)].append(row)
    summaries=[]
    for (kind,k),rows in sorted(stages.items()):
        rollouts=sum(len(r['root']['actions'])*k for r in rows)
        counts=collections.Counter(s for r in rows for action in r['statuses'] for s in action)
        if sum(counts.values())!=rollouts:raise ValueError('outcome denominator differs')
        entry=dict(kind=kind,mode=rows[0]['mode'],K=k,roots=len(rows),rollouts=rollouts,statuses=dict(counts),
                   cpu_seconds=sum(r['cpu_seconds'] for r in rows),wall_seconds=sum(r['wall_seconds'] for r in rows))
        entry.update(cpu_seconds_per_rollout=entry['cpu_seconds']/rollouts,
                     wall_seconds_per_root=entry['wall_seconds']/len(rows))
        if k==256:
            diagnostics={}
            for n in ('64','256'):
                correlations=[];separable=0;resolved=0;pair_ses=[]
                for row in rows:
                    pairs=row['pairs'][n]
                    rho=[p['rho'] for p in pairs if p['rho'] is not None]
                    if rho:correlations.append(statistics.mean(rho))
                    all_known=all(p['separable'] is not None for p in pairs)
                    resolved+=all_known;separable+=any(p['separable'] is True for p in pairs)
                    pair_ses.extend(p['terminal_pair_se'] for p in pairs if p['terminal_pair_se'] is not None)
                diagnostics[n]=dict(roots_with_separable_pair=separable,measured_roots=len(rows),
                    uncensored_roots=resolved,median_root_mean_rho=statistics.median(correlations) if correlations else None,
                    roots_with_nondegenerate_rho=len(correlations),median_complete_pair_se=statistics.median(pair_ses) if pair_ses else None,
                    se_caveat='Complete-pair diagnostic only; censored pairs are not declared separable.')
            entry['diagnostics']=diagnostics
        summaries.append(entry)
    smoke_path=root/'runtime-smoke.json';smoke=json.loads(smoke_path.read_bytes()) if smoke_path.exists() else []
    qualified=len(smoke)==2 and {r['kind'] for r in smoke}=={'self','foulplay'} and all(r['status']=='PASS' for r in smoke)
    final_path=root/'summary.json';final=json.loads(final_path.read_bytes()) if final_path.exists() else None
    failure_path=root/'failure-summary.json';failure=json.loads(failure_path.read_bytes()) if failure_path.exists() else None
    complete=bool(not failure_path.exists() and qualified and final and final['status']=='COMPLETE' and len(done[64]&done[256])==50 and len(ready)==50)
    return dict(status='COMPLETE' if complete else 'FAILED' if failure_path.exists() or final else 'RUNNING_OR_UNOBSERVED',
                experiment_completion_verified=complete,expected_roots=50,prepared=len(prepared),ready=len(ready),
                refused=len(prepared)-len(ready),completed_both_modes=len(done[64]&done[256]),
                unbatched_completed=len(done[64]),batched_completed=len(done[256]),runtime_qualified=qualified,
                final=final,failure=failure,groups=summaries,
                scope='Cost and CRN measurement under pinned champion/proxy policies; no strength or real-Foul-Play gain claim.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--out')
    args=p.parse_args(argv);report=readout(args.directory)
    if args.out:write(args.out,report)
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0 if report['experiment_completion_verified'] else 2


if __name__=='__main__':raise SystemExit(main())
