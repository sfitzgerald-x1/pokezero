"""Prepare offline-only genuine privileged rows, not models/games/training.

Execute this FILE with the original bank source. Uses accepted reconstructed
roots and ONE explicit original-chance child/grid transition, never an adjacent
or inferred latent state. Original public tensors must match before swapping.
No actor input, truth policy or original outcome is changed.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _file_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def prepare(config,source,output,*,enable_prepare=False):
    if enable_prepare is not True:raise ValueError('explicit private offline row preparation required')
    base=Path(__file__).resolve().parents[1]
    helper=_file_module('_priv_rows_truth_helper',base/'gpu_engine'/'device_rollout_truth.py')
    critic=_file_module('_priv_rows_rules',base/'src'/'pokezero'/'v5_privileged_critic.py')
    worker,schema=helper._import_original(source)
    decl,batches,games,labels,positions,groups,index,bank_manifest=helper.bank_manifest(config,schema)
    summary=helper.read(config['truth_summary']);truth_manifest=summary['manifest']
    if any(truth_manifest[k]!=bank_manifest[k]for k in ('declaration','positions','siblings','source_commit','declared_children')):raise ValueError('same original truth bank required')
    if truth_manifest['critic_split_seed']!=2103700000:raise ValueError('same predeclared five-fold seed required')
    prepared,pool,gender=helper.prepare_original(worker,decl,config['runtime'])
    from gpu_engine import device_root_snapshot as snapshots,v5_device_evaluation_bank as bank
    import jax
    import numpy as np
    import inspect
    from pokezero import v5_contract as contract
    root_index=helper.read(config['reconstructed_index'])
    if root_index['namespace']!=index['namespace']or [e['selector']for e in root_index['cost_roots']]!=[p['selector']for p in positions]:raise ValueError('same complete verified-root index required')
    if Path(config['reconstructed_index']['path']).resolve()!=Path(summary['truth']['path']).resolve().parent/'reconstructed-root-index.json':raise ValueError('actual truth reconstructed-index path required')
    def lines(ref):
        blob=critic.artifact(ref,json_value=False)
        return [json.loads(x)for x in blob.splitlines()]
    truth_rows=lines(summary['truth']);header=truth_rows.pop(0);grids=lines(summary['head_grids'])
    if header!=truth_manifest or len(truth_rows)!=bank_manifest['declared_children']or len(grids)!=len(positions):raise ValueError('complete truth/grid denominator required')
    expected=[(g['selector'],b['action'])for g in groups for b in g['branches']]
    if [(r['selector'],r['action'])for r in truth_rows]!=expected or [g['selector']for g in grids]!=[p['selector']for p in positions]:raise ValueError('same ordered root/child/grid coordinates required')
    executor=bank.make_executor(prepared,None) # child_view/advance ONLY; query/model NEVER called.
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=[];arrays={k:[]for k in critic.KEYS};grid_records=[];truth_iter=iter(truth_rows)
    def same_bytes(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    def view_docs(state,public):
        views=jax.block_until_ready(executor.child_view(state,public))
        return views
    def private(state,public,owner,views):
        if any(int(v.error[0])!=0 for v in views):raise ValueError('actual both-seat source view refused')
        mapping=critic.physical_slot_map(np.asarray(public.physical_to_public[0,1-owner]),np.asarray(public.own_order[0,1-owner]))
        docs=[{k:np.asarray(getattr(v,k)[0])for k in critic.KEYS}for v in views]
        return critic.privileged_arrays(docs[owner],docs[1-owner],mapping)
    def add(kind,selector,*,data=None,reason=None,**facts):
        identity=dict(kind=kind,selector=selector,**{k:v for k,v in facts.items()if k in ('action','reply')})
        row=dict(row_id=helper.rules().digest(identity),kind=kind,selector=selector,available=data is not None,array_row=None,reason=reason,**facts)
        if data is not None:
            row['array_row']=len(arrays[critic.KEYS[0]])
            for k in critic.KEYS:arrays[k].append(data[k])
        records.append(row);return row
    def original_view_matches(state,public,views,owner,position):
        return helper.accept_root(state,public,views[owner],position['selector'],position,bank,schema)
    for entry,p,g,grid in zip(root_index['cost_roots'],positions,groups,grids,strict=True):
        root=fields=views=None;reason=entry['reason'];data=None
        if entry['status']=='SAVED':
            root,fields=snapshots.load(entry,root_index);views=view_docs(root.state,root.public)
            if not original_view_matches(root.state,root.public,views,p['selector']['owner'],p):raise ValueError('verified original root changed; no adjacent-state substitution')
            try:data=private(root.state,root.public,p['selector']['owner'],views)
            except ValueError as error:reason=str(error)
        terminal=p['terminal'];outcome=None
        if terminal is not None and terminal['status']=='COMPLETE':outcome=0 if terminal['winner']==2 else 1 if terminal['winner']==p['selector']['owner']else -1
        add('root',p['selector'],data=data,reason=reason,target=None,outcome=outcome)
        for branch in g['branches']:
            actual=next(truth_iter)
            if any(actual[k]!=g[k]for k in ('root_id','world_id','reply','chance_id')):raise ValueError('same actual child truth provenance required')
            target=helper.rules().summarize_rollouts(actual['rollouts'])['value'];data=None;why='MISSING_VERIFIED_ROOT'
            if root is not None:
                out=helper.child_step(executor,root,fields,g,branch['action'],g['reply'])
                child_views=view_docs(out.step.state,out.trackers)
                if bool(out.step.invalid_actions[0]):why='ACTUAL_CHILD_REFUSED'
                elif branch['status']=='OBSERVED'and not bool(out.step.state.done[0])and bank._observation(child_views[p['selector']['owner']])==branch['observation']:
                    try:data=private(out.step.state,out.trackers,p['selector']['owner'],child_views);why=None
                    except ValueError as error:why=str(error)
                elif branch['status']=='COMPLETE':
                    actual=dict(done=bool(out.step.state.done[0]),winner=int(out.step.state.winner[0]),state_id=schema.digest(bank._host(out.step.state)))
                    if actual!=branch['terminal']:raise ValueError('same original actual child terminal required')
                    why='ACTUAL_TERMINAL_NO_HEAD_OBSERVATION'
                else:raise ValueError('actual stored child public bytes differ')
            add('child',p['selector'],action=branch['action'],reply=g['reply'],data=data,reason=why,target=target,rollouts=actual['rollouts'])
        gr=dict(grid,rows=[])
        tensors=None
        if grid.get('artifact')is not None:
            import io
            blob=critic.artifact(grid['artifact'],json_value=False)
            with np.load(io.BytesIO(blob),allow_pickle=False)as archive:tensors={k:np.array(archive[k],copy=True)for k in (*critic.KEYS,'legal_action_mask')}
        for n,branch in enumerate(grid.get('rows',[])):
            data=None;why='MISSING_VERIFIED_ROOT'
            if root is not None and branch['status']=='OBSERVED':
                out=helper.child_step(executor,root,fields,g,branch['action'],branch['reply']);child_views=view_docs(out.step.state,out.trackers);view=child_views[p['selector']['owner']]
                if bool(out.step.invalid_actions[0])or bool(out.step.state.done[0])or tensors is None or any(not same_bytes(np.asarray(getattr(view,k)[0]),tensors[k][n])for k in tensors):raise ValueError('same original head-grid physical/public branch required')
                try:data=private(out.step.state,out.trackers,p['selector']['owner'],child_views);why=None
                except ValueError as error:why=str(error)
            elif root is not None and branch['status']=='COMPLETE':
                out=helper.child_step(executor,root,fields,g,branch['action'],branch['reply']);winner=int(out.step.state.winner[0]);owner=p['selector']['owner']
                value=0 if winner==2 else 1 if winner==owner else -1
                if bool(out.step.invalid_actions[0])or not bool(out.step.state.done[0])or winner not in (0,1,2)or value!=branch['terminal_value']:raise ValueError('same actual terminal head-grid leaf required')
                why='ACTUAL_TERMINAL_NO_HEAD_OBSERVATION'
            r=add('grid',p['selector'],action=branch['action'],reply=branch['reply'],data=data,reason=why,target=None)
            gr['rows'].append(dict(branch,row_id=r['row_id']))
        grid_records.append(gr)
    shards=[]
    if arrays[critic.KEYS[0]]:
        path=output/'privileged-rows.npz'
        with path.open('xb')as f:np.savez_compressed(f,**{k:np.stack(v)for k,v in arrays.items()})
        shards=[dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())]
    manifest=dict(schema='pokezero.privileged-critic.rows.v1',view='PRIVILEGED_OFFLINE_ONLY',truth_summary=config['truth_summary'],bank=bank_manifest,
        reconstructed_index=config['reconstructed_index'],split_seed=2103700000,folds=helper.rules().folds([p['selector']for p in positions],seed=2103700000),
        category_vocabulary_sha256=decl['category_vocabulary_sha256'],label_vocabulary_sha256=labels.sha256,records=records,shards=shards,grids=grid_records,
        original_privileged_contract_sha256=hashlib.sha256(inspect.getsource(contract.privileged_view).encode()).hexdigest(),
        field_binding=dict(source_commit=decl['source_commit'],opponent_physical_to_public='Trackers.physical_to_public[0,1-owner]',peer_order='Trackers.own_order[0,1-owner]',
            public_swap='Opponent tokens7:13 <- actual peer-own tokens1:7 by explicit complete physical permutation; known encounter slots preserved; unseen physical members fill unused slots ascending OFFLINE ONLY'),
        targets='Only child64-complete rollout means fit; roots/original outcomes and head-only grids are evaluation-only. Missing tensors/targets kept.',new_games=0)
    return helper.publish(output/'rows.json',manifest)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--expected-config-sha256',required=True)
    parser.add_argument('--original-source',required=True);parser.add_argument('--output',required=True);parser.add_argument('--enable-prepare',action='store_true');a=parser.parse_args(argv)
    if not a.enable_prepare:raise ValueError('explicit offline row preparation required')
    blob=Path(a.config).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=a.expected_config_sha256:raise ValueError('declared private preparation config differs')
    prepare(json.loads(blob),a.original_source,a.output,enable_prepare=True)

if __name__=='__main__':main()
