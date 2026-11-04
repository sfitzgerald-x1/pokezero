"""Checkpoint evaluation on shared frozen truth; no new trajectories/training.

Run under the evaluated head's OWN source environment. The observations and
outcomes are immutable bank data. No evaluated head can influence truth policy,
keys, subset, terminal labels or the full head-only reply-grid denominator.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from pokezero import rollout_truth as rules
from .device_rollout_truth import read,publish,label_vocabulary_sha256


def load_predictor(head,bank):
    import jax
    import jax.numpy as j
    from . import champion,v5_model
    if head['kind']=='champion23':
        params,c,meta=champion.load(head['archive']['path'])
        if (meta['weights_archive_sha256']!=head['archive']['sha256']or meta['checkpoint_sha256']!=head['checkpoint_sha256']or c.tokens!=23 or c.vocabulary!=bank['category_count']
                or head['category_vocabulary_sha256']!=bank['category_vocabulary_sha256']):
            raise ValueError('same frozen head archive/checkpoint required')
        def forward(docs):
            out=champion.forward(params,j.asarray([d['numeric_features'][:23]for d in docs],j.float32),j.asarray([d['categorical_ids'][:23]for d in docs],j.int32),
                j.asarray([d['token_type_ids'][:23]for d in docs],j.int32),j.asarray([d['attention_mask'][:23]for d in docs],bool),c,compute_dtype='bfloat16')
            return out.value,j.all(j.isfinite(out.policy_logits),axis=-1)&j.isfinite(out.value)
    elif head['kind']=='public-v5':
        params,c,meta=v5_model.load(head['archive']['path'],expected_archive_sha256=head['archive']['sha256'],expected_checkpoint_sha256=head['checkpoint_sha256'],
            expected_category_vocabulary_sha256=bank['category_vocabulary_sha256'],expected_label_vocabulary_sha256=label_vocabulary_sha256(bank['label_vocabulary']),enable_v5=True)
        def forward(docs):
            inputs={k:j.asarray([d[k]for d in docs],dtype)[:,None]for k,dtype in [('numeric_features',j.float32),('categorical_ids',j.int32),('token_type_ids',j.int32),('attention_mask',bool)]}
            inputs['history_mask']=j.ones((len(docs),1),bool)
            out=v5_model.forward(params,inputs,c,compute_dtype='bfloat16')
            return out.value,out.valid&j.isfinite(out.value)&j.all(j.isfinite(out.policy_logits),axis=-1)
    else:raise ValueError('explicit public-v5/champion23 head required; privileged fitting is a separate partitioned step')
    # Batch shape does not select or reorder data; outputs are accepted only as
    # finite bounded predictions. This head never supplies rollout labels.
    def predict(docs):
        result=[]
        for start in range(0,len(docs),256):
            values,valid=jax.block_until_ready(forward(docs[start:start+256]))
            result.extend(float(v)if bool(ok)and np.isfinite(v)and abs(v)<=1 else None for v,ok in zip(np.asarray(values),np.asarray(valid),strict=True))
        return result
    return predict


def backup_from_grid(grid,predictions):
    """All required reply mass, no survivor renormalization; max legal own Q."""
    if grid['status']!='OBSERVED':return None
    if len(predictions)!=len(grid['rows']):raise ValueError('every declared grid branch prediction required')
    if [(r['action'],r['reply'])for r in grid['rows']]!=[(a,b)for a in grid['own_actions']for b in grid['replies']]:raise ValueError('every legal own/reply coordinate required once')
    if len(grid['replies'])!=len(grid['reply_weights'])or len(set(grid['own_actions']))!=len(grid['own_actions'])or len(set(grid['replies']))!=len(grid['replies']):raise ValueError('unique full-grid declarations required')
    prior=dict(zip(grid['replies'],grid['reply_weights'],strict=True))
    actions={}
    for row,v in zip(grid['rows'],predictions,strict=True):
        if row['status']=='COMPLETE':v=row['terminal_value']
        elif row['status']!='OBSERVED':return None
        if v is None or not np.isfinite(v)or abs(v)>1:return None
        if row['weight']!=prior[row['reply']]:raise ValueError('same frozen reply policy for every action required')
        actions.setdefault(row['action'],[]).append((row['reply'],row['weight'],v))
    expected=None;values=[]
    for rows in actions.values():
        ids=tuple(x[0]for x in rows)
        if len(set(ids))!=len(ids)or (expected is not None and ids!=expected):raise ValueError('same complete reply grid required')
        expected=ids
        weights=[r[1]for r in rows]
        if any(not np.isfinite(w)or w<0 for w in weights)or not np.isclose(sum(weights),1.,atol=1e-12,rtol=0):raise ValueError('full frozen reply prior required')
        values.append(sum(w*v for _,w,v in rows))
    return max(values)if values else None


def evaluate(config,summary_reference,head,output):
    """Only genuine stored observations/refusals. All bulk remains at caller."""
    from pokezero import v5_device_evaluation_bank as schema
    from .device_rollout_truth import bank_manifest
    _,_,_,_,positions,groups,_,manifest=bank_manifest(config,schema)
    summary=read(summary_reference)
    if any(summary['manifest'][k]!=manifest[k]for k in ('declaration','positions','siblings','declared_children','source_commit')):
        raise ValueError('truth and head bank differ')
    def lines(ref):
        blob=Path(ref['path']).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=ref['sha256']:raise ValueError('immutable truth/grid bytes differ')
        return [json.loads(x)for x in blob.splitlines()]
    records=lines(summary['truth']);header=records.pop(0);grids=lines(summary['head_grids'])
    if header!=summary['manifest']or len(records)!=manifest['declared_children'] or len(grids)!=manifest['declared_slots']:raise ValueError('complete original truth denominator required')
    if summary['manifest']['prespec']!=config['prespec']or summary['manifest']['recipe']!=vars(rules.Recipe(config['seed'],max_requests=config['max_requests']))or summary['manifest']['raw_actor']!=config['raw_actor']:
        raise ValueError('same shared truth policy/keys/prespec required')
    expected=[(g['selector'],b['action'])for g in groups for b in g['branches']]
    if [(r['selector'],r['action'])for r in records]!=expected:raise ValueError('truth selector/action order changed')
    if [g['selector']for g in grids]!=[p['selector']for p in positions]:raise ValueError('grid root order changed')
    for record,group in zip(records,[g for g in groups for _ in g['branches']],strict=True):
        if any(record[k]!=group[k]for k in ('reply','chance_id','root_id','world_id')):raise ValueError('same recorded child world/reply/chance required')
    bank=schema.read_artifact(config['declaration']);predict=load_predictor(head,bank)
    root_docs=[p['observation']for p in positions if p['observation']is not None];rp=iter(predict(root_docs));roots=[];grid_predictions=[]
    for p,g in zip(positions,grids,strict=True):
        outcome=None
        if p['terminal']is not None and p['terminal']['status']=='COMPLETE':
            winner=p['terminal']['winner'];outcome=0 if winner==2 else 1 if winner==p['selector']['owner']else -1
        root=dict(selector=p['selector'],prediction=None if p['observation']is None else next(rp),outcome=outcome,backup=None)
        values=None
        if g['status']=='OBSERVED':
            blob=Path(g['artifact']['path']).read_bytes()
            if hashlib.sha256(blob).hexdigest()!=g['artifact']['sha256']:raise ValueError('grid tensor bytes differ')
            import io
            with np.load(io.BytesIO(blob),allow_pickle=False)as a:
                keys=('numeric_features','categorical_ids','token_type_ids','attention_mask','legal_action_mask')
                if set(a.files)!=set(keys)or any(len(a[k])!=len(g['rows'])for k in keys):raise ValueError('full typed grid tensor fields required')
                docs=[{k:a[k][i].tolist()for k in keys}for i,r in enumerate(g['rows'])if r['status']=='OBSERVED']
                pv=iter(predict(docs));values=[next(pv)if r['status']=='OBSERVED'else r['terminal_value']for r in g['rows']]
            root['backup']=backup_from_grid(g,values)
        roots.append(root)
        grid_predictions.append(dict(selector=g['selector'],status=g['status'],predictions=values))
    docs=[b['observation']for g in groups for b in g['branches']if b['status']=='OBSERVED'];cp=iter(predict(docs));children=[]
    for record,branch in zip(records,[b for g in groups for b in g['branches']],strict=True):
        value=None
        if branch['status']=='OBSERVED':value=next(cp)
        # Original terminal children without a stored head observation have
        # known rollout truth, but no invented head prediction. They remain
        # unknown in head calibration (terminal values still serve backup).
        children.append(dict(selector=record['selector'],action=record['action'],prediction=value,rollouts=record['rollouts']))
    # Preserve the scalar outputs already used above. A stored-result reviewer
    # can join these to the immutable truth/grid records without another model
    # forward. Missing predictions and the original ordering remain explicit.
    predictions=publish(str(output)+'.predictions.json',dict(schema='pokezero.rollout-truth.head-predictions.v1',
        head=head,truth_summary=summary_reference,bank=manifest,roots=roots,grids=grid_predictions,
        children=[{k:row[k]for k in ('selector','action','prediction')}for row in children]))
    result=dict(schema='pokezero.rollout-truth.head-score.v1',head=head,truth_summary=summary_reference,bank=manifest,predictions=predictions,
        metrics=rules.metrics(roots,children),all_checkpoints_share_truth=True,new_games=0,
        privileged_fit='NOT_RUN; manifest game-clustered5fold crossfit is fixed before outcomes; actual B checkpoint/frozen trunk/public-head warmstart; small steps/epoch0 eligible; train fits, validation selects, every test prediction OUT OF FOLD; no claim for a final all-bank fitted critic')
    publish(output,result);return result
