"""Offline-only B-trunk privileged critic; never an actor or game collector.

The original B checkpoint loader stays strict. Original public-head weights are
warmstart/epoch0, trunk is frozen/eval, only value_head/value_query are fitted.
Every reported prediction is game-clustered OUT OF FOLD. No work on import.
"""
from dataclasses import dataclass,asdict
from contextlib import nullcontext
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
def truth():
    # Also usable as a source-only array adapter in the quarantined ORIGINAL
    # bank namespace; do not install new modules over its original package.
    import importlib.util,sys
    name='_privileged_critic_truth_rules'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('rollout_truth.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]

KEYS=('numeric_features','categorical_ids','token_type_ids','attention_mask')
SHAPES={'numeric_features':(29,132),'categorical_ids':(29,41),'token_type_ids':(29,),'attention_mask':(29,)}
DTYPES={'numeric_features':np.dtype('float32'),'categorical_ids':np.dtype('int32'),'token_type_ids':np.dtype('int32'),'attention_mask':np.dtype('bool')}


def artifact(ref,*,json_value=True):
    blob=Path(ref['path']).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=ref['sha256']:raise ValueError('immutable data/checkpoint bytes differ')
    return json.loads(blob)if json_value else blob


def physical_slot_map(physical_to_public,peer_order):
    """Known public encounter slots first; unseen slots from actual physical IDs.

    Completion is explicitly PRIVILEGED ONLY. It neither claims unseen public
    identity nor changes the actor/public rows. Source physical IDs are0..5.
    """
    p=np.asarray(physical_to_public);order=np.asarray(peer_order)
    if p.shape!=(6,)or order.shape!=(6,)or p.dtype!=np.int32 or order.dtype!=np.int32 or set(order.tolist())!=set(range(6)):
        raise ValueError('actual original six-member int32 physical namespace required')
    if np.any((p < -1)|(p > 5)):raise ValueError('public encounter slot outside original namespace')
    known=[int(x)for x in p if x>=0]
    if len(known)!=len(set(known)):raise ValueError('duplicate public encounter physical join')
    slots=[None]*6
    for physical,slot in enumerate(p):
        if slot>=0:slots[int(slot)]=physical
    remaining=iter(i for i in range(6)if p[i]<0)
    slots=[next(remaining)if x is None else x for x in slots]
    return tuple(order.tolist().index(physical)for physical in slots)


def privileged_arrays(public,peer,slot_map):
    """Exact v5_contract.privileged_view swap with device tensors kept float32.

    Only cats/numeric/attention opponent tokens7:13 change. Public belief tails,
    owner/action/field/history and role IDs remain byte-identical. Actor surface
    is deliberately not called or returned; these are offline sidecar arrays.
    """
    if tuple(sorted(slot_map))!=tuple(range(6)):raise ValueError('complete peer physical permutation required')
    for doc in (public,peer):
        if set(doc)!=set(KEYS):raise ValueError('exact typed source observation fields required')
        for k in KEYS:
            a=np.asarray(doc[k])
            if a.shape!=SHAPES[k]or a.dtype!=DTYPES[k]:raise ValueError('same actual source tensor type/shape required')
        if not np.isfinite(doc['numeric_features']).all():raise ValueError('nonfinite source view')
    out={k:np.array(public[k],copy=True)for k in KEYS}
    for k in ('numeric_features','categorical_ids','attention_mask'):out[k][7:13]=np.asarray(peer[k])[1+np.asarray(slot_map)]
    return out


@dataclass(frozen=True)
class FitConfig:
    seed:int=2103700000
    epochs:int=10
    learning_rate:float=1e-5
    batch_size:int=256
    device:str='cuda'
    precision:str='bfloat16'
    def __post_init__(self):
        if type(self.seed)is not int or self.seed!=2103700000:raise ValueError('original predeclared five-fold seed required')
        if type(self.epochs)is not int or not 1<=self.epochs<=20 or type(self.batch_size)is not int or self.batch_size<=0:
            raise ValueError('bounded predeclared head-only fit recipe required')
        if type(self.learning_rate)not in (float,int)or not math.isfinite(self.learning_rate)or not 0<self.learning_rate<=1e-4:
            raise ValueError('small finite privileged-head learning rate required')
        if type(self.device)is not str or not self.device or self.precision not in ('bfloat16','float32'):raise ValueError('explicit fit device/precision required')


def fold_indices(records,*,fold,seed=2103700000):
    roles=truth().partition([r['selector']for r in records],seed=seed,holdout_fold=fold)
    result={k:[]for k in ('train','validation','test')}
    for i,r in enumerate(records):
        role=roles[r['selector']['stream']]
        if role=='test':result[role].append(i) # ALL test rows, including missing.
        elif r['kind']=='child'and r['available']and r.get('target')is not None:result[role].append(i)
    return result


def select_epoch(history):
    """Validation only, epoch0 included; earliest epoch wins exact ties."""
    if not history or history[0]['epoch']!=0 or [r['epoch']for r in history]!=list(range(len(history))):raise ValueError('warmstart epoch0 and ordered epochs required')
    valid=[r for r in history if r['validation_mse']is not None and math.isfinite(r['validation_mse'])]
    return None if not valid else min(valid,key=lambda r:(r['validation_mse'],r['epoch']))['epoch']


def fit_fold(train,validation,epochs,*,train_epoch,validate,snapshot,restore):
    """Controller never receives test rows/labels; warmstart is epoch0."""
    history=[dict(epoch=0,validation_mse=validate(validation))];best=snapshot();chosen=0;updates=0
    if train and history[0]['validation_mse']is not None:
        for epoch in range(1,epochs+1):
            updates+=train_epoch(train,epoch)
            history.append(dict(epoch=epoch,validation_mse=validate(validation)))
            if select_epoch(history)==epoch:chosen=epoch;best=snapshot()
    restore(best)
    return dict(selected_epoch=chosen,history=history,optimizer_updates=updates)


def production_binding(config):
    """Original production declaration + durable checkpoint publication, not smoke."""
    from . import v5_run as runs
    declaration=artifact(config['production_declaration']);start=artifact(config['run_started_event']);event=artifact(config['checkpoint_event'])
    for e in (start,event):
        body=dict(e);sha=body.pop('sha256')
        if truth().digest(body)!=sha:raise ValueError('original checkpoint journal event integrity differs')
    if (declaration['schema']!=runs.SCHEMA or declaration['model']['arm']!='B'
            or sum(len(b['games'])for it in declaration['iterations']for b in it)!=2_000_000
            or start['kind']!='RUN_STARTED'or start['sequence']!=0 or start['previous']is not None
            or start['payload']['declaration_sha256']!=truth().digest(declaration)
            or event['kind']!='CHECKPOINT'or event['payload']['steps']<=0 or event['payload']['rows']<=0
            or event['payload']['sha256']!=config['checkpoint']['sha256']):
        raise ValueError('actual trained production B checkpoint required, no diagnostic/scratch trunk')
    journal=Path(config['run_started_event']['path']).resolve().parent
    if Path(config['checkpoint_event']['path']).resolve().parent!=journal or Path(config['production_declaration']['path']).resolve().parent!=journal.parent:
        raise ValueError('same original production run publication paths required')
    if (journal.parent/event['payload']['path']).resolve()!=Path(config['checkpoint']['path']).resolve():raise ValueError('checkpoint path differs from original publication')
    artifact(config['checkpoint'],json_value=False)
    return declaration,event['payload']


def load_rows(reference):
    manifest=artifact(reference);records=manifest['records'];directory=Path(reference['path']).resolve().parent
    arrays={};expected=0
    for shard in manifest['shards']:
        blob=artifact(shard,json_value=False)
        import io
        with np.load(io.BytesIO(blob),allow_pickle=False)as z:
            if set(z.files)!=set(KEYS):raise ValueError('exact private sidecar tensors required')
            n=len(z[KEYS[0]])
            for k in KEYS:
                if z[k].shape!=(n,)+SHAPES[k]or z[k].dtype!=DTYPES[k]:raise ValueError('typed private shard differs')
                arrays.setdefault(k,[]).append(np.array(z[k],copy=True))
            expected+=n
    arrays={k:np.concatenate(v,axis=0)for k,v in arrays.items()}if arrays else{}
    seen=[];identities=set()
    for r in records:
        identity=dict(kind=r['kind'],selector=r['selector'],**{k:r[k]for k in ('action','reply')if k in r})
        if r['kind']not in ('root','child','grid')or r['row_id']!=truth().digest(identity)or r['row_id']in identities:raise ValueError('same unique original private row identity required')
        identities.add(r['row_id'])
        if r['kind']=='child':
            if r['target']!=truth().summarize_rollouts(r['rollouts'])['value']:raise ValueError('critic target differs from actual64 rollout truth')
        elif r.get('target')is not None:raise ValueError('root/grid cannot add critic fitting labels')
        if r['available']:
            if type(r['array_row'])is not int:raise ValueError('actual private tensor index required')
            seen.append(r['array_row'])
        elif r['array_row']is not None:raise ValueError('missing private row cannot carry invented tensors')
    if sorted(seen)!=list(range(expected)):raise ValueError('every private tensor must name exactly one original position')
    if expected and (not np.isfinite(arrays['numeric_features']).all()or (arrays['categorical_ids']<0).any()):raise ValueError('finite genuine privileged rows required')
    return manifest,records,arrays


def head_parameter(name):return name=='value_query'or name.startswith('value_head.')


def _parameter_bytes(model,*,trunk):
    return {name:p.detach().cpu().contiguous().numpy().tobytes()for name,p in model.state_dict().items()
        if (not head_parameter(name))==trunk}


def crossfit(config,rows_reference,output,*,enable_fit=False):
    """Actual scientific fitter. Explicit opt-in before any Torch/model access."""
    if enable_fit is not True:raise ValueError('explicit offline privileged critic fitting required')
    fit=FitConfig(**config['fit']);decl,event=production_binding(config)
    manifest,records,arrays=load_rows(rows_reference)
    if manifest['view']!='PRIVILEGED_OFFLINE_ONLY'or manifest['split_seed']!=fit.seed or manifest['truth_summary']!=config['truth_summary']:raise ValueError('predeclared private sidecar/folds required')
    from . import v5_training as training
    from .v5_model import MODEL_INPUTS
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    # Digest and loader consume the SAME private immutable byte snapshot, so
    # a changing source path cannot race the original strict checkpoint load.
    snapshot=output/'production-input.pt'
    with snapshot.open('xb')as f:f.write(artifact(config['checkpoint'],json_value=False))
    snapshot.chmod(0o444)
    runtime=training.load_checkpoint(snapshot,enable_v5=True,
        **({'expected_learning_rate_window':training.DeclaredGameWindow(**config['learning_rate_window'])}if config.get('learning_rate_window')is not None else{}))
    if (runtime.model_config.arm!='B'or runtime.steps!=event['steps']or runtime.rows!=event['rows']
            or truth().digest(runtime.model_config.metadata())!=truth().digest(decl['model'])
            or runtime.model_config.category_vocabulary_sha256!=manifest['category_vocabulary_sha256']
            or runtime.model_config.label_vocabulary.sha256!=manifest['label_vocabulary_sha256']):raise ValueError('same actual B model/source/namespace required')
    torch=training._torch();model=runtime.model.to(fit.device);model.eval()
    for name,p in model.named_parameters():p.requires_grad_(head_parameter(name))
    trainable=[p for p in model.parameters()if p.requires_grad]
    if not trainable:raise ValueError('original B value head required')
    initial=copy.deepcopy(model.state_dict());trunk=_parameter_bytes(model,trunk=True)
    def precision():return torch.autocast(device_type=torch.device(fit.device).type,dtype=torch.bfloat16)if fit.precision=='bfloat16'else nullcontext()
    predictions=[None]*len(records);fold_reports=[];predicted=set()
    def evaluate(indices):
        result=[]
        for start in range(0,len(indices),fit.batch_size):
            selected=indices[start:start+fit.batch_size];available=[i for i in selected if records[i]['available']]
            values={}
            if available:
                tensors={k:torch.as_tensor(arrays[k][[records[i]['array_row']for i in available]],dtype=torch.float32 if k=='numeric_features'else torch.bool if k=='attention_mask'else torch.long,device=fit.device)[:,None]for k in KEYS}
                tensors['history_mask']=torch.ones((len(available),1),dtype=torch.bool,device=fit.device)
                with torch.no_grad(),precision():v=model(**tensors,include_auxiliary=False).value.float().detach().cpu().numpy()
                values={i:float(x)if np.isfinite(x)and abs(x)<=1 else None for i,x in zip(available,v,strict=True)}
            result.extend(values.get(i)for i in selected)
        return result
    def mse(indices):
        values=evaluate(indices)
        if not indices or any(v is None for v in values):return None
        return float(np.mean([(v-records[i]['target'])**2 for i,v in zip(indices,values,strict=True)]))
    for fold in range(5):
        model.load_state_dict(initial,strict=True);model.eval();indices=fold_indices(records,fold=fold,seed=fit.seed)
        optimizer=torch.optim.AdamW(trainable,lr=fit.learning_rate,weight_decay=0.)
        def train_epoch(train,epoch):
            updates=0
            order=np.random.default_rng(fit.seed+100*fold+epoch).permutation(train).tolist()
            for start in range(0,len(order),fit.batch_size):
                ids=order[start:start+fit.batch_size]
                tensors={k:torch.as_tensor(arrays[k][[records[i]['array_row']for i in ids]],dtype=torch.float32 if k=='numeric_features'else torch.bool if k=='attention_mask'else torch.long,device=fit.device)[:,None]for k in KEYS}
                tensors['history_mask']=torch.ones((len(ids),1),dtype=torch.bool,device=fit.device)
                target=torch.as_tensor([records[i]['target']for i in ids],dtype=torch.float32,device=fit.device)
                optimizer.zero_grad(set_to_none=True)
                with precision():value=model(**tensors,include_auxiliary=False).value
                loss=(value.float()-target).square().mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite critic loss; no test-based recovery')
                loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.);optimizer.step();updates+=1
            return updates
        fitted=fit_fold(indices['train'],indices['validation'],fit.epochs,train_epoch=train_epoch,validate=mse,
            snapshot=lambda:copy.deepcopy(model.state_dict()),restore=lambda state:model.load_state_dict(state,strict=True))
        history=fitted['history'];chosen=fitted['selected_epoch'];updates=fitted['optimizer_updates']
        if _parameter_bytes(model,trunk=True)!=trunk:raise ValueError('frozen B trunk mutated')
        values=evaluate(indices['test'])
        for i,v in zip(indices['test'],values,strict=True):
            if i in predicted:raise ValueError('test prediction reused across folds')
            predicted.add(i);predictions[i]=v
        path=output/('fold-%d-head.pt'%fold)
        with path.open('xb')as f:torch.save({k:v.detach().cpu()for k,v in model.state_dict().items()if head_parameter(k)},f)
        fold_reports.append(dict(fold=fold,optimizer_updates=updates,train_rows=len(indices['train']),validation_rows=len(indices['validation']),test_rows=len(indices['test']),selected_epoch=chosen,
            history=history,fit_status='FIT'if indices['train']and history[0]['validation_mse']is not None else'WARMSTART_ONLY_MISSING_TRAIN_OR_VALIDATION',
            head=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    if predicted!=set(range(len(records))):raise ValueError('every declared row needs one OOS disposition')
    metrics=score_predictions(manifest,records,predictions)
    assignments=truth().folds([r['selector']for r in records],seed=fit.seed)
    result=dict(metrics=metrics,fit_complete=all(f['optimizer_updates']>0 for f in fold_reports),schema='pokezero.privileged-critic.crossfit.v1',rows=rows_reference,production_checkpoint=config['checkpoint'],production_declaration=config['production_declaration'],fit=asdict(fit),
        folds=fold_reports,predictions=[dict(row_id=r['row_id'],selector=r['selector'],kind=r['kind'],prediction=p,fold=assignments[r['selector']['stream']])for r,p in zip(records,predictions,strict=True)],
        implementation='Original Torch B builder/strict loader, explicit fit precision, deterministic eval frozen trunk; separately scored JAX public baselines may have framework-rounding differences, not an isolated causal conditioning effect',scope='Every root/child/grid prediction OUT OF FOLD; frozen B trunk/public-head epoch0 warmstart; not a final all-bank trained critic or actor',new_games=0,scientific_qualification=False)
    with (output/'crossfit.json').open('x')as f:json.dump(result,f,sort_keys=True,allow_nan=False)
    return result


def score_predictions(manifest,records,predictions):
    from gpu_engine.device_rollout_truth_score import backup_from_grid
    if len(records)!=len(predictions):raise ValueError('full OOS prediction denominator required')
    by_id={r['row_id']:v for r,v in zip(records,predictions,strict=True)}
    if len(by_id)!=len(records):raise ValueError('duplicate private row identity')
    roots=[];children=[]
    backups={tuple(g['selector'][k]for k in ('stream','frame','owner')):backup_from_grid(g,[by_id[r['row_id']]for r in g.get('rows',[])])if g['status']=='OBSERVED'else None for g in manifest['grids']}
    for r,v in zip(records,predictions,strict=True):
        if r['kind']=='root':roots.append(dict(selector=r['selector'],prediction=v,outcome=r['outcome'],backup=backups[tuple(r['selector'][k]for k in ('stream','frame','owner'))]))
        elif r['kind']=='child':children.append(dict(selector=r['selector'],action=r['action'],prediction=v,rollouts=r['rollouts']))
    return truth().metrics(roots,children)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--expected-config-sha256',required=True)
    parser.add_argument('--rows',required=True);parser.add_argument('--expected-rows-sha256',required=True);parser.add_argument('--output',required=True);parser.add_argument('--enable-fit',action='store_true');a=parser.parse_args(argv)
    if not a.enable_fit:raise ValueError('explicit offline fit enablement required')
    config=artifact(dict(path=a.config,sha256=a.expected_config_sha256))
    crossfit(config,dict(path=a.rows,sha256=a.expected_rows_sha256),a.output,enable_fit=True)

if __name__=='__main__':main()
