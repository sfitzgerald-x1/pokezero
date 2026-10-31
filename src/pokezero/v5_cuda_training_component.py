"""Finite NON_GAME full-size CUDA optimizer wiring, never study collection.

Imports and fixture preparation allocate no model or device. Only explicit CLI
enablement executes the original trainer against two already-saved unit rows.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

SCHEMA='pokezero.v5.cuda_training_component.v1'
ROLES=(('A',0),('B',0),('D',0),('B',1))
STATUS='NON_GAME_CUDA_OPTIMIZER_COMPONENT_NOT_PHASE_C_QUALIFICATION'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def document(record):
    path=Path(record['path'])
    if sha(path)!=record['sha256']:raise ValueError('bound component input differs: '+str(path))
    return json.loads(path.read_bytes())


def validate_plan(plan):
    if (plan.get('schema')!=SCHEMA or plan.get('fixture_kind')!='NON_GAME'
            or plan.get('games')!=0 or plan.get('updates_per_arm')!=2
            or plan.get('rows_per_arm')!=2 or plan.get('unit_schedule_coordinates')!=
                dict(start_game=0,end_game=1,total_games=2000000)
            or [(x['arm'],x['replicate'])for x in plan.get('arms',())]!=list(ROLES)):
        raise ValueError('exact finite NON_GAME four-arm/two-row/two-update component required')
    root=Path(plan['remote_new_root'])
    if (not root.is_absolute() or root.parent!=Path('/shared/scott')
            or not root.name.startswith('pokezero-v5-cuda-training-component-')):
        raise ValueError('exclusive NON_GAME remote component root required')


def fixture_batch(columns,model_config,*,input_sha256):
    """Existing tensor/cache API with explicit unit metadata, no fake outcome.

    D's uniform zero-Q tensor is solely a NON_GAME loss fixture. It is not an
    original teacher record, sampled world, eligibility decision or game label.
    Terminal, survival, hidden-set and semantic-opponent truth remain unknown.
    """
    import numpy as np
    from . import v5_contract as c,v5_data as data
    from .dataset import TrainingBatch
    n=2;legal=np.asarray(columns['legal_action_mask'])
    if legal.shape!=(n,9) or legal.dtype!=np.bool_ or not legal.any(1).all():
        raise ValueError('two original nonempty legal unit rows required')
    c._binding(input_sha256)
    targets=legal.astype(np.float32)/legal.sum(1)[:,None]
    selected=np.argmax(legal,axis=1)
    cols={name:np.zeros((n,*shape),dtype=np.bool_ if 'mask' in name else
        np.int32 if name in ('opponent_semantic_action_targets','hidden_set_species','privileged_categorical_ids','privileged_token_type_ids') else np.float32)
        for name,shape in data.ARRAY_SHAPES.items()}
    records=[]
    for mask in legal:
        teacher=c.RootImprovement(tuple(float(x)for x in mask/mask.sum()),0.,
            tuple(0. if x else None for x in mask),2)
        teacher.validate(tuple(bool(x)for x in mask));records.append(asdict(teacher))
    search=model_config.arm=='D'
    cols.update(schema=c.TRAINING_CONTRACT_SCHEMA,observation_schema=c.OBSERVATION_SCHEMA_VERSION_V5,
        category_vocabulary_sha256=model_config.category_vocabulary_sha256,
        label_vocabulary_sha256=model_config.label_vocabulary.sha256,
        source_binding_sha256=model_config.label_vocabulary.source_binding_sha256,
        label_vocabulary=asdict(model_config.label_vocabulary),
        addresses=tuple(dict(stream=i+1,frame=1,owner=i)for i in range(n)),
        source_frame_sha256=(input_sha256,)*n,terminal_source_sha256=(None,)*n,
        terminal_status=('IN_PROGRESS',)*n,search_status=('LABELED'if search else'UNSELECTED',)*n,
        search_refusal=(None,)*n,root_improvement_records=tuple(records)if search else(None,)*n)
    return TrainingBatch(**{name:np.asarray(columns[name])for name in
        ('categorical_ids','numeric_features','token_type_ids','attention_mask','history_mask')},
        legal_action_mask=legal,action_indices=selected,rewards=(0.,)*n,returns=(0.,)*n,
        value_estimates=(0.,)*n,value_estimate_mask=(True,)*n,
        ppo_advantages=(.5,-.5),ppo_advantage_mask=(True,)*n,
        ppo_value_targets=(.25,-.25),ppo_value_target_mask=(True,)*n,
        opponent_action_indices=(-1,)*n,opponent_action_mask=(False,)*n,
        action_probabilities=tuple(float(targets[i,selected[i]])for i in range(n)),
        action_probability_mask=(True,)*n,training_weights=(1.,)*n,
        battle_ids=('NON_GAME_CUDA_UNIT_ROW_0','NON_GAME_CUDA_UNIT_ROW_1'),
        seeds=(0,0),format_ids=('NON_GAME',)*n,player_ids=('p1','p2'),turn_indices=(0,0),
        terminal_capped=(False,)*n,step_metadata=tuple(dict(fixture_kind='NON_GAME',
            observed_outcome=False,actual_teacher_eligible=False,fixture_input_sha256=input_sha256)for _ in range(n)),
        search_policy_targets=targets,search_policy_target_mask=(search,)*n,
        search_value_targets=(0.,)*n,search_value_target_mask=(False,)*n,v5_columns=cols)


def tensor_tree_sha(value,torch):
    """Compact comparison of actual model/AdamW state; bulk stays remote."""
    digest=hashlib.sha256()
    def add(item):
        if torch.is_tensor(item):
            x=item.detach().cpu().contiguous()
            digest.update(str((str(x.dtype),tuple(x.shape))).encode());digest.update(x.numpy().tobytes())
        elif isinstance(item,dict):
            for key in sorted(item,key=lambda x:str(x)):
                digest.update(repr(key).encode());add(item[key])
        elif isinstance(item,(tuple,list)):
            digest.update(str(type(item)).encode())
            for x in item:add(x)
        else:digest.update(repr(item).encode())
    add(value);return digest.hexdigest()


def execute(plan,out,*,enable_non_game=False):
    if enable_non_game is not True:raise ValueError('explicit NON_GAME component enablement required')
    validate_plan(plan);out=Path(out)
    if out!=Path(plan['remote_new_root'])/'attempt':raise ValueError('exclusive declared component attempt path required')
    # Bind every input before CUDA/model initialization. No dataset/collector,
    # constructor, team bank, engine, world or model0 entrypoint is imported.
    summary=document(plan['prior_transfer_summary'])
    if (summary['status']!='FULLSHAPE_TRANSFER_COMPONENT_PASS_NOT_C_QUALIFICATION'
            or summary['games']!=0 or summary['training_updates']!=0 or summary['owning_rows']!=8):
        raise ValueError('original untrained fullshape component differs')
    references={x['role']:x for x in summary['arms']}
    for row in plan['arms']:
        prior=references[row['arm']+str(row['replicate'])]
        if sha(row['inputs']['path'])!=row['inputs']['sha256'] or row['inputs']['sha256']!=prior['input_sha256']:
            raise ValueError('original NON_GAME public29 fixture differs')
    category_data=document(plan['category_vocabulary']);document(plan['control_inventory'])
    source_root=Path(__file__).resolve().parents[2]
    for name,digest in plan['source_files_sha256'].items():
        if sha(source_root/name)!=digest:raise ValueError('original component source bytes differ: '+name)
    from . import v5_worker as worker
    worker._source(plan['runtime_validation']);worker._memory(plan['runtime_validation']['memory_policy'],apply=True)
    worker._probe(plan['runtime_validation'])
    import numpy as np
    import torch
    from . import v5_model as model,v5_training as training,v5_phasec as phase,v5_data as data
    from .category_vocab import CategoryVocabulary
    from gpu_engine import v5_model as export
    category=CategoryVocabulary(tuple(category_data['tokens']),category_data['oov_buckets'],category_data['aliases'])
    profile=phase.ChampionProfile.read(plan['control_inventory']['path'],expected_sha256=plan['control_inventory']['sha256'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'GB300'not in torch.cuda.get_device_name():
        raise ValueError('one actual GB300 CUDA device required')
    out.mkdir(exist_ok=False);reports=[];sequence=0
    def completed(label,**extra):
        nonlocal sequence
        torch.cuda.synchronize();sequence+=1
        value=dict(sequence=sequence,completed_phase=label,at=time.time(),fixture_kind='NON_GAME',games=0,**extra)
        path=out/'progress.json';tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,sort_keys=True)+'\n');tmp.replace(path)
        print(json.dumps(value,sort_keys=True),flush=True)
    try:
        for row in plan['arms']:
            role=row['arm']+str(row['replicate']);prior=references[role]
            config=model.V5ModelConfig.from_metadata(prior['model'])
            chosen=profile.configurations(config.label_vocabulary,category_vocabulary=category,arm=row['arm'],replicate=row['replicate'])
            if chosen['model']!=config or (config.width,config.layers,config.expert_width)!=(512,3,512):
                raise ValueError('exact original full-size arm model differs')
            train=training.TrainConfig(**chosen['training'])
            if train.seed!=row['replicate'] or train.precision!='bfloat16' or train.device!='cuda:0':
                raise ValueError('original CUDA BF16 arm seed/control differs')
            directory=out/role;directory.mkdir();(directory/'NON_GAME_ONLY.json').write_text(json.dumps(dict(
                fixture_kind='NON_GAME',games=0,actual_teacher_eligible=False,
                schedule_coordinates='UNIT_API_ONLY_NOT_OBSERVED_OR_DECLARED_GAME_PROGRESS',
                terminal_outcomes='ALL_UNKNOWN',production_use=False),sort_keys=True)+'\n')
            with np.load(row['inputs']['path'],allow_pickle=False)as arrays:
                batch=fixture_batch({name:arrays[name]for name in arrays.files},config,input_sha256=row['inputs']['sha256'])
            cache=directory/'NON_GAME_CACHE';data.write_cache(cache,batch)
            runtime=training.create(config,chosen['loss'],train,enable_v5=True)
            initial=tensor_tree_sha(runtime.model.state_dict(),torch)
            completed(role+'_ORIGINAL_MODEL_INITIALIZED',parameters=sum(x.numel()for x in runtime.model.parameters()))
            window=training.DeclaredGameWindow(**plan['unit_schedule_coordinates'])
            first=training.train_caches(runtime,[cache],max_updates=1,learning_rate_window=window)
            changed=tensor_tree_sha(runtime.model.state_dict(),torch)
            if initial==changed or first['steps']!=1 or first['rows']!=2 or first['updates'][0]['gradient_norm']<=0:
                raise ValueError('actual first optimizer update/gradient missing')
            before_optimizer=tensor_tree_sha(runtime.optimizer.state_dict(),torch)
            checkpoint=training.save_checkpoint(directory/'NON_GAME-update1.pt',runtime)
            completed(role+'_CUDA_UPDATE1_CHECKPOINT_COMPLETE')
            del runtime;gc.collect();torch.cuda.empty_cache()
            if sha(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('component checkpoint changed')
            runtime=training.load_checkpoint(checkpoint['path'],enable_v5=True,expected_model_config=config,
                expected_loss_config=chosen['loss'],expected_train_config=train,expected_learning_rate_window=window)
            if (tensor_tree_sha(runtime.model.state_dict(),torch)!=changed
                    or tensor_tree_sha(runtime.optimizer.state_dict(),torch)!=before_optimizer):
                raise ValueError('original CUDA checkpoint model/AdamW restore differs')
            second=training.train_caches(runtime,[cache],max_updates=1,learning_rate_window=window)
            if second['steps']!=2 or second['rows']!=4 or second['updates'][0]['gradient_norm']<=0:
                raise ValueError('actual resumed optimizer update missing')
            checkpoint=training.save_checkpoint(directory/'NON_GAME-update2.pt',runtime)
            del runtime;gc.collect();torch.cuda.empty_cache()
            archive=directory/'NON_GAME-actor.npz'
            exported=export.export(checkpoint['path'],archive,expected_checkpoint_sha256=checkpoint['sha256'],enable_v5=True)
            params,loaded,binding=export.load(archive,expected_archive_sha256=exported['weights_archive_sha256'],
                expected_checkpoint_sha256=checkpoint['sha256'],expected_category_vocabulary_sha256=config.category_vocabulary_sha256,
                expected_label_vocabulary_sha256=config.label_vocabulary.sha256,enable_v5=True)
            if loaded!=config:raise ValueError('original actor export/load model differs')
            result=dict(role=role,status='PASS_NON_GAME_CUDA_COMPONENT',games=0,unique_fixture_rows=2,processed_fixture_rows=4,
                optimizer_updates=2,model=config.metadata(),loss=chosen['loss'].metadata(),training=training.train_config_metadata(train),
                first_update=first['updates'][0],resumed_update=second['updates'][0],
                initial_model_sha256=initial,update1_model_sha256=changed,restored_optimizer_sha256=before_optimizer,
                checkpoint_sha256=checkpoint['sha256'],archive_sha256=binding['weights_archive_sha256'],
                unit_D_policy_rows=2 if config.arm=='D' else 0,unit_D_value_rows=0,
                observed_outcome_rows=0,survival_rows=0,auxiliary_truth_rows=0,
                limits='Two fixed NON_GAME rows, not production batch/memory/throughput or training-signal/strength proof.')
            reports.append(result);(directory/'result.json').write_text(json.dumps(result,sort_keys=True)+'\n')
            import jax
            jax.block_until_ready(params)
            del params,batch;gc.collect()
            jax.clear_caches();completed(role+'_UPDATE_RESUME_EXPORT_COMPLETE')
        result=dict(status=STATUS,fixture_kind='NON_GAME',games=0,models=4,unique_fixture_rows=8,
            optimizer_updates=8,arms=reports,limits=plan['limits'])
        (out/'summary.json').write_text(json.dumps(result,sort_keys=True)+'\n');completed('ALL_FOUR_NON_GAME_COMPONENTS_COMPLETE')
        return result
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps(dict(status='COMPONENT_FAILED_NO_QUALIFICATION',fixture_kind='NON_GAME',
            games=0,completed_arms=reports,error_type=type(error).__name__,error=str(error)),sort_keys=True)+'\n')
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-non-game-component',action='store_true')
    parser.add_argument('--prespec',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    if args.enable_non_game_component is not True:raise ValueError('explicit NON_GAME enablement required before input access')
    result=execute(json.loads(args.prespec.read_bytes()),args.out,enable_non_game=True)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':main()
