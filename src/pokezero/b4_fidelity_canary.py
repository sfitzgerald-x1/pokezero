"""Opt-in original-v4 champion fidelity canary, separate from scratch v5 arms.

This module declares no seeds or artifacts. The caller binds the original
prepared engine/pool and the complete prospective ordered game and FP banks.
Unknown attempts are never refilled; only literal complete games enter PPO.
"""
from dataclasses import asdict, dataclass, replace, fields
import hashlib
import io
import json
import math
from pathlib import Path
import time

from . import neural_policy as neural

CHAMPION_SHA = '0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be'
DECLARED_GAMES = 200000
PANEL_SEEDS = 500


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False,
                      default=lambda x: str(x) if isinstance(x, Path) else asdict(x)).encode()


def sha(value):return hashlib.sha256(encoded(value)).hexdigest()


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):result.update(block)
    return result.hexdigest()


def require(condition, reason):
    if not condition:raise ValueError(reason)


@dataclass(frozen=True)
class Game:
    stream_id:int
    battle_id:str
    seed:int
    team_indices:tuple[int,int]

    def __post_init__(self):
        object.__setattr__(self,'team_indices',tuple(self.team_indices))
        require(type(self.stream_id) is int and 0<self.stream_id<2**31 and
                type(self.seed) is int and 0<=self.seed<2**32 and type(self.battle_id) is str and self.battle_id and
                len(self.team_indices)==2 and all(type(x) is int and x>=0 for x in self.team_indices), 'declared original game')


@dataclass(frozen=True)
class Batch:
    games:tuple[Game,...]
    constructor_seed:int
    sampling_seed:int
    gender_seed:int
    gender_domain:int
    gender_root_ids:tuple[int,...]

    def __post_init__(self):
        object.__setattr__(self,'games',tuple(self.games));object.__setattr__(self,'gender_root_ids',tuple(self.gender_root_ids))
        require(self.games and all(type(x) is Game for x in self.games) and
                all(type(x) is int and 0<=x<2**32 for x in (self.constructor_seed,self.sampling_seed,self.gender_seed,self.gender_domain)) and
                len(self.gender_root_ids)==len(self.games) and all(type(x) is int and 0<=x<2**32 for x in self.gender_root_ids),
                'explicit finite batch seeds and original gender roots')


@dataclass(frozen=True)
class Plan:
    iterations:tuple[tuple[Batch,...],...]
    pool_sha256:str
    source_commit:str
    prepared_binding:str
    max_requests:int
    fp_configs:tuple[dict,dict]
    champion_sha256:str=CHAMPION_SHA
    stall_floor_seconds:float=1800.
    stall_multiplier:float=5.
    backstop_seconds:float=30*86400.

    def __post_init__(self):
        object.__setattr__(self,'iterations',tuple(tuple(x) for x in self.iterations))
        object.__setattr__(self,'fp_configs',tuple(dict(x) for x in self.fp_configs))
        games=[g for iteration in self.iterations for batch in iteration for g in batch.games]
        require(self.iterations and all(iteration for iteration in self.iterations) and len(games)==DECLARED_GAMES,
                'exact prospective 200k declared games, no outcome-based budget')
        for name in ('stream_id','battle_id','seed'):
            require(len({getattr(g,name) for g in games})==len(games),'duplicate declared '+name)
        teams=[i for g in games for i in g.team_indices]
        require(len(set(teams))==len(teams),'no team reuse/refill')
        genders=[i for iteration in self.iterations for batch in iteration for i in batch.gender_root_ids]
        require(len(set(genders))==len(genders),'unique declared original gender roots')
        for value in (self.pool_sha256,self.prepared_binding,self.champion_sha256):
            require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'artifact SHA256')
        require(self.champion_sha256==CHAMPION_SHA and len(self.source_commit)==40,'original9375/source binding')
        require(type(self.max_requests) is int and self.max_requests>0 and self.stall_floor_seconds==1800. and
                self.stall_multiplier==5. and self.backstop_seconds==30*86400.,'finite cap/adaptive watchdog')
        require(len(self.fp_configs)==2,'both original FP seats')
        first=self.fp_configs[0]
        for side,config in zip(('p1','p2'),self.fp_configs):
            require(config['games']==PANEL_SEEDS and config['pokezero_player']==side and config['policy_mode']=='raw' and
                    config['search_time_ms']==1000 and config.get('opponent_policy_mode','foul-play')=='foul-play' and
                    config.get('capture_driver','checkpoint')=='checkpoint' and config.get('encoder_kind','legacy-mixed')=='legacy-mixed' and
                    config.get('value_checkpoint') is None and
                    not config.get('live_continuation_smoke',False) and not config.get('live_continuation_oracle',False),
                    'fixed500 seeds x both seats, original raw Python/Showdown champion vs FP1000ms')
            require(config['seed_start']==first['seed_start'] and config['foulplay_random_seed']==first['foulplay_random_seed'],
                    'paired FP original seed bank')
            require(type(config['seed_start']) is int and type(config['foulplay_random_seed']) is int,'explicit FP seeds')
            require(all(0<=config[k]<config[k]+PANEL_SEEDS<=2**32 for k in ('seed_start','foulplay_random_seed')),'finite original uint32 FP banks')

    def manifest(self):
        return dict(schema='pokezero.b4.original_champion_fidelity_canary.v1',**asdict(self),
                    schedule='Original mit-thesis fresh0..1 over declared200k',optimizer='fresh AdamW: original checkpoint has no optimizer',
                    unknown_training='Exclude caps/refusals/failures, retain full declared denominator; no refill',
                    evaluation='Final declared checkpoint only;1000games/model paired against untouched9375; no automatic pass/equivalence margin')


def load_plan(path,*,expected_sha256):
    require(file_sha(path)==expected_sha256,'original declaration file digest')
    document=json.loads(Path(path).read_bytes());values={field.name:document[field.name] for field in fields(Plan)}
    values['iterations']=tuple(tuple(Batch(**{**b,'games':tuple(Game(**g) for g in b['games'])}) for b in batches)
                               for batches in values['iterations'])
    plan=Plan(**values)
    require(encoded(plan.manifest())==encoded(document),'no missing/extra/reinterpreted controls in full declaration')
    return plan


class Journal:
    def __init__(self,path,plan,*,resume=False):
        self.path=Path(path);self.events=[];blob=encoded(plan.manifest());declaration=self.path/'declaration.json'
        if resume:require(declaration.read_bytes()==blob,'original accepted declaration differs')
        else:
            self.path.mkdir(parents=True,exist_ok=False);declaration.write_bytes(blob)
        self.journal=self.path/'journal.jsonl'
        if self.journal.exists():
            for line in self.journal.read_bytes().splitlines():
                event=json.loads(line);digest=event.pop('sha256')
                require(event['sequence']==len(self.events) and event['previous']==(self.events[-1]['sha256'] if self.events else None)
                        and sha(event)==digest,'append-only journal differs')
                self.events.append(dict(event,sha256=digest))

    def emit(self,kind,**payload):
        import os
        event=dict(sequence=len(self.events),previous=self.events[-1]['sha256'] if self.events else None,kind=kind,payload=payload)
        event['sha256']=sha(event)
        with self.journal.open('ab') as stream:stream.write(encoded(event)+b'\n');stream.flush();os.fsync(stream.fileno())
        self.events.append(event)

    def matching(self,kind,**fields):
        return [e['payload'] for e in self.events if e['kind']==kind and all(e['payload'].get(k)==v for k,v in fields.items())]


class Progress:
    def __init__(self,plan):self.plan=plan;self.start=self.last=time.monotonic();self.longest=0.
    def completed(self,duration):
        require(type(duration) in (int,float) and math.isfinite(duration) and duration>=0,'actual completed-game timing')
        self.longest=max(self.longest,duration);self.last=time.monotonic()
    def check(self):
        now=time.monotonic()
        if now-self.start>self.plan.backstop_seconds:raise TimeoutError('distant canary backstop')
        if now-self.last>max(self.plan.stall_floor_seconds,self.plan.stall_multiplier*self.longest):
            raise TimeoutError('no actual completed game within adaptive window')


def controls(payload,*,start,end,device):
    """Only the prespecified fresh schedule/device differs from direct9375."""
    config=neural.TransformerTrainingConfig(**payload)
    require(config.objective=='ppo' and config.ppo_target_mode=='gae' and config.discount==.9999 and
            config.gae_lambda==.754 and config.value_clip_range==.0184 and config.learning_rate==5.9e-5 and
            config.entropy_coef==.0147 and config.clip_epsilon==.0829 and config.value_loss_weight==.4375 and
            config.batch_size==512 and config.epochs==5 and config.max_grad_norm==.543 and
            config.opponent_action_loss_weight==.1 and config.shaping_weights is None and
            config.hp_delta_return_weight==config.faint_delta_return_weight==config.turn_penalty==0 and
            config.turn_penalty_after is None and config.capped_terminal_value==-.25 and
            not config.freeze_non_value_parameters and not config.freeze_policy_heads,'direct9375 controls differ')
    return replace(config,learning_rate_schedule='mit-thesis',learning_rate_schedule_total_games=DECLARED_GAMES,
                   learning_rate_progress_start=start/DECLARED_GAMES,learning_rate_progress_end=end/DECLARED_GAMES,
                   learning_rate_warmup_progress=0.,learning_rate_resume_warmup_start_progress=None,
                   learning_rate_resume_warmup_end_progress=None,learning_rate_resume_warmup_start_lr=None,device=device)


def dataset_controls(config):
    from .dataset import TrajectoryDatasetConfig
    return TrajectoryDatasetConfig(window_size=config.window_size,discount=config.discount,
        capped_terminal_value=config.capped_terminal_value,hp_delta_return_weight=config.hp_delta_return_weight,
        faint_delta_return_weight=config.faint_delta_return_weight,turn_penalty_after=config.turn_penalty_after,
        turn_penalty=config.turn_penalty,ppo_target_mode=config.ppo_target_mode,gae_lambda=config.gae_lambda)


def checkpoint_payload(path,*,expected_sha256):
    """Original immutable byte-snapshot helper; hash and deserialize SAME bytes."""
    from gpu_engine import champion
    blob,digest=champion.snapshot(path)
    require(digest==expected_sha256,'immutable original checkpoint snapshot differs')
    return neural.require_torch().load(io.BytesIO(blob),map_location='cpu',weights_only=True)


def run(plan,path,*,checkpoint,prepared,pool,gender_policy,device='cuda',resume=False,enable_canary=False):
    """Real source path; no default runtime, artifacts, seed bank or v5 model."""
    require(enable_canary is True and type(plan) is Plan,'explicit original champion canary required')
    require(file_sha(checkpoint)==CHAMPION_SHA,'exact untouched original9375 checkpoint required')
    from gpu_engine import b4_canary_collection as engine,champion
    from . import dataset
    book=Journal(path,plan,resume=resume);progress=Progress(plan)
    original=checkpoint_payload(checkpoint,expected_sha256=CHAMPION_SHA)
    require('optimizer' not in original and 'optimizer_state_dict' not in original,'unexpected optimizer: choose explicit resume semantics')
    require(original.get('value_calibration_transform') is None,'raw champion required; calibration transform cannot be silently dropped')
    model_config=neural.parse_transformer_model_config(original)
    require(model_config.observation_schema_version=='pokezero.observation.v4' and model_config.token_count==23 and
            model_config.numeric_feature_count==132 and model_config.categorical_feature_count==41 and
            model_config.embedding_dim==512 and model_config.transformer_layers==3 and model_config.feedforward_dim==2048 and
            model_config.reward_shaping is None,'original v4 architecture/reward contract required')
    backend=engine.Backend(prepared,pool,gender_policy=gender_policy,plan=plan,model_config=model_config)
    current=Path(checkpoint);declared=0
    for iteration,batches in enumerate(plan.iterations):
        end=declared+sum(len(batch.games) for batch in batches);train=controls(original['training_config'],start=declared,end=end,device=device)
        finished=book.matching('TRAIN_END',iteration=iteration)
        if finished:
            require(len(finished)==1,'duplicate trained iteration');current=book.path/finished[0]['checkpoint']
            require(file_sha(current)==finished[0]['checkpoint_sha256'],'immutable completed checkpoint differs');declared=end;continue
        if book.matching('TRAIN_START',iteration=iteration):raise RuntimeError('interrupted legacy optimizer attempt UNKNOWN; no automatic retrain')
        caches=[]
        def actual_end(stream,status,duration,source_sha256):
            book.emit('GAME_END',stream_id=stream,status=status,duration_seconds=duration,source_sha256=source_sha256)
            if status=='COMPLETE':progress.completed(duration)
        for index,batch in enumerate(batches):
            done=book.matching('COLLECT_END',iteration=iteration,batch=index)
            if done:
                require(len(done)==1,'duplicate collected batch');result=done[0]
            elif book.matching('COLLECT_START',iteration=iteration,batch=index):
                # Original attempted cells remain unknown; never reconstruct/replay.
                endings=[e for e in book.matching('GAME_END') if e.get('stream_id') in {g.stream_id for g in batch.games}]
                complete=[e['stream_id'] for e in endings if e['status']=='COMPLETE']
                require(len(set(e['stream_id'] for e in endings))==len(endings),'duplicate actual cell ending')
                result=dict(iteration=iteration,batch=index,cache=None,status='INTERRUPTED_UNKNOWN',games=len(batch.games),
                    complete=len(complete),unknown=len(batch.games)-len(complete),preserved_completed_streams=complete,
                    note='Original ending receipts and spooled records retained; interrupted batch cache not automatically reconstructed')
                book.emit('COLLECT_END',**result)
            else:
                book.emit('BATCH_STARTED',iteration=iteration,batch=index)
                book.emit('COLLECT_START',iteration=iteration,batch=index,streams=[g.stream_id for g in batch.games],checkpoint_sha256=file_sha(current))
                result=backend.collect(batch,current,book.path/f'collection-{iteration}-{index}',
                    dataset_config=dataset_controls(train),on_end=actual_end,
                    check_progress=progress.check)
                result=dict(result,iteration=iteration,batch=index);book.emit('COLLECT_END',**result)
            if result.get('cache'):
                cache=Path(result['cache']);require(file_sha(cache/'metadata.json')==result['cache_metadata_sha256'],'immutable cache receipt differs')
                caches.append(cache)
        if not caches:
            book.emit('CANARY_UNKNOWN',iteration=iteration,reason='no terminal-complete engine trajectories; no invented training')
            return dict(status='UNKNOWN_NO_COMPLETE_CACHE',declared_games=end)
        book.emit('TRAIN_START',iteration=iteration,declared_start=declared,declared_end=end,initial_checkpoint_sha256=file_sha(current),controls=train.to_dict())
        payload=checkpoint_payload(current,expected_sha256=file_sha(current))
        require(neural.parse_transformer_model_config(payload)==model_config,'original model configuration drift')
        model=neural.EntityTokenTransformerPolicy(model_config)
        model.load_state_dict(payload['state_dict'],strict=True)  # No fresh-head compatibility path.
        def epoch_progress(_,partial):
            require(0<len(partial.epochs)<=train.epochs,'original completed PPO epoch count')
            book.emit('TRAIN_PROGRESS',steps=iteration*train.epochs+len(partial.epochs),iteration=iteration,completed_epochs=len(partial.epochs))
        train_started=time.monotonic()
        model,result=neural.train_transformer_policy(caches,model_config=model_config,training_config=train,initial_model=model,epoch_callback=epoch_progress)
        # Collection stall time excludes the separately supervised training phase.
        progress.last+=time.monotonic()-train_started
        current=book.path/f'checkpoint-{iteration}.pt';require(not current.exists(),'exclusive checkpoint output')
        neural.save_transformer_checkpoint(current,model,result=result)
        archive=book.path/f'weights-{iteration}.npz';champion.export(current,archive)
        book.emit('TRAIN_END',iteration=iteration,checkpoint=current.name,checkpoint_sha256=file_sha(current),archive=archive.name,archive_sha256=file_sha(archive))
        declared=end
    ending=dict(declared_games=declared,checkpoint=str(current.relative_to(book.path)),checkpoint_sha256=file_sha(current))
    prior=book.matching('CANARY_TRAINING_END')
    if prior:require(prior==[ending],'immutable final checkpoint differs')
    else:book.emit('CANARY_TRAINING_END',**ending)
    return dict(status='FINAL_CHECKPOINT_NOT_EVALUATED',checkpoint=current,declared_games=declared)


async def evaluate(plan,path,*,champion_checkpoint,final_checkpoint,enable_canary=False):
    """Fixed paired raw FP endpoints only; attempted cells never repeat on resume."""
    require(enable_canary is True and type(plan) is Plan,'explicit separate canary endpoint')
    require(file_sha(champion_checkpoint)==CHAMPION_SHA,'untouched paired control checkpoint')
    book=Journal(path,plan,resume=True)
    final=book.matching('CANARY_TRAINING_END')
    require(final and file_sha(final_checkpoint)==final[-1]['checkpoint_sha256'],'final declared checkpoint only, no selection')
    from . import foulplay_bridge as fp,v5_foulplay
    progress=Progress(plan);rows=[];book.emit('BATCH_STARTED',stage='FP_ENDPOINT')
    for offset in range(PANEL_SEEDS):
        for seat,raw in enumerate(plan.fp_configs):
            for arm,checkpoint in (('CONTROL',champion_checkpoint),('CANARY',final_checkpoint)):
                key=dict(arm=arm,offset=offset,seat=seat);done=book.matching('FP_END',**key)
                if done:require(len(done)==1,'duplicate FP cell');rows.append(done[0]);continue
                if book.matching('FP_START',**key):
                    row=dict(**key,status='UNKNOWN',score=None,reason='attempted interrupted cell; no replay');book.emit('FP_END',**row);rows.append(row);continue
                progress.check();values={**raw,'checkpoint':Path(checkpoint),'games':1,
                    'seed_start':raw['seed_start']+offset,'foulplay_random_seed':raw['foulplay_random_seed']+offset}
                for field in fields(fp.ControlledFoulPlayConfig):
                    if field.name in values and values[field.name] is not None and 'Path' in str(field.type):values[field.name]=Path(values[field.name])
                config=fp.ControlledFoulPlayConfig(**values)
                book.emit('FP_START',**key,seed=config.seed_start,checkpoint_sha256=file_sha(checkpoint));start=time.monotonic()
                game=None
                try:
                    result=await fp.run_controlled_foulplay_benchmark(config)
                    require(len(result.games)==1,'one original declared FP cell')
                    game=result.games[0].to_dict();score=v5_foulplay._score(config,game)
                    row=dict(**key,status='COMPLETE' if score is not None else 'UNKNOWN',score=score,source_game=game)
                    if score is not None:
                        duration=time.monotonic()-start;progress.completed(duration)
                        book.emit('GAME_END',phase='FP',**key,status='COMPLETE',duration_seconds=duration,source_sha256=sha(game))
                except Exception as error:row=dict(**key,status='UNKNOWN',score=None,source_game=game,reason=type(error).__name__+': '+str(error))
                book.emit('FP_END',**row);rows.append(row)
    paired=[]
    for offset in range(PANEL_SEEDS):
        for seat in (0,1):
            pair={r['arm']:r for r in rows if r['offset']==offset and r['seat']==seat}
            paired.append(None if any(x['status']!='COMPLETE' for x in pair.values()) else pair['CANARY']['score']-pair['CONTROL']['score'])
    known=[x for x in paired if x is not None];unknown=len(paired)-len(known)
    report=dict(status='ENDPOINT_RECORDED',original_paired_cells=PANEL_SEEDS*2,known_pairs=len(known),unknown_pairs=unknown,
                paired_mean=sum(known)/len(known) if known else None,
                identification_bounds=[(sum(known)-unknown)/len(paired),(sum(known)+unknown)/len(paired)],
                automatic_fidelity_pass=False,uncertainty='No equivalence margin or significance pass inferred; paired interval/result review required',rows=rows)
    report['paired_seed_means']=[None if paired[2*i] is None or paired[2*i+1] is None else
        (paired[2*i]+paired[2*i+1])/2 for i in range(PANEL_SEEDS)]
    report['interval_unit']='Original paired seed block (both seats); no IID-cell significance claim'
    book.emit('FP_ENDPOINT',report=report);return report


def supervise_owned(command,*,plan,path,**kwargs):
    """Reuse the reviewed owned-process cleanup and durable-progress watchdog."""
    from types import SimpleNamespace
    from .v5_run import supervise_owned as supervise,Watchdog
    operational=SimpleNamespace(evaluation=None,iterations=plan.iterations,
        watchdog=Watchdog(initial_window_seconds=plan.stall_floor_seconds,longest_multiplier=plan.stall_multiplier,
                          backstop_seconds=plan.backstop_seconds))
    return supervise(command,plan=operational,path=path,**kwargs)
