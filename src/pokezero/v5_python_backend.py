"""Default-off owning Showdown collector for the original finite v5 runner.

Preparation reads bound source data only. Actual games require an explicit
worker choice. The device and retained-engine backends are never fallbacks.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import threading
import time
from types import MappingProxyType
import uuid

import numpy as np

from . import v5_run as runs
from .env import BattleStartOverride
from . import v5_python_collection as collection


@dataclass(frozen=True)
class Prepared:
    arrays: object
    metadata: object
    pool_sha256: str
    source: object
    compiled_source: object
    dex: object
    vocabulary: object
    contract: object
    public_config: object
    gender_policy: object


def prepare(inputs):
    """Read the literal pool, including IVs, without constructing any game."""
    from .category_vocab import CategoryVocabulary
    from .dex import load_showdown_dex_cached
    from .randbat import load_gen3_randbat_source_cached
    from .live_protocol_actor import ActorContract
    from .local_showdown import LocalShowdownConfig
    from .observation import ObservationFeatureMasks
    from .showdown import V4_REPLAY_OBSERVATION_SPEC
    from gpu_engine import public_belief, team_pool
    runtime=inputs.runtime
    ref=runtime['constructor_pool'];payload=Path(ref['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=ref['sha256'] or ref['sha256']!=inputs.plan.pool_sha256:
        raise ValueError('original constructor pool bytes differ')
    arrays={}
    with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
        metadata=json.loads(archive['metadata_utf8'].tobytes())
        count=metadata['completed_teams']
        for name,(shape,dtype) in team_pool.FIELDS.items():
            value=archive[name]
            if value.shape!=(count,*shape) or value.dtype!=np.dtype(dtype):
                raise ValueError('original constructor field geometry differs: '+name)
            arrays[name]=np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)
        for name in ('gender','happiness'):
            if name in archive.files:
                value=archive[name]
                if value.shape!=(count,6) or value.dtype.kind not in 'iu':
                    raise ValueError('literal optional constructor field differs: '+name)
                arrays[name]=np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)
    if metadata['compiled_tree']['sha256']!=runtime['showdown']['compiled_tree_sha256']:
        raise ValueError('original compiled constructor source differs')
    directory=Path(__file__).resolve().parents[2]/'gpu_engine'
    code="const p=require(process.argv[1]); console.log(JSON.stringify(p.compiledTree(process.argv[2])));"
    show=runtime['showdown']
    compiled=json.loads(subprocess.check_output([show['node']['path'],'-e',code,str(directory/'gender_policy.cjs'),
        str(Path(show['root'])/'dist')],text=True))
    if compiled!=metadata['compiled_tree']:raise ValueError('actual compiled Showdown tree differs')
    v=inputs.vocabulary
    vocabulary=CategoryVocabulary(tuple(v['tokens']),v['oov_buckets'],v['aliases'])
    contract=ActorContract.champion()
    source=load_gen3_randbat_source_cached(show['root'])
    if source.metadata.source_hash!=contract.belief_set_source_hash:
        raise ValueError('original public belief source differs')
    masks=dict(contract.feature_masks)
    masks['opponent_tendency_stats_block']=masks.pop('stats_block')
    config=LocalShowdownConfig(showdown_root=show['root'],node_binary=show['node']['path'],
        observation_spec=V4_REPLAY_OBSERVATION_SPEC,feature_masks=ObservationFeatureMasks(**masks),
        category_vocab=vocabulary,set_belief_source=True)
    if inputs.gender_policy is not None:
        provenance=inputs.gender_policy.get('provenance',{})
        if (provenance.get('showdown_commit')!=contract.showdown_revision
                or provenance.get('compiled_tree')!=compiled or provenance.get('pool_sha256')!=ref['sha256']
                or provenance.get('generator_sources')!={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                    for name in('gender_policy.cjs','generate_teams.cjs')}):
            raise ValueError('original gender source/provenance differs')
    return Prepared(MappingProxyType(arrays),metadata,ref['sha256'],source,
        public_belief.compile_source(source),load_showdown_dex_cached(show['root']),
        vocabulary,contract,config,inputs.gender_policy)


def _genders(prepared,batch):
    """Reuse the declared root/seat/party gender addresses, with no redraw."""
    from gpu_engine.gender import initialize_genders
    from gpu_engine.source_provenance import _GenderInput
    indices=np.asarray([g.team_indices for g in batch.games],np.int64)
    species=prepared.arrays['species'][indices]
    explicit=(prepared.arrays['gender'][indices] if 'gender' in prepared.arrays
        else np.full(species.shape,-1,np.int32))
    if batch.gender_seed is None:
        if not np.isin(explicit,[0,1,2]).all():
            raise ValueError('unresolved constructor gender; explicit recipe required')
        return explicit
    policy=prepared.gender_policy
    if policy is None or policy.get('provenance',{}).get('pool_sha256')!=prepared.pool_sha256:
        raise ValueError('original pool-bound gender policy required')
    state=_GenderInput(species,np.full(species.shape,-1,np.int32))
    return np.asarray(initialize_genders(state,policy,prepared.metadata['vocabulary']['species'],
        seed=batch.gender_seed,domain=batch.gender_domain,root_ids=batch.gender_root_ids,
        explicit_genders=explicit).gender)


def packed_start(prepared,game,genders):
    """Exact original constructor sets; private teams go only to Showdown."""
    from .showdown_fixture import FixturePokemon, pack_pokemon
    arrays=prepared.arrays;v=prepared.metadata['vocabulary'];stats=('hp','atk','def','spa','spd','spe')
    if np.asarray(genders).shape!=(2,6) or not np.isin(genders,[0,1,2]).all():
        raise ValueError('two complete literal gender arrays required')
    teams={}
    for side,index in enumerate(game.team_indices):
        if not 0<=index<len(arrays['species']):raise ValueError('declared team index outside pool')
        mons=[]
        for slot in range(6):
            def label(field,kind):
                value=int(arrays[field][index,slot])
                if not 0<=value<len(v[kind]):raise ValueError('constructor namespace differs')
                return '' if value==0 else v[kind][value]
            moves=[v['moves'][int(value)]for value in arrays['moves'][index,slot]if value]
            mon=FixturePokemon(species=label('species','species'),moves=moves,
                item=label('item','items'),ability=label('ability','abilities'),
                level=int(arrays['level'][index,slot]),nature=label('nature','natures'),
                ivs=dict(zip(stats,map(int,arrays['ivs'][index,slot]))),
                evs=dict(zip(stats,map(int,arrays['evs'][index,slot]))),
                gender=('N','M','F')[int(genders[side,slot])])
            fields=pack_pokemon(mon).split('|')
            # Same explicit default as the original pool constructor: the
            # pinned generator omits happiness and Pokemon defaults to255.
            happiness=int(arrays['happiness'][index,slot])if 'happiness'in arrays else 255
            if not 0<=happiness<=255:raise ValueError('literal constructor happiness out of range')
            fields[11]=str(happiness);mons.append('|'.join(fields))
        teams['p'+str(side+1)]=']'.join(mons)
    return BattleStartOverride(teams,observation_format_id='gen3randombattle')


def chance_seed(batch,game):
    """Prospective Showdown domain, independent of dispatch/filtered row order."""
    return int(runs._sha(dict(schema='pokezero.showdown_constructor_seed.v5',
        constructor_seed=batch.constructor_seed,stream=game.stream_id,cell_seed=game.seed))[:8],16)


def _snapshot_binding(snapshot):
    return {name:snapshot[name]for name in('archive_sha256','checkpoint_sha256','model_sha256')}


def _expected_actors(current,assignment):
    side=assignment['learner_side']
    if type(side)is not int or side not in(0,1):raise ValueError('original learner owner required')
    pair=[_snapshot_binding(assignment['opponent'])]*2;pair[side]=_snapshot_binding(current)
    return pair


def _factory(prepared,snapshot,store):
    from .v5_phasec import category_binding
    from gpu_engine import v4_encoder
    from gpu_engine.v5_live_protocol import load_factory
    contract=replace(prepared.contract,checkpoint_sha256=snapshot['checkpoint_sha256'],
        model_config_sha256=snapshot['model_sha256'])
    catalog=v4_encoder.compile_encoder(prepared.compiled_source,prepared.dex,prepared.vocabulary,
        checkpoint_sha256=contract.checkpoint_sha256,feature_masks=dict(contract.feature_masks))
    return load_factory(store.resolve(snapshot['archive'],snapshot['archive_sha256']),
        catalog=catalog,contract=contract,expected_archive_sha256=snapshot['archive_sha256'],
        expected_checkpoint_sha256=snapshot['checkpoint_sha256'],
        expected_category_vocabulary_sha256=category_binding(prepared.vocabulary),
        expected_label_vocabulary_sha256=store.plan.model.label_vocabulary.sha256,
        compute_dtype='bfloat16',learner_seconds=60.,deterministic=False,enable_v5=True)


def _write_game_cache(path,decisions,plan):
    from . import dataset,v5_data
    from gpu_engine.v5_collection_data import cache_batches
    batches=tuple(cache_batches(decisions,loss_config=plan.loss,batch_size=plan.train.batch_size,discount=plan.discount))
    if not batches:raise ValueError('no actual game rows to cache')
    batch=dataset._combine_training_batches(batches)
    # The original multi-batch combiner stores search targets as float32;
    # its single-batch fast path leaves Python tuples as float64. A/B has
    # no search labels. Canonicalize only those inactive zero columns so
    # game boundaries cannot change the merged cache's storage dtype.
    replacements={}
    for target,mask in(('search_policy_targets','search_policy_target_mask'),
            ('search_value_targets','search_value_target_mask')):
        values=np.asarray(getattr(batch,target))
        if np.any(getattr(batch,mask))or np.any(values!=0):
            raise ValueError('Python A/B cannot contain search-label targets')
        replacements[target]=np.asarray(values,np.float32)
    return v5_data.write_cache(path,replace(batch,**replacements))


class RuntimeBackend:
    """Original runner callbacks, with starts deferred to actual dispatch."""
    deferred_game_starts=True

    def __init__(self,prepared,*,workers=1):
        if type(prepared)is not Prepared or type(workers)is not int or workers<1:
            raise ValueError('explicit prepared Python backend and positive worker count required')
        self.prepared=prepared;self.workers=workers;self.factories={};self._plan=None
        self._games=None;self._locations=None

    def check_plan(self,plan):
        from .v5_phasec import category_binding
        from .v5_contract import source_variant_census
        if type(plan)is not runs.Plan or plan.model.arm not in ('A','B'):
            raise runs.MissingDependency('Python fallback is explicit raw A/B only; D is deferred')
        if (plan.pool_sha256!=self.prepared.pool_sha256
                or plan.model.category_vocabulary_sha256!=category_binding(self.prepared.vocabulary)
                or plan.model.category_vocab_size!=self.prepared.vocabulary.size
                or plan.model.label_vocabulary.source_binding_sha256!=source_variant_census(self.prepared.source)['source_rows_sha256']):
            raise ValueError('same original constructor/category/source-label namespace required')
        if self._plan is not None and self._plan!=plan:raise ValueError('one original finite Plan required')
        if self._plan is None:
            self._games={g.stream_id:g for g in plan.games}
            self._locations={g.stream_id:(i,j)for i,iteration in enumerate(plan.iterations)
                for j,batch in enumerate(iteration)for g in batch.games}
        self._plan=plan

    def _actor(self,snapshot,store):
        key=snapshot['archive_sha256']
        if snapshot['model_sha256']!=runs._sha(store.plan.model.metadata()):
            raise ValueError('original snapshot model configuration differs')
        if key not in self.factories:self.factories[key]=_factory(self.prepared,snapshot,store)
        factory=self.factories[key]
        metadata=json.loads(factory.metadata)
        if factory.config!=store.plan.model or metadata['checkpoint_sha256']!=snapshot['checkpoint_sha256']:
            raise ValueError('cached immutable actor checkpoint differs')
        return factory

    def prepare_batch(self,plan,batch,current,assignments,store):
        self.check_plan(plan)
        if store.plan!=plan or [a['stream_id']for a in assignments]!=[g.stream_id for g in batch.games]:
            raise ValueError('original ordered assignments and run required')
        locations={self._locations.get(g.stream_id)for g in batch.games}
        if len(locations)!=1 or None in locations:raise ValueError('one original batch and its pending suffix required')
        iteration,index=next(iter(locations));full=plan.iterations[iteration][index]
        if any(self._games[g.stream_id]!=g for g in batch.games):raise ValueError('original declared game differs')
        positions=[full.games.index(g)for g in batch.games]
        if positions!=sorted(set(positions)):raise ValueError('original game dispatch order required')
        roots=tuple(full.gender_root_ids[p]for p in positions)if full.gender_seed is not None else ()
        if (batch.constructor_seed!=full.constructor_seed or batch.sampling_seed!=full.sampling_seed
                or batch.gender_seed!=full.gender_seed or batch.gender_domain!=full.gender_domain
                or batch.gender_root_ids!=roots):
            raise ValueError('original constructor/action/gender recipe required')
        required={current['archive_sha256'],*(a['opponent']['archive_sha256']for a in assignments)}
        self.factories={k:v for k,v in self.factories.items()if k in required}
        learner=self._actor(current,store);pairs=[];actor_bindings=[]
        for row in assignments:
            side=row['learner_side']
            if type(side)is not int or side not in(0,1):raise ValueError('original learner seat required')
            opponent=self._actor(row['opponent'],store)
            pair={0:learner,1:opponent}if side==0 else{0:opponent,1:learner}
            actual=[collection.actor_binding(pair[owner])for owner in(0,1)]
            if actual!=_expected_actors(current,row):raise ValueError('actual factories differ from original assignment')
            pairs.append((pair,side));actor_bindings.append(actual)
        return dict(plan=plan,batch=batch,iteration=iteration,index=index,original=full,
            pairs=tuple(pairs),actor_bindings=tuple(actor_bindings),genders=_genders(self.prepared,batch))

    def _one(self,prepared,row,check_progress,cancel):
        game=prepared['batch'].games[row];factories,side=prepared['pairs'][row]
        start=packed_start(self.prepared,game,prepared['genders'][row]);started=time.monotonic()
        def check():
            if cancel.is_set():raise RuntimeError('owned Python collection cancelled')
            check_progress()
        env=collection.OwningEnv(self.prepared.public_config)
        try:
            result=collection.collect_game(env,collection.Game(game.stream_id,game.battle_id,game.seed,
                prepared['batch'].sampling_seed),factories=factories,max_requests=prepared['plan'].max_requests,
                collect_owners=(side,),start_override=start,chance_seed=chance_seed(prepared['batch'],game),
                check_progress=check,enable_v5=True)
        finally:env.close()
        return result,time.monotonic()-started

    def collect(self,prepared,store,on_end,check_progress,*,on_start):
        self.check_plan(prepared['plan']);check_progress()
        cancel=threading.Event();active={};cursor=0
        executor=ThreadPoolExecutor(max_workers=self.workers,thread_name_prefix='showdown-owning')
        try:
            while cursor<len(prepared['batch'].games)or active:
                while cursor<len(prepared['batch'].games)and len(active)<self.workers:
                    check_progress();game=prepared['batch'].games[cursor]
                    on_start(game.stream_id)
                    active[executor.submit(self._one,prepared,cursor,check_progress,cancel)]=game
                    cursor+=1
                done,_=wait(active,timeout=.25,return_when=FIRST_COMPLETED)
                if not done:check_progress();continue
                for future in sorted(done,key=lambda f:prepared['batch'].games.index(active[f])):
                    game=active.pop(future);result,duration=future.result()
                    expected=collection.Game(game.stream_id,game.battle_id,game.seed,prepared['batch'].sampling_seed)
                    if (type(result)is not collection.GameCapture or result.game!=expected
                            or result.status not in('COMPLETE','CAPPED','REFUSED','FAILED')
                            or type(result.frames)is not int or not 0<=result.frames<=prepared['plan'].max_requests):
                        raise ValueError('actual original game capture required')
                    row=prepared['batch'].games.index(game);_,side=prepared['pairs'][row]
                    bindings=prepared['actor_bindings'][row]
                    if list(result.source.get('actor_bindings',()))!=bindings:
                        raise ValueError('captured factory identity differs from actual assigned actors')
                    for captured in result.decisions:
                        address=captured.row.observation.address;meta=captured.row.step.metadata.get('v5_capture',{})
                        if (address.stream!=game.stream_id or address.owner!=side
                                or meta.get('actor_binding')!=bindings[side]
                                or meta.get('opponent_actor_binding')!=bindings[1-side]):
                            raise ValueError('captured behavior does not belong to the assigned learner')
                    part=None
                    if result.decisions:
                        (store.path/'python-games').mkdir(exist_ok=True)
                        relative='python-games/'+str(game.stream_id)+'-'+uuid.uuid4().hex
                        _write_game_cache(store.path/relative,result.decisions,store.plan)
                        part=dict(path=relative,sha256=runs._file_sha(store.path/relative/'metadata.json'))
                    facts=dict(stream_id=game.stream_id,status=result.status,duration_seconds=duration,
                        iteration=prepared['iteration'],batch=prepared['index'],requests=result.frames,reason=result.reason)
                    document=dict(schema='pokezero.python-game-publication.v5',game=game.__dict__,
                        facts=facts,cache=part,source=result.source,auxiliary_refusals=result.auxiliary_refusals)
                    ref=store.artifact('python-games/'+str(game.stream_id)+'-'+uuid.uuid4().hex+'.json',
                        runs._json(document).encode(),kind='PYTHON_GAME_PUBLISHED',stream_id=game.stream_id,
                        iteration=prepared['iteration'],batch=prepared['index'])
                    on_end(game.stream_id,result.status,duration_seconds=duration,requests=result.frames,
                        reason=result.reason,python_publication=ref)
            return self._cache(store,prepared['iteration'],prepared['index'],prepared['original'])
        finally:
            cancel.set();executor.shutdown(wait=True,cancel_futures=True)

    def _publications(self,store,*,exclude=(),**fields):
        result={}
        if self._games is None:self._games={g.stream_id:g for g in store.plan.games}
        for ref in store.matching('PYTHON_GAME_PUBLISHED',**fields):
            if ref['stream_id']in exclude:continue
            document=json.loads(store.resolve(ref['path'],ref['sha256']).read_text())
            stream=ref['stream_id']
            if (stream in result or document.get('schema')!='pokezero.python-game-publication.v5'
                    or document['game']['stream_id']!=stream or document['facts']['stream_id']!=stream
                    or document['facts']['iteration']!=ref['iteration']or document['facts']['batch']!=ref['batch']):
                raise ValueError('original Python game publication changed or duplicated')
            game=self._games.get(stream)
            if game is None or json.loads(runs._json(game))!=document['game']:
                raise ValueError('published game is not the declared cell')
            facts=document['facts'];batch=store.plan.iterations[facts['iteration']][facts['batch']]
            if game not in batch.games or facts['status']not in('COMPLETE','CAPPED','REFUSED','FAILED'):
                raise ValueError('published game has no original terminal disposition/batch')
            expected_seed=chance_seed(batch,game);source=document['source']
            starts=store.matching('GAME_STARTED',stream_id=stream)
            if len(starts)!=1:raise ValueError('publication requires exactly one actual start')
            sequence=starts[0].get('batch_sequence')
            if type(sequence)is not int or not 0<=sequence<len(store.events):
                raise ValueError('actual start lacks its original batch binding')
            event=store.events[sequence];binding=event['payload']
            if (event['kind']!='BATCH_STARTED'or binding.get('deferred_starts')is not True
                    or binding['iteration']!=facts['iteration']or binding['batch']!=facts['batch']
                    or binding['streams'].count(stream)!=1):
                raise ValueError('actual start/batch identity differs')
            assigned=[a for a in binding['assignments']if a['stream_id']==stream]
            if len(assigned)!=1 or 'actor'not in binding:raise ValueError('original learner/actor assignment missing')
            actors=_expected_actors(binding['actor'],assigned[0]);owner=assigned[0]['learner_side']
            if source.get('actor_bindings')!=actors:raise ValueError('published actual actors differ from original assignment')
            if (document['cache']is not None or facts['status']=='COMPLETE'or 'chance_seed'in source):
                if source.get('chance_seed')!=expected_seed:raise ValueError('published constructor chance address differs')
            if facts['status']=='COMPLETE'and source.get('terminal',{}).get('winner')not in(0,1,2):
                raise ValueError('complete publication requires its literal terminal source')
            if document['cache']is not None:
                part=document['cache'];folder=store.resolve(part['path'],part['sha256'])
                meta=json.loads((folder/'metadata.json').read_text())
                addresses=meta['v5_rows']['addresses'];outer=meta['base_json']['step_metadata']
                steps=[s.get('v5_capture',{})for s in outer]
                frames=meta['v5_rows']['source_frame_sha256']
                if len(addresses)!=meta['example_count']or len(steps)!=len(addresses)or len(frames)!=len(addresses):
                    raise ValueError('cached original row denominator differs')
                static=meta['v5_static']
                if (static['category_vocabulary_sha256']!=store.plan.model.category_vocabulary_sha256
                        or static['label_vocabulary_sha256']!=store.plan.model.label_vocabulary.sha256):
                    raise ValueError('cached original actor namespace differs')
                for address,step,frame_sha in zip(addresses,steps,frames):
                    if (type(address['stream'])is not int or address['stream']!=stream
                            or type(address['owner'])is not int or address['owner']!=owner
                            or type(address['frame'])is not int or not 0<=address['frame']<facts['requests']
                            or step.get('stream')!=stream or step.get('owner')!=owner or step.get('frame')!=address['frame']
                            or step.get('source_frame_sha256')!=frame_sha
                            or step.get('actor_binding')!=actors[owner]
                            or step.get('opponent_actor_binding')!=actors[1-owner]):
                        raise ValueError('cached row owner/frame/actual actor differs from original learner')
                checks={'battle_ids':[game.battle_id]*len(addresses),'seeds':[game.seed]*len(addresses),
                    'player_ids':['p'+str(owner+1)]*len(addresses),'turn_indices':[a['frame']for a in addresses],
                    'action_indices':[s['choice']for s in steps],
                    'action_probabilities':[s['behavior_probability']for s in steps],
                    'value_estimates':[s['value_estimate']for s in steps]}
                for name,expected in checks.items():
                    key='base__'+name;file=folder/(key+'.npy');description=meta['arrays'][key]
                    if runs._file_sha(file)!=description['sha256']:raise ValueError('cached behavior array bytes changed')
                    value=np.load(file,allow_pickle=False)
                    if not np.array_equal(value,np.asarray(expected,dtype=value.dtype)):
                        raise ValueError('cached behavior/game identity differs from actual row')
            result[stream]=(document,ref)
        return result

    def recover_game_ends(self,store):
        """Recover only already-published actual results, never rerun a game."""
        ended={r['stream_id']for r in store.matching('GAME_END')}
        started={r['stream_id']for r in store.matching('GAME_STARTED')}
        for stream,(document,ref)in self._publications(store,exclude=ended).items():
            if stream not in started:raise ValueError('Python publication has no original actual start')
            if stream not in ended:
                store.emit('GAME_END',**document['facts'],python_publication=ref,recovered_from_publication=True)

    def _cache(self,store,iteration,index,batch):
        from gpu_engine.v5_device_spool import merge_game_parts
        parts=[];publications=self._publications(store,iteration=iteration,batch=index)
        for row,game in enumerate(batch.games):
            if game.stream_id not in publications:continue
            document,_=publications[game.stream_id]
            if (document['facts']['iteration'],document['facts']['batch'])!=(iteration,index):
                raise ValueError('original Python batch location differs')
            part=document['cache']
            if part is not None:parts.append((row,store.resolve(part['path'],part['sha256']),part['sha256']))
        if not parts:return None
        (store.path/'caches').mkdir(exist_ok=True)
        relative='caches/python-'+uuid.uuid4().hex
        merge_game_parts(store.path/relative,parts,tuple(g.stream_id for g in batch.games))
        return relative,runs._file_sha(store.path/relative/'metadata.json')

    def recover_caches(self,store,statuses):
        """Finish only the missing merge of a completely closed Python batch."""
        declared={(p['iteration'],p['batch'])for p in store.matching('BATCH_STARTED')if p.get('deferred_starts')is True}
        for iteration,index in sorted(declared):
            batch=store.plan.iterations[iteration][index]
            if (store.matching('CACHE',iteration=iteration,batch=index)
                    or any(statuses[g.stream_id]in('NOT_ATTEMPTED','RUNNING')for g in batch.games)):continue
            result=self._cache(store,iteration,index,batch)
            if result is not None:store.emit('CACHE',iteration=iteration,batch=index,path=result[0],sha256=result[1],recovered_python_merge=True)

    def close(self):
        self.factories.clear()
