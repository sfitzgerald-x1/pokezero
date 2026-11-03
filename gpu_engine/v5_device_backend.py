"""Explicit original v5 run backend using device public rollout, never FULL.

Import/preparation creates no actor, game or training update. The original run
owns declaration attempts, historical assignments, trainer and resume rules.
Only a supplied fixed batch is collected, with no replacement or refill.
"""
from dataclasses import dataclass
import hashlib
import json
import time
import uuid

import jax
import numpy as np
from pokezero import v5_run as runs,v5_data
from . import transition,engine_device_observation as encoder,source_provenance,v5_model
from . import v5_device_rollout as rollout
from .v5_device_spool import Spool


@dataclass(frozen=True)
class Prepared:
    catalog:object
    source:object
    engine_tables:object
    observation_tables:object
    pool_sha256:str
    metadata_json:str
    source_binding_json:str
    category_sha256:str
    category_size:int

    @property
    def source_sha256(self):return hashlib.sha256(self.source_binding_json.encode()).hexdigest()


def prepare(catalog,source,pool,*,source_commit):
    """Compile static device tables once from the same original public source."""
    receipt=source_provenance._pool(pool,rehash=True);metadata=pool.metadata()
    if catalog.source.source_hash!=source.metadata.source_hash:raise ValueError('same original public source required')
    binding=dict(source_commit=source_commit,public_source_sha256=source.metadata.source_hash,
        pool_metadata_sha256=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    from pokezero.v5_phasec import category_binding
    return Prepared(catalog,source,transition.tables_from_metadata(metadata),encoder.compile_tables(metadata,catalog=catalog),
        receipt['pool_sha256'],json.dumps(metadata,sort_keys=True),json.dumps(binding,sort_keys=True,separators=(',',':')),
        category_binding(catalog.vocabulary),catalog.vocabulary.size)


class RuntimeBackend:
    """Device-mode backend for the unchanged v5_run.run callback contract."""
    def __init__(self,prepared,pool,*,gender_policy=None,chunk_steps=8,teacher=None,measure_runtime=False,cache_workers=0):
        if type(prepared)is not Prepared or type(chunk_steps)is not int or chunk_steps<=0:
            raise ValueError('prepared device runtime and finite chunk required')
        if type(measure_runtime)is not bool:raise ValueError('explicit runtime timing choice required')
        if type(cache_workers)is not int or cache_workers<0:raise ValueError('explicit nonnegative CPU cache worker count required')
        self.cache_workers=cache_workers;self._cache_pool=None;self._cache_merger=None
        self.measure_runtime=measure_runtime;self._initializer_executables={}
        self.teacher=teacher
        self.prepared=prepared;self.pool=pool;self.gender_policy=gender_policy;self.chunk_steps=chunk_steps
        self.actors={};self._plan=None;self.labels=None;self.declaration_sha256=None;self.model_sha256=None
        self.initialize=jax.jit(lambda state:transition.initialize_with_public(state,prepared.engine_tables,
            belief_tables=prepared.observation_tables.belief))

    def close(self):
        if self._cache_pool is not None:self._cache_pool.close()
        if self._cache_merger is not None:self._cache_merger.shutdown(wait=True,cancel_futures=True)

    def require_teacher(self):
        from .v5_device_teacher import Provider
        if type(self.teacher)is not Provider or self.teacher.prepared is not self.prepared:
            raise runs.MissingDependency('device D requires a genuine configured public-world teacher; no FULL fallback')

    def check_plan(self,plan):
        if plan.model.arm=='D':
            self.require_teacher()
            if self.teacher.plan!=plan:raise ValueError('same original D teacher Plan required')
        elif self.teacher is not None:raise ValueError('D teacher cannot label A/B treatment')
        if self.prepared.pool_sha256!=plan.pool_sha256:raise ValueError('original declared constructor pool differs')
        if self._plan is not None and plan is not self._plan and plan!=self._plan:
            raise ValueError('one original worker plan required')
        if self._plan is None:
            from pokezero.v5_contract import source_variant_census
            if (plan.model.category_vocabulary_sha256!=self.prepared.category_sha256
                    or plan.model.category_vocab_size!=self.prepared.category_size
                    or source_variant_census(self.prepared.source)['source_rows_sha256']!=plan.model.label_vocabulary.source_binding_sha256):
                raise ValueError('same original actor category/full source-label namespace required')
            self.labels=rollout.compile_labels(json.loads(self.prepared.metadata_json),plan.model.label_vocabulary)
            self.declaration_sha256=runs._sha(plan.manifest());self.model_sha256=runs._sha(plan.model.metadata());self._plan=plan

    def _actor(self,snapshot,store):
        digest=snapshot['archive_sha256'];checkpoint=snapshot['checkpoint_sha256'];plan=store.plan
        if snapshot['model_sha256']!=self.model_sha256:raise ValueError('original historical model configuration differs')
        if digest not in self.actors:
            path=store.resolve(snapshot['archive'],digest)
            params,config,metadata=v5_model.load(path,expected_archive_sha256=digest,
                expected_checkpoint_sha256=checkpoint,expected_category_vocabulary_sha256=plan.model.category_vocabulary_sha256,
                expected_label_vocabulary_sha256=plan.model.label_vocabulary.sha256,enable_v5=True)
            if config!=plan.model:raise ValueError('loaded original actor configuration differs')
            self.actors[digest]=(params,metadata)
        params,metadata=self.actors[digest]
        if metadata['checkpoint_sha256']!=checkpoint:raise ValueError('cached actor checkpoint provenance differs')
        return params

    def prepare_batch(self,plan,batch,current,assignments,store):
        self.check_plan(plan)
        if store.plan!=plan:raise ValueError('same original run store plan required')
        if [a['stream_id'] for a in assignments]!=[g.stream_id for g in batch.games]:
            raise ValueError('declared assignment order differs')
        required=tuple(dict.fromkeys([current['archive_sha256'],*(a['opponent']['archive_sha256'] for a in assignments)]))
        self.actors={key:value for key,value in self.actors.items() if key in required}
        self._actor(current,store)
        for assignment in assignments:self._actor(assignment['opponent'],store)
        bindings=rollout.Bindings(plan.model.category_vocabulary_sha256,plan.model.label_vocabulary,
            self.prepared.source_sha256,(current['archive_sha256'],)*2,(True,True),self.declaration_sha256)
        # Original planner opponents are Snapshot dataclasses. Use exactly
        # the original run-journal JSON representation, retaining all fields
        # and order, before the bank's plain immutable JSON binding.
        recorded_assignments=json.loads(runs._json(assignments))
        bank=rollout.actor_bank_from_assignments(bindings,required,current_archive=current['archive_sha256'],assignments=recorded_assignments)
        options={} if plan.model.arm!='D' else dict(retain_teacher_roots=True)
        params=tuple(self.actors[key][0] for key in required)
        if self.teacher is not None:self.teacher.bind(bank,params)
        collector=rollout.make_collector(self.prepared.engine_tables,self.prepared.observation_tables,plan.model,self.labels,bindings,
            sampling_seed=batch.sampling_seed,chunk_steps=self.chunk_steps,max_requests=plan.max_requests,actor_bank=bank,**options)
        # One callable per prepared assignment bank; reused for every device
        # chunk. Group topology/shape changes may require a new compilation.
        return dict(plan=plan,batch=batch,bindings=bindings,bank=bank,collector=collector,
            params=params)

    def _construct(self,batch):
        indices=np.asarray([game.team_indices for game in batch.games],np.int32)
        raw=transition.state_from_pool(self.pool.arrays,indices,seed=batch.constructor_seed)
        from .engine_request_source import _gender
        gender_initialization=None
        if batch.gender_seed is not None:
            if self.gender_policy is None:raise ValueError('original declared gender policy required')
            gender_initialization=dict(policy=self.gender_policy,seed=batch.gender_seed,domain=batch.gender_domain,
                root_ids=np.asarray(batch.gender_root_ids,np.uint32))
        # Original pure constructor-field helper, including retained explicit
        # genders and its source binding; no lifecycle or captured writer.
        raw=_gender(raw,self.pool,indices,gender_initialization)
        truth=rollout.constructor_truth(raw,self.pool,indices,source=self.prepared.source,
            compiled_source=self.prepared.catalog.source,vocabulary=self._plan.model.label_vocabulary)
        if self.measure_runtime:
            signature=tuple((x.shape,str(x.dtype))for x in jax.tree.leaves(raw))
            if signature not in self._initializer_executables:
                started=time.perf_counter();lowered=self.initialize.lower(raw)
                self._phase['initializer_lower_seconds']+=time.perf_counter()-started
                started=time.perf_counter();self._initializer_executables[signature]=lowered.compile()
                self._phase['initializer_compile_seconds']+=time.perf_counter()-started
                self._phase['initializer_compile_calls']+=1
            initialized=jax.block_until_ready(self._initializer_executables[signature](raw))
        else:initialized=self.initialize(raw)
        return rollout.initialize(initialized,np.asarray([g.stream_id for g in batch.games],np.int32),truth=truth)

    def collect(self,prepared,store,on_end,check_progress):
        plan=prepared['plan'];batch=prepared['batch'];self.check_plan(plan);check_progress()
        started=time.monotonic();wall_started=time.perf_counter()
        self._phase={key:0. for key in ('initializer_lower_seconds','initializer_compile_seconds','collector_lower_seconds','collector_compile_seconds',
            'constructor_wall_seconds','kernel_blocked_seconds','teacher_wall_seconds','host_transfer_wall_seconds','spool_write_wall_seconds','game_cache_wall_seconds','cache_merge_wall_seconds')}
        self._phase.update(initializer_compile_calls=0,collector_compile_calls=0)
        construction=time.perf_counter();carry=self._construct(batch);closed=set()
        self._phase['constructor_wall_seconds']=time.perf_counter()-construction
        execute=prepared['collector']
        if self.measure_runtime:
            from .v5_device_timing import Counts
            counts=Counts(len(batch.games))
            if 'compiled_collector'not in prepared:
                clock=time.perf_counter();lowered=execute.lower(carry,prepared['params'])
                self._phase['collector_lower_seconds']=time.perf_counter()-clock
                clock=time.perf_counter();prepared['compiled_collector']=lowered.compile()
                self._phase['collector_compile_seconds']=time.perf_counter()-clock;self._phase['collector_compile_calls']=1
            execute=prepared['compiled_collector']
        bank=prepared['bank'];binding=dict(schema='pokezero.device_actor_bank.v1',source_binding=json.loads(self.prepared.source_binding_json),
            declaration_sha256=prepared['bindings'].declaration_sha256,assignment_sha256=bank.assignment_sha256,
            actor_archives=bank.actor_archives,streams=bank.streams,actor_indices=bank.actor_indices,learner_mask=bank.learner_mask,
            games=[dict(stream=g.stream_id,battle_id=g.battle_id,seed=g.seed,team_indices=g.team_indices)for g in batch.games],
            constructor_seed=batch.constructor_seed,sampling_seed=batch.sampling_seed,gender_seed=batch.gender_seed,
            gender_domain=batch.gender_domain,gender_root_ids=batch.gender_root_ids)
        receipt=store.artifact(f'rollouts/device-bank-{uuid.uuid4().hex}.json',runs._json(binding).encode(),kind='DEVICE_ACTOR_BANK')
        options={} if self.teacher is None else dict(teacher_recipe=self.teacher.recipe)
        spool=Spool(store,bank,bank_sha256=receipt['sha256'],**options)
        assembly=None
        if self.cache_workers:
            from concurrent.futures import ThreadPoolExecutor
            from .v5_cache_pipeline import CpuWorkers,BankAssembly
            if self._cache_pool is None:
                self._cache_pool=CpuWorkers(self.cache_workers)
                self._cache_merger=ThreadPoolExecutor(max_workers=1,thread_name_prefix='cache-merge')
            assembly=BankAssembly(self._cache_pool,spool,bindings=prepared['bindings'],loss_config=plan.loss,discount=plan.discount)
        warm_loop_started=time.perf_counter()
        for _ in range((plan.max_requests+self.chunk_steps-1)//self.chunk_steps):
            check_progress()
            if assembly is not None:assembly.check()
            clock=time.perf_counter();device=execute(carry,prepared['params'])
            if self.measure_runtime:device=jax.block_until_ready(device)
            self._phase['kernel_blocked_seconds']+=time.perf_counter()-clock;carry=device.carry
            # Keep the live world/public carry on device. The original batch
            # adapter needs raw Frames plus literal terminal fields, not a
            # duplicate hidden world/constructor truth on the host each chunk.
            clock=time.perf_counter()
            host=rollout.boundary(device._replace(carry=carry._replace(state=None,public=None,truth=None),roots=None))
            self._phase['host_transfer_wall_seconds']+=time.perf_counter()-clock
            if self.measure_runtime:counts.add(host.frames.requested)
            clock=time.perf_counter()
            labels=None if self.teacher is None else self.teacher.label_chunk(device.roots,host.frames,bank,prepared['params'])
            self._phase['teacher_wall_seconds']+=time.perf_counter()-clock
            del device
            clock=time.perf_counter()
            if self.teacher is None:spool.append(host)
            else:spool.append(host,teacher_records=labels)
            self._phase['spool_write_wall_seconds']+=time.perf_counter()-clock
            for row,game in enumerate(batch.games):
                status=int(host.carry.status[row])
                if status!=rollout.RUNNING and game.stream_id not in closed:
                    on_end(game.stream_id,rollout.STATUS[status],duration_seconds=time.monotonic()-started,
                        requests=int(host.carry.frames[row]),terminal_frame=int(host.carry.terminal_frame[row]),
                        terminal_winner=int(host.carry.winner[row]) if status==rollout.COMPLETE else None,
                        actor_bank_sha256=receipt['sha256'])
                    closed.add(game.stream_id)
                    clock=time.perf_counter()
                    if assembly is None:
                        spool.finalize_game(row,game,bindings=prepared['bindings'],loss_config=plan.loss,discount=plan.discount)
                    else:assembly.submit(row,game)
                    self._phase['game_cache_wall_seconds']+=time.perf_counter()-clock
            if len(closed)==len(batch.games):break
        if len(closed)!=len(batch.games):raise ValueError('finite device collector left a declared game unclosed')
        if assembly is not None:
            # Freeze this bank's timing and terminal state before the owner
            # starts the next bank. Workers never touch the run journal.
            phases=self._phase;status=np.array(host.carry.status,copy=True)
            kernel_end=time.perf_counter();relative=f'caches/cache-{uuid.uuid4().hex}'
            (store.path/'caches').mkdir(exist_ok=True)
            def finish():
                timing=assembly.finish(store.path/relative)
                timing['finished']=time.perf_counter();return timing
            def publish(timing):
                phases['game_cache_wall_seconds']=timing['worker_seconds']
                phases['cache_merge_wall_seconds']=timing['merge_seconds']
                total=timing['finished']-wall_started
                if self.measure_runtime:
                    report=counts.report(status,kernel_seconds=phases['kernel_blocked_seconds'],collection_seconds=total)
                    warm=timing['finished']-warm_loop_started
                    store.emit('DEVICE_RUNTIME_TIMING',**report,phases=phases,bank_streams=list(bank.streams),
                        warm_loop_including_cache_io_seconds=warm,
                        completed_per_warm_loop_wall_second=None if warm<=0 else report['complete']/warm,
                        cache_pipeline=dict(workers=self.cache_workers,maximum_pending_banks=2,
                            tail_wait_seconds=timing['wait_seconds'],kernel_bank_wall_seconds=kernel_end-wall_started,
                            assembly_worker_seconds=timing['worker_seconds'],
                            worker_dispatch_inclusive_seconds=timing['dispatch_seconds'],bank_finish_monotonic=timing['finished']),
                        timing_scope='Blocked actual kernels; CPU game_cache is aggregate worker busy time, not wall. Merge and tail writes may overlap next bank; bank durations must not be summed into iteration wall.',
                        actor_load_scope='verified original archive loading is batch preparation before collect; not kernel wall')
                if timing['metadata']is None:return None
                return relative,runs._file_sha(store.path/relative/'metadata.json')
            return runs.PendingCache(self._cache_merger.submit(finish),publish)
        result=None
        if spool.parts:
            (store.path/'caches').mkdir(exist_ok=True)
            relative=f'caches/cache-{uuid.uuid4().hex}';clock=time.perf_counter();spool.merge(store.path/relative)
            self._phase['cache_merge_wall_seconds']=time.perf_counter()-clock
            result=relative,runs._file_sha(store.path/relative/'metadata.json')
        if self.measure_runtime:
            report=counts.report(host.carry.status,kernel_seconds=self._phase['kernel_blocked_seconds'],collection_seconds=time.perf_counter()-wall_started)
            warm_loop_seconds=time.perf_counter()-warm_loop_started
            store.emit('DEVICE_RUNTIME_TIMING',**report,phases=self._phase,bank_streams=list(bank.streams),
                warm_loop_including_cache_io_seconds=warm_loop_seconds,
                completed_per_warm_loop_wall_second=None if warm_loop_seconds<=0 else report['complete']/warm_loop_seconds,
                timing_scope='explicit lower/compile before actual chunks; blocked actual kernel calls, no warmup games; constructor wall includes initializer lower/compile',
                actor_load_scope='verified original archive loading is batch preparation before collect; not kernel wall')
        return result
