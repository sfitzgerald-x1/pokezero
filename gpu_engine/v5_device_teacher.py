"""Opt-in actual device Arm-D chunk consumer and original-cache target bridge.

Only requested learner PRE roots are quarter-selected by the original Recipe.
Actor trees are the verified ActorBank archives. Public source sampling and all
reply/leaf arithmetic are the existing device teacher, never a private critic.
Raw logits and every selected/refused address survive in the disk spool.
"""
from dataclasses import asdict
import hashlib
import json
import time
import jax
import numpy as np
from pokezero import v5_contract as c, v5_gumbel as original
from . import device_public_worlds as worlds, device_public_world_state as states
from . import device_public_teacher as teacher, v5_model

SCHEMA='pokezero.device_teacher_records.v1'
INPUTS=('categorical_ids','numeric_features','token_type_ids','attention_mask')


def binding(frames,bank,step,row,owner):
    """Bind the exact original PRE bytes, including nonfinite refused inputs."""
    if frames.raw_policy_logits is None:raise ValueError('actual raw root logits required')
    h=hashlib.sha256()
    for name in INPUTS+('legal_action_mask','raw_policy_logits','policy_valid','errors'):
        a=np.asarray(getattr(frames,name)[step,row,owner])
        h.update(json.dumps((name,a.dtype.str,a.shape),separators=(',',':')).encode());h.update(a.tobytes())
    return dict(schema=SCHEMA,address=dict(stream=int(frames.streams[step,row]),frame=int(frames.frames[step,row]),owner=owner),
        source_sha256=bank.bindings.source_sha256,declaration_sha256=bank.bindings.declaration_sha256,
        assignment_sha256=bank.assignment_sha256,
        actor_archive_sha256=bank.actor_archives[int(frames.actor_indices[step,row,owner])],
        opponent_archive_sha256=bank.actor_archives[int(frames.actor_indices[step,row,1-owner])],
        input_sha256=h.hexdigest())


def _requests(frames,bank):
    from .v5_device_rollout import ActorBank
    if type(bank)is not ActorBank:raise ValueError('original actor bank required')
    if frames.raw_policy_logits is None or frames.raw_policy_logits.shape!=frames.requested.shape+(9,) or frames.raw_policy_logits.dtype!=np.float32:
        raise ValueError('typed original raw root logits geometry required')
    for name in ('requested','policy_valid','trainable'):
        value=np.asarray(getattr(frames,name))
        if value.shape!=frames.requested.shape or value.dtype!=np.bool_:raise ValueError('typed original learner/request masks required')
    actors=np.broadcast_to(np.asarray(bank.actor_indices,np.int32),frames.requested.shape)
    learners=np.broadcast_to(np.asarray(bank.learner_mask,bool),frames.requested.shape)
    if (not np.array_equal(frames.actor_indices,actors)or not np.array_equal(frames.learner_mask,learners)
            or not np.array_equal(frames.streams,np.broadcast_to(bank.streams,frames.streams.shape))
            or not np.array_equal(frames.trainable,frames.policy_valid&learners)):
        raise ValueError('same original actor-bank rows required')
    return np.argwhere(frames.requested&learners)


def validate_records(frames,bank,recipe,records):
    """No dropped/refilled/renormalized slots, and no foreign PRE/actor labels."""
    if type(recipe)is not original.Recipe:raise ValueError('original declared Recipe required')
    expected={}
    for step,row,owner in _requests(frames,bank):
        owner=int(owner);b=binding(frames,bank,int(step),int(row),owner);key=tuple(b['address'][k]for k in ('stream','frame','owner'))
        if key in expected:raise ValueError('duplicate actual requested decision')
        expected[key]=(b,bool(frames.trainable[step,row,owner]),tuple(map(bool,frames.legal_action_mask[step,row,owner])))
    result={}
    for record in records:
        if type(record)is not dict or type(record.get('address'))is not dict:raise ValueError('literal teacher record required')
        a=record['address']
        if set(a)!=set(('stream','frame','owner'))or any(type(a.get(k))is not int for k in a):raise ValueError('typed exact teacher address required')
        key=tuple(a[k]for k in ('stream','frame','owner'))
        if key not in expected or key in result:raise ValueError('foreign/duplicate teacher address')
        b,valid,legal=expected[key]
        if any(record.get(k)!=v for k,v in b.items())or type(record.get('recipe_seed'))is not int or record.get('recipe_seed')!=recipe.seed:
            raise ValueError('teacher original PRE/actor/source binding differs')
        selected=recipe.selected(c.DecisionAddress(*key));status=record.get('status');improvement=record.get('improvement');refusal=record.get('refusal')
        if not selected:
            if status!='UNSELECTED'or improvement is not None or refusal is not None:raise ValueError('unselected address was priced')
        elif status=='REFUSED':
            if improvement is not None or type(refusal)is not str or not refusal:raise ValueError('literal selected refusal required')
        elif status=='LABELED':
            if not valid or refusal is not None or type(improvement)is not dict:raise ValueError('valid selected original improvement required')
            target=c.RootImprovement(**improvement);target.validate(legal)
            if target.worlds!=2:raise ValueError('both declared source worlds required; no survivor pricing')
        else:raise ValueError('selected address must be labeled or refused')
        result[key]=record
    if set(result)!=set(expected):raise ValueError('missing requested learner teacher slot')
    return result


class Provider:
    """Construct once from genuine Prepared+Plan, before any real constructor."""
    def __init__(self,prepared,plan,*,batch_joint=True,frame_batch=False,chunk_batch=False,vectorized=False,batched_prep=False):
        from .v5_device_backend import Prepared
        from pokezero.v5_run import Plan,_sha
        if type(prepared)is not Prepared or type(plan)is not Plan or plan.model.arm!='D':
            raise ValueError('actual prepared device D Plan required')
        if type(batch_joint)is not bool:raise ValueError('explicit joint batching bool required')
        if type(frame_batch)is not bool or frame_batch and not batch_joint:
            raise ValueError('frame batching requires the original mapped joint executor')
        if type(chunk_batch)is not bool or chunk_batch and not frame_batch:
            raise ValueError('chunk batching requires the explicit frame teacher')
        if type(vectorized)is not bool or vectorized and not frame_batch:raise ValueError('vectorized pricing requires frame batching')
        if type(batched_prep)is not bool or batched_prep and not vectorized:raise ValueError('batched preparation requires vectorized D3 pricing')
        self.vectorized=vectorized;self.batched_prep=batched_prep
        self.chunk_batch=chunk_batch
        if prepared.pool_sha256!=plan.pool_sha256 or prepared.category_sha256!=plan.model.category_vocabulary_sha256 or prepared.category_size!=plan.model.category_vocab_size:
            raise ValueError('same declared device source/pool/category required')
        self.prepared=prepared;self.plan=plan;self.recipe=original.Recipe(plan.selection_seed)
        self.declaration_sha256=_sha(plan.manifest())
        self.source=worlds.compile_source(prepared.source,json.loads(prepared.metadata_json),
            expected_source_hash=prepared.source.metadata.source_hash,
            expected_source_rows_sha256=plan.model.label_vocabulary.source_binding_sha256)
        self.materializer=states.compile_tables(self.source,json.loads(prepared.metadata_json))
        self.executor=teacher.make_executor(prepared.engine_tables,prepared.observation_tables,
            enable_device_teacher=True,batch_joint=batch_joint)
        self.batch_joint=batch_joint;self.forwards={};self.joint_forwards={}
        self.frame_batch=frame_batch;self.frame_counts={};self.batch_timings=[]
        if frame_batch:
            from .device_public_teacher_batch import FramePricer
            self.frame_pricer=FramePricer(self.executor,plan.model,actor_capacity=plan.historical_window+1,
                **(dict(capacity=4096,vectorized=True,host_prep=batched_prep)if vectorized else{}))

    def reset_metrics(self):
        self.frame_counts={};self.batch_timings=[];self.root_prepare_seconds={}

    def record_metrics(self,records):
        if not hasattr(self,'frame_counts'):self.reset_metrics()
        for record in records:
            frame=record['address']['frame'];counts=self.frame_counts.setdefault(frame,dict(requested=0,selected=0,labeled=0,refused=0,unselected=0))
            counts['requested']+=1;status=record['status']
            if status=='UNSELECTED':counts['unselected']+=1
            else:
                counts['selected']+=1
                counts['labeled'if status=='LABELED'else'refused']+=1

    def metrics(self):
        from .device_public_teacher_batch import FRAME_WORK_KEYS
        report=dict(strategy=('CHUNK_BATCH'if getattr(self,'chunk_batch',False)else'FRAME_BATCH')if getattr(self,'frame_batch',False)else'SERIAL_ROOTS',
            by_frame={str(frame):dict(value,refused_share_selected=None if not value['selected']else value['refused']/value['selected'])for frame,value in sorted(self.frame_counts.items())},
            batch_timing_totals={key:sum(x.get(key,0)for x in self.batch_timings)for key in ('roots','world_prepare_seconds','batch_pack_and_device_seconds','device_invocations','lower_seconds','compile_seconds','device_seconds')+FRAME_WORK_KEYS})
        if getattr(self,'vectorized',False):
            report['strategy']='D3_VECTORIZED_FRAME_QUEUE'
            report['frame_calls']=[dict(x)for x in self.batch_timings]
            report['root_prepare_by_frame']={str(k):v for k,v in getattr(self,'root_prepare_seconds',{}).items()}
        if getattr(self,'batched_prep',False):
            report['batched_preparation']=True
            report['root_prepare_timing_scope']='Per-root planning wall plus equal-per-pending-root share of chunk detach wall; cold detach/readout compilation is included, not isolated.'
        return report

    def bind(self,bank,params):
        if (bank.bindings.source_sha256!=self.prepared.source_sha256
                or bank.bindings.category_sha256!=self.plan.model.category_vocabulary_sha256
                or bank.bindings.label_vocabulary!=self.plan.model.label_vocabulary
                or bank.bindings.declaration_sha256!=self.declaration_sha256
                or len(params)!=len(bank.actor_archives)):
            raise ValueError('same verified prepared ActorBank required')
        # New snapshot banks can evict old callbacks, just like the worker's
        # verified archive cache. No per-game model or parameter-tree copies.
        self.forwards={k:v for k,v in self.forwards.items()if k in bank.actor_archives}
        for digest,p in zip(bank.actor_archives,params):
            if digest not in self.forwards:
                self.forwards[digest]=(p,jax.jit(lambda inputs,p=p:v5_model.forward(p,inputs,self.plan.model,compute_dtype='bfloat16')))
            elif self.forwards[digest][0]is not p:raise ValueError('verified archive parameter identity changed')
        if getattr(self,'batch_joint',False):
            self.joint_forwards={k:v for k,v in self.joint_forwards.items()if k in bank.actor_archives}
            for digest in bank.actor_archives:
                if digest not in self.joint_forwards:
                    self.joint_forwards[digest]=jax.jit(jax.vmap(self.forwards[digest][1]))

    def label_chunk(self,device_roots,frames,bank,params):
        from .v5_device_rollout import RootContext
        if type(device_roots)is not RootContext:raise ValueError('actual retained PRE root contexts required')
        if getattr(self,'frame_batch',False):return self._label_chunk_batched(device_roots,frames,bank,params)
        self.bind(bank,params);records=[];contexts={}
        for step,row,owner in _requests(frames,bank):
            step,row,owner=map(int,(step,row,owner));base=binding(frames,bank,step,row,owner)
            record=dict(base,recipe_seed=self.recipe.seed,status='UNSELECTED',refusal=None,improvement=None,diagnostics=None)
            records.append(record);address=c.DecisionAddress(**base['address'])
            if not self.recipe.selected(address):continue
            record.update(status='REFUSED',refusal='invalid-actual-root-input')
            if not frames.trainable[step,row,owner]:continue
            try:
                # Slice the scan axis on device. detach_root subsequently
                # indexes ONLY own/private + safe root public knowledge before
                # host conversion. Never device_get this whole root context.
                if step not in contexts:contexts[step]=jax.tree.map(lambda x:x[step],device_roots)
                context=contexts[step]
                if int(context.streams[row])!=address.stream or int(context.frames[row])!=address.frame:
                    raise ValueError('PRE root context address differs')
                root=worlds.detach_root(context,self.source,row=row,owner=owner,enable_device_worlds=True)
                plans=worlds.plan_starts(root,self.source,self.recipe)
                inputs={name:getattr(frames,name)[step,row,owner][None,None]for name in INPUTS}
                inputs['history_mask']=np.ones((1,1),bool)
                options={}if not getattr(self,'batch_joint',False)else dict(
                    learner_batch_forward=self.joint_forwards[base['actor_archive_sha256']])
                result=teacher.price(plans,self.materializer,self.executor,self.recipe,
                    root_logits=tuple(map(float,frames.raw_policy_logits[step,row,owner])),
                    root_legal=tuple(map(bool,frames.legal_action_mask[step,row,owner])),root_inputs=inputs,
                    opponent_forward=self.forwards[base['opponent_archive_sha256']][1],
                    learner_forward=self.forwards[base['actor_archive_sha256']][1],enable_device_teacher=True,**options)
                record['diagnostics']=dict(world_reports=result.world_reports,missing_worlds=result.missing_worlds,
                    actual_teacher_calls=result.actual_teacher_calls,reply_input_approximation=result.reply_input_approximation)
                if result.status=='LABELED':
                    record.update(status='LABELED',refusal=None,improvement=asdict(result.reduction.improvement))
                elif result.status=='REFUSED':record['refusal']=result.reason or 'unknown-required-world-or-reply'
                else:raise ValueError('selected original recipe returned an unselected result')
            except (ValueError,RuntimeError,TypeError,KeyError,IndexError)as error:
                record.update(status='REFUSED',refusal=type(error).__name__+': '+str(error),improvement=None)
        validate_records(frames,bank,self.recipe,records)
        self.record_metrics(records)
        return records

    def _prepare_batched(self,device_roots,frames,bank):
        """The per-root preparation of _label_chunk_batched, with all roots of a scan step detached
        in ONE batched pass (worlds.detach_roots). Records, refusals, requests and their grouping
        order are the same as the per-root loop's."""
        from .device_public_teacher_batch import Request
        records=[];pending=[];contexts={};hosts={}
        for step,row,owner in _requests(frames,bank):
            step,row,owner=map(int,(step,row,owner));base=binding(frames,bank,step,row,owner)
            record=dict(base,recipe_seed=self.recipe.seed,status='UNSELECTED',refusal=None,improvement=None,diagnostics=None)
            records.append(record);address=c.DecisionAddress(**base['address'])
            if not self.recipe.selected(address):continue
            record.update(status='REFUSED',refusal='invalid-actual-root-input')
            if not frames.trainable[step,row,owner]:continue
            pending.append((record,step,row,owner,address))
        started=time.perf_counter();roots={}
        for step in sorted({x[1] for x in pending}):
            try:
                if step not in contexts:contexts[step]=jax.tree.map(lambda x:x[step],device_roots)
                context=contexts[step];hosts[step]=jax.device_get((context.streams,context.frames))
            except (ValueError,RuntimeError,TypeError,KeyError,IndexError)as error:
                hosts[step]=error
        batch={}
        for record,step,row,owner,address in pending:
            host=hosts[step]
            if isinstance(host,Exception):continue
            streams,frames_=host
            if not (0<=row<len(streams))or int(streams[row])!=address.stream or int(frames_[row])!=address.frame:continue
            batch.setdefault(step,[]).append((row,owner))
        for step,items in batch.items():
            # Rows, owners and addresses are checked above and every primitive is shape-uniform, so a
            # detach_roots failure is systemic (e.g. device memory), never one root's: it stops the
            # study instead of refusing the whole step.
            for (row,owner),root in zip(items,worlds.detach_roots(contexts[step],self.source,rows=[r for r,_ in items],
                    owners=[o for _,o in items],enable_device_worlds=True)):roots[(step,row,owner)]=root
        detach_seconds=time.perf_counter()-started;grouped={}
        for record,step,row,owner,address in pending:
            preparation_started=time.perf_counter()
            try:
                host=hosts[step]
                if isinstance(host,Exception):raise host
                streams,frames_=host
                if int(streams[row])!=address.stream or int(frames_[row])!=address.frame:
                    raise ValueError('PRE root context address differs')
                root=roots[(step,row,owner)]
                plans=worlds.plan_starts(root,self.source,self.recipe)
                inputs={name:getattr(frames,name)[step,row,owner][None,None]for name in INPUTS};inputs['history_mask']=np.ones((1,1),bool)
                request=Request(plans,tuple(map(float,frames.raw_policy_logits[step,row,owner])),
                    tuple(map(bool,frames.legal_action_mask[step,row,owner])),inputs,
                    int(frames.actor_indices[step,row,owner]),int(frames.actor_indices[step,row,1-owner]))
                grouped.setdefault(record['address']['frame'],[]).append((record,request))
            except (ValueError,RuntimeError,TypeError,KeyError,IndexError)as error:
                record.update(status='REFUSED',refusal=type(error).__name__+': '+str(error),improvement=None)
            finally:
                frame=record['address']['frame']
                self.root_prepare_seconds[frame]=self.root_prepare_seconds.get(frame,0.)+time.perf_counter()-preparation_started+detach_seconds/len(pending)
        return records,grouped

    def _label_chunk_batched(self,device_roots,frames,bank,params):
        from .device_public_teacher_batch import Request
        if getattr(self,'batched_prep',False):
            self.bind(bank,params)
            if not hasattr(self,'root_prepare_seconds'):self.root_prepare_seconds={}
            records,grouped=self._prepare_batched(device_roots,frames,bank)
            return self._price_groups(records,grouped,frames,bank,params)
        self.bind(bank,params);records=[];grouped={};contexts={}
        for step,row,owner in _requests(frames,bank):
            step,row,owner=map(int,(step,row,owner));base=binding(frames,bank,step,row,owner)
            record=dict(base,recipe_seed=self.recipe.seed,status='UNSELECTED',refusal=None,improvement=None,diagnostics=None)
            records.append(record);address=c.DecisionAddress(**base['address'])
            if not self.recipe.selected(address):continue
            record.update(status='REFUSED',refusal='invalid-actual-root-input')
            if not frames.trainable[step,row,owner]:continue
            preparation_started=time.perf_counter()
            try:
                if step not in contexts:contexts[step]=jax.tree.map(lambda x:x[step],device_roots)
                context=contexts[step]
                if int(context.streams[row])!=address.stream or int(context.frames[row])!=address.frame:
                    raise ValueError('PRE root context address differs')
                root=worlds.detach_root(context,self.source,row=row,owner=owner,enable_device_worlds=True)
                plans=worlds.plan_starts(root,self.source,self.recipe)
                inputs={name:getattr(frames,name)[step,row,owner][None,None]for name in INPUTS};inputs['history_mask']=np.ones((1,1),bool)
                request=Request(plans,tuple(map(float,frames.raw_policy_logits[step,row,owner])),
                    tuple(map(bool,frames.legal_action_mask[step,row,owner])),inputs,
                    int(frames.actor_indices[step,row,owner]),int(frames.actor_indices[step,row,1-owner]))
                grouped.setdefault(record['address']['frame']if getattr(self,'vectorized',False)else 0 if getattr(self,'chunk_batch',False)else step,[]).append((record,request))
            except (ValueError,RuntimeError,TypeError,KeyError,IndexError)as error:
                record.update(status='REFUSED',refusal=type(error).__name__+': '+str(error),improvement=None)
            finally:
                if getattr(self,'vectorized',False):
                    if not hasattr(self,'root_prepare_seconds'):self.root_prepare_seconds={}
                    frame=record['address']['frame']
                    self.root_prepare_seconds[frame]=self.root_prepare_seconds.get(frame,0.)+time.perf_counter()-preparation_started
        return self._price_groups(records,grouped,frames,bank,params)

    def _price_groups(self,records,grouped,frames,bank,params):
        for step,group in grouped.items():
            # Queue mode groups original selected requests across chunk frames.
            # Capacity changes only packing, never world keys or per-root math.
            capacity=self.frame_pricer.capacity
            for offset in range(0,len(group),capacity):
                items=group[offset:offset+capacity]
                # Device/producer failures stop the study, never fast-refuse a
                # whole frame or chunk to improve apparent throughput.
                results=self.frame_pricer.price([request for _,request in items],self.materializer,self.recipe,params)
                if len(results)!=len(items):raise ValueError('frame pricing dropped a selected root')
                self.batch_timings.append(dict(self.frame_pricer.last_timing,frames=sorted({r['address']['frame']for r,_ in items})))
                for (record,_),result in zip(items,results):
                    record['diagnostics']=dict(world_reports=result.world_reports,missing_worlds=result.missing_worlds,
                        actual_teacher_calls=result.actual_teacher_calls,reply_input_approximation=result.reply_input_approximation)
                    if result.status=='LABELED':record.update(status='LABELED',refusal=None,improvement=asdict(result.reduction.improvement))
                    elif result.status=='REFUSED':record['refusal']=result.reason or'unknown-required-world-or-reply'
                    else:raise ValueError('selected original recipe returned an unselected result')
        validate_records(frames,bank,self.recipe,records);self.record_metrics(records)
        return records
