"""Opt-in actual device Arm-D chunk consumer and original-cache target bridge.

Only requested learner PRE roots are quarter-selected by the original Recipe.
Actor trees are the verified ActorBank archives. Public source sampling and all
reply/leaf arithmetic are the existing device teacher, never a private critic.
Raw logits and every selected/refused address survive in the disk spool.
"""
from dataclasses import asdict
import hashlib
import json
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
    def __init__(self,prepared,plan,*,batch_joint=True):
        from .v5_device_backend import Prepared
        from pokezero.v5_run import Plan,_sha
        if type(prepared)is not Prepared or type(plan)is not Plan or plan.model.arm!='D':
            raise ValueError('actual prepared device D Plan required')
        if type(batch_joint)is not bool:raise ValueError('explicit joint batching bool required')
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
        return records
