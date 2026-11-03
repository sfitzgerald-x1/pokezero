"""Frame-batched D prices, retaining the serial source sampler and reduction.

The device call maps independent ONE-ROW public worlds, then their81 joint
branches. It never flattens public tracker namespaces into ordinary battles.
Bounded root blocks keep large frames inside one device invocation. All random
keys, missing worlds, positive reply mass and original completed-Q math remain.
"""
from dataclasses import dataclass,replace
import time

import jax
import jax.numpy as j
import numpy as np

from pokezero import v5_gumbel as original
from . import device_public_teacher as serial,device_public_world_state as mapping,v5_model

SERIAL_INPUT_ORDER=('numeric_features','categorical_ids','token_type_ids','attention_mask','history_mask')
FRAME_WORK_KEYS=('evaluated_root_slots','evaluated_world_slots','evaluated_joint_slots',
    'evaluated_peer_forward_rows','evaluated_leaf_forward_rows','padding_root_slots',
    'padding_world_slots','substituted_world_slots','pre_refused_available_worlds')

@dataclass(frozen=True)
class Request:
    plans:object
    root_logits:tuple
    root_legal:tuple
    root_inputs:object
    learner_actor:int
    opponent_actor:int


def make_kernel(executor,model,*,params,capacity,block_rows,mixed_learners=False):
    """Preserve original actor programs, inside one frame device invocation.

    Transitions map roots/worlds/joint branches together. Actor call geometry
    stays one-row for peer and fixed81 mapped one-row for leaf, as in serial.
    BF16 XLA programs with different row/parameter batching are NOT equivalent
    at the fixed tolerance. Device maps retain those original programs while
    removing host-per-root dispatch; archive weights remain verified constants.
    """
    if capacity%block_rows or executor.joint is None:raise ValueError('bounded mapped joint executor required')
    def forward(params,inputs):return v5_model.forward(params,inputs,model,compute_dtype='bfloat16')
    peers=tuple(jax.jit(lambda inputs,p=p:forward(p,inputs))for p in params)
    leaves=tuple(jax.jit(jax.vmap(peer))for peer in peers)
    def views(state,public,owner):
        masks=executor.legal(state);seat=j.asarray([owner],j.int32)
        own=executor.observe(state,public,seat,masks[:,owner])
        peer=executor.observe(state,public,1-seat,masks[:,1-owner])
        return own.observation,peer.observation,own.inputs,peer.inputs
    def compute_block(a):
        b=block_rows;n=2*b
        states,publics=a['states'],a['publics']
        flatten=lambda x:x.reshape((n,)+x.shape[2:])
        states=jax.tree.map(flatten,states);publics=jax.tree.map(flatten,publics)
        owners=j.repeat(a['owner'],2)
        own,peer,own_inputs,peer_inputs=jax.vmap(views)(states,publics,owners)
        root_legal=j.repeat(a['legal'],2,axis=0)
        legal=peer.legal_action_mask[:,0];requested=j.any(legal,axis=-1)
        pre=j.where(own.error[:,0]!=0,1,0)
        pre=j.where((pre==0)&~j.all(own.legal_action_mask[:,0]==root_legal,axis=-1),2,pre)
        if set(own_inputs)!=set(a['expected']):pre=j.where(pre==0,10,pre)
        for index,key in enumerate(k for k in SERIAL_INPUT_ORDER if k in a['expected']):
            if key not in own_inputs:continue
            expected=j.repeat(a['expected'][key],2,axis=0);value=own_inputs[key]
            same=j.all(value==expected,axis=tuple(range(1,value.ndim)))
            pre=j.where((pre==0)&~same,3+index,pre)
        pre=j.where((pre==0)&(peer.error[:,0]!=0),8,pre)
        # Original peer callback observes ONE row, with its verified archive
        # captured exactly as in Provider.bind. No dynamic parameter batching.
        prediction=jax.lax.map(lambda pair:jax.lax.switch(pair[0],peers,pair[1]),
            (j.repeat(a['peer_actor'],2),peer_inputs))
        logits=prediction.policy_logits.reshape(n,9);pv=prediction.value.reshape(n);valid=prediction.valid.reshape(n)
        usable=valid&j.all(j.isfinite(logits),axis=-1)&j.isfinite(pv)&(j.abs(pv)<=1)
        pre=j.where((pre==0)&requested&~usable,9,pre)
        logits=j.where(requested[:,None],logits,0.)
        # EXACT original action-order x ascending replies, padded with the
        # first original pair. Keep actor row positions as well as row shapes.
        # Changing BF16 batch row layout must not alter a target's rounding.
        reply_count=j.where(requested,j.sum(legal,axis=-1),1)
        reply_order=j.argsort(j.where(legal,j.arange(9),j.arange(9)+9),axis=-1)
        reply_order=j.where(requested[:,None],reply_order,0)
        order=j.repeat(a['order'],2,axis=0);sizes=j.repeat(a['order_size'],2)
        ordinal=j.broadcast_to(j.arange(81),(n,81));required=sizes*reply_count
        action_row=j.where(ordinal<required[:,None],ordinal//reply_count[:,None],0)
        reply_row=j.where(ordinal<required[:,None],ordinal%reply_count[:,None],0)
        pairs=j.stack((j.take_along_axis(order,action_row,axis=1),
            j.take_along_axis(reply_order,reply_row,axis=1)),axis=-1)
        chance=jax.vmap(executor.sample)(states)
        invalid,done,winner,error,inputs=jax.vmap(executor.joint)(states,publics,own,peer,
            owners[:,None],requested[:,None],pairs,chance)
        invalid,done,winner,error=(x[:,:,0]for x in (invalid,done,winner,error))
        # Keep the original fixed81 actor vmap intact for each world. The
        # physical/public transition grid above is still fully root-batched.
        if mixed_learners:
            prediction=jax.lax.map(lambda pair:jax.lax.switch(pair[0],leaves,pair[1]),
                (j.repeat(a['learner_actor'],2),inputs))
        else:prediction=jax.lax.map(leaves[0],inputs)
        values=prediction.value.reshape(n,81);valid=prediction.valid.reshape(n,81)
        leaf_logits=prediction.policy_logits.reshape(n,81,9)
        usable=valid&j.all(j.isfinite(leaf_logits),axis=-1)&j.isfinite(values)&(j.abs(values)<=1)
        terminal_ok=(winner>=0)&(winner<=2)
        errors=j.where(invalid,1,j.where(done&~terminal_ok,2,j.where(~done&(error!=0),3,j.where(~done&~usable,4,0))))
        terminal=j.where(winner==2,0.,j.where(winner==owners[:,None],1.,-1.))
        values=j.where(done,terminal,values)
        return dict(pre=pre.reshape(b,2),peer_logits=logits.reshape(b,2,9),peer_legal=legal.reshape(b,2,9),
            requested=requested.reshape(b,2),values=values.reshape(b,2,81),errors=errors.reshape(b,2,81),
            done=done.reshape(b,2,81))
    def empty():
        b=block_rows
        return dict(pre=j.zeros((b,2),j.int32),peer_logits=j.zeros((b,2,9),j.float32),peer_legal=j.zeros((b,2,9),bool),
            requested=j.zeros((b,2),bool),values=j.zeros((b,2,81),j.float32),errors=j.zeros((b,2,81),j.int32),done=j.zeros((b,2,81),bool))
    def run(data):
        blocks=jax.tree.map(lambda x:x.reshape((capacity//block_rows,block_rows)+x.shape[1:]),data)
        result=jax.lax.map(lambda a:jax.lax.cond(j.any(a['active']),lambda:compute_block(a),empty),blocks)
        return jax.tree.map(lambda x:x.reshape((capacity,)+x.shape[2:]),result)
    return jax.jit(run)


def _host_worlds(requests,materializer,recipe):
    """Unchanged source materialization on CPU; only already-redacted roots.

    Fresh per-root keys still come from original empty_state/Recipe. Moving
    these functional preparation operations to CPU avoids one-row GPU traffic;
    no private opponent state is transferred or borrowed.
    """
    worlds=[];cpu=jax.devices('cpu')[0]
    with jax.default_device(cpu):
        for request in requests:
            root=request.plans.root
            public=jax.device_put(root.public,cpu)
            safe=replace(root,public=public)
            worlds.append(tuple(mapping.materialize(safe,start,materializer,recipe)for start in request.plans.starts))
    return worlds


class FramePricer:
    def __init__(self,executor,model,*,capacity=2048,block_rows=32,actor_capacity=33):
        if (type(capacity)is not int or type(block_rows)is not int or capacity<=0 or block_rows<=0
                or capacity%block_rows or type(actor_capacity)is not int or actor_capacity<=0):
            raise ValueError('finite explicit root/actor capacities required')
        self.executor=executor;self.model=model;self.capacity=capacity;self.block_rows=block_rows;self.actor_capacity=actor_capacity
        self.kernels={};self.param_key=None;self.params=None;self.source_params=None;self.last_timing=None

    def bind(self,params):
        if not 0<len(params)<=self.actor_capacity:raise ValueError('declared actor bank exceeds fixed capacity')
        key=tuple(map(id,params))
        if key!=self.param_key:
            self.params=tuple(params);self.param_key=key;self.kernels={}
            # Retain identities as well as integer ids: a later archive bank
            # cannot accidentally reuse a collected Python object's id.
            self.source_params=tuple(params)

    def price(self,requests,materializer,recipe,params):
        requests=tuple(requests)
        if not requests:return ()
        if type(recipe)is not original.Recipe or type(materializer)is not mapping.Tables:
            raise ValueError('original materializer and random recipe required')
        if len(requests)>self.capacity:raise ValueError('one frame exceeds declared root capacity')
        self.bind(params);started=time.perf_counter()
        for request in requests:
            p=request.plans
            if (not p.selected or p.selected!=recipe.selected(p.root.address)or len(p.starts)!=2
                    or tuple(x.world for x in p.starts)!=(0,1)
                    or p.source_hash!=materializer.source.source_hash
                    or p.source_rows_sha256!=materializer.source.source_rows_sha256
                    or p.root.source_binding_sha256!=materializer.source.binding_sha256):
                raise ValueError('same selected root/source/world/recipe required')
            original._probabilities(tuple(request.root_logits),original._mask(request.root_legal))
            if not request.root_inputs or set(request.root_inputs)-set(SERIAL_INPUT_ORDER):
                raise ValueError('original encoder input keys required')
            if any(type(x)is not int or not 0<=x<len(params)for x in (request.learner_actor,request.opponent_actor)):
                raise ValueError('verified original actor index required')
        worlds=_host_worlds(requests,materializer,recipe);prepared=time.perf_counter()
        available=[w for pair in worlds for w in pair if w.state is not None]
        output=None;lower_seconds=compile_seconds=device_seconds=0.
        if available:
            first=available[0];flat=[w if w.state is not None else first for pair in worlds for w in pair]
            def stacked(values,world_axis=False):
                values=jax.device_get(tuple(values))
                def pack(*leaves):
                    value=np.stack(leaves);length=self.capacity*(2 if world_axis else 1)
                    padded=np.empty((length,)+value.shape[1:],value.dtype);padded[:len(value)]=value;padded[len(value):]=value[0]
                    return padded.reshape((self.capacity,2)+value.shape[1:])if world_axis else padded
                return jax.tree.map(pack,*values)
            count=len(requests)
            def vector(values,dtype,default=0):
                result=np.full(self.capacity,default,dtype);result[:count]=values;return result
            data=dict(states=stacked([w.state for w in flat],True),publics=stacked([w.public for w in flat],True),
                expected=stacked([r.root_inputs for r in requests]),
                legal=np.zeros((self.capacity,9),bool),order=np.zeros((self.capacity,9),np.int32),
                order_size=np.ones(self.capacity,np.int32),active=np.arange(self.capacity)<count,
                owner=vector([r.plans.root.address.owner for r in requests],np.int32),
                peer_actor=vector([r.opponent_actor for r in requests],np.int32),
                learner_actor=vector([r.learner_actor for r in requests],np.int32))
            data['legal'][:count]=[r.root_legal for r in requests];data['legal'][count:,0]=True
            for i,r in enumerate(requests):
                order=recipe.pricing_order(r.plans.root.address,r.root_logits,r.root_legal)
                data['order'][i]=order+(order[0],)*(9-len(order));data['order_size'][i]=len(order)
            mixed=any(r.learner_actor!=0 for r in requests)
            device_data=jax.device_put(data)
            if mixed not in self.kernels:
                fn=make_kernel(self.executor,self.model,params=self.params,capacity=self.capacity,block_rows=self.block_rows,mixed_learners=mixed)
                clock=time.perf_counter();lowered=fn.lower(device_data);lower_seconds=time.perf_counter()-clock
                clock=time.perf_counter();self.kernels[mixed]=lowered.compile();compile_seconds=time.perf_counter()-clock
            # Exactly one device invocation for the entire selected frame.
            clock=time.perf_counter();output=jax.device_get(jax.block_until_ready(self.kernels[mixed](device_data)))
            device_seconds=time.perf_counter()-clock
        evaluated=0 if output is None else ((len(requests)+self.block_rows-1)//self.block_rows)*self.block_rows
        work=dict(evaluated_root_slots=evaluated,evaluated_world_slots=2*evaluated,evaluated_joint_slots=162*evaluated,
            evaluated_peer_forward_rows=2*evaluated,evaluated_leaf_forward_rows=162*evaluated,
            padding_root_slots=0 if output is None else evaluated-len(requests),
            padding_world_slots=0 if output is None else 2*(evaluated-len(requests)),
            substituted_world_slots=0 if output is None else 2*len(requests)-len(available),
            pre_refused_available_worlds=0 if output is None else sum(w.state is not None and int(output['pre'][i,wi])!=0 for i,pair in enumerate(worlds)for wi,w in enumerate(pair)))
        finished=time.perf_counter();self.last_timing=dict(roots=len(requests),world_prepare_seconds=prepared-started,
            batch_pack_and_device_seconds=finished-prepared,device_invocations=int(output is not None),block_rows=self.block_rows,
            lower_seconds=lower_seconds,compile_seconds=compile_seconds,device_seconds=device_seconds,
            actor_strategy='ORIGINAL_ROW1_PEER_FIXED81_LEAF_DEVICE_MAP',**work)
        return tuple(self._result(i,r,pair,output,recipe)for i,(r,pair)in enumerate(zip(requests,worlds)))

    def _result(self,index,request,worlds,output,recipe):
        reports=[];prices=[];approximations=[];order=recipe.pricing_order(request.plans.root.address,request.root_logits,request.root_legal)
        counts=dict(reply_policy=0,public_leaf=0,chance=0,transitions=0,strategy='FRAME_ROOT_WORLD_JOINT',shared_frame_invocations=int(output is not None),
            scope='LOGICAL_RETAINED_ROOT_FIXED81_WORLD_ROWS; actual evaluated padding/substitution/pre-refusal work reported ONCE per frame, not here')
        pre_reasons={1:'current direct-encoder input error',2:'same actual root legal-action denominator required',8:'current direct-encoder input error',9:'current public actor output refused/nonfinite',10:'actual learner root input fields differ'}
        error_reasons={1:'hypothetical original action route invalid',2:'hypothetical terminal winner unavailable',3:'current direct-encoder input error',4:'current public actor output refused/nonfinite'}
        for wi,(start,world)in enumerate(zip(request.plans.starts,worlds)):
            report=dict(world=start.world,posterior_seed=start.seed,kernel_seed=world.kernel_seed,status='REFUSED',reason=None,
                required_pairs=None,completed_pairs=0,unknown_pairs=None,joint_values=None);reports.append(report)
            if world.state is None:
                report['reason']='physical-materialization:'+','.join(world.refusals);continue
            code=int(output['pre'][index,wi])
            if code:
                reason=pre_reasons.get(code)
                if reason is None:reason='actual learner root input differs: '+tuple(k for k in SERIAL_INPUT_ORDER if k in request.root_inputs)[code-3]
                report['reason']='ValueError: '+reason;continue
            requested=bool(output['requested'][index,wi]);legal=tuple(map(bool,output['peer_legal'][index,wi]))
            logits=tuple(map(float,output['peer_logits'][index,wi]));replies=tuple(i for i,v in enumerate(legal)if v)if requested else(0,)
            counts['reply_policy']+=int(requested);counts['chance']+=1;counts['transitions']+=81;counts['public_leaf']+=81
            if not world.reply_private_history_available:approximations.append(dict(world=start.world,**serial.REPLY_HISTORY_APPROXIMATION))
            grid=[[None]*9 for _ in range(9)];errors=[];completed=0
            ordinal=0
            for action in order:
                for reply in replies:
                    error=int(output['errors'][index,wi,ordinal])
                    if error:errors.append((action,reply,'ValueError: '+error_reasons[error]))
                    else:grid[action][reply]=float(output['values'][index,wi,ordinal]);completed+=1
                    ordinal+=1
            required=len(order)*len(replies)
            report.update(status='COMPLETE'if not errors else'REFUSED',required_pairs=required,completed_pairs=completed,
                unknown_pairs=required-completed,joint_values=grid,opponent_requested=requested,reply_legal=legal,reply_logits=logits,
                reason=None if not errors else'unknown-required-public-leaf',errors=tuple(errors),
                execution=dict(strategy='FRAME_ROOT_WORLD_JOINT',transition_rows=81,actor_forward_rows=81,padding_rows=81-required))
            prices.append(original.WorldPrices(requested,legal,logits,grid))
        missing=2-len(prices)
        if missing:return serial.Result('REFUSED',None,tuple(reports),tuple(worlds),missing,'unknown-required-world-or-reply-denominator',counts,tuple(approximations))
        reduced=original.reduce_completed_q(root_logits=request.root_logits,root_legal=request.root_legal,worlds=prices)
        return serial.Result(reduced.status,reduced,tuple(reports),tuple(worlds),0,reduced.reason,counts,tuple(approximations))
