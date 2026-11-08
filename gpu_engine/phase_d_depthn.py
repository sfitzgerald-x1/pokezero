"""Explicit R7 batching fallback behind the original protocol constructor.

DepthN supplies only execution geometry. Worlds, branch domains, physical
switch routing, evaluator precision and the original completed-Q rule stay
owned by phase_d_ladder_search. No D3 selection or world sampler runs here.
"""
from typing import NamedTuple
import math

import jax
import jax.numpy as j
import numpy as np

from . import device_public_teacher as teacher, device_depthn_pricer as depthn
from . import depthn_backups, transition, engine_device_observation as observation
from .phase_d_ladder_search import Search as SerialSearch, domain_seed
from .phase_d_search import probabilities
from .phase_d_live_worlds import Refusal


class AddressedState(NamedTuple):
    state: object
    seeds: object
    ply: object

    @property
    def done(self): return self.state.done

    @property
    def winner(self): return self.state.winner


def folded_state(wrapped):
    """The SAME domain fold before both sampling and stepping, at each ply."""
    salt=wrapped.seeds[:,wrapped.ply[0]]
    return wrapped.state._replace(keys=jax.vmap(jax.random.fold_in)(wrapped.state.keys,salt))


def rename_inputs(inputs):
    """Aliases/history dimension only; the champion still reads the same23 rows."""
    keys=(('numeric','numeric_features'),('categorical','categorical_ids'),
          ('token_types','token_type_ids'),('attention','attention_mask'))
    return {**{target:inputs[source][:,None] for source,target in keys},
            'history_mask':j.ones((inputs['numeric'].shape[0],1),bool)}


def original_inputs(inputs):
    return {target:inputs[source][:,0] for source,target in
            (('numeric_features','numeric'),('categorical_ids','categorical'),
             ('token_type_ids','token_types'),('attention_mask','attention'))}


def addressed_executor(legal,observe,dispatch,sample,advance):
    """Keep one-world tracker namespaces inside DepthN's mapped programs."""
    def view(w,p,owner,mask):
        result=observe(w.state,p,owner,mask)
        # Keep the original encoder error. Only surface-specific guards may
        # ignore a waiting seat; a value leaf always requires its own view.
        return result._replace(inputs=rename_inputs(result.inputs))
    def step(w,p,actions,chance):
        result=advance(folded_state(w),p,actions,chance)
        new=AddressedState(result.step.state,w.seeds,w.ply+1)
        return result._replace(step=result.step._replace(state=new))
    return teacher.Executor(jax.jit(lambda w:legal(w.state)),jax.jit(view),
        jax.jit(lambda obs,w,owner,a:dispatch(obs,w.state,owner,a)),
        jax.jit(lambda w:sample(folded_state(w))),jax.jit(step))


class PreparedPricer:
    """One fixed world count; grouping adjacent worlds is geometry, not a draw."""
    def __init__(self,executor,forward,*,worlds,block_lanes=2187,blocks_per_call=16):
        if worlds not in (2,8,16,32):raise ValueError('Declared even world budget required')
        self.worlds=worlds
        self.pricer=depthn.DepthNPricer(executor,None,capacity=worlds//2,depth=2,
            k_own=9,k_opp=9,k1_own=9,k1_opp=9,block_lanes=block_lanes,
            blocks_per_call=blocks_per_call,selection='prior',fixed_capacity=True,
            forward=lambda _,inputs:forward(original_inputs(inputs)),actor_capacity=1,serial_surface_guards=True)
        self.last_timing=None

    def price(self,nodes,legal):
        if len(nodes)!=self.worlds or len(legal)!=9 or not any(legal):
            raise Refusal('ORIGINAL_R7_WORLD_ACTION_DENOMINATOR')
        if any(n.owner!=nodes[0].owner or n.prefix!=nodes[0].prefix or n.world!=i
               for i,n in enumerate(nodes)):
            raise Refusal('ORIGINAL_R7_WORLD_ORDER_OR_ADDRESS')
        count=self.worlds//2
        wrapped=[AddressedState(n.state,j.asarray([[domain_seed(n.prefix,n.world,p,0,
            domain='sampled-chance') for p in (0,1)]],j.uint32),j.zeros(1,j.int32)) for n in nodes]
        def paired(values):
            stacked=jax.tree.map(lambda *xs:j.stack(xs),*values)
            return jax.tree.map(lambda x:x.reshape((count,2)+x.shape[1:]),stacked)
        order=tuple(a for a,v in enumerate(legal)if v);padded=order+(order[0],)*(9-len(order))
        data=dict(states=paired(wrapped),publics=paired([n.public for n in nodes]),
            expected={},legal=np.tile(np.asarray(legal,bool),(count,1)),
            order=np.tile(np.asarray(padded,np.int32),(count,1)),
            order_size=np.full(count,len(order),np.int32),available=np.ones((count,2),bool),
            owner=np.full(count,nodes[0].owner,np.int32),
            peer_rows=(np.arange(self.worlds,dtype=np.int32),),
            leaf_rows=(np.arange(self.worlds,dtype=np.int32),))
        frame=self.pricer.price_prepared(data,count=count,params=(None,))
        self.last_timing=frame.timing
        reports=[]
        for world in range(self.worlds):
            root,wi=divmod(world,2)
            if frame.pre[root,wi]!=0:raise Refusal('R7_DEPTHN_REQUIRED_ROOT_SURFACE',str(frame.pre[root,wi]))
            peer=bool(frame.root_peer_requested[root,wi])
            mask=tuple(map(bool,frame.root_peer_legal[root,wi]));replies=tuple(i for i,v in enumerate(mask)if v)if peer else(0,)
            probs=probabilities(frame.root_peer_logits[root,wi],mask)if peer else None
            q={}
            for a in order:
                terms=[]
                for r in replies:
                    value=depthn_backups.node_value(frame,root,wi,1,a*9+r)
                    if value is None or not math.isfinite(value):raise Refusal('R7_DEPTHN_REQUIRED_BRANCH_UNKNOWN')
                    terms.append((probs[r]if peer else 1.)*value)
                value=math.fsum(terms)
                if abs(value)>1+1e-12:raise Refusal('COMPLETED_Q_OUT_OF_RANGE')
                q[a]=min(1.,max(-1.,value))
            reports.append(q)
        return [max(r.values())for r in reports],reports


class Search(SerialSearch):
    """Opt-in R7 only. Original protocol/server/request result surfaces retained."""
    def __init__(self,*args,**kwargs):
        if kwargs.get('rung')!='R7' or kwargs.get('constructor_worlds')is not None:
            raise ValueError('DepthN fallback is explicitly the original R7 only')
        super().__init__(*args,**kwargs)
        def advance(s,p,a,c):
            return transition.step_with_public(s,a,self.engine_tables,p,
                belief_tables=self.observation_tables.belief,chance_inputs=c)
        executor=addressed_executor(self.legal,self.observe,observation.map_actions,
                                   transition.sample_chance_inputs,advance)
        def forward(inputs):
            result=self.forward(inputs)
            valid=j.all(j.isfinite(result.policy_logits),axis=-1)&j.isfinite(result.value)&(j.abs(result.value)<=1)
            return Prediction(result.policy_logits,result.value,valid)
        self.depthn=PreparedPricer(executor,forward,worlds=32)

    def _price(self,nodes,depth,ply):
        if depth!=2 or ply!=0 or self.rung!='R7':raise Refusal('ONLY_ORIGINAL_R7_BATCHING_AUTHORIZED')
        # Original own request order; per-world equality remains checked by
        # DepthN and the unchanged process_many completed-Q denominator check.
        node=nodes[0];masks=self.legal(node.state)
        view=self.observe(node.state,node.public,j.asarray([node.owner],j.int32),masks[:,node.owner])
        legal=tuple(map(bool,np.asarray(jax.block_until_ready(view.observation.legal_action_mask))[0]))
        return self.depthn.price(nodes,legal)


class Prediction(NamedTuple):
    policy_logits: object
    value: object
    valid: object
