"""Explicit champion breadth/chance/depth/rollout budgets on one GPU.

These are separate sampled-chance ladder rungs. The original exact-chance
Phase D prerequisite and D3 training-label provider remain unchanged. Each
request retains every declared world/action/reply/sample; one required failure
refuses the request. A CPU panel may explicitly fall back to raw and counts it.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import time

import jax
import jax.numpy as j
import numpy as np

from pokezero.live_protocol_actor import ActorContract,_packet
from pokezero.recorded_encoder_runner import JaxRecordedAdapter
from pokezero.v5_contract import DecisionAddress
from pokezero.v5_teacher_public_root import capture_root
from .phase_d_search import Search as OriginalSearch,probabilities,_host,SCHEMA
from . import phase_d_live_worlds as worlds,phase_d_protocol_world as protocol
from . import transition,device_public_leaf as leaf,engine_device_observation as observation
from .benchmark_device_selfplay import request_seats

RUNGS={
 'R1':dict(worlds=2,chance_samples=1,depth=1,leaf='VALUE',rollout_leaves=0),
 'R3':dict(worlds=32,chance_samples=1,depth=1,leaf='VALUE',rollout_leaves=0),
 'R4':dict(worlds=32,chance_samples=8,depth=1,leaf='VALUE',rollout_leaves=0),
 'R7':dict(worlds=32,chance_samples=1,depth=2,leaf='VALUE',rollout_leaves=0),
 'R5':dict(worlds=8,chance_samples=1,depth=1,leaf='ROLLOUT',rollout_leaves=8),
}


def domain_seed(prefix,world,ply,sample,*,domain):
    """No rung/budget in the key: all shared draws are exact budget prefixes."""
    if type(prefix)is not str or len(prefix)!=64 or any(type(v)is not int or v<0 for v in (world,ply,sample)):
        raise ValueError('fixed original root/world/ply/draw address required')
    return int.from_bytes(hashlib.sha256(json.dumps((prefix,world,ply,sample,domain),separators=(',',':')).encode()).digest()[:4],'big')


def reduce_q(legal,reports,*,requested_worlds):
    """Same mean completed-Q root operator for every rung, never softmax+Q."""
    if requested_worlds not in (2,8,16,32)or len(reports)!=requested_worlds or len(legal)!=9:
        raise ValueError('all fixed world slots required')
    q=[]
    for action,active in enumerate(legal):
        if not active:q.append(None);continue
        values=[]
        for report in reports:
            value=report[action]
            if value is None or not math.isfinite(value)or abs(value)>1:raise ValueError('required world action unknown')
            values.append(value)
        q.append(math.fsum(values)/requested_worlds)
    if not any(legal):raise ValueError('literal root legal action required')
    return tuple(q),max((a for a,v in enumerate(legal)if v),key=lambda a:(q[a],-a))


@dataclass
class Node:
    state:object
    public:object
    owner:int
    prefix:str
    world:int
    request:int


class Search(OriginalSearch):
    def __init__(self,*args,rung,owning_config,rollout_max_requests=1024,constructor_worlds=None,**kwargs):
        if rung not in RUNGS or type(rollout_max_requests)is not int or rollout_max_requests<=0:
            raise ValueError('declared champion rung and finite rollout cap required')
        # Original initialization loads exactly one actual9375 archive/catalog.
        super().__init__(*args,count=32,**kwargs)
        self.rung=rung;self.recipe=dict(RUNGS[rung]);self.count=self.recipe['worlds']
        if constructor_worlds not in (None,16,32):raise ValueError('constructor-only16/32 budget required')
        self.constructor_only=constructor_worlds is not None
        if self.constructor_only:self.count=constructor_worlds
        self.owning_config=owning_config;self.rollout_max_requests=rollout_max_requests
        def step(s,p,a,c):
            result=transition.step_with_public(s,a,self.engine_tables,p,
                belief_tables=self.observation_tables.belief,chance_inputs=c)
            return result.step.invalid_actions,result.step.state,result.trackers
        self.step_many=jax.jit(jax.vmap(step))
        self.sample_many=jax.jit(jax.vmap(transition.sample_chance_inputs))
        def rollout(s,p,owner):
            def condition(carry):
                state,_,steps,bad=carry
                return (~state.done[0])&(steps<self.rollout_max_requests)&(~bad)
            def body(carry):
                state,public,steps,bad=carry
                requested=request_seats(state)
                masks=transition.action_mask(state,self.engine_tables)&requested[...,None]
                encoded=[leaf.from_public(state,public,self.observation_tables,observer_sides=j.asarray([side],j.int32),
                    legal_action_mask=masks[:,side],enable_device_leaf=True,legacy_champion=True)for side in (0,1)]
                inputs={key:j.concatenate([e.inputs[key]for e in encoded],axis=0)for key in encoded[0].inputs}
                predictions=self.forward(inputs)
                bad|=(~j.isfinite(predictions.policy_logits)).any()|(~j.isfinite(predictions.value)).any()
                actions=[]
                for side,e in enumerate(encoded):
                    chosen=j.argmax(j.where(masks[0,side],predictions.policy_logits[side],-j.inf)).astype(j.int32)
                    physical=leaf.map_actions(e,state,chosen[None])[0]
                    actions.append(j.where(requested[0,side],physical,-1))
                    bad|=requested[0,side]&(e.observation.error[0]!=0)
                result=transition.step_with_public(state,j.stack(actions)[None,:],self.engine_tables,public,
                    belief_tables=self.observation_tables.belief)
                bad|=result.step.invalid_actions[0]
                return result.step.state,result.trackers,steps+1,bad
            final,public,steps,bad=jax.lax.while_loop(condition,body,(s,p,j.asarray(0,j.int32),j.asarray(False)))
            known=final.done[0]&(~bad)&((final.winner[0]>=0)&(final.winner[0]<=2))
            value=j.where(final.winner[0]==2,0.,j.where(final.winner[0]==owner,1.,-1.))
            return known,value,steps
        self.rollouts=jax.jit(jax.vmap(rollout))

    def _prepare_protocol(self,request,index):
        required={'schema','request_id','battle_id','owner','stream_id','decision_index','generation','prefix_sha256','lines','own_request_line'}
        if set(request)!=required or request['schema']!=SCHEMA or request['owner']not in ('p1','p2'):
            raise worlds.Refusal('MALFORMED_SEARCH_REQUEST')
        if (type(request['stream_id'])is not int or not 0<request['stream_id']<2**31
                or type(request['decision_index'])is not int or request['decision_index']<0
                or type(request['battle_id'])is not str or not request['battle_id'].strip()
                or type(request['generation'])is not int or request['generation']<=0):
            raise worlds.Refusal('MALFORMED_SEARCH_ADDRESS')
        lines=tuple(request['lines']);prefix=hashlib.sha256(''.join(x+'\n'for x in lines).encode()).hexdigest()
        if prefix!=request['prefix_sha256']or not lines or lines[-1]!=request['own_request_line']:
            raise worlds.Refusal('SEARCH_PREFIX_OR_REQUEST_MISMATCH')
        key=(self.rung,request['battle_id'],request['owner'],request['stream_id']);address=(key,request['decision_index'],prefix)
        if address in self.seen:raise worlds.Refusal('SEARCH_REQUEST_ALREADY_ATTEMPTED')
        self.seen.add(address)
        owner=int(request['owner'][1])-1
        prior=None if self.saved_root_admission else self.sessions.get(key)
        if prior is None:
            adapter=JaxRecordedAdapter(self.catalog,owner_side=owner,compiled_functions=self.functions)
            adapter.reset(stream_id=request['stream_id'],battle_id=request['battle_id']);old=()
        else:adapter,old=prior
        if lines[:len(old)]!=old or len(lines)<=len(old):raise worlds.Refusal('SEARCH_PREFIX_NOT_ORIGINAL_SUCCESSOR')
        # Original Python public-root capture checks literal owning request and
        # exact-foe/split contamination before either constructor can consume it.
        public=capture_root(config=self.owning_config,contract=ActorContract.champion(),lines=lines,
            address=DecisionAddress(request['stream_id'],request['decision_index'],owner),battle_id=request['battle_id'],seed=self.seed&0xffffffff)
        packet=_packet(request['own_request_line'],request['owner'])
        adapter.feed(tuple(x for x in lines[len(old):]if not x.startswith('|request|')))
        if not self.saved_root_admission:self.sessions[key]=(adapter,lines)
        detached=worlds.from_b2(adapter,packet,self.source,self.metadata,address=public.address)
        made=protocol.sample(public,detached,lines=lines,catalog=self.catalog,source=self.source,metadata=self.metadata,
            config=self.owning_config,battle_id=request['battle_id'],seed=self.seed,count=self.count)
        return detached,made,tuple(map(bool,adapter.encode(packet,owner_side=owner)['legal_action_mask']))

    def _surfaces(self,nodes):
        surfaces=[];rows=[]
        for node in nodes:
            requested=_host(request_seats(node.state))[0]
            masks=self.legal(node.state)
            encoded=[self.observe(node.state,node.public,j.asarray([side],j.int32),masks[:,side])for side in (0,1)]
            for side,e in enumerate(encoded):
                if requested[side]and int(_host(e.observation.error)[0]):raise worlds.Refusal('REQUIRED_PUBLIC_ENCODER_REFUSED')
            rows.extend(e.inputs for e in encoded)
            maps=[];legal=[]
            for side,e in enumerate(encoded):
                valid=tuple(map(bool,_host(masks)[0,side]));legal.append(valid)
                if requested[side]and not any(valid):raise worlds.Refusal('REQUESTED_ACTION_SURFACE_EMPTY')
                maps.append(tuple(int(_host(leaf.map_actions(e,node.state,j.asarray([a],j.int32)))[0])for a in range(9)))
            surfaces.append(dict(requested=requested,legal=legal,maps=maps))
        outputs=self._forward_rows(rows)
        for i,surface in enumerate(surfaces):surface['predictions']=outputs[2*i:2*i+2]
        return surfaces

    def _steps(self,branches):
        output=[]
        stack=lambda values:jax.tree.map(lambda *x:j.stack(x),*values)
        for start in range(0,len(branches),self.lanes):
            real=branches[start:start+self.lanes];padded=real+[real[0]]*(self.lanes-len(real))
            states=[b['node'].state for b in padded]
            # chance keys are prospective root/world/ply/sample domains shared
            # across all actions and replies; no rung name enters these keys.
            states=[s._replace(keys=jax.vmap(jax.random.fold_in)(s.keys,j.full((1,),b['chance_seed'],j.uint32)))for s,b in zip(states,padded)]
            sampled=self.sample_many(stack(states))
            invalid,after,public=jax.block_until_ready(self.step_many(stack(states),stack([b['node'].public for b in padded]),
                j.stack([j.asarray([b['actions']],j.int32)for b in padded]),sampled))
            for i,b in enumerate(real):
                if bool(_host(invalid[i])[0]):raise worlds.Refusal('REQUIRED_SAMPLED_BRANCH_REFUSED')
                node=b['node'];state=jax.tree.map(lambda value:value[i],after);carry=jax.tree.map(lambda value:value[i],public)
                output.append(Node(state,carry,node.owner,node.prefix,node.world,node.request))
        return output

    def _rollout_values(self,nodes,ply):
        branches=[]
        for node in nodes:
            for draw in range(self.recipe['rollout_leaves']):
                seed=domain_seed(node.prefix,node.world,ply,draw,domain='raw-policy-rollout')
                state=node.state._replace(keys=jax.vmap(jax.random.fold_in)(node.state.keys,j.full((1,),seed,j.uint32)))
                branches.append(Node(state,node.public,node.owner,node.prefix,node.world,node.request))
        values=[];stack=lambda rows:jax.tree.map(lambda *x:j.stack(x),*rows)
        for start in range(0,len(branches),self.lanes):
            real=branches[start:start+self.lanes];padded=real+[real[0]]*(self.lanes-len(real))
            known,value,steps=jax.block_until_ready(self.rollouts(stack([x.state for x in padded]),stack([x.public for x in padded]),j.asarray([x.owner for x in padded],j.int32)))
            if not np.asarray(known[:len(real)]).all():raise worlds.Refusal('ROLLOUT_LEAF_UNKNOWN_OR_REQUEST_CAP')
            values.extend(map(float,np.asarray(value[:len(real)])))
        k=self.recipe['rollout_leaves']
        return [math.fsum(values[i:i+k])/k for i in range(0,len(values),k)]

    def _price(self,nodes,depth,ply):
        if not nodes:return [],[]
        surfaces=self._surfaces(nodes);branches=[];shape=[]
        for i,(node,surface)in enumerate(zip(nodes,surfaces)):
            owner=node.owner;foe=1-owner;requested=surface['requested'];legal=surface['legal']
            own=tuple(a for a,v in enumerate(legal[owner])if v)if requested[owner]else(-1,)
            replies=tuple(a for a,v in enumerate(legal[foe])if v)if requested[foe]else(-1,)
            probs=probabilities(surface['predictions'][foe][0],legal[foe])if requested[foe]else None
            shape.append((own,replies,probs))
            for a in own:
                for reply in replies:
                    actions=[-1,-1]
                    if a>=0:actions[owner]=surface['maps'][owner][a]
                    if reply>=0:actions[foe]=surface['maps'][foe][reply]
                    for draw in range(self.recipe['chance_samples']):
                        branches.append(dict(node=node,parent=i,action=a,reply=reply,actions=tuple(actions),
                            chance_seed=domain_seed(node.prefix,node.world,ply,draw,domain='sampled-chance')))
        if len(branches)>self.cap:raise worlds.Refusal('LADDER_BRANCH_CAP',f'{len(branches)}>{self.cap}')
        children=self._steps(branches);values=[None]*len(children);pending=[];indices=[]
        for i,node in enumerate(children):
            if bool(_host(node.state.done)[0]):
                winner=int(_host(node.state.winner)[0])
                if winner not in (0,1,2):raise worlds.Refusal('TERMINAL_WINNER_UNAVAILABLE')
                values[i]=0. if winner==2 else 1. if winner==node.owner else -1.
            else:pending.append(node);indices.append(i)
        if depth>1:
            next_values,_=self._price(pending,depth-1,ply+1)
        elif self.recipe['leaf']=='ROLLOUT':next_values=self._rollout_values(pending,ply+1)
        else:
            rows=[]
            for node in pending:
                mask=self.legal(node.state)
                observed=self.observe(node.state,node.public,j.asarray([node.owner],j.int32),mask[:,node.owner])
                if int(_host(observed.observation.error)[0]):raise worlds.Refusal('REQUIRED_PUBLIC_VALUE_LEAF_REFUSED')
                rows.append(observed.inputs)
            next_values=[value for logits,value in self._forward_rows(rows)]if rows else[]
        for i,value in zip(indices,next_values):values[i]=value
        grouped=[{}for _ in nodes]
        for branch,value in zip(branches,values):
            if value is None or not math.isfinite(value):raise worlds.Refusal('REQUIRED_LEAF_UNAVAILABLE')
            pair=grouped[branch['parent']].setdefault((branch['action'],branch['reply']),[]);pair.append(value)
        reports=[];best=[]
        for node,parts,(own,replies,probs)in zip(nodes,grouped,shape):
            q={}
            for action in own:
                terms=[]
                for reply in replies:
                    samples=parts.get((action,reply),())
                    if len(samples)!=self.recipe['chance_samples']:raise worlds.Refusal('SAMPLED_CHANCE_DENOMINATOR_CHANGED')
                    terms.append((1. if reply<0 else probs[reply])*math.fsum(samples)/len(samples))
                total=math.fsum(terms)
                if abs(total)>1+1e-12:raise worlds.Refusal('COMPLETED_Q_OUT_OF_RANGE')
                q[action]=min(1.,max(-1.,total))
            reports.append(q);best.append(max(q.values()))
        return best,reports

    def process_many(self,requests):
        # Each root retains its own failure surface. GPU forwards/transitions
        # within a root are dense branch batches; the transport serializes the
        # GPU owner across concurrent games rather than allocating per-game GPUs.
        results=[]
        for index,request in enumerate(requests):
            started=time.monotonic();made=None
            try:
                detached,made,legal=self._prepare_protocol(request,index)
                if not made.fully_constructed:
                    raise worlds.Refusal('WORLD_CONSTRUCTOR_REFUSED',';'.join(sorted({r for w in made.worlds for r in w.refusals})))
                if self.constructor_only:
                    results.append(dict(schema=SCHEMA,request_id=request.get('request_id'),archive_sha256=self.archive_sha256,
                        status='COMPLETE',action_index=None,q_values=None,refusal=None,requested_worlds=self.count,
                        constructed_worlds=self.count,decision_seconds=None,constructor_seconds=time.monotonic()-started,
                        scope='CONSTRUCTOR_ONLY_NOT_SEARCH_LATENCY_ADMISSION'));continue
                b2=detached.view['_b2_root_inputs']
                raw_input={target:j.asarray(b2[source])[None]for source,target in (('numeric_features','numeric'),('categorical_ids','categorical'),('token_type_ids','token_types'),('attention_mask','attention'))}
                raw_logits,raw_value=self._forward_rows([raw_input])[0]
                nodes=[Node(w.state,w.public,detached.address.owner,request['prefix_sha256'],w.index,index)for w in made.worlds]
                _,reports=self._price(nodes,self.recipe['depth'],0)
                if any(tuple(a for a in range(9)if legal[a])!=tuple(r)for r in reports):raise worlds.Refusal('OWN_REQUEST_PHYSICAL_LEGALITY_DIFFERS')
                q,action=reduce_q(legal,[tuple(r.get(a)for a in range(9))for r in reports],requested_worlds=self.count)
                result=dict(status='COMPLETE',action_index=action,q_values=q,refusal=None,requested_worlds=self.count,
                    constructed_worlds=self.count,decision_seconds=time.monotonic()-started,
                    rung=self.rung,budget=self.recipe,raw_policy_logits=raw_logits,raw_value=raw_value,
                    world_action_q=[tuple(r.get(a)for a in range(9))for r in reports],
                    world_seeds=made.seeds,search_seed=self.seed,rollout_max_requests=self.rollout_max_requests,root_operator='ARGMAX_MEAN_COMPLETED_Q_COMMON_ALL_RUNGS')
            except (ValueError,KeyError,TypeError,IndexError,AttributeError)as error:
                result=dict(status='REFUSED',action_index=None,q_values=None,refusal=str(error),requested_worlds=self.count,
                    constructed_worlds=0 if made is None else sum(w.state is not None for w in made.worlds),decision_seconds=None,rung=self.rung,budget=self.recipe,
                    world_refusals=[]if made is None else[dict(world=w.index,kinds=list(w.refusals))for w in made.worlds if w.refusals])
            results.append(dict(schema=SCHEMA,request_id=request.get('request_id'),archive_sha256=self.archive_sha256,**result))
        return results
