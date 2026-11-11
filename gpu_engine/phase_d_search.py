"""One shared GPU: breadth-first 16/32-world, depth-one exact-chance search.

All concurrent panel requests enter one process and one batching queue. Every
positive exact chance/reply/world branch is required; an unsupported branch
refuses the request. No per-game GPU, CPU model or sampled-chance fallback.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import time

import jax
import jax.numpy as j
import numpy as np

from pokezero.live_protocol_actor import ActorContract, _packet
from pokezero.recorded_encoder_runner import JaxRecordedAdapter
from pokezero.live_protocol_runtime import CompiledEncoderFunctions
from pokezero.v5_contract import DecisionAddress
from . import champion, transition, device_public_leaf as leaf
from . import phase_d_live_worlds as worlds, phase_d_chance as chance
from . import device_public_world_state as states
from .benchmark_device_selfplay import request_seats

SCHEMA = 'pokezero.phase-d.shared-gpu-search.v1'


def _host(value):
    return np.asarray(jax.block_until_ready(value))


def probabilities(logits, legal):
    if len(logits)!=9 or len(legal)!=9 or not any(legal) or any(not math.isfinite(float(v)) for v in logits):
        raise ValueError('finite nine-action logits and nonempty literal legality required')
    maximum=max(float(v) for v,active in zip(logits,legal) if active)
    raw=[math.exp(float(v)-maximum) if active else 0. for v,active in zip(logits,legal)]
    total=math.fsum(raw)
    return tuple(x/total for x in raw)


def completed_q(legal, reports, *, requested_worlds):
    """Exact chance first, reply policy second, equal worlds last; no dropping."""
    if requested_worlds not in (16,32) or len(reports)!=requested_worlds:
        raise ValueError('all requested16/32 worlds required')
    q=[[] for _ in range(9)]
    for report in reports:
        replies=report['replies'];p=report['probabilities'];pairs=report['pairs']
        if len(p)!=len(replies) or not replies or abs(math.fsum(p)-1)>1e-12:
            raise ValueError('full unit-mass reply surface required')
        for a,active in enumerate(legal):
            if not active:continue
            for r,weight in zip(replies,p):
                cases=pairs.get((a,r))
                if not cases or any(type(m) is not Fraction or m<=0 or v is None or not math.isfinite(v) or abs(v)>1 for m,v in cases):
                    raise ValueError('unknown required exact chance/public leaf')
                if sum((m for m,_ in cases),Fraction())!=1:
                    raise ValueError('incomplete exact chance mass')
                q[a].append(weight*math.fsum(float(m)*v for m,v in cases)/requested_worlds)
    result=tuple(math.fsum(v) if active else None for v,active in zip(q,legal))
    return result, max((a for a,active in enumerate(legal) if active),key=lambda a:(result[a],-a))


@dataclass
class Pending:
    request: dict
    started: float
    root: object = None
    worldset: object = None
    reports: object = None
    required_chance_branches: int = 0
    legal: object = None
    result: object = None


class Search:
    """Construct once on the allocated GPU; real champion archive load once."""
    def __init__(self, archive, *, expected_archive_sha256, catalog, source,
                 metadata, engine_tables, observation_tables, count=16, seed=20261103,
                 chance_cap=262144, lanes=4096, compute_dtype='bfloat16', saved_root_admission=False):
        devices=jax.devices()
        if len(devices)!=1 or devices[0].platform!='gpu':
            raise ValueError('exactly ONE visible GPU required; no CPU search substitute')
        if count not in (16,32) or type(lanes) is not int or lanes<=0 or type(chance_cap) is not int or chance_cap<=0:
            raise ValueError('explicit16/32 worlds, branch cap and positive GPU batch lanes required')
        if compute_dtype not in ('float32','bfloat16'):
            raise ValueError('explicit evaluator precision required')
        params,config,meta=champion.load(archive)
        if (meta['weights_archive_sha256']!=expected_archive_sha256
                or meta['checkpoint_sha256']!=ActorContract.champion().checkpoint_sha256
                or catalog.checkpoint_sha256!=meta['checkpoint_sha256'] or config.tokens!=23):
            raise ValueError('real original9375 champion and exact archive/catalog binding required')
        self.catalog,self.source,self.metadata=catalog,source,metadata
        self.engine_tables,self.observation_tables=engine_tables,observation_tables
        self.tables=states.compile_tables(source,metadata)
        self.functions=CompiledEncoderFunctions(catalog)
        self.count,self.seed,self.cap,self.lanes=count,seed,chance_cap,lanes
        self.archive_sha256=expected_archive_sha256
        self.forward=jax.jit(lambda inputs:champion.forward(params,inputs['numeric'],inputs['categorical'],
            inputs['token_types'],inputs['attention'],config,compute_dtype=compute_dtype))
        self.legal=jax.jit(lambda s:transition.action_mask(s,engine_tables)&request_seats(s)[...,None])
        self.observe=jax.jit(lambda s,p,o,m:leaf.from_public(s,p,observation_tables,observer_sides=o,
            legal_action_mask=m,enable_device_leaf=True,legacy_champion=True))
        def advance(s,p,a,c,o):
            result=transition.step_with_public(s,a,engine_tables,p,
                belief_tables=observation_tables.belief,chance_inputs=c)
            mask=transition.action_mask(result.step.state,engine_tables)&request_seats(result.step.state)[...,None]
            root=leaf.from_public(result.step.state,result.trackers,observation_tables,observer_sides=o,
                legal_action_mask=mask[:,o[0]],enable_device_leaf=True,legacy_champion=True)
            return result.step.invalid_actions,result.step.state.done,result.step.state.winner,root.observation.error,root.inputs
        self.advance=jax.jit(jax.vmap(advance))
        if type(saved_root_admission) is not bool:raise ValueError('explicit saved-root mode required')
        self.saved_root_admission=saved_root_admission
        self.sessions={};self.seen=set();self.batch_calls=0

    def _prepare(self, pending):
        request=pending.request
        required={'schema','request_id','battle_id','owner','stream_id','decision_index','generation','prefix_sha256','lines','own_request_line'}
        if set(request)!=required or request['schema']!=SCHEMA or request['owner'] not in ('p1','p2'):
            raise worlds.Refusal('MALFORMED_SEARCH_REQUEST')
        if (type(request['stream_id']) is not int or not 0<request['stream_id']<2**31
                or type(request['decision_index']) is not int or request['decision_index']<0
                or type(request['battle_id']) is not str or not request['battle_id'].strip()
                or type(request['generation']) is not int or request['generation']<=0):
            raise worlds.Refusal('MALFORMED_SEARCH_ADDRESS')
        key=(request['battle_id'],request['owner'],request['stream_id'])
        address=(key,request['decision_index'],request['prefix_sha256'])
        if address in self.seen:raise worlds.Refusal('SEARCH_REQUEST_ALREADY_ATTEMPTED')
        self.seen.add(address)
        lines=tuple(request['lines'])
        if any(type(x) is not str or not x.startswith('|') or '\n' in x or '\r' in x for x in lines):
            raise worlds.Refusal('LITERAL_OWNING_PREFIX_REQUIRED')
        digest=hashlib.sha256(b''.join(x.encode()+b'\n' for x in lines)).hexdigest()
        requests=[x for x in lines if x.startswith('|request|')]
        if digest!=request['prefix_sha256'] or not requests or requests[-1]!=request['own_request_line']:
            raise worlds.Refusal('SEARCH_PREFIX_OR_REQUEST_MISMATCH')
        owner=int(request['owner'][1])-1
        previous=None if self.saved_root_admission else self.sessions.get(key)
        if previous is None:
            adapter=JaxRecordedAdapter(self.catalog,owner_side=owner,compiled_functions=self.functions)
            adapter.reset(stream_id=request['stream_id'],battle_id=request['battle_id'])
            old=()
        else:adapter,old=previous
        if lines[:len(old)]!=old or len(lines)<=len(old):
            raise worlds.Refusal('SEARCH_PREFIX_NOT_ORIGINAL_SUCCESSOR')
        # Same original packet validation and public fold, with no model.
        # Session state is committed only on successful own-line ingestion.
        peer='p2' if owner==0 else 'p1'
        for line in lines[len(old):]:
            if line.startswith('|split|'):raise worlds.Refusal('OMNISCIENT_SPLIT_FORBIDDEN')
            if line.startswith('|request|'):_packet(line,request['owner'])
            pieces=line.split('|');health=[]
            if len(pieces)>4 and pieces[1] in ('switch','drag','replace'):health=[(pieces[2],pieces[4])]
            elif len(pieces)>3 and pieces[1] in ('-damage','-heal','-sethp'):
                health=[(pieces[2],pieces[3])]
                if pieces[1]=='-sethp' and len(pieces)>5:health.append((pieces[4],pieces[5]))
            if any(ident.startswith(peer) and '/' in hp and hp.split()[0].split('/')[1]!='100' for ident,hp in health):
                raise worlds.Refusal('TRUE_OPPONENT_HP_FORBIDDEN')
        adapter.feed(tuple(x for x in lines[len(old):] if not x.startswith('|request|')))
        if not self.saved_root_admission:self.sessions[key]=(adapter,lines)
        packet=_packet(request['own_request_line'],request['owner'])
        root=worlds.from_b2(adapter,packet,self.source,self.metadata,
            address=DecisionAddress(request['stream_id'],request['decision_index'],owner))
        pending.root=root
        pending.worldset=worlds.sample_worlds(root,self.source,self.tables,count=self.count,seed=self.seed)
        if not pending.worldset.fully_constructed:
            raise worlds.Refusal('WORLD_CONSTRUCTOR_REFUSED',';'.join(sorted({r for w in pending.worldset.worlds for r in w.refusals})))
        pending.legal=tuple(map(bool,adapter.encode(packet,owner_side=owner)['legal_action_mask']))
        pending.reports=[]

    def _forward_rows(self, rows):
        # One actual GPU invocation per bounded flattened batch; rows may come
        # from many concurrent panel pairs and many posterior worlds.
        output=[]
        for start in range(0,len(rows),self.lanes):
            chunk=rows[start:start+self.lanes];n=len(chunk)
            padded=chunk+[chunk[0]]*(self.lanes-n)
            inputs={k:j.concatenate([r[k] for r in padded],axis=0) for k in padded[0]}
            result=jax.block_until_ready(self.forward(inputs));self.batch_calls+=1
            logits,value=map(_host,(result.policy_logits,result.value))
            for i in range(n):
                if not np.isfinite(logits[i]).all() or not np.isfinite(value[i]) or abs(float(value[i]))>1:
                    raise worlds.Refusal('NONFINITE_CHAMPION_EVALUATOR')
                output.append((tuple(map(float,logits[i])),float(value[i])))
        return output

    def process_many(self, requests):
        """One scheduler batch shared across all current CPU actor/FP pairs."""
        pending=[Pending(r,time.monotonic()) for r in requests]
        for item in pending:
            try:self._prepare(item)
            except (ValueError,TypeError,KeyError,IndexError,AttributeError) as error:self._refuse(item,error)
        peer_rows=[];world_rows=[]
        for item in pending:
            if item.result is not None:continue
            try:
                for world in item.worldset.worlds:
                    owner=item.root.address.owner
                    mask=self.legal(world.state)
                    own=self.observe(world.state,world.public,j.asarray([owner],j.int32),mask[:,owner])
                    peer=self.observe(world.state,world.public,j.asarray([1-owner],j.int32),mask[:,1-owner])
                    if np.any(_host(own.observation.error)) or np.any(_host(peer.observation.error)):
                        raise worlds.Refusal('ROOT_PUBLIC_ENCODER_REFUSED')
                    if tuple(map(bool,_host(own.observation.legal_action_mask)[0]))!=item.legal:
                        raise worlds.Refusal('OWN_REQUEST_PHYSICAL_LEGALITY_DIFFERS')
                    # Reply policy's private history is explicitly unknown as
                    # in D; actual opponent exact supplemental inputs are absent.
                    peer_rows.append(peer.inputs);world_rows.append((item,world,own,peer))
            except (ValueError,TypeError,KeyError,IndexError,AttributeError) as error:self._refuse(item,error)
        try:peer_outputs=self._forward_rows(peer_rows) if peer_rows else []
        except (ValueError,TypeError,KeyError,IndexError,AttributeError) as error:
            for item in pending:
                if item.result is None:self._refuse(item,error)
            peer_outputs=[]
        branches=[]
        for (item,world,own,peer),(logits,_) in zip(world_rows,peer_outputs):
            if item.result is not None:continue
            try:
                owner=item.root.address.owner
                peer_legal=tuple(map(bool,_host(peer.observation.legal_action_mask)[0]))
                requested=bool(_host(request_seats(world.state))[0,1-owner])
                replies=tuple(i for i,v in enumerate(peer_legal) if v) if requested else (0,)
                policy=probabilities(logits,peer_legal) if requested else (1.,)*9
                report=dict(replies=replies,probabilities=tuple(policy[r] for r in replies),pairs={})
                item.reports.append(report)
                for a,active in enumerate(item.legal):
                    if not active:continue
                    for r in replies:
                        actions=j.full((1,2),-1,j.int32)
                        from . import engine_device_observation as observation
                        actions=actions.at[0,owner].set(observation.map_actions(own.observation,world.state,j.asarray([owner],j.int32),j.asarray([a],j.int32))[0])
                        if requested:
                            actions=actions.at[0,1-owner].set(observation.map_actions(peer.observation,world.state,j.asarray([1-owner],j.int32),j.asarray([r],j.int32))[0])
                        plan=chance.plan_for_state(world.state,0,tuple(map(int,_host(actions)[0])),self.engine_tables,self.metadata,cap=self.cap,exact_move_slots=True)
                        if not isinstance(plan,chance.ConsumerPlan):
                            raise worlds.Refusal('EXACT_CHANCE_PLAN_REFUSED',';'.join(g.code+':'+g.detail[:160] for g in getattr(plan,'gaps',())))
                        item.required_chance_branches+=plan.cardinality
                        if item.required_chance_branches>self.cap:
                            raise worlds.Refusal('EXACT_CHANCE_DECISION_CAP',f'required_so_far={item.required_chance_branches};cap={self.cap};no partial reduction')
                        cases=plan.bounded_cases();report['pairs'][(a,r)]=[]
                        for case in cases:
                            branches.append((item,world,owner,actions,case,report['pairs'][(a,r)]))
            except (ValueError,TypeError,KeyError,IndexError,AttributeError) as error:self._refuse(item,error)
        branches=[row for row in branches if row[0].result is None]
        for start in range(0,len(branches),self.lanes):
            chunk=branches[start:start+self.lanes]
            # Failed roots keep their slots; no branch is used in a reduction
            # unless every declared world/reply/chance branch completed.
            try:
                values=[]
                for _,_,_,_,case,_ in chunk:
                    v=dict(case.values)
                    values.append(transition.ChanceInputs(j.asarray([[[v[f'draws.{s}.{k}'] for k in range(12)] for s in range(2)]],j.int32),
                        **{name:j.asarray([[v[f'{name}.{s}'] for s in range(2)]],j.int32) for name in ('phazing','secondary','sleep_talk','entry_ties')},
                        initial_tie=j.asarray([v['initial_tie']],j.int32)))
                padded=chunk+[chunk[0]]*(self.lanes-len(chunk));values+= [values[0]]*(self.lanes-len(values))
                stack=lambda rows:jax.tree.map(lambda *x:j.stack(x),*rows)
                result=jax.block_until_ready(self.advance(stack([r[1].state for r in padded]),stack([r[1].public for r in padded]),
                    j.stack([r[3] for r in padded]),stack(values),j.asarray([[r[2]] for r in padded],j.int32)))
                invalid,done,winner,error,inputs=result
                evaluated=self._forward_rows([{k:v[i] for k,v in inputs.items()} for i in range(len(chunk))])
                for i,(item,world,owner,actions,case,prices) in enumerate(chunk):
                    if bool(_host(invalid[i])[0]) or int(_host(error[i])[0]):
                        self._refuse(item,worlds.Refusal('REQUIRED_EXACT_CHANCE_BRANCH_REFUSED'));continue
                    terminal=bool(_host(done[i])[0]);win=int(_host(winner[i])[0])
                    if terminal and win not in (0,1,2):
                        self._refuse(item,worlds.Refusal('TERMINAL_WINNER_UNAVAILABLE'));continue
                    value=(0. if win==2 else 1. if win==owner else -1.) if terminal else evaluated[i][1]
                    prices.append((case.mass,value))
            except (ValueError,TypeError,KeyError,IndexError,AttributeError) as error:
                for item,*_ in chunk:
                    if item.result is None:self._refuse(item,error)
        for item in pending:
            if item.result is not None:continue
            try:
                q,action=completed_q(item.legal,item.reports,requested_worlds=self.count)
                item.result=dict(status='COMPLETE',action_index=action,q_values=q,refusal=None,
                    requested_worlds=self.count,constructed_worlds=self.count,archive_sha256=self.archive_sha256,
                    decision_seconds=time.monotonic()-item.started)
            except (ValueError,TypeError,KeyError,IndexError,AttributeError) as error:self._refuse(item,error)
        return [dict(request_id=item.request.get('request_id'),schema=SCHEMA,**item.result) for item in pending]

    def _refuse(self,item,error):
        item.result=dict(status='REFUSED',action_index=None,q_values=None,refusal=str(error),
            requested_worlds=self.count,constructed_worlds=0 if item.worldset is None else sum(w.state is not None for w in item.worldset.worlds),
            archive_sha256=self.archive_sha256,decision_seconds=None)
