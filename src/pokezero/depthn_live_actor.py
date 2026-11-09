"""DepthN live actor over the unchanged controlled B2 Foul Play owning path.

Per decision: at most one bounded search request, sent as soon as the snapshot
is encoded (on a helper thread, whose socket wait uses no CPU), while the
champion's greedy action is computed on the actor's own core; the decision waits
for both, so a legal base action always exists. Search's critical path therefore
carries the encode but not the greedy forward (the lead's actor-time check: the
greedy forward is about 0.5 s on one pinned core). Any refusal, client deadline,
transport failure, illegal action or bad Q plays that greedy action with one
counted cause. A single-legal decision is never searched. No retries,
ever. Arm RAW is the same actor with search disabled: greedy only, no client.
The budget clock starts when the owning snapshot reaches encode. Import loads no
model, JAX program or GPU stack; this module never imports gpu_engine.

Played Q: the server's q is an equal-weight mean of world values in [-1, 1], so
a forced win or loss is exactly +-1 or within a few ulp of it (fsum rounding).
A searched action is played only if its q (int or float, never bool) is finite
and |q| <= 1 + 1e-9.

Own recharge (the lead's decision): the GPU engine has no recharge or locked-move
state, so its worlds price a recharge move (RECHARGE_MOVES) as if its user acted
freely on the next turn. A search override is never played INTO one: when the
searched action is a move lane whose own move (own_moves, from the owner's
|request| line) is a recharge move and greedy's action is not, greedy is played,
source greedy_fallback, cause OWN_RECHARGE_UNREPRESENTED; the row keeps the
COMPLETE response and names the guarded lane (recharge_guard). An override AWAY
from a recharge move greedy chose stays allowed. The recharge turn itself is
single-legal and never searched.

Unrepresented effects (the lead's decision): a COMPLETE response carries the
server's effect guard (pokezero.depthn_effect_guard: the guard record, with a
bool `tripped`; the client refuses a COMPLETE response without one,
server_error). When it tripped, greedy is played, source greedy_fallback, cause
UNREPRESENTED_EFFECT, the full response kept. The opponent-recharge refusal
(the server refuses the root) and the own-recharge guard stay as they are: when
several guards trip the first cause stands (OWN_RECHARGE_UNREPRESENTED before
UNREPRESENTED_EFFECT) and guard_causes lists every guard that tripped.

Against design note section 6: greedy logits/probabilities are NOT logged. The
champion's raw policy is pinned with record_policy_distribution=False, and
neither the observation nor the encode path carries them. Each row keeps what an
offline recompute needs: battle_id, the battle turn (null before the first
|turn| line), decision_index and the champion archive_sha256 (the server's on
server responses, else the pinned champion archive), plus the greedy action,
behavior probability, value estimate and timings, and the client's deadline and reserve (None on arm RAW). A row that cannot be written
because of server content is replaced by a minimal server_error row and the
greedy decision is played: select() never raises on anything the server sends.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import socket
import struct
import time
import uuid

from . import phase_d_search_server as wire
from .live_protocol_actor import ActorContract,RawPolicy
from .live_protocol_bridge import OwningActorFactory,require_actor_factory
from .phase_d_ladder_panel import encoded
from .policy import PolicyDecision

SEARCH_ARMS=('DN2W8','DN2W16')
ARMS=('RAW',)+SEARCH_ARMS
POLICY_IDS={'RAW':'champion-raw-ours',**{arm:'champion-depthn-'+arm for arm in SEARCH_ARMS}}
ROW_SCHEMA='depthn-live.decision.v1'
SERVER_CAUSES=('constructor_refused','depthn_refused','expired_in_queue','server_error','bad_request','class_refused')
# The certified bf16 champion archive (gpu_engine/declared_bounds/champion-9375-bf16-*.json).
CHAMPION_ARCHIVE_SHA256='e450515351bce1497ab50a712955fe17b761bbb5f424b23ec299e96cda5b165c'
OWN_RECHARGE='OWN_RECHARGE_UNREPRESENTED'
CAUSES=SERVER_CAUSES+('timeout','connection_error','illegal_action','bad_q',OWN_RECHARGE,'UNREPRESENTED_EFFECT')
SOURCES=('search','greedy_fallback','single_legal','raw_arm')
REQUEST_KINDS=('move','switch','forced')
Q_BOUND=1.+1e-9
clock=time.monotonic


class _Refused(Exception):
    def __init__(self,cause,message):super().__init__(message);self.cause=cause


class Client:
    """One connection per request to ONE address resolved here, never per request.

    Connect, send and every read get the remaining budget as their timeout.
    Anything the server sends maps to a REFUSED cause; request() never raises
    on server content.
    """
    def __init__(self,host,port,*,rung,deadline_seconds=3.,reserve_seconds=.15):
        if (rung not in SEARCH_ARMS or type(host)is not str or not host or type(port)is not int or not 0<port<65536
                or any(type(x)not in (int,float)or not math.isfinite(x)for x in (deadline_seconds,reserve_seconds))
                or not 0<deadline_seconds<=60 or not 0<=reserve_seconds<deadline_seconds):
            raise ValueError('declared DepthN rung and bounded client deadline required')
        # A slow lookup or a second dead address must never spend a decision's budget:
        # resolve here once and connect to ONE address (the first IPv4 one, if any).
        found=socket.getaddrinfo(host,port,type=socket.SOCK_STREAM)
        self.family,_,_,_,self.address=next((x for x in found if x[0]==socket.AF_INET),found[0])
        self.host=host;self.rung=rung;self.deadline=float(deadline_seconds);self.reserve=float(reserve_seconds)

    def budget(self,started):
        """Deadline from the decision's start (encode), less elapsed and the reserve."""
        return self.deadline-(clock()-started)-self.reserve

    def request(self,snapshot,*,decision_index,started):
        budget=self.budget(started)
        request=dict(schema=wire.SCHEMA,request_id=uuid.uuid4().hex,battle_id=snapshot.battle_id,owner=snapshot.owner,
            stream_id=snapshot.stream_id,decision_index=decision_index,generation=snapshot.generation,
            prefix_sha256=snapshot.prefix_sha256,lines=list(snapshot.lines),own_request_line=snapshot.own_request_line,
            rung=self.rung,budget_seconds=budget)
        sent=clock()
        try:
            if not budget>0:raise TimeoutError('no client budget left after the greedy decision')
            try:packet=wire.encode(request)
            except (ValueError,TypeError)as error:raise _Refused('bad_request',str(error))
            response=self._exchange(packet,sent,budget);self._check(response,request)
            response=dict(response,client_refused=False)
        except _Refused as error:response=self._refused(request,error.cause,error)
        except TimeoutError as error:response=self._refused(request,'timeout',error)
        except OSError as error:response=self._refused(request,'connection_error',error)
        except Exception as error:response=self._refused(request,'server_error',error)
        return dict(response,budget_seconds=budget,client_wall_seconds=clock()-sent)

    def _exchange(self,packet,sent,budget):
        def remaining():
            left=budget-(clock()-sent)
            if left<=0:raise TimeoutError('client search deadline reached; no retry')
            return left
        with socket.socket(self.family,socket.SOCK_STREAM)as stream:
            stream.settimeout(remaining());stream.connect(self.address)
            stream.settimeout(remaining());stream.sendall(packet)
            def exact(n):
                output=bytearray()
                while len(output)<n:
                    stream.settimeout(remaining());piece=stream.recv(n-len(output))
                    if not piece:raise _Refused('server_error','search server closed without a response; no retry')
                    output.extend(piece)
                return bytes(output)
            length=struct.unpack('!I',exact(4))[0]
            if not 0<length<=wire.MAX_PACKET:raise ValueError('bounded search response required')
            return json.loads(exact(length))

    def _check(self,response,request):
        if (type(response)is not dict or response.get('schema')!=wire.SCHEMA
                or response.get('request_id')!=request['request_id']or response.get('rung')!=self.rung):
            raise ValueError('same schema/request_id/rung response required')
        status,cause=response.get('status'),response.get('cause')
        if not (status=='COMPLETE'and cause is None or status=='REFUSED'and cause in SERVER_CAUSES):
            raise ValueError('literal status and declared refusal cause required')
        guard=response.get('effect_guard')
        if status=='COMPLETE'and(type(guard)is not dict or type(guard.get('tripped'))is not bool):
            raise ValueError('a COMPLETE response carries the effect guard record (bool tripped)')
        # Finite JSON only (NaN, Infinity and 1e999 never reach play or telemetry), checked
        # at the depth the decision row will hold it: the row nests the response once.
        encoded(dict(search=response))

    def _refused(self,request,cause,error):
        return dict(schema=wire.SCHEMA,request_id=request['request_id'],rung=self.rung,status='REFUSED',
            action_index=None,q_values=None,refusal=(type(error).__name__+': '+str(error))[:1000],cause=cause,
            requested_worlds=None,constructed_worlds=None,decision_seconds=None,client_refused=True)


def request_kind(legal_action_mask):
    """forced: one legal action; switch: no move legal (a forceSwitch request); else move."""
    legal=tuple(map(bool,legal_action_mask))
    if len(legal)!=9 or not any(legal):raise ValueError('nine-action legal mask required')
    if sum(legal)==1:return 'forced'
    return 'move'if any(legal[:4])else'switch'


def searched_action(response,observation):
    """(action, None) only for a COMPLETE legal action with finite |q| <= Q_BOUND.

    q may be an int or a float, never a bool; a huge JSON integer is compared
    exactly (no float conversion), so it is refused without overflow.
    """
    if response['status']!='COMPLETE':return None,response['cause']
    action=response.get('action_index');q=response.get('q_values')
    if type(action)is not int or not 0<=action<9 or not observation.legal_action_mask[action]:return None,'illegal_action'
    if type(q)is not list or len(q)!=9:return None,'bad_q'
    value=q[action]
    if (type(value)not in (int,float)or type(value)is float and not math.isfinite(value)
            or not abs(value)<=Q_BOUND):return None,'bad_q'
    return action,None


# RECHARGE_MOVES, own_moves, recharge_lane and recharge_guarded live in the shared guard module (depthn_effect_guard), the
# one the re-pricer and the RR scorer import too; they are re-exported here under their old names.
from .depthn_effect_guard import CAUSE as EFFECT_GUARD,RECHARGE_MOVES,own_moves,recharge_guarded,recharge_lane  # noqa: E402,F401


def battle_turn(lines):
    """The last |turn|N in the owner's lines; None before the first turn line."""
    for line in reversed(lines):
        if line.startswith('|turn|'):
            value=line.split('|')[2]
            return int(value)if value.isascii()and value.isdigit()else None
    return None


def factory(config,*,raw_factory,client,telemetry_path,arm,cell=None,archive_sha256=CHAMPION_ARCHIVE_SHA256):
    if (type(raw_factory)is not OwningActorFactory or raw_factory.contract!=ActorContract.champion()or arm not in ARMS
            or (client is not None if arm=='RAW'else type(client)is not Client or client.rung!=arm)
            or type(archive_sha256)is not str or len(archive_sha256)!=64 or any(c not in '0123456789abcdef'for c in archive_sha256)):
        raise ValueError('champion raw factory, matching DepthN arm/client and champion archive required')
    require_actor_factory(config,raw_factory)
    path=Path(telemetry_path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb'):pass  # One fresh decision log per game; never appended across cells.
    cell=None if cell is None else json.loads(encoded(cell))
    policy_id=POLICY_IDS[arm];pending={}
    def emit(blob):
        with path.open('ab')as stream:stream.write(blob+b'\n');stream.flush();os.fsync(stream.fileno())
    def backend(contract,*,env_config,**identity):
        pending.clear();pending['index']=0
        provider=raw_factory.backend_factory(contract,env_config=env_config,**identity)
        encode=provider.encode
        def capture(snapshot):
            started=clock();output=encode(snapshot);pending.update(snapshot=snapshot,started=started);return output
        provider.encode=capture
        return provider
    def policy(contract,*,model,result):
        fallback=raw_factory.policy_factory(contract,model=model,result=result)
        searcher=None if client is None else ThreadPoolExecutor(1,thread_name_prefix='depthn-search')
        class LivePolicy(RawPolicy):
            def claim(self,c):fallback.claim(c);super().claim(c)
            def close(self):
                try:fallback.close()
                finally:
                    if searcher is not None:searcher.shutdown(wait=False)
        def select(observation,*,rng):
            entered=clock();snapshot=pending.pop('snapshot',None);started=pending.pop('started',None)
            if snapshot is None or started is None or observation.metadata.get('owner_prefix_sha256')!=snapshot.prefix_sha256:
                raise ValueError('same current B2 owning root required')
            index=pending['index'];pending['index']+=1;kind=request_kind(observation.legal_action_mask)
            # The search request leaves now (its budget counts from encode), the greedy forward runs meanwhile.
            searching=None if client is None or kind=='forced'else searcher.submit(client.request,snapshot,decision_index=index,started=started)
            greedy=fallback.select_action(observation,rng=rng);greedy_seconds=clock()-entered
            response=action=cause=guard=None;guards=[]
            if client is None:source='raw_arm'
            elif kind=='forced':source='single_legal'
            else:
                response=searching.result()
                action,cause=searched_action(response,observation)
                if action is not None:
                    moves=own_moves(snapshot.own_request_line);base=int(greedy.action_index)
                    if recharge_guarded(moves,action,base):
                        guard=dict(searched_action=action,searched_move=moves[action],greedy_move=moves[base]if 0<=base<4 else None)
                        guards.append(OWN_RECHARGE)
                    if (response.get('effect_guard')or {}).get('tripped')is True:guards.append(EFFECT_GUARD)
                    if guards:action,cause=None,guards[0]   # the first cause stands; guard_causes keeps them all
                source='search'if action is not None else'greedy_fallback'
            if action is not None:
                decision=PolicyDecision(action,policy_id,action_probability=1.,value_estimate=float(response['q_values'][action]),
                    metadata=dict(depthn_live=dict(arm=arm,source=source,cause=None,request_id=response['request_id'])))
            else:
                decision=replace(greedy,metadata={**dict(greedy.metadata),'depthn_live':dict(arm=arm,source=source,cause=cause,
                    request_id=None if response is None else response['request_id'])})
            served=response is not None and response.get('client_refused')is False
            row=dict(schema=ROW_SCHEMA,arm=arm,policy_id=policy_id,cell=cell,battle_id=snapshot.battle_id,owner=snapshot.owner,
                stream_id=snapshot.stream_id,generation=snapshot.generation,turn=battle_turn(snapshot.lines),decision_index=index,
                request_kind=kind,prefix_sha256=snapshot.prefix_sha256,
                archive_sha256=response.get('archive_sha256')if served else archive_sha256,archive_source='server'if served else'pinned',
                deadline_seconds=None if client is None else client.deadline,reserve_seconds=None if client is None else client.reserve,
                encode_seconds=entered-started,
                greedy=dict(action_index=greedy.action_index,policy_id=greedy.policy_id,action_probability=greedy.action_probability,
                    value_estimate=greedy.value_estimate,seconds=greedy_seconds),
                search=response,played_action_index=decision.action_index,source=source,cause=cause,
                search_differs=None if action is None else action!=greedy.action_index,recharge_guard=guard,guard_causes=guards,
                select_seconds=clock()-started)
            try:blob=encoded(row)
            except Exception as error:
                if response is None:raise
                # Server content the row cannot hold: a minimal server_error row, and greedy is played.
                rid=response.get('request_id');rid=rid if type(rid)is str else None
                response=dict(schema=wire.SCHEMA,request_id=rid,status='REFUSED',cause='server_error',client_refused=True,
                    refusal=('telemetry row: '+type(error).__name__+': '+str(error))[:1000],
                    **{k:response.get(k)if type(response.get(k))is float else None for k in ('budget_seconds','client_wall_seconds')})
                source,cause='greedy_fallback','server_error'
                decision=replace(greedy,metadata={**dict(greedy.metadata),'depthn_live':dict(arm=arm,source=source,cause=cause,request_id=rid)})
                row.update(search=response,played_action_index=decision.action_index,source=source,cause=cause,search_differs=None,
                    recharge_guard=None,guard_causes=[],archive_sha256=archive_sha256,archive_source='pinned')
                blob=encoded(row)
            emit(blob)
            return decision
        return LivePolicy(contract,select,fallback.reset)
    value=OwningActorFactory('jax-owning',raw_factory.contract,backend,policy)
    from .owning_worker_runtime import register_factory
    # The champion's raw factory remains the owner of the shared compiled B2 functions.
    register_factory(value,lambda:{})
    return require_actor_factory(config,value)


class Runtime:
    """One declared cell through ChampionWorkerRuntime and the original bridge."""
    def __init__(self,*,client,telemetry_path,arm,cell,original=None,archive_sha256=CHAMPION_ARCHIVE_SHA256):
        from .owning_worker_runtime import ChampionWorkerRuntime
        if (arm not in ARMS or (client is None)!=(arm=='RAW')or type(cell)is not dict
                or any(type(cell.get(k))is not int for k in ('pair_ordinal','seed','seat','fp_seed'))or cell['seat']not in (0,1)):
            raise ValueError('declared arm, matching client and literal cell required')
        self.original=ChampionWorkerRuntime()if original is None else original
        self.client=client;self.telemetry_path=telemetry_path;self.arm=arm;self.cell=cell;self.factory=None
        self.archive_sha256=archive_sha256

    async def run(self,config):
        from . import foulplay_bridge as bridge
        cell=self.cell
        if self.factory is not None:raise ValueError('one game per DepthN runtime; no replay')
        if (config.games!=1 or config.seed_start!=cell['seed']or config.foulplay_random_seed!=cell['fp_seed']
                or config.pokezero_player!=('p1','p2')[cell['seat']]):
            raise ValueError('game configuration differs from the declared cell')
        original=self.original.prepare(config)
        self.factory=factory(config,raw_factory=original,client=self.client,telemetry_path=self.telemetry_path,arm=self.arm,cell=cell,
            archive_sha256=self.archive_sha256)
        model,result,env,spec,vocab,dex,masks,digest=self.original.runtime
        policy_id=POLICY_IDS[self.arm]
        policy=bridge._build_policy(config=config,model=model,result=result,value_model=model,value_result=result,
            env_config=env,rollout_config=bridge.RolloutConfig(max_decision_rounds=config.max_decision_rounds,
            format_id=config.format_id),policy_id=policy_id)
        return await bridge._run_controlled_foulplay_games(config,policy=policy,policy_id=policy_id,
            vocab=vocab,dex=dex,observation_spec=spec,feature_masks=masks,checkpoint_sha256=digest,
            owning_actor_factory=self.factory,actor_runtime=(model,result,env))
