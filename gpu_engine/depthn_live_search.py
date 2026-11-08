"""DepthN depth-2 live search: Codex's ladder search-server requests answered over ProtocolConstructor worlds (gap 5).

Design. A DepthNLiveSearch owns one ProtocolConstructor (W = 8 or 16 worlds: rung DN2W8 / DN2W16), one DepthNPricer per
capacity bucket of roots (make_pricers: depth 2, full width k 9 and k1 9/9, `prior`, fixed_capacity, the champion as
forward=, one shared program table) and the champion bank. phase_d_search_server.Server's single queue consumer calls
process_many(requests) on its one executor thread with at most max_requests requests and checks the responses' order and
request_id. Per batch (process_many chunks a longer list into batches of max_requests):
- validate: exactly Codex's ten ladder keys (what ProtocolConstructor.prepare accepts) plus ours, `rung` (this server's)
  and `budget_seconds` (finite > 0), and the server queue's private ARRIVAL stamp; anything else is REFUSED bad_request.
- expired_in_queue: a request whose queue wait (ARRIVAL -> the batch's start, both time.monotonic; 0 without a stamp)
  leaves less than min_price_seconds of its budget is refused unpriced and unprepared (its address stays unattempted).
- the rest, stripped to Codex's keys: constructor_worlds.prepare_roots (CPU device) -> build (only_complete: Codex's all-W
  rule refuses a root with any refused world before any device work) -> ONE price call on the smallest bucket holding the
  batch's blocks -> reduce (learner_max backups; the K blocks composed by the equal-weight mean of completed-Q). DepthN's
  capacity counts requests, one per block of two worlds (count > capacity raises; n = 2 x capacity world slots), so the
  bucket of b roots is a pricer of capacity b x K, K = W/2.
- per root: COMPLETE plays `action_mean_q` (ARGMAX_MEAN_COMPLETED_Q, the lead's decision: argmax over the legal actions of
  the mean completed-Q, lower action on ties; constructor_worlds.root_actions) and reports `action_policy_target`, never
  played. A root prepare() refused, or one with a refused world, is constructor_refused; a refusal by build's root checks
  (root view, legal mask, root forward) or by the reduction (a DepthN block cause) is depthn_refused. The reduction runs
  root by root: reduce_completed_q's range check raises ValueError when a block's completed-Q rounds outside [-1, 1]
  (an action that wins, or loses, against every reply in both worlds of a block, its reply prior summing to 1 + ulp),
  and constructor_worlds.reduce does not catch it; here it refuses that root only (depthn_refused, REDUCTION_ERROR).
  Any other exception in the shared batch (prepare_roots, build, price, reduce) refuses every request of the batch
  server_error with its text; MemoryError propagates. Every response must pass phase_d_search_server.encode (else
  server_error), so no value breaks the wire.
Response, per the live contract: schema, request_id, rung, archive_sha256, status, action_index, q_values (9, None on
illegal lanes), refusal, cause, requested_worlds, constructed_worlds, decision_seconds (batch start -> result, COMPLETE
only); ours: root_operator, policy_target, root_value, action_policy_target, block_causes, world_refusals, timings
(queue_seconds, prepare_seconds (the root's own), build/price/reduce_seconds (the shared batch's)), batch_size, bucket,
tree_commit, pricer. Fields a request never reached are None.
Sessions: Codex's constructor keeps `sessions` (the incremental protocol fold per (rung, battle, owner, stream)) and `seen`
(every attempted address) forever. Here the battles least recently sent to prepare beyond session_cap lose both. A battle
of the current batch is never evicted (session_cap >= max_requests), so a live battle keeps its fold and its no-retry
rule; an evicted battle that returns folds its full prefix again from a fresh adapter (the same state, more host time).
warmup(requests): per bucket, the full path on saved Showdown requests forced to that bucket (cold: compiles P1 and the
selection per capacity, the gather per pool, the shared leaf block program, the root forward), then once more (warm).
Session state is cleared around every pass. The server is READY only after it.
"""
from collections import OrderedDict
from dataclasses import replace
import math
import time

from pokezero.phase_d_search_server import SCHEMA,encode
from . import champion_evaluator as evaluators,constructor_worlds as cw,device_depthn_pricer as dn

RUNGS={'DN2W8':8,'DN2W16':16}
ROOT_OPERATOR='ARGMAX_MEAN_COMPLETED_Q'
CAUSES=('constructor_refused','depthn_refused','expired_in_queue','server_error','bad_request')
PROTOCOL_KEYS=('schema','request_id','battle_id','owner','stream_id','decision_index','generation','prefix_sha256','lines','own_request_line')
OURS=('rung','budget_seconds')
ARRIVAL='_arrival_monotonic'   # stamped by the server's queue (time.monotonic at arrival); a client's value is overwritten
STAGES=('queue_seconds','prepare_seconds','build_seconds','price_seconds','reduce_seconds')
DEPTH=2;WIDTH=9
FATAL=(MemoryError,)
WARMUP_BUDGET=3600.


def make_pricers(executor,forward,params,*,worlds,buckets=(1,2,4),block_lanes=2187,blocks_per_call=16):
    """The live pricers: {bucket b: DepthNPricer(capacity b x W/2, depth 2, k 9, k1 9/9, prior, fixed_capacity)} with `forward`
    (the champion) and one program table shared after binding `params` (block programs compile once, P1 once per bucket)."""
    if forward is None:raise ValueError('an explicit evaluator (the champion) required')
    table={};out={}
    for b in buckets:
        p=dn.DepthNPricer(executor,None,capacity=b*(worlds//2),depth=DEPTH,k_own=WIDTH,k_opp=WIDTH,k1_own=WIDTH,k1_opp=WIDTH,block_lanes=block_lanes,
            blocks_per_call=blocks_per_call,selection='prior',fixed_capacity=True,forward=forward,actor_capacity=len(params))
        p.bind(params);p.programs=table;out[b]=p
    return out


def refusal_cause(reason):
    """constructor_refused for a root prepare() refused or with a refused world (Codex's all-W rule), else depthn_refused."""
    return('constructor_refused'if reason.startswith(('constructor-root:','WORLD_CONSTRUCTOR_REFUSED'))or reason=='materialization'
        else'depthn_refused')


def _battle(key):
    """The battle of a ProtocolConstructor session key (rung, battle, owner, stream), else None."""
    return key[1]if type(key)is tuple and len(key)==4 else None


def _real(x):
    return type(x)in(int,float)and math.isfinite(x)


class DepthNLiveSearch:
    def __init__(self,*,constructor,pricers,params,binding,recipe,tree_commit,min_price_seconds=1.,max_requests=None,session_cap=256,
            archive_sha256=None,device='cpu',clock=time.monotonic):
        """`pricers`: make_pricers' {bucket: DepthNPricer}; `binding`: the pricer side's WorldBinding (constructor_worlds.binding of
        the executor's source, engine metadata and catalog); `recipe`: the lane-order Recipe; `clock` must be the server's
        arrival clock (time.monotonic)."""
        W=constructor.count;self.rung=next((r for r,w in RUNGS.items()if w==W),None)
        if self.rung is None:raise ValueError(f'a constructor of W in {tuple(RUNGS.values())} worlds required, got {W!r}')
        self.W,self.K=W,W//2;self.buckets=tuple(sorted(pricers))
        if not self.buckets or any(type(b)is not int or b<=0 for b in self.buckets):raise ValueError('positive integer capacity buckets required')
        first=pricers[self.buckets[0]]
        for b in self.buckets:
            p=pricers[b]
            if(type(p)is not dn.DepthNPricer or p.depth!=DEPTH or p.selection!='prior'or p.fixed_capacity is not True or p.capacity!=b*self.K
                    or p.k_own!=(WIDTH,)or p.k_opp!=(WIDTH,)or(p.k1_own,p.k1_opp)!=(WIDTH,WIDTH)or p.executor is not first.executor
                    or p.forward is not first.forward):
                raise ValueError(f'bucket {b}: a fixed-capacity DepthNPricer of capacity {b}x{self.K}, depth 2, k 9, k1 9/9, prior, sharing the executor and evaluator required')
        self.forward=first.forward;ident=evaluators.identity(self.forward)
        if ident[0]!='champion':raise ValueError('the champion evaluator required (forward=)')
        max_requests=self.buckets[-1]if max_requests is None else max_requests
        if type(max_requests)is not int or not 1<=max_requests<=self.buckets[-1]:raise ValueError(f'max_requests in 1..{self.buckets[-1]} (the largest bucket) required')
        if type(session_cap)is not int or session_cap<max_requests:raise ValueError('a session cap of at least max_requests battles required')
        if not _real(min_price_seconds)or min_price_seconds<0:raise ValueError('finite min_price_seconds >= 0 required')
        if type(tree_commit)is not str or not tree_commit:raise ValueError('the tree commit required')
        if type(params)is not tuple or not params:raise ValueError('the champion bank (params,) required')
        if cw.constructor_binding(constructor)!=binding:raise ValueError('the constructor builds under another binding than the pricer\'s')
        if isinstance(self.forward,evaluators.ChampionEvaluator):evaluators.check_binding(self.forward,constructor.catalog)
        self.constructor,self.pricers,self.params,self.binding,self.recipe=constructor,dict(pricers),params,binding,recipe
        self.tree_commit,self.min_price_seconds,self.max_requests,self.session_cap=tree_commit,float(min_price_seconds),max_requests,session_cap
        self.archive_sha256=archive_sha256 or getattr(self.forward,'archive_sha256',None)or ident[1]
        self.device,self.clock=device,clock
        self.battles=OrderedDict();self.evicted=0;self.warm=None
        self.pricer_identity=dict(engine='DepthNPricer',depth=DEPTH,k=[WIDTH,WIDTH],k1=[WIDTH,WIDTH],selection='prior',W=W,K=self.K,
            buckets=list(self.buckets),capacities=[b*self.K for b in self.buckets],fixed_capacity=True,block_lanes=first.block_lanes,
            blocks_per_call=first.blocks_per_call,precision=getattr(self.forward,'precision',None)or(ident[2]if len(ident)>2 else None),
            evaluator=evaluators.describe(self.forward)['identity'],world_source=cw.SOURCE,backup='learner_max',root_operator=ROOT_OPERATOR)

    @property
    def programs(self):
        """Every compiled program of the live pricers (the union of their tables; one shared table from make_pricers)."""
        return{k for p in self.pricers.values()for k in p.programs}

    # Requests -----------------------------------------------------------------------------------------------------------------------
    def _response(self,request_id,n):
        return dict(schema=SCHEMA,request_id=request_id,rung=self.rung,archive_sha256=self.archive_sha256,status='REFUSED',action_index=None,q_values=None,
            refusal=None,cause=None,requested_worlds=self.W,constructed_worlds=0,decision_seconds=None,root_operator=ROOT_OPERATOR,policy_target=None,
            root_value=None,action_policy_target=None,block_causes=None,world_refusals=None,timings=dict.fromkeys(STAGES),batch_size=n,bucket=None,
            tree_commit=self.tree_commit,pricer=dict(self.pricer_identity))

    @staticmethod
    def _problem(request):
        """Why a request is malformed (MALFORMED_REQUEST: ...), or None."""
        if not isinstance(request,dict):return'MALFORMED_REQUEST: a JSON object required'
        keys=set(request)-{ARRIVAL};want=set(PROTOCOL_KEYS)|set(OURS)
        if keys!=want:return'MALFORMED_REQUEST: missing '+','.join(sorted(want-keys))+'; unexpected '+','.join(sorted(map(str,keys-want)))
        if request['schema']!=SCHEMA:return'MALFORMED_REQUEST: schema'
        if type(request['request_id'])is not str or not request['request_id']:return'MALFORMED_REQUEST: request_id'
        if type(request['rung'])is not str:return'MALFORMED_REQUEST: rung'
        if not _real(request['budget_seconds'])or request['budget_seconds']<=0:return'MALFORMED_REQUEST: budget_seconds (finite > 0)'
        if ARRIVAL in request and not _real(request[ARRIVAL]):return'MALFORMED_REQUEST: arrival stamp'
        return None

    def process_many(self,requests):
        """One response per request, in order (module docstring); batches of at most max_requests."""
        requests=list(requests);out=[]
        for start in range(0,len(requests),self.max_requests):out.extend(self._batch(requests[start:start+self.max_requests]))
        return out

    def _batch(self,requests,*,bucket=None):
        started=self.clock();n=len(requests);out=[None]*n;todo=[]
        for i,request in enumerate(requests):
            rid=request.get('request_id')if isinstance(request,dict)else None
            row=self._response(rid if type(rid)is str else None,n);problem=self._problem(request)
            if problem is None and request['rung']!=self.rung:problem=f'RUNG_NOT_SERVED: {request["rung"]!r} (this server serves {self.rung})'
            if problem is not None:out[i]=dict(row,cause='bad_request',refusal=problem);continue
            wait=max(0.,started-request.get(ARRIVAL,started));row['timings']['queue_seconds']=wait
            if request['budget_seconds']-wait<self.min_price_seconds:
                out[i]=dict(row,cause='expired_in_queue',refusal=f'EXPIRED_IN_QUEUE: waited {wait:.3f}s of a {float(request["budget_seconds"]):.3f}s budget, '
                    f'under the {self.min_price_seconds:.3f}s minimum to price');continue
            todo.append((i,{k:request[k]for k in PROTOCOL_KEYS},row))
        if todo:
            try:done=self._search(todo,started,bucket)
            except FATAL:raise
            except Exception as error:done={i:dict(row,cause='server_error',refusal=f'{type(error).__name__}: {error}')for i,_,row in todo}
            finally:self._evict()
            for i,row in done.items():out[i]=row
        for i,row in enumerate(out):
            try:encode(row)
            except(ValueError,TypeError)as error:
                out[i]=dict(self._response(row.get('request_id'),n),cause='server_error',refusal=f'RESPONSE_NOT_ENCODABLE: {type(error).__name__}: {error}')
        return out

    def _search(self,todo,started,bucket):
        """prepare -> build -> one price -> reduce for the batch's valid requests; {response index: response}."""
        for _,p,_ in todo:
            if type(p['battle_id'])is str:self.battles[p['battle_id']]=None;self.battles.move_to_end(p['battle_id'])
        prepared=cw.prepare_roots(self.constructor,[p for _,p,_ in todo],device=self.device);t_prepare=time.perf_counter()
        made=cw.build(self.constructor,prepared,self.pricers[self.buckets[0]],self.params,binding=self.binding,recipe=self.recipe,only_complete=True)
        t_build=time.perf_counter();chosen=frame=None
        if made.requests:
            chosen=bucket if bucket is not None else next((b for b in self.buckets if len(made.requests)<=b*self.K),None)
            if chosen is None:raise ValueError(f'{len(made.requests)} blocks exceed the largest bucket\'s capacity')
            frame,_=cw.price(self.pricers[chosen],made,self.binding,self.recipe,self.params)
        t_price=time.perf_counter();results=[]
        for root in made.roots:   # one root at a time: a reduction ValueError refuses that root, not the batch
            try:results.extend(cw.reduce(frame,replace(made,roots=(root,))))
            except ValueError as error:
                results.append(dict(request_id=root['request_id'],status='REFUSED',cause=f'REDUCTION_ERROR: {type(error).__name__}: {error}',block_causes=None))
        t_reduce=time.perf_counter();decided=self.clock()-started
        if len(prepared)!=len(todo)or len(results)!=len(todo)or any(r['request_id']!=p['request_id']for r,(_,p,_)in zip(results,todo)):
            raise ValueError('the reduction does not answer the batch\'s requests in order')
        out={}
        for(i,_,row),item,result in zip(todo,prepared,results):
            worlds=item.made.worlds if type(item)is cw.Prepared else()
            timings=dict(row['timings'],prepare_seconds=item.seconds,build_seconds=t_build-t_prepare,price_seconds=t_price-t_build,reduce_seconds=t_reduce-t_price)
            base=dict(row,constructed_worlds=sum(w.state is not None for w in worlds),world_refusals=[dict(world=w.index,kinds=list(w.refusals))for w in worlds if w.refusals],
                timings=timings,bucket=chosen,block_causes=None if result['block_causes']is None else list(result['block_causes']))
            if result['status']=='LABELED':
                out[i]=dict(base,status='COMPLETE',action_index=result['action_mean_q'],q_values=list(result['completed_q']),policy_target=list(result['policy_target']),
                    root_value=result['root_value'],action_policy_target=result['action_policy_target'],decision_seconds=decided)
            else:out[i]=dict(base,cause=refusal_cause(result['cause']),refusal=result['cause'])
        return out

    # Sessions -----------------------------------------------------------------------------------------------------------------------
    def _evict(self):
        """Drop the sessions and attempted addresses of the least recently prepared battles beyond session_cap."""
        drop=set()
        while len(self.battles)>self.session_cap:drop.add(self.battles.popitem(last=False)[0])
        if not drop:return
        sessions,seen=self.constructor.sessions,self.constructor.seen
        for key in[k for k in sessions if _battle(k)in drop]:del sessions[key]
        seen.difference_update([a for a in seen if type(a)is tuple and len(a)==3 and _battle(a[0])in drop])
        self.evicted+=len(drop)

    def _reset_sessions(self):
        self.constructor.seen.clear();self.constructor.sessions.clear();self.battles.clear()

    # Warmup -------------------------------------------------------------------------------------------------------------------------
    def warmup(self,requests):
        """The full path on saved requests (their eight address keys) once cold and once warm per bucket, each forced to that
        bucket, so every program compiles before READY. Windows of the saved requests are tried until one prices at the bucket
        (RuntimeError if none does). Returns dict(buckets={b: seconds, compiles and outcomes}, programs, seconds)."""
        keys=PROTOCOL_KEYS[2:];pool=[]
        for r in requests:
            if not isinstance(r,dict)or any(k not in r for k in keys):raise ValueError('saved requests with '+', '.join(keys)+' required')
            pool.append({k:r[k]for k in keys})
        if not pool:raise ValueError('at least one saved request required')
        report={};started=time.perf_counter()
        for b in self.buckets:
            for attempt in range(-(-len(pool)//b)):
                batch=[dict(pool[(attempt*b+j)%len(pool)],schema=SCHEMA,request_id=f'warmup-b{b}-{attempt}-{j}',rung=self.rung,budget_seconds=WARMUP_BUDGET)
                    for j in range(b)]
                self._reset_sessions();clock=time.perf_counter();rows=self._batch(batch,bucket=b);cold=time.perf_counter()-clock
                if any(r['bucket']==b for r in rows):break
            else:
                self._reset_sessions()
                raise RuntimeError(f'warmup: no saved request priced at bucket {b}: '+'; '.join(sorted({f'{r["cause"]}: {r["refusal"]}'for r in rows})))
            compiles=self.pricers[b].compiles
            self._reset_sessions();clock=time.perf_counter();again=self._batch(batch,bucket=b);warm=time.perf_counter()-clock
            report[b]=dict(cold_seconds=cold,warm_seconds=warm,attempts=attempt+1,roots=b,complete=sum(r['status']=='COMPLETE'for r in rows),
                causes=sorted({r['cause']for r in rows if r['cause']}),compiled=len(compiles),lower_seconds=sum(c[1]for c in compiles),
                compile_seconds=sum(c[2]for c in compiles),warm_compiled=len(self.pricers[b].compiles)if any(r['bucket']==b for r in again)else None)
        self._reset_sessions()
        self.warm=dict(buckets=report,programs=len(self.programs),seconds=time.perf_counter()-started)
        return self.warm
