#!/usr/bin/env python3
"""One long-lived GPU listener answering ladder search requests with DepthN depth 2 over ProtocolConstructor worlds.

Design. build() loads Codex's pinned constructor inputs (phase_d_protocol_constructor.load_inputs: B2 catalog, bound
source, engine metadata, owning config), the champion (champion_evaluator's flags: archive and checkpoint sha256s, bf16 by
default), one ProtocolConstructor (W from the rung, sessions keyed by the rung), the executor over the same metadata and
catalog, and one DepthNPricer per capacity bucket (depthn_live_search.make_pricers), and returns a DepthNLiveSearch
(--refuse-classes passed through: those root classes are refused before pricing; recorded in the header and READY). run()
warms it up on --warmup-requests (every bucket's programs compile while nothing listens), wraps it for --telemetry and
serves it until stopped (SIGTERM or an interrupt); it is never restarted per game.
- Server is phase_d_search_server.Server set up through its own _initialize, as phase_d_ladder_transport.Server: Codex's
  framing, handle loop, bounded queue, batch window, single consumer and single executor thread, unchanged. Its one
  change is the queue. ArrivalQueue stamps each request dict with the arrival time the handle loop itself takes and
  queues beside it (the item's third member: the time.monotonic() its server_wall_seconds counts from) as the item enters
  the queue, through asyncio.Queue._put, the hook put and put_nowait share. process_many reads the stamp for
  expired_in_queue; a client cannot forge it (the stamp overwrites any value sent). The queue is the one point the handle
  loop and the consumer share, so hooking it stamps arrival without copying (and so forking) Codex's handle loop.
- READY: one JSON line (the run header, each bucket's cold (compile and run) and warm seconds, the program count, the carry
  reference root (address and classes), the full warmup report) printed once warmup is done AND this server's own listener has accepted a probe connection: the probe
  counts only when Codex's handler registers it in this server's `clients` while serving, so another process already
  listening on the port never triggers READY (the bind then fails and serve raises). The Server (Codex's queue and
  batch checks) is built before the warmup, so a bad queue setting fails before any compile.
- Telemetry (--telemetry PATH, appended): a `ready` row (the READY line), then one `request` row per request (the
  response, the request's address, rung, budget and line count, its arrival and the server-side seconds to its result),
  written, flushed and fsynced per batch by a writer thread, off the executor's critical path.
- server_id (--server-id, default HOSTNAME:PORT): this process's name, in READY and in every response it sends, so a client
  behind a pool (scripts/depthn_live_proxy) records which process answered and the panel readout breaks latency down by it.
Batched serving (--prepare-workers N > 0; off by default, when everything above is exactly as before): prepare runs in N worker
processes (gpu_engine.prepare_pool: each its own constructor, built by worker_constructor from this server's own arguments, on
the CPU, battle affinity) and requests go through gpu_engine.depthn_batching.Pipeline (a scheduler, a device thread and a
reduce thread; its docstring has the flush rule). Server.consume is then the pipeline's consumer: it takes each queued request
as it arrives while fewer than --queue-limit are in flight (so the handle loop's put still backpressures), and sets each
request's future when its response is ready (after the request_id check Codex's consumer makes). Responses may complete out of
arrival order across connections; each connection still has one request in flight and gets its own answer (Codex's handle
loop: put, then await that request's future, then write it). On the queue's None (serve's exit, SIGTERM) it admits nothing
more, answers everything accepted, then stops the pipeline and the workers. Warmup prepares through the workers (every worker
first prepares the saved requests once, so its first-use compiles happen before READY). Telemetry rows also count JAX backend
compiles since the previous row (this process: `compiles`; the request's prepare worker: pipeline.prepare_compiles).
JAX: x64 pinned off (as Codex's ladder server) and the matmul precision hard-pinned to --matmul-precision (default
'highest', the precision the champion gates R1/R2/R3 were certified under, through the environment's
JAX_DEFAULT_MATMUL_PRECISION; Codex's ladder server uses 'highest' too). The server refuses to start when the environment
names another precision (checked before anything heavy loads) and again when JAX's own jax_default_matmul_precision differs
once it is imported (in build, before any warmup); READY and the telemetry header record it, and depthn_live_search
records it in every response's pricer identity. The champion's own precision flag (bf16 by default) sets its forward.
"""
import argparse
import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path
import queue
from types import SimpleNamespace
import signal
import socket
import subprocess
import threading
import time

from pokezero import phase_d_search_server as original
from gpu_engine import atomic_compilation_cache,champion_evaluator as evaluators,depthn_live_search as live


class ArrivalQueue(asyncio.Queue):
    """asyncio.Queue stamping request[live.ARRIVAL] = arrival on each (request, future, arrival) item it accepts, and keeping,
    per connection handler task (the putter), the future of the request whose answer it owes (owing: a handler has one request
    in flight; the entry goes when its answer is written, AnswerMark, or the task ends), so a stop can wait for exactly the
    handlers that still have an answer to write."""
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.owing={}

    def _put(self,item):
        if item is not None:
            item[0][live.ARRIVAL]=item[2];task=asyncio.current_task()
            if task is not None:
                if task not in self.owing:task.add_done_callback(lambda done:self.owing.pop(done,None))
                self.owing[task]=item[1]
        super()._put(item)


class AnswerMark:
    """A connection's StreamWriter for Codex's handle: write() (the handle loop's one write per request: the answer) also
    clears its task's debt in `owing`; everything else is the writer's."""
    def __init__(self,writer,owing):self._writer=writer;self._owing=owing

    def write(self,data):
        self._writer.write(data);self._owing.pop(asyncio.current_task(),None)

    def __getattr__(self,name):return getattr(self._writer,name)


class Server(original.Server):
    def __init__(self,search,*,max_requests=4,batch_wait_seconds=.005,queue_limit=256,pipeline=None):
        if not callable(getattr(search,'process_many',None)):raise ValueError('a search object with process_many required')
        self._initialize(search,max_requests,batch_wait_seconds,queue_limit)
        self.queue=ArrivalQueue(queue_limit);self.loop=self.task=None;self.stopping=False;self.limit=queue_limit
        self.pipeline=pipeline   # with one, consume() is the pipeline's (module docstring); without, Codex's
        self.inflight=0

    async def consume(self):
        """Codex's consumer, or with a pipeline: each queued request submitted as it arrives while fewer than queue_limit are in
        flight; its future set when its response is ready; on None, everything accepted answered, then the pipeline stopped."""
        if self.pipeline is None:return await super().consume()
        if self.started:raise ValueError('one shared GPU queue consumer required')
        self.started=True;loop=asyncio.get_running_loop();slots=asyncio.Semaphore(self.limit)
        while True:
            await slots.acquire()
            item=await self.queue.get()
            if item is None:
                slots.release();self.queue.task_done();break
            request,future,_=item;self.inflight+=1
            self.pipeline.submit(request,lambda response,request=request,future=future:loop.call_soon_threadsafe(self._settle,request,future,response,slots))
        for _ in range(self.limit):await slots.acquire()   # drain: every accepted request answered
        await loop.run_in_executor(None,self.pipeline.close)

    def _settle(self,request,future,response,slots):
        """On the loop: one request's end (Codex's consumer's checks: a response for another request_id is an error)."""
        self.inflight-=1;slots.release();self.queue.task_done()
        if future.done():return
        if isinstance(response,BaseException):future.set_exception(RuntimeError(type(response).__name__+': '+str(response)));return
        if not isinstance(response,dict)or response.get('request_id')!=request.get('request_id'):
            future.set_exception(RuntimeError('ValueError: shared GPU result order/address differs'));return
        finish=getattr(self.search,'finish',None)
        future.set_result(finish(request,response)if callable(finish)else response)

    async def serve(self,host,port,*,on_ready=None):
        """Codex's serve (listener and consumer; on exit it drains accepted work and shuts the executor down), plus on_ready()
        once the listener accepts a connection, and stop() from any thread (SIGTERM when serving on the main thread)."""
        self.loop=asyncio.get_running_loop();self.stop_event=asyncio.Event();self.task=asyncio.ensure_future(self._serve(host,port))
        if threading.current_thread()is threading.main_thread():self.loop.add_signal_handler(signal.SIGTERM,self.stop)
        try:
            if on_ready is not None and await self._listening(host,port):on_ready()
            await self.task
        except asyncio.CancelledError:
            if not self.stopping:raise

    async def _listening(self,host,port,*,patience=1.):
        """True once THIS server accepted a probe connection on (host, port): Codex's handle registers its task in
        self.clients, so the probe counts only when a new task appears there while serving (its empty read then ends the
        handler quietly). A connection accepted by anything else on the port never counts. False, or the listener's own
        error (e.g. the port in use), if serving ended first."""
        while not self.task.done():
            known=set(self.clients)
            try:_,writer=await asyncio.open_connection(host,port)
            except OSError:await asyncio.wait({self.task},timeout=.02);continue
            try:
                deadline=self.loop.time()+patience
                while not self.task.done()and self.loop.time()<deadline:
                    if set(self.clients)-known:return True
                    await asyncio.wait({self.task},timeout=.01)
            finally:
                writer.close()
                with suppress(OSError):await writer.wait_closed()
        self.task.result();return False

    async def _serve(self,host,port):
        """Codex's serve (phase_d_search_server.Server.serve), its steps in its order, stopped by stop()'s event instead of by
        cancelling the listener's serve_forever: since CPython 3.13.(14..18) a cancelled serve_forever also closes every client
        connection at once (asyncio.Server.close_clients), so the answers owed to requests already accepted were lost (the
        pipelined server's drain test failed on CI). Here the listener only stops accepting; then, as Codex's: closing set,
        the queue's sentinel, the consumer drained (every accepted request's future set), each handler owed an answer waited
        for while it writes it and ends (closing: at most 10 s), the other handlers (idle on their connections) ended, the
        executor shut down."""
        consumer=asyncio.create_task(self.consume());server=None
        try:
            server=await asyncio.start_server(lambda reader,writer:self.handle(reader,AnswerMark(writer,self.queue.owing)),host,port)
            await self.stop_event.wait()
        finally:
            if server is not None:server.close()   # no new connections; the accepted ones stay open until answered
            self.closing=True
            await self.queue.put(None)
            await consumer
            owed=[task for task,future in self.queue.owing.items()if future.done()and not task.done()]
            if owed:await asyncio.wait(owed,timeout=10.)   # each writes the answer it owes, then (closing) leaves its loop
            for task in tuple(self.clients):task.cancel()
            await asyncio.gather(*self.clients,return_exceptions=True)
            if server is not None:await server.wait_closed()
            self.executor.shutdown(wait=True,cancel_futures=False)

    def stop(self):
        """Stop serving, from any thread: the listener closes, accepted requests are answered, then serve returns."""
        self.stopping=True
        if self.loop is not None:self.loop.call_soon_threadsafe(self.stop_event.set)


def request_row(request,response,finished):
    """A telemetry row: the response, the request's address (not its lines, which grow every decision: their count), its
    arrival stamp and the server-side seconds from arrival to the result."""
    q=request if isinstance(request,dict)else{};arrival=q.get(live.ARRIVAL)
    return dict(kind='request',response=response,address={k:q.get(k)for k in('battle_id','owner','stream_id','decision_index','generation','prefix_sha256')},
        rung=q.get('rung'),budget_seconds=q.get('budget_seconds'),lines=len(q['lines'])if isinstance(q.get('lines'),list)else None,
        arrival_monotonic=arrival,server_seconds=finished-arrival if type(arrival)in(int,float)else None)


class CompileCounter:
    """This process's JAX backend compiles (jax.monitoring's backend_compile_duration events), persistent compilation cache
    hits and misses (its cache_hits events; cache_misses: an entry compiled and written), and cache entries it could not read
    (JAX's 'Error reading persistent compilation cache entry' warnings, as shown), read as deltas: run() reads the warmup's
    at READY (recorded there), so each request row's are its own."""
    EVENT='/jax/core/compile/backend_compile_duration';HIT='/jax/compilation_cache/cache_hits';MISS='/jax/compilation_cache/cache_misses'
    UNREADABLE='Error reading persistent compilation cache entry'
    KEYS=('compiles','compile_seconds','cache_hits','cache_misses','cache_read_errors')

    def __init__(self):
        self.now=dict.fromkeys(self.KEYS,0);self.now['compile_seconds']=0.;self.read=dict(self.now);self.lock=threading.Lock()
        try:
            import jax
            jax.monitoring.register_event_duration_secs_listener(self._listen);jax.monitoring.register_event_listener(self._event)
        except Exception:pass   # noqa: BLE001 -- a JAX without monitoring: the counts stay 0
        import warnings
        shown=warnings.showwarning
        def showwarning(message,category,filename,lineno,file=None,line=None):
            if self.UNREADABLE in str(message):
                with self.lock:self.now['cache_read_errors']+=1
            return shown(message,category,filename,lineno,file,line)
        warnings.showwarning=showwarning

    def _listen(self,event,duration,**kw):
        if event==self.EVENT:
            with self.lock:self.now['compiles']+=1;self.now['compile_seconds']+=duration

    def _event(self,event,**kw):
        key={self.HIT:'cache_hits',self.MISS:'cache_misses'}.get(event)
        if key is not None:
            with self.lock:self.now[key]+=1

    def delta(self):
        """{compiles, compile_seconds, cache_hits, cache_misses, cache_read_errors} since the previous read."""
        with self.lock:
            out={k:self.now[k]-self.read[k]for k in self.KEYS};self.read=dict(self.now)
        return out


class Recorder:
    """What the Server calls: the search's process_many, its responses stamped with server_id (when one is given), plus telemetry
    rows queued for a writer thread (each with the backend compiles since the previous row)."""
    def __init__(self,search,path=None,*,server_id=None,compiles=None):
        if server_id is not None and (type(server_id)is not str or not server_id):raise ValueError('a non-empty server_id required')
        self.search=search;self.server_id=server_id;self.rows=self.thread=self.file=None;self.compiles=compiles
        if path is not None:
            self.file=open(path,'a',encoding='utf-8');self.rows=queue.SimpleQueue()
            self.thread=threading.Thread(target=self._write,name='depthn-live-telemetry',daemon=True);self.thread.start()

    def process_many(self,requests):
        requests=list(requests);responses=self.search.process_many(requests);finished=time.monotonic()
        if self.server_id is not None:responses=[dict(r,server_id=self.server_id)if isinstance(r,dict)else r for r in responses]
        if self.rows is not None:self.rows.put(self._rows(requests,responses,finished))
        return responses

    def finish(self,request,response):
        """One pipelined request's response (Server._settle): stamped and recorded as process_many's are."""
        finished=time.monotonic()
        if self.server_id is not None and isinstance(response,dict):response=dict(response,server_id=self.server_id)
        if self.rows is not None:self.rows.put(self._rows([request],[response],finished))
        return response

    def _rows(self,requests,responses,finished):
        rows=[request_row(q,r,finished)for q,r in zip(requests,responses)]
        if self.compiles is not None and rows:rows[0].update(self.compiles.delta())
        return rows

    def record(self,row):
        if self.rows is not None:self.rows.put([row])

    def _write(self):
        while(rows:=self.rows.get())is not None:
            self.file.write(''.join(json.dumps(r,sort_keys=True,default=repr)+'\n'for r in rows));self.file.flush();os.fsync(self.file.fileno())

    def close(self):
        if self.rows is not None:self.rows.put(None);self.thread.join();self.file.close();self.rows=None


def _print(row):print(json.dumps(row,sort_keys=True,default=repr),flush=True)


PRECISION_ENVIRONMENT='JAX_DEFAULT_MATMUL_PRECISION'


def check_precision_environment(p,args):
    """Before anything heavy loads: JAX reads JAX_DEFAULT_MATMUL_PRECISION at import, and it must name the pinned precision.
    An early fail-fast only: the guard is build()'s check of JAX's own setting, which also catches a process that imported
    or reconfigured JAX before this check could see it."""
    found=os.environ.get(PRECISION_ENVIRONMENT)
    if found!=args.matmul_precision:
        p.error(f'{PRECISION_ENVIRONMENT} must be {args.matmul_precision!r} (the precision the champion gates were certified under); found {found!r}')


def matmul_precision():
    """JAX's own default matmul precision, as configured in this process."""
    import jax
    return str(jax.config.jax_default_matmul_precision)


def run(search,*,host,port,warmup_requests,max_requests,batch_wait_seconds=.005,queue_limit=256,telemetry=None,header=None,log=_print,on_server=None,
        server_id=None,pool=None,pipeline_options=None,warm_in=()):
    """Build the Server (Codex's queue checks), warm `search` up, then serve until stopped; READY is logged (and recorded)
    once it is warm AND its own listener accepts connections. `on_server(server)` sees the Server before the warmup (its
    stop() ends run once serving). With `pool` (a started PreparePool of this search's constructor), the search prepares
    through it and serves through a depthn_batching.Pipeline (pipeline_options: its keywords); the pool is closed on exit.
    `warm_in`: prepare's warm-in list (DepthNLiveSearch.warmup's: this process's own prepare), when there is one."""
    compiles=CompileCounter();recorder=None
    try:   # the pool's workers are up: everything from here closes them on the way out
        recorder=Recorder(search,telemetry,server_id=server_id,compiles=compiles)
        server=Server(recorder,max_requests=max_requests,batch_wait_seconds=batch_wait_seconds,queue_limit=queue_limit)
        if on_server is not None:on_server(server)
        workers=None
        local=pool is not None and(pipeline_options or{}).get('local_battles',0)>0;warm_saved=False
        if local:
            pool.attach_local(search.constructor)   # batch 1 in process: this search's own prepare, warmed below as the serial server's
            warm_saved=True   # as a worker's two passes: warmup's own prepares go to the workers once the pool is attached, so
            # this slot's prepare warms the warmup requests here (leniently, as the workers do), then the list (strictly)
        elif pool is not None:warm_in=()   # nothing prepares in this process: only the workers warm in (at their start)
        if pool is not None:
            started=time.perf_counter();pool.wait_ready();search.attach_pool(pool)
            workers=dict(workers=pool.size,ready_wait_seconds=time.perf_counter()-started,warm_in={w.index:(w.ident or{}).get('warm_in')for w in pool.workers},
                atomic_cache={w.index:(w.ident or{}).get('atomic_cache')for w in pool.workers})
            if(pipeline_options or{}).get('reply_forward')=='device'and getattr(search,'reply',None)is not None:
                search.reply_batches=tuple(search.buckets)   # warmup compiles each bucket's batched reply forward, both seats
        warm=search.warmup(warmup_requests,**({'warm_in':warm_in}if warm_in else{}),**({'warm_in_saved':True}if warm_saved else{}));buckets=warm.get('buckets',{})
        if pool is not None:
            from gpu_engine.depthn_batching import Pipeline
            server.pipeline=Pipeline(search,pool,max_batch=max_requests,**(pipeline_options or{})).start()
            warm=dict(warm,prepare_workers=dict(workers,rss_bytes=pool.rss()),pipeline=dict(pipeline_options or{},max_batch=max_requests))
        ready=dict(event='READY',host=host,port=port,server_id=server_id,**(header or{}),cold_seconds={b:x.get('cold_seconds')for b,x in buckets.items()},
            warm_seconds={b:x.get('warm_seconds')for b,x in buckets.items()},programs=warm.get('programs'),carry_reference=warm.get('carry_reference'),warmup=warm,
            warmup_compiles=compiles.delta(),   # everything up to READY: each request row's compiles and cache counts are serving's own
            atomic_cache=dict(atomic_compilation_cache.STATUS))   # this process's whole-entry cache writes (the workers': prepare_workers.atomic_cache)
        def announce():log(ready);recorder.record(dict(ready,kind='ready'))
        asyncio.run(server.serve(host,port,on_ready=announce))
    finally:
        try:
            if pool is not None:pool.close()
        finally:
            if recorder is not None:recorder.close()


# Command line ---------------------------------------------------------------------------------------------------------------------
def buckets_arg(text):
    try:values=tuple(sorted({int(x)for x in text.split(',')}))
    except ValueError:values=()
    if not values or values[0]<=0:raise argparse.ArgumentTypeError('comma-separated positive integers required')
    return values


def classes_arg(text):
    values=tuple(x for x in text.split(',')if x)
    if any(x not in live.CLASSES for x in values):raise argparse.ArgumentTypeError(f'comma-separated root classes from {", ".join(live.CLASSES)} required')
    return values


def parser():
    p=argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for name in('showdown-root','node-binary','metadata','expected-metadata-sha256','vocabulary','expected-vocabulary-sha256'):p.add_argument('--'+name,required=True)
    evaluators.add_arguments(p)
    p.add_argument('--rung',choices=tuple(live.RUNGS),default='DN2W8',help='DN2W8 (W=8) or DN2W16 (W=16); DN2W32 / DN2W64 for offline replays')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,required=True)
    p.add_argument('--buckets',type=buckets_arg,default=(1,2,4),help='capacity buckets, in roots per price call (default 1,2,4)')
    p.add_argument('--max-requests',type=int,help='requests per batch (default the largest bucket; set from gap 4B)')
    p.add_argument('--batch-wait-seconds',type=float,default=.005);p.add_argument('--queue-limit',type=int,default=256)
    p.add_argument('--min-price-seconds',type=float,default=1.,help='refuse expired_in_queue when the queue wait leaves less of the budget')
    p.add_argument('--session-cap',type=int,default=256,help='constructor sessions kept, by least-recent battle')
    p.add_argument('--refuse-classes',type=classes_arg,default=(),help='root classes refused before pricing (class_refused), comma-separated from '
        +', '.join(live.CLASSES)+' (default none)')
    p.add_argument('--seed',type=int,default=20261103,help='constructor world seed (Codex\'s default)')
    p.add_argument('--recipe-seed',type=int,default=20261103,help='the pricer\'s lane-order recipe seed')
    p.add_argument('--saved-root-admission',action='store_true',help='Codex\'s stateless mode: every prefix folded from scratch, no sessions')
    p.add_argument('--incremental-roots',action='store_true',help='keep each session\'s public-root fold (RootSession) and fold only the new '
        'lines; classify reuses the root prepare captured. The same roots as capture_root from scratch, at the cost of the new lines '
        '(needs sessions: not with --saved-root-admission); the header records incremental_roots')
    p.add_argument('--eager-peer-fold',action='store_true',help='advance each battle\'s peer wire fold on every root (with the redaction '
        'on), so the first private reveal folds only the new lines instead of the whole battle (constructor_worlds.advance_peer_fold): the '
        'same fold and beliefs, without the 1-2 s first-reveal spike; the header records eager_peer_fold')
    p.add_argument('--ability-guard',action='store_true',help='the effect guard also guards lanes whose outcome depends on an ability the '
        'engine does not model (Wonder Guard, Blaze/Torrent/Overgrow/Swarm, Poison Point/Flame Body/Effect Spore: '
        'pokezero.depthn_effect_guard.ability_evaluate under its ABILITY_ARMS and PINCH_ROOT); the header records ability_guard')
    p.add_argument('--redact-private-reveals',action='store_true',help='clear the owner\'s abilities revealed only on Showdown\'s side-private '
        'Gen 3 Pressure line from the peer\'s view of the owner side (constructor_worlds.redact_private_reveals; off by default)')
    p.add_argument('--peer-view-b',action='store_true',help='view b in the peer\'s view (constructor_worlds\' Peer view b; off by default): it '
        'implies --redact-private-reveals and projects the root\'s two peer damage cells to Showdown\'s public percent; the run header '
        'records peer_view_b (and redact_private_reveals true), each built response its peer_view_b record')
    p.add_argument('--prepare-workers',type=int,default=0,help='batched serving: prepare in this many worker processes and serve through '
        'gpu_engine.depthn_batching (default 0: off, the serial path)')
    p.add_argument('--prepare-cpus',help='the workers\' CPUs: a CPU list (e.g. 8-15) split evenly across the workers in order (default: '
        'this process\'s own CPUs, shared)')
    p.add_argument('--prepare-timeout-seconds',type=float,default=30.,help='a prepare running longer kills its worker (server_error)')
    p.add_argument('--batch-window-seconds',type=float,default=.05,help='batched serving: the longest a ready root waits for others while the '
        'device is idle')
    p.add_argument('--flush-guard-seconds',type=float,default=.15,help='batched serving: the deadline rule\'s margin (gpu_engine.depthn_batching)')
    p.add_argument('--local-battles',type=int,default=8,help='batched serving: at most this many live battles prepared in this process (a new '
        'battle arriving with nothing else in flight while the server runs about one decision at a time; gpu_engine.depthn_batching), so '
        'batch 1 pays nothing for batching; 0: every battle in a worker')
    p.add_argument('--split-worlds',type=int,default=0,help='MEASUREMENT ONLY (gpu_engine.world_split; never in production before worlds\' '
        'world-range seam): a lone in-process decision of W >= --split-min-worlds builds its worlds across up to this many idle workers '
        '(default 0: off)')
    p.add_argument('--split-min-worlds',type=int,default=32,help='the smallest W --split-worlds splits (default 32)')
    p.add_argument('--reply-forward',choices=('reduce','device'),default='reduce',help='batched serving: the reply prior\'s forward per root '
        'on the reduce thread (default; as the serial search) or once per batch on the device thread. device is NOT bitwise the per-root '
        'forward: the batch\'s forward has another batch shape, and the backend may pick another matmul (CPU: 0 of 194 roots equal, max '
        '|logit difference| 0.052); never on a served config until shown batch-invariant on its GPU')
    p.add_argument('--block-lanes',type=int,default=2187);p.add_argument('--blocks-per-call',type=int,default=16)
    p.add_argument('--leaf-layout',choices=('blocks','wide'),default='blocks',help='the DepthN leaf program (device_depthn_pricer.LEAF_LAYOUTS): '
        'blocks (default, the programs as before) or wide (one iteration of --blocks-per-call forward chunks of the blocks\' B*K rows per call, '
        'only enumerated lanes packed). Placement and digest are the blocks layout\'s by construction, and so are leaf status and error '
        'except the forward-usability code (error 4); that code and the leaf values match only while the forward is row-position invariant: '
        'gated on GB300 under the pinned stack, not on CPU, so a run serving wide passes scripts/depthn_wide_leaf_frame_gate.py at bind time. '
        'The pricer identity records the layout when not blocks')
    p.add_argument('--wide-nodes',type=int,help='node slots per wide leaf call (default its lanes // 16; only with --leaf-layout wide)')
    p.add_argument('--warmup-requests',required=True,help='JSONL of saved Showdown requests, one object per line with the eight address keys of '
        'Codex\'s ladder request (battle_id, owner, stream_id, decision_index, generation, prefix_sha256, lines, own_request_line)')
    p.add_argument('--prepare-warm-in',help='JSONL of saved requests (the eight address keys, as --warmup-requests) that prepare runs once '
        'each, in order, during warmup (DepthNLiveSearch.warmup\'s warm_in): their first-use CPU compiles happen before READY instead of on '
        'live roots. With --prepare-workers, every worker also runs them before it is ready (after the warmup requests). With '
        '--prepare-warm-in-sha256')
    p.add_argument('--prepare-warm-in-sha256',help='sha256 of --prepare-warm-in\'s bytes: the same list on every startup (the server refuses '
        'another)')
    p.add_argument('--telemetry',help='server-side JSONL, appended, fsynced per batch')
    p.add_argument('--server-id',help='this process\'s name in READY and every response (default HOSTNAME:PORT)')
    p.add_argument('--tree-commit',help='the code commit served (default: this checkout\'s HEAD, +dirty when tracked files differ)')
    p.add_argument('--compilation-cache',help='absolute directory for JAX\'s persistent compilation cache')
    p.add_argument('--reply-archive',help='br_fp: a reply model (a champion_evaluator .npz export, e.g. pi_FP-red2) whose logits on each '
        'world\'s peer view replace root_peer_logits before averaging; default off (the champion\'s own averaging)')
    p.add_argument('--reply-archive-sha256',help='sha256 of --reply-archive\'s bytes (required with it)')
    p.add_argument('--root-operator',choices=live.ROOT_OPERATORS,default=live.ROOT_OPERATOR,help='the live root operator: '
        'ARGMAX_MEAN_COMPLETED_Q (default: averaging plays, or br_fp under --reply-archive) or NOISE_GATED_K2 (that choice only past '
        'a 2-standard-error per-world gap over the root logits\' argmax; screen-r4a variant B with --reply-archive)')
    p.add_argument('--reply-checkpoint-sha256',help='the reply export\'s checkpoint sha256 (required with --reply-archive)')
    p.add_argument('--reply-views',choices=('eager','jit','check'),default='eager',help='how --reply-archive builds the peer inputs: eager '
        '(default; bit-exact with the RR re-pricer), jit (one vmapped program per root), or check (eager plays; jit reported beside)')
    p.add_argument('--matmul-precision',choices=('highest','high','default'),default='highest',
        help='the pinned JAX default matmul precision (default highest, the gates\' certified precision); the server refuses to start '
        'unless JAX_DEFAULT_MATMUL_PRECISION and JAX\'s own setting both name it')
    return p


def check_reply_arguments(p,args):
    """--reply-archive and its two sha256 flags go together, as 64-hex sha256s (the server's and the re-pricer's options)."""
    reply=[getattr(args,k,None)for k in('reply_archive','reply_archive_sha256','reply_checkpoint_sha256')]
    if any(reply)and not all(reply):p.error('--reply-archive, --reply-archive-sha256 and --reply-checkpoint-sha256 go together')
    if not any(reply)and getattr(args,'reply_views','eager')!='eager':p.error('--reply-views goes with --reply-archive')
    if all(reply)and not all(type(x)is str and len(x)==64 and all(c in'0123456789abcdef'for c in x)for x in reply[1:]):
        p.error('--reply-archive-sha256 and --reply-checkpoint-sha256 must be 64-hex sha256s')


def reply_view(meta):
    """The peer view a reply archive was trained on: its export metadata's extra.view (fp_reply_train --view), 'env' when absent."""
    return ((meta or {}).get('extra') or {}).get('view') or 'env'


def check_reply_view(meta,args):
    """A reply model trained on the builder's seat view b (view 'b': Foul Play's view without the [silent]-only reveals and with
    the damage columns at public precision) is served only with --peer-view-b on, so its inputs are the view it learned; ValueError
    otherwise (the server's and the re-pricer's loading). One-way: an env-view archive is still served with --peer-view-b."""
    if reply_view(meta)=='b'and not getattr(args,'peer_view_b',False):
        raise ValueError('the reply archive was trained on peer view b: serve it with --peer-view-b')


def options(argv=None):
    p=parser();args=p.parse_args(argv);evaluators.check_arguments(p,args);check_precision_environment(p,args)
    # V2 self_moves: the server encodes what its champion contract declares (the contract's masks carry no self_moves, so
    # False); an OBS_SELF_MOVES override is for measurement tools and refuses a serving start.
    from pokezero.observation import require_declared_self_moves
    try:require_declared_self_moves(False,context='depthn live server')
    except ValueError as error:p.error(str(error))
    if not args.champion_archive:p.error('--champion-archive with its two sha256 flags is required: the champion prices every root and leaf')
    check_reply_arguments(p,args)
    if getattr(args,'eager_peer_fold',False)and not(args.redact_private_reveals or args.peer_view_b):
        p.error('--eager-peer-fold folds the peer wire for the redaction: it needs --redact-private-reveals or --peer-view-b')
    if args.prepare_workers<0:p.error('--prepare-workers must be >= 0')
    if args.prepare_workers==0 and(args.prepare_cpus or args.reply_forward!='reduce'):p.error('--prepare-cpus and --reply-forward go with --prepare-workers')
    if args.local_battles<0:p.error('--local-battles must be >= 0')
    if args.split_worlds<0 or args.split_worlds and(not args.prepare_workers or not args.local_battles):
        p.error('--split-worlds (measurement only) goes with --prepare-workers and --local-battles')
    if bool(getattr(args,'prepare_warm_in',None))!=bool(getattr(args,'prepare_warm_in_sha256',None)):
        p.error('--prepare-warm-in and --prepare-warm-in-sha256 go together')
    if args.wide_nodes is not None and(args.leaf_layout!='wide'or args.wide_nodes<=0):
        p.error('--wide-nodes is a positive node capacity of --leaf-layout wide')
    if args.prepare_cpus:
        try:prepare_cpu_sets(args.prepare_cpus,args.prepare_workers)
        except ValueError as error:p.error(f'--prepare-cpus: {error}')
    for name in('prepare_timeout_seconds','batch_window_seconds','flush_guard_seconds'):
        value=getattr(args,name)
        if not(value==value and 0<=value<float('inf'))or name=='prepare_timeout_seconds'and value<=0:p.error(f'--{name.replace("_","-")} must be finite and >= 0 (> 0 for the timeout)')
    if args.max_requests is None:args.max_requests=args.buckets[-1]
    if not 1<=args.max_requests<=args.buckets[-1]:p.error('--max-requests must be in 1..the largest bucket')
    try:   # Codex's own queue/batch checks (phase_d_search_server.Server._initialize), before anything loads or compiles
        Server(SimpleNamespace(process_many=list),max_requests=args.max_requests,batch_wait_seconds=args.batch_wait_seconds,
            queue_limit=args.queue_limit).executor.shutdown()
    except ValueError as error:p.error(f'--max-requests/--batch-wait-seconds/--queue-limit: {error}')
    return args


def read_requests(path):
    """The saved requests of a JSONL file: one JSON object per non-empty line, at least one."""
    with open(path,encoding='utf-8')as f:return parse_requests(f.read(),path)


def parse_requests(text,name):
    """Saved requests from JSONL text: one JSON object per non-empty line, at least one (ValueError naming `name` otherwise)."""
    # split on '\n' only (as line-by-line file reading did): str.splitlines() also splits on U+2028/U+2029/U+0085,
    # which a JSON string may hold unescaped (ensure_ascii=False)
    rows=[json.loads(line)for line in text.split('\n')if line.strip()]
    if not rows or not all(isinstance(r,dict)for r in rows):raise ValueError(f'{name}: one JSON object per line, at least one, required')
    return rows


def read_warm_in(args):
    """--prepare-warm-in's saved requests, its bytes held to --prepare-warm-in-sha256 (ValueError otherwise); () without it. The
    requests are parsed from the very bytes the sha256 checked (the file is read once), as phase_d_protocol_constructor.read_bound."""
    if not getattr(args,'prepare_warm_in',None):return()
    import hashlib
    blob=Path(args.prepare_warm_in).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=args.prepare_warm_in_sha256:raise ValueError('--prepare-warm-in differs from --prepare-warm-in-sha256')
    return tuple(parse_requests(blob.decode('utf-8'),args.prepare_warm_in))


def tree_commit():
    root=str(Path(__file__).resolve().parents[1])
    head=subprocess.check_output(['git','-C',root,'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',root,'status','--porcelain','--untracked-files=no'],text=True).strip()
    return head+('+dirty'if dirty else'')


def build(args):
    """The DepthNLiveSearch the arguments name, on the one visible GPU (module docstring)."""
    import jax
    jax.config.update('jax_enable_x64',False)   # pinned off, as phase_d_ladder_server
    if matmul_precision()!=args.matmul_precision:   # the hard pin, again once JAX is imported, before anything loads or warms up
        raise ValueError(f'JAX default matmul precision {matmul_precision()!r} differs from the pinned {args.matmul_precision!r}')
    if not atomic_compilation_cache.install()and args.compilation_cache:   # whole entries only, for every process sharing the cache
        raise ValueError(f'--compilation-cache needs whole-entry cache writes: {atomic_compilation_cache.STATUS["reason"]}')
    if args.compilation_cache:
        from pokezero.owning_worker_runtime import configure_compilation_cache
        configure_compilation_cache(args.compilation_cache)
    if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':raise ValueError('ONE visible GPU required')
    from pokezero import v5_gumbel
    from gpu_engine import constructor_worlds as cw
    constructor=build_constructor(args);catalog,source,metadata=constructor.catalog,constructor.source,constructor.metadata
    evaluator=evaluators.load_from_arguments(args);W=live.RUNGS[args.rung]
    from gpu_engine.private_self_moves import refuse_search_serving   # V2 self_moves: fail closed until the search threading lands
    refuse_search_serving(catalog=catalog,meta=getattr(evaluator,'meta',None),context='depthn live server')
    executor=cw.live_executor(metadata,catalog)   # the constructor's own engine metadata and catalog (the depth-N gate's CONSTRUCTOR_ROOTS uses the same)
    params=(evaluator.params,)
    from gpu_engine import depthn_guard_inputs as guard_inputs
    pricers=live.make_pricers(executor,evaluator,params,worlds=W,buckets=args.buckets,block_lanes=args.block_lanes,blocks_per_call=args.blocks_per_call,
        labels=guard_inputs.ability_labels if getattr(args,'ability_guard',False)else guard_inputs.child_labels,
        leaf_layout=getattr(args,'leaf_layout','blocks'),wide_nodes=getattr(args,'wide_nodes',None))
    reply=None
    if getattr(args,'reply_archive',None):   # br_fp: the reply model at the champion's precision, on the same executor's views
        from gpu_engine.depthn_reply_prior import ReplyPrior
        model=evaluators.load(args.reply_archive,expected_archive_sha256=args.reply_archive_sha256,
            expected_checkpoint_sha256=args.reply_checkpoint_sha256,precision=evaluators.PRECISIONS[args.champion_precision])
        check_reply_view(getattr(model,'meta',None),args)
        refuse_search_serving(meta=getattr(model,'meta',None),context='depthn live server reply model')
        reply=ReplyPrior(executor,model,mode=args.reply_views,identity=dict(archive_sha256=args.reply_archive_sha256,
            checkpoint_sha256=args.reply_checkpoint_sha256,view=reply_view(getattr(model,'meta',None))))
    return live.DepthNLiveSearch(constructor=constructor,pricers=pricers,params=params,binding=cw.binding(source,metadata,catalog),
        recipe=v5_gumbel.Recipe(args.recipe_seed),tree_commit=args.tree_commit or tree_commit(),min_price_seconds=args.min_price_seconds,
        max_requests=args.max_requests,session_cap=args.session_cap,refuse_classes=args.refuse_classes,
        redact_private_reveals=bool(getattr(args,'redact_private_reveals',False)),reply=reply,
        root_operator=getattr(args,'root_operator',live.ROOT_OPERATOR),
        **({'peer_view_b':True}if getattr(args,'peer_view_b',False)else{}),**({'ability_guard':True}if getattr(args,'ability_guard',False)else{}))


def build_constructor(args):
    """The ProtocolConstructor the arguments name (Codex's pinned inputs, W from the rung, sessions keyed by the rung): build()'s,
    and every prepare worker's (worker_constructor)."""
    from gpu_engine.phase_d_protocol_constructor import ProtocolConstructor,load_inputs
    catalog,source,metadata,config=load_inputs(args)
    return ProtocolConstructor(catalog=catalog,source=source,metadata=metadata,owning_config=config,count=live.RUNGS[args.rung],seed=args.seed,rung=args.rung,
        saved_root_admission=args.saved_root_admission,incremental_roots=bool(getattr(args,'incremental_roots',False)),
        eager_peer_fold=bool(getattr(args,'eager_peer_fold',False)))


CONSTRUCTOR_ARGUMENTS=('showdown_root','node_binary','metadata','expected_metadata_sha256','vocabulary','expected_vocabulary_sha256','rung','seed',
    'saved_root_admission','incremental_roots','eager_peer_fold')


def worker_constructor(payload):
    """A prepare worker's constructor (gpu_engine.prepare_pool's factory): build_constructor of the server's own arguments."""
    return build_constructor(SimpleNamespace(**payload))


def prepare_cpu_sets(text,workers):
    """--prepare-cpus: a CPU list ('8-15' or '8,9,12-15') split into `workers` consecutive, near-equal sets."""
    cpus=[]
    for part in text.split(','):
        lo,_,hi=part.partition('-')
        try:lo,hi=int(lo),int(hi or lo)
        except ValueError:raise ValueError(f'{part!r} is not a CPU or a CPU range')from None
        if lo<0 or hi<lo:raise ValueError(f'{part!r} is not a CPU range')
        cpus.extend(range(lo,hi+1))
    if workers<1 or len(cpus)<workers:raise ValueError(f'{len(cpus)} CPUs for {workers} workers')
    size,extra=divmod(len(cpus),workers);out=[];at=0
    for i in range(workers):
        n=size+(i<extra);out.append(tuple(cpus[at:at+n]));at+=n
    return out


def make_pool(args):
    """The prepare-worker pool of --prepare-workers, its workers starting (None at 0): started before build() so the workers
    build their constructors while this process loads the champion; run() waits for them."""
    if not args.prepare_workers:return None
    from gpu_engine.prepare_pool import PreparePool
    payload={k:getattr(args,k)for k in CONSTRUCTOR_ARGUMENTS}
    redaction={**({'redact_private_reveals':True}if args.redact_private_reveals or args.peer_view_b else{}),**({'peer_view_b':True}if args.peer_view_b else{})}
    cache={}if not args.compilation_cache else dict(jax_enable_compilation_cache=True,jax_compilation_cache_dir=str(Path(args.compilation_cache).resolve()),
        jax_persistent_cache_min_compile_time_secs=0.)   # the workers' prepare uses the server's persistent cache (written whole, as the server)
    return PreparePool('scripts.depthn_live_server:worker_constructor',payload,workers=args.prepare_workers,device='cpu',redaction=redaction,
        cpus=prepare_cpu_sets(args.prepare_cpus,args.prepare_workers)if args.prepare_cpus else None,
        timeout_seconds=args.prepare_timeout_seconds,warm_in=warm_in_requests(args),jax_config=cache).start(wait=False)


def warm_in_requests(args):
    """Every prepare worker's warm-in: two passes, every session forgotten between them (so the lists may share an address or
    go back in a battle): the warmup requests, then --prepare-warm-in's (its bytes held to the sha256), as protocol requests
    (warm-in-<j> request ids)."""
    keys=live.PROTOCOL_KEYS[2:];passes=(read_requests(args.warmup_requests),list(read_warm_in(args)));out=[];j=0
    for rows in passes:
        if rows:out.append(tuple(dict({k:r[k]for k in keys},schema=original.SCHEMA,request_id=f'warm-in-{j+i}')for i,r in enumerate(rows)));j+=len(rows)
    return tuple(out)


def header(args,search):
    """The run's identity for READY and the telemetry: code, champion, pricer, constructor and server settings."""
    import jax
    from pokezero import depthn_effect_guard as effect_guard
    return dict(tree_commit=search.tree_commit,rung=search.rung,max_requests=search.max_requests,min_price_seconds=search.min_price_seconds,
        session_cap=search.session_cap,refuse_classes=list(search.refuse_classes),
        redact_private_reveals=bool(getattr(search,'redact_private_reveals',False)),**({'peer_view_b':True}if getattr(search,'peer_view_b',False)is True else{}),
        batch_wait_seconds=args.batch_wait_seconds,queue_limit=args.queue_limit,constructor_seed=args.seed,
        recipe_seed=args.recipe_seed,saved_root_admission=args.saved_root_admission,**({'incremental_roots':True}if getattr(args,'incremental_roots',False)else{}),
        **({'eager_peer_fold':True}if getattr(args,'eager_peer_fold',False)else{}),
        **({}if not getattr(search,'ability_guard',False)else dict(ability_guard=dict(arms={f:list(a)for f,a in effect_guard.ABILITY_ARMS.items()},
            pinch_root=effect_guard.PINCH_ROOT,families=dict(effect_guard.ABILITY_FAMILIES)))),
        metadata_sha256=args.expected_metadata_sha256,
        vocabulary_sha256=args.expected_vocabulary_sha256,champion_archive_sha256=args.champion_archive_sha256,
        champion_checkpoint_sha256=args.champion_checkpoint_sha256,champion_precision=args.champion_precision,pricer=search.pricer_identity,
        **({}if not getattr(args,'reply_archive',None)else dict(root_operator='br_fp',reply_archive_sha256=args.reply_archive_sha256,
            reply_checkpoint_sha256=args.reply_checkpoint_sha256,reply_views=args.reply_views,reply_archive_view=reply_view(getattr(getattr(getattr(search,'reply',None),'evaluator',None),'meta',None)))),
        jax_enable_x64=bool(jax.config.jax_enable_x64),matmul_precision=matmul_precision(),matmul_precision_pin=args.matmul_precision,
        **({}if getattr(args,'root_operator',live.ROOT_OPERATOR)==live.ROOT_OPERATOR else dict(live_root_operator=args.root_operator)),
        **({}if not getattr(args,'prepare_workers',0)else dict(prepare_workers=args.prepare_workers,prepare_cpus=args.prepare_cpus,
            prepare_timeout_seconds=args.prepare_timeout_seconds,batch_window_seconds=args.batch_window_seconds,
            flush_guard_seconds=args.flush_guard_seconds,reply_forward=args.reply_forward,local_battles=args.local_battles,
            **({}if not args.split_worlds else dict(split_worlds=args.split_worlds,split_min_worlds=args.split_min_worlds)))),
        **({}if not getattr(args,'prepare_warm_in_sha256',None)else dict(prepare_warm_in_sha256=args.prepare_warm_in_sha256)))


def main(argv=None):
    args=options(argv);saved=read_requests(args.warmup_requests);warm_in=read_warm_in(args);pool=make_pool(args)
    try:   # the workers start before build(): whatever fails from here (build, header, run) never orphans them (close is idempotent)
        search=build(args)
        run(search,host=args.host,port=args.port,warmup_requests=saved,max_requests=args.max_requests,batch_wait_seconds=args.batch_wait_seconds,
            queue_limit=args.queue_limit,telemetry=args.telemetry,header=header(args,search),server_id=args.server_id or f'{socket.gethostname()}:{args.port}',
            pool=pool,pipeline_options=None if pool is None else dict(window_seconds=args.batch_window_seconds,guard_seconds=args.flush_guard_seconds,
                reply_forward=args.reply_forward,local_battles=args.local_battles,split_worlds=args.split_worlds,
                split_min_worlds=args.split_min_worlds),warm_in=warm_in)
    finally:
        if pool is not None:pool.close()


if __name__=='__main__':main()
