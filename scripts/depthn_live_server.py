#!/usr/bin/env python3
"""One long-lived GPU listener answering ladder search requests with DepthN depth 2 over ProtocolConstructor worlds.

Design. build() loads Codex's pinned constructor inputs (phase_d_protocol_constructor.load_inputs: B2 catalog, bound
source, engine metadata, owning config), the champion (champion_evaluator's flags: archive and checkpoint sha256s, bf16 by
default), one ProtocolConstructor (W from the rung, sessions keyed by the rung), the executor over the same metadata and
catalog, and one DepthNPricer per capacity bucket (depthn_live_search.make_pricers), and returns a DepthNLiveSearch
(--refuse-classes passed through: those root classes are refused before pricing; recorded in the header and READY;
--root-operator passed through: ARGMAX_MEAN_COMPLETED_Q, the default, TARGET_ARGMAX or NOISE_GATED_K2, recorded in the pricer identity the
header, READY and every response carry, in every response's root_operator and, when not the default, in the header's own
root_operator). run()
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
from gpu_engine import champion_evaluator as evaluators,depthn_live_search as live


class ArrivalQueue(asyncio.Queue):
    """asyncio.Queue stamping request[live.ARRIVAL] = arrival on each (request, future, arrival) item it accepts."""
    def _put(self,item):
        if item is not None:item[0][live.ARRIVAL]=item[2]
        super()._put(item)


class Server(original.Server):
    def __init__(self,search,*,max_requests=4,batch_wait_seconds=.005,queue_limit=256):
        if not callable(getattr(search,'process_many',None)):raise ValueError('a search object with process_many required')
        self._initialize(search,max_requests,batch_wait_seconds,queue_limit)
        self.queue=ArrivalQueue(queue_limit);self.loop=self.task=None;self.stopping=False

    async def serve(self,host,port,*,on_ready=None):
        """Codex's serve (listener and consumer; on exit it drains accepted work and shuts the executor down), plus on_ready()
        once the listener accepts a connection, and stop() from any thread (SIGTERM when serving on the main thread)."""
        self.loop=asyncio.get_running_loop();self.task=asyncio.ensure_future(super().serve(host,port))
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

    def stop(self):
        """Stop serving, from any thread."""
        self.stopping=True
        if self.loop is not None:self.loop.call_soon_threadsafe(self.task.cancel)


def request_row(request,response,finished):
    """A telemetry row: the response, the request's address (not its lines, which grow every decision: their count), its
    arrival stamp and the server-side seconds from arrival to the result."""
    q=request if isinstance(request,dict)else{};arrival=q.get(live.ARRIVAL)
    return dict(kind='request',response=response,address={k:q.get(k)for k in('battle_id','owner','stream_id','decision_index','generation','prefix_sha256')},
        rung=q.get('rung'),budget_seconds=q.get('budget_seconds'),lines=len(q['lines'])if isinstance(q.get('lines'),list)else None,
        arrival_monotonic=arrival,server_seconds=finished-arrival if type(arrival)in(int,float)else None)


class Recorder:
    """What the Server calls: the search's process_many, its responses stamped with server_id (when one is given), plus telemetry
    rows queued for a writer thread."""
    def __init__(self,search,path=None,*,server_id=None):
        if server_id is not None and (type(server_id)is not str or not server_id):raise ValueError('a non-empty server_id required')
        self.search=search;self.server_id=server_id;self.rows=self.thread=self.file=None
        if path is not None:
            self.file=open(path,'a',encoding='utf-8');self.rows=queue.SimpleQueue()
            self.thread=threading.Thread(target=self._write,name='depthn-live-telemetry',daemon=True);self.thread.start()

    def process_many(self,requests):
        requests=list(requests);responses=self.search.process_many(requests);finished=time.monotonic()
        if self.server_id is not None:responses=[dict(r,server_id=self.server_id)if isinstance(r,dict)else r for r in responses]
        if self.rows is not None:self.rows.put([request_row(q,r,finished)for q,r in zip(requests,responses)])
        return responses

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
        server_id=None):
    """Build the Server (Codex's queue checks), warm `search` up, then serve until stopped; READY is logged (and recorded)
    once it is warm AND its own listener accepts connections. `on_server(server)` sees the Server before the warmup (its
    stop() ends run once serving)."""
    recorder=Recorder(search,telemetry,server_id=server_id)
    try:
        server=Server(recorder,max_requests=max_requests,batch_wait_seconds=batch_wait_seconds,queue_limit=queue_limit)
        if on_server is not None:on_server(server)
        warm=search.warmup(warmup_requests);buckets=warm.get('buckets',{})
        ready=dict(event='READY',host=host,port=port,server_id=server_id,**(header or{}),cold_seconds={b:x.get('cold_seconds')for b,x in buckets.items()},
            warm_seconds={b:x.get('warm_seconds')for b,x in buckets.items()},programs=warm.get('programs'),carry_reference=warm.get('carry_reference'),warmup=warm)
        def announce():log(ready);recorder.record(dict(ready,kind='ready'))
        asyncio.run(server.serve(host,port,on_ready=announce))
    finally:recorder.close()


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
    p.add_argument('--rung',choices=tuple(live.RUNGS),default='DN2W8',help='DN2W8 (W=8) or DN2W16 (W=16)')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,required=True)
    p.add_argument('--buckets',type=buckets_arg,default=(1,2,4),help='capacity buckets, in roots per price call (default 1,2,4)')
    p.add_argument('--max-requests',type=int,help='requests per batch (default the largest bucket; set from gap 4B)')
    p.add_argument('--batch-wait-seconds',type=float,default=.005);p.add_argument('--queue-limit',type=int,default=256)
    p.add_argument('--min-price-seconds',type=float,default=1.,help='refuse expired_in_queue when the queue wait leaves less of the budget')
    p.add_argument('--session-cap',type=int,default=256,help='constructor sessions kept, by least-recent battle')
    p.add_argument('--refuse-classes',type=classes_arg,default=(),help='root classes refused before pricing (class_refused), comma-separated from '
        +', '.join(live.CLASSES)+' (default none)')
    p.add_argument('--root-operator',choices=live.ROOT_OPERATORS,default=live.ROOT_OPERATOR,
        help='the root operator that picks the played action (default ARGMAX_MEAN_COMPLETED_Q: averaging\'s argmax; TARGET_ARGMAX: the RR '
        'scorer\'s target_argmax on the reported policy_target; NOISE_GATED_K2: the RR scorer\'s noise_gated k=2, averaging\'s action only '
        'past the per-world gap gate over the root logits\' greedy); recorded in the pricer identity of READY, the header and every response')
    p.add_argument('--seed',type=int,default=20261103,help='constructor world seed (Codex\'s default)')
    p.add_argument('--recipe-seed',type=int,default=20261103,help='the pricer\'s lane-order recipe seed')
    p.add_argument('--saved-root-admission',action='store_true',help='Codex\'s stateless mode: every prefix folded from scratch, no sessions')
    p.add_argument('--block-lanes',type=int,default=2187);p.add_argument('--blocks-per-call',type=int,default=16)
    p.add_argument('--warmup-requests',required=True,help='JSONL of saved Showdown requests, one object per line with the eight address keys of '
        'Codex\'s ladder request (battle_id, owner, stream_id, decision_index, generation, prefix_sha256, lines, own_request_line)')
    p.add_argument('--telemetry',help='server-side JSONL, appended, fsynced per batch')
    p.add_argument('--server-id',help='this process\'s name in READY and every response (default HOSTNAME:PORT)')
    p.add_argument('--tree-commit',help='the code commit served (default: this checkout\'s HEAD, +dirty when tracked files differ)')
    p.add_argument('--compilation-cache',help='absolute directory for JAX\'s persistent compilation cache')
    p.add_argument('--matmul-precision',choices=('highest','high','default'),default='highest',
        help='the pinned JAX default matmul precision (default highest, the gates\' certified precision); the server refuses to start '
        'unless JAX_DEFAULT_MATMUL_PRECISION and JAX\'s own setting both name it')
    return p


def options(argv=None):
    p=parser();args=p.parse_args(argv);evaluators.check_arguments(p,args);check_precision_environment(p,args)
    if not args.champion_archive:p.error('--champion-archive with its two sha256 flags is required: the champion prices every root and leaf')
    if args.max_requests is None:args.max_requests=args.buckets[-1]
    if not 1<=args.max_requests<=args.buckets[-1]:p.error('--max-requests must be in 1..the largest bucket')
    try:   # Codex's own queue/batch checks (phase_d_search_server.Server._initialize), before anything loads or compiles
        Server(SimpleNamespace(process_many=list),max_requests=args.max_requests,batch_wait_seconds=args.batch_wait_seconds,
            queue_limit=args.queue_limit).executor.shutdown()
    except ValueError as error:p.error(f'--max-requests/--batch-wait-seconds/--queue-limit: {error}')
    return args


def read_requests(path):
    """The saved requests of a JSONL file: one JSON object per non-empty line, at least one."""
    with open(path,encoding='utf-8')as f:rows=[json.loads(line)for line in f if line.strip()]
    if not rows or not all(isinstance(r,dict)for r in rows):raise ValueError(f'{path}: one JSON object per line, at least one, required')
    return rows


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
    if args.compilation_cache:
        from pokezero.owning_worker_runtime import configure_compilation_cache
        configure_compilation_cache(args.compilation_cache)
    if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':raise ValueError('ONE visible GPU required')
    from pokezero import v5_gumbel
    from gpu_engine import constructor_worlds as cw
    from gpu_engine.phase_d_protocol_constructor import ProtocolConstructor,load_inputs
    catalog,source,metadata,config=load_inputs(args);evaluator=evaluators.load_from_arguments(args);W=live.RUNGS[args.rung]
    constructor=ProtocolConstructor(catalog=catalog,source=source,metadata=metadata,owning_config=config,count=W,seed=args.seed,rung=args.rung,
        saved_root_admission=args.saved_root_admission)
    executor=cw.live_executor(metadata,catalog)   # the constructor's own engine metadata and catalog (the depth-N gate's CONSTRUCTOR_ROOTS uses the same)
    params=(evaluator.params,)
    pricers=live.make_pricers(executor,evaluator,params,worlds=W,buckets=args.buckets,block_lanes=args.block_lanes,blocks_per_call=args.blocks_per_call)
    return live.DepthNLiveSearch(constructor=constructor,pricers=pricers,params=params,binding=cw.binding(source,metadata,catalog),
        recipe=v5_gumbel.Recipe(args.recipe_seed),tree_commit=args.tree_commit or tree_commit(),min_price_seconds=args.min_price_seconds,
        max_requests=args.max_requests,session_cap=args.session_cap,refuse_classes=args.refuse_classes,
        root_operator=getattr(args,'root_operator',live.ROOT_OPERATOR))


def header(args,search):
    """The run's identity for READY and the telemetry: code, champion, pricer, constructor and server settings."""
    import jax
    return dict(tree_commit=search.tree_commit,rung=search.rung,max_requests=search.max_requests,min_price_seconds=search.min_price_seconds,
        session_cap=search.session_cap,refuse_classes=list(search.refuse_classes),batch_wait_seconds=args.batch_wait_seconds,queue_limit=args.queue_limit,constructor_seed=args.seed,
        recipe_seed=args.recipe_seed,saved_root_admission=args.saved_root_admission,metadata_sha256=args.expected_metadata_sha256,
        vocabulary_sha256=args.expected_vocabulary_sha256,champion_archive_sha256=args.champion_archive_sha256,
        champion_checkpoint_sha256=args.champion_checkpoint_sha256,champion_precision=args.champion_precision,pricer=search.pricer_identity,
        jax_enable_x64=bool(jax.config.jax_enable_x64),matmul_precision=matmul_precision(),matmul_precision_pin=args.matmul_precision,
        **({}if getattr(search,'root_operator',live.ROOT_OPERATOR)==live.ROOT_OPERATOR else dict(root_operator=search.root_operator)))


def main(argv=None):
    args=options(argv);saved=read_requests(args.warmup_requests);search=build(args)
    run(search,host=args.host,port=args.port,warmup_requests=saved,max_requests=args.max_requests,batch_wait_seconds=args.batch_wait_seconds,
        queue_limit=args.queue_limit,telemetry=args.telemetry,header=header(args,search),server_id=args.server_id or f'{socket.gethostname()}:{args.port}')


if __name__=='__main__':main()
