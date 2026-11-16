"""The live server on CPU: fake (and one real fake-engine) search objects through Codex's real Server queue over localhost."""
import asyncio
from contextlib import ExitStack,redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pokezero import phase_d_search_server as original
PINNED={'JAX_DEFAULT_MATMUL_PRECISION':'highest'}
from gpu_engine import depthn_live_search as live
from scripts import depthn_live_server as S


def free_port():
    with socket.socket()as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]


def request(i,**kw):return{**dict(schema=original.SCHEMA,request_id=f'r{i}',battle_id=f'b{i}',lines=[f'|line|{i}']),**kw}


def blocking_request(port,request,timeout=10.):
    """Codex's client framing on a blocking socket (phase_d_ladder_transport.Client.request's shape)."""
    with socket.create_connection(('127.0.0.1',port),timeout=timeout)as stream:
        stream.sendall(original.encode(request))
        def exact(n):
            out=bytearray()
            while len(out)<n:
                piece=stream.recv(n-len(out))
                if not piece:raise ConnectionError('closed without a result')
                out.extend(piece)
            return bytes(out)
        return json.loads(exact(struct.unpack('!I',exact(4))[0]))


class FakeSearch:
    """process_many answers each request in order after `delay`, echoing the arrival stamp it saw; warmup waits for `release`."""
    def __init__(self,delay=0.,release=None):self.batches=[];self.delay=delay;self.release=release;self.warmed=False

    def warmup(self,requests):
        if self.release is not None and not self.release.wait(10):raise TimeoutError('never released')
        self.warmed=True;self.saved=list(requests)
        return dict(buckets={1:dict(cold_seconds=1.5,warm_seconds=.5),2:dict(cold_seconds=2.5,warm_seconds=.75)},programs=7,
            carry_reference=dict(battle_id='w',decision_index=3,root_classes=['sleep']))

    def process_many(self,requests):
        self.batches.append((time.monotonic(),[dict(r)for r in requests]));time.sleep(self.delay)
        return[dict(schema=original.SCHEMA,request_id=r['request_id'],status='COMPLETE',warmed=self.warmed,arrival=r.get(live.ARRIVAL))for r in requests]


class FakePipeline:
    """A batching pipeline stand-in: holds what it is submitted; the test completes requests in any order it likes."""
    def __init__(self):self.submitted=[];self.lock=threading.Lock();self.closed=False

    def submit(self,request,done):
        with self.lock:self.submitted.append((request,done))

    def count(self):
        with self.lock:return len(self.submitted)

    def complete(self,i,**extra):
        request,done=self.submitted[i];done(dict(schema=original.SCHEMA,request_id=request['request_id'],status='COMPLETE',n=i,**extra))

    def close(self):self.closed=True


async def serve_forever_closing_clients(self):
    """CPython 3.13.18's asyncio.Server.serve_forever (Lib/asyncio/base_events.py), installed by the stop tests whatever this
    Python's patch level: a cancelled serve_forever also closes every client connection (close_clients), before an accepted
    request is answered. The server's stop must not rely on cancelling it."""
    if self._serving_forever_fut is not None:raise RuntimeError(f'server {self!r} is already being awaited on serve_forever()')
    if self._sockets is None:raise RuntimeError(f'server {self!r} is closed')
    self._start_serving();self._serving_forever_fut=self._loop.create_future()
    try:await self._serving_forever_fut
    except asyncio.CancelledError:
        try:self.close();self.close_clients();await self.wait_closed()
        finally:raise
    finally:self._serving_forever_fut=None


def newer_asyncio():
    import asyncio.base_events
    return patch.object(asyncio.base_events.Server,'serve_forever',serve_forever_closing_clients)


async def until(test,timeout=10.):
    deadline=time.monotonic()+timeout
    while not test():
        if time.monotonic()>deadline:raise TimeoutError('condition never held')
        await asyncio.sleep(.01)


class PipelinedServerTests(unittest.TestCase):
    """Server.consume with a pipeline (--prepare-workers): out-of-order completion, backpressure, drain on stop."""
    def scenario(self,body,*,queue_limit=8,pipeline=None):
        pipeline=pipeline or FakePipeline()
        async def run():
            server=S.Server(FakeSearch(),max_requests=1,queue_limit=queue_limit,pipeline=pipeline);port=free_port();ready=asyncio.Event()
            task=asyncio.create_task(server.serve('127.0.0.1',port,on_ready=ready.set));await asyncio.wait_for(ready.wait(),10)
            try:return await body(server,port,pipeline,task)
            finally:
                if not task.done():server.stop();await asyncio.wait_for(task,10)
        return asyncio.run(run())

    @staticmethod
    async def client(port,i,got):
        reader,writer=await asyncio.open_connection('127.0.0.1',port)
        writer.write(original.encode(request(i)));await writer.drain()
        row=await asyncio.wait_for(original.receive(reader),20);got.append(row['request_id'])
        writer.close();await writer.wait_closed();return row

    def test_completions_out_of_order_reach_their_own_connections(self):
        async def body(server,port,pipeline,task):
            got=[];clients=[asyncio.create_task(self.client(port,i,got))for i in range(2)]
            await until(lambda:pipeline.count()==2)
            order=sorted(range(2),key=lambda k:pipeline.submitted[k][0]['request_id'],reverse=True)   # r1 first, then r0
            for k in order:pipeline.complete(k);await asyncio.sleep(.05)
            rows=await asyncio.gather(*clients);return rows,got
        rows,got=self.scenario(body)
        self.assertEqual([r['request_id']for r in rows],['r0','r1']);self.assertEqual(got,['r1','r0'])   # each its own answer, as completed
        self.assertTrue(all(r['server_wall_seconds']>=0 for r in rows))

    def test_a_response_for_another_request_closes_that_connection(self):
        async def body(server,port,pipeline,task):
            got=[];client=asyncio.create_task(self.client(port,0,got));await until(lambda:pipeline.count()==1)
            request_,done=pipeline.submitted[0];done(dict(schema=original.SCHEMA,request_id='someone-else',status='COMPLETE'))
            with self.assertRaises((asyncio.IncompleteReadError,ConnectionError)):await client
            return got
        self.assertEqual(self.scenario(body),[])

    def test_in_flight_is_capped_at_the_queue_limit(self):
        async def body(server,port,pipeline,task):
            got=[];clients=[asyncio.create_task(self.client(port,i,got))for i in range(5)]
            await until(lambda:pipeline.count()==2);await asyncio.sleep(.3)
            capped=(pipeline.count(),server.inflight)   # the rest wait in Codex's bounded queue (and the handle loop's put)
            for k in range(5):
                await until(lambda:pipeline.count()>k);pipeline.complete(k)
            await asyncio.gather(*clients);return capped,sorted(got)
        capped,got=self.scenario(body,queue_limit=2)
        self.assertEqual(capped,(2,2));self.assertEqual(got,[f'r{i}'for i in range(5)])

    def test_stop_answers_everything_accepted_then_closes_the_pipeline(self):
        async def body(server,port,pipeline,task):
            got=[];clients=[asyncio.create_task(self.client(port,i,got))for i in range(3)]
            await until(lambda:pipeline.count()==3)
            server.stop();await asyncio.sleep(.2)
            self.assertFalse(task.done());self.assertFalse(pipeline.closed)   # draining: three answers owed
            for k in range(3):pipeline.complete(k)
            rows=await asyncio.gather(*clients);await asyncio.wait_for(task,10)
            return rows,pipeline.closed
        with newer_asyncio():rows,closed=self.scenario(body)   # CI's CPython 3.13.18 closed the three connections at stop()
        self.assertEqual(sorted(r['request_id']for r in rows),['r0','r1','r2']);self.assertTrue(closed)

    def test_a_fatal_error_closes_its_connection(self):
        async def body(server,port,pipeline,task):
            got=[];client=asyncio.create_task(self.client(port,0,got));await until(lambda:pipeline.count()==1)
            pipeline.submitted[0][1](MemoryError('out'))
            with self.assertRaises((asyncio.IncompleteReadError,ConnectionError)):await client
            return got
        self.assertEqual(self.scenario(body),[])


class ServerTests(unittest.TestCase):
    def test_stop_answers_the_requests_already_accepted(self):
        """Serial serving (no pipeline): a stop while accepted requests are being searched answers them first, then serve
        returns, under CPython 3.13.18's serve_forever too (it closes client connections when cancelled)."""
        async def scenario():
            search=FakeSearch(delay=.4);server=S.Server(search,max_requests=4);port=free_port();ready=asyncio.Event()
            task=asyncio.create_task(server.serve('127.0.0.1',port,on_ready=ready.set));await asyncio.wait_for(ready.wait(),10)
            async def client(i):
                reader,writer=await asyncio.open_connection('127.0.0.1',port)
                writer.write(original.encode(request(i)));await writer.drain()
                row=await asyncio.wait_for(original.receive(reader),20);writer.close();await writer.wait_closed();return row
            clients=[asyncio.create_task(client(i))for i in range(2)]
            await until(lambda:sum(len(b)for _,b in search.batches)==2)   # both accepted and being searched
            server.stop();rows=await asyncio.gather(*clients);await asyncio.wait_for(task,10)
            return rows,server.closing
        with newer_asyncio():rows,closing=asyncio.run(scenario())
        self.assertEqual(sorted(r['request_id']for r in rows),['r0','r1']);self.assertTrue(closing)

    def test_requests_batch_through_codexs_queue_with_arrival_stamps(self):
        async def scenario():
            search=FakeSearch(delay=.05);server=S.Server(search,max_requests=4,batch_wait_seconds=.05);port=free_port();ready=asyncio.Event()
            task=asyncio.create_task(server.serve('127.0.0.1',port,on_ready=ready.set))
            await asyncio.wait_for(ready.wait(),10);before=time.monotonic()
            async def client(i):
                reader,writer=await asyncio.open_connection('127.0.0.1',port)
                writer.write(original.encode(request(i,**({live.ARRIVAL:-5.}if i==0 else{}))));await writer.drain()   # a forged stamp
                row=await asyncio.wait_for(original.receive(reader),10);writer.close();await writer.wait_closed();return row
            rows=await asyncio.gather(*(client(i)for i in range(6)))
            big=await asyncio.to_thread(blocking_request,port,request(9,lines=['|big|'+'x'*100]*20000))   # a long prefix, Codex's framing
            server.stop();await asyncio.wait_for(task,10)
            return search,rows,big,before
        search,rows,big,before=asyncio.run(scenario())
        self.assertEqual([r['request_id']for r in rows],[f'r{i}'for i in range(6)]);self.assertEqual(big['request_id'],'r9')   # each client its own answer
        for row in rows+[big]:
            self.assertEqual((row['schema'],row['status']),(original.SCHEMA,'COMPLETE'))
            self.assertGreaterEqual(row['arrival'],before);self.assertGreaterEqual(row['server_wall_seconds'],.05)   # stamped by the queue, never the client's
        sizes=[len(batch)for _,batch in search.batches];self.assertEqual(sum(sizes),7);self.assertLessEqual(max(sizes),4);self.assertGreater(max(sizes),1)
        for started,batch in search.batches:
            self.assertTrue(all(before<=r[live.ARRIVAL]<=started for r in batch))   # arrival precedes the batch that serves it
        self.assertEqual(len(search.batches[-1][1][0]['lines']),20000)

    def test_ready_only_after_warmup_then_serving_and_telemetry(self):
        release=threading.Event();search=FakeSearch(release=release);port=free_port();logs=[];servers=[]
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'telemetry.jsonl'
            fsync=os.fsync;synced=[]
            def counting(fd):synced.append(fd);return fsync(fd)
            thread=threading.Thread(target=S.run,kwargs=dict(search=search,host='127.0.0.1',port=port,warmup_requests=[dict(battle_id='w')],max_requests=4,
                telemetry=str(path),header=dict(tree_commit='abc'),log=logs.append,on_server=servers.append,server_id='gpu0:7001'),daemon=True)
            patcher=patch.object(S.os,'fsync',side_effect=counting);patcher.start();self.addCleanup(patcher.stop)
            thread.start();time.sleep(.3)
            self.assertEqual((logs,search.warmed,len(servers)),([],False,1))   # warming: the Server is built (its checks ran), no READY, nothing listening
            with self.assertRaises(OSError):socket.create_connection(('127.0.0.1',port),timeout=1).close()
            release.set();deadline=time.monotonic()+10
            while not logs and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(len(logs),1);ready=logs[0]
            self.assertEqual((ready['event'],ready['tree_commit'],ready['port'],ready['programs'],ready['cold_seconds'],ready['warm_seconds']),
                ('READY','abc',port,7,{1:1.5,2:2.5},{1:.5,2:.75}))
            self.assertEqual(ready['carry_reference'],dict(battle_id='w',decision_index=3,root_classes=['sleep']))   # the warmup's reference root
            self.assertTrue(search.warmed);self.assertEqual(search.saved,[dict(battle_id='w')])
            row=blocking_request(port,request(1,budget_seconds=2.5));self.assertTrue(row['warmed'])   # served only after warmup
            self.assertEqual((ready['server_id'],row['server_id']),('gpu0:7001','gpu0:7001'))   # which process answered, for a pool's readout
            servers[0].stop();thread.join(10);self.assertFalse(thread.is_alive())
            rows=[json.loads(line)for line in path.read_text().splitlines()];self.assertGreaterEqual(len(synced),2)   # each written batch fsynced
        self.assertEqual([r['kind']for r in rows],['ready','request']);self.assertEqual(rows[0]['event'],'READY')
        self.assertEqual((rows[1]['response']['request_id'],rows[1]['address']['battle_id'],rows[1]['lines'],rows[1]['budget_seconds']),('r1','b1',1,2.5))
        self.assertEqual(rows[1]['arrival_monotonic'],rows[1]['response']['arrival']);self.assertGreaterEqual(rows[1]['server_seconds'],0)
        self.assertEqual(rows[1]['response']['server_id'],'gpu0:7001')
        keys=set(S.CompileCounter.KEYS);self.assertEqual(set(ready['warmup_compiles']),keys);self.assertLessEqual(keys,set(rows[1]))   # READY: the warmup's; a row: its own

    def test_the_compile_counter_reads_compiles_cache_hits_misses_and_unreadable_entries_as_deltas(self):
        import jax,warnings
        counter=S.CompileCounter();counter.delta()
        jax.monitoring.record_event_duration_secs(S.CompileCounter.EVENT,.25);jax.monitoring.record_event(S.CompileCounter.HIT)
        jax.monitoring.record_event(S.CompileCounter.HIT);jax.monitoring.record_event(S.CompileCounter.MISS)
        with warnings.catch_warnings():
            warnings.simplefilter('always')   # shown (through the counter's hook), however often this test runs
            warnings.warn("Error reading persistent compilation cache entry for 'jit_x': error: truncated")
        got=counter.delta();self.assertEqual({k:got[k]for k in('compiles','cache_hits','cache_misses')},dict(compiles=1,cache_hits=2,cache_misses=1))
        self.assertGreaterEqual(got['compile_seconds'],.25);self.assertGreaterEqual(got['cache_read_errors'],1)
        self.assertEqual(counter.delta(),dict(compiles=0,compile_seconds=0.,cache_hits=0,cache_misses=0,cache_read_errors=0))

    def test_run_hands_prepare_its_warm_in_list(self):
        seen=[]
        class Warming(FakeSearch):
            def warmup(self,requests,warm_in=()):seen.append((list(requests),list(warm_in)));return super().warmup(requests)
        with patch.object(S.asyncio,'run',lambda coroutine:coroutine.close()):   # no serving: warmup, READY's setup, done
            S.run(Warming(),host='127.0.0.1',port=free_port(),warmup_requests=[dict(battle_id='w')],max_requests=4,log=lambda row:None,
                warm_in=(dict(battle_id='x'),))
            S.run(FakeSearch(),host='127.0.0.1',port=free_port(),warmup_requests=[dict(battle_id='w')],max_requests=4,log=lambda row:None)   # no list: warmup(requests)
        self.assertEqual(seen,[([dict(battle_id='w')],[dict(battle_id='x')])])

    def test_the_in_process_slot_warms_in_as_a_worker_does(self):
        """With a pool and an in-process slot (local_battles), warmup's prepares run in this process: its warm-in is the
        workers' two passes (warmup_saved: the warmup requests, leniently as the workers; then the list), or the warmup requests
        alone without a list (R1: the slot warm at READY without --prepare-warm-in). Without the slot nothing prepares here, so
        there is no warm-in."""
        seen=[]
        class Warming(FakeSearch):
            constructor=object()
            def warmup(self,requests,warm_in=(),warm_in_saved=False):seen.append((tuple(warm_in),warm_in_saved));return super().warmup(requests)
            def attach_pool(self,pool):self.pool=pool
        class Pool:
            size=1;workers=();closed=False
            def attach_local(self,constructor):self.local=constructor
            def wait_ready(self):pass
            def rss(self):return 0
            def close(self):self.closed=True
        class Pipeline:
            def __init__(self,*a,**kw):pass
            def start(self):return self
        import gpu_engine.depthn_batching as batching
        warmup,listed=[dict(battle_id='w')],(dict(battle_id='x'),dict(battle_id='y'))
        with patch.object(S.asyncio,'run',lambda coroutine:coroutine.close()),patch.object(batching,'Pipeline',Pipeline):
            for kw in(dict(warm_in=listed),dict()):
                pool=Pool();S.run(Warming(),host='127.0.0.1',port=free_port(),warmup_requests=warmup,max_requests=4,log=lambda row:None,
                    pool=pool,pipeline_options=dict(local_battles=2),**kw)
                self.assertTrue(pool.closed)
            S.run(Warming(),host='127.0.0.1',port=free_port(),warmup_requests=warmup,max_requests=4,log=lambda row:None,pool=Pool(),
                pipeline_options=dict(local_battles=0),warm_in=listed)   # workers only: they warm in at their start
        self.assertEqual(seen,[(listed,True),((),True),((),False)])

    def test_main_closes_the_pool_whatever_fails_after_it_starts(self):
        """The prepare workers start before build(): a build, a header or a telemetry file that raises closes them (no
        orphaned workers warming in); so does a run that ends."""
        class Pool:
            def close(self):closed.append(True)
        args=NS(warmup_requests='w.jsonl',host='127.0.0.1',port=free_port(),max_requests=4,batch_wait_seconds=.005,queue_limit=256,telemetry=None,server_id='s',
            batch_window_seconds=.05,flush_guard_seconds=.15,reply_forward='reduce',local_battles=0,split_worlds=0,split_min_worlds=0)
        for name,kw in(('build',dict(build=RuntimeError('no device'))),('header',dict(header=RuntimeError('no header'))),
                       ('telemetry',dict(Recorder=OSError('telemetry unwritable')))):
            with self.subTest(failing=name):
                closed=[];stand={k:patch.object(S,k,side_effect=v)for k,v in kw.items()}
                with patch.object(S,'options',return_value=args),patch.object(S,'read_requests',return_value=[]),patch.object(S,'read_warm_in',return_value=()),\
                        patch.object(S,'make_pool',return_value=Pool()),patch.object(S,'build',return_value=FakeSearch()),patch.object(S,'header',return_value={}):
                    with stand[next(iter(kw))],self.assertRaises((RuntimeError,OSError)):S.main([])
                self.assertGreaterEqual(len(closed),1)

    def test_server_id_stamps_responses_only_when_given(self):
        plain=S.Recorder(FakeSearch()).process_many([request(1)]);self.assertNotIn('server_id',plain[0])   # no id: the search's responses, unchanged
        named=S.Recorder(FakeSearch(),server_id='h:1').process_many([request(1),request(2)])
        self.assertEqual([r['server_id']for r in named],['h:1','h:1'])
        for bad in('',3):
            with self.subTest(bad=bad),self.assertRaisesRegex(ValueError,'server_id'):S.Recorder(FakeSearch(),server_id=bad)

    def test_a_foreign_listener_on_the_port_never_triggers_ready(self):
        """The review's repro (ready_race.py): a stale listener already holds the port. Its accepted probe must not count as
        this server's: no READY, and serve fails cleanly on its own bind."""
        async def scenario():
            stale=socket.socket();stale.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);stale.bind(('127.0.0.1',0));stale.listen(16)
            port=stale.getsockname()[1];fired=[];server=S.Server(FakeSearch())
            try:
                with self.assertRaises(OSError):await asyncio.wait_for(server.serve('127.0.0.1',port,on_ready=lambda:fired.append('READY')),10)
            finally:stale.close()
            return fired,server
        for _ in range(5):
            fired,server=asyncio.run(scenario());self.assertEqual(fired,[]);self.assertEqual(server.clients,set())
        with tempfile.TemporaryDirectory()as tmp,socket.socket()as stale:   # through run(): warm, then the bind fails; no READY line
            stale.bind(('127.0.0.1',0));stale.listen(4);logs=[];search=FakeSearch()
            with self.assertRaises(OSError):S.run(search,host='127.0.0.1',port=stale.getsockname()[1],warmup_requests=[{}],max_requests=4,
                telemetry=str(Path(tmp)/'t.jsonl'),log=logs.append)
            self.assertEqual((logs,search.warmed,(Path(tmp)/'t.jsonl').read_text()),([],True,''))

    def test_queue_settings_fail_before_the_warmup(self):
        search=FakeSearch()
        for bad in(dict(batch_wait_seconds=.5),dict(queue_limit=2),dict(max_requests=0)):
            with self.subTest(**bad),self.assertRaisesRegex(ValueError,'bounded shared search queue'):
                S.run(search,host='127.0.0.1',port=free_port(),warmup_requests=[{}],**{'max_requests':4,**bad})
        self.assertFalse(search.warmed)

    def test_build_refuses_a_shared_cache_without_whole_entry_writes(self):
        """--compilation-cache with an atomic_compilation_cache.install() that cannot patch this JAX: build() refuses, before it
        configures the cache or touches the device; without the flag the server starts (nothing shares a cache)."""
        from gpu_engine import atomic_compilation_cache as atomic
        base=['--showdown-root','s','--node-binary','n','--metadata','m','--expected-metadata-sha256','a'*64,'--vocabulary','v',
            '--expected-vocabulary-sha256','b'*64,'--port','9000','--warmup-requests','w.jsonl','--champion-archive','c.npz','--champion-archive-sha256','c'*64,
            '--champion-checkpoint-sha256','d'*64,'--tree-commit','t'*40]
        with tempfile.TemporaryDirectory()as tmp:
            with patch.dict(os.environ,PINNED):args=S.options(base+['--compilation-cache',tmp])
            with patch.object(atomic,'install',return_value=False),patch.object(S,'matmul_precision',return_value='highest'):
                with self.assertRaisesRegex(ValueError,'whole-entry cache writes'):S.build(args)

    def test_build_passes_the_command_line_through(self):
        """build() on stand-ins for the GPU, Codex's inputs, the champion and the pricers: --refuse-classes (and the queue
        settings) reach the DepthNLiveSearch it returns."""
        import jax
        from gpu_engine import constructor_worlds as cw,device_public_teacher as teacher,engine_device_observation,phase_d_protocol_constructor as pc,transition
        base=['--showdown-root','s','--node-binary','n','--metadata','m','--expected-metadata-sha256','a'*64,'--vocabulary','v',
            '--expected-vocabulary-sha256','b'*64,'--port','9000','--warmup-requests','w.jsonl','--champion-archive','c.npz','--champion-archive-sha256','c'*64,
            '--champion-checkpoint-sha256','d'*64,'--refuse-classes','wish,sleep','--max-requests','2','--session-cap','64','--tree-commit','t'*40]
        with patch.dict(os.environ,PINNED):args=S.options(base)
        made={}
        stand_ins=((S,'matmul_precision',lambda:'highest'),(jax,'devices',lambda:[NS(platform='gpu')]),(pc,'load_inputs',lambda a:('catalog','source','metadata','config')),
            (S.evaluators,'load_from_arguments',lambda a:NS(params='champion')),(pc,'ProtocolConstructor',lambda **kw:NS(**kw)),
            (teacher,'make_executor',lambda *a,**k:'executor'),(transition,'tables_from_metadata',lambda m:'tables'),
            (engine_device_observation,'compile_tables',lambda m,catalog:'observation'),(S.live,'make_pricers',lambda *a,**k:'pricers'),
            (cw,'binding',lambda *a:'binding'),(S.live,'DepthNLiveSearch',lambda **kw:made.update(kw)or'search'))
        with ExitStack()as stack:
            for target,name,value in stand_ins:stack.enter_context(patch.object(target,name,side_effect=value))
            self.assertEqual(S.build(args),'search')
        self.assertEqual((made['refuse_classes'],made['max_requests'],made['session_cap'],made['tree_commit'],made['binding']),(('wish','sleep'),2,64,'t'*40,'binding'))
        self.assertIs(made['redact_private_reveals'],False)   # off by default
        with patch.dict(os.environ,PINNED):args=S.options(base+['--redact-private-reveals'])
        with ExitStack()as stack:
            for target,name,value in stand_ins:stack.enter_context(patch.object(target,name,side_effect=value))
            S.build(args)
        self.assertIs(made['redact_private_reveals'],True)
        self.assertNotIn('peer_view_b',made)   # off: the search is built exactly as before the flag
        with patch.dict(os.environ,PINNED):args=S.options(base+['--peer-view-b'])
        with ExitStack()as stack:
            for target,name,value in stand_ins:stack.enter_context(patch.object(target,name,side_effect=value))
            S.build(args)
        self.assertEqual((made['peer_view_b'],made['redact_private_reveals']),(True,False))   # the search applies the implied redaction
        self.assertIsNone(made['reply'])   # br_fp off by default: the champion's own averaging
        from gpu_engine import depthn_reply_prior as rp
        reply=['--reply-archive','r.npz','--reply-archive-sha256','e'*64,'--reply-checkpoint-sha256','f'*64]
        with patch.dict(os.environ,PINNED):args=S.options(base+reply)
        loaded=[]
        with ExitStack()as stack:
            for target,name,value in stand_ins+((S.evaluators,'load',lambda path,**kw:loaded.append((path,kw))or NS(params='reply')),
                    (rp,'ReplyPrior',lambda executor,model,mode,identity:('reply-prior',executor,model.params,mode,identity))):
                stack.enter_context(patch.object(target,name,side_effect=value))
            self.assertEqual(S.build(args),'search')
        self.assertEqual(loaded,[('r.npz',dict(expected_archive_sha256='e'*64,expected_checkpoint_sha256='f'*64,precision=S.evaluators.PRECISIONS['bf16']))])
        self.assertEqual(made['reply'][0],'reply-prior');self.assertEqual((made['reply'][2],made['reply'][3]),('reply','eager'))
        self.assertEqual(made['reply'][4],dict(archive_sha256='e'*64,checkpoint_sha256='f'*64,view='env'))   # the reply pin's identity
        with patch.dict(os.environ,PINNED):args=S.options(base+reply+['--peer-view-b'])   # both: br_fp on view b's worlds
        with ExitStack()as stack:
            for target,name,value in stand_ins+((S.evaluators,'load',lambda path,**kw:NS(params='reply')),
                    (rp,'ReplyPrior',lambda executor,model,mode,identity:('reply-prior',executor,model.params,mode,identity))):
                stack.enter_context(patch.object(target,name,side_effect=value))
            S.build(args)
        self.assertEqual((made['peer_view_b'],made['reply'][0],made['root_operator']),(True,'reply-prior','ARGMAX_MEAN_COMPLETED_Q'))
        with patch.dict(os.environ,PINNED):args=S.options(base+reply+['--peer-view-b','--root-operator','NOISE_GATED_K2'])   # variant B
        with ExitStack()as stack:
            for target,name,value in stand_ins+((S.evaluators,'load',lambda path,**kw:NS(params='reply')),
                    (rp,'ReplyPrior',lambda executor,model,mode,identity:('reply-prior',executor,model.params,mode,identity))):
                stack.enter_context(patch.object(target,name,side_effect=value))
            S.build(args)
        self.assertEqual((made['root_operator'],made['reply'][0]),('NOISE_GATED_K2','reply-prior'))

    def test_a_view_b_reply_archive_needs_peer_view_b(self):
        self.assertEqual((S.reply_view(None),S.reply_view({}),S.reply_view(dict(extra=dict(view='env')))),('env','env','env'))
        self.assertEqual(S.reply_view(dict(extra=dict(view='b',redaction='peer-view-v2'))),'b')
        with self.assertRaisesRegex(ValueError,'--peer-view-b'):S.check_reply_view(dict(extra=dict(view='b')),NS())
        with self.assertRaisesRegex(ValueError,'--peer-view-b'):S.check_reply_view(dict(extra=dict(view='b')),NS(peer_view_b=False))
        S.check_reply_view(dict(extra=dict(view='b')),NS(peer_view_b=True));S.check_reply_view(dict(extra=dict(view='env')),NS())
        S.check_reply_view(None,NS())   # an archive without export metadata (red2's era): served as before

    def test_the_matmul_precision_is_hard_pinned(self):
        """The environment is checked before anything loads; JAX's own setting again in build, before any input or champion."""
        from gpu_engine import phase_d_protocol_constructor as pc
        base=['--showdown-root','s','--node-binary','n','--metadata','m','--expected-metadata-sha256','a'*64,'--vocabulary','v',
            '--expected-vocabulary-sha256','b'*64,'--port','9000','--warmup-requests','w.jsonl','--champion-archive','c.npz','--champion-archive-sha256','c'*64,
            '--champion-checkpoint-sha256','d'*64]
        for environment in({},{'JAX_DEFAULT_MATMUL_PRECISION':'default'},{'JAX_DEFAULT_MATMUL_PRECISION':'high'}):
            with self.subTest(environment=environment),patch.dict(os.environ,environment,clear=False),redirect_stderr(io.StringIO())as err:
                if not environment:os.environ.pop('JAX_DEFAULT_MATMUL_PRECISION',None)
                with self.assertRaises(SystemExit):S.options(base)
                self.assertIn('JAX_DEFAULT_MATMUL_PRECISION',err.getvalue())
        with patch.dict(os.environ,{'JAX_DEFAULT_MATMUL_PRECISION':'high'}):self.assertEqual(S.options(base+['--matmul-precision','high']).matmul_precision,'high')
        with patch.dict(os.environ,PINNED):args=S.options(base)
        self.assertEqual(args.matmul_precision,'highest')
        with patch.object(S,'matmul_precision',return_value='None'),patch.object(pc,'load_inputs',side_effect=AssertionError('nothing loads'))as load,\
                self.assertRaisesRegex(ValueError,'differs from the pinned'):
            S.build(args)
        load.assert_not_called()
        search=NS(tree_commit='t',rung='DN2W8',max_requests=4,min_price_seconds=1.,session_cap=256,refuse_classes=(),pricer_identity={})
        with patch.object(S,'matmul_precision',return_value='highest'):header=S.header(args,search)
        self.assertEqual((header['matmul_precision'],header['matmul_precision_pin']),('highest','highest'))

    def test_the_arrival_queue_stamps_every_put_and_passes_the_sentinel(self):
        async def scenario():
            q=S.ArrivalQueue(4);a={'x':1};b={live.ARRIVAL:-1.}
            await q.put((a,None,12.5));q.put_nowait((b,None,13.));await q.put(None)
            return[await q.get()for _ in range(3)],a,b
        items,a,b=asyncio.run(scenario())
        self.assertEqual((a[live.ARRIVAL],b[live.ARRIVAL],items[2]),(12.5,13.,None))
        with self.assertRaisesRegex(ValueError,'process_many'):S.Server(object())
        with self.assertRaisesRegex(ValueError,'bounded'):S.Server(FakeSearch(),max_requests=0)   # Codex's own bounds still apply
        self.assertIsInstance(S.Server(FakeSearch()).queue,S.ArrivalQueue)

    def test_a_real_live_search_behind_the_server(self):
        from tests import test_gpu_engine_depthn_live_search as L
        L.LiveSearchTests.setUpClass();fixture=L.LiveSearchTests('test_configuration_is_checked');search=fixture.search()
        outcomes={('s',0):fixture.outcome(0),('t',0):fixture.outcome(1)}
        async def scenario():
            server=S.Server(search,max_requests=search.max_requests,batch_wait_seconds=.05);port=free_port();ready=asyncio.Event()
            task=asyncio.create_task(server.serve('127.0.0.1',port,on_ready=ready.set));await asyncio.wait_for(ready.wait(),10)
            rows=await asyncio.gather(asyncio.to_thread(blocking_request,port,dict(L.request('s',0),**{live.ARRIVAL:0.})),   # forged: would be expired
                asyncio.to_thread(blocking_request,port,L.request('t',0,budget=.5)))   # under min_price_seconds: expired
            server.stop();await asyncio.wait_for(task,30);return rows
        # The fake engine's carries are kept as built (LiveSearchTests.setUp's rule; this fixture instance never runs its setUp).
        with L.Codex(outcomes),patch.object(L.cw,'world_completer',return_value=lambda world,owner,**kw:world):rows=asyncio.run(scenario())
        self.assertEqual([(r['request_id'],r['status'],r['cause'])for r in rows],[('s:0','COMPLETE',None),('t:0','REFUSED','expired_in_queue')])
        self.assertTrue(0<=rows[0]['timings']['queue_seconds']<1);self.assertGreater(rows[0]['server_wall_seconds'],0)
        self.assertEqual(set(rows[0])-{'server_wall_seconds'},L.KEYS)

    def test_command_line(self):
        base=['--showdown-root','s','--node-binary','n','--metadata','m','--expected-metadata-sha256','a'*64,'--vocabulary','v',
            '--expected-vocabulary-sha256','b'*64,'--port','9000','--warmup-requests','w.jsonl']
        champion=['--champion-archive','c.npz','--champion-archive-sha256','c'*64,'--champion-checkpoint-sha256','d'*64]
        environment=patch.dict(os.environ,PINNED);environment.start();self.addCleanup(environment.stop)
        args=S.options(base+champion)
        self.assertEqual((args.rung,args.buckets,args.max_requests,args.champion_precision,args.min_price_seconds,args.session_cap,args.host),
            ('DN2W8',(1,2,4),4,'bf16',1.,256,'127.0.0.1'))
        args=S.options(base+champion+['--rung','DN2W16','--buckets','2,1','--max-requests','1','--refuse-classes','wish,forced_switch'])
        self.assertEqual((args.rung,args.buckets,args.max_requests,args.refuse_classes),('DN2W16',(1,2),1,('wish','forced_switch')))
        self.assertEqual(S.options(base+champion).refuse_classes,())
        search=NS(tree_commit='t',rung='DN2W8',max_requests=4,min_price_seconds=1.,session_cap=256,refuse_classes=('forced_switch','wish'),pricer_identity={})
        self.assertEqual(S.header(args,search)['refuse_classes'],['forced_switch','wish'])   # recorded in READY and the telemetry header
        self.assertIs(S.header(args,search)['redact_private_reveals'],False)
        self.assertIs(S.header(args,NS(vars(search),redact_private_reveals=True))['redact_private_reveals'],True)
        self.assertNotIn('live_root_operator',S.header(args,search))   # the default operator: the header is unchanged
        gated=S.options(base+champion+['--root-operator','NOISE_GATED_K2'])
        self.assertEqual((args.root_operator,gated.root_operator),('ARGMAX_MEAN_COMPLETED_Q','NOISE_GATED_K2'))
        self.assertEqual(S.header(gated,search)['live_root_operator'],'NOISE_GATED_K2')
        with self.assertRaises(SystemExit),redirect_stderr(io.StringIO()):S.options(base+champion+['--root-operator','TARGET_ARGMAX'])
        self.assertNotIn('peer_view_b',S.header(args,search));self.assertNotIn('peer_view_b',S.header(args,NS(vars(search),peer_view_b=False)))
        on=S.header(args,NS(vars(search),peer_view_b=True,redact_private_reveals=True))   # view b's search: the redaction it implies is on
        self.assertEqual((on['peer_view_b'],on['redact_private_reveals']),(True,True))
        self.assertIs(S.options(base+champion+['--peer-view-b']).peer_view_b,True);self.assertIs(S.options(base+champion).peer_view_b,False)
        self.assertNotIn('root_operator',S.header(args,search))
        reply=['--reply-archive','r.npz','--reply-archive-sha256','e'*64,'--reply-checkpoint-sha256','f'*64]
        self.assertEqual({k:S.header(S.options(base+champion+reply),search)[k]for k in('root_operator','reply_archive_sha256','reply_checkpoint_sha256')},
            dict(root_operator='br_fp',reply_archive_sha256='e'*64,reply_checkpoint_sha256='f'*64))
        self.assertEqual(S.header(S.options(base+champion+reply+['--reply-views','check']),search)['reply_views'],'check')
        for bad in(reply[:2],reply[:4],reply[:2]+['--reply-archive-sha256','x'*64,'--reply-checkpoint-sha256','f'*64],['--reply-views','jit']):
            with self.subTest(bad=bad),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):S.options(base+champion+bad)
        both=S.header(S.options(base+champion+reply+['--peer-view-b']),NS(vars(search),peer_view_b=True,redact_private_reveals=True))
        self.assertEqual((both['root_operator'],both['peer_view_b'],both['redact_private_reveals']),('br_fp',True,True))
        for bad in(base,base+champion[:2],base+champion+['--max-requests','5'],base+champion+['--buckets','0,1'],base+champion+['--rung','R1'],
                base+champion[:2]+['--champion-archive-sha256','x'*64,'--champion-checkpoint-sha256','d'*64],base[:-2]+champion,
                base+champion+['--batch-wait-seconds','.5'],base+champion+['--queue-limit','2'],base+champion+['--refuse-classes','nap']):   # the last two: Codex's queue checks, before any load
            with self.subTest(bad=bad[len(base)-2:]),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):S.options(bad)
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'w.jsonl';path.write_text('{"battle_id":"a"}\n\n{"battle_id":"b"}\n')
            self.assertEqual(S.read_requests(path),[dict(battle_id='a'),dict(battle_id='b')])
            # prepare's warm-in list: with its sha256 only, held to it, recorded in the header
            sha=hashlib.sha256(path.read_bytes()).hexdigest();listed=S.options(base+champion+['--prepare-warm-in',str(path),'--prepare-warm-in-sha256',sha])
            self.assertEqual(S.read_warm_in(listed),(dict(battle_id='a'),dict(battle_id='b')));self.assertEqual(S.read_warm_in(args),())
            self.assertEqual(S.header(listed,search)['prepare_warm_in_sha256'],sha);self.assertNotIn('prepare_warm_in_sha256',S.header(args,search))
            for bad in(['--prepare-warm-in',str(path)],['--prepare-warm-in-sha256',sha]):
                with self.subTest(bad=bad),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):S.options(base+champion+bad)
            with self.assertRaisesRegex(ValueError,'differs from --prepare-warm-in-sha256'):
                S.read_warm_in(S.options(base+champion+['--prepare-warm-in',str(path),'--prepare-warm-in-sha256','0'*64]))
            empty=Path(tmp)/'empty.jsonl';empty.write_text('\n')   # an empty list (its own sha256 matches): refused
            with self.assertRaisesRegex(ValueError,'one JSON object per line, at least one'):
                S.read_warm_in(S.options(base+champion+['--prepare-warm-in',str(empty),'--prepare-warm-in-sha256',hashlib.sha256(b'\n').hexdigest()]))
            with self.assertRaises(FileNotFoundError):   # a missing list
                S.read_warm_in(S.options(base+champion+['--prepare-warm-in',str(Path(tmp)/'none.jsonl'),'--prepare-warm-in-sha256',sha]))
            # parsed from the bytes the sha256 checked: a file that changes after the check is never read again
            with patch.object(S,'read_requests',side_effect=AssertionError('the list file was read twice')):
                self.assertEqual(S.read_warm_in(listed),(dict(battle_id='a'),dict(battle_id='b')))
            path.write_text('[1]\n')
            with self.assertRaisesRegex(ValueError,'one JSON object per line'):S.read_requests(path)


if __name__=='__main__':unittest.main()
