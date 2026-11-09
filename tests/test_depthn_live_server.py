"""The live server on CPU: fake (and one real fake-engine) search objects through Codex's real Server queue over localhost."""
import asyncio
from contextlib import ExitStack,redirect_stderr
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


class ServerTests(unittest.TestCase):
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
                    (rp,'ReplyPrior',lambda executor,model,mode:('reply-prior',executor,model.params,mode))):
                stack.enter_context(patch.object(target,name,side_effect=value))
            self.assertEqual(S.build(args),'search')
        self.assertEqual(loaded,[('r.npz',dict(expected_archive_sha256='e'*64,expected_checkpoint_sha256='f'*64,precision=S.evaluators.PRECISIONS['bf16']))])
        self.assertEqual(made['reply'][0],'reply-prior');self.assertEqual((made['reply'][2],made['reply'][3]),('reply','eager'))
        with patch.dict(os.environ,PINNED):args=S.options(base+reply+['--peer-view-b'])   # both: br_fp on view b's worlds
        with ExitStack()as stack:
            for target,name,value in stand_ins+((S.evaluators,'load',lambda path,**kw:NS(params='reply')),
                    (rp,'ReplyPrior',lambda executor,model,mode:('reply-prior',executor,model.params,mode))):
                stack.enter_context(patch.object(target,name,side_effect=value))
            S.build(args)
        self.assertEqual((made['peer_view_b'],made['reply'][0]),(True,'reply-prior'))

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
            path.write_text('[1]\n')
            with self.assertRaisesRegex(ValueError,'one JSON object per line'):S.read_requests(path)


if __name__=='__main__':unittest.main()
