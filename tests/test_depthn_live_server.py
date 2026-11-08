"""The live server on CPU: fake (and one real fake-engine) search objects through Codex's real Server queue over localhost."""
import asyncio
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import unittest

from pokezero import phase_d_search_server as original
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
        self.warmed=True;self.saved=list(requests);return dict(buckets={1:dict(cold_seconds=1.5),2:dict(cold_seconds=2.5)},programs=7)

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
            thread=threading.Thread(target=S.run,kwargs=dict(search=search,host='127.0.0.1',port=port,warmup_requests=[dict(battle_id='w')],max_requests=4,
                telemetry=str(path),header=dict(tree_commit='abc'),log=logs.append,on_server=servers.append),daemon=True)
            thread.start();time.sleep(.3)
            self.assertEqual((logs,search.warmed,servers),([],False,[]))   # warming: no READY, no server, nothing listening
            with self.assertRaises(OSError):socket.create_connection(('127.0.0.1',port),timeout=1).close()
            release.set();deadline=time.monotonic()+10
            while not logs and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(len(logs),1);ready=logs[0]
            self.assertEqual((ready['event'],ready['tree_commit'],ready['port'],ready['programs'],ready['compile_seconds']),('READY','abc',port,7,{1:1.5,2:2.5}))
            self.assertTrue(search.warmed);self.assertEqual(search.saved,[dict(battle_id='w')])
            row=blocking_request(port,request(1,budget_seconds=2.5));self.assertTrue(row['warmed'])   # served only after warmup
            servers[0].stop();thread.join(10);self.assertFalse(thread.is_alive())
            rows=[json.loads(line)for line in path.read_text().splitlines()]
        self.assertEqual([r['kind']for r in rows],['ready','request']);self.assertEqual(rows[0]['event'],'READY')
        self.assertEqual((rows[1]['response']['request_id'],rows[1]['address']['battle_id'],rows[1]['lines'],rows[1]['budget_seconds']),('r1','b1',1,2.5))
        self.assertEqual(rows[1]['arrival_monotonic'],rows[1]['response']['arrival']);self.assertGreaterEqual(rows[1]['server_seconds'],0)

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
        with L.Codex(outcomes):rows=asyncio.run(scenario())
        self.assertEqual([(r['request_id'],r['status'],r['cause'])for r in rows],[('s:0','COMPLETE',None),('t:0','REFUSED','expired_in_queue')])
        self.assertTrue(0<=rows[0]['timings']['queue_seconds']<1);self.assertGreater(rows[0]['server_wall_seconds'],0)
        self.assertEqual(set(rows[0])-{'server_wall_seconds'},L.KEYS)

    def test_command_line(self):
        base=['--showdown-root','s','--node-binary','n','--metadata','m','--expected-metadata-sha256','a'*64,'--vocabulary','v',
            '--expected-vocabulary-sha256','b'*64,'--port','9000','--warmup-requests','w.jsonl']
        champion=['--champion-archive','c.npz','--champion-archive-sha256','c'*64,'--champion-checkpoint-sha256','d'*64]
        args=S.options(base+champion)
        self.assertEqual((args.rung,args.buckets,args.max_requests,args.champion_precision,args.min_price_seconds,args.session_cap,args.host),
            ('DN2W8',(1,2,4),4,'bf16',1.,256,'127.0.0.1'))
        args=S.options(base+champion+['--rung','DN2W16','--buckets','2,1','--max-requests','1'])
        self.assertEqual((args.rung,args.buckets,args.max_requests),('DN2W16',(1,2),1))
        for bad in(base,base+champion[:2],base+champion+['--max-requests','5'],base+champion+['--buckets','0,1'],base+champion+['--rung','R1'],
                base+champion[:2]+['--champion-archive-sha256','x'*64,'--champion-checkpoint-sha256','d'*64],base[:-2]+champion):
            with self.subTest(bad=bad[len(base)-2:]),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):S.options(bad)
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'w.jsonl';path.write_text('{"battle_id":"a"}\n\n{"battle_id":"b"}\n')
            self.assertEqual(S.read_requests(path),[dict(battle_id='a'),dict(battle_id='b')])
            path.write_text('[1]\n')
            with self.assertRaisesRegex(ValueError,'one JSON object per line'):S.read_requests(path)


if __name__=='__main__':unittest.main()
