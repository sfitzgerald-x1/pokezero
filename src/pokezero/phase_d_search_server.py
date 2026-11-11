"""Bounded shared-GPU request transport for concurrent CPU Foul Play pairs.

Import does not start a listener or load a model. The one explicit Search
instance owns the GPU and champion. CPU clients send owning prefixes and await
one action; they never load a device evaluator or allocate a GPU per game.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import math
import socket
import struct
import time
import uuid

SCHEMA = 'pokezero.phase-d.shared-gpu-search.v1'
MAX_PACKET = 16 * 1024 * 1024


def encode(value):
    blob=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if not 0<len(blob)<=MAX_PACKET:raise ValueError('bounded nonempty search packet required')
    return struct.pack('!I',len(blob))+blob


async def receive(reader):
    length=struct.unpack('!I',await reader.readexactly(4))[0]
    if not 0<length<=MAX_PACKET:raise ValueError('search packet exceeds bound')
    return json.loads(await reader.readexactly(length))


class Server:
    def __init__(self, search, *, max_requests=32, batch_wait_seconds=.005, queue_limit=256):
        from gpu_engine.phase_d_search import Search
        if type(search) is not Search:
            raise ValueError('one real champion GPU Search required')
        self._initialize(search,max_requests,batch_wait_seconds,queue_limit)

    def _initialize(self,search,max_requests,batch_wait_seconds,queue_limit):
        if (type(max_requests) is not int or not 1<=max_requests<=256 or type(queue_limit) is not int
                or queue_limit<max_requests or type(batch_wait_seconds) not in (int,float)
                or not math.isfinite(batch_wait_seconds) or not 0<=batch_wait_seconds<=.1):
            raise ValueError('bounded shared search queue/batch window required')
        self.search,self.max_requests,self.wait=search,max_requests,batch_wait_seconds
        self.queue=asyncio.Queue(queue_limit)
        # ONE executor thread and ONE queue consumer; no concurrent JAX owners.
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='phase-d-shared-gpu')
        self.clients=set();self.closing=False;self.started=False

    async def handle(self,reader,writer):
        task=asyncio.current_task();self.clients.add(task)
        try:
            while not self.closing:
                request=await receive(reader)
                if request.get('schema')!=SCHEMA or type(request.get('request_id')) is not str:
                    raise ValueError('explicit Phase D request identity required')
                future=asyncio.get_running_loop().create_future()
                arrival=time.monotonic()
                await self.queue.put((request,future,arrival))
                # Disconnect/cancel never causes a replay. The queued request
                # remains owned by the single consumer until it completes.
                result=await asyncio.shield(future)
                result=dict(result,server_wall_seconds=time.monotonic()-arrival)
                writer.write(encode(result));await writer.drain()
        except (asyncio.IncompleteReadError,ConnectionError):
            pass
        finally:
            self.clients.discard(task);writer.close();await writer.wait_closed()

    async def consume(self):
        if self.started:raise ValueError('one shared GPU queue consumer required')
        self.started=True
        while True:
            first=await self.queue.get()
            if first is None:
                self.queue.task_done();return
            batch=[first];end=time.monotonic()+self.wait;finish=False
            while len(batch)<self.max_requests:
                remaining=end-time.monotonic()
                if remaining<=0:break
                try:item=await asyncio.wait_for(self.queue.get(),remaining)
                except TimeoutError:break
                if item is None:
                    self.queue.task_done();finish=True;break
                batch.append(item)
            try:
                result=await asyncio.get_running_loop().run_in_executor(self.executor,self.search.process_many,[r for r,_,_ in batch])
                if len(result)!=len(batch) or any(r.get('request_id')!=item[0]['request_id'] for r,item in zip(result,batch)):
                    raise ValueError('shared GPU result order/address differs')
                for row,(_,future,arrival) in zip(result,batch):
                    if not future.done():future.set_result(row)
            except BaseException as error:
                for _,future,_ in batch:
                    if not future.done():future.set_exception(RuntimeError(type(error).__name__+': '+str(error)))
                if not isinstance(error,Exception):raise
            finally:
                for _ in batch:self.queue.task_done()
            if finish:return

    async def serve(self,host,port):
        """One explicit listener; caller owns its lifecycle and GPU allocation."""
        consumer=asyncio.create_task(self.consume())
        try:
            async with await asyncio.start_server(self.handle,host,port) as server:
                async with server:await server.serve_forever()
        finally:
            self.closing=True
            # Drain accepted work and retain unknown cells on external failure;
            # caller's scientific owner supplies the original stall/backstop.
            await self.queue.put(None)
            await consumer
            for task in tuple(self.clients):task.cancel()
            await asyncio.gather(*self.clients,return_exceptions=True)
            self.executor.shutdown(wait=True,cancel_futures=False)


class Client:
    """Synchronous client for the existing B2 actor's synchronous decide seam."""
    def __init__(self,host,port,*,timeout_seconds=60.):
        if type(timeout_seconds) not in (int,float) or not math.isfinite(timeout_seconds) or not 0<timeout_seconds<=60:
            raise ValueError('original learner bound at most60seconds required')
        self.address=(host,port);self.timeout=timeout_seconds

    def request(self,snapshot,*,decision_index):
        """Send only the owner's literal snapshot; no true opponent inputs."""
        request=dict(schema=SCHEMA,request_id=uuid.uuid4().hex,battle_id=snapshot.battle_id,owner=snapshot.owner,
            stream_id=snapshot.stream_id,decision_index=decision_index,generation=snapshot.generation,
            prefix_sha256=snapshot.prefix_sha256,lines=snapshot.lines,own_request_line=snapshot.own_request_line)
        started=time.monotonic()
        def exact(stream,n):
            result=bytearray()
            while len(result)<n:
                remaining=self.timeout-(time.monotonic()-started)
                if remaining<=0:raise TimeoutError('shared GPU search exceeded original learner limit')
                stream.settimeout(remaining);piece=stream.recv(n-len(result))
                if not piece:raise ConnectionError('shared GPU closed without a result; no retry')
                result.extend(piece)
            return bytes(result)
        with socket.create_connection(self.address,timeout=self.timeout) as stream:
            stream.sendall(encode(request))
            length=struct.unpack('!I',exact(stream,4))[0]
            if not 0<length<=MAX_PACKET:raise ValueError('shared GPU response exceeds bound')
            response=json.loads(exact(stream,length))
        if response.get('schema')!=SCHEMA or response.get('request_id')!=request['request_id']:
            raise ValueError('shared GPU response binding differs')
        if response.get('status')!='COMPLETE':
            raise ValueError('Phase D search refused; original cell unknown: '+str(response.get('refusal')))
        return response
