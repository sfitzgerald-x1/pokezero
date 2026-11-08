"""Single-owner ladder server and explicit refused-to-raw client telemetry."""
import json
import math
import socket
import struct
import time
import uuid
from . import phase_d_search_server as original

SCHEMA=original.SCHEMA


class Server(original.Server):
    def __init__(self,search,*,max_requests=32,batch_wait_seconds=.005,queue_limit=256):
        from gpu_engine.phase_d_ladder_search import Search
        from gpu_engine.phase_d_depthn import Search as DepthNSearch
        if type(search)not in (Search,DepthNSearch):raise ValueError('one genuine champion ladder GPU owner required')
        self._initialize(search,max_requests,batch_wait_seconds,queue_limit)


class Client:
    def __init__(self,host,port,*,rung,timeout_seconds=45.):
        if rung not in ('R1','R3','R4','R7','R5')or type(timeout_seconds)not in (int,float)or not math.isfinite(timeout_seconds)or not 0<timeout_seconds<60:
            raise ValueError('fixed rung and bounded search time leaving raw fallback time required')
        self.address=(host,port);self.rung=rung;self.timeout=timeout_seconds

    def request(self,snapshot,*,decision_index):
        request=dict(schema=SCHEMA,request_id=uuid.uuid4().hex,battle_id=snapshot.battle_id,owner=snapshot.owner,
            stream_id=snapshot.stream_id,decision_index=decision_index,generation=snapshot.generation,
            prefix_sha256=snapshot.prefix_sha256,lines=snapshot.lines,own_request_line=snapshot.own_request_line)
        started=time.monotonic()
        def exact(stream,n):
            output=bytearray()
            while len(output)<n:
                remaining=self.timeout-(time.monotonic()-started)
                if remaining<=0:raise TimeoutError('fixed ladder search timeout; no retry')
                stream.settimeout(remaining);piece=stream.recv(n-len(output))
                if not piece:raise ConnectionError('search server closed; no retry')
                output.extend(piece)
            return bytes(output)
        try:
            with socket.create_connection(self.address,timeout=self.timeout)as stream:
                stream.sendall(original.encode(request));length=struct.unpack('!I',exact(stream,4))[0]
                if not 0<length<=original.MAX_PACKET:raise ValueError('bounded search response required')
                response=json.loads(exact(stream,length))
            if response.get('schema')!=SCHEMA or response.get('request_id')!=request['request_id']or response.get('rung')!=self.rung:
                raise ValueError('same declared rung/request response required')
            if response.get('status')not in ('COMPLETE','REFUSED'):raise ValueError('literal search status required')
        except (ValueError,TypeError,KeyError,TimeoutError,ConnectionError,OSError)as error:
            response=dict(status='REFUSED',refusal=type(error).__name__+': '+str(error),requested_worlds=None,constructed_worlds=None,
                decision_seconds=None,rung=self.rung)
        return dict(response,client_wall_seconds=time.monotonic()-started)
