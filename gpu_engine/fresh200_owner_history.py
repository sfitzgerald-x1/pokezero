"""Literal owner-wire chronology and current-reference binding, no encoder.

    Reference sessions ingest every emitted cut, including waits, into both
    existing backends. Only actual actionable requests may own an encoding.
    Extra public lines after a request make an older encoded prefix stale.
"""
from dataclasses import dataclass
import hashlib
import json


def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate literal owning request key')
        result[key]=value
    return result


@dataclass(frozen=True)
class OwnerCut:
    generation:int
    prefix_sha256:str
    request_line:str
    actionable:bool
    lines:tuple


class OwnerHistory:
    def __init__(self,owner):
        if owner not in ('p1','p2'):raise ValueError('literal owner seat required')
        self.owner=owner;self.lines=[];self.hash=hashlib.sha256();self.generation=0;self.current=None;self.reference=None

    @property
    def prefix_sha256(self):return self.hash.hexdigest()

    def append(self,lines):
        if type(lines) not in (list,tuple):raise ValueError('literal ordered Native owner lines required')
        cuts=[]
        for line in lines:
            if type(line) is not str or not line.startswith('|') or '\n' in line or '\r' in line:
                raise ValueError('literal single Native owner line required')
            request=line.startswith('|request|');packet=None
            if request:
                packet=json.loads(line[len('|request|'):],object_pairs_hook=_pairs)
                if packet is not None and (type(packet) is not dict or type(packet.get('side')) is not dict
                        or packet['side'].get('id')!=self.owner):raise ValueError('request is not the literal owning seat')
                if packet is not None:
                    if 'wait' in packet and type(packet['wait']) is not bool:raise ValueError('literal Native wait bit required')
                    force=packet.get('forceSwitch');active=packet.get('active')
                    if force is not None and (type(force) is not list or len(force)>1 or any(type(x) is not bool for x in force)):
                        raise ValueError('literal Native singles forced request required')
                    if active is not None and (type(active) is not list or len(active)>1 or any(type(x) is not dict for x in active)):
                        raise ValueError('literal Native singles active request required')
            self.lines.append(line);self.hash.update((line+'\n').encode())
            if request:
                self.generation+=1;self.reference=None
                actionable=packet is not None and not packet.get('wait',False) and (bool(packet.get('active')) or any(packet.get('forceSwitch') or []))
                self.current=OwnerCut(self.generation,self.prefix_sha256,line,bool(actionable),tuple(self.lines))
                cuts.append(self.current)
        return cuts

    def current_cut(self):
        if self.current is None or not self.current.actionable:raise ValueError('no current actual actionable owner request')
        return OwnerCut(self.current.generation,self.prefix_sha256,self.current.request_line,True,tuple(self.lines))

    def bind_reference(self,cut:OwnerCut,value):
        current=self.current_cut()
        if cut!=current:raise ValueError('reference does not belong to the current literal owner prefix')
        self.reference=(cut,value)

    def endpoint(self):
        current=self.current_cut()
        if self.reference is None or self.reference[0]!=current:raise ValueError('stale or missing current owner reference')
        return self.reference[1]
