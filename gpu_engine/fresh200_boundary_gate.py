"""Enforce Native freeze before expected boundary values are decoded.

The production sealer writes/validates the complete typed Native artifact and
returns its source-bound receipt. This gate never loads expected values until
that receipt is present for the exact stream/step/role. It is not a projector,
an expected-state surrogate or a scientific qualification result.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class BoundaryAddress:
    stream:int
    step:int
    role:str

    def __post_init__(self):
        if (type(self.stream) is not int or self.stream<=0 or type(self.step) is not int or self.step<0
                or self.role not in ('INITIALIZED','PRE','STEP','POST')):raise ValueError('exact physical boundary address required')


class BoundaryGate:
    def __init__(self,address:BoundaryAddress):
        if not isinstance(address,BoundaryAddress):raise ValueError('typed physical boundary address required')
        self.address=address;self.receipt=None

    def freeze(self,projection,sealer):
        if self.receipt is not None:raise ValueError('Native physical boundary is already frozen')
        receipt=sealer(self.address,projection)
        if (type(receipt) is not dict or set(receipt)!={'address','reference','field_count','source_registry_hash'}
                or receipt['address']!=self.address or receipt['field_count']!=73
                or type(receipt['source_registry_hash']) is not str or re.fullmatch('[a-f0-9]{64}',receipt['source_registry_hash']) is None):
            raise ValueError('complete source-bound Native projection receipt required')
        reference=receipt['reference']
        if (type(reference) is not dict or set(reference)!={'path','bytes','sha256'}
                or type(reference['path']) is not str or not reference['path'] or type(reference['bytes']) is not int or reference['bytes']<=0
                or type(reference['sha256']) is not str or re.fullmatch('[a-f0-9]{64}',reference['sha256']) is None):
            raise ValueError('sealed typed Native artifact identity required')
        self.receipt=dict(receipt,reference=dict(reference))
        return dict(self.receipt,reference=dict(reference))

    def read_expected(self,loader):
        if self.receipt is None:raise ValueError('expected physical values cannot be read before Native projection is sealed')
        return loader(self.address,dict(self.receipt,reference=dict(self.receipt['reference'])))
