"""P1 host packing preserves every byte across CPU buffer forms and padding."""
from types import SimpleNamespace as S
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import device_depth2_pricer as d2
from tests.fixtures import depth2_host_reference as old


def inputs(kind,*,count=3,missing=()):
    cpu=jax.devices('cpu')[0];ordinal=0
    def convert(x):
        nonlocal ordinal
        ordinal+=1
        if kind=='numpy' or(kind=='mixed' and ordinal%2):return x
        if kind=='uncommitted':return j.asarray(x)
        device=cpu if kind in ('cpu','mixed') else jax.devices('gpu')[0]
        return jax.device_put(x,device)
    def tree(index):
        return dict(payload=convert(np.array([0x80000000,0x7fc12345,0x3f800000+index],np.uint32).view(np.float32)),
            small=convert(np.array([index,-index],np.int8)),unsigned=convert(np.array([index,255],np.uint8)),
            flags=convert(np.array([True,False])),scalar=convert(np.float32(-0.)),
            bf16=convert(np.array([1.,-0.,float('nan')],dtype=j.bfloat16)),wide=convert(np.array([index,2**32-1],np.uint32)))
    requests=[];worlds=[]
    for i in range(count):
        requests.append(S(root_inputs=tree(i),root_logits=(0.,)*9,root_legal=(True,False,True)+(False,)*6,
            plans=S(root=S(address=S(owner=i%2))),learner_actor=i%3,opponent_actor=(i+1)%3))
        pair=[]
        for index in (2*i,2*i+1):
            pair.append(S(state=None,public=None) if index in missing else S(state=tree(index),public=tree(index+1)))
        worlds.append(tuple(pair))
    recipe=S(pricing_order=lambda address,logits,legal:tuple(i for i,live in enumerate(legal) if live))
    return requests,worlds,recipe,({},{},{})


def packed(fn,data,capacity,fixed):
    requests,worlds,recipe,params=data
    return fn(requests,worlds,recipe,capacity,params,k1_own=2,fixed_capacity=fixed)


def equal_bytes(case,a,b):
    la,ta=jax.tree.flatten(a);lb,tb=jax.tree.flatten(b);case.assertEqual(ta,tb)
    for x,y in zip(la,lb):
        case.assertEqual(x.shape,y.shape);case.assertEqual(x.dtype,y.dtype);case.assertEqual(x.tobytes(),y.tobytes())


class HostPackTests(unittest.TestCase):
    def test_cpu_buffer_forms_and_padding_match_frozen_oracle(self):
        for kind in ('numpy','cpu','uncommitted','mixed'):
            for count,capacity,missing in ((1,1,()),(3,3,(0,)),(3,5,(1,4))):
                for fixed in (False,True):
                    with self.subTest(kind=kind,count=count,capacity=capacity,missing=missing,fixed=fixed):
                        data=inputs(kind,count=count,missing=missing)
                        equal_bytes(self,packed(old.pack_turn1,data,capacity,fixed),packed(d2.pack_turn1,data,capacity,fixed))

    def test_no_available_worlds(self):
        for count in (0,2):
            data=inputs('numpy',count=count,missing=tuple(range(2*count)))
            self.assertIsNone(packed(old.pack_turn1,data,4,True));self.assertIsNone(packed(d2.pack_turn1,data,4,True))

    def test_numpy_float_payload_bits_survive(self):
        result=packed(d2.pack_turn1,inputs('numpy',count=1),2,True)
        np.testing.assert_array_equal(result['states']['payload'][0,0].view(np.uint32),[0x80000000,0x7fc12345,0x3f800000])
        np.testing.assert_array_equal(result['states']['payload'][1,0].view(np.uint32),result['states']['payload'][0,0].view(np.uint32))

    def test_cpu_arrays_under_transfer_guard(self):
        data=inputs('cpu',count=1)
        with jax.transfer_guard_device_to_host('disallow'):
            expected=packed(old.pack_turn1,data,2,True);actual=packed(d2.pack_turn1,data,2,True)
        equal_bytes(self,expected,actual)


if __name__=='__main__':unittest.main()
