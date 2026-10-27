"""Whole-state memo controls; every chance case and its mass remain present."""
from collections import namedtuple
from fractions import Fraction
import importlib.util
from types import SimpleNamespace
import unittest
from unittest import mock

from gpu_engine import fidelity_panel as panel, future_state as mapping
from tests.test_gpu_engine_fidelity_panel import FixedPlan, future


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'requires numpy')
class SuccessorCacheTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np=np
        self.Raw=namedtuple('Raw','hp keys queued_actions pending_residual residual_draws')
        self.calls=[]
        outer=self
        class Canon:
            metadata={'version':1}
            native_input=SimpleNamespace(certificate=('root',1))
            def canonical_successor(self,host,index):
                outer.calls.append(index)
                value=int(host.hp[index].flat[0])
                if value<0:return mapping.Refused((mapping.Gap('hp','UNSUPPORTED','raw retained'),))
                return future(value)
        self.plan=Canon()

    def host(self,values):
        n=len(values);np=self.np
        return self.Raw(np.asarray(values,dtype=np.int32).reshape(n,1),
            np.zeros((n,2),np.uint32),np.zeros((n,2),np.int32),
            np.zeros(n,bool),np.zeros((n,2,12),np.int32))

    def cache(self,size,limit=1024):
        return panel._PhysicalSuccessorCache(self.Raw._fields,size,limit=limit)

    def test_cross_batch_hits_keep_all_results_and_padding_out(self):
        cache=self.cache(3)
        self.assertEqual(len(cache.canonicalize(self.host([1,1,99]),2,self.plan)),2)
        self.assertEqual(len(self.calls),1)
        out=cache.canonicalize(self.host([1,2,99]),2,self.plan)
        self.assertEqual([x.key for x in out],[future(1).key,future(2).key])
        self.assertEqual(len(self.calls),2)
        self.assertEqual(cache.canonicalize(self.host([2,2,99]),2,self.plan),(future(2),)*2)
        self.assertEqual(len(self.calls),2)

    def test_every_state_field_including_future_capsule_and_keys_is_in_key(self):
        np=self.np;fields=tuple(mapping.SHAPES)
        self.assertEqual(set(fields),set(mapping.FIELD_POLICY))
        Raw=namedtuple('CompleteRaw',fields)
        data={name:np.zeros((2,)+tuple(shape),dtype=np.int32) for name,shape in mapping.SHAPES.items()}
        data['keys']=data['keys'].astype(np.uint32)
        cache=panel._PhysicalSuccessorCache(fields,2)
        cache.canonicalize(Raw(**data),1,self.plan)
        for name in fields:
            changed={k:v.copy() for k,v in data.items()};changed[name].flat[0]=1
            cache.canonicalize(Raw(**changed),1,self.plan)
        self.assertEqual(len(self.calls),1+len(fields))

    def test_dtype_and_shape_are_part_of_equal_byte_layouts(self):
        np=self.np;Raw=namedtuple('OneField','hp');cache=panel._PhysicalSuccessorCache(Raw._fields,2)
        a=Raw(np.asarray([[1,0],[1,0]],np.int8));b=Raw(np.asarray([[1],[1]],np.int16))
        self.assertEqual(a.hp[0].tobytes(),b.hp[0].tobytes())
        cache.canonicalize(a,1,self.plan);cache.canonicalize(b,1,self.plan)
        self.assertEqual(len(self.calls),2)

    def test_context_changes_recheck_existing_correspondence(self):
        cache=self.cache(1);host=self.host([1])
        cache.canonicalize(host,1,self.plan)
        self.plan.metadata['version']=2;cache.canonicalize(host,1,self.plan)
        self.plan.native_input.certificate=('other-root',2);cache.canonicalize(host,1,self.plan)
        self.assertEqual(len(self.calls),3)

    def test_eviction_recomputes_without_losing_cases(self):
        cache=self.cache(1,limit=2)
        out=[cache.canonicalize(self.host([v]),1,self.plan)[0] for v in (1,2,3,1)]
        self.assertEqual(out,[future(v) for v in (1,2,3,1)])
        self.assertEqual(len(self.calls),4);self.assertEqual(len(cache.values),2)

    def test_separate_executors_never_share_results(self):
        host=self.host([1]);one,two=self.cache(1),self.cache(1)
        one.canonicalize(host,1,self.plan);two.canonicalize(host,1,self.plan)
        self.assertEqual(len(self.calls),2)

    def test_cached_and_uncached_complete_laws_match_and_late_refusal_is_whole_slot(self):
        values=[1,1,2,1]
        class Four(FixedPlan):
            cardinality=4
            def bounded_cases(self):return iter(SimpleNamespace(mass=Fraction(1,4),value=v) for v in values)
        cache=self.cache(1)
        def run(use_cache):
            def execute(batch):
                host=self.host([batch[0].value])
                return cache.canonicalize(host,1,self.plan) if use_cache else (self.plan.canonical_successor(host,0),)
            return panel.compare_slot(Four(),execute,lambda:[(.75,future(1)),(.25,future(2))],
                batch_size=1,mass_tolerance=1e-5,probability_tolerance=1e-5)
        self.assertEqual(run(True),run(False))
        values[-1]=-1
        a,b=run(True),run(False)
        self.assertEqual(a,b);self.assertEqual(a['status'],'UNKNOWN')
        self.assertFalse(a['partial_laws_accepted']);self.assertEqual(a['completed_cases'],3)

    def test_missing_field_and_object_array_never_hit_cache(self):
        cache=self.cache(1);cache.canonicalize(self.host([1]),1,self.plan)
        with self.assertRaisesRegex(ValueError,'complete'):
            cache.canonicalize(SimpleNamespace(_asdict=lambda:{'hp':self.np.array([[1]])}),1,self.plan)
        host=self.host([1])._replace(hp=self.np.array([[1]],dtype=object))
        with self.assertRaisesRegex(ValueError,'non-object'):cache.canonicalize(host,1,self.plan)


class RepresentativeTests(unittest.TestCase):
    def test_first_actual_representative_is_serialized_once_per_key_and_engine(self):
        from dataclasses import asdict
        values=iter(('first','second'))
        def execute(batch):return (mapping.CanonicalFuture(future(1).key,next(values)),)
        native=lambda:[(.5,mapping.CanonicalFuture(future(1).key,'native-first')),
                       (.5,mapping.CanonicalFuture(future(1).key,'native-second'))]
        with mock.patch.object(panel,'asdict',wraps=asdict) as serializer:
            result=panel.compare_slot(FixedPlan(),execute,native,batch_size=1,
                mass_tolerance=1e-5,probability_tolerance=1e-5)
        self.assertEqual(serializer.call_count,2)
        self.assertEqual(result['status'],'AGREE');self.assertEqual(result['exact_cases'],2)
        reps=result['outcomes'][0]['raw_representatives']
        self.assertEqual(reps['jax']['certificate'],'first')
        self.assertEqual(reps['native']['certificate'],'native-first')
