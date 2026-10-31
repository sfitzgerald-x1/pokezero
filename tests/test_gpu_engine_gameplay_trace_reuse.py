"""Immutable measurement-data reuse only; no engine, model or game calls."""
from collections import namedtuple
from copy import deepcopy
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import numpy as np
from gpu_engine import engine_gameplay_trace as trace
from gpu_engine import engine_retained_encoder_driver as driver
from gpu_engine.state import BattleState
import test_gpu_engine_gameplay_stream_comparison as fixture

_RECORDS = {}

def immutable(value):
    if isinstance(value, np.ndarray):return jax.device_put(value)
    if isinstance(value, SimpleNamespace):
        keys = tuple(vars(value))
        cls = _RECORDS.setdefault(keys, namedtuple('LiteralBoundaryRecord', keys))
        return cls(*(immutable(getattr(value, key)) for key in keys))
    if type(value) is tuple:return tuple(map(immutable, value))
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return type(value)(*(immutable(x) for x in value))
    if isinstance(value, np.generic):return value.item()
    return value


def result(batch=2, frame=0, done=False):
    value, _ = fixture.result(batch, frame=frame, hp=100-frame, done=done)
    core = driver.Core(**{name:immutable(getattr(value.state.core, name, None)) for name in driver.Core._fields})
    owners = tuple(driver.OwnerResult(**{name:immutable(getattr(owner, name, None)) for name in driver.OwnerResult._fields}) for owner in value.owners)
    return driver.Result(driver.DriverState(core, ()), owners)


def header(batch=2):
    return dict(runtime_binding='r'*64, source={'commit':'unit-source'},pool_receipt={'pool_sha256':'a'*64},
        team_indices=np.arange(2*batch).reshape(batch,2),stream_ids=np.arange(101,101+batch),
        max_requests=3,constructor_seed=401,sampling_seed=402,action_names=('', 'tackle', 'struggle'))


def trace_run(path, reuse, *, mutable=False, cap=10**6):
    def factory(batch, **options):
        value=result(batch,**options)
        if mutable:
            owner=value.owners[0];observation=owner.encoded.observation
            writable=SimpleNamespace(**{name:np.array(leaf,copy=True)if isinstance(leaf,jax.Array)else leaf
                for name,leaf in zip(observation._fields,observation,strict=True)})
            value=value._replace(owners=(owner._replace(encoded=SimpleNamespace(observation=writable)),value.owners[1]))
        return value,None
    initial,_=factory(2);a,_=factory(2,frame=1);b,_=factory(2,frame=2,done=True)
    ticks=itertools.count(0,1000000)
    with patch.object(trace,'time',SimpleNamespace(perf_counter_ns=lambda:next(ticks))):
        writer=trace.TraceWriter(path,max_bytes=cap,reuse_boundary_data=reuse)
        writer.begin(initial=initial,**header())
        for request,(before,after)in enumerate(((initial,a),(a,b))):
            if mutable and request==1:
                a.owners[0].encoded.observation.numeric_features[0,0,0]=37
            writer.decision(request=request,before=before,after=after,actions=np.zeros((2,2),np.int32),
                active=np.ones(2,bool),executed=np.ones(2,bool),invalid=np.zeros(2,bool),chance=fixture.chance(2))
        summary=writer.finish([dict(stream_id=101+i,team_indices=[2*i,2*i+1],status='COMPLETED',executed_steps=2)for i in range(2)])
    return Path(path).read_bytes(),summary,writer


class TraceReuseTests(unittest.TestCase):
    def writer(self, path, value, *, batch=2):
        writer=trace.TraceWriter(path);writer.begin(initial=value,**header(batch));return writer

    def test_byte_identical_full_trace_summary_and_fewer_boundary_reads(self):
        with TemporaryDirectory()as directory:
            path=Path(directory)/'trace'
            with patch.object(trace,'boundary',wraps=trace.boundary)as calls:
                fresh,summary,_=trace_run(path,False);fresh_count=calls.call_count
            path.rename(Path(directory)/'fresh-preserved')
            with patch.object(trace,'boundary',wraps=trace.boundary)as calls:
                reused,actual,writer=trace_run(path,True);reuse_count=calls.call_count
            self.assertEqual(fresh,reused)
            self.assertEqual(trace.encoded(summary),trace.encoded(actual))
            self.assertEqual((fresh_count,reuse_count),(10,6))
            records=[json.loads(line)for line in reused.splitlines()]
            self.assertTrue(all(len(r['payload']['after']['full'])==2 for r in records if r['kind']=='DECISION'))
            self.assertEqual(writer._boundary_packets,{})
            self.assertIsNone(writer._boundary_result)

    def test_mutable_numpy_fixture_mutation_keeps_exact_fresh_trace(self):
        with TemporaryDirectory()as directory:
            path=Path(directory)/'trace'
            fresh,summary,_=trace_run(path,False,mutable=True);path.rename(Path(directory)/'fresh')
            with patch.object(trace,'boundary',wraps=trace.boundary)as calls:
                reused,actual,_=trace_run(path,True,mutable=True)
            self.assertEqual(fresh,reused);self.assertEqual(trace.encoded(summary),trace.encoded(actual))
            self.assertEqual(calls.call_count,10)
            decisions=[json.loads(line)['payload']for line in reused.splitlines()if json.loads(line)['kind']=='DECISION']
            changed=next(row for row in decisions if row['request']==1 and row['stream_id']==101)
            self.assertEqual(changed['before']['observations'][0]['arrays']['numeric_features'][0][0],37)

    def test_nested_mutable_records_numpy_and_unknown_leaves_never_cached(self):
        good=result();self.assertIsNotNone(trace._boundary_proof(good))
        owner=good.owners[0]
        variants=[good._replace(owners=(owner._replace(encoded=SimpleNamespace(observation=owner.encoded.observation)),good.owners[1])),
            good._replace(state=good.state._replace(core=good.state.core._replace(engine=good.state.core.engine._replace(hp=np.asarray(good.state.core.engine.hp).copy())))),
            good._replace(state=good.state._replace(core=good.state.core._replace(unsafe=[None,None])))]
        for value in variants:self.assertIsNone(trace._boundary_proof(value))
        # Unserialized constructor holders are not a claim of Core immutability.
        unused=good._replace(state=good.state._replace(core=good.state.core._replace(constructors=({'mutable':'not captured'},))))
        self.assertIsNotNone(trace._boundary_proof(unused))

    def test_changed_result_row_runtime_and_mutated_return_packet_fall_back(self):
        with TemporaryDirectory()as directory:
            initial=result();writer=self.writer(Path(directory)/'trace',initial)
            self.assertEqual(set(writer._boundary_packets),{0,1})
            packet=writer._before_boundary(initial,0);packet['frame']=999
            self.assertEqual(writer._before_boundary(initial,0)['frame'],0)
            changed=initial._replace(state=initial.state._replace(core=initial.state.core._replace(frame_number=jax.device_put(np.array([7,7],np.int32)))))
            with patch.object(trace,'boundary',wraps=trace.boundary)as fresh:
                self.assertEqual(writer._before_boundary(changed,0)['frame'],7)
                self.assertEqual(fresh.call_count,1)
            del writer._boundary_packets[1]
            with patch.object(trace,'boundary',wraps=trace.boundary)as fresh:
                writer._before_boundary(initial,1);self.assertEqual(fresh.call_count,1)
            writer.header['runtime_binding']='s'*64
            with patch.object(trace,'boundary',wraps=trace.boundary)as fresh:
                writer._before_boundary(initial,0);self.assertEqual(fresh.call_count,1)
            writer.finish([])

    def test_deleted_array_and_changed_namedtuple_descriptor_fall_back(self):
        with TemporaryDirectory()as directory:
            initial=result();writer=self.writer(Path(directory)/'trace',initial)
            cursor=initial.state.core.cursor;cls=type(cursor);original=cls.stream_id
            try:
                cls.stream_id=cls.turn
                with patch.object(trace,'boundary',return_value={'fresh':True})as fresh:
                    self.assertEqual(writer._before_boundary(initial,0),{'fresh':True});fresh.assert_called_once()
            finally:cls.stream_id=original
            initial.state.core.engine.hp.delete()
            with patch.object(trace,'boundary',return_value={'deleted_fresh_attempt':True})as fresh:
                self.assertEqual(writer._before_boundary(initial,0),{'deleted_fresh_attempt':True});fresh.assert_called_once()
            writer.finish([])

    def test_mutable_python_jax_subclass_and_missing_native_type_are_fresh(self):
        class MutableArray(jax.Array):
            def __init__(self, value):self.value=np.array(value,copy=True)
            @property
            def shape(self):return self.value.shape
            @property
            def dtype(self):return self.value.dtype
            def is_deleted(self):return False
            def __getitem__(self,index):return self.value[index]
        original=result();self.assertIs(type(original.state.core.engine.hp),trace._CANONICAL_JAX_ARRAY_IMPL)
        # Matching name/module text is not an implementation identity.
        MutableArray.__name__=type(original.state.core.engine.hp).__name__
        MutableArray.__module__=type(original.state.core.engine.hp).__module__
        fake=MutableArray(np.full((2,2,6),100,np.int32))
        value=original._replace(state=original.state._replace(core=original.state.core._replace(
            engine=original.state.core.engine._replace(hp=fake))))
        self.assertIsInstance(fake,jax.Array)
        self.assertIsNone(trace._boundary_proof(value))
        with TemporaryDirectory()as directory:
            writer=self.writer(Path(directory)/'trace',value)
            self.assertEqual(writer._boundary_packets,{})
            fake.value[0,0,0]=7
            with patch.object(trace,'boundary',wraps=trace.boundary)as fresh:
                self.assertEqual(writer._before_boundary(value,0)['state']['hp'][0][0][0],7)
                fresh.assert_called_once()
            writer.finish([])
            path=Path(directory)/'unavailable'
            fresh,summary,_=trace_run(path,False);path.rename(Path(directory)/'fresh-preserved')
            with patch.object(trace,'_CANONICAL_JAX_ARRAY_IMPL',None),patch.object(trace,'boundary',wraps=trace.boundary)as calls:
                actual,result_summary,_=trace_run(path,True)
            self.assertEqual(actual,fresh);self.assertEqual(trace.encoded(result_summary),trace.encoded(summary))
            self.assertEqual(calls.call_count,10)

    def test_only_latest_result_active_rows_retained_and_capacity_same(self):
        with TemporaryDirectory()as directory:
            initial=result();after=result(frame=1);writer=self.writer(Path(directory)/'trace',initial)
            writer.decision(request=0,before=initial,after=after,actions=np.zeros((2,2),np.int32),
                active=np.array([True,False]),executed=np.array([True,False]),invalid=np.zeros(2,bool),chance=fixture.chance(2))
            self.assertIs(writer._boundary_result,after);self.assertEqual(set(writer._boundary_packets),{0})
            writer.finish([]);self.assertIsNone(writer._boundary_result)
            golden,_,_=trace_run(Path(directory)/'size-baseline',False)
            lines=golden.splitlines(keepends=True)
            # Admit HEADER/INITIALs/one DECISION, then cap with a populated
            # immutable cache. Later terminal evidence/attempts still survive.
            cap=sum(map(len,lines[:4]))+len(lines[4])//2
            path=Path(directory)/'capacity'
            fresh,summary,_=trace_run(path,False,cap=cap);path.rename(Path(directory)/'fresh')
            reused,actual,writer=trace_run(path,True,cap=cap)
            self.assertEqual(fresh,reused);self.assertEqual(trace.encoded(summary),trace.encoded(actual))
            self.assertFalse(actual['complete']);self.assertEqual(writer._boundary_packets,{})
            self.assertEqual(actual['attempts'],{101:2,102:2})
            self.assertEqual(actual['written'],{101:1,102:0})
            self.assertEqual(len(actual['terminals']),2)

if __name__=='__main__':unittest.main()
