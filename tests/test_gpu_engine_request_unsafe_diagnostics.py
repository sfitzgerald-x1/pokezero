"""Original static rules + literal trace packets only; no engine/model/game."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from gpu_engine import engine_gameplay_trace as trace
from gpu_engine import engine_request_source as request
from gpu_engine import benchmark_retained_games as benchmark
import test_gpu_engine_request_source as static_fixture
import test_gpu_engine_gameplay_stream_comparison as fake


class UnsafeHistoryDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        static_fixture.RequestSourceTests.setUpClass()
        cls.f=static_fixture.RequestSourceTests

    def test_original_static_namespace_and_lookup_are_bound_not_request_ids(self):
        snapshot=trace.request_history_rules(self.f.compiled)
        self.assertEqual(snapshot['engine_move_names'],self.f.metadata['vocabulary']['moves'])
        self.assertEqual(snapshot['engine_species_names'],self.f.metadata['vocabulary']['species'])
        np.testing.assert_array_equal(snapshot['rules_binding'],self.f.rules.device.binding)
        np.testing.assert_array_equal(snapshot['engine_binding'],self.f.rules.device.engine_binding)
        np.testing.assert_array_equal(snapshot['unsupported_history'],self.f.rules.device.unsupported_history)
        self.assertEqual(len(snapshot['request_binding']),8)
        flags=np.asarray(self.f.rules.device.unsupported_history)
        self.assertEqual(snapshot['unsupported_history_sha256'],hashlib.sha256(flags.tobytes()).hexdigest())
        self.assertTrue(snapshot['unsupported_history'][snapshot['engine_move_names'].index('hyperbeam')])
        self.assertFalse(snapshot['unsupported_history'][snapshot['engine_move_names'].index('surf')])
        with self.assertRaises(ValueError):trace.request_history_rules(request.Compiled(self.f.compiled.binding))
        record=request._checked(self.f.compiled);original=record['metadata'];changed=json.loads(original)
        changed['vocabulary']['moves'][1]='changed-diagnostic-namespace'
        try:
            record['metadata']=json.dumps(changed)
            with self.assertRaisesRegex(ValueError,'vocabulary binding'):trace.request_history_rules(self.f.compiled)
        finally:record['metadata']=original

    def test_b8_first_actual_per_owner_transition_and_literal_pre_post_operands(self):
        before,_=fake.result(batch=8);after,_=fake.result(batch=8,frame=1)
        # This control needs literal source fields, not large fake observations.
        for result in (before,after):
            for owner in result.owners:owner.encoded.observation.admitted=np.zeros(8,bool)
        old=(np.zeros(8,bool),np.zeros(8,bool));new=(old[0].copy(),old[1].copy())
        new[0][7]=True;new[1][3]=True
        before.state.core.unsafe=old;after.state.core.unsafe=new
        hyper=self.f.metadata['vocabulary']['moves'].index('hyperbeam')
        species=np.zeros((8,2,6),np.int32);species[:,:,0]=1
        for result in (before,after):result.state.core.engine=result.state.core.engine._replace(species=species.copy())
        engine=after.state.core.engine
        history=engine.last_move.copy();history[3,1,0]=hyper;history[7,0,0]=-1
        active=np.zeros((8,2),np.int32);active[3,1]=4
        charge=engine.charge_move.copy();charge[3,1,4]=hyper
        encore=engine.encore_move.copy();encore[3,1,4]=2
        counter=engine.encore_counter.copy();counter[3,1,4]=3
        after.state.core.engine=engine._replace(last_move=history,active=active,
            charge_move=charge,encore_move=encore,encore_counter=counter)
        literal=trace.state_packet(after.state.core.engine,3)
        with TemporaryDirectory() as directory:
            path=Path(directory)/'trace';writer=trace.TraceWriter(path,max_bytes=10**6)
            writer.begin(initial=before,runtime_binding='r',source={'commit':'literal'},pool_receipt={},
                team_indices=np.arange(16).reshape(8,2),stream_ids=np.arange(101,109),max_requests=2,
                constructor_seed=1,sampling_seed=2,action_names=('', 'surf'),request_compiled=self.f.compiled)
            for index,a,z in ((0,before,after),(1,after,after)):
                writer.decision(request=index,before=a,after=z,actions=np.zeros((8,2),np.int32),
                    active=np.ones(8,bool),executed=np.ones(8,bool),invalid=np.zeros(8,bool),chance=fake.chance(8))
            descriptor=writer.finish([dict(stream_id=i,status='UNKNOWN_REFUSED')for i in range(101,109)])
            self.assertTrue(descriptor['complete'],descriptor['failures']);rows=[json.loads(x)for x in path.read_text().splitlines()]
        header=rows[0]['payload']['request_history_rules']
        self.assertEqual(len(header['rules_binding']),8)
        decisions=[x['payload']for x in rows if x['kind']=='DECISION']
        first=[x for x in decisions if 'unsafe_history_first_transition'in x]
        self.assertEqual([(x['stream_id'],x['unsafe_history_first_transition']['owners'])for x in first],[(104,[1]),(108,[0])])
        self.assertEqual(first[0]['before']['unsafe'],[False,False]);self.assertEqual(first[0]['after']['unsafe'],[False,True])
        np.testing.assert_array_equal(first[0]['after']['state']['last_move'],literal['last_move'])
        for name in ('active','charge_move','charge_counter','encore_move','encore_counter'):
            self.assertEqual(first[0]['after']['state'][name],literal[name])
        lookup=first[0]['unsafe_history_first_transition']['after_lookup'][1]
        self.assertEqual((lookup['last_move'],lookup['last_move_valid'],lookup['unsupported_history']),(hyper,True,True))
        invalid=first[1]['unsafe_history_first_transition']['after_lookup'][0]
        self.assertEqual((invalid['last_move'],invalid['last_move_valid'],invalid['unsupported_history']),(-1,False,None))
        self.assertEqual(new[0].tolist(),[False]*7+[True]);self.assertEqual(new[1].tolist(),[False]*3+[True]+[False]*4)

    def test_missing_or_malformed_diagnostics_never_change_actual_carries(self):
        actual,_=fake.result();self.assertIsNone(trace.boundary(actual,0)['unsafe'])
        actual.state.core.unsafe=(np.array([1],np.int32),np.array([False]))
        with TemporaryDirectory()as directory:
            writer=trace.TraceWriter(Path(directory)/'trace')
            writer.begin(initial=actual,runtime_binding='r',source={},pool_receipt={},team_indices=[[0,1]],
                stream_ids=[101],max_requests=1,constructor_seed=1,sampling_seed=2,action_names=('', 'surf'))
            receipt=writer.finish([dict(stream_id=101,status='NOT_ATTEMPTED')])
        self.assertFalse(receipt['complete']);self.assertIn('literal bool',receipt['failures'][0]['message'])
        self.assertEqual(actual.state.core.unsafe[0].tolist(),[1]);self.assertTrue(bool(actual.state.core.admitted[0]))

    def test_benchmark_only_passes_original_request_to_opt_in_trace(self):
        import subprocess
        baseline=ast.parse(subprocess.check_output(['git','show','40204d52:gpu_engine/benchmark_retained_games.py'],text=True))
        current=ast.parse(Path(benchmark.__file__).read_text())
        before=deepcopy(current)
        calls=[n for n in ast.walk(before)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)
            and n.func.attr=='begin'and isinstance(n.func.value,ast.Name)and n.func.value.id=='gameplay_trace']
        self.assertEqual(len(calls),1)
        kw=next(k for k in calls[0].keywords if k.arg=='request_compiled')
        self.assertEqual(ast.unparse(kw.value),"prepared_record.get('request_compiled')")
        calls[0].keywords.remove(kw)
        self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(baseline,include_attributes=False))


if __name__=='__main__':unittest.main()
