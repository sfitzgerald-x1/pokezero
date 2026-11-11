"""Typed stored transport + fake numerical worker. No actual engine/model."""
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import jax.numpy as j
import numpy as np

from gpu_engine import device_recorded_reencoding as r, device_observation_collector as c
from gpu_engine.device_observation_parity import AgreementParity, SHAPES
from gpu_engine.state import empty_state

LEGACY_SOURCE_COMMIT='9d5b6f72d1db13027c6716957c5c571689611e41'


def view(value=0):
    out={key:np.zeros(shape,np.float32 if kind=='f'else bool if kind=='b'else np.int32)
         for key,(shape,kind)in SHAPES.items()}
    out['numeric_features'][0,0]=value
    return out


def state(root=0,step=0):
    s=empty_state(1)
    return s._replace(done=j.array([step>=2]),turn=j.array([step],j.int32),
        hp=j.full((1,2,6),100+root-step,j.int32),
        keys=j.array([[2**31+1,2**32-1]],j.uint32))


def packet(s):
    return dict(state={key:np.asarray(getattr(s,key)).tolist()for key in s._fields},
        requested=[not bool(s.done[0])]*2)


def chance():
    from gpu_engine.transition import ChanceInputs
    return {key:np.zeros((1,2,12)if key=='draws'else(1,)if key=='initial_tie'else(1,2,4)if key=='effect'else(1,2),np.int32).tolist()
        for key in ChanceInputs._fields}


class Runtime:
    def __init__(self,*,bad_after=False,bad_initial=False):
        self.bindings={'test':'new-source'};self.calls=[];self.bad_after=bad_after;self.bad_initial=bad_initial
    def boundary(self,s):return packet(s)
    def initialize(self,s):
        if self.bad_initial:s=s._replace(gender=np.full((1,2,6),2,np.int32))
        return NS(state=s,trackers=0)
    def observe(self,s,t):
        self.calls.append(('observe',int(s.turn[0]),t))
        v={k:x[None]for k,x in view().items()};v['error']=np.zeros(1,np.int32)
        return None,np.array([[True,True]]),None,(NS(**v),NS(**v))
    def advance(self,s,a,t,ch):
        self.calls.append(('advance',int(s.hp[0,0,0]),ch,a.copy()))
        after=s._replace(turn=s.turn+1,hp=s.hp-1,done=np.array([int(s.turn[0])+1>=2]))
        if self.bad_after and int(s.hp[0,0,0])==100 and int(s.turn[0])==0:
            after=after._replace(status_counter=np.ones((1,2,6),np.int32))
        return NS(step=NS(state=after,invalid_actions=np.array([False])),trackers=t+1)


def dataset(folder,*,reference_value=0,routing_mismatch=False):
    root=Path(folder)/'old';writer=c.Evidence(root)
    cells=[dict(stream_id=10+n,root_id=20+n,native_seed=30+n,team_indices=[2*n,2*n+1])for n in range(2)]
    refdoc=dict(path='/not-loaded',sha256='a'*64)
    plan=dict(schema=c.SCHEMA,source_commit=LEGACY_SOURCE_COMMIT,pool=refdoc,vocabulary=refdoc,labels=refdoc,gender_policy=refdoc,
        showdown_root='/not-loaded',node='/not-loaded',node_sha256='b'*64,constructor_seed=1,sampling_seed=2,
        gender_seed=3,gender_domain=4,max_requests=2,max_turns=100,cells=cells,
        policy='PCG64_SEEDSEQUENCE_UNIFORM_PHYSICAL_LEGAL_V1')
    oldsource=subprocess.check_output(['git','show',LEGACY_SOURCE_COMMIT+':gpu_engine/device_observation_parity.py'],cwd=c.ROOT)
    (root/'original-reducer.py').write_bytes(oldsource)
    bindings={'test':'original','source_commit':plan['source_commit'],
        'source_files':{'gpu_engine/device_observation_parity.py':hashlib.sha256(oldsource).hexdigest()}}
    old=r._frozen_reducer(bindings)(source_bindings=bindings);final=[]
    flags=[[True,None],[False,True]]
    for n,cell in enumerate(cells):
        writer.event(dict(kind='START',**cell,status='ATTEMPTED'))
        writer.boundary(dict(kind='CONSTRUCTOR',stream_id=cell['stream_id'],raw=packet(state(n)),initialized=packet(state(n))))
        for step in range(2):
            record=dict(kind='STEP',stream_id=cell['stream_id'],request=step,
                before=packet(state(n,step)),after=packet(state(n,step+1)),
                pre_agreement=flags[n][step],actions=[[0,1]],chance=chance())
            sha=writer.boundary(record)
            for side in (0,1):
                row=dict(decision_id=f"{cell['stream_id']}-{step}-{side}",stream_id=cell['stream_id'],
                    root_id=cell['root_id'],request=step,owner_side=side,turn=step,agreement=flags[n][step],
                    line_kinds=['move'],refusal=None,physical_binding_sha256=sha,candidate_error=0,reference_metadata={'literal':'unchanged'})
                original_view=view(reference_value)
                if routing_mismatch:original_view['switch_targets']=np.arange(1,6,dtype=np.int32)
                writer.decision(row,original_view,original_view)
                old.add(decision_id=row['decision_id'],turn=step,line_kinds=['move'],agreement=row['agreement'],
                    reference=original_view,candidate=original_view)
        value=dict(cell,status='TERMINAL',executed_steps=2);final.append(value);writer.event(dict(kind='END',**value))
    writer.close()
    (root/'plan.json').write_text(c._json(plan)+'\n');(root/'source-bindings.json').write_text(c._json(bindings)+'\n')
    summary=old.report();summary.update(declared_cells=2,cells=final,collection_schema=c.SCHEMA,
        artifact_files={name:dict(sha256=c.digest(root/name),bytes=(root/name).stat().st_size)
            for name in ('plan.json','source-bindings.json','physical.jsonl','decisions.jsonl','journal.jsonl')})
    (root/'summary.json').write_text(c._json(summary)+'\n')
    return root,plan,summary,bindings


class DeviceRecordedReencodingTests(unittest.TestCase):
    def run_stored(self,folder,*,runtime=None,reference_value=0):
        root,plan,summary,bindings=dataset(folder,reference_value=reference_value)
        writer=c.Evidence(Path(folder)/'new');runtime=Runtime()if runtime is None else runtime
        try:result=r.reencode_records(runtime,root,plan,summary,bindings,root/'physical.jsonl',root/'decisions.jsonl',writer)
        finally:writer.close()
        return result,runtime,root,Path(folder)/'new'

    def test_same_rows_flags_refs_continuous_public_and_no_reference_leak(self):
        with tempfile.TemporaryDirectory()as folder:
            result,runtime,old,new=self.run_stored(folder,reference_value=7)
            self.assertEqual(result['counts']['attempted'],8);self.assertEqual(result['counts']['agreement'],4)
            self.assertEqual(result['counts']['compared'],4);self.assertEqual(result['counts']['joint_match'],0)
            self.assertEqual(result['counts']['state_agreement_unknown'],2);self.assertEqual(result['counts']['state_disagreement'],2)
            self.assertTrue(result['original_report_recomputed_exact']);self.assertEqual(result['new_games'],0)
            self.assertEqual([x[2]for x in runtime.calls if x[0]=='observe'],[0,1,0,1])
            oldrows=[x for x,_ in r._records(old/'decisions.jsonl')];newrows=[x for x,_ in r._records(new/'decisions.jsonl')]
            self.assertEqual([x['original_metadata']for x in newrows],oldrows)
            for row in newrows:
                with np.load(new/row['tensor_path'],allow_pickle=False)as arrays:
                    self.assertEqual(arrays['reference__numeric_features'][0,0],7)
                    self.assertEqual(arrays['candidate__numeric_features'][0,0],0)
                self.assertEqual(row['original_tensor_sha256'],c.digest(row['original_tensor_path']))

    def test_numeric_mismatch_stops_root_refuses_but_keeps_original_agreement(self):
        with tempfile.TemporaryDirectory()as folder:
            result,runtime,old,new=self.run_stored(folder,runtime=Runtime(bad_after=True))
            self.assertEqual(result['counts']['agreement'],4);self.assertEqual(result['counts'].get('comparison_refused',0),0)
            self.assertEqual(result['counts']['compared'],4)
            self.assertEqual(result['reencoding_cells'][0]['reencoding_stopped']['differing_state_fields'],['status_counter'])
            self.assertIsNone(result['reencoding_cells'][1]['reencoding_stopped'])
            self.assertEqual(len([x for x in runtime.calls if x[0]=='advance']),3)
            self.assertFalse(result['replay_integrity_requirement_met'])
            self.assertEqual(result['replay_integrity_failed_roots'],1)
            self.assertEqual(result['provenance_qualification_status'],'INELIGIBLE_REPLAY_INTEGRITY_FAILURE')
            self.assertFalse(result['collection_provenance_independently_verified']);self.assertFalse(result['full_B4_pass'])
        with tempfile.TemporaryDirectory()as folder:
            result,runtime,old,new=self.run_stored(folder,runtime=Runtime(bad_initial=True))
            self.assertFalse(any(x[0]=='advance'for x in runtime.calls))
            self.assertEqual(result['counts']['agreement'],4);self.assertEqual(result['counts']['comparison_refused'],4)

    def test_typed_state_unsigned_chance_actions_and_missing_after_never_redraw(self):
        original=state();decoded=r.decode_state(packet(original)['state'])
        self.assertEqual(decoded.keys.dtype,np.uint32);self.assertEqual(decoded.keys.tolist(),[[2**31+1,2**32-1]])
        bad=packet(original)['state'];del bad['hp']
        with self.assertRaises(ValueError):r.decode_state(bad)
        bad=chance();bad['draws'][0][0][0]=2
        with self.assertRaises(ValueError):r.decode_chance(bad)
        bad=chance();bad['phazing']=[[2**32,0]]
        with self.assertRaises(ValueError):r.decode_chance(bad)
        with self.assertRaises(ValueError):r.decode_actions([[True,False]])
        runtime=Runtime();replay=r.Replay(runtime);replay.constructor(dict(raw=packet(original),initialized=packet(original)))
        record=dict(before=packet(original),actions=[[0,1]],chance=chance(),engine_error={'literal':'original'})
        candidates,bits,pre,post=replay.step(record)
        self.assertIsNotNone(candidates[0]);self.assertIsNone(pre);self.assertIn('NO_NEW_OUTCOME',post['reason'])
        self.assertFalse(any(x[0]=='advance'for x in runtime.calls))

    def test_final_owner_manifest_and_dataset_hash_guards(self):
        with tempfile.TemporaryDirectory()as folder:
            root,plan,summary,bindings=dataset(folder)
            runtime_plan=copy.deepcopy(plan);runtime_plan['source_commit']='b'*40
            (root/'runtime-plan.json').write_text(c._json(runtime_plan))
            job=dict(metadata=dict(uid='actual-job'),status=dict(conditions=[dict(type='Complete',status='True')]))
            pod=dict(metadata=dict(ownerReferences=[dict(kind='Job',uid='actual-job')]),status=dict(phase='Succeeded',
                containerStatuses=[dict(state=dict(terminated=dict(exitCode=0)))]))
            (root/'job.json').write_text(c._json(job));(root/'pod.json').write_text(c._json(pod))
            ref=lambda path:dict(path=str(path),sha256=c.digest(path))
            manifest=dict(schema=r.SCHEMA,original_root=str(root),files={k:ref(root/v)for k,v in r.FILES.items()},
                final_owner=dict(job=ref(root/'job.json'),pod=ref(root/'pod.json')),runtime_plan=ref(root/'runtime-plan.json'),
                original_reducer=ref(root/'original-reducer.py'))
            r.validate_inputs(manifest,expected_cells=2)
            with self.assertRaises(ValueError):r.validate_inputs(manifest)  # PUBLIC entry requires512
            runtime_plan['sampling_seed']+=1;(root/'runtime-plan.json').write_text(c._json(runtime_plan))
            manifest['runtime_plan']=ref(root/'runtime-plan.json')
            with self.assertRaises(ValueError):r.validate_inputs(manifest,expected_cells=2)
            pod['status']['containerStatuses'][0]['state']['terminated']['exitCode']=False
            with self.assertRaises(ValueError):r._owner_complete(job,pod)
            job['status']['conditions']=[]
            with self.assertRaises(ValueError):r._owner_complete(job,pod)
            row=next(r._records(root/'decisions.jsonl'))[0]
            with (root/row['tensor_path']).open('ab')as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):r._load_row(root,row)
            (root/'physical.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):r._verify_file(manifest['files']['physical'])

    def test_frozen_legacy_bytes_before_new_routing_mode_and_no_substituted_source(self):
        with tempfile.TemporaryDirectory()as folder:
            root,plan,summary,bindings=dataset(folder,routing_mismatch=True)
            cls=r._frozen_reducer(bindings,reference=dict(path=str(root/'original-reducer.py'),sha256=c.digest(root/'original-reducer.py')))
            with self.assertRaises(TypeError):cls(source_bindings=bindings,comparison_mode='source_routing_coordinates')
            writer=c.Evidence(Path(folder)/'new');runtime=Runtime()
            try:out=r.reencode_records(runtime,root,plan,summary,bindings,root/'physical.jsonl',root/'decisions.jsonl',writer)
            finally:writer.close()
            self.assertEqual(out['comparison_mode'],'source_routing_coordinates')
            self.assertTrue(out['original_report_recomputed_exact']);self.assertEqual(out['original_counts']['joint_match'],4)
            self.assertEqual(out['counts']['joint_match'],4)
            self.assertEqual(out['switch_target_routing_coordinates']['raw_differing_views'],4)
            self.assertEqual(out['switch_target_routing_coordinates']['raw_differing_slots'],[4]*5)
            current=Path(r.__file__).with_name('device_observation_parity.py')
            with self.assertRaises(ValueError):r._frozen_reducer(bindings,reference=dict(path=str(current),sha256=c.digest(current)))
            # Equal-valued different JSON number types cannot count as byte-
            # exact original report reproduction. No new source input runs.
            summary['counts']['attempted']=8.0;runtime=Runtime();writer=c.Evidence(Path(folder)/'bad')
            try:
                with self.assertRaisesRegex(ValueError,'canonical bytes'):
                    r.reencode_records(runtime,root,plan,summary,bindings,root/'physical.jsonl',root/'decisions.jsonl',writer)
            finally:writer.close()
            self.assertEqual(runtime.calls,[])

    def test_enablement_no_runtime_and_no_native_sampler_constructor_calls(self):
        with patch.object(c,'prepare',side_effect=AssertionError('must not prepare')):
            with self.assertRaises(ValueError):r.run({},out='/shared/fixture/not-created')
        tree=ast.parse(Path(r.__file__).read_text())
        forbidden={'sample','sample_chance_inputs','construct','pool_sets','pack_sets','game','step_with_chance',
            'OwningShowdownEnv','ChanceControl','compare_trace','captured_words','build_v5_model','choose_legal'}
        for call in (x for x in ast.walk(tree)if isinstance(x,ast.Call)):
            name=call.func.attr if isinstance(call.func,ast.Attribute)else call.func.id if isinstance(call.func,ast.Name)else''
            self.assertNotIn(name,forbidden)


if __name__=='__main__':unittest.main()
