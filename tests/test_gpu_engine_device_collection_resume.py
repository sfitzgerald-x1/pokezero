"""Interrupted inert controllers only; no engine, model or simulator runs."""
from contextlib import ExitStack,contextmanager
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from gpu_engine import device_observation_collector as c
import test_gpu_engine_device_observation_collector as f


def plan():
    ref=dict(path='/inert-not-read',sha256='a'*64)
    return dict(schema=c.SCHEMA,source_commit='b'*40,pool=ref,vocabulary=ref,labels=ref,gender_policy=ref,
        showdown_root='/inert',node='/inert',node_sha256='c'*64,constructor_seed=1,sampling_seed=2,
        gender_seed=3,gender_domain=4,max_requests=3,max_turns=100,
        cells=[dict(f.CELL,stream_id=12+i,root_id=22+i,native_seed=32+i,team_indices=[2*i,2*i+1]) for i in range(3)],
        policy='PCG64_SEEDSEQUENCE_UNIFORM_PHYSICAL_LEGAL_V1')


class Runtime(f.Runtime):
    def __init__(self):
        super().__init__();self.bindings={'source_commit':'b'*40};self.started=[];self.constructed=[]
        self.current=None;self.interrupt=True;self.manifest_calls=0
    def construct(self,cell):
        if cell['stream_id'] not in self.started:raise AssertionError('constructor preceded durable START')
        self.current=cell['stream_id'];self.constructed.append(self.current)
        return super().construct(cell)
    def advance(self,s,a,t,chance):
        if self.interrupt and self.current==13 and int(s.turn[0])==1:raise KeyboardInterrupt('inert interruption')
        return super().advance(s,a,t,chance)
    def column_manifests(self,trackers):
        self.manifest_calls+=1
        return [dict(owner_side=s,status='UNKNOWN',columns=None,error={'reason':'inert no actual trackers'}) for s in (0,1)]


@contextmanager
def harness(runtime):
    original=c.collect_game;Evidence=c.Evidence;sync=os.fsync;synced=[]
    def fsync(fd):synced.append(fd);return sync(fd)
    class Durable(Evidence):
        def event(self,row):
            before=len(synced);super().event(row)
            if row['kind']=='START':
                if self.events.fileno() not in synced[before:]:raise AssertionError('START not fsynced')
                runtime.started.append(row['stream_id'])
    def inert(*args,**kwargs):return original(*args,**kwargs,env_factory=f.Env,coupling_factory=f.Coupling)
    with ExitStack() as stack:
        stack.enter_context(patch.object(c,'prepare',return_value=runtime))
        stack.enter_context(patch.object(Path,'is_relative_to',return_value=True))
        stack.enter_context(patch.object(c,'Evidence',Durable))
        stack.enter_context(patch.object(c.os,'fsync',side_effect=fsync))
        stack.enter_context(patch.object(c,'collect_game',side_effect=inert))
        stack.enter_context(patch('gpu_engine.finite_stream_comparison.pack_sets',return_value=['set1','set2']))
        stack.enter_context(patch('gpu_engine.finite_gameplay_comparison.native_evidence',side_effect=lambda snapshot,env:dict(turn=snapshot.turn)))
        stack.enter_context(patch('gpu_engine.finite_gameplay_comparison.native_actions',return_value={'p1':0,'p2':0}))
        yield


def start_interrupted(root,runtime,p):
    with harness(runtime),unittest.TestCase().assertRaises(KeyboardInterrupt):
        c.collect(p,out=root,enable_collection=True,comparison_mode='source_routing_coordinates')


class DeviceCollectionResumeTests(unittest.TestCase):
    def test_complete_and_started_cells_orphan_and_partial_tail_are_never_replayed(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';runtime=Runtime();p=plan();start_interrupted(out,runtime,p)
            self.assertEqual(runtime.constructed,[12,13]);self.assertEqual(runtime.manifest_calls,1)
            first_end=next(line for line in (out/'journal.jsonl').read_bytes().splitlines() if b'"kind":"END"' in line)
            tensors={str(path):path.read_bytes() for path in (out/'rows').glob('*.npz')}
            manifest=(out/'column-manifest.json').read_bytes()
            with (out/'decisions.jsonl').open('ab') as stream:stream.write(b'{"unfinished":')
            before=(out/'decisions.jsonl').read_bytes()
            orphan=out/'rows/resume-000000000500-unknown.npz'
            with orphan.open('xb') as stream:np.savez(stream,literal=np.array([9],np.int32))
            orphan_bytes=orphan.read_bytes();runtime.interrupt=False
            with harness(runtime):
                result=c.collect(p,out=out,enable_collection=True,comparison_mode='source_routing_coordinates',resume=True)
            report=json.loads(Path(result['summary_path']).read_text())
            strict=json.loads(Path(result['legacy_strict_report']['path']).read_text())
            self.assertEqual(runtime.constructed,[12,13,14]);self.assertEqual(runtime.manifest_calls,1)
            self.assertEqual([cell['status'] for cell in report['cells']],['TERMINAL','INTERRUPTED_UNKNOWN','TERMINAL'])
            self.assertEqual(report['counts']['attempted'],14);self.assertEqual(strict['counts']['attempted'],14)
            self.assertIsNone(report['joint_match_fraction_on_ALL_agreement_views'])
            self.assertEqual(report['joint_match_fraction_on_RECORDED_agreement_views'],1)
            self.assertFalse(report['numerical_gate_only'])
            self.assertIsNone(report['cells'][1]['final_turn']);self.assertTrue(report['cells'][1]['counts_unknown'])
            self.assertEqual(report['resume_recovery']['orphan_tensor_files'],1)
            self.assertEqual(report['resume_recovery']['unparsed_trailing_fragments'],1)
            self.assertIn(first_end,(out/'journal.jsonl').read_bytes().splitlines())
            self.assertEqual((out/'column-manifest.json').read_bytes(),manifest)
            for path,blob in tensors.items():self.assertEqual(Path(path).read_bytes(),blob)
            recovery=out/'recovery/000001';self.assertEqual((recovery/'orphans'/orphan.name).read_bytes(),orphan_bytes)
            tail=report['resume_recovery']['tail_repairs'][0]
            restored=(out/'decisions.jsonl').read_bytes()[:tail['prefix_bytes']]+(out/tail['preserved_path']).read_bytes()
            self.assertEqual(restored,before)
            resumed=list((out/'rows').glob('resume-*.npz'))
            self.assertEqual(len(resumed),6);self.assertGreaterEqual(min(int(p.name.split('-')[1]) for p in resumed),501)
            # Re-running resume after completion rebuilds the same raw rows,
            # publishes a separate report, and calls no constructor/chance.
            samples=len(runtime.samples)
            with harness(runtime):again=c.collect(p,out=out,enable_collection=True,comparison_mode='source_routing_coordinates',resume=True)
            self.assertEqual(runtime.constructed,[12,13,14]);self.assertEqual(len(runtime.samples),samples)
            self.assertNotEqual(result['summary_path'],again['summary_path'])
            self.assertEqual(json.loads(Path(again['summary_path']).read_text())['counts'],report['counts'])

    def test_exact_plan_source_mode_and_midstream_or_tensor_corruption_refuse_before_constructor(self):
        for change in ('plan','mode','source','journal','tensor','turn','eligibility'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                out=Path(d)/'out';runtime=Runtime();p=plan();start_interrupted(out,runtime,p)
                selected=copy.deepcopy(p);mode='source_routing_coordinates'
                if change=='plan':selected['sampling_seed']=55
                if change=='mode':mode='legacy_strict'
                if change=='source':runtime.bindings={'source_commit':'c'*40}
                if change=='journal':
                    path=out/'journal.jsonl';path.write_bytes(b'not-json\n'+path.read_bytes())
                if change=='tensor':
                    path=next((out/'rows').glob('*.npz'));path.write_bytes(path.read_bytes()+b'changed')
                if change=='turn':
                    path=out/'decisions.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]['turn']+=10;path.write_text(''.join(c._json(row)+'\n' for row in rows))
                if change=='eligibility':
                    path=out/'physical.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
                    next(row for row in rows if row['kind']=='STEP').pop('pre_agreement')
                    path.write_text(''.join(c._json(row)+'\n' for row in rows))
                preserved={str(path):path.read_bytes() for path in (out/'rows').glob('*.npz')}
                journal=(out/'journal.jsonl').read_bytes();runtime.interrupt=False
                with harness(runtime),self.assertRaises(ValueError):
                    c.collect(selected,out=out,enable_collection=True,comparison_mode=mode,resume=True)
                self.assertEqual(runtime.constructed,[12,13]);self.assertEqual((out/'journal.jsonl').read_bytes(),journal)
                for path,blob in preserved.items():self.assertEqual(Path(path).read_bytes(),blob)
                self.assertFalse((out/'recovery').exists())

    def test_start_only_and_complete_unterminated_line_preserve_unknown_and_original_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';out.mkdir();(out/'rows').mkdir();p=plan();runtime=Runtime();runtime.interrupt=False
            from gpu_engine import device_observation_resume as recovery
            for name,value in {'plan.json':p,'source-bindings.json':runtime.bindings,
                    'comparison-config.json':c._comparison_choice(p,'source_routing_coordinates')}.items():
                recovery.durable(out/name,(c._json(value)+'\n').encode())
            start=dict(kind='START',**p['cells'][0],status='ATTEMPTED',executed_steps=0,requested_views=0,
                candidate_error_views=0,candidate_error_kind_views={})
            original=c._json(start).encode();(out/'journal.jsonl').write_bytes(original)
            (out/'decisions.jsonl').touch();(out/'physical.jsonl').touch()
            with harness(runtime):result=c.collect(p,out=out,enable_collection=True,comparison_mode='source_routing_coordinates',resume=True)
            report=json.loads(Path(result['summary_path']).read_text())
            self.assertEqual(runtime.constructed,[13,14]);self.assertEqual(runtime.manifest_calls,0)
            self.assertEqual(report['cells'][0]['status'],'INTERRUPTED_UNKNOWN')
            self.assertEqual(report['installed_column_manifest']['status'],'UNKNOWN')
            preserved=out/'recovery/000001/journal.jsonl.tail.bin'
            self.assertEqual(preserved.read_bytes(),original)
            self.assertTrue((out/'journal.jsonl').read_bytes().startswith(original+b'\n'))
            self.assertEqual(report['counts']['attempted'],12)

    def test_interruption_during_quarantine_preserves_intent_and_unknown_mass(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';runtime=Runtime();p=plan();start_interrupted(out,runtime,p)
            with (out/'decisions.jsonl').open('ab') as stream:stream.write(b'{"partial":')
            orphan=out/'rows/unknown.npz'
            with orphan.open('xb') as stream:np.savez(stream,literal=np.ones(1))
            runtime.interrupt=False
            with harness(runtime),patch.object(Path,'rename',side_effect=KeyboardInterrupt('inert recovery interruption')),self.assertRaises(KeyboardInterrupt):
                c.collect(p,out=out,enable_collection=True,comparison_mode='source_routing_coordinates',resume=True)
            self.assertTrue((out/'recovery/000001/intent.json').exists())
            self.assertFalse((out/'recovery/000001/report.json').exists())
            self.assertEqual(runtime.constructed,[12,13])
            with harness(runtime):result=c.collect(p,out=out,enable_collection=True,comparison_mode='source_routing_coordinates',resume=True)
            report=json.loads(Path(result['summary_path']).read_text())
            self.assertEqual(runtime.constructed,[12,13,14])
            self.assertIsNone(report['joint_match_fraction_on_ALL_agreement_views'])
            self.assertIn('recovery/000001/intent.json',report['resume_recovery']['prior_recoveries'])
            self.assertEqual((out/'recovery/000001/decisions.jsonl.tail.bin').read_bytes(),b'{"partial":')

    def test_empty_present_view_is_not_guessed_to_be_absent_on_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';runtime=Runtime();p=plan()
            runtime.factory=NS(game=lambda **kw:NS(feed=lambda lines:None,view=lambda:({},None),metadata=lambda:{}))
            start_interrupted(out,runtime,p);runtime.interrupt=False
            with harness(runtime):result=c.collect(p,out=out,enable_collection=True,comparison_mode='source_routing_coordinates',resume=True)
            report=json.loads(Path(result['summary_path']).read_text())
            self.assertEqual(report['counts']['comparison_refused'],14)
            self.assertTrue(report['refusals_by_kind_column'])
            self.assertTrue(all(row['reason']=='MALFORMED_VIEW_FIELD' for row in report['refusals_by_kind_column']))


if __name__=='__main__':unittest.main()
