"""Immutable first8 diagnostic controls with fake stored numerical worker."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from gpu_engine import device_recorded_reencoding as r, device_observation_collector as c
import test_gpu_engine_device_recorded_reencoding as fixture

BASE='9beed4fd95e5fbabd756e44115e7bfc81a03efd0'


def setup(folder):
    folder=Path(folder);(folder/'seed').mkdir()
    seedroot,plan,_,bindings=fixture.dataset(folder/'seed')
    plan=copy.deepcopy(plan)
    plan['cells']=[dict(stream_id=10+n,root_id=1000+n,native_seed=2000+n,team_indices=[2*n,2*n+1])for n in range(512)]
    bindings=copy.deepcopy(bindings)
    name='gpu_engine/engine_device_observation.py'
    payload=subprocess.check_output(['git','show',plan['source_commit']+':'+name],cwd=c.ROOT)
    bindings['source_files'][name]=hashlib.sha256(payload).hexdigest()
    for key in ('pool','vocabulary','labels','gender_policy'):bindings[key]=plan[key]
    snapshot=folder/'snapshot';writer=c.Evidence(snapshot)
    for n,cell in enumerate(plan['cells'][:8]):
        writer.event(dict(kind='START',status='ATTEMPTED',**cell))
        if n!=2:
            writer.boundary(dict(kind='CONSTRUCTOR',stream_id=cell['stream_id'],
                raw=fixture.packet(fixture.state(n)),initialized=fixture.packet(fixture.state(n))))
            for step in range(2):
                agreement=(None if n==1 else False if n==3 else True)
                value=dict(kind='STEP',stream_id=cell['stream_id'],request=step,
                    before=fixture.packet(fixture.state(n,step)),after=fixture.packet(fixture.state(n,step+1)),
                    pre_agreement=agreement,actions=[[0,1]],chance=fixture.chance())
                sha=writer.boundary(value)
                for side in (0,1):
                    row=dict(decision_id=f"{cell['stream_id']}-{step}-{side}",stream_id=cell['stream_id'],
                        root_id=cell['root_id'],request=step,owner_side=side,turn=step,agreement=agreement,
                        line_kinds=['move'],refusal=None,physical_binding_sha256=sha,candidate_error=0,reference_metadata={})
                    writer.decision(row,fixture.view(),fixture.view())
        final=dict(executed_steps=0 if n==2 else 2,requested_views=0 if n==2 else 4,
            candidate_error_views=0,candidate_error_kind_views={})
        if n!=2:final.update(final_turn=2,final_done=True,final_winner=int(fixture.state(n,2).winner[0]))
        writer.event(dict(kind='END',status='COLLECTION_CELL_FAILED'if n==2 else'TERMINAL',**cell,**final))
    writer.close()
    (folder/'original-plan.json').write_text(c._json(plan)+'\n')
    runtime=copy.deepcopy(plan);runtime['source_commit']='b'*40
    (folder/'runtime-plan.json').write_text(c._json(runtime)+'\n')
    (folder/'source-bindings.json').write_text(c._json(bindings)+'\n')
    (folder/'original-reducer.py').write_bytes((seedroot/'original-reducer.py').read_bytes())
    job=dict(metadata=dict(uid='running-job'),status=dict(active=1,conditions=[]))
    pod=dict(metadata=dict(ownerReferences=[dict(kind='Job',uid='running-job')]),status=dict(phase='Running',
        containerStatuses=[dict(state=dict(running=dict(startedAt='literal-snapshot-time')))]))
    (folder/'job.json').write_text(c._json(job));(folder/'pod.json').write_text(c._json(pod))
    ref=lambda p:dict(path=str(p),sha256=c.digest(p))
    last=list(r._records(snapshot/'journal.jsonl'))[-1]
    manifest=dict(schema=r.PREFIX_SCHEMA,original_plan=ref(folder/'original-plan.json'),
        original_source_bindings=ref(folder/'source-bindings.json'),original_reducer=ref(folder/'original-reducer.py'),
        runtime_plan=ref(folder/'runtime-plan.json'),source_owner=dict(job=ref(folder/'job.json'),pod=ref(folder/'pod.json')),
        selection=dict(policy='FIRST_DECLARATION_ROWS_ALL_DISPOSITIONS',count=8,original_declared_cells=512),
        snapshot=dict(root=str(snapshot),files={key:ref(snapshot/r.FILES[key])for key in ('physical','decisions','journal')},
            end_boundary=dict(stream_id=plan['cells'][7]['stream_id'],journal_end_sha256=last[1])))
    return manifest


class DeviceRecordedPrefixTests(unittest.TestCase):
    def test_original_end_error_and_final_boundary_facts_are_joined(self):
        for change in ('error','final'):
            with self.subTest(change=change),tempfile.TemporaryDirectory()as folder:
                manifest=setup(folder);root=Path(manifest['snapshot']['root'])
                key='decisions'if change=='error'else'journal';path=root/(key+'.jsonl')
                rows=[row for row,_ in r._records(path)]
                if change=='error':rows[0]['candidate_error']=1
                else:next(row for row in rows if row['kind']=='END')['final_done']=False
                path.write_text(''.join(c._json(row)+'\n'for row in rows))
                manifest['snapshot']['files'][key]['sha256']=c.digest(path)
                with self.assertRaisesRegex(ValueError,'prefix END'):r.validate_prefix_inputs(manifest)

    def test_running_immutable_first8_retains_failed_roots_and_final_is_not_accepted(self):
        with tempfile.TemporaryDirectory()as folder:
            manifest=setup(folder);root,plan,bindings,runtime,ends=r.validate_prefix_inputs(manifest)
            self.assertEqual(len(plan['cells']),512);self.assertEqual(len(ends),8)
            self.assertEqual(ends[2]['status'],'COLLECTION_CELL_FAILED')
            self.assertEqual([x['stream_id']for x in ends],[x['stream_id']for x in plan['cells'][:8]])
            with self.assertRaises(ValueError):r.validate_inputs(manifest)
            for count in (7,9,8.0):
                bad=copy.deepcopy(manifest);bad['selection']['count']=count
                with self.assertRaises(ValueError):r.validate_prefix_inputs(bad)
            bad=copy.deepcopy(manifest);bad['snapshot']['end_boundary']['journal_end_sha256']='f'*64
            with self.assertRaises(ValueError):r.validate_prefix_inputs(bad)
            journal=Path(manifest['snapshot']['files']['journal']['path']);payload=journal.read_bytes()
            journal.write_bytes(b'\n'.join(payload.splitlines()[:-1])+b'\n')
            partial=copy.deepcopy(manifest);partial['snapshot']['files']['journal']['sha256']=c.digest(journal)
            with self.assertRaisesRegex(ValueError,'actual END'):r.validate_prefix_inputs(partial)
            journal.write_bytes(payload)
            # Keep immutable bytes even if statuses are errors, never choose
            # a successful replacement or silently include the ninth root.
            path=Path(manifest['snapshot']['files']['decisions']['path'])
            with path.open('a')as stream:stream.write(c._json(dict(stream_id=18))+'\n')
            manifest['snapshot']['files']['decisions']['sha256']=c.digest(path)
            with self.assertRaisesRegex(ValueError,'out-of-prefix'):r.validate_prefix_inputs(manifest)

    def test_prefix_full_pipeline_is_diagnostic_no_gate_no_plan_rewrite(self):
        with tempfile.TemporaryDirectory()as folder:
            manifest=setup(folder);out=Path(folder)/'out';old_hash=c.digest(manifest['original_plan']['path'])
            original=Path.is_relative_to
            def only_test_output(path,base):
                if path==out and base in (Path('/shared/scott'),Path('/store')):return True
                return original(path,base)
            with patch.object(c,'prepare',return_value=fixture.Runtime()),patch.object(Path,'is_relative_to',only_test_output):
                result=r.run_prefix(manifest,out=out,enable_reencoding=True)
            report=json.loads(Path(result['summary_path']).read_text())
            self.assertEqual(report['original_declared_roots'],512);self.assertEqual(report['selected_diagnostic_roots'],8)
            self.assertEqual(report['unselected_roots'],504);self.assertIn('UNKNOWN_NOT_EVALUATED',report['unselected_diagnostic_status'])
            self.assertEqual(report['original_cells'][2]['status'],'COLLECTION_CELL_FAILED')
            self.assertEqual(report['counts']['attempted'],28);self.assertEqual(report['counts']['agreement'],20)
            self.assertEqual(report['counts']['state_agreement_unknown'],4);self.assertEqual(report['counts']['state_disagreement'],4)
            self.assertTrue(report['original_report_recomputed_exact']);self.assertFalse(report['numerical_gate_only'])
            self.assertFalse(report['full_B4_pass']);self.assertFalse(report['collection_provenance_independently_verified'])
            self.assertIsNone(report['original_full_cohort_counts']);self.assertEqual(report['new_games'],0)
            self.assertEqual(c.digest(manifest['original_plan']['path']),old_hash)
            baseline=json.loads((out/'prefix-original-legacy-summary.json').read_text())
            self.assertFalse(baseline['numerical_gate_only']);self.assertFalse(baseline['full_B4_pass'])
            self.assertEqual(baseline['frozen_original_reducer_report']['counts']['attempted'],28)

    def test_missing_end_running_owner_and_strict_final_source_are_fail_closed(self):
        with tempfile.TemporaryDirectory()as folder:
            manifest=setup(folder)
            pod=json.loads(Path(manifest['source_owner']['pod']['path']).read_text());pod['status']['phase']='Succeeded'
            Path(manifest['source_owner']['pod']['path']).write_text(c._json(pod))
            manifest['source_owner']['pod']['sha256']=c.digest(manifest['source_owner']['pod']['path'])
            with self.assertRaises(ValueError):r.validate_prefix_inputs(manifest)
            with patch.object(c,'prepare',side_effect=AssertionError('must not prepare')):
                with self.assertRaises(ValueError):r.run_prefix(manifest,out='/shared/fixture/not-created')
        tree=ast.parse(Path(r.__file__).read_text())
        previous=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/device_recorded_reencoding.py'],cwd=c.ROOT,text=True))
        old={x.name:ast.dump(x,include_attributes=False)for x in previous.body if isinstance(x,ast.FunctionDef)}
        current={x.name:ast.dump(x,include_attributes=False)for x in tree.body if isinstance(x,ast.FunctionDef)}
        for name in ('run','validate_inputs','_owner_complete','reencode_records','Replay' if 'Replay'in old else 'decode_state'):
            self.assertEqual(current[name],old[name])


if __name__=='__main__':unittest.main()
