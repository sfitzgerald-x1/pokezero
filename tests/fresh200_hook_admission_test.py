"""NEW policy/causal/transport composition guards; no prior suite is rerun."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from gpu_engine import fresh200_witness_audit as native
from gpu_engine.fresh200_hook_admission import HookCursor,HookPolicy,HookAdmissionBinding,seal_parsed_catalog
from gpu_engine.fresh200_native_consumer import ClosedCommandVerifier,CommandSpec
from gpu_engine.fresh200_native_recording import NativeCommandRecorder


def canonical(value):return (json.dumps(value,ensure_ascii=False,separators=(',',':'),sort_keys=True)+'\n').encode()


class HookAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=tempfile.TemporaryDirectory(prefix='new-hook-admission-');root=Path(cls.folder.name)
        cls.causal=root/'causal';cls.closed=root/'closed';node=os.environ['FRESH200_TEST_NODE']
        for script,destination,extra in [('tests/fresh200_hook_catalog_fixture.cjs',cls.causal,[]),
                ('tests/fresh200_consumer_fixture.cjs',cls.closed,['--registration-policy'])]:
            result=subprocess.run([node,script,str(destination),*extra],capture_output=True,timeout=30)
            if result.returncode:raise RuntimeError('new synthetic fixture exit'+str(result.returncode)+': '+result.stderr.decode())
        cls.declaration=json.loads((cls.causal/'catalog.json').read_text());assert cls.declaration['synthetic_only']
        cls.m=json.loads((cls.closed/'manifest.json').read_text());assert cls.m['synthetic_only'] and cls.m['no_actual_native_game']
        cls.parser_sha=native.digest(b'synthetic manually declared full-function spans; NOT a real Native AST catalog')

    @classmethod
    def tearDownClass(cls):cls.folder.cleanup()

    def policy(self,declaration=None):
        d=self.declaration if declaration is None else declaration
        catalog=seal_parsed_catalog(d['source_pins'],d['functions'],parser_sha256=self.parser_sha)
        return HookPolicy(catalog,tuple(d['bootstrap_operations']),tuple(d['instance_operations']))

    def causal_rows(self,index):
        receipts=(self.causal/'receipts.jsonl').read_bytes().splitlines(keepends=True)
        receipt=native.document(receipts[index].decode());lines=(self.causal/'audit.jsonl').read_bytes().splitlines(keepends=True)
        rows=[(line,native.document(line.decode())) for line in lines[receipt['firstSequence']:receipt['lastSequence']+1]]
        return rows,receipts[index]

    def mutated(self,index,mutate):
        rows,receipt=self.causal_rows(index);rows=copy.deepcopy(rows);mutate([value for _,value in rows])
        return [(canonical(value),value) for _,value in rows],receipt

    def test_full_parsed_function_bytes_and_source_pins_are_checked_without_normalization(self):
        self.policy()
        d=copy.deepcopy(self.declaration);path=next(iter(d['source_pins']));d['source_pins'][path]='e'*64
        with self.assertRaisesRegex(ValueError,'source file identity differs'):self.policy(d)
        d=copy.deepcopy(self.declaration);d['functions'][0]['locations'][0]['end']-=1
        with self.assertRaisesRegex(ValueError,'full function span/hash differs'):self.policy(d)
        d=copy.deepcopy(self.declaration);d['functions'][0]['locations'][0]['source_form']='approximated-body'
        with self.assertRaisesRegex(ValueError,'unsupported'):self.policy(d)

    def test_admitted_policy_is_immutable_and_retains_identical_body_multiplicity(self):
        d=copy.deepcopy(self.declaration);policy=self.policy(d);sha=policy.sha256
        d['functions'][0]['originalSha256']='0'*64;d['bootstrap_operations'].clear()
        self.assertEqual(policy.sha256,sha);self.assertTrue(policy.bootstrap_operations)
        cursor=HookCursor();rows,_=self.causal_rows(0);cursor=cursor.candidate_command(rows,policy=policy,phase='start')
        callbacks=[e for e in cursor.hooks if e['kind']=='callback-own-apply']
        self.assertGreater(len(callbacks),len({e['operation'] for e in callbacks}));self.assertEqual(len({e['target_token'] for e in callbacks}),len(callbacks))

    def test_two_actual_recorder_commands_reconstruct_digest_before_causal_receipt_admission(self):
        policy=self.policy();hooks=HookCursor();causal=native.Cursor();controls=(self.causal/'controls.jsonl').read_bytes().splitlines()
        for index,phase in enumerate(('start','choices')):
            binding=hooks.binding(policy);hooks.admit_binding(binding,policy);rows,receipt=self.causal_rows(index)
            candidate=hooks.candidate_command(rows,policy=policy,phase=phase)
            checked=native.verify_command(io.BytesIO(b''.join(w for w,_ in rows)),receipt,
                dict(control=native.document(controls[index].decode()),battle_id='b',hook_identity_digest=candidate.digest),cursor=causal,
                source_registry_hash=self.declaration['source_registry_hash'],call_operations=policy.operations,extra_operations=('HOOK_REGISTERED',))
            self.assertEqual(len(checked),len(rows));self.assertNotEqual(candidate.head,native.ZERO);hooks=candidate
        self.assertEqual(hooks.commands,2);self.assertTrue(hooks.callbacks)

    def test_unused_missing_registration_cannot_adopt_the_runtime_roster_digest(self):
        policy=self.policy();rows,receipt=self.causal_rows(0);values=[copy.deepcopy(v) for _,v in rows]
        callback_rows=[r for r in values if r['operation']=='HOOK_REGISTERED' and r['evidence']['kind']=='callback-own-apply']
        missing=callback_rows[1];values.remove(missing);counter=0
        for row in values:
            if row['operation']=='HOOK_REGISTERED':row['evidence']['hook_id']=counter;counter+=1
        candidate=HookCursor().candidate_command([(canonical(v),v) for v in values],policy=policy,phase='start')
        self.assertNotEqual(candidate.digest,native.document(receipt.decode())['hookIdentityDigest'])
        controls=(self.causal/'controls.jsonl').read_bytes().splitlines()
        with self.assertRaisesRegex(ValueError,'incorrectly bound witness receipt'):
            native.verify_command(io.BytesIO(b''.join(canonical(v) for v in values)),receipt,
                dict(control=native.document(controls[0].decode()),battle_id='b',hook_identity_digest=candidate.digest),cursor=native.Cursor(),
                source_registry_hash=self.declaration['source_registry_hash'],call_operations=policy.operations,extra_operations=('HOOK_REGISTERED',))

    def test_discovery_result_index_and_exact_function_objects_are_not_interchangeable(self):
        policy=self.policy()
        def wrong_index(rows):
            next(r for r in rows if r['operation']=='HOOK_REGISTERED' and r['evidence']['discovery_site']['kind']=='handler-result')['evidence']['discovery_site']['index']=0
        def wrong_token(rows):
            next(r for r in rows if r['operation']=='CALL_ENTER' and r['frame']['operation'].startswith('Callback:'))['evidence']['function_token']='h999999'
        def wrong_call(rows):
            next(r for r in rows if r['operation']=='HOOK_REGISTERED' and r['evidence']['kind']=='callback-own-apply')['evidence']['discovery_call_id']=False
        for mutate in (wrong_index,wrong_token,wrong_call):
            with self.subTest(mutate=mutate.__name__):
                rows,_=self.mutated(0,mutate);cursor=HookCursor()
                with self.assertRaises(ValueError):cursor.candidate_command(rows,policy=policy,phase='start')
                self.assertEqual(cursor.hooks,[]);self.assertEqual(cursor.head,native.ZERO)

    def test_unknown_function_body_and_registration_order_fail_source_admission(self):
        policy=self.policy()
        def body(rows):next(r for r in rows if r['operation']=='HOOK_REGISTERED')['evidence']['originalSha256']='f'*64
        def order(rows):next(r for r in rows if r['operation']=='HOOK_REGISTERED')['evidence']['hook_id']=1
        for mutate in (body,order):
            with self.subTest(mutate=mutate.__name__):
                rows,_=self.mutated(0,mutate)
                with self.assertRaises(ValueError):HookCursor().candidate_command(rows,policy=policy,phase='start')

    def test_sealed_policy_binding_refuses_changed_catalog_or_registration_parents(self):
        policy=self.policy();cursor=HookCursor();binding=cursor.binding(policy)
        for name in vars(binding):
            values=vars(binding).copy();values[name]='f'*64
            with self.subTest(field=name),self.assertRaisesRegex(ValueError,'pre-submission'):
                cursor.admit_binding(HookAdmissionBinding(**values),policy)
        rows,_=self.causal_rows(0);after=cursor.candidate_command(rows,policy=policy,phase='start')
        with self.assertRaisesRegex(ValueError,'pre-submission'):after.admit_binding(binding,policy)

    def closed_verifier(self):
        policy=self.policy(self.m['registration_catalog']);m=self.m
        return ClosedCommandVerifier(session=m['session'],battle_id=m['battle_id'],source_registry_hash=m['source_registry_hash'],
            expected_methods=m['expected_methods'],call_operations=policy.operations,hook_policy=policy)

    def closed_spec(self,index,verifier,**overrides):
        c=self.m['commands'][index]
        values=dict(stdin_wire=(self.closed/c['stdin']['path']).read_bytes(),native_control_wire=(self.closed/c['native_control']['path']).read_bytes(),
            transport_control_wire=(self.closed/c['transport_control']['path']).read_bytes(),battle_id='b',
            hook_identity_digest=None,hook_admission=verifier.hook_cursor.binding(verifier.hook_policy));values.update(overrides)
        return CommandSpec(**values)

    def closed_streams(self):return {name:io.BytesIO((self.closed/(name+'.jsonl')).read_bytes()) for name in ('audit','events','stdout')}

    def closed_verify(self,index,verifier,streams,**overrides):
        c=self.m['commands'][index];values=dict(spec=self.closed_spec(index,verifier),native_receipt_wire=(self.closed/c['native_receipt']['path']).read_bytes(),
            transport_receipt_wire=(self.closed/c['transport_receipt']['path']).read_bytes(),**streams);values.update(overrides)
        return verifier.verify(**values)

    def test_source_policy_closed_command_and_transport_cursors_compose_through_natural_eof(self):
        verifier=self.closed_verifier();streams=self.closed_streams()
        for index in range(5):self.assertFalse(self.closed_verify(index,verifier,streams)['qualification'])
        result=verifier.finish_artifacts(**streams,expected_command_ids=[x['command_id'] for x in verifier.completed])
        self.assertFalse(result['qualification']);self.assertEqual(verifier.hook_cursor.commands,5);self.assertEqual(len(verifier.hook_cursor.hooks),2)

    def test_post_registration_transport_failure_rolls_back_every_cursor_and_saved_stream(self):
        verifier=self.closed_verifier();streams=self.closed_streams();c=self.m['commands'][0]
        receipt=json.loads((self.closed/c['transport_receipt']['path']).read_bytes());receipt['stdout_frontier']+=1;receipt.pop('hash')
        receipt['hash']=native.digest(canonical(receipt)[:-1]);bad=canonical(receipt)
        with self.assertRaisesRegex(ValueError,'summary differs'):
            self.closed_verify(0,verifier,streams,transport_receipt_wire=bad)
        self.assertEqual(verifier.hook_cursor.head,native.ZERO);self.assertEqual(verifier.hook_cursor.hooks,[])
        self.assertEqual(verifier.native_cursor.sequence,0);self.assertEqual(verifier.transport_cursor.sequence,0)
        self.assertTrue(all(s.tell()==0 for s in streams.values()));self.assertEqual(verifier.completed,[])
        self.assertFalse(self.closed_verify(0,verifier,streams)['qualification'])

    def test_exact_receipt_digest_and_policy_forms_are_exclusive_before_submission(self):
        verifier=self.closed_verifier();spec=self.closed_spec(0,verifier,hook_identity_digest=self.m['hook_identity_digest'])
        with self.assertRaises(ValueError):verifier.admit_spec(spec)
        plain=self.closed_spec(0,verifier,hook_identity_digest=self.m['hook_identity_digest'],hook_admission=None)
        with self.assertRaisesRegex(ValueError,'cannot adopt'):verifier.admit_spec(plain)

    def test_borrowed_callback_apply_preserves_original_work_but_poisons_admission(self):
        result=json.loads((self.causal/'borrowed.json').read_text())
        self.assertEqual((result['original_result'],result['original_calls']),(17,1));self.assertTrue(result['poisoned'])
        self.assertFalse(result['qualification']);self.assertTrue(result['guarded_pointer_unchanged'])

    def test_running_recorder_seals_policy_bindings_before_original_stdin_and_closes_all_cursors(self):
        # Real pipes to a synthetic saved-trace child only; never Native.
        with tempfile.TemporaryDirectory(prefix='new-policy-recorder-') as directory:
            root=Path(directory);bridge=root/'bridge';bridge.mkdir();read_fd,write_fd=os.pipe()
            completion=os.fdopen(read_fd,'rb',buffering=0);child=None;recorder=None
            try:
                child=subprocess.Popen([sys.executable,'tests/fresh200_recording_child.py',str(self.closed),str(bridge),'normal',str(write_fd)],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,pass_fds=(write_fd,));os.close(write_fd);write_fd=None
                verifier=self.closed_verifier();registry=(json.dumps(dict(registry=dict(synthetic=True),sourceRegistryHash=self.m['source_registry_hash']),separators=(',',':'))+'\n').encode()
                recorder=NativeCommandRecorder(directory=bridge,artifacts=root/'artifacts',child=child,completion=completion,verifier=verifier,
                    expected_registry_wire=registry,deadline_monotonic=time.monotonic()+30,max_commands=5)
                for index in range(5):recorder.record(self.closed_spec(index,verifier))
                result=recorder.finish(expected_command_ids=[x['command_id'] for x in recorder.records])
                self.assertFalse(result['qualification']);self.assertEqual(result['child_exit_code'],0);self.assertEqual(verifier.hook_cursor.commands,5)
            finally:
                if write_fd is not None:os.close(write_fd)
                if child is not None:
                    if child.poll() is None:child.terminate()
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=3)
                    if recorder is not None:
                        for capture in recorder._captures.values():capture.thread.join(timeout=3)
                    for stream in (child.stdin,child.stdout,child.stderr):stream.close()
                completion.close()


if __name__=='__main__':unittest.main()
