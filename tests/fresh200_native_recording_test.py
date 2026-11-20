"""NEW finite recording guards with real pipes and a synthetic child only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from gpu_engine.fresh200_native_consumer import ClosedCommandVerifier,CommandSpec
from gpu_engine.fresh200_native_recording import NativeCommandRecorder


class RecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=tempfile.TemporaryDirectory(prefix='new-recording-fixture-');cls.root=Path(cls.fixture.name)
        result=subprocess.run([os.environ['FRESH200_TEST_NODE'],'tests/fresh200_consumer_fixture.cjs',str(cls.root)],capture_output=True,timeout=30)
        if result.returncode:raise RuntimeError('synthetic fixture exit'+str(result.returncode)+': '+result.stderr.decode())
        cls.m=json.loads((cls.root/'manifest.json').read_text())

    @classmethod
    def tearDownClass(cls):cls.fixture.cleanup()

    def setUp(self):
        self.attempt=tempfile.TemporaryDirectory(prefix='new-recording-attempt-');self.addCleanup(self.attempt.cleanup)
        self.directory=Path(self.attempt.name)/'bridge';self.directory.mkdir();self.child=None;self.completion=None;self.recorder=None
        self.addCleanup(self.release_synthetic_child)

    def release_synthetic_child(self):
        # Test-owned fake child only. The recorder itself never stops a child
        # after a harness/observer error; actual science uses its real guards.
        if self.child is not None:
            if self.child.poll() is None:self.child.terminate()
            try:self.child.wait(timeout=3)
            except subprocess.TimeoutExpired:self.child.kill();self.child.wait(timeout=3)
            if self.recorder is not None:
                for capture in self.recorder._captures.values():capture.thread.join(timeout=3)
            for stream in (self.child.stdin,self.child.stdout,self.child.stderr):stream.close()
        if self.completion is not None:self.completion.close()

    def create(self,mode='normal',seconds=30):
        read_fd,write_fd=os.pipe();self.completion=os.fdopen(read_fd,'rb',buffering=0)
        try:
            self.child=subprocess.Popen([sys.executable,'tests/fresh200_recording_child.py',str(self.root),str(self.directory),mode,str(write_fd)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,pass_fds=(write_fd,))
        finally:os.close(write_fd)
        m=self.m;verifier=ClosedCommandVerifier(session=m['session'],battle_id=m['battle_id'],source_registry_hash=m['source_registry_hash'],
            expected_methods=m['expected_methods'],call_operations=set(m['call_operations']))
        registry=(json.dumps(dict(registry=dict(synthetic=True),sourceRegistryHash=m['source_registry_hash']),separators=(',',':'))+'\n').encode()
        self.recorder=NativeCommandRecorder(directory=self.directory,artifacts=Path(self.attempt.name)/'artifacts',child=self.child,
            completion=self.completion,verifier=verifier,expected_registry_wire=registry,deadline_monotonic=time.monotonic()+seconds,max_commands=5)
        return self.recorder

    def spec(self,index):
        command=self.m['commands'][index]
        return CommandSpec(stdin_wire=(self.root/command['stdin']['path']).read_bytes(),
            native_control_wire=(self.root/command['native_control']['path']).read_bytes(),
            transport_control_wire=(self.root/command['transport_control']['path']).read_bytes(),battle_id='b',hook_identity_digest=self.m['hook_identity_digest'])

    def complete(self,recorder):
        for index in range(5):recorder.record(self.spec(index))
        return recorder.finish(expected_command_ids=[x['command_id'] for x in recorder.records])

    def test_original_real_pipes_controls_snapshots_and_natural_close_are_preserved(self):
        recorder=self.create();result=self.complete(recorder)
        self.assertEqual(result['status'],'RAW_CLOSED_ARTIFACT_SET_RECORDED_ONLY');self.assertFalse(result['qualification']);self.assertEqual(result['child_exit_code'],0)
        self.assertEqual((recorder.artifacts/'stdout.bin').read_bytes(),(self.root/'stdout.jsonl').read_bytes())
        self.assertEqual((recorder.artifacts/'stderr.bin').read_bytes(),b'original unparsed stderr \xff\n')
        for index,record in enumerate(recorder.records):
            for name,key in (('stdin.jsonl','stdin'),('native-control.json','native_control'),('transport-control.json','transport_control')):
                self.assertEqual((recorder.artifacts/('command-'+str(index))/name).read_bytes(),(self.root/self.m['commands'][index][key]['path']).read_bytes())
                self.assertEqual(record['inputs'][name]['sha256'],self.m['commands'][index][key]['sha256'])
        self.assertTrue(all(x.eof for x in recorder._captures.values()))

    def test_extra_stdout_is_fatal_to_recording_but_never_stops_the_child(self):
        recorder=self.create('bad-stdout')
        with self.assertRaisesRegex(ValueError,'stdout frontier differs'):recorder.record(self.spec(0))
        self.assertIsNone(self.child.poll());self.assertFalse(recorder.failed['child_stopped_by_recorder'])
        self.assertTrue((recorder.artifacts/'stdout.bin').read_bytes().startswith(b'{"observer":"extra"}'))
        self.assertEqual(recorder.verifier.completed,[])
        with self.assertRaises(ValueError):recorder.record(self.spec(1))
        self.assertIsNone(self.child.poll())

    def test_expected_source_registry_is_not_adopted_from_the_child(self):
        recorder=self.create('bad-registry')
        with self.assertRaisesRegex(ValueError,'independent prelaunch source admission'):recorder.record(self.spec(0))
        self.assertIsNone(self.child.poll());self.assertEqual(recorder.verifier.completed,[])

    def test_dedicated_fd_bytes_must_equal_durable_transport_receipt(self):
        recorder=self.create('bad-notification')
        with self.assertRaisesRegex(ValueError,'durable completion differs'):recorder.record(self.spec(0))
        self.assertIsNone(self.child.poll());self.assertFalse(recorder.failed['child_stopped_by_recorder'])

    def test_next_control_parent_is_checked_before_any_stdin_submission(self):
        recorder=self.create()
        with self.assertRaisesRegex(ValueError,'before stdin write'):recorder.record(self.spec(1))
        self.assertEqual(recorder.records,[]);self.assertFalse((self.directory/'witness-audit.jsonl').exists());self.assertIsNone(self.child.poll())

    def test_original_chance_audit_is_durable_before_existing_retirement(self):
        recorder=self.create();recorder.record(self.spec(0))
        original=b'{"kind":"synthetic","value":0}\n';source=self.directory/'chance-audit.jsonl';source.write_bytes(original)
        def retire():
            saved=recorder.artifacts/'command-0/chance-audit.jsonl';self.assertEqual(saved.read_bytes(),original)
            self.assertEqual(json.loads((recorder.artifacts/'recording.json').read_text())['commands'][0]['chance_audit']['sha256'],hashlib.sha256(original).hexdigest())
            source.unlink();return [json.loads(original)]
        self.assertEqual(recorder.archive_chance_before_retirement(retire=retire),[dict(kind='synthetic',value=0)])
        self.assertFalse(source.exists());self.assertIsNone(self.child.poll())

    def test_nonzero_original_child_exit_is_a_failed_recording(self):
        recorder=self.create('nonzero')
        for index in range(5):recorder.record(self.spec(index))
        with self.assertRaisesRegex(ValueError,'nonzero: 7'):
            recorder.finish(expected_command_ids=[x['command_id'] for x in recorder.records])
        manifest=json.loads((recorder.artifacts/'recording.json').read_text());self.assertEqual(manifest['status'],'FAILED_RAW_RECORDING')
        self.assertFalse(manifest['failure']['child_stopped_by_recorder'])

    def test_missing_completion_uses_finite_deadline_and_does_not_kill_child(self):
        recorder=self.create('no-completion',seconds=2.5)
        with self.assertRaisesRegex(TimeoutError,'wall deadline'):recorder.record(self.spec(0))
        self.assertIsNone(self.child.poll());self.assertFalse(recorder.failed['child_stopped_by_recorder'])
        self.assertTrue((recorder.artifacts/'command-0/stdin.jsonl').exists())

    def test_late_unaccounted_stdout_is_not_an_implicit_success(self):
        recorder=self.create('late-tail')
        with self.assertRaisesRegex(ValueError,'stdout frontier differs|unaccounted'):
            self.complete(recorder)
        self.assertFalse(recorder.failed['child_stopped_by_recorder'])


if __name__=='__main__':unittest.main()
