"""Exercise the pre-submit artifact binding with a synthetic bridge."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from gpu_engine.fresh200_hook_admission import HookPolicy, seal_parsed_catalog
from gpu_engine.fresh200_native_consumer import ClosedCommandVerifier, CommandSpec
from gpu_engine.fresh200_native_recording import NativeCommandRecorder
from gpu_engine.fresh200_witness_audit import digest


class CheckedStdin:
    def __init__(self, stdin):
        self.stdin = stdin
        self.check = None

    def write(self, data):
        if self.check is not None:
            self.check()
        return self.stdin.write(data)

    def flush(self):
        self.stdin.flush()

    def close(self):
        self.stdin.close()


class RecordingInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.fixture = Path(directory.name)
        subprocess.run(
            [os.environ['FRESH200_TEST_NODE'], 'tests/fresh200_consumer_fixture.cjs',
             str(cls.fixture), '--registration-policy'],
            check=True, timeout=15,
        )
        cls.manifest = json.loads((cls.fixture / 'manifest.json').read_bytes())
        declaration = cls.manifest['registration_catalog']
        catalog = seal_parsed_catalog(
            declaration['source_pins'], declaration['functions'],
            parser_sha256=digest(b'synthetic source declaration'),
        )
        cls.policy = HookPolicy(catalog, tuple(declaration['bootstrap_operations']), ())

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bridge = self.root / 'bridge'
        self.bridge.mkdir()
        read_fd, write_fd = os.pipe()
        completion = os.fdopen(read_fd, 'rb', buffering=0)
        self.addCleanup(completion.close)
        try:
            child = subprocess.Popen(
                [sys.executable, 'tests/fresh200_recording_child.py', str(self.fixture),
                 str(self.bridge), 'normal', str(write_fd)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                pass_fds=(write_fd,),
            )
        finally:
            os.close(write_fd)
        child.stdin = self.stdin = CheckedStdin(child.stdin)
        self.addCleanup(self._close_child, child)
        manifest = self.manifest
        verifier = ClosedCommandVerifier(
            session=manifest['session'], battle_id=manifest['battle_id'],
            source_registry_hash=manifest['source_registry_hash'],
            expected_methods=manifest['expected_methods'],
            call_operations=self.policy.operations, hook_policy=self.policy,
        )
        registry = json.dumps(dict(
            registry=dict(synthetic=True), sourceRegistryHash=manifest['source_registry_hash'],
        ), separators=(',', ':')).encode() + b'\n'
        self.recorder = NativeCommandRecorder(
            directory=self.bridge, artifacts=self.root / 'artifacts', child=child,
            completion=completion, verifier=verifier, expected_registry_wire=registry,
            deadline_monotonic=time.monotonic() + 30,
        )

    def _close_child(self, child):
        child.stdin.close()
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=3)
        if hasattr(self, 'recorder'):
            for capture in self.recorder._captures.values():
                capture.thread.join(timeout=3)
        child.stdout.close()
        child.stderr.close()

    def spec(self, index):
        command = self.manifest['commands'][index]
        values = [(self.fixture / command[key]['path']).read_bytes()
                  for key in ('stdin', 'native_control', 'transport_control')]
        return CommandSpec(*values, 'b', hook_admission=self.recorder.verifier.hook_cursor.binding(self.policy))

    def test_admission_and_exact_chance_bytes_are_durable_before_submission(self):
        initial = b'{"initial_tie":0, "genders":[[0,0,0,0,0,0],[0,0,0,0,0,0]],"allow_inert_gender":false}'
        (self.bridge / 'initializing.json').write_bytes(initial)
        spec = self.spec(0)

        def check():
            manifest = json.loads((self.recorder.artifacts / 'recording.json').read_bytes())
            record = manifest['commands'][0]
            self.assertEqual(record['status'], 'SEALED_BEFORE_STDIN')
            self.assertEqual(record['inputs']['initializing.json']['sha256'], digest(initial))
            admission = Path(record['inputs']['hook-admission.json']['path']).read_bytes()
            self.assertEqual(json.loads(admission), vars(spec.hook_admission))
            for name, expected in [('hook-policy.json', self.policy.sha256), ('hook-catalog.json', self.policy.catalog.sha256)]:
                artifact = manifest['admission_files'][name]
                self.assertEqual(digest(Path(artifact['path']).read_bytes()), expected)

        self.stdin.check = check
        self.recorder.record(spec, chance_control_wire=initial)
        self.assertEqual((self.recorder.artifacts / 'command-0/initializing.json').read_bytes(), initial)

    def test_unsealed_controls_never_reach_stdin(self):
        initial = b'{"initial_tie":0}'
        (self.bridge / 'initializing.json').write_bytes(initial)
        self.stdin.check = lambda: self.fail('original stdin must not be written')
        with self.assertRaisesRegex(ValueError, 'requires independently sealed bytes'):
            self.recorder.record(self.spec(0))
        self.assertEqual(self.recorder.records, [])

    def test_resident_chance_bytes_must_match_before_submission(self):
        (self.bridge / 'initializing.json').write_bytes(b'{"initial_tie":0}')
        self.stdin.check = lambda: self.fail('original stdin must not be written')
        with self.assertRaisesRegex(ValueError, 'resident chance control differs'):
            self.recorder.record(self.spec(0), chance_control_wire=b'{"initial_tie":1}')
        self.assertEqual(self.recorder.records, [])

    def test_literal_chance_nonce_is_required_even_when_resident_bytes_match(self):
        self.recorder.record(self.spec(0))
        packet = b'{"nonce":"incorrect"}'
        (self.bridge / 'chance.json').write_bytes(packet)
        self.stdin.check = lambda: self.fail('original stdin must not be written')
        with self.assertRaisesRegex(ValueError, 'chance control nonce differs'):
            self.recorder.record(self.spec(1), chance_control_wire=packet)
        self.assertEqual(len(self.recorder.records), 1)

    def test_finished_recording_binds_causal_and_transport_files(self):
        for i in range(5):
            self.recorder.record(self.spec(i))
        result = self.recorder.finish(expected_command_ids=[r['command_id'] for r in self.recorder.records])
        for name in ('witness-audit.jsonl', 'witness-transport-events.jsonl', 'witness-receipts.jsonl',
                     'witness-transport-receipts.jsonl', 'witness-source-registry.json'):
            self.assertEqual(result['files'][name]['sha256'], digest((self.bridge / name).read_bytes()))


if __name__ == '__main__':
    unittest.main()
