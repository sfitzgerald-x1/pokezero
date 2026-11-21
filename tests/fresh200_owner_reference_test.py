"""Synthetic provider/order guards; no Native, Python/B2 runtime or corpus run."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from gpu_engine import fresh200_owner_reference as reference

SHAPES = dict(categorical_ids=(23,41), numeric_features=(23,132), token_type_ids=(23,),
    attention_mask=(23,), legal_action_mask=(9,))
BINDINGS = dict(producer_commit='a'*40, runtime_sha256='b'*64, native_source_registry_hash='c'*64)


def request(*, wait=False, rqid=1, owner='p1'):
    packet = dict(side=dict(id=owner), rqid=rqid)
    if wait:packet['wait'] = True
    else:packet['active'] = [dict(moves=[dict(id='literal')])]
    return '|request|'+json.dumps(packet, separators=(',', ':'))


def arrays(device):
    result = {}
    for name, shape in SHAPES.items():
        dtype = np.bool_ if name.endswith('mask') else np.float32 if name == 'numeric_features' and device else np.float64 if name == 'numeric_features' else np.int32
        result[name] = np.full(shape, True if name.endswith('mask') else 0.1 if name == 'numeric_features' else 0, dtype=dtype)
    return result


class Contract:
    @staticmethod
    def champion():return SimpleNamespace(namespace='synthetic-test-only', categorical_vocab_size=916, token_type_vocab_size=16)


@dataclass(frozen=True)
class Snapshot:
    namespace: str
    battle_id: str
    owner: str
    seed: int
    stream_id: int
    generation: int
    lines: tuple
    prefix_sha256: str
    request_count: int
    own_request_line: str


class Python:
    def __init__(self, contract, **kwargs):
        self.contract = contract;self.calls = [];self.seen = [];self.encoded = [];self.values = arrays(False)
        self.closed = False;self.fail_close = False
    def ingest(self, lines):self.calls.append(tuple(lines));self.seen.extend(lines)
    def encode(self, snapshot):
        assert tuple(self.seen) == snapshot.lines
        self.encoded.append(snapshot);return SimpleNamespace(**self.values), (1,2,3,4,5)
    def close(self):
        self.closed = True
        if self.fail_close:raise RuntimeError('synthetic close failure')


class B2(Python):
    def __init__(self, contract, **kwargs):
        super().__init__(contract, **kwargs);self.values = arrays(True);self.adapter = SimpleNamespace(own_lines=None)
    def encode(self, snapshot):
        assert self.adapter.own_lines == snapshot.lines
        _, targets = super().encode(snapshot)
        return self.values, targets


@contextmanager
def literal_history(adapter, lines):
    adapter.own_lines = tuple(lines)
    try:yield adapter
    finally:adapter.own_lines = None


def validate_lines(lines, owner):
    # This double checks preflight invocation/order. Production loads the
    # existing owning_python_reference._lines privacy/identity validator.
    requests = [x for x in lines if x.startswith('|request|')]
    if not requests:raise ValueError('actual request required')
    for line in requests:
        value = json.loads(line[len('|request|'):])
        if type(value) is not dict or value.get('side', {}).get('id') != owner:
            raise ValueError('foreign or null request')
    if '|-damage|p2a: peer|12/110' in lines:raise ValueError('exact peer HP')
    return tuple(lines), requests


class ReferenceGuards(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void').start()
        self.runtime = SimpleNamespace(ActorContract=Contract, OwnerSnapshot=Snapshot, PythonBackend=Python,
            JaxBackend=B2, literal_history=literal_history, validate_lines=validate_lines,
            terminal=lambda line:line == '|tie' or line.startswith('|win|'), shapes=SHAPES)
        self.binding_mock = patch.object(reference, '_runtime_bindings', return_value=self.runtime).start()

    def session(self):
        value = reference.OwnerReferenceSession(None, None, battle_id='synthetic', owner='p1', seed=7,
            stream_id=101, bindings=BINDINGS);self.addCleanup(value.close);return value

    def test_import_is_independent_of_engine_jax_torch_and_pokezero(self):
        script = '''
import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('jax','torch','pokezero'):
   raise AssertionError('runtime imported: '+fullname)
sys.meta_path.insert(0,Deny())
from gpu_engine import fresh200_owner_reference
'''
        p = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout+p.stderr)

    def test_every_actual_request_flushes_public_segments_in_identical_order(self):
        s = self.session();self.assertEqual(s.ingest(['|start']), ())
        self.assertEqual(s.python.calls, [])
        lines = [request(), '|turn|1', request(wait=True, rqid=2), '|upkeep', request(rqid=3)]
        created = s.ingest(lines)
        self.assertEqual([x.cut.generation for x in created], [1, 3])
        expected = [('|start',), (lines[0],), ('|turn|1',), (lines[2],), ('|upkeep',), (lines[4],)]
        self.assertEqual(s.python.calls, expected);self.assertEqual(s.b2.calls, expected)
        self.assertEqual(s.python.seen, ['|start', *lines]);self.assertEqual(s.b2.seen, s.python.seen)
        self.assertEqual(s.endpoint().cut.request_line, lines[-1]);self.assertIsNone(s.b2.adapter.own_lines)

    def test_wait_invalidates_endpoint_without_encoding_or_a_peer_reference(self):
        s = self.session();s.ingest([request()]);s.ingest([request(wait=True, rqid=2)])
        self.assertEqual(len(s.python.encoded), 1);self.assertEqual(len(s.b2.encoded), 1)
        self.assertEqual(s.history.generation, 2)
        with self.assertRaisesRegex(ValueError, 'actionable'):s.endpoint()
        s.ingest([request(rqid=3)]);self.assertEqual(s.endpoint().cut.generation, 3)

    def test_public_tail_requires_explicit_refresh_and_unchanged_cut_is_reused(self):
        s = self.session();s.ingest([request()]);first = s.endpoint()
        self.assertIs(s.reencode_current(), first);self.assertEqual(len(s.python.encoded), 1)
        s.ingest(['|turn|1'])
        with self.assertRaisesRegex(ValueError, 'stale'):s.endpoint()
        refreshed = s.reencode_current();self.assertEqual(refreshed.cut.generation, first.cut.generation)
        self.assertNotEqual(refreshed.cut.prefix_sha256, first.cut.prefix_sha256)
        self.assertEqual(refreshed.cut.lines[-1], '|turn|1');self.assertEqual(len(s.python.encoded), 2)
        self.assertIs(s.endpoint(), refreshed)

    def test_cpu_and_source_identity_are_required_before_runtime_construction(self):
        kwargs = dict(battle_id='synthetic', owner='p1', seed=7, stream_id=101, bindings=BINDINGS)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'):
            with self.assertRaisesRegex(ValueError, 'CPU-only'):reference.OwnerReferenceSession(None, None, **kwargs)
            with self.assertRaisesRegex(ValueError, 'CPU-only'):reference.compile_catalog(None, None, None)
        for bindings in ({}, dict(BINDINGS, producer_commit='x'*40)):
            with self.assertRaisesRegex(ValueError, 'bindings'):
                reference.OwnerReferenceSession(None, None, **dict(kwargs, bindings=bindings))
        self.binding_mock.assert_not_called()

    def test_frozen_tensors_preserve_python_binary64_and_b2_float32(self):
        s = self.session();s.ingest([request()]);pair = s.endpoint()
        self.assertEqual(pair.python['numeric_features'].dtype, np.dtype('float64'))
        self.assertEqual(pair.b2['numeric_features'].dtype, np.dtype('float32'))
        saved = pair.python['numeric_features'].tobytes();s.python.values['numeric_features'][:] = 99
        self.assertEqual(pair.python['numeric_features'].tobytes(), saved)
        with self.assertRaises(ValueError):pair.python['numeric_features'].flags.writeable = True
        with self.assertRaises(TypeError):pair.python['numeric_features'] = np.zeros((23,132))

    def test_foreign_request_or_exact_peer_hp_poison_before_backend_mutation(self):
        for bad in (request(owner='p2'), '|-damage|p2a: peer|12/110', '|request|null'):
            s = self.session();s.ingest([request()]);previous = tuple(s.python.seen)
            with self.subTest(bad=bad), self.assertRaises(ValueError):s.ingest([bad])
            self.assertEqual(tuple(s.python.seen), previous);self.assertEqual(tuple(s.b2.seen), previous)
            self.assertEqual(s.status, 'POISONED');self.assertEqual(len(s.records), 1)
            with self.assertRaisesRegex(ValueError, 'POISONED'):s.endpoint()

    def test_failed_backend_output_cannot_publish_partial_reference(self):
        s = self.session();s.b2.values['numeric_features'] = np.zeros((23,132), np.float64)
        with self.assertRaisesRegex(ValueError, 'canonical B2'):s.ingest([request()])
        self.assertEqual(s.status, 'POISONED');self.assertEqual(s.records, [])
        self.assertIsNone(s.b2.adapter.own_lines)
        with self.assertRaisesRegex(ValueError, 'POISONED'):s.reencode_current()

    def test_sealer_failure_can_retry_without_reencoding_or_replaying(self):
        s = self.session();s.ingest([request()]);pair = s.endpoint();seen = []
        def broken(value, identity):
            seen.append(value);raise OSError('synthetic artifact writer failure')
        with self.assertRaises(OSError):s.seal_current(broken)
        self.assertEqual(s.status, 'ACTIVE');self.assertIs(s.endpoint(), pair)
        expected = dict(path='synthetic-only.json', bytes=2, sha256='d'*64)
        def seal(value, identity):
            seen.append(value);self.assertFalse(identity['actual_device_observations_loaded']);return expected
        self.assertEqual(s.seal_current(seal), expected)
        self.assertEqual(s.seal_current(lambda *args:self.fail('must reuse sealed artifact')), expected)
        self.assertEqual(len(s.python.encoded), 1);self.assertEqual(len(s.b2.encoded), 1)
        self.assertEqual(len(seen), 2)

    def test_terminal_lifecycle_invalidates_endpoint_and_close_attempts_both_backends(self):
        s = self.session();s.ingest([request(), '|win|literal'])
        self.assertTrue(s.terminal);self.assertEqual(len(s.python.encoded), 1)
        with self.assertRaisesRegex(ValueError, 'terminal'):s.endpoint()
        s.python.fail_close = True
        with self.assertRaisesRegex(RuntimeError, 'close failure'):s.close()
        self.assertTrue(s.python.closed and s.b2.closed);self.assertEqual(s.status, 'CLOSED')


if __name__ == '__main__':unittest.main()
