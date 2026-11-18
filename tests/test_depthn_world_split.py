"""The measurement adapter dispatches the seam and cannot silently run serially."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import unittest
from unittest.mock import patch

from gpu_engine import constructor_worlds as cw, phase_d_protocol_world as pw
from gpu_engine import prepare_pool as pp, world_split as ws
from tests import test_depthn_world_range as fixtures


class WorldSplit(unittest.TestCase):
    def setUp(self):
        self.original = pw.world_range
        self.installed = dict(ws._installed)
        self.fixture = fixtures.WorldRanges()
        self.con, self.prepared = self.fixture.fixture()

    def tearDown(self):
        pw.world_range = self.original
        ws._installed.clear();ws._installed.update(self.installed)
        ws._local.backend = None

    def dispatch(self, blobs):
        self.ranges.extend(pp.pickle.loads(blob)[4:6] for blob in blobs)
        # Helpers have a separate thread-local context, just as remote workers do.
        with ThreadPoolExecutor(max_workers=2) as pool:
            return list(pool.map(lambda blob: ws.worker_range(self.con, blob), blobs))

    def test_multiple_ranges_and_timing_through_real_prepare_and_transport(self):
        self.ranges = []
        with self.fixture.patches(), patch.object(self.con, 'prepare_root', return_value=self.prepared), \
                patch.object(pw, 'sample', side_effect=AssertionError('legacy sample hook used')):
            _, serial, legal = self.con.prepare({}, 0)
            for k in (1, 2, 4):
                self.ranges.clear();backend = ws.Backend(k, self.dispatch)
                with ws.split(backend):
                    root, made, got_legal = self.con.prepare({}, 0)
                self.assertIs(root, self.prepared.root)
                self.assertEqual(got_legal, legal)
                self.assertEqual(self.fixture.signature(made), self.fixture.signature(serial))
                self.assertEqual(self.ranges, [(16*i//k, 16*(i+1)//k) for i in range(k)])
                self.assertEqual(backend.parts['k'], k)
                self.assertGreater(backend.parts['bytes'], 0)
                for name in ('ship', 'helpers', 'gather'):
                    self.assertGreater(backend.parts[name], 0)
                # A prior successful request must not mask a subsequent bypass.
                with self.assertRaisesRegex(RuntimeError, 'serial fallback forbidden'), ws.split(backend):
                    pass

    def test_missing_hook_raises(self):
        backend = ws.Backend(2, self.dispatch)
        with patch.object(pw, 'world_range', None), self.assertRaisesRegex(RuntimeError, 'seam hook'):
            with ws.split(backend):pass
        ws.install()
        with patch.object(pw, 'world_range', self.original), self.assertRaisesRegex(RuntimeError, 'missing or replaced'):
            with ws.split(backend):pass

    def test_synchronous_helper_fallback_preserves_results_and_owner_context(self):
        with self.fixture.patches(), patch.object(self.con, 'prepare_root', return_value=self.prepared):
            _, serial, legal = self.con.prepare({}, 0)
            for k in (1, 2, 4):
                bounds = []
                def local(blobs):
                    replies = []
                    for blob in blobs:
                        bounds.append(pp.pickle.loads(blob)[4:6])
                        replies.append(ws.worker_range(self.con, blob))
                        self.assertIs(ws._local.backend, backend)
                    return replies
                backend = ws.Backend(k, local)
                with ws.split(backend):
                    _, made, got_legal = self.con.prepare({}, 0)
                self.assertEqual(self.fixture.signature(made), self.fixture.signature(serial))
                self.assertEqual(got_legal, legal)
                self.assertEqual(bounds, [(16*i//k, 16*(i+1)//k) for i in range(k)])
                self.assertEqual(backend.parts['k'], k)
                self.assertIsNone(ws._local.backend)

    def test_failed_local_helper_restores_owner_context_before_retry(self):
        def local(blobs):
            def fail(*args, **kwargs):
                raise ValueError('helper failed')
            with patch.dict(ws._installed, world_range=fail):
                with self.assertRaisesRegex(ValueError, 'helper failed'):
                    ws.worker_range(self.con, blobs[0])
            self.assertIs(ws._local.backend, backend)
            replies = [ws.worker_range(self.con, blob) for blob in blobs]
            self.assertIs(ws._local.backend, backend)
            return replies
        backend = ws.Backend(2, local)
        with self.fixture.patches(), patch.object(self.con, 'prepare_root', return_value=self.prepared):
            with ws.split(backend):
                _, made, _ = self.con.prepare({}, 0)
        self.assertEqual(len(made.worlds), 16)
        self.assertEqual(backend.parts['k'], 2)
        self.assertIsNone(ws._local.backend)

    def test_bypassed_hook_raises_even_when_serial_prepare_succeeds(self):
        backend = ws.Backend(2, self.dispatch)
        with self.fixture.patches(), patch.object(self.con, 'prepare_root', return_value=self.prepared):
            with self.assertRaisesRegex(RuntimeError, 'serial fallback forbidden'):
                with ws.split(backend), patch.object(pw, 'world_range', self.original):
                    self.con.prepare({}, 0)

    def test_refusal_before_sampling_preserves_its_classification(self):
        backend = ws.Backend(2, lambda blobs: self.fail('a refused root dispatched worlds'))
        item = cw.RootRefusal('request', 'PUBLIC_ROOT_REFUSED', 'refused', 0., None)
        with ws.split(backend):backend.check_result(item)
        self.assertEqual(backend.parts, {})
        self.assertIsNone(ws._local.backend)

    def test_missing_or_misrouted_helper_reply_raises(self):
        self.ranges = []
        with self.fixture.patches(), patch.object(self.con, 'prepare_root', return_value=self.prepared):
            for dispatch, message in ((lambda blobs: [], 'every dispatched range'),
                    (lambda blobs: list(reversed(self.dispatch(blobs))), 'does not match')):
                with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                    with ws.split(ws.Backend(2, dispatch)):self.con.prepare({}, 0)
                self.assertIsNone(ws._local.backend)

    def test_decode_and_assembly_errors_cannot_become_root_refusals(self):
        self.ranges = []
        with self.fixture.patches(), patch.object(self.con, 'prepare_root', return_value=self.prepared), \
                patch.object(cw, 'constructor_binding', return_value='binding'):
            for problem in ('missing-seed', 'missing-world', 'bad-tuple'):
                def malformed(blobs):
                    replies = self.dispatch(blobs)
                    header, shells, packed = pp.pickle.loads(replies[0])
                    if problem == 'missing-seed':header = replace(header, seeds=())
                    elif problem == 'missing-world':shells = shells[:-1]
                    replies[0] = pp.dumps((header,) if problem == 'bad-tuple' else (header, shells, packed))
                    return replies
                backend = ws.Backend(2, malformed)
                with self.subTest(problem=problem), self.assertRaisesRegex(RuntimeError, 'dispatch/gather failed'):
                    with ws.split(backend):
                        item = cw.prepare_roots(self.con, [{}], device='cpu')[0]
                        backend.check_result(item)
                self.assertEqual(backend.parts, {})
                self.assertIsNone(ws._local.backend)

    def _assert_stale_replies_rejected(self, k):
        self.ranges = []
        def stale(blobs):
            previous = []
            for blob in blobs:
                root, *rest = pp.pickle.loads(blob)
                root.owner_lines_sha256 = 'previous-request'
                previous.append(pp.dumps((root, *rest)))
            return self.dispatch(previous)
        backend = ws.Backend(k, stale)
        with self.fixture.patches(), patch.object(self.con, 'prepare_root', return_value=self.prepared), \
                patch.object(pp, 'unpack_worlds', side_effect=AssertionError('stale worlds unpacked')):
            with self.assertRaisesRegex(RuntimeError, 'does not match its dispatched root'):
                with ws.split(backend):self.con.prepare({}, 0)
        self.assertEqual(backend.parts, {})
        self.assertIsNone(ws._local.backend)

    def test_single_stale_helper_reply_raises_before_unpack(self):
        self._assert_stale_replies_rejected(1)

    def test_complete_stale_reply_set_raises_before_unpack(self):
        self._assert_stale_replies_rejected(2)


if __name__ == '__main__':unittest.main()
