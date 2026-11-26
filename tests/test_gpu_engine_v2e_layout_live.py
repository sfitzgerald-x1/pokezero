"""Pod test: the committed layout description (tests/fixtures/v2e_layout_description.json) against the live oracle
pytrees, and the header's digests against live_digests. Needs the real tables: set V2E_CONFIG and V2E_REFERENCE_RUN
(paths are environment arguments, never built in); skipped otherwise. Fails on any added, removed, retyped or reshaped
leaf, so a stale header cannot bind a different number of operands."""
import os
from types import SimpleNamespace
import unittest

from gpu_engine import v2e_layout as L

CONFIG = os.environ.get('V2E_CONFIG');RUN = os.environ.get('V2E_REFERENCE_RUN')


@unittest.skipUnless(CONFIG and RUN, 'V2E_CONFIG and V2E_REFERENCE_RUN not set (pod-only test)')
class LiveDescriptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gpu_engine import v2e_census as cz
        cls.ctx = cz.load_context(SimpleNamespace(config=CONFIG, reference_run=RUN, arm='B'))

    def test_fixture_matches_the_live_oracle(self):
        state, actions, trackers = self.ctx['template'];belief = getattr(self.ctx['observation_tables'], 'belief', None)
        description = L.load_description(L.DEFAULT_DESCRIPTION)
        self.assertEqual(L.check_live(description, state, actions, trackers, self.ctx['engine_tables'], belief), [])
        live = L.live_description(state, actions, trackers, self.ctx['engine_tables'], belief)
        for key in ('state', 'actions', 'trackers', 'tables'):self.assertEqual(live[key], description[key], key)
        full = L.description_from_context(self.ctx)
        self.assertEqual(full['step_output'], description['step_output'])

    def test_header_digests_equal_live_digests(self):
        state, actions, trackers = self.ctx['template'];belief = getattr(self.ctx['observation_tables'], 'belief', None)
        spec = L.load_spec(L.DEFAULT_SPEC) if L.DEFAULT_SPEC.exists() else None
        live = L.live_digests(state, actions, trackers, self.ctx['engine_tables'], belief, spec);text = L.HEADER.read_text()
        for name, (d64, hexdigest) in live.items():
            self.assertIn(hexdigest, text, name)
        self.assertIn(f'constexpr uint64_t kLayoutDigest = 0x{live["kLayoutDigest"][0]:016x}ull;', text)

    def test_pack_tables_matches_the_header(self):
        description = L.load_description(L.DEFAULT_DESCRIPTION);belief = getattr(self.ctx['observation_tables'], 'belief', None)
        buffer = L.pack_tables(description, self.ctx['engine_tables'], belief)
        rows, total = L.table_layout(description['tables']);self.assertEqual(buffer.shape, (total,))
        import numpy as np
        from gpu_engine import mechanics
        e, offset, nbytes = rows[-1];counts = np.asarray(mechanics._ACCURACY_COUNTS, np.int32)
        self.assertTrue(np.array_equal(buffer[offset:offset + nbytes].view(np.int32).reshape(counts.shape), counts))


if __name__ == '__main__':unittest.main()
