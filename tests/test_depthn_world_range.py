"""Range partitions preserve global seeds, keys, refusals and completion order."""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import hashlib
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import phase_d_protocol_world as pw, phase_d_protocol_constructor as pc
from gpu_engine import v5_public_world
from pokezero.v5_contract import DecisionAddress


class WorldRanges(unittest.TestCase):
    def test_root_recapture_reuses_the_constructor_actor_contract(self):
        constructor, prepared = self.fixture()
        constructor.actor_contract = NS(feature_masks=(('set_posterior_narrowing', True),))
        request = dict(lines=prepared.lines, stream_id=5, decision_index=7, owner='p1', battle_id='range-test')
        with patch.object(pc, 'capture_root', return_value=prepared.public) as capture:
            self.assertIs(constructor.public_root(request), prepared.public)
        self.assertIs(capture.call_args.kwargs['contract'], constructor.actor_contract)

    def fixture(self, count=16):
        address = DecisionAddress(5, 7, 0)
        public = NS(context=NS(battle_id='range-test'), owner_lines_sha256='wire', address=address)
        root = NS(address=address, source_binding_sha256='bound')
        con = pc.ProtocolConstructor(catalog='catalog', source=NS(binding_sha256='bound', original='source'), metadata={},
            owning_config=None, count=count, functions=NS(catalog='catalog'), seed=17)
        prepared = pc.PreparedRoot(public, root, ('|turn|3',), (True, False), count, 17)
        return con, prepared

    def sampler(self, **kw):
        draw = kw['rng'].randrange(2**31)
        return (None, 'named-refusal') if draw % 5 == 0 else (draw, None)

    def materialize(self, root, start, *, seed, host_key, **kw):
        state = (np.asarray([start, seed], np.uint64), np.asarray(host_key))
        return pw.ProtocolWorld(state, np.asarray([seed], np.uint32), (), mapper_diagnostics=({'kind': 'test'},))

    def signature(self, made):
        return (made.seeds, made.mapper_diagnostics, tuple((w.index, w.kernel_seed, w.refusals,
            None if w.state is None else tuple((x.dtype.str, x.shape, x.tobytes()) for x in (*w.state, w.public))) for w in made.worlds))

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(pw, '_gen3_randbat_belief_start_override_result', side_effect=self.sampler))
        stack.enter_context(patch.object(v5_public_world, 'materialize', side_effect=self.materialize))
        return stack

    def test_partitions_match_serial_and_budget_prefix_including_empty_and_uneven_ranges(self):
        results = {}
        with self.patches():
            for count in (16, 32, 64):
                con, prepared = self.fixture(count)
                with patch.object(pw, '_carried_root', return_value=prepared.root):
                    serial = pw.sample(prepared.public, prepared.root, lines=prepared.lines, catalog=con.catalog, source=con.source,
                        metadata=con.metadata, config=None, battle_id='range-test', seed=con.seed, count=count)
                for boundaries in ((0, count), (0, count//2, count), (0, 1, 7, count-1, count), (0, 0, count)):
                    parts = [con.sample_range(prepared, a, b) for a, b in zip(boundaries, boundaries[1:])]
                    self.assertEqual(self.signature(serial), self.signature(pw.assemble(reversed(parts), count=count)))
                results[count] = serial
        for count in (16, 32):
            full = results[64]
            self.assertEqual(self.signature(results[count]), self.signature(replace(full, worlds=full.worlds[:count],
                seeds=full.seeds[:count], mapper_diagnostics=full.mapper_diagnostics[:count])))

    def test_completion_runs_on_each_constructed_world_before_gather(self):
        con, prepared = self.fixture();seen = []
        def complete(world):
            seen.append(world.index)
            return replace(world, public=world.public + np.uint32(1))
        with self.patches():
            raw = con.sample_range(prepared, 0, 16)
            parts = [con.sample_range(prepared, a, b, complete=complete) for a, b in ((0, 5), (5, 16))]
        made = pw.assemble(parts, count=16)
        self.assertEqual(seen, [w.index for w in raw.worlds if w.state is not None])
        self.assertEqual(raw.seeds, made.seeds)
        for a, b in zip(raw.worlds, made.worlds):
            self.assertEqual(a.refusals, b.refusals)
            if a.state is not None:np.testing.assert_array_equal(a.public + np.uint32(1), b.public)

    def test_bad_ranges_and_mixed_results_fail_closed(self):
        con, prepared = self.fixture()
        with self.patches():
            for a, b in ((-1, 2), (3, 2), (0, 17), (True, 3), (0, 2.0)):
                with self.subTest(a=a,b=b), self.assertRaises(ValueError):con.sample_range(prepared, a, b)
            left, right = con.sample_range(prepared, 0, 8), con.sample_range(prepared, 8, 16)
            for parts in ((left,), (left, left), (replace(left, start=True), right),
                          (left, replace(right, binding=('different',))), (left, replace(right, requested_worlds=32)),
                          (left, replace(right, seeds=())), (left, replace(right, worlds=tuple(reversed(right.worlds)))),
                          (left, replace(right, mapper_diagnostics=()))):
                with self.subTest(parts=repr(parts)[:60]), self.assertRaises(ValueError):pw.assemble(parts, count=16)
            for bad in (replace(prepared, seed=18), replace(prepared, count=32), object()):
                with self.assertRaises(ValueError):con.sample_range(bad, 0, 8)

    def test_prepare_root_folds_once_and_ranges_do_not_touch_sessions(self):
        con, prepared = self.fixture();events = [];checked = [False]
        class Adapter:
            @contextmanager
            def checked_catalog(self):
                self_test.assertFalse(checked[0]);checked[0] = True
                try:yield
                finally:checked[0] = False
            def reset(self, **kw):self_test.assertTrue(checked[0]);events.append('reset')
            def feed(self, lines):self_test.assertTrue(checked[0]);events.append('feed')
            def encode(self, *a, **kw):
                self_test.assertTrue(checked[0]);events.append('encode');return {'legal_action_mask': prepared.legal}
        self_test = self
        lines = ('|turn|3', '|request|p1|{}')
        request = dict(schema=pc.SCHEMA, request_id='r', battle_id='range-test', owner='p1', stream_id=5,
            decision_index=7, generation=1, prefix_sha256=hashlib.sha256(''.join(x+'\n' for x in lines).encode()).hexdigest(),
            lines=lines, own_request_line=lines[-1])
        with self.patches(), patch.object(pc, 'JaxRecordedAdapter', return_value=Adapter()), \
                patch.object(con, '_capture', return_value=prepared.public) as capture, \
                patch.object(pc, '_packet', return_value={}), patch.object(pw.tail, 'facts', return_value=None), \
                patch.object(pc.worlds, 'from_b2', return_value=prepared.root), \
                patch.object(pw, '_carried_root', return_value=prepared.root) as carry:
            got = con.prepare_root(request)
            con.sample_range(got, 0, 8);con.sample_range(got, 8, 16)
            self.assertEqual(events, ['reset', 'feed', 'encode']);capture.assert_called_once();carry.assert_called_once()
            with self.assertRaisesRegex(ValueError, 'SEARCH_REQUEST_ALREADY_ATTEMPTED'):con.prepare_root(request)


if __name__ == '__main__':unittest.main()
