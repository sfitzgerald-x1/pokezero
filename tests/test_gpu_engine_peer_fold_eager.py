"""constructor_worlds' eager peer fold (advance_peer_fold, _peer_fold) with a fake B2 adapter (no engine, catalog or Showdown):
the same peer wire lines are folded, only split across more feeds; an error drops the fold and is not raised; prepare_roots
advances the fold only with the redaction on and the constructor's eager_peer_fold."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gpu_engine import constructor_worlds as cw

LINES = ('|start', '|switch|p1a: Absol|Absol, L87, F|255/255', '|-ability|p1a: Absol|Pressure|[silent]', '|move|p2a: Raikou|Thunderbolt|p1a: Absol',
         '|turn|2', '|request|{"active":[]}', '|switch|p2a: Gligar|Gligar, L80|100/100', '|turn|3', '|request|{"active":[]}')


class FakeAdapter:
    made, fail_next = [], False

    def __init__(self, catalog, *, owner_side, compiled_functions):
        self.fed, self.fail = [], FakeAdapter.fail_next;FakeAdapter.fail_next = False;FakeAdapter.made.append(self)

    def reset(self, *, stream_id, battle_id):
        self.reset_to = (stream_id, battle_id)

    def feed(self, lines):
        if self.fail:raise ValueError('JAX public fold refused boundary')
        if getattr(self, 'partway', False):   # a failure after part of the feed (B2's feed commits nothing then; the fake keeps the part)
            self.fed.append(tuple(lines[:1]));raise ValueError('JAX public fold refused boundary')
        self.fed.append(tuple(lines))


class Stop(Exception):
    pass


def request(n, battle='b1'):
    return dict(battle_id=battle, stream_id=1, lines=list(LINES[:n]))


class EagerPeerFoldTests(unittest.TestCase):
    def setUp(self):
        FakeAdapter.made, FakeAdapter.fail_next = [], False
        self.c = SimpleNamespace(catalog=SimpleNamespace(source=object()), functions=object())
        cw._PEER_FOLDS.clear();self.addCleanup(cw._PEER_FOLDS.clear)
        p = patch('pokezero.recorded_encoder_runner.JaxRecordedAdapter', FakeAdapter);p.start();self.addCleanup(p.stop)

    def test_advancing_folds_the_same_lines_and_the_belief_s_fold_feeds_only_the_new_ones(self):
        self.assertTrue(cw.advance_peer_fold(self.c, request(5), 0));self.assertTrue(cw.advance_peer_fold(self.c, request(6), 0))
        adapter = cw._peer_fold(self.c, request(9), 0, self.c.catalog, self.c.functions)
        self.assertEqual(FakeAdapter.made, [adapter])   # one fold for the battle, never refolded from scratch
        self.assertEqual(sum(adapter.fed, ()), cw.peer_wire_lines(LINES, 0))   # every peer line once, in order (the owner's [silent] line out)
        self.assertEqual(adapter.fed[-1], cw.peer_wire_lines(LINES, 0)[4:])   # the last feed: only the new lines
        self.assertEqual(len(adapter.fed), 2)   # request(6) added only a |request| line: no feed
        cw._peer_fold(self.c, request(9), 0, self.c.catalog, self.c.functions);self.assertEqual(len(adapter.fed), 2)   # nothing new: no feed

    def test_a_fold_that_raises_is_dropped_never_raised_and_the_next_call_folds_from_scratch(self):
        FakeAdapter.fail_next = True
        self.assertFalse(cw.advance_peer_fold(self.c, request(5), 0));self.assertEqual(cw._PEER_FOLDS, {})
        adapter = cw._peer_fold(self.c, request(9), 0, self.c.catalog, self.c.functions)
        self.assertEqual(len(FakeAdapter.made), 2);self.assertEqual(adapter.fed, [cw.peer_wire_lines(LINES, 0)])
        FakeAdapter.fail_next = True   # the lazy path still raises where it always did
        with self.assertRaises(ValueError):cw._peer_fold(self.c, request(9, battle='b2'), 0, self.c.catalog, self.c.functions)

    def test_an_eager_feed_failing_partway_on_the_first_reveal_s_own_decision_leaves_nothing_the_lazy_fold_reuses(self):
        cw.advance_peer_fold(self.c, request(5), 0);clean = cw._peer_fold(self.c, request(9), 0, self.c.catalog, self.c.functions)
        cw._PEER_FOLDS.clear();FakeAdapter.made = []
        cw.advance_peer_fold(self.c, request(5), 0);FakeAdapter.made[0].partway = True
        self.assertFalse(cw.advance_peer_fold(self.c, request(9), 0));self.assertEqual(cw._PEER_FOLDS, {})   # dropped, partial feed and all
        lazy = cw._peer_fold(self.c, request(9), 0, self.c.catalog, self.c.functions)   # the same decision's belief: from scratch
        self.assertIsNot(lazy, FakeAdapter.made[0]);self.assertEqual(lazy.fed, [cw.peer_wire_lines(LINES, 0)])
        self.assertEqual(sum(lazy.fed, ()), sum(clean.fed, ()))   # the same lines as the clean run's fold, none twice

    def test_no_catalog_or_no_lines_folds_nothing(self):
        for c, r in ((SimpleNamespace(catalog=None, functions=None), request(5)), (self.c, dict(battle_id='b1', stream_id=1)), (self.c, None)):
            with self.subTest(r=r):self.assertFalse(cw.advance_peer_fold(c, r, 0))
        self.assertEqual(FakeAdapter.made, [])

    def test_a_non_successor_starts_a_new_fold_as_before(self):
        cw.advance_peer_fold(self.c, request(5), 0);other = dict(request(9), lines=['|start', '|turn|9'])
        cw.advance_peer_fold(self.c, other, 0);self.assertEqual(len(FakeAdapter.made), 2);self.assertEqual(FakeAdapter.made[1].fed, [('|start', '|turn|9')])


class PrepareRootsAdvanceTests(unittest.TestCase):
    def run_prepare(self, *, eager, peer_view_b, redact=False):
        constructor = SimpleNamespace(eager_peer_fold=eager, prepare=Mock(return_value=(SimpleNamespace(address=SimpleNamespace(owner=1)), None, ())))
        r = request(9)
        with patch.object(cw, 'advance_peer_fold') as advance, patch.object(cw, 'constructor_binding', return_value='stamp'), \
                patch.object(cw, 'world_completer', side_effect=Stop):
            with self.assertRaises(Stop):cw.prepare_roots(constructor, [r], device=None, peer_view_b=peer_view_b, redact_private_reveals=redact)
        return advance, constructor, r

    def test_prepare_roots_advances_only_with_the_flag_and_the_redaction(self):
        advance, constructor, r = self.run_prepare(eager=True, peer_view_b=True);advance.assert_called_once_with(constructor, r, 1)
        self.run_prepare(eager=True, peer_view_b=False, redact=True)[0].assert_called_once()   # the redaction alone also reads the fold
        for eager, view_b in ((False, True), (True, False)):
            with self.subTest(eager=eager, view_b=view_b):self.run_prepare(eager=eager, peer_view_b=view_b)[0].assert_not_called()

    def test_a_refused_root_advances_the_fold_too_with_the_request_s_owner(self):
        for eager, called in ((True, True), (False, False)):
            constructor = SimpleNamespace(eager_peer_fold=eager, prepare=Mock(side_effect=ValueError('PUBLIC_SUBSTITUTE_NOT_FRESH_OR_PASSED')))
            r = dict(request(9), owner='p2', request_id='x')
            with self.subTest(eager=eager), patch.object(cw, 'advance_peer_fold') as advance, patch.object(cw, 'constructor_binding', return_value='stamp'):
                out = cw.prepare_roots(constructor, [r], device=None, peer_view_b=True)
                self.assertEqual((type(out[0]).__name__, out[0].detail), ('RootRefusal', 'PUBLIC_SUBSTITUTE_NOT_FRESH_OR_PASSED'))   # the refusal as before
                if called:advance.assert_called_once_with(constructor, r, 1)
                else:advance.assert_not_called()
        for eager in (True, False):   # (review) an owner that is not a seat string, even unhashable, refuses its root alone, as before
            constructor = SimpleNamespace(eager_peer_fold=eager, prepare=Mock(side_effect=ValueError('MALFORMED_SEARCH_REQUEST')))
            for owner in ('p3', ['p1'], {'p': 1}, None):
                with self.subTest(eager=eager, owner=owner), patch.object(cw, 'advance_peer_fold') as advance, \
                        patch.object(cw, 'constructor_binding', return_value='stamp'):
                    out = cw.prepare_roots(constructor, [dict(request(9), owner=owner)], device=None, peer_view_b=True)
                    self.assertEqual((type(out[0]).__name__, out[0].detail), ('RootRefusal', 'MALFORMED_SEARCH_REQUEST'));advance.assert_not_called()

    def test_a_memory_error_in_the_eager_fold_is_fatal_as_everywhere_in_the_live_search(self):
        c = SimpleNamespace(catalog=SimpleNamespace(source=object()), functions=object())
        with patch.object(cw, '_peer_fold', side_effect=MemoryError()), self.assertRaises(MemoryError):cw.advance_peer_fold(c, request(5), 0)
        with patch.object(cw, '_peer_fold', side_effect=RuntimeError('anything else')):self.assertFalse(cw.advance_peer_fold(c, request(5), 0))


if __name__ == '__main__':
    unittest.main()
