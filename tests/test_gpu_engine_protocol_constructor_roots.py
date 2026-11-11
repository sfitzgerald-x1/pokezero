"""ProtocolConstructor's incremental public roots (_capture, public_root, drop_roots) and the live search's eviction and reset of
them, with a fake RootSession and capture_root (no engine, catalog or Showdown)."""
from collections import OrderedDict
import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gpu_engine import depthn_live_search as live
from gpu_engine import phase_d_protocol_constructor as pc


class FakeSession:
    made = []

    def __init__(self, *, config, contract, battle_id, owner, stream_id):
        self.key, self.lines, self.open, self.fail = (battle_id, owner, stream_id), (), True, False;FakeSession.made.append(self)

    def extends(self, lines):
        return self.open and len(lines) > len(self.lines) and tuple(lines[:len(self.lines)]) == self.lines

    def root(self, *, contract, lines, address, battle_id, seed):
        if self.fail:
            self.close();raise ValueError('boom')
        self.lines = tuple(lines);return ('held', self.key, address.frame, lines[-1])

    def close(self):
        self.open = False


def fresh(*, lines, address, battle_id, seed, **_):
    return ('fresh', battle_id, address.frame, tuple(lines)[-1])


def request(battle='b1', index=0, lines=('|start', '|request|0'), stream=1, owner='p1'):
    return dict(battle_id=battle, owner=owner, stream_id=stream, decision_index=index, lines=list(lines))


def address(c, r):
    lines = tuple(r['lines']);prefix = hashlib.sha256(''.join(x + '\n' for x in lines).encode()).hexdigest()
    return (c.rung, r['battle_id'], r['owner'], r['stream_id']), r['decision_index'], prefix


class IncrementalRootTests(unittest.TestCase):
    def setUp(self):
        FakeSession.made = []
        self.c = object.__new__(pc.ProtocolConstructor)
        self.c.__dict__.update(incremental_roots=True, root_sessions={}, roots=OrderedDict(), owning_config=object(), seed=7, rung='DN2W16')
        for target, value in (('RootSession', FakeSession), ('capture_root', Mock(side_effect=fresh))):
            p = patch.object(pc, target, value);p.start();self.addCleanup(p.stop)

    def capture(self, r):
        key, index, prefix = address(self.c, r);return self.c._capture(key, (key, index, prefix), r, tuple(r['lines']), 0)

    def test_off_every_root_is_capture_root_s_and_nothing_is_held(self):
        self.c.incremental_roots = False
        self.assertEqual(self.capture(request())[0], 'fresh');self.assertEqual((self.c.roots, self.c.root_sessions, FakeSession.made), ({}, {}, []))

    def test_a_session_serves_successors_and_a_non_successor_starts_a_new_one(self):
        first = request();second = request(index=1, lines=('|start', '|request|0', '|turn|2', '|request|1'))
        self.capture(first);self.capture(second);self.assertEqual(len(FakeSession.made), 1)
        self.capture(request(index=2));self.assertEqual(len(FakeSession.made), 2);self.assertFalse(FakeSession.made[0].open)
        self.assertEqual(len(self.c.roots), 3)

    def test_an_error_drops_the_session_and_propagates(self):
        r = request();self.capture(r);FakeSession.made[0].fail = True
        with self.assertRaisesRegex(ValueError, 'boom'):self.capture(request(index=1, lines=('|start', '|request|0', '|request|1')))
        self.assertEqual(self.c.root_sessions, {})

    def test_public_root_returns_the_held_root_only_for_its_exact_integer_address(self):
        r = request();held = self.capture(r);pc.capture_root.reset_mock()
        self.assertIs(self.c.public_root(r), held);pc.capture_root.assert_not_called()
        self.assertEqual(self.c.public_root(request(index=1))[0], 'fresh')
        self.assertEqual(self.c.public_root(request(stream=True))[0], 'fresh')   # True == 1 must not reach the held root
        self.assertEqual(self.c.public_root(dict(r, decision_index=False))[0], 'fresh')

    def test_held_roots_are_capped_most_recent_first(self):
        for i in range(pc.HELD_ROOTS + 6):self.capture(request(battle=f'b{i}'))
        self.assertEqual(len(self.c.roots), pc.HELD_ROOTS);self.assertNotIn('b0', {a[0][1] for a in self.c.roots})

    def test_drop_roots_closes_the_battles_sessions_and_forgets_their_roots(self):
        self.capture(request('b1'));self.capture(request('b2'))
        self.c.drop_roots({'b1'})
        self.assertEqual({k[1] for k in self.c.root_sessions}, {'b2'});self.assertEqual({a[0][1] for a in self.c.roots}, {'b2'})
        self.assertFalse(FakeSession.made[0].open);self.assertTrue(FakeSession.made[1].open)
        self.c.drop_roots();self.assertEqual((self.c.root_sessions, dict(self.c.roots)), ({}, {}));self.assertFalse(FakeSession.made[1].open)


class SearchEvictionTests(unittest.TestCase):
    def test_eviction_and_reset_drop_the_incremental_roots_too(self):
        constructor = SimpleNamespace(sessions={('R', 'b0', 'p1', 1): 0, ('R', 'b2', 'p1', 1): 0}, seen={(('R', 'b0', 'p1', 1), 0, 'x')}, drop_roots=Mock())
        search = SimpleNamespace(battles=OrderedDict((f'b{i}', None) for i in range(3)), session_cap=1, constructor=constructor, evicted=0)
        live.DepthNLiveSearch._evict(search)
        constructor.drop_roots.assert_called_once_with({'b0', 'b1'});self.assertEqual(list(constructor.sessions), [('R', 'b2', 'p1', 1)])
        live.DepthNLiveSearch.reset_sessions(search);constructor.drop_roots.assert_called_with()


if __name__ == '__main__':
    unittest.main()
