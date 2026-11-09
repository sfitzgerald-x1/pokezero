"""gpu_engine.depthn_guard_inputs: the observer seam on the DepthN pricer (bitwise identical frames with it off and on) and
the effect guard record built from a real frame and its observed turn-1 children (the fake engine of the constructor-world
tests; a fake label gather stands in for the BattleState one)."""
from contextlib import ExitStack
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from pokezero import depthn_effect_guard as G
from gpu_engine import constructor_worlds as cw, depthn_guard_inputs as gi, device_depthn_pricer as dn
from tests import test_gpu_engine_constructor_worlds as CW

ROOTS = ((0, 0), (2, 0))   # the fixed roots: CW made(i, owner, CYCLE modes) for (i, owner); W = 8, depth 2, k 2/2
SURF, TACKLE, REST, PROTECT = 2, 1, 3, 4   # the fixture's engine move ids (CW metadata: ['', 'tackle', 'surf', 'rest', 'protect'])


def fake_labels(states, *, children_only=False, peer_moves=(SURF, TACKLE, REST, PROTECT)):
    """The label gather on the fake engine's states [N, 1]: both seats' active slot 0, species Alpha (seat 0) / Beta (seat 1),
    seat 0 knowing tackle/rest/protect and seat 1 `peer_moves` (all usable), nobody statused. children_only: seat 1 knows
    them only on P1's flat children (N > 16), plain tackle at the roots."""
    n = int(np.asarray(states.value).shape[0]);moves = np.zeros((n, 2, 4), np.int32);moves[:, 0] = (TACKLE, REST, PROTECT, 0)
    moves[:, 1] = peer_moves if not children_only or n > 16 else (TACKLE, 0, 0, 0)
    party = np.zeros((n, 2, 6), np.int32);party[:, 0, 0], party[:, 1, 0] = 1, 2
    return dict(active=np.zeros((n, 2), np.int32), species=party[:, :, 0].copy(), moves=moves, charge=np.zeros((n, 2), np.int32),
        usable=np.ones((n, 2, 4), bool), party=party, status=np.zeros((n, 2, 6), np.int32), alive=np.ones((n, 2, 6), bool), rest=np.zeros((n, 2, 6), bool))


def varied_labels(states, *, seat=1, sleeper=None, move=SURF):
    """Labels that vary per row and per seat: on P1's flat children (N > 16) `seat` knows `move` on lane 0 only on the rows whose
    lane (row % 81) is odd (tackle elsewhere; the other seat tackle/rest/protect everywhere), and seat `sleeper`'s slot 0 sleeps."""
    out = fake_labels(states, peer_moves=(TACKLE, REST, PROTECT, 0));n = out['moves'].shape[0]
    out['moves'][:, seat, 0] = TACKLE
    if n > 16:out['moves'][(np.arange(n) % dn.LANES) % 2 == 1, seat, 0] = move
    if sleeper is not None:out['status'][:, sleeper, 0] = 5
    return out


class SeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CW.ConstructorWorldsTests.setUpClass();cls.cw = CW.ConstructorWorldsTests('test_both_root_moves_are_reported_and_can_differ')

    def setUp(self):
        keep = patch.object(cw, 'world_completer', return_value=lambda world, owner, **kw: world);keep.start();self.addCleanup(keep.stop)

    def batch(self):
        W = 8;roots = {f'r{i}': self.cw.made(i, owner, self.cw.modes(i, W)) for i, owner in ROOTS}
        constructor, prepared, _ = self.cw.prepared(roots, W)
        return constructor, prepared

    def price(self, pricer, observe=False):
        constructor, prepared = self.batch()
        with self.cw.forward_patch():
            b = cw.build(constructor, prepared, pricer, CW.PARAMS, binding=self.cw.binding, recipe=self.cw.recipe)
            if observe:pricer.observe()
            frame, _ = cw.price(pricer, b, self.cw.binding, self.cw.recipe, CW.PARAMS)
            children = pricer.observed() if observe else None
        return b, frame, children

    def make(self, cls=dn.DepthNPricer, **kw):
        return cls(CW.carry_executor(), object(), capacity=8, block_lanes=16, actor_capacity=3, depth=2, k_own=(2,), k_opp=(2,), **kw)

    def test_the_seam_is_observer_only_bitwise_off_and_on(self):
        """The fixed roots (ROOTS) priced by the base DepthNPricer, by ObservedDepthNPricer with the flag off, and with it on:
        every frame array equal bit for bit, the same program keys, and children captured only when on."""
        self.assertTrue(gi.check_seam())
        base = self.make();_, reference, _ = self.price(base)
        off = self.make(gi.ObservedDepthNPricer, labels=fake_labels);_, frame_off, _ = self.price(off)
        self.assertEqual((off.observing, off.children), (False, []))
        on = self.make(gi.ObservedDepthNPricer, labels=fake_labels);b, frame_on, (children, notes) = self.price(on, observe=True)
        for frame in (frame_off, frame_on):self.cw.assert_frames_equal(reference, frame)
        self.assertEqual(sorted(map(repr, base.programs)), sorted(map(repr, on.programs)));self.assertEqual(sorted(map(repr, base.programs)), sorted(map(repr, off.programs)))
        self.assertEqual((len(children), notes, on.observing), (1, [], False))
        self.assertEqual(children[0]['lane_cell'].shape, (16, dn.LANES));self.assertEqual(children[0]['labels']['moves'].shape, (16 * dn.LANES, 2, 4))
        # a P1 output the seam cannot read leaves a note and no children: the guard then fails closed
        broken = self.make(gi.ObservedDepthNPricer, labels=lambda states: 1 / 0);_, frame_broken, (none, why) = self.price(broken, observe=True)
        self.cw.assert_frames_equal(reference, frame_broken);self.assertEqual((none, why[0][:30]), ([], 'P1 children: ZeroDivisionError'))

    def record(self, labels, *, children=None, choice=None, lines=(), request_line=None, entry=('A', 'test', None), found_out=None):
        pricer = self.make(gi.ObservedDepthNPricer, labels=labels);b, frame, (found, notes) = self.price(pricer, observe=True)
        if found_out is not None:found_out.extend(found)
        root = b.roots[0];worlds = [w for r in root['rows'] for w in b.worlds.pairs[r]]
        request = dict(lines=list(lines), own_request_line=request_line)
        with ExitStack() as stack:   # surf guarded (tier A) for the test: the fixture pool has no guarded move
            stack.enter_context(patch.dict(G.GUARDED, surf=entry));stack.enter_context(patch.object(G, 'WATCHED', G.WATCHED | {'surf'}))
            return gi.effect_record(frame, root, worlds, found if children is None else children, request, 0, self.cw.metadata['vocabulary'], choice,
                labels=labels, notes=notes), frame, root

    def test_the_peers_turn_1_and_turn_2_priors_trip_the_guard(self):
        record, frame, root = self.record(fake_labels)
        self.assertEqual((record['tripped'], record['b_turn1'], record['b_turn2']), (True, True, True))
        turn1 = [x for x in record['reasons'] if x['source'] == 'turn1']
        self.assertEqual([(x['move'], x['side'], x['tier']) for x in turn1], [('surf', 'opp', 'A')])
        p = G.masses([float(x) for x in frame.root_peer_logits[0, 0]], [bool(x) for x in frame.root_peer_legal[0, 0]])
        self.assertAlmostEqual(next(m['mass'] for m in record['masses']['turn1'] if m['world'] == 0), p[0], places=12)   # the frame's own prior
        self.assertTrue(turn1[0]['worlds'] and set(turn1[0]['worlds']) <= set(range(8)))
        self.assertGreater(record['turn2_cells']['expanded'], 0);self.assertEqual(record['turn2_cells']['unlabelled'], 0)
        self.assertEqual(record['turn2_cells']['p1_captures'], 1)
        only2, _, _ = self.record(lambda s: fake_labels(s, children_only=True))   # the peer knows Surf only at turn-1 children
        self.assertEqual((only2['b_turn1'], only2['b_turn2']), (False, True))
        cells = [m for m in only2['masses']['turn2'] if m['move'] == 'surf']
        self.assertTrue(cells and all(m['side'] == 'opp' and 0 <= m['cell'] < dn.LANES for m in cells))
        clear, _, _ = self.record(lambda s: fake_labels(s, peer_moves=(TACKLE, REST, PROTECT, 0)))
        self.assertEqual((clear['tripped'], clear['reasons'], clear['a']), (False, [], False))
        missing, _, _ = self.record(lambda s: fake_labels(s, peer_moves=(TACKLE, REST, PROTECT, 0)), children=[])
        self.assertEqual((missing['tripped'], [x['source'] for x in missing['reasons']]), (True, ['turn2_unavailable']))   # fail closed

    def test_turn_2_reads_each_cells_own_child_row(self):
        """Each expanded cell reads the child row its lane holds (P1 row w, lane l: w*81 + l, through lane_cell), not w*81 + cell: the
        peer's Surf sits on odd lanes only, so the guarded (world, cell) set is exactly theirs."""
        found = [];record, frame, root = self.record(varied_labels, found_out=found)
        lane_cell = found[0]['lane_cell'];expected = set();discriminating = False
        for k, r in enumerate(root['rows']):
            for jw in (0, 1):
                w = 2 * r + jw
                for lane in range(dn.LANES):
                    cell = int(lane_cell[w, lane])
                    if cell >= dn.LANES:continue
                    discriminating |= lane % 2 != cell % 2
                    a, b = divmod(cell, 9)
                    if (lane % 2 == 1 and int(frame.t1_status[r, jw, a, b]) == 2 and bool(frame.t1_peer_requested[r, jw, a, b])
                            and bool(frame.t1_peer_legal[r, jw, a, b][0])):expected.add((2 * k + jw, cell))
        got = {(m['world'], m['cell']) for m in record['masses']['turn2'] if m['move'] == 'surf' and m['side'] == 'opp'}
        self.assertTrue(discriminating and expected);self.assertEqual(got, expected)
        self.assertEqual((record['b_turn1'], record['b_turn2']), (False, True))

    def test_each_side_at_a_child_reads_its_own_seat(self):
        """Our Surf is a sleep move here (Sleep Clause): guarded at a child only while the OPPONENT's seat has a sleeper there."""
        clause = ('C', 'status', 'sleep_clause')
        theirs, _, _ = self.record(lambda s: varied_labels(s, seat=0, sleeper=1), entry=clause)
        self.assertEqual({(x['move'], x['side'], x['source']) for x in theirs['reasons']}, {('surf', 'own', 'turn2')})
        ours, _, _ = self.record(lambda s: varied_labels(s, seat=0, sleeper=0), entry=clause)   # only our own seat sleeps: no clause
        self.assertEqual(ours['reasons'], [])

    def test_c_reads_our_requests_lanes(self):
        line = '|request|{"side":{"id":"p1","pokemon":[{"ident":"p1: Alpha","details":"Alpha, L80","condition":"100/100","active":true}]},' \
            '"active":[{"moves":[{"id":"surf"},{"id":"tackle"}]}]}'
        record, _, _ = self.record(lambda s: fake_labels(s, peer_moves=(TACKLE, REST, PROTECT, 0)), choice=0, request_line=line)
        self.assertEqual((record['c'], record['tripped'], record['own_lanes'][:2]), (True, True, ['surf', 'tackle']))
        self.assertEqual(record['own_guarded_lanes'], [['A', 'test'], None, None, None])   # our lanes are the request's
        self.assertEqual((G.trips(record, 0)['c'], G.trips(record, 1)['c'], G.trips(record, 4)['c']), (True, False, False))


if __name__ == '__main__':
    unittest.main()
