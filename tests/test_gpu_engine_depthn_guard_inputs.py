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

    def batch(self, modes=None):
        W = 8;roots = {f'r{i}': self.cw.made(i, owner, self.cw.modes(i, W) if modes is None else modes) for i, owner in ROOTS}
        constructor, prepared, _ = self.cw.prepared(roots, W)
        return constructor, prepared

    def price(self, pricer, observe=False, modes=None):
        constructor, prepared = self.batch(modes)
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

    def record(self, labels, *, children=None, choice=None, lines=(), request_line=None, entry=('A', 'test', None), found_out=None, modes=None,
               reply=None, edit=None):
        pricer = self.make(gi.ObservedDepthNPricer, labels=labels);b, frame, (found, notes) = self.price(pricer, observe=True, modes=modes)
        if edit is not None:frame = edit(frame)
        if found_out is not None:found_out.extend(found)
        root = b.roots[0];worlds = [w for r in root['rows'] for w in b.worlds.pairs[r]]
        request = dict(lines=list(lines), own_request_line=request_line)
        with ExitStack() as stack:   # surf guarded (tier A) for the test: the fixture pool has no guarded move
            stack.enter_context(patch.dict(G.GUARDED, surf=entry));stack.enter_context(patch.object(G, 'WATCHED', G.WATCHED | {'surf'}))
            return gi.effect_record(frame, root, worlds, found if children is None else children, request, 0, self.cw.metadata['vocabulary'], choice,
                labels=labels, notes=notes, reply_logits=None if reply is None else reply(frame, root)), frame, root

    @staticmethod
    def champion_reply(frame, root, lane=None, high=50.):
        """Per world in block order: the frame's own peer logits for a requested world (None when waiting); with `lane`, that
        lane's logit raised to `high` (the reply model puts its mass there)."""
        out = []
        for r in root['rows']:
            for jw in (0, 1):
                if not bool(frame.root_peer_requested[r, jw]):out.append(None);continue
                logits = [float(x) for x in frame.root_peer_logits[r, jw]]
                if lane is not None:logits[lane] = high
                out.append(logits)
        return out

    def test_the_union_guard_is_the_champion_record_when_the_reply_model_is_the_champion(self):
        plain, frame, root = self.record(fake_labels)
        union, _, _ = self.record(fake_labels, reply=self.champion_reply)
        self.assertEqual(union['union']['champion'], union['union']['reply']);self.assertEqual(union['tripped'], plain['tripped'])
        self.assertFalse(union['union_only'])
        self.assertEqual({k: v for k, v in union.items() if k not in ('union', 'union_only')}, plain)   # every champion field unchanged
        clear, _, _ = self.record(lambda s: fake_labels(s, peer_moves=(TACKLE, REST, PROTECT, 0)), reply=self.champion_reply)
        self.assertEqual((clear['tripped'], clear['union']['tripped']), (False, False))
        with self.assertRaisesRegex(ValueError, 'reply logits for'):self.record(fake_labels, reply=lambda f, r:self.champion_reply(f, r)[:-1])
        def hole(f, r):   # a requested world without logits, and one with too few: both refused (the live search's error record trips)
            out = self.champion_reply(f, r);w = next(i for i, x in enumerate(out) if x is not None);out[w] = None;return out
        with self.assertRaisesRegex(ValueError, 'requested world'):self.record(fake_labels, reply=hole)
        def short(f, r):
            out = self.champion_reply(f, r);w = next(i for i, x in enumerate(out) if x is not None);out[w] = out[w][:8];return out
        with self.assertRaisesRegex(ValueError, '8 values, not 9'):self.record(fake_labels, reply=short)

    def test_a_reply_model_moving_turn_1_peer_mass_onto_a_guarded_move_trips_the_union_only(self):
        """Surf on the peer's lane 1 (a legal lane of the fixture's peer) at the roots only (no turn-2 child knows it), and the
        champion's turn-1 logit on lane 1 lowered to -50 in every world: the champion's record stays clear; a reply model raising
        lane 1 trips the union only."""
        lane = 1
        def roots_only(states):
            out = fake_labels(states, peer_moves=(TACKLE, SURF, PROTECT, 0))
            if out['moves'].shape[0] > 16:out['moves'][:, 1] = (TACKLE, REST, PROTECT, 0)   # the children: no Surf
            return out
        def lowered(frame):
            logits = np.array(frame.root_peer_logits, np.float32);logits[..., lane] = -50.
            return replace(frame, root_peer_logits=logits)
        champion, frame, root = self.record(roots_only, edit=lowered)
        self.assertTrue(any(bool(frame.root_peer_legal[r, jw][lane]) and bool(frame.root_peer_requested[r, jw]) for r in root['rows'] for jw in (0, 1)))
        self.assertEqual((champion['tripped'], champion['b_turn1']), (False, False))
        union, _, _ = self.record(roots_only, edit=lowered, reply=lambda f, r:self.champion_reply(f, r, lane))
        self.assertEqual((union['tripped'], union['union']['champion']['tripped'], union['union']['reply']['tripped']), (True, False, True))
        self.assertTrue(union['union']['reply']['b_turn1']);self.assertEqual(union['b_turn1'], False)   # the record's own fields: the champion's
        self.assertTrue(union['union_only'])
        lowered_reply, _, _ = self.record(roots_only, edit=lowered, reply=lambda f, r:self.champion_reply(f, r, lane, high=-50.))
        self.assertEqual((lowered_reply['tripped'], lowered_reply['union']['reply']['tripped']), (False, False))

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
        self.assertEqual([x['source'] for x in missing['any_cell_reasons']], ['turn2_unavailable'])

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
        self.assertEqual({(x['move'], x['side'], x['source']) for x in theirs['any_cell_reasons']}, {('surf', 'own', 'turn2')})
        # reasons list only what tripped: the reach on our side (M_own crosses THRESHOLD here), its tier C mass from Surf
        self.assertGreaterEqual(theirs['m_own'], G.THRESHOLD);self.assertEqual((theirs['m_opp'], theirs['tiers']), (0., ['C']))
        self.assertEqual([(x['side'], x['source'], x['tiers'], list(x['moves'])) for x in theirs['reasons']], [('own', 'reach', {'C': theirs['m_own']}, ['surf'])])
        ours, _, _ = self.record(lambda s: varied_labels(s, seat=0, sleeper=0), entry=clause)   # only our own seat sleeps: no clause
        self.assertEqual((ours['reasons'], ours['any_cell_reasons'], ours['m_own']), ([], [], 0.))

    def test_turn_2_cells_are_our_lane_times_9_plus_the_reply(self):
        """Every turn-2 mass entry's cell is a * 9 + b on the frame's [own lane a, reply b] axes (what the reach reads), its mass
        the prior of that cell's own logits; a world whose opponent waits at the root expands only its (a, 0) cells."""
        record, frame, root = self.record(fake_labels)
        for e in record['masses']['turn2']:
            k, jw = divmod(e['world'], 2);r = root['rows'][k];a, b = divmod(e['cell'], 9);own = e['side'] == 'own'
            self.assertEqual(int(frame.t1_status[r, jw, a, b]), 2)
            self.assertTrue(bool((frame.t1_own_requested if own else frame.t1_peer_requested)[r, jw, a, b]))
            p = G.masses([float(x) for x in (frame.t1_own_logits if own else frame.t1_peer_logits)[r, jw, a, b]],
                [bool(x) for x in (frame.t1_own_legal if own else frame.t1_peer_legal)[r, jw, a, b]])
            self.assertIn(e['mass'], p);self.assertEqual((e['guarded'], e['tier'], e['user']), (e['move'] == 'surf', 'A' if e['move'] == 'surf' else None,
                'beta' if not own else 'alpha'))
        waiting, frame, root = self.record(fake_labels, modes=(6,) * 8)   # the opponent waits at the root in every world
        self.assertFalse(frame.root_peer_requested[root['rows'][0]].any())
        cells = {e['cell'] for e in waiting['masses']['turn2']}
        self.assertTrue(cells and all(c % 9 == 0 for c in cells))   # (a, 0) only
        self.assertFalse(any(x['source'] == 'turn1' and x['side'] == 'opp' for x in waiting['reasons']))

    def test_c_reads_our_requests_lanes(self):
        line = '|request|{"side":{"id":"p1","pokemon":[{"ident":"p1: Alpha","details":"Alpha, L80","condition":"100/100","active":true}]},' \
            '"active":[{"moves":[{"id":"surf"},{"id":"tackle"}]}]}'
        record, _, _ = self.record(lambda s: fake_labels(s, peer_moves=(TACKLE, REST, PROTECT, 0)), choice=0, request_line=line)
        self.assertEqual((record['c'], record['tripped'], record['own_lanes'][:2]), (True, True, ['surf', 'tackle']))
        self.assertEqual(record['own_guarded_lanes'], [['A', 'test'], None, None, None])   # our lanes are the request's
        self.assertEqual((G.trips(record, 0)['c'], G.trips(record, 1)['c'], G.trips(record, 4)['c']), (True, False, False))


if __name__ == '__main__':
    unittest.main()
