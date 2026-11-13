"""Engine-held volatiles at the constructor: Yawn's projection and Perish Song's count (phase_d_protocol_world), and our own
holders' admission by from_b2 per route. The projection and count tests are pure payload and lines; the world and admission tests
use the engine state and the original B2 adapter (pod or CI runtime)."""
import json
from pathlib import Path
import unittest

from gpu_engine import phase_d_protocol_world as protocol


def payload(p1=(), p2=()):
    return {'sides': {'p1': {'volatiles': list(p1), 'pokemon': []}, 'p2': {'volatiles': list(p2), 'pokemon': []}}}


class YawnProjectionTests(unittest.TestCase):
    start = ['|turn|3', '|move|p2a: Slowbro|Yawn|p1a: Snorlax', '|-start|p1a: Snorlax|move: Yawn|[of] p2a: Slowbro']

    def test_the_count_is_two_before_the_residual_and_one_after(self):
        """Showdown's Yawn: duration 2 at |-start|...|move: Yawn, -1 at each residual (its |upkeep|): 2 at a replacement
        boundary of that turn, 1 at the next decision. The volatile leaves the payload the CPU materializer sees."""
        for lines, count in ((self.start, 2), (self.start + ['|', '|upkeep', '|turn|4'], 1)):
            with self.subTest(count=count):
                projected, yawns = protocol._defer_yawn(payload(p1=('yawn', 'leechseed')), lines)
                self.assertEqual(yawns, ((0, count),))
                self.assertEqual(projected['sides']['p1']['volatiles'], ['leechseed'])

    def test_no_yawn_is_the_same_payload_and_an_unexplained_one_refuses(self):
        source = payload(p2=('leechseed',))
        projected, yawns = protocol._defer_yawn(source, self.start)
        self.assertEqual((projected, yawns), (source, ()))
        with self.assertRaisesRegex(ValueError, 'PUBLIC_YAWN_START_UNAVAILABLE'):
            protocol._defer_yawn(payload(p2=('yawn',)), self.start)   # the start line names p1's active
        with self.assertRaisesRegex(ValueError, 'PUBLIC_YAWN_COUNT_UNAVAILABLE'):
            protocol._defer_yawn(payload(p1=('yawn',)), self.start + ['|upkeep', '|turn|4', '|upkeep', '|turn|5'])


class PerishCountTests(unittest.TestCase):
    """phase_d_protocol_world._perish_counts on a real r4a3 prefix (ordinal 482, decision 5) and its mutants: Showdown sets
    duration 4 and prints a silent perish3 at the use; the residual decrements, then prints perishN before |upkeep|."""
    lines = json.loads((Path(__file__).parent / 'fixtures' / 'phase_d_held_volatiles' / 'perish_song_482_5.json').read_text())['lines']
    use = lines.index('|move|p2a: Misdreavus|Perish Song|p2a: Misdreavus')
    upkeep = lines.index('|upkeep', use)
    passed = ['|move|p1a: Blissey|Baton Pass|p1a: Blissey', '|switch|p1a: Chansey|Chansey, L80, F|300/300|[from] Baton Pass']

    def test_after_the_residual_the_shown_count(self):
        self.assertEqual(protocol._perish_counts(self.lines), (3, 3))

    def test_a_boundary_in_the_songs_own_turn_holds_four(self):
        """Mutant: the prefix cut after the use's silent perish3 lines (a mid-turn replacement before that turn's residual)."""
        cut = self.lines[:self.use + 4]
        self.assertEqual(cut[-1], '|-fieldactivate|move: Perish Song')
        self.assertEqual(protocol._perish_counts(cut), (4, 4))

    def test_a_baton_pass_carries_the_count_to_the_recipient(self):
        """Mutants: Blissey Baton Passes in the Song's own turn (the recipient holds 4) and after the residual (it holds 3);
        the side's latest line is still the passer's, as Showdown prints none for the recipient until the next residual."""
        same_turn = self.lines[:self.use + 4] + self.passed
        self.assertEqual(protocol._perish_counts(same_turn), (4, 4))
        later = self.lines[:self.upkeep + 2] + self.passed
        self.assertEqual(later[self.upkeep + 1], '|turn|6')
        self.assertEqual(protocol._perish_counts(later), (3, 3))

    def test_the_world_holds_four_in_the_songs_turn_and_refuses_an_unexplained_count(self):
        """_write_held_counts: map_spec wrote the shown 3 on both actives; in the Song's own turn the world holds 4, after the
        residual 3; Yawn's count lands on its side's active; a shown count without a perish line refuses."""
        import jax.numpy as j
        from gpu_engine.state import empty_state
        state = empty_state(1)._replace(active=j.asarray([[0, 2]], j.int32))
        state = state._replace(perish_counter=state.perish_counter.at[0, 0, 0].set(3).at[0, 1, 2].set(3))
        out = protocol._write_held_counts(state, ((1, 2),), protocol._perish_counts(self.lines[:self.use + 4]))
        self.assertEqual((int(out.perish_counter[0, 0, 0]), int(out.perish_counter[0, 1, 2])), (4, 4))
        self.assertEqual(int(out.yawn_counter[0, 1, 2]), 2)
        after = protocol._write_held_counts(state, (), protocol._perish_counts(self.lines))
        self.assertEqual((int(after.perish_counter[0, 0, 0]), int(after.perish_counter[0, 1, 2])), (3, 3))
        with self.assertRaisesRegex(ValueError, 'PUBLIC_PERISH_COUNT_UNAVAILABLE'):
            protocol._write_held_counts(state, (), protocol._perish_counts(self.lines[:self.use]))

    def test_no_perish_line_is_none_and_the_next_residual_counts_down(self):
        self.assertEqual(protocol._perish_counts(self.lines[:self.use]), (None, None))
        nxt = self.lines + ['|', '|-start|p2a: Misdreavus|perish2', '|-start|p1a: Blissey|perish2', '|upkeep', '|turn|7']
        self.assertEqual(protocol._perish_counts(nxt), (2, 2))


FIXTURES = Path(__file__).parent / 'fixtures' / 'phase_d_held_volatiles'
TOLERANCE = 0.10   # pre-stated: each value of a uniform support within 10% of its expected count


class EncorePosteriorTests(unittest.TestCase):
    """phase_d_protocol_world.encore_support against Showdown f76228a1's measured Encore counts (a pinned simulation table),
    _encore_facts on real r4a3 lines and mutants, and _write_encores' per-world draw."""
    sim = json.loads((FIXTURES / 'encore_showdown_f76228a1.json').read_text())

    def assert_uniform(self, counts, support):
        self.assertEqual(sorted(counts), sorted(support))
        expected = sum(counts.values()) / len(support)
        for value, n in counts.items():self.assertLessEqual(abs(n - expected), TOLERANCE * expected, (value, n, expected))

    def test_the_support_is_showdowns_measured_one(self):
        """For b = 0 and 1: the start duration (k = 0) and the remaining count after every k the battles reached land exactly on
        encore_support(k, b), with flat frequencies."""
        cells = 0
        for b in (0, 1):
            table = self.sim[f'b{b}']
            self.assert_uniform({int(v): n for v, n in table['residuals_held'].items()}, protocol.encore_support(0, b))
            for k, counts in table['remaining_after_k_residuals'].items():
                self.assert_uniform({int(v): n for v, n in counts.items()}, protocol.encore_support(int(k), b));cells += 1
            self.assertEqual(protocol.encore_support(len(table['remaining_after_k_residuals']) + 1, b), ())   # nobody outlasts 6 + b
        self.assertEqual(cells, 11)

    def test_facts_on_real_lines_and_mutants(self):
        """r4a3 65/8: Wobbuffet Encores Octillery after Octillery's Thunder Wave that turn (b = 1), one residual since (k = 1);
        602/13: Pikachu Encores Jirachi before it acts (b = 0), two residuals since. Mutants: the prefix cut at the start line
        (a boundary in the Encore's own turn, k = 0) and the acted-first line removed (b = 0)."""
        for name, side, facts in (('encore_65_8', 0, (1, 1)), ('encore_602_13', 1, (2, 0))):
            with self.subTest(name):
                lines = json.loads((FIXTURES / (name + '.json')).read_text())['lines']
                got = protocol._encore_facts(lines);self.assertEqual(got[side], facts);self.assertIsNone(got[1 - side])
                start = max(i for i, l in enumerate(lines) if l.startswith('|-start|') and l.endswith('|Encore'))
                self.assertEqual(protocol._encore_facts(lines[:start + 1])[side], (0, facts[1]))
        lines = json.loads((FIXTURES / 'encore_65_8.json').read_text())['lines']
        start = max(i for i, l in enumerate(lines) if l.startswith('|-start|') and l.endswith('|Encore'))
        acted = [i for i in range(start) if lines[i].startswith('|move|p1a: Octillery')][-1]
        self.assertEqual(protocol._encore_facts(lines[:acted] + lines[acted + 1:])[0], (1, 0))
        # A Gen 3 confusion self-hit uses the action and prints only -activate|X|confusion: b = 1 (the replacement rule).
        hurt = lines[:acted] + ['|-activate|p1a: Octillery|confusion'] + lines[acted + 1:]
        self.assertEqual(protocol._encore_facts(hurt)[0], (1, 1))
        # Lines that show the Encore over, or a start naming another mon than the active, give no facts (the write refuses).
        self.assertIsNone(protocol._encore_facts(lines + ['|-end|p1a: Octillery|Encore'])[0])
        renamed = lines[:start] + [lines[start].replace('Octillery', 'Qwilfish')] + lines[start + 1:]
        self.assertIsNone(protocol._encore_facts(renamed)[0])

    @property
    def j(self):
        import jax.numpy as j
        return j

    def world(self, *, pp=5, kind=2, slot=1):
        import jax.numpy as j
        from gpu_engine.state import empty_state
        s = empty_state(1)._replace(active=j.asarray([[0, 2]], j.int32))
        s = s._replace(moves=s.moves.at[0, 1, 2].set(j.asarray([11, 12, 13, 0], j.int32)), pp=s.pp.at[0, 1, 2].set(j.asarray([5, pp, 5, 0], j.int32)),
                       last_action_kind=s.last_action_kind.at[0, 1].set(kind), last_action_index=s.last_action_index.at[0, 1].set(slot),
                       last_move=s.last_move.at[0, 1, 2].set(12), encore_move=s.encore_move.at[0, 1, 2].set(12),
                       encore_counter=s.encore_counter.at[0, 1, 2].set(1))
        return s

    def test_the_world_draw_is_uniform_on_the_support_and_deterministic(self):
        """Over 4,000 world seeds the drawn encore_counter (side 1's active) hits exactly encore_support(k, b) with flat
        frequencies; the same seed draws the same count; the lock stays the mapped move."""
        s = self.world();encored = protocol._mapped_encores(s);self.assertEqual(encored, ((1, 2, 12, 1),))
        for facts in ((1, 1), (2, 0), (0, 0)):
            with self.subTest(facts=facts):
                counts = {}
                for seed in range(4000):
                    out = protocol._write_encores(s, encored, (None, facts), seed, replacement=False, owner=0)
                    n = int(out.encore_counter[0, 1, 2]);counts[n] = counts.get(n, 0) + 1
                    self.assertEqual(int(out.encore_move[0, 1, 2]), 12)
                self.assert_uniform(counts, protocol.encore_support(*facts))
        once = lambda seed: int(protocol._write_encores(s, encored, (None, (1, 1)), seed, replacement=False, owner=0).encore_counter[0, 1, 2])
        self.assertEqual([once(7)] * 3, [once(7), once(7), once(7)])

    def test_refusals(self):
        """No start line, a support the lines rule out, a last action that is not the lock, and the locked move at 0 PP at an
        ordinary decision refuse; at a replacement boundary (before that turn's residual) 0 PP is drawn as usual."""
        s = self.world();encored = protocol._mapped_encores(s)
        for facts, world, kind in (((None, None), s, 'PUBLIC_ENCORE_START_UNAVAILABLE'), ((None, (6, 0)), s, 'PUBLIC_ENCORE_COUNT_UNAVAILABLE'),
                                   ((None, (1, 0)), self.world(slot=2), 'PUBLIC_ENCORE_LOCK_DISAGREEMENT'),
                                   ((None, (1, 0)), self.world(kind=1), 'PUBLIC_ENCORE_LOCK_DISAGREEMENT'),
                                   ((None, (1, 0)), self.world(pp=0), 'PUBLIC_ENCORE_ZERO_PP')):
            with self.subTest(kind):
                with self.assertRaisesRegex(ValueError, kind):protocol._write_encores(world, encored, facts, 3, replacement=False, owner=0)
        out = protocol._write_encores(self.world(pp=0), encored, (None, (1, 0)), 3, replacement=True, owner=0)
        self.assertIn(int(out.encore_counter[0, 1, 2]), protocol.encore_support(1, 0))

    def test_our_own_encored_active_keeps_no_request_disable_flags(self):
        """Our own request marks every move but the lock disabled (Encore's onDisableMove); with exactly that pattern the
        flags are cleared, so the engine's own Encore clock (and Choice and PP) decide once the drawn count ends. Another
        pattern refuses."""
        s = self.world();s = s._replace(disabled=s.disabled.at[0, 1, 2].set(self.j.asarray([True, False, True, False])))
        encored = protocol._mapped_encores(s)
        out = protocol._write_encores(s, encored, (None, (1, 0)), 3, replacement=False, owner=1)
        self.assertFalse(bool(self.j.any(out.disabled[0, 1, 2])));self.assertIn(int(out.encore_counter[0, 1, 2]), protocol.encore_support(1, 0))
        foe = protocol._write_encores(s, encored, (None, (1, 0)), 3, replacement=False, owner=0)   # the opponent's: untouched
        self.assertTrue(bool(self.j.array_equal(foe.disabled, s.disabled)))
        for flags in ([False, False, True, False], [True, True, True, False]):
            with self.subTest(flags=flags):
                bad = s._replace(disabled=s.disabled.at[0, 1, 2].set(self.j.asarray(flags)))
                with self.assertRaisesRegex(ValueError, 'PUBLIC_ENCORE_REQUEST_DISAGREEMENT'):
                    protocol._write_encores(bad, encored, (None, (1, 0)), 3, replacement=False, owner=1)


class OwnHeldVolatileAdmissionTests(unittest.TestCase):
    def test_own_destiny_bond_on_every_route_perish_and_yawn_only_with_protocol_facts(self):
        """from_b2 on the original B2 adapter: our active's Destiny Bond rides the own volatile (DESTINY_BOND) on every route;
        a Perish or Yawn count is admitted only on the protocol route (protocol_facts), the one that writes it, and the older
        route still refuses OWN_VOLATILE_OR_LATENT_CLOCK_UNAVAILABLE."""
        from types import SimpleNamespace as NS
        from test_phase_d_device_worlds import WorldsTests
        from pokezero.v5_contract import DecisionAddress
        from gpu_engine import phase_d_live_worlds as live, phase_d_tail as tail
        from gpu_engine.state import DESTINY_BOND
        WorldsTests.setUpClass();helper=WorldsTests();owner=0;own,foe='p1','p2'
        cases={'destinybond':(f'|-singlemove|{own}a: Alpha|Destiny Bond',),'perish3':(f'|-start|{own}a: Alpha|perish3|[silent]',),
               'yawn':(f'|-start|{own}a: Alpha|move: Yawn|[of] {foe}a: Beta',)}
        for name,wire in cases.items():
            with self.subTest(name):
                adapter=helper.adapter(owner);packet=helper.request(owner);adapter.feed(wire)
                payload=dict(turn=1,selfPlayer=own,selfRequestKind='move',pendingBatonPassSides=[],wishSetTurns={},
                    sides={own:dict(pokemon=[dict(species='Alpha',condition='291/291',active=True)],volatiles=[]),
                           foe:dict(pokemon=[dict(species='Beta',condition='100/100',active=True)],volatiles=[])})
                lines=('|start',f'|switch|{own}a: Alpha|Alpha, L80|291/291',f'|switch|{foe}a: Beta|Beta, L80|100/100','|turn|1')+wire+('|request|'+json.dumps(packet),)
                facts=tail.facts(NS(materialization_payload=payload),packet,lines);address=DecisionAddress(1,0,owner)
                protocol_root=live.from_b2(adapter,packet,helper.source,helper.metadata,address=address,defer_sleep=True,protocol_facts=facts)
                at=int(protocol_root.own['active'])
                if name=='destinybond':
                    older=live.from_b2(adapter,packet,helper.source,helper.metadata,address=address)
                    for root in (older,protocol_root):self.assertEqual(int(root.own['volatile'][at])&DESTINY_BOND,DESTINY_BOND)
                else:
                    self.assertEqual(int(protocol_root.own['volatile'][at])&DESTINY_BOND,0)
                    with self.assertRaisesRegex(ValueError,'OWN_VOLATILE_OR_LATENT_CLOCK_UNAVAILABLE.*'+name):
                        live.from_b2(adapter,packet,helper.source,helper.metadata,address=address)


if __name__ == '__main__':
    unittest.main()
