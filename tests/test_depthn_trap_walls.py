"""The constructor's trap walls, pure and light (named test_depthn_* so the depthn-light CI job runs it): wall 1, a request trap
reproduced by the world (preflight admission, our Mean Look's link), and wall 2, a partial trap's hidden clock (wrap_support
against Showdown f76228a1's measured Wrap durations, _wrap_facts on authored lines, the per-world draw, its refusals and the
lines/payload/B2 agreement), and the engine's own masks for the traps the worlds write. Real roots: phase_d_trap_walls_replay."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np

from gpu_engine import device_public_world_state as states, phase_d_live_worlds as live, phase_d_protocol_world as protocol
from gpu_engine.state import BattleState

FIXTURES = Path(__file__).parent / 'fixtures' / 'phase_d_trap_walls'
TOLERANCE = 0.1
SIM = json.loads((FIXTURES / 'wrap_showdown_f76228a1.json').read_text())

START = ('|switch|p1a: Arbok|Arbok, L80, M|100/100', '|switch|p2a: Snorlax|Snorlax, L80, F|100/100', '|turn|1',
         '|move|p1a: Arbok|Wrap|p2a: Snorlax', '|-damage|p2a: Snorlax|96/100', '|-activate|p2a: Snorlax|move: Wrap|[of] p1a: Arbok',
         '|move|p2a: Snorlax|Splash|p2a: Snorlax', '|-nothing')


def tick(n):
    """n turns after the start turn's actions: each residual's tick, |upkeep| and |turn|; then both Splash."""
    out = []
    for k in range(n):
        out += ['|', f'|-damage|p2a: Snorlax|{90 - 6 * k}/100|[from] move: Wrap|[partiallytrapped]', '|upkeep', f'|turn|{k + 2}',
                '|move|p1a: Arbok|Splash|p1a: Arbok', '|-nothing', '|move|p2a: Snorlax|Splash|p2a: Snorlax', '|-nothing']
    return tuple(out)


def host_world(active=(0, 0)):
    """An empty host (NumPy) world: both actives living, the given party slots."""
    arrays = states._empty_arrays()
    arrays['active'][0] = active;arrays['hp'][0, :, :] = 100;arrays['max_hp'][0, :, :] = 100;arrays['done'][0] = False
    return BattleState(**arrays)


def assert_uniform(test, counts, support):
    test.assertEqual(sorted(counts), sorted(support))
    expected = sum(counts.values()) / len(support)
    for value, n in counts.items():test.assertLessEqual(abs(n - expected), TOLERANCE * expected, (value, n, expected))


class WrapSupportTests(unittest.TestCase):
    def test_the_support_is_showdowns_measured_one(self):
        """The remaining duration after every k ticks the 4,000 simulated traps reached lands exactly on wrap_support(k), with
        flat frequencies (the start, k = 0: random(3, 7)); every trap ticked D - 1 times, and every holder request showed
        trapped. Nothing outlasts six residuals."""
        self.assertEqual(SIM['battles'], 4000);self.assertEqual(SIM['ticks_equal_duration_minus_one'], 4000)
        self.assertEqual(SIM['holder_request_trapped'], 4000)
        assert_uniform(self, {int(d): n for d, n in SIM['durations'].items()}, protocol.wrap_support(0))
        for k, counts in SIM['remaining_after_k_ticks'].items():
            with self.subTest(k=k):assert_uniform(self, {int(r): n for r, n in counts.items()}, protocol.wrap_support(int(k)))
        self.assertEqual(sorted(map(int, SIM['remaining_after_k_ticks'])), [0, 1, 2, 3, 4, 5])
        self.assertEqual(protocol.wrap_support(6), ());self.assertEqual(protocol.wrap_support(13), ())

    def test_the_simulated_protocol_is_the_one_the_facts_read(self):
        """Showdown's own start, tick and end lines (and the trapper-leaves silent end) drive _wrap_facts as authored."""
        start, damage, end = SIM['sample_lines']
        self.assertEqual(start, '|-activate|p2a: Snorlax|move: Wrap|[of] p1a: Arbok')
        lines = ['|switch|p1a: Arbok|Arbok, M|261/261', '|switch|p2a: Snorlax|Snorlax, M|461/461', '|turn|1', start, '|', damage, '|upkeep', '|turn|2']
        self.assertEqual(protocol._wrap_facts(lines), (None, dict(k=1, upkeeps=1, turns=1, trapper_active=True)))
        for ending in (end, *SIM['trapper_leaves']):
            with self.subTest(ending):self.assertEqual(protocol._wrap_facts(lines + ['|', ending, '|upkeep']), (None, None))
        self.assertEqual(SIM['trapper_leaves'], {'|-end|p2a: Snorlax|Wrap|[partiallytrapped]|[silent]': 200})


class WrapFactsTests(unittest.TestCase):
    def test_counts_from_the_start(self):
        """At a boundary before the start turn's residual nothing has ticked; each later turn adds a tick, an |upkeep| and a
        |turn|. A repeat hit on the trapped mon prints no new start and changes nothing."""
        self.assertEqual(protocol._wrap_facts(START), (None, dict(k=0, upkeeps=0, turns=0, trapper_active=True)))
        for n in (1, 2, 5):
            with self.subTest(n=n):self.assertEqual(protocol._wrap_facts(START + tick(n))[1], dict(k=n, upkeeps=n, turns=n, trapper_active=True))
        again = START + tick(1) + ('|move|p1a: Arbok|Wrap|p2a: Snorlax', '|-damage|p2a: Snorlax|80/100') + tick(1)[1:]
        self.assertEqual(protocol._wrap_facts(again)[1], dict(k=2, upkeeps=2, turns=2, trapper_active=True))
        self.assertEqual(protocol._wrap_facts(START + ('|upkeep', '|turn|2'))[1]['k'], 0)   # an unticked residual: k != upkeeps

    def test_every_partial_trap_move_and_both_sides(self):
        for move in ('Wrap', 'Bind', 'Clamp', 'Fire Spin', 'Whirlpool', 'Sand Tomb'):
            with self.subTest(move):
                lines = [x.replace('Wrap', move) for x in START + tick(2)]
                self.assertEqual(protocol._wrap_facts(lines)[1]['k'], 2)
        mirrored = [x.replace('p1a', 'pXa').replace('p2a', 'p1a').replace('pXa', 'p2a') for x in START + tick(1)]
        self.assertEqual(protocol._wrap_facts(mirrored), (dict(k=1, upkeeps=1, turns=1, trapper_active=True), None))
        self.assertEqual(protocol._wrap_facts(START[:5]), (None, None))   # no start line: no trap

    def test_lines_that_end_or_move_the_trap(self):
        """Over: a later -end for the holder (the duration, Rapid Spin, a Substitute's silent end, the trapper gone at a residual;
        a move: prefix is read too) or the holder's faint. Not this active's: a holder that switched out (Baton Pass, a drag) or a
        start naming another mon. The trapper switched out (no residual since) keeps the fact, marked not active."""
        base = START + tick(1)
        for extra in ('|-end|p2a: Snorlax|Wrap|[partiallytrapped]', '|-end|p2a: Snorlax|Wrap|[partiallytrapped]|[silent]',
                      '|-end|p2a: Snorlax|move: Wrap|[partiallytrapped]', '|faint|p2a: Snorlax',
                      '|switch|p2a: Blissey|Blissey, L80, F|100/100', '|drag|p2a: Blissey|Blissey, L80, F|100/100'):
            with self.subTest(extra):self.assertIsNone(protocol._wrap_facts(base + (extra,))[1])
        renamed = tuple(x.replace('|-activate|p2a: Snorlax', '|-activate|p2a: Blissey') for x in base)
        self.assertIsNone(protocol._wrap_facts(renamed)[1])
        # Another side's end line leaves this trap alone.
        self.assertEqual(protocol._wrap_facts(base + ('|-end|p1a: Arbok|Wrap|[partiallytrapped]',))[1]['k'], 1)
        moved = protocol._wrap_facts(base + ('|switch|p1a: Smeargle|Smeargle, L80, M|100/100',))[1]
        self.assertEqual(moved, dict(k=1, upkeeps=1, turns=1, trapper_active=False))
        unnamed = tuple(x.replace('|[of] p1a: Arbok', '') for x in base)
        self.assertFalse(protocol._wrap_facts(unnamed)[1]['trapper_active'])


class DeferTests(unittest.TestCase):
    def payload(self, p1=(), p2=(), meanlook=(), trapped=True):
        sides = {seat: dict(volatiles=list(vols), meanlookTrap=seat in meanlook, pokemon=[]) for seat, vols in (('p1', p1), ('p2', p2))}
        return dict(selfPlayer='p1', selfActiveRequestState=dict(trapped=True) if trapped else {}, sides=sides)

    def test_nothing_to_defer_is_the_same_payload(self):
        payload = self.payload(p1=('leechseed',), meanlook=('p2',))
        self.assertIs(protocol._defer_traps(payload, 0)[0], payload);self.assertEqual(protocol._defer_traps(payload, 0)[1:], (False, ()))

    def test_partial_traps_and_our_mean_look_leave_the_projection(self):
        payload = self.payload(p1=('partiallytrapped', 'leechseed'), p2=('partiallytrapped',), meanlook=('p1',))
        projected, meanlook, wrapped = protocol._defer_traps(payload, 0)
        self.assertEqual((meanlook, wrapped), (True, (0, 1)))
        self.assertEqual(projected['sides']['p1']['volatiles'], ['leechseed']);self.assertEqual(projected['sides']['p2']['volatiles'], [])
        self.assertFalse(projected['sides']['p1']['meanlookTrap']);self.assertFalse(projected['selfActiveRequestState']['trapped'])
        self.assertTrue(payload['selfActiveRequestState']['trapped']);self.assertTrue(payload['sides']['p1']['meanlookTrap'])   # untouched

    def test_the_request_trap_stays_unless_its_cause_is_ours(self):
        """Only the opponent wrapped: our request's trap (an ability trap, say) still goes to the CPU materializer's check."""
        projected, meanlook, wrapped = protocol._defer_traps(self.payload(p2=('partiallytrapped',)), 0)
        self.assertEqual((meanlook, wrapped), (False, (1,)));self.assertTrue(projected['selfActiveRequestState']['trapped'])
        projected, _, wrapped = protocol._defer_traps(self.payload(p2=('partiallytrapped',)), 1)   # p2 as the owner: ours
        self.assertEqual(wrapped, (1,));self.assertFalse(projected['selfActiveRequestState']['trapped'])


class PreflightTests(unittest.TestCase):
    def kinds(self, payload):
        return {(r['kind'], r.get('side')): r['status'] for r in protocol.preflight_checks(payload, {})}

    def test_a_request_trap_is_not_a_preflight_refusal(self):
        payload = dict(selfPlayer='p1', selfActiveRequestState=dict(trapped=True), pendingBatonPassSides=[],
                       sides=dict(p1=dict(volatiles=[], pokemon=[]), p2=dict(volatiles=[], pokemon=[])))
        rows = self.kinds(payload)
        self.assertNotIn('native-probed-hard-trapped-own-request', {kind for kind, _ in rows})
        self.assertEqual({rows[('native-probed-or-hidden-volatile-clock', s)] for s in ('p1', 'p2')}, {'CLEAR'})

    def test_mean_look_is_admitted_on_both_named_sides(self):
        for owner, foe in (('p1', 'p2'), ('p2', 'p1')):
            with self.subTest(owner=owner):
                payload = dict(selfPlayer=owner, pendingBatonPassSides=[], sides={owner: dict(volatiles=[], meanlookTrap=True, pokemon=[]),
                                                                                foe: dict(volatiles=[], meanlookTrap=True, pokemon=[])})
                rows = self.kinds(payload)
                self.assertEqual(rows[('native-probed-or-hidden-volatile-clock', owner)], 'CLEAR')
                self.assertEqual(rows[('native-probed-or-hidden-volatile-clock', foe)], 'CLEAR')
        unowned = dict(pendingBatonPassSides=[], sides=dict(p1=dict(volatiles=[], meanlookTrap=True, pokemon=[])))
        self.assertEqual(self.kinds(unowned)[('native-probed-or-hidden-volatile-clock', 'p1')], 'TRIGGERED')   # no owner named
        for volatile in ('trapped', 'solarbeam'):
            payload = dict(selfPlayer='p1', pendingBatonPassSides=[], sides=dict(p1=dict(volatiles=[volatile], pokemon=[])))
            self.assertEqual(self.kinds(payload)[('native-probed-or-hidden-volatile-clock', 'p1')], 'TRIGGERED')


class WriteWrapsTests(unittest.TestCase):
    def fact(self, k, **kw):
        return dict(dict(k=k, upkeeps=k, turns=k, trapper_active=True), **kw)

    def test_the_draw_is_uniform_on_the_support_and_the_trapper_is_the_opposing_active(self):
        """Over 4,000 world seeds the drawn count hits exactly wrap_support(k), flat; the source is the opposing active's party
        index in this world; the same seed draws the same count, and each side has its own RNG (drawing one side changes
        nothing on the other)."""
        world = host_world(active=(2, 4))
        for k in range(6):
            with self.subTest(k=k):
                counts = Counter()
                for seed in range(4000):
                    out = protocol._write_wraps(world, (1,), (None, self.fact(k)), seed)
                    counts[int(out.partial_trap_counter[0, 1, 4])] += 1
                    self.assertEqual(int(out.partial_trap_source[0, 1, 4]), 2)
                assert_uniform(self, counts, protocol.wrap_support(k))
        both = protocol._write_wraps(world, (0, 1), (self.fact(1), self.fact(2)), 11)
        alone = protocol._write_wraps(world, (1,), (None, self.fact(2)), 11)
        self.assertEqual(int(both.partial_trap_counter[0, 1, 4]), int(alone.partial_trap_counter[0, 1, 4]))
        self.assertEqual((int(both.partial_trap_source[0, 0, 2]), int(both.partial_trap_source[0, 1, 4])), (4, 2))
        self.assertIn(int(both.partial_trap_counter[0, 0, 2]), protocol.wrap_support(1))
        untouched = protocol._write_wraps(world, (), (self.fact(1), None), 11)
        self.assertIs(untouched, world)
        self.assertEqual(int(world.partial_trap_counter[0, 1, 4]), 0)   # the input world is not changed

    def test_refusals(self):
        world = host_world()
        for facts, kind in (((None, None), 'PUBLIC_WRAP_START_UNAVAILABLE'), ((None, self.fact(6)), 'PUBLIC_WRAP_COUNT_UNAVAILABLE'),
                            ((None, self.fact(1 + 7)), 'PUBLIC_WRAP_COUNT_UNAVAILABLE'),
                            ((None, self.fact(1, upkeeps=2)), 'PUBLIC_WRAP_TICK_DISAGREEMENT'),
                            ((None, self.fact(1, trapper_active=False)), 'PUBLIC_WRAP_TRAPPER_NOT_ACTIVE')):
            with self.subTest(kind=kind, facts=facts), self.assertRaisesRegex(ValueError, kind):
                protocol._write_wraps(world, (1,), facts, 3)


class AgreementTests(unittest.TestCase):
    bit = live.hooks.VOLATILE_NAMES.index('partiallytrapped')

    def detached(self, own=(False, 0), foe=(False, 0), owner=0, foe_slot=3):
        own_bag = np.zeros((1, 6, len(live.hooks.VOLATILE_NAMES)), bool);foe_bag = np.zeros_like(own_bag)
        own_bag[0, 1, self.bit] = own[0];foe_bag[0, foe_slot, self.bit] = foe[0]
        clock = lambda at, n: np.asarray([[n if k == at else 0 for k in range(6)]], np.int32)
        view = {'active': np.asarray([[k == foe_slot for k in range(6)]]), 'own_volatiles': own_bag, 'volatiles': foe_bag,
                'own_mon_features': {'wrap_trap_turns': clock(1, own[1])}, 'mon_features': {'wrap_trap_turns': clock(foe_slot, foe[1])}}
        return NS(view=view, own={'active': np.asarray(1)})

    def test_agreement_and_each_disagreement(self):
        fact = dict(k=2, upkeeps=2, turns=2, trapper_active=True)
        protocol._wrap_agreement((None, None), (), self.detached(), 0)
        protocol._wrap_agreement((None, fact), (1,), self.detached(foe=(True, 2)), 0)
        protocol._wrap_agreement((fact, None), (0,), self.detached(own=(True, 2)), 0)
        protocol._wrap_agreement((None, fact), (1,), self.detached(own=(True, 2)), 1)   # p2 as the owner: own is side 1
        protocol._wrap_agreement((None, None), (1,), self.detached(foe=(True, 0)), 0)   # no lines: _write_wraps refuses START
        for facts, wrapped, detached, kind in (
                ((None, fact), (), self.detached(foe=(True, 2)), 'PUBLIC_WRAP_PAYLOAD_DISAGREEMENT'),
                ((None, fact), (1,), self.detached(), 'PUBLIC_WRAP_B2_DISAGREEMENT'),
                ((None, None), (), self.detached(foe=(True, 0)), 'PUBLIC_WRAP_B2_DISAGREEMENT'),
                ((None, None), (), self.detached(own=(True, 0)), 'PUBLIC_WRAP_B2_DISAGREEMENT'),
                ((None, fact), (1,), self.detached(foe=(True, 1)), 'PUBLIC_WRAP_B2_DISAGREEMENT'),
                ((fact, None), (0,), self.detached(own=(True, 3)), 'PUBLIC_WRAP_B2_DISAGREEMENT')):
            with self.subTest(kind=kind, facts=facts, wrapped=wrapped), self.assertRaisesRegex(ValueError, kind):
                protocol._wrap_agreement(facts, wrapped, detached, 0)


class OwnMeanLookTests(unittest.TestCase):
    def test_the_link_names_this_worlds_trapper(self):
        """from_b2 wrote the link as 0; the world's opposing active is the trapper (here party slot 5). No link is no change;
        B2 and the payload must agree."""
        world = host_world(active=(1, 5))
        linked = NS(own={'active': np.asarray(1), 'trap_source': np.asarray([-1, 0, -1, -1, -1, -1], np.int32)})
        free = NS(own={'active': np.asarray(1), 'trap_source': np.full(6, -1, np.int32)})
        out = protocol._write_own_meanlook(world, linked, True, 0)
        self.assertEqual(int(out.trap_source[0, 0, 1]), 5);self.assertEqual(int(out.trap_source[0, 1, 5]), -1)
        self.assertIs(protocol._write_own_meanlook(world, free, False, 0), world)
        for detached, meanlook in ((linked, False), (free, True)):
            with self.subTest(meanlook=meanlook), self.assertRaisesRegex(ValueError, 'OWN_MEANLOOK_B2_PAYLOAD_DISAGREEMENT'):
                protocol._write_own_meanlook(world, detached, meanlook, 0)


class EngineMaskTests(unittest.TestCase):
    """The device's own request masks for the traps the worlds write (state.legal_actions, transition.action_mask): a partial
    trap holds while its source is the foe's living active, a Mean Look link likewise; neither touches the trapper's lanes."""
    def switches(self, world, side):
        import jax.numpy as jnp
        state = BattleState(**{k: jnp.asarray(v) for k, v in world._asdict().items()})
        from gpu_engine.state import legal_actions
        return np.asarray(legal_actions(state))[0, side, 4:].tolist()

    def test_written_traps_close_the_holders_switches(self):
        world = host_world(active=(0, 2))
        self.assertEqual(self.switches(world, 1), [True] * 5)
        fact = dict(k=1, upkeeps=1, turns=1, trapper_active=True)
        wrapped = protocol._write_wraps(world, (1,), (None, fact), 9)
        self.assertEqual(self.switches(wrapped, 1), [False] * 5);self.assertEqual(self.switches(wrapped, 0), [True] * 5)
        gone = wrapped._replace(active=np.asarray([[3, 2]], np.int32))   # the trapper left: the trap no longer holds
        self.assertEqual(self.switches(gone, 1), [True] * 5)
        hp = wrapped.hp.copy();hp[0, 0, 0] = 0   # the trapper fainted: the trap no longer holds either
        self.assertEqual(self.switches(wrapped._replace(hp=hp), 1), [True] * 5)
        detached = NS(own={'active': np.asarray(0), 'trap_source': np.asarray([0, -1, -1, -1, -1, -1], np.int32)})
        linked = protocol._write_own_meanlook(world, detached, True, 0)
        self.assertEqual(self.switches(linked, 0), [False] * 5);self.assertEqual(self.switches(linked, 1), [True] * 5)
        self.assertEqual(self.switches(linked._replace(active=np.asarray([[0, 4]], np.int32)), 0), [True] * 5)


if __name__ == '__main__':
    unittest.main()
