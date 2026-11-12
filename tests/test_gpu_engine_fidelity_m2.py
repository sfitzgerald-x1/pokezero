"""Fidelity patch M2: Perish Song, Yawn, Wrap (partial trap) and the residual scheduler, on authored states.

The rules follow Showdown f76228a1 as cited in gpu_engine/transition.py (data/moves.ts perishsong, yawn, wrap; the gen4/gen5
partiallytrapped condition; sim/battle.ts fieldEvent); the two-engine probes (gpu_engine/effect_probes/{perishsong,yawn,wrap}*.json)
are the Showdown gate. Light by design (heavy-test rule): one jitted step_with_chance at B=8 shared by every transition test
(table and arrays are arguments, compiled once), eager device hooks against B2's public_active fold (no step_with_public
compile), and pure-Python / numpy checks for the tables, future_state, the scheduler order and the regression rule.
"""
import unittest

import numpy as np
import jax
import jax.numpy as j

import test_gpu_engine_device_observation_boundary_parity as bp

TYPES = ['_none', 'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug',
         'rock', 'ghost', 'dragon', 'dark', 'steel']
NORMAL, POISON, GHOST = 1, 8, 14
MOVES = ['_none', 'struggle', 'splash', 'perishsong', 'yawn', 'wrap', 'tackle', 'protect', 'substitute', 'rapidspin', 'batonpass',
         'explosion', 'toxic']
STRUGGLE, SPLASH, PERISH, YAWN, WRAP, TACKLE, PROTECT, SUB, SPIN, BATON, EXPLOSION, TOXIC = range(1, 13)
ABILITIES = ['_none', 'soundproof', 'insomnia', 'vitalspirit']
SOUNDPROOF, INSOMNIA, VITALSPIRIT = range(1, 4)
ITEMS = ['_none', 'leftovers', 'lumberry']
LEFTOVERS, LUM = 1, 2
PAR, PSN, TOX, SLP = 1, 3, 4, 5
N = 8          # every transition test runs one fixed batch shape: one compile


def metadata():
    def row(name, power, typ, category, accuracy=100, flags=None, priority=0):
        return dict(id=name, base_power=power, type=typ, source_type=TYPES[typ].title(), category=category, pp=10,
                    accuracy=accuracy, priority=priority, flags={'protect': 1} if flags is None else flags)
    rows = [None, row('struggle', 50, NORMAL, 'Physical', True, {'contact': 1}), row('splash', 0, NORMAL, 'Status', True, {}),
            row('perishsong', 0, NORMAL, 'Status', True, {'sound': 1, 'distance': 1}),
            row('yawn', 0, NORMAL, 'Status', True, {'protect': 1, 'reflectable': 1, 'mirror': 1}),
            row('wrap', 15, NORMAL, 'Physical', 85, {'contact': 1, 'protect': 1, 'mirror': 1}),
            row('tackle', 35, NORMAL, 'Physical', 95, {'contact': 1, 'protect': 1}),
            row('protect', 0, NORMAL, 'Status', True, {}, priority=3), row('substitute', 0, NORMAL, 'Status', True, {}),
            row('rapidspin', 20, NORMAL, 'Physical', 100, {'contact': 1, 'protect': 1}), row('batonpass', 0, NORMAL, 'Status', True, {}),
            row('explosion', 250, NORMAL, 'Physical', 100, {'protect': 1}), row('toxic', 0, POISON, 'Status', 85, {'protect': 1})]
    chart = np.ones((17, 17));chart[NORMAL - 1, GHOST - 1] = 0
    return {'vocabulary': dict(moves=MOVES, abilities=ABILITIES, items=ITEMS, species=['_none', 'a', 'b', 'c', 'd', 'e', 'f'],
                               types=TYPES), 'moves': rows, 'type_chart': chart.tolist()}


class TableTests(unittest.TestCase):
    """Tables, the partial-trap legal mask, future_state and the chance inventories: no step compile."""
    def test_move_flags(self):
        from gpu_engine import effects
        t = effects.from_metadata(metadata())
        self.assertEqual(effects.MoveEffects._fields[-3:], ('perishsong', 'yawn', 'wrap'))
        for name, move in (('perishsong', PERISH), ('yawn', YAWN), ('wrap', WRAP)):
            self.assertEqual(np.flatnonzero(np.asarray(getattr(t, name))).tolist(), [move])

    def test_partial_trap_blocks_switches_only_while_its_source_is_the_living_foe_active(self):
        """onTrapPokemon (data/conditions.ts partiallytrapped) traps only while effectState.source is active; a fainted
        source is inactive; a forced replacement ignores it."""
        from gpu_engine.state import empty_state, legal_actions, partially_trapped
        s = empty_state(5)
        s = s._replace(hp=j.full_like(s.hp, 100), done=j.zeros(5, bool), pp=j.full_like(s.pp, 5),
                       partial_trap_counter=s.partial_trap_counter.at[:, 1, 0].set(3),
                       partial_trap_source=s.partial_trap_source.at[:, 1, 0].set(j.asarray([0, 2, 0, 0, 0])))
        s = s._replace(hp=s.hp.at[2, 0, 0].set(0), partial_trap_counter=s.partial_trap_counter.at[3, 1, 0].set(0),
                       forced_switch=s.forced_switch.at[4, 1].set(True))
        self.assertEqual(np.asarray(partially_trapped(s))[:, 1].tolist(), [True, False, False, False, True])
        switches = np.asarray(legal_actions(s))[:, 1, 4:].any(-1)
        self.assertEqual(switches.tolist(), [False, True, True, True, True])
        self.assertFalse(bool(np.asarray(partially_trapped(s))[:, 0].any()))

    def test_future_state_policy_shapes_and_refusals(self):
        from gpu_engine import future_state as f
        from gpu_engine.state import BattleState
        names = ('perish_counter', 'yawn_counter', 'partial_trap_counter', 'partial_trap_source')
        self.assertEqual(BattleState._fields[-4:], names)
        for name in names:
            self.assertIn(name, f.FIELD_POLICY);self.assertIn(name, f.NATIVE_PATHS);self.assertEqual(f.SHAPES[name], (2, 6))
        source = open(f.__file__).read()
        for code in ('PERISH_UNREPRESENTED', 'YAWN_UNREPRESENTED', 'PARTIAL_TRAP_UNREPRESENTED'):self.assertIn(code, source)

    def test_chance_inventories_name_the_new_consumers(self):
        from gpu_engine import chance_consumers, phase_d_chance
        for module in (chance_consumers, phase_d_chance):
            c = module.CONSUMERS
            for s in range(2):
                self.assertIn('Wrap duration 3+draw', c[f'draws.{s}.7'][0])
                self.assertIn('Yawn sleep duration 2+draw', c[f'effect.{s}.2'][0])
            self.assertIn('partial-trap', c['effect.0.3'][0]);self.assertIn('Perish', c['effect.1.3'][0])


class SchedulerTests(unittest.TestCase):
    """Design section 10: the 14-event order-10 list keeps the old 10 events' relative order (argsort equality)."""
    def test_old_events_keep_their_order_under_the_new_keys(self):
        rng = np.random.default_rng(20261111);lanes = 4096
        speed = rng.integers(1, 400, (lanes, 2));speed[: lanes // 4, 1] = speed[: lanes // 4, 0]        # a quarter tied
        speed[lanes // 8: lanes // 4] = rng.integers(1, 4, (lanes // 8, 2))                         # near ties
        coins = rng.integers(0, 2, (lanes, 7));sides = np.arange(2)[None, :]
        old_sides = np.array([0, 1] * 5);old_sub = np.array([3, 3, 4, 4, 5, 5, 6, 6, 14, 14])
        old_tie = np.concatenate([coins[:, k, None] ^ sides for k in (0, 1, 2, 3, 5)], axis=1)
        old_keys = -speed[:, old_sides] * 32 + old_sub[None, :] * 2 + old_tie
        new_sides = np.array([0, 1] * 7);new_sub = np.array([3, 3, 4, 4, 5, 5, 6, 6, 9, 9, 14, 14, 19, 19])
        new_tie = np.concatenate([coins[:, k, None] ^ sides for k in (0, 1, 2, 3, 4, 5)] + [np.broadcast_to(sides, (lanes, 2))], axis=1)
        new_keys = -speed[:, new_sides] * 64 + new_sub[None, :] * 2 + new_tie
        for keys in (old_keys, new_keys):self.assertTrue(all(len(set(row)) == len(row) for row in keys.tolist()))   # distinct
        old_order = np.asarray(j.argsort(j.asarray(old_keys), axis=1))
        new_order = np.asarray(j.argsort(j.asarray(new_keys), axis=1))
        # Old event index -> new index: 0-7 unchanged, Encore 8,9 -> 10,11.
        remap = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10, 11])
        restricted = np.array([[e for e in row if e in set(remap.tolist())] for row in new_order.tolist()])
        np.testing.assert_array_equal(restricted, remap[old_order])
        # The new events sit by Speed, then subOrder: a side's wrap tick (9) after its status (6), before its Encore (14).
        first = np.where(speed[:, 0] > speed[:, 1], 0, 1)
        lane = int(np.flatnonzero(speed[:, 0] != speed[:, 1])[0]);order = new_order[lane].tolist();f = int(first[lane])
        self.assertLess(order.index(6 + f), order.index(8 + f));self.assertLess(order.index(8 + f), order.index(10 + f))
        self.assertLess(order.index(10 + f), order.index(12 + f))


class TransitionTests(unittest.TestCase):
    """step_with_chance on authored 8-lane states: side 0 (faster) uses its slot-0 move, side 1 its slot 0 (Splash by
    default). One jit, the table and every array as arguments, compiled once (asserted last)."""
    @classmethod
    def setUpClass(cls):
        from gpu_engine import transition
        from gpu_engine.state import BattleState, empty_state
        cls.t, cls.State, cls.empty = transition, BattleState, staticmethod(empty_state)
        cls.table = transition.tables_from_metadata(metadata())
        # A dict, not a class attribute: a jitted function is a descriptor and would bind the test instance.
        cls.jitted = dict(step=jax.jit(transition.step_with_chance))

    def build(self, lanes, move=SPLASH, foe_move=SPLASH):
        s = self.empty(N, seed=4246)
        s = s._replace(hp=j.full_like(s.hp, 320), max_hp=j.full_like(s.max_hp, 320),
            stats=j.full_like(s.stats, 200).at[..., 0].set(320).at[:, 0, :, 5].set(301),
            level=j.full_like(s.level, 100), types=s.types.at[..., 0].set(NORMAL), base_types=s.base_types.at[..., 0].set(NORMAL),
            species=j.broadcast_to(j.arange(1, 7, dtype=j.int32), (N, 2, 6)),
            moves=j.full_like(s.moves, SPLASH).at[:, 0, :, 0].set(move).at[:, 1, :, 0].set(foe_move),
            pp=j.full_like(s.pp, 10), done=j.zeros(N, bool))
        arrays = {k: np.array(v) for k, v in s._asdict().items()}
        assert len(lanes) <= N
        for lane, edits in enumerate(lanes):
            for field, writes in edits.items():
                for index, value in writes:arrays[field][(lane,) + tuple(index)] = value
        arrays['base_types'] = arrays['types'].copy()
        return self.State(**{k: j.asarray(v, dtype=getattr(s, k).dtype) for k, v in arrays.items()})

    def chance(self, draws=(), effect=()):
        """Hit, no crit, max damage roll, no thaw/paralysis/self-hit, duration roll 0, Protect succeeds, effect 0."""
        d = np.zeros((N, 2, 12), np.int32);d[:, :, 1:3] = 99;d[:, :, 4] = 15;d[:, :, 5] = 15
        e = np.zeros((N, 2, 4), np.int32);e[:, :, 0] = 15
        for lane, edits in enumerate(draws):
            for (side, column), value in edits.items():d[lane, side, column] = value
        for lane, edits in enumerate(effect):
            for (side, column), value in edits.items():e[lane, side, column] = value
        z = j.zeros((N, 2), j.int32)
        return self.t.ChanceInputs(draws=j.asarray(d), phazing=z, secondary=j.full((N, 2), 99, j.int32), sleep_talk=z, entry_ties=z,
                                   initial_tie=j.zeros(N, j.int32), effect=j.asarray(e))

    def play(self, s, chance, a0=0, a1=0):
        actions = j.zeros((N, 2), j.int32).at[:, 0].set(j.asarray(a0, j.int32)).at[:, 1].set(j.asarray(a1, j.int32))
        out = self.jitted['step'](s, actions, self.table, chance)
        self.assertFalse(bool(np.asarray(out.invalid_actions).any()), np.asarray(out.invalid_actions))
        return out.state

    def at(self, s, name, side, mon=0):
        return np.asarray(getattr(s, name))[:, side, mon]

    def test_perish_song_counts_soundproof_reuse_and_both_faints(self):
        """Lanes: 0 both count (4, then 3 after order 12); 1 a Soundproof foe and 2 a Soundproof user are -immune; 3 the foe
        already counts (no reset: 2 -> 1); 4 the user sleeps (no move). Lanes 5-7 hold count 1 on both (Splash): both
        perish0 ends resolve before faintMessages, so 6 (side 0 at its last mon) still faints side 1 too (side 1 wins with
        its bench) and 7 (both last mons) is a tie."""
        bench_dead = [((side, k), 0) for side in (0, 1) for k in range(1, 6)]
        lanes = [{}, {'ability': [((1, 0), SOUNDPROOF)]}, {'ability': [((0, 0), SOUNDPROOF)]}, {'perish_counter': [((1, 0), 2)]},
                 {'status': [((0, 0), SLP)], 'status_counter': [((0, 0), 3)]},
                 {'perish_counter': [((0, 0), 1), ((1, 0), 1)], 'moves': [((0, 0, 0), SPLASH)]},
                 {'perish_counter': [((0, 0), 1), ((1, 0), 1)], 'moves': [((0, 0, 0), SPLASH)], 'hp': bench_dead[:5]},
                 {'perish_counter': [((0, 0), 1), ((1, 0), 1)], 'moves': [((0, 0, 0), SPLASH)], 'hp': bench_dead}]
        s = self.play(self.build(lanes, move=PERISH), self.chance())
        self.assertEqual(self.at(s, 'perish_counter', 0).tolist()[:5], [3, 3, 0, 3, 0])
        self.assertEqual(self.at(s, 'perish_counter', 1).tolist()[:5], [3, 0, 3, 1, 0])
        hp = np.asarray(s.hp);self.assertTrue((hp[5:, :, 0] == 0).all())
        self.assertEqual(np.asarray(s.perish_counter)[5:].max(), 0)                     # faint clears
        self.assertEqual(np.asarray(s.winner)[5:].tolist(), [-1, 1, 2]);self.assertEqual(np.asarray(s.done)[5:].tolist(), [False, True, True])

    def test_perish_yawn_and_trap_follow_switches_and_baton_pass(self):
        """Lane 0: the foe switches out (switchOut's clearVolatile ends perish and yawn; a trapped foe could not switch).
        Lane 1: the foe Baton Passes (copyVolatileFrom carries perishsong and partiallytrapped with its source; yawn is
        noCopy): the replacement request installs the recipient, then the residual ticks the copies."""
        held = {'perish_counter': [((0, 0), 3), ((1, 0), 3)], 'yawn_counter': [((1, 0), 1)],
                'partial_trap_counter': [((1, 0), 4)], 'partial_trap_source': [((1, 0), 0)]}
        free = {'perish_counter': [((0, 0), 3), ((1, 0), 3)], 'yawn_counter': [((1, 0), 1)]}
        lanes = [free, dict(held, moves=[((1, 0, 0), BATON)])] + [{}] * 6
        s = self.play(self.build(lanes), self.chance(), a0=1, a1=j.asarray([4, 0, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(np.asarray(s.perish_counter)[0].tolist(), [[2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        self.assertEqual(int(np.asarray(s.yawn_counter)[0].sum()), 0)
        self.assertEqual((int(np.asarray(s.partial_trap_counter)[0].sum()), int(np.asarray(s.partial_trap_source)[0, 1].max())), (0, -1))
        self.assertTrue(bool(np.asarray(s.pending_residual)[1]))                          # the Baton Pass waits for a target
        s = self.play(s, self.chance(), a0=0, a1=j.asarray([0, 4, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(int(np.asarray(s.active)[1, 1]), 1)
        self.assertEqual(np.asarray(s.perish_counter)[1, :, :2].tolist(), [[2, 0], [0, 2]])          # 3 -> 2 at this residual
        self.assertEqual(np.asarray(s.yawn_counter)[1, 1, :2].tolist(), [0, 0])
        self.assertEqual(np.asarray(s.partial_trap_counter)[1, 1, :2].tolist(), [0, 3])          # copied 4, ticked to 3
        self.assertEqual(np.asarray(s.partial_trap_source)[1, 1, :2].tolist(), [-1, 0])          # the original trapper
        self.assertEqual(int(np.asarray(s.hp)[1, 1, 1]), 320 - 20)

    def test_yawn_lands_only_where_showdown_lets_it(self):
        """Lanes: 0 plain (2 -> 1); 1 a paralysed foe; 2 Insomnia and 3 Vital Spirit (-immune); 4 a Substitute; 5 the foe's
        Protect; 6 the foe is already drowsy (no reset: it sleeps at this residual, 2+effect[1][2]); 7 a Ghost (a Status move
        ignores type immunity: it lands)."""
        lanes = [{}, {'status': [((1, 0), PAR)]}, {'ability': [((1, 0), INSOMNIA)]}, {'ability': [((1, 0), VITALSPIRIT)]},
                 {'substitute_hp': [((1, 0), 80)]}, {'moves': [((1, 0, 0), PROTECT)]}, {'yawn_counter': [((1, 0), 1)]},
                 {'types': [((1, 0, 0), GHOST)]}]
        s = self.play(self.build(lanes, move=YAWN), self.chance(effect=[{}] * 6 + [{(1, 2): 2}]))
        self.assertEqual(self.at(s, 'yawn_counter', 1).tolist(), [1, 0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(self.at(s, 'status', 1).tolist(), [0, PAR, 0, 0, 0, 0, SLP, 0])
        self.assertEqual(int(self.at(s, 'status_counter', 1)[6]), 4)

    def test_yawn_sleep_at_its_end(self):
        """A drowsy foe (count 1) at the residual: 0 sleeps (2+effect[1][2]); 1 a paralysed holder stays paralysed; 2 Sleep
        Clause (a benched foe-made sleeper) refuses it; 3 a Rest sleeper and 5 a fainted sleeper do not count; 4 a Lum Berry
        cures it at once; 6 both sides drowsy (the inert pair) both sleep; 7 effect[1][2]=3."""
        lanes = [{}, {'status': [((1, 0), PAR)]}, {'status': [((1, 2), SLP)], 'status_counter': [((1, 2), 2)]},
                 {'status': [((1, 2), SLP)], 'status_counter': [((1, 2), 2)], 'rest_origin': [((1, 2), True)]},
                 {'item': [((1, 0), LUM)]}, {'status': [((1, 2), SLP)], 'hp': [((1, 2), 0)]},
                 {'yawn_counter': [((0, 0), 1)]}, {}]
        lanes = [dict(edits, yawn_counter=edits.get('yawn_counter', []) + [((1, 0), 1)]) for edits in lanes]
        s = self.play(self.build(lanes), self.chance(effect=[{}] * 6 + [{(0, 2): 1, (1, 2): 1}, {(1, 2): 3}]))
        self.assertEqual(self.at(s, 'status', 1).tolist(), [SLP, PAR, 0, SLP, 0, SLP, SLP, SLP])
        self.assertEqual(self.at(s, 'status_counter', 1).tolist(), [2, 0, 0, 2, 0, 2, 3, 5])
        self.assertFalse(bool(self.at(s, 'rest_origin', 1).any()))
        self.assertEqual(int(self.at(s, 'item', 1)[4]), 0)                                        # Lum eaten
        self.assertEqual((int(self.at(s, 'status', 0)[6]), int(self.at(s, 'status_counter', 0)[6])), (SLP, 3))
        self.assertEqual(int(np.asarray(s.yawn_counter).sum()), 0)

    def test_wrap_duration_tick_leftovers_and_failures(self):
        """Lanes 0-3: duration 3+draws[0][7] (0..3), one tick this turn (d-1 left). 4 a miss; 5 Leftovers heals before the
        tick (subOrder 4 < 9); 6 a Ghost is immune; 7 a Substitute takes the hit: neither traps."""
        lanes = [{}, {}, {}, {}, {}, {'item': [((1, 0), LEFTOVERS)]}, {'types': [((1, 0, 0), GHOST)]}, {'substitute_hp': [((1, 0), 80)]}]
        draws = [{(0, 7): 0}, {(0, 7): 1}, {(0, 7): 2}, {(0, 7): 3}, {(0, 6): 99}, {}, {}, {}]
        s = self.play(self.build(lanes, move=WRAP), self.chance(draws=draws))
        self.assertEqual(self.at(s, 'partial_trap_counter', 1).tolist(), [2, 3, 4, 5, 0, 2, 0, 0])
        self.assertEqual(self.at(s, 'partial_trap_source', 1).tolist(), [0, 0, 0, 0, -1, 0, -1, -1])
        hp = self.at(s, 'hp', 1);dealt = 320 - int(hp[0]) - 20
        self.assertTrue(dealt > 0);self.assertTrue((hp[:4] == hp[0]).all())
        self.assertEqual(int(hp[5]), min(320, 320 - dealt + 20) - 20)
        self.assertEqual(hp[4:].tolist()[0], 320);self.assertEqual(int(hp[6]), 320);self.assertEqual(int(hp[7]), 320)

    def test_wrap_expiry_silent_release_spin_substitute_and_tick_ko(self):
        """A trap carried in (source: side 0's active): 0 count 1 ends (no tick); 1 ticks (3 -> 2, 20 HP); 2 a source that is
        not the foe's active releases silently; 3 the trapper switched out this turn: released; 4 Rapid Spin and 5 the
        holder's own Substitute end it before the residual; 6 a lethal tick faints the holder (its leaves cleared); 7 a
        Speed tie both ways (no new trap): unaffected by the trap."""
        lanes = [{'partial_trap_counter': [((1, 0), c)], 'partial_trap_source': [((1, 0), src)]}
                 for c, src in ((1, 0), (3, 0), (3, 3), (3, 0), (3, 0), (3, 0), (3, 0), (0, -1))]
        lanes[4]['moves'] = [((1, 0, 0), SPIN)];lanes[5]['moves'] = [((1, 0, 0), SUB)];lanes[6]['hp'] = [((1, 0), 15)]
        s = self.play(self.build(lanes), self.chance(), a0=j.asarray([1, 1, 1, 4, 1, 1, 1, 1]))
        self.assertEqual(self.at(s, 'partial_trap_counter', 1).tolist(), [0, 2, 0, 0, 0, 0, 0, 0])
        self.assertEqual(self.at(s, 'partial_trap_source', 1).tolist(), [-1, 0, -1, -1, -1, -1, -1, -1])
        self.assertEqual(self.at(s, 'hp', 1).tolist(), [320, 300, 320, 320, 320, 240, 0, 320])
        self.assertTrue(bool(np.asarray(s.forced_switch)[6, 1]))

    def test_wrap_releases_for_a_trapper_fainted_in_this_residual_or_new_this_turn(self):
        """M2 review (RW8, RW9): the holder's handler releases silently, with no tick, when its trapper is no longer held
        (data/mods/gen5/conditions.ts onResidual: source.hp <= 0 or !source.activeTurns). Lane 0: the faster trapper, at
        1 HP and poisoned, faints at its own subOrder-6 poison earlier in this residual. Lane 1: control, it ticks (3 -> 2,
        320/16 = 20). Lane 2: the source switches in this turn (activeTurns 0). Lane 3: a slower trapper fainting at its own
        poison after the holder's tick: the tick stands, the trap stays until the next residual."""
        trap = lambda src=0: {'partial_trap_counter': [((1, 0), 3)], 'partial_trap_source': [((1, 0), src)]}
        lanes = [dict(trap(), hp=[((0, 0), 1)], status=[((0, 0), PSN)]), trap(), trap(2),
                 dict(trap(), hp=[((0, 0), 1)], status=[((0, 0), PSN)], stats=[((0, 0, 5), 100)])] + [{}] * 4
        s = self.play(self.build(lanes), self.chance(), a0=j.asarray([0, 0, 5, 0, 0, 0, 0, 0]))
        self.assertEqual(self.at(s, 'hp', 0)[[0, 3]].tolist(), [0, 0])                 # both poisoned trappers fainted
        self.assertEqual(int(np.asarray(s.active)[2, 0]), 2)                            # lane 2: the source is the new active
        self.assertEqual(self.at(s, 'partial_trap_counter', 1)[:4].tolist(), [0, 2, 0, 2])
        self.assertEqual(self.at(s, 'partial_trap_source', 1)[:4].tolist(), [-1, 0, -1, 0])
        self.assertEqual(self.at(s, 'hp', 1)[:4].tolist(), [320, 300, 320, 300])

    def test_residual_pair_ties(self):
        """Equal Speed, both last mons. Lanes 0/1: both trapped with lethal ticks; the subOrder-9 pair takes effect[0][3]:
        the first tick's KO ends the game, so the coin decides the winner, and the survivor's handler never runs. Lanes 2/3:
        both at Perish 1: both resolve, a tie either way. Lane 4: both drowsy: both sleep (fixed order, inert)."""
        last = [((side, k), 0) for side in (0, 1) for k in range(1, 6)]
        trap = {'partial_trap_counter': [((0, 0), 3), ((1, 0), 3)], 'partial_trap_source': [((0, 0), 0), ((1, 0), 0)],
                'hp': last + [((0, 0), 15), ((1, 0), 15)], 'stats': [((0, 0, 5), 200)]}
        perish = {'perish_counter': [((0, 0), 1), ((1, 0), 1)], 'hp': last, 'stats': [((0, 0, 5), 200)]}
        yawn = {'yawn_counter': [((0, 0), 1), ((1, 0), 1)], 'stats': [((0, 0, 5), 200)]}
        lanes = [trap, trap, perish, perish, yawn, {}, {}, {}]
        s = self.play(self.build(lanes), self.chance(effect=[{(0, 3): 0}, {(0, 3): 1}, {(1, 3): 0}, {(1, 3): 1}]))
        self.assertEqual(np.asarray(s.winner)[:4].tolist(), [1, 0, 2, 2])
        self.assertEqual(np.asarray(s.hp)[0, :, 0].tolist(), [0, 15]);self.assertEqual(np.asarray(s.hp)[1, :, 0].tolist(), [15, 0])
        self.assertEqual(int(np.asarray(s.partial_trap_counter)[0, 1, 0]), 3)           # the survivor never ticked
        self.assertEqual(np.asarray(s.status)[4, :, 0].tolist(), [SLP, SLP])

    def test_faint_clears_every_m2_leaf(self):
        """faintMessages -> clearVolatile: a KO'd holder loses perish, yawn and its trap (design 2.4)."""
        lanes = [{'hp': [((1, 0), 1)], 'perish_counter': [((1, 0), 3)], 'yawn_counter': [((1, 0), 1)],
                  'partial_trap_counter': [((1, 0), 3)], 'partial_trap_source': [((1, 0), 0)]}] + [{}] * 7
        s = self.play(self.build(lanes, move=TACKLE), self.chance())
        self.assertEqual(int(self.at(s, 'hp', 1)[0]), 0)
        for name, value in (('perish_counter', 0), ('yawn_counter', 0), ('partial_trap_counter', 0), ('partial_trap_source', -1)):
            self.assertEqual(int(self.at(s, name, 1)[0]), value, name)

    def test_jitted_step_equals_an_eager_step(self):
        """Positive control for the shared compiled step: one lane set, eager against jitted, every leaf."""
        lanes = [{'partial_trap_counter': [((1, 0), 3)], 'partial_trap_source': [((1, 0), 0)], 'yawn_counter': [((0, 0), 1)]}] * N
        s = self.build(lanes, move=PERISH);chance = self.chance(draws=[{(0, 7): k % 4} for k in range(N)])
        actions = j.zeros((N, 2), j.int32)
        fast = self.jitted['step'](s, actions, self.table, chance).state
        with jax.disable_jit():
            slow = self.t.step_with_chance(s, actions, self.table, chance).state
        for name in s._fields:np.testing.assert_array_equal(np.asarray(getattr(fast, name)), np.asarray(getattr(slow, name)), err_msg=name)

    def test_zz_one_compile(self):
        """Every transition test used the one compiled step (no recompile)."""
        self.assertEqual(self.jitted['step']._cache_size(), 1)


class DeviceB2ParityTests(unittest.TestCase):
    """The device's M2 producers (eager hooks) against B2's public_active fold of the same Showdown lines (design 2.5):
    perishN, yawn and partiallytrapped bag entries and WRAP_TRAP_TURNS (column 33)."""
    setUpClass = classmethod(bp.DeviceBoundaryParityTests.setUpClass.__func__)
    state = bp.DeviceBoundaryParityTests.state
    b2_active = bp.DeviceBoundaryParityTests.b2_active
    compare_active = bp.DeviceBoundaryParityTests.compare_active
    b2_lines = bp.DeviceBoundaryParityTests.b2_lines

    def setUp(self):
        from gpu_engine import engine_device_public_hooks as h
        self.h = h
        self.a, self.catalog, self.fact, self.project = self.b2_active()
        self.base = self.state();self.c = h.initialize(self.base, self.tables.belief);self.yes = j.ones(2, bool)
        self.fact(self.a.ARRIVE, 0);self.fact(self.a.ARRIVE, 1);self.turn = 0

    def with_(self, state, **counters):
        out = state
        for name, writes in counters.items():
            value = getattr(out, name)
            for (side, mon), v in writes:value = value.at[:, side, mon].set(v)
            out = out._replace(**{name: value})
        return out

    def end_turn(self, before, after, *lines):
        self.b2_lines(self.a, self.catalog, self.fact, *lines)
        self.h.turn_end(self.c, before, after, self.yes);self.turn += 1;self.fact(self.a.TURN, -1, self.turn)
        return self.check()

    def check(self):
        got = self.project();self.compare_active(self.c.trackers, got, self.a);return got

    def clock(self, got, side):
        return float(np.asarray(got.numeric)[0, side, self.a.NUMERIC_COLUMNS[self.a.CORE.index('WRAP_TRAP_TURNS')]])

    def test_perish_bag_follows_perish_start_and_the_faint(self):
        sung = self.with_(self.base, perish_counter=[((0, 0), 4), ((1, 0), 4)])
        self.h.public_move_state(self.c, self.base, sung, 0, j.full(2, PERISH, j.int32), self.yes)
        self.b2_lines(self.a, self.catalog, self.fact, '|-start|p1a: Snorlax|perish3|[silent]', '|-start|p2a: Absol|perish3|[silent]')
        self.check()
        state = sung
        for n in (3, 2, 1):
            after = self.with_(sung, perish_counter=[((0, 0), n), ((1, 0), n)])
            got = self.end_turn(state, after, f'|-start|p1a: Snorlax|perish{n}', f'|-start|p2a: Absol|perish{n}');state = after
            self.assertTrue(bool(np.asarray(got.categorical)[0, 0, j.asarray(self.a.CATEGORY_COLUMNS)].any()))
        gone = self.with_(state, perish_counter=[((0, 0), 0), ((1, 0), 0)], hp=[((0, 0), 0), ((1, 0), 0)])
        self.b2_lines(self.a, self.catalog, self.fact, '|-start|p1a: Snorlax|perish0', '|-start|p2a: Absol|perish0')
        self.fact(self.a.FAINT, 0);self.fact(self.a.FAINT, 1)
        self.h.turn_end(self.c, state, gone, j.zeros(2, bool));got = self.check()
        self.assertFalse(bool(np.asarray(got.categorical)[0, :, j.asarray(self.a.CATEGORY_COLUMNS)].any()))

    def test_baton_pass_carries_perish_and_the_partial_trap_with_its_clock(self):
        held = self.with_(self.base, perish_counter=[((1, 0), 3)], partial_trap_counter=[((1, 0), 4)], partial_trap_source=[((1, 0), 0)])
        self.h.public_move_state(self.c, self.base, held, 0, j.full(2, WRAP, j.int32), self.yes)
        self.b2_lines(self.a, self.catalog, self.fact, '|-start|p2a: Absol|perish3|[silent]', '|-activate|p2a: Absol|move: Wrap|[of] p1a: Snorlax')
        got = self.end_turn(held, held, '|-start|p2a: Absol|perish3');self.assertAlmostEqual(self.clock(got, 1), 0.2)
        self.fact(self.a.ARRIVE, 1, slot=1, bp=True)
        passed = self.with_(held, perish_counter=[((1, 1), 3)], partial_trap_counter=[((1, 1), 4)], partial_trap_source=[((1, 1), 0)])
        passed = passed._replace(active=passed.active.at[:, 1].set(1))
        self.h.install(self.c, passed, 1, self.yes);self.h.baton_copy(self.c, passed, 1, held.active[:, 1], self.yes)
        self.h.withdraw(self.c, passed, 1, held.active[:, 1], self.yes)
        got = self.check();self.assertAlmostEqual(self.clock(got, 1), 0.2)
        got = self.end_turn(passed, passed, '|-start|p2a: Gardevoir|perish3');self.assertAlmostEqual(self.clock(got, 1), 0.4)

    def test_yawn_bag_follows_start_and_the_silent_end(self):
        drowsy = self.with_(self.base, yawn_counter=[((1, 0), 2)])
        self.h.public_move_state(self.c, self.base, drowsy, 0, j.full(2, YAWN, j.int32), self.yes)
        self.b2_lines(self.a, self.catalog, self.fact, '|-start|p2a: Absol|move: Yawn|[of] p1a: Snorlax');got = self.check()
        self.assertTrue(bool(np.asarray(got.categorical)[0, 1, j.asarray(self.a.CATEGORY_COLUMNS)].any()))
        one = self.with_(drowsy, yawn_counter=[((1, 0), 1)]);self.end_turn(drowsy, one)
        asleep = self.with_(one, yawn_counter=[((1, 0), 0)], status=[((1, 0), SLP)], status_counter=[((1, 0), 2)])
        got = self.end_turn(one, asleep, '|-end|p2a: Absol|move: Yawn|[silent]', '|-status|p2a: Absol|slp')
        self.assertFalse(bool(np.asarray(got.categorical)[0, 1, j.asarray(self.a.CATEGORY_COLUMNS)].any()))

    def test_partial_trap_bag_and_clock_follow_activate_turn_and_every_end(self):
        """-activate adds the entry (ADD); each TURN adds a turn while held; -end (expiry, a silent release, Rapid Spin, a
        Substitute) is REMOVE + WRAP_END; a faint clears both."""
        trap = lambda n: self.with_(self.base, partial_trap_counter=[((1, 0), n)], partial_trap_source=[((1, 0), 0 if n else -1)])
        self.h.public_move_state(self.c, self.base, trap(5), 0, j.full(2, WRAP, j.int32), self.yes)
        self.b2_lines(self.a, self.catalog, self.fact, '|-activate|p2a: Absol|move: Wrap|[of] p1a: Snorlax')
        self.assertEqual(self.clock(self.check(), 1), 0.0)
        self.assertAlmostEqual(self.clock(self.end_turn(trap(5), trap(4)), 1), 0.2)
        self.assertAlmostEqual(self.clock(self.end_turn(trap(4), trap(3)), 1), 0.4)
        # Rapid Spin by the holder mid-step: removeVolatile's -end resets the clock at once.
        self.h.public_move_state(self.c, trap(3), trap(0), 1, j.full(2, SPIN, j.int32), self.yes)
        self.b2_lines(self.a, self.catalog, self.fact, '|-end|p2a: Absol|Wrap|[partiallytrapped]')
        self.assertEqual(self.clock(self.check(), 1), 0.0)
        self.end_turn(trap(0), trap(0))
        # Trapped again; the expiry at a residual (-end, not silent) and then a silent release both reset.
        self.h.public_move_state(self.c, trap(0), trap(3), 0, j.full(2, WRAP, j.int32), self.yes)
        self.b2_lines(self.a, self.catalog, self.fact, '|-activate|p2a: Absol|move: Wrap|[of] p1a: Snorlax')
        self.end_turn(trap(3), trap(2));self.end_turn(trap(2), trap(1))
        self.assertEqual(self.clock(self.end_turn(trap(1), trap(0), '|-end|p2a: Absol|Wrap|[partiallytrapped]'), 1), 0.0)
        self.h.public_move_state(self.c, trap(0), trap(4), 0, j.full(2, WRAP, j.int32), self.yes)
        self.b2_lines(self.a, self.catalog, self.fact, '|-activate|p2a: Absol|move: Wrap|[of] p1a: Snorlax')
        self.end_turn(trap(4), trap(3))
        self.assertEqual(self.clock(self.end_turn(trap(3), trap(0), '|-end|p2a: Absol|Wrap|[partiallytrapped]|[silent]'), 1), 0.0)
        # Trapped, then KO'd: FAINT clears the entry and the clock.
        self.h.public_move_state(self.c, trap(0), trap(5), 0, j.full(2, WRAP, j.int32), self.yes)
        self.b2_lines(self.a, self.catalog, self.fact, '|-activate|p2a: Absol|move: Wrap|[of] p1a: Snorlax')
        self.end_turn(trap(5), trap(4))
        self.fact(self.a.FAINT, 1)
        self.h.turn_end(self.c, trap(4), trap(0)._replace(hp=self.base.hp.at[:, 1, 0].set(0)), j.zeros(2, bool))
        self.assertEqual(self.clock(self.check(), 1), 0.0)


class RegressRuleTests(unittest.TestCase):
    """v2e_regress.m2_rule on one lane's arrays (numpy): each counter needs its own post-state witness."""
    def lane(self, **over):
        z = lambda v=0: np.full((2, 6), v, np.int32)
        w = dict(perish=z(), yawn=z(), trap=z(), source=z(-1), ability=z(), clock=z(), bag=np.zeros((2, 6, 6), bool),
                 p2p=np.broadcast_to(np.arange(6), (2, 6)).copy(), post_active=np.zeros(2, np.int32), post_hp=z(100),
                 done=False, ran=True, ordinary=True, perish_used=np.zeros(2, bool), yawn_by=np.zeros(2, bool),
                 yawn_target=np.full(2, -1), blocked=np.zeros(2, bool), wrap_by=np.zeros(2, bool), wrap_user=np.full(2, -1),
                 wrap_target=np.full(2, -1), draws7=np.zeros(2, np.int32), soundproof={SOUNDPROOF}, sleep_immune={INSOMNIA, VITALSPIRIT},
                 active_turns=np.ones(2, np.int32), max_hp=z(320), base_hp=None)
        w.update(over);return w

    def rule(self, entries, w):
        from gpu_engine import v2e_regress as r
        return r.m2_rule(set(entries), w)

    def test_perish_needs_the_move_a_living_non_soundproof_active_the_value_and_its_bag_bit(self):
        perish = np.zeros((2, 6), np.int32);perish[:, 0] = 3;bag = np.zeros((2, 6, 6), bool);bag[:, 0, 3] = True
        entries = [('state.perish_counter', 0), ('state.perish_counter', 1)]
        rest, done, cells, _ = self.rule(entries, self.lane(perish=perish, bag=bag, perish_used=np.array([True, False])))
        self.assertEqual((rest, len(done)), (set(), 2));self.assertTrue(cells[:, 0, :4].all())
        self.assertEqual(len(self.rule(entries, self.lane(perish=perish, bag=bag))[0]), 2)                     # no move line
        ability = np.zeros((2, 6), np.int32);ability[1, 0] = SOUNDPROOF
        self.assertEqual(self.rule(entries, self.lane(perish=perish, bag=bag, ability=ability, perish_used=np.array([True, False])))[0],
                         {('state.perish_counter', 1)})
        self.assertEqual(len(self.rule(entries, self.lane(perish=perish, perish_used=np.array([True, False])))[0]), 2)   # no bag
        deferred = perish.copy();deferred[:, 0] = 4
        self.assertEqual(len(self.rule(entries, self.lane(perish=deferred, bag=bag, ran=False, perish_used=np.array([True, False])))[0]), 0)
        self.assertEqual(len(self.rule(entries, self.lane(perish=deferred, bag=bag, perish_used=np.array([True, False])))[0]), 2)

    def test_yawn_needs_an_unblocked_clear_line_at_that_target(self):
        yawn = np.zeros((2, 6), np.int32);yawn[1, 0] = 1;bag = np.zeros((2, 6, 6), bool);bag[1, 0, 4] = True
        ok = dict(yawn=yawn, bag=bag, yawn_by=np.array([True, False]), yawn_target=np.array([0, -1]))
        entries = [('state.yawn_counter', 1)]
        self.assertEqual(self.rule(entries, self.lane(**ok))[0], set())
        self.assertEqual(self.rule(entries, self.lane(**dict(ok, blocked=np.array([True, False]))))[0], set(entries))
        self.assertEqual(self.rule(entries, self.lane(**dict(ok, yawn_by=np.zeros(2, bool))))[0], set(entries))
        ability = np.zeros((2, 6), np.int32);ability[1, 0] = INSOMNIA
        self.assertEqual(self.rule(entries, self.lane(**dict(ok, ability=ability)))[0], set(entries))
        self.assertEqual(self.rule(entries, self.lane(**dict(ok, bag=np.zeros((2, 6, 6), bool))))[0], set(entries))

    def test_wrap_needs_hp_damage_the_duration_draw_the_source_and_the_clock(self):
        trap = np.zeros((2, 6), np.int32);trap[1, 0] = 4;source = np.full((2, 6), -1, np.int32);source[1, 0] = 2
        bag = np.zeros((2, 6, 6), bool);bag[1, 0, 5] = True;clock = np.zeros((2, 6), np.int32);clock[1, 0] = 1
        ok = dict(trap=trap, source=source, bag=bag, clock=clock, wrap_by=np.array([True, False]), wrap_user=np.array([2, -1]),
                  wrap_target=np.array([0, -1]), draws7=np.array([2, 0]), post_active=np.array([2, 0], np.int32))
        entries = [('state.partial_trap_counter', 1), ('state.partial_trap_source', 1), (r_clock(), 1)]
        self.assertEqual(self.rule(entries, self.lane(**ok))[0], set())
        fainted = np.full((2, 6), 100, np.int32);fainted[0, 2] = 0
        for bad in (dict(wrap_by=np.zeros(2, bool)), dict(draws7=np.array([1, 0])), dict(wrap_user=np.array([3, -1])),
                    dict(clock=np.zeros((2, 6), np.int32)), dict(bag=np.zeros((2, 6, 6), bool)), dict(ordinary=False),
                    # M2 review: a tick (4 = 3+2-1) needs the trapper held at the residual: still active, alive, activeTurns > 0.
                    dict(post_active=np.array([1, 0], np.int32)), dict(post_hp=fainted), dict(active_turns=np.array([0, 1], np.int32))):
            self.assertEqual(self.rule(entries, self.lane(**dict(ok, **bad)))[0], set(entries), bad)
        untick = dict(ok, trap=np.where(trap > 0, 5, 0), post_hp=fainted, ran=False)       # no residual: no tick, no trapper check
        self.assertEqual(self.rule(entries, self.lane(**untick))[0], set())

    def test_the_trapper_is_read_at_the_holders_handler_when_recorded(self):
        """M2 review (GPU, 40 lanes at 15f2af3d): a slower trapper fainting at its own residual after the holder's tick ends
        the step fainted; the handler cutpoint (tick_seen/tick_trapper/tick_held at the holder's side) shows it held at the
        tick, which stands. A trapper not held there, another mon there, or no handler run with a ticked count fail."""
        trap = np.zeros((2, 6), np.int32);trap[1, 0] = 4;source = np.full((2, 6), -1, np.int32);source[1, 0] = 2
        bag = np.zeros((2, 6, 6), bool);bag[1, 0, 5] = True;clock = np.zeros((2, 6), np.int32);clock[1, 0] = 1
        fainted = np.full((2, 6), 100, np.int32);fainted[0, 2] = 0
        ok = dict(trap=trap, source=source, bag=bag, clock=clock, wrap_by=np.array([True, False]), wrap_user=np.array([2, -1]),
                  wrap_target=np.array([0, -1]), draws7=np.array([2, 0]), post_active=np.array([2, 0], np.int32), post_hp=fainted,
                  tick_seen=np.array([False, True]), tick_trapper=np.array([-1, 2]), tick_held=np.array([False, True]))
        entries = [('state.partial_trap_counter', 1), ('state.partial_trap_source', 1), (r_clock(), 1)]
        self.assertEqual(self.rule(entries, self.lane(**ok))[0], set())
        for bad in (dict(tick_held=np.array([False, False])), dict(tick_trapper=np.array([-1, 3])), dict(tick_seen=np.array([False, False]))):
            self.assertEqual(self.rule(entries, self.lane(**dict(ok, **bad)))[0], set(entries), bad)
        without = {k: v for k, v in ok.items() if not k.startswith('tick_')}
        self.assertEqual(self.rule(entries, self.lane(**without))[0], set(entries))         # the step-end stand-in: fainted

    def test_mon_digest_is_the_base_pass_side_digest(self):
        from gpu_engine import v2e_regress as r
        hp = np.random.default_rng(5).integers(0, 400, (3, 2, 6)).astype(np.int32)
        jax_words = np.asarray(r.digest_named({'state.hp': j.asarray(hp)}, 3)[1])[:, 0]        # [L, 2 sides, 2 words]
        for lane in range(3):
            for side in range(2):self.assertEqual(r.mon_digest(hp[lane, side]).tolist(), jax_words[lane, side].tolist())

    def wrap_tick_lane(self, *, tick=20, base_hp=320, now=None, **over):
        """Side 0's Wrap (user 2) struck side 1's active 0 (maxhp 320): counter 4 = 3+2-1 after the tick, the base's side-1
        HP digest taken with `base_hp` there."""
        from gpu_engine import v2e_regress as r
        trap = np.zeros((2, 6), np.int32);trap[1, 0] = 4;source = np.full((2, 6), -1, np.int32);source[1, 0] = 2
        bag = np.zeros((2, 6, 6), bool);bag[1, 0, 5] = True;clock = np.zeros((2, 6), np.int32);clock[1, 0] = 1
        hp = np.full((2, 6), 320, np.int32);hp[1, 0] = base_hp - tick if now is None else now
        base = np.full((2, 6), 320, np.int32);base[1, 0] = base_hp
        base_words = np.stack([r.mon_digest(base[0]), r.mon_digest(base[1])])
        return self.lane(**dict(dict(trap=trap, source=source, bag=bag, clock=clock, wrap_by=np.array([True, False]),
            wrap_user=np.array([2, -1]), wrap_target=np.array([0, -1]), draws7=np.array([2, 0]),
            post_active=np.array([2, 0], np.int32), post_hp=hp, base_hp=base_words), **over))

    def test_wrap_tick_hp_needs_the_exact_tick_at_the_trapped_active(self):
        """M2 review (RW7): state.hp of the struck side is explained only by adding max(1, maxhp//16) back at the trapped
        active and matching the base digest; a Substitute's HP never; both stay `must` entries when unexplained."""
        trap = [('state.partial_trap_counter', 1), ('state.partial_trap_source', 1), (r_clock(), 1)]
        hp1, sub1 = ('state.hp', 1), ('state.substitute_hp', 1)
        rest, _, _, must = self.rule(trap + [hp1], self.wrap_tick_lane())
        self.assertEqual((rest, must), (set(), {hp1}))
        for lane in (self.wrap_tick_lane(tick=40),                               # baseMaxhp/8 (RW7)
                     self.wrap_tick_lane(tick=19)):                              # one short
            self.assertEqual(self.rule(trap + [hp1], lane)[0], {hp1})
        self.assertEqual(self.rule(trap + [hp1, sub1], self.wrap_tick_lane())[0], {sub1})
        unticked = dict(trap=np.where(self.wrap_tick_lane()['trap'] > 0, 5, 0), ran=False)
        self.assertEqual(self.rule(trap + [hp1], self.wrap_tick_lane(**unticked))[0], {hp1})       # no tick: HP must not move
        held_not = self.wrap_tick_lane(active_turns=np.array([0, 1], np.int32))
        self.assertEqual(self.rule(trap + [hp1], held_not)[0], set(trap) | {hp1})
        # A lethal tick: the faint cleared the trap; the base kept 1..20 HP there.
        cleared = dict(trap=np.zeros((2, 6), np.int32), source=np.full((2, 6), -1, np.int32))
        for base_hp, want in ((13, set()), (20, set()), (21, {hp1})):
            self.assertEqual(self.rule([hp1], self.wrap_tick_lane(base_hp=base_hp, now=0, **cleared))[0], want, base_hp)
        self.assertEqual(self.rule([hp1], self.wrap_tick_lane(base_hp=13, now=0, ran=False, **cleared))[0], {hp1})
        self.assertEqual(self.rule([hp1], self.wrap_tick_lane(base_hp=13, now=0, post_active=np.array([1, 0], np.int32), **cleared))[0], {hp1})

    def test_reveals_need_a_holder_of_the_immunity(self):
        """M2 review: a reveal entry is explained only at a Soundproof active on a Perish Song lane, or an Insomnia/Vital
        Spirit active the other side's unblocked Yawn line met; otherwise it stays (no allowance covers it)."""
        reveal = [('trackers.shown_ability', 1), ('trackers.belief_mask', 1)]
        sound = np.zeros((2, 6), np.int32);sound[1, 0] = SOUNDPROOF;awake = np.zeros((2, 6), np.int32);awake[1, 0] = VITALSPIRIT
        perish = dict(perish_used=np.array([True, False]))
        yawn = dict(yawn_by=np.array([True, False]), yawn_target=np.array([0, -1]))
        self.assertEqual(self.rule(reveal, self.lane(ability=sound, **perish))[0], set())
        self.assertEqual(self.rule(reveal, self.lane(ability=awake, **yawn))[0], set())
        for bad in (dict(**perish), dict(ability=awake, **perish), dict(ability=sound), dict(ability=sound, **yawn),
                    dict(ability=awake, **dict(yawn, blocked=np.array([True, False]))), dict(ability=awake)):
            self.assertEqual(self.rule(reveal, self.lane(**bad))[0], set(reveal), bad)


def r_clock():
    from gpu_engine import v2e_regress as r
    return r.WRAP_CLOCK


if __name__ == '__main__':
    unittest.main()
