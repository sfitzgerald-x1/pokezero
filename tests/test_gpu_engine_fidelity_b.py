"""Fidelity patch B: the chance secondaries (Iron Tail, Steel Wing, Meteor Mash, Headbutt, Needle Arm, Signal Beam) and
confusion, switched on, on authored states.

The rules follow Showdown f76228a1 as cited in gpu_engine/secondary.py (apply_chance_effects) and gpu_engine/transition.py;
the two-engine probes (gpu_engine/effect_probes/{irontail,steelwing,meteormash,headbutt,needlearm,signalbeam,confusion}*.json)
are the Showdown gate. Device: B2's -end|X|confusion clear and the self-hit's HP line (also in
test_gpu_engine_device_observation_boundary_parity.py). Also the B regression plug-in's declared allowances.
"""
import importlib.util
import unittest

import numpy as np

TYPES = ['_none', 'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug',
         'rock', 'ghost', 'dragon', 'dark', 'steel']
NORMAL, GRASS, BUG, GHOST, DARK, STEEL = 1, 5, 12, 14, 16, 17
MOVES = ['_none', 'struggle', 'splash', 'irontail', 'steelwing', 'meteormash', 'headbutt', 'needlearm', 'signalbeam',
         'crunch', 'silverwind', 'mudshot', 'rockslide', 'tackle', 'batonpass']
(STRUGGLE, SPLASH, IRONTAIL, STEELWING, METEORMASH, HEADBUTT, NEEDLEARM, SIGNALBEAM, CRUNCH, SILVERWIND, MUDSHOT, ROCKSLIDE,
 TACKLE, BATONPASS) = range(1, 15)
ABILITIES = ['_none', 'serenegrace', 'shielddust', 'clearbody', 'whitesmoke', 'hypercutter', 'owntempo', 'innerfocus', 'keeneye']
SERENEGRACE, SHIELDDUST, CLEARBODY, WHITESMOKE, HYPERCUTTER, OWNTEMPO, INNERFOCUS, KEENEYE = range(1, 9)
ITEMS = ['_none', 'leftovers', 'lumberry']
LUM = 2
B = 24
HAS_JAX = bool(importlib.util.find_spec('jax'))


def metadata():
    def row(name, power, typ, category, accuracy=100, flags=None):
        return dict(id=name, base_power=power, type=typ, source_type=TYPES[typ].title(), category=category, pp=10,
                    accuracy=accuracy, priority=0, flags=flags or {'protect': 1})
    rows = [None, row('struggle', 50, NORMAL, 'Physical', True, {'contact': 1}), row('splash', 0, NORMAL, 'Status', True, {}),
            row('irontail', 100, STEEL, 'Physical', 75, {'contact': 1, 'protect': 1}),
            row('steelwing', 70, STEEL, 'Physical', 90, {'contact': 1, 'protect': 1}),
            row('meteormash', 100, STEEL, 'Physical', 85, {'contact': 1, 'protect': 1}),
            row('headbutt', 70, NORMAL, 'Physical', 100, {'contact': 1, 'protect': 1}),
            row('needlearm', 60, GRASS, 'Special', 100, {'contact': 1, 'protect': 1}),
            row('signalbeam', 75, BUG, 'Physical'), row('crunch', 80, DARK, 'Special', 100, {'contact': 1, 'protect': 1}),
            row('silverwind', 60, BUG, 'Physical'), row('mudshot', 55, 9, 'Physical', 95),
            row('rockslide', 75, 13, 'Physical', 90), row('tackle', 35, NORMAL, 'Physical', 95, {'contact': 1, 'protect': 1}),
            row('batonpass', 0, NORMAL, 'Status', True, {})]
    chart = np.ones((17, 17));chart[NORMAL - 1, GHOST - 1] = 0
    return {'vocabulary': dict(moves=MOVES, abilities=ABILITIES, items=ITEMS, species=['_none', 'a', 'b', 'c', 'd', 'e', 'f'],
                               types=TYPES), 'moves': rows, 'type_chart': chart.tolist()}


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class TableTests(unittest.TestCase):
    def test_rows(self):
        from gpu_engine import mechanics as m, secondary
        t = secondary.from_metadata(metadata())
        chance = np.asarray(t.chance)
        expect = {IRONTAIL: 30, STEELWING: 10, METEORMASH: 20, HEADBUTT: 30, NEEDLEARM: 30, SIGNALBEAM: 10,
                  CRUNCH: 20, SILVERWIND: 10, MUDSHOT: 100, ROCKSLIDE: 30}
        self.assertEqual(chance.tolist(), [expect.get(i, 0) for i in range(len(MOVES))])
        self.assertEqual(np.flatnonzero(np.asarray(t.flinch)).tolist(), [HEADBUTT, NEEDLEARM, ROCKSLIDE])
        self.assertFalse(np.asarray(t.status)[[HEADBUTT, NEEDLEARM, IRONTAIL, SIGNALBEAM]].any())
        drops = np.asarray(t.target_drops)
        self.assertEqual(drops.shape, (len(MOVES), 7));self.assertEqual(drops[IRONTAIL].tolist(), [0, -1, 0, 0, 0, 0, 0])
        self.assertEqual(np.flatnonzero(drops.any(1)).tolist(), [IRONTAIL])
        raises = np.asarray(t.self_boosts)
        self.assertEqual(raises[STEELWING].tolist(), [0, 1, 0, 0, 0, 0, 0]);self.assertEqual(raises[METEORMASH].tolist(), [1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(np.flatnonzero(raises.any(1)).tolist(), [STEELWING, METEORMASH])
        self.assertEqual(np.flatnonzero(np.asarray(t.confusion)).tolist(), [SIGNALBEAM])
        self.assertEqual(np.flatnonzero(np.asarray(t.own_tempo)).tolist(), [OWNTEMPO])
        # The original subset keeps its own fields and values (one appended field each, no reorder).
        self.assertEqual(np.flatnonzero(np.asarray(t.spd_drop)).tolist(), [CRUNCH])
        self.assertEqual(np.flatnonzero(np.asarray(t.speed_drop)).tolist(), [MUDSHOT])
        self.assertEqual(np.flatnonzero(np.asarray(t.self_omni)).tolist(), [SILVERWIND])
        self.assertEqual(secondary.SecondaryTable._fields[-4:], ('target_drops', 'self_boosts', 'confusion', 'own_tempo'))
        self.assertEqual(int(np.asarray(t.status)[CRUNCH]), m.NONE)


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class ChanceEffectTests(unittest.TestCase):
    """secondary.apply_chance_effects on one authored lane per case (side 0 acts on side 1)."""
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import secondary
        from gpu_engine.state import empty_state
        cls.j, cls.sec = j, secondary
        cls.table = secondary.from_metadata(metadata())
        cls.empty = staticmethod(empty_state)
        cls.apply = staticmethod(jax.jit(lambda s, side, move, roll, dur, el, sub: secondary.apply_chance_effects(
            s, side, move, roll, dur, cls.table, eligible=el, substitute_hit=sub)))

    def test_gates_caps_and_untouched_leaves(self):
        j = self.j
        # (move, roll, duration roll, edits {field: [(index, value)]}, eligible, substitute hit)
        cases = [
            (IRONTAIL, 29, 0, {}, True, False),                                              # 0  drop lands
            (IRONTAIL, 30, 0, {}, True, False),                                              # 1  no trigger
            (IRONTAIL, 0, 0, {'ability': [((1, 0), SHIELDDUST)]}, True, False),              # 2  Shield Dust
            (IRONTAIL, 0, 0, {}, True, True),                                                # 3  Substitute hit
            (IRONTAIL, 0, 0, {'hp': [((1, 0), 0)]}, True, False),                            # 4  fainted target
            (IRONTAIL, 0, 0, {'ability': [((1, 0), CLEARBODY)]}, True, False),               # 5  Clear Body
            (IRONTAIL, 0, 0, {'ability': [((1, 0), WHITESMOKE)]}, True, False),              # 6  White Smoke
            (IRONTAIL, 0, 0, {'ability': [((1, 0), HYPERCUTTER)]}, True, False),             # 7  Hyper Cutter: Atk only
            (IRONTAIL, 0, 0, {'boosts': [((1, 0, 1), -6)]}, True, False),                    # 8  at -6
            (IRONTAIL, 59, 0, {'ability': [((0, 0), SERENEGRACE)]}, True, False),            # 9  Serene Grace: 60
            (IRONTAIL, 0, 0, {}, False, False),                                              # 10 not eligible (miss)
            (STEELWING, 9, 0, {'ability': [((1, 0), SHIELDDUST)], 'hp': [((1, 0), 0)]}, True, True),  # 11 self entry survives
            (STEELWING, 10, 0, {}, True, False),                                             # 12 no trigger
            (METEORMASH, 19, 0, {'boosts': [((0, 0, 0), 5)]}, True, False),                  # 13 +5 -> +6
            (METEORMASH, 0, 0, {'boosts': [((0, 0, 0), 6)]}, True, False),                   # 14 at +6
            (METEORMASH, 0, 0, {'hp': [((0, 0), 0)]}, True, False),                          # 15 fainted user
            (SIGNALBEAM, 9, 3, {}, True, False),                                             # 16 confusion 2+3
            (SIGNALBEAM, 9, 0, {'ability': [((1, 0), OWNTEMPO)]}, True, False),              # 17 Own Tempo
            (SIGNALBEAM, 9, 0, {'confusion_counter': [((1, 0), 2)]}, True, False),           # 18 already confused
            (SIGNALBEAM, 9, 0, {'ability': [((1, 0), SHIELDDUST)]}, True, False),            # 19 Shield Dust
            (SIGNALBEAM, 9, 0, {}, True, True),                                              # 20 Substitute hit
            (SIGNALBEAM, 10, 0, {}, True, False),                                            # 21 no trigger
            (SIGNALBEAM, 19, 1, {'ability': [((0, 0), SERENEGRACE)]}, True, False),          # 22 Serene Grace: 20
            (HEADBUTT, 0, 0, {}, True, False),                                               # 23 flinch is apply()'s
        ]
        self.assertEqual(len(cases), B)
        s = self.empty(B, seed=7)
        arrays = {k: np.array(v) for k, v in s._asdict().items()}
        arrays['hp'][:] = 100;arrays['max_hp'][:] = 100;arrays['done'][:] = False
        for lane, (_, _, _, edits, _, _) in enumerate(cases):
            for field, writes in edits.items():
                for index, value in writes:arrays[field][(lane,) + tuple(index)] = value
        s = type(s)(**{k: j.asarray(v) for k, v in arrays.items()})
        col = lambda i: j.asarray([c[i] for c in cases])
        out, attempted = self.apply(s, j.zeros(B, j.int32), col(0).astype(j.int32), col(1).astype(j.int32),
                                    col(2).astype(j.int32), col(4), col(5))
        boosts, conf = np.asarray(out.boosts), np.asarray(out.confusion_counter)
        target_def = boosts[:, 1, 0, 1].tolist()
        self.assertEqual(target_def[:11], [-1, 0, 0, 0, 0, 0, 0, -1, -6, -1, 0])
        self.assertEqual(boosts[11:13, 0, 0, 1].tolist(), [1, 0])
        self.assertEqual(boosts[13:16, 0, 0, 0].tolist(), [6, 6, 0])
        self.assertEqual(conf[16:23, 1, 0].tolist(), [5, 0, 2, 0, 0, 0, 3])
        self.assertFalse(boosts[23].any());self.assertFalse(conf[23].any())
        self.assertEqual(np.asarray(attempted).tolist(),
            [True, False, False, False, False, True, True, True, True, True, False,
             True, False, True, True, False, True, True, True, False, False, False, True, False])
        changed = {'boosts', 'confusion_counter'}
        for name in out._fields:
            if name not in changed:
                np.testing.assert_array_equal(np.asarray(getattr(out, name)), np.asarray(getattr(s, name)), name)
        # Only the acting lanes' named mons changed.
        mask = np.zeros((B, 2, 6), bool);mask[:, :, 0] = True
        self.assertFalse((np.asarray(out.boosts) != np.asarray(s.boosts))[~mask].any())

    def test_flinch_rows_use_the_existing_path(self):
        """Headbutt/Needle Arm through secondary.apply: < chance (doubled by Serene Grace) flinches; Inner Focus, Shield Dust,
        a Substitute hit and a fainted target block it."""
        j = self.j
        from gpu_engine.state import FLINCH
        cases = [(HEADBUTT, 29, {}, False), (HEADBUTT, 30, {}, False), (HEADBUTT, 59, {'ability': [((0, 0), SERENEGRACE)]}, False),
                 (NEEDLEARM, 0, {'ability': [((1, 0), INNERFOCUS)]}, False), (NEEDLEARM, 0, {'ability': [((1, 0), SHIELDDUST)]}, False),
                 (NEEDLEARM, 0, {}, True), (NEEDLEARM, 0, {'hp': [((1, 0), 0)]}, False), (NEEDLEARM, 29, {}, False)]
        n = len(cases);s = self.empty(n, seed=9)
        arrays = {k: np.array(v) for k, v in s._asdict().items()}
        arrays['hp'][:] = 100;arrays['max_hp'][:] = 100;arrays['done'][:] = False
        for lane, (_, _, edits, _) in enumerate(cases):
            for field, writes in edits.items():
                for index, value in writes:arrays[field][(lane,) + tuple(index)] = value
        s = type(s)(**{k: j.asarray(v) for k, v in arrays.items()})
        out = self.sec.apply(s, j.zeros(n, j.int32), j.asarray([c[0] for c in cases], j.int32), j.asarray([c[1] for c in cases], j.int32),
                             self.table, eligible=j.ones(n, bool), substitute_hit=j.asarray([c[3] for c in cases]))
        flinched = (np.asarray(out.volatile)[:, 1, 0] & FLINCH) != 0
        self.assertEqual(flinched.tolist(), [True, False, True, False, False, False, False, True])


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class TransitionTests(unittest.TestCase):
    """step_with_chance / step_with_public on authored lanes: side 0 (faster) uses its slot-0 move, side 1 Splash."""
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import transition
        from gpu_engine.state import BattleState, FLINCH, empty_state
        cls.jax, cls.j, cls.t, cls.State, cls.FLINCH = jax, j, transition, BattleState, FLINCH
        cls.empty = staticmethod(empty_state)
        cls.table = transition.tables_from_metadata(metadata())
        cls.step = staticmethod(jax.jit(lambda s, a, c: transition.step_with_chance(s, a, cls.table, c)))

    def build(self, lanes, move=SIGNALBEAM):
        j = self.j;n = len(lanes);s = self.empty(n, seed=4243)
        s = s._replace(hp=j.full_like(s.hp, 300), max_hp=j.full_like(s.max_hp, 300),
            stats=j.full_like(s.stats, 200).at[..., 0].set(300).at[:, 0, :, 5].set(301),
            level=j.full_like(s.level, 100), types=s.types.at[..., 0].set(NORMAL), base_types=s.base_types.at[..., 0].set(NORMAL),
            species=j.broadcast_to(j.arange(1, 7, dtype=j.int32), (n, 2, 6)),
            moves=j.full_like(s.moves, SPLASH).at[:, 0, 0, 0].set(move), pp=j.full_like(s.pp, 10), done=j.zeros(n, bool))
        arrays = {k: np.array(v) for k, v in s._asdict().items()}
        for lane, edits in enumerate(lanes):
            for field, writes in edits.items():
                for index, value in writes:arrays[field][(lane,) + tuple(index)] = value
        return self.State(**{k: j.asarray(v) for k, v in arrays.items()})

    def chance(self, n, draws=(), secondary=0):
        """Hit, no crit, max damage roll, no thaw/paralysis/self-hit, duration roll 0; the secondary roll for side 0."""
        j = self.j;d = np.zeros((n, 2, 12), np.int32);d[:, :, 1:3] = 99;d[:, :, 4] = 15;d[:, :, 5] = 15
        for lane, edits in enumerate(draws):
            for (side, column), value in edits.items():d[lane, side, column] = value
        z = j.zeros((n, 2), j.int32)
        sec = np.full((n, 2), 99, np.int32);sec[:, 0] = secondary
        return self.t.ChanceInputs(draws=j.asarray(d), phazing=z, secondary=j.asarray(sec), sleep_talk=z, entry_ties=z,
                                   initial_tie=j.zeros(n, j.int32), effect=j.zeros((n, 2, 4), j.int32))

    def play(self, s, chance, a0=0):
        actions = self.j.zeros((s.hp.shape[0], 2), self.j.int32).at[:, 0].set(a0)
        out = self.step(s, actions, chance)
        self.assertFalse(bool(np.asarray(out.invalid_actions).any()))
        return out.state

    def test_signal_beam_starts_confusion_and_the_slower_target_decrements_it_that_turn(self):
        lanes = [{}, {}, {}, {}, {'ability': [((1, 0), OWNTEMPO)]}, {'item': [((1, 0), LUM)]}, {'substitute_hp': [((1, 0), 50)]}, {}]
        s = self.build(lanes)
        d = [{(0, 7): 0}, {(0, 7): 3}, {(0, 7): 3, (1, 3): 99}, {(0, 7): 1, (1, 3): 50}, {}, {}, {}, {(0, 6): 0}]
        out = self.play(s, self.chance(8, d, secondary=9))
        conf, hp, pp = np.asarray(out.confusion_counter)[:, 1, 0], np.asarray(out.hp)[:, 1, 0], np.asarray(out.pp)[:, 1, 0, 0]
        # 2+d, then Splash's BeforeMove decrement; a self-hit (draws[1][3] >= 50) spends no PP.
        self.assertEqual(conf.tolist(), [1, 4, 4, 2, 0, 0, 0, 1])
        self.assertEqual(pp.tolist(), [9, 9, 10, 10, 9, 9, 9, 9])
        self.assertGreater(300 - int(hp[0]), 0)
        self.assertGreater(int(hp[1]) - int(hp[2]), 0)                     # the self-hit's own damage
        self.assertEqual(int(hp[2]), int(hp[3]))                            # both self-hit lanes take the same
        self.assertEqual(np.asarray(out.item)[5, 1, 0], 0)                  # Lum eaten at once
        self.assertLess(int(np.asarray(out.substitute_hp)[6, 1, 0]), 50)   # the hit met the Substitute
        none = self.play(s, self.chance(8, d, secondary=10))
        self.assertFalse(np.asarray(none.confusion_counter).any())

    def test_snap_out_self_hit_and_blocks(self):
        lanes = [{'confusion_counter': [((1, 0), 1)]}, {'confusion_counter': [((1, 0), 3)]},
                 {'confusion_counter': [((1, 0), 3)], 'status': [((1, 0), 5)], 'status_counter': [((1, 0), 3)]},
                 {'confusion_counter': [((1, 0), 3)], 'volatile': [((1, 0), 1)]},
                 {'confusion_counter': [((1, 0), 3)], 'substitute_hp': [((1, 0), 40)]},
                 {'confusion_counter': [((1, 0), 3)], 'hp': [((1, 0), 3)]}, {}, {}]
        s = self.build(lanes, move=SPLASH)
        out = self.play(s, self.chance(8, [{(1, 3): 99}] * 8, secondary=99))
        conf, hp, pp = np.asarray(out.confusion_counter)[:, 1, 0], np.asarray(out.hp)[:, 1, 0], np.asarray(out.pp)[:, 1, 0, 0]
        # 1 -> 0 snaps out and moves even at roll 99; asleep or flinched: no decrement; a Substitute does not take the self-hit.
        self.assertEqual(conf.tolist()[:5], [0, 2, 3, 3, 2])
        self.assertEqual(pp.tolist()[:5], [9, 10, 10, 10, 10])
        self.assertEqual(int(hp[0]), 300);self.assertLess(int(hp[1]), 300);self.assertEqual(int(hp[2]), 300)
        self.assertEqual(int(np.asarray(out.substitute_hp)[4, 1, 0]), 40);self.assertEqual(int(hp[4]), int(hp[1]))
        self.assertEqual(int(hp[5]), 0)                                        # a self-KO
        self.assertEqual((np.asarray(out.volatile)[3, 1, 0] & self.FLINCH), 0)   # the flinch ends with the turn

    def test_iron_tail_and_headbutt_through_the_step(self):
        s = self.build([{}, {}, {}, {}], move=IRONTAIL)
        out = self.play(s, self.chance(4, [{}, {}, {(0, 6): 75}, {}], secondary=29), a0=0)
        self.assertEqual(np.asarray(out.boosts)[:3, 1, 0, 1].tolist(), [-1, -1, 0])
        h = self.build([{}, {'stats': [((1, 0, 5), 400)]}], move=HEADBUTT)
        out = self.play(h, self.chance(2, [], secondary=0))
        # The slower target flinches (no Splash, PP kept); the faster one moved first.
        self.assertEqual(np.asarray(out.pp)[:, 1, 0, 0].tolist(), [10, 9])
        self.assertFalse((np.asarray(out.volatile) & self.FLINCH).any())

    def test_device_self_hit_hp_line_and_snap_out_clear(self):
        """step_with_public: a self-hit publishes its HP line to the absent-item snapshot; a snap-out drops the bag entry."""
        j = self.j
        from gpu_engine import engine_device_trackers as tr
        from gpu_engine.public_active import NAMES
        lanes = [{'confusion_counter': [((1, 0), 3)]}, {'confusion_counter': [((1, 0), 3)]}, {'confusion_counter': [((1, 0), 1)]}, {}]
        s = self.build(lanes, move=SPLASH)
        init = self.t.initialize_with_public(s, self.table)
        t0 = init.trackers;slot = np.asarray(t0.active_slot)[:, 1]
        bag = np.asarray(t0.public_volatiles).copy();bag[np.arange(4), 1, slot, NAMES.index('confusion')] = [True, True, True, False]
        clock = np.asarray(t0.mon_features['confusion_turns']).copy();clock[np.arange(4), 1, slot] = [2, 2, 2, 0]
        t0 = t0._replace(public_volatiles=j.asarray(bag), mon_features={**t0.mon_features, 'confusion_turns': j.asarray(clock)})
        d = [{(1, 3): 99}, {(1, 3): 0}, {(1, 3): 99}, {}]
        out = self.jax.jit(lambda st, a, tk, c: self.t.step_with_public(st, a, self.table, tk, chance_inputs=c))(
            init.state, j.zeros((4, 2), j.int32), t0, self.chance(4, d, secondary=99))
        st, tk = out.step.state, out.trackers
        hp = np.asarray(st.hp)[:, 1, 0]
        percent = np.asarray(tk.negative_items.percent)[np.arange(4), 1, slot]
        self.assertLess(int(hp[0]), 300);self.assertEqual(hp[1:].tolist(), [300, 300, 300])
        self.assertEqual(int(percent[0]), int(tr._percent(j.asarray(hp[0]), j.asarray(300))))
        self.assertEqual(percent[1:].tolist(), [100, 100, 100])
        conf = np.asarray(st.confusion_counter)[:, 1, 0].tolist();self.assertEqual(conf, [2, 2, 0, 0])
        bag = np.asarray(tk.public_volatiles)[np.arange(4), 1, slot, NAMES.index('confusion')]
        self.assertEqual(bag.tolist(), [True, True, False, False])
        clock = np.asarray(tk.mon_features['confusion_turns'])[np.arange(4), 1, slot]
        self.assertEqual(clock.tolist(), [3, 3, 0, 0])                    # +1 at the TURN while confused; reset by the -end


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class SelfHitNegativeItemTests(unittest.TestCase):
    def test_self_hit_hp_line_is_action_phase_evidence_like_b2(self):
        """B2 (pokezero.belief _ACTION_PHASE_ONLY_HP_TAGS: '[from] confusion') takes the self-hit's -damage line as
        action-phase HP: a confused Exploud at 80% that does not heal rules Leftovers out. The device's HP_ACTION line
        at the self-hit gives the same narrowing; without it (the old device) nothing is learned."""
        import jax
        import jax.numpy as j
        from gpu_engine import engine_device_public_hooks as h, engine_device_negative_items as n
        from test_gpu_engine_device_negative_items import fixture, reference
        s, tables, leftovers, source = fixture(survivors=(0, 1), count=8)
        def run(state, line):
            c = h.initialize(state, tables)
            after = state._replace(hp=state.hp.at[:, 1, 0].set(160))
            if line:h.hp_line(c, after, 1, j.ones(2, bool), source_class=h.HP_ACTION)
            else:h.observe(c, after, j.ones(2, bool))
            h.turn_end(c, state, after, j.zeros(2, bool), upkeep=j.ones(2, bool), leftovers=leftovers)
            return c.trackers.belief_mask
        with_line, without = jax.jit(lambda st: run(st, True))(s), jax.jit(lambda st: run(st, False))(s)
        ref = reference(source, ['|-activate|p2a: Exploud|confusion', '|-damage|p2a: Exploud|80/100|[from] confusion'])
        self.assertIn('leftovers', ref.ruled_out_items)
        self.assertEqual(tuple(int(v['variant_id']) for v in ref.candidate_variants), (0, 1))
        self.assertEqual(np.asarray(with_line)[:, 1, 0].tolist(), [[3, 0]] * 2)
        self.assertEqual(np.asarray(without)[:, 1, 0].tolist(), [[255, 0]] * 2)
        self.assertEqual(n.HP_ACTION, h.HP_ACTION)


class RegressPluginTests(unittest.TestCase):
    def test_b_effects_declare_allowed_fields(self):
        from gpu_engine.v2e_regress import ALL, ALLOWED, PLUGINS, lane_verdict
        self.assertTrue(all(e in ALLOWED for e in PLUGINS['B']))
        for effect in ('headbutt', 'needlearm', 'signalbeam', 'confusion'):self.assertIs(ALLOWED[effect], ALL)
        self.assertEqual(lane_verdict({('state.boosts', 1), ('trackers.boosts', 1)}, {'irontail': [0]}, None)[0], [])
        for bad in (('state.boosts', 0), ('state.hp', 1), ('state.confusion_counter', 1), ('trackers.public_volatiles', 1)):
            self.assertEqual(lane_verdict({bad}, {'irontail': [0]}, None)[0], [bad])
        ok = {('state.boosts', 0), ('state.hp', 0), ('state.substitute_hp', 0), ('state.item', 0), ('state.hp', 1),
              ("trackers.mon_features['sub_present']", 0), ("trackers.field_features['wish_pending']", 0)}
        self.assertEqual(lane_verdict(ok, {'steelwing': [0]}, None)[0], [])
        for bad in (('state.status', 1), ('state.status', 0), ('state.confusion_counter', 1), ('trackers.public_volatiles', 1)):
            self.assertEqual(lane_verdict({bad}, {'steelwing': [0]}, None)[0], [bad])
        self.assertEqual(lane_verdict({('state.boosts', 0)}, {'meteormash': [0]}, None)[0], [])
        for bad in (('state.hp', 0), ('state.boosts', 1), ('state.hp', 1)):
            self.assertEqual(lane_verdict({bad}, {'meteormash': [0]}, None)[0], [bad])


if __name__ == '__main__':unittest.main()
