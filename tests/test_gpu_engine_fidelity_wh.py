"""Fidelity WH: White Herb, Sacred Fire's thaw and Facade (C-lite), on authored states.

The rules follow Showdown f76228a1's Gen 3 code as cited in gpu_engine/transition.py (_white_herb), items.white_herb,
variable_moves.facade_power and the defrost block; the two-engine probes (gpu_engine/effect_probes/{wh,sacredfire,
facade}_*.json) are the Showdown gate. Light by design (heavy-test rule): one jitted step_with_chance at B=8 shared by
every transition test (table and arrays are arguments, compiled once, asserted last), and pure-Python / numpy checks for
the tables, Facade's base power and the regression rule.
"""
import unittest

import numpy as np
import jax
import jax.numpy as j

TYPES = ['_none', 'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug',
         'rock', 'ghost', 'dragon', 'dark', 'steel']
NORMAL, FIRE, FIGHTING, PSYCHIC, GHOST = 1, 2, 7, 11, 14
MOVES = ['_none', 'struggle', 'splash', 'superpower', 'psychoboost', 'charm', 'tackle', 'explosion', 'sacredfire', 'facade',
         'strength', 'slam', 'protect', 'batonpass']
STRUGGLE, SPLASH, SUPERPOWER, PSYCHOBOOST, CHARM, TACKLE, EXPLOSION, SACREDFIRE, FACADE, STRENGTH, SLAM, PROTECT, BATON = range(1, 14)
ABILITIES = ['_none', 'intimidate', 'guts', 'pressure']
INTIMIDATE, GUTS = 1, 2
ITEMS = ['_none', 'leftovers', 'whiteherb', 'lumberry']
LEFTOVERS, HERB = 1, 2
PAR, BRN, PSN, TOX, SLP, FRZ = 1, 2, 3, 4, 5, 6
N = 8          # every transition test runs one fixed batch shape: one compile


def metadata():
    def row(name, power, typ, category, accuracy=100, flags=None, priority=0):
        return dict(id=name, base_power=power, type=typ, source_type=TYPES[typ].title(), category=category, pp=10,
                    accuracy=accuracy, priority=priority, flags={'protect': 1} if flags is None else flags)
    hit = {'protect': 1, 'contact': 1}
    rows = [None, row('struggle', 50, NORMAL, 'Physical', True, {'contact': 1}), row('splash', 0, NORMAL, 'Status', True, {}),
            row('superpower', 120, FIGHTING, 'Physical', 100, hit), row('psychoboost', 140, PSYCHIC, 'Special', 90),
            row('charm', 0, NORMAL, 'Status', 100), row('tackle', 35, NORMAL, 'Physical', 100, hit),
            row('explosion', 250, NORMAL, 'Physical', 100), row('sacredfire', 100, FIRE, 'Special', 95, {'protect': 1, 'defrost': 1}),
            row('facade', 70, NORMAL, 'Physical', 100, hit), row('strength', 140, NORMAL, 'Physical', 100, hit),
            row('slam', 70, NORMAL, 'Physical', 100, hit), row('protect', 0, NORMAL, 'Status', True, {}, priority=3),
            row('batonpass', 0, NORMAL, 'Status', True, {})]
    chart = np.ones((17, 17));chart[NORMAL - 1, GHOST - 1] = 0
    return {'vocabulary': dict(moves=MOVES, abilities=ABILITIES, items=ITEMS, species=['_none', 'a', 'b', 'c', 'd', 'e', 'f'],
                               types=TYPES), 'moves': rows, 'type_chart': chart.tolist()}


class TableTests(unittest.TestCase):
    """Tables, Facade's base power and White Herb's restore rule: no step compile."""
    def test_tables_append_their_fields(self):
        from gpu_engine import effects, items, variable_moves
        meta = metadata()
        self.assertEqual(items.ItemTable._fields[-1], 'white_herb');self.assertEqual(effects.MoveEffects._fields[-1], 'defrost')
        self.assertEqual(variable_moves.VariableMoveTable._fields[-1], 'facade')
        self.assertEqual(np.flatnonzero(np.asarray(items.from_metadata(meta).white_herb)).tolist(), [HERB])
        self.assertEqual(np.flatnonzero(np.asarray(effects.from_metadata(meta).defrost)).tolist(), [SACREDFIRE])   # the move flag
        self.assertEqual(np.flatnonzero(np.asarray(variable_moves.from_metadata(meta).facade)).tolist(), [FACADE])

    def test_facade_doubles_for_a_set_status_other_than_slp(self):
        """data/moves.ts facade onBasePower: pokemon.status && pokemon.status !== 'slp' (frz included literally; a frozen
        user never reaches Facade frozen). Other moves are untouched."""
        from gpu_engine import variable_moves
        t = variable_moves.from_metadata(metadata())
        status = j.arange(7)
        self.assertEqual(np.asarray(variable_moves.facade_power(t, j.full(7, FACADE), j.full(7, 70), status)).tolist(),
                         [70, 140, 140, 140, 140, 70, 140])
        self.assertEqual(np.asarray(variable_moves.facade_power(t, j.full(7, SLAM), j.full(7, 70), status)).tolist(), [70] * 7)

    def test_white_herb_restores_only_negative_stages_once_and_only_alive(self):
        from gpu_engine import items
        t = items.from_metadata(metadata())
        item = j.asarray([HERB, HERB, HERB, LEFTOVERS, HERB])
        boosts = j.asarray([[-1, -1, 0, 0, 0, 0, 0], [2, -1, 0, 0, -6, 0, 1], [1, 0, 0, 0, 0, 0, 0], [-2, 0, 0, 0, 0, 0, 0],
                            [-2, 0, 0, 0, 0, 0, 0]])
        hp = j.asarray([10, 10, 10, 10, 0])
        new_item, new_boosts, used = items.white_herb(t, item, boosts, hp)
        self.assertEqual(np.asarray(used).tolist(), [True, True, False, False, False])
        self.assertEqual(np.asarray(new_item).tolist(), [0, 0, HERB, LEFTOVERS, HERB])
        self.assertEqual(np.asarray(new_boosts).tolist(), [[0] * 7, [2, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0],
                                                           [-2, 0, 0, 0, 0, 0, 0], [-2, 0, 0, 0, 0, 0, 0]])


class TransitionTests(unittest.TestCase):
    """step_with_chance on authored 8-lane states: side 0 (faster) uses its slot-0 move, side 1 its slot 0 (Splash by
    default); side 1's party 1 is an Intimidate mon. One jit, the table and every array as arguments, compiled once."""
    @classmethod
    def setUpClass(cls):
        from gpu_engine import transition
        from gpu_engine.state import BattleState, empty_state
        cls.t, cls.State, cls.empty = transition, BattleState, staticmethod(empty_state)
        cls.table = transition.tables_from_metadata(metadata())
        cls.jitted = dict(step=jax.jit(transition.step_with_chance))

    def build(self, lanes, move=SPLASH, foe_move=SPLASH):
        s = self.empty(N, seed=4247)
        s = s._replace(hp=j.full_like(s.hp, 400), max_hp=j.full_like(s.max_hp, 400),
            stats=j.full_like(s.stats, 200).at[..., 0].set(400).at[:, 0, :, 5].set(301),
            level=j.full_like(s.level, 100), types=s.types.at[..., 0].set(NORMAL), base_types=s.base_types.at[..., 0].set(NORMAL),
            species=j.broadcast_to(j.arange(1, 7, dtype=j.int32), (N, 2, 6)), ability=s.ability.at[:, 1, 1].set(INTIMIDATE),
            moves=j.full_like(s.moves, SPLASH).at[:, 0, :, 0].set(move).at[:, 1, :, 0].set(foe_move),
            pp=j.full_like(s.pp, 10), done=j.zeros(N, bool))
        arrays = {k: np.array(v) for k, v in s._asdict().items()}
        assert len(lanes) <= N
        for lane, edits in enumerate(lanes):
            for field, writes in edits.items():
                for index, value in writes:arrays[field][(lane,) + tuple(index)] = value
        arrays['base_types'] = arrays['types'].copy();arrays['base_ability'] = arrays['ability'].copy()
        return self.State(**{k: j.asarray(v, dtype=getattr(s, k).dtype) for k, v in arrays.items()})

    def chance(self, draws=(), secondary=()):
        """Hit, no crit, max damage roll, no thaw/paralysis/self-hit, duration roll 0, Protect succeeds, no secondary."""
        d = np.zeros((N, 2, 12), np.int32);d[:, :, 1:3] = 99;d[:, :, 4] = 15;d[:, :, 5] = 15
        sec = np.full((N, 2), 99, np.int32)
        for lane, edits in enumerate(draws):
            for (side, column), value in edits.items():d[lane, side, column] = value
        for lane, edits in enumerate(secondary):
            for side, value in edits.items():sec[lane, side] = value
        z = j.zeros((N, 2), j.int32);e = np.zeros((N, 2, 4), np.int32);e[:, :, 0] = 15
        return self.t.ChanceInputs(draws=j.asarray(d), phazing=z, secondary=j.asarray(sec), sleep_talk=z, entry_ties=z,
                                   initial_tie=j.zeros(N, j.int32), effect=j.asarray(e))

    def play(self, s, chance, a0=0, a1=0):
        actions = j.zeros((N, 2), j.int32).at[:, 0].set(j.asarray(a0, j.int32)).at[:, 1].set(j.asarray(a1, j.int32))
        out = self.jitted['step'](s, actions, self.table, chance)
        self.assertFalse(bool(np.asarray(out.invalid_actions).any()), np.asarray(out.invalid_actions))
        return out.state

    def at(self, s, name, side, mon=0):
        return np.asarray(getattr(s, name))[:, side, mon]

    def test_after_move_restores_both_actives(self):
        """onAnyAfterMove after every move that passed BeforeMove, for both actives. Lanes: 0 Superpower's -1/-1 restored
        (item gone); 1 the foe's Protect blocks it: no drop (herb kept); 2 +2 Atk, then Superpower: +1 Atk kept, Def restored; 3 the
        faster side's Charm on the holding foe, restored at Charm's own AfterMove; 4 a standing -1 restored by the holder's
        Splash; 5 Explosion: the user is at 0 HP at AfterMove (no useItem); 6 Leftovers holds nothing back; 7 the holder
        sleeps (no AfterMove of its own) and the foe's Splash restores it."""
        drop = [((0, 0, 0), -1)]
        lanes = [{'item': [((0, 0), HERB)]}, {'item': [((0, 0), HERB)], 'moves': [((1, 0, 0), PROTECT)]},
                 {'item': [((0, 0), HERB)], 'boosts': [((0, 0, 0), 2)]},
                 {'item': [((1, 0), HERB)], 'moves': [((0, 0, 0), CHARM)]},
                 {'item': [((0, 0), HERB)], 'boosts': drop, 'moves': [((0, 0, 0), SPLASH)]},
                 {'item': [((0, 0), HERB)], 'boosts': drop, 'moves': [((0, 0, 0), EXPLOSION)]},
                 {'item': [((0, 0), LEFTOVERS)], 'boosts': drop, 'moves': [((0, 0, 0), SPLASH)]},
                 {'item': [((0, 0), HERB)], 'boosts': drop, 'status': [((0, 0), SLP)], 'status_counter': [((0, 0), 3)]}]
        s = self.play(self.build(lanes, move=SUPERPOWER), self.chance())
        b0 = np.asarray(s.boosts)[:, 0, 0];b1 = np.asarray(s.boosts)[:, 1, 0]
        self.assertEqual(b0[:, :2].tolist(), [[0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [-1, 0], [0, 0]])
        self.assertEqual(self.at(s, 'item', 0).tolist(), [0, HERB, 0, 0, 0, HERB, LEFTOVERS, 0])
        self.assertEqual((b1[3].tolist(), int(self.at(s, 'item', 1)[3])), ([0] * 7, 0))
        self.assertEqual(int(self.at(s, 'hp', 0)[5]), 0)

    def test_psycho_boost_and_a_destiny_bond_ko(self):
        """Psycho Boost's -2 SpA is restored at once (lane 0; lane 1 without the herb keeps it). Lane 2: the bonded foe at 1
        HP falls to it; the user restores at AfterMove and only then faints at faintMessages (its herb is used)."""
        bond = {'hp': [((1, 0), 1)], 'volatile': [((1, 0), 64)]}
        lanes = [{'item': [((0, 0), HERB)]}, {'item': [((0, 0), LEFTOVERS)]}, dict(bond, item=[((0, 0), HERB)]),
                 dict(bond, item=[((0, 0), LEFTOVERS)])]
        s = self.play(self.build(lanes, move=PSYCHOBOOST), self.chance())
        self.assertEqual(np.asarray(s.boosts)[:2, 0, 0, 2].tolist(), [0, -2])
        self.assertEqual(self.at(s, 'item', 0)[:4].tolist(), [0, LEFTOVERS, 0, LEFTOVERS])
        self.assertEqual(self.at(s, 'hp', 0)[2:4].tolist(), [0, 0]);self.assertEqual(self.at(s, 'hp', 1)[2:4].tolist(), [0, 0])

    def test_intimidate_switch_in_is_restored_only_later(self):
        """Gen 3 runSwitch: SwitchIn (White Herb's onAnySwitchIn) before the entrant's Intimidate. Lanes 0/1: side 1 switches
        in its Intimidate mon; side 0's Tackle hits it at -1 with or without the herb (equal damage), then the herb restores
        at AfterMove. Lanes 2/3: both switch (side 0 first), no move: the herb restores at residual order 29, unless its
        holder (poisoned at 1 HP) faints at its order-10 poison first (the herb stays). Lanes 4-6 (initial leads, side 1's
        lead Intimidate): 4 the slower holder restores at its own runSwitch (its Tackle at 0); 5 the faster holder keeps -1
        into its Tackle; 6 control without the herb."""
        lead = {'ability': [((1, 0), INTIMIDATE)], 'entry_pending': [((0,), True), ((1,), True)]}
        slow = [((0, 0, 5), 150)]
        lanes = [{'item': [((0, 0), HERB)]}, {'item': [((0, 0), LEFTOVERS)]}, {'item': [((0, 1), HERB)]},
                 {'item': [((0, 1), HERB)], 'hp': [((0, 1), 1)], 'status': [((0, 1), PSN)]},
                 dict(lead, item=[((0, 0), HERB)], stats=slow), dict(lead, item=[((0, 0), HERB)]), dict(lead, item=[((0, 0), LEFTOVERS)], stats=slow),
                 {}]
        a0 = j.asarray([0, 0, 4, 4, 0, 0, 0, 0]);a1 = j.asarray([4, 4, 4, 4, 0, 0, 0, 0])
        s = self.play(self.build(lanes, move=TACKLE), self.chance(), a0=a0, a1=a1)
        hp = np.asarray(s.hp)
        self.assertEqual(hp[0, 1, 1], hp[1, 1, 1]);self.assertLess(hp[0, 1, 1], 400)              # both hit at -1
        self.assertEqual(np.asarray(s.boosts)[:2, 0, 0, 0].tolist(), [0, -1])
        self.assertEqual(self.at(s, 'item', 0)[:2].tolist(), [0, LEFTOVERS])
        self.assertEqual((int(np.asarray(s.boosts)[2, 0, 1, 0]), int(self.at(s, 'item', 0, 1)[2])), (0, 0))
        self.assertEqual((int(hp[3, 0, 1]), int(self.at(s, 'item', 0, 1)[3])), (0, HERB))
        self.assertEqual(np.asarray(s.boosts)[4:7, 0, 0, 0].tolist(), [0, 0, -1])
        self.assertEqual(self.at(s, 'item', 0)[4:7].tolist(), [0, 0, LEFTOVERS])
        self.assertGreater(400 - int(hp[4, 1, 0]), 400 - int(hp[5, 1, 0]))                        # lane 4 hit at 0
        self.assertEqual(int(hp[5, 1, 0]), int(hp[6, 1, 0]))                                       # 5 and 6 at -1

    def test_baton_passed_drops_are_restored_on_entry(self):
        """Side 0 at -2 Atk Baton Passes (the turn waits, every lane); its recipient (party 1) holds the herb in lane 0,
        Leftovers in lane 1: the herb restores at the recipient's runSwitch, the plain recipient keeps -2."""
        lanes = [{'boosts': [((0, 0, 0), -2)], 'item': [((0, 1), HERB)]}, {'boosts': [((0, 0, 0), -2)], 'item': [((0, 1), LEFTOVERS)]}]
        s = self.play(self.build(lanes, move=BATON), self.chance())
        self.assertTrue(bool(np.asarray(s.pending_residual)[0]))
        s = self.play(s, self.chance(), a0=4)
        self.assertEqual(np.asarray(s.active)[:2, 0].tolist(), [1, 1])
        self.assertEqual(np.asarray(s.boosts)[:2, 0, 1, 0].tolist(), [0, -2])
        self.assertEqual(self.at(s, 'item', 0, 1)[:2].tolist(), [0, LEFTOVERS])

    def test_sacred_fire_thaws_its_frozen_user(self):
        """Lanes: 0 the thaw roll fails, the defrost move thaws and hits; 1 it misses (95% accuracy) and still thaws; 2 the
        20% roll thaws first; 3 a frozen Tackle user stays frozen (no move); 4 into the foe's Protect, thawed; 5 an unfrozen
        user (control); 6 its 50% burn lands (secondary 0); 7 not on an already paralysed target."""
        frozen = {'status': [((0, 0), FRZ)]}
        lanes = [frozen, frozen, frozen, dict(frozen, moves=[((0, 0, 0), TACKLE)]), dict(frozen, moves=[((1, 0, 0), PROTECT)]),
                 {}, {}, {'status': [((1, 0), PAR)]}]
        s = self.play(self.build(lanes, move=SACREDFIRE),
                      self.chance(draws=[{}, {(0, 6): 99}, {(0, 1): 0}], secondary=[{}] * 6 + [{0: 0}, {0: 0}]))
        self.assertEqual(self.at(s, 'status', 0).tolist(), [0, 0, 0, FRZ, 0, 0, 0, 0])
        hp = self.at(s, 'hp', 1)
        self.assertEqual([int(x) < 400 for x in hp], [True, False, True, False, False, True, True, True])
        self.assertEqual(self.at(s, 'status', 1).tolist(), [0, 0, 0, 0, 0, 0, BRN, PAR])
        self.assertEqual(int(hp[0]), int(hp[5]))

    def test_facade_doubles_under_status_and_keeps_gen3_burn_halving(self):
        """Damage to side 1 (equal): 0 a paralysed Facade = 1 a healthy 140-power Strength; 2 a burned Facade without Guts =
        3 a burned Strength (both halved); 4 a burned Guts Facade = 5 a burned Guts Strength (Atk x1.5, no halving);
        6 a healthy Facade = 7 a healthy 70-power Slam."""
        burn, guts = [((0, 0), BRN)], [((0, 0), GUTS)]
        lanes = [{'status': [((0, 0), PAR)]}, {'moves': [((0, 0, 0), STRENGTH)]}, {'status': burn}, {'status': burn, 'moves': [((0, 0, 0), STRENGTH)]},
                 {'status': burn, 'ability': guts}, {'status': burn, 'ability': guts, 'moves': [((0, 0, 0), STRENGTH)]},
                 {}, {'moves': [((0, 0, 0), SLAM)]}]
        s = self.play(self.build(lanes, move=FACADE), self.chance())
        dealt = 400 - self.at(s, 'hp', 1)
        for a, b in ((0, 1), (2, 3), (4, 5), (6, 7)):self.assertEqual(int(dealt[a]), int(dealt[b]), (a, b))
        self.assertGreater(int(dealt[0]), int(dealt[6]));self.assertGreater(int(dealt[4]), int(dealt[0]))
        self.assertLess(int(dealt[2]), int(dealt[0]))

    def test_jitted_step_equals_an_eager_step(self):
        """Positive control for the shared compiled step: one lane set, eager against jitted, every leaf."""
        lanes = [{'item': [((0, 0), HERB), ((1, 0), HERB)], 'boosts': [((1, 0, 3), -1)], 'status': [((0, 0), BRN)]}] * N
        s = self.build(lanes, move=FACADE, foe_move=SUPERPOWER);chance = self.chance()
        actions = j.zeros((N, 2), j.int32)
        fast = self.jitted['step'](s, actions, self.table, chance).state
        with jax.disable_jit():
            slow = self.t.step_with_chance(s, actions, self.table, chance).state
        for name in s._fields:np.testing.assert_array_equal(np.asarray(getattr(fast, name)), np.asarray(getattr(slow, name)), err_msg=name)

    def test_zz_one_compile(self):
        """Every transition test used the one compiled step (no recompile)."""
        self.assertEqual(self.jitted['step']._cache_size(), 1)


class RegressRuleTests(unittest.TestCase):
    """v2e_regress.wh_rule on one lane's arrays (numpy): each White Herb write needs a recorded use and a per-mon witness."""
    def lane(self, **over):
        z = lambda v=0: np.full((2, 6), v, np.int32)
        used = np.zeros((2, 6), bool);used[0, 0] = True
        raised = np.zeros((2, 6, 7), np.int32);raised[0, 0, :2] = 1
        item = z();item[0, 0] = 0;base_item = z();base_item[0, 0] = HERB
        boosts = np.zeros((2, 6, 7), np.int32);base_boosts = boosts.copy();base_boosts[0, 0, :2] = -1
        shown = z();shown[0, 0] = HERB
        removed = np.zeros((2, 6), bool);removed[0, 0] = True
        wit = {'state.item': item, 'state.item_knocked': np.zeros((2, 6), bool), 'state.boosts': boosts, 'trackers.shown_item': shown,
               'trackers.item_removed': removed, 'trackers.boosts': boosts.copy(),
               "trackers.field_features['items_removed_credit']": np.zeros(2, np.int32)}
        base = {'state.item': base_item, 'state.item_knocked': np.zeros((2, 6), bool), 'state.boosts': base_boosts,
                'trackers.shown_item': z(), 'trackers.item_removed': np.zeros((2, 6), bool), 'trackers.boosts': base_boosts.copy(),
                "trackers.field_features['items_removed_credit']": np.zeros(2, np.int32)}
        acting = np.zeros((2, 6), bool);acting[:, 0] = True
        w = dict(used=used, raise_=raised, bad=False, herb=HERB, p2p=np.broadcast_to(np.arange(6), (2, 6)).copy(),
                 ko=np.zeros((2, 6), bool), acting=acting, wit=wit, base=base)
        for k, v in over.items():
            if k in ('wit', 'base'):w[k] = dict(w[k], **v)
            else:w[k] = v
        w['raise'] = w.pop('raise_');return w

    def rule(self, entries, w):
        from gpu_engine import v2e_regress as r
        return r.wh_rule(set(entries), w)

    ENTRIES = [('state.item', 0), ('state.boosts', 0), ('trackers.shown_item', 0), ('trackers.item_removed', 0), ('trackers.boosts', 0)]

    def test_a_recorded_restore_explains_its_holder_per_mon(self):
        rest, done = self.rule(self.ENTRIES, self.lane())
        self.assertEqual((rest, len(done)), (set(), 5))
        self.assertEqual(self.rule(self.ENTRIES, self.lane(bad=True))[0], set(self.ENTRIES))               # malformed call
        self.assertEqual(self.rule(self.ENTRIES, self.lane(used=np.zeros((2, 6), bool)))[0], set(self.ENTRIES))   # no use
        wrong = np.zeros((2, 6, 7), np.int32);wrong[0, 0, :] = 1                                              # a positive reset too
        self.assertIn(('state.boosts', 0), self.rule(self.ENTRIES, self.lane(raise_=wrong))[0])
        bench = np.zeros((2, 6), np.int32);bench[0, 0] = 0;bench[0, 3] = 0                                   # a bench mon's item moved
        base_item = np.zeros((2, 6), np.int32);base_item[0, 0] = HERB;base_item[0, 3] = LEFTOVERS
        self.assertIn(('state.item', 0), self.rule(self.ENTRIES, self.lane(base={'state.item': base_item}))[0])
        shown = np.zeros((2, 6), np.int32);shown[0, 0] = HERB;shown[0, 4] = HERB                             # a reveal off the actives
        self.assertIn(('trackers.shown_item', 0), self.rule(self.ENTRIES, self.lane(wit={'trackers.shown_item': shown}))[0])

    def test_a_base_knock_off_and_a_base_trick_partner(self):
        """The base kept the herb and then lost it to Knock Off (item_knocked and the removal credit differ only there), or
        handed it to the foe's Trick (the partner active holds nothing here, the herb in the base)."""
        knocked = np.zeros((2, 6), bool);knocked[0, 0] = True
        both_none = np.zeros((2, 6), np.int32)
        lane = self.lane(base={'state.item': both_none, 'state.item_knocked': knocked,
                               "trackers.field_features['items_removed_credit']": np.array([1, 0], np.int32)})
        entries = [('state.item_knocked', 0), ("trackers.field_features['items_removed_credit']", 0), ('state.boosts', 0)]
        self.assertEqual(self.rule(entries, lane)[0], set())
        self.assertEqual(self.rule(entries, self.lane(base={'state.item_knocked': knocked}))[0] & {entries[1]}, {entries[1]})
        partner = np.zeros((2, 6), np.int32);partner[1, 0] = HERB;partner[0, 0] = HERB
        lane = self.lane(base={'state.item': partner})
        self.assertEqual(self.rule([('state.item', 0), ('state.item', 1)], lane)[0], set())
        lane = self.lane(base={'state.item': partner}, acting=np.zeros((2, 6), bool))
        self.assertEqual(self.rule([('state.item', 1)], lane)[0], {('state.item', 1)})                    # not an active

    def test_facade_keeps_only_sides_whose_line_doubled(self):
        from gpu_engine.v2e_regress import facade_actors
        self.assertEqual(facade_actors([0, 1], np.array([True, False])), [0])
        self.assertEqual(facade_actors([1], np.array([True, False])), [])


if __name__ == '__main__':
    unittest.main()
