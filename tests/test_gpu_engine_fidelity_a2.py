"""Fidelity patch A2 (Knock Off, Giga Drain/Liquid Ooze, Counter/Mirror Coat, Destiny Bond) on authored states.

Each against the Showdown f76228a1 rule it ports (cited in gpu_engine/transition.py, damage_moves.py and
items.py). The two-engine probes (gpu_engine/effect_probes/*.json) are the Showdown gate; these are the
component checks, the device/B2 tracker checks (removal credit, the Destiny Bond bag, reveals, the item
narrowing reset of design 13 item 4) and the A2 regression plug-in. No chance is drawn beyond the fixed block.
"""
import importlib.util
import json
import shutil
import subprocess
import unittest

import numpy as np

from _showdown_root import requires_showdown, showdown_root

MOVES = ['_none', 'knockoff', 'gigadrain', 'counter', 'mirrorcoat', 'destinybond', 'hiddenpowerfire', 'tackle', 'surf',
         'splash', 'protect', 'substitute', 'struggle', 'sleeptalk', 'pursuit', 'endure']
(KNOCKOFF, GIGADRAIN, COUNTER, MIRRORCOAT, DESTINYBOND, HPFIRE, TACKLE, SURF, SPLASH, PROTECT, SUBSTITUTE, STRUGGLE,
 SLEEPTALK, PURSUIT, ENDURE) = range(1, 16)
TYPES = ['_none', 'normal', 'fighting', 'flying', 'poison', 'ground', 'rock', 'bug', 'ghost',
         'steel', 'fire', 'water', 'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark']
NORMAL, FIGHTING, GHOST, FIRE, WATER, GRASS, PSYCHIC, DARK = 1, 2, 8, 10, 11, 12, 14, 17
ABILITIES = ['_none', 'stickyhold', 'liquidooze', 'insomnia']
STICKYHOLD, LIQUIDOOZE, INSOMNIA = 1, 2, 3
ITEMS = ['_none', 'leftovers', 'lumberry', 'choiceband', 'salacberry']
LEFTOVERS, LUM, BAND, SALAC = 1, 2, 3, 4
NONE, PAR, BRN, PSN, TOX, SLP, FRZ = range(7)
B = 8
POWER = {KNOCKOFF: 20, GIGADRAIN: 60, TACKLE: 35, SURF: 95, STRUGGLE: 50, PURSUIT: 40}
TYPE = {KNOCKOFF: DARK, GIGADRAIN: GRASS, COUNTER: FIGHTING, MIRRORCOAT: PSYCHIC, DESTINYBOND: GHOST, SURF: WATER,
        HPFIRE: NORMAL, PURSUIT: DARK}
PHYSICAL = {TACKLE, STRUGGLE, COUNTER}
ALWAYS = {DESTINYBOND, SPLASH, PROTECT, SUBSTITUTE, STRUGGLE, SLEEPTALK, ENDURE}
PRIORITY = {COUNTER: -5, MIRRORCOAT: -5, PROTECT: 3, ENDURE: 4}
PROTECTABLE = {KNOCKOFF, GIGADRAIN, COUNTER, MIRRORCOAT, HPFIRE, TACKLE, SURF, STRUGGLE, PURSUIT}
CONTACT = {KNOCKOFF, COUNTER, TACKLE, STRUGGLE, PURSUIT}
BYPASS = {DESTINYBOND}


def metadata():
    rows = [None] + [{'id': name, 'source_type': 'Normal', 'category': 'Physical' if i in PHYSICAL else
                      ('Special' if i in POWER or i in (MIRRORCOAT, HPFIRE) else 'Status'),
                      'flags': {'protect': int(i in PROTECTABLE), 'contact': int(i in CONTACT), 'bypasssub': int(i in BYPASS)}}
                     for i, name in enumerate(MOVES) if i]
    return {'vocabulary': {'moves': MOVES, 'types': TYPES, 'abilities': ABILITIES, 'items': ITEMS,
                           'species': ['_none', 'a', 'b', 'c', 'd', 'e', 'f']}, 'moves': rows}


def components(pursuit=False):
    import jax.numpy as j
    from gpu_engine import (damage_moves, effects, items, pursuit as intercepts, sleep_talk, transition,
                            variable_moves)
    meta = metadata();n = len(MOVES)
    num = np.ones((18, 18), np.int32);num[NORMAL, GHOST] = 0;num[FIGHTING, GHOST] = 0;num[PSYCHIC, DARK] = 0
    table = transition.MoveTable(
        j.asarray([POWER.get(i, 0) for i in range(n)], j.int32), j.asarray([TYPE.get(i, NORMAL) for i in range(n)], j.int32),
        j.asarray([i in PHYSICAL for i in range(n)]), j.asarray([100] * n, j.int32),
        j.asarray([i in ALWAYS for i in range(n)]), j.asarray([PRIORITY.get(i, 0) for i in range(n)], j.int32),
        j.asarray(num), j.ones((18, 18), j.int32), j.array(STRUGGLE),
        effects=effects.from_metadata(meta), items=items.from_metadata(meta), damage_moves=damage_moves.from_metadata(meta),
        variable_moves=variable_moves.from_metadata(meta), sleep_talk=sleep_talk.from_metadata(meta),
        pursuit=intercepts.from_metadata(meta) if pursuit else None)
    return meta, table


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires JAX')
class FidelityA2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import transition
        from gpu_engine.state import BattleState, DESTINY_BOND, empty_state
        cls.jax, cls.j, cls.t, cls.State, cls.DB = jax, j, transition, BattleState, DESTINY_BOND
        cls.empty = staticmethod(empty_state)
        cls.meta, cls.table = components()
        cls.step = staticmethod(jax.jit(transition.step_with_chance))

    # -- authored batch: B lanes, side 0 faster, both actives at full 300 HP using Splash.
    def build(self, lanes):
        """lanes: per-lane {field: [(index without lane, value), ...]}; unlisted lanes are the base."""
        j = self.j;s = self.empty(B, seed=4242)
        s = s._replace(hp=j.full_like(s.hp, 300), max_hp=j.full_like(s.max_hp, 300),
            stats=j.full_like(s.stats, 200).at[..., 0].set(300).at[:, 0, :, 5].set(301),
            level=j.full_like(s.level, 100), types=s.types.at[..., 0].set(NORMAL), base_types=s.base_types.at[..., 0].set(NORMAL),
            species=j.broadcast_to(j.arange(1, 7, dtype=j.int32), (B, 2, 6)),
            moves=j.full_like(s.moves, SPLASH), pp=j.full_like(s.pp, 10), done=j.zeros(B, bool))
        arrays = {k: np.array(v) for k, v in s._asdict().items()}
        for lane, edits in enumerate(lanes):
            for field, writes in edits.items():
                for index, value in writes:arrays[field][(lane,) + tuple(index)] = value
        return self.State(**{k: j.asarray(v) for k, v in arrays.items()})

    def chance(self, lanes=()):
        """No thaw/full paralysis/confusion hit, no crit, max damage roll, accuracy roll 0, Protect succeeds, side 0
        wins ties. lanes: per-lane {(side, column): value}."""
        j = self.j;draws = np.zeros((B, 2, 12), np.int32);draws[:, :, 1:3] = 99;draws[:, :, 4] = 15;draws[:, :, 5] = 15
        for lane, edits in enumerate(lanes):
            for (side, column), value in edits.items():draws[lane, side, column] = value
        z = j.zeros((B, 2), j.int32)
        return self.t.ChanceInputs(draws=j.asarray(draws), phazing=z, secondary=j.full((B, 2), 99, j.int32), sleep_talk=z,
                                   entry_ties=z, initial_tie=j.zeros(B, j.int32))

    def play(self, s, a0=0, a1=0, chance=None, table=None):
        actions = self.j.zeros((B, 2), self.j.int32).at[:, 0].set(a0).at[:, 1].set(a1)
        out = self.step(s, actions, self.table if table is None else table, self.chance() if chance is None else chance)
        self.assertFalse(bool(np.asarray(out.invalid_actions).any()))
        return out.state

    def uses(self, move0, move1=SPLASH):
        return {'moves': [((0, 0, 0), move0), ((1, 0, 0), move1)]}

    @staticmethod
    def merge(*parts):
        out = {}
        for part in parts:
            for k, v in part.items():out.setdefault(k, []).extend(v)
        return out

    def slower0(self):
        """Side 1 moves first (side 0's speed below side 1's)."""
        return {'stats': [((0, 0, 5), 150)]}

    # -- tables ---------------------------------------------------------------------------------
    def test_tables(self):
        d, i, e = self.table.damage_moves, self.table.items, self.table.effects
        self.assertEqual(np.asarray(d.drain_den).nonzero()[0].tolist(), [GIGADRAIN]);self.assertEqual(int(d.drain_den[GIGADRAIN]), 2)
        self.assertEqual(np.asarray(d.liquid_ooze).nonzero()[0].tolist(), [LIQUIDOOZE])
        self.assertEqual((int(d.counter_kind[COUNTER]), int(d.counter_kind[MIRRORCOAT])), (1, 2))
        self.assertEqual(int(np.asarray(d.counter_kind).astype(bool).sum()), 2)
        self.assertEqual(np.asarray(d.hidden_power).nonzero()[0].tolist(), [HPFIRE])
        self.assertEqual(np.asarray(i.knock_off).nonzero()[0].tolist(), [KNOCKOFF])
        self.assertEqual(np.asarray(i.sticky_hold).nonzero()[0].tolist(), [STICKYHOLD])
        self.assertEqual(np.asarray(e.destinybond).nonzero()[0].tolist(), [DESTINYBOND])
        self.assertEqual(self.DB, 64)
        from gpu_engine import damage_moves as dm
        self.assertEqual(np.asarray(dm.drain_amount(self.j.asarray([1, 2, 7, 57, 57]), self.j.asarray([False, False, False, True, False]))).tolist(),
                         [1, 1, 3, 29, 28])
        self.assertEqual(np.asarray(dm.counter_damage(self.j.asarray([0, 1, 40]))).tolist(), [1, 2, 80])

    # -- A2.1 Knock Off ---------------------------------------------------------------------------
    def test_knock_off_removes_the_item_for_good(self):
        u = self.uses(KNOCKOFF);hurt = {'hp': [((1, 0), 200)]}
        s = self.build([
            self.merge(u, hurt, {'item': [((1, 0), LEFTOVERS)]}),                     # knocked: no Leftovers heal
            self.merge(u, hurt, {'item': [((1, 0), LEFTOVERS)], 'ability': [((1, 0), STICKYHOLD)]}),  # blocked, heals
            self.merge(u, {'item': [((1, 0), LEFTOVERS)], 'substitute_hp': [((1, 0), 75)]}),          # Substitute: kept
            self.merge(self.uses(KNOCKOFF, PROTECT), {'item': [((1, 0), LEFTOVERS)]}),                # Protect: kept
            self.merge(u, {'item': [((1, 0), NONE)]}),                                                # nothing to knock
            self.merge(u, {'item': [((1, 0), LEFTOVERS)], 'hp': [((1, 0), 1)]}),                      # KO: still knocked
            self.merge(u, {'item': [((1, 0), SALAC)], 'hp': [((1, 0), 80)]}),                         # berry gone before its pinch
            self.merge(u, {'item': [((1, 0), LEFTOVERS)], 'ability': [((1, 0), STICKYHOLD)], 'hp': [((1, 0), 1)]}),
        ])
        out = self.play(s)
        self.assertEqual(np.asarray(out.item)[:, 1, 0].tolist(), [NONE, LEFTOVERS, LEFTOVERS, LEFTOVERS, NONE, NONE, NONE, LEFTOVERS])
        self.assertEqual(np.asarray(out.item_knocked)[:, 1, 0].tolist(), [True, False, False, False, False, True, True, False])
        self.assertEqual(int(np.asarray(out.item_knocked).sum()), 3)
        hp = np.asarray(out.hp)[:, 1, 0];base = np.asarray(s.hp)[:, 1, 0]
        self.assertLess(hp[0], 200);self.assertEqual(int(hp[1]), min(300, int(hp[0]) + 300 // 16))   # same hit, then heal
        self.assertLess(int(out.substitute_hp[2, 1, 0]), 75);self.assertEqual(int(hp[2]), 300)
        self.assertEqual(int(hp[3]), 300);self.assertEqual(int(hp[5]), 0)
        self.assertEqual(np.asarray(out.boosts)[6, 1, 0, 4], 0)                                       # no Salac boost
        self.assertEqual(np.asarray(out.item)[:, 0, 0].tolist(), [NONE] * B)

    def test_choice_band_lock_ends_at_end_of_turn_after_knock_off(self):
        # Side 1 (faster) already locked into Tackle with Choice Band; side 0's Knock Off removes the Band.
        locked = {'item': [((1, 0), BAND)], 'choice_move': [((1, 0), TACKLE)], 'moves': [((1, 0, 0), TACKLE)]}
        s = self.build([self.merge({'moves': [((0, 0, 0), KNOCKOFF)]}, self.slower0(), locked),
                        self.merge({'moves': [((0, 0, 0), SPLASH)]}, self.slower0(), locked)])
        out = self.play(s)
        self.assertEqual(np.asarray(out.choice_move)[:2, 1, 0].tolist(), [0, TACKLE])
        self.assertEqual(np.asarray(out.item)[:2, 1, 0].tolist(), [NONE, BAND])
        # A pending replacement defers endTurn's DisableMove: the lock survives the step that requests it.
        s = self.build([self.merge({'moves': [((0, 0, 0), KNOCKOFF)]}, self.slower0(), locked, {'hp': [((0, 0), 1)]})])
        out = self.play(s)
        self.assertEqual(int(out.hp[0, 0, 0]), 0);self.assertTrue(bool(out.forced_switch[0, 0]))
        self.assertEqual(int(out.choice_move[0, 1, 0]), TACKLE)          # side 1 hit first; side 0 never knocked

    # -- A2.2 Giga Drain --------------------------------------------------------------------------
    def test_giga_drain_heals_half_and_liquid_ooze_reverses(self):
        u = self.uses(GIGADRAIN);low = {'hp': [((0, 0), 100)]}
        s = self.build([
            u,                                                                           # full HP: no heal
            self.merge(u, low),
            self.merge(u, low, {'substitute_hp': [((1, 0), 51)]}),                      # sub (51 < the 52 damage): ceil(51/2)
            self.merge(u, {'ability': [((1, 0), LIQUIDOOZE)]}),                         # Ooze at full HP
            self.merge(u, {'ability': [((1, 0), LIQUIDOOZE)], 'hp': [((0, 0), 5)]}),    # Ooze KO
            self.merge(u, low, {'hp': [((1, 0), 7)]}),                                  # capped by the target's HP
            self.merge(u, low, {'hp': [((1, 0), 1)]}),                                  # min 1
            self.merge(self.uses(GIGADRAIN, PROTECT), low),                             # Protect: nothing
        ])
        out = self.play(s)
        foe = 300 - np.asarray(out.hp)[:, 1, 0]
        own = np.asarray(out.hp)[:, 0, 0]
        dealt = int(foe[1])
        self.assertGreater(dealt, 2)
        self.assertEqual(int(own[0]), 300)
        self.assertEqual(int(own[1]), 100 + dealt // 2)
        self.assertEqual((int(out.substitute_hp[2, 1, 0]), int(own[2])), (0, 100 + 26))
        self.assertEqual(int(own[3]), 300 - dealt // 2)
        self.assertEqual(int(own[4]), 0);self.assertTrue(bool(out.forced_switch[4, 0]))
        self.assertEqual(int(own[5]), 100 + 3);self.assertEqual(int(own[6]), 101);self.assertEqual(int(own[7]), 100)

    # -- A2.3 Counter / Mirror Coat ---------------------------------------------------------------
    def test_counter_and_mirror_coat_double_the_qualifying_hit(self):
        first = self.slower0()
        def lane(own, foe, *extra):return self.merge({'moves': [((0, 0, 0), own), ((1, 0, 0), foe)]}, first, *extra)
        s = self.build([
            lane(COUNTER, TACKLE), lane(COUNTER, SURF), lane(MIRRORCOAT, SURF), lane(MIRRORCOAT, TACKLE),
            lane(COUNTER, HPFIRE, {'hidden_power_type': [((1, 0), FIRE)]}),
            lane(MIRRORCOAT, HPFIRE, {'hidden_power_type': [((1, 0), FIRE)]}),
            lane(COUNTER, SPLASH), lane(COUNTER, TACKLE, {'substitute_hp': [((0, 0), 120)]}),
        ])
        out = self.play(s)
        taken = 300 - np.asarray(out.hp)[:, 0, 0];given = 300 - np.asarray(out.hp)[:, 1, 0]
        self.assertTrue(all(int(taken[k]) > 0 for k in (0, 1, 2, 3, 4, 5)))
        self.assertEqual(given.tolist(), [2 * taken[0], 0, 2 * taken[2], 0, 2 * taken[4], 0, 0, 0])
        self.assertLess(int(out.substitute_hp[7, 0, 0]), 120);self.assertEqual(int(out.hp[7, 0, 0]), 300)
        self.assertEqual(np.asarray(out.pp)[:, 0, 0, 0].tolist(), [9] * B)        # PP spent on every failure

    def test_counter_immunity_substitute_target_and_called_counter(self):
        first = self.slower0()
        s = self.build([
            self.merge({'moves': [((0, 0, 0), COUNTER), ((1, 0, 0), TACKLE)], 'types': [((1, 0, 0), GHOST)]}, first),
            self.merge({'moves': [((0, 0, 0), MIRRORCOAT), ((1, 0, 0), SURF)], 'types': [((1, 0, 0), DARK)]}, first),
            self.merge({'moves': [((0, 0, 0), COUNTER), ((1, 0, 0), TACKLE)], 'substitute_hp': [((1, 0), 40)]}, first),
            # Sleep Talk calling Counter: beforeTurnCallback added no counter volatile, so it fails.
            self.merge({'moves': [((0, 0, 0), SLEEPTALK), ((0, 0, 1), COUNTER), ((0, 0, 2), 0), ((0, 0, 3), 0),
                                  ((1, 0, 0), TACKLE)], 'status': [((0, 0), SLP)], 'status_counter': [((0, 0), 3)]}, first),
            # Side 0 is faster here, but Counter's -5 still follows the foe's hit; a same-priority reply
            # would not: Mirror Coat after the foe's Mirror Coat (no record yet) fails.
            {'moves': [((0, 0, 0), COUNTER), ((1, 0, 0), TACKLE)]},
            {'moves': [((0, 0, 0), MIRRORCOAT), ((1, 0, 0), MIRRORCOAT)]},
        ])
        out = self.play(s)
        given = 300 - np.asarray(out.hp)[:, 1, 0]
        self.assertEqual(given[:2].tolist(), [0, 0])
        taken = 300 - int(out.hp[2, 0, 0])
        self.assertEqual((int(out.substitute_hp[2, 1, 0]), int(given[2])), (0, 0))     # 2x taken > 40 breaks the sub only
        self.assertGreater(taken, 20)
        self.assertEqual(int(given[3]), 0)
        self.assertEqual(int(given[4]), 2 * (300 - int(out.hp[4, 0, 0])))             # priority -5 still replies
        self.assertEqual((int(out.hp[5, 0, 0]), int(out.hp[5, 1, 0])), (300, 300))
        self.assertEqual(np.asarray(out.pp)[3, 0, 0].tolist(), [9, 10, 10, 10])         # Sleep Talk spends its own PP

    # -- A2.4 Destiny Bond ------------------------------------------------------------------------
    def test_destiny_bond_lifetime_and_trigger(self):
        bond = {'volatile': [((0, 0), 64)]};frail = {'hp': [((0, 0), 1)]}
        first = self.slower0()
        s = self.build([
            self.merge(self.uses(DESTINYBOND, TACKLE), frail),                          # bonds first, KO'd: both faint
            self.merge(self.uses(SPLASH, TACKLE), bond, frail),                         # moved again: bond gone
            self.merge(self.uses(SPLASH, TACKLE), bond, frail, first),                  # KO'd before moving: bond holds
            self.merge(self.uses(SPLASH, TACKLE), bond, frail, {'status': [((0, 0), SLP)], 'status_counter': [((0, 0), 3)]}),
            self.merge(self.uses(DESTINYBOND, TACKLE), bond, frail),                    # consecutive use holds
            self.merge(self.uses(DESTINYBOND), {'hp': [((0, 0), 1)], 'status': [((0, 0), PSN)]}),   # residual KO: no trigger
            self.merge(self.uses(DESTINYBOND, TACKLE)),                                 # survives: bond kept
            self.merge(self.uses(DESTINYBOND, TACKLE), frail, {'substitute_hp': [((0, 0), 50)]}),   # hit lands on the sub
        ])
        out = self.play(s)
        hp = np.asarray(out.hp);vol = np.asarray(out.volatile)
        self.assertEqual(hp[:, 0, 0].tolist()[:6], [0, 0, 0, 0, 0, 0])
        self.assertEqual(hp[:, 1, 0].tolist()[:6], [0, 300, 0, 300, 0, 300])
        self.assertEqual((vol[:, 0, 0] & 64).tolist(), [0, 0, 0, 0, 0, 0, 64, 64])     # fainted holders cleared
        self.assertEqual(int(hp[7, 0, 0]), 1);self.assertEqual(bool(out.done[0]), False)
        self.assertTrue(bool(out.forced_switch[0, 1]))
        # Switching out clears it; Baton Pass never copies it (baton.py keeps only Leech Seed).
        s = self.build([self.merge({'volatile': [((0, 0), 64)]})])
        out = self.play(s, a0=4)
        self.assertEqual(int(out.volatile[0, 0, 0]) & 64, 0);self.assertEqual(int(out.volatile[0, 0, 1]) & 64, 0)

    def test_destiny_bond_both_last_mons_tie(self):
        last = {'hp': [((side, k), 0) for side in (0, 1) for k in range(1, 6)]}
        s = self.build([self.merge(self.uses(DESTINYBOND, TACKLE), last, {'hp': [((0, 0), 1)]})])
        out = self.play(s)
        self.assertTrue(bool(out.done[0]));self.assertEqual(int(out.winner[0]), 2)

    def test_pursuit_ko_of_a_bonded_switcher_faints_the_pursuer(self):
        _, table = components(pursuit=True)
        s = self.build([self.merge({'moves': [((1, 0, 0), PURSUIT)], 'volatile': [((0, 0), 64)], 'hp': [((0, 0), 1)]}),
                        self.merge({'moves': [((1, 0, 0), PURSUIT)], 'hp': [((0, 0), 1)]})])
        out = self.play(s, a0=4, table=table)
        self.assertEqual(np.asarray(out.hp)[:2, 0, 0].tolist(), [0, 0])
        self.assertEqual(np.asarray(out.hp)[:2, 1, 0].tolist(), [0, 300])
        self.assertEqual(np.asarray(out.active)[:2, 0].tolist(), [1, 1])                # the chosen switch completes

    # -- device trackers --------------------------------------------------------------------------
    def device(self, s, a0=0, a1=0, table=None, chance=None):
        table = self.table if table is None else table
        init = self.t.initialize_with_public(s, table)
        actions = self.j.zeros((B, 2), self.j.int32).at[:, 0].set(a0).at[:, 1].set(a1)
        out = self.jax.jit(lambda st, a, tk, c: self.t.step_with_public(st, a, table, tk, chance_inputs=c))(
            init.state, actions, init.trackers, self.chance() if chance is None else chance)
        return init.trackers, out

    def test_device_knock_off_reveal_removal_credit_and_sticky_hold(self):
        from gpu_engine import engine_device_trackers as tr, public_active
        u = self.uses(KNOCKOFF)
        s = self.build([self.merge(u, {'item': [((1, 0), LEFTOVERS)]}),
                        self.merge(u, {'item': [((1, 0), LEFTOVERS)], 'ability': [((1, 0), STICKYHOLD)]}),
                        self.merge(u, {'item': [((1, 0), LEFTOVERS)], 'substitute_hp': [((1, 0), 75)]}),
                        self.merge(self.uses(GIGADRAIN), {'ability': [((1, 0), LIQUIDOOZE)]}),
                        self.merge(self.uses(DESTINYBOND))])
        _, out = self.device(s)
        t = out.trackers
        # Lane 1 keeps its item (Sticky Hold) and heals at the residual, which reveals it the usual way.
        self.assertEqual(np.asarray(t.shown_item)[:3, 1, 0].tolist(), [LEFTOVERS, LEFTOVERS, 0])
        self.assertEqual(np.asarray(t.item_removed)[:3, 1, 0].tolist(), [True, False, False])
        self.assertEqual(np.asarray(t.field_features['items_removed_credit'])[:4].tolist(), [[0, 1], [0, 0], [0, 0], [0, 0]])
        self.assertEqual(np.asarray(t.shown_ability)[:4, 1, 0].tolist(), [0, STICKYHOLD, 0, LIQUIDOOZE])
        for observer in (0, 1):
            view = tr.readout(t, self.j.full(B, observer, self.j.int32))
            self.assertEqual(float(view.field_features['self_items_removed_credit'][0]), [0, 1][observer])
            self.assertEqual(float(view.field_features['opp_items_removed_credit'][0]), [1, 0][observer])
        name = public_active.NAMES.index('destinybond')
        self.assertEqual(np.asarray(t.public_volatiles)[:, 0, 0, name].tolist(), [False] * 4 + [True] + [False] * 3)
        # The next move attempt expires it (B2 EXPIRE_ACTION on the holder's move line); a second
        # Destiny Bond sets it again (-singlemove). Same trackers, next step.
        state = out.step.state
        state = state._replace(moves=state.moves.at[:, 0, 0, 0].set(SPLASH).at[5, 0, 0, 0].set(DESTINYBOND),
            volatile=state.volatile.at[5, 0, 0].set(64))
        step = self.jax.jit(lambda st, a, tk, c: self.t.step_with_public(st, a, self.table, tk, chance_inputs=c))
        again = step(state, self.j.zeros((B, 2), self.j.int32), t, self.chance())
        self.assertEqual(np.asarray(again.trackers.public_volatiles)[:, 0, 0, name].tolist(), [False] * 5 + [True] + [False] * 2)
        self.assertEqual((np.asarray(again.step.state.volatile)[:, 0, 0] & 64).tolist(), [0] * 5 + [64] + [0] * 2)


class KnockOffNarrowingB2Tests(unittest.TestCase):
    """Design 13 item 4: after a Knock Off the device's hard mask equals B2's rebuilt belief, whose matching
    item key is 0 for a mutated item (public_items.matching_item, item_belief_narrowing=False)."""
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_item_context as fixture
        fixture.PublicItemContextTests.setUpClass()
        cls.f = fixture.PublicItemContextTests()

    def tables(self):
        """Device belief tables over a tiny engine namespace, built from B2's own compiled source (the
        engine_device_belief.compile_tables join for these fields)."""
        from types import SimpleNamespace
        import jax.numpy as j
        host = self.f.host;src = host.device
        species = ['_none', 'Snorlax', 'Absol'];items = ['_none', 'Leftovers', 'Choice Band'];moves = ['_none', 'surf', 'bodyslam']
        sid = np.asarray([host.species_id(n) if i else 0 for i, n in enumerate(species)], np.int32)
        def predicates(kind, names):
            ids = [host.predicate_id(kind, n) if i else 0 for i, n in enumerate(names)]
            out = np.asarray(src.predicates)[ids][:, sid].transpose(1, 0, 2).copy()
            out[:, 0] = np.asarray(src.pool)[sid];return j.asarray(out)
        return SimpleNamespace(species_initial_mask=j.asarray(np.asarray(src.pool)[sid]), move_mask=predicates('move', moves),
            ability_mask=predicates('ability', ['_none', 'Immunity']), item_mask=predicates('item', items))

    def test_knock_off_drops_item_narrowing_like_b2(self):
        import jax.numpy as j
        from gpu_engine import engine_device_public_hooks as h
        from gpu_engine.state import empty_state
        prefix = [*self.f.prefix, '|move|p1a: Snorlax|Surf|p2a: Absol', '|-heal|p1a: Snorlax|90/100|[from] item: Leftovers']
        knocked = prefix + ['|-enditem|p1a: Snorlax|Leftovers|[from] move: Knock Off|[of] p2a: Absol']
        b2 = {}
        for name, lines in (('revealed', prefix), ('knocked', knocked)):
            result, _ = self.f.run_lines(lines)
            _, summary = self.f.b.materialize(result.beliefs, self.f.host.device)
            b2[name] = np.asarray(summary.effective[0, 1, 0])                 # p1's Snorlax seen by p2
            if name == 'knocked':
                self.assertTrue(bool(result.context.items.mutated[0, 0, 0]));self.assertTrue(bool(result.context.items.removed[0, 0, 0]))
        self.assertFalse(np.array_equal(b2['revealed'], b2['knocked']))
        tables = self.tables()
        s = empty_state(1)
        s = s._replace(species=s.species.at[0, 0, 0].set(1).at[0, 1, 0].set(2), level=j.full_like(s.level, 80),
            hp=j.full_like(s.hp, 100), max_hp=j.full_like(s.max_hp, 100), item=s.item.at[0, 0, 0].set(1), done=j.zeros(1, bool))
        c = h.initialize(s, tables);yes = j.ones(1, bool)
        h.move(c, s, 0, j.ones(1, j.int32), yes, pp_use=True)                     # Surf (engine move 1)
        h.reveal_item(c, s, 0, j.ones(1, j.int32), yes)                            # Leftovers
        np.testing.assert_array_equal(np.asarray(c.trackers.belief_mask[0, 0, 0]), b2['revealed'])
        after = s._replace(item=s.item.at[0, 0, 0].set(0), item_knocked=s.item_knocked.at[0, 0, 0].set(True))
        h.knock_off(c, s, after, 0, yes)
        np.testing.assert_array_equal(np.asarray(c.trackers.belief_mask[0, 0, 0]), b2['knocked'])
        np.testing.assert_array_equal(np.asarray(c.trackers.visible_belief.visible_hard[0, 0, 0]), b2['knocked'])
        self.assertEqual((int(c.trackers.shown_item[0, 0, 0]), bool(c.trackers.item_removed[0, 0, 0])), (1, True))
        self.assertEqual(c.trackers.field_features['items_removed_credit'].tolist(), [[1, 0]])


class CureClauseParityTests(unittest.TestCase):
    """Review L4: B2's Sleep Clause ledger (public_sleep_clause) against the device port for Aromatherapy's
    -cureteam (clears the inducing side's holder and the cured seat's victims) and Refresh's -curestatus of a
    non-sleep status (leaves another member's clause entries)."""
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_sleep_clause as fixture
        fixture.SleepClauseTests.setUpClass()
        cls.f = fixture.SleepClauseTests()

    def device(self, events):
        """events over p1 Snorlax (slot 0) and p1 Gardevoir (slot 1) vs p2 Absol: ('status', mon, code), ('switch', mon),
        ('cureteam',). Returns the device readout's four clause fields for both observers."""
        import jax.numpy as j
        from gpu_engine import engine_device_public_hooks as h, engine_device_trackers as tr
        from gpu_engine.state import empty_state
        s = empty_state(1)
        s = s._replace(species=s.species.at[0, 0, :2].set(j.asarray([1, 2], j.int32)).at[0, 1, 0].set(3),
            level=j.full_like(s.level, 80), hp=j.full_like(s.hp, 100), max_hp=j.full_like(s.max_hp, 100), done=j.zeros(1, bool))
        c = h.initialize(s, None);yes = j.ones(1, bool)
        for event in events:
            if event[0] == 'status':
                s = s._replace(status=s.status.at[0, 0, event[1]].set(event[2]));h.observe(c, s, yes)
            elif event[0] == 'switch':
                old = s.active[:, 0];s = s._replace(active=s.active.at[0, 0].set(event[1]))
                h.install(c, s, 0, yes);h.withdraw(c, s, 0, old, yes)
            else:
                before = s;s = s._replace(status=j.zeros_like(s.status).at[0, 1].set(s.status[0, 1]))
                h.party_cure(c, before, s, 0, yes)
        out = {}
        for owner in (0, 1):
            view = tr.readout(c.trackers, j.full(1, owner, j.int32))
            out[owner] = [bool(view.field_features[k][0]) for k in
                          ('self_sleep_clause', 'opp_sleep_clause', 'sleep_clause_blocks_self', 'sleep_clause_blocks_opp')]
        return out

    def b2(self, lines):
        ledger, _ = self.f.run_prefixes(lines)
        out = {}
        for owner in (0, 1):
            got = self.f.c.project(ledger, self.f.tables, observer_side=owner,
                                   expected_vocabulary=self.f.catalog.species.device.vocabulary_binding)
            out[owner] = [bool(x) for x in np.asarray(got.numeric[0])[list(self.f.c.COLUMNS)]]
        return out

    def test_aromatherapy_cureteam_and_refresh_curestatus(self):
        prefix = [*self.f.f.public.prefix, '|turn|1', '|-status|p1a: Snorlax|slp',
                  '|switch|p1a: Gardevoir|Gardevoir, L80|100/100', '|-status|p1a: Gardevoir|par']
        events = [('status', 0, SLP), ('switch', 1), ('status', 1, PAR)]
        for lines, more in ((['|-curestatus|p1a: Gardevoir|par|[msg]'], [('status', 1, NONE)]),
                            (['|-cureteam|p1a: Gardevoir|[from] move: Aromatherapy'], [('cureteam',)])):
            with self.subTest(lines[-1]):
                b2, device = self.b2(prefix + lines), self.device(events + more)
                self.assertEqual(device, b2)
        self.assertTrue(any(self.b2(prefix + ['|-curestatus|p1a: Gardevoir|par|[msg]'])[0]))       # the clause survives Refresh
        self.assertFalse(any(self.b2(prefix + ['|-cureteam|p1a: Gardevoir|[from] move: Aromatherapy'])[0]))


@requires_showdown()
@unittest.skipUnless(shutil.which('node'), 'requires node')
class ShowdownAbilityLineTests(unittest.TestCase):
    """The Showdown f76228a1 lines behind the device's A2 reveals, read by B2's attribution: Sticky Hold's
    -activate on a blocked Knock Off and Liquid Ooze's -damage [from] ability [of] the drained target."""
    def test_b2_attributes_sticky_hold_and_liquid_ooze(self):
        from pokezero.belief import _confirmed_ability_from_event
        from pokezero.showdown import _public_event_from_line
        code = """const {Battle}=require(process.argv[1]+'/dist/sim/battle.js');
function mon(species,ability,moves,item){return {name:species,species,ability,item:item||'',moves,nature:'Hardy',level:80,gender:'M',
 evs:{hp:84,atk:84,def:84,spa:84,spd:84,spe:84},ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31}};}
function run(p1,p2,choices){const b=new Battle({formatid:'gen3customgame@@@sleepclausemod',seed:[1,2,3,4]});
 b.setPlayer('p1',{name:'A',team:p1});b.setPlayer('p2',{name:'B',team:p2});for(const [x,y] of choices)b.makeChoices(x,y);return b.log;}
console.log(JSON.stringify({
 sticky:run([mon('Sableye','Keen Eye',['knockoff'])],[mon('Muk','Sticky Hold',['splash'],'Leftovers')],[['move 1','move 1']]),
 knocked:run([mon('Sableye','Keen Eye',['knockoff'])],[mon('Snorlax','Thick Fat',['splash'],'Leftovers')],[['move 1','move 1']]),
 ooze:run([mon('Exeggutor','Chlorophyll',['gigadrain'])],[mon('Tentacruel','Liquid Ooze',['splash'])],[['move 1','move 1']])}));"""
        logs = json.loads(subprocess.check_output(['node', '-e', code, str(showdown_root())], text=True))
        def revealed(log):
            found = set()
            for line in log:
                ident, name = _confirmed_ability_from_event(_public_event_from_line(line))
                if ident and name and ident.startswith('p2'):found.add(name)
            return found
        self.assertEqual(revealed(logs['sticky']), {'Sticky Hold'})
        self.assertEqual(revealed(logs['knocked']), set())
        self.assertIn('|-enditem|p2a: Snorlax|Leftovers|[from] move: Knock Off|[of] p1a: Sableye', logs['knocked'])
        self.assertEqual(revealed(logs['ooze']), {'Liquid Ooze'})
        self.assertTrue(any(line.startswith('|-damage|p1a: Exeggutor') and 'Liquid Ooze' in line for line in logs['ooze']))


class RegressPluginTests(unittest.TestCase):
    """A2's plug-in allowances and the tightened cascade rules (review L1-L3)."""
    def test_a2_effects_declare_allowed_fields(self):
        from gpu_engine.v2e_regress import ALLOWED, PLUGINS, lane_verdict
        self.assertTrue(all(e in ALLOWED for e in PLUGINS['A2']))
        ok = {('state.item', 1), ('state.item_knocked', 1), ("trackers.field_features['items_removed_credit']", 1),
              ('state.choice_move', 1), ('trackers.shown_item', 1), ('trackers.item_removed', 1), ('state.hp', 0)}
        self.assertEqual(lane_verdict(ok, {'knockoff': [0]}, None)[0], [])
        self.assertEqual(lane_verdict({('state.item_knocked', 0)}, {'knockoff': [0]}, None)[0], [('state.item_knocked', 0)])
        self.assertEqual(lane_verdict({('state.volatile', 0), ('trackers.public_volatiles', 0)}, {'destinybond': [0]}, None)[0], [])
        self.assertEqual(lane_verdict({('state.volatile', 1)}, {'destinybond': [0]}, None)[0], [('state.volatile', 1)])
        self.assertEqual(lane_verdict({('state.types', 1), ('trackers.shown_ability', 1)}, {'counter': [0]}, None)[0], [])
        # Counter/Mirror Coat now break a target's Substitute (device presence and bag) and can decide whether its
        # side's pending Wish lands with a heal line.
        sub = {("trackers.mon_features['sub_present']", 1), ('trackers.public_volatiles', 1), ('state.substitute_hp', 1),
               ("trackers.field_features['wish_pending']", 1), ("trackers.field_features['wish_turns']", 1)}
        for effect in ('counter', 'mirrorcoat'):
            self.assertEqual(lane_verdict(sub, {effect: [0]}, None)[0], [])
            user = ("trackers.mon_features['sub_present']", 0)       # the replier's own Substitute is not its effect
            self.assertEqual(lane_verdict({user}, {effect: [0]}, None)[0], [user])
        self.assertEqual(lane_verdict({('state.hp', 0), ('trackers.shown_ability', 1)}, {'gigadrain': [0]}, None)[0], [])
        # L3: a cure's port allowance is the sleep-clause entries only.
        self.assertEqual(lane_verdict({('trackers.public_port.holder', 1), ('trackers.public_port.victims', 0)},
                                      {'refresh': [0]}, None)[0], [])
        self.assertEqual(lane_verdict({('trackers.public_port.blocked', 0)}, {'aromatherapy': [0]}, None)[0],
                         [('trackers.public_port.blocked', 0)])

    def test_link_leaves_per_mon_and_phase_shift_tracker_families(self):
        from gpu_engine.v2e_regress import active_mask, departed_mask, lane_verdict, link_ok
        ko = np.zeros((2, 6), bool);ko[1, 2] = True
        acting = active_mask(np.array([0, 2])) | active_mask(np.array([0, 2]))
        stay = departed_mask(np.array([0, 2]), np.array([0, 2]))
        # last_move: the fainted mon's and the actives' entries only (L2).
        self.assertEqual(link_ok('state.last_move', 0, None, None, ko, acting, departed=stay).tolist(), [True] + [False] * 5)
        self.assertEqual(link_ok('state.last_move', 1, None, None, ko, acting, departed=stay).tolist(),
                         [False, False, True, False, False, False])
        # trap_source: a link pointing at the fainted foe is released too; an active that stayed (side 0's slot 0)
        # no longer passes (R carries review L2: only a fainted or departed mon releases a link).
        a = np.array([-1, 2, -1, -1, -1, -1]);b = np.array([-1, -1, -1, -1, -1, 4])
        self.assertEqual(link_ok('state.trap_source', 0, a, b, ko, acting, departed=stay).tolist(),
                         [False, True, False, False, False, False])
        effects = {'curse': [0]}
        mon_ok = lambda leaf, side: side == 0
        cascade = dict(ko_sides=np.array([True, False]), mon_ok=mon_ok)
        self.assertEqual(lane_verdict({('state.last_move', 0), ('state.trap_source', 0)}, effects, None, cascade=cascade)[0], [])
        self.assertEqual(lane_verdict({('state.last_move', 1)}, effects, None, cascade=cascade)[0], [('state.last_move', 1)])
        # L1: a phase shift accepts the residual handlers' tracker families on the non-KO seat, not every tracker.
        phase = dict(cascade, families=('residual', 'residual_trackers'))
        self.assertEqual(lane_verdict({('trackers.shown_item', 1), ("trackers.mon_features['turns_active']", 1),
                                       ('trackers.public_port.post_upkeep', 1), ('state.item', 1)}, effects, None, cascade=phase)[0], [])
        self.assertEqual(lane_verdict({('trackers.observed_moves', 1), ('trackers.own_order', 1)}, effects, None, cascade=phase)[0],
                         [('trackers.observed_moves', 1), ('trackers.own_order', 1)])
        self.assertEqual(lane_verdict({('trackers.observed_moves', 0)}, effects, None, cascade=phase)[0], [])


if __name__ == '__main__':
    unittest.main()
