"""Fidelity patch M1: Hyper Beam's recharge (with Truant) and Trick, on authored states.

The rules follow Showdown f76228a1 as cited in gpu_engine/transition.py (data/moves.ts hyperbeam and trick, data/conditions.ts
mustrecharge and choicelock, sim/pokemon.ts takeItem/getLockedMove); the two-engine probes
(gpu_engine/effect_probes/{hyperbeam,trick}*.json) are the Showdown gate. Device: the mustrecharge bag (B2's stale bit), the
recharge request collapse, Trick's item ledger and per-stint Choice flags (B2 parity cases live in
test_gpu_engine_device_observation_boundary_parity.py). Also the M1 regression plug-in and its post-state rules.
"""
import importlib.util
import unittest

import numpy as np

TYPES = ['_none', 'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug',
         'rock', 'ghost', 'dragon', 'dark', 'steel']
NORMAL, PSYCHIC, GHOST, DARK = 1, 11, 14, 16
MOVES = ['_none', 'struggle', 'splash', 'hyperbeam', 'trick', 'tackle', 'protect', 'roar', 'explosion', 'return', 'encore']
STRUGGLE, SPLASH, HYPERBEAM, TRICK, TACKLE, PROTECT, ROAR, EXPLOSION, RETURN, ENCORE = range(1, 11)
ABILITIES = ['_none', 'truant', 'stickyhold', 'keeneye']
TRUANT, STICKYHOLD, KEENEYE = range(1, 4)
ITEMS = ['_none', 'leftovers', 'lumberry', 'choiceband']
LEFTOVERS, LUM, BAND = range(1, 4)
SLP, PSN = 5, 3
HAS_JAX = bool(importlib.util.find_spec('jax'))


def metadata():
    def row(name, power, typ, category, accuracy=100, flags=None, priority=0):
        return dict(id=name, base_power=power, type=typ, source_type=TYPES[typ].title(), category=category, pp=10,
                    accuracy=accuracy, priority=priority, flags={'protect': 1} if flags is None else flags)
    rows = [None, row('struggle', 50, NORMAL, 'Physical', True, {'contact': 1}), row('splash', 0, NORMAL, 'Status', True, {}),
            row('hyperbeam', 150, NORMAL, 'Physical', 90, {'recharge': 1, 'protect': 1, 'mirror': 1}),
            row('trick', 0, PSYCHIC, 'Status', 100, {'protect': 1, 'mirror': 1}),
            row('tackle', 35, NORMAL, 'Physical', 95, {'contact': 1, 'protect': 1}),
            row('protect', 0, NORMAL, 'Status', True, {}, priority=3),
            row('roar', 0, NORMAL, 'Status', 100, {'sound': 1, 'bypasssub': 1}, priority=-6),
            row('explosion', 250, NORMAL, 'Physical', 100, {'protect': 1}),
            row('return', 102, NORMAL, 'Physical', 100, {'contact': 1, 'protect': 1}),
            row('encore', 0, NORMAL, 'Status', 100, {'protect': 1, 'failencore': 1})]
    chart = np.ones((17, 17));chart[NORMAL - 1, GHOST - 1] = 0;chart[PSYCHIC - 1, DARK - 1] = 0
    return {'vocabulary': dict(moves=MOVES, abilities=ABILITIES, items=ITEMS, species=['_none', 'a', 'b', 'c', 'd', 'e', 'f'],
                               types=TYPES), 'moves': rows, 'type_chart': chart.tolist()}


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class TableAndMaskTests(unittest.TestCase):
    def test_rows(self):
        from gpu_engine import effects, items
        meta = metadata()
        self.assertEqual(np.flatnonzero(np.asarray(effects.from_metadata(meta).recharge)).tolist(), [HYPERBEAM])
        t = items.from_metadata(meta)
        self.assertEqual(np.flatnonzero(np.asarray(t.trick)).tolist(), [TRICK])
        self.assertEqual(items.ItemTable._fields[-3:], ('knock_off', 'sticky_hold', 'trick'))
        self.assertIn('recharge', effects.MoveEffects._fields)   # M1 appended it; later tiers append after it (M2: perishsong, yawn, wrap)

    def test_recharge_request_is_lane_zero_and_trapped(self):
        """getMoves(lockedMove 'recharge') is one Recharge row and getMoveRequestData sets trapped: only lane 0, over Choice
        and Encore filters and empty PP; a forced replacement of a fainted active is unaffected."""
        import jax.numpy as j
        from gpu_engine.state import empty_state, legal_actions, usable_moves, recharging
        s = empty_state(4)
        s = s._replace(hp=j.full_like(s.hp, 100), max_hp=j.full_like(s.max_hp, 100), done=j.zeros(4, bool),
            moves=j.broadcast_to(j.asarray([HYPERBEAM, TACKLE, SPLASH, RETURN], j.int32), s.moves.shape), pp=j.full_like(s.pp, 5))
        s = s._replace(recharge_counter=s.recharge_counter.at[:3, 0, 0].set(1),
            choice_move=s.choice_move.at[1, 0, 0].set(TACKLE), pp=s.pp.at[2, 0, 0].set(0),
            encore_move=s.encore_move.at[3, 0, 0].set(TACKLE), encore_counter=s.encore_counter.at[3, 0, 0].set(2),
            last_action_kind=s.last_action_kind.at[3, 0].set(2), last_action_index=s.last_action_index.at[3, 0].set(1))
        mask = np.asarray(legal_actions(s))
        for lane in range(3):
            self.assertEqual(mask[lane, 0].tolist(), [True] + [False] * 8, lane)
            self.assertEqual(mask[lane, 1].tolist(), [True] * 9, lane)
        self.assertEqual(mask[3, 0].tolist(), [False, True, False, False] + [True] * 5)     # not recharging: Encore as before
        self.assertEqual(np.asarray(recharging(s))[:, 0].tolist(), [True, True, True, False])
        self.assertEqual(np.asarray(usable_moves(s))[:3, 0].tolist(), [[True, False, False, False]] * 3)
        fainted = s._replace(hp=s.hp.at[0, 0, 0].set(0), recharge_counter=s.recharge_counter.at[0, 0, 0].set(0))
        self.assertEqual(np.asarray(legal_actions(fainted))[0, 0].tolist(), [False] * 4 + [True] * 5)


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class TransitionTests(unittest.TestCase):
    """step_with_chance on authored lanes: side 0 (faster) uses its slot-0 move, side 1 its slot 0 (Splash by default)."""
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import transition
        from gpu_engine.state import BattleState, DESTINY_BOND, empty_state
        cls.jax, cls.j, cls.t, cls.State, cls.DB = jax, j, transition, BattleState, DESTINY_BOND
        cls.empty = staticmethod(empty_state)
        cls.table = transition.tables_from_metadata(metadata())
        cls.step = staticmethod(jax.jit(lambda s, a, c: transition.step_with_chance(s, a, cls.table, c)))
        cls.mask = staticmethod(jax.jit(lambda s: transition.action_mask(s, cls.table)))

    def build(self, lanes, move=HYPERBEAM):
        j = self.j;n = len(lanes);s = self.empty(n, seed=4244)
        s = s._replace(hp=j.full_like(s.hp, 300), max_hp=j.full_like(s.max_hp, 300),
            stats=j.full_like(s.stats, 200).at[..., 0].set(300).at[:, 0, :, 5].set(301),
            level=j.full_like(s.level, 100), types=s.types.at[..., 0].set(NORMAL), base_types=s.base_types.at[..., 0].set(NORMAL),
            species=j.broadcast_to(j.arange(1, 7, dtype=j.int32), (n, 2, 6)),
            moves=j.full_like(s.moves, SPLASH).at[:, 0, 0, 0].set(move).at[:, 0, 0, 1].set(TACKLE),
            pp=j.full_like(s.pp, 10), done=j.zeros(n, bool))
        arrays = {k: np.array(v) for k, v in s._asdict().items()}
        for lane, edits in enumerate(lanes):
            for field, writes in edits.items():
                for index, value in writes:arrays[field][(lane,) + tuple(index)] = value
        arrays['base_types'] = arrays['types'].copy()
        return self.State(**{k: j.asarray(v) for k, v in arrays.items()})

    def chance(self, n, draws=()):
        """Hit, no crit, max damage roll, no thaw/paralysis/self-hit, duration roll 0, Protect succeeds."""
        j = self.j;d = np.zeros((n, 2, 12), np.int32);d[:, :, 1:3] = 99;d[:, :, 4] = 15;d[:, :, 5] = 15
        for lane, edits in enumerate(draws):
            for (side, column), value in edits.items():d[lane, side, column] = value
        z = j.zeros((n, 2), j.int32)
        return self.t.ChanceInputs(draws=j.asarray(d), phazing=z, secondary=j.full((n, 2), 99, j.int32), sleep_talk=z, entry_ties=z,
                                   initial_tie=j.zeros(n, j.int32), effect=j.zeros((n, 2, 4), j.int32))

    def play(self, s, chance, a0=0, a1=0):
        actions = self.j.zeros((s.hp.shape[0], 2), self.j.int32).at[:, 0].set(a0).at[:, 1].set(a1)
        out = self.step(s, actions, chance)
        self.assertFalse(bool(np.asarray(out.invalid_actions).any()))
        return out.state

    def test_hyper_beam_sets_recharge_on_a_hit_only(self):
        lanes = [{}, {}, {'types': [((1, 0, 0), GHOST)]}, {'substitute_hp': [((1, 0), 75)]}, {'hp': [((1, 0), 1)]},
                 {'moves': [((1, 0, 0), PROTECT)], 'stats': [((1, 0, 5), 400)]}, {'ability': [((0, 0), TRUANT)]}]
        s = self.build(lanes)
        out = self.play(s, self.chance(7, [{}, {(0, 6): 95}]))
        counter = np.asarray(out.recharge_counter)
        # Hit 2 -> residual 1; miss, Ghost immunity, Protect: none; a Substitute hit counts; a KO defers the residual (2).
        self.assertEqual(counter[:, 0, 0].tolist(), [1, 0, 0, 1, 2, 0, 1])
        self.assertFalse(counter[:, 1].any());self.assertFalse(counter[:, 0, 1:].any())
        self.assertEqual(np.asarray(out.pending_residual).tolist(), [False] * 4 + [True, False, False])
        self.assertLess(int(np.asarray(out.substitute_hp)[3, 1, 0]), 75)
        # The request that follows: lane 0 only, trapped (the KO lane's foe replaces first).
        mask = np.asarray(self.mask(out))
        for lane in (0, 3, 6):self.assertEqual(mask[lane, 0].tolist(), [True] + [False] * 8, lane)
        self.assertEqual(mask[1, 0].tolist(), [True, True, True, True] + [True] * 5)

    def test_recharge_turn_preempts_sleep_confusion_truant_and_destiny_bond(self):
        """The recharge turn: cant|recharge at BeforeMove priority 11, so no sleep or confusion tick, no Truant line (the
        loaf clock still flips at the residual), Destiny Bond ends (MoveAborted); no PP, no move, no history."""
        one = {'recharge_counter': [((0, 0), 1)]}
        lanes = [dict(one), dict(one, status=[((0, 0), SLP)], status_counter=[((0, 0), 3)]),
                 dict(one, confusion_counter=[((0, 0), 3)]), dict(one, ability=[((0, 0), TRUANT)], truant_loaf=[((0, 0), True)]),
                 dict(one, volatile=[((0, 0), 64)]),
                 # A Band holder's lock survives the recharge turn (without a Choice item it is stale and ends at endTurn).
                 dict(one, choice_move=[((0, 0), HYPERBEAM)], item=[((0, 0), ITEMS.index('choiceband'))]),
                 {'ability': [((0, 0), TRUANT)], 'truant_loaf': [((0, 0), True)]}]
        s = self.build(lanes)
        s = s._replace(last_action_kind=s.last_action_kind.at[:, 0].set(2), last_action_index=s.last_action_index.at[:, 0].set(0))
        out = self.play(s, self.chance(7, [{}, {}, {(0, 3): 99}]))
        self.assertFalse(np.asarray(out.recharge_counter)[:6].any())
        hp = np.asarray(out.hp);pp = np.asarray(out.pp)
        self.assertEqual(hp[:6, 1, 0].tolist(), [300] * 6);self.assertEqual(hp[:6, 0, 0].tolist(), [300] * 6)
        self.assertEqual(pp[:6, 0, 0, 0].tolist(), [10] * 6)
        self.assertEqual(int(np.asarray(out.status_counter)[1, 0, 0]), 3);self.assertEqual(int(np.asarray(out.status)[1, 0, 0]), SLP)
        self.assertEqual(int(np.asarray(out.confusion_counter)[2, 0, 0]), 3)
        self.assertEqual(np.asarray(out.truant_loaf)[[3, 6], 0, 0].tolist(), [False, False])    # the residual flip, both lanes
        self.assertEqual(int(np.asarray(out.volatile)[4, 0, 0]) & self.DB, 0)
        self.assertEqual(int(np.asarray(out.choice_move)[5, 0, 0]), HYPERBEAM)
        self.assertEqual(np.asarray(out.last_action_index)[:6, 0].tolist(), [0] * 6)
        self.assertEqual(np.asarray(out.last_move)[:6, 0, 0].tolist(), [0] * 6)
        self.assertEqual(int(np.asarray(out.pp)[6, 0, 0, 0]), 10)                       # the plain loaf, for comparison

    def test_recharge_pseudo_move_is_priority_zero_at_the_users_speed(self):
        """The foe Encores at priority 0. When the recharge goes first (faster recharger, or a Speed tie the coin draws[0][0]
        gives side 0), Encore finds no queued move left and lasts one turn longer (3+0+1); otherwise 3 (Showdown: the
        recharge action is a priority-0 move action at the user's Speed; its move-order tie is the same draws[0][0] coin)."""
        one = {'recharge_counter': [((0, 0), 1)], 'moves': [((1, 0, 0), ENCORE)]}
        lanes = [dict(one), dict(one, stats=[((1, 0, 5), 400)]), dict(one, stats=[((1, 0, 5), 301)]),
                 dict(one, stats=[((1, 0, 5), 301)])]
        s = self.build(lanes)
        s = s._replace(last_action_kind=s.last_action_kind.at[:, 0].set(2), last_action_index=s.last_action_index.at[:, 0].set(0))
        out = self.play(s, self.chance(4, [{}, {}, {(0, 0): 0}, {(0, 0): 1}]))
        self.assertFalse(np.asarray(out.recharge_counter).any())
        self.assertEqual(np.asarray(out.encore_move)[:, 0, 0].tolist(), [HYPERBEAM] * 4)
        # Encore's residual already ran once this turn (counter - 1).
        self.assertEqual(np.asarray(out.encore_counter)[:, 0, 0].tolist(), [3, 2, 3, 2])

    def test_a_faint_cancels_the_recharge_which_then_expires_at_the_residual(self):
        """The faster foe explodes (the recharger is a Ghost): Gen 3 cancels the queued recharge; mustrecharge stays 1 until
        the deferred residual of the replacement step, which ends it silently."""
        lane = {'recharge_counter': [((0, 0), 1)], 'types': [((0, 0, 0), GHOST)], 'moves': [((1, 0, 0), EXPLOSION)],
                'stats': [((1, 0, 5), 400)]}
        s = self.build([lane, lane])
        out = self.play(s, self.chance(2))
        self.assertEqual(np.asarray(out.hp)[:, 1, 0].tolist(), [0, 0]);self.assertEqual(np.asarray(out.hp)[:, 0, 0].tolist(), [300, 300])
        self.assertEqual(np.asarray(out.recharge_counter)[:, 0, 0].tolist(), [1, 1])
        self.assertTrue(np.asarray(out.pending_residual).all())
        after = self.play(out, self.chance(2), a0=0, a1=4)
        self.assertEqual(np.asarray(after.recharge_counter)[:, 0, 0].tolist(), [0, 0])
        self.assertEqual(np.asarray(after.active)[:, 1].tolist(), [1, 1])
        mask = np.asarray(self.mask(after))[:, 0]
        self.assertTrue(mask[:, :4].all())                                                    # free again

    def test_switch_out_and_faint_clear_the_counter(self):
        """Roar after the hit (priority -6) drags the user out: switchOut's clearVolatile. A KO after the hit: the faint's."""
        lanes = [{'moves': [((1, 0, 0), ROAR)]}, {'moves': [((1, 0, 0), EXPLOSION)], 'hp': [((0, 0), 10)]}]
        s = self.build(lanes)
        out = self.play(s, self.chance(2))
        self.assertFalse(np.asarray(out.recharge_counter).any())
        self.assertNotEqual(int(np.asarray(out.active)[0, 0]), 0)
        self.assertEqual(int(np.asarray(out.hp)[1, 0, 0]), 0)

    def test_trick_swaps_items_and_fails_where_showdown_fails(self):
        lanes = [
            {'item': [((0, 0), BAND), ((1, 0), LEFTOVERS)]},                                         # 0 swap
            {'item': [((0, 0), BAND), ((1, 0), LEFTOVERS)], 'ability': [((1, 0), STICKYHOLD)]},       # 1 Sticky Hold
            {'item': [((0, 0), BAND), ((1, 0), LEFTOVERS)], 'substitute_hp': [((1, 0), 75)]},         # 2 Substitute
            {'item': [((0, 0), BAND)], 'item_knocked': [((1, 0), True)]},                             # 3 target knocked off
            {'item': [((1, 0), LEFTOVERS)], 'item_knocked': [((0, 0), True)]},                        # 4 user knocked off
            {},                                                                                       # 5 neither holds one
            {'item': [((0, 0), BAND), ((1, 0), BAND)]},                                               # 6 Band <-> Band
            {'item': [((0, 0), BAND), ((1, 0), LEFTOVERS)], 'types': [((1, 0, 0), DARK)]},            # 7 Dark: hit anyway
            {'item': [((1, 0), LEFTOVERS)]},                                                          # 8 nothing for Leftovers
            {'item': [((0, 0), BAND), ((1, 0), LEFTOVERS)], 'moves': [((1, 0, 0), PROTECT)], 'stats': [((1, 0, 5), 400)]},  # 9
            {'item': [((0, 0), BAND), ((1, 0), LEFTOVERS)], 'stats': [((1, 0, 5), 400)]},             # 10 faster receiver
            {'item': [((0, 0), BAND), ((1, 0), LEFTOVERS)], 'choice_move': [((0, 0), TRICK)]},        # 11 locked Tricker
        ]
        s = self.build(lanes, move=TRICK)
        out = self.play(s, self.chance(len(lanes)))
        item = np.asarray(out.item)[:, :, 0];lock = np.asarray(out.choice_move)[:, :, 0]
        self.assertEqual(item.tolist(), [[LEFTOVERS, BAND], [BAND, LEFTOVERS], [BAND, LEFTOVERS], [BAND, 0], [0, LEFTOVERS], [0, 0],
            [BAND, BAND], [LEFTOVERS, BAND], [LEFTOVERS, 0], [BAND, LEFTOVERS], [LEFTOVERS, BAND], [LEFTOVERS, BAND]])
        # Locks: the Tricker locks only while it still holds the Band (a failed Trick, Band <-> Band); a slower receiver
        # locks into its move this turn, a faster one did not lock (it moved before the Band came); a stale lock ends at
        # endTurn (lane 11).
        self.assertEqual(lock.tolist(), [[0, SPLASH], [TRICK, 0], [TRICK, 0], [TRICK, 0], [0, 0], [0, 0], [TRICK, SPLASH],
            [0, SPLASH], [0, 0], [TRICK, 0], [0, 0], [0, SPLASH]])
        self.assertFalse(np.asarray(out.item_knocked)[[0, 6, 7, 8]].any())
        self.assertEqual(np.asarray(out.pp)[:, 0, 0, 0].tolist(), [9] * len(lanes))

    def test_tricked_lum_is_eaten_at_once_and_band_damage_follows_the_item(self):
        """A poisoned Tricker takes a Lum Berry and eats it after the move (Update); a slower receiver of the Band hits harder."""
        lanes = [{'item': [((0, 0), LEFTOVERS), ((1, 0), LUM)], 'status': [((0, 0), PSN)]},
                 {'item': [((0, 0), BAND)], 'moves': [((1, 0, 0), TACKLE)]}, {'moves': [((1, 0, 0), TACKLE)], 'item': [((0, 0), LEFTOVERS)]}]
        s = self.build(lanes, move=TRICK)
        out = self.play(s, self.chance(3))
        self.assertEqual(int(np.asarray(out.status)[0, 0, 0]), 0);self.assertEqual(np.asarray(out.item)[0, :, 0].tolist(), [0, LEFTOVERS])
        hp = np.asarray(out.hp)[:, 0, 0]
        self.assertLess(int(hp[1]), int(hp[2]) - 16)                          # Band x1.5 on the receiver's Tackle (lane 2: no Band)


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class DeviceTests(unittest.TestCase):
    """step_with_public: the mustrecharge bag, the recharge request collapse and Trick's ledger and flags."""
    @classmethod
    def setUpClass(cls):
        TransitionTests.setUpClass.__func__(cls)
        from gpu_engine.public_active import NAMES
        cls.RECHARGE = NAMES.index('mustrecharge')
        cls.pstep = staticmethod(cls.jax.jit(lambda st, a, tk, c: cls.t.step_with_public(st, a, cls.table, tk, chance_inputs=c)))

    build, chance = TransitionTests.build, TransitionTests.chance

    def run_public(self, s, chance, a0=0, a1=0, trackers=None):
        j = self.j
        if trackers is None:
            init = self.t.initialize_with_public(s, self.table);s, trackers = init.state, init.trackers
        actions = j.zeros((s.hp.shape[0], 2), j.int32).at[:, 0].set(a0).at[:, 1].set(a1)
        out = self.pstep(s, actions, trackers, chance)
        return out.step.state, out.trackers

    def slot(self, t, side):
        return np.asarray(t.active_slot)[:, side]

    def test_mustrecharge_bag_set_cleared_and_stale_like_b2(self):
        s = self.build([{}, {}, {'hp': [((1, 0), 1)]}])
        st, tk = self.run_public(s, self.chance(3))
        rows = np.arange(3);slot = self.slot(tk, 0)
        bag = np.asarray(tk.public_volatiles)[rows, 0, slot, self.RECHARGE]
        self.assertEqual(bag.tolist(), [True, True, True])                          # -mustrecharge, a KO included
        # Lane 0: the recharge turn's cant clears it. Lane 1: a faster foe explodes (the user a Ghost) and cancels the
        # recharge: the counter expires silently at the replacement's residual and B2 keeps the bit (its stale bit).
        st = st._replace(types=st.types.at[1, 0, 0, 0].set(GHOST), base_types=st.base_types.at[1, 0, 0, 0].set(GHOST),
            moves=st.moves.at[1, 1, 0, 0].set(EXPLOSION), stats=st.stats.at[1, 1, 0, 5].set(400))
        st2, tk2 = self.run_public(st, self.chance(3), a1=self.j.asarray([0, 0, 4]), trackers=tk)
        bag = np.asarray(tk2.public_volatiles)[rows, 0, slot, self.RECHARGE]
        self.assertEqual(bag[:2].tolist(), [False, True])
        self.assertEqual(np.asarray(st2.recharge_counter)[:2, 0, 0].tolist(), [0, 1])
        st3, tk3 = self.run_public(st2, self.chance(3), a1=self.j.asarray([0, 4, 0]), trackers=tk2)
        self.assertEqual(int(np.asarray(st3.recharge_counter)[1, 0, 0]), 0)
        self.assertTrue(bool(np.asarray(tk3.public_volatiles)[1, 0, slot[1], self.RECHARGE]))       # stale, as in B2

    def test_trick_ledger_flags_and_choice_lock_line(self):
        lanes = [{'item': [((0, 0), BAND), ((1, 0), LEFTOVERS)]}, {'item': [((0, 0), BAND), ((1, 0), BAND)]},
                 {'item': [((0, 0), BAND)]}, {'item': [((0, 0), BAND), ((1, 0), LEFTOVERS)], 'ability': [((1, 0), STICKYHOLD)]}]
        s = self.build(lanes, move=TRICK)
        st, tk = self.run_public(s, self.chance(4))
        rows = np.arange(4);f = tk.mon_features
        def mon(name, side):return np.asarray(getattr(tk, name) if name in ('shown_item', 'item_removed') else f[name])[rows, side, self.slot(tk, side)].tolist()
        # B2: revealed = received, else the item lost; removed = nothing received; mutated; original = the item held.
        self.assertEqual(mon('shown_item', 0), [LEFTOVERS, BAND, BAND, 0]);self.assertEqual(mon('shown_item', 1), [BAND, BAND, BAND, 0])
        self.assertEqual(mon('item_removed', 0), [False, False, True, False]);self.assertEqual(mon('item_removed', 1), [False] * 4)
        self.assertEqual(mon('item_mutated', 0), [1, 1, 1, 0]);self.assertEqual(mon('item_mutated', 1), [1, 1, 1, 0])
        self.assertEqual(mon('item_original', 0), [BAND, BAND, BAND, 0]);self.assertEqual(mon('item_original', 1), [LEFTOVERS, BAND, 0, 0])
        # Seat flags: -item is ITEM_SET (swapped, a received Choice item public), -enditem ITEM_END; the slower receiver's
        # own move line (Splash, after the -item) locks it publicly (CHOICE_MOVE).
        self.assertEqual(mon('item_swapped', 0), [1, 1, 0, 0]);self.assertEqual(mon('item_swapped', 1), [1, 1, 1, 0])
        self.assertEqual(mon('choice_public', 0), [0, 1, 0, 0]);self.assertEqual(mon('choice_public', 1), [1, 1, 1, 0])
        self.assertEqual(mon('choice_locked', 0), [0, 0, 0, 0]);self.assertEqual(mon('choice_locked', 1), [1, 1, 1, 0])
        # The Tricker's next own move line locks it publicly only when it received a Choice item (lane 1).
        st2, tk2 = self.run_public(st._replace(moves=st.moves.at[:, 0, 0, 0].set(SPLASH), choice_move=st.choice_move.at[:, 0].set(0)),
            self.chance(4), trackers=tk)
        locked = np.asarray(tk2.mon_features['choice_locked'])[rows, 0, self.slot(tk2, 0)].tolist()
        self.assertEqual(locked, [0, 1, 0, 0])

    def test_tricked_items_keep_the_ledger_and_consumption_ends_the_flags(self):
        """The Tricked-in Leftovers heals (its reveal keeps the ledger; the no-narrowing half is TrickNarrowingB2Tests); a
        consumption's -enditem (here the Tricked-in Lum Berry eaten at once) ends the seat flags (ITEM_END), not the ledger."""
        lanes = [{'item': [((0, 0), BAND), ((1, 0), LEFTOVERS)], 'hp': [((0, 0), 200)]},
                 {'item': [((0, 0), LEFTOVERS), ((1, 0), LUM)], 'status': [((0, 0), PSN)]}]
        s = self.build(lanes, move=TRICK)
        st, tk = self.run_public(s, self.chance(2))
        rows = np.arange(2);slot = self.slot(tk, 0)
        self.assertGreater(int(np.asarray(st.hp)[0, 0, 0]), 200)                             # Leftovers healed
        self.assertEqual(np.asarray(tk.item_removed)[rows, 0, slot].tolist(), [False, True])
        self.assertEqual(np.asarray(tk.mon_features['item_swapped'])[rows, 0, slot].tolist(), [1, 0])
        self.assertEqual(np.asarray(tk.mon_features['item_mutated'])[rows, 0, slot].tolist(), [1, 1])
        self.assertEqual(np.asarray(tk.shown_item)[rows, 0, slot].tolist(), [LEFTOVERS, LUM])



@unittest.skipUnless(HAS_JAX, 'requires JAX')
class TrickNarrowingB2Tests(unittest.TestCase):
    """Design 13 item 4 for Trick: B2's ledger after a Trick's lines (public_items TRICK_RECEIVE / REMOVE: revealed,
    removed, mutated, the partner-credited original) and its rebuilt belief, whose matching item key is 0 for a mutated
    item; the device's trick hook and a later reveal of the Tricked-in item (reveal_item) against them."""
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_fidelity_a2 as a2
        a2.KnockOffNarrowingB2Tests.setUpClass()
        cls.helper = a2.KnockOffNarrowingB2Tests();cls.f = cls.helper.f

    def b2(self, lines):
        result, _ = self.f.run_lines(lines)
        _, summary = self.f.b.materialize(result.beliefs, self.f.host.device)
        names = self.f.item_catalog.names;items = result.context.items
        ledger = {side: dict(revealed=names[int(items.revealed[0, side, 0])], original=names[int(items.original[0, side, 0])],
            removed=bool(items.removed[0, side, 0]), mutated=bool(items.mutated[0, side, 0])) for side in (0, 1)}
        # effective[batch, observer, slot]: p1's Snorlax as p2 sees it is [0, 1, 0]; p2's Absol as p1 sees it [0, 0, 0].
        return ledger, {0: np.asarray(summary.effective[0, 1, 0]), 1: np.asarray(summary.effective[0, 0, 0])}

    def test_trick_ledger_and_narrowing_match_b2(self):
        import jax.numpy as j
        from gpu_engine import engine_device_public_hooks as h
        from gpu_engine.state import empty_state
        names = {1: 'leftovers', 2: 'choiceband', 0: ''}
        prefix = [*self.f.prefix, '|move|p1a: Snorlax|Surf|p2a: Absol', '|-heal|p1a: Snorlax|90/100|[from] item: Leftovers']
        trick = prefix + ['|-activate|p1a: Snorlax|move: Trick|[of] p2a: Absol', '|-item|p2a: Absol|Leftovers|[from] move: Trick',
                          '|-item|p1a: Snorlax|Choice Band|[from] move: Trick']
        healed = trick + ['|-heal|p2a: Absol|90/100|[from] item: Leftovers']
        before_ledger, before = self.b2(prefix)
        trick_ledger, after = self.b2(trick)
        heal_ledger, heal = self.b2(healed)
        self.assertEqual(trick_ledger[0], dict(revealed='choiceband', original='leftovers', removed=False, mutated=True))
        self.assertEqual(trick_ledger[1], dict(revealed='leftovers', original='choiceband', removed=False, mutated=True))
        self.assertFalse(np.array_equal(before[0], after[0]))                       # Snorlax's Leftovers narrowing is gone
        np.testing.assert_array_equal(heal[1], after[1])                             # the Tricked-in Leftovers does not narrow
        tables = self.helper.tables()
        s = empty_state(1)
        s = s._replace(species=s.species.at[0, 0, 0].set(1).at[0, 1, 0].set(2), level=j.full_like(s.level, 80),
            hp=j.full_like(s.hp, 100), max_hp=j.full_like(s.max_hp, 100), item=s.item.at[0, 0, 0].set(1).at[0, 1, 0].set(2),
            done=j.zeros(1, bool))
        c = h.initialize(s, tables);yes = j.ones(1, bool)
        h.move(c, s, 0, j.ones(1, j.int32), yes, pp_use=True)                       # Surf (engine move 1)
        h.reveal_item(c, s, 0, j.ones(1, j.int32), yes)                              # Leftovers
        np.testing.assert_array_equal(np.asarray(c.trackers.belief_mask[0, 0, 0]), before[0])
        swapped = s._replace(item=s.item.at[0, 0, 0].set(2).at[0, 1, 0].set(1))
        h.trick(c, s, swapped, 0, yes, j.asarray([False, False, True]))
        for side in (0, 1):
            np.testing.assert_array_equal(np.asarray(c.trackers.belief_mask[0, side, 0]), after[side], err_msg=str(side))
            f = c.trackers.mon_features
            device = dict(revealed=names[int(c.trackers.shown_item[0, side, 0])], original=names[int(f['item_original'][0, side, 0])],
                removed=bool(c.trackers.item_removed[0, side, 0]), mutated=bool(f['item_mutated'][0, side, 0]))
            self.assertEqual(device, trick_ledger[side], side)
        h.reveal_item(c, swapped._replace(hp=swapped.hp.at[0, 1, 0].set(90)), 1, j.ones(1, j.int32), yes)
        np.testing.assert_array_equal(np.asarray(c.trackers.belief_mask[0, 1, 0]), heal[1])

    def test_one_sided_trick_ledger_matches_b2(self):
        """Nothing comes back: the giver's -enditem [silent] (REMOVE) reveals what it lost and marks it removed."""
        import jax.numpy as j
        from gpu_engine import engine_device_public_hooks as h
        from gpu_engine.state import empty_state
        names = {1: 'leftovers', 2: 'choiceband', 0: ''}
        lines = [*self.f.prefix, '|-activate|p1a: Snorlax|move: Trick|[of] p2a: Absol', '|-item|p2a: Absol|Choice Band|[from] move: Trick',
                 '|-enditem|p1a: Snorlax|Choice Band|[silent]|[from] move: Trick']
        ledger, _ = self.b2(lines)
        self.assertEqual(ledger[0], dict(revealed='choiceband', original='choiceband', removed=True, mutated=True))
        self.assertEqual(ledger[1], dict(revealed='choiceband', original='', removed=False, mutated=True))
        s = empty_state(1)
        s = s._replace(species=s.species.at[0, 0, 0].set(1).at[0, 1, 0].set(2), level=j.full_like(s.level, 80),
            hp=j.full_like(s.hp, 100), max_hp=j.full_like(s.max_hp, 100), item=s.item.at[0, 0, 0].set(2), done=j.zeros(1, bool))
        c = h.initialize(s, None);yes = j.ones(1, bool)
        h.trick(c, s, s._replace(item=s.item.at[0, 0, 0].set(0).at[0, 1, 0].set(2)), 0, yes, j.asarray([False, False, True]))
        f = c.trackers.mon_features
        for side in (0, 1):
            device = dict(revealed=names[int(c.trackers.shown_item[0, side, 0])], original=names[int(f['item_original'][0, side, 0])],
                removed=bool(c.trackers.item_removed[0, side, 0]), mutated=bool(f['item_mutated'][0, side, 0]))
            self.assertEqual(device, ledger[side], side)


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class FutureStateTests(unittest.TestCase):
    def test_policy_and_refusal(self):
        from gpu_engine import future_state as f
        from test_gpu_engine_future_state import FAKE, fixture
        self.assertIn('recharge_counter', f.FIELD_POLICY);self.assertEqual(f.SHAPES['recharge_counter'], (2, 6))
        self.assertIn('MUSTRECHARGE', f.NATIVE_PATHS['recharge_counter'][0])
        data, meta = fixture()
        self.assertIsInstance(f.native_input_from_jax(data, None, meta, module=FAKE), f.NativeInput)
        data['recharge_counter'][0][0] = 1
        out = f.native_input_from_jax(data, None, meta, module=FAKE)
        self.assertIsInstance(out, f.Refused);self.assertIn('RECHARGE_UNREPRESENTED', {g.code for g in out.gaps})
        data, meta = fixture();del data['recharge_counter']                     # optional in older data
        self.assertIsInstance(f.native_input_from_jax(data, None, meta, module=FAKE), f.NativeInput)


class RegressPluginTests(unittest.TestCase):
    def test_m1_effects_declare_allowed_fields(self):
        from gpu_engine.v2e_regress import ALL, ALLOWED, PLUGINS, lane_verdict
        self.assertTrue(all(e in ALLOWED for e in PLUGINS['M1']))
        self.assertIs(ALLOWED['recharge'], ALL)
        self.assertEqual(ALLOWED['hyperbeam'], dict(user=(), target=(), glob=()))
        for bad in (('state.recharge_counter', 0), ('trackers.public_volatiles', 0), ('state.hp', 1)):
            self.assertEqual(lane_verdict({bad}, {'hyperbeam': [0]}, None)[0], [bad])
        ok = {('state.item', 0), ('state.item', 1), ('state.choice_move', 0), ('state.choice_move', 1), ('state.hp', 0),
              ("trackers.mon_features['item_mutated']", 1), ("trackers.mon_features['choice_locked']", 1), ('trackers.shown_item', 0),
              ('trackers.belief_mask', 1), ('trackers.visible_belief.visible_hard', 1)}
        self.assertEqual(lane_verdict(ok, {'trick': [0]}, None)[0], [])
        # M1 review: a swapped berry's stages and cure and the bag are never an allowance (trick_rule / bag_rule witnesses).
        for bad in (('state.types', 1), ('state.trap_source', 0), ('state.encore_counter', 1), ('state.recharge_counter', 0),
                    ('state.volatile', 1), ('state.item_knocked', 1), ('state.status', 0), ('state.confusion_counter', 1),
                    ('state.boosts', 1), ('trackers.boosts', 0), ('trackers.public_volatiles', 0), ('trackers.status', 1),
                    ("trackers.mon_features['confusion_turns']", 1), ("trackers.mon_features['sub_present']", 1),
                    ('trackers.public_port.victims', 0)):
            self.assertEqual(lane_verdict({bad}, {'trick': [0]}, None)[0], [bad])

    def test_recharge_rule_needs_a_hit_and_the_bag_bit(self):
        """M1 review: the counter is explained only with post-state evidence the move connected and the bag bit set."""
        from gpu_engine.v2e_regress import recharge_rule
        counter = np.zeros((2, 6), np.int64);counter[0, 0] = 1
        bag = np.zeros((2, 6), bool);bag[0, 2] = True                    # the active (physical 0) holds public slot 2
        p2p = np.tile(np.array([2, 0, 1, 3, 4, 5]), (2, 1));post_active = np.zeros(2, np.int64);hp = np.full((2, 6), 100)
        departed = np.zeros((2, 6), bool)
        entries = {('state.recharge_counter', 0), ('trackers.public_volatiles', 0), ('state.hp', 1)}
        rule = lambda **kw: recharge_rule(entries, **dict(dict(counter=counter, bag=bag, hyperbeam=np.array([True, False]),
            evidence=np.array([True, False]), residual_ran=True, post_active=post_active, post_hp=hp, physical_to_public=p2p,
            departed=departed), **kw))
        rest, explained, cells = rule()
        self.assertEqual(rest, {('trackers.public_volatiles', 0), ('state.hp', 1)})   # the bag is bag_rule's, per cell
        self.assertEqual(explained, [('state.recharge_counter', 0)]);self.assertEqual(np.flatnonzero(cells[0]).tolist(), [2])
        self.assertEqual(rule(evidence=np.array([False, False]))[1], [])                  # RMISS: acted, never connected
        self.assertFalse(rule(evidence=np.array([False, False]))[2].any())
        self.assertEqual(rule(bag=np.zeros((2, 6), bool))[1], [])                         # RNOBAG: no bit at its slot
        stray = np.zeros((2, 6), bool);stray[0, 4] = True
        self.assertEqual(rule(bag=stray)[1], [])                                         # the bit on another slot only
        self.assertEqual(rule(hyperbeam=np.array([False, True]))[1], [])                  # not this side's Hyper Beam
        self.assertEqual(rule(residual_ran=False)[1], [])                                 # deferred: must still read 2
        wrong = counter.copy();wrong[0, 3] = 2
        self.assertEqual(rule(counter=wrong)[1], [])                                      # a second mon's counter
        gone = np.zeros((2, 6), np.int64);dep = departed.copy();dep[0, 0] = True
        rest, explained, cells = rule(counter=gone, departed=dep, post_active=np.array([1, 0]))
        self.assertEqual(explained, []);self.assertEqual(np.flatnonzero(cells[0]).tolist(), [2])   # dragged after the hit

    def test_bag_rule_explains_only_witnessed_cells(self):
        from gpu_engine.v2e_regress import bag_rule
        base = np.zeros((2, 6, 5), bool);bag = base.copy();bag[0, 2, 1] = True
        allowed = np.zeros((2, 6, 5), bool);allowed[0, 2, 1] = True
        entries = {('trackers.public_volatiles', 0)}
        self.assertEqual(bag_rule(entries, bag, base, allowed)[0], set())
        extra = bag.copy();extra[0, 2, 3] = True                                         # TWIDE: another column at that slot
        self.assertEqual(bag_rule(entries, extra, base, allowed)[0], entries)
        self.assertEqual(bag_rule(entries, bag, None, allowed)[0], entries)                # no stored base bag: never

    def test_trick_witnesses_are_per_mon(self):
        from gpu_engine.v2e_regress import item_witnesses, trick_rule
        LUM, LIECHI, LEFT = 2, 3, 1;pinch = {LIECHI: 0}
        pre_active = post_active = np.zeros(2, np.int64);p2p = np.tile(np.arange(6), (2, 1))
        pre_item = np.zeros((2, 6), np.int64);pre_item[0, 0] = LIECHI;pre_item[1, 0] = LUM
        post_item = np.zeros((2, 6), np.int64);post_item[0, 0] = 0;post_item[1, 0] = LIECHI   # the Tricker got the Lum, ate it
        zero = np.zeros((2, 6), np.int64);conf = zero.copy();base_conf = zero.copy();base_conf[0, 0] = 2
        boosts = np.zeros((2, 6, 7), np.int64);base_boosts = boosts.copy()
        w = item_witnesses(pre_active, post_active, pre_item, post_item, zero, conf, zero, zero, zero, LUM, pinch)
        self.assertEqual(np.flatnonzero(w['lum'][0]).tolist(), [0]);self.assertEqual(w['stats'], {0})
        ko = {'state.confusion_counter': conf, 'state.status': zero, 'state.boosts': boosts}
        base_ko = {'state.confusion_counter': base_conf, 'state.status': zero, 'state.boosts': base_boosts}
        entries = {('state.confusion_counter', 0), ("trackers.mon_features['confusion_turns']", 0)}
        rest, explained, confused = trick_rule(entries, w, ko, base_ko, {}, {}, p2p, pre_item, post_item, pinch)
        self.assertEqual(rest, set());self.assertEqual(np.flatnonzero(confused[0]).tolist(), [0])
        other = base_conf.copy();other[0, 3] = 1                                          # a bench mon differs too
        self.assertEqual(trick_rule(entries, w, ko, dict(base_ko, **{'state.confusion_counter': other}), {}, {}, p2p, pre_item,
                                    post_item, pinch)[0], entries)
        # Stages: the Tricker gave away its own Liechi, which the base ate (this tree -1 Atk against it): explained;
        # a second stat, or -1 Atk where neither active held an Atk berry (TWIDE), is not.
        stage = base_boosts.copy();stage[0, 0, 0] = 1
        entries = {('state.boosts', 0)}
        self.assertEqual(trick_rule(entries, w, ko, dict(base_ko, **{'state.boosts': stage}), {}, {}, p2p, pre_item, post_item,
                                    pinch)[0], set())
        two = stage.copy();two[0, 0, 1] = 1
        self.assertEqual(trick_rule(entries, w, ko, dict(base_ko, **{'state.boosts': two}), {}, {}, p2p, pre_item, post_item,
                                    pinch)[0], entries)
        no_berry = pre_item.copy();no_berry[0, 0] = LEFT
        w2 = item_witnesses(pre_active, post_active, no_berry, post_item, zero, conf, zero, zero, zero, LUM, pinch)
        self.assertEqual(trick_rule(entries, w2, ko, dict(base_ko, **{'state.boosts': stage}), {}, {}, p2p, no_berry, post_item,
                                    pinch)[0], entries)
        # A foe that switched in before the Trick is its target: its pre-step berry is in the pool (GPU pass, 4 lanes).
        switched = no_berry.copy();switched[1, 0] = LEFT;switched[1, 3] = LIECHI
        w3 = item_witnesses(pre_active, np.array([0, 3]), switched, post_item, zero, conf, zero, zero, zero, LUM, pinch)
        self.assertEqual(w3['stats'], {0});self.assertTrue(w3['acts'][1, 3])

    def test_ledger_rule_checks_the_knocked_mons(self):
        from gpu_engine.v2e_regress import LEDGER_LEAVES, ledger_rule
        p2p = np.tile(np.arange(6), (2, 1))
        pre, post = np.zeros((2, 6), bool), np.zeros((2, 6), bool);post[1, 0] = True
        item = np.zeros((2, 6), np.int64);item[1, 0] = 7
        mutated, original = np.zeros((2, 6), np.int64), np.zeros((2, 6), np.int64);mutated[1, 0] = 1;original[1, 0] = 7
        entries = {(LEDGER_LEAVES[0], 1), (LEDGER_LEAVES[1], 1)}
        self.assertEqual(ledger_rule(entries, mutated, original, pre, post, item, p2p)[0], set())
        bad = original.copy();bad[1, 0] = 3
        self.assertEqual(ledger_rule(entries, mutated, bad, pre, post, item, p2p)[0], entries)
        self.assertEqual(ledger_rule(entries, mutated, original, post, post, item, p2p)[0], entries)   # knocked before the step
        self.assertEqual(ledger_rule(entries, mutated, original, pre, post, item, p2p, trick=True)[0], entries)


if __name__ == '__main__':unittest.main()
