"""Fidelity patch A1 (status, boost and cure effects) on authored component states.

Curse, Superpower, Charm, Lovely Kiss, Sleep Clause Mod, Insomnia/Vital Spirit,
Aromatherapy, Refresh and the Substitute cleared at a faint, each against the
Showdown f76228a1 rule it ports (cited in gpu_engine/effects.py and transition.py).
The two-engine probes (gpu_engine/effect_probes/*.json) are the Showdown gate;
these are the component checks. No chance is drawn beyond the fixed block.
"""
import importlib.util
import json
import shutil
import subprocess
import unittest

import numpy as np

from _showdown_root import requires_showdown, showdown_root

MOVES = ['_none', 'curse', 'superpower', 'charm', 'lovelykiss', 'spore', 'rest', 'aromatherapy', 'refresh',
         'healbell', 'splash', 'protect', 'substitute', 'tackle', 'struggle', 'sleeptalk', 'doubleteam']
(CURSE, SUPERPOWER, CHARM, LOVELYKISS, SPORE, REST, AROMATHERAPY, REFRESH, HEALBELL, SPLASH, PROTECT, SUBSTITUTE,
 TACKLE, STRUGGLE, SLEEPTALK, DOUBLETEAM) = range(1, 17)
TYPES = ['_none', 'normal', 'fighting', 'flying', 'poison', 'ground', 'rock', 'bug', 'ghost',
         'steel', 'fire', 'water', 'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark']
NORMAL, FIGHTING, GHOST, GRASS, PSYCHIC = 1, 2, 8, 12, 14
ABILITIES = ['_none', 'insomnia', 'vitalspirit', 'clearbody', 'whitesmoke', 'hypercutter', 'soundproof', 'limber']
INSOMNIA, VITALSPIRIT, CLEARBODY, WHITESMOKE, HYPERCUTTER, SOUNDPROOF = range(1, 7)
NONE, PAR, BRN, PSN, TOX, SLP, FRZ = range(7)
B = 8
POWER = {SUPERPOWER: 120, TACKLE: 35, STRUGGLE: 50}
TYPE = {SUPERPOWER: FIGHTING, SPORE: GRASS, REST: PSYCHIC, AROMATHERAPY: GRASS}
ACCURACY = {LOVELYKISS: 75}
ALWAYS = {CURSE, REST, AROMATHERAPY, REFRESH, HEALBELL, SPLASH, PROTECT, SUBSTITUTE, STRUGGLE, SLEEPTALK, DOUBLETEAM}
PROTECTABLE = {SUPERPOWER, CHARM, LOVELYKISS, SPORE, TACKLE, STRUGGLE}
CONTACT = {SUPERPOWER, TACKLE, STRUGGLE}


def metadata():
    rows = [None] + [{'id': name, 'source_type': 'Normal', 'category': 'Physical' if i in POWER else 'Status',
                      'flags': {'protect': int(i in PROTECTABLE), 'contact': int(i in CONTACT)}}
                     for i, name in enumerate(MOVES) if i]
    return {'vocabulary': {'moves': MOVES, 'types': TYPES, 'abilities': ABILITIES, 'items': ['_none', 'leftovers'],
                           'species': ['_none', 'a', 'b', 'c', 'd', 'e', 'f']}, 'moves': rows}


def components():
    import jax.numpy as j
    from gpu_engine import effects, secondary, sleep_talk, transition
    meta = metadata();n = len(MOVES)
    num = np.ones((18, 18), np.int32);num[NORMAL, GHOST] = 0;num[FIGHTING, GHOST] = 0
    table = transition.MoveTable(
        j.asarray([POWER.get(i, 0) for i in range(n)], j.int32), j.asarray([TYPE.get(i, NORMAL) for i in range(n)], j.int32),
        j.asarray([i in POWER for i in range(n)]), j.asarray([ACCURACY.get(i, 100) for i in range(n)], j.int32),
        j.asarray([i in ALWAYS for i in range(n)]), j.asarray([3 if i == PROTECT else 0 for i in range(n)], j.int32),
        j.asarray(num), j.ones((18, 18), j.int32), j.array(STRUGGLE),
        effects=effects.from_metadata(meta), secondary=secondary.from_metadata(meta), sleep_talk=sleep_talk.from_metadata(meta))
    return meta, table


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires JAX')
class FidelityA1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import effects, transition
        from gpu_engine.state import BattleState, empty_state
        cls.jax, cls.j, cls.e, cls.t, cls.State = jax, j, effects, transition, BattleState
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
        """No thaw, full paralysis or confusion self-hit (roll 0), no crit, max damage roll, accuracy roll 0,
        sleep roll 0, Protect succeeds, side 0 wins ties, no secondary. lanes: per-lane {(side, column): value}."""
        j = self.j;draws = np.zeros((B, 2, 12), np.int32);draws[:, :, 1:3] = 99;draws[:, :, 4] = 15;draws[:, :, 5] = 15
        for lane, edits in enumerate(lanes):
            for (side, column), value in edits.items():draws[lane, side, column] = value
        z = j.zeros((B, 2), j.int32)
        return self.t.ChanceInputs(j.asarray(draws), z, j.full((B, 2), 99, j.int32), z, z, j.zeros(B, j.int32))

    def play(self, s, a0=0, a1=0, chance=None):
        actions = self.j.zeros((B, 2), self.j.int32).at[:, 0].set(a0).at[:, 1].set(a1)
        out = self.step(s, actions, self.table, self.chance() if chance is None else chance)
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

    def boosts(self, s, side):
        return np.asarray(s.boosts)[:, side, 0].tolist()

    # -- tables -----------------------------------------------------------------------------
    def test_tables(self):
        e = self.table.effects
        self.assertEqual(np.asarray(e.self_boosts[CURSE]).tolist(), [1, 1, 0, 0, -1, 0, 0])
        self.assertTrue(bool(e.curse[CURSE]));self.assertEqual(int(e.ghost_type), GHOST)
        self.assertEqual(np.asarray(e.hit_self_boosts[SUPERPOWER]).tolist(), [-1, -1, 0, 0, 0, 0, 0])
        self.assertEqual(np.asarray(e.target_boosts[CHARM]).tolist(), [-2, 0, 0, 0, 0, 0, 0])
        self.assertEqual(int(np.asarray(e.hit_self_boosts).astype(bool).sum()), 2)     # Superpower only
        self.assertEqual(int(np.asarray(e.target_boosts).astype(bool).sum()), 1)       # Charm only
        self.assertFalse(bool(e.hit_spa_drop[SUPERPOWER]))
        self.assertEqual(int(e.status[LOVELYKISS]), SLP)
        block = np.asarray(e.drop_block)
        self.assertEqual(block[CLEARBODY].tolist(), [True] * 7);self.assertEqual(block[WHITESMOKE].tolist(), [True] * 7)
        self.assertEqual(block[HYPERCUTTER].tolist(), [True] + [False] * 6);self.assertFalse(block[INSOMNIA].any())
        self.assertTrue(bool(e.aromatherapy[AROMATHERAPY]));self.assertTrue(bool(e.refresh[REFRESH]))
        immune = np.asarray(self.table.secondary.ability_immunity)
        self.assertEqual(immune[INSOMNIA].tolist(), [False] * 5 + [True, False])
        self.assertEqual(immune[VITALSPIRIT].tolist(), [False] * 5 + [True, False])
        # A vocabulary without the new names still builds (zero rows, no Ghost type).
        small = dict(vocabulary=dict(moves=['_none', 'tackle'], types=[t for t in TYPES if t != 'ghost'], abilities=['_none']),
                     moves=[None, {'flags': {}}])
        e2 = self.e.from_metadata(small)
        self.assertEqual(int(e2.ghost_type), -1);self.assertFalse(bool(np.asarray(e2.curse).any()))

    # -- A1.1 Curse ---------------------------------------------------------------------------
    def test_curse_non_ghost_self_boosts_caps_protect_and_substitute(self):
        u = self.uses(CURSE)
        s = self.build([
            u,
            self.merge(u, {'boosts': [((0, 0, 0), 6), ((0, 0, 1), 6), ((0, 0, 4), -6)]}),
            self.merge(u, {'boosts': [((0, 0, 0), 5)]}),
            self.uses(CURSE, PROTECT),
            self.merge(u, {'substitute_hp': [((0, 0), 75)]}),
            self.merge(u, {'types': [((0, 0, 0), GHOST)]}),                       # Ghost user: unported placeholder
            self.merge(u, {'boosts': [((0, 0, 4), 6)]}),
        ])
        out = self.play(s)
        self.assertEqual(self.boosts(out, 0), [[1, 1, 0, 0, -1, 0, 0], [6, 6, 0, 0, -6, 0, 0], [6, 1, 0, 0, -1, 0, 0],
            [1, 1, 0, 0, -1, 0, 0], [1, 1, 0, 0, -1, 0, 0], [0] * 7, [1, 1, 0, 0, 5, 0, 0], [0] * 7])
        self.assertEqual(self.boosts(out, 1), [[0] * 7] * B)
        self.assertEqual(np.asarray(out.pp)[:7, 0, 0, 0].tolist(), [9] * 7)
        self.assertEqual(int(out.substitute_hp[4, 0, 0]), 75)

    # -- A1.2 Superpower ----------------------------------------------------------------------
    def test_superpower_self_drop_after_a_successful_hit(self):
        u = self.uses(SUPERPOWER)
        s = self.build([
            u,
            self.merge(u, {'boosts': [((1, 0, 6), 6)]}),                           # evasion +6 with roll 99: miss
            self.uses(SUPERPOWER, PROTECT),
            self.merge(u, {'types': [((1, 0, 0), GHOST)]}),                        # Fighting into Ghost: immune
            self.merge(u, {'substitute_hp': [((1, 0), 10)]}),                      # Substitute hit (and broken)
            self.merge(u, {'hp': [((1, 0), 1)]}),                                  # KO
            self.merge(u, {'boosts': [((0, 0, 0), -6), ((0, 0, 1), -6)]}),        # silent at -6
            self.merge(u, {'boosts': [((0, 0, 0), 2)]}),
        ])
        out = self.play(s, chance=self.chance([{}, {(0, 6): 99}]))
        drop = [-1, -1, 0, 0, 0, 0, 0]
        self.assertEqual(self.boosts(out, 0), [drop, [0] * 7, [0] * 7, [0] * 7, drop, drop, [-6, -6, 0, 0, 0, 0, 0],
            [1, -1, 0, 0, 0, 0, 0]])
        hp = np.asarray(out.hp)[:, 1, 0]
        self.assertLess(hp[0], 300);self.assertEqual(hp[1:4].tolist(), [300, 300, 300])
        self.assertEqual((int(hp[4]), int(out.substitute_hp[4, 1, 0])), (300, 0))
        self.assertEqual(int(hp[5]), 0)
        self.assertEqual(np.asarray(out.pp)[:, 0, 0, 0].tolist(), [9] * B)
        self.assertEqual(self.boosts(out, 1)[1], [0, 0, 0, 0, 0, 0, 6])

    def test_superpower_drop_lowers_the_users_defense_against_a_slower_hit(self):
        # The -1 Def is in place when the slower foe's Tackle lands in the same turn.
        s = self.build([self.uses(SUPERPOWER, TACKLE), self.merge(self.uses(SPLASH, TACKLE), {'boosts': [((0, 0, 1), -1)]})])
        out = self.play(s)
        taken = 300 - np.asarray(out.hp)[:2, 0, 0]
        self.assertEqual(int(taken[0]), int(taken[1]))

    # -- A1.3 Charm ---------------------------------------------------------------------------
    def test_charm_target_attack_drop_and_blocks(self):
        u = self.uses(CHARM)
        s = self.build([
            u,
            self.merge(u, {'types': [((1, 0, 0), GHOST)]}),                        # status moves ignore type immunity
            self.merge(u, {'substitute_hp': [((1, 0), 50)]}),
            self.uses(CHARM, PROTECT),
            self.merge(u, {'ability': [((1, 0), HYPERCUTTER)]}),
            self.merge(u, {'ability': [((1, 0), CLEARBODY)]}),
            self.merge(u, {'ability': [((1, 0), WHITESMOKE)]}),
            self.merge(u, {'boosts': [((1, 0, 0), -5)]}),
        ])
        out = self.play(s)
        self.assertEqual([b[0] for b in self.boosts(out, 1)], [-2, -2, 0, 0, 0, 0, 0, -6])
        self.assertEqual(self.boosts(out, 0), [[0] * 7] * B)
        self.assertEqual(int(out.substitute_hp[2, 1, 0]), 50)
        s = self.build([
            self.merge(u, {'boosts': [((1, 0, 0), -6)]}),
            self.merge(u, {'boosts': [((1, 0, 6), 6)]}),                           # evasion +6, roll 99: miss
            self.merge(u, {'boosts': [((1, 0, 6), 6)]}),                           # evasion +6, roll 0: hit
            self.merge(u, {'ability': [((1, 0), HYPERCUTTER)], 'boosts': [((1, 0, 1), 1)]}),
        ])
        out = self.play(s, chance=self.chance([{}, {(0, 6): 99}]))
        self.assertEqual([b[0] for b in self.boosts(out, 1)][:4], [-6, 0, -2, 0])
        self.assertEqual(self.boosts(out, 1)[3][1], 1)

    # -- A1.4 Lovely Kiss and A1.6 Insomnia / Vital Spirit ------------------------------------
    def test_lovely_kiss_sleep_accuracy_and_immunity(self):
        u = self.uses(LOVELYKISS)
        s = self.build([u, u, u, u, u,
            self.merge(u, {'substitute_hp': [((1, 0), 50)]}),
            self.merge(u, {'ability': [((1, 0), INSOMNIA)]}),
            self.merge(u, {'ability': [((1, 0), VITALSPIRIT)]})])
        out = self.play(s, chance=self.chance([{(0, 7): 0, (0, 6): 74}, {(0, 7): 1}, {(0, 7): 2}, {(0, 7): 3}, {(0, 6): 75}]))
        self.assertEqual(np.asarray(out.status)[:, 1, 0].tolist(), [SLP] * 4 + [NONE] * 4)
        # 2 + roll, then the slower sleeper's own move attempt this turn spends one.
        self.assertEqual(np.asarray(out.status_counter)[:4, 1, 0].tolist(), [1, 2, 3, 4])
        self.assertFalse(bool(np.asarray(out.rest_origin).any()))
        self.assertEqual(np.asarray(out.pp)[:, 0, 0, 0].tolist(), [9] * B)

    def test_sleep_immunity_covers_every_sleep_move_and_rest(self):
        lanes = [self.merge(self.uses(move), {'ability': [((1, 0), ability)]})
                 for move in (SPORE, LOVELYKISS) for ability in (INSOMNIA, VITALSPIRIT)]
        rest = {'hp': [((0, 0), 150)]}
        lanes += [self.merge(self.uses(REST), rest, {'ability': [((0, 0), INSOMNIA)]}),
                  self.merge(self.uses(REST), rest, {'ability': [((0, 0), VITALSPIRIT)]}),
                  self.merge(self.uses(REST), rest), self.uses(SPORE)]
        out = self.play(self.build(lanes))
        self.assertEqual(np.asarray(out.status)[:4, 1, 0].tolist(), [NONE] * 4)
        self.assertEqual(np.asarray(out.status)[4:7, 0, 0].tolist(), [NONE, NONE, SLP])
        self.assertEqual(np.asarray(out.hp)[4:7, 0, 0].tolist(), [150, 150, 300])
        self.assertEqual(np.asarray(out.pp)[:, 0, 0, 0].tolist(), [9] * B)
        self.assertEqual(int(out.status[7, 1, 0]), SLP)

    # -- A1.5 Sleep Clause Mod ----------------------------------------------------------------
    def test_sleep_clause(self):
        bench = lambda **extra: self.merge({'status': [((1, 3), SLP)], 'status_counter': [((1, 3), 2)]},
                                           *({k: v} for k, v in extra.items()))
        sp = self.uses(SPORE)
        s = self.build([
            self.merge(sp, bench()),                                               # blocked
            self.merge(sp, bench(rest_origin=[((1, 3), True)])),                   # Rest sleeper: allowed
            self.merge(sp, bench(hp=[((1, 3), 0)])),                                # fainted sleeper: allowed
            sp,                                                                    # first sleep: allowed
            self.merge(self.uses(LOVELYKISS), bench()),                            # blocked
            self.merge(sp, {'status': [((0, 3), SLP)], 'status_counter': [((0, 3), 2)]}),   # own-side sleeper: allowed
            self.merge(sp, bench(ability=[((1, 0), INSOMNIA)])),                  # blocked either way
            self.merge(sp, {'status': [((1, 3), FRZ)]}),                           # a frozen benched mon: allowed
        ])
        out = self.play(s, chance=self.chance([{(0, 7): 3}] * B))
        self.assertEqual(np.asarray(out.status)[:, 1, 0].tolist(), [NONE, SLP, SLP, SLP, NONE, SLP, NONE, SLP])
        self.assertEqual(np.asarray(out.status_counter)[:, 1, 0].tolist(), [0, 4, 4, 4, 0, 4, 0, 4])
        self.assertEqual(np.asarray(out.pp)[:, 0, 0, 0].tolist(), [9] * B)            # PP is spent when blocked
        self.assertEqual(np.asarray(out.status)[:, 1, 3].tolist(), [SLP, SLP, SLP, NONE, SLP, NONE, SLP, FRZ])
        clause = self.e.sleep_clause_blocked(s, self.j.ones(B, self.j.int32))
        self.assertEqual(np.asarray(clause).tolist(), [True, False, False, False, True, False, True, False])

    # -- A1.7 Aromatherapy --------------------------------------------------------------------
    def test_aromatherapy_cures_the_living_party_including_soundproof(self):
        party = {'status': [((0, k), v) for k, v in enumerate([PAR, SLP, PSN, BRN, TOX, FRZ])],
                 'status_counter': [((0, 1), 3), ((0, 4), 2)], 'sleep_skipped': [((0, 1), 1)], 'rest_origin': [((0, 1), True)],
                 'ability': [((0, 2), SOUNDPROOF), ((0, 0), SOUNDPROOF)], 'hp': [((0, 5), 0)],
                 'confusion_counter': [((0, 0), 2)]}
        foe = {'status': [((1, k), PSN) for k in range(6)]}
        s = self.build([self.merge(self.uses(AROMATHERAPY), party, foe), self.merge(self.uses(HEALBELL), party, foe),
                        self.merge(self.uses(AROMATHERAPY), foe)])
        out = self.play(s)
        st = np.asarray(out.status)
        self.assertEqual(st[0, 0].tolist(), [NONE] * 5 + [FRZ])                       # fainted member unchanged
        self.assertEqual(st[1, 0].tolist(), [PAR, NONE, PSN, NONE, NONE, FRZ])        # Heal Bell keeps Soundproof
        self.assertEqual(st[2, 0].tolist(), [NONE] * 6)
        self.assertEqual(st[:3, 1].tolist(), [[PSN] * 6] * 3)                         # the foe's party untouched
        self.assertEqual(np.asarray(out.status_counter)[0, 0, :5].tolist(), [0] * 5)
        self.assertEqual((int(out.sleep_skipped[0, 0, 1]), bool(out.rest_origin[0, 0, 1])), (0, False))
        self.assertEqual(int(out.confusion_counter[0, 0, 0]), 1)                        # kept (one move attempt spent)
        self.assertEqual(int(out.pp[0, 0, 0, 0]), 9)
        cured = self.e.cure_party(s, self.j.zeros(B, self.j.int32), self.table.effects, eligible=self.j.ones(B, bool))
        self.assertEqual(np.asarray(cured.status)[0, 0].tolist(), [NONE] * 5 + [FRZ])

    # -- A1.8 Refresh -------------------------------------------------------------------------
    def test_refresh_cures_par_brn_psn_tox_not_sleep(self):
        own = lambda status, counter=0: {'status': [((0, 0), status)], 'status_counter': [((0, 0), counter)]}
        r = self.uses(REFRESH)
        talk = {'moves': [((0, 0, 0), SLEEPTALK), ((0, 0, 1), REFRESH), ((0, 0, 2), 0), ((0, 0, 3), 0)]}
        s = self.build([self.merge(r, own(PAR)), self.merge(r, own(BRN)), self.merge(r, own(PSN)), self.merge(r, own(TOX, 3)),
                        r, self.merge(talk, own(SLP, 3)), self.merge(r, own(PAR), {'status': [((0, 2), BRN)]})])
        out = self.play(s)
        self.assertEqual(np.asarray(out.status)[:7, 0, 0].tolist(), [NONE] * 5 + [SLP, NONE])
        self.assertEqual(np.asarray(out.status_counter)[:5, 0, 0].tolist(), [0] * 5)
        self.assertEqual(int(out.status_counter[5, 0, 0]), 2)                          # Sleep Talk's Refresh fails on slp
        self.assertEqual(int(out.status[6, 0, 2]), BRN)                                # the user only
        self.assertEqual(np.asarray(out.pp)[:5, 0, 0, 0].tolist(), [9] * 5)

    # -- device: B2 attributes an ability from a -fail/-immune [from] ability line ------------
    def test_device_reveals_follow_showdown_ability_lines(self):
        """Charm blocked by Clear Body or Hyper Cutter prints -fail [from] ability; at -6 the capped drop
        is zero and nothing prints (White Smoke row). A sleep move into Insomnia prints -immune [from]
        ability (before Sleep Clause's message: the ability handler sorts first by Speed), and Rest by a
        Vital Spirit holder -fail [from] ability. A landed Charm or an ordinary Rest reveals nothing."""
        charm = self.uses(CHARM)
        s = self.build([
            self.merge(charm, {'ability': [((1, 0), CLEARBODY)]}),
            self.merge(charm, {'ability': [((1, 0), HYPERCUTTER)]}),
            self.merge(charm, {'ability': [((1, 0), WHITESMOKE)], 'boosts': [((1, 0, 0), -6)]}),
            self.merge(charm, {'ability': [((1, 0), INSOMNIA)]}),
            self.merge(self.uses(SPORE), {'ability': [((1, 0), INSOMNIA)], 'status': [((1, 3), SLP)]}),
            self.merge(self.uses(SPLASH, REST), {'ability': [((1, 0), VITALSPIRIT)], 'hp': [((1, 0), 150)]}),
            self.merge(self.uses(SPLASH, REST), {'hp': [((1, 0), 150)]}),
        ])
        init = self.t.initialize_with_public(s, self.table)
        actions = self.j.zeros((B, 2), self.j.int32)
        out = self.jax.jit(lambda st, a, tk, c: self.t.step_with_public(st, a, self.table, tk, chance_inputs=c))(
            init.state, actions, init.trackers, self.chance())
        shown = np.asarray(out.trackers.shown_ability)[:, 1, 0]
        self.assertEqual((shown != 0).tolist(), [True, True, False, False, True, True, False, False])
        self.assertEqual(shown[[0, 1, 4, 5]].tolist(), [CLEARBODY, HYPERCUTTER, INSOMNIA, VITALSPIRIT])
        state = out.step.state
        self.assertEqual(np.asarray(state.boosts)[:4, 1, 0, 0].tolist(), [0, 0, -6, -2])
        self.assertEqual(np.asarray(state.status)[[4, 5, 6], 1, 0].tolist(), [NONE, NONE, SLP])

    # -- design 13: the own Substitute at a faint ---------------------------------------------
    def test_a_fainted_holder_loses_its_substitute(self):
        no_pp = {'pp': [((0, 0, k), 0) for k in range(4)]}
        s = self.build([
            self.merge(self.uses(TACKLE), no_pp, {'hp': [((0, 0), 1)], 'substitute_hp': [((0, 0), 75)]}),   # Struggle recoil KO
            self.merge(self.uses(TACKLE), no_pp, {'substitute_hp': [((0, 0), 75)]}),                       # survives
            self.merge({'hp': [((1, 0), 1)], 'status': [((1, 0), PSN)], 'substitute_hp': [((1, 0), 75)]}),  # poison KO
            {'status': [((1, 0), PSN)], 'substitute_hp': [((1, 0), 75)]},                                    # survives
        ])
        out = self.play(s)
        self.assertEqual((int(out.hp[0, 0, 0]), int(out.substitute_hp[0, 0, 0])), (0, 0))
        self.assertEqual(int(out.substitute_hp[1, 0, 0]), 75)
        self.assertEqual((int(out.hp[2, 1, 0]), int(out.substitute_hp[2, 1, 0])), (0, 0))
        self.assertGreater(int(out.hp[3, 1, 0]), 0);self.assertEqual(int(out.substitute_hp[3, 1, 0]), 75)
        self.assertTrue(bool(out.forced_switch[0, 0]));self.assertTrue(bool(out.forced_switch[2, 1]))


@requires_showdown()
@unittest.skipUnless(shutil.which('node'), 'requires node')
class ShowdownAbilityLineTests(unittest.TestCase):
    """The Showdown f76228a1 lines the device reveals above stand for, read by B2's attribution."""
    def test_b2_attributes_the_ability_lines_the_device_reveals(self):
        from pokezero.belief import _confirmed_ability_from_event
        from pokezero.showdown import _public_event_from_line
        code = """const {Battle}=require(process.argv[1]+'/dist/sim/battle.js');
function mon(species,ability,moves){return {name:species,species,ability,item:'',moves,nature:'Hardy',level:80,gender:'M',
 evs:{hp:84,atk:84,def:84,spa:84,spd:84,spe:84},ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31}};}
function run(p1,p2,choices){const b=new Battle({formatid:'gen3customgame@@@sleepclausemod',seed:[1,2,3,4]});
 b.setPlayer('p1',{name:'A',team:p1});b.setPlayer('p2',{name:'B',team:p2});for(const [x,y] of choices)b.makeChoices(x,y);return b.log;}
const smeargle=mon('Smeargle','Own Tempo',['charm','spore','seismictoss','splash']);
console.log(JSON.stringify({
 clearbody:run([smeargle],[mon('Metagross','Clear Body',['splash'])],[['move 1','move 1']]),
 hypercutter:run([smeargle],[mon('Kingler','Hyper Cutter',['splash'])],[['move 1','move 1']]),
 landed:run([smeargle],[mon('Hypno','Insomnia',['splash'])],[['move 1','move 1']]),
 insomnia:run([smeargle],[mon('Blissey','Serene Grace',['splash']),mon('Hypno','Insomnia',['splash'])],
  [['move 2','move 1'],['move 2','switch 2']]),
 rest:run([smeargle],[mon('Primeape','Vital Spirit',['rest','splash'])],[['move 3','move 2'],['move 4','move 1']])}));"""
        logs = json.loads(subprocess.check_output(['node', '-e', code, str(showdown_root())], text=True))
        def revealed(log):
            found = set()
            for line in log:
                ident, name = _confirmed_ability_from_event(_public_event_from_line(line))
                if ident and name and ident.startswith('p2'):found.add(name)
            return found
        self.assertEqual(revealed(logs['clearbody']), {'Clear Body'})
        self.assertEqual(revealed(logs['hypercutter']), {'Hyper Cutter'})
        self.assertEqual(revealed(logs['landed']), set())
        self.assertIn('|-unboost|p2a: Hypno|atk|2', logs['landed'])
        self.assertEqual(revealed(logs['insomnia']), {'Insomnia'})                 # before Sleep Clause's handler
        self.assertFalse(any('Sleep Clause Mod activated' in line for line in logs['insomnia']))
        self.assertEqual(revealed(logs['rest']), {'Vital Spirit'})
        self.assertFalse(any(line.startswith('|-status|p2a: Primeape') for line in logs['rest']))


class RegressPluginTests(unittest.TestCase):
    """A1's regression plug-in: the Substitute-at-faint predicate and its allowed fields."""
    def test_substitute_holder_and_allowed_fields(self):
        from gpu_engine.v2e_regress import ALLOWED, PLUGINS, Pre, lane_verdict
        vocab = dict(moves=['_none', 'tackle', 'substitute', 'sleeptalk'], abilities=['_none'], items=['_none'])
        L = 4;z = lambda *shape: np.zeros(shape, np.int32)
        s = {k: z(L, 2, 6) for k in ('hp', 'status', 'charge_move', 'encore_move', 'encore_counter', 'ability', 'item',
                                     'confusion_counter', 'substitute_hp')}
        s.update(rest_origin=np.zeros((L, 2, 6), bool), volatile=np.zeros((L, 2, 6), np.uint32), active=z(L, 2),
                 pending_residual=np.zeros(L, bool), queued_actor=np.full(L, -1), queued_actions=z(L, 2), last_action_kind=z(L, 2),
                 last_action_index=np.full((L, 2), -1), forced_switch=np.zeros((L, 2), bool), moves=np.full((L, 2, 6, 4), 1))
        s['substitute_hp'][0, 1, 0] = 40                                  # lane 0: side 1's active holds one
        s['substitute_hp'][1, 0, 3] = 40                                  # lane 1: a benched value is not the active's
        s['moves'][2, 0, 0, 0] = 2                                        # lane 2: side 0 uses Substitute
        named = {f'state.{k}': v for k, v in s.items()};named['actions'] = np.zeros((L, 2), np.int32)
        p = Pre(named, vocab, np.array([False, True, True, False]), np.zeros((L, 2), bool))
        got = PLUGINS['A1']['sub_faint'](p)
        self.assertEqual(got.tolist(), [[False, True], [False, False], [True, False], [False, False]])
        self.assertTrue(all(e in ALLOWED for e in PLUGINS['A1']))
        self.assertEqual(lane_verdict({('state.substitute_hp', 1)}, {'sub_faint': [1]}, None)[0], [])
        self.assertEqual(lane_verdict({('state.substitute_hp', 0)}, {'sub_faint': [1]}, None)[0], [('state.substitute_hp', 0)])
        self.assertEqual(lane_verdict({('state.hp', 1)}, {'sub_faint': [1]}, None)[0], [('state.hp', 1)])
        ok = {('trackers.public_port.holder', 1), ('trackers.public_port.victims', 0), ('trackers.visible_belief.visible_hard', 0),
              ('trackers.status', 0), ('state.status', 0)}
        for effect in ('aromatherapy', 'refresh'):
            self.assertEqual(lane_verdict(ok, {effect: [0]}, None)[0], [])
            self.assertEqual(lane_verdict({('state.status', 1)}, {effect: [0]}, None)[0], [('state.status', 1)])

    def test_trace_copying_insomnia_and_any_install_expose_a_sleep_target(self):
        from gpu_engine.v2e_regress import Pre, sleep_immunity
        vocab = dict(moves=['_none', 'tackle', 'hypnosis', 'batonpass'], abilities=['_none', 'insomnia', 'trace'], items=['_none'])
        L = 4;z = lambda *shape: np.zeros(shape, np.int32)
        s = {k: z(L, 2, 6) for k in ('hp', 'status', 'charge_move', 'encore_move', 'encore_counter', 'ability', 'item',
                                     'confusion_counter', 'substitute_hp')}
        s.update(hp=np.full((L, 2, 6), 50), rest_origin=np.zeros((L, 2, 6), bool), volatile=np.zeros((L, 2, 6), np.uint32),
                 active=z(L, 2), pending_residual=np.zeros(L, bool), queued_actor=np.full(L, -1), queued_actions=z(L, 2),
                 last_action_kind=z(L, 2), last_action_index=np.full((L, 2), -1), forced_switch=np.zeros((L, 2), bool),
                 moves=np.full((L, 2, 6, 4), 1))
        s['moves'][:, 0, 0, 0] = 2                                        # side 0 uses Hypnosis on every lane
        s['ability'][:, 0, 0] = 1                                         # ... and holds Insomnia (Trace's donor)
        s['ability'][0, 1, 3] = 2                                         # lane 0: a benched Trace holder comes in
        s['ability'][1, 1, 3] = 2                                         # lane 1: the same, side 1 attacks (no install)
        s['ability'][2, 1, 3] = 1;s['moves'][2, 1, 0, 0] = 3              # lane 2: a benched Insomnia holder, side 1 Baton Passes
        s['ability'][3, 1, 3] = 2;s['ability'][3, 0, 0] = 0               # lane 3: Trace with no Insomnia to copy
        actions = np.zeros((L, 2), np.int32);actions[0, 1] = 6;actions[3, 1] = 6
        named = {f'state.{k}': v for k, v in s.items()};named['actions'] = actions
        got = sleep_immunity(Pre(named, vocab, np.array([False, True, True, True]), np.zeros((L, 2), bool)))
        self.assertEqual(got[:, 0].tolist(), [True, False, True, False])

    def test_ko_cascade_turn_line_and_phase_shift(self):
        from gpu_engine.v2e_regress import lane_verdict
        effects = {'curse': [0]}
        cascade = dict(ko_sides=np.array([True, False]), mon_ok=lambda leaf, side: side == 0)
        turn_line = {('trackers.turn', None), ("trackers.mon_features['turns_active']", 1), ('state.last_move', 1),
                     ('trackers.public_port.post_upkeep', None), ('state.trap_source', 1)}
        self.assertEqual(lane_verdict(turn_line, effects, None, cascade=cascade)[0], [])
        self.assertEqual(lane_verdict(turn_line, effects, None)[0], sorted(turn_line, key=lambda x: (x[0], -1 if x[1] is None else x[1])))
        # A residual write on the non-fainted side is a consequence only on a phase-shift lane.
        residual = {('state.status_counter', 1), ('state.item', 1), ('trackers.shown_item', 1), ('state.volatile', 1)}
        self.assertEqual(len(lane_verdict(residual, effects, None, cascade=cascade)[0]), 4)
        phase = dict(cascade, ko_sides=np.array([True, True]), families=('residual',))
        self.assertEqual(lane_verdict(residual, effects, None, cascade=phase)[0], [])
        self.assertEqual(lane_verdict({('state.status', 1), ('state.turn', None), ('state.moves', 1)}, effects, None, cascade=phase)[0],
                         [('state.moves', 1), ('state.status', 1), ('state.turn', None)])


if __name__ == '__main__':
    unittest.main()
