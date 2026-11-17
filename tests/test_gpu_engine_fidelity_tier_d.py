"""Fidelity tier D: Wonder Guard, the pinch abilities (Blaze, Torrent, Overgrow, Swarm) and the contact-status abilities
(Static, Poison Point, Flame Body, Effect Spore), on authored tables and lanes.

The rules follow Showdown f76228a1 as cited in gpu_engine/combat.py (wonder_guard_blocks, base_power) and
gpu_engine/secondary.py (apply_contact); the two-engine probes (gpu_engine/effect_probes/{wonderguard,pinch_,contact_,
effectspore_}*.json) are the Showdown gate. The plans' chance rows are in test_gpu_engine_chance_consumers.py.
"""
import importlib.util
import unittest

import numpy as np

TYPES = ['_none', 'normal', 'fighting', 'flying', 'poison', 'ground', 'rock', 'bug', 'ghost', 'steel', 'fire', 'water',
         'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark']
NORMAL, FIGHTING, POISON, BUG, GHOST, STEEL, FIRE, WATER, GRASS = 1, 2, 4, 7, 8, 9, 10, 11, 12
MOVES = ['_none', 'struggle', 'splash', 'tackle', 'flamethrower', 'toxic', 'seismictoss', 'surf', 'megahorn', 'earthquake']
STRUGGLE, SPLASH, TACKLE, FLAMETHROWER, TOXIC, SEISMICTOSS, SURF, MEGAHORN, EARTHQUAKE = range(1, 10)
ABILITIES = ['_none', 'wonderguard', 'blaze', 'torrent', 'overgrow', 'swarm', 'thickfat', 'poisonpoint', 'flamebody',
             'effectspore', 'static', 'limber', 'insomnia', 'immunity', 'waterveil', 'vitalspirit', 'roughskin', 'synchronize']
(WONDERGUARD, BLAZE, TORRENT, OVERGROW, SWARM, THICKFAT, POISONPOINT, FLAMEBODY, EFFECTSPORE, STATIC, LIMBER, INSOMNIA,
 IMMUNITY, WATERVEIL, VITALSPIRIT, ROUGHSKIN, SYNCHRONIZE) = range(1, 18)
HAS_JAX = bool(importlib.util.find_spec('jax'))


def metadata():
    def row(name, power, typ, category, flags=None):
        return dict(id=name, base_power=power, type=typ, source_type=TYPES[typ].title(), category=category, pp=10,
                    accuracy=100, priority=0, flags=flags or {'protect': 1})
    contact = {'contact': 1, 'protect': 1}
    rows = [None, row('struggle', 50, NORMAL, 'Physical', contact), row('splash', 0, NORMAL, 'Status', {}),
            row('tackle', 35, NORMAL, 'Physical', contact), row('flamethrower', 95, FIRE, 'Special'),
            row('toxic', 0, POISON, 'Status'), row('seismictoss', 0, FIGHTING, 'Physical', contact),
            row('surf', 95, WATER, 'Special'), row('megahorn', 120, BUG, 'Physical', contact),
            row('earthquake', 100, 5, 'Physical')]
    return {'vocabulary': dict(moves=MOVES, abilities=ABILITIES, items=['_none', 'leftovers', 'lumberry'],
                               species=['_none', 'a', 'b', 'c'], types=TYPES),
            'moves': rows, 'type_chart': np.ones((17, 17)).tolist()}


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class TableTests(unittest.TestCase):
    def test_rows(self):
        from gpu_engine import combat, mechanics as m, secondary
        meta = metadata()
        c = combat.from_metadata(meta)
        self.assertEqual(np.flatnonzero(np.asarray(c.wonder_guard)).tolist(), [WONDERGUARD])
        self.assertEqual(np.asarray(c.pinch_type).tolist(),
                         [0, 0, FIRE, WATER, GRASS, BUG] + [0] * (len(ABILITIES) - 6))
        self.assertEqual(np.flatnonzero(np.asarray(c.status_move)).tolist(), [SPLASH, TOXIC])
        # The pre-tier-D fields keep their positions (appended fields only).
        self.assertEqual(combat.CombatTable._fields[-4:], ('compound_eyes', 'wonder_guard', 'pinch_type', 'status_move'))
        s = secondary.from_metadata(meta)
        self.assertEqual(np.asarray(s.contact_status).tolist(),
                         [m.PSN if i == POISONPOINT else m.BRN if i == FLAMEBODY else 0 for i in range(len(ABILITIES))])
        self.assertEqual(np.flatnonzero(np.asarray(s.effect_spore)).tolist(), [EFFECTSPORE])
        self.assertEqual(np.flatnonzero(np.asarray(s.static)).tolist(), [STATIC])
        self.assertEqual(secondary.SecondaryTable._fields[-2:], ('contact_status', 'effect_spore'))


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class WonderGuardTests(unittest.TestCase):
    def test_blocks_exactly_the_typed_damaging_hits_that_are_not_super_effective(self):
        import jax.numpy as j
        from gpu_engine import combat
        c = combat.from_metadata(metadata())
        # (holder ability, move, resolved type, num, den, blocked)
        cases = [
            (WONDERGUARD, FLAMETHROWER, FIRE, 2, 1, False),    # super-effective
            (WONDERGUARD, FLAMETHROWER, FIRE, 4, 1, False),    # doubly super-effective
            (WONDERGUARD, SURF, WATER, 1, 1, True),            # neutral
            (WONDERGUARD, EARTHQUAKE, 5, 1, 2, True),          # resisted
            (WONDERGUARD, TACKLE, NORMAL, 0, 1, True),         # immune (WG prints its own line first)
            (WONDERGUARD, MEGAHORN, BUG, 2, 2, True),          # 2 x 0.5: runEffectiveness 1 - 1 = 0
            (WONDERGUARD, SEISMICTOSS, FIGHTING, 1, 2, True),  # fixed damage is not exempt
            (WONDERGUARD, TOXIC, POISON, 1, 1, False),         # Status move: exempt
            (WONDERGUARD, STRUGGLE, 0, 1, 1, False),           # typeless ??? (Struggle): exempt
            (0, SURF, WATER, 1, 1, False),                     # no Wonder Guard
            (STATIC, SURF, WATER, 1, 2, False),
        ]
        col = lambda i: j.asarray([x[i] for x in cases])
        out = combat.wonder_guard_blocks(c, col(0), col(1), col(2), col(3), col(4))
        self.assertEqual(np.asarray(out).tolist(), [x[5] for x in cases])


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class PinchTests(unittest.TestCase):
    def test_threshold_type_and_the_thick_fat_chain(self):
        import jax.numpy as j
        from gpu_engine import combat, mechanics as m
        c = combat.from_metadata(metadata())
        # (target ability, attacker ability, resolved type, power, hp, max hp, expected base power)
        cases = [
            (0, BLAZE, FIRE, 95, 79, 237, 142),          # hp == floor(237/3): 1.5x, 142.5 rounds down
            (0, BLAZE, FIRE, 95, 80, 237, 95),           # one above
            (0, BLAZE, WATER, 95, 1, 237, 95),           # other type
            (0, TORRENT, WATER, 95, 94, 284, 142),
            (0, OVERGROW, GRASS, 70, 83, 249, 105),
            (0, SWARM, BUG, 70, 90, 272, 105),           # Hidden Power Bug: the resolved type decides
            (0, SWARM, BUG, 120, 91, 272, 120),
            (THICKFAT, BLAZE, FIRE, 69, 85, 257, 52),    # one chain: 3072/4096 -> 52 (two modifies: 51)
            (THICKFAT, BLAZE, FIRE, 95, 85, 257, 71),
            (THICKFAT, BLAZE, FIRE, 95, 86, 257, 47),    # Thick Fat alone (47.5 rounds down)
            (THICKFAT, 0, FIRE, 69, 1, 257, 34),
            (0, 0, FIRE, 95, 1, 257, 95),
        ]
        col = lambda i: j.asarray([x[i] for x in cases])
        out = combat.base_power(c, col(0), col(2), col(3), attacker_ability=col(1), hp=col(4), max_hp=col(5))
        self.assertEqual(np.asarray(out).tolist(), [x[6] for x in cases])
        self.assertEqual(int(m.modify(m.modify(69, 3, 2), 1, 2)), 51)

    def test_without_a_pinch_the_chain_is_the_old_thick_fat_modify(self):
        import jax.numpy as j
        from gpu_engine import combat, mechanics as m
        c = combat.from_metadata(metadata())
        power = j.arange(0, 400)
        for target in (0, THICKFAT):
            for typ in (FIRE, WATER, 15):
                old = m.modify(power, 1, 2 if target == THICKFAT and typ in (FIRE, 15) else 1)
                new = combat.base_power(c, j.full_like(power, target), j.full_like(power, typ), power,
                    attacker_ability=j.full_like(power, BLAZE), hp=j.full_like(power, 100), max_hp=j.full_like(power, 200))
                legacy = combat.base_power(c, j.full_like(power, target), j.full_like(power, typ), power)
                self.assertEqual(np.asarray(new).tolist(), np.asarray(old).tolist())
                self.assertEqual(np.asarray(legacy).tolist(), np.asarray(old).tolist())


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class ContactTests(unittest.TestCase):
    """secondary.apply_contact on one authored lane per case: side 0 makes contact with side 1's active (slot 0)."""
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import secondary
        from gpu_engine.state import empty_state
        cls.j, cls.empty = j, staticmethod(empty_state)
        cls.table = secondary.from_metadata(metadata())
        cls.apply = staticmethod(jax.jit(lambda s, move, roll, sleep, el: secondary.apply_contact(
            s, 0, move, roll, cls.table, eligible=el, sleep_roll=sleep)))

    def test_rolls_statuses_immunities_and_the_sleep_clause(self):
        from gpu_engine import mechanics as m
        j = self.j
        # (holder ability, attacker edits {field: [(index, value)]}, move, roll, sleep roll, eligible,
        #  expected attacker status, counter)
        A = lambda **kw: {k: [((0, 0), v)] for k, v in kw.items()}
        cases = [
            (STATIC, {}, TACKLE, 19, 0, True, m.PAR, 0),                       # 0  1/3
            (STATIC, {}, TACKLE, 20, 0, True, 0, 0),                           # 1
            (STATIC, A(ability=LIMBER), TACKLE, 0, 0, True, 0, 0),             # 2  Limber
            (POISONPOINT, {}, TACKLE, 19, 0, True, m.PSN, 0),                  # 3
            (POISONPOINT, {}, TACKLE, 20, 0, True, 0, 0),                      # 4
            (POISONPOINT, {'types': [((0, 0, 0), POISON)]}, TACKLE, 0, 0, True, 0, 0),   # 5  Poison type
            (POISONPOINT, {'types': [((0, 0, 1), STEEL)]}, TACKLE, 0, 0, True, 0, 0),    # 6  Steel type
            (POISONPOINT, A(ability=IMMUNITY), TACKLE, 0, 0, True, 0, 0),      # 7  Immunity
            (POISONPOINT, A(status=m.BRN), TACKLE, 0, 0, True, m.BRN, 0),      # 8  already statused
            (POISONPOINT, A(hp=0), TACKLE, 0, 0, True, 0, 0),                  # 9  fainted attacker
            (POISONPOINT, {}, SURF, 0, 0, True, 0, 0),                         # 10 no contact
            (POISONPOINT, {}, TACKLE, 0, 0, False, 0, 0),                      # 11 not a damaging hit
            (FLAMEBODY, {}, TACKLE, 0, 0, True, m.BRN, 0),                     # 12
            (FLAMEBODY, {'types': [((0, 0, 0), FIRE)]}, TACKLE, 0, 0, True, 0, 0),       # 13 Fire type
            (FLAMEBODY, A(ability=WATERVEIL), TACKLE, 0, 0, True, 0, 0),       # 14 Water Veil
            (EFFECTSPORE, {}, TACKLE, 0, 3, True, m.SLP, 5),                   # 15 slp, 2+3
            (EFFECTSPORE, {}, TACKLE, 1, 0, True, m.SLP, 2),                   # 16
            (EFFECTSPORE, {}, TACKLE, 2, 0, True, m.PAR, 0),                   # 17
            (EFFECTSPORE, {}, TACKLE, 3, 0, True, m.PAR, 0),                   # 18
            (EFFECTSPORE, {}, TACKLE, 4, 0, True, m.PSN, 0),                   # 19
            (EFFECTSPORE, {}, TACKLE, 5, 0, True, m.PSN, 0),                   # 20
            (EFFECTSPORE, {}, TACKLE, 6, 0, True, 0, 0),                       # 21 9/10 nothing
            (EFFECTSPORE, A(ability=INSOMNIA), TACKLE, 0, 0, True, 0, 0),      # 22 Insomnia
            (EFFECTSPORE, A(ability=VITALSPIRIT), TACKLE, 1, 0, True, 0, 0),   # 23 Vital Spirit
            (EFFECTSPORE, A(ability=LIMBER), TACKLE, 2, 0, True, 0, 0),        # 24 Limber
            (EFFECTSPORE, {'types': [((0, 0, 0), POISON)]}, TACKLE, 4, 0, True, 0, 0),   # 25 Poison type
            # Sleep Clause Mod: another living member of the attacker's side sleeps from a foe.
            (EFFECTSPORE, {'status': [((0, 1), m.SLP)]}, TACKLE, 0, 0, True, 0, 0),      # 26
            (EFFECTSPORE, {'status': [((0, 1), m.SLP)], 'rest_origin': [((0, 1), True)]}, TACKLE, 0, 2, True, m.SLP, 4),  # 27 Rest
            (EFFECTSPORE, {'status': [((0, 1), m.SLP)], 'hp': [((0, 1), 0)]}, TACKLE, 0, 1, True, m.SLP, 3),  # 28 fainted
            (EFFECTSPORE, {'status': [((0, 1), m.SLP)]}, TACKLE, 2, 0, True, m.PAR, 0),  # 29 the clause is sleep only
            (0, {}, TACKLE, 0, 0, True, 0, 0),                                  # 30 no ability
            (ROUGHSKIN, {}, TACKLE, 0, 0, True, 0, 0),                          # 31 Rough Skin: HP only
        ]
        b = len(cases)
        s = self.empty(b, seed=3)
        arrays = {k: np.array(v) for k, v in s._asdict().items()}
        arrays['hp'][:] = 160;arrays['max_hp'][:] = 160;arrays['done'][:] = False
        arrays['types'][:, :, :, 0] = NORMAL
        for lane, (holder, edits, *_rest) in enumerate(cases):
            arrays['ability'][lane, 1, 0] = holder
            for field, writes in edits.items():
                for index, value in writes:arrays[field][(lane,) + tuple(index)] = value
        s = type(s)(**{k: j.asarray(v) for k, v in arrays.items()})
        col = lambda i: j.asarray([c[i] for c in cases])
        out = self.apply(s, col(2), col(3), col(4), col(5))
        status, counter = np.asarray(out.status)[:, 0, 0], np.asarray(out.status_counter)[:, 0, 0]
        self.assertEqual(status.tolist(), [c[6] for c in cases])
        self.assertEqual(counter.tolist(), [c[7] for c in cases])
        # A fresh sleep starts a clean sleep record; nothing else on either side moves but Rough Skin's HP.
        slept = [i for i, c in enumerate(cases) if c[6] == m.SLP and not c[1].get('status')]
        self.assertTrue(slept and not np.asarray(out.sleep_skipped)[slept, 0, 0].any())
        self.assertFalse(np.asarray(out.rest_origin)[slept, 0, 0].any())
        hp = np.asarray(out.hp)
        self.assertEqual(hp[31, 0, 0], 150);self.assertTrue((np.delete(hp[:, 0, 0], [9, 31]) == 160).all())
        self.assertTrue((hp[:, 1] == np.asarray(s.hp)[:, 1]).all())
        self.assertTrue((np.asarray(out.status)[:, 1] == np.asarray(s.status)[:, 1]).all())

    def test_reveal_masks_for_the_pursuit_intercept(self):
        """_reveal_masks: the holder's reveal on a -status [from] ability line (not Gen 3's plain slp line) or a Rough Skin
        line, the attacker's on its Synchronize reflection; the state is the plain call's."""
        from gpu_engine import mechanics as m, secondary
        j = self.j
        # (holder, attacker ability, roll): Poison Point psn, Effect Spore slp, Effect Spore par, Rough Skin, Static
        # into a Synchronize attacker (reflected onto the holder), no proc.
        cases = [(POISONPOINT, 0, 0), (EFFECTSPORE, 0, 0), (EFFECTSPORE, 0, 2), (ROUGHSKIN, 0, 59), (STATIC, SYNCHRONIZE, 0),
                 (POISONPOINT, 0, 20)]
        s = self.empty(len(cases), seed=5)
        arrays = {k: np.array(v) for k, v in s._asdict().items()}
        arrays['hp'][:] = 160;arrays['max_hp'][:] = 160;arrays['done'][:] = False;arrays['types'][:, :, :, 0] = NORMAL
        for lane, (holder, attacker, _) in enumerate(cases):
            arrays['ability'][lane, 1, 0] = holder;arrays['ability'][lane, 0, 0] = attacker
        s = type(s)(**{k: j.asarray(v) for k, v in arrays.items()})
        args = (s, 0, j.full(len(cases), TACKLE), j.asarray([c[2] for c in cases]), self.table)
        kw = dict(eligible=True, sleep_roll=j.zeros(len(cases), j.int32))
        plain = secondary.apply_contact(*args, **kw)
        out, holder, attacker = secondary.apply_contact(*args, **kw, _reveal_masks=True)
        self.assertEqual(np.asarray(holder).tolist(), [True, False, True, True, True, False])
        self.assertEqual(np.asarray(attacker).tolist(), [False, False, False, False, True, False])
        self.assertEqual(np.asarray(out.status)[4].tolist()[0][0], m.PAR);self.assertEqual(np.asarray(out.status)[4, 1, 0], m.PAR)
        for name in out._fields:
            self.assertTrue(np.array_equal(np.asarray(getattr(out, name)), np.asarray(getattr(plain, name))), name)

    def test_sleep_roll_is_required(self):
        j = self.j
        from gpu_engine import secondary
        s = self.empty(1)
        with self.assertRaises(ValueError):
            secondary.apply_contact(s, 0, j.asarray([TACKLE]), j.asarray([0]), self.table, eligible=True)


class SourceTests(unittest.TestCase):
    def test_synchronize_reflects_repeats_after_status_event_inline(self):
        """secondary.synchronize_reflects (the Pursuit intercept's reveal masks) repeats after_status_event's reflection
        statements, which stay inline there: test_gpu_engine_engine_item_callbacks certifies that function's AST against
        b8092a85. The two copies must stay equal (pure AST, no engine import)."""
        import ast
        from pathlib import Path
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'gpu_engine/secondary.py').read_text())
        fn = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        names = {'rows', '(own, target)', 'reflected', 'reflects'}
        pick = lambda f: [ast.dump(s) for s in f.body if isinstance(s, ast.Assign) and {ast.unparse(t) for t in s.targets} & names]
        helper = pick(fn['synchronize_reflects'])
        self.assertEqual(len(helper), 4)
        self.assertEqual(pick(fn['after_status_event']), helper)

if __name__ == '__main__':
    unittest.main()
