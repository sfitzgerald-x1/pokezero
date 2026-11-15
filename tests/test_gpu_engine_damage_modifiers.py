"""Damage modifiers: Silk Scarf and Twisted Spoon (Gen 3 type items), Marvel Scale, and Gen 3's Plus/Minus.

The rules follow Showdown f76228a1 as cited in gpu_engine/items.py (attack_modifier) and gpu_engine/combat.py
(attack_modifier, defense_modifier); the two-engine probes (gpu_engine/effect_probes/{silkscarf,twistedspoon,marvelscale,
plusminus}_*.json) are the Showdown gate. The Showdown reference here is sim/battle.ts's chainModify / modify in binary64,
as JavaScript evaluates them.
"""
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import unittest

import numpy as np

TYPES = ['_none', 'normal', 'fighting', 'flying', 'poison', 'ground', 'rock', 'bug', 'ghost', 'steel', 'fire', 'water',
         'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark']
NORMAL, GHOST, ELECTRIC, PSYCHIC = 1, 8, 13, 14
ABILITIES = ['_none', 'guts', 'hugepower', 'purepower', 'marvelscale', 'plus', 'minus', 'hustle']
GUTS, HUGEPOWER, PUREPOWER, MARVELSCALE, PLUS, MINUS, HUSTLE = range(1, 8)
ITEMS = ['_none', 'choiceband', 'thickclub', 'lightball', 'souldew', 'silkscarf', 'twistedspoon', 'leftovers']
CHOICEBAND, THICKCLUB, LIGHTBALL, SOULDEW, SILKSCARF, TWISTEDSPOON, LEFTOVERS = range(1, 8)
SPECIES = ['_none', 'mew', 'marowak', 'pikachu', 'latias']
MEW, MAROWAK, PIKACHU, LATIAS = range(1, 5)
MOVES = ['_none', 'tackle', 'shadowball', 'psychic', 'thunderbolt', 'struggle', 'splash']
TACKLE, SHADOWBALL, PSYCHICMOVE, THUNDERBOLT, STRUGGLE, SPLASH = range(1, 7)
HAS_JAX = bool(importlib.util.find_spec('jax'))
NONE, PAR, BRN, PSN, TOX, SLP, FRZ = range(7)


def metadata():
    return {'vocabulary': dict(moves=MOVES, abilities=ABILITIES, items=ITEMS, species=SPECIES, types=TYPES)}


# --- Showdown f76228a1, transcribed (sim/battle.ts:2334-2356; JS numbers are binary64, Math.trunc is int()). ---

def sd_chain(factors):
    """runEvent's event.modifier after each handler's chainModify(factor), in handler order."""
    modifier = 1.0
    for factor in factors:
        previous = int(modifier * 4096)
        nxt = int(factor * 4096)
        modifier = ((previous * nxt + 2048) >> 12) / 4096
    return modifier


def sd_modify(value, modifier):
    q = int(modifier * 4096)
    return int((int(value * q) + 2048 - 1) / 4096)


def sd_handlers(ability, item, species, move_type, physical, status, foe_ability, foe_hp):
    """The ModifyAtk/ModifySpA handlers that return chainModify for this hit, sorted as runEvent sorts them
    (priority high to low; the ability's subOrder 7 before the item's 8 on a tie). (priority, subOrder, factor)."""
    out = []
    if physical:
        if ability == GUTS and status != NONE:out.append((5, 7, 1.5))        # data/abilities.ts guts
        if ability in (HUGEPOWER, PUREPOWER):out.append((5, 7, 2))           # data/abilities.ts hugepower/purepower
        if item == CHOICEBAND:out.append((1, 8, 1.5))                        # data/items.ts choiceband
        if item == THICKCLUB and species == MAROWAK:out.append((1, 8, 2))    # data/mods/gen4/items.ts thickclub
        if item == SILKSCARF and move_type == NORMAL:out.append((1, 8, 1.1))  # data/mods/gen3/items.ts silkscarf
    else:
        if item == LIGHTBALL and species == PIKACHU:out.append((1, 8, 2))    # data/mods/gen3/items.ts lightball
        if item == SOULDEW and species == LATIAS:out.append((1, 8, 1.5))     # data/mods/gen6/items.ts souldew
        if item == TWISTEDSPOON and move_type == PSYCHIC:out.append((1, 8, 1.1))  # data/mods/gen3/items.ts twistedspoon
        partner = {PLUS: MINUS, MINUS: PLUS}.get(ability)
        if partner is not None and foe_ability == partner and foe_hp > 0:     # data/mods/gen3/abilities.ts plus/minus
            out.append((0, 7, 1.5))
    out.sort(key=lambda h: (-h[0], h[1]))
    return [factor for _, _, factor in out]


def old_fraction(ability, item, species, physical, status):
    """The engine's pre-damage-modifier (num, den): items.attack_modifier's fraction times combat.attack_modifier's."""
    num, den = 1, 1
    if physical and item == CHOICEBAND:num, den = 3, 2
    if physical and item == THICKCLUB and species == MAROWAK:num, den = 2, 1
    if not physical and item == LIGHTBALL and species == PIKACHU:num, den = 2, 1
    if not physical and item == SOULDEW and species == LATIAS:num, den = 3, 2
    if physical and ability == GUTS and status != NONE:return num * 3, den * 2
    if physical and ability in (HUGEPOWER, PUREPOWER):return num * 2, den
    return num, den


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class TableTests(unittest.TestCase):
    def test_rows_and_appended_fields(self):
        from gpu_engine import combat, items
        meta = metadata()
        i = items.from_metadata(meta)
        self.assertEqual(np.asarray(i.type_boost).tolist(), [0, 0, 0, 0, 0, NORMAL, PSYCHIC, 0])
        self.assertEqual(items.ItemTable._fields[-2:], ('white_herb', 'type_boost'))
        c = combat.from_metadata(meta)
        self.assertEqual(np.flatnonzero(np.asarray(c.marvel_scale)).tolist(), [MARVELSCALE])
        self.assertEqual(np.flatnonzero(np.asarray(c.plus)).tolist(), [PLUS])
        self.assertEqual(np.flatnonzero(np.asarray(c.minus)).tolist(), [MINUS])
        # Appended only: tier D's fields keep their positions.
        self.assertEqual(combat.CombatTable._fields[13:], ('compound_eyes', 'wonder_guard', 'pinch_type', 'status_move',
                                                           'marvel_scale', 'plus', 'minus'))
        with self.assertRaisesRegex(ValueError, 'type vocabulary'):
            items.from_metadata({'vocabulary': dict(items=['_none', 'silkscarf'], species=['_none'])})
        # A vocabulary without the type items (or without types at all) has no type item.
        plain = items.from_metadata({'vocabulary': dict(items=['_none', 'leftovers'], species=['_none'])})
        self.assertEqual(np.asarray(plain.type_boost).tolist(), [0, 0])


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class ChainTests(unittest.TestCase):
    """Every (ability, item) Atk/SpA pair: the engine's Q12 chain against Showdown's handler chain, and against the
    engine's old fraction wherever that pair existed before."""
    STATS = np.arange(1, 1024)

    def engine(self, items_t, combat_t, ability, item, species, move_type, physical, status, foe_ability, foe_hp):
        import jax.numpy as j
        from gpu_engine import combat, items
        item_num, item_den = items.attack_modifier(items_t, j.asarray(item), j.asarray(species), physical,
                                                   move_type=j.asarray(move_type))
        return combat.attack_modifier(combat_t, j.asarray(ability), j.asarray(status), physical, item_num=item_num,
                                      item_den=item_den, foe_ability=j.asarray(foe_ability), foe_hp=j.asarray(foe_hp))

    def test_every_pair_matches_showdown_and_the_old_fraction(self):
        import jax.numpy as j
        from gpu_engine import combat, items, mechanics as m
        meta = metadata()
        items_t, combat_t = items.from_metadata(meta), combat.from_metadata(meta)
        combos = [c for c in itertools.product(
            range(len(ABILITIES)), range(len(ITEMS)), (MEW, MAROWAK, PIKACHU, LATIAS),
            ((NORMAL, True), (GHOST, True), (PSYCHIC, False), (ELECTRIC, False), (0, True)),
            (NONE, PAR, SLP), ((0, 100), (PLUS, 100), (MINUS, 100), (MINUS, 0)))
            if c[0] != HUSTLE]  # Hustle modifies the staged stat directly (gen3_damage), outside the chain
        col = lambda f: j.asarray([f(c) for c in combos])
        num, den = self.engine(items_t, combat_t, col(lambda c: c[0]), col(lambda c: c[1]), col(lambda c: c[2]),
                               col(lambda c: c[3][0]), col(lambda c: c[3][1]), col(lambda c: c[4]),
                               col(lambda c: c[5][0]), col(lambda c: c[5][1]))
        self.assertEqual(set(np.asarray(den).tolist()), {4096})
        stats = j.asarray(self.STATS)
        modified = {}
        old_pairs = new_pairs = 0
        for combo, q in zip(combos, np.asarray(num).tolist()):
            ability, item, species, (move_type, physical), status, (foe_ability, foe_hp) = combo
            factors = sd_handlers(ability, item, species, move_type, physical, status, foe_ability, foe_hp)
            modifier = sd_chain(factors)
            self.assertEqual(q, int(modifier * 4096), combo)
            if q not in modified:
                got = np.asarray(m.modify(stats, q, 4096)).tolist()
                # runEvent's single modify of the stat by the chained modifier (an identity chain modifies by 1).
                self.assertEqual(got, [sd_modify(v, modifier) for v in self.STATS.tolist()], q)
                modified[q] = got
            if 1.1 in factors or (not physical and ability in (PLUS, MINUS) and foe_hp > 0
                                  and foe_ability == {PLUS: MINUS, MINUS: PLUS}[ability]):
                new_pairs += 1
                continue
            old_num, old_den = old_fraction(ability, item, species, physical, status)
            key = (q, old_num, old_den)
            if key not in modified:
                self.assertEqual(modified[q], np.asarray(m.modify(stats, old_num, old_den)).tolist(), combo)
                modified[key] = True
            old_pairs += 1
        self.assertGreater(old_pairs, 5000)
        self.assertGreater(new_pairs, 500)
        # The distinct chains seen: the old exact ones and the new inexact ones.
        self.assertTrue({4096, 6144, 8192, 9216, 12288, 16384, 4505, 6758, 9010} <= set(modified))

    def test_the_inexact_chains(self):
        import jax.numpy as j
        from gpu_engine import combat, items
        meta = metadata()
        items_t, combat_t = items.from_metadata(meta), combat.from_metadata(meta)
        cases = [  # (ability, item, move type, physical, status, foe ability, chained Q12)
            (0, SILKSCARF, NORMAL, True, NONE, 0, 4505),
            (HUGEPOWER, SILKSCARF, NORMAL, True, NONE, 0, 9010),   # 2 x 1.1 = 22/10 would be 9011
            (PUREPOWER, SILKSCARF, NORMAL, True, NONE, 0, 9010),
            (GUTS, SILKSCARF, NORMAL, True, PAR, 0, 6758),         # 4505 x 3/2 as a fraction would be 6757
            (GUTS, SILKSCARF, NORMAL, True, NONE, 0, 4505),
            (0, TWISTEDSPOON, PSYCHIC, False, NONE, 0, 4505),
            (PLUS, TWISTEDSPOON, PSYCHIC, False, NONE, MINUS, 6758),
            (MINUS, 0, ELECTRIC, False, NONE, PLUS, 6144),
            (PLUS, 0, ELECTRIC, False, NONE, PLUS, 4096),          # the partner must hold the other ability
            (PLUS, 0, NORMAL, True, NONE, MINUS, 4096),            # SpA only
            (0, SILKSCARF, GHOST, True, NONE, 0, 4096),
            (0, SILKSCARF, 0, True, NONE, 0, 4096),                # Struggle / the typeless self-hit
            (0, TWISTEDSPOON, ELECTRIC, False, NONE, 0, 4096),
        ]
        for ability, item, move_type, physical, status, foe_ability, q in cases:
            num, den = self.engine(items_t, combat_t, ability, item, MEW, move_type, physical, status, foe_ability, 100)
            self.assertEqual((int(num), int(den)), (q, 4096), (ABILITIES[ability], ITEMS[item], move_type))
        self.assertEqual(int(1.1 * 4096), 4505)
        # The item alone keeps its fraction; None (the self-hit, Pursuit) is no type item.
        self.assertEqual([int(x) for x in items.attack_modifier(items_t, j.asarray(SILKSCARF), j.asarray(MEW), True,
                                                                move_type=j.asarray(NORMAL))], [4505, 4096])
        self.assertEqual([int(x) for x in items.attack_modifier(items_t, j.asarray(SILKSCARF), j.asarray(MEW), True)],
                         [1, 1])
        self.assertEqual([int(x) for x in items.attack_modifier(items_t, j.asarray(CHOICEBAND), j.asarray(MEW), True,
                                                                move_type=j.asarray(NORMAL))], [3, 2])

    def test_hustle_is_applied_before_the_chain(self):
        from gpu_engine import mechanics as m
        # Hustle's handler returns modify(atk, 1.5) at priority 5; the chain (Silk Scarf) applies once at the end.
        kwargs = dict(level=82, power=80, attack=271, defense=200, roll=100, stab=True)
        got = int(m.gen3_damage(**kwargs, hustle=True, attack_num=4505, attack_den=4096))
        atk = sd_modify(sd_modify(271, 1.5), 4505 / 4096)
        expected = int(m.gen3_damage(**dict(kwargs, attack=atk)))
        self.assertEqual(got, expected)


# Every chained handler on Gen 3's stat and BasePower events that a Gen 3 Random Battle can reach (Showdown f76228a1, resolved
# gen3 -> gen4 -> ... -> base): (holder kind, id, hook) -> (priority, factor, applies(move id, resolved type, physical)).
# The hook's prefix says whose: onX is the event target's own (the attacker on ModifyAtk/SpA/BasePower, the defender on
# ModifyDef/SpD), onSourceX the other side's. Hustle's onModifyAtk returns this.modify directly (gen3_damage), not a
# chainModify, so it is listed but chains nothing. Status, HP and species gates are "may hold" here (a superset).
PHYSICAL_TYPES = ('normal', 'fighting', 'flying', 'poison', 'ground', 'rock', 'bug', 'ghost', 'steel')
CHAIN = {
    ('ability', 'guts', 'onModifyAtk'): (5, 1.5, lambda m, t, ph: ph),          # data/abilities.ts:1725-1732
    ('ability', 'hugepower', 'onModifyAtk'): (5, 2, lambda m, t, ph: ph),       # data/abilities.ts:1841-1846
    ('ability', 'purepower', 'onModifyAtk'): (5, 2, lambda m, t, ph: ph),       # data/abilities.ts:3562-3567
    ('ability', 'hustle', 'onModifyAtk'): (5, None, lambda m, t, ph: ph),       # data/abilities.ts:1863-1868 (direct)
    ('ability', 'plus', 'onModifySpA'): (0, 1.5, lambda m, t, ph: not ph),      # data/mods/gen3/abilities.ts:125-134
    ('ability', 'minus', 'onModifySpA'): (0, 1.5, lambda m, t, ph: not ph),     # (gen4 abilities.ts:270, 325: priority 0)
    ('ability', 'marvelscale', 'onModifyDef'): (6, 1.5, lambda m, t, ph: ph),   # data/abilities.ts:2488-2498
    ('ability', 'blaze', 'onBasePower'): (2, 1.5, lambda m, t, ph: t == 'fire'),       # data/mods/gen4/abilities.ts:25-35
    ('ability', 'overgrow', 'onBasePower'): (2, 1.5, lambda m, t, ph: t == 'grass'),   # :307-317
    ('ability', 'swarm', 'onBasePower'): (2, 1.5, lambda m, t, ph: t == 'bug'),        # :457-467
    ('ability', 'torrent', 'onBasePower'): (2, 1.5, lambda m, t, ph: t == 'water'),    # :502-512
    ('ability', 'thickfat', 'onSourceBasePower'): (1, 0.5, lambda m, t, ph: t in ('fire', 'ice')),  # :491-500
    ('item', 'choiceband', 'onModifyAtk'): (1, 1.5, lambda m, t, ph: ph),       # data/items.ts:959-979
    ('item', 'thickclub', 'onModifyAtk'): (1, 2, lambda m, t, ph: ph),          # data/mods/gen4/items.ts:410-417
    ('item', 'silkscarf', 'onModifyAtk'): (1, 1.1, lambda m, t, ph: t == 'normal'),    # data/mods/gen3/items.ts:337-345
    ('item', 'lightball', 'onModifySpA'): (1, 2, lambda m, t, ph: not ph),      # data/mods/gen3/items.ts:187-195
    ('item', 'souldew', 'onModifySpA'): (1, 1.5, lambda m, t, ph: not ph),      # data/mods/gen6/items.ts:166-180
    ('item', 'souldew', 'onModifySpD'): (2, 1.5, lambda m, t, ph: not ph),
    ('item', 'twistedspoon', 'onModifySpA'): (1, 1.1, lambda m, t, ph: t == 'psychic'),  # data/mods/gen3/items.ts:406-415
    ('move', 'facade', 'onBasePower'): (0, 2, lambda m, t, ph: m == 'facade'),          # data/moves.ts:5029-5046
    ('move', 'solarbeam', 'onBasePower'): (0, 0.5, lambda m, t, ph: m == 'solarbeam'),  # data/moves.ts:17227-17260
}
# gen3 teams.ts getItem (452-513) assigns exactly these items.
POOL_ITEMS = ('choiceband', 'leftovers', 'liechiberry', 'lightball', 'lumberry', 'petayaberry', 'salacberry', 'silkscarf',
              'souldew', 'stick', 'thickclub', 'twistedspoon', 'whiteherb')
EVENTS = ('BasePower', 'ModifyAtk', 'ModifySpA', 'ModifyDef', 'ModifySpD')
HP_TYPES = ('fighting', 'flying', 'poison', 'ground', 'rock', 'bug', 'ghost', 'steel', 'fire', 'water', 'grass',
            'electric', 'psychic', 'ice', 'dragon', 'dark')


def reachable_chains():
    """{event: {factor multiset (sorted handler keys) seen with >= 1 handler}} over every reachable (move, type) and every
    holder combination, each holder slot (target's ability, target's item, source's ability, source's item, the move)
    holding at most one effect. Gen 3's category is its type's (Hidden Power: any of its 16 types)."""
    out = {e: set() for e in EVENTS}
    moves = [('hiddenpower', t) for t in HP_TYPES] + [('facade', 'normal'), ('solarbeam', 'grass')]
    moves += [('other', t) for t in HP_TYPES + ('normal',)]
    for event in EVENTS:
        own, source = f'on{event}', f'onSource{event}'
        for move, typ in moves:
            physical = typ in PHYSICAL_TYPES
            if event in ('ModifyAtk', 'ModifyDef') and not physical:continue
            if event in ('ModifySpA', 'ModifySpD') and physical:continue
            fits = lambda kind, hook: [None] + [k for k, (_, _, ok) in CHAIN.items()
                                                if k[0] == kind and k[2] == hook and ok(move, typ, physical)]
            slots = [fits('ability', own), fits('item', own), fits('ability', source), fits('item', source),
                     [None] + [k for k, (_, _, ok) in CHAIN.items() if k[0] == 'move' and k[2] == own and ok(move, typ, physical)]]
            for combo in itertools.product(*slots):
                handlers = tuple(sorted(k for k in combo if k is not None))
                if handlers:out[event].add(handlers)
    return out


class ChainOrderTests(unittest.TestCase):
    """No reachable Gen 3 stat or BasePower chain has three or more chained factors, so none is order-sensitive (with
    two, (a*b+2048)>>12 is commutative); the engine follows runEvent's priority order regardless."""

    def test_no_reachable_chain_is_order_sensitive(self):
        chains = reachable_chains()
        sizes = {}
        sensitive = []
        for event, sets in chains.items():
            for handlers in sets:
                factors = [CHAIN[k][1] for k in handlers if CHAIN[k][1] is not None]
                sizes[event] = max(sizes.get(event, 0), len(factors))
                results = {int(sd_chain(order) * 4096) for order in itertools.permutations(factors)}
                if len(results) > 1:sensitive.append((event, handlers, sorted(results)))
        self.assertEqual(sensitive, [])
        self.assertEqual(sizes, {'BasePower': 2, 'ModifyAtk': 2, 'ModifySpA': 2, 'ModifyDef': 1, 'ModifySpD': 1})
        two = {e: sorted(tuple(k[1] for k in h if CHAIN[k][1] is not None) for h in sets
                         if sum(CHAIN[k][1] is not None for k in h) == 2) for e, sets in chains.items()}
        self.assertEqual(sorted(two['BasePower']), [('blaze', 'thickfat'), ('overgrow', 'solarbeam')])
        self.assertEqual(sorted(two['ModifyAtk']), sorted((a, i) for a in ('guts', 'hugepower', 'purepower')
                                                        for i in ('choiceband', 'silkscarf', 'thickclub')))
        self.assertEqual(sorted(two['ModifySpA']), sorted((a, i) for a in ('minus', 'plus')
                                                        for i in ('lightball', 'souldew', 'twistedspoon')))
        # Within each two-factor set the priorities differ, so runEvent's order is fixed anyway (ability 5 then item 1 on
        # Atk; item 1 then Plus/Minus 0 on SpA; pinch 2 then Thick Fat 1 or the move's own 0 on BasePower).
        for event in ('BasePower', 'ModifyAtk', 'ModifySpA'):
            for handlers in chains[event]:
                chained = [k for k in handlers if CHAIN[k][1] is not None]
                if len(chained) == 2:self.assertNotEqual(CHAIN[chained[0]][0], CHAIN[chained[1]][0], handlers)

    @unittest.skipUnless((os.environ.get('NODE') or shutil.which('node')) and os.environ.get('POKEZERO_SHOWDOWN_ROOT'),
                         'requires node and the pinned Showdown checkout')
    def test_the_table_is_every_reachable_chain_handler_in_the_dex(self):
        """Every reachable ability (sets.json), item (getItem's 13), move and their conditions, plus the status, volatile
        and weather conditions, read from the built gen3 dex: exactly CHAIN's handlers, at CHAIN's priorities."""
        script = r"""
const root = process.argv[1], items = JSON.parse(process.argv[2]), d = require(root + '/dist/sim/dex.js').Dex.mod('gen3');
const sets = require(root + '/data/random-battles/gen3/sets.json'), id = s => s.toLowerCase().replace(/[^a-z0-9]/g, '');
const abilities = new Set(), moves = new Set(['struggle']);
for (const sp in sets) for (const set of sets[sp].sets) {
  for (const a of set.abilities || []) abilities.add(id(a));
  for (const m of set.movepool) moves.add(m.startsWith('hiddenpower') ? 'hiddenpower' : m);
}
const re = /^on(Source|Ally|Any|Foe)?(BasePower|ModifyAtk|ModifySpA|ModifyDef|ModifySpD)$/, out = [];
const scan = (kind, name, x) => {
  if (!x || !x.exists) throw new Error(kind + ' ' + name + ' missing');
  for (const [where, obj] of [[kind, x], [kind + '.condition', x.condition || {}]])
    for (const k of Object.keys(obj)) if (re.test(k) && typeof obj[k] === 'function') out.push([where, name, k, obj[k + 'Priority'] || 0]);
};
for (const a of abilities) scan('ability', a, d.abilities.get(a));
for (const i of items) scan('item', i, d.items.get(i));
for (const m of moves) scan('move', m, d.moves.get(m));
for (const c of ['brn', 'par', 'psn', 'tox', 'slp', 'frz', 'confusion', 'flinch', 'trapped', 'partiallytrapped',
                 'mustrecharge', 'choicelock', 'sunnyday', 'raindance', 'sandstorm', 'hail']) scan('condition', c, d.conditions.get(c));
process.stdout.write(JSON.stringify({abilities: abilities.size, moves: moves.size, hooks: out}));
"""
        node = os.environ.get('NODE') or shutil.which('node')
        got = json.loads(subprocess.run([node, '-e', script, os.environ['POKEZERO_SHOWDOWN_ROOT'], json.dumps(POOL_ITEMS)],
                                        capture_output=True, text=True, check=True, timeout=60).stdout)
        self.assertEqual((got['abilities'], got['moves']), (71, 114))
        self.assertEqual(sorted((kind, name, hook, priority) for kind, name, hook, priority in got['hooks']),
                         sorted((kind, name, hook, prio) for (kind, name, hook), (prio, _, _) in CHAIN.items()))


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class MarvelScaleTests(unittest.TestCase):
    def test_every_status_on_def_only(self):
        import jax.numpy as j
        from gpu_engine import combat, items
        meta = metadata()
        c = combat.from_metadata(meta)
        for status in range(7):
            for physical in (True, False):
                num, den = combat.defense_modifier(c, j.asarray(MARVELSCALE), j.asarray(status), physical)
                scaled = physical and status != NONE
                self.assertEqual((int(num), int(den)), (3, 2) if scaled else (1, 1), (status, physical))
            num, den = combat.defense_modifier(c, j.asarray(GUTS), j.asarray(status), True)
            self.assertEqual((int(num), int(den)), (1, 1))
        # Soul Dew's SpD fraction passes through unchanged on a special hit (the two never share a stat).
        i = items.from_metadata(meta)
        item_num, item_den = items.defense_modifier(i, j.asarray(SOULDEW), j.asarray(LATIAS), False)
        num, den = combat.defense_modifier(c, j.asarray(MARVELSCALE), j.asarray(PAR), False, item_num=item_num,
                                           item_den=item_den)
        self.assertEqual((int(num), int(den)), (3, 2))


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class StepTests(unittest.TestCase):
    """The wiring in transition.step_with_chance: side 0 attacks (or hits itself in confusion), side 1 uses Splash; one
    authored lane per case, each roll explicit, against mechanics.gen3_damage with the Showdown chain."""

    @classmethod
    def setUpClass(cls):
        import jax.numpy as j
        from gpu_engine import combat, items, mechanics as m, transition
        from gpu_engine.state import empty_state
        cls.j, cls.m, cls.t, cls.empty = j, m, transition, staticmethod(empty_state)
        meta = metadata()
        physical = [False, True, True, False, False, True, False]
        cls.power = [0, 40, 80, 90, 95, 50, 0]
        cls.mtype = [0, NORMAL, GHOST, PSYCHIC, ELECTRIC, NORMAL, NORMAL]
        cls.physical = physical
        cls.table = transition.MoveTable(j.array(cls.power), j.array(cls.mtype), j.array(physical),
            j.full(7, 100), j.ones(7, bool), j.zeros(7, j.int32), j.ones((18, 18), j.int32), j.ones((18, 18), j.int32),
            j.array(STRUGGLE), items=items.from_metadata(meta), combat=combat.from_metadata(meta))

    def run_cases(self, cases):
        """cases: (own ability, own item, own status, own confusion, move, foe ability, foe status, atk, roll, self_hit)."""
        j, m = self.j, self.m
        b = len(cases)
        s = self.empty(b, seed=5)
        a = {k: np.array(v) for k, v in s._asdict().items()}
        a['stats'][:] = 200;a['stats'][..., 0] = 1000;a['stats'][:, 0, :, 5] = 300
        a['level'][:] = 100;a['hp'][:] = 1000;a['max_hp'][:] = 1000;a['pp'][:] = 10;a['done'][:] = False
        a['moves'][:, :, :, 0] = SPLASH;a['types'][..., 0] = 17   # Dark: no STAB on any case's move
        a['species'][:] = MEW
        draws = np.zeros((b, 2, 12), np.int32)
        draws[:, :, 1] = 99;draws[:, :, 2] = 99;draws[:, :, 4] = 15
        for lane, (ability, item, status, confusion, move, foe_ability, foe_status, atk, roll, self_hit) in enumerate(cases):
            a['ability'][lane, 0, 0] = ability;a['item'][lane, 0, 0] = item;a['status'][lane, 0, 0] = status
            a['confusion_counter'][lane, 0, 0] = confusion;a['moves'][lane, 0, 0, 0] = move
            a['ability'][lane, 1, 0] = foe_ability;a['status'][lane, 1, 0] = foe_status
            a['status_counter'][lane, 1, 0] = 3 if foe_status == SLP else 0
            a['stats'][lane, 0, 0, 1] = atk;a['stats'][lane, 0, 0, 3] = atk
            draws[lane, 0, 5] = roll;draws[lane, 0, 3] = 60 if self_hit else 0
        s = type(s)(**{k: j.asarray(v) for k, v in a.items()})
        z = j.zeros((b, 2), j.int32)
        chance = self.t.ChanceInputs(j.asarray(draws), z, z, z, z, j.zeros(b, j.int32), effect=j.zeros((b, 2, 4), j.int32))
        out = self.t.step_with_chance(s, j.zeros((b, 2), j.int32), self.table, chance).state
        return np.asarray(out.hp)[:, :, 0]

    def expected(self, case, *, attack=None):
        """(own HP lost, foe's hit damage, foe's residual); attack=(num, den) replaces the Showdown chain (a wrong composition, to show the
        lane discriminates)."""
        m = self.m
        ability, item, status, confusion, move, foe_ability, foe_status, atk, roll, self_hit = case
        if self_hit:
            scaled = ability == MARVELSCALE and status != NONE
            factors = sd_handlers(ability, item, MEW, 0, True, status, 0, 0)
            damage = int(m.gen3_damage(level=100, power=40, attack=atk, defense=200, typeless=True,
                burned=status == BRN, guts=ability == GUTS, attack_num=int(sd_chain(factors) * 4096), attack_den=4096,
                defense_num=3 if scaled else 1, defense_den=2 if scaled else 1, roll=85 + roll))
            return damage, 0, 0
        typ = 0 if move == STRUGGLE else self.mtype[move]
        physical = self.physical[move]
        factors = sd_handlers(ability, item, MEW, typ, physical, status, foe_ability, 1000)
        num, den = attack or (int(sd_chain(factors) * 4096), 4096)
        scaled = foe_ability == MARVELSCALE and foe_status != NONE and physical
        damage = int(m.gen3_damage(level=100, power=self.power[move], attack=atk, defense=200, physical=physical,
            typeless=typ == 0, burned=status == BRN, guts=ability == GUTS, attack_num=num, attack_den=den,
            defense_num=3 if scaled else 1, defense_den=2 if scaled else 1, roll=85 + roll))
        residual = int(m.major_status_residual(1000 - damage, 1000, foe_status, 0).damage)
        recoil = max(1, damage // 4) if move == STRUGGLE else 0   # Struggle's own recoil [1,4] (the engine's core)
        return recoil, damage, residual

    def test_the_four_modifiers_in_the_step(self):
        cases = []
        for roll in (0, 7, 15):
            cases += [
                # Silk Scarf: a Normal hit, a Ghost control, Struggle's ??? control; chained with Huge Power and Guts.
                (0, SILKSCARF, NONE, 0, TACKLE, 0, NONE, 271, roll, False),
                (0, SILKSCARF, NONE, 0, SHADOWBALL, 0, NONE, 271, roll, False),
                (0, SILKSCARF, NONE, 0, STRUGGLE, 0, NONE, 271, roll, False),
                (HUGEPOWER, SILKSCARF, NONE, 0, TACKLE, 0, NONE, 368, roll, False),
                (GUTS, SILKSCARF, PAR, 0, TACKLE, 0, NONE, 227, roll, False),
                (0, 0, NONE, 0, TACKLE, 0, NONE, 271, roll, False),
                # Twisted Spoon: a Psychic hit and an Electric control.
                (0, TWISTEDSPOON, NONE, 0, PSYCHICMOVE, 0, NONE, 271, roll, False),
                (0, TWISTEDSPOON, NONE, 0, THUNDERBOLT, 0, NONE, 271, roll, False),
                # Plus/Minus against the opposing active; a physical hit and a non-partner are controls.
                (PLUS, 0, NONE, 0, THUNDERBOLT, MINUS, NONE, 271, roll, False),
                (MINUS, 0, NONE, 0, THUNDERBOLT, PLUS, NONE, 271, roll, False),
                (PLUS, TWISTEDSPOON, NONE, 0, PSYCHICMOVE, MINUS, NONE, 271, roll, False),
                (PLUS, 0, NONE, 0, THUNDERBOLT, PLUS, NONE, 271, roll, False),
                (PLUS, 0, NONE, 0, TACKLE, MINUS, NONE, 271, roll, False),
                # The confusion self-hit: a statused Marvel Scale holder's own Def; unstatused and a Silk Scarf control.
                (MARVELSCALE, 0, PAR, 2, TACKLE, 0, NONE, 271, roll, True),
                (MARVELSCALE, 0, NONE, 2, TACKLE, 0, NONE, 271, roll, True),
                (0, SILKSCARF, NONE, 2, TACKLE, 0, NONE, 271, roll, True),
            ]
            # Marvel Scale under every status, a physical hit and a special control.
            for status in range(7):
                cases.append((0, 0, NONE, 0, TACKLE, MARVELSCALE, status, 271, roll, False))
                cases.append((0, 0, NONE, 0, THUNDERBOLT, MARVELSCALE, status, 271, roll, False))
        hp = self.run_cases(cases)
        for lane, case in enumerate(cases):
            own, damage, residual = self.expected(case)
            self.assertEqual((1000 - int(hp[lane, 0]), 1000 - int(hp[lane, 1])), (own, damage + residual), case)
        # The chained lanes discriminate: Huge Power + Silk Scarf (Atk 368) against 2 x 11/10 = 22/10 (9011), and Guts +
        # Silk Scarf (Atk 227) against the item's 4505/4096 times 3/2 (6757), at some roll.
        for ability, wrong in ((HUGEPOWER, (22, 10)), (GUTS, (4505 * 3, 4096 * 2))):
            lanes = [c for c in cases if c[:2] == (ability, SILKSCARF)]
            self.assertTrue(any(self.expected(c) != self.expected(c, attack=wrong) for c in lanes), ABILITIES[ability])
        # Each modifier moves its lane's hit at every roll; each control does not.
        hit = lambda *case: self.expected(case)[:2]
        for roll in (0, 7, 15):
            for modified, control, moved in (
                    ((0, SILKSCARF, NONE, 0, TACKLE, 0, NONE), (0, 0, NONE, 0, TACKLE, 0, NONE), True),
                    ((0, SILKSCARF, NONE, 0, SHADOWBALL, 0, NONE), (0, 0, NONE, 0, SHADOWBALL, 0, NONE), False),
                    ((0, SILKSCARF, NONE, 0, STRUGGLE, 0, NONE), (0, 0, NONE, 0, STRUGGLE, 0, NONE), False),
                    ((0, TWISTEDSPOON, NONE, 0, PSYCHICMOVE, 0, NONE), (0, 0, NONE, 0, PSYCHICMOVE, 0, NONE), True),
                    ((0, TWISTEDSPOON, NONE, 0, THUNDERBOLT, 0, NONE), (0, 0, NONE, 0, THUNDERBOLT, 0, NONE), False),
                    ((PLUS, 0, NONE, 0, THUNDERBOLT, MINUS, NONE), (0, 0, NONE, 0, THUNDERBOLT, 0, NONE), True),
                    ((MINUS, 0, NONE, 0, THUNDERBOLT, PLUS, NONE), (0, 0, NONE, 0, THUNDERBOLT, 0, NONE), True),
                    ((PLUS, 0, NONE, 0, THUNDERBOLT, PLUS, NONE), (0, 0, NONE, 0, THUNDERBOLT, 0, NONE), False),
                    ((PLUS, 0, NONE, 0, TACKLE, MINUS, NONE), (0, 0, NONE, 0, TACKLE, 0, NONE), False),
                    ((MARVELSCALE, 0, PAR, 2, TACKLE, 0, NONE), (MARVELSCALE, 0, NONE, 2, TACKLE, 0, NONE), True),
                    ((0, SILKSCARF, NONE, 2, TACKLE, 0, NONE), (0, 0, NONE, 2, TACKLE, 0, NONE), False),
                    *(((0, 0, NONE, 0, TACKLE, MARVELSCALE, st), (0, 0, NONE, 0, TACKLE, MARVELSCALE, NONE), True)
                      for st in range(1, 7)),
                    *(((0, 0, NONE, 0, THUNDERBOLT, MARVELSCALE, st), (0, 0, NONE, 0, THUNDERBOLT, 0, NONE), False)
                      for st in range(7))):
                self_hit = modified[3] > 0
                a, b = hit(*modified, 271, roll, self_hit), hit(*control, 271, roll, self_hit)
                self.assertEqual(a != b, moved, (modified, control, roll))


if __name__ == '__main__':
    unittest.main()
