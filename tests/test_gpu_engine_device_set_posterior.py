"""The device strike likelihood (gpu_engine/device_set_posterior.py) against set_posterior's damage core.

The rolls are differential: every case is evaluated by the device kernel (the engine's gen3_damage and modifier helpers
over authored tables) and by tier2._strike_damage, the core set_posterior calls, with a small authored dex. The verdict
order, the survivor bits and narrow_pin are checked on authored strikes. The production parity (every r5a strike) is
scripts/set_posterior_device_parity.py; the compile of the real source runs when a Showdown root is configured.
"""
import importlib.util
import os
import random
import unittest
from types import SimpleNamespace

import numpy as np

HAS_JAX = bool(importlib.util.find_spec('jax'))
TYPES = ['_none', 'normal', 'fighting', 'flying', 'poison', 'ground', 'rock', 'bug', 'ghost', 'steel', 'fire', 'water',
         'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark']
T = {name: i for i, name in enumerate(TYPES)}
SPECIAL = {'fire', 'water', 'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark'}
# (name, base power, type); category by type (Gen 3), Status when power is None.
MOVE_ROWS = [('struggle', 50, 'normal'), ('tackle', 35, 'normal'), ('earthquake', 100, 'ground'), ('flamethrower', 95, 'fire'),
             ('surf', 95, 'water'), ('icebeam', 95, 'ice'), ('hiddenpower', 0, 'normal'), ('hiddenpowerfire', 70, 'fire'),
             ('hiddenpowerice', 70, 'ice'), ('hiddenpowerground', 70, 'ground'), ('flail', 0, 'normal'), ('eruption', 150, 'fire'),
             ('solarbeam', 120, 'grass'), ('facade', 70, 'normal'), ('explosion', 250, 'normal'), ('pursuit', 40, 'dark'),
             ('seismictoss', 0, 'fighting'), ('return', 0, 'normal'), ('doublekick', 30, 'fighting'), ('toxic', None, 'poison'),
             ('megahorn', 120, 'bug'), ('thunderbolt', 95, 'electric'), ('bodyslam', 85, 'normal')]
MOVES = ['_none'] + [row[0] for row in MOVE_ROWS]
M = {name: i for i, name in enumerate(MOVES)}
ABILITIES = ['_none', 'guts', 'hugepower', 'hustle', 'thickfat', 'blaze', 'swarm', 'overgrow', 'torrent', 'cloudnine',
             'marvelscale', 'plus', 'minus', 'flashfire', 'levitate', 'wonderguard', 'static']
A = {name: i for i, name in enumerate(ABILITIES)}
ITEMS = ['_none', 'leftovers', 'choiceband', 'silkscarf', 'thickclub', 'souldew', 'lightball', 'charcoal']
I = {name: i for i, name in enumerate(ITEMS)}
SPECIES = ['_none', 'alpha', 'marowak', 'latias', 'pikachu']
WEATHERS = {0: '', 1: 'raindance', 2: 'sunnyday', 3: 'sandstorm'}
STATUS = ['', 'par', 'brn', 'psn', 'tox', 'slp', 'frz']
# Effectiveness (attacking, defending) -> multiplier; everything else neutral.
CHART = {('fire', 'grass'): 2, ('fire', 'water'): .5, ('fire', 'fire'): .5, ('water', 'fire'): 2, ('water', 'water'): .5,
         ('ground', 'flying'): 0, ('ground', 'fire'): 2, ('ice', 'grass'): 2, ('ice', 'fire'): .5, ('normal', 'ghost'): 0,
         ('normal', 'rock'): .5, ('grass', 'water'): 2, ('bug', 'grass'): 2, ('electric', 'water'): 2, ('electric', 'ground'): 0,
         ('fighting', 'normal'): 2, ('dark', 'ghost'): 2}


def metadata():
    rows = [None]
    for name, power, typ in MOVE_ROWS:
        category = 'Status' if power is None else 'Special' if typ in SPECIAL else 'Physical'
        rows.append(dict(id=name, base_power=power or 0, type=T[typ], source_type=typ.title(), category=category, accuracy=100,
                         priority=0, pp=10, flags={'protect': 1}))
    chart = [[CHART.get((a, d), 1) for d in TYPES[1:]] for a in TYPES[1:]]
    return {'vocabulary': dict(moves=MOVES, abilities=ABILITIES, items=ITEMS, species=SPECIES, types=TYPES), 'moves': rows,
            'type_chart': chart}


class FakeDex:
    """The dex surface tier2._strike_damage reads, over the same authored rows."""

    def __init__(self, species_types):
        self.species_types = species_types

    def move_info(self, move):
        for name, power, typ in MOVE_ROWS:
            if name == move:
                category = 'Status' if power is None else 'Special' if typ in SPECIAL else 'Physical'
                return SimpleNamespace(id=name, type=typ.title(), base_power=power or 0, gen3_category=category)
        return None

    def species_info(self, key):
        types = self.species_types.get(key)
        return None if types is None else SimpleNamespace(types=tuple(t.title() for t in types))

    def effectiveness(self, move_type, defender_types):
        value = 1.0
        for d in defender_types:
            value *= CHART.get((move_type.lower(), d.lower()), 1)
        return value


def variant_arrays(cases):
    """One source species per case (row s = case s + 1) with that case's single variant at index 0."""
    n = len(cases) + 1
    arrays = dict(count=np.zeros(n, np.int32), species_types=np.zeros((n, 2), np.int32), engine_species=np.zeros(n, np.int32),
                  valid=np.zeros((n, 64), bool), participant=np.zeros((n, 64), bool), level=np.zeros((n, 64), np.int32),
                  stats=np.zeros((n, 64, 6), np.int32), ability=np.zeros((n, 64), np.int32), item=np.zeros((n, 64), np.int32),
                  moves=np.full((n, 64, 4), -1, np.int32), hp_type=np.zeros((n, 64), np.int32), hp_power=np.zeros((n, 64), np.int32))
    for s, c in enumerate(cases, 1):
        arrays['count'][s] = 1;arrays['valid'][s, 0] = arrays['participant'][s, 0] = True
        arrays['species_types'][s] = [T[t] for t in c['species_types']] + [0] * (2 - len(c['species_types']))
        arrays['engine_species'][s] = SPECIES.index(c['species'])
        arrays['level'][s, 0] = c['level'];arrays['stats'][s, 0] = c['stats']
        arrays['ability'][s, 0] = A[c['ability']];arrays['item'][s, 0] = I[c['item']]
        hp_key = M['hiddenpower']
        arrays['moves'][s, 0, :len(c['moves'])] = [hp_key if x.startswith('hiddenpower') else M[x] for x in c['moves']]
        hidden = next((x for x in c['moves'] if x.startswith('hiddenpower') and len(x) > len('hiddenpower')), None)
        if hidden:
            arrays['hp_type'][s, 0] = T[hidden[len('hiddenpower'):]];arrays['hp_power'][s, 0] = 70
    return arrays


def tables_for(cases):
    from gpu_engine import device_set_posterior as dsp
    flail = SimpleNamespace(id='flail', base_power=0)
    return dsp.assemble(variant_arrays(cases), metadata(), flail=flail, source_hash='authored')


def inputs_for(cases):
    import jax.numpy as j
    from gpu_engine import device_set_posterior as dsp
    col = lambda name, dtype=np.int32:j.asarray(np.asarray([c[name] for c in cases], dtype))
    return dsp.StrikeInputs(species=j.arange(1, len(cases) + 1, dtype=j.int32), move=col('move'), crit=col('crit', bool),
        weather=col('weather'), pursuit=col('pursuit', bool), attacker_status=col('attacker_status'),
        attacker_boosts=col('attacker_boosts'), attacker_percent=col('percent'), attacker_flash_fire=col('flash_fire', bool),
        defender_stats=col('defender_stats'), defender_boosts=col('defender_boosts'), defender_types=col('defender_types'),
        defender_ability=col('defender_ability'), defender_item=col('defender_item'), defender_species=col('defender_species'),
        defender_status=col('defender_status'), hp_before=col('hp_before'), observed=col('observed'), ko=col('ko', bool),
        effectiveness=col('effectiveness'), precondition=col('precondition'))


def tier2_rolls(c):
    """tier2._strike_damage over the same case: (rolls, ambiguous), or None when the core gives no rolls."""
    from pokezero import tier2
    dex = FakeDex({'alpha': c['species_types'], 'marowak': c['species_types'], 'latias': c['species_types'],
                   'pikachu': c['species_types']})
    boosts = lambda b:dict(zip(('atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'), b))
    names = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
    attacker = tier2.StrikeParticipant(species_key=c['species'], level=c['level'], stats=dict(zip(names, c['stats'])),
        ability=c['ability'] if c['ability'] != '_none' else '', item=c['item'] if c['item'] != '_none' else '', moves=tuple(c['moves']))
    defender = tier2.StrikeParticipant(species_key=SPECIES[c['defender_species']], level=80, stats=dict(zip(names, c['defender_stats'])),
        ability=ABILITIES[c['defender_ability']] if c['defender_ability'] else '', item=ITEMS[c['defender_item']] if c['defender_item'] else '')
    context = tier2.StrikeContext(attacker_species=c['species'], defender_species=SPECIES[c['defender_species']],
        attacker_status=STATUS[c['attacker_status']] or None, defender_status=STATUS[c['defender_status']] or None,
        attacker_boosts=boosts(c['attacker_boosts']), defender_boosts=boosts(c['defender_boosts']),
        attacker_hp_fraction=c['percent'] / 100, attacker_flash_fire=c['flash_fire'])
    token = SimpleNamespace(crit=c['crit'], weather=WEATHERS[c['weather']], pursuit_intercept=c['pursuit'])
    result = tier2._strike_damage(attacker=attacker, defender=defender,
        defender_types=tuple(TYPES[t].title() for t in c['defender_types'] if t), observed_move=c['observed_move'], token=token,
        context=context, dex=dex, require_move_in_set=True, pinch_ambiguity_band=tier2.Tier2Config().pinch_ambiguity_band,
        split_choice_band_baseline=False)
    if result is None or len(result.rolls) != 16:return None
    return list(result.rolls), result.ambiguous


def base_case(**over):
    c = dict(species='alpha', species_types=('normal',), level=80, stats=[250, 200, 150, 180, 160, 170], ability='_none',
             item='_none', moves=['tackle', 'flamethrower', 'hiddenpowerfire', 'earthquake'], move=M['tackle'], observed_move='tackle',
             crit=False, weather=0, pursuit=False, attacker_status=0, attacker_boosts=[0] * 7, percent=100, flash_fire=False,
             defender_stats=[300, 180, 170, 160, 150, 140], defender_boosts=[0] * 7, defender_types=[T['grass'], 0],
             defender_ability=0, defender_item=0, defender_species=1, defender_status=0, hp_before=300, observed=20, ko=False,
             effectiveness=0, precondition=0)
    c.update(over)
    return c


def random_cases(n, seed):
    rng = random.Random(seed)
    moves = [('tackle', 'tackle'), ('earthquake', 'earthquake'), ('flamethrower', 'flamethrower'), ('surf', 'surf'),
             ('icebeam', 'icebeam'), ('hiddenpowerfire', 'hiddenpower'), ('hiddenpowerice', 'hiddenpower'),
             ('hiddenpowerground', 'hiddenpower'), ('flail', 'flail'), ('solarbeam', 'solarbeam'), ('facade', 'facade'),
             ('explosion', 'explosion'), ('pursuit', 'pursuit'), ('return', 'return'), ('megahorn', 'megahorn'),
             ('thunderbolt', 'thunderbolt'), ('bodyslam', 'bodyslam'), ('eruption', 'eruption')]
    cases = []
    for _ in range(n):
        held, observed_move = rng.choice(moves)
        species = rng.choice(['alpha', 'alpha', 'marowak', 'latias', 'pikachu'])
        ability = rng.choice(['_none', 'guts', 'hugepower', 'hustle', 'blaze', 'swarm', 'overgrow', 'torrent', 'cloudnine',
                              'plus', 'minus', 'static'])
        status = rng.choice([0, 0, 1, 2, 3, 5])
        if ability == 'guts' and status and (observed_move == 'hiddenpower' or held in ('flamethrower', 'surf', 'icebeam',
                                             'solarbeam', 'pursuit', 'thunderbolt', 'eruption')):
            status = 0   # tier2 also multiplies SpA for Guts (Showdown: Atk only); a known, counted departure
        defender_types = rng.choice([[T['grass'], 0], [T['fire'], 0], [T['water'], T['ground']], [T['normal'], 0],
                                     [T['flying'], T['water']], [T['rock'], T['bug']], [T['fire'], T['grass']]])
        c = base_case(species=species, species_types=rng.choice([('normal',), ('fire',), ('water', 'ground'), ('grass',),
            ('bug', 'fighting'), ('dark',)]), level=rng.randint(50, 100),
            stats=[rng.randint(150, 400)] + [rng.randint(40, 450) for _ in range(5)], ability=ability,
            item=rng.choice(['_none', 'leftovers', 'choiceband', 'silkscarf', 'thickclub', 'souldew', 'lightball', 'charcoal']),
            moves=[held, 'tackle', 'toxic', 'surf'], move=M[observed_move], observed_move=observed_move,
            crit=rng.random() < .2, weather=rng.choice([0, 0, 1, 2, 3]), pursuit=observed_move == 'pursuit' and rng.random() < .5,
            attacker_status=status, attacker_boosts=[rng.randint(-6, 6) for _ in range(7)], percent=rng.randint(1, 100),
            flash_fire=rng.random() < .2, defender_stats=[rng.randint(150, 500)] + [rng.randint(40, 450) for _ in range(5)],
            defender_boosts=[rng.randint(-6, 6) for _ in range(7)], defender_types=defender_types,
            defender_ability=A[rng.choice(['_none', 'thickfat', 'marvelscale', 'minus', 'plus', 'cloudnine', 'static'])],
            defender_item=I[rng.choice(['_none', 'leftovers', 'souldew'])], defender_species=SPECIES.index(rng.choice(['alpha', 'latias'])),
            defender_status=rng.choice([0, 0, 1, 2]), hp_before=rng.randint(1, 500), observed=rng.randint(1, 200),
            ko=rng.random() < .3)
        cases.append(c)
    return cases


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class RollParityTests(unittest.TestCase):
    def test_random_strikes_match_tier2_core_roll_for_roll(self):
        import jax
        from gpu_engine import device_set_posterior as dsp
        cases = random_cases(600, 20261115)
        result = jax.device_get(jax.jit(dsp.strike)(tables_for(cases), inputs_for(cases)))
        compared = 0
        for n, c in enumerate(cases):
            want = tier2_rolls(c)
            rolls = np.asarray(result.rolls)[n, 0].tolist()
            if want is None:
                # tier2 has no rolls: an immune type (the device's count-0 rule) or no damage channel.
                self.assertTrue(bool(result.immune[n, 0]) or bool(result.unsupported[n, 0]), (n, c))
                self.assertEqual(int(result.counts[n, 0]), 0)
                continue
            self.assertFalse(bool(result.unsupported[n, 0]), (n, c))
            self.assertEqual(rolls, want[0], (n, c))
            self.assertEqual(bool(result.ambiguous[n, 0]), want[1], (n, c))
            fit = sum(r >= c['hp_before'] for r in want[0]) if c['ko'] else sum(r == c['observed'] for r in want[0])
            self.assertEqual(int(result.counts[n, 0]), fit)
            compared += 1
        self.assertGreater(compared, 400)

    def test_named_modifiers(self):
        import jax
        from gpu_engine import device_set_posterior as dsp
        cases = [
            base_case(item='choiceband'), base_case(ability='hugepower'), base_case(ability='guts', attacker_status=1),
            base_case(ability='guts', attacker_status=2), base_case(attacker_status=2),           # burn halves, Guts exempts
            base_case(ability='hustle'), base_case(item='silkscarf'), base_case(defender_ability=A['marvelscale'], defender_status=1),
            base_case(defender_ability=A['thickfat'], move=M['flamethrower'], observed_move='flamethrower'),
            base_case(ability='blaze', percent=20, move=M['flamethrower'], observed_move='flamethrower'),
            base_case(ability='blaze', percent=33, move=M['flamethrower'], observed_move='flamethrower'),   # band: ambiguous
            base_case(ability='blaze', percent=20, move=M['flamethrower'], observed_move='flamethrower', defender_ability=A['thickfat']),
            base_case(weather=2, move=M['flamethrower'], observed_move='flamethrower'),
            base_case(weather=2, ability='cloudnine', move=M['flamethrower'], observed_move='flamethrower'),
            base_case(weather=2, defender_ability=A['cloudnine'], move=M['flamethrower'], observed_move='flamethrower'),
            base_case(weather=1, moves=['solarbeam'], move=M['solarbeam'], observed_move='solarbeam'),
            base_case(flash_fire=True, move=M['flamethrower'], observed_move='flamethrower'),
            base_case(moves=['explosion'], move=M['explosion'], observed_move='explosion'),
            base_case(moves=['pursuit'], move=M['pursuit'], observed_move='pursuit', pursuit=True),
            base_case(moves=['facade'], move=M['facade'], observed_move='facade', attacker_status=1),
            base_case(moves=['return'], move=M['return'], observed_move='return'),
            base_case(moves=['flail'], move=M['flail'], observed_move='flail', percent=4),
            base_case(moves=['flail'], move=M['flail'], observed_move='flail', percent=10),        # straddles 48ths: ambiguous
            base_case(species='marowak', item='thickclub'), base_case(species='pikachu', item='lightball', move=M['flamethrower'],
                observed_move='flamethrower'),
            base_case(species='latias', item='souldew', move=M['flamethrower'], observed_move='flamethrower'),
            base_case(ability='plus', defender_ability=A['minus'], move=M['flamethrower'], observed_move='flamethrower'),
            base_case(crit=True, attacker_boosts=[-2, 0, 0, 0, 0, 0, 0], defender_boosts=[0, 3, 0, 0, 0, 0, 0]),
            base_case(move=M['hiddenpower'], observed_move='hiddenpower'),
            base_case(move=M['hiddenpower'], observed_move='hiddenpower', moves=['hiddenpowerground'], defender_types=[T['flying'], 0]),
        ]
        result = jax.device_get(jax.jit(dsp.strike)(tables_for(cases), inputs_for(cases)))
        for n, c in enumerate(cases):
            want = tier2_rolls(c)
            if want is None:
                self.assertTrue(bool(result.immune[n, 0]), n);continue
            self.assertEqual(np.asarray(result.rolls)[n, 0].tolist(), want[0], n)
            self.assertEqual(bool(result.ambiguous[n, 0]), want[1], n)
        # The engine-unmodeled flags.
        self.assertTrue(bool(result.typeboost[6, 0]));self.assertTrue(bool(result.marvel[7, 0]))
        self.assertTrue(bool(result.plusminus[26, 0]));self.assertFalse(bool(result.typeboost[0, 0]))
        # An immune Hidden Power variant counts 0 (set_posterior v2), it is not unsupported.
        self.assertTrue(bool(result.immune[-1, 0]));self.assertFalse(bool(result.unsupported[-1, 0]))
        self.assertEqual(int(result.counts[-1, 0]), 0)

    def test_ko_is_censored_and_a_crit_is_a_known_condition(self):
        import jax
        from gpu_engine import device_set_posterior as dsp
        plain = base_case(observed=1, hp_before=10_000)
        want = tier2_rolls(plain)[0]
        cases = [base_case(observed=want[5], hp_before=500), base_case(ko=True, hp_before=want[10], observed=want[10]),
                 base_case(crit=True, observed=want[5], hp_before=500)]
        result = jax.device_get(jax.jit(dsp.strike)(tables_for(cases), inputs_for(cases)))
        self.assertEqual(int(result.counts[0, 0]), want.count(want[5]))
        self.assertEqual(int(result.counts[1, 0]), sum(r >= want[10] for r in want))
        crit = tier2_rolls(cases[2])[0]
        self.assertEqual(int(result.counts[2, 0]), crit.count(want[5]))


def two_variant_tables(**over):
    """One species, variants 0 (no item) and 1 (Choice Band), plus an HP-Ice variant 2 and a Blaze variant 3."""
    from gpu_engine import device_set_posterior as dsp
    c = base_case(**over)
    arrays = variant_arrays([c])
    for v, (item, ability, moves) in enumerate([('_none', '_none', c['moves']), ('choiceband', '_none', c['moves']),
                                                ('_none', '_none', ['tackle', 'hiddenpowerice', 'surf', 'toxic']),
                                                ('_none', 'blaze', c['moves'])]):
        arrays['valid'][1, v] = arrays['participant'][1, v] = True
        arrays['level'][1, v] = c['level'];arrays['stats'][1, v] = c['stats']
        arrays['item'][1, v] = I[item];arrays['ability'][1, v] = A[ability]
        arrays['moves'][1, v] = -1
        arrays['moves'][1, v, :len(moves)] = [M['hiddenpower'] if x.startswith('hiddenpower') else M[x] for x in moves]
        hidden = next((x for x in moves if x.startswith('hiddenpower') and len(x) > 11), None)
        arrays['hp_type'][1, v] = T[hidden[11:]] if hidden else 0;arrays['hp_power'][1, v] = 70 if hidden else 0
    arrays['count'][1] = 4
    return dsp.assemble(arrays, metadata(), flail=SimpleNamespace(id='flail', base_power=0), source_hash='authored'), c


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class VerdictTests(unittest.TestCase):
    def run_one(self, hard, *, guard=False, **over):
        import jax
        import jax.numpy as j
        from gpu_engine import device_set_posterior as dsp
        tables, c = two_variant_tables(**over)
        result = jax.device_get(dsp.strike(tables, inputs_for([c])))
        bits = np.zeros((1, 64), bool);bits[0, list(hard)] = True
        reduced = dsp.reduce(result, dsp.pack(j.asarray(bits)), j.asarray([c['precondition']], j.int32), guard=guard)
        return result, dsp.REASONS[int(reduced.reason[0])], np.flatnonzero(np.asarray(dsp.unpack(reduced.fit))[0]).tolist()

    def test_choice_band_hit_excludes_the_unbanded_variant(self):
        rolls = tier2_rolls(base_case(item='choiceband'))[0]
        plain = tier2_rolls(base_case())[0]
        observed = next(r for r in rolls if r not in plain)
        result, reason, fit = self.run_one((0, 1), observed=observed)
        self.assertEqual(reason, '');self.assertEqual(fit, [1])

    def test_refusal_order(self):
        from gpu_engine import device_set_posterior as dsp
        # A caller fact beats a move class: called before struggle.
        _, reason, fit = self.run_one((0, 1), move=M['struggle'], moves=['struggle'], precondition=dsp.bit('called'))
        self.assertEqual((reason, fit), ('called', []))
        _, reason, _ = self.run_one((0, 1), move=M['struggle'], moves=['struggle'])
        self.assertEqual(reason, 'struggle')
        _, reason, _ = self.run_one((0, 1), move=M['seismictoss'], moves=['seismictoss'], precondition=dsp.bit('hit-substitute'))
        self.assertEqual(reason, 'fixed-damage')
        _, reason, _ = self.run_one((0, 1), move=M['doublekick'], moves=['doublekick'])
        self.assertEqual(reason, 'multi-hit')
        _, reason, _ = self.run_one((0,))
        self.assertEqual(reason, 'single-variant')
        _, reason, _ = self.run_one(())
        self.assertEqual(reason, 'no-support')
        # Eruption below full HP is conditional power.
        _, reason, _ = self.run_one((0, 1), move=M['eruption'], moves=['eruption'], observed_move='eruption', percent=99)
        self.assertEqual(reason, 'conditional-power')
        # The first hard variant in source order names the core refusal: variant 2 lacks the move (unsupported) and
        # variant 3's Blaze sits in the band; variant 2 comes first.
        _, reason, _ = self.run_one((0, 2, 3), move=M['flamethrower'], observed_move='flamethrower', percent=33)
        self.assertEqual(reason, 'core-unsupported')
        _, reason, _ = self.run_one((0, 3), move=M['flamethrower'], observed_move='flamethrower', percent=33)
        self.assertEqual(reason, 'ambiguous-band')

    def test_effectiveness_disagreement_and_immune_hidden_power(self):
        # Hidden Power: variant 0 is Fire (super-effective on Grass), variant 2 Ice (also super-effective on Grass).
        fire = tier2_rolls(base_case(move=M['hiddenpower'], observed_move='hiddenpower'))[0]
        _, reason, fit = self.run_one((0, 2), move=M['hiddenpower'], observed_move='hiddenpower', observed=fire[0], effectiveness=1)
        self.assertEqual(reason, '');self.assertIn(0, fit)
        # The same hit printed without a super-effective line disagrees with a variant whose rolls fit.
        _, reason, _ = self.run_one((0, 2), move=M['hiddenpower'], observed_move='hiddenpower', observed=fire[0], effectiveness=0)
        self.assertEqual(reason, 'effectiveness-disagreement')
        # Against a Fire defender Hidden Power Ground... is super-effective; a Flying defender makes it immune: count 0,
        # the strike stays admitted on the other variants (set_posterior v2).
        result, reason, fit = self.run_one((0, 1), move=M['hiddenpower'], observed_move='hiddenpower',
            moves=['hiddenpowerground', 'tackle'], defender_types=[T['flying'], 0], observed=5)
        self.assertTrue(bool(result.immune[0, 0]));self.assertEqual(reason, '');self.assertEqual(fit, [])

    def test_guard_refuses_engine_unmodeled_modifiers_in_tree(self):
        _, reason, _ = self.run_one((0, 1), defender_ability=A['marvelscale'], defender_status=1)
        self.assertEqual(reason, '')
        _, reason, fit = self.run_one((0, 1), guard=True, defender_ability=A['marvelscale'], defender_status=1)
        self.assertEqual((reason, fit), ('engine-unmodeled:marvelscale', []))
        _, reason, _ = self.run_one((0, 1), guard=True)
        self.assertEqual(reason, '')


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class PinTests(unittest.TestCase):
    def test_narrow_pin_semantics(self):
        import jax.numpy as j
        from gpu_engine import device_set_posterior as dsp, public_belief
        bits = lambda *xs:dsp.pack(j.asarray(np.isin(np.arange(64), xs)[None]))
        pin, active, conflicts = bits(), j.zeros(1, bool), j.zeros(1, j.int32)
        pin, active, conflicts, changed = dsp.narrow(pin, active, conflicts, bits(1, 2, 40), j.asarray([True]))
        self.assertEqual(np.flatnonzero(np.asarray(dsp.unpack(pin))[0]).tolist(), [1, 2, 40]);self.assertTrue(bool(active[0]))
        pin, active, conflicts, _ = dsp.narrow(pin, active, conflicts, bits(2, 40, 41), j.asarray([True]))
        self.assertEqual(np.flatnonzero(np.asarray(dsp.unpack(pin))[0]).tolist(), [2, 40])
        pin2, _, conflicts2, changed = dsp.narrow(pin, active, conflicts, bits(), j.asarray([True]))   # empty: declined
        self.assertFalse(bool(changed[0]));self.assertEqual(int(conflicts2[0]), 0)
        pin3, _, conflicts3, _ = dsp.narrow(pin, active, conflicts, bits(5), j.asarray([True]))       # disjoint: conflict
        self.assertTrue(np.array_equal(np.asarray(pin3), np.asarray(pin)));self.assertEqual(int(conflicts3[0]), 1)
        # The bit layout is the belief mask's (public_belief._pack).
        self.assertEqual(np.asarray(bits(1, 2, 40))[0].tolist(), public_belief._pack(np.isin(np.arange(64), (1, 2, 40))).tolist())

    def test_percent_rules(self):
        from gpu_engine import device_set_posterior as dsp
        t = dsp._percent_tables(SimpleNamespace(id='flail', base_power=0))
        # tier2's float rule |3x/100 - 1| <= 0.04: 3 * 0.32 = 0.9600000000000001 leaves 32% just outside the band. Outside
        # it the shown percent decides 3 * hp <= maxhp exactly (32% <=> hp/maxhp <= 0.32; 35% <=> > 0.34).
        ambiguous = np.flatnonzero(t['pinch_ambiguous']).tolist()
        self.assertEqual(ambiguous, [33, 34])
        self.assertTrue(t['pinch_on'][:35].all());self.assertFalse(t['pinch_on'][35:].any())
        self.assertEqual(int(t['flail_power'][100]), 20);self.assertEqual(int(t['flail_power'][1]), 200)


class ConstantTests(unittest.TestCase):
    def test_move_classes_are_set_posteriors(self):
        if not importlib.util.find_spec('jax'):self.skipTest('requires JAX (module import)')
        try:
            from pokezero import set_posterior as sp
        except ImportError:
            self.skipTest('set_posterior not merged into this tree')
        from gpu_engine import device_set_posterior as dsp
        for name in ('FIXED_DAMAGE_MOVES', 'OHKO_MOVES', 'CONDITIONAL_POWER_MOVES', 'MULTI_HIT_MOVES'):
            self.assertEqual(getattr(dsp, name), getattr(sp, name), name)


@unittest.skipUnless(HAS_JAX and os.environ.get('POKEZERO_SHOWDOWN_ROOT'), 'requires JAX and a Showdown root')
class SourceTableTests(unittest.TestCase):
    def test_participants_are_set_posteriors(self):
        from pokezero.dex import load_showdown_dex_cached, normalize_id
        from pokezero.randbat import load_gen3_randbat_source_cached
        from gpu_engine import device_set_posterior as dsp, public_belief
        root = os.environ['POKEZERO_SHOWDOWN_ROOT']
        dex, source = load_showdown_dex_cached(root), load_gen3_randbat_source_cached(root)
        tables = public_belief.compile_source(source)
        names = {kind: ['_none'] for kind in ('moves', 'abilities', 'items', 'types', 'species')}
        for variants in tables.variants[1:]:
            for v in variants:
                names['moves'] += [normalize_id(m) for m in v.moves];names['abilities'].append(normalize_id(v.ability))
                names['items'].append(normalize_id(v.item))
        names['moves'] += ['hiddenpower', 'struggle', 'flail'];names['types'] += [t.lower() for t in
            ('Normal', 'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel', 'Fire', 'Water', 'Grass',
             'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark')]
        vocab = {k: list(dict.fromkeys(v)) for k, v in names.items()}
        rows = [None] + [dict(id=m, base_power=0, type=0, source_type='Normal', category='Physical', accuracy=100, priority=0,
                              pp=10, flags={}) for m in vocab['moves'][1:]]
        meta = {'vocabulary': vocab, 'moves': rows, 'type_chart': np.ones((17, 17)).tolist()}
        compiled, unknown = dsp.compile_tables(source, dex, meta, source_tables=tables)
        self.assertEqual(unknown['ability'], ());self.assertEqual(unknown['item'], ());self.assertEqual(unknown['move'], ())
        try:
            from pokezero import set_posterior as sp
        except ImportError:
            return
        level, stats = np.asarray(compiled.level), np.asarray(compiled.stats)
        for sid, variants in enumerate(tables.variants):
            for v, variant in enumerate(variants):
                p = sp._variant_participant(source, tables.species_names[sid], variant)
                self.assertIsNotNone(p)
                self.assertEqual(int(level[sid, v]), p.level)
                self.assertEqual(stats[sid, v].tolist(), [p.stats[s] for s in ('hp', 'atk', 'def', 'spa', 'spd', 'spe')])


if __name__ == '__main__':
    unittest.main()
