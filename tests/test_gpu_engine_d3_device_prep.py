"""D3 device preparation, CPU side: layout contract, CPython MT19937 model, source tables and the
sampling order of plan_starts, checked against the oracle on a synthetic source over many seeds."""
import json
import random
import re
import unittest
from pathlib import Path
from types import MappingProxyType

import numpy as np

from pokezero import v5_contract as contract
from pokezero.dex import normalize_id
from pokezero.determinization import _gen3_randbat_fixture_spread, _normalize_species_id, _sample_gender
from pokezero.engine_world import hidden_power_engine_id
from pokezero.gen3_damage import gen3_hp_stat, gen3_stat
from pokezero.randbat import Gen3RandbatSource
from pokezero.v5_gumbel import Recipe
from gpu_engine import device_public_source_tables as tables_
from gpu_engine import device_public_world_state as mapping
from gpu_engine import device_public_worlds as plans

HEADER = Path(__file__).resolve().parents[1] / 'gpu_engine' / 'cuda' / 'v2e_prep.cuh'


# ---- exact CPython MT19937, transliterated from gpu_engine/cuda/v2e_mt19937.cuh ----------------
class MT:
    N, M = 624, 397

    def __init__(self, seed):
        words = list(np.frombuffer(int(seed).to_bytes(32, 'little'), np.uint32))
        used = len(words)
        while used > 1 and words[used - 1] == 0:
            used -= 1
        self.mt = [0] * self.N
        self._init_genrand(19650218)
        i, jj = 1, 0
        for _ in range(max(self.N, used)):
            self.mt[i] = ((self.mt[i] ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) * 1664525)) + int(words[jj]) + jj) & 0xFFFFFFFF
            i += 1;jj += 1
            if i >= self.N:self.mt[0] = self.mt[self.N - 1];i = 1
            if jj >= used:jj = 0
        for _ in range(self.N - 1):
            self.mt[i] = ((self.mt[i] ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) * 1566083941)) - i) & 0xFFFFFFFF
            i += 1
            if i >= self.N:self.mt[0] = self.mt[self.N - 1];i = 1
        self.mt[0] = 0x80000000
        self.index = self.N

    def _init_genrand(self, s):
        self.mt[0] = s
        for i in range(1, self.N):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.index = self.N

    def genrand(self):
        if self.index >= self.N:
            mt = self.mt
            for kk in range(self.N):
                y = (mt[kk] & 0x80000000) | (mt[(kk + 1) % self.N] & 0x7fffffff)
                mt[kk] = mt[(kk + self.M) % self.N] ^ (y >> 1) ^ (0x9908b0df if y & 1 else 0)
            self.index = 0
        y = self.mt[self.index];self.index += 1
        y ^= y >> 11;y ^= (y << 7) & 0x9d2c5680;y ^= (y << 15) & 0xefc60000;y ^= y >> 18
        return y & 0xFFFFFFFF

    def getrandbits(self, k):
        return self.genrand() >> (32 - k)

    def randbelow(self, n):
        k = n.bit_length();r = self.getrandbits(k)
        while r >= n:r = self.getrandbits(k)
        return r


# ---- synthetic source ----------------------------------------------------------------------
VOCABULARY = dict(
    species=['', 'Pikachu', 'Gyarados', 'Unown', 'Unown-C', 'Shedinja', 'Ditto', 'Snorlax', 'Magikarp', 'Slakoth', 'Zapdos'],
    moves=['', 'Thunderbolt', 'Surf', 'Hidden Power Ice', 'Hidden Power Flying', 'Hidden Power Psychic',
        'Hidden Power Fighting', 'Substitute', 'Volt Tackle', 'Brick Break', 'Return', 'Quick Attack', 'Dragon Dance',
        'Earthquake', 'Double-Edge', 'Shadow Ball', 'Silver Wind', 'Transform', 'Body Slam', 'Rest', 'Sleep Talk',
        'Curse', 'Self-Destruct', 'Struggle', 'Splash', 'Tackle', 'Slack Off', 'Drill Peck', 'Thunder Wave'],
    abilities=['', 'Static', 'Intimidate', 'Levitate', 'Wonder Guard', 'Limber', 'Thick Fat', 'Swift Swim', 'Truant', 'Pressure'],
    items=['', 'Light Ball', 'Choice Band', 'Leftovers', 'Lum Berry'],
    types=['', 'Normal', 'Electric', 'Water', 'Flying', 'Psychic', 'Bug', 'Ghost', 'Ice', 'Fighting', 'Fire', 'Grass',
        'Ground', 'Rock', 'Dark', 'Dragon', 'Steel', 'Poison'])
BASE = dict(hp=60, atk=70, spa=80, spd=75, spe=65)
BASE['def'] = 55
SOURCE_PAYLOAD = dict(
    metadata=dict(format_id='gen3randombattle', generation=3, showdown_root=None, sets_path=None, generator_path=None,
        source_hash='synthetic-d3-device-prep'),
    universes=dict(
        pikachu=dict(species='Pikachu', level=88, variants=[
            dict(variant_id='pika-1', source_set_id='a', role='Fast Attacker', level=88,
                moves=['Thunderbolt', 'Surf', 'Hidden Power Ice', 'Substitute'], ability='Static', item='Light Ball'),
            dict(variant_id='pika-2', source_set_id='b', role='Wallbreaker', level=88,
                moves=['Volt Tackle', 'Brick Break', 'Return', 'Quick Attack'], ability='Static', item='Choice Band')]),
        gyarados=dict(species='Gyarados', level=78, variants=[
            dict(variant_id='gyara-1', source_set_id='a', role='Setup', level=78,
                moves=['Dragon Dance', 'Earthquake', 'Hidden Power Flying', 'Double-Edge'], ability='Intimidate', item='Leftovers')]),
        unown=dict(species='Unown', level=100, variants=[
            dict(variant_id='unown-1', source_set_id='a', role='Unknown', level=100,
                moves=['Hidden Power Psychic'], ability='Levitate', item='Twisted Spoon')]),
        shedinja=dict(species='Shedinja', level=90, variants=[
            dict(variant_id='shed-1', source_set_id='a', role='Wall', level=90,
                moves=['Shadow Ball', 'Silver Wind', 'Hidden Power Fighting', 'Substitute'], ability='Wonder Guard', item='Lum Berry')]),
        ditto=dict(species='Ditto', level=95, variants=[
            dict(variant_id='ditto-1', source_set_id='a', role='Copy', level=95, moves=['Transform'], ability='Limber', item='Choice Band')]),
        snorlax=dict(species='Snorlax', level=80, variants=[
            dict(variant_id='lax-1', source_set_id='a', role='Rest', level=80,
                moves=['Body Slam', 'Earthquake', 'Rest', 'Sleep Talk'], ability='Thick Fat', item='Leftovers'),
            dict(variant_id='lax-2', source_set_id='b', role='Curse', level=80,
                moves=['Curse', 'Return', 'Shadow Ball', 'Self-Destruct'], ability='Thick Fat', item='Leftovers'),
            dict(variant_id='lax-3', source_set_id='c', role='Odd', level=80,
                moves=['Body Slam', 'Fake Move', 'Rest', 'Sleep Talk'], ability='Thick Fat', item='Leftovers')]),
        magikarp=dict(species='Magikarp', level=100, variants=[
            dict(variant_id='karp-1', source_set_id='a', role='None', level=100, moves=[], ability='Swift Swim', item='')]),
        slakoth=dict(species='Slakoth', level=100, variants=[
            dict(variant_id='slak-1', source_set_id='a', role='Loaf', level=100, moves=['Slack Off', 'Return'], ability='Truant', item='Leftovers')]),
        zapdos=dict(species='Zapdos', level=76, variants=[
            dict(variant_id='zap-1', source_set_id='a', role='Bird', level=76,
                moves=['Thunderbolt', 'Drill Peck', 'Thunder Wave', 'Rest'], ability='Pressure', item='Leftovers'),
            dict(variant_id='zap-2', source_set_id='b', role='Bird2', level=76,
                moves=['Thunderbolt', 'Drill Peck', 'Hidden Power Ice', 'Substitute'], ability='Pressure', item='Leftovers')])),
    move_metadata={normalize_id(m): dict(category='Special' if m in ('Thunderbolt', 'Surf', 'Shadow Ball') else 'Physical',
        type='Normal') for m in VOCABULARY['moves'][1:]},
    species_metadata=dict(
        pikachu=dict(baseStats=dict(BASE, hp=35, spe=90), types=['Electric'], gender=None),
        gyarados=dict(baseStats=dict(BASE, hp=95, atk=125), types=['Water', 'Flying'], gender=''),
        unown=dict(baseStats=dict(BASE, hp=48), types=['Psychic'], gender='N'),
        shedinja=dict(baseStats=dict(BASE, hp=1), types=['Bug', 'Ghost'], gender='N'),
        ditto=dict(baseStats=dict(BASE, hp=48), types=['Normal'], gender='n'),
        snorlax=dict(baseStats=dict(BASE, hp=160, atk=110), types=['Normal'], gender='M'),
        magikarp=dict(baseStats=dict(BASE, hp=20), types=['Water'], gender='F'),
        slakoth=dict(baseStats=dict(BASE, hp=60), types=['Normal']),
        zapdos=dict(baseStats=dict(BASE, hp=90, spa=125), types=['Electric', 'Flying'], gender='N')))
METADATA = dict(vocabulary=VOCABULARY, moves=[{}] + [dict(pp=5 + 5 * (i % 4)) for i in range(1, len(VOCABULARY['moves']))])


def synthetic():
    source = Gen3RandbatSource.from_payload(SOURCE_PAYLOAD)
    census = contract.source_variant_census(source)
    bound = plans.compile_source(source, METADATA, expected_source_hash=source.metadata.source_hash,
        expected_source_rows_sha256=census['source_rows_sha256'])
    tables = mapping.compile_tables(bound, METADATA)
    return bound, tables, tables_.compile_device_source(tables)


# ---- a pure-Python model of the kernel's plan stage (v2e_prep.cu) --------------------------------
def model_plan(device, view, seed):
    """(plan string or None, fixtures [(universe, variant, gender)], revealed indices)."""
    U, V, R = tables_.U, tables_.V, tables_.R
    uni, var, misc = device.uni, device.var, device.misc
    S, M, A, I = (int(misc[tables_.MISC[k]]) for k in ('MISC_SPECIES', 'MISC_MOVES', 'MISC_ABILITIES', 'MISC_ITEMS'))
    Vn = int(misc[tables_.MISC['MISC_VARIANTS']]);struggle = int(misc[tables_.MISC['MISC_STRUGGLE']])
    strings = device.strings
    features = view.get('mon_features');swapped = None if features is None else features.get('item_swapped')
    if int(view['error'][0]):return strings[R['R_TRACKER_ERROR']], [], ()
    revealed = [];seen = set()
    for slot in range(6):
        sp = int(view['species'][0, slot]);m0, m1 = (int(x) for x in view['belief_mask'][0, slot])
        if not view['present'][0, slot]:
            if sp != 0 or m0 or m1:return strings[R['R_UNREVEALED_EXPOSES']], [], ()
            continue
        if not 1 <= sp < S:return strings[R['R_NS_SPECIES']], [], ()
        u = int(device.species_to_universe[sp])
        if u < 0 or uni[u, U['U_VAR_CNT']] == 0:return strings[R['R_NO_SOURCE']], [], ()
        if u in seen:return strings[R['R_DUPLICATE_SPECIES']], [], ()
        seen.add(u);count = int(uni[u, U['U_VAR_CNT']]);raw = m0 | m1 << 32
        if count > 64 or raw >> count:return strings[R['R_BITS_OUTSIDE']], [], ()
        if not raw:return strings[R['R_ZERO_MASS']], [], ()
        shown = []
        for v in view['revealed_moves'][0, slot]:
            v = int(v)
            if not v:continue
            if not 1 <= v < M:return strings[R['R_NS_MOVES']], [], ()
            shown.append(v)
        ab = int(view['ability'][0, slot])
        if not 0 <= ab < A:return strings[R['R_NS_ABILITIES']], [], ()
        it = int(view['item'][0, slot])
        if not 0 <= it < I:return strings[R['R_NS_ITEMS']], [], ()
        if it and swapped is None:return strings[R['R_ITEM_SWAP_UNAVAILABLE']], [], ()
        if swapped is not None and swapped[0, slot]:it = 0
        off = int(uni[u, U['U_VAR_OFF']]);level = int(view['level'][0, slot]);eligible = []
        for index in range(count):
            if not raw >> index & 1:continue
            vr = var[0, off + index];ok = True
            for m in shown:
                if m == struggle:continue
                if device.move_is_hp[m]:ok = ok and bool(vr[V['V_HAS_HP']])
                else:ok = ok and m in [int(vr[V['V_MOVE0'] + k]) for k in range(4)]
            if ab and vr[V['V_ABILITY']] != ab:ok = False
            if it and vr[V['V_ITEM']] != it:ok = False
            if vr[V['V_LEVEL']] != level:ok = False
            if ok:eligible.append(index)
        if not eligible:return strings[R['R_NO_CONSISTENT']], [], ()
        g = int(view['gender'][0, slot])
        if g not in (-1, 0, 1, 2):return strings[R['R_GENDER_NS']], [], ()
        revealed.append((slot, u, eligible, g))
    rng = MT(seed);fixtures = [];indices = [];used = set()
    for slot, u, eligible, g in revealed:
        if g < 0:
            fixed = int(uni[u, U['U_GENDER']]);g = fixed if fixed >= 0 else (1 if rng.randbelow(2) == 0 else 2)
        k = eligible[rng.randbelow(len(eligible))]
        if var[1, int(uni[u, U['U_VAR_OFF']]) + k, V['V_FIXTURE_NONE']]:return strings[R['R_REVEALED_FIXTURE']], fixtures, tuple(indices)
        fixtures.append((u, k, g));indices.append((slot, k));used.add(u)
    candidates = [u for u in range(len(uni)) if uni[u, U['U_VAR_CNT']] > 0 and u not in used]
    for _ in range(6 - len(fixtures)):
        if not candidates:return strings[R['R_HIDDEN_SOURCE']], fixtures, tuple(indices)
        u = candidates.pop(rng.randbelow(len(candidates)))
        k = rng.randbelow(int(uni[u, U['U_VAR_CNT']]))
        fixed = int(uni[u, U['U_GENDER']]);g = fixed if fixed >= 0 else (1 if rng.randbelow(2) == 0 else 2)
        fixtures.append((u, k, g))
    return None, fixtures, tuple(indices)


def random_view(rng, device, bound):
    """A synthetic public readout (host arrays as detach_roots freezes them) for plan_starts."""
    uni = device.uni;U = tables_.U
    species = np.zeros((1, 6), np.int32);present = np.zeros((1, 6), bool);masks = np.zeros((1, 6, 2), np.uint32)
    moves = np.zeros((1, 6, 4), np.int32);ability = np.zeros((1, 6), np.int32);item = np.zeros((1, 6), np.int32)
    gender = np.full((1, 6), -1, np.int32);level = np.zeros((1, 6), np.int32);swapped = np.zeros((1, 6), np.int32)
    vocab = bound.vocabulary;universes = list(bound.original.universes.values())
    species_ids = [i for i in range(1, len(vocab['species']))]
    chosen = []
    for slot in range(6):
        if rng.random() < 0.55:
            present[0, slot] = True
            sp = rng.choice(species_ids);u = int(device.species_to_universe[sp])
            if u >= 0 and u in chosen and rng.random() < 0.8:
                sp = rng.choice(species_ids);u = int(device.species_to_universe[sp])
            chosen.append(u);species[0, slot] = sp
            count = int(uni[u, U['U_VAR_CNT']]) if u >= 0 else 1
            raw = rng.randrange(1, 1 << count) if rng.random() < 0.9 else rng.randrange(0, 1 << (count + 1))
            masks[0, slot] = (raw & 0xFFFFFFFF, raw >> 32)
            if u >= 0 and count:
                variant = universes[u].variants[rng.randrange(count)]
                shown = [m for m in variant.moves if rng.random() < 0.5]
                if rng.random() < 0.15:shown.append(rng.choice(vocab['moves'][1:]))
                for k, name in enumerate(shown[:4]):
                    ids = [i for i, v in enumerate(vocab['moves']) if normalize_id(v) == normalize_id(name)]
                    moves[0, slot, k] = ids[0] if ids else 0
                if rng.random() < 0.4:
                    ids = [i for i, v in enumerate(vocab['abilities']) if normalize_id(v) == normalize_id(variant.ability)]
                    ability[0, slot] = ids[0] if ids else rng.randrange(len(vocab['abilities']))
                if rng.random() < 0.4:
                    ids = [i for i, v in enumerate(vocab['items']) if normalize_id(v) == normalize_id(variant.item)]
                    item[0, slot] = ids[0] if ids else rng.randrange(len(vocab['items']))
                level[0, slot] = universes[u].level if rng.random() < 0.95 else 50
            gender[0, slot] = rng.choice([-1, -1, 0, 1, 2, 3]) if rng.random() < 0.3 else -1
            swapped[0, slot] = int(rng.random() < 0.2)
        elif rng.random() < 0.03:
            species[0, slot] = 1
    view = dict(error=np.asarray([int(rng.random() < 0.02)], np.int32), species=species, present=present, belief_mask=masks,
        revealed_moves=moves, ability=ability, item=item, gender=gender, level=level,
        mon_features=None if rng.random() < 0.1 else dict(item_swapped=swapped))
    return plans._freeze(view)


class HeaderLayoutTest(unittest.TestCase):
    def test_python_layout_matches_cuda_header(self):
        text = HEADER.read_text()
        values = {}
        for body in re.findall(r'enum\s+\w+\s*\{(.*?)\}\s*;', text, re.S):
            body = re.sub(r'//.*', '', body);next_value = 0
            for entry in body.split(','):
                entry = entry.strip()
                if not entry:continue
                if '=' in entry:
                    name, expression = (x.strip() for x in entry.split('=', 1))
                    next_value = eval(expression, {}, dict(values))
                else:name = entry
                values[name] = next_value;next_value += 1
        for line in re.findall(r'constexpr int ([^;]+);', text):
            for entry in line.split(','):
                name, expression = (x.strip() for x in entry.split('='))
                values[name] = eval(expression, {}, dict(values))
        expected = tables_.header_enumerations()
        for name, value in expected.items():
            self.assertIn(name, values, name)
            self.assertEqual(values[name], value, name)
        self.assertEqual(len(tables_.STATIC_REFUSALS), len(tables_.STATIC_REFUSAL_NAMES))


class Mt19937ModelTest(unittest.TestCase):
    def seeds(self):
        rng = np.random.default_rng(7);recipe = Recipe(20261104)
        seeds = [0, 1, 2, 2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**64 - 1, 2**64, 2**224 - 1, 2**255, 2**256 - 1, 2**32 * 7]
        seeds += [int.from_bytes(rng.bytes(32), 'little') for _ in range(6)]
        seeds += [recipe.world_seed(contract.DecisionAddress(1803000001, 12, 1), w) for w in (0, 1)]
        return seeds

    def test_getrandbits_matches_random(self):
        for seed in self.seeds():
            model = MT(seed);reference = random.Random(seed)
            self.assertEqual([model.getrandbits(32) for _ in range(2000)], [reference.getrandbits(32) for _ in range(2000)], seed)

    def test_randrange_matches_random(self):
        for seed in self.seeds()[:10]:
            for n in (1, 2, 3, 7, 100, 255, 256, 257, 1000, 2**31 - 1, 2**31, 2**32 - 1):
                model = MT(seed);reference = random.Random(seed)
                self.assertEqual([model.randbelow(n) for _ in range(300)], [reference.randrange(n) for _ in range(300)], (seed, n))

    def test_mixed_draws_match_random(self):
        for seed in self.seeds():
            model = MT(seed);reference = random.Random(seed);draws = random.Random(seed ^ 1)
            for _ in range(500):
                n = draws.choice((1, 2, 3, 5, 12, 64, 200, 4096))
                self.assertEqual(model.randbelow(n), reference.randrange(n))


class SourceTablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bound, cls.tables, cls.device = synthetic()

    def test_universe_order_and_variant_offsets(self):
        original = self.bound.original;device = self.device;U = tables_.U
        self.assertEqual(device.universe_keys, tuple(original.universes))
        offset = 0
        for u, universe in enumerate(original.universes.values()):
            self.assertEqual((int(device.uni[u, U['U_VAR_OFF']]), int(device.uni[u, U['U_VAR_CNT']])), (offset, len(universe.variants)))
            offset += len(universe.variants)
        self.assertEqual(device.var.shape, (2, offset, len(tables_.VAR_COLUMNS)))

    def test_species_join_matches_universe_for(self):
        original = self.bound.original;keys = list(original.universes)
        for sp, name in enumerate(self.bound.vocabulary['species']):
            universe = original.universe_for(name) if sp else None
            expected = -1 if universe is None else keys.index(_normalize_species_id(universe.species))
            self.assertEqual(int(self.device.species_to_universe[sp]), expected, name)
        self.assertEqual(int(self.device.species_to_universe[3]), int(self.device.species_to_universe[4]))  # Unown, Unown-C

    def test_gender_codes_match_sample_gender(self):
        original = self.bound.original;U = tables_.U
        for u, universe in enumerate(original.universes.values()):
            drawn = {_sample_gender(universe.species, set_source=original, rng=random.Random(s)) for s in range(40)}
            code = int(self.device.uni[u, U['U_GENDER']])
            if code < 0:self.assertEqual(drawn, {'M', 'F'}, universe.species)
            else:self.assertEqual(drawn, {{0: 'N', 1: 'M', 2: 'F'}[code]}, universe.species)

    def test_variant_rows_match_oracle_derivation(self):
        """Hidden-case writes re-derived from the oracle functions directly; refusals name the namespace."""
        original = self.bound.original;tables = self.tables;device = self.device;V = tables_.V;names = tables.namespaces
        v = 0
        for universe in original.universes.values():
            for variant in universe.variants:
                row = device.var[0, v];v += 1
                self.assertEqual([int(row[V['V_MOVE0'] + k]) for k in range(4)],
                    [names['moves'].get(normalize_id(m), 0) for m in variant.moves] + [0] * (4 - len(variant.moves)))
                self.assertEqual(int(row[V['V_LEVEL']]), variant.level)
                self.assertEqual(int(row[V['V_ABILITY']]), names['abilities'].get(normalize_id(variant.ability), 0))
                self.assertEqual(int(row[V['V_ITEM']]), names['items'].get(normalize_id(variant.item), 0))
                refusal = device.strings[int(row[V['V_REFUSAL']])] if row[V['V_REFUSAL']] else None
                unknown_move = [m for m in variant.moves if normalize_id(m) not in names['moves']]
                if not variant.moves:
                    self.assertEqual(refusal, 'sampled source moves unavailable');continue
                if normalize_id(variant.item) and normalize_id(variant.item) not in names['items']:
                    self.assertEqual(refusal, 'sampled source namespace unavailable: items:' + normalize_id(variant.item));continue
                if unknown_move:
                    self.assertEqual(refusal, 'sampled source namespace unavailable: moves:' + normalize_id(unknown_move[0]));continue
                self.assertIsNone(refusal, variant.variant_id)
                spread = _gen3_randbat_fixture_spread({}, species=variant.species, moves=tuple(variant.moves), item=variant.item,
                    level=variant.level, set_source=original)
                base = original.species_metadata[_normalize_species_id(variant.species)]['baseStats']
                evs, ivs = spread['evs'], spread['ivs']
                maximum = 1 if base['hp'] == 1 else gen3_hp_stat(base['hp'], ivs['hp'], evs['hp'], variant.level)
                stats = [maximum] + [gen3_stat(base[k], ivs[k], evs[k], variant.level) for k in ('atk', 'def', 'spa', 'spd', 'spe')]
                self.assertEqual([int(x) for x in row[V['V_STAT0']:V['V_STAT5'] + 1]], stats, variant.variant_id)
                ids = [names['moves'][normalize_id(m)] for m in variant.moves]
                self.assertEqual([int(x) for x in row[V['V_WMOVE0']:V['V_WMOVE3'] + 1]], ids + [0] * (4 - len(ids)))
                self.assertEqual([int(x) for x in row[V['V_PP0']:V['V_PP3'] + 1]], [tables.max_pp[m] for m in ids] + [0] * (4 - len(ids)))
                self.assertEqual(int(row[V['V_WNMOVES']]), len(variant.moves))
                types = original.species_metadata[_normalize_species_id(variant.species)]['types']
                self.assertEqual([int(x) for x in row[V['V_TYPE0']:V['V_TYPE1'] + 1]],
                    [names['types'][normalize_id(t)] for t in types] + [0] * (2 - len(types)))
                hp = [m for m in variant.moves if normalize_id(m).startswith('hiddenpower')]
                if hp:
                    resolved = hidden_power_engine_id(normalize_id(hp[-1]), ivs)
                    self.assertEqual(int(row[V['V_HP_TYPE']]), names['types'][resolved[len('hiddenpower'):-2]])
                    self.assertEqual(int(row[V['V_HP_POWER']]), int(resolved[-2:]))
                else:self.assertEqual((int(row[V['V_HP_TYPE']]), int(row[V['V_HP_POWER']])), (0, 70))
                self.assertEqual(int(row[V['V_CHOICEBAND']]), int(normalize_id(variant.item) == 'choiceband'))
                self.assertEqual(int(row[V['V_SPECIES']]), names['species'][normalize_id(variant.species)])
                self.assertEqual(int(row[V['V_HAS_HP']]), int(bool(hp)))
                revealed = device.var[1, v - 1]
                self.assertEqual(int(revealed[V['V_SPECIES']]), 0)
                self.assertEqual(revealed[V['V_STAT0']:V['V_REFUSAL']].tolist(), row[V['V_STAT0']:V['V_REFUSAL']].tolist())

    def test_fixture_none_only_for_the_revealed_case(self):
        V = tables_.V;device = self.device
        none = [v for v in range(device.var.shape[1]) if device.var[1, v, V['V_FIXTURE_NONE']]]
        self.assertEqual([device.variant_ids[u][k] for u in range(len(device.variant_ids)) for k in range(len(device.variant_ids[u]))
            if sum(map(len, device.variant_ids[:u])) + k in none], ['karp-1'])
        self.assertFalse(device.var[0, :, V['V_FIXTURE_NONE']].any())

    def test_misc_table(self):
        misc = self.device.misc;MISC = tables_.MISC;names = self.tables.namespaces
        self.assertEqual(int(misc[MISC['MISC_STRUGGLE']]), names['moves']['struggle'])
        self.assertEqual(int(misc[MISC['MISC_TRUANT']]), names['abilities']['truant'])
        header = len(tables_.MISC_COLUMNS);volatile = misc[header:header + self.device.volatile_count]
        from gpu_engine.public_active import NAMES
        for kind, name in enumerate(NAMES):
            if name == 'leechseed':self.assertEqual(int(volatile[kind]), tables_.VOL_LEECH_SEED)
            elif name == 'ingrain':self.assertEqual(int(volatile[kind]), tables_.VOL_INGRAIN)
            elif name == 'trapped':self.assertEqual(int(volatile[kind]), tables_.VOL_TRAPPED)
            else:self.assertEqual(self.device.strings[int(volatile[kind])], 'public volatile/latent clock not materialized: ' + name)
        for f, name in enumerate(tables_.FEATURES):
            self.assertEqual(self.device.strings[int(misc[header + self.device.volatile_count + f])], 'public current feature unavailable: ' + name)
        self.assertTrue(all(normalize_id(self.bound.vocabulary['moves'][m]).startswith('hiddenpower') == bool(self.device.move_is_hp[m])
            for m in range(len(self.bound.vocabulary['moves']))))


class PlanOrderTest(unittest.TestCase):
    """The kernel's sampling model reproduces plan_starts (gender, variant, hidden universe pop, gender)."""

    @classmethod
    def setUpClass(cls):
        cls.bound, cls.tables, cls.device = synthetic()

    def describe(self, fixture):
        return (_normalize_species_id(fixture.species), tuple(fixture.moves), normalize_id(fixture.ability or ''),
            normalize_id(fixture.item or ''), fixture.level, fixture.gender)

    def describe_model(self, sampled):
        universes = list(self.bound.original.universes.values())
        u, k, g = sampled;variant = universes[u].variants[k]
        return (_normalize_species_id(variant.species), tuple(variant.moves), normalize_id(variant.ability),
            normalize_id(variant.item), variant.level, {0: 'N', 1: 'M', 2: 'F'}[g])

    def test_model_matches_plan_starts(self):
        recipe = Recipe(99);rng = random.Random(2026);compared = 0;refused = 0;outcomes = set()
        for trial in range(1200):
            address = contract.DecisionAddress(trial, rng.randrange(60), rng.randrange(2))
            if not recipe.selected(address):continue
            view = random_view(rng, self.device, self.bound)
            root = plans.Root(address, MappingProxyType({}), view, None, self.bound.binding_sha256, False)
            result = plans.plan_starts(root, self.bound, recipe)
            for start in result.starts:
                failure, fixtures, indices = model_plan(self.device, view, start.seed)
                outcomes.add(failure)
                self.assertEqual(start.refusals, () if failure is None else (failure,), (trial, start.world))
                self.assertEqual(start.revealed_source_indices, indices)
                if failure is None:
                    self.assertEqual([self.describe(f) for f in start.opponent_fixtures], [self.describe_model(f) for f in fixtures])
                    compared += 1
                else:
                    self.assertIsNone(start.opponent_fixtures);refused += 1
        self.assertGreater(compared, 60)
        self.assertGreater(refused, 10)
        self.assertGreaterEqual(len(outcomes), 4, outcomes)


if __name__ == '__main__':
    unittest.main()
