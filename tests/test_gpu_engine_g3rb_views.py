"""g3rb_views' log reading: decision points and labels, censoring, per-seat private lines, the facts scan, the request's
party order, HP, PP and disabled moves, and the own team's random party order and imputed genders (fake dex and generator
tables; no Showdown, model or pinned source)."""
import collections
import random
from types import SimpleNamespace as NS
import unittest

from _showdown_root import requires_showdown, showdown_root
from gpu_engine import g3rb_views as gv

LOG = [
    '|player|p1|ua', '|player|p2|ub', '|start',
    '|switch|p1a: Gengar|Gengar, L76, M|230/230', '|-ability|p1a: Gengar|Pressure|[silent]',
    '|switch|p2a: Snorlax|Snorlax, L72, F|400/400', '|-ability|p2a: Snorlax|Pressure|[silent]', '|turn|1',
    '|switch|p2a: Starmie|Starmie, L77|260/260',
    '|move|p1a: Gengar|Thunderbolt|p2a: Starmie', '|-damage|p2a: Starmie|180/260', '|-hint|a private hint', '|turn|2',
    '|move|p2a: Starmie|Surf|p1a: Gengar', '|-damage|p1a: Gengar|100/230', '|cant|p1a: Gengar|par', '|turn|3',
    '|move|p1a: Gengar|Thunderbolt|p2a: Starmie', '|-damage|p2a: Starmie|0 fnt', '|faint|p2a: Starmie', '|upkeep',
    '|switch|p2a: Snorlax|Snorlax, L72, F|400/400', '|turn|4',
    '|move|p2a: Snorlax|Body Slam|p1a: Gengar', '|-damage|p1a: Gengar|0 fnt', '|faint|p1a: Gengar',
    '|move|p1a: Gengar|Thunderbolt|p2a: Snorlax',   # never happens in a real log; must not count once Gengar fainted
    '|switch|p1a: Dugtrio|Dugtrio, L80, M|180/180', '|turn|5',
    '|move|p1a: Dugtrio|Earthquake|p2a: Snorlax|[from]lockedmove', '|move|p2a: Snorlax|Baton Pass|p2a: Snorlax',
    '|switch|p2a: Starmie|Starmie, L77|10/260', '|turn|6',
]


class LogTests(unittest.TestCase):
    def test_decision_points_labels_and_censoring(self):
        got = [(lines_end, seat, kind, label, forced, censor, turn) for lines_end, seat, kind, label, forced, censor, turn in gv.decisions(LOG)]
        by = {(seat, turn, kind): (label, forced, censor) for _, seat, kind, label, forced, censor, turn in got}
        self.assertEqual(by[('p2', 1, 'move')], ('switch starmie', False, 'none'))   # a voluntary switch before any own action
        self.assertEqual(by[('p1', 1, 'move')], ('move thunderbolt', False, 'none'))
        self.assertEqual(by[('p1', 2, 'move')][2], 'cant')
        self.assertEqual(by[('p2', 3, 'switch')], ('switch snorlax', False, 'none'))   # the forced replacement after a faint
        self.assertEqual(by[('p1', 4, 'move')][2], 'fainted_before_acting')
        self.assertEqual(by[('p1', 4, 'switch')], ('switch dugtrio', False, 'none'))
        self.assertEqual(by[('p1', 5, 'move')], ('move earthquake', True, 'none'))   # a locked move: labelled, forced
        self.assertEqual(by[('p2', 5, 'move')], ('move batonpass', False, 'none'))
        self.assertEqual(by[('p2', 5, 'switch')], ('switch starmie', False, 'none'))   # Baton Pass's switch is its own decision
        ends = {(seat, turn, kind): end for end, seat, kind, *_, turn in got}
        self.assertEqual(LOG[ends[('p1', 1, 'move')] - 1], '|turn|1')   # a move decision's prefix ends at its turn line
        self.assertTrue(LOG[ends[('p2', 3, 'switch')]].startswith('|switch|p2a: Snorlax'))   # a switch decision's prefix stops before it

    def test_double_faints_pursuit_kos_and_recharge(self):
        log = ['|start', '|switch|p1a: Gengar|Gengar, L76|230/230', '|switch|p2a: Snorlax|Snorlax, L72|400/400', '|turn|1',
               '|move|p1a: Gengar|Explosion|p2a: Snorlax', '|-damage|p2a: Snorlax|0 fnt', '|faint|p2a: Snorlax', '|faint|p1a: Gengar', '|upkeep',
               '|switch|p2a: Tyranitar|Tyranitar, L75|300/300', '|switch|p1a: Dugtrio|Dugtrio, L80|180/180', '|turn|2',
               '|-activate|p1a: Dugtrio|move: Pursuit', '|move|p2a: Tyranitar|Pursuit|p1a: Dugtrio', '|-damage|p1a: Dugtrio|0 fnt', '|faint|p1a: Dugtrio',
               '|switch|p1a: Starmie|Starmie, L77|260/260', '|move|p2a: Tyranitar|Hyper Beam|p1a: Starmie', '|turn|3', '|cant|p2a: Tyranitar|recharge', '|turn|4']
        got = gv.decisions(log);by = {(seat, turn, kind): (end, label, censor) for end, seat, kind, label, locked, censor, turn in got}
        first = log.index('|switch|p2a: Tyranitar|Tyranitar, L75|300/300')
        self.assertEqual((by[('p2', 1, 'switch')], by[('p1', 1, 'switch')]),
                         ((first, 'switch tyranitar', 'none'), (first, 'switch dugtrio', 'none')))   # both chose at once: one shared prefix
        self.assertEqual(by[('p1', 2, 'move')][1:], ('switch starmie', 'none'))   # the switch chosen at turn start, carried out after the KO
        self.assertNotIn(('p1', 2, 'switch'), by)   # and no fabricated forced-switch decision for it
        self.assertEqual(by[('p2', 3, 'move')][2], 'recharge')

    def test_a_replacement_lost_inside_the_batch_is_its_own_decision(self):
        head = ['|start', '|switch|p1a: Gengar|Gengar, L76|230/230', '|switch|p2a: Snorlax|Snorlax, L72|400/400', '|turn|1',
                '|move|p1a: Gengar|Explosion|p2a: Snorlax', '|-damage|p2a: Snorlax|0 fnt', '|faint|p2a: Snorlax', '|faint|p1a: Gengar', '|upkeep']
        # (b) p2's first replacement dies to Spikes before p1's switch runs: p1 chose without seeing it; p2's re-replacement saw p1's switch
        spikes = head + ['|switch|p2a: Tyranitar|Tyranitar, L75|300/300', '|-damage|p2a: Tyranitar|0 fnt|[from] Spikes', '|faint|p2a: Tyranitar',
                         '|switch|p1a: Dugtrio|Dugtrio, L80|180/180', '|switch|p2a: Starmie|Starmie, L77|260/260', '|turn|2']
        ends = [(seat, end, label) for end, seat, kind, label, *_ in gv.decisions(spikes) if kind == 'switch']
        first = spikes.index('|switch|p2a: Tyranitar|Tyranitar, L75|300/300')
        self.assertEqual(ends, [('p2', first, 'switch tyranitar'), ('p1', first, 'switch dugtrio'),
                                ('p2', spikes.index('|switch|p2a: Starmie|Starmie, L77|260/260'), 'switch starmie')])
        # (a) both replace, then sandstorm KOs p1's replacement later that turn: that re-replacement sees the turn so far
        sand = head + ['|switch|p2a: Tyranitar|Tyranitar, L75|300/300', '|switch|p1a: Dugtrio|Dugtrio, L80|10/180', '|-weather|Sandstorm|[upkeep]',
                       '|-damage|p1a: Dugtrio|0 fnt|[from] Sandstorm', '|faint|p1a: Dugtrio', '|switch|p1a: Starmie|Starmie, L77|260/260', '|turn|2']
        last = [(end, label) for end, seat, kind, label, *_ in gv.decisions(sand) if kind == 'switch' and seat == 'p1'][-1]
        self.assertEqual(last, (sand.index('|switch|p1a: Starmie|Starmie, L77|260/260'), 'switch starmie'))
        # a Baton Passer KOed by Pursuit still gets its replacement decision
        bp = ['|start', '|switch|p1a: Gengar|Gengar, L76|230/230', '|switch|p2a: Snorlax|Snorlax, L72|400/400', '|turn|1',
              '|move|p1a: Gengar|Baton Pass|p1a: Gengar', '|-activate|p1a: Gengar|move: Pursuit', '|move|p2a: Snorlax|Pursuit|p1a: Gengar',
              '|-damage|p1a: Gengar|0 fnt', '|faint|p1a: Gengar', '|switch|p1a: Dugtrio|Dugtrio, L80|180/180', '|turn|2']
        self.assertIn(('p1', 'switch', 'switch dugtrio'), [(seat, kind, label) for _, seat, kind, label, *_ in gv.decisions(bp)])

    def test_each_seat_loses_the_others_private_lines(self):
        p1 = gv.seat_lines(LOG, 'p1');p2 = gv.seat_lines(LOG, 'p2')
        self.assertIn('|-ability|p1a: Gengar|Pressure|[silent]', p1);self.assertNotIn('|-ability|p2a: Snorlax|Pressure|[silent]', p1)
        self.assertIn('|-ability|p2a: Snorlax|Pressure|[silent]', p2);self.assertNotIn('|-ability|p1a: Gengar|Pressure|[silent]', p2)
        self.assertFalse(any(l.startswith('|-hint') for l in p1 + p2))
        struggle = ['|-activate|p2a: Snorlax|move: Struggle', '|-activate|p1a: Gengar|move: Struggle']
        self.assertEqual(gv.seat_lines(struggle, 'p1'), struggle[1:])

    def test_facts_collect_every_revealed_fact(self):
        mons, lead = gv.facts(LOG + ['|-enditem|p2a: Starmie|Leftovers|[eat]', '|-heal|p1a: Dugtrio|180/180|[from] item: Leftovers'])
        self.assertEqual(lead, {'p1': 'Gengar', 'p2': 'Snorlax'})
        self.assertEqual(mons['p1']['Gengar']['moves'], ['thunderbolt']);self.assertEqual(mons['p1']['Gengar']['ability'], 'pressure')
        self.assertEqual(mons['p2']['Snorlax']['moves'], ['bodyslam', 'batonpass'])
        self.assertEqual(mons['p1']['Dugtrio']['moves'], [])   # a [from]lockedmove line is not a fresh reveal
        self.assertEqual((mons['p2']['Starmie']['item'], mons['p1']['Dugtrio']['item'], mons['p1']['Dugtrio']['level']), ('leftovers', 'leftovers', 80))


class RequestTests(unittest.TestCase):
    """request() with a fake dex and generator tables."""
    def tables(self):
        info = NS(base_stats=dict(hp=60, atk=60, **{'def': 60}, spa=60, spd=60, spe=60))
        moves = {'thunderbolt': ('Special', 24), 'icepunch': ('Physical', 24), 'hypnosis': ('Status', 32), 'explosion': ('Physical', 8),
                 'hiddenpower': ('Special', 24)}
        dex = NS(species_info=lambda s: info, move_info=lambda m: NS(name=m.title(), gen3_category=moves[m][0], max_pp=moves[m][1], target='normal') if m in moves else None)
        from gpu_engine import fp_reply_reweight as rw
        return NS(rw=rw, dex=dex, gen=dict(fixed_damage=frozenset(), hp_ivs={}))

    def team(self, item='leftovers'):
        mk = lambda nick, moves: dict(nick=nick, species=nick.lower(), level=80, details=f'{nick}, L80', moves=moves, ability='levitate', item=item, revealed=True)
        return [mk('Gengar', ['thunderbolt', 'icepunch', 'hypnosis', 'explosion']), mk('Dugtrio', ['thunderbolt']), mk('Starmie', ['thunderbolt'])]

    def test_party_order_hp_pp_and_disabled_moves(self):
        lines = ['|start', '|switch|p1a: Gengar|Gengar, L80|230/230', '|turn|1', '|move|p1a: Gengar|Thunderbolt|p2a: X',
                 '|switch|p1a: Starmie|Starmie, L80|200/250 par', '|switch|p1a: Gengar|Gengar, L80|230/230', '|-damage|p1a: Gengar|115/230',
                 '|move|p1a: Gengar|Ice Punch|p2a: X', '|-start|p1a: Gengar|Disable|Hypnosis', '|turn|3']
        t = self.tables();req = gv.request(t, self.team(), lines, 'p1', 'move')
        party = req['side']['pokemon']
        # Starmie came in (swapped with slot 0), then Gengar came back (swapped again): Gengar, Dugtrio, Starmie -> Starmie, Dugtrio, Gengar -> Gengar, Dugtrio, Starmie
        self.assertEqual([m['ident'] for m in party], ['p1: Gengar', 'p1: Dugtrio', 'p1: Starmie'])
        self.assertEqual([m['active'] for m in party], [True, False, False])
        self.assertEqual((party[0]['condition'], party[2]['condition']), ('115/230', '200/250 par'))   # the log's exact own HP
        entries = {e['id']: e for e in req['active'][0]['moves']}
        self.assertEqual((entries['thunderbolt']['pp'], entries['icepunch']['pp']), (23, 23))   # one use each
        self.assertTrue(entries['hypnosis']['disabled']);self.assertFalse(entries['thunderbolt']['disabled'])
        locked = gv.request(t, self.team(item='choiceband'), lines, 'p1', 'move')
        self.assertEqual([e['disabled'] for e in locked['active'][0]['moves']], [True, False, True, True])   # locked into Ice Punch
        self.assertEqual((gv.choice(req, 'move icepunch'), gv.choice(req, 'move hiddenpower')), ('move 2', None))
        hp = self.team();hp[0]['moves'][3] = 'hiddenpowerfire';hp_req = gv.request(t, hp, lines, 'p1', 'move')
        self.assertEqual(hp_req['active'][0]['moves'][3]['id'], 'hiddenpower');self.assertEqual(gv.choice(hp_req, 'move hiddenpower'), 'move 4')
        self.assertEqual((gv.choice(req, 'switch starmie'), gv.choice(req, 'switch gengar')), ('switch 3', None))   # the active never
        struggle = gv.request(t, self.team(), lines, 'p1', 'move', struggle=True)
        self.assertEqual([e['id'] for e in struggle['active'][0]['moves']], ['struggle']);self.assertEqual(gv.choice(struggle, 'move struggle'), 'move 1')
        hp2 = self.team(item='choiceband');hp2[0]['moves'][3] = 'hiddenpowerfire'
        hpl = ['|start', '|switch|p1a: Gengar|Gengar, L80|230/230', '|turn|1', '|move|p1a: Gengar|Hidden Power|p2a: X', '|move|p1a: Gengar|Hidden Power|p2a: X',
               '|-cureteam|p1a: Gengar|[from] move: Aromatherapy', '|turn|3']
        r = gv.request(t, hp2, hpl, 'p1', 'move');e = r['active'][0]['moves']
        self.assertEqual((e[3]['pp'], [x['disabled'] for x in e]), (22, [True, True, True, False]))   # the log's family id: counted and locked on
        self.assertEqual(r['side']['pokemon'][0]['moves'][3], 'hiddenpowerfire')
        gone = gv.request(t, hp2, hpl[:-1] + ['|-enditem|p1a: Gengar|Choice Band|[from] move: Knock Off', '|turn|3'], 'p1', 'move')
        self.assertEqual((gone['side']['pokemon'][0]['item'], [x['disabled'] for x in gone['active'][0]['moves']]), ('', [False] * 4))   # lock lifted
        empty = self.team();empty[0]['moves'][3] = 'explosion'
        boom = gv.request(t, empty, ['|start', '|switch|p1a: Gengar|Gengar, L80|230/230'] + ['|move|p1a: Gengar|Explosion|p2a: X'] * 8 + ['|turn|9'], 'p1', 'move')
        self.assertEqual((boom['active'][0]['moves'][3]['pp'], boom['active'][0]['moves'][3]['disabled']), (0, True))   # no PP left: disabled
        sick = gv.request(t, self.team(), lines[:5] + ['|-cureteam|p1a: Starmie|[from] move: Aromatherapy', '|turn|2'], 'p1', 'move')
        self.assertEqual(sick['side']['pokemon'][0]['condition'], '200/250')   # Starmie's paralysis cured with the team
        forced = gv.request(t, self.team(), lines, 'p1', 'switch')
        self.assertEqual((forced['forceSwitch'], 'active' in forced), ([True], False))


class TeamTests(unittest.TestCase):
    """draw_team: the lead first, the other five in a seeded random order (the first switch-in's slot is not degenerate), imputed
    mons gendered as Showdown genders a random-battle mon (fixed, genderless, else 50/50), revealed details as logged, Species
    Clause by base species, moves in a random order, and the variants exactly those of the order-by-appearance draw."""
    SPECIES = ('gengar', 'starmie', 'dugtrio', 'snorlax', 'magneton', 'tauros', 'chansey', 'jynx')
    MOVES = ('thunderbolt', 'icepunch', 'hypnosis', 'explosion', 'hiddenpowerfire')

    def tables(self, species=SPECIES, base=None):
        variants = tuple(NS(moves=tuple(self.MOVES[j:j + 4]), ability=f'ability{j}', item=f'item{j}') for j in range(2))
        class Universe:
            level = 80
            def __init__(self):self.variants = variants
            def filter_variants(self, revealed_moves=(), revealed_ability=None, revealed_item=None):
                return [v for v in variants if set(revealed_moves) <= set(v.moves)]
        req = RequestTests.tables(RequestTests())
        dex = NS(species_info=lambda s: NS(name=s.title(), base_stats=dict(hp=60, atk=60, **{'def': 60}, spa=60, spd=60, spe=60)),
                 move_info=req.dex.move_info)
        fixed = dict(magneton='N', tauros='M', chansey='F', jynx='F', starmie='N')
        gender = {s: dict(gender=fixed.get(s, ''), base=(base or {}).get(s, s)) for s in species}
        return NS(universes={s: Universe() for s in species}, dex=dex, gender=gender, rw=req.rw, gen=req.gen)

    def facts(self):
        f = lambda sp, details: dict(species=sp, details=details, level=80, moves=['thunderbolt'], ability=None, item=None)
        return {'Gengar': f('gengar', 'Gengar, L80, M'), 'Starmie': f('starmie', 'Starmie, L80'), 'Dugtrio': f('dugtrio', 'Dugtrio, L80, F')}

    def test_the_lead_first_and_the_rest_in_a_seeded_random_order(self):
        t = self.tables();slots = collections.Counter();lines = ['|start', '|switch|p1a: Gengar|Gengar, L80, M|230/230', '|turn|1']
        for seed in range(600):
            team = gv.draw_team(t, self.facts(), 'Gengar', random.Random(seed))
            self.assertEqual((len(team), team[0]['nick']), (6, 'Gengar'))
            req = gv.request(t, team, lines, 'p1', 'move')
            slots[gv.choice(req, 'switch starmie')] += 1   # Starmie is the first switch-in: its slot used to be 2 every time
        self.assertEqual(set(slots), {f'switch {k}' for k in range(2, 7)})
        self.assertTrue(all(70 < c < 170 for c in slots.values()), slots)   # about 120 each
        same = [[(m['nick'], tuple(m['moves'])) for m in gv.draw_team(t, self.facts(), 'Gengar', random.Random(7))] for _ in range(2)]
        self.assertEqual(same[0], same[1])   # seeded

    def test_variants_are_those_of_the_order_by_appearance_draw(self):
        def reference(tables, seat_facts, lead, rng):   # draw_team before the order and gender fix
            team = []
            for nick in [lead] + [n for n in seat_facts if n != lead]:
                f = seat_facts[nick];u = tables.universes[f['species']];v = rng.choice(list(u.filter_variants(revealed_moves=tuple(f['moves']))))
                team.append((nick, f['species'], frozenset(v.moves), v.ability, v.item))
            pool = sorted(s for s in tables.universes if s not in {m[1] for m in team})
            while len(team) < 6 and pool:
                s = pool.pop(rng.randrange(len(pool)));v = rng.choice(list(tables.universes[s].variants))
                team.append((s.title(), s, frozenset(v.moves), v.ability, v.item))
            return sorted(team)
        t = self.tables()
        for seed in range(200):
            got = sorted((m['nick'], m['species'], frozenset(m['moves']), m['ability'], m['item'])
                         for m in gv.draw_team(t, self.facts(), 'Gengar', random.Random(seed)))
            self.assertEqual(got, reference(t, self.facts(), 'Gengar', random.Random(seed)), seed)

    def test_moves_in_a_random_order(self):
        t = self.tables();orders = collections.Counter()
        for seed in range(200):
            gengar = gv.draw_team(t, self.facts(), 'Gengar', random.Random(seed))[0]
            orders[tuple(gengar['moves'])] += 1;self.assertEqual(len(set(gengar['moves'])), 4)
        self.assertGreater(len(orders), 12)   # 24 orders of 4 moves (per variant): not the movepool's own order every time

    def test_imputed_mons_are_gendered_as_showdown_does_and_revealed_details_stay_as_logged(self):
        t = self.tables();seen = collections.defaultdict(collections.Counter)
        for seed in range(400):
            for m in gv.draw_team(t, self.facts(), 'Gengar', random.Random(seed)):
                seen[m['species']][m['details'].split(', ')[-1] if m['details'].count(',') == 2 else ''] += 1
                if m['revealed']:self.assertEqual(m['details'], self.facts()[m['nick']]['details'])
        self.assertEqual([set(seen[s]) for s in ('magneton', 'tauros', 'chansey', 'jynx')], [{''}, {'M'}, {'F'}, {'F'}])
        snorlax = seen['snorlax'];self.assertTrue(.35 < snorlax['M'] / sum(snorlax.values()) < .65, snorlax)   # 50/50, not the 87.5% ratio
        self.assertEqual(set(seen['starmie']), {''})   # revealed without a gender in its log: kept so

    def test_species_clause_by_base_species(self):
        species = self.SPECIES + ('deoxys', 'deoxysattack', 'deoxysdefense', 'deoxysspeed')
        t = self.tables(species, base={s: 'deoxys' for s in species if s.startswith('deoxys')})
        facts = dict(self.facts(), Deoxys=dict(species='deoxysspeed', details='Deoxys-Speed, L72', level=72, moves=['thunderbolt'], ability=None, item=None))
        for seed in range(300):
            team = gv.draw_team(t, facts, 'Gengar', random.Random(seed))
            self.assertEqual([m['species'] for m in team if m['species'].startswith('deoxys')], ['deoxysspeed'])   # no second Deoxys
            self.assertEqual(len({m['nick'] for m in team}), 6)   # so no nickname clash in request()'s by_nick

    def test_species_clause_after_an_imputed_forme_is_drawn(self):
        species = ('gengar', 'starmie', 'dugtrio', 'tauros', 'chansey', 'deoxys', 'deoxysattack', 'deoxysdefense', 'deoxysspeed')
        t = self.tables(species, base={s: 'deoxys' for s in species if s.startswith('deoxys')});drawn = 0
        for seed in range(300):
            team = gv.draw_team(t, self.facts(), 'Gengar', random.Random(seed))   # 3 slots from 2 others and 4 Deoxys formes
            formes = [m['species'] for m in team if m['species'].startswith('deoxys')]
            self.assertLessEqual(len(formes), 1);self.assertEqual(len(team), 6);drawn += len(formes)
        self.assertGreater(drawn, 100)   # a forme is drawn often, and its siblings are then dropped from the pool

    def test_gender_draw(self):
        table = {'tauros': dict(gender='M'), 'magneton': dict(gender='N'), 'snorlax': dict(gender='')}
        rng = random.Random(0);draws = collections.Counter(gv.draw_gender(table, 'snorlax', rng) for _ in range(4000))
        self.assertEqual((gv.draw_gender(table, 'tauros', rng), gv.draw_gender(table, 'magneton', rng)), ('M', ''))
        self.assertTrue(.47 < draws['M'] / 4000 < .53);self.assertEqual(set(draws), {'M', 'F'})
        with self.assertRaises(KeyError):gv.draw_gender(table, 'unknownmon', rng)   # every pinned species must be in the table

    @requires_showdown('needs a built Showdown checkout')
    @unittest.skipUnless(__import__('shutil').which('node'), 'needs node')
    def test_gender_table_from_showdown(self):
        if not (showdown_root() / 'dist' / 'sim').exists():self.skipTest(f'no built dist/sim in {showdown_root()}')
        table = gv.gender_table(showdown_root())
        self.assertEqual({s: table[s]['gender'] for s in ('tauros', 'starmie', 'chansey', 'snorlax', 'magneton')},
                         dict(tauros='M', starmie='N', chansey='F', snorlax='', magneton='N'))
        self.assertEqual((table['deoxysspeed']['base'], table['gengar']['base']), ('deoxys', 'gengar'))

if __name__ == '__main__':
    unittest.main()
