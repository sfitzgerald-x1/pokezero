"""pokezero.depthn_effect_guard on exact Gen 3 protocol lines: the tier-C conditions, (b) at turn 1 and turn 2, (c) per choice."""
import json
import math
import unittest

from pokezero import depthn_effect_guard as G
from pokezero import depthn_live_actor as actor


def switch(seat, species, hp='100/100', details=None):
    return f'|switch|{seat}a: {species}|{details or species}, L80|{hp}'


def request(members, *, side='p1', moves=('spore', 'gigadrain', 'leechseed', 'substitute')):
    """Our |request| line: members [(species, condition, item, ability)], the first active."""
    pokemon = [dict(ident=f'{side}: {s}', details=f'{s}, L80', condition=c, item=i, ability=a, baseAbility=a, active=k == 0)
               for k, (s, c, i, a) in enumerate(members)]
    return '|request|' + json.dumps(dict(side=dict(id=side, pokemon=pokemon), active=[dict(moves=[dict(id=m, pp=8, maxpp=8) for m in moves])]))


OURS = [('Breloom', '100/100', 'leftovers', 'effectspore'), ('Snorlax', '300/300', 'leftovers', 'thickfat')]
BASE = [switch('p1', 'Breloom'), switch('p2', 'Snorlax'), '|turn|1']


class SleepClauseTests(unittest.TestCase):
    """Showdown f76228a1 data/rulesets.ts sleepclausemod: a sleep move fails while the target's side has a non-fainted Pokemon
    asleep from a non-allied source; Rest's sleep (|-status|X|slp|[from] move: Rest) is allied, Yawn's is not."""
    def clause(self, body, owner=0, ours=OURS):
        c = G.conditions(BASE + body, owner, request(ours))
        return c['own']['sleep_clause'], c['opp']['sleep_clause']

    def test_move_rest_and_yawn_sleep(self):
        self.assertEqual(self.clause([]), (False, False))
        self.assertEqual(self.clause(['|move|p1a: Breloom|Spore|p2a: Snorlax', '|-status|p2a: Snorlax|slp', '|turn|2']), (True, False))
        self.assertEqual(self.clause(['|move|p2a: Snorlax|Rest|p2a: Snorlax', '|-status|p2a: Snorlax|slp|[from] move: Rest',
            '|-heal|p2a: Snorlax|100/100 slp|[silent]', '|turn|2']), (False, False))   # Rest: its own, allied source
        self.assertEqual(self.clause(['|move|p2a: Snorlax|Yawn|p1a: Breloom', '|-start|p1a: Breloom|move: Yawn|[of] p2a: Snorlax', '|turn|2',
            '|-end|p1a: Breloom|move: Yawn|[silent]', '|-status|p1a: Breloom|slp|[from] move: Yawn', '|turn|3'],
            ours=[('Breloom', '100/100 slp', 'leftovers', 'effectspore')] + OURS[1:]), (False, True))   # Yawn counts (our request shows the sleep)

    def test_wake_faint_and_cure_end_it(self):
        slept = ['|move|p1a: Breloom|Spore|p2a: Snorlax', '|-status|p2a: Snorlax|slp', '|turn|2']
        self.assertEqual(self.clause(slept + ['|cant|p2a: Snorlax|slp', '|turn|3', '|-curestatus|p2a: Snorlax|slp|[msg]', '|turn|4']), (False, False))
        self.assertEqual(self.clause(slept + ['|-damage|p2a: Snorlax|0 fnt', '|faint|p2a: Snorlax', '|turn|3']), (False, False))
        self.assertEqual(self.clause(slept + ['|-cureteam|p2a: Snorlax|[from] move: HealBell', '|turn|3']), (False, False))
        self.assertEqual(self.clause(slept + [switch('p2', 'Gengar'), '|turn|3']), (True, False))   # a benched sleeper still binds it
        self.assertEqual(self.clause(slept + [switch('p2', 'Gengar'), '|turn|3', switch('p2', 'Snorlax', '80/100 slp'), '|turn|4']), (True, False))
        # our own sleeper (the request's condition) binds the opponent's sleep moves; Rest's would not
        rested = G.conditions(BASE + ['|-status|p1a: Breloom|slp|[from] move: Rest'], 0, request([('Breloom', '100/100 slp', '', 'effectspore')]))
        self.assertEqual((rested['opp']['sleep_clause'], rested['own']['members']['breloom']['sleep_source']), (False, 'rest'))
        hit = G.conditions(BASE + ['|-status|p1a: Breloom|slp'], 0, request([('Breloom', '100/100 slp', '', 'effectspore')]))
        self.assertTrue(hit['opp']['sleep_clause'])

    def test_a_rest_sleep_keeps_its_source_across_a_switch(self):
        """Review B2: a Rester that switches out and comes back asleep ('100/100 slp' on its switch or drag line) is still Rest's
        sleeper, on either seat; a sleeper first seen on a switch line counts."""
        for seat, owner, kind in (('p2', 0, 'switch'), ('p2', 0, 'drag'), ('p1', 1, 'switch'), ('p1', 1, 'drag'), ('p1', 0, 'switch'), ('p2', 1, 'drag')):
            with self.subTest(seat=seat, kind=kind):
                lines = BASE + [f'|move|{seat}a: Snorlax|Rest|{seat}a: Snorlax', f'|-status|{seat}a: Snorlax|slp|[from] move: Rest', '|turn|2',
                    f'|switch|{seat}a: Gengar|Gengar, L80|100/100', '|turn|3', f'|{kind}|{seat}a: Snorlax|Snorlax, L80|100/100 slp', '|turn|4']
                if seat == 'p1':lines = [x.replace('p1a: Breloom', 'p1a: Snorlax').replace('Breloom, L80', 'Snorlax, L80') for x in lines]
                c = G.conditions(lines, owner, None)
                rester = 'opp' if (seat == 'p2') == (owner == 0) else 'own'
                self.assertEqual(c[rester]['members']['snorlax']['sleep_source'], 'rest')
                self.assertFalse(c['opp' if rester == 'own' else 'own']['sleep_clause'])
        unknown = G.conditions([switch('p1', 'Breloom'), switch('p2', 'Snorlax', '100/100 slp')], 0, None)
        self.assertTrue(unknown['own']['sleep_clause'])   # no Rest seen: a non-allied source, conservatively

    def test_the_guard_reads_the_clause_for_sleep_moves_only(self):
        c = G.conditions(BASE + ['|-status|p2a: Snorlax|slp'], 0, request(OURS))
        self.assertEqual(G.guarded('spore', c['own'], 'breloom', other=c['opp']), ('C', 'status'))
        self.assertEqual(G.guarded('hypnosis', c['opp'], 'snorlax', other=c['own']), None)   # our side has no sleeper
        self.assertEqual(G.guarded('perishsong', c['own'], 'breloom'), ('A', 'volatile'))   # tier A: always
        self.assertIsNone(G.guarded('destinybond', c['own'], 'breloom'))   # retired: never guarded
        self.assertIsNone(G.guarded('irontail', c['own'], 'breloom'))   # tier B: never guarded


class UserConditionTests(unittest.TestCase):
    def test_white_herb(self):
        lines = [switch('p1', 'Snorlax'), switch('p2', 'Deoxys', details='Deoxys-Attack'), '|turn|1']
        c = G.conditions(lines, 0, request([('Snorlax', '300/300', 'leftovers', 'thickfat')]))
        self.assertEqual(G.guarded('psychoboost', c['opp'], 'deoxysattack'), ('C', 'self_stat_drop'))   # the pool: always White Herb
        gone = G.conditions(lines + ['|move|p2a: Deoxys|Psycho Boost|p1a: Snorlax', '|-unboost|p2a: Deoxys|spa|2',
            '|-enditem|p2a: Deoxys|White Herb', '|turn|2'], 0, request([('Snorlax', '300/300', 'leftovers', 'thickfat')]))
        self.assertIsNone(G.guarded('psychoboost', gone['opp'], 'deoxysattack'))   # consumed
        knocked = G.conditions(lines + ['|-enditem|p2a: Deoxys|White Herb|[from] move: Knock Off|[of] p1a: Snorlax'], 0, None)
        self.assertIsNone(G.guarded('psychoboost', knocked['opp'], 'deoxysattack'))
        ours = G.conditions([switch('p1', 'Deoxys'), switch('p2', 'Snorlax')], 0, request([('Deoxys', '200/200', 'whiteherb', 'pressure')],
            moves=('psychoboost', 'superpower', 'shadowball', 'recover')))
        self.assertEqual(G.guarded('psychoboost', ours['own'], 'deoxys'), ('C', 'self_stat_drop'))
        used = G.conditions([switch('p1', 'Deoxys'), switch('p2', 'Snorlax')], 0, request([('Deoxys', '200/200', '', 'pressure')]))
        self.assertIsNone(G.guarded('psychoboost', used['own'], 'deoxys'))   # the request's item: consumed

    def test_refresh_facade_and_sacred_fire_statuses(self):
        ours = [('Ursaring', '200/200 brn', 'flameorb', 'guts'), ('Ho-Oh', '300/300 frz', 'leftovers', 'pressure'), ('Milotic', '250/250', 'leftovers', 'marvelscale')]
        c = G.conditions([switch('p1', 'Ursaring'), switch('p2', 'Swellow'), '|-status|p2a: Swellow|par', '|turn|2'], 0, request(ours))
        self.assertEqual(G.guarded('facade', c['own'], 'ursaring'), ('C', 'other'))   # Guts (the request) and a burn
        self.assertEqual(G.guarded('facade', c['opp'], 'swellow'), ('C', 'other'))   # every pool Facade user has Guts
        self.assertEqual(G.guarded('refresh', c['opp'], 'swellow'), ('C', 'status'))
        self.assertEqual(G.guarded('sacredfire', c['own'], 'hooh'), ('C', 'status'))   # frozen
        self.assertIsNone(G.guarded('refresh', c['own'], 'milotic'));self.assertIsNone(G.guarded('sacredfire', c['own'], 'ursaring'))
        noguts = G.conditions([switch('p1', 'Ursaring'), switch('p2', 'Swellow')], 0, request([('Ursaring', '200/200 brn', '', 'quickfeet')]))
        self.assertIsNone(G.guarded('facade', noguts['own'], 'ursaring'))
        cured = G.conditions([switch('p1', 'Ursaring'), switch('p2', 'Swellow'), '|-status|p2a: Swellow|par', '|-curestatus|p2a: Swellow|par|[msg]'], 0,
            request(ours))
        self.assertIsNone(G.guarded('refresh', cured['opp'], 'swellow'))
        slept = G.conditions([switch('p1', 'Ursaring'), switch('p2', 'Swellow'), '|-status|p2a: Swellow|slp'], 0, request(ours))
        self.assertIsNone(G.guarded('facade', slept['opp'], 'swellow'))   # Facade does not double while asleep

    def test_an_unrevealed_opponent_at_a_child_takes_the_pool_facts(self):
        """Review B1: a world's opponent member the root never saw (sampled) takes the pool's White Herb (Deoxys formes) and
        Guts (Facade users) at its turn-2 child, as a revealed one does at the root; ours never take them."""
        c = G.conditions(BASE, 0, request(OURS))
        herb = G.child_side(c['opp'], [('deoxysattack', 0, True, False)], opponent=True)
        self.assertEqual(G.guarded('psychoboost', herb, 'deoxysattack'), ('C', 'self_stat_drop'))
        guts = G.child_side(c['opp'], [('ursaring', 2, True, False)], opponent=True)   # burned at turn 1 in this child
        self.assertEqual(G.guarded('facade', guts, 'ursaring'), ('C', 'other'))
        self.assertIsNone(G.guarded('facade', G.child_side(c['opp'], [('ursaring', 0, True, False)], opponent=True), 'ursaring'))
        ours = G.child_side(c['own'], [('deoxys', 0, True, False), ('ursaring', 2, True, False)])
        self.assertEqual((G.guarded('psychoboost', ours, 'deoxys'), G.guarded('facade', ours, 'ursaring')), (None, None))

    def test_aromatherapy_reads_the_party(self):
        c = G.conditions([switch('p1', 'Meganium'), switch('p2', 'Vileplume'), '|-status|p2a: Vileplume|psn', switch('p2', 'Bellossom')], 0,
            request([('Meganium', '200/200', 'leftovers', 'overgrow'), ('Snorlax', '300/300', 'leftovers', 'thickfat')]))
        self.assertEqual((G.guarded('aromatherapy', c['opp'], 'bellossom'), G.guarded('aromatherapy', c['own'], 'meganium')), (('C', 'status'), None))
        mine = G.conditions([switch('p1', 'Meganium')], 0, request([('Meganium', '200/200', '', 'overgrow'), ('Snorlax', '300/300 par', '', 'thickfat')]))
        self.assertEqual(G.guarded('aromatherapy', mine['own'], 'meganium'), ('C', 'status'))
        fainted = G.conditions([switch('p1', 'Meganium')], 0, request([('Meganium', '200/200', '', 'overgrow'), ('Snorlax', '0 fnt', '', 'thickfat')]))
        self.assertIsNone(G.guarded('aromatherapy', fainted['own'], 'meganium'))


def prior(mass_on, mass, legal=None):
    """Logits giving lane `mass_on` probability `mass` among the legal lanes (the rest share the remainder equally)."""
    import math
    legal = legal or [True] * 4 + [False] * 5;others = sum(legal) - 1
    rest = (1 - mass) / others
    return [math.log(mass) if a == mass_on else math.log(rest) if legal[a] else 0. for a in range(9)], legal


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.cond = G.conditions(BASE, 0, request(OURS))
        self.own_lanes = ['spore', 'gigadrain', 'leechseed', 'substitute']   # Giga Drain is tier A

    def record(self, *, own=(1, .01), peer=(), turn2=(), available=True, choice=None):
        logits, legal = prior(*own)
        return G.evaluate(self.cond, own_turn1=dict(lanes=self.own_lanes, logits=logits, legal=legal), peer_turn1=list(peer), turn2=list(turn2),
            turn2_available=available, choice=choice)

    def peer(self, world, lanes, on, mass, requested=True):
        logits, legal = prior(on, mass);return dict(world=world, lanes=lanes, logits=logits, legal=legal, requested=requested, user='snorlax')

    def test_b_at_turn_1(self):
        r = self.record(own=(1, .2))
        self.assertEqual((r['tripped'], r['b'], r['b_turn1'], r['b_turn2'], r['c']), (True, True, True, False, False))
        self.assertEqual([(x['move'], x['side'], x['source'], x['tier'], round(x['max_mass'], 6)) for x in r['reasons']], [('gigadrain', 'own', 'turn1', 'A', .2)])
        low = self.record(own=(1, .04999))   # below 5%
        self.assertEqual((low['tripped'], low['a'], low['a_moves']), (False, True, ['gigadrain']))   # (a): legal, reported only
        peer = self.record(peer=[self.peer(0, ['bodyslam', 'curse', 'rest', 'sleeptalk'], 1, .3), self.peer(1, ['bodyslam', 'curse', 'rest', 'sleeptalk'], 1, .02)])
        self.assertEqual([(x['move'], x['side'], x['worlds']) for x in peer['reasons']], [('curse', 'opp', [0])])
        waiting = self.record(peer=[self.peer(0, ['curse', None, None, None], 0, .9, requested=False)])
        self.assertFalse(waiting['b'])   # a waiting opponent has no prior
        tier_b = self.record(peer=[self.peer(3, ['irontail', 'earthquake', 'rest', 'protect'], 0, .9)])
        self.assertEqual((tier_b['b'], tier_b['tier_b_legal'], tier_b['tier_b_moves']), (False, True, ['irontail']))
        masses = [m for m in tier_b['masses']['turn1'] if m['side'] == 'opp']
        self.assertEqual([(m['move'], m['world'], round(m['mass'], 6)) for m in masses], [('irontail', 3, .9)])

    def two_worlds(self):
        """World 0: the opponent requested, its reply prior .5 on lane 0; world 1: it waits (reply 0). p_w = 1/2."""
        return [self.peer(0, ['bodyslam', 'rest', 'sleeptalk', 'earthquake'], 0, .5), dict(world=1, lanes=[None] * 4, logits=[0.] * 9,
            legal=[False] * 9, requested=False, user='snorlax')]

    def test_b_at_turn_2_is_the_reach_over_each_childs_conditions(self):
        """Live rule: turn 2 trips on M_s >= 5%, each entry guarded under its own child's conditions (Sleep Clause where the turn-1
        Spore landed; a Rest sleep never binds); a heavy guarded mass on a cell the priors barely reach does not trip any more,
        though the any-cell arm (kept for analysis) still does."""
        child = lambda members_opp: dict(own_side=G.child_side(self.cond['own'], [('breloom', 0, True, False)]),
            opp_side=G.child_side(self.cond['opp'], members_opp, opponent=True))
        logits, legal = prior(0, .5)
        cell = lambda a, b, members: dict(world=0, cell=a * 9 + b, side='own', lanes=['spore', 'machpunch', 'leechseed', 'substitute'], logits=logits,
            legal=legal, user='breloom', **child(members))
        run = lambda *cells: self.record(own=(0, .97), peer=self.two_worlds(), turn2=list(cells))
        asleep = run(cell(0, 0, [('snorlax', 5, True, False)]))   # our Spore at (lane 0, reply 0) where the opponent now sleeps
        m = .5 * .97 * .5 * .5   # p_w x pi_own(0) x pi_peer,0(0) x q
        self.assertAlmostEqual(asleep['m_own'], m, places=15);self.assertEqual(asleep['m_opp'], 0.)
        self.assertEqual((asleep['b'], asleep['b_turn1'], asleep['b_turn2'], asleep['arms'], asleep['tripped']), (True, False, True, ['reach'], True))
        self.assertEqual((asleep['masses']['turn2'][0]['guarded'], asleep['any_cell_turn2']), (True, True))
        reach, = asleep['reasons']   # only what tripped: the reach on our side, with its tiers, moves and worlds
        self.assertEqual((reach['source'], reach['side'], list(reach['tiers']), reach['worlds'], asleep['tiers']), ('reach', 'own', ['C'], [0], ['C']))
        self.assertAlmostEqual(reach['tiers']['C'], m, places=15)
        self.assertAlmostEqual(reach['max_mass'], m, places=15);self.assertAlmostEqual(reach['moves']['spore'], m, places=15)
        self.assertEqual([(x['source'], x['move']) for x in asleep['any_cell_reasons']], [('turn2', 'spore')])
        for name, members in (('awake', [('snorlax', 0, True, False)]), ('rested', [('snorlax', 5, True, True)]), ('fainted', [('snorlax', 5, False, False)])):
            with self.subTest(name):
                r = run(cell(0, 0, members))
                self.assertEqual((r['m_own'], r['b'], r['any_cell_b'], r['masses']['turn2'][0]['guarded']), (0., False, False, False))
        faint = run(cell(3, 0, [('snorlax', 5, True, False)]))   # the same guarded mass where we play lane 3 (prior .01)
        self.assertAlmostEqual(faint['m_own'], .5 * .01 * .5 * .5, places=15)
        self.assertEqual((faint['b'], faint['tripped'], faint['any_cell_turn2'], faint['any_cell_b'], faint['any_cell_tripped']), (False, False, True, True, True))
        self.assertEqual((faint['reasons'], faint['tiers'], len(faint['any_cell_reasons'])), ([], [], 1))   # nothing tripped: no reason
        waiting = self.record(own=(0, .97), peer=self.two_worlds(), turn2=[dict(cell(0, 0, [('snorlax', 5, True, False)]), world=1)])
        self.assertAlmostEqual(waiting['m_own'], .5 * .97 * 1. * .5, places=15)   # the waiting world: reply 0 at mass 1
        missing = self.record(available=False, peer=self.two_worlds())
        self.assertEqual((missing['tripped'], missing['b_turn2'], missing['arms'], missing['m_own'], [x['source'] for x in missing['reasons']]),
            (True, True, ['unavailable'], None, ['turn2_unavailable']))
        self.assertEqual([x['source'] for x in missing['any_cell_reasons']], ['turn2_unavailable'])

    def test_the_live_m_is_the_exact_reach_of_its_own_record(self):
        """Agreement with #1873's exact path: re-running reach on the record (as the RR scorer does from a row: the root prior and
        each world's logits and kept lanes) gives the live m_own and m_opp bit for bit."""
        child = dict(own_side=G.child_side(self.cond['own'], [('breloom', 0, True, False)]),
            opp_side=G.child_side(self.cond['opp'], [('snorlax', 5, True, False)], opponent=True))
        cells = [dict(world=w, cell=a * 9 + b, side=side, lanes=lanes, logits=prior(k, q)[0], legal=prior(k, q)[1], user=user, **child)
                 for w, a, b, side, lanes, user, k, q in ((0, 0, 0, 'own', ['spore', 'gigadrain', None, None], 'breloom', 0, .3),
                     (0, 1, 2, 'opp', ['curse', 'rest', 'facade', None], 'snorlax', 1, .7), (1, 2, 0, 'own', ['substitute', 'gigadrain', 'spore', None], 'breloom', 2, .6),
                     (1, 0, 0, 'opp', ['perishsong', None, None, None], 'snorlax', 0, .9))]
        own = prior(0, .4);record = G.evaluate(self.cond, own_turn1=dict(lanes=self.own_lanes, logits=own[0], legal=own[1]), peer_turn1=self.two_worlds(),
            turn2=cells)
        worlds = [dict(requested=w['requested'], logits=w['logits'], kept=w['legal']) for w in self.two_worlds()]
        again = G.reach(json.loads(json.dumps(record)), own[0], own[1], worlds)
        self.assertTrue(again['exact']);self.assertEqual((again['m_own'], again['m_opp'], again['tripped']), (record['m_own'], record['m_opp'], record['b_turn2']))
        self.assertGreater(record['m_own'], 0);self.assertGreater(record['m_opp'], 0)

    def test_a_reach_trip_lists_every_contributing_tier(self):
        """Neither tier crosses 5% alone, their sum does: the reach reason lists both with their own masses."""
        child = dict(own_side=G.child_side(self.cond['own'], [('breloom', 0, True, False)]),
            opp_side=G.child_side(self.cond['opp'], [('snorlax', 5, True, False)], opponent=True))
        logits, legal = prior(0, .5)
        cell = dict(world=0, cell=0, side='own', lanes=['spore', 'gigadrain', None, None], logits=logits, legal=legal, user='breloom', **child)
        r = self.record(own=(0, .97), peer=self.two_worlds(), turn2=[cell])   # spore .5 (C), gigadrain 1/6 (A) at (0, 0)
        reach, = r['reasons'];w = .5 * .97 * .5
        self.assertEqual((reach['source'], sorted(reach['tiers']), r['tiers']), ('reach', ['A', 'C'], ['A', 'C']))
        self.assertAlmostEqual(reach['tiers']['C'], w * .5, places=15);self.assertAlmostEqual(reach['tiers']['A'], w / 6, places=15)
        self.assertAlmostEqual(r['m_own'], w * .5 + w / 6, places=15)

    def test_tier_c_child_conditions_in_the_reach(self):
        """Frozen (Sacred Fire), statused (Refresh; Facade with Guts) and a consumed White Herb (Psycho Boost), each decided at its child."""
        lines = BASE[:-1] + [switch('p2', 'Deoxys', details='Deoxys-Attack'), '|-enditem|p2a: Deoxys|White Herb', switch('p2', 'Snorlax'), '|turn|2']
        cond = G.conditions(lines, 0, request([('Ho-Oh', '300/300', 'leftovers', 'pressure'), ('Ursaring', '200/200', '', 'guts')],
            moves=('sacredfire', 'refresh', 'facade', 'recover')))
        logits, legal = prior(0, .5)
        def entry_at(side, lanes, user, own_members, opp_members):
            return dict(world=0, cell=0, side=side, lanes=lanes, logits=logits, legal=legal, user=user,
                own_side=G.child_side(cond['own'], own_members), opp_side=G.child_side(cond['opp'], opp_members, opponent=True))
        def run(*cells):
            return G.evaluate(cond, own_turn1=dict(lanes=[None] * 4, logits=prior(0, .97)[0], legal=prior(0, .97)[1]), peer_turn1=self.two_worlds(),
                turn2=list(cells))
        cases = (('frozen Ho-Oh: Sacred Fire', 'own', ['sacredfire', None, None, None], 'hooh', [('hooh', 6, True, False)], True),
                 ('thawed', 'own', ['sacredfire', None, None, None], 'hooh', [('hooh', 0, True, False)], False),
                 ('paralysed: Refresh', 'own', ['refresh', None, None, None], 'hooh', [('hooh', 1, True, False)], True),
                 ('burned Guts user: Facade', 'own', ['facade', None, None, None], 'ursaring', [('ursaring', 2, True, False)], True),
                 ('asleep Guts user: Facade does not double', 'own', ['facade', None, None, None], 'ursaring', [('ursaring', 5, True, False)], False))
        for name, side, lanes, user, members, want in cases:
            with self.subTest(name):
                r = run(entry_at(side, lanes, user, members, [('snorlax', 0, True, False)]))
                self.assertEqual((r['masses']['turn2'][0]['guarded'], r['b_turn2']), (want, want))
        herb = run(entry_at('opp', ['psychoboost', None, None, None], 'deoxysattack', [('hooh', 0, True, False)], [('deoxysattack', 0, True, False)]))
        self.assertEqual((herb['masses']['turn2'][0]['guarded'], herb['m_opp'], herb['b_turn2']), (False, 0., False))   # consumed at the root

    def test_c_per_choice(self):
        r = self.record(choice=1)
        self.assertEqual((r['tripped'], r['b'], r['c']), (True, False, True))
        self.assertEqual([(x['move'], x['source']) for x in r['reasons']], [('gigadrain', 'choice')])
        self.assertEqual(r['own_guarded_lanes'], [None, ['A', 'other'], None, None])   # Spore: no clause at this root
        self.assertEqual([G.trips(r, a) for a in (0, 1, 4, None)], [dict(tripped=False, b=False, c=False), dict(tripped=True, b=False, c=True),
            dict(tripped=False, b=False, c=False), dict(tripped=False, b=False, c=False)])
        b = self.record(own=(1, .3), choice=0)
        self.assertEqual((b['c'], G.trips(b, 0)['tripped'], G.trips(b, 4)), (False, True, dict(tripped=True, b=True, c=False)))   # (b): every choice
        self.assertTrue(json.loads(json.dumps(r)) == r)   # JSON-safe as the wire carries it
        self.assertEqual(G.error_record('boom')['tripped'], True)

    def test_masses_and_lanes(self):
        p = G.masses([0., 0., 1., 0., 0., 0., 0., 0., 0.], [True, False, True] + [False] * 6)
        self.assertAlmostEqual(sum(p), 1.);self.assertEqual(p[1], 0.);self.assertGreater(p[2], p[0])
        self.assertEqual(G.masses([float('nan')] * 9, [True] * 9), [0.] * 9)
        self.assertEqual(G.lanes_of(['solarbeam', 'surf', None, None], 'solarbeam'), ['solarbeam', None, None, None])
        self.assertEqual(G.lanes_of(['surf', 'curse', None, None], None, [False] * 4), ['struggle', None, None, None])
        self.assertEqual(G.lanes_of(['surf', 'curse'], None), ['surf', 'curse', None, None])

    def test_the_actor_and_this_module_share_the_recharge_rule(self):
        self.assertIs(actor.RECHARGE_MOVES, G.RECHARGE_MOVES);self.assertIs(actor.own_moves, G.own_moves)
        self.assertIs(actor.recharge_guarded, G.recharge_guarded);self.assertEqual(actor.EFFECT_GUARD, 'UNREPRESENTED_EFFECT')
        self.assertTrue({'hyperbeam', 'gigadrain', 'aromatherapy'} <= G.WATCHED and not G.WATCHED & {'surf', 'earthquake'})
        self.assertEqual(sorted(k for k, v in G.GUARDED.items() if v[0] == 'A'), sorted(('hyperbeam', 'transform', 'yawn', 'wrap',
            'perishsong', 'trick', 'counter', 'mirrorcoat', 'curse', 'superpower', 'knockoff', 'charm', 'lovelykiss', 'bonemerang', 'gigadrain')))


# The record of a fixed input as a62fb20b's guard wrote it (before turn-2 mass entries gained guarded / tier / user): every
# existing field must stay exactly this.
GOLDEN = json.loads('{"a": true, "a_moves": ["curse", "gigadrain"], "b": true, "b_turn1": true, "b_turn2": true, "c": true, "choice": 1, "conditions": {"opp": {"active": "snorlax", "members": {"snorlax": {"fainted": false, "guts": true, "sleep_source": null, "status": "", "white_herb": false}}, "party_statused": false, "sleep_clause": false}, "own": {"active": "breloom", "members": {"breloom": {"fainted": false, "guts": false, "sleep_source": null, "status": "", "white_herb": false}, "snorlax": {"fainted": false, "guts": false, "sleep_source": null, "status": "", "white_herb": false}}, "party_statused": false, "sleep_clause": false}}, "masses": {"turn1": [{"mass": 0.26666666666666666, "move": "spore", "side": "own", "world": null}, {"mass": 0.20000000000000004, "move": "gigadrain", "side": "own", "world": null}, {"mass": 0.3, "move": "curse", "side": "opp", "world": 0}, {"mass": 0.23333333333333334, "move": "irontail", "side": "opp", "world": 0}, {"mass": 0.02, "move": "curse", "side": "opp", "world": 1}, {"mass": 0.32666666666666666, "move": "irontail", "side": "opp", "world": 1}], "turn2": [{"cell": 10, "mass": 0.5, "move": "spore", "side": "own", "world": 2}, {"cell": 10, "mass": 0.16666666666666669, "move": "gigadrain", "side": "own", "world": 2}, {"cell": 11, "mass": 0.5, "move": "curse", "side": "opp", "world": 3}, {"cell": 11, "mass": 0.16666666666666669, "move": "facade", "side": "opp", "world": 3}, {"cell": 11, "mass": 0.16666666666666669, "move": "irontail", "side": "opp", "world": 3}]}, "own_guarded_lanes": [null, ["A", "other"], null, null], "own_lanes": ["spore", "gigadrain", "leechseed", "substitute"], "reasons": [{"effect_class": "other", "max_mass": 1.0, "move": "gigadrain", "side": "own", "source": "choice", "tier": "A", "worlds": []}, {"effect_class": "self_boost", "max_mass": 0.3, "move": "curse", "side": "opp", "source": "turn1", "tier": "A", "worlds": [0]}, {"effect_class": "other", "max_mass": 0.20000000000000004, "move": "gigadrain", "side": "own", "source": "turn1", "tier": "A", "worlds": []}, {"effect_class": "self_boost", "max_mass": 0.5, "move": "curse", "side": "opp", "source": "turn2", "tier": "A", "worlds": [3]}, {"effect_class": "other", "max_mass": 0.16666666666666669, "move": "gigadrain", "side": "own", "source": "turn2", "tier": "A", "worlds": [2]}, {"effect_class": "status", "max_mass": 0.5, "move": "spore", "side": "own", "source": "turn2", "tier": "C", "worlds": [2]}], "schema": "pokezero.depthn-effect-guard.v1", "threshold": 0.05, "tier_b_legal": true, "tier_b_moves": ["irontail"], "tiers": ["A", "C"], "tripped": true, "turn2_available": true}')


def golden_inputs():
    cond = G.conditions(BASE, 0, request(OURS))
    own_logits, own_legal = prior(1, .2)
    peer = [dict(world=w, lanes=['bodyslam', 'curse', 'irontail', 'sleeptalk'], logits=prior(1, m)[0], legal=prior(1, m)[1], requested=True, user='snorlax')
            for w, m in ((0, .3), (1, .02))]
    child = lambda members: dict(own_side=G.child_side(cond['own'], [('breloom', 0, True, False)]), opp_side=G.child_side(cond['opp'], members, opponent=True))
    l2, g2 = prior(0, .5)
    turn2 = [dict(world=2, cell=10, side='own', lanes=['spore', 'machpunch', 'leechseed', 'gigadrain'], logits=l2, legal=g2, user='breloom', **child([('snorlax', 5, True, False)])),
             dict(world=3, cell=11, side='opp', lanes=['curse', 'facade', 'rest', 'irontail'], logits=l2, legal=g2, user='snorlax', **child([('snorlax', 0, True, False)]))]
    return cond, dict(lanes=['spore', 'gigadrain', 'leechseed', 'substitute'], logits=own_logits, legal=own_legal), peer, turn2


class GoldenTests(unittest.TestCase):
    def test_the_new_turn_2_fields_change_nothing_else(self):
        cond, own, peer, turn2 = golden_inputs()
        record = json.loads(json.dumps(G.evaluate(cond, own_turn1=own, peer_turn1=peer, turn2=turn2, choice=1)))
        added = [{k: e.pop(k) for k in ('guarded', 'tier', 'user')} for e in record['masses']['turn2']]
        live = {k: record.pop(k) for k in ('rule', 'm_own', 'm_opp', 'arms', 'reach', 'any_cell_turn2', 'any_cell_b', 'any_cell_tripped', 'any_cell_reasons')}
        # reasons and tiers list only what tripped (turn 1, the choice; no reach here); the any-cell turn-2 hits moved to any_cell_reasons
        turn2 = [r for r in GOLDEN['reasons'] if r['source'] == 'turn2']
        self.assertEqual((live['any_cell_reasons'], record.pop('reasons'), record.pop('tiers')),
            (turn2, [r for r in GOLDEN['reasons'] if r['source'] != 'turn2'], ['A']))
        # The live turn-2 arm is the reach now (its cells' worlds 2 and 3 have no turn-1 prior here: M 0); the any-cell arm the
        # golden record decided with is kept under any_cell_*, and b / tripped hold through turn 1 (Giga Drain at 20%).
        self.assertEqual((live['any_cell_turn2'], live['any_cell_b'], live['any_cell_tripped']), (GOLDEN['b_turn2'], GOLDEN['b'], GOLDEN['tripped']))
        self.assertEqual((record.pop('b_turn2'), live['m_own'], live['m_opp'], live['arms'], live['rule']), (False, 0., 0., ['turn1', 'choice'], 'reach'))
        self.assertEqual(record, {k: v for k, v in GOLDEN.items() if k not in ('b_turn2', 'reasons', 'tiers')})   # every other field unchanged
        # spore (Sleep Clause binds: the child's opponent sleeps) and gigadrain (A); curse (A), facade (C, unstatused), irontail (B)
        self.assertEqual(added, [dict(guarded=True, tier='C', user='breloom'), dict(guarded=True, tier='A', user='breloom'),
            dict(guarded=True, tier='A', user='snorlax'), dict(guarded=False, tier='C', user='snorlax'), dict(guarded=False, tier='B', user='snorlax')])
        self.assertTrue(all('guarded' not in e for e in record['masses']['turn1']))


def reach_record(entries, *, opp_members=None, own_members=None, available=True):
    return dict(masses=dict(turn1=[], turn2=entries), turn2_available=available,
                conditions=dict(own=dict(members=own_members or {}), opp=dict(members=opp_members or {})))


def entry(world, a, b, side, move, mass, guarded=None):
    out = dict(side=side, world=world, move=move, mass=mass, cell=a * 9 + b)
    if guarded is not None:out.update(guarded=guarded, tier=G.tier_of(move), user='x')
    return out


class ReachTests(unittest.TestCase):
    """M_s = sum_w p_w sum_(a,b) pi_own(a) pi_peer,w(b) q_(w,ab),s on two worlds: our prior 3/4, 1/4 on lanes 0, 1; world 0's
    reply prior 1/4, 3/4 on lanes 0, 1; world 1's opponent waits (reply 0 at mass 1); p_w = 1/2."""
    root_logits = [math.log(3), 0.] + [0.] * 7
    root_legal = [True, True] + [False] * 7
    worlds = [dict(requested=True, logits=[0., math.log(3)] + [0.] * 7, kept=[True, True] + [False] * 7),
              dict(requested=False, logits=[0.] * 9, kept=[False] * 9)]

    def entries(self, flags):
        g = (lambda x: x) if flags else (lambda x: None)
        return [entry(0, 0, 1, 'opp', 'perishsong', .4, g(True)),   # tier A
                entry(0, 1, 0, 'own', 'spore', .2, g(False)),   # tier C, its condition false at that child
                entry(0, 0, 1, 'own', 'spore', .3, g(True)),   # tier C, true there
                entry(1, 1, 0, 'opp', 'facade', .5, g(True)),   # the waiting world's cell (a=1, b=0)
                entry(1, 0, 1, 'opp', 'curse', .9, g(True)),   # a reply the waiting world never has: weight 0
                entry(0, 0, 0, 'opp', 'irontail', .6, g(False))]   # tier B: never counted

    def reach(self, entries, **kw):
        return G.reach(reach_record(entries, **kw), self.root_logits, self.root_legal, self.worlds)

    def test_the_exact_path_by_hand(self):
        r = self.reach(self.entries(True))
        m_opp = .5 * (.75 * .75 * .4) + .5 * (.25 * 1. * .5);m_own = .5 * (.75 * .75 * .3)
        self.assertTrue(r['exact']);self.assertAlmostEqual(r['m_opp'], m_opp, places=15);self.assertAlmostEqual(r['m_own'], m_own, places=15)
        self.assertEqual((r['m_opp_lower'], r['m_own_lower']), (r['m_opp_upper'], r['m_own_upper']))
        self.assertEqual((r['tripped'], r['tripped_lower'], r['tripped_upper']), (True, True, True))
        self.assertAlmostEqual(r['tiers']['lower']['opp']['A'], .5 * .75 * .75 * .4, places=15)
        self.assertAlmostEqual(r['tiers']['lower']['opp']['C'], .5 * .25 * .5, places=15)

    def test_the_bounds_by_hand(self):
        r = self.reach(self.entries(False))
        self.assertFalse(r['exact']);self.assertIsNone(r['m_opp'])
        self.assertAlmostEqual(r['m_opp_lower'], .5 * .75 * .75 * .4, places=15)   # tier A only
        self.assertAlmostEqual(r['m_opp_upper'], .5 * .75 * .75 * .4 + .5 * .25 * .5, places=15)   # every tier-C mass too
        self.assertEqual(r['m_own_lower'], 0.)
        self.assertAlmostEqual(r['m_own_upper'], .5 * (.25 * .25 * .2 + .75 * .75 * .3), places=15)
        self.assertEqual(r['tripped'], True)   # the lower bound already trips

    def test_an_ambiguous_row_and_a_clear_one(self):
        straddle = [e for e in self.entries(False) if e['move'] != 'perishsong']   # lower 0, upper above the threshold
        r = self.reach(straddle)
        self.assertEqual((r['tripped'], r['tripped_lower'], r['tripped_upper']), (None, False, True))
        clear = self.reach([entry(0, 0, 1, 'opp', 'spore', .01), entry(1, 1, 0, 'own', 'hyperbeam', .001)])
        self.assertEqual((clear['tripped'], clear['tripped_lower'], clear['tripped_upper']), (False, False, False))

    def test_cells_and_the_waiting_world(self):
        # a = our lane (cell // 9), b = the reply (cell % 9): (0, 1) weighs 3/4 x 3/4, (1, 0) weighs 1/4 x 1/4 in world 0
        one = lambda a, b, w=0: self.reach([entry(w, a, b, 'opp', 'curse', 1.)])['m_opp_lower']
        self.assertAlmostEqual(one(0, 1), .5 * .75 * .75, places=15);self.assertAlmostEqual(one(1, 0), .5 * .25 * .25, places=15)
        self.assertAlmostEqual(one(1, 0, 1), .5 * .25, places=15)   # the waiting world: reply 0 at mass 1
        self.assertEqual(one(1, 1, 1), 0.)   # and no other reply

    def test_psycho_boost_counts_in_the_lower_bound_only_under_a_revealed_white_herb_holder(self):
        pb = [entry(0, 0, 1, 'opp', 'psychoboost', .2)]
        none = self.reach(pb)   # no Deoxys form revealed: it may be Deoxys-Speed (no White Herb), so upper only
        self.assertEqual(none['m_opp_lower'], 0.);self.assertAlmostEqual(none['m_opp_upper'], .5 * .75 * .75 * .2, places=15)
        held = self.reach(pb, opp_members={'deoxysattack': dict(white_herb=True)})
        self.assertAlmostEqual(held['m_opp_lower'], .5 * .75 * .75 * .2, places=15)
        gone = self.reach(pb, opp_members={'deoxysattack': dict(white_herb=False)})
        self.assertEqual((gone['m_opp_lower'], gone['m_opp_upper'] > 0), (0., True))   # consumed: upper only
        ours = self.reach([entry(0, 0, 1, 'own', 'psychoboost', .2)], own_members={'deoxys': dict(white_herb=True)})
        self.assertGreater(ours['m_own_lower'], 0)
        self.assertEqual(self.reach([entry(0, 0, 1, 'own', 'psychoboost', .2)])['m_own_lower'], 0.)   # ours without a Deoxys: no herb

    def test_a_revealed_deoxys_speed_keeps_lower_at_most_exact(self):
        """The review's repro: a revealed opponent Deoxys-Speed (Psycho Boost, no White Herb) with 0.6 of its turn-2 prior on it."""
        cond = G.conditions([switch('p1', 'Snorlax'), switch('p2', 'Deoxys', details='Deoxys-Speed'), '|turn|1'], 0, None)
        hit = G.guarded('psychoboost', G.child_side(cond['opp'], [('deoxysspeed', 0, True, False)], opponent=True), 'deoxysspeed')
        self.assertIsNone(hit)   # the guard itself: not guarded
        e = entry(0, 0, 1, 'opp', 'psychoboost', .6);flagged = dict(e, guarded=hit is not None, tier='C', user='deoxysspeed')
        exact = G.reach(dict(reach_record([flagged]), conditions=cond), self.root_logits, self.root_legal, self.worlds)
        bounds = G.reach(dict(reach_record([e]), conditions=cond), self.root_logits, self.root_legal, self.worlds)
        self.assertEqual((exact['m_opp'], exact['tripped']), (0., False))
        self.assertEqual(bounds['m_opp_lower'], 0.);self.assertAlmostEqual(bounds['m_opp_upper'], .5 * .75 * .75 * .6, places=15)
        self.assertEqual(bounds['tripped'], None)   # the bounds straddle 5%: ambiguous, never a false trip

    def test_lower_at_most_exact_at_most_upper_on_random_records(self):
        """Randomized records whose turn-2 entries carry the guard's own flags (G.guarded on each child's sides, as evaluate decides):
        the bounds of the same record without the flags bracket its exact M, both sides."""
        import random
        rng = random.Random(20261109);moves = sorted(G.WATCHED);status = (0, 0, 0, 1, 2, 3, 4, 5, 6)
        opp_pool = ['snorlax', 'ursaring', 'swellow', 'gengar', 'deoxys', 'deoxysattack', 'deoxysspeed', 'bellossom']
        for trial in range(300):
            deoxys = rng.choice(['deoxys', 'deoxysattack', 'deoxysspeed', None])   # the opponent's one Deoxys form, if any
            revealed = rng.sample([x for x in opp_pool if not x.startswith('deoxys')], 2) + ([deoxys] if deoxys and rng.random() < .5 else [])
            lines = [switch('p1', 'Breloom')] + [switch('p2', x.title(), details=x) for x in revealed]
            if deoxys in revealed and rng.random() < .3:lines.append(f'|-enditem|p2a: {deoxys.title()}|White Herb')
            ours = [('Breloom', rng.choice(['100/100', '100/100 par', '50/100 brn']), 'leftovers', 'effectspore'),
                    ('Deoxys', '100/100', rng.choice(['whiteherb', '']), 'pressure'), ('Ursaring', rng.choice(['100/100', '100/100 psn']), '', rng.choice(['guts', 'quickfeet']))]
            cond = G.conditions(lines, 0, request(ours))
            own_species = ['breloom', 'deoxys', 'ursaring'];opp_species = [x for x in opp_pool if not x.startswith('deoxys')] + ([deoxys] if deoxys else [])
            entries = []
            for _ in range(rng.randint(1, 8)):
                side = rng.choice(['own', 'opp']);move = rng.choice(moves)
                users = own_species if side == 'own' else opp_species
                user = (('deoxys' if side == 'own' else deoxys) if move == 'psychoboost' else rng.choice(users))
                if user is None:continue   # no Deoxys form on that side: nobody knows Psycho Boost
                children = {s: G.child_side(cond[s], [(x, rng.choice(status), rng.random() < .9, rng.random() < .3) for x in (own_species if s == 'own' else opp_species)],
                                            opponent=s == 'opp') for s in ('own', 'opp')}
                hit = G.guarded(move, children[side], user, other=children['opp' if side == 'own' else 'own'])
                e = entry(rng.randint(0, 1), rng.randint(0, 1), rng.randint(0, 1), side, move, rng.random())
                entries.append(dict(e, guarded=hit is not None, tier=G.tier_of(move), user=user))
            exact = G.reach(dict(reach_record(entries), conditions=cond), self.root_logits, self.root_legal, self.worlds)
            bounds = G.reach(dict(reach_record([{k: v for k, v in e.items() if k not in ('guarded', 'tier', 'user')} for e in entries]), conditions=cond),
                             self.root_logits, self.root_legal, self.worlds)
            for side in ('own', 'opp'):
                with self.subTest(trial=trial, side=side):
                    self.assertLessEqual(bounds[f'm_{side}_lower'], exact[f'm_{side}'] + 1e-12)
                    self.assertLessEqual(exact[f'm_{side}'], bounds[f'm_{side}_upper'] + 1e-12)
            if bounds['tripped'] is not None:self.assertEqual(bounds['tripped'], exact['tripped'])   # the bounds never decide wrong

    def test_a_record_without_turn_2_masses_fails_closed(self):
        for record in (G.error_record('x'), reach_record([], available=False), None):
            r = G.reach(record, self.root_logits, self.root_legal, self.worlds)
            self.assertEqual((r['tripped'], r['tripped_lower'], r['tripped_upper'], r['unavailable'], r['m_own_lower']), (True, True, True, True, None))


if __name__ == '__main__':
    unittest.main()


class RetiredTests(unittest.TestCase):
    """RETIRED (destinybond since A2's merge 286a5ffe): watched and recorded, never guarded, never in the reach; a record written
    while it was tier A keeps its own M."""
    def test_destiny_bond_is_watched_never_guarded(self):
        self.assertEqual(G.RETIRED, {'destinybond': ('volatile', '286a5ffe')})
        self.assertNotIn('destinybond', G.GUARDED);self.assertIn('destinybond', G.WATCHED);self.assertEqual(G.tier_of('destinybond'), 'R')
        self.assertFalse(set(G.RETIRED) & (set(G.GUARDED) | set(G.TIER_B)))

    def test_a_retired_lane_is_recorded_and_never_trips(self):
        cond = G.conditions(BASE, 0, request(OURS, moves=('destinybond', 'leechseed', 'substitute', 'protect')))
        lanes = ['destinybond', 'leechseed', 'substitute', 'protect'];own = prior(0, .97)
        child = dict(own_side=G.child_side(cond['own'], [('breloom', 0, True, False)]),
            opp_side=G.child_side(cond['opp'], [('snorlax', 0, True, False)], opponent=True))
        logits, legal = prior(0, .9)
        peer = [dict(world=0, lanes=['destinybond', 'rest', 'sleeptalk', 'earthquake'], logits=logits, legal=legal, requested=True, user='snorlax')]
        turn2 = [dict(world=0, cell=0, side=side, lanes=['destinybond', 'rest', None, None], logits=logits, legal=legal, user=user, **child)
                 for side, user in (('own', 'breloom'), ('opp', 'snorlax'))]
        r = G.evaluate(cond, own_turn1=dict(lanes=lanes, logits=own[0], legal=own[1]), peer_turn1=peer, turn2=turn2, choice=0)
        self.assertEqual((r['tripped'], r['b'], r['c'], r['a'], r['m_own'], r['m_opp'], r['reasons']), (False, False, False, False, 0., 0., []))
        self.assertEqual(r['own_guarded_lanes'], [None] * 4)
        self.assertEqual(sorted((m['side'], m['move']) for m in r['masses']['turn1']), [('opp', 'destinybond'), ('own', 'destinybond')])
        self.assertEqual([(m['side'], m['guarded'], m['tier']) for m in r['masses']['turn2']], [('own', False, 'R'), ('opp', False, 'R')])
        self.assertEqual(G.trips(r, 0), dict(tripped=False, b=False, c=False))

    def test_an_exact_record_from_before_the_shrink_keeps_its_m(self):
        root_logits, root_legal = ReachTests.root_logits, ReachTests.root_legal
        before = dict(entry(0, 0, 1, 'opp', 'destinybond', .4), guarded=True, tier='A', user='x')   # as the tier-A guard wrote it
        r = G.reach(reach_record([before]), root_logits, root_legal, ReachTests.worlds)
        self.assertTrue(r['exact']);self.assertAlmostEqual(r['m_opp'], .5 * .75 * .75 * .4, places=15);self.assertTrue(r['tripped'])
        now = G.reach(reach_record([entry(0, 0, 1, 'opp', 'destinybond', .4, False)]), root_logits, root_legal, ReachTests.worlds)
        self.assertEqual((now['m_opp'], now['tripped']), (0., False))
        old = G.reach(reach_record([entry(0, 0, 1, 'opp', 'destinybond', .4)]), root_logits, root_legal, ReachTests.worlds)
        self.assertEqual((old['exact'], old['m_opp_lower'], old['m_opp_upper']), (False, 0., 0.))   # a pre-`guarded` record: this module's tiers
