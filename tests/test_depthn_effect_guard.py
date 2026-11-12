"""pokezero.depthn_effect_guard on exact Gen 3 protocol lines: the tier-C conditions, (b) at turn 1 and turn 2, (c) per choice."""
import contextlib
import json
import math
import unittest
from unittest import mock

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
        self.assertEqual((c['opp']['members']['swellow']['status'], G.guarded('refresh', c['opp'], 'swellow')), ('par', None))   # Refresh: retired (A1)
        self.assertEqual(G.guarded('sacredfire', c['own'], 'hooh'), ('C', 'status'))   # frozen
        self.assertIsNone(G.guarded('sacredfire', c['own'], 'ursaring'))
        noguts = G.conditions([switch('p1', 'Ursaring'), switch('p2', 'Swellow')], 0, request([('Ursaring', '200/200 brn', '', 'quickfeet')]))
        self.assertIsNone(G.guarded('facade', noguts['own'], 'ursaring'))
        cured = G.conditions([switch('p1', 'Ursaring'), switch('p2', 'Swellow'), '|-status|p2a: Swellow|par', '|-curestatus|p2a: Swellow|par|[msg]'], 0,
            request(ours))
        self.assertEqual(cured['opp']['members']['swellow']['status'], '')   # cured
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
        self.assertEqual((c['opp']['party_statused'], c['own']['party_statused']), (True, False))   # the facts stay recorded
        mine = G.conditions([switch('p1', 'Meganium')], 0, request([('Meganium', '200/200', '', 'overgrow'), ('Snorlax', '300/300 par', '', 'thickfat')]))
        self.assertTrue(mine['own']['party_statused'])
        fainted = G.conditions([switch('p1', 'Meganium')], 0, request([('Meganium', '200/200', '', 'overgrow'), ('Snorlax', '0 fnt', '', 'thickfat')]))
        self.assertFalse(fainted['own']['party_statused'])
        self.assertEqual([G.guarded('aromatherapy', x['own'], 'meganium') for x in (c, mine)], [None, None])   # Aromatherapy: retired (A1)


def prior(mass_on, mass, legal=None):
    """Logits giving lane `mass_on` probability `mass` among the legal lanes (the rest share the remainder equally)."""
    import math
    legal = legal or [True] * 4 + [False] * 5;others = sum(legal) - 1
    rest = (1 - mass) / others
    return [math.log(mass) if a == mass_on else math.log(rest) if legal[a] else 0. for a in range(9)], legal


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.cond = G.conditions(BASE, 0, request(OURS))
        self.own_lanes = ['spore', 'bonemerang', 'leechseed', 'substitute']   # Bonemerang is tier A

    def record(self, *, own=(1, .01), peer=(), turn2=(), available=True, choice=None):
        logits, legal = prior(*own)
        return G.evaluate(self.cond, own_turn1=dict(lanes=self.own_lanes, logits=logits, legal=legal), peer_turn1=list(peer), turn2=list(turn2),
            turn2_available=available, choice=choice)

    def peer(self, world, lanes, on, mass, requested=True):
        logits, legal = prior(on, mass);return dict(world=world, lanes=lanes, logits=logits, legal=legal, requested=requested, user='snorlax')

    def test_b_at_turn_1(self):
        r = self.record(own=(1, .2))
        self.assertEqual((r['tripped'], r['b'], r['b_turn1'], r['b_turn2'], r['c']), (True, True, True, False, False))
        self.assertEqual([(x['move'], x['side'], x['source'], x['tier'], round(x['max_mass'], 6)) for x in r['reasons']], [('bonemerang', 'own', 'turn1', 'A', .2)])
        low = self.record(own=(1, .04999))   # below 5%
        self.assertEqual((low['tripped'], low['a'], low['a_moves']), (False, True, ['bonemerang']))   # (a): legal, reported only
        peer = self.record(peer=[self.peer(0, ['bodyslam', 'charm', 'rest', 'sleeptalk'], 1, .3), self.peer(1, ['bodyslam', 'charm', 'rest', 'sleeptalk'], 1, .02)])
        self.assertEqual([(x['move'], x['side'], x['worlds']) for x in peer['reasons']], [('charm', 'opp', [0])])
        waiting = self.record(peer=[self.peer(0, ['charm', None, None, None], 0, .9, requested=False)])
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
                 for w, a, b, side, lanes, user, k, q in ((0, 0, 0, 'own', ['spore', 'bonemerang', None, None], 'breloom', 0, .3),
                     (0, 1, 2, 'opp', ['charm', 'rest', 'facade', None], 'snorlax', 1, .7), (1, 2, 0, 'own', ['substitute', 'bonemerang', 'spore', None], 'breloom', 2, .6),
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
        cell = dict(world=0, cell=0, side='own', lanes=['spore', 'bonemerang', None, None], logits=logits, legal=legal, user='breloom', **child)
        r = self.record(own=(0, .97), peer=self.two_worlds(), turn2=[cell])   # spore .5 (C), bonemerang 1/6 (A) at (0, 0)
        reach, = r['reasons'];w = .5 * .97 * .5
        self.assertEqual((reach['source'], sorted(reach['tiers']), r['tiers']), ('reach', ['A', 'C'], ['A', 'C']))
        self.assertAlmostEqual(reach['tiers']['C'], w * .5, places=15);self.assertAlmostEqual(reach['tiers']['A'], w / 6, places=15)
        self.assertAlmostEqual(r['m_own'], w * .5 + w / 6, places=15)

    def test_tier_c_child_conditions_in_the_reach(self):
        """Frozen (Sacred Fire), statused (Facade with Guts; Refresh, retired) and a consumed White Herb (Psycho Boost), each decided at its child."""
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
                 ('paralysed: Refresh, retired (A1): recorded, never guarded', 'own', ['refresh', None, None, None], 'hooh', [('hooh', 1, True, False)], False),
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
        self.assertEqual([(x['move'], x['source']) for x in r['reasons']], [('bonemerang', 'choice')])
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
        self.assertEqual(sorted(k for k, v in G.GUARDED.items() if v[0] == 'A'), sorted(('transform', 'yawn', 'wrap',
            'perishsong', 'superpower', 'knockoff', 'charm', 'bonemerang')))
        self.assertEqual(sorted(k for k, v in G.GUARDED.items() if v[0] == 'C'), sorted(('spore', 'sleeppowder', 'hypnosis', 'sacredfire',
            'psychoboost', 'facade', 'trick')))
        self.assertEqual(G.GUARDED['trick'], ('C', 'item_removal', 'item_knocked'))


# The record of a fixed input as a62fb20b's guard wrote it (before turn-2 mass entries gained guarded / tier / user): every
# existing field must stay exactly this.
GOLDEN = json.loads('{"a": true, "a_moves": ["curse", "gigadrain"], "b": true, "b_turn1": true, "b_turn2": true, "c": true, "choice": 1, "conditions": {"opp": {"active": "snorlax", "members": {"snorlax": {"fainted": false, "guts": true, "sleep_source": null, "status": "", "white_herb": false}}, "party_statused": false, "sleep_clause": false}, "own": {"active": "breloom", "members": {"breloom": {"fainted": false, "guts": false, "sleep_source": null, "status": "", "white_herb": false}, "snorlax": {"fainted": false, "guts": false, "sleep_source": null, "status": "", "white_herb": false}}, "party_statused": false, "sleep_clause": false}}, "masses": {"turn1": [{"mass": 0.26666666666666666, "move": "spore", "side": "own", "world": null}, {"mass": 0.20000000000000004, "move": "gigadrain", "side": "own", "world": null}, {"mass": 0.3, "move": "curse", "side": "opp", "world": 0}, {"mass": 0.23333333333333334, "move": "irontail", "side": "opp", "world": 0}, {"mass": 0.02, "move": "curse", "side": "opp", "world": 1}, {"mass": 0.32666666666666666, "move": "irontail", "side": "opp", "world": 1}], "turn2": [{"cell": 10, "mass": 0.5, "move": "spore", "side": "own", "world": 2}, {"cell": 10, "mass": 0.16666666666666669, "move": "gigadrain", "side": "own", "world": 2}, {"cell": 11, "mass": 0.5, "move": "curse", "side": "opp", "world": 3}, {"cell": 11, "mass": 0.16666666666666669, "move": "facade", "side": "opp", "world": 3}, {"cell": 11, "mass": 0.16666666666666669, "move": "irontail", "side": "opp", "world": 3}]}, "own_guarded_lanes": [null, ["A", "other"], null, null], "own_lanes": ["spore", "gigadrain", "leechseed", "substitute"], "reasons": [{"effect_class": "other", "max_mass": 1.0, "move": "gigadrain", "side": "own", "source": "choice", "tier": "A", "worlds": []}, {"effect_class": "self_boost", "max_mass": 0.3, "move": "curse", "side": "opp", "source": "turn1", "tier": "A", "worlds": [0]}, {"effect_class": "other", "max_mass": 0.20000000000000004, "move": "gigadrain", "side": "own", "source": "turn1", "tier": "A", "worlds": []}, {"effect_class": "self_boost", "max_mass": 0.5, "move": "curse", "side": "opp", "source": "turn2", "tier": "A", "worlds": [3]}, {"effect_class": "other", "max_mass": 0.16666666666666669, "move": "gigadrain", "side": "own", "source": "turn2", "tier": "A", "worlds": [2]}, {"effect_class": "status", "max_mass": 0.5, "move": "spore", "side": "own", "source": "turn2", "tier": "C", "worlds": [2]}], "schema": "pokezero.depthn-effect-guard.v1", "threshold": 0.05, "tier_b_legal": true, "tier_b_moves": ["irontail"], "tiers": ["A", "C"], "tripped": true, "turn2_available": true}')


# The base guard's entries (8b212312 and before) for the moves retired since A1 / A2 / R: tests of a record written under that guard.
BASE_RETIRED = {'curse': ('A', 'self_boost', None), 'aromatherapy': ('C', 'status', 'party_statused'), 'refresh': ('C', 'status', 'user_brn_par_psn'),
                'lovelykiss': ('A', 'status', None), 'gigadrain': ('A', 'other', None), 'mirrorcoat': ('A', 'retaliation', None),
                'counter': ('A', 'retaliation', None)}


@contextlib.contextmanager
def base_tiers():
    with mock.patch.dict(G.GUARDED, BASE_RETIRED), mock.patch.object(G, 'RETIRED', {m: v for m, v in G.RETIRED.items() if m not in BASE_RETIRED}):
        yield


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
        with base_tiers():   # the golden record's own guard: Curse and Giga Drain tier A (retired since A1 / A2)
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
        # the Trick condition's fields (the M1 guard shrink) are additions: none knocked off here
        knocked = [(side.pop('item_knocked'), [m.pop('knocked') for m in side['members'].values()]) for side in record['conditions'].values()]
        self.assertEqual(knocked, [(False, [False] * len(side['members'])) for side in record['conditions'].values()])
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
        the bounds of the same record without the flags bracket its exact M, both sides. Moves whose tier changed since records
        carried no flags (the RETIRED ones, and Trick: tier C since M1) are left out: a record without the flags was written before
        that change."""
        import random
        rng = random.Random(20261109);moves = sorted(m for m in G.WATCHED if G.PRE_FLAG_TIERS.get(m, G.tier_of(m)) == G.tier_of(m))
        status = (0, 0, 0, 1, 2, 3, 4, 5, 6)
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


class RetiredTests(unittest.TestCase):
    """RETIRED (destinybond since A2's merge 286a5ffe, hyperbeam since M1's 8b212312): watched and recorded, never guarded, never in
    the reach; a record written while it was tier A keeps its own M (and so does one with Trick, tier A until M1)."""
    def test_destiny_bond_is_watched_never_guarded(self):
        self.assertEqual(G.RETIRED, {'destinybond': ('volatile', '286a5ffe'), 'hyperbeam': ('multi_turn', '8b212312'),
            'curse': ('self_boost', '4ee5d3ee'), 'aromatherapy': ('status', '4ee5d3ee'), 'refresh': ('status', '4ee5d3ee'),
            'lovelykiss': ('status', '4ee5d3ee'), 'gigadrain': ('other', '286a5ffe'), 'mirrorcoat': ('retaliation', '286a5ffe'),
            'counter': ('retaliation', '0d661773')})   # A1 4ee5d3ee, A2 286a5ffe, R 0d661773, M1 8b212312: the engine merges
        self.assertEqual((G.tier_of('hyperbeam'), G.guarded('hyperbeam', G.conditions(BASE, 0, request(OURS))['own'], 'breloom')), ('R', None))
        self.assertIn('hyperbeam', G.RECHARGE_MOVES)   # the live actor's own-recharge guard is separate and unchanged
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
        self.assertFalse(old['exact'])   # a pre-`guarded` record: its bounds keep the tiers it was written under (tier A)
        self.assertEqual((old['m_opp_lower'], old['m_opp_upper'], old['tripped']), (.5 * .75 * .75 * .4, .5 * .75 * .75 * .4, True))
        bare = dict(entry(0, 0, 1, 'opp', 'destinybond', .4), guarded=True)   # flagged but no tier: counted (bucket A)
        self.assertAlmostEqual(G.reach(reach_record([bare]), root_logits, root_legal, ReachTests.worlds)['tiers']['lower']['opp']['A'], .1125, places=15)

    def test_old_trick_records_keep_tier_a(self):
        root_logits, root_legal = ReachTests.root_logits, ReachTests.root_legal
        exact = dict(entry(0, 0, 1, 'opp', 'trick', .4), guarded=True, tier='A', user='x')   # as the base guard wrote it
        r = G.reach(reach_record([exact]), root_logits, root_legal, ReachTests.worlds)
        self.assertAlmostEqual(r['tiers']['lower']['opp']['A'], .1125, places=15);self.assertEqual(r['tiers']['lower']['opp']['C'], 0.)
        old = G.reach(reach_record([entry(0, 0, 1, 'opp', 'trick', .4)]), root_logits, root_legal, ReachTests.worlds)
        self.assertEqual((old['m_opp_lower'], old['m_opp_upper']), (.1125, .1125))   # pre-flag: tier A, so in the lower bound too

    def test_the_pre_flag_tiers_are_the_guard_before_retirement(self):
        frozen = {**dict.fromkeys(('hyperbeam', 'destinybond', 'transform', 'yawn', 'wrap', 'perishsong', 'trick', 'counter', 'mirrorcoat',
            'curse', 'superpower', 'knockoff', 'charm', 'lovelykiss', 'bonemerang', 'gigadrain'), 'A'),
            **dict.fromkeys(('spore', 'sleeppowder', 'hypnosis', 'refresh', 'sacredfire', 'psychoboost', 'facade', 'aromatherapy'), 'C')}
        self.assertEqual(G.PRE_FLAG_TIERS, frozen)   # base 8b212312's GUARDED tiers, a literal
        changed = {'trick': ('A', 'C')}   # M1: tier C under the knocked-off condition
        self.assertEqual({m: t for m, (t, _, _) in G.GUARDED.items() if m not in changed},
                         {m: t for m, t in G.PRE_FLAG_TIERS.items() if m not in G.RETIRED and m not in changed})
        self.assertEqual({m: (G.PRE_FLAG_TIERS[m], G.GUARDED[m][0]) for m in changed}, changed)
        self.assertEqual({m: G.PRE_FLAG_TIERS[m] for m in G.RETIRED}, {**dict.fromkeys(('destinybond', 'hyperbeam', 'curse', 'lovelykiss',
            'gigadrain', 'mirrorcoat', 'counter'), 'A'), 'aromatherapy': 'C', 'refresh': 'C'})
        self.assertEqual({m: c for m, (_, c, _) in BASE_RETIRED.items()}, {m: G.RETIRED[m][0] for m in BASE_RETIRED})   # same classes


KNOCK = ['|move|p1a: Breloom|Knock Off|p2a: Snorlax', '|-damage|p2a: Snorlax|80/100',
         '|-enditem|p2a: Snorlax|Leftovers|[from] move: Knock Off|[of] p1a: Breloom', '|turn|2']


class TrickConditionTests(unittest.TestCase):
    """Trick (tier C since M1, 8b212312): guarded for either side while either side has a non-fainted Pokemon whose item was
    publicly knocked off (live roots do not carry item_knocked; Showdown's takeItem fails on a knocked-off item, user or target)."""
    def test_a_public_knock_off_guards_both_sides_trick(self):
        c = G.conditions(BASE + KNOCK, 0, request(OURS))
        self.assertEqual((c['opp']['members']['snorlax']['knocked'], c['opp']['item_knocked'], c['own']['item_knocked']), (True, True, False))
        self.assertEqual(G.guarded('trick', c['own'], 'breloom', other=c['opp']), ('C', 'item_removal'))   # ours, onto the knocked target
        self.assertEqual(G.guarded('trick', c['opp'], 'snorlax', other=c['own']), ('C', 'item_removal'))   # theirs, from the knocked user
        mine = G.conditions([switch('p1', 'Breloom'), switch('p2', 'Kecleon'), '|turn|1', '|move|p2a: Kecleon|Knock Off|p1a: Breloom',
            '|-enditem|p1a: Breloom|Leftovers|[from] move: Knock Off|[of] p2a: Kecleon', '|turn|2'], 0, request(OURS))
        self.assertEqual((mine['own']['item_knocked'], G.guarded('trick', mine['opp'], 'kecleon', other=mine['own'])), (True, ('C', 'item_removal')))

    def test_no_knock_off_no_guard(self):
        c = G.conditions(BASE, 0, request(OURS))
        self.assertIsNone(G.guarded('trick', c['own'], 'breloom', other=c['opp']))
        eaten = G.conditions(BASE + ['|-enditem|p2a: Snorlax|Chesto Berry|[eat]', '|turn|2'], 0, request(OURS))   # consumed, not knocked off
        self.assertEqual((eaten['opp']['item_knocked'], G.guarded('trick', eaten['own'], 'breloom', other=eaten['opp'])), (False, None))
        fainted = G.conditions(BASE + KNOCK + ['|faint|p2a: Snorlax'], 0, request(OURS))   # the knocked Pokemon is gone
        self.assertEqual((fainted['opp']['item_knocked'], G.guarded('trick', fainted['own'], 'breloom', other=fainted['opp'])), (False, None))

    def test_a_child_keeps_the_root_facts_with_its_own_faints(self):
        c = G.conditions(BASE + KNOCK, 0, request(OURS))
        alive = G.child_side(c['opp'], [('snorlax', 0, True, False)], opponent=True)
        gone = G.child_side(c['opp'], [('snorlax', 0, False, False)], opponent=True)
        self.assertEqual((alive['item_knocked'], gone['item_knocked']), (True, False))
        self.assertEqual(G.guarded('trick', G.child_side(c['own'], [('breloom', 0, True, False)]), 'breloom', other=gone), None)

    def test_a_side_from_before_the_condition(self):
        old = dict(active='breloom', members={'breloom': dict(status='', fainted=False, sleep_source=None, white_herb=False, guts=False)},
                   sleep_clause=False, party_statused=False)
        self.assertIsNone(G.guarded('trick', old, 'breloom', other=old))
        self.assertFalse(G.child_side(old, [('breloom', 0, True, False)])['item_knocked'])

    def test_the_guard_trips_on_trick_only_under_the_condition(self):
        lanes = ['trick', 'leechseed', 'substitute', 'protect'];own = prior(0, .9)
        for lines, tripped in ((BASE, False), (BASE + KNOCK, True)):
            with self.subTest(knocked=tripped):
                c = G.conditions(lines, 0, request(OURS, moves=lanes))
                r = G.evaluate(c, own_turn1=dict(lanes=lanes, logits=own[0], legal=own[1]), peer_turn1=[], turn2=[], choice=0)
                self.assertEqual((r['tripped'], r['b_turn1'], r['c'], r['own_guarded_lanes'][0]), (tripped, tripped, tripped, ['C', 'item_removal'] if tripped else None))
                logits, legal = prior(1, .8)   # the opponent's Trick at turn 1, and at a turn-2 child where nobody fainted
                peer = [dict(world=0, lanes=['bodyslam', 'trick', 'rest', None], logits=logits, legal=legal, requested=True, user='snorlax')]
                child = dict(own_side=G.child_side(c['own'], [('breloom', 0, True, False)]), opp_side=G.child_side(c['opp'], [('snorlax', 0, True, False)], opponent=True))
                t2 = [dict(world=0, cell=0, side='opp', lanes=['bodyslam', 'trick', 'rest', None], logits=logits, legal=legal, user='snorlax', **child)]
                quiet = prior(2, .9)
                r = G.evaluate(c, own_turn1=dict(lanes=lanes, logits=quiet[0], legal=quiet[1]), peer_turn1=peer, turn2=t2, choice=2)
                self.assertEqual(([x['side'] for x in r['reasons'] if x['source'] == 'turn1'], r['masses']['turn2'][0]['guarded']),
                                 (['opp'] if tripped else [], tripped))


def chart_of(pairs):
    """A Gen 3 type chart by name: neutral everywhere but `pairs` {(attack, defend): multiplier}."""
    names = ('normal', 'fighting', 'ground', 'ghost', 'bug', 'fire', 'water', 'grass', 'dragon', 'ice', 'dark', 'psychic', 'electric')
    return {a: {d: pairs.get((a, d), 1.) for d in names} for a in names}


CHART = chart_of({('normal', 'ghost'): 0., ('fighting', 'ghost'): 0., ('fighting', 'bug'): .5, ('ground', 'bug'): .5, ('ghost', 'ghost'): 2.,
    ('dark', 'ghost'): 2., ('fire', 'bug'): 2., ('fire', 'grass'): 2., ('fire', 'fire'): .5, ('fire', 'water'): .5, ('water', 'fire'): 2.,
    ('bug', 'ghost'): .5, ('bug', 'fire'): .5, ('bug', 'grass'): 2.})
CATALOG = {m: dict(category=c, type=t, contact=k) for m, c, t, k in (
    ('tackle', 'Physical', 'normal', True), ('seismictoss', 'Physical', 'fighting', True), ('earthquake', 'Physical', 'ground', False),
    ('dragonrage', 'Special', 'dragon', False), ('sheercold', 'Special', 'ice', False), ('flamethrower', 'Special', 'fire', False),
    ('firepunch', 'Special', 'fire', True), ('shadowball', 'Physical', 'ghost', False), ('crunch', 'Special', 'dark', True),
    ('struggle', 'Physical', 'normal', True), ('beatup', 'Physical', 'dark', False), ('toxic', 'Status', 'poison', False),
    ('hiddenpowerbug', 'Physical', 'bug', False), ('surf', 'Special', 'water', False), ('megahorn', 'Physical', 'bug', True))}
SHEDINJA = dict(ability='wonderguard', hp=1, max_hp=1, types=['bug', 'ghost'], hp_type=None)
BRELOOM = dict(ability='effectspore', hp=200, max_hp=200, types=['grass', 'fighting'], hp_type=None)


def mon(ability, hp=100, top=300, types=('normal',), hp_type=None):
    return dict(ability=ability, hp=hp, max_hp=top, types=list(types), hp_type=hp_type)


class AbilityLaneTests(unittest.TestCase):
    """ability_lanes: Showdown f76228a's Gen 3 Wonder Guard (gen4 abilities.ts:530-543), pinch abilities (gen4 :25-35) and contact
    statusers (gen3 abilities.ts:12-30, 135-144), lane by lane."""
    def lanes(self, user, target, moves, pinch='third'):
        return G.ability_lanes(user, target, list(moves), CATALOG, CHART, pinch=pinch)

    def test_wonder_guard_blocks_every_damaging_move_that_is_not_super_effective(self):
        user = mon('pressure')
        wg = self.lanes(user, SHEDINJA, ('tackle', 'dragonrage', 'flamethrower', 'shadowball'))['WG']
        self.assertEqual(wg, [False, True, False, False])   # power-0 Dragon Rage (neutral) blocked, the engine KOs; immune Normal: 0 in both; Fire, Ghost hit
        wg = self.lanes(user, SHEDINJA, ('sheercold', 'seismictoss', 'crunch', 'earthquake'))['WG']
        self.assertEqual(wg, [True, False, False, True])   # OHKO (neutral) and Ground x0.5 blocked; immune Seismic Toss: 0 in both; Dark x2 hits
        wg = self.lanes(user, SHEDINJA, ('struggle', 'beatup', 'toxic', None))['WG']
        self.assertEqual(wg, [False] * 4)   # typeless moves pass Wonder Guard; a Status move is not its business; no move

    def test_hidden_power_takes_the_users_own_type(self):
        self.assertEqual(self.lanes(mon('pressure', hp_type='fire'), SHEDINJA, ('hiddenpowerbug',))['WG'][0], False)   # fire: super-effective
        self.assertEqual(self.lanes(mon('pressure', hp_type='water'), SHEDINJA, ('hiddenpowerbug',))['WG'][0], True)
        self.assertEqual(self.lanes(mon('pressure'), SHEDINJA, ('hiddenpowerbug',))['WG'][0], True)   # no sampled type: the row's (bug, x0.5)

    def test_pinch_at_a_third_or_two_thirds(self):
        moves = ('flamethrower', 'tackle', 'firepunch', 'toxic')
        for hp, third, two in ((100, [True, False, True, False], [True, False, True, False]), (101, [False] * 4, [True, False, True, False]),
                               (200, [False] * 4, [True, False, True, False]), (201, [False] * 4, [False] * 4), (0, [False] * 4, [False] * 4)):
            with self.subTest(hp=hp):
                self.assertEqual(self.lanes(mon('blaze', hp, 300), None, moves)['PINCH'], third)
                self.assertEqual(self.lanes(mon('blaze', hp, 300), None, moves, 'two_thirds')['PINCH'], two)
        self.assertEqual(self.lanes(mon('swarm', 1, 300), None, ('megahorn', 'hiddenpowerbug', 'tackle'))['PINCH'][:3], [True, True, False])
        self.assertEqual(self.lanes(mon('blaze', 1, 300, hp_type='fire'), None, ('hiddenpowerbug',))['PINCH'][0], True)   # Hidden Power Fire
        self.assertEqual(self.lanes(mon('torrent', 1, 300), None, ('flamethrower',))['PINCH'][0], False)

    def test_contact_statusers_and_what_is_modelled(self):
        moves = ('tackle', 'earthquake', 'toxic', 'firepunch')
        self.assertEqual(self.lanes(mon('pressure'), BRELOOM, moves)['CONTACT'], [True, False, False, True])
        for name in ('poisonpoint', 'flamebody'):
            self.assertEqual(self.lanes(mon('pressure'), mon(name), moves)['CONTACT'], [True, False, False, True], name)
        for name in ('static', 'roughskin', 'pressure', None):   # Static and Rough Skin the engine models
            self.assertEqual(self.lanes(mon('pressure'), mon(name), moves), {f: [False] * 4 for f in G.FAMILIES}, name)
        self.assertEqual(G.damaging_lanes(['tackle', 'toxic', None, 'unknownmove'], CATALOG), [True, False, False, False])
        self.assertEqual([G.holder_family(mon(a)) for a in ('wonderguard', 'flamebody', 'blaze', None)], ['WG', 'CONTACT', None, None])


def ability_root(world, own=None, opp=None, dmg=((True,) * 4, (True,) * 4)):
    """A turn-1 ability entry: `own` / `opp` {family: [bool] * 4} for both PINCH rules (two_thirds overrides with 'PINCH2')."""
    side = lambda m, d: dict(third={f: list((m or {}).get(f, [False] * 4)) for f in G.FAMILIES},
        two_thirds={f: list((m or {}).get('PINCH2' if f == 'PINCH' else f, (m or {}).get(f, [False] * 4))) for f in G.FAMILIES}, damaging=list(d))
    return dict(world=world, own=side(own, dmg[0]), opp=side(opp, dmg[1]))


class AbilityEvaluateTests(unittest.TestCase):
    """ability_evaluate on one world (p_w 1) and two (1/2 each): our prior and the peer's, the arms one by one."""
    def priors(self, own=(0, .9), peer=(0, .6), requested=True, worlds=(0,)):
        o = prior(*own, legal=[True] * 9);p = prior(*peer, legal=[True] * 9)
        return dict(logits=o[0], legal=o[1]), [dict(world=w, logits=p[0], legal=p[1], requested=requested) for w in worlds]

    def test_turn_1_both_sides_and_the_record(self):
        own, peer = self.priors()
        r = G.ability_evaluate(own, peer, dict(turn1=[ability_root(0, own=dict(WG=[True, False, False, False]))]))
        self.assertEqual((r['tripped'], r['b_turn1'], r['b_reach'], r['b_switchin'], r['c']), (True, True, False, False, False))
        reason, = r['reasons'];self.assertEqual((reason['source'], reason['family'], reason['side'], reason['worlds']), ('ability_turn1', ['WG'], 'own', [0]))
        self.assertAlmostEqual(reason['max_mass'], .9, places=12);self.assertAlmostEqual(r['families']['WG']['turn1']['own'], .9, places=12)
        theirs = G.ability_evaluate(own, peer, dict(turn1=[ability_root(0, opp=dict(CONTACT=[True, False, False, False]))]), arms=G.ABILITY_ARMS_FULL)
        self.assertEqual([(x['side'], x['family']) for x in theirs['reasons']], [('opp', ['CONTACT'])])
        declared = G.ability_evaluate(own, peer, dict(turn1=[ability_root(0, opp=dict(CONTACT=[True, False, False, False]))]))
        self.assertEqual((declared['tripped'], declared['families']['CONTACT']['turn1']['opp']), (False, theirs['families']['CONTACT']['turn1']['opp']))   # choice-only: recorded
        waiting = G.ability_evaluate(own, self.priors(requested=False)[1], dict(turn1=[ability_root(0, opp=dict(CONTACT=[True] * 4))]), arms=G.ABILITY_ARMS_FULL)
        self.assertFalse(waiting['tripped'])   # a waiting opponent has no prior
        low = G.ability_evaluate(*self.priors(own=(0, .04)), dict(turn1=[ability_root(0, own=dict(WG=[True, False, False, False]))]))
        self.assertFalse(low['b_turn1'])   # 4% < 5%

    def test_the_root_pinch_rule(self):
        root = [ability_root(0, own=dict(PINCH2=[True, False, False, False]), dmg=((True,) * 4, (False,) * 4))]   # their lanes: no damage
        own, peer = self.priors()
        full = G.ABILITY_ARMS_FULL
        self.assertFalse(G.ability_evaluate(own, peer, dict(turn1=root), arms=full, pinch_root='two_thirds')['tripped'])   # nothing can bring us to a third this turn
        root = [ability_root(0, own=dict(PINCH2=[True, False, False, False]))]
        self.assertTrue(G.ability_evaluate(own, peer, dict(turn1=root), arms=full, pinch_root='two_thirds')['b_turn1'])
        self.assertFalse(G.ability_evaluate(own, peer, dict(turn1=root), arms=full, pinch_root='third')['tripped'])   # exact rule: hp above a third
        declared = G.ability_evaluate(own, peer, dict(turn1=root))
        self.assertEqual((declared['pinch_root'], declared['tripped']), ('third', False))   # the declared root rule is the exact one
        at_third = [ability_root(0, own=dict(PINCH=[True, False, False, False]))]
        self.assertTrue(G.ability_evaluate(own, peer, dict(turn1=at_third), arms=full)['b_turn1'])
        declared = G.ability_evaluate(own, peer, dict(turn1=at_third), choice=1)   # declared: PINCH choice-only
        self.assertEqual((declared['tripped'], declared['families']['PINCH']['turn1_third']['own'] > .8), (False, True))   # recorded
        self.assertTrue(G.ability_evaluate(own, peer, dict(turn1=at_third), choice=0)['c'])   # our chosen pinch lane

    def test_the_choice_and_choice_only_families(self):
        own, peer = self.priors(own=(0, .01))
        inputs = dict(turn1=[ability_root(0, own=dict(CONTACT=[True, False, False, False]))])
        r = G.ability_evaluate(own, peer, inputs, choice=0)
        self.assertEqual((r['tripped'], r['b'], r['c'], r['own_guarded_lanes'][0]), (True, False, True, ['CONTACT']))
        only = G.ability_evaluate(*self.priors(), inputs, choice=1, arms=dict(G.ABILITY_ARMS, CONTACT=('choice',)))
        self.assertEqual((only['tripped'], only['families']['CONTACT']['turn1']['own'] > .8), (False, True))   # recorded, not tripping
        self.assertTrue(G.ability_evaluate(*self.priors(), inputs, choice=0, arms=dict(G.ABILITY_ARMS, CONTACT=('choice',)))['c'])

    def test_the_turn_1_switch_in(self):
        own, peer = self.priors(own=(4, .5), peer=(0, .6))   # we switch to lane 4 at .5; they attack with lane 0 at .6
        entry = dict(side='own', lane=4, family='WG', holder='wonderguard', against=[True, False, False, False])
        r = G.ability_evaluate(own, peer, dict(turn1=[ability_root(0)], switchin={0: [entry]}))
        self.assertAlmostEqual(r['families']['WG']['switchin']['opp'], .5 * .6, places=12)   # credited to the attacker
        self.assertEqual((r['b_switchin'], [(x['source'], x['side'], x['family']) for x in r['reasons']]), (True, [('ability_switchin', 'opp', ['WG'])]))
        again = G.ability_evaluate(own, peer, json.loads(json.dumps(dict(turn1=[ability_root(0)], switchin={0: [entry]}))))
        self.assertEqual(again['families'], r['families'])   # world keys survive a JSON round trip
        theirs = dict(side='opp', lane=4, family='CONTACT', holder='flamebody', against=[True, False, False, False])
        own2, peer2 = self.priors(own=(0, .3), peer=(4, .5))
        self.assertAlmostEqual(G.ability_evaluate(own2, peer2, dict(switchin={0: [theirs]}))['families']['CONTACT']['switchin']['own'], .15, places=12)
        self.assertFalse(G.ability_evaluate(own, self.priors(requested=False)[1], dict(switchin={0: [entry]}))['tripped'])   # they wait
        self.assertFalse(G.ability_evaluate(own, peer, dict(switchin={0: [entry]}), arms=dict(G.ABILITY_ARMS, WG=('choice',)))['b_switchin'])

    def test_the_reach_by_hand(self):
        own, peer = self.priors(own=(0, .75), peer=(1, .75), worlds=(0, 1))
        q = prior(0, .4, legal=[True] * 9)
        cell = lambda w, a, b, side, lanes: dict(world=w, cell=a * 9 + b, side=side, lanes=lanes, logits=q[0], legal=q[1])
        contact = {f: [f == 'CONTACT', False, False, False] for f in G.FAMILIES}
        pinch = {f: [f == 'PINCH', f == 'PINCH', False, False] for f in G.FAMILIES}
        r = G.ability_evaluate(own, peer, dict(turn2=[cell(0, 0, 1, 'opp', contact), cell(1, 0, 1, 'own', pinch)]), arms=G.ABILITY_ARMS_FULL)
        w = .5 * .75 * .75   # p_w x pi_own(0) x pi_peer(1)
        self.assertAlmostEqual(r['m_opp'], w * .4, places=12);self.assertAlmostEqual(r['m_own'], w * (.4 + .075), places=12)
        self.assertEqual((r['b_reach'], sorted(x['side'] for x in r['reasons'])), (True, ['opp', 'own']))
        self.assertAlmostEqual(r['families']['PINCH']['reach']['own'], w * .475, places=12)
        off = G.ability_evaluate(own, peer, dict(turn2=[cell(0, 0, 1, 'opp', contact)]), arms=dict(G.ABILITY_ARMS, CONTACT=('choice',)))
        self.assertEqual((off['b_reach'], off['m_opp']), (False, 0.));self.assertAlmostEqual(off['families']['CONTACT']['reach']['opp'], w * .4, places=12)

    def test_any_configuration_from_the_record_alone(self):
        """ability_trips: the record's own configuration reproduces its decision; every other one (choice-only families, the exact
        root PINCH rule) equals a fresh evaluation under it, from the record alone (reach_subsets, turn1_third, choice_third)."""
        own, peer = self.priors(own=(0, .75), peer=(1, .75), worlds=(0, 1))
        q = prior(0, .4, legal=[True] * 9)
        cell = lambda w, a, b, side, lanes: dict(world=w, cell=a * 9 + b, side=side, lanes=lanes, logits=q[0], legal=q[1])
        both = {f: [f in ('CONTACT', 'PINCH'), f == 'PINCH', False, False] for f in G.FAMILIES}   # lane 0 guarded by two families
        inputs = dict(turn1=[ability_root(0, own=dict(PINCH2=[True, False, False, False])), ability_root(1)],
                      switchin={0: [dict(side='own', lane=4, family='WG', holder='wonderguard', against=[True, False, False, False])]},
                      turn2=[cell(0, 0, 1, 'opp', both), cell(1, 0, 1, 'own', {f: [f == 'WG'] + [False] * 3 for f in G.FAMILIES})])
        full = dict(arms=G.ABILITY_ARMS_FULL, pinch_root='two_thirds')   # the record keeps every arm's masses and both root rules
        # a config names what differs from the record's own (ability_trips reads an absent key from the record), the declared one included
        configs = [dict(), dict(arms=G.ABILITY_ARMS, pinch_root=G.PINCH_ROOT), dict(arms=dict(G.ABILITY_ARMS_FULL, CONTACT=('choice',))),
                   dict(arms=dict(G.ABILITY_ARMS_FULL, WG=('switchin', 'choice'))), dict(pinch_root='third'),
                   dict(arms=dict(G.ABILITY_ARMS_FULL, CONTACT=('choice',), PINCH=('choice',)), pinch_root='third')]
        record = G.ability_evaluate(own, peer, inputs, choice=0, **full)
        w = .5 * .75 * .75
        self.assertAlmostEqual(record['reach_subsets']['PINCH+CONTACT']['opp'], w * .475, places=12)   # the union, not the sum
        self.assertAlmostEqual(record['families']['PINCH']['reach']['opp'] + record['families']['CONTACT']['reach']['opp'], w * .875, places=12)
        for kw in configs:
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                fresh = G.ability_evaluate(own, peer, inputs, choice=0, **dict(full, **kw))
                self.assertEqual(G.ability_trips(record, **kw), {k: fresh[k] for k in ('tripped', 'b_turn1', 'b_reach', 'b_switchin', 'c')})
        self.assertEqual(G.ability_trips(record)['tripped'], record['tripped'])
        with self.assertRaisesRegex(ValueError, 'threshold'):G.ability_trips(record, threshold=.07)
        with self.assertRaisesRegex(ValueError, 'pinch_root'):G.ability_trips(record, pinch_root='half')
        third = G.ability_evaluate(own, peer, inputs, choice=0, pinch_root='third')
        with self.assertRaisesRegex(ValueError, 'two_thirds'):G.ability_trips(third, pinch_root='two_thirds')
        self.assertEqual(G.ability_trips(third, arms=G.ABILITY_ARMS_FULL), {k: G.ability_evaluate(own, peer, inputs, choice=0, arms=G.ABILITY_ARMS_FULL,
            pinch_root='third')[k] for k in ('tripped', 'b_turn1', 'b_reach', 'b_switchin', 'c')})   # a 'third' record still decides its arms

    def test_the_declared_configuration(self):
        """Gate 2's decisions (the lead's rules on the trip-share replay): PINCH and CONTACT choice-only, WG every arm, PINCH's exact
        root rule. A record written under it carries them, so ability_trips reproduces it, and the same record decides every arm."""
        self.assertEqual(G.ABILITY_ARMS, dict(WG=('turn1', 'reach', 'switchin', 'choice'), PINCH=('choice',), CONTACT=('choice',)))
        self.assertEqual(G.PINCH_ROOT, 'third')
        own, peer = self.priors()   # our lane 0 at .9, theirs at .6
        q = prior(0, .4, legal=[True] * 9)
        contact = {f: [f == 'CONTACT', False, False, False] for f in G.FAMILIES}
        inputs = dict(turn1=[ability_root(0, own=dict(PINCH=[True, False, False, False]), opp=dict(CONTACT=[True, False, False, False]))],
                      turn2=[dict(world=0, cell=0, side='opp', lanes=contact, logits=q[0], legal=q[1])])   # a CONTACT-only reach cell
        keys = ('tripped', 'b_turn1', 'b_reach', 'b_switchin', 'c')
        r = G.ability_evaluate(own, peer, inputs, choice=1)   # a lane that is neither pinch nor contact
        self.assertEqual((r['arms'], r['pinch_root']), ({f: list(a) for f, a in G.ABILITY_ARMS.items()}, 'third'))
        self.assertEqual({k: r[k] for k in keys}, dict(tripped=False, b_turn1=False, b_reach=False, b_switchin=False, c=False))
        self.assertEqual(G.ability_trips(r), {k: r[k] for k in keys})   # the record's own configuration
        full = G.ability_evaluate(own, peer, inputs, choice=1, arms=G.ABILITY_ARMS_FULL)
        self.assertEqual((full['b_turn1'], full['b_reach']), (True, True))   # PINCH turn 1 (ours), CONTACT reach (theirs): unguarded now
        self.assertEqual(G.ability_trips(r, arms=G.ABILITY_ARMS_FULL), {k: full[k] for k in keys})   # a 'third' record decides every arm
        self.assertTrue(G.ability_evaluate(own, peer, inputs, choice=0)['c'])   # our chosen pinch lane stays guarded

    def test_evaluate_folds_the_ability_record_in(self):
        cond = G.conditions(BASE, 0, request(OURS));lanes = ['tackle', 'leechseed', 'substitute', 'protect'];own = prior(1, .9)   # lane 0: 3%
        plain = G.evaluate(cond, own_turn1=dict(lanes=lanes, logits=own[0], legal=own[1]), peer_turn1=[], turn2=[], choice=0)
        self.assertNotIn('ability', plain);self.assertFalse(plain['tripped'])
        with_ab = G.evaluate(cond, own_turn1=dict(lanes=lanes, logits=own[0], legal=own[1]), peer_turn1=[], turn2=[], choice=0,
            ability=dict(turn1=[ability_root(0, own=dict(WG=[True, False, False, False]))]))
        self.assertEqual((with_ab['tripped'], with_ab['b'], with_ab['arms'], with_ab['ability']['c']), (True, False, ['ability'], True))
        self.assertEqual(G.trips(with_ab, 0), dict(tripped=True, b=False, c=True));self.assertEqual(G.trips(with_ab, 1), dict(tripped=False, b=False, c=False))
        b_only = G.evaluate(cond, own_turn1=dict(lanes=lanes, logits=prior(0, .9)[0], legal=prior(0, .9)[1]), peer_turn1=[], turn2=[], choice=1,
            ability=dict(turn1=[ability_root(0, own=dict(WG=[True, False, False, False]))]))   # the ability guard's (b) alone
        self.assertEqual((b_only['b'], b_only['ability']['b'], G.trips(b_only, 1)), (False, True, dict(tripped=True, b=True, c=False)))
        self.assertEqual({k: v for k, v in with_ab.items() if k not in ('ability', 'tripped', 'arms')}, {k: v for k, v in plain.items() if k not in ('tripped', 'arms')})



if __name__ == '__main__':
    unittest.main()
