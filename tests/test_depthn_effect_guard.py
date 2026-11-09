"""pokezero.depthn_effect_guard on exact Gen 3 protocol lines: the tier-C conditions, (b) at turn 1 and turn 2, (c) per choice."""
import json
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
        self.assertEqual(G.guarded('destinybond', c['own'], 'breloom'), ('A', 'volatile'))   # tier A: always
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

    def test_b_at_turn_2_reads_the_childs_conditions(self):
        child = lambda members_opp: dict(own_side=G.child_side(self.cond['own'], [('breloom', 0, True, False)]),
            opp_side=G.child_side(self.cond['opp'], members_opp))
        logits, legal = prior(0, .5)
        cell = dict(world=2, cell=10, side='own', lanes=['spore', 'machpunch', 'leechseed', 'substitute'], logits=logits, legal=legal, user='breloom')
        awake = self.record(turn2=[dict(cell, **child([('snorlax', 0, True, False)]))])
        self.assertFalse(awake['b'])   # Spore while no opponent sleeps: no clause, not guarded
        asleep = self.record(turn2=[dict(cell, **child([('snorlax', 5, True, False)]))])   # the turn-1 Spore landed in this child
        self.assertEqual((asleep['b'], asleep['b_turn1'], asleep['b_turn2']), (True, False, True))
        self.assertEqual([(x['move'], x['source'], x['worlds']) for x in asleep['reasons']], [('spore', 'turn2', [2])])
        rested = self.record(turn2=[dict(cell, **child([('snorlax', 5, True, True)]))])   # asleep from its own Rest: no clause
        self.assertFalse(rested['b'])
        self.assertEqual(asleep['masses']['turn2'][0]['cell'], 10)
        missing = self.record(available=False)
        self.assertEqual((missing['tripped'], missing['b_turn2'], [x['source'] for x in missing['reasons']]), (True, True, ['turn2_unavailable']))

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
        self.assertEqual(sorted(k for k, v in G.GUARDED.items() if v[0] == 'A'), sorted(('hyperbeam', 'destinybond', 'transform', 'yawn', 'wrap',
            'perishsong', 'trick', 'counter', 'mirrorcoat', 'curse', 'superpower', 'knockoff', 'charm', 'lovelykiss', 'bonemerang', 'gigadrain')))


if __name__ == '__main__':
    unittest.main()
