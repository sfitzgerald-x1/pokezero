"""Measured likelihood extensions: protocol boundaries, roll inequalities, direction and public HP bins."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pokezero import set_posterior as SP, tier2
from tests.test_depthn_set_posterior import Base, Edits, fixture


def prefix(*middle, owner='p1'):
    foe='p2' if owner=='p1' else 'p1'
    return [f'|switch|{owner}a: Snorlax|Snorlax, L80|400/400',
            f'|switch|{foe}a: Rhydon|Rhydon, L80|100/100', '|turn|1', *middle, '|request|{"wait":true}']


def sub_prefix(*hits, owner='p1'):
    return prefix(f'|move|{owner}a: Snorlax|Substitute|{owner}a: Snorlax',
                  f'|-start|{owner}a: Snorlax|Substitute', f'|-damage|{owner}a: Snorlax|300/400', *hits, owner=owner)


class ParserTests(unittest.TestCase):
    def test_only_the_exact_sleep_talk_source_is_allowed(self):
        for owner in ('p1','p2'):
            foe='p2' if owner=='p1' else 'p1'
            for tag,allowed in (('Sleep Talk',True),('move: Sleep Talk',True),('lockedmove',False),
                                ('Metronome',False),('Sleep Talk extra',False)):
                with self.subTest(owner=owner,tag=tag):
                    lines=prefix(f'|move|{foe}a: Rhydon|Earthquake|{owner}a: Snorlax|[from] {tag}',
                                 f'|-damage|{owner}a: Snorlax|300/400',owner=owner)
                    strike,=SP.parse_strikes(lines,owner)
                    self.assertEqual(strike['reason'],None if allowed else 'called')
                    self.assertEqual(strike['sleep_talk'],allowed)

    def test_only_first_fresh_nonbreaking_substitute_hit_has_a_capacity(self):
        for owner in ('p1','p2'):
            foe='p2' if owner=='p1' else 'p1'
            hit=[f'|move|{foe}a: Rhydon|Earthquake|{owner}a: Snorlax',f'|-activate|{owner}a: Snorlax|Substitute|[damage]']
            strikes=SP.parse_strikes(sub_prefix(*hit,*hit,owner=owner),owner)
            self.assertEqual([(s['reason'],s['sub_hp'],s['observed']) for s in strikes],[(None,100,None),('hit-substitute',None,None)])
            # An unsupported first interaction still spends freshness.
            lines=sub_prefix(*hit,*hit,owner=owner);lines[6]=lines[6].replace('Earthquake','Seismic Toss')
            strikes=SP.parse_strikes(lines,owner)
            self.assertEqual([s['reason'] for s in strikes],['fixed-damage','hit-substitute'])

    def test_broken_passed_replaced_and_mixed_substitute_events_refuse(self):
        move='|move|p2a: Rhydon|Earthquake|p1a: Snorlax'
        activation='|-activate|p1a: Snorlax|Substitute|[damage]'
        cases=[sub_prefix(move,'|-end|p1a: Snorlax|Substitute'),
               prefix('|-start|p1a: Snorlax|Substitute',move,activation),
               sub_prefix('|switch|p1a: Snorlax|Snorlax, L80|300/400',move,activation),
               sub_prefix(move,activation,'|-damage|p1a: Snorlax|200/400')]
        for lines in cases:
            with self.subTest(lines=lines):self.assertEqual(SP.parse_strikes(lines,'p1')[-1]['reason'],'hit-substitute')
        # Open windows remain unclassified even when the capacity is known.
        self.assertEqual(SP.parse_strikes(sub_prefix(move,activation)[:-1],'p1')[-1]['reason'],'window-open')

    def test_outgoing_attaches_to_the_defender_and_requires_current_full_public_hp(self):
        for owner in ('p1','p2'):
            foe='p2' if owner=='p1' else 'p1'
            attack=[f'|move|{owner}a: Snorlax|Body Slam|{foe}a: Rhydon',f'|-damage|{foe}a: Rhydon|99/100']
            strike,=SP.parse_strikes(prefix(*attack,owner=owner),owner,include_outgoing=True)
            self.assertEqual((strike['posterior_key'],strike['attacker_seat'],strike['defender_seat']),('rhydon',owner,foe))
            self.assertEqual((strike['observed'],strike['after_percent'],strike['observation_kind']),(None,99,'public-hp'))
            for event in (f'|-damage|{foe}a: Rhydon|99/100',f'|-sethp|{owner}a: Snorlax|300/400|{foe}a: Rhydon|50/100',
                          f'|-heal|{foe}a: Rhydon|100/200',f'|-heal|{foe}a: Rhydon|unknown'):
                self.assertEqual(SP.parse_strikes(prefix(event,*attack,owner=owner),owner,include_outgoing=True),[])
            restored=prefix(f'|-damage|{foe}a: Rhydon|50/100',f'|-heal|{foe}a: Rhydon|100/100',*attack,owner=owner)
            self.assertEqual(len(SP.parse_strikes(restored,owner,include_outgoing=True)),1)

    def test_mixed_directions_keep_chronology_and_other_guards(self):
        lines=prefix('|move|p1a: Snorlax|Body Slam|p2a: Rhydon','|-damage|p2a: Rhydon|80/100',
                     '|move|p2a: Rhydon|Earthquake|p1a: Snorlax','|-damage|p1a: Snorlax|300/400')
        strikes=SP.parse_strikes(lines,'p1',include_outgoing=True)
        self.assertEqual([s['posterior_key'] for s in strikes],['rhydon','rhydon'])
        self.assertEqual([s['attacker_seat'] for s in strikes],['p1','p2'])
        self.assertEqual([s['line'] for s in strikes],sorted(s['line'] for s in strikes))
        screened=prefix('|-sidestart|p2: foe|Reflect',*lines[3:-1])
        self.assertEqual(SP.parse_strikes(screened,'p1',include_outgoing=True)[0]['reason'],'unvalidated:screen')


class BinTests(unittest.TestCase):
    def test_all_bins_are_the_exact_inverse_including_99_tail_and_ko(self):
        for maximum in (1,99,100,101,257,513):
            for hp in range(maximum+1):
                shown=0 if hp==0 else 100 if hp==maximum else min(99,(100*hp+maximum-1)//maximum)
                lo,hi=SP._public_hp_bounds(shown,maximum)
                self.assertLessEqual(lo,hp);self.assertLessEqual(hp,hi)
            for shown in (0,1,98,99,100):
                lo,hi=SP._public_hp_bounds(shown,maximum)
                for hp in range(maximum+1):
                    printed=0 if hp==0 else 100 if hp==maximum else min(99,(100*hp+maximum-1)//maximum)
                    self.assertEqual(lo<=hp<=hi,printed==shown)


class LikelihoodTests(Edits):
    def test_sleep_talk_uses_the_same_declared_move_damage_counts(self):
        baseline,_=self.reason(self.lines)
        for tag in ('|[from] Sleep Talk','|[from] move: Sleep Talk'):
            strike,_=self.reason(self.moved('Body Slam',tag))
            self.assertTrue(strike.admitted);self.assertEqual(strike.counts,baseline.counts)

    def test_substitute_counts_are_strict_inequalities_and_whole_strike_guards_remain(self):
        raw=SP.parse_strikes(self.lines,self.me)[0]
        raw.update(observation_kind='substitute',sub_hp=100,observed=None,ko=False)
        variants=self.variants('snorlax');hard=tuple(True for _ in variants)
        team=tier2.own_team_from_request(SP._first_request(self.lines))
        damage=NS(rolls=(0,99,100,101)*4,ambiguous=False)
        with patch.object(tier2,'_strike_damage',return_value=damage),patch.object(SP,'_effectiveness_line',return_value=raw['eff']):
            reason,counts=SP._counts(raw,'snorlax',variants,hard,team,source=self.src,dex=self.dex)
        self.assertIsNone(reason);self.assertEqual(set(counts),{4})
        with patch.object(tier2,'_strike_damage',return_value=NS(rolls=damage.rolls,ambiguous=True)):
            reason,counts=SP._counts(raw,'snorlax',variants,hard,team,source=self.src,dex=self.dex)
        self.assertEqual((reason,counts),('ambiguous-band',None))

    def test_real_outgoing_hit_keeps_the_other_seat_true_variant(self):
        data=fixture('hp_immune_kingler');posts=self.post(data);post=posts['kingler']
        outgoing=[s for s in post.strikes if s.observation_kind=='public-hp']
        self.assertTrue(outgoing);self.assertEqual(outgoing[0].observed,None)
        true=data['census']['species']['kingler']['true_variant'];ids=[v.variant_id for v in self.variants('kingler')]
        self.assertGreater(post.weights[ids.index(true)],0)
        self.assertTrue(all(isinstance(w,int) for w in post.weights))
