"""Ledger arithmetic and evidence references, not gameplay qualification."""
import copy
from pathlib import Path
import unittest

from gpu_engine.coverage import ledger,registry


class CoverageTests(unittest.TestCase):
    def fixture(self):
        return dict(move_slots=100,move_frequency=[
            {'id':'earthquake','count':40},{'id':'hiddenpowergrass','count':20},
            {'id':'brickbreak','count':10},{'id':'rapidspin','count':10},
            {'id':'transform','count':20}])

    def test_partial_and_placeholders_receive_no_selected_effect_credit(self):
        r=ledger(self.fixture(),root=Path(__file__).resolve().parents[1])
        self.assertEqual(r['weighted_status_shares'],dict(SELECTED_EFFECT_IMPLEMENTED=.6,PARTIAL=.2,PLACEHOLDER=.2))
        self.assertIsNone(r['game_qualified_move_share'])
        self.assertIn('not full-game qualification',r['rows'][0]['evidence'])

    def test_hidden_power_alias_counts_once_at_its_actual_slot_frequency(self):
        m=self.fixture();m['move_frequency'][1]['id']='hiddenpowerice'
        r=ledger(m,root=Path(__file__).resolve().parents[1])
        self.assertEqual(sum(x['count'] for x in r['rows']),100)
        self.assertEqual(r['rows'][1]['count'],20)

    def test_missing_reference_is_refused(self):
        with self.assertRaisesRegex(ValueError,'missing source'):
            ledger(self.fixture(),root=Path('/nonexistent-authored-fixture-root'))

    def test_duplicate_padding_and_bad_totals_are_refused(self):
        for change in ('duplicate','padding','count','total'):
            m=copy.deepcopy(self.fixture())
            if change=='duplicate':m['move_frequency'][1]['id']='earthquake'
            elif change=='padding':m['move_frequency'][1]['id']='_none'
            elif change=='count':m['move_frequency'][1]['count']=0
            else:m['move_slots']=101
            with self.subTest(change=change),self.assertRaises(ValueError):
                ledger(m,root=Path(__file__).resolve().parents[1])

    def test_intrinsic_missing_effects_are_explicit(self):
        r=registry()
        self.assertEqual(r['brickbreak']['status'],'PARTIAL')
        self.assertIn('Screen',r['brickbreak']['missing_intrinsic_effect'])
        self.assertIn('Partial-trap',r['rapidspin']['missing_intrinsic_effect'])

    def test_pain_split_registration_is_not_fidelity_credit(self):
        for name in ('painsplit','reversal','flail'):
            result=ledger(dict(move_slots=1,move_frequency=[{'id':name,'count':1}]),
                          root=Path(__file__).resolve().parents[1])
            self.assertEqual(result['rows'][0]['status'],'SELECTED_EFFECT_IMPLEMENTED')
            self.assertIsNone(result['game_qualified_move_share'])

    def test_dragon_dance_source_registration_does_not_award_fidelity_credit(self):
        result=ledger(dict(move_slots=100,move_frequency=[{'id':'dragondance','count':100}]),
                      root=Path(__file__).resolve().parents[1])
        self.assertEqual(result['rows'][0]['status'],'SELECTED_EFFECT_IMPLEMENTED')
        self.assertIsNone(result['game_qualified_move_share'])
        self.assertIn('not full-game qualification',result['rows'][0]['evidence'])
