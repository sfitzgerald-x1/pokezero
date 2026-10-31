"""Synthetic comparator/reducer contracts; no Battle, game, generator or model."""
from collections import namedtuple
from types import SimpleNamespace as NS
import json
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import finite_stream_comparison as comparison


def captured(rows):
    words = np.zeros((1, 64, 16), np.int32)
    words[0, :len(rows)] = rows
    return words, np.arange(64)[None] < len(rows), np.array([False])


def observation():
    return dict(numeric_features=np.zeros((23, 132), np.float32),
        categorical_ids=np.zeros((23, 41), np.int32), token_type_ids=np.zeros(23, np.int32),
        attention_mask=np.ones(23, bool), legal_action_mask=np.array([True]+[False]*8))


def writer(state, line, **_):
    if 'unsupported' in line:
        raise ValueError('unhandled actual raw family')
    value = 7 if line.startswith('|switch|') else 8
    return state, NS(events=[NS(words=lambda:[value]+[0]*15)])


class FiniteStreamComparisonTests(unittest.TestCase):
    def test_only_literal_current_actual_owner_bridge_messages_are_captured(self):
        env = comparison.OwningShowdownEnv(comparison.LocalShowdownConfig())
        env._battle_token = 'b1'
        with patch.object(comparison.LocalShowdownEnv, '_apply_event', return_value=False) as base:
            for stream, token, line in [('p1','b1','|switch|p1a: Mew|Mew|341/341'),
                    ('p2','b1','|switch|p1a: Mew|Mew|100/100'),
                    ('omniscient','b1','|switch|p1a: Mew|Mew|341/341'),
                    ('p1','old','|switch|p1a: foreign|foreign|1/1')]:
                env._apply_event(dict(type='stream',stream=stream,battleId=token,lines=[line]))
            self.assertEqual(base.call_count,4)
            self.assertEqual(env.owner_transcripts,[['|switch|p1a: Mew|Mew|341/341'],
                ['|switch|p1a: Mew|Mew|100/100']])
            with self.assertRaisesRegex(ValueError,'literal'):
                env._apply_event(dict(type='stream',stream='p1',battleId='b1',lines=[123]))
        with patch.object(env,'_reset') as reset:
            env.reset_pool_sets(['original-p1','original-p2'],seed=123)
            self.assertEqual(env.owner_transcripts,[[],[]])
            self.assertEqual(reset.call_args.kwargs['format_id'],'gen3randombattle')
            self.assertEqual(reset.call_args.kwargs['start_override'].player_teams,
                {'p1':'original-p1','p2':'original-p2'})
        env.close()

    def test_exact_original_pool_sets_are_packed_without_regeneration(self):
        arrays = {name:np.ones((2,6),np.int32) for name in ('species','level','item','ability','nature')}
        arrays.update(moves=np.tile([1,2,0,0],(2,6,1)), ivs=np.full((2,6,6),30,np.int32),
            evs=np.full((2,6,6),84,np.int32))
        vocab = dict(species=['_none','mew'],moves=['_none','hiddenpowerfire','tackle'],
            items=['_none','leftovers'],abilities=['_none','synchronize'],natures=['_none','serious'])
        pool = NS(arrays=arrays,metadata=lambda:dict(vocabulary=vocab))
        genders=np.array([[0,1,2,0,1,2],[2,1,0,2,1,0]],np.int32)
        with patch.object(comparison.source_provenance,'_pool') as checked:
            parties=comparison.pool_sets(pool,[1,0],genders,np.full((2,6),123,np.int32))
            checked.assert_called_once_with(pool,rehash=True)
            self.assertEqual(parties[0][2]['gender'],'F')
            self.assertEqual(parties[1][2]['gender'],'N')
            self.assertEqual(parties[0][0]['moves'],['hiddenpowerfire','tackle'])
            self.assertEqual(parties[0][0]['happiness'],123)
            self.assertEqual(parties[0][0]['ivs']['spa'],30)
            self.assertEqual(parties[0][0]['evs']['hp'],84)
            with self.assertRaises(ValueError):comparison.pool_sets(pool,[0,0],genders)
            with self.assertRaises(ValueError):comparison.pool_sets(pool,[0,1],np.full((2,6),-1,np.int32))
        with patch.object(comparison.subprocess,'check_output',return_value='["p1-packed","p2-packed"]') as pack:
            self.assertEqual(comparison.pack_sets(parties,showdown_root='/pinned',node='node'),
                ['p1-packed','p2-packed'])
            command=pack.call_args.args[0]
            self.assertIn('Teams.pack',command[2])
            self.assertNotIn('Teams.generate',command[2])
            self.assertEqual(json.loads(pack.call_args.kwargs['input']),parties)

    def test_actual_reference_refusal_is_nonfatal_but_never_silently_dropped(self):
        lines=['|gen|3','|switch|p1a: Mew|Mew|100/100','|unsupported|x','|turn|1','|request|{}']
        with patch.object(comparison.v4_encoder,'wire_kwargs',return_value={}),\
                patch.object(comparison.public_wire,'project_line',side_effect=writer) as projected:
            result=comparison.reference_words(lines,None,constructor=True)
            self.assertEqual(projected.call_count,3)
            self.assertIsNone(result['words'])
            self.assertEqual(result['refused_lines'][0]['kind'],'unsupported')
            self.assertEqual(result['setup_kinds_outside_frame'],{'gen':1})
            self.assertEqual(result['private_request_lines'],{'request':1})
            self.assertEqual(result['projected_word_count'],2)
            missing=comparison.reference_words(['|gen|3'],None,constructor=True)
            self.assertEqual(missing['refused_lines'][0]['kind'],'MISSING_INITIAL_SWITCH')

    def test_all_original_chunks_are_compared_and_unfinished_or_holey_capture_refuses(self):
        words=np.arange(2*64*16,dtype=np.int32).reshape(2,64,16)
        valid=np.ones((2,64),bool);valid[1,3:]=False
        more=np.array([True,False])
        self.assertEqual(len(comparison.engine_words(words,valid,more)),67)
        self.assertEqual(comparison.engine_words(words,valid,more)[64],words[1,0].tolist())
        for badvalid,badmore in ((valid,np.array([False,False])),
                (valid,np.array([True,True])),(valid.copy(),more)):
            if badvalid is not valid:badvalid[1,1]=False
            with self.assertRaises(ValueError):comparison.engine_words(words,badvalid,badmore)
        with self.assertRaises(ValueError):comparison.engine_words(*captured([]))

    def test_stream_and_all_173_column_comparisons_keep_mismatches_and_encoder_unknowns(self):
        lines=['|switch|p1a: Mew|Mew|100/100','|turn|1']
        raw=[[7]+[0]*15,[8]+[0]*15]
        actual, reference=observation(),observation()
        actual['numeric_features'][2,9]=.1
        actual['categorical_ids'][3,7]=1
        with patch.object(comparison.v4_encoder,'wire_kwargs',return_value={}),\
                patch.object(comparison.public_wire,'project_line',side_effect=writer):
            result=comparison.compare_owner(lines,captured(raw),None,constructor=True,
                actual_observation=actual,reference_observation=reference)
            self.assertEqual(result['status'],'MATCH')
            self.assertEqual(result['columns']['numeric_bad'][9],1)
            self.assertEqual(result['columns']['categorical_bad'][7],1)
            changed=[r[:] for r in raw];changed[1][4]=123
            result=comparison.compare_owner(lines,captured(changed),None,constructor=True)
            self.assertEqual(result['status'],'MISMATCH')
            self.assertEqual(result['first_difference']['index'],1)
            actual['legal_action_mask'][:]=False
            result=comparison.compare_owner(lines,captured(raw),None,constructor=True,
                actual_observation=actual,reference_observation=reference)
            self.assertEqual(result['status'],'MATCH')
            self.assertEqual(result['columns']['reason'],'ENCODER_REFUSED')

    def test_fixed_roster_conditional_rate_unknown_bounds_and_columns_never_normalize_refusals(self):
        col=dict(status='COMPARED',numeric_bad=[1]+[0]*131,numeric_max=[.1]+[0]*131,
            categorical_bad=[0]*41)
        rows=[dict(root_id=10,state_status='AGREE',owners=[dict(status='MATCH',columns=col),
                dict(status='UNKNOWN',refusals=[dict(kind='-status')],columns=dict(reason='ENCODER_REFUSED'))]),
            dict(root_id=11,state_status='DIVERGE',owners=[dict(status='MATCH'),dict(status='MATCH')]),
            dict(root_id=12,state_status='UNKNOWN')]
        result=comparison.reduce_cohort(rows,root_ids=[10,11,12,13])
        self.assertEqual((result['original_roots'],result['original_owner_views'],result['not_attempted_roots']),(4,8,1))
        conditional=result['conditional_on_measured_state_agreement']
        self.assertEqual((conditional['owner_views'],conditional['matched'],conditional['unknown']),(2,1,1))
        self.assertEqual(conditional['identified_match_rate'],1.)
        self.assertEqual(conditional['match_fraction_bounds'],[.5,1.])
        self.assertEqual(result['refusals_by_event_kind'],{'-status':1})
        self.assertEqual(len(result['numeric_columns']),132)
        self.assertEqual(len(result['categorical_columns']),41)
        self.assertEqual(result['numeric_columns'][0]['unknown_cells'],7*23)
        self.assertEqual(result['numeric_columns'][0]['expected_cells'],8*23)
        self.assertEqual(result['numeric_columns'][0]['mismatched_cells'],1)
        self.assertFalse(result['full_B4_qualified'])
        self.assertIsNone(comparison.reduce_cohort([],root_ids=[10])[
            'conditional_on_measured_state_agreement']['identified_match_rate'])
        with self.assertRaises(ValueError):comparison.reduce_cohort(rows+rows[:1],root_ids=[10,11,12,13])
        malformed=dict(col,numeric_bad=[1])
        with self.assertRaisesRegex(ValueError,'per-column'):
            comparison.reduce_cohort([dict(root_id=10,state_status='AGREE',
                owners=[dict(status='MATCH',columns=malformed)])],root_ids=[10])

    def test_collector_continues_next_fixed_roots_after_native_failure_without_games_in_test(self):
        State=namedtuple('State','turn gender happiness')
        state=State(np.zeros(3,np.int32),np.ones((3,2,6),np.int32),np.full((3,2,6),255,np.int32))
        pool=NS(arrays={'species':np.ones((6,6),np.int32)},metadata=lambda:dict(vocabulary={'species':['_none','mew']}))
        allcaptures=tuple(np.stack([np.stack([part,part]) for _ in range(3)]) for part in captured([[7]+[0]*15]))
        attempts=[]
        class FakeEnv:
            def __init__(self,_):
                self.index=len(attempts);attempts.append(self)
                self.owner_transcripts=[['|switch|p1a: Mew|Mew|100/100']]*2
                self.seed=None
            def reset_pool_sets(self,packed,*,seed):
                self.seed=seed
                if self.index==0:raise RuntimeError('authored native failure')
            def snapshot(self):return object()
            def close(self):
                if self.index==2:raise RuntimeError('authored close failure')
        with patch.object(comparison,'OwningShowdownEnv',FakeEnv),\
                patch.object(comparison,'pool_sets',return_value=[[],[]]),\
                patch.object(comparison,'pack_sets',return_value=['packed1','packed2']),\
                patch.object(comparison.fidelity,'compare',return_value=[]),\
                patch.object(comparison.v4_encoder,'wire_kwargs',return_value={}),\
                patch.object(comparison.public_wire,'project_line',side_effect=writer):
            result=comparison.collect_constructors(pool,[[0,1],[2,3],[4,5]],state,allcaptures,None,
                root_ids=[101,102,103],showdown_root='/not-opened',node='not-started',native_seed_start=700)
        self.assertEqual([env.seed for env in attempts],[700,701,702])
        self.assertEqual([r['state_status'] for r in result['rows']],['UNKNOWN','AGREE','AGREE'])
        self.assertEqual(result['summary']['attempted_roots'],3)
        self.assertEqual(result['summary']['conditional_on_measured_state_agreement']['matched'],4)
        self.assertEqual(result['rows'][0]['error']['type'],'RuntimeError')
        self.assertIn('close_error',result['rows'][2])


if __name__ == '__main__':
    unittest.main()
