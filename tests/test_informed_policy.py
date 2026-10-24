"""Gameplay seam contracts; fakes do not establish native fidelity or strength."""
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.informed_policy import InformedPolicyConfig, InformedRootPolicy
from pokezero.informed_root import InformedOnePlyResult, solve_informed_root_lp
from pokezero.shallow_exact import ExactSearchRefused


class Adapter:
    def __init__(self,**kwargs):
        self.kwargs=kwargs;self._dex='dex';self._set_source='source';self._module='engine'
        self._tables_json=json.dumps(dict(layout=dict(token_count=1,
            categorical_feature_count=2,numeric_feature_count=2)))
        self.stats=SimpleNamespace(worlds_attempted=0)
        self.reset_calls=0
    def reset(self):self.reset_calls+=1
    def _validate_model_root_observation(self,observation):pass
    def _advance_live_fold(self,context):return SimpleNamespace(to_payload=lambda:{'prefix':'public'})
    def _root_inputs_json(self,context):
        return json.dumps(dict(observation_metadata=dict(showdown_slot=context.player_id)))
    def _public_effect_signals(self,context):return {},{},{},{},{}
    def _recharging_slots(self,context):return ()
    def _truant_loaf_slots(self,context):return ()
    def _native(self):return 'model'
    def _choice_vocabulary(self,context):
        return SimpleNamespace(action_index=lambda a:{'first':0,'second':1}.get(a))


class Native:
    FoldState=SimpleNamespace(from_payload=lambda p:('fold',p))
    LeafEncoder=staticmethod(lambda *args:('encoder',args))


class Backend:
    def __init__(self,*args,**kwargs):self.stage_seconds=Counter()
    def actions(self,position,*,root):
        assert root
        return (('first','second'),('opponent',)) if position.self_side=='side_one' \
            else (('opponent',),('first','second'))


class InformedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'decisions.jsonl'
        self.patches=[]
        for patcher in (patch.dict(os.environ,POKEZERO_ENUMERATE_ROLLS='1'),
                        patch('pokezero.informed_policy.EngineMctsPolicy',Adapter),
                        patch('pokezero.informed_policy.NativeExactBackend',Backend)):
            patcher.start();self.addCleanup(patcher.stop)
        self.draws=[]
        def draw(**kwargs):
            self.draws.append(kwargs)
            return SimpleNamespace(player_teams={'p1':'our-known-team','p2':'sampled-team'}),None
        self.sampler=patch('pokezero.informed_policy._gen3_randbat_belief_start_override_result',
                           side_effect=draw).start()
        self.addCleanup(patch.stopall)
        self.world=SimpleNamespace(spec='spec',party_species={'p1':('own',),'p2':('sample',)},
            slot_sides={'p1':'side_one','p2':'side_two'})
        patch('pokezero.informed_policy.world_battle_spec',return_value=self.world).start()
        patch('pokezero.informed_policy.build_poke_engine_state',
              return_value=SimpleNamespace(to_string=lambda:'native-state')).start()

    def policy(self,**kwargs):
        result=InformedRootPolicy(dex='dex',set_source='source',checkpoint_path='checkpoint',
            model_path='model',tables_path='tables',device='cuda',trace_path=self.path,
            native=Native,**kwargs)
        self.addCleanup(result.close)
        return result

    def context(self,side='p1'):
        return SimpleNamespace(player_id=side,decision_round_index=7,battle_id='battle-17',
            seed=17,observation=SimpleNamespace(legal_action_mask=(True,True)),
            public_materialization_state=SimpleNamespace(replay=SimpleNamespace(turn_number=5)))

    def result(self,side=0):
        eq=solve_informed_root_lp([[[0],[1]]]*4)
        return InformedOnePlyResult(side,('first','second'),(('opponent',),)*4,
            (((0.,),(1.,)),)*4,eq,4,8,8,8,1,0.)

    def records(self):return [json.loads(s) for s in self.path.read_text().splitlines()]

    def test_four_draws_sample_then_map_exact_actor_action_and_persist(self):
        for side in (0,1):
            with self.subTest(side=side):
                if self.path.exists():self.path=Path(self.directory.name)/'second-seat.jsonl'
                p=self.policy()
                with patch('pokezero.informed_policy.exact_informed_one_ply',
                           return_value=self.result(side)) as search:
                    result=p.select_action_with_context(self.context('p1' if side==0 else 'p2'),
                                                        rng=random.Random(17))
                self.assertEqual(result.action_index,1)
                self.assertEqual(result.action_probability,1)
                self.assertEqual(p._adapter.stats.worlds_attempted,4)
                self.assertEqual(search.call_args.kwargs['acting_side'],side)
                record=self.records()[0]
                self.assertEqual(record['status'],'COMPLETE')
                self.assertEqual(record['sampling']['mix'],[0.,1.])
                self.assertEqual(record['sampling']['action'],'second')
                self.assertEqual(record['sampling']['support_floor'],'NONE')
                self.assertEqual(record['sampling']['sampling_domain'][-3:],[5,side,7])
                self.assertEqual(len(record['root_state_sha256']),4)
                self.assertNotIn('failure_witness',record)
                self.assertEqual(p.stats.completed_decisions,1)

    def test_failed_draw_never_retries_or_searches_surviving_worlds(self):
        p=self.policy()
        good=SimpleNamespace(player_teams={'p1':'own','p2':'sample'})
        self.sampler.side_effect=[(good,None),(None,'unsupported-bound')]
        with patch('pokezero.informed_policy.exact_informed_one_ply') as search:
            with self.assertRaisesRegex(ExactSearchRefused,'draw 1 refused'):
                p.select_action_with_context(self.context(),rng=random.Random(17))
            search.assert_not_called()
        self.assertEqual(self.sampler.call_count,2)
        row=self.records()[0]
        self.assertEqual(row['status'],'REFUSED')
        self.assertEqual(len(row['failure_witness']['draws']),2)
        self.assertEqual(row['failure_witness']['draws'][0]['state_str'],'native-state')
        self.assertEqual(row['failure_witness']['draws'][1]['refusal'],'unsupported-bound')
        self.assertIn('error_traceback',row)
        self.assertEqual(p.stats.refusals,1)
        self.assertEqual(json.loads(json.dumps(p.stats.to_dict()))['refusal_reasons'],
                         {'ExactSearchRefused':1})

    def test_construction_error_preserves_actual_draw_without_replacement(self):
        p=self.policy()
        with patch('pokezero.informed_policy.world_battle_spec',side_effect=ValueError('fixture condition')):
            with self.assertRaisesRegex(ValueError,'fixture condition'):
                p.select_action_with_context(self.context(),rng=random.Random(17))
        row=self.records()[0]
        self.assertEqual(self.sampler.call_count,1)
        self.assertEqual(row['failure_witness']['draws'][0]['player_teams']['p2'],'sampled-team')
        self.assertIn('rng_state_before_draw',row['failure_witness']['draws'][0])

    def test_native_baseexception_preserves_decision_witness_and_propagates(self):
        class AuthoredNativePanic(BaseException):pass
        p=self.policy()
        with patch('pokezero.informed_policy.exact_informed_one_ply',
                   side_effect=AuthoredNativePanic('authored panic')):
            with self.assertRaisesRegex(AuthoredNativePanic,'authored panic'):
                p.select_action_with_context(self.context(),rng=random.Random(17))
        row=self.records()[0]
        self.assertEqual(row['error_type'],'AuthoredNativePanic')
        self.assertEqual(len(row['failure_witness']['draws']),4)
        self.assertIn('root_inputs',row['failure_witness'])
        self.assertIn('fold_payload',row['failure_witness'])
        self.assertEqual(p.stats.refusals,1)

    def test_full_own_action_surface_disagreement_refuses_before_search(self):
        p=self.policy();context=self.context();context.observation.legal_action_mask=(True,False)
        with patch('pokezero.informed_policy.exact_informed_one_ply') as search:
            with self.assertRaisesRegex(ExactSearchRefused,'action surface disagrees'):
                p.select_action_with_context(context,rng=random.Random(17))
            search.assert_not_called()
        self.assertEqual(self.records()[0]['status'],'REFUSED')

    def test_reset_preserves_stats_and_trace_and_no_context_never_falls_back(self):
        p=self.policy();p.stats.decisions=3;p.reset()
        self.assertEqual(p.stats.decisions,3)
        self.assertEqual(p._adapter.reset_calls,1)
        with self.assertRaises(ExactSearchRefused):p.select_action(None,rng=random.Random(0))
        with self.assertRaises(FileExistsError):self.policy()

    def test_missing_public_context_or_fold_refuses_with_record(self):
        p=self.policy();context=self.context();context.public_materialization_state=None
        with self.assertRaisesRegex(ExactSearchRefused,'no public materialization'):
            p.select_action_with_context(context,rng=random.Random(0))
        self.assertEqual(self.records()[0]['failure_witness'],{})

    def test_config_bounds_and_fresh_process_flag(self):
        for changes in (dict(worlds=3),dict(worlds=True),dict(batch_size=4097),
                        dict(bulk_encode_workers=33),dict(max_seconds=0)):
            with self.assertRaises(ValueError):InformedPolicyConfig(**changes)
        with patch.dict(os.environ,POKEZERO_ENUMERATE_ROLLS='0'):
            with self.assertRaisesRegex(ExactSearchRefused,'fresh-process'):
                self.policy()

    def test_external_bridge_failure_is_durable_and_not_clean_completion(self):
        p=self.policy()
        p.record_failure('benchmark',ValueError('stream close fixture'))
        row=self.records()[0]
        self.assertEqual(row['schema'],'pokezero.informed-root-run-failure.v1')
        self.assertEqual(row['status'],'REFUSED')
        self.assertEqual(row['stage'],'benchmark')
        self.assertEqual(p.stats.refusals,1)
        self.assertEqual(p.stats.decisions,0)

    def test_root_encoder_keeps_existing_exact_toxic_provenance_handoff(self):
        p=self.policy();context=self.context()
        context.public_materialization_state.replay.toxic_stage_zero_after_upkeep={'p1':True,'p2':'bad'}
        positions,turn=p._positions(context,random.Random(17))
        ctx=json.loads(positions[0].encoder[1][2])
        self.assertIs(ctx['toxic_stage_zero_after_upkeep']['p1']['proof'],True)
        self.assertIsNone(ctx['toxic_stage_zero_after_upkeep']['p2']['proof'])


if __name__=='__main__':unittest.main()
