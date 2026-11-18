"""Owning-evidence transport and whole-team sampler integration."""
from dataclasses import replace
import hashlib
import json
import random
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pokezero import determinization as D,set_posterior as SP
from pokezero.joint_serving import JointSession,opponent_team,JointPlan
from pokezero.joint_generator import JointMember
from tests.test_depthn_set_posterior_worlds import ProtocolRoute
from tests.test_depthn_set_posterior import fixture


class JointProtocol(ProtocolRoute):
    # Reuse root/source construction, not inherited verification cases.
    def test_joint_draw_bypasses_both_independent_draws(self):
        _,root=self.root('cb_collapse')
        source=self.src
        (original,_),_=self.start(root,37)
        from pokezero.engine_world import unpack_team
        view=D.player_belief_view_from_payload(root.context.observation.metadata['belief_view'])
        fixtures=unpack_team(original.player_teams[view.opponent_slot])
        with patch.object(D,'_opponent_team_from_belief_result',side_effect=AssertionError('second draw')):
            start,reason=D._gen3_randbat_belief_start_override_result(context=root.context,set_source=source,
                rng=random.Random(37),joint_opponent_team=lambda rng:fixtures)
        self.assertIsNone(reason)
        self.assertEqual(start.player_teams[view.opponent_slot],original.player_teams[view.opponent_slot])

    def test_generation_slot_is_opening_lead_only(self):
        self.assertEqual(JointSession._generation_slots(('|switch|p2a: A|Ampharos, L83|100/100',
            '|switch|p2a: B|Snorlax, L70|100/100','|drag|p2a: C|Rhydon, L80|100/100'),'p2'),{0:'ampharos'})

    def test_feature_off_does_not_read_or_mutate_payload(self):
        class Poison:
            def __getattribute__(self,name):raise AssertionError(name)
        self.assertFalse(SP.apply_feature_payload(Poison(),Poison(),enabled=False,game_sha256='x',owner='p1'))

    def test_original_team_mapping_and_explicit_fallback(self):
        data,root=self.root('cb_collapse')
        session=JointSession(self.src,producer='shared-battle')
        own=session._own_initial(data['request']['lines'])
        self.assertEqual(len(own),6)
        session._own_initial=lambda lines: (_ for _ in ()).throw(ValueError('custom own team'))
        plan=session.prepare(root.context,data['request']['lines'],dex=self.dex,seed=5,count=16,features=True)
        self.assertIsNone(plan.teams)
        self.assertEqual(plan.diagnostic['mode'],'stage1_fallback')
        SP.feature_rows(plan.feature_payload,self.src,game_sha256=plan.game_sha256)
        broken=json.loads(json.dumps(plan.feature_payload));broken['rows'][next(iter(broken['rows']))]['weights'][0]+=1
        with self.assertRaisesRegex(ValueError,'provenance'):SP.feature_rows(broken,self.src,game_sha256=plan.game_sha256)
