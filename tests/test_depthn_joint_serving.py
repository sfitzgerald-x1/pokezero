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


class JointEvidenceArithmetic(unittest.TestCase):
    def test_fallback_features_use_the_actual_restricted_stage1_law(self):
        session=JointSession.__new__(JointSession)
        variants=tuple(NS(item='Leftovers',to_summary=lambda i=i:{'index':i})for i in range(2))
        session.source=NS(universe_for=lambda name:NS(variants=variants))
        belief=NS(species='a',condition='42/42',gender=None,original_public_item=None)
        view=NS(opponent_slot='p2',opponent_pokemon=(belief,))
        post=SP.EncounterPosterior('p2','a',(True,True),(True,False),(3,0),(),(),'fixture')
        fallback={}
        with patch.object(D,'_public_opponent_move_slot_constraints',return_value={}), \
             patch.object(D,'_condition_max_hp',return_value=42), \
             patch.object(D,'_variant_public_max_hp',side_effect=lambda payload,**kw:42 if payload['index']==1 else 99):
            evidence=session._evidence(None,view,{'a':post},stage1_posts=fallback)
        self.assertEqual(evidence,{'a':(0,0)})
        self.assertEqual(fallback['a'].weights,(0,1))
        self.assertEqual(fallback['a'].hard,post.hard)


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

    def test_original_named_item_conditions_joint_evidence(self):
        data,root=self.root('refusal_item_changed')
        session=JointSession(self.src,producer='shared-battle')
        view=D.player_belief_view_from_payload(root.context.observation.metadata['belief_view'])
        posts=SP.posterior(data['request']['lines'],None,source=self.src,owner=view.self_slot,
            beliefs=view.opponent_pokemon,format_id=root.context.format_id,dex=self.dex)
        first=view.opponent_pokemon[0];variants=self.src.universe_for(first.species).variants
        named=variants[0].item
        changed=replace(first,item_mutated=True,original_public_item=named)
        view=replace(view,opponent_pokemon=(changed,*view.opponent_pokemon[1:]))
        from pokezero.randbat import canonical_gen3_randbat_species_id
        from pokezero.dex import normalize_id
        before=session._evidence(root.context,D.player_belief_view_from_payload(root.context.observation.metadata['belief_view']),posts)
        after=session._evidence(root.context,view,posts)
        key=canonical_gen3_randbat_species_id(first.species)
        self.assertEqual(after[key],tuple(w if normalize_id(v.item or '')==normalize_id(named or '')else 0
            for v,w in zip(variants,before[key])))

    def test_joint_materialization_preserves_public_cosmetic_species(self):
        _,root=self.root('hp_neutral_unown')
        names=['unown']+[s for s in self.src.universes if s!='unown'][:5]
        team=tuple(JointMember(s,0,i,0)for i,s in enumerate(names))
        plan=JointPlan((team,),{},None,{},self.src.metadata.source_hash,'fixture')
        fixtures=opponent_team(plan,0,context=root.context,source=self.src,rng=random.Random(11))
        self.assertEqual([f.species for f in fixtures if f.species.lower().startswith('unown')],['Unown-I'])

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
