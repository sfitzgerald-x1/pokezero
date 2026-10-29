"""Substitute initial-size source parity and namespace/own-HP isolation."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicSubstituteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_v4_encoder import V4EncoderTests
        from gpu_engine import public_substitute
        V4EncoderTests.setUpClass();cls.f=V4EncoderTests();cls.s=public_substitute
        cls.j,cls.jax,cls.catalog=cls.f.j,cls.f.jax,cls.f.catalog
        cls.tables=cls.s.compile_substitute(cls.catalog.active,cls.catalog.species,cls.f.vocab)

    def initial(self):
        return self.f.ingest([*self.f.public.prefix,'|-start|p1a: Snorlax|Substitute','|-start|p2a: Absol|Substitute'])

    def packet(self,request,owner):
        from gpu_engine import private_team
        p=private_team.parse_team_facts(request,owner_side=owner,vocabulary=self.f.vocab,dex=self.catalog.dex)
        return self.jax.tree.map(lambda x:self.j.stack([x]),p)

    def test_both_source_seats_private_active_order_and_exact_mask(self):
        import json
        import numpy as np
        from dataclasses import replace
        from pokezero import showdown as source
        state=self.initial()
        lines=[*self.f.public.prefix,'|-start|p1a: Snorlax|Substitute','|-start|p2a: Absol|Substitute']
        project=self.jax.jit(lambda a,m,p,o,x:self.s.project(a,self.catalog.active.device,m,p,self.tables,
            observer_side=o,expected_binding=self.tables.binding,exact_state=x))
        for owner in (0,1):
            for flip in (False,True):
                for condition in ('123/247','0 fnt','123/247.0'):
                    request=self.f.request(owner)
                    for i,row in enumerate(request['side']['pokemon']):
                        row['condition']=condition if i==1 else '125/253'
                        row['active']=(i==0) if flip else (i==1)
                    packet=self.packet(request,owner)
                    replay=source.parse_showdown_replay([*lines,'|request|'+json.dumps(request)],battle_id='substitute-fixture')
                    normalized=source.normalize_for_player(replay,player_id=f'p{owner+1}',configured_showdown_slot=f'p{owner+1}',
                        format_id='gen3randombattle',set_source=self.f.public.source)
                    for exact in (False,True):
                        got=project(state.context.active,state.mons,packet,self.j.array([owner],self.j.int32),self.j.array([exact]))
                        self.assertEqual(got.own.admitted.tolist(),[True]);self.assertEqual(got.opponent.admitted.tolist(),[True])
                        reference=source.observation_from_player_state(normalized,category_vocab=self.f.vocab,
                            spec=source.V4_REPLAY_OBSERVATION_SPEC,dex=self.catalog.dex,
                            feature_masks=replace(source.DEFAULT_OBSERVATION_FEATURE_MASKS,exact_state=exact,transition_token_budget=0))
                        np.testing.assert_allclose(np.asarray(got.own.numeric[0,:,self.s.COLUMN]),
                            np.asarray(reference.numeric_features)[1:7,self.s.COLUMN],atol=1e-6,rtol=0)
                        np.testing.assert_allclose(np.asarray(got.opponent.numeric[0,:,self.s.COLUMN]),
                            np.asarray(reference.numeric_features)[7:13,self.s.COLUMN],atol=1e-6,rtol=0)

    def test_mixed_namespace_owner_hp_refusal_and_public_fixed_fraction(self):
        import numpy as np
        state=self.initial();repeat=lambda x:self.j.repeat(x,8,axis=0)
        active=self.jax.tree.map(repeat,state.context.active);mons=self.jax.tree.map(repeat,state.mons)
        packet=self.jax.tree.map(repeat,self.packet(self.f.request(),0))
        run=self.jax.jit(lambda a,m,p:self.s.project(a,self.catalog.active.device,m,p,self.tables,
            observer_side=self.j.zeros(8,self.j.int32),expected_binding=self.tables.binding))
        result=run(active,mons,packet)
        np.testing.assert_array_equal(np.asarray(result.opponent.numeric[:,:,self.s.COLUMN]),np.tile([.25,0,0,0,0,0],(8,1)))
        for bad in (packet._replace(vocabulary_binding=packet.vocabulary_binding.at[0,0].set(packet.vocabulary_binding[0,0]^self.j.uint32(1))),
                    packet._replace(owner_side=packet.owner_side.at[0].set(1)),
                    packet._replace(actual_stats=packet.actual_stats.at[0,1,0].set(self.j.nan)),
                    packet._replace(actual_stats=packet.actual_stats.at[0,1,0].set(247.5))):
            got=run(active,mons,bad)
            self.assertEqual(got.own.admitted.tolist(),[False]+[True]*7)
            self.assertFalse(bool(got.own.numeric[0].any()));self.assertFalse(bool(got.opponent.numeric[0].any()))
            np.testing.assert_array_equal(got.opponent.numeric[1:],result.opponent.numeric[1:])
        with self.assertRaisesRegex(ValueError,'shape/dtype'):
            run(active,mons,packet._replace(actual_stats=packet.actual_stats.astype(self.j.int32)))


if __name__=='__main__':unittest.main()
