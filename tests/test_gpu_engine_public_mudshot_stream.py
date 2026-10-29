import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class MudShotStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataclasses import replace
        from types import SimpleNamespace
        from test_gpu_engine_v4_encoder import V4EncoderTests
        from pokezero.dex import MoveInfo,ShowdownDex
        from pokezero.category_vocab import CategoryVocabulary
        from gpu_engine import public_belief
        V4EncoderTests.setUpClass();f=V4EncoderTests();raw=f.public.source;universes=dict(raw.universes)
        old=universes['absol']
        universes['absol']=replace(old,variants=tuple(replace(old.variants[0],variant_id='mudstream'+str(i),ability=name,item=item)
            for i,(name,item) in enumerate((('Shield Dust','Leftovers'),('Pressure','Choice Band')))))
        raw=replace(raw,universes=universes);source=public_belief.compile_source(raw)
        mud=MoveInfo('mudshot','Mud Shot','Ground','Special','Physical',55,95,0,False,False,False,None,{},'normal',False,
            effect_chance=100,effect_label='lower_foe_spe',pp=15)
        dex=ShowdownDex(dict(f.catalog.dex.moves,mudshot=mud),dict(f.catalog.dex.species),dict(f.catalog.dex.type_chart))
        labels={'move:mudshot','belief:possible_move:mudshot','move_effect:lower_foe_spe','ability:shielddust','belief:possible_ability:shielddust',
            'belief:possible_item:choiceband','belief:possible_item:leftovers'}
        vocab=CategoryVocabulary(tuple(sorted(set(f.vocab.tokens)|labels)))
        f.catalog=f.e.compile_encoder(source,dex,vocab,checkpoint_sha256='a'*64,feature_masks=f.masks)
        f.source,f.vocab=source,vocab;f.public=SimpleNamespace(source=raw)
        cls.f,cls.j,cls.jax=f,f.j,f.jax
        step=f.jax.jit(lambda s,c:f.e.fold_chunk(s,c,f.catalog))
        def ingest(lines):
            state=f.e.initialize(f.catalog,stream_ids=f.j.array([1],f.j.int32))
            for chunk in f.chunks(f.projections(lines)):
                state,error=step(state,chunk);f.assertEqual(error.tolist(),[0])
            return state
        f.ingest=ingest
        cls.prefix=['|switch|p1a: Snorlax|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100']

    def test_raw_mudshot_resolves_before_next_actor_and_matches_all_source_columns(self):
        lines=[*self.prefix,'|move|p1a: Snorlax|Mud Shot|p2a: Absol','|-damage|p2a: Absol|80/100',
            '|move|p2a: Absol|Surf|p1a: Snorlax']
        state=self.f.ingest(lines)
        dust=self.f.catalog.abilities.ability_id('Shield Dust')
        self.assertEqual(int(state.context.abilities.original[0,1,0]),dust)
        self.assertEqual(state.context.mudshot.side.tolist(),[-1])
        self.f.compare_source(lines)

    def test_post_nonproc_cache_tagged_damage_and_any_unboost_suppress_claims(self):
        import numpy as np
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.showdown import _public_event_from_line
        move='|move|p1a: Snorlax|Mud Shot|p2a: Absol'
        cases=(['|-damage|p2a: Absol|80/100','|upkeep'],
            ['|-damage|p2a: Absol|80/100|[from] psn','|move|p2a: Absol|Surf|p1a: Snorlax'],
            ['|-damage|p2a: Absol|80/100','|-unboost|p2a: Absol|def|1','|move|p2a: Absol|Surf|p1a: Snorlax'])
        for suffix in cases:
            lines=[*self.prefix,move,*suffix];state=self.f.ingest(lines)
            engine=PublicBattleBeliefEngine.from_events([_public_event_from_line(line) for line in lines],
                format_id='gen3randombattle',set_source=self.f.public.source)
            peer=engine.snapshot().side('p2')[0]
            self.assertFalse(peer.revealed_ability)
            self.assertEqual(int(state.context.abilities.original[0,1,0]),0)
            self.assertEqual(state.context.mudshot.side.tolist(),[-1])
            self.assertIsNone(engine._pending_mudshot)
            cache=state.context.refresh.controller.visible
            self.assertEqual(int(cache.summary.count[0,0,0]),peer.candidate_set_count)
            np.testing.assert_array_equal(cache.summary.effective[0,0,0],self.f.source.pin_mask('Absol',peer.candidate_variants))
        # The first suffix removed the Shield Dust/Leftovers variant before
        # resolving the pending secondary, leaving only Pressure/Choice Band.
        state=self.f.ingest([*self.prefix,move,*cases[0]])
        self.assertEqual(int(state.context.refresh.controller.visible.summary.count[0,0,0]),1)

    def test_called_and_transformed_mudshot_queue_before_next_real_move(self):
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.showdown import _public_event_from_line
        for transformed in (False,True):
            lines=[*self.prefix,*(['|-transform|p1a: Snorlax|p2a: Absol'] if transformed else []),
                '|move|p1a: Snorlax|Mud Shot|p2a: Absol|[from] move: Sleep Talk','|-damage|p2a: Absol|80/100']
            state=self.f.ingest(lines)
            engine=PublicBattleBeliefEngine.from_events([_public_event_from_line(line) for line in lines],
                format_id='gen3randombattle',set_source=self.f.public.source)
            self.assertIsNotNone(engine._pending_mudshot)
            self.assertEqual(state.context.mudshot.side.tolist(),[1])
            self.assertEqual(state.context.mudshot.saw_damage.tolist(),[True])
            self.assertTrue(engine._pending_mudshot['saw_damage'])
            final=[*lines,'|move|p2a: Absol|Surf|p1a: Snorlax'];state=self.f.ingest(final)
            self.assertEqual(int(state.context.abilities.original[0,1,0]),self.f.catalog.abilities.ability_id('Shield Dust'))
