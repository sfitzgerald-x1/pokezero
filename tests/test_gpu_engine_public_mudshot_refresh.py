import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class MudShotRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataclasses import replace
        from types import SimpleNamespace
        from test_gpu_engine_v4_encoder import V4EncoderTests
        from gpu_engine import public_belief,public_mudshot,public_mudshot_refresh
        from pokezero.category_vocab import CategoryVocabulary
        V4EncoderTests.setUpClass();f=V4EncoderTests();raw=f.public.source;universes=dict(raw.universes)
        old=universes['absol']
        universes['absol']=replace(old,variants=tuple(replace(old.variants[0],variant_id='mud'+str(i),ability=name)
            for i,name in enumerate(('Shield Dust','Pressure'))))
        raw=replace(raw,universes=universes);source=public_belief.compile_source(raw)
        vocab=CategoryVocabulary(tuple(sorted(set(f.vocab.tokens)|{'ability:shielddust','belief:possible_ability:shielddust'})))
        f.catalog=f.e.compile_encoder(source,f.catalog.dex,vocab,checkpoint_sha256='a'*64,feature_masks=f.masks)
        f.source,f.vocab=source,vocab;f.public=SimpleNamespace(source=raw)
        cls.f,cls.m,cls.x,cls.b,cls.j,cls.jax=f,public_mudshot,public_mudshot_refresh,public_belief,f.j,f.jax
        cls.tables=cls.m.compile_mudshot(source,f.catalog.species,f.catalog.abilities)
        cls.prefix=['|switch|p1a: Snorlax|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100']

    def initial(self):
        # This component receives a typed pending record; raw move scheduling
        # remains separate. The public HP/ability basis is the actual prefix.
        state=self.f.ingest([*self.prefix,'|-damage|p2a: Absol|80/100'])
        pending=self.m.initialize(1,self.tables)._replace(side=self.j.array([1],self.j.int32),
            slot=self.j.array([0],self.j.int32),encounter=self.j.array([1],self.j.int32),saw_damage=self.j.array([True]))
        return state,pending

    def test_authoritative_claim_and_checkpoint_match_source_even_when_borrowing_or_pinned(self):
        import numpy as np
        from gpu_engine.public_context import basis
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.showdown import _public_event_from_line
        for borrowed,pin in ((False,False),(True,False),(False,True)):
            state,pending=self.initial()
            engine=PublicBattleBeliefEngine.from_events([_public_event_from_line(line) for line in (
                *self.prefix,'|move|p1a: Snorlax|Mud Shot|p2a: Absol','|-damage|p2a: Absol|80/100')],
                format_id='gen3randombattle',set_source=self.f.public.source)
            if borrowed:
                pokemon=engine.snapshot().side('p2')[0]
                engine._running_ability[pokemon.key]='Pressure'
                abilities=state.context.abilities._replace(running=state.context.abilities.running.at[0,1,0].set(
                    self.f.catalog.abilities.ability_id('Pressure')))
                state=state._replace(context=state.context._replace(abilities=abilities))
            if pin:
                survivor=next(v for v in self.f.source.variants[self.f.source.species_id('Absol')] if v.ability=='Pressure').to_summary()
                self.assertTrue(engine.narrow_candidate_variants('p2:absol',[survivor]))
                beliefs,_=self.b.narrow_pin(state.beliefs,self.f.source.pin_mask('Absol',[survivor]),
                    valid=self.j.zeros_like(state.beliefs.species,bool).at[0,0,0].set(True))
                state=state._replace(beliefs=beliefs)
            result=self.jax.jit(lambda c,n,b:self.x.resolve(c,n,b,state.mons,self.f.catalog.source.device,
                self.f.catalog.context,self.tables,producer_binding=self.tables.binding))(state.context,pending,basis(state.context,state.beliefs))
            self.assertEqual(result.admitted.tolist(),[True]);self.assertEqual(result.revealed.tolist(),[True])
            engine._resolve_pending_mudshot();pokemon=engine.snapshot().side('p2')[0]
            self.assertEqual(pokemon.revealed_ability,'Shield Dust')
            self.assertEqual(int(result.context.abilities.original[0,1,0]),int(self.tables.shield_dust_id))
            self.assertEqual(int(result.context.abilities.running[0,1,0]),int(state.context.abilities.running[0,1,0]))
            cache=result.context.refresh.controller.visible
            self.assertEqual(int(cache.generation[0,0,0]),int(state.context.refresh.controller.visible.generation[0,0,0])+1)
            self.assertEqual(int(cache.summary.count[0,0,0]),pokemon.candidate_set_count)
            np.testing.assert_array_equal(cache.summary.effective[0,0,0],self.f.source.pin_mask('Absol',pokemon.candidate_variants))
            self.assertEqual(int(result.beliefs.conflicts[0,0,0]),engine.variant_pin_conflicts.get('p2:absol',0))
            self.assertEqual(result.state.side.tolist(),[-1])

    def test_cached_guard_and_species_or_binding_refusals_preserve_state(self):
        import numpy as np
        from gpu_engine.public_context import basis
        state,pending=self.initial();base=basis(state.context,state.beliefs)
        for mons,binding,valid,want in ((state.mons,self.tables.binding,False,True),
                (state.mons._replace(source_species=state.mons.source_species.at[0,1,0].set(0)),self.tables.binding,True,False),
                (state.mons,self.tables.binding.at[0].set(self.tables.binding[0]^self.j.uint32(1)),True,False)):
            result=self.x.resolve(state.context,pending,base,mons,self.f.catalog.source.device,self.f.catalog.context,
                self.tables,producer_binding=binding,valid=valid)
            self.assertEqual(result.admitted.tolist(),[want]);self.assertFalse(bool(result.revealed.any()))
            for new,old in ((result.context,state.context),(result.state,pending),(result.basis,base)):
                for a,b in zip(self.jax.tree.leaves(new),self.jax.tree.leaves(old)):np.testing.assert_array_equal(a,b)
        cancelled=pending._replace(cancelled=self.j.array([True]))
        result=self.x.resolve(state.context,cancelled,base,state.mons,self.f.catalog.source.device,self.f.catalog.context,
            self.tables,producer_binding=self.tables.binding)
        self.assertEqual(result.admitted.tolist(),[True]);self.assertEqual(result.revealed.tolist(),[False])
        self.assertEqual(result.state.side.tolist(),[-1])
        for a,b in zip(self.jax.tree.leaves(result.context),self.jax.tree.leaves(state.context)):np.testing.assert_array_equal(a,b)
