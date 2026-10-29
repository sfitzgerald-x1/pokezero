import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class NonprocRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataclasses import replace
        from types import SimpleNamespace
        from test_gpu_engine_v4_encoder import V4EncoderTests
        from gpu_engine import public_belief,public_item_nonproc,public_nonproc_refresh
        from pokezero.category_vocab import CategoryVocabulary
        V4EncoderTests.setUpClass();f=V4EncoderTests();raw=f.public.source;universes=dict(raw.universes)
        old=universes['snorlax'];names=(*public_item_nonproc.NAMES,'choiceband')
        universes['snorlax']=replace(old,variants=tuple(replace(old.variants[0],variant_id='nonproc'+str(i),item=name)
            for i,name in enumerate(names)))
        raw=replace(raw,universes=universes);source=public_belief.compile_source(raw)
        vocab=CategoryVocabulary(tuple(sorted(set(f.vocab.tokens)|{'belief:possible_item:'+n for n in names})))
        f.catalog=f.e.compile_encoder(source,f.catalog.dex,vocab,checkpoint_sha256='a'*64,feature_masks=f.masks)
        f.source,f.vocab=source,vocab;f.public=SimpleNamespace(source=raw,prefix=('|switch|p1a: Snorlax|Snorlax, L80|100/100',
            '|switch|p2a: Absol|Absol, L80|100/100'))
        cls.f,cls.n,cls.x,cls.b=f,public_item_nonproc,public_nonproc_refresh,public_belief
        cls.j,cls.jax=f.j,f.jax
        cls.tables=cls.n.compile_nonproc(source,f.catalog.species,f.catalog.items,f.catalog.context.sleep)
        cls.lines=[*f.public.prefix,'|-damage|p1a: Snorlax|20/100','|-status|p1a: Snorlax|par','|-damage|p2a: Absol|10/100']

    def initial(self):
        state=self.f.ingest(self.lines)
        ledger=self.n.initialize(1,species_binding=self.tables.species_binding)._replace(encounter=state.mons.encounter,
            hp=state.mons.hp_numerator,maximum=state.mons.hp_denominator)
        return state,ledger

    def test_each_upkeep_checkpoint_matches_actual_source_including_standing_pin(self):
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.showdown import _public_event_from_line
        from gpu_engine.public_context import basis
        for pin in (False,True):
            state,ledger=self.initial();engine=PublicBattleBeliefEngine.from_events([_public_event_from_line(line) for line in self.lines],
                format_id='gen3randombattle',set_source=self.f.public.source)
            if pin:
                survivor=next(v for v in self.f.source.variants[self.f.source.species_id('Snorlax')] if v.item=='leftovers').to_summary()
                self.assertTrue(engine.narrow_candidate_variants('p1:snorlax',[survivor]))
                mask=self.j.zeros_like(state.beliefs.species,bool).at[0,1,0].set(True)
                beliefs,_=self.b.narrow_pin(state.beliefs,self.f.source.pin_mask('Snorlax',[survivor]),valid=mask)
                state=state._replace(beliefs=beliefs)
            calls=[];original=engine._replace_belief
            def record(pokemon,**changes):
                result=original(pokemon,**changes)
                if 'ruled_out_items' in changes:calls.append(result)
                return result
            engine._replace_belief=record
            result=self.jax.jit(lambda c,n,b:self.x.upkeep(c,n,b,state.mons,self.f.catalog.source.device,
                self.f.catalog.context,self.tables,producer_binding=self.tables.binding))(state.context,ledger,basis(state.context,state.beliefs))
            self.assertEqual(result.admitted.tolist(),[True]);engine.ingest_event(_public_event_from_line('|upkeep'))
            scheduled=[(side,stage) for side in (0,1) for stage in range(3) if result.callbacks[0,side,stage]]
            self.assertEqual(len(scheduled),len(calls));self.assertEqual(len(calls),5)
            for (side,stage),pokemon in zip(scheduled,calls):
                self.assertEqual(pokemon.showdown_slot,f'p{side+1}');index=side*3+stage
                self.assertEqual(int(result.counts[0,index,1-side,0]),pokemon.candidate_set_count)
                self.assertAlmostEqual(float(result.summaries.uncertainty[0,index,1-side,0]),pokemon.uncertainty,places=6)
                import numpy as np
                np.testing.assert_array_equal(result.summaries.effective[0,index,1-side,0],
                    self.f.source.pin_mask(pokemon.species,pokemon.candidate_variants))
            for side in (0,1):
                key=f'p{side+1}:'+('snorlax' if side==0 else 'absol')
                self.assertEqual(int(result.beliefs.conflicts[0,1-side,0]),engine.variant_pin_conflicts.get(key,0))
                expected=sum(owner==side for owner,_ in scheduled)
                self.assertEqual(int(result.context.refresh.controller.visible.generation[0,1-side,0]),
                    int(state.context.refresh.controller.visible.generation[0,1-side,0])+expected)
            self.assertFalse(bool(result.state.maximum.any()))

    def test_bad_namespace_and_disabled_rows_do_not_mutate_cache_or_ledger(self):
        import numpy as np
        from gpu_engine.public_context import basis
        state,ledger=self.initial();base=basis(state.context,state.beliefs)
        for binding,valid in ((self.tables.binding.at[0].set(self.tables.binding[0]^self.j.uint32(1)),True),(self.tables.binding,False)):
            result=self.x.upkeep(state.context,ledger,base,state.mons,self.f.catalog.source.device,self.f.catalog.context,
                self.tables,producer_binding=binding,valid=valid)
            self.assertEqual(result.admitted.tolist(),[not valid]);self.assertFalse(bool(result.callbacks.any()))
            for a,b in zip(self.jax.tree.leaves(result.context),self.jax.tree.leaves(state.context)):np.testing.assert_array_equal(a,b)
            for a,b in zip(self.jax.tree.leaves(result.state),self.jax.tree.leaves(ledger)):np.testing.assert_array_equal(a,b)

    def test_stale_persistent_exclusion_cannot_skip_a_source_callback(self):
        import numpy as np
        from gpu_engine.public_context import basis
        state,ledger=self.initial();base=basis(state.context,state.beliefs)
        stale=ledger._replace(excluded=ledger.excluded.at[0,0,0,0].set(True))
        result=self.x.upkeep(state.context,stale,base,state.mons,self.f.catalog.source.device,self.f.catalog.context,
            self.tables,producer_binding=self.tables.binding)
        self.assertEqual(result.admitted.tolist(),[False]);self.assertFalse(bool(result.callbacks.any()))
        for a,b in zip(self.jax.tree.leaves(result.context),self.jax.tree.leaves(state.context)):np.testing.assert_array_equal(a,b)
        for a,b in zip(self.jax.tree.leaves(result.state),self.jax.tree.leaves(stale)):np.testing.assert_array_equal(a,b)

    def test_species_and_source_availability_join_refuses_only_bad_batch_rows(self):
        import numpy as np
        from gpu_engine.public_context import basis
        state,ledger=self.initial()
        repeat=lambda x:self.j.repeat(x,8,axis=0) if x.ndim and x.shape[0]==1 else x
        state,ledger=self.jax.tree.map(repeat,(state,ledger));base=basis(state.context,state.beliefs)
        other=self.f.source.species_id('Absol')
        mons=state.mons._replace(source_species=state.mons.source_species.at[1,0,0].set(other).at[5,0,0].set(other))
        base=base._replace(species=base.species.at[2,1,0].set(other),
            source_available=base.source_available.at[3,1,0].set(False))
        visible=state.context.refresh.controller.visible
        visible=visible._replace(source_available=visible.source_available.at[4,1,0].set(False))
        controller=state.context.refresh.controller._replace(visible=visible)
        context=state.context._replace(refresh=state.context.refresh._replace(controller=controller))
        valid=self.j.array([True,True,True,True,True,False,True,True])
        result=self.jax.jit(lambda c,n,b,m,v:self.x.upkeep(c,n,b,m,self.f.catalog.source.device,
            self.f.catalog.context,self.tables,producer_binding=self.tables.binding,valid=v))(context,ledger,base,mons,valid)
        self.assertEqual(result.admitted.tolist(),[True,False,False,False,False,True,True,True])
        self.assertEqual(result.callbacks.sum(axis=(1,2)).tolist(),[5,0,0,0,0,0,5,5])
        for new,old in ((result.context,context),(result.state,ledger),(result.basis,base)):
            for a,b in zip(self.jax.tree.leaves(new),self.jax.tree.leaves(old)):
                if a.ndim and a.shape[0]==8:
                    np.testing.assert_array_equal(a[1:6],b[1:6])
                else:np.testing.assert_array_equal(a,b)
        for bad in (mons._replace(source_species=mons.source_species.astype(self.j.float32)),
                mons._replace(species=mons.species[:,:,:1])):
            with self.assertRaisesRegex(ValueError,'checkpoint shape/dtype'):
                self.x.upkeep(context,ledger,base,bad,self.f.catalog.source.device,self.f.catalog.context,
                    self.tables,producer_binding=self.tables.binding)
