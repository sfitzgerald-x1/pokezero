"""Intermediate item callbacks compared with the actual belief engine."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class ItemCallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_item_context import PublicItemContextTests
        from gpu_engine import public_item_callbacks
        PublicItemContextTests.setUpClass();cls.f=PublicItemContextTests();cls.x=public_item_callbacks
        cls.j,cls.jax=cls.f.j,cls.f.jax

    def compare(self,prefix,line,narrow=False):
        from pokezero import showdown
        import numpy as np
        f=self.f;initial,_=f.run_lines(prefix,narrow)
        engine=f.Reference.from_events([showdown._public_event_from_line(raw) for raw in prefix],
            format_id='gen3randombattle',set_source=f.source,item_belief_narrowing=narrow)
        calls=[];original=engine._with_set_summary;replace_belief=engine._replace_belief
        recording=[False]
        def replace_item(pokemon,**changes):
            previous=recording[0]
            recording[0]=bool(set(changes)&{'revealed_item','original_public_item','item_mutated','item_removed','current_public_item'})
            try:return replace_belief(pokemon,**changes)
            finally:recording[0]=previous
        def record(pokemon):
            result=original(pokemon)
            if recording[0]:calls.append(result)
            return result
        # The helper owns item-specific replacements. Common HP/status
        # callbacks on the same raw line belong to their installed consumer.
        engine._replace_belief=replace_item;engine._with_set_summary=record
        projection=f.items.project_line(line,items=f.item_catalog)
        state=initial.context.items;kwargs={}
        if projection.trick_idents:
            kwargs=dict(actor_side=0,actor_slot=0,actor_encounter=1,target_side=1,target_slot=0,target_encounter=1)
        state,ok=f.items.begin_event(state,initial.mon_state,f.item_catalog.device,
            item_event=projection.item_event,trick=bool(projection.trick_idents),**kwargs)
        self.assertTrue(bool(ok.all()));scheduled=[]
        for ident,operation,item in projection.facts:
            side=int(ident[1])-1
            result=self.jax.jit(lambda s:self.x.project(s,initial.mon_state,f.item_catalog.device,
                side=side,slot=0,encounter=1,item_id=item,operation=operation,
                item_binding=f.item_catalog.device.binding,is_eat='[eat]' in line))(state)
            self.assertTrue(bool(result.admitted.all()))
            if result.mutation_callback[0]:scheduled.append((result.mutation,side,0))
            if result.reveal_callback[0]:scheduled.append((result.reveal,side,0))
            for owner,slot in np.argwhere(np.asarray(result.partner_callback[0])):
                scheduled.append((result.partner,int(owner),int(slot)))
            self.assertEqual(bool(result.berry_ate[0]),'[eat]' in line)
            state=result.partner
        engine.ingest_event(showdown._public_event_from_line(line))
        self.assertEqual(len(scheduled),len(calls),line)
        species=self.j.zeros((1,2,6),self.j.int32).at[:,0,0].set(f.host.species_id('Absol')).at[:,1,0].set(f.host.species_id('Snorlax'))
        base=f.b.initialize(species,f.host.device)
        context=f.c.initialize(base,f.contexts[narrow],species_binding=f.species_catalog.device.binding)
        for (stage,side,slot),reference in zip(scheduled,calls):
            self.assertEqual(reference.showdown_slot,f'p{side+1}',line)
            for name,attribute in (('revealed','revealed_item'),('original','original_public_item'),('current','current_public_item')):
                raw=getattr(reference,attribute);want=f.item_catalog.item_id(raw) if raw else 0
                self.assertEqual(int(getattr(stage,name)[0,side,slot]),want,(line,name))
            self.assertEqual(bool(stage.mutated[0,side,slot]),reference.item_mutated)
            self.assertEqual(bool(stage.removed[0,side,slot]),reference.item_removed)
            belief,admitted=f.c.rebuild(context._replace(items=stage),base,f.host.device,f.contexts[narrow])
            self.assertTrue(bool(admitted.all()));_,summary=f.b.materialize(belief,f.host.device)
            self.assertEqual(int(summary.count[0,1-side,slot]),reference.candidate_set_count,line)
            self.assertAlmostEqual(float(summary.uncertainty[0,1-side,slot]),reference.uncertainty,places=6)
        return state,scheduled

    def test_raw_removal_mutation_then_named_reveal_and_duplicate_callbacks(self):
        base=list(self.f.prefix)
        known=base+['|-item|p1a: Snorlax|Leftovers']
        cases=((base,'|-heal|p1a: Snorlax|100/100|[from] item: Leftovers'),
            (known,'|-heal|p1a: Snorlax|100/100|[from] item: Leftovers'),
            (base,'|-enditem|p1a: Snorlax|Leftovers|[from] move: Knock Off'),
            (known,'|-enditem|p1a: Snorlax|Leftovers|[from] move: Knock Off'),
            (known+['|-enditem|p1a: Snorlax|Leftovers|[from] move: Knock Off'],
                '|-enditem|p1a: Snorlax|Leftovers|[from] move: Knock Off'),
            (known,'|-enditem|p1a: Snorlax|Sitrus Berry|[eat]'),
            (known,'|-enditem|p1a: Snorlax|Choice Band|[from] move: Unknown Move'),
            (known,'|-item|p1a: Snorlax|Leftovers'),
            (known,'|-item|p1a: Snorlax|Choice Band|[from] item: Leftovers'))
        for narrow in (False,True):
            for prefix,line in cases:self.compare(prefix,line,narrow)

    def test_trick_subject_then_partner_credit_in_both_assignment_modes(self):
        base=list(self.f.prefix)+['|-activate|p1a: Snorlax|move: Trick|[of] p2a: Absol']
        for narrow in (False,True):
            self.compare(base,'|-item|p2a: Absol|Choice Band|[from] move: Trick',narrow)
            self.compare(base+['|-item|p2a: Absol|Choice Band|[from] move: Trick'],
                '|-item|p1a: Snorlax|Leftovers|[from] move: Trick',narrow)
            self.compare(list(self.f.prefix),'|-item|p1a: Snorlax|Choice Band|[from] move: Trick',narrow)

    def test_typed_batch_isolation_namespace_and_disabled_rows(self):
        import numpy as np
        initial,_=self.f.run_lines(self.f.prefix);repeat=lambda x:self.j.repeat(x,8,axis=0)
        mons=self.jax.tree.map(repeat,initial.mon_state);state=self.jax.tree.map(repeat,initial.context.items)
        tables=self.f.item_catalog.device;item=self.f.item_catalog.item_id('Leftovers')
        run=self.jax.jit(lambda s:self.x.project(s,mons,tables,side=0,slot=0,encounter=1,item_id=item,
            operation=self.f.items.REMOVE,item_binding=tables.binding))
        good=run(state);self.assertEqual(good.admitted.tolist(),[True]*8)
        bad=state._replace(binding=state.binding.at[0,0].set(state.binding[0,0]^self.j.uint32(1)))
        bad=bad._replace(current=bad.current.at[1,0,0].set(item))
        result=run(bad);self.assertEqual(result.admitted.tolist(),[False,False]+[True]*6)
        for values,old in zip(self.jax.tree.leaves(result.partner),self.jax.tree.leaves(bad)):
            np.testing.assert_array_equal(values[:2],old[:2])
        np.testing.assert_array_equal(result.partner.revealed[2:],good.partner.revealed[2:])
        self.assertFalse(bool(result.mutation_callback[:2].any()));self.assertFalse(bool(result.reveal_callback[:2].any()))
        with self.assertRaisesRegex(ValueError,'shape/dtype'):run(state._replace(revealed=state.revealed.astype(self.j.float32)))
        disabled=self.x.project(state,mons,tables,side=0,slot=0,encounter=1,item_id=item,operation=self.f.items.REMOVE,
            item_binding=tables.binding,valid=False)
        self.assertEqual(disabled.admitted.tolist(),[True]*8)
        for actual,old in zip(self.jax.tree.leaves(disabled.partner),self.jax.tree.leaves(state)):np.testing.assert_array_equal(actual,old)
        self.assertFalse(bool(disabled.mutation_callback.any()))
