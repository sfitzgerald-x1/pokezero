"""Real raw MOVE callbacks execute inside the atomic public event fold."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicRefreshContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_tendency_context as fixtures
        from gpu_engine import public_sleep,public_belief_refresh,public_belief_snapshot
        fixtures.PublicTendencyContextTests.setUpClass.__func__(cls)
        cls.snapshot=public_belief_snapshot
        cls.sleep_tables=public_sleep.compile_sleep(cls.species_catalog,cls.vocab)
        cls.refresh_tables=public_belief_refresh.compile_refresh(cls.host,cls.catalog)
        cls.contexts={n:cls.c.compile_context(cls.host,cls.abilities,cls.catalog,items=cls.item_catalog,
            item_belief_narrowing=n,field_catalog=cls.field_catalog,active_catalog=cls.active_catalog,
            sleep_tables=cls.sleep_tables,tendency_tables=cls.tables,refresh_tables=cls.refresh_tables) for n in (False,True)}
        cls.kwargs=dict(cls.kwargs,sleep_tables=cls.sleep_tables)

    project=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.project
    fold=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.fold
    carry=staticmethod(__import__('test_gpu_engine_public_item_context').PublicItemContextTests.carry)
    run_lines=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.run_lines
    same=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.same

    def initial(self,narrow=False,batch=1):
        beliefs=self.b.initialize(self.j.zeros((batch,2,6),self.j.int32),self.host.device)
        binding=self.species_catalog.device.binding
        return (self.i.initialize(self.j.arange(1,batch+1,dtype=self.j.int32),source_binding=self.host.device.binding),
            beliefs,self.m.initialize(batch,species_binding=binding),self.c.initialize(beliefs,self.contexts[narrow],
                species_binding=binding,tendency_complete_prefix=True,source_tables=self.host.device))

    def chunks(self,projections,events=None):
        import numpy as np
        p=projections[0]
        return list(self.i.pack_chunks(np.array([1],np.int32),[events if events is not None else [e for p in projections for e in p.events]],
            source_binding=self.host.device.binding,**{n:np.asarray(getattr(p,n),np.uint32) for n in (
                'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding',
                'active_binding','sleep_binding','tendency_binding')}))

    def reference_engine(self,lines,narrow=False):
        from pokezero.belief import PublicBattleBeliefEngine
        engine=PublicBattleBeliefEngine(format_id='gen3randombattle',set_source=self.source,item_belief_narrowing=narrow)
        calls=[];original=engine._with_set_summary
        def observe(p):
            out=original(p);calls.append(out.key);return out
        engine._with_set_summary=observe
        for raw in lines:engine.ingest_event(self.reference_module._public_event_from_line(raw))
        return engine,calls

    def compare(self,result,engine,calls):
        summary,ok=self.snapshot.read(result.context.refresh.controller.visible,result.beliefs,self.host.device)
        self.assertTrue(bool(ok.all()))
        for side in (0,1):
            for slot,pokemon in enumerate(engine.snapshot().side(f'p{side+1}')):
                self.assertEqual(int(summary.count[0,1-side,slot]),pokemon.candidate_set_count)
                self.assertAlmostEqual(float(summary.uncertainty[0,1-side,slot]),pokemon.uncertainty,places=6)
                self.assertEqual(int(result.context.refresh.controller.visible.generation[0,1-side,slot]),calls.count(pokemon.key))
                self.assertEqual(int(result.beliefs.conflicts[0,1-side,slot]),engine._variant_pin_conflicts.get(pokemon.key,0))
        self.assertEqual(result.context.refresh.family.tolist(),[0])
        self.assertEqual(result.context.refresh.main_seen.tolist(),[0])
        self.assertEqual(result.context.refresh.controller.visible.source_binding.shape,(8,))

    def test_actual_raw_source_callbacks_and_choice_both_modes(self):
        from pokezero import showdown
        self.reference_module=showdown
        lines=[*self.prefix,'|turn|1','|-ability|p2a: Absol|Pressure',
            '|move|p1a: Snorlax|Body Slam|p2a: Absol','|move|p1a: Snorlax|Surf|p2a: Absol',
            '|move|p1a: Snorlax|Surf|p2a: Absol|[from] move: Metronome',
            '|move|p1a: Snorlax|Surf|p2a: Absol|[from] move: Sleep Talk','|',
            '|move|p1a: Snorlax|Baton Pass|p1a: Snorlax',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100',
            '|-ability|p1a: Gardevoir|Pressure|[from] ability: Trace|[of] p2a: Absol',
            '|turn|2','|switch|p1a: Snorlax|Snorlax, L80|100/100']
        for narrow in (False,True):
            result,_=self.run_lines(lines,narrow);engine,calls=self.reference_engine(lines,narrow);self.compare(result,engine,calls)

    def test_pin_registered_between_raw_chunks_retains_exact_callback_order(self):
        from pokezero import showdown
        self.reference_module=showdown
        result,_=self.run_lines(self.prefix);engine,calls=self.reference_engine(self.prefix)
        variants=self.host.variants[self.host.species_id('Snorlax')]
        victim=next(v for v in variants if not any(x.lower()=='surf' for x in v.moves))
        survivor=victim.to_summary();self.assertTrue(engine.narrow_candidate_variants('p1:snorlax',[survivor]))
        mask=self.j.zeros_like(result.beliefs.species,bool).at[:,1,0].set(True)
        beliefs,_=self.b.narrow_pin(result.beliefs,self.host.pin_mask('Snorlax',[survivor]),valid=mask)
        current=(result.cursor,beliefs,result.mon_state,result.context)
        host,_=self.project(self.prefix)
        for raw in ('|move|p1a: Snorlax|Surf|p2a: Absol','|move|p1a: Snorlax|Surf|p2a: Absol'):
            host,ps=self.project([raw],state=host)
            for chunk in self.chunks(ps):
                result=self.fold(current,chunk);self.assertEqual(result.error.tolist(),[0]);current=self.carry(result)
            engine.ingest_event(showdown._public_event_from_line(raw));self.compare(result,engine,calls)
        self.assertEqual(int(result.beliefs.conflicts[0,1,0]),4)

    def test_source_main_proof_survives_chunk_continuation(self):
        from pokezero import showdown
        self.reference_module=showdown
        for count in range(64):
            lines=[*self.prefix,*(['|']*count),'|move|p1a: Snorlax|Surf|p2a: Absol']
            _,ps=self.project(lines);events=[e for p in ps for e in p.events]
            index=next(i for i,e in enumerate(events) if e.kind==self.i.Kind.MOVE)
            if index%64==63:break
        else:self.fail('fixture failed to place MOVE at chunk end')
        chunks=self.chunks(ps);current=self.initial();pending_seen=False
        for chunk in chunks:
            result=self.fold(current,chunk);self.assertEqual(result.error.tolist(),[0]);current=self.carry(result)
            if int(result.context.refresh.main_seen[0]):
                pending_seen=True;self.assertTrue(bool(result.cursor.needs_more[0]))
                self.assertTrue(bool(result.context.tendency_raw_pending[0]))
        self.assertTrue(pending_seen);engine,calls=self.reference_engine(lines);self.compare(result,engine,calls)

    def test_raw_family_drop_wrong_family_and_late_row_error_are_atomic(self):
        from dataclasses import replace
        lines=[*self.prefix,'|move|p1a: Snorlax|Surf|p2a: Absol']
        _,ps=self.project(lines);events=[e for p in ps for e in p.events]
        move=next(i for i,e in enumerate(events) if e.kind==self.i.Kind.MOVE)
        missing=[replace(e,sequence=n+1) for n,e in enumerate(events[:move]+events[move+1:])]
        out=self.fold(self.initial(),self.chunks(ps,missing)[0]);self.assertNotEqual(out.error.tolist(),[0]);self.same(self.initial(),self.carry(out))
        chunk=self.chunks(ps)[0];opener=next(i for i,e in enumerate(events) if e.kind==self.i.Kind.RAW_BEGIN)
        bad=chunk._replace(words=chunk.words.at[0,opener,11].set(0))
        out=self.fold(self.initial(),bad);self.assertNotEqual(out.error.tolist(),[0]);self.same(self.initial(),self.carry(out))
        # A valid different move cannot silently disagree with its closing
        # raw action, even though both IDs are in the frozen namespace.
        changed=chunk._replace(words=chunk.words.at[0,move,9].set(self.catalog.move_id('Body Slam')))
        out=self.fold(self.initial(),changed);self.assertNotEqual(out.error.tolist(),[0]);self.same(self.initial(),self.carry(out))
        batch=self.initial(batch=2)
        mixed=chunk._replace(stream_id=self.j.array([1,2],self.j.int32),words=self.j.concatenate([chunk.words,chunk.words]),
            valid=self.j.concatenate([chunk.valid,chunk.valid]),more=self.j.zeros(2,bool))
        mixed=mixed._replace(words=mixed.words.at[0,len(events)-1,1].set(999))
        out=self.jax.jit(lambda c:self.fold(batch,c))(mixed)
        self.assertNotEqual(int(out.error[0]),0);self.assertEqual(int(out.error[1]),0)
        for before,after in zip(self.jax.tree.leaves(batch),self.jax.tree.leaves(self.carry(out))):
            if before.shape[:1]==(2,):self.assertTrue(bool(self.j.array_equal(before[0],after[0])))
        self.assertEqual(out.context.refresh.controller.visible.source_binding.shape,(8,))

    def test_remaining_source_families_and_legacy_namespace_refuse(self):
        host,_=self.project(self.prefix)
        for raw in ('|upkeep','|move|p1a: Snorlax|Mud Shot|p2a: Absol','|-damage|p1a: Snorlax|50/100',
                '|cant|p1a: Snorlax|slp','|-item|p1a: Snorlax|Leftovers'):
            with self.assertRaises(ValueError):self.project([raw],state=host)
        beliefs=self.b.initialize(self.j.zeros((1,2,6),self.j.int32),self.host.device)
        with self.assertRaises(ValueError):self.c.initialize(beliefs,self.contexts[False],species_binding=self.species_catalog.device.binding)

    def test_invalid_saved_selection_refuses_noop_row_at_entry(self):
        _,ps=self.project(['|']);chunk=self.chunks(ps)[0]
        initial=self.initial(batch=2);cursor,beliefs,mons,context=initial
        for value in (-1,2**30):
            controller=context.refresh.controller._replace(first_selected=context.refresh.controller.first_selected.at[0,0,0].set(value))
            bad=context._replace(refresh=context.refresh._replace(controller=controller))
            mixed=chunk._replace(stream_id=self.j.array([1,2],self.j.int32),words=self.j.concatenate([chunk.words,chunk.words]),
                valid=self.j.concatenate([chunk.valid,chunk.valid]),more=self.j.zeros(2,bool))
            before=(cursor,beliefs,mons,bad)
            out=self.jax.jit(lambda ch:self.fold(before,ch))(mixed)
            self.assertEqual(out.error.tolist(),[self.i.Error.SOURCE,0])
            for a,b in zip(self.jax.tree.leaves(before),self.jax.tree.leaves(self.carry(out))):
                if a.shape[:1]==(2,):self.assertTrue(bool(self.j.array_equal(a[0],b[0])))


if __name__=='__main__':unittest.main()
