"""Ordered real move callbacks compared with instrumented Python summaries."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicBeliefRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import dataclasses
        from gpu_engine import public_belief_refresh as r
        import test_gpu_engine_public_moves as fixtures
        fixtures.PublicMovesTests.setUpClass.__func__(cls)
        old=cls.source.universes['snorlax']
        a,b=old.variants
        variants=(a,dataclasses.replace(b,item='Choice Band'),dataclasses.replace(b,
            variant_id='both',moves=('Body Slam','Surf','Rest','Protect'),item='Choice Band'))
        cls.source=dataclasses.replace(cls.source,universes=dict(cls.source.universes,
            snorlax=dataclasses.replace(old,variants=variants)))
        cls.host=cls.b.compile_source(cls.source)
        cls.catalog=cls.p.compile_moves(cls.host,cls.dex,cls.vocab);cls.tables=cls.catalog.device
        cls.r=r;cls.refresh_tables=r.compile_refresh(cls.host,cls.catalog)

    same=__import__('test_gpu_engine_public_moves').PublicMovesTests.same_tree
    move_initial=__import__('test_gpu_engine_public_moves').PublicMovesTests.initial

    def initial(self,batch=1):
        from gpu_engine import public_belief_snapshot as snapshot
        move,full,mons=self.move_initial(batch=batch)
        empty=self.b.initialize(self.j.zeros_like(full.species),self.host.device)
        state=self.r.initialize(empty,self.host.device,self.refresh_tables)
        visible=state.visible
        # Source arrival upsert + _on_switch_in replace, for both encounters.
        for _ in range(2):visible,full,ok=snapshot.refresh(visible,full,self.host.device,source_refresh=full.species>0)
        self.assertTrue(bool(ok.all()))
        return state._replace(visible=visible),move,full,full,mons

    def reference(self,narrow=False):
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.showdown import _public_event_from_line
        engine=PublicBattleBeliefEngine(format_id='gen3randombattle',set_source=self.source,item_belief_narrowing=narrow)
        calls=[];original=engine._with_set_summary
        def observe(pokemon):
            out=original(pokemon);calls.append((out.key,out.candidate_set_count,out.uncertainty,
                engine._variant_pin_conflicts.get(out.key,0)))
            return out
        engine._with_set_summary=observe
        for raw in ('|switch|p1a: Snorlax|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100'):
            engine.ingest_event(_public_event_from_line(raw))
        return engine,calls

    def apply(self,carry,line,*,narrow=False,mutated=None,revealed=None,pending=None,transformed=False,valid=True,tables=None):
        state,move,base,full,mons=carry
        caller,has_from=self.p.lexical_move_tags(line)
        z=self.j.zeros((mons.species.shape[0],2,6),bool)
        return self.r.record_move(state,move,base,full,mons,self.host.device,self.tables,self.refresh_tables if tables is None else tables,
            side=0,slot=0,encounter=1,move_id=self.catalog.move_id(line.split('|')[3]),caller=caller,
            has_from=has_from,transformed=transformed,item_mutated=z if mutated is None else mutated,
            item_revealed=z if revealed is None else revealed,sleep_pending=z if pending is None else pending,
            item_belief_narrowing=narrow,valid=valid)

    @staticmethod
    def carry(out,mons):return out.refresh,out.move_update.state,out.basis,out.beliefs,mons

    def compare(self,out,engine,calls):
        from gpu_engine import public_belief_snapshot as snapshot
        source=engine.snapshot().side('p1')[0]
        summary,ok=snapshot.read(out.refresh.visible,out.beliefs,self.host.device)
        self.assertTrue(bool(ok.all()));self.assertTrue(bool(out.admitted.all()))
        self.assertEqual(int(summary.count[0,1,0]),source.candidate_set_count)
        self.assertAlmostEqual(float(summary.uncertainty[0,1,0]),source.uncertainty,places=6)
        self.assertEqual(int(out.beliefs.conflicts[0,1,0]),engine._variant_pin_conflicts.get(source.key,0))
        self.assertEqual(int(out.refresh.visible.generation[0,1,0]),sum(key==source.key for key,*_ in calls))

    def test_intermediate_pin_conflicts_follow_old_old_new_then_repeat_and_called(self):
        from pokezero.showdown import _public_event_from_line
        carry=self.initial();engine,calls=self.reference()
        variant=self.host.variants[self.host.species_id('Snorlax')][0].to_summary()
        changed=engine.narrow_candidate_variants('p1:snorlax',[variant]);self.assertTrue(changed)
        mask=self.j.zeros_like(carry[2].species,bool).at[:,1,0].set(True)
        pin=self.host.pin_mask('Snorlax',[variant])
        state,_=self.b.narrow_pin(carry[2],pin,valid=mask)
        carry=(carry[0],carry[1],state,state,carry[4])
        for raw,delta in [('|move|p1a: Snorlax|Surf|p2a: Absol',1),
                ('|move|p1a: Snorlax|Surf|p2a: Absol',3),
                ('|move|p1a: Snorlax|Body Slam|p2a: Absol|[from] move: Metronome',1)]:
            old=int(carry[3].conflicts[0,1,0]);out=self.apply(carry,raw)
            engine.ingest_event(_public_event_from_line(raw));self.compare(out,engine,calls)
            self.assertEqual(int(out.beliefs.conflicts[0,1,0])-old,delta)
            carry=self.carry(out,carry[4])

    def test_choice_exclusion_once_between_upsert_and_pp_in_both_modes(self):
        from pokezero.showdown import _public_event_from_line
        for narrow in (False,True):
            carry=self.initial();engine,calls=self.reference(narrow)
            for name in ('Body Slam','Surf','Protect','Surf'):
                raw='|move|p1a: Snorlax|'+name+'|p2a: Absol'
                engine.ingest_event(_public_event_from_line(raw));out=self.apply(carry,raw,narrow=narrow)
                self.compare(out,engine,calls);carry=self.carry(out,carry[4])
            self.assertEqual(bool(out.refresh.choice_excluded[0,0,0]),narrow)

    def test_called_locked_struggle_transformed_sleep_skip_and_item_guards(self):
        from pokezero.showdown import _public_event_from_line
        for suffix in ('','|[from] move: Sleep Talk','|[from] move: Metronome','|[from]lockedmove','|[from]other'):
            for name in ('Surf','Struggle'):
                carry=self.initial();engine,calls=self.reference()
                raw='|move|p1a: Snorlax|'+name+'|p2a: Absol'+suffix
                engine.ingest_event(_public_event_from_line(raw));out=self.apply(carry,raw)
                self.compare(out,engine,calls)
        carry=self.initial();engine,calls=self.reference()
        # Actual sleeping-turn callbacks establish the pending public fact.
        engine.ingest_event(_public_event_from_line('|cant|p1a: Snorlax|slp'))
        from gpu_engine import public_belief_snapshot as snapshot
        cache=carry[0].visible;full=carry[3];mask=self.j.zeros_like(full.species,bool).at[:,1,0].set(True)
        for _ in range(2):cache,full,_=snapshot.refresh(cache,full,self.host.device,source_refresh=mask)
        carry=(carry[0]._replace(visible=cache),carry[1],full,full,carry[4])
        pending=self.j.zeros((1,2,6),bool).at[:,0,0].set(True)
        raw='|move|p1a: Snorlax|Sleep Talk|p2a: Absol'
        engine.ingest_event(_public_event_from_line(raw));out=self.apply(carry,raw,pending=pending);self.compare(out,engine,calls)
        self.assertEqual(int(out.refresh.first_selected[0,0,0]),self.catalog.move_id('Sleep Talk'))
        # Transformed moves still perform the initial upsert, with no set PP.
        before=self.initial();out=self.apply(before,'|move|p1a: Snorlax|Surf|p2a: Absol',transformed=True)
        self.assertEqual(int(out.refresh.visible.generation[0,1,0]),3)
        self.assertEqual(int(out.move_update.charge[0]),0)
        self.assertEqual(int(out.refresh.first_selected[0,0,0]),0)
        # Public original-item mutation/reveal freezes the exclusion conclusion.
        for field in ('mutated','revealed'):
            carry=self.initial();first=self.apply(carry,'|move|p1a: Snorlax|Body Slam|p2a: Absol',narrow=True)
            carry=self.carry(first,carry[4]);flag=self.j.zeros((1,2,6),bool).at[:,0,0].set(True)
            out=self.apply(carry,'|move|p1a: Snorlax|Surf|p2a: Absol',narrow=True,**{field:flag})
            self.assertFalse(bool(out.refresh.choice_excluded.any()));self.assertEqual(int(out.refresh.visible.generation[0,1,0]),8)

    def test_jit_late_overflow_rolls_back_every_ledger_in_only_bad_row(self):
        carry=self.initial(batch=2)
        variant=self.host.variants[self.host.species_id('Snorlax')][0].to_summary()
        mask=self.j.zeros_like(carry[2].species,bool).at[:,1,0].set(True)
        full,_=self.b.narrow_pin(carry[2],self.host.pin_mask('Snorlax',[variant]),valid=mask)
        full=full._replace(conflicts=full.conflicts.at[0,1,0].set(2**31-1))
        carry=(carry[0],carry[1],full,full,carry[4])
        raw='|move|p1a: Snorlax|Surf|p2a: Absol'
        out=self.jax.jit(lambda x:self.apply(x,raw))(carry)
        self.assertEqual(out.admitted.tolist(),[False,True])
        after=self.carry(out,carry[4])
        for a,b in zip(self.jax.tree.leaves(carry),self.jax.tree.leaves(after)):
            if a.shape[:1]==(2,):self.assertTrue(bool(self.j.array_equal(a[0],b[0])))
        self.assertEqual(int(out.beliefs.conflicts[1,1,0]),1)
        self.assertEqual(int(out.refresh.visible.generation[1,1,0]),5)
        # Wrong callback namespace refuses without updating unrelated ledgers.
        bad=carry[0]._replace(binding=carry[0].binding.at[:,0].add(1))
        refused=self.apply((bad,*carry[1:]),raw)
        self.assertFalse(bool(refused.admitted.any()));self.same((bad,*carry[1:]),self.carry(refused,carry[4]))
        with self.assertRaises(ValueError):self.apply(carry,raw,pending=True)

    def test_namespace_types_and_stale_cache_refuse_before_move_side_effects(self):
        carry=self.initial();raw='|move|p1a: Snorlax|Surf|p2a: Absol'
        binding=carry[0].binding
        for bad in (binding[0],(binding+1).astype(self.j.float32),binding.astype(self.j.int32)):
            with self.assertRaises(ValueError):self.apply((carry[0]._replace(binding=bad),*carry[1:]),raw)
            with self.assertRaises(ValueError):self.r.reset_stay(carry[0]._replace(binding=bad),carry[4],
                self.refresh_tables,side=0,slot=0,encounter=1)
        stale=carry[3]._replace(has_evidence=carry[3].has_evidence.at[:,1,0].set(True))
        torn=(carry[0],carry[1],stale,stale,carry[4]);out=self.apply(torn,raw)
        self.assertFalse(bool(out.admitted.any()));self.same(torn,self.carry(out,carry[4]))
        bad=self.refresh_tables._replace(choice_predicate=self.j.array(2**31-1,self.j.int32))
        out=self.apply(carry,raw,narrow=True,tables=bad)
        self.assertFalse(bool(out.admitted.any()));self.same(carry,self.carry(out,carry[4]))
        for bad in (self.refresh_tables._replace(binding=self.refresh_tables.binding.astype(self.j.float32)),
                self.refresh_tables._replace(sleep_usable=self.refresh_tables.sleep_usable.astype(self.j.int32))):
            with self.assertRaises(ValueError):self.apply(carry,raw,tables=bad)

    def test_entry_resets_only_live_active_stay_and_keeps_certain_exclusion(self):
        carry=self.initial();state=carry[0]._replace(first_selected=carry[0].first_selected.at[:,0,0].set(1),
            choice_excluded=carry[0].choice_excluded.at[:,0,0].set(True))
        reset,ok=self.r.reset_stay(state,carry[4],self.refresh_tables,side=0,slot=0,encounter=1)
        self.assertTrue(bool(ok.all()));self.assertEqual(int(reset.first_selected[0,0,0]),0)
        self.assertTrue(bool(reset.choice_excluded[0,0,0]))
        mons=carry[4]._replace(encounter=carry[4].encounter.at[:,0,1].set(2))
        state=state._replace(first_selected=state.first_selected.at[:,0,1].set(1))
        out,ok=self.r.reset_stay(state,mons,self.refresh_tables,side=0,slot=1,encounter=2)
        self.assertFalse(bool(ok.any()));self.same(state,out)


if __name__=='__main__':unittest.main()
