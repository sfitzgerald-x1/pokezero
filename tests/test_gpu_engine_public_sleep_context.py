"""Real sleep facts share lifecycle, clocks and atomic stream rollback."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicSleepContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_field_context as fixtures
        from test_gpu_engine_public_truant import configure_truant
        from gpu_engine import public_sleep,public_damage
        fixtures.PublicFieldContextTests.setUpClass.__func__(cls)
        configure_truant(cls)
        cls.sl=public_sleep;cls.sleep_tables=public_sleep.compile_sleep(cls.species_catalog,cls.vocab)
        cls.catalog=cls.p.compile_moves(cls.host,cls.dex,cls.vocab)
        cls.item_catalog=cls.items.compile_items(cls.host,cls.vocab,item_names=('Mystery Orb',))
        cls.field_catalog=cls.f.compile_field(cls.vocab);cls.damage_tables=public_damage.compile_damage(cls.vocab)
        cls.contexts={n:cls.c.compile_context(cls.host,cls.abilities,cls.catalog,items=cls.item_catalog,
            item_belief_narrowing=n,field_catalog=cls.field_catalog,active_catalog=cls.active_catalog,
            damage_tables=cls.damage_tables,sleep_tables=cls.sleep_tables) for n in (False,True)}
        cls.kwargs=dict(source=cls.host,species_catalog=cls.species_catalog,moves=cls.catalog,
            abilities=cls.abilities,items=cls.item_catalog,field_catalog=cls.field_catalog,
            active_catalog=cls.active_catalog,damage_tables=cls.damage_tables,sleep_tables=cls.sleep_tables)

    project=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.project
    initial=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.initial
    fold=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.fold
    carry=staticmethod(__import__('test_gpu_engine_public_item_context').PublicItemContextTests.carry)
    run_lines=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.run_lines
    same=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.same

    def chunks(self,projections):
        import numpy as np
        p=projections[0]
        return list(self.i.pack_chunks(np.array([1],np.int32),[[e for p in projections for e in p.events]],
            source_binding=self.host.device.binding,**{n:np.asarray(getattr(p,n),np.uint32) for n in (
                'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding',
                'active_binding','damage_binding','sleep_binding')}))

    def compare(self,result,lines):
        from test_gpu_engine_public_sleep import PublicSleepTests
        check=PublicSleepTests();check.sleep,check.sleep_tables,check.reference=self.sl,self.sleep_tables,self.s
        check.catalog,check.vocab=self.species_catalog,self.vocab
        engine=self.Reference.from_events([self.s._public_event_from_line(x) for x in lines],
            format_id='gen3randombattle',set_source=self.source)
        book=tuple({self.s._normalize_identifier(self.species_catalog.names[int(result.mon_state.species[0,s,n])]):n
            for n in range(6) if int(result.mon_state.encounter[0,s,n])>0} for s in (0,1))
        check.compare(result.context.sleep,result.mon_state,engine,lines,book)
        self.assertEqual(int(result.context.sleep.turn[0]),int(result.cursor.turn[0]))

    def test_actual_sleep_attempts_refunds_and_shared_upkeep_cross_chunks(self):
        lines=[*self.prefix,'|switch|p1a: Gardevoir|Gardevoir, L80|100/100',
            '|switch|p1a: Snorlax|Snorlax, L80|100/100','|turn|1','|-status|p1a: Snorlax|slp|[from] move: Rest',
            '|cant|p1a: Snorlax|slp','|move|p1a: Snorlax|Sleep Talk|p2a: Absol',
            '|move|p1a: Snorlax|Body Slam|p2a: Absol|[from] move: Sleep Talk',
            '|-damage|p2a: Absol|80/100',*(['|']*27),'|upkeep','|turn|2',
            '|cant|p1a: Snorlax|slp','|move|p1a: Snorlax|Snore|p2a: Absol',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100',
            '|switch|p1a: Snorlax|Snorlax, L80|100/100 slp']
        for narrow in (False,True):
            result,p=self.run_lines(lines,narrow);self.compare(result,lines)
            self.assertGreater(len(self.chunks(p)),1)
            upkeep=next(x for x in p if x.raw_line=='|upkeep')
            self.assertEqual([e.kind for e in upkeep.events],[self.i.Kind.RAW_BEGIN,self.i.Kind.UPKEEP])
            self.assertEqual(int(result.context.sleep.count[0,0,0]),0)
            self.assertAlmostEqual(float(result.context.damage.last_taken[0,1].sum()),.2,places=6)

    def test_bench_team_cures_pending_and_faint_age_match_source(self):
        lines=[*self.prefix,'|turn|1','|-status|p1a: Snorlax|slp|[from] move: Rest',
            '|cant|p1a: Snorlax|slp','|-curestatus|p1: Snorlax|slp|[silent]',
            '|-status|p1a: Snorlax|slp|[from] move: Rest','|cant|p1a: Snorlax|slp',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100',
            '|-curestatus|p1: Snorlax|slp|[silent]','|switch|p1a: Snorlax|Snorlax, L80|100/100',
            '|-status|p2a: Absol|slp|[from] move: Rest','|cant|p2a: Absol|slp',
            '|-cureteam|p2a: Absol','|move|p2a: Absol|Sleep Talk|p1a: Snorlax',
            '|faint|p2a: Absol','|turn|2','|upkeep']
        result,_=self.run_lines(lines);self.compare(result,lines)

    def test_wrong_headers_after_sleep_fact_roll_back_every_ledger_and_isolate_rows(self):
        lines=[*self.prefix,'|switch|p1a: Gardevoir|Gardevoir, L80|100/100',
            '|switch|p1a: Snorlax|Snorlax, L80|100/100','|turn|1','|-status|p1a: Snorlax|slp|[from] move: Rest',
            '|cant|p1a: Snorlax|slp','|move|p2a: Absol|Body Slam|p1a: Snorlax','|-damage|p1a: Snorlax|80/100']
        _,p=self.project(lines);chunk=self.chunks(p)[0];initial=self.initial()
        for bad in (chunk._replace(sleep_binding=None),chunk._replace(sleep_binding=chunk.sleep_binding.at[0].add(1))):
            result=self.fold(initial,bad);self.assertEqual(result.error.tolist(),[self.i.Error.SOURCE]);self.same(initial,self.carry(result))
        index=next(n for n in range(64) if int(chunk.words[0,n,0])==self.i.Kind.SLEEP)
        fn=self.jax.jit(lambda cur,b,m,c,ch:self.fold((cur,b,m,c),ch))
        for changes in ({8:99},{9:self.sl.ARRIVE},{10:99},{11:7},{12:1},{7:5}):
            words=chunk.words
            for col,value in changes.items():words=words.at[0,index,col].set(value)
            result=fn(*initial,chunk._replace(words=words));self.assertNotEqual(int(result.error[0]),0);self.same(initial,self.carry(result))
        cant=next(n for n in range(64) if int(chunk.words[0,n,0])==self.i.Kind.SLEEP and int(chunk.words[0,n,9])==self.sl.CANT_SLEEP)
        words=chunk.words.at[0,cant,7].set(1).at[0,cant,8].set(2)
        result=fn(*initial,chunk._replace(words=words));self.assertEqual(result.error.tolist(),[self.i.Error.PAYLOAD]);self.same(initial,self.carry(result))
        batch=self.initial(batch=2);words=self.j.concatenate([chunk.words,chunk.words]).at[0,0,1].set(99)
        mixed=chunk._replace(stream_id=self.j.array([1,2],self.j.int32),words=words,
            valid=self.j.concatenate([chunk.valid,chunk.valid]),more=self.j.zeros(2,bool))
        result=self.fold(batch,mixed);self.assertEqual(result.error.tolist(),[self.i.Error.SEQUENCE,0])
        for before,after in zip(self.jax.tree.leaves(batch),self.jax.tree.leaves(self.carry(result))):
            self.assertTrue(bool(self.j.array_equal(before[0],after[0])))
        self.assertEqual(int(result.context.sleep.count[1,0,0]),1)

    def test_sleep_namespace_and_clock_cannot_drop_on_continuation(self):
        wire,_=self.project(self.prefix)
        legacy=self.c.compile_context(self.host,self.abilities,self.catalog,items=self.item_catalog,
            field_catalog=self.field_catalog,active_catalog=self.active_catalog,damage_tables=self.damage_tables)
        for changes in (dict(sleep_tables=None),dict(sleep_tables=None,context_tables=legacy),dict(context_tables=legacy)):
            kwargs=dict(self.kwargs,context_tables=self.contexts[False]);kwargs.update(changes)
            with self.assertRaises(ValueError):self.wire.project_line(wire,'|',**kwargs)
        good,_=self.run_lines([*self.prefix,'|turn|1']);_,tail=self.project(['|'],state=self.project([*self.prefix,'|turn|1'])[0])
        wrong=good.context._replace(sleep=good.context.sleep._replace(turn=self.j.array([0],self.j.int32)))
        initial=(good.cursor,good.beliefs,good.mon_state,wrong);result=self.fold(initial,self.chunks(tail)[0])
        self.assertEqual(result.error.tolist(),[self.i.Error.SOURCE]);self.same(initial,self.carry(result))


if __name__=='__main__':unittest.main()
