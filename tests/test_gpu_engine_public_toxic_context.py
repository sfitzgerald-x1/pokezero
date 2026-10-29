"""Toxic state consumes each raw transaction once with the other public ledgers."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicToxicContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_field_context as fixtures
        from test_gpu_engine_public_truant import configure_truant
        from gpu_engine import public_toxic,public_types,public_damage
        from pokezero.category_vocab import CategoryVocabulary
        fixtures.PublicFieldContextTests.setUpClass.__func__(cls)
        cls.vocab=CategoryVocabulary(tuple(dict.fromkeys((*cls.vocab.tokens,'type:normal','type:water','type:grass'))))
        configure_truant(cls)
        cls.tx=public_toxic;cls.toxic_tables=public_toxic.compile_toxic(cls.species_catalog,cls.vocab)
        cls.catalog=cls.p.compile_moves(cls.host,cls.dex,cls.vocab)
        cls.item_catalog=cls.items.compile_items(cls.host,cls.vocab,item_names=('Mystery Orb',))
        cls.field_catalog=cls.f.compile_field(cls.vocab);cls.damage_tables=public_damage.compile_damage(cls.vocab)
        cls.type_catalog=public_types.compile_types(cls.species_catalog,cls.dex,cls.vocab)
        cls.contexts={n:cls.c.compile_context(cls.host,cls.abilities,cls.catalog,items=cls.item_catalog,
            item_belief_narrowing=n,field_catalog=cls.field_catalog,active_catalog=cls.active_catalog,
            damage_tables=cls.damage_tables,type_catalog=cls.type_catalog,toxic_tables=cls.toxic_tables) for n in (False,True)}
        cls.kwargs=dict(source=cls.host,species_catalog=cls.species_catalog,moves=cls.catalog,abilities=cls.abilities,
            items=cls.item_catalog,field_catalog=cls.field_catalog,active_catalog=cls.active_catalog,
            damage_tables=cls.damage_tables,type_catalog=cls.type_catalog,toxic_tables=cls.toxic_tables)

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
                species_binding=binding,toxic_hp_visibility=(2,2)))

    def chunks(self,projections,events=None):
        import numpy as np
        p=projections[0]
        return list(self.i.pack_chunks(np.array([1],np.int32),[events if events is not None else [e for p in projections for e in p.events]],
            source_binding=self.host.device.binding,**{n:np.asarray(getattr(p,n),np.uint32) for n in (
                'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding',
                'active_binding','damage_binding','type_binding','toxic_binding')}))

    def compare(self,result,lines):
        from test_gpu_engine_public_toxic import PublicToxicTests
        check=PublicToxicTests();check.t,check.tables,check.reference,check.dex=self.tx,self.toxic_tables,self.s,self.dex
        wire,_=self.project(lines)
        check.ids={entry.ident:entry.address.encounter for entries in wire.encounters for entry in entries}
        parser=self.s._ReplayParser(complete_prefix=False,hp_visibility={'p1':'percentage','p2':'percentage'})
        for line in lines:parser._feed_line(line)
        check.compare(result.context.toxic,parser)
        self.assertEqual(int(result.context.toxic.turn[0]),int(result.cursor.turn[0]))
        self.assertFalse(bool(result.context.toxic_raw_pending.any()))

    def test_status_reentry_residual_and_joint_consumers_match_actual_source(self):
        lines=[*self.prefix,'|turn|1','|-status|p1a: Snorlax|tox',
            '|move|p1a: Snorlax|Body Slam|p2a: Absol','|-damage|p2a: Absol|80/100',
            '|-start|p2a: Absol|typechange|Water','|upkeep','|turn|2',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100 tox',
            '|-damage|p1a: Gardevoir|94/100 tox|[from] psn','|t:|123',
            '|-status|p1a: Gardevoir|slp|[from] move: Rest','|-curestatus|p1: Snorlax|tox|[silent]',
            '|-cureteam|p1a: Gardevoir']
        for narrow in (False,True):
            result,_=self.run_lines(lines,narrow);self.compare(result,lines)
            self.assertAlmostEqual(float(result.context.damage.last_taken[0,1].sum()),.2,places=6)

    def test_pending_raw_control_can_cross_chunk_boundary_without_double_clock(self):
        core=[*self.prefix,'|turn|1','|-status|p1a: Snorlax|tox']
        _,p=self.project(core);count=sum(len(x.events) for x in p)
        if count%2==0:core+=['|-status|p1a: Snorlax|tox'];_,p=self.project(core);count=sum(len(x.events) for x in p)
        self.assertEqual(count%2,1)
        core+=['|']*((63-count%64)%64//2)
        lines=[*core,'|-damage|p1a: Snorlax|94/100 tox|[from] psn','|upkeep','|turn|2']
        _,p=self.project(lines);chunks=self.chunks(p);self.assertGreater(len(chunks),1)
        self.assertEqual(int(chunks[0].words[0,63,0]),self.i.Kind.RAW_BEGIN)
        first=self.fold(self.initial(),chunks[0]);self.assertEqual(first.error.tolist(),[0])
        self.assertTrue(bool(first.context.toxic_raw_pending[0]));self.assertTrue(bool(first.cursor.needs_more[0]))
        final=self.fold(self.carry(first),chunks[1]);self.assertEqual(final.error.tolist(),[0]);self.compare(final,lines)

    def test_missing_or_duplicate_raw_consumption_refuses_and_rolls_back_all_ledgers(self):
        from dataclasses import replace
        _,p=self.project([*self.prefix,'|turn|1','|-status|p1a: Snorlax|tox'])
        events=[e for x in p for e in x.events];initial=self.initial()
        def renumber(items):return [replace(e,sequence=i+1) for i,e in enumerate(items)]
        controls=[i for i,e in enumerate(events) if e.kind==self.i.Kind.TOXIC];index=controls[-1]
        for changed in (events[:index]+events[index+1:],events[:index]+[events[index]]+events[index:],
                        events[:index-2]+events[index-1:],events[:index-2]+events[index-1:index]+events[index+1:]):
            chunk=self.chunks(p,renumber(changed))[0]
            result=self.fold(initial,chunk);self.assertNotEqual(int(result.error[0]),0);self.same(initial,self.carry(result))
        _,move_p=self.project([*self.prefix,'|turn|1','|move|p1a: Snorlax|Body Slam|p2a: Absol'])
        move_events=[e for x in move_p[:-1] for e in x.events]+[e for e in move_p[-1].events if e.kind not in (self.i.Kind.RAW_BEGIN,self.i.Kind.TOXIC)]
        result=self.fold(initial,self.chunks(move_p,renumber(move_events))[0])
        self.assertNotEqual(int(result.error[0]),0);self.same(initial,self.carry(result))
        # Explicit source evidence is not itself a raw protocol line.
        fact=self.i.PublicEvent(self.i.Kind.SET_REVEAL,len(events)+1,1,self.i.Phase.ACTION,
            target_side=0,target_slot=0,payload=(self.host.predicate_ids[('move','bodyslam')],0,0,0,0,0,0,0))
        result=self.fold(initial,self.chunks(p,[*events,fact])[0]);self.assertEqual(result.error.tolist(),[0])
        chunk=self.chunks(p)[0]
        for bad in (chunk._replace(toxic_binding=None),chunk._replace(toxic_binding=chunk.toxic_binding.at[0].add(1))):
            result=self.fold(initial,bad);self.assertEqual(result.error.tolist(),[self.i.Error.SOURCE]);self.same(initial,self.carry(result))
        batch=self.initial(batch=2);words=self.j.concatenate([chunk.words,chunk.words]).at[0,index,15].set(1)
        mixed=chunk._replace(stream_id=self.j.array([1,2],self.j.int32),words=words,
            valid=self.j.concatenate([chunk.valid,chunk.valid]),more=self.j.zeros(2,bool))
        result=self.jax.jit(lambda a,b,m,c,ch:self.fold((a,b,m,c),ch))(*batch,mixed)
        self.assertNotEqual(int(result.error[0]),0);self.assertEqual(int(result.error[1]),0)
        for old,new in zip(self.jax.tree.leaves(batch),self.jax.tree.leaves(self.carry(result))):self.assertTrue(bool(self.j.array_equal(old[0],new[0])))

    def test_source_namespace_continuation_and_unknown_raw_consumers_cannot_drop(self):
        wire,_=self.project(self.prefix)
        legacy=self.c.compile_context(self.host,self.abilities,self.catalog,items=self.item_catalog,
            field_catalog=self.field_catalog,active_catalog=self.active_catalog,damage_tables=self.damage_tables,type_catalog=self.type_catalog)
        for changes in (dict(toxic_tables=None),dict(toxic_tables=None,context_tables=legacy),dict(context_tables=legacy)):
            kwargs=dict(self.kwargs,context_tables=self.contexts[False]);kwargs.update(changes)
            with self.assertRaises(ValueError):self.wire.project_line(wire,'|',**kwargs)
        for line in ('|unimplemented|p1a: Snorlax','|upkeep\r','|turn','|turn|03'):
            with self.assertRaises(ValueError):self.project([line],state=wire)


if __name__=='__main__':unittest.main()
