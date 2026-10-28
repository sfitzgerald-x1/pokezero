"""Known and unknown Truant phases in the canonical ordered public stream."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicTruantContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_field_context as fixtures
        from test_gpu_engine_public_truant import configure_truant
        fixtures.PublicFieldContextTests.setUpClass.__func__(cls)
        configure_truant(cls)
        cls.catalog=cls.p.compile_moves(cls.host,cls.dex,cls.vocab)
        cls.item_catalog=cls.items.compile_items(cls.host,cls.vocab,item_names=('Mystery Orb',))
        cls.field_catalog=cls.f.compile_field(cls.vocab)
        cls.contexts={n:cls.c.compile_context(cls.host,cls.abilities,cls.catalog,items=cls.item_catalog,
            item_belief_narrowing=n,field_catalog=cls.field_catalog,active_catalog=cls.active_catalog) for n in (False,True)}
        cls.kwargs=dict(source=cls.host,species_catalog=cls.species_catalog,moves=cls.catalog,abilities=cls.abilities,
            items=cls.item_catalog,field_catalog=cls.field_catalog,active_catalog=cls.active_catalog)
        cls.prefix=('|switch|p1a: Slaking|Slaking, L80|100/100','|switch|p2a: Gardevoir|Gardevoir, L80|100/100')

    project=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.project
    initial=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.initial
    fold=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.fold
    carry=staticmethod(__import__('test_gpu_engine_public_item_context').PublicItemContextTests.carry)
    run_lines=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.run_lines
    same=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.same
    chunks=__import__('test_gpu_engine_public_active_context').PublicActiveContextTests.chunks

    def compare(self,result,lines):
        from test_gpu_engine_public_truant import PublicTruantTests
        check=PublicTruantTests();check.s,check.active,check.tables=self.s,self.active,self.active_catalog.device
        check.compare(result.context.active,result.mon_state,lines)
        self.assertEqual(int(result.context.active.turn[0]),int(result.cursor.turn[0]))

    def test_native_trace_anchor_and_unknown_phases_match_the_real_parser(self):
        events=['|turn|1','|-ability|p2a: Gardevoir|Truant|[from] ability: Trace|[of] p1a: Slaking',
            '|upkeep','|turn|2','|cant|p2a: Gardevoir|slp','|cant|p2a: Gardevoir|ability: Truant',
            '|turn|3','|move|p2a: Gardevoir|Body Slam|p1a: Slaking|[from] move: Sleep Talk',
            '|move|p2a: Gardevoir|Body Slam|p1a: Slaking',
            '|-ability|p2a: Gardevoir|Truant|[from] ability: Trace|[of] p1a: Slaking','|upkeep','|turn|4']
        for count in (4,8,len(events)):
            lines=[*self.prefix,*events[:count]];result,_=self.run_lines(lines);self.compare(result,lines)

    def test_post_upkeep_replacement_skip_and_upkeep_event_cross_chunks(self):
        lines=[*self.prefix,'|turn|1',*(['|']*28),'|upkeep','|faint|p1a: Slaking',
            '|switch|p1a: Slakoth|Slakoth, L80|100/100','|turn|2']
        result,projections=self.run_lines(lines);self.compare(result,lines)
        self.assertGreater(len(self.chunks(projections)),1)
        self.assertTrue(bool(result.context.active.truant_loaf[0,0]))
        self.assertFalse(bool(result.context.active.truant_skip_flip.any()))
        upkeep=next(p for p in projections if p.raw_line=='|upkeep')
        self.assertEqual([e.kind for e in upkeep.events],[self.i.Kind.RAW_BEGIN,self.i.Kind.UPKEEP])

    def test_truant_namespace_and_post_upkeep_cursor_mismatch_refuse_atomically(self):
        _,p=self.project([*self.prefix,'|turn|1'])
        initial=self.initial();chunk=self.chunks(p)[0];good=self.fold(initial,chunk)
        self.assertEqual(good.error.tolist(),[0])
        wire,_=self.project([*self.prefix,'|turn|1']);_,tail=self.project(['|'],state=wire)
        wrong=good.context._replace(active=good.context.active._replace(post_upkeep=self.j.ones_like(good.context.active.post_upkeep)))
        result=self.fold((good.cursor,good.beliefs,good.mon_state,wrong),self.chunks(tail)[0])
        self.assertEqual(result.error.tolist(),[self.i.Error.SOURCE])
        self.same((good.cursor,good.beliefs,good.mon_state,wrong),self.carry(result))
        bad=self.contexts[False]._replace(active=self.active_catalog.device._replace(
            public_species_binding=self.active_catalog.device.public_species_binding.at[0].add(1)))
        with self.assertRaises(ValueError):self.c.initialize(initial[1],bad,species_binding=self.species_catalog.device.binding)
        for mutated in (chunk._replace(active_binding=None),chunk._replace(active_binding=chunk.active_binding.at[0].add(1))):
            result=self.fold(initial,mutated);self.assertEqual(result.error.tolist(),[self.i.Error.SOURCE]);self.same(initial,self.carry(result))

    def test_native_mode_cannot_drop_or_inject_an_upkeep_as_a_targeted_fact(self):
        wire,_=self.project(self.prefix)
        legacy=self.active.compile_active(self.vocab)
        tables=self.c.compile_context(self.host,self.abilities,self.catalog,items=self.item_catalog,
            field_catalog=self.field_catalog,active_catalog=legacy)
        for changes in (dict(active_catalog=legacy),dict(active_catalog=legacy,context_tables=tables)):
            kwargs=dict(self.kwargs,context_tables=self.contexts[False]);kwargs.update(changes)
            with self.assertRaises(ValueError):self.wire.project_line(wire,'|',**kwargs)
        with self.assertRaises(ValueError):self.i.PublicEvent(self.i.Kind.ACTIVE,1,0,self.i.Phase.ACTION,
            target_side=0,target_slot=0,payload=(1,self.active.UPKEEP,0,0,0,0,0,0)).words()
        _,p=self.project([*self.prefix,'|turn|1','|cant|p1a: Slaking|ability: Truant'])
        chunk=self.chunks(p)[0];index=next(n for n in range(64) if int(chunk.words[0,n,0])==self.i.Kind.ACTIVE)
        bad=chunk._replace(words=chunk.words.at[0,index,9].set(self.active.UPKEEP))
        initial=self.initial();result=self.fold(initial,bad)
        self.assertNotEqual(int(result.error[0]),0);self.same(initial,self.carry(result))


if __name__=='__main__':unittest.main()
