"""Active inputs share the real raw stream and its atomic chunk commit."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicActiveContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataclasses import replace
        import test_gpu_engine_public_field_context as fixtures
        fixtures.PublicFieldContextTests.setUpClass.__func__(cls)
        from gpu_engine import public_active, public_team
        from pokezero.category_vocab import CategoryVocabulary
        from pokezero.dex import ShowdownDex
        cls.active=public_active
        cls.vocab=CategoryVocabulary(cls.vocab.tokens+tuple('volatile:'+n for n in public_active.NAMES[1:]))
        moves=dict(cls.dex.moves)
        for name in ('Baton Pass','Protect','Detect','Endure','Solar Beam'):
            key=cls.s._normalize_identifier(name)
            moves[key]=replace(moves['wish'],id=key,name=name)
        cls.dex=ShowdownDex(moves,cls.dex.species,cls.dex.type_chart)
        cls.catalog=cls.p.compile_moves(cls.host,cls.dex,cls.vocab)
        cls.abilities=cls.a.compile_abilities(cls.host,cls.vocab,ability_names=('Pressure','Trace','Immunity','Levitate','Sand Stream'))
        cls.item_catalog=cls.items.compile_items(cls.host,cls.vocab,item_names=('Mystery Orb',))
        cls.species_catalog=public_team.compile_species(cls.host,cls.dex,cls.vocab,cls.abilities)
        cls.field_catalog=cls.f.compile_field(cls.vocab)
        cls.active_catalog=public_active.compile_active(cls.vocab)
        cls.contexts={n:cls.c.compile_context(cls.host,cls.abilities,cls.catalog,items=cls.item_catalog,
            item_belief_narrowing=n,field_catalog=cls.field_catalog,active_catalog=cls.active_catalog) for n in (False,True)}
        cls.kwargs=dict(source=cls.host,species_catalog=cls.species_catalog,moves=cls.catalog,abilities=cls.abilities,
            items=cls.item_catalog,field_catalog=cls.field_catalog,active_catalog=cls.active_catalog)

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
                'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding','active_binding')}))

    def compare_active(self,result,lines):
        from test_gpu_engine_public_active import PublicActiveTests
        checker=PublicActiveTests()
        checker.s,checker.a,checker.catalog,checker.tables,checker.vocab=(
            self.s,self.active,self.active_catalog,self.active_catalog.device,self.vocab)
        checker.compare(result.context.active,result.mon_state,lines)
        self.assertEqual(int(result.context.active.turn[0]),int(result.cursor.turn[0]))
        self.assertEqual(int(result.context.field.turn[0]),int(result.cursor.turn[0]))

    def test_actual_prefixes_share_arrival_baton_faint_and_turn_consumers(self):
        events=['|turn|1','|-start|p1a: Snorlax|confusion','|-start|p1a: Snorlax|Encore',
            '|-activate|p1a: Snorlax|move: Wrap|[of] p2a: Absol','|-activate|p1a: Snorlax|trapped',
            '|move|p2a: Absol|Protect|p2a: Absol','|-singleturn|p2a: Absol|Protect','|turn|2',
            '|move|p1a: Snorlax|Baton Pass|p1a: Snorlax','|switch|p1a: Gardevoir|Gardevoir, L80|100/100',
            '|-mustrecharge|p2a: Absol','|-start|p1a: Gardevoir|perish3',
            '|-singlemove|p1a: Gardevoir|perish2','|turn|3','|-end|p1a: Gardevoir|Wrap|[partiallytrapped]',
            '|cant|p2a: Absol|recharge','|faint|p1a: Gardevoir',
            '|switch|p1a: Snorlax|Snorlax, L80|100/100','|turn|4']
        # Check actual parser state at several boundaries without replaying a scientific root.
        for count in (8,11,len(events)):
            lines=[*self.prefix,*events[:count]]
            result,_=self.run_lines(lines)
            self.compare_active(result,lines)

    def test_active_lifecycle_and_expiry_cross_chunk_boundaries(self):
        lines=[*self.prefix,*(['|']*30),'|turn|1','|-singlemove|p1a: Snorlax|Destiny Bond',
            '|move|p1a: Snorlax|Solar Beam||[still]','|-prepare|p1a: Snorlax|Solar Beam',
            '|-anim|p1a: Snorlax|Solar Beam|p2a: Absol','|-mustrecharge|p2a: Absol',
            '|turn|2','|cant|p2a: Absol|recharge']
        result,p=self.run_lines(lines)
        self.assertGreater(len(self.chunks(p)),1);self.compare_active(result,lines)
        move=next(x for x in p if x.raw_line.startswith('|move|'))
        self.assertEqual([e.kind for e in move.events],
            [self.i.Kind.RAW_BEGIN,self.i.Kind.ACTIVE,self.i.Kind.ACTIVE,self.i.Kind.ACTIVE,self.i.Kind.MOVE])

    def test_malformed_payload_namespace_or_clock_roll_back_the_whole_chunk(self):
        _,p=self.project([*self.prefix,'|turn|1','|-start|p1a: Snorlax|confusion','|-mustrecharge|p2a: Absol'])
        initial=self.initial();chunk=self.chunks(p)[0]
        for bad in (chunk._replace(active_binding=None),chunk._replace(active_binding=chunk.active_binding.at[0].add(1))):
            result=self.fold(initial,bad);self.assertEqual(result.error.tolist(),[self.i.Error.SOURCE]);self.same(initial,self.carry(result))
        index=next(n for n in range(64) if int(chunk.words[0,n,0])==self.i.Kind.ACTIVE)
        for offset,value in ((4,0),(7,1),(7,6),(8,999),(9,self.active.ARRIVE),(10,0),(11,1)):
            bad=chunk._replace(words=chunk.words.at[0,index,offset].set(value))
            result=self.jax.jit(lambda cur,b,m,c,ch:self.fold((cur,b,m,c),ch))(*initial,bad)
            self.assertNotEqual(int(result.error[0]),0);self.same(initial,self.carry(result))
        good=self.fold(initial,chunk);self.assertEqual(good.error.tolist(),[0])
        state,_=self.project([*self.prefix,'|turn|1','|-start|p1a: Snorlax|confusion','|-mustrecharge|p2a: Absol'])
        _,more=self.project(['|'],state=state)
        broken=good.context._replace(active=good.context.active._replace(turn=self.j.zeros_like(good.context.active.turn)))
        result=self.fold((good.cursor,good.beliefs,good.mon_state,broken),self.chunks(more)[0])
        self.assertEqual(result.error.tolist(),[self.i.Error.SOURCE])
        downgraded=self.contexts[False]._replace(field=None)
        with self.assertRaises(ValueError):self.c.initialize(initial[1],downgraded,species_binding=self.species_catalog.device.binding)
        with self.assertRaises(ValueError):self.i.fold_chunk(initial[0],initial[1],self.host.device,
            chunk._replace(field_binding=None),mon_state=initial[2],context=initial[3]._replace(field=None),context_tables=downgraded)

    def test_raw_continuation_cannot_drop_active_and_source_active_first_attribution(self):
        state,_=self.project(self.prefix)
        without=self.c.compile_context(self.host,self.abilities,self.catalog,items=self.item_catalog,field_catalog=self.field_catalog)
        for changes in (dict(active_catalog=None),dict(active_catalog=None,context_tables=without),dict(context_tables=without)):
            kwargs=dict(self.kwargs,context_tables=self.contexts[False]);kwargs.update(changes)
            with self.assertRaises(ValueError):self.wire.project_line(state,'|',**kwargs)
        state,_=self.project(['|switch|p1a: Gardevoir|Gardevoir, L80|100/100'],state=state)
        # These source consumers attribute by public SIDE, as does the common
        # active-first resolver. A stale nickname is not a party-slot join.
        _,facts=self.project(['|-start|p1a: Snorlax|confusion'],state=state)
        active=next(e for e in facts[0].events if e.kind==self.i.Kind.ACTIVE)
        self.assertEqual(active.target_slot,state.active[0])
        self.assertEqual(active.payload[0],state.encounters[0][state.active[0]].address.encounter)
        state,_=self.project(['|faint|p1a: Gardevoir'],state=state)
        with self.assertRaises(ValueError):self.project(['|-start|p1a: Gardevoir|confusion'],state=state)
        with self.assertRaises(ValueError):self.project(['|-unknown|p1a: Gardevoir|unimplemented'],state=state)

    def test_choice_item_raw_events_share_the_current_encounter_and_boundaries(self):
        lines=[*self.prefix,'|turn|1','|-item|p1a: Snorlax|Choice Band|[from] move: Trick',
            '|move|p1a: Snorlax|Body Slam|p2a: Absol|[from] move: Sleep Talk',
            '|move|p1a: Snorlax|Body Slam|p2a: Absol','|',
            '|-item|p2a: Absol|Leftovers|[from] move: Trick','|turn|2',
            '|-enditem|p1a: Snorlax|Choice Band|[from] move: Knock Off']
        result,projections=self.run_lines(lines);self.compare_active(result,lines)
        first=next(p for p in projections if p.raw_line.startswith('|-item|p1'))
        self.assertEqual([e.kind for e in first.events],[self.i.Kind.RAW_BEGIN,self.i.Kind.ITEM,self.i.Kind.ACTIVE])
        self.assertTrue(bool(result.context.active.item_swapped[0,1]))
        _,projections=self.project([*self.prefix,'|-item|p1a: Snorlax|Choice Band'])
        initial=self.initial();chunk=self.chunks(projections)[0]
        index=next(n for n in range(64) if int(chunk.words[0,n,0])==self.i.Kind.ACTIVE)
        for value in (-1,4,2**31-1):
            result=self.fold(initial,chunk._replace(words=chunk.words.at[0,index,10].set(value)))
            self.assertNotEqual(int(result.error[0]),0);self.same(initial,self.carry(result))


if __name__=='__main__':unittest.main()
