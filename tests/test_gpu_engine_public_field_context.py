"""Ordered field facts commit atomically with public item/move/ability state."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class PublicFieldContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_item_context as fixtures
        fixtures.PublicItemContextTests.setUpClass.__func__(cls)
        from gpu_engine import public_field as f, public_team, private_requests
        from pokezero.category_vocab import CategoryVocabulary
        from pokezero.dex import ShowdownDex,MoveInfo
        cls.f,cls.requests,cls.s=f,private_requests,cls.reference
        cls.vocab=CategoryVocabulary(cls.vocab.tokens+('field','request_kind:wait','weather:raindance','weather:sandstorm'))
        moves=dict(cls.dex.moves)
        moves['wish']=MoveInfo('wish','Wish','Normal','Status','Status',0,100,0,False,False,False,None,{},'self',False,pp=10)
        cls.dex=ShowdownDex(moves,cls.dex.species,cls.dex.type_chart)
        cls.catalog=cls.p.compile_moves(cls.host,cls.dex,cls.vocab)
        cls.abilities=cls.a.compile_abilities(cls.host,cls.vocab,
            ability_names=('Pressure','Trace','Immunity','Levitate','Sand Stream'))
        cls.item_catalog=cls.items.compile_items(cls.host,cls.vocab,item_names=('Mystery Orb',))
        cls.species_catalog=public_team.compile_species(cls.host,cls.dex,cls.vocab,cls.abilities)
        cls.field_catalog=f.compile_field(cls.vocab);cls.tables=cls.field_catalog.device
        cls.contexts={n:cls.c.compile_context(cls.host,cls.abilities,cls.catalog,items=cls.item_catalog,
            item_belief_narrowing=n,field_catalog=cls.field_catalog) for n in (False,True)}
        cls.kwargs=dict(source=cls.host,species_catalog=cls.species_catalog,moves=cls.catalog,
            abilities=cls.abilities,items=cls.item_catalog,field_catalog=cls.field_catalog)

    project=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.project
    initial=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.initial
    fold=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.fold
    carry=staticmethod(__import__('test_gpu_engine_public_item_context').PublicItemContextTests.carry)
    run_lines=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.run_lines
    same=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.same
    assert_writer=__import__('test_gpu_engine_public_field').PublicFieldTests.assert_writer

    def chunks(self,projections):
        import numpy as np
        p=projections[0]
        return list(self.i.pack_chunks(np.array([1],np.int32),[[e for p in projections for e in p.events]],
            source_binding=self.host.device.binding,**{name:np.asarray(getattr(p,name),np.uint32) for name in
                ('species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding')}))

    def test_weather_spikes_wish_and_turn_clock_compose_with_actual_writer(self):
        lines=[*self.prefix,'|turn|1','|-sidestart|p1: Alice|Spikes','|-sidestart|p2: Bob|move: Spikes',
               '|-weather|RainDance','|move|p1a: Snorlax|Wish|p1a: Snorlax','|','|-weather|RainDance|[upkeep]',
               '|upkeep','|turn|2','|-heal|p1a: Snorlax|100/100|[from] move: Wish',
               '|-sideend|p2: Bob|Spikes','|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Absol']
        result,projections=self.run_lines(lines)
        for observer in (0,1):
            for exact in (False,True):self.assert_writer(result.context.field,lines,observer=observer,exact=exact)
        self.assertEqual(int(result.context.field.turn[0]),int(result.cursor.turn[0]))
        self.assertEqual([e.kind for e in projections[-1].events],
            [self.i.Kind.RAW_BEGIN,self.i.Kind.ABILITY,self.i.Kind.FIELD])
        self.assertEqual([e.kind for e in projections[6].events],
            [self.i.Kind.RAW_BEGIN,self.i.Kind.FIELD,self.i.Kind.MOVE])

    def test_boundaries_and_field_facts_cross_chunks_without_losing_either_clock(self):
        lines=[*self.prefix,*(['|']*30),'|turn|1','|-weather|RainDance',
               '|move|p2a: Absol|Wish|p2a: Absol','|turn|2','|-weather|RainDance|[upkeep]']
        result,projections=self.run_lines(lines)
        self.assertGreater(len(self.chunks(projections)),1)
        self.assertEqual(int(result.cursor.turn[0]),1+1)
        self.assert_writer(result.context.field,lines,observer=1)
        self.assertTrue(bool(result.context.field.weather_set[0]))
        self.assertEqual(int(result.context.field.weather_upkeeps[0]),1)

    def test_wrong_field_headers_config_or_malformed_suffix_roll_back_all_state(self):
        _,projections=self.project([*self.prefix,'|turn|1','|-weather|RainDance','|-item|p1a: Snorlax|Leftovers'])
        initial=self.initial();chunk=self.chunks(projections)[0]
        for bad in (chunk._replace(field_binding=None),chunk._replace(field_binding=chunk.field_binding.at[0].add(1))):
            result=self.fold(initial,bad);self.assertEqual(result.error.tolist(),[self.i.Error.SOURCE]);self.same(initial,self.carry(result))
        # A direct typed caller cannot manufacture a field-only downgrade of
        # a context whose digest was compiled for the item-enabled composition.
        malformed_tables=self.contexts[False]._replace(items=None,item_predicate=None,item_narrowing=None)
        malformed_context=initial[3]._replace(items=None)
        with self.assertRaises(ValueError):
            self.c.initialize(initial[1],malformed_tables,species_binding=self.species_catalog.device.binding)
        with self.assertRaises(ValueError):
            self.i.fold_chunk(initial[0],initial[1],self.host.device,
                chunk._replace(item_binding=None,context_binding=None),mon_state=initial[2],
                context=malformed_context,context_tables=malformed_tables)
        field_index=next(i for i in range(64) if int(chunk.words[0,i,0])==self.i.Kind.FIELD)
        bad=chunk._replace(words=chunk.words.at[0,field_index,9].set(2**31-1))
        result=self.jax.jit(lambda cur,b,m,c,ch:self.fold((cur,b,m,c),ch))(*initial,bad)
        self.assertNotEqual(int(result.error[0]),0);self.same(initial,self.carry(result))
        good=self.fold(initial,chunk);self.assertEqual(good.error.tolist(),[0])
        outofsync=good.context._replace(field=good.context.field._replace(turn=self.j.zeros_like(good.context.field.turn)))
        state,_=self.project([*self.prefix,'|turn|1','|-weather|RainDance','|-item|p1a: Snorlax|Leftovers'])
        _,more=self.project(['|'],state=state)
        bad=self.fold((good.cursor,good.beliefs,good.mon_state,outofsync),self.chunks(more)[0])
        self.assertEqual(bad.error.tolist(),[self.i.Error.SOURCE])

    def test_continuation_cannot_drop_field_namespace_and_unhandled_sides_still_refuse(self):
        state,_=self.project(self.prefix)
        without=self.c.compile_context(self.host,self.abilities,self.catalog,items=self.item_catalog)
        for changes in (dict(field_catalog=None,context_tables=without),dict(field_catalog=None),
                        dict(context_tables=without)):
            with self.assertRaises(ValueError):self.project(['|'],state=state,**changes)
        for line in ('|-sidestart|p1: Alice|Reflect','|-weather|UnknownWeather','|-sidestart|p3: Bad|Spikes'):
            with self.subTest(line=line),self.assertRaises(ValueError):self.project([line],state=state)
        event=self.i.PublicEvent(self.i.Kind.FIELD,1,0,self.i.Phase.ACTION,
            target_side=0,payload=(self.f.WEATHER,1,0,0,0,0,0,0))
        with self.assertRaises(ValueError):event.words()


if __name__=='__main__':unittest.main()
