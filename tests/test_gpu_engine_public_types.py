"""Public live retyping compared with the actual replay and token writer."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicTypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_team as fixtures
        fixtures.PublicTeamTests.setUpClass.__func__(cls)
        from gpu_engine import public_types as t
        from pokezero.dex import ShowdownDex,SpeciesInfo
        from pokezero.category_vocab import CategoryVocabulary
        species=dict(cls.dex.species)
        species.update({key:SpeciesInfo(key,name,types,{}) for key,name,types in (
            ('castform','Castform',('Normal',)),('castformsunny','Castform-Sunny',('Fire','Flying')),
            ('kecleon','Kecleon',('Normal',)))})
        cls.dex=ShowdownDex({},species,{})
        cls.vocab=CategoryVocabulary(('pokemon:opponent','species:castform','type:normal','type:fire',
            'type:water','type:ice','type:grass','type:flying','type:psychic'))
        cls.catalog=cls.public.compile_species(cls.host,cls.dex,cls.vocab,cls.abilities)
        cls.t=t;cls.types=t.compile_types(cls.catalog,cls.dex,cls.vocab);cls.tables=cls.types.device

    def initial(self,batch=1):
        return self.t.initialize(batch,self.tables,species_binding=self.catalog.device.binding),self.m.initialize(batch,species_binding=self.catalog.device.binding)

    def apply(self,state,mons,op,side,first=0,second=0,**kwargs):
        enc=int(mons.encounter[0,side,int(mons.active[0,side])])
        return self.t.apply(state,self.tables,mons,operation=op,side=side,encounter=enc,
            first=first,second=second,producer_binding=self.tables.binding,**kwargs)

    def consume(self,state,mons,line,identities):
        p=line.split('|');kind=p[1];slot=self.reference._slot_from_ident(p[2]) if len(p)>2 else None
        side=int(slot[1])-1 if slot in ('p1','p2') else -1
        if kind in ('switch','drag','replace'):
            key=p[2].split(':',1)[-1].strip();book=identities[side]
            if key not in book:book[key]=len(book)
            index=book[key]
            species=p[3].split(',')[0]
            mons,ok=self.m.arrive(mons,side=side,slot=index,encounter=index+1,
                species=self.catalog.species_id(species),source_species=0,level=80,gender=0,
                hp_numerator=100,hp_denominator=100,status=0)
            self.assertTrue(bool(ok.all()));state,ok=self.apply(state,mons,self.t.ARRIVE,side)
            self.assertTrue(bool(ok.all()))
        if kind=='faint':
            index=int(mons.active[0,side]);mons,ok=self.m.faint(mons,side=side,slot=index,encounter=index+1)
            self.assertTrue(bool(ok.all()))
        raw=self.t.project_line(line,catalog=self.types)
        self.assertFalse(raw.full_line_consumers_complete)
        for op,seat,x,y in raw.facts:
            state,ok=self.apply(state,mons,op,seat,x,y);self.assertTrue(bool(ok.all()))
        return state,mons

    def compare(self,state,mons,lines):
        replay=self.reference.parse_showdown_replay(lines,battle_id='public-type-fixture')
        got=self.t.project(state,self.tables,mons,expected_vocabulary=self.tables.vocabulary_binding)
        self.assertEqual(got.admitted.tolist(),[True])
        for side in (0,1):
            seat=f'p{side+1}';candidate=replay.public_active.get(seat)
            if candidate is None:continue
            from dataclasses import replace
            candidate=replace(candidate,active=True,live_type_source=replay.live_type_override.get(seat))
            cats=[['']*51];nums=[[0.]*200]
            self.reference._encode_pokemon_tokens(cats,nums,0,[candidate],role='opponent',limit=1,
                dex=self.dex,schema_v3=True,schema_v4=True)
            pair=self.reference._resolve_live_type_slots(candidate.live_type_source,self.dex)
            self.assertEqual(bool(state.overridden[0,side]),pair is not None)
            self.assertEqual(int(got.write_mask[0,side].sum()),2 if pair else 0)
            if pair:
                for col in self.t.COLUMNS:self.assertEqual(int(got.categorical[0,side,col]),self.vocab.encode(cats[0][col]))
            else:self.assertFalse(bool(got.categorical[0,side].any()))

    def prefixes(self,events):
        state,mons=self.initial();lines=[];identities=({}, {})
        for line in ['|switch|p1a: Cloud|Castform, L80|100/100','|switch|p2a: Lizard|Kecleon, L80|100/100',*events]:
            state,mons=self.consume(state,mons,line,identities);lines.append(line)
            self.compare(state,mons,lines)
        return state,mons,lines

    def test_forecast_dex_types_absent_fallback_base_reversion_and_unresolved(self):
        self.prefixes(['|-formechange|p1a: Cloud|Castform-Sunny',
            '|-formechange|p1a: Cloud|Castform-Rainy','|-formechange|p1a: Cloud|Castform-Snowy',
            '|-formechange|p1a: Cloud|Castform','|-formechange|p1a: Cloud|Missing-Forme',
            '|-formechange|p1a: Cloud|Castform-Sunny','|-formechange|p1a: Cloud|'])

    def test_typechange_dual_empty_unresolved_unknown_oov_and_explicit_end(self):
        self.prefixes(['|-start|p2a: Lizard|typechange|Water / Grass / Fire',
            '|-start|p2a: Lizard|typechange|','|-start|p2a: Lizard|typechange| / / ',
            '|-start|p2a: Lizard|typechange|Mystery','|-end|p2a: Lizard|typechange'])

    def test_switch_drag_replace_clear_faint_and_blank_preserve_other_side(self):
        self.prefixes(['|-start|p1a: Cloud|typechange|Ice','|-start|p2a: Lizard|typechange|Grass',
            '|faint|p1a: Cloud','|','|switch|p1a: Spare|Absol, L80|100/100',
            '|-start|p1a: Spare|typechange|Water','|drag|p1a: Cloud|Castform, L80|100/100',
            '|-start|p1a: Cloud|typechange|Water','|replace|p1a: Cloud|Castform, L80|100/100'])

    def test_jit_mixed_rows_namespaces_wrong_encounter_and_category_refuse(self):
        state,mons,_=self.prefixes([])
        state=self.jax.tree.map(lambda x:self.j.concatenate([x,x]),state)
        mons=self.jax.tree.map(lambda x:self.j.concatenate([x,x]),mons)
        x=self.vocab.encode('type:fire')
        fn=self.jax.jit(lambda st,m,enc,first:self.t.apply(st,self.tables,m,operation=self.t.TYPE,side=0,
            encounter=enc,first=first,producer_binding=self.tables.binding))
        for enc,first in [(self.j.array([99,1],self.j.int32),self.j.array([x,x],self.j.int32)),
            (self.j.array([1,1],self.j.int32),self.j.array([1,x],self.j.int32))]:
            got,ok=fn(state,mons,enc,first);self.assertEqual(ok.tolist(),[False,True])
            for a,b in zip(self.jax.tree.leaves(state),self.jax.tree.leaves(got)):self.assertTrue(bool(self.j.array_equal(a[0],b[0])))
        broken=state._replace(species_binding=state.species_binding.at[0,0].add(1))
        got=self.t.project(broken,self.tables,mons,expected_vocabulary=self.tables.vocabulary_binding)
        self.assertEqual(got.admitted.tolist(),[False,True]);self.assertFalse(bool(got.write_mask[0].any()))

    def test_bindings_shape_and_integer_transport_are_strict(self):
        state,mons,_=self.prefixes([])
        with self.assertRaises(ValueError):self.t.initialize(1,self.tables,species_binding=self.j.zeros(8,self.j.uint32))
        with self.assertRaises(ValueError):self.apply(state,mons,self.t.TYPE,0,first=1.0)
        with self.assertRaises(ValueError):self.t.project(state._replace(categories=state.categories.astype(self.j.float32)),
            self.tables,mons,expected_vocabulary=self.tables.vocabulary_binding)
        broken=state._replace(categories=state.categories.at[0,0,1].set(1))
        got=self.t.project(broken,self.tables,mons,expected_vocabulary=self.tables.vocabulary_binding)
        self.assertEqual(got.admitted.tolist(),[False])

    def test_compiled_raw_fallback_is_immune_to_caller_dex_mutation(self):
        from pokezero.dex import ShowdownDex,SpeciesInfo
        species=dict(self.dex.species);dex=ShowdownDex({},species,{})
        catalog=self.t.compile_types(self.catalog,dex,self.vocab)
        before=self.t.project_line('|-formechange|p1a: Cloud|Unown-Z',catalog=catalog)
        species['unown']=SpeciesInfo('unown','Unown',('Fire',),{})
        after=self.t.project_line('|-formechange|p1a: Cloud|Unown-Z',catalog=catalog)
        self.assertEqual(before,after)

    def test_changed_dex_refuses_and_caller_vocabulary_cannot_change_bound_production(self):
        from pokezero.dex import ShowdownDex,SpeciesInfo
        from pokezero.category_vocab import CategoryVocabulary
        species=dict(self.dex.species);species['castformsunny']=SpeciesInfo('castformsunny','Castform-Sunny',('Water',),{})
        with self.assertRaises(ValueError):self.t.compile_types(self.catalog,ShowdownDex({},species,{}),self.vocab)
        vocab=CategoryVocabulary(self.vocab.tokens,self.vocab.oov_buckets,dict(self.vocab.aliases))
        catalog=self.t.compile_types(self.catalog,self.dex,vocab)
        line='|-start|p1a: Cloud|typechange|Water'
        before=self.t.project_line(line,catalog=catalog)
        vocab._index['type:water']=vocab.encode('type:fire')
        self.assertEqual(before,self.t.project_line(line,catalog=catalog))
        with self.assertRaises(TypeError):catalog.vocabulary._index['type:water']=1


if __name__=='__main__':unittest.main()
