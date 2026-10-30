"""Named public base columns against the real champion token writer."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class PublicTeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_context as fixtures
        fixtures.PublicContextTests.setUpClass.__func__(cls)
        from gpu_engine import public_team as p
        from pokezero import showdown as s
        from pokezero.dex import ShowdownDex,SpeciesInfo
        from pokezero.category_vocab import CategoryVocabulary
        cls.public,cls.reference=p,s
        cls.dex=ShowdownDex({}, {name:SpeciesInfo(name,name,types,dict(zip(p.STATS,stats)))
            for name,types,stats in [('snorlax',('Normal',),(160,110,65,65,110,30)),
                ('absol',('Dark',),(65,130,60,75,60,75)),
                ('gardevoir',('Psychic',),(68,65,65,125,115,80)),
                ('unown',('Psychic',),(48,72,48,72,48,48))]}, {})
        cls.vocab=CategoryVocabulary(('pokemon:opponent','species:snorlax','status:none','status:par','type:normal'),
            aliases={'species:unown-z':'species:unown'})
        cls.catalog=p.compile_species(cls.host,cls.dex,cls.vocab,cls.abilities,extra_species=('Unown-Z',))

    def state(self):
        mons=self.m.initialize(1,species_binding=self.catalog.device.binding)
        entries=[(0,0,'Snorlax','100/100',80,1),(1,0,'Absol','70/100 par',80,2),
                 (1,1,'Gardevoir','100/100 tox',100,0)]
        from gpu_engine.public_mon_protocol import condition_words
        for side,slot,name,condition,level,gender in entries:
            hp,maximum,status,fainted=condition_words(condition)
            mons,ok=self.m.arrive(mons,side=side,slot=slot,encounter=slot+1,
                species=self.catalog.species_id(name),source_species=self.host.species_id(name),
                level=level,gender=gender,hp_numerator=hp,hp_denominator=maximum,status=status)
            self.assertTrue(bool(ok.all()))
        mons,_=self.m.boost_update(mons,side=1,stat=0,amount=2)
        mons,_=self.m.boost_update(mons,side=1,stat=4,amount=-6)
        return mons,self.a.initialize(1,self.abilities.device,species_binding=self.catalog.device.binding)

    def project(self,mons,abilities,observer=0):
        return self.public.project_opponent(mons,abilities,self.catalog.device,observer_side=observer,
                                           expected_binding=self.catalog.device.binding)

    def compare(self,got,candidates,**kwargs):
        numeric,categories=[[0.0]*200 for _ in range(6)],[['']*60 for _ in range(6)]
        self.reference._encode_pokemon_tokens(categories,numeric,0,candidates,role='opponent',limit=6,
            dex=self.dex,schema_v3=True,schema_v4=True,**kwargs)
        for row in range(6):
            for name,col in zip(self.public.CORE,self.public.NUMERIC_COLUMNS):
                self.assertAlmostEqual(float(got.numeric[0,row,col]),numeric[row][getattr(self.reference,'NUMERIC_'+name)],
                    places=6,msg=f'{name} row{row}')
            for col in self.public.CATEGORY_COLUMNS:
                self.assertEqual(int(got.categorical[0,row,col]),self.vocab.encode(categories[row][col]),f'cat{col} row{row}')

    def mon(self,name,condition,active,level=80,gender=''):
        details=f'{name}, L{level}'+(f', {gender}' if gender else '')
        return self.reference.ShowdownPokemon(ident=f'p2a: {name}',showdown_slot='p2',species=name,
            condition=condition,active=active,details=details)

    def test_default_namespace_carries_source_unown_forms_without_category_relabeling(self):
        from pokezero.randbat_vocab import UNOWN_FORMES
        catalog=self.public.compile_species(self.host,self.dex,self.vocab,self.abilities)
        for name in UNOWN_FORMES:
            identity=catalog.species_id(name)
            self.assertEqual(catalog.names[identity],name)
            self.assertEqual(int(catalog.device.species_category[identity]),self.vocab.encode('species:'+name))
            self.assertEqual(catalog.device.base_stats[identity].tolist(),[48,72,48,72,48,48])
            # Source's exact-dex HP lookup intentionally stays distinct from
            # its base-form fallback; do not silently repair that old quirk.
            self.assertEqual(float(catalog.device.original_hp[identity]),0)
        with self.assertRaises(ValueError):catalog.species_id('Unown-A')
        with self.assertRaises(ValueError):catalog.species_id('Unown-Unknown')
        mons=self.m.initialize(1,species_binding=catalog.device.binding)
        mons,ok=self.m.arrive(mons,side=1,slot=0,encounter=1,species=catalog.species_id('Unown-S'),
            source_species=self.host.species_id('Unown-S'),level=80,gender=0,
            hp_numerator=258,hp_denominator=258,status=self.m.NONE)
        self.assertTrue(bool(ok.all()))
        abilities=self.a.initialize(1,self.abilities.device,species_binding=catalog.device.binding)
        got=self.public.project_opponent(mons,abilities,catalog.device,observer_side=0,expected_binding=catalog.device.binding)
        self.assertTrue(bool(got.admitted.all()))
        self.compare(got,[self.mon('Unown-S','258/258',True)])

    def test_all18_numeric_five_category_columns_and_padding_match_actual_writer(self):
        mons,abilities=self.state();got=self.project(mons,abilities)
        self.compare(got,[self.mon('Absol','70/100 par',False,gender='F'),
            self.mon('Gardevoir','100/100 tox',True,level=100)],active_boosts={'atk':2,'spe':-6})
        self.assertEqual(int(got.numeric_columns.sum()),18)
        self.assertEqual(int(got.categorical_columns.sum()),5)
        self.assertTrue(bool(got.admitted.all()))
        self.assertTrue(bool((got.numeric[0,2:]==0).all()))

    def test_transform_uses_public_donor_but_retains_original_hp_level_gender(self):
        from pokezero.belief import RevealedPokemonBelief
        mons,abilities=self.state()
        abilities,ok=self.a.transform(abilities,mons,self.abilities.device,
            side=1,slot=1,encounter=2,donor_side=0,donor_slot=0,donor_encounter=1)
        self.assertTrue(bool(ok.all()))
        got=self.project(mons,abilities)
        transformed=RevealedPokemonBelief(showdown_slot='p2',species='Gardevoir',transformed=True,transform_species='Snorlax')
        self.compare(got,[self.mon('Absol','70/100 par',False,gender='F'),
            self.mon('Gardevoir','100/100 tox',True,level=100)],
            exact_beliefs_by_species={'gardevoir':transformed},active_boosts={'atk':2,'spe':-6})

    def test_faint_retains_public_pointer_and_forme_dex_fallback_preserves_vocab(self):
        mons,abilities=self.state()
        mons,_=self.m.faint(mons,side=1,slot=1,encounter=2)
        self.compare(self.project(mons,abilities),[self.mon('Absol','70/100 par',False,gender='F'),
            self.mon('Gardevoir','0 fnt',True,level=100)],active_boosts={'atk':2,'spe':-6})
        mons,_=self.m.arrive(mons,side=1,slot=2,encounter=3,species=self.catalog.species_id('Unown-Z'),
            source_species=0,level=80,gender=0,hp_numerator=17,hp_denominator=31,status=0)
        self.compare(self.project(mons,abilities),[self.mon('Absol','70/100 par',False,gender='F'),
            self.mon('Gardevoir','0 fnt',False,level=100),self.mon('Unown-Z','17/31',True)])

    def test_jit_both_observers_refuse_invalid_namespace_and_hide_all_rows(self):
        mons,abilities=self.state()
        mons=self.jax.tree.map(lambda x:self.j.concatenate([x,x,x]),mons)
        abilities=self.jax.tree.map(lambda x:self.j.concatenate([x,x,x]),abilities)
        fn=self.jax.jit(lambda m,a,o,d:self.public.project_opponent(m,a,self.catalog.device,
            observer_side=o,expected_binding=d))
        got=fn(mons,abilities,self.j.array([0,1,2],self.j.int32),self.catalog.device.binding)
        self.assertEqual(got.admitted.tolist(),[True,True,False])
        self.assertTrue(bool((got.numeric[2]==0).all()))
        self.assertEqual(int(got.categorical[1,0,self.reference.CATEGORY_PRIMARY]),self.vocab.encode('species:snorlax'))
        wrong=fn(mons,abilities,self.j.zeros(3,self.j.int32),self.catalog.device.binding.at[0].add(1))
        self.assertFalse(bool(wrong.admitted.any()))
        self.assertTrue(bool((wrong.categorical==0).all()))
        bad=abilities._replace(binding=abilities.binding.at[0,0].add(1))
        self.assertEqual(fn(mons,bad,self.j.zeros(3,self.j.int32),self.catalog.device.binding).admitted.tolist(),[False,True,True])
        for owner in (True,.5,self.j.array(True),self.j.array(.5)):
            with self.assertRaises(ValueError):self.project(mons,abilities,owner)

    def test_static_compile_does_not_count_unused_oov_as_observed_drift(self):
        from pokezero.category_vocab import CategoryVocabulary
        vocab=CategoryVocabulary(('pokemon:opponent',))
        before=vocab.observed_oov_tokens
        self.public.compile_species(self.host,self.dex,vocab,self.abilities,extra_species=('Unown-Z',))
        self.assertEqual(vocab.observed_oov_tokens,before)

    def test_same_source_reordered_catalog_cannot_reinterpret_stored_species_or_donors(self):
        mons,abilities=self.state()
        changed=self.public.compile_species(self.host,self.dex,self.vocab,self.abilities,
            extra_species=('Unown-Z','Aaa'))
        got=self.public.project_opponent(mons,abilities,changed.device,observer_side=0,expected_binding=changed.device.binding)
        self.assertFalse(bool(got.admitted.any()))
        self.assertTrue(bool((got.numeric==0).all()))
        other=self.m.initialize(1,species_binding=changed.device.binding)
        other,_=self.m.arrive(other,side=0,slot=0,encounter=1,species=changed.species_id('Snorlax'),
            source_species=0,level=80,gender=0,hp_numerator=100,hp_denominator=100,status=0)
        _,ok=self.a.reveal(abilities,other,self.abilities.device,side=0,slot=0,encounter=1,
            ability_id=self.abilities.ability_id('Pressure'),authoritative=True)[:2]
        self.assertFalse(bool(ok.all()))

    def test_transport_requires_bound_namespace_and_rolls_back_changed_continuation(self):
        import numpy as np
        beliefs=self.b.initialize(self.j.zeros((1,2,6),self.j.int32),self.host.device)
        context=self.c.initialize(beliefs,self.context_tables,species_binding=self.catalog.device.binding)
        mons=self.m.initialize(1,species_binding=self.catalog.device.binding)
        cursor=self.i.initialize(self.j.array([1],self.j.int32),source_binding=self.host.device.binding)
        event=self.i.PublicEvent(self.i.Kind.SWITCH,1,0,self.i.Phase.ACTION,target_side=1,target_slot=0,
            payload=(1,self.catalog.species_id('Absol'),self.host.species_id('Absol'),80,0,100,100,0))
        def run(binding,**headers):
            options=dict(ability_binding=self.abilities.device.binding,move_binding=self.context_tables.moves.binding)
            options.update(headers)
            chunk=next(self.i.pack_chunks(np.array([1],np.int32),[[event]],source_binding=self.host.device.binding,
                species_binding=binding,**options))
            return self.i.fold_chunk(cursor,beliefs,self.host.device,chunk,mon_state=mons,
                                     context=context,context_tables=self.context_tables)
        for binding in (None,self.catalog.device.binding.at[0].add(1)):
            result=run(binding)
            self.assertEqual(result.error.tolist(),[self.i.Error.SOURCE])
            self.assertEqual(int(result.cursor.sequence[0]),0)
            self.assertFalse(bool(result.mon_state.encounter.any()))
        result=run(self.catalog.device.binding)
        self.assertEqual(result.error.tolist(),[0])
        self.assertEqual(int(result.mon_state.encounter[0,1,0]),1)
        for headers in ({'ability_binding':None},{'move_binding':None},
                        {'ability_binding':self.abilities.device.binding.at[0].add(1)},
                        {'move_binding':self.context_tables.moves.binding.at[0].add(1)}):
            result=run(self.catalog.device.binding,**headers)
            self.assertEqual(result.error.tolist(),[self.i.Error.SOURCE])
            self.assertEqual(int(result.cursor.sequence[0]),0)
            self.assertFalse(bool(result.mon_state.encounter.any()))


if __name__=='__main__':unittest.main()
