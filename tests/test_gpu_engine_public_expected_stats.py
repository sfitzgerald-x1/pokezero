"""Expected stat columns against the actual v4 writer and spread core."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicExpectedStatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataclasses import replace
        import test_gpu_engine_public_team as fixtures
        fixtures.PublicTeamTests.setUpClass.__func__(cls)
        from gpu_engine import public_expected_stats as expected, public_team
        from pokezero.dex import ShowdownDex,MoveInfo
        cls.e=expected
        universes=dict(cls.source.universes);original=universes['snorlax'];seed=original.variants[0]
        variants=[(('Return',),'Leftovers'),(('Hidden Power Fire','Rest'),'Leftovers'),
            (('Substitute','Flail'),'Salac Berry'),(('Belly Drum','Body Slam'),'Sitrus Berry'),
            (('Reversal','Substitute'),'Liechi Berry'),(('Hidden Power Ice','Body Slam'),'Choice Band')]
        universes['snorlax']=replace(original,variants=tuple(replace(seed,variant_id='expected-'+str(i),moves=moves,item=item)
            for i,(moves,item) in enumerate(variants)))
        cls.source=replace(cls.source,universes=universes);cls.host=cls.b.compile_source(cls.source)
        moves=dict(cls.dex.moves)
        for name,power,kind in [('Return',0,'Physical'),('Hidden Power Fire',70,'Special'),
            ('Hidden Power Ice',70,'Special'),('Rest',0,'Status'),('Body Slam',85,'Physical'),
            ('Substitute',0,'Status'),('Flail',0,'Physical'),('Reversal',0,'Physical'),('Belly Drum',0,'Status')]:
            key=cls.reference._normalize_identifier(name)
            moves[key]=MoveInfo(key,name,'Normal',kind,kind,power,100,0,False,False,False,None,{},'normal',False,pp=10)
        cls.dex=ShowdownDex(moves,cls.dex.species,cls.dex.type_chart)
        cls.abilities=cls.a.compile_abilities(cls.host,cls.vocab,ability_names=('Immunity','Pressure','Trace','Levitate'))
        cls.species_catalog=public_team.compile_species(cls.host,cls.dex,cls.vocab,cls.abilities,extra_species=('Unown-Z',))
        cls.catalog=expected.compile_expected(cls.host,cls.dex,cls.species_catalog);cls.tables=cls.catalog.device

    def state(self,*,observer=0,level=80,species='Snorlax',selected=None,available=True):
        mons=self.m.initialize(1,species_binding=self.species_catalog.device.binding)
        sid=self.host.species_id(species);pub=self.species_catalog.species_id(species)
        mons,ok=self.m.arrive(mons,side=1-observer,slot=0,encounter=1,species=pub,source_species=sid,
            level=level,gender=0,hp_numerator=100,hp_denominator=100,status=0)
        self.assertTrue(bool(ok.all()))
        species_ids=self.j.zeros((1,2,6),self.j.int32).at[0,observer,0].set(sid)
        states=self.b.initialize(species_ids,self.host.device,source_available=available)
        if selected is not None:
            mask=self.host.pin_mask(species,[self.host.variants[sid][i].to_summary() for i in selected])
            states=states._replace(hard=states.hard.at[0,observer,0].set(mask))
        states,summary=self.b.materialize(states,self.host.device)
        abilities=self.a.initialize(1,self.abilities.device,species_binding=self.species_catalog.device.binding)
        return states,summary,mons,abilities

    def project(self,current,*,observer=0,targets=None,exact=True,tables=None):
        tables=tables or self.tables
        return self.e.project_opponent(*current,tables,observer_side=observer,expected_binding=tables.binding,
            own_targets=targets,exact_state=exact)

    def reference_row(self,*,species='Snorlax',level=80,selected=None,transformed=False,target=None):
        from pokezero.belief import RevealedPokemonBelief
        sid=self.host.species_id(species)
        variants=self.host.variants[sid] if selected is None else [self.host.variants[sid][i] for i in selected]
        exact=RevealedPokemonBelief('p2',species,candidate_variants=tuple(v.to_summary() for v in variants))
        row=[0.]*self.reference.V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT
        self.reference._encode_expected_stats(row,self.dex,base_species=species,battle_species=species,
            details=f'{species}, L{level}',belief=exact,exact_spreads=True,transformed=transformed,transform_target=target)
        return [row[getattr(self.reference,'NUMERIC_'+n)] for n in self.e.CORE]

    def assert_row(self,patch,want):
        self.assertEqual(patch.admitted.tolist(),[True])
        for col,value in zip(self.e.NUMERIC_COLUMNS,want):self.assertAlmostEqual(float(patch.numeric[0,0,col]),value,places=6)
        self.assertEqual(int(patch.numeric_columns.sum()),10);self.assertFalse(bool(patch.categorical_columns.any()))
        self.assertFalse(bool(patch.numeric[0,1:].any()))

    def own_request(self,owner=0,*,stats=None,duplicate=None):
        members=[{'ident':f'p{owner+1}: Absol','details':'Absol, L80','condition':'100/100',
            'stats':stats or {'atk':171,'def':111,'spa':121,'spd':131,'spe':141},'moves':['tackle'],'active':True}]
        if duplicate is not None:members.append({'ident':f'p{owner+1}: second','details':'Absol, L80',
            'condition':'100/100','stats':duplicate,'moves':['tackle'],'active':False})
        return {'wait':True,'side':{'id':f'p{owner+1}','pokemon':members}}

    def test_every_level_and_source_variant_matches_integer_core_and_writer(self):
        import numpy as np
        cases=[(level,index) for level in range(1,101) for index in range(6)]
        base=self.state();batch=len(cases)
        current=list(self.jax.tree.map(lambda x:self.j.repeat(x,batch,axis=0),base))
        levels=self.j.array([x[0] for x in cases],self.j.int32)
        current[2]=current[2]._replace(level=current[2].level.at[:,1,0].set(levels))
        sid=self.host.species_id('Snorlax')
        masks=np.stack([self.host.pin_mask('Snorlax',[self.host.variants[sid][index].to_summary()]) for _,index in cases])
        current[0]=current[0]._replace(hard=current[0].hard.at[:,0,0].set(masks))
        current[0],current[1]=self.b.materialize(current[0],self.host.device)
        actual=self.jax.jit(lambda b,s,m,a:self.e.project_opponent(b,s,m,a,self.tables,
            observer_side=0,expected_binding=self.tables.binding))(*current)
        valid=invalid=0
        for row,(level,index) in enumerate(cases):
            try:want=self.reference_row(level=level,selected=[index])
            except ValueError:
                self.assertFalse(bool(actual.admitted[row]));invalid+=1;continue
            self.assertTrue(bool(actual.admitted[row]));valid+=1
            np.testing.assert_allclose(np.asarray(actual.numeric[row,0,list(self.e.NUMERIC_COLUMNS)]),want,rtol=0,atol=1e-6)
        self.assertEqual(valid+invalid,600);self.assertGreater(valid,500)

    def test_posteriors_bounds_no_source_unknown_and_mask_match_source(self):
        for observer in (0,1):
            for subset in (None,[0],[1],[2,3],[0,1,4,5]):
                self.assert_row(self.project(self.state(observer=observer,selected=subset),observer=observer),self.reference_row(selected=subset))
        self.assert_row(self.project(self.state(available=False)),self.reference_row(selected=[]))
        self.assertFalse(bool(self.project(self.state(),exact=False).numeric.any()))
        self.assert_row(self.project(self.state(species='Unown-Z')),self.reference_row(species='Unown-Z',selected=[]))

    def test_transform_uses_own_target_stats_last_duplicate_and_original_hp(self):
        for observer in (0,1):
            current=list(self.state(observer=observer))
            donor=self.species_catalog.species_id('Absol')
            current[3]=current[3]._replace(transformed=current[3].transformed.at[0,1-observer,0].set(True),
                transform_species=current[3].transform_species.at[0,1-observer,0].set(donor))
            self.assert_row(self.project(current,observer=observer),[0.]*10)
            for duplicate in (None,{'atk':711,'def':710,'spa':709,'spd':708,'spe':707},{'atk':300}):
                raw=self.own_request(observer,duplicate=duplicate)
                targets=self.e.parse_own_targets(raw,owner_side=observer,catalog=self.catalog)
                target=self.reference._self_team_from_request(raw,f'p{observer+1}')[-1]
                self.assert_row(self.project(current,observer=observer,targets=targets),self.reference_row(transformed=True,target=target))

    def test_source_unavailable_before_illegal_spread_falls_back_in_source_order(self):
        from unittest.mock import patch
        real=self.reference._variant_spread_stats
        def controlled(base,level,variant,physical):
            moves=variant['moves']
            if 'Return' in moves:return None
            if 'Rest' in moves:raise ValueError('fixture illegal spread')
            return real(base,level,variant,physical)
        with patch.object(self.reference,'_variant_spread_stats',side_effect=controlled):
            catalog=self.e.compile_expected(self.host,self.dex,self.species_catalog)
            for selected in ([0,1],[0,2],[2]):
                self.assert_row(self.project(self.state(selected=selected),tables=catalog.device),self.reference_row(selected=selected))
            with self.assertRaises(ValueError):self.reference_row(selected=[1,2])
            rejected=self.project(self.state(selected=[1,2]),tables=catalog.device)
            self.assertFalse(bool(rejected.admitted.any()));self.assertFalse(bool(rejected.numeric.any()))

    def test_shape_namespace_ownership_and_materialized_posterior_guards(self):
        state=list(self.state());targets=self.e.parse_own_targets(self.own_request(),owner_side=0,catalog=self.catalog)
        for bad in (targets._replace(owner_side=self.j.array(1,self.j.int32)),
                    targets._replace(species_binding=targets.species_binding.at[0].add(1)),
                    targets._replace(vocabulary_binding=targets.vocabulary_binding.at[0].add(1)),
                    targets._replace(stats=targets.stats.at[0,1].set(float('nan')))):
            got=self.project(state,targets=bad);self.assertFalse(bool(got.admitted.any()));self.assertFalse(bool(got.numeric.any()))
        for index,bad in ((1,state[1]._replace(count=state[1].count.at[0,0,0].add(1))),
                          (0,state[0]._replace(hard=state[0].hard.at[0,0,0,1].set(self.j.uint32(1<<31)))),
                          (2,state[2]._replace(source_species=state[2].source_species.at[0,1,0].set(0))),
                          (2,state[2]._replace(species=state[2].species.at[0,1,0].set(self.species_catalog.species_id('Absol')))),
                          (2,state[2]._replace(level=state[2].level.at[0,1,0].set(0))),
                          (3,state[3]._replace(transformed=state[3].transformed.at[0,1,0].set(True))),
                          (3,state[3]._replace(binding=state[3].binding.at[0,0].add(1)))):
            current=list(state);current[index]=bad;got=self.project(current)
            self.assertFalse(bool(got.admitted.any()));self.assertFalse(bool(got.numeric.any()))
        for bad in (True,.5,self.j.array([0.]),self.j.zeros((1,1),self.j.int32)):
            with self.assertRaises(ValueError):self.project(state,observer=bad)
        with self.assertRaises(ValueError):self.project(state,targets=targets._replace(known=targets.known.astype(self.j.int32)))
        with self.assertRaises(ValueError):self.project(state,targets=targets._replace(species=targets.species[:,None]))
        with self.assertRaises(ValueError):self.e.parse_own_targets(self.own_request(stats={'atk':10**100}),owner_side=0,catalog=self.catalog)
        # The actual source decoder discards raw float stats before this
        # component sees them; preserve missingness instead of inventing a value.
        omitted=self.e.parse_own_targets(self.own_request(stats={'atk':1e100}),owner_side=0,catalog=self.catalog)
        self.assertFalse(bool(omitted.known[0,1]));self.assertEqual(float(omitted.stats[0,1]),0.)
        from pokezero.dex import ShowdownDex
        from dataclasses import replace
        species=dict(self.dex.species);species['snorlax']=replace(species['snorlax'],base_stats=dict(species['snorlax'].base_stats,hp=200))
        with self.assertRaises(ValueError):self.e.compile_expected(self.host,ShowdownDex(self.dex.moves,species,self.dex.type_chart),self.species_catalog)

    def test_stacked_own_targets_jit_refuses_only_wrong_namespace_row(self):
        current=list(self.jax.tree.map(lambda x:self.j.repeat(x,2,axis=0),self.state()))
        donor=self.species_catalog.species_id('Absol')
        current[3]=current[3]._replace(transformed=current[3].transformed.at[:,1,0].set(True),
            transform_species=current[3].transform_species.at[:,1,0].set(donor))
        packet=self.e.parse_own_targets(self.own_request(),owner_side=0,catalog=self.catalog)
        packet=self.jax.tree.map(lambda x:self.j.stack((x,x)),packet)
        packet=packet._replace(vocabulary_binding=packet.vocabulary_binding.at[1,0].add(1))
        out=self.jax.jit(lambda st,su,mo,ab,targets:self.e.project_opponent(st,su,mo,ab,self.tables,
            observer_side=self.j.array([0,0],self.j.int32),expected_binding=self.tables.binding,own_targets=targets))(*current,packet)
        self.assertEqual(out.admitted.tolist(),[True,False]);self.assertTrue(bool(out.numeric[0].any()))
        self.assertFalse(bool(out.numeric[1].any()))


if __name__=='__main__':unittest.main()
