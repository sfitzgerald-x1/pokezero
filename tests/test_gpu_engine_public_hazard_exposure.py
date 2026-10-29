"""Future Spikes exposure compared with the actual field-credit source."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicHazardExposureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import dataclasses
        import test_gpu_engine_public_belief_features as fixtures
        fixtures.PublicBeliefFeatureTests.setUpClass.__func__(cls)
        from gpu_engine import public_hazard_exposure as h,public_field,public_team
        from pokezero.dex import ShowdownDex,SpeciesInfo
        from pokezero.category_vocab import CategoryVocabulary
        cls.h,cls.field=h,public_field
        universes=dict(cls.source.universes);old=universes['gardevoir']
        universes['gardevoir']=dataclasses.replace(old,variants=tuple(dataclasses.replace(old.variants[0],
            variant_id='hazard-'+a,ability=a) for a in ('Trace','Levitate')))
        cls.source=dataclasses.replace(cls.source,universes=universes);cls.host=cls.b.compile_source(cls.source)
        species={key:SpeciesInfo(key,name,types,{}) for key,name,types in (
            ('snorlax','Snorlax',('Normal',)),('absol','Absol',('Dark',)),
            ('gardevoir','Gardevoir',('Psychic',)),('unown','Unown',('Flying',)))}
        cls.dex=ShowdownDex(cls.dex.moves,species,{})
        cls.vocab=CategoryVocabulary(tuple(dict.fromkeys((*cls.vocab.tokens,'type:normal','type:dark','type:psychic','type:flying'))))
        cls.catalog=cls.p.compile_moves(cls.host,cls.dex,cls.vocab)
        cls.abilities=cls.a.compile_abilities(cls.host,cls.vocab,ability_names=('Pressure','Trace','Immunity','Levitate'))
        cls.item_catalog=cls.items.compile_items(cls.host,cls.vocab,item_names=('Mystery Orb',))
        cls.species_catalog=public_team.compile_species(cls.host,cls.dex,cls.vocab,cls.abilities,extra_species=('Unown-Z',))
        cls.field_catalog=public_field.compile_field(cls.vocab)
        cls.feature_tables=cls.f.compile_features(cls.host,cls.abilities,cls.item_catalog,cls.catalog,cls.vocab,species_catalog=cls.species_catalog)
        cls.hazard_catalog=h.compile_hazard(cls.species_catalog,cls.dex,cls.vocab,
            feature_tables=cls.feature_tables,field_tables=cls.field_catalog.device,abilities=cls.abilities)
        cls.tables=cls.hazard_catalog.device
        cls.contexts={n:cls.c.compile_context(cls.host,cls.abilities,cls.catalog,items=cls.item_catalog,
            item_belief_narrowing=n,field_catalog=cls.field_catalog) for n in (False,True)}
        cls.kwargs=dict(species_catalog=cls.species_catalog,source=cls.host,moves=cls.catalog,abilities=cls.abilities,
            items=cls.item_catalog,field_catalog=cls.field_catalog)

    project=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.project
    initial=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.initial
    fold=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.fold
    carry=staticmethod(__import__('test_gpu_engine_public_item_context').PublicItemContextTests.carry)
    run_lines=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.run_lines
    chunks=__import__('test_gpu_engine_public_field_context').PublicFieldContextTests.chunks

    def request(self,rows,owner=0):
        return {'side':{'id':f'p{owner+1}','pokemon':[dict(ident=f'p{owner+1}: {name}',details=name+', L80',
            condition=condition,active=active,**ability) for name,condition,active,ability in rows]}}

    def visible(self,result):
        from gpu_engine import public_belief_snapshot as snapshot
        state=self.b.initialize(self.j.zeros_like(result.beliefs.species),self.host.device)
        cache=snapshot.initialize(state,self.host.device)
        cache,beliefs,ok=snapshot.refresh(cache,result.beliefs,self.host.device,
            source_refresh=self.j.ones_like(result.beliefs.species,bool))
        self.assertTrue(bool(ok.all()));return cache,beliefs

    def encode(self,result,request,observer=0,own=None,visible=None):
        own=own or self.h.parse_own_hazard(request,owner_side=observer,catalog=self.hazard_catalog,vocabulary=self.vocab)
        if visible is None:visible,beliefs=self.visible(result)
        else:beliefs=result.beliefs
        return self.h.project(own,result.mon_state,result.context,beliefs,visible,self.host.device,result.context.field,
            self.tables,self.feature_tables,observer_side=observer,expected_binding=self.tables.binding)

    def compare(self,result,lines,request,observer=0,ref=None,visible=None):
        from types import SimpleNamespace
        ref=ref or self.Reference.from_events([self.reference._public_event_from_line(x) for x in lines],
            format_id='gen3randombattle',set_source=self.source)
        replay=self.reference.parse_showdown_replay(lines)
        beliefs={self.reference._normalize_identifier(x.species):x for x in ref.snapshot().side(f'p{2-observer}')}
        state=SimpleNamespace(self_team=self.reference._self_team_from_request(request,f'p{observer+1}'),
            opponent_team=replay.public_revealed.get(f'p{2-observer}',()),
            belief_view=SimpleNamespace(opponent_by_species=lambda:beliefs),
            self_side_condition_counts={'spikes':int(result.context.field.spikes[0,observer])},
            opponent_side_condition_counts={'spikes':int(result.context.field.spikes[0,1-observer])},
            self_hazard_damage_suffered=0,opponent_hazard_damage_suffered=0,self_items_removed=0,opponent_items_removed=0)
        expected=self.reference.field_credit_values(state,dex=self.dex)
        got=self.encode(result,request,observer,visible=visible);self.assertEqual(got.admitted.tolist(),[True])
        for name,col in zip(('self_hazard_expected','opponent_hazard_expected'),self.h.COLUMNS):
            self.assertAlmostEqual(float(got.numeric[0,col]),expected[name],places=7,msg=name)
        self.assertEqual(int(got.numeric_columns.sum()),2)
        return got

    def test_layers_unseen_faint_and_owner_condition_facts_match_source(self):
        request=self.request([('Snorlax','100/100',True,{}),('Gardevoir','0/100',False,{}),
            ('Unown','100/100',False,{}),('Unown-Z','100/100',False,{}),
            ('Absol','0 fnt',False,{}),('Absol','',False,{'ability':'','baseAbility':'Levitate'})])
        prefix=['|switch|p1a: Snorlax|Snorlax, L80|100/100',
            '|switch|p2a: Gardevoir|Gardevoir, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100',
            '|faint|p2a: Absol','|switch|p2a: Unown-Z|Unown-Z, L80|100/100']
        for layers in range(4):
            lines=[*prefix,*(['|-sidestart|p1: own|Spikes','|-sidestart|p2: foe|Spikes']*layers)]
            result,_=self.run_lines(lines);self.compare(result,lines,request)
        result,_=self.run_lines(prefix);self.compare(result,prefix,None)

    def test_original_reveal_singleton_and_ambiguous_levitate_use_public_posterior(self):
        lines=['|switch|p1a: Snorlax|Snorlax, L80|100/100','|switch|p2a: Gardevoir|Gardevoir, L80|100/100',
            '|switch|p2a: Absol|Absol, L80|100/100','|-sidestart|p2: foe|Spikes']
        request=self.request([('Snorlax','100/100',True,{})])
        result,_=self.run_lines(lines);before=self.compare(result,lines,request)
        visible,_=self.visible(result)
        running=result.context.abilities._replace(running=result.context.abilities.running.at[0,1,0].set(self.abilities.ability_id('Levitate')))
        borrowed=result._replace(context=result.context._replace(abilities=running))
        self.assertEqual(self.encode(borrowed,request).numeric.tolist(),before.numeric.tolist())
        ref=self.Reference.from_events([self.reference._public_event_from_line(x) for x in lines],
            format_id='gen3randombattle',set_source=self.source)
        survivors=[v.to_summary() for v in self.host.variants[self.host.species_id('Gardevoir')] if v.ability=='Levitate']
        mask=self.host.pin_mask('Gardevoir',survivors);valid=self.j.zeros((1,2,6),bool).at[0,0,0].set(True)
        beliefs,_=self.b.narrow_pin(result.beliefs,mask,valid=valid);result=result._replace(beliefs=beliefs)
        key=next(x.key for x in ref.snapshot().side('p2') if x.species=='Gardevoir');ref.narrow_candidate_variants(key,survivors)
        pending=self.compare(result,lines,request,ref=ref,visible=visible)
        self.assertEqual(pending.numeric.tolist(),before.numeric.tolist())
        # Pin registration alone leaves Python's cached snapshot unchanged.
        # Compare the already-materialized posterior at the same evidence epoch.
        refresh=['|switch|p2a: Gardevoir|Gardevoir, L80|100/100',
            '|switch|p2a: Absol|Absol, L80|100/100']
        host,_=self.project(lines);_,extra=self.project(refresh,state=host)
        result=self.fold(self.carry(result),self.chunks(extra)[0]);self.assertEqual(result.error.tolist(),[0])
        for raw in refresh:ref.ingest_event(self.reference._public_event_from_line(raw))
        lines.extend(refresh)
        from gpu_engine import public_belief_snapshot as snapshot
        refresh_mask=self.j.zeros_like(result.beliefs.species,bool).at[0,0,:2].set(True)
        visible,beliefs,ok=snapshot.refresh(visible,result.beliefs,self.host.device,source_refresh=refresh_mask)
        self.assertTrue(bool(ok.all()));result=result._replace(beliefs=beliefs)
        after=self.compare(result,lines,request,ref=ref,visible=visible)
        self.assertLess(float(after.numeric[0,self.h.COLUMNS[1]]),float(before.numeric[0,self.h.COLUMNS[1]]))
        # Running copied Levitate must not become original revealed Levitate.
        running=result.context.abilities._replace(running=result.context.abilities.running.at[0,1,0].set(self.abilities.ability_id('Levitate')))
        changed=result._replace(context=result.context._replace(abilities=running))
        self.assertEqual(self.encode(changed,request).numeric.tolist(),after.numeric.tolist())
        for name in ('Levitate','Immunity'):
            extra=[*lines,f'|-ability|p2a: Gardevoir|{name}'];candidate,_=self.run_lines(extra)
            self.compare(candidate,extra,request)

    def test_private_active_both_seats_and_all_active_rows_are_excluded(self):
        lines=['|switch|p1a: Snorlax|Snorlax, L80|100/100','|switch|p2a: Gardevoir|Gardevoir, L80|100/100',
            '|-sidestart|p1: own|Spikes','|-sidestart|p2: foe|Spikes']
        result,_=self.run_lines(lines)
        for observer in (0,1):
            request=self.request([('Gardevoir','100/100',True,{}),('Snorlax','100/100',True,{}),
                ('Absol','0/100',False,{}),('Absol','100/100',False,{})],owner=observer)
            self.compare(result,lines,request,observer)

    def test_packet_namespace_summary_and_jit_mixed_row_refusals(self):
        lines=['|switch|p1a: Snorlax|Snorlax, L80|100/100','|switch|p2a: Gardevoir|Gardevoir, L80|100/100','|-sidestart|p2: foe|Spikes']
        result,_=self.run_lines(lines);request=self.request([('Absol','100/100',False,{})])
        own=self.h.parse_own_hazard(request,owner_side=0,catalog=self.hazard_catalog,vocabulary=self.vocab)
        for bad in (own._replace(owner_side=self.j.array(1,self.j.int32)),
                own._replace(species_binding=own.species_binding.at[0].add(1)),
                own._replace(vocabulary_binding=own.vocabulary_binding.at[0].add(1))):
            got=self.encode(result,request,own=bad);self.assertEqual(got.admitted.tolist(),[False]);self.assertFalse(bool(got.numeric.any()))
        batch=self.jax.tree.map(lambda x:self.j.concatenate([x,x],axis=0),result)
        visible,beliefs=self.visible(batch)
        packet=self.jax.tree.map(lambda x:self.j.stack([x,x]),own)._replace(owner_side=self.j.array([1,0],self.j.int32))
        fn=self.jax.jit(lambda p:self.h.project(p,batch.mon_state,batch.context,beliefs,visible,self.host.device,batch.context.field,
            self.tables,self.feature_tables,observer_side=self.j.array([0,0],self.j.int32),expected_binding=self.tables.binding))
        out=fn(packet);self.assertEqual(out.admitted.tolist(),[False,True]);self.assertFalse(bool(out.numeric[0].any()))
        with self.assertRaises(ValueError):self.h.parse_own_hazard({'side':{'id':'p2'}},owner_side=0,catalog=self.hazard_catalog,vocabulary=self.vocab)
        bad_features=self.feature_tables._replace(variant_ability=self.feature_tables.variant_ability.at[
            self.host.species_id('Gardevoir'),0].set(0))
        with self.assertRaises(ValueError):self.h.compile_hazard(self.species_catalog,self.dex,self.vocab,
            feature_tables=bad_features,field_tables=self.field_catalog.device,abilities=self.abilities)


if __name__=='__main__':unittest.main()
