"""Belief buckets and PP alignment compared with the actual opponent writer."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class PublicBeliefFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import dataclasses
        import test_gpu_engine_public_item_context as fixtures
        fixtures.PublicItemContextTests.setUpClass.__func__(cls)
        from gpu_engine import public_belief_features as features, public_team
        from pokezero.category_vocab import CategoryVocabulary
        cls.f=features;cls.fixtures=fixtures
        universes=dict(cls.source.universes);original=universes['snorlax'];rows=[]
        for i in range(12):
            pair=(30,31) if i==0 else (30,29) if i==11 else (2*(i-1),2*(i-1)+1)
            rows.append(dataclasses.replace(original.variants[0],variant_id=f'features-{i}',
                moves=('Rest','Protect',*(f'move{x:02d}' for x in pair)),
                item='Choice Band' if i%2 else 'Leftovers'))
        universes['snorlax']=dataclasses.replace(original,variants=tuple(rows))
        cls.source=dataclasses.replace(cls.source,universes=universes);cls.host=cls.b.compile_source(cls.source)
        labels=['field','pokemon:opponent','species:snorlax','species:absol','status:none']
        for kind,names in (('possible_move',[f'move{i:02d}' for i in range(40)]+['rest','protect','shadowball','bodyslam']),
                           ('possible_ability',['immunity','pressure','trace','levitate']),
                           ('possible_item',['leftovers','choiceband','mysteryorb'])):
            labels.extend('belief:'+kind+':'+n for n in names)
        cls.vocab=CategoryVocabulary(tuple(labels))
        cls.catalog=cls.p.compile_moves(cls.host,cls.dex,cls.vocab)
        cls.abilities=cls.a.compile_abilities(cls.host,cls.vocab,ability_names=('Immunity','Pressure','Trace','Levitate'))
        cls.item_catalog=cls.items.compile_items(cls.host,cls.vocab,item_names=('Mystery Orb',))
        cls.species_catalog=public_team.compile_species(cls.host,cls.dex,cls.vocab,cls.abilities)
        cls.contexts={n:cls.c.compile_context(cls.host,cls.abilities,cls.catalog,
            items=cls.item_catalog,item_belief_narrowing=n) for n in (False,True)}
        cls.kwargs=dict(species_catalog=cls.species_catalog,source=cls.host,moves=cls.catalog,
            abilities=cls.abilities,items=cls.item_catalog)
        cls.tables=features.compile_features(cls.host,cls.abilities,cls.item_catalog,cls.catalog,cls.vocab,
            species_catalog=cls.species_catalog)

    # Reuse the reviewed real-wire transport helper, not another event adapter.
    project=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.project
    initial=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.initial
    chunks=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.chunks
    fold=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.fold
    carry=staticmethod(__import__('test_gpu_engine_public_item_context').PublicItemContextTests.carry)
    run_lines=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.run_lines

    def encode(self,result,observer=1,exact=True):
        beliefs,summary=self.b.materialize(result.beliefs,self.host.device)
        return self.f.project_opponent(beliefs,summary,result.mon_state,result.context,self.tables,
            observer_side=observer,expected_binding=self.tables.binding,exact_state=exact)

    def compare(self,result,lines,*,observer=1,exact=True,ref=None):
        ref=ref or self.Reference.from_events([self.reference._public_event_from_line(x) for x in lines],
            format_id='gen3randombattle',set_source=self.source)
        replay=self.reference.parse_showdown_replay(lines,battle_id='feature-fixture')
        target=f'p{2-observer}';expected=ref.snapshot().side(target)[0]
        mons=replay.public_revealed[target]
        nums=[[0.]*200 for _ in range(6)];cats=[['']*60 for _ in range(6)]
        from dataclasses import replace
        self.reference._encode_pokemon_tokens(cats,nums,0,mons,role='opponent',limit=6,
            beliefs_by_species={self.reference._normalize_identifier(expected.species):expected},
            exact_beliefs_by_species={self.reference._normalize_identifier(expected.species):expected},
            dex=self.dex,schema_v2_1=True,schema_v3=True,schema_v4=True,
            masks=replace(self.reference.DEFAULT_OBSERVATION_FEATURE_MASKS,exact_state=exact))
        output=self.encode(result,observer,exact);self.assertEqual(output.patch.admitted.tolist(),[True])
        for name,c in zip(self.f.CORE,self.f.NUMERIC_COLUMNS):
            self.assertAlmostEqual(float(output.patch.numeric[0,0,c]),nums[0][getattr(self.reference,'NUMERIC_'+name)],places=6,msg=name)
        for offset in (self.reference.NUMERIC_OPP_MOVE_PP_OFFSET,self.reference.NUMERIC_OPP_MOVE_PP_VALID_OFFSET):
            for i in range(16):
                self.assertAlmostEqual(float(output.patch.numeric[0,0,self.reference.v4_numeric_index(offset+i)]),nums[0][offset+i],places=6)
        for c in self.f.CATEGORY_COLUMNS:
            self.assertEqual(int(output.patch.categorical[0,0,c]),self.vocab.encode(cats[0][c]),f'category{c}')
        bucket=self.reference._compact_belief_values(self.reference._prioritized_belief_moves(
            expected.revealed_moves,expected.possible_moves,16),limit=16)
        self.assertEqual([self.catalog.names[i] for i in output.bucket_move_ids[0,0].tolist() if i],
            [self.reference._normalize_identifier(n) for n in bucket])
        weights=output.posterior_weights[0,0].tolist();variants=self.host.variants[self.host.species_id(expected.species)]
        self.assertEqual([v.variant_id for v,w in zip(variants,weights) if w>0],
            [v['variant_id'] for v in expected.candidate_variants])
        self.assertAlmostEqual(sum(weights),1. if expected.candidate_variants else 0.,places=6)
        self.assertEqual(int(output.patch.numeric_columns.sum()),40);self.assertEqual(int(output.patch.categorical_columns.sum()),24)
        self.assertFalse(bool(output.patch.numeric[0,1:].any()));self.assertFalse(bool(output.patch.categorical[0,1:].any()))
        return output

    def test_source_buckets_narrowing_reveals_items_and_pp_align_with_actual_writer(self):
        cases=[[],['|-ability|p1a: Snorlax|Immunity'],['|-item|p1a: Snorlax|Leftovers'],
            ['|move|p1a: Snorlax|move00|p2a: Absol']*20,
            ['|-ability|p2a: Absol|Pressure','|move|p1a: Snorlax|move00|p2a: Absol'],
            ['|-item|p1a: Snorlax|Leftovers','|-enditem|p1a: Snorlax|Leftovers|[eat]']]
        for suffix in cases:
            lines=[*self.prefix,*suffix];result,_=self.run_lines(lines)
            for exact in (False,True):self.compare(result,lines,exact=exact)
            self.compare(result,lines,observer=0)

    def test_off_source_reveals_have_priority_over_large_possible_union(self):
        lines=[*self.prefix,'|move|p1a: Snorlax|move39|p2a: Absol']
        result,_=self.run_lines(lines);output=self.compare(result,lines)
        names=[self.catalog.names[i] for i in output.bucket_move_ids[0,0].tolist() if i]
        self.assertIn('move39',names);self.assertEqual(len(names),16)
        self.assertEqual(int(output.reason[0,0]),self.b.INCONSISTENT_REVEALS)

    def test_standing_pin_filters_source_order_instead_of_reordering_from_kept_variants(self):
        lines=list(self.prefix);result,_=self.run_lines(lines)
        variants=self.host.variants[self.host.species_id('Snorlax')]
        survivors=[v.to_summary() for v in variants[1:]]
        pin=self.host.pin_mask('Snorlax',survivors)
        valid=self.j.zeros((1,2,6),bool).at[0,1,0].set(True)
        full=self.j.broadcast_to(pin,result.beliefs.hard.shape)
        beliefs,_=self.b.narrow_pin(result.beliefs,full,valid=valid)
        result=result._replace(beliefs=beliefs)
        ref=self.Reference.from_events([self.reference._public_event_from_line(x) for x in lines],
            format_id='gen3randombattle',set_source=self.source)
        ref.narrow_candidate_variants(ref.snapshot().side('p1')[0].key,survivors)
        # The source applies a standing pin when a public reveal re-summarizes;
        # registering the pin alone does not rewrite its existing snapshot.
        refresh='|-ability|p1a: Snorlax|Immunity'
        host,_=self.project(lines)
        _,extra=self.project([refresh],state=host)
        result=self.fold(self.carry(result),self.chunks(extra)[0])
        self.assertEqual(result.error.tolist(),[0])
        ref.ingest_event(self.reference._public_event_from_line(refresh));lines.append(refresh)
        output=self.compare(result,lines,ref=ref)
        self.assertIn('move30',[self.catalog.names[i] for i in output.bucket_move_ids[0,0].tolist() if i])

    def test_jit_two_observers_and_wrong_catalog_namespace_zero_invalid_rows(self):
        lines=[*self.prefix,'|-item|p1a: Snorlax|Mystery Orb|[from] move: Thief']
        result,_=self.run_lines(lines)
        doubled=self.jax.tree.map(lambda x:self.j.concatenate([x,x],axis=0),result)
        beliefs,summary=self.b.materialize(doubled.beliefs,self.host.device)
        fn=self.jax.jit(lambda m:self.f.project_opponent(beliefs,summary,m,doubled.context,self.tables,
            observer_side=self.j.array([1,0],self.j.int32),expected_binding=self.tables.binding))
        output=fn(doubled.mon_state);self.assertEqual(output.patch.admitted.tolist(),[True,True])
        broken=doubled.mon_state._replace(species_binding=doubled.mon_state.species_binding.at[0,0].add(1))
        bad=fn(broken);self.assertEqual(bad.patch.admitted.tolist(),[False,True])
        self.assertFalse(bool(bad.patch.numeric[0].any()));self.assertFalse(bool(bad.patch.categorical[0].any()))
        for observer in (True,.5,self.j.zeros((2,1),self.j.int32)):
            with self.assertRaises(ValueError):self.f.project_opponent(beliefs,summary,doubled.mon_state,
                doubled.context,self.tables,observer_side=observer,expected_binding=self.tables.binding)
        bad=self.f.project_opponent(beliefs,summary,doubled.mon_state,doubled.context,self.tables,
            observer_side=1,expected_binding=self.tables.binding.at[0].add(1))
        self.assertFalse(bool(bad.patch.admitted.any()))


    def test_no_source_retains_known_reveals_and_malformed_summaries_refuse(self):
        lines=[*self.prefix,'|move|p1a: Snorlax|move39|p2a: Absol']
        result,_=self.run_lines(lines)
        beliefs=result.beliefs._replace(source_available=self.j.zeros_like(result.beliefs.source_available),
            hard=self.j.zeros_like(result.beliefs.hard))
        no_source=result._replace(beliefs=beliefs)
        ref=self.Reference.from_events([self.reference._public_event_from_line(x) for x in lines],
            format_id='gen3randombattle',set_source=None)
        output=self.compare(no_source,lines,ref=ref)
        self.assertEqual(int(output.reason[0,0]),self.b.NO_SOURCE)
        self.assertFalse(bool(output.posterior_weights.any()))
        b,summary=self.b.materialize(result.beliefs,self.host.device)
        for bad in (summary._replace(effective=summary.effective.at[0,1,0,0].set(self.j.uint32(1<<31))),
                    summary._replace(uncertainty=summary.uncertainty.at[0,1,0].set(self.j.nan)),
                    summary._replace(reason=summary.reason.at[0,1,0].set(99))):
            out=self.f.project_opponent(b,bad,result.mon_state,result.context,self.tables,
                observer_side=1,expected_binding=self.tables.binding)
            self.assertEqual(out.patch.admitted.tolist(),[False]);self.assertFalse(bool(out.patch.numeric.any()))
        for bad in (summary._replace(count=summary.count[...,None]),
                    summary._replace(count=summary.count.astype(bool)),
                    summary._replace(effective=summary.effective.astype(self.j.int32))):
            with self.assertRaises(ValueError):self.f.project_opponent(b,bad,result.mon_state,result.context,self.tables,
                observer_side=1,expected_binding=self.tables.binding)


if __name__=='__main__':unittest.main()
