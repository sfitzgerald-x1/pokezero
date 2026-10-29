"""Role placement and owner-order joins against source section writers."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class V4AssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_private_actions as fixture
        fixture.PrivateActionTests.setUpClass.__func__(cls)
        from gpu_engine import v4_assembly as assembly,private_team,public_field,public_tendency
        from pokezero.category_vocab import CategoryVocabulary
        labels=set(cls.vocab.tokens)|{'field'}|{'request_kind:'+n for n in
            ('none','wait','team_preview','force_switch','move','unknown')}|{'weather:'+n for n in
            ('raindance','sunnyday','sandstorm','hail')}
        cls.vocab=CategoryVocabulary(tuple(sorted(labels)))
        cls.catalog=cls.p.compile_actions(cls.dex,cls.vocab)
        cls.assembly,cls.pt,cls.field,cls.tendency=assembly,private_team,public_field,public_tendency
        cls.field_tables=public_field.compile_field(cls.vocab).device

    request=__import__('test_gpu_engine_private_actions').PrivateActionTests.request
    expected=__import__('test_gpu_engine_private_actions').PrivateActionTests.expected

    def blocks(self,raws):
        facts=[self.p.parse_action_facts(raw,owner_side=i,vocabulary=self.vocab,catalog=self.catalog)
            for i,raw in enumerate(raws)]
        facts=self.jax.tree.map(lambda *x:self.j.stack(x),*facts);b=len(raws);observer=self.j.arange(b,dtype=self.j.int32)
        own=self.pt.project_team(facts.team,observer_side=observer,expected_vocabulary_binding=self.pt.vocabulary_binding(self.vocab))
        field=self.field.project(self.field.initialize(b,self.field_tables),self.field_tables,facts.request,
            observer_side=observer,expected_vocabulary=self.pt.vocabulary_binding(self.vocab))
        actions=self.p.project_actions(facts,self.catalog.device,observer_side=observer)
        opponent=self.pt.TeamFeaturePatch(self.j.zeros((b,6,132),self.j.float32),self.j.zeros((b,6,41),self.j.int32),
            self.j.zeros(132,bool),self.j.zeros(41,bool),self.j.ones(b,bool))
        tendency=self.tendency.TendencyProjection(self.j.zeros((b,6,132),self.j.float32),
            self.j.zeros((b,132),self.j.float32),self.j.zeros(b,self.j.int32),self.j.zeros(132,bool),
            self.j.zeros(132,bool),self.j.zeros(b,bool),self.j.ones(b,bool))
        return dict(field=field,own=own,opponent=opponent,actions=actions,tendency=tendency,request=facts.request,
            observer_side=observer,opponent_present=self.j.zeros((b,6),bool),category_count=self.vocab.size)

    def test_real_request_sections_and_source_token_attention_both_seats(self):
        from dataclasses import replace
        raws=[self.request(size=3,active=1,owner=i,moves=[{'id':'surf','pp':0,'maxpp':24,'disabled':True}]) for i in (0,1)]
        blocks=self.blocks(raws);count=blocks.pop('category_count')
        out=self.jax.jit(lambda x:self.assembly.assemble_sections(**x,category_count=count))(blocks)
        self.assertEqual(out.admitted.tolist(),[True,True])
        self.assertEqual(out.numeric_features.shape,(2,23,132));self.assertEqual(out.categorical_ids.shape,(2,23,41))
        for owner,raw in enumerate(raws):
            num,cat,legal=self.expected(raw,owner)
            self.assertTrue(bool(self.j.allclose(out.numeric_features[owner,13:22],num,atol=1e-6,rtol=0)))
            self.assertTrue(bool(self.j.array_equal(out.categorical_ids[owner,13:22],cat)))
            self.assertEqual(out.legal_action_mask[owner].tolist(),list(legal))
            nums=[[0.]*200 for _ in range(6)];cats=[['']*60 for _ in range(6)]
            team=self.s._self_team_from_request(raw,f'p{owner+1}')
            self.s._encode_pokemon_tokens(cats,nums,0,team,role='self',limit=6,dex=self.dex,
                schema_v3=True,schema_v4=True,masks=replace(self.s.DEFAULT_OBSERVATION_FEATURE_MASKS,exact_state=False))
            for slot in range(6):
                for name,column in zip(self.pt.CORE,self.pt.NUMERIC_COLUMNS):
                    self.assertAlmostEqual(float(out.numeric_features[owner,slot+1,column]),nums[slot][getattr(self.s,'NUMERIC_'+name)],places=6)
            self.assertEqual(out.attention_mask[owner].tolist(),[True]+[True]*3+[False]*9+[True]*9+[False])
            self.assertEqual(out.token_type_ids[owner].tolist(),list(self.s._token_type_ids(self.s.DEFAULT_REPLAY_OBSERVATION_SPEC))[:23])
        self.assertTrue(bool(out.numeric_written[:,13:22].all()))
        self.assertFalse(bool(out.numeric_written.all()))  # no completeness from a partial writer union

    def test_owner_late_overlay_and_category_bound_refuse_without_legal_actions(self):
        blocks=self.blocks([self.request(size=3,active=0)])
        out=self.assembly.assemble_sections(**blocks)
        patch=blocks['own']._replace(categorical=blocks['own'].categorical.at[0,0,0].set(self.vocab.size))
        bad=self.jax.jit(lambda x,p:self.assembly.overlay(x,p,section='own'))(out,patch)
        self.assertEqual(bad.admitted.tolist(),[False]);self.assertFalse(bool(bad.legal_action_mask.any()))
        wrong=dict(blocks,observer_side=self.j.array([1],self.j.int32))
        bad=self.assembly.assemble_sections(**wrong);self.assertEqual(bad.admitted.tolist(),[False])
        self.assertEqual(bad.switch_targets.tolist(),[[-1]*5])
        tendency=blocks['tendency']._replace(stats_numeric=blocks['tendency'].stats_numeric.at[0,0].set(.75))
        bad=self.assembly.assemble_sections(**dict(blocks,tendency=tendency))
        self.assertEqual(bad.admitted.tolist(),[False])
        column=self.s.v4_numeric_index(self.s.NUMERIC_MON_TURNS_ACTIVE_TOTAL)
        tendency=blocks['tendency']._replace(numeric=blocks['tendency'].numeric.at[0,2,column].set(.5),
            numeric_columns=blocks['tendency'].numeric_columns.at[column].set(True))
        out=self.assembly.assemble_sections(**dict(blocks,tendency=tendency))
        self.assertEqual(float(out.numeric_features[0,9,column]),.5)
        self.assertTrue(bool(out.numeric_written[0,9,column]))

    def test_public_own_order_join_does_not_borrow_opponent_order_or_categories(self):
        from types import SimpleNamespace
        j=self.j;b=2;binding=j.broadcast_to(j.arange(8,dtype=j.uint32),(b,8))
        species=j.array([[[7,3,0,0,0,0],[3,7,0,0,0,0]],[[7,3,0,0,0,0],[3,7,0,0,0,0]]],j.int32)
        mons=SimpleNamespace(species=species,encounter=(species>0).astype(j.int32),species_binding=binding)
        own=SimpleNamespace(owner_side=j.array([0,1],j.int32),species=j.array([[3,7,9,0,0,0]]*b,j.int32),
            present=j.array([[True,True,True,False,False,False]]*b),species_binding=binding)
        nums=j.zeros((b,2,6,132),j.float32).at[:,:,:,0].set(j.array([[[.7,.3,0,0,0,0],[.3,.7,0,0,0,0]]]*b))
        patch=self.pt.TeamFeaturePatch(nums,j.zeros((b,2,6,41),j.int32),j.arange(132)==0,j.zeros(41,bool),j.ones(b,bool))
        out=self.assembly.remap_public_own(patch,mons,own,observer_side=j.array([0,1],j.int32))
        self.assertEqual(out.admitted.tolist(),[True,True]);self.assertTrue(bool(j.allclose(out.numeric[:,:,0],j.array([[.3,.7,0,0,0,0]]*b))))
        self.assertEqual(out.numeric_columns[:,:,0].tolist(),[[True,True,False,False,False,False]]*b)
        own.species=own.species.at[0,1].set(3)
        out=self.assembly.remap_public_own(patch,mons,own,observer_side=j.array([0,1],j.int32))
        self.assertEqual(out.admitted.tolist(),[False,True])

    def test_active_category_columns_and_boosts_use_actual_source_writers(self):
        from gpu_engine import public_sleep
        import test_gpu_engine_public_refresh_context as fixture
        fixture.PublicRefreshContextTests.setUpClass()
        helper=fixture.PublicRefreshContextTests('test_actual_raw_source_callbacks_and_choice_both_modes')
        lines=[*helper.prefix,'|-boost|p1a: Snorlax|atk|2',
            '|move|p1a: Snorlax|Body Slam|p2a: Absol',
            '|-ability|p1a: Snorlax|Pressure|[from] ability: Trace|[of] p2a: Absol']
        result,_=helper.run_lines(lines)
        parsed=self.s.parse_showdown_replay(lines,battle_id='active-assembly')
        cats=self.assembly.project_active_categories(result.context,helper.contexts[False],result.mon_state)
        self.assertEqual(cats.admitted.tolist(),[True])
        expected=['']*41;nums=[0.]*200
        self.s._encode_active_feature_pack(expected,nums,truant_loaf=False,last_used_move=parsed.last_used_move['p1'],
            arrived_by_baton_pass=False,choice_locked=False,item_swapped=False,traced_ability=parsed.traced_ability['p1'],
            last_damage_dealt=0.,last_damage_taken=0.)
        for column in (self.s.CATEGORY_LAST_USED_MOVE,self.s.CATEGORY_TRACED_ABILITY):
            self.assertEqual(int(cats.categorical[0,0,column]),helper.vocab.encode(expected[column]))
        dark=self.assembly.project_active_categories(result.context,helper.contexts[False],result.mon_state,feature_pack_last_move=False)
        self.assertEqual(int(dark.categorical[0,0,self.s.CATEGORY_LAST_USED_MOVE]),0)
        self.assertEqual(int(dark.categorical[0,0,self.s.CATEGORY_TRACED_ABILITY]),int(cats.categorical[0,0,self.s.CATEGORY_TRACED_ABILITY]))
        boost=self.assembly.project_own_boosts(result.mon_state)
        self.s._encode_active_boosts(nums,{'atk':2})
        column=self.s.v4_numeric_index(self.s.NUMERIC_BOOST_ATK)
        self.assertAlmostEqual(float(boost.numeric[0,0,column]),nums[self.s.NUMERIC_BOOST_ATK],places=6)
        raw={'side':{'id':'p1','pokemon':[{'ident':'p1: Snorlax','details':'Snorlax, L80','condition':'100/100','active':True}]}}
        own=public_sleep.parse_own_status(raw,owner_side=0,species_catalog=helper.species_catalog,vocabulary=helper.vocab)
        own=self.jax.tree.map(lambda x:x[None],own)
        # Public active columns must not spread over unseen own bench rows.
        full=self.pt.TeamFeaturePatch(boost.numeric[:,:,None,:]*self.j.ones((1,2,6,1)),
            self.j.zeros((1,2,6,41),self.j.int32),boost.numeric_columns,self.j.zeros(41,bool),boost.admitted)
        remapped=self.assembly.remap_public_own(full,result.mon_state,own,observer_side=self.j.array([0],self.j.int32))
        self.assertEqual(remapped.admitted.tolist(),[True]);self.assertFalse(bool(remapped.numeric_columns[:,1:].any()))

    def test_real_opponent_tables_from_a_different_vocabulary_refuse(self):
        import test_gpu_engine_public_refresh_context as fixture
        fixture.PublicRefreshContextTests.setUpClass()
        helper=fixture.PublicRefreshContextTests('test_actual_raw_source_callbacks_and_choice_both_modes')
        public,_=helper.run_lines(helper.prefix)
        raw=self.request(size=3,active=0)
        facts=self.p.parse_action_facts(raw,owner_side=0,vocabulary=self.vocab,catalog=self.catalog)
        facts=self.jax.tree.map(lambda x:x[None],facts)
        context=public.context._replace(field=self.field.initialize(1,self.field_tables))
        args=dict(facts=facts,action_tables=self.catalog.device,mons=public.mon_state,context=context,
            team_tables=helper.species_catalog.device,field_tables=self.field_tables,
            tendency=self.blocks([raw])['tendency'],observer_side=self.j.array([0],self.j.int32),
            vocabulary_binding=self.pt.vocabulary_binding(self.vocab),category_count=self.vocab.size)
        out=self.jax.jit(lambda facts:self.assembly.project_base_sections(**dict(args,facts=facts)))(facts)
        self.assertEqual(out.admitted.tolist(),[False]);self.assertFalse(bool(out.legal_action_mask.any()))


if __name__=='__main__':unittest.main()
