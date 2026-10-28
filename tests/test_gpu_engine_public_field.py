"""Device field events against actual public parser and retained v4 writer."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class PublicFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import public_field as f, private_requests as requests
        from pokezero import showdown as s
        from pokezero.category_vocab import CategoryVocabulary
        cls.jax,cls.j,cls.f,cls.requests,cls.s = jax,j,f,requests,s
        cls.vocab=CategoryVocabulary(tuple(['field']+
            ['request_kind:'+n for n in ('none','wait','team_preview','force_switch','move','unknown')]+
            ['weather:'+n for n in ('raindance','sunnyday','sandstorm','hail')]))
        cls.catalog=f.compile_field(cls.vocab); cls.tables=cls.catalog.device

    def consume(self,state,line):
        p=self.f.project_line(line,catalog=self.catalog)
        self.assertEqual(p.raw_line,line);self.assertFalse(p.full_line_consumers_complete)
        for operation,side,value,upkeep,ability in p.facts:
            state,ok=self.f.apply(state,self.tables,operation=operation,side=side,value=value,
                upkeep=upkeep,from_ability=ability,producer_binding=self.tables.binding)
            self.assertTrue(bool(ok.all()))
        return state

    def assert_writer(self,state,lines,*,exact=True,observer=0,raw=None):
        from dataclasses import replace
        raw=raw or {'wait':True}
        replay=self.s.parse_showdown_replay(lines,battle_id='field-fixture')
        source=self.s.normalize_for_player(replay,player_id='observer',configured_showdown_slot=f'p{observer+1}')
        source=replace(source,request_kind=self.s._request_kind(raw))
        cats=[[None]*51 for _ in range(23)];nums=[[0.]*self.s.V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT for _ in range(23)]
        masks=replace(self.s.DEFAULT_OBSERVATION_FEATURE_MASKS,exact_state=exact)
        self.s._encode_field_token(cats,nums,source,masks=masks,schema_v3=True,schema_v4=True)
        request=self.requests.parse_action_request(raw,owner_side=observer)
        actual=self.f.project(state,self.tables,request,observer_side=observer,
            expected_vocabulary=self.tables.vocabulary_binding,exact_state=exact)
        self.assertEqual(actual.admitted.tolist(),[True])
        expected_num=self.s._project_v4_numeric_rows(nums)[0]
        for c in self.f.NUMERIC_COLUMNS:
            self.assertAlmostEqual(float(actual.numeric[0,c]),expected_num[c],places=6)
        for c in self.f.CATEGORY_COLUMNS:
            self.assertEqual(int(actual.categorical[0,c]),self.vocab.encode(cats[0][c]))
        self.assertEqual(int(actual.numeric_columns.sum()),10)
        self.assertEqual(int(actual.categorical_columns.sum()),3)

    def test_move_weather_upkeep_permanent_recast_and_clear_match_source_writer(self):
        cases=[['|turn|1','|-weather|RainDance', '|-weather|RainDance|[upkeep]', '|upkeep','|turn|2'],
            ['|turn|1','|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Tyranitar']+
                ['|-weather|Sandstorm|[upkeep]']*7,
            ['|turn|1','|-weather|SunnyDay']+['|-weather|SunnyDay|[upkeep]']*6+
                ['|-weather|RainDance','|-weather|RainDance|[upkeep]','|-weather|none'],
            ['|-weather|Hail|[upkeep]','|-weather|Hail'],
            ['|-weather|RainDance|[from] ability: Drizzle','|-weather|']]
        for lines in cases:
            state=self.f.initialize(1,self.tables);prefix=[]
            for line in lines:
                state=self.consume(state,line);prefix.append(line)
                for exact in (False,True):self.assert_writer(state,prefix,exact=exact)

    def test_spikes_counts_saturate_end_and_both_observer_orientations_match(self):
        state=self.f.initialize(1,self.tables);lines=[]
        for line in ['|-sidestart|p1: Alice|Spikes']*4+['|-sidestart|p2: Bob|move: Spikes']*2+['|-sideend|p1: Alice|Spikes']:
            lines.append(line);state=self.consume(state,line)
            for observer in (0,1):self.assert_writer(state,lines,observer=observer)
        self.assertEqual(state.spikes.tolist(),[[0,2]])

    def test_wish_repeated_declaration_side_pass_landing_and_expiry_match(self):
        cases=[['|turn|1','|move|p1a: Snorlax|Wish|p1a: Snorlax','|turn|2',
                '|move|p1a: Snorlax|Wish|p1a: Snorlax','|switch|p1a: Absol|Absol, L80|100/100',
                '|-heal|p1a: Absol|100/100|[from] move: Wish','|turn|3'],
            ['|turn|1','|move|p2a: Absol|Wish|p2a: Absol','|turn|2','|turn|3',
             '|move|p2a: Absol|Wish|p2a: Absol','|-sethp|p2a: Absol|100/100|[from] move: Wish']]
        for lines in cases:
            state=self.f.initialize(1,self.tables);prefix=[]
            for line in lines:
                prefix.append(line);state=self.consume(state,line)
                for observer in (0,1):
                    for exact in (False,True):self.assert_writer(state,prefix,observer=observer,exact=exact)

    def test_batched_jit_admission_preserves_unrelated_or_invalid_rows(self):
        state=self.f.initialize(3,self.tables)
        fn=self.jax.jit(lambda x,side,value:self.f.apply(x,self.tables,operation=self.f.SPIKES,
            side=side,value=value,producer_binding=self.tables.binding))
        updated,ok=fn(state,self.j.array([0,1,2],self.j.int32),self.j.array([1,1,1],self.j.int32))
        self.assertEqual(ok.tolist(),[True,True,False]);self.assertEqual(updated.spikes.tolist(),[[1,0],[0,1],[0,0]])
        wrong,ok=self.f.apply(state,self.tables,operation=self.f.WEATHER,value=1,
            producer_binding=self.tables.binding.at[0].add(1))
        self.assertFalse(bool(ok.any()))
        for x,y in zip(state,wrong):self.assertTrue(bool(self.j.array_equal(x,y)))
        for bad in (True,.5,self.j.array(True),self.j.array(.5)):
            with self.assertRaises(ValueError):self.f.apply(state,self.tables,operation=self.f.WEATHER,
                value=bad,producer_binding=self.tables.binding)
        request=self.requests.parse_action_request({'wait':True},owner_side=0)
        actual=self.jax.jit(lambda x:self.f.project(x,self.tables,request,observer_side=0,
            expected_vocabulary=self.tables.vocabulary_binding))(updated)
        self.assertEqual(actual.admitted.tolist(),[True]*3)
        for changes in (dict(observer_side=1),dict(expected_vocabulary=self.tables.vocabulary_binding.at[0].add(1))):
            args=dict(observer_side=0,expected_vocabulary=self.tables.vocabulary_binding);args.update(changes)
            bad=self.f.project(updated,self.tables,request,**args)
            self.assertFalse(bool(bad.admitted.any()));self.assertFalse(bool(bad.numeric.any()));self.assertFalse(bool(bad.categorical.any()))

    def test_row_shapes_and_private_request_dtype_aliases_refuse(self):
        state=self.f.initialize(2,self.tables)
        defaults=dict(operation=self.f.SPIKES,side=0,value=1,upkeep=False,from_ability=False,valid=True)
        for name in defaults:
            bad=self.j.zeros((2,1),bool if name in ('upkeep','from_ability','valid') else self.j.int32)
            args=dict(defaults);args[name]=bad
            with self.subTest(name=name),self.assertRaises(ValueError):
                self.f.apply(state,self.tables,producer_binding=self.tables.binding,**args)
        request=self.requests.parse_action_request({'wait':True},owner_side=0)
        for name in ('owner_side','kind'):
            for value in (True,False,0.,self.j.array(True),self.j.array(0.)):
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                    self.f.project(state,self.tables,request._replace(**{name:value}),observer_side=0,
                        expected_vocabulary=self.tables.vocabulary_binding)
            with self.assertRaises(ValueError):
                self.f.project(state,self.tables,request._replace(**{name:self.j.zeros((2,1),self.j.int32)}),
                    observer_side=0,expected_vocabulary=self.tables.vocabulary_binding)
        for args in (dict(observer_side=self.j.zeros((2,1),self.j.int32)),
                     dict(exact_state=self.j.ones((2,1),bool))):
            kw=dict(observer_side=0,expected_vocabulary=self.tables.vocabulary_binding);kw.update(args)
            with self.assertRaises(ValueError):self.f.project(state,self.tables,request,**kw)

    def test_lexical_refusals_and_unimplemented_consumer_remain_explicit(self):
        for line in ('|-weather|UnknownWeather','|request|{}','|turn|01','|turn|2147483648',
                     '|-sidestart|p3: Missing|Spikes','|move|p3: Missing|Wish|p3: Missing'):
            with self.subTest(line=line),self.assertRaises(ValueError):self.f.project_line(line,catalog=self.catalog)
        for line in ('|-sidestart|p1: Alice|Reflect','|-weatherx|RainDance','|-damage|p1a: Snorlax|80/100'):
            self.assertEqual(self.f.project_line(line,catalog=self.catalog).facts,())
        state=self.f.initialize(1,self.tables)
        for op,side,value in ((self.f.TURN,-1,2),(self.f.SPIKES,0,4),(self.f.WISH,-1,1),(self.f.WEATHER,-1,999)):
            after,ok=self.f.apply(state,self.tables,operation=op,side=side,value=value,producer_binding=self.tables.binding)
            self.assertFalse(bool(ok.any()))
            for x,y in zip(state,after):self.assertTrue(bool(self.j.array_equal(x,y)))


if __name__=='__main__':unittest.main()
