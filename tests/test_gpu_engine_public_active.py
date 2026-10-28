"""Public active inputs against the real parser and retained v4 writer."""
import importlib.util
import unittest
import warnings


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicActiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import public_active as a, public_mon_state as m
        from pokezero import showdown as s
        from pokezero.category_vocab import CategoryVocabulary
        cls.jax,cls.j,cls.a,cls.m,cls.s=jax,j,a,m,s
        cls.vocab=CategoryVocabulary(tuple('volatile:'+n for n in a.NAMES[1:]))
        cls.catalog=a.compile_active(cls.vocab);cls.tables=cls.catalog.device
        cls.sb=j.arange(8,dtype=j.uint32)+1

    def initial(self,batch=1):
        return (self.a.initialize(batch,self.tables,species_binding=self.sb),
            self.m.initialize(batch,species_binding=self.sb))

    def fact(self,state,mons,op,side=-1,value=0,*,enc=None,bp=False,valid=True):
        if enc is None: enc=0 if side<0 else int(mons.encounter[0,side,int(mons.active[0,side])])
        updated,ok=self.a.apply(state,self.tables,mons,operation=op,side=side,encounter=enc,
            value=value,baton_pass=bp,producer_binding=self.tables.binding,valid=valid)
        self.assertTrue(bool(ok.all()))
        return updated

    def consume(self,state,mons,line,pending):
        p=line.split('|');kind=p[1];seat=self.s._slot_from_ident(p[2]) if len(p)>2 else None
        side=int(seat[1])-1 if seat in ('p1','p2') else -1
        if kind in ('switch','drag','replace'):
            # Test fixture resolves each public nickname to a positional encounter.
            slot=0 if p[2].split(': ')[-1] in ('Snorlax','Absol') else 1
            bp=kind=='switch' and (side in pending or '[from] Baton Pass' in line)
            pending.discard(side)
            mons,ok=self.m.arrive(mons,side=side,slot=slot,encounter=slot+1,species=slot+1,
                source_species=0,level=80,gender=0,hp_numerator=100,hp_denominator=100,status=0,
                kind=self.m.BATON_PASS if bp else self.m.SWITCH)
            self.assertTrue(bool(ok.all()))
            state=self.fact(state,mons,self.a.ARRIVE,side,bp=bp)
        elif kind=='faint':
            state=self.fact(state,mons,self.a.FAINT,side)
        if kind=='turn':pending.clear()
        projection=self.a.project_line(line,catalog=self.catalog)
        self.assertFalse(projection.full_line_consumers_complete)
        self.assertEqual(projection.raw_line,line)
        for op,target,value in projection.facts:state=self.fact(state,mons,op,target,value)
        if kind=='move' and len(p)>3 and self.s._normalize_identifier(p[3])=='batonpass':pending.add(side)
        return state,mons

    def compare(self,state,mons,lines):
        replay=self.s.parse_showdown_replay(lines,battle_id='active-public-fixture')
        got=self.a.project(state,self.tables,mons,expected_vocabulary=self.tables.vocabulary_binding)
        self.assertEqual(got.admitted.tolist(),[True])
        for side in (0,1):
            slot=f'p{side+1}'
            bag=tuple(replay.volatiles.get(slot,()))
            self.assertEqual(set(self.catalog.names[i] for i in range(len(self.catalog.names)) if state.volatiles[0,side,i]),set(bag))
            self.assertEqual(int(state.stall[0,side]),replay.stall_counter[slot])
            self.assertEqual(state.elapsed[0,side].tolist(),[getattr(replay,n)[slot] for n in (
                'confusion_elapsed','encore_elapsed','wrap_trap_elapsed')])
            self.assertEqual(bool(state.recharge[0,side]),replay.must_recharge[slot])
            self.assertEqual(bool(state.meanlook[0,side]),replay.meanlook_trap[slot])
            cats=[[None]*51];nums=[[0.]*self.s.V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT]
            candidate=self.s.ShowdownPokemon(ident=slot+'a: fixture',showdown_slot=slot,species='Snorlax',
                condition='100/100',active=True,details='Snorlax, L80')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',self.s.VolatileBucketOverflowWarning)
                self.s._encode_pokemon_tokens(cats,nums,0,[candidate],role='opponent',limit=1,
                    active_volatiles=bag,active_stall_counter=replay.stall_counter[slot],
                    active_confusion_elapsed=replay.confusion_elapsed[slot],active_encore_elapsed=replay.encore_elapsed[slot],
                    active_wrap_trap_elapsed=replay.wrap_trap_elapsed[slot],active_meanlook_trap=replay.meanlook_trap[slot],
                    active_must_recharge=replay.must_recharge[slot],schema_v3=True,schema_v4=True)
            for name,col in zip(self.a.CORE,self.a.NUMERIC_COLUMNS):
                self.assertAlmostEqual(float(got.numeric[0,side,col]),nums[0][getattr(self.s,'NUMERIC_'+name)],places=6,msg=name)
            for col in self.a.CATEGORY_COLUMNS:self.assertEqual(int(got.categorical[0,side,col]),self.vocab.encode(cats[0][col]))
            visible=len(set(bag)|({'mustrecharge'} if replay.must_recharge[slot] else set()))
            self.assertEqual(bool(got.overflow[0,side]),visible>self.s.VOLATILE_BUCKET_COUNT)
        self.assertEqual(int(got.numeric_columns.sum()),5)
        self.assertEqual(int(got.categorical_columns.sum()),6)

    def prefixes(self,events):
        state,mons=self.initial();lines=[];pending=set()
        for line in ['|switch|p1a: Snorlax|Snorlax, L80|100/100',
                     '|switch|p2a: Absol|Absol, L80|100/100']+events:
            state,mons=self.consume(state,mons,line,pending);lines.append(line)
            # Both public seats have been introduced before writer comparison.
            if len(lines)>=2:self.compare(state,mons,lines)
        return state,mons,lines

    def test_all_tracked_names_perish_exclusion_and_sorted_six_bucket_overflow(self):
        events=[]
        for name in sorted(self.s.TRACKED_VOLATILES):events.append('|-start|p1a: Snorlax|'+name)
        events+=['|-start|p1a: Snorlax|perish3','|-singlemove|p1a: Snorlax|perish2',
            '|-singlemove|p1a: Snorlax|perish1','|-start|p1a: Snorlax|perish2']
        events+=['|-mustrecharge|p1a: Snorlax','|cant|p1a: Snorlax|recharge']
        for name in sorted(self.s.TRACKED_VOLATILES):events.append('|-end|p1a: Snorlax|'+name)
        self.prefixes(events)

    def test_single_move_charge_expiry_abort_and_sunny_animation_match_parser(self):
        self.prefixes(['|-singlemove|p1a: Snorlax|Destiny Bond','|-singlemove|p1a: Snorlax|Grudge',
            '|-activate|p1a: Snorlax|confusion','|-singlemove|p1a: Snorlax|Destiny Bond',
            '|move|p1a: Snorlax|Solar Beam||[still]','|-prepare|p1a: Snorlax|Solar Beam',
            '|-anim|p1a: Snorlax|Solar Beam|p2a: Absol','|-prepare|p1a: Snorlax|Solar Beam',
            '|cant|p1a: Snorlax|par','|-singlemove|p1a: Snorlax|Grudge',
            '|move|p1a: Snorlax|Tackle|p2a: Absol'])

    def test_duration_resets_baton_pass_and_source_meanlook_quirk(self):
        self.prefixes(['|turn|1','|-start|p1a: Snorlax|confusion','|-start|p1a: Snorlax|Encore',
            '|-activate|p1a: Snorlax|move: Wrap|[of] p2a: Absol','|-activate|p1a: Snorlax|trapped',
            '|turn|2','|-start|p1a: Snorlax|confusion','|-mustrecharge|p1a: Snorlax',
            '|move|p1a: Snorlax|Baton Pass|p1a: Snorlax','|switch|p1a: Blissey|Blissey, L80|100/100',
            '|turn|3','|-end|p1a: Blissey|partiallytrapped','|turn|4',
            '|-activate|p1a: Blissey|move: Wrap','|turn|5','|-end|p1a: Blissey|Wrap|[partiallytrapped]',
            '|-end|p1a: Blissey|confusion','|-start|p2a: Absol|Encore','|turn|6','|faint|p2a: Absol',
            '|switch|p2a: Snorlax|Snorlax, L80|100/100','|drag|p1a: Snorlax|Snorlax, L80|100/100'])

    def test_stall_success_failure_called_moves_cant_saturation_and_unrelated_tags(self):
        events=['|turn|1']
        for move,tag in [('Protect','Protect'),('Detect','Protect'),('Endure','move: Endure')]*4:
            events+=['|move|p2a: Absol|'+move+'|p2a: Absol','|-singleturn|p2a: Absol|'+tag]
        events+=['|-singleturn|p2a: Absol|Magic Coat','|move|p2a: Absol|Protect|p2a: Absol',
            '|-fail|p2a: Absol','|-singleturn|p2a: Absol|Protect','|cant|p2a: Absol|slp',
            '|move|p2a: Absol|Protect|p2a: Absol','|-singleturn|p2a: Absol|Protect',
            '|move|p2a: Absol|Tackle|p1a: Snorlax|[from] move: Sleep Talk','|-fail|p2a: Absol']
        self.prefixes(events)

    def test_batched_jit_rows_namespaces_shapes_and_payloads_refuse_atomically(self):
        state,mons,_=self.prefixes(['|turn|1'])
        state=self.jax.tree.map(lambda x:self.j.repeat(x,3,axis=0),state)
        mons=self.jax.tree.map(lambda x:self.j.repeat(x,3,axis=0),mons)
        fn=self.jax.jit(lambda x,side,value:self.a.apply(x,self.tables,mons,operation=self.a.ADD,
            side=side,encounter=1,value=value,producer_binding=self.tables.binding))
        out,ok=fn(state,self.j.array([0,1,2],self.j.int32),self.j.array([self.catalog.ids['confusion']]*3,self.j.int32))
        self.assertEqual(ok.tolist(),[True,True,False]);self.assertTrue(bool(out.volatiles[0,0,self.catalog.ids['confusion']]))
        self.assertTrue(bool(out.volatiles[1,1,self.catalog.ids['confusion']]))
        cases=[dict(operation=self.a.ADD,side=0,encounter=1,value=0),
            dict(operation=self.a.ADD,side=0,encounter=1,value=self.a.RECHARGE_ID),
            dict(operation=self.a.ADD,side=0,encounter=2,value=1),
            dict(operation=self.a.TRAPPED,side=0,encounter=1,value=1),
            dict(operation=self.a.TURN,value=3),dict(operation=self.a.TURN,value=2,baton_pass=True)]
        for kwargs in cases:
            before,ok=self.a.apply(state,self.tables,mons,producer_binding=self.tables.binding,**kwargs)
            self.assertFalse(bool(ok.any()))
            for x,y in zip(before,state):self.assertTrue(bool(self.j.array_equal(x,y)))
        for bad in (True,.5,self.j.array([1.]),self.j.ones((3,1),self.j.int32)):
            with self.assertRaises(ValueError):self.a.apply(state,self.tables,mons,operation=bad,
                producer_binding=self.tables.binding)
        bad=state._replace(species_binding=state.species_binding.at[:,0].add(1))
        got=self.a.project(bad,self.tables,mons,expected_vocabulary=self.tables.vocabulary_binding)
        self.assertFalse(bool(got.admitted.any()));self.assertFalse(bool(got.numeric.any()))
        for bad in (state._replace(elapsed=state.elapsed.astype(self.j.float32)),
                    state._replace(turn=state.turn[:,None])):
            with self.assertRaises(ValueError):self.a.project(bad,self.tables,mons,expected_vocabulary=self.tables.vocabulary_binding)

    def test_counter_overflow_and_lifecycle_mismatch_refuse_with_no_cross_row_change(self):
        state,mons,_=self.prefixes(['|turn|1','|-start|p1a: Snorlax|confusion'])
        state=state._replace(elapsed=state.elapsed.at[0,0,0].set(2**31-1))
        before,ok=self.a.apply(state,self.tables,mons,operation=self.a.TURN,value=2,producer_binding=self.tables.binding)
        self.assertFalse(bool(ok.any()))
        for x,y in zip(before,state):self.assertTrue(bool(self.j.array_equal(x,y)))
        forged=mons._replace(encounter=mons.encounter.at[0,0,0].set(2))
        got=self.a.project(state,self.tables,forged,expected_vocabulary=self.tables.vocabulary_binding)
        self.assertFalse(bool(got.admitted.any()))
        with self.assertRaises(ValueError):self.a.project_line('|request|{}',catalog=self.catalog)
        with self.assertRaises(ValueError):self.a.project_line('|turn|1.0',catalog=self.catalog)
        self.assertEqual(self.a.project_line('|faint|p1a: Snorlax',catalog=self.catalog).facts,())
        self.assertEqual(self.a.project_line('|switch|p1a: Snorlax|Snorlax|100/100',catalog=self.catalog).facts,())


if __name__=='__main__':unittest.main()
