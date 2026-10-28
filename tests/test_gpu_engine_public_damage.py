"""Public damage input bookkeeping against the real parser and v4 writers."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicDamageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import public_damage as d,public_mon_state as m
        from pokezero import showdown as s
        from pokezero.category_vocab import CategoryVocabulary
        cls.jax,cls.j,cls.d,cls.m,cls.s=jax,j,d,m,s
        cls.tables=d.compile_damage(CategoryVocabulary(()));cls.sb=j.arange(8,dtype=j.uint32)+1

    def initial(self,batch=1):
        return self.d.initialize(batch,self.tables,species_binding=self.sb),self.m.initialize(batch,species_binding=self.sb)

    def apply(self,state,mons,op,*,side=-1,slot=-1,enc=0,**kwargs):
        state,ok=self.d.apply(state,self.tables,mons,operation=op,side=side,slot=slot,encounter=enc,
            producer_binding=self.tables.binding,**kwargs)
        self.assertTrue(bool(ok.all()));return state

    def consume(self,state,mons,line):
        from gpu_engine.public_mon_protocol import condition_words
        p=line.split('|');kind=p[1];side=self.s._slot_from_ident(p[2]) if len(p)>2 else None
        side=int(side[1])-1 if side in ('p1','p2') else -1
        if kind in ('switch','drag','replace'):
            slot=0 if p[2].split(': ')[-1] in ('Snorlax','Absol') else 1
            hp,maximum,status,fainted=condition_words(p[4])
            mons,ok=self.m.arrive(mons,side=side,slot=slot,encounter=slot+1,species=slot+1,source_species=0,
                level=80,gender=0,hp_numerator=hp,hp_denominator=maximum,status=status)
            self.assertTrue(bool(ok.all()));state=self.apply(state,mons,self.d.ARRIVE,side=side,slot=slot,enc=slot+1)
        elif kind=='turn':state=self.apply(state,mons,self.d.TURN,value=int(p[2]))
        elif kind=='move':
            slot=int(mons.active[0,side]);enc=int(mons.encounter[0,side,slot])
            state=self.apply(state,mons,self.d.MOVE,side=side,slot=slot,enc=enc)
        for op,target in self.d.control_facts(line):state=self.apply(state,mons,op,side=target)
        if kind in ('-damage','-heal','-sethp'):
            slot=int(mons.active[0,side]);enc=int(mons.encounter[0,side,slot])
            hp,maximum,status,fainted=condition_words(p[3])
            damage,tagged,spikes=self.d.condition_tags(line)
            if damage:state=self.apply(state,mons,self.d.DAMAGE,side=side,slot=slot,enc=enc,new_hp=hp,tagged=bool(tagged),spikes=bool(spikes))
            mons,ok=self.m.condition_update(mons,side=side,slot=slot,encounter=enc,numerator=hp,
                denominator=maximum,status=status,fainted=bool(fainted));self.assertTrue(bool(ok.all()))
        elif kind=='faint':
            slot=int(mons.active[0,side]);enc=int(mons.encounter[0,side,slot])
            mons,ok=self.m.faint(mons,side=side,slot=slot,encounter=enc);self.assertTrue(bool(ok.all()))
        return state,mons

    def compare(self,state,mons,lines):
        replay=self.s.parse_showdown_replay(lines,battle_id='damage-public-fixture')
        for field in self.d.SUM_FIELDS:
            actual=getattr(state,field).sum(axis=-1)
            refname={'current_dealt':'current_damage_dealt','current_taken':'current_damage_taken',
                'last_dealt':'last_damage_dealt','last_taken':'last_damage_taken','hazard_suffered':'hazard_damage_suffered'}[field]
            for side in (0,1):self.assertAlmostEqual(float(actual[0,side]),getattr(replay,refname)[f'p{side+1}'],places=6,msg=field)
        self.assertEqual(state.items_removed.tolist(),[[replay.items_removed['p1'],replay.items_removed['p2']]])
        active=self.d.project_active(state,self.tables,mons,expected_vocabulary=self.tables.vocabulary_binding)
        self.assertEqual(active.admitted.tolist(),[True]);self.assertEqual(int(active.numeric_columns.sum()),2)
        for side in (0,1):
            nums=[0.]*self.s.V4_PRIVATE_WRITER_NUMERIC_FEATURE_COUNT;cats=[None]*51
            self.s._encode_active_feature_pack(cats,nums,truant_loaf=False,last_used_move=None,arrived_by_baton_pass=False,
                choice_locked=False,item_swapped=False,traced_ability=None,last_damage_dealt=replay.last_damage_dealt[f'p{side+1}'],
                last_damage_taken=replay.last_damage_taken[f'p{side+1}'])
            for name,col in zip(self.d.MON_CORE,self.d.MON_COLUMNS):
                self.assertAlmostEqual(float(active.numeric[0,side,col]),nums[getattr(self.s,'NUMERIC_'+name)],places=6)
        for observer in (0,1):
            source=self.s.normalize_for_player(replay,player_id='observer',configured_showdown_slot=f'p{observer+1}')
            values=self.s.field_credit_values(source,dex=None)
            patch=self.d.project_credits(state,self.tables,mons,observer_side=observer,expected_vocabulary=self.tables.vocabulary_binding)
            self.assertEqual(patch.admitted.tolist(),[True]);self.assertEqual(int(patch.numeric_columns.sum()),4)
            for col,key in zip(self.d.FIELD_COLUMNS,('self_hazard_credit','opponent_hazard_credit','self_items_removed_credit','opponent_items_removed_credit')):
                self.assertAlmostEqual(float(patch.numeric[0,col]),values[key],places=6)

    def prefixes(self,events):
        state,mons=self.initial();lines=[]
        for line in ['|switch|p1a: Snorlax|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100']+events:
            state,mons=self.consume(state,mons,line);lines.append(line)
            if len(lines)>=2:self.compare(state,mons,lines)
        return state,mons,lines

    def test_old_hp_multi_hit_heal_and_changed_denominator_match_source(self):
        self.prefixes(['|turn|1','|move|p1a: Snorlax|Body Slam|p2a: Absol','|-damage|p2a: Absol|80/100',
            '|-damage|p2a: Absol|75/100','|-heal|p2a: Absol|95/100','|-damage|p2a: Absol|20/50',
            '|-damage|p1a: Snorlax|50/100|[from] recoil','|turn|2','|','|move|p2a: Absol|Tackle|p1a: Snorlax',
            '|-damage|p1a: Snorlax|30/100','|-damage|p1a: Snorlax|0 fnt','|faint|p1a: Snorlax','|turn|3'])

    def test_confusion_cant_and_arrival_close_but_blank_and_faint_do_not(self):
        self.prefixes(['|turn|1','|move|p1a: Snorlax|Tackle|p2a: Absol','|-activate|p2a: Absol|confusion',
            '|-damage|p2a: Absol|90/100','|move|p1a: Snorlax|Tackle|p2a: Absol','|cant|p2a: Absol|par',
            '|-damage|p2a: Absol|80/100','|move|p1a: Snorlax|Tackle|p2a: Absol','|',
            '|-damage|p2a: Absol|70/100','|faint|p1a: Snorlax','|-damage|p2a: Absol|60/100',
            '|switch|p1a: Blissey|Blissey, L80|100/100','|-damage|p2a: Absol|50/100','|turn|2'])

    def test_literal_hazard_and_knockoff_credit_side_orientation_survives_switches(self):
        self.prefixes(['|turn|1','|-damage|p1a: Snorlax|87/100|[from] Spikes',
            '|-damage|p2a: Absol|88/100| [from] Spikes','|-damage|p2a: Absol|77/100|[from] spikes',
            '|-damage|p2a: Absol|66/100|other [from] Spikes','|-enditem|p1a: Snorlax|Leftovers|[from] move: Knock Off',
            '|-enditem|p2a: Absol|Choice Band|[from] move: Trick','|-enditem|p2a: Absol|Leftovers|[from] move: Knock Off',
            '|switch|p1a: Blissey|Blissey, L80|100/100','|-damage|p1a: Blissey|75/100|[from] Spikes','|turn|2'])

    def test_compensated_jit_long_prefix_agrees_with_source_without_float64(self):
        state,mons,_=self.prefixes(['|turn|1'])
        maximum=2**31-1
        mons=mons._replace(hp_numerator=mons.hp_numerator.at[0,1,0].set(maximum),hp_denominator=mons.hp_denominator.at[0,1,0].set(maximum))
        def step(st,_):
            out,ok=self.d.apply(st,self.tables,mons,operation=self.d.DAMAGE,side=1,slot=0,encounter=1,
                new_hp=maximum-1,tagged=True,spikes=True,producer_binding=self.tables.binding)
            return out,ok
        out,accepted=self.jax.jit(lambda x:self.jax.lax.scan(step,x,None,length=10000))(state)
        self.assertTrue(bool(accepted.all()))
        lines=['|switch|p2a: Absol|Absol, L80|'+f'{maximum}/{maximum}', '|turn|1']
        lines+=([f'|-damage|p2a: Absol|{maximum-1}/{maximum}|[from] Spikes',f'|-heal|p2a: Absol|{maximum}/{maximum}']*10000)
        ref=self.s.parse_showdown_replay(lines,battle_id='damage-numeric-control')
        self.assertAlmostEqual(float(out.hazard_suffered[0,1].sum()),ref.hazard_damage_suffered['p2'],places=10)
        self.assertAlmostEqual(float(out.current_taken[0,1].sum()),ref.current_damage_taken['p2'],places=10)

    def test_jit_type_namespace_overflow_and_invalid_rows_preserve_all_fields(self):
        state,mons,_=self.prefixes(['|turn|1'])
        state=self.jax.tree.map(lambda x:self.j.repeat(x,2,axis=0),state);mons=self.jax.tree.map(lambda x:self.j.repeat(x,2,axis=0),mons)
        out,ok=self.jax.jit(lambda st:self.d.apply(st,self.tables,mons,operation=self.d.DAMAGE,side=1,
            slot=self.j.array([1,0],self.j.int32),encounter=1,new_hp=50,producer_binding=self.tables.binding))(state)
        self.assertEqual(ok.tolist(),[False,True])
        for x,y in zip(out,state):self.assertTrue(bool(self.j.array_equal(x[0],y[0])))
        self.assertAlmostEqual(float(out.current_taken[1,1].sum()),.5)
        cases=[dict(operation=self.d.DAMAGE,side=1,slot=0,encounter=1,new_hp=50,spikes=True),
            dict(operation=self.d.TURN,value=3),dict(operation=self.d.REMOVED,side=1,slot=0),
            dict(operation=self.d.MOVE,side=1,slot=0,encounter=2),dict(operation=self.d.CLOSE,new_hp=1)]
        for kwargs in cases:
            out,ok=self.d.apply(state,self.tables,mons,producer_binding=self.tables.binding,**kwargs)
            self.assertFalse(bool(ok.any()))
            for x,y in zip(out,state):self.assertTrue(bool(self.j.array_equal(x,y)))
        broken=state._replace(items_removed=state.items_removed.at[:,1].set(2**31-1))
        out,ok=self.d.apply(broken,self.tables,mons,operation=self.d.REMOVED,side=1,producer_binding=self.tables.binding)
        self.assertFalse(bool(ok.any()));self.assertTrue(bool(self.j.array_equal(out.items_removed,broken.items_removed)))
        for broken in (state._replace(species_binding=state.species_binding.at[:,0].add(1)),
            state._replace(last_taken=state.last_taken.at[:,0,0].set(float('nan')))):
            patch=self.d.project_active(broken,self.tables,mons,expected_vocabulary=self.tables.vocabulary_binding)
            self.assertFalse(bool(patch.admitted.any()));self.assertFalse(bool(patch.numeric.any()))
        for bad in (True,.5,self.j.ones((2,1),self.j.int32)):
            with self.assertRaises(ValueError):self.d.apply(state,self.tables,mons,operation=bad,producer_binding=self.tables.binding)
        with self.assertRaises(ValueError):self.d.project_active(state._replace(turn=state.turn[:,None]),self.tables,mons,expected_vocabulary=self.tables.vocabulary_binding)


if __name__=='__main__':unittest.main()
