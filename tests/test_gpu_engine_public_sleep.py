"""Public sleep counters versus actual belief ingestion and exact-state writer."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicSleepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_team as fixtures
        fixtures.PublicTeamTests.setUpClass.__func__(cls)
        from gpu_engine import public_sleep as s
        from pokezero.belief import PublicBattleBeliefEngine
        cls.sleep=s;cls.Engine=PublicBattleBeliefEngine
        cls.sleep_tables=s.compile_sleep(cls.catalog,cls.vocab)

    def initial(self,batch=1):
        return self.sleep.initialize(batch,self.sleep_tables,species_binding=self.catalog.device.binding),self.m.initialize(batch,species_binding=self.catalog.device.binding)

    def own(self,mons,side):
        return self.sleep.OwnStatusFacts(self.j.array(side,self.j.int32),self.sleep_tables.vocabulary_binding,
            self.catalog.device.binding,mons.species[:,side],mons.encounter[:,side]>0,mons.status[:,side],
            (self.j.arange(6)[None,:]==mons.active[:,side,None])&(mons.encounter[:,side]>0))

    def fact(self,state,mons,op,side=-1,slot=-1,x=0,y=0,enc=None):
        if enc is None:enc=0 if slot<0 else int(mons.encounter[0,side,slot])
        state,ok=self.sleep.apply(state,self.sleep_tables,mons,operation=op,side=side,slot=slot,
            encounter=enc,first=x,second=y,producer_binding=self.sleep_tables.binding)
        self.assertTrue(bool(ok.all()));return state

    def consume(self,state,mons,line,book):
        from gpu_engine.public_mon_protocol import condition_words
        p=line.split('|');kind=p[1];seat=self.reference._slot_from_ident(p[2]) if len(p)>2 else None
        side=int(seat[1])-1 if seat in ('p1','p2') else -1
        active=int(mons.active[0,side]) if side>=0 else -1
        status_ids={'none':0,'slp':1,'par':2,'brn':3,'frz':4,'psn':5,'tox':6}
        if kind in ('switch','drag','replace'):
            name=p[3].split(',')[0];key=self.reference._normalize_identifier(name)
            if key not in book[side]:book[side][key]=len(book[side])
            active=book[side][key];hp,mx,status,fnt=condition_words(p[4])
            mons,ok=self.m.arrive(mons,side=side,slot=active,encounter=active+1,
                species=self.catalog.species_id(name),source_species=self.host.species_id(name),level=80,gender=0,
                hp_numerator=hp,hp_denominator=mx,status=status)
            self.assertTrue(bool(ok.all()))
            state=self.fact(state,mons,self.sleep.ARRIVE,side,active,x=int(p[4]=='0 fnt'))
        elif kind=='turn':state=self.fact(state,mons,self.sleep.TURN,x=int(p[2]))
        elif line=='|upkeep':state=self.fact(state,mons,self.sleep.UPKEEP)
        elif kind=='-status':
            status=status_ids[p[3].strip()];mons,ok=self.m.status_update(mons,side=side,slot=active,encounter=active+1,status=status)
            self.assertTrue(bool(ok.all()));state=self.fact(state,mons,self.sleep.STATUS,side,active,x=status,y=int('move: Rest' in line))
        elif kind=='-curestatus':
            bench=not self.reference._ident_has_position(p[2])
            target=book[side][self.reference._normalize_identifier(p[2].split(':')[-1])] if bench else active
            mons,ok=self.m.status_update(mons,side=side,slot=target,encounter=target+1,status=0)
            self.assertTrue(bool(ok.all()));state=self.fact(state,mons,self.sleep.CURE,side,target,x=int(bench))
        elif kind=='-cureteam':
            mons,ok=self.m.cure_team(mons,side=side);self.assertTrue(bool(ok.all()))
            state=self.fact(state,mons,self.sleep.CURE_TEAM,side)
        elif kind=='faint':
            mons,ok=self.m.faint(mons,side=side,slot=active,encounter=active+1)
            self.assertTrue(bool(ok.all()));state=self.fact(state,mons,self.sleep.FAINT,side,active)
        elif kind in ('-damage','-heal','-sethp'):
            hp,mx,status,fnt=condition_words(p[3]);mons,ok=self.m.condition_update(mons,side=side,slot=active,
                encounter=active+1,numerator=hp,denominator=mx,status=status,fainted=bool(fnt))
            condition=p[3].strip() if kind=='-sethp' else p[3]
            flags=int(condition=='0 fnt')|int(condition.split()==['0','fnt'])*2
            self.assertTrue(bool(ok.all()));state=self.fact(state,mons,self.sleep.CONDITION,side,active,x=status,y=flags)
        for op,x,y in self.sleep.lexical_facts(line):state=self.fact(state,mons,op,side,active,x=x,y=y)
        return state,mons

    def compare(self,state,mons,engine,lines,book):
        replay=self.reference.parse_showdown_replay(lines,battle_id='sleep-input-fixture')
        status_ids={'none':0,'slp':1,'par':2,'brn':3,'frz':4,'psn':5,'tox':6}
        for side in (0,1):
            entries=engine.snapshot().side(f'p{side+1}')
            for exact in entries:
                key=self.reference._normalize_identifier(exact.species);index=book[side][key]
                self.assertEqual(int(state.count[0,side,index]),exact.sleep_turns)
                self.assertEqual(bool(state.rest[0,side,index]),exact.rest_sleep)
                self.assertEqual(int(state.skipped[0,side,index]),exact.sleep_skipped_turns)
                self.assertEqual(int(state.age[0,side,index]),exact.turns_active)
                self.assertEqual(bool(state.pending[0,side,index]),exact.key in engine._sleep_cant_pending)
                self.assertEqual(int(state.status[0,side,index]),status_ids.get(exact.status or 'none',0))
                for observer in (0,1):
                    request=dict(side=dict(id=f'p{observer+1}',pokemon=[dict(ident=x.ident,details=x.details,
                        condition=x.condition,active=x.active) for x in replay.public_revealed.get(f'p{observer+1}',())]))
                    own=self.sleep.parse_own_status(request,owner_side=observer,species_catalog=self.catalog,vocabulary=self.vocab)
                    got=self.sleep.project(state,self.sleep_tables,mons,observer_side=observer,expected_vocabulary=self.sleep_tables.vocabulary_binding,own_status=own)
                    self.assertEqual(got.admitted.tolist(),[True])
                    candidate=next(x for x in replay.public_revealed[f'p{side+1}'] if self.reference._normalize_identifier(x.species)==key)
                    condition=self.reference._condition_features(candidate.condition)
                    role='self' if observer==side else 'opponent'
                    status=(exact.status if role=='opponent' and exact.status is not None else condition.status)
                    numeric=[0.]*200
                    self.reference._encode_mon_exact_state(numeric,candidate,exact,role=role,status=status,fainted=condition.fainted)
                    for name,col in zip(self.sleep.CORE,self.sleep.COLUMNS):
                        self.assertAlmostEqual(float(got.numeric[0,side,index,col]),numeric[getattr(self.reference,'NUMERIC_'+name)],places=6,msg=name)
                    self.assertFalse(bool(got.numeric_columns[self.reference.v4_numeric_index(self.reference.NUMERIC_WAKE_KNOWN)]))

    def prefixes(self,events):
        state,mons=self.initial();book=({},{});lines=[];engine=self.Engine(format_id='gen3randombattle',set_source=self.source)
        for line in ['|switch|p1a: Snorlax|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100',*events]:
            state,mons=self.consume(state,mons,line,book);lines.append(line)
            engine.ingest_event(self.reference._public_event_from_line(line));self.compare(state,mons,engine,lines,book)
        return state,mons,engine,book

    def test_natural_sleep_rest_attempts_called_moves_and_reentry_refunds(self):
        self.prefixes(['|turn|1','|-status|p1a: Snorlax|slp|[from] move: Rest',
            '|cant|p1a: Snorlax|slp','|move|p1a: Snorlax|Sleep Talk|p2a: Absol',
            '|move|p1a: Snorlax|Body Slam|p2a: Absol|[from] move: Sleep Talk','|upkeep','|turn|2',
            '|cant|p1a: Snorlax|slp','|move|p1a: Snorlax|Snore|p2a: Absol',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100','|turn|3',
            '|switch|p1a: Snorlax|Snorlax, L80|100/100 slp','|cant|p1a: Snorlax|slp','|upkeep'])

    def test_pending_survives_turn_cure_faint_and_other_moves(self):
        self.prefixes(['|turn|1','|-status|p2a: Absol|slp','|cant|p2a: Absol|slp','|turn|2',
            '|move|p2a: Absol|Body Slam|p1a: Snorlax','|-curestatus|p2a: Absol|slp',
            '|move|p2a: Absol|Sleep Talk|p1a: Snorlax','|-status|p2a: Absol|slp|[from] move: Rest',
            '|cant|p2a: Absol|slp','|faint|p2a: Absol','|turn|3','|upkeep'])

    def test_explicit_status_differs_from_condition_and_literal_zero_fnt_active_age(self):
        self.prefixes(['|turn|1','|-status|p2a: Absol|slp|[from] move: Rest','|cant|p2a: Absol|slp',
            '|-damage|p2a: Absol|80/100','|turn|2','|-damage|p2a: Absol|0 fnt',
            '|turn|3','|faint|p2a: Absol','|turn|4'])

    def test_bench_and_team_cures_retire_exit_state_and_preserve_pending(self):
        self.prefixes(['|turn|1','|-status|p1a: Snorlax|slp|[from] move: Rest','|cant|p1a: Snorlax|slp',
            '|-curestatus|p1: Snorlax|slp|[silent]','|-status|p1a: Snorlax|slp|[from] move: Rest','|cant|p1a: Snorlax|slp',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100','|-curestatus|p1: Snorlax|slp|[silent]',
            '|switch|p1a: Snorlax|Snorlax, L80|100/100','|-status|p1a: Snorlax|slp|[from] move: Rest',
            '|cant|p1a: Snorlax|slp','|-cureteam|p1a: Snorlax','|move|p1a: Snorlax|Sleep Talk|p2a: Absol','|upkeep'])

    def test_lexical_cant_substring_rest_literal_and_any_from_marker(self):
        self.prefixes(['|-status|p1a: Snorlax|slp|[from] move: rest','|cant|p1a: Snorlax|par|slp',
            '|move|p1a: Snorlax|Sleep Talk|p2a: Absol|[from]lockedmove',
            '|move|p1a: Snorlax|Snore|p2a: Absol|[from]','|upkeep'])

    def test_exact_state_mask_and_noncanonical_faint_condition(self):
        state,mons,_,_=self.prefixes(['|turn|1','|-status|p1a: Snorlax|slp|[from] move: Rest',
            '|cant|p1a: Snorlax|slp','|-damage|p1a: Snorlax|0 fnt ',
            '|-cureteam|p1a: Snorlax','|turn|2'])
        state=self.jax.tree.map(lambda x:self.j.concatenate([x,x]),state)
        mons=self.jax.tree.map(lambda x:self.j.concatenate([x,x]),mons)
        got=self.jax.jit(lambda s,m,mask:self.sleep.project(s,self.sleep_tables,m,
            observer_side=0,expected_vocabulary=self.sleep_tables.vocabulary_binding,own_status=self.own(m,0),exact_state=mask))(
                state,mons,self.j.array([True,False]))
        self.assertEqual(got.admitted.tolist(),[True,True])
        self.assertTrue(bool((got.numeric[1]==0).all()));self.assertGreater(float(got.numeric[0].sum()),0.)
        for invalid in (0,1,.5,self.j.array(1)):
            with self.assertRaises((TypeError,ValueError)):
                self.sleep.project(state,self.sleep_tables,mons,observer_side=0,
                    expected_vocabulary=self.sleep_tables.vocabulary_binding,own_status=self.own(mons,0),exact_state=invalid)

    def test_own_request_silent_cure_and_private_order_do_not_use_public_status(self):
        state,mons,_,_=self.prefixes(['|-status|p1a: Snorlax|slp|[from] move: Rest','|cant|p1a: Snorlax|slp',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100'])
        request=dict(side=dict(id='p1',pokemon=[dict(ident='p1: Gardevoir',details='Gardevoir, L80',condition='100/100'),
            dict(ident='p1: Snorlax',details='Snorlax, L80',condition='100/100')]))
        own=self.sleep.parse_own_status(request,owner_side=0,species_catalog=self.catalog,vocabulary=self.vocab)
        def project(packet,observer=0):return self.sleep.project(state,self.sleep_tables,mons,
            observer_side=observer,expected_vocabulary=self.sleep_tables.vocabulary_binding,own_status=packet)
        got=project(own);self.assertTrue(bool(got.admitted.all()))
        self.assertEqual(int(state.count[0,0,0]),1);self.assertTrue(bool(state.rest[0,0,0]))
        self.assertEqual(int(mons.status[0,0,0]),self.sleep.SLEEP)
        expected=[0.]*200
        from pokezero.belief import RevealedPokemonBelief
        self.reference._encode_mon_exact_state(expected,self.reference._self_team_from_request(request,'p1')[1],
            RevealedPokemonBelief(showdown_slot='p1',species='Snorlax',sleep_turns=1,rest_sleep=True),
            role='self',status='none',fainted=False)
        for name,col in zip(self.sleep.CORE,self.sleep.COLUMNS):
            self.assertEqual(float(got.numeric[0,0,0,col]),expected[getattr(self.reference,'NUMERIC_'+name)])
        other=project(self.own(mons,1),observer=1)
        self.assertAlmostEqual(float(other.numeric[0,0,0,self.sleep.COLUMNS[0]]),.2,places=6)
        for bad in (own._replace(owner_side=self.j.array(1,self.j.int32)),
                    own._replace(species_binding=own.species_binding.at[0].add(1)),
                    own._replace(present=own.present.at[1].set(False),species=own.species.at[1].set(0)),
                    own._replace(species=own.species.at[1].set(own.species[0]))):
            result=project(bad);self.assertEqual(result.admitted.tolist(),[False]);self.assertFalse(bool(result.numeric.any()))
        with self.assertRaises(ValueError):self.sleep.parse_own_status(request,owner_side=1,species_catalog=self.catalog,vocabulary=self.vocab)

    def test_sethp_strip_and_cure_faint_normalization_keep_literal_gate(self):
        from gpu_engine.public_sleep_protocol import facts
        for kind in ('-damage','-heal','-sethp'):
            line=f'|{kind}|p2a: Absol|0 fnt '
            self.assertEqual(facts(line)[0][2],3 if kind=='-sethp' else 2)
            self.prefixes(['|-status|p2a: Absol|slp|[from] move: Rest','|cant|p2a: Absol|slp',line,
                '|-curestatus|p2a: Absol|slp','|-status|p2a: Absol|slp|[from] move: Rest',
                '|cant|p2a: Absol|slp','|-cureteam|p2a: Absol'])

    def test_own_age_uses_request_active_flag_independently_of_public_pointer(self):
        state,mons,engine,_=self.prefixes(['|turn|1','|switch|p1a: Gardevoir|Gardevoir, L80|100/100'])
        request=dict(side=dict(id='p1',pokemon=[dict(ident='p1: Snorlax',details='Snorlax, L80',condition='100/100',active=True),
            dict(ident='p1: Gardevoir',details='Gardevoir, L80',condition='100/100',active=False)]))
        packet=self.sleep.parse_own_status(request,owner_side=0,species_catalog=self.catalog,vocabulary=self.vocab)
        got=self.sleep.project(state,self.sleep_tables,mons,observer_side=0,
            expected_vocabulary=self.sleep_tables.vocabulary_binding,own_status=packet)
        for slot,candidate in enumerate(self.reference._self_team_from_request(request,'p1')):
            exact=next(x for x in engine.snapshot().side('p1') if x.species==candidate.species)
            numeric=[0.]*200
            self.reference._encode_mon_exact_state(numeric,candidate,exact,role='self',status='none',fainted=False)
            self.assertEqual(float(got.numeric[0,0,slot,self.sleep.COLUMNS[2]]),numeric[self.reference.NUMERIC_TURNS_ACTIVE])
        self.assertAlmostEqual(float(got.numeric[0,0,0,self.sleep.COLUMNS[2]]),1/64)

    def test_jit_bad_rows_shapes_overflow_and_duplicate_species_refuse(self):
        state,mons,_,_=self.prefixes([])
        state=self.jax.tree.map(lambda x:self.j.concatenate([x,x]),state);mons=self.jax.tree.map(lambda x:self.j.concatenate([x,x]),mons)
        fn=self.jax.jit(lambda s,m,e:self.sleep.apply(s,self.sleep_tables,m,operation=self.sleep.CANT_SLEEP,
            side=0,slot=0,encounter=e,producer_binding=self.sleep_tables.binding))
        got,ok=fn(state,mons,self.j.array([99,1],self.j.int32));self.assertEqual(ok.tolist(),[False,True])
        for a,b in zip(self.jax.tree.leaves(state),self.jax.tree.leaves(got)):self.assertTrue(bool(self.j.array_equal(a[0],b[0])))
        broken=state._replace(count=state.count.at[0,0,0].set(2**31-1));got,ok=fn(broken,mons,self.j.ones(2,self.j.int32))
        self.assertEqual(ok.tolist(),[False,True]);self.assertEqual(int(got.count[0,0,0]),2**31-1)
        duplicate=mons._replace(encounter=mons.encounter.at[0,0,1].set(2),species=mons.species.at[0,0,1].set(mons.species[0,0,0]))
        got,ok=fn(state,duplicate,self.j.ones(2,self.j.int32));self.assertEqual(ok.tolist(),[False,True])
        bad_status=mons._replace(status=mons.status.at[0,0,0].set(99))
        got,ok=self.sleep.apply(state,self.sleep_tables,bad_status,operation=self.sleep.ARRIVE,
            side=0,slot=0,encounter=1,producer_binding=self.sleep_tables.binding)
        self.assertEqual(ok.tolist(),[False,True])
        for a,b in zip(self.jax.tree.leaves(state),self.jax.tree.leaves(got)):self.assertTrue(bool(self.j.array_equal(a[0],b[0])))
        with self.assertRaises(ValueError):self.sleep.project(state._replace(age=state.age.astype(self.j.float32)),self.sleep_tables,mons,
            observer_side=0,expected_vocabulary=self.sleep_tables.vocabulary_binding,own_status=self.own(mons,0))


if __name__=='__main__':unittest.main()
