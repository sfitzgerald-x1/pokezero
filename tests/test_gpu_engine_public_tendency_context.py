"""Tendency raw controls compose with the ordered public input transaction."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicTendencyContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_active_context as fixtures
        fixtures.PublicActiveContextTests.setUpClass.__func__(cls)
        from gpu_engine import public_tendency as t
        from pokezero import transitions
        from pokezero.dex import ShowdownDex
        cls.t,cls.ref=t,transitions
        moves=dict(cls.dex.moves);moves['pursuit']=moves['bodyslam']
        cls.dex=ShowdownDex(moves,cls.dex.species,cls.dex.type_chart)
        cls.catalog=cls.p.compile_moves(cls.host,cls.dex,cls.vocab)
        cls.tendency_catalog=t.compile_tendency(cls.species_catalog,cls.catalog,cls.vocab)
        cls.tables=cls.tendency_catalog.device
        cls.contexts={n:cls.c.compile_context(cls.host,cls.abilities,cls.catalog,items=cls.item_catalog,
            item_belief_narrowing=n,field_catalog=cls.field_catalog,active_catalog=cls.active_catalog,
            tendency_tables=cls.tables) for n in (False,True)}
        cls.kwargs=dict(cls.kwargs,moves=cls.catalog,tendency_catalog=cls.tendency_catalog)

    project=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.project
    fold=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.fold
    carry=staticmethod(__import__('test_gpu_engine_public_item_context').PublicItemContextTests.carry)
    run_lines=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.run_lines
    same=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.same

    def initial(self,narrow=False,batch=1):
        beliefs=self.b.initialize(self.j.zeros((batch,2,6),self.j.int32),self.host.device)
        binding=self.species_catalog.device.binding
        return (self.i.initialize(self.j.arange(1,batch+1,dtype=self.j.int32),source_binding=self.host.device.binding),
            beliefs,self.m.initialize(batch,species_binding=binding),self.c.initialize(beliefs,self.contexts[narrow],
                species_binding=binding,tendency_complete_prefix=True))

    def chunks(self,projections,events=None):
        import numpy as np
        p=projections[0]
        return list(self.i.pack_chunks(np.array([1],np.int32),[events if events is not None else [e for p in projections for e in p.events]],
            source_binding=self.host.device.binding,**{n:np.asarray(getattr(p,n),np.uint32) for n in (
                'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding',
                'active_binding','tendency_binding')}))

    def compare(self,result,lines):
        from test_gpu_engine_public_tendency import PublicTendencyTests
        check=PublicTendencyTests()
        for n in ('t','tables','ref','reference','j','vocab','species_catalog'):setattr(check,n,getattr(self,n))
        check.compare(result.context.tendency,lines)
        self.assertFalse(bool(result.context.tendency_raw_pending.any()))
        self.assertEqual(int(result.context.tendency.turn[0]),int(result.cursor.turn[0]))

    def test_actual_source_windows_opportunities_switches_and_weather_both_modes(self):
        lines=[*self.prefix,'|turn|1','|move|p1a: Snorlax|Surf|p2a: Absol',
            '|-activate|p2a: Absol|Protect','|-immune|p2a: Absol','|',
            '|cant|p2a: Absol|slp','|move|p2a: Absol|Sleep Talk|p1a: Snorlax',
            '|move|p2a: Absol|Body Slam|p1a: Snorlax|[from] move: Sleep Talk','|upkeep','|turn|2',
            '|switch|p2a: Gardevoir|Gardevoir, L80|100/100',
            '|-weather|RainDance|[from] ability: Sand Stream|[of] p2a: Gardevoir',
            '|-activate|p1a: Snorlax|move: Pursuit','|','|move|p2a: Gardevoir|Pursuit|p1a: Snorlax',
            '|upkeep','|turn|3','|move|p1a: Snorlax|Solar Beam|p2a: Gardevoir',
            '|-prepare|p1a: Snorlax|Solar Beam','|upkeep','|turn|4',
            '|move|p1a: Snorlax|Solar Beam|p2a: Gardevoir']
        for narrow in (False,True):
            result,_=self.run_lines(lines,narrow);self.compare(result,lines)

    def test_turn_control_crosses_transport_boundary_after_shared_cursor_clock(self):
        lines=[*self.prefix,*(['|']*40),'|turn|1']
        _,p=self.project(lines);chunks=self.chunks(p);self.assertEqual(len(chunks),3)
        self.assertEqual(int(chunks[1].words[0,63,0]),self.i.Kind.TURN)
        current=self.initial()
        for chunk in chunks[:2]:result=self.fold(current,chunk);self.assertEqual(result.error.tolist(),[0]);current=self.carry(result)
        self.assertEqual(int(result.cursor.turn[0]),1);self.assertEqual(int(result.context.tendency.turn[0]),0)
        self.assertTrue(bool(result.context.tendency_raw_pending[0]))
        final=self.fold(current,chunks[2]);self.assertEqual(final.error.tolist(),[0]);self.compare(final,lines)

    def test_missing_duplicate_or_malformed_controls_roll_back_all_row_ledgers(self):
        from dataclasses import replace
        lines=[*self.prefix,'|turn|1','|move|p1a: Snorlax|Body Slam|p2a: Absol']
        _,p=self.project(lines);events=[e for row in p for e in row.events]
        initial=self.initial();idx=next(i for i,e in enumerate(events) if e.kind==self.i.Kind.TENDENCY)
        for changed in (events[:idx]+events[idx+1:],events[:idx]+[events[idx]]+events[idx:],
                [e for e in events if e.kind not in (self.i.Kind.RAW_BEGIN,self.i.Kind.TENDENCY)]):
            changed=[replace(e,sequence=i+1) for i,e in enumerate(changed)]
            result=self.fold(initial,self.chunks(p,changed)[0]);self.assertNotEqual(int(result.error[0]),0);self.same(initial,self.carry(result))
        chunk=self.chunks(p)[0]
        bad=chunk._replace(words=chunk.words.at[0,idx,13].set(32))
        result=self.fold(initial,bad);self.assertNotEqual(int(result.error[0]),0);self.same(initial,self.carry(result))
        turn_idx=next(i for i,e in enumerate(events) if e.kind==self.i.Kind.TENDENCY and e.payload[0]==self.t.TURN)
        # A later correct control cannot conceal corruption at an earlier close.
        arrival_bad=chunk._replace(words=chunk.words.at[0,idx,9].set(self.species_catalog.species_id('Absol')))
        clock_bad=chunk._replace(words=chunk.words.at[0,turn_idx,8:].set(self.j.array([self.t.OTHER,0,0,-1,0,0,0,9],self.j.int32)))
        for corrupt in (arrival_bad,clock_bad):
            result=self.fold(initial,corrupt);self.assertNotEqual(int(result.error[0]),0);self.same(initial,self.carry(result))
        mixed=chunk._replace(stream_id=self.j.array([1,2],self.j.int32),
            words=self.j.concatenate([bad.words,chunk.words]),valid=self.j.concatenate([chunk.valid,chunk.valid]),more=self.j.zeros(2,bool))
        batch=self.initial(batch=2);result=self.jax.jit(lambda a,b,m,c,ch:self.fold((a,b,m,c),ch))(*batch,mixed)
        self.assertNotEqual(int(result.error[0]),0);self.assertEqual(int(result.error[1]),0)
        for old,new in zip(self.jax.tree.leaves(batch),self.jax.tree.leaves(self.carry(result))):self.assertTrue(bool(self.j.array_equal(old[0],new[0])))

    def test_namespaces_producer_drop_and_unknown_raw_noops_refuse(self):
        wire,_=self.project(self.prefix)
        with self.assertRaises(ValueError):self.project(['|'],state=wire,tendency_catalog=None)
        for line in ('|unimplemented|p1a: Snorlax',
                '|unimplemented|p1a: Snorlax|x|[from] ability: Pressure|[of] p2a: Absol',
                '|unimplemented|p1a: Snorlax|x|[from] item: Leftovers',
                '|switch|p2a: absol|absol, L80|100/100','|turn|01'):
            with self.assertRaises(ValueError):self.project([line],state=wire)
        _,p=self.project([*self.prefix,'|turn|1']);chunk=self.chunks(p)[0];initial=self.initial()
        for bad in (chunk._replace(tendency_binding=None),chunk._replace(tendency_binding=chunk.tendency_binding.at[0].add(1))):
            result=self.fold(initial,bad);self.assertEqual(result.error.tolist(),[self.i.Error.SOURCE]);self.same(initial,self.carry(result))

    def test_writer_uses_owner_request_active_and_public_token_order(self):
        from gpu_engine.public_sleep import parse_own_status
        lines=[*self.prefix,'|turn|1','|switch|p2a: Gardevoir|Gardevoir, L80|100/100',
            '|switch|p2a: Absol|Absol, L80|100/100','|move|p2a: Absol|Body Slam|p1a: Snorlax',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100']
        result,_=self.run_lines(lines)
        request={'side':{'id':'p1','pokemon':[
            {'ident':'p1: Gardevoir','details':'Gardevoir, L80','condition':'100/100','active':False},
            {'ident':'p1: Snorlax','details':'Snorlax, L80','condition':'100/100','active':True}]}}
        own=parse_own_status(request,owner_side=0,species_catalog=self.species_catalog,vocabulary=self.vocab)
        def write(packet,observer=0):
            return self.t.project_from_request(result.context.tendency,self.tables,result.mon_state,
                observer_side=observer,own_status=packet,expected_vocabulary=self.tables.vocabulary_binding)
        out=write(own);self.assertEqual(out.admitted.tolist(),[True])
        self.assertEqual(float(out.numeric[0,0,self.t.MON_COLUMNS[3]]),1/8)
        wrong=self.t.project(result.context.tendency,self.tables,observer_side=0,
            opponent_species=result.mon_state.species[:,1],own_active_species=self.species_catalog.species_id('Gardevoir'),
            expected_vocabulary=self.tables.vocabulary_binding)
        self.assertEqual(float(wrong.numeric[0,0,self.t.MON_COLUMNS[3]]),0)
        for packet,observer in ((own,1),(own._replace(species_binding=own.species_binding.at[0].add(1)),0),
                (own._replace(vocabulary_binding=own.vocabulary_binding.at[0].add(1)),0)):
            bad=write(packet,observer);self.assertEqual(bad.admitted.tolist(),[False])
            self.assertFalse(bool(bad.numeric.any()));self.assertFalse(bool(bad.stats_numeric.any()));self.assertFalse(bool(bad.stats_attention.any()))


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicTendencyJointContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        PublicTendencyContextTests.setUpClass.__func__(cls)
        from test_gpu_engine_public_truant import configure_truant
        from pokezero.category_vocab import CategoryVocabulary
        from gpu_engine import public_toxic,public_sleep,public_types,public_damage
        cls.vocab=CategoryVocabulary(tuple(dict.fromkeys((*cls.vocab.tokens,'stats','type:normal','type:water'))))
        configure_truant(cls)
        cls.catalog=cls.p.compile_moves(cls.host,cls.dex,cls.vocab)
        cls.item_catalog=cls.items.compile_items(cls.host,cls.vocab,item_names=('Mystery Orb',))
        cls.field_catalog=cls.f.compile_field(cls.vocab)
        cls.damage_tables=public_damage.compile_damage(cls.vocab)
        cls.type_catalog=public_types.compile_types(cls.species_catalog,cls.dex,cls.vocab)
        cls.sleep_tables=public_sleep.compile_sleep(cls.species_catalog,cls.vocab)
        cls.toxic_tables=public_toxic.compile_toxic(cls.species_catalog,cls.vocab)
        cls.tendency_catalog=cls.t.compile_tendency(cls.species_catalog,cls.catalog,cls.vocab);cls.tables=cls.tendency_catalog.device
        cls.contexts={n:cls.c.compile_context(cls.host,cls.abilities,cls.catalog,items=cls.item_catalog,
            item_belief_narrowing=n,field_catalog=cls.field_catalog,active_catalog=cls.active_catalog,
            damage_tables=cls.damage_tables,type_catalog=cls.type_catalog,sleep_tables=cls.sleep_tables,
            toxic_tables=cls.toxic_tables,tendency_tables=cls.tables) for n in (False,True)}
        cls.kwargs=dict(source=cls.host,species_catalog=cls.species_catalog,moves=cls.catalog,abilities=cls.abilities,
            items=cls.item_catalog,field_catalog=cls.field_catalog,active_catalog=cls.active_catalog,
            damage_tables=cls.damage_tables,type_catalog=cls.type_catalog,sleep_tables=cls.sleep_tables,
            toxic_tables=cls.toxic_tables,tendency_catalog=cls.tendency_catalog)
        cls.prefix=('|switch|p1a: Slaking|Slaking, L80|100/100','|switch|p2a: Gardevoir|Gardevoir, L80|100/100')

    project=PublicTendencyContextTests.project
    fold=PublicTendencyContextTests.fold
    carry=staticmethod(PublicTendencyContextTests.carry)
    run_lines=PublicTendencyContextTests.run_lines
    same=PublicTendencyContextTests.same
    compare=PublicTendencyContextTests.compare

    def initial(self,narrow=False,batch=1):
        beliefs=self.b.initialize(self.j.zeros((batch,2,6),self.j.int32),self.host.device)
        binding=self.species_catalog.device.binding
        return (self.i.initialize(self.j.arange(1,batch+1,dtype=self.j.int32),source_binding=self.host.device.binding),
            beliefs,self.m.initialize(batch,species_binding=binding),self.c.initialize(beliefs,self.contexts[narrow],
                species_binding=binding,tendency_complete_prefix=True,toxic_hp_visibility=(2,2)))

    def chunks(self,projections,events=None):
        import numpy as np
        p=projections[0]
        return list(self.i.pack_chunks(np.array([1],np.int32),[events if events is not None else [e for p in projections for e in p.events]],
            source_binding=self.host.device.binding,**{n:np.asarray(getattr(p,n),np.uint32) for n in (
                'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding',
                'active_binding','damage_binding','type_binding','sleep_binding','toxic_binding','tendency_binding')}))

    def test_all_mode_source_parity_independent_closes_continuation_and_rollback(self):
        from test_gpu_engine_public_toxic import PublicToxicTests
        from test_gpu_engine_public_damage import PublicDamageTests
        from test_gpu_engine_public_truant import PublicTruantTests
        lines=[*self.prefix,'|-ability|p1a: Slaking|Truant',
            '|-ability|p2a: Gardevoir|Truant|[from] ability: Trace|[of] p1a: Slaking','|turn|1',
            '|move|p1a: Slaking|Body Slam|p2a: Gardevoir','|-damage|p2a: Gardevoir|80/100',
            '|-start|p2a: Gardevoir|typechange|Water','|-status|p1a: Slaking|tox','|','|upkeep','|turn|2',
            '|cant|p1a: Slaking|ability: Truant','|cant|p2a: Gardevoir|ability: Truant','|upkeep','|turn|3',
            '|-status|p2a: Gardevoir|slp|[from] move: Rest','|cant|p2a: Gardevoir|slp',
            '|move|p2a: Gardevoir|Sleep Talk|p1a: Slaking',
            '|move|p2a: Gardevoir|Body Slam|p1a: Slaking|[from] move: Sleep Talk',
            '|-damage|p1a: Slaking|94/100 tox|[from] psn','|upkeep','|turn|4']
        for narrow in (False,True):
            result,p=self.run_lines(lines,narrow);self.compare(result,lines)
            self.assertFalse(bool(result.context.toxic_raw_pending.any()))
            wire,_=self.project(lines,narrow)
            check=PublicToxicTests();check.t,check.tables,check.reference,check.dex=self.t.__class__,self.toxic_tables,self.s,self.dex
            from gpu_engine import public_toxic
            check.t=public_toxic
            check.ids={entry.ident:entry.address.encounter for entries in wire.encounters for entry in entries}
            parser=self.s._ReplayParser(complete_prefix=False,hp_visibility={'p1':'percentage','p2':'percentage'})
            for line in lines:parser._feed_line(line)
            check.compare(result.context.toxic,parser)
            damage=PublicDamageTests();damage.d=__import__('gpu_engine.public_damage',fromlist=['']);damage.s=self.s;damage.tables=self.damage_tables
            damage.compare(result.context.damage,result.mon_state,lines)
            truant=PublicTruantTests();truant.s,truant.active,truant.tables=self.s,self.active,self.active_catalog.device
            truant.compare(result.context.active,result.mon_state,lines)
        # Force a chunk to end between the independently required raw closes.
        base=[*self.prefix,'|turn|1'];status='|-status|p1a: Slaking|tox'
        _,p=self.project([*base,status]);events=[e for row in p for e in row.events]
        idx=next(i for i,e in enumerate(events) if e.kind==self.i.Kind.TOXIC)
        padding=next(n for n in range(64) if (idx+3*n)%64==63)
        lines=[*base,*(['|']*padding),status];_,p=self.project(lines);chunks=self.chunks(p);current=self.initial()
        for chunk in chunks:
            before=current;result=self.fold(current,chunk);self.assertEqual(result.error.tolist(),[0]);current=self.carry(result)
            if int(chunk.words[0,63,0])==self.i.Kind.TOXIC:
                self.assertFalse(bool(result.context.toxic_raw_pending[0]));self.assertTrue(bool(result.context.tendency_raw_pending[0]))
        self.compare(result,lines);self.assertFalse(bool(result.context.toxic_raw_pending.any()))
        for header in ('toxic_binding','tendency_binding'):
            bad=self.chunks(p)[0]._replace(**{header:None});initial=self.initial();result=self.fold(initial,bad)
            self.assertEqual(result.error.tolist(),[self.i.Error.SOURCE]);self.same(initial,self.carry(result))


if __name__=='__main__':unittest.main()
