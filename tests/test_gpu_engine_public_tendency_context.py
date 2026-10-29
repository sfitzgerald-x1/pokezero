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
        for line in ('|unimplemented|p1a: Snorlax','|switch|p2a: absol|absol, L80|100/100','|turn|01'):
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


if __name__=='__main__':unittest.main()
