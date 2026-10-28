"""Damage reads old HP atomically with the actual ordered condition stream."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicDamageContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_field_context as fixtures
        fixtures.PublicFieldContextTests.setUpClass.__func__(cls)
        from gpu_engine import public_damage as d
        cls.d=d;cls.damage_tables=d.compile_damage(cls.vocab)
        cls.contexts={n:cls.c.compile_context(cls.host,cls.abilities,cls.catalog,items=cls.item_catalog,
            item_belief_narrowing=n,field_catalog=cls.field_catalog,damage_tables=cls.damage_tables) for n in (False,True)}
        cls.kwargs=dict(cls.kwargs,damage_tables=cls.damage_tables)

    project=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.project
    initial=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.initial
    fold=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.fold
    carry=staticmethod(__import__('test_gpu_engine_public_item_context').PublicItemContextTests.carry)
    run_lines=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.run_lines
    same=__import__('test_gpu_engine_public_item_context').PublicItemContextTests.same

    def chunks(self,projections):
        import numpy as np
        p=projections[0]
        return list(self.i.pack_chunks(np.array([1],np.int32),[[e for p in projections for e in p.events]],
            source_binding=self.host.device.binding,**{n:np.asarray(getattr(p,n),np.uint32) for n in (
                'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding','damage_binding')}))

    def compare(self,result,lines):
        from test_gpu_engine_public_damage import PublicDamageTests
        check=PublicDamageTests();check.d,check.s,check.tables=self.d,self.s,self.damage_tables
        check.compare(result.context.damage,result.mon_state,lines)
        self.assertEqual(int(result.context.damage.turn[0]),int(result.cursor.turn[0]))

    def test_pre_hp_attribution_controls_and_credits_match_actual_parser(self):
        lines=[*self.prefix,'|turn|1','|move|p1a: Snorlax|Body Slam|p2a: Absol','|-damage|p2a: Absol|80/100',
            '|-heal|p2a: Absol|90/100','|-damage|p2a: Absol|75/100|[from] Spikes','|',
            '|cant|p2a: Absol|par','|-damage|p2a: Absol|70/100',
            '|-enditem|p2a: Absol|Leftovers|[from] move: Knock Off','|turn|2',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100','|move|p2a: Absol|Body Slam|p1a: Gardevoir',
            '|-damage|p1a: Gardevoir|60/100','|-activate|p1a: Gardevoir|confusion',
            '|-damage|p1a: Gardevoir|40/100','|turn|3']
        result,projections=self.run_lines(lines);self.compare(result,lines)
        self.assertAlmostEqual(float(result.context.damage.last_dealt[0,1].sum()),.4,places=6)
        self.assertAlmostEqual(float(result.context.damage.last_taken[0,0].sum()),.6,places=6)
        damage=next(p for p in projections if p.raw_line=='|-damage|p2a: Absol|75/100|[from] Spikes')
        self.assertEqual(damage.events[-1].kind,self.i.Kind.CONDITION)
        self.assertEqual(damage.events[-1].payload[5:],(1,1,1))

    def test_move_window_and_precondition_metadata_cross_chunk_boundary(self):
        lines=[*self.prefix,'|turn|1','|move|p1a: Snorlax|Body Slam|p2a: Absol',*(['|']*27),
            '|-damage|p2a: Absol|75/100','|-damage|p2a: Absol|50/100|[from] Spikes',
            '|-enditem|p2a: Absol|Leftovers|[from] move: Knock Off','|turn|2']
        result,projections=self.run_lines(lines);self.compare(result,lines)
        chunks=self.chunks(projections);self.assertGreater(len(chunks),1)
        self.assertEqual(int(chunks[0].words[0,63,0]),self.i.Kind.CONDITION)
        self.assertEqual(chunks[0].words[0,63,13:16].tolist(),[1,0,0])
        self.assertAlmostEqual(float(result.context.damage.last_dealt[0,0].sum()),.25,places=6)
        self.assertAlmostEqual(float(result.context.damage.last_taken[0,1].sum()),.5,places=6)

    def test_malformed_metadata_or_namespace_rolls_back_every_state_leaf(self):
        lines=[*self.prefix,'|switch|p2a: Gardevoir|Gardevoir, L80|100/100',
            '|switch|p2a: Absol|Absol, L80|100/100','|turn|1','|move|p1a: Snorlax|Body Slam|p2a: Absol',
            '|-damage|p2a: Absol|80/100','|turn|2']
        _,p=self.project(lines);chunk=self.chunks(p)[0];initial=self.initial()
        for bad in (chunk._replace(damage_binding=None),chunk._replace(damage_binding=chunk.damage_binding.at[0].add(1))):
            got=self.fold(initial,bad);self.assertEqual(got.error.tolist(),[self.i.Error.SOURCE]);self.same(initial,self.carry(got))
        index=next(n for n in range(64) if int(chunk.words[0,n,0])==self.i.Kind.CONDITION)
        for changes in ({13:2},{14:2},{15:1},{9:101},{7:1,8:2}):
            words=chunk.words
            for column,value in changes.items():words=words.at[0,index,column].set(value)
            got=self.jax.jit(lambda cur,b,m,c,ch:self.fold((cur,b,m,c),ch))(*initial,chunk._replace(words=words))
            self.assertNotEqual(int(got.error[0]),0);self.same(initial,self.carry(got))
        good=self.fold(initial,chunk);self.assertEqual(good.error.tolist(),[0])
        wire,_=self.project(lines);_,more=self.project(['|'],state=wire)
        broken=good.context._replace(damage=good.context.damage._replace(turn=self.j.zeros_like(good.context.damage.turn)))
        got=self.fold((good.cursor,good.beliefs,good.mon_state,broken),self.chunks(more)[0])
        self.assertEqual(got.error.tolist(),[self.i.Error.SOURCE])

    def test_no_consumer_or_downgraded_continuation_cannot_swallow_damage_metadata(self):
        wire,_=self.project(self.prefix)
        legacy=self.c.compile_context(self.host,self.abilities,self.catalog,items=self.item_catalog,field_catalog=self.field_catalog)
        for changes in (dict(damage_tables=None),dict(damage_tables=None,context_tables=legacy),dict(context_tables=legacy)):
            kwargs=dict(self.kwargs,context_tables=self.contexts[False]);kwargs.update(changes)
            with self.assertRaises(ValueError):self.wire.project_line(wire,'|',**kwargs)
        _,modern=self.project([*self.prefix,'|turn|1','|-damage|p2a: Absol|75/100'])
        initial=self.initial()
        bad_tables=self.contexts[False]._replace(field=None)
        with self.assertRaises(ValueError):self.c.initialize(initial[1],bad_tables,species_binding=self.species_catalog.device.binding)
        with self.assertRaises(ValueError):self.i.fold_chunk(initial[0],initial[1],self.host.device,
            self.chunks(modern)[0]._replace(field_binding=None),mon_state=initial[2],
            context=initial[3]._replace(field=None),context_tables=bad_tables)
        # A legacy bound mon consumer must reject nonzero damage metadata rather
        # than apply the new HP and silently omit its attribution consumer.
        _,old=self.project([*self.prefix,'|-damage|p2a: Absol|75/100'],damage_tables=None,context_tables=legacy)
        import numpy as np
        p=old[0]
        ch=list(self.i.pack_chunks(np.array([1],np.int32),[[e for x in old for e in x.events]],
            source_binding=self.host.device.binding,**{n:np.asarray(getattr(p,n),np.uint32) for n in (
                'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding')}))[0]
        index=next(n for n in range(64) if int(ch.words[0,n,0])==self.i.Kind.CONDITION)
        ctx=self.c.initialize(initial[1],legacy,species_binding=self.species_catalog.device.binding)
        got=self.i.fold_chunk(initial[0],initial[1],self.host.device,ch._replace(words=ch.words.at[0,index,13].set(1)),
            mon_state=initial[2],context=ctx,context_tables=legacy)
        self.assertNotEqual(int(got.error[0]),0)

    def test_damage_and_active_consumers_commit_together_under_the_same_turn(self):
        import numpy as np
        from gpu_engine import public_active
        active=public_active.compile_active(self.vocab)
        tables=self.c.compile_context(self.host,self.abilities,self.catalog,items=self.item_catalog,
            field_catalog=self.field_catalog,active_catalog=active,damage_tables=self.damage_tables)
        lines=[*self.prefix,'|turn|1','|-start|p2a: Absol|confusion',
            '|move|p1a: Snorlax|Body Slam|p2a: Absol','|-damage|p2a: Absol|80/100',
            '|-activate|p2a: Absol|confusion','|-damage|p2a: Absol|70/100','|turn|2']
        _,projections=self.project(lines,active_catalog=active,context_tables=tables)
        p=projections[0]
        chunks=list(self.i.pack_chunks(np.array([1],np.int32),[[e for p in projections for e in p.events]],
            source_binding=self.host.device.binding,**{n:np.asarray(getattr(p,n),np.uint32) for n in (
                'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding','active_binding','damage_binding')}))
        initial=self.initial();ctx=self.c.initialize(initial[1],tables,species_binding=self.species_catalog.device.binding)
        result=self.i.fold_chunk(initial[0],initial[1],self.host.device,chunks[0],mon_state=initial[2],context=ctx,context_tables=tables)
        self.assertEqual(result.error.tolist(),[0]);self.compare(result,lines)
        self.assertEqual(int(result.context.active.elapsed[0,1,0]),1)
        self.assertEqual(int(result.context.damage.turn[0]),int(result.context.active.turn[0]))
        self.assertAlmostEqual(float(result.context.damage.last_dealt[0,0].sum()),.2,places=6)
        bad=chunks[0]._replace(damage_binding=None)
        rejected=self.i.fold_chunk(initial[0],initial[1],self.host.device,bad,mon_state=initial[2],context=ctx,context_tables=tables)
        self.assertEqual(rejected.error.tolist(),[self.i.Error.SOURCE])
        self.same((initial[0],initial[1],initial[2],ctx),self.carry(rejected))


if __name__=='__main__':unittest.main()
