"""Pending Mud Shot reducer against actual public-source evidence."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class MudShotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataclasses import replace
        from test_gpu_engine_public_switch_evidence import SwitchEvidenceTests
        from test_gpu_engine_v4_encoder import V4EncoderTests
        from gpu_engine import public_mudshot as x,public_belief,public_abilities,public_team
        SwitchEvidenceTests.setUpClass();f=SwitchEvidenceTests();v=V4EncoderTests()
        universes=dict(f.raw.universes)
        for species,names in (('snorlax',('Pressure',)),('absol',('Shield Dust','Pressure'))):
            original=universes[species]
            universes[species]=replace(original,variants=tuple(replace(original.variants[0],variant_id=species+str(i),ability=name)
                for i,name in enumerate(names)))
        f.raw=replace(f.raw,universes=universes);f.source=public_belief.compile_source(f.raw)
        f.abilities=public_abilities.compile_abilities(f.source,v.vocab,ability_names=('Shield Dust','Levitate'))
        f.species=public_team.compile_species(f.source,v.catalog.dex,v.vocab,f.abilities)
        cls.f,cls.x,cls.j,cls.jax=f,x,f.j,f.jax
        cls.tables=x.compile_mudshot(f.source,f.species,f.abilities)
        cls.prefix=['|switch|p1a: Nick|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100']

    def apply(self,state,engine,op,**kwargs):
        m,a,e,l=self.f.snapshot(engine)
        return self.jax.jit(lambda s:self.x.apply(s,m,a,self.f.source.device,self.tables,e,l,
            operation=op,producer_binding=self.tables.binding,**kwargs))(state)

    def test_real_plain_tagged_unboost_faint_and_original_reveal_paths(self):
        from pokezero.showdown import _public_event_from_line
        cases=([],['|-damage|p2a: Absol|80/100'],
            ['|-damage|p2a: Absol|80/100|[from] psn'],
            ['|-damage|p2a: Absol|80/100','|-unboost|p2a: Absol|def|1'],
            ['|-damage|p2a: Absol|0 fnt','|faint|p2a: Absol'],
            ['|-damage|p2a: Absol|80/100','|-ability|p2a: Absol|Pressure'],
            ['|-damage|p2a: Absol|80/100','|-ability|p2a: Absol|Shield Dust'])
        for suffix in cases:
            engine=self.f.engine(self.prefix+['|move|p1a: Nick|Mud Shot|p2a: Absol'])
            result=self.apply(self.x.initialize(1,self.tables),engine,self.x.QUEUE,side=1,slot=0,encounter=1)
            self.assertTrue(bool(result.admitted.all()));state=result.state
            for line in suffix:
                engine.ingest_event(_public_event_from_line(line));kind=line.split('|')[1]
                if kind in ('-damage','-unboost','faint'):
                    op={'-damage':self.x.DAMAGE,'-unboost':self.x.UNBOOST,'faint':self.x.FAINT}[kind]
                    kwargs={'tagged':'[from]' in line} if kind=='-damage' else {}
                    result=self.apply(state,engine,op,side=1,slot=0,encounter=1,**kwargs)
                    self.assertTrue(bool(result.admitted.all()));state=result.state
                pending=engine._pending_mudshot
                self.assertEqual(state.side.tolist(),[1 if pending else -1])
                if pending:
                    self.assertEqual(state.saw_damage.tolist(),[pending['saw_damage']])
                    self.assertEqual(state.cancelled.tolist(),[pending['cancelled']])
            result=self.apply(state,engine,self.x.RESOLVE)
            before=engine.snapshot().side('p2')[0].revealed_ability
            engine._resolve_pending_mudshot();after=engine.snapshot().side('p2')[0].revealed_ability
            self.assertEqual(result.reveal.tolist(),[not bool(before) and after=='Shield Dust'],suffix)
            self.assertEqual(result.state.side.tolist(),[-1]);self.assertIsNone(engine._pending_mudshot)

    def test_cached_candidate_guard_empty_fallback_and_no_live_target(self):
        engine=self.f.engine(self.prefix);m,a,e,l=self.f.snapshot(engine)
        pending=self.apply(self.x.initialize(1,self.tables),engine,self.x.QUEUE,side=1,slot=0,encounter=1).state
        pending=self.apply(pending,engine,self.x.DAMAGE,side=1,slot=0,encounter=1).state
        pid=self.f.abilities.ability_id('Pressure');predicate=self.f.abilities.device.predicate[pid]
        pressure=self.f.source.device.predicates[int(predicate),self.f.source.species_id('Absol')]
        for mask,want in ((e.at[0,0,0].set(pressure),False),(self.j.zeros_like(e),True)):
            result=self.x.apply(pending,m,a,self.f.source.device,self.tables,mask,l,
                operation=self.x.RESOLVE,producer_binding=self.tables.binding)
            self.assertEqual(result.reveal.tolist(),[want]);self.assertEqual(result.state.side.tolist(),[-1])
        empty=self.x.initialize(1,self.tables)
        result=self.apply(empty,engine,self.x.QUEUE)
        self.assertTrue(bool(result.admitted.all()));self.assertEqual(result.state.side.tolist(),[-1])

    def test_batch_binding_identity_disabled_and_typed_controls(self):
        import numpy as np
        engine=self.f.engine(self.prefix);m,a,e,l=self.f.snapshot(engine);repeat=lambda v:self.j.repeat(v,8,axis=0)
        m,a,e,l=(self.jax.tree.map(repeat,value) for value in (m,a,e,l))
        state=self.x.initialize(8,self.tables)
        run=lambda s,**kw:self.x.apply(s,m,a,self.f.source.device,self.tables,e,l,
            operation=self.x.QUEUE,side=1,slot=0,encounter=1,producer_binding=self.tables.binding,**kw)
        good=self.jax.jit(run)(state);self.assertEqual(good.admitted.tolist(),[True]*8)
        bad=state._replace(binding=state.binding.at[0,0].set(state.binding[0,0]^self.j.uint32(1)),
            slot=state.slot.at[1].set(0))
        result=run(bad);self.assertEqual(result.admitted.tolist(),[False,False]+[True]*6)
        for actual,old in zip(self.jax.tree.leaves(result.state),self.jax.tree.leaves(bad)):np.testing.assert_array_equal(actual[:2],old[:2])
        disabled=run(state,valid=False)
        for actual,old in zip(self.jax.tree.leaves(disabled.state),self.jax.tree.leaves(state)):np.testing.assert_array_equal(actual,old)
        with self.assertRaisesRegex(ValueError,'shape/dtype'):run(state._replace(encounter=state.encounter.astype(self.j.float32)))
