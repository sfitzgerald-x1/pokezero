"""Non-proc upkeep stages against actual Python source transcripts."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class ItemNonprocTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_item_context import PublicItemContextTests
        from gpu_engine import public_item_nonproc
        PublicItemContextTests.setUpClass();cls.f=PublicItemContextTests();cls.x=public_item_nonproc
        cls.j,cls.jax=cls.f.j,cls.f.jax

    def run_case(self,lines):
        from types import SimpleNamespace
        from gpu_engine.public_mon_protocol import condition_words
        from pokezero import showdown
        import numpy as np
        f=self.f;initial,_=f.run_lines(f.prefix);mons=initial.mon_state;items=initial.context.items
        state=self.x.initialize(1,species_binding=f.species_catalog.device.binding)._replace(encounter=mons.encounter)
        for side in (0,1):
            state,ok=self.x.record(state,mons,operation=self.x.ARRIVE,side=side,slot=0,encounter=1,hp=100,maximum=100)
            self.assertTrue(bool(ok.all()))
        sleep=SimpleNamespace(species_binding=mons.species_binding,encounter=mons.encounter,active=mons.active,
            live=self.j.ones((1,2),bool),status=mons.status)
        engine=f.Reference.from_events([showdown._public_event_from_line(raw) for raw in f.prefix],
            format_id='gen3randombattle',set_source=f.source)
        calls=[];replace=engine._replace_belief
        def capture(pokemon,**changes):
            value=replace(pokemon,**changes)
            if 'ruled_out_items' in changes:calls.append(value)
            return value
        engine._replace_belief=capture
        sweeps=0
        for line in lines:
            facts=self.x.lexical_facts(line);parts=line.split('|');kind=parts[1]
            operation=self.x.NOTE;side=slot=-1;enc=hp=maximum=0
            if facts.hp_line or facts.leftovers_heal or facts.berry_eat:
                side=int(parts[2][1])-1;slot=0;enc=1
            if facts.hp_line:
                operation=self.x.HP;hp,maximum,status,faint=condition_words(parts[3])
                mons=mons._replace(hp_numerator=mons.hp_numerator.at[0,side,0].set(hp),
                    hp_denominator=mons.hp_denominator.at[0,side,0].set(maximum),
                    hp_known=mons.hp_known.at[0,side,0].set(maximum>0),fainted=mons.fainted.at[0,side,0].set(bool(faint)))
            if kind=='-status':sleep.status=sleep.status.at[0,int(parts[2][1])-1,0].set({'par':2,'psn':5}[parts[3]])
            projection=f.items.project_line(line,items=f.item_catalog)
            for ident,op,item in projection.facts:
                items,ok=f.items.record(items,mons,f.item_catalog.device,side=int(ident[1])-1,slot=0,encounter=1,
                    item_id=item,item_binding=f.item_catalog.device.binding,operation=op)
                self.assertTrue(bool(ok.all()))
            state,ok=self.jax.jit(lambda s:self.x.record(s,mons,operation=operation,side=side,slot=slot,encounter=enc,
                hp=hp,maximum=maximum,phase=facts.phase,action_phase_hp=facts.action_phase_hp,
                residual_phase_hp=facts.residual_phase_hp,leftovers_heal=facts.leftovers_heal,berry_eat=facts.berry_eat))(state)
            self.assertTrue(bool(ok.all()),line)
            calls.clear();engine.ingest_event(showdown._public_event_from_line(line))
            if facts.upkeep:
                result=self.jax.jit(lambda s:self.x.sweep(s,mons,items,sleep,item_tables=f.item_catalog.device))(state)
                self.assertTrue(bool(result.admitted.all()));scheduled=[]
                for owner in (0,1):
                    for stage in range(3):
                        if result.callbacks[0,owner,stage]:scheduled.append((owner,result.stages[0,owner,stage]))
                self.assertEqual(len(scheduled),len(calls),line)
                for (owner,ledger),pokemon in zip(scheduled,calls):
                    self.assertEqual(pokemon.showdown_slot,f'p{owner+1}')
                    actual={name for i,name in enumerate(self.x.NAMES) if ledger[owner,0,i]}
                    self.assertEqual(actual,set(pokemon.ruled_out_items)&set(self.x.NAMES))
                state=result.state;sweeps+=1
                self.assertFalse(bool(state.maximum.any()));self.assertFalse(bool(state.leftovers_healed.any()))
                self.assertFalse(bool(state.berry_ate.any()))
            for owner in (0,1):
                pokemon=engine.snapshot().side(f'p{owner+1}')[0]
                actual={name for i,name in enumerate(self.x.NAMES) if state.excluded[0,owner,0,i]}
                self.assertEqual(actual,set(pokemon.ruled_out_items)&set(self.x.NAMES),line)
        self.assertGreater(sweeps,0)
        return state

    def test_real_hp_phase_threshold_status_and_stale_snapshot_transcripts(self):
        cases=(['|-damage|p1a: Snorlax|20/100','|-status|p1a: Snorlax|par','|','|upkeep','|turn|2','|','|upkeep'],
            ['|-damage|p1a: Snorlax|20/100','|','|-heal|p1a: Snorlax|100/100|[from] move: Wish','|upkeep'],
            ['|','|-damage|p1a: Snorlax|20/100|[from] psn','|upkeep'],
            ['|-damage|p1a: Snorlax|20/100','|','|-heal|p1a: Snorlax|90/100|[silent]','|upkeep'],
            ['|-damage|p1a: Snorlax|20/100','|','|-damage|p1a: Snorlax|10/100|[from] ability: Liquid Ooze','|upkeep'],
            ['|','|-damage|p1a: Snorlax|20/100|[from] confusion','|upkeep'],
            ['|-damage|p1a: Snorlax|30/100','|-damage|p2a: Absol|25/100','|-status|p2a: Absol|psn','|upkeep'],
            ['|-damage|p1a: Snorlax|20/100','|-heal|p1a: Snorlax|26/100|[from] item: Leftovers','|upkeep'],
            ['|-damage|p1a: Snorlax|20/100','|-enditem|p1a: Snorlax|Sitrus Berry|[eat]','|upkeep'])
        for lines in cases:self.run_case(lines)

    def test_exact_integer_full_hp_boundary_and_mixed_refusal(self):
        import numpy as np
        from types import SimpleNamespace
        f=self.f;initial,_=f.run_lines(f.prefix);repeat=lambda a:self.j.repeat(a,8,axis=0)
        mons=self.jax.tree.map(repeat,initial.mon_state);items=self.jax.tree.map(repeat,initial.context.items)
        state=self.x.initialize(8,species_binding=f.species_catalog.device.binding)._replace(encounter=mons.encounter)
        state=state._replace(hp=state.hp.at[:,0,0].set(2**31-2),maximum=state.maximum.at[:,0,0].set(2**31-1))
        sleep=SimpleNamespace(species_binding=mons.species_binding,encounter=mons.encounter,active=mons.active,
            live=self.j.ones((8,2),bool),status=mons.status)
        good=self.jax.jit(lambda s:self.x.sweep(s,mons,items,sleep,item_tables=f.item_catalog.device))(state)
        self.assertTrue(bool(good.callbacks[:,0,0].all()))
        bad=state._replace(species_binding=state.species_binding.at[0,0].set(state.species_binding[0,0]^self.j.uint32(1)),
            hp=state.hp.at[1,0,0].set(-1))
        result=self.x.sweep(bad,mons,items,sleep,item_tables=f.item_catalog.device)
        self.assertEqual(result.admitted.tolist(),[False,False]+[True]*6)
        for a,b in zip(self.jax.tree.leaves(result.state),self.jax.tree.leaves(bad)):np.testing.assert_array_equal(a[:2],b[:2])
        self.assertFalse(bool(result.callbacks[:2].any()))
        disabled=self.x.sweep(state,mons,items,sleep,item_tables=f.item_catalog.device,valid=False)
        for a,b in zip(self.jax.tree.leaves(disabled.state),self.jax.tree.leaves(state)):np.testing.assert_array_equal(a,b)
        with self.assertRaisesRegex(ValueError,'shape/dtype'):self.x.sweep(state._replace(hp=state.hp.astype(self.j.float32)),mons,items,sleep,item_tables=f.item_catalog.device)
        with self.assertRaisesRegex(ValueError,'private'):self.x.lexical_facts('|request|{}')

    def test_stale_active_item_namespace_and_reveal_id_refuse_before_upkeep(self):
        import numpy as np
        from types import SimpleNamespace
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.showdown import _public_event_from_line
        f=self.f;initial,_=f.run_lines(f.prefix);mons=initial.mon_state;items=initial.context.items
        state=self.x.initialize(1,species_binding=f.species_catalog.device.binding)._replace(encounter=mons.encounter)
        state=state._replace(hp=state.hp.at[0,0,0].set(20),maximum=state.maximum.at[0,0,0].set(100))
        sleep=SimpleNamespace(species_binding=mons.species_binding,encounter=mons.encounter,active=mons.active,
            live=self.j.ones((1,2),bool),status=mons.status)
        variants=((mons._replace(active=mons.active.at[0,0].set(-1)),items),
            (mons,items._replace(revealed=items.revealed.at[0,0,0].set(-1))),
            (mons,items._replace(revealed=items.revealed.at[0,0,0].set(f.item_catalog.device.predicate.shape[0]))),
            (mons,items._replace(binding=items.binding.at[0,0].set(items.binding[0,0]^self.j.uint32(1)))))
        for public,item in variants:
            result=self.x.sweep(state,public,item,sleep,item_tables=f.item_catalog.device)
            self.assertEqual(result.admitted.tolist(),[False]);self.assertFalse(bool(result.callbacks.any()))
            for a,b in zip(self.jax.tree.leaves(result.state),self.jax.tree.leaves(state)):np.testing.assert_array_equal(a,b)
        for marker in ('|','| ','|  '):
            engine=PublicBattleBeliefEngine();engine.ingest_event(_public_event_from_line(marker))
            self.assertEqual(self.x.lexical_facts(marker).phase,self.x.RESIDUAL_PHASE)
            self.assertTrue(engine._in_residual_phase)

    def test_static_exclusion_predicates_bind_exact_source_item_and_sleep_namespaces(self):
        from gpu_engine import public_sleep
        f=self.f;sleep_tables=public_sleep.compile_sleep(f.species_catalog,f.vocab)
        tables=self.x.compile_nonproc(f.host,f.species_catalog,f.item_catalog,sleep_tables)
        self.assertEqual(tables.predicates.tolist(),[f.host.predicate_id('item',name) for name in self.x.NAMES])
        self.assertEqual(tables.source_binding.tolist(),f.host.device.binding.tolist())
        self.assertEqual(tables.species_binding.tolist(),f.species_catalog.device.binding.tolist())
        self.assertEqual(tables.binding.shape,(8,))
        bad=sleep_tables._replace(species_binding=sleep_tables.species_binding.at[0].set(sleep_tables.species_binding[0]^self.j.uint32(1)))
        with self.assertRaisesRegex(ValueError,'one frozen'):self.x.compile_nonproc(f.host,f.species_catalog,f.item_catalog,bad)
        changed=self.x.compile_nonproc(f.host,f.species_catalog,f.item_catalog,sleep_tables._replace(
            binding=sleep_tables.binding.at[0].set(sleep_tables.binding[0]^self.j.uint32(1))))
        self.assertFalse(bool(self.j.array_equal(changed.binding,tables.binding)))
