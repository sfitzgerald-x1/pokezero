"""Raw pending-switch chronology and checkpoint parity in the complete fold."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class SwitchContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataclasses import replace
        from types import SimpleNamespace
        from test_gpu_engine_v4_encoder import V4EncoderTests
        from gpu_engine import public_belief
        from pokezero.category_vocab import CategoryVocabulary
        V4EncoderTests.setUpClass();f=V4EncoderTests();raw=f.public.source;universes=dict(raw.universes)
        old=universes['snorlax']
        universes['snorlax']=replace(old,variants=tuple(replace(old.variants[0],variant_id='switch'+str(i),ability=n)
            for i,n in enumerate(('Intimidate','Levitate'))))
        raw=replace(raw,universes=universes);source=public_belief.compile_source(raw)
        labels=set(f.vocab.tokens)|{'ability:intimidate','belief:possible_ability:intimidate'}
        vocab=CategoryVocabulary(tuple(sorted(labels)))
        catalog=f.e.compile_encoder(source,f.catalog.dex,vocab,checkpoint_sha256='a'*64,feature_masks=f.masks)
        f.source,f.vocab,f.catalog=source,vocab,catalog
        f.public=SimpleNamespace(source=raw,prefix=('|switch|p1a: Nick|Snorlax, L80|100/100',
            '|switch|p2a: Absol|Absol, L80|100/100'))
        original=f.request
        def request(owner=0):
            packet=original(owner)
            for pokemon in packet['side']['pokemon']:pokemon['ability']='Pressure'
            return packet
        f.request=request
        cls.f,cls.j,cls.jax,cls.catalog=f,f.j,f.jax,catalog
        cls.step=cls.jax.jit(lambda s,c:f.e.fold_chunk(s,c,catalog))
        cls.lines=[*f.public.prefix,'|-ability|p2a: Absol|Pressure',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100','|switch|p1a: Nick|Snorlax, L80|100/100',
            '|-boost|p2a: Absol|atk|1','|-damage|p2a: Absol|80/100','|',
            '|move|p1a: Nick|Surf|p2a: Absol','|switch|p1a: Gardevoir|Gardevoir, L80|100/100',
            '|switch|p1a: Nick|Snorlax, L80|100/100','|']

    def test_every_prefix_matches_actual_pending_ledger_and_source_refreshes(self):
        from gpu_engine import public_wire,public_belief_snapshot
        from pokezero import showdown
        from pokezero.belief import PublicBattleBeliefEngine
        state=self.f.e.initialize(self.catalog,stream_ids=self.j.array([1],self.j.int32));host=public_wire.WireState()
        engine=PublicBattleBeliefEngine(format_id='gen3randombattle',set_source=self.f.public.source)
        calls=[];original=engine._with_set_summary
        def record(pokemon):
            result=original(pokemon);calls.append(result.key);return result
        engine._with_set_summary=record
        for line in self.lines:
            candidate,projection=public_wire.project_line(host,line,**self.f.e.wire_kwargs(self.catalog))
            self.assertEqual([event.kind.name for event in projection.events[:2]],['RAW_BEGIN','SWITCH_PRELUDE'])
            for chunk in self.f.chunks([projection]):
                state,error=type(self).step(state,chunk);self.assertEqual(error.tolist(),[0],line)
            host=candidate;engine.ingest_event(showdown._public_event_from_line(line))
            summary,ok=public_belief_snapshot.read(state.context.refresh.controller.visible,state.beliefs,self.catalog.source.device)
            self.assertTrue(bool(ok.all()))
            for side in (0,1):
                for slot,pokemon in enumerate(engine.snapshot().side(f'p{side+1}')):
                    self.assertEqual(int(summary.count[0,1-side,slot]),pokemon.candidate_set_count,line)
                    self.assertEqual(int(state.context.refresh.controller.visible.generation[0,1-side,slot]),calls.count(pokemon.key),line)
                    self.assertEqual(bool(state.context.switches.ruled_out[0,side,slot]),'Intimidate' in pokemon.ruled_out_abilities,line)
            pending=engine._pending_switches
            if pending:
                side=int(pending[0].showdown_slot[1])-1
                slot=next(i for i,p in enumerate(engine.snapshot().side(pending[0].showdown_slot)) if p.species==pending[0].species)
                self.assertEqual((int(state.context.switches.side[0]),int(state.context.switches.slot[0])),(side,slot),line)
            else:self.assertEqual(state.context.switches.side.tolist(),[-1],line)
        self.assertTrue(bool(state.context.switches.ruled_out[0,0,0]))
        # Consistent narrowing and inconsistent fallback are both actual source
        # observations. Hazard uses raw possible abilities; Exact subtracts the
        # explicit ledger. Their legacy distinction must survive registration.
        for end in (5,8,10,11,len(self.lines)):
            self.f.compare_source(self.lines[:end])
        belief=engine.snapshot().side('p1')[0]
        self.assertEqual(set(belief.possible_abilities),{'Intimidate','Levitate'})
        self.assertTrue(showdown._is_grounded_for_spikes(showdown.ShowdownPokemon(ident='p1: Snorlax',showdown_slot='p1',species='Snorlax'),
            belief=belief,dex=self.catalog.dex))

    def test_missing_duplicate_prelude_namespace_and_late_chunk_refuse_atomically(self):
        import numpy as np
        from dataclasses import replace
        from gpu_engine import public_inputs
        initial=self.f.e.initialize(self.catalog,stream_ids=self.j.array([1],self.j.int32))
        projections=self.f.projections(self.lines[:3])
        events=[event for p in projections for event in p.events]
        def pack(items,batch=1):
            first=projections[0]
            bindings={name:np.asarray(getattr(first,name),np.uint32) for name in (
                'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding',
                'active_binding','damage_binding','type_binding','sleep_binding','toxic_binding','tendency_binding')}
            items=[replace(event,sequence=i+1) for i,event in enumerate(items)]
            return list(public_inputs.pack_chunks(np.arange(1,batch+1,dtype=np.int32),[items]*batch,
                source_binding=self.catalog.source.device.binding,**bindings))
        index=next(i for i,event in enumerate(events) if event.kind==public_inputs.Kind.SWITCH_PRELUDE)
        malformed=(events[:index]+events[index+1:],events[:index]+[events[index]]+events[index:])
        for items in malformed:
            out,error=type(self).step(initial,pack(items)[0]);self.assertNotEqual(error.tolist(),[0])
            for a,b in zip(self.jax.tree.leaves(initial),self.jax.tree.leaves(out)):
                np.testing.assert_array_equal(a,b)
        # Typed state/table admission is checked even on a truly empty chunk.
        empty=pack(events)[0]._replace(valid=self.j.zeros((1,64),bool),more=self.j.zeros(1,bool))
        bad=initial._replace(context=initial.context._replace(switches=initial.context.switches._replace(
            binding=initial.context.switches.binding.at[0,0].set(initial.context.switches.binding[0,0]^self.j.uint32(1)))))
        out,error=type(self).step(bad,empty);self.assertEqual(error.tolist(),[int(public_inputs.Error.SOURCE)])
        np.testing.assert_array_equal(out.context.switches.binding,bad.context.switches.binding)
        for state,tables in ((initial._replace(context=initial.context._replace(switches=None)),self.catalog.context),
                (initial,self.catalog.context._replace(switches=None))):
            result=public_inputs.fold_chunk(state.cursor,state.beliefs,self.catalog.source.device,empty,
                mon_state=state.mons,context=state.context,context_tables=tables)
            self.assertEqual(result.error.tolist(),[int(public_inputs.Error.SOURCE)])

    def test_observation_boundary_is_ephemeral_and_preserves_next_actual_trigger(self):
        import numpy as np
        from gpu_engine import public_belief_snapshot
        state=self.f.ingest(self.lines[:5])
        self.assertEqual(state.context.switches.side.tolist(),[0])
        before=[np.asarray(value).copy() for value in self.jax.tree.leaves(state)]
        own=self.f.e.parse_own_requests(self.catalog,[self.f.request(1)],owner_sides=[1])
        encode=self.jax.jit(lambda s:self.f.e.encode(s,own,self.catalog,observer_sides=self.j.array([1],self.j.int32)))
        first=encode(state);second=encode(state)
        self.assertEqual(first.observation.admitted.tolist(),[True])
        for x,y in zip(self.jax.tree.leaves(first),self.jax.tree.leaves(second)):np.testing.assert_array_equal(x,y)
        for old,new in zip(before,self.jax.tree.leaves(state)):np.testing.assert_array_equal(old,new)
        self.assertFalse(bool(state.context.switches.ruled_out[0,0,0]))
        # The next real Intimidate trigger resolves the still-live queue; the
        # ephemeral observed negative must not become a persistent exclusion.
        lines=self.lines[:5]+['|-ability|p1a: Nick|Intimidate']
        projection=self.f.projections(lines)[-1]
        for chunk in self.f.chunks([projection]):state,error=type(self).step(state,chunk);self.assertEqual(error.tolist(),[0])
        self.assertEqual(state.context.switches.side.tolist(),[-1])
        self.assertFalse(bool(state.context.switches.ruled_out[0,0,0]))
        summary,ok=public_belief_snapshot.read(state.context.refresh.controller.visible,state.beliefs,self.catalog.source.device)
        self.assertTrue(bool(ok.all()));self.assertEqual(int(summary.count[0,1,0]),1)
        self.assertEqual(int(state.context.abilities.original[0,0,0]),self.catalog.abilities.ability_id('Intimidate'))
        self.f.compare_source(lines)
