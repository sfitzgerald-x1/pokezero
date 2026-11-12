"""One original self-boost count with two disjoint registered source handlers."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from types import SimpleNamespace
import test_gpu_engine_calm_mind_components as fixture
from gpu_engine import transition as tr,engine_lifecycle_callbacks as life
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_swords_dance_callbacks as swords,engine_source_frame_witness as witness
from gpu_engine import engine_full_v4_event_device as full,public_wire,v4_encoder as v4,engine_public_events as events


class SelfBoostUnionTests(unittest.TestCase):
    def test_both_registered_sources_one_literal_assignment_and_both_full_drains(self):
        fixture.CalmMindComponentTests.setUpClass();f=fixture.CalmMindComponentTests()
        sword=swords.compile_swords_dance(engine_catalog=f.producer,metadata=f.metadata)
        f.risks=risk.compile_risks(f.producer,f.account,calm_mind_compiled=f.helper,swords_dance_compiled=sword)
        sword_context=swords.Context(sword.device,f.account.device)
        state,cursor,consumers=f.initial()
        state=state._replace(moves=state.moves.at[0,1,0,:].set(f.ids['swordsdance']).at[1,0,0,:].set(f.ids['swordsdance']))
        chance=fixture.fixtures.engine_fixture.EnginePublicEventsTests().chance(2)
        tape=life._tape(cursor,f.producer,64)._replace(refusal=j.ones(2,j.int32))
        tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=f.account),f.account.device,active=j.ones(2,bool))
        carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,f.risks.device),tape,f.risks.device,active=j.ones(2,bool))
        tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,f.risks.device)))
        actual=tr.step_with_chance(state,j.zeros((2,2),j.int32),f.table,chance,_public_trace=tape,_lifecycle_trace=True,
            _calm_mind_context=f.context,_swords_dance_context=sword_context)
        plain=tr.step_with_chance(state,j.zeros((2,2),j.int32),f.table,chance)
        for a,z in zip(jax.tree.leaves(actual.step.state),jax.tree.leaves(plain.state)):np.testing.assert_array_equal(a,z)
        self.assertEqual(actual.callbacks.execution.risks.carry.counts[:,int(risk.Branch.SELF_BOOST)-1].tolist(),[2,2])
        self.assertEqual(actual.callbacks.execution.risks.carry.risk.tolist(),[0,0]);self.assertEqual(actual.callbacks.refusal.tolist(),[1,1])
        tables=SimpleNamespace(risk=f.risks.device,secondary_weather=j.zeros(len(f.table.power),bool),direct_hit_binding=j.zeros(8,j.uint32),substitute_binding=j.zeros(8,j.uint32),toxic_binding=j.zeros(8,j.uint32))
        self.assertTrue(bool(witness._raw_complete(actual.callbacks,tables).all()))
        expected,_,geometry=witness._geometry(SimpleNamespace(frame=j.ones(2,j.int32)),actual.callbacks)
        self.assertTrue(bool(geometry.all()));np.testing.assert_array_equal(expected,actual.callbacks.execution.risks.carry.counts)
        frame=life.LifecycleResult(actual.step.state,actual.callbacks.cursor,actual.callbacks,life.snapshot(cursor),life.snapshot(actual.callbacks.cursor),j.zeros(2,bool),j.zeros(2,j.int32))
        closed=[]
        for owner,(stream,public) in enumerate(consumers):
            ledger,result=ex.consume_frame_device(actual.callbacks.execution.ledger,frame,stream,public,active=j.ones(2,bool),tables=f.account.device,adapter_tables=f.adapter.device,catalog=f.catalog);closed.append(ledger)
            self.assertEqual(result.projection.error.tolist(),[0,0]);self.assertEqual(ledger.site[:,0].tolist(),[int(ex.Site.TURNLOOP_PREFIX)]*2);self.assertEqual(ledger.stage[:,:3].tolist(),[[3,3,3],[3,3,3]])
            self.assertEqual(result.public.mons.boosts[:,:,0].tolist(),[[0,2],[2,0]])
            self.assertEqual(result.public.mons.boosts[:,:,2:4].tolist(),[[[1,1],[0,0]],[[0,0],[1,1]]])
            self.assertFalse(bool(result.projection.B4_qualified.any()))
            for row in (0,1):
                names=tuple(f.catalog.species.names[int(f.adapter.device.causal.species[i])] for i in (1,2));ident={s:f'p{s+1}a: '+names[s] for s in (0,1)}
                prefix=[f'|switch|{ident[s]}|{names[s]}, L80, M|100/100' for s in (0,1)]+['|turn|1']
                wire,_=public_wire.project_lines(public_wire.WireState(),prefix,**v4.wire_kwargs(f.catalog));lines=[]
                for index,w in enumerate(np.array(actual.callbacks.words[row,:int(actual.callbacks.count[row])])):
                    kind=int(w[0]);side=int(w[1])
                    if kind==events.MOVE:
                        move=f.metadata['vocabulary']['moves'][int(w[5])];label=f.catalog.dex.moves[move].name;lines.append(f'|move|{ident[side]}|{label}|{ident[side]}')
                    elif kind==30:lines.append(f'|-boost|{ident[side]}|'+('spa' if int(w[4])==2 else 'spd')+'|1')
                    elif kind==32:lines.append(f'|-boost|{ident[side]}|atk|2')
                    elif kind==life.TURNLOOP_PREFIX:
                     self.assertEqual(w.tolist(),[35,-1,-1,-1,-1,0,0,0,0,0])
                     lines.extend(['|','|t:|1'])
                    elif kind==life.ACTION_END:lines.append('|')
                    elif kind==life.UPKEEP:lines.append('|upkeep')
                    elif kind==life.TURN:lines.append(f'|turn|{int(actual.callbacks.event_turn[row,index])}')
                    else:self.fail('unexpected raw '+str(kind))
                _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(f.catalog));reference=np.array([e.words() for packet in packets for e in packet.events],np.int32)
                np.testing.assert_array_equal(f.f.words(result.projection,row),reference)
        _,ok=ex.join_consumed_owners(*closed,tables=f.account.device);self.assertTrue(bool(ok.all()))
