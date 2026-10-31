"""Tiny actual setter/original writer components, not sessions or games."""
import ast
from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import secondary,engine_secondary_status_callbacks as callbacks
from gpu_engine import engine_direct_hit_controls as direct,engine_execution_completeness as ex
from gpu_engine import engine_lifecycle_callbacks as life,engine_request_source as request
from gpu_engine import engine_full_v4_event_device as full,engine_public_events as public
from gpu_engine import public_wire,v4_encoder,mechanics as m
from gpu_engine import engine_source_frame_witness as witness
from test_gpu_engine_gameplay_stream_comparison import state as literal_state
import test_gpu_engine_no_damage_owning as fixtures
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected


class SecondaryStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NoDamageOwningTests.setUpClass();cls.f=fixtures.NoDamageOwningTests
        cls.rules=request._checked(cls.f.req)['rules']
        cls.compiled=direct.compile_direct_hit_controls(engine_catalog=cls.f.producer,metadata=cls.f.metadata,source_rules=cls.rules)
        cls.account=cls.f.runtime['execution_compiled']
        cls.context=direct.RegisteredContext(cls.compiled.device,cls.account.device)
        cls.kw=cls.f.kw
        cls.fast=staticmethod(jax.jit(lambda state,tape,move,roll,table:
            secondary.apply(state,j.zeros(len(tape.count),j.int32),move,roll,table,eligible=j.ones(len(tape.count),bool),
                item_table=cls.f.table.items,_status_trace=tape,_secondary_status_context=cls.context)))
        cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.f.runtime['public'].device,p,cls.f.catalog)))

    def component(self,b=2):
        state=jax.tree.map(j.asarray,literal_state(b))
        abi=self.f.metadata['vocabulary']['abilities'].index('pressure')
        state=state._replace(species=j.ones_like(state.species),ability=j.full_like(state.ability,abi),base_ability=j.full_like(state.ability,abi),
            gender=j.ones_like(state.gender),types=j.ones_like(state.types))
        cursor=life.register(j.arange(25001,25001+b,dtype=j.int32),**self.kw)
        before=life.snapshot(cursor);tape=life._tape(cursor,self.f.producer,64)
        for side in (0,1):
            tape=life._reveal(tape,state,j.full((b,),side,j.int32),j.full((b,),-1,j.int32),j.zeros(b,j.int32),
                eligible=j.ones(b,bool),replacement=j.zeros(b,bool),baton=j.zeros(b,bool),initial=True)
        tape=tape._replace(cursor=tape.cursor._replace(initialized=j.ones(b,bool)))
        tape=life._turn(tape,j.ones(b,bool))
        initial=life.LifecycleResult(state,tape.cursor,tape,before,life.snapshot(tape.cursor),j.zeros(b,bool),j.zeros(b,j.int32))
        observers=[]
        for side in (0,1):
            stream=full.register(tape.stream_id,j.full((b,),side,j.int32),compiled=self.f.runtime['public'])
            pub=v4_encoder.initialize(self.f.catalog,stream_ids=tape.stream_id,owner_sides=j.full((b,),side,j.int32))
            folded=self.fold(stream,initial,pub)
            self.assertEqual(folded.projection.error.tolist(),[0]*b)
            observers.append((folded.projection.state,folded.public))
        empty=life._tape(tape.cursor,self.f.producer,64)
        ledger=ex.initialize(tape.stream_id,compiled=self.account)
        empty=ex.begin_on_tape(empty,ledger,self.account.device,active=j.ones(b,bool))
        move=j.full((b,),self.f.metadata['vocabulary']['moves'].index('thunderbolt'),j.int32)
        empty=public.move(empty,j.zeros(b,j.int32),j.zeros(b,j.int32),j.zeros(b,j.int32),move,eligible=j.ones(b,bool))
        return state,empty,move,observers

    def test_original_compiler_and_default_numeric_ast(self):
        data=self.compiled.device.secondary_status;self.assertIs(direct.check_compiled(self.compiled),self.compiled.device)
        thunder=self.f.metadata['vocabulary']['moves'].index('thunderbolt')
        fire=self.f.metadata['vocabulary']['moves'].index('flamethrower')
        self.assertTrue(bool(data.candidates[thunder]));self.assertTrue(bool(data.candidates[fire]))
        self.assertEqual(int(data.status[thunder]),m.PAR);self.assertEqual(int(data.status[fire]),m.BRN)
        legacy=direct.compile_direct_hit_controls(engine_catalog=self.f.producer)
        self.assertIsNone(legacy.device.secondary_status)
        with self.assertRaises(ValueError):direct.compile_direct_hit_controls(engine_catalog=self.f.producer,metadata=self.f.metadata)
        for name in ('secondary.py','transition.py'):
            before=ast.parse(subprocess.check_output(['git','show','27fa443a:gpu_engine/'+name],text=True))
            after=ast.parse(Path(secondary.__file__).with_name(name).read_text())
            self.assertEqual(dump_default_program(before),dump_default_program(after),name)
            if name=='transition.py':assert_hp_rng_mutations_detected(self,after,dump_default_program(before))

    def test_empty_secondary_context_witness_path_remains_defined(self):
        cursor=life.register(j.array([25001],j.int32),**self.kw)
        tape=life._tape(cursor,self.f.producer,64)
        ledger=ex.initialize(cursor.stream_id,compiled=self.account)
        tape=tape._replace(execution=ex.Context(ledger,self.account.device))
        tables=witness.check_compiled(self.f.runtime['frame_witness_compiled'])
        result=jax.jit(witness._raw_complete)(tape,tables)
        self.assertEqual(result.shape,(1,));self.assertEqual(result.dtype,j.bool_)
        # Raw emptiness is not complete source/input admission.
        self.assertFalse(bool(ledger.opened.any()))

    def test_actual_setter_both_original_consumers_and_plain_writer(self):
        state,tape,move,observers=self.component()
        numeric,recorded=self.fast(state,tape,move,j.zeros(2,j.int32),self.f.table.secondary)
        expected=secondary.apply(state,j.zeros(2,j.int32),move,j.zeros(2,j.int32),self.f.table.secondary,
            eligible=j.ones(2,bool),item_table=self.f.table.items)
        for a,b in zip(numeric,expected):np.testing.assert_array_equal(a,b)
        self.assertEqual(numeric.status[:,1,0].tolist(),[m.PAR]*2)
        self.assertEqual(recorded.words[:,:3,0].tolist(),[[1,9,36]]*2)
        self.assertEqual(recorded.execution.ledger.stage[:,0].tolist(),[2,2])
        self.assertEqual(recorded.refusal.tolist(),[1,1])
        self.assertTrue(bool(callbacks.owned_pair(recorded,2).all()))
        frame=life.LifecycleResult(numeric,recorded.cursor,recorded,life.snapshot(tape.cursor),life.snapshot(recorded.cursor),j.zeros(2,bool),j.zeros(2,j.int32))
        for side,(stream,pub) in enumerate(observers):
            ledger,folded=ex.consume_frame_device(recorded.execution.ledger,frame,stream,pub,active=j.ones(2,bool),
                tables=self.account.device,adapter_tables=self.f.runtime['public'].device,catalog=self.f.catalog)
            self.assertEqual(folded.projection.error.tolist(),[0,0]);self.assertEqual(ledger.stage[:,0].tolist(),[3,3])
            self.assertTrue(bool(ex.report(ledger,tables=self.account.device).scoped_drained.all()))
            words=np.asarray(folded.projection.frames.words)[:,0][np.asarray(folded.projection.frames.valid)[:,0]]
            self.assertEqual(int(np.count_nonzero(words[:,0]==11)),1)
            # Complete original public writer, including actual MOVE before STATUS.
            name=self.f.catalog.species.names[int(self.f.runtime['public'].device.causal.species[1])]
            lines=[f'|switch|p1a: {name}|{name}, L80, M|100/100',f'|switch|p2a: {name}|{name}, L80, M|100/100','|turn|1']
            wire,_=public_wire.project_lines(public_wire.WireState(),lines,**v4_encoder.wire_kwargs(self.f.catalog))
            _,projections=public_wire.project_lines(wire,[f'|move|p1a: {name}|Thunderbolt|p2a: {name}',f'|-status|p2a: {name}|par'],**v4_encoder.wire_kwargs(self.f.catalog))
            wanted=np.asarray([e.words()for p in projections for e in p.events]);np.testing.assert_array_equal(words,wanted)

    def test_mixed_refusal_install_drift_and_old_bits(self):
        state,tape,move,_=self.component()
        state=state._replace(status=state.status.at[1,1,0].set(m.PSN))
        tape=tape._replace(refusal=tape.refusal.at[0].set(129))
        numeric,recorded=self.fast(state,tape,move,j.zeros(2,j.int32),self.f.table.secondary)
        self.assertEqual(recorded.count.tolist(),[3,1]);self.assertTrue(int(recorded.refusal[0])&128)
        self.assertEqual(recorded.execution.ledger.count.tolist(),[1,0])
        altered=self.f.table.secondary._replace(chance=self.f.table.secondary.chance.at[int(move[0])].add(1))
        _,bad=self.fast(state,tape,move,j.zeros(2,j.int32),altered)
        self.assertEqual(bad.count.tolist(),[1,1]);self.assertTrue(int(bad.refusal[0])&128)


if __name__=='__main__':unittest.main()
