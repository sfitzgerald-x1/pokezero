"""Literal Own Tempo empty-update admission; no games/models/Native/replay."""
from _showdown_root import showdown_root_str
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import os
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import entry,transition,engine_public_events as events
from gpu_engine import engine_lifecycle_callbacks as life,engine_execution_completeness as execution
from gpu_engine import engine_kernel_frame_risk as risk,engine_full_v4_event_device as full
from gpu_engine import engine_public_event_device as base,v4_encoder as v4,public_wire
import test_gpu_engine_pressure_entry_owning as fixture
from gpu_engine_default_observer_ast import dump_default_program


class OwnTempoNoOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.PressureEntryOwningTests.setUpClass();cls.f=fixture.PressureEntryOwningTests()
        cls.metadata=deepcopy(cls.f.metadata)
        cls.metadata['vocabulary']['abilities'].append('owntempo')
        cls.own_tempo=cls.metadata['vocabulary']['abilities'].index('owntempo')
        cls.table=entry.from_metadata(cls.metadata);cls.engine=cls.f.engine._replace(entry=cls.table)
        cls.catalog=cls.f.catalog
        cls.producer=events.compile_catalog(cls.metadata,cls.engine,public_moves=cls.catalog.moves,
            dex=cls.catalog.dex,exposure=fixture.engine_fixture.EnginePublicEventsTests.exposure)
        projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,
            expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32))
        cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog)
        cls.account=execution.compile_accounting(cls.adapter)
        cls.risks=risk.compile_risks(cls.producer,cls.account)

    def initial(self,b=2):
        state=self.f.state(b)._replace(done=j.zeros(b,bool),active=j.zeros((b,2),j.int32),entry_pending=j.ones((b,2),bool),
            ability=j.zeros((b,2,6),j.int32).at[:,1,0].set(self.own_tempo),
            base_ability=j.zeros((b,2,6),j.int32).at[:,1,0].set(self.own_tempo),
            stats=j.full((b,2,6,6),142,j.int32).at[:,1,:,5].set(131))
        streams=j.arange(9101,9101+b,dtype=j.int32)
        cursor=life.register(streams,catalog=self.producer,expected_binding=self.producer.binding,
            expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32))
        actual=life.initialize_with_lifecycle(state,self.engine,cursor=cursor,catalog=self.producer,
            expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32),
            tie_coin=j.zeros(b,j.int32),execution_compiled=self.account,kernel_risk_compiled=self.risks)
        self.assertTrue(bool(actual.cursor.initialized.all()))
        return state,actual

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):np.testing.assert_array_equal(x,y)

    def test_original_constructor_empty_confusion_actual16_word_full_both_owners(self):
        state,actual=self.initial();legacy=transition.initialize(state,self.engine,tie_coin=j.zeros(2,j.int32))
        self.same(actual.state,legacy)
        carry=actual.callbacks.execution.risks.carry
        self.assertEqual(carry.risk.tolist(),[0,0])
        self.assertEqual(carry.counts[:,risk.Branch.ENTRY_CONFUSION_CURE-1].tolist(),[2,2])
        self.assertEqual(actual.callbacks.count.tolist(),[3,3])
        self.assertEqual(actual.callbacks.words[0,:3,0].tolist(),[8,8,4])
        self.assertEqual((actual.callbacks.refusal&~events.UNINSTRUMENTED).tolist(),[0,0])
        ledger=actual.callbacks.execution.ledger
        consume=jax.jit(lambda stream,public:execution.consume_frame_device(ledger,actual,stream,public,
            active=j.ones(2,bool),tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog))
        owners=[]
        names=[self.catalog.species.names[int(self.adapter.device.causal.species[i])] for i in (1,2)]
        lines=[f'|switch|p1a: {names[0]}|{names[0]}, L80, M|100/100',
            f'|switch|p2a: {names[1]}|{names[1]}, L80, M|100/100','|turn|1']
        _,packets=public_wire.project_lines(public_wire.WireState(),lines,**v4.wire_kwargs(self.catalog))
        expected=np.asarray([event.words() for packet in packets for event in packet.events],np.int32)
        self.assertEqual(expected.shape[1],16)
        for owner in (0,1):
            stream=full.register(actual.cursor.stream_id,j.full(2,owner,j.int32),compiled=self.adapter)
            public=v4.initialize(self.catalog,stream_ids=actual.cursor.stream_id,owner_sides=j.full(2,owner,j.int32))
            accounted,result=consume(stream,public);owners.append(accounted)
            self.assertEqual(result.projection.error.tolist(),[0,0])
            self.assertFalse(bool(result.projection.B4_qualified.any()))
            for row in (0,1):np.testing.assert_array_equal(self.f.words(result.projection,row),expected)
        _,joined=execution.join_consumed_owners(*owners,tables=self.account.device)
        self.assertEqual(joined.tolist(),[True,True])

    def test_actual_entry_nonzero_negative_inactive_nonown_copied_and_prior_stay_unknown(self):
        _,frame=self.initial(8);state=frame.state;mask=j.array([True]*4+[False]+[True]*3)
        state=state._replace(confusion_counter=state.confusion_counter.at[1,1,0].set(3).at[2,1,0].set(-1)
            .at[4,1,0].set(3).at[5,1,0].set(3),
            ability=state.ability.at[5,1,0].set(0),base_ability=state.base_ability.at[5,1,0].set(0).at[6,1,0].set(0),
            hp=state.hp.at[7,1,0].set(0))
        context=frame.callbacks.execution.risks
        tape=frame.callbacks._replace(execution=frame.callbacks.execution._replace(risks=context._replace(
            carry=context.carry._replace(risk=context.carry.risk.at[3].set(risk.SELF_HP)))))
        run=jax.jit(lambda s,t:entry.enter(s,1,self.table,eligible=mask,hazards=False,_weather_trace=t))
        actual,_,out=run(state,tape);legacy,_=entry.enter(state,1,self.table,eligible=mask,hazards=False)
        self.same(actual,legacy);values=np.asarray(out.execution.risks.carry.risk)
        self.assertEqual(values[0],0)
        self.assertTrue(values[1]&risk.FIELD_VOLATILE);self.assertTrue(values[2]&risk.FIELD_VOLATILE)
        self.assertTrue(values[3]&risk.SELF_HP);self.assertEqual(values[4],0);self.assertEqual(values[5],0)
        self.assertTrue(values[6]&risk.UPSTREAM);self.assertEqual(values[7],0)
        self.assertEqual(actual.confusion_counter[:,1,0].tolist(),[0,0,0,0,3,3,0,0])
        self.assertEqual(out.count.tolist(),tape.count.tolist()) # no invented source line
        self.assertEqual(out.execution.risks.carry.counts[4].tolist(),context.carry.counts[4].tolist())

    def test_effective_original_callback_empty_has_no_line_and_default_AST_unchanged(self):
        root=Path(os.environ.get('POKEZERO_SHOWDOWN_ROOT',showdown_root_str()))
        self.assertEqual(subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),events.SHOWDOWN)
        self.assertTrue((root/'dist/sim/dex.js').is_file(),'pinned compiled Showdown root required')
        script=r'''
const path=require('node:path');const {Dex}=require(path.join(process.argv[1],'dist/sim/dex.js'));
const ability=Dex.mod('gen3').abilities.get('owntempo');const result=[];
for (const confusion of [false,true]) {const lines=[],removed=[];
const pokemon={volatiles:confusion?{confusion:{}}:{},toString:()=> 'p2a: Slowbro',removeVolatile:x=>removed.push(x)};
ability.onUpdate.call({add:(...x)=>lines.push(x.map(String))},pokemon);result.push({lines,removed});}
console.log(JSON.stringify(result));'''
        node=shutil.which('node');self.assertIsNotNone(node,'original callback runtime required')
        result=subprocess.run([node,'-e',script,str(root)],
            text=True,capture_output=True,check=True)
        rows=json.loads(result.stdout)
        self.assertEqual(rows[0],dict(lines=[],removed=[]))
        self.assertEqual(rows[1],dict(lines=[['-activate','p2a: Slowbro','ability: Own Tempo']],removed=['confusion']))
        original=subprocess.check_output(['git','show','b8092a85:gpu_engine/entry.py'],text=True)
        self.assertEqual(dump_default_program(ast.parse(original)),dump_default_program(ast.parse(Path('gpu_engine/entry.py').read_text())))
        self.assertEqual(hashlib.sha256(Path('gpu_engine/transition.py').read_bytes()).hexdigest(),
            hashlib.sha256(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'])).hexdigest())
