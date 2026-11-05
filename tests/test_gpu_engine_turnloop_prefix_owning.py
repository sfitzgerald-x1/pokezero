"""Prospective tiny components only; no consumed scientific root is replayed."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from dataclasses import replace

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_lifecycle_callbacks as life, engine_execution_completeness as execution
from gpu_engine import engine_source_frame_witness as witness, public_wire, v4_encoder
from gpu_engine import engine_full_v4_event_device as full
import test_gpu_engine_source_frame_witness as fixtures
from test_gpu_engine_gameplay_stream_comparison import state as literal_state


class TurnLoopPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.SourceFrameWitnessTests.setUpClass.__func__(cls)

    actual=fixtures.SourceFrameWitnessTests.actual
    same=fixtures.SourceFrameWitnessTests.same

    def tape(self,b=8,active=None):
        r=self.runtime;streams=j.arange(9801,9801+b,dtype=j.int32)
        cursor=life.register(streams,catalog=r['request']['lifecycle'],
            expected_binding=r['request']['lifecycle'].binding,
            expected_engine_binding=r['request']['rules'].device.engine_binding)
        cursor=cursor._replace(turn=j.ones(b,j.int32),initialized=j.ones(b,bool),
            physical_active=j.zeros((b,2),j.int32),public_slot=j.zeros((b,2),j.int32),
            encounter=j.ones((b,2),j.int32),established=j.ones((b,2),bool),
            physical_to_public=j.full((b,2,6),-1,j.int32).at[:,:,0].set(0),next_slot=j.ones((b,2),j.int32))
        tape=life._tape(cursor,r['request']['lifecycle'],64)
        ledger=execution.initialize(streams,compiled=r['execution_compiled'],capacity=32,receipt_capacity=64)
        ledger=ledger._replace(frame=j.zeros(b,j.int32))
        mask=j.ones(b,bool) if active is None else active
        return execution.begin_on_tape(tape,ledger,r['execution_compiled'].device,active=mask)

    def test_current_open_frame_b8_inactive_closed_invalid_and_prior_refusal(self):
        active=j.array([1,0,1,1,1,1,1,1],bool);tape=self.tape(active=active)
        state=literal_state(8)
        live=j.array([1,1,0,1,1,1,1,1],bool)
        invalid=j.array([0,0,0,1,0,0,0,0],bool)
        tape=tape._replace(refusal=tape.refusal|j.array([0,0,0,0,256,0,0,0],j.int32))
        entered=jax.jit(life.begin)(tape,state,live,invalid)
        self.assertEqual(entered.count.tolist(),[1,0,0,0,1,1,1,1])
        self.assertEqual(entered.words[0,0].tolist(),[35,-1,-1,-1,-1,0,0,0,0,0])
        self.assertEqual(entered.execution.ledger.stage[:,0].tolist(),[2,0,0,0,2,2,2,2])
        self.assertTrue(int(entered.refusal[4])&256)
        self.assertEqual(entered.execution.tables.engine_binding.shape,(8,))
        diagnostic=tape._replace(execution=None)
        self.assertEqual(life.begin(diagnostic,state,live,invalid).count.tolist(),[0]*8)
        constructor=tape._replace(cursor=tape.cursor._replace(initialized=j.zeros(8,bool)))
        self.assertEqual(life.begin(constructor,state,live,invalid).count.tolist(),[0]*8)

    def test_prefix_relocation_generation_and_missing_capture_are_sticky(self):
        tape=self.tape(2);state=literal_state(2);active=j.ones(2,bool)
        entered=life.begin(tape,state,active,j.zeros(2,bool))
        moved=tape._replace(count=j.ones(2,j.int32))
        bad=life.begin(moved,state,active,j.zeros(2,bool))
        self.assertTrue(bool((bad.execution.ledger.refusal&execution.RECEIPT).all()))
        self.assertEqual(bad.execution.ledger.stage[:,0].tolist(),[1,1])
        self.assertEqual(entered.event_turn[:,0].tolist(),[1,1])
        for row in (0,1):
            self.assertEqual(entered.event_generation[row,0].tolist(),tape.cursor.generation[row].tolist())
        opened,ticket=execution.open_attempt(tape,execution.Site.TURNLOOP_PREFIX,0,active=active)
        appended=life._clock(opened,life.TURNLOOP_PREFIX,active)
        changed=appended._replace(event_generation=appended.event_generation.at[0,0,0].add(1))
        rejected=execution.capture_turnloop_prefix(changed,ticket,requested=active)
        self.assertEqual(rejected.execution.ledger.stage[:,0].tolist(),[1,2])
        self.assertTrue(int(rejected.execution.ledger.refusal[0])&execution.RECEIPT)

    def wire_before(self,sequence,turn,phase=0):
        kwargs=v4_encoder.wire_kwargs(self.runtime['catalog'])
        name=self.runtime['catalog'].species.names[1]
        before,_=public_wire.project_lines(public_wire.WireState(),[
            f'|switch|p1a: {name}|{name}, L80, M|100/100',
            f'|switch|p2a: {name}|{name}, L80, M|100/100','|turn|1'],**kwargs)
        return replace(before,next_sequence=sequence,turn=turn,phase=phase)

    def wire_before_stream(self,stream):
        return self.wire_before(int(stream.sequence[0])+1,
            int(stream.causal.cursor.turn[0]),int(stream.causal.cursor.phase[0]))

    def test_original_twelve_controls_timestamp_neutrality_and_continuation(self):
        kwargs=v4_encoder.wire_kwargs(self.runtime['catalog'])
        before=self.wire_before(19,1)
        after,packets=public_wire.project_lines(before,['|','|t:|1793451704'],**kwargs)
        actual=[event.words().tolist() for packet in packets for event in packet.events]
        self.assertEqual(len(actual),12)
        self.assertEqual([x[0]for x in actual],[21,30,32,2,31,29,21,30,32,28,31,29])
        self.assertEqual(actual[0],[21,19,1,0,-1,-1,-1,-1,256,0,0,1,0,0,0,0])
        middle,first=public_wire.project_line(before,'|',**kwargs)
        continued,last=public_wire.project_line(middle,'|t:|1',**kwargs)
        self.assertEqual(after,continued)
        self.assertEqual(actual,[e.words().tolist() for packet in (first,last)for e in packet.events])
        table=self.runtime['public'].device
        key=int(table.kinds[28]);size=int(table.length[key])
        template=np.array(table.templates[key,:size]);template[:,1]=np.arange(19,19+size)
        self.assertEqual(template.tolist(),actual)
        self.assertEqual(int(table.main[key]),9)
        registered=full.register(j.array([51,52],j.int32),j.array([0,1],j.int32),compiled=self.runtime['public'])
        baseline=self.wire_before_stream(registered)
        self.assertEqual((baseline.next_sequence,baseline.turn,baseline.phase),(1,0,0))

    def test_physical_default_sources_and_all_source_pins(self):
        folder=Path(life.__file__).parent
        for name in ('transition.py','entry.py','mechanics.py','state.py'):
            original=subprocess.check_output(['git','show','93c88ae5:'+str(Path('gpu_engine')/name)])
            self.assertEqual((folder/name).read_bytes(),original)
        for name,digest in witness.SOURCE_PINS.items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),digest,name)

    def test_actual_two_owner_full_drain_then_next_frame_continuation(self):
        built,core,frame,streams,publics=self.actual()
        self.assertFalse(bool((frame.callbacks.words[:,:,0]==life.TURNLOOP_PREFIX).any()))
        carry=witness.initialize_from_constructed(self.compiled,built)
        initial=self.fast(carry,core,frame,streams,publics,j.ones(2,bool))
        self.assertEqual(initial.report.frame_inputs_complete.tolist(),[True,True])
        core=core._replace(execution=initial.ledger)
        import test_gpu_engine_kernel_risk_runtime as kernel_fixture
        chance=kernel_fixture.KernelRiskDriverPreparationTests.base().chance()
        streams=tuple(x.projection.state for x in initial.owners);publics=tuple(x.public for x in initial.owners)
        for number in (1,2):
            actual=self.runtime['frame'](core,j.zeros((2,2),j.int32),chance,j.ones(2,bool))
            tape=actual.callbacks
            self.assertEqual(tape.words[:,0,0].tolist(),[35,35])
            self.assertEqual(tape.execution.ledger.site[:,0].tolist(),[21,21])
            self.assertEqual(tape.execution.ledger.stage[:,0].tolist(),[2,2])
            after=core._replace(engine=actual.state,cursor=actual.cursor,execution=tape.execution.ledger,
                risks=tape.execution.risks.carry,frame_number=j.full((2,),number,j.int32),weather_profile=actual.weather_profile)
            step=self.fast(initial.carry if number==1 else step.carry,after,actual,streams,publics,j.ones(2,bool))
            self.assertEqual(step.report.frame_inputs_complete.tolist(),[True,True],msg=json.dumps(dict(
                frame=number,refusal=step.report.refusal.tolist(),tape=tape.refusal.tolist(),risk=after.risks.risk.tolist(),
                ledger=step.ledger.refusal.tolist(),public=[x.projection.error.tolist() for x in step.owners])))
            self.assertEqual(step.ledger.stage[:,0].tolist(),[3,3])
            if number==1:
                # This same genuine frame must still roll BOTH owner folds
                # back on a one-row prefix/claim mismatch or duplicated owner.
                damaged=actual._replace(callbacks=tape._replace(words=tape.words.at[0,0,0].set(life.ACTION_END)))
                refused=self.fast(initial.carry,after,damaged,streams,publics,j.ones(2,bool))
                self.assertEqual(refused.report.frame_inputs_complete.tolist(),[False,True])
                duplicate=self.fast(initial.carry,after,actual,(streams[0],streams[0]),publics,j.ones(2,bool))
                self.assertEqual(duplicate.report.frame_inputs_complete.tolist(),[False,False])
                for side,owner in enumerate(refused.owners):
                    self.assertEqual(int(owner.projection.state.sequence[0]),int(streams[side].sequence[0]))
                    self.assertEqual(int(owner.public.cursor.sequence[0]),int(publics[side].cursor.sequence[0]))
            for side,owner in enumerate(step.owners):
                before=self.wire_before_stream(streams[side])
                _,packets=public_wire.project_lines(before,['|','|t:|1'],**v4_encoder.wire_kwargs(self.runtime['catalog']))
                expected=[e.words().tolist()for packet in packets for e in packet.events]
                words=np.array(owner.projection.frames.words)[:,0]
                valid=np.array(owner.projection.frames.valid)[:,0]
                self.assertEqual(words[valid][:12].tolist(),expected)
            streams=tuple(x.projection.state for x in step.owners);publics=tuple(x.public for x in step.owners)
            core=after._replace(execution=step.ledger)


if __name__=='__main__':unittest.main()
