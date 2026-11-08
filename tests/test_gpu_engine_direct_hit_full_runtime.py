"""Actual critical callback -> original FULL words/both owners, no coverage claim."""
from dataclasses import replace
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_direct_hit_controls as hit
from gpu_engine import engine_full_v4_event_device as full, engine_lifecycle_callbacks as life
from gpu_engine import public_wire, public_inputs as inputs, transition, v4_encoder as v4
from gpu_engine import engine_retained_encoder_driver as driver, engine_execution_completeness as accounting
import test_gpu_engine_engine_full_v4_event_device as fixture


class DirectHitFullRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FullV4EventDeviceTests.setUpClass()
        cls.f=fixture.FullV4EventDeviceTests.f
        cls.compiled=fixture.FullV4EventDeviceTests.compiled
        cls.hit=hit.compile_direct_hit_controls(engine_catalog=cls.f.lifecycle)
        cls.fast=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.compiled.device,p,cls.f.catalog)))

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def initial(self):
        arrays={k:np.array(v,copy=True) for k,v in self.f.arrays.items()}
        arrays['item'][:]=0  # literal fixture has no unrecorded residual item work
        source=self.f.construct(self.f.make_pool(arrays))
        kw=dict(catalog=self.f.lifecycle,expected_binding=self.f.lifecycle.binding,
            expected_engine_binding=np.array(self.f.lifecycle.engine_binding,np.uint32))
        before=life.register(source.cursor.stream_id,**kw)
        frame=life.LifecycleResult(source.state,source.cursor,source.callbacks,
            life.snapshot(before),source.registry,j.zeros(2,bool),j.zeros(2,j.int32))
        return frame,kw

    def reference(self,stream,frame,wire):
        extra=self.lower_context(frame) if hasattr(self,'lower_context') else {}
        lower=full.causal.project_frame(stream.causal,frame,self.compiled.causal.device,**extra)
        self.assertEqual(lower.error.tolist(),[0,0])
        rows=[];next_wire=[]
        statuses=('',' slp',' par',' brn',' frz',' psn',' tox')
        for row in range(2):
            lines=[];active=dict(wire[row].active_idents())
            for i,w in enumerate(np.asarray(lower.chunk.words[row])[np.asarray(lower.chunk.valid[row])]):
                kind,_,turn,_,actor,_,target,_,*p=map(int,w)
                raw=np.asarray(frame.callbacks.words[row,i]);rawkind=int(raw[0])
                if rawkind==life.TURNLOOP_PREFIX:
                    self.assertEqual(kind,int(inputs.Kind.ABILITY))
                    self.assertEqual(raw.tolist(),[35,-1,-1,-1,-1,0,0,0,0,0])
                    lines.extend(['|','|t:|1'])
                elif kind==inputs.Kind.ABILITY and rawkind==24:
                    # Pressure Start is an owner-only silent protocol reveal.
                    # The peer's semantic envelope deliberately emits no line.
                    if int(stream.causal.owner[row])==target:
                        lines.append(f'|-ability|{active[target]}|Pressure|[silent]')
                elif kind==inputs.Kind.SWITCH:
                    name=self.f.catalog.species.names[p[1]];label=f'p{target+1}a: '+name
                    active[target]=label
                    detail=name+f', L{p[3]}'+('' if p[4]==0 else ', M' if p[4]==1 else ', F')
                    lines.append(f'|switch|{label}|{detail}|{p[5]}/{p[6]}'+statuses[p[7]])
                elif kind==inputs.Kind.MOVE:
                    lines.append(f'|move|{active[actor]}|{self.f.catalog.moves.names[p[1]]}|{active[target]}')
                elif kind==inputs.Kind.NONPROC:
                    label={hit.CRITICAL:'-crit',hit.SUPER_EFFECTIVE:'-supereffective',hit.RESISTED:'-resisted'}[rawkind]
                    lines.append(f'|{label}|{active[int(raw[3])]}')
                elif kind==inputs.Kind.CONDITION:
                    line=f'|-damage|{active[target]}|{p[1]}/{p[2]}'+statuses[p[3]]
                    if rawkind==17:line=f'|-heal|{active[target]}|{p[1]}/{p[2]}'+statuses[p[3]]+'|[from] item: Leftovers'
                    lines.append(line)
                elif kind==inputs.Kind.TURN:lines.append(f'|turn|{turn}')
                elif kind==inputs.Kind.UPKEEP:lines.append('|upkeep')
                elif kind==inputs.Kind.ACTION_CHUNK_END:lines.append('|')
                else:self.fail('unexpected actual receipt '+str(kind))
            new,ps=public_wire.project_lines(wire[row],lines,**v4.wire_kwargs(self.f.catalog))
            rows.append([e for p in ps for e in p.events]);next_wire.append(new)
        ns={n:np.array(getattr(self.compiled.device.namespaces,n)) for n in full._NAMES if n!='source_binding'}
        chunks=list(inputs.pack_chunks(np.array(frame.cursor.stream_id),rows,
            source_binding=np.array(self.f.catalog.source.device.binding),**ns))
        self.assertEqual(len(chunks),1)
        return chunks[0],next_wire

    def test_actual_diagnostic_hit_keeps_unknown_and_full_frame_refused(self):
        initial,kw=self.initial();draws=j.zeros((2,2,12),j.int32).at[:,:,4].set(1).at[0,0,4].set(0)
        chance=transition.ChanceInputs(draws,*[j.zeros((2,2),j.int32) for _ in range(4)],j.zeros(2,j.int32))
        actions=j.array([[0,8],[0,8]],j.int32)
        frame=life.trace_step(initial.state,actions,self.f.table,cursor=initial.cursor,
            chance_inputs=chance,direct_hit_compiled=self.hit,**kw)
        legacy=transition.step_with_chance(initial.state,actions,self.f.table,chance)
        self.same(frame.state,legacy.state)
        raw=np.asarray(frame.callbacks.words);valid=np.arange(64)[None,:]<np.asarray(frame.callbacks.count)[:,None]
        controls=np.flatnonzero(valid[0] & (raw[0,:,0]==hit.CRITICAL));self.assertEqual(len(controls),1)
        at=int(controls[0]);self.assertEqual(int(raw[0,at+1,0]),full.engine.MAIN_HP)
        self.assertEqual(int(raw[0,at,7]),int(raw[0,at+1,5]))
        self.assertFalse(bool(np.any(valid[1] & (raw[1,:,0]==hit.CRITICAL))))
        self.assertTrue(bool((frame.callbacks.refusal & hit.UNKNOWN_DIRECT_HIT).all()))
        for owner in (0,1):
            ids=initial.cursor.stream_id;sides=j.full(2,owner,j.int32)
            stream=full.register(ids,sides,compiled=self.compiled)
            public=v4.initialize(self.f.catalog,stream_ids=ids,owner_sides=sides)
            expected,wire=self.reference(stream,initial,[public_wire.WireState()]*2)
            init=self.fast(stream,initial,public);self.same(init.projection.chunk,expected)
            result=self.fast(init.projection.state,frame,init.public)
            self.assertEqual(result.projection.error.tolist(),[int(inputs.Error.SOURCE)]*2)
            self.same(result.public,init.public)
            self.assertFalse(bool(result.projection.full_line_consumers_complete.any()))
            self.assertFalse(bool(result.projection.B4_qualified.any()))

    def test_registered_actual_span_both_owner_words_and_consumption(self):
        import test_gpu_engine_execution_runtime_plumbing as owning_fixture
        owning_fixture.ExecutionRuntimePlumbingTests.setUpClass()
        base=owning_fixture.ExecutionRuntimePlumbingTests
        original=driver._checked(base.prepared)
        compiled=hit.compile_direct_hit_controls(engine_catalog=original['request']['lifecycle'])
        prepared=driver.prepare(original['request_compiled'],original['own_compiled'],
            public_mode='retained-full',leftovers_compiled=original['leftovers_compiled'],
            leftovers_mode='retained',account_source_sites=True,direct_hit_compiled=compiled,
            direct_hit_mode='kernel-event',track_kernel_risks=True,
            fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(7,j.int32)))
        runtime=driver._checked(prepared);self.compiled=runtime['public']
        self.f=SimpleNamespace(catalog=runtime['catalog'])
        fixture_instance=base();session=fixture_instance.construct(prepared=prepared);before=session.state
        self.lower_context=lambda frame:dict(weather_hp=(
            before.core.weather_profile if frame.weather_profile is None else frame.weather_profile,
            runtime['speed_profile'].device),leftovers=runtime['leftovers_context'].data)
        actual_constructor=driver._session(session)['constructed']
        catalog=runtime['request']['lifecycle']
        cursor=life.register(actual_constructor.cursor.stream_id,catalog=catalog,
            expected_binding=catalog.binding,expected_engine_binding=np.asarray(catalog.engine_binding,np.uint32))
        initial=life.LifecycleResult(actual_constructor.state,actual_constructor.cursor,actual_constructor.callbacks,
            life.snapshot(cursor),actual_constructor.registry,j.zeros(2,bool),j.zeros(2,j.int32))
        wires=[]
        for owner in (0,1):
            observer=full.register(cursor.stream_id,j.full(2,owner,j.int32),compiled=self.compiled)
            _,wire=self.reference(observer,initial,[public_wire.WireState()]*2);wires.append(wire)
        chance=fixture_instance.chance()._replace(draws=fixture_instance.chance().draws.at[0,0,4].set(0))
        actions=j.array([[1,8],[1,8]],j.int32);frames=[];actual=runtime['frame']
        def capture(*args):
            frame=actual(*args);frames.append(frame);return frame
        runtime['frame']=capture
        try:result=driver.advance(session,actions,chance_inputs=chance)
        finally:runtime['frame']=actual
        self.assertEqual(len(frames),1);frame=frames[0]
        legacy=transition.step_with_chance(before.core.engine,actions,runtime['request']['engine_table'],chance)
        self.same(result.state.core.engine,legacy.state)
        self.assertEqual(result.state.core.admitted.tolist(),[True,True])
        self.assertEqual(result.state.core.risks.risk.tolist(),[0,0])
        ledger=result.state.core.execution
        self.assertTrue(bool(accounting.report(ledger,tables=runtime['execution_compiled'].device).scoped_drained.all()))
        direct=(ledger.site==int(accounting.Site.DIRECT_HIT))&(ledger.serial>0)
        self.assertEqual(direct.sum(axis=1).tolist(),[1,1])
        self.assertTrue(bool(j.all(j.where(direct,ledger.stage==3,True))))
        self.assertEqual((frame.callbacks.refusal & hit.UNKNOWN_DIRECT_HIT).tolist(),[0,0])
        for owner,(old,new) in enumerate(zip(before.observers,result.state.observers)):
            expected,_=self.reference(old.projection,frame,wires[owner])
            expected_public,error=v4.fold_chunk(old.retained,expected,self.f.catalog)
            self.assertEqual(error.tolist(),[0,0]);self.same(new.retained,expected_public)
            self.assertEqual(new.retained_healthy.tolist(),[True,True])
            self.assertFalse(bool(result.owners[owner].scored.any()))
            self.assertFalse(bool(result.owners[owner].B4_qualified.any()))
        fault_session=fixture_instance.construct(prepared=prepared);prior=fault_session.state
        original_public=runtime['public_fast']
        def corrupt_second_owner(observer,frame,profile,active):
            if int(observer.owner[0])==1:
                words=np.asarray(frame.callbacks.words[0]);where=np.flatnonzero(words[:,0]==hit.CRITICAL)
                self.assertEqual(len(where),1)
                tape=frame.callbacks._replace(words=frame.callbacks.words.at[0,int(where[0]),6].add(1))
                frame=frame._replace(callbacks=tape)
            return original_public(observer,frame,profile,active)
        runtime['public_fast']=corrupt_second_owner
        try:failed=driver.advance(fault_session,actions,chance_inputs=chance)
        finally:runtime['public_fast']=original_public
        self.assertEqual(failed.state.core.admitted.tolist(),[False,True])
        for old,new,good in zip(prior.observers,failed.state.observers,result.state.observers):
            self.same(jax.tree.map(lambda x:x[0],new.retained),jax.tree.map(lambda x:x[0],old.retained))
            self.same(jax.tree.map(lambda x:x[1],new.retained),jax.tree.map(lambda x:x[1],good.retained))
        slots=failed.state.core.execution.site==int(accounting.Site.DIRECT_HIT)
        self.assertEqual(failed.state.core.execution.stage[slots].tolist(),[2,3])

    def test_original_helper_and_fresh_outer_jit_guards(self):
        initial,kw=self.initial();actions=j.full((2,2),8,j.int32)
        call=lambda:life.trace_step(initial.state,actions,self.f.table,cursor=initial.cursor,
            direct_hit_compiled=self.hit,**kw)
        with self.assertRaisesRegex(ValueError,'outside outer JIT'):jax.jit(lambda:call())()
        with self.assertRaisesRegex(ValueError,'outside outer JIT'):jax.jit(lambda unused:call())(j.array(0))
        with self.assertRaises(ValueError):hit.check_compiled(replace(self.hit))
        with patch.object(hit,'check_compiled',side_effect=ValueError('changed original helper')):
            with self.assertRaisesRegex(ValueError,'changed original helper'):call()


if __name__=='__main__':unittest.main()
