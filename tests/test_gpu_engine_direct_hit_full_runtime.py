"""Actual critical callback -> original FULL words/both owners, no coverage claim."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_direct_hit_controls as hit
from gpu_engine import engine_full_v4_event_device as full, engine_lifecycle_callbacks as life
from gpu_engine import public_wire, public_inputs as inputs, transition, v4_encoder as v4
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
        lower=full.causal.project_frame(stream.causal,frame,self.compiled.causal.device)
        self.assertEqual(lower.error.tolist(),[0,0])
        rows=[];next_wire=[]
        statuses=('',' slp',' par',' brn',' frz',' psn',' tox')
        for row in range(2):
            lines=[];active=dict(wire[row].active_idents())
            for i,w in enumerate(np.asarray(lower.chunk.words[row])[np.asarray(lower.chunk.valid[row])]):
                kind,_,turn,_,actor,_,target,_,*p=map(int,w)
                raw=np.asarray(frame.callbacks.words[row,i]);rawkind=int(raw[0])
                if kind==inputs.Kind.SWITCH:
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
                    lines.append(f'|-damage|{active[target]}|{p[1]}/{p[2]}'+statuses[p[3]])
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

    def test_actual_hit_full_words_both_owner_leaves_and_matching_hp_faults(self):
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
            expected,wire=self.reference(init.projection.state,frame,wire)
            expected_public,error=v4.fold_chunk(init.public,expected,self.f.catalog)
            self.assertEqual(error.tolist(),[0,0])
            result=self.fast(init.projection.state,frame,init.public)
            self.assertEqual(result.projection.error.tolist(),[0,0])
            self.same(result.projection.chunk,expected);self.same(result.public,expected_public)
            self.assertFalse(bool(result.projection.full_line_consumers_complete.any()))
            self.assertFalse(bool(result.projection.B4_qualified.any()))
            # Literal enum-valid faults cannot borrow a later HP write or old
            # observed HP. A bad row rolls back the whole real FULL frame.
            for column,index,value in ((6,at,int(raw[0,at,6])-1),
                    (5,at+1,int(raw[0,at+1,5])+1),(5,at,2)):
                tape=frame.callbacks._replace(words=frame.callbacks.words.at[0,index,column].set(value))
                bad=self.fast(init.projection.state,frame._replace(callbacks=tape),init.public)
                self.assertNotEqual(int(bad.projection.error[0]),0)
                self.assertEqual(int(bad.projection.error[1]),0)
                self.same(jax.tree.map(lambda x:x[0],bad.public),jax.tree.map(lambda x:x[0],init.public))
                self.same(jax.tree.map(lambda x:x[1],bad.public),jax.tree.map(lambda x:x[1],result.public))

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
