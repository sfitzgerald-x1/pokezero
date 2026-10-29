import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class NonprocStreamGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_nonproc_stream import NonprocStreamTests
        NonprocStreamTests.setUpClass();cls.f=NonprocStreamTests.f;cls.lines=NonprocStreamTests.lines
        cls.j,cls.jax=cls.f.j,cls.f.jax

    def pack(self,streams):
        import numpy as np
        from dataclasses import replace
        from gpu_engine import public_inputs
        p=self.f.projections(self.lines[:1])[0]
        bindings={name:np.asarray(getattr(p,name),np.uint32) for name in (
            'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding',
            'active_binding','damage_binding','type_binding','sleep_binding','toxic_binding','tendency_binding')}
        streams=[[replace(event,sequence=i+1) for i,event in enumerate(events)] for events in streams]
        return list(public_inputs.pack_chunks(np.arange(1,len(streams)+1,dtype=np.int32),streams,
            source_binding=self.f.catalog.source.device.binding,**bindings))

    def test_missing_duplicate_reordered_and_inconsistent_facts_roll_back_mixed_batch(self):
        import numpy as np
        from dataclasses import replace
        from gpu_engine import public_inputs
        events=[e for p in self.f.projections([*self.lines,'|upkeep']) for e in p.events]
        end=max(i for i,e in enumerate(events) if e.kind==public_inputs.Kind.NONPROC)
        hp=next(i for i,e in enumerate(events) if e.kind==public_inputs.Kind.NONPROC and e.payload[0]==2)
        changed=list(events);d=changed[hp].payload;changed[hp]=replace(changed[hp],payload=(*d[:6],d[6]^4,d[7]))
        false_hp=list(events);d=false_hp[hp].payload;false_hp[hp]=replace(false_hp[hp],payload=(*d[:4],d[4]+1,*d[5:]))
        late=list(events);late[end],late[end+1]=late[end+1],late[end]
        streams=[events,events[:end]+events[end+1:],events[:end]+[events[end]]+events[end:],changed,false_hp,late,events,events]
        chunks=self.pack(streams);self.assertEqual(len(chunks),1)
        initial=self.f.e.initialize(self.f.catalog,stream_ids=self.j.arange(1,9,dtype=self.j.int32))
        out,error=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.f.catalog))(initial,chunks[0])
        self.assertEqual((error==0).tolist(),[True,False,False,False,False,False,True,True])
        for a,b in zip(self.jax.tree.leaves(out),self.jax.tree.leaves(initial)):
            if a.ndim and a.shape[0]==8:np.testing.assert_array_equal(a[1:6],b[1:6])
            else:np.testing.assert_array_equal(a,b)
        self.assertEqual(out.cursor.upkeeps.tolist(),[1,0,0,0,0,0,1,1])

    def test_ledger_binding_identity_and_empty_chunk_cannot_bypass_admission(self):
        import numpy as np
        from dataclasses import replace
        from gpu_engine import public_inputs
        events=[e for p in self.f.projections(self.lines[:2]) for e in p.events]
        chunk=self.pack([events]*8)[0]
        state=self.f.e.initialize(self.f.catalog,stream_ids=self.j.arange(1,9,dtype=self.j.int32))
        ledger=state.context.nonproc._replace(encounter=state.context.nonproc.encounter.at[1,0,0].set(1))
        context=state.context._replace(nonproc=ledger,nonproc_binding=state.context.nonproc_binding.at[0,0].set(
            state.context.nonproc_binding[0,0]^self.j.uint32(1)))
        state=state._replace(context=context)
        empty=chunk._replace(valid=self.j.zeros_like(chunk.valid),more=self.j.zeros_like(chunk.more))
        step=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.f.catalog))
        for packet in (empty,chunk):
            out,error=step(state,packet);self.assertEqual(error.tolist(),[int(public_inputs.Error.SOURCE)]*2+[0]*6)
            np.testing.assert_array_equal(out.cursor.sequence[:2],state.cursor.sequence[:2])
            np.testing.assert_array_equal(out.context.nonproc.encounter[:2],state.context.nonproc.encounter[:2])
        for catalog,bad in ((self.f.catalog,state._replace(context=context._replace(nonproc=None))),
                (replace(self.f.catalog,context=self.f.catalog.context._replace(nonproc=None)),state)):
            out,error=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,catalog))(bad,empty)
            self.assertTrue(bool((error!=0).all()))

    def test_real64_event_cut_after_exclusions_preserves_pending_upkeep_and_global_cache(self):
        import numpy as np
        from gpu_engine import public_inputs,public_refresh_stream
        flatten=lambda lines:[e for p in self.f.projections(lines) for e in p.events]
        base=len(flatten(self.lines));last=self.f.projections([*self.lines,'|upkeep'])[-1].events
        offset=next(i for i,e in enumerate(last) if e.kind==public_inputs.Kind.NONPROC)
        pad='|-item|p1a: Snorlax|Choice Band|[from] item: Leftovers'
        n=len(self.f.projections([*self.lines,pad])[-1].events);m=len(self.f.projections([*self.lines,'|'])[-1].events)
        choices=[(n*a+m*b,a,b) for a in range(64) for b in range(64) if (base+n*a+m*b+offset)%64==63]
        self.assertTrue(choices);_,a,b=min(choices)
        selected=flatten([*self.lines,*([pad]*a),*(['|']*b),'|upkeep'])
        self.assertEqual(max(i for i,e in enumerate(selected) if e.kind==public_inputs.Kind.NONPROC)%64,63)
        chunks=self.pack([selected]*8);self.assertGreaterEqual(len(chunks),2)
        state=self.f.e.initialize(self.f.catalog,stream_ids=self.j.arange(1,9,dtype=self.j.int32))
        step=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.f.catalog));saw_cut=False
        for chunk in chunks:
            state,error=step(state,chunk);self.assertEqual(error.tolist(),[0]*8)
            if int(chunk.words[0,63,0])==int(public_inputs.Kind.NONPROC):
                saw_cut=True;self.assertEqual(state.context.refresh.family.tolist(),[public_refresh_stream.UPKEEP]*8)
                self.assertEqual(state.context.refresh.nonproc_seen.tolist(),[1]*8)
                self.assertFalse(bool(state.context.nonproc.maximum.any()))
                np.testing.assert_array_equal(state.context.refresh.controller.visible.source_binding,self.f.source.device.binding)
        self.assertTrue(saw_cut);self.assertEqual(state.cursor.upkeeps.tolist(),[1]*8)
        self.assertEqual(state.context.refresh.family.tolist(),[0]*8)

    def test_same_hp_other_side_cannot_redirect_hp_berry_or_arrival_facts(self):
        import numpy as np
        from dataclasses import replace
        from gpu_engine import public_inputs
        prefix=self.lines[:2];pairs=[]
        for raw,operation in (('|-damage|p1a: Snorlax|100/100',2),
                ('|-enditem|p1a: Snorlax|Salac Berry|[eat]',0),
                ('|switch|p1a: Snorlax|Snorlax, L80|100/100',1)):
            events=[e for p in self.f.projections([*prefix,raw]) for e in p.events]
            index=max(i for i,e in enumerate(events) if e.kind==public_inputs.Kind.NONPROC)
            self.assertEqual(events[index].payload[0],operation)
            bad=list(events);d=bad[index].payload
            bad[index]=replace(bad[index],payload=(d[0],1,0,1,*d[4:]))
            pairs.append((bad,events))
        base=[e for p in self.f.projections(prefix) for e in p.events]
        streams=[*(bad for bad,_ in pairs),*(good for _,good in pairs),base,base]
        chunks=self.pack(streams);self.assertEqual(len(chunks),1)
        initial=self.f.e.initialize(self.f.catalog,stream_ids=self.j.arange(1,9,dtype=self.j.int32))
        out,error=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.f.catalog))(initial,chunks[0])
        self.assertEqual((error==0).tolist(),[False]*3+[True]*5)
        for a,b in zip(self.jax.tree.leaves(out),self.jax.tree.leaves(initial)):
            if a.ndim and a.shape[0]==8:np.testing.assert_array_equal(a[:3],b[:3])
            else:np.testing.assert_array_equal(a,b)
