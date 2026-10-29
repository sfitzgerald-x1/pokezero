"""Atomic typed item frame admission and transport continuation."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class ItemStreamGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_item_stream import ItemStreamTests
        ItemStreamTests.setUpClass();cls.f=ItemStreamTests.f;cls.j,cls.jax=cls.f.j,cls.f.jax

    def pack(self,events,batch=1):
        import numpy as np
        from dataclasses import replace
        from gpu_engine import public_inputs
        p=self.f.projections([self.f.public.prefix[0]])[0]
        bindings={name:np.asarray(getattr(p,name),np.uint32) for name in (
            'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding',
            'active_binding','damage_binding','type_binding','sleep_binding','toxic_binding','tendency_binding')}
        events=[replace(e,sequence=i+1) for i,e in enumerate(events)]
        return list(public_inputs.pack_chunks(np.arange(1,batch+1,dtype=np.int32),[events]*batch,
            source_binding=self.f.catalog.source.device.binding,**bindings))

    def test_missing_duplicate_bad_eat_proof_and_late_event_rollback(self):
        import numpy as np
        from dataclasses import replace
        from gpu_engine import public_inputs
        f=self.f;lines=[*f.public.prefix,'|-item|p1a: Snorlax|Leftovers']
        events=[e for p in f.projections(lines) for e in p.events]
        index=next(i for i,e in enumerate(events) if e.kind==public_inputs.Kind.ITEM)
        opener=max(i for i in range(index) if events[i].kind==public_inputs.Kind.RAW_BEGIN)
        bad_eat=list(events);payload=bad_eat[opener].payload
        bad_eat[opener]=replace(bad_eat[opener],payload=(payload[0]|16,*payload[1:]))
        invalid=(events[:index]+events[index+1:],events[:index]+[events[index]]+events[index:],bad_eat)
        initial=f.e.initialize(f.catalog,stream_ids=self.j.array([1],self.j.int32))
        for stream in invalid:
            chunks=self.pack(stream);self.assertEqual(len(chunks),1)
            out,error=self.jax.jit(lambda s,c:f.e.fold_chunk(s,c,f.catalog))(initial,chunks[0])
            self.assertNotEqual(error.tolist(),[0])
            for a,b in zip(self.jax.tree.leaves(initial),self.jax.tree.leaves(out)):np.testing.assert_array_equal(a,b)

    def test_pending_item_counter_namespace_empty_refusal_and_batch_isolation(self):
        import numpy as np
        from gpu_engine import public_inputs
        f=self.f;events=[e for p in f.projections(f.public.prefix) for e in p.events]
        chunk=self.pack(events,batch=8)[0]
        initial=f.e.initialize(f.catalog,stream_ids=self.j.arange(1,9,dtype=self.j.int32))
        bad=initial._replace(context=initial.context._replace(refresh=initial.context.refresh._replace(
            item_expected=initial.context.refresh.item_expected.at[0].set(3))))
        empty=chunk._replace(valid=self.j.zeros_like(chunk.valid),more=self.j.zeros_like(chunk.more))
        for payload in (empty,chunk):
            out,error=self.jax.jit(lambda s,c:f.e.fold_chunk(s,c,f.catalog))(bad,payload)
            self.assertEqual(error.tolist(),[int(public_inputs.Error.SOURCE)]+[0]*7)
            for a,b in zip(self.jax.tree.leaves(bad.context.refresh),self.jax.tree.leaves(out.context.refresh)):
                # Cache namespace arrays are global, all other values carry B.
                if a.shape[:1]==(8,) and a.dtype!=self.j.uint32:np.testing.assert_array_equal(a[0],b[0])
            self.assertEqual(out.cursor.sequence[0],bad.cursor.sequence[0])

    def test_item_after_common_or_reversed_two_fact_frame_refuses_atomically(self):
        import numpy as np
        from gpu_engine import public_inputs
        f=self.f
        for raw in ('|-heal|p1a: Snorlax|90/100|[from] item: Leftovers',
                '|-item|p1a: Snorlax|Choice Band|[from] item: Leftovers'):
            events=[e for p in f.projections([*f.public.prefix,raw]) for e in p.events]
            indices=[i for i,e in enumerate(events) if e.kind==public_inputs.Kind.ITEM]
            if len(indices)==1:
                item=events.pop(indices[0]);events.insert(len(events)-1,item)
            else:
                self.assertEqual(len(indices),2)
                events[indices[0]],events[indices[1]]=events[indices[1]],events[indices[0]]
            initial=f.e.initialize(f.catalog,stream_ids=self.j.array([1],self.j.int32))
            chunks=self.pack(events);self.assertEqual(len(chunks),1)
            out,error=self.jax.jit(lambda s,c:f.e.fold_chunk(s,c,f.catalog))(initial,chunks[0])
            self.assertNotEqual(error.tolist(),[0])
            for a,b in zip(self.jax.tree.leaves(initial),self.jax.tree.leaves(out)):np.testing.assert_array_equal(a,b)

    def test_item_word_family_coherence_and_inactive_literal_source_lookup(self):
        from gpu_engine import public_inputs,public_wire,public_refresh_stream
        f=self.f
        for word,code in ((1,public_refresh_stream.NOOP),(10,public_refresh_stream.ITEM),(4,public_refresh_stream.ITEM)):
            with self.assertRaisesRegex(ValueError,'raw boundary'):
                public_inputs.PublicEvent(public_inputs.Kind.RAW_BEGIN,1,0,public_inputs.Phase.ACTION,
                    payload=(word,0,0,code,0,0,0,0)).words()
        lines=['|switch|p1a: Nick|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100','|faint|p1a: Nick']
        host,_=public_wire.project_lines(public_wire.WireState(),lines,**f.e.wire_kwargs(f.catalog))
        with self.assertRaisesRegex(ValueError,'inactive item lookup'):
            public_wire.project_line(host,'|-immune|p1a: Nick|[from] item: Leftovers',**f.e.wire_kwargs(f.catalog))
        # The actual source's literal species fallback can reuse the old row.
        _,projected=public_wire.project_line(host,'|-immune|p1a: Snorlax|[from] item: Leftovers',**f.e.wire_kwargs(f.catalog))
        event=next(e for e in projected.events if e.kind==public_inputs.Kind.ITEM)
        self.assertEqual((event.target_side,event.target_slot,event.payload[0]),(0,0,1))
