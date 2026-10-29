"""Pending-switch raw frames survive arbitrary chunk cuts and row refusals."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class SwitchStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_switch_context import SwitchContextTests
        SwitchContextTests.setUpClass.__func__(cls)

    def pack(self,events,projection,batch=1):
        import numpy as np
        from gpu_engine import public_inputs
        bindings={name:np.asarray(getattr(projection,name),np.uint32) for name in (
            'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding',
            'active_binding','damage_binding','type_binding','sleep_binding','toxic_binding','tendency_binding')}
        return list(public_inputs.pack_chunks(np.arange(1,batch+1,dtype=np.int32),[events]*batch,
            source_binding=self.catalog.source.device.binding,**bindings))

    def same(self,left,right):
        import numpy as np
        for a,b in zip(self.jax.tree.leaves(left),self.jax.tree.leaves(right)):np.testing.assert_array_equal(a,b)

    def test_cut_after_header_and_prelude_then_late_chunk_rolls_back(self):
        import numpy as np
        from gpu_engine import public_inputs
        projections=self.f.projections(self.lines)
        events=[event for p in projections for event in p.events]
        target=next(i for i,event in enumerate(events) if i>10 and event.kind==public_inputs.Kind.RAW_BEGIN
            and event.payload[3]==2)
        parts=(events[:target+1],events[target+1:target+2],events[target+2:])
        chunks=[chunk for part in parts for chunk in self.pack(part,projections[0])]
        chunks=[chunk._replace(more=self.j.array([i<len(chunks)-1])) for i,chunk in enumerate(chunks)]
        state=self.f.e.initialize(self.catalog,stream_ids=self.j.array([1],self.j.int32))
        for index,chunk in enumerate(chunks):
            before=state
            if index==1:
                for word,value in ((2,7),(3,99)):
                    malformed=chunk._replace(words=chunk.words.at[0,0,word].set(value))
                    refused,error=type(self).step(before,malformed)
                    self.assertEqual(error.tolist(),[int(public_inputs.Error.HEADER)])
                    self.same(before,refused)
            if index==len(chunks)-1:
                last=int(np.asarray(chunk.valid[0]).sum())-1
                corrupted=chunk._replace(words=chunk.words.at[0,last,0].set(999))
                refused,error=type(self).step(before,corrupted)
                self.assertNotEqual(error.tolist(),[0]);self.same(before,refused)
            state,error=type(self).step(state,chunk);self.assertEqual(error.tolist(),[0])
            if index==0:
                self.assertEqual(state.context.refresh.switch_seen.tolist(),[0]);self.assertTrue(bool(state.cursor.needs_more[0]))
            if index==1:
                self.assertEqual(state.context.refresh.switch_seen.tolist(),[1]);self.assertTrue(bool(state.cursor.needs_more[0]))
        self.assertEqual(state.context.refresh.family.tolist(),[0]);self.assertEqual(state.context.refresh.switch_seen.tolist(),[0])
        self.assertTrue(bool(state.context.switches.ruled_out[0,0,0]))
        self.assertFalse(bool(state.cursor.needs_more[0]))

    def test_batch8_bad_namespace_and_post_attribution_error_preserve_other_rows(self):
        import numpy as np
        projections=self.f.projections(self.lines[:8]);events=[event for p in projections for event in p.events]
        chunks=self.pack(events,projections[0],batch=8)
        initial=self.f.e.initialize(self.catalog,stream_ids=self.j.arange(1,9,dtype=self.j.int32))
        step=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.catalog))
        good=initial
        for chunk in chunks:good,error=step(good,chunk);self.assertEqual(error.tolist(),[0]*8)
        bad=initial._replace(context=initial.context._replace(switches=initial.context.switches._replace(
            binding=initial.context.switches.binding.at[0,0].set(initial.context.switches.binding[0,0]^self.j.uint32(1)))))
        bad_initial=bad
        for index,chunk in enumerate(chunks):
            if index==len(chunks)-1:
                last=int(np.asarray(chunk.valid[1]).sum())-1
                chunk=chunk._replace(words=chunk.words.at[1,last,0].set(999))
            bad,error=step(bad,chunk)
        self.assertNotEqual(int(error[0]),0);self.assertNotEqual(int(error[1]),0)
        self.assertEqual(error.tolist()[2:],[0]*6)
        # Rows0/1 are unchanged at their rejected chunk; in this short roster
        # all events fit one chunk, including the negative checkpoint.
        self.assertEqual(len(chunks),1)
        paths,_=self.jax.tree_util.tree_flatten_with_path(bad)
        for (path,actual),start,expected in zip(paths,self.jax.tree.leaves(bad_initial),self.jax.tree.leaves(good)):
            names=[getattr(key,'name','') for key in path]
            global_namespace='visible' in names and names[-1] in ('source_binding','schema_binding')
            if global_namespace:
                np.testing.assert_array_equal(actual,expected)
            else:
                np.testing.assert_array_equal(actual[:2],start[:2])
                np.testing.assert_array_equal(actual[2:],expected[2:])
        self.assertEqual(bad.context.refresh.controller.visible.source_binding.shape,(8,))

    def test_encode_refuses_corrupt_or_missing_switch_namespace(self):
        from dataclasses import replace
        state=self.f.ingest(self.lines[:3])
        own=self.f.e.parse_own_requests(self.catalog,[self.f.request()],owner_sides=[0])
        encode=self.jax.jit(lambda s:self.f.e.encode(s,own,self.catalog,observer_sides=self.j.array([0],self.j.int32)))
        self.assertEqual(encode(state).observation.admitted.tolist(),[True])
        bad_states=[state._replace(context=state.context._replace(switches=None))]
        for name in ('binding','species_binding'):
            value=getattr(state.context.switches,name)
            corrupt=state.context.switches._replace(**{name:value.at[0,0].set(value[0,0]^self.j.uint32(1))})
            bad_states.append(state._replace(context=state.context._replace(switches=corrupt)))
        for bad in bad_states:
            out=encode(bad)
            self.assertEqual(out.observation.admitted.tolist(),[False])
            for values in (out.observation.numeric_features,out.observation.categorical_ids,out.observation.legal_action_mask,
                    out.observation.attention_mask,out.posterior_mask,out.posterior_weights):self.assertFalse(bool(values.any()))
        missing=replace(self.catalog,context=self.catalog.context._replace(switches=None))
        out=self.f.e.encode(state,own,missing,observer_sides=self.j.array([0],self.j.int32))
        self.assertEqual(out.observation.admitted.tolist(),[False])
