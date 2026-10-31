"""Tiny original hypothetical components; no model/Native/game/scientific bank."""
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np

from gpu_engine import v5_hypothetical_batch as batch,v5_public_world as world,v5_public_branch as branch
from gpu_engine import v5_observation as observation,transition,private_team
from pokezero import v5_contract as contract
from pokezero.v5_model import V5ModelConfig

# An independent API bootstrap can be supplied while the separate provider
# lane is not yet merged. It is immutable ORIGINAL source, never a fake grid.
if importlib.util.find_spec('gpu_engine.v5_depth1_provider') is None and os.environ.get('POKEZERO_V5_GRID_API_DIR'):
    import gpu_engine
    gpu_engine.__path__.append(os.environ['POKEZERO_V5_GRID_API_DIR'])
from gpu_engine.v5_depth1_provider import GridCell,GridBatch


class LayoutTests(unittest.TestCase):
    def test_b8_global8_layout_and_drift_are_not_row_inferred(self):
        from typing import NamedTuple
        class Row(NamedTuple):
            row:object
            binding:object
        global8=j.arange(8,dtype=j.uint32)
        original=Row(j.array([[9,10]],j.int32),global8)
        stacked,layout=batch._stack([original]*8)
        self.assertEqual(stacked.row.shape,(8,2));self.assertIs(stacked.binding,global8)
        restored=batch._take(stacked,layout,7,8)
        self.assertEqual(restored.row.shape,(1,2));self.assertIs(restored.binding,global8)
        with self.assertRaises(ValueError):batch._stack([original,original._replace(binding=global8+1)])
        with self.assertRaises(ValueError):batch._take(stacked._replace(binding=j.tile(global8,(8,1))),layout,0,8)

    def test_chance_bounds_and_outer_trace_refuse_before_execution(self):
        good=transition.ChanceInputs(j.zeros((1,2,12),j.int32),*[j.zeros((1,2),j.int32) for _ in range(4)],j.zeros(1,j.int32))
        batch._chance(good)
        with self.assertRaises(ValueError):batch._chance(good._replace(draws=good.draws.at[0,0,0].set(2)))
        with self.assertRaises(ValueError):jax.jit(lambda dummy:batch._host() or dummy)(j.array(0))

    def test_transport_is_distinct_deterministic_and_preserves_original_global_and_physics(self):
        from types import SimpleNamespace as NS
        from typing import NamedTuple
        class Cursor(NamedTuple):
            stream_id:object
            binding:object
        class Record(NamedTuple):
            stream_ids:object
            origin_binding:object
        class Core(NamedTuple):
            cursor:object
            weather_profile:object
            physical:object
        global8=j.arange(8,dtype=j.uint32);cursor=Cursor(j.array([3401],j.int32),global8)
        record=Record(cursor.stream_id,j.arange(8,dtype=j.uint32)[None,:])
        physical=j.array([[3,4,5]],j.int32)
        row=batch._PreparedRow(Core(cursor,record,physical),(cursor,cursor),(cursor,cursor),record)
        root=NS(history=NS(stream_id=3401),origin_sha256='a'*64)
        cell=GridCell(contract.DecisionAddress(3401,2,1),0,0,1,2)
        transported=[]
        for index in range(8):
            value,mapping=batch._transport(row,root,cell,index+1,index,padding=index if index>3 else None)
            repeated,again=batch._transport(row,root,cell,index+1,index,padding=index if index>3 else None)
            self.assertEqual(mapping,again);self.assertEqual(value.core.cursor.stream_id.tolist(),[index+1])
            self.assertIs(value.core.cursor.binding,global8);self.assertIs(value.core.physical,physical)
            self.assertEqual(mapping['original_stream'],3401);self.assertEqual(mapping['address']['owner'],1)
            np.testing.assert_array_equal(value.record.origin_binding,repeated.record.origin_binding)
            transported.append(value)
        batch._unique_transport(transported)
        with self.assertRaisesRegex(ValueError,'unique'):batch._unique_transport([transported[0]]*8)
        with self.assertRaises(ValueError):batch._transport(row,root,cell,2**31,0)
        self.assertEqual(row.core.cursor.stream_id.tolist(),[3401])

    def test_mixed_owner_selection_uses_two_genuine_row_fields_and_never_global8(self):
        zero=observation.Observation(*(j.zeros((8,2),j.int32) for _ in range(12)))
        one=observation.Observation(*(j.ones((8,2),j.int32) for _ in range(12)))
        owners=j.array([0,1,0,1,1,0,1,0],j.int32)
        result=batch._own_view((zero,one),owners)
        for value in result:np.testing.assert_array_equal(value,j.repeat(owners[:,None],2,axis=1))
        with self.assertRaises(ValueError):batch._own_view((zero,one._replace(error=j.arange(8,dtype=j.uint32))),owners)


class ActualBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_v5_public_world as fixture
        fixture.PublicWorldTests.setUpClass();cls.f=fixture.PublicWorldTests();cls.runtime=cls.f.runtime
        census=contract.source_variant_census(cls.f.source)
        labels=contract.LabelVocabulary(census['source_rows_sha256'],tuple(cls.f.catalog.moves.names[1:]),census['species_variants'])
        category=np.asarray(private_team.vocabulary_binding(cls.f.catalog.vocabulary),dtype='>u4').tobytes().hex()
        config=V5ModelConfig(cls.f.catalog.vocabulary.size,category,labels,width=8,layers=1,attention_heads=2,
            expert_width=8,value_hidden=(8,8))
        cls.adapter=observation.compile_adapter(cls.f.catalog,model_config=config)
        root,start,history,static=cls.f.fixture(spent=True,slow_own=True)
        retained=world.public_carry(history,cls.runtime);roots=[]
        for seed in (5,6):
            value=world.materialize(root,start,prepared=cls.f.f.prepared,history=history,own_static=static,seed=seed,retained=retained)
            assert value.branch is not None,value.refusals;roots.append(value.branch)
        cls.roots=tuple(roots)
        draws=j.zeros((1,2,12),j.int32).at[:,:,4].set(1).at[:,:,5].set(0)
        cls.chance=transition.ChanceInputs(draws,*[j.zeros((1,2),j.int32) for _ in range(4)],j.zeros(1,j.int32))

    def test_batched_original_kernel_full_public29_equals_scalar_and_preserves_unknown_cell(self):
        # TWO distinct worlds/actions share one real B2 numerical kernel call.
        # This fixture's switched branch is refused by the actual pipeline;
        # it remains an executed UNKNOWN, never dropped or called an invalid route.
        cells=(GridCell(self.roots[0].root.address,0,0,0,0),GridCell(self.roots[1].root.address,1,1,1,0),
            GridCell(self.roots[0].root.address,0,0,4,0))
        declared=GridBatch(self.roots,(self.chance,self.chance),cells)
        executor=batch._executor(self.roots[0].prepared,self.adapter)
        # Numerical comparisons use device output via the immutable executor,
        # then public API chunks reuse the compiled stages from the same object.
        original_contexts=[executor.prepare_row(self.roots[i]) for i in (0,1)]
        contexts=[batch._transport(value,self.roots[i],cells[i],i+1,i)[0] for i,value in enumerate(original_contexts)]
        core,layout=batch._stack([x.core for x in contexts]);pub,_=batch._stack([x.publics for x in contexts])
        streams,_=batch._stack([x.streams for x in contexts]);profiles,_=batch._stack([x.record for x in contexts])
        chance,_=batch._stack([self.chance,self.chance]);choices=j.array([[0,0],[1,0]],j.int32)
        routes=executor.route(core,choices);self.assertTrue(bool((routes[0].admitted&routes[1].admitted).all()))
        engine_actions=j.stack([x.engine_action for x in routes],axis=1)
        result=jax.block_until_ready(executor.run(core,pub,streams,profiles,engine_actions,chance,j.ones(2,bool)))
        scalar_results=[branch.step(self.roots[row],choice,chance_inputs=self.chance,hypothetical_profiles=True)
            for row,choice in enumerate(((0,0),(1,0)))]
        if not bool(result.good.all()):
            comparison=[]
            for row,scalar in enumerate(scalar_results):
                different=[]
                for name in result.frame.state._fields:
                    if not np.array_equal(np.asarray(getattr(result.frame.state,name))[row:row+1],np.asarray(getattr(scalar.state,name))):different.append(name)
                comparison.append(dict(state_different=different,
                    batch_raw=np.asarray(result.frame.callbacks.words)[row,:int(result.frame.callbacks.count[row])].tolist(),
                    scalar_raw=np.asarray(scalar.raw_frame.callbacks.words)[0,:int(scalar.raw_frame.callbacks.count[0])].tolist(),
                    initial_public=[dict(hp=np.asarray(x.hp).tolist(),maximum=np.asarray(x.maximum).tolist(),status=np.asarray(x.status).tolist(),owner=np.asarray(x.causal.owner).tolist()) for x in contexts[row].streams]))
            Path('/tmp/pokezero-v5-batch-diagnostic.json').write_text(json.dumps(comparison))
        self.assertEqual(result.good.tolist(),[True,True],msg=str((result.diagnostics.tolist(),
            [(x.refusals,x.diagnostics) for x in scalar_results])))
        from gpu_engine import engine_execution_completeness as accounting
        from gpu_engine import v5_hypothetical_profile as profiles
        scalar_consume=jax.jit(lambda source,stream,public:accounting.consume_frame_device(
            source.callbacks.execution.ledger,source,stream,public,active=j.ones(1,bool),
            tables=self.runtime['execution_compiled'].device,adapter_tables=self.runtime['public'].device,
            catalog=self.runtime['catalog'],leftovers=self.runtime['leftovers_context'].data,
            leftovers_no_output=self.runtime['no_output_compiled'].device,
            hypothetical_hp=profiles.Context(source.weather_profile,self.runtime['speed_profile'].device)))
        for row,scalar in enumerate(scalar_results):
            self.assertFalse(scalar.refusals,(scalar.refusals,scalar.diagnostics))
            for actual,expected in zip(jax.tree.leaves(result.frame.state),jax.tree.leaves(scalar.state),strict=True):
                np.testing.assert_array_equal(actual[row:row+1],expected)
            for owner in (0,1):
                source=scalar.raw_frame
                prior=original_contexts[row]
                _,consumed=jax.block_until_ready(scalar_consume(source,prior.streams[owner],prior.publics[owner]))
                original_words=np.asarray(consumed.projection.frames.words)[:,0][np.asarray(consumed.projection.frames.valid)[:,0]]
                batched_words=np.asarray(result.frames[owner].words)[:,row][np.asarray(result.frames[owner].valid)[:,row]]
                np.testing.assert_array_equal(batched_words,original_words)
                expected=observation.append_device(scalar.encoded[owner],scalar.public[owner],self.adapter,observer_sides=j.array([owner],j.int32))
                for name in ('numeric_features','categorical_ids','token_type_ids','attention_mask','legal_action_mask'):
                    np.testing.assert_array_equal(getattr(result.appended[owner],name)[row:row+1],getattr(expected,name),err_msg=name)
        chunks=list(batch.iter_execute(declared,self.adapter,max_rows=2))
        self.assertEqual(tuple(cell for chunk in chunks for cell in chunk.cells),cells)
        self.assertEqual([chunk.admitted.tolist() for chunk in chunks],[[True,True],[False]])
        self.assertEqual(chunks[0].model_indices,(0,1));self.assertEqual(chunks[0].inputs['numeric_features'].shape,(2,1,29,132))
        self.assertIsNone(chunks[1].inputs)
        self.assertEqual(chunks[1].refusals,(('actual-hypothetical-kernel-FULL-or-public29-refused',),))
        self.assertEqual((chunks[0].device_rows,chunks[0].active_rows,chunks[0].padding),(2,2,0))
        self.assertEqual((chunks[1].device_rows,chunks[1].active_rows,chunks[1].padding),(2,1,1))
        self.assertEqual(chunks[1].diagnostics['planned_transport'][0]['grid_ordinal'],2)
        self.assertEqual(chunks[1].diagnostics['planned_transport'][0]['stream'],3)
        # The actual refused cell still executes. Its denominator remains and
        # it never enters a model. A separate one-cell chunk tests frozen padding.
        mixed=GridBatch(self.roots,(self.chance,self.chance),(cells[0],cells[2]))
        chunk=next(batch.iter_execute(mixed,self.adapter,max_rows=2))
        self.assertEqual(chunk.admitted.tolist(),[True,False]);self.assertEqual(chunk.model_indices,(0,))
        self.assertEqual((chunk.device_rows,chunk.active_rows,chunk.padding),(2,2,0))
        self.assertEqual(chunk.inputs['history_mask'].shape,(1,1))
        padded=next(batch.iter_execute(GridBatch(self.roots,(self.chance,self.chance),(cells[0],)),self.adapter,max_rows=2))
        self.assertEqual(padded.admitted.tolist(),[True]);self.assertEqual(padded.model_indices,(0,))
        self.assertEqual((padded.device_rows,padded.active_rows,padded.padding),(2,1,1))
        self.assertEqual(padded.diagnostics['padding'][0]['stream'],2)
        with patch.object(executor,'run',lambda *a:None):
            with self.assertRaisesRegex(ValueError,'original fixed'):next(batch.iter_execute(declared,self.adapter,max_rows=2))
