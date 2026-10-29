"""Raw Sleep Clause proofs and whole-frame commit in the actual v4 encoder."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class SleepClauseContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_sleep_clause import SleepClauseTests
        SleepClauseTests.setUpClass.__func__(cls)

    def begin(self,batch=1):
        return self.f.e.initialize(self.catalog,stream_ids=self.j.arange(1,batch+1,dtype=self.j.int32))

    def ingest(self,lines,batch=1):
        state=self.begin(batch);host=self.w.WireState()
        step=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.catalog))
        for line in lines:
            host,p=self.w.project_line(host,line,**self.f.e.wire_kwargs(self.catalog))
            for chunk in self.f.chunks([p],batch=batch):
                state,error=step(state,chunk);self.assertEqual(error.tolist(),[0]*batch)
        return state,host

    def test_real_encoder_field_columns_match_every_source_prefix_both_seats(self):
        import json
        import numpy as np
        from pokezero import showdown as source
        from dataclasses import replace
        state=self.begin();host=self.w.WireState();prefix=[]
        step=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.catalog))
        encode=self.jax.jit(lambda s,o,p:self.f.e.encode(s,o,self.catalog,observer_sides=p))
        lines=['|switch|p1a: Sleepy|Snorlax, L80|100/100','|switch|p2a: Absol|Absol, L80|100/100',
            '|turn|1','|-status|p1a: Sleepy|slp|[from] ability: Effect Spore',
            '|switch|p1a: Fairy|Gardevoir, L80|100/100','|-curestatus|p1: Snorlax|slp|[silent]',
            '|-cureteam|p1a: Fairy','|-status|p2a: Absol|slp','|-curestatus|p2a: Absol|slp',
            '|-status|p2a: Absol|slp|[from] move: Rest','|-curestatus|p2a: Absol|slp']
        for line in lines:
            host,p=self.w.project_line(host,line,**self.f.e.wire_kwargs(self.catalog))
            for chunk in self.f.chunks([p]):state,error=step(state,chunk);self.assertEqual(error.tolist(),[0])
            prefix.append(line)
            for owner in (0,1):
                raw=self.f.request(owner);packet=self.f.e.parse_own_requests(self.catalog,[raw],owner_sides=[owner])
                actual=encode(state,packet,self.j.array([owner],self.j.int32))
                self.assertEqual(actual.observation.admitted.tolist(),[True])
                self.assertEqual(actual.pending_numeric_columns.tolist(),[]);self.assertFalse(bool(actual.full_encoder_admitted[0]))
                replay=source.parse_showdown_replay([*prefix,'|request|'+json.dumps(raw)],battle_id='clause-integration-fixture')
                normalized=source.normalize_for_player(replay,player_id=f'p{owner+1}',configured_showdown_slot=f'p{owner+1}',
                    format_id='gen3randombattle',set_source=self.source)
                reference=source.observation_from_player_state(normalized,category_vocab=self.f.vocab,spec=source.V4_REPLAY_OBSERVATION_SPEC,
                    dex=self.catalog.dex,feature_masks=replace(source.DEFAULT_OBSERVATION_FEATURE_MASKS,exact_state=True,transition_token_budget=0))
                np.testing.assert_array_equal(np.asarray(actual.observation.numeric_features[0,0])[list(self.c.COLUMNS)],
                    np.asarray(reference.numeric_features[0])[list(self.c.COLUMNS)],err_msg=str((line,owner)))

    def test_cure_raw_nickname_proof_legacy_refusal_and_late_frame_rollback(self):
        import numpy as np
        from dataclasses import replace
        from gpu_engine import public_inputs,public_refresh_stream as refresh
        lines=[*self.f.public.prefix,'|-status|p1a: Snorlax|slp',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100']
        state,host=self.ingest(lines,batch=2)
        _,p=self.w.project_line(host,'|-curestatus|p1: Snorlax|slp|[silent]',**self.f.e.wire_kwargs(self.catalog))
        opener=p.events[0];self.assertEqual(opener.kind,public_inputs.Kind.RAW_BEGIN)
        self.assertEqual(opener.payload[3],refresh.CURE);self.assertEqual(opener.payload[7],1)
        meta=self.j.array([opener.payload[4:]],self.j.int32);code=self.j.array([refresh.CURE],self.j.int32)
        self.assertEqual(refresh.metadata_ok(code,meta,lifecycle=True,clause=True).tolist(),[True])
        self.assertEqual(refresh.metadata_ok(code,meta,lifecycle=True,clause=False).tolist(),[False])
        bad=[*p.events[:-1],replace(p.events[-1],sequence=p.events[-1].sequence+1)]
        bindings={n:np.asarray(getattr(p,n),np.uint32) for n in ('species_binding','ability_binding','move_binding',
            'item_binding','context_binding','field_binding','active_binding','damage_binding','type_binding',
            'sleep_binding','toxic_binding','tendency_binding')}
        chunks=list(public_inputs.pack_chunks(np.array([1,2],np.int32),[bad,list(p.events)],
            source_binding=self.catalog.source.device.binding,**bindings))
        self.assertEqual(len(chunks),1)
        result,error=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.catalog))(state,chunks[0])
        self.assertNotEqual(int(error[0]),0);self.assertEqual(int(error[1]),0)
        old,new=state.context.sleep_clause,result.context.sleep_clause
        for a,b in zip(self.jax.tree.leaves(old),self.jax.tree.leaves(new)):np.testing.assert_array_equal(a[0],b[0])
        self.assertTrue(bool(old.victims[1,0,0]));self.assertFalse(bool(new.victims[1,0,0]))
        self.assertEqual(new.holder[1].tolist(),[-1,-1])

    def test_bad_clause_entry_refuses_noop_empty_and_presence_mismatch(self):
        import numpy as np
        from dataclasses import replace
        from gpu_engine import public_inputs
        state,host=self.ingest(self.f.public.prefix,batch=8)
        _,p=self.w.project_line(host,'|-boost|p1a: Snorlax|atk|2',**self.f.e.wire_kwargs(self.catalog))
        chunks=self.f.chunks([p],batch=8);self.assertEqual(len(chunks),1);chunk=chunks[0]
        empty=chunk._replace(valid=self.j.zeros_like(chunk.valid),more=self.j.zeros_like(chunk.more))
        step=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.catalog))
        original=state.context.sleep_clause
        for clause in (original._replace(binding=original.binding.at[0,0].set(original.binding[0,0]^self.j.uint32(1))),
                       original._replace(encounter=original.encounter.at[0,0,0].set(9))):
            bad=state._replace(context=state.context._replace(sleep_clause=clause))
            for payload in (chunk,empty):
                result,error=step(bad,payload)
                self.assertEqual(error.tolist(),[int(public_inputs.Error.SOURCE)]+[0]*7)
                self.assertEqual(result.cursor.sequence[0],bad.cursor.sequence[0])
                np.testing.assert_array_equal(result.mons.boosts[0],bad.mons.boosts[0])
                for a,b in zip(self.jax.tree.leaves(clause),self.jax.tree.leaves(result.context.sleep_clause)):
                    np.testing.assert_array_equal(a[0],b[0])
                np.testing.assert_array_equal(result.context.refresh.controller.visible.source_binding,
                    self.catalog.source.device.binding)
                if payload is chunk:self.assertEqual(int(result.mons.boosts[1,0,0]),2)
        missing=state._replace(context=state.context._replace(sleep_clause=None))
        result,error=step(missing,chunk)
        self.assertEqual(error.tolist(),[int(public_inputs.Error.SOURCE)]*8)
        np.testing.assert_array_equal(result.cursor.sequence,missing.cursor.sequence)
        np.testing.assert_array_equal(result.mons.boosts,missing.mons.boosts)
        without_tables=replace(self.catalog,context=self.catalog.context._replace(sleep_clause=None))
        result,error=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,without_tables))(state,empty)
        self.assertEqual(error.tolist(),[int(public_inputs.Error.SOURCE)]*8)
        np.testing.assert_array_equal(result.cursor.sequence,state.cursor.sequence)
        # State/table symmetry also applies to a legacy context with no raw
        # refresh controller, so an empty chunk cannot bypass this admission.
        legacy=replace(self.catalog,context=self.catalog.context._replace(sleep_clause=None,refresh=None,lifecycle=None))
        orphan=state._replace(context=state.context._replace(refresh=None,lifecycle=None))
        with self.assertRaisesRegex(ValueError,'pending switch evidence requires ordered source and sleep context'):
            public_inputs.fold_chunk(orphan.cursor,orphan.beliefs,self.catalog.source.device,empty,
                mon_state=orphan.mons,context=orphan.context,context_tables=legacy.context)
        # Legacy core has neither the raw controller nor its switch consumer.
        # Remove both sides of that optional pair to exercise Clause symmetry.
        legacy=replace(legacy,context=legacy.context._replace(switches=None))
        orphan=orphan._replace(context=orphan.context._replace(switches=None))
        direct=self.jax.jit(lambda s,c:public_inputs.fold_chunk(s.cursor,s.beliefs,self.catalog.source.device,c,
            mon_state=s.mons,context=s.context,context_tables=legacy.context))(orphan,empty)
        self.assertEqual(direct.error.tolist(),[int(public_inputs.Error.SOURCE)]*8)
        np.testing.assert_array_equal(direct.cursor.sequence,state.cursor.sequence)
        with self.assertRaisesRegex(ValueError,'shape/dtype'):
            step(state._replace(context=state.context._replace(sleep_clause=original._replace(holder=original.holder.astype(self.j.float32)))),chunk)


if __name__=='__main__':unittest.main()
