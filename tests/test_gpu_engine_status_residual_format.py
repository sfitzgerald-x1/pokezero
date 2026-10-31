"""Original FULL word controls; diagnostic formatting, not source admission."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_status_residual_format as d,engine_status_residual_callbacks as r
from gpu_engine import engine_full_v4_event_device as full,engine_lifecycle_callbacks as life
from gpu_engine import public_inputs as inputs,public_wire,v4_encoder as v4,mechanics as m
import test_gpu_engine_engine_full_v4_event_device as fixture


class StatusResidualFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FullV4EventDeviceTests.setUpClass();cls.f=fixture.FullV4EventDeviceTests
        cls.helper=r.compile_status_residual(cls.f.f.metadata,engine_catalog=cls.f.f.lifecycle)
        cls.compiled=d.compile_formatter(cls.helper,full_adapter=cls.f.compiled)
        cls.tables=d.check_compiled(cls.compiled)
        initial=cls.f().initial();switched=cls.f().step(initial,[[8,8]]*2)
        ids=j.arange(171,175,dtype=j.int32);owners=j.array([0,0,1,1],j.int32)
        cursor=jax.tree.map(lambda a:j.tile(a,(2,)+(1,)*(a.ndim-1)),switched.cursor)._replace(stream_id=ids)
        cls.cursor=cursor;cls.owners=owners;cls.sides=j.array([0,1,0,1],j.int32)
        cls.statuses=j.array([m.BRN,m.BRN,m.PSN,m.PSN],j.int32)
        cls.maximum=j.full(4,1001,j.int32);cls.old=j.full(4,999,j.int32);cls.new=j.full(4,998,j.int32)
        cls.labels=[];rows=[];wires=[];projections=[]
        kwargs=v4.wire_kwargs(cls.f.f.catalog)
        for row in range(4):
            names=[]
            for side in (0,1):
                names.append(tuple(cls.f.f.catalog.dex.species_info(cls.f.f.metadata['vocabulary']['species'][
                    int(initial.state.species[row%2,side,physical])]).name for physical in (0,5)))
            lines=[f'|switch|p{s+1}a: {names[s][0]}|{names[s][0]}, L80|100/100' for s in (0,1)]
            lines+=['|turn|1']+[f'|switch|p{s+1}a: {names[s][1]}|{names[s][1]}, L80|100/100' for s in (0,1)]
            lines+=['|upkeep','|turn|2']
            side=int(cls.sides[row]);ident=f'p{side+1}a: {names[side][1]}';cls.labels.append(ident)
            token='brn' if row<2 else 'psn';lines+=[f'|-status|{ident}|{token}']
            own=int(owners[row])==side;hp=999 if own else 99;maximum=1001 if own else 100
            lines+=[f'|-damage|{ident}|{hp}/{maximum} {token}','|']
            wire,ps=public_wire.project_lines(public_wire.WireState(),lines,**kwargs)
            wires.append(wire);rows.append([e for p in ps for e in p.events]);projections.append(ps)
        cls.wires=wires;cls.kwargs=kwargs
        chunks=list(inputs.pack_chunks(np.array(ids),rows,**{key:np.array(getattr(cls.tables.namespaces,key)) for key in full._NAMES}))
        public=v4.initialize(cls.f.f.catalog,stream_ids=ids,owner_sides=owners)
        fold=jax.jit(lambda p,c:v4.fold_chunk(p,c,cls.f.f.catalog))
        for chunk in chunks:
            public,error=jax.block_until_ready(fold(public,chunk));assert error.tolist()==[0]*4,error
        stream=full.register(ids,owners,compiled=cls.f.compiled)
        stream=stream._replace(causal=stream.causal._replace(cursor=cursor,sequence=public.cursor.sequence),
            sequence=public.cursor.sequence,live=j.ones((4,2),bool),species=public.mons.species,
            status=public.mons.status,hp=public.mons.hp_numerator,maximum=public.mons.hp_denominator)
        cls.stream,cls.public=stream,public
        tape=life._tape(cursor,cls.f.f.lifecycle,64)
        cls.tape=r.record_status_residual_device(tape,cls.sides,j.full(4,5,j.int32),cls.old,cls.new,cls.maximum,
            cls.statuses,j.zeros(4,j.int32),j.zeros(4,j.int32),cls.helper.device,eligible=True)
        cls.fast=staticmethod(jax.jit(lambda t,s,p:d.format_device(t,0,s,p,cls.tables,eligible=True)))

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def reference(self,tape=None):
        tape=self.tape if tape is None else tape;rows=[]
        for row in range(4):
            w=np.array(tape.words[row,0]);own=int(self.owners[row])==int(w[3]);hp=int(w[6]);maximum=int(w[7])
            if not own:
                hp=min(99,(100*hp+maximum-1)//maximum);maximum=100
            token={m.BRN:'brn',m.PSN:'psn',m.TOX:'tox'}[int(w[8])]
            tag='brn' if int(w[9])==r.BURN else 'psn'
            _,p=public_wire.project_line(self.wires[row],f'|-damage|{self.labels[row]}|{hp}/{maximum} {token}|[from] {tag}',**self.kwargs)
            rows.append(list(p.events))
        return next(inputs.pack_chunks(np.array(self.cursor.stream_id),rows,
            **{key:np.array(getattr(self.tables.namespaces,key)) for key in full._NAMES}))

    def test_brn_psn_exact_original_full_words_both_owner_displays_nonphysical_encounters(self):
        result=jax.block_until_ready(self.fast(self.tape,self.stream,self.public))
        self.assertEqual(result.format_coherent.tolist(),[True]*4)
        self.assertEqual(result.refusal.tolist(),[257]*4)
        self.same(result.chunk,self.reference())
        self.assertEqual(self.cursor.physical_active.tolist(),[[5,5]]*4)
        self.assertEqual(self.public.mons.active.tolist(),[[1,1]]*4)
        self.assertFalse(bool(result.chunk.more.any()))
        checked=d.format_receipt(self.tape,0,self.stream,self.public,compiled=self.compiled,eligible=True)
        self.same(checked,result)
        # Test-only original consumer folding does not register this formatter.
        fold=jax.jit(lambda p,c:v4.fold_chunk(p,c,self.f.f.catalog))
        got,error=fold(self.public,result.chunk);want,werror=fold(self.public,self.reference())
        self.assertEqual(error.tolist(),[0]*4);self.same(got,want);self.same(error,werror)

    def test_private_counter_changes_do_not_enter_public_words_and_toxic_is_closed(self):
        baseline=self.fast(self.tape,self.stream,self.public)
        tape=self.tape._replace(words=self.tape.words.at[:,0,1:3].set(7))
        self.same(self.fast(tape,self.stream,self.public),baseline)
        tox=self.tape._replace(words=self.tape.words.at[:,0,8].set(m.TOX).at[:,0,9].set(r.TOXIC))
        rows=j.arange(4)
        status=self.stream.status.at[rows,self.sides,1].set(6)
        tox_stream=self.stream._replace(status=status)
        tox_public=self.public._replace(mons=self.public.mons._replace(status=status),
            context=self.public.context._replace(toxic=self.public.context.toxic._replace(
                condition_tox=self.public.context.toxic.condition_tox.at[rows,self.sides].set(True),
                known=j.ones_like(self.public.context.toxic.known),stage=j.ones_like(self.public.context.toxic.stage))))
        got=self.fast(tox,tox_stream,tox_public)
        self.assertEqual(got.format_coherent.tolist(),[False]*4);self.assertFalse(bool(got.chunk.valid.any()))
        self.assertEqual(got.refusal.tolist(),[257]*4)
        for status in (m.NONE,m.PAR,m.SLP,m.FRZ):
            tape=self.tape._replace(words=self.tape.words.at[:,0,8].set(status).at[:,0,9].set(0))
            self.assertFalse(bool(self.fast(tape,self.stream,self.public).format_coherent.any()))

    def test_all21_original_lexical_templates_are_exact_unqualified_controls(self):
        name=self.f.f.catalog.species.names[1];ident='p1a: '+name
        prior,_=public_wire.project_lines(public_wire.WireState(),
            [f'|switch|{ident}|{name}, L80|100/100','|turn|1','|'],**self.kwargs)
        for reason,tag in ((r.BURN,'brn'),(r.POISON,'psn'),(r.TOXIC,'psn')):
            for status,token in enumerate(('', 'slp','par','brn','frz','psn','tox')):
                _,p=public_wire.project_line(prior,f'|-damage|{ident}|71/100'+(' '+token if token else '')+'|[from] '+tag,**self.kwargs)
                words=np.array([e.words() for e in p.events]);n=int(self.tables.length[reason,status])
                np.testing.assert_array_equal(np.array(self.tables.templates[reason,status,:n]),words)

    def test_malformed_before_ledger_payload_stamps_prefix_and_refusals_are_atomic(self):
        for column,value in ((0,19),(1,-1),(2,1),(3,2),(4,0),(5,998),(6,0),(7,998),(8,m.PSN),(9,r.POISON)):
            tape=self.tape._replace(words=self.tape.words.at[0,0,column].set(value))
            got=self.fast(tape,self.stream,self.public)
            self.assertFalse(bool(got.format_coherent[0]));self.assertEqual(got.format_coherent[1:].tolist(),[True]*3)
            self.assertFalse(bool(got.chunk.valid[0].any()));self.assertEqual(int(got.refusal[0]),257)
        cases=[(self.tape._replace(event_turn=self.tape.event_turn.at[0,0].add(1)),self.stream,self.public),
            (self.tape._replace(event_generation=self.tape.event_generation.at[0,0,0].add(1)),self.stream,self.public),
            (self.tape._replace(refusal=self.tape.refusal.at[0].set(385)),self.stream,self.public),
            (self.tape,self.stream,self.public._replace(cursor=self.public.cursor._replace(needs_more=self.public.cursor.needs_more.at[0].set(True)))),
            (self.tape,self.stream,self.public._replace(context=self.public.context._replace(nonproc=self.public.context.nonproc._replace(residual=self.public.context.nonproc.residual.at[0].set(False))))),
            (self.tape,self.stream,self.public._replace(context=self.public.context._replace(refresh=self.public.context.refresh._replace(family=self.public.context.refresh.family.at[0].set(12))))),
            (self.tape,self.stream,self.public._replace(mons=self.public.mons._replace(hp_numerator=self.public.mons.hp_numerator.at[0,0,1].add(1))))]
        for tape,stream,public in cases:
            got=self.fast(tape,stream,public);self.assertFalse(bool(got.format_coherent[0]));self.assertFalse(bool(got.chunk.valid[0].any()))
        for index in (-1,64,True,np.array([2**32],np.uint64)):
            got=d.format_device(self.tape,index,self.stream,self.public,self.tables,eligible=True)
            self.assertFalse(bool(got.chunk.valid.any()));self.assertEqual(got.refusal.tolist(),[257]*4)

    def test_host_original_edges_metadata_and_allclosed_guards(self):
        with self.assertRaises(ValueError):d.check_compiled(replace(self.compiled))
        original=self.compiled.device
        for key in ('templates','length','encoder_binding','species_map','binding'):
            for value in (np.array(getattr(original,key)),j.array(getattr(original,key),copy=True)):
                try:
                    object.__setattr__(self.compiled,'device',original._replace(**{key:value}))
                    with self.assertRaises(ValueError):d.check_compiled(self.compiled)
                finally:object.__setattr__(self.compiled,'device',original)
        metadata=self.f.f.metadata;old=metadata['schema'];metadata['schema']='changed'
        try:
            with self.assertRaises(ValueError):d.check_compiled(self.compiled)
        finally:metadata['schema']=old
        calls=(lambda:d.compile_formatter(self.helper,full_adapter=self.f.compiled),
            lambda:d.check_compiled(self.compiled),lambda:d.format_receipt(self.tape,0,self.stream,self.public,compiled=self.compiled,eligible=True))
        with patch.object(r,'check_compiled',side_effect=AssertionError('host reached')) as guard:
            for call in calls:
                for fast,args in ((jax.jit(call),()),(jax.jit(lambda unused:call()),(j.array(0,j.int32),))):
                    with self.assertRaisesRegex(ValueError,'host boundary cannot be traced'):fast(*args)
            guard.assert_not_called()

    def test_b8_inactive_global_namespaces_final_cursor_and_hot_jit_no_callbacks(self):
        tile=lambda x:j.tile(x,(2,)+(1,)*(x.ndim-1)) if x.ndim and x.shape[0]==4 else x
        tape,stream,public=jax.tree.map(tile,(self.tape,self.stream,self.public))
        ids=j.arange(271,279,dtype=j.int32)
        tape=tape._replace(stream_id=ids,cursor=tape.cursor._replace(stream_id=ids))
        stream=stream._replace(causal=stream.causal._replace(cursor=stream.causal.cursor._replace(stream_id=ids)))
        public=public._replace(cursor=public.cursor._replace(stream_id=ids))
        mask=j.array([True,False]*4)
        call=lambda t:d.format_device(t,0,stream,public,self.tables,eligible=mask)
        fast=jax.jit(call);expected=call(tape)
        self.same(jax.block_until_ready(fast(tape)),expected);self.same(fast(tape),expected)
        self.assertEqual(fast._cache_size(),1);self.assertEqual(expected.format_coherent.tolist(),mask.tolist())
        self.assertEqual(expected.refusal.tolist(),[257]*8)
        for key in full._NAMES:self.assertEqual(getattr(expected.chunk,key).shape,(8,))
        changed=tape._replace(source_binding=tape.source_binding.at[7].add(j.uint32(1)))
        self.assertFalse(bool(call(changed).chunk.valid.any()))
        clean=tape._replace(refusal=j.zeros(8,j.int32))
        self.assertEqual(call(clean).refusal.tolist(),[256,0]*4)
        # Earlier receipt joins BEFORE cursor. Later/final tape cursor fields
        # are never substituted for the earlier declaration's identity.
        final=tape._replace(cursor=tape.cursor._replace(turn=tape.cursor.turn+1))
        self.same(call(final),expected)
        names=set()
        def walk(v):
            if isinstance(v,(tuple,list)):
                for x in v:walk(x)
            elif isinstance(v,dict):
                for x in v.values():walk(x)
            elif hasattr(v,'eqns'):
                for e in v.eqns:names.add(e.primitive.name);walk(e.params)
            elif hasattr(v,'jaxpr'):walk(v.jaxpr)
        walk(jax.make_jaxpr(call)(tape));self.assertFalse(names&{'io_callback','pure_callback','debug_callback'})


if __name__=='__main__':unittest.main()
