import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class MudShotStreamGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_mudshot_stream import MudShotStreamTests
        MudShotStreamTests.setUpClass();cls.f=MudShotStreamTests.f;cls.prefix=MudShotStreamTests.prefix
        cls.j,cls.jax=cls.f.j,cls.f.jax
        cls.lines=[*cls.prefix,'|move|p1a: Snorlax|Mud Shot|p2a: Absol','|-damage|p2a: Absol|80/100',
            '|move|p2a: Absol|Surf|p1a: Snorlax']

    def events(self,lines):return [e for p in self.f.projections(lines) for e in p.events]

    def pack(self,streams):
        import numpy as np
        from dataclasses import replace
        from gpu_engine import public_inputs
        p=self.f.projections(self.prefix[:1])[0]
        bindings={name:np.asarray(getattr(p,name),np.uint32) for name in (
            'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding',
            'active_binding','damage_binding','type_binding','sleep_binding','toxic_binding','tendency_binding')}
        streams=[[replace(e,sequence=i+1) for i,e in enumerate(events)] for events in streams]
        return list(public_inputs.pack_chunks(np.arange(1,len(streams)+1,dtype=np.int32),streams,
            source_binding=self.f.catalog.source.device.binding,**bindings))

    def test_missing_duplicate_late_queue_and_tag_facts_refuse_mixed_batch(self):
        import numpy as np
        from dataclasses import replace
        from gpu_engine import public_inputs
        good=self.events(self.lines)
        index=next(i for i,e in enumerate(good) if e.kind==public_inputs.Kind.MUDSHOT and e.payload[4]==1)
        late=list(good);fact=late.pop(index);late.insert(index+1,fact)
        noqueue=list(good);d=noqueue[index].payload;noqueue[index]=replace(noqueue[index],payload=(d[0],-1,-1,0,0,0,0,0))
        raw=max(i for i in range(index) if good[i].kind==public_inputs.Kind.RAW_BEGIN)
        d=noqueue[raw].payload;noqueue[raw]=replace(noqueue[raw],payload=(d[0]^(1<<16),*d[1:]))
        tagged=list(good);hit=next(i for i,e in enumerate(tagged) if e.kind==public_inputs.Kind.MUDSHOT and e.payload[0]==1)
        d=tagged[hit].payload;tagged[hit]=replace(tagged[hit],payload=(*d[:5],1,0,0))
        raw=max(i for i in range(hit) if tagged[i].kind==public_inputs.Kind.RAW_BEGIN)
        d=tagged[raw].payload;tagged[raw]=replace(tagged[raw],payload=(d[0]^(1<<17),*d[1:]))
        streams=[good,good[:index]+good[index+1:],good[:index]+[good[index]]+good[index:],late,noqueue,tagged,good,good]
        chunks=self.pack(streams);self.assertEqual(len(chunks),1)
        initial=self.f.e.initialize(self.f.catalog,stream_ids=self.j.arange(1,9,dtype=self.j.int32))
        out,error=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.f.catalog))(initial,chunks[0])
        self.assertEqual((error==0).tolist(),[True,False,False,False,False,False,True,True])
        for a,b in zip(self.jax.tree.leaves(out),self.jax.tree.leaves(initial)):
            if a.ndim and a.shape[0]==8:np.testing.assert_array_equal(a[1:6],b[1:6])
            else:np.testing.assert_array_equal(a,b)

    def test_unboost_receipt_and_damage_target_cannot_be_dropped_or_redirected(self):
        import numpy as np
        from dataclasses import replace
        from gpu_engine import public_inputs
        good=self.events([*self.lines[:-1],'|-unboost|p2a: Absol|def|1',self.lines[-1]])
        boost=next(i for i,e in enumerate(good) if e.kind==public_inputs.Kind.BOOST)
        dropped=good[:boost]+good[boost+1:]
        redirected=list(good);hit=next(i for i,e in enumerate(good) if e.kind==public_inputs.Kind.MUDSHOT and e.payload[0]==1)
        d=redirected[hit].payload;redirected[hit]=replace(redirected[hit],payload=(d[0],0,0,1,*d[4:]))
        chunks=self.pack([dropped,redirected,good,good,good,good,good,good]);self.assertEqual(len(chunks),1)
        initial=self.f.e.initialize(self.f.catalog,stream_ids=self.j.arange(1,9,dtype=self.j.int32))
        out,error=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.f.catalog))(initial,chunks[0])
        self.assertEqual((error==0).tolist(),[False,False]+[True]*6)
        for a,b in zip(self.jax.tree.leaves(out),self.jax.tree.leaves(initial)):
            if a.ndim and a.shape[0]==8:np.testing.assert_array_equal(a[:2],b[:2])
            else:np.testing.assert_array_equal(a,b)

    def test_consistent_forged_note_or_unboost_cannot_disagree_with_body(self):
        import numpy as np
        from dataclasses import replace
        from gpu_engine import public_inputs,public_mudshot_protocol,public_mudshot
        def forged(events,old,new):
            events=list(events)
            i=next(i for i,e in enumerate(events) if e.kind==public_inputs.Kind.MUDSHOT and e.payload[0]==old)
            data=events[i].payload
            payload=(new,-1,-1,0,0,0,0,0) if new==public_mudshot_protocol.NOTE else (new,1,0,1,0,0,0,0)
            events[i]=replace(events[i],payload=payload)
            h=max(k for k in range(i) if events[k].kind==public_inputs.Kind.RAW_BEGIN)
            data=events[h].payload
            word=(data[0]&~(31<<13))|((new+1)<<13)
            events[h]=replace(events[h],payload=(word,*data[1:]))
            return events
        unboost=self.events([*self.lines[:-1],'|-unboost|p2a: Absol|def|1',self.lines[-1]])
        boost=self.events([*self.lines[:-1],'|-boost|p2a: Absol|def|1',self.lines[-1]])
        bad_boost=list(boost)
        # Find the boost frame's NOTE rather than earlier arrival NOTE packets.
        body=next(i for i,e in enumerate(boost) if e.kind==public_inputs.Kind.BOOST)
        h=max(i for i in range(body) if boost[i].kind==public_inputs.Kind.RAW_BEGIN)
        sub=forged(boost[h:],public_mudshot_protocol.NOTE,public_mudshot.UNBOOST)
        bad_boost=boost[:h]+sub
        streams=[forged(self.events(self.lines),public_mudshot.DAMAGE,public_mudshot_protocol.NOTE),
            forged(unboost,public_mudshot.UNBOOST,public_mudshot_protocol.NOTE),bad_boost,
            self.events(self.lines),unboost,boost,unboost,boost]
        chunks=self.pack(streams);self.assertEqual(len(chunks),1)
        initial=self.f.e.initialize(self.f.catalog,stream_ids=self.j.arange(1,9,dtype=self.j.int32))
        out,error=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.f.catalog))(initial,chunks[0])
        self.assertEqual((error==0).tolist(),[False]*3+[True]*5)
        for a,b in zip(self.jax.tree.leaves(out),self.jax.tree.leaves(initial)):
            if a.ndim and a.shape[0]==8:np.testing.assert_array_equal(a[:3],b[:3])
            else:np.testing.assert_array_equal(a,b)

    def test_literal_unboost_zero_and_negative_differ_from_signed_boost(self):
        for kind,amount,cancels in (('-unboost','0',True),('-unboost','-1',True),
                ('-boost','0',False),('-boost','-1',False)):
            lines=[*self.lines[:-1],f'|{kind}|p2a: Absol|def|{amount}',self.lines[-1]]
            events=self.events(lines)
            from gpu_engine import public_inputs
            event=next(e for e in events if e.kind==public_inputs.Kind.BOOST)
            self.assertEqual(event.payload[4],int(cancels))
            state=self.f.ingest(lines)
            expected=0 if cancels else self.f.catalog.abilities.ability_id('Shield Dust')
            self.assertEqual(int(state.context.abilities.original[0,1,0]),expected)
            self.f.compare_source(lines)

    def test_pending_binding_identity_and_absence_refuse_empty_chunks(self):
        import numpy as np
        from dataclasses import replace
        from gpu_engine import public_inputs
        state=self.f.ingest(self.lines[:-1]);repeat=lambda x:self.j.repeat(x,8,axis=0) if x.ndim and x.shape[0]==1 else x
        state=self.jax.tree.map(repeat,state);state=state._replace(cursor=state.cursor._replace(stream_id=self.j.arange(1,9,dtype=self.j.int32)))
        pending=state.context.mudshot
        bad=pending._replace(binding=pending.binding.at[0,0].set(pending.binding[0,0]^self.j.uint32(1)),
            encounter=pending.encounter.at[1].set(999))
        state=state._replace(context=state.context._replace(mudshot=bad))
        template=self.pack([self.events(self.prefix)]*8)[0]
        empty=template._replace(valid=self.j.zeros_like(template.valid),more=self.j.zeros_like(template.more))
        out,error=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.f.catalog))(state,empty)
        self.assertEqual(error.tolist(),[int(public_inputs.Error.SOURCE)]*2+[0]*6)
        for a,b in zip(self.jax.tree.leaves(out),self.jax.tree.leaves(state)):np.testing.assert_array_equal(a,b)
        for catalog,bad_state in ((self.f.catalog,state._replace(context=state.context._replace(mudshot=None))),
                (replace(self.f.catalog,context=self.f.catalog.context._replace(mudshot=None,mudshot_move_id=None)),state)):
            _,error=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,catalog))(bad_state,empty)
            self.assertTrue(bool((error!=0).all()))

    def test_real64_cut_resolves_before_next_actor_with_global_cache_preserved(self):
        import numpy as np
        from gpu_engine import public_inputs,public_refresh_stream
        before=self.lines[:-1];next_move=self.lines[-1]
        base=len(self.events(before));offset=next(i for i,e in enumerate(self.f.projections([*before,next_move])[-1].events)
            if e.kind==public_inputs.Kind.MUDSHOT)
        pad='|-item|p1a: Snorlax|Choice Band|[from] item: Leftovers'
        n=len(self.f.projections([*before,pad])[-1].events);m=len(self.f.projections([*before,'|'])[-1].events)
        choices=[(n*a+m*b,a,b) for a in range(64) for b in range(64) if (base+n*a+m*b+offset)%64==63]
        self.assertTrue(choices);_,a,b=min(choices)
        events=self.events([*before,*([pad]*a),*(['|']*b),next_move])
        cut=max(i for i,e in enumerate(events) if e.kind==public_inputs.Kind.MUDSHOT)
        self.assertEqual(cut%64,63);chunks=self.pack([events]*8)
        state=self.f.e.initialize(self.f.catalog,stream_ids=self.j.arange(1,9,dtype=self.j.int32))
        step=self.jax.jit(lambda s,c:self.f.e.fold_chunk(s,c,self.f.catalog));saw_cut=False
        for chunk in chunks:
            state,error=step(state,chunk);self.assertEqual(error.tolist(),[0]*8)
            if int(chunk.words[0,63,0])==int(public_inputs.Kind.MUDSHOT):
                saw_cut=True;self.assertEqual(state.context.refresh.family.tolist(),[public_refresh_stream.MOVE]*8)
                self.assertEqual(state.context.refresh.mudshot_seen.tolist(),[1]*8)
                self.assertEqual(state.context.refresh.mudshot_effect_seen.tolist(),[0]*8)
                self.assertEqual(state.context.mudshot.side.tolist(),[-1]*8)
                self.assertEqual(state.context.abilities.original[:,1,0].tolist(),[self.f.catalog.abilities.ability_id('Shield Dust')]*8)
                np.testing.assert_array_equal(state.context.refresh.controller.visible.source_binding,self.f.source.device.binding)
        self.assertTrue(saw_cut);self.assertEqual(state.context.refresh.family.tolist(),[0]*8)
